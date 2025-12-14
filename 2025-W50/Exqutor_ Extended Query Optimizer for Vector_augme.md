# Exqutor: Extended Query Optimizer for Vector-augmented Analytical Queries

**Authors**: Hyunjoon Kim, Chaerim Lim, Hyeonjun An, Rathijit Sen, Kwanghyun Park

**Published**: 2025-12-10 14:42:52

**PDF URL**: [https://arxiv.org/pdf/2512.09695v2](https://arxiv.org/pdf/2512.09695v2)

## Abstract
Vector similarity search is becoming increasingly important for data science pipelines, particularly in Retrieval-Augmented Generation (RAG), where it enhances large language model inference by enabling efficient retrieval of relevant external knowledge. As RAG expands with table-augmented generation to incorporate structured data, workloads integrating table and vector search are becoming more prevalent. However, efficiently executing such queries remains challenging due to inaccurate cardinality estimation for vector search components, leading to suboptimal query plans. In this paper, we propose Exqutor, an extended query optimizer for vector-augmented analytical queries. Exqutor is a pluggable cardinality estimation framework designed to address this issue, leveraging exact cardinality query optimization techniques to enhance estimation accuracy when vector indexes (e.g., HNSW, IVF) are available. In scenarios lacking these indexes, we employ a sampling-based approach with adaptive sampling size adjustment, dynamically tuning the sample size to balance estimation accuracy and sampling overhead. This allows Exqutor to efficiently approximate vector search cardinalities while minimizing computational costs. We integrate our framework into pgvector, VBASE, and DuckDB, demonstrating performance improvements of up to four orders of magnitude on vector-augmented analytical queries.

## Full Text


<!-- PDF content starts -->

Exqutor: Extended Query Optimizer for
Vector-augmented Analytical Queries
Hyunjoon Kim∗, Chaerim Lim∗, Hyeonjun An∗, Rathijit Sen†, Kwanghyun Park∗‡
∗Yonsei University BDAI Lab,†Microsoft Gray Systems Lab
{wns41559, chaerim.lim, hyeonjun.an}@yonsei.ac.kr, rathijit.sen@microsoft.com, kwanghyun.park@yonsei.ac.kr
Abstract—Vector similarity search is becoming increasingly
important for data science pipelines, particularly in Retrieval-
Augmented Generation (RAG), where it enhances large language
model inference by enabling efficient retrieval of relevant external
knowledge. As RAG expands with table-augmented generation
to incorporate structured data, workloads integrating table and
vector search are becoming more prevalent. However, efficiently
executing such queries remains challenging due to inaccurate
cardinality estimation for vector search components, leading to
suboptimal query plans.
In this paper, we propose Exqutor, an ex tended qu ery op-
timizer for vector -augmented analytical queries. Exqutor is a
pluggable cardinality estimation framework designed to address
this issue, leveraging exact cardinality query optimization tech-
niques to enhance estimation accuracy when vector indexes (e.g.,
HNSW, IVF) are available. In scenarios lacking these indexes,
we employ a sampling-based approach with adaptive sampling
size adjustment, dynamically tuning the sample size to balance
estimation accuracy and sampling overhead. This allows Exqutor
to efficiently approximate vector search cardinalities while mini-
mizing computational costs. We integrate our framework into
pgvector, VBASE, and DuckDB, demonstrating performance
improvements of up to four orders of magnitude on vector-
augmented analytical queries.
Index Terms—Vector Similarity Search, Query Optimization
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) workflows have
become essential for integrating Large Language Models
(LLMs) into domains such as e-commerce, healthcare, and
recommendation systems [1]–[3]. By leveraging relevant exter-
nal knowledge, RAG significantly enhances LLM performance
and mitigates common challenges such as hallucinations and
outdated information. In a traditional RAG pipeline, a vector
similarity search (VSS) is a primary operator to retrieve
the top-k most semantically similar documents, which are
already embedded and saved in database systems. It can be
performed either within specialized vector database systems
optimized for vector data or within generalized vector database
systems, which extend traditional relational databases with
vector search capabilities.
Advances in vector similarity search.While traditional RAG
relies on simple top-k retrieval over unstructured vector data,
real-world analytical workloads increasingly demand the joint
retrieval of both structured and unstructured data [1], [4]–[6].
Consequently, significant advancements have been made in
filtered vector search and vector range search. The filtered
‡Corresponding author.vector search augments vector similarity search with attribute-
based filtering. This pattern captures many real-world appli-
cations, such as filtering by price and brand in e-commerce.
The vector range search generalizes the simple top-k vector
search by retrieving all vectors within a specified threshold
from the target vector. It is commonly used to retrieve all
sufficiently similar vectors, such as in plagiarism detection [7].
Those two techniques can be used together to form more
complex analytical queries like vector range search with filters.
Companies such as Alibaba [1], Apple [8], Huawei [9], and
Microsoft [10] have already restructured Approximate Nearest
Neighbor (ANN) indexes and database systems to efficiently
incorporate those types of queries. Building on these develop-
ments, modern vector database systems are expected to support
traditional query operators with VSS together [7], [10]–[14].
Vector-augmented analytical queries.Despite recent ad-
vances in attribute-based filtering and range-based vector
search, existing work has assumed that vector embeddings
and structured attributes are stored together within a sin-
gle relation [1], [7], [8], [10], [11]. However, in real-world
deployments where the data scales in volume and schema
complexity, this assumption becomes impractical [15]–[17].
Modern database systems commonly partition data across mul-
tiple relations, requiring join operations to execute analytical
queries on large datasets.
 
SELECT ps_comment, l_extendedprice, l_discount
FROM partsupp, lineitem, orders
WHERE
l_partkey = ps_partkey
AND o_orderkey = l_orderkey
AND o_orderdate BETWEEN DATE (2024-01-01, 2024-12-31)
AND ps_embedding <-> ${image_embedding} < ${D}
ORDER BY
ps_embedding <-> ${image_embedding}; 
Listing 1: An example V AQ generated for a user prompt in a
RAG pipeline (see Figure 1).
To bridge this gap, we introduce Vector-augmented Analyt-
ical Queries (V AQs), which integrate vector similarity search
with relational operators such as filters, aggregates, and joins.
Listing 1 presents an example V AQ that retrieves parts from
many suppliers visually similar to a given product image,
incorporating vector similarity search (ps_embedding <->
${image_embedding}) with relational joins, where<->
denotes the Euclidean distance operator. The query applies a
vector distance thresholdDto ensure that image embeddings
close enough to the query embedding are selected. Figure 1arXiv:2512.09695v2  [cs.DB]  11 Dec 2025

(1) User Prompt
Generalized Vector Database System(4) Answer(3) Ask to LLM
Analyzingsuppliercommentsrevealsthat70%aresatisfiedwiththecurrentprice,while30%suggestadiscount,anda15%discountcouldincreaseordercountssignificantly.(2) RetrievalStructuredVectorFindsupplierpartsthatarevisuallysimilartoagivenheadphoneimageusing2024data,andsuggestanoptimaldiscountratebyanalyzingassociatedcustomercommentsandpurchasebehaviorregardingdiscountrates.SELECTps_comment, l_extendedprice, l_discountFROMpartsupp, lineitem, ordersWHEREl_partkey= ps_partkeyANDo_orderkey= l_orderkeyANDo_orderdateBETWEENDATE (2024-01-01, 2024-12-31) ANDps_embedding${image_embedding}< ${D}ORDER BY ps_embedding${image_embedding};Vector-augmented Analytical QueryUser PromptRelevant Contexts
Image
Fig. 1: Extended RAG pipeline integrating vector search with structured data. This four-stage pipeline retrieves structured
and vector-based contexts to generate informed responses to user prompts. (1) The user provides a prompt, requesting optimal
discount rate. (2) The user transforms the prompt into a V AQ to retrieve the relevant structured and vector data. (3) The retrieved
contexts, along with the user prompt, are provided as input to the LLM. (4) The LLM generates a response, delivering analytical
insights based on the combined structured and vector-based retrieval results.
illustrates how data analysts leverage a RAG pipeline that
integrates vector search with structured data to derive rec-
ommended discount rates based on supplier feedback and
purchasing patterns [18]. Generalized vector database systems,
such as pgvector [13], VBASE [10], and DuckDB [19], are
inherently better suited for executing V AQs, as they provide
multi-relational queries such as complex filtering and aggre-
gation. In contrast, specialized vector database systems [20]
like Milvus [11] and Qdrant [12] are limited to single relation
queries. This often necessitates coupling with an external
RDBMS, which introduces additional system complexity and
risks of data silos [21].
Optimization opportunities.We observe that generalized
vector database systems currently suffer from limited perfor-
mance on V AQs. As shown in section III, this issue stems
primarily from suboptimal query execution plans caused by
inaccurate cardinality estimation, especially for VSS opera-
tors. Our investigation further reveals that several prominent
generalized vector database systems rely on fixed default
selectivity values when estimating VSS cardinality. This ob-
servation highlights a key optimization opportunity: improving
VSS cardinality estimation can significantly enhance V AQ
performance in generalized vector database systems.
Contributions.We propose Exqutor, an open-source frame-
work1designed to optimize query plans for V AQs by im-
proving cardinality estimation of vector similarity searches.
Exqutor introduces Exact Cardinality Query Optimization
(ECQO) for queries with vector indexes, executing Approxi-
mate Nearest Neighbor (ANN) searches during query planning
to obtain accurate cardinality estimates. Since ECQO can
incur substantial computational overhead, Exqutor addresses
this challenge by visiting only tables with vector embeddings
and reusing the obtained ANN results during execution. For
queries without vector indexes, Exqutor employs an adaptive
sampling-based approach that estimates predicate selectivity
by evaluating similarity over a subset of data. This approach
1https://github.com/BDAI-Research/Exqutoradaptively adjusts the sample size using momentum-based
feedback [22] and learning rate scheduling to balance estima-
tion accuracy and planning efficiency. By addressing the crit-
ical challenge of cardinality estimation for high-dimensional
vector data, this work not only delivers practical performance
gains but also opens up new avenues for advanced query
optimization research for vector database systems.
To the best of our knowledge, we introduce thefirst ana-
lytics benchmark that integrates both vector embeddings and
structured datato assess query optimization strategies for
V AQs. In addition, Exqutor is thefirst framework to explicitly
address VAQs in real analytical scenarios, highlighting and
mitigating the effects of incorrect cardinality estimates on
query performance. Our contributions are as follows:
•Vector-augmented SQL analytics (section IV): An ana-
lytical benchmark that extends TPC-H [23] and TPC-DS [24]
with widely used vector datasets by introducing embedding
columns. This design enables systematic evaluation of V AQs
under both classical relational and modern hybrid settings, and
the benchmark is publicly available as open-source1
•Exact cardinality query optimization with vector index
(subsection V-A): An optimization technique for V AQs with
vector indexes, integrating ECQO into query planning by
restricting ANN searches to vector tables. The results are
reused at execution time without incurring redundant overhead.
•Adaptive sampling for cardinality estimation in V AQs
without vector indexes (subsection V-B): A sampling-based
cardinality estimation approach for V AQs without vector in-
dexes. Exqutor employs adaptive sampling to improve accu-
racy by dynamically adjusting the sample size based on Q-
error through a momentum-based adjustment mechanism and
a learning rate scheduler.
•Evaluation of vector-augmented SQL analytics with
diverse datasets across multiple systems (section VI): Com-
prehensive evaluation using vector-augmented SQL analytics
with real-world vector datasets. Exqutor achieves speedups of
up to three orders of magnitude on pgvector, four orders of
magnitude on VBASE, and 37.2×on DuckDB.

II. BACKGROUND
Specialized vector database systems.A specialized vector
database system is a dedicated system designed for managing
high-dimensional vector data, particularly the embeddings
generated by machine learning models [25]. Systems such
as Milvus [11], Pinecone [26], Qdrant [12], Manu [27], and
ChromaDB [28] are representative examples. These systems
are architected with a strong focus on optimizing vector sim-
ilarity search performance, incorporating advanced indexing
techniques to accelerate approximate nearest neighbor search.
While they deliver high efficiency in vector operations, their
support for structured query capabilities remains limited [29].
In practice, they mainly support filtered vector queries, which
combine vector similarity search with attribute-based filters
over metadata, and these queries are restricted to execution
over a single collection.
Generalized vector database systems.Extending a traditional
database system to support vector-type operators is referred to
as a generalized vector database system. Examples include
pgvector [13], PASE [30], VBASE [10], and AnalyticDB [1],
which are built on PostgreSQL as an extension, as well as
DuckDB-VSS [31], which is based on DuckDB [19]. Other
systems, such as SingleStore-V [7] and ClickHouse [14],
are also built on relational database systems. Additionally,
NoSQL-based systems including Vespa [32], Neo4j [33], and
Redis [34] can be classified in this category. These systems are
not designed solely for vector operations and generally exhibit
lower vector similarity search performance than specialized
vector database systems [29]. However, they provide robust
support for managing a wide range of traditional data types.
Nearest neighbor search and vector index.Nearest Neighbor
(NN) search is a core operation in vector similarity search,
used to identify the most similar data points to a given query
vector based on distance metrics such as cosine similarity or
Euclidean distance. Two primary variants of this search exist:
K-Nearest Neighbor (KNN) [35]–[37], which offers exact
results, and Approximate Nearest Neighbor (ANN), which im-
proves efficiency by sacrificing some accuracy. KNN performs
an exhaustive comparison with all vectors in the dataset, guar-
anteeing precise results. However, its computational cost scales
poorly with data size, making it impractical for large-scale or
real-time applications. In contrast, ANN reduces search cost
by leveraging pre-built vector indexes that approximate the
nearest neighbor computation. This trade-off between accuracy
and performance makes ANN suitable for data-intensive work-
loads such as RAG and real-time recommendation systems.
Most vector database systems support both KNN and ANN,
allowing users to select the appropriate method depending on
application needs. Efficient ANN search depends heavily on
the underlying indexing technique. Common approaches in-
clude graph-based [38]–[41], hash-based [42], [43], tree-based
[44]–[46], and quantization-based methods [47]–[49]. Among
them, HNSW [38] is one of the most widely adopted due to
its high efficiency in high-dimensional spaces. HNSW builds
a layered proximity graph and uses greedy traversal strategies
Original PlanOptimal PlanHash⨝Seq ScanσLineitemIndex ScanσMerge⨝Index Scanσ
OrdersPartsuppNested Loop⨝Index ScanσLineitemIndex ScanσNested Loop⨝Index Scanσ
PartsuppOrders111111111111
Original planOptimal planExecution time ㏙ms㏚Log scale(a) Query execution time(b) Query plan100101102103104105Fig. 2: Execution time and generated query plan for the V AQ in
Listing 1 on pgvector. The optimal plan is generated by query
optimizer with true cardinality of vector similarity search. The
psembedding column in the partsupp table (80M) has vector
embeddings from the DEEP dataset.
to locate approximate neighbors, significantly reducing search
latency while maintaining acceptable accuracy.
Exact cardinality query optimization (ECQO).Query opti-
mizers often generate suboptimal plans, especially when cardi-
nality estimation is challenging due to data skew and complex
predicates [50], [51]. Vector similarity search exemplifies this
challenge, as multi-dimensional spaces, large datasets, and
query-dependent similarity thresholds complicate estimation
[51], [52]. ECQO is the method that uses intermediate result
sizes by actually executing queries during query plan gener-
ation rather than relying solely on simplifying assumptions
and coarse metadata. Although this approach ensures accurate
cardinality values for optimal plans, it has traditionally been
considered an offline technique due to computational overhead
[50], [53]. This overhead arises from evaluating all relevant
relations in a query. To address this, ECQO is selectively
applied to critical expressions that are expected to significantly
influence join ordering or plan cost estimation and used for
relation pruning, reducing computational costs [50], [54].
This is particularly beneficial for multi-join queries, where
cardinality estimation errors often degrade performance. In
such scenarios, the additional overhead introduced by ECQO
is justified by its considerable impact on the decrease in total
execution time [55], [56].
III. MOTIVATION
V AQs can be executed on two types of database systems:
specialized or generalized vector database systems. Special-
ized vector database systems, dedicated to managing vector
data such as Milvus [11] and Qdrant [12], support relational
metadata storage and attribute-based filtering within a sin-
gle collection. However, when handling a large number of
structured attributes, these systems often require integration
with an external RDBMS [21], resulting in increased com-
plexity for development and query processing overhead [1].
Consequently, they become less suitable for applications that
require seamless and efficient management of a large volume
of structured attributes alongside vector data.
Generalized vector database systems, such as pgvector [13],
VBASE [10], AnalyticDB [1], and DuckDB [19], integrate
vector data directly within relational databases, making them

TABLE I: Heuristic fixed selectivity values used by general-
ized vector database systems for estimating the cardinality of
VSS operators.Card(T)is the number of rows in the table T
containing vector data.
DBMS Selectivity Estimated Cardinality
pgvector Fixed0.333×Card(T)
VBASE Fixed0.500×Card(T)
DuckDB Fixed1.000×Card(T)
better suited to address these limitations [57]. However, they
still face significant challenges, especially the lack of ac-
curate cardinality estimation for vector similarity predicates.
Although these systems can leverage existing cost-based query
optimizers to generate efficient execution plans [29], their
approaches mainly target simple filter queries over a single
relation [1], [7], [10]. In contrast, multi-relation V AQs with
complex joins require more precise cardinality estimation [58],
which current systems have not yet fully addressed. As
illustrated in Figure 2, inaccurate cardinality estimation in
vector similarity search can lead to suboptimal query execution
plans. The root cause is that most generalized vector database
systems employ heuristic methods, as shown in Table I.
Database systems such as pgvector, VBASE, and DuckDB use
simplistic fixed selectivity values for VSS operators regardless
of query predicates or data distribution, which often misleads
the optimizer and results in significant suboptimal query plans.
In our empirical analysis, incorporating accurate cardinality
estimates for vector similarity search resulted in significantly
more efficient query plans. Improvements in join order, join
type, and scan strategy led to substantial performance gains by
reducing intermediate computation and avoiding unnecessary
work. These results underscore the importance of precise
cardinality estimation in optimizing query execution for V AQs.
This improvement becomes even more critical as dataset sizes
scale [55], [56], highlighting the essential role of accurate
cardinality estimation in optimizing V AQs within generalized
vector database systems [58].
IV. VECTOR-AUGMENTEDSQLANALYTICS
V AQs extend traditional analytical workloads by integrat-
ing relational operations with VSS. While existing ANN
benchmarks effectively evaluate vector indexing and search
algorithms, they primarily focus on high-recall and high-
QPS scenarios over a single relation with limited structured
filtering. Consequently, they fail to reflect the complexity of
real-world analytical workloads, which often require multi-
attribute filtering, relational joins, and fine-grained similarity
conditions as part of end-to-end query execution. This gap
limits the applicability of such benchmarks for evaluating
the performance and functionality of modern vector-aware
database systems. To address this limitation, we design a
benchmark calledVector-augmented SQL analytics, which
extends the widely used TPC-H [23] and TPC-DS [24]
benchmarks by incorporating realistic vector data into both
datasets and queries.This benchmark is designed to capture the complexity of
real-world workloads by enabling systematic evaluation of
V AQs in hybrid settings that combine structured relational data
with high-dimensional vector representations. We augment
thepartsupptable in TPC-H with two embedding columns
(psimage embeddingandps text embedding) and an addi-
tional tag column (ps tag) to simulate multimodal charac-
teristics of parts provided by suppliers. These embeddings
are derived from widely used datasets such as DEEP [59],
SIFT [60], SimSearchNet++ [61], YFCC [62], [63], and
WIKI [64], ensuring diverse and realistic vector characteristics
across image and text domains. We further extend the schema
by adding a text embedding column (p text embedding) to the
parttable, enabling evaluation of queries that involve multiple
vector predicates across relations as well as combinations of
vector similarity and attribute-based filtering.
To construct vector-augmented analytical queries, we extend
representative TPC-H queries: Q3, Q5, Q8, Q9, Q10, Q11,
Q12, and Q20. These queries were selected to reflect a
diverse range of query characteristics, including variations in
the number of joins, the join paths between tables, and the
presence of complex filtering [65]. We further chose queries
that directly use or are well-suited for augmentation with
thepartsupptable. We preserve the original relational filter
conditions to maintain semantic consistency, while adding
vector-based predicates to reflect modern hybrid workloads.
The resulting queries integrate traditional filters with vector
similarity conditions, creating realistic analytical scenarios that
jointly stress relational and vector processing components.
A key feature ofVector-augmented SQL analyticsis
the introduction of avector distance threshold, denoted as
ps_embedding <-> ${image_embedding}<${D}.
This enables range-based vector similarity search tailored
to real-world use cases such as anomaly detection, contex-
tual recommendations, and similarity-based visual filtering.
Whereas top-ksearch always returns a fixed number of results
regardless of their actual similarity, it may yield results that
are either semantically weak or fail to capture the full set
of strongly related vectors. In contrast, vector range queries
apply a user-defined similarity thresholdDto retrieve only
those records that meet a minimum similarity requirement.
Furthermore, this threshold condition also provides bench-
marking flexibility, allowing one to vary query selectivity and
test system performance under different constraints. Unlike
top-ksearch with fixed cardinality, vector range queries return
a variable number of results based on the threshold. This
makes cardinality estimation significantly more challenging,
providing a practical basis for optimizers facing uncertainty
in modern hybrid workloads.
In addition, we extend our benchmark design to TPC-
DS [24], following a similar approach to TPC-H. Specifi-
cally, we augment theitemtable with an embedding column
(iembedding) and queries (Q7, Q12, Q19, Q20, Q42, Q72,
and Q98) are selected across diverse query classes [66] and
varying selectivities to evaluate V AQs under more complex
analytical scenarios.

Generalized Vector Database SystemResultExqutorVectorSimilaritySearchWith IndexSampling-based Cardinality EstimationExact Cardinality Query Optimization❶Original Plan❸Re-optimized PlanParserExecutorPlanner / OptimizerVAQAnalyzer
❷VectorCardinalityCalculationQ-error Results& Sample SizeWithoutIndexVector Index Search ResultsFig. 3: Integration of Exqutor into a generalized vector
database system. When a V AQ is processed, the original
query plan is forwarded to Exqutor (➊), which calculates
vector cardinality using ECQO or sampling-based cardinality
estimation (➋). The estimated cardinality for vector range
search is then returned to the query optimizer, allowing it to
generate a more accurate and efficient execution plan (➌).
V. EXQUTOR
We present Exqutor, a system designed to enhance the query
optimization process for V AQs in generalized vector database
systems by estimating the cardinality of vector range search.
As illustrated in Figure 3, Exqutor integrates seamlessly into
existing query optimizer pipelines. It supports both ANN and
KNN queries and operates during the query planning phase
to influence the selection of optimal execution strategies. For
V AQs with vector indexes, Exqutor employs Exact Cardinality
Query Optimization (ECQO), which performs lightweight
index-based searches and derives accurate cardinality val-
ues for vector range queries during query planning (subsec-
tion V-A). For V AQs without index, Exqutor uses a sampling-
based approach to approximate selectivity (subsection V-B).
A. Vector Index-based ECQO
When a V AQ involves a vector similarity predicate and a
corresponding index is available, Exqutor applies a strategy
called ECQO. The key idea behind ECQO is to execute
a lightweight vector index search during query planning to
compute the exact number of vectors among the retrieved
candidates that satisfy the similarity threshold. The resulting
cardinality is incorporated into the optimizer’s cost model,
guiding the selection of both join ordering and scan strategies
for more efficient plan generation. By replacing heuristic
assumptions with precise estimates, ECQO helps the optimizer
construct execution plans.
Many vector database systems support ANN indexes such as
HNSW and IVF, which are primarily designed to accelerate
similarity search during query execution. ECQO repurposes
these index structures during the planning phase by issuing
a range query over the index using the actual query vector
and user-defined similarity threshold. Since ANN indexes
such as HNSW are designed to limit the search space using
hierarchical or graph-based traversal strategies, even when in-
voked during planning, the index search completes quickly and
introduces negligible overhead. Although ANN indexes areapproximate by design, they can exhibit deterministic behavior
with fixed configurations. For instance, in the case of HNSW,
the result of a search remains deterministic as long as the index
graph, entry point, search parameters (e.g.,ef search), and
query vector remain unchanged. This property allows ECQO
to treat the cardinality obtained during query planning as an
exact input to the optimizer.
In addition to its accuracy, ECQO improves efficiency by
reusing computation across planning and execution. Since
the index search used during planning retrieves the same
candidate vectors as those needed at execution time, Exqutor
can cache the results and eliminate redundant searches. As a
result, the ANN vector search only needs to be performed
once per query, effectively reducing execution latency and
avoiding unnecessary computation. Collectively, the accuracy,
determinism, and low overhead of ECQO make it a practical
and effective mechanism for optimizing vector-aware queries
in systems that support vector indexing.
Implementation in generalized vector database systems.
We integrated ECQO into multiple systems based on pgvector,
VBASE, and DuckDB by extending their query optimizers to
incorporate exact cardinality estimates obtained from vector
indexes. In pgvector and VBASE, we integrated ECQO by
extending the planner’s hook to identify V AQs with vector
range predicates and trigger index-based similarity searches
during query planning. To minimize redundant computation
across query stages, the candidate set retrieved during planning
is cached and reused during execution. Given that ANN index
searches produce deterministic results under fixed parameters,
this reuse ensures consistency while reducing overhead. In
DuckDB, we implemented ECQO by modifying the logical op-
timizer rules responsible for cardinality estimation. DuckDB’s
in-process architecture enables efficient propagation of this es-
timate to downstream components of the optimizer, including
join reordering and scan strategy selection.
B. Sampling-based Cardinality Estimation without Vector In-
dex
When a V AQ lacks a vector index, the query optimizer must
rely on either an index over structured attributes or perform a
full sequential scan. In the case of a sequential scan, evaluating
the similarity predicate requires computing distances between
the query vector and all vectors in the dataset. This exhaustive
KNN search is highly expensive, making it unsuitable for
direct execution during query planning, unlike the approach
used in ECQO. To address this, Exqutor adopts a sampling-
based cardinality estimation approach specifically for KNN
queries, where it approximates the number of qualifying tuples
by evaluating similarity over a small subset of the data. This
enables the optimizer to obtain meaningful cardinality esti-
mates at a fraction of the cost of a full scan, making sampling
a practical alternative for query planning in the absence of
vector indexes. Similar to ECQO, the estimated cardinality is
integrated into the optimizer’s cost model, allowing it to select
execution plans that better reflect the selectivity of the vector
range predicate.

To determine an appropriate sample size, Exqutor uses a
statistical formula derived from classical sampling theory [67].
The required number of samplesNis computed as:
N=&
z2·ˆP·(1− ˆP)
e2'
(1)
zcritical value corresponding to the desired confidence level
(e.g.,z= 1.96for 95% confidence).
ˆPestimated proportion of data points expected to fall within
the similarity threshold.
edesired margin of error (e.g.,e= 0.05for 5% error).
Adaptive sampling size adjustment.While fixed sample
sizes provide statistical guarantees, they may not be equally
effective across datasets with varying distributions or dimen-
sionalities. In high-dimensional or skewed datasets, a fixed
sample size may be unnecessarily large, resulting in wasted
resources, or too small, leading to inaccurate estimates. To
address this, Exqutor introduces an adaptive sampling mecha-
nism that dynamically adjusts the sample size based on estima-
tion accuracy observed after query execution. This mechanism
aims to balance estimation precision with computational cost,
adapting to the workload characteristics.
Exqutor employs a momentum-based adjustment algorithm
combined with a learning rate scheduler to adapt the sampling
size over time. Momentum smooths fluctuations in adjustment,
preventing instability, while the learning rate scheduler grad-
ually reduces update magnitude to ensure convergence. The
adjustment is guided by the Q-error [68]–[70], which measures
the deviation between the estimated and true cardinality:
Q-error= maxCard esti
Card true,Card true
Card esti
(2)
Using this metric, Exqutor tracks recent estimation accuracy
and updates the sample size according to the following rule:
δ=α·(Q-error−β)−(100−α)·sampling ratio (3)
Vt=m·V t−1+η t·δ(4)
sampling sizet+1=sampling sizet+V t (5)
Here,δis the adjustment factor computed from estimation
error and the current sampling ratio, which determines the
direction and magnitude of sample size updates.V tis the
momentum term at iterationt,mis the momentum coefficient,
andη tis the learning rate.αbalances the contribution between
Q-error and the sampling ratio, andβis a tunable threshold
representing acceptable Q-error.
The learning rate is decayed at each iteration using:
ηt+1=γ·η t (6)
whereγis the decay factor (0< γ <1) that progressively
reduces the adjustment magnitude.
This adaptive mechanism enables Exqutor to respond effec-
tively to changing query workloads and data characteristics.
When estimation remains accurate with low Q-error, the
sample size is reduced to save computation. Conversely, higher
Q-error triggers an increase in sample size to restore accuracy.This feedback-driven adaptation ensures that sampling remains
both efficient and reliable over time.
By combining statistical sampling theory with adaptive
learning techniques, Exqutor delivers a practical and robust
solution for cardinality estimation in vector similarity queries
without index support. This method is particularly effective for
exploratory and analytical queries on large datasets, where full
scans are infeasible and traditional estimates are insufficient.
Implementation in generalized vector database systems.
We implemented sampling-based cardinality estimation in
pgvector by extending the query optimizer to support dynamic
sample size adjustment during planning. When a V AQ with
a vector range predicate lacks index support, the optimizer
invokes a sampling routine that evaluates the similarity pred-
icate over a representative subset of the data, rather than
performing a full KNN scan. To support adaptive sampling,
we extended the optimizer to track estimation accuracy using
the Q-error metric. After each query, the system compares the
estimated cardinality with the actual value observed during
execution and uses the resulting Q-error as feedback to adjust
the sample size for future queries, expanding it when accuracy
is insufficient and shrinking it when estimates remain stable.
Additionally, the optimizer maintains separate sample size
states for each table, allowing it to adapt to the specific
distributional characteristics of different datasets.
VI. EVALUATION
We evaluate the performance of Exqutor by integrating
it into pgvector, VBASE, and DuckDB, demonstrating its
generality and pluggability across fundamentally different
database system architectures. Our evaluation focuses on im-
provements in cardinality estimation accuracy, query execution
performance, and scalability across varying data and workload
characteristics. In particular, we show how Exqutor enhances
cardinality estimation for vector range predicates in Vector-
augmented Analytical Queries, enabling more effective query
optimization and yielding substantial execution-time improve-
ments.
To assess the impact of ECQO and sampling-based car-
dinality estimation, we design experiments for two types of
V AQs. First, we evaluate the benefit of executing lightweight
vector index searches during query planning for V AQs (sub-
section VI-A). Second, we analyze how sampling-based car-
dinality estimation improves V AQs without index support
(subsection VI-B). We then extend the evaluation to diverse
workloads, including multi-vector queries, correlation-aware
queries, and TPC-DS (subsection VI-C), and conclude with
an in-depth analysis of Exqutor covering overhead, scalability
(subsection VI-D), and limitations (subsection VI-E). Our key
findings are summarized as follows:
•For V AQs utilizing vector indexes, Exqutor significantly
improves query execution performance. In PostgreSQL-
based systems, pgvector and VBASE, ECQO achieves
speedups ranging from 1.01×to four orders of magnitude.
In DuckDB, ECQO yields speedups from 1.5×to 37.2×.

•For V AQs without vector indexes, sampling-based cardinal-
ity estimation improves performance from 1.2×to 3.2×.
The adaptive strategy converges to dataset-specific sample
sizes, balancing estimation accuracy and planning overhead
across diverse vector distributions.
•Exqutor consistently improves cardinality estimation ac-
curacy under diverse query conditions, including different
selectivities, data scales, and vector characteristics, while
incurring negligible overhead, enabling robust and efficient
query planning.
Datasets and V AQs.We conduct experiments on TPC-H and
TPC-DS based Vector-augmented SQL analytics using widely
used vector datasets: DEEP (96 dimensions) [59], SIFT (128
dimensions) [60], SimSearchNet++ (256 dimensions) [61].
These datasets not only vary in embedding dimensionality
but also cover diverse data distributions, including skewed
and normal [71], which are commonly observed in real-world
vector workloads. These datasets represent realistic use cases
in multimedia retrieval and allow us to investigate how vector
dimensionality affects query performance. For each dataset,
we construct V AQs1that perform filtering alongside vector
similarity search, reflecting practical analytics scenarios.
To ensure consistency across experiments, we configure
range thresholds in V AQs such that the expected number
of matches is controlled. For index-based V AQs, the range
threshold is tuned to return 200 vectors. For sampling-based
V AQs, the threshold is set to retrieve 1% of total rows. This
setting reflects typical usage where the user seeks only closely
related vectors and avoids biasing the experiments toward
either Exqutor or the baselines.
System setup.We conduct our experiments using pgvector,
VBASE, and DuckDB withDuckDB-VSS. The system is
equipped with Intel Xeon Gold 6530 configured with 128
vCPUs and 1.0 TB of RAM.
Each experiment begins with a warm-up execution, which
is excluded from the reported results to eliminate caching
effects. We then report the trimmed mean of execution
times over ten runs, removing the lowest and highest run-
times. We run with default settings, where PostgreSQL used
8 formax worker processesand DuckDB used 128 for
worker threads.
Indexing and sampling parameter configuration.For V AQs
involving vector indexes, we use HNSW [38] as the underlying
ANN structure in both Exqutor and the baseline systems. We
configure HNSW with the same vector index parameters (M=
16,ef construction= 200,ef search= 400).
For sampling-based cardinality estimation, we initially com-
pute the number of samplesNusing the sample size formula
(Equation 1) for sample size estimation [67], given a 95%
confidence level (z= 1.96), a proportion estimate ˆP= 0.5,
and a 5% margin of error (e= 0.05). Applying the formula
yields a fixed sample size ofN= 385.
For adaptive sampling, we extend the optimizer with
momentum-based feedback control. Parameter values are se-
lected based on prior work on adaptive query estimation
[22], [70]: we set the momentum coefficientm= 0.9,initial learning rateη 0= 0.1, weighting factorα= 50,
and target Q-errorβ= 1.5. These values balance Q-error
minimization and sample size stability. The learning rate decay
factorγ= 0.99gradually reduces adjustment magnitude to
ensure convergence. Sample size updates are triggered every
50 queries.
A. Vector Index-based Exact Cardinality Query Optimization
In this section, we evaluate the performance of Exqutor
when executing V AQs with a vector index using an ANN
search, specifically with HNSW [38]. The experiments com-
pare two configurations: (i) baseline generalized vector
database systems, and (ii) execution with ECQO enabled.
We evaluate Exqutor integrated into pgvector, VBASE, and
DuckDB, analyzing its effectiveness in optimizing query plan
by injecting exact cardinality and reducing redundant compu-
tations through lightweight index-based vector search.
Performance gains from ECQO.Figure 4 shows end-to-
end query execution times for TPC-H based Vector-augmented
SQL analytics using three datasets. Each figure compares
six system configurations, including pgvector, VBASE, and
DuckDB, with and without Exqutor integration. Across all
datasets and systems, ECQO consistently improves query per-
formance by enabling accurate cardinality estimation during
planning.
The most substantial performance gains are observed in
pgvector and VBASE. When Exqutor is integrated into pgvec-
tor, it achieves up to three orders of magnitude speedup over
the baseline. Similarly, applying Exqutor to VBASE yields
speedups of up to four orders of magnitude by enabling
accurate cardinality estimation for vector range search. These
improvements stem from two key mechanisms. First, ECQO
performs a lightweight vector index probe during planning
using the HNSW structure, which returns the exact number
of qualifying tuples. This cardinality is incorporated into the
cost model, improving scan method selection and join ordering
decisions. In all evaluated queries, thepartsupptable, which
contains the vector predicate, is selected as the first input in
the join plan, enabling early filtering and reducing intermediate
result sizes. Second, Exqutor caches the retrieved index results
during planning and reuses them during execution, avoiding
redundant similarity computations.
In DuckDB, which already features a highly optimized
execution engine for analytic queries, ECQO still yields
measurable improvements. While the baseline query plans in
DuckDB are already efficient, ECQO enhances query planning
by exposing the true selectivity of vector range search. This re-
sults in improved join strategies and execution time reductions
for V AQs.
Cardinality estimation and query plan optimization.In
PostgreSQL-based systems, pgvector and VBASE, the query
optimizer lacks native support for vector-aware statistics. Con-
sequently, cardinality estimates for vector predicates default to
fixed or arbitrary values, regardless of data size or threshold
selectivity. This leads to poor plan choices such as unneces-
sary hash joins or late application of selective filters. With

1111111
Q3Q5Q8Q9Q10Q11Q12Q20End㎿to㎿end Time ㏙ms㏚Log scalepgvectorpgvector 㚉 ExqutorVBASEVBASE     㚉 ExqutorDuckDBDuckDB   㚉 Exqutor
100101102103104106105
1111111
Q3Q5Q8Q9Q10Q11Q12Q20End㎿to㎿end Time ㏙ms㏚Log scalepgvectorpgvector 㚉 ExqutorVBASEVBASE     㚉 ExqutorDuckDBDuckDB   㚉 Exqutor(a) DEEP
1111111
Q3Q5Q8Q9Q10Q11Q12Q20End㎿to㎿end Time ㏙ms㏚Log scale100101102103104105
(b) SIFT
1111111
Q3Q5Q8Q9Q10Q11Q12Q20End㎿to㎿end Time ㏙ms㏚Log scale100101102103104105
(c) SimSearchNet++
Fig. 4: Query execution time for TPC-H V AQs with a vector
index using three different datasets (SF100). Each subfigure
compares query latency with and without Exqutor integration
in pgvector, VBASE and DuckDB.
Exqutor, the optimizer receives precise cardinality derived
from the actual vector index search. This improves plan quality
by enabling the planner to choose nested loop joins when
beneficial and to push down vector filters early. Sequential
scans are often replaced with index scans, reducing I/O and
computation.
DuckDB exhibits similar issues, as it also lacks built-in
selectivity estimation for vector data. Without ECQO, its
optimizer must assume worst-case cardinality for similarity
predicates. By integrating ECQO through Exqutor, DuckDB
can leverage index-based estimates during planning, producing
more accurate join orders and improving execution. While the
relative gains are smaller than in PostgreSQL, the improve-
ment is still consistent across datasets.
Interestingly, our evaluation reveals that pgvector and
VBASE with Exqutor can outperform baseline DuckDB on
certain queries. This is due to PostgreSQL’s ability to benefit
more from ECQO’s index scan enablement, which offsets its
slower executor. These results show that optimizer enhance-
ments like ECQO can shift performance bottlenecks, making
even traditionally slower systems competitive with modernexecution engines.
Impact of vector dimensionality and query characteristics.
With ECQO, the availability of accurate cardinality estimates
for vector range search allows the optimizer to avoid unneces-
sary operations such as full scans and hash joins, significantly
reducing the cost of non-vector operators. As a result, HNSW-
based vector search emerges as the dominant component of
query execution time. Because the cost of HNSW search
increases with vector dimensionality due to more expensive
distance computations, we observe a corresponding rise in
overall query time as the dimensionality grows.
Despite these improvements, PostgreSQL’s cost model re-
mains insufficiently equipped to accurately reflect the per-
formance characteristics of ANN-based vector indexes. In
pgvector, although the index leverages PostgreSQL’s internal
buffer pool, ANN indexes typically incur high space ampli-
fication [72], often exceeding the size of the base table. At
the same time, structures like HNSW achieveO(logCard(T))
sublinear search times [38], which the current cost model
fails to capture. As a result, the optimizer may overestimate
the cost of an HNSW index scan and fail to select it, even
in cases where it would be the better access method [73].
As a result, the system significantly overestimates the cost
of index scans. VBASE, on the other hand, decouples ANN
index access from the primary buffer pool and manages it
with an independent memory structure, making precise cost
estimation even more difficult. These limitations can still result
in suboptimal plan choices, indicating that further refinements
to the underlying cost model are needed to fully capitalize on
accurate cardinality estimates.
Query characteristics also influence ECQO’s effectiveness.
Most evaluated queries benefit from better join ordering and
early application of vector filters. However, in queries like
Q20, where the dominant cost arises from a full scan of
thelineitemtable, ECQO’s impact is limited. While ECQO
improves join ordering, the total benefit is limited by the
large fixed cost of scanning unrelated data. This suggests that
ECQO is most effective when vector predicates contribute
significantly to overall selectivity.
B. Sampling-based Cardinality Estimation
In this section, we evaluate the performance of Exqutor
applied to TPC-H V AQs that perform KNN searches without
vector indexes, where cardinality estimation is handled via
sampling. We compare three configurations: (i) the baseline
pgvector execution without sampling, which uses default
selectivity estimates, (ii) Exqutor with a fixed sample size
derived from statistical confidence bounds, and (iii) Exqutor
with adaptive sampling that dynamically adjusts the sample
size based on query feedback and dataset properties. For
consistency, we focus on a subset of V AQs where the optimizer
selects a sequential scan on thepartsupptable which means
KNN search forms a dominant component of execution cost.
These queries are representative of realistic cases in vector
analytics without index support.

010203040506070
Q3Q10Q12Q3Q10Q12Q3Q9Q10DEEPSIFTSimsearch㚉㚉End㎿to㎿end time ㏙sec㏚pgvectorpgvector 㚉 Exqutor ㏙fixed sampling㏚pgvector 㚉 Exqutor ㏙adaptive sampling㏚Fig. 5: Query execution time on pgvector for TPC-H V AQs
without vector index (SF100). The fixed sample size uses
a constant sample size ofN= 385, whereas the adaptive
sampling strategy dynamically adjusts the sample size based
on Q-error.
Performance gains from sampling-based cardinality esti-
mation.Figure 5 shows that both fixed and adaptive sampling
significantly improve execution time compared to the baseline.
Fixed sampling achieves speedups from 1.2×to 3.2×, demon-
strating that even a small, uniform sample provides better
cardinality feedback. However, fixed-size sampling does not
consider data distribution or vector dimensionality. For exam-
ple, when working with dense clusters or high-dimensional
embeddings, the fixed sample may misrepresent selectivity,
leading to misestimation and suboptimal plans.
Adaptive sampling overcomes this limitation by modifying
the sample size based on query feedback. It tracks Q-error
over time and adjusts the number of sampled rows accordingly.
When the error is high, indicating that the estimate diverges
from observed cardinality, the sample size is increased to
enhance accuracy. Conversely, when estimates stabilize, the
sample size is reduced to conserve computation. Adaptive
sampling delivers up to 1.4×speedup over fixed sampling,
consistently outperforming fixed sampling across datasets with
varying distributional properties. The ability to react to query
conditions makes adaptive sampling especially effective for
dynamic workloads and heterogeneous data.
Effect of adaptive sampling.To understand how adaptive
sampling evolves over time, Figure 6 illustrates how Exqutor
adjusts the sample size in response to Q-error. Initially, the
system starts with a statistically determined sample size using
Equation 1 and evaluates estimation accuracy after each query.
When the Q-error exceeds a predefined threshold, Exqutor in-
creases the sample size using momentum-based updates, which
smooth out fluctuations and promote stable convergence. As
updates continue, the learning rate decays gradually, leading
to smaller adjustments over time. This feedback loop enables
the system to maintain estimation accuracy while minimizing
unnecessary computation.
This behavior demonstrates that Exqutor effectively bal-
ances estimation accuracy and planning efficiency. The sample
size trajectory varies depending on the dataset: for DEEP and
SimSearchNet++, the sample size decreases over time as Q-
error stabilizes, allowing the system to reduce planning cost
without loss of accuracy. In contrast, for SIFT, the sample
355365375385395405415
01002003004005006007008009001000Sample sizeIterationAdaptive sampling ㏙DEEP㏚Adaptive sampling ㏙SIFT㏚Adaptive sampling ㏙SimsearchNet㚉㚉㏚Fixed sampling㏙SimSearchNet㚉㚉㏚Fig. 6: Convergence of adaptive sample size over time on
the DEEP, SIFT, and SimSearchNet++ (SF100). The plot
illustrates how Exqutor adaptively adjusts sample sizes for
V AQs compared against fixed sampling.
size increases to satisfy higher estimation demands due to its
more complex distribution. Ultimately, Exqutor converges to a
dataset-specific equilibrium that reflects the selectivity patterns
and estimation difficulty of each workload.
Accuracy and optimizer impact.The effectiveness of
sampling-based estimation directly translates to improved
query plans. Without sampling, pgvector relies on fixed default
cardinality values, which often result in suboptimal join order-
ing and delayed application of vector predicates. For example,
due to underestimated selectivity, the optimizer may place
the join with thepartsupptable later in the execution plan,
which results in unnecessarily large intermediate results and
degraded performance. In contrast, sampling allows Exqutor
to provide more accurate cardinality estimates, enabling the
optimizer to apply vector filters earlier and improve the plan
across multiple aspects. Hash joins are often replaced with
nested loop joins that better exploit early filtering, and scan
strategies also improve, with the optimizer selecting index-
based access over full sequential scans.
Overall, sampling-based cardinality estimation is a
lightweight yet powerful technique for improving query
optimization in V AQs without index support. Fixed sampling
provides a simple, statistically grounded baseline that already
improves execution time. Adaptive sampling further enhances
this by learning from query performance and dynamically
tuning sampling effort. Together, these methods allow
Exqutor to apply selectivity-aware optimization even when
vector indexes are unavailable, while also ensuring stable
performance under shifting workloads.
C. Performance on Diverse Workloads
To further validate the generality of Exqutor, we evaluate
it on more diverse workloads. We extend TPC-H with multi-
vector and correlation-aware V AQs that combine vector simi-
larity with tag filtering, and further assess its effectiveness on
TPC-DS based Vector-augmented SQL analytics. We utilize
three widely used embedding datasets, DEEP (96 dimen-
sions) [59], YFCC (192 dimensions) [62], [63] and WIKI (768
dimensions) [64].
V AQs with correlation.We evaluate Exqutor on V AQs with
correlation that combine vector search and tag-based filtering

1111111111
Q3Q5Q8Q9Q10Q11Q12Q20Execution time ㏙ms㏚Log scalepgvectorpgvector 㚉 Exqutor
100101102103104Fig. 7: Query execution time for TPC-
H V AQs on thepartsupptable with
YFCC and tag filtering (SF10).
1111111111
Q3Q5Q8Q9Q10Q11Q12Q20Execution time ㏙ms㏚Log scalepgvectorpgvector 㚉 Exqutor
100101102103104Fig. 8: Query execution time for TPC-
H V AQs on thepartsupptable with
DEEP and WIKI (SF10).
1111111111
Q3Q5Q8Q9Q10Q11Q12Q20Execution time ㏙ms㏚Log scalepgvectorpgvector 㚉 Exqutor
100101102103104Fig. 9: Query execution time for TPC-
H V AQs on thepartsupptable with
DEEP and theparttable with WIKI
(SF10).
11111111
Q7Q12Q19Q20Q42Q72Q98Execution time ㏙ms㏚Log scalepgvectorpgvector 㚉 Exqutor103102101100
Fig. 10: Query execution time for TPC-DS V AQs with vector
indexes on the DEEP using pgvector and Exqutor (SF10).
using the YFCC. In this setting, the associated tags are stored
inps_tag, and V AQs are extended with conditions requiring
the presence of specific tags. As shown in Figure 7, query
execution times improve significantly, achieving up to 301.5×
speedup. Exqutor provides accurate cardinality estimates for
nodes that jointly apply vector predicates and tag filters,
enabling the optimizer to generate more efficient plans.
Multi-vector V AQs.We further evaluate Exqutor on multi-
vector query workloads, where embeddings from multiple
sources are integrated into analytical queries. As shown in
Figure 8, when both DEEP and WIKI datasets are stored
in thepartsupptable, we observe substantial performance
improvements, with query execution times accelerated by
factors ranging from 1.07×to 479.4×. Figure 9 illustrates the
scenario where DEEP embeddings are stored inpartsupp
while WIKI embeddings are stored in theparttable. Even
in this more complex join setting, we still observe speedups
from 1.07×to 254×. These results confirm that Exqutor
effectively adapts to queries involving multiple vector columns
across relations, consistently optimizing execution plans and
achieving significant gains.
Evaluation on TPC-DS.To further validate the effectiveness
of Exqutor on diverse workloads, we conducted experiments
on the TPC-DS based Vector-augmented SQL analytics, a
widely used benchmark that provides rich and complex query
templates. As shown in Figure 10, the results demonstrate con-
sistent performance improvements, with query execution times
achieving speedups of up to 109.6×. Moreover, we observed
query plan transformations similar to those identified in TPC-
H based Vector-augmented SQL analytics, confirming that
Exqutor effectively adapts to workload diversity in realistic
decision-support scenarios.
Hash⨝Hash⨝HNSW Index ScanσPartsuppHash⨝Seq ScanσCustomerSeq ScanσSeq ScanσLineitemOrders5143.16519142.32435681.413807.72136684.12910.59924379.815Execution Time 52595.309 ms(a) pgvector
⨝NestedLoop⨝Index ScanσCustomerNestedLoop⨝Index ScanσOrdersIndex ScanσHNSW Index ScanσPartsuppLineitem0.0030.3662.0390.0012.9840.0013.408NestedLoopExqutorOverhead 1.88 msExecution Time 15.571 ms (b) pgvector + Exqutor
Fig. 11: Query plan comparison for Q3 on the DEEP dataset
(SF100). (a) shows the execution plan using pgvector, and (b)
shows the optimized plan with Exqutor, where ECQO enables
accurate cardinality estimation using HNSW index probing
during planning. Both plans display individual operator nodes
along with their execution times, reported in milliseconds.
D. Discussion
Query time analysis.Figure 11 illustrates the execution plans
for Q3 on the DEEP dataset before and after applying Exqutor
with ECQO on pgvector. The baseline pgvector plan relies
heavily on full table scans and parallel hash joins, resulting in
high scan and join costs, with total query time exceeding 52
seconds. In contrast, the optimized plan with Exqutor elimi-
nates expensive join operations and replaces sequential scans
with selective index scans, enabled by accurate cardinality
feedback from ECQO. Notably, the HNSW-based vector range
search is executed once during query planning and the result
is reused during execution. This reuse reduces the runtime
cost of HNSW index scan from 10.6 ms to only 0.366 ms. In
addition, ECQO itself contributes only 6.82 ms for cardinality
estimation and 1.88 ms as integration overhead. As a result,
the total query execution time drops dramatically to 15.571
milliseconds. This confirms that exact cardinality injection via
ECQO enables the optimizer to generate execution plans that
are highly efficient, eliminating unnecessary operations and
prioritizing selective access paths. Beyond this single query
example, the most common improvements come from join
reordering, changes in join methods (e.g., hash joins replaced
with nested-loop joins), and scan method transitions from full
or bitmap-index scans to index-based access. These patterns
consistently appear across different workloads such as TPC-
DS, confirming that the benefits of Exqutor generalize beyond
a single benchmark. Further illustrations of query execution

1111111111
Q3Q5Q8Q9Q10Q11Q12Q20Execution time ㏙ms㏚Log scalepgvector 㚉 SelNet ㏙learned estimator㏚pgvector 㚉 Exqutor103102101100104Fig. 12: Query execution time of SelNet (learned estimator)
and Exqutor for TPC-H V AQs with vector indexes on the
DEEP dataset (SF10).
improvements are provided in supplementary figures2.
Comparison with learned cardinality estimator.Figure 12
compares Exqutor with SelNet [74], a learned selectivity es-
timator. Exqutor achieves speedups up to 16.1×speedup over
SelNet. SelNet requires 77 ms for a single-query cardinality
estimation and depends on offline training, which becomes
increasingly costly with the dataset size. In addition, it relies
on external model management, introducing further overhead
and complexity. When compared with the sampling-based
approach, Exqutor achieves an average Q-error of 1.69, while
SelNet yields a higher Q-error of 5.53. These results highlight
the advantages of Exqutor in delivering accurate cardinality
estimates with lightweight overhead, ensuring both efficiency
and robustness in query optimization.
Cardinality estimation accuracy.As previously discussed in
subsection VI-A, PostgreSQL lacks statistical summaries for
vector data, causing it to assign arbitrarily large fixed values
for estimated cardinalities. This fixed estimation, irrespective
of the given predicate, leads to substantial errors across all
datasets. With static sampling, Exqutor significantly reduces
estimation errors, achieving Q-errors ranging from 1.04 to
1.57. Adaptive sampling further improves accuracy by adapt-
ing the sample size based on Q-error feedback, achieving a
lower Q-error range of 1.02 to 1.19, closely aligning with the
true cardinality. Figure 6 shows that in DEEP and SimSearch-
Net++, the sample size was reduced while maintaining a low
Q-error, demonstrating the efficiency of adaptive sampling. In
contrast, SIFT required an increased sample size to achieve
higher accuracy, highlighting the adaptive nature of Exqutor
in adjusting to dataset-specific characteristics.
Query time of sampling method across selectivity levels.
Figure 13 shows the query execution time of both pgvector
and Exqutor under the sampling-based estimation method,
evaluated at three different selectivity levels: 0.1%, 1%, and
10%. As pgvector uses a static heuristic for cardinality esti-
mation, its execution plans remain largely unchanged across
different selectivity levels. This leads to inefficient plans and
consistently high query latency, especially when selectivity is
low. In contrast, Exqutor dynamically adjusts its sampling size
and accurately estimates cardinality, enabling the optimizer
to generate plans that better reflect the selectivity of vector
predicates. This allows Exqutor to significantly outperform
pgvector at low selectivities, where the benefit of avoiding
2https://github.com/BDAI-Research/Exqutor/tree/main/query plans
020406080100120140160
Q3Q10Q12Q3Q10Q12Q3Q10Q120.10㚠1㚠10㚠Execution time ㏙sec㏚Selectivitypgvector ㏙default㏚pgvector ㏙10㚠㏚pgvector ㏙1㚠㏚pgvector ㏙0.1㚠㏚pgvector ㏙0.01㚠㏚pgvector㚉ExqutorFig. 13: Query execution time for TPC-H V AQs without vector
indexes under varying selectivities on the DEEP dataset with
SF100 using pgvector and Exqutor. The values in parentheses
indicate the adjusted fixed selectivity parameter values used
by pgvector’s heuristic selectivity.
TABLE II: Overhead and reduced execution time of ECQO
and fixed sampling methods on pgvector for Query Q3 with
DEEP dataset. The relative overhead indicates the ratio of
optimization overhead to execution time reduction.
Method DatasetOverhead
(ms)Reduced
Time (s)Relative
Overhead (%)
ECQODEEP 1.88 43.34 0.0043
SIFT 1.89 41.65 0.0045
SimSearchNet++ 1.96 43.66 0.0045
Fixed
SamplingDEEP 28.17 38.87 0.0724
SIFT 33.23 47.12 0.0705
SimSearchNet++ 72.83 26.67 0.2730
over-scanning is most pronounced. At 10% selectivity, the
performance gap narrows because pgvector’s heuristic estimate
approaches the actual selectivity, reducing the relative gain
from sampling.
We further extended this analysis by varying the parameter
shown in Table I and comparing the resulting execution
times. As shown in Figure 13, relying on a fixed sampling
ratio fails to consistently yield optimal performance across
different selectivities. The key determinant is the alignment
between estimated and actual cardinalities, showing that a
fixed parameter alone cannot ensure robust performance across
varying selectivities.
Overhead of Exqutor.Table II presents the overhead intro-
duced by ECQO and sampling-based estimation in Exqutor
for Q3 on the DEEP, SIFT, and SimSearchNet++ datasets.
ECQO incurs minimal cost as it leverages a single lightweight
traversal of the vector index during query planning. As shown,
ECQO’s overhead is consistently low across all datasets, rang-
ing from 1.88 to 1.96 milliseconds, while delivering substantial
reductions in execution time, with savings between 41.65 and
43.66 seconds. The relative overhead remains extremely small,
between 0.0043% and 0.0045%, indicating that ECQO offers
high returns for negligible planning effort. Sampling-based es-
timation introduces significant execution time savings, ranging
from 26.67 to 47.12 seconds across the evaluated datasets. In
return for these improvements, it introduces moderate planning
overhead, measured between 28.17 and 72.83 ms, which arises
from evaluating the similarity predicate on a sampled subset

111111111111
110100Executiontime ㏙ms㏚Log scaleScale factorpgvector ㎿ Q3pgvector㚉Exqutor ㎿ Q3pgvector ㎿ Q5pgvector㚉Exqutor ㎿ Q5pgvector㎿ Q20pgvector㚉Exqutor ㎿ Q20
100101102103104105Fig. 14: Scalability of Exqutor (ECQO) on pgvector for TPC-
H V AQs using the DEEP dataset, evaluated while increasing
the scale factor.
during query planning. The relative overhead remains low,
from 0.0705% to 0.273%, with the highest value observed
in the SimSearchNet++ dataset. This pattern is attributed to
the higher computational cost of similarity evaluation in high-
dimensional vector. Nevertheless, the overall overhead remains
minimal in comparison to the achieved performance gains,
validating the practicality of the approach.
Data scalability.To evaluate the scalability of Exqutor, we
conducted experiments on the DEEP dataset using scale factors
of 1, 10, and 100. Figure 14 shows that pgvector experiences
near-linear growth in execution time as dataset size increases.
In contrast, Exqutor demonstrates highly stable performance
across all scale factors by leveraging the property that HNSW
search complexity does not grow linearly with data size.
Notably, all evaluated queries except Q20 exhibited similar
trends, where Exqutor consistently mitigated the impact of
dataset scaling. In the case of Q20, execution time increases
linearly due to the full scan of thelineitemtable. However,
Exqutor still achieves up to 1.31×speedups, demonstrating
that Exqutor effectively mitigates the effect of increasing data
size by leveraging accurate cardinality and efficient query
planning, ensuring stable performance for large-scale V AQs.
E. Limitations
While Exqutor demonstrates substantial performance im-
provements across diverse workloads, several limitations re-
main. In high-dimensional spaces, the overhead of sampling
increases because of the higher cost of distance computations,
which may reduce the efficiency of our adaptive sampling
strategy. Moreover, our approach relies on cost models that
fail to fully capture the performance characteristics of ANN
indexes, so even with accurate cardinality estimates the op-
timizer may still choose suboptimal plans. Addressing these
issues through more efficient sampling in high dimensions
and refined cost models for V AQ optimization remains an
important direction for future work.
VII. RELATEDWORK
Filtered vector search.As vector similarity search becomes
more prevalent, many systems [11], [12], [27] store vector
embeddings alongside structured metadata, enabling filteredvector search. This trend has also emerged in ANN bench-
marks [63], [75], highlighting the growing importance of
efficient filtering techniques. Several studies have optimized
filtered vector queries by restructuring ANN indexes to support
filtering constraints more effectively. ACORN [76], SeRF [77],
HQANN [78], and diskANN [41] enhance ANN search by
integrating attribute filtering directly into the index structure,
improving retrieval efficiency. However, these methods are
limited to filtering within a single relation or collection,
making them less effective for large-scale analytical workloads
that involve complex joins across multiple datasets.
Query optimization in generalized vector database systems.
Several generalized vector database systems have extended
traditional query processing techniques to support vector oper-
ations. AnalyticDB [1] optimizes filtered vector searches using
a cost-based model, while SingleStore [7] integrates filters
directly into vector index scans to improve retrieval efficiency.
However, these optimizations primarily target simple filter
queries rather than complex analytical workloads involving
multi-way joins and nested queries. As a result, they do not
effectively address the challenges of optimizing V AQs, where
inaccurate cardinality estimation can severely degrade query
performance.
One technique in query optimization for efficiently estimat-
ing selectivity and cost is sampling. Early works introduced
random sampling for join size estimation [79], [80], while
later approaches refined these ideas with adaptive sampling
strategies [81]. The method in [81] adjusts the sample size
dynamically until a desired confidence level is reached, but
does not consider sampling overhead or optimize it dynami-
cally based on query characteristics.
VIII.CONCLUSION
In this paper, we introduce Exqutor, an extended query
optimizer designed to improve the performance of vector-
augmented analytical queries by addressing the challenges
of inaccurate cardinality estimation in vector searches. By
leveraging exact cardinality query optimization and adaptive
sampling, Exqutor significantly enhances query performance,
achieving speedups of up to four orders of magnitude. Through
integration with pgvector, VBASE, and DuckDB, Exqutor
extends the ability of generalized vector database systems to
efficiently handle vector-augmented analytical queries, con-
tributing to the optimization of emerging data science pipelines
like retrieval-augmented generation (RAG). This work demon-
strates the critical role of accurate cardinality estimation and
query optimization in enhancing the scalability and efficiency
of modern data workflows.
ACKNOWLEDGMENT
This research was supported by NRF grants (No. RS-2025-
16068623, No. RS-2024-NR121334), the Korea Basic Science
Institute (National Research Facilities and Equipment Center)
grant funded by the Ministry of Science and ICT (No. RS-
2024-00403860), Advanced Database System Infrastructure
(NFEC-2024-11-300458), and the Yonsei University Research
Fund (2025-22-0057).

REFERENCES
[1] C. Wei, B. Wu, S. Wang, R. Lou, C. Zhan, F. Li, and Y . Cai, “Analyticdb-
v: a hybrid analytical engine towards query fusion for structured and
unstructured data,”Proceedings of the VLDB Endowment, vol. 13,
no. 12, p. 3152–3165, Aug. 2020.
[2] C. Su, J. Wen, J. Kang, Y . Wang, Y . Su, H. Pan, Z. Zhong, and
M. S. Hossain, “Hybrid rag-empowered multi-modal llm for secure data
management in internet of medical things: A diffusion-based contract
approach,”IEEE Internet of Things Journal, pp. 1–1, 2024.
[3] Y . Yang and C. Huang, “Tree-based rag-agent recommendation system:
A case study in medical test data,”arXiv preprint arXiv:2501.02727,
2025.
[4] Y . Lu, A. Chowdhery, S. Kandula, and S. Chaudhuri, “Accelerating
machine learning inference with probabilistic predicates,” inProceedings
of the 2018 International Conference on Management of Data, ser.
SIGMOD ’18. New York, NY , USA: Association for Computing
Machinery, May 2018, p. 1493–1508. [Online]. Available: https:
//doi.org/10.1145/3183713.3183751
[5] V . Sanca and A. Ailamaki, “Analytical engines with context-rich pro-
cessing: Towards efficient next-generation analytics,” in2023 IEEE 39th
International Conference on Data Engineering (ICDE). IEEE, 2023,
pp. 3699–3707.
[6] V . Sanca, M. Chatzakis, and A. Ailamaki, “Context-enhanced relational
operators with vector embeddings,” no. arXiv:2312.01476, Dec. 2023,
arXiv:2312.01476. [Online]. Available: http://arxiv.org/abs/2312.01476
[7] C. Chen, C. Jin, Y . Zhang, S. Podolsky, C. Wu, S. Wang, E. Hanson,
Z. Sun, R. Walzer, and J. Wang, “SingleStore-V: An Integrated
Vector Database System in SingleStore,”Proceedings of the VLDB
Endowment, vol. 17, no. 12, pp. 3772–3785, 2024. [Online]. Available:
https://doi.org/10.14778/3685800.3685805
[8] J. Mohoney, A. Pacaci, S. R. Chowdhury, A. Mousavi, I. F. Ilyas, U. F.
Minhas, J. Pound, and T. Rekatsinas, “High-throughput vector similarity
search in knowledge graphs,”Proc. ACM Manag. Data, vol. 1, no. 2,
Jun. 2023. [Online]. Available: https://doi.org/10.1145/3589777
[9] T. Wang, X. Xue, G. Li, and Y . Wang, “Andb: Breaking boundaries
with an ai-native database for universal semantic analysis,” 2025.
[Online]. Available: https://arxiv.org/abs/2502.13805
[10] Q. Zhang, S. Xu, Q. Chen, G. Sui, J. Xie, Z. Cai, Y . Chen,
Y . He, Y . Yang, F. Yang, M. Yang, and L. Zhou, “VBASE:
Unifying online vector similarity search and relational queries
via relaxed monotonicity,” 2023, p. 377–395. [Online]. Available:
https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi
[11] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li,
X. Xu, K. Yu, Y . Yuan, Y . Zou, J. Long, Y . Cai, Z. Li, Z. Zhang, Y . Mo,
J. Gu, R. Jiang, Y . Wei, and C. Xie, “Milvus: A purpose-built vector
data management system,” inProceedings of the 2021 International
Conference on Management of Data, ser. SIGMOD ’21. Association
for Computing Machinery, 2021, p. 2614–2627. [Online]. Available:
https://doi.org/10.1145/3448016.3457550
[12] Qdrant, “Qdrant: High-performance vector search at scale,” https://
qdrant.tech/, 2024, accessed: 2024-12-01.
[13] A. Kane, “pgvector: Open-source vector similarity search for post-
gresql,” http://github.com/pgvector, 2021, accessed: 2024-11-29.
[14] R. Schulze, T. Schreiber, I. Yatsishin, R. Dahimene, and A. Milovidov,
“Clickhouse - lightning fast analytics for everyone,”Proc. VLDB
Endow., vol. 17, no. 12, p. 3731–3744, Nov. 2024. [Online]. Available:
https://doi.org/10.14778/3685800.3685802
[15] K. Sawarkar, A. Mangal, and S. R. Solanki, “Blended rag: Improving rag
(retriever-augmented generation) accuracy with semantic search and hy-
brid query-based retrievers,” in2024 IEEE 7th International Conference
on Multimedia Information Processing and Retrieval (MIPR). IEEE,
2024, pp. 155–161.
[16] P. Zezula, “Scalable similarity search for big data,” inScalable Informa-
tion Systems, J. J. Jung, C. Badica, and A. Kiss, Eds. Cham: Springer
International Publishing, 2015, p. 3–12.
[17] F. Lei, J. Chen, Y . Ye, R. Cao, D. Shin, H. Su, Z. Suo, H. Gao, W. Hu,
P. Yin, V . Zhong, C. Xiong, R. Sun, Q. Liu, S. Wang, and T. Yu, “Spider
2.0: Evaluating language models on real-world enterprise text-to-sql
workflows,” no. arXiv:2411.07763, Nov. 2024, arXiv:2411.07763 [cs].
[Online]. Available: http://arxiv.org/abs/2411.07763
[18] M. Wang, L. Lv, X. Xu, Y . Wang, Q. Yue, and J. Ni, “An efficient and
robust framework for approximate nearest neighbor search with attribute
constraint,” inAdvances in Neural Information Processing Systems,A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
Eds., vol. 36. Curran Associates, Inc., 2023, p. 15738–15751.
[Online]. Available: https://proceedings.neurips.cc/paper files/paper/
2023/file/32e41d6b0a51a63a9a90697da19d235d-Paper-Conference.pdf
[19] M. Raasveldt and H. M ¨uhleisen, “Duckdb: an embeddable analytical
database,” inProceedings of the 2019 international conference on
management of data, 2019, pp. 1981–1984.
[20] Y . Zhang, S. Liu, and J. Wang, “Are there fundamental limitations in
supporting vector data management in relational databases? a case study
of postgresql,” in2024 IEEE 40th International Conference on Data
Engineering (ICDE), 2024, pp. 3640–3653.
[21] Milvus Team, “DNA Sequence Classifi-
cation with Milvus,” https://milvus.io/blog/
2021-09-06-dna-sequence-classification-based-on-milvus.md, 2024,
accessed: 2024-11-30.
[22] I. Sutskever, J. Martens, G. Dahl, and G. Hinton, “On the importance
of initialization and momentum in deep learning,” inInternational
conference on machine learning. PMLR, 2013, pp. 1139–1147.
[23] M. Poess and C. Floyd, “New tpc benchmarks for decision support and
web commerce,”ACM Sigmod Record, vol. 29, no. 4, pp. 64–71, 2000.
[24] R. O. Nambiar and M. Poess, “The making of tpc-ds.” inVLDB, vol. 6,
2006, pp. 1049–1058.
[25] J. J. Pan, J. Wang, and G. Li, “Vector database management techniques
and systems,” inCompanion of the 2024 International Conference
on Management of Data, ser. SIGMOD/PODS ’24. Association
for Computing Machinery, 2024, p. 597–604. [Online]. Available:
https://doi.org/10.1145/3626246.3654691
[26] I. Pinecone Systems, “Pinecone: The vector database to build knowl-
edgeable ai,” https://www.pinecone.io/, 2024, accessed: 2024-12-01.
[27] R. Guo, X. Luan, L. Xiang, X. Yan, X. Yi, J. Luo, Q. Cheng, W. Xu,
J. Luo, F. Liu, Z. Cao, Y . Qiao, T. Wang, B. Tang, and C. Xie, “Manu:
A cloud native vector database management system,” 2022. [Online].
Available: https://arxiv.org/abs/2206.13843
[28] Chroma, “Chroma: The open-source ai application database,” https://
www.trychroma.com/, 2024, accessed: 2024-12-01.
[29] J. J. Pan, J. Wang, and G. Li, “Survey of vector database management
systems,”The VLDB Journal, vol. 33, no. 5, p. 1591–1615, Sep. 2024.
[30] W. Yang, T. Li, G. Fang, and H. Wei, “Pase: Postgresql ultra-
high-dimensional approximate nearest neighbor search extension,” in
Proceedings of the 2020 ACM SIGMOD International Conference on
Management of Data. ACM, Jun. 2020, p. 2241–2253. [Online].
Available: https://dl.acm.org/doi/10.1145/3318464.3386131
[31] D. D. Team, “Duckdb-vss: Vector similarity search extension for
duckdb,” https://github.com/duckdb/duckdb vss, 2024, accessed: 2024-
12-01.
[32] Vespa.ai, “Vespa: Ai + data, online at any scale,” https://vespa.ai/, 2024,
accessed: 2024-12-01.
[33] Neo4j, “Neo4j vector index and search,” https://neo4j.com/labs/
genai-ecosystem/vector-search/, 2024, accessed: 2024-12-01.
[34] Redis, “Redis for vector database,” https://redis.io/solutions/
vector-database/, 2024, accessed: 2024-12-01.
[35] L. E. Peterson, “K-nearest neighbor,”Scholarpedia, vol. 4, no. 2, p.
1883, 2009.
[36] S. Dhanabal and S. Chandramathi, “A review of various k-nearest neigh-
bor query processing techniques,”International Journal of Computer
Applications, vol. 31, no. 7, pp. 14–22, 2011.
[37] J. C. Bezdek, S. K. Chuah, and D. Leep, “Generalized k-nearest neighbor
rules,”Fuzzy Sets and Systems, vol. 18, no. 3, pp. 237–256, 1986.
[38] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approxi-
mate nearest neighbor search using hierarchical navigable small world
graphs,”IEEE transactions on pattern analysis and machine intelligence,
vol. 42, no. 4, pp. 824–836, 2018.
[39] C. Fu, C. Xiang, C. Wang, and D. Cai, “Fast approximate nearest
neighbor search with the navigating spreading-out graph,”Proc. VLDB
Endow., vol. 12, no. 5, p. 461–474, Jan. 2019. [Online]. Available:
https://doi.org/10.14778/3303753.3303754
[40] W. Zhao, S. Tan, and P. Li, “Song: Approximate nearest neighbor
search on gpu,” in2020 IEEE 36th International Conference on Data
Engineering (ICDE), 2020, pp. 1033–1044.
[41] S. Gollapudi, N. Karia, V . Sivashankar, R. Krishnaswamy, N. Begwani,
S. Raz, Y . Lin, Y . Zhang, N. Mahapatro, P. Srinivasan, A. Singh, and
H. V . Simhadri, “Filtered-diskann: Graph algorithms for approximate
nearest neighbor search with filters,” inProceedings of the ACM
Web Conference 2023, ser. WWW ’23. Association for Computing

Machinery, 2023, p. 3406–3416. [Online]. Available: https://doi.org/10.
1145/3543507.3583552
[42] B. Zheng, X. Zhao, L. Weng, N. Q. V . Hung, H. Liu, and
C. S. Jensen, “Pm-lsh: A fast and accurate lsh framework for
high-dimensional approximate nn search,”Proc. VLDB Endow.,
vol. 13, no. 5, p. 643–655, Jan. 2020. [Online]. Available:
https://doi.org/10.14778/3377369.3377374
[43] Y . Park, M. Cafarella, and B. Mozafari, “Neighbor-sensitive hashing,”
Proc. VLDB Endow., vol. 9, no. 3, p. 144–155, Nov. 2015. [Online].
Available: https://doi.org/10.14778/2850583.2850589
[44] M. Muja and D. G. Lowe, “Scalable nearest neighbor algorithms for high
dimensional data,”IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 36, no. 11, pp. 2227–2240, 2014.
[45] K. Lu, H. Wang, W. Wang, and M. Kudo, “Vhp: approximate nearest
neighbor search via virtual hypersphere partitioning,”Proc. VLDB
Endow., vol. 13, no. 9, p. 1443–1455, May 2020. [Online]. Available:
https://doi.org/10.14778/3397230.3397240
[46] E. Bernhardsson, “Annoy: Approximate nearest neighbors in
c++/python,” https://github.com/spotify/annoy, 2013, accessed: 2024-
12-01.
[47] H. J ´egou, M. Douze, and C. Schmid, “Product quantization for nearest
neighbor search,”IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 33, no. 1, pp. 117–128, 2011.
[48] F. A. Research, “Faiss: A library for efficient similarity search and
clustering of dense vectors,” https://github.com/facebookresearch/faiss,
2017, accessed: 2024-12-01.
[49] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,”IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547,
2019.
[50] I. Trummer, “Exact cardinality query optimization with bounded
execution cost,” inProceedings of the 2019 International Conference
on Management of Data. Amsterdam Netherlands: ACM, Jun. 2019,
p. 2–17. [Online]. Available: https://dl.acm.org/doi/10.1145/3299869.
3300087
[51] J. Sun, G. Li, and N. Tang, “Learned cardinality estimation
for similarity queries,” inProceedings of the 2021 International
Conference on Management of Data. Virtual Event China: ACM,
Jun. 2021, p. 1745–1757. [Online]. Available: https://dl.acm.org/doi/10.
1145/3448016.3452790
[52] L. Doshi, V . Zhuang, G. Jain, R. Marcus, H. Huang, D. Altinb ¨uken,
E. Brevdo, and C. Fraser, “Kepler: Robust learning for parametric query
optimization,”Proceedings of the ACM on Management of Data, vol. 1,
no. 1, p. 1–25, May 2023.
[53] S. Chaudhuri, V . Narasayya, and R. Ramamurthy, “Exact cardinality
query optimization for optimizer testing,”Proceedings of the VLDB
Endowment, vol. 2, no. 1, pp. 994–1005, 2009.
[54] A. Datta, B. Tsan, Y . Izenov, and F. Rusu, “Analyzing query optimizer
performance in the presence and absence of cardinality estimates,”
no. arXiv:2311.17293, Nov. 2023, arXiv:2311.17293 [cs]. [Online].
Available: http://arxiv.org/abs/2311.17293
[55] K. Lee, A. Dutt, V . Narasayya, and S. Chaudhuri, “Analyzing the impact
of cardinality estimation on execution plans in microsoft sql server,”
vol. 16, p. 2871–2883, Jul. 2023.
[56] V . Markl, V . Raman, D. Simmen, G. Lohman, H. Pirahesh,
and M. Cilimdzic, “Robust query processing through progressive
optimization,” inProceedings of the 2004 ACM SIGMOD international
conference on Management of data, ser. SIGMOD ’04. New York, NY ,
USA: Association for Computing Machinery, Jun. 2004, p. 659–670.
[Online]. Available: https://dl.acm.org/doi/10.1145/1007568.1007642
[57] J. Xian, T. Teofili, R. Pradeep, and J. Lin, “Vector search with openai
embeddings: Lucene is all you need,” inProceedings of the 17th ACM
International Conference on Web Search and Data Mining, 2024, pp.
1090–1093.
[58] H. Lan, Z. Bao, and Y . Peng, “A survey on advancing the dbms query
optimizer: Cardinality estimation, cost model, and plan enumeration,”
Data Science and Engineering, vol. 6, pp. 86–101, 2021.
[59] A. Babenko and V . Lempitsky, “Efficient indexing of billion-scale
datasets of deep descriptors,” inProceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, 2016, pp. 2055–2063.
[60] H. J ´egou, R. Tavenard, M. Douze, and L. Amsaleg, “Searching in one
billion vectors: re-rank with source coding,” in2011 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2011, pp. 861–864.[61] Meta, “Here’s how we’re using AI to help
detect misinformation,” https://ai.meta.com/blog/
heres-how-were-using-ai-to-help-detect-misinformation/, 2020,
accessed: 2024-11-30.
[62] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland,
D. Borth, and L.-J. Li, “Yfcc100m: The new data in multimedia
research,”Communications of the ACM, vol. 59, no. 2, pp. 64–73, 2016.
[63] H. V . Simhadri, M. Aum ¨uller, A. Ingber, M. Douze, G. Williams,
M. D. Manohar, D. Baranchuk, E. Liberty, F. Liu, B. Landrumet al.,
“Results of the big ann: Neurips’23 competition,”arXiv preprint
arXiv:2409.17424, 2024.
[64] Cohere, “Wikipedia (december 2022) dataset,” https://huggingface.co/
datasets/Cohere/wikipedia-22-12, accessed: 2025-09-20.
[65] M. Dreseler, M. Boissier, T. Rabl, and M. Uflacker, “Quantifying tpc-
h choke points and their optimizations,”Proceedings of the VLDB
Endowment, vol. 13, no. 8, pp. 1206–1220, 2020.
[66] M. Poess, R. O. Nambiar, and D. Walrath, “Why you should run tpc-ds:
A workload analysis.” inVLDB, vol. 7, 2007, pp. 1138–1149.
[67] G. D. Israelet al., “Determining sample size,” 1992.
[68] A. Kipf, T. Kipf, B. Radke, V . Leis, P. Boncz, and A. Kemper, “Learned
cardinalities: Estimating correlated joins with deep learning,”arXiv
preprint arXiv:1809.00677, 2018.
[69] B. Hilprecht, A. Schmidt, M. Kulessa, A. Molina, K. Kersting, and
C. Binnig, “Deepdb: Learn from data, not from queries!”arXiv preprint
arXiv:1909.00607, 2019.
[70] A. Dutt, C. Wang, A. Nazi, S. Kandula, V . Narasayya, and S. Chaudhuri,
“Selectivity estimation for range predicates using lightweight models,”
Proceedings of the VLDB Endowment, vol. 12, no. 9, pp. 1044–1057,
2019.
[71] L. Kuffo, E. Krippner, and P. Boncz, “Pdx: A data layout for vector
similarity search,”Proceedings of the ACM on Management of Data,
vol. 3, no. 3, pp. 1–26, 2025.
[72] R. Cheng, Y . Peng, X. Wei, H. Xie, R. Chen, S. Shen, and H. Chen,
“Characterizing the dilemma of performance and index size in billion-
scale vector search and breaking it with second-tier memory,”arXiv
preprint arXiv:2405.03267, 2024.
[73] A. El-Helw, I. F. Ilyas, and C. Zuzarte, “Statadvisor: Recommending
statistical views,”Proceedings of the VLDB Endowment, vol. 2, no. 2,
pp. 1306–1317, 2009.
[74] Y . Wang, C. Xiao, J. Qin, R. Mao, M. Onizuka, W. Wang, R. Zhang,
and Y . Ishikawa, “Consistent and flexible selectivity estimation for high-
dimensional data,” inProceedings of the 2021 International Conference
on Management of Data, 2021, pp. 2319–2327.
[75] myscale, “vector-db-benchmark,” https://github.com/myscale/
vector-db-benchmark, 2024, accessed: 2024-12-01.
[76] L. Patel, P. Kraft, C. Guestrin, and M. Zaharia, “Acorn: Performant and
predicate-agnostic search over vector embeddings and structured data,”
Proceedings of the ACM on Management of Data, vol. 2, no. 3, pp.
1–27, 2024.
[77] C. Zuo, M. Qiao, W. Zhou, F. Li, and D. Deng, “Serf: Segment
graph for range-filtering approximate nearest neighbor search,”Proc.
ACM Manag. Data, vol. 2, no. 1, Mar. 2024. [Online]. Available:
https://doi.org/10.1145/3639324
[78] W. Wu, J. He, Y . Qiao, G. Fu, L. Liu, and J. Yu, “Hqann: Efficient
and robust similarity search for hybrid queries with structured and
unstructured constraints,” inProceedings of the 31st ACM International
Conference on Information & Knowledge Management, 2022, pp. 4580–
4584.
[79] Z. Zhao, R. Christensen, F. Li, X. Hu, and K. Yi, “Random sampling
over joins revisited,” inProceedings of the 2018 International Confer-
ence on Management of Data, 2018, pp. 1525–1539.
[80] P. J. Haas, J. F. Naughton, S. Seshadri, and A. N. Swami, “Selectivity
and cost estimation for joins based on random sampling,”Journal of
Computer and System Sciences, vol. 52, no. 3, pp. 550–569, 1996.
[81] R. J. Lipton, J. F. Naughton, and D. A. Schneider, “Practical selectivity
estimation through adaptive sampling,” inProceedings of the 1990 ACM
SIGMOD international conference on Management of data, 1990, pp.
1–11.