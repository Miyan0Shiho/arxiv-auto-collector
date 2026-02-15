# Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis

**Authors**: Abylay Amanbayev, Brian Tsan, Tri Dang, Florin Rusu

**Published**: 2026-02-11 23:40:26

**PDF URL**: [https://arxiv.org/pdf/2602.11443v1](https://arxiv.org/pdf/2602.11443v1)

## Abstract
Retrieval-Augmented Generation (RAG) applications increasingly rely on Filtered Approximate Nearest Neighbor Search (FANNS) to combine semantic retrieval with metadata constraints. While algorithmic innovations for FANNS have been proposed, there remains a lack of understanding regarding how generic filtering strategies perform within Vector Databases. In this work, we systematize the taxonomy of filtering strategies and evaluate their integration into FAISS, Milvus, and pgvector. To provide a robust benchmarking framework, we introduce a new relational dataset, \textit{MoReVec}, consisting of two tables, featuring 768-dimensional text embeddings and a rich schema of metadata attributes. We further propose the \textit{Global-Local Selectivity (GLS)} correlation metric to quantify the relationship between filters and query vectors.
  Our experiments reveal that algorithmic adaptations within the engine often override raw index performance. Specifically, we find that: (1) \textit{Milvus} achieves superior recall stability through hybrid approximate/exact execution; (2) \textit{pgvector}'s cost-based query optimizer frequently selects suboptimal execution plans, favoring approximate index scans even when exact sequential scans would yield perfect recall at comparable latency; and (3) partition-based indexes (IVFFlat) outperform graph-based indexes (HNSW) for low-selectivity queries. To facilitate this analysis, we extend the widely-used \textit{ANN-Benchmarks} to support filtered vector search and make it available online. Finally, we synthesize our findings into a set of practical guidelines for selecting index types and configuring query optimizers for hybrid search workloads.

## Full Text


<!-- PDF content starts -->

Filtered Approximate Nearest Neighbor Search in Vector
Databases: System Design and Performance Analysis
Abylay Amanbayev, Brian Tsan, Tri Dang, Florin Rusu
{amanbayev, btsan, tridang, frusu}@ucmerced.edu
University of California Merced
February 2026
Abstract
Retrieval-Augmented Generation (RAG) applications increasingly rely on Filtered Approximate Nearest Neigh-
bor Search (FANNS) to combine semantic retrieval with metadata constraints. While algorithmic innovations for
FANNS have been proposed, there remains a lack of understanding regarding how generic filtering strategies perform
within Vector Databases. In this work, we systematize the taxonomy of filtering strategies and evaluate their integra-
tion into FAISS, Milvus, and pgvector. To provide a robust benchmarking framework, we introduce a new relational
dataset,MoReVec, consisting of two tables, featuring 768-dimensional text embeddings and a rich schema of meta-
data attributes. We further propose theGlobal-Local Selectivity (GLS)correlation metric to quantify the relationship
between filters and query vectors.
Our experiments reveal that algorithmic adaptations within the engine often override raw index performance.
Specifically, we find that: (1)Milvusachieves superior recall stability through hybrid approximate/exact execution;
(2)pgvector’s cost-based query optimizer frequently selects suboptimal execution plans, favoring approximate index
scans even when exact sequential scans would yield perfect recall at comparable latency; and (3) partition-based
indexes (IVFFlat) outperform graph-based indexes (HNSW) for low-selectivity queries. To facilitate this analysis, we
extend the widely-usedANN-Benchmarksto support filtered vector search and make it available online. Finally, we
synthesize our findings into a set of practical guidelines for selecting index types and configuring query optimizers
for hybrid search workloads.
1 INTRODUCTION
Vector Databases have emerged as an important component in the modern AI stack, serving as the long-term memory
for Large Language Models (LLMs) [25]. In particular, they mitigate hallucinations by providing relevant context from
external knowledge bases to Retrieval-Augmented Generation (RAG) applications [22, 29]. This is done by retrieval
from unstructured data, such as text, images, and audio, stored as vector embeddings, usingApproximate Nearest
Neighbor Search (ANNS)algorithms. While standard ANNS is a well-studied problem with efficient algorithms
[35, 15, 36, 20] and established benchmarks [2, 6, 55], real-world deployment reveals that semantic similarity alone is
rarely sufficient for context retrieval.
1.1 Motivation
As vector search systems mature from prototypes to production, they are expected to handlehybrid queries, where
irrelevant search results are excluded by filters over structured metadata [60, 30]. This requirement is ubiquitous across
domains: a medical RAG system must find patients with similar symptoms (vector) who are also within a specific age
range (filter); a legal search engine must locate relevant cases (vector) within a specific jurisdiction (filter); and e-
commerce platforms must recommend visually similar products (vector) that are highly rated (filter) [58, 10].
Despite the universality of these queries, the interaction between structured filtering and vector index traversal
remains a critical blind spot in current research. While recent algorithmic work has proposed specialized fusion
1arXiv:2602.11443v1  [cs.DB]  11 Feb 2026

methods (e.g., ACORN [45], Filtered-DiskANN [15], HQANN [61]), these approaches often sacrifice flexibility,
requiring the index to be tuned for specific attributes or workloads. In contrast, general-purpose Vector Databases aim
to be schema-agnostic and data-agnostic, necessitating the use of generic strategies likepre-filteringandpost-filtering.
This creates a gap between algorithmic theory and real-world applications. First, the literature lacks a unified tax-
onomy to describe these strategies, leading to ambiguous definitions and inconsistent implementations across systems.
Second, while generic strategies are flexible, their performance limitations and interplay with system components
remain underexplored.
Furthermore, the community lacks the tooling to properly benchmark these scenarios. Existing benchmarks fail to
stress these systems realistically; they are predominantly ”flat,” relying on image-based datasets (like SIFT or GIST)
or pure text corpora (like BEIR [54]) with synthetic or limited metadata. Crucially, they lack robust correlation metrics
to distinguish whether a performance drop is due to the system’s architecture or the intrinsic ”hardness” of the filter-
vector distribution. Without a rigorous evaluation of these factors, it remains unclear how architectural choices of
Vector Databases impact the trade-off between recall and query latency in filtered workloads.
1.2 Contributions
To address these challenges, we present a comprehensive evaluation of generic filtering strategies within vector
database systems. Our specific contributions are as follows:
•Taxonomy and System Analysis.We systematize the taxonomy of filtering strategies (Pre-, Post-, and Runtime-
filtering) and analyze their implementation in three widely-used systems:FAISS,Milvus, andpgvector. We
highlight how system-level architectural decisions, such as adaptive query execution and query planning, often
override raw algorithmic properties in production environments.
•Global-Local Selectivity (GLS) Metric.We propose theGlobal-Local Selectivity (GLS)correlation metric
to rigorously quantify the independence between metadata filters and vector neighborhoods. Unlike previous
distance-based metrics, GLS normalizes local filter prevalence against global selectivity, providing a robust
signal for indexing decisions.
•FANNS Dataset and Workload.We conduct our experiments on a new relational dataset,MoReVec, comprising
two tables:MoviesandReviews. This dataset features 768-dimensional dense text embeddings and a rich schema
of scalar and categorical metadata, enabling the evaluation of filtered ANNS queries. Furthermore, the relational
schema is designed to support future benchmarking of ANNS queries with join predicates.
•Open-source Benchmarking Framework.We release an extension of the industry-standardANN-Benchmarks
framework capable of executing filtered queries with dynamic selectivity targets, facilitating future research into
hybrid search performance. Our extended framework, the MoReVec dataset, and GLS correlation analysis tools
are publicly available [3].
•Empirical Insights on System Behavior.Through extensive experimentation, we reveal counter-intuitive per-
formance characteristics in modern Vector Databases:
–Algorithmic Adaptation: Milvusachieves superior recall stability at low selectivities compared to standard
implementations by employing a hybrid execution model that dynamically switches between “Dual-Pool”
graph traversal and adaptive brute-force fallbacks.
–Optimizer Limitations:We identify thatpgvector’s cost-based optimizer frequently misjudges execution
costs, favoring approximate index scans even when exact sequential scans would yield perfect recall at
comparable latency.
–Index Capabilities:Contrary to standard ANNS wisdom, we demonstrate that partition-based indexes
(IVFFlat) often outperform graph-based indexes (HNSW) in low-selectivity regimes due to efficient cluster
pruning.
2

– Practical Guidelines.We synthesize our findings into a set of actionable guidelines for index selection,
parameter tuning, and query plan verification, providing practitioners with a roadmap for optimizing hybrid
search workloads in production.
Organization.The remainder of this paper is organized as follows. We begin with Section 2, which reviews
recent work in this field, followed by the necessary background on Approximate Nearest Neighbor Search (ANNS) in
Section 3. Section 4 describes our proposed Global-Local Selectivity (GLS) correlation metric. Section 5 systematizes
the taxonomy of filtering strategies and details the specific architectures of the systems evaluated. Section 6 outlines
our benchmarking framework, introducing the MoReVec dataset and workload. Section 7 presents the experimental
results, analyzing the impact of selectivity and system architecture on performance. Finally, Section 8 concludes with
a summary of our findings and potential future directions.
2 RELATED WORK
The systematization of Filtered ANNS (FANNS) draws upon high-dimensional indexing, hybrid search algorithms,
and the evolving architecture of modern Vector Databases.
Table 1: Comparison between this work and other recent FANNS benchmarks.
Paper Description
Research Goal & Framing
This Work System-Level Analysis: Focuses on production Vector Databases (FAISS, Milvus, pgvec-
tor). Systematizes generic filtering (Pre/Post/Runtime) and investigates how architectural
choices (algorithmic adaptability, query planning) override raw algorithmic performance.
Shi et al.[51]Unified Benchmark: Focuses on robustness and fairness. Establishes a standardized tuning
protocol and classification of strategy families (Filter-then, Search-then, Hybrid) to mitigate
implementation bias.
Li et al.[31]Component Taxonomy: Focuses on internal index levers. Analyzes performance variability
through specific components like pruning heuristics, entry-point selection, and edge-filter
costs.
Iff et al.[19]Algorithm Survey: Focuses on specialized/fusion methods. Benchmarks state-of-the-art
algorithms (ACORN, Filtered-DiskANN) specifically on Transformer-based embeddings.
Dataset Model
This Work Relational (IMDb): Two tables (Movies, Reviews) with 768d text embeddings and several
real-world attributes. Has 1-to-many foreign key between tables, can be used for ANNS
with JOIN-filtering.
Shi et al.[51]Real-World Curated: Selection of diverse datasets, unified into a single harness to test
general robustness across varying data distributions.
Li et al.[31]Synthetic + Real: 4 datasets (up to 10M vectors) mixing real and synthetic attributes to
allow precise control over selectivity sweeps (0.1%–100%).
Iff et al.[19]Flat (ArXiv): Single table with 2.7M vectors and 11 real-world attributes. Large-scale text
data designed to test fusion algorithms.
Key Insights
This Work Architecture Dominates: Milvus’s algorithmic adaptability and brute-force fallbacks sta-
bilize recall; pgvector’s optimizer often chooses suboptimal approximate scans. Introduces
GLS Correlationto quantify filter independence.
Continued on next page
3

Table 1 – continued from previous page
Paper Description
Shi et al.[51]Tuning Sensitivity: Standardized tuning significantly alters the ranking of strategies;
”Filter-then” vs. ”Search-then” trade-offs are highly sensitive to implementation details.
Li et al.[31]Pruning Drivers: Performance is driven by pruning granularity and edge-traversal costs;
provides practical tuning guidelines per attribute type.
Iff et al.[19]Scale Failures: Fusion methods (like ACORN) show promise but can struggle with index
size and build times at scale; no universal winner across all selectivities.
2.1 Vector Database Architectures
The integration of ANNS into database systems has followed two paths:Specialized Vector DatabasesandRe-
lational Extensions. Systems likeMilvus[57] andPinecone[48] utilize distributed architectures optimized for
horizontal scaling and data segmentation. In contrast, extensions likepgvector[46] andAnalyticDB-V[60] embed
vector types into the relational core, leveraging existing cost-based optimizers (CBO). Alternatively, other approaches
within the PostgreSQL ecosystem explore decoupling the vector search engine to bypass the overhead inherent from
page-based design [23]. A critical research gap exists in understanding how these architectural choices—rather than
just the underlying algorithms—impact performance. Recent studies comparing RDBMS and specialized Vector DBs
[66] suggest that the performance gap is narrowing, yet the specific failure modes of query optimizers in the presence
of high-dimensional vectors are not well-documented.
2.2 Filtered-ANN Algorithms
Recent algorithmic work has attempted to integrate metadata constraints directly into the index structure to avoid
the “recall cliff” associated with post-filtering.Filtered-DiskANN[15] introduced a filter-aware graph construction
strategy, whileHQANN[61] utilized joint quantization of vectors and metadata. More recently,ACORN[45] pro-
posed using hierarchical Navigable Small World graphs to maintain connectivity even under low selectivity predicates.
While these “fusion” methods offer high efficiency, they often require the index to be specialized for specific attributes,
limiting their utility in general-purpose databases that must support ad-hoc queries over arbitrary schemas.
2.3 ANNS Benchmarking
The foundations of ANNS are built upon four algorithmic pillars:graph-based(e.g., HNSW [35], Vamana [21]),
partition-based(e.g., IVFFlat, LSH [20]),quantization-based(e.g., PQ [26]), andtree-basedmethods (e.g., Annoy
[5]). The performance of these algorithms is typically evaluated via theANN-Benchmarksframework [6], which for-
malizes the recall-vs-throughput trade-off. However, traditional benchmarks focus almost exclusively on “flat” vector
search, utilizing datasets like SIFT and GIST that lack structured metadata. Consequently, the impact of selection
predicates remains under-explored in standard literature.
2.4 Differentiation from Recent FANNS Benchmarks
Recent studies have begun to address the complexities of Filtered ANNS. Notably,Shi et al.[51],Li et al.[31] andIff
et al.[19] provide valuable benchmarks for hybrid vector search. However, our work diverges from these studies in
three critical aspects, which are detailed in Table 1.
System Architecture vs. Algorithmic Fusion.While recent benchmarks primarily evaluate specialized algo-
rithms, including fusion-based approaches (e.g., ACORN, UNG) that tightly couple metadata with index structures,
we focus on thesystem-level integrationof generic filtering strategies (Pre-, Post-, and Runtime-filtering). We contend
that fusion methods, while theoretically efficient, lack the schema-agnosticism required for general-purpose Vector
Databases. Our results demonstrate that system architectural choices, such asMilvus’s algorithmic adaptability or
pgvector’s query optimizer cost models, often override raw algorithmic performance.
4

Relational vs. Flat Data Models.Unlike prior benchmarks that rely on ”flat” datasets with simple tag-based
metadata [19, 31], we introduce a relational dataset withMoviesandReviewstables connected via a foreign key,
enabling the evaluation of complex query plans (e.g., joins) inherent to real-world database workloads.
Query-Filter Relationship Study.We propose theGlobal-Local Selectivity (GLS)correlation, a normalized
metric that explicitly isolates whether a filter enriches or depletes the semantic neighborhood, providing a clearer signal
for index selection. To the best of our knowledge, prior benchmarks have not explicitly quantified this correlation
between filter predicates and query vector neighborhoods.
3 PRELIMINARIES
In this section, we outline the fundamental concepts of vector similarity search, the indexing structures evaluated in
this work, and the existing metrics for quantifying filter-query correlations.
3.1 Vector Representations and Embeddings
Modern information retrieval relies heavily ondense vector embeddings[27] — fixed-dimensional representations of
unstructured data (text, images, audio) where semantic similarity is preserved as geometric proximity [37]. Unlike tra-
ditional sparse methods (e.g., BM25 [50], TF-IDF), dense embeddings capture semantic nuance, enabling the retrieval
of conceptually related items that may not share exact keywords.
The current state-of-the-art for text embeddings utilizes Transformer-based architectures (e.g., GPT [42, 8], BERT
[11]) and contrastive learning [33, 65]. These models map variable-length text into a fixed-size vectorv∈Rd(com-
monlyd= 768ord= 1024).
MTEB Leaderboard.TheMassive Text Embedding Benchmark (MTEB)[39, 40] has emerged as the standard
for evaluating embedding quality across diverse tasks, including retrieval [54], clustering, and classification. It ranks
models based on their generalized performance, guiding the selection of embedding models for production systems.
3.2 Similarity Metrics
Vector similarity metrics quantify the semantic relationship between a query vectorqand a database vectorv. These
metrics generally fall into two categories:distance-based metrics, which measure the geometric distance in vector
space, andangle-based metrics, which measure directional alignment. While recent works have proposed fusion
distances that aggregate vector similarity with metadata distinctness [61], generic Vector Databases predominantly
rely on three standard metrics:
Lp-norm. This metric generalizes the standard Euclidean (L 2) and Manhattan (L 1) distances. It is defined as
d(v, q) =||v−q||p= (Pn
i=1|vi−qi|p)1/p. Euclidean distance (L 2) is the most ubiquitous metric. However,
it is susceptible to the “Curse of Dimensionality”. As dimensionality increases, the volume of the space expands
exponentially, causing the ratio of distances between the nearest and farthest points to approach 1. This loss of contrast
can make distance-based discrimination difficult in extremely high-dimensional sparse spaces, though modern dense
embeddings (e.g., 768 dimensions) generally retain sufficient structural locality.
Inner Product. Defined asd(v, q) =⟨v, q⟩=Pn
i=1viqi, this metric projectsvontoqscaled by the magnitude of
q. It is computationally efficient but sensitive to vector magnitude; for example, two identical large vectors will yield
a higher similarity score than two identical small vectors, which may not be desirable for all semantic tasks.
Cosine Similarity. This metric measures the cosine of the angle between two vectors, effectively normalizing the
inner product:d(v, q) =⟨v,q⟩
||v||||q||. This is mathematically equivalent to the Inner Product ofL 2-normalized vectors
[1]. It is widely favored in text retrieval as it captures semantic orientation independent of document length (vector
magnitude); consequently, it serves as the standard metric for the majority of embedding models evaluated on the
MTEB leaderboard [40].
5

Table 2: Notation and key parameters.
Symbol Description
Dataset and Query
D, NDataset of vectors and its cardinality (|D|=N)
QSet of query vectors
dVector dimensionality (768 in our experiments)
kNumber of nearest neighbors to retrieve
Filtering and Selectivity
ϕ(v)Filter predicate function, returns 1 if valid, 0 otherwise
σg Global Selectivity: Fraction ofDsatisfyingϕ
σl Local Selectivity: Fraction ofk-NN satisfyingϕ
rSelectivity Ratio:σ l/σg(as proposed in [62])
ρq GLS Correlation: Per-query filter-vector independence
¯ρMean GLS correlation acrossQ
Index Parameters
MHNSW: Maximum number of edges per node
efconst HNSW: Candidate queue size during construction
efsearch HNSW: Candidate queue size during search
nprobe IVFFlat: Number of clusters (centroids) to search
3.3 Indexing and Search Quality
Thek-NN problem involves scanning through the entire set of vectors while maintaining a record of thek-nearest
neighbors found thus far. This exhaustive search requires the distance calculation between allNvectors with the
query vector, resulting in a time complexity ofO(Nd)for a dataset ofNvectors. Although polynomial, this approach
demands substantial computational resources, especially for high-workload systems with large vector collections.
To address that, ANNS algorithms have been developed to trade small amounts of accuracy for significant speed
improvements [32].
A wide variety of these indexing strategies exists, supported by established evaluation frameworks [2]. They are
usually classified in four categories:graph-based, such as HNSW, Vamana,partition-basedsuch as IVF, Locality
Sensitive Hashing,quantization-based, such as PQ, andtree-based, such as ANNOY , k-d tree algorithms [21, 26,
20, 7, 41]. Some algorithms may combine multiple approaches into one. Below we will discuss the details of some
commonly implemented ANNS algorithms: Inverted File Index, which is a partition-based algorithm, and HNSW
which is graph-based algorithm.
TheInverted File Index(IVF) is a data structure used in information retrieval systems to locate documents con-
taining specific terms efficiently. It establishes a link between the termst iand the set of documentsD icontaining
them. For the ANNS, Inverted file indexes are often complemented by V oronoi diagrams or k-means clustering to
partition a space into regions based on proximity to a set of points or objects [26, 53]. By associating each vector
with a point in the V oronoi diagram or cluster centroid, queries can efficiently pinpoint relevant documents, thereby
optimizing the retrieval process. This leverages spatial relationships between documents and also decreases the search
space for ANNS problems. When the association is based on pure distance (L2 or Cosine) this algorithm is known as
IVFFlat.
Hierarchical Navigable Small-Worlds (HNSW)is a graph-based indexing algorithm that extends the Navigable
Small World (NSW) model to enable logarithmic scalability for high-dimensional vector search [35]. The standard
NSW algorithm iteratively builds a graph by connecting each vector to its nearest neighbors and adding some long-
range links that connect different regions of the vector space.
The core idea of HNSW is to separate connections by length: long-range links (shortcuts) are placed in upper layers
to facilitate rapid traversal across the vector space, while short-range links remain in the lower layers for fine-grained
local search. For that, vectors are inserted consecutively into the graph structure. For every inserted element, an integer
maximum layerlis randomly selected with an exponentially decaying probability. This probabilistic assignment
6

ensures that while all vectors are present in the lowest layer, only a sparse subset of ”hub” vertices populates the upper
layers, acting as expressways into the graph.
The search process begins at the highest layer and proceeds top-down. The algorithm greedily traverses the graph
elements in the current layer until a nearest neighbor is reached, at which point it descends to the next layer below.
This ”zoom-in” strategy allows the algorithm to ignore the vast majority of distance computations. Upon reaching the
base layer, a candidate priority queue is maintained to track the nearest neighbors found. The size of this queue is
controlled by the parameteref search , which serves as a tunable trade-off between search latency and recall quality.
Recall@kserves as the primary metric for evaluating the accuracy of the approximate search. It measures the
proportion of the true nearest neighbors that are successfully retrieved by the index and defined as:
Recall@k=TP
TP+FN=TP
k(1)
.
Here,True Positives(TP) represent the number of relevant, whileFalse Negatives(FN) represent the true neighbors
missed by the approximation. Since TP+FN=k, this simplifies to the fraction of the top-kground truth found.
There is also another important characteristic, which is the speed of query processing, which is measured as
number of queries processed per second orQPS(Queries Per Second). Indexing methods are usually evaluated as a
graph where QPS is a function of Recall [6].
3.4 Filter Types
Filtering predicates in vector search generally fall into two primary categories based on the nature of the constraint:
metadata-based and distance-based.
Metadata-based filtersenforce constraints on the structured attributes associated with vector embeddings (e.g.,
tags, timestamps, or unique identifiers). These are the direct equivalent of theWHEREclause in relational SQL and are
now standard across modern Vector Databases [38, 48, 46, 49]. They can be subdivided into:
•Categorical filters: Exact matches on discrete labels (e.g.,category = ’electronics’) or set member-
ship checks (e.g.,tags CONTAINS ’urgent’). There is a benchmark dedicated specifically for such filters
with many algorithms aimed to solve it [52, 36, 24].
•Scalar filters: Numerical range queries on continuous values (e.g.,price > 50.0oryear BETWEEN
2020 AND 2024).
•String filters: Pattern matching operations using standard substring search (e.g.,LIKE) or regular expressions.
Distance-based filters, often referred to as a separate problem, known asRange Search, leverage the distance
between vectors to prune results. Instead of a fixedk-nearest neighbors retrieval, these filters return all vectors falling
within a specified radius of the query (e.g.,cosine similarity > 0.8) [9].
A third, emerging paradigm is Semantic filtering, which operates outside these traditional categories. Here, con-
straints are defined by natural language predicates rather than structured schema fields. These systems typically employ
a secondary LLM step to evaluate conditions that require semantic reasoning (e.g., “find papers associated withvector
databases” where the topic is inferred rather than explicitly tagged) [44].
In this work, we focus exclusively onMetadata-based filters—specifically scalar inequalities—as they represent
the standard integration point for Hybrid Search in production Vector Databases (see Section 6.4).
Throughout this work, we adopt the standard database definition where selectivityσrepresents thefractionof valid
rows satisfying filter predicates. Consequently, we use the termlow selectivityto denote a strict filter (whereσ→0,
e.g.,1%), andhigh selectivityto denote a relaxed filter (whereσ→1, e.g.,50%).
3.5 Query-Filter Correlation: Distance-based Approach
In FANNS, the integration of filters with vector similarity introduces challenges in balancing recall and latency. These
challenges are amplified when filter attributes exhibit a correlation (or anti-correlation) with the spatial distribution of
the vector embeddings.
7

a) Distance-Based Correlation (e.g., ACORN)
df
dr
Correlation:C(D, Q)≈0⇐⇒d f≈dr
Interpretation:No correlation.
Observation:Thoughd f≈d r, majority of
valid filtered points are far away(sensitive to
local density).b) Global-Local Selectivity (GLS) Correlation
Q
Local selectivity:σ l≈0.33
Global selectivity:σ g≈0.2
Selectivity ratio:r≈1.65
Correlation:ρ≈0.245
Interpretation:ρ >0→More valid candi-
dates around queryQ(normalizes for density)
QueryQ
Random SampleFiltered (Valid)
k-Nearest Neighbors (filter disregarded)Unfiltered df=g(q, X pi)- distance to nearest filtered
dr=g(q, R i)- distance to nearest in sample
Figure 1: Comparison between correlation metrics. (a) The distance-based correlation [45] is sensitive to geometric
density, potentially misidentifying dense clusters as non-correlated. (b) The proposed GLS correlation normalizes for
density by comparing local selectivityσ lto the global baselineσ g.
State-of-the-art approaches, such as ACORN [45], quantify this relationship usingdistance-based approach. In
particular, they compare the distance distribution of filtered results against a random baseline, defining the query-filter
correlationC(D,Q)for a workloadQover datasetDas the expected difference in minimum distances between the
query vector and the equally-sized filtered and random subsets ofD:
C(D,Q) =E (xi,pi)∈Q[ERi[g(xi, Ri)]−g(x i, Xpi)],(2)
where:
•Xpiis the subset of vectors satisfying the filter predicatep i.
•Riis a random subset drawn uniformly fromD, with|R i|=|X pi|.
•g(x, S) = min y∈Sdist(x, y)is the distance from queryxto the nearest vector in setS.
Intuitively, a positiveC(D,Q)indicates that the filtered vectors are spatially clustered around the query (closer
than random chance), suggesting high local density. Conversely, a negative value implies the filtered subset is sparser
or more distant than a random sample. While this metric effectively captures geometric “hardness,” it is susceptible
to the influence of thelocal neighborhood density, the varying average distance from each queryqto its neighbors.
By averaging minimum distances, that metric conflates global clustering with per-query locality, failing to distinguish
enrichment (high local selectivity) from mere proximity in dense regions. Furthermore, estimating such a correlation
metric for query optimization remains an open challenge.
4 GLOBAL-LOCAL SELECTIVITY CORRELATION
We introduce theGlobal-Local Selectivity (GLS) Correlation, a metric that quantifies the relationship between a
filter’s global selectivity and its selectivity within the query’s local neighborhood. It explicitly reveals whether filters
enrich or deplete the relevant neighborhood around the query vector on a per-query level.
Consider a datasetDcomprisingNvectorsv∈Rd. A filterϕ:D →0,1is applied to select vectors satisfying
a predicate, with the assumption that the filtered subset is non-empty (i.e.,|v∈ D |ϕ(v) = 1|>0; empty filters are
8

ignored in analysis). Theglobal filter selectivity, denotedσ g, quantifies the baseline prevalence of the filter across the
entire dataset:
σg=|{v∈ D |ϕ(v) = 1}|
N∈(0,1](3)
This serves as a corpus-level prior, independent of any specific query.
For a set of query vectorsQwhere eachq∈ Q ⊆Rd, we define the local neighborhoodN qas thek-nearest
neighbors ofqinD, with|N q|=k. Thelocal filter selectivityσ lthen measures the fraction of this neighborhood that
passes the filter:
σl=|{v∈ N q|ϕ(v) = 1}|
k∈[0,1](4)
Here,σ l> σ gindicates local enrichment (filter attributes cluster with similar vectors), whileσ l< σ gsuggests
depletion (anti-correlation).
To derive a directional correlation from these selectivities, we compute the rawselectivity ratio:
r=σl
σg∈[0,∞),(5)
which normalizes the local prevalence against the global baseline:r= 1denotes neutrality,r >1enrichment, and
0< r <1depletion. We note that recent work has independently identified this ratio as a critical factor for estimating
the computational cost of filtered ANN search [62]. However, the unbounded nature ofrlimits its utility in two
key ways. First, it complicatesaggregation: a simple mean is highly sensitive to outliers, where rare global filters
appearing in local neighborhoods can yield exponentially large ratios that skew the dataset-level statistic. Second, for
optimization, cost models typically require normalized inputs (e.g.,[−1,1]or[0,1]) to establish stable thresholds
for plan switching. An unbounded metric makes it difficult to define consistent heuristics for when to prefer specific
indexing strategies. We thus apply the bilinear (M ¨obius) transformation to obtain the per-queryfilter-vector selectivity
correlationρ q:
ρq=r−1
r+ 1∈[−1,1)(6)
This maps the ratio symmetrically to[−1,1), whereρ q= 0signals neutrality,ρ q>0positive correlation (enrichment
strength), andρ q<0negative correlation (depletion strength). The transformation is monotonic and differentiable,
with derivativef′(r) =2
(r+1)2>0. The inverse, for denormalization, isr= (1 +ρ q)/(1−ρ q)forρ q∈[−1,1).
Finally, to aggregate across queries and assess systematic entanglement, we define themean filter-vector selectivity
correlation
¯ρ=1
|Q|X
q∈Qρq∈[−1,1)(7)
We can also obtain a decent estimate of this metric by finding the selectivity of approximatek-nearest neighbors
using an ANN-index(approximate local selectivity)and the selectivity of a sample drawn uniformly from our dataset
(approximate global selectivity).
5 TAXONOMY OF FILTERING STRATEGIES AND SYSTEMS
5.1 FANNS Strategies
When the set of returned results must also satisfy certain metadata (or structured data) constraints, such a query is
generally known as ahybrid queryorfiltered ANNSorfiltered vector search. In these cases, we need to apply a selec-
tion operation on the subset of elements that meet the given condition, which is commonly referred to asfiltering. In
relational database management systems (RDBMS), this operation is known as the application of aselection predicate.
9

QP1P2 P3
P4P5P6
P7P8P9
P10P11P121
2
3
45Starting point (passing filter)
Points passing filter
Irrelevant points
Top-3 true results
Top-3 search results
Query point
True result max-distance circle
Graph traversal path
Candidate paths
Step Current Results Neighbors Candidates Visited
0 P2 [] P3, P4, P11 P3, P4, P11 P2, P3, P4, P11
1 P3 P5, P9, P2 P2, P5, P9 P5, P9, P4 P2, P3, P4, P5, P9, P11
2 P5 P1, P5, P10 P1, P3, P10 P1, P10, P9 P1, P2, P3, P4, P5, P9, P10, P11
3 P1 P1, P5, P10 P5, P6, P9 P10, P9∗P1, P2, P3, P4, P5, P6, P9, P10, P11
4 P10 P1, P5, P10 P4, P5, P7 P9∗P1, P2, P3, P4, P5, P6, P7, P9, P10, P11
5 P9 P1, P5, P10 P1, P3, P6 []∗P1, P2, P3, P4, P5, P6, P7, P9, P10, P11
∗Candidates are not updated since there is no neighbor pointn∈Neighbors, s.t.d(n, q)< max(d(Results, q))or
|Results|< ef search
Figure 2: Graph traversal using thepre-filteringstrategy in 1-layer HNSW index withm= 3and search parameters
k= 3,ef search= 3.
In this work, we will use the termsSelection PredicateandFilterinterchangeably, assuming they are equivalent across
different systems.
We draw from recent surveys on vector databases [17, 43] to outline the primary strategies for Filtered ANNS
and provide their exact definitions. There are generally three main methods used for filtering in hybrid queries:pre-
filtering,runtime-filtering, andpost-filtering. While these terms may appear intuitively straightforward, the literature
reveals a lack of systematic understanding, resulting in varying definitions and conceptual overlaps across works [43,
34, 58, 51], particularly with respect to pre-filtering. The specifics of each method depend on the underlying index
type (e.g., graph-based or partition-based).
We excludefusion-basedmethods [45, 61], where metadata values are fused into either the index construction or
the distance calculation. We omit these approaches from this study primarily because they lack schema-agnosticism.
By tightly coupling metadata with the vector index, these methods compromise system flexibility and scalability.
Fusion methods struggle with sparse or high-dimensional attribute data; if many vectors lack certain attributes or if the
total number of attributes is large, fusing all of them into a single, compact vector space or metric becomes a complex,
non-trivial problem. Furthermore, fusion methods may require a priori knowledge of the query workload to effectively
integrate metadata into the index or distance metric, limiting their applicability to arbitrary schemas or ad-hoc queries.
For these reasons, they are not treated here as generic filtering strategies for existing ANNS indexes.
5.1.1 Pre-filtering + ANNS
The filter predicate is evaluated upfront to prepare a bitset indicating which vectors pass the filter criteria. This bitset is
then used to guide the vector search, ideally reducing the effective search space by masking irrelevant points. However,
in graph-based indices, this does not necessarily hold.
Forgraph-based indexes, distances must still be calculated during traversal to ensure the connectivity of the graph
and navigate toward the query vector. The bitset is consulted to determine whether a visited vector should be added to
the priority queue of candidates.
Forpartition-based indexes(e.g., IVFFlat), the bitset allows checking the filter before computing distances for each
vector. This approach prunes the search space early, avoiding unnecessary distance computations on non-qualifying
points.
10

Pre-filtering (Milvus, FAISS):
Start
Filter Metadata→Bitset
Loop: Traverse Graph
Check Bitset
Update Results
Check Stopping Criteria
Return Top-KContinueStopRuntime-filtering:
Start
Loop: Traverse Graph
Filter Metadata
Update Results
Check Stopping Criteria
Return Top-KContinueStopPost-filtering (pgvector):
Start
Loop: Traverse Graph
Update Results
Check Stopping Criteria
Filter Metadata
Update Results
Return Top-KContinueStop
Figure 3: Filtering strategies for ANNS.
Note onPre-filtering + kNNS.Sometimes, for queries with low selectivities, vector search engines can apply filter
and search exhaustively among survived points. It is important to distinguish it from Pre-filtering + ANNS, as it solves
problem exactly rather than approximately.
5.1.2 Runtime Filtering
In this approach, attribute retrieval and predicate evaluation occur lazily, triggered only when a candidate vector
is accessed during the search. While this on-demand pattern can introduce latency due to random I/O access, it
significantly reduces the total count of predicate evaluations. This trade-off is particularly advantageous in memory-
constrained environments or when vector filters are computationally expensive, such as complex regular expressions,
string pattern matching, or conditions involving joins across multiple relations.
Ingraph-based indexes, attributes are fetched for the vectors traversed during the graph navigation. To mitigate
the I/O overhead from these repeated fetches, batching strategies can be employed, such as traversing multiple hops
and fetching attributes for all visited points at once.
Inpartition-based indexes, attributes are fetched only for vectors within the search space (relevant clusters in
IVFFlat). This targeted fetching typically incurs no additional I/O overhead beyond the standard cluster access, as the
attributes of all points within the target cluster are loaded together.
We exclude Runtime Filtering from our primary evaluation as it is not natively exposed by the standard APIs of the
evaluated in-memory systems (FAISS, pgvector). While integral to disk-based systems like DiskANN [15] to mask I/O
latency, adapting it for in-memory HNSW requires custom index modification that deviates from the ’off-the-shelf’
system behavior this paper seeks to systematize.
5.1.3 Post-filtering
The filter is applied after the vector search retrieves a set of candidate vectors. This decouples the filtering from the
search but risks including false candidates (similar vectors that fail the filter), which are discarded post-search. It often
performs well in terms of search latency but may suffer from reduced recall. This method is commonly implemented
in relational DBMS systems, such as pgvector in PostgreSQL, and potentially in proprietary systems like Oracle
Database.
Ingraph-based indexes, filtering occurs only on the limited set of vectors in the candidate priority queue (e.g.,
in HNSW, a larger queue size increases traversal latency). For low selectivity predicates uncorrelated with vector
11

similarities, this can lead to incomplete top-kresults (fewer thanksurvivors).
Inpartition-based indexes, search parameters typically allow probing a larger candidate set (e.g., settingnprobe
in IVFFlat to check multiple clusters), reducing the risk of incomplete results compared to graph indexes, though it
persists for extreme selectivity.
5.1.4 Query Optimization for FANNS
Integration of vector search into relational DBMS introduces additional possibilities to write more complex queries,
involving relational operators, such as joins with other tables or filters applied across joined relations. This introduces a
fundamental dilemma in query optimization: whether to first do vector search and then execute the ”relational” part of
query, or to first complete the ”relational” part of the query to determine the set of relevant vectors before performing
the similarity search.
The query optimizer relies on cardinality estimates to choose between execution plans. However, to prioritize
vector search, the optimizer requires accurateselectivityestimates to determine the expansion factor—calculating
k/selectivity candidates to ensurekresults remain after filtering. However, without reliable estimates, the system is
forced to complete the relational join or filter first to generate the subset of qualifying IDs. This subset is then passed
as a constraint to the vector engine, ensuring accuracy at the cost of potential latency.
5.2 Vector Search Systems
All systems designed for vector similarity search can be divided into three main categories: Vector Extensions for
Relational Databases (e.g., PG-Vector [46], PASE [63], VBASE [64]), Specialized Vector Databases (e.g., Milvus [57],
Pinecone [48], Weaviate [59]), and Vector Search Libraries (e.g., FAISS [12], ParlayANN). We will test representative
systems from each category.
System Scalar Categ. Str. Match. Filtering Strategies
Extensions
PG-Vector✓ ✓ ✓Cost-based (Post / Pre + ANNS)
PASE✓ ✓ ✓IVFFlat / HNSW integrated w/ PG
VBASE✓ ✓ ✓Relaxed Monotonicity (Kernel-integrated)
Specialized DBs
Milvus✓ ✓ ✓(regex) Pre-filtering + ANNS
Qdrant✓ ✓ ✓(exact/substr) Pre-filtering (Payload Indexing)
Weaviate✓ ✓ ✓(BM25) Pre-filtering (Inverted Index)
Pinecone✓ ✓ ✓(prefix) Pre-filtering (Metadata Index)
Vespa✓ ✓ ✓(ling) Multi-stage (Pre-filter/Rank)
Elasticsearch✓ ✓ ✓(full-text) Hybrid (Pre/Post/Rescore)
Libraries
FAISS (w/ NumPy)✓× ×Pre-filtering + ANNS (IDSelector)
HNSWlib× × ×Post-filtering (during traversal)
ParlayANN× × ×Parallel Graph-based Search
Table 3: Comparison of vector search systems.
5.2.1 Vector Extensions
Vector Extensionsintegrate vector data types directly into relational database columns, thereby retaining core relational
capabilities such as ACID compliance and queries withJOIN,GROUP BY-predicates via SQL syntax. A key benefit
is seamless adoption within existing relational infrastructures, obviating the need for data migration. Nonetheless,
these extensions may exhibit functional constraints compared to purpose-built solutions. We selectPG-Vector[46]
for its active maintenance and widespread adoption as a PostgreSQL extension [47]. It supports a comprehensive
12

range of filters, encompassing essentially all predicates available in PostgreSQL (e.g., scalar, categorical and string
operations viaWHEREclauses). The cost-based model of the query optimizer allows switching between filtering
strategies, however it doesn’t always choose optimal Filtered ANNS queries as shown in Section 7.
Other notable candidates includeVBASE[64], a system built on PostgreSQL modifies the database kernel to
support”relaxed monotonicity”for smarter search termination check, andPASE[63], an Alibaba-developed extension
on PostgreSQL with IVFFlat and HNSW indexes.
5.2.2 Specialized Vector Databases
Specialized Vector Databaseseschew traditional relational schemas and usually don’t support SQL-like syntax. They
range from schema-less key-value stores to those enforcing predefined schemas for vector collection and often include
native support for embedding generation. Compared to vector extensions, specialized vector databases offer perfor-
mance advantages in indexing and querying, though these can often be mitigated to insignificant levels [66]. We opt
forMilvus[57, 38] due to its open-source nature and robust maintenance. It supports a wide array of filters, includ-
ing scalar and categorical predicates, and string matching via regular expressions.Knowhere, Milvus’s vector search
engine, is built on top of FAISS with customized optimizations. Milvus primarily utilizes a pre-filtering + ANNS
strategy, and employs a logic-based query optimizer, which is evolving in newer versions.
Other representative systems in this category include:
•Pinecone[48]: A proprietary, closed-source cloud database. While implementation details are opaque, it claims
to utilize a custom graph-based index that integrates metadata filtering directly into the index structure. This
approach aims to prevent the graph disconnectivity issues typically associated with aggressive pre-filtering on
standard HNSW graphs.
•Qdrant[49]: An open-source engine (Rust) that employs a ”segment-based” architecture. It maintains indepen-
dent data structures for metadata (e.g., HashMaps for keywords, B-Trees for numeric ranges). During filtered
search, these metadata indices generate a bitset used as a mask to identify qualifying points during vector re-
trieval.
•Weaviate[59]: An open-source engine (Go) that couples an HNSW vector index with a traditional inverted
index (posting lists) for scalar data. For filtered queries, the system first consults the inverted index to retrieve an
allow-list of object IDs (or a bitmap), which is then used to filter candidates during the HNSW graph exploration.
•Vespa[56]: An open-source tensor compute engine. Unlike pure vector stores, Vespa executes queries as tensor
operations. It can also automatically switch to exact search to process the queries with low selectivity filters.
•Elasticsearch[13]: Built on Apache Lucene, it utilizes the Lucene implementation of HNSW. The system
assesses filter selectivity to decide between an approximate search (checking filters during graph traversal) or
an exact brute-force scan of the filtered document set.
5.2.3 Vector Search Libraries
Vector Search Libraries, such as FAISS [12, 14], are lightweight, schema-agnostic tools optimized for vector indexing
and retrieval. Unlike full databases, they do not handle persistent storage; instead, they operate on pre-loaded vectors
for indexing. Filtered ANNS queries require an external bitset to delineate vectors satisfying the filter criteria. We
employFAISSdue to its open-source nature and widespread adoption in the field.
Other notable libraries in this category include:
•HNSWlib[35]: A header-only C++ library implementing the HNSW graph algorithm. It serves as the core
indexing engine for several vector databases (e.g., Chroma).
•ScaNN[16]: Developed by Google, this library introducesanisotropic vector quantization. Unlike standard
product quantization (PQ) which minimizes Euclidean reconstruction error, ScaNN optimizes a loss function
that prioritizes directional accuracy, which is critical for Inner Product search.
13

•ParlayANN[36]: A research library focusing on parallel algorithms for ANNS, including graph indexes (e.g.,
HNSW, Vamana). Its architecture is designed to minimize thread contention and ensure deterministic behavior
on high-concurrency multi-core systems.
6 BENCHMARKING FRAMEWORK
6.1 ANN-Benchmarks
Evaluating ANNS algorithms is notoriously difficult due to varying hardware and implementation nuances. ANN-
Benchmarks [6] addresses this by providing a containerized testing environment for each algorithm, accompanied by
a YAML configuration file that defines build instructions, hyperparameters, and indexing procedures. Once validated,
these are executed on a fixed set of public datasets to evaluate trade-offs between recall (fraction of true nearest neigh-
bors retrieved) and throughput (queries per second, QPS), typically under resource constraints like single-threaded
CPU execution.
The benchmarks employ a curated list of diverse, high-dimensional vector datasets, spanning domains like images
and text, including word embeddings like GloVe-100 (angular distance, 100 dimensions) and image features like SIFT-
128 (Euclidean distance, 128 dimensions), as well as sparse sets such as Kosarak (Jaccard distance, up to 27,983
dimensions), as summarized in Table 4. Evaluations are conducted on standardized hardware, such as AWS r6i
instances, to ensure reproducibility, with performance metrics plotted on logarithmic scales to illustrate the recall-
throughput trade-off alongside secondary factors like index build time and memory footprint. This framework has
benchmarked over 30 algorithms, including prominent libraries such as FAISS, Milvus, PG-Vector, HNSWlib, and
Annoy, with contributions from the community keeping the results up to date.
6.2 Limitations of Existing Datasets
Dataset Name Dim. Cardinality Queries Distance Domain Embedding type|V|∗|A|∗∗
SIFT-1M 128 1,000,000 10,000 L2 Images Hand-crafted 1 0
GIST-1M 960 1,000,000 1,000 L2 Images Hand-crafted 1 0
GLOVE-100 100 1,183,514 10,000 Cosine Text Count-based 1 0
MNIST-784 784 60,000 10,000 L2 Images Raw pixels 1 0
NYTimes-256-ang 256 290,000 10,000 Cosine Text Word2Vec embeddings 1 0
Kosarak 27,983 74,962 500 Jaccard Web Sparse binary 1 0
LAION-5B∗∗∗7685.85×109– Cosine Multi-modal CLIP (contrastive) 2 6
ArXiv∗∗∗768 2,700,000 10,000 Cosine Text Transformer-based 1 10
Wiki-22-12-en 768 35,200,000 – Cosine Text Transformer-based 1 5
YFCC∗∗∗4,096 100,000,000 – L2 Images Deep CNN 1 5
Youtube audio∗∗∗128 6,100,000 – L2 Audio Hand-crafted 1 2
Youtube video∗∗∗1,024 6,100,000 – L2 Video Hand-crafted 1 2
Table 4: Datasets used for ANNS benchmarking [4].
∗|V|– number of vectors per each record.∗∗|A|– number of metadata attributes per each record.∗∗∗Not included in
ANN-Benchmarks; added here for comparison with emerging text-focused datasets.
Despite their rigor, existing ANN datasets suffer from a key shortfall: they are predominantly derived from non-text
embeddings (e.g., image or generic word vectors), limiting relevance to modern NLP tasks like dense text retrieval.
The LAION-5B dataset, a popular large-scale alternative, mitigates scale issues with billions of image-text pairs but
restricts attributes to basic categorical labels (e.g., aesthetics scores, content flags), lacking the continuous or versatile
metadata needed for nuanced evaluations.
ArXivdataset is a notable exception, with transformer-based embeddings from over 2.7 million arXiv abstracts
alongside 11 structured attributes (e.g., numerical citation counts, categorical domains, and temporal metadata). Sim-
ilarly,Wikipedia-22-12-en-embeddingsoffers a massive scale of 35 million transformer-based vectors derived from
Wikipedia. While it includes valuable structured metadata, it introduces a significant structural complexity: unlike
14

standard 1-to-1 mappings, this dataset assigns an arbitrary number of vectors to each logical entity (article) because
a separate vector is generated for each paragraph. This creates a ”variable cardinality” problem for entity retrieval,
complicating benchmarking efforts that must distinguish between vector-level (paragraph) similarity and item-level
(article) relevance.
However, these datasets fall short because they cannot support complex joins. Their limited scope—primarily
numerical and categorical fields without deeper relational structures—precludes extensibility for query compositions
like temporal-range joins with categorical hierarchies. Moreover, attributes like author counts, category listings, and
version histories, though scholarly in nature, hold marginal value for broader retrieval tasks beyond niche academic
filtering. For completeness, datasets likeYFCC100Mand theYouTube audio/videosets, included here for multimodal
comparison, are evidently inadequate: YFCC relies on image-centric deep CNN embeddings, while the YouTube
datasets use hand-crafted features and minimal metadata.
6.3 Dataset Design
To address these limitations, we introduceMoReVec(Movies and Reviews Vectors), a dataset designed to facilitate
optimization over vector embeddings and structured attributes.
Drawing from the rich ecosystem of IMDb data used for construction of Join-Ordering Benchmark (JOB) [28]
for query optimizers evaluation in RDBMS, we merge text embeddings of movie synopsis and reviews with available
web-sourced metadata. In particular, we generate 768-dimensional vectors using thegte-base-en-v1.5model [33],
a high-performing embedding method from the Massive Text Embedding Benchmark (MTEB) leaderboard, applied
to textual fields such as movie synopsis and user reviews. All embeddings areL 2-normalized, ensuring that cosine
similarity is equivalent to inner product for all evaluated systems.
The resulting datasets adhere to extensible schemas detailed in Table 5, enabling advanced hybrid queries that
combine semantic similarity searches with attribute-based filtering and extensible join operations. To enable scalability
studies, we instantiate each dataset at three cardinalities:
• Small:|Movies|= 9,999;|Reviews|= 247,286;
• Medium:|Movies|= 99,560;|Reviews|= 1,496,493;
• Large:|Movies|= 551,155,|Reviews|= 2,598,267;
Field Description Data Type
mid Unique alphanu-
meric identifier of
the movieVARCHAR
title Movie title VARCHAR
year Year when movie
was releasedINTEGER
genre Movie genre VARCHAR
avgrating Average rating of
the movieFLOAT
numvotes Number of votes for
the movieINTEGER
mvector Movie synopsis em-
beddingsVECTOR(768)
(a) Movies tableField Description Data Type
rid Unique alphanumeric identifier of
the reviewVARCHAR
uid Alphanumeric identifier of a user
who left reviewVARCHAR
mid Foreign key to mid in movies table VARCHAR
rvector Vector embedding for the review to
a movieVECTOR(768)
movierating Rating assigned to movie by user INTEGER
totalvotes Total votes for the review INTEGER
likeshare Share of positive votes for the re-
viewFLOAT
quality Quality of reviews based on their
totalvotes and likeshareVARCHAR
(b) Reviews table
Table 5: Schema of the proposed dataset.
15

0.013 0.059 0.110 0.224 0.521
Selectivity1.0
0.5
0.00.51.0Distance Correlation
avgrating - (numerical)
0.067 0.086 0.174 0.184 0.249
Selectivity1.0
0.5
0.00.51.0
genre - (categorical)
0.010 0.050 0.101 0.210 0.509
Selectivity1.0
0.5
0.00.51.0
totalvotes - (numerical)
0.021 0.109 0.137 0.350 0.384
Selectivity1.0
0.5
0.00.51.0
quality - (categorical)
0.013 0.059 0.110 0.224 0.521
Selectivity1.0
0.5
0.00.51.0GLS Correlation
avgrating - (numerical)
0.067 0.086 0.174 0.184 0.249
Selectivity1.0
0.5
0.00.51.0
genre - (categorical)
0.010 0.050 0.101 0.210 0.509
Selectivity1.0
0.5
0.00.51.0
totalvotes - (numerical)
0.021 0.109 0.137 0.350 0.384
Selectivity1.0
0.5
0.00.51.0
quality - (categorical)Figure 4: Distribution of per-query correlation valuesρ qbetween metadata attributes and vector embeddings at differ-
ent target selectivities.
6.4 Workload
Filters. Figure 4 shows that the meanGLS correlation¯ρbetween vectors and metadata attributes tends to 0 for all
examined attributes across the breadth of the selectivity range in both datasets. At low selectivity levels (σ g≤0.1), we
observe a notable deviation toward negative values for some attributes. This can be partially attributed to a sampling
artifact arising from the finite neighborhood size (|N q|= 2048); when global selectivity is sufficiently low, the
expected number of valid neighbors within the fixed neighborhood is small. Consequently, queries often yield fewer
valid local neighbors than the expected value (σ l< σg), drivingρ qto−1and skewing the mean.
Accounting for this artifact, the results confirm that metadata attributes are mostly independent of the embedding
space, allowing us to study the isolated effect of filter selectivity on filtered ANN performance. We therefore apply nu-
meric/scalar filters on theavgratingcolumn (Movies dataset) andtotalvotescolumn (Reviews dataset). Filter
thresholds are chosen such that the expected selectivityσ g(fraction of the dataset passing the filter) approximately
matches the following target values:
• Small dataset:σ g≈ {0.01; 0.03; 0.05; 0.1; 0.2; 0.5}
• Medium dataset:σ g≈ {0.0003; 0.001; 0.01; 0.03; 0.05; 0.1; 0.2}
• Large dataset:σ g≈ {0.0003; 0.001; 0.01; 0.03; 0.05; 0.1}
1.0
 0.5
 0.0 0.5 1.0
Cosine Similarity01234Probability DensityVar: 0.008Movies-Movies
Min: -0.072
Max: 0.791
Mean: 0.311
1.0
 0.5
 0.0 0.5 1.0
Cosine Similarity012345Probability DensityVar: 0.007Reviews-Reviews
Min: -0.103
Max: 0.790
Mean: 0.294
1.0
 0.5
 0.0 0.5 1.0
Cosine Similarity012345Probability DensityVar: 0.005Movies-Reviews
Min: -0.155
Max: 0.620
Mean: 0.241
Figure 5: Distribution of pairwise cosine distances for 1,000,000 randomly sampled vector pairs. Left: Movie embed-
dings only; Middle: Review embeddings only; Right: Mixed pairs with one movie and one review embedding each.
16

Queries. Since semantic-search queries are derived from the same embedding model as the data, we assume that
distributions of vectors fromMoviesandReviewstables are similar. This is supported by Figure 5, which shows the
distribution of pairwise cosine distances for vector pairs from each table and their mixture.
Accordingly, we randomly sample 1,000 vectors from each dataset to serve as the query set. Each query seman-
tically corresponds to”find top-kclosest vectors to query vectorqwhere value of scalar attribute is greater than or
equal to thresholdt”. We test four different values ofk:k∈ {1,10,40,100}. Each of the 1000 query vectors is
executed with every filter condition and all fourk, yielding1000×7×4 = 28,000filteredk-NN queries per dataset
(including no filter runs).
We did not include multi-tableJOINpredicates to our query workload. While pgvector supports them syntacti-
cally, its query optimizer consistently reverts to sequential scans for such queries, bypassing the vector index entirely.
Conversely, schema-agnostic systems like Milvus and FAISS lack the relational operators to execute joins natively.
7 EMPIRICAL EV ALUATION
We perform extensive empirical comparison of FANNS performance acrossFAISS,Milvus, andpgvector. We focus
specifically on the trade-offs between query throughput and recall, as well as the underlying query planning and
execution strategies employed by each system. Our main goal is to determine how generic filtering strategies (pre-
filtering and post-filtering) perform within real-world Vector Databases, and to what extent system-level architectural
choices override the expected algorithmic behavior of the underlying index structures. To this end, our evaluation
investigates the following questions:
•How does filter selectivity impact the recall-throughput trade-off?We examine whether low-selectivity filters
universally degrade performance, or whether certain index types and filtering strategies adapt better to sparse
result sets.
•Does the traditional dominance of HNSW over IVFFlat hold under filtered search?We test whether the graph-
based traversal of HNSW remains superior to the cluster-pruning mechanics of IVFFlat across varying selectiv-
ity regimes.
•How do system-level architectural decisions affect performance?We investigate whether factors such as algo-
rithmic adaptability, data segmentation (Milvus) and cost-based query optimization (pgvector) have a greater
impact on filtered ANNS performance than the choice of index algorithm itself.
•Can query optimizers in relational vector extensions make effective plan selections for hybrid queries?We
analyze whether pgvector’s cost-based optimizer reliably selects execution plans that balance recall and latency.
To facilitate reproducibility and further research, our extendedFiltered ANN-Benchmarksframework, the MoReVec
relational datasets, and our GLS correlation analysis tools are available on GitHub [3].
Table 6: Software versions used in experiments.
System Server/Library Version Notes
Milvus v2.6.6 pymilvus 2.6.3
FAISS 1.12.0 (faiss hnsw) CPU version
pgvector 0.8.1 (git main) PostgreSQL 16
7.1 Experimental Setup
Indexes.We evaluate two widely implemented ANN indexing methods:HNSWandIVFFlat. For theHNSWindex,
we explore the following hyperparameter configurations:
• Construction parameters:(M, ef construction )∈ {(5,25),(10,50),(15,75)}
17

• Search parameter:ef search∈ {1,10,40,100,200,500,1000}
For theIVFFlatindex, we vary the number of centroids to control the trade-off depending on dataset size:
• Construction parameters:clusters={small: (100,500), medium: (300,1200), large: (750,1600)}.
These ranges roughly follow the commonC≈√
Nheuristic, whereCis number of clusters, andNis the
dataset size.
• Search parameters:n probes ={1,5,10,20,50,150,300}
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Movies, @k=10 (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
Reviews, @k=10 (large)(a) HNSW
(b) IVFFlat
FAISS g={S: 0.02, M: 0.01, L: 0.001}
FAISS g={S: 0.11, M: 0.06, L: 0.02}
FAISS g={S: 0.51, M: 0.21, L: 0.11}
FAISS g={S: 1.00, M: 1.00, L: 1.00}
Milvus g={S: 0.02, M: 0.01, L: 0.001}
Milvus g={S: 0.11, M: 0.06, L: 0.02}
Milvus g={S: 0.51, M: 0.21, L: 0.11}
Milvus g={S: 1.00, M: 1.00, L: 1.00}
pgvector g={S: 0.02, M: 0.01, L: 0.001}
pgvector g={S: 0.11, M: 0.06, L: 0.02}
pgvector g={S: 0.51, M: 0.21, L: 0.11}
pgvector g={S: 1.00, M: 1.00, L: 1.00}
Figure 6: QPS–Recall curves at different selectivity levels.
Changes to ANN-Benchmarks Implementation.We extended the ANN-Benchmarks experimental loop to gen-
erate YAML configurations dynamically: one “run group” per index parameter set during preprocessing. In the
query phase, we test filter/k/index/search-parameter combinations on the same built index without rebuilds, encod-
ing filters in HDF5 directory hierarchies and aggregating results in a comprehensive CSV (runtimes and recalls per
filter/k/query/index/search-parameter).
Execution Protocol.Following the ANN-Benchmarks methodology, all queries are executedsingle-threaded, one
query at a time. This design isolates the algorithmic and architectural performance characteristics of each system from
18

concurrency effects, enabling controlled comparisons. All evaluated systems operate inin-memorymode; indexes and
datasets are fully loaded into RAM before query execution begins. We do not evaluate disk-based or streaming work-
loads. Each query is executed once without repetition within the measurement phase; throughput (QPS) is computed
as the inverse of the measured single-query latency. No batch query execution is employed—each query is submitted
and completed individually.
Ground Truth for Filtered Recall.For filtered queries, ground truth is computed by first applying the filter
predicate to the full dataset, then performing anexactk-NN search (brute-force distance computation) over the filtered
subset. Formally, for a query vectorqand filter predicateϕ, we define the ground truth as thekvectors from{v∈ D |
ϕ(v) = 1}with the smallest distance toq.
Hardware and Software Configuration.As in ANN-Benchmarks [6], we run all experiments in a Docker con-
tainer, which includes all necessary dependencies for the evaluated systems. We use a machine running Ubuntu 24.04
LTS equipped with a 56-core Intel Xeon E5-2660 (2.00 GHz) and 256 GB of RAM. Table 6 lists the exact versions of
all evaluated systems. All experiments use the specified versions to ensure reproducibility.
Scope.Our experiments focus exclusively onscalar inequality filters(e.g.,rating >= threshold). Cate-
gorical filters, string pattern matching, and multi-attribute conjunctions are excluded from the experimental workload.
Additionally, we evaluate only in-memory execution; disk-based and runtime filtering strategies are out of scope.
7.2 Results
7.2.1 Impact of Filter Selectivity
As illustrated in Figure 6, an increase in filter selectivity consistently exerts negative pressure on recall for both HNSW
and IVFFlat implementations. This degradation occurs regardless of whether a pre-filtering or post-filtering strategy is
employed, though the mechanisms differ. In pre-filtering, low selectivity filters aggressively sparsify the distribution
of true nearest neighbors, including the region surrounding the query vector. Conversely, in post-filtering, the effective
candidate pool is drastically reduced after the index scan; if the initial search does not retrieve a sufficient number
of candidates satisfying the predicate, the final result set is inevitably truncated.Milvusis the notable exception,
maintaining higher recall at low selectivity due to its hybrid architecture (see Section 7.2.7).
Despite yielding slight recall decrease at lower selectivity levels in general, IVFFlat may demonstrate superior QPS
performance, as indicated by the elevated QPS-Recall curve in Figure 15. This result is expected, as the cluster-pruning
mechanism of IVFFlat reduces the computational overhead of distance calculations.
However, system throughput (QPS) in graph-based indexes remains largely insensitive to variations in selectivity,
as the underlying graph structure necessitates consistent traversal efforts regardless of filter presence. This stability
indicates that the overhead of filtering is negligible compared to the expensive distance computations required during
graph traversal, particularly in large datasets.
Predicate evaluation in FAISS. To satisfy the bitset re-
quirement, we implemented a wrapper around the FAISS
index that handles predicate evaluation using NumPy. For
every query, we first compute a boolean mask indicating
which vectors satisfy the filter, and then convert this mask
into a FAISSIDSelector. This process occurs on-the-
fly with negligible overhead (<1ms on the large datasets).
Figure 7 summarizes the added latency across representa-
tive workloads.
0.1M 0.5M 1.0M 1.5M 2.0M 2.5M 3.0M
Dataset Size0.000.250.500.75Runtime (ms)
Mean ± Std (10 samples)Figure 7: Bitset generation runtime in FAISS.
Takeaway.Filter selectivity acts as a universal constraint on recall. However, its impact on throughput varies
by architecture: graph-based indices exhibit QPS invariance, confirming that traversal costs mask filtering overheads,
whereas partition-based indices (IVFFlat) leverage pruning to achieve superior throughput in low-selectivity regimes.
Consequently, performance in filtered ANNS is determined less by metadata evaluation costs and more by the specific
mechanics of the chosen vector index—balancing traversal robustness against pruning efficiency.
19

7.2.2 Impact ofk
Figure 14 analyzes the relationship between the number of requested nearest neighbors (k) and system recall. Given
that search parameters (such asefSearchornprobe) were held constant, the raw size of the candidate pool
generated by the index remains static. Askincreases, the system must extract a larger number of valid results from
this fixed-size candidate set. This leads to a systematic decrease in recall, as the probability of the fixed set containing
kvalid neighbors diminishes.
This effect is particularly pronounced inpgvector’s HNSW implementation using post-filtering, except when recall
jumps to 1.0 at lower selectivities (triggered by the query optimizer switching to a sequential scan). Since filter is
applied only to the final set of nearest neighbors bounded byefSearch, there is an increased statistical likelihood
that a majority of candidates retrieved by the raw vector search will fail the predicate. Consequently, many true
neighbors are never examined because they were not among the top-efSearchcandidates in the unfiltered vector
space, causing a precipitous drop in recall.
Takeaway.To sustain high recall when increasing the retrieval sizek, it is imperative to proportionally scale the
search parameters (e.g., increasingefSearch). However, this necessitates a trade-off, as larger search parameters
incur higher computational costs and increased query latency.
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Movies, @k=10, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Movies, @k=10, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Movies, @k=10, (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Reviews, @k=10, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Reviews, @k=10, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103104QPS
Reviews, @k=10, (large)
FAISS, M=5
FAISS, M=10
FAISS, M=15Milvus, M=5
Milvus, M=10
Milvus, M=15pgvector, M=5
pgvector, M=10
pgvector, M=15
Figure 8: QPS–Recall curves for HNSW at different values ofM.
7.2.3 Filtering Strategies in HNSW
Post-filtering approach traverses the HNSW graph agnostically, without guidance from the predicate. This results in the
admission of numerous candidates that are ultimately discarded, thereby reducing the effective recall achievable within
theefSearchbudget. Due to that, inferior recall for low-selectivity queries is expected. However, post-filtering
implementation inpgvectorshows comparable and even slightly better performance than pre-filtering implementation
ofFAISSfor non-filtered or high-selectivity queries in terms of runtime. We attribute this topgvectoravoiding longer
graph traversal and early filter evaluation overhead.
AlthoughMilvus’s Knowhere engine is built uponFAISS, its performance characteristics differ substantially, due
to its hybrid architecture (see Section 7.2.7).
Takeaway.While pre-filtering is generally expected to provide superior recall, post-filtering may be the optimal
choice for filters with high selectivity.
20

7.2.4 HNSW-Graph Density
Increasing the HNSW construction parameterM(and implicitlyefConstruction) significantly increases the
index building time inpgvectorandFAISS, as shown in Figure 9. AlthoughMilvusdemonstrates invariance to con-
struction parameters, this also results from its hybrid architecture (Section 7.2.7). HNSW inherently incurs higher
construction costs with largerMvalues, as internal vector searches for node insertion become computationally expen-
sive. This increase in graph density yields only negligible gains in recall and has minimal (or even negative) impact
on search latency across all systems (Figure 8), unless the graph is excessively sparse (e.g.,M= 5). Therefore, when
rapid index construction is prioritized, “lighter” graphs with smallerMare preferable.
5 10 15
M - max degree of the graph101102103104Index Build Time (s)Movies
5 10 15
M - max degree of the graph101102103104Index Build Time (s)Reviews
Milvus pgvector FAISS
Figure 9: Index build times for HNSW.
1.0
 0.5
 0.0 0.5 1.0
Correlation0100200300CountDistribution of Correlations
Distance
GLSFigure 10: Distribution of GLS correlation
and distance-based correlation values.
7.2.5 IVFFlat index
IVFFlat is traditionally considered inferior to HNSW for general ANNS tasks due to lower recall-throughput effi-
ciency. However, our results indicate specific edge cases—particularly under low selectivity filters—where IVFFlat
outperforms HNSW. This is likely due to the cluster-pruning nature of IVFFlat, which may adapt better to sparse
regions of the vector space than the graph-traversal mechanics of HNSW.
While HNSW maintains dominance in standard benchmarks, its superiority is not absolute in filtered search. We
observed that for each system, specific selectivity thresholds exist where the QPS-Recall curves of HNSW and IVFFlat
intersect (Figure 11).
Takeaway.HNSW is challenged in low-selectivity scenarios where the sparsity of valid neighbors necessitates
traversing distant nodes. This can trigger premature search termination due to lack of updates in the candidate set
(see Fig. 2). In such cases, the partition-based approach of IVFFlat offers a robust alternative, suggesting that: 1)
workload-aware index selection can be beneficial for query performance, and 2) hybrid indexing methods may be
necessary for optimal performance across the full selectivity spectrum. This may include switching between graph
and partition methods based on query predicates, or utilizing adaptable search parameters contingent on estimated
selectivity.
7.2.6 The Impact of Query-Filter Correlation on Performance
While filter selectivity (σ g) serves as a primary driver of performance, our analysis of theGlobal-Local Selectivity
(GLS) correlation reveals a more nuanced relationship between metadata and recall. As shown in Figure 12, retrieval
accuracy improves significantly as the GLS correlation shifts from low to high, while query latency (or QPS) remains
invariant across all three correlation scenarios. This relationship persists because a high GLS correlation implies that
valid neighbors are spatially clustered near the query, making them easier for index structures to locate. Conversely,
low or negative GLS correlation indicates that valid neighbors are ”pushed” to the periphery of the query’s natural
neighborhood, significantly challenging both graph-based and partition-based traversal.
Finally, Figure 10 demonstrates the sensitivity of our metric: while distance-based correlations only capture a
narrow range of interactions (approximately[−0.3,0.3]), GLS correlation spans nearly the full spectrum from−1to
1. This confirms that GLS is a more expressive tool for identifying ”hard” queries. By categorizing queries into low
21

0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (large)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (small)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (medium)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (large)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (small)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (medium)
0.6 0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Movies (large)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (small)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Recall@k=10101102103QPS
Reviews (large)(a) FAISS
(b) Milvus
(c) pgvector
HNSW g={movies: 0.001, reviews: 0.001}
HNSW g={movies: 0.02, reviews: 0.02}
HNSW g={movies: 0.11, reviews: 0.10}
HNSW g={movies: 1.00, reviews: 1.00}
IVF g={movies: 0.001, reviews: 0.001}
IVF g={movies: 0.02, reviews: 0.02}
IVF g={movies: 0.11, reviews: 0.10}
IVF g={movies: 1.00, reviews: 1.00}
Figure 11: QPS–Recall curves for HNSW and IVFFlat indexes at different selectivity levels.
22

0.0 0.2 0.4 0.6 0.8 1.0
Recall102103QPS
FAISS, Movies
0.0 0.2 0.4 0.6 0.8 1.0
Recall102QPS
Milvus, Movies
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102103QPS
pgvector, Movies
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102QPS
FAISS, Reviews
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102QPS
Milvus, Reviews
0.0 0.2 0.4 0.6 0.8 1.0
Recall100101102103QPS
pgvector, Reviews
HNSW, High GLS Correlation: q>0.3
HNSW, Average GLS Correlation: q[0.3,0.3]
HNSW, Low GLS Correlation: q<0.3
IVF, High GLS Correlation: q>0.3
IVF, Average GLS Correlation: q[0.3,0.3]
IVF, Low GLS Correlation: q<0.3
Figure 12: QPS–Recall curves for different levels of GLS correlation.
(ρq<−0.3), medium (−0.3≤ρ q≤0.3), and high (ρ q>0.3) GLS brackets, we show that lower GLS consistently
results in lower recall across all tested systems. This does not contradict our previous assertion that selectivity can
be studied as an isolated variable for system-level planning; rather, it suggests that whileσ gdetermines theaverage
system behavior, GLS correlation explains theper-queryvariance that individual users may experience.
7.2.7 Milvus: Hybrid Architecture and Adaptive Search
TheMilvusvector database operates on a specialized engine calledKnowhere, which diverges from standard FAISS
wrappers to ensure stability during filtered search.Knowhereemploys FAISS for graph construction but executes all
search operations through a heavily modified fork ofhnswlib[18]. Contrary to our initial hypothesis, the system’s high
recall is not primarily driven by data segmentation or ”Scatter-Gather” mechanics. Our ablation studies demonstrate
thatMilvusmaintains near perfect recall regardless of whether the data is partitioned into many 1GB segments or
stored in a single 16GB monolith (Figure 13). Instead, this robustness stems from a layered algorithmic strategy
within the search engine itself.
Dual-Pool Graph Traversal.The most critical differentiator is Knowhere’s handling of the candidate queue
during graph traversal. Standard FAISS HNSW implementations use a single priority queue where filtered (invalid)
nodes compete for slots with valid nodes. At high filter rates (e.g.,90%filtered out), the search beam becomes
saturated with invalid navigation nodes, causing recall to plummet. Knowhere addresses this by implementing aDual-
Poolstrategy (NeighborSetDoublePopList). It maintains two separate priority queues during traversal: one
for valid results and another for invalid nodes used strictly for navigation. This ensures that the search budget (ef)
is not cannibalized by filtered vectors, allowingMilvusto maintain near-perfect recall even at moderate selectivities
(e.g.,10%− −5%) where standard HNSW implementations degrade.
Adaptive Fallback Mechanisms.To handle highly restrictive filters,Milvusemploys a deterministic fallback
strategy rather than a cost-based optimizer. The engine checks filter selectivity prior to search; if the ratio of filtered
vectors exceeds a specific threshold (set to93%in the codebase), it bypasses the HNSW index entirely and performs
a brute-force scan (Pre-filtering + kNNS). Given the reduced search space, this operation remains computationally
inexpensive (1–5ms for typical segments) while guaranteeing100%recall. Furthermore, a ”safety net” fallback is
triggered post-search if the graph traversal fails to return the requestedkneighbors, ensuring consistency even in
sparse vector regions.
23

0.0 0.2 0.4 0.6 0.8 1.0
Recall102
6×1012×102QPS
HNSW , Movies, M=10
0.0 0.2 0.4 0.6 0.8 1.0
Recall102
4×1016×1012×102QPS
HNSW , Reviews, M=10
0.0 0.2 0.4 0.6 0.8 1.0
Recall102QPS
IVF-Flat, Movies, nlist=750
0.0 0.2 0.4 0.6 0.8 1.0
Recall101102QPS
IVF-Flat, Reviews, nlist=750
1 GB  g0.001
16 GB g0.001
1 GB  g0.02
16 GB g0.02
1 GB  g0.11
16 GB g0.11
1 GB  g1.00
16 GB g1.00
Figure 13: QPS–Recall curves for different segment sizes in Milvus.
This adaptive behavior extends toIVFFlatindexes, explaining the anomaly whereMilvusachieves 1.0 recall even
withnprobe= 1under restrictive filters. While standard inverted indices fail in this regime because the few matching
vectors are scattered across unvisited clusters, Knowhere detects the high selectivity and switches to a linear scan of
the valid vectors, effectively treating the query as a small-data problem. Consequently,Milvusexhibits ”Exact Search”
behavior for highly filtered queries regardless of the underlying index type.
Construction Efficiency.The robustness of index construction is driven by a batch-parallel build pipeline using a
persistent thread pool (Folly), rather than simple segment parallelization. Knowhere also performs a post-build graph
repair step to reconnect vectors isolated during concurrent insertion, which explains the higher base recall compared
to vanilla FAISS.
Attribute Index.Milvusperformance remains less sensitive to the presence of an attribute index (Figure 16). This
contrasts sharply withpgvector, which may change its query execution plan depending on configurations. Our analysis
confirms thatMilvusdoes not employ a cost-based Query Optimizer in the traditional relational sense, instead relying
on this static pre-filtering and per-segment fallback mechanism.
Takeaway.Milvus prioritizes recall stability through a sophisticated ”hybrid” engine design. Its performance
profile is a byproduct of its hybridized algorithmic specialization: theDual-Pooltraversal prevents search starvation
under moderate filters, while adaptive brute-force fallbacks guarantee accuracy under strict filters for both HNSW and
IVFFlat. This design allows the system to behave like an exact search engine, delivering perfect recall for queries with
high selectivity and hard-to-find neighbors.
7.2.8 PG-Vector: Query Optimization for Filtered ANNS
Falling back to kNN.The PostgreSQL Query Optimizer is designed to select the execution plan with the lowest esti-
mated total cost. To integrate vector similarity search,pgvectorintroduces specific cost functions for each supported
vector index type. These calculations rely heavily on cardinality estimates—standard in relational DBMS optimiz-
ers—to predict the number of tuples involved in the ANNS operation. The overall cost function is sensitive to the
parameterkand the estimated cardinality of the dataset following the application of the selection predicateS σ. We
observed two distinct query plans:
24

•ANNS + Post-filtering:In this plan, the vector search on index is executed first. For HNSW, this involves
populating a queue of sizeef search , while for IVFFlat, it involves probingn pclusters. Selection predicates are
subsequently applied to the tuples returned by the index scan. Below is an illustrative plan for a query with
r= 9.5andk= 20using the HNSW index:
Limit
-> Index Scan using vector_idx on dataset
Order By: (vector <=> $0)
Filter: (attribute > 8.5)
•Pre-Filtering + kNNS:Here, the selection predicates are applied first via a sequential scan, followed by an exact
k-NN search performed by calculating distances between the query vector and the filtered tuples. While this
fallback to brute-force search guarantees theoretically perfect recall (Exact kNN), it can be computationally
suboptimal for larger subsets. Below is the execution plan for the query withr= 9.5andk= 21:
Limit
-> Sort
Sort Key: ((dataset.vector <=> $0))"
Sort Method: top-N heapsort Memory: XYZ kB"
-> Seq Scan on dataset
Filter: (attribute > 8.5)"
We subsequently replicated the experiment with an active B-Tree index on the metadata attribute used for filtering.
In this configuration, the query optimizer is presented with an additional execution path:
•Attribute Index Scan + kNNS: This plan leverages the index onaverageratingto filter the dataset ef-
ficiently, after which it identifies thek-nearest neighbors by strictly calculating distances to the query vector.
Below is the execution plan for the query withr= 9.5andk= 10:
Limit
-> Sort
Sort Key: ((dataset.vector <=> $0))
Sort Method: top-N heapsort
-> Bitmap Heap Scan on dataset
Recheck Cond: (attribute > 8.5)
-> Bitmap Index Scan on idx_attribute
Index Cond: (attribute > 8.5::double precision)
As illustrated in Figure 17, theAttribute Index Scan + kNNSplan consistently yields perfect recall. However,
the automatic plan selection inpgvectorappears suboptimal; for many low-selectivity queries, the query optimizer
overlooks sequential scan based plans in favor of vector search on indexes. Our data indicates that there are cases,
when forcing theAttribute Index Scan + kNNS(orPre-Filtering + kNNS) plan would achieve perfect recall at latencies
comparable to the selected approximate plans.
Takeaway: Indexing metadata inpgvectorsignificantly alters query execution planning, yet exposes limitations
in the optimizer’s cost model. Specifically, the optimizer often prioritizes approximate vector index searches even in
scenarios where filtering followed by exact distance computation would offer guaranteed recall at a similar or lower
computational cost. This suggests that the cost weighting between high-dimensional vector operations and scalar index
scans requires further tuning to avoid recall-suboptimal plan selections.
7.3 Summary
Our extensive empirical evaluation of several open-source systems, includingFAISS,Milvus, andpgvector, yields
several critical insights into the interplay between filter selectivity, index topology, and system architecture:
25

The Selectivity-Efficiency Trade-off:We observed divergent behavior across index types. While graph-based
indexes exhibit degrading QPS-Recall curves at low selectivities regardless of filtering strategy, partition-based indexes
(IVFFlat) demonstrate significant efficiency gains in some cases. As such, in FAISS, the pre-filtering bitset allows the
search to bypass expensive distance computations for invalid vectors within probed clusters. Although absolute recall
at a fixedn probe may decrease due to a constrained candidate pool, the reduced computational overhead drastically
increases throughput (QPS). This results in a superior QPS-Recall trade-off curve of IVFFlat compared to higher
selectivity scenarios.
Index Inversion:We demonstrated that the traditional dominance of HNSW is challenged in low-selectivity
scenarios. In such cases, partition-based indexes (IVFFlat) can offer greater robustness due to their efficient cluster-
pruning mechanics.
Algorithmic Robustness and the Latency Floor:Our analysis ofMilvusreveals that recall stability is achieved
through algorithmic specialization acrossbothgraph and partition-based indexes. The system employs adaptive brute-
force fallbacks and ”Dual-Pool” traversal to achieve high recall for both HNSW and IVFFlat. However, this robustness
effectively enforces a ”latency floor,” sacrificing peak throughput to maintain exact-search accuracy under strict filters.
Optimizer Fragility:We exposed some limitations inpgvector’s cost-based optimizer, which frequently fails to
switch to sequential scans or attribute-index scans when they would provide perfect recall at competitive latencies.
This highlights the need for better cardinality and cost estimation models in relational vector extensions.
Query-Filter correlation: Our findings suggest that while metadata and vector distributions are weakly correlated
on average (mean GLS≈0), individual query performance is highly sensitive to local correlation shifts. Using our
GLS metric, we demonstrate that higher GLS correlation serves as a predictor for increased recall, as it indicates
a spatial clustering of valid neighbors. However, the high variance and near-zero mean across the dataset suggest
that while correlation explains per-query recall fluctuations, it remains too volatile for global cost estimation. Con-
sequently, query optimizers can reliably use global selectivity for plan selection, but must account for the fact that
low-correlation ”hard” queries may require more aggressive search parameters to maintain target recall levels.
7.4 Practical Guidelines for Filtered Vector Search
Based on our experimental findings, we distill the following actionable recommendations for practitioners deploying
filtered vector search systems:
1.Select index type based on expected filter selectivity.Do not assume HNSW universally dominates IVFFlat.
For workloads where expected selectivityσ g≲5%, consider IVFFlat; forσ g≳20%or predominantly unfiltered
queries, HNSW remains preferable. For mixed workloads, maintain both index types and select per query when
possible.
2.Plan for recall degradation under filtering.Filtered ANNS inherently degrades recall as selectivity decreases—
regardless of system or index. Treat recall loss as a budgeted resource. For critical queries, expose search-parameter
overrides or allow fallback to exact search when candidate exhaustion is detected.
3.Scale search parameters withk.Since the candidate pool size is fixed by search parameters (efSearch,
nprobe), increasingkwithout scaling these parameters systematically reduces recall. ScaleefSearch∝k
(ornprobe∝k) when recall matters.
4.Expect a throughput penalty for robustness.Engines that guarantee high recall under filtering (e.g.,Milvus)
prevent the fail-fast” behavior seen in standard libraries. While this ensures accuracy, it precludes the ultra-high
QPS observed in systems that allow recall to degrade (e.g.,FAISS,pgvector). Practitioners transitioning to robust
engines should budget for this latency floor, as the system will refuse to return fast, low-quality results.
5.Avoid over-tuning graph density.Increasing HNSW construction parameters (M,efConstruction) yields
diminishing returns under filtering, with marginal recall gains but significant build-time increases. Prefer lighter
graphs and allocate tuning budget to search-time parameters.
26

6.Index metadata attributes aggressively (pgvector).In relational vector extensions, metadata indexes unlock
execution plans that the optimizer can exploit for filtered queries. Without them, the system often commits to
recall-degrading post-filtering strategies.
7.Do not blindly trust query optimizers.Verify query plans manually (EXPLAIN ANALYZE) for critical work-
loads. The optimizer may favor approximate index scans even when exact scans would yield perfect recall at
comparable latency. For very selective filters, consider forcing attribute-index or sequential scan plans.
8.Use global selectivity for plan selection, but monitor correlation.Since metadata attributes tend to be weakly
correlated with the embedding space on average (mean GLS≈0), system throughput is dominated by global
selectivity (σ g). Base primary plan selection onσ gandk: lowσ gfavors exact or pre-filtered search, while highσ g
makes ANN-first plans safe. However, be aware that queries with low GLS correlation (where valid neighbors are
”pushed” away from the query) will suffer from lower recall. In recall-critical applications, consider using GLS as
a diagnostic tool to identify these ”hard” query regions and dynamically increase search depth (e.g.,ef search).
8 CONCLUSIONS AND FUTURE WORK
In this work, we addressed the widening gap between algorithmic theory and system-level implementation in FANNS.
First, we systematized the ambiguous landscape of FANNS by formalizing a taxonomy of Pre-, Post-, and Runtime-
filtering for both graph-based and partition-based indexes. Second, we established a rigorous evaluation framework
for modern Vector Databases handling predicate-based queries. To support this, we introduced MoReVec, a new
dataset comprising two relations:MoviesandReviews. These relations are connected via a foreign key and feature
transformer-based embeddings alongside structured metadata attributes. Finally, we proposed the Global-Local Se-
lectivity (GLS) metric to quantify the independence between metadata attributes and vector embeddings—a critical
distinction for isolating the impact of selectivity from semantic clustering.
Limitations.We acknowledge several factors that may affect the generalizability of our findings:
Internal Validity.Our conclusions are sensitive to the specific hyperparameter ranges explored (efSearch,
nprobe,M). Different parameter sweeps may yield different crossover points between index types. All experi-
ments are conducted in-memory with datasets fully loaded into RAM; results may differ under memory pressure or
disk-based execution. Our evaluation inherits the ANN-Benchmarks single-threaded execution model, which isolates
algorithmic behavior but does not capture production concurrency effects.
External Validity.Results may not generalize to: (1) other embedding models (e.g., CLIP, E5, OpenAI em-
beddings) that produce different vector distributions; (2) alternative distance metrics (inner product,L 2) on non-
normalized vectors; (3) workloads with highly correlated metadata, where filter attributes are semantically aligned
with the embedding space; and (4) distributed or multi-node deployments, where network latency and data partition-
ing introduce additional variables. Our conclusions apply primarily to generic, schema-agnostic filtering strategies
within single-node Vector Databases—not to fusion methods (e.g., ACORN, HQANN) or workload-aware indexes.
Future Work.The integration of vector search into relational systems remains a frontier for database research.
While current systems like pgvector support hybrid queries syntactically, their optimizers often lack the sophistica-
tion to maintain performance under varied workloads. We identify the development of a Selectivity-Aware Query
Optimizer for Filtered-ANNS as a primary direction. Such an optimizer must go beyond simple heuristics to perform:
•Optimal Index & Parameter Selection: Dynamically choosing between graph-based (e.g., HNSW) or partition-
based (e.g., IVF) indices, and tuning search parameters (e.g.,ef search ,nprobe ) at runtime based on specific filter
constraints and budget.
•Correlation-Aware Estimates: Investigating estimated GLS correlations to predict the ”reachability” of filter-
passing neighbors. This allows the engine to account for cases where attribute distributions are non-uniformly
clustered in the vector space, impacting the search effort required to identify valid candidates.
•Complex Relational Interops: Extending optimization logic to JOIN + Vector Search scenarios, when immediate
attribute filtering is infeasible. In such cases the optimizer must evaluate not only the cost-efficiency of compet-
ing execution plans, but also estimated Recall of competing execution plans: (1) Join-First, where the relational
27

join is executed to narrow the candidate pool prior to vector search; or (2) ANNS-First, which performs a vector
search to retrieve a speculatively oversized candidate set, followed by a late-stage join to resolve remaining
relational constraints.
Additionally, we intend to investigate hybrid indexing strategies that integrate multiple algorithmic approaches
(e.g., graph-structured partitions) into a single, unified design. In this framework, real-time selectivity estimates do
not trigger a switch between disparate indices but rather guide internal traversal logic and parameters. Such an index
is intended to remain robust under restrictive filter constraints and the distribution shifts typical of evolving relational
datasets.
Acknowledgments
This work was supported by NSF award number 2008815.
References
[1] C. C. Aggarwal, A. Hinneburg, and D. A. Keim. On the surprising behavior of distance metrics in high dimensional spaces.
InProceedings of the 8th International Conference on Database Theory, ICDT ’01, page 420–434, Berlin, Heidelberg, 2001.
Springer-Verlag.
[2] ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.https://ann-benchmark
s.com/. Accessed: 2026-01-20.
[3] ANN Benchmarks Extension for Filtered Search.https://github.com/aabylay/ANN-benchmark-HQ. Ac-
cessed: 2026-01-20.
[4] Evaluation of Approximate Nearest Neighbors: Large Datasets.https://corpus-texmex.irisa.fr/. Accessed:
2026-01-20.
[5] Annoy: Approximate Nearest Neighbors Oh Yeah.https://github.com/spotify/annoy. Accessed: 2026-01-20.
[6] M. Aum ¨uller, E. Bernhardsson, and A. Faithfull. Ann-benchmarks: A benchmarking tool for approximate nearest neighbor
algorithms.CoRR, arXiv:1807.05614, 2018.
[7] J. L. Bentley. Multidimensional binary search trees used for associative searching.Commun. ACM, 18(9):509–517, Sept.
1975.
[8] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agar-
wal, A. Herbert-V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen,
E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Lan-
guage models are few-shot learners. InProceedings of the 34th International Conference on Neural Information Processing
Systems, NIPS ’20, Red Hook, NY , USA, 2020. Curran Associates Inc.
[9] E. Ch ´avez, G. Navarro, R. Baeza-Yates, and J. L. Marroqu ´ın. Searching in metric spaces.ACM Comput. Surv., 33(3):273–321,
Sept. 2001.
[10] P. Covington, J. Adams, and E. Sargin. Deep neural networks for youtube recommendations. InProceedings of the 10th ACM
Conference on Recommender Systems, RecSys ’16, page 191–198, New York, NY , USA, 2016. Association for Computing
Machinery.
[11] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language
understanding.CoRR, arXiv:1810.04805, 2019.
[12] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e, M. Lomeli, L. Hosseini, and H. J ´egou. The faiss
library.IEEE Transactions on Big Data, pages 1–17, 2025.
[13] Elasticsearch: Free and Open Source, Distributed, RESTful Search Engine.https://github.com/elastic/elast
icsearch. Accessed: 2026-01-20.
28

[14] FAISS: A Library for Efficient Similarity Search.https://github.com/facebookresearch/faiss. Accessed:
2026-01-20.
[15] S. Gollapudi, N. Karia, V . Sivashankar, R. Krishnaswamy, N. Begwani, S. Raz, Y . Lin, Y . Zhang, N. Mahapatro, P. Srinivasan,
A. Singh, and H. V . Simhadri. Filtered-diskann: Graph algorithms for approximate nearest neighbor search with filters. In
Proceedings of the ACM Web Conference 2023, WWW ’23, page 3406–3416, New York, NY , USA, 2023. Association for
Computing Machinery.
[16] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, and S. Kumar. Accelerating large-scale inference with anisotropic
vector quantization. InProceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.
[17] Y . Han, C. Liu, and P. Wang. A comprehensive survey on vector database: Storage and retrieval technique, challenge.CoRR,
arXiv:2310.11703, 2023.
[18] Header-only C++/python library for fast approximate nearest neighbors.https://github.com/nmslib/hnswlib.
Accessed: 2026-02-09.
[19] P. Iff, P. Bruegger, M. Chrapek, M. Besta, and T. Hoefler. Benchmarking filtered approximate nearest neighbor search
algorithms on transformer-based embedding vectors.CoRR, arXiv:2507.21989, 2025.
[20] O. Jafari, P. Maurya, P. Nagarkar, K. M. Islam, and C. Crushev. A survey on locality sensitive hashing algorithms and their
applications.CoRR, arXiv:2102.08942, 2021.
[21] S. Jayaram Subramanya, F. Devvrit, H. V . Simhadri, R. Krishnawamy, and R. Kadekodi. Diskann: Fast accurate billion-
point nearest neighbor search on a single node. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch ´e-Buc, E. Fox, and
R. Garnett, editors,Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.
[22] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang, A. Madotto, and P. Fung. Survey of hallucination in natural
language generation.ACM Comput. Surv., 55(12), Mar. 2023.
[23] C. Jin, Y . Zhang, J. Liu, and J. Wang. Fast vector search in postgresql: A decoupled approach. InConference on Innovative
Data Systems Research (CIDR), 2026.
[24] Y . Jin, Y . Wu, W. Hu, B. M. Maggs, J. Yang, X. Zhang, and D. Zhuo. Curator: Efficient vector search with low-selectivity
filters.CoRR, arXiv:2601.01291, 2026.
[25] Z. Jing, Y . Su, Y . Han, B. Yuan, H. Xu, C. Liu, K. Chen, and M. Zhang. When large language models meet vector databases:
A survey.CoRR, arXiv:2402.01763, 2024.
[26] H. J ´egou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search.IEEE Transactions on Pattern Analysis
and Machine Intelligence, 33(1):117–128, 2011.
[27] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih. Dense passage retrieval for open-domain
question answering. In B. Webber, T. Cohn, Y . He, and Y . Liu, editors,Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online, Nov. 2020. Association for Computational
Linguistics.
[28] V . Leis, A. Gubichev, A. Mirchev, P. Boncz, A. Kemper, and T. Neumann. How good are query optimizers, really?Proc.
VLDB Endow., 9(3):204–215, Nov. 2015.
[29] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. K ¨uttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel,
and D. Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks.CoRR, arXiv:2005.11401, 2021.
[30] J. Li, H. Liu, C. Gui, J. Chen, Z. Ni, N. Wang, and Y . Chen. The design and implementation of a real time visual search
system on jd e-commerce platform. InProceedings of the 19th International Middleware Conference Industry, Middleware
’18, page 9–16, New York, NY , USA, 2018. Association for Computing Machinery.
[31] M. Li, X. Yan, B. Lu, Y . Zhang, J. Cheng, and C. Ma. Attribute filtering in approximate nearest neighbor search: An in-depth
experimental study.CoRR, arXiv:2508.16263, 2025.
[32] W. Li, Y . Zhang, H. Wang, K. Lu, and M. Kudo. Approximate nearest neighbor search on high dimensional data: Experiments,
analyses, and improvement.IEEE Transactions on Knowledge and Data Engineering, 32(8):1475–1488, 2020.
29

[33] Z. Li, X. Zhang, Y . Zhang, D. Long, P. Xie, and M. Zhang. Towards general text embeddings with multi-stage contrastive
learning.CoRR, arXiv:2308.03281, 2023.
[34] Y . Lin, K. Zhang, Z. He, Y . Jing, and X. S. Wang. Survey of filtered approximate nearest neighbor search over the vector-scalar
hybrid data.CoRR, arXiv:2505.06501, 2025.
[35] Y . A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable
small world graphs.CoRR, arXiv:1603.09320, 2018.
[36] M. D. Manohar, Z. Shen, G. E. Blelloch, L. Dhulipala, Y . Gu, H. V . Simhadri, and Y . Sun. Parlayann: Scalable and determin-
istic parallel graph-based approximate nearest neighbor search algorithms.CoRR, arXiv:2305.04359, 2024.
[37] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. Distributed representations of words and phrases and their
compositionality. InProceedings of the 27th International Conference on Neural Information Processing Systems - Volume
2, NIPS’13, page 3111–3119, Red Hook, NY , USA, 2013. Curran Associates Inc.
[38] Milvus: A Cloud-Native Vector Database.https://milvus.io/. Accessed: 2026-01-20.
[39] Massive Text Embedding Benchmark Leaderboard.https://huggingface.co/spaces/mteb/leaderboard.
Accessed: 2026-01-20.
[40] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers. MTEB: Massive text embedding benchmark. In A. Vlachos and
I. Augenstein, editors,Proceedings of the 17th Conference of the European Chapter of the Association for Computational
Linguistics, pages 2014–2037, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics.
[41] M. Muja and D. G. Lowe. Scalable nearest neighbor algorithms for high dimensional data.IEEE Transactions on Pattern
Analysis and Machine Intelligence, 36(11):2227–2240, 2014.
[42] OpenAI. Gpt-4 technical report.CoRR, arXiv:2303.08774, 2023.
[43] J. J. Pan, J. Wang, and G. Li. Survey of vector database management systems.The VLDB Journal, 33(5):1591–1615, July
2024.
[44] L. Patel, S. Jha, M. Pan, H. Gupta, P. Asawa, C. Guestrin, and M. Zaharia. Semantic operators: A declarative model for rich,
ai-based data processing.CoRR, arXiv:2407.11418, 2025.
[45] L. Patel, P. Kraft, C. Guestrin, and M. Zaharia. Acorn: Performant and predicate-agnostic search over vector embeddings and
structured data.CoRR, arXiv:2403.04871, 2024.
[46] pgvector: Open-source vector similarity search for Postgres.https://github.com/pgvector/pgvector. Ac-
cessed: 2026-01-20.
[47] Supercharging vector search performance and relevance with pgvector 0.8.0 on Amazon Aurora PostgreSQL.https:
//aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relev
ance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/. Accessed: 2026-01-20.
[48] Pinecone: Vector Database for Machine Learning.https://www.pinecone.io/. Accessed: 2026-01-20.
[49] Qdrant: Vector Similarity Search Engine and Vector Database.https://github.com/qdrant/qdrant. Accessed:
2026-01-20.
[50] S. E. Robertson and S. Walker. Some simple effective approximations to the 2-poisson model for probabilistic weighted re-
trieval. InProceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR ’94, page 232–241, Berlin, Heidelberg, 1994. Springer-Verlag.
[51] J. Shi, Y . Cai, and W. Zheng. Filtered approximate nearest neighbor search: A unified benchmark and systematic experimental
study [experiment, analysis & benchmark].CoRR, arXiv:2509.07789, 2025.
[52] H. V . Simhadri, M. Aum ¨uller, A. Ingber, M. Douze, G. Williams, M. D. Manohar, D. Baranchuk, E. Liberty, F. Liu, B. Lan-
drum, M. Karjikar, L. Dhulipala, M. Chen, Y . Chen, R. Ma, K. Zhang, Y . Cai, J. Shi, Y . Chen, W. Zheng, Z. Wan, J. Yin, and
B. Huang. Results of the big ann: Neurips’23 competition.CoRR, arXiv:2409.17424, 2024.
30

[53] Sivic and Zisserman. Video google: a text retrieval approach to object matching in videos. InProceedings Ninth IEEE
International Conference on Computer Vision, pages 1470–1477 vol.2, 2003.
[54] N. Thakur, N. Reimers, A. R ¨uckl´e, A. Srivastava, and I. Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. In J. Vanschoren and S. Yeung, editors,Proceedings of the Neural Information Processing
Systems Track on Datasets and Benchmarks, volume 1, 2021.
[55] Vector Database Benchmarks.https://zilliz.com/vdbbench-leaderboard. Accessed: 2026-01-20.
[56] Vespa: The Open Big Data Serving Engine.https://vespa.ai/. Accessed: 2026-01-20.
[57] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu, K. Yu, Y . Yuan, Y . Zou, J. Long, Y . Cai,
Z. Li, Z. Zhang, Y . Mo, J. Gu, R. Jiang, Y . Wei, and C. Xie. Milvus: A purpose-built vector data management system. In
Proceedings of the 2021 International Conference on Management of Data, SIGMOD ’21, page 2614–2627, New York, NY ,
USA, 2021. Association for Computing Machinery.
[58] M. Wang, L. Lv, X. Xu, Y . Wang, Q. Yue, and J. Ni. Navigable proximity graph-driven native hybrid queries with structured
and unstructured constraints.CoRR, arXiv:2203.13601, 2022.
[59] Weaviate: Open-source Vector Database.https://github.com/weaviate. Accessed: 2026-01-20.
[60] C. Wei, B. Wu, S. Wang, R. Lou, C. Zhan, F. Li, and Y . Cai. Analyticdb-v: a hybrid analytical engine towards query fusion
for structured and unstructured data.Proc. VLDB Endow., 13(12):3152–3165, Aug. 2020.
[61] W. Wu, J. He, Y . Qiao, G. Fu, L. Liu, and J. Yu. Hqann: Efficient and robust similarity search for hybrid queries with
structured and unstructured constraints.CoRR, arXiv:2207.07940, 2022.
[62] W. Xia, M. Yang, W. Li, and W. Wang. Filtered approximate nearest neighbor search cost estimation.CoRR,
arXiv:2602.06721, 2026.
[63] W. Yang, T. Li, G. Fang, and H. Wei. Pase: Postgresql ultra-high-dimensional approximate nearest neighbor search exten-
sion. InProceedings of the 2020 ACM SIGMOD International Conference on Management of Data, SIGMOD ’20, page
2241–2253, New York, NY , USA, 2020. Association for Computing Machinery.
[64] Q. Zhang, S. Xu, Q. Chen, G. Sui, J. Xie, Z. Cai, Y . Chen, Y . He, Y . Yang, F. Yang, M. Yang, and L. Zhou. VBASE: Unifying
online vector similarity search and relational queries via relaxed monotonicity. In17th USENIX Symposium on Operating
Systems Design and Implementation (OSDI 23), pages 377–395, Boston, MA, July 2023. USENIX Association.
[65] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou. Qwen3
embedding: Advancing text embedding and reranking through foundation models.CoRR, arXiv:2506.05176, 2025.
[66] Y . Zhang, S. Liu, and J. Wang. Are there fundamental limitations in supporting vector data management in relational
databases? a case study of postgresql. In2024 IEEE 40th International Conference on Data Engineering (ICDE), pages
3640–3653, 2024.
31

A ADDITIONAL EXPERIMENTS
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Movies, (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity0.20.61.0Recall
Reviews, (large)(a) HNSW
(b) IVFFlat
FAISS k=1
FAISS k=10
FAISS k=40
FAISS k=100Milvus k=1
Milvus k=10
Milvus k=40
Milvus k=100pgvector k=1
pgvector k=10
pgvector k=40
pgvector k=100
Figure 14: Recall–Selectivity curves for different values ofk.
32

0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, @m=10, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, @m=10, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, @m=10, (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, @m=10, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, @m=10, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, @m=10, (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Movies, (large)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, (small)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, (medium)
0.0 0.2 0.4 0.6 0.8 1.0
Selectivity101102103QPS
Reviews, (large)(a) HNSW
(b) IVFFlat
FAISS k=1
FAISS k=10
FAISS k=40
FAISS k=100Milvus k=1
Milvus k=10
Milvus k=40
Milvus k=100pgvector k=1
pgvector k=10
pgvector k=40
pgvector k=100Figure 15: QPS–Selectivity curves for different values ofk.
33

0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Movies (small)
0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Movies (medium)
0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Movies (large)
0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Reviews (small)
0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Reviews (medium)
0.80 0.85 0.90 0.95 1.00
Recall@k=1025262728QPS
Reviews (large)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (small)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (medium)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Movies (large)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (small)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (medium)
0.7 0.8 0.9 1.0
Recall@k=10101102QPS
Reviews (large)(a) HNSW
(b) IVFFlat
att_idx=0, g={movies: 0.001}
att_idx=0, g={movies: 0.06}
att_idx=0, g={movies: 1.00}
att_idx=1, g={movies: 0.001}
att_idx=1, g={movies: 0.06}
att_idx=1, g={movies: 1.00}
Figure 16: QPS–Recall curves for HNSW and IVFFlat inMilvuswith and without index on the filter attribute.
34

0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Movies (small)
0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Movies (medium)
0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Movies (large)
0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Reviews (small)
0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Reviews (medium)
0.00 0.25 0.50 0.75 1.00
Recall@k=10101102103QPS
Reviews (large)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Movies (small)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Movies (medium)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Movies (large)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Reviews (small)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Reviews (medium)
0.00 0.25 0.50 0.75 1.00
Recall@k=10100101102103QPS
Reviews (large)(a) HNSW
(b) IVFFlat
att_idx=0, g={movies: 0.001}
att_idx=0, g={movies: 0.06}
att_idx=0, g={movies: 1.00}
att_idx=1, g={movies: 0.001}
att_idx=1, g={movies: 0.06}
att_idx=1, g={movies: 1.00}
Figure 17: QPS–Recall curves for HNSW and IVFFlat inpgvectorwith and without index on the filter attribute.
35