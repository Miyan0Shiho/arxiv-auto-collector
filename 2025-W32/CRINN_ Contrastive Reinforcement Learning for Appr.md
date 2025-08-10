# CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Chris Shum, Jiwei Li

**Published**: 2025-08-04 05:57:46

**PDF URL**: [http://arxiv.org/pdf/2508.02091v1](http://arxiv.org/pdf/2508.02091v1)

## Abstract
Approximate nearest-neighbor search (ANNS) algorithms have become
increasingly critical for recent AI applications, particularly in
retrieval-augmented generation (RAG) and agent-based LLM applications. In this
paper, we present CRINN, a new paradigm for ANNS algorithms. CRINN treats ANNS
optimization as a reinforcement learning problem where execution speed serves
as the reward signal. This approach enables the automatic generation of
progressively faster ANNS implementations while maintaining accuracy
constraints. Our experimental evaluation demonstrates CRINN's effectiveness
across six widely-used NNS benchmark datasets. When compared against
state-of-the-art open-source ANNS algorithms, CRINN achieves best performance
on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and
GloVe-25-angular), and tied for first place on two of them (SIFT-128-Euclidean
and GloVe-25-angular). The implications of CRINN's success reach well beyond
ANNS optimization: It validates that LLMs augmented with reinforcement learning
can function as an effective tool for automating sophisticated algorithmic
optimizations that demand specialized knowledge and labor-intensive manual
refinement.Code can be found at https://github.com/deepreinforce-ai/CRINN

## Full Text


<!-- PDF content starts -->

CRINN: Contrastive Reinforcement Learning for Approximate
Nearest Neighbor Search
Xiaoya Li♣,♠, Xiaofei Sun♠, Albert Wang♠, Chris Shum♠and Jiwei Li♠
♣University of Washington,♠DeepReinforce Team
github.com/deepreinforce-ai/crinn
Abstract
Approximate nearest-neighbor search (ANNS) algorithms have become increasingly critical for recent AI applications,
particularly in retrieval-augmented generation (RAG) and agent-based LLM applications. In this paper, we present CRINN,
a new paradigm for ANNS algorithms. CRINN treats ANNS optimization as a reinforcement learning problem where
execution speed serves as the reward signal. This approach enables the automatic generation of progressively faster ANNS
implementations while maintaining accuracy constraints. Our experimental evaluation demonstrates CRINN’s effectiveness
across six widely-used NNS benchmark datasets. When compared against state-of-the-art open-source ANNS algorithms,
CRINN achieves best performance on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular), and
tied for first place on two of them (SIFT-128-Euclidean and GloVe-25-angular). The implications of CRINN’s success reach
well beyond ANNS optimization: It validates that LLMs augmented with reinforcement learning can function as an effective
tool for automating sophisticated algorithmic optimizations that demand specialized knowledge and labor-intensive manual
refinement.B
0.5 0.9 0.99 0.999 0.9999 1103104
(a) sift-128-euclidean
0.1 0.5 0.9 0.99 0.999102103104
(b) gist-960-euclidean
0.9 0.99 0.999 0.9999 1103104
(c) mnist-784-euclidean
0.1 0.5 0.9 0.99 0.999 0.9999 1102103104
(d) glove-100-angular
0.1 0.5 0.9 0.99 0.999 0.9999 1103104105
(e) glove-25-angular
0.1 0.5 0.9 0.99 0.999102103104105
(f) nytimes-256-angular
RecallQueries per second (1/s)GLASS NN-Descent ParlayANN PyNNDescent Vearch Voyager CRINN
Figure 1: QPS versus recall curves for different models across six datasets. CRINN achieves achieves best-in-class performance
on three out of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular) and matching state-of-the-art results
on two (SIFT-128 and GloVe-25).
BEmail: xxiaoya@uw.edu, {xiaofei_sun, albert_wang, chris_shum, jiwei_li}@deep-reinforce.com
1arXiv:2508.02091v1  [cs.LG]  4 Aug 2025

1 Introduction
Approximate nearest-neighbor search (ANNS) [ 20,21,6,7,19,12,5] aims at finding data points that are closest to a query
point in high-dimensional spaces while trading off a small amount of accuracy for significant speedup over exact search methods.
ANNS is of growing importance due to the unprecedented popularity of retrieval-augmented generation (RAG) [ 16,10,23]
and agent-based LLM applications [ 25], which require fast retrieval of relevant information from massive vector databases.
Existing widely-used open-source projects use fundamental algorithms such as Vamana [ 12] and HNSW [ 20] as the backbone:
DiskANN and ParlayANN [ 21] build upon Vamana, while FAISS [ 6], Vespa1, Weaviate2, Qdrant3, Milvus4, and GLASS [ 29]5
leverage HNSW, proposing different levels of optimization to cater for various scenarios.
Existing optimizations for ANNS are mostly manual, where humans identify bottlenecks through profiling tools, analyze cache
miss patterns, hand-tune parameters like graph degree and search beam width, carefully design data structures for memory
locality, and iteratively experiment with different algorithmic variants based on hardware characteristics. This process requires
deep expertise in computer architecture, parallel programming, and the mathematical properties of ANNS algorithms. With
the increasing power of LLMs in code generation [ 9,11], a natural question arises: can LLMs facilitate optimization by
automatically generating and testing algorithmic improvements, learning from the execution speeds of previous implementations
to propose better solutions, and discovering novel optimization patterns that human engineers might overlook?
In this paper, we propose CRINN, a reinforcement learning-augmented LLM framework for automated ANNS optimization.
The core of CRINN is a contrastive RL model that performs comparative analysis of previously generated code variants
alongside their execution metrics, enabling the model to learn by distinguishing between effective and ineffective optimization
strategies and generate better solutions. Through this contrastive learning approach, CRINN develops an understanding of
which code patterns lead to performance improvements and which cause degradation. The generated code variants are evaluated
based on execution time, with faster implementations receiving higher rewards in the RL training process. This reward signal
drives the LLM to iteratively generate progressively more efficient ANNS implementations. By learning from the performance
outcomes of its own generated code, CRINN effectively transforms the manual optimization process into an automated search
through the space of possible implementations.
Our experimental evaluation demonstrates CRINN’s effectiveness across six widely-used NNS benchmark datasets [ 1]. When
compared against state-of-the-art open-source ANNS algorithms, CRINN achieves best performance on three of them (GIST-
960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular); tied for first place on two of them (SIFT-128-Euclidean and
GloVe-25-angular). More importantly, the success of CRINN carries broader implications beyond ANNS optimization. It
demonstrates that RL-augmented LLMs can serve as powerful tools for automating complex algorithmic optimizations that
traditionally require deep domain expertise and extensive manual tuning. As the demand for efficient vector search continues to
grow with the proliferation of RAG and agent-based LLM applications, automated optimization frameworks like CRINN will
become increasingly valuable for maintaining competitive performance across evolving hardware architectures and application
requirements.
2 Background: HNSW
We use HNSW (Hierarchical Navigable Small World) [ 20] as the optimization backbone. HNSW is a state-of-the-art graph-
based ANNS algorithm that achieves high recall rates with logarithmic search complexity. Here we describe the core modules
that most HNSW-based implementations adopt, on which our RL optimization is performed.
2.1 Graph Construction
In HNSW, we first need to build a graph for base vectors, where each vector is represented as a node in the graph. As the search
is performed through this graph for a given query, graph construction is a key step in HNSW.
The graph construction module builds the graph through incremental insertion of vectors. HNSW constructs a multi-layer graph
where each element is assigned to multiple layers based on an exponentially decaying probability distribution. Each new base
vector is assigned to layer 0 with probability 1, and to each subsequent layer lwith probability pl, where pis typically set to
1/2 ln(2) . This creates a hierarchical structure similar to skip lists, enabling efficient navigation during search.
1https://docs.vespa.ai/en/nearest-neighbor-search.html
2https://weaviate.io/blog/vector-search-explained
3https://github.com/qdrant/qdrant
4https://github.com/milvus-io/milvus
5https://github.com/hhy3/pyglass
2

For each layer where a vector is present, the algorithm performs a greedy search from the layer’s entry point to find the M
nearest neighbors. The search parameter efcontrols the number of candidates explored during this neighbor selection process.
The algorithm maintains Mconnections for upper layers and M×2connections for the bottom layer (layer 0) to ensure rich
connectivity. It is worth noting that efis a crucial parameter in HNSW, controlling the tradeoff between recall and speed.
Larger values of efmean exploring more candidates during the search phase, thus achieving higher recall at the cost of reduced
speed. HNSW employs a heuristic pruning strategy to optimize the graph structure. This strategy maintains connectivity while
promoting the small-world property by prioritizing diverse connections over purely nearest neighbors. Redundant edges that
don’t contribute to search efficiency are removed, resulting in a graph that balances accuracy with traversal efficiency.
2.2 Search
Given a constructed graph and an input query, the search module performs k-nearest neighbor search through a two-phase
hierarchical process. The search begins with the upper layer, starting from the global entry point at the top layer. The algorithm
greedily traverses each layer to find the single nearest neighbor, using this neighbor as the entry point for the next lower layer.
This process continues until reaching the bottom layer (layer 0), effectively narrowing down the search space.
At layer 0, the algorithm switches to a more exhaustive exploration strategy. It initializes a candidate set with the entry point
from layer 1 and maintains a priority queue of efnearest candidates (where ef≥k). The search iteratively explores neighbors
of the closest unexplored candidate, updating the result set whenever closer neighbors are discovered. The process terminates
when no remaining candidate can improve the current results, ensuring that the knearest neighbors have been found with high
probability.
2.3 Refinement
Most HNSW-based algorithms incorporate refinement modules that enhance search efficiency through various optimization
strategies. One common approach is quantized preliminary search, where vectors are compressed to int4 or int8 representations
for rapid similarity estimation. Product quantization further improves this by dividing vectors into subspaces and quantizing
each independently. Asymmetric distance computation keeps queries in full precision while using quantized database vectors,
balancing speed and accuracy. Hierarchical pruning strategies form another class of refinements. These include layer-wise
filtering that uses coarse distance estimates to prune entire graph regions early in the search process. Batch processing amortizes
memory access costs by handling multiple queries simultaneously, while adaptive beam search dynamically adjusts the search
width based on query difficulty.
3 Improving ANNS with Contrastive Reinforcement Learning
In this section, we describe how we optimize an ANNS algorithm using contrastive RL [18] in detail.
3.1 Overview
An ANSS algorithm usually contains multiple modules, e.g., graph construction, search, refinement as described in Section 2.
We treat each module in ANSS as independent and optimize module by module sequentially using contrastive RL.
The core idea of contrastive RL is to transform the traditional reinforcement learning paradigm by integrating both exemplar
code snippets and their execution times directly into the LLM prompt. This enables the LLM to explicitly reason about why
certain ANNS implementations outperform others, learning to identify performance-critical patterns through direct comparison.
Given these speed-annotated examples, the LLM conducts comparative analysis to understand the factors driving efficiency
differences, then synthesizes new module code that incorporate these insights. The generated code is evaluated based on
execution time, which serves as the reward signal for updating the LLM parameters within the RL framework. This creates a
feedback loop where the model continuously improves its ability to both analyze performance characteristics and generate
optimized implementations.
3.2 Prompt Construction
Here we describe the construction of prompts provided to the LLM during contrastive-RL training. The prompt comprises the
following structured components:
1.Task Description: A detailed description of the current ANNS module to optimize, including input/output requirements,
performance metrics, and optimization objectives with emphasis on execution speed.
3

Prompt Template Used in CRINN
Task Description
You are an approximate nearest neighbor search optimization expert specializing in high-performance similarity
search algorithms. Given reference implementations for search, your objective is to create an accelerated version
that maintains identical functionality. You will receive previous module implementations accompanied by their
scores indicating the general speed. Higher scores indicate higher speed. Conduct a comparative analysis of these
implementations and use the insights to develop optimized graph construction code.
Previous Implementations with Speed
1// Implementation 1 (Score: 1.42)
2class Module_v1 {
3 void build_index(const float *data, int n, int d) {
4 ...
5 }
6 void search(const float *query, int k, int *indices, float *distances) {
7 ...
8 }
9};
10// Implementation 2 (Score: 1.34)
11class Module_v2 {
12 void build_index(const float *data, int n, int d) {
13 ...
14 }
15 void search(const float *query, int k, int *indices, float *distances) {
16 ...
17 }
18};
Generation Protocol
You MUST use exactly two hash symbols (##) at the beginning of each section.
## Performance Analysis : Compare ANNS implementations above and articulate on:
1.Which implementations achieve superior query throughput and what algorithmic factors contribute to their
fast execution?
2. What indexing structures or search strategies demonstrate the best speed-accuracy tradeoffs?
3. What are the primary bottlenecks limiting query performance in slower implementations?
4. Which vectorization, parallelization, or caching techniques remain unexploited?
## Algorithm Design : Describe your optimization strategy as numbered points outlining key techniques
and improvements for accelerating execution speed
## Code : Your code implementation
Requirements and Constraints
## Critical Requirements :
1.Search quality must match the reference implementation exactly (same recall, precision). Failure to maintain
search accuracy will result in a score of 0.
2.The module must support the same interface: build_index() and search() methods with identical parameters.
3. Results must be deterministic and reproducible across runs.
Table 1: Prompt template used in CRINN for RL training.
4

2.Previous Implementations with Speed: Previously generated ANNS implementations paired with their scores indicating
execution speed.
3.Generation Protocol: Explicit instructions defining the required output format.
4.Critical Requirements: Explicit instructions for the required output content.
The model’s response must contain three structured components:
1.Performance Analysis: A comparative analysis identifying which previous ANNS module implementations achieved
superior speed performance and the underlying algorithmic or implementation factors contributing to their fast execution.
2.Algorithm Design: A high-level description of the proposed optimization strategy, outlining key techniques and
improvements for accelerating execution speed as numbered points in natural language.
3.Code Implementation: The detailed code implementation.
A detailed example of the prompt structure is shown in Table 1. For contrastive exemplar selection, we adopted the similar
strategy to [ 18,26], where we maintain a performance-indexed database of all successful code samples. We sample exemplars
from the dataset using a temperature-scaled softmax distribution:
P(Bi) =exp (( si−µ)/τ)P
jexp (( sj−µ)/τ)(1)
where sidenotes the score of code sample in the database, µdenotes the mean score across all codes in the database, and τis
the temperature parameter governing the exploration-exploitation tradeoff.
3.3 Speed Reward
Giving a generated code a proper reward, which must be a scalar , is crucial in training our system. The reward serves two
important purposes: (1) guiding parameter updates in reinforcement learning and (2) constructing prompts for contrastive
analysis.
Unfortunately, constructing a reward that effectively captures general code speed performance in ANNS is not as straightforward
as it seems. In the ANNS setting, performance is characterized by two interdependent metrics: queries per second (QPS) and
recall. The only parameter we can adjust is ef, which controls the number of neighbors explored during search: higher values
ofeflead to lower QPS and higher recall. However, this creates a comparison problem: two implementations with identical ef
can exhibit different combinations of QPS and recall. Simply using QPS as the reward would be unfair without accounting for
recall differences.
This challenge explains why the ANNS literature rarely reports single quantitative metrics (unlike traditional ML metrics such
as accuracy or F1 score). Instead, researchers typically present QPS-recall curves (as in Figure 1) that visualize the entire
performance tradeoff space. A seemingly straightforward solution would be to fix one metric and optimize the other—either
fix recall and maximize QPS, or fix QPS and maximize recall. Unfortunately, this approach is infeasible because eftakes
discrete values. We cannot continuously tune efto achieve specific target values of QPS or recall; instead, we can only obtain a
discrete set of (QPS, recall) points corresponding to different efsettings. This discretization prevents us from making fair
comparisons at fixed performance levels. This discrete nature creates other evaluation challenges: some algorithms may excel
in low-recall regions but perform poorly in high-recall regions, while others show the opposite pattern; some algorithms may
not even produce data points in certain regions of the spectrum. For example, certain high-quality algorithms often cannot
achieve low recall values regardless of efvalues. This incomplete coverage of the performance spectrum adds significant
complexity to reward estimation, where the RL framework requires the reward to be a single scalar value.
To address these challenges, we employ the following evaluation strategy: given a module implementation, we first vary ef
across different values to obtain a comprehensive set of (QPS, recall) points. We then filter these points to retain only those
within the recall range of [0.85, 0.95] and compute the area under the curve formed by these points. This area serves as our
scalar reward.
We choose the [0.85, 0.95] recall range for several reasons. First, we primarily care about algorithms that achieve reasonable
recall levels—performance at very low recall is generally not useful for practical applications. Second, in the high-recall region
above 0.95, data points become increasingly sparse, and some algorithms may not even produce points in this range, leading to
unstable reward estimation. The [0.85, 0.95] range thus represents a sweet spot where most algorithms have sufficient data
points and where performance differences are most meaningful for real-world deployment.
5

3.4 RL Training
For reinforcement learning training, we employ the Group Relative Policy Optimization (GRPO) approach [ 27]. For each input
prompt qaugmented with selected demonstrations, we generate Gcode completions from the current policy πold, represented as
{d1, d2, . . . , d G}. The corresponding reward scores are denoted by r= (r1, r2, . . . , r G). To ensure training stability, rewards
undergo smoothing following [18]. Following the GRPO methodology, we normalize rewards within each group:
ˆri=ri−mean (r)
std(r)(2)
The GRPO training objective maximizes the following loss function:
LGRPO(θ) =Eq∼P(q),{di}G
i=1∼πθold(d|q)
1
GGX
i=11
|di||di|X
t=1
minπθ(di,t|q, di,<t)
πθold(di,t|q, di,<t)ˆri,
clipπθ(di,t|q, di,<t)
πθold(di,t|q, di,<t),1−ε,1 +ε
ˆri
−βDKL[πθ∥πref]
(3)
Here:
•πθrepresents the policy network under optimization
•πθolddenotes the policy from the preceding training step
•εcontrols the clipping range for policy updates
•βis a regularization parameter balancing exploration and adherence to the reference policy
•DKLmeasures the Kullback-Leibler divergence between current and reference distributions
For comprehensive details on GRPO, we direct readers to [ 27]. The model’s parameters are updated through this GRPO
objective, utilizing contrastive prompts enriched with comparative exemplars.RetryClaude can make mistakes. Please double-
check responses.
3.5 Using Glass as the RL Starting Point
Instead of implementing an ANSS algorithm from scratch, we use an existing open-source ANSS algorithm as a starting
point and progressively optimize its constituent modules. We selected GLASS as our initial baseline due to its balanced
performance across diverse evaluation datasets. GLASS is a recent graph-based ANNS algorithm that achieves efficient graph
construction while maintaining competitive search performance. It is important to note that CRINN is a general optimization
framework—while we demonstrate its effectiveness using GLASS, it can take any existing open-source ANNS algorithm as a
starting point and progressively evolve its implementation for improved performance. The choice of GLASS simply provides a
strong foundation for showcasing CRINN’s capabilities across different optimization scenarios. We will update the results
based on another strong baseline ParlayANN in the upcoming version of this project. Using GLASS, we sequentially optimize
the code across the three key modules in HNSW: graph construction ,search , and refinement .
4 Experiments and Analysis
4.1 Datasets and Baselines
We use the following six widely used datasets for evaluation: SIFT-128-Euclidean, GIST-960-Euclidean, MNIST-784-Euclidean,
GloVe-100-Angular, GloVe-25-Angular, and NYTimes-256-Angular. SIFT-128-Euclidean consists of 128-dimensional SIFT
features extracted from images. GIST-960-Euclidean contains 960-dimensional GIST global descriptors for images. MNIST-
784-Euclidean is composed of 784-dimensional vectors representing flattened 28 ×28 pixel images of handwritten digits.
GloVe-100-Angular and GloVe-25-Angular are datasets of word embeddings trained by the GloVe algorithm on a large text
corpus, with dimensions 100 and 25, respectively. NYTimes-256-Angular contains 256-dimensional bag-of-words vectors from
New York Times articles. The statistics for these datasets are summarized in Table 2.
To maintain a single generalizable codebase, we train our reinforcement learning model exclusively based on rewards from the
SIFT-128 dataset. The resulting implementation is then evaluated across all datasets without modification. It is worth noting
that SIFT-128 uses Euclidean distance, which means our RL model is trained only on Euclidean rewards. This might cause
problems for angular-distance datasets. However, as we will show in the experiments, the code trained on Euclidean distance
6

Dataset D LID Base Vectors Query Vectors
Euclidean Distance
SIFT-128 128 9.3 1,000,000 10,000
GIST-960 960 20.5 1,000,000 1,000
MNIST-784 784 14.1 60,000 10,000
Angular Distance
GloVe-25 25 9.9 1,183,514 10,000
GloVe-100 100 12.3 1,183,514 10,000
NYTimes-256 256 12.5 290,000 10,000
Table 2: Statistics of the benchmark datasets. D is the vector dimension, and LID is the average Local Intrinsic Dimension.
generalizes well to angular-distance datasets, demonstrating the strong generalization capability of CRINN. Incorporating both
Euclidean and angular distances as training rewards constitutes future work.
We employ the following widely used open-source projects as baselines:
•Glass [29]: A graph-based approximate nearest neighbor search library developed by Zilliz, utilizing hierarchical
navigable small world (HNSW) graphs with optimizations for high-dimensional vector search and hardware acceleration.
Glass serves as the starting point for our RL training process, making it a natural baseline for performance comparison.
•ParlayANN [22]: A parallel approximate nearest neighbor library that leverages multi-core parallelism and cache-efficient
algorithms, offering implementations of graph-based search methods optimized for shared-memory multiprocessors. In
the updated version of this project, we will include the experimental setup that uses ParlayANN as the reinforcement
learning starting point.
•NNDescent [24]: An implementation of the NN-Descent algorithm that constructs approximate k-nearest neighbor
graphs through iterative local search, efficiently handling both dense and sparse data with minimal memory overhead.
•PyNNDescent6: A Python implementation of NN-Descent that provides fast approximate nearest neighbor queries and
KNN graph construction, with support for a wide variety of distance metrics and efficient handling of high-dimensional
data.
•Vearch [17]7: A distributed vector search system designed for large-scale similarity search, combining graph-based
indexing with inverted file structures to support billion-scale vector retrieval in production environments.
•Voyager8: A library developed by Spotify for approximate nearest neighbor search, implementing hierarchical navigable
small world graphs with optimizations for music and recommendation system workloads.
5 Results
5.1 Major Results
Figure 1 presents the QPS versus recall curves for different models across six datasets. CRINN outperforms all baselines on three
benchmarks: GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular. The improvement is particularly substantial
for MNIST-784-Euclidean. On SIFT-128-Euclidean and GloVe-25-angular, CRINN achieves performance comparable to the
best baseline—matching ParlayANN on SIFT-128 and Vearch on GloVe-25. Among the six datasets, CRINN underperforms
the best baseline only on NYTimes-256. This performance gap likely stems from the fundamental differences between distance
metrics. Since the RL-optimized code was trained exclusively on SIFT-128 using Euclidean distance, it may not effectively
capture the optimization patterns required for angular similarity computations. Notably, it is common for strong models
to exhibit dataset-specific weaknesses. For instance, ParlayANN performs poorly on GloVe-25-angular despite its strong
performance elsewhere. When comparing CRINN with GLASS, which serves as the RL starting point, CRINN demonstrates
substantial improvements. This consistent performance gain indicates that contrastive RL can robustly and progressively
optimize the code.
6https://github.com/lmcinnes/pynndescent
7https://github.com/vearch/vearch
8https://github.com/spotify/voyager
7

Dataset Recall CRINN Best Baseline Baseline Improvement
QPS QPS
Euclidean Distance
SIFT-128 0.900 36,876 ParlayANN 29,368 +25.57%
0.950 27,499 ParlayANN 23,057 +19.26%
0.990 13,014 ParlayANN 11,808 +10.21%
0.999 5,158 ParlayANN 4,996 +3.25%
GIST-960 0.900 4,288 ParlayANN 3,788 +13.20%
0.950 2,925 ParlayANN 2,348 +24.59%
0.990 1,149 ParlayANN 666 +72.68%
MNIST-784 0.900 24,826 ParlayANN 19,324 +28.47%
0.950 22,008 ParlayANN 17,293 +27.26%
0.990 17,457 ParlayANN 11,728 +48.85%
0.999 10,600 ParlayANN 5,722 +85.25%
Angular Distance
GloVe-100 0.900 5,947 Vearch 5,768 +3.09%
0.950 3,024 ParlayANN 3,212 -5.84%
GloVe-25 0.900 37,474 Glass 31,611 +18.55%
0.950 28,909 Glass 21,899 +32.01%
0.990 13,574 Glass 11,804 +14.99%
0.999 4,588 Glass 4,549 +0.87%
NYTimes-256 0.900 1,623 ParlayANN 9,459 -82.85%
Table 3: Performance comparison of CRINN against best baselines across different datasets and recall levels. QPS (Queries Per
Second) measures throughput.
5.2 QPS with Fixed Recall
To give a quantitative comparison, Table 3 presents the QPS (Queries Per Second) performance of CRINN against the best-
performing baselines across six benchmark datasets at various recall levels (0.9, 0.99, 0.999). For cases where performances are
absent for specific recall levels, it indicates that none of the tested methods could reach the target recall threshold. The results
demonstrate that CRINN consistently outperforms state-of-the-art methods across most configurations, with improvements
ranging from modest gains of 3.09% to substantial speedups of 85.25%. CRINN shows particularly strong performance on
MNIST-784, achieving up to 85.25% improvement at 0.999 recall, and on GIST-960 at high recall levels, with a 72.68%
improvement at 0.99 recall. The SIFT-128 dataset, which was used for training the RL agent, shows consistent improvements
across all recall levels, with gains decreasing as recall requirements become more stringent. Among angular distance datasets,
GloVe-25 exhibits significant improvements of up to 32.01%, while GloVe-100 shows mixed results, including a slight
degradation of 5.84% at 0.95 recall. As mentioned above, CRINN achieves poor performance on NYTimes-256, where CRINN
underperforms the best baseline by 82.85% for the 0.9 recall setup.
5.3 Progressive Improvements for Different Modules
In CRINN, optimization proceeds sequentially through three modules: graph construction, search, and refinement. To evaluate
the individual contribution of each module, we examine the progressive performance improvements at each optimization stage.
We demonstrate these incremental gains by measuring the average QPS improvement across fixed recall values (0.90, 0.95,
0.99, and 0.999).
Table 4 presents the result. As can be seen, CRINN demonstrates substantial gains through all its three optimization stages.
The graph construction module yields the most significant individual improvements, averaging 22.11% across all recall values
and datasets, with particularly impressive results on gist-960-euclidean (58.26%) and mnist-784-euclidean (45.85%). The
search optimization module contributes an additional 18.30% on average, maintaining strong performance especially for
high-dimensional datasets. The refinement module, showing more modest individual gains of 9.69%. The diminishing returns
8

across the three stages can be attributed to two primary factors. First, the fundamental importance hierarchy of these stages
naturally leads to different improvement potentials. Graph construction is the foundation that determines the entire search space
structure. In contrast, the refinement stage serves a more specialized role of fine-tuning results, where the candidates are already
of reasonable quality, thus offering less dramatic improvement potential. Second, the optimization order plays a significant
role—earlier stages have more optimization opportunities available, while later stages operate on an already-improved system.
The graph construction stage works with the raw, unoptimized baseline, allowing techniques like adaptive search, multi-level
prefetching, and multi-entry points to capture the "low-hanging fruit" of performance improvements. By the time the refinement
stage is reached, many inefficiencies have already been addressed, leaving less room for dramatic improvement.
The only outlier is nytimes-256-angular, which shows performance degradation across all stages, suggesting that the optimization
techniques may need dataset-specific tuning for certain angular distance computations. Overall, the results validate the
effectiveness of the progressive optimization strategy, with five out of six datasets showing substantial cumulative improvements
ranging from 16% to 134%.
DatasetGraph Construction Search Refinement
Individual Cumulative Individual Cumulative Individual Cumulative
sift-128-euclidean +30.12% +30.12% +25.86% +55.98% +11.19% +67.17%
gist-960-euclidean +58.26% +58.26% +46.43% +104.69% +29.63% +134.32%
mnist-784-euclidean +45.85% +45.85% +44.49% +90.34% +18.30% +108.64%
glove-100-angular +13.19% +13.19% +19.03% +32.22% +5.86% +38.08%
glove-25-angular +6.94% +6.94% +6.52% +13.46% +2.70% +16.16%
nytimes-256-angular -21.68% -21.68% -32.54% -54.22% -9.56% -63.78%
Overall Average +22.11% +22.11% +18.30% +40.41% +9.69% +50.10%
Table 4: Average QPS improvement across different recall levels.
6 Analysis
Here, we conduct a detailed analysis of each of the three optimization modules, examining the specific changes introduced by
contrastive RL and how they contribute to improved performance.
6.1 Graph Construction
We identify the following three optimization strategies discovered by contrastive RL in the graph construction module.
Adaptive Search with Dynamic EF Scaling , which adjusts search effort based on target recall requirements, replacing the
fixed ef_construction parameter in the original code with an adaptive value. This strategy helps allocate computational resources
proportionally to quality requirements.
1// Old: Fixed search budget
2size_t ef = ef_construction; // Always constant
3
4// New: Adaptive search budget based on recall needs
5if (target_recall > critical_threshold)
6 dynamic_ef = ef_search *(1.0 + recall_excess *14.5);
7else
8 dynamic_ef = ef_search;
Zero-Overhead Multi-Level Prefetching , which replaces fixed prefetching with an adaptive multi-level strategy that considers
neighbor density and search layer. It reduces memory access latency compared to the fixed prefetch window in the old
implementation
1// Old: Fixed prefetch window
2for (int j = 0; j < min(5, size); ++j)
3 computer.prefetch(neighbors[j], 1);
4
5// New: Adaptive multi-level prefetching
6prefetch_depth = min(adaptive_depth, size); // 24-48 based on performance
9

7for (int j = 0; j < prefetch_depth; ++j)
8 computer.prefetch(neighbors[j], 3); // L1 cache
9if (high_recall_needed)
10 // Additional L2 prefetch for more neighbors
Multi-Entry Point Search Architecture , which maintains multiple diverse entry points for parallel exploration instead of a
single global entry point. This strategy improves recall for the same computational budget by exploring diverse graph regions
simultaneously.
1// Old: Single entry point
2start_node = enterpoint_node;
3results = search(start_node, query);
4
5// New: Multiple diverse entry points (up to 9)
6for (node : strategic_entrypoints) {
7 if (distance_to_others(node) > threshold)
8 entry_points.add(node);
9}
10 // Search from multiple starting points
11 for (ep : entry_points)
12 results.merge(search(ep, query));
6.2 Search
For search, we identify the following three optimization strategies proposed by contrastive RL.
Multi-Tier Entry Point Selection , which replaces single entry point initialization with a sophisticated multi-tier system that
selects from primary, secondary, and tertiary entry points based on search budget. This strategy improves search quality by
starting from diverse, high-quality nodes.
1// Old: Single entry point
2initialize_search(single_entry_point)
3
4// New: Multi-tier entry selection
5add_entry(primary_entry_point)
6if search_budget > threshold_1:
7 add_entry(secondary_entry_point)
8if search_budget > threshold_2:
9 add_entry(tertiary_entry_point)
Batch Processing with Adaptive Prefetching , which optimizes neighbor processing by collecting edges into batches and
using enhanced prefetch strategies. This reduces random memory access and improves cache utilization.
1// Old: Fixed prefetching
2for i in range(prefetch_count):
3 prefetch(neighbor[i])
4
5// New: Adaptive batch prefetching
6prefetch_size = prefetch_count *batch_factor
7for i in range(adaptive_prefetch_size):
8 prefetch(neighbor[i])
9 if processing_node[j]:
10 prefetch(neighbor[j + prefetch_size]) // Look ahead
Intelligent Early Termination with Convergence Detection , which monitors search progress and terminates early when
convergence is detected, avoiding unnecessary exploration while maintaining quality.
1// Old: Explore until pool exhausted
2while has_candidates():
3 process_neighbor()
4
5// New: Smart termination
6no_improvement_count = 0
7while has_candidates():
10

8 improvements = process_neighbor()
9 if improvements == 0:
10 no_improvement_count++
11 if check_convergence(no_improvement_count):
12 break // Early termination
6.3 Refinement
For the refinement module, RL proposed the following two optimization strategies.
Adaptive Memory Prefetching , which replaces basic hierarchical search with an intelligent prefetching system that adapts
based on edge patterns and node characteristics. This strategy significantly reduces memory latency during the refinement
process.
1// Old: Basic traversal without prefetching
2for each edge v in node_edges:
3 if distance(v) < best_distance:
4 update best_node
5
6// New: Adaptive prefetching with lookahead
7if should_prefetch:
8 prefetch(edges[0])
9for i, edge v in node_edges:
10 prefetch(edges[i + lookahead]) // Prefetch future edges
11 if distance(v) < best_distance:
12 update best_node
Pre-computed Edge Metadata with Pattern Recognition , which enhances the refiner by pre-computing and storing edge
counts for each node level. This eliminates redundant computations and enables pattern-based optimizations during refinement.
1// Old: Runtime edge counting
2count = 0
3for each edge in node:
4 if edge != -1:
5 count++
6
7// New: Pre-computed metadata access
8metadata = get_precomputed_metadata(level, node)
9edge_count = metadata.count
10 pattern_score = metadata.intelligence_score
11 // Use metadata for optimization decisions
12 if pattern_score > threshold:
13 apply_pattern_optimization()
7 Related Work
The past year has witnessed a surge of interest in leveraging LLMs and RL-augmented LLM models for code optimization. This
includes significant advances in compiler optimization [ 4], assembly code optimization [ 30], and CUDA kernel optimization
[13,2,18]. Reinforcement learning frameworks such as CodeRL [ 14] and PPOCoder [ 28] have emerged as powerful tools
for enhancing LLM performance in code generation and optimization tasks. Notably, RLEF [ 8] demonstrates that end-to-end
reinforcement learning can effectively train models to utilize execution feedback during code synthesis, achieving state-of-the-art
performance on competitive programming benchmarks.
In assembly code optimization, recent breakthroughs show that PPO-trained LLMs can achieve remarkable results—reaching
96.0% test pass rates and delivering 1.47 ×speedups compared to the gcc -O3 baseline [ 30]. Similarly, Meta’s LLM Compiler
[3] achieves 77% of the optimization potential of exhaustive autotuning searches, validating the effectiveness of LLMs for
optimizing compiler intermediate representations. For GPU-accelerated computing, CUrator [ 15] introduces an efficient LLM
execution engine that seamlessly integrates CUDA libraries such as cuBLAS and CUTLASS, optimizing performance for
modern language models. Complementing this, ComputeEval provides an open-source benchmark framework specifically
designed to evaluate LLM capabilities in CUDA programming tasks, establishing standardized metrics for this emerging field.
11

8 Conclusion
In this paper, we presented CRINN, a novel framework that employs contrastive reinforcement learning-augmented LLMs to
automatically optimize approximate nearest-neighbor search algorithms. By treating ANNS optimization as a reinforcement
learning problem where execution speed serves as the reward signal, CRINN successfully transforms the traditionally manual
and expertise-intensive optimization process into an automated search through the space of possible implementations. Our
experimental results demonstrate CRINN’s effectiveness across diverse benchmark datasets, achieving best-in-class performance
on three out of six benchmarks and matching state-of-the-art results on two others. The success of CRINN carries broader
implications beyond ANNS optimization. It demonstrates that RL-augmented LLMs can serve as powerful tools for automating
complex algorithmic optimizations that traditionally require deep domain expertise and extensive manual tuning. As the demand
for efficient vector search continues to grow with the proliferation of RAG and agent-based LLM applications, automated
optimization frameworks like CRINN will become increasingly valuable for maintaining competitive performance across
evolving hardware architectures and application requirements.
References
[1]AUMÜLLER , M., B ERNHARDSSON , E., AND FAITHFULL , A. J. Ann-benchmarks: A benchmarking tool for approximate
nearest neighbor algorithms. Inf. Syst. 87 (2018).
[2]CHEN, W., Z HU, J., F AN, Q., M A, Y., AND ZOU, A. Cuda-llm: Llms can write efficient cuda kernels. arXiv preprint
arXiv:2506.09092 (2025).
[3]CUMMINS , C., S EEKER , V., G RUBISIC , D., R OZIERE , B., G EHRING , J., S YNNAEVE , G., AND LEATHER , H. Meta
large language model compiler: Foundation models of compiler optimization. arXiv preprint arXiv:2407.02524 (2024).
[4]CUMMINS , C., S EEKER , V., G RUBISIC , D., R OZIERE , B., G EHRING , J., S YNNAEVE , G., AND LEATHER , H. Llm
compiler: Foundation language models for compiler optimization. In Proceedings of the 34th ACM SIGPLAN International
Conference on Compiler Construction (2025), pp. 141–153.
[5]DONG, W., M OSES , C., AND LI, K. Efficient k-nearest neighbor graph construction for generic similarity measures. In
Proceedings of the 20th international conference on World wide web (2011), pp. 577–586.
[6]DOUZE , M., G UZHVA , A., D ENG, C., J OHNSON , J., S ZILVASY , G., M AZARÉ , P.-E., L OMELI , M., H OSSEINI , L.,
AND JÉGOU , H. The faiss library. arXiv preprint arXiv:2401.08281 (2024).
[7]FU, C., X IANG , C., W ANG, C., AND CAI, D.Fast approximate nearest neighbor search with the navigating spreading-out
graph. arXiv preprint arXiv:1707.00143 (2017).
[8]GEHRING , J., Z HENG , K., C OPET , J., M ELLA , V., C ARBONNEAUX , Q., C OHEN , T., AND SYNNAEVE , G. Rlef:
Grounding code llms in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089 (2024).
[9]GUO, D., Z HU, Q., Y ANG , D., X IE, Z., D ONG , K., Z HANG , W., C HEN, G., B I, X., W U, Y., L I, Y. K., L UO, F.,
XIONG , Y., AND LIANG , W. Deepseek-coder: When the large language model meets programming - the rise of code
intelligence. ArXiv abs/2401.14196 (2024).
[10] GUU, K., L EE, K., T UNG , Z., P ASUPAT , P., AND CHANG , M.-W. Realm: Retrieval-augmented language model
pre-training. ArXiv abs/2002.08909 (2020).
[11] HUI, B., Y ANG , J., C UI, Z., Y ANG , J., L IU, D., Z HANG , L., L IU, T., Z HANG , J., Y U, B., D ANG , K., Y ANG , A.,
MEN, R., H UANG , F., Q UAN, S., R EN, X., R EN, X., Z HOU , J., AND LIN, J.Qwen2.5-coder technical report. ArXiv
abs/2409.12186 (2024).
[12] JAYARAM SUBRAMANYA , S., D EVVRIT , F., S IMHADRI , H. V., K RISHNAWAMY , R., AND KADEKODI , R. Diskann:
Fast accurate billion-point nearest neighbor search on a single node. Advances in neural information processing Systems
32(2019).
[13] LANGE , R. T., P RASAD , A., S UN, Q., F ALDOR , M., T ANG, Y., AND HA, D. The ai cuda engineer: Agentic cuda kernel
discovery, optimization and composition. Tech. rep., 2025.
[14] LE, H., W ANG , Y., G OTMARE , A. D., S AVARESE , S., AND HOI, S. C. H. Coderl: Mastering code generation through
pretrained models and deep reinforcement learning. ArXiv abs/2207.01780 (2022).
12

[15] LEE, Y. N., Y U, Y., AND PARK, Y. Curator: An efficient llm execution engine with optimized integration of cuda
libraries. In Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization (2025),
pp. 209–224.
[16] LEWIS , P., P EREZ , E., P IKTUS , A., P ETRONI , F., K ARPUKHIN , V., G OYAL , N., K UTTLER , H., L EWIS , M., TAU YIH,
W., R OCKTÄSCHEL , T., R IEDEL , S., AND KIELA , D. Retrieval-augmented generation for knowledge-intensive nlp tasks.
ArXiv abs/2005.11401 (2020).
[17] LI, J., L IU, H., G UI, C., C HEN, J., N I, Z., AND WANG, N. The design and implementation of a real time visual search
system on jd e-commerce platform, 2019.
[18] LI, X., S UN, X., W ANG, A., L I, J., AND SHUM , C.Cuda-l1: Improving cuda optimization via contrastive reinforcement
learning. arXiv preprint arXiv:2507.14111 (2025).
[19] LIN, J., M A, X., L IN, S.-C., Y ANG , J.-H., P RADEEP , R., AND NOGUEIRA , R. Pyserini: A python toolkit for
reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th international
ACM SIGIR conference on research and development in information retrieval (2021), pp. 2356–2362.
[20] MALKOV , Y. A., AND YASHUNIN , D. A. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42 , 4 (2018), 824–836.
[21] MANOHAR , M. D., S HEN, Z., B LELLOCH , G., D HULIPALA , L., G U, Y., S IMHADRI , H. V., AND SUN, Y.Parlayann:
Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms. In Proceedings of the
29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (2024), pp. 270–285.
[22] MANOHAR , M. D., S HEN, Z., B LELLOCH , G. E., D HULIPALA , L., G U, Y., S IMHADRI , H. V., AND SUN, Y.Parlayann:
Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms. Proceedings of the 29th
ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (2023).
[23] MENG, Y., L I, X., Z HENG , X., W U, F., S UN, X., Z HANG , T., AND LI, J.Fast nearest neighbor machine translation. In
Findings (2021).
[24] ONO, N., AND MATSUI , Y.Relative nn-descent: A fast index construction for graph-based approximate nearest neighbor
search. In Proceedings of the 31st ACM International Conference on Multimedia (2023), pp. 1659–1667.
[25] PARK, J. S., O’B RIEN , J. C., C AI, C. J., M ORRIS , M. R., L IANG , P., AND BERNSTEIN , M. S. Generative agents:
Interactive simulacra of human behavior. Proceedings of the 36th Annual ACM Symposium on User Interface Software
and Technology (2023).
[26] ROMERA -PAREDES , B., B AREKATAIN , M., N OVIKOV , A., B ALOG , M., K UMAR , M. P., D UPONT , E., R UIZ, F. J.,
ELLENBERG , J. S., W ANG , P., F AWZI , O., ET AL .Mathematical discoveries from program search with large language
models. Nature 625 , 7995 (2024), 468–475.
[27] SHAO, Z., W ANG , P., Z HU, Q., X U, R., S ONG , J., B I, X., Z HANG , H., Z HANG , M., L I, Y., W U, Y., ET AL .
Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300
(2024).
[28] SHOJAEE , P., J AIN, A., T IPIRNENI , S., AND REDDY , C. K. Execution-based code generation using deep reinforcement
learning. ArXiv abs/2301.13816 (2023).
[29] W ANG , Z. Graph library for approximate similarity search, 4 2025.
[30] WEI, A., S URESH , T., T AN, H., X U, Y., S INGH , G., W ANG, K., AND AIKEN , A.Improving assembly code performance
with large language models via reinforcement learning. arXiv preprint arXiv:2505.11480 (2025).
13