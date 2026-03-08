# Core-based Hierarchies for Efficient GraphRAG

**Authors**: Jakir Hossain, Ahmet Erdem Sarıyüce

**Published**: 2026-03-05 14:17:30

**PDF URL**: [https://arxiv.org/pdf/2603.05207v1](https://arxiv.org/pdf/2603.05207v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge. However, existing vector-based methods often fail on global sensemaking tasks that require reasoning across many documents. GraphRAG addresses this by organizing documents into a knowledge graph with hierarchical communities that can be recursively summarized. Current GraphRAG approaches rely on Leiden clustering for community detection, but we prove that on sparse knowledge graphs, where average degree is constant and most nodes have low degree, modularity optimization admits exponentially many near-optimal partitions, making Leiden-based communities inherently non-reproducible. To address this, we propose replacing Leiden with k-core decomposition, which yields a deterministic, density-aware hierarchy in linear time. We introduce a set of lightweight heuristics that leverage the k-core hierarchy to construct size-bounded, connectivity-preserving communities for retrieval and summarization, along with a token-budget-aware sampling strategy that reduces LLM costs. We evaluate our methods on real-world datasets including financial earnings transcripts, news articles, and podcasts, using three LLMs for answer generation and five independent LLM judges for head-to-head evaluation. Across datasets and models, our approach consistently improves answer comprehensiveness and diversity while reducing token usage, demonstrating that k-core-based GraphRAG is an effective and efficient framework for global sensemaking.

## Full Text


<!-- PDF content starts -->

Core-based Hierarchies for Efficient GraphRAG
Jakir Hossain
University at Buffalo
Buffalo, NY, USA
mh267@buffalo.eduAhmet Erdem Sarıyüce
University at Buffalo
Buffalo, NY, USA
erdem@buffalo.edu
Abstract
Retrieval-Augmented Generation (RAG) enhances large language
models by incorporating external knowledge. However, existing
vector-based methods often fail on global sensemaking tasks that
require reasoning across many documents. GraphRAG addresses
this by organizing documents into a knowledge graph with hierar-
chical communities that can be recursively summarized. Current
GraphRAG approaches rely on Leiden clustering for community de-
tection, but we prove that on sparse knowledge graphs, where aver-
age degree is constant and most nodes have low degree, modularity
optimization admits exponentially many near-optimal partitions,
making Leiden-based communities inherently non-reproducible. To
address this, we propose replacing Leiden with𝑘-core decomposi-
tion, which yields a deterministic, density-aware hierarchy in linear
time. We introduce a set of lightweight heuristics that leverage the 𝑘-
core hierarchy to construct size-bounded, connectivity-preserving
communities for retrieval and summarization, along with a token-
budget–aware sampling strategy that reduces LLM costs. We evalu-
ate our methods on real-world datasets including financial earnings
transcripts, news articles, and podcasts, using three LLMs for an-
swer generation and five independent LLM judges for head-to-head
evaluation. Across datasets and models, our approach consistently
improves answer comprehensiveness and diversity while reduc-
ing token usage, demonstrating that 𝑘-core-based GraphRAG is an
effective and efficient framework for global sensemaking.
Keywords
GraphRAG, k-core, Leiden, RAG
1 Introduction
Retrieval-Augmented Generation (RAG) enhances large language
models (LLMs) with external knowledge, enabling more grounded
and accurate responses to complex queries [ 2,7,16,17,33]. It is
particularly useful when the corpus is too large to fit within an
LLM’s context window [ 15,18]. In a typical RAG setup, a small set
of relevant records is retrieved and fed, along with the query, to
the LLM for response generation. RAG has been widely adopted in
domains such as healthcare, law, finance, and education, improving
faithfulness, reducing hallucinations, mitigating privacy risks, and
enhancing robustness [19, 30, 31, 34, 35].
Most RAG systems rely on lexical/semantic retrieval over text
and are effective only for queries answerable from a few documents.
These approaches struggle with multi-hop reasoning or queries re-
quiring synthesis across semantically diverse documents or global
corpus understanding [ 15,18]. For instance, answering “How have
patient outcomes in cancer treatment evolved in response to multi-
center clinical trials over the last 15 years?” demands aggregating
evidence across many sources, which conventional RAG methods
cannot easily support. Similarly, a financial analyst asking “Whatcommon strategies have semiconductor companies adopted in re-
sponse to supply chain disruptions over the past decade?” must
synthesize themes across hundreds of earnings call transcripts, a
task that no single retrieved passage can answer. We refer to such
queries asglobal sensemakingtasks: they require the system to
identify recurring themes, reconcile conflicting perspectives, and
synthesize evidence distributed across the entire corpus, capabilities
that go beyond standard RAG.
Graph-based Retrieval-Augmented Generation (GraphRAG) has
recently emerged as a promising approach for global sensemaking
tasks by representing the corpus as a knowledge graph and organiz-
ing it into a hierarchy of communities that can be recursively sum-
marized [ 6]. The GraphRAG framework introduced by Edge et al .
relies on modularity-based community detection, such as Leiden
[28]. However, Leiden often produces hierarchies that are shallow,
overly fragmented, and/or dominated by a few large communities.
Building on an observation by Good, de Montjoye, and Clauset[ 9],
we prove that when the average degree is constant and most nodes
have low degree—the regime of typical knowledge graphs—the
number of near-optimal modularity partitions is exponential in the
graph size (Theorem 1). This implies that Leiden-based communi-
ties are inherently non-reproducible on such graphs, compounding
known issues with resolution limits and sensitivity to initialization.
In practice, this instability causes communities to unpredictably
merge or fragment semantically meaningful structures.
Motivated by these limitations, we explore 𝑘-core–based alterna-
tivesfor hierarchical graph construction in GraphRAG. The 𝑘-core
decomposition organizes a network into nested layers of increasing
minimum degree [ 23]. Each node is assigned a core number: the
largest𝑘for which it belongs to a subgraph where every node has
at least𝑘neighbors. This creates a deterministic, density-aware
hierarchy that can be computed in a single 𝑂(|𝐸|) pass. Unlike
modularity-driven methods, the 𝑘-core hierarchy naturally captures
progressively denser and more cohesive substructures, making it
well suited for global sensemaking tasks that require integrating
evidence across multiple interconnected regions of the corpus. In
knowledge graphs specifically, higher 𝑘-cores correspond to enti-
ties connected through multiple distinct relational paths, providing
a natural proxy for topical centrality that modularity’s comparison
to a degree-preserving null model cannot capture.
We operationalize this hierarchy through a set of lightweight, in-
terpretable community construction heuristics that operate over dif-
ferent levels of the 𝑘-core decomposition. Our approach constructs
balanced hierarchical communities while explicitly controlling clus-
ter sizes to respect LLM context constraints. These heuristics enable
comprehensive and diverse summaries without excessive token us-
age. Additionally, we introduce a token-budget–aware sampling
strategy that further reduces LLM costs while preserving retrieval
quality. We make the following key contributions:arXiv:2603.05207v1  [cs.IR]  5 Mar 2026

Jakir Hossain and Ahmet Erdem Sarıyüce
•We introduce 𝑘-core decomposition as a drop-in replace-
ment for Leiden in GraphRAG, yielding deterministic, density-
aware hierarchies in linear time.
•We prove that modularity optimization on sparse graphs
admits exponentially many near-optimal partitions (Theo-
rem 1), formally explaining why Leiden-based community
detection is unreliable on knowledge graphs.
•We propose multiple hierarchical heuristic strategies that
exploit different regions of the induced 𝑘-core hierarchy to
balance coverage, granularity, and efficiency.
•We conduct extensive evaluations on three real-world datasets,
using three LLMs for answer generation and five indepen-
dent LLM judges for head-to-head evaluation.
2 Related Work and Background
In this section, we review the literature on Graph-based RAG and
remind key concepts of GraphRAG workflows and 𝑘-core decom-
position.
Graph-based RAG.Traditional vector-based RAG retrieves rel-
evant passages to augment LLM responses, but is largely limited
to single-hop, fact-based queries where the answer resides within
one or a few contiguous passages. When reasoning must span mul-
tiple documents or synthesize information across a corpus, these
approaches fall short [10, 20].
To address this, several methods incorporate graph structures
into the retrieval process. One line of work constructs query-focused
subgraphs from knowledge graphs and performs reasoning over
them. Systems such as KG-GPT [ 14], G-Retriever [ 11], GRAG [ 12],
and HLG [ 8] follow this paradigm, extracting relevant subgraphs
and using LLMs to reason over the retrieved structure, thereby
improving multi-hop answer quality. A complementary line of
work enhances the retrieval step itself with graph signals, using
graph topology to guide which documents or passages are se-
lected [ 20,25,29]. However, both subgraph-reasoning and graph-
enhanced retrieval methods are primarily designed for queries that
require a bounded number of reasoning hops. They do not address
global sensemaking, tasks where the answer requires understanding
themes, patterns, or relationships that emerge only when consider-
ing an entire corpus.
GraphRAG [ 6] addresses this gap by applying Leiden-based hi-
erarchical community detection on a knowledge graph and using
the resulting communities for global sensemaking (see Section 2.1
for details). A key limitation, however, is that modularity-based
clustering methods like Leiden can produce highly uneven hier-
archies, where community sizes vary widely across levels. This
imbalance can lead to inconsistent summary quality and makes it
difficult to control the granularity of retrieved context. Our work
addresses this limitation by replacing community detection with
𝑘-core decomposition, which yields a naturally nested and more
balanced hierarchy of dense subgraphs.
2.1 Community-based GraphRAG Overview
Edge et al .[6] introduced a graph-based approach for global sense-
making question answering by a community-driven GraphRAG
workflow. Here, we briefly describe the two main stages of theGraphRAG workflow: indexing and query-time answer generation,
followed by their evaluation conditions.
Indexing.Large text corpora are first split into manageable chunks,
from which entities, relationships, and claims are extracted using
LLMs. These elements are aggregated into a knowledge graph,
where nodes represent entities or claims and edges represent rela-
tionships, weighted by frequency. The graph is hierarchically parti-
tioned into communities using Leiden community detection [ 28],
recursively identifying sub-communities until reaching leaf-level
communities that cannot be further split. Leaf-level communities
are summarized first by prioritizing elements according to edge
prominence, adding source and target node descriptions, edges,
and related claims into the LLM context until the token limit is
reached. For higher-level communities, if all element summaries fit
within the context window, they are summarized directly; other-
wise, sub-community summaries iteratively replace longer element-
level descriptions until they fit. This process, from knowledge graph
construction to hierarchical community summaries, is referred to
as indexing.
Query-Time Answer Generation.Given a user query, indexed
community summaries are used to generate a final answer through
a multi-stage Map-Reduce process. First, summaries are randomly
shuffled and divided into chunks of a pre-specified token size to
ensure relevant information is distributed across multiple contexts
rather than concentrated in a single window. Next, intermediate
answers are generated for each chunk in parallel, with the LLM also
assigning a helpfulness score between 0 and 100. These intermediate
answers are then sorted by helpfulness and iteratively added into
a new context window until the token limit is reached. The LLM
uses this aggregated context to produce the final global answer.
By leveraging the hierarchical structure of the communities, this
approach allows answers to be generated from different levels and
enables robust aggregation of information from all relevant sub-
communities.
Evaluation for GraphRAG.Evaluating RAG systems for multi-
document reasoning presents distinct challenges beyond standard
QA benchmarks. Recent work assesses output quality along dimen-
sions such as comprehensiveness, diversity, empowerment, and
directness [ 6]. Edge et al .tested six conditions, including four hi-
erarchical community levels (C0–C3), source-text summarization,
and a vector-based RAG approach. The hierarchical configurations
differ in granularity.C0uses only top-level communities, yielding
the smallest set of summaries,C1uses communities one level below
the root for a slightly finer decomposition and so on. Results on two
datasets show that all community-level approaches (C0–C3) out-
perform both source-text summarization and vector RAG for global
sensemaking. Among these,C2andC3consistently achieve the
best performance, hence we consider those two for our evaluation.
2.2 Hierarchical𝑘-core Decomposition
In this work, we consider the knowledge graph as an unweighted
graph𝐺=(𝑉,𝐸) , where𝑉denotes the set of nodes and 𝐸de-
notes the set of edges. The degree of a node 𝑢within a subgraph
𝑆⊆𝐺 is denoted𝑑𝑒𝑔(𝑢,𝑆) . A𝑘-core is the maximal connected
subgraph𝑆⊆𝐺 in which every vertex has at least 𝑘neighbors,
i.e.,𝑑𝑒𝑔(𝑢,𝑆)≥𝑘∀𝑢∈𝑆 . Each node is assigned acore number,

Core-based Hierarchies for Efficient GraphRAG
the largest𝑘for which it belongs to a 𝑘-core. The𝑘-shell is the
set of nodes with core number 𝑘[4]. The𝑘-cores for all 𝑘can be
computed by recursively removing nodes with degree less than 𝑘
and their edges, which runs in 𝑂(|𝐸|) time [ 1]. This process of iden-
tifying all𝑘-cores and assigning core numbers is referred to as core
decomposition. Since nodes can belong to multiple 𝑘-cores with
different𝑘values, the decomposition naturally forms a hierarchy
of nested dense subgraphs. This hierarchy can be represented as a
tree, with nodes denoting subgraphs and edges encoding contain-
ment relationships. The root represents the entire graph (1-core),
while child nodes correspond to subgraphs with increasing core
numbers. Internal nodes may contain denser subgraphs with higher
core values, forming a multi-level structure that captures the nested
organization of densely connected regions within the graph.
3 Why Modularity Optimization is Unreliable
on Sparse Knowledge Graphs?
Modularity-based community detection methods such as Leiden [ 28]
compare observed within-community edge density to the expec-
tation under a degree-preserving null model (the configuration
model), an approach that works well in dense networks but de-
grades in sparse ones. Good, de Montjoye, and Clauset [ 9] provided
a systematic characterization of this degradation. They showed that
the modularity function 𝑄typically admits an exponentially large
number of distinct high-scoring partitions while lacking a clear
global maximum: a phenomenon they termed thedegeneracy of mod-
ularity. Using real-world metabolic networks, they demonstrated
that these near-optimal partitions can fundamentally disagree on
key properties such as the composition of the largest modules and
the distribution of module sizes, meaning that any optimizer (in-
cluding Louvain and Leiden) is effectively selecting one solution
from a vast equivalence class, guided more by random seeds and
tie-breaking rules than by genuine structure. They further showed
that𝑄𝑚𝑎𝑥depends strongly on both network size and the num-
ber of modules, and that degeneracy is most severe in sparse and
hierarchically structured networks.
Modularity Degeneracy.We give a key structural observation:
the low-degree nodes (degree at most a fixed constant 𝑑) have
vanishing modularity sensitivity in sparse graphs, because their
few edges contribute negligibly to the objective. Denote by 𝑛≤d=
|{𝑖∈𝑉 :𝑘𝑖≤𝑑}| the number of such nodes. For small 𝑑, these
nodes can be reassigned across communities with negligible impact
on modularity, creating an exponentially large degenerate set.
Recall that the modularity of a partition 𝜎of a graph𝐺=(𝑉,𝐸)
with𝑛nodes and𝑚edges is𝑄(𝜎)=1
2𝑚Í
𝑖,𝑗[𝐴𝑖𝑗−𝑘𝑖𝑘𝑗/2𝑚]𝛿(𝜎𝑖,𝜎𝑗),
where𝑘𝑖is the degree of node 𝑖. Let𝑄∗=max𝜎𝑄(𝜎) denote the op-
timal modularity. For 𝜀>0, we define the 𝜀-degeneracyof 𝐺as the
number of structurally distinct partitions achieving near-optimal
modularity:D(𝜀)=|{𝜎∈P(𝐺):𝑄∗−𝑄(𝜎)<𝜀}|.
Theorem 1 (Modularity Degeneracy in Sparse Graphs).Let
𝐺be a graph with 𝑛nodes,𝑚edges, average degree ¯𝑘=2𝑚/𝑛=𝑂( 1),
and𝑛≤d=Θ(𝑛). Then for any𝜀>𝑑(2+ ¯𝑘)/(2𝑚),
D(𝜀) ≥2𝑛≤d/(𝑑+1).In particular, the number of near-optimal partitions is exponential
in𝑛, and the tolerance threshold 𝜀required to trigger this blowup is
𝑂(1/𝑛).
Proof sketch and the full proof are given in Appendix A.
Knowledge graphs (KG) are particularly well-suited to 𝑘-core
decomposition over modularity-based methods for two structural
reasons. First, their heavy-tailed but low-mean degree distributions
mean that𝑛≤𝑑=Θ(𝑛) , placing them squarely in the regime where
Theorem 1 bites hardest. Second, KG edges are semantically mean-
ingful rather than stochastic, so the 𝑘-core criterion, requiring each
node to have at least 𝑘neighbors, directly captures genuine rela-
tional richness in a way that comparison to a degree-preserving
null model does not.
4 A Robust Alternative:𝑘-core Decomposition
Theorem 1 exposes a fundamental instability: for a graph with low
average degree; any modularity optimizer, Leiden included, selects
one partition from an exponentially large near-optimal plateau.
Different seeds, tie-breaking rules, or minor edge perturbations
yield structurally different communities.
The𝑘-core decomposition sidesteps this problem entirely. It
isunique: for every graph 𝐺and integer 𝑘, the𝑘-core𝐻𝑘is the
unique maximal subgraph with minimum degree ≥𝑘, computed by
the deterministic peeling process; no optimization landscape, no
stochasticity. It isrobust: adding or removing a few edges changes
the shells by at most a proportional number of nodes, whereas
Theorem 1 implies that even a single edge can shift the optimal par-
tition across the entire degenerate set. And 𝑘-core decomposition
provides anatural hierarchy: the nested shells 𝐻1⊇𝐻 2⊇··· re-
flect structural connectivity rather than comparison to a null model.
In the sparse regime where 𝑘𝑖·𝑘𝑗/2𝑚≈ 0for most pairs, the null
model is nearly vacuous yet 𝑘-core structure remains informative.
The 2-core captures the backbone of multiply-connected nodes
while degree-1 nodes form the periphery. In knowledge graphs, this
distinction is semantically meaningful: multiple relational paths
between entities signal genuine topical relatedness. Last, but not
least, computing 𝑘-core is much cheaper than Leiden; a single pass
over the graph suffices to compute all and a hierarchy [21].
In GraphRAG, community assignments determine which entities
are summarized together and which retrieval units are constructed.
Exponential degeneracy means Leiden-based pipelines produce
non-reproducible summaries: the same knowledge graph under
different random seeds yields different communities and different
retrieval behavior, with communities that are either too fragmented
(splitting apart a natural topic) or too arbitrary (grouping unrelated
peripheral nodes together because the optimization found a mar-
ginal modularity gain). The 𝑘-core decomposition eliminates this
variance. Its nested shells provide a small number of progressively
tighter subgraphs where the innermost cores represent the most
interconnected, and therefore most semantically central — concepts,
while the outer shells supply context. This maps naturally onto a
summarization hierarchy: summarize the dense core first, then ex-
pand outward, producing stable, deterministic retrieval units from
which summaries and indices can be reliably built.
Remark.A natural refinement of 𝑘-core is the 𝑘-truss [ 5], which
requires every edge to participate in at least 𝑘−2triangles, yielding

Jakir Hossain and Ahmet Erdem Sarıyüce
e
m
o pn
k
lgf1-core
a-p
2-core
f-p
ba1-core
2-core3-core
m-pRes
f-hi
j
cd3-coreRes
a, b
Res
i, jh2-hop
c, d
Res
k, l
SGLN
e
Figure 1:𝑘-core decomposition (left) and corresponding hier-
archy tree produced byRkH(right).
tighter subgraphs. However, knowledge graphs are triangle-poor:
edges typically connect entities through distinct relation types
(e.g.,born_in,capital_of), producing bipartite-like local structure
with global clustering coefficients well below 0.05. Even the 3-truss
discards the vast majority of the graph, making the degree-based
𝑘-core criterion the right granularity for sparse relational graphs.
Motivated by these advantages, we propose a set of heuristics
for efficient global sensemaking in GraphRAG. These heuristics
leverage the 𝑘-core hierarchy to generate size-constrained clusters,
merge small two-hop clusters, and handle residual components.
4.1 Handling Residuals in𝑘-core Hierarchy
Our first heuristic builds on the 𝑘-core hierarchy while address-
ing practical issues such as singleton nodes and oversized clusters
by separating dense cores from sparse residuals. At each level,
dense components are recursively partitioned into size-bounded,
connectivity-preserving clusters, while low-core residuals are han-
dled separately to avoid distorting core structure. The result is
a deterministic hierarchy that captures both central regions and
peripheral context, enabling reliable hierarchical summarization.
Algorithm 1 presents the resultingResidual-aware 𝑘-core Hi-
erarchyheuristic, RkHin short. It takes two parameters: the input
graph (𝐺), and the maximum cluster size ( 𝑀).RkHbegins by ex-
tracting the largest connected component and removing self-loops
(line 1), as done in GraphRAG [ 6]. It then computes the core number
for each node (line 2). The hierarchy construction is initialized by
settingℓ=1, enqueuing full node set 𝑉intoQ, and initializing an
empty cluster setCand global singleton setG 𝑠(lines 3–4).
The graph is then processed iteratively over increasing core
levelsℓuntil the maximum core is reached (line 5), splitting each
cluster𝑆into core (𝑐(𝑣)≥ℓ ) and residual ( 𝑐(𝑣)<ℓ ) nodes (line 8).
Core components smaller than or equal to 𝑀are added directly to C
and the queue; larger components are split into size-bounded clus-
ters viaSplitbefore adding (lines 9–11). The maximum cluster size
(𝑀) is set by dividing the input token limit by the average tokens per
node, providing a practical estimate that keeps most clusters within
the allowed input size. Residual nodes are handled similarly, either
added directly orSplitas needed (lines 13–14); since these residual
subgraphs correspond to leaf clusters, they are not pushed to the
next-level queue. TheSplitprocedure breaks each oversized cluster
into smaller, size-bounded clusters by greedily growing each cluster
from a high-degree seed node, prioritizing neighbors that maximize
internal connectivity. This ensures that dense regions remain intactAlgorithm 1: RkH: Residual-aware 𝑘-core Hierarchy ( 𝐺,𝑀)
Require:Graph𝐺=(𝑉,𝐸), max cluster size𝑀
Ensure:Hierarchical clustersC
1:𝐺←LCC(𝐺); remove self-loops
2:Compute core number𝑐(𝑣)for all𝑣∈𝑉
3:Initialize queueQ←{(𝑉)}and levelℓ←1
4:Initialize cluster setC←∅, Global singletonG 𝑠←∅
5:whileℓ≤max𝑐(𝑣)do
6:Q′←∅
7:foreach cluster𝑆∈Qdo
8:𝑆 𝑐←{𝑣∈𝑆|𝑐(𝑣)≥ℓ},𝑆 𝑟←𝑆\𝑆 𝑐
9:foreach connected component𝑅of𝐺[𝑆 𝑐]do
10:𝑅′←(
{𝑅},|𝑅|≤𝑀
Split(𝐺,𝑅,𝑀)otherwise
11:C←C∪𝑅′,Q′←Q′∪𝑅′
12:end for
13:foreach connected component𝑅of𝐺[𝑆 𝑟]do
14:C←C∪(
{𝑅},|𝑅|≤𝑀
Split(𝐺,𝑅,𝑀)otherwise
15:end for
16:𝑆 single←{𝑅∈C||𝑅|=1}
17:H←{𝑅⊆𝑆 single|𝑅is 2-hop connected in𝐺}
18:G 𝑠←𝑆 single\Ð
𝑅∈H𝑅
19:foreach 2-hop component𝑅ofHdo
20:C←C∪(
{𝑅},|𝑅|≤𝑀
Split-2hop(𝐺,𝑅,𝑀)otherwise
21:end for
22:end for
23:Q←Q′,ℓ←ℓ+1
24:end while
25:Attach each ofG 𝑠to neighboring clusters inC
26:returnC
while respecting the maximum cluster size; full algorithmic details
are provided in Algorithm 3 in Appendix B.1.
Now, we extract singleton clusters from the existing set C(line 16)
into𝑆single and form new clusters Hconsisting of nodes that are
2-hop connected in 𝐺(line 17). A set 𝑅is 2-hop connected if each
node𝑢∈𝑅 has a path of length ≤2in𝐺to some other node 𝑣∈𝑅 .
Any remaining singleton nodes not included in Hare added to
global setG𝑠(line 18). Now, each 2-hop component from His either
added directly toCor split into smaller clusters viaSPLIT-2HOPif
it exceeds size𝑀(lines 19–20).
TheSPLIT-2HOPprocedure greedily splits a 2-hop connected
component, ensuring that nodes sharing common anchors are
grouped together. Starting from a seed node with the largest anchor
set, the algorithm greedily grows a cluster by adding nodes that
share anchors with the current cluster, continuing until the cluster
reaches the maximum size or no eligible nodes remain. Anchor
nodes connected to multiple cluster members are then included to
maintain structural coherence. The process produces size-bounded
clusters that respect both 2-hop connectivity and anchor relation-
ships. Full details are provided in Algorithm 4 in Appendix B.2.
After all clusters at the current level are processed, the queue
is updated with the newly formed clusters, and the level counter
is incremented (line 23). Once all core levels have been processed,
a final attachment step assigns each node in G𝑠to a neighboring
cluster inC, ensuring that no isolated nodes remain in the hierarchy

Core-based Hierarchies for Efficient GraphRAG
(line 25). Finally, the algorithm returns the hierarchical cluster
setC(line 26). This approach ensures size-bounded, connectivity-
preserving clusters while capturing both dense core structures and
sparse residual nodes.
Figure 1 shows an example hierarchy produced by RkH. The
1-core contains all nodes (a–p). At the next level, the 2-core consists
of nodes (f–p), while the remaining nodes (a–e) form different
clusters, including a connected component (a–b), a 2-hop connected
group (c–d), and a singleton (e). At the 3-core level, only nodes
(m–p) remain, and the residual nodes (f–l) are partitioned into
three connected components (f–h), (i–j), and (k–l). Note that the
singleton (e) is attached to its neighboring cluster (f–h) at the end.
This recursive decomposition yields a hierarchy where higher 𝑘-
cores capture denser subgraphs and residuals are organized into
progressively finer-grained structures, as formalized in Algorithm 1.
4.2 Handling Small Clusters
Due to the sparse nature of knowledge graphs, many tiny clusters
(often containing only two nodes) can arise from previous heuris-
tic. As a result, when GraphRAG evaluates community relevance
during query answering, these small clusters usually receive low
scores. Consequently, they are often excluded from the final answer
generation, which can affect overall sensemaking performance. We
propose to explicitly merge such small clusters.
Merging 2-hop Small Clusters.In RkH, some 2-hop clusters gen-
erated in lines 19–20 may contain only two nodes. These very
small clusters can fragment the hierarchy and reduce connectivity
if left unprocessed. To address this, we apply a post-processing step
after Algorithm 1 that merges small 2-hop clusters into larger clus-
ters whenever possible, or creates new clusters when no suitable
neighbors exist. The merging procedure is detailed in Algorithm 2,
referred to as M2hC . First, all 2-hop clusters of size two are sepa-
rated from the remaining clusters (lines 1–2). While small clusters
remain (line 3), the algorithm selects the small cluster with the
most connections to existing clusters (line 4) and counts its neigh-
bors in each cluster (line 5). If neighbors exist, the small cluster
is merged in-place into the cluster with which it shares the most
edges (lines 6–8). Otherwise, a new cluster is created for the small
cluster (line 9). After each iteration, the processed small cluster is
removed from the pool (line 12). Finally, the algorithm returns the
updated cluster set with all small clusters integrated (line 14).
Merging Residual Clusters.In addition to the previous heuristics,
we observe that many residual connected components produced
during RkH(lines 13–14) have size two. To address this, we introduce
an extended heuristic, referred to as MRC.MRCis conceptually similar
toM2hC , but instead of focusing solely on 2-hop clusters, it handles
all small residual clusters of size two. The only modification occurs
in line 1 of Algorithm 2, where both residual clusters and 2-hop
clusters of size two are collected into the set Sfor processing. The
rest of the procedure remains unchanged: each small cluster is
merged into a neighboring parent cluster when possible; otherwise,
a new cluster is created only if no suitable neighbors exist.Algorithm 2:M2hC: Merge 2-hop Clusters (𝐺,C)
Require:Graph𝐺=(𝑉,𝐸), existing cluster setC
Ensure:Updated cluster setCwith small clusters merged
1:S←All 2-hop clusters of size 2 inC
2:L←C\S
3:whileS≠∅do
4:Pick small cluster𝐶 𝑠∈Swith the most neighbors inL
5:Count neighbors of𝐶 𝑠in each cluster of𝐿∈L
6:ifany neighbors existthen
7:Find𝐿 𝑏𝑒𝑠𝑡∈Lwith most neighbors of𝐶 𝑠
8:𝐿𝑏𝑒𝑠𝑡←𝐿𝑏𝑒𝑠𝑡∪𝐶𝑠
9:else
10:L←L∪𝐶 𝑠
11:end if
12:Remove𝐶 𝑠fromS
13:end while
14:returnL
Note that, for M2hC andMRC, we focus specifically on clusters
of size two, as they account for the majority of small clusters ob-
served in practice. Clusters of size three or larger occur infrequently,
and trying to merge would not yield measurable improvements;
moreover, merging them would unnecessarily inflate neighboring
clusters and weaken size constraints.
4.3 Token Efficiency via Sampling
In large hierarchical graph decompositions, leaf-level communi-
ties can be densely connected, and many nodes/edges within the
same community frequently carry overlapping information. Pass-
ing all of this content to an LLM for retrieval or summarization can
therefore be costly in terms of tokens, leading to inefficiency and
redundancy. To address this challenge, we propose theRound-Robin
Token-Constrained Selection(RRTC) heuristic, which efficiently re-
duces token usage while preserving the most informative edges
from each community. RRTC operates on leaf-level communities
produced by any hierarchy heuristic (e.g., RkH,M2hC , orMRC) and
selects a representative subset of edges under a fixed token budget.
Within each community, edges are ranked by the combined degree
of their endpoints to reflect overall prominence. Selection then
proceeds in a round-robin fashion across leaf-level communities,
traversing from higher to lower 𝑘-shells, until the token budget
is exhausted. By sampling a representative subset of edges from
each community, RRTC captures the essential information without
including all nodes/edges.
5 Experimental Setup
In this section, we describe the experimental setup, including the
datasets and evaluation approach. Since our study targets the global
sensemaking task, we selected three datasets requiring reasoning
over multiple documents. The first two align with the original
Edge et al .’s study, while the third also requires multi-document
reasoning. Dataset sizes range from approximately 1M to 6M tokens,
all processed using the same knowledge graph construction pipeline
used by [6]. Details of these datasets are reported in Table 1.
Podcast transcripts ( podcast ).Public transcripts ofBehind the
Tech with Kevin Scott, a podcast featuring conversations between

Jakir Hossain and Ahmet Erdem Sarıyüce
Table 1: Summary statistics of datasets used in our experi-
ments, showing model, dataset, number of documents, and
graph information ( |𝑉|: nodes,|𝐸|: edges,𝐾𝑚: maximum core
number). Post-cutoff data is used for GPT-3.5-turbo (cutoff:
Sep 1, 2021) and GPT-4o-mini (cutoff: Oct 1, 2023), while full
data is used forGPT-5-mini.
Model Dataset # Docs |V| |E|K 𝑚
GPT-3.5-turbo
(post-cutoff)podcast36 1791 2651 6
news609 10655 15368 6
semiconductor183 4324 9546 13
GPT-4o-mini
(post-cutoff)podcast13 628 882 5
news331 5312 7316 6
semiconductor63 2029 4249 9
GPT-5-mini
(full)podcast72 3323 7761 7
news609 4499 7941 8
microsoft45 790 1702 7
Microsoft CTO Kevin Scott and various thought leaders in science
and technology [ 22]. The corpus contains 72 documents with ap-
proximately 1 million tokens in total.
News articles ( news ).A benchmark dataset of news articles span-
ning multiple categories, including entertainment, business, sports,
technology, health, and science [ 26]. It is a multi-document QA
dataset containing 609 documents and nearly 1.4 million tokens.
S&P 500 earnings transcripts.This dataset consists of earnings
call transcripts from publicly listed S&P 500 companies spanning
2013–2025 [ 3], with over 20,000 documents totaling 232M tokens.
ForGPT-3.5-turbo andGPT-4o-mini , to focus on multi-document
synthesis and keep the dataset manageable, we use only post-cutoff
transcripts from the Semiconductor industry ( semiconductor ),
yielding 183 and 63 documents, respectively. For GPT-5-mini , since
the full semiconductor dataset is large (nearly 6M tokens), we
use Microsoft earnings calls from 2013–2024 ( microsoft ) to limit
corpus size.
5.1 Evaluation Criteria
Following the methodology of [ 6], we generated 125 sensemaking
questions using GPT-5-mini (details are in Appendix C.0.1). Con-
sistent with prior work [ 6,13,27], we evaluate answers using two
primary criteria:Comprehensiveness, which measures how thor-
oughly an answer addresses all relevant aspects of a question, and
Diversity, which measures the extent to which an answer captures
varied perspectives and insights. Edge et al .[6] also reportEm-
powerment(ability to support informed judgment) andDirectness
(degree to which an answer directly addresses the question); results
for these metrics are provided in the Appendix (Table 12).
Retrieval approaches and evaluation.We focus onC2and
C3from Section 2, which perform best overall, and evaluate all 𝑘-
core heuristics— RkH,M2hC , and MRC—against them. Higher 𝑘-core
levels typically yield finer-grained communities, improving answer
quality via more detailed summaries (see Appendix, Table 8). Hence,
for all heuristics, we consider the Leaf and immediate parent levels,
labeled LF and L1 (e.g., RkHLF). For evaluation, we use a head-to-
head framework where LLMs compare answers to select a winner,
loser, or tie. To improve reliability, we use multiple evaluators and
perform repeated assessments, with majority voting determining
the final outcome—suitable for global sensemaking tasks without
gold-standard references (see Appendix C.0.2 for details).Configuration.For fair comparison with [ 6], we performed
indexing using a 600-token chunking window with 100-token over-
lap and an 8k-token context window for community summaries,
following their approach. We utilized the public Azure OpenAI end-
point and the OpenAI API for GPT-3.5-turbo ,GPT-4o-mini , and
GPT-5-mini . For the multiple-judge evaluation, GPT-5-mini was
accessed via Azure OpenAI, while the other four LLMs (Gemini
3 Pro Preview, Gemini 2.5 Pro, Qwen3 Next 80B, and DeepSeek
v3.2) were accessed via GCP. All LLM-based evaluations were per-
formed on a GCP virtual machine equipped with 32 GB memory
and 16 vCPUs (Intel Haswell architecture). Detailed prompts for
graph construction, community summaries, and global answer gen-
eration (following [ 6]), along with our code, are available here:
https://github.com/jakir-sust/Kcore-GraphRAG.
6 Results and Analysis
Knowledge cutoff is a key consideration in RAG evaluation [ 24,32],
as models may have already seen the corpus if it predates their
training data. To minimize such contamination, we prioritizepost-
cutoff datawhen selecting both datasets and evaluation models.
Our primary evaluation (Section 6.1) uses GPT-3.5-turbo and
GPT-4o-mini across three post-cutoff datasets: podcast ,news , and
semiconductor . This setup ensures a fair comparison of GraphRAG
configurations under realistic retrieval and sensemaking conditions,
while minimizing the influence of memorized knowledge. To fur-
ther verify that our heuristics generalize to more recent, stronger
models, we conduct a secondary evaluation using GPT-5-mini on
the full datasets, compensating for the limited post-cutoff content
(Section 6.2). We then report the statistical significance of our 𝑘-
core–based heuristics (Section 6.3). Finally, we analyze token effi-
ciency compared to Edge et al. [ 6] and evaluate ourRound-Robin
Token-Constrained Selection(RRTC) heuristic that aims to reduce to-
ken usage while maintaining competitive performance (Section 6.4).
6.1 Results on Post–Cutoff Data
In this section, we evaluate the performance of our heuristics vs.
Leiden-based GraphRAG [ 6] on three datasets under a post–knowledge-
cutoff setting, using GPT-3.5-turbo andGPT-4o-mini , according
to the two metrics defined earlier: comprehensiveness and diversity.
Table 2 summarizes the GPT-3.5-turbo results. We report head-
to-head comparisons of our 𝑘-core based heuristics (in rows) vs.
Leiden-based GraphRAG [ 6] (columns) on post-knowledge-cutoff
content (September 2021), covering 183 documents for semiconductor ,
609 for news , and 36 for podcast . For each of our heuristics, we
consider the leaf (LF) and parent-of-leaf (L1) levels, and for Leiden-
based GraphRAG, we consider C2 (third from top) and C3 (fourth
from top) levels, which consistently perform well (see Section 2 for
details). Results on C0 and C1 are given in the Appendix (Table 10).
Across all datasets and configurations, our 𝑘-core–based heuris-
tics achieve higher win rates than the C2 and C3 configurations
of Edge et al .in approximately70–75%of comparisons. Ties are
generally infrequent (typically below10%) and nearly absent for
semiconductor , indicating decisive preferences. M2hC LF is the
most consistently strong configuration in the evaluation. It
never records a negative net win rate across any dataset, condition,

Core-based Hierarchies for Efficient GraphRAG
Table 2: GPT-3.5-turbo post-cutoff: Head-to-head win rates (%), for comprehensiveness and diversity metrics. C2 and C3 are
Leiden community levels from Edge et al. [6]. LF is leaf-level communities, and L1 is the level immediately above the leaf.
GPT-3.5-turbo podcast news semiconductor
results for C2 C3 C2 C3 C2 C3
Comprehensiveness Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 56 36 8 50 42 8 44 44 12 52 38 10 48 48 4 56 42 2
RkHLF 58 42 0 44 52 4 52 42 6 49 47 4 50 50 0 50 48 2
M2hCL1 58 36 6 52 42 6 56 34 10 50 36 14 58 42 0 48 42 10
M2hCLF 58 40 2 52 44 4 54 38 8 56 38 6 54 44 2 54 42 4
MRCL1 50 44 6 48 48 4 50 42 8 42 46 12 54 40 6 50 46 4
MRCLF 50 50 0 53 43 4 42 48 10 48 48 4 60 40 0 67 29 4
Diversity Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 46 50 4 52 44 4 46 52 2 36 58 6 52 48 0 48 52 0
RkHLF 56 42 2 44 52 4 48 44 8 56 44 0 46 52 2 44 50 6
M2hCL1 64 32 4 56 42 2 44 52 4 44 50 6 54 46 0 50 50 0
M2hCLF 66 34 0 55 44 1 56 42 2 61 37 2 52 48 0 54 36 10
MRCL1 50 50 0 46 54 0 62 38 0 48 44 8 56 40 4 54 44 2
MRCLF 52 48 0 49 51 0 52 48 0 56 44 0 62 38 0 69 28 3
or metric. On comprehensiveness, it maintains steady positive mar-
gins (+8 to +18 net) across all three datasets against both C2 and C3,
with no sharp degradation between conditions. Performance is even
stronger for diversity: it achieves the highest podcast win rate in
the table (66% against C2), delivers strong news results that actu-
ally improve from C2 to C3 (reaching 61/37 in C3), and posts solid
semiconductor gains especially in C3 (+18 net). Dataset-wise, gains
are largest on semiconductor , where MRCLF averages∼64%wins
and peaks at67–69%against C3. For podcast andnews ,M2hC LF
performs best, averaging ∼57%and∼56%, respectively. Across
datasets, our heuristics outperform C3 more strongly than C2 by
roughly3–6%on average.
Averaged across datasets and community levels, leaf-level (LF)
variants consistently outperform their L1 counterparts by5–10per-
centage points, confirming that finer-grained communities lead to
more informative and diverse summaries. This effect is particularly
pronounced on podcast andsemiconductor , where LF configura-
tions frequently exceed55–60%win rates across both metrics. Over-
all,M2hC LFand MRCLFemerge as the strongest heuristics: M2hC LF
shows the most consistent gains for diversity on podcast andnews ,
whileMRCLF dominates comprehensiveness onsemiconductor.
We conduct the same evaluation using GPT-4o-mini , restricting
all datasets to content published after its knowledge cutoff (October
2023). This results in 63 documents for semiconductor , 331 docu-
ments for news , and 13 documents for podcast . Overall, the trends
remain consistent with GPT-3.5-turbo , though performance mar-
gins are narrower and ties more frequent, as expected given that
GPT-4o-mini is a stronger model and the post-cutoff subsets are
smaller. The detailed results are given in the Appendix (Table 11).
6.2 Evaluation on Full Data byGPT-5-mini
In our secondary experiments, we use GPT-5-mini to examine
whether the trends observed with earlier models persist with a
stronger, more recent evaluator. Due to GPT-5-mini ’s later train-
ing window, none of the datasets have sufficient post-cutoff content.
Hence, we evaluate on the full news ,podcast , and microsoft (since
semiconductorwould create an excessively large corpus).
Table 3 reports head-to-head results between Leiden C2 and
our𝑘-core heuristics; C3 results are given in Appendix (Table 9).
While our heuristics continue to show improvements, the high
proportion of ties suggests that prior knowledge reduces the dis-
criminative power of head-to-head evaluation. Across datasets andTable 3: GPT-5-mini full: Head-to-head win rates (%), for com-
prehensiveness and diversity metrics against Leiden C2 level
from Edge et al. [ 6]. LF indicates leaf-level communities, and
L1 indicates the level immediately above the leaf.
GPT-5-mini podcast news microsoft
Comprehen. Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 41 47 12 56 42 2 43 43 14
RkHLF 50 48 2 50 45 5 45 40 15
M2hCL1 40 52 8 40 34 26 40 45 15
M2hCLF 52 42 6 40 40 20 48 50 2
MRCL1 48 50 2 42 38 20 45 48 7
MRCLF 54 44 2 42 36 22 45 35 20
Diversity Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 40 44 16 50 44 6 48 50 2
RkHLF 50 48 2 45 43 12 38 45 17
M2hCL1 37 51 12 40 38 22 45 38 17
M2hCLF 56 36 8 40 37 23 52 46 2
MRCL1 48 50 2 36 40 24 55 42 3
MRCLF 54 44 2 37 37 26 40 43 17
metrics, leaf-level variants generally outperform L1 by 2–6 per-
centage points, with M2hC LF showing the strongest gains for di-
versity on podcast andMRCLF performing best for comprehensive-
ness on microsoft . While M2hC LF still achieves the strongest
overall performance, MRCLF edges out M2hC LF on podcast and
microsoft . Overall win rates remain around45–55%, reflecting
narrower margins than with GPT-3.5-turbo and slightly wider
than with GPT-4o-mini . Several L1 configurations fall below 50%,
indicating that gains are concentrated in leaf-level variants. Never-
theless, the directional consistency with earlier evaluators suggests
that𝑘-core-based heuristics retain an advantage, albeit a modest
one, even under a stronger model whose prior knowledge reduces
the discriminative power of head-to-head evaluation.
6.3 Statistical Analysis
To assess statistical significance, we follow the procedure of [ 6]
and report p-values for pairwise comparisons of each heuristic
against C2/C3 baselines on the comprehensiveness metric using
GPT-3.5-turbo post-cutoff data in Table 4 (details and diversity
results are in Appendix D). Values with 𝑝<0.005are highlighted
in bold. Overall, RkHandM2hC are significant on podcast andnews ,
while M2hC andMRCperform best on semiconductor . Notably, M2hC LF
is the only heuristic significant ( 𝑝<0.005) against both C2 and C3
across all datasets.
Summary.The results above show that the performance gap be-
tween our heuristics and Leiden-based C2/C3 is systematic rather

Jakir Hossain and Ahmet Erdem Sarıyüce
than incidental. Knowledge graphs exhibit extremely sparse graph
structures; e.g., average degrees in our datasets range from 2.88
to 4.42 and 55-60% of nodes have a degree of just 1. This high
prevalence of degree-1 nodes means that the majority of nodes
have only a single connection, providing insufficient local structure
for Leiden’s modularity optimization to form meaningful commu-
nities, as the algorithm relies on dense local neighborhoods to
effectively aggregate nodes into clusters. As Theorem 1 shows,
Leiden’s degeneracy on such sparse graphs makes it unlikely to
find the “right” partition, as there are exponentially many near-
equivalent solutions, most of which group peripheral nodes incoher-
ently. In contrast, 𝑘-core–based heuristics produce a deterministic
hierarchy that is structurally grounded, explaining the consistent
advantage of our methods across GPT-3.5-turbo ,GPT-4o-mini ,
andGPT-5-mini evaluations. Note that nearly 45% of clusters from
RkHhave size 2, so M2hC andMRChelp improve performance by merg-
ing these, reducing fragmentation. Clusters of size 3 or 4 account for
only 10–12%, and merging them offers little benefit while inflating
neighboring clusters, so they were excluded from our small-cluster
definition. Under GPT-3.5-turbo , these differences are statisti-
cally significant across all three datasets ( 𝑝< 0.005for M2hC LF),
and directionally consistent under GPT-4o-mini andGPT-5-mini ,
where stronger prior knowledge narrows the margins but does not
eliminate the advantage.
6.4 Token Utilization and Impact of RRTC
Finally, we analyze how the evaluated methods consume tokens,
and how our round-robin token-constrained selection (RRTC) mech-
anism (see Section 4.3) performs. We first analyze the number of
communities produced by each heuristic, along with the percent-
age of source tokens contained in those communities. These are
the communities and tokens that are used to answer queries via
LLM calls: fewer communities reduce the number of LLM calls, and
lower token counts reduce overall LLM token usage. Results on
GPT-3.5-turbo post-cutoff data are shown in Table 5 for leaf-level,
which typically contain more communities and use more tokens
than L1. Notably, MRCreduces both the number of communities and
the tokens used: it consistently produces the fewest communities
and lowest coverage (55–60%), meaning it aggressively consolidates
information. These reductions indicate that our heuristics use fewer
tokens and require fewer LLM calls, while achieving comparable
or improved performance on global sensemaking, as shown in the
previous section.
Impact of Round-Robin Token-Constrained Selection (RRTC).
Table 6 reports results for M2hC LF with RRTC across three datasets,
Table 4: Wilcoxon signed-rank test p-values for comprehen-
siveness ( GPT-3.5-turbo ) across datasets and baselines. Val-
ues with𝑝<0.005are highlighted in bold.
Cond. podcast news semiconductor
C2 C3 C2 C3 C2 C3
RkHL1<0.001 <0.001 1.000<0.001 1.000<0.001
RkHLF<0.001 <0.001 <0.0010.370 0.982 0.363
M2hC L1<0.001 <0.001 <0.001 <0.001 <0.0010.006
M2hC LF<0.001 <0.001 <0.001 <0.001 <0.001 <0.001
MRCL1 0.0071.000 <0.0010.069 <0.0010.073
MRCLF 0.982<0.001 0.007 1.000 <0.001 <0.001Table 5: Community statistics across datasets, showing the
number of communities and the percentage of source text
used at different hierarchy levels and methods.
Dataset # Tokens Metric C2 C3RkHLFM2hCLFMRCLF
podcast180,267 # Comm. 291 301 351 311 194
Coverage (%) 68.93 70.87 83.70 75.47 58.08
news1,340,078 # Comm. 2,019 2,021 2,205 1,863 1,357
Coverage (%) 65.90 71.26 76.46 65.74 55.08
semic.438,819 # Comm. 627 696 847 705 468
Coverage (%) 66.81 72.67 90.18 76.45 59.94
compared against Leiden C2/C3 under three edge budgets: 80%,
70%, and 60% of the original graph. The table reports token us-
age relative to C2/C3 and head-to-head win rates against them for
comprehensiveness and diversity, where win rates are computed
against losses only (ties excluded); thus, a 55% win rate implies a
45% loss rate.
Across datasets, RRTC maintains competitive performance even
when selecting only 60% of graph edges. Comprehensiveness win
rates remain in the 50–56% range, with the highest values on
semiconductor at the 80% budget, reaching 56% and 58% against C2
and C3, respectively. Diversity is similarly preserved, often exceed-
ing 55% on podcast andsemiconductor despite substantial edge
reduction. As the edge budget decreases from 80% to 60%, win rates
gradually decline, with comprehensiveness on podcast dropping
below 50%, indicating that a 60% budget may be too aggressive for
some dataset–metric combinations. Overall, RRTC reduces token
usage by up to∼40%relative to Leiden C2/C3 while maintaining
competitive comprehensiveness and diversity, providing a practical,
token-efficient alternative under moderate budget constraints.
Table 6: RRTC ( M2hC Leaf) vs. Leiden C2/C3: Head-to-head
win rates (%) on comprehensiveness and diversity
Dataset Edge Relative token Comp. Div.
budget usage (%) win rate (%) win rate (%)
(%) C2 C3 C2 C3 C2 C3
podcast80 90 88 57 55 64 55
70 79 77 54 52 56 52
60 68 66 47 46 51 49
news80 81 75 54 54 55 58
70 71 65 49 54 52 56
60 60 56 52 50 46 50
semic.80 92 85 56 58 55 60
70 81 74 53 53 50 57
60 69 64 55 50 52 54
7 Conclusion
We proved that modularity optimization is inherently unreliable
on the sparse graphs typical of GraphRAG knowledge graphs, and
proposed𝑘-core–based heuristics that yield deterministic, size-
bounded hierarchies in linear time. Across three datasets, three
generator LLMs, and five LLM judges, our approach consistently
improves comprehensiveness and diversity while reducing token
usage compared to Leiden-based GraphRAG. Future work includes
extending the framework to dynamic knowledge graphs and ex-
ploring its applicability beyond global sensemaking tasks.

Core-based Hierarchies for Efficient GraphRAG
References
[1] Vladimir Batagelj and Matjaž Zaversnik. 2011. Fast algorithms for determining
(generalized) core groups in social networks.Advances in Data Analysis and
Classification5, 2 (2011), 129–145.
[2] Tal Baumel, Matan Eyal, and Michael Elhadad. 2018. Query focused abstractive
summarization: Incorporating query relevance, multi-document coverage, and
summary length constraints into seq2seq models.arXiv preprint arXiv:1801.07704
(2018).
[3] Michele Ca’ Zorzi, Gianluigi Lopardo, and Ana-Simona Manu. 2025. Verba Volant,
Transcripta Manent: What Corporate Earnings Calls Reveal About the AI Stock
Rally. 3093 (2025). https://glopardo.com/corporatetalks/
[4] Shai Carmi, Shlomo Havlin, Scott Kirkpatrick, Yuval Shavitt, and Eran Shir. 2007.
A model of Internet topology using k-shell decomposition.Proceedings of the
National Academy of Sciences104, 27 (2007), 11150–11154.
[5] J. Cohen. 2008. Trusses: Cohesive subgraphs for social network analysis. National
Security Agency Technical Report (2008).
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[8] Abdellah Ghassel, Ian Robinson, Gabriel Tanase, Hal Cooper, Bryan Thompson,
Zhen Han, Vassilis Ioannidis, Soji Adeshina, and Huzefa Rangwala. 2025. Hi-
erarchical Lexical Graph for Enhanced Multi-Hop Retrieval. InProceedings of
the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2
(Toronto ON, Canada)(KDD ’25). Association for Computing Machinery, New
York, NY, USA, 4457–4466. https://doi.org/10.1145/3711896.3737233
[9] Benjamin H Good, Yves-Alexandre De Montjoye, and Aaron Clauset. 2010. Per-
formance of modularity maximization in practical contexts.Physical Review
E—Statistical, Nonlinear, and Soft Matter Physics81, 4 (2010), 046106.
[10] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi
He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and
Jiliang Tang. 2025. Retrieval-Augmented Generation with Graphs (GraphRAG).
arXiv:2501.00309 [cs.IR] https://arxiv.org/abs/2501.00309
[11] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering.Advances in
Neural Information Processing Systems37 (2024), 132876–132907.
[12] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao.
2025. GRAG: Graph Retrieval-Augmented Generation. arXiv:2405.16506 [cs.LG]
https://arxiv.org/abs/2405.16506
[13] Boyoung Kim, Dosung Lee, Sumin An, Jinseong Jeong, and Paul Hongsuck
Seo. 2025. ReTAG: Retrieval-Enhanced, Topic-Augmented Graph-Based Global
Sensemaking. InFindings of the Association for Computational Linguistics: EMNLP
2025. 22249–22277.
[14] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. Kg-gpt: A general
framework for reasoning on knowledge graphs using large language models.
arXiv preprint arXiv:2310.11220(2023).
[15] Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Dmitry Sorokin, Artyom Sorokin,
and Mikhail Burtsev. 2024. In search of needles in a 11m haystack: Recurrent
memory finds what llms miss.arXiv preprint arXiv:2402.10790(2024).
[16] Md Tahmid Rahman Laskar, Enamul Hoque, and Jimmy Huang. 2020. Query fo-
cused abstractive summarization via incorporating query relevance and transfer
learning with transformer models. InCanadian conference on artificial intelligence.
Springer, 342–348.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[18] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts.Transactions of the association for computational linguistics12
(2024), 157–173.
[19] Fatma Miladi, Valéry Psyché, and Daniel Lemire. 2024. Leveraging gpt-4 for ac-
curacy in education: A comparative study on retrieval-augmented generation in
moocs. InInternational Conference on Artificial Intelligence in Education. Springer,
427–434.
[20] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey.
ACM Transactions on Information Systems44, 2 (2025), 1–52.
[21] Ahmet Erdem Sariyüce and Ali Pinar. 2016. Fast hierarchy construction for dense
subgraphs. 10, 3 (Nov. 2016), 97–108.[22] Kevin Scott. 2024. Behind the Tech. https://www.microsoft.com/en-us/behind-
the-tech. Accessed: 2026-01-01.
[23] Stephen B Seidman. 1983. Network structure and minimum degree.Social
networks5, 3 (1983), 269–287.
[24] Minjae Seo, Wonwoo Choi, Myoungsung You, and Seungwon Shin. 2025. Au-
toPatch: Multi-Agent Framework for Patching Real-World CVE Vulnerabilities.
arXiv preprint arXiv:2505.04195(2025).
[25] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M Ni, Heung-Yeung Shum, and Jian Guo. 2023. Think-on-graph:
Deep and responsible reasoning of large language model on knowledge graph.
arXiv preprint arXiv:2307.07697(2023).
[26] Yixuan Tang and Yi Yang. 2024. Multihop-rag: Benchmarking retrieval-
augmented generation for multi-hop queries.arXiv preprint arXiv:2401.15391
(2024).
[27] Yash Tiwari, Owais Ahmad Lone, and Mayukha Pal. 2025. OntoRAG: Enhancing
Question-Answering through Automated Ontology Derivation from Unstruc-
tured Knowledge Bases.arXiv preprint arXiv:2506.00664(2025).
[28] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to
Leiden: guaranteeing well-connected communities.Scientific reports9, 1 (2019),
1–12.
[29] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr.
2024. Knowledge graph prompting for multi-document question answering. In
Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 19206–19214.
[30] Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stew-
art Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, and Bruno
Fleisch. 2024. CBR-RAG: case-based reasoning for retrieval augmented gen-
eration in LLMs for legal question answering. InInternational Conference on
Case-Based Reasoning. Springer, 445–460.
[31] Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Bowen Jin, May Dongmei Wang,
Joyce Ho, and Carl Yang. 2024. Ram-ehr: Retrieval augmentation meets clini-
cal predictions on electronic health records. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 2: Short Papers).
754–765.
[32] Zhuoyi Yang, Yurun Song, Iftekhar Ahmed, and Ian Harris. 2026. Fine-Tuning vs.
RAG for Multi-Hop Question Answering with Novel Knowledge.arXiv preprint
arXiv:2601.07054(2026).
[33] Jin-ge Yao, Xiaojun Wan, and Jianguo Xiao. 2017. Recent advances in document
summarization.Knowledge and Information Systems53, 2 (2017), 297–336.
[34] Boyu Zhang, Hongyang Yang, Tianyu Zhou, Muhammad Ali Babar, and Xiao-
Yang Liu. 2023. Enhancing financial sentiment analysis via retrieval augmented
large language models. InProceedings of the fourth ACM international conference
on AI in finance. 349–356.
[35] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou,
Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al .2023. A survey
of large language models.arXiv preprint arXiv:2303.182231, 2 (2023).

Jakir Hossain and Ahmet Erdem Sarıyüce
A Proof of Theorem 1
Proof sketch.
Step 1: Low-degree nodes are weakly coupled.A node with
degree𝑘𝑖≤𝑑participates in at most 𝑑edges. Moving it between
communities changes modularity by at most 𝑂(𝑑/𝑚) : the adjacency
term shifts by at most 𝑑/𝑚, and the null-model penalty shifts by
𝑂(𝑘𝑖·𝐾𝑟/(2𝑚)2)where𝐾𝑟is the total degree of the destination
community. In the sparse regime where 𝑚=Θ(𝑛) , this is𝑂(1/𝑛),
so the node’s community assignment barely affects the objective.
Step 2: Weakly coupled nodes can be independently reas-
signed.Reassigning one low-degree node perturbs another’s sensi-
tivity by only 𝑂(1/𝑛2). By selecting a large independent set among
the low-degree nodes, which is straightforward since these nodes
have bounded degree, we obtain Θ(𝑛) nodes whose reassignments
are approximately independent. Each can be placed in at least two
communities without meaningful modularity loss, yielding2Θ(𝑛)
distinct near-optimal partitions.
Step 3: Sparsity makes this severe.When ¯𝑘< 3, standard re-
sults on Poisson and power-law degree distributions guarantee that
𝑛≤d=Θ(𝑛) . This is the regime of knowledge graphs in GraphRAG.
In contrast, for dense graphs with large ¯𝑘,𝑛≤d/𝑛→ 0and most
nodes are strongly coupled to their communities, collapsing the
degeneracy.
Full proof.
Proof. We first remind modularity in its standard community-
sum form:
𝑄(𝜎)=∑︁
𝑐"
𝑒𝑐
𝑚−𝐾𝑐
2𝑚2#
,
where𝑒𝑐counts edges internal to community 𝑐and𝐾𝑐=Í
𝑗∈𝑐𝑘𝑗
is its total degree. Moving node 𝑖from community 𝑠to community
𝑟produces a new partition 𝜎𝑖→𝑟. We write𝑑(𝑐)
𝑖for the number
of neighbors of 𝑖in community 𝑐. We also define themodularity
sensitivityof node 𝑖under partition 𝜎isΔ𝑖(𝜎)=max 𝑟≠𝜎𝑖|𝑄(𝜎)−
𝑄(𝜎𝑖→𝑟)|, where𝜎𝑖→𝑟moves𝑖to community 𝑟with all other as-
signments fixed. Node𝑖is𝜀-weakly coupledifΔ 𝑖(𝜎)<𝜀.
Next, we give two lemmas that will help with the proof
Lemma 2 (Moving a low-degree node).For any partition 𝜎
and any node 𝑖with degree𝑘𝑖≤𝑑, moving𝑖to any other community
changes modularity by at most
Δ𝑖(𝜎) ≤2𝑘𝑖
𝑚+𝑘2
𝑖
2𝑚2=𝑂1
𝑛
.
Proof. Moving𝑖from community 𝑠to𝑟affects only those two
terms in the sum. There are two contributions:
Edge term.Community 𝑠loses𝑑(𝑠)
𝑖internal edges; 𝑟gains𝑑(𝑟)
𝑖. Both
are at most𝑘 𝑖, so
|Δedge|=|𝑑(𝑟)
𝑖−𝑑(𝑠)
𝑖|
𝑚≤𝑘𝑖
𝑚.
Degree-penalty term.The total degree of 𝑠drops by𝑘𝑖and that of𝑟
rises by𝑘 𝑖. Expanding the squares and simplifying:
|Δpenalty|=|𝑘𝑖[2(𝐾𝑟−𝐾𝑠)+2𝑘𝑖]|
(2𝑚)2≤𝑘𝑖
𝑚+𝑘2
𝑖
2𝑚2,using|𝐾𝑟−𝐾𝑠|≤2𝑚.
Adding the two and substituting 𝑘𝑖≤𝑑and𝑚=𝑛 ¯𝑘/2=Θ(𝑛) gives
Δ𝑖≤2𝑑/𝑚+𝑑2/(2𝑚2). Both terms are𝑂(1/𝑛).□
Lemma 3 (Non-adjacent low-degree nodes).Let 𝑖,𝑗be non-
adjacent nodes with degrees ≤𝑑. Reassigning 𝑗changes node 𝑖’s
sensitivity by at most
|Δ𝑖(𝜎)−Δ𝑖(𝜎′)| ≤2𝑑2
(2𝑚)2=𝑂1
𝑛2
.
Proof. Since𝑖and𝑗share no edge, reassigning 𝑗does not
change𝑖’s neighbor counts 𝑑(𝑐)
𝑖in any community, so the edge
term for𝑖is unaffected. The only effect is that 𝑗’s move shifts the
total degrees 𝐾𝑟,𝐾𝑠by±𝑘𝑗≤𝑑, which perturbs 𝑖’s penalty term by
at most𝑘𝑖·2𝑑/(2𝑚)2≤2𝑑2/(2𝑚)2=𝑂(1/𝑛2).□
Now, we give the proof of Theorem 1. The argument has three
parts.
1. Find a large independent set of weakly coupled nodes.By
Lemma 2, every node with 𝑘𝑖≤𝑑has sensitivity 𝑂(1/𝑛), so for
𝜀>𝑑( 2+¯𝑘)/(2𝑚)it is𝜀-weakly coupled. Let 𝑛≤𝑑denote the number
of such nodes. The subgraph they induce has maximum degree 𝑑,
so a greedy algorithm yields an independent set𝑆of size
|𝑆| ≥𝑛≤𝑑
𝑑+1.
2. Every combination of reassignments stays near-optimal.
Start from an optimal partition 𝜎∗. Each node 𝑖∈𝑆 can stay or
move to an alternative community: two choices per node. For any
subset𝑇⊆𝑆 of nodes that are reassigned, the total modularity loss
is bounded by summing the individual sensitivities plus pairwise
cross-terms (using Lemma 3):
|𝑄∗−𝑄(𝜎𝑇)| ≤∑︁
𝑖∈𝑇Δ𝑖(𝜎∗)
|      {z      }
individual+∑︁
{𝑖,𝑗}⊆𝑇2𝑑2
(2𝑚)2
|          {z          }
cross-terms.
Bounding each sum with|𝑇|≤|𝑆|≤𝑛 ≤𝑑/(𝑑+1)and𝑚=𝑛 ¯𝑘/2:
Individual sum≤𝑛≤𝑑
𝑑+1·𝑑(2+ ¯𝑘)
𝑛¯𝑘≤𝑑(2+ ¯𝑘)
(𝑑+1) ¯𝑘=:𝐶1,
Cross-term sum≤1
2𝑛≤𝑑
𝑑+12
·2𝑑2
𝑛2¯𝑘2≤𝑑2
(𝑑+1)2¯𝑘2=:𝐶2.
Both𝐶1and𝐶2are constants (independent of 𝑛), so setting 𝜀′=
𝐶1+𝐶 2keeps every such partition within𝜀′of𝑄∗.
3. Count and conclude.Each of the |𝑆|nodes has two valid as-
signments, giving2|𝑆|structurally distinct near-optimal partitions:
D(𝜀′) ≥2|𝑆|≥2𝑛≤𝑑/(𝑑+1).
In the sparse regime ( ¯𝑘< 3), standard degree distributions guar-
antee𝑛≤𝑑=Θ(𝑛) . For example, under a Poisson distribution with
¯𝑘=2and𝑑=2, the fraction of low-degree nodes is5 𝑒−2≈0.68;
under a power law with exponent 𝛾=2.5, degree-1 nodes alone
comprise∼74%of the graph. ThereforeD(𝜀′)≥2Θ(𝑛).□

Core-based Hierarchies for Efficient GraphRAG
B Splitting Oversized Components
B.1 SPLIT: Splitting Large Connected
Components
When a connected component 𝑅exceeds the maximum cluster size
𝑀, it is split into smaller clusters using theSplitprocedure. The
algorithm (Algorithm 3) grows each cluster greedily from a seed
node with the highest degree (line 3), adding neighboring nodes
that maximize internal connectivity (line 6) until the cluster reaches
size𝑀. Each completed cluster is added to the cluster set (line 10)
and removed from the remaining node set (line 11). The output is a
set of size-constrained clusters that preserve internal connectivity
within𝐶.
Algorithm 3:Split(G, R, M)
Require:Graph𝐺=(𝑉,𝐸), connected comp.𝑅, max size𝑀
Ensure:List of split clustersC
1:Initialize cluster setC←∅
2:while𝑅≠∅do
3:Pick seed node𝑠∈𝑅with highest degree
4:𝑆←{𝑠},𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟←𝑁(𝑠)∩𝑅
5:while|𝑆|<𝑀and𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟≠∅do
6:Pick𝑣∈𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟maximizing𝑁(𝑣)∩𝑆
7:𝑆←𝑆∪𝑣
8:𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟←𝑁(𝑣)∩(𝑅\𝑆)
9:end while
10:C←C∪𝑆
11:Remove𝑆from𝑅
12:end while
13:returnC
B.2 SPLIT-2HOP: Splitting Large 2-Hop
Connected Components
When a 2-hop connected component 𝐻exceeds the maximum
cluster size𝑀, it is partitioned into smaller clusters using theSplit-
2HOPprocedure (Algorithm 4). The algorithm first computes the
anchor set as the union of neighbors of all 2-hop nodes and precom-
putes anchor neighborhoods for each node (lines 2–3). Clusters are
grown greedily by selecting a seed node with the largest anchor set
(line 5) and initializing a cluster with this seed (line 6). Candidate
nodes that share at least one anchor with the seed are added to the
frontier (line 7). The cluster is expanded by repeatedly selecting
the frontier node with the highest overlap in anchor connectivity
with the current cluster (lines 8–10), until the cluster reaches size
𝑀or no eligible nodes remain. After cluster growth, anchor nodes
connected to at least two nodes in the cluster are selected and in-
cluded (line 12), and the resulting cluster is added to the output set
(line 13). The assigned nodes are then removed from the remaining
2-hop groups set (line 14). This process repeats until all nodes in
the 2-hop groups are assigned.
C Evaluation Criteria Details
C.0.1 Question Generation.Following methodology described in
[6], we generated 125 sensemaking questions using GPT-5-mini .
Consistent with their approach, we first prompted GPT-5-mini toAlgorithm 4:Split-2hop(G, H, M)
Require:Graph𝐺=(𝑉,𝐸), 2-hop groups𝐻, max size𝑀
Ensure:List of split clustersC
1:Initialize cluster setC←∅
2:Compute anchors𝐴←Ð
𝑢∈𝐻𝑁(𝑢)
3:Precompute𝐴 𝑢←𝑁(𝑢)∩𝐴for𝑢∈𝐻
4:while𝐻≠∅do
5:Pick𝑠𝑒𝑒𝑑∈𝐻with max|𝐴 𝑠𝑒𝑒𝑑|
6:𝑆←{𝑠𝑒𝑒𝑑}
7:𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟←{𝑣∈𝐻\{𝑠𝑒𝑒𝑑}|𝐴 𝑣∩𝐴𝑠𝑒𝑒𝑑≠∅}
8:while|𝑆|<𝑀and𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟≠∅do
9:Pick𝑣∈𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟maximizingÍ
𝑢∈𝑆|𝐴𝑣∩𝐴𝑢|
10:𝑆←𝑆∪{𝑣};
𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟←(𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟∪{𝑢∈𝐻|𝐴 𝑢∩𝐴𝑣≠∅})\𝑆
11:end while
// Select anchors linked to≥2nodes in𝑆
12:𝐴′←{𝑣∈𝐴𝑠.𝑡.𝑁(𝑣)∩𝑆≥2}
13:C←C∪{𝑆∪𝐴′}
14:𝐻←𝐻\𝑆
15:end while
16:returnC
create personas of hypothetical users for each corpus and then gen-
erated 5 tasks per user. For each user-task pair, we use GPT-5-mini to
generate high-level questions that require understanding of the en-
tire corpus without relying on low-level fact retrieval. This process
ensures that the resulting questions assess comprehensive, corpus-
wide reasoning, aligned with prior work.
C.0.2 Evaluation Approach.We evaluate models using a head-to-
head framework, where LLMs compare pairs of answers to identify
a winner, loser, or tie. To enhance reliability, multiple evaluators are
used, and repeated assessments are performed. The final outcome is
determined via majority voting, making this approach suitable for
global sensemaking tasks without gold-standard references. Each
comparison proceeds as follows:
(1)Generate answers to the same query using two approaches.
(2)Randomize answer order and present the pair to the evalu-
ator.
(3)Each of five independent LLM evaluators ( GPT-5-mini ,
Gemini 3 Pro Preview, Gemini 2.5 Pro, Qwen3 Next 80B,
and DeepSeek v3.2) assesses the same answer pair three
times.
(4)Apply majority voting within each evaluator to determine
their decision.
(5)Apply a second majority vote across evaluators to deter-
mine the final judgment.
This procedure ensures robust, repeatable head-to-head compar-
isons across methods.
D Statistical Analysis (p-values)
To assess the significance of observed differences, we follow a pro-
cedure similar to Edge et al. [ 6]. For each dataset, evaluator, and
metric, we assign scores for each head-to-head comparison: the
winning method receives a score of 100, the losing method receives
0, and in the event of a tie, both methods receive 50. These scores

Jakir Hossain and Ahmet Erdem Sarıyüce
are then averaged over all evaluators, repeated runs, and questions,
as detailed in Section 5.1. Following [ 6], we use non-parametric
Wilcoxon signed-rank tests to evaluate pairwise performance dif-
ferences between methods. Holm-Bonferroni adjustment is applied
to correct for multiple comparisons.
Table 7: Wilcoxon signed-rank test p-values for comparisons
across datasets and baselines on Diversity ( GPT-3.5-turbo ).
Significant values are highlighted in bold for𝑝<0.005.
Conditionpodcast news semiconductor
C2 C3 C2 C3 C2 C3
RkHL1 0.073<0.001 0.008<0.001 0.083 0.083
RkHLF<0.001 <0.001 0.071<0.001 0.008 0.007
M2hCL1<0.001 <0.001 <0.0010.007 <0.0010.982
M2hCLF<0.001 <0.001 <0.001 <0.001 0.083<0.001
MRCL1 0.982<0.001 <0.0010.071 <0.001 <0.001
MRCLF 0.083 0.393 0.083<0.001 <0.001 <0.001
C3<0.001 <0.001 <0.001 <0.001 <0.001 <0.001
D.1 Head-to-Head Comparison on Leiden C2
and DifferentM2hCLevels
Table 8 shows the head-to-head win rates (%) for GPT-3.5-turbo on
microsoft data, comparing the Leiden C2 baseline against various
M2hC level variants. The two subtables report metrics for (a) Com-
prehensiveness and (b) Diversity.
Here, the M2hC variants correspond to different k-core levels
within the graph hierarchy: LF represents the leaf-level communi-
ties, LF-2 corresponds to two levels above the leaf, LF-4 to four levels
above the leaf, and LF-6 to six levels above the leaf. We observe that
higher k-core levels consistently achieve better win rates against
C2, particularly in Comprehensiveness, indicating that the more
central nodes capture global context more effectively. This justifies
focusing on leaf-level (LF) and the level immediately above leaf (L1)
for all of our evaluations, as they provide better performance.
Table 8: GPT-3.5-turbo Head-to-Head Win Rates (%) using
Leiden C2 andM2hCLevel Variants onmicrosoftdata
(a) Comprehensiveness
Leiden C2 M2hCLF-6 M2hCLF-4 M2hCLF-2 M2hCLF
Leiden C2 50 56 51 48 39
M2hCLF-6 44 50 45 43 33
M2hCLF-4 49 55 50 46 40
M2hCLF-2 52 57 54 50 47
M2hCLF 61 67 60 53 50
(b) Diversity
Leiden C2 M2hCLF-6 M2hCLF-4 M2hCLF-2 M2hCLF
Leiden C2 50 62 54 46 39
M2hCLF-6 38 50 42 38 36
M2hCLF-4 46 58 50 49 41
M2hCLF-2 54 62 51 50 51
M2hCLF 61 64 59 49 50
E Additional Results
Here, we present additional results from GPT-3.5-turbo ,GPT-4o-mini ,
andGPT-5-mini , including the results from Leiden C0, C1, and
other metrics such as Empowerment and Directness.E.1GPT-4o-miniResults on Post-cutoff Data
Table 11 reports head-to-head win rates for comprehensiveness and
diversity under the same experimental conditions as in GPT-3.5-turbo .
Across datasets and configurations, our heuristics still achieve
higher win rates than Leiden C2/C3 in a majority of comparisons,
averaging approximately45–55%, particularly on semiconductor .
Among our methods, M2hC LF still achieves the strongest overall
performance, with average win rates of approximately50–52%
across datasets and metrics, followed by RkHLF and MRCLF at48–
50%. Dataset-wise, gains are most pronounced on semiconductor ,
where M2hC LF reaches up to64%wins for diversity (C2) and main-
tains58%against C3, while comprehensiveness peaks at52%. For
podcast andnews , win rates remain closer to parity, typically fluc-
tuating between45–50%across heuristics and Leiden levels.
Comparing Leiden resolutions, differences between C2 and C3
are smaller than for GPT-3.5-turbo , with average win-rate gaps
generally within2–4%. Averaged across datasets and metrics, leaf-
level (LF) variants continue to outperform L1, but by a narrower
margin of2–5percentage points. Overall, GPT-4o-mini narrows
the gap between 𝑘-core heuristics and Leiden GraphRAG, with
M2hC LFremaining the most robust, especially on semiconductor ,
and generally matching or slightly surpassing Leiden baselines
despite more ties.
Note that performance on the podcast is less reliable due to the
limited number of post-cutoff documents. With only 13 documents
available, the resulting graphs are sparse, and our heuristics of-
ten show minimal improvement, performing similarly to C2/C3
configurations.
Table 9: GPT-5-mini full: Head-to-head win rates (%), for com-
prehensiveness and diversity metrics. C3 is the Leiden com-
munity level from Edge et al. [ 6]. LF indicates leaf-level com-
munities, and L1 indicates the level immediately above the
leaf.
GPT-5-mini podcast news microsoft
results for C3 C3 C3
Comprehensiveness Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 41 41 18 52 39 9 38 43 19
RkHLF 43 43 14 47 45 8 42 48 10
M2hCL1 48 48 4 49 37 14 48 42 10
M2hCLF 50 47 3 47 35 18 47 48 5
MRCL1 45 45 10 32 48 20 45 40 15
MRCLF 50 37 13 36 46 18 45 40 15
Diversity Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 44 44 12 47 38 15 33 53 14
RkHLF 48 45 7 37 34 29 48 44 8
M2hCL1 40 42 18 42 38 20 45 48 7
M2hCLF 48 41 11 32 33 35 52 40 8
MRCL1 47 46 7 45 29 24 53 35 12
MRCLF 56 31 13 34 30 36 54 38 8

Core-based Hierarchies for Efficient GraphRAG
Table 10: GPT-3.5-turbo post-cutoff: Head-to-head win rates (%), for comprehensiveness and diversity metrics. C0 and C1 are
Leiden community levels from Edge et al. [ 6]. LF indicates leaf-level communities, and L1 indicates the level immediately
above the leaf.
GPT-3.5-turbo podcast news semiconductor
results for C0 C1 C0 C1 C0 C1
Comprehensiveness Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 50 42 8 68 24 8 56 32 12 48 42 10 66 30 4 72 26 2
RkHLF 58 42 0 52 44 4 40 50 10 42 44 14 68 32 0 52 46 2
M2hCL1 48 46 6 66 28 6 50 40 10 50 36 14 54 46 0 62 28 10
M2hCLF 58 40 2 68 28 4 44 48 8 44 49 7 60 38 2 72 24 4
MRCL1 58 36 6 48 48 4 50 38 12 46 44 10 54 40 6 64 32 4
MRCLF 64 36 0 57 39 4 56 34 10 61 35 4 48 42 10 55 43 2
Diversity Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 68 28 4 60 36 4 60 38 2 56 38 6 62 38 0 66 34 0
RkHLF 52 46 2 56 40 4 48 48 4 50 44 6 62 36 2 60 34 6
M2hCL1 42 54 4 70 28 2 56 40 4 60 34 6 64 36 0 70 30 0
M2hCLF 52 48 0 72 27 1 60 38 2 67 32 1 62 38 0 52 38 10
MRCL1 66 34 0 60 40 0 50 48 2 42 52 6 52 44 4 52 46 2
MRCLF 60 40 0 65 35 0 64 36 0 64 36 0 60 40 0 56 44 0
Table 11: GPT-4o-mini post-cutoff : Head-to-head win rates (%), for comprehensiveness and diversity metrics. C2 and C3 are
Leiden community levels from Edge et al. [ 6]. LF indicates leaf-level communities, and L1 indicates the level immediately
above the leaf.
GPT-4o-mini podcast news semiconductor
results for C2 C3 C2 C3 C2 C3
Comprehensiveness Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 50 50 0 50 50 0 38 42 20 40 42 18 46 46 8 39 36 25
RkHLF 50 48 2 46 48 6 40 40 20 43 40 17 52 41 7 49 32 19
M2hCL1 48 46 6 48 44 8 42 42 16 44 44 12 44 40 16 51 30 19
M2hCLF 50 48 2 50 46 4 44 40 16 46 42 12 52 38 10 50 24 26
MRCL1 48 44 8 44 42 14 38 36 26 38 32 30 36 42 22 46 44 10
MRCLF 48 46 6 46 48 6 40 35 25 40 30 30 48 42 10 44 44 12
Diversity Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 50 48 2 46 48 6 47 40 13 52 39 9 38 54 8 45 48 7
RkHLF 50 50 0 48 50 2 50 38 12 55 37 8 47 49 4 52 36 12
M2hCL1 50 50 0 50 50 0 42 48 10 44 46 10 50 38 12 49 36 15
M2hCLF 49 46 5 46 48 6 44 46 10 46 44 10 64 32 4 58 34 8
MRCL1 48 48 4 48 44 8 43 34 23 46 38 16 46 46 8 45 45 10
MRCLF 49 50 1 48 52 0 45 32 23 48 36 16 54 44 2 54 45 1
Table 12: GPT-3.5-turbo post-cutoff: Head-to-head win rates (%), for empowerment and directness metrics. C2/C3 are Leiden
community levels, LF is leaf-level communities, and L1 is the level immediately above the leaf
GPT-3.5-turbo podcast news semiconductor
results for C2 C3 C2 C3 C2 C3
Comprehensiveness Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 48 50 2 52 45 3 46 52 2 63 37 0 52 48 0 47 52 1
RkHLF 52 43 5 54 43 3 46 50 4 57 40 3 50 50 0 50 50 0
M2hCL1 54 44 2 59 39 2 63 35 2 55 45 0 40 59 1 53 43 4
M2hCLF 50 46 4 54 46 0 46 51 3 46 54 0 61 38 1 52 44 4
MRCL1 39 54 7 40 55 5 38 60 2 38 60 2 51 48 2 57 37 6
MRCLF 36 56 8 47 50 3 50 40 10 65 30 5 53 46 1 59 36 5
Directness Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie Win Loss Tie
RkHL1 62 36 2 40 50 10 37 63 0 45 45 10 35 65 0 56 43 1
RkHLF 42 55 3 35 55 10 40 57 3 48 42 10 35 65 0 58 41 1
M2hCL1 54 44 2 64 28 8 56 44 0 56 31 13 63 36 1 53 43 4
M2hCLF 52 44 4 62 31 7 46 54 0 58 31 11 61 38 1 53 44 3
MRCL1 43 50 7 50 44 6 48 50 2 59 28 13 50 50 0 47 53 0
MRCLF 58 36 6 47 50 3 48 52 0 61 28 11 50 50 0 45 55 0