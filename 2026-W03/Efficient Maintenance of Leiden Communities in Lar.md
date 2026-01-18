# Efficient Maintenance of Leiden Communities in Large Dynamic Graphs

**Authors**: Chunxu Lin, Yumao Xie, Yixiang Fang, Yongmin Hu, Yingqian Hu, Chen Cheng

**Published**: 2026-01-13 13:39:22

**PDF URL**: [https://arxiv.org/pdf/2601.08554v2](https://arxiv.org/pdf/2601.08554v2)

## Abstract
As a well-known community detection algorithm, Leiden has been widely used in various scenarios such as large language model generation (e.g., Graph-RAG), anomaly detection, and biological analysis. In these scenarios, the graphs are often large and dynamic, where vertices and edges are inserted and deleted frequently, so it is costly to obtain the updated communities by Leiden from scratch when the graph has changed. Recently, one work has attempted to study how to maintain Leiden communities in the dynamic graph, but it lacks a detailed theoretical analysis, and its algorithms are inefficient for large graphs. To address these issues, in this paper, we first theoretically show that the existing algorithms are relatively unbounded via the boundedness analysis (a powerful tool for analyzing incremental algorithms on dynamic graphs), and also analyze the memberships of vertices in communities when the graph changes. Based on theoretical analysis, we develop a novel efficient maintenance algorithm, called Hierarchical Incremental Tree Leiden (HIT-Leiden), which effectively reduces the range of affected vertices by maintaining the connected components and hierarchical community structures. Comprehensive experiments in various datasets demonstrate the superior performance of HIT-Leiden. In particular, it achieves speedups of up to five orders of magnitude over existing methods.

## Full Text


<!-- PDF content starts -->

Efficient Maintenance of Leiden Communities in Large Dynamic
Graphs
Chunxu Lin
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, China
chunxulin1@link.cuhk.edu.cnYumao Xie
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, China
yumaoxie@link.cuhk.edu.cnYixiang Fang
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, China
fangyixiang@cuhk.edu.cn
Yongmin Hu
ByteDancen
Hangzhou, China
huyongmin@bytedance.comYingqian Hu
ByteDance
Hangzhou, China
huyingqian@bytedance.comChen Cheng
ByteDance
Singapore, Singapore
chencheng.sg@bytedance.com
Abstract
As a well-known community detection algorithm, Leiden has been
widely used in various scenarios such as large language model
(LLM) generation, anomaly detection, and biological analysis. In
these scenarios, the graphs are often large and dynamic, where
vertices and edges are inserted and deleted frequently, so it is costly
to obtain the updated communities by Leiden from scratch when
the graph has changed. Recently, one work has attempted to study
how to maintain Leiden communities in the dynamic graph, but
it lacks a detailed theoretical analysis, and its algorithms are in-
efficient for large graphs. To address these issues, in this paper,
we first theoretically show that the existing algorithms are rela-
tively unbounded via the boundedness analysis (a powerful tool for
analyzing incremental algorithms on dynamic graphs), and also an-
alyze the memberships of vertices in communities when the graph
changes. Based on theoretical analysis, we develop a novel efficient
maintenance algorithm, calledHierarchical Incremental Tree Lei-
den( HIT-Leiden ), which effectively reduces the range of affected
vertices by maintaining the connected components and hierarchi-
cal community structures. Comprehensive experiments in various
datasets demonstrate the superior performance of HIT-Leiden . In
particular, it achieves speedups of up to five orders of magnitude
over existing methods.
CCS Concepts
•Information systems →Clustering;Data stream mining;•
Theory of computation→Dynamic graph algorithms.
Keywords
Incremental graph algorithms, community detection, Leiden algo-
rithm
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGMOD ’26, Bengaluru, India
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXACM Reference Format:
Chunxu Lin, Yumao Xie, Yixiang Fang, Yongmin Hu, Yingqian Hu, and Chen
Cheng. 2026. Efficient Maintenance of Leiden Communities in Large Dy-
namic Graphs. InProceedings of Make sure to enter the correct conference
title from your rights confirmation email (SIGMOD ’26).ACM, New York, NY,
USA, 18 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
𝑣!𝑣"𝐶#𝑣$𝑣%𝑣#𝑣&𝑣'𝑣(𝐶&
(a) A static graph𝐺
𝑣!𝑣"𝐶#𝑣$𝑣%𝑣#𝑣&𝑣'𝑣(𝐶& (b) A dynamic graph𝐺′
Figure 1: Illustrating community maintenance, where ( 𝑣1,𝑣3)
is a newly inserted edge and (𝑣 3,𝑣5) is a newly deleted edge.
As one of the fundamental measures in network science, modu-
larity [ 60] effectively measures the strength of division of a network
into modules (also called communities). Essentially, it captures the
difference between the actual number of edges within a community
and the expected number of such edges if connections were random.
By maximizing the modularity of a graph, it can reveal all the com-
munities in the graph. In Figure 1(a), for example, by maximizing
the modularity of the graph, we can obtain two communities 𝐶1and
𝐶2. As shown in the literature [ 13,78], the graph communities have
found a wide range of applications in recommendation systems,
social marketing, and biological analysis.
One of the most popular community detection (CD) algorithms
that use modularity maximization is Louvain [ 10], which partitions
a graph into disjoint communities. As shown in Figure 2(a), Louvain
employs an iterative process with each iteration having two phases,
calledmovementandaggregation, to adjust the community struc-
ture and improve modularity. Specifically, in the movement phase,
each vertex is relocated to a suitable community to maximize the
modularity of the graph. In the aggregation phase, all the vertices
belonging to the same community are merged into a supervertex to
form a supergraph for the next iteration. Since a supervertex corre-
sponds to a set of vertices, the communities of a graph naturally
form a tree-like hierarchical structure. In practice, to balance mod-
ularity gains against the running time, users often limit Louvain to
𝑃iterations, where𝑃is a pre-defined parameter.

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
MovementInput𝑃iterationsOutputAggregation
(a) The process of the Louvain algorithm [10].
MovementAggregationInput𝑃iterationsOutputRefinement
(b) The process of the Leiden algorithm [80].
Figure 2: Illustrating the Louvain and Leiden algorithms.
Despite its popularity, Louvain may produce communities that
are internally disconnected. This typically occurs during the move-
ment phase, where a vertex that serves as a bridge within a com-
munity may be moved to a different community that has stronger
connections, thereby breaking the connectivity of the original com-
munity. To overcome this issue, Traag et al. [ 80] proposed theLeiden
algorithm1, which introduces an additional phase, calledrefine-
ment, between the movement and aggregation phases, as shown
in Figure 2(b). Specifically, during the refinement phase, vertices
explore merging with their neighbors within the same community
to form sub-communities. By adding this additional phase, Leiden
produces communities with higher quality than Louvain, since its
communities well preserve the connectivity.
As shown in the literature, Leiden has recently received plenty of
attention because of its applications in many areas, including large
language model (LLM) generation [ 43,54,55,63,104], anomaly
detection [ 27,38,65,73,82], and biological analysis [ 1,8,28,47,99].
For example, Microsoft has recently developed Graph-RAG [ 54],
a retrieval-augmented generation (RAG) method that enhances
prompts by searching external knowledge to improve the accuracy
and trustworthiness of LLM generation, and builds a hierarchical
index by using the communities detected by Leiden. As another
example, Liu et al. introduced eRiskComm [ 48], a community-based
fraud detection system that assists regulators in identifying high-
risk individuals from social networks by using Louvain to partition
communities, and Leiden can be naturally applied in this context.
In the aforementioned application scenarios, the graphs often
evolve frequently over time, with many insertions and deletions
of vertices and edges. For instance, in Wikipedia, the number of
English articles increases by about 15,000 per month as of July
20242, making their contributors form a massive and continuously
evolving collaboration graph, where nodes represent users. In these
settings, changes to the underlying graph can significantly alter the
communities produced by Leiden, thereby affecting downstream
tasks and decision-making. However, the original Leiden algorithm
is designed for static graphs, so it is very costly to recompute the
communities from scratch using Leiden whenever a graph change
occurs, especially for large graphs. Hence, it is strongly desirable to
develop efficient algorithms for maintaining the up-to-date Leiden
communities in large dynamic graphs.
Prior works.To maintain Louvain communities in dynamic
graphs, several algorithms have been developed, such as DF-Louvain
[69], Delta-Screening [ 97], DynaMo [ 105], and Batch [ 18]. However,
little attention has been paid to maintaining Leiden communities. To
the best of our knowledge, [ 70] is the only work that achieves this.
It first uses some optimizations for the first iteration of DF-Leiden ,
1As of July 2025, Leiden has received over 5,000 citations according to Google Scholar.
2https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia
Opt-movementAggregationInputOutputOpt-refinementLeiden(a) The process of the increment algorithms in [70].
Inc-movementInc-aggregationInput𝑃iterationsOutputInc-refinement
(b) The process of ourHIT-Leidenalgorithm.
Figure 3: Algorithms for maintaining Leiden communities.
and then invokes the original Leiden algorithm for the remaining
iterations, as depicted in Figure 3(a). Following the optimized move-
ment phase ( opt-movement ), the refinement phase in DF-Leiden
separates communities affected by edge or vertex changes into mul-
tiple sub-communities, while leaving unchanged communities as
single sub-communities. The aggregation phase remains identical
to that of the Leiden algorithm. After constructing the aggregated
graph, the standard Leiden algorithm is applied to complete the
remaining CD process. The author has also developed two variants
ofDF-Leiden , called ND-Leiden andDS-Leiden , by using differ-
ent optimizations for the movement phase of the first iteration.
Nevertheless, there is a lack of detailed theoretical analysis for
these algorithms, and they are inefficient for large graphs with few
changes.
Our contributions.To address the above limitations, we first
theoretically analyze the time cost of existing algorithms for main-
taining Leiden communities and theoretically show that they are
relatively unbounded via the boundedness analysis, which is a
powerful tool for analyzing the time complexity of incremental
algorithms on dynamic graphs. We further analyze the membership
of vertices in communities and sub-communities when the graph
edges change, and observe that the procedure for maintaining these
memberships generalizes naturally to all the supergraphs generated
by Leiden. The above analysis not only lays a solid foundation for us
to comprehend existing algorithms but also offers us opportunities
to improve upon them.
Based on the above analyses, we develop a novel efficient mainte-
nance algorithm, called Hierarchical Incremental Tree Leiden (HIT-
Leiden), which effectively reduces the range of affected vertices by
maintaining the connected components and hierarchical commu-
nity structures. As depicted in Figure 3(b), HIT-Leiden is an itera-
tive algorithm with each iteration having three key phases, namely
incremental movement, incremental refinement, and incremental
aggregation, abbreviated as inc-movement ,inc-refinement , and
inc-aggregation , respectively. More specifically, inc-movement
extends the movement phase from [ 70] by incorporating hierar-
chical community structures [ 80]. Unlike prior approaches, it op-
erates on a supergraph where each supervertex represents a sub-
community, focusing on hierarchical dependencies between com-
munities and their nested substructures. Inspired by the key tech-
nique of maintaining the connected components in dynamic graphs
[90],inc-refinement maintains sub-communities by using tree-
based structures to efficiently track changes in sub-communities.
Inc-aggregation updates the supergraph by computing structural
changes based on the outputs of the previous two phases.
We have evaluated HIT-Leiden on several large-scale real-world
dynamic graph datasets. The experimental results show that our
algorithm achieves comparable community quality with the state-
of-the-art algorithms for maintaining Leiden communities, while

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
achieving up to five orders of magnitude faster than DF-Leiden . In
addition, we have deployed our algorithm in real-world applications
at ByteDance.
Outline.We first review related work in Section 2. We then for-
mally introduce some preliminaries, including the Leiden algorithm
and problem definition in Section 3, provide some theoretical anal-
ysis in Section 4, and present our proposed HIT-Leiden algorithm
in Section 5. Finally, we present the experimental results in Section
6 and conclude in Section 7.
2 Related Work
In this section, we first review the existing works of CD for both
static and dynamic graphs. We simply classify these works as mod-
ularity and other metrics-based CD methods.
•Modularity-based CD.Modularity-based CD methods aim to
partition a graph such that communities exhibit high internal con-
nectivity relative to a null model. Among these methods, Louvain
[10] is the most popular one due to its high efficiency and scalability
as shown in some comparative analyses [ 4,39,94]. Leiden [ 80] im-
proves upon Louvain by resolving the problem of disconnected com-
munities, yielding higher-quality results with comparable runtime.
Other modularity heuristics [ 19,56,58] or incorporate simulated
annealing [ 11,37], spectral techniques [ 59], and evolutionary strate-
gies [ 42,49]. Further refinements explore multi-resolution [ 77], ro-
bust optimization [ 5], normalized modularity [ 52], and clustering
cost frameworks [ 35]. Recent neural approaches have integrated
modularity objectives into deep learning models [ 9,12,89,93,100],
enhancing representation learning for CD.
Besides, some recent works have studied how to incrementally
maintain modularity-based communities when the graph is changed.
Aynaud et al. [ 6] proposed one of the earliest approaches by reusing
previous community assignments to warm-start the Louvain al-
gorithm. Subsequent works extended this idea to both Louvain
[18,20,53,62,69,74,75,97] and Leiden [ 70], incorporating mecha-
nisms such as edge-based impact screening or localized modular-
ity updates. Nevertheless, the existing algorithms of maintaining
Leiden communities lack in-depth theoretical analysis, and their
practical efficiency is poor. Other methods based on modularity,
including extensions to spectral clustering [ 17], multi-step CD [ 7],
and label propagation-based methods [ 61,86–88] have been studied
on dynamic graphs.
•Other metrics-based CD.Beyond modularity, various CD
methods have been developed by using different optimization pur-
poses, such as similarity, statistical inference, spectral clustering,
and neural networks. The similarity-based methods like SCAN
[23,83,92] identify dense regions from the graph via structural
similarity. Statistical inference approaches, including stochastic
block models [ 2,29,36,64], infer communities by fitting genera-
tive probabilistic models to observed networks. Spectral clustering
methods [ 3,22,57] exploit the eigenstructure of graph Laplacians
to group nodes with similar structural roles. Deep learning-based
methods for CD have recently gained traction. Graph convolutional
networks [ 21,31,32,40,50,76,91,101,103], and graph attention
networks [ 26,34,51,81,84,96] have demonstrated strong perfor-
mance in learning expressive node embeddings for CD tasks. For
more details, please refer to recent survey papers of CD [13, 78].Table 1: Frequently used notations and their meanings.
Notation Meaning
𝐺=(𝑉,𝐸) A graph with vertex set𝑉and edge set𝐸
𝑁(𝑣),𝑁 2(𝑣) The vertex𝑣’s 1- and 2-hop neighbor sets, resp.
𝑤(𝑣𝑖,𝑣𝑗) The weight of edge between𝑣 𝑖and𝑣𝑗
𝑑(𝑣) The weighted degree of vertex𝑣
𝑚 The total weight of all edges in𝐺
C A set of communities forming a partition of𝐺
𝑄 The modularity of the graph𝐺with partitionC
𝐺𝑝=(𝑉𝑝,𝐸𝑝) The supergraph in the𝑝-th iteration of Leiden
Δ𝑄(𝑣→𝐶′,𝛾) Modularity gain by moving𝑣from𝐶to𝐶′with𝛾
𝑓(·):𝑉→C A mapping from vertices to communities
𝑓𝑝(·):𝑉𝑃→C A mapping from supervertices to communities
𝑠𝑝(·):
𝑉𝑝→𝑉𝑝+1A mapping from supervertices in 𝑝-th level to
supervertices in(𝑝+ 1)-th level (sub-communities)
Δ𝐺 The set of changed edges in the dynamic graph
Besides, many of the above methods have also been extended for
dynamic graphs. Ruan et al. [ 68] and Zhang et al. [ 98] have studied
structural graph clustering on dynamic graphs, which is based
on structural similarity. Temporal spectral methods [ 16,17] and
dynamic stochastic block models [ 45,72] enable statistical modeling
of evolving community structures over time. Recent deep learning
approaches also support dynamic CD through mechanisms such as
temporal embeddings [ 102], variational inference [ 41], contrastive
learning [ 15,24,85], and generative modeling [ 33]. These models
capture temporal dependencies and structural evolution.
3 Preliminaries
In this section, we first formally present the problem we study,
and then briefly introduce the original Leiden algorithm. Table 1
summarizes the notations frequently used throughout this paper.
3.1 Problem definition
We consider anundirected and weighted graph 𝐺=(𝑉,𝐸) ,
where𝑉and𝐸are the sets of vertices and edges, respectively. Each
vertex𝑣’s neighbor set is denoted by 𝑁(𝑣) . Each edge(𝑣𝑖,𝑣𝑗)is
associated with a positive weight 𝑤(𝑣𝑖,𝑣𝑗)> 0. The degree of 𝑣𝑖is
given by𝑑(𝑣𝑖)=∑︁
𝑣𝑗∈𝑁(𝑣𝑖)𝑤(𝑣𝑖,𝑣𝑗). Denote by 𝑚the total weight
of all edges in𝐺, i.e.,𝑚=∑︁
(𝑣𝑖,𝑣𝑗)∈𝐸𝑤(𝑣𝑖,𝑣𝑗).
Given a graph 𝐺=(𝑉,𝐸) , the CD process aims to partition all
the vertices of 𝑉into some disjoint sets C, each of which is called
a community, corresponding to a set of vertices that are densely
connected. This process can be modeled as a mapping function
𝑓(·) :𝑉→C , such that each 𝑣belongs to a community 𝑓(𝑣) of
the partition C. For each vertex 𝑣, the total weight between 𝑣and a
community𝐶is denoted by𝑤(𝑣,𝐶)=∑︁
𝑣′∈𝑁(𝑣)∩𝐶𝑤(𝑣,𝑣′).
As a well-known CD metric, the modularity measures the differ-
ence between the actual number of edges in a community and the
expected number of such edges.
Definition 1 (Modularity [ 10]).Given a graph 𝐺=(𝑉,𝐸) and
a community partition Cover𝑉, the modularity 𝑄(𝐺,C,𝛾) of the
graph𝐺with the partitionCis defined as:
𝑄(𝐺,C,𝛾)=∑︂
𝐶∈C(︄
1
2𝑚∑︂
𝑣∈𝐶𝑤(𝑣,𝐶)−𝛾(︃𝑑(𝐶)
2𝑚)︃2)︄
,(1)

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
Algorithm 1:Leiden algorithm [71, 79]
Input:𝐺,𝑓(·),𝑃,𝛾
Output:Updated𝑓(·)
1𝐺1←𝐺,𝑓1(·)←𝑓(·);
2for𝑝=1to𝑃do
3𝑓𝑝(·)←𝑀𝑜𝑣𝑒(𝐺𝑝,𝑓𝑝(·),𝛾);
4𝑠𝑝(·)←𝑅𝑒𝑓𝑖𝑛𝑒(𝐺𝑝,𝑓𝑝(·),𝛾);
5ifp < Pthen
6𝐺𝑝+1,𝑓𝑝+1(·)←𝐴𝑔𝑔𝑟𝑒𝑔𝑎𝑡𝑒(𝐺𝑝,𝑓𝑝(·),𝑠𝑝(·));
7Update𝑓(·)using𝑠1(·),···,𝑠𝑃(·);
8return𝑓(·);
where𝑑(𝐶) is the total degree of all vertices in a community 𝐶, and
𝛾>0is a superparameter.
Note that the parameter 𝛾> 0controls the granularity of the
detected communities [ 67]. A higher𝛾favors smaller, finer-grained
communities. In practice, 𝛾is often set to 0.5, 1, 4, or 32, as shown
in [46]. Besides, to guide community updates, the concept of modu-
larity gain is often used to capture the changed modularity when a
vertex is moved from one community to another.
Definition 2 (Modularity gain [ 10]).Given a graph 𝐺, a par-
titionC, and a vertex 𝑣that belongs to a community 𝐶, the modularity
gain of moving𝑣from𝐶to another community𝐶′is defined as:
Δ𝑄(𝑣→𝐶′,𝛾)=𝑤(𝑣,𝐶′)−𝑤(𝑣,𝐶)
2𝑚
+𝛾·𝑑(𝑣)·(𝑑(𝐶)−𝑑(𝑣)−𝑑(𝐶′))
(2𝑚)2.(2)
In this paper, we focus on the dynamic graph with insertions and
deletions of both vertices and edges. Since a vertex insertion (resp.
deletion) can be modeled as a sequence of edge insertions (resp.
deletions), we simply focus on edge changes. Given a set of edge
changes Δ𝐺to a graph𝐺=(𝑉,𝐸) , we obtain an updated graph
𝐺′=(𝑉′,𝐸′). Since there are two types of edge updates, we let
Δ𝐺=Δ𝐺+∪Δ𝐺−, where Δ𝐺+=𝐸′\𝐸andΔ𝐺−=𝐸\𝐸′denote the
sets of inserted and deleted edges, respectively. We denote updated
edges(𝑣𝑖,𝑣𝑗,𝛼)∈Δ𝐺+and(𝑣𝑖,𝑣𝑗,−𝛼)∈Δ𝐺−, where𝛼is positive,
i.e.,𝛼> 0. We use𝐺⊕Δ𝐺 to denote applying Δ𝐺to𝐺, yielding an
updated graph𝐺′.
We now formally introduce the problem studied in this paper.
Problem 1 (Maintenance of Leiden communities [ 70]).Given
a graph𝐺with its Leiden communities C, and some edge updates Δ𝐺,
return the updated Leiden communities after applyingΔ𝐺to𝐺.
We illustrate our problem via Example 1.
Example 1.In Figure 1(a), the original graph 𝐺with unit edge
weights contains two Leiden communities: 𝐶1={𝑣 1,𝑣2}and𝐶2=
{𝑣3,𝑣4,𝑣5,𝑣6,𝑣7,𝑣8}. After inserting a new edge (𝑣1,𝑣3)and deleting
an existing edge(𝑣3,𝑣5)into𝐺, we obtain an updated graph 𝐺′,
which has two updated communities 𝐶1={𝑣 1,𝑣2,𝑣3,𝑣4}and𝐶2=
{𝑣5,𝑣6,𝑣7,𝑣8}.
3.2 Leiden algorithm
Algorithm 1 presents Leiden [ 71,79], following the process in Fig-
ure 2(b). Given a graph 𝐺, and an initial mapping 𝑓(·) (w.l.o.g.,
𝑓(𝑣)={𝑣} ), it first initializes the level-1 supergraph 𝐺1, lets level-1
𝑣!"𝑣#$"𝑣##"𝑣#""𝑣#%&𝑣#&&𝑣"#𝑣##𝑣%#𝑣&#𝑣'#𝑣(#𝑣)#𝑣*#(a) All the communities.
𝑣!!𝑣"!𝑣#!𝑣$!𝑣%!𝑣&!𝑣'"𝑣!!"𝑣!""𝑣!((𝑣!)(𝑣)!𝑣(!𝑣!*" (b) A tree-like structure.
Figure 4: The process of Leiden for the graph 𝐺in Figure 1(a).
mapping𝑓1(·)be𝑓(·), and sets up the sub-community mapping
𝑠(·)(line 1). Next, it iterates𝑃times, each having three phases.
(1)Movement phase(line 3): for each supervertex 𝑣𝑝in the
supergraph 𝐺𝑝, it attempts to move 𝑣𝑝to a neighboring
community that yields the maximum positive modularity
gain, resulting in an updated community mapping𝑓𝑝(·).
(2)Refinement phase(line 4): it splits each community into
some sub-communities such that each of them corresponds
to a connected component, producing a sub-community map-
ping𝑠𝑝(·).
(3)Aggregation phase(line 6): when 𝑝<𝑃 , it aggregates each
sub-community as a supervertex and builds a new graph
𝐺𝑝+1.
Finally, after 𝑃iterations, we update 𝑓(·)and obtain the commu-
nities (lines 7-8). Note that 𝑓(·)is updated using 𝑠𝑃(·)rather than
𝑓𝑃(·)since sub-communities guarantee connectivity with com-
parable modularity. Besides, we use the terms supervertex and
sub-community interchangeably in this paper. A superedge is an
edge between two supervertices, and its weight is the sum of the
weights of edges between the supervertices.
Clearly, the vertices assigned to a sub-community will be further
aggregated as a supervertex, so all the vertices and supervertices
generated naturally form a tree-like hierarchical structure. The
total time complexity of Leiden is 𝑂(𝑃·(|𝑉|+|𝐸|)) [71], since each
iteration costs𝑂(|𝑉|+|𝐸|)time.
Example 2.Figure 4 (a) depicts the process of Leiden with 𝑃=3
for the graph in Figure 1. Denote by 𝑣𝑝
𝑖the supervertex (i.e., sub-
community) in the 𝑝-th iteration of Leiden. It generates three levels
of supergraphs: 𝐺1,𝐺2, and𝐺3, with𝐺1=𝐺. The vertices of these
supergraphs form a tree-like structure, as shown in Figure 4(b).
Take the first iteration as an example depicted in Figure 5. In
the movement phase, it generates three communities 𝐶1={𝑣1
1,𝑣1
2},
𝐶2={𝑣1
5,𝑣1
6,𝑣1
7,𝑣1
8}and𝐶3={𝑣1
3,𝑣1
4}. In the refinement phase, 𝐶2is
split into two sub-communities 𝑣2
11={𝑣1
5,𝑣1
6}and𝑣2
12={𝑣1
7,𝑣1
8}, and
𝐶1and𝐶2are unchanged. In the aggregation phase, all vertices are
aggregated into supervertices based on their sub-community mem-
berships, resulting in𝐺2.
4 Theoretical Analysis of Leiden
In this section, we first analyze the boundedness of existing al-
gorithms, then study how vertex behavior impacts community
structure under graph updates, and extend it to supergraphs.
4.1 Boundedness analysis
We first introduce some concepts related to boundedness.
•Notation.Let Θdenote the CD query applied to a graph 𝐺,
where Θ(𝐺)=C is the set of detected communities. The new graph

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
MovementRefinementAggregation𝐺!𝐺!𝐺!𝐺"𝑣#𝑣$𝑣%𝑣&𝑣!𝑣"𝑣'𝑣(𝑣#𝑣$𝐶!𝑣%𝑣&𝑣!𝑣"𝑣'𝑣(𝐶'𝑣!𝑣"𝑣)"𝑣'𝑣(𝑣!*"𝑣#𝑣$𝑣%𝑣&𝑣!!"𝑣!""𝐶!𝐶'𝐶"𝐶"𝐶!𝐶'𝑣!""𝑣)"𝑣!*"𝑣!!"𝐶"
(a) The process of hierarchical partitions at the first iteration on the graph.
𝑣!!"𝑣#!𝑣$!𝑣%"𝑣!!𝑣"!𝐶!𝐶&𝑣!""𝑣'!𝑣(!𝑣!)"𝑣*!𝑣&!𝐶" (b) The tree-like structure.
Figure 5: The process of hierarchical partitions of Figure 4 at level-1 with the Leiden algorithm.
is𝐺⊕Δ𝐺 , and the updated community is Θ(𝐺⊕Δ𝐺) . We denote
the output difference asΔC, whereΘ(𝐺⊕Δ𝐺)=Θ(𝐺)⊕ΔC.
•Concepts of boundedness.The notion of boundedness [ 66]
evaluates the effectiveness of an incremental algorithm using the
metric CHANGED , defined as CHANGED=Δ𝐺+ΔC , which leads to
|CHANGED|=|Δ𝐺|+|ΔC|.
Definition 3 (Boundedness [ 25,66]).An incremental algorithm
is bounded if its computational cost can be expressed as a polynomial
function of|CHANGED|and|Θ|. Otherwise, it is unbounded.
•Concepts of relative boundedness.In real-world dynamic
graphs,|CHANGED| is often small, yet some unbounded algorithms
can be solved in polynomial time using measures comparable to
|CHANGED| , making these algorithms feasible. To assess these incre-
mental algorithms effectively, Fan et al. [ 25] introduced the concept
of relative boundedness, which leverages a more refined cost model
called the affected region. Let AFFdenote the affected part, the re-
gion of the graph actually processed by the incremental algorithm.
Definition 4 ( AFF[25]).Given a graph 𝐺, a query Θ, and the
input update Δ𝐺to𝐺,AFFsignifies the cost difference of the static
algorithm between computingΘ(𝐺)andΘ(𝐺⊕Δ𝐺).
Unlike CHANGED ,AFFcaptures the concrete portion of the graph
touched by an incremental algorithm, providing a tighter bound
on its computational cost. This leads to the following definition.
Definition 5 (Relative boundedness [ 25]).An incremental
graph algorithm is relatively bounded to the static algorithm if its
cost is polynomial in|Θ|and|AFF|.
We now analyze the boundedness of existing incremental Leiden
algorithms.
Theorem 1.When processing an edge deletion or insertion, the
incremental Leiden algorithms proposed in [ 70] all cost𝑂(𝑃·(|𝑉|+
|𝐸|)).
Table 2: Incremental Leiden algorithms
Method Time complexityRelative
boundedness
ST-Leiden[70] 𝑂(𝑃·(|𝑉|+|𝐸|)) ✗
DS-Leiden[70] 𝑂(𝑃·(|𝑉|+|𝐸|)) ✗
DF-Leiden[70] 𝑂(𝑃·(|𝑉|+|𝐸|)) ✗
HIT-Leiden 𝑂(|𝑁 2(CHANGED)|+|𝑁 2(AFF)|) ✓
By Theorem 1, the existing algorithms for maintaining Leiden
communities are both unbounded and relatively unbounded as
shown in Table 2. They are very costly for large graphs, even with
a small update. Following, we review the property of Leiden and
then identifyAFFof Leiden in the end.4.2 Vertex optimality and subpartition 𝛾-density
As shown in the literature [ 10,80], if𝑠𝑃(·)=𝑓𝑃(·)after𝑃iterations,
Leiden is guaranteed to satisfy the following two properties:
•Vertex optimality:All the vertices are vertex optimal.
•Subpartition 𝛾-density:All the communities are subparti-
tion𝛾-dense.
To design an efficient and effective maintenance algorithm for
Leiden communities, we analyze the behaviors of vertices and com-
munities when the graph changes as follows.
•Analysis of vertex optimality.We begin with a key concept.
Definition 6 (Vertex optimality [ 10]).A community 𝐶∈C
is called vertex optimality if for each vertex 𝑣∈𝐶 and𝐶′∈C, the
modularity gainΔ𝑄(𝑣→𝐶′,𝛾)≤0.
Next, we introduce an assumption in the maintenance of Louvain
communities [69, 97]:
Assumption 1.The sum of weights of the updated edges is suffi-
ciently small relative to the graph size𝑚.
Based on Assumption 1, prior studies suggest that when the num-
ber of edge updates is small relative to the graph size, three heuris-
tics hold: (1) intra-community edge deletions and inter-community
edge insertions could affect vertex-level community membership [ 69,
97]; (2) Inter-community edge deletions and intra-community edge
insertions can be ignored [ 69,97]; (3) Vertices directly involved
in such edge changes are the most likely to alter their communi-
ties [ 69]. The heuristics are stated in Observation 1, which can be
proved based on Definition 6.
Observation 1 ([ 69]).Given an intra-community edge deletion
(𝑣𝑖,𝑣𝑗,−𝛼) or a cross-community edge insertion (𝑣𝑖,𝑣𝑗,𝛼), its effect on
the community memberships of vertices 𝑣𝑖and𝑣𝑗can not be ignored.
We further derive the propagation of community changes from
Observation 1.
Lemma 1.When a vertex𝑣changes its community to𝐶, then the
communities of its neighbors not in 𝐶in the updated graph could be
affected.
Proof. Assuming𝑣changes its community from 𝐶𝑖to𝐶, there
are three cases:
(1)For each neighbor 𝑣𝑖in𝐶𝑖, the edge(𝑣,𝑣𝑖)is adeleted intra-
communityedge and an inserted cross-community edge;
(2)For each neighbor 𝑣𝑗in𝐶, the edge(𝑣,𝑣𝑗)is a deleted cross-
community edge and an inserted intra-community edge;
(3)For each other neighbor 𝑣𝑘, edge(𝑣,𝑣𝑘)is a deleted cross-
community edge and aninserted cross-communityedge.

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
𝑣!𝑣"𝑣#
(a) A triangle.
𝑣!𝑣"𝑣# (b) Delete an edge.
𝑣!𝑣"𝑣# (c) Delete two edges.
Figure 6: An example for illustrating subpartition 𝛾-density.
Since only the first and third cases meet the conditions in Observa-
tion 1, all the neighbors of 𝑣that are not in 𝐶are likely to change
their communities.□
Based on these analyses, we develop a novel movement phase,
called inc-movement inHIT-Leiden to preserve vertex optimality,
which will be introduced in Section 5.1.
•Analysis of subpartition 𝛾-density.For simplified analy-
sis, we first introduce 𝛾-order and𝛾-connectivity, which are key
concepts for defining subpartition𝛾-density.
Definition 7 ( 𝛾-order).Given two vertex sequences 𝑋and𝑌of
a graph𝐺, let𝑋⊗𝑌 represent that 𝑌is merged into 𝑋such that2𝑚·
𝑤(𝑋,𝑌)≥𝛾·𝑑(𝑋)·𝑑(𝑌) , where𝑤(𝑋,𝑌)=∑︁
𝑣𝑖∈𝑋∑︁
𝑣𝑗∈𝑌𝑤(𝑣𝑖,𝑣𝑗).
A𝛾-order of a vertex sequence 𝑈={𝑣 1,···,𝑣𝑥}represents the merged
sequence starting from singleton sequences{𝑣 1},···,{𝑣𝑥}.
We can maintain one 𝛾-order per sub-community from Leiden ,
which is represented by the sequence of vertices merging into the
sub-community inrefinementphase ofLeiden.
Definition 8 ( 𝛾-connectivity [ 80]).Given a graph 𝐺, a vertex
sequence𝑈is𝛾-connected if 𝑈can be generated from at least one
𝛾-order.
Definition 9 (Subpartition 𝛾-density [ 80]).A vertex sub-
sequence𝑈⊆𝐶∈C is subpartition 𝛾-dense if𝑈is𝛾-connected,
and any intermediate vertex sequence 𝑋is locally optimized, i.e.,
Δ𝑄(𝑋→∅,𝛾)≤0.
Notably, Δ𝑄(𝑋→∅,𝛾) ≤ 0denotes the modularity gain of
moving𝑋from𝐶to an empty set, whose calculation follows the
same formula as the standard modularity gain in Equation (2).
Example 3.The triangle in Figure 6(a) is subpartition 𝛾-dense
with𝛾= 1since there are six different 𝛾-orders. For instance, one is
{𝑣3}⊗({𝑣 1}⊗{𝑣 2}), which represents that 𝑣2is merged into{𝑣1}
generating sequence {𝑣1,𝑣2}, and then{𝑣1,𝑣2}merges into 𝑣3gen-
erating{𝑣1,𝑣2,𝑣3}. After deleting the edge (𝑣1,𝑣2), although{𝑣3}⊗
({𝑣 1}⊗{𝑣 2})is not a𝛾-order, the update graph is still subpartition
𝛾-dense since{𝑣1}⊗({𝑣 2}⊗{𝑣 3})is a𝛾-order in the update graph.
After continuing to delete the edge (𝑣2,𝑣3), the updated graph is not
subpartition𝛾-dense since𝑣 2is not connected to𝑣 1and𝑣 3.
In essence, each community 𝐶(or sub-community 𝑆) of Leiden
is subpartition 𝛾-dense, since (1) any sub-community in 𝐶(or𝑆)
is locally optimized, and (2) all sub-communities are 𝛾-connected.
Notably, as shown in Figure 3(b), vertex optimality ensures the first
condition by design since any sub-community will be a supervertex
ininc-movement of the next iteration. Thus, we will develop a new
refinement algorithm, inc-refinement , to preserve 𝛾-connectivity
of sub-communities.
Next, we analyze the 𝛾-connectivity property under two kinds
of graph updates, i.e.,edge deletionandedge insertion. For any
vertex𝑣𝑖within a sub-community 𝑆𝑖with a𝛾-order, we denote an
intermediate subsequence of the 𝛾-order containing 𝑣𝑖by𝐼𝑖⊆𝑆𝑖,and the subsequence 𝑈𝑖=𝐼𝑖\{𝑣𝑖}is an intermediate subsequence
of the𝛾-order before merging 𝑣𝑖. For lack of space, all the proofs of
lemmas are shown in the appendix of the full version [ 44] of this
paper.
(1) Edge deletion. We consider the deletions of both intra-sub-
community edges and cross-sub-community edges:
Lemma 2.Given an intra-sub-community edge deletion (𝑣𝑖,𝑣𝑗,−𝛼) ,
assume𝑣𝑗is before𝑣𝑖in the𝛾-order of the sub-community. The effects
of the edge deletion can be described by the following four cases:
(1)𝑣𝑖could be removed from its sub-community only if 𝛼>
2𝑚·𝑤(𝑣𝑖,𝑈𝑖)−𝛾·𝑑(𝑣𝑖)·𝑑(𝑈𝑖)
4𝑚+2𝑤(𝑣𝑖,𝑈𝑖);
(2)𝑣𝑗could be removed from its sub-community only if 𝛼>𝑚−
𝛾·𝑑(𝑣𝑗)·𝑑(𝑈𝑗)
2𝑤(𝑣𝑗,𝑈𝑗);
(3)For any𝑣𝑘∈𝑆𝑖(𝑘≠𝑖,𝑗 ), it could be removed from its sub-
community only if𝛼>𝑚−𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
2𝑤(𝑣𝑘,𝑈𝑘);
(4)For any𝑣𝑙∉𝑆𝑖, it should be removed from its sub-community
if and only if𝛼>𝑚−𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙))
2𝑤(𝑣𝑙,𝑈𝑙).
Lemma 3.Given a cross-sub-community edge deletion (𝑣𝑖,𝑣𝑗,−𝛼) ,
the effects of the edge deletion can be described by the following four
cases:
(1)𝑣𝑖could be removed from its sub-community only if 𝛼>𝑚−
𝛾·𝑑(𝑣𝑖)·𝑑(𝑈𝑖)
2𝑤(𝑣𝑖,𝑈𝑖);
(2)𝑣𝑗holds similar behavior with𝑣 𝑖;
(3)For any𝑣𝑘∈𝑆𝑖∪𝑆𝑗(𝑘≠𝑖,𝑗 ), it could be removed its sub-
community only if𝛼>𝑚−𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
2𝑤(𝑣𝑘,𝑈𝑘);
(4)For any𝑣𝑙∉𝑆𝑖∪𝑆𝑗, it could be removed from its sub-community
only if𝛼>𝑚−𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
2𝑤(𝑣𝑙,𝑈𝑙).
(2) Edge insertion. We consider the insertion of edges containing
the insertions of both intra-sub-community edges and cross-sub-
community edges:
Lemma 4.Given an edge insertion (𝑣𝑖,𝑣𝑗,𝛼), the effects of the edge
insertion can be described by the following four cases:
(1)𝑣𝑖could be removed from its sub-community only if 𝛼>4
𝛾𝑚−
𝑑(𝐼𝑖)or𝛼>2𝑤(𝑣𝑖,𝑈𝑖)
𝛾·𝑑(𝑈𝑖)·𝑚−𝑑(𝑣𝑖);
(2)𝑣𝑗could be removed from its sub-community, only if 𝛼>
2𝑤(𝑣𝑗,𝑈𝑗)
𝛾·𝑑(𝑈𝑗)·𝑚−𝑑(𝑣 𝑗);
(3)For any𝑣𝑘∈𝑆𝑖∪𝑆𝑗(𝑘≠𝑖,𝑗 ), it could be removed from its
sub-community, only if𝛼>𝑤(𝑣𝑘,𝑈𝑘)
𝛾·𝑑(𝑣𝑘)·𝑚−1
2𝑑(𝑈𝑘);
(4)For any𝑣 𝑙∉𝑆𝑖∪𝑆𝑗, it is unaffected.
Observation 2.In the refinement phase of Leiden algorithms,
each vertex𝑣is likely to be merged into the sub-community (interme-
diate subsequence 𝑈), offering more edge weights 𝑤(𝑣,𝑈) and smaller
degrees𝑑(𝑈) . Therefore, the differences of the values of 𝑑(𝑣) ,𝑤(𝑣,𝑈) ,
and𝑑(𝑈) are very small when the traversal order of vertices to be
merged into sub-communities is in ascending order of vertex degree.
By the above observation, 𝛼is unlikely to satisfy the conditions
in cases (2)-(4) of Lemma 2, all the cases of Lemma 3, and the
conditions in cases (1)-(3) of Lemma 4 when 𝛼≪𝑚 (which is often
true as stated in Assumption 1). As a result, when designing the

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
maintenance algorithm, we only need to consider the effect of intra-
sub-community edge deletions on𝑣 𝑖, which cannot be ignored.
Besides, our experiments show the following observation, which
shows that the case (1) of Lemma 2 can also be ignored.
Observation 3.Given an updated graph with its previous sub-
community memberships, for any sub-community 𝑆, we treat each
connected component in 𝑆as a new sub-community. Most of the
maintained communities are subpartition𝛾-dense.
The above observation holds because Leiden only offers us a
𝛾-order from the refinement phase, and a subgraph often exists
with multiple distinct 𝛾-orders as shown in Example 3. Besides, if a
vertex is a candidate affecting 𝛾-connectivity, it is often a candidate
affecting vertex optimality, e.g., the vertex 𝑣2in Figure 6(c). In this
case, the vertex is likely to change its community before verifying
whether the vertex needs to move out of its sub-community. Hence,
the case (1) of Lemma 2 can be ignored if the intra-sub-community
edge deletion does not cause the sub-community to be disconnected.
Based on Observations 2-3, we develop a novel refinement al-
gorithm, called inc-refinement , inHIT-Leiden , which will be
introduced in Section 5.2. As shown in Figures 13 and Figure 14(b),
over99%maintained communities from HIT-Leiden are subparti-
tion𝛾-dense.
Extension to supergraphs.Changes at the lower level propa-
gate upward to superedge changes in the higher-level supergraph,
as Leiden constructs a list of supergraphs in a bottom-up manner.
This motivates us to develop an incremental aggregation phase,
namely inc-aggregation , to compute the superedge changes in
Section 5.3.
Example 4.In Figure 1, communities 𝐶1and𝐶2are treated as su-
pervertices. Deleting an edge (𝑣3,𝑣5,1)and inserting an edge (𝑣1,𝑣3,1)
cause𝑣3and𝑣4to move from 𝐶2to𝐶1. This results in the deletion of
(𝐶2,𝐶2,−2)and insertion of(𝐶 1,𝐶1,2)in the supergraph.
Therefore, we treat each supergraph as a set of facing edge
changes from the previous Leiden community and process them
using a consistent procedure as shown in Figure 3(b).
Characterization of AFF.Based on these analyses, we define
the supervertices that change their communities or sub-communities
as the affected areaAFFof Leiden.
5 Our HIT-Leiden algorithm
Observation1Observation3Assumption1Inc-movementInc-refinementLemma 2Lemma 3Lemma 4Lemma1Observation2
Figure 7: The design rationale for inc-movement and
inc-refinement.
In this section, we first introduce the three key components,
namely inc-movement ,inc-refinement , and inc-aggregation
of our HIT-Leiden . Figure 7 shows the assumption, lemmas, and
observations used in these components. Then, we present an auxil-
iary procedure, called deferred update, abbreviated as def-update .
Afterward, we give an overview of HIT-Leiden , and finally analyze
the boundedness ofHIT-Leiden.Algorithm 2:Inc-movement
Input:𝐺,Δ𝐺,𝑓(·),𝑠(·),Ψ,𝛾
Output:Updated𝑓(·),Ψ,𝐵,𝐾
1𝐴←∅,𝐵←∅,𝐾←∅;
2for(𝑣𝑖,𝑣𝑗,𝛼)∈Δ𝐺do
3if𝛼>0and𝑓(𝑣 𝑖)≠𝑓(𝑣𝑗)then
4𝐴.𝑎𝑑𝑑(𝑣 𝑖);𝐴.𝑎𝑑𝑑(𝑣 𝑗);
5if𝛼<0and𝑓(𝑣 𝑖)=𝑓(𝑣𝑗)then
6𝐴.𝑎𝑑𝑑(𝑣 𝑖);𝐴.𝑎𝑑𝑑(𝑣 𝑗);
7if𝑠(𝑣 𝑖)=𝑠(𝑣𝑗)and𝑢𝑝𝑑𝑎𝑡𝑒_𝑒𝑑𝑔𝑒(︁𝐺Ψ,(𝑣𝑖,𝑣𝑗,𝛼))︁then
8𝐾.𝑎𝑑𝑑(𝑣 𝑖);𝐾.𝑎𝑑𝑑(𝑣 𝑗);
9for𝐴≠∅do
10𝑣𝑖←𝐴.𝑝𝑜𝑝();
11𝐶∗←𝑎𝑟𝑔𝑚𝑎𝑥 𝐶∈C∪∅ Δ𝑄(𝑣𝑖→𝐶,𝛾);
12ifΔ𝑄(𝑣 𝑖→𝐶∗,𝛾)>0then
13𝑓(𝑣 𝑖)←𝐶∗;𝐵.𝑎𝑑𝑑(𝑣𝑖);
14for𝑣 𝑗∈𝑁(𝑣𝑖)do
15if𝑓(𝑣 𝑗)≠𝐶∗then
16𝐴.𝑎𝑑𝑑(𝑣 𝑗);
17for𝑣 𝑗∈𝑁(𝑣𝑖)∧𝑠(𝑣𝑖)=𝑠(𝑣𝑗)do
18if𝑢𝑝𝑑𝑎𝑡𝑒_𝑒𝑑𝑔𝑒(︁𝐺Ψ,(𝑣𝑖,𝑣𝑗,−𝑤(𝑣𝑖,𝑣𝑗)))︁then
19𝐾.𝑎𝑑𝑑(𝑣 𝑖);𝐾.𝑎𝑑𝑑(𝑣 𝑗);
20return𝑓(·),Ψ,𝐵,𝐾;
5.1 Inc-movement
The goal of inc-movement is to preserve vertex optimality. As an-
alyzed in Section 4.2, the endpoints of a deleted intra-community
edge or an inserted cross-community edge may affect their com-
munity memberships. If an affected vertex changes its community,
its neighbors outside the target community may also be affected.
Note that any vertex that changes its community has to change its
sub-community, since each sub-community is a subset of its com-
munity. Hence, sub-community memberships are also considered
ininc-movement.
We first introduce the data structures used to maintain a dynamic
sub-community. According to Observation 3, each connected com-
ponent of a sub-community is treated as a 𝛾-connected subset.
When edge updates or vertex movements split a sub-community
into multiple connected components, we re-assign each result-
ing component as a new sub-community, and the largest sub-
community succeeds the original sub-community’s ID.
𝑣!𝑣"𝑣#𝑆!
(a) Original graph.
𝑣!𝑣"𝑣#𝑆!𝑆# (b) Delete two edges.
𝑣!𝑣"𝑣#𝑆!𝑆# (c) Move out a vertex.
Figure 8: Illustrating the process that a sub-community 𝑆1is
split into two sub-communities𝑆 1and𝑆 2.
Example 5.Figure 8 shows the sub-community 𝑆1is split into two
sub-communities 𝑆1={𝑣 1,𝑣3}and𝑆2={𝑣 2}. The component{𝑣1,𝑣3}
retains the original sub-community ID 𝑆1, since it is larger than {𝑣2}.
The separation can occur either due to the deletion of edges (𝑣1,𝑣2)
and(𝑣2,𝑣3)during graph updates, as shown in Figure 8(b), or due
to the removal of vertex 𝑣2during the movement phase, as shown in
Figure 8(c).

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
Algorithm 3:Inc-refinement
Input:𝐺,𝑓(·),𝑠(·),Ψ,𝐾,𝛾
Output:Updated𝑠(·),Ψ,𝑅,
1𝑅←∅;
2for𝑣𝑖∈𝐾do
3if𝑣𝑖is not in the largest connected component of𝑠(𝑣)then
4Map all vertices in the connected component into a new
sub-community and add them into𝑅;
5for𝑣𝑖∈𝑅do
6if𝑣𝑖is in singleton sub-communitythen
7T←{𝑠(𝑣)|𝑣∈𝑁(𝑣 𝑖)∩𝑓(𝑣𝑖),Δ𝑄(𝑠(𝑣)→∅,𝛾)≤0};
8𝑆∗←𝑎𝑟𝑔𝑚𝑎𝑥 𝑆∈TΔ𝑀(𝑣𝑖→𝑆,𝛾);
9ifΔ𝑀(𝑣 𝑖→𝑆∗,𝛾)>0then
10𝑠(𝑣 𝑖)←𝑆∗;
11for𝑣 𝑗∈𝑁(𝑣𝑖)do
12if𝑠(𝑣 𝑖)=𝑠(𝑣𝑗)then
13𝑢𝑝𝑑𝑎𝑡𝑒_𝑒𝑑𝑔𝑒(︁𝐺Ψ,(𝑣𝑖,𝑣𝑗,𝑤(𝑣𝑖,𝑣𝑗)))︁;
14return𝑠(·),Ψ,𝑅;
To preserve the structure under such changes, we leverage dy-
namic connected component maintenance techniques. Various index-
based methods have been proposed for this purpose, such as D-Tree
[14], DND-Tree [ 90], and HDT [ 30]. LetΨdenote a connected com-
ponent index, abbreviated as CC-index. The graph 𝐺Ψstores the
subgraph of 𝐺consisting only of intra-sub-community edges based
on𝑠(·).
Algorithm 2 shows inc-movement . Given an updated graph
𝐺, a set of graph changes Δ𝐺, community mappings 𝑓(·), sub-
community mappings 𝑠(·), and a CC-index Ψ, it first initializes three
empty sets: 𝐴,𝐵and𝐾(line 1). Here, 𝐴keeps the vertices whose
community memberships may be changed, 𝐵keeps the vertices that
have changed their community memberships, and 𝐾records the
endpoints on edges whose deletion disconnects the connected com-
ponent in𝐺Ψ. Subsequently, vertices involved in intra-community
edge deletion or cross-community edge insertion are added to 𝐴,
and edges in 𝐺Ψare updated according to intra-sub-community
changes (lines 2-7) based on Observations 1 and 3, respectively. If
an edge update in 𝐺Ψcauses a connected component to split (i.e.,
𝑢𝑝𝑑𝑎𝑡𝑒_𝑒𝑑𝑔𝑒(·) returns𝑡𝑟𝑢𝑒), its endpoints are added to 𝐾(line 8).
It then processes vertices in 𝐴until the set is empty (line 9). For
each vertex 𝑣𝑖, it identifies the target community 𝐶∗that yields the
highest modularity gain (lines 10-11). If Δ𝑄(𝑣𝑖→𝐶∗)> 0,𝑓(𝑣𝑖)
is updated to 𝐶∗,𝑣𝑖is added into 𝐵, and the neighbors of 𝑣𝑖not in
𝐶∗are added to 𝐴(lines 12-16), which implements the property in
Lemma 1. Besides, the intra-sub-community edges involving 𝑣𝑖are
deleted from 𝐺Ψ, and the vertices involved in component splits are
added to𝐾(lines 17-19). Finally, it returns 𝑓(·),Ψ,𝐵, and𝐾(line
20).
5.2 Inc-refinement
As discussed in Section 5.1 and Observation 3, we treat each con-
nected component in 𝐺Ψmaintained in inc-movement as a sub-
community. Therefore, we design inc-refinement for re-assigning
each new connected component in 𝐺Ψas a sub-community. Addi-
tionally, we attempt to merge singleton sub-communities whoseprocess is the same as the process of the refinement phase in Leiden
with𝐺 Ψmaintenance.
Algorithm 3 presents its pseudocode. Given an updated graph
𝐺, community mappings 𝑓(·)and sub-community mapping 𝑠(·),
a CC-index Ψ, and a set𝐾, it first initializes 𝑅as an empty list
to track vertices that have changed their sub-communities (line
1). Note that 𝑅is an ordered list sorted in ascending vertex de-
gree mentioned in Observation 2. It then traverses 𝐾to identify
split connected components in 𝐺Ψusing breadth-first search or
depth-first search. If a connected component is not the largest in
its original sub-community, all its vertices are re-mapped to a new
sub-community, and added to 𝑅(lines 2-4). If multiple components
tie for the largest component, one of them is randomly selected
to represent the original sub-community. For each vertex 𝑣𝑖∈𝑅
that is in a singleton sub-community, inc-refinement uses a set
Tto store the locally optimized neighboring sub-communities of
𝑣𝑖within the same community (lines 5-7). Then, it attempts to re-
assign𝑣𝑖to a sub-community 𝑆∗∈T, which offers the highest
modularity gain to eliminate singleton sub-communities (line 8).
Notably,Δ𝑀(𝑣 𝑖→𝑆,𝛾)denotes the modularity gain of moving𝑣 𝑖
from𝑠(𝑣𝑖)to𝑆, whose calculation follows the same formula as the
standard modularity gain. If the gain is positive, 𝑠(𝑣𝑖)is updated to
𝑆∗, and the corresponding intra-sub-community edges are inserted
into𝐺Ψ(lines 9-13). Finally, inc-refinement returns the 𝑠(·),Ψ,
and𝑅(line 14).
5.3 Inc-aggregation
Given an updated graph 𝐺and its edge changes Δ𝐺, modifications
to edges and sub-community memberships are reflected as changes
to superedges and supervertices in the supergraph 𝐻. Let𝑠𝑝𝑟𝑒(·)
(resp.𝑠𝑐𝑢𝑟(·)) denotes the vertex-to-supervertex mappings before
(resp. after) inc-refinement . Any edge change(𝑣𝑖,𝑣𝑗,𝛼)inΔ𝐺cor-
responds to a superedge change (𝑠𝑝𝑟𝑒(𝑣𝑖),𝑠𝑝𝑟𝑒(𝑣𝑗),𝛼) in𝐻, since
the weight of a superedge is the sum of weights of edges between
their sub-communities. Besides, a vertex 𝑣migration from 𝑠pre(𝑣)
to𝑠cur(𝑣)requires updating these weights. Specifically, the original
sub-community 𝑠𝑝𝑟𝑒(𝑣)must decrease the superedge weights cor-
responding to the edge incident to 𝑣, and the new sub-community
𝑠𝑐𝑢𝑟(𝑣)must increase them under the new assignment.
Example 6.Following Example 4, the initial superedge changes
due to edge changes are (𝐶1,𝐶2,1)and(𝐶2,𝐶2,−1). Then, vertices 𝑣3
and𝑣 4move from𝐶 2to𝐶 1, and there are three cases:
(1)𝐶 1gains edges to the neighbors of 𝑣3, resulting in two updates:
(𝐶1,𝐶1,1)and(𝐶 1,𝐶1,1);
(2)𝐶 2loses edges to the neighbor of 𝑣3are(𝐶1,𝐶2,−1)and(𝐶2,𝐶2,−1);
(3)The effect of 𝑣4is skipped to avoid duplicate updates, since its
only neighbor𝑣 3already changed.
After compressing the above six superedge changes, we obtain the
final superedge changes, which are(𝐶 1,𝐶1,2)and(𝐶 2,𝐶2,−2).
Algorithm 4 presents inc-aggregation . Initially, the set of chan-
gesΔ𝐻of𝐻is empty (line 1). Then, it maps the edge changes Δ𝐺
to superedge changes using 𝑠𝑝𝑟𝑒(·)(lines 2-4). Following, it updates
superedges for vertices that switch sub-communities by removing
edges from the old community and adding edges to the new one. For
any vertex𝑣𝑖in𝑅, if updates superedges with each neighbor 𝑣𝑗if

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
Algorithm 4:Inc-aggregation
Input:𝐺,Δ𝐺,𝑠 𝑝𝑟𝑒(·),𝑠𝑐𝑢𝑟(·),𝑅
Output:Δ𝐻,𝑠 𝑝𝑟𝑒(·)
1Δ𝐻←∅;
2for(𝑣𝑖,𝑣𝑗,𝛼)∈Δ𝐺do
3𝑟𝑖←𝑠𝑝𝑟𝑒(𝑣𝑖),𝑟𝑗←𝑠𝑝𝑟𝑒(𝑣𝑗);
4Δ𝐻.𝑎𝑑𝑑((𝑠 𝑖,𝑠𝑗,𝛼));
5for𝑣𝑖∈𝑅do
6for𝑣 𝑗∈𝑁(𝑣𝑗)do
7if𝑠 𝑐𝑢𝑟(𝑣𝑗)=𝑠𝑝𝑟𝑒(𝑣𝑗)or𝑖<𝑗then
8Δ𝐻.𝑎𝑑𝑑((𝑠 𝑝𝑟𝑒(𝑣𝑖),𝑠𝑝𝑟𝑒(𝑣𝑗),−𝑤(𝑣𝑖,𝑣𝑗)));
9Δ𝐻.𝑎𝑑𝑑((𝑠 𝑐𝑢𝑟(𝑣𝑖),𝑠𝑐𝑢𝑟(𝑣𝑗),𝑤(𝑣𝑖,𝑣𝑗)));
10Δ𝐻.𝑎𝑑𝑑((𝑠 𝑝𝑟𝑒(𝑣𝑖),𝑠𝑝𝑟𝑒(𝑣𝑖),−𝑤(𝑣𝑖,𝑣𝑖)));
11Δ𝐻.𝑎𝑑𝑑((𝑠 𝑐𝑢𝑟(𝑣𝑖),𝑠𝑐𝑢𝑟(𝑣𝑖),𝑤(𝑣𝑖,𝑣𝑖)));
12for𝑣𝑖∈𝑅do
13𝑠𝑝𝑟𝑒(𝑣𝑖)←𝑠𝑐𝑢𝑟(𝑣𝑖);
14𝐶𝑜𝑚𝑝𝑟𝑒𝑠𝑠(Δ𝐻);
15returnΔ𝐻,𝑠 𝑝𝑟𝑒(·);
either𝑠𝑐𝑢𝑟(𝑣𝑗)=𝑠𝑝𝑟𝑒(𝑣𝑗)or𝑖<𝑗 to avoid duplicate updates (lines
5-9). Besides, it updates the self-loop for the sub-community of 𝑣𝑖
(lines 10-11). Finally, it locally updates 𝑠𝑝𝑟𝑒(·)to match𝑠𝑐𝑢𝑟(·)for
the next time step (lines 12-13), and compresses entries by summing
the weight of identical superedges inΔ𝐻(line 14).
5.4 Overall HIT-Leiden algorithm
𝑣!""𝑣!#"𝑣!!𝑣$!𝑣%!𝑣&!𝑣'!𝑣(!𝑣#!𝑣"!𝑣)$𝑣!!$𝑣!$$𝑣!*$
(a) Before maintenance.
𝑣!""𝑣!#"𝑣!!𝑣$!𝑣%!𝑣&!𝑣'!𝑣(!𝑣#!𝑣"!𝑣)$𝑣!!$𝑣!$$𝑣!*$ (b) After maintenance.
Figure 9: The hierarchical partitions changes of Figure 1.
Before presenting our overall HIT-Leiden algorithm, we intro-
duce an optimization technique to further improve the efficiency of
the vertices’ membership update. Specifically, when a supervertex
changes its community membership, all the lower-level superver-
tices associated with it have to update their community membership.
As shown in Figure 9, when 𝑣2
10changes its community, 𝑣1
3and𝑣1
4
also update their community memberships to the community con-
taining𝑣2
10. However, during the iteration process of HIT-Leiden ,
a supervertex that changes its community does not automatically
trigger updates of the community memberships of its constituent
lower-level supervertices.
To resolve the above inconsistency, we perform a post-processing
step to synchronize the community memberships across all levels,
as described in Algorithm 5. Let {𝐵𝑃}denote a sequence of 𝑃sets
{𝐵1,···,𝐵𝑃},{𝑠𝑃(·)}denote a sequence of𝑃adajcent-level super-
vertex mappings{𝑠1(·),···,𝑠𝑃(·)}, and{𝑓𝑃(·)}denote a sequence
of𝑃community mappings {𝑓1(·),···,𝑓𝑃(·)}. Note, each 𝐵𝑝in
{𝐵𝑃}collects supervertices at level- 𝑝whose community member-
ships have changed, each 𝑠𝑝(·)in{𝑠𝑃(·)}maps from level- 𝑝super-
vertices to their parent supervertices at level-( 𝑝+1), and each 𝑓𝑝(·)
in{𝑓𝑃(·)}maps from level- 𝑝supervertices to their communities.
A supervertex is added to 𝐵𝑝for one of two reasons: (1) it changesAlgorithm 5:def-update
Input:{𝑓𝑃(·)},{𝑠𝑃(·)},{𝐵𝑃},𝑃
Output:Updated{𝑓𝑃(·)}
1for𝑝from𝑃to1do
2if𝑝≠𝑃then
3for𝑣𝑝
𝑖∈𝐵𝑝do
4𝑓𝑝(𝑣𝑖
𝑝)=𝑓𝑝+1(𝑠𝑝(𝑣𝑖
𝑝));
5if𝑝≠1then
6for𝑣𝑝
𝑖∈𝐵𝑝do
7𝐵𝑝−1.𝑎𝑑𝑑(𝑠−𝑝(𝑣𝑝
𝑖));
8return{𝑓𝑃(·)};
its community during inc-movement , or (2) its higher-level super-
vertex changes community. Hence, for each level 𝑝,def-update
updates each supervertex in 𝐵𝑝by re-mapping its community mem-
bership of its parent using 𝑠𝑝(·)and𝑓𝑝+1(·)when𝑝≠𝑃 (lines 1-4),
and adds its constituent vertices to 𝐵𝑝−1for the next level updates
where𝑠−𝑝(·)is the inverse mapping of 𝑠𝑝(·)when𝑝≠ 1(lines 5-7).
This algorithm also supports updating the mappings {𝑔𝑃(·)}from
each level supervertex to its level-𝑃ancestor.
•Overall HIT-Leiden.After introducing all the key compo-
nents, we present our overall HIT-Leiden in Algorithm 6. The
algorithm proceeds over 𝑃hierarchical levels, where each level- 𝑝
operates on a corresponding supergraph 𝐺𝑝. Besides the commu-
nity membership 𝑓(·),HIT-Leiden also maintains supergraphs
{𝐺𝑃}, community mappings {𝑓𝑃(·)}, sub-community mappings
{𝑔𝑃(·)},{𝑠𝑃
𝑝𝑟𝑒(·)}and{𝑠𝑃
𝑐𝑢𝑟(·)}, and CC-indices{Ψ𝑃}to maintain
sub-community memberships for each level. Note, {𝑠𝑃
𝑝𝑟𝑒(·)}are the
mappings from the previous time step, and {𝑠𝑃
𝑐𝑢𝑟(·)}are the in-time
mappings to track sub-community memberships as they evolve at
the current time step.
Specifically, it initializes {𝑠𝑃
𝑐𝑢𝑟(·)}={𝑠𝑃
𝑝𝑟𝑒(·)}. Given the graph
change Δ𝐺, it first initializes the first-level update Δ𝐺toΔ𝐺1(line
1). It then proceeds through 𝑃iterations, each including three phases
after updating the supergraph𝐺𝑝(line 3).
(1)Inc-movement (line 4): it re-assigns community member-
ships of affected vertices to achieve vertex optimality, which
yields𝑓𝑝(·),Ψ,𝐵𝑝, and𝐾.
(2)Inc-refinement (line 5): it re-maps the supervertices of
split connected components in Ψto new sub-communities,
producing𝑠𝑝
𝑐𝑢𝑟(·),Ψ, and𝑅𝑝.
(3)Inc-aggregation (line 7): it calculates the next level’s su-
peredge changes Δ𝐺𝑝+1, and synchronizes 𝑠𝑝
𝑝𝑟𝑒(·)to match
𝑠𝑝
𝑐𝑢𝑟(·).
After𝑃iterations, def-update (Algorithm 5) synchronizes com-
munity mappings{𝑓𝑃(·)}and sub-community mappings {𝑔𝑃(·)}
across levels (lines 8-9). The final output 𝑓(·)is set to𝑔1(·)(line 10).
Besides, we also return {𝐺𝑃},{𝑓𝑃(·)},{𝑔𝑃(·)},{𝑠𝑃
𝑝𝑟𝑒(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},
and{Ψ𝑃}for the next graph evolution (line 11).
Example 7.Consider the result in Figure 4. The graph undergoes an
edge deletion(𝑣1
3,𝑣1
5,−1)and an edge insertions (𝑣1
1,𝑣1
3,1). Resulting
community and sub-community changes are shown in Figure 10,
with hierarchical changes in Figure 9. Take the second iteration as
an example. In inc-movement , the supervertex 𝑣2
10is reassigned to

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
𝑣!"𝑣#$"𝑣##"𝑣#""𝑣#%&𝑣#&&𝑣"#𝑣##𝑣%#𝑣&#𝑣'#𝑣(#𝑣)#𝑣*#HIT-Leiden𝑣!"𝑣##"𝑣#""𝑣#%&𝑣#&&𝑣"#𝑣##𝑣#$"𝑣%#𝑣&#𝑣'#𝑣(#𝑣)#𝑣*#𝐶#𝐶"𝐶#𝐶"
(a) Community maintain byHIT-Leiden
Update…𝑣!"𝑣#$$𝑣#%"𝑣##"𝑣#""𝑣#&$𝑣!"𝑣#$$𝑣##"𝑣#""𝑣#&$𝑣#%"𝑣#'$𝑣!"𝑣#$$𝑣##"𝑣#""𝑣#&$𝑣#%"Inc-movementInc-aggregation…Inc-refinement𝐶#𝐶"𝐶#𝐶"𝐶#𝐶" (b) The process ofHIT-Leidenin iteration two
Figure 10: An example ofHIT-Leiden
Algorithm 6:HIT-Leiden
Input:{𝐺𝑃},Δ𝐺,{𝑓𝑃(·)},{𝑔𝑃(·)},{𝑠𝑃
𝑝𝑟𝑒(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},{Ψ𝑃},
𝑃,𝛾
Output:𝑓(·),{𝐺𝑃},{𝑓𝑃(·)},{𝑓𝑃(·)},{𝑠𝑃
𝑝𝑟𝑒(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},
{Ψ𝑃}
1Δ𝐺1←Δ𝐺;
2for𝑝from1to𝑃do
3𝐺𝑝←𝐺𝑝⊕Δ𝐺𝑝;
4𝑓𝑝(·),Ψ,𝐵𝑝,𝐾←
inc-movement(𝐺𝑝,Δ𝐺𝑝,𝑓𝑝(·),𝑠𝑝
𝑐𝑢𝑟(·),Ψ,𝛾);
5𝑠𝑝
𝑐𝑢𝑟(·),Ψ,𝑅𝑝←
inc-refinement(𝐺𝑝,𝑓𝑝(·),𝑠𝑝
𝑐𝑢𝑟(·),Ψ,𝐾,𝛾);
6ifp < Pthen
7Δ𝐺𝑝+1,𝑠𝑝
𝑝𝑟𝑒(·)←
inc-aggregation(𝐺𝑝,Δ𝐺𝑝,𝑠𝑝
𝑝𝑟𝑒(·),𝑠𝑝
𝑐𝑢𝑟(·),𝑅𝑝);
8{𝑓𝑃(·)}←def-update({𝑓𝑃(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},{𝐵𝑃},𝑃);
9{𝑔𝑃(·)}←def-update({𝑔𝑃(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},{𝑅𝑃},𝑃);
10𝑓(·)←𝑔1(·);
11return𝑓(·),{𝐺𝑃},{𝑓𝑃(·)},{𝑔𝑃(·)},{𝑠𝑃
𝑝𝑟𝑒(·)},{𝑠𝑃
𝑐𝑢𝑟(·)},
{Ψ𝑃};
𝑣3
15due to disconnection, and migrates from community 𝐶2to𝐶1. In
inc-refinement ,𝑣2
10is merged into 𝑣3
13. Then, inc-aggregation
calculates superedge changes for level-3, including edge insertion
(𝑣3
13,𝑣3
13,2)and edge deletions(𝑣3
14,𝑣3
14,−2).
•Complexity analysis.We now analyze the time complexity
ofHIT-Leiden over𝑃iterations. Let Γ𝑝denote the set of superver-
tices involved in superedge changes, and let Λ𝑝track the superver-
tices that change their communities or sub-communities at level- 𝑝.
Therefore, for each level- 𝑝,inc-movement ,inc-refinement , and
inc-aggregation complete in𝑂(|𝑁 2(Γ𝑝)|+|𝑁 2(Λ𝑝)|),𝑂(|𝑁(Γ𝑝)|+
|𝑁(Λ𝑝)|), and𝑂(|𝑁(Γ𝑝)|+|𝑁(Λ𝑝)|), respectively. Besides, the time
cost of def-update is𝑂(︂∑︁𝑃
𝑝=1|Λ𝑝|)︂
. Hence, the total time cost of
HIT-Leiden is𝑂(∑︁𝑃
𝑝=1(|𝑁 2(Γ𝑝)|+|𝑁 2(Λ𝑝)|))=𝑂(|𝑁 2(CHANGED)|+
|𝑁2(AFF)|) , as analyzed in Section 4.2. As a result, our HIT-Leiden
is bounded relative to Leiden.
6 Experiments
We now present our experimental results. Section 6.1 introduces the
experimental setup. Section 6.2 and 6.3 evaluate the effectiveness
and efficiency ofHIT-Leiden, respectively.Table 3: Datasets used in our experiments.
Dataset Abbr. |𝑉| |𝐸| Timestamp
dblp-coauthor DC 1.8M 29.4M Yes
yahoo-song YS 1.6M 256.8M Yes
sx-stackoverflow SS 2.6M 63.4M Yes
it-2004 IT 41.2M 1.0B No
risk RS 201.0M 4.0B Yes
6.1 Setup
Datasets.We use four real-world dynamic datasets, includingdblp-
coauthor1(academic collaboration),yahoo-song1(user-song inter-
actions),sx-stackoverflow2(developer Q&A), andrisk(financial
transactions) provided by ByteDance. All these dynamic edges are
associated with real timestamps. We also use one static datasetit-
20043(a large-scale web graph), but randomly insert or delete some
edges to simulate a dynamic graph. All the graphs are treated as
undirected graphs. For each real-world dynamic graph, we collect a
sequence of batch updates by sorting the edges in ascending order
of their timestamps; forit-2004, which lacks timestamps, we ran-
domly shuffle its edge order. Table 3 summarizes the key statistics
of the above datasets.
Algorithms.We test the following maintenance algorithms:
•ST-Leiden : A naive baseline that executes the static Leiden
algorithm from scratch when the graph changes.
•ND-Leiden : A simple maintenance algorithm in [ 70], which
processes all vertices during the movement phase, initialized
with previous community memberships.
•DS-Leiden : A maintenance algorithm based on [ 70], which
uses the delta-screening technique [ 97] to restrict the num-
ber of vertices considered in the movement phase.
•DF-Leiden : An advanced maintenance algorithm from [ 70],
which adopts the dynamic frontier approach [ 69] to support
localized updates.
•HIT-Leiden: Our proposed method.
Dynamic graph settings.As the temporal span varies across
datasets (e.g., 62 years fordblp-coauthorversus 8 years forsx-
stackoverflow), we apply a sliding edge window, avoiding reliance
on fixed valid time intervals that are hard to standardize. Initially,
we construct a static graph using the first 80% of edges. Then, we se-
lect a window size 𝑏∈{ 10,102,103,104,105}, denoting the number
of updated edges in an updated batch. Next, we slide this window
𝑟= 9times, so we update 9 batches of edges for each dataset. Note
that by default, we set𝑏=103.
1http://konect.cc/networks/
2https://snap.stanford.edu/data/
3https://networkrepository.com/

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
All the algorithms are implemented in C++ and compiled with
the gcc 8.3.0 compiler using the -O0 optimization level. We set 𝛾= 1
and use𝑃= 10iterations. Before running the Leiden community
maintenance algorithms, we obtain the communities by running
the Leiden algorithm, and HIT-Leiden requires an additional pro-
cedure to build auxiliary structures. Due to the limited number of
iterations, the community structure has not fully converged, so the
maintenance algorithms usually take more time in the first two
batches than in other batches. Therefore, we exclude the first two
batches from efficiency evaluations. Experiments are conducted on
a Linux server running Debian 5.4.56, equipped with an Intel(R)
Xeon(R) Platinum 8336C CPU @ 2.30GHz and 2.0 TB of RAM.
6.2 Effectiveness evaluation
To evaluate the effectiveness of different maintenance algorithms,
we compare the modularity value and proportion of subpartition
𝛾-dense communities for their returned communities. We also eval-
uate the long-term effectiveness of community maintenance and
present a case study.
•Modularity.Figure 11 depicts the average modularity values
of all the maintenance algorithms, where the batch size ranges from
10 to105. Figure 12 depicts the modularity value across all the 9
batches, where the batch size is fixed as 1,000. Across all datasets,
the expected fluctuation in modularity for ST-Leiden is around
0.02due to its inherent randomness. These maintenance algorithms
achieve equivalent quality in modularity, since the difference in
their modularity values is within0 .01. Overall, our HIT-Leiden
achieves comparable modularity with other methods.
•Proportion of subpartition𝛾-density.After runningHIT-
Leiden , for each returned community, we try to re-find its 𝛾-order
such that any intermediate vertex set in the 𝛾-order is locally opti-
mized, according to Definition 9. If we can find a valid𝛾-order for
a community, we classify it as a subpartition 𝛾-dense community.
We report the proportion of subpartition 𝛾-dense communities in
Figure 13. The proportions of subpartition 𝛾-dense communities
among these Leiden algorithms are almost 1, and they are within
the expected fluctuation (around 0.0001) caused by the inherent
randomness of the measure method. Thus, HIT-Leiden achieves a
comparable percentage of subpartition𝛾-density with others.
•Long-term effectiveness.To demonstrate the long-term ef-
fectiveness of maintaining communities, we enlarge the number 𝑟
of batches from 9 to 999 and set 𝑏= 10,000. Figure 14(a)-(c) presents
the modularity, proportion of subpartition 𝛾-dense communities,
and runtime on the sx-stackoverflow dataset. We observe that incre-
mental Leiden algorithms exhibit higher stability than ST-Leiden
in modularity since they use previous community memberships,
andHIT-Leidenis faster than other algorithms.
•A case study.Our HIT-Leiden has been deployed at ByteDance
to support several real applications. Here, we briefly introduce the
application of Graph-RAG. To augment the LLM generation for
answering a question, people often retrieve relevant information
from an external corpus. To facilitate retrieval, Graph-RAG builds
an offline index: It first builds a graph for the corpus, then clus-
ters the graph hierarchically using Leiden, and finally associates
a summary for each community, which is generated by an LLM
with some token cost. In practice, since the underlying corpus oftenchanges, the communities and their summaries need to be updated
as well. Our HIT-Leiden can not only dynamically update the com-
munities efficiently, but also save the token cost since we only need
to regenerate the summaries for the updated communities.
To experiment, we use the HotpotQA [ 95] dataset, which con-
tains Wikipedia-based question-answer (QA) pairs. We randomly
select 9,500 articles to build the initial graph, and insert 9 batches
of new articles, each with 5 articles. The LLM we use is doubao-
1.5-pro-32k. To support a dynamic corpus, we adapt the static
Graph-RAG method by updating communities using ST-Leiden
andHIT-Leiden , respectively. These two RAG methods are denoted
byST-Leiden-RAG andHIT-Leiden-RAG , respectively. Note that
ND-Leiden ,DS-Leiden , and DF-Leiden are not fit to maintain the
hierarchical communities of Graph-RAG since they lack hierarchi-
cal maintenance. We report their runtime, token cost, and accuracy
in Figure 14(d)-(f). Clearly, HIT-Leiden-RAG is56.1×faster than
ST-Leiden-RAG . Moreover, it significantly reduces the summary
token cost while preserving downstream QA accuracy, since its
token cost is only 0.8% of the token cost of ST-Leiden-RAG . Hence,
HIT-Leiden is effective for supporting Graph-RAG on a dynamic
corpus.
6.3 Efficiency evaluation
In this section, we first present the overall efficiency results, then
analyze the time cost of each component, and finally evaluate the
effects of some hyperparameters.
•Overall results.Figure 15 presents the overall efficiency re-
sults where 𝑏is set to its default value1 ,000. Clearly, HIT-Leiden
achieves the best efficiency on datasets, especially on the it-2004
dataset, since it is up to three orders of magnitude faster than the
state-of-the-art algorithms. That is mainly because the ratio of
updated edges to total edges in it-2004 is larger than those in
dblp-coauthor,yahoo-song, andsx-stackoverflow.
•Time cost of different components in HIT-Leiden .There
are three key components, i.e., inc-movement ,inc-refine , and
inc-aggregation , inHIT-Leiden . We evaluate the proportion of
time cost for each component and present the results in Figure 16.
Note that some operations (e.g., def-update inHIT-Leiden ) may
not be included by the above three components, so we put them into
the "Others" component. Notably, in HIT-Leiden , the refinement
phase contributes minimally to the overall runtime. Besides, the
combined proportion of time spent in its movement and aggregation
phase is comparable to that of other algorithms. Inc-movement ,
inc-refinement , and inc-aggregation consistently outperform
their counterparts in other algorithms across all datasets, achieving
lower absolute runtime costs according to Figure 15.
•Effect of𝑏.We vary the batch size 𝑏∈{ 10,102,103,104,105}
and report the efficiency in Figure 17. We see that HIT-Leiden is
up to five orders of magnitude faster than other algorithms. Also, it
exhibits a notable increase as 𝑏becomes smaller because it is a rela-
tively bounded algorithm. In contrast, ND-Leiden ,DS-Leiden , and
DF-Leidenstill need to process the entire graph when processing
a new batch.
•Effect of𝑟.Recall that after fixing the batch size 𝑏, we update
the graph for 𝑟batches. Figure 18 shows the efficiency, where 𝑏is

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden
1011021031041050.750.760.770.78
batch sizeModularity
(a)DC1011021031041050.3600.3650.370
batch size
(b)YS1011021031041050.4450.4500.455
batch size
(c)SS1011021031041050.9710.9720.973
batch size
(d)IT1011021031041050.3550.3600.365
batch size
(e)RS
Figure 11: Modularity values on dynamic graphs.
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden
01234567890.740.750.760.770.78
batchModularity
(a)DC01234567890.3600.3650.370
batch
(b)YS01234567890.4450.4500.455
batch
(c)SS01234567890.9710.9720.973
batch
(d)IT01234567890.3550.3600.365
batch
(e)RS
Figure 12: Modularity changes w.r.t. the number of update batches.
DC YS SS IT RS979899100% communityST-Leiden ND-Leiden DS-Leiden
DF-Leiden HIT-Leiden
Figure 13: Percentage of subpartition 𝛾-dense communities.
ST-Leiden ND-Leiden DS-Leiden
DF-Leiden HIT-Leiden ST-Leiden-RAG
HIT-Leiden-RAG
0 250 500 750 9990.40.420.440.460.480.5
batchModularity
(a) Modularity0 250 500 750 99999.699.8100
batch% Community
(b) Subpartition𝛾-density
0 250 500 750 999101102103104105
batchRuntime (ms)
(c) Runtime0123456789105106107108
batchRuntime (ms)
(d) Runtime
0123456789104105106107108109
batch# of tokens
(e) Token cost01234567890.50.520.540.560.580.6
batchAccuracy
(f) Accuracy
Figure 14: Subfigures (a)–(c) show the effectiveness of HIT-
Leiden over 999 update batches, and subfigures (d)–(f)
compare ST-Leiden-RAG and HIT-Leiden-RAG over 9 update
batches.DC YS SS IT RS102105108Runtime (ms)ST-Leiden ND-Leiden DS-Leiden
DF-Leiden HIT-Leiden
Figure 15: Efficiency of all Leiden algorithms on all datasets.
fixed as 1,000, but 𝑟ranges from 1 to 9. We observe that the incre-
mental speedup is limited in the first few batches because 𝑃= 10is
small, and additional iterations may slightly improve the commu-
nity membership. As a result, all the maintenance algorithms often
require more time for the second batch to adjust the community
structure. Once high-quality community structure is established,
the speedup becomes significant. In addition, HIT-Leiden incurs a
slightly higher runtime to record more information and construct
the CC-index.
7 Conclusions
In this paper, we develop an efficient algorithm for maintaining Lei-
den communities in a dynamic graph. We first theoretically analyze
the boundedness of existing algorithms and how supervertex behav-
iors affect community membership under graph update. Building
on these analyses, we further develop a relative boundedness algo-
rithm, called HIT-Leiden , which consists of three key components,
i.e.,inc-movement ,inc-refinement , and inc-aggregation . Ex-
tensive experiments on five real-world dynamic graphs show that
HIT-Leiden not only preserves the properties of Leiden and achieves
comparable modularity quality with Leiden, but also runs faster
than state-of-the-art competitors. In future work, we will extend
our algorithm to handle directed graphs and also evaluate it in a
distributed environment.
References
[1]2020. A single-cell transcriptomic atlas characterizes ageing tissues in the mouse.
Nature583, 7817 (2020), 590–595.
[2]Edo M Airoldi, David Blei, Stephen Fienberg, and Eric Xing. 2008. Mixed
membership stochastic blockmodels.Advances in neural information processing
systems21 (2008).

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
Inc-movement Inc-refinement Inc-aggregation Others
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden020406080100Time proportion (%)
(a)DCST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden020406080100
(b)YSST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden020406080100
(c)SSST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden020406080100
(d)ITST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden020406080100
(e)RS
Figure 16: Proportion of time cost of each component for the Leiden algorithms on all datasets.
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden
101102103104105102103104105
batch sizeRuntime (ms)
(a)DC101102103104105101102103104105106
batch size
(b)YS101102103104105101103105
batch size
(c)SS101102103104105100102104106
batch size
(d)IT101102103104105103105107
batch size
(e)RS
Figure 17: Runtime on dynamic graphs.
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden
0123456789102103104105
batchRuntime (ms)
(a)DC0123456789103105107
batch
(b)YS0123456789101103105107
batch
(c)SS0123456789101103105107
batch
(d)IT0123456789104105106107108
batch
(e)RS
Figure 18: Runtime w.r.t. the number of update batches.
[3]Arash A Amini, Aiyou Chen, Peter J Bickel, and Elizaveta Levina. 2013. Pseudo-
likelihood methods for community detection in large sparse networks. (2013).
[4]Abdelouahab Amira, Abdelouahid Derhab, Elmouatez Billah Karbab, and Omar
Nouali. 2023. A survey of malware analysis using community detection algo-
rithms.Comput. Surveys56, 2 (2023), 1–29.
[5]LN Fred Ana and Anil K Jain. 2003. Robust data clustering. In2003 IEEE Computer
Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings.,
Vol. 2. IEEE, II–II.
[6]Thomas Aynaud and Jean-Loup Guillaume. 2010. Static community detection
algorithms for evolving networks. In8th international symposium on modeling
and optimization in mobile, ad hoc, and wireless networks. IEEE, 513–519.
[7]Thomas Aynaud and Jean-Loup Guillaume. 2011. Multi-step community detec-
tion and hierarchical time segmentation in evolving networks. InProceedings of
the 5th SNA-KDD workshop, Vol. 11.
[8]Trygve E Bakken, Nikolas L Jorstad, Qiwen Hu, Blue B Lake, Wei Tian, Brian E
Kalmbach, Megan Crow, Rebecca D Hodge, Fenna M Krienen, Staci A Sorensen,
et al.2021. Comparative cellular analysis of motor cortex in human, marmoset
and mouse.Nature598, 7879 (2021), 111–119.
[9]Vandana Bhatia and Rinkle Rani. 2018. Dfuzzy: a deep learning-based fuzzy
clustering model for large graphs.Knowledge and Information Systems57 (2018),
159–181.
[10] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefeb-
vre. 2008. Fast unfolding of communities in large networks.Journal of statistical
mechanics: theory and experiment2008, 10 (2008), P10008.
[11] Stefan Boettcher and Allon G Percus. 2002. Optimization with extremal dynam-
ics.complexity8, 2 (2002), 57–62.
[12] Biao Cai, Yanpeng Wang, Lina Zeng, Yanmei Hu, and Hongjun Li. 2020. Edge
classification based on convolutional neural networks for community detection
in complex network.Physica A: statistical mechanics and its applications556
(2020), 124826.
[13] Tanmoy Chakraborty, Ayushi Dalmia, Animesh Mukherjee, and Niloy Ganguly.
2017. Metrics for community analysis: A survey.ACM Computing Surveys
(CSUR)50, 4 (2017), 1–37.
[14] Qing Chen, Sven Helmer, Oded Lachish, and Michael Bohlen. 2022. Dynamic
spanning trees for connectivity queries on fully-dynamic undirected graphs.
(2022).[15] Jiafeng Cheng, Qianqian Wang, Zhiqiang Tao, Deyan Xie, and Quanxue Gao.
2021. Multi-view attribute graph convolution networks for clustering. InProceed-
ings of the twenty-ninth international conference on international joint conferences
on artificial intelligence. 2973–2979.
[16] Yun Chi, Xiaodan Song, Dengyong Zhou, Koji Hino, and Belle L Tseng. 2007.
Evolutionary spectral clustering by incorporating temporal smoothness. In
Proceedings of the 13th ACM SIGKDD international conference on Knowledge
discovery and data mining. 153–162.
[17] Yun Chi, Xiaodan Song, Dengyong Zhou, Koji Hino, and Belle L Tseng. 2009.
On evolutionary spectral clustering.ACM Transactions on Knowledge Discovery
from Data (TKDD)3, 4 (2009), 1–30.
[18] Wen Haw Chong and Loo Nin Teow. 2013. An incremental batch technique
for community detection. InProceedings of the 16th international conference on
information fusion. IEEE, 750–757.
[19] Aaron Clauset, Mark EJ Newman, and Cristopher Moore. 2004. Finding commu-
nity structure in very large networks.Physical Review E—Statistical, Nonlinear,
and Soft Matter Physics70, 6 (2004), 066111.
[20] Mário Cordeiro, Rui Portocarrero Sarmento, and Joao Gama. 2016. Dynamic com-
munity detection in evolving networks using locality modularity optimization.
Social Network Analysis and Mining6 (2016), 1–20.
[21] Ganqu Cui, Jie Zhou, Cheng Yang, and Zhiyuan Liu. 2020. Adaptive graph
encoder for attributed graph embedding. InProceedings of the 26th ACM SIGKDD
international conference on knowledge discovery & data mining. 976–985.
[22] Siemon C de Lange, Marcel A de Reus, and Martijn P van den Heuvel. 2014. The
Laplacian spectrum of neural networks.Frontiers in computational neuroscience
7 (2014), 189.
[23] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al .1996. A density-
based algorithm for discovering clusters in large spatial databases with noise.
Inkdd, Vol. 96. 226–231.
[24] Shaohua Fan, Xiao Wang, Chuan Shi, Emiao Lu, Ken Lin, and Bai Wang. 2020.
One2multi graph autoencoder for multi-view graph clustering. Inproceedings
of the web conference 2020. 3070–3076.
[25] Wenfei Fan, Chunming Hu, and Chao Tian. 2017. Incremental graph compu-
tations: Doable and undoable. InProceedings of the 2017 ACM International
Conference on Management of Data. 155–169.
[26] Xinyu Fu, Jiani Zhang, Ziqiao Meng, and Irwin King. 2020. Magnn: Metap-
ath aggregated graph neural network for heterogeneous graph embedding. In
Proceedings of the web conference 2020. 2331–2341.

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
[27] László Gadár and János Abonyi. 2024. Explainable prediction of node labels
in multilayer networks: a case study of turnover prediction in organizations.
Scientific Reports14, 1 (2024), 9036.
[28] Michael S Haney, Róbert Pálovics, Christy Nicole Munson, Chris Long, Pa-
trik K Johansson, Oscar Yip, Wentao Dong, Eshaan Rawat, Elizabeth West,
Johannes CM Schlachetzki, et al .2024. APOE4/4 is linked to damaging lipid
droplets in Alzheimer’s disease microglia.Nature628, 8006 (2024), 154–161.
[29] Paul W Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. 1983. Sto-
chastic blockmodels: First steps.Social networks5, 2 (1983), 109–137.
[30] Jacob Holm, Kristian De Lichtenberg, and Mikkel Thorup. 2001. Poly-logarithmic
deterministic fully-dynamic algorithms for connectivity, minimum spanning
tree, 2-edge, and biconnectivity.Journal of the ACM (JACM)48, 4 (2001), 723–
760.
[31] Ruiqi Hu, Shirui Pan, Guodong Long, Qinghua Lu, Liming Zhu, and Jing Jiang.
2020. Going deep: Graph convolutional ladder-shape networks. InProceedings
of the AAAI Conference on Artificial Intelligence, Vol. 34. 2838–2845.
[32] Xiao Huang, Jundong Li, and Xia Hu. 2017. Accelerated attributed network
embedding. InProceedings of the 2017 SIAM international conference on data
mining. SIAM, 633–641.
[33] Yuting Jia, Qinqin Zhang, Weinan Zhang, and Xinbing Wang. 2019. Commu-
nitygan: Community detection with generative adversarial nets. InThe world
wide web conference. 784–794.
[34] Baoyu Jing, Chanyoung Park, and Hanghang Tong. 2021. Hdmi: High-order
deep multiplex infomax. InProceedings of the web conference 2021. 2414–2424.
[35] Ravi Kannan, Santosh Vempala, and Adrian Vetta. 2004. On clusterings: Good,
bad and spectral.Journal of the ACM (JACM)51, 3 (2004), 497–515.
[36] Brian Karrer and Mark EJ Newman. 2011. Stochastic blockmodels and commu-
nity structure in networks.Physical Review E—Statistical, Nonlinear, and Soft
Matter Physics83, 1 (2011), 016107.
[37] Scott Kirkpatrick, C Daniel Gelatt Jr, and Mario P Vecchi. 1983. Optimization
by simulated annealing.science220, 4598 (1983), 671–680.
[38] Sadamori Kojaku, Giacomo Livan, and Naoki Masuda. 2021. Detecting anoma-
lous citation groups in journal networks.Scientific Reports11, 1 (2021), 14524.
[39] Andrea Lancichinetti and Santo Fortunato. 2009. Community detection algo-
rithms: a comparative analysis.Physical Review E—Statistical, Nonlinear, and
Soft Matter Physics80, 5 (2009), 056117.
[40] Ron Levie, Federico Monti, Xavier Bresson, and Michael M Bronstein. 2018. Cay-
leynets: Graph convolutional neural networks with complex rational spectral
filters.IEEE Transactions on Signal Processing67, 1 (2018), 97–109.
[41] Bentian Li, Dechang Pi, Yunxia Lin, and Lin Cui. 2021. DNC: A deep neural
network-based clustering-oriented network embedding algorithm.Journal of
Network and Computer Applications173 (2021), 102854.
[42] Zhangtao Li and Jing Liu. 2016. A multi-agent genetic algorithm for commu-
nity detection in complex networks.Physica A: Statistical Mechanics and its
Applications449 (2016), 336–347.
[43] Xujian Liang and Zhaoquan Gu. 2025. Fast think-on-graph: Wider, deeper and
faster reasoning of large language model on knowledge graph. InProceedings of
the AAAI Conference on Artificial Intelligence, Vol. 39. 24558–24566.
[44] Chunxu Lin, YiXiang Fang, Yumao Xie, Yongming Hu, Yingqian Hu, and Chen
Cheng. 2025. Efficient Maintenance of Leiden Communities in Large Dynamic
Graphs (full version). https://anonymous.4open.science/r/HIT_Leiden-2DC1.
[45] Yu-Ru Lin, Yun Chi, Shenghuo Zhu, Hari Sundaram, and Belle L Tseng. 2008.
Facetnet: a framework for analyzing communities and their evolutions in dy-
namic networks. InProceedings of the 17th international conference on World
Wide Web. 685–694.
[46] Rik GH Lindeboom, Kaylee B Worlock, Lisa M Dratva, Masahiro Yoshida, David
Scobie, Helen R Wagstaffe, Laura Richardson, Anna Wilbrey-Clark, Josephine L
Barnes, Lorenz Kretschmer, et al .2024. Human SARS-CoV-2 challenge uncovers
local and systemic response dynamics.Nature631, 8019 (2024), 189–198.
[47] Monika Litviňuková, Carlos Talavera-López, Henrike Maatz, Daniel Reichart,
Catherine L Worth, Eric L Lindberg, Masatoshi Kanda, Krzysztof Polanski,
Matthias Heinig, Michael Lee, et al .2020. Cells of the adult human heart.Nature
588, 7838 (2020), 466–472.
[48] Fanzhen Liu, Zhao Li, Baokun Wang, Jia Wu, Jian Yang, Jiaming Huang, Yiqing
Zhang, Weiqiang Wang, Shan Xue, Surya Nepal, et al .2022. eRiskCom: an e-
commerce risky community detection platform.The VLDB Journal31, 5 (2022),
1085–1101.
[49] Fanzhen Liu, Jia Wu, Chuan Zhou, and Jian Yang. 2019. Evolutionary community
detection in dynamic social networks. In2019 International Joint Conference on
Neural Networks (IJCNN). IEEE, 1–7.
[50] Yanbei Liu, Xiao Wang, Shu Wu, and Zhitao Xiao. 2020. Independence promoted
graph disentangled networks. InProceedings of the AAAI Conference on Artificial
Intelligence, Vol. 34. 4916–4923.
[51] Linhao Luo, Yixiang Fang, Xin Cao, Xiaofeng Zhang, and Wenjie Zhang. 2021.
Detecting communities from heterogeneous graphs: A context path-based graph
neural network model. InProceedings of the 30th ACM international conference
on information & knowledge management. 1170–1180.[52] Aaron F McDaid, Derek Greene, and Neil Hurley. 2011. Normalized mutual
information to evaluate overlapping community finding algorithms.arXiv
preprint arXiv:1110.2515(2011).
[53] Xiangfeng Meng, Yunhai Tong, Xinhai Liu, Shuai Zhao, Xianglin Yang, and
Shaohua Tan. 2016. A novel dynamic community detection algorithm based on
modularity optimization. In2016 7th IEEE international conference on software
engineering and service science (ICSESS). IEEE, 72–75.
[54] Microsoft. 2025. GraphRAG: A Structured, Hierarchical Approach to Retrieval
Augmented Generation. https://microsoft.github.io/graphrag/. Accessed: 2025-
03-31.
[55] Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, WA Redmond, Hiteshi
Sharma, Robert Ness, Nebojsa Jojic, Hamid Palangi, and Jonathan Larson. [n. d.].
Evaluating Cognitive Maps and Planning in Large Language Models with Co-
gEval (Supplementary Materials). ([n. d.]).
[56] Mark EJ Newman. 2004. Fast algorithm for detecting community structure in
networks.Physical Review E—Statistical, Nonlinear, and Soft Matter Physics69, 6
(2004), 066133.
[57] Mark EJ Newman. 2006. Finding community structure in networks using the
eigenvectors of matrices.Physical Review E—Statistical, Nonlinear, and Soft
Matter Physics74, 3 (2006), 036104.
[58] Mark EJ Newman. 2006. Modularity and community structure in networks.
Proceedings of the national academy of sciences103, 23 (2006), 8577–8582.
[59] Mark EJ Newman. 2013. Spectral methods for community detection and graph
partitioning.Physical Review E—Statistical, Nonlinear, and Soft Matter Physics
88, 4 (2013), 042822.
[60] Mark EJ Newman and Michelle Girvan. 2004. Finding and evaluating community
structure in networks.Physical review E69, 2 (2004), 026113.
[61] Nam P Nguyen, Thang N Dinh, Sindhura Tokala, and My T Thai. 2011. Overlap-
ping communities in dynamic networks: their detection and mobile applications.
InProceedings of the 17th annual international conference on Mobile computing
and networking. 85–96.
[62] Nam P Nguyen, Thang N Dinh, Ying Xuan, and My T Thai. 2011. Adaptive
algorithms for detecting community structure in dynamic social networks. In
2011 Proceedings IEEE INFOCOM. IEEE, 2282–2290.
[63] Alexandru Oarga, Matthew Hart, Andres M Bran, Magdalena Lederbauer, and
Philippe Schwaller. 2024. Scientific knowledge graph and ontology generation
using open large language models. InAI for Accelerated Materials Design-NeurIPS
2024.
[64] Shashank Pandit, Duen Horng Chau, Samuel Wang, and Christos Faloutsos.
2007. Netprobe: a fast and scalable system for fraud detection in online auction
networks. InProceedings of the 16th international conference on World Wide Web.
201–210.
[65] Songtao Peng, Jiaqi Nie, Xincheng Shu, Zhongyuan Ruan, Lei Wang, Yunxuan
Sheng, and Qi Xuan. 2022. A multi-view framework for BGP anomaly detection
via graph attention network.Computer Networks214 (2022), 109129.
[66] Ganesan Ramalingam and Thomas Reps. 1996. On the computational complexity
of dynamic graph problems.Theoretical Computer Science158, 1-2 (1996), 233–
277.
[67] Jörg Reichardt and Stefan Bornholdt. 2006. Statistical mechanics of community
detection.Physical Review E—Statistical, Nonlinear, and Soft Matter Physics74, 1
(2006), 016110.
[68] Boyu Ruan, Junhao Gan, Hao Wu, and Anthony Wirth. 2021. Dynamic structural
clustering on graphs. InProceedings of the 2021 International Conference on
Management of Data. 1491–1503.
[69] Subhajit Sahu. 2024. DF Louvain: Fast Incrementally Expanding Approach for
Community Detection on Dynamic Graphs.arXiv preprint arXiv:2404.19634
(2024).
[70] Subhajit Sahu. 2024. A Starting Point for Dynamic Community Detection with
Leiden Algorithm.arXiv preprint arXiv:2405.11658(2024).
[71] Subhajit Sahu, Kishore Kothapalli, and Dip Sankar Banerjee. 2024. Fast Leiden
Algorithm for Community Detection in Shared Memory Setting. InProceedings
of the 53rd International Conference on Parallel Processing. 11–20.
[72] Arindam Sarkar, Nikhil Mehta, and Piyush Rai. 2020. Graph representation
learning via ladder gamma variational autoencoders. InProceedings of the AAAI
Conference on Artificial Intelligence, Vol. 34. 5604–5611.
[73] Akrati Saxena, Yulong Pei, Jan Veldsink, Werner van Ipenburg, George Fletcher,
and Mykola Pechenizkiy. 2021. The banking transactions dataset and its com-
parative analysis with scale-free networks. InProceedings of the 2021 IEEE/ACM
International Conference on Advances in Social Networks Analysis and Mining.
283–296.
[74] Jiaxing Shang, Lianchen Liu, Xin Li, Feng Xie, and Cheng Wu. 2016. Targeted
revision: A learning-based approach for incremental community detection in
dynamic networks.Physica A: Statistical Mechanics and its Applications443
(2016), 70–85.
[75] Jiaxing Shang, Lianchen Liu, Feng Xie, Zhen Chen, Jiajia Miao, Xuelin Fang,
and Cheng Wu. 2014. A real-time detecting algorithm for tracking community
structure of dynamic networks.arXiv preprint arXiv:1407.2683(2014).

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
[76] Oleksandr Shchur and Stephan Günnemann. 2019. Overlapping community
detection with graph neural networks.arXiv preprint arXiv:1909.12201(2019).
[77] Stanislav Sobolevsky, Riccardo Campari, Alexander Belyi, and Carlo Ratti. 2014.
General optimization technique for high-quality community detection in com-
plex networks.Physical Review E90, 1 (2014), 012811.
[78] Xing Su, Shan Xue, Fanzhen Liu, Jia Wu, Jian Yang, Chuan Zhou, Wenbin Hu,
Cecile Paris, Surya Nepal, Di Jin, et al .2022. A comprehensive survey on
community detection with deep learning.IEEE transactions on neural networks
and learning systems35, 4 (2022), 4682–4702.
[79] Tencent. 2019.Tencent Graph Computing (TGraph) Officially Open Sourced
High-Performance Graph Computing Framework: Plato. Accessed: 2025-04-17.
[80] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to
Leiden: guaranteeing well-connected communities.Scientific reports9, 1 (2019),
1–12.
[81] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you
need.Advances in neural information processing systems30 (2017).
[82] Lewen Wang, Haozhe Zhao, Cunguang Feng, Weiqing Liu, Congrui Huang,
Marco Santoni, Manuel Cristofaro, Paola Jafrancesco, and Jiang Bian. 2023.
Removing camouflage and revealing collusion: Leveraging gang-crime pattern
in fraudster detection. InProceedings of the 29th ACM SIGKDD conference on
knowledge discovery and data mining. 5104–5115.
[83] Shu Wang, Yixiang Fang, and Wensheng Luo. 2025. Searching and Detect-
ing Structurally Similar Communities in Large Heterogeneous Information
Networks.Proceedings of the VLDB Endowment18, 5 (2025), 1425–1438.
[84] Xiao Wang, Nian Liu, Hui Han, and Chuan Shi. 2021. Self-supervised heteroge-
neous graph neural network with co-contrastive learning. InProceedings of the
27th ACM SIGKDD conference on knowledge discovery & data mining. 1726–1736.
[85] Wei Xia, Qianqian Wang, Quanxue Gao, Xiangdong Zhang, and Xinbo Gao.
2021. Self-supervised graph convolutional network for multi-view clustering.
IEEE Transactions on Multimedia24 (2021), 3182–3192.
[86] Jierui Xie, Mingming Chen, and Boleslaw K Szymanski. 2013. LabelrankT:
Incremental community detection in dynamic networks via label propagation.
InProceedings of the workshop on dynamic networks management and mining.
25–32.
[87] Jierui Xie and Boleslaw K Szymanski. 2013. Labelrank: A stabilized label propa-
gation algorithm for community detection in networks. In2013 IEEE 2nd Network
Science Workshop (NSW). IEEE, 138–143.
[88] Jierui Xie, Boleslaw K Szymanski, and Xiaoming Liu. 2011. Slpa: Uncovering
overlapping communities in social networks via a speaker-listener interaction
dynamic process. In2011 ieee 11th international conference on data mining
workshops. IEEE, 344–349.
[89] Yu Xie, Maoguo Gong, Shanfeng Wang, and Bin Yu. 2018. Community discovery
in networks with deep sparse filtering.Pattern Recognition81 (2018), 50–59.
[90] Lantian Xu, Dong Wen, Lu Qin, Ronghua Li, Ying Zhang, and Xuemin Lin. 2024.
Constant-time Connectivity Querying in Dynamic Graphs.Proceedings of the
ACM on Management of Data2, 6 (2024), 1–23.
[91] Rongbin Xu, Yan Che, Xinmei Wang, Jianxiong Hu, and Ying Xie. 2020. Stacked
autoencoder-based community detection method via an ensemble clustering
framework.Information sciences526 (2020), 151–165.
[92] Xiaowei Xu, Nurcan Yuruk, Zhidan Feng, and Thomas AJ Schweiger. 2007.
Scan: a structural clustering algorithm for networks. InProceedings of the 13th
ACM SIGKDD international conference on Knowledge discovery and data mining.
824–833.
[93] Liang Yang, Xiaochun Cao, Dongxiao He, Chuan Wang, Xiao Wang, and Weix-
iong Zhang. 2016. Modularity based community detection with deep learning..
InIJCAI, Vol. 16. 2252–2258.
[94] Zhao Yang, René Algesheimer, and Claudio J Tessone. 2016. A comparative
analysis of community detection algorithms on artificial networks.Scientific
reports6, 1 (2016), 30750.
[95] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for di-
verse, explainable multi-hop question answering.arXiv preprint arXiv:1809.09600
(2018).
[96] Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. 2016.
Image captioning with semantic attention. InProceedings of the IEEE conference
on computer vision and pattern recognition. 4651–4659.
[97] Neda Zarayeneh and Ananth Kalyanaraman. 2021. Delta-screening: a fast and
efficient technique to update communities in dynamic graphs.IEEE transactions
on network science and engineering8, 2 (2021), 1614–1629.
[98] Fangyuan Zhang and Sibo Wang. 2022. Effective indexing for dynamic structural
graph clustering.Proceedings of the VLDB Endowment15, 11 (2022), 2908–2920.
[99] Meng Zhang, Xingjie Pan, Won Jung, Aaron R Halpern, Stephen W Eichhorn,
Zhiyun Lei, Limor Cohen, Kimberly A Smith, Bosiljka Tasic, Zizhen Yao, et al .
2023. Molecularly defined and spatially resolved cell atlas of the whole mouse
brain.Nature624, 7991 (2023), 343–354.
[100] Tianqi Zhang, Yun Xiong, Jiawei Zhang, Yao Zhang, Yizhu Jiao, and Yangyong
Zhu. 2020. CommDGI: community detection oriented deep graph infomax. InProceedings of the 29th ACM international conference on information & knowledge
management. 1843–1852.
[101] Xiaotong Zhang, Han Liu, Xiao-Ming Wu, Xianchao Zhang, and Xinyue Liu.
2021. Spectral embedding network for attributed graph clustering.Neural
Networks142 (2021), 388–396.
[102] Yao Zhang, Yun Xiong, Yun Ye, Tengfei Liu, Weiqiang Wang, Yangyong Zhu,
and Philip S Yu. 2020. SEAL: Learning heuristics for community detection
with generative adversarial networks. InProceedings of the 26th ACM SIGKDD
international conference on knowledge discovery & data mining. 1103–1113.
[103] Han Zhao, Xu Yang, Zhenru Wang, Erkun Yang, and Cheng Deng. 2021. Graph
debiased contrastive learning with joint representation clustering.. InIJCAI.
3434–3440.
[104] Yingli Zhou, Qingshuo Guo, Yi Yang, Yixiang Fang, Chenhao Ma, and Laks
Lakshmanan. 2024. In-depth Analysis of Densest Subgraph Discovery in a
Unified Framework.arXiv preprint arXiv:2406.04738(2024).
[105] Di Zhuang, J Morris Chang, and Mingchen Li. 2019. DynaMo: Dynamic com-
munity detection by incrementally maximizing modularity.IEEE Transactions
on Knowledge and Data Engineering33, 5 (2019), 1934–1945.
Appendix
A Proof of lemmas
A.1 Proof of Lemma 2
Proof. We analyze the modularity gain Δ𝑀(𝑣→∅,𝛾) for any
vertex𝑣, which denotes the modularity gain of moving 𝑣from the
intermediate subsequence 𝐼to∅, whose calculation follows the
same formula as the standard modularity gain.
According to Definition 8, if Δ𝑀(𝑣→∅,𝛾)> 0, the intermediate
subsequence 𝐼could not be 𝛾-connected and 𝑣has to leave 𝐼. It is
different from maintaining vertex optimality (mentioned in Defi-
nition 6): If there exists a community 𝐶′such that the modularity
gain of moving 𝑣from its community 𝐶to𝐶′is positive,𝑣is not
locally optimized and has to be removed from𝐶.
Case 1:𝑣 𝑖is inserted into𝑆after𝑣 𝑗,i.e.,𝑣𝑗∈𝐼𝑖. The old mod-
ularity gain𝑀 𝑜𝑙𝑑(𝑣𝑖→∅,𝛾)<0before deletion is:
𝑀𝑜𝑙𝑑(𝑣𝑖→∅,𝛾)=−𝑤(𝑣𝑖,𝑈𝑖)
2𝑚+𝛾·𝑑(𝑣𝑖)·𝑑(𝑈𝑖)
4𝑚2≤0. (3)
Where𝑈𝑖=𝐼𝑖\{𝑣𝑖}. We multiply right side of Equation (3) by
4𝑚2and obtain𝑋(3):
𝑋(3)=−2𝑚·𝑤(𝑣 𝑖,𝑈𝑖)+𝛾·𝑑(𝑣 𝑖)·𝑑(𝑈𝑖)≤0(4)
After the deletion, the new modularity gain 𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾)
formulates:
Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾)=−𝑤(𝑣𝑖,𝑈𝑖)−2𝛼
2(𝑚−𝛼)
+𝛾·(𝑑(𝑣𝑖)−𝛼)·(𝑑(𝑈 𝑖)−𝛼)
4(𝑚−𝛼)2.(5)
We multiply right side of Equation (5) by4 (𝑚−𝛼)2and obtain
𝑌(5):
𝑌(5)=−2(𝑚−𝛼)·(𝑤(𝑣 𝑖,𝑈𝑖)−2𝛼)
+𝛾·(𝑑(𝑣 𝑖)−𝛼)·(𝑑(𝑈 𝑖)−𝛼)
=𝑋(3)+𝛼·(4𝑚+2𝑤(𝑣 𝑖,𝑈𝑖)−4𝑎−𝛾·(𝑑(𝐼 𝑖)−𝛼))
<𝑋(3)+𝛼·(4𝑚+2𝑤(𝑣 𝑖,𝑈𝑖))(6)
If𝑋(3)+𝛼·( 4𝑚+ 2𝑤(𝑣𝑖,𝑈𝑖))> 0,Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾) could
be positive; Otherwise, Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾) must be non-positive.
Therefore,𝑣𝑖could be removed from its sub-community only if
𝛼>2𝑚·𝑤(𝑣𝑖,𝑈𝑖)−𝛾·𝑑(𝑣𝑖)·𝑑(𝑈𝑖)
4𝑚+2𝑤(𝑣𝑖,𝑈𝑖).

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
Case 2:𝑣 𝑗is inserted into𝑆before𝑣 𝑖.In this case, we have
𝑣𝑗∈𝐼𝑗,𝑣𝑖∉𝐼𝑗, and the edge deletion does not affect intra-edges
within𝑈𝑗. The old modularity gain 𝑀𝑜𝑙𝑑(𝑣𝑖→∅,𝛾)< 0before
deletion is:
𝑀𝑜𝑙𝑑(𝑣𝑗→∅,𝛾)=−𝑤(𝑣𝑗,𝑈𝑗)
2𝑚+𝛾·𝑑(𝑣𝑗)·𝑑(𝑈𝑗)
4𝑚2. (7)
We multiply right side of Equation (3) by4𝑚2and obtain𝑋(3):
𝑋(7)=−2𝑚·𝑤(𝑣 𝑗,𝑈𝑗)+𝛾·𝑑(𝑣 𝑗)·𝑑(𝑈𝑗)<0(8)
The new modularity gain after the edge deletion becomes:
Δ𝑀𝑛𝑒𝑤(𝑣𝑗→∅,𝛾)=−𝑤(𝑣𝑗,𝑈𝑗)
2(𝑚−𝛼)
+𝛾·(𝑑(𝑣𝑗)−𝛼)·𝑑(𝑈 𝑗)
4(𝑚−𝛼)2(9)
We multiply right side of Equation (9) by4 (𝑚−𝛼)2and obtain
𝑌(9):
𝑌(9)=−2(𝑚−𝛼)·𝑤(𝑣 𝑗,𝑈𝑗)+𝛾·(𝑑(𝑣 𝑗)−𝛼)·𝑑(𝑈 𝑗)
=𝑋(7)+2𝛼·(︁𝑤(𝑣𝑗,𝑈𝑗)−𝛾·𝑑(𝑈 𝑗))︁
<𝑋(7)+2𝛼·𝑤(𝑣 𝑗,𝑈𝑗)(10)
Hence,𝑣𝑗could be removed from its sub-community only if
𝛼>𝑚−𝛾·𝑑(𝑣𝑗)·𝑑(𝑈𝑗)
2𝑤(𝑣𝑗.𝑈𝑗).
Generalization to other vertices. Consider other vertices 𝑣𝑘
and𝑣𝑙such that𝑣𝑘∈𝑆𝑖,𝑘≠𝑖,𝑗 and𝑣𝑙∉𝑆𝑖. The old modularity
gains𝑀𝑜𝑙𝑑(𝑣𝑘→∅,𝛾)< 0and𝑀𝑜𝑙𝑑(𝑣𝑙→∅,𝛾)< 0before deletion
are:
𝑀𝑜𝑙𝑑(𝑣𝑘→∅,𝛾)=−𝑤(𝑣𝑘,𝑈𝑘)
2𝑚+𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
4𝑚2. (11)
𝑀𝑜𝑙𝑑(𝑣𝑙→∅,𝛾)=−𝑤(𝑣𝑙,𝑈𝑙)
2𝑚+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
4𝑚2. (12)
We multiply right side of Equation (11) and (12) by4 𝑚2respec-
tively to obtain𝑋 (11)and𝑋(12):
𝑋(11)=−2𝑚·𝑤(𝑣 𝑘,𝑈𝑘)+𝛾·𝑑(𝑣 𝑘)·𝑑(𝑈𝑘)≤0(13)
𝑋(12)=−2𝑚·𝑤(𝑣 𝑙,𝑈𝑙)+𝛾·𝑑(𝑣 𝑙)·𝑑(𝑈𝑙)≤0(14)
After the edge deletion, their new modularity gains are satisfied:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)≤−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
4(𝑚−𝛼)2. (15)
Δ𝑀𝑛𝑒𝑤(𝑣𝑙→∅,𝛾)=−𝑤(𝑣𝑙,𝑈𝑙)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
4(𝑚−𝛼)2.(16)
𝑣𝑘could be merged before 𝑣𝑖and𝑣𝑗, between𝑣𝑖and𝑣𝑗, as well
as after𝑣𝑖and𝑣𝑗.Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾) can be formulated as follows:
(1)𝑣𝑘is merged before𝑣 𝑖and𝑣𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)=−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
4(𝑚−𝛼)2;(17)
(2)𝑣𝑘is merged between𝑣 𝑖and𝑣𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)=−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑘)·(𝑑(𝑈𝑘)−𝛼)
4(𝑚−𝛼)2;(18)(3)𝑣𝑘is merged after𝑣 𝑖and𝑣𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)=−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑘)·(𝑑(𝑈𝑘)−2𝛼)
4(𝑚−𝛼)2.(19)
Therefore, the equivalent of Equation (15) holds if and only if 𝑣𝑘
is merged before 𝑣𝑖and𝑣𝑗. Then, We multiply right side of Equation
(15) and (16) by4(𝑚−𝛼)2respectively and obtain𝑌 (15)and𝑌(16):
𝑌(15)=−2(𝑚−𝛼)·𝑤(𝑣 𝑘,𝑈𝑘)
+𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
=𝑋 13+2𝛼·𝑤(𝑣 𝑘,𝑈𝑘),(20)
𝑌(16)=𝑋 14+2𝛼·𝑤(𝑣 𝑙,𝑈𝑙), (21)
Only if𝛼>𝑚−𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
2𝑤(𝑣𝑘,𝑈𝑘),𝑣𝑘could be removed from its sub-
community; 𝑣𝑙should be removed from its sub-community if and
only if𝛼>𝑚−𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
2𝑤(𝑣𝑙,𝑈𝑙).
□
A.2 Proof of Lemma 3
Proof. We adopt the same notations as in the proof of Lemma
2, with the exception that 𝑣𝑘now denotes a vertex residing in the
same sub-community as either 𝑣𝑖or𝑣𝑗. Based on this setup, the
modularity gain after the edge deletion is shown as follows.
Case 1: Consider the endpoint𝑣 𝑖:
Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾)=−𝑤(𝑣𝑖,𝑈𝑖)
2(𝑚−𝛼)
+𝛾·(𝑑(𝑣𝑖)−𝛼)·𝑑(𝑈 𝑖)
4(𝑚−𝛼)2.(22)
We multiply right side of Equation (22) by4 (𝑚−𝛼)2and obtain
𝑌(22):
𝑌(22)=−2(𝑚−𝛼)·𝑤(𝑣 𝑖,𝑈𝑖)
+𝛾·(𝑑(𝑣 𝑖)−𝛼)·𝑑(𝑈 𝑖)
=𝑋(3)+𝛼·(2𝑤(𝑣 𝑖,𝑈𝑖)−𝛾·𝑑(𝑈 𝑖))
<𝑋(3)+𝛼·2𝑤(𝑣 𝑖,𝑈𝑖)(23)
Only if𝛼>𝑚−𝛾·𝑑(𝑣𝑖)·𝑑(𝑈𝑖)
2𝑤(𝑣𝑖,𝑈𝑖),𝑣𝑖could be removed from its sub-
community.𝑣 𝑗holds similar behavior.
Case 2: Consider the vertex𝑣 𝑘∈𝑆𝑖∪𝑆𝑗,𝑘≠𝑖,𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)≤−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
4(𝑚−𝛼)2. (24)
For Equation (24), 𝑣𝑘could be merged before 𝑣𝑖or𝑣𝑗, as well as
after𝑣𝑖or𝑣𝑗. Its equivalent holds if and only if 𝑣𝑘is merged before
𝑣𝑖or𝑣𝑗. We multiply right side of Equation (24) by4 (𝑚−𝛼)2and
obtain𝑌(24):
𝑌(24)=−2(𝑚−𝛼)·𝑤(𝑣 𝑘,𝑈𝑘)+𝛾·𝑑(𝑣 𝑘)·𝑑(𝑈𝑘)
=𝑋(11)+2𝛼·𝑤(𝑣 𝑘,𝑈𝑘)(25)
Only if𝛼>𝑚−𝛾·𝑑(𝑣𝑘)·𝑑(𝑈𝑘)
2𝑤(𝑣𝑘,𝑈𝑘),𝑣𝑘could be removed from its
sub-community.
Case 3: Consider the vertex𝑣 𝑙∉𝑆𝑖∪𝑆𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑙→∅,𝛾)=−𝑤(𝑣𝑙,𝑈𝑙)
2(𝑚−𝛼)+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
4(𝑚−𝛼)2. (26)

Efficient Maintenance of Leiden Communities in Large Dynamic Graphs SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India
Similar toCase 2, if and only if 𝛼>𝑚−𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
2𝑤(𝑣𝑙,𝑈𝑙),𝑣𝑙should
be removed from its sub-community.
□
A.3 Proof of Lemma 4
Proof. First, we analyze theinsertion of intra-sub-community
edges. We adopt the same notations as in the proof of Lemma 2.
Based on this setup, the modularity gain after the edge insertion is
shown as follows.
Case 1: Consider the endpoint𝑣 𝑖, which is the latter merged
endpoint:
Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾)=−𝑤(𝑣𝑖,𝑈𝑖)+2𝛼
2(𝑚+𝛼)
+𝛾·(𝑑(𝑣𝑖)+𝛼)·(𝑑(𝑈 𝑖)+𝛼)
4(𝑚+𝛼)2.(27)
We multiply right side of Equation (27) by4 (𝑚+𝛼)2and obtain
𝑌(27):
𝑌(27)=−2(𝑚+𝛼)(𝑤(𝑣 𝑖,𝑈𝑖)+2𝛼)
+𝛾·(𝑑(𝑣𝑖)+𝛼)·(𝑑(𝑈 𝑖)+𝛼)
=𝑋(3)+𝛼·(︁𝛾·(𝑑(𝐼𝑖)+𝛼)−2𝑤(𝑣 𝑖,𝑈𝑖)−4𝛼−4𝑚)︁
<𝑋(3)+𝛼·(︁𝛾·(𝑑(𝐼𝑖)+𝛼)−4𝑚)︁(28)
Obviously, only if 𝛾·(𝑑(𝐼𝑖)+𝛼)− 4𝑚> 0, i.e.,𝛼>4
𝛾𝑚−𝑑(𝐼𝑖),
𝑌(27)could be positive.
Case 2: Consider the endpoint𝑣 𝑗, which is the former merged
endpoint:
Δ𝑀𝑛𝑒𝑤(𝑣𝑗→∅,𝛾)=−𝑤(𝑣𝑗,𝑈𝑖)
2(𝑚+𝛼)
+𝛾·(𝑑(𝑣𝑗)+𝛼)·𝑑(𝑈 𝑖)
4(𝑚+𝛼)2.(29)
We multiply right side of Equation (29) by4 (𝑚+𝛼)2and obtain
𝑌(29):
𝑌(29)=−2(𝑚+𝛼)·𝑤(𝑣 𝑗,𝑈𝑗)
+𝛾·(𝑑(𝑣 𝑗)+𝛼)·𝑑(𝑈 𝑗)
=𝑋(7)+𝛼·(𝛾·𝑑(𝑈 𝑗)−𝑤(𝑣𝑗,𝑈𝑗))
<𝑋(7)+𝛼·𝛾·𝑑(𝑈 𝑗)(30)
Only if𝛼>2𝑤(𝑣𝑗,𝑈𝑗)
𝛾·𝑑(𝑈𝑗)·𝑚−𝑑(𝑣 𝑗),𝑣𝑗could be removed from its
sub-community.
Case 3: Consider other vertex𝑣 𝑘∈𝑆𝑖,𝑘≠𝑖,𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)≤−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚+𝛼)
+𝛾·𝑑(𝑣𝑘)·(𝑑(𝑈𝑘)+2𝛼)
4(𝑚+𝛼)2.(31)
The equivalent of Equation (31) holds if and only if 𝑣𝑘is merged
after𝑣𝑖and𝑣𝑗. We multiply right side of Equation (31) by4 (𝑚+𝛼)2
and obtain𝑌(31):
𝑌(31)=−2(𝑚+𝛼)·𝑤(𝑣 𝑘,𝑈𝑘)
+𝛾·𝑑(𝑣𝑘)·(𝑑(𝑈𝑘)+2𝛼)
=𝑋(11)+𝛼·(︁2𝛾·𝑑(𝑣𝑘)−2𝑤(𝑣𝑘,𝑈𝑘))︁
<𝑋(11)+2𝛼·𝛾·𝑑(𝑣 𝑘)(32)Only if𝛼>𝑤(𝑣𝑘,𝑈𝑘)
𝛾·𝑑(𝑣𝑘)·𝑚−1
2𝑑(𝑈𝑘),𝑣𝑘could be removed from its
sub-community.
Case 4: Consider other vertex𝑣 𝑙∉𝑆𝑖:
Δ𝑀𝑛𝑒𝑤(𝑣𝑙→∅,𝛾)≤−𝑤(𝑣𝑙,𝑈𝑙)
2(𝑚+𝛼)
+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
4(𝑚+𝛼)2.(33)
Equation (33) holds if and only if 𝑣𝑗is merged after 𝑣𝑖and𝑣𝑗. We
multiply right side of Equation (33) by4(𝑚+𝛼)2and obtain𝑌(31):
𝑌(33)=−2(𝑚+𝛼)·𝑤(𝑣 𝑙,𝑈𝑙)
+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
=𝑋(12)−2𝛼·𝑤(𝑣 𝑙,𝑈𝑙)<0(34)
𝑣𝑙is not affected by the intra-sub-community insertion.
Now, we consider theinsertion of cross-sub-community
edges. We adopt the same notations as in the proof of Lemma
3. Based on this setup, the modularity gain after the edge insertion
is shown as follows.
Case 5: Consider the endpoint𝑣 𝑖:
Δ𝑀𝑛𝑒𝑤(𝑣𝑖→∅,𝛾)=−𝑤(𝑣𝑖,𝑈𝑖)
2(𝑚+𝛼)
+𝛾·(𝑑(𝑣𝑖)+𝛼)·𝑑(𝑈 𝑖)
4(𝑚+𝛼)2.(35)
We multiply right side of Equation (35) by4 (𝑚+𝛼)2and obtain
𝑌(35):
𝑌(35)=−2(𝑚+𝛼)·𝑤(𝑣 𝑖,𝑈𝑖)
+𝛾·(𝑑(𝑣 𝑖)+𝛼)·𝑑(𝑈 𝑖)
=𝑋(3)+𝛼·(︁𝛾·𝑑(𝑈𝑖)−2𝑤(𝑣𝑖,𝑈𝑖))︁
<𝑋(3)+𝛼·𝛾·𝑑(𝑈 𝑖)(36)
Only if𝛼>2𝑤(𝑣𝑖,𝑈𝑖)
𝛾·𝑑(𝑈𝑖)·𝑚−𝑑(𝑣 𝑖),𝑣𝑖could be removed from its
sub-community.𝑣 𝑗is the same.
Case 6: Consider other vertex𝑣 𝑘∈𝑆𝑖∪𝑆𝑗,𝑘≠𝑖,𝑗:
Δ𝑀𝑛𝑒𝑤(𝑣𝑘→∅,𝛾)≤−𝑤(𝑣𝑘,𝑈𝑘)
2(𝑚+𝛼)
+𝛾·𝑑(𝑣𝑘)·(𝑑(𝑈𝑘)+𝛼)
4(𝑚+𝛼)2.(37)
The equivalent of Equation (37) holds if and only if 𝑣𝑘is merged
after𝑣𝑖or𝑣𝑗. We multiply right side of Equation (37) by4 (𝑚+𝛼)2
and obtain𝑌(37):
𝑌(37)=−2(𝑚+𝛼)·𝑤(𝑣 𝑖,𝑈𝑖)
+𝛾·𝑑(𝑣𝑖)·(𝑑(𝑈𝑖)+𝛼)
=𝑋(3)+𝛼·(︁𝛾·𝑑(𝑣𝑖)−2𝑤(𝑣𝑖,𝑈𝑖))︁
<𝑋(3)+𝛼·𝛾·𝑑(𝑣 𝑖)
<𝑋(3)+2𝛼·𝛾·𝑑(𝑣 𝑖)(38)
𝑣𝑘could be removed from its sub-community only if 𝛼>𝑤(𝑣𝑘,𝑈𝑘)
𝛾𝑑(𝑣𝑘)·
𝑚−1
2𝑑(𝑈𝑘).
Case 7: Consider other vertex𝑣 𝑙∉𝑆𝑖:

SIGMOD ’26, May 03–June 05, 2026, Bengaluru, India Lin et al.
ST-Leiden ND-Leiden DS-Leiden DF-Leiden HIT-Leiden
0.5 2 8 320.650.70.750.8
𝛾Runtime (ms)
(a)DC0.5 2 8 3200.10.20.30.4
𝛾
(b)YS0.5 2 8 320.10.20.30.40.5
𝛾
(c)SS0.5 2 8 320.960.981
𝛾
(d)IT0.5 2 8 320.30.320.340.360.380.4
𝛾
(e)RS
Figure 19: Runtime w.r.t.𝛾.
Δ𝑀𝑛𝑒𝑤(𝑣𝑙→∅,𝛾)≤−𝑤(𝑣𝑙,𝑈𝑙)
2(𝑚+𝛼)
+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
4(𝑚+𝛼)2.(39)
Equation (39) holds if and only if 𝑣𝑗is merged after 𝑣𝑖and𝑣𝑗. We
multiply right side of Equation (39) by4(𝑚+𝛼)2and obtain𝑌(37):
𝑌(39)=−2(𝑚+𝛼)·𝑤(𝑣 𝑙,𝑈𝑙)
+𝛾·𝑑(𝑣𝑙)·𝑑(𝑈𝑙)
=𝑋(12)−2𝛼·𝑤(𝑣 𝑘,𝑈𝑙)<0(40)
𝑣𝑙is not affected by the cross-sub-community insertion.
Conclusively, the effects of these edge insertions are:
(1)𝑣𝑖could be removed from its sub-community only if 𝛼>
4
𝛾𝑚−𝑑(𝐼𝑖)or𝛼>2𝑤(𝑣𝑖,𝑈𝑖)
𝛾·𝑑(𝑈𝑖)·𝑚−𝑑(𝑣𝑖)according toCase 1
and 5.(2)𝑣𝑗could be removed from its sub-community, only if 𝛼>
2𝑤(𝑣𝑗,𝑈𝑗)
𝛾·𝑑(𝑈𝑗)·𝑚−𝑑(𝑣 𝑗)according toCase 2 and 5.
(3)𝑣𝑘∈𝑆𝑖∪𝑆𝑗(𝑘≠𝑖,𝑗 ) could be removed from its sub-
community only if 𝛼>𝑤(𝑣𝑘,𝑈𝑘)
𝛾·𝑑(𝑣𝑘)·𝑚−1
2𝑑(𝑈𝑘)according
toCase 3 and 6.
(4)𝑣𝑙∉𝑆𝑖∪𝑆𝑗is unaffected according toCase 4 and 7.
□
B Inaddtional experiments
•Effect of𝛾on modularity.Figure 19 shows the average modu-
larity values for all maintenance algorithms, with the parameter
𝛾∈{ 0.5,2,8,32}across all 9 batches, and with the batch size fixed
at 1000. Across all datasets, these maintenance algorithms achieve
equivalent quality in modularity, since the difference in their mod-
ularity values is within 0.01. Overall, our HIT-Leiden still achieves
comparable modularity with other methods across different𝛾.