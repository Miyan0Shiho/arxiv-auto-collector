# Welfarist Formulations for Diverse Similarity Search

**Authors**: Siddharth Barman, Nirjhar Das, Shivam Gupta, Kirankumar Shiragur

**Published**: 2026-02-09 14:42:28

**PDF URL**: [https://arxiv.org/pdf/2602.08742v1](https://arxiv.org/pdf/2602.08742v1)

## Abstract
Nearest Neighbor Search (NNS) is a fundamental problem in data structures with wide-ranging applications, such as web search, recommendation systems, and, more recently, retrieval-augmented generations (RAG). In such recent applications, in addition to the relevance (similarity) of the returned neighbors, diversity among the neighbors is a central requirement. In this paper, we develop principled welfare-based formulations in NNS for realizing diversity across attributes. Our formulations are based on welfare functions -- from mathematical economics -- that satisfy central diversity (fairness) and relevance (economic efficiency) axioms. With a particular focus on Nash social welfare, we note that our welfare-based formulations provide objective functions that adaptively balance relevance and diversity in a query-dependent manner. Notably, such a balance was not present in the prior constraint-based approach, which forced a fixed level of diversity and optimized for relevance. In addition, our formulation provides a parametric way to control the trade-off between relevance and diversity, providing practitioners with flexibility to tailor search results to task-specific requirements. We develop efficient nearest neighbor algorithms with provable guarantees for the welfare-based objectives. Notably, our algorithm can be applied on top of any standard ANN method (i.e., use standard ANN method as a subroutine) to efficiently find neighbors that approximately maximize our welfare-based objectives. Experimental results demonstrate that our approach is practical and substantially improves diversity while maintaining high relevance of the retrieved neighbors.

## Full Text


<!-- PDF content starts -->

Welfarist Formulations for Diverse Similarity Search
Siddharth Barman* Nirjhar Das†Shivam Gupta‡Kirankumar Shiragur§
Abstract
Nearest Neighbor Search (NNS) is a fundamental problem in data structures with wide-ranging
applications, such as web search, recommendation systems, and, more recently, retrieval-augmented
generations (RAG). In such recent applications, in addition to the relevance (similarity) of the returned
neighbors, diversity among the neighbors is a central requirement. In this paper, we develop principled
welfare-based formulations in NNS for realizing diversity across attributes. Our formulations are based
on welfare functions—from mathematical economics—that satisfy central diversity (fairness) and rel-
evance (economic efficiency) axioms. With a particular focus on Nash social welfare, we note that our
welfare-based formulations provide objective functions that adaptively balance relevance and diversity
in a query-dependent manner. Notably, such a balance was not present in the prior constraint-based
approach, which forced a fixed level of diversity and optimized for relevance. In addition, our formu-
lation provides a parametric way to control the trade-off between relevance and diversity, providing
practitioners with flexibility to tailor search results to task-specific requirements. We develop efficient
nearest neighbor algorithms with provable guarantees for the welfare-based objectives. Notably, our
algorithm can be applied on top of any standard ANN method (i.e., use standard ANN method as a
subroutine) to efficiently find neighbors that approximately maximize our welfare-based objectives.
Experimental results demonstrate that our approach is practical and substantially improves diversity
while maintaining high relevance of the retrieved neighbors.
*Indian Institute of Science.barman@iisc.ac.in
†Indian Institute of Science.nirjhardas@iisc.ac.in
‡Indian Institute of Science.shivamgupta2@iisc.ac.in
§Microsoft Research India.kshiragur@microsoft.com
1arXiv:2602.08742v1  [cs.DS]  9 Feb 2026

Table of Contents
1 Introduction 3
2 Problem Formulation and Main Results 6
2.1 Our Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 NaNNS in the Single-Attribute Setting 8
3.1 Proofs of Theorem 1 and Corollary 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4 NaNNS in the Multi-Attribute Setting 12
4.1 Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2 Algorithm for the Multi-Attribute Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
4.3 Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
5 Experimental Evaluations 15
5.1 Metrics for Measuring Relevance and Diversity . . . . . . . . . . . . . . . . . . . . . . . . . 16
5.2 Experimental Setup and Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
5.3 Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
5.4 Results: Balancing Relevance and Diversity . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
5.5 A Faster Heuristic for the Single Attribute Setting:p-FetchUnion-ANN. . . . . . . . . . . . 22
6 Conclusion 23
Acknowledgement 23
A Proofs of Examples 1 and 2 27
B Extensions forp-NNS 28
C Experimental Evaluation and Analysis 33
C.1 Balancing Relevance and Diversity: Single-attribute Setting . . . . . . . . . . . . . . . . . 34
C.1.1 Performance ofNash-ANN. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
C.1.2 Performance ofp-mean-ANN. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
C.1.3 Approximation Ratio Versus Inverse Simpson Index . . . . . . . . . . . . . . . . . . 35
C.1.4 Approximation Ratio Versus Distinct Attribute Count . . . . . . . . . . . . . . . . . 36
C.1.5 Recall Versus Entropy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
C.2 Balancing Relevance and Diversity: Multi-attribute Setting . . . . . . . . . . . . . . . . . . 37
C.3 More Experiments forp-FetchUnion-ANN. . . . . . . . . . . . . . . . . . . . . . . . . . . 39
2

1 Introduction
Nearest Neighbor Search (NNS) is a fundamental problem in computer science with wide-ranging appli-
cations in diverse domains, including computer vision [WWZ+12], data mining [CKPS10], information
retrieval [MRS08], classification [FH89], and recommendation systems [DSM+21]. The relevance of
NNS has grown further in recent years with the advent of retrieval-augmented generation (RAG); see,
e.g., [MSB+24], [WXC+24], and references therein. Formally, given a set of vectorsP⊂Rd, in ambient
dimensiond, and a query vectorq∈Rd, the objective in NNS is to find a subsetSofk(input) vectors
fromPthat are most similar toqunder a similarity functionσ:Rd×Rd→R +. That is, NNS corresponds
to the optimization problemarg maxS⊆P:|S|=kP
v∈Sσ(q, v). Note that, while most prior works in neigh-
bor search express the problem in terms of minimizing distances, we work with the symmetric version of
maximizing similarity.1
In practice, the input vectors are high dimensional; in many of the above-mentioned applications the
ambient dimensiondis close to a thousand. Furthermore, most applications involve a large number of
input vectors. This scale makes exact NNS computationally expensive, since applications require, for
real-time queriesq, NNS solutions in time (sub)linear in the number of input vectors|P|. To address
this challenge, the widely studied framework of Approximate Nearest Neighbor (ANN) search relaxes the
requirement of exactness and instead seeks neighbors whose similarities are approximately close to the
optimal ones.
ANN search has received substantial attention over the past three decades. Early techniques re-
lied on space-partitioning methods, including Locality-Sensitive Hashing (LSH) [IM98; AI08], k-d trees
[AMN+98], and cover trees [BKL06]. More recent industry-scale systems adopt clustering-based [JDJ17;
BBM18] and graph-based [MY16; FXWC19; SKI16; SDK+19] approaches, along with other practically-
efficient methods [SSD+23; SAI+24].
While relevance—measured in terms of a similarity functionσ(·,·)—is a primary objective in NNS,
prior work has shown thatdiversityin the retrieved set of vectors is equally important for user expe-
rience, fairness, and reducing redundancy [CG98]. For instance, in 2019 Google announced a policy
update to limit the number of results from a single domain, thereby reducing redundancy [Lia19]. Sim-
ilarly, Microsoft recently introduced diversity constraints in ad recommendation systems to ensure that
advertisements from a single seller do not dominate the results [AIK+25]. Such an adjustment was cru-
cial for improving user experience and promoting fairness for advertisers. These examples highlight how
diversity, in addition to enhancing fairness and reducing redundancy, directly contributes to improved
search quality for end users.
A natural way to formalize diversity in these settings is to associate each input vector with one or
moreattributes. Diversity can then be measured with respect to these attributes, complementing the
similarity-based relevance. Building on this idea, the current work develops a principled framework
for diversity in neighbor search by drawing on the theory of collective welfare from mathematical eco-
nomics [Mou04]. This perspective enables the design of performance metrics (i.e., optimization criteria)
that balance similarity-based relevance and attribute-based diversity in a theoretically grounded manner.
This formulation is based on the perspective thatalgorithms can be viewed as economic policies.Indeed,
analogous to economic policies, numerous deployed algorithms induce utility (monetary or otherwise)
among the participating agents. For instance, an ANN algorithm—deployed to select display advertise-
ments for search queries—impacts the exposure and, hence, the sales of the participating advertisers.
Notably, there are numerous other application domains wherein the outputs of the underlying algorithms
impact the utilities of individuals; see [ALMK22] and [KR19] for multiple examples. Hence, in contexts
where fairness (diversity) and welfare are important considerations, it is pertinent to evaluate algorithms
analogous to how one evaluates economic policies that induce welfare.
In mathematical economics, welfare functions,f:Rc7→R, provide a principled approach to aggre-
1This enables us to directly apply welfare functions.
3

Figure 1: Neighbor search results (k= 9) on the Amazon dataset. From left:FirstandSecondimages -
ANN and Nash-based results for query “shirts”, respectively.ThirdandFourthimages - ANN and Nash-
based results for query “blue shirt”, respectively. Note that the Nash-based method selects diverse colors
for the query “shirts” but conforms to the blue color for the query “blue shirt”.
gate the utilities ofc∈Z +agents into a single measure. Specifically, if an algorithm induces utilities
u1, u2, . . . , u camong a population ofcagents, then the collective welfare isf(u 1, u2, . . . , u c). A utilitarian
way of aggregation is by considering the arithmetic mean (average) of the utilitiesu ℓs. However, note
that the arithmetic mean is not an ideal criterion if we are required to be fair among thecagents: the
utilitarian welfare (arithmetic mean) can be high even if the utility of only one agent, sayu 1, is large and
all the remaining utilities,u 2, . . . , u c, are zero. The theory of collective welfare develops meaningful al-
ternatives to the arithmetic mean by identifying welfare functions,fs, that satisfy fairness and efficiency
axioms.
Among such alternatives, Nash social welfare (NSW) is an exemplar that upholds multiple fairness ax-
ioms, including symmetry, independence of unconcerned agents, scale invariance, and the Pigou-Dalton
transfer principle [Mou04]. Nash social welfare is obtained by setting the functionfas the geometric
mean,NSW(u 1, . . . , u c):= Qc
ℓ=1uℓ1/c. The fact that NSW strikes a balance between fairness and eco-
nomic efficiency is supported by the observation that it sits between egalitarian and utilitarian welfare:
the geometric mean is at least as large as the minimum value,min 1≤ℓ≤c uℓ, and it is also at most the
arithmetic mean1
cPc
ℓ=1uℓ(the AM-GM inequality).
The overarching goal of this work is to realize diversity (fairness) across attributes in nearest neighbor
search while maintaining relevance of the returnedkvectors. Our modeling insight here is to equate
attributes with agents and apply Nash social welfare.
In particular, consider a setting where we havec∈Z +different attributes (across the input vectors),
and letSbe any subset ofkvectors (neighbors) among the input setP. In our model, each included
vectorv∈S, with attributeℓ∈[c], contributes to the utilityu ℓ(see Section 2.1), and the Nash social
welfare (NSW) induced bySis the geometric mean of these utilities,u 1, u2, . . . , u c. Our objective is to
find a size-ksubset,S∗⊆P, of input vectors with as large NSW as possible.
The following two instantiations highlight the applicability of our model in NNS settings: In a display-
advertising context withcsellers, each selected advertisementv∈Sof a sellerℓ∈[c]contributes toℓ’s
exposure (utility)u ℓ. Similarly, in an apparel-search setup withccolors in total, each displayed product
v∈Swith colorℓ∈[c]contributes to the utilityu ℓ.
Prior work [AIK+25] imposed constraints for achieving diversity in NNS. These constraints enforced
that, for eachℓ∈[c]and among thekreturned vectors, at mostk′many can have attributeℓ. Such hard
constraints rely on a fixed ad hoc quota parameterk′and may fail to adapt to the intent expressed in
the query. In contrast, our NSW-based approach balances relevance and diversity in a query-dependent
manner. For example, in the apparel-search setup, if the search query is “blue shirt,” then a constraint
on the color attribute ‘blue’ (i.e., whenℓstands for ‘blue’) would limit the relevance by excluding valid
vectors. NSW, however, for the “blue shirt” query, is free to select all thekvectors with attribute ‘blue’
upholding relevance; see Figure 1 for supporting empirical results. On the other hand, if the apparel-
search query is just “shirts,” then NSW criterion is inclined to select vectors with different color attributes.
These features of NSW are substantiated by the stylized instances given in Examples 1 and 2 (Section
4

2.1).
We reiterate that our formulation does not require a quota parameterk′to force diversity. For NSW,
diversity (fairness) across attributes is obtained via normative properties of Nash social welfare. Hence,
with axiomatic support, NSW stands as a meaningful criterion in neighbor search, as it is in the context
of economic and allocation policies.
Our welfarist formulation extends further to control the trade-off between relevance and diversity.
Specifically, we also considerp-mean welfare. Formally, for exponent parameterp∈(−∞,1], thepth
meanM p(·), ofcutilitiesu 1, u2, . . . , u c∈R +, is defined asM p(u1, . . . , u c):= 1
cPc
ℓ=1up
ℓ1/p. Thep-mean
welfare,M p(·), captures a range of objectives with different values ofp: it corresponds to the utilitarian
welfare (arithmetic mean) whenp= 1, the NSW (geometric mean) withp→0, and the egalitarian
welfare whenp→−∞. Notably, settingp= 1, we get back the standard nearest neighbor objective, i.e.,
maximizingM 1(·)corresponds to finding theknearest neighbors and this objective is not concerned with
diversity across attributes. At the other extreme,p→ −∞aims to find as attribute-diverse a set ofk
vectors as possible (while paying scarce attention to relevance).
We study, both theoretically and experimentally, two diversity settings: (i) single-attribute setting and
(ii) multi-attribute setting. In the single-attribute setting, each input vectorv∈Pis associated with
exactly one attributeℓ∈[c]– this captures, for instance, the display-advertisement setup, wherein each
advertisementvbelongs to exactly one sellerℓ. In the more general multi-attribute setting, each input
vectorv∈Pcan have more than one attribute; in apparel-search, for instance, the products can be
associated with multiple attributes, such as color, brand, and price.
We note that the constraint-based formulation for diversity considered in [AIK+25] primarily ad-
dresses single-attribute setting. In fact, generalizing such constraints to the multi-attribute context leads
to a formulation wherein it is NP-hard even to determine whether there existkvectors that satisfy the
constraints, i.e., it would be computationally hard to find any size-kconstraint-feasible subsetS, let alone
an optimal one.2
By contrast, our NSW formulation does not run into such a feasibility barrier. Here, for any candidate
subsetSofkvectors, each included vectorv∈Scontributes to the utilityu ℓof every attributeℓassociated
withv. As before, the NSW induced bySis the geometric mean of the induced utilities,u 1, u2, . . . , u c,
and the objective is to find a subset ofkvectors with as large NSW as possible.
Our Contributions
• We view the NSW formulation for diversity, in both single-attribute and multi-attribute settings, as
a key contribution of the current paper. Another relevant contribution of this work is the general-
ization top-mean welfare, which provides a systematic way to trade off relevance and diversity.
• We also develop efficient algorithms, with provable guarantees, for the NSW andp-mean welfare
formulations. For the single-attribute setting, we develop an efficient greedy algorithm for finding
kvectors that optimize the Nash social welfare among thecattributes (Theorem 1). In addition,
this algorithm can be provably combined with any sublinear ANN method (as a subroutine) to find
near-optimal solutions for the Nash objective in sublinear time (Corollary 2).
• For the multi-attribute setting, we first show that finding the set ofkvectors that maximize the
Nash social welfare is NP-hard (Theorem 3). We complement this hardness result, by developing a
polynomial-time approximation algorithm that achieves an approximation ratio of(1−1/e)≈0.63
for maximizing the logarithm of the Nash social welfare (Theorem 4).
• We complement our theoretical results with experiments on both real-world and semi-synthetic
datasets. These experiments demonstrate that the NSW objective effectively captures the trade-off
2This hardness result follows via a reduction from the Maximum Independent Set problem.
5

between diversity and relevance in a query-dependent manner. We further analyze the behavior of
thep-mean welfare objective across different values ofp∈(−∞,1], observing that it interpolates
smoothly between prioritizing for diversity, whenpis small, and focusing on relevance, whenp
is large. Finally, we benchmark the solution quality and running times of various algorithms for
solving the NSW andp-mean formulations proposed in this work.
2 Problem Formulation and Main Results
We are interested in neighbor search algorithms that not only achieve a high relevance, but also find a
diverse set of vectors for each query. To quantify diversity we work with a model wherein each input
vectorv∈Pis assigned one or more attributes from the set[c] ={1,2, . . . , c}. In particular, write
atb(v)⊆[c]to denote the attributes assigned to vectorv∈P. Also, letD ℓ⊆Pdenote the subset of
vectors that are assigned attributeℓ∈[c], i.e.,D ℓ:={v∈P|ℓ∈atb(v)}.
2.1 Our Results
An insight of this work is to equate thesecattributes withcdistinct agents. Here, the output of a neighbor
search algorithm—i.e., the selected subsetS⊆P—induces utility among these agents. With this per-
spective, we define the Nash Nearest Neighbor Search problem (NaNNS) below. This novel formulation
for diversity is a key contribution of this work. For any queryq∈Rdand subsetS⊆P, we define utility
uℓ(S):=P
v∈S∩D ℓσ(q, v), for eachℓ∈[c]. That is,u ℓ(S)is equal to the cumulative similarity betweenq
and the vectors inSthat belong toD ℓ. Equivalently,u ℓ(S)is the cumulative similarity of the vectors inS
that have attributeℓ.3
We employ Nash social welfare to identify size-ksubsetsSthat are both relevant (with respect to
similarity) and support diversity among thecattribute classes. The Nash social welfare amongcagents is
defined as the geometric mean of the agents’ utilities. Specifically, in the above-mentioned utility model
and with a smoothening parameterη >0, the Nash social welfare (NSW) induced by any subsetS⊆P
among thecattributes is defined as
NSW(S) := (cY
ℓ=1(uℓ(S) +η))1/c.(1)
Throughout,η >0will be a fixed smoothing constant that ensures that NSW remains nonzero.
Definition 1(NaNNS).Nash nearest neighbor search (NaNNS) corresponds to the following the opti-
mization problemarg maxS⊆P:|S|=k NSW(S), or, equivalently,
arg max
S⊆P:|S|=klog NSW(S)(2)
Here, we havelog NSW(S) =1
cP
ℓ∈[c]log(u ℓ(S) +η).
To further appreciate the welfarist approach, note that one recovers the standard nearest neighbor
problem, NNS, in the single-attribute setting, if—instead of the geometric mean—we maximize the arith-
metic mean. That is, maximizing the utilitarian social welfare gives usmax S⊆P:|S|=kPc
ℓ=1uℓ(S) =
max S⊆P:|S|=kP
v∈Sσ(q, v).
As stated in the introduction, depending on the query and the problem instance, solutions obtained
via NaNNS can adjust between the ones obtained through standard NNS and those obtained via hard con-
straints. This feature is illustrated in the following stylized examples; see Appendix A for the associated
proofs.
3Note that in the above-mentioned display-advertising example,u ℓ(·)is the cumulative similarity between the (search) query
and the selected advertisements that are from sellerℓ.
6

The first example shows that if all vectors have same similarity, then an optimal solution,S∗, for
NaNNS is completely diverse, i.e., all the vectors inS∗have different attributes.
Example 1(Complete Diversity via NaNNS).Consider an instance in which, for a given queryq∈Rd,
all vectors inPare equally similar with the query:σ(q, v) = 1for allv∈P. Also, let|atb(v)|= 1for all
v∈P, and writeS∗∈arg maxS⊆P:|S|=k NSW(S). Ifc≥k, then here it holds that|S∗∩D ℓ| ≤1for all
ℓ∈[c].
The second example shows that if the vectors of only one attribute have high similarity with the given
query, then a Nash optimal solutionS∗contains only vectors with that attribute.
Example 2(Complete Relevance via NaNNS).Consider an instance in which, for a given queryq∈Rd
and a particularℓ∗∈[c], only vectorsv∈D ℓ∗have similarityσ(q, v) = 1and all other vectorsv′∈P\D ℓ∗
have similarityσ(q, v′) = 0. Also, suppose that|atb(v)|= 1for eachv∈P, along with|D ℓ∗| ≥k. Then,
for a Nash optimal solutionS∗∈arg maxS⊆P,|S|=k NSW(S), it holds that|S∗∩D ℓ∗|=k. That is, for all
otherℓ∈[c]\ {ℓ∗}we have|S∗∩D ℓ|= 0.
With the above-mentioned utility model for thecattributes, we also identify an extended formulation
based on generalizedp-means. Specifically, for exponent parameterp∈(−∞,1], thepth meanM p(·), of
cnonnegative numbersw 1, w2, . . . , w c∈R +, is defined as
Mp(w1, . . . , w c):= 
1
ccX
ℓ=1wp
ℓ!1/p
(3)
Note thatM 1(w1, . . . , w c)is the arithmetic mean1
cPc
ℓ=1wℓ. Here, whenp→0, we obtain the geometric
mean (Nash social welfare):M 0(w1, . . . , w c) = (Qc
ℓ=1wℓ)1/c. Further,p→ −∞gives us egalitarian
welfare,M −∞(w1, . . . , w ℓ) = min 1≤ℓ≤c wℓ.
Analogous to NaNNS, we consider a fixed smoothing constantη >0, and, for each subsetS⊆P,
defineM p(S):=M p(u1(S) +η, . . . , u c(S) +η).
With these constructs in hand and generalizing both NNS and NaNNS, we have thep-mean nearest
neighbor search (p-NNS) problem defined as follows.
Definition 2(p-NNS).For any exponent parametersp∈(−∞,1], thep-mean nearest neighbor search
(p-NNS) corresponds to the following the optimization problem
max
S⊆P:|S|=kMp(S) = max
S⊆P:|S|=k 
1
ccX
ℓ=1(uℓ(S) +η)p!1/p
(4)
Diversity in Single- and Multi-Attribute Settings.The current work addresses two diversity settings:
the single-attribute setup and, the more general, the multi-attribute one. The single-attribute setting
refers to the case wherein|atb(v)|= 1for each input vectorv∈Pand, hence, the attribute classesD ℓs
are pairwise disjoint. In the more general multi-attribute setting, we have|atb(v)| ≥1; here, the sets
Dℓ-s intersect.4Notably, the NaNNS seamlessly applies to both these settings.
Algorithmic Results for Single-Attribute NaNNS andp-NNS.In addition to introducing the NaNNS
andp-NNS formulations for capturing diversity, we develop algorithmic results for these problems, thereby
demonstrating the practicality of our approach in neighbor search. In particular, in the single-attribute
setting, we show that both NaNNS andp-NNS admit efficient algorithms.
4For a motivating instantiation for multi-attributes, note that, in the apparel-search context, it is possible for a product (input
vector)vto have multiple attributes based onv’s seller and its color(s).
7

Theorem 1.In the single-attribute setting, given any queryq∈Rdand an (exact) oracleENNforkmost
similar vectors from any set, Algorithm 1 (Nash-ANN) returns an optimal solution for NaNNS, i.e., it returns
a size-ksubsetALG⊆Pthat satisfiesALG∈arg maxS⊆P:|S|=k NSW(S). Furthermore, the algorithm runs
in timeO(kc) +Pc
ℓ=1ENN(D ℓ, q), whereENN(D ℓ, q)is the time required by the exact oracle to findkmost
similar vectors toqinD ℓ.
Further, to establish the practicality of our formulations, we present an approximate algorithm for
NaNNS that leverages any standard ANN algorithm as an oracle (subroutine), i.e., works with anyα-
approximate ANN oracle (α∈(0,1)) which returns a subsetScontainingkvectors satisfyingσ(q, v (i))≥
α σ(q, v∗
(i)), for alli∈[k], wherev (i)andv∗
(i)are thei-th most similar vectors toqinSandP, respectively.
Formally,
Corollary 2.In the single-attribute setting, given any queryq∈Rdand anα-approximate oracleANNfork
most similar vectors from any set, Algorithm 1 (Nash-ANN) returns anα-approximate solution for NaNNS,
i.e., it returns a size-ksubsetALG⊆PwithNSW(ALG)≥αmax S⊆P:|S|=k NSW(S). The algorithm runs in
timeO(kc) +Pc
ℓ=1ANN(D ℓ, q), whereANN(D ℓ, q)is the time required by the oracle to findksimilar vectors
toqinD ℓ.
Furthermore, both Theorem 1 and Corollary 2 generalize top-NNS problem with a slight modification
in Algorithm 1. Specifically, there exists exact, efficient algorithm (Algorithm 3) for thep-NNS problem
(Theorem 11 and Corollary 12). Appendix B details this algorithm and the results forp-NNS in the
single-attribute setting.
Algorithmic Results for Multi-Attribute NaNNS.Next, we address the multi-attribute setting. While
the optimization problem (2) in the single attribute setting can be solved efficiently, the problem is NP-
hard in the multi-attribute setup (see Section 4.1 for the proof).
Theorem 3.In the multi-attribute setting, with parameterη= 1, NaNNS isNP-hard.
Complementing this hardness result, we show that, considering the logarithm of the objective, NaNNS
in the multi-attribute setting admits a polynomial-time 
1−1
e
-approximation algorithm. This result is
established in Section 4.3.
Theorem 4.In the multi-attribute setting with parameterη= 1, there exists a polynomial-time algo-
rithm (Algorithm 2) that, given any queryq∈Rd, finds a size-ksubsetALG⊆Pwithlog NSW(ALG)≥ 
1−1
e
log NSW(OPT); here,OPTdenotes an optimal solution for the optimization problem (2).
Experimental Validation of our Formulation and Algorithms.We complement our theoretical results
with several experiments on real-world datasets. Our findings highlight that the Nash-based formulation
strikes a balance between diversity and relevance. Specifically, we find that, across datasets and in both
single- and multi-attribute settings, the Nash formulation maintains relevance and consistently achieves
high diversity. By contrast, the hard-constrained formulation from [AIK+25] is highly sensitive to the
choice of the constraint parameterk′, and in some cases incurs a substantial drop in relevance. Section 5
details our experimental evaluations.
3 NaNNS in the Single-Attribute Setting
In this section, we first provide our exact, efficient algorithm (Algorithm 1) for NaNNS in the single-
attribute setting and then present the proof of optimality of the algorithm (i.e., proof of Theorem 1
and Corollary 2).
8

Algorithm 1:Nash-ANN: Algorithm for NaNNS in the single-attribute setting
Input:Queryq∈Rdand, for each attributeℓ∈[c], the set of input vectorsD ℓ⊂Rd
1For eachℓ∈[c], fetch bDℓ, thek(exact or approximate) nearest neighbors ofq∈RdfromD ℓ.
2For everyℓ∈[c]and each indexi∈[k], letvℓ
(i)denote theith most similar vector toqin bDℓ.
3Initialize subset ALG← ∅, along with countk ℓ←0and utilityw ℓ←0, for eachℓ∈[c].
4while|ALG|< kdo
5a←arg max
ℓ∈[c]
log 
wℓ+η+σ(q, vℓ
(kℓ+1))
−log(w ℓ+η)
.▷Ties broken arbitrarily
6ALG←ALG∪ {va
(ka+1)}.w a←w a+σ(q, va
(ka+1)).ka←k a+ 1.
7returnALG
The algorithm has two parts: a prefetching step and a greedy, iterative selection. In the prefetching
step, for each attributeℓ∈[c], we populatekvectors from withinD ℓ5that are most similar to the given
queryq∈Rd. Such a size-ksubset, for eachℓ∈[c], can be obtained by executing any nearest neighbor
search algorithm withinD ℓand with respect to queryq. Alternatively, we can execute any standard ANN
algorithm as a subroutine and find sufficiently good approximations for theknearest neighbors (ofq)
within eachD ℓ.
Write bDℓ⊆D ℓto denote thek—exact or approximate—nearest neighbors ofq∈RdinD ℓ. We
note that our algorithm is robust to the choice of the search algorithm (subroutine) used for finding
bDℓs: IfbDℓs are exact nearest neighbors, then Algorithm 1 optimally solves NaNNS in the single-attribute
setting (Theorem 1). Otherwise, if bDℓs are obtained via an ANN algorithm with approximation guarantee
α∈(0,1), then Algorithm 1 achieves an approximation ratio ofα(Corollary 2).
The algorithm then considers the vectors with each bDℓin decreasing order of their similarity withq.
Confining to this order, the algorithm populates thekdesired vectors iteratively. In each iteration, the
algorithm greedily selects a new vector based on maximizing the marginal increase inlog NSW(·); see
Lines 5 and 6 in Algorithm 1. Theorem 1 and Corollary 2 (stated previously) provide our main results for
Algorithm 1. Proofs of Theorem 1 and Corollary 2 are presented below.
3.1 Proofs of Theorem 1 and Corollary 2
Now we state the proof of optimality of Algorithm 1. To obtain the proof, we make use of two lemmas
stated below. We defer their proofs till after the proof of Theorem 1 and Corollary 2.
As in Algorithm 1, write bDℓto denote theknearest neighbors of the given queryqin the setD ℓ.
Recall that in the single-attribute setting the setsD ℓs are disjoint acrossℓ∈[c]. Also,vℓ
(j)∈bDℓdenotes
thejthmost similar vector toqin bDℓ, for each indexj∈[k]. For a given attributeℓ∈[c], we define the
logarithm of cumulative similarity upto theithmost similar vector as
Fℓ(i):= log
iX
j=1σ(q, vℓ
(j)) +η
 (5)
Note thatF ℓ(i)is also equal to the logarithm of the cumulative similarity of theimost similar (toq)
vectors inD ℓwhen the neighbor search oracle is exact. The lemma below shows thatF ℓ(·)satisfies a
useful decreasing marginals property.
Lemma 5(Decreasing Marginals).For all attributesℓ∈[c]and indicesi′, i∈[k], withi′< i, it holds that
Fℓ(i′)−F ℓ(i′−1)≥F ℓ(i)−F ℓ(i−1).
5Recall that in the single-attribute setting, the input vectorsPare partitioned into subsetsD 1, . . . , D c, whereD ℓdenotes the
subset of input vectors with attributeℓ∈[c].
9

The following lemma asserts the Nash optimality of the subset returned by Algorithm 1, ALG, within
a relevant class of solutions.
Lemma 6.In the single-attribute setting, letALGbe the subset of vectors returned by Algorithm 1 and
Sbe any subset of input vectors with the property that|S∩D ℓ|=|ALG∩D ℓ|, for eachℓ∈[c]. Then,
NSW(ALG)≥NSW(S).
Now we are ready to state the proof of Theorem 1.
Theorem 1.In the single-attribute setting, given any queryq∈Rdand an (exact) oracleENNforkmost
similar vectors from any set, Algorithm 1 (Nash-ANN) returns an optimal solution for NaNNS, i.e., it returns
a size-ksubsetALG⊆Pthat satisfiesALG∈arg maxS⊆P:|S|=k NSW(S). Furthermore, the algorithm runs
in timeO(kc) +Pc
ℓ=1ENN(D ℓ, q), whereENN(D ℓ, q)is the time required by the exact oracle to findkmost
similar vectors toqinD ℓ.
Proof.The runtime of Algorithm 1 can be established by noting that Line 1 requiresPc
ℓ=1ENN(D ℓ, q)
time to populate the subsets bDℓs, and the while-loop (Lines 4-6) iteratesktimes and each iteration
(specifically, Line 5) runs inO(c)time. Hence, as stated, the time complexity of the algorithm isO(kc) +Pc
ℓ=1ENN(D ℓ, q).
Next, we prove the optimality of the returned set ALG. Let OPT∈arg maxS⊆P:|S|=k NSW(S)be an
optimal solution with attribute counts|OPT∩D ℓ|as close to|ALG∩D ℓ|as possible. That is, among the
optimal solutions, it is one that minimizesPc
ℓ=1|k∗
ℓ−kℓ|, wherek∗
ℓ=|OPT∩D ℓ|andk ℓ=|ALG∩D ℓ|, for
eachℓ∈[c]. We will prove that OPTsatisfiesk∗
ℓ=kℓfor eachℓ∈[c]. This guarantee, along with Lemma
6, implies that, as desired, ALGis a Nash optimal solution.
Assume, towards a contradiction, thatk∗
ℓ̸=kℓfor someℓ∈[c]. Since|OPT|=|ALG|=k, there exist
attributesx, y∈[c]with the property thatk∗
x< kxandk∗
y> ky. For a given attributeℓ∈[c], write the
logarithm of cumulative similarity upto theithmost similar vector asF ℓ(i):= log Pi
j=1σ(q, vℓ
(j)) +η
,
wherevℓ
(j)is defined in Line 2 of Algorithm 1.
Next, note that for any attributeℓ∈[c], if Algorithm 1, at any point during its execution, has included
k′
ℓvectors of attributeℓin ALG, then at that point the maintained utilityw ℓ=Pk′
ℓ
j=1σ(q, vℓ
(j)). Hence, at
the beginning of any iteration of the algorithm, if thek′
ℓdenotes the number of selected vectors of each
attributeℓ∈[c], then the marginals considered in Line 5 areF ℓ(k′
ℓ+ 1)−F ℓ(k′
ℓ). These observations
and the selection criterion in Line 5 of the algorithm give us the following inequality for the counts
kx=|ALG∩D x|andk y=|ALG∩D y|of the returned solution ALG:
Fx(kx)−F x(kx−1)≥F y(ky+ 1)−F y(ky)(6)
Specifically, equation (6) follows by considering the iteration in whichkth
x(last) vector of attributexwas
selected by the algorithm. Before that iteration the algorithm had selected(k x−1)vectors of attributex,
and letk′
ydenote the number of vectors with attributeythat have been selected till that point. Note that
k′
y≤ky. The fact that thekth
xvector was (greedily) selected in Line 5, instead of including an additional
vector of attributey, givesF x(kx)−F x(kx−1)≥F y(k′
y+ 1)−F y(k′
y)≥F y(ky+ 1)−F y(ky); here, the
last inequality follows from Lemma 5. Therefore we have,
Fx(k∗
x+ 1)−F x(k∗
x)(i)
≥F x(kx)−F x(kx−1)(ii)
≥F y(ky+ 1)−F y(ky)(iii)
≥F y(k∗
y)−F y(k∗
y−1)(7)
Here, inequality (i) follows fromk∗
x< k xand Lemma 5, inequality (ii) is due to equation (6), and
inequality (iii) is viak∗
y> kyand Lemma 5.
Next, observe that the definition of bDℓensures thatvℓ
(i)is, in fact, theithmost similar (toq) vector
among the ones that have attributeℓ, i.e.,ithmost similar in all ofD ℓ. Since OPTis an optimal solution,
thek∗
ℓ=|OPT∩D ℓ|vectors of attributeℓin OPTare the most similark∗
ℓvectors fromD ℓ. That is,
10

OPT∩D ℓ={vℓ
(1), . . . , vℓ
(k∗
ℓ)}, for eachℓ∈[c]. This observation and the definition ofF ℓ(·)imply that the
logarithm of OPT’s NSW satisfieslog NSW(OPT) =1
cPc
ℓ=1Fℓ(k∗
ℓ). Now, consider a subset of vectorsS
obtained from OPTby including vectorvx
(k∗x+1)and removingvy
(k∗y), i.e.,S=
OPT∪n
vx
(k∗x+1)o
\n
vy
(k∗y)o
.
Note that
log NSW(S)−log NSW(OPT) =1
c
Fx(k∗
x+ 1)−F x(k∗
x)
+1
c
Fy(k∗
y−1)−F y(k∗
y)
≥0,
where the last inequality is via equation (7). Hence,NSW(S)≥NSW(OPT). Given that OPTis a
Nash optimal solution, the last inequality must hold with an equality,NSW(S) = NSW(OPT), i.e.,Sis
an optimal solution as well. This, however, contradicts the choice of OPTas an optimal solution that
minimizesPc
ℓ=1|k∗
ℓ−kℓ|; note thatPc
ℓ=1bkℓ−kℓ<Pc
ℓ=1|k∗
ℓ−kℓ|, where bkℓ:=|S∩D ℓ|.
Therefore, by way of contradiction, we obtain that|OPT∩D ℓ|=|ALG∩D ℓ|for eachℓ∈[c]. As
mentioned previously, this guarantee along with Lemma 6 imply that ALGis a Nash optimal solution.
This completes the proof of the theorem.
Corollary 2.In the single-attribute setting, given any queryq∈Rdand anα-approximate oracleANNfork
most similar vectors from any set, Algorithm 1 (Nash-ANN) returns anα-approximate solution for NaNNS,
i.e., it returns a size-ksubsetALG⊆PwithNSW(ALG)≥αmax S⊆P:|S|=k NSW(S). The algorithm runs in
timeO(kc) +Pc
ℓ=1ANN(D ℓ, q), whereANN(D ℓ, q)is the time required by the oracle to findksimilar vectors
toqinD ℓ.
Proof.The running time of the algorithm follows via an argument similar to the one used in the proof
of Theorem 1. Therefore, we only argue correctness here.
For everyℓ∈[c], let theα-approximate oracle return bDℓ. Recall thatvℓ
(i),i∈[k], denotes theithmost
similar point toqin the set bDℓ. Further, for everyℓ∈[c], letD∗
ℓbe the set ofkmost similar points toq
withinD ℓand definev∗ℓ
(i),i∈[k], to be theithmost similar point toqinD∗
ℓ. Recall that by the guarantee
of theα-approximate NNS oracle, we haveσ(q, vℓ
(i))≥α·σ(q, v∗ℓ
(i))for alli∈[k]. Let OPTbe an optimal
solution to the NaNNS problem containingk∗
ℓmost similar points of attributeℓfor everyℓ∈[c].
Finally, let dOPTbe the optimal solution to the NaNNS problem when the set of vectors to search over
isP=∪ ℓ∈[c]bDℓ.
By an argument similar to the proof of Theorem 1, we haveNSW(ALG) = NSW( dOPT). Therefore
NSW(ALG) = NSW( dOPT)
≥
Y
ℓ∈[c]
k∗
ℓX
i=1σ(q, vℓ
(i)) +η

1
c
(S
ℓ∈[c]:k∗
ℓ≥1{vℓ
(1), . . . , vℓ
(k∗
ℓ)}is a feasible solution)
≥
Y
ℓ∈[c]
k∗
ℓX
i=1ασ(q, v∗ℓ
(i)) +η

1
c
(byα-approximate guarantee of the oracle;k∗
ℓ≤k)
≥
Y
ℓ∈[c]α
k∗
ℓX
i=1σ(q, v∗ℓ
(i)) +η

1
c
(α∈(0,1))
=αNSW(OPT)(definition of OPT)
Hence, the corollary stands proved.
We complete this section by stating the proofs of Lemmas 5 and 6.
11

Proof of Lemma 5.Note thatexp(F ℓ(i)) =Pi
j=1σ(q, vℓ
(j)) +η= exp(F ℓ(i−1)) +σ(q, vℓ
(i)). Therefore, we
have
exp(F ℓ(i)−F ℓ(i−1)) =exp(F ℓ(i))
exp(F ℓ(i−1))=exp(F ℓ(i−1)) +σ(q, vℓ
(i))
exp(F ℓ(i−1))= 1 +σ(q, vℓ
(i))
exp(F ℓ(i−1))(8)
Similarly, we haveexp(F ℓ(i′)−F ℓ(i′−1)) = 1 +σ(q,vℓ
(i′))
exp(F ℓ(i′−1)).
In addition, the indexing of the vectorsvℓ
(j)ensures thatσ(q, vℓ
(i′))≥σ(q, vℓ
(i))fori′< i. Moreover,
exp(F ℓ(i))is non-decreasing since it is the cumulative sum of non-negative similarities uptoithvectorvℓ
(i).
Hence,exp(F ℓ(i))≥exp(F ℓ(i′))fori′< i. Combining these inequalities, we obtain
σ(q, vℓ
(i′))
exp(F ℓ(i′−1))≥σ(q, vℓ
(i))
exp(F ℓ(i−1)).
That is,1 +σ(q,vℓ
(i′))
exp(F ℓ(i′−1))≥1 +σ(q,vℓ
(i))
exp(F ℓ(i−1)). Hence, equation (8) gives us
exp(F ℓ(i′)−F ℓ(i′−1))≥exp(F ℓ(i)−F ℓ(i−1))(9)
Sinceexp(·)is an increasing function, inequality (9) implies
Fℓ(i′)−F ℓ(i′−1)≥F ℓ(i)−F ℓ(i−1).
The lemma stands proved.
Proof of Lemma 6.Assume, towards a contradiction, that there exists a subset of input vectorsSthat
satisfies|S∩D ℓ|=|ALG∩D ℓ|, for eachℓ∈[c], and still induces NSW strictly greater than that of
ALG. This strict inequality implies that there exists an attributea∈[c]with the property that the utility
ua(S)> u a(ALG).
X
t∈S∩D aσ(q, t)>X
v∈ALG∩D aσ(q, v)(10)
On the other hand, note that the construction of Algorithm 1 and the definition of bDaensure that the
vectors in ALG∩D aare, in fact, the most similar toqamong all the vectors inD a. This observation and
the fact that|S∩D a|=|ALG∩D a|gives usP
v∈ALG∩D aσ(q, v)≥P
t∈S∩D aσ(q, t). This equation, however,
contradicts the strict inequality (10).
Therefore, by way of contradiction, we obtain that there does not exist a subsetSsuch that|S∩D ℓ|=
|ALG∩D ℓ|, for eachℓ∈[c], andNSW(ALG)<NSW(S). The lemma stands proved.
4 NaNNS in the Multi-Attribute Setting
In this section, we first establish the NP-Hardness of the NaNNS problem in the multi-attribute setting by
stating the proof of Theorem 3. Thereafter, we describe an efficient algorithm that obtains a constant ap-
proximation in terms of the logarithm of the Nash Social Welfare objective, and prove the approximation
ratio (Theorem 4).
12

4.1 Proof of Theorem 3
Recall that in the multi-attribute setting, input vectorsv∈Pare associated with one or more attributes,
|atb(v)| ≥1.
Theorem 3.In the multi-attribute setting, with parameterη= 1, NaNNS isNP-hard.
Proof.Consider the decision version of the optimization problem: given a thresholdW∈Q, decide
whether there exists a size-ksubsetS⊆Psuch thatlog NSW(S)≥W. We will refer to this problem
asNaNNS. Note that the input in anNaNNSinstance consists of: a set ofnvectorsP⊂Rd, a similarity
functionσ:Rd×Rd→R +, an integerk∈N, the setsD ℓ={p∈P:ℓ∈atb(p)}for every attributeℓ∈[c],
a query pointq∈Rd, and thresholdW∈Q. We will show thatNaNNSis NP-complete by reducing EXACT
REGULARSETPACKING(ERSP) to it.ERSPis known to be NP-complete [GJ90] and is also W[1]-hard with
respect to solution size [ADP80].
InERSP, we are given a universe ofnelements,U={1,2, . . . , n}, an integerk∈N, and a collection
ofmsubsetsS={S 1, . . . , S m}, with each subsetS i⊆ Uof cardinalityτ(i.e.,|S i|=τfor eachi∈[m]).
The objective here is to decide whether there exists a size-ksub-collectionI⊆ Ssuch that for all distinct
S, S′∈Iwe haveS∩S′=∅.
For the reduction, we start with the given instance ofERSPand construct an instance ofNaNNS: Con-
siderUas the set of attributes, i.e., setc=n. In addition, we set the input vectorsP=1
τ1S|S∈ S	
;
here,1 S∈Rnis the characteristic vector (inRn) of the subsetS, i.e., for eachi∈[n], thei-th coordinate
of1 Sis1{i∈S}. Note that the set of vectorsPis of cardinalitym.
Furthermore, we set the query vectorq=1as the all-ones vector inRn. In thisNaNNSinstance, each
input vector1
τ1Sis assigned attributeℓ∈[n]iff elementℓ∈S. That is,D ℓ={1
τ1S|S∈ Sandℓ∈S}.
The number of neighbors to be found in the constructedNaNNSis equal tok, which is the count in the
given theERSPinstance. Also, the similarity functionσ:Rn×Rn→Ris taken to be the standard
dot-product. Finally, we set the thresholdW=τklog 2
c.
Note that the reduction takes time polynomial innandm. In addition, for each input vectorv∈Pit
holds thatv=1
τ·1Sfor someS∈ Sand, hence,σ(q, v) =⟨1
τ1S,1⟩= 1.
Now we establish the correctness of the reduction.
Forward direction “⇒”: Suppose the givenERSPinstance admits a (size-k) solutionI∗⊂ S. Consider the
subset of vectorsN∗:={1
τ1S|S∈I∗}. Indeed,N∗⊆Pand|N∗|=k, henceN∗is a feasible set of the
NaNNSproblem. Now, sinceI∗is a solution to theERSPinstance, for distinctS,S′∈I∗we haveS∩S′=∅.
In particular, if for an elementℓ∈[c], we haveℓ∈Sfor someS∈I∗, thenℓ /∈S′for allS′∈I∗\ {S}.
Therefore,|N∗∩D ℓ| ≤1for allℓ∈[c], which in turn implies thatu ℓ(N∗)is either1or0for everyℓ∈[c].
Finally, note that each vectorv∈Pbelongs to exactlyτattributes, i.e.,|atb(v)|=τ. Hence,
log NSW(N∗) =1
ccX
ℓ=1log(1 +u ℓ(N∗)) =1
cX
v∈N∗X
ℓ∈atb(v)log(1 + 1) =τklog 2
c.
Therefore, if the givenERSPinstance admits a solution (exact packing), then the constructedNaNNS
instance haskneighbors with sufficiently highlog NSW.
Reverse direction “⇐”: SupposeN∗⊆Pis a solution in the constructedNaNNSinstance with|N∗|=k
andlog NSW(N∗)≥W. DefineI∗:={S|1
τ·1S∈N∗}and note that|I∗|=k. We will show thatI∗is a
solution for the givenERSPinstance, i.e., it consists of disjoint subsets.
Towards this, first note thatN∗induces social welfare:
X
ℓ∈[c]uℓ(N∗) =X
ℓ∈[c]X
v∈N∗∩Dℓσ(q, v) =X
v∈N∗X
ℓ∈atb(v)σ(q, v) =τk(11)
Furthermore, any attributeℓ∈[c]has a non-zero utility underN∗iffℓ∈Sfor some subsetS∈I∗.
Hence,A :=∪ S∈I∗Scorresponds to the set of attributes with non-zero utility underN∗. We have
13

1≤ |A| ≤τk. Next, using the fact thatlog NSW(N∗)≥W=τklog 2
cwe obtain
τklog 2
c≤log NSW(N∗) =1
cX
ℓ∈[c]log(1 +u ℓ(N∗))
=1
cX
ℓ∈Alog(1 +u ℓ(N∗))
=|A|
c1
|A|X
ℓ∈Alog(1 +u ℓ(N∗))
≤|A|
clog 
1
|A|X
ℓ∈A(1 +u ℓ(N∗))!
(concavity oflog)
=|A|
clog
1 +P
ℓ∈Auℓ(N∗)
|A|
=|A|
clog
1 +τk
|A|
(via (11))
≤τklog 2
c.
Here, the last inequality follows from Lemma 7 (stated and proved below). Hence, all the inequalities
in the derivation above must hold with equality. In particular, we must have|A|=τkby the quality
condition of Lemma 7. Therefore, for distinct setsS, S′∈I∗it holds thatS∩S′=∅. Hence, as desired,
I∗is a solution of theERSPinstance.
This completes the correctness of the reduction, and the theorem stands proved.
Lemma 7.For anya >0and for allx∈(0, a]it holds thatxlog(1+a
x)≤alog 2. Furthermore, the equality
holds iffx=a.
Proof.Writef(x) :=xlog(1 +a
x). At the end points of the domain(0, a], the functionf(·)satisfies:
f(a) =alog(2)and
lim
x→0+f(x) = lim
x→0+xlog(a+x)−xlogx= lim
x→0+xlog(a+x)−lim
x→0+xlog(x) = 0−0 = 0.
Note thatf′(x) = log(1 +a
x)−a
a+x. We will show thatf′(x)>0for allx∈(0, a]which will conclude the
proof.
Case 1:x∈(0,a
2]. We havelog(1 +a
x)≥log(1 +a
a/2) = log(3)>1. On the other hand,a
a+x≤1.
Case 2:x∈(a
2, a]. In this case,log(1 +a
x)≥log(1 +a
a) = log(2)>0.693. However,a
a+x<a
a+a
2=2
3≤
0.667.
Therefore,f′(x) = log(1 +a
x)−a
a+x>0for allx∈(0, a], which completes the proof.
4.2 Algorithm for the Multi-Attribute Setting
This section details Algorithm 2, based on which we obtain Theorem 4. The algorithm greedily selects
thekneighbors for the given queryq. Specifically, the algorithm iteratesktimes, and in each iteration,
it selects a new vector (from the given setP) whose inclusion in the current solution ALGyields that
maximum increase inlog NSW(Line 3). Afterkiterations, the algorithm returns thekselected vectors
ALG.
14

Algorithm 2:MultiNashANN: Approximation algorithm in the multi-attribute setting
Input:Queryq∈Rd, and the set of input vectorsP⊂Rd
1Initialize ALG=∅.
2fori= 1tokdo
3Setbv= arg maxv∈P\ALG 
log NSW(ALG∪ {v})−log NSW(ALG)
.
4Update ALG←ALG∪ {bv}.
5returnALG
4.3 Proof of Theorem 4
We now establish Theorem 4, which provides the approximation ratio achieved by Algorithm 2.
Theorem 4.In the multi-attribute setting with parameterη= 1, there exists a polynomial-time algo-
rithm (Algorithm 2) that, given any queryq∈Rd, finds a size-ksubsetALG⊆Pwithlog NSW(ALG)≥ 
1−1
e
log NSW(OPT); here,OPTdenotes an optimal solution for the optimization problem (2).
Proof.For each subsetS⊆P, write functionf(S) := log NSW(S). Since parameterη= 1, we have
f(∅) = 0and the function is nonnegative. Moreover, we will show that this set functionf: 2P→R +
is monotone and submodular. Given that Algorithm 2 follows the marginal-gain greedy criterion in each
iteration, it achieves a(1−1
e)-approximation for the submodular maximization problem (2).
To establish the monotonicity off, consider any pair of subsetsS⊆T⊆P. Here, for eachℓ∈[c], we
haveD ℓ∩S⊆D ℓ∩T. Hence,u ℓ(S)≤u ℓ(T). Further, sincelogis an increasing function, it holds that
log(u ℓ(S) + 1)≤log(u ℓ(T) + 1), for eachℓ∈[c]. Hence,f(S)≤f(T), and we obtain thatfis monotone.
For submodularity, letS⊆T⊆Pbe any two subsets and letw∈P\T. WriteS+wandT+wto
denote the setsS∪ {w}andT∪ {w}, respectively. Here, we have
f(S+w)−f(S)−f(T+w) +f(T)
=1
cX
ℓ∈[c]log 
1 +P
v∈D ℓ∩(S+w) σ(q, v)
1 +P
v∈D ℓ∩Sσ(q, v)·1 +P
v∈D ℓ∩Tσ(q, v)
1 +P
v∈D ℓ∩(T+w) σ(q, v)!
=1
cX
ℓ∈atb(w)log
 
1 +σ(q, w)
1 +P
v∈D ℓ∩Sσ(q, v)!
· 
1 +σ(q, w)
1 +P
v∈D ℓ∩Tσ(q, v)!−1

=1
cX
ℓ∈atb(w)log 
1 +σ(q, w)
1 +u ℓ(S)
·
1 +σ(q, w)
1 +u ℓ(T)−1!
≥0(u ℓ(S)≤u ℓ(T)forS⊆T)
Therefore, upon rearranging, we obtainf(S+w)−f(S)≥f(T+w)−f(T), i.e.,fis submodular.
Hence, Algorithm 2, which follows marginals-gain greedy selection, achieves a(1−1
e)-approximation
[NWF78] for the optimization problem (2).
5 Experimental Evaluations
In this section, we validate the welfare-based formulations and the performance of our proposed algo-
rithms against existing methods on a variety of real and semi-synthetic datasets. We perform three sets
of experiments:
15

• In the first set of experiments (Figure 4), we compareNash-ANN(Algorithm 1) with prior work on
hard-constraint based diversity [AIK+25]. Here, we show thatNash-ANNstrikes a balance between
relevance and diversity both in the single- and multi-attribute settings.
• In the second set of experiments (Figure 5), we study the effect of varying the exponent parameter
pin thep-NNS objective on relevance and diversity, in both single- and multi-attribute settings.
• In the final set of experiments (Tables 2 and 3), we compare our algorithm,Nash-ANN(with prov-
able guarantees), and a heuristic we propose to improve the runtime ofNash-ANN. The heuristic
directly utilizes a standardANNalgorithm to first fetch a sufficiently large candidate set of vectors
(irrespective of their attributes). Then, it applies the greedy procedure for Nash social (orp-mean)
welfare maximization (similar to Lines 4-6 in Algorithm 1) only within this set.
In what follows, we provide the details of the experimental set-ups, the baseline algorithms and the
results of the experiments.
Additional plots for the experiments appear in Appendices C.1 and C.2.
5.1 Metrics for Measuring Relevance and Diversity
Relevance Metrics: In our experiments, we capture the relevance of a solution to the query through two
metrics detailed below.
1.Approximation Ratio: For a given queryq, letSbe the set ofkvectors returned by an NNS
algorithm that we wish to study, and letObe thekmost similar vectors toqinP. Then the approx-
imation ratio of the algorithm is defined as the ratioP
v∈Sσ(q,v)P
v∈Oσ(q,v). Therefore, a higher approximation
ratio indicates a more relevant solution. Note that the highest possible value of this metric is1.
2.Recall: For a given queryq, letSbe the set ofkvectors returned by an NNS algorithm that we wish
to study and letObe thekmost similar vectors toqinP. The recall of the algorithm is defined
as the quantity|S∩O|
|O|. Therefore, higher the recall, more relevant the solution, and the maximum
possible value of recall is1.
Remark1.Although recall is a popular metric in the context of the standard NNS problem, it is important
to note that it is a fragile metric when the objective is to retrieve a relevant and diverse set of vectors for
a given query. This can be illustrated with the following stylized example in the single-attribute setting.
Suppose for a given queryq, all the vectors in the similarity-wise optimal setOhave similarity1and
share the same attributeℓ∗∈[c], i.e., for eachu∈Owe haveσ(q, u) = 1andatb(u) =ℓ∗. That is, the
setOof thekmost similar vectors toqare not at all diverse. However, it is possible to have another
setSofkvectors each with a distinct attribute andσ(q, v) = 0.99for eachv∈S. Such a set provides
a highly relevant set of vectors that are also completely diverse. However, for the setS, the recall is
actually0(sinceS∩O=∅), but the approximation ratio is0.99. Hence, in the context of diverse near-
est neighbor search problem, approximation ratio may be a more meaningful relevance metric than recall.
Diversity Metrics: To measure the diversity of the solutions obtained by various algorithms, we consider
the following metrics.
•Entropy: LetS⊆Pbe a size-ksubset computed by an algorithm. Then the entropy of the set
Sin the single-attribute setting is given by the quantityP
ℓ∈[c]:p ℓ>0−pℓlog(p ℓ)wherep ℓ=|S∩D ℓ|
|S|.
Note that a higher entropy value indicates greater diversity. Moreover, it is not hard to see that the
highest possible value of entropy islog(k)(achieved whenScontains at most1vector from each
attribute).
16

Table 1: Summary of considered datasets. For synthetic attributes, we use two strategies: clustering-
based (suffixed byClus) and distribution-based (suffixed byProb), see Section 5.2 for details.
Dataset # Input Vectors # Query Vectors Dimension Attributes
Amazon 92,092 8,956 768 product color
ArXiv 200,000 50,000 1536 year, paper category
Sift1m 1,000,000 10,000 128 synthetic
Deep1b 9,990,000 10,000 96 synthetic
•Inverse Simpson Index: For a given setS⊆Pin the single-attribute setting, the inverse Simpson
index is defined as1Pc
ℓ=1p2
ℓwherep ℓis the same as in the definition of entropy above. A higher
value of this metric indicates greater diversity.
•Distinct Attribute Count: In the single-attribute setting, the distinct attribute count of a setS⊆P
is the number of different attributes that have at least one vector inS, i.e., the count is equal to
|{ℓ∈[c]| |S∩D ℓ|>0}|.
Note that the diversity metrics defined above are for the single-attribute setting. In the multi-attribute
setting, in our experiments, we focus on settings where the attribute set[c]is partitioned intomsets
{Ci}m
i=1(i.e.,[c] =⊔m
i=1Ci) and every input vectorv∈Pis associated with exactly one attribute from
eachC i. In particular,|atb(v)|=mand|atb(v)∩C i|= 1for each1≤i≤m. We call eachC i
an attribute class. To measure diversity in the multi-attribute setting, we consider the aforementioned
diversity metrics like entropy and inverse Simpson index restricted to an attribute classC i. For instance,
the entropy a setS⊆Prestricted to a particularC iis given byP
ℓ∈Ci:pℓ>0−pℓlog(p ℓ), wherep ℓ=|S∩D ℓ|
|S|.
Similarly, the inverse Simpson index of a setS⊆Prestricted toC iis given by1P
ℓ∈Cip2
ℓ.
5.2 Experimental Setup and Datasets
Hardware Details.All the experiments were performed in memory on an Intel(R) Xeon(R) Silver4314
CPU (64cores,2.40GHz) with128GB RAM. We set the number of threads to32.
Datasets.We report results on both semi-synthetic and real-world datasets consistent with prior works
[AIK+25]. These are summarized in Table 1 and detailed below.
1.Amazon Products Dataset(Amazon): The dataset, also known as the Shopping Queries Image
Dataset (SQID) [GCT24], includes vector embeddings of about190,000product images and about
9,000text queries by users. The embeddings of both product images and query texts are obtained
via OpenAI’s CLIP model [RKH+21], which maps both images and texts to a shared vector space.
Given this dataset, our task is to retrieve relevant and diverse product images for a given text
query. SQID also contains metadata for every product image, such as product image url, product id,
product description, product title, and product color. The dataset is publicly available on Hugging
Face platform.6
For this dataset, we choose the set of all possible product colors in the dataset as our set of attributes
[c]. We noted that for a lot of products, the color of the product in its image did not match the
product color in the associated metadata. Hence, to associate a clean attribute (color) to each
vector (product) in the dataset, we use the associated metadata as follows: we assign to the vector
the majority color among the colors listed in the product color, product description, and title of
the product. In case of a tie, we assign a separate attribute (color) called ‘color mix’. Further, we
remove from consideration product images whose metadata does not contain any valid color names.
6https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset
17

Figure 2: Distribution of product colors in the processed (cleaned)Amazondataset.
The processed dataset contains92,092vector embeddings of product images and constitutes our
setP. Note that the dataset exhibits a skewed color distribution, shown in Figure 2, with dominant
colors such as black and white. No processing is applied to the query set which contains8,956
vectors. The vector embeddings of both images and queries ared= 768dimensional. We use
σ(u, v) = 1 +u⊤v
∥u∥·∥v∥as the similarity function between two vectorsuandv. Since the CLIP model
was trained using the cosine similarity metric in the loss function (see [GCT24], Section 4.2), this
similarity function is a natural choice for theAmazondataset.
2.ArXiv OpenAI Embedding(ArXiv): This dataset published by Cornell University consists of vector
embeddings and metadata for approximately250,000machine learning papers from arXiv.org [Wes22].
The vector embeddings of the papers were generated via OpenAI’stext-embedding-ada-002model,
executed on each paper’s title, authors, year, and abstract. The dataset is publicly available on Kag-
gle [Wes22].7Note that there are no user queries prespecified in this dataset.
We use this dataset to study both the single- and multi-attribute setting. For the single-attribute
setting, we only consider the year in which a paper was last updated as its attribute. For the multi-
attribute setting, the paper’s update year and its arXiv category are the two associated attributes.
Therefore, we havem= 2attribute classes: update years and arXiv categories. Specifically, for
the experiments we only consider papers with update-year between2012and2025and belonging
to one or more of the following arXiv categories:cs.ai,math.oc,cs.lg,cs.cv,stat.ml,cs.ro,
cs.cl,cs.ne,cs.ir,cs.sy,cs.hc,cs.cr,cs.cy,cs.sd,eess.as, andeess.iv.
(a)
 (b)
Figure 3: Distribution of (a) arXiv categories (b) last update year in theArXivdataset.
7https://www.kaggle.com/datasets/awester/arxiv-embeddings
18

Since this dataset does not contain predefined queries, we randomly split the dataset in4 : 1ratio,
and use the larger part as the setPand the smaller part as the queries. Here, by using the vector
embeddings of papers themselves as queries, we aim to simulate the task of finding papers similar to
a given query paper. The similarity function used for this dataset is the reciprocal of the Euclidean
distance, i.e.,σ(u, v) =1
∥u−v∥+δ, for any two vectorsuandv; here,δ >0is a small constant to
avoid division by zero in case∥u−v∥= 0. Typically, we setδ=η, whereηis the smoothening
parameter used in the definition ofNSW(·). The distribution of the attributes across the vectors is
shown in Figure 3.
3.SIFT Embeddings: This is a standard benchmarking dataset for approximate nearest neighbor
search in the Euclidean distance metric [Ten25]. The dataset consists of SIFT vector embeddings
ind= 128dimensional space. In particular, here input setPcontains1,000,000vectors and we
have10,000vectors as queries. The embeddings are available at [Ten25].8Note that this dataset
does not contain any metadata that can be naturally used to assign attributes to the given vectors.
Therefore, we utilize the following two methods for synthetically assigning attributes to the input
vectors.
•Clustering-based(Sift1m- (Clus)): Since attributes such as colors often occupy distinct re-
gions in the embedding space, we apply k-means clustering [Llo82; McQ67] to identify20
clusters. Each cluster is then assigned a unique index, and vectors in the same cluster are
associated with the index of the cluster, which serves as our synthetic attribute. This simulates
a single-attribute setting withc= 20.
To simulate the multi-attribute setting, we extend the above method of attribute generation.
Given an input vectorvof128dimensions, we split it into4vectors{vi}4
i=1, each of32dimen-
sions. In particular, the vectorv1consists of the first32components ofv, the vectorv2is ob-
tained from the next32components ofv, so on. Next, for eachi∈[4], we apply k-means clus-
tering onPi:={vi:v∈P}to identify20clusters. We writeC i={ci
1, . . . , ci
20}to denote these
20clusters forPiand if vectorvibelongs to a clusterci
j, thenviis assigned the attributeci
j,
i.e., we setatb(vi) =ci
j. Finally, for a vectorv∈P, we assignatb(v) ={atb(v1), . . . ,atb(v4)}.
Note that this method of simulating the multi-attribute setting yieldsm= 4attribute classes.
•Probability distribution-based(Sift1m- (Prob)): As in prior work [AIK+25], we also con-
sider a setting wherein the input vectors have randomly assigned attributes. Specifically, we
consider the single-attribute setting withc= 20. Here, we assign each vectorv∈Pan at-
tribute as follows: with probability0.9, select an attribute from{1,2,3}uniformly at random,
otherwise, with the remaining0.1probability, select an attribute from{4, . . . ,20}uniformly at
random. This results in a skewed distribution of vectors among attributes that mimics real-
world settings (e.g., market dominance by a few sellers).
4.Deep Descriptor Embeddings: This is another standard dataset for nearest neighbor search [Ten25].9
The version of the dataset used in the current work contains9,990,000input vectors and10,000
separate query vectors. Here, the vectors are96dimensional and the distances between them are
evaluated using the cosine distance.
As in the case of SIFT dataset, input vectors inDeep1bdo not have predefined attributes. Hence,
we use the above-mentioned methods (based on clustering and randomization) to synthetically
assign attributes to the vectors. This gives us the clustering-basedDeep1b- (Clus)and probability
distribution-basedDeep1b- (Prob)versions of the dataset.
8https://www.tensorflow.org/datasets/catalog/sift1m
9https://www.tensorflow.org/datasets/catalog/deep1b
19

Choice of Parameterη:For our methods, we tune and set the smoothing parameter,η, to0.01for
theArXiv,Sift1m- (Clus)andSift1m- (Prob)datasets, and set it to0.0001to analyzep-NNS. For other
datasets, namelyAmazon,Deep1b- (Clus)andDeep1b- (Prob), we setηto50.
5.3 Algorithms
Next, we describe the algorithms executed in the experiments.
1.ANN: This is the standard ANN algorithm that aims to maximize the similarity of the retrieved vectors
to the given query without any diversity considerations. In our experiments, we use the graph
based DiskANN method of [SDK+19] as the standard ANN algorithm. We instantiate DiskANN with
candidate list sizeL= 2000and the maximum graph degree as128.10Here, we also set the pruning
factor at1.3, which is consistent with the existing recommendation in [AIK+25].
2.Div-ANN: This refers to the algorithm of [AIK+25] that solves the hard- constraint-based formulation
for diversity in the single-attribute setting. Recall that [AIK+25] aims to maximize the similarity of
the retrieved vectors to the given query subject to the constraint that no more thank′vectors in the
retrieved set should have the same attribute. Note that the smaller the value ofk′, the more diverse
the retrieved set of vectors. Moreover,k′has to be provided as an input to this algorithm. In our
experiments, we set different valuesk′, such ask′∈ {1,2,5}whenk= 10, andk′∈ {1,2,5,10}
whenk= 50.
3.Nash-ANNandp-mean-ANN:Nash-ANNrefers to Algorithm 1 andp-mean-ANNrefers to Algorithm 3
(stated in Appendix B). Recall that Algorithm 1 and Algorithm 3 optimally solve the NaNNS and
thep-NNS problems, respectively, in the single-attribute setting given access to an exact nearest
neighbor search oracle (Theorems 1 and 11). Further note that thep-mean welfare function,M p(·),
reduces to the Nash social welfare (geometric mean) when the exponent parameterp→0+. For
readability and at required places, we will writep= 0to denoteNash-ANN. We conduct experiments
with varying values ofp∈ {−10,−1,−0.5,0,0.5,1}.
4.Multi Nash-ANNandMulti Div-ANN: In the multi-attribute setting, there are no prior methods to
address diversity. Hence, for comparisons, we first fetchL= 10000candidate vectors fromPfor
each queryq, using the standardANNmethod, and then apply the following algorithms on the candi-
date vectors: (i) our algorithm for the NaNNS problem in the multi-attribute setting (Algorithm 2),
which we term asMulti Nash-ANN, and (ii) an adaptation of the algorithm of [AIK+25], referred
hereon asMulti Div-ANN, which greedily selects the most similar vectors to the query subject to
the constraint that there are no more thank′vectors from each attribute.11We compareMulti
Nash-ANN(p= 0) againstMulti Div-ANNunder different choices ofk′.
5.Multi p-mean-ANN: In the multi-attribute setting, we also implement an analogue ofMulti Nash-ANN
that (in lieu ofNSW) focuses on thep-mean welfare,M p. The objective in these experiments is
to understand the impact of varying the parameterpand the resulting tradeoff between relevance
and diversity. Here, for each given query, we first fetch a set ofL= 10000candidate vectors using
ANN. Then, we populate a set ofkvectors by executing a marginal-gains greedy method over theL
candidate vectors. In particular, we iteratektimes, and in each iteration, select a new candidate
vector that: (i) forp∈(0,1], yields the maximum increase inM p(·)p, or (ii) forp <0, leads to the
maximum decrease inM p(·)p.
10Both these choices are sufficiently larger than the standard values,L= 200and maximum graph degree64.
11Note that one vector can have multiple attributes, hence contributing to the constraint of multiple attributes. Therefore, the
issue of identifying an appropriatek′is exacerbated on moving to the multi-attribute setting.
20

Figure 4:Columns 1 and 2- Comparison of approximation ratio versus entropy trade-offs betweenNash-ANN,
andDiv-ANNwith varyingk′, fork=50onAmazonandDeep1b- (Clus)datasets in the single-attribute setting.
Columns 3 and 4- Comparison of approximation ratio versus entropy trade-offs (across attribute classesC 1and
C2) betweenMulti Nash-ANN, andMulti Div-ANNwith varyingk′onSift1m- (Clus)dataset withk= 50in the
multi-attribute setting.
5.4 Results: Balancing Relevance and Diversity
Single-attribute setting.We first compare, in the single-attribute setting, the performance of our al-
gorithm,Nash-ANN, withANNandDiv-ANN(with different values ofk′). The results for theAmazonand
Deep1b- (Clus)datasets withk= 50are shown in Figure 4 (columns one and two). Here,ANNfinds the
most relevant set of neighbors (approximation ratio close to1), albeit with the lowest entropy (diversity).
Moreover, as can be seen in the plots, the most diverse (highest entropy) solution is obtained when we
set, inDiv-ANN,k′= 1; this restricts eachℓ∈[c]to contribute at most one vector in the output ofDiv-ANN.
Also, note that one can increase the approximation ratio (i.e., increase relevance) ofDiv-ANNwhile in-
curring a loss in entropy (diversity), by increasing the value of the constraint parameterk′. However,
selecting a ‘right’ value fork′is non-obvious, since this choice needs to be tailored to the dataset and,
even within it, to queries (recall the “blue shirt” query in Figure 1).
By contrast,Nash-ANNdoes not require such ad hoc adjustments and, by design, finds a balance be-
tween relevance and diversity. Indeed, as can be seen in Figure 4 (columns 1 and 2),Nash-ANNmaintains
an approximation ratio close to1while achieving diversity similar toDiv-ANNwithk′= 1. Moreover,
Nash-ANNPareto dominatesDiv-ANNwithk′= 2forAmazondataset andk′= 5forDeep1b- (Clus)
dataset on the fronts of approximation ratio and entropy. The results for other datasets and metrics
follow similar trends and are given in Appendix C.1.
Multi-attribute setting.In the multi-attribute setting, we report results forMulti Nash-ANNandMulti
Div-ANNon theSift1m- (Clus)dataset (Figure 4, columns 3 and 4) fork= 50andc= 80. These
eighty attributes are partitioned into four sets,{C i}4
i=1, with each set of size|C i|= 20, i.e.,[c] =∪4
i=1Ci.
Further, each input vectorvis associated with four attributes (|atb(v)|= 4), one from eachC i; see
Section 5.2 for further details. Here, to quantify diversity, we separately consider for eachi∈[4], the
entropy across attributes within aC i. In Figure 4 (columns 3 and 4), we compare the approximation ratio
versus entropy trade-offs ofMulti Nash-ANNagainstMulti Div-ANNwith varyingk′. Here we show the
results for attribute classesC 1(column 1) andC 2(column 2) whereas the results forC 3andC 4are
given in Figure 30. We observe thatMulti Nash-ANNmaintains a high approximation ratio (relevance)
while simultaneously achieving significantly higher entropy (higher diversity) thanANN. By contrast, in
the constraint-based methodMulti Div-ANN, low values ofk′lead to a notable drop in the approximation
ratio, whereas increasingk′reduces entropy. For example, fork′below15, one obtains approximation
ratio less than0.8, and to reach an approximation ratio comparable toMulti Nash-ANN, one needsk′
as high as30. Additional results for theArXivdataset in the multi-attribute setting are provided in
Appendix C.2, and they exhibit trends similar to the ones in Figure 4. These findings demonstrate that
Multi Nash-ANNachieves a balance between relevance and diversity. In summary
21

Figure 5:Columns 1 and 2- Approximation ratio versus entropy trade-offs forp-mean-ANNat variouspvalues,
fork=50onAmazonandDeep1b- (Clus)datasets in the single-attribute setting.Columns 3 and 4- Approxima-
tion ratio versus entropy trade-offs (across attribute classesC 1andC 2) forMulti p-mean-ANNwith varyingpon
Sift1m- (Clus)dataset withk=50in the multi-attribute setting.
Across datasets, and in both single- and multi-attribute settings, the Nash formulation consis-
tently improves entropy (diversity) overANN, while maintaining an approximation ratio (rele-
vance) of roughly above0.9. By contrast, the hard-constrained formulation is highly sensitive
to the choice of the constraint parameterk′, and in some cases, incurs a substantial drop in
approximation ratio (even lower than0.2).
Results forp-NNS.Recall that, selecting the exponent parameterp∈(−∞,1]enables us to interpolate
p-NNS between the standard NNS problem (p= 1), NaNNS (p= 0), and optimizing solely for diversity
(p→ −∞). We executep-mean-ANNforp∈ {−10,−1,−0.5,0,0.5,1}in both single- and multi-attribute
settings and show that a trade-off between relevance (approximation ratio) and diversity (entropy) can
be achieved by varyingp.
For the single-attribute setting, Figure 5 (columns 1 and 2), and for the multi-attribute setting, Figure
5 (columns 3 and 4) capture this feature onSift1m- (Clus)dataset withk= 50: For lower values ofp,
we have higher entropy but lower approximation ratio, whilep= 1matchesANN. For the multi-attribute
setting, we show results for attribute classesC 1(column 3) andC 2(column 4) in Figure 5, whereas the
results forC 3andC 4are shown in Figure 30. Note that in the multi-attribute setting,Multip-mean-ANN
withp=−10Pareto dominatesMulti Div-ANNwithk′= 1in terms of approximation ratio and entropy.
Moreover, analogous results are obtained for other datasets and metrics; see Appendix C.1 and C.2.
5.5 A Faster Heuristic for the Single Attribute Setting:p-FetchUnion-ANN
Further, we empirically study a faster heuristic algorithm for our NSW andp-mean welfare formulations.
Specifically, the heuristic—calledp-FetchUnion-ANN—first fetches a sufficiently large candidate set of
vectors (irrespective of their attributes) using theANNalgorithm. Then, it applies the Nash (orp-mean)
selection (similar to Line 5 in Algorithm 1 or Lines 6-8 in Algorithm 3) within this set. That is, instead of
starting out withkneighbors for eachℓ∈[c](as in Line 1 of Algorithm 1), the alternative here is to work
with sufficiently many neighbors from the set∪c
ℓ=1Dℓ.
Table 2 shows that this heuristic consistently achieves performance comparable top-mean-ANNin
terms of approximation ratio and diversity in theSift1m- (Clus)dataset. Sincep-FetchUnion-ANN
retrieves a larger pool of vectors with high similarity, it achieves improved approximation ratio over
p-Mean-ANN. However, it comes at the cost of reduced entropy, which can be explained by the fact that
in restricting its search to an initially fetched large pool of vectors,p-FetchUnion-ANNmay miss a more
diverse solution that exists over the entire dataset. Another important aspect ofp-FetchUnion-ANNis
that, because it retrieves all neighbors from the union at once, the heuristic delivers substantially higher
throughput (measured as queries answered per second, QPS) and therefore lower latency, which can
be seen in Table 3 for theSift1m- (Clus)dataset. In particular,p-FetchUnion-ANNserves almost10×
22

Table 2: Comparison of performance acrosspvalues forSift1m- (Clus)atk= 50.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.749±0.051 0.810±0.045 0.812±0.043 0.846±0.036 0.932±0.028 1.000±0.000
p-FetchUnion-ANN0.979±0.014 0.980±0.013 0.980±0.013 0.981±0.012 0.983±0.011 1.000±0.000
ANN1.000±0.000
Div-ANN(k′=1) 0.315±0.021
Entropyp-Mean-ANN4.285±0.012 4.293±0.002 4.293±0.001 4.197±0.045 3.506±0.275 0.892±0.663
p-FetchUnion-ANN2.235±0.802 2.238±0.802 2.239±0.802 2.239±0.802 2.231±0.800 0.892±0.663
ANN0.892±0.663
Div-ANN(k′=1) 4.289±0.053
Table 3: Comparison of performance on QPS and Latency acrossponSift1m- (Clus)dataset fork= 50.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Query per Secondp-Mean-ANN120.86 115.78 107.01 135.98 122.59 122.59
p-FetchUnion-ANN1324.53 1324.62 1337.28 1442.03 1443.38 1327.03
Latency (µs)p-Mean-ANN264566.00 276129.00 298804.00 230318.00 235144.00 260800.00
p-FetchUnion-ANN24133.80 24134.00 23907.00 22170.20 22149.30 28990.40
99.9th percentile of Latencyp-Mean-ANN484601.00 513036.00 478821.00 477925.00 482777.00 479132.00
p-FetchUnion-ANN52943.40 53474.70 54283.40 56128.70 53082.20 24088.70
more queries onSift1m- (Clus)dataset thanp-mean-ANN. The latency values exhibit a similar trend with
reductions of similar magnitude. In summary, these observations position the heuristic as a notably fast
method for NaNNS andp-NNS, particularly whencis large. We provide comparison ofp-FetchUnion-ANN
withp-mean-ANNon other datasets in Tables 4 to 9 in Appendix C.3.
6 Conclusion
In this work, we formulated diversity in neighbor search with a welfarist perspective, using Nash social
welfare (NSW) andp-mean welfare as the underlying objectives. Our NSW formulation balances diver-
sity and relevance in a query-dependent manner, satisfies several desirable axiomatic properties, and is
naturally applicable in both single-attribute and multi-attribute settings. With these properties, our for-
mulation overcomes key limitations of the prior hard-constrained approach [AIK+25]. Furthermore, the
more generalp-mean welfare interpolates between complete relevance (p= 1) and complete diversity
(p=−∞), offering practitioners a tunable parameter for real-world needs. Our formulations also ad-
mit provable and practical algorithms suited for low-latency scenarios. Experiments on real-world and
semi-synthetic datasets validate their effectiveness in balancing diversity and relevance against existing
baselines.
An important direction for future work is the design of sublinear-time approximation algorithms, in
both single- and multi-attribute settings, that directly optimize our welfare objectives as part of ANN
algorithms, thereby further improving efficiency. Another promising avenue is to extend welfare-based
diversity objectives to settings without explicit attributes.
Acknowledgement
Siddharth Barman, Nirjhar Das, and Shivam Gupta acknowledge the support of the Walmart Center for
Tech Excellence (CSR WMGT-23-0001) and an Ittiam CSR Grant (OD/OTHR-24-0032).
23

References
[ADP80] Giorgio Ausiello, Alessandro D. D’Atri, and Marco Protasi. Structure preserving reductions
among convex optimization problems.Journal of Computer and System Sciences, 21(1):136–
153, 1980.
[AI08] Alexandr Andoni and Piotr Indyk. Near-optimal hashing algorithms for approximate nearest
neighbor in high dimensions.Communications of the ACM, 51(1):117–122, 2008.
[AIK+25] Piyush Anand, Piotr Indyk, Ravishankar Krishnaswamy, Sepideh Mahabadi, Vikas C. Raykar,
Kirankumar Shiragur, and Haike Xu. Graph-based algorithms for diverse similarity search.
InForty-second International Conference on Machine Learning, 2025.
[ALMK22] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias. InEthics of data
and analytics, pages 254–264. Auerbach Publications, 2022.
[AMN+98] Sunil Arya, David M Mount, Nathan S Netanyahu, Ruth Silverman, and Angela Y Wu. An
optimal algorithm for approximate nearest neighbor searching fixed dimensions.Journal of
the ACM (JACM), 45(6):891–923, 1998.
[BBM18] Dmitry Baranchuk, Artem Babenko, and Yury Malkov. Revisiting the inverted indices for
billion-scale approximate nearest neighbors. InComputer Vision – ECCV 2018: 15th European
Conference, Munich, Germany, September 8–14, 2018, Proceedings, Part XII, page 209–224,
Berlin, Heidelberg, 2018. Springer-Verlag.
[BKL06] Alina Beygelzimer, Sham Kakade, and John Langford. Cover trees for nearest neighbor. In
Proceedings of the 23rd International Conference on Machine Learning, ICML ’06, page 97–104,
New York, NY, USA, 2006. Association for Computing Machinery.
[CG98] Jaime Carbonell and Jade Goldstein. The use of mmr, diversity-based reranking for reorder-
ing documents and producing summaries. InProceedings of the 21st annual international
ACM SIGIR conference on Research and development in information retrieval, pages 335–336,
1998.
[CKPS10] A. Camerra, E. Keogh, T. Palpanas, and J. Shieh. isax 2.0: Indexing and mining one billion
time series. In2013 IEEE 13th International Conference on Data Mining, pages 58–67, Los
Alamitos, CA, USA, dec 2010. IEEE Computer Society.
[DSM+21] Kunal Dahiya, Deepak Saini, Anshul Mittal, Ankush Shaw, Kushal Dave, Akshay Soni, Himan-
shu Jain, Sumeet Agarwal, and Manik Varma. Deepxml: A deep extreme multi-label learning
framework applied to short text documents. InProceedings of the 14th International Confer-
ence on Web Search and Data Mining, WSDM ’21, New York, NY, USA, 2021. Association for
Computing Machinery.
[FH89] Evelyn Fix and J. L. Hodges. Discriminatory analysis. nonparametric discrimination: Con-
sistency properties.International Statistical Review / Revue Internationale de Statistique,
57(3):238–247, 1989.
[FXWC19] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. Fast approximate nearest neighbor
search with the navigating spreading-out graphs.PVLDB, 12(5):461 – 474, 2019.
[GCT24] Marie Al Ghossein, Ching-Wei Chen, and Jason Tang. Shopping queries image dataset (sqid):
An image-enriched esci dataset for exploring multimodal learning in product search.arXiv
preprint arXiv:2405.15190, 2024.
24

[GJ90] Michael R. Garey and David S. Johnson.Computers and Intractability; A Guide to the Theory
of NP-Completeness. W. H. Freeman & Co., USA, 1990.
[IM98] Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the
curse of dimensionality. InProceedings of the thirtieth annual ACM symposium on Theory of
computing, pages 604–613, 1998.
[JDJ17] Jeff Johnson, Matthijs Douze, and Herv ´e J´egou. Billion-scale similarity search with gpus.
arXiv preprint arXiv:1702.08734, 2017.
[KR19] Michael Kearns and Aaron Roth.The ethical algorithm: The science of socially aware algorithm
design. Oxford University Press, 2019.
[Lia19] Search Liaison. Google announces site diversity change to search results, 2019.
[Llo82] Stuart Lloyd. Least squares quantization in pcm.IEEE transactions on information theory,
28(2):129–137, 1982.
[McQ67] James B McQueen. Some methods of classification and analysis of multivariate observations.
InProc. of 5th Berkeley Symposium on Math. Stat. and Prob., pages 281–297, 1967.
[Mou04] Herv ´e Moulin.Fair division and collective welfare. MIT press, 2004.
[MRS08] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Sch ¨utze.Introduction to Infor-
mation Retrieval. Cambridge University Press, USA, 2008.
[MSB+24] Magdalen Dobson Manohar, Zheqi Shen, Guy Blelloch, Laxman Dhulipala, Yan Gu, Har-
sha Vardhan Simhadri, and Yihan Sun. Parlayann: Scalable and deterministic parallel graph-
based approximate nearest neighbor search algorithms. InProceedings of the 29th ACM SIG-
PLAN Annual Symposium on Principles and Practice of Parallel Programming, pages 270–285,
2024.
[MY16] Yury A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search
using hierarchical navigable small world graphs.CoRR, abs/1603.09320, 2016.
[NWF78] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher. An analysis of approximations for maximiz-
ing submodular set functions–i.Math. Program., 14(1):265–294, December 1978.
[RKH+21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agar-
wal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision. In Marina
Meila and Tong Zhang, editors,Proceedings of the 38th International Conference on Machine
Learning, volume 139 ofProceedings of Machine Learning Research, pages 8748–8763. PMLR,
18–24 Jul 2021.
[SAI+24] Harsha Vardhan Simhadri, Martin Aum ¨uller, Amir Ingber, Matthijs Douze, George Williams,
Magdalen Dobson Manohar, Dmitry Baranchuk, Edo Liberty, Frank Liu, Ben Landrum, et al.
Results of the big ann: Neurips’23 competition.arXiv preprint arXiv:2409.17424, 2024.
[SDK+19] Suhas Jayaram Subramanya, Fnu Devvrit, Rohan Kadekodi, Ravishankar Krishnawamy, and
Harsha Vardhan Simhadri. Diskann: Fast accurate billion-point nearest neighbor search on
a single node. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alch ´e-
Buc, Emily B. Fox, and Roman Garnett, editors,Advances in Neural Information Processing
Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019,
8-14 December 2019, Vancouver, BC, Canada, pages 13748–13758, 2019.
25

[SKI16] Kohei Sugawara, Hayato Kobayashi, and Masajiro Iwasaki. On approximately searching for
similar word embeddings. InAnnual Meeting of the Association for Computational Linguistics,
pages 2265–2275, 01 2016.
[SSD+23] Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, and Sanjiv Kumar. Soar: Improved
indexing for approximate nearest neighbor search. InNeural Information Processing Systems,
2023.
[Ten25] TensorFlow. Tensorflow datasets.https://www.tensorflow.org/datasets/catalog/,
2025. Accessed: 2025-07-10.
[Wes22] August Wester. arxiv openai embeddings.https://www.kaggle.com/datasets/awester/
arxiv-embeddings, 2022. Accessed: 2025-07-10.
[WWZ+12] J. Wang, J. Wang, G. Zeng, Z. Tu, R. Gan, and S. Li. Scalable k-nn graph construction
for visual descriptors. In2012 IEEE Conference on Computer Vision and Pattern Recognition,
pages 1106–1113, June 2012.
[WXC+24] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue
Liu, Tei-Wei Kuo, Nan Guan, et al. Retrieval-augmented generation for natural language
processing: A survey.arXiv preprint arXiv:2407.13193, 2024.
26

A Proofs of Examples 1 and 2
This section provides the proofs for Examples 1 and 2. Recall that these stylized examples highlight how
NaNNS dynamically recovers complete diversity or complete relevance based on the data.
Example 1(Complete Diversity via NaNNS).Consider an instance in which, for a given queryq∈Rd,
all vectors inPare equally similar with the query:σ(q, v) = 1for allv∈P. Also, let|atb(v)|= 1for all
v∈P, and writeS∗∈arg maxS⊆P:|S|=k NSW(S). Ifc≥k, then here it holds that|S∗∩D ℓ| ≤1for all
ℓ∈[c].
Proof.Towards a contradiction, suppose there existsT∈arg maxS⊆P:|S|=k NSW(S)such that|T∩D ℓ∗|>
1for someℓ∗∈[c]. Note that according to the setting specified in the example,u ℓ(T) =|T∩D ℓ|+ηfor
allℓ∈[c].
Sincec≥kand|T∩D ℓ∗|>1, there existsℓ′∈[c]such that|T∩D ℓ′|= 0. Letv∗∈T∩D ℓ∗and
v′∈D ℓ′be two vectors. Consider the setT′= (T\ {v∗})∪ {v′}. We have
NSW(T′)
NSW(T)=
(uℓ′(T′) +η)
(uℓ′(T) +η)·(uℓ∗(T′) +η)
(uℓ∗(T) +η)Y
ℓ∈[c]\{ℓ∗,ℓ′}(uℓ(T′) +η)
(uℓ(T) +η)
1
c
=
(1 +η)
η·(uℓ∗(T)−1 +η)
(uℓ∗(T) +η)Y
ℓ∈[c]\{ℓ∗,ℓ′}(uℓ(T) +η)
(uℓ(T) +η)
1
c
=(1 +η)
η·(uℓ∗(T)−1 +η)
(uℓ∗(T) +η)1
c
=uℓ∗(T)−1 +ηu ℓ∗(T) +η2
ηuℓ∗(T) +η21
c
>1(u ℓ∗(T)≥2)
Therefore, we haveNSW(T′)>NSW(T), which contradicts the optimality ofT. Hence, we must have
|T∩D ℓ| ≤1for allℓ∈[c], which proves the claim.
Example 2(Complete Relevance via NaNNS).Consider an instance in which, for a given queryq∈Rd
and a particularℓ∗∈[c], only vectorsv∈D ℓ∗have similarityσ(q, v) = 1and all other vectorsv′∈P\D ℓ∗
have similarityσ(q, v′) = 0. Also, suppose that|atb(v)|= 1for eachv∈P, along with|D ℓ∗| ≥k. Then,
for a Nash optimal solutionS∗∈arg maxS⊆P,|S|=k NSW(S), it holds that|S∗∩D ℓ∗|=k. That is, for all
otherℓ∈[c]\ {ℓ∗}we have|S∗∩D ℓ|= 0.
Proof.Towards a contradiction, suppose there existsT∈arg maxS⊆P:|S|=k NSW(S)such that|T∩D ℓ∗|<
k. Therefore, there existsℓ′∈[c]\ {ℓ∗}such that|T∩D ℓ′| ≥1. Letv∗∈D ℓ∗\Tand letv′∈T∩D ℓ′.
Note thatu ℓ′(T) = 0sinceσ(q, v) = 0for allv∈D ℓfor anyℓ∈[c]\ {ℓ∗}. Moreover, we also have
uℓ∗(T) =|T∩D ℓ∗|.
Consider the setT′= (T\ {v′})∪ {v∗}. We have,
NSW(T′)
NSW(T)=
(uℓ′(T′) +η)
(uℓ′(T) +η)·(uℓ∗(T′) +η)
(uℓ∗(T) +η)Y
ℓ∈[c]\{ℓ∗,ℓ′}(uℓ(T′) +η)
(uℓ(T) +η)
1
c
=
(uℓ′(T)−σ(q, v′) +η)
(uℓ′(T) +η)·(uℓ∗(T) +σ(q, v∗) +η)
(uℓ∗(T) +η)Y
ℓ∈[c]\{ℓ∗,ℓ′}(uℓ(T) +η)
(uℓ(T) +η)
1
c
=(0−0 +η)
0 +η·(|T∩D ℓ∗|+ 1 +η)
(|T∩D ℓ∗|+η)1
c
27

=
1 +1
|T∩D ℓ∗|+η1
c
>1.
Therefore, we have obtainedNSW(T′)>NSW(T), which contradicts the optimality ofT. Hence, it
must be the case that|T∩D ℓ∗|=k, which proves the claim.
B Extensions forp-NNS
This section extends our NaNNS results top-mean nearest neighbor search (p-NNS) in the single-attribute
setting. We state our algorithm (Algorithm 3) and present corresponding guarantees (Theorem 11
and Corollary 12) for finding an optimal solution for thep-NNS problem.
Recall that, for anyp∈(−∞,1], thep-mean welfare ofcagents with utilitiesw 1, . . . , w cis given by
Mp(w1, . . . , w c):= 1
cPc
ℓ=1wp
ℓ1
p.
Here, as in Section 2.1, the utility is defined asu ℓ(S) =P
v∈S∩D ℓσ(q, v), for any subset of vec-
torsSand attributeℓ∈[c]. Also, with a fixed smoothing constantη >0, we will writeM p(S):=
Mp(u1(S) +η, . . . , u c(S) +η), and thep-NNS problem is defined as
max
S⊆P:|S|=kMp(S)(12)
Throughout this section, we will write letF ℓ(i):=Pi
j=1σ(q, vℓ
(j)) +ηp
, for each attributeℓ∈[c];
here, as before,vℓ
(j)denotes thejthmost similar vector toqin bDℓ, the set ofkmost similar vectors toq
fromD ℓ.
Lemma 8(Decreasing Marginals forp >0).Fix anyp∈(0,1]and attributeℓ∈[c]. Then, for indices
1≤i′< i≤k, it holds that
Fℓ(i′)−F ℓ(i′−1)≥F ℓ(i)−F ℓ(i−1).
Proof.WriteG(j) :=F ℓ(j)−F ℓ(j−1)for allj∈[k], andf ℓ(i) =Pi
j=1σ(q, vℓ
(j))fori∈[k]. We will
establish the lemma by showing thatG(j)is decreasing inj. Towards this, note that, for indicesj≥2,
we have
G(j−1)−G(j)
=F ℓ(j−1)−F ℓ(j−2)−F ℓ(j) +F ℓ(j−1)
= 2F ℓ(j−1)−(F ℓ(j) +F ℓ(j−2))
= 2(f ℓ(j−1) +η)p−((f ℓ(j) +η)p+ (f ℓ(j−2) +η)p)
= 2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
−1
2 
1 + 
1 +σ(q, vℓ
(j−1)) +σ(q, vℓ
(j))
fℓ(j−2) +η!p!!
≥2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
−1
2 
1 + 
1 +2σ(q, vℓ
(j−1))
fℓ(j−2) +η!p!!
(σ(q, vℓ
(j−1))≥σ(q, vℓ
(j));x7→xpis increasing forp∈(0,1]andx≥0)
≥2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
− 
1
2·1 +1
2· 
1 +2σ(q, vℓ
(j−1))
fℓ(j−2) +η!!p!
(x7→xpis concave forp∈(0,1]andx≥0)
= 0.
Therefore,G(j)≤G(j−1)for each2≤j≤k. Equivalently, for indices1≤i′< i≤k, it holds that
G(i′)≥G(i). This completes the proof of the lemma.
28

Algorithm 3:p-Mean-ANN: Algorithm forp-NNS in the single-attribute setting
Input:Queryq∈Rdand, for each attributeℓ∈[c], the set of input vectorsD ℓ⊂Rdand
parameterp∈(−∞,1]\ {0}
1For eachℓ∈[c], fetch thek(exact or approximate) nearest neighbors ofq∈RdfromD ℓ. Write
bDℓ⊆D ℓto denote these sets.
2For everyℓ∈[c]and each indexi∈[k], letvℓ
(i)denote theith most similar vector toqin bDℓ.
3Initialize subset ALG=∅, along with countk ℓ= 0and utilityw ℓ= 0, for eachℓ∈[c].
4while|ALG|< kdo
5ifp∈(0,1]then
6a←arg max
ℓ∈[c]
(wℓ+η+σ(q, vℓ
(kℓ+1)))p−(w ℓ+η)p
.▷Ties broken arbitrarily
7else ifp <0then
8a←arg min
ℓ∈[c]
(wℓ+η+σ(q, vℓ
(kℓ+1)))p−(w ℓ+η)p
.▷Ties broken arbitrarily
9Update ALG←ALG∪ {va
(ka+1)}along withw a←w a+σ(q, va
(ka+1))andk a←k a+ 1.
10returnALG
Lemma 9(Increasing Marginals forp <0).Fix any parameterp∈(−∞,0)and attributeℓ∈[c]. Then,
for indices1≤i′< i≤k, it holds that
Fℓ(i′)−F ℓ(i′−1)≤F ℓ(i)−F ℓ(i−1).
Proof.The proof is similar to that of Lemma 8, except that we now seek the reverse inequality. In
particular, writeG(j) :=F ℓ(j)−F ℓ(j−1)for allj∈[k], andf ℓ(i) =Pi
j=1σ(q, vℓ
(j))fori∈[k].
To show thatG(j)≥G(j−1)for all indices2≤j≤k, note that
G(j−1)−G(j)
= 2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
−1
2 
1 + 
1 +σ(q, vℓ
(j−1)) +σ(q, vℓ
(j))
fℓ(j−2) +η!p!!
≤2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
−1
2 
1 + 
1 +2σ(q, vℓ
(j−1))
fℓ(j−2) +η!p!!
(σ(q, vℓ
(j−1))≥σ(q, vℓ
(j));x7→xpis decreasing forp∈(−∞,0)andx≥0)
≤2(f ℓ(j−2) +η)p  
1 +σ(q, vℓ
(j−1))
fℓ(j−2) +η!p
− 
1
2·1 +1
2· 
1 +2σ(q, vℓ
(j−1))
fℓ(j−2) +η!!p!
(x7→xpis convex forp∈(−∞,0)andx≥0)
= 0.
Therefore, we haveG(j)≥G(j−1)for all2≤j≤k. That is,G(i′)≤G(i)for indices1≤i′< i≤k.
Hence, the lemma stands proved.
Lemma 10.In the single-attribute setting, letALGbe the subset of vectors returned by Algorithm 3 and
Sbe any subset of input vectors with the property that|S∩D ℓ|=|ALG∩D ℓ|, for eachℓ∈[c]. Then,
Mp(ALG)≥M p(S).
29

Proof.Assume, towards a contradiction, that there exists a subset of input vectorsSthat satisfies|S∩
Dℓ|=|ALG∩D ℓ|, for eachℓ∈[c], and still induces p-mean welfare strictly greater than that of ALG. This
strict inequality combined with the fact thatM p(w1, . . . , w c)is an increasing function ofw is implies that
there exists an attributea∈[c]with the property that the utilityu a(S)> u a(ALG). That is,
X
t∈S∩D aσ(q, t)>X
v∈ALG∩D aσ(q, v)(13)
On the other hand, note that the construction of Algorithm 3 and the definition of bDaensure that the
vectors in ALG∩D aare, in fact, the most similar toqamong all the vectors inD a. This observation and
the fact that|S∩D a|=|ALG∩D a|gives usP
v∈ALG∩D aσ(q, v)≥P
t∈S∩D aσ(q, t). This equation, however,
contradicts the strict inequality (13). Therefore, by way of contradiction, we obtain that there does not
exist a subsetSsuch that|S∩D ℓ|=|ALG∩D ℓ|, for eachℓ∈[c], andM p(ALG)< M p(S). The lemma
stands proved.
Theorem 11.In the single-attribute setting, given any queryq∈Rdand an (exact) oracleENNforkmost
similar vectors from any set, Algorithm 3 (p-mean-ANN) returns an optimal solution forp-NNS, i.e., it returns
a size-ksubsetALG⊆Pthat satisfiesALG∈arg maxS⊆P:|S|=k Mp(S). Furthermore, the algorithm runs in
timeO(kc) +Pc
ℓ=1ENN(D ℓ, q), whereENN(D ℓ, q)is the time required by the exact oracle to findkmost
similar vectors toqinD ℓ.
Proof.The running time of the algorithm follows via arguments similar to the ones used in the proof
of Theorem 1.
For the correctness analysis, we divide the proof into two:p∈(0,1]andp <0.
Case 1:p∈(0,1]. Sincex7→xpis an increasing function forx≥0, the problemmax S⊆P,|S|=k Mp(S)is
equivalent tomax S⊆P,|S|=k Mp(S)p= max S⊆P,|S|=k1
cPc
ℓ=1(uℓ(S) +η)p.
The proof here is similar to that of Theorem 1. Letk ℓ=|ALG∩D ℓ|for allℓ∈[c]. Further, let
OPT∈arg maxS⊆P,|S|=k1
cPc
ℓ=1(uℓ(S) +η)pandk∗
ℓ=|OPT∩D ℓ|for allℓ∈[c], where OPTis chosen such
thatPc
ℓ=1|k∗
ℓ−kℓ|is minimized.
We will prove that OPTsatisfiesk∗
ℓ=k ℓfor eachℓ∈[c]. This guarantee, along with Lemma 10,
implies that, as desired, ALGis a p-mean welfare optimal solution.
Assume, towards a contradiction, thatk∗
ℓ̸=kℓfor someℓ∈[c]. Since|OPT|=|ALG|=k, there exist
attributesx, y∈[c]with the property that
k∗
x< kx andk∗
y> ky (14)
Recall that via Lemma 8, we have for any pair of indices1≤i′< i≤k,
Fℓ(i′)−F ℓ(i′−1)≥F ℓ(i)−F ℓ(i−1)(15)
Next, note that for any attributeℓ∈[c], if Algorithm 3, at any point during its execution, has included
k′
ℓvectors of attributeℓin ALG, then at that point the maintained utilityw ℓ=Pk′
ℓ
j=1σ(q, vℓ
(j)). Hence, at
the beginning of any iteration of the algorithm, if thek′
ℓdenotes the number of selected vectors of each
attributeℓ∈[c], then the marginals considered in Line 6 areF ℓ(k′
ℓ+ 1)−F ℓ(k′
ℓ). These observations
and the selection criterion in Line 6 of the algorithm give us the following inequality for the counts
kx=|ALG∩D x|andk y=|ALG∩D y|of the returned solution ALG:
Fx(kx)−F x(kx−1)≥F y(ky+ 1)−F y(ky)(16)
Specifically, equation (16) follows by considering the iteration in whichkth
x(last) vector of attributexwas
selected by the algorithm. Before that iteration the algorithm had selected(k x−1)vectors of attributex,
30

and letk′
ydenote the number of vectors with attributeythat have been selected till that point. Note that
k′
y≤ky. The fact that thekth
xvector was (greedily) selected in Line 6, instead of including an additional
vector of attributey, givesF x(kx)−F x(kx−1)≥F y(k′
y+ 1)−F y(k′
y)≥F y(ky+ 1)−F y(ky); here, the
last inequality follows from equation (15). Hence, equation (16) holds.
Moreover,
Fx(k∗
x+ 1)−F x(k∗
x)≥F x(kx)−F x(kx−1)(via eqns. (14) and (15))
≥F y(ky+ 1)−F y(ky)(via eqn. (16))
≥F y(k∗
y)−F y(k∗
y−1)(17)
The last inequality follows from equations (14) and (15).
Recall thatvℓ
(i)denotes theithmost similar (toq) vector in the set bDℓ. The definition of bDℓensures
thatvℓ
(i)is, in fact, theithmost similar (toq) vector among the ones that have attributeℓ, i.e.,ithmost
similar in all ofD ℓ. Since OPTis an optimal solution, thek∗
ℓ=|OPT∩D ℓ|vectors of attributeℓin OPT
are the most similark∗
ℓvectors fromD ℓ. That is, OPT∩D ℓ=n
vℓ
(1), . . . , vℓ
(k∗
ℓ)o
, for eachℓ∈[c]. This
observation and the definition ofF ℓ(·)imply that thep-th power of OPT’sp-mean welfare satisfies
Mp(OPT)p=1
ccX
ℓ=1Fℓ(k∗
ℓ).(18)
Now, consider a subset of vectorsSobtained from OPTby including vectorvx
(k∗x+1)and removingvy
(k∗y),
i.e.,S=
OPT∪n
vx
(k∗x+1)o
\n
vy
(k∗y)o
. Note that
Mp(S)p−M p(OPT)p=1
c
Fx(k∗
x+ 1)−F x(k∗
x)
+1
c
Fy(k∗
y−1)−F y(k∗
y)
≥0(via eqn. (17))
Hence,M p(S)≥M p(OPT). Given that OPTis ap-mean welfare optimal solution, the last inequality must
hold with equality,M p(S) =M p(OPT), i.e.,Sis an optimal solution as well. This, however, contradicts
the choice of OPTas an optimal solution that minimizesPc
ℓ=1|k∗
ℓ−kℓ|– note thatPc
ℓ=1bkℓ−kℓ<
Pc
ℓ=1|k∗
ℓ−kℓ|, where bkℓ:=|S∩D ℓ|.
Therefore, by way of contradiction, we obtain that|OPT∩D ℓ|=|ALG∩D ℓ|for eachℓ∈[c]. As men-
tioned previously, this guarantee, along with Lemma 10, implies that ALGis a p-mean welfare optimal
solution. This completes the proof of the theorem for the casep∈(0,1].
Case 2:p <0. The arguments here are similar to the ones used in the previous case. However, since the
exponent parameterpis negative, the key inequalities are reversed. For completeness, we present the
proof below.
Note that, with thep <0, the mapx7→xpis a decreasing function forx≥0. Hence, the optimization
problemmax S⊆P,|S|=k Mp(S)is equivalent tomin S⊆P,|S|=k Mp(S)p= min S⊆P,|S|=k1
cPc
ℓ=1(uℓ(S) +η)p.
Letk ℓ=|ALG∩D ℓ|for allℓ∈[c]. Further, let OPT∈arg minS⊆P,|S|=k1
cPc
ℓ=1(uℓ(S) +η)pand
k∗
ℓ=|OPT∩D ℓ|for allℓ∈[c], where OPTis chosen such thatPc
ℓ=1|k∗
ℓ−kℓ|is minimized.
We will prove that OPTsatisfiesk∗
ℓ=k ℓfor eachℓ∈[c]. This guarantee, along with Lemma 10,
implies that, as desired, ALGis a p-mean welfare optimal solution.
Assume, towards a contradiction, thatk∗
ℓ̸=kℓfor someℓ∈[c]. Since|OPT|=|ALG|=k, there exist
attributesx, y∈[c]with the property that
k∗
x< kx andk∗
y> ky (19)
31

Recall that via Lemma 9, we have for any pair of indices1≤i′< i≤k,
Fℓ(i′)−F ℓ(i′−1)≤F ℓ(i)−F ℓ(i−1)(20)
Next, note that for any attributeℓ∈[c], if Algorithm 3, at any point during its execution, has included
k′
ℓvectors of attributeℓin ALG, then at that point the maintained utilityw ℓ=Pk′
ℓ
j=1σ(q, vℓ
(j)). Hence, at
the beginning of any iteration of the algorithm, if thek′
ℓdenotes the number of selected vectors of each
attributeℓ∈[c], then the marginals considered in Line 8 areF ℓ(k′
ℓ+ 1)−F ℓ(k′
ℓ). These observations
and the selection criterion in Line 8 of the algorithm give us the following inequality for the counts
kx=|ALG∩D x|andk y=|ALG∩D y|of the returned solution ALG:
Fx(kx)−F x(kx−1)≤F y(ky+ 1)−F y(ky)(21)
Specifically, equation (21) follows by considering the iteration in whichkth
x(last) vector of attributexwas
selected by the algorithm. Before that iteration the algorithm had selected(k x−1)vectors of attributex,
and letk′
ydenote the number of vectors with attributeythat have been selected till that point. Note that
k′
y≤ky. The fact that thekth
xvector was (greedily) selected in Line 8, instead of including an additional
vector of attributey, givesF x(kx)−F x(kx−1)≤F y(k′
y+ 1)−F y(k′
y)≤F y(ky+ 1)−F y(ky); here, the
last inequality follows from equation (20). Hence, equation (21) holds.
Moreover,
Fx(k∗
x+ 1)−F x(k∗
x)≤F x(kx)−F x(kx−1)(via eqns. (19) and (20))
≤F y(ky+ 1)−F y(ky)(via eqn. (21))
≤F y(k∗
y)−F y(k∗
y−1)(22)
The last inequality follows from equations (19) and (20).
Recall thatvℓ
(i)denotes theithmost similar (toq) vector in the set bDℓ. The definition of bDℓensures
thatvℓ
(i)is, in fact, theithmost similar (toq) vector among the ones that have attributeℓ, i.e.,ithmost
similar in all ofD ℓ. Since OPTis an optimal solution, thek∗
ℓ=|OPT∩D ℓ|vectors of attributeℓin OPT
are the most similark∗
ℓvectors fromD ℓ. That is, OPT∩D ℓ=n
vℓ
(1), . . . , vℓ
(k∗
ℓ)o
, for eachℓ∈[c]. This
observation and the definition ofF ℓ(·)imply that thep-th power of OPT’sp-mean welfare satisfies
Mp(OPT)p=1
ccX
ℓ=1Fℓ(k∗
ℓ).(23)
Now, consider a subset of vectorsSobtained from OPTby including vectorvx
(k∗x+1)and removingvy
(k∗y),
i.e.,S=
OPT∪n
vx
(k∗x+1)o
\n
vy
(k∗y)o
. Note that
Mp(S)p−M p(OPT)p=1
c
Fx(k∗
x+ 1)−F x(k∗
x)
+1
c
Fy(k∗
y−1)−F y(k∗
y)
≤0(via eqn. (22))
Hence,M p(S)≥M p(OPT). Given that OPTis ap-mean welfare optimal solution, the last inequality must
hold with equality,M p(S) =M p(OPT), i.e.,Sis an optimal solution as well. This, however, contradicts
the choice of OPTas an optimal solution that minimizesPc
ℓ=1|k∗
ℓ−kℓ|– note thatPc
ℓ=1bkℓ−kℓ<
Pc
ℓ=1|k∗
ℓ−kℓ|, where bkℓ:=|S∩D ℓ|.
Therefore, by way of contradiction, we obtain that|OPT∩D ℓ|=|ALG∩D ℓ|for eachℓ∈[c]. As
mentioned previously, this guarantee, along with Lemma 10, implies that ALGis a p-mean welfare optimal
solution. This completes the proof of the theorem for the casep <0.
Combining the two cases, we have that the theorem holds for allp∈(−∞,1]\ {0}.
32

Corollary 12.In the single-attribute setting, given any queryq∈Rdand anα-approximate oracleANNfork
most similar vectors from any set, Algorithm 3 (p-mean-ANN) returns anα-approximate solution forp-NNS,
i.e., it returns a size-ksubsetALG⊆PwithM p(ALG)≥αmax S⊆P:|S|=k Mp(S). The algorithm runs in time
O(kc) +cP
ℓ=1ANN(D ℓ, q), withANN(D ℓ, q)being the time required by the approximate oracle to findksimilar
vectors toqinD ℓ.
Proof.The running time of the algorithm follows via an argument similar to one used in the proof of The-
orem 11. Hence, we only argue correctness here.
For everyℓ∈[c], let theα-approximate oracle return bDℓ. Recall thatvℓ
(i),i∈[k], denotes theithmost
similar point toqin the set bDℓ. Further, for everyℓ∈[c], letD∗
ℓbe the set ofkmost similar points toq
withinD ℓand, for eachi∈[k], definev∗ℓ
(i)to be theithmost similar point toqinD∗
ℓ. Recall that by the
guarantee of theα-approximate NNS oracle, we haveσ(q, vℓ
(i))≥α σ(q, v∗ℓ
(i))for eachi∈[k]. Let OPTbe
an optimal solution to thep-NNS problem. Note that, for each attributeℓin[c], the optimal solution OPT
contains in it thek∗
ℓmost similar vectors with attributeℓ.
Finally, let dOPTbe the optimal solution to thep-NNS problem when the set of vectors to search over
isP=∪ ℓ∈[c]bDℓ.
By arguments similar to the ones used in the proof of Theorem 11, we haveM p(ALG) =M p(dOPT).
Therefore
Mp(ALG) =M p(dOPT)
≥
1
cX
ℓ∈[c]
k∗
ℓX
i=1σ(q, vℓ
(i)) +η
p
1
p
(∪ℓ∈[c]:k∗
ℓ≥1{vℓ
(1), . . . , vℓ
(k∗
ℓ)}is a feasible solution)
≥
1
cX
ℓ∈[c]
k∗
ℓX
i=1ασ(q, v∗ℓ
(i)) +η
p
1
p
(byα-approximate guarantee of the oracle, andM pis increasing in its argument;k∗
ℓ≤k)
≥
1
cX
ℓ∈[c]αp
k∗
ℓX
i=1σ(q, v∗ℓ
(i)) +η
p
1
p
(α∈(0,1))
=α M p(OPT)(definition of OPT)
The corollary stands proved.
C Experimental Evaluation and Analysis
In this section, we present additional experimental results to further validate the performance ofNash-ANN
and compare it with existing methods. The details of the evaluation metrics and the description of the
datasets used in our study are already presented in Section 5. Here, we report results for the single-
attribute setting (Section C.1), where we compare the approximation ratio alongside all diversity metrics
fork= 10andk= 50. We also include recall values for bothk= 10andk= 50(Section C.1.5). The key
observation in all these plots is that the NSW objective effectively strikes a balance between relevance and
diversity without having to specify any ad hoc constraints. Furthermore, we report experimental results
for the multi-attribute setting on both a synthetic dataset (Sift1m) and a real-world dataset (ArXiv).
Finally, we experimentally validate the performance-efficiency trade-offs of a faster heuristic variant of
p-mean-ANN(detailed in Appendix 5.5) that can be used in addition to any existing (standard) ANN
algorithm.
33

Figure 6: The plots show approximation ratio versus entropy trade-offs for various algorithms fork=10
(Left)andk=50(Right)in single-attribute setting onSift1m- (Clus)dataset.
C.1 Balancing Relevance and Diversity: Single-attribute Setting
In this experiment, we evaluate how wellp-mean-ANN(and the special case ofp=0,Nash-ANN) balances
relevance and diversity in the single-attribute setting. We begin by examining the tradeoff between
approximation ratio and entropy achieved by our algorithms on additional datasets beyond those used in
the main paper. Moreover, we also report results for other diversity metrics such as the inverse Simpson
index (Section C.1.3) and the number of distinct attributes appearing in thekneighbors (Section C.1.4)
retrieved by our algorithms. These experiments corroborate the findings in the main paper, namely,
Nash-ANNandp-mean-ANNare able to strike a balance between relevance and diversity whereasANN
only optimizes for relevance (hence low diversity) andDiv-ANNonly optimizes for diversity (hence low
relevance).
C.1.1 Performance ofNash-ANN
We report the results for different datasets in Figures 6, 7, 8, and 9. On theSift1m- (Clus)dataset
(Figure 6),Nash-ANNachieves entropy close to that of the most diverse solution (Div-ANNwithk′= 1)
in bothk= 10andk= 50cases. Moreover,Nash-ANNachieves significantly higher approximation ratio
thanDiv-ANNin bothk= 10andk= 50cases whenk′= 1. Fork= 10case,Nash-ANNPareto dominates
Div-ANNeven with the relaxed constraint ofk′= 5fork= 10. When the number of required neighbors
is increased tok= 50, no other method Pareto dominatesNash-ANN. Similar observations hold for the
Sift1m- (Prob)(Figure 7) andDeep1b- (Prob)(Figure 8) datasets. In the results on theArXivdataset
(Figure 9) withk= 10, we observe thatDiv-ANNalready achieves a high approximation ratio. However,
Nash-ANNmatches the entropy ofDiv-ANNwithk′= 1while improving on the approximation ratio. For
k= 50,Nash-ANNnearly matches the entropy ofDiv-ANNwithk′= 1,2whereas it significantly improves
on the approximation ratio. In summary, the experimental results clearly demonstrate the ability of
Nash-ANNto adapt to the varying nature of queries and consistently strike a balance between relevance
and diversity.
C.1.2 Performance ofp-mean-ANN
In this set of experiments, we study the effect on trade-off between approximation ratio and entropy when
the parameterpin thep-NNS objective is varied over a range. Recall that thep-NNS problem withp→0
corresponds to the NaNNS problem, and withp= 1, corresponds to the NNS problem. We experiment
34

Figure 7: The plots show approximation ratio versus entropy trade-offs for various algorithms fork=10
(Left)andk=50(Right)in single-attribute setting onSift1m- (Prob)dataset.
Figure 8: The plots show approximation ratio versus entropy trade-offs for various algorithms fork=10
(Left)andk=50(Right)in single-attribute setting onDeep1b- (Prob)dataset.
with values ofp∈ {−10,−1,−0.5,0,0.5,1}by running our algorithmp-mean-ANN(Algorithm 3) on the
various datasets. The results are shown in Figures 10, 11, 12, and 13. We observe across all datasets for
bothk= 10andk= 50that, aspdecreases from1, the entropy increases but the approximation ratio
decreases. This highlights the key intuition that aspdecreases, the behavior changes from utilitarian
welfare (p= 1aligns exactly withANN) to egalitarian welfare (more attribute-diverse). In other words,
the parameterpallows us to smoothly interpolate between complete relevance (the standard NNS with
p= 1) and complete diversity (p→ −∞).
C.1.3 Approximation Ratio Versus Inverse Simpson Index
We also report results (Figures 14, 15, 16 and 17) on approximation ratio versus inverse Simpson index
for all the aforementioned datasets, comparingNash-ANNwithDiv-ANNwith various choices of constraint
parameterk′. The trends are similar to those for approximation ratio versus entropy.
35

Figure 9: The plots show approximation ratio versus entropy trade-offs for various algorithms fork=10
(Left)andk=50(Right)in single-attribute setting onArXivdataset.
Figure 10: The plots report the approximation ratio versus entropy trade-off ofp-mean-ANN, aspvaries,
fork=10(Left)andk=50(Right)onSift1m- (Clus)dataset in the single-attribute setting.
C.1.4 Approximation Ratio Versus Distinct Attribute Count
We also report the number of distinct attributes appearing in the set of vectors returned by different
algorithms. Note thatDiv-ANNby design always returns a set where the number of distinct attributes
is at least(k/k′). We plot approximation ratio versus number of distinct attributes and the results are
shown in Figures 18, 19, 20, and 21. The results show that whileDiv-ANNwithk′= 1has high number
of distinct attributes (by design), its approximation ratio is quite low. On the other hand,Nash-ANNhas
almost equal or slightly lower number of distinct attributes but achieves very high approximation ratio.
C.1.5 Recall Versus Entropy
We also report results for another popular relevance metric in the nearest neighbor search literature,
namely, recall. The results for different datasets are shown in Figures 22, 23, 24, 25, 26, and 27.
Note that as discussed earlier (Section 5.1, Remark 1), recall can be a fragile metric when the goal
is to balance between diversity and relevance. However, we still report recall to be consistent with prior
36

Figure 11: The plots report the approximation ratio versus entropy trade-off ofp-mean-ANN, aspvaries,
fork=10(Left)andk=50(Right)onSift1m- (Prob)dataset in single-attribute setting. We omit points
corresponding top∈ {−1,−0.5,0.5}since they were extremely close to the pointsp=−10orp= 0.
Figure 12: The plots report the approximation ratio versus entropy trade-off ofp-mean-ANN, aspvaries,
fork=10(Left)andk=50(Right)onDeep1b- (Prob)dataset in single-attribute setting. Fork= 50, we
omit points corresponding top∈ {−1,−0.5,0,0.5}since they were extremely close to the pointp=−10.
Due to the same reasons, we omitp∈ {−1,−0.5}fork=10.
literature and to demonstrate thatNash-ANNdoes not perform poorly. In fact, it is evident from the plots
thatNash-ANN’s recall value (relevance) surpasses that ofDiv-ANNwithk′= 1(most attribute diverse
solution) while achieving a similar entropy. As already noted, the approximation ratio forNash-ANN
remains sufficiently high, indicating that the retrieved set of neighbors lies within a reasonably good
neighborhood of the true nearest neighbors of a given query.
C.2 Balancing Relevance and Diversity: Multi-attribute Setting
Recall that our welfarist formulation seamlessly extends to the multi-attribute setting. In Section 5, we
discussed the performance ofMulti Nash-ANNandMulti Div-ANNonSift1m- (Clus), where each input
vector was associated with four attributes. In this section, we repeat the same set of experiments on
one of the real-world datasets, namelyArXiv, which naturally contains two attribute classes (m= 2;
37

Figure 13: The plots report the approximation ratio versus entropy trade-off ofp-mean-ANN, aspvaries,
fork=10(Left)andk=50(Right)onArXivdataset in single-attribute setting. Fork= 50, we omit
points corresponding top∈ {−1,−0.5,0,0.5}since they were extremely close to the pointsp=−10. Due
to similar reasons we omitp∈ {−1,−0.5,0.5}fork=10.
Figure 14: The plots show approximation ratio versus inverse Simpson index trade-offs for various algo-
rithms fork=10(Left)andk=50(Right)in single-attribute setting onSift1m- (Clus)dataset.
see Section 5.1, Diversity Metrics): update year (|C 1|= 14) and paper category (|C 2|= 16). Therefore,
c=|C 1|+|C 2|= 30. The results fork= 50are presented in Figure 28. Note that, in each plot, we
restrict the entropy to one of the attribute classes (C 1orC 2) so that the diversity within a class can be
understood from these plots. The results indicate thatMulti Nash-ANNachieves an approximation ratio
very close to one while maintaining entropy levels comparable toMulti Div-ANNwithk′= 1or2for
both the attribute classes. In fact,Multi Nash-ANNPareto dominatesMulti Div-ANNwithk′= 5.
We also study the effect of varyingpinp-NNS problem in the multi-attribute setting. The results for
performance ofMulti p-mean-ANN(an analogue ofMulti Nash-ANN, detailed in Section 5.3) forp∈
{−10,−1,−0.5,0,0.5,1}are shown in Figures 29 and 30. Interestingly, we observe that with decreasing
p, the entropy (acrossC 1orC 2) increases but the approximation ratio remains nearly the same and very
close to1. On the other hand,Multi Div-ANNwithk′= 1has very low approximation ratio. In fact,
Multi p-mean-ANNwithp=−1and−10Pareto dominatesMulti Div-ANNwithk′= 1.
38

Figure 15: The plots show approximation ratio versus inverse Simpson index trade-offs for various algo-
rithms fork=10(Left)andk=50(Right)in single-attribute setting onSift1m- (Prob)dataset.
Figure 16: The plots show approximation ratio versus inverse Simpson index trade-offs for various algo-
rithms fork=10(Left)andk=50(Right)in single-attribute setting onDeep1b- (Prob)dataset.
C.3 More Experiments forp-FetchUnion-ANN
In this section, we empirically study a faster heuristic algorithm for NSW andp-mean welfare formula-
tions. Specifically, the heuristic—calledp-FetchUnion-ANN—first fetches a sufficiently large candidate
set of vectors (irrespective of their attributes) using theANNalgorithm. Then, it applies the Nash (or
p-mean) selection (similar to Line 5 in Algorithm 1 or Lines 6-8 in Algorithm 3) within this set. That is,
instead of starting out withkneighbors for eachℓ∈[c](as in Line 1 of Algorithm 1), the alternative here
is to work with sufficiently many neighbors from the set∪c
ℓ=1Dℓ.
We empirically show (in Tables 4 to 8) that this heuristic consistently achieves performance compara-
ble top-Mean-ANNacross nearly all datasets and evaluation metrics. Sincep-FetchUnion-ANNstarts with
a large pool of vectors (with high similarity to the query) retrieved by theANNalgorithm without diversity
considerations, it achieves improved approximation ratio overp-Mean-ANN. This trend is clearly evident
in two datasets, namelyDeep1b- (Clus)andSift1m- (Clus). However, the improvement in approxima-
tion ratio comes at the cost of reduced entropy, which can be explained by the fact that in restricting its
search to an initially fetched large pool of vectors,p-FetchUnion-ANNmay miss out on a more diverse so-
39

Figure 17: The plots show approximation ratio versus inverse Simpson index trade-offs for various algo-
rithms fork=10(Left)andk=50(Right)in single-attribute setting onArXivdataset.
Figure 18: The plots show approximation ratio versus distinct counts trade-offs for various algorithms for
k=10(Left)andk=50(Right)in single-attribute setting onSift1m- (Clus)dataset.
lution that exists over the entire dataset. Another important aspect ofp-FetchUnion-ANNis that, because
it retrieves all neighbors from the union at once, the heuristic delivers substantially higher throughput
(measured as queries answered per second, QPS) and therefore lower latency. The results validating
these findings are reported in Tables 3 and 9 for theSift1m- (Clus)andAmazondatasets, respectively. In
particular, it serves almost10×more queries onSift1m- (Clus)and3×more queries onAmazondataset.
The latency values exhibit a similar trend with reductions of similar magnitude. In summary, these ob-
servations position the heuristic as a notably fast method for NaNNS andp-NNS, particularly whencis
large.
40

Figure 19: The plots show approximation ratio versus distinct counts trade-offs for various algorithms for
k=10(Left)andk=50(Right)in single-attribute setting onSift1m- (Prob)dataset.
Figure 20: The plots show approximation ratio versus distinct counts trade-offs for various algorithms for
k=10(Left)andk=50(Right)in single-attribute setting onDeep1b- (Prob)dataset.
Table 4: Comparison of relevance and diversity ofp-mean-ANN,p-FetchUnion-ANNacross different values
ofpagainstANNandDiv-ANN(k′= 1) forAmazonatk= 50.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.865±0.045 0.909±0.029 0.922±0.027 0.938±0.023 0.961±0.018 1.000±0.000
p-FetchUnion-ANN0.907±0.033 0.912±0.030 0.921±0.027 0.935±0.024 0.958±0.019 1.000±0.000
ANN1.000±0.000
Div-ANN(k′=1) 0.813±0.053
Entropyp-Mean-ANN5.644±0.000 5.382±0.135 5.252±0.153 5.058±0.178 4.687±0.227 2.782±0.684
p-FetchUnion-ANN5.364±0.156 5.333±0.149 5.261±0.150 5.099±0.171 4.736±0.221 2.782±0.684
ANN2.782±0.684
Div-ANN(k′=1) 5.594±0.049
41

Figure 21: The plots show approximation ratio versus distinct counts trade-offs for various algorithms for
k=10(Left)andk=50(Right)in single-attribute setting onArXivdataset.
Figure 22: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onAmazondataset.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.985±0.010 0.985±0.010 0.985±0.010 0.986±0.009 0.989±0.008 1.000±0.001
p-FetchUnion-ANN0.989±0.007 0.989±0.007 0.989±0.007 0.990±0.006 0.991±0.006 1.000±0.001
ANN1.000±0.001
Div-ANN(k′=1) 0.293±0.007
Entropyp-Mean-ANN3.793±0.002 3.793±0.002 3.793±0.002 3.793±0.002 3.793±0.002 2.790±0.510
p-FetchUnion-ANN3.704±0.167 3.704±0.166 3.704±0.166 3.704±0.166 3.704±0.166 2.790±0.510
ANN2.790±0.510
Div-ANN(k′=1) 3.799±0.029
Table 5: Comparison of relevance and diversity ofp-mean-ANN,p-FetchUnion-ANNacross different values
ofpagainstANNandDiv-ANN(k′= 1) forArXivatk= 50.
42

Figure 23: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onDeep1b- (Clus)dataset.
Figure 24: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onSift1m- (Clus)dataset.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.784±0.071 0.815±0.065 0.831±0.063 0.858±0.060 0.904±0.049 1.000±0.000
p-FetchUnion-ANN0.958±0.033 0.961±0.030 0.962±0.029 0.963±0.028 0.968±0.024 1.000±0.000
ANN1.000±0.000
Div-ANN(k′=1) 0.286±0.041
Entropyp-Mean-ANN4.293±0.000 4.200±0.052 4.105±0.091 3.887±0.155 3.349±0.267 0.746±0.717
p-FetchUnion-ANN2.101±1.214 2.101±1.214 2.099±1.212 2.095±1.207 2.068±1.179 0.746±0.717
ANN0.746±0.717
Div-ANN(k′=1) 4.191±0.234
Table 6: Comparison of relevance and diversity ofp-mean-ANN,p-FetchUnion-ANNacross different values
ofpagainstANNandDiv-ANN(k′= 1) forDeep1b- (Clus)atk= 50.
43

Figure 25: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onSift1m- (Prob)dataset.
Figure 26: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onDeep1b- (Prob)dataset.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.958±0.019 0.960±0.017 0.961±0.016 0.963±0.014 0.969±0.010 1.000±0.000
p-FetchUnion-ANN0.958±0.019 0.960±0.017 0.961±0.016 0.963±0.014 0.969±0.010 1.000±0.000
ANN1.000±0.000
Div-ANN(k′=1) 0.395±0.010
Entropyp-Mean-ANN4.293±0.000 4.292±0.005 4.288±0.010 4.275±0.020 4.217±0.068 2.070±0.208
p-FetchUnion-ANN4.293±0.001 4.292±0.005 4.288±0.010 4.275±0.020 4.217±0.068 2.070±0.207
ANN2.070±0.207
Div-ANN(k′=1) 4.322±0.002
Table 7: Comparison of relevance and diversity ofp-mean-ANN,p-FetchUnion-ANNacross different values
ofpagainstANNandDiv-ANN(k′= 1) forDeep1b- (Prob)atk= 50.
44

Figure 27: The plots show recall versus entropy trade-offs fork=10(Left)andk=50(Right)in the
single-attribute setting onArXivdataset.
Figure 28: The plots show approximation ratio versus entropy trade-offs for various algorithms onArXiv
dataset across attribute classesC 1(Left)andC 2(Right)in the multi-attribute setting fork= 50.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Approx. Ratiop-Mean-ANN0.975±0.010 0.977±0.008 0.979±0.008 0.980±0.008 0.982±0.006 1.000±0.000
p-FetchUnion-ANN0.975±0.010 0.977±0.008 0.979±0.008 0.980±0.008 0.982±0.006 1.000±0.000
ANN1.000±0.000
Div-ANN(k′=1) 0.404±0.004
Entropyp-Mean-ANN4.292±0.006 4.292±0.003 4.293±0.002 4.293±0.002 4.269±0.020 2.068±0.205
p-FetchUnion-ANN4.292±0.006 4.292±0.003 4.293±0.002 4.293±0.003 4.269±0.020 2.068±0.205
ANN2.068±0.205
Div-ANN(k′=1) 4.322±0.005
Table 8: Comparison of relevance and diversity ofp-mean-ANN,p-FetchUnion-ANNacross different values
ofpagainstANNandDiv-ANN(k′= 1) forSift1m- (Prob)atk= 50.
45

Figure 29: The plots show approximation ratio versus entropy trade-offs forp-mean-ANN, aspvaries, for
k=50across attribute classesC 1(Left)andC 2(Right)in the multi-attribute setting onArXivdataset.
Metric Algorithmp=−10p=−1p=−0.5p= 0p= 0.5p= 1
Query per Secondp-Mean-ANN198.08 195.97 199.08 179.03 171.22 189.31
p-FetchUnion-ANN620.27 610.62 551.02 608.76 572.57 591.76
Latency (µs)p-Mean-ANN161385.00 163121.00 160503.00 178555.00 186780.00 168856.00
p-FetchUnion-ANN51539.90 52362.30 58028.60 52521.60 55843.70 54030.80
99.9th percentile of Latencyp-Mean-ANN433434.00 407151.00 418147.00 421725.00 475474.00 404477.00
p-FetchUnion-ANN146632.00 144989.00 145620.00 145657.00 143627.00 146464.00
Table 9: Comparison of Queries per second and Latency ofp-mean-ANN,p-FetchUnion-ANNacross differ-
ent values ofpagainstANNandDiv-ANN(k′= 1) forAmazondataset fork= 50.
46

Figure 30:Top Row: The plots show approximation ratio versus entropy trade-offs forNash-ANNagainst
Div-ANNwith varying values ofk′for attribute classC 3(left)andC 4(right).Bottom Row: The plots
show approximation ratio versus entropy trade-offs forp-mean-ANN, aspvaries, across attribute classes
C3(left)andC 4(right). Both the rows correspond tok= 50onSift1m- (Clus)dataset in the multi-
attribute setting.
47