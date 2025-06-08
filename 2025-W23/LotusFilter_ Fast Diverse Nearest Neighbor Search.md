# LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table

**Authors**: Yusuke Matsui

**Published**: 2025-06-05 09:17:30

**PDF URL**: [http://arxiv.org/pdf/2506.04790v1](http://arxiv.org/pdf/2506.04790v1)

## Abstract
Approximate nearest neighbor search (ANNS) is an essential building block for
applications like RAG but can sometimes yield results that are overly similar
to each other. In certain scenarios, search results should be similar to the
query and yet diverse. We propose LotusFilter, a post-processing module to
diversify ANNS results. We precompute a cutoff table summarizing vectors that
are close to each other. During the filtering, LotusFilter greedily looks up
the table to delete redundant vectors from the candidates. We demonstrated that
the LotusFilter operates fast (0.02 [ms/query]) in settings resembling
real-world RAG applications, utilizing features such as OpenAI embeddings. Our
code is publicly available at https://github.com/matsui528/lotf.

## Full Text


<!-- PDF content starts -->

arXiv:2506.04790v1  [cs.CV]  5 Jun 2025LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table
Yusuke Matsui
The University of Tokyo
matsui@hal.t.u-tokyo.ac.jp
Abstract
Approximate nearest neighbor search (ANNS) is an essen-
tial building block for applications like RAG but can some-
times yield results that are overly similar to each other. In
certain scenarios, search results should be similar to the
query and yet diverse. We propose LotusFilter, a post-
processing module to diversify ANNS results. We precom-
pute a cutoff table summarizing vectors that are close to
each other. During the filtering, LotusFilter greedily looks
up the table to delete redundant vectors from the candidates.
We demonstrated that the LotusFilter operates fast (0.02
[ms/query]) in settings resembling real-world RAG appli-
cations, utilizing features such as OpenAI embeddings. Our
code is publicly available at https://github.com/
matsui528/lotf .
1. Introduction
An approximate nearest neighbor search (ANNS) algo-
rithm, which finds the closest vector to a query from
database vectors [8, 29, 31], is a crucial building block
for various applications, including image retrieval and in-
formation recommendation. Recently, ANNS has become
an essential component of Retrieval Augmented Genera-
tion (RAG) approaches, which integrate external informa-
tion into Large Language Models [5].
The essential problem with ANNS is the lack of diver-
sity. For example, consider the case of image retrieval us-
ing ANNS. Suppose the query is an image of a cat, and the
database contains numerous images of the same cat. In that
case, the search results might end up being almost uniform,
closely resembling the query. However, users might prefer
more diverse results that differ from one another.
Diverse nearest neighbor search (DNNS) [15, 41, 50]
is a classical approach to achieving diverse search results
but often suffers from slow performance. Existing DNNS
methods first obtain Scandidates (search step) and then se-
lectK(< S)results to ensure diversity (filter step). This
approach is slow for three reasons. First, integrating mod-
ern ANN methods is often challenging. Second, selecting
𝒒𝒙13
𝒙28𝒙25(a) Usual ANNS
𝒒𝜀
𝒙25𝒙61
𝒙19𝒙𝑖−𝒙𝑗22≥𝜀 (b) DNNS with the LotusFilter
Figure 1. (a) Usual ANNS. The search results are close to the
query qbut similar to each other. (b) DNNS with the proposed
LotusFilter. The obtained vectors are at least√εapart from each
other. The results are diverse despite being close to the query.
Kitems from Scandidates is a subset selection problem,
which is NP-hard. Lastly, existing methods require access
to the original vectors during filtering, which often involves
slow disk access if the vectors are not stored in memory.
We propose a fast search result diversification approach
called LotusFilter , which involves precomputing a cutoff ta-
ble and using it to filter search results. Diverse outputs are
ensured by removing vectors too close to each other. The
data structure and algorithm are both simple and highly ef-
ficient (Fig. 1), with the following contributions:
• As LotusFilter is designed to operate as a pure post-
processing module, one can employ the latest ANNS
method as a black-box backbone. This design provides
a significant advantage over existing DNNS methods.
• We introduce a strategy to train the hyperparameter, elim-
inating the need for complex parameter tuning.
• LotusFilter demonstrates exceptional efficiency for large-
scale datasets, processing queries in only 0.02 [ms/query]
for9×1051536 -dimensional vectors.
2. Related work
2.1. Approximate nearest neighbor search
Approximate nearest neighbor search (ANNS) has been
extensively studied across various fields [29, 31]. Since

around 2010, inverted indices [4, 7, 13, 24, 30] and graph-
based indices [20, 28, 34, 36, 46, 48] have become the
standard, achieving search times under a millisecond for
datasets of approximately 106items. These modern ANNS
methods are significantly faster than earlier approaches, im-
proving search efficiency by orders of magnitude.
2.2. Diverse nearest neighbor search
The field of recommendation systems has explored di-
verse nearest neighbor search (DNNS), especially during
the 2000s [9, 15, 41, 50]. Several approaches propose ded-
icated data structures as solutions [16, 39], indicating that
modern ANNS methods have not been fully incorporated
into DNNS. Hirata et al. stand out as the only ones to use
modern ANNS for diverse inner product search [22].
Most existing DNNS methods load Sinitial search re-
sults (the original D-dimensional vectors) and calculate all
possible combinations even if approximate. This approach
incurs a diversification cost of at least O(DS2). In contrast,
our LotusFilter avoids loading the original vectors or per-
forming pairwise computations, instead scanning Sitems
directly. This design reduces the complexity to O(S), mak-
ing it significantly faster than traditional approaches.
2.3. Learned data structure
Learned data structures [17, 26] focus on enhancing clas-
sical data structures by integrating machine learning tech-
niques. This approach has been successfully applied to
well-known data structures such as B-trees [10, 18, 19, 49],
KD-trees [12, 21, 33], and Bloom Filters [27, 32, 42, 47].
Our proposed method aligns with this trend by constructing
a data structure that incorporates data distribution through
learned hyperparameters for thresholding, similar to [10].
3. Preliminaries
Let us describe our problem setting. Considering that we
haveN D -dimensional database vectors {xn}N
n=1, where
xn∈RD. Given a query vector q∈RD, our task is to
retrieve Kvectors that are similar to qyet diverse, i.e., dis-
similar to each other. We represent the obtained results as a
set of identifiers, K ⊆ { 1, . . . , N }, where |K|=K.
The search consists of two steps. First, we run ANNS
and obtain S(≥K)vectors close to q. These initial search
results are denoted as S ⊆ { 1, . . . , N }, where |S|=S. The
second step is diversifying the search results by selecting a
subset K(⊆ S)from the candidate set S. This procedure
is formulated as a subset selection problem. The objective
here is to minimize the evaluation function f: 2S→R.
argmin
K⊆S,|K|=Kf(K). (1)
Here, fevaluates how good Kis, regarding both “proximityto the query” and “diversity”, formulated as follows.
f(K) =1−λ
KX
k∈K∥q−xk∥2
2−λmin
i,j∈K, i̸=j∥xi−xj∥2
2.
(2)
The first term is the objective function of the nearest neigh-
bor search itself, which indicates how close qis to the se-
lected vectors. The second term is a measure of the diver-
sity. Following [3, 22], we define it as the closest distance
among the selected vectors. Here λ∈[0,1]is a parameter
that adjusts the two terms. If λ= 0, the problem is a near-
est neighbor search. If λ= 1, the equation becomes the
MAX-MIN diversification problem [40] that evaluates the
diversity of the set without considering a query. This for-
mulation is similar to the one used in [9, 22, 39] and others.
Let us show the computational complexity of Eq. (1) is
O(T+ S
K
DK2), indicating that it is slow. First, since it’s
not easy to represent the cost of ANNS, we denote ANNS’s
cost as O(T), where Tis a conceptual variable govern-
ing the behavior of ANNS. The first term in Eq. (2) takes
O(DK), and the second term takes O(DK2)for a naive
pairwise comparison. When calculating Eq. (1) naively, it
requires S
K
computations for subset enumeration. There-
fore, the total cost is O(T+ S
K
DK2).
There are three main reasons why this operation is slow.
First, it depends on D, making it slow for high-dimensional
vectors since it requires maintaining and scanning original
vectors. Second, the second term calculates all pairs of ele-
ments in K(costing O(K2)), which becomes slow for large
K. Lastly, subset enumeration, S
K
, is unacceptably slow.
In the next section, we propose an approximate and efficient
solution with a complexity of O(T+S+KL), where Lis
typically less than 100 for N= 9×105.
4. LotusFilter Algorithm
In this section, we introduce the algorithm of the proposed
LotusFilter. The basic idea is to pre-tabulate the neighbor-
ing points for each xnand then greedily prune candidates
by looking up this table during the filtering step.
Although LotusFilter is extremely simple, it is unclear
whether the filtering works efficiently. Therefore, we in-
troduce a data structure called OrderedSet to achieve fast
filtering with a theoretical guarantee.
4.1. Preprocessing
Algorithm 1 illustrates a preprocessing step. The inputs
consist of database vectors {xn}N
n=1and the threshold for
the squared distance, ε∈R. InL1, we first construct I, the
index for ANNS. Any ANNS methods, such as HNSW [28]
for faiss [14], can be used here.
Next, we construct a cutoff table in L2-3 . For each xn,
we collect the set of IDs whose squared distance from xnis

Algorithm 1: BUILD
Input: {xn}N
n=1⊆RD, ε∈R
1I ← BUILD INDEX ({xn}N
n=1) # ANNS
2forn∈ {1, . . . , N }do
3Ln← {i∈ {1, . . . , N } | ∥xn−xi∥2
2< ε, n ̸=i}
4return I,{Ln}N
n=1
Algorithm 2: SEARCH AND FILTER
Input: q∈RD, S, K (≤S),I,{Ln}N
n=1
1S ← I .SEARCH (q, S) #S ⊆ { 1, . . . , N }
2K ←∅
3while|K|< K do # At most Ktimes
4 k←POP(S) #O(L)
5K ← K ∪ { k} #O(1)
6S ← S \ L k #O(L)
7return K #K ⊆ S where |K|=K
less than ε. The collected IDs are stored as Ln. We refer to
these{Ln}N
n=1as a cutoff table (an array of integer arrays).
We perform a range search for each xnto create the cut-
off table. Assuming that the cost of the range search is also
O(T), the total cost becomes O(NT). As demonstrated
later in Tab. 2, the runtime for N= 9×105is approxi-
mately one minute at most.
4.2. Search and Filtering
The search and filtering process is our core contribution and
described in Algorithm 2 and Fig. 2. The inputs are a query
q∈RD, the number of initial search results S(≤N), the
number of final results K(≤S), the ANNS index I, and
the cutoff table {Ln}N
n=1.
As the search step, we first run ANNS in L1(Fig. 2a) to
obtain the candidate set S ⊆ { 1, . . . , N }. InL2, we prepare
an empty integer set Kto store the final results.
The filtering step is described in L3-6 where IDs are
added to the set Kuntil its size reaches K. InL4, we pop
the ID kfromS, where xkis closest to the query, and add
it toKinL5. Here, L6is crucial: for the current focus
k, the IDs of vectors close to xkare stored in Lk. Thus,
by removing LkfromS, we can eliminate vectors similar
toxk(Fig. 2b). Repeating this step (Fig. 2c) ensures that
elements in Kare at least√εapart from each other.1
Here, the accuracy of the top-1 result (Recall@1) after
filtering remains equal to that of the initial search results.
This is because the top-1 result from the initial search is
always included in KinL4during the first iteration.
1The filtering step involves removing elements within a circle centered
on a vector (i.e., eliminating points inside the green circle in Figs. 2b
and 2c). This process evokes the imagery of lotus leaves, which inspired
us to name the proposed method “LotusFilter”.Note that the proposed approach is faster than existing
methods for the following intuitive reasons:
• The filtering step processes candidates sequentially
(O(S)) in a fast, greedy manner. Many existing meth-
ods determine similar items in Sby calculating distances
on the fly, requiring O(DS2)for all pairs, even when ap-
proximated. In contrast, our approach precomputes dis-
tances, eliminating on-the-fly calculations and avoiding
pairwise computations altogether.
• The filtering step does not require the original vectors,
making it a pure post-processing step for any ANNS mod-
ules. In contrast, many existing methods depend on re-
taining the original vectors and computing distances dur-
ing the search. Therefore, they cannot be considered pure
post-processing, especially since modern ANNS methods
often use compressed versions of the original vectors.
In Sec. 5, we discuss the computational complexity in detail
and demonstrate that it is O(T+S+KL).
4.3. Memory consumption
With L=1
NPN
n=1|Ln|being the average length of Ln,
the memory consumption of the LotusFilter is 64LN [bit]
with the naive implementation using 64 bit integers. It is
because, from Algorithm 2, the LotusFilter requires only a
cutoff table {Ln}N
n=1as an auxiliary data structure.
This result demonstrates that the memory consumption
of our proposed LotusFilter can be accurately estimated in
advance. We will later show in Tab. 1 that, for N= 9×105,
the memory consumption is 1.14×109[bit]= 136 [MiB].
4.4. Theoretical guarantees on diversity
For the results obtained by Algorithm 2, the diversity term
(second term) of the objective function Eq. (2) is bounded
by−εas follows. We construct the final results of Algo-
rithm 2, K, by adding an element one by one in L4. For
each loop, given a new kinL4, all items whose squared
distance to kis less than εmust be contained in Lk. Such
close items are removed from the candidates SinL6. Thus,
for all i, j∈ K where i̸=j,∥xi−xj∥2
2≥εholds, resulting
in−mini,j∈K,i̸=j∥xi−xj∥2
2≤ −ε.
This result shows that the proposed LotusFilter can al-
ways ensure diversity, where we can adjust the degree of
diversity using the parameter ε.
4.5. Safeguard against over-pruning
Filtering can sometimes prune too many candidates from
S. To address this issue, a safeguard mode is available as
an option. Specifically, if LkinL6is large and |S|drops to
zero, no further elements can be popped. If this occurs, K
returned by Algorithm 2 may have fewer elements than K.
With the safeguard mode activated, the process will ter-
minate immediately when excessive pruning happens in L6.
The remaining elements in Swill be added to K. This

12
3
6
45
14
810
9 1213
711𝒒𝑘ℒ𝑘
14, 8
2
311, 14
41, 8
510
6
712
81, 4
9
10 5
11 3, 14
12 7
13
14 3, 11𝒮=5,10,3,14,7,11(a) Initial search result
912
3
6
414
81213
711510
𝒒𝑘ℒ𝑘
14, 8
2
311, 14
41, 8
510
6
712
81, 4
9
10 5
11 3, 14
12 7
13
14 3, 11𝒮=5,10,3,14,7,11
𝒦={5}Pop (b) Accept the 1stcandidate. Cutoff.
910
12
6
481213
75𝑘ℒ𝑘
14, 8
2
311, 14
41, 8
510
6
712
81, 4
9
10 5
11 3, 14
12 7
13
14 3, 11 3
1411𝒒𝒮=3,14,7,11
𝒦={5,3}Pop (c) Accept the 2ndcandidate. Cutoff.
Figure 2. Overview of the proposed LotusFilter ( D= 2, N = 14, S= 6, K = 2)
safeguard ensures that the final result meets the condition
|K|=K. In this scenario and only in this scenario, the
theoretical result discussed in Sec. 4.4 does not hold.
5. Complexity Analysis
We prove that the computational complexity of Algorithm 2
isO(T+S+KL)on average. This is fast because just
accessing the used variables requires the same cost.
The filtering step of our LotusFilter ( L3-L6 in Algo-
rithm 2) is quite simple, but it is unclear whether it can
be executed efficiently. Specifically, for S,L4requires a
pop operation, and L6removes an element. These two op-
erations cannot be efficiently implemented with basic data
structures like arrays, sets, or priority queues.
To address this, we introduce a data structure called Or-
deredSet. While OrderedSet has a higher memory con-
sumption, it combines the properties of both a set and an
array. We demonstrate that by using OrderedSet, the opera-
tions in the while loop at L3can be run in O(L).
5.1. Main result
Proposition 5.1. The computational complexity of the
search and filter algorithm in Algorithm 2 is O(T+S+KL)
on average using the OrderedSet data structure for S.
Proof. InL1, the search takes O(T), and the initialization
ofStakesO(S). The loop in L3is executed at most K
times. Here, the cost inside the loop is O(L). That is, P OP
onStakesO(L)inL4. Adding an element to a set takes
O(1)inL5. The Ltimes deletion for SinL6takesO(L).
In total, the computational cost is O(T+S+KL).
To achieve the above, we introduce the data structure
called OrderedSet to represent S. An OrderedSet satisfies
O(S)for initialization, O(L)for P OP, andO(1)for the
deletion of a single item.
5.2. OrderedSet
OrderedSet, as its name suggests, is a data structure repre-
senting a set while maintaining the order of the input ar-ray. OrderedSet combines the best aspects of arrays and
sets at the expense of memory consumption. See the swift-
collections package2in the Swift language for the reference
implementation. We have found that this data structure im-
plements the P OPoperation in O(L).
For a detailed discussion of the implementation, here-
after, we consider the input to OrderedSet as an array v=
[v[1], v[2], . . . , v [V]]withVelements (i.e., the input to Sin
L1of Algorithm 2 is an array of integers).
Initialization: We show that the initialization of Ordered-
Set takes O(V). OrderedSet takes an array vof length V
and converts it into a set (hash table) V:
V ← SET(v). (3)
This construction takes O(V). Then, a counter c∈
{1, . . . , V }indicating the head position is prepared and ini-
tialized to c←1. The OrderedSet is a simple data structure
that holds v,V, and c. OrderedSet has high memory con-
sumption because it retains both the original array vand its
set representation V. An element in Vmust be accessed and
deleted in constant time on average. We utilize a fast open-
addressing hash table boost::unordered flat set
in our implementation3. InL1of Algorithm 2, this initial-
ization takes O(S).
Remove: The operation to remove an element afrom Or-
deredSet is implemented as follows with an average time
complexity of O(1):
V ← V \ { a}. (4)
In other words, the element is deleted only from V. As the
element in vremains, the deletion is considered shallow. In
L6of Algorithm 2, the Lremovals result in an O(L)cost.
2https : / / swiftpackageindex . com / apple /
swift - collections / 1 . 1 . 0 / documentation /
orderedcollections/orderedset
3https://www.boost.org/doc/libs/master/libs/
unordered/doc/html/unordered/intro.html

Pop: Finally, the P OPoperation, which removes the first
element, is realized in O(∆)as follows:
• Step 1: Repeat c←c+ 1untilv[c]∈ V
• Step 2: V ← V \ { v[c]}
• Step 3: Return v[c]
• Step 4: c←c+ 1
Step 1 moves the counter until a valid element is found.
Here, the previous head (or subsequent) elements might
have been removed after the last call to P OP. In such cases,
the counter must move along the array until it finds a valid
element. Let ∆be the number of such moves; this counter
update takes O(∆). In Step 2, the element is removed in
O(1)on average. In Step 3, the removed element is re-
turned, completing the P OPoperation. Step 4 updates the
counter position accordingly.
Thus, the total time complexity is O(∆). Here, ∆repre-
sents the “number of consecutively removed elements from
the previous head position since the last call to P OP”. In our
problem setting, between two calls to P OP, at most Lele-
ments can be removed (refer to L6in Algorithm 2). Thus,
∆≤L. (5)
Therefore, the P OPoperation is O(L)in Algorithm 2.
Using other data structures, achieving both P OPand R E-
MOVE operations efficiently is challenging. With an array,
POPcan be accomplished in O(∆) in the same way. How-
ever, removing a specific element requires a linear search,
which incurs a cost of O(V). On the other hand, if we use
a set (hash table), deletion can be done in O(1), but P OP
cannot be implemented. Please refer to the supplemental
material for a more detailed comparison of data structures.
6. Training
The proposed method intuitively realizes diverse search by
removing similar items from the search results, but it is un-
clear how it contributes explicitly to the objective function
Eq. (1). Here, by learning the threshold εin advance, we
ensure that our LotusFilter effectively reduces Eq. (1).
First, let’s confirm the parameters used in our approach;
λ, S, K, andε. Here, λis set by the user to balance the pri-
ority between search and diversification. Kis the number
of final search results and must also be set by the user. S
governs the accuracy and speed of the initial search. Setting
Sis not straightforward, but it can be determined based on
runtime requirements, such as setting S= 3K. The param-
eterεis less intuitive; a larger εincreases the cutoff table
sizeL, impacting both results and runtime. The user should
setεminimizing f, but this setting is not straightforward.
To find the optimal ε, we rewrite the equations as fol-
lows. First, since Sis the search result of q, we can write
S= NN( q, S). Here, we explicitly express the solution f∗
of Eq. (1) as a function of εandqas follows.f∗(ε,q) = argmin
K⊆NN(q, S),|K|=Kf(K). (6)
We would like to find εthat minimizes the above. Since q
is a query data provided during the search phase, we cannot
know it beforehand. Therefore, we prepare training query
dataQtrain⊂RDin the training phase. This training query
data can usually be easily prepared using a portion of the
database vectors. Assuming that this training query data is
drawn from a distribution similar to the test query data, we
solve the following.
ε∗= argmin
εE
q∈Qtrain[f∗(ε,q)]. (7)
This problem is a nonlinear optimization for a single
variable without available gradients. One could apply a
black-box optimization [1] to solve this problem, but we use
a more straightforward approach, bracketing [25], which re-
cursively narrows the range of the variable. See the supple-
mentary material for details. This simple method achieves
sufficient accuracy as shown later in Fig. 4.
7. Evaluation
In this section, we evaluate the proposed LotusFil-
ter. All experiments were conducted on an AWS EC2
c7i.8xlarge instance (3.2GHz Intel Xeon CPU, 32 vir-
tual cores, 64GiB memory). We ran preprocessing us-
ing multiple threads while the search was executed using
a single thread. For ANNS, we used HNSW [28] from
the faiss library [14]. The parameters of HNSW were
efConstruction=40 ,efSearch=16 , and M=256 .
LotusFilter is implemented in C++17 and called from
Python using nanobind [23]. Our code is publicly available
athttps://github.com/matsui528/lotf .
We utilized the following datasets:
• OpenAI Dataset [35, 45]: This dataset comprises 1536-
dimensional text features extracted from WikiText us-
ing OpenAI’s text embedding model. It consists of
900,000 base vectors and 100,000 query vectors. We use
this dataset for evaluation, considering that the proposed
method is intended for application in RAG systems.
• MS MARCO Dataset [6]: This dataset includes Bing
search logs. We extracted passages from the v1.1 vali-
dation set, deriving 768-dimensional BERT features [11],
resulting in 38,438 base vectors and 1,000 query vectors.
We used this dataset to illustrate redundant texts.
• Revisited Paris Dataset [37]: This image dataset fea-
tures landmarks in Paris, utilizing 2048-dimensional R-
GeM [38] features with 6,322 base and 70 query vectors.
It serves as an example of data with many similar images.
We used the first 1,000 vectors from base vectors for hyper-
parameter training ( Qtrain in Eq. (7)).

Cost function ( ↓) Runtime [ms/query] ( ↓) Memory overhead [bit] ( ↓)
Filtering Search Diversification Final ( f) Search Filter Total {xn}N
n=1 {Ln}K
k=1
None (Search only) 0.331 −0.107 0 .200 0 .855 - 0.855 - -
Clustering 0.384 −0.152 0 .223 0 .941 6 .94 7 .88 4 .42×1010-
GMM [40] 0.403 −0.351 0 .177 0.977 13 .4 14 .4 4 .42×1010-
LotusFilter (Proposed) 0.358 −0.266 0.171 1.00 0 .02 1 .03 - 1.14×109
Table 1. Comparison with existing methods for the OpenAI dataset. The parameters are λ= 0.3, K = 100 , S= 500 , ε∗= 0.277,and
L= 19.8. The search step is with HNSW [28]. Bold and underlined scores represent the best and second-best results, respectively.
7.1. Comparison with existing methods
Existing methods We compare our methods with existing
methods in Tab. 1. The existing methods are the ANNS
alone (i.e., HNSW only), clustering, and the GMM [3, 40].
• ANNS alone (no filtering): An initial search is performed
to obtain Kresults. We directly use them as the output.
• Clustering: After obtaining the initial search result S, we
cluster the vectors {xs}s∈SintoKgroups using k-means
clustering. The nearest neighbors of each centroid form
the final result K. Clustering serves as a straightforward
approach to diversifying the initial search results with the
running cost of O(DKS ). To perform clustering, we re-
quire the original vectors {xn}N
n=1.
• GMM: GMM is a representative approach for extracting
a diverse subset from a set. After obtaining the initial
search result S, we iteratively add elements to Kaccord-
ing to j∗= arg max j∈S\K 
mini∈K∥xi−xj∥2
2
, updat-
ingKasK ← K∪{ j∗}in each step. This GMM approach
produces the most diverse results from the set S. With a
bit of refinement, GMM can be computed in O(DKS ).
Like k-means clustering, GMM also requires access to the
original vectors {xn}N
n=1.
We consider the scenario of obtaining Susing modern
ANNS methods like HNSW, followed by diversification.
Since no existing methods can be directly compared in this
context, we use simple clustering and GMM as baselines.
Well-known DNNS methods, like Maximal Marginal
Relevance (MMR) [9], are excluded from comparison due
to their inability to directly utilize ANNS, resulting in slow
performance. Directly solving Eq. (1) is also excluded be-
cause of its high computational cost. Note that MMR can
be applied to Srather than the entire database vectors. This
approach is similar to the GMM described above and can be
considered an extension that takes the distance to the query
into account. Although it has a similar runtime as GMM, its
score was lower, so we reported the GMM score.
In the “Cost function” of Tab. 1, the “Search” refers to
the first term in Eq. (2), and the “Diversification” refers to
the second term. The threshold εis the value obtained from
Eq. (7). The runtime is the average of three trials.Results From Tab. 1, we observe the following results:
• In the case of NN search only, it is obviously the fastest;
however, the results are the least diverse (with a diversifi-
cation term of −0.107).
• Clustering is simple but not promising. The final score is
the worst ( f= 0.223), and it takes 10 times longer than
search-only ( 7.88[ms/query]).
• GMM achieves the most diverse results ( −0.351), attain-
ing the second-highest final performance ( f= 0.177).
However, GMM is slow ( 14.4[ms/query]), requiring ap-
proximately 17 times the runtime of search-only.
• The proposed LotusFilter achieves the highest perfor-
mance ( f= 0.171). It is also sufficiently fast ( 1.03
[ms/query]), with the filtering step taking only 0.02
[ms/query]. As a result, it requires only about 1.2 times
the runtime of search-only.
• Clustering and GMM consume 40 times more memory
than LotusFilter. Clustering and GMM require the orig-
inal vectors, costing 32ND [bits] using 32-bit floating-
points, which becomes especially large for datasets with
a high D. In contrast, the memory cost of the proposed
method is 64LNusing 64-bit integers.
The proposed method is an effective filtering approach re-
garding performance, runtime, and memory efficiency, es-
pecially for high-dimensional vectors. For low-dimensional
vectors, simpler baselines may be more effective. Please
see the supplemental material for details.
7.2. Impact of the number of initial search results
When searching, users are often interested in knowing how
to set S, the size of the initial search result. We evaluated
this behavior for the OpenAI dataset in Fig. 3. Here, λ=
0.3, andεis determined by solving Eq. (7) for each point.
Taking more candidates in the initial search (larger S)
results in the following:
• Overall performance improves (lower f), as having more
candidates is likely to lead to better solutions.
• On the other hand, the runtime gradually increases. Thus,
there is a clear trade-off in S’s choice.

102103
S0.180.200.22f
0.951 [ms/query]
1.0111.0950.894 [ms/query]
0.948
1.020K= 200
K= 100Figure 3. Fix K, vary S
0.2 0.3 0.4 0.5
ε0.330.34f
From test query
ε∗by Eq. 7 Figure 4. Evaluate ε∗by Eq. (7)Runtime [s]
N λ ε∗L Train Build
9×1030.3 0.39 8.7 96 0.16
0.5 0.42 19.6 99 0.17
9×1040.3 0.33 10.1 176 3.8
0.5 0.36 23.5 177 3.9
9×1050.3 0.27 18.4 1020 54
0.5 0.29 29.3 1087 54
Table 2. Train and build
7.3. Effectiveness of training
We investigated how hyperparameter tuning in the training
phase affects final performance using the OpenAI dataset.
While simple, we found that the proposed training proce-
dure achieves sufficiently good performance.
The training of εas described in Sec. 6 is shown in Fig. 4
(λ= 0.3, K= 100 , S= 500 ). Here, the blue dots repre-
sent the actual calculation of fusing various εvalues with
the test queries. The goal is to obtain εthat achieves the
minimum value of this curve in advance using training data.
The red line represents the ε∗obtained from the training
queries via Eq. (7). Although not perfect, we can obtain
a reasonable solution. These results demonstrate that the
proposed data structure can perform well by learning the
parameters in advance using training data.
7.4. Preprocessing time
Tab. 2 shows the training and construction details (building
the cutoff table) with K= 100 andS= 500 for the OpenAI
dataset. Here, we vary the number of database vectors N.
For each condition, εis obtained by solving Eq. (7). The
insights obtained are as follows:
• AsNincreases, the time for training and construction in-
creases, and Lalso becomes larger, whereas ε∗decreases.
• As λincreases, ε∗andLincrease, and training and con-
struction times slightly increase.
•Lis at most 30 within the scope of this experiment.
• Training and construction each take a maximum of ap-
proximately 1,100 seconds and 1 minute, respectively.
This runtime is sufficiently fast but could potentially be
further accelerated using specialized hardware like GPUs.
7.5. Qualitative evaluation for texts
This section reports qualitative results using the MS
MARCO dataset (Tab. 3). This dataset contains many short,
redundant passages, as anticipated for real-world use cases
of RAG. We qualitatively compare the results of the NNS
and the proposed DNNS on such a redundant dataset. The
parameters are K= 10 ,S= 50 ,λ= 0.3, andε∗= 18.5.Query : “Tonsillitis is a throat infection that occurs on the tonsil.”
Results by nearest neighbor search
1: “Tonsillitis refers to the inflammation of the pharyngeal tonsils and
is the primary cause of sore throats.”
2: “Strep throat is a bacterial infection in the throat and the tonsils.”
3: “Strep throat is a bacterial infection of the throat and tonsils.”
4: “Strep throat is a bacterial infection of the throat and tonsils.”
5: “Mastoiditis is an infection of the spaces within the mastoid bone.”
Results by diverse nearest neighbor search (proposed)
1: “Tonsillitis refers to the inflammation of the pharyngeal tonsils and
is the primary cause of sore throats.“
2: “Strep throat is a bacterial infection in the throat and the tonsils.“
3: “Mastoiditis is an infection of the spaces within the mastoid bone.“
4: “Tonsillitis (enlarged red tonsils) is caused by a bacterial (usually
strep) or viral infection.“
5: “Spongiotic dermatitis is a usually uncomfortable dermatological
condition which most often affects the skin of the chest, abdomen,
and buttocks.“
Table 3. Qualitative evaluation on text data using MS MARCO.
Simple NNS results displayed nearly identical second,
third, and fourth-ranked results (highlighted in red), while
the proposed LotusFilter eliminates this redundancy. This
tendency to retrieve similar data from the scattered dataset
is common if we run NNS. Eliminating such redundant re-
sults is essential for real-world RAG systems. See the sup-
plemental material for more examples.
The proposed LotusFilter is effective because it obtains
diverse results at the data structure level. While engineer-
ing solutions can achieve diverse searches, such solutions
are complex and often lack runtime guarantees. In contrast,
LotusFilter is a simple post-processing module with com-
putational guarantees. This simplicity makes it an advan-
tageous building block for complex systems, especially in
applications like RAG.
7.6. Qualitative evaluation for images
This section reports qualitative evaluations of images. Here,
we consider an image retrieval task using image features ex-
tracted from the Revisited Paris dataset (Fig. 5). The param-
eters are set to K= 10 ,S= 100 ,λ= 0.5, andε∗= 1.14.

NN
Search
Diverse 
Search 
(Proposed)Query1st 2nd                               3rd                                      4th                                        5th 
1st 2nd                             3rd                              4th                                        5th 
1st 2nd                               3rd                                 4th                                 5th 
1st 2nd                                             3rd                            4th                  5th NN
Search
Diverse 
Search 
(Proposed)QuerySimilar Similar Similar SimilarSimilarSimilar
SimilarFigure 5. Qualitative evaluation on image data using Revisited Paris.
In the first example, a windmill image is used as a query
to find similar images in the dataset. The NNS results are
shown in the upper row, while the proposed diverse search
results are in the lower row. The NNS retrieves images close
to the query, but the first, second, and fifth images show
windmills from similar angles, with the third and fourth im-
ages differing only in sky color. In a recommendation sys-
tem, such nearly identical results would be undesirable. The
proposed diverse search, however, provides more varied re-
sults related to the query.
In the second example, the query image is a photograph
of the Pompidou Center taken from a specific direction. In
this case, all the images retrieved by the NNS have almost
identical compositions. However, the proposed approach
can retrieve images captured from various angles.
It is important to note that the proposed LotusFilter is
simply a post-processing module, which can be easily re-
moved. For example, if the diverse search results are less
appealing, simply deactivating LotusFilter would yield the
standard search results. Achieving diverse search through
engineering alone would make it more difficult to switch
between results in this way.
7.7. Limitations and future works
The limitations and future works are as follows:
• LotusFilter involves preprocessing steps. Specifically, we
optimize εfor parameter tuning, and a cutoff table needs
to be constructed in advance.
• During εlearning, Kneeds to be determined in advance.In practical applications, there are many cases where K
needs to be varied. If Kis changed during the search, it
is uncertain whether ε∗is optimal.
• A theoretical bound has been established for the diver-
sification term in the cost function; however, there is no
theoretical guarantee for the total cost.
• Unlike ANNS alone, LotusFilter requires additional
memory for a cutoff table. Although the memory usage is
predictable at 64LN [bits], it can be considerable, espe-
cially for large values of N.
• When Dis small, more straightforward methods (such as
GMM) may be the better option.
• The proposed method determines a global threshold ε.
Such a single threshold may not work well for challeng-
ing datasets.
• The end-to-end evaluation of the RAG system is planned
for future work. Currently, the accuracy is only assessed
by Eq. (2), and the overall performance within the RAG
system remains unmeasured. A key future direction is
employing LLM-as-a-judge to evaluate search result di-
versity comprehensively.
8. Conclusions
We introduced the LotusFilter, a fast post-processing mod-
ule for DNNS. The method entails creating and using a cut-
off table for pruning. Our experiments showed that this ap-
proach achieves diverse searches in a similar time frame to
the most recent ANNS.

Acknowledgement
We thank Daichi Amagata and Hiroyuki Deguchi for re-
viewing this paper, and we appreciate Naoki Yoshinaga for
providing the inspiration for this work.
References
[1] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. Optuna: A next-generation hy-
perparameter optimization framework. In Proc. ACM KDD ,
2019. 5
[2] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. Optuna: A next-generation
hyperparameter optimization framework. In Proceedings of
the 25th ACM SIGKDD International Conference on Knowl-
edge Discovery and Data Mining , 2019. 1
[3] Daichi Amagata. Diversity maximization in the presence of
outliers. In Proc. AAAI , 2023. 2, 6
[4] Fabien Andr ´e, Anne-Marie Kermarrec, and Nicolas Le
Scouarnec. Quicker adc: Unlocking the hidden potential of
product quantization with simd. IEEE TPAMI , 43(5):1666–
1677, 2021. 2
[5] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen.
Acl2023 tutorial on retrieval-based language models and ap-
plications, 2023. 1
[6] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jian-
feng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNa-
mara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song,
Alina Stoica, Saurabh Tiwary, and Tong Wang. Ms marco:
A human generated machine reading comprehension dataset.
arXiv , 1611.09268, 2016. 5
[7] Dmitry Baranchuk, Artem Babenko, and Yury Malkov. Re-
visiting the inverted indices for billion-scale approximate
nearest neighbors. In Proc. ECCV , 2018. 2
[8] Sebastian Bruch. Foundations of Vector Retrieval . Springer,
2024. 1
[9] Jaime Carbonell and Jade Goldstein. The use of mmr,
diversity-based reranking for reordering documents and pro-
ducing summaries. In Proc. SIGIR , 1998. 2, 6
[10] Daoyuan Chen, Wuchao Li, Yaliang Li, Bolin Ding, Kai
Zeng, Defu Lian, and Jingren Zhou. Learned index with dy-
namic ϵ. InProc. ICLR , 2023. 2
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. In Proc. NAACL-HLT ,
2019. 5
[12] Jialin Ding, Vikram Nathan, Mohammad Alizadeh, and Tim
Kraska. Tsunami: A learned multi-dimensional index for
correlated data and skewed workloads. In Proc. VLDB , 2020.
2
[13] Matthijs Douze, Alexandre Sablayrolles, and Herv ´e J´egou.
Link and code: Fast indexing with graphs and compact re-
gression codes. In Proc. IEEE CVPR , 2018. 2
[14] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar ´e, Maria
Lomeli, Lucas Hosseini, and Herv ´e J´egou. The faiss library.
arXiv , 2401.08281, 2024. 2, 5[15] Marina Drosou and Evaggelia Pitoura. Search result diversi-
fication. In Proc. SIGMOD , 2010. 1, 2
[16] Marina Drosou and Evaggelia Pitoura. Disc diversity: Result
diversification based on dissimilarity and coverage. In Proc.
VLDB , 2012. 2
[17] Paolo Ferragina and Giorgio Vinciguerra. Learned Data
Structures . Springer International Publishing, 2020. 2
[18] Paolo Ferragina and Giorgio Vinciguerra. The pgmindex:
a fully dynamic compressed learned index with provable
worst-case bounds. In Proc. VLDB , 2020. 2
[19] Paolo Ferragina, Fabrizio Lillo, and Giorgio Vinciguerra.
Why are learned indexes so effective? In Proc. ICML , 2020.
2
[20] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. Fast
approximate nearest neighbor search with the navigating
spreading-out graph. In Proc. VLDB , 2019. 2
[21] Fuma Hidaka and Yusuke Matsui. Flexflood: Efficiently up-
datable learned multi-dimensional index. In Proc. NeurIPS
Workshop on ML for Systems , 2024. 2
[22] Kohei Hirata, Daichi Amagata, Sumio Fujita, and Takahiro
Hara. Solving diversity-aware maximum inner product
search efficiently and effectively. In Proc. RecSys , 2022. 2
[23] Wenzel Jakob. nanobind: tiny and efficient c++/python bind-
ings, 2022. https://github.com/wjakob/nanobind. 5
[24] Herv ´e J´egou, Matthijis Douze, and Cordelia Schmid. Prod-
uct quantization for nearest neighbor search. IEEE TPAMI ,
33(1):117–128, 2011. 2
[25] Mykel J. Kochenderfer and Tim A. Wheeler. Algorithms for
Optimization . The MIT Press, 2019. 5, 1
[26] Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, and
Neoklis Polyzotis. The case for learned index structures. In
Proc. SIGMOD , 2018. 2
[27] Qiyu Liu, Libin Zheng, Yanyan Shen, and Lei Chen. Stable
learned bloom filters for data streams. In Proc. VLDB , 2020.
2
[28] Yury A. Malkov and Dmitry A. Yashunin. Efficient and ro-
bust approximate nearest neighbor search using hierarchical
navigable small world graphs. IEEE TPAMI , 42(4):824–836,
2020. 2, 5, 6, 3
[29] Yusuke Matsui, Takuma Yamaguchi, and Zheng Wang.
Cvpr2020 tutorial on image retrieval in the wild, 2020. 1
[30] Yusuke Matsui, Yoshiki Imaizumi, Naoya Miyamoto, and
Naoki Yoshifuji. Arm 4-bit pq: Simd-based acceleration for
approximate nearest neighbor search on arm. In Proc. IEEE
ICASSP , 2022. 2
[31] Yusuke Matsui, Martin Aum ¨uller, and Han Xiao. Cvpr2023
tutorial on neural search in action, 2023. 1
[32] Michael Mitzenmacher. A model for learned bloom filters,
and optimizing by sandwiching. In Proc. NeurIPS , 2018. 2
[33] Vikram Nathan, Jialin Ding, Mohammad Alizadeh, and Tim
Kraska. Learning multi-dimensional indexes. In Proc. SIG-
MOD , 2020. 2
[34] Yutaro Oguri and Yusuke Matsui. General and practical tun-
ing method for off-the-shelf graph-based index: Sisap index-
ing challenge report by team utokyo. In Proc. SISAP , 2023.
2

[35] Yutaro Oguri and Yusuke Matsui. Theoretical and empiri-
cal analysis of adaptive entry point selection for graph-based
approximate nearest neighbor search. arXiv , 2402.04713,
2024. 5
[36] Naoki Ono and Yusuke Matsui. Relative nn-descent: A
fast index construction for graph-based approximate nearest
neighbor search. In Proc. MM , 2023. 2
[37] Filip Radenovi ´c, Ahmet Iscen, Giorgos Tolias, Yannis
Avrithis, and Ond ˇrej Chum. Revisiting oxford and paris:
Large-scale image retrieval benchmarking. In Proc. IEEE
CVPR , 2018. 5
[38] Filip Radenovi ´c, Giorgos Tolias, and Ond ˇrej Chum. Fine-
tuning cnn image retrieval with no human annotation. IEEE
TPAMI , 41(7):1655–1668, 2018. 5
[39] Vidyadhar Rao, Prateek Jain, and C.V . Jawahar. Diverse yet
efficient retrieval using locality sensitive hashing. In Proc.
ICMR , 2016. 2
[40] Sekharipuram S. Ravi, Daniel J. Rosenkrantz, and Giri Ku-
mar Tayi. Heuristic and special case algorithms for disper-
sion problems. Operations Research , 542(2):299–310, 1994.
2, 6, 3
[41] Rodrygo L. T. Santos, Craig Macdonald, and Iadh Ounis.
Search result diversification. Foundations and Trends in In-
formation Retrieval , 9(1):1–90, 2015. 1, 2
[42] Atsuki Sato and Yusuke Matsui. Fast partitioned learned
bloom filter. In Proc. NeurIPS , 2023. 2
[43] Xuan Shan, Chuanjie Liu, Yiqian Xia, Qi Chen, Yusi
Zhang, Kaize Ding, Yaobo Liang, Angen Luo, and Yuxiang
Luo. Glow: Global weighted self-attention network for web
search. In Proc. IEEE Big Data , 2021. 2
[44] Harsha Vardhan Simhadri, George Williams, Martin
Aum ¨uller, Matthijs Douze, Artem Babenko, Dmitry
Baranchuk, Qi Chen, Lucas Hosseini, Ravishankar Krish-
naswamny, Gopal Srinivasa, Suhas Jayaram Subramanya,
and Jingdong Wang. Results of the neurips’21 challenge on
billion-scale approximate nearest neighbor search. In Proc.
PMLR , 2022. 2
[45] Harsha Vardhan Simhadri, Martin Aum ¨uller, Amir Ing-
ber, Matthijs Douze, George Williams, Magdalen Dobson
Manohar, Dmitry Baranchuk, Edo Liberty, Frank Liu, Ben
Landrum, Mazin Karjikar, Laxman Dhulipala, Meng Chen,
Yue Chen, Rui Ma, Kai Zhang, Yuzheng Cai, Jiayang Shi,
Yizhuo Chen, Weiguo Zheng, Zihao Wan, Jie Yin, and Ben
Huang. Results of the big ann: Neurips’23 competition.
arXiv , 2409.17424, 2024. 5
[46] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan
Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi.
Diskann: Fast accurate billion-point nearest neighbor search
on a single node. In Proc. NeurIPS , 2019. 2
[47] Kapil Vaidya, Eric Knorr, Michael Mitzenmacher, and Tim
Kraska. Partitioned learned bloom filters. In Proc. ICLR ,
2021. 2
[48] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang
Wang. A comprehensive survey and experimental compari-
son of graph-based approximate nearest neighbor search. In
Proc. VLDB , 2021. 2[49] Jiacheng Wu, Yong Zhang, Shimin Chen, Jin Wang, Yu
Chen, and Chunxiao Xing. Updatable learned index with
precise positions. In Proc. VLDB , 2021. 2
[50] Kaiping Zheng, Hongzhi Wang, Zhixin Qi, Jianzhong Li,
and Hong Gao. A survey of query result diversification.
Knowledge and Information Systems , 51:1–36, 2017. 1, 2

LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table
Supplementary Material
A. Selection of data structures
We introduce alternative data structures for Sand demon-
strate that the proposed OrderedSet is superior. As intro-
duced in Sec. 5.2, an input array v= [v[1], v[2], . . . , v [V]]
containing Velements is given. The goal is to realize a data
structure that efficiently performs the following operations:
• POP: Retrieve and remove the foremost element while
preserving the order of the input array.
• R EMOVE : Given an element as input, delete it from the
data structure.
The average computational complexity of these opera-
tions for various data structures, including arrays, sets, pri-
ority queues, lists, and their combinations, are summarized
in Tab. A.
Array When using an array directly, the P OPoperation
follows the same procedure as OrderedSet. However, ele-
ment removal incurs a cost of O(V). This removal is im-
plemented by performing a linear search and marking the
element with a tombstone. Due to the inefficiency of this
removal process, arrays are not a viable option.
Set If we convert the input array into a set (e.g.,
std::unordered set in C++ or set in Python), el-
ement removal can be achieved in O(1). However, since
the set does not maintain element order, we cannot perform
the P OPoperation, making this approach unsuitable.
List Consider converting the input array into a list (e.g., a
doubly linked list such as std::list in C++). The first
position in the list is always accessible, and removal from
this position is straightforward, so P OPcan be executed in
O(1). However, for R EMOVE , a linear search is required to
locate the element, resulting in a cost of O(V). Hence, this
approach is slow.
Priority queue A priority queue is a commonly used data
structure for implementing P OP. C++ STL has a standard
implementation such as std::priority queue . If the
input array is converted into a priority queue, the P OPop-
eration can be performed in O(logV). However, priority
queues are not well-suited for removing a specified element,
as this operation requires a costly full traversal in a naive
implementation. Thus, priority queues are inefficient for
this purpose.List + dictionary Combining a list with a dictionary
(hash table) achieves both P OPand R EMOVE operations in
O(1), making it the fastest from a computational complex-
ity perspective. The two data structures, a list and a dic-
tionary, are created in the construction step. First, the input
array is converted into a list to maintain order. Next, the dic-
tionary is created with a key corresponding to an element in
the array, and a value is a pointer pointing to a correspond-
ing node in the list.
During removal, the element is removed from the dic-
tionary, and the corresponding node in the list is also re-
moved. This node removal is possible since we know its
address from the dictionary. This operation ensures the list
maintains the order of remaining elements. For P OP, the
first element in the list is extracted and removed, and the
corresponding element in the dictionary is also removed.
While the list + dictionary combination achieves the best
complexity, its constant factors are significant. Construct-
ing two data structures during initialization is costly. R E-
MOVE must also remove elements from both data structures.
Furthermore, in our target problem (Algorithm 2), the cost
of set deletions within the for-loop ( L6) is already O(L).
Thus, even though P OPisO(1), it does not improve the
overall computational complexity. Considering these fac-
tors, we opted for OrderedSet.
OrderedSet As summarized in Tab. A, our OrderedSet
introduced in Sec. 5.2 combines the advantages of arrays
and hash tables. During initialization, only a set is con-
structed. The only operation required for removals is dele-
tion from the set, resulting in smaller constant factors than
other methods.
B. Details of training
In Algorithm A, we describe our training approach for ε
(Eq. (7)) in detail. The input to the algorithm is the train-
ing query vectors Qtrain⊂RD, which can be prepared by
using a part of the database vectors. The output is ε∗that
minimizes the evaluation function f∗(ε,q)defined in Eq.
(6). Since this problem is a non-linear single-variable opti-
mization, we can apply black-box optimization [2], but we
use a more straightforward approach, bracketing [25].
Algorithm A requires several hyperparameters:
•εmax∈R: The maximum range of ε. This value can be
estimated by calculating the inter-data distances for sam-
pled vectors.
•W∈R: The number of search range divisions. We will
discuss this in detail later.

Method P OP() R EMOVE (a) Constant factor Overall complexity of Algorithm 2
Array O(∆) O(V) O(T+KLS )
Set (hash table) - O(1) N/A
List O(1) O(V) O(T+KLS )
Priority queue O(logV)O(V) O(T+KLS )
List + dictionary (hash table) O(1) O(1) Large O(T+S+KL)
OrderedSet: array + set (hash table) O(∆) O(1) O(T+S+KL)
Table A. The average computational complexity to achieve operations on S
Algorithm A: Training for ε
Input: Qtrain⊂RD
Hyper params: εmax∈R,W∈R,I,
λ∈[0,1],S,K
Output: ε∗∈R
1εleft←0 # Lower bound
2εright←εmax # Upper bound
3r←εright−εleft # Search range
4ε∗← ∞
5f∗← ∞
6repeat 5 times do
# Sampling Wcandidates at
equal intervals from the search
range
7E ←n
εleft+iεleft−εright
W|i∈ {0, . . . , W }o
# Evaluate all candidates and
find the best one
8 forε∈ Edo
9 f←E
q∈Qtrain[f∗(ε,q)]
10 iff < f∗then
11 ε∗←ε
12 f∗←f
13 r←r/2 # Shrink the range
# Update the bounds
14 εleft←max( ε∗−r,0)
15 εright←min(ε∗+r, εmax)
16return ε∗
• LotusFilter parameters: These include I,λ∈[0,1],S,
andK. Notably, this training algorithm fixes λ,S, and
K, and optimizes εunder these conditions.
First, the search range for the variable εis initialized in
L1andL2. Specifically, we consider the range [εleft, εright]
and examine the variable within this range ε∈[εleft, εright].
The size of this range is recorded in L3. The optimization
loop is executed in L6, where we decided to perform five
iterations. In L7,Wcandidates are sampled at equal inter-
vals from the current search range of the variable. We eval-uate all candidates in L8-12 , selecting the best one. Subse-
quently, the search range is narrowed in L13-15 . Here, the
size of the search range is gradually reduced in L13. The
search range for the next iteration is determined by center-
ing around the current optimal value ε∗and extending rin
both directions ( L14-15 ).
The parameter Wis not a simple constant but is dynam-
ically scheduled. Wis set to 10 for the first four iterations
to enable coarse exploration over a wide range. In the final
iteration, Wis increased to 100 to allow fine-tuned adjust-
ments after the search range has been adequately narrowed.
The proposed training method adopts a strategy similar
to beam search, permitting some breadth in the candidate
pool while greedily narrowing the range recursively. This
approach avoids the complex advanced machine learning
algorithms, making it simple and fast (as shown in Table
2, the maximum training time observed in our experiments
was less than approximately 1100 seconds on CPUs). As il-
lustrated in Fig. 4, this training approach successfully iden-
tifies an almost optimal parameter.
C. Experiments on memory-efficient datasets
We present the experimental results on a memory-efficient
dataset and demonstrate that simple baselines can be vi-
able choices. Here, we use the Microsoft SpaceV 1M
dataset [44]. This dataset consists of web documents rep-
resented by features extracted using the Microsoft SpaceV
Superion model [43]. While the original dataset contains
N= 109vectors, we used the first 107vectors for our ex-
periments. We utilized the first 103entries from the query
set for query data. The dimensionality of the vectors is 100,
which is relatively low-dimensional, and each element is
represented as an 8-bit integer. Therefore, compared to fea-
tures like those from CLIP, which are represented in float
and often exceed 1000 dimensions, this dataset is signifi-
cantly more memory-efficient.
Tab. B shows the results. While the overall trends are
similar to those observed with the OpenAI dataset in Ta-
ble 1, there are key differences:
• LotusFilter remains faster than Clustering and GMM, but
the runtime advantage is minor. This result is because Lo-

Cost function ( ↓) Runtime [ms/query] ( ↓) Memory overhead [bit] ( ↓)
Filtering Search Diversification Final ( f) Search Filter Total {xn}N
n=1{Ln}K
k=1
None (Search only) 10197 −778 6904 0 .241 - 0.241 - -
Clustering 11384 −2049 7354 0 .309 0 .372 0 .681 8 ×109-
GMM [40] 12054 −9525 5580 0.310 0 .367 0 .677 8 ×109-
LotusFilter (Proposed) 10648 −5592 5776 0.310 0 .016 0 .326 - 3.7×1010
Table B. Comparison with existing methods for the MS SpaceV 1M dataset. The parameters are λ= 0.3, K = 100 , S= 300 , ε∗= 5869 ,
andL= 58.3. The search step is with HNSW [28]. Bold and underlined scores represent the best and second-best results, respectively.
tusFilter’s performance does not depend on D, whereas
Clustering and GMM are D-dependent, and thus their
performance improves relatively as Ddecreases.
• Memory usage is higher for LotusFilter. This is due to the
dataset being represented as memory-efficient 8-bit inte-
gers, causing the cutoff table of LotusFilter to consume
more memory in comparison.
From the above, simple methods, particularly GMM, are
also suitable for memory-efficient datasets.
D. Additional results of qualitative evaluation
on texts
In Tab. C, we present additional results of a diverse search
on text data as conducted in Sec 7.5. Here, we introduce the
results for three queries as follows.
For the first query, “This condition...”, three identical
results appear at the first three results. When considering
RAG, it is typical for the information source to contain re-
dundant data like this. Removing such redundant data be-
forehand can sometimes be challenging. For instance, if the
data sources are continuously updated, it may be impossible
to check for redundancy every time new data is added. The
proposed LotusFilter helps eliminate such duplicate data as
a simple post-processing. LotusFilter does not require mod-
ifying the data source or the nearest neighbor search algo-
rithm.
For the second query, “Psyllium...”, the first and second
results, as well as the third and fourth results, are almost
identical. This result illustrates that multiple types of redun-
dant results can often emerge during the search. Without
using the proposed LotusFilter, removing such redundant
results during post-processing is not straightforward.
For the third query, “In the United...”, while there is no
perfect match, similar but not identical sentences are filtered
out. We can achieve it because LotusFilter identifies redun-
dancies based on similarity in the feature space. As shown,
LotusFilter can effectively eliminate similar results that can-
not necessarily be detected through exact string matching.

Query : “This condition is usually caused by bacteria entering the bloodstream and infecting the heart.”
Results by nearest neighbor search
1: “It is a common symptom of coronary heart disease, which occurs when vessels that carry blood to the heart become narrowed and blocked
due to atherosclerosis.”
2: “It is a common symptom of coronary heart disease, which occurs when vessels that carry blood to the heart become narrowed and blocked
due to atherosclerosis.”
3: “It is a common symptom of coronary heart disease, which occurs when vessels that carry blood to the heart become narrowed and blocked
due to atherosclerosis.”
4: “Cardiovascular disease is the result of the build-up of plaques in the blood vessels and heart.”
5: “The most common cause of myocarditis is infection of the heart muscle by a virus.”
Results by diverse nearest neighbor search (proposed)
1: “It is a common symptom of coronary heart disease, which occurs when vessels that carry blood to the heart become narrowed and blocked
due to atherosclerosis.”
2: “Cardiovascular disease is the result of the build-up of plaques in the blood vessels and heart.”
3: “The most common cause of myocarditis is infection of the heart muscle by a virus.”
4: “The disease results from an attack by the body’s own immune system, causing inflammation in the walls of arteries.”
5: “The disease disrupts the flow of blood around the body, posing serious cardiovascular complications.”
Query : “Psyllium fiber comes from the outer coating, or husk of the psyllium plant’s seeds.”
Results by nearest neighbor search
1: “Psyllium is a form of fiber made from the Plantago ovata plant, specifically from the husks of the plant’s seed.”
2: “Psyllium is a form of fiber made from the Plantago ovata plant, specifically from the husks of the plant’s seed.”
3: “Psyllium husk is a common, high-fiber laxative made from the seeds of a shrub.”
4: “Psyllium seed husks, also known as ispaghula, isabgol, or psyllium, are portions of the seeds of the plant Plantago ovata, (genus Plantago), a
native of India and Pakistan.”
5: “Psyllium seed husks, also known as ispaghula, isabgol, or psyllium, are portions of the seeds of the plant Plantago ovata, (genus Plantago), a
native of India and Pakistan.”
Results by diverse nearest neighbor search (proposed)
1: “Psyllium is a form of fiber made from the Plantago ovata plant, specifically from the husks of the plant’s seed.”
2: “Psyllium husk is a common, high-fiber laxative made from the seeds of a shrub.”
3: “Flaxseed oil comes from the seeds of the flax plant (Linum usitatissimum, L.).”
4: “The active ingredients are the seed husks of the psyllium plant.”
5: “Sisal fibre is derived from the leaves of the plant.”
Query : “In the United States there are grizzly bears in reserves in Montana, Idaho, Wyoming and Washington.”
Results by nearest neighbor search
1: “In the United States there are grizzly bears in reserves in Montana, Idaho, Wyoming and Washington.”
2: “In North America, grizzly bears are found in western Canada, Alaska, Wyoming, Montana, Idaho and a potentially a small population in
Washington.”
3: “In the United States black bears are common in the east, along the west coast, in the Rocky Mountains and parts of Alaska.”
4: “Major populations of Canadian lynx, Lynx canadensis, are found throughout Canada, in western Montana, and in nearby parts of Idaho and
Washington.”
5: “Major populations of Canadian lynx, Lynx canadensis, are found throughout Canada, in western Montana, and in nearby parts of Idaho and
Washington.”
Results by diverse nearest neighbor search (proposed)
1: “In the United States there are grizzly bears in reserves in Montana, Idaho, Wyoming and Washington.”
2: “In the United States black bears are common in the east, along the west coast, in the Rocky Mountains and parts of Alaska.”
3: “Major populations of Canadian lynx, Lynx canadensis, are found throughout Canada, in western Montana, and in nearby parts of Idaho and
Washington.”
4: “Today, gray wolves have populations in Alaska, northern Michigan, northern Wisconsin, western Montana, northern Idaho, northeast Oregon
and the Yellowstone area of Wyoming.”
5: “There are an estimated 7,000 to 11,200 gray wolves in Alaska, 3,700 in the Great Lakes region and 1,675 in the Northern Rockies.”
Table C. Additional qualitative evaluation on text data using MS MARCO.