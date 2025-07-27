# Threshold-Protected Searchable Sharing: Privacy Preserving Aggregated-ANN Search for Collaborative RAG

**Authors**: Ruoyang Rykie Guo

**Published**: 2025-07-23 04:45:01

**PDF URL**: [http://arxiv.org/pdf/2507.17199v1](http://arxiv.org/pdf/2507.17199v1)

## Abstract
LLM-powered search services have driven data integration as a significant
trend. However, this trend's progress is fundamentally hindered, despite the
fact that combining individual knowledge can significantly improve the
relevance and quality of responses in specialized queries and make AI more
professional at providing services. Two key bottlenecks are private data
repositories' locality constraints and the need to maintain compatibility with
mainstream search techniques, particularly Hierarchical Navigable Small World
(HNSW) indexing for high-dimensional vector spaces. In this work, we develop a
secure and privacy-preserving aggregated approximate nearest neighbor search
(SP-A$^2$NN) with HNSW compatibility under a threshold-based searchable sharing
primitive. A sharable bitgraph structure is constructed and extended to support
searches and dynamical insertions over shared data without compromising the
underlying graph topology. The approach reduces the complexity of a search from
$O(n^2)$ to $O(n)$ compared to naive (undirected) graph-sharing approach when
organizing graphs in the identical HNSW manner.
  On the theoretical front, we explore a novel security analytical framework
that incorporates privacy analysis via reductions. The proposed
leakage-guessing proof system is built upon an entirely different interactive
game that is independent of existing coin-toss game design. Rather than being
purely theoretical, this system is rooted in existing proof systems but goes
beyond them to specifically address leakage concerns and standardize leakage
analysis -- one of the most critical security challenges with AI's rapid
development.

## Full Text


<!-- PDF content starts -->

Threshold-Protected Searchable Sharing:
Privacy Preserving Aggregated-ANN Search for Collaborati ve RAG
Ruoyang Rykie Guo
gruoyang@stevens.edu
Abstract —LLM-powered search services have driven data in-
tegration as a signiﬁcant trend. However, this trend’s prog ress
is fundamentally hindered, despite the fact that combining
individual knowledge can signiﬁcantly improve the relevan ce
and quality of responses in specialized queries and make AI
more professional at providing services. Two key bottlenec ks
are private data repositories’ locality constraints and th e need
to maintain compatibility with mainstream search techniqu es,
particularly Hierarchical Navigable Small World (HNSW) in-
dexing for high-dimensional vector spaces. In this work, we
develop a secure and privacy-preserving aggregated approx i-
mate nearest neighbor search (SP-A2NN) with HNSW compat-
ibility under a threshold-based searchable sharing primit ive.
A sharable bitgraph structure is constructed and extended t o
support searches and dynamical insertions over shared data
without compromising the underlying graph topology. The
approach reduces the complexity of a search from O(n2)to
O(n)compared to naive (undirected) graph-sharing approach
when organizing graphs in the identical HNSW manner.
On the theoretical front, we explore a novel security
analytical framework that incorporates privacy analysis v ia
reductions. The proposed leakage-guessing proof system is built
upon an entirely different interactive game that is indepen dent
of existing coin-toss game design. Rather than being purely
theoretical, this system is rooted in existing proof system s but
goes beyond them to speciﬁcally address leakage concerns an d
standardize leakage analysis — one of the most critical secu rity
challenges with AI’s rapid development.
1. Introduction
As LLM-search systems scale to new heights, leveraging
data integration from diverse individual and institutional
sources empowers AI agents (e.g., GPTs) to decode complex
professional contexts with enhanced accuracy and depth,
especially within domain-speciﬁc ﬁelds such as biomedical
laboratory research, independent research institutions, and
expert-level query-response (QA) environments. As private
data is commonly stored in isolated and conﬁdential local
servers, signiﬁcant privacy concerns prevent these special-
ized domains from sharing and integrating their critical data
resources, limiting the realization of collaborative multi-user
LLM-search platforms. While existing single-user/multi-
tenancy RAG [1] architectures help LLM chatbots access in-
ternal private data, they fail to support multi-user knowledgesharing platforms in a collaborative pattern, where data is
remained conﬁdential for each individual data owner without
physically extracting the data from its original location.
To realize such a collaborative RAG environment
requires an aggregated approximate nearest neighbors
(A2NN) proximity search, given that standard RAG systems
rely on ANN similarity search as their core mechanism for
retrieving relevant contexts. While considering that existing
ANN search techniques adopted in RAG systems heavily
depend on hierarchical navigable small world (HNSW) [2]
indexing, the mainstream approach for high-dimensional
vector indexing, this dependency makes the Aggregated-
ANN problem extremely challenging under security and
privacy requirements.
In the realm of cryptographic protection techniques, the
series of multi-party computation (MPC) [3] techniques
seemingly offers intuitively viable solutions that enable the
aggregated setting by keeping local data in place while
allowing participating users (i.e., parties) to jointly per-
form search calculations. However, the HNSW indexing
method involves multilayer graphs, and this sophisticated
structure requires multiple rounds of interactions if directly
calculating each vertex of the graphs via MPC protocols.
This introduces overwhelming complexity since each ver-
tex corresponds to high-dimensional vector representations.
In comparison, fully homomorphic encryption is unsuit-
able due to efﬁciency concerns. Another category of non-
cryptographic approaches is inherently insufﬁcient under
this problem context, such as differential privacy or anony-
mous technologies, since authoritative data in specialized
domains demands the highest security standards and cannot
be utilized in real-world applications without rigorously
proven security guarantees. Existing secure search schemes
based on the searchable encryption (SSE) [4] technical line
cannot simultaneously satisfy both graph-based indexing
requirements [5] and distributed private calculation demands
in this setting. No methods are currently designed for either
direct HNSW structures or indirect graph-based searches
that can be adapted to HNSW indexing for vector search
over encrypted data.
CONTRIBUTIONS . In this work, we develop SP-A2NN,
a secure and privacy-preserving approximate nearest neigh-
bors search that is compatible with HNSW indexing and
supports distributed local data storage across individual
users. To achieve this goal, we ﬁrst formally formulate the
dynamic searchable sharing threshold primitive for the SP-arXiv:2507.17199v1  [cs.CR]  23 Jul 2025

A2NN problem. A pattern combining arithmetic and secret
sharing is utilized to perform distance comparisons during
searches, while a sharable bit-graph structure is designed to
minimize complexity, signiﬁcantly decreasing search com-
plexity from quadratic tolinear compared to distributing
original HNSW graphs in shared format. Interestingly, after
transforming from an original undirected graph (i.e., with
selective preservation of the original graph topology) to a
bit-graph while using the proposed two rule designs at-hand-
detour and honeycomb-neighbor , the searching walks re-
main nearly identical performance as in the original graphs.
On the theoretical analysis perspective, we adopt a novel
quantiﬁable reduction-based security analytical framework
that incorporates leakage analysis for validating SP-A2NN’s
security and privacy. The critical need for introducing a new
leakage-guessing proof system arises from a fundamental
gap on leakage between existing proof systems based on
adaptive security and their application to encrypted search
schemes. Current approaches address this gap through leak-
age functions that capture non-quantiﬁable states and cal-
culate detailed leakage ranges under threat models where
attackers possess varying knowledge (typically derived from
prior datasets, index, or their interconnections) to make
leakage measurable. In this work, we claim that this leakage
gap can be simulated in security environments through
standardized calculations via formal reductions, while this
still represents an uncharted theoretical ﬁeld, the formal
deﬁnitions and presentations are not entirely complete. A
high-level overview of such a proof system by reduction is
illustrated in Appendix A.1
2. Background
2.1. ANN Search
An appropriate nearest neighbor (ANN) search ﬁnds el-
ements in a dataset that are approximately closest to a given
query. The algorithm takes as input a query q, a dataset of
vectorsDand other parameters, and outputs approximate
nearest-neighbor IDs. We begin with revisiting the basic
ANN search for processing a query in brute-force way
in Functionality 2.1.1. A threshold θconstrains the query
range by setting either the maximum allowable neighbor
distance or desired count of vectors to be returned. Next,
we explore the HNSW indexing method that organizes high-
dimensional vectors (e.g., embeddings) in a hierarchical
structure to accelerate search.
2.2. HNSW Graph-Based Indexing for ANN Search
A hierarchical navigable small world (HNSW) algorithm
organizes a dataset Dusing a multilayered graph-based
indexI, with each layer being an undirected graph with
vertices as data elements. The layers compose a hierarchical
structure from top to down where each upper layer is
extracted from the layer below it with a certain probability.
Within each layer, graph construction follows a distance-
priority way in which elements closer in distance are moreFunctionality 2.1.1: ANN Search
Input: queryq, query threshold θfor either
distance/range or nearest neighbor count, vector
datasetD={vi}1≤i≤|D|, and other param
-eters, i.e., distance computation metric Distance ,
vector dimension dwithq,vi∈Rd.
Output: Nearest neighbors.
Brute-Force Procedure:
1:a←nearest neighbor to qinDB via brute-force
search
2:ifDistance(a,q)> θ then
3: the client outputs Null and⊥.
4:else
5: the client outputs a vector aand⊥.
6:end if
Functionality 2.2.1: HNSW Indexing Sketch for
ANN Search
Input : queryq, query threshold θfor either
distance/range or nearest neighbor count, vector
datasetD={vi}1≤i≤|D|with indexI, and other
parameters, i.e., distance metric Distance , vector
dimensions dwithq,vi∈Rd.
Output : Nearest neighbors.
HNSW Procedure :
1:Path←a routing path that connects layers of Iby
probabilistically skipping vectors according to their
distances (near or far)
2:a←nearest neighbor to qin the0th later found via
Path
3:if{Distance(a,q)> θ}then
4: the client outputs Null and⊥.
5:else
6: the client outputs a vector aand⊥.
likely to be connected through an edge. Generating such a
multi-layer index is a dynamic process of inserting dataset
elements (i.e., vectors) one by one from the top layer down
to the bottom layer, where the bottom layer contains the
complete dataset.
When processing a query, the algorithm traverses from
top layer to the most bottom layer until a query range of
appropriate nearest neighbors is reached. Given a query
element, HNSW search ﬁnds the nearest element in each
top layer (excluding the bottom layer). The nearest element
found in the Lth layer becomes the starting anchor for the
next lower layer ( (L-1)th), then the search ﬁnds its nearest
element in that layer; and this process continues layer by
layer until reaching the bottom layer, where the ﬁnal nearest
neighbors are identiﬁed within a speciﬁc range. Finally, the
IDs of neighbors satisfying the threshold θare returned
as the result. The search sketch is shown in Functionality
2.2.1. We recommend referring to Fig. 1 of ref [2] for
a better understanding of the search process. In brief, the
proximity graph structure replaces the probabilistic skip list
2

[6], maintaining a constant limit on edges (i.e., connections)
per layer, which enables HNSW search to achieve fast
logarithmic complexity for nearest neighbor queries, even
with high-dimensional vector data.
Following real-world database architecture that orga-
nizes data via index, it is established that a database consists
of two parts: dataset and index as
DB=D+I. (1)
3. Redeﬁne Problem
In this section, we deﬁne the system, security and threat
model of a secure and privacy-preserving aggregated ap-
proximate nearest neighbor (SP-A2NN) search problem.
3.1. System, Security and Threats Model of SP-
A2NN Search Scheme
Participating Parties. A party can be a client such as
a service provider (e.g., biomedical laboratory) using col-
laborative computing services, or an endpoint user (e.g.,
platform-agnostic worker) seeking to establish a collabora-
tion network with others. Take the RAG frameworks for ex-
ample, each party conﬁgures a database typically as vectors
based on their individual knowledge (e.g., ﬁles) to leverage
external AI retrieval services such as language models. The
computing task is to retrieve relevant knowledge across all
parties, creating a collaborative knowledge database to let
language models easily draw upon when producing answers.
For brevity, our framework focuses only on outputting the
retrieved data, excluding the process of parties forwarding
the results to a language model.
A setUofnparties participate in executing an aggre-
gated SP-A2NN search. Imagine that all parties integrate
their data and jointly establish/update a global index, main-
taining an idealized collaborative database C-DB together,
in which a global index C-Iorganizes a uniﬁed dataset
C-Dacross parties. Searches using this global index are
completed through interactions among parties, with each
element of the uniﬁed dataset accessible via a unique pointer
in the global index regardless of this element ownership. The
ﬁnal search result aggregates the queried nearest neighbors
from all parties. The structure is represented as
C-DB=C-D+C-I (2)
Security and Threat Model. In a SP-A2NN search, par-
ticipating parties do not trust one another and seek to keep
their individual databases conﬁdential from other parties. We
consider honest-but-curious security environments, where
parties follow the protocol honestly but may attempt to
infer other parties’ data during execution. While this can
be extended to prevent active adversaries through additional
processes consistency veriﬁcation, we omit this from the
current work. While the threat model traditionally concerns
attackers’ prior knowledge, in this work, we employ a pri-
vacy triplet setting to connect standardizable threat patternswith leakage analysis. The objective is to make privacy anal-
ysis as the foundation for the security analysis framework.
4. Preliminaries
In this section, we deﬁne a basic cryptographic primitive,
dynamic searchable sharing threshold (SST), for formulating
the problem of SP-A2NN search. Under this primitive, we
provide related security deﬁnitions and related constructions
in Sec 4.1, along with the existing cryptographic building
blocks used in this realization in Sec 4.2. Sec 4.3 deﬁnes
privacy triplet.
4.1. Dynamic SST
Dynamic SST evolves from dynamic SSE capabilities
for searches and updates (such as insertion and deletion),
adapting its deﬁnitions to work in a database environment
where data is distributed across multiple separate parties.
Conceptual Settings. As in SSE schemes, EDB denotes
the encrypted database that combines encrypted index and
encrypted data blocks (i.e., storage units), but with expanded
scope in dynamic SST. We introduce C-EDB as an abstract
construct, representing an idealized collaborative database in
encrypted form that integrates an uniﬁed encrypted dataset
(i.e.,C-ED) with a global encrypted index (i.e., C-EI),
that is
C-EDB=C-ED+C-EI. (3)
From a real perspective, C-EDB integrates separate dataset
segments, along with index segments, distributed among and
maintained by parties,
C-EDB=n/summationdisplay
1EDBi (4)
=n/summationdisplay
1EDi+n/summationdisplay
1EIi (5)
whereEDBiis the portion of C-EDB that is physically
stored in party ui, including data and index segment, EDi
andEIirespectively.
Dynamic SST Deﬁnition. A dynamic SST problem
Σ ={Setup,Search,Update}is comprised of interactive
protocols as:
Setup(1λ)→K,σ,C -EDB : It takes as input database
DB andλ, the computational security parameter of the
scheme (i.e., security should hold against attackers running
in time≈2λ). The outputs are collectively maintained by
all parties. Kis secret key of the scheme, analogous to the
arithmetic protection applied to data (e.g., the constructed
polynomial formula in Shamir’s secret sharing [7]). σis an
chronological state agreed across parties, and C-EDB is an
encrypted (initially empty) collaborative database.
Search(K,σ,q;C-EDB)→C-EDB(q): It represents
a protocol for querying the collaborative database. We as-
sume that a search query qis initiated by party v. The
protocol outputs C-DB(q), meaning that the elements that
3

are relevant to q(i.e., appropriate nearest neighbors in vector
format) are returned.
Insert/Delete(K,σ,in;C-EDB)→K,σ,C -EDB : It
is a protocol for inserting an element ininto (or deleting it
from) the collaborative database. The element inis a vector
owned by the party who requests an update. The protocol
ends with a new state where all parties jointly conﬁrm if
C-EDB contains the element inor not.
The above deﬁnitions extend the APIs of common dy-
namic SSE [8] to adapt the database structure, speciﬁcally
representing data storage blocks (e.g., dataset C-ED) and
index (e.g., C-EI). TheSearch algorithm’s result shows
which nearest neighbors are retrieved in response to a given
query, while the process is independent of how parties
subsequently forward the results to a language model (Sec
3.1).
QUANTIFIABLE CORRECTNESS . A dynamic SST prob-
lemΣ = (Setup,Search,Update)is correct if it returns the
correct results for any query with allowable deviation.
Correctness is a relative term that quantiﬁes identical
search results by comparing them to a baseline search as
reference, where this baseline is generally a search over
plaintext data under the same index. While searching in any
applied secure scheme, result deviation inevitably occurs,
making it difﬁcult to maintain the same identical search
results that an Enc/Dec oracle can achieve. Therefore, we
introduce a concept of deviation to deﬁne correctness below.
QUANTIFIABLE SECURITY . A dynamic SST problem
Σ = (Setup,Search,Update)is secure with bounded leak-
age if it is proven to satisfy an allowable privacy budget of
a certain value that can be standardized for measurement.
Both the privacy and threshold-based security analysis
of SST are captured simultaneously using a reduction-based
leakage-guessing proof system: The I DEAL experiment ex-
presses the layer where the basic security scheme achieves
provable threshold-based security, while the R EAL experi-
ment represents any applied scheme (such as our proposed
SP2ANN scheme) that, as the outermost layer, must capture
leakage. Although direct reduction w.r.t security from R EAL
to I DEAL can identify threshold-based security, it cannot
locate where/what level of leakage occurs. A new M IRROR
environment is then introduced as an intermediary bridge
that enables comparison with the I DEAL environment for
threshold-based security analysis and with the R EAL envi-
ronment for leakage analysis.
Deﬁnition 4.1.1 (∆-bounded Deviation-Controlled Correct-
ness of Dynamic SST ).A dynamic SST problem Πis∆-
correct ifffor all efﬁcientA, there exists a stateful efﬁcient
S, such that
AdvSST-Cor
Π,A(λ,ρ) =
AdvSST-Cor
Πbas,A(λ)+AdvSST-Cor
ΠM,A(λ)+AdvSST-Cor
Π,A(λ,ρ)(6)
where the ﬁrst two functions satisfy
AdvSST-Cor
ΠM,A(λ) ={MIRRORA
ΠM(λ)} ≡
{IDEALA
ΠBas(λ)}=AdvSST-Cor
Πbas,A(λ)(7)and are negligible , and
AdvSST-Cor
Π,A(λ,ρ) ={REALA
Π(λ)}−{ SIMA
∆(Π,ΠM),S(λ,ρ)}
(8)
is aunnegligible function in terms of the allowable deviation
ρ.{·}means the probability that adversary wins in the
experiment.
Deﬁnition 4.1.2 (L(ǫ)-bounded Threshold Security of Dy-
namic SST ).A dynamic SST problem ΠisL-secure ifffor
all efﬁcientA, there exists a stateful efﬁcient S′′,S′andS,
such that
AdvSST-Sec
Π,A(λ,ǫ) =
AdvSST-Threshold
ΠBas,A(λ)+AdvSST-Threshold
ΠM,A(λ)+AdvSST-Privacy
Π,A(λ,ǫ)
(9)
where
AdvSST-Threshold
ΠBas,A,S′′(λ) ={IDEALA
ΠBas(λ)}−{SIMA
L(ΠBas,SS),S′′(λ)}
(10)
AdvSST-Threshold
ΠM,A,S′(λ) ={MIRRORA
ΠM(λ)}−{SIMA
L(ΠM,ΠBas),S′(λ)}
(11)
are both negligible functions, and
AdvSST-Privacy
Π,A,S(λ,ǫ) ={REALA
Π(λ)}−{SIMA
L(Π,ΠM),S(λ,ǫ)}
(12)
is aunnegligible function in terms of the allowable privacy
budgetǫ.{·}follows the same meaning.
EXPERIMENTS DEFINITION . Importantly, the above ex-
periments extend existing query-response games for adaptive
data security. For instance, {REALA
Π(λ)}(Def 4.1.2.(12))
deﬁnes adversary’s adaptive security advantage against en-
crypted data in a real scheme Π, with this advantage is
veriﬁed through game-based experiments. In this work, we
assume that part of the security has been validated; therefore,
we omit the query-response game experiments for adaptive
security while looking forward a little bit. Instead, we par-
ticularly focus on a gap in current security proof systems:
privacy simulation.
In Deﬁnition 4.1.2, {SIMA
L(Π,ΠM),S(λ,ǫ)}isA’s advan-
tage on an environment for simulating Π. With the same
logic, this advantage is established based on but extends
beyond an adaptive security environment, where a simulator
Ssimulates a reduction from ΠtoΠM, and the output of this
environment is the leakage L(Π,ΠM) =L(ǫ). Analogously,
{SIMA
L(ΠM,ΠBas),S′(λ)}is an environment for simulating ΠM
that is provided by {MIRRORA
ΠM}; andA’s advantage is
calculated via S′’s simulation of the reduction from ΠMto
ΠBas, where the leakage L(ΠM,ΠBas) =L(λ)meaning that
it is allowable.
In Deﬁnition 4.1.1, {SIMA
∆(Π,ΠM),S(λ,ρ)}is an envi-
ronment for simulating Πthat is provided by {REALA
Π(λ)}
w.r.t correctness, where ∆is a function for captur-
ing result deviation between ΠandΠM). Of particular
note,{MIRRORA
ΠM(λ)}establishes the correctness baseline,
meaning it satisﬁes complete correctness equivalence with
{IDEALA
ΠBas(λ)}.
4

Basic Construction. Let a(t,n)-threshold secret shar-
ing conﬁguration SS serve as an encryption scheme of
(Enc,Dec),F1be polynomial formula (i.e., keys) for pro-
ducing shares and F2be an arithmetic circuit for calculating
ciphers. We have our basic static construction ΠSSfor the
dynamic SST problem in Fig 1.
Theorem 4.1.3 .A basic scheme ΠSSis correct and
threshold-secure iffthe(t,n)-threshold secret sharing mech-
anismSSis information-theoretically secure.
Mirror Construction. Let a(t,n)-threshold secret shar-
ing conﬁguration SS serve as an encryption scheme of
(Enc,Dec), andI-hnsw be the HNSW index to organize
C-EDB .F1andF2use the same representations in ΠSS.
We have our mirror construction ΠI-hnsw
SS in Fig 2.
Theorem 4.1.4 .A mirror scheme ΠI-hnsw
SS is correct andL-
secure iffΠSSis threshold-secure and the reduction from
ΠI-hnsw
SS toΠSSw.r.t leakage isL(ΠI-hnsw
SS,ΠSS)-secure.
The proofs for Theorem 4.1.3 and Theorem 4.1.4 are
provided in Appendix A.2.1 and A.2.2 respectively.
4.2. Cryptographic Building Blocks
Shamir’s t-out-of-nSecret Sharing Scheme. Within this
mechanism, any subset of tshares enables recovery of the
complete secret sthat has been divided into nparts, while
any collection of up to t−1shares yields no information
abouts. The generation of shares is parameterized over
a ﬁnite ﬁeld Fof sizel >2k(i.e.,kis the security
parameter of the scheme), where, e.g., F=Zpfor some
public prime p1. In our scheme, parties share their data as
vectors. As a result, we constrain this ﬁeld size parameter
byl≥Max(2k,⌈10ρ⌉,n), where10ρrepresents the scaling
factor applied to a vector. The scheme is composed of two
algorithms SS= (SS.Sharet
n,SS.Recont
n), one for sharing
a secret with parties and the other for reconstructing a share
from a subset of parties.
The sharing algorithm SS.Sharet
n(s)→ {(u,su)}u∈U
takes as input a secret s, a setUwith size|U|=nfor
parties, a threshold t≤n, and it produces a set of shares,
representing a party uholds a share sufor all parties inU.
The reconstruction algorithm SS.Recont
n({(u,su)}u∈V)→
sinputs a threshold tand shares related to a set V ⊆U
with|V|≥t, and outputs the secret sas a ﬁeld element.
CORRECTNESS . At-out-of-nsecret sharing scheme SS
correctly shares a secret sif it always reconstructs s.
SECURITY . At-out-of-nsecret sharing scheme privately
shares a secret sif∀s,s′∈Fand anyV⊆U with|V|< t,
there exists the view of parties in Vduring an execution of
SSfor sharing sand that view for s′, such that
{VIEWSS
u∈V({(u,s′
u)u∈U})}≡{ VIEWSS
u∈V({(u,su)u∈U})},
(13)
1. The selection of a prime pis constrained by some public integer r
withl=pr,pr>2kwhere{(u,s′
u)u∈U} ←SS.Sharet
n(s′),{(u,su)u∈U} ←
SS.Sharet
n(s), and≡denotes computational indistinguisha-
bility (bounded by distributions).
4.3. Privacy Triplet for Leakage Analysis
An intuition is that the ratio of inferrable data (e.g.,
based on prior knowledge) to the complete database provides
a measure of leakage severity. We deﬁne prior knowledge
as information already known to an adversary excluding
publicly available information such as open-source indexing
algorithms. When an adversary attempts to deduce addi-
tional information from a private database, we assume their
prior knowledge is limited to a single, randomly chosen data
entry. This approach allows us to measure the fraction of the
database that becomes exposed when following deduction
paths from a single, randomly selected data entry. The re-
sulting ratio of inferrable data provides a standardized metric
for comparing leakage across different security schemes.
For formulating this, we redeﬁne privacy leakage via an
analytical framework, named Privacy Triplet . This triplet
deﬁnes three interfaces (I-III) of measurable leakage with
dependently progressive strength as follows. A complete in-
ference trajectory, traversing from the starting interface (I) to
the ﬁnal interface (III), traces a linking path to its impacted
inferrable data, beginning from a single, randomly selected
data entry. A complete trajectory following a privacy triplet
is deﬁned as:
I.Data-to-Index privacy interface LI.It leaks the
index nodes that match a chosen data item.
II.Index-to-Index privacy interface LI.It leaks the
index nodes that can be deduced through other nodes already
linked to the chosen data item.
III.Index-to-Data privacy interface LD.It leaks ad-
ditional data (or indirect information of data) that can be
connected to the inferred index nodes from I and II.
Deﬁnition 4.3.1 (Privacy Triplet ).Given any search scheme
(e.g., dynamic SST Σ) build on an encrypted database
EDB , a privacy triplet standardizes the leakage disclosure
ofEDB by taking an individual, randomly chosen data item
wthrough a complete I-III trajectory as:
I-Data-to-Index :LI(w) =L′(DB-Inx(w),w).
II-Index-to-Index :LI(w′) =L′(DB-Inx(w′),DB-Inx(w)).
III-Index-to-Data :LD(w) =L′(w′,DB-Inx(w))
whereL′is a stateless function.
Cooperating with Existing Privacy Norms. An privacy
triplet creates an analytical approach for bounding leakage
severity, making it not conﬂict with established metrics
that identify patterns of leak-inducing behaviors. Existing
leakage patterns describe leak-inducing behaviors occurring
during search, update and access operations. Update and
access patterns have been relatively well explored and de-
ﬁned. Access patterns captures the observable sequence of
data locations that are accessed during searches. Update
patterns are typically associated with forward and backward
5

Setup(1λ,σ)
1:sdi$←{0,1}λallocate list L
2:Initiate Counter σ:c←0
3:Ki←F1(sd1,c)
4:AddKiinto listL(in lex order)
5:OutputK= (Ki,σ)Insert(K,σ,q;C-EDB)
1:(partyu){qu}U←Enc(q,K1)
2:Set
C-EI←C-EI.Add({loc(qu)}U);
C-ED←C-ED.Add({qu}U);
σ:c++
3:Output
C-EDB= (C-EI,C-ED,σ)Search(K,σ,q;C-EDB)
1:(partyv){qu}U←Enc(q,K2)
2:On input{q}Uand
C-EDB= (C-EI,C-ED,σ)
3:Forc= 0 untilBruteForce return⊥,
{vu}U←
BruteForce (C-EI;C-ED,{q}U,F2)
4:v←Dec({v}U,K)
5:Outputv
Figure 1: Basic Scheme ΠSS
Setup(1λ,σ)
1:sdi$←{0,1}λallocate list L
2:Initiate Counter σ:c←0
3:Ki←F1(sd1,c)
4:AddKiinto listL(in lex order)
5:OutputK= (Ki,σ)Insert(K,σ,q;C-EDB)
1:(partyu){qu}U←Enc(q,K1)
2:Set
C-EI←
C-EI.Add(I-hnsw:{loc(qu)}U);
C-ED←C-ED.Add({qu}U);
σ:c++
3:Output
C-EDB= (C-EI,C-ED,σ)Search(K,σ,q;C-EDB)
1:(partyv){qu}U←Enc(q,K2)
2:On input
{q}←v:Enc(q)
C-EDB= (C-EI,C-ED,σ)
3:Forc= 0 untilHNSW return⊥,
{vu}U←
HNSW(C-EI;C-ED,{q}U,F2)
4:v←Dec({v}U,K)
5:Outputv
Figure 2: Mirror Scheme ΠI-hnsw
SS
privacy, which address the leakage incurred by earlier and
later updates (i.e., insertions and deletions). Existing works
deﬁne search patterns in a more ﬂexible way to explore how
correlations between previously executed and subsequent
queries affect information exposure. We illustrate possible
locations within the leakage-guessing analytical framework
where these patterns can be integrated, as shown in the
Appendix A.1.
5. Technical Intuitions
We lay the technical foundation and preparations in this
section for constructing an efﬁcient aggregated approximate
nearest neighbor search scheme (SP-A2NN) with security
and privacy guarantees, detailed in Sec 6. This involves a
novel storage structure termed bitgraph along with its essen-
tial functionalities in Sec 5.2 to enable subsequent effective
aggregated searching. In Sec 5.1, we examine the efﬁciency
dilemma that emerges when naively distributing HNSW
graphs (undirected graphs) to construct a sharable index,
thereby justifying our bitgraph approach. Through complex-
ity analysis comparing unmodiﬁed graphs with bitgraphs
for executing aggregated queries, we show that bitgraphs
achieve a reduction from quadratic tolinear complexity.
By design, the bitgraph structure signiﬁcantly decreases
the number of invocations of Shamir’s secret sharing by us-
ing minimal information to convey the complete structure of
HNSW graphs. The additional optimizations further imple-
ment search/update functionalities by introducing the most
minimal changes possible based on the bitgraph framework.5.1. Establishing the Critical Need for Bitgraph
In what follows, we ﬁrst analyze theoretical arguments
on computational complexity that unavoidably arises when
sharing undirected graphs without proper conversions, and
then discuss the inherent tensions between this cost, search
functionality, and efﬁciency.
COMPLEXITY ANALYSIS . We begin with a scenario
wherein an undirected graph, comprising vertices and their
connecting edges, is to be distributed across multiple parties
in such a way that all participants (at least t) possess
sufﬁcient information to reconstruct the complete graph.
A graph’s topological structure is captured in its vertex
connection pattern, with each edge serving as a link between
two vertices. Thus, when quantifying the complexity of
graph sharing, the metric is the minimum number of vertices
that must be distributed in shares and exchanged among
parties to convey the graph’s complete structure.
If we consider the sharing operation on a vertex as a
one-time pad encryption, then this vertex cannot be traced
back after it has been checked during searching. This means
connections related to a vertex must be collaboratively
recorded when sharing it. In consequence, the link between
two vertices must be shared repeatedly among parties even
only considering minimum sharing patterns. We deﬁne such
structures as sharable sets for representing the complete
structure of a graph, and we only focus on the minimum
sharable set of any given graph.
Deﬁnition 5.1.1 (Minimum Sharable Set (MSS) ).For sharing
any undirected graph G= (V,E) :{Vas a set of vertices
andEfor a set of edges }among parties, there exists a
6

minimum sharable set Sthat contains all distinct directed
connections between vertices, and we have the size of this
set satisﬁes twice the number of edges, that is Size(S) =
2|E|.
Based on the deﬁnition on MSS, the complexity analysis
for sharing an undirected graph becomes more straightfor-
ward. We validate this through several logical deductions.
Deduction 5.1.2 .For any undirected graph G={V,E}and
its minimum sharable set S, when sharing Gamongnpar-
ties using a t-out-of-nsecret sharing mechanism SS.Sharet
n,
the computational complexity is:
O(SS.Sharet
n(G)) =Size(S)·n·O(SS.Sharet
n(·)),(14)
whereO(SS.Sharet
n(·))denotes the computational cost of an
invoking the sharing algorithm for each element in S. The
relation holds because this complexity is proportional to the
count of invoking sharing algorithms, which is proportional
to size of its minimum sharable set. (Note that when we dis-
cuss complexity in this work, we are speciﬁcally measuring
only the computational burden placed on a single party.)
Take the graph with two edges in Fig 3 as an instance, its
minimum sharable set is {a-b,a-c,b-a,c-a}. In this case,
the computational cost of conveying the complete structure
of it among parties is linear to at least four times the count
of sharing operations.
a
bc
Figure 3: A simple graph example
Whereas, when characterizing the computational com-
plexity of sharing a complete graph, it is more appropriate to
express this in terms of the number of vertices rather than the
number of edges. This metric is suitable because the vertex
count directly corresponds to both the count of sharing op-
erations and the database size (where each vertex represents
a vector record). By analyzing the relations between edges
and vertices (Def 5.1.1), we establish an upper bound on the
size of this minimum set, that is Size(S)≤|V|2. (Referring
to a basic deduction2in the context of graph theory [9].)
Deduction 5.1.3 .For any undirected graph G={V,E}
and its minimum sharable set S, when sharing Gamong
nparties using SS.Sharet
n, assuming any vertex vin the
vertex space satisﬁes the format v∈Rd, the computational
complexity is:
O(SS.Sharet
n(G)) =d|V|2·n·O(pt), (15)
whereddenotes vector dimension, pis modular of the ﬁnite
ﬁeld arithmetic in which the mechanism SS.Sharet
noperates
andtis the degree of the polynomial used for calculating
shares.
2. For any graph G= (V,E), the maximum degree of any vertex is
at most|V| −1, and then the total number of edges |E|cannot exceed
|V|(|V|−1)
2.Reconciling Search Functionality Challenge. Efﬁciency
is impeded by the quadratic invocations for generating
shares, and although not the primary factor, this contributes
to signiﬁcant computational ﬂows during search operations.
The process of repeatedly sharing vertices and their as-
sociation relations leads to quadratic distance comparison
calculations over shares, an essential requirement for search
functionality to determine if a speciﬁc vertex is being
queried.
Real-world searches over feature vectors commonly uti-
lize three distance operators for similarity evaluation: Eu-
clidean Distance ,Inner Product andCosine Similarity . As-
suming the compatibility with existing arithmetic protocols
to implement distance calculations over shares, selecting
an arithmetic circuit that supports a full set of calculation
operators to compute shares becomes an unavoidable step.
However, the computational burden imposed by any arith-
metic protocol that offering universal calculation capabil-
ities (i.e., both addition and multiplication) is substantial
enough to impact any single search, regardless of the speciﬁc
protocol chosen — even without considering the additional
authentication costs required to address malicious security
threats.
We visualize the complexity of searching on shares
through the following deduction.
Deduction 5.1.4 .Letcdenote the number of vertices in
a searching walk over any undirected graph G={V,E}
withc≤ |V|. Then, when sharing Gamongnparties
usingSS.Sharet
nand based on previous deductions, the
computational complexity of a search operation is:
O(SearchonShares({u,Gu}u∈U)) =dc2·n·O(pt)·O(AC),
(16)
whereO(AC)represents the complexity of arithmetic cir-
cuits that are required for performing distance similarity and
comparisons between two vertices.
In this work, we address the search efﬁciency dilemma
under this problem by minimizing the number of arithmetic
circuit invocations, rather than reducing the cost of each
invocation. In the next subsection, we propose a novel
storage structure, bitgraph, to support linear complexity for
sharing, calculating distance over shares, and reconstructing
search results, making practically effective index and search
systems possible.
5.2. Sharable Bitgraph Structure
A sharable bitgraph is the key storage structure required
for constructing a sharable index to realize aggregated ANN
search. This bitgraph structure, derived from HNSW graphs
(i.e., undirected graphs), maintains the integrity of original
inserting and searching walks. The proposed bitgraph elim-
inates the process of sharing entire graphs with their full
set of vertices and edges, signiﬁcantly reducing complexity
(i.e., sharing times) from quadratic tolinear .
Intuitions. We explain the efﬁciency of bitgraph sharing
by showing how it eliminates redundant vertex connec-
tion records through a strategic decomposition of graph
7

information. This design signiﬁcantly reduces complexity
while preserving complete graph representation. A bitgraph
replaces the traditional vertex-edge structure with four com-
ponents: vertices ,sequences ,post-positive degrees , and par-
allel branches . The sequence component extracts the graph’s
fundamental structure: an ordered path that visits all vertices
where any vertex in this path keeps a single forward con-
nection with its pre-positive vertex, while the path obviously
and potentially omitting some edges. Post-positive degrees
record only the backward connections of each vertex along
this sequence. Conceptually, if we view a graph as a system
of connected branches (i.e., subgraphs), the ﬁrst three com-
ponents fully describe individual branches without capturing
inter-branch connections. The fourth component, parallel
branches, records these branch-to-branch relationships. With
all four components, we can completely reconstruct the
original graph starting from any vertex.
Search Intuitions. We introduce a conceptual intuition
for searching: the search behaves like a walk that winds
through the graph structure in a hexagonal honeycomb
pattern, advancing toward deeper regions of a graph. This
design leverages the previously described information to
establish a natural bidirectional search trajectory, with one
direction following forward connections and the other fol-
lowing backward connections, both guided by the estab-
lished vertex sequences.
Bitgraph Roadmap. In what follows, we ﬁrst discuss
two pre-requisite deﬁnitions, subgraphs and its partition in
undirected graphs and the isomorphism relations how graphs
relate to bitgraphs. A bitgraph construction is then presented
with a helpful example. Next, we provide algorithms for
insertion and search operations on bitgraphs, with searches
built on HNSW principles and enhanced with bitgraph-
speciﬁc optimizations. Throughout, examples demonstrating
these operations are provided. We conclude by establishing
correctness proofs for search result consistency and ana-
lyzing the complexity of bitgraph sharing involved in con-
structing/searching bitgraphs across multiple shares. (Note
that graphs in this context have practical interpretations: a
vertex represents a high-dimensional vector, while an edge
corresponds to distance metrics between these vectors.)
Deﬁnition 5.2.1 (Subgraph ).Any undirected graph with
ordered vertices and edges G={V,E}can be expressed
as a set of subgraphs generated by applying a partitioning
functionΓas
GΓ=/uniondisplay
Subgraph( G)Γ=/uniondisplay
i(Gi), (17)
such that this union set contains the complete vertices (pos-
sibly repeated), edges, and the original ordered structure of
G.
Deﬁnition 5.2.2 (Bitgraph Isomorphism ).A bitgraph iso-
morphism ffrom an undirected graph Gto a bitgraph H
is a bijection (i.e., one-to-one correspondence) between the
subgraph set of Gand the branch set of H, that is
f:/uniondisplay
Subgraph( G)→/uniondisplay
Branch(H), (18)such that each branch of His the image of exactly one
subgraph of G. To further explore the isomorphism fi, which
maps the vertex and edge structure of a subgraph Gito its
corresponding branch Hi, we introduce
fi:Gi→Hi. (19)
Speciﬁcally, we deﬁne the branch set for a bitgraph Has
/uniondisplay
i(Hi) =/uniondisplay
Branch(H) =H. (20)
Bitgraph Construction. Given the above isomorphism
between a bitgraph Hand undirected graph G={V,E},
{fi:Gi→Hi|GΓ=/uniontext
i(Gi),H=/uniontext
i(Hi)}, we construct
such a bitgraph by constructing a partition rule Γon its
isomorphic graph Gand components of each branch Hi.
The partitioning function Γoperates on Gas follows:
When traversing vertices according to their order, the func-
tion evaluates whether an edge exists between a vertex vi
and its next vertex vi+1(i.e., adjacent in order). If no
edge connects viandvi+1, then the vertex preceding vi+1
(denoted as vj) serves as a split vertex that generates a sub-
graph. This procedure is recursively applied to each resulting
subgraph (replacing the original graph Gwith the subgraph
in the rule) until no vertices violate the connectivity rule.
Note that within each subgraph, the partitioning rule remains
consistent, but the vertex ordering refers to the sequence
within the subgraph rather than in the original graph.
The composition of each branch is an ordered set Hi=
(Vi,seq(Vi),postd(Vi,V),parb(Vi))consisting of:
•Vi, a set of vertices that forms a subset of V;
•seq(Vi), a set containing the sequence of vertices in
Hi;
•postd(Vi,V), a set containing post-positive degree
of each vertex in Viwith respect to the traversal
sequence in V, deﬁned as
postd(Vi,V) ={postd(y)|y=x,x∈Vi,y∈V}.
(21)
•parb(Vi), the set of branches in which a vertex
fromViserves as the split vertex that creates a new
branch.
Demonstrative Example. To illustrate the construction,
we present an example demonstrating how an undirected
graphGis partitioned and how its corresponding bitgraph
His calculated. Consider an undirected graph Gwith six
vertices as shown in Fig 4, where the alphabetic order (i.e.,
a,b,c,d, ...) represents the vertex ordering in G. According
to the partitioning function Γ, it can be observed that vertex
cand its next vertex dare not adjacent in G’s vertex
ordering. Therefore, the vertex preceding d, namely vertex
a, serves as a split vertex. This creates a subgraph G2that
follows the edge connection from atod.
Let’s examine this from a panoramic time-sequential per-
spective. At this moment, vertices a,b,ccomprise subgraph
G1, while vertices a,dform subgraph G2. When partition
Γis applied recursively and independently to the subgraphs,
8

we consider the established ordering within each subgraph,
whereG1contains vertices a,b,cin positions 0th,1st,2nd
respectively, while H2contains vertices a,din positions
0thand1strespectively. Within G1, since vertex eis the
next vertex in sequence adjacent to cand they share a
connecting edge, the incorporation of einto the subgraph
extendsG1rather than creating a new subgraph. Likewise,
the addition of vertex ftoG2extendsG2without creating
any new subgraph. Consequently, Gis divided into exactly
two subgraphs.
In calculating the branches of Hfrom the two subgraphs
ofG, we follow the principle that G1yieldsH1andG2
yieldsH2(as deﬁned in Def. 5.2.2). To construct branches
ofH, such as branch H1, we record triplet information
from its isomorphic subgraph G1. For instance, vertex a,
being the 0thenter vertex in H1with one incident edge
(connecting to its postpositive vertices) and creating a new
branchH2, is recorded as entry (a,0,1,H2)whereaalready
indicates its position in graph G’s ordered sequence; For
vertexb, which is the 1stvertex in H1and connects to
two post-positive vertices (i.e., c,e) without producing a
new branch, the recorded entry is (b,1,2,Ø). In the same
manner, other vertices are converted to their triplets in H1
andH2. In particular, when vertex gforms connections
with the tail vertexes (i.e., e,f) in both subgraphs G1
andG2simultaneously, it is recorded in both H1andH2.
Examination of Fig 4 also reveals that when all branches in
Hare combined, they uniquely determine the reconstruction
of the isomorphic graph G.
Functionalities on Bitgraph Construction. In the fol-
lowing text, we present algorithms for core bitgraph op-
erations: Bitgraph.Insert andBitgraph.Search . These algo-
rithms demonstrate how partition rules enable bitgraph rep-
resentation of vertex insertion into a graph and how nearest
neighbors are searched. We provide high-level intuitions for
each algorithm design and visualize the algorithms using
extreme cases where vertices are added/searched across
bitgraph branches for easier understanding.
The insertion algorithm’s goal is to place a new vertex
qin branch Hiwith pre-positive vertex set Wi, as shown
inBitgraph.Insert (Alg 5.2.1). This process operationalizes
+a, 0, 1,  !
b, 1, 2
c, 2, 1
e, 3, 0"#: "!: ":
 #:
a, 0, 1
d, 1, 1
f, 2, 0 !:a
bd
c
ef$$a
b
ce+a
d
f
 :
a, 0, 1
b, 1, 2
c, 2, 1
e, 3, 1a, 0, 1
d, 1, 1
f, 2, 1
Figure 4: A bigraph construction examplepartition rule Γto decide vertex position in bitgraph, which
depends on Wicharacteristics and whether a continuously
adjacent vertex subset exists when traversing backwards. If
found, vertices in this subset won’t initiate new branches,
but remaining vertices outside the subset will each create
new branches. In the absence of such subsets, each vertex
generates a new branch. The input Wiis commonly derived
from the neighbor search results for vertex qbefore its
insertion into Hi; we omit this process.
The search algorithm follows the basic logic of the
original search in HNSW to replicate the searching walks
for ﬁnding nearest neighbors of a vertex. However, paral-
lel branches with cross-entering vertices make the original
searching walks extremely difﬁcult to maintain consistency
when searching over bitgraph. We begin with the HNSW
search intuition and then show how to preserve the searching
logic with minimal modiﬁcations, tracing the same search-
ing walks to hold result consistency.
During traversal, HNSW search maintains two dynamic
queues while traversing vertices: C, sequentially storing dis-
tinct vertices checked during search walks, and W, contain-
ing identiﬁed nearest neighbors. The path of search walks
is determined by evaluating neighboring vertices to navigate
deeper into the graph. All distance comparisons utilize these
queues, with Cconsistently providing the vertex currently
nearest to query vertex q, andWcontributing the furthest
vertex within query-range threshold θ; for example, vertices
candfrespectively. Each comparison evaluates whether a
new nearest neighbor exists by comparing distances (c,q)
and(f,q)to update W. The search process terminates when,
after ﬁnding sufﬁcient nearest neighbors in W, the nearest
vertex in Cis further than the furthest vertex in W.
To eliminate the uncertainty in search walk progres-
sion caused by cross-entering vertices, we modiﬁes two
places, and the complete algorithm is in Bitgraph.Search
(Alg 5.2.3): The ﬁrst is the Bitgraph.HoneycombNeighbors
algorithm (Alg 5.2.2), rewriting the vertex neighbor identi-
ﬁcation to governs which vertex enters the Cqueue next.
Through this algorithm, we identify all vertices honeycomb-
adjacent to the currently examined vertex c, regardless of
whether they reside in the current branch or in parallel
branches (the latter scenario occurring when cfunctions
as a split vertex). This method alone proves insufﬁcient
when search progress reaches a branch endpoint without
triggering the termination condition, with searching still
active. To address this limitation, our second modiﬁcation
introduces an at-hand-detour function (colored blue, Lines
10-14) that reverts to the previously nearest vertex in queue
Cwhen the examined vertex cis determined to be at its
branch tail. This traceback approach establishes a speciﬁc
pathway to vertices that should maintain connectivity in the
original graph structure but have been segmented across
different branches. In speciﬁc, we explain how Algorithm
5.2.2 identiﬁes honeycomb-adjacent neighbors of its input
vertexc. On branch Hi, the honeycomb around vertex c
consists of post-positive vertices (recorded in postd) and a
single pre-positive vertex (recorded in seq). Besides its role
within the current branch, we must consider cases where
9

vertexcfunctions as a split vertex connecting to other
forked branches (recorded in parb). In these cases, the
honeycomb involves only a single post-positive vertex that
follows the current split vertex in sequence. Here, one step
to the next vertex is enough to keep the search moving
forward. A search trajectory example through honeycomb
neighbors on a branch is visualized in Figure 5, where
the numerical values are the sequence ordering of vertices
within the branch.
Algorithm 5.2.2: Bitgraph.HoneycombNeighbors (c,Hi)
Input : a vertex c, its located branch Hi
Output : neighbors of vertex c
Finding Neighbors across Branches —
1:Neighbors←Ø// set of neighbors of vertex c
2:cparb←get the parallel branches set of cinHi
3:// ifcis head vertex in Hi,
seqin line 5 starts from c.seq+1toc.postd
4:// ifcis tail vertex in Hi,
seqin line 5 is assigned c.seq−1
5:forseq←c.seq−1,c.seq+1...c.seq+c.postdin
Hido
6:(v,locHi)←get theseqthvertex in Hi
7:Neighbors =Neighbors/uniontext(v,locHi)
8:end for
9:foreach branch Hjincparbdo
10: // get the next vertex of head vertex in Hj
11:(v,locHj)←get the1stvertex of Hj
12:Neighbors =Neighbors/uniontext(v,locHj)
13:end for
14:returnNeighbors
We also provide illustrative examples to clarify the func-
tionalities.
Example with Insert/Search Functionality. Figure 6
shows how values change when vertex gis inserted into
branches H′
1,H′
3,H′
4and vertex hinto branch H′
2, where
2
3
4
5
612
3
4
5
61
2
3
4
5
61
2
3
4
5
612
3
4
5
61
Figure 5: A search trajectory through honeycomb neighbors
on a branchAlgorithm 5.2.1: Bitgraph.Insert(q;H,(Wi,locHi))
Input : a new vertex q; a bitgraph H, andq’s pre-positive
vertices set Wiwith an ordered sequence, its branch
locationHi(i.e.,Wi⊆Hi).
Output : update related entries of bitgraph Hafter
inserting q(i.e., add connections from neighbors
(Wi,locHi)toqin a given bitgraph.)
Insert Procedure on Hi—
1:S←Ø// set of new split vertices
2:{Hj:Hj←Ø}// set of new branches produced from
the split vertices in Hi
3:T←get set of vertices having continuous sequences
fromWiin a back-to-front order
4:t←get last element from T
5:h←get last element from Hi
6:ift=hthen
7: for each vertex v∈Tdo
8:thisEntry .vpostd←thisEntry .vpostd+1
// update v’s entry in Hi
9: end for
10:(Entry)qseq,qpostd,qparb←|Hi|,0,Ø
11:Hi←Hi/uniontextthisEntry .q// addqto the tail of Hi
12: ifWi/T/n⌉}ationslash= Ø then
13:S←S/uniontextWi/T
14: end if
15:end if
16:ift/n⌉}ationslash=hthen
17:S←S/uniontextWi
18:end if
19:foreach vertex v∈Sdo
20:Hj←instantiate a new branch
21:thisEntry .vparb←
thisEntry .vparb/uniontextlocHj// record parallel
branches location of vinHi
22:(Entry)vseq,vpostd,vparb←0,1,Ø
23:Hj←Hj/uniontextthisEntry .v
24:(Entry)qseq,qpostd,qparb←1,0,Ø
25:Hj←Hj/uniontextthisEntry .q
26:end for
detailed insertion procedures is bypassed. We focus on the
search process for a query vertex (represented by a green tri-
angle). In the original graph structure, the search walk would
proceed by entering vertex a, analyzing a’s neighbors, then
examining g’s neighbors to identify nearest neighbors (likely
gande). However, in the bitgraph context, standard search
protocols cannot establish a connection from gtoein
this situation. Here, the at-hand-detour function provides
the solution by backtracking to vertex band examining its
neighborhood to successfully reach vertex e.
CORRECTNESS . The walk isomorphism deﬁnition is de-
rived from bitgraph isomorphism, providing the theoretical
basis to validate that substituting bitgraphs in the search for
graphs achieves correctness (i.e., identical search results)
with acceptable deviation.
10

Algorithm 5.2.3: Bitgraph.Search(q,θ;H,(ev,locHa))
Input : a query q, maximum nearest neighbor number θ; a
bitgraphH, with an enter vertex evand its branch
locationlocHa(i.e., it is not necessarily the head
entry).
Output : nearest neighbor vertices to a query q.
Search Procedure on H—
1:E←(ev,locHa)// set of evaluated vertices and
their branch locations
2:C←(ev,locHa)// queue of candidates and their
branch locations
3:W←(ev,locHa)// queue of found nearest
neighbors
4:while|C|>0do
5:(c,locHi)←extract nearest element from C
6:f←get furthest element from Wtoq
7: ifDistance(c,q)>Distance(f,q)then
8: break
9: end if
10: //at-hand-detour rule
11: whilecpostd= 0 do
12: remove nearest element from C
13:(c,locHi)←extract nearest element from C
14: end while
15: foreach(v,locHj)∈
Bitgraph.HoneyCombNeighbors (c,locHi)do
16: if(v,locHj)/∈Ethen
17: E←E∪(v,locHj)
18: f←get furthest element from W
19: ifDistance(v,q)<Distance(f,q)
20: C←C∪(v,locHj)
21: W←W∪(v,locHj)
22: if|W|> θ then
23: remove furthest element from W
24: end if
25: end if
26: end for
27:end while
28:returnW
Deduction 5.2.3 (Walk Isomorphism ).For any undirected
graphGand its isomorphic bitgraph H, there always exists
at least one isomorphic walk in Hthat covers any given
walk inG, regardless of which vertex in His selected to
split branches. That is, there exists
f: walk(G)→walk(H), (22)
such that the set of vertices in a walk of Gforms a subset
of the vertices in the corresponding walk of H.
COMPLEXITY ANALYSIS . Following Sec. 5.1, we quan-
tify bitgraph sharing complexity by considering the min-
imum requirement for collective computation: all parties
must together hold shares of at least all branch’s vertices.
The complexity metric becomes the sum of entries across all
branches in a bitgraph, which characterizes the complexity
for both sharing a bitgraph and searching distributed shares.a
b d
ce
a, 0, 1,  !", #"
b, 1, 2,  $"
c, 2, 1
e, 3, 0+1
g, 4, 0 %&": %!":%"=%.Insert ( ,!"#;$,!%#; ,!&#; ,!'#)
*"#:
a, 0, 1
d, 1, 1+1
f, 2, 0+1
h, 3, 0 *%#:g
f
b, 0, 1
g, 1, 0*&#:!&#:
+
+ +h
b
cea
g bga
g+d
fha
+!'#:
a, 0, 1
g, 1, 0*'#:
+:-  
Figure 6: A bigraph insert/search example
Deduction 5.2.4 .For any bitgraph Hand undirected graph
G={V,E},{fi:Gi→Hi|GΓ=/uniontext
i(Gi),H=/uniontext
i(Hi)},
assuming v∈Rdfor any vertex v∈V, when sharing H
amongnparties using SS.Sharet
nandiffthe vertices set of
Hbeing shared, the computational complexity is:
O(SS.Sharet
n(H)) =O(SS.Sharet
n(¯V))
=d(|V|+|H|)·n·O(pt)
∼=d|V|·n·O(pt),(23)
where|H|is the number of branches, corresponding to
the count of split vertices; and ¯VandVare vertices
(including repeated vertices) contained in Hand distinct
vertices respectively. In practical graph applications, this
value is typically treated as a constant in the structure, with
components like edges capturing meaningful relationships
such as distances.
Deduction 5.2.5 .Letcdenote the number of vertices in
a searching walk over any undirected graph G={V,E}
with its isomorphic bitgraph H,{fi:Gi→Hi|GΓ=/uniontext
i(Gi),H=/uniontext
i(Hi)}, andv∈Rdfor any vertex v∈V.
Then, when sharing Hamongnparties using SS.Sharet
n
and iffits vertices set ¯Vbeing shared, the computational
complexity of a search operation over H’s shares is:
O(SearchonShares({u,Hu}u∈U))
=O(SearchonShares({u,¯Vu}u∈U))
=d(c+a)·n·O(pt)·O(AC)
∼=dc·n·O(pt)·O(AC),(24)
whereais the number of additional vertices introduced by
theat-hand-detour function.
11

6. A SP-A2NN Search Scheme Based on
HNSW
6.1. Construction
Prior to presenting the complete algorithms, this sec-
tion covers two key aspects: the storage structure detailing
stored index components, data repository, and their inter-
connections; and the search/update intuition for operating
shared bitgraphs that are organized in the HNSW-indexing
pattern, which explains the contents of bitgraph storage
units that enable direct query token matching for SP-A2NN
searches. We then brieﬂy discuss potential security enhance-
ments to this work. Finally, we analyze the scheme in terms
of complexity, parameter-related correctness, and security
guarantees.
Structure – Encrypted Database C-EDB . The SP-
A2NN search scheme adheres to the conceptual structure
deﬁned in Sec 4.1, Formula (3), organizing its database
into separate index and data repositories. Like HNSW’s
multilayer graph index, which arranges vertices in layers
of increasing density from sparse at the top to dense at
the bottom, the encrypted collaborative index in SP-A2NN
also consists of multiple hierarchical layers (i.e., C-EI-l).
Similarly, the bottom layer serves as the data repository
containing vectors. The fundamental difference lies in that
each layer uses an encrypted bitgraph (i.e., C-EH-l) in a
shared pattern instead of an undirected graph employed in
HNSW. We have
C-EDB=C-EI+C-ED
=L/summationdisplay
1C-EI-l+C-ED
=L/summationdisplay
1C-EH-l+C-ED.(25)
Search/Update Intuition – From Bitgraph to Shared
Bitgraph in HNSW Organization. Searching in SP-A2NN’s
collaborative encrypted index (i.e., C-EI) proceeds layer-
wise from top to bottom in the same manner as HNSW,
ﬁnding query vector nearest neighbors in each layer until
retrieving all bottom-layer data vectors. In a similar manner,
insertion of a new element into C-EIrelies on the search
algorithm to preliminarily identify nearest neighbors from
top to bottom, establishing the optimal placement for the
new element. The overall search/insert architecture is shown
in Algorithm B.2 &B.5.
Let us now focus on searching within each layer where
a bitgraph resides. The problem becomes clear given that
Sec 5 already addresses searching and inserting vertices in
bitgraphs. The operation of searching or inserting vertices
within a layer (e.g, C-EI-l) is equivalent to operating
on a shared bitgraph (e.g., C-EH-l) where vertices are
distributed among participating parties. This architectural
choice emphasizes data protection at the expense of leaving
graph connectivity unencrypted in terms of access patterns
at every party’s view, where connectivity means partialedges. Any party can easily recognize whether two ver-
tices are adjacent by observing their storage locations and
sequences, but cannot determine what the vertex content
actually represents.
In particular, search operations implement a shared ver-
sion that retains the core search logic from Bitgraph.Search
in Sec 5, while this algorithm itself is an unshared, un-
protected plaintext version for identifying neighbors of a
query vector within the bitgraph network. Direct query token
matching in the shared scheme is implemented via share-
based calculations given the distributed nature of vertices
across parties. The algorithm of SearchLayer is detailed
in Algorithm B.4. Insert operations InsertLayer (Alg B.1)
follow a similar approach, mirroring the Bitgraph.Insert
algorithm.
Optimization, Detail and Discussion. During search
execution, a small adjustment for efﬁciency is made to the
wayBitgraph.Search (Alg 5.2.3) tracks evaluated vertices.
Rather than maintaining the set and determining vertex
membership status, this is replaced with bit-based infor-
mation recording. That is, a standard bitgraph vertex quad
(v,vseq,vpostd,vparb)is extended by adding a visit
bitvethat marks whether a vertex has been accessed
during search, preventing repeated vertex evaluations. The
complete algorithms are provided in Appendix B. (We omit
the SetUp algorithms for parties agreeing on keys and the
state of each execution.)
Note that this version only covers insert scenarios,
with delete methods left out of scope. SP-A2NN’s local
database setting makes privacy concerns from dynamic up-
dates, such as forward/backward privacy , acceptable. Con-
sequently, ﬂexible deletion processes are possible, such as
having all parties collectively mark a unit as deleted. How-
ever, when considering whether updating a unit belonging to
one party might leak information about that unit’s content
to other parties, this introduces a distinct issue requiring
further study. Beyond the current scope, adding consistency
veriﬁcation mechanisms for this work’s solutions can extend
the scheme to provide protection against active adversaries.
6.2. Analysis
COMPLEXITY ANALYSIS . The complexity for a SP-
A2NN search can be simpliﬁed to the complexity of search-
ing on shares of a bitgraph (Deduction 5.2.5) by the follow-
ing reductions.
Deduction 6.2.1 .According to the component structure in
Formula (25) of the multilayer C-EDB , the computational
complexity of SP-A2NN search can be decomposed into
the complexity sum of searching each individual layer.
Generally, we focus on the lthlayer’s encrypted bitgraph
C-EH-lin a shared pattern, which is constrained to sharing
only vertices and thus adheres to the complexity pattern of
SearchonShares .
Let¯Vrepresent the set of vertices actually traversed in
a search walk over C-EH-l. This set’s size is formulated
using three parameters: cis the vertex count in a search walk
12

Setup(1λ,σ)
1:sdi$←{0,1}λallocate list L
2:Initiate Counter σ:c←0
3:Ki←F1(sd1,c)
4:AddKiinto listL(in lex order)
5:OutputK= (Ki,σ)Insert(K,σ,q;C-EDB)
1:(partyu){qu}U←Enc(q,K1)
2:Set
C-EI←
C-EI.Add(I-hnsw -bitg:{loc(qu)}U,
qseq,qpostd,qparb;qe);
C-ED←C-ED.Add({qu}U,
qseq,qpostd,qparb;qe);
σ:c++
3:Output
C-EDB= (C-EI,C-ED,σ)Search(K,σ,q;C-EDB)
1:(partyv){qu}U←Enc(q,K2)
2:On input
{q}←v:Enc(q)
C-EDB= (C-EI,C-ED,σ)
3:Forc= 0 untilSP-A2NN(HNSW-Bitgraph
index) return⊥,
{vu}U←
SP-A2NN(C-EI;C-ED,{q}U,F2)
4:v←Dec({v}U,K)
5:Outputv
Figure 7: Real Scheme ΠI-hnsw -bitg
SS
over the original HNSW graph that is converted from its
isomorphic bitgraph ( H-l), which quantiﬁes the complexity
of standard HNSW search; ais the count of additional
vertices triggered by at-hand-detour functions; and ois
the number of deviated vertices incurred by honeycomb-
neighbors tracing walks compared to the original HNSW
walks. Thus, the computational complexity of SP-A2NN
search compared to reference HNSW search is:
O(SP-A2NN.Search({q}U;C-EDB))
=O(SP-A2NN.SearchLayer({q}U;L/summationdisplay
1C-EH-l,C-ED))
= (L+1)·O(SearchonShares({u,¯Vu}u∈U)))
= (L+1)·d(c+a+o)·n·O(pt)·O(AC)
∼=(L+1)·dc·n·O(pt)·O(AC)
=O(HNSW.Search(q;DB))·n·O(pt)·O(AC)
(26)
Observe that SP-A2NN search introduces only the ad-
ditional overhead of computing distance comparisons via
arithmetic circuits, relative to standard HNSW search on
unencrypted data.
CORRECTNESS AND SECURITY ANALYSIS . We validate
the correctness and security of a SP-2ANN search scheme
via a real instantiated construction as follows.
Tunable Parameters Impact on Correctness. In the bit-
graph structure, the same vertex may be located in multi-
ple branches, resulting in duplicate vertices from different
branches potentially being recorded in the queue W, which
dynamically maintains search results during the search op-
eration. Additionally, the search results contain additional
deviation vertices compared to the original HNSW results,
since the actual search walks traverse (a+o)additional
vertices. Despite occasionally missing some vertices relative
to thecvertices in the original search walks, the scheme
design guarantees tracing the original paths as faithfully
as possible. We view this deviation as quite acceptable
since, in real-world scenarios, the search results in queue W
act as candidate sets, with ﬁnal elements selected through
closest-ﬁrst or heuristic selection methods. Thus, in real
applications, the queue size |W|(i.e.,θ) can be tuned toa relatively large range to avoid having repeated vertex
positions negatively impact the expected search results. The
impact of this part is considered and incorporated when
validating correctness of Theorem 6.2.2.
Real SP-A2NN-Instantiated Construction. Let a
(t,n)-threshold secret sharing conﬁguration SSserve as an
encryption scheme of (Enc,Dec), andI-hnsw -bitg be the
bitgraph-based HNSW index to organize C-EDB .F1and
F2are the same as in ΠSS. We have our real construction
ΠI-hnsw -bitg
SSin Fig 7.
Theorem 6.2.2 .A real scheme ΠI-hnsw -bitg
SS is∆(ρ)-correct
iffthe reduction from ΠI-hnsw -bitg
SStoΠI-hnsw
SS w.r.t correct-
ness is∆(ΠI-hnsw -bitg
SS ,ΠI-hnsw
SS)-correct.
The proof for Theorem 6.2.2 is omitted, while the impact
from the tunable parameters is used to measure ∆(ρ), and
we consider this impact on deviation allowable.
Theorem 6.2.3 .A real scheme ΠI-hnsw -bitg
SS isL(ǫ)-secure
iffthe reduction from ΠI-hnsw -bitg
SStoΠI-hnsw
SS w.r.t security
andw.r.t leakage isL(ΠI-hnsw -bitg
SS ,ΠI-hnsw
SS)-secure.
The proof for Theorem 6.2.3 is in Appendix A.2.3.
13

References
[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Pe troni, Vladimir
Karpukhin, Naman Goyal, Heinrich K¨ uttler, Mike Lewis, Wen -tau
Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe Kiela. R etrieval-
augmented generation for knowledge-intensive NLP tasks. I nAdvances
in Neural Information Processing Systems 33: Annual Confer ence on
Neural Information Processing Systems 2020, NeurIPS 2020, Decem-
ber 6-12, 2020, virtual , 2020.
[2] Yury A. Malkov and Dmitry A. Yashunin. Efﬁcient and robus t
approximate nearest neighbor search using hierarchical na vigable small
world graphs. IEEE Trans. Pattern Anal. Mach. Intell. , 42(4):824–836,
2020.
[3] Ueli M. Maurer. Secure multi-party computation made sim ple.Discret.
Appl. Math. , 154(2):370–381, 2006.
[4] Dawn Xiaodong Song, David A. Wagner, and Adrian Perrig. P ractical
techniques for searches on encrypted data. In 2000 IEEE Symposium
on Security and Privacy, Berkeley, California, USA, May 14- 17, 2000 ,
pages 44–55. IEEE Computer Society, 2000.
[5] Sacha Servan-Schreiber, Simon Langowski, and Srinivas Devadas.
Private approximate nearest neighbor search with sublinea r commu-
nication. In 43rd IEEE Symposium on Security and Privacy, SP 2022,
San Francisco, CA, USA, May 22-26, 2022 , pages 911–929. IEEE,
2022.
[6] Wikipedia. Skip list. https://en.wikipedia.org/wiki /Skip list.
[7] Adi Shamir. How to share a secret. Commun. ACM , 22(11):612–613,
1979.
[8] Rapha¨ el Bost, Brice Minaud, and Olga Ohrimenko. Forwar d and back-
ward private searchable encryption from constrained crypt ographic
primitives. In Proceedings of the 2017 ACM SIGSAC Conference on
Computer and Communications Security, CCS 2017, Dallas, TX , USA,
October 30 - November 03, 2017 , pages 1465–1482. ACM, 2017.
[9] Wikipedia. Graph theory. https://en.wikipedia.org/w iki/Graph theory.Appendix A.
A Leakage-Guessing Proof System for Privacy
Analysis
A.1. Logical Reduction Framework
A.2. Proof – Reduction from ΠtoΠBas
A.2.1. Proof for Theorem 4.1.3 – fromΠBastoSS.This
part is assumed validated.
A.2.2. Proof for Theorem 4.1.4 – fromΠMtoΠBas.
SETTINGS : We introduce a chess-play game aided by
an oracle deﬁnition to complete the secruity proof.
Deﬁnition A.2.1 (OracleF).The oracleFis deﬁned as:
F: (x,f)→{f(x,y) :y∈Connected (x)} (27)
wherex∈Xis the query element, the oracle returns the set
of function values over all elements connected to x, andf:
X×X→{0,1}is the underlying link existence function.
Proposition A.2.2 (Chess GameC′).LetCbe a oracle-
F-aided interaction-based protocol between simulators S1,
S2, a guard Gand an adversary Ato identify the related
information between two ciphers encrypted by two different
schemes from an identical message.
Init:S2runsΠI-hnsw
SS.Init,ΠI-hnsw
SS.Insert in Fig. 2 and
S1runsΠSS.Init,ΠSS.Insert in Fig. 1, both in a completed
execution state.
Chess-Play: [A QUERYS]S1takes any cipher
element equeried fromAand outputs a result
(e,C-ED;C-EI1)toA. Following the same pattern, S2
outputs(e,C-ED;C-EI2)toA.
[SREQUEST G] BothS1,S2backup their outputs to A
and hand them over to G. ForS1,Gtakes over S1’sΠSS.Init
to decode efromC-ED, denoted as a message me, and
reversesC-EI1toC-I1. ThenGinvokes oracle-Fwith
input(me,C-I1:∪f)to extract the complete adjacency
information linking meto all related elements, yielding a
set of elements{me1}. (Here,fcontains the position and
pointer structure of each element in C-I1, while∪fcaptures
all structures in C-I1.)Gdecodes this set to {e1}via keys in
ΠSS.Initand outputs a pair. The pair (e,{e1};fc)represents
that there exists a connection from element eto elements
in the set{e1}withinC-EI1, where the connection is
formulated with fc:={loc(e1)}according to the execution
in Fig 1 (Line 2).
Analogously, for S2,Goutputs(e,{e2};fc)that repre-
sents there exists a connection from eto{e2}inC-EI2,
where the connection is formulated with fc:={I-hnsw:
loc(e2)}according to the execution in Fig 2 (Line 2).
[GRESPONSE S]Goutputs the response to S.
[SANSWERA]Sanswers the outputs to A.
Bonus (Leakage): WhatAobtains deﬁnes the bonus
he wins in the game.
14

 Instance !of 
Real Problem "
Instance of 
#$%&&'& 
“break” 
( )Solution to !
Let’s play two chess games  !
 
 (for "!
"") and  
 
 (for "(#)
#)) to guess leakage: 
" $%
A simulator, a guard and an adversary Reduction "!
Reduction 
"(#) Instance of 
Scheme &'*+, Reduction "!!
Instance of 
&-+. 
Figure 8: A high-level overview of reduction framework for privacy analysis
SECURITY PROOF : The proof follows a two-step re-
duction pattern: ﬁrst, we verify that the leakage reduction
forL(ΠI-hnsw
SS,ΠSS)-security holds, then we show that
complete equivalence reduces from ΠI-hnsw
SS toΠSS(i.e.,
security of all practically meaningful encrypted data, e.g.,
C-ED).
Step-1: Leakage Reduction. To extract the leakage in-
curred when reducing from ΠI-hnsw
SS toΠSS, we begin with
several deduction sketches as:
Deduction A.2.1 .The leakage between (ΠI-hnsw
SS,ΠSS)
reduces to data of C-EDB ’s leakage between
(ΠI-hnsw
SS,ΠSS).
Deduction A.2.2 .TheC-EDB ’s leakage between
(ΠI-hnsw
SS ,ΠSS) reduces to leakage between ( ΠI-hnsw
SS ’s
C-EI,ΠSS’sC-EI)ifcomplete equivalence holds in
(ΠI-hnsw
SS ’sC-ED,ΠSS’sC-ED).
Deduction A.2.3 .The leakage between ( ΠI-hnsw
SS ’sC-EI,
ΠSS’sC-EI) equals the total leakage of all elements in
C-ED that occur in both C-EIs.
Deduction A.2.4 .The leakage of any element in C-ED
that occur in both C-EIs can be calculated via a privacy-
guessing game.
To validate Deduction A.2.4, we construct a simulator
S′
0to simulate an individual party’s view (e.g., party i),
which invokes a leakage-guessing game called chess game
C′(Proposition A.2.2) for any element eofC-ED as
inputs. From a high-level respective, party i’s view captures
the highest level of privilege, not only allowing complete
observation of ciphers ﬂowing both within/through party i’s
territory, but also invoking Enc/Dec oracle for reversing any
cipher to its message. This privilege is transferred to a guard
roleGofC′. We haveLi(ΠI-hnsw
SS,ΠSS)
=Li(ΠI-hnsw
SS: (C-EI,C-ED),ΠSS: (C-EI,C-ED))
=Li(ΠI-hnsw
SS:C-EI,ΠSS:C-EI;C-ED)
=/summationdisplay
eLi(C-EI2,C-EI1;C-ED,e)fore∈C-ED
=/summationdisplay
eS′
0.C′(C-EI2,C-EI1;C-ED,e)fore∈C-ED
(28)
where the equalities are justiﬁed as follows: the 1st by
deduction of A.2.1, the 2nd by by deduction of A.2.2, the
3th by by deduction of A.2.3, and the 4th by deduction of
A.2.4 and the followed deduction. (For e’s format, taking
Fig. 1 as an example, eis a share of{qi}Uhold by party
i.))
Step-2: complete C-ED equivalence. It can be observed
that the validation of Deduction A.2.2 relies on complete
equivalence on ( ΠI-hnsw
SS ’sC-ED,ΠSS’sC-ED). Exam-
ining the execution of inserting any element into C-ED is
identical in both Fig 1 and Fig 2, this complete equivalence
holds.
CORRECTNESS BASELINE . As in formula (7) (Def
4.1.1), the correctness of mirror construction ΠI-hnsw
SS is the
reference baseline under the problem Σ.
CONCLUSION . Back to Theorem 4.1.4, we have its L-
15

secure claim on (ΠM,ΠBas)is hold as:
L(ΠM,ΠBas)
=/summationdisplay
e(e,{e2};I-hnsw:{loc(e2)})def
−(e,{e1};{loc(e1)})
=/summationdisplay
e(e,{e2};I-hnsw:{loc(e2)})
(29)
where the 2nd equality is justiﬁed by the independence
of storage locations due to no indexing occurring in
e,{e1};{loc(e1)}.
The correctness claim on (ΠM,ΠBas)holds by referring
to above correctness baseline.
A.2.3. Proof for Theorem 4.1.3 – fromΠtoΠM.
SETTINGS : A chess-play game Canalogous to that in
Appendix A.2.2 is deﬁned, where the difference lies in:
givenA’s querye, during Init,S2runsΠI-hnsw -bitg
SS’sInit
andInsert algorithms in Fig. 7, and S1runsΠI-hnsw
SS ’s
Init,Insert in Fig. 2, with both achieving a completed
execution state. The ﬁnal output of the game Cis
(e,{e2};fc2)and(e,{e1};fc1), wherefc2is
{I-hnsw -bitg:loc(e2),e2seq,e2postd,e2parb;e2e}
andfc1is{I-hnsw:loc(e1)}.
SECURITY PROOF : The proof also uses a two-step re-
duction pattern as in Appendix A.2.2: L-security reduction
and complete equivalence reduction (exclusively in terms of
C-ED) fromΠI-hnsw -bitg
SS toΠI-hnsw
SS .
Step-1: Leakage Reduction. Adopting the same frame-
work for reductions (Apx A.2.2 Step-1), we let S′
0simulate
the view for an individual party iand invokeC, yielding
Li(ΠI-hnsw -bitg
SS,ΠI-hnsw
SS)
=Li(ΠI-hnsw -bitg
SS : (C-EI,C-ED),ΠI-hnsw
SS: (C-EI,C-ED))
=Li(ΠI-hnsw -bitg
SS : (C-EI),ΠI-hnsw
SS: (C-EI: (C-EI);C-ED)
=/summationdisplay
eLi(C-EI2,C-EI1;C-ED,e)fore∈C-ED
=/summationdisplay
eS0.C(C-EI2,C-EI1;C-ED,e)fore∈C-ED
(30)
Step-2: complete C-ED equivalence. The equivalence
on (ΠI-hnsw
SS ’sC-ED,ΠSS’sC-ED) is validated since the
identical execution of inserting any element into C-ED
occurs in both Fig 7 and Fig 2.
Back to Theorem 6.2.3, we have its L-secure claim on
(Π,ΠM)is hold as:
L(Π,ΠM)
=/summationdisplay
e(e,{e2};fc2)def
−(e,{e1};{I-hnsw:{loc(e2)})
=/summationdisplay
e(e,{e2};fc2)
(31)wherefc2is
{I-hnsw -bitg:loc(e2),e2seq,e2postd,e2parb;e2e}.
Drawing upon the deﬁnition of privacy triplet (Def
4.3.1), for any individual, randomly chosen data element e
where its message is assumed to be learned, we can calculate
the leakage exposure L(ǫ)incurred by eon the multilayer-
bitgraph organized C-EDB , speciﬁcally/summationtextL
1C-EH-l+
C-ED. Through a I-III trajectory, we have
LI-hnsw -bitg
I(e) =(L+1)×(1+postd+parb)
C-ED.
where the ﬁrst 1counts for the prior one of ein sequence
of a layer.
LI-hnsw -bitg
II ({e2}) =/summationtext
|{e2}|(L+1)×(1+postd+parb)
C-ED
Having traversed interfaces I, II, we can determine the
number of elements that exhibit connections with e.To
further measure this connection, based on knowledge of
L(Π,ΠM)and public parameters, we can state
LD
III(e,{e2}) =
{Distance(θ)|/summationtext
|{e2}|+1(L+1)×(1+postd+parb)}
C-ED
whereDistance(θ)measures similarity distance of two el-
ements (i.e., vectors) given that we treat the query range
threshold θis public (although we examine θas the maxi-
mum number of neighbors a query contains in our work).
CONCLUSION . Back to Theorem 6.2.3, we have its L(ǫ)-
secure claim on (Π,ΠM)is hold as:
L(ǫ) =LD
III(e,{e2}) =
{Distance(θ)|/summationtext
|{e2}|+1(L+1)×(1+postd+parb)}
C-ED.
Appendix B.
SP-A2NN Algorithms
16

Algorithm B.2: SP-A2NN.Insert(KSS,AC,σ,{q}U,l′;
C-EDB)
Input : a new vector{q}Usubmitted by party v, this new
element’s level l′;
C-EDB : multiple bitgraphs, and its an enter vector
{ev}Ushared from party u, which is located in
branchHaof top layer’s bitgraph (i.e., the Lthlayer).
(Locations of bitgraph, branches, and units are public
parameters.)
Output :
Insert Procedure —
1:({ev}U,locHa)←get enter vector for C-EDB
2:forl←L...l′+1do
3:{W}U←Search -Layer({q}U,θ= 1;
C-EI-l,({ev}U,locHa))
4:({ev}U,locHa)←get ﬁrst element from {W}U
5:end for
6:forl←l′...0do
7:{W}U←Search -Layer({q}U,θ;
C-EI-l,({ev}U,locHa))
8:(ev,locHa)←get ﬁrst element from {W}U
9:end forAlgorithm B.1: Insert -Layer({q}U;
C-EI-l,({Wi}U,locHi))
Clients Input : a new vector{q}Ushared from party v;
C-EI-l,({Wi}U,locHi): thelthlayer’s bitgraph, and
{q}U’s pre-positive vertices set {Wi}Uwith an
ordered sequence, its branch location locHi
(i.e.,{Wi}U⊆Hi).
Clients Output : update branch HiinC-EI-lafter
inserting q.
Insert Procedure on a Layer —
all parties inU:
1:S←Ø// set of new split vertices
2:{Hj:Hj←Ø}// set of new branches produced from
the split vertices in Hi
3:{T}U←agree on set of vertices having continuous
sequences from{Wi}Uin a back-to-front order
4:{t}U←get last element from {T}U
5:{h}U←get last element from Hi
6:ifAC.Evaluate({t}U={h}U)then
7: foreach vertex{v}U∈{T}Udo
8: (partyubroadcasts: )
thisUnit.vpostd←thisUnit.vpostd+1
9: end for
10: (partyvbroadcasts: )(Unit)
qseq,qpostd,qparb,qe←|Hi|,0,Ø,0
11:Hi←Hi/uniontextthisUnit.{q}U
12: if{Wi}U/{T}U/n⌉}ationslash= Ø then
13:S←S/uniontext{Wi}U/{T}U
14: end if
15:end if
16:if{t}U/n⌉}ationslash={h}Uthen
17:S←S/uniontext{Wi}U
18:end if
19:foreach vertex{v}U∈Sdo
20:Hj←instantiate a new branch
21:thisEntry .vparb←
thisEntry .vparb/uniontextlocHj// record parallel
branches location of {v}UinHi
22:(Entry)vseq,vpostd,vparb←0,1,Ø
23:Hj←Hj/uniontextthisEntry .{v}U
24:(Entry)qseq,qpostd,qparb←1,0,Ø
25:Hj←Hj/uniontextthisEntry .{q}U
26:end for
17

Algorithm B.3: Seach -Layer.HoneycombNeighbors ({c}U,Hi)
Input : a vertex{c}U, its located branch Hi
Output : neighbors of vertex {c}U
Finding Neighbors across Branches —
1:Neighbors←Ø// set of neighbors of vertex {c}U
2:cparb←get the parallel branches set of cinHi
3:// ifcis head vertex in Hi,
seqin line 5 starts from c.seq+1toc.postd
4:// ifcis tail vertex in Hi,
seqin line 5 is assigned c.seq−1
5:forseq←c.seq−1,c.seq+1...c.seq+c.postdin
Hido
6:({v}U,locHi)←get theseqthvertex in Hi
7:Neighbors =Neighbors/uniontext({v}U,locHi)
8:end for
9:foreach branch Hjincparbdo
10: // get the next vertex of head vertex in Hj
11:({v}U,locHj)←get the1stvertex of Hj
12:Neighbors =Neighbors/uniontext({v}U,locHj)
13:end for
14:returnNeighbors
Algorithm B.5: SP-A2NN.Search(KSS,AC,σ,{q}U,θ;
C-EDB)
Input : a query{q}Usubmitted by party v, maximum
nearest neighbor number θ;
C-EDB : multiple bitgraphs, and its an enter vector
{ev}Ushared from party u, which is located in
branchHaof top layer’s bitgraph (i.e., the Lthlayer).
(θand locations of bitgraph, branches, and units are
public parameters.)
Output :θnearest neighbor vectors to {q}U.
Search Procedure —
1:({ev}U,locHa)←get enter vector for C-EDB
2:forl←L...1do
3:{W}U←Search -Layer(
{q}U,θ= 1;C-EI-l,({ev}U,locHa))
4:({ev}U,locHa)←get ﬁrst element from {W}U
5:end for
6:{W}U←Search -Layer(
{q}U,θ;C-ED,({ev}U,locHa))//0thlayer
7:W←AC.Evaluate(SS.Recon({W}U))
8:returnWAlgorithm B.4: Search -Layer({q}U,θ;
C-EI-l,({ev}U,locHa))
Clients Input : a query{q}Usubmitted by party v,
maximum nearest neighbor number θ;
C-EI-l,({ev}U,locHa): a bitgraph of layer l, and
an enter vector{ev}Uwith its location locHa.
Clients Output : nearest neighbor vectors to {q}U.
Search Procedure on a Layer :
partyu:
1:{C}U←({ev}U,locHa)// queue of candidates and
their branch locations
2:{W}U←({ev}U,locHa)// queue of found nearest
neighbors
all parties inU:
3:while|C|>0do
4:({c}U,locHi)←extract ﬁrst element from {C}U
5:{f}U←get last element from {W}U
6: ifAC.Distance({c}U,{q}U)>
AC.Distance({f}U,{q}U)then
7: break
8: end if
9: whileAC.Evaluate(SS.Recon({cpostd}U) = 0)
do
10: remove ﬁrst element from {C}U
11:({c}U,locHi)←extract ﬁrst element from
{C}U
12: end while
13: for each({v}U,locHj)∈
Search -Layer.HoneycombNeighbors ({c}U,Hi)do
14: (partyubroadcasts: )thisUnit.ve←1// record
this vertex as ‘ evaluated ’
15:{f}U←get last element from {W}U
16: ifAC.Distance({v}U,{q}U)<
AC.Distance({f}U,{q}U)then
17:{C}U←{C}U/uniontext({v}U,locHj)
18:{W}U←{W}U/uniontext({v}U,locHj)
19: ifAC.Agree(|{W}U|> θ)then
20: remove last element of {W}U
21: end if
22: end if
23: end for
24:end while
18