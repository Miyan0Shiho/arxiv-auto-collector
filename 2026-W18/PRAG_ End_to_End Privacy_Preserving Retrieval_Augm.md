# PRAG: End-to-End Privacy-Preserving Retrieval-Augmented Generation

**Authors**: Zhijun Li, Minghui Xu, Huayi Qi, Wenxuan Yu, Tingchuang Zhang, Qiao Zhang, GuangYong Shang, Zhen Ma, Xiuzhen Cheng

**Published**: 2026-04-29 10:46:45

**PDF URL**: [https://arxiv.org/pdf/2604.26525v2](https://arxiv.org/pdf/2604.26525v2)

## Abstract
Retrieval-Augmented Generation (RAG) is essential for enhancing Large Language Models (LLMs) with external knowledge, but its reliance on cloud environments exposes sensitive data to privacy risks. Existing privacy-preserving solutions often sacrifice retrieval quality due to noise injection or only provide partial encryption. We propose PRAG, an end-to-end privacy-preserving RAG system that achieves end-to-end confidentiality for both documents and queries without sacrificing the scalability of cloud-hosted RAG. PRAG features a dual-mode architecture: a non-interactive PRAG-I utilizes homomorphic-friendly approximations for low-latency retrieval, while an interactive PRAG-II leverages client assistance to match the accuracy of non-private RAG. To ensure robust semantic ordering, we introduce Operation-Error Estimation (OEE), a mechanism that stabilizes ranking against homomorphic noise. Experiments on large-scale datasets demonstrate that PRAG achieves competitive recall (72.45%-74.45%), practical retrieval latency, and strong resilience against graph reconstruction attacks while maintaining end-to-end confidentiality. This work confirms the feasibility of secure, high-performance RAG at scale.

## Full Text


<!-- PDF content starts -->

1
PRAG: End-to-End Privacy-Preserving
Retrieval-Augmented Generation
Zhijun Li, Minghui Xu, Member, IEEE, Huayi Qi, Wenxuan Yu, Tingchuang Zhang, Qiao Zhang, GuangYong
Shang, Zhen Ma, Xiuzhen Cheng, Fellow, IEEE
This work has been submitted to the IEEE for possible
publication. Copyright may be transferred without notice, after
which this version may no longer be accessible.
Abstract—Retrieval-Augmented Generation (RAG) is essential
for enhancing Large Language Models (LLMs) with external
knowledge, but its reliance on cloud environments exposes sensi-
tive data to privacy risks. Existing privacy-preserving solutions
often sacrifice retrieval quality due to noise injection or only
provide partial encryption. We propose PRAG, an end-to-end
privacy-preserving RAG system that achieves end-to-end confi-
dentiality for both documents and queries without sacrificing the
scalability of cloud-hosted RAG. PRAG features a dual-mode
architecture: a non-interactive PRAG-I utilizes homomorphic-
friendly approximations for low-latency retrieval, while an inter-
active PRAG-II leverages client assistance to match the accuracy
of non-private RAG. To ensure robust semantic ordering, we
introduce Operation-Error Estimation (OEE), a mechanism that
stabilizes ranking against homomorphic noise. Experiments on
large-scale datasets demonstrate that PRAG achieves competitive
recall (72.45%–74.45%), practical retrieval latency, and strong
resilience against graph reconstruction attacks while maintaining
end-to-end confidentiality. This work confirms the feasibility of
secure, high-performance RAG at scale.
Index Terms—Privacy-preserving RAG, homomorphic encryp-
tion, secure similarity retrieval, access-pattern leakage, encrypted
HNSW.
I. INTRODUCTION
With the rapid proliferation of Large Language Models
(LLMs) in question-answering and decision-support settings,
combined with recent advances in knowledge-intensive ar-
tificial intelligence systems, an increasing number of users
are investigating the use of Retrieval-Augmented Generation
(RAG) to construct private knowledge-base applications over
proprietary data [1], [2]. In particular, RAG facilitates the
integration of LLMs with intelligent retrieval methods, thereby
enabling the deployment of knowledge-centric applications in
a flexible, rapid, and low-code fashion [3]–[5].
Motivated by these advantages, cloud-based RAG has
emerged as the predominant deployment paradigm [6], [7].
Major cloud providers, including Amazon, Google, and Al-
ibaba Cloud, now offer native capabilities for seamlessly
Zhijun Li, Minghui Xu, Wenxuan Yu, Tingchuang Zhang, Qiao
Zhang, and Xiuzhen Cheng are with Shandong University. (e-mail:
richikun2014@gmail.com, mhxu@sdu.edu.cn, 202115114@mail.sdu.edu.cn,
tczhang@mail.sdu.edu.cn, qiao.zhang@sdu.edu.cn, xzcheng@sdu.edu.cn).
Huayi Qi is with Tsinghua University. (e-mail: huayiqi@tsinghua.edu.cn).
GuangYong Shang and Zhen Ma are with Inspur Yunzhou In-
dustrial Internet Co., Ltd. (e-mail: shangguangyong@inspur.com,
mazhenrj@inspur.com).
Corresponding author: Minghui Xuintegrating personal and enterprise document repositories into
RAG pipelines [8]–[10]. This allows organizations to rapidly
deploy AI agents with scalable, low-latency access to pro-
prietary knowledge, but also causes sensitive document and
retrieval indices to reside persistently on cloud servers, ex-
posing a new attack surface. In practice, authorized users
may even access such cloud-hosted knowledge bases from
mobile devices, further extending both the convenience and
the exposure surface of this deployment model. A series
of large-scale data breaches since 2024 demonstrates the
vulnerability of cloud-hosted data at scale. Notably, the Mother
of All Breaches (MOAB) exposed over 26 billion credential
records [11], while other incidents targeted universities, gov-
ernment contractors, and major technology vendors, including
Huawei and Google [12]–[15]. These events demonstrate that
storing sensitive data in the cloud without strong cryptographic
protection poses a systemic and ongoing privacy risk.
TABLE I
COMPARISON OFPRIVACY-PRESERVINGRAG SCHEMES
Scheme Method Doc (D)
ProtectionQuery (Q)
ProtectionProvable Guarantee
RemoteRAG [7] DP1Plaintext DP-perturbedε-DP (Q)
DP-RAG [16],
DPV oteRAG [17],
Text-DP [18]DP DP-
perturbedPlaintextε-DP (D)
SANNS [19],
TipToe [20]HE2Plaintext Encrypted Semantic security (Q)
Pacmann [21] PIR3Plaintext PIR-protected PIR security (Q)
Fortify [22] TEE Plaintext TEE-protected Trusted hardware (Q)
PRAG (Ours)HE Encrypted Encrypted Semantic security
(D,Q)
1DP: Differential Privacy.
2HE: Homomorphic Encryption.
3PIR: Private Information Retrieval.
Current research into privacy-preserving RAG explores a
spectrum of solutions ranging from local deployment to sta-
tistical noise injection and hardware-based execution environ-
ments, each offering different trade-offs between security and
utility as shown in Table I. To avoid exposing documents and
queries to the cloud, local deployments [1], [2], [6] keep all
data on the client side. However, this design is ill-suited for
shared, cloud-hosted RAG settings because it lacks support
for multi-user collaboration, centralized administration, and
scalable management of large-scale knowledge bases.
To address these limitations while remaining in the cloud,
some systems adopt Differential Privacy (DP) to reduce infor-arXiv:2604.26525v2  [cs.CR]  30 Apr 2026

2
mation leakage by injecting noise into documents [16]–[18].
While these approaches protect statistical privacy, they do not
provide provable data confidentiality; the cloud still stores per-
turbed document representations in plaintext and can observe
access patterns during retrieval. RemoteRAG [7] improves
upon this by perturbing user queries, preventing the cloud from
directly observing query contents. Nevertheless, the underlying
documents remain stored and processed in plaintext, leaving
the hosted knowledge base visible to the cloud provider;
SANNS [19], TipToe [20], and Pacmann [21] have the same
limitation in that they protect query privacy but not document
privacy, which may be acceptable for public dataset but not
for private knowledge bases. Alternatively, TEE-based RAG
systems [22] protect query processing by executing retrieval
within trusted hardware enclaves. This significantly reduces
data exposure during computation, though its security remains
tied to specific hardware trust assumptions. This raises a new
problem:
Is it possible to achieve end-to-end confidentiality for both
documents and queries without sacrificing the scalability of
cloud-hosted RAG?
A practical solution must maintain the corpus in the cloud
while providing robust cryptographic protection and preserv-
ing the full functionality required by RAG. In particular,
a privacy-preserving RAG system must support three core
operations:secure similarity computationbetween a query
embedding and document embeddings,efficient rankingto
select the top-kcontexts for generation, andreal-time updates
to keep the knowledge base synchronized with continuously
changing data. Approximate Nearest Neighbor (ANN) search,
with its sub-linear query complexity and favorable recall-
latency trade-off, is naturally suited for retrieval over large-
scale embedding corpora in modern RAG systems. To ensure
the cloud learns nothing about document contents, these opera-
tions cannot be performed on plaintext and must instead be ex-
ecuted entirely in the ciphertext domain. Under this end-to-end
encrypted cloud setting, these requirements effectively leave
Secure Similarity Retrieval (SSR)[19], [23]–[26] as the most
viable design direction, since it enables similarity retrieval
directly over encrypted data in the cloud. However, existing
Secure Similarity Retrieval schemes still have limitations in
practical RAG deployments.
A. Why SSR Fails in Privacy-Preserving RAG
Partial Support for RAG Functionality in Secure Sim-
ilarity Retrieval.Althoughsecure similarity computation,
efficient ranking, andreal-time updatesare the pillars of
privacy-preserving RAG systems, current SSR schemes fail
to provide these three capabilities concurrently, as shown in
Table II. The following analysis identifies the specific gaps in
existing solutions.
First, several schemes support similarity computation but
rely on encryption primitives that have been shown to be
insecure under practical attack models, such as ASPE-based
constructions [23], [24]. Although these approaches support
encrypted similarity computation, their security guarantees areTABLE II
COMPARISON OFSECURESIMILARITYRETRIEVALSCHEMES
Scheme Secure Similarity
ComputationEfficient
RankingReal-Time
Updates
Huang et al. [23]
SPE-Sim [24]
SESR [25]
PDQ [26]
MSecKNN [28]
FSkNN [29]
PRAG (Ours)
: Supported; : Not supported; : Partial support under specific
settings or with performance constraints.
limited for protecting knowledge bases [27]. Second, efficient
RAG retrieval requires ranking similarity scores to select
the most relevant context. However, schemes such as SPE-
Sim [24] and SESR [25] do not support numeric comparison in
the ciphertext domain. Without native encrypted comparison,
these systems cannot perform top-kranking without leaking
sensitive score distributions or requiring trusted-party decryp-
tion. Third, the dynamic nature of RAG requires support for
evolving datasets, yet many schemes, such as PDQ [26] and
SESR [25], lack real-time update capabilities. Their index
structures often rely on global organization or rigid parti-
tioning, making new insertions computationally prohibitive.
This limitation affects the handling of continuously evolving
knowledge bases.
Distortion of retrieval ranking.Even when encrypted
comparison is available, or when the system relies on an HE-
friendly ranking surrogate, the ordering of similarity scores can
change under CKKS-style approximate arithmetic. In privacy-
preserving RAG, each similarity score is computed through
multiple steps, including packed inner products, rescaling
and rounding, and in fully homomorphic settings, polynomial
approximations for selection. These steps introduce errors
that may vary across candidates and accumulate during re-
trieval [30], [31].
This leads to two challenges. First, top-kretrieval is sen-
sitive to score differences [32], [33]. If two candidates have
similar true scores, even a small error can flip their order.
Second, error depends on both input data and retrieval path.
Different candidates may accumulate different error due to
different operation counts, such as different traversal lengths,
making a single absolute error bound insufficient. The core
challenge is therefore to keep total perturbation below the sim-
ilarity margin that separates relevant documents from near ties,
so encrypted retrieval preserves ranking with high probability.
B. Our Contribution
To address these challenges, we propose PRAG, an end-
to-end privacy-preserving RAG framework that enables high-
accuracy retrieval over encrypted knowledge bases. Our con-
tributions are summarized as follows:
•Bridging secure similarity retrieval and basic RAG
functionality.To address the incompatibility between ex-
isting secure similarity retrieval schemes and basic RAG

3
requirements, we develop a CKKS-compatible encrypted
retrieval backend that supports (1) similarity scoring
via homomorphic inner products, (2) ranking through
Chebyshev polynomial approximations or client-assisted
comparisons, and (3) real-time updates using a ciphertext-
resident hierarchical index with homomorphic clustering
and encrypted HNSW navigation.
•An end-to-end privacy-preserving RAG system with
dual-mode operation.We propose PRAG, an end-to-
end privacy-preserving RAG system that supports two
deployable modes within a unified encrypted architecture:
a non-interactive PRAG-I mode that performs retrieval
fully in the ciphertext domain via homomorphic-friendly
approximations, and an interactive, client-assisted PRAG-
II mode that leverages limited client-side assistance to im-
prove ranking precision and retrieval accuracy, achieving
average top-10 retrieval latencies of under 1.29 seconds
and around 7.91 seconds, respectively, on a 100,000-
sample dataset. In our experiments, the two modes
achieve recall rates of 72.45% and 74.45%, respectively,
while sharing the same encrypted corpus and index and
enabling on-demand switching without re-encrypting data
or rebuilding indices. The system further incorporates
access-pattern mitigation against graph reconstruction at-
tacks, together with formal leakage analysis and empirical
evaluation of the resulting security-utility trade-off.
•Noise-aware ranking stabilization via Operation-
Error Estimation (OEE).We introduce OEE, a noise
control method that models how approximation error and
CKKS noise affect similarity scores during encrypted
retrieval. Rather than focusing only on numeric accuracy,
OEE targets preservation of the relative order of similarity
scores, which significantly impacts retrieval quality in
RAG. Specifically, OEE accounts for: (1) polynomially
approximated comparisons, which occur only in PRAG-
I; and (2) homomorphic noise and hierarchical index
traversal, which affect both PRAG-I and PRAG-II. Using
OEE, we guide parameter selection to bound ranking
errors, making ranking stability an explicit design goal
for encrypted retrieval.
The remainder of this paper details the system architecture,
security model, and protocol design, followed by experimental
evaluation and discussion.
II. RELATEDWORK
A. Privacy-Preserving RAG (PRAG)
Existing privacy-preserving RAG research can be broadly
organized into three routes: statistical perturbation of the
retrieval pipeline, system-level trust decomposition, and se-
lective plaintext assistance during retrieval or ranking. This
taxonomy clarifies the central gap in the literature: prior
systems usually strengthen one aspect of privacy, but they do
not simultaneously provide end-to-end corpus confidentiality,
accurate semantic retrieval, and practical cloud deployment.
RAG first introduced by Lewis et al. [34] and later re-
fined by Active RAG [35] established the retrieval-generation
paradigm, but these works do not consider the privacy risksof sensitive corpora. Subsequent analyses by Huang et al. [36]
and Zeng et al. [37] showed that retrieved knowledge and
training signals can leak private information. One line of
defense therefore adds statistical noise to the retrieval pipeline.
DP-based approaches [16]–[18] and embedding perturbation
methods such as PRESS [38] improve privacy for queries or
outputs, but they inevitably distort similarity relationships and
still leave the outsourced corpus exposed in plaintext.
A second route reduces reliance on a single cloud server
through architectural redesign. Federated systems such as
FRAG [39], blockchain-based systems such as D-RAG [40],
split execution [2], and isolated-enclave solutions [1], [6]
all limit the trust placed in one component of the system.
However, these designs typically introduce substantial com-
munication or coordination overhead and still do not provide
an efficient encrypted vector index that supports RAG-style
semantic search.
A third route keeps part of the pipeline encrypted but reveals
intermediate information to recover accuracy or efficiency.
Privacy-Aware RAG [41], for example, requires selective de-
cryption during ranking, which breaks end-to-end confiden-
tiality and reintroduces exposure during the most ranking-
sensitive stage of retrieval.
In summary, the three main RAG directions correspond to
three incomplete trade-offs: perturbation-based methods sacri-
fice utility, trust-decomposition methods sacrifice practicality,
and plaintext-assisted methods sacrifice full confidentiality.
PRAG is designed to close this gap by supporting clustering,
HNSW traversal, and ranking over a fully encrypted corpus
under an untrusted server model.
B. Secure Similarity Retrieval
Existing secure similarity retrieval (SSR) work can likewise
be summarized along three routes: query-private retrieval over
plaintext corpora, encrypted similarity computation with weak
or limited ranking semantics, and structured encrypted indexes
that still fall short of RAG’s ANN requirements. This lens
makes clear why current SSR primitives cannot be plugged
into RAG directly without losing either security or function-
ality.
Early works such as SANNS [19] and TipToe [20] protect
the query while leaving the document embeddings exposed
on the server. Pacmann [21] shifts crucial computation back
to the client because it cannot support server-side comparison
and similarity computation securely. These systems therefore
do not meet the end-to-end confidentiality requirement of
outsourced RAG.
Huang et al. [23] proposed an ASPE-based method, but
ASPE is not secure against ciphertext-only attacks [27].
Subsequent efforts extended query semantics to support set
similarities, as in SPE-Sim [24] and SESR [25], or numerical
comparisons over integers, as in PDQ [26]. Even when these
systems support secure arithmetic, they do not provide the con-
tinuous, ranking-preserving comparisons needed for semantic
vector retrieval in RAG.
Some methods [42] use clustering or quantization, but
they partition the feature space rigidly and break semantic

4
continuity across clusters. Real-time update mechanisms from
keyword-based systems [43]–[45] are not directly compatible
with continuous embedding retrieval. Recent designs based on
R-trees [46], [47] or specialized encodings [48], [49] improve
structure, yet none jointly support ANN graph navigation,
iterative centroid refinement, and semantically faithful vector
operations.
Overall, existing SSR schemes each cover only a sub-
set of the requirements for encrypted RAG. For example,
SANNS [19] requires multiple online parties, and update
operations can disrupt cluster balance and trigger costly recon-
struction. As a result, no prior SSR scheme simultaneously sat-
isfies the navigational, semantic, dynamic, and cryptographic
requirements of practical privacy-preserving RAG.
III. PRELIMINARIES
A. Graph-Based ANN Search
Approximate Nearest Neighbor (ANN) search is a funda-
mental operation in high-dimensional vector databases that
trades exactness for efficiency, returning approximate nearest
neighbors with sub-linear query complexity. Among ANN
approaches, graph-based methods have emerged as the state-
of-the-art for high-dimensional similarity search due to their
superior recall-latency trade-off. These methods construct a
proximity graph over the dataset where each node represents
a data point and edges connect nearby points. Search proceeds
by greedily traversing the graph from an entry point, iteratively
moving to neighbors closer to the query until reaching a local
optimum. Compared to tree-based methods (e.g., KD-trees)
that suffer from the curse of dimensionality, and hashing-based
methods (e.g., LSH) that require parameter tuning for specific
distance distributions, graph-based methods adaptively capture
the intrinsic data manifold and achieve robust performance
across diverse datasets.
Hierarchical Navigable Small World (HNSW) is a repre-
sentative graph-based ANN algorithm that extends the basic
proximity graph with a multi-layered hierarchical structure. It
organizes data points into a multi-layered graph to perform
search in a coarse-to-fine manner, where sparse upper layers
facilitate rapid global navigation and dense lower layers enable
accurate local refinement. At each layer, the search follows a
greedy routing rule: starting from a designated entry node,
the algorithm iteratively evaluates the distances to the current
node’s neighbors and moves to the neighbor that is closest
to the query vector. This process continues until no adjacent
neighbor is closer to the query than the current node, reaching
a local optimum that serves as the entry point for the next
lower layer.
To illustrate this mechanism concretely, consider a typical
three-layer HNSW configuration consisting of a sparse top
layer (L 2), a middle layer (L 1), and a dense base layer (L 0)
containing all data points, as illustrated in Fig. 1. The retrieval
process is initialized at a predefined global entry node in
L2, where the algorithm performs a rapid greedy traversal to
”zoom in” on the general region of the query while bypassing
large portions of irrelevant data. Once a local optimum is iden-
tified inL 2, the search transitions toL 1, using that optimumas the new entry point. SinceL 1possesses a denser topology,
it provides a finer-grained routing path to further narrow the
search space. Finally, the search descends into the densest
layer,L 0, to execute a precise localized greedy search, often
maintaining a dynamic candidate list to pinpoint the exact top-
knearest neighbors. This hierarchical transition ensures that
the search complexity scales sub-linearly, typicallyO(logN),
while maintaining high recall in high-dimensional spaces.
L2
L1
L0Entry
Fig. 1. Schematic diagram of the HNSW structure.
B. K-means Clustering
K-means clustering is a widely used unsupervised learning
algorithm for partitioning|D|data points intokclusters, where
each data point belongs to the cluster with the nearest centroid.
The algorithm iteratively minimizes the within-cluster sum
of squared distances by updating cluster assignments and
centroids until convergence. Formally, given a set of data
points{x 1, x2, . . . , x n}in ad-dimensional space, the goal is
to find centroids{µ 1, . . . , µ k}that minimize
nX
i=1min
j∈{1,...,k}∥xi−µj∥2.
C. CKKS and Primitives
We briefly introduce the CKKS homomorphic encryption
abstraction used in this work. CKKS supports approximate
arithmetic on packed vectors, enabling homomorphic addition,
multiplication, and slot rotations. We use the following high-
level primitives:
•KeyGen(λ)→(pk,sk,evk).
•Encrypt(pk,v)→ct vandDecrypt(sk,ct)→v′.
•Add,Mul,Rotate: homomorphic slot-wise operations.
D. Chebyshev Polynomial Approximation (ChebyApprox)
Lets= (s 1, . . . , s C)∈RCbe similarity scores or distance
scores, and letτ >0be a scaling parameter. Since CKKS
supports only additions and multiplications efficiently, we
approximate a monotone comparison surrogate over a bounded
interval with a degree-dChebyshev polynomial
Cd(x) =dX
m=0amTm(x),

5
whereT m(·)is them-th Chebyshev polynomial of the first
kind. We then define
ChebyApprox(s i) :=w i=Cd(−si/τ)PC
j=1Cd(−sj/τ), i∈[C].
The resulting weightsw= (w 1, . . . , w C)satisfyPC
i=1wi= 1
and provide an HE-friendly surrogate for comparison-based
ranking.
IV. PROBLEMSTATEMENT
A. System Setting
Our proposed framework is designed for an environment
where data providers can contribute to a shared encrypted
RAG database. The architecture comprises two main entities:
a trusted Client, and a semi-honest Cloud Server.
A trusted Client serves two roles. As the data owner, it
normalizes private documents into embedding vectors, builds
the search index, encrypts the entire dataset and index struc-
ture, and uploads the ciphertexts to the Cloud Server. As
an authorized user, it transforms a query into multiple sub-
queries, encrypts them, and submits them to the server; upon
receiving the encrypted retrieval results from the cloud, it
decrypts them and forwards the plaintext context to the large
language model for response generation.
The Cloud Server is a semi-honest provider that hosts
the encrypted database. Upon receiving an encrypted query,
it performs all similarity-matching computations directly on
ciphertexts, without any decryption key access, and returns
the top-kencrypted candidates to the Client. The Client then
decrypts and reranks the results locally.
B. Threat Model
We model the cloud server as a Probabilistic Polynomial-
Time (PPT) adversaryAunder the semi-honest model. The
adversary:
•Infers sensitive information from all observable data;
•Holds no secret keyskand does not collude with others.
•Correctly executes all protocols using provided keys;
The adversary’s view consists of:
ViewA=
{ct(i)
v}|D|
i=1,{ct(j)
q}Q
j=1,CT inter,Params,L
,(1)
where{ct(i)
v},{ct(j)
q}are encrypted database, queries,CT inter
are intermediate ciphertexts generated during computation,
Paramsare public parameters such asλ,N, andτ, andL
is the leakage function defined in Appendix A.
C. Design Goals
•Data Privacy (Semantic Confidentiality).All document
embeddings, query embeddings, similarity scores, and
final rankings must remain semantically hidden from the
cloud server. Formally, the CKKS-encrypted ciphertexts
ensure IND-CPA security: for any two datasetsV 0,V1
and queriesq 0, q1of equal size, the adversary’s views
are computationally indistinguishable. We formalize this
as semantic confidentiality.Definition IV .1(Semantic Confidentiality).Our frame-
work achieves semantic confidentiality if for any PPT
adversaryA, there exists a PPT simulatorSsuch that:
n
ViewA(D,{q(i)})oc≈n
S
L(D,{q(i)}),1λo
,(2)
wherec≈denotes computational indistinguishability.
Additionally, the permitted leakageLconsists of access
patterns during HNSW traversal and cluster probing.
PRAG explicitly bounds this leakage via random access
perturbation to prevent graph reconstruction attacks. We
categorize access-pattern-based graph reconstruction at-
tacks as a class of LAAs, where the adversary exploits
observable retrieval traces to infer hidden index topology.
•Ranking Stability under Noise.The system ensures
that semantic ranking is robust to approximation error
and homomorphic noise. In particular, retrieval quality
is preserved by explicitly controlling ranking deviations
rather than only minimizing numeric error.
•Practical and Efficient Dual-mode System Design.
The overall design targets real-world RAG workloads by
supporting efficient and accurate retrieval on large-scale
datasets with real-time updates, and by enabling dual-
mode operation to trade latency for ranking accuracy,
including fully homomorphic non-interactive retrieval and
client-assisted retrieval.
TABLE III
SUMMARY OFCORENOTATIONS
Symbol Description
|D|Dataset size
dVector dimension
CNumber of clusters
TK-means iterations
τChebyshev scaling parameter
ct(·) Encrypted ciphertext, such as vectorct vor scorect s
NSet of neighbors
Cprobe Number of clusters probed per query
ksub Number of sub-queries generated per cluster
SIP Noise of a single homomorphic inner product
v,qPlaintext vectors (data and query)
N, λCKKS polynomial modulus degree and security pa-
rameter
pk,sk,evkPublic key, secret key, and evaluation keys (rlk,galk)
L, M,DephHNSW parameters: max layers, max degree, and
search depth
V. PRIVACY-PRESERVINGRAG
In this section, we present two complementary privacy-
preserving RAG frameworks as shown in Fig. 2: a non-
interactive PRAG-I that achieves end-to-end encrypted re-
trieval, and an interactive PRAG-II that enhances ranking
precision. Both frameworks share the same data preprocessing
and response generation stages, but differ in Setup, Retrieve
and Update stages. The key notations used in our protocols
are shown in Table III.

6
Client
LLM&
OpenAIEmb
Enc( Pk, Vi)
ChebyApprox
DistCompWeighted Sum& 
HomoNorm
TiterationsHomolPInit Enc
CentroidsAsstDistCompQuery
Sub-Query
Operations 
only for 
PRAG-IIOperations 
only for 
PRAG-I
Server
Cluster A
Cluster B,Same process asabove A
Cluster C,Same process asabove AEmbedding( Vi)
① ② ③Update Return
Top-k         
results② ③Enc Enc EncFile
Fig. 2. Architecture of PRAG-I and PRAG-II. Note: Numbers in green circles
denote steps exclusive to PRAG-I, while numbers in brick-red circles indicate
steps specific to PRAG-II; the remaining steps are common to both schemes.
A. PRAG-I
The PRAG-I framework performs retrieval homomorphi-
cally on the cloud using CKKS, and uses a Chebyshev
polynomial approximation for ranking.
1) Detailed Protocols:We present the pseudocode for
ProtocolsΠ Setup,ΠRetrieval , andΠ Update in Section V-A1, and
ProtocolΠ LocalRAG in Appendix C.
Setup Stage: Encrypted Upload and Index Construction.
As shown in Protocol 1, the client encrypts each vector
vi∈ Vwith its identifierID iunderpkto obtain ciphertextsct i.
CKKS parameters, including the polynomial modulus degree
N, the moduli chainq i, and the initial scaling∆, are chosen
to match the multiplicative depth of the homomorphic inner-
product circuit. The client then uploads{ct i}and evaluation
keysevkto the untrusted server, which stores them as a two-
level encrypted indexI.
The server then runs encrypted K-means. It initializes
centroids by samplingCvectors and encrypting them as
{ctµ1, . . . ,ct µC}. ForTiterations, it computes homomorphic
inner productsct sjbetween each data pointct iand all
centroids, appliesChebyApproxwith scaling parameterτ
to obtain weightsct wj, accumulates weighted sums in each
partitionP j, and updates centroids viaHomoNorm. Details
ofHomoNormare given in Appendix C.
Once clustering is complete, the encrypted centroids form
the cluster-level indexI cluster . For the second level, the server
constructs an encrypted HNSW graphG jfor each cluster
by iteratively inserting assigned encrypted vectors using the
secure insertion procedure in Protocol 3. The full indexI
combines both levels. Finally, encrypted centroids are returned
to the client and decrypted into plaintext cluster metadata
Cmeta, which contains centroids and semantic labels. Because
this metadata is aggregate, it does not reveal individual data
points and can be safely used for subsequent retrieval.
Retrieve Stage: Hierarchical and Fused Secure Re-
trieval.As shown in Protocol 2, retrieval uses a hierarchical
index and query transformation while keeping server-side
computation efficient. The process starts with client-side query
decomposition. The original queryqis embedded intov qvia
OpenAIEmbeddings (OpenAI Emb). Using plaintext clusterAlgorithm 1:ProtocolΠ Setup
KwIn :Plaintext vectorsV, public keypk, evaluation keysevk,
parametersC, T, τ.
KwOut :Encrypted indexIat the server and encrypted centroids
{ctµj}returned to the client.
// Client-Side Encryption
1CTV← {(ID i,Enc(pk,v i))|(ID i,vi)∈ V};
2UploadCT Vandevkto the server;
// Level-1: Encrypted Chebyshev-Weighted
K-means Clustering
3Randomly initialize encrypted centroids{ct µ1, . . . ,ct µC};
4fort←1toTdo
5InitializeP j.sum←0andP j.count←0for allj∈[C];
6foreach(ID i,cti)∈ CT Vdo
7Computect sj←HomoIP(ct i,ctµj)for allj;
8Computect wj←ChebyApprox({ct sj}, τ);
9UpdateP j.sum← P j.sum+ct i·ctwjand
Pj.count← P j.count+ct wj;
10end
11forj←1toCdo
12ifP j.count̸= 0then
13ct µj←HomoNorm(P j.sum,P j.count);
14end
15end
16end
17Store encrypted centroids asI cluster ;
// Level-2: Encrypted HNSW Construction
18forj←1toCdo
19Construct encrypted HNSW graphG jusingSecureInsert;
20end
21I ←(I cluster,IHNSW );
22Send encrypted centroids{ct µj}to the client and outputI;
metadataC meta, which is decrypted from the server during
setup, the client computes similarities betweenv qand cluster
centroids, then selects the top-C probe clustersJ bestfor semantic
routing without revealing query content to the server.
To improve recall, the client performs query fusion within
each selected cluster. For each clusterj∗, it retrieves the
semantic label fromC meta and uses a local LLM via
LLM.Decomposeto generatek subdiverse sub-queries tai-
lored to the cluster topic and original query. Each sub-query is
embedded by OpenAI Emb, encrypted intoct sub, and paired
with its target cluster ID. The encrypted sub-query set is sent
to the server with the desired top-kparameter.
On the server side, retrieval is targeted and efficient. For
each encrypted sub-queryct suband designated clusterj∗, the
server retrieves the corresponding encrypted HNSW graph
Gj∗fromI HNSW . It then runs greedy search from entry point
epto find top-knearest neighbors in the encrypted domain
using homomorphic distance computations. Partial results con-
taining candidate identifiers and scores are aggregated and
returned to the client. Restricting search to selected clusters
and parallelizing sub-queries reduces server overhead while
preserving homomorphic security. The client then fuses and
reranks aggregated results in Appendix 4 to produce final
context.
To reduce access-pattern leakage during retrieval, PRAG-I
further augments the above search process with randomized
protection mechanisms. First, for each real encrypted sub-
query, the client injects aρ-fraction of dummy traversal
requests that target randomly selected clusters and trigger ran-
dom walks of matched depth in the corresponding encrypted

7
Algorithm 2:ProtocolΠ SecureRetrieval
KwIn :Queryq, top-kparameterK, parameters(C probe, ksub),
client metadataC meta, and scaling parameterτ.
KwOut :Aggregated encrypted retrieval results.
// Client-Side Query Decomposition
1vq←OpenAI Emb(q);
2Compute similarity scores betweenv qand cluster centroids using
Cmeta;
3Select top-C probe clusters and denote them asJ best;
// Encrypted Sub-query Generation
4EncSubQueries← ∅;
5foreachj∗∈ J bestdo
6Retrieve semantic label fromC meta[j∗];
7Generatek subsub-queries viaLLM.Decompose;
8foreachq subin generated sub-queriesdo
9v sub←OpenAI Emb(q sub);
10ct sub←Enc(pk,v sub);
11Append(ct sub, j∗)toEncSubQueries;
12end
13end
14SendEncSubQueriesandKto the server;
// Server-Side Targeted Encrypted Search
15AggregatedResults← ∅;
16foreach(ct sub, j∗)∈EncSubQueriesdo
17Retrieve encrypted HNSW graphG j∗;
18Perform greedy encrypted ANN search onG j∗withct suband
obtain top-kidentifiers;
19Append(j∗,PartialResults)toAggregatedResults;
20end
21returnAggregatedResults;
HNSW graphs. In addition, each real traversal is padded to a
fixed observable lengthℓ max by appending random neighbor
visits, so the server observes a mixture of real and dummy
paths with aligned lengths rather than a clean trace of the true
greedy search. Second, the encrypted index is periodically re-
encrypted and re-clustered after every epoch ofEqueries, us-
ing fresh ciphertext randomness, fresh K-means initialization,
and freshly rebuilt HNSW layers. This prevents the server from
accumulating stable long-term traversal statistics over a fixed
graph. Together, these protections are part of the operational
design of PRAG-I, while their formal leakage model and
security proof are deferred to Section VI-A3.
Update Stage: Secure Data Insertion and Dele-
tion.As shown in Protocol 3, this protocol supports dy-
namic encrypted-index updates for evolving datasets. The
SecureInsertprocedure adapts standard HNSW insertion
to homomorphic encryption. For a new encrypted node
(ID new,ctnew), a random maximum layerLis assigned fol-
lowing HNSW probabilistic layering. From the top layer
down toL+ 1, the protocol performs a preliminary search:
from entry pointep, it retrieves candidate neighbors and
computes encrypted distancesct dkvia homomorphic inner
products. Because discrete selection is incompatible with HE,
ChebyApproxassigns Chebyshev-derived weightsct wkbased
on distances, withτcontrolling the approximation scale. The
next entry point is computed as a weighted sum of candidates,
yielding a continuous approximation for descent.
From layerLdownward to the base layer (0), a more
thorough greedy search is conducted with depth controlled by
a depth heuristic, denoted as Dephto find potential neighbors.
Again, Chebyshev-derived weights are applied to these neigh-Algorithm 3:ProtocolΠ SecureUpdate
KwIn :Encrypted HNSW graphG, new encrypted node
(IDnew,ctnew), scaling parameterτ, search depth Deph,
and maximum degreeM.
KwOut :Updated encrypted HNSW graphG.
// Procedure SecureInsert
1L←RandomLayer(),ep←G.entryPoint,
topLayer←G.maxLayer;
2forl=topLayer, . . . , L+ 1do
3N ←Neighbors(G,ep, l);
4Computect dk←HomoIP(ct new,N[k]);
5Computect wk←ChebyApprox({ct dk}, τ);
6ep←P
kctwk· N[k];
7end
8forl= min(L,topLayer), . . . ,0do
9N ←GreedySearch(G,ct new,ep,Deph, l);
10W ←ChebyApprox(N, τ);
11N′←WeightedTopM(N,W, M);
12G.link(ct new,N′, l);
13ep←PW · N′;
14end
15Output updated graphG;
// Procedure SecureDelete
16ct node←FindNodeByID(I,ID del);
17ifct node is foundthen
18Mark node as deleted using a secure flag; keep it for routing;
19end
20returndeletion status;
bors, and the topMare selected using a weighted ranking
WeightedTopM to establish bidirectional links in the graph.
The entry point for the next layer is updated as a weighted
aggregate.
TheSecureDeleteprocedure addresses deletion without
compromising security. Rather than physically removing a
node, which could disrupt graph connectivity and potentially
leak information through observable changes in access pat-
terns, the protocol locates the encrypted node by its ID and
sets a deleted flag. This ”mark-and-sweep” strategy allows the
node to remain as a routing point, ensuring ongoing search
efficiency.
B. PRAG-II
To reduce approximation errors from the Chebyshev poly-
nomial surrogate, we introduce an interactive variant of our
PRAG. In this mode, the cloud computes all distance-related
operations homomorphically, while the client partially de-
crypts intermediate encrypted distances, such as cluster dis-
tances or HNSW layer candidates, to decide the next nav-
igation step. This design preserves the hierarchical greedy
structure of HNSW and achieves retrieval accuracy closely
approximating plaintext search, at the cost of multiple client-
server communication rounds. The preprocessing and final
RAG response phases are the same as those in the PRAG-
I, so they will not be described again in this subsection.
Interactive Setup.This procedure builds the encrypted two-
level indexIon an untrusted server under an interactive
client-assisted protocol, sharing the same overall structure as
the PRAG-I index construction protocolΠ Setup but replacing
Chebyshev polynomial approximations with client-guided ex-
act decisions. The client-side encryption and upload of vectors
Vto produce ciphertextsCT Vremain identical.

8
On the server side, during each K-means iteration, the
process diverges: instead of usingChebyApproxfor weighted
assignments, the server computes exact encrypted distances
from eachct ito all centroids usingHomoDist, which contex-
tually utilizesHomoIPto generate encrypted similarity scores
for precise client-side comparison, and sends these ciphertexts
to the client. The client decrypts them, identifies the nearest
clusterj∗in plaintext, and sends back only the cluster index.
The server then aggregates the assigned ciphertexts homo-
morphically and updates centroids viaHomoNorm, which
essentially performs a homomorphic divisionct sum/ctcount
using a polynomial approximation of the reciprocal function
x7→1/xto compute the new centroid of the aggregated
vectors, mirroring the aggregation step in the PRAG-I but with
hard assignments.
After clustering, the construction of encrypted HNSW
graphs per cluster follows the interactive insertion procedure
described below, rather than the PRAG-I. Finally, the en-
crypted centroids are sent to the client for decryption into
plaintext metadataC meta, as in the PRAG-I. This achieves
identical clustering results to plaintext K-means without ap-
proximations, while preserving privacy, at the expense of
O(NCT)communication rounds during indexing.
Interactive Retrieval.This phase extends the PRAG-I
retrieval protocolΠ SecureRetrieval into an interactive protocol
that removes Chebyshev polynomial approximations in favor
of exact client-side ranking decisions. The client-side query
preparation, including embeddingqintov qand encrypting it
toct q, is unchanged.
However, cluster routing differs: the server computes en-
crypted distances fromct qto all centroids and sends them to
the client, who decrypts and selects the top-C probe clusters in
plaintext, returning their indices. This replaces homomorphic
Chebyshev-weighted routing with exact client-side selection.
For each selected cluster, HNSW traversal is guided interac-
tively: at each layer, the server computes encrypted distances
fromct qto candidate nodes and sends them to the client, who
decrypts, selects the nearest entry point, or at lower layers
selects the top-Mneighbors, and returns the choices. The
final layer-0 greedy search on the server produces exact top-k
candidates, matching plaintext HNSW behavior.
Interactive Update.To support dynamic updates, this ex-
tends the PRAG-I update protocolΠ SecureUpdate to an interac-
tive form, replacing Chebyshev weighting with client-assisted
exact selections. During insertion, layer assignment and entry
point initialization remain the same.
At each layer, the server computes encrypted distances from
the newct newto candidates usingHomoDistand sends them
to the client, who decrypts and selects the nearest neighbor
as the next entry point or selects the top-Mclosest nodes for
linking, returning the encrypted indices. This ensures exact
HNSW topology without approximations. Deletion uses the
same non-interactive mark-and-sweep strategy: flagging the
node as deleted while keeping it for routing, with optional
offline reconstruction. Overall, this preserves the graph’s struc-
tural integrity under encryption, with client-guided decisions
adding communication but guaranteeing accuracy equivalent
to plaintext operations.C. Operation-Error Estimation (OEE)
A key challenge in deploying CKKS-based retrieval systems
is managing computational noise and approximation errors.
As established in our contribution summary, retrieval quality
hinges on preserving the relative order of similarity scores
rather than achieving pointwise numeric accuracy. To formally
guarantee this property, we abstract the OEE mechanism intro-
duced earlier into aRanking-Preserving Correctness Model.
This model provides a theoretical framework that quantifies
how homomorphic noise propagates through similarity com-
putations, and derives explicit bounds to ensure that ranking
inversions occur with negligible probability under properly
tuned CKKS parameters.
To make this dependency explicit, PRAG decomposes the
encrypted score of candidated ias
ˆsi=si+ξpoly
i+ξhe
i+ξpath
i,
wheres i=⟨q, d i⟩is the true similarity,ξpoly
i is the ap-
proximation error introduced by Chebyshev-based comparison
surrogates,ξhe
icaptures CKKS arithmetic noise from inner
products, rescaling, and rotations, andξpath
i captures path-
dependent accumulation caused by different traversal depths
or update paths. In PRAG-I, all three terms may appear; in
PRAG-II, client-assisted comparisons eliminate the dominant
comparison-approximation term for interactive decisions, leav-
ing CKKS arithmetic noise and path-dependent accumulation
as the main sources.
OEE shifts the correctness target from minimizing average
numerical error to preserving the margin between competing
candidates. Letqbe a query andd i, djtwo documents with
true similarity gapδ=|⟨q, d i⟩ − ⟨q, d j⟩|. Even if each
encrypted score has error less thanδ/2, asymmetric noise or
approximation can still flip their order. Such ranking errors
break top-kretrieval and harm RAG output, but are invisible
to standard metrics such as mean square error (MSE).
Ranking is preserved if, whenever the true similarity gap
exceeds a margin∆(where∆depends on homomorphic noise
and approximation error), the encrypted retrieval maintains the
correct order with overwhelming probability. Concretely, for
documentsd iandd j, ranking is stable whenever:
|(ˆsi−ˆsj)−(s i−sj)| ≤ϵ OEE<∆/2.
By choosing parameters so that the total perturbation is less
than∆/2, PRAG ensures that ranking errors occur only
when documents are semantically indistinguishable, meaning
any ambiguity reflects true embedding similarity rather than
encryption-induced leakage.
OEE also reveals that noise is unevenly distributed across
phases. Table IV shows that online retrieval usually re-
quires only hundreds of independent inner-product evaluations,
whereas encrypted K-means and full index construction re-
quire millions to tens of millions. Thus, the dominant risk is
not one query overflowing the noise budget; it is that offline
construction determines the minimum CKKS parameter budget
needed for the whole system.
This phase-aware view directly guides mitigation. At the
cryptographic level, we choose a sufficiently large polynomial

9
TABLE IV
COMPARATIVE NOISE ANALYSIS OF OPERATIONS IN ENCRYPTED VECTOR SIMILARITY SEARCH
Operation Inner Product Count Order of Magnitude Noise Characteristics
Single inner product1O(1)Baseline unit (S IP)
Single queryC+C probe·L·log(M)O(102)Independent computations
Single HNSW insertionL·Deph O(102)Independent comparisons
K-means (one iteration)N·C O(106)−O(107)Dominant noise source
Full index constructionT· |D| ·C+|D| ·L·Deph O(107)−O(108)Critical bottleneck
modulus degreeN, an adequate ciphertext moduli chain{q i},
and a calibrated scaling factor∆ scale so that the full con-
struction circuit fits within the available noise budget. At the
algorithmic level, we limit K-means to a small number of iter-
ations and adopt conservative HNSW construction parameters
such as a smaller Deph. These choices reduce multiplicative
depth and operation count in the dominant offline phase while
keeping retrieval quality acceptable. Additional derivations are
provided in Appendix D.
VI. THEORETICALANALYSIS
We first present a formal security analysis under the honest-
but-curious adversary model. Building on CKKS IND-CPA
security and game-based proofs, we establish view indistin-
guishability and analyze data privacy, query privacy, and result
privacy, including ranking confidentiality, for both operating
modes, PRAG-I and PRAG-II. We then compare PRAG’s
theoretical complexity with representative SSR schemes in
Table V.
A. Security Analysis
1) Security Definitions and Leakage Definitions:We for-
malize PRAG’s security along two dimensions. The first is
data-level security, which protects ciphertext contents. The
second isaccess-pattern-level security, which bounds the
information leaked through observable traversal patterns. Let
Adenote the honest-but-curious cloud server adversary with
polynomial-time capabilities.
Definition VI.1(IND-CPA Security of CKKS).The
CKKS scheme is IND-CPA secure if for any polynomial-
time adversaryA, the advantageAdvIND-CPA
A =
|Pr[A(Enc(m 0)) = 0]−Pr[A(Enc(m 1)) = 1]|is negligible
in the security parameterλ, wherem 0,m1are chosen byA.
This provides the foundational guarantee that all encrypted
embeddings, queries, similarity scores, and rankings are se-
mantically hidden.
Definition VI.2(Leakage Function).The permitted leakage
Lcharacterizes the access-pattern information observable by
the server. However, PRAG introducesrandom access per-
turbationto suppress the fidelity of this leakage. Formally,
letL truedenote the true access pattern andL obsdenote the
observed perturbed access pattern. PRAG ensures thatL obsis
a noisy version ofL true. The perturbation is parameterized by
the obfuscation rateρ, which denotes the fraction of dummy
accesses, and by the re-encryption epochE:
L=L obs=Perturb(L true;ρ, E).(3)The perturbed leakage includes: the noisy sequence of ac-
cessed nodes in encrypted HNSW graphs and the perturbed
set of probed clusters, mixed with dummy accesses. Notably,
Ldoes not include plaintext values, similarity scores, or exact
rankings. We do not explicitly include update-stage leakage
inL, as index updates are treated as offline operations with
observable effects limited to local index connectivity.
Definition VI.3(Semantic Confidentiality under Perturbed
Leakage).PRAG achieves semantic confidentiality under
perturbed leakage if: (1)Data-level:Acannot distinguish
plaintext embeddings, queries, or result rankings with non-
negligible advantage; (2)Access-pattern-level: the server ob-
serves only the perturbed leakageLinduced by random access
perturbation, and any graph reconstruction attempt is therefore
limited to what can be inferred from this noisy view. Formally,
for any two datasetsV 0,V1and queriesq 0, q1of equal size, the
viewsView A(V0, q0)andView A(V1, q1)are computationally
indistinguishable, conditioned on equivalent perturbed leakage
L(V 0, q0) =L(V 1, q1).
This two-tier definition captures four properties: data pri-
vacy through IND-CPA indistinguishability of ciphertexts,
query privacy through indistinguishability of query intents,
result privacy through indistinguishability of scores and rank-
ings, and access-pattern mitigation through perturbed leakage.
To formalize how OEE masks fine-grained ranking infor-
mation, lets i=⟨q, d i⟩be true cosine similarities and let the
server observe approximationss′
i=si+ξi, where|ξ i| ≤ϵ OEE.
Define the ranking ambiguity margin∆ := 2ϵ OEE.
Lemma VI.4(Ranking Indistinguishability under OEE).For
any two documentsd i, dj, if|s i−sj|<∆, then the
cloud server cannot determine their relative order with non-
negligible advantage.
The lemma states that OEE converts sufficiently small score
gaps into ranking ambiguity, so close candidates do not reveal
stable semantic orderings to the server. The full proof is
deferred to Appendix B.
2) Security of PRAG-I and PRAG-II:In PRAG-I, all op-
erations are performed homomorphically using Chebyshev
polynomial approximations to replace comparisons.
Theorem VI.5.Assuming CKKS is IND-CPA secure and the
OEE is bounded such that ranking distortion is limited by an
ambiguity margin∆, PRAG-I achieves semantic confidentiality
with leakageL.
Proof sketch.The proof uses a standard hybrid sequence. We
first replace real encryptions with encryptions of random vec-

10
TABLE V
COMPLEXITY OFSECURESIMILARITYSEARCHSCHEMES
Scheme Setup Retrieval Update Communication
SESR [25]O(|D| ·dlog|D|)O(dlog|D|)NAO(dlog|D|)
PDQ [26]O(|D| ·d)O(|D| ·d)NAO(|D| ·d)
MSecKNN [28]O(|D| ·d)O(|D|(d+ log2|D|))O(d)O(|D|(d+ log2|D|))
FSkNN [29]O(|D| ·d)O(|D|(d+k))O(d)O(|D| ·k)
PRAGO(|D| ·dlog|D|)O(d)O(d)O(d)
Notations:|D|: dataset size;d: vector dimension;k: top-kretrieval parameter (FSkNN).
tors and then simulate homomorphic additions and multiplica-
tions on these random ciphertexts; IND-CPA security of CKKS
makes both transitions negligible. We then use Lemma VI.4
to show that Chebyshev-driven routing decisions reveal only
ambiguity-preserving order information when score gaps fall
inside the OEE margin. Finally, a simulator that depends only
on perturbed leakageLreproduces the observed paths and
intermediates, so the adversary’s final view is independent
of the underlying plaintexts up to negligible advantage. A
complete proof is given in Appendix B.
PRAG-II uses client-assisted decryption to resolve com-
parisons, introducing bounded additional leakage of local
orderings.
Theorem VI.6.Assuming CKKS is IND-CPA secure, PRAG-
II achieves semantic confidentiality with leakageLextended
to include revealed orderings of interacted candidates.
Proof sketch.The hybrid structure is the same as in The-
orem 1 for all non-interactive ciphertext operations. The
difference is that the simulator must additionally reproduce
the bounded local orderings revealed by client-assisted com-
parisons, such as selected clusters or candidate neighbors
at a given HNSW layer. Because the server observes only
the choices and not the decrypted values themselves, these
interactions can be simulated as orderings over candidate
sets whose sizes match the protocol, yielding only localized
leakage and no exposure of global plaintext rankings. Thus,
PRAG-II preserves data and query privacy while incurring
only the explicitly modeled ordering leakage. The full proof
appears in Appendix B.
3) Proof of Access-Pattern Mitigation:While Theorems 1
and 2 establish data-level semantic confidentiality, the server
can still observe retrieval paths over encrypted HNSW graphs.
In PRAG-I, this leakage is operationally mitigated by query
obfuscation with dummy traversals and by periodic re-
encryption/re-clustering across epochs, as described in Sec-
tion V-A. We now formalize the security of this mitigation
and bound the adversary’s graph-reconstruction advantage.
We now formally prove that the combination of query
obfuscation and periodic re-encryption effectively defends
against Leakage-Abuse Attacks (LAAs).
Theorem VI.7(LAA Resilience).LetAbe a PPT adversary
that observesQ Equeries within a single re-encryption epoch
of lengthE. Under PRAG’s access-pattern mitigation withobfuscation rateρand epoch lengthE, the adversary’s ad-
vantage in reconstructing the HNSW graph topology satisfies:
Advgraph-recon
A ≤QE
(1 +ρ)· |D|+ negl(λ).(4)
Proof sketch.The analysis again proceeds by hybrids. First,
dummy traversals are computationally indistinguishable from
real traversals because both execute the same encrypted op-
erations over ciphertext nodes; thus the adversary cannot
reliably separate them. Second, periodic re-encryption and
re-clustering make different epochs statistically independent,
so graph evidence cannot be accumulated indefinitely across
epochs. Finally, recovering an edge requires observing it
inside a real traversal, whose expected count is diluted by
both the dataset size and the obfuscation factor, yielding the
boundQE
(1+ρ)|D|+ negl(λ). The detailed proof is deferred to
Appendix B.
Corollary VI.8(Epoch Length Selection).To achieve graph
reconstruction advantage at mostαfor a dataset of size|D|,
the epoch length should satisfyE≤α·(1 +ρ)· |D|queries.
For the 100,000-sample setting used in Section VII-B, taking
ρ= 0.3andα= 10−2givesE≤1,300queries per epoch,
placing the experimentally evaluatedE= 1,000setting within
the theoretical bound.
RemarkVI.9 (Security Interpretation).Combining Theo-
rems 1–3 with the empirical results in Section VII-B, we
conclude that PRAG achieves semantic confidentiality while
making access-pattern-based graph reconstruction attacks dif-
ficult both theoretically and empirically.
B. Complexity Analysis
In this subsection, we first analyze the computational com-
plexity and then investigate the communication complexity.
Setup.PRAG builds a two-level encrypted index with setup
complexityO(|D| ·dlog|D|), covering encrypted clustering
initialization and per-cluster HNSW construction. Thelog|D|
factor arises from HNSW’s probabilistic layering, which re-
quires each of the|D|vectors to be inserted acrossO(log|D|)
layers. As shown in Table V, this is comparable to SESR,
which also relies on hierarchical index construction. By con-
trast, MSecKNN, FSkNN, and PDQ have setup complexity
O(|D| ·d), because they avoid hierarchical ANN index con-
struction. PRAG therefore incurs higher upfront indexing cost
but enables substantially more efficient sublinear retrieval.

11
Retrieval.As summarized in Table V, PRAG achieves
sublinear retrieval complexityO(d)by performing encrypted
hierarchical navigation over cluster-level centroids followed
by localized HNSW graph traversal. In contrast, SESR has
complexityO(dlog|D|)with logarithmic dependency on the
full dataset size and lacks support for encrypted similarity
ranking within graph-based ANN structures. More importantly,
PDQ, MSecKNN, and FSkNN remain effectively linear in the
dataset size during query processing, with costsO(|D| ·d),
O(|D|(d+ log2|D|)), andO(|D|(d+k)), respectively. This
reflects the fact that they evaluate secure distance computa-
tions over the outsourced dataset without a hierarchical ANN
pruning mechanism, making them less suitable than PRAG for
large-scale RAG retrieval.
Update.PRAG supports efficient incremental updates with
complexityO(d), independent of the dataset size|D|, by
enabling localized encrypted insertions into existing HNSW
graphs. In contrast, SESR and PDQ do not support real-time
updates in the compared settings. MSecKNN and FSkNN have
record-level update costs ofO(d)because they do not maintain
a hierarchical encrypted index, but this advantage comes with
a structural trade-off: query processing continues to scale
linearly with the outsourced dataset rather than benefiting from
localized ANN navigation.
Communication.PRAG incurs communication overhead of
onlyO(d)per query, as it transmits a single encrypted query
vector and returns encrypted similarity scores. SESR increases
communication cost toO(dlog|D|)to support unlinkability
through multi-path traversal. By comparison, PDQ requires
O(|D| ·d)communication, MSecKNN requiresO(|D|(d+
log2|D|)), and FSkNN requiresO(|D|·k), since these schemes
exchange information proportional to the outsourced dataset
during secure query evaluation. This makes PRAG consid-
erably more communication-efficient for cloud-scale RAG
workloads.
VII. EVALUATIONS
Experimental Setup.We evaluate PRAG on a 100,000-
sample subset of TriviaQA [50], which serves as the default
dataset scale throughout this section. All experiments are
conducted on an Ubuntu 24.04.2 LTS server equipped with
8 Intel Xeon Gold vCPUs, 200 GB RAM, and 500 GB SSD.
For secure computation, we adopt the CKKS scheme with
a 128-bit security level. Unless otherwise specified, retrieval
and communication use a top-10 setting, and the graph-
reconstruction study usesQ= 10,000queries.
Implementation.The PRAG prototype is a pure
C++ implementation developed on top of the Microsoft
SEAL 4.1 library. The source code are available at
https://github.com/richikun2014-bit/PRAG. We utilize
LangChain for initial document chunking and OpenAI Emb
for embedding generation, with Qwen-3 serving as the
response generation model1. A key technical feature of
our implementation is the use of Chebyshev polynomials
specifically for encrypted-domain numerical comparisons;
1https://huggingface.co/Qwen/Qwen3-32B-GGUFthis mechanism facilitates secure decision making during
HNSW hierarchical navigation and routing path selection.
A. System Performance
We evaluate PRAG in both PRAG-I and PRAG-II con-
figurations against baselines that satisfy currently acceptable
security assumptions, focusing on retrieval efficiency, update
cost, communication overhead, and resilience to graph recon-
struction attacks. Unless otherwise specified, the retrieval and
communication results reported below use a top-10 retrieval
setting. Following this criterion, Huang et al. [23] and SPE-
Sim [24] are excluded from experimental baselines due to
known security weaknesses.
Setup.FSkNN at25.902s and MSecKNN at44.678s show
the fastest setup, as illustrated in Fig. 3(a). This advantage
mainly comes from their lightweight preprocessing pipelines,
which avoid costly tree/graph index construction and rely on
simpler encryption/partition routines. In contrast, PRAG-II at
289.2956s, PRAG-I at398.0s, PDQ at217.524s, and SESR
at986.385s incur higher upfront construction overhead due to
more complex secure indexing procedures. Nevertheless, this
additional setup cost in PRAG is amortized by substantially
lower retrieval and communication overhead in the online
phase, especially compared with heavier alternatives such as
PDQ and SESR.
Retrieval.PRAG-I achieves the best retrieval latency at
1.29s, followed by PRAG-II at7.91s, as shown in Fig. 3(b).
Both outperform SESR at166.11s, PDQ at4107.111s,
MSecKNN at2137.65s, and FSkNN at1978s. PDQ is sub-
stantially slower because it executes a full comparison circuit
entirely in the ciphertext domain. MSecKNN and FSkNN are
also slower because secure ranking requires many cross-server
interaction rounds. Overall, PRAG achieves the most favorable
retrieval efficiency among the compared secure schemes.
0123456789 1 010-1100101102103Build time (s)
Number of data entries  (×104) FSkNN
 MSecKNN
  PDQ
 SESR
 PRAG-I 
 PRAG-II 
(a) Setup time
0123456789 1 010-410-310-210-1100101102103104Query time (s)
Number of data entries  (×104) FSkNN
 MSecKNN
  PDQ
 SESR
 PRAG-I 
 PRAG-II (b) Retrieval time
Fig. 3. Setup time and retrieval time comparison across schemes.
Update.PRAG-II achieves5.44ms and PRAG-I achieves
7.28ms for updates as shown in Fig. 4(a), while FSkNN and
MSecKNN are about7.42ms and12.198ms. PDQ and SESR
do not support updates.
Communication.PRAG-I achieves the lowest communica-
tion cost at4.1175MB, as shown in Fig. 4(b). SESR and
PDQ follow at8.3055MB and41.07111MB, while PRAG-II
reaches78.5478MB due to layer-wise client-assisted interac-
tions during encrypted traversal. FSkNN and MSecKNN are
much higher at434.7MB and3206.475MB, respectively,
because their secure ranking procedures require substantially

12
0123456789 1 048121620Update time (ms)
Number of data entries  (×104) FSkNN
 MSecKNN
 PRAG-I 
 PRAG-II 
(a) Update time
0123456789 1 010-310-210-1100101102103Amount of traffic (MB)
Number of data entries  (×104) FSkNN
 MSecKNN
  PDQ
 SESR
 PRAG-I 
 PRAG-II (b) Communication cost
Fig. 4. Update time and communication cost comparison across schemes.
TABLE VI
ADVERSARY’SGRAPHRECONSTRUCTIONSUCCESS UNDERPRAG’S
ACCESS-PATTERNMITIGATION.
ρEpochEEdge
RecoveryFalse
PositiveRecall
@10Latency
Overhead
0∞34.7% 8.2% 72.45% 1.00×
0.1∞18.3% 21.6% 72.31% 1.10×
0.2∞11.5% 35.4% 72.18% 1.19×
0.3∞6.8% 48.7% 71.92% 1.28×
0.5∞3.1% 62.3% 71.54% 1.47×
0 5000 19.2% 14.5% 72.45% 1.00×
0 2000 9.6% 28.3% 72.45% 1.00×
0 1000 5.1% 41.7% 72.45% 1.00×
0.3 2000 2.4% 71.8% 71.92% 1.28×
0.3 1000 1.1% 83.5% 71.92% 1.28×
EpochE=∞denotes no periodic re-encryption.
more cross-party message exchanges. Overall, PRAG-I pro-
vides the most communication-efficient retrieval path, and
PRAG-II preserves stronger ranking fidelity with moderate but
still practical bandwidth overhead.
PRAG offers the strongest end-to-end trade-off across setup,
retrieval, update, and communication. Although its offline
setup cost is higher than FSkNN and MSecKNN, it substan-
tially reduces online retrieval latency and bandwidth, which
dominate continuous RAG serving. Compared with PDQ and
SESR, PRAG also avoids severe update bottlenecks. Within
PRAG, PRAG-I is preferable for latency- and bandwidth-
sensitive deployments, while PRAG-II is preferable when
stronger ranking fidelity is required with still-practical commu-
nication overhead. This dual-mode design improves deploya-
bility over single-regime alternatives in cloud RAG workloads.
B. Resilience Against Graph Reconstruction Attacks
We evaluate PRAG’s access-pattern mitigation (Sec-
tion VI-A3) by simulating an IHOP-style leakage-abuse ad-
versary [51]–[53] that reconstructs the encrypted HNSW graph
from observed traversal paths. We report edge recovery rate,
false positive rate, and Recall@10 degradation under obfusca-
tion ratesρ∈ {0,0.1,0.2,0.3,0.5}and re-encryption epochs
E∈ {∞,5000,2000,1000}.
Table VI shows that unmitigated access leakage is substan-
tial: atρ= 0andE=∞, the adversary recovers 34.7%
of true edges. Either defense already helps: query obfuscation
withρ= 0.3lowers recovery to 6.8% at a modest Recall@10drop from 72.45% to 71.54% and 1.28×latency overhead,
while periodic re-encryption withE= 1,000lowers recovery
to 5.1% without affecting retrieval quality. When both are
enabled (ρ= 0.3,E= 1,000), recovery falls to 1.1%
and the false positive rate rises to 83.5%, indicating that
the inferred graph is dominated by spurious edges. Overall,
PRAG’s mitigation sharply limits graph reconstruction while
preserving retrieval utility, consistent with Theorem 3.
VIII. CONCLUSION ANDFUTUREWORK
We present PRAG, a privacy-preserving RAG framework
for untrusted clouds that combines dual-mode encrypted re-
trieval, OEE-based ranking stabilization, and access-pattern
mitigation. Experiments and formal analysis show that PRAG
achieves practical latency, efficient updates, competitive re-
call, and stronger resilience to graph reconstruction attacks
while preserving end-to-end confidentiality. PRAG currently
assumes a single-client key setting. Future work will extend it
to multi-key secure computation and further reduce homomor-
phic overhead through algorithmic and hardware acceleration.
REFERENCES
[1] N. I. Khan and V . Filkov, “Evidencebot: A privacy-preserving, customiz-
able rag-based tool for enhancing large language model interactions,”
inProceedings of the 33rd ACM International Conference on the
Foundations of Software Engineering, 2025, pp. 1188–1192.
[2] Y . Wei, P. Xia, Y . Ni, and J. Li, “Privacy-preserving llm-based rag
with split inference and masked privacy recovery,” in2025 IEEE/CIC
International Conference on Communications in China (ICCC). IEEE,
2025, pp. 1–6.
[3] A. Xu, T. Yu, M. Du, P. Gundecha, Y . Guo, X. Zhu, M. Wang, P. Li,
and X. Chen, “Generative ai and retrieval-augmented generation (rag)
systems for enterprise,” inProceedings of the 33rd ACM International
Conference on Information and Knowledge Management, ser. CIKM
’24. New York, NY , USA: Association for Computing Machinery,
2024, pp. 5599–602. [Online]. Available: https://doi.org/10.1145/3627
673.3680117
[4] X. Zhao, T. Sun, S. Ren, J. Yang, and Y . Liu, “Rag-based ai agents for
enterprise software development: Implementation patterns and produc-
tion deployment,”Frontiers in Artificial Intelligence Research, vol. 2,
no. 3, pp. 501–520, 2025.
[5] T. Yu, W. Zhou, L. Leiyang, A. Shukla, M. Mmadugula, P. Gundecha,
N. Burnett, A. Xu, V . Viseth, T. Tbaret al., “Ekrag: Benchmark rag for
enterprise knowledge question answering,” inProceedings of the 4th
International Workshop on Knowledge-Augmented Methods for Natural
Language Processing, 2025, pp. 152–159.
[6] T. B. Weerasekara, C. Chandeepa, O. S. Amarasuriya, and C. Het-
tiarachchi, “Privacy-preserving medical advising system on mobile
devices: On-device phi anonymization, medical report retrieval, and
cloud-based rag,” in2025 IEEE/ACM Conference on Connected Health:
Applications, Systems and Engineering Technologies (CHASE). IEEE,
2025, pp. 447–452.
[7] Y . Cheng, L. Zhang, J. Wang, M. Yuan, and Y . Yao, “Remoterag: A
privacy-preserving llm cloud rag service,” inFindings of the Association
for Computational Linguistics: ACL 2025, 2025, pp. 3820–3837.
[8] Amazon Web Services, “Knowledge bases for amazon bedrock,” https:
//docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html,
Amazon.com, Inc., 2026.
[9] Google Cloud, “Rag engine overview,” https://docs.cloud.google.com/
vertex-ai/generative-ai/docs/rag-engine/rag-overview?, Google LLC,
2026.
[10] Alibaba Cloud, “What is model studio,” https://help.aliyun.com/zh/mod
el-studio/what-is-model-studio, Alibaba Group Holding Limited, 2026.
[11] C. Team, “Mother of all breaches (moab) reveals 26 billion records,”
https://cybernews.com/security/billions-passwords-credentials-leaked-m
other-of-all-breaches/, Cybernews, 2024.
[12] Western Sydney University, “Cyber incident update: October 23, 2025,”
https://www.westernsydney.edu.au/news/cyber-details/october-23-2025,
Western Sydney University, 2025.

13
[13] G. Radauskas, “Huawei source code and data breach reported,” https://
cybernews.com/security/huawei-source-code-data-breach/, Cybernews,
2025.
[14] Z. Doffman, “13 billion unique passwords exposed in extensive data
leak,” https://www.forbes.com/sites/zakdoffman/2025/11/06/13-billion
-unique-passwords-exposed-in-extensive-data-leak/, Forbes, 2025.
[15] L. Abrams, “Sedgwick confirms breach at government contractor sub-
sidiary,” https://www.bleepingcomputer.com/news/security/sed
gwick- confirms- breach- at- government- contractor- subsidiary/,
BleepingComputer, 2026.
[16] N. Grislain, “Rag with differential privacy,” in2025 IEEE Conference
on Artificial Intelligence (CAI). IEEE, 2025, pp. 847–852.
[17] T. Koga, R. Wu, and K. Chaudhuri, “Privacy-preserving retrieval-
augmented generation with differential privacy,”arXiv preprint
arXiv:2412.04697, 2024.
[18] J. Yu, J. Zhou, Y . Ding, L. Zhang, Y . Guo, and H. Sato, “Textual
differential privacy for context-aware reasoning with large language
model,” in2024 IEEE 48th Annual Computers, Software, and Appli-
cations Conference (COMPSAC). IEEE, 2024, pp. 988–997.
[19] H. Chen, I. Chillotti, Y . Dong, O. Poburinnayaet al., “Sanns: Scaling up
secure approximate k-nearest neighbors search,”29th USENIX Security
Symposium (USENIX Security 20), pp. 1515–1532, 2020.
[20] A. Henzinger, E. Dauterman, H. Corrigan-Gibbs, and N. Zeldovich,
“Private web search with tiptoe,”Proceedings of the 29th Symposium
on Operating Systems Principles, 2023. [Online]. Available: https:
//api.semanticscholar.org/CorpusID:263304868
[21] M. Zhou, E. Shi, and G. Fanti, “Pacmann: Efficient private approximate
nearest neighbor search,”IACR Cryptol. ePrint Arch., vol. 2024, p. 1600,
2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:
273202108
[22] M. Chrapek, A. Vahldiek-Oberwagner, M. Spoczynski, S. Constable,
M. Vij, and T. Hoefler, “Fortify your foundations: Practical privacy and
security for foundation model deployments in the cloud,”arXiv preprint
arXiv:2410.05930, 2024.
[23] Z. Huang, M. Zhang, and Y . Zhang, “Toward efficient encrypted image
retrieval in cloud environment,”IEEE Access, vol. 7, pp. 174 541–
174 550, 2019.
[24] Y . Zheng, R. Lu, Y . Guan, J. Shao, and H. Zhu, “Achieving efficient and
privacy-preserving exact set similarity search over encrypted data,”IEEE
Transactions on Dependable and Secure Computing, vol. 19, no. 2, pp.
1090–1103, 2020.
[25] N. Wang, W. Zhou, J. Wang, Y . Guo, J. Fu, and J. Liu, “Secure and
efficient similarity retrieval in cloud computing based on homomorphic
encryption,”IEEE Transactions on Information Forensics and Security,
vol. 19, pp. 2454–2469, 2024.
[26] B. H. M. Tan, H. T. Lee, H. Wang, S. Ren, and K. M. M. Aung,
“Efficient private comparison queries over encrypted databases using
fully homomorphic encryption with finite fields,”IEEE Transactions on
Dependable and Secure Computing, vol. 18, no. 6, pp. 2861–2874, 2021.
[27] R. Li, A. X. Liu, Y . Liu, H. Xu, and H. Yuan, “Insecurity and hardness of
nearest neighbor queries over encrypted data,” inProc. IEEE Conference
on Data Engineering (ICDE’19), 2019, pp. 1614–1617.
[28] Z. Li, H. Wang, W. Zhang, Y . Su, and W. Susilo, “Msecknn:
Maliciously secure outsourced knn classification under multiple
distance metrics,”IEEE Transactions on Information Forensics and
Security, vol. 20, pp. 11 279–11 294, 2025. [Online]. Available:
https://api.semanticscholar.org/CorpusID:282161289
[29] Y . Fukuchi, S. Hashimoto, K. Sakai, S. Fukumoto, M.-T. Sun,
and W.-S. Ku, “Secure knn for distributed cloud environment
using fully homomorphic encryption,”IEEE Transactions on Cloud
Computing, vol. 13, pp. 721–736, 2025. [Online]. Available: https:
//api.semanticscholar.org/CorpusID:277872246
[30] J.-W. Lee, E. Lee, Y . Lee, Y .-S. Kim, and J.-S. No, “High-
precision bootstrapping of rns-ckks homomorphic encryption using
optimal minimax polynomial approximation and inverse sine function,”
inInternational Conference on the Theory and Application of
Cryptographic Techniques, 2021. [Online]. Available: https://api.sema
nticscholar.org/CorpusID:223605897
[31] S. Bae, J. H. Cheon, A. Kim, and D. Stehl ´e, “Bootstrapping bits with
ckks,” inAdvances in Cryptology – EUROCRYPT 2024, ser. Lecture
Notes in Computer Science. Springer, 2024.
[32] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
and P. Liang, “Lost in the middle: How language models use long
contexts,” 2023. [Online]. Available: https://arxiv.org/abs/2307.03172
[33] T. Yu, A. Xu, and R. Akkiraju, “In defense of rag in the
era of long-context language models,” 2024. [Online]. Available:
https://arxiv.org/abs/2409.01666[34] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Kuttler, M. Lewis, W. tau Yih, T. Rocktaschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive nlp
tasks.”Advances in Neural Information Processing Systems (NeurIPS),
vol. 33, pp. 9459–9474, 2020.
[35] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Yu, Y . Yang, J. Callan,
and G. Neubig, “Active retrieval augmented generation,” 2023.
[36] Y . Huang, S. Gupta, Z. Zhong, K. Li, and D. Chen, “Privacy implications
of retrieval-based language models,” in2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023. Association
for Computational Linguistics (ACL), 2023, pp. 14 887–14 902.
[37] S. Zeng, J. Zhang, P. He, Y . Xing, Y . Liu, H. Xu, J. Ren, S. Wang, D. Yin,
Y . Changet al., “The good and the bad: Exploring privacy issues in
retrieval-augmented generation (rag),”arXiv preprint arXiv:2402.16893,
2024.
[38] J. He, C. Liu, G. Hou, W. Jiang, and J. Li, “Press: Defending privacy
in retrieval-augmented generation via embedding space shifting,” in
ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.
[39] D. Zhao, “Frag: Toward federated vector database management for
collaborative and secure retrieval-augmented generation,”arXiv preprint
arXiv:2410.13272, 2024.
[40] T. E Andersen, A. M. Avalos, G. G Dagher, and M. Long, “D-rag: A
privacy-preserving framework for decentralized rag using blockchain,”
2025.
[41] P. Zhou, Y . Feng, and Z. Yang, “Privacy-aware rag: Secure and isolated
knowledge retrieval,”arXiv preprint arXiv:2503.15548, 2025.
[42] Z. Xia, Q. Ji, Q. Gu, C. Yuan, and F. Xiao, “A format-compatible
searchable encryption scheme for jpeg images using bag-of-words,”
ACM Transactions on Multimedia Computing, Communications, and
Applications (TOMM), vol. 18, no. 3, pp. 1–18, 2022.
[43] E. Stefanov, C. Papamanthou, and E. Shi, “Practical dynamic searchable
encryption with small leakage.” inProc. ISOC Network and Distributed
System Security Symposium (NDSS’14), vol. 71, 2014, pp. 72–75.
[44] M. Etemad, A. K ¨upc ¸¨u, C. Papamanthou, and D. Evans, “Efficient
dynamic searchable encryption with forward privacy,”Proc. Privacy
Enhancing Technologies, vol. 1, pp. 5–20, 2018.
[45] H. Dou, Z. Dan, P. Xu, W. Wang, S. Xu, T. Chen, and H. Jin, “Dynamic
searchable symmetric encryption with strong security and robustness,”
IEEE Transactions on Information Forensics and Security, 2024.
[46] Z. Li, J. Ma, Y . Miao, X. Wang, J. Li, and C. Xu, “Enabling efficient
privacy-preserving spatio-temporal location-based services for smart
cities,”IEEE Internet of Things Journal, 2023.
[47] Z. Wang, X. Ding, J. Lu, L. Zhang, P. Zhou, K.-K. R. Choo, and
H. Jin, “Efficient location-based skyline queries with secure r-tree over
encrypted data,”IEEE Transactions on Knowledge and Data Engineer-
ing, 2023.
[48] P. Sun, D. Simcha, D. Dopson, R. Guo, and S. Kumar, “Soar: Improved
indexing for approximate nearest neighbor search,” inAdvances in
Neural Information Processing Systems (NeurIPS’23), vol. 36. Curran
Associates, Inc., 2023, pp. 3189–3204.
[49] H. Lee, S. Jang, J. Gwak, J. Park, and Y . Kim, “Bit-level semantics:
Scalable rag retrieval with neurosymbolic hyperdimensional computing,”
in2025 34th International Conference on Parallel Architectures and
Compilation Techniques (PACT). IEEE, 2025, pp. 347–358.
[50] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension,”
arXiv preprint arXiv:1705.03551, 2017.
[51] S. Oya and F. Kerschbaum, “IHOP: Improved statistical query recovery
against searchable symmetric encryption through quadratic optimiza-
tion,” in31st USENIX Security Symposium (USENIX Security 22), 2022,
pp. 2407–2424.
[52] M. Damie, F. Hahn, and A. Peter, “A highly accurate query-recovery
attack against searchable encryption using non-indexed documents,” in
30th USENIX Security Symposium (USENIX Security 21), 2021, pp.
143–160.
[53] E. M. Kornaropoulos, C. Papamanthou, and R. Tamassia, “The state of
the uniform: Attacks on encrypted databases beyond the uniform query
distribution,” inIEEE Symposium on Security and Privacy (S&P), 2021,
pp. 1223–1240.
[54] G. Kellaris, G. Kollios, K. Nissim, and A. O’Neill, “Generic attacks on
secure outsourced databases,” inProceedings of the 2016 ACM SIGSAC
Conference on Computer and Communications Security (CCS), 2016,
pp. 1329–1340.

14
APPENDIXA
LEAKAGEFUNCTION
Definition A.1(Leakage Function (with Access-Pattern Per-
turbation)).The leakage functionL= (L setup,Lquery)charac-
terizes theperturbedside-channel information observable by
the adversary after applying PRAG’s access-pattern mitigation
(Section VI-A3):
Setup Phase:
Lsetup(D) =
|D|, d, C,n
C(t)
joC,T
j=1,t=1
(5)
where|D|is the dataset size,dis the embedding dimension,C
is the number of clusters, andC(t)
jcaptures cluster assignment
patterns at iterationt. Note that setup leakage is invalidated
after each re-encryption epoch.
Query Phase (Perturbed):
Lquery(q(i),D) = 
eJ(i)
probe,
gPath(i)
j
j∈eJ(i)
probe, K!
(6)
where eJ(i)
probe =J(i)
probe∪ J(i)
dummy includes both real probed
clusters and dummy cluster probes, gPath(i)
jincludes real
traversal paths padded to fixed lengthℓ max and interspersed
with dummy random walks, andKis the result set size. The
adversary cannot distinguish real paths from dummy paths.
Crucially,Ldoes NOT include: plaintext embeddings{v i},
queries{q(i)}, similarity scores{⟨q(i),vj⟩}, exact ranking
relationships, or the distinction between real and dummy
accesses.
APPENDIXB
SUPPLEMENTARYSECURITYPROOFS
A. Proof of Lemma VI.4
Proof.The encrypted similarity difference satisfies
s′
i−s′
j= (s i−sj) + (ξ i−ξj).
By the triangle inequality,
|(ξi−ξj)| ≤2ϵ OEE= ∆.
Hence, when|s i−sj|<∆, the sign ofs′
i−s′
jis dominated
by encryption noise and approximation error. As a result, the
probability thats′
i> s′
jis negligibly close to the probability
thats′
j> s′
i.
Therefore, the relative ordering ofd iandd jis statistically
indistinguishable from a random permutation from the adver-
sary’s perspective.
B. Proof of Theorem 1
Proof.We prove via a sequence of hybrid games that the
adversary’s advantage in breaking semantic confidentiality is
negligible.
Game 0:Real execution.AobservesCT V={Enc(v i)},
encrypted centroids, HNSW graphs, and intermediates from
Chebyshev-based clustering and retrieval.
Game 1:Replace all encryptions with encryptions of random
vectorsr i. By IND-CPA of CKKS, the difference inA’s
success probability between Game 0 and Game 1 is negligible.Game 2:Simulate homomorphic operations on random ci-
phertexts. Since homomorphic additions and multiplications
preserve IND-CPA, and the Chebyshev surrogate is a polyno-
mial function, intermediates remain indistinguishable.
Ranking Hiding via OEE-Induced Ambiguity.By
Lemma VI.4, whenever two candidates fall within the∆-
margin, the cloud server cannot determine their relative order
with non-negligible advantage.
Implication for Access Patterns.All routing decisions in
clustering and HNSW traversal are driven by Chebyshev-
derived weights computed from{s′
i}. By Lemma VI.4,
whenever competing candidates fall within the∆-margin,
these routing decisions behave as randomized choices. Conse-
quently, the observed access patterns are statistically indepen-
dent of the true semantic ranking, up to negligible probability.
Game 3:Simulate access patterns using a leakage simulator
SimLthat generates paths based solely on index size and
structure, such as random cluster probes and graph traversals
matching the real distribution. Since real paths depend only
on approximated Chebyshev-derived weights rather than exact
rankings, and OEE ensures that noise and approximation
bounds∆make close rankings ambiguous, the simulated
paths are statistically close to real ones. The difference inA’s
success probability between Game 2 and Game 3 is negligible.
In Game 3,A’s view is independent of plaintexts, proving
data and query privacy. Result privacy follows from ranking
ambiguity: since the Chebyshev surrogate smooths decisions,
exact orderings are not revealed, and∆bounds inferable
semantic relationships.
Thus,Advsem-conf
A ≤negl(λ).
C. Proof of Theorem 2
Proof.The proof follows a similar game sequence as in
Theorem 1. The key difference is that during clustering, inser-
tion, and retrieval, the client decrypts limited intermediates to
resolve comparisons, revealing exact local orderings such as
the selected cluster and the nearest neighbor.
Game 0:Real PRAG-II execution. During clustering and
insertion, for each distance set{ct dij}, the client decrypts and
returns exact assignments or links. For retrieval, per layer, the
client decrypts candidate distances and returns exact nearest or
top-Mchoices.Aobserves these choices, such as the selected
clusterj∗and the entry points.
Game 1:As in PRAG-I Game 1, replace encryptions with ran-
dom vectors. By IND-CPA of CKKS,|Pr[Awins Game 0]−
Pr[Awins Game 1]| ≤negl(λ).
Game 2:Simulate homomorphic computations such as
HomoDist. Identical to PRAG-I Game 2: the difference is
0.
Game 3:Simulate access patterns and interacted orderings
withSim L. For non-interacted parts, use the simulation from
PRAG-I. For interactions, sinceAdoes not observe decrypted
distances and observes only the resulting choices, simulate
choices as random orderings over candidate sets whose sizes
match the real protocol, such as sizeCfor clusters and
layer-dependent sizes for HNSW. Leakage is bounded: per
query, at mostO(L)orderings are revealed, whereLis

15
the maximum number of HNSW layers, and these orderings
are localized to small candidate sets. Since candidates are
random from Game 1, revealed orderings do not correlate with
global plaintext rankings, and thus|Pr[Awins Game 2]−
Pr[Awins Game 3]| ≤negl(λ).
In Game 3, the adversary’s view is independent of plaintexts
except for explicit, bounded ordering leakage, proving data and
query privacy. Result privacy holds because the final top-kset
is computed client-side and never revealed to the server, while
intermediate ciphertexts hide similarity scores.
The non-interactive encryption components are identical to
those in PRAG-I, so their security follows directly. Therefore,
Advsem-conf
A ≤negl(λ).
D. Proof of Theorem 3
Proof.We prove via a sequence of games that the adversary’s
graph reconstruction advantage is bounded.
Game 0 (Real Execution with Mitigation):The adversary
observes the perturbed access patternsPath obsforQ Equeries
within a single epoch. Each observation consists of real
traversal paths mixed with aρ-fraction of dummy paths.
Game 1 (Indistinguishability of Real and Dummy Paths):
We replace all real query traversal paths with independent
random walks. Since CKKS ciphertexts are IND-CPA secure,
the computational operations performed during real traversals
and dummy traversals are indistinguishable to the server.
Specifically, both real and dummy accesses involve reading
encrypted nodes, computing homomorphic inner products,
and returning encrypted results. Without the secret key, the
server cannot determine whether a traversal follows the greedy
HNSW routing or a random walk. The distinguishing advan-
tage between Game 0 and Game 1 is bounded bynegl(λ)
via a standard hybrid argument over theQ E·(1 +ρ)·ℓ max
ciphertext operations.
Game 2 (Independent Epochs):Since each epoch uses fresh
keys, fresh random seeds for clustering, and fresh HNSW level
assignments, the access patterns in epocheare statistically
independent of the graph structure in epoche′̸=e. Thus, the
adversary’s observation is limited toQ Equeries per epoch.
Graph Reconstruction Bound:Consider the adversary’s task
of recovering an edge(u, v)in the HNSW graph. An edge
can only be detected if bothuandvappear consecutively in
a real non-dummy traversal path. Within a single epoch, the
probability that a specific edge is traversed by a random query
is at most1
|D|. WithQ Eobserved queries, each containing
(1 +ρ)-diluted paths, the expected number of observations of
any specific edge is
E[obs(u, v)]≤QE
(1 +ρ)· |D|.(7)
For LAA reconstruction to succeed with high confidence,
the adversary needsΩ(|D|)observations per edge [54]. By
choosing epoch lengthEsuch thatQ E≪(1 +ρ)· |D|, the
adversary’s reconstruction advantage remains negligible.
Combining the three games, we obtain
Advgraph-recon
A ≤QE
(1 +ρ)· |D|+ negl(λ).APPENDIXC
PROTOCOLSSUPPLEMENT
A. Homomorphic Normalization (HomoNorm)
Letct sumdenote a ciphertext encrypting a vectoru∈Rd,
and letct count denote a ciphertext encrypting a positive scalar
c∈R >0. The homomorphic normalization operation, denoted
byHomoNorm, computes
ctµ←HomoNorm(ct sum,ctcount) =Encu
c
,
using a polynomial approximation of the reciprocal function
x7→1/x.
B. Additional Protocol
Algorithm 4:ProtocolΠ LocalRAG
KwIn :Aggregated retrieval results, cluster scores, original query
q, local databaseDB local, and target sizeK.
KwOut :Generated responseR.
1doc score← ∅;
2foreach(j∗,PartialResults)in aggregated resultsdo
3cluster weight←cluster score[j∗];
4foreachID iat positionrankinPartialResultsdo
5rrf score←1/(60 +rank);
6doc score[ID i]←
docscore[ID i] +rrf score·cluster weight;
7end
8end
9Select top-kdocument identifiers fromdoc scoreasCandidates;
10{ID∗
1, . . . ,ID∗
K} ←FuseRerank(q,Candidates, DB local, K);
11Context←””;
12foreachID∗
iin selected identifiersdo
13d∗
i←GetLocalData(DB local,ID∗
i);
14Context←Context⊕ExtractText(d∗
i);
15end
16R←LLM.Gen(Prompt);
17returnR;
APPENDIXD
SUPPLEMENTARYOEE ANALYSIS
This appendix complements the main-text discussion in Sec-
tion V-C by giving the phase-wise derivation behind the noise
counts and mitigation rationale.
A. Understanding CKKS Noise Characteristics
Before comparing noise across different operational phases,
we first characterize how fundamental homomorphic opera-
tions contribute to noise growth in CKKS. Homomorphic ad-
dition exhibits linear, gradual noise growth: Noise(ct 1+ct2)≈
Noise(ct 1)+Noise(ct 2). In contrast, homomorphic multiplica-
tion is the primary driver of noise accumulation, with quadratic
behavior: Noise(ct 1×ct 2)≈C·Noise(ct 1)·Noise(ct 2).
Rescaling operations, required after multiplication to maintain
numerical stability, introduce minor noise while increasing
relative noise. Rotation operations contribute moderate noise,
typically greater than addition but less severe than multiplica-
tion.
A single homomorphic inner product⟨ct q,ctdb⟩between two
encryptedd-dimensional vectors requires one component-wise
multiplication, one relinearization operation for key switching,

16
one rescaling operation, and approximatelylog(d)rotations
and additions for summation. We denote the noise introduced
by this complete circuit as our basic unitS IP, which serves as
the fundamental measure for comparing computational costs
across different phases.
B. Comparative Analysis of Noise Sources
The total noise in our system originates from two primary
phases: the online retrieval phase and the offline index con-
struction phase. To understand which phase dominates the
noise budget, we analyze the computational complexity of
each in terms of the number of noise-intensive homomorphic
operations.
1) Noise in the Retrieval Phase:A single query execution
involves a sequence of independentHomoIPcomputations.
The total number of such operations consists of two compo-
nents. First, during cluster pruning, the encrypted query vector
is compared with allCencrypted cluster centroids, incurring
C× S IPof noise spread acrossCindependent computations.
Second, during HNSW search within the topC probe selected
clusters, a typical greedy traversal from the top layer to
the bottom involves approximatelyL·log(M)inner product
operations per cluster, whereLdenotes the number of layers
andMrepresents the maximum node degree.
The overall computational cost can be expressed as:
Querycost≈C+C probe·L·log(M)·ct sj (8)
Crucially, the noise from these operations does not accu-
mulate sequentially. Each similarity score is computed in-
dependently. The primary risk here is that the noise in any
singleHomoIPcomputation could be large enough to flip
the ranking between two closely scored vectors. However, the
total computational load for a single query remains relatively
modest, typically in the order of hundreds of operations.
2) Noise in the Index Construction Phase:In stark con-
trast, the offline index construction phase represents a greater
computational burden and is the dominant source of noise
concern. This phase comprises two major components, each
with notably different noise profiles.
K-means Clustering NoiseThe K-means algorithm is the
most computationally intensive component. Each iteration has
two steps. In the assignment step, every data point is compared
with all cluster centroids, requiringN·CindependentHomoIP
computations per iteration. For example, with a database of
105vectors and100clusters, one iteration requires107inner-
product evaluations. The update step computes cluster means
through(|D|/C−1)additions per cluster, followed by scalar
multiplication for normalization. Although less noisy than
assignment, it still contributes to the overall budget.
WithTiterations, the total K-means cost is:
K-means cost=T· |D| ·C·ct sj (9)
HNSW Construction NoiseBuilding the HNSW graph
structure requires inserting each of the|D|data points into
the hierarchical graph. Each insertion involves a layer-wise
greedy search from top to bottom, with neighbor selection at
each layer using search depth Deph. This process requiresapproximatelyL·Dephinner products per insertion. The
cumulative construction cost is:
HNSW cost=|D| ·L·Deph·ctsj (10)
3) Quantitative Comparison:Table IV compares noise
across system operations and highlights a clear disparity in
computational burden. The dominant sources are K-means
iterations and full index construction. Since full index con-
struction is unavoidable, mitigation should prioritize reducing
noise from K-means iterations.
4) Conclusion of Analysis:The analysis shows that index
construction, especially iterative K-means clustering, domi-
nates the required noise budget. The K-means assignment step
alone executes 4–5 orders of magnitude more inner products
than a single query. The resulting volume of homomorphic
multiplications in this offline phase strongly influences the
minimum cryptographic parameters needed for correct system
operation. CKKS parameters should therefore be sized for
offline construction, not for the comparatively light online
query phase. Once sized for construction, the query phase
remains within budget. This implies that mitigation should
focus primarily on noise generated during index construction.