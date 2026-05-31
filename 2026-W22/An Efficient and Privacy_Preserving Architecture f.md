# An Efficient and Privacy-Preserving Architecture for Cross-Institutional Collaborative RAG

**Authors**: Chenxin Mao, Shangyu Liu, Zhenzhe Zheng, Fan Wu, Jie Wu, Guihai Chen

**Published**: 2026-05-25 11:18:19

**PDF URL**: [https://arxiv.org/pdf/2605.25716v1](https://arxiv.org/pdf/2605.25716v1)

## Abstract
Retrieval-Augmented Generation (RAG) empowers LLMs with external knowledge, making cross-institutional domain-specific knowledge base integration a highly promising deployment paradigm. Despite this potential, strict privacy regulations create severe "data silos" that obstruct such collaboration. Building federated RAG systems requires distributed inference, but the Transformer's self-attention mechanism fundamentally conflicts with this by mandating cross-node access to distributed Key-Value caches. To address this challenge, we present FedRAG, a high-throughput, privacy-preserving federated RAG framework. At its core is a novel Scrambled Distributed Attention protocol that utilizes numerically stable feature scrambling and token permutation. By dynamically delegating scrambled computations to collaborating nodes, our system successfully decouples attention execution from data localization without exposing plaintext. Crucially, our approach requires no specialized hardware or model retraining, circumventing the prohibitive latency and communication overheads of cryptographic solutions while robustly defending against intermediate state inversion attacks. Extensive evaluations demonstrate our framework preserves negligible (<0.1\%) model utility degradation and achieves up to a 62$\times$ latency reduction over existing secure baselines, sustaining practical, human-reading throughput for cross-institutional knowledge synergy.

## Full Text


<!-- PDF content starts -->

An Efficient and Privacy-Preserving Architecture
for Cross-Institutional Collaborative RAG
Chenxin Mao†, Shangyu Liu†, Zhenzhe Zheng†, Fan Wu†, Jie Wu‡, Guihai Chen†
†Shanghai Jiao Tong University‡Cloud Computing Research Institute, China Telecom
Abstract
Retrieval-Augmented Generation (RAG) empowers LLMs
with external knowledge, making cross-institutional domain-
specific knowledge base integration a highly promising de-
ployment paradigm. Despite this potential, strict privacy reg-
ulations create severe "data silos" that obstruct such collabo-
ration. Building federated RAG systems requires distributed
inference, but the Transformer’s self-attention mechanism fun-
damentally conflicts with this by mandating cross-node access
to distributed Key-Value caches. To address this challenge,
we present FedRAG, a high-throughput, privacy-preserving
federated RAG framework. At its core is a novel Scrambled
Distributed Attention protocol that utilizes numerically stable
feature scrambling and token permutation. By dynamically
delegating scrambled computations to collaborating nodes,
our system successfully decouples attention execution from
data localization without exposing plaintext. Crucially, our ap-
proach requires no specialized hardware or model retraining,
circumventing the prohibitive latency and communication
overheads of cryptographic solutions while robustly defend-
ing against intermediate state inversion attacks. Extensive
evaluations demonstrate our framework preserves negligible
(<0.1%) model utility degradation and achieves up to a 62 ×
latency reduction over existing secure baselines, sustaining
practical, human-reading throughput for cross-institutional
knowledge synergy.
1 Introduction
In recent years, transformer-based Large Language Models
(LLMs) have achieved unprecedented breakthroughs across
both academia and industry, demonstrating exceptional capa-
bilities in multilingual interaction, text generation, and logical
reasoning [58]. This paradigm shift has led to their seamless
integration into a myriad of applications, significantly opti-
mizing various workflows, such as software development [6],
conversational agents [36], knowledge management [38], and
healthcare [42]. It is foreseeable that as their foundationalcapabilities continue to evolve, LLMs will progressively pen-
etrate more complex, multi-domain applications.
Despite possessing vast pre-trained knowledge bases and
exhibiting remarkable generalization, generalized LLMs en-
counter significant challenges when tasked with domain-
specific or highly time-sensitive knowledge. A prominent
issue is "hallucination," whereby models, lacking adequate
context, are prone to generating plausible yet factually incor-
rect information [56]. Furthermore, maintaining the currency
of the model’s knowledge via continual training entails pro-
hibitive computational and temporal costs. To mitigate these
inherent limitations, the Retrieval-Augmented Generation
(RAG) architecture has emerged as a promising paradigm [29].
By dynamically integrating external, authoritative knowledge
bases, RAG endows LLMs with the ability to access spe-
cialized and current information without requiring parameter
retraining. This approach effectively curtails the incidence
of hallucinations while simultaneously ensuring the accuracy
and timeliness of the generated content.
Beyond open-domain applications that leverage publicly
accessible web data, many organizations increasingly rely on
their proprietary data to construct specialized RAG systems.
However, the efficacy of such RAG systems is intrinsically
contingent upon the scale and comprehensiveness of their
underlying knowledge bases [4]. In highly specialized do-
mains, a single organization typically possesses limited and
fragmented proprietary data, which is insufficient to support
robust, well-rounded LLM reasoning. Consequently, there
is a compelling need to collaborate and synthesize knowl-
edge across multiple institutions to construct a more capable
and comprehensive RAG system. For example, healthcare
providers can enhance complex diagnostics by integrating
electronic medical records from multiple hospitals, financial
institutions can conduct robust risk assessments utilizing trans-
action logs across various banks, and the manufacturing sector
can optimize production processes by analyzing operational
data from multiple factories. Despite a strong mutual will-
ingness to collaborate, organizations are strictly precluded
from directly sharing raw documents due to stringent privacy
1arXiv:2605.25716v1  [cs.CR]  25 May 2026

regulations and the imperative to protect commercial secrets,
which inevitably leads to the formation of "data silos."
Therefore, it is imperative to explore and establish privacy-
preserving Federated RAG frameworks to dismantle these
informational barriers and enable secure, cross-institutional
knowledge synergy. However, developing such collaborative
systems entails addressing three primary challenges:
•Privacy Protection.Early works attempted to preserve
privacy by transmitting model intermediate states rather than
raw text. However, recent studies reveal that the intermedi-
ate states of LLMs exhibit high semantic sparsity, rendering
them vulnerable to inversion attacks capable of reconstructing
partial original text [11]. For instance, V ocabulary Mapping
Attacks (VMA) [44], which rely on traversing the vocabu-
lary, can effectively bypass simple permutation-based protec-
tion schemes on intermediate states, including both sequence-
dimension and hidden-dimension permutations. The emer-
gence of such sophisticated attacks imposes significantly more
stringent requirements on the design of privacy-preserving
algorithms.
•Inference Overhead.Existing privacy-preserving dis-
tributed LLM inference architectures based on cryptographic
methods suffer from prohibitive inference overhead. Even
state-of-the-art frameworks utilizing Secure Multi-Party Com-
putation (MPC), such as PUMA [12], require several min-
utes to generate a single token using an 8-billion parameter
model. Furthermore, excessive communication overhead re-
mains a pressing issue. Taking Homomorphic Encryption
(HE) schemes (e.g., Paillier [37] or CKKS [7]) as an example,
the ciphertext volume typically expands by tens to hundreds
of times compared to the plaintext. For massive intermediate
state tensors, such transmission costs are practically unaccept-
able. The severe latency induced by both computation and
transmission renders cryptography-based secure inference
impractical for real-world RAG systems.
•Universality and Rapid Deployment.Many distributed
secure inference architectures impose stringent prerequisites
on the execution environment or the model itself. Hardware-
based solutions relying on Trusted Execution Environments
(TEEs) [41] necessitate specialized hardware support and
complex environmental attestation configurations, which are
often difficult to satisfy in heterogeneous distributed systems.
Conversely, other secure inference architectures require re-
training the model [26, 39] or substituting specific operators
with approximate counterparts. However, retraining hinders
the seamless adoption of the latest high-performance models,
while approximate operators inevitably lead to a degradation
in inference accuracy.
To address these challenges, we propose FedRAG, a high-
throughput, privacy-preserving federated RAG framework
designed specifically for cross-institutional knowledge syn-
ergy. At its core, FedRAG introduces a novel Scrambled Dis-
tributed Attention protocol. Instead of transmitting highly
vulnerable intermediate states or relying on heavyweight cryp-tographic protocols, participating institutions dynamically ne-
gotiate a numerically stable feature scrambling mechanism.
This specially designed scrambling structure provides robust
privacy guarantees while sustaining highly efficient and pre-
cise computation on hardware accelerators. By transmitting
these scrambled tensors over the network to dynamically dele-
gate computations to an independent compute node, FedRAG
successfully decouples the self-attention execution from data
localization. This innovative design ensures that no single
entity can access the plaintext contexts of others, while fully
leveraging the abundant computational resources and high-
speed interconnects typical of institutional consortiums.
Our core contributions can be summarized as follows:
•We present FedRAG, a collaborative generative system
that breaks down cross-institutional data silos. Our framework
requires no model retraining, relies on no approximate opera-
tors, and eliminates the need for specialized hardware (e.g.,
TEEs), enabling seamless and rapid deployment of state-of-
the-art LLMs.
•We design a highly efficient, non-cryptographic privacy-
preserving attention protocol. It provides robust mathematical
and empirical defense against sophisticated intermediate state
inversion attacks while replacing prohibitive cryptographic
overhead with hardware-friendly linear transformations.
•Extensive evaluations across various open-source LLMs
and diverse benchmarks validate the promise of our system.
FedRAG achieves up to a 62 ×latency reduction over ex-
isting secure baselines, sustaining practical, human-reading
throughput. Crucially, this immense efficiency gain comes
with near-zero model utility degradation across six question
answering and summarization benchmarks.
2 Background
2.1 Transformers and Self-Attention
The Transformer [46] serves as the foundational architec-
ture underlying contemporary LLMs, representing a highly
efficient and high-performance neural network architecture
for processing long sequences. A typical Transformer model
is composed of a stack of layers, with each layer primarily
consisting of a multi-head self-attention mechanism and a
feed-forward network. At the core of this multi-head mecha-
nism is the scaled dot-product attention. During computation,
the model first linearly projects each token’s hidden state into
three matrices1: Queries ( Q), Keys ( K), and Values ( V). Since
Q,K, and Vare all linearly projected from the same input
sequence, this mechanism is referred to asself-attention, and
the attention output is computed as:
Attention(Q,K,V) =softmax
QK⊤/√
d
V.(1)
1Throughout this paper, vectors are represented as row vectors, and d
denotes the dimension of the attention head.
2

While this dense cross-token interaction enables the model
to effectively understand long contexts, it inherently intro-
duces a strict data dependency: the inference for any individ-
ual token dictates mandatory access to the representations (i.e.,
KandV) of all other relevant tokens in the sequence. This
intrinsic architectural trait forms a fundamental bottleneck
when attempting to scale LLM inference across distributed or
privacy-sensitive data silos, making the decoupling of atten-
tion computation and data localization a critical prerequisite
for collaborative generation.
2.2 Retrieval-Augmented Generation
RAG [29] equips LLMs with external knowledge to mitigate
the poor performance and hallucinations caused by static in-
ternal weights on domain-specific or time-sensitive tasks, all
without requiring costly retraining. A standard query-based
RAG pipeline typically executes in three steps. First, an em-
bedding model maps the query into a dense vector space to
perform coarse-grained document retrieval via approximate
nearest neighbor search. Next, a cross-encoder reranker con-
ducts precision filtering by evaluating fine-grained semantic
interactions between the query and candidate documents. Fi-
nally, during response synthesis, the top-ranked documents
are concatenated with the user query and jointly fed into the
LLM to generate a grounded answer.
However, deploying this pipeline in federated scenarios
introduces a fundamental architectural conflict. When knowl-
edge bases are distributed across mutually distrustful institu-
tions, centralizing retrieved documents violates strict data pri-
vacy constraints. This data fragmentation severely impairs the
reranking and generation stages, both of which mandate inten-
sive cross-token attention computation between the user query
and the retrieved documents to capture fine-grained seman-
tic dependencies. Because the Transformer’s self-attention
strictly demands access to all context tokens, executing this
computation across distributed, plaintext-sensitive contexts
naturally transforms into a challenging cross-node privacy
problem, motivating the federated RAG pipeline formalized
in §3.
3 System Model and Problem Formulation
In this section, we delineate the unique characteristics of
our target deployment scenario, formalize the federated RAG
pipeline, and define the threat model that guides our system
design.
3.1 System Model and Target Scenario
While RAG significantly enhances the performance of LLMs
in specialized domains, organizations possessing vast amounts
of domain-specific data often face severe “data silo” prob-
lems due to privacy concerns. Bridging this gap, our work tar-gets a highly practical and increasingly prevalent deployment
scenario:cross-institutional collaborative RAG within a
consortium. In this scenario, institutions are willing to share
information to leverage LLMs for generating macroscopic
insights, but they must strictly avoid exposing granular, record-
level details to other participating entities. Consequently, we
define the primary privacy objective of this work as preventing
leakage of the documents’ raw plaintext.
In such a cross-institutional setting, the participating nodes
are large organizations rather than individual users. They
typically operate within enterprise-grade infrastructure char-
acterized by three main features:
•Powerful Hardware Resources.Each participating node
operates enterprise-grade servers equipped with high-end AI
accelerators and massive storage systems, which provide the
computational capacity required to execute large-scale LLM
inference.
•High-Speed and Predictable Interconnects.These in-
stitutions are often interconnected via dedicated optical net-
works or co-located within Data Center Colocation facilities,
which provide high-speed connections of multi-Gbps and
stable millisecond-level low latency.
•Multi-Node Participation.The system inherently in-
volves three or more independent participating nodes. Ac-
cording to the threat model introduced later, we assume no
collusion occurs between any pair of these nodes.
Existing privacy-preserving LLM inference frame-
works predominantly focus on the traditional “edge-cloud”
paradigm, which typically involves a resource-constrained
edge device communicating over high-latency public net-
works [30, 60]. As a result, most existing work overlooks
the critical need for inter-institutional federated inference. To
fill this void, our method fully leverages the aforementioned
advantages in system resources and multi-participant architec-
tures, proposing a privacy-preserving federated RAG frame-
work designed to satisfy the high-throughput and long-context
demands of modern LLMs.
3.2 Problem Formulation
To elucidate the system architecture, we formally define the
problem of Federated RAG. Consider a consortium of Minsti-
tutions, where each institution imaintains a private, localized
document corpus Di. The global conceptual corpus is the
union of these isolated silos:D=SM
i=1Di.
When an authorized user submits a query q, the federated
RAG pipeline aims to generate a comprehensive response y
conditioned on both qand the most relevant documents dis-
tributed across D. The end-to-end pipeline can be broadly
abstracted into two primary stages: global retrieval and dis-
tributed generation. During theglobal retrievalstage, the
system collaboratively retrieves and aggregates the most rel-
evant documents across the global conceptual corpus Dto
3

form a globally optimal context pool:
Cf inal=Retrieve(q,D 1,D2,...,D M).(2)
Subsequently, in thedistributed generationstage, an LLM
synthesizes the final response. For each generation step t, the
model computes the attention outputs over the distributed
context Cf inaland the query q, generating the next token yt
based on the previously generated tokensy <t:
yt=LLM(C f inal,q,y<t).(3)
Problem Statement:During the distributed generation
(and collaborative re-ranking, if applied), the core operation
is the cross-node attention computation Attention(Q,K,V) .
The fundamental problem our system addresses is how to
compute this cross-node attention collaboratively when Qand
K,V belong to mutually distrustful nodes. The system must
ensure that neither party can reconstruct the other’s exact
plaintext tensors, all while maintaining a strict end-to-end
latency constraint suitable for real-time applications.
3.3 Threat Model and Security Assumptions
Given the consortium-based scenario, our system operates
under a highly practical threat model tailored for cross-
institutional collaborations.
Honest-but-Curious (Semi-Honest) Adversaries:We as-
sume all participating nodes arehonest-but-curious. Nodes
will strictly adhere to the predefined cryptographic proto-
cols, correct execution of neural network forward passes, and
timely transmission of intermediate states. However, they are
“curious” and may attempt to passively log the exchanged
intermediate activations (e.g., scrambled Q,K,V, and atten-
tion outputs) to infer or reconstruct the original private text of
other institutions. Our primary defensive objective is to thwart
such passive reconstruction attacks, notably intermediate state
inversion [11] and V ocabulary Mapping Attacks (VMA) [44].
Reputation-Aware and Non-Collusion Constraints:Un-
like anonymous P2P networks, institutions in our scenario
(e.g., hospitals, banks) are bound by stringent legal frame-
works (e.g., HIPAA, GDPR) and commercial contracts. There-
fore, we introduce aReputation-Awareassumption: partici-
pants are highly risk-averse regarding their institutional rep-
utation. An adversary will actively refrain from any mali-
cious behavior (such as deliberately feeding poisoned data,
intentionally altering model weights, or violating protocol
rules) if such actions have a high probability of leaving ver-
ifiable traces subject to future auditing. Active attacks are
thus considered out of scope. Furthermore, this reputation-
aware nature inherently enforces aNon-Collusionassumption
among participating institutions. Any attempt to collude—
where two organizations explicitly coordinate to share crypto-
graphic keys or intermediate tensors to decrypt a third party’s
data—inevitably leaves severe digital footprints (e.g., abnor-
mal out-of-band communication records). More importantly,such malicious coordination exposes each colluding party to
the catastrophic risk of being reported or audited by the other.
Thus, rational, reputation-bound institutions will inherently
refrain from collusion.
4 Scrambled Distributed Attention
4.1 Distributed Attention Computation
In distributed RAG systems, the KV Cache of the LLM is
partitioned across multiple nodes, requiring cross-node col-
laboration to execute the attention mechanism. To minimize
cross-node communication, we build on the online-softmax
decomposition widely adopted in efficient attention imple-
mentations [10, 32], which decomposes attention into two
stages: per-node local computation followed by a lightweight
global aggregation. In what follows, we formalize this decom-
position in our federated setting; this framework will serve
as the substrate on which our privacy-preserving protocol
operates (§4.4).
Per-Node Local Computation.Assume that the relevant
KV Cache is distributed across Mnodes. Let KiandVide-
note the key and value matrices stored on node i, respectively.
Given a query matrix Q, node idirectly applies the standard
attention computation to its local (Ki,Vi)producing two quan-
tities:
•Node-Local Attention Output.Applying Eq. (1)to the
locally availableKandVmatrices gives
Oi=softmaxQK⊤
i√
d
Vi.(4)
This output is normalized only over the keys stored on node i
and thus does not yet reflect the global attention distribution.
•Local Normalization Factor.To recover the global at-
tention distribution across all nodes, we also compute the
corresponding normalization factor. Letw i∈RLQdenote the
vector whose ℓ-th entry aggregates the unnormalized softmax
weights on nodei:
[wi]ℓ=∑LVi
t=1expQK⊤
i√
d
ℓ,t.(5)
Global Aggregation.Finally, the local results from all
nodes are communicated and aggregated to reconstruct the
global attention output. The global attention output is recon-
structed as
Attention(Q,K,V) =∑M
i=1wi⊙O i
∑M
i=1wi,(6)
where⊙and the division are applied row-wise.
This two-stage decomposition decouples the attention com-
putation across nodes while preserving mathematical equiv-
alence to the centralized formulation. Crucially, it exposes
4

a clean interface—each node only exchanges ( Oi,wi) with
the aggregator—on which we layer our privacy-preserving
transformation in §4.4.
4.2 Privacy-Preserving Attention Computa-
tion
Random Feature Scrambling.Let Φ∈Rd×dbe an invertible
matrix, which we call ascrambling matrix. Given a vector
x∈Rd, we refer to xΦorxΦ−⊤as ascrambled vector; the
choice between ΦandΦ−⊤depends on the role of xin the
downstream attention computation, as will become clear in
§4.4. The KV Cache of the LLM is partitioned across multi-
ple nodes, requiring cross-node collaboration to execute the
attention mechanism. We defer the concrete construction of
Φto the next paragraph, since the following two algebraic
properties hold for any invertibleΦ.
Lemma 1(Inner Product Preservation).For any two vectors
x1,x2∈Rdand any invertible matrix Φ∈Rd×d, scrambling
with the dual matrices ΦandΦ−⊤preserves the inner prod-
uct:
⟨x1Φ,x 2Φ−⊤⟩= (x 1Φ)(x 2Φ−⊤)⊤=⟨x 1,x2⟩.(7)
Lemma 2(Linearity).For any vectors x1,...,x n∈Rdscram-
bled with the same matrix Φ∈Rd×d, and any scalar weights
w1,...,w n∈R:
∑n
i=1wi(xiΦ) = ∑n
i=1wixi
Φ.(8)
These two properties are chosen by design: they corre-
spond precisely to the two core operations of scaled dot-
product attention. Lemma 1 preserves the Q-K inner products
that determine attention logits, while Lemma 2 preserves the
V-weighted summation that produces attention outputs. To-
gether, they enable a computation party to execute attention on
scrambled (Q,K,V) and return a result that, after output-side
descrambling, equals the plaintext attention—without ever
observing the plaintext tensors. We formalize this protocol in
§4.4.
Constructing a Numerically Stable Scrambling Matrix.
While Lemmas 1 and 2 hold for any invertible Φ, a naive
choice—such as a fully random dense matrix sampled from
a continuous distribution—fails in practice. Modern LLM
inference runs in low-precision floating-point formats (BF16
or FP16), which offer only 7-10 bits of mantissa. A random
dense Φproduces entries of widely varying magnitudes, caus-
ing the matrix-vector products xΦandxΦ−⊤to suffer from
severe round-off and cancellation errors. These errors accu-
mulate across the dozens of transformer layers, degrading
generation quality to the point of unusability (please refer
to Appendix A for an empirical demonstration). A practical
scrambling matrix must therefore reconcile two seemingly
conflicting goals: sufficient mixing to obscure the plaintext,
and numerical stability under low-precision arithmetic.Our construction.We construct the scrambling matrix as:
Φ=S 1P1HP2S2,(9)
where H∈Rd×dis a normalized Hadamard matrix, P1,P2
are independent random permutation matrices, and S1,S2are
random diagonal matrices whose entries possess random signs
and variable magnitudes. Each component serves a distinct
role:
•Normalized Hadamard matrix Hsimultaneously ad-
dresses both goals from above. Its base {−1,+1} entries
allow matrix multiplication to be computed via stable sign-
flipping additions followed by a single global 1/√
dscaling,
avoiding the floating-point errors of arbitrary dense multiplica-
tions (numerical stability). At the same time, its orthogonality
ensures that each output coordinate is a ±1/√
d-weighted
sum of all input coordinates (sufficient mixing).
•Permutation matrices P1,P2randomize the otherwise
deterministic H. A fixed Halone is publicly known; sandwich-
ing it between two random permutations yields a distribution
over mixing matrices. Permutations are exact operations, so
they preserve numerical stability.
•Scaling matrices S1,S2distort distances. Pure
permutation-and-Hadamard preserves L2distances between
rows, exposing topological information (e.g., kNN structure)
that enables graph-matching attacks (§6). The random
scaling factors severely disrupt these distance relationships.
Their magnitudes are sampled from an empirical range
(e.g., [1/8,8] ), representing a fundamental trade-off between
privacy enhancement and floating-point precision.
The two-layer sandwich structure ( S1P1·H·P 2S2) ensures
that both the input and output sides of Hare randomized, so
partial knowledge of one layer does not help an attacker invert
the other.
Security under structured Φ.A natural concern is whether
this structured Φ—which has fewer effective degrees of free-
dom than a fully random matrix—provides sufficient privacy.
In §6, we show that it does: the highly collinear geometry of
LLM hidden states [15, 16] fundamentally limits any alge-
braic inversion attack, making the structural constraints on
Φlargely inconsequential in practice. We defer the detailed
analysis to §6.
Random Token Permutation.In addition to scrambling
the feature dimension, we introduce random permutation in
the token dimension. Due to the permutation-equivariant na-
ture of attention computation, applying a consistent permuta-
tion to the KandVmatrices yields an attention output from
which the original result can be exactly recovered. Similarly,
permuting the Qmatrix results in an attention output that,
upon applying the inverse permutation, matches the original
output. This token-level shuffling significantly enhances sys-
tem resilience against known-plaintext attacks and brute-force
enumeration.
Scrambled Attention Protocol.Consider three matrices
Q= [q 1;...;q LQ]∈RLQ×d,K= [k 1;...;k LK]∈RLK×d, and
5

Coordinator𝐾𝐾/𝑉𝑉/𝑄𝑄 Attn. OutputCollaborative Nodes
Compute Node
Ranking
Scores𝐾𝐾/𝑉𝑉/𝑄𝑄 Attn. Output② Re-ranking
Collaborative Nodes Coordinator Compute Node
𝐾𝐾/𝑉𝑉Attn.
Output𝑄𝑄Attention
KV CacheAnswerLLM Docs
③ GenerationAttn.
Output𝑄𝑄SimilarityEmbedding
Vecto r
Distributed DatabaseCoordinatorCandidate
DocumentsQuery① Embedding RetrieveFigure 1: System architecture of the federated RAG system.
V= [v 1;...;v LV]∈RLV×d, where LV=L K. We generate two
random scrambling matrices ΦKQ,ΦV∈Rd×dand two per-
mutation matrices PQ∈RLQ×LQ,PKV∈RLV×LVto form the
key setΘ. The workflow is as follows:
1. Plaintext Matrix Scrambling:
Enc(Q,K,V,Θ) = (P QQΦ KQ,PKVKΦ−⊤
KQ,PKVVΦ V)
= (Q′,K′,V′).(10)
2. Scrambled Attention Computation:
Attention(Q′,K′,V′) =softmax
Q′K′⊤/√
d
V′
=softmax
PQQΦ KQ(PKVKΦ−⊤
KQ)⊤/√
d
(PKVVΦ V)
=P Qsoftmax
QK⊤/√
d
P⊤
KV(PKVVΦ V)
=P QAttention(Q,K,V)Φ V.(11)
3. Output Descrambling:
Dec(Attention(Q′,K′,V′),Θ)
=P−1
QAttention(Q′,K′,V′)Φ−1
V
=Attention(Q,K,V).(12)
In this design, no single node can access the plaintext data
of matrices from other nodes, yet the system correctly exe-
cutes the attention computation.
5 The RAG Pipeline
This section details the lifecycle of a single user request within
the federated RAG system. When an authorized user from a
participating organization requires generative services, they
submit a query to their local node. This receiving node as-
sumes the role of theCoordinatorfor the request, orches-
trating subsequent operations alongside other collaborating
nodes in the federation system.5.1 System Overview and Node Roles
A complete RAG request lifecycle consists of three main
phases: vector retrieval, re-ranking, and generation. The sys-
tem architecture is illustrated in Figure 1. In the vector re-
trieval phase, theCoordinatorfirst converts the received user
query into a vector representation using an embedding model
and broadcasts it to all collaborating nodes. Each node per-
forms vector retrieval against its local documents, returning
the identifiers and similarity scores of the top- kmost rele-
vant documents to theCoordinator. TheCoordinatorthen
aggregates and ranks these scores to select the top- kglobal
candidates for the subsequent re-ranking stage. During re-
ranking, theCoordinatorand the collaborating nodes jointly
execute the re-ranking model inference to determine the final
top-m(m<k ) documents that constitute the context for the
RAG request. Finally, in the generation phase, theCoordina-
torcollaboratively performs LLM inference with other nodes
holding the retrieved contexts to generate the final response.
Since both the re-ranking and generation phases require
cross-node attention computation, we first formalize the roles
of participating nodes. To execute the privacy-preserving at-
tention mechanism across the federated network, we catego-
rize the nodes into three distinct roles based on their functions
during each cross-node attention computation:
•Inquirer.The node currently running the forward pass for
unprocessed tokens. It possesses the plaintext Query required
for the attention computation.
•Context Owner.The node possessing the plaintext K
andVmatrices required for the attention computation.
•Compute Node.The node responsible for executing the
actual attention computation. It receives scrambled matrices
from the Inquirer and Context Owner, computes the result,
and returns it to the Inquirer.
For each cross-node attention computation, the Inquirer
and Context Owner share the scrambling key set Θ. The
Compute Node does not possess these keys and thus can-
not reconstruct the information, ensuring the security of the
privacy-preserving protocol. These roles are applied dynami-
cally across both the re-ranking and generation stages.
5.2 Collaborative Re-ranking
Building upon the vector retrieval results, the re-ranking
stage employs specialized re-ranking models (e.g., BGE
Reranker [5], mGTE [55]) to recalibrate the relevance scores
between candidate documents and the query. Unlike the bi-
encoder architecture used in the preceding vector retrieval
phase, which independently maps queries and documents into
a shared vector space to compute coarse-grained similarities,
this stage adopts a cross-encoder approach. It directly models
the fine-grained, token-level interactions between each query-
document pair, thereby significantly improving retrieval qual-
ity.
6

Linear Projection 
Scaled Dot-Product
Attention
Feed Forward NetworkLayer Input
Layer OutputHidden State
Local Attention
Descrambling & AggregationK/V Q Scrambling Scrambling
Local AttentionK/VContext Owner
Compute NodeFigure 2: Forward pass of a decoder layer and the distributed
attention computation in the generation phase.
Input Construction.In this stage, the model input is
a concatenation of the query and the candidate document:
[CLS]⊕query⊕[SEP]⊕document⊕[SEP] , where [CLS]
and[SEP] are special tokens functioning as the global se-
quence representation and the segment separator, respec-
tively. In a federated RAG system, the query and documents
may reside on different nodes, necessitating cross-node atten-
tion computation. We utilize the scrambling-based privacy-
preserving attention method introduced in §4 to execute col-
laborative re-ranker inference without leaking private infor-
mation.
Privacy-Preserving Forward Pass.Most classical re-
ranking models typically adopt an encoder-only architecture
utilizing bidirectional attention [48, 52]. This implies that
the attention computation for every single token requires full
access to the intermediate states of the entire sequence (i.e.,
both preceding and succeeding contexts). Consequently, par-
ticipating nodes simultaneously act as both Inquirers (seeking
attention outputs for their own tokens) and Context Owners
(providing their KandVmatrices to others) during the for-
ward pass of each layer. Nodes independently compute the
Q,K, and Vmatrices for their local tokens based on the out-
put of the previous layer. These matrices are scrambled and
transmitted to the designated Compute Node responsible for
the attention calculation. Once the computation is complete,
this node sends the corresponding attention outputs back to
the respective Inquirer nodes owning the tokens. The partic-
ipating nodes then descramble the results and proceed with
the subsequent Feed-Forward Network (FFN) computations
to complete the layer’s inference. This process repeats until
all layers are processed, requiring bidirectional communica-
tion with the Compute Node Lreranker times (where Lreranker is
the number of model layers). Finally, the hidden state of the
[CLS]token is mapped to a relevance score.
5.3 Collaborative Generation
In the RAG generation phase, the retrieved results are fed into
the LLM along with the user input. Modern LLMs predom-
inantly rely on a decoder-only architecture. In the forwardpass of a decoder-only transformer, each token only computes
attention with its preceding tokens, rather than all tokens in
the sequence as is characteristic of encoder architectures. Dur-
ing the generation process, the intermediate state (i.e., the
KV Cache) of each token remains unchanged once computed.
For distributed generation in our system, this implies that the
roles of nodes participating in the decoder’s attention com-
putation are asymmetric: a Context Owner does not need to
perform layer-by-layer attention computation synchronously
with the Inquirer possessing subsequent tokens. Instead, it
can complete the KV Cache computation for its local tokens
in advance and transmit them to the Compute Node ahead of
time. We illustrate the forward pass of a single decoder layer
and our distributed attention computation in Figure 2.
The generation process comprises a prefill phase and a
decoding phase. The former builds the KV Cache for the
input tokens, while the latter generates new tokens step-by-
step based on the prefilled cache. Unlike centralized RAG
systems, the prefill and decoding attention computations in
a federated RAG system must be coordinated across nodes
due to the distributed nature of documents and queries. We
define a continuous sequence of plaintext tokens held by a
single node as aninput segment. The global input sequence
consists of ordered input segments T1,T2,...,T ns, owned by
nodes O(T 1),O(T 2),...,O(T ns), respectively. Nodes owning
these segments sequentially perform the forward pass to com-
plete attention caching based on the segment’s position in the
sequence.
Prefill Phase.When the tokens in segment Tiundergo
the forward pass, the owners of all preceding segments,
O(T 1),...,O(T i−1), act as Context Owners. They scramble
theKandVof their already-prefilled tokens and transmit
them to the Compute Node in advance. Concurrently, the
owner of Tiassumes the role of the Inquirer. For each trans-
former layer, the Inquirer computes the local Q,K, and V
matrices for its current segment and performs a partitioned
attention calculation. As mathematically formulated in §4.1,
it first derives the local attention output utilizing its own K,V,
andQmatrices. Next, the Inquirer scrambles its local Query
matrix and dispatches it to the Compute Node. The Compute
Node executes the attention computation between this scram-
bled Query and the scrambled K,Vpreviously received from
the Context Owners, and subsequently returns the result to
the Inquirer. Upon receiving this result, the Inquirer descram-
bles it and aggregates it with the previously computed local
attention output to finalize the attention computation for the
current segment. The aggregated attention output is then fed
into the subsequent FFN to complete the forward pass for
that specific layer. This layer-wise process is iterated until
the forward computation for all layers is concluded. Finally,
during transmission idle periods, the Inquirer scrambles the K
andVmatrices of its newly processed segment and forwards
them to the Compute Node, making them available as context
for subsequent prefill and decoding stages.
7

Decoding Phase.This phase commences after the last input
segment completes prefill. Since the final input segment ex-
actly corresponds to the user’s initial query, its owner, O(T ns),
is inherently theCoordinatorof the RAG request. Having
already assumed the Inquirer role during the prefill of this
final segment, theCoordinatornaturally retains this identity
to commence generating new tokens, while the owners of all
preceding segments serve as Context Owners. The decoding
forward pass mirrors the prefill phase, with the distinction that
it processes one newly generated token at a time to predict
the next token. Furthermore, the Inquirer does not need to
send the newly computed KandVof the generated token to
the Compute Node immediately. For each layer, the Inquirer
scrambles the new token’s Query vector, sends it to the Com-
pute Node (which already holds the scrambled KV Cache
from the prefill phase), receives the result, descrambles it, and
proceeds with the FFN. Upon completing the final decoder
layer, the Inquirer samples the new token and initiates the
next generation step. This process repeats until a termination
condition is met (e.g., reaching a maximum length limit or
generating an end-of-sequence token).
5.4 Dynamic Role Assignment
Our privacy-preserving attention scheme employs a
symmetric-like scrambling mechanism, implying that any
node capable of scrambling is also capable of descrambling.
This necessitates a strict constraint: a node acting as a
Compute Node in a specific attention computation must not
simultaneously be a Context Owner or an Inquirer. Otherwise,
the Compute Node could use the shared key to reverse the
counterparty’s data, violating privacy. This constraint implies
that our federated RAG system requires a minimum of three
independent physical nodes.
If there exists a node in the system that does not own any
input segments, it can permanently serve as the Compute
Node for all cross-node attention computations in that request.
However, in scenarios with few nodes and a uniform docu-
ment distribution, all nodes may own input segments. This
requires dynamically assigning the Compute Node role to
at least two different physical nodes during a request. When
physical node NE
1acts as a Context Owner, NE
2must serve
as the Compute Node, and vice versa. This implies that the
KV Cache for certain segments must be replicated to both
NE
1andNE
2, incurring additional communication overhead.
LetmaxSeg(N i) =max O(T j)=N ijdenote the highest index
of an input segment owned by physical node Ni. Consider
two physical nodes assigned the Compute Node role, NE
1and
NE
2, with maxSeg(NE
1)<maxSeg(NE
2). To minimize the
overhead of redundant transmission, we strategically select
the node with the smallest maximum segment index as one
of the Compute Nodes. Specifically, if there exists a physical
nodeNisuch that maxSeg(N i) =1 (i.e., it owns only the very
first input segment), we can permanently assign it as a Com-pute Node. This strategy eliminates the need to redundantly
transmit theKandVinformation of any input segment.
6 Privacy Analysis
State Inversion & Text Recovery.State inversion attacks
aim to recover original tokens by accessing the model’s in-
termediate activation states. Representative methods include
VMA [44], white-box optimization-based inversion [11, 14],
and black-box generation-based inversion [11, 35]. VMA re-
lies on traversing the vocabulary and utilizing the sorted L1-
distance for ANN collisions. However, the dense mixing in-
troduced by the Hadamard matrix Hand the non-isometric
scaling by S1,S2in our scrambling scheme thoroughly destroy
the distance relationships between the plaintext and ciphertext,
rendering VMA ineffective. White-box optimization-based
inversion requires calculating the loss via forward propaga-
tion to construct a differentiable distance function; since the
computational node lacks the scrambling matrix Φ, back-
propagation cannot be performed. Furthermore, black-box
generation-based inversion relies on a stationary feature space
to train a Seq2Seq inverse mapping model. In contrast, our
system dynamically negotiates Φfor each layer at the be-
ginning of every RAG request, preventing the attacker from
obtaining a stable feature space for model convergence. Con-
sequently, to execute the aforementioned text recovery attacks,
an adversary must satisfy an absolute prerequisite: unmixing
the dense scrambled representation manifold to extract the
plaintext intermediate states. This forces the attacker to resort
to low-level algebraic unmixing attacks.
Defeating Independent Component Analysis (ICA).ICA
is a classic blind source separation algorithm that aims to ex-
tract original underlying signals from their linear mixtures
without prior knowledge of the mixing parameters [21]. We
evaluated the resistance of our method against ICA attacks, as
shown in Figure 3a. The five curves represent the Hungarian
Mean Absolute Cosine Similarity between the attack results
and the targets under different scenarios: no leakage (know-
ing only the scrambling structure), known P2, known S2, a
random guess baseline based on the empirical distribution of
intermediate states, and a positive control using independent
Laplace sources. These results demonstrate that even under
pessimistic assumptions of partial information leakage, the
recovery similarities plateau near the random guess baseline
and fail to approach an exploitable degree. In contrast, the
IID Laplace positive control can be almost perfectly recov-
ered. This disparity demonstrates that our scheme naturally
resists ICA as the highly collinear intermediate states (i.e.,
the anisotropy problem) of modern LLMs intrinsically violate
the independent source assumption [15, 16] upon which the
algorithm relies.
Graph Matching Attacks.Graph matching attacks are
non-parametric attack schemes that attempt to reconstruct fea-
tures by aligning the relative distance matrices of the scram-
8

24
25
26
27
28
29
210
211
212
Sequence Length0.20.40.60.81.0Matched Mean |Cos|
No Leakage
Known P2
Known S2
IID Laplace
Rand. Guess(a) Resistance to ICA.
24
25
26
27
28
29
210
211
212
Sequence Length0.00.20.40.60.8Neighbor Overlap
Key
Query
Value
Random (b) Resistance to graph matching.
Figure 3: Security evaluation of the scrambling mechanism
against algebraic and topological attacks.
bled space and the known plaintext spaces. Since exact sub-
graph matching is fundamentally an NP-Hard problem [8],
existing efficient graph matching algorithms heavily rely on
the topological stability of local spatial features [43]. As il-
lustrated in Figure 3b, experiments demonstrate that the kNN
overlap rate between the plaintext and ciphertext spaces drops
rapidly as the sequence length Lincreases. At L=4096 , the
local neighborhood overlap for Q,K, and Vall plummet be-
low 15%, which is vastly below the theoretical threshold that
existing efficient alignment methods can exploit [9].
Malicious Prompt.To counter the threat of request ini-
tiators attempting to steal other nodes’ documents via mali-
ciously crafted prompts, our system employs a game-theoretic
defense mechanism based on ciphertext auditing. The initiat-
ing node of a RAG request (i.e., the Coordinator) must send
the scrambled query to the computational nodes, and its keys
are shared with at least one other node in the system. In the
event of a dispute, the federation can collaboratively audit
these scrambled records. This ensures proactive suppression
of malicious attacks under a “reputation-aware” mechanism.
7 Evaluation
7.1 Evaluation Setup
We implement our privacy-preserving mechanism as an API-
compatible, drop-in replacement for the standard attention
function in the Hugging Face transformers library. We con-
duct our evaluations on a server equipped with one NVIDIA
PRO 6000 and two NVIDIA RTX 5090 GPUs. To enforce
inter-node isolation, we deploy the participating nodes within
separate Docker containers, and we employ Toxiproxy as
the network proxy between containers to simulate diverse
network environments with varying latency and bandwidth
constraints.
In this section, we first evaluate the end-to-end latency,
throughput, and communication volume of our method com-
pared to existing privacy-preserving LLM inference tech-
niques in RAG scenarios. Subsequently, we analyze the im-
pact of various factors—such as the number of protected lay-Table 1: Length configurations in throughput evaluations.
Config Context Query Output
Short 256+256 32 32
Medium 1024+1024 64 64
Long 4096+4096 128 128
ers, network latency, and network bandwidth—on system ef-
ficiency. Next, we benchmark the generation quality of our
method across six diverse datasets to demonstrate that our
scrambling mechanism incurs negligible degradation in model
accuracy. Finally, we investigate the accuracy-efficiency trade-
offs of quantizing intermediate states to provide insights for
future optimizations.
Baselines.We select SCX [53] and PermLLM [59] as our
primary baselines, as they share similar characteristics with
our approach: they do not require model retraining, do not
mandate TEEs, and avoid approximate operators. For SCX,
we implement it according to the non-TEE environment set-
tings described in their paper. For PermLLM, to ensure a fair
comparison, we strictly apply its cryptographic operations
solely to the attention computation.
Models.We evaluate our system using multiple state-of-
the-art open-source LLMs, including the Qwen 3 [49], Llama
3.1 [17], Ministral 3 [31], and GPT-OSS series [1], with pa-
rameter sizes ranging from 4B to 20B. All models are ob-
tained from the Hugging Face model hub and executed using
their default data types. For the RAG retrieval phase, we uni-
formly employ bge-m3 and bge-reranker-v2-m3 [5] as the
embedding and reranking models, respectively. During the
reranking stage, we apply 4-bit quantization to the intermedi-
ate states, which we discuss in further detail in §7.5.
Datasets.For throughput and latency evaluations, we gener-
ate input token sequences of fixed lengths, with configurations
detailed in Table 1. Our experimental setup distributes con-
text tokens across two nodes, each hosting two equal-length
documents. For the reranking phase, we use 10 candidate
documents for recalibration.
For accuracy evaluation, we select several prominent
question-answering (QA) and summarization datasets
equipped with reference contexts. These benchmarks cover
a diverse range of domains, context lengths, and reason-
ing complexities. Specifically, we evaluate our system on
five standard QA datasets—SQuAD [40], HotpotQA [51],
MuSiQue [45], MS MARCO [3], and AmbigQA [34]—along
with one long-form meeting summarization benchmark, QM-
Sum [61]. Detailed descriptions of these datasets are deferred
to Appendix B.
9

Long Med. Short101102103
Qwen3 4B
Fully ProtectedLong Med. Short101102103
Llama3.1 8B
Fully ProtectedLong Med. Short101102103
Ministral3 14B
Fully ProtectedLong Med. Short101102103
GPT-OSS 20B
Fully ProtectedLong Med. Short100101102
Qwen3 4B
2 Protected LayersLong Med. Short100101102
Llama3.1 8B
2 Protected LayersLong Med. Short100101102
Ministral3 14B
2 Protected LayersLong Med. Short101102
GPT-OSS 20B
2 Protected LayersT otal Time (s)Ours SCX PermLLMFigure 4: End-to-end latency of RAG requests.
Table 2: Performance evaluation for fully protected generation.
Length Methodmeta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-20b
TTFT (s)↓Dec. TPS↑Traff. (MiB)↓Comm. Rounds↓TTFT (s)↓Dec. TPS↑Traff. (MiB)↓Comm. Rounds↓
ShortOurs3.23 11.70 301.17 1147 1.39 10.19 220.66 867
PermLLM 11.93 1.62 4721.73 10851 9.68 1.79 4224.67 8163
SCX 6.22 5.13 1791.25 3458 3.52 5.94 1181.29 2618
MediumOurs4.83 11.79 1107.61 2171 2.45 10.28 804.73 1635
PermLLM 89.23 1.46 45815.71 21091 73.47 1.62 38419.13 15843
SCX 15.09 5.48 6752.25 6530 10.13 5.98 4454.75 4922
LongOurs12.00 11.85 4238.54 4219 7.72 13.54 3037.73 1647
PermLLM 1243.09 0.96 587393.74 41571 966.51 1.16 483843.42 31203
SCX 60.15 5.50 26202.25 12674 39.13 6.01 17274.69 9530
7.2 End-to-End Latency and Communication
Overhead
Figure 4 illustrates the total end-to-end latency of a single
RAG request for our method and the baselines across varying
prompt lengths. Although prior studies have demonstrated
that transmitting plaintext intermediate states from any single
layer can lead to privacy leakage (as discussed in §6), we
follow the partial-layer protection strategy adopted by SCX
and introduce an additional configuration that only protects
two layers for comparison, with results presented in the sec-
ond row of Figure 4. In practice, this partial-layer protection
strategy serves as a viable trade-off option between system
efficiency and privacy, allowing users to configure the system
according to their specific operational requirements.
As shown in Figure 4, our method achieves the lowest end-
to-end latency across all six configurations and four models,
yielding a 1.71 ×to 62.96 ×speedup over the baselines. Table
2 details the comprehensive efficiency metrics for the Llama-
3.1-8B model, with results for other models provided in Ap-
pendix C. The table highlights that our approach achieves
significantly superior results in Time to First Token (TTFT),
Decode TPS, communication rounds, and communication vol-
ume. Crucially, the overhead of our method exhibits a linear
growth trend with respect to sequence length, endowing our
framework with excellent scalability for increasingly preva-
lent long-context workloads. We attribute this substantial ef-
ficiency improvement to the core design principle that wemaximize the computational payload delegated to the Com-
pute Node within a single cross-node communication round,
and we rely exclusively on hardware-efficient matrix multipli-
cations for both plaintext encryption and ciphertext compu-
tation, thereby minimizing latency induced by cryptography
and communication.
7.3 Throughput and Network Sensitivity
Figure 5 plots the Decode TPS metrics under varying network
latencies. Here, the number of nodes denotes the physical
machines that act as Compute Nodes during a given RAG
request; as articulated in §5.4, if a request requires all nodes
in the system to supply tokens, we must dynamically assign
the Compute Node role to two distinct nodes to guarantee pri-
vacy protection. The empirical results demonstrate that with
a Round-Trip Time (RTT) of approximately 1ms, our method
sustains Decode TPS greater than 10 under the full-layer
protection configuration. For collaborative RAG scenarios
deployed across physical servers within the same data center,
across availability zones in the same region, or across depart-
ments within a research institution, our system can deliver
output speeds matching human reading rates while providing
uncompromising privacy [27]. When the RTT increases to
10ms, the Decode TPS drops to single digits; in such scenar-
ios, operators can resort to offline generation or intentionally
trade a portion of privacy for enhanced throughput. Neverthe-
10

24816 36010203040Decode TPS
(a) Qwen3 4B, 1 node
248 16 32015304560
(b) Llama 3.1 8B, 1 node
24816 4008162432
(c) Qwen3 14B, 1 node
24816 4008162432
(d) Ministral 3 14B, 1 node
248 16 2405101520
(e) GPT-OSS 20B, 1 node
24816 36
Protected Layers010203040Decode TPS
(f) Qwen3 4B, 2 nodes
248 16 32
Protected Layers015304560
(g) Llama 3.1 8B, 2 nodes
24816 40
Protected Layers08162432
(h) Qwen3 14B, 2 nodes
24816 40
Protected Layers08162432
(i) Ministral 3 14B, 2 nodes
248 16 24
Protected Layers05101520
(j) GPT-OSS 20B, 2 nodes
0ms 0.5ms 1ms 5ms 10ms Unprotected Local Decoding BaselineFigure 5: Decode throughput under different protected-layer counts and network settings.
Qwen3 4B Llama 3.1 8B Qwen3 14B Ministral 3 14B GPT-OSS 20B
0.1 1 10 100
Bandwidth (Gbps)050010001500Prefill TPS
(a) Prefill TPS.
0.1 1 10 100
Bandwidth (Gbps)5.07.510.012.5Decode TPS
 (b) Decode TPS.
Figure 6: Throughput vs. bandwidth.
less, our throughput remains significantly higher than existing
cryptographic solutions, which typically require several min-
utes to generate a single token [12].
The throughput for configurations employing two Compute
Nodes is depicted in the second row of Figure 5. Compared
to the single Compute Node setup, the throughput exhibits an
average degradation of roughly 10%. This decline primarily
occurs because, during the decoding phase, the Inquirer must
transmit the query vector of the token to the second Compute
Node and wait for both nodes to finalize their computations
before proceeding to the subsequent inference step.
We further investigate the impact of network bandwidth on
throughput, with the results shown in Figure 6. The two subfig-
ures illustrate the throughput trends for the prefill and decod-
ing phases, respectively. We observe that network bandwidth
significantly affects the prefill phase, whereas the Decode
TPS does not exhibit noticeable degradation until the band-
width drops to 100Mbps. This discrepancy is due to the fact
that the prefill phase involves fewer communication rounds
but transmits massive volumes of data per round, whereas the
decoding phase is the exact opposite; consequently, network
latency serves as the predominant performance bottleneck for
decoding.7.4 Utility Preservation on Downstream Tasks
In §4, we theoretically proved that our privacy-preserving
method is mathematically equivalent to the standard attention
computation. However, in practical deployments, floating-
point arithmetic inevitably introduces numerical precision
errors, which can cause deviations in the model’s final output.
To evaluate the impact of these deviations on model capabil-
ities, we run benchmark evaluations across six datasets. We
select three instruction-tuned models from the previous tests
and additionally introduce the Qwen2.5-32B-Instruct model;
possessing 64 hidden layers, it better exposes the cumulative
effects of precision loss during the forward pass. We design
ablation baselines for the benchmark, sequentially introduc-
ing the distributed attention computation and the intermediate
state scrambling mechanism to analyze their respective im-
pacts on model utility.
The experimental results are summarized in Table 3. The
accuracy reported in the table denotes the proportion of sam-
ples where the ground truth answer appears within the model’s
output; for the meeting summarization dataset QMSum, we
use the F1 score instead of accuracy. Furthermore, we mea-
sure the ROUGE-L overlap between the generated output and
the ground truth, as well as the perplexity (PPL) of the ground
truth given the model. The empirical results reveal that we
maintain a virtually identical PPL compared to the standard
attention baseline, with accuracy and ROUGE-L experiencing
negligible average drops of roughly 0.06% and 0.05%, respec-
tively. These findings confirm that our precision-preserving
scrambling implementation enables privacy protection with
near-zero accuracy degradation.
7.5 Accuracy-Efficiency Trade-offs via Quan-
tization
Quantization serves as a vital optimization to reduce com-
munication overhead, which is particularly prominent during
11

Table 3: Impact of Distributed Attention and Feature Scrambling on Downstream Task Performance.
MethodSQUAD HOTPOTQA MARCO AMBIGQA MUSIQUE QMSUM
Acc↑R-L↑PPL↓Acc↑R-L↑PPL↓Acc↑R-L↑PPL↓Acc↑R-L↑PPL↓Acc↑R-L↑PPL↓F1↑R-L↑PPL↓
Baseline89.4389.67 3.0669.40 78.26 3.07 22.16 45.92 4.23 49.94 64.0217.063.93 66.79 3.3326.29 20.24 16.1
+Distributed 89.4289.683.06 69.35 78.24 3.07 22.09 45.84 4.23 49.89 63.95 17.0 63.82 66.66 3.3326.55 20.38 16.1
+Scrambler 89.42 89.683.0669.37 78.21 3.07 22.11 45.84 4.24 49.79 63.9116.963.79 66.67 3.34 26.53 20.32 16.1
168765432
Quantization Bits0.20.40.60.81.0Rerank Accuracy
Acc@3 (FP32)
Acc@1 (FP32)
Acc@3
Acc@1
(a) Accuracy under quantization.
168765432
Quantization Bits0123Score L1 Distance
Score L1 Dist
Rank Edit Dist
02468
Rank Edit Distance
 (b) Deviation from the FP32
baseline.
Figure 7: Impact of intermediate activation quantization on
the Reranker.
the reranking phase (consuming ∼70% of the total trans-
mission using the model’s default precision). To investigate
this accuracy-efficiency trade-off, we apply standard affine
min-max quantization [23] to the intermediate states during
reranking. Using the SQuAD dataset, we retrieve the top-10
candidate documents via dense embeddings and re-score them
using the quantized reranker.
Figures 7a and 7b illustrate the impact of varying quantiza-
tion bit-widths on retrieval quality and raw output deviations.
While absolute score deviations (L1 and Rank Edit distances)
increase progressively below 8-bit precision, the actual rerank-
ing accuracy (Acc@1 and Acc@3) remains remarkably robust
down to 4-bit quantization, maintaining performance compa-
rable to the FP32 baseline. This robustness stems from the
fact that reranking efficacy depends strictly on the relative
ordering of scores rather than their absolute logits, providing
a sufficiently large margin to absorb quantization errors. At
3-bit and below, however, inconsistencies rapidly spike, ren-
dering the results unusable. Thus, we strongly recommend
4-bit quantization for the reranking phase as the optimal bal-
ance between communication efficiency and retrieval quality.
Furthermore, we briefly investigate intermediate state quan-
tization during the generation phase. Directly applying 8-bit
quantization to the scrambled states severely degrades gen-
eration quality because the two random scaling operations
(S1andS2) significantly expand the tensor’s numerical distri-
bution range. For deployments under extremely constrained
bandwidth, operators can adopt a workaround by deliberately
removing the post-Hadamard scaling matrix ( S2). Under this
relaxed privacy configuration, 8-bit quantization incurs only
marginal utility penalties (average drops of ∼0.26% in accu-
racy and ∼0.23% in ROUGE-L, as detailed in Appendix E),allowing systems to dynamically trade rigorous privacy for
communication efficiency.
8 Discussion
System Scalability.In a typical RAG pipeline, the number
of documents utilized during the reranking and generation
phases is bounded by a predefined constant (i.e., top- kand
top-mdocuments). Consequently, the number of participating
nodes in these two stages does not scale indefinitely with the
overall system size. As the institutional consortium expands,
the primary performance bottleneck shifts to the global vector
retrieval phase. Fortunately, a substantial body of existing
work has extensively investigated efficient distributed vector
search in large-scale systems [24, 47]. These established tech-
niques can be orthogonally integrated into our framework to
alleviate retrieval bottlenecks.
Architectural Compatibility.Since our privacy-
preserving mechanism exclusively modifies the attention
computation, it is inherently compatible with a wide spectrum
of multimodal models that rely on the standard Transformer
architecture. Furthermore, for emerging sub-quadratic archi-
tectures (e.g., Mamba [18], DeltaNet [50]), our framework
remains adaptable. For layers utilizing these alternative
mechanisms, the intermediate states can be directly trans-
mitted to the Inquirer node for local computation, while the
remaining standard attention layers can continue to leverage
our distributed, scrambled attention protocol.
Stronger Privacy Guarantees.While our previous eval-
uations primarily explored trading a portion of privacy for
enhanced throughput under constrained networks, our scram-
bling methodology can be naturally extended to provide
stronger privacy guarantees in scenarios with abundant band-
width or stringent confidentiality requirements. By deliber-
ately exchanging communication overhead for heightened
security, we can apply more aggressive obfuscation to the
intermediate states, particularly in shallower layers. Poten-
tial extensions include elevating the transmission precision
of intermediate states to 32-bit floating-point coupled with a
fully random dense scrambling matrix, as well as applying
dimensional expansion to the KandQmatrices padded with
orthogonal random noise. These enhancements would further
resist sophisticated algebraic unmixing attempts.
Future Work.Our work represents an early exploration
into privacy-preserving federated RAG systems, leaving sev-
12

eral avenues for future research. Currently, our decoding
phase processes the generation of a single token sequentially.
Future iterations could incorporate advanced mechanisms
such as Speculative Decoding [28] to generate multiple tokens
per cross-node communication round, thereby significantly
boosting system throughput. Additionally, while our current
defense against malicious prompts relies on reactive cipher-
text auditing logs, future designs could integrate automated
detection and mitigation mechanisms to proactively block
adversarial queries prior to execution.
9 Related Work
Retrieval-Augmented Generation.RAG was initially pro-
posed to mitigate LLM hallucinations and bypass costly
retraining by fetching relevant, non-parametric knowledge
from external data stores prior to generation [29]. Existing
RAG architectures can be fundamentally categorized into
three paradigms based on their augmentation methodolo-
gies [57].Query-based RAGdirectly concatenates retrieved
plaintext information with user inputs at the prompt level
(e.g., REALM [19], SELF-RAG [2]).Latent representation-
based RAGdeeply fuses retrieved contexts into the genera-
tive model’s hidden states via cross-attention mechanisms
(e.g., FiD [22], RETRO [4]). Finally,Logit-based RAGinter-
polates retrieval probabilities into the generator’s step-wise
decoding distribution during inference (e.g., kNN-LM [25],
TRIME [62]).
Privacy-Preserving LLM Inference.Recent work has
actively explored privacy-preserving LLM inference to safe-
guard data in collaborative deployments. Cryptography-based
methods [12, 20] deliver rigorous mathematical security by
executing neural network operations over encrypted data or
secret shares, primarily leveraging Fully Homomorphic En-
cryption (FHE) and Secure Multi-Party Computation (SMPC).
Alternatively, TEE-based methods [33] leverage hardware en-
claves to protect critical data and sensitive intermediate states.
Furthermore, statistical perturbation mechanisms [13,54] mit-
igate feature inversion risks while avoiding extensive cryp-
tographic computational overhead by applying Differential
Privacy (DP) principles to inject controlled noise into inter-
mediate LLM activations.
10 Conclusion
In this paper, we present a privacy-preserving federated RAG
framework designed to overcome the prevalent data silo prob-
lem and the prohibitive latency constraints of existing cryp-
tographic inference techniques. By leveraging a numerically
stable feature scrambling mechanism alongside dynamic role
delegation, our system facilitates secure, cross-node atten-
tion computation without exposing plaintext contexts. Exten-
sive evaluations confirm that our approach delivers orders ofmagnitude latency reduction and sustains practical, human-
reading throughput with near-zero model utility degradation.
Ultimately, this work eliminates critical infrastructure barri-
ers, paving the way for secure, cross-institutional knowledge
synergy in highly regulated and privacy-sensitive domains.
References
[1]Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman,
Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai,
Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-
oss-20b model card.arXiv preprint arXiv:2508.10925,
2025.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-rag: Learning to retrieve, gen-
erate, and critique through self-reflection. InThe Twelfth
International Conference on Learning Representations,
2023.
[3]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, et al. Ms
marco: A human generated machine reading comprehen-
sion dataset.arXiv preprint arXiv:1611.09268, 2016.
[4]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George Bm
Van Den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, et al. Improving language mod-
els by retrieving from trillions of tokens. InInterna-
tional conference on machine learning, pages 2206–
2240. PMLR, 2022.
[5]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text embeddings
through self-knowledge distillation.arXiv preprint
arXiv:2402.03216, 4(5), 2024.
[6]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri
Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman,
et al. Evaluating large language models trained on code.
arXiv preprint arXiv:2107.03374, 2021.
[7]Jung Hee Cheon, Andrey Kim, Miran Kim, and Yongsoo
Song. Homomorphic encryption for arithmetic of ap-
proximate numbers. InInternational conference on the
theory and application of cryptology and information
security, pages 409–437. Springer, 2017.
[8]Donatello Conte, Pasquale Foggia, Carlo Sansone, and
Mario Vento. Thirty years of graph matching in pattern
recognition.International journal of pattern recognition
and artificial intelligence, 18(03):265–298, 2004.
13

[9]Daniel Cullina and Negar Kiyavash. Improved achiev-
ability and converse bounds for erdos-rényi graph match-
ing.ACM SIGMETRICS performance evaluation review,
44(1):63–72, 2016.
[10] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and
Christopher Ré. Flashattention: Fast and memory-
efficient exact attention with io-awareness.Advances
in neural information processing systems, 35:16344–
16359, 2022.
[11] Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen,
Zhen Liu, and Haojin Zhu. Depth gives a false sense
of privacy: {LLM}internal states inversion. In34th
USENIX Security Symposium (USENIX Security 25),
pages 1629–1648, 2025.
[12] Ye Dong, Wen-jie Lu, Yancheng Zheng, Haoqi Wu,
Derun Zhao, Jin Tan, Zhicong Huang, Cheng Hong,
Tao Wei, Wen-Guang Chen, et al. Puma: Secure in-
ference of llama-7b in five minutes.Security and Safety,
4:2025014, 2025.
[13] Minxin Du, Xiang Yue, Sherman SM Chow, Tianhao
Wang, Chenyu Huang, and Huan Sun. Dp-forward: Fine-
tuning and inference on language models with differen-
tial privacy in forward pass. InProceedings of the 2023
ACM SIGSAC Conference on Computer and Communi-
cations Security, pages 2665–2679, 2023.
[14] Ege Erdo ˘gan, Alptekin Küpçü, and A Ercüment Çiçek.
Unsplit: Data-oblivious model inversion, model stealing,
and label inference attacks against split learning. In
Proceedings of the 21st Workshop on Privacy in the
Electronic Society, pages 115–124, 2022.
[15] Kawin Ethayarajh. How contextual are contextualized
word representations? comparing the geometry of bert,
elmo, and gpt-2 embeddings. InProceedings of the 2019
conference on empirical methods in natural language
processing and the 9th international joint conference on
natural language processing (EMNLP-IJCNLP), pages
55–65, 2019.
[16] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse:
Simple contrastive learning of sentence embeddings.
InProceedings of the 2021 conference on empirical
methods in natural language processing, pages 6894–
6910, 2021.
[17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.[18] Albert Gu and Tri Dao. Mamba: Linear-time sequence
modeling with selective state spaces.arXiv preprint
arXiv:2312.00752, 2023.
[19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Mingwei Chang. Retrieval augmented language
model pre-training. InInternational conference on ma-
chine learning, pages 3929–3938. PMLR, 2020.
[20] Meng Hao, Hongwei Li, Hanxiao Chen, Pengzhi Xing,
Guowen Xu, and Tianwei Zhang. Iron: Private infer-
ence on transformers.Advances in neural information
processing systems, 35:15718–15731, 2022.
[21] Aapo Hyvärinen and Erkki Oja. Independent component
analysis: algorithms and applications.Neural networks,
13(4-5):411–430, 2000.
[22] Gautier Izacard and Edouard Grave. Leveraging pas-
sage retrieval with generative models for open domain
question answering. InProceedings of the 16th con-
ference of the european chapter of the association for
computational linguistics: main volume, pages 874–880,
2021.
[23] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong
Zhu, Matthew Tang, Andrew Howard, Hartwig Adam,
and Dmitry Kalenichenko. Quantization and training
of neural networks for efficient integer-arithmetic-only
inference. InProceedings of the IEEE conference on
computer vision and pattern recognition, pages 2704–
2713, 2018.
[24] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-
scale similarity search with gpus.IEEE transactions on
big data, 7(3):535–547, 2019.
[25] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. Generalization through
memorization: Nearest neighbor language models.
arXiv preprint arXiv:1911.00172, 2019.
[26] Brian Knott, Shobha Venkataraman, Awni Hannun,
Shubho Sengupta, Mark Ibrahim, and Laurens van der
Maaten. Crypten: Secure multi-party computation meets
machine learning.Advances in Neural Information Pro-
cessing Systems, 34:4961–4973, 2021.
[27] Woosuk Kwon.vLLM: An Efficient Inference Engine
for Large Language Models. PhD thesis, UC Berkeley,
2025.
[28] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast
inference from transformers via speculative decoding. In
International Conference on Machine Learning, pages
19274–19286. PMLR, 2023.
14

[29] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474, 2020.
[30] Senyao Li, Haozhao Wang, Wenchao Xu, Rui Zhang,
Song Guo, Jingling Yuan, Xian Zhong, Tianwei Zhang,
and Ruixuan Li. Collaborative inference and learning
between edge slms and cloud llms: A survey of algo-
rithms, execution, and open challenges.arXiv preprint
arXiv:2507.16731, 2025.
[31] Alexander H Liu, Kartik Khandelwal, Sandeep Subra-
manian, Victor Jouault, Abhinav Rastogi, Adrien Sadé,
Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexan-
dre Gavaudan, et al. Ministral 3.arXiv preprint
arXiv:2601.08584, 2026.
[32] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring at-
tention with blockwise transformers for near-infinite
context.arXiv preprint arXiv:2310.01889, 2023.
[33] Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou,
Yuke Hu, Chenxu Zhao, Zhihao Liu, and Zhan Qin.
Shadow in the cache: Unveiling and mitigating pri-
vacy risks of kv-cache in llm inference.arXiv preprint
arXiv:2508.09442, 2025.
[34] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and
Luke Zettlemoyer. Ambigqa: Answering ambiguous
open-domain questions. InProceedings of the 2020
conference on empirical methods in natural language
processing (EMNLP), pages 5783–5797, 2020.
[35] John Morris, V olodymyr Kuleshov, Vitaly Shmatikov,
and Alexander M Rush. Text embeddings reveal (al-
most) as much as text. InProceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing, pages 12448–12460, 2023.
[36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
Training language models to follow instructions with
human feedback.Advances in neural information pro-
cessing systems, 35:27730–27744, 2022.
[37] Pascal Paillier. Public-key cryptosystems based on com-
posite degree residuosity classes. InInternational con-
ference on the theory and applications of cryptographic
techniques, pages 223–238. Springer, 1999.
[38] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu
Wang, and Xindong Wu. Unifying large language mod-
els and knowledge graphs: A roadmap.IEEE Transac-tions on Knowledge and Data Engineering, 36(7):3580–
3599, 2024.
[39] Qi Pang, Jinhao Zhu, Helen Möllering, Wenting Zheng,
and Thomas Schneider. Bolt: Privacy-preserving, accu-
rate and efficient inference for transformers. In2024
IEEE Symposium on Security and Privacy (SP), pages
4753–4771. IEEE, 2024.
[40] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. Squad: 100,000+ questions for machine
comprehension of text. InProceedings of the 2016
conference on empirical methods in natural language
processing, pages 2383–2392, 2016.
[41] Mohamed Sabt, Mohammed Achemlal, and Abdelmad-
jid Bouabdallah. Trusted execution environment:
What it is, and what it is not. In2015 IEEE Trust-
com/BigDataSE/Ispa, volume 1, pages 57–64. IEEE,
2015.
[42] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al.
Large language models encode clinical knowledge.Na-
ture, 620(7972):172–180, 2023.
[43] Anders Søgaard, Sebastian Ruder, and Ivan Vuli ´c. On
the limitations of unsupervised bilingual dictionary in-
duction. InProceedings of the 56th Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pages 778–788, 2018.
[44] Rahul Krishna Thomas, Louai Zahran, Erica Choi, Ak-
ilesh Potti, Micah Goldblum, and Arka Pal. Hidden no
more: Attacking and defending private third-party llm
inference. InForty-second International Conference on
Machine Learning, 2025.
[45] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. Musique: Multihop questions
via single-hop question composition.Transactions of
the Association for Computational Linguistics, 10:539–
554, 2022.
[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,
and Illia Polosukhin. Attention is all you need.Advances
in neural information processing systems, 30, 2017.
[47] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin,
Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo,
Chengming Li, Xiaohai Xu, et al. Milvus: A purpose-
built vector data management system. InProceedings
of the 2021 international conference on management of
data, pages 2614–2627, 2021.
15

[48] Zhichao Xu, Fengran Mo, Zhiqi Huang, Crystina Zhang,
Puxuan Yu, Bei Wang, Jimmy Lin, and Vivek Srikumar.
A survey of model architectures in information retrieval.
arXiv preprint arXiv:2502.14822, 2025.
[49] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chen-
gen Huang, Chenxu Lv, et al. Qwen3 technical report.
arXiv preprint arXiv:2505.09388, 2025.
[50] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen,
and Yoon Kim. Parallelizing linear transformers with
the delta rule over sequence length.Advances in neu-
ral information processing systems, 37:115491–115522,
2024.
[51] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. Hotpotqa: A dataset for diverse, ex-
plainable multi-hop question answering. InProceedings
of the 2018 conference on empirical methods in natural
language processing, pages 2369–2380, 2018.
[52] Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. Pre-
trained transformers for text ranking: Bert and beyond.
InProceedings of the 14th ACM International Confer-
ence on web search and data mining, pages 1154–1156,
2021.
[53] Mu Yuan, Lan Zhang, Liekang Zeng, Siyang Jiang, Bu-
fang Yang, Di Duan, and Guoliang Xing. Scx: Stateless
kv-cache encoding for cloud-scale confidential trans-
former serving. InProceedings of the ACM SIGCOMM
2025 Conference, pages 39–54, 2025.
[54] Ziqian Zeng, Jianwei Wang, Junyao Yang, Zhengdong
Lu, Haoran Li, Huiping Zhuang, and Cen Chen. Pri-
vacyrestore: Privacy-preserving inference in large lan-
guage models via privacy removal and restoration. In
Proceedings of the 63rd Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long
Papers), pages 10821–10855, 2025.
[55] Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, et al. mgte: Generalized long-
context text representation and reranking models for
multilingual text retrieval. InProceedings of the 2024
Conference on Empirical Methods in Natural Language
Processing: Industry Track, pages 1393–1412, 2024.
[56] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yu-
long Chen, et al. Siren’s song in the ai ocean: A survey
on hallucination in large language models.Computa-
tional Linguistics, 51(4):1373–1418, 2025.[57] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wen-
tao Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented
generation for ai-generated content: A survey.Data
Science and Engineering, pages 1–29, 2026.
[58] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xi-
aolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang,
Junjie Zhang, Zican Dong, et al. A survey of large lan-
guage models.arXiv preprint arXiv:2303.18223, 1(2):1–
124, 2023.
[59] Fei Zheng, Chaochao Chen, Zhongxuan Han, and Xi-
aolin Zheng. Permllm: Private inference of large lan-
guage models within 3 seconds under wan.arXiv
preprint arXiv:2405.18744, 2024.
[60] Yue Zheng, Yuhao Chen, Bin Qian, Xiufang Shi, Yuan-
chao Shu, and Jiming Chen. A review on edge large
language models: Design, execution, and applications.
ACM Computing Surveys, 57(8):1–35, 2025.
[61] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia
Mutuma, Rahul Jha, Ahmed Hassan, Asli Celikyilmaz,
Yang Liu, Xipeng Qiu, et al. Qmsum: A new bench-
mark for query-based multi-domain meeting summa-
rization. InProceedings of the 2021 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies,
pages 5905–5921, 2021.
[62] Zexuan Zhong, Tao Lei, and Danqi Chen. Training lan-
guage models with memory augmentation. InProceed-
ings of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 5657–5673, 2022.
16

A Numerical Stability Evaluation
We evaluate the numerical stability of using a random dense
matrix for feature scrambling. Table 4 presents the relative
error between the scrambled attention outputs and the orig-
inal plaintext results. We observe that scrambling with a
random dense matrix introduces a relative error of approx-
imately 12%. In contrast, our proposed structured scram-
bling method ( S1P1HP2S2) significantly reduces this error to
roughly 1.5%, maintaining consistent stability across varying
sequence lengths.
Table 4: Relative attention-output error.
Method Seq=128 Seq=512 Seq=2048
Dense Random Matrix 12.61% 11.68% 12.62%
S1P1HP2S2 1.49% 1.52% 1.63%
B Dataset Descriptions
For accuracy evaluation, we select several prominent question-
answering (QA) and summarization datasets equipped with
reference contexts. These benchmarks cover a diverse range
of domains, context lengths, and reasoning complexities:
SQuAD [40].SQuAD is a large-scale reading comprehen-
sion dataset comprising over 100,000 Wikipedia-based ques-
tions, where each answer is a continuous text span extracted
directly from the provided passage.
MuSiQue [45].MuSiQue is a challenging multihop QA
dataset constructed by composing single-hop questions. It
minimizes dataset biases to ensure models perform genuine
reasoning across multiple contexts rather than relying on
shortcuts.
QMSum [61].QMSum is a query-based meeting sum-
marization benchmark containing annotated transcripts from
various domains, designed to evaluate the extraction and sum-
marization of relevant context from long-form conversations.
HotpotQA [51].HotpotQA is a diverse, explainable multi-
hop QA dataset that requires models to reason across multiple
Wikipedia documents and synthesize answers while providing
sentence-level supporting facts.
MS MARCO [3].MS MARCO is a large-scale reading
comprehension dataset built from anonymized Bing search
queries, featuring real-world questions paired with human-
generated answers derived from multiple web documents.
AmbigQA [34].AmbigQA focuses on ambiguous open-
domain questions that yield multiple valid interpretations,
evaluating a system’s ability to identify ambiguity and gener-
ate comprehensive answers covering various plausible con-
texts.C Overall Performance Details
Detailed performance data for the evaluations conducted in
§ 7.3 are presented in Table 5. We report the following evalua-
tion metrics: Time to First Token (TTFT, in seconds), Decode
Throughput (Dec. TPS, in tokens/s), Network Traffic (Traff.,
in MiB), and Communication Rounds . The optimal value
for each metric is highlighted in bold. As demonstrated, our
method consistently outperforms the baselines across all con-
figurations and evaluation metrics.
D Benchmark Accuracy Details
Detailed results for the benchmark evaluations discussed in
§ 7.4 are provided in Table 6. For the HotpotQA, MuSiQue,
and MS MARCO datasets, we default to filtering the refer-
ence contexts using the supporting evidence (ground truth)
provided by the datasets. This filtering mechanism simulates
a practical RAG pipeline where a retriever isolates relevant
passages from irrelevant distractors. For comprehensive com-
parison, we also evaluate the models by feeding all provided
contexts (including distractors); these results are denoted with
anunfilteredsuffix.
Regarding our accuracy metric, a response is considered
correct if it contains the ground truth string as a substring.
Consequently, the reported accuracy may occasionally exceed
the ROUGE-L score. Note that QMSum is a meeting sum-
marization dataset; since strict substring matching yields an
accuracy of 0 across all models, we report the F1 score in-
stead. Additionally, we conducted identical evaluations using
the closed-source GPT-5.4 model, with results summarized
in Table 7. It is important to note that model performance on
these benchmarks is highly sensitive to prompt design and
text preprocessing. Therefore, the accuracy figures reported
herein are intended solely for relative comparison within the
scope of our experimental setup.
E Evaluation of Quantization Effects
Table 8 illustrates the utility degradation resulting from 8-bit
quantization of LLM intermediate states, as discussed in § 7.5.
For these experiments, we evaluated only the first 128 samples
from each dataset. As noted in the main text, naively apply-
ing quantization introduces significant performance penalties.
However, deliberately removing the outermost scaling ma-
trixS2effectively mitigates these quantization-induced losses.
Furthermore, our evaluation currently employs a rudimentary
affine min-max quantization scheme; integrating more ad-
vanced quantization techniques could potentially yield even
better accuracy-efficiency trade-offs in future implementa-
tions.
17

Table 5: Performance evaluation detailed results.
Length Methodprotected_layers= 2protected_layers= full
TTFT (s)↓Dec. TPS↑Traff. (MiB)↓Comm. Rounds↓TTFT (s)↓Dec. TPS↑Traff. (MiB)↓Comm. Rounds↓
Qwen/Qwen3-4B-Instruct-2507
ShortOurs0.25 36.15 156.47 75 3.65 9.91 328.01 1287
PermLLM 0.85 16.38 452.59 683 12.97 1.42 5135.92 12195
SCX 5.20 25.66 1418.02 628 5.20 4.86 1677.80 3878
MediumOurs0.37 36.51 619.11 139 5.39 9.96 1210.85 2439
PermLLM 6.02 15.32 3650.17 1323 92.52 1.28 49936.56 23715
SCX 14.44 25.84 5494.15 820 15.20 4.84 6345.42 7334
LongOurs1.05 36.79 2462.82 267 13.15 10.02 4639.35 4743
PermLLM 77.53 11.75 42196.11 2603 1341.59 0.86 640416.59 46755
SCX 52.82 26.22 21620.40 1204 55.41 4.83 24644.67 14246
meta-llama/Llama-3.1-8B-Instruct
ShortOurs0.25 47.04 141.01 75 3.23 11.70 301.17 1147
PermLLM 0.86 18.18 439.03 683 11.93 1.62 4721.73 10851
SCX 4.86 30.97 1526.88 580 6.22 5.13 1791.25 3458
MediumOurs0.42 46.99 555.65 139 4.83 11.79 1107.61 2171
PermLLM 5.93 16.94 3592.90 1323 89.23 1.46 45815.71 21091
SCX 17.44 28.67 5891.63 772 15.09 5.48 6752.25 6530
LongOurs1.15 44.49 2207.37 267 12.00 11.85 4238.54 4219
PermLLM 77.51 12.74 41963.67 2603 1243.09 0.96 587393.74 41571
SCX 58.06 31.44 23159.13 1156 60.15 5.50 26202.25 12674
mistralai/Ministral-3-14B-Instruct-2512-BF16
ShortOurs0.20 33.77 173.14 75 1.64 9.12 357.67 1427
PermLLM 0.93 19.68 466.64 558 14.73 1.29 5620.91 13539
SCX 9.09 24.45 2117.09 676 9.06 4.07 2406.17 4298
MediumOurs0.43 33.05 683.78 139 3.51 9.23 1316.92 2707
PermLLM 6.07 14.62 3721.02 1323 106.02 1.52 53448.83 20139
SCX 19.91 24.56 8172.84 868 21.65 4.18 9083.67 8138
LongOurs1.29 31.59 2719.49 267 12.51 9.17 5042.98 5267
PermLLM 76.65 11.23 42475.80 2603 1464.34 3.39 673616.17 12839
SCX 79.65 24.75 32134.34 1252 80.99 4.25 35278.67 15818
openai/gpt-oss-20b
ShortOurs0.22 20.67 46.00 75 1.39 10.19 220.66 867
PermLLM 0.98 13.25 379.52 683 9.68 1.79 4224.67 8163
SCX 3.13 17.79 941.40 484 3.52 5.94 1181.29 2618
MediumOurs0.44 20.66 160.84 139 2.45 10.28 804.73 1635
PermLLM 6.21 12.56 3295.09 1323 73.47 1.62 38419.13 15843
SCX 9.29 17.90 3643.90 676 10.13 5.98 4454.75 4922
LongOurs1.35 21.88 611.33 140 7.72 13.54 3037.73 1647
PermLLM 77.91 10.24 40677.89 2603 966.51 1.16 483843.42 31203
SCX 36.42 18.06 14331.90 1060 39.13 6.01 17274.69 9530
18

Table 6: Detailed Benchmark Performance under Distributed Attention and Feature Scrambling.
Model MethodSQUAD HOTPOTQA MARCO
Acc (%)↑Rouge-L↑PPL↓Acc (%)↑Rouge-L↑PPL↓Acc (%)↑Rouge-L↑PPL↓
Llama 3.1
8BBaseline88.33 89.751.8566.58 75.111.92 24.11 48.14 2.05
+Distributed 88.29 89.74 1.85 66.47 75.01 1.92 24.13 48.11 2.05
+Scrambler 88.30 89.721.8566.50 75.001.92 24.18 48.16 2.05
Qwen 3
4BBaseline 89.21 88.22 6.43 69.63 79.305.94 24.25 46.34 7.01
+Distributed89.2388.23 6.4369.68 79.385.97 24.11 46.25 7.02
+Scrambler 89.1888.24 6.4369.67 79.32 5.95 24.23 46.34 7.05
Ministral 3
14BBaseline86.44 89.761.60 68.41 78.51 1.72 18.03 42.442.35
+Distributed 86.40 89.75 1.60 68.4378.521.7218.09 42.462.35
+Scrambler 86.41 89.741.60 68.4778.501.7218.02 42.35 2.35
Qwen 2.5
32BBaseline 93.72 90.94 2.3572.99 80.142.6922.25 46.755.50
+Distributed 93.75 91.002.3572.82 80.062.6922.00 46.56 5.51
+Scrambler93.80 91.022.35 72.83 80.03 2.69 22.00 46.495.50
Model MethodAMBIGQA MUSIQUE QMSUM
Acc (%)↑Rouge-L↑PPL↓Acc (%)↑Rouge-L↑PPL↓F1 (%)↑Rouge-L↑PPL↓
Llama 3.1
8BBaseline56.79 63.783.9259.3362.63 1.9830.02 23.0811.61
+Distributed 56.59 63.64 3.92 59.21 62.47 1.98 29.97 23.07 11.61
+Scrambler 56.44 63.533.9259.2162.64 1.9729.87 22.9211.60
Qwen 3
4BBaseline58.04 61.0946.02 57.10 62.137.12 25.7519.66 16.65
+Distributed58.0461.05 45.9357.34 62.307.13 25.7019.81 16.65
+Scrambler 57.79 61.0245.4057.10 62.03 7.16 25.65 19.60 16.69
Ministral 3
14BBaseline 59.24 64.264.0965.70 66.80 1.9122.59 17.7316.49
+Distributed59.2964.22 4.10 65.58 66.601.9122.52 17.6516.46
+Scrambler59.29 64.374.1165.87 66.931.91 22.54 17.65 16.46
Qwen 2.5
32BBaseline25.67 66.97 14.11 73.60 75.582.31 26.81 20.5019.80
+Distributed 25.62 66.90 14.11 73.15 75.27 2.31 28.01 20.98 19.82
+Scrambler 25.62 66.72 14.13 72.98 75.062.31 28.09 21.0919.83
Model MethodHOTPOTQA (Unfiltered) MUSIQUE (Unfiltered) MARCO (Unfiltered)
Acc (%)↑Rouge-L↑PPL↓Acc (%)↑Rouge-L↑PPL↓Acc (%)↑Rouge-L↑PPL↓
Llama 3.1
8BBaseline 56.87 66.012.14 35.7940.112.9318.66 34.04 2.22
+Distributed56.96 66.022.1535.79 40.152.9318.77 34.042.22
+Scrambler 56.87 65.92 2.1535.7940.00 2.93 18.66 34.032.22
Qwen 3
4BBaseline 61.01 69.5911.6230.41 35.5715.72 17.58 29.4312.85
+Distributed61.08 69.6111.6330.49 35.7015.73 17.48 29.4212.84
+Scrambler 60.93 69.56 11.83 30.33 35.60 15.91 17.52 29.39 12.95
Ministral 3
14BBaseline 61.63 71.411.89 47.62 49.862.48 14.24 27.31 2.99
+Distributed61.74 71.511.90 47.29 49.60 2.4814.34 27.34 2.99
+Scrambler 61.66 71.39 1.89 47.37 49.782.4814.25 27.30 2.99
Qwen 2.5
32BBaseline66.08 73.95 2.95 47.54 52.80 3.54 17.0832.647.19
+Distributed 65.56 73.44 2.96 46.83 52.23 3.55 16.85 32.68 7.19
+Scrambler 65.64 73.48 2.96 47.08 52.37 3.55 16.9232.697.20
19

Table 7: Benchmark Performance of GPT-5.4.
Dataset Acc (%)↑F1 (%)↑Rouge-L↑
SQUAD 94.24 91.98 91.79
HOTPOTQA 76.41 84.19 84.66
MARCO 21.28 45.32 46.86
AMBIGQA 65.93 69.50 70.23
MUSIQUE 82.25 83.39 83.49
QMSUM 0.00 24.38 17.89
HOTPOTQA (Unfiltered) 75.21 83.09 83.54
MUSIQUE (Unfiltered) 78.03 79.59 79.97
MARCO (Unfiltered) 16.94 29.34 30.31
20

Table 8: Benchmark accuracy degradation introduced by different scrambling methods under 8-bit quantization.
Model MethodSQUAD HOTPOTQA MARCO
∆Acc∆Rouge-L∆PPL∆Acc∆Rouge-L∆PPL∆Acc∆Rouge-L∆PPL
Llama 3.1
8BS1& S 2-29.69 -31.12 +0.49 -39.84 -37.06 +3.09 -12.50 -28.12 +0.84
S1only+0.00 -0.39 +0.05 +2.34 +2.14 +0.05 -1.56 -0.02 +0.01
Qwen 3
4BS1& S 2-20.31 -25.49-0.51-27.34 -36.31 +2.75 -8.59 -22.27-2.12
S1only+0.78 -1.95-0.47-0.78 -1.01 -0.72 +0.00 +0.97-0.12
Ministral 3
14BS1& S 2-3.13 -1.03 +0.03 -2.34 -3.33 +0.09-1.56-2.52 +0.08
S1only+0.00 +0.00 +0.00 +0.78 +0.78 +0.00 -1.56 -2.22 -0.01
Qwen 2.5
32BS1& S 2-4.69 -3.31 +0.30 -3.13 -0.82 +1.03 +0.00 -5.26 +0.04
S1only+0.00 +0.65 -0.01 +0.00 -0.13 -0.04 +0.78 +0.09 -0.07
Model MethodAMBIGQA MUSIQUE QMSUM
∆Acc∆Rouge-L∆PPL∆Acc∆Rouge-L∆PPL∆F1∆Rouge-L∆PPL
Llama 3.1
8BS1& S 2-39.06 -57.97 +83.23 -43.75 -38.78 +2.55 -21.53 -11.94 +71.53
S1only+0.78 +1.35 +0.06 +0.78 -0.36 +0.03 -0.64 -0.88 +0.01
Qwen 3
4BS1& S 2-31.25 -49.15 +50.30 -35.94 -34.60 +1.75 -13.59 -5.99 +37.00
S1only-1.56 -1.20 -7.22 -5.47 -3.45 -0.48 -0.27 -0.05 -0.25
Ministral 3
14BS1& S 2-7.03 -9.58-0.02-10.16 -6.90 +0.25 -1.67 -0.94 +0.85
S1only-0.78 -1.04+0.01+0.00 +0.52 +0.01 -0.05 +0.13 +0.06
Qwen 2.5
32BS1& S 2-10.94 -16.86 +10.81 -14.06 -9.56 +1.79 -12.06 -5.82 +35.23
S1only+0.78 +0.33 -0.53 +0.00 +0.39 -0.04 +0.22 +0.13 -0.29
21