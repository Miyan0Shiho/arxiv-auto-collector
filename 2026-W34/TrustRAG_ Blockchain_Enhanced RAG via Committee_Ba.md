# TrustRAG: Blockchain-Enhanced RAG via Committee-Based Credibility Scoring

**Authors**: Baixiang Liu, Haotian Che, Yuan Li

**Published**: 2026-08-20 14:31:10

**PDF URL**: [https://arxiv.org/pdf/2608.20097v1](https://arxiv.org/pdf/2608.20097v1)

## Abstract
Retrieval-Augmented Generation (RAG) lets Large Language Models (LLMs) pull in up-to-date, domain-specific information instead of relying only on what they were trained on. Yet most RAG systems still draw from centralized databases with limited oversight, making it difficult to verify where a document came from, whether it has been tampered with, or whether it should be trusted at all. This is a serious problem in domains where both the timeliness and accuracy of retrieved content are critical, such as healthcare, finance, logistics, and legal case law, where a wrong or manipulated document can directly lead to bad decisions.
  We present TrustRAG, a committee-based, blockchain-backed RAG system: before a document is used, it is certified by a committee of domain experts through a zero-knowledge protocol, and the committee's hidden scores are combined via secure multi-party computation into a trust score that any client can verify. These scores, along with the underlying document data, are maintained jointly across chains through hash commitments, so no document or score can be silently altered or dropped, and every ranking can be independently replayed and checked.

## Full Text


<!-- PDF content starts -->

TrustRAG: Blockchain-Enhanced RAG via
Committee-Based Credibility Scoring
Baixiang Liu
Fudan University
bxliu@fudan.edu.cnHaotian Che
Fudan University
23210240106@m.fudan.edu.cnYuan Li
Fudan University
yuan li@fudan.edu.cn
Abstract—Retrieval-Augmented Generation (RAG) lets Large
Language Models (LLMs) pull in up-to-date, domain-specific
information instead of relying only on what they were trained on.
Yet most RAG systems still draw from centralized databases with
limited oversight, making it difficult to verify where a document
came from, whether it has been tampered with, or whether it
should be trusted at all. This is a serious problem in domains
where both the timeliness and accuracy of retrieved content are
critical, such as healthcare, finance, logistics, and legal case law,
where a wrong or manipulated document can directly lead to
bad decisions.
We present TrustRAG, a committee-based, blockchain-backed
RAG system: before a document is used, it is certified by a
committee of domain experts through a zero-knowledge protocol,
and the committee’s hidden scores are combined via secure
multi-party computation into a trust score that any client can
verify. These scores, along with the underlying document data,
are maintained jointly across chains through hash commitments,
so no document or score can be silently altered or dropped, and
every ranking can be independently replayed and checked.
I. Introduction
Large Language Models have achieved remarkable capabil-
ities across reasoning, generation, and knowledge-intensive
tasks, yet their reliance on static training datasets fundamentally
constrains their ability to provide accurate, up-to-date, and
verifiable information. Addressing this limitation, Retrieval-
Augmented Generation (RAG) has emerged as a promising
paradigm that enriches LLMs with external knowledge re-
trieval, thereby enhancing factual accuracy and contextual
relevance [ 1]. RAG promises more accurate, up-to-date, and
domain-specific responses. This paradigm has therefore been
widely adopted in areas such as healthcare, finance, and law,
where strong generative capabilities must coexist with strict
requirements on privacy, compliance, and auditability.
However, despite its growing role in improving factual
grounding, the architecture of modern RAG systems exhibits
a structural vulnerability: the retrieval pipelines, indexing
services, and augmentation logic remain heavily centralized.
Data flows, update policies, and ranking heuristics are typically
controlled by a small number of providers or internal platform
teams, making it difficult for downstream users or regulators to
inspect, verify, or challenge the retrieval layer’s behavior. As
a result, the very mechanism meant to reduce hallucinations
and ground outputs in fact is itself implemented as an opaque
subsystem.The evolution of RAG infrastructures echoes earlier trends
in Internet architecture: systems that begin with open, protocol-
based principles often drift toward platform-centric consol-
idation. Early protocols such as TCP/IP were designed to
distribute control and enable interoperability without cen-
tralized gatekeepers, yet were gradually overshadowed by
proprietary platforms that centralized data ownership and
shaped information flows through opaque algorithms [ 2],
[3]. RAG infrastructures now show the same trajectory: a
modular, evidence-driven mechanism increasingly resembles a
closed, platform-governed subsystem whose behavior cannot
be independently audited or verified.
This opacity gives rise to two challenges in contemporary
RAG systems: evidence trustworthiness and process gover-
nance. The first is that retrieved sources are opaque, making it
hard to verify their integrity, legality, or authority. Centralized
pipelines offer no reliable way to confirm where a passage
originated, whether it was modified, or whether it complies
with copyright and data-protection requirements. This opacity
is exploitable: attackers can dilute relevant evidence by inject-
ing superficially similar but irrelevant text [ 4], [5], or distribute
contradictory statements across sources to obscure authoritative
information and induce ambiguous or incorrect outputs [ 6], [7],
[8]. Without verifiable audit trails or cryptographic provenance,
it is difficult to tell whether anomalous retrieval behavior
stems from system limitations, data-quality issues, or deliberate
manipulation.
The second challenge is that the retrieval-and-augmentation
layer itself is opaque, obscuring why a system produces
harmful outputs or refuses to respond. This layer acts as
a gatekeeper over what conditions the model’s responses, yet
centralized ranking and filtering heuristics offer little visibility
into how conflicting or harmful content is prioritized or
suppressed. Such opacity introduces further risks: insufficient
filtering can let unsafe or toxic content shape generations [ 9],
[10], while targeted denial-of-service cues can push RAG
pipelines into refusing to answer even when relevant evidence
is available [ 11], [12]. Because these behaviors often resemble
benign limitations, operators struggle to pinpoint the cause,
complicating efforts to improve robustness and accountability.
A further difficulty arises when document trustworthiness
must be evaluated collectively, e.g., by distributed validators,
expert committees, or multiple stakeholders, rather than by
a single curator. Collecting such scores in plaintext risks
arXiv:2608.20097v1  [cs.CR]  20 Aug 2026

leaking evaluator preferences or enabling strategic manip-
ulation; aggregating them inside a closed backend merely
turns the trust value into another opaque platform output.
Trustworthy RAG therefore requires not only decentralized
provenance records, but also a privacy-preserving mechanism
for combining distributed trust inputs into verifiable retrieval
metadata.
Taken together, these developments reveal a growing tension
at the core of contemporary RAG practice. Mainstream
AI infrastructure, from model training to retrieval, remains
highly centralized, yet the data that matters most, e.g., time-
sensitive, high-accuracy, access-controlled, and frequently
updated, often originates from inherently distributed and decen-
tralized sources: hospitals, regulators, financial institutions, and
expert communities that do not share a single trusted curator.
Reconciling this mismatch calls for a blockchain-enhanced
RAG architecture, one that provides decentralized provenance,
privacy-preserving trust aggregation, and verifiable retrieval
for high-stakes domains where corrupted evidence or unac-
countable filtering carry serious operational and regulatory
costs.
To address these challenges, we propose TrustRAG, a
decentralized, verifiable, and auditable RAG framework that
combines immutable document registration with reusable
credibility computation. Documents are first registered with
content and embedding hashes as immutable provenance
anchors. Validators then submit privacy-preserving quality
evaluations bound to document identities and set hashes,
enforcing membership, uniqueness, and score correctness
without disclosing votes. During consolidation, hidden score
components are split into Shamir secret shares and processed
by committee nodes via an MP-SPDZ-based secure aggregation
interface, yielding chain-local tallies and credibility values
without revealing any individual vote or blinding factor. Rather
than generating a recursive global aggregation proof, the
system binds per-chain outputs through hash commitments
and exposes sufficient metadata for deterministic replay of
the final ranking. The resulting proofs and aggregate sum-
maries are reused during later retrieval and audit, yielding
a verifiable response package while keeping the protocol
substantially simpler than recursive-proof alternatives. This
design targets domains where knowledge is time-sensitive,
accuracy-critical, access-controlled, and continuously updated,
such as healthcare, finance, breaking news, logistics and traffic,
and legal precedents, where data is naturally distributed across
independent, mutually distrusting sources.
A. Related Work
Recent work has turned to blockchain as a mechanism for
improving data reliability and trustworthiness in RAG systems.
Andersen, Avalos, Dagher and Long proposed D-RAG [ 13],
a blockchain-based framework that organizes knowledge into
domain-specific communities, where data is validated by field
experts prior to database inclusion via a Privacy-Preserving
Knowledge Incorporation (PPKI) protocol. PPKI employs zero-
knowledge proofs and homomorphic encryption to realize adouble-blind consensus, hiding both the proposer’s identity
and members’ votes to mitigate validation bias. A separate
Retrieval Blockchain then coordinates LLM-assisted document
ranking and response generation across communities. Unlike
prior work that detects malicious content only at retrieval
time, D-RAG addresses data integrity at the source, preventing
unsafe data from entering the knowledge base in the first place.
Yu and Sato proposed DeRAG [ 14], a decentralized multi-
source RAG system that adapts the Pyth Network’s oracle
technology for knowledge-intensive applications via the RAG-
Optimized Pyth Consensus (ROPC) protocol. ROPC extends
oracle-based validation to multi-dimensional textual data
through tensor-based representations and semantic consistency
checks, coordinated by a domain-specialized validator network
organized as a directed acyclic graph. This design demonstrates
that decentralized oracle mechanisms originally developed for
financial data feeds can be effectively repurposed to improve
data integrity and retrieval reliability in RAG pipelines.
Lu, Tan, Johnson, Jung and Jiang proposed a decentralized
RAG system with adynamic reliability scoring mecha-
nism[ 15], in which each data source is assigned cumulative
reliability and usefulness scores updated via sentence-level
importance estimation and user feedback. By incorporating
these scores into both source sampling and document reranking,
their system progressively prioritizes higher-quality sources
without requiring centralized data management. Notably, their
approach achieves performance comparable to centralized
systems operating on fully reliable data, despite operating
over independently maintained, potentially noisy sources.
Lin, Cui, Zhou, et al. proposed VeriRAG [ 16], a zero-
knowledge proof framework that provides efficient integrity
guarantees for the retrieval step of RAG systems. VeriRAG
allows a service provider to prove that its returned top-
𝑘documents are genuinely the output of an Approximate
Nearest Neighbor Search (ANNS) executed faithfully over its
committed corpus, without revealing the underlying dataset.
To make this practical, the authors introduce a protocol that
sidesteps the costly verification of the sorting process itself,
together with vector-lookup and chunk-merging optimizations
that jointly reduce proving overhead. The resulting system
scales to a 37GB dataset with a 96-second prover time and a
3-second verifier time, demonstrating that succinct, privacy-
preserving proofs of retrieval correctness are practical even at
scale.
Shukla and Joshi proposed Proof-Carrying Answers
(PCA) [ 17], a protocol that attaches a signature and a Merkle
proof to every retrieved chunk and admits a generated claim
only if all its supporting chunks pass verification, abstaining
otherwise. This is an elegant shift in RAG’s default posture,
from ”trust, then verify” to ”verify, then trust”: cryptographic
grounding is checked before an answer is released, rather
than being left to post-hoc auditing, and the use of Merkle
proofs keeps this check lightweight enough for practical
deployment. The protocol relies on a single signer and a
Merkle-committed corpus snapshot, treating trustworthiness
as a matter of provenance – that a chunk is unmodified and

traceable to a registered source.
B. Our Contributions
Our paper makes the following contributions.
1)Committee-Certified Knowledge Base.We introduce
a pre-certification architecture in which documents are
endorsed by an expert validator committee at registration
time, producing reusable trust artifacts that downstream
retrieval inherits without repeating the scoring path on
every query.
2)Privacy-Preserving Multi-Party Trust Aggregation.
We combine the zero-knowledge scoring workflow with
Pedersen commitments, Shamir secret sharing, and an
MP-SPDZ secure-sum interface so that committee nodes
can aggregate hidden validator inputs without revealing
individual scores or blinding factors.
3)Cross-Chain Binding Without Recursive Proofs.We
replace a global aggregation proof with a lightweight
binding layer that hashes each per-chain scoring sum-
mary and candidate list into a single global digest,
preventing weight substitution or chain omission while
reducing system complexity.
4)Deterministic Replay for Ranking Verification.We
publish sufficient metadata for any client to indepen-
dently recompute the ranking scores, reproduce the top-
𝑘document selection, and verify that the final answer is
consistent with the retrieved evidence, without relying
on a recursive proof.
Our core contribution is an end-to-end verifiable pipeline
that connects committee-based certification at registration
time to cryptographically auditable retrieval at query time.
Documents are endorsed by an expert validator committee
through a privacy-preserving voting protocol at registration,
producing reusable trust artifacts that persist with each
document identity. At query time, these artifacts directly
inform ranking, and the entire process—from individual votes
through score aggregation to final document selection—can be
independently replayed and verified by any third party without
relying on a trusted intermediary or a recursive aggregation
proof. To our knowledge, this is the first decentralized RAG
framework that integrates privacy-preserving certification,
cross-chain binding, and deterministically replayable ranking
into a unified auditable system.
TrustRAG is designed for high-stakes domains where the
cost of misinformation is severe and the need for accountable
knowledge provenance is paramount. Examples include:
•Healthcare.Clinical guidelines and medical literature
must be endorsed by qualified experts before influencing
diagnostic or treatment recommendations.
•Legal practice.Case law and regulatory documents
require professional validation to ensure jurisdictional
relevance and currency.
•Finance.Market data and financial disclosures must be
current and traceable to a qualified source, since a stale
or manipulated figure can directly distort trading and
investment decisions.•Logistics and traffic.Routing, capacity, and disruption
information changes continuously, and outdated or unver-
ified reports can propagate through downstream planning
decisions.
•News.Breaking coverage places a premium on both timeli-
ness and accuracy, and expert or outlet-level endorsement
helps prevent unverified or fabricated reports from shaping
generated summaries.
•Government and public administration.Policy docu-
ments and official records must carry verifiable prove-
nance to prevent the citation of forged or superseded
versions.
Across all these settings, TrustRAG provides a framework
in which domain experts collectively certify knowledge at the
source, and that certification remains cryptographically bound
to every subsequent retrieval and ranking decision.
II. Challenges and Design Overview
Today’s RAG systems face a fundamental governance
problem:retrieval is centralized, trust signals are opaque, and
the ranking process is unauditable.Addressing this requires
resolving three concrete technical challenges.
Challenge 1: secure candidate scoring under untrusted
validators.:Trust metadata should be derived from distributed
validators rather than from a single curator, but the system
must prevent unauthorized participation, duplicate scoring, and
out-of-range manipulation within each scoring round bound
to a designated document set. At the same time, individual
scores and validator identities should remain hidden.
Challenge 2: privacy-preserving credibility with public
verifiability.:Even if individual votes are hidden, the system
still needs a publicly auditable way to prove that each finalized
credibility value is consistent with the submitted commitments
and with the exact document identities used at retrieval time.
Challenge 3: cross-chain binding and ranking verification
without recursive proofs.:Once per-chain credibility values
are available, the system must still prevent an untrusted
service from changing candidate sets, omitting required chains
from a declared retrieval round, or reranking documents
incorrectly. A practical design should therefore provide end-
to-end verifiability without incurring the implementation and
proving complexity of a recursive global proof.
TrustRAG addresses these challenges by combining im-
mutable document registration, reusable zero-knowledge scor-
ing, committee-side MPC aggregation, and replay-oriented
verification. Validators first prove authorized scoring over regis-
tered document identifiers using zero knowledge. Their hidden
score components are then aggregated through committee-side
MPC, and the resulting document-level values are checked
against the underlying commitments and exposed through
finalized chain records. These scores are finalized ahead of
query time and remain bound to stable document identities
rather than to any single future query. At retrieval time, the
service derives a candidate set, hashes the exact returned list,
fetches the already finalized per-document scores for that list,
and binds them through the Aggregator. Finally, the retrieval

layer publishes the finalized scores, and the associated zero-
knowledge proofs, allowing any client to verify the validity of
the scoring and aggregation and to confirm that the returned
answer hash is correctly bound to them.
Compared with representative centralized and decentralized
RAG architectures discussed in the literature, TrustRAG
emphasizes private candidate scoring, chain-local verifiable
tallying, cross-chain binding, and proof-based verification.
III. Preliminaries
A. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ 1] enhances Large
Language Models (LLMs) by combining external knowledge
retrieval with text generation. Given a query 𝑄and a document
collectionD={𝑑 1,...,𝑑𝑛}, the retriever applies an encoder
𝑓(·) to obtain dense vector representations q=𝑓(𝑄) and
d𝑖=𝑓(𝑑𝑖), and computes their similarity via cosine similarity:
sim(q,d𝑖)=q·d𝑖
∥q∥∥d𝑖∥.
The top-𝑘documentsD𝑘⊆D with the highest similarity
scores are selected and concatenated with 𝑄as additional
context for the generator 𝐺(·), which produces the final
response:
𝑅=𝐺(𝑄,D 𝑘).
This retrieval-generation pipeline enables LLMs to access
up-to-date information without retraining, improving factual
accuracy and reducing hallucination [18], [19].
B. Zero-Knowledge Proofs
A zero-knowledge proof (ZKP) [ 20] is a protocol that allows
a prover𝑃to convince a verifier 𝑉that a statement 𝑥is true,
i.e., there exists a witness 𝑤such that𝑅(𝑥,𝑤)=1 , without
revealing any information about 𝑤beyond the validity of the
statement.
Modern succinct constructions such as Groth16 [ 21] and
PLONK [ 22], along with their subsequent developments,
enable short proofs and fast verification, and are widely used
for privacy-preserving and verifiable computation.
C. Secure Multi-Party Computation
Secure Multi-Party Computation (MPC) allows multiple
parties to jointly compute a function over their private inputs
without revealing those inputs to one another. Each party holds
its own input and participates in a sequence of interactive
computation rounds; at no point during these rounds does any
party observe another party’s raw input or any intermediate
value in the clear. The final result is produced jointly, so that
no single party ever reconstructs it alone.
In our framework, MPC protects the aggregation stage
after zero-knowledge voting: committee nodes’ hidden score
components are combined jointly, revealing only the final
aggregate while each node’s contribution stays hidden.IV. Threat Model and Trust Assumptions
This section formalizes the threat model and security objec-
tives RAG framework. We explicitly characterize the system
participants, adversarial capabilities, and the information that
must remain confidential in order to guarantee robustness,
privacy, and security.
A. System Model
The system consists of five main components: (i) users
who issue queries and receive generated responses, (ii) a
RAG Service responsible for candidate retrieval and response
generation, (iii) multiple Data Chains that maintain document
provenance and finalized credibility metadata together with the
corresponding scoring records, (iv) per-chain MPC committees
that aggregate hidden scores, and (v) an Aggregator that
records cross-chain binding hashes but does not produce a
global recursive proof.
The RAG Service is not assumed to be fully trusted. Instead,
its behavior is constrained through cryptographic commitments,
chain-local verifiable state, cross-chain binding hashes, and a
deterministic replay rule for ranking.
B. Adversary Model
We consider a probabilistic polynomial-time adversary with
the following capabilities:
•observing all on-chain data and public network commu-
nication;
•injecting malicious documents or metadata into Data
Chains;
•controlling a subset of voters or validators and attempting
collusion or Sybil attacks;
•corrupting a subset of committee nodes involved in secure
aggregation; and
•deviating from the prescribed protocol execution.
We assume that, within each consensus domain, an honest
majority or honest threshold of participants is maintained. For
committee aggregation, privacy and correctness follow the
threshold assumption of the underlying 𝑡-of-𝑛Shamir-sharing
protocol.
To avoid ambiguity, we state three trust assumptions
explicitly.
•Scores≠Truth.A document’s credibility score reflects
committee endorsement, not truth. A high score means
validators approved the document through the scoring
protocol; it does not mean the content is factually correct
or safe.
•Completeness w.r.t. a Declared Set.Completeness is
defined relative to a declared chain set. Before each
retrieval round rid, the service and verifier agree on an
ordered set of participating chains Jrid. A verifier can
only check for omitted or reordered chains within this
declared set, not against chains outside it.
•Detectability, Not Availability.Our guarantees cover
tampering detection, not service availability. If the RAG
Service alters data or deviates from the protocol, this is

publicly detectable. However, we do not guarantee that
the service will return a result at all; it may simply refuse
to respond.
C. Security and Privacy Objectives
Our design aims to satisfy the following security and privacy
objectives.
•Security Property 1 (Data Integrity).Once a chain-
local tally is accepted, it must match the commitments
recorded on-chain; results cannot be altered after the fact.
•Security Property 2 (Vote Privacy).Beyond the final
aggregate, no one should learn an honest validator’s score
or identity.
•Security Property 3 (Trust-Metadata Integrity).A
document’s credibility score can only rise through valid,
accepted votes. An adversary cannot inflate it without
either corrupting enough of the trust domain or breaking
the underlying cryptographic assumptions.
•Security Property 4 (Retrieval Verifiability).Clients
can independently verify that the credibility scores and
final ranking come from finalized chain state, computed
by the declared deterministic rule—not fabricated by the
service.
D. Confidentiality and Public Verifiability
To meet the objectives above, some information must stay
hidden during the protocol, while other information must be
public enough for anyone to check that nothing was tampered
with.
Individual scores and voter identities must be hidden. If a
score were public, a validator could be bribed or threatened
into voting a certain way. If a voter’s identity were linked to
their score, they could face retaliation or be profiled based
on their voting history. Hiding both prevents coercion, vote-
buying, and collusion.
User queries, and anything that could reveal what a user
was looking for, are also kept off-chain. We do not offer a
dedicated query-hiding protocol beyond this; queries simply
never touch the chain. Likewise, the retrieval process itself is
not shown to outside observers in full: candidate document
sets are hashed into 𝑙𝑗, and only this hash is published. In
short, the only public information consists of cryptographic
commitments, chain-level tally records, zero-knowledge proofs,
and cross-chain binding hashes—nothing more.
Even with all this hidden, the system remains fully verifiable.
Each vote is published as a commitment plus a zero-knowledge
proof showing the vote was valid (from a registered validator,
not a duplicate, within the allowed score range) without
revealing the score itself. Each chain publishes its aggregate
results: the credibility scores for its documents, along with
a digest binding these scores to the corresponding document
identities. The Aggregator publishes, for each chain, a hash
combining that chain’s score digest with its candidate-list
digest, and then combines all of these per-chain hashes into a
single global digest. The service similarly publishes hashes of
the final ranking and the response. Together, these let anyoneverify that each chain’s tally is correct, no chain was skipped,
and the final ranking is consistent—all without ever seeing an
individual vote.
E. Out of Scope
Our system does not guarantee that the LLM’s output
is factually correct, and it does not defend against prompt
injection or other attacks on generation itself. It also cannot
help if a validation domain is fully compromised—for example,
if enough validators and committee members collude to
endorse malicious content—or if a validator simply makes a
poor judgment call when scoring a document.
What we do guarantee is detectability, not prevention: as
long as the honest-threshold assumption holds, any attempt to
tamper with document provenance, forge aggregation results,
omit required chains, or manipulate trust scores can be publicly
caught and audited.
V. System Overview
A. Architecture
As shown in Fig. 1, the system has three main layers:
Data Chains, theAggregator Binding Layer, and theRAG
Service Layer. Each Data Chain has its own MPC committee
for secure tallying and optional auditing. The retrieval service
itself is treated as untrusted and not necessarily decentralized.
Document provenance is fixed at registration, trust scores
are produced through protected scoring and aggregation, and
cross-chain consistency is enforced via hash binding and
deterministic replay, without relying on recursive proofs. In
our prototype, the expensive cryptographic scoring is done
once at ingestion or score-submission time, and the resulting
artifacts are simply reused during later retrieval.
1) Data Chains:Each data chain manages its own corpus
and validation logic independently. When a document is
ingested, the chain records an immutable provenance anchor –
its identifier and hash – along with the trust artifacts produced
by validator scoring. When a query arrives, the retrieval
service selects a set of candidate documents from the chain,
and the chain exposes the finalized credibility scores for
those candidates. The service then computes a hash over
the exact candidate list, binding it to the scores it received.
These records serve as verifiable attestations of chain-local
credibility and form the basis for ranking. In our prototype,
the zero-knowledge score proofs are generated once when
scores are submitted, and reused across later queries rather
than regenerated each time.
Score consolidation itself relies on committee-assisted
MPC. Instead of revealing raw score components to a single
aggregator, validators split their hidden score values into shares
distributed across committee nodes, which jointly compute the
sum and authorize its release. Only the aggregate document-
level score is derived and recorded on-chain, together with
artifacts needed for later verification; individual validator
scores stay hidden. Committee-side threshold authorization
and secret-sharing details mainly come into play during score
finalization or post-hoc auditing.

Fig. 1. System Overview
2) Aggregator Layer:The Aggregator is the cross-chain
binding layer. Instead of verifying one big recursive proof, it
hashes each chain’s score hash and candidate-list hash together
into a single digest per chain, then combines all chains’ digests
into one global digest. This binding step prevents a chain from
being skipped, a document’s score from being swapped, or
a candidate list from being replaced – while keeping the
cross-chain logic simple and lightweight.
MPC only plays a role earlier, inside each chain, as a privacy-
preserving step for consolidating scores. By the time results
reach the Aggregator, each Data Chain has already verified
its own aggregated scores and finalized its per-document
credibility values. The Aggregator simply binds each chain’s
finalized score hash and candidate-list hash together – it does
not repeat any of that chain-local verification.
3) RAG Service Layer:The RAG Service Layer is the off-
chain interface between users and the verifiable multi-chain
knowledge base. Given a query, it runs semantic retrieval to get
candidate documents from each chain, fetches their credibility
scores and proof references from the finalized chain state,
and pulls the binding digest from the Aggregator. A trust-
aware ranking step then combines semantic relevance with
document credibility, so results reflect both how relevant a
document is and how trustworthy its source is. The top-ranked
documents are passed to the LLM generator, and the final
output is bundled into a ProofPack containing the metadata a
client needs to verify it.
Together, these three layers form a unified architecture: data
quality assessment is decentralized across chains, integrity is
enforced through binding hashes, and the off-chain retrieval
service itself stays accountable and verifiable.TABLE I
Notation used in the query flow.
Symbol Meaning
𝑄User query
CSet of chains participating in the query
𝐷𝑗 Candidate document set from chain𝑗
docId Document identifier
𝑤𝑗 Document credibility score on chain𝑗
ℎ𝑗 Binding digest for chain𝑗(score + candidate list)
𝐺Global digest across all chains
𝐷★Final evidence set passed to the generator
𝑅Generated response
ΠProof package returned to the client
B. Query Flow
Figure 2 shows how a query flows through the system: from
the RAG Service, to the Data Chains, through the Aggregator
binding layer, and finally to the LLM, with committee-side
MPC aggregation happening inside each chain’s trust path.
The key design choice is to separate document registration and
trust scoring from online retrieval: documents are registered
and scored ahead of time, and committee authorization is only
invoked when scores are finalized or audited. This means that
during normal query processing, the service just consumes
already-finalized credibility scores for the retrieved candidates
– it does not redo any scoring. Global coordination at query
time is then reduced to two simple things: binding hashes
together and replaying the ranking over the declared set of
chainsC.
Steps 1–3: Query submission, embedding, and candidate-
set binding.:When the RAG Service receives a query 𝑄, it
first encodes it into a vector q=𝑓(𝑄) and retrieves a candidate

UserTrustRAG
ServiceRetriever Data ChainsAggregator
(Binding)LLM
1) query𝑄
2) embed + semantic retrieve
3) candidate sets{𝐷 𝑗}
4) request local tally / proof
5) scores𝑤 𝑗, digestℎ 𝑗
6) submitℎ 𝑗bindings
7) return global digest𝐻
8) replayable evidence𝐷★
9) grounded response𝑅
10)𝑅+Π
Verification relies on local state checks, hash binding, and deterministic replay; no global recursive proof is used.
Fig. 2. End-to-end query flow. Candidate sets are matched to local trust artifacts, then bound across chains through hashes and verified by replay.
document set 𝐷𝑗from each participating chain 𝑗∈C . For
each chain, the service also computes a list hash 𝑙𝑗=𝐻(𝐷𝑗),
which binds the exact set of candidates used in this round.
This hash is computed fresh at query time – it is not fixed
in advance, since the candidate set depends on the specific
query.
Steps 4–6: Chain-local trust retrieval, optional tallying,
and local checks.:For each candidate document 𝑑𝑡∈𝐷𝑗, the
service retrieves its finalized credibility score from chain 𝑗. In
the full design, validators on chain 𝑗submit hidden scores to-
gether with zero-knowledge proofs of membership, uniqueness,
range validity, and commitment correctness. Committee nodes
then aggregate these hidden scores via MPC to derive each
document’s credibility 𝑤𝑡, without revealing any individual
validator’s score.
Let𝑊𝑗={(docId𝑡,𝑤𝑡)}𝑑𝑡∈𝐷𝑗denote the set of scores for
the retrieved candidates on chain 𝑗, and𝑠𝑗=𝐻(𝑊𝑗)its hash.
In our prototype, scores are computed once when validators
submit them, and simply reused at query time. So at query
time, the service just fetches the relevant subset 𝑊𝑗, computes
𝑠𝑗, and checks it against the chain’s on-chain records. In short:
score computation happens ahead of time, while candidate-set
binding happens at query time.
Steps 7–8: Cross-chain binding and deterministic rerank-
ing.:Once the chain-local outputs are ready, the Aggregator
combines them into a single digest:
ℎ𝑗=𝐻(𝑠𝑗∥𝑙𝑗), 𝐺=𝐻(ℎ 1∥ℎ2∥···∥ℎ𝑚),
where the chains are taken in a fixed canonical order over C.
This makes it detectable if any chain is dropped or reordered
relative to the declared configuration. The RAG Servicethen ranks each candidate document by combining semantic
relevance with credibility:
score(𝑑𝑖)=𝛼·sim(𝑄,𝑑 𝑖)+𝛽·𝑤𝑖.
The top-ranked documents form the final evidence set 𝐷★,
which is bound into a replay hash:
𝑟=𝐻 𝑄∥C∥{𝑠 𝑗}𝑗∥{𝑙𝑗}𝑗∥𝐷★.
𝐷★is then passed to the LLM, which produces the final
grounded response:
𝑅=𝐺(𝑄,𝐷★).
Steps 9–10: Returning a verifiable response package.:
The response is bundled with a verifiable proof package:
Π= C,{𝑊𝑗}𝑗,{𝑠𝑗}𝑗,{𝑙𝑗}𝑗, 𝐺, 𝑟, 𝐻(𝑅),
while the underlying chain-local records remain retrievable
from their source chains. The response and Πare returned to
the user, who can independently verify chain-local correctness,
cross-chain completeness, ranking correctness, and response
integrity.
This query flow provides end-to-end verifiability without a
global recursive proof: correctness is instead established by
checking chain-local state, binding hashes across chains, and
replaying the ranking computation.
VI. Protocol Design
A. Document Registration
Before the system handles any query, each Data Chain
registers its documents up front. For a document 𝑑, registration

just needs a content hash, an embedding hash, and a metadata
hash. The chain stores this as a tuple
(docId,contentHash,embeddingHash,owner),
which becomes the document’s permanent identity for every-
thing that happens later – retrieval, scoring, verification, all
of it.
Registration itself doesn’t produce a trust score. Its only
job is to fix a stable identity for the document on-chain, so
that when scores and verification happen later, they’re always
anchored to something concrete rather than to raw content
that could shift or be reinterpreted.
B. Zero-Knowledge Quality-Scoring Protocol
RAG systems depend on the quality of their documents.
Public expert ratings, however, expose validators to bribery,
coercion, and retaliation. We therefore require a scoring
protocol that is verifiable without being public.
The zero-knowledge scoring protocol operates over the
candidate set 𝐷𝑗on chain𝑗for a given round. It enforces
three properties simultaneously: only registered validators may
vote; each validator may score a given document at most once
per round; and every accepted score lies in [0,𝑆 max]. None
of these properties requires disclosing the voter’s identity or
vote value.
The protocol’s output is a document-level weight 𝑤𝑡, bound
to a stable document identifier. At query time, the service
discloses the weights for retrieved documents and binds them
into a score hash 𝑠𝑗together with the candidate-list hash 𝑙𝑗;
no rescoring occurs on the query path.
Once a vote is proven valid, an MPC layer sums the
hidden scores (Section VI-C ). This does not replace the zero-
knowledge layer; it ensures the intermediate values used in
tallying are never exposed to a single party in plaintext.
1) Participants:The protocol involves three roles.
•Validatorsare domain experts who score documents.
Each holds a secret key 𝑠𝑘𝑖and is listed in a registry,
proving membership via a Merkle tree rather than
revealing identity.
•Data Chainsgovern a domain corpus. They verify proofs,
maintain tally records, and reject duplicate or malformed
submissions, without ever observing a score or validator
identity in plaintext.
•Committee nodesjointly aggregate hidden scores under
a threshold scheme. The prototype uses 𝑛=4 nodes with
reconstruction threshold𝑡=3.
2) Cryptographic Building Blocks:Three primitives support
the protocol.
Pedersen commitmentshide the score. A validator commits
to score𝑠𝑖,𝑡with blinding factor𝑟 𝑖,𝑡:
Com𝑖,𝑡=𝑔𝑠𝑖,𝑡ℎ𝑟𝑖,𝑡mod𝑝.
The commitment reveals nothing about 𝑠𝑖,𝑡and cannot later be
altered. Its multiplicative structure allows many commitments
to be combined into a sum without opening individual scores,
which is essential for tallying.Merkle membershipproves registration without identifying
the validator. All validators are committed into a tree with
rootRoot𝑟𝑒𝑔; a validator supplies an authentication path as
part of the proof.
Nullifiersprevent duplicate voting. Each validator computes
Null𝑖,𝑡=Poseidon(𝑠𝑘 𝑖,rid,docId 𝑡),
bound to their key, the round rid, and the document. The
nullifier is published on-chain, so a repeated vote on the same
document in the same round is detected, without revealing
which validator cast it.
Each vote additionally carries a succinct zk-SNARK
(Groth16 or PLONK) attesting that all constraints above hold,
at constant proof size and low verification cost.
3) Formal Relation:For validator 𝑣𝑖scoring document
docId𝑡on chain𝑗, the public statement is
𝑥𝑖,𝑡=(Root𝑟𝑒𝑔,Com𝑖,𝑡,Null𝑖,𝑡,rid,docId 𝑡, 𝑆max),
and the private witness is
𝑤𝑖,𝑡=(𝑠𝑖,𝑡, 𝑟𝑖,𝑡, 𝑠𝑘𝑖,MerklePath 𝑖).
The relationRscore holds iff the following four constraints
are satisfied:
Root𝑟𝑒𝑔=MerkleRoot(ℓ 𝑖,MerklePath 𝑖)(1)
Null𝑖,𝑡=Poseidon(𝑠𝑘 𝑖,rid,docId 𝑡)(2)
Com𝑖,𝑡=𝑔𝑠𝑖,𝑡ℎ𝑟𝑖,𝑡mod𝑝(3)
0≤𝑠𝑖,𝑡≤𝑆 max (4)
Constraint (1)enforces membership, (2)enforces vote unique-
ness, (3)enforces commitment correctness, and (4)enforces
range validity.
The validator generates 𝜋𝑖,𝑡←Prove(𝑝𝑘 score,𝑥𝑖,𝑡,𝑤𝑖,𝑡)
and submits(Com𝑖,𝑡,Null𝑖,𝑡,𝜋𝑖,𝑡). The Data Chain accepts
the submission only ifVerify(𝑣𝑘 score,𝑥𝑖,𝑡,𝜋𝑖,𝑡)=1.
C. Credibility and Cross-Chain Binding
Once each chain has finalized its document-level scores, the
protocol discloses the subset relevant to the current candidate
set and binds these per-chain outputs together, without a
recursive global proof.
The chain-local inputs to this step come from secure MPC
aggregation: committee-side secure sum strengthens the local
tally, while cross-chain coordination reduces to hash binding
over the published outputs. The resulting aggregate is checked
against the accumulated Pedersen commitment product before
the chain derives per-document scores and publishes the
records later consumed at retrieval.
1) Chain-Local Tally and Weight Computation:For each
chain𝑗∈C, let
𝑙𝑗=𝐻(𝐷𝑗)
bind the candidate set for the current round. For each document
𝑑𝑡∈𝐷𝑗, the committee computes the hidden tally
𝑆(𝑗)
𝑡=∑︁
𝑖𝑠(𝑗)
𝑖,𝑡,

and the aggregated commitment
ComTally(𝑗)
𝑡=Ö
𝑖Com(𝑗)
𝑖,𝑡=𝑔𝑆(𝑗)
𝑡ℎ𝑅(𝑗)
𝑡mod𝑝.
The credibility value for each candidate is
𝑤(𝑗)
𝑡=𝑆(𝑗)
𝑡
𝑛𝑗·𝑆max,
where𝑛𝑗is the validator count for chain 𝑗. This keeps 𝑤(𝑗)
𝑡∈
[0,1] and makes weights comparable across documents and
chains. The chain then forms
W𝑗={(docId𝑡,𝑤(𝑗)
𝑡)}𝑑𝑡∈𝐷𝑗, 𝑠𝑗=𝐻(W𝑗).
2) Chain-Local Verifiable Relation:Each chain exposes
enough finalized state for a verifier to check its local tallies and
derived weights. In the prototype, this consists of finalized tally
records, aggregate openings, and contract-verifiable checks.
The relationR(𝑗)
allholds iff:
all accepted votes satisfyR score,(5)
ComTally(𝑗)
𝑡=Î
𝑖Com(𝑗)
𝑖,𝑡,∀𝑑𝑡∈𝐷𝑗,(6)
𝑆(𝑗)
𝑡=Í
𝑖𝑠(𝑗)
𝑖,𝑡,∀𝑑𝑡∈𝐷𝑗,(7)
𝑤(𝑗)
𝑡=𝑆(𝑗)
𝑡
𝑛𝑗·𝑆max,∀𝑑𝑡∈𝐷𝑗,(8)
𝑠𝑗=𝐻(W𝑗).(9)
Thus the disclosed chain-local state certifies both the validity
of the underlying votes and the correctness of the derived
weights, for the exact candidate set hashed into𝑙 𝑗.
3) Binding Layer and RAG Integration:The Aggregator
stores only binding hashes:
ℎ𝑗=𝐻(𝑠𝑗∥𝑙𝑗),
𝐺=𝐻(ℎ 1∥ℎ2∥···∥ℎ𝑚).
At retrieval, the service obtains {W𝑗}𝑗,{𝑠𝑗}𝑗, and{𝑙𝑗}𝑗from
the Data Chains, and 𝐺from the Aggregator. Any client can
then verify the supporting chain-local state, recompute ℎ𝑗and
𝐺, and replay the ranking procedure. The Aggregator thus acts
only as a binding layer, preventing substitution or omission
relative to the declared chain set C, without introducing a
recursive proof.
D. Trust-Aware Retrieval and Verifiable Response Generation
Once each chain has published (W𝑗,𝑠𝑗,𝑙𝑗)along with its
supporting on-chain state, the RAG Service Layer performs
provenance-aware retrieval and generates a grounded, veri-
fiable response. Unlike conventional RAG, which treats all
retrieved documents as equally trustworthy, our system folds
cryptographically verified credibility into ranking: semantic
relevance and provenance jointly determine which documents
shape the output.
Algorithm 1 gives the steps.
Each returned answer is thus conditioned only on documents
whose provenance and credibility were independently verified,
and comes with a replayable certificate any client can check.Algorithm 1Trust Retrieval and Verifiable Response Genera-
tion
Require: Query𝑄, round id rid, chain setC, proof interfaces
for all Data Chains, binding digest𝐺
Ensure:Response𝑅and proof packageΠ
1:q←𝑓(𝑄)
2:{𝐷𝑗}𝑗∈C←SemanticRetrieveByChain(q)
3:for all𝑗∈Cdo
4:𝑙𝑗←𝐻(𝐷𝑗)
5:(W 𝑗,𝑠𝑗)←GetLocalState(𝑗,rid,𝐷 𝑗)
6:ifVerifyLocalState(𝑗,rid,𝑙 𝑗,𝑠𝑗,W𝑗)=0then
7:abortwithInvalidLocalState
8:end if
9:ℎ𝑗←𝐻(𝑠𝑗∥𝑙𝑗)
10:end for
11:if𝐻(ℎ 1∥···∥ℎ𝑚)≠𝐺then
12:abortwithBindingMismatch
13:end if
14:for all𝑑(𝑗)
𝑖∈Ð
𝑗𝐷𝑗do
15:sim 𝑖←sim(𝑄,𝑑(𝑗)
𝑖)
16:score 𝑖←𝛼·sim𝑖+𝛽·𝑤(𝑗)
𝑖
17:end for
18:𝐷★←Top-𝑘(score 𝑖)
19:ctx←𝑄∥C∥{𝑠 𝑗}𝑗∥{𝑙𝑗}𝑗∥𝐷★
20:𝑟←𝐻(ctx)
21:𝑅←𝐺(𝑄,𝐷★)
22:ℎ𝑅←𝐻(𝑅)
23:Π← rid,C,{W 𝑗}𝑗,{𝑠𝑗}𝑗,
{𝑙𝑗}𝑗, 𝐺, 𝑟, ℎ 𝑅
24:EmitResponse(𝑅,Π)
VII. Security and Privacy Analysis
We analyze the security and privacy of TrustRAG, showing
that the protocol achieves data integrity, vote privacy, trust-
metadata integrity, retrieval verifiability, chain-level consis-
tency, vote uniqueness, cross-chain binding integrity, retrieval
atomicity, and liveness, under standard cryptographic assump-
tions.
A. Assumptions
•A1. Cryptographic hardness.The hash function is colli-
sion resistant, the commitment scheme is computationally
binding and hiding, and the zero-knowledge proof system
is complete, sound, and zero-knowledge.
•A2. Honest-threshold trust domains.In each chain,
the fraction of corrupted validators and committee nodes
stays below the threshold required by the local voting
and aggregation procedures.
•A3. Declared chain set and deterministic ordering.
For each query, all honest parties agree on the ordered
chain setCagainst which completeness and binding are
evaluated.
•A4. Deterministic verification logic.Given the same
public input, all honest contracts and verifiers return

the same result for vote proofs, chain-local state checks,
binding checks, and replay checks.
•A5. Eventual delivery.Messages among honest parties
and access to required chain state are eventually available.
B. Proofs
Theorem 1(Data Integrity).For any document docId𝑡∈𝐷𝑗,
if a Data Chain accepts an aggregate opening (𝑆𝑡,𝑅𝑡)and its
tally commitment ComTally𝑡, the accepted tally is consistent
with the exact set of vote commitments recorded on-chain for
docId𝑡.
Proof.Each vote commitment is recorded on-chain only
after its zero-knowledge proof is verified. The submitted
aggregate is accepted only if the chain verifies the Pedersen
consistency equation 𝑔𝑆𝑡ℎ𝑅𝑡=Î
𝑖Com𝑖,𝑡together with vote-
count and authorization checks. To make the chain accept a
different tally, an adversary would need to alter the finalized
commitment set, produce a second valid opening for the same
commitment product, or bypass verification – contradicting
chain immutability, computational binding, and soundness,
respectively.□
Theorem 2(Vote Privacy).For any honest validator, the
adversary learns nothing about the private vote value 𝑠𝑖,𝑡
beyond what the public protocol output reveals.
Proof.Each validator publishes only a commitment, a
nullifier, and a zero-knowledge proof 𝜋vote. The commitment
is computationally hiding, so it conceals (𝑠𝑖,𝑡,𝑟𝑖,𝑡); the proof
system is zero-knowledge, so 𝜋votereveals nothing beyond the
statement’s truth.□
Theorem 3(Trust-Metadata Integrity Under Poisoning).An
adversary cannot raise a document’s finalized credibility score
beyond what accepted votes imply, unless it corrupts the trust
domain beyond the tolerated threshold or breaks the protocol’s
cryptographic assumptions.
Proof.Inflating a score requires either injecting favorable
votes or manipulating the tally or binding outcome. The first
is blocked by vote validity: only registered validators can vote,
nullifiers prevent duplicate votes, commitment correctness
binds each vote to a concrete score, and range checks reject
out-of-domain values. The second is blocked because a
local aggregate is accepted only if it matches the accepted
commitments, and 𝑊𝑗is accepted only if consistent with the
finalized tally for the candidate set hashed into 𝑙𝑗and𝑠𝑗; the
global digest 𝐺then prevents substituting 𝑊𝑗or the candidate
list undetected. Thus credibility can rise only through accepted
votes, threshold corruption, or broken assumptions – this does
not claim honest validators always score poisoned content
correctly, only that published metadata faithfully reflects the
accepted scoring process.□
Theorem 4(Retrieval Verifiability).For any response returned
withΠ=(C,{𝑊 𝑗}𝑗,{𝑠𝑗}𝑗,{𝑙𝑗}𝑗,𝐺,𝑟,𝐻(𝑅)) , any client can
efficiently verify that the credibility values and final ranking
originated from accepted chain state.Proof.The client fetches finalized chain state for all 𝑗∈C
and checks that each published (𝑊𝑗,𝑠𝑗,𝑙𝑗)is consistent
with accepted local commitments and tallies (soundness
of vote proofs, A4). It recomputes ℎ𝑗=𝐻(𝑠𝑗∥𝑙𝑗)and
𝐺′=𝐻(ℎ 1∥···∥ℎ𝑚), checking𝐺′=𝐺. Using the disclosed
candidates, chain set, and weights, it recomputes ranking
scores, replays the top- 𝑘selection, checks the result against
𝑟, and verifies the response hash. If the service tampers
with scores, substitutes candidates, omits a chain, reranks
incorrectly, or alters the response, one of these checks fails. □
Theorem 5(Chain-Level Consistency).If an honest party
accepts a finalized state 𝑆for a round, no other honest party
accepts a conflicting state𝑆′≠𝑆for that round.
Proof.Local acceptance requires a verified proof and
commitment tuple; since verification is deterministic and
sound, two conflicting local states cannot both be accepted. At
the binding layer, 𝐺is a deterministic function of the ordered
ℎ𝑗’s, so a conflicting global binding would require forging a
local proof, finding a hash collision, or violating finality.□
Theorem 6(Local Vote Validity and Uniqueness).If a
commitment Com𝑖,𝑡is accepted for document docId𝑡, it came
from a registered validator, is bound to a score within range,
and that validator cannot cast a second valid vote for the
same document.
Proof.The vote circuit enforces membership, uniqueness
(via nullifier Null𝑖,𝑡=Poseidon(𝑠𝑘 𝑖,docId𝑡)), commitment
correctness, and range validity simultaneously. By soundness,
any accepted proof implies all four hold; nullifier reuse blocks
double voting, and the range check blocks malformed scores.
□
Theorem 7(Cross-Chain Binding Integrity).If the Aggregator
publishes𝐺over the declared chain set C, every(𝑠𝑗,𝑙𝑗)pair
is bound to a unique ℎ𝑗, and any change to a score, candidate
set, or chain membership is detectable.
Proof.The Aggregator computes ℎ𝑗=𝐻(𝑠𝑗∥𝑙𝑗)for each
chain inCand𝐺over that ordered sequence. Replacing a
score or candidate-list hash changes ℎ𝑗except with negligible
probability under collision resistance; omitting or reordering
a chain likewise changes𝐺.□
Theorem 8(Retrieval Atomicity).For any query 𝑄, the
response𝑅is generated only from a fully verified evidence
set𝐷★with a validΠ, or no response is accepted.
Proof.The service does not finalize generation on partial
outputs; it waits until all required (𝑊𝑗,𝑠𝑗,𝑙𝑗)tuples are
obtained and 𝐺is published. Only after verifying local state
and the global binding does it derive the ranking, compute
𝑟, and invoke the generator. If any required result is invalid,
missing, or inconsistent, local verification, the binding check,
or the replay check fails, and no validΠcan be formed.□
Theorem 9(Liveness).Assume corrupted validators stay
below threshold in each chain and messages between honest

parties are eventually delivered. Then every valid query
eventually yields either a finalized response with a valid Π,
or an explicit rejection from proof failure, binding mismatch,
or insufficient evidence.
Proof.A valid query is dispatched to C. Under A2, A4,
A5, each honest chain eventually exposes valid local state
or a failure indication. These are relayed to the Aggregator,
which deterministically computes 𝐺once all inputs arrive in
canonical order. The service then either receives valid chain-
local state with a matching 𝐺and completes generation, or
detects failure and returns rejection.□
VIII. Implementation and Evaluation
We implemented a prototype realizing the main components
of the design and evaluate its overhead. The vote-validity
and score-consolidation path uses a zero-knowledge circuit
incircom 2.1.6 , with on-chain verification via Solidity
contracts on a Hardhat EDR simulated network. Committee-
side aggregation uses Shamir secret sharing and an MP-SPDZ-
compatible secure-sum interface, with threshold authorization
performed off-chain before aggregate submission.1
A. Prototype Components
The prototype consists of four components.
•Document and retrieval service.An off-chain service
stores document contents, maintains the retrieval index,
derives per-chain candidate sets, and performs replayable
reranking.
•Validator-scoring path.A zero-knowledge vote circuit
enforces validator membership, nullifier uniqueness, com-
mitment correctness, and score range validity before a
vote is accepted.
•Committee aggregation path.Committee nodes receive
Shamir shares of hidden score components, perform
secure-sum aggregation, and authorize the final aggregate
once the threshold is met.
•On-chain verification and binding path.Solidity
contracts maintain commitment products, accepted vote
counts, aggregate summaries, and the cross-chain binding
digests(ℎ𝑗,𝐺), rejecting any submission that fails the
Pedersen consistency or authorization checks.
B. Prototype Boundary
The implementation directly exercises the security-critical
chain-local tally path and the publication of replay metadata
consumed by the service layer. Committee aggregation uses a
𝑡-of-𝑛Shamir-sharing domain with 𝑛=4 ,𝑡=3 . The prototype
fully implements the vote-validity circuit, the contract-level
commitment-consistency checks, the disclosure of finalized
per-document weights, and the binding metadata (𝑠𝑗,𝑙𝑗,𝐺)
used for replay – sufficient to demonstrate the end-to-end
vote-validation path, the commitment-consistency check, and
retrieval-time use of authenticated replay metadata, without
recursive proof aggregation.
1https://github.com/1Vastsky/trustRAGTABLE II
Proving time comparison for the vote circuit under different Merkle
depths.
Depth Groth16 (ms) PLONK (ms)
8 865.73 5,183.92
12 1,023.82 9,930.46
16 1,486.58 9,998.97
20 1,624.51 9,946.89
C. Experimental Setup
We evaluate two parts of the system. First, we benchmark
the zero-knowledge vote-validity circuit under Groth16 and
PLONK. Second, we measure the cost of the committee-
based chain-local tally path, including Pedersen commitments,
Shamir share splitting and reconstruction, and threshold
authorization. Unless otherwise stated, we use a score bound
𝑆max=10, a committee size of 4, and threshold 3.
All experiments were conducted on a MacBook Pro
equipped with an Apple M1 Pro processor and 32GB of
memory, running macOS. The zk modules are implemented
with Circom and snarkjs , while the blockchain path is
implemented with Solidity contracts and exercised through
Hardhat. We evaluate the protocol in a modular manner: vote-
proof generation, chain-local tallying, threshold authorization,
and committee reconstruction are measured separately and
then interpreted as composable building blocks of the full
system.
D. Proving Time
We first compare the proving latency of Groth16 and
PLONK for the same Rvotecircuit under different Merkle
depths. Table II summarizes the results. As expected, Groth16
achieves substantially lower proving time across all tested
depths, while PLONK incurs a much larger overhead due to
its universal constraint system and polynomial-commitment
machinery.
Even at depth 20, Groth16 remains in the low-second regime,
which is acceptable because proof generation happens only
at vote-submission time rather than on the online query path.
In contrast, PLONK becomes significantly more expensive
in this setting, suggesting that Groth16 is the more practical
choice for the current prototype.
E. Circuit Scaling with Merkle Depth
To evaluate scalability with respect to validator-set size,
we vary the Merkle tree depth while keeping the rest
of the vote circuit unchanged. Concretely, we instantiate
VoteCircuit(depth) for depths ranging from 8to50,
compile each variant to R1CS, and record the resulting circuit
artifacts.
The artifact sizes exhibit a stable near-linear growth trend as
the Merkle depth increases. In particular, the R1CS size grows
from 702 KB at depth 8to 3.6 MB at depth 50, while the
compiled WASM artifact remains within a relatively narrow
range of approximately 1.7–1.9 MB. This suggests that the

TABLE III
Circuit artifact scaling with Merkle tree depth.
Depth #Constraints Nonlinear Linear Wires R1CS size (KB) WASM size (KB)
8 5,251 2,488 2,763 5,263 702 1,738
12 7,339 3,472 3,867 7,355 981 1,751
16 9,427 4,456 4,971 9,447 1,260 1,763
20 11,515 5,440 6,075 11,539 1,538 1,774
24 13,604 6,486 7,118 13,628 1,812 1,786
28 15,694 7,493 8,201 15,721 2,092 1,799
32 17,785 8,523 9,262 17,812 2,365 1,813
40 21,968 10,537 11,431 22,002 2,926 1,842
50 27,181 13,045 14,136 27,221 3,622 1,876
TABLE IV
Chain-local tally overhead with committee size4and threshold3.
Votes Latency (ms) Commit (ms) Split+Recon (ms) BLS (ms)
20 1123.24 0.10 0.40 1122.66
50 1117.12 0.24 0.87 1115.87
100 1118.86 0.47 1.74 1116.39
dominant scaling cost comes from the constraint system itself
rather than from an explosion in executable representation
size.
F. Chain-Local Tally Overhead
We next measure the cost of the chain-local tally path under
committee-based secure aggregation. Table IV reports end-
to-end latency for 20, 50, and 100 validators. The results
show that the dominant cost in the current prototype comes
from threshold authorization, while Pedersen commitments
and Shamir operations remain small compared with the total
tally delay.
This profile is consistent with the intended design: commit-
ment and Shamir operations scale gently with the number of
votes, while the dominant cost in the current prototype lies in
finalizing the chain-local tally.
G. Committee Scaling
To complement the fixed-size tally benchmark above, we
also evaluate how the two committee-critical subroutines scale
with committee size: BLS threshold-signature aggregation and
Shamir reconstruction. Figure 3 visualizes the measurements
for committee sizes from 4to48under three threshold ratios,
𝑡/𝑛∈{50%,75%,100%}.
Two patterns are clear. First, both subroutines scale approx-
imately linearly with the effective threshold size. Second, the
dominant committee-side cost comes from BLS aggregation
rather than Shamir reconstruction. Even in the most conser-
vative configuration with 𝑛=48 and𝑡=𝑛 , BLS aggregation
remains below 300 ms and Shamir reconstruction remains
close to 1.1 ms. These results indicate that the committee layer
remains practical at moderate scale and does not invalidate
the system’s low-latency objective.TABLE V
Ablation results for the chain-local tally path at 100 votes.
Variant Lat. Msgs Tamper Unauth. Dropout
(ms) det. det. resil.
Full 1118.86 400✓ ✓ ✓
No Pedersen 1115.26 400✗✓ ✓
No Shamir 1113.20 0✓ ✓✗
No BLS 2.43 400✓✗✓
H. Ablation Study
To understand the role of each cryptographic mechanism,
we compare the full system against three ablated variants:
one without Pedersen commitment consistency checking,
one without Shamir sharing, and one without threshold
authorization. Table V summarizes the results for 100 votes
in the chain-local tally path.
Removing Pedersen commitments disables detection of
tampered aggregate openings. Removing Shamir sharing elim-
inates threshold-based resilience to committee-node dropout.
Removing threshold authorization sharply reduces latency,
but at the cost of losing the ability to detect unauthorized
aggregate submission. These results are consistent with the
intended role of each mechanism in the overall trust pipeline.
I. Discussion
These results show that TrustRAG achieves a strong
security profile for verifiable RAG. The protocol provides
state consistency through deterministic on-chain verification,
input legitimacy through vote-validity proofs and nullifier-
based uniqueness, cross-chain integrity through hash binding,
retrieval atomicity by generating only after verification, privacy
through hidden commitments and zero-knowledge proofs, and
auditability through publicly verifiable replay metadata. The
overhead measurements above indicate that these guarantees
can be added to a RAG pipeline without prohibitive cost: proof
generation and committee authorization – the dominant costs
– occur off the online query path, while online retrieval and
cross-chain binding remain lightweight.
IX. Conclusion
This paper presented TrustRAG, a committee-based, de-
centralized, and verifiable RAG architecture. Documents are

0 10 20 30 40 500123·105
Committee size𝑛Aggregation cost (𝜇s)𝑡/𝑛=50%
𝑡/𝑛=75%
𝑡/𝑛=100%
BLS aggregation0 10 20 30 40 5005001,000
Committee size𝑛Reconstruction cost (𝜇s)𝑡/𝑛=50%
𝑡/𝑛=75%
𝑡/𝑛=100%
Shamir reconstruction
Fig. 3. Committee-scaling costs under different threshold ratios. Left: BLS threshold-signature aggregation. Right: Shamir reconstruction.
certified by an expert committee through zero-knowledge
scoring and commitment-consistent secure aggregation, pro-
ducing reusable trust scores that are bound across chains via
lightweight hash commitments and verified through determinis-
tic replay, without recursive aggregation proofs. Our evaluation
shows the dominant overhead falls in the score-finalization
and committee-authorization stages, while online retrieval and
cross-chain coordination remain lightweight. TrustRAG is well
suited to high-stakes domains such as healthcare, where the
provenance and trustworthiness of retrieved knowledge are
critical and require verification by domain experts.
Acknowledgment
This work was supported by the Henan Province Key
Research and Development Special Project (Project No.
251111210400).
The authors used ChatGPT and Claude to assist with
language polishing and editing of this manuscript. The authors
take full responsibility for the accuracy and correctness of the
paper.
References
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. K¨ uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel,et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[2]N. Garg, “Evaluating copa congestion control for improved video
performance.” https://engineering.fb.com/2019/11/17/video-engineering/
copa/?utm source=chatgpt.com, Nov. 17 2019. Engineering at Meta
blog post.
[3]J. Abreu, P. Bergeron, and S. Aneja, “Should bbr be the default tcp
congestion control protocol?,”arXiv preprint arXiv:2510.22461, 2025.
Submitted on 25 Oct 2025.
[4]T. Chen, H. Wang, S. Chen, W. Yu, K. Ma, X. Zhao, H. Zhang, and
D. Yu, “Dense x retrieval: What retrieval granularity should we use?,”
inProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, pp. 15159–15177, 2024.
[5]F. Fang, Y. Bai, S. Ni, M. Yang, X. Chen, and R. Xu, “Enhancing
noise robustness of retrieval-augmented language models with adaptive
adversarial training,”arXiv preprint arXiv:2405.20978, 2024.
[6]K. Wu, E. Wu, and J. Zou, “Clasheval: Quantifying the tug-of-war
between an llm’s internal prior and external evidence,”arXiv preprint
arXiv:2404.10198, 2024.
[7]Y. Liu, L. Huang, S. Li, S. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun,
“Recall: A benchmark for llms robustness against external counterfactual
knowledge,”arXiv preprint arXiv:2311.XXXXX, 2023.[8]Y. Zhou, Y. Liu, X. Li, J. Jin, H. Qian, Z. Liu, C. Li, Z. Dou, T.-Y.
Ho, and P. S. Yu, “Trustworthiness in retrieval-augmented generation
systems: A survey,”arXiv preprint arXiv:2409.10102, 2024.
[9]A. Deshpande, V. Murahari, T. Rajpurohit, A. Kalyan, and
K. Narasimhan, “Toxicity in chatgpt: Analyzing persona-assigned
language models,” inFindings of the Association for Computational
Linguistics, pp. 1236–1270, 2023.
[10] F. Perez and I. Ribeiro, “Ignore previous prompt: Attack techniques for
language models,”arXiv preprint arXiv:2211.09527, 2022.
[11] H. Chaudhari, G. Severi, J. Abascal, M. Jagielski, C. A. Choquette-
Choo, M. Nasr, C. Nita-Rotaru, and A. Oprea, “Phantom: General trigger
attacks on retrieval augmented language generation,”arXiv preprint
arXiv:2405.20485, 2024.
[12] A. Shafran, R. Schuster, and V. Shmatikov, “Machine against the rag:
Jamming retrieval-augmented generation with blocker documents,”arXiv
preprint arXiv:2406.05870, 2024.
[13] T. E. Andersen, A. M. Avalos, G. G. Dagher, and M. Long, “D-RAG: A
privacy-preserving framework for decentralized RAG using blockchain,”
inProceedings of the 15th International Conference on Computer
Science and Information Technology (CSIT), pp. 183–198, Academy &
Industry Research Collaboration, February 2025.
[14] J. Yu and H. Sato, “Derag: Decentralized multi-source rag system with
optimized pyth network,” in2024 IEEE International Symposium on
Parallel and Distributed Processing with Applications (ISPA), pp. 106–
115, IEEE, 2024.
[15] Y. Lu, W. Tang, M. Johnson, T. Jung, and M. Jiang, “A decentralized
retrieval augmented generation system with source reliabilities secured
on blockchain,” 2025.
[16] C. Lin, Y. Cui, Z. Zhou, C. Hong, Y. Wang, Z. Chen, and M. Li,
“VeriRAG: Efficient zero-knowledge proofs for verifiable retrieval-
augmented generation.” Cryptology ePrint Archive, Paper 2026/637,
2026.
[17] S. Shukla and H. Joshi, “Proof-carrying answers: A systematic protocol
for verifiable retrieval-augmented generation with cryptographic prove-
nance,” in2025 Annual Computer Security Applications Conference
Workshops (ACSAC Workshops), pp. 405–413, IEEE, 2025.
[18] G. Izacard and E. Grave, “Leveraging passage retrieval with gener-
ative models for open-domain question answering,”arXiv preprint
arXiv:2007.01282, 2020.
[19] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[20] S. Goldwasser, S. Micali, and C. Rackoff, “The knowledge complexity
of interactive proof systems,”SIAM Journal on Computing, vol. 18,
no. 1, pp. 186–208, 1989.
[21] J. Groth, “On the size of pairing-based non-interactive arguments,” in
EUROCRYPT, pp. 305–326, 2016.
[22] A. Gabizon, Z. J. Williamson, and O. Ciobotaru, “Plonk: Permuta-
tions over lagrange-bases for oecumenical noninteractive arguments of
knowledge,” inIACR ePrint Archive, 2019.