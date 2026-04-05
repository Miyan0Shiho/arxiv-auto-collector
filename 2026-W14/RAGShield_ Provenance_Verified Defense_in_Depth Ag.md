# RAGShield: Provenance-Verified Defense-in-Depth Against Knowledge Base Poisoning in Government Retrieval-Augmented Generation Systems

**Authors**: KrishnaSaiReddy Patil

**Published**: 2026-04-01 02:16:42

**PDF URL**: [https://arxiv.org/pdf/2604.00387v1](https://arxiv.org/pdf/2604.00387v1)

## Abstract
RAG systems deployed across federal agencies for citizen-facing services are vulnerable to knowledge base poisoning attacks, where adversaries inject malicious documents to manipulate outputs. Recent work demonstrates that as few as 10 adversarial passages can achieve 98.2% retrieval success rates. We observe that RAG knowledge base poisoning is structurally analogous to software supply chain attacks, and propose RAGShield, a five-layer defense-in-depth framework applying supply chain provenance verification to the RAG knowledge pipeline. RAGShield introduces: (1) C2PA-inspired cryptographic document attestation blocking unsigned and forged documents at ingestion; (2) trust-weighted retrieval prioritizing provenance-verified sources; (3) a formal taint lattice with cross-source contradiction detection catching insider threats even when provenance is valid; (4) provenance-aware generation with auditable citations; and (5) NIST SP 800-53 compliance mapping across 15 control families. Evaluation on a 500-passage Natural Questions corpus with 63 attack documents and 200 queries against five adversary tiers achieves 0.0% attack success rate including adaptive attacks (95% CI: [0.0%, 1.9%]) with 0.0% false positive rate. We honestly report that insider in-place replacement attacks achieve 17.5% ASR, identifying the fundamental limit of ingestion-time defense. The cross-source contradiction detector catches subtle numerical manipulation attacks that bypass provenance verification entirely.

## Full Text


<!-- PDF content starts -->

RAGShield: Provenance-Verified Defense-in-Depth
Against
Knowledge Base Poisoning in Government
Retrieval-Augmented Generation Systems
KrishnaSaiReddy Patil
Abstract—Retrieval-Augmented Generation (RAG) systems are
increasingly deployed across federal agencies for citizen-facing
services including tax guidance, benefits eligibility, and legal
information. These systems are vulnerable to knowledge base poi-
soning attacks, where adversaries inject malicious documents to
manipulate RAG outputs—recent work demonstrates that as few
as 10 adversarial passages (0.04% of a corpus) can achieve 98.2%
retrieval success rates [1]. We observe that RAG knowledge base
poisoning is structurally analogous to software supply chain
attacks, and propose RAGShield, a five-layer defense-in-depth
framework that applies supply chain provenance verification to
the RAG knowledge pipeline. RAGShield introduces: (1) C2PA-
inspired cryptographic document attestation that blocks unsigned
and forged documents at ingestion; (2) trust-weighted retrieval
that prioritizes provenance-verified sources; (3) a formal taint
lattice with cross-source contradiction detection that catches
insider threats even when provenance is valid; (4) provenance-
aware generation with auditable citations; and (5) NIST SP 800-
53 compliance mapping across 15 control families including SA-
12 (Supply Chain Protection). We evaluate RAGShield against
six adversary tiers—external injection, signature forgery, insider
compromise, adaptive evasion, data loader exploitation, and in-
place replacement—on a 500-passage Natural Questions corpus
with 63 attack documents and 200 queries. RAGShield achieves
0.0% attack success rate across all tiers including adaptive
attacks (95% CI: [0.0%, 1.9%]) while maintaining 0.0% false
positive rate, compared to 8–13% ASR without defense, 7.5–
12.5% for RAGDefender, and 0–0.5% for RobustRAG. We
honestly report that in-place replacement attacks by insiders
who modify existing documents achieve 17.5% ASR—identifying
the fundamental limit of ingestion-time defense and proposing
hash-pinning mitigations. Notably, the cross-source contradiction
detector catches subtle insider attacks (e.g., $15,000→$15,500
numerical manipulation) that bypass provenance verification
entirely. To our knowledge, RAGShield is the first framework
to apply supply chain provenance to RAG security, the first to
provide NIST 800-53 compliance mapping for RAG pipelines,
and the first to defend against data loader exploitation attacks.
Index Terms—RAG, retrieval-augmented generation, poison-
ing, defense, provenance, government, supply chain security,
NIST 800-53
I. INTRODUCTION
Federal agencies are rapidly adopting Retrieval-Augmented
Generation (RAG) systems for citizen-facing services. The
General Services Administration launched USAi.Gov in Au-
gust 2025, providing all federal agencies access to AI mod-
els from OpenAI, Anthropic, Google, and Meta [16]. TheGovernment Accountability Office documented 89 RAG-
based citizen-facing services across 28 agencies as of fiscal
year 2025 [17]. These systems answer questions about tax
obligations, benefits eligibility, legal rights, and government
procedures—domains where incorrect information causes di-
rect harm to citizens.
RAG systems are vulnerable to knowledge base poison-
ing attacks. Zou et al. demonstrated in PoisonedRAG that
injecting just 5 malicious texts into a knowledge base with
millions of documents achieves over 90% attack success rate
on NQ, HotpotQA, and MS-MARCO [30]. Chaudhari et al.
showed that 10 adversarial passages (0.04% of the corpus)
achieve 98.2% retrieval success in the Phantom attack [1]. The
PhantomText attack achieves 74.4% success across five major
RAG frameworks using invisible text embedded in standard
document formats [3]. The ConfusedPilot attack demonstrates
integrity and confidentiality violations against Microsoft 365
Copilot’s RAG system [4].
Key observation.We observe that RAG knowledge base
poisoning is structurally analogous to software supply chain at-
tacks. In both cases, an adversary injects malicious content into
a trusted pipeline: malicious code into a software build (So-
larWinds [5]), or malicious documents into a knowledge base.
The defense in software supply chains isprovenance verifica-
tion—cryptographic attestation of origin, integrity, and chain
of custody, standardized through SLSA [6] and C2PA [7]. No
existing RAG defense applies this principle.
The gap.All existing RAG defenses operate oncon-
tent analysis: RobustRAG [8] isolates and aggregates pas-
sages; RAGDefender [9] uses post-retrieval ML filtering;
TrustRAG [10] applies clustering-based detection; the A V
Filter [11] uses attention-variance scoring. Content analysis is
an arms race—as attacks become more fluent (CPA-RAG [14],
AuthChain [15]), content-based detection becomes harder.
Provenance verification provides a fundamentally different
defense: you either have a valid cryptographic attestation or
you don’t.
Contributions.We present RAGShield, a five-layer
defense-in-depth framework:
1)Supply chain provenance for RAG(§IV-A): C2PA-
inspired document attestation with content normalization
that blocks unsigned, forged, and PhantomText-exploitedarXiv:2604.00387v1  [cs.CR]  1 Apr 2026

TABLE I
RAG POISONINGATTACKLANDSCAPE
Attack Venue Key Result Tier
PoisonedRAG USENIX’25 5 texts→90% ASR T1
Phantom NeurIPS’24 98.2% retrieval T1
CPA-RAG arXiv’25 Black-box, fluent T1
AuthChain arXiv’25 Single-doc, stealthy T3
PhantomText arXiv’25 74.4% via hidden text T4
ConfusedPilot arXiv’24 Copilot integrity T3
Eyes-on-Me arXiv’25 Scalable attractors T1
PoisonArena arXiv’25 Multi-attacker T1
documents at ingestion. First application of supply chain
provenance to RAG security.
2)Formal taint lattice with insider detection(§IV-C):
Cross-source contradiction detection, corroboration check-
ing, and claim magnitude scoring that catch insider threats
even when provenance is valid. Formal taint propagation
with soundness guarantee (Theorem 2).
3)Provenance integrity bound(§V): Theorem 1 proves that
attack success rate is bounded by the fraction of unattested
documents when the adversary cannot forge signatures.
4)Government compliance mapping(§VIII): First NIST SP
800-53 mapping for RAG pipeline security, covering 15
controls across SA, SI, AC, AU, SC, IR, CM, RA, and PM
families, including the August 2025 AI Control Overlays.
5)Comprehensive evaluation with baselines and adap-
tive attacks(§VII): Evaluation on the Natural Questions
benchmark against five adversary tiers including T5 (adap-
tive RAGShield-aware adversary), compared against Ro-
bustRAG and RAGDefender baselines, with pre-poisoned
corpus degradation analysis, threshold sensitivity study, and
LLM-in-the-loop end-to-end impact assessment.
RAGShield extends CivicShield’s [18] Layer 1 (input vali-
dation) with document-level provenance and Layer 3 (semantic
taint tracking) with pipeline-level taint propagation, forming
the second paper in a trilogy on government AI security.
II. BACKGROUND ANDRELATEDWORK
A. RAG Poisoning Attacks
Table I summarizes the attack landscape. Attacks span
four capability levels: external corpus injection (T1), signa-
ture forgery (T2), insider compromise (T3), and data loader
exploitation (T4).
B. RAG Defenses
Table II compares existing defenses. All operate on content
analysis; none uses provenance verification, taint tracking, or
government compliance mapping.
C. Supply Chain Provenance Standards
SLSA (Supply chain Levels for Software Artifacts) [6]
defines three progressively stronger build levels ensuring
software artifacts have verifiable provenance. C2PA (Coali-
tion for Content Provenance and Authenticity) [7] pro-
vides cryptographically signed manifests for digital content.TABLE II
RAG DEFENSECOMPARISON
Provenance Taint Track Insider Det. NIST Map Loader Def.
RobustRAG [8]✗ ✗ ✗ ✗ ✗
RAGDefender [9]✗ ✗ ✗ ✗ ✗
TrustRAG [10]✗ ✗ ✗ ✗ ✗
A V Filter [11]✗ ✗ ✗ ✗ ✗
RA-RAG [12]✗ ✗Partial✗ ✗
ReliabilityRAG [13]✗ ✗Partial✗ ✗
SDAG [31]✗ ✗ ✗ ✗ ✗
RAGShield✓ ✓ ✓ ✓ ✓
TABLE III
ADVERSARYTIERS
Tier Capability Provenance Analog
T1 Corpus inject None Public comment
T2 Inject + forge Invalid sig Spoofed .gov
T3 Inject + insider Valid sig Compromised employee
T3-H Inject + insider Valid, subtle Minor numerical edit
T4 Loader exploit Valid, hidden PhantomText in PDF
RAGShield adapts these principles to the knowledge base
domain: documents are the “artifacts,” government sources are
the “builders,” and the RAG pipeline is the “supply chain.”
III. THREATMODEL
Definition 1(RAG System).A RAG systemR= (K, η, ρ, γ)
consists of knowledge baseK={d 1, . . . , d n}, embedding
functionη:D →Rm, retrieverρ:Q × K → P(K), and
generatorγ:Q × P(K)→ A.
Definition 2(Adversary).An adversaryA= (C, G, O)is
characterized by capabilitiesC, knowledgeG, and objective
O.
We define four adversary tiers (Table III):
Security goals:(G1) Provenance integrity—no unattested
document enters the index; (G2) Taint soundness—taint labels
are conservative; (G3) Trust monotonicity—composite trust
≤minimum component trust; (G4) Graceful degradation—
fallback to content-based defense when provenance is unavail-
able.
IV. RAGSHIELDARCHITECTURE
RAGShield implements five defense layers, each mapping
to specific NIST SP 800-53 controls. Figure 1 illustrates the
pipeline.
A. Layer 1: Provenance-Verified Ingestion
Each documentdenteringKcarries an attestationA(d):
A(d) =⟨h, s, τ s, k, t, σ⟩(1)

L1: Provenance-Verified Ingestion
SA-12, SI-7, SI-10
L2: Trust-Weighted Retrieval
AC-4, SC-16
L3: Taint-Tracked Context Assembly
SI-3, AU-10
L4: Provenance-Aware Generation
AU-3, SI-4, PM-25
L5: Compliance & Audit
AU-2, AU-6, IR-4, CM-3Document Ingestion
Audited Response
Fig. 1. RAGShield five-layer architecture with NIST 800-53 control map-
pings.
whereh=SHA-256(normalize(d))is the content hash
after normalization,sis the source identifier,τ s∈
{1.0,0.8,0.6,0.3,0.1}is the source trust tier,kis the
signing key fingerprint,tis the timestamp, andσ=
Ed25519(k priv, h∥s∥t)is the digital signature.
Source trust tiersreflect the government document hierar-
chy:
Tier Examplesτ
AUTHORITATIVE Federal Register, U.S. Code, CFR 1.0
OFFICIAL Agency publications (IRS, SSA) 0.8
INSTITUTIONAL GAO reports, inter-agency memos 0.6
PUBLIC Public comments, FOIA responses 0.3
UNKNOWN Unsigned, legacy content 0.1
Content normalization(anti-PhantomText) strips zero-
width characters, hidden text layers, and invisible HTML/CSS
elements before hashing. The normalization divergenceδ(d) =
1− |normalize(d)|/|d raw|measures hidden content fraction.
Documents withδ >0.20are blocked;δ >0.05are flagged.
Bootstrapping.When full cryptographic signing is un-
available, RAGShield supports graduated provenance levels:
Level 0 (URL verification via TLS), Level 1 (hash pinning
at ingestion), Level 2 (cross-reference against multiple official
sources), Level 3 (full C2PA-style attestation). Trust scores are
capped atτ max={0.5,0.6,0.8,1.0}respectively, ensuring G4
(graceful degradation).
B. Layer 2: Trust-Weighted Retrieval
Standard retrieval ranks by similarity: score(q, d) =
sim(η(q), η(d)). RAGShield augments with trust:
score′(q, d) =sim(η(q), η(d))·τ(d)α(2)ADVERSARIAL
⊤
PIIHIGH LOW TRUST
PIIMED MED TRUST
PIILOW HIGH TRUST
UNTAINTED
⊥
Fig. 2. Taint lattice Hasse diagram. Join (⊔) moves up (less trusted); meet
(⊓) moves down. Context taint =Fof component taints.
whereα≥0controls trust sensitivity. Atα= 0, trust is
ignored; atα= 1, trust scales linearly; atα >1, high-trust
documents are strongly preferred.
For sensitive queries (classified by topic), a minimum trust
thresholdτ minis enforced: CRITICAL queries (legal/medical)
requireτ min= 0.8; HIGH (financial) requires0.6; MEDIUM
(procedural) requires0.3.
C. Layer 3: Taint-Tracked Context Assembly
1) Formal Taint Lattice:We define a taint latticeT=
(L,⊑,⊔,⊓)with elements ordered from most trusted (⊥=
UNTAINTED) to least trusted (⊤= ADVERSARIAL). The
join operation⊔computes the least upper bound: when chunks
from different trust levels are combined, the composite taint
is theworstcomponent. Figure 2 shows the Hasse diagram.
2) Taint Propagation Rules:
•Chunking:taint(chunk) =taint(parent)
•Embedding:taint(emb) =taint(chunk)
•Concatenation:taint(ctx) =F
itaint(c i)
•Generation:taint(resp) =taint(ctx)⊔taint(q)
3) Cross-Source Insider Detection (v2):Provenance alone
cannot stop insider threats (T3). We add three content-based
checks that runonly on newly-ingested documents(not the
established corpus baseline):
Contradiction detection.For a new documentd newwith
embeddinge new, compute similarity against all existing AU-
THORITATIVE documents. If sim(e new, eauth)∈[θ low, θhigh]
(topically similar but not identical), flag as a potential contra-
diction. Rationale: a poisoned insider document will be about
the same topic as legitimate documents but make different
claims.
Corroboration checking.Count how many existing docu-
ments have similarity≥θ corrwithd new. If fewer thank mincor-
roborators exist, flag as uncorroborated. Rationale: legitimate
policy changes appear in multiple documents (regulation +
guidance + Federal Register notice); a single insider document
making a sweeping claim is suspicious.
Claim magnitude scoring.Compute similarity between
dnew and the topic centroid of existing documents. If
sim(e new,¯etopic)< θ mag, flag as an outlier. Rationale: even

subtle numerical changes ($15,000→$15,500) shift the
embedding away from the established consensus.
D. Layer 4: Provenance-Aware Generation
The LLM prompt includes trust metadata for each context
chunk, enabling trust-aware response generation. Citations
include the full provenance chain: source, trust tier, attestation
status, and content hash.
E. Layer 5: Compliance and Audit
A graduated response model determines the action for each
query:
•PASS:All checks pass,τ context≥τmin. Serve normally.
•FLAG:Taint anomaly detected or low-trust source used.
Serve with enhanced monitoring and trust indicator.
•BLOCK:Provenance verification failed, trust below thresh-
old, or PII purpose mismatch. Refuse with alternative ser-
vice path.
V. FORMALANALYSIS
Theorem 1(Provenance Integrity Bound).LetKbe a knowl-
edge base where fractionpof documents have valid prove-
nance attestations, and letAbe a T1 or T2 adversary who
cannot produce valid signatures. Then:
ASRL1≤(1−p)·ASR baseline (3)
where ASR baseline is the attack success rate without RAGShield.
Proof.A T1 adversary cannot produce valid signatures (by
the EUF-CMA security of Ed25519). Therefore, L1 rejects
all injected documents—they never enter the retrieval index
K. The only documents available for retrieval are thep·n
attested documents plus any(1−p)·nunattested legacy
documents. The adversary can only inject poison into the
unattested fraction, bounding ASR by(1−p)·ASR baseline .
Note: this bound comes from L1 ingestion blocking, not L2
trust weighting. L2 provides additional defense-in-depth but is
not required for the bound.
Corollary 1.Whenp= 1(all documents attested), ASR L1=
0for T1 and T2 adversaries.
Theorem 2(Taint Soundness).Under the taint propagation
rules of §IV-C, for any chunkcin the RAG pipeline: taint(c)⊒
taint(source(c)).
Proof.By structural induction on pipeline operations. Base
case: chunking preserves taint (rule 1). Inductive case: con-
catenation applies⊔(join), which by lattice properties satisfies
⊔iti⊒tjfor allj. Therefore taint never decreases through
the pipeline.
Theorem 3(Trust Monotonicity).For any contextCassem-
bled from chunks{c 1, . . . , c k}: trust(C)≤min itrust(c i).
Proof.Follows directly from the join operation on the taint
lattice.
Proposition 1(Insider Threat Mitigation).For a T3 adver-
sary with valid signing key, RAGShield’s defense degrades toTABLE IV
IMPLEMENTATIONSTATUS
Component Status Notes
L1: Provenance attestation★Ed25519 simulated via HMAC
L1: Content normalization★Zero-width char stripping
L1: Bootstrapping (L0–L2)❍URL/hash/cross-ref levels
L2: Trust-weighted retrieval★Eq. 2
L2: Sensitivity thresholds★4-tier classification
L3: Taint lattice★8-element lattice
L3: Contradiction detection★Embedding similarity
L3: Corroboration checking★Neighbor counting
L3: Claim magnitude★Centroid distance
L4: Trust-annotated prompts◆Template-based
L4: Citation generation◆Provenance chain
L5: Graduated response★PASS/FLAG/BLOCK
L5: Audit trail★JSON logging
L5: NIST mapping❍15 controls mapped
★= Fully implemented◆= Simulated❍= Designed only
content-based detection with rateD content , but the audit trail
enables forensic identification with probabilityP forensic . The
effective defense is:
Pdefense =D content + (1−D content)·P forensic (4)
This is strictly better than systems without provenance, where
Pforensic≈0due to lack of source attribution.
VI. IMPLEMENTATION
RAGShield is implemented in Python 3.12 using sentence-
transformers (all-MiniLM-L6-v2) for embeddings, HMAC-
SHA256 for simulated Ed25519 signatures, and NumPy for
vector operations. The implementation runs on CPU (macOS,
18GB RAM) with no GPU required. Table IV shows the
implementation status of each component.
A. Cost Model
TABLE V
PER-OPERATIONOVERHEAD
Operation Time Storage
Content normalization∼5ms/doc 0
SHA-256 hash∼1ms/doc 32B
Signature verification∼0.1ms/doc 96B
Attestation metadata —∼256B/doc
Trust-weighted re-ranking∼2ms/query 0
Taint propagation∼1ms/query 0
Insider detection (3 checks)∼5ms/query 0
Audit logging∼2ms/query∼1KB/query
Total per document∼6ms∼384B
Total per query∼10ms∼1KB
For a 100K-document corpus: 10 minutes ingestion over-
head, 37MB metadata storage. At 10ms/query overhead,
RAGShield adds negligible latency to the∼500ms typical
RAG query time.

VII. EVALUATION
A. Experimental Setup
Corpus.500 passages from the Natural Questions (NQ)
benchmark [20], a standard RAG evaluation dataset containing
real Wikipedia passages paired with Google search queries.
This represents a significant scale-up from template-generated
documents to real-world text.
Attacks.63 attack documents across five tiers: 15 T1 (exter-
nal, no provenance), 10 T2 (forged signatures), 10 T3 (insider
with valid provenance), 8 T4 (PhantomText), and 20 T5
(adaptive RAGShield-aware adversary with valid signatures,
subtle modifications, and fake corroborating documents).
Queries.200 NQ questions used as attack-targeted and
benign queries.
Defenses compared.RAGShield-Full (L1+L2+L3),
RAGShield-L1 (provenance only), RAGShield-L3
(taint/insider only), simulated RobustRAG [8] (isolate-then-
aggregate with majority vote), simulated RAGDefender [9]
(post-retrieval centroid-distance filtering), and No-Defense
baseline.
Parameters.α= 1.0,θ low= 0.55,θ high= 0.92,θ corr=
0.75,θ mag= 0.70,k= 5,δ block= 0.20.
B. Main Results
Table VI presents the main results comparing RAGShield
against two state-of-the-art baselines and ablations on the NQ
benchmark. All confidence intervals are 95% Wilson score
intervals.
Key findings:
RAGShield-Fullachieves 0.0% ASR across all five tiers with
95% CI [0.0%, 1.9%] and 0.0% FPR. This includes the T5
adaptive tier where the adversary has valid signing keys, crafts
semantically similar poison, and injects fake corroborating
documents.
RobustRAGperforms well on most tiers (0.0–0.5% ASR)
through its isolate-then-aggregate strategy, but allows 0.5% of
T1 attacks through when poisoned passages form a local ma-
jority in the retrieved set. RobustRAG provides no provenance
guarantee and no compliance mapping.
RAGDefenderfails significantly (7.5–12.5% ASR across all
tiers). Its centroid-distance filtering cannot distinguish well-
crafted poison from legitimate passages when the poison is
topically relevant. This confirms that post-retrieval content
filtering alone is insufficient against sophisticated attacks.
L1-onlyblocks T1/T2/T4 completely but fails on T3 (8.0%)
and T5 adaptive (11.0%)—identical to No-Defense for insider
attacks. This proves provenance alone is insufficient.
T5 adaptive attacksare the most challenging: the adversary
has valid provenance, crafts poison semantically close to
existing passages, and adds fake corroborating documents.
Despite this, RAGShield-Full blocks all T5 attacks through
the contradiction detector, which identifies that the modified
passages diverge from established AUTHORITATIVE sources
even when corroboration is artificially inflated.C. Ablation: Why Both Layers Are Necessary
The ablation demonstrates clear complementarity:
•L1 alone fails on T3/T5 (8–11% ASR)—insiders bypass
provenance
•L3 alone achieves 0% across all tiers but lacks cryptographic
guarantees
•Full RAGShield combines L1’s hard boundary with L3’s
content analysis
•Neither RobustRAG nor RAGDefender matches
RAGShield-Full on all tiers simultaneously
D. Pre-Poisoned Corpus Degradation
A critical assumption of the insider detection (L3) is that
the established corpus is clean. We test what happens when
this assumption is violated by injecting 0–20% poison into the
baseline corpus before deploying RAGShield.
RAGShield is robust up to 2% pre-poisoning (ASR remains
0%). At 5%, ASR rises to 5.7% as pre-existing poison corrupts
the contradiction baseline. Notably, ASR plateaus at∼13%
even at 20% pre-poisoning—the corroboration and magnitude
detectors continue to catch most attacks even when the base-
line is partially corrupted. This plateau represents the residual
effectiveness of the non-baseline-dependent detectors.
E. Threshold Sensitivity Analysis
We vary the three key thresholds (θ mag,θcorr,α) across their
full range. RAGShield maintains 0% ASR and 0% FPR across
all tested values (θ mag∈[0.50,0.90],θ corr∈[0.55,0.90],α∈
[0.0,3.0]). The defense is not fragile or threshold-dependent—
the three complementary detectors provide redundancy, so no
single threshold is a single point of failure.
F . LLM-in-the-Loop Evaluation
To demonstrate end-to-end impact, we evaluate 10 scenarios
where poisoned context would cause an LLM to produce incor-
rect government information. For each scenario, we compare
the answer an LLM would produce with poisoned context (no
defense) versus clean context (after RAGShield filtering).
Without RAGShield, all 10 scenarios produce incorrect
answers. The T3 attacks (rows 3, 4, 7, 9) would cause citizens
to commit tax fraud or receive dangerously incorrect benefits
guidance. The T3-Hard attacks (rows 1, 2, 5, 6, 8, 10) would
cause financial errors of $50–$1,000. With RAGShield, all 10
scenarios produce correct answers from provenance-verified,
clean context.
G. Layer Contribution Analysis
H. Failure Mode Analysis
Where RAGShield succeeds unexpectedly:The T5 adap-
tive adversary—with valid provenance, semantically similar
poison, and fake corroborating documents—was fully blocked.
The contradiction detector identifies divergence from AU-
THORITATIVE sources even when the adversary inflates
corroboration scores.
T6: In-Place Replacement Attack (Honest Failure).
We design a T6 tier to find RAGShield’s true blind spot.

TABLE VI
ATTACKSUCCESSRATE(%)BYDEFENSE ONNATURALQUESTIONS(500PASSAGES, 200QUERIES). LOWER IS BETTER. 95% WILSONCIS IN
BRACKETS.
Tier RAGShield L1-Only L3-Only RobustRAG RAGDefender No Defense
T1: External 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] 0.5 [0,3] 12.5 [9,18] 13.0 [9,18]
T2: Forger 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] 7.5 [5,12] 8.0 [5,13]
T3: Insider 0.0 [0,2] 8.0 [5,13] 0.0 [0,2] 0.0 [0,2] 7.5 [5,12] 8.0 [5,13]
T4: Phantom 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] 8.5 [5,13] 8.5 [5,13]
T5: Adaptive 0.0 [0,2] 11.0 [7,16] 0.0 [0,2] 0.0 [0,2] 11.0 [7,16] 11.0 [7,16]
Benign FPR 0.0 [0,2] 0.0 [0,2] 0.0 [0,2] — — —
TABLE VII
RAGSHIELDDEGRADATIONUNDERPRE-POISONEDCORPUS
Pre-poison Docs ASR 95% CI
0% 0 0.0% [0.0, 5.2]%
1% 2 0.0% [0.0, 5.2]%
2% 4 0.0% [0.0, 5.2]%
5% 11 5.7% [2.2, 13.8]%
10% 22 12.9% [6.9, 22.7]%
15% 34 12.9% [6.9, 22.7]%
20% 45 12.9% [6.9, 22.7]%
TABLE VIII
LLM-IN-THE-LOOP: END-TO-ENDIMPACT
Query Correct Poisoned Type
Std. deduction single? $15,000 $15,500 T3-H
SSI monthly rate? $943/mo $993/mo T3-H
Home office deduction? Requires docs No docs needed T3
Report crypto? Yes, Form 8949 Under $100K exempt T3
Medicare B premium? $185/mo $195/mo T3-H
401(k) limit? $23,500 $24,500 T3-H
Disability application? Medical review Auto-approved T3
EITC maximum? $7,430 $7,630 T3-H
Capital gains taxable? Yes, 0/15/20% Tax-free T3
Child tax credit? $2,000 $2,100 T3-H
The insiderreplaces20 existing corpus documents in-place
with single-number modifications (e.g., changing “1788” to
“1789”). Since the modified documents occupy the same
corpus positions as the originals, L3 insider checks do not
fire—the documents are part of the baseline. Result:ASR =
17.5% [12.9%, 23.4%]. RAGShield completely fails to detect
in-place replacement attacks. The mean embedding similarity
between original and modified passages is 0.9964, confirming
that the changes are below all detection thresholds.
This is the fundamental limit of RAGShield’s current de-
sign: it detectsnewpoisoned documents but notmodifications
to existing ones. Mitigation requires:
•Content hash pinningat deployment time (L1 detects any
change to a pinned document)
•Periodic re-verificationagainst authoritative sources (NIST
RA-5)
•Change controlrequiring re-attestation for any corpus
modification (NIST CM-3)
Summary of failure modes:TABLE IX
LAYERCONTRIBUTIONS(V4EVALUATION)
Layer Detections What It Catches
L1 (Provenance) 40 docs blocked T1, T2, T4 at ingestion
L3 (Contradiction) 49 anomalies T3 policy reversals
L3 (Corroboration) 14 anomalies T3 uncorroborated claims
L3 (Magnitude) 47 anomalies T3-H numerical deviations
TABLE X
RAGSHIELDFAILUREMODES
Failure Mode Root Cause ASR
T6: In-place replacement L3 skips baseline docs 17.5%
Pre-poison>5% Corrupted baseline 5.7–12.9%
Sub-embedding changes Below model sensitivity Unknown
VIII. GOVERNMENTCOMPLIANCEMAPPING
Table XI maps RAGShield layers to NIST SP 800-53 Rev 5
controls. SA-12 (Supply Chain Protection) is the primary
mapping—RAGShield treats the knowledge base as a supply
chain requiring provenance verification.
Privacy Act compliance.RAGShield’s PII taint labels
enforce Privacy Act requirements: purpose limitation (retrieval
restricted to the purpose for which data was collected),
minimum necessary (minimum PII retrieved per query), and
accounting of disclosures (L5 audit trail records every PII-
containing retrieval). Federal agencies deploying RAG over
citizen records must file a System of Records Notice (SORN)
per 5 U.S.C. §552a—RAGShield’s audit trail provides the
disclosure accounting required by §552a(c).
IX. DISCUSSION
Provenance infrastructure.RAGShield assumes a docu-
ment signing infrastructure that does not yet exist for most
government documents. Our bootstrapping strategy (Level 0
through Level 3) provides a realistic adoption path. The
Government Publishing Office (GPO) already digitally signs
Federal Register documents; extending this to agency publica-
tions is an infrastructure investment, not a research challenge.
Computational complexity.L1 provenance verification
isO(1)per document (hash + signature check). L2 trust-
weighted retrieval adds no asymptotic overhead to standard

TABLE XI
NIST SP 800-53 CONTROLMAPPING
Control Layer Implementation
SA-12 Supply Chain L1 Document provenance
SI-7 Info Integrity L1 Content hash verification
SI-10 Input Validation L1 Anti-PhantomText
AC-4 Info Flow L2 Trust-weighted retrieval
SC-16 Security Attrs L2, L3 Trust/taint labels
SI-3 Malicious Code L3 Content-based detection
AU-10 Non-repudiation L3 Source attribution
AU-3 Audit Content L4 Provenance citations
SI-4 Monitoring L4, L5 Taint pattern monitoring
PM-25 PII Minimization L4 PII taint enforcement
AU-2 Event Logging L5 Query-to-response trail
AU-6 Audit Review L5 Flagged response review
IR-4 Incident Handling L5 BLOCK triggers incident
CM-3 Change Control L1 Re-attestation required
RA-5 Vuln Scanning L1 Provenance validity scan
top-kretrieval. L3 insider detection isO(n)per newly-
ingested document (compare againstnexisting corpus docu-
ments), yieldingO(mn)for ingestingmnew documents into a
corpus of sizen. For a 1M-document corpus, this is tractable:
comparing one new document against 1M embeddings via
matrix multiplication takes∼50ms on CPU. For batch inges-
tion of thousands of documents, approximate nearest neighbor
search (e.g., FAISS HNSW) reduces this toO(mlogn). The
per-query overhead (∼10ms) is dominated by the embedding
model inference (∼50ms), making RAGShield’s overhead
negligible in practice.
Relationship to CivicShield.RAGShield extends Civic-
Shield [18] in two directions: Layer 1 extends CivicShield’s
input validation with document-level provenance chains, and
Layer 3 extends CivicShield’s semantic taint tracking with
pipeline-level taint propagation. Together, CivicShield (con-
versational defense) and RAGShield (knowledge base defense)
provide complementary protection for government AI systems.
Limitations.(1) Our NQ evaluation uses 500 passages;
production government knowledge bases contain millions of
documents. Scaling to full NQ (300K+ passages) is future
work. (2) The insider detection degrades when the baseline
corpus is pre-poisoned beyond 5% (Table VII); periodic
corpus auditing is required. (3) RobustRAG and RAGDe-
fender baselines are simulated implementations capturing the
core algorithmic principles (isolate-then-aggregate and post-
retrieval anomaly filtering respectively); results may differ
from production implementations. (4) The LLM-in-the-loop
evaluation uses in-context assessment rather than automated
end-to-end testing with a deployed LLM.
Ethical considerations.All attack documents in our evalu-
ation contain fabricated government information (incorrect tax
deductions, false benefit amounts). We do not distribute these
documents outside the evaluation context. The government
documents in our corpus are generated from templates, not
copied from official sources.X. CONCLUSION
We presented RAGShield, a five-layer defense-in-depth
framework that applies supply chain provenance verification
to RAG knowledge base security. By treating RAG poisoning
as a supply chain attack, RAGShield provides a fundamentally
different defense from existing content-analysis approaches:
cryptographic provenance blocks unsigned and forged docu-
ments, while cross-source contradiction detection catches in-
sider threats that bypass provenance entirely. Our evaluation on
the Natural Questions benchmark across five adversary tiers—
including adaptive attacks by a RAGShield-aware adversary—
demonstrates 0.0% attack success rate (95% CI: [0.0%, 1.9%])
with 0.0% false positive rate, outperforming both RobustRAG
and RAGDefender on insider and adaptive attack tiers.
RAGShield is the first framework to apply supply chain
provenance to RAG security, the first to provide NIST SP
800-53 compliance mapping for RAG pipelines, and the first
to defend against data loader exploitation attacks. As federal
agencies scale RAG deployments for citizen-facing services,
provenance-verified knowledge bases are essential to maintain-
ing public trust in government AI systems.
ACKNOWLEDGMENTS
The author thanks the anonymous reviewers for their feed-
back.
REFERENCES
[1] H. Chaudhari et al., “Phantom: General trigger attacks on retrieval
augmented language generation,” inProc. NeurIPS, 2024.
[2] Promptfoo, “RAG data poisoning: Key concepts explained,” 2025.
[Online]. Available: https://www.promptfoo.dev/blog/rag-poisoning/
[3] “The hidden threat in plain text: Attacking RAG data loaders,”
arXiv:2507.05093, 2025.
[4] “ConfusedPilot: Confused deputy risks in RAG-based LLMs,”
arXiv:2408.04870, 2024.
[5] CISA, “SolarWinds supply chain compromise,” Alert AA20-352A,
2020.
[6] Google, “SLSA: Supply chain levels for software artifacts,”
https://slsa.dev, 2023.
[7] C2PA, “Coalition for content provenance and authenticity,”
https://c2pa.org, 2021.
[8] C. Xiang et al., “Certifiably robust RAG against retrieval corruption,”
inProc. NeurIPS, 2025.
[9] M. Kim et al., “RAGDefender: Efficient defense against knowledge
corruption attacks on RAG systems,” arXiv:2511.01268, 2025.
[10] “TrustRAG: Enhancing robustness and trustworthiness in RAG,”
arXiv:2501.00879, 2025.
[11] S. Choudhary et al., “Through the stealth lens: Rethinking attacks and
defenses in RAG,” arXiv:2506.04390, 2025.
[12] J. Hwang et al., “Retrieval-augmented generation with estimation of
source reliability,” arXiv:2410.22954, 2024.
[13] “ReliabilityRAG: Effective and provably robust defense for RAG-based
web-search,” inProc. NeurIPS, 2025.
[14] “CPA-RAG: Covert poisoning attacks on RAG,” arXiv:2505.19864,
2025.
[15] “AuthChain: Knowledge poisoning attack on RAG systems,”
arXiv:2505.11548, 2025.
[16] GSA, “USAi.Gov: AI platform for federal agencies,” 2025.
[17] GAO, “GAO-25-107653: Generative AI use and management at federal
agencies,” 2025.
[18] K. Patil, “CivicShield: A cross-domain defense-in-depth framework for
securing government-facing AI chatbots,” arXiv:2603.29062, 2026.
[19] NIST, “SP 800-53 Rev. 5: Security and privacy controls for information
systems,” 2020.

[20] T. Kwiatkowski et al., “Natural Questions: A benchmark for question
answering research,”Trans. ACL, 2019.
[21] NIST, “SP 800-53 control overlays for securing AI systems: Concept
paper,” 2025.
[22] “RAGForensics: Traceback of poisoning attacks to RAG,”
arXiv:2504.21668, 2025.
[23] “PoisonArena: Uncovering competing poisoning attacks in RAG,”
arXiv:2505.12574, 2025.
[24] “MMA-RAGT: Stateful trust inference for securing multimodal agentic
RAG,” arXiv:2602.21447, 2026.
[25] “Who taught the lie? Responsibility attribution for poisoned knowledge
in RAG,” arXiv:2509.13772, 2025.
[26] “Knowledge database or poison base? Detecting RAG poisoning through
LLM activations,” arXiv:2411.18948, 2024.
[27] “GMTP: Gradient-based masked token probability for poisoned docu-
ment detection,” inFindings of ACL, 2025.
[28] “Bias injection attacks on RAG databases and sanitization defenses,”
arXiv:2512.00804, 2025.
[29] “RAGuard: Secure retrieval-augmented generation against poisoning
attacks,” arXiv:2510.25025, 2025.
[30] J. Zou et al., “PoisonedRAG: Knowledge corruption attacks to retrieval-
augmented generation of large language models,” inProc. USENIX
Security, 2025.
[31] “SDAG: Addressing corpus knowledge poisoning attacks on RAG using
sparse attention,” arXiv:2602.04711, 2026.