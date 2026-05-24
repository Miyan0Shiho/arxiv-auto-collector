# Auditing Privacy in Multi-Tenant RAG under Account Collusion

**Authors**: Florian A. D. Burnat, Brittany I. Davidson

**Published**: 2026-05-19 13:41:59

**PDF URL**: [https://arxiv.org/pdf/2605.19847v1](https://arxiv.org/pdf/2605.19847v1)

## Abstract
Multi-tenant retrieval-augmented generation (RAG) services advertise per-account differential privacy as the operative leakage boundary: each account's queries are guaranteed to satisfy $(\varepsilon_{\text{acc}}, δ_{\text{acc}})$-DP with respect to the index. We identify same-index multi-account collusion as a privacy-boundary failure: for $k$ same-tenant accounts coordinating against the tenant's index -- the operative regime -- known DP composition theory implies joint leakage degrades unconditionally at rate $Θ(\sqrt{k} \cdot \varepsilon_{\text{acc}})$ for Gaussian-noised retrieval. Cross-tenant and external collusion match the rate only under explicit access-control failure (M4); without M4 these regimes have zero leakage by design and reduce to an architectural audit, not a DP audit. We exhibit an attack realizing the rate and derive a RAG-specific MIA prediction we test empirically. To make this per-account/joint gap auditable, we design the first audit protocol that operates against unmodified RAG deployments and issues a quantitative $(\textsf{PASS}, \varepsilon_{\text{audit}})$ verdict for the retrieval-score channel -- the noise-then-select step the per-account DP guarantee actually covers -- without index disclosure, pipeline redesign, or model-weight exposure. Generation-channel privacy (LLM output conditioned on selected documents) is a separate audit predicate that should compose with ours; we explicitly scope it out. The protocol composes generic cryptographic primitives (Merkle ledgers, ZK function-application proofs, Gaussian noise attestations) with six RAG-specific primitives (embedder commitment, index-content vector commitment, per-account query ledger, noise-then-select attestation, cross-tenant containment proof, coalition-size estimator) and supports both closed-form audit bounds and Rényi-DP moments-accountant tracking.

## Full Text


<!-- PDF content starts -->

Auditing Privacy in Multi-T enant RAG under
Account Collusion
Florian Burnat∗Brittany Davidson†
May 20, 2026
Abstract
Multi-tenant retrieval-augmented generation (RAG) services advertise per-account
differential privacy as the operative leakage boundary: each account’s queries
are guaranteed to satisfy (εacc, δ acc)-DP with respect to the index. W e identify
same-index multi-account collusion as a privacy-boundary failure: for k same-
tenant accounts coordinating against the tenant’s index — the operative regime
— known DP composition theory implies joint leakage degrades unconditionally at
rateΘ(√
k·εacc)for Gaussian-noised retrieval. Cross-tenant and external collusion
match the rate only under explicit access-control failure (M4); without M4 these
regimes have zero leakage by design and reduce to an architectural audit, not a
DP audit. W e exhibit an attack realizing the rate and derive a RAG-specific MIA
prediction we test empirically . T o make this per-account/joint gap auditable, we
design the first audit protocol that operates against unmodified RAG deployments
and issues a quantitative (P ASS , ε audit)verdict for the retrieval-score channel —
the noise-then-select step the per-account DP guarantee actually covers — without
index disclosure, pipeline redesign, or model-weight exposure. Generation-channel
privacy (LLM output conditioned on selected documents) is a separate audit pred-
icate that should compose with ours; we explicitly scope it out. The protocol com-
poses generic cryptographic primitives (Merkle ledgers, ZK function-application
proofs, Gaussian noise attestations) with six RAG-specific primitives (embedder
commitment, index-content vector commitment, per-account query ledger, noise-
then-select attestation, cross-tenant containment proof, coalition-size estimator)
and supports both closed-form audit bounds and Rényi-DP moments-accountant
tracking. W e validate the predicted√
kscaling against two falsification gates plus
a diagnostic on a synthetic multi-tenant deployment, empirically observe the√
k
trend surviving top- K post-processing and a trained embedder at the parameters
∗University of Bath, Bath, UK. fadb20@bath.ac.uk .
†University of Bath, Bath, UK. bid23@bath.ac.uk .
1arXiv:2605.19847v1  [cs.CR]  19 May 2026

tested, confirm alternative-adversary robustness, and discuss the implications for
production RAG services covered by EU AI Act and DSA disclosure obligations.
Keywords: differential privacy , retrieval-augmented generation, multi-tenant, ac-
count collusion, audit protocol, zero-knowledge proofs, membership inference

1 Introduction
Production retrieval-augmented generation services — Microsoft 365 Copilot, OpenAI
Assistants, Anthropic Files API, and the managed-RAG products built atop Pinecone,
W eaviate, and similar vector databases — are overwhelmingly multi-tenant. A single
deployed embedder, a single deployed generator, and a fleet of per-tenant indices serve
thousands to millions of customer accounts under a shared infrastructure. The DP-RAG
designs proposed for and audited at this architecture reason at per-account granularity:
each account is rate-limited, each query is scored against only that account’s tenant index,
and retrieval scores are noised to deliver an (εacc, δ acc)-differential-privacy guarantee per
account over an audit window (Cheng et al., 2025 ; Zeng et al., 2024 ). Our concern is the
adequacy of the per-account framing where it does apply , not that every named service
formally advertises such a guarantee.
This per-account framing is computationally convenient and aligns with how providers
reason about API contracts. It is also the framing implicit in the recent wave of single-
attacker membership-inference work: (Li et al., 2024 ; W ang et al., 2025 ; F eng et al., 2025 ;
Liu et al., 2024 ; Naseh et al., 2025 ; Gao et al., 2025 ) all treat one isolated attacker as the
unit of analysis, competing on AUC at fixed query budget under a k= 1 threat model.
Existing defenses (Cheng et al., 2025 ) likewise modify the deployment to leak less per
query; none formalize what changes when the assumption of a single attacker is relaxed.
The gap. Production RAG accounts are cheap. An adversary with a budget for k
disposable accounts — whether a malicious customer creating sock puppets, a malicious
bystander tenant probing across the boundary , or an external attacker buying API access
through resellers — can pool kaccounts’ query budgets against a single target tenant.
The provider’s per-account DP guarantee no longer says anything direct about what the
coalition observes jointly . W orse, the right benchmark is non-obvious: a basic group-
privacy bound gives εjoint≤k·εacc, but this is loose; a naive “per-account independence”
reading gives εjoint≈εacc, but this is wrong.
1

Contributions. W e characterize the gap with a tight rate, an empirical attack, and an
audit protocol that makes the bound verifiable in deployment.
C1. Threat model: same-index multi-account col lusion as a privacy-boundary failure (Sec-
tions 2and 3). W e define (ε, δ, k )-collusion DP for multi-tenant RAG and identify
same-index account pooling as the regime in which per-account (εacc, δ acc)-DP guaran-
tees structurally understate joint leakage. Instantiating known DP-composition the-
ory (Dwork, Rothblum, et al., 2010 ; Kairouz et al., 2017 ; V adhan and W ang, 2021 )
for this setting gives the degradation rate Θ(√
k·εacc)for Gaussian-noised retrieval
(Theorems 3.4 and 3.5 and corollary 3.6 ), unconditional for same-tenant collusion and
under M4 (Remark 2.1 ) otherwise. The composition rate is a specialization of stan-
dard theory; the contribution is the threat model itself and the bridge to RAG-specific
MIA.
C2. F alsifiable empirical predictions (Sections 3and 4). The DP-to-MIA reduction (Corol-
lary 3.12 ) translates the bound into a predicted membership-inference AUC of1
2+
Θ(√
k·εacc)at fixed per-account budget (the coalition-wide query count grows as kn ;
the per-account budget εacc is held constant). W e commit to two falsification gates
(P1, P3 in Section 3.8 : slope and tightness) plus one diagnostic (P2: scale collapse,
resolved at higher trial counts); a gate failure would invalidate the audit-protocol
guarantees that follow. W e test these on a multi-tenant F AISS-based harness and
report the empirical curve.
C3. V erifier-runnable audit protocol for unmodified deployments (Section 5). A four-phase
protocol (commitment, per-query attestation, coalition-size estimation, verification)
issuing a (P ASS , ε audit)verdict for the retrieval-score channel (noise-then-select) with-
out disclosing the index. Generation-channel privacy is out of scope. The proto-
col composes five generic cryptographic primitives we factor into a reusable library
(cryptographic-audit-protocols ) with six RAG-specific primitives. Theorem 5.1
shows the verdict implies (εaudit, δ policy)-collusion-DP for any coalition up to the policy
cap. W e also identify and correct a subtle δ-accounting error in naive multi-account
2

composition (the well-formedness constraint δpolicy > k maxδacc), making the final joint
failure probability exactly the policy target.
C4. Numerical privacy tracking via RDP (Sections 3.7 and 4). The RDP-based analysis
(Theorem 3.9 ) admits exact numerical evaluation via the moments accountant, giving
tighter realized (ε, δ)pairs than the closed-form bound at moderate k— the closed-
form for advertised guarantees, the accountant for in-deployment audit reporting.
Why now. Three trends make the gap operationally pressing. First, RAG-specific
membership-inference attacks have matured from one-off case studies to a competitive
sub-field over 2024–2025; the single-attacker performance ceiling is now suﬀicient for re-
viewers to demand a stronger threat model. Second, regulatory regimes that require
auditable disclosure of AI system behavior — the EU AI Act (Articles 13, 50, 86), the
Digital Services Act, and emerging US sectoral frameworks — are explicitly contemplat-
ing third-party audit primitives that do not require model or data disclosure; our protocol
fits this slot directly . Third, multi-account abuse is already an operational reality at pro-
duction RAG providers (rate-limit evasion, sock-puppet account creation), but is treated
as fraud rather than a privacy boundary failure; we argue it is both.
Distinction from prior threat models. Our framing differs from three adjacent
literatures along orthogonal axes. Single-attacker RAG MIA (Naseh et al., 2025 ; Gao
et al., 2025 ; Li et al., 2024 ; W ang et al., 2025 ; Liu et al., 2024 ) relaxes neither k= 1
nor the assumption of a stateless API; we relax the former while keeping the latter.
F ederated learning col lusion (Lyu et al., 2023 ; Pasquini et al., 2022 ) models coordinated
adversaries but at training time, against gradient aggregation; we model coordination at
inference time, against retrieval over a stateless API. DP-RAG defenses (Cheng et al.,
2025 ) modify deployments to leak less per query; we leave deployment intact and provide
an audit primitive over its existing privacy claims.
Roadmap. Section 2formalizes the multi-tenant RAG threat model and the (ε, δ, k )-
collusion-DP notion. Section 3proves the√
ktight bound, the matching lower bound,
3

and the empirical predictions. Section 4reports the empirical AUC–vs– kcurve against
the predictions. Section 5specifies the audit protocol and its soundness. W e discuss
responsible disclosure, limitations, and policy implications in the closing section.
2 Threat Model
W e formalize multi-tenant retrieval-augmented generation (RAG) under coordinated ac-
count collusion as a Stackelberg-like game between a service provider P and an adversary
A controlling kaccounts. The provider commits to a privacy policy — per-account rate
limits and a differentially-private retrieval scoring mechanism — and the adversary best-
responds with a coordinated query strategy . Our analysis (Section 3) shows the provider’s
per-account composition bound understates total leakage, and the audit protocol (Sec-
tion 5) closes the resulting gap.
2.1 System model
A multi-tenant RAG service is a tuple P= (D, Emb,LLM,Policy )where:
•D={Dt}t∈Tis a family of per-tenant indices, Dt⊆ X , partitioned by tenant identifier
t∈T.
• Emb :X →Rdis a shared embedder.
• LLM :Q×XK→ Y is a shared generator that conditions on a query qand the top- K
retrieved documents.
• Policy = (r, ε acc, δ acc, σ)specifies the per-account rate limit r(queries per audit win-
dow), per-account privacy budget (εacc, δ acc), and noise parameter σon retrieval scores.
Figure 1sketches the data flow. Accounts (rounded squares) live under tenants
(dashed boundary boxes) and issue queries through a shared service; the service em-
beds each query , scores it against the requesting tenant’s index, perturbs the scores with
Gaussian noise before top-K selection, and feeds the selected documents to the LLM. A
4

tenant t1
tenant t2RAG service
Emb→ Score →+N(0, σ2)→ TopKLLMoutput
channeltop-Kcoalition kper-account rate limit r
Figure 1: Multi-tenant RAG with a k-account coalition. The service applies the rate limit
per account and adds DP noise before top- K; coalitions evade both bounds by pooling
outputs after retrieval (red dashed link).
k-coalition (red dashed link) is a subset of accounts that pool their responses through
a shared output channel after the LLM has returned them, which the service cannot
observe and the per-account rate limiter does not constrain.
F or a query qfrom account awith tenant t(a), the service runs the retrieval mechanism
Retr(q, a) = TopK 
Score (q, D t(a)) +N(0, σ2I)
, (2.1)
adding Gaussian noise before top-K selection,1and then returns RAG(q, a) = LLM(q,Retr(q, a)).
HereN(0, σ2I)denotes i.i.d. Gaussian noise on each coordinate of the |Dt(a)|-dimensional
score vector; the scalar lower-bound mechanism M∗of Theorem 3.5 is the projection onto
the single coordinate corresponding to the differing document. The per-account (εacc, δ acc)
guarantee and Theorem 3.4 govern Retr ; generation is out of scope (Section 7.1 ).
2.2 Provider strategy
The provider commits, before the audit window opens, to a policy Policy together with
public commitments
C emb=H(Emb∥prompt _tmpl∥tokenizer ∥schema (D)), (2.2)
C ledger ,0=H(empty ) (2.3)
1T op- K on noisy scores is a post-processing step over the DP output, so the privacy loss is non-
increasing. Selecting first and noising only the K winners breaks this — the unnoised ordering leaks
through the selected indices. The audit protocol attests the correct ordering.
5

where C ledger is the root of an append-only Merkle ledger of per-account query records
(Section 5). The provider’s claimed privacy guarantee is that, for any single account a
over the audit window, the joint distribution of a’s retrieval-channel transcripts ( Retr
outputs of ( 2.1 )) satisfies (εacc, δ acc)-DP w.r.t. neighboring index pairs Dt(a)∼D′
t(a)
differing by the addition, removal, or replacement of exactly one document (the stan-
dard add/remove/replace relation, |Dt(a)△D′
t(a)| ≤2); the per-account L2 sensitivity of
Score (q,·)over this relation is bounded by 1for unit-norm embeddings. Generation-
channel privacy is out of scope (Section 7.1 ).
2.3 Adversary model
The adversary A controls k≥1accounts {a1, . . . , a k}, each registered to a tenant t(ai)
chosen by A. W e consider three regimes:
Same-tenant collusion ( t(ai) =t∗∀i)
Acoordinates within a single tenant to extract that tenant’s index. Models a malicious
customer.
Cross-tenant collusion ( t(ai)varies)
A coordinates across tenants to extract another tenant’s documents via shared infras-
tructure. Models a malicious bystander tenant.
External collusion ( t(ai) /∈Tvictim )
A has no legitimate access to the victim tenant’s index but uses kunrelated accounts.
Models an external attacker.
The adversary’s capabilities are:
(C1) Shared posterior : all kaccounts share a single probabilistic belief over the target
index contents.
(C2) Adaptive coordinated queries : the query of account aiat time tmay depend on the
full transcript of all accounts up to time t−1.
6

(C3) Rate-limit-aware scheduling :A knows rand schedules queries to remain within
per-account limits while saturating aggregate budget.
(C4) Shared output channel : all responses across accounts are pooled before A’s estimator
runs.
Modeling assumptions. The analysis of Section 3treats the following as standing
assumptions on the deployment: ( M1 )Independent fresh randomness : each query con-
sumes an independent draw of Gaussian noise; in particular, no two accounts (colluding
or otherwise) share noise samples. ( M2 )Audit-window disjointness : each audit window
W has its own per-account budget; the per-account (εacc, δ acc)resets at window open, and
adversary coordination is scoped to a single window unless explicitly noted. ( M3 )Em-
bedder determinism within a window : the embedder Emb and its parameters are fixed for
the duration of W and identical across tenants. ( M5 )Honest-account receipt sample : at
least one non-colluding account submits its query-time receipts to V; ledger completeness
(Section 5.4 ) is verified against this sample. These are standard for the DP-RAG setting
and are enforced cryptographically by the audit protocol of Section 5viaC emb, C policy ,
and R2 receipts.
Remark 2.1 (M4: same-index access for cross-tenant/external regimes) .Equation ( 2.1 )
confines an account’s queries to Dt(a), so under normal access control cross-tenant and
external accounts have zero leakage about the victim index Dt∗. W e retain those regimes
only under an explicit access-control-failure assumption — ( M4 )Same-index access : the
attacker’s accounts share retrieval access to Dt∗via a shared backend (e.g., a misconfig-
ured router or an over-broad shared index). Under M4, all three regimes are formally
identical for Theorem 3.4 ; the formal claims and audit protocol apply unconditionally
only to same-tenant collusion. The § 4.7 experiment is conducted under M4 and reads
as an upper bound under access-control failure, not a guarantee about ordinary external
attackers.
7

2.4 Adversary goals
W e consider three goals, in increasing severity:
(G1) Membership inference (MIA) : given a candidate document x∗, decide whether x∗∈
Dtfor a target tenant t. Measured as advantage over chance, area under ROC, or
queries-to-confidence at fixed false-positive rate.
(G2) Document reconstruction : output a string ˆxsuch that ˆxis suﬀiciently close (e.g.,
ROUGE-L ≥τ) to some x∈Dt. Measured as reconstruction rate at fixed query
budget.
(G3) Cross-tenant leakage : extract any document of tenant t∗̸=t(ai)∀i. Measured as
reconstruction rate across the tenant boundary .
2.5 Game formulation
The interaction unfolds in three phases over an audit window W :
Phase 1 (commitment)
P publishes Policy ,C emb , and C ledger ,0.
Phase 2 (query)
A issues queries from kaccounts subject to Policy . Each query updates C ledger via
append-only Merkle insertion.
Phase 3 (audit)
Vverifies, against P’s ZK-attested transcript, that (i) every retrieval respected C emb ,
(ii) noise was applied with the committed σbefore top- K, (iii) the ledger root is
consistent with the accounts’ query counts, and (iv) ˆk≤kmax for the policy-declared
cap; the joint leakage bound εaudit is then computed from kmax (not the empirical ˆk),
keeping soundness independent of estimator robustness (Section 5.6 ).
Definition 2.2 ((ε, δ, k )-collusion DP) .A multi-tenant RAG mechanism RAG is(ε, δ, k )-
collusion-DP with respect to tenant t∗if, for any kadaptively-coordinated accounts
8

{a1, . . . , a k}satisfying (C1) –(C4) , any neighboring index pair D∼t∗D′differing in one
document of tenant t∗, and any output set S,
Pr[TranscriptA(D)∈S]≤eε·Pr[TranscriptA(D′)∈S] +δ. (2.4)
Definition 2.2 sits between standard per-account DP (which is the k= 1 special case)
and the operationally-meaningless ε-pure DP against an unbounded coalition. Our central
result (Section 3) bounds the achievable εunder coordination capabilities (C1) –(C4) and
shows it grows as Θ(√
k)rather than the Θ(k)bound a naive per-account composition
argument would suggest.
2.6 Distinction from prior threat models
Our model differs from the literature on four axes:
• vs. single-attacker RAG MIA (e.g., Naseh et al. 2025; Gao et al. 2025): we relax
thek= 1 assumption, exposing a sublinear-in- kattack surface that single-attacker
analyses cannot detect.
• vs. FL col lusion (e.g., Lyu et al. 2023; Pasquini et al. 2022): prior collusion analyses
target training-time aggregation. W e attack inference-time retrieval over a stateless
API.
• vs. DP-RAG defenses (e.g., Cheng et al. 2025, RemoteRAG): prior defenses modify
the deployment to leak less per query; we provide an audit primitive that any verifier
can run against an unmodified deployment to obtain a leakage bound, without index
disclosure.
• vs. embedding-inversion attacks (e.g., Morris et al. (Morris et al., 2023 )): inversion
attacks recover input text from embedding vectors and are an orthogonal threat to
membership inference. W e model membership of documents given retrieval responses;
inversion-augmented attacks compose with our bound (a stronger adversary against
the embedder still pays the√
kDP cost on the retrieval mechanism).
9

3 DP Composition Under Collusion
This section specializes standard DP composition / concurrent-composition theory (Dwork,
Rothblum, et al., 2010 ; Kairouz et al., 2017 ; V adhan and W ang, 2021 ) to multi-tenant
RAG: a service whose per-account (εacc, δ acc)-DP guarantee comes from per-query Gaus-
sian noise (Definition A.1 ) leaks Θ(√
k·εacc)under k-account collusion against the same
index. The bound is sublinear in krelative to the trivial group-privacy bound k·εacc,
but strictly larger than the per-account budget the provider advertises — the gap is the
privacy debt the provider implicitly takes on by failing to bound coalition size.
3.1 Preliminaries
Definition 3.1 (Differential privacy (Dwork and Roth, 2014 )).A randomized mechanism
M:X∗→ Y is(ε, δ)-DP if for all neighboring datasets D∼D′and all measurable S⊆ Y ,
Pr[M(D)∈S]≤eεPr[M(D′)∈S] +δ.
W e use the advanced composition theorem in the form of (Dwork, Rothblum, et al.,
2010 ):
Lemma 3.2 (Advanced composition) .F or any ε0∈(0,1] andδ0, δ′∈(0,1), the m-fold
adaptive composition of (ε0, δ0)-DP mechanisms on the same database is (ε′, mδ 0+δ′)-DP
forε′=p
2mlog(1/δ′)·ε0+mε 0(eε0−1).F orε0smal l, the second term is O(mε2
0)and
dominated by the first; we abbreviate ε′=O p
mlog(1/δ′)·ε0
.
3.2 Per-account guarantee and per-query calibration
The service implements DP retrieval scoring via Gaussian noise on similarity scores (Sec-
tion 2). F or a single account aissuing nadaptive queries against Dt(a), the per-account
guarantee (εacc, δ acc)is achieved by calibrating the per-query noise so that each query is
(εq, δq)-DP with
εq=εaccp
2nlog(1/δacc), δ q=δacc/n, (3.1)
10

so that Lemma 3.2 applied to a’snqueries yields the advertised (εacc, δ acc)-DP transcript
guarantee.2
W e call services satisfying ( 3.1 ) via nindependent per-query Gaussian mechanisms
the Gaussian-noised score-release class (Definition A.1 ). Theorem 3.4 is stated for this
class only; opaque transcript bounds without per-query accounting yield only the loose
k·εacc group-privacy bound. Class membership is verified cryptographically by the audit
protocol of Section 5before εaudit is issued.
3.3 Coordination reduces to a single adaptive analyst
LetA= (a1, . . . , a k)be a k-collusion adversary satisfying capabilities (C1) –(C4) from
Section 2. The transcript τApools all responses from all accounts into a single sequence
visible to a unified estimator.
Lemma 3.3 (Coordination reduction) .F or any k-col lusion adversary A against Dt∗,
there exists a single adaptive analyst A†with query budget kn against Dt∗such that
τAd=τA†.
Sketch – ful l proof in appendix. Capability (C4) givesA†access to the same response
sequence as A. Capability (C1) ensures both adversaries condition on the same posterior
overDt∗. Capability (C2) permits A†to schedule queries in any order consistent with the
per-account rate limit (capability (C3) ). The simulator interleaves the kaccounts’ query
schedules into a single kn -query sequence; rate-limit constraints translate into a per-step
admissibility constraint that does not change the joint distribution of transcripts. The
reduction applies unconditionally to same-tenant collusion; for cross-tenant and external
collusion, it requires assumption M4 (Remark 2.1 ) that each colluding account can cause
queries to be scored against Dt∗via a shared backend — without M4 the regimes have
zero leakage about Dt∗and the reduction is vacuous.
2The calibration absorbs per-query failure probabilities into the composition slack via δq=δacc/n
and δ′=δacc , giving a post-composition guarantee of (εacc,2δacc)in the strictest reading. An equal
δ-split ( δq=δacc/(2n),δ′=δacc/2) recovers the exact (εacc, δ acc)guarantee with a constant tightening
ofp
log(2/δacc)/ log(1/δacc)≈1.025 atδacc= 10−6— numerically negligible at the regime we test. The
privacy filter and odometer constructions of (Rogers et al., 2016 ) target the adaptive-budget setting and
do not improve fixed-budget constants; the leading-order tightening of the Gaussian mechanism itself
would come from the analytic calibration of (Balle and W ang, 2018 ) (see Section 3.5 ).
11

3.4 Main result
Theorem 3.4 ((ε, δ, k )-collusion DP upper bound for Gaussian-noised score release) .
Let RAG be in the Gaussian-noised score-release class (Definition A.1 ) with per-account
guarantee (εacc, δ acc)via calibration (3.1 ). F or any δ≥δacc,RAG is(εk, δ+kδ acc)-col lusion-
DP (Definition 2.2 ) with
εk=√
k·εacc·s
log(1/δ)
log(1/δacc)+Ok·ε2
acc
log(1/δacc)
. (3.2)
F orδ=δacc andεacc∈(0,1] this simplifies to εk= Θ(√
k·εacc).
Sketch – ful l proof in appendix. By Lemma 3.3 , it suﬀices to bound the privacy of kn
adaptive (εq, δq)-DP queries against Dt∗. Apply Lemma 3.2 with m=kn ,ε0=εq,
δ0=δq,δ′=δ. Substituting ( 3.1 ):
εk≤p
2kn log(1/δ)·εq+kn·εq(eεq−1)
=p
2kn log(1/δ)·εaccp
2nlog(1/δacc)+O(knε2
q)
=√
k·εacc·s
log(1/δ)
log(1/δacc)+Ok·ε2
acc
log(1/δacc)
.
The total δslack absorbs kn·δq=kδ acc from the per-query failure probabilities plus δ
from the composition slack.
3.5 Tightness
W e exhibit M∗in the Gaussian-noised score-release class saturating the per-account
guarantee together with a k-collusion adversary achieving MIA advantage Ω(√
k·εacc).
By the advantage-to-DP reduction (Dwork and Roth, 2014 ), any (ε, δ)characterizing
M∗hasε= Ω(√
k·εacc)up to logs in δacc, δq; rate matches Theorem 3.4 , constants differ
(Corollary 3.6 and remark 3.7 ).
12

Construction. Fix tenant t∗with index Dt∗⊆Rdof unit-norm embeddings. Let
neighboring indices D∼t∗D′differ in a single document: D=D0∪{x∗},D′=D0∪{x′
∗},
with∥x∗∥2=∥x′
∗∥2= 1 . Define a probe query q∗(unit-norm) such that the similarity
scores satisfy ⟨q∗, x∗⟩−⟨q∗, x′
∗⟩= ∆ for some ∆∈(0,1](achievable with ∆ = 1 by taking
x∗=q∗andx′
∗orthogonal to q∗, or smaller ∆ by interpolation toward x∗=x′
∗).
The mechanism M∗releases, per query q, the Gaussian-noised scalar
M∗(q;Dt∗) =⟨q, x q⟩+Z, Z ∼ N (0, σ2), (3.3)
where xqis the (deterministic) document at the position of query qinDt∗, and the noise
scale is
σ=∆p
2nlog(1/δacc)·2log(1.25/δq)
εacc, (3.4)
with δq=δacc/n.M∗releases the noisy similarity score directly — a strictly stronger
adversary surface than top- K-only (whose output is post-processing of the same noisy
score vector), so lower bounds against M∗characterize the score-release class itself, not
top-K.
Per-account privacy of M∗. By the standard Gaussian mechanism Dwork and Roth,
2014 , Theorem A.1, a single query of M∗is(εq, δq)-DP with εq= ∆p
2log(1.25/δq)/σ.
Substituting ( 3.4 ),εq=εacc/p
2nlog(1/δacc),which matches the per-query calibration
(3.1 ). Thus n adaptive queries from a single account compose to (εacc, δ acc)-DP via
Lemma 3.2 , satisfying the advertised per-account guarantee.
Adversary A∗. Each of the kaccounts issues the same probe query q∗for all nrounds,
yielding kn samples
yi,j=⟨q∗, x(b)
∗⟩+Zi,j, i ∈[k], j∈[n],
where x(b)
∗=x∗ifD is in use ( b= 0 ) and x(b)
∗=x′
∗ifD′(b= 1 ), and the Zi,j are
iidN(0, σ2). The adversary computes the empirical mean ¯y=1
knP
i,jyi,j and outputs
13

ˆb= 1[¯y <1
2(⟨q∗, x∗⟩+⟨q∗, x′
∗⟩)]. Capabilities (C1) –(C4) of Section 2suﬀice: posterior
is shared (both branches assume Gaussian likelihood with known variance), queries are
coordinated and rate-limit-aware (each account stays within nqueries), and the output
channel is pooled.
Distinguishing advantage. Under either index, ¯yis Gaussian with variance σ2/(kn)
and means separated by ∆. The advantage of the optimal test is
Adv(A∗) = 2 Φ 
∆√
kn
2σ!
−1,
where Φis the standard normal CDF. Substituting ( 3.4 ) and simplifying,
∆√
kn
2σ=√
k·εacc
4p
log(1/δacc)log(1.25/δq).
F orεacc∈(0,1] andδacc, δqin any constant-polynomial regime, Φis approximately linear
at the origin ( Φ(z)−1/2≥z/√
2πforz∈[0,1]), so
Adv(A∗) = Ω √
k·εaccp
log(1/δacc)!
. (3.5)
Theorem 3.5 (Lower bound) .F or any k≥1and per-account budget (εacc, δ acc)with
εacc∈(0,1], the mechanism M∗in (3.3 )–(3.4 )satisfies the per-account (εacc, δ acc)-
DP guarantee, yet admits a k-col lusion adversary A∗with indistinguishability advan-
tage Adv(A∗) = Ω(√
k·εacc/p
log(1/δacc)log(1/δq)) between neighboring indices, with
δq=δacc/n.
Corollary 3.6 (Rate match within the score-release class) .Combining Theorems 3.4
and 3.5 , the joint privacy parameter satisfies εk=O(√
k·εacc)and an explicit attack
achieves MIA advantage Ω(√
k·εacc)on the same construction. Rates match; constants
do not (advantage-to-DP reduction vs. privacy-loss-distribution analysis), so we claim
rate-tightness, not exact-DP-parameter tightness.
Remark 3.7 (T op- K transfer: upper bound, not tightness) .A top- K-only output is a
14

deterministic function (post-processing) of M∗’s noisy score vector, so the top- K adver-
sary’s advantage is bounded above by M∗’s — the Θ(√
kε acc)rate is an upper bound
on top- K leakage, not a proven achievable rate. The empirical transfer is tested in
Sections 4.4 and 4.8 , where constants shrink but slope is preserved at the scales tested.
Collusion regimes. The construction realizes same-tenant collusion directly; cross-
tenant and external collusion reduce identically only under M4 (Remark 2.1 ), as flagged
in Section 2.3 .
A tighter lower-bound constant is achievable via the analytic Gaussian mechanism of
Balle and W ang (Balle and W ang, 2018 ), which removes thep
log(1.25/δq)factor and
matches the upper bound’s leading constant. W e leave this constant improvement to
follow-up work as the rate Θ(√
k)is unchanged.
3.6 Operational reading
F or services in Definition A.1 , the advertised per-account (εacc, δ acc)guarantee sharpens
to a joint (√
k·εacc,·)bound under k-collusion. Three corollaries:
1. Hidden privacy debt. A1-DP-per-account service is only√
10-DP≈3.2-DP under
k= 10 collusion — a 3.2× degradation invisible in the provider’s marketing.
2. Quadratic break-even. T o match a target joint ε∗
joint , the provider must shrink the
per-account budget to εacc=ε∗
joint/√kmax , where kmax is the assumed coalition size —
which the provider does not currently attempt to bound or audit.
3. Sublinear, not flat. Membership-inference advantage is non-decreasing in kat rate
Θ(√
k). Hence empirical AUC vs kat fixed per-account budget should follow a√
k
curve — the shape we test in Section 4.
4. Linear δdegradation. The failure-probability budget in Theorem 3.4 satisfies δjoint≤
δ+kδ acc, growing linearly inkeven as εgrows only as√
k. At a production setting
ofδacc= 10−6andk= 20 , the joint δis2×10−5— still policy-acceptable for most
15

regulators, but degrading faster than the headline ε-rate suggests. Providers reporting
both εandδshould track both under the coalition policy , not just the headline budget.
3.7 Alternative analysis via Rényi DP
The advanced-composition proof of Theorem 3.4 is convenient for exposition but loses
constants relative to a Rényi-DP analysis (Mironov, 2017 ). W e re-derive the rate via
RDP both to sharpen the constant and to expose the moments-accountant numerics we
use in Section 4for empirical privacy tracking.
Definition 3.8 (Rényi DP (Mironov, 2017 )).F orα > 1, a mechanism M is(α, ρ(α))-
RDP if for all neighboring D∼D′,Dα(M(D)∥M(D′))≤ρ(α),where Dαis the Rényi
divergence of order α.
Three facts about RDP we use without proof (all from (Mironov, 2017 )):
(R1) Gaussian mechanism. The Gaussian mechanism with sensitivity ∆ and noise
N(0, σ2)is(α, α∆2/(2σ2))-RDP for all α > 1.
(R2) Composition. Them-fold adaptive composition of (α, ρ i(α))-RDP mechanisms on
the same database is (α,P
iρi(α))-RDP — linear in m, with nop
log(1/δ)over-
head.
(R3) Conversion. (α, ρ)-RDP implies (ρ+log(1/δ)/(α−1), δ)-DP for any δ >0.
Theorem 3.9 (Sharper upper bound via RDP) .Under the hypotheses of Theorem 3.4 ,
the joint mechanism is (εRDP
k, δ)-col lusion-DP with
εRDP
k =∆p
2kn log(1/δ)
σ+kn∆2
2σ2. (3.6)
F or the calibration σ= ∆p
2nlog(1/δacc)/εacc that achieves per-account (εacc, δ acc)-DP
via RDP (R3) onnqueries, this simplifies to
εRDP
k =√
k·εacc·s
log(1/δ)
log(1/δacc)+Ok·ε2
acc
log(1/δacc)
. (3.7)
16

Sketch – ful l proof in appendix. By Lemma 3.3 , the joint mechanism is the kn -fold com-
position of the per-query Gaussian mechanism. By (R1) and (R2) , the composed mecha-
nism is (α, knα ∆2/(2σ2))-RDP . Applying (R3) and optimizing over α > 1,
εRDP
k(δ) = min
α>1knα∆2
2σ2+log(1/δ)
α−1
.
The optimum is at α∗= 1 + σp
2log(1/δ)/(kn∆2), yielding ( 3.6 ) after simplification.
Substituting the calibration σgives ( 3.7 ); constants match Theorem 3.4 on the leading
order, and the error term coincides with the small- εqresidual.
Improvement over Theorem 3.4 . Leading rate identical; the RDP route shrinks
the residual by a constant factor of 4(from kε2
acc/log(1/δacc)tokε2
acc/(4 log(1/δacc)),n-
independent, linear in k); see Section A.3 .
Numerical privacy tracking. The RDP form admits exact numerical evaluation via
the moments accountant of (Abadi et al., 2016 ): we update ρ(α)on a grid α∈ {2, . . . , 64}
per query and convert to (ε, δ)at audit-window close. This gives tighter realized (ε, δ)
pairs than the closed-form bound at moderate kand is what the audit pipeline reports.
W e do not claim that query correlation reduces worst-case DP cost (it does not for a fixed
Gaussian mechanism with fixed sensitivity); the tightening relative to the closed-form is
purely arithmetic.
3.8 Bridge to membership-inference advantage
The bounds of Theorems 3.4 ,3.5 and 3.9 characterize leakage in DP-theoretic terms. The
empirical attacks of Section 4measure the membership-inference advantage of k-collusion
against a held-out target document. This subsection states the standard reduction from
(ε, δ)-DP to MIA advantage (Y eom et al., 2018 ; Kairouz et al., 2017 ; Dwork and Roth,
2014 ) and applies it to obtain a falsifiable prediction for the AUC–vs– kcurve we test
empirically .
17

Definition 3.10 (Membership-inference advantage) .F or a mechanism M , target docu-
ment x∗, and k-collusion adversary A, the MIA advantage is Adv MIA(A;x∗) :=Pr[A(τ) =
1|x∗∈D]−Pr[A(τ) = 1 |x∗/∈D],where τis the pooled transcript across the kac-
counts.
Lemma 3.11 (DP-to-MIA reduction (Y eom et al., 2018 )).IfM is(ε, δ)-DP, then for
anyk-col lusion adversary A and any target x∗,
Adv MIA(A;x∗)≤eε−1 + 2 δ
eε+ 1≤ tanh (ε/2) + δ. (3.8)
F orε≤1, the right-hand side is further bounded by ε/2 + δ.
Combining the bound ( 3.8 ) from Lemma 3.11 with Theorem 3.4 :
Corollary 3.12 (MIA upper bound under k-collusion) .F or a multi-tenant RAG service
satisfying per-account (εacc, δ acc)-DP with εacc≤1, any k-col lusion adversary against a
target tenant satisfies
Adv MIA≤ tanh
1
2√
k·εacc
+O(kδ acc). (3.9)
Corollary 3.13 (MIA lower bound under k-collusion) .F or the explicit construction of
Theorem 3.5 , the constructed adversary A∗achieves
Adv MIA(A∗)≥1√
2π·√
k·εacc
4p
log(1/δacc)log(1.25/δq)
= Ω √
k·εaccp
log(1/δacc)!
.(3.10)
AUC prediction. T ranslating MIA advantage into ROC AUC (Y eom et al., 2018 ) gives
AUC =1
2(1 + Adv MIA). Combining Corollaries 3.12 and 3.13 , we predict an empirical
curve
AUC k=1
2+ Θ √
k·εacc
at fixed per-account ε-budget, (3.11)
saturating at AUC →1once√
k·εacc approaches 1. The headline experiment in Section 4
sweeps k∈ {1,2,5,10,20}andεacc∈ {0.5,1.0,2.0}to test ( 3.11 ). At deployment values
18

δacc= 10−6, n= 104, the closed form gives concrete audit verdicts: εaudit≈3.16 at
(k, ε acc) = (10 ,1),7.07 at(50,1),14.14 at(50,2),10.00 at(100,1);3the matched lower-
bound MIA advantage at production score gap ∆ = 0 .10 stays below 0.025 across the full
grid and below 0.01 at empirically tested εacc≤2(Corollary 3.13 scaled per Section 4.5 ),
reflecting that the worst-case DP bound is much looser than the realized advantage at
small ∆.
Empirical checks. The theory makes three predictions; P1, P3 are falsification gates
(a violation invalidates the bound); P2 is a diagnostic (not resolvable at our finite T).
(P1) Slope (gate) :∂AUC /∂√
kis positive and approximately constant for√
k·εacc≲1.
(P2) Scale col lapse (diagnostic) : AUC curves at different εacc collapse onto one when
plotted against√
k·εacc; resolved at higher T.
(P3) Tightness (gate) : empirical AUC sits within a constant factor of ( 3.9 ) but strictly
above 1/2 .
If P1 or P3 fails, the theory needs revision before the audit-protocol guarantees can be
trusted. P1+P3 passing is necessary , not suﬀicient evidence; P2 (when resolved) sharpens
it further.
4 Coordinated Attacks
This section validates the predictions P1–P3 from Section 3.8 empirically . W e instantiate
the Theorem 3.5 construction directly — kaccounts each issuing nprobe queries against
the scalar Gaussian mechanism M∗of ( 3.3 ) — and measure the membership-inference
AUC against neighboring indices. This is the most theory-bound experiment we can run;
deviations between empirical and predicted AUC isolate finite-sample noise rather than
modeling error.
3Generated by experiments/scripts/epsilon_audit_table.py ; full lookup grid at
paper/tables/eps-audit-lookup.tex .
19

4.1 Experimental setup
Mechanism. F or each pair of neighboring indices D, D′differing in one document of
tenant t∗, the mechanism returns, per query , the scalar ⟨q∗, x(b)
∗⟩+N(0, σ2)as in ( 3.3 ), with
σcalibrated by ( 3.4 ) so that nadaptive queries from one account compose to per-account
(εacc, δ acc)-DP via Lemma 3.2 . W e bypass the actual top- K retrieval pipeline of Section 2
for the headline experiment because (i) top- K post-processing of Gaussian-noised scores
cannot decrease privacy beyond the noise itself (the remark after Theorem 3.5 ), and
(ii) the closed-form predictions from Theorem 3.5 and corollary 3.12 hold against M∗
directly , providing the cleanest theory-empirics binding. A separate experiment validating
the rate transfer to top- K is reported in Section 4.4 .
Adversary . The lower-bound adversary A∗from Section 3.5 :kaccounts each issue
the same probe query q∗fornrounds, the empirical mean of all kn noisy samples is the
test statistic, and the optimal Bayes threshold is at the midpoint between the two world
means. Capabilities (C1) –(C4) are exercised: the accounts share posterior (Gaussian
likelihood with known variance), schedule queries within the per-account rate limit, and
pool responses through a single estimator.
Sweep. Coalition size k∈ {1,2,5,10,20}, per-account budget εacc∈ {1,2,4},δacc=
10−6, queries-per-account n= 10 000 (chosen to mirror a multi-day audit window at
production rate-limit budgets), score gap ∆ = 1 (max for unit-norm embeddings). Each
(k, ε acc)cell uses T= 10 000 paired-worlds Monte Carlo trials, giving a normal-approximation
standard error on AUC ofp
0.25/T≈0.005 , which resolves the smallest predicted ad-
vantage in the sweep ( ≈0.026 atk= 1, ε acc= 1 ) at∼5σ;nis fixed at production-window
scale rather than tuned for statistical power. Each trial draws fresh Gaussian random-
ness for both worlds and reports the pooled mean noisy score; AUC is computed by the
Mann-Whitney U statistic over the T×T pairs. T otal runtime is 87 seconds on a 2024
Mac M4. Code and config are in src/rag_collusion_privacy_audit/attacks/ and
experiments/configs/auc-vs-k.yaml respectively .
20

Scope. The §4.2–§4.6 experiments instantiate the same-tenant collusion regime of Sec-
tion 2.3 :kaccounts within the same tenant boundary coordinate against the tenant’s
index. The external regime (under M4, Remark 2.1 ) is reported separately in Section 4.7 ,
which compares both regimes head-to-head at matched (k, ε acc). The cross-tenant regime
is not separately swept; conditional on M4, its rate matches the others by Theorem 3.4 ,
and its empirical constant gap to the same-tenant baseline is bounded above by the
external-regime gap. Without M4 the cross-tenant and external regimes have zero leak-
age about Dt∗and are out of scope.
4.2 Headline result
Figure 2(a) reports empirical AUC against kfor each εacc on the scalar mechanism. Fig-
ure 2(b) reports the same attack against the full multi-tenant F AISS harness with top- K
retrieval and post-processing, validating the rate-transfer claim of Section 3.5 (see Sec-
tion 4.4 ).
The empirical and predicted curves agree to within one standard error in every cell. At
εacc= 4 , AUC climbs from 0.535 atk= 1 to0.640 atk= 20 — the empirical advantage
growing 4.0× across the sweep against a predicted growth factor of√
20≈4.5, with
the residual gap accounted for by mild Φ-saturation at the high- kend. At εacc= 2 the
curve is monotone but compressed (advantages 0.026 to0.150 ); at εacc= 1 the absolute
advantage is below 0.08 across the sweep, consistent with the calibrated σon the order of
thousands at production query budgets. Higher- Tresolution at small εacc would sharpen
these regimes; we report only the cells where the empirical signal sits above one stderr.
4.3 F alsifiability check
W e stated two falsification gates (P1, P3) and one diagnostic (P2) in Section 3.8 ; the
data inform each.
(P1) Slope. The prediction is that ∂AUC /∂√
kis positive and approximately con-
stant for√
k·εacc≲1. The high-signal εacc= 4 row clears this cleanly (advantages
21

12 5 10 20
Coalition size k0.500.550.600.650.700.750.800.85Membership-inference AUC
(a) Scalar mechanism *
acc=1
acc=2
acc=4
12 5 10 20
Coalition size k0.500.550.600.650.700.750.800.85
(b) Full top-K harness
acc=4
acc=8
acc=16
Figure 2: Empirical membership-inference AUC against coalition size k. (a) Scalar
mechanism M∗: the lower-bound construction of Theorem 3.5 atn= 10 000
queries, εacc∈ {1,2,4},T= 10 000 trials. Empirical curves match the predicted
1
2+1
2Φ(√
k ε acc/(4p
log(1/δacc)log(1.25/δq))) within one standard error in every cell. (b)
F ull top- K harness: same probe attack issued through MultiTenantRAGService with
K= 5 ,50 background documents, embedding dimension 32,n= 200 ,T= 2 000 trials.
T est statistic is the count of queries for which the target document index appears in the
returned top- K. Looser εacc∈ {4,8,16}is required to resolve the curve at finite T (see
Section 4.4 ). The√
kscaling is preserved through the top- K post-processing, with a
constant-factor degradation that we quantify below. Markers are empirical means with
normal-approximation ±1standard-error bars; dotted line at AUC = 1/2 marks chance.
δacc= 10−6throughout.
0.070,0.099,0.145,0.181,0.281 atk= 1,2,5,10,20). The low-signal rows are consistent
with the prediction but within finite-sample noise; cleanly falsifying at small εacc would
require T≥104, outside this paper’s scope. F ailure of P1 would indicate that the as-
sumption of independent per-query noise (M1) is violated — the only way the rate of
pooled-mean information growth diverges from√
kn/σis for the noise draws to correlate
across queries.
(P2) Scale collapse (diagnostic). Plotting AUC against√
k·εacc should collapse the
three εacc curves onto one. At T= 10,000 , Monte Carlo noise at the small-advantage end
is comparable to the inter-curve separation, so the data are silent on P2; a T≥105rerun
would resolve it. W e treat P2 as a diagnostic, not a gate: its absence does not loosen the
audit-protocol guarantees, which rest on P1 and P3.
22

(P3) Tightness. The empirical AUC sits within one standard error of the theoretical
lower bound ( 3.10 ) across all 15 cells, and is strictly above the chance line 1/2 for every
(k, ε acc)cell with εacc≥2. Combined with the upper bound ( 3.9 ), this confirms the
prediction ( 3.11 ) that the empirical AUC is Θ(√
k ε acc)within the regime tested — the
strongest empirical statement that finite-sample MIA experiments can support. F ailure
of P3 in the conservative direction (empirical AUC below the lower bound) would expose
either a finite-sample eﬀiciency loss the moments-accountant numerics in Section 3.7 can-
not recover, or a misspecification of the lower-bound adversary; failure in the aggressive
direction (above the upper bound) would falsify Theorem 3.4 .
4.4 Rate transfer through top- K post-processing
Post-processing only guarantees that top- K-only leakage is no larger than score-release
leakage; it does not preserve the rate. Figure 2(b) reports the empirical transfer: the
same probe query is issued through MultiTenantRAGService with the binary “target
appears in returned top- K” indicator as test statistic — strictly less informative than the
scalar adversary’s noisy score, since top- K collapses a continuous score to a discrete in-or-
out signal. W e observe, but do not prove, that the√
ktrend survives at the parameters
tested.
Empirical findings. (i) The√
kshape is preserved: at εacc= 16 the empirical AUC
climbs from 0.583 atk= 1 to0.811 atk= 20 — an advantage growth of 3.7× against a
predicted√
20≈4.5, the residual attributable to top- K post-processing’s discretization
cost on the binary hit indicator. (ii) The constant factor is substantially worse than the
scalar attack: at matched εacc= 4 , the harness AUC at k= 20 is0.593 vs the scalar’s
0.640 , a roughly 30% smaller advantage. (iii) Resolving the curve at production-realistic
εacc∈ {0.5,1,2}would require either substantially more trials or a stronger test statistic
(e.g., the noisy score conditional on top- K membership) than the binary hit indicator we
use here; this resolution is outside the present scope.
23

Operational reading. T op-K post-processing is, in practice, a meaningful defense-in-
depth layer beyond the per-account DP guarantee — attenuating the empirical advantage
by an observed factor of ∼0.9at our parameters relative to what the bare scalar mech-
anism leaks — but does not change the leakage rate’s dependence on k. A provider
committing to per-account (εacc, δ acc)cannot rely on the top- K bottleneck to mask co-
ordinated attack growth; the joint guarantee against coalition still degrades as√
kink,
with an empirical constant we do not bound analytically . The audit protocol of Section 5
certifies the worst-case bound (the scalar bound); in deployment, the realized leakage
may sit below the audit bound by a constant factor that depends on the retrieval K, the
index size, and the embedder.
4.5 T rained-embedder check
The §4.1–§4.4 experiments use random unit-sphere embeddings at dimension d= 32 . T o
check that the√
krate transfers to a trained embedder (and quantify the constant-factor
change), we re-run the scalar-mechanism attack of Section 4.1 with BAAI/bge-small-en-v1.5
(384-dim, MTEB-standard sentence embedder) over a hardcoded 50-paragraph factual
corpus. W e pick a target/decoy pair from the corpus with realized score gap closest to
∆ = 0 .40 (vs. the ∆ = 1 .0used in §4.2, which is the degenerate maximum-signal case):
the picker selects “The Earth orbits the Sun …” as target and “The Renaissance began in
Italy …” as decoy , yielding ∆ = 0 .41. The probe vector is the target embedding (match-
ing the §3.5 construction q∗=x∗). Other parameters match §4.1: εacc= 4 ,δacc= 10−6,
n= 200 queries per account, T= 2 000 trials per cell.
Figure 3reports the result. The empirical advantage grows from 0.027 atk= 1 to
0.130 atk= 20 — a growth factor of 4.84 against the predicted√
20≈4.47. The√
kscal-
ing rate is preserved within finite-sample noise. The absolute advantage at any matched k
is smaller than the random-embedder baseline at ∆ = 1 : atk= 20 , the bge-small AUC is
0.565 versus the random-embedder’s 0.640 . The ratio of advantages ( 0.130/0 .281 = 0 .46)
is close to the ratio of realized ∆s (0.41/1.00) within ∼12%, consistent with the closed-
form prediction that advantage scales linearly in ∆ in the small-advantage regime up to
24

1 2 5 10 20
Coalition size k0.5000.5250.5500.5750.6000.625Empirical AUC
advantage grows 4.84×, predicted 204.47
Real vs random embedder (scalar mechanism, acc=4)
Random embedder, =1.0
bge-small, =0.41 (realised)
Figure 3: Real-embedder rate transfer ( bge-small-en-v1.5 , 384-dim, n= 200 ,T=
2 000 ). At realized ∆ = 0 .41, the empirical advantage grows 4.84× across k∈
{1,2,5,10,20}against a predicted√
20≈4.47 — the√
krate is preserved. Absolute
advantages sit below the ∆ = 1 random-embedder baseline (grey squares) by a constant
factor consistent with the ratio of realized ∆.
finite-sample noise.
4.6 Alternative-adversary robustness check
The repeated-probe attack of Section 4.1 saturates the lower-bound construction by de-
sign. T o check that no nearby alternative adversary beats the√
krate, we run two
non-trivial variants at matched (k, ε acc)against the same scalar mechanism.
Adversary B — Bayes LR. The first variant replaces the pooled-mean statistic with
the exact Bayes log-likelihood-ratio T(y) = (∆/ σ2
eff)y−∆2/(2σ2
eff)with σ2
eff=σ2/(kn).
Because T is a monotone-increasing function of y, AUC under the LR test equals AUC
under the pooled mean trial-by-trial. Figure 4confirms this empirically: at every k∈
{1,2,5,10,20}, baseline AUC and LR AUC agree to four decimal places on the same
trial draws. The mean test is therefore already at the family-optimum within the LR-test
family .
25

1 2 5 10 20
Coalition size k0.5000.5250.5500.5750.6000.625Empirical AUC
Alternative adversaries (acc=4, n=104, T=5000)
Baseline (=1, pooled mean)
Bayes LR (monotone in mean)
Split =0.50
Split =0.25
Figure 4: Alternative-adversary AUC vs. katεacc= 4 ,n= 104,T= 5 000 trials per cell,
on the scalar mechanism of Theorem 3.5 . Baseline (blue circles) and Bayes LR (green
squares) coincide trial-by-trial (monotonicity identity). Diversified split adversaries at
ρ= 0.50 (orange triangles) and ρ= 0.25 (red triangles) trail baseline at every k >1. No
alternative adversary tested exceeds the baseline√
krate.
Adversary C — Diversified multi-target split. The second variant addresses the
named multi-target concern. A fraction ρ∈(0,1] of the kcolluders point their probes at
the true target; the remaining k(1−ρ)probe a random orthogonal direction (modeling
an adversary hedging across candidate targets without knowing which one is real). All
accounts spend the same per-account DP budget regardless of probe direction. W e sweep
ρ∈ {0.50,0.25}. The orthogonal probes contribute zero signal under both worlds; their
queries are wasted noise. Empirically (Figure 4): at k= 20 , baseline AUC = 0.638 ,
ρ= 0.50 gives 0.593 ,ρ= 0.25 gives 0.580 . Diversification strictly degrades the advantage
at every k >1; the effect grows with kas the wasted budget compounds.
The Bayesian-active-query and embedder-inversion threats named in Theorem 3.4 ’s
coverage are not exercised here — they would require an adaptive query loop and an ex-
plicit inversion oracle respectively — but the LR check rules out a wide class of statistic-
side improvements, and the split check rules out a wide class of probe-side hedging strate-
26

gies. Combined with the Theorem 3.4 upper bound on any adversary obeying the per-
account budget, we conclude that the√
krate is empirically tight in the neighborhood
of the §4.2 attack.
4.7 External vs. same-tenant collusion
Under M4 (Remark 2.1 ) the three collusion regimes reduce to the same per-query analysis
at the same rate Θ(√
k). W e test the M4-failure regime empirically: external models k
attacker accounts outside the victim tenant, each with a fresh per-account budget, query-
ing the victim index via a shared retrieval backend (M4 active). Each attacker account
applies fresh Gaussian noise per query calibrated by ( 3.1 ); sweep parameters match §4.4:
k∈ {1,2,5,10,20},εacc∈ {4,8,16},n= 200 ,T= 2,000 trials, 50 background docs,
K= 5 .
Figure 5shows the two regimes are statistically indistinguishable under M4 : across
all 15 cells the absolute AUC difference is at most 0.022 (median 0.010 ), within one stderr
(≈0.011 ). At the headline cell k= 20, ε acc= 16 , external AUC = 0.810 vs same-tenant
0.811 . Empirical√
kgrowth across k= 1→20: external 3.95×, same-tenant 4.11×, pre-
dicted√
20≈4.47×. Read correctly: this experiment quantifies external-collusion damage
conditional on access-control failure , not against ordinary external attackers, who have
zero same-index access by Equation ( 2.1 ) and zero leakage about Dt∗. T enant-boundary
access control isthe first line of defense; the audit-protocol coalition estimator (clustering
by query similarity , not by tenant) is the second.
4.8 Production-scale HNSW: user-observable vs auditor-observable
T o test whether the√
ktrend transfers to production ANN retrieval, we run the at-
tack against 106MS MARCO passages embedded with bge-small-en-v1.5 (384-dim,
L2-normalised), retrieved by HNSW (Malkov and Y ashunin, 2018 ) (M=64 ,ef cstr=200 ,
ef q=128 ). T arget/decoy are screened for in-corpus uniqueness (no other passage within
cosine 0.90); realized ∆ = 0 .133 . Sweep parameters match §4.4. W e report two statistics
distinguished by who observes them: the user-observable hit indicator (planted slot in re-
27

1 2 5 10 20
Coalition size k0.500.550.600.650.700.750.80Empirical AUC
External vs same-tenant collusion (top-K harness, T=2,000)
Same-tenant, =4
External, =4
Same-tenant, =8
External, =8
Same-tenant, =16
External, =16
Figure 5: External vs same-tenant collusion against the top- K harness at matched
(k, ε acc). Same-tenant baseline (open squares, solid) and external (filled circles, dashed)
at three εacc. Curves overlap across the full sweep; absolute AUC differences are bounded
by0.022 (median 0.010 ), within one stderr at T= 2,000 .
turned top- 5), and the auditor-observable pooled noisy score at the planted slot , available
only under the audit protocol’s attestations.
Figure 6reports both. User-observable: chance, marginal growth at highest εacc
(0.500→0.539 overk= 1→20) — attackers observing only top- K get near-zero leakage
at scale. Auditor-observable pooled noisy-score: at εacc= 16 , AUC climbs from 0.508
atk= 1 to0.582 atk= 20 , advantage growing from 0.016 to0.164 .The latter is
an audit-monitoring signal, not an attacker capability . Production multi-tenant
RAG without an audit primitive is harder to attack via top- K alone than the score-release
class would imply; the audit’s role is to expose the noise-then-select internals that recover
the leakage signal.
4.9 What the experiment does not test
T wo caveats bound the empirical claim. First , the alternative adversaries of Section 4.6
cover statistic-side optimization (Bayes LR) and probe-side hedging (multi-target split),
28

1 2 5 10 20
Coalition size k0.50.60.70.8Empirical AUC
(a) Hit indicator (§4.4 statistic)
Synthetic N=50, =4
HNSW N=106, =4
Synthetic N=50, =8
HNSW N=106, =8
Synthetic N=50, =16
HNSW N=106, =16
1 2 5 10 20
Coalition size k0.500.520.540.560.58Empirical AUC
(b) Pooled noisy score at slotid
=4
=8
=16
Figure 6: Production-scale HNSW on 106MS MARCO passages, bge-small-en-v1.5 em-
bedder, HNSW (M=64,ef cstr=200 ,ef q=128) ,∆ = 0 .133 ,T= 2,000 . (a) User-observable
hit-indicator AUC: chance at scale (curves flat across k). (b) Auditor-observable pooled
noisy-score at the planted slot: monotone curve preserved at εacc∈ {8,16}. The two
panels show different observation surfaces: (a) is what an attacker account sees; (b) is
exposed only by the audit protocol’s per-query attestations.
but not adaptive query loops where colluder i+ 1 ’s probe depends on colluder i’s ob-
servations. Theorem 3.4 bounds any such adaptive adversary; an explicit empirical in-
stantiation is left to follow-up work. Second , the trained-embedder check in Section 4.5
uses a single target/decoy pair from a 50-paragraph synthetic corpus at one realized ∆;
production deployments span a continuous distribution of ∆ across document pairs and
tenants. W e expect the linear scaling of advantage in ∆ to compose with the√
kscaling
in coalition size (Corollary 3.12 ); a full embedder-distribution sweep is outside the present
paper’s scope.
5 Audit Protocol
This section specifies a three-party audit protocol that lets a verifier V certify the joint
leakage bound of Corollary 3.6 against a deployed RAG service without requiring the
provider to disclose the index. The protocol is the contribution that distinguishes our
work from purely defensive RAG-DP literature ( e.g., (Cheng et al., 2025 )): existing
defenses modify the deployment to leak less per query; ours leaves the deployment intact
and produces a verifiable leakage bound that any third party can check.
29

5.1 T oolkit boundary
The protocol composes generic cryptographic primitives factored into upstream cryptographic-audit-protocols
with RAG-specific primitives shipping in this paper.
Upstream (generic, reusable):
(G1) ZK commitment over hashable artifacts; (G2) append-only Merkle ledger with
O(logn)inclusion + non-inclusion proofs; (G3) ZK proof template “ fwith public θ
was applied to private x”; (G4) committed-seed Gaussian noise: provider pre-commits
C seed=H(sW)before W ; per-query zq=DGS(H(sW∥recq);σ)via a specified discrete
Gaussian sampler DGS over the ledger record recq;sW is opened at audit close for
re-derivation; (G5) prover/verifier transcript schema.
This paper (RAG-specific):
(R1) embedder commitment C emb ; (R2) per-account query ledger schema + query-
time receipts ( 5.4 ) binding each served query to a public ledger position; (R3) noise-
then-select retrieval attestation πord ; (R4) cross-tenant containment proof πtenant ; (R5)
coalition-size estimator ˆkwith bound certificate πcoal ; (R6) per-tenant index-content
vector commitment C idx binding πord to deployed documents.
Each R-primitive is a composition of one or more G-primitives parameterized by
RAG-specific witnesses.
5.2 Setup and parties
Three parties:
• Provider P: knows the index D, embedder weights, and the secret randomness used for
DP noise. Wishes to demonstrate compliance with a privacy policy without revealing
D.
• V erifier V: a third party (regulator, auditor, customer’s representative) with no prior
trust relationship to P. Outputs a binary verdict and a leakage bound εaudit .
30

• Observer O: anyone who reads the public ledger. May submit audit queries to test
the protocol.
The protocol is parameterized by a hash function H, a non-interactive zero-knowledge
proof system Π (e.g., Groth16 or PLONK; we treat it abstractly), the public DP pa-
rameters (εacc, δ acc, σ)from Section 2, a policy-declared coalition cap kmax , and a policy-
declared joint failure-probability target δpolicy≥δacc at which the auditor reports the joint
(εaudit, δ policy)-collusion-DP guarantee. Both kmax andδpolicy are fixed at audit-window
opening as part of C policy (Section 5.3 ).
5.3 Phase A: commitment
Before the audit window W opens, P publishes:
C emb=H 
Emb∥prompt _tmpl∥tokenizer ∥schema (D)
(R1)
C idx={CDt}t∈T, C Dt=VCom 
{Emb(d) :d∈Dt}
(R6)
C seed=H(sW), s W∼ U({0,1}λ), λ≥256 (G4)
C ledger ,0=H(⊥) (G2)
C policy =H 
εacc∥δacc∥σ∥kmax∥δpolicy∥W
(5.1)
C emb binds embedder, prompt template, tokenizer, and per-tenant schema (cardinalities,
boundaries) — publishable metadata. C idx binds per-tenant document embeddings: veri-
fier learns only |Dt|, but every per-query proof must open positions of CDt(a)rather than
a free witness index. VCom is instantiable as a Merkle commitment over H(Emb(d))
(succinct O(log|Dt|)openings) or as a vector commitment with smaller commitment
size (Catalano and Fiore, 2013 ); we leave the choice to deployment (cost in Section 5.8 ).
C seed binds a fresh λ-bit seed sWdriving per-query noise (G4); pre-window publication pre-
vents post-hoc noise selection. C policy binds privacy parameters + audit window against
retroactive renegotiation, with the well-formedness constraint δpolicy > k maxδacc (required
for the audit’s δreparametrization in Section 5.6 ).
31

5.4 Phase B: per-query attestation
F or each query qsubmitted by account aduring W ,Pruns the standard retrieval pipeline
(Section 2) and additionally produces a zero-knowledge proof πqattesting four claims
jointly:
(A1) Embedder consistency : the embedding of qwas computed under the artifact bound
byC emb . (G3 instantiated with f=Emb , public input C emb , private witness Emb .)
(A2) Noise derivation from committed seed (G4): zq= DGS(H(sW∥recq);σ)forσin
C policy , with recq= (a, t(a), H(q),ts)andsW under C seed .DGS is a deterministic
discrete Gaussian sampler whose encoding is part of the audit-protocol specification.
T ogether with pre-window binding of C seed , this pins zqto the sampler output on
the committed input; P cannot cherry-pick noise.
(A3) Noise-then-select over committed index (R3 + R6): top- K indices are the K largest
of Score (q, E t(a)) +z, where Et(a)={Emb(d) :d∈Dt(a)}is the per-tenant em-
bedding vector opened from CDt(a). The proof carries openings suﬀicient to attest
top-K, binding retrieval to the deployed index — tying the deployment to the
per-query DP guarantee ( 3.1 ).
(A4) Cross-tenant containment (R4): every position opened in πqcomes from CDt(a),
never from CDt′with t′̸=t(a), i.e., the response respects the tenant boundary of
the issuing account.
Pappends rq= (a, t(a), H(q), H(top-K),ts, πq)to the Merkle ledger (G2), publishes
the new root C ledger ,t, and co-issues to aa signed query-time receipt ρq= (a, H (q), H(top-K),posq, C ledger ,t,SigP)
where posqis the leaf position and Sig uses a long-term identity key . Honest accounts (M5)
submit receipts to V at audit close; a missing or contradicted receipt is a non-inclusion
witness against the ledger. Past records are provable in O(logn)time; raw content is
stored only as hashes.
32

5.5 Phase C: coalition-size estimation
At the end of the audit window, P commits a coalition-size estimate ˆktogether with a
certificate πcoal (R5) attesting that an agreed-upon estimator was applied to the ledger
contents.
Estimator (abuse-detection signal). W e use a query-correlation detector: cluster
queries in W by hash collisions and approximate-nearest-neighbor proximity in Emb ’s
representation (which C emb binds, so the verifier can re-derive the clustering). Clusters
of≥2distinct accounts within a sliding window of length τare candidate coalitions; let
cbe the largest such cluster. The audit’s pass condition is c≤kmax . The bound εaudit
in ( 5.2 ) is computed from kmax , not from c: the estimator gates policy compliance but is
not a soundness condition on the DP guarantee. Section 5.7 discusses adversarial-evasion
limits.
Calibration. W e calibrate the cosine-similarity threshold θon a toy harness with A=
30 accounts, n= 100 queries per account, embedding dimension d= 32 , sweeping θ∈
[0.20,0.95] over 16 grid points with 200 trials per cell. The null distribution draws iid
uniform unit-sphere queries; the alternative replaces ktrue∈ {2,5,10,20}accounts with a
coalition that follows one of three patterns: ( P-A ) all colluders issue an identical probe
vector q∗; (P-B ) jittered probes q(i)
t=normalize (q∗+ζη(i)
t)with ζ= 0.10 andη∼ N (0, I);
(P-C ) probes drawn iid from a fixed paraphrase intent set of size 5. W e declare coalition
detection when ˆk≥2and pick the smallest θachieving null FPR ≤5%.
Figure 7(a) shows the null FPR as a function of θ: below θ≈0.70 the random unit-
sphere null is over-connected (FPR = 1 ), the transition is sharp ( 0.595 atθ= 0.75),
and the operating point lands at θ∗= 0.80 with achieved FPR = 0.040 . Figure 7(b)
shows ˆkrecovery against θatktrue= 10 : the estimator transitions from a single merged
cluster ( ˆk=A) to exact recovery ( ˆk= 10 ) over the same window. At θ∗, the estimator
achieves TPR = 1.00 at all four tested ktrue values across all three patterns, with ˆk
recovering ktrue exactly (zero variance across 200 trials). The estimator therefore meets
33

0.2 0.4 0.6 0.8
Proximity threshold 
0.00.20.40.60.81.0False-positive rate
(a) FPR on null ledger
FPR target = 0.05
0.2 0.4 0.6 0.8
Proximity threshold 
1015202530k (mean over trials)
(b) k at ktrue=10
P-A (identical probe)
P-B (jittered, =0.10)
P-C (5-paraphrase set)
ktrue=10Figure 7: Coalition-size estimator calibration on the toy harness ( A= 30 ,n= 100 ,
d= 32 ,200 trials per cell). (a) Null FPR as a function of the proximity threshold θ;
the operating point θ∗= 0.80 achieves FPR = 0.040 .(b) Mean ˆkatktrue= 10 for the
three attack patterns; the estimator transitions from a merged graph ( ˆk=Aat low θ) to
exact recovery ( ˆk= 10 ) across the same window. At θ∗, TPR = 1.00 at all four tested
ktrue∈ {2,5,10,20}for all three patterns, with ˆkexactly recovering ktrue .
the policy specification on these three patterns; the limitation is that all three share
high within-coalition cosine similarity . An adversary who deliberately diversifies queries
below θin the embedder’s representation evades detection by construction — this is the
residual gap that Section 5.7 acknowledges and that motivates the embedder-binding step
of Section 5.3 .
Certificate. πcoal is a ZK proof that ˆkis the output of the agreed estimator on the ledger
contents (witness: the queries and their cluster assignments; public input: C ledger ,W and
the estimator’s parameters bound by C policy ). This forces P to commit to a coalition
estimate consistent with the public ledger.
5.6 Phase D: verification
The verifier Vexecutes:
1. Commitment integrity : all of C emb, C idx, C seed, C policy, C ledger ,0were published before W
opened. At audit close P opens sW;V checks H(sW) =C seed and re-derives zqfor
each sampled proof. Index openings in πqverify against CDt(a).
2. Per-query soundness : for each rqin the ledger, Π.Verify (πq,·) = 1 for claims (A1) –(A4) .
34

Sampled mode: Vverifies suniformly-random records and reports a property-testing
guarantee — any β-fraction of non-conforming records is detected with probability
≥1−(1−β)s(Section 5.8 ). This does not certify worst-case soundness; rare bad
events ( βnear 1/|W|) need full verification.
3. Ledger integrity + receipt consistency :C ledger ,W is a valid append-only extension of
C ledger ,0(G2). F or each submitted receipt ρq,V verifies SigP+ Merkle inclusion at
posqinC ledger ,W, and that the receipted (H(q), H(top-K)) match the ledger record. A
receipt without a consistent ledger record is a forgery witness; the audit fails.
4. Policy-cap check :Π.Verify (πcoal,·) = 1 and the estimator’s largest valid cluster c≤
kmax . Ifc > k max the audit fails: the deployment violates its declared coalition cap.
5. Leakage bound :δpolicy is the target final joint failure probability . Set the composition
slack δcomp =δpolicy−kmaxδacc (C policy enforces δpolicy > k maxδacc). Plug kmax, δ comp into
Theorem 3.4 :
εaudit =p
kmax·εacc·s
log(1/δcomp)
log(1/δacc)+Okmaxε2
acc
log(1/δacc)
. (5.2)
The reparametrization makes the final joint failure probability exactly δpolicy ;kmax
rather than ckeeps soundness independent of estimator adversarial robustness.
Voutputs (P ASS , ε audit)if all four checks pass, else (F AIL,·).
5.7 Security argument
Theorem 5.1 (Audit soundness) .Meaningful-verdict conditions: non-trivial kmax , M4
(Remark 2.1 ) for cross-tenant or external regimes, and coalitions within the estimator’s
pattern class; generation-channel privacy is out of scope (Section 7.1 ). Under these
andΠ sound, H col lision-resistant, VCom binding, DGS statistical ly close to N(0, σ2I):
Mode A. With ful l per-record verification and at least one honest-account receipt sample
(M5): if V outputs (P ASS , ε audit), the deployed retrieval-channel mechanism Retr (2.1 )
is(εaudit, δ policy)-col lusion-DP for any coalition ≤kmax against any single tenant in W .
35

Mode B. With suniformly-sampled records: any β-fraction violation of (A1) –(A4) is
detected with probability ≥1−(1−β)s;εaudit is conditional on conformance of unsampled
records.
Sketch – ful l proof in appendix. Soundness of Π+(A1) forces the artifact bound by C emb .
(A2) + pre-window C seed pinszqto the sampler output on the committed seed and ledger
record, statistically close to a fresh N(0, σ2I)draw. Claim (A3) pins the ordering to noise-
then-select, so the per-query mechanism is exactly the Gaussian mechanism of ( 3.3 ),
which is (εq, δq)-DP per ( 3.1 ). Claim (A4) confines responses to Dt(a), so neighboring-
index changes outside tenant t∗do not affect the transcript — privacy reduces to the
k-collusion analysis of Section 3. The policy-cap check rejects audits where the estimator
detects c > k max ; an audit that passes therefore applies to deployments respecting their
declared cap. Applying Theorem 3.4 with k=kmax yields the claimed bound.
What the protocol does not guarantee. The audit cannot detect adversaries who
evade the estimator (R5 trades sensitivity for false positives; tightening it is empirical,
Section 4). Ledger completeness rests on honest-account receipts (R2, M5): a coalition
where every account colludes with P can drop its own queries from the ledger, but only
by sacrificing every customer-facing audit trail. The protocol also assumes embedder de-
terminism within W (M3); detecting surreptitious mid-window embedder changes would
require a per-query embedder-version attestation, deferred to follow-up work.
5.8 Cost
W e estimate costs from a model anchored on Groth16 (Groth, 2016 ) per-constraint tim-
ings: proof size ∼192 bytes (3 BN254 elements), verification ∼ 3 pairings (constant in
N), proving ∼1µs/constraint (single-core M-class); cross-validated against Setty (Setty ,
2020 )’s R1CS measurements on similarly-sized circuits (up to 220constraints). Linear
scaling in constraint count; sub-linear FFT/MSM corrections absorbed for our index
sizes. Model in experiments/scripts/zk_cost_model.py ; two circuit modes.
36

T able 1: Estimated Groth16 per-query costs for the noise-then-select audit circuit at
d= 384 , top- K= 5 . Optimized circuit pre-commits C sc,qand proves score-openings
consistent with CDt(a)onO(K logN)positions; naive recomputes all N inner products in-
circuit. Proving and setup derived from ∼1µs/constraint; verification constant per proof.
PLONK (no trusted setup) and ST ARK (post-quantum) alternatives carry comparable
proving costs on the same circuit. Measured numbers in the camera-ready microbench-
mark.
Mode Index size N Constraints Prove (s) Setup (s)
Optimized 1039.8×1040.1 0.05
Optimized 1048.9×1050.9 0.4
Optimized 1058.8×1068.8 4.4
Optimized 1068.8×10788 44
Naive 1038.7×1050.9 0.4
Naive 1048.6×1068.6 4.3
Naive 1058.6×10786 43
Naive 1068.6×108856 428
Circuit modes. The naive circuit recomputes ⟨q, x i⟩in-circuit for every i, paying 2dN
constraints ( d= 384 ). The optimized circuit has P pre-commit a per-query score vec-
torC sc,q= VCom (Score (q, E t(a))) binding all N scores, and proves: (i) G4 noise deriva-
tion for every zi(O(64N)); (ii) noisy-top- K selection on the full noise-added vector
(O(8N log2K)); (iii) score-opening consistency for K+O(logN)positions touched by (ii)
— opened scores equal in-circuit inner products against CDt(a)openings. The remaining
N−K−O(logN)scores are bound only by VCom , not re-checked against embeddings:
P commits to all N scores, the circuit constrains opened positions only .
Estimated costs. T able 1reports model outputs for index sizes N∈ {103,104,105,106}
atK= 5 and the embedding dimension above. At a production-scale index of N= 105
documents, the optimized circuit is ∼8.8M constraints with proving time ∼9s per query;
the naive circuit is ∼86M constraints ( ∼86 s). At N= 106, optimized is ∼88M constraints
(∼88 s); naive crosses into the hundreds of seconds and is no longer per-query feasible.
V erification is constant at ∼8ms per proof (3 pairings); setup is one-off, on the order of
half the proving cost.
Sampling reduces verifier cost (property-testing soundness). F ull per-query at-
testation at production query rates ( 107queries per audit window per tenant is realistic)
37

is infeasible at present hardware. The sampling-based verification of Section 5.6 reports a
property-testing guarantee: srandom samples detect a fraction- βviolation at confidence
1−ηwith s≥ log(1/η)/β(i.e., 1−(1−β)s≥1−η). A policy of “detect β≥0.01 at
η= 2−20” yields s≈1,400 samples per window, ∼11 s of verifier time, independent of
|W|. Crucially , sampling does not certify worst-case soundness: a single non-conforming
query in |W|is detected with probability ≈s/|W|, not 1−2−s. Adversary-resistant
deployments need parallel provers or recursive SNARKs.
Storage. Ledger storage is O(|W|)in queries (Merkle leaves at 32 bytes plus per-query
witness). Each Groth16 proof is 192 bytes uncompressed on BN254. At |W|= 106,
public ledger ∼30 MB; the verifier downloads only sampled proofs ( ∼1,400 atβ= 0.01,
η= 2−20) plus the Merkle root. C idx (R6) adds O(|Dt|)provider-side storage for a Merkle
instantiation (sub-MB up to |Dt|= 105); a vector-commitment instantiation (Catalano
and Fiore, 2013 ) trades smaller commitment size against larger openings.
External calibration. The 1µs/constraint figure anchors T able 1against Setty (Setty ,
2020 )’s Groth16 baseline ( ∼220R1CS in seconds on a single core; our model gives ∼1s at
106constraints). The model is a within-order-of-magnitude estimate; absolute numbers
would require a direct arkworks Groth16 microbenchmark.
6 Related W ork
Our work composes three sub-fields: RAG-specific privacy attacks, coordinated-attacker
analyses in federated learning, and cryptographic audit primitives. The formal core of
Section 3— after Lemma 3.3 reduces kaccounts to one kn -query analyst — is standard
advanced-composition / RDP accounting; our contributions are the RAG-specific threat
modeling, the audit protocol, and the empirical rate-transfer investigation.
38

6.1 Single-attacker privacy attacks on RAG
The recent wave of RAG-specific membership-inference attacks establishes the threat
surface but stops at k= 1 . Zeng et al. (Zeng et al., 2024 ) provide a foundational privacy
taxonomy for RAG; their threat model is descriptive rather than adversarial. Li et al. (Li
et al., 2024 ) introduced S2MIA, exploiting semantic similarity between generated text
and indexed documents. W ang et al. (W ang et al., 2025 ) sharpened this with diﬀiculty-
calibrated MIA, and F eng et al. (F eng et al., 2025 ) identified RAG-specific generation
artifacts that betray index membership. Liu et al. (Liu et al., 2024 ) introduced a mask-
based MIA targeting document-specific tokens. The current state of the art is Naseh et
al. (Naseh et al., 2025 ), achieving stealthy membership inference in only 30 queries, and
Gao et al. (Gao et al., 2025 ), reporting 97.4% AUC via differential calibration. All six
papers operate within the same threat model: one isolated attacker, one stateless API,
no coordination. Our analysis subsumes this threat model at k= 1 (where Theorem 3.4
reduces to standard per-account DP) and extends it to the regime production deployments
actually face.
Adjacent attacks complement the RAG-MIA literature. Morris et al. (Morris et al.,
2023 ) invert text embeddings. Cohen et al. (Cohen et al., 2024 ) are the only published
RAG-privacy work to consider any multi-actor effect, but via self-replicating prompt-
injection worms rather than strategic coordination — their setting is propagation, not
collusion-as-DP-degradation.
Defenses are nascent. Cheng et al. (Cheng et al., 2025 ) (RemoteRAG) propose a
privacy-preserving cloud-RAG design that modifies the retrieval pipeline; our protocol
leaves the deployment intact and audits existing claims. W u et al. (W u et al., 2025 )
address the related but distinct multi-query DP setting (single user, multiple queries,
per-document budget) — they fix k= 1 and vary query count, we fix queries and vary k.
39

6.2 Coordinated-attacker analyses in federated learning
F ederated learning has a mature literature on colluding clients, but it targets training-
time aggregation, not inference-time API access. Lyu et al. (Lyu et al., 2023 ) (AAAI) is
the only paper we found with an explicit colluded-attacker model: kclients coordinate
backdoor injection through the FL aggregation channel. Pasquini et al. (Pasquini et
al., 2022 ) (CCS) demonstrate that multi-attacker coordination breaks cryptographic FL
defenses via correlated parameter updates — methodologically transferable but designed
against a stateful aggregator, not a stateless inference API. The MIA survey of Hu et
al. (Hu et al., 2022 ) enumerates colluding-client threats in FL but does not model them
against inference. Production RAG sits structurally between the FL and single-attacker
MIA worlds: stateless like the latter, multi-account like the former, but not directly
modeled by either.
Concurrent composition. The Θ(√
k)rate of Theorem 3.4 is not a new composi-
tion phenomenon. V adhan and W ang (V adhan and W ang, 2021 ) establish concurrent-
composition bounds for kanalysts with (ε, δ)-DP quotas against a shared database with
the same Θ(p
klog(1/δ′)·ε)leading rate; Rogers et al. (Rogers et al., 2016 ) give adaptive-
budget odometers; Kaiser et al. (Kaiser et al., 2026 ) (SaTML 2026) study related collu-
sion vulnerabilities in individual -DP . Our contribution is not the composition theorem
itself but identifying that this phenomenon governs per-account RAG privacy budgets
under multi-account pooling, deriving the RAG-specific MIA prediction, and designing a
verifier-runnable audit for coalition-aware claims.
6.3 Cryptographic audit primitives
The audit-protocol toolkit (Section 5) draws on the mature ZK-SNARK literature origi-
nating with Groth16 (Groth, 2016 ) and Ben-Sasson et al. (Sasson et al., 2014 ), without
inheriting any one paper’s full apparatus. The closest related construction in spirit is the
body of secure-aggregation work for federated learning, which audits training-side aggre-
gation rather than inference-time retrieval; our setting requires the inverse abstraction.
40

DP-side foundations are well established. Dwork and Roth (Dwork and Roth, 2014 )
provide the canonical reference for DP and the per-query Gaussian mechanism; Dwork et
al. (Dwork, Rothblum, et al., 2010 ) introduced advanced composition; Kairouz et al. (Kairouz
et al., 2017 ) gave the tight composition theorem we use as the closed-form benchmark
in Theorem 3.4 . Mironov (Mironov, 2017 ) introduced Rényi DP , which underpins our
sharper analysis (Theorem 3.9 ); the moments-accountant of Abadi et al. (Abadi et al.,
2016 ) is the empirical-tracking machinery we propose for audit-window numerics. Y eom et
al. (Y eom et al., 2018 ) provide the DP-to-MIA reduction (Lemma 3.11 ) that bridges Sec-
tion 3’s bounds to Section 4’s empirical AUC.
7 Discussion
7.1 Limitations
The bound and protocol of Sections 3and 5rest on five modeling assumptions (M1–M5,
Section 2.3 ) whose violation in deployment shifts the meaning of a P ASS verdict.
Scope of the DP claim. Theorems 3.4 and 5.1 are statements about Retr (2.1 ), not
the full RAG output LLM(q,Retr(q, a)): the generation step re-accesses private documents
post-selection (not post-processing of a DP score vector). W e certify only the selected-
document-IDs / noisy-scores channel; generation-channel privacy is a complementary
audit predicate that should be composed with ours (existing single-attacker RAG-MIA
defenses sit there).
Coalition cap is policy-declared, not derived. The protocol verifies (εaudit, δ policy)-
collusion DP for coalitions up to kmax . The provider sets kmax during the commitment
phase; the auditor checks that the coalition-size estimator ˆkdoes not exceed it. A provider
who declares kmax= 1 recovers the per-account guarantee with a passing audit but no pro-
tection against any coordinated adversary . W e see two paths forward: (i) regulatory floors
that mandate kmax≥kfloor as a condition of publishing a passing audit (e.g., kfloor≥10
41

for production multi-tenant services, calibrated to observed sock-puppet creation rates),
and (ii) auditor-led adversarial estimators that statistically lower-bound the true coali-
tion size from the public ledger, with the divergence between the auditor’s estimate and
the provider’s ˆkas the real audit signal. Both are deferred to future work.
Coalition-size estimator can be gamed. Our query-correlation estimator (R5 in
Section 5.1 ) clusters queries by embedding proximity within a sliding time window. An
adversary aware of the estimator’s threshold can dilute coordination signal by paraphras-
ing probes or staggering query timing across accounts. W e validate the estimator on the
toy harness in Phase 2 of the experimental work, but a robustness study that adversar-
ially trains query-paraphrasing strategies against a fixed estimator threshold is outside
the present scope. A natural defense is to recurse the audit: prove statistically that
any coalition exceeding ˆkwould have been detected with high probability , against an
adversary class that includes paraphrased coordination.
ZK overhead is asymptotically benign but practically heavy . Section 5.8 sketches
a Groth16-based instantiation with ∼10–60 seconds per query proof and ∼106con-
straints for a noise-then-select circuit on indices of size ∼105. At production query
rates ( 107queries per audit window per tenant is realistic), full per-query attestation
is infeasible at present hardware. The sampling-based verification in Section 5.6 gives
a property-testing guarantee (detect any fraction- βviolation at confidence 1−ηwith
s=⌈log(1/η)/β⌉samples); it does not certify worst-case soundness against rare bad
events. A microbenchmark replacing predicted asymptotics with measured numbers is
left to follow-up implementation work.
Embedder trust is delegated to the commitment. The protocol binds the em-
bedder via C emb but does not audit whether the embedder is itself privacy-preserving
(e.g., resistant to embedding inversion (Morris et al., 2023 )). A provider committing to a
known-leaky embedder passes our audit while leaking heavily through retrieval hits. W e
treat embedder choice as a separate audit predicate that should be composed with ours;
42

embedding-inversion-aware audit is a natural follow-up.
7.2 Responsible disclosure
The vulnerability we describe is structural, not implementation-specific. Every multi-
tenant RAG deployment that publishes a per-account (εacc, δ acc)-DP guarantee is affected;
no patch beyond an architectural change to bound coalition size or shrink the per-account
budget by√kmax closes the gap. Because the gap is structural rather than exploitable
as a one-off, we do not believe it warrants an embargoed coordinated-disclosure cycle
in the conventional sense. Our disclosure plan is to (i) release the arXiv preprint two
weeks before submission to allow the major multi-tenant RAG providers (Microsoft, Ope-
nAI, Anthropic, Pinecone, W eaviate) to update their documentation and customer-facing
guarantees, (ii) contact each provider directly at the time of arXiv release with a copy
of the audit protocol and a one-page summary of the bound and its operational reading
from Section 3, and (iii) publish the toolkit ( cryptographic-audit-protocols ) with
the camera-ready version. W e are open to revising the disclosure timeline if any specific
provider requests a longer window.
7.3 Policy implications and adoption pathway
F raming. W e present the audit as a technical foundation, not a deployed-ready specifi-
cation. The bound, protocol, and cost model are concrete; the regulatory and commercial
path to adoption is forward-looking and depends on actors outside this paper’s scope.
Regulatory slot. The audit fits a slot several frameworks contemplate but have not
specified: a third-party predicate producing a verifiable guarantee without auditor access
to model or training data. EU AI Act Articles 13 and 50 are silent on how to operationalize
privacy claims checkable externally; the protocol’s (P ASS , ε audit)output is a natural target.
The DSA’s Articles 34–35 similarly admit cryptographic audit as one mode of compliance.
At the time of writing, no European DP A, the U.S. FTC, or the U.K. ICO has published a
formal call for RAG-specific privacy audit; the protocol sits ahead of mandate rather than
43

in response to one. Our position is that εaudit is the right unit of regulatory disclosure —
not the per-account budget the provider currently advertises.
Adoption incentives. Absent mandate, three voluntary incentives apply . Reputation:
passing audit is a verifiable trust signal the per-account budget cannot supply , particularly
for enterprise customers in regulated sectors. Early adoption ahead of mandate: the AI
Act’s high-risk rules phase in through 2026–2027; instrumenting now is cheaper than
under a deadline. Liability reduction: a passing audit creates a contemporaneous record
defensible in privacy litigation. None is suﬀicient alone; we observe their joint weight may
suﬀice for at least one major provider as a competitive differentiator.
What we do not claim. The protocol does not bind the provider’s choice of kmax :
kmax= 1 trivially passes, recovering the (insuﬀicient) status quo. The audit’s value
derives from kmax being meaningful; that is a contractual choice the protocol enforces
but does not motivate. Closing this gap requires regulatory floors or industry-consortium
norms, neither of which is in our purview to deliver.
7.4 F uture work
Beyond Sections 4.9 and 7.1 : (i) analytic-Gaussian RDP would close the residual constant-
factor gap between Theorem 3.9 and Theorem 3.5 ; (ii) extension to RAG-augmented
agents with persistent memory , where temporal accumulation composes with cross-account
coalition; (iii) Stackelberg analysis of the auditor-provider interaction is a natural sequel;
(iv) the most operationally important open question is whether the coalition-size estima-
tor can be made adversarially robust without prohibitive ZK machinery .
References
Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., T alwar, K., and Zhang,
L. (2016) Deep Learning with Differential Privacy . In: Proceedings of the 2016 ACM
SIGSAC Conference on Computer and Communications Security (CCS) . A vailable at:
https://doi.org/10.1145/2976749.2978318 .
44

Balle, B., and W ang, Y.-X. (2018) Improving the Gaussian Mechanism for Differential Pri-
vacy: Analytical Calibration and Optimal Denoising . A vailable at: ICML 2018; arXiv
preprint arXiv:1805.06530. A vailable at: https://doi.org/10.48550/arxiv.1805.06530 .
Catalano, D., and Fiore, D. (2013) V ector Commitments and Their Applications. In:
Public-Key Cryptography – PKC 2013 . Lecture Notes in Computer Science. Springer.
A vailable at: https://doi.org/10.1007/978-3-642-36362-7_5 .
Cheng, Y., Zhang, L., W ang, J., Y uan, M., and Y ao, Y. (2025) RemoteRAG: A Privacy-
Preserving LLM Cloud RAG Service. In: Findings of the Association for Computa-
tional Linguistics: ACL 2025 . A vailable at: https : / / doi . org / 10 . 18653 / v1 / 2025 .
findings-acl.197 .
Cohen, S., Bitton, R., and Nassi, B. (2024) Here Comes The AI W orm: Unleashing Zero-
click W orms that T arget GenAI-Powered Applications . A vailable at: arXiv preprint
arXiv:2403.02817. A vailable at: https://doi.org/10.48550/arxiv.2403.02817 .
Dwork, C., and Roth, A. (2014) The Algorithmic F oundations of Differential Privacy .
Now Publishers. A vailable at: https://doi.org/10.1561/0400000042 .
Dwork, C., Rothblum, G. N., and V adhan, S. (2010) Boosting and Differential Privacy . In:
51st Annual IEEE Symposium on F oundations of Computer Science (FOCS) . A vailable
at: https://doi.org/10.1109/FOCS.2010.12 .
F eng, K., Zhang, G., Tian, H., Xu, H., Zhang, Y., Zhu, T., Ding, M., and Liu, B. (2025)
RAGLeak: Membership Inference Attacks on RAG-Based Large Language Models. In:
Australasian Conference on Information Security and Privacy (ACISP) . A vailable at:
https://doi.org/10.1007/978-981-96-9101-2_8 .
Gao, X., Meng, X., Dong, Y., Li, Z., and Guo, S. (2025) DCMI: A Differential Cali-
bration Membership Inference Attack Against Retrieval-Augmented Generation. In:
ACM SIGSAC Conference on Computer and Communications Security (CCS) . A vail-
able at: https://doi.org/10.1145/3719027.3765103 .
Groth, J. (2016) On the Size of Pairing-Based Non-interactive Arguments. In: Advances
in Cryptology – EUROCR YPT 2016 . A vailable at: https://doi.org/10.1007/978- 3-
662-49896-5_11 .
Hu, H., Salčić, Z., Sun, L., Dobbie, G., Y u, P . S., and Zhang, X. (2022) Membership
Inference Attacks on Machine Learning: A Survey . ACM Computing Surveys . A vailable
at: https://doi.org/10.1145/3523273 .
Kairouz, P ., Oh, S., and Viswanath, P . (2017) The Composition Theorem for Differential
Privacy . IEEE T ransactions on Information Theory . A vailable at: https://doi.org/10.
1109/TIT.2017.2685505 .
Kaiser, J., Ziller, A., T riantafillou, E., Rückert, D., and Kaissis, G. (2026) Y our Pri-
vacy Depends on Others: Col lusion V ulnerabilities in Individual Differential Privacy .
A vailable at: arXiv preprint.
Li, Y., Liu, G., W ang, C., and Y ang, Y. (2024) Generating Is Believing: Membership
Inference Attacks against Retrieval-Augmented Generation. In: ICASSP 2025 (IEEE
International Conference on Acoustics, Speech and Signal Processing) . A vailable at:
https://doi.org/10.1109/icassp49660.2025.10889013 .
Liu, M., Zhang, S., and Long, C. (2024) Mask-based Membership Inference Attacks for
Retrieval-Augmented Generation. In: Proceedings of the ACM on W eb Conference
2025 (TheW ebConf/WWW) . A vailable at: https://doi.org/10.1145/3696410.3714771 .
Lyu, X., Han, Y., W ang, W., Liu, J., W ang, B., Liu, J., and Zhang, X. (2023) Poisoning
with Cerberus: Stealthy and Colluded Backdoor Attack against F ederated Learning.
45

In: Proceedings of the AAAI Conference on Artificial Intel ligence . A vailable at: https:
//doi.org/10.1609/aaai.v37i7.26083 .
Malkov, Y. A., and Y ashunin, D. A. (2018) Eﬀicient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small W orld Graphs. IEEE T ransac-
tions on Pattern Analysis and Machine Intel ligence . A vailable at: https://doi.org/10.
1109/TPAMI.2018.2889473 .
Mironov, I. (2017) Rényi Differential Privacy . In: 2017 IEEE 30th Computer Security
F oundations Symposium (CSF) . A vailable at: https://doi.org/10.1109/CSF.2017.11 .
Morris, J. X., Kuleshov, V., Shmatikov, V., and Rush, A. M. (2023) T ext Embeddings
Reveal (Almost) As Much As T ext . A vailable at: arXiv preprint arXiv:2310.06816;
EMNLP 2023. A vailable at: https://doi.org/10.48550/arxiv.2310.06816 .
Naseh, A., Peng, Y., Suri, A., Chaudhari, H., Oprea, A., and Houmansadr, A. (2025)
Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation.
In: ACM SIGSAC Conference on Computer and Communications Security (CCS) .
A vailable at: https://doi.org/10.1145/3719027.3744840 .
Pasquini, D., F rancati, D., and Ateniese, G. (2022) Eluding Secure Aggregation in F ed-
erated Learning via Model Inconsistency . In: Proceedings of the 2022 ACM SIGSAC
Conference on Computer and Communications Security . A vailable at: https : / / doi .
org/10.1145/3548606.3560557 .
Rogers, R., Roth, A., and Ullman, J. (2016) Privacy Odometers and Filters: Pay-as-you-
Go Composition . A vailable at: arXiv. A vailable at: https://doi.org/10.48550/arxiv.
1605.08294 .
Sasson, E. B., Chiesa, A., and Garman, C. (2014) Zerocash: Decentralized Anonymous
Payments from Bitcoin. IEEE S&P . A vailable at: https://doi.org/10.1109/sp.2014.36 .
Setty , S. (2020) Spartan: Eﬀicient and General-Purpose zkSNARKs Without T rusted
Setup. In: Advances in Cryptology – CR YPTO 2020 . A vailable at: https://doi.org/10.
1007/978-3-030-56877-1_25 .
V adhan, S., and W ang, T. (2021) Concurrent Composition of Differential Privacy . In:
Theory of Cryptography Conference (TCC 2021) . Lecture Notes in Computer Science.
Springer. A vailable at: https://doi.org/10.1007/978-3-030-90453-1_20 .
W ang, G., He, J., Li, H., Zhang, M., and F eng, D. (2025) RAG-leaks: diﬀiculty-calibrated
membership inference attacks on retrieval-augmented generation. Science China In-
formation Sciences . A vailable at: https://doi.org/10.1007/s11432-024-4441-4 .
W u, R., W ang, E., and W ang, Y.-X. (2025) Beyond Per-Question Privacy: Multi-Query
Differential Privacy for RAG Systems. In: NeurIPS 2025 W orkshop on Reliable ML
from Unreliable Data .
Y eom, S., Giacomelli, I., F redrikson, M., and Jha, S. (2018) Privacy Risk in Machine
Learning: Analyzing the Connection to Overfitting. In: 2018 IEEE 31st Computer
Security F oundations Symposium (CSF) . A vailable at: https://doi.org/10.1109/CSF.
2018.00027 .
Zeng, S., Zhang, J., He, P ., Xing, Y., Liu, Y., Xu, H., Ren, J., W ang, S., Yin, D., Chang, Y.,
and T ang, J. (2024) The Good and The Bad: Exploring Privacy Issues in Retrieval-
Augmented Generation (RAG). In: Findings of the Association for Computational
Linguistics: ACL 2024 . A vailable at: https : / / doi . org / 10 . 18653 / v1 / 2024 . findings -
acl.267 .
46

A F ull proofs
This appendix expands the four proof sketches of Sections 3and 5.7 into self-contained
arguments. W e retain the body sketches as quick references; nothing in the appendix
changes the theorem statements.
Definition A.1 (Gaussian-noised score-release class) .A RAG mechanism is in this class
if its per-account (εacc, δ acc)guarantee is obtained by composing nindependent per-query
Gaussian-noised score mechanisms with sensitivity ∆ and noise scale σcalibrated to per-
query (εq, δq)-DP as in ( 3.1 ), with per-query noise draws independent across queries and
accounts (M1).
A.1 Proof of Lemma 3.3 (Coordination reduction)
Proof. W e construct an adaptive analyst A†that, given the same posterior and output
channel as A, produces a transcript with the same distribution. Under M1, if colluders
shared randomness the simulator’s kn independent draws would carry more information
and the equality τAd=τA†would degrade to a one-sided inequality — still a valid upper
bound.
Construction. A†schedules kvirtual accounts {ai}with per-account rate r. At each
round it chooses an account aiwith nonzero rate budget and a query qfromA’s posterior
((C1) ), issues q, receives response r, and appends (ai, q, r)to its transcript.
Distribution. The Gaussian mechanism of ( 3.3 ) has per-query privacy depending only
onqandDt∗, not on the account label. So Pr [τA†]and Pr [τA]marginalize to the same
distribution; capabilities (C1) ,(C4) ,(C2) ,(C3) make the conditioning, channel, and
scheduling latitude identical. Joint transcripts agree.
Three col lusion regimes. Same-tenant collusion: queries target Dt∗directly , the con-
struction applies verbatim. Cross-tenant/external collusion: under M4 (Remark 2.1 ), col-
luding accounts share retrieval access to Dt∗and the reduction goes through identically;
without M4 the regimes have zero leakage about Dt∗and the reduction is vacuous.
A.2 Proof of Theorem 3.4
Proof. By Lemma 3.3 , it suﬀices to upper-bound the privacy of kn adaptive (εq, δq)-DP
queries on Dt∗.
Composition. Apply Lemma 3.2 with m=kn ,ε0=εq,δ0=δq,δ′chosen below:
εk(δ′)≤p
2kn log(1/δ′)·εq+kn·εq(eεq−1). (A.1)
F orεq∈(0,1],eεq−1≤2εq, so the second term is bounded by 2knε2
q.
Substituting the per-query calibration. Equation ( 3.1 ) gives εq=εacc/p
2nlog(1/δacc).
Identify δ′with δ(the body theorem’s parameter is the composition slack; the final joint
failure probability is δ+kδ acc, tracked below). Then
p
2kn log(1/δ)·εq=p
2kn log(1/δ)·εaccp
2nlog(1/δacc)
=√
k·εacc·s
log(1/δ)
log(1/δacc),
47

matching ( 3.2 )’s dominant term. The second term of ( A.1 ) satisfies 2knε2
q=kε2
acc
log(1/δacc).
F ailure-probability accounting. The total δbudget consists of (i) the composition
slack δ, and (ii) the per-query failure probabilities, kn·δq=kδ acc by ( 3.1 ). Summing,
δjoint=δ+kδ acc,which matches the theorem statement.
A.3 Proof of Theorem 3.9
Proof. By Lemma 3.3 , the joint mechanism is the kn -fold composition of the per-query
Gaussian mechanism of ( 3.3 ) onDt∗.
RDP composition. By (R1) , the per-query mechanism is (α, α∆2/(2σ2))-RDP for all
α > 1. By (R2) , the kn -fold composition is (α, knα ∆2/(2σ2))-RDP .
Optimal-order conversion. Apply (R3) toεRDP
k(α, δ) =knα∆2/(2σ2) + log(1/δ)/(α−
1). Setting ∂/∂α= 0 gives α∗−1 =σp
2log(1/δ)/(kn∆2); substituting yields εRDP
k(α∗, δ) =
∆p
2kn log(1/δ)/σ+kn∆2/(2σ2),which is ( 3.6 ).
Calibration substitution. The same σfrom ( 3.1 ) (σ= ∆p
2nlog(1/δacc)/εacc) satis-
fies the RDP route at m=n, equating to (εacc, δ acc). Substituting in the bound above:
∆p
2kn log(1/δ)/σ=√
k ε accp
log(1/δ)/log(1/δacc)and the residual kn∆2/(2σ2) =kε2
acc/(4 log(1/δacc)),
a constant-factor improvement (1/4 vs. 1) over Theorem 3.4 ’s residual, with leading-order
coeﬀicient matching ( 3.2 ) exactly . This gives ( 3.7 ).
A.4 Proof of Theorem 5.1
Proof. Five attestations compose. (A1) Opening C emb at audit close, plus collision-
resistance of H and soundness of Π on (A1) , pins retrieval to the committed embedder.
(A2) (A2) verifies zq=DGS(H(sW∥recq);σ); pre-window binding of C seed rules out post-
hoc seed selection, so zqis statistically close to N(0, σ2I)(Sasson et al., 2014 ). (A3) (A3)
pins TopK to score-then-noise over the embeddings opened from C idx; binding-of- VCom
onCDt(a)(R6) prevents the prover from substituting a private witness index for the de-
ployed one. (A4) (A4) restricts every C idx-opening to CDt(a). (A5) πcoal attests c≤kmax ;
the FPR/TPR profile (Section 5.5 ) bounds detection failure under tested patterns, with
evasive coalitions a residual gap (Section 5.7 ). Combining (A1)–(A4) pins every per-query
mechanism to ( 3.3 ) on the deployed index; applying Theorem 3.4 atk=kmax ,δ=δpolicy
yields ( 5.2 ). The bound is independent of empirical c: the estimator gates, it does not
enter soundness.
48