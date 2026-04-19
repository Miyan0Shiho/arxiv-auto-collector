# Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge

**Authors**: Andre Bacellar

**Published**: 2026-04-15 23:56:35

**PDF URL**: [https://arxiv.org/pdf/2604.14488v1](https://arxiv.org/pdf/2604.14488v1)

## Abstract
In any domain where knowledge accumulates under formal authority -- law, drug regulation, software security -- a later document can formally void an earlier one while remaining semantically distant from it. We formalize this as Controlling Authority Retrieval (CAR): recovering the active frontier front(cl(A_k(q))) of the authority closure of the semantic anchor set -- a different mathematical problem from argmax_d s(q,d). The two central results are: Theorem 4 (CAR-Correctness Characterization) gives necessary-and-sufficient conditions on any retrieved set R for TCA(R,q)=1 -- frontier inclusion and no-ignored-superseder -- independent of how R was produced. Proposition 2 (Scope Identifiability Upper Bound) establishes phi(q) as a hard worst-case ceiling: for any scope-indexed algorithm, TCA@k <= phi(q) * R_anchor(q), proved by an adversarial permutation argument. Three independent real-world corpora validate the proved structure: security advisories (Dense TCA@5=0.270, two-stage 0.975), SCOTUS overruling pairs (Dense=0.172, two-stage 0.926), FDA drug records (Dense=0.064, two-stage 0.774). A GPT-4o-mini experiment shows the downstream cost: Dense RAG produces explicit "not patched" claims for 39% of queries where a patch exists; Two-Stage cuts this to 16%. Four benchmark datasets, domain adapters, and a single-command scorer are released at https://github.com/andremir/car-retrieval.

## Full Text


<!-- PDF content starts -->

Controlling Authority Retrieval:
A Missing Retrieval Objective for Authority-Governed Knowledge
Andre Bacellar
andremi@gmail.com
Abstract
In any domain where knowledge accumulates
under formal authority—law, drug regulation,
software security—a later document can for-
mally void an earlier one while remaining se-
mantically distant from it. We formalize this
asControlling Authority Retrieval (CAR):
recovering theactive frontier front(cl(A k(q)))
of the authority closure of the semantic anchor
set—a different mathematical problem from
arg max ds(q, d).
The two central results are:Theorem 4(CAR-
Correctness Characterization) gives necessary-
and-sufficient conditions onanyretrieved set
RforTCA(R, q) = 1 —frontier inclusion and
no-ignored-superseder—independent of how R
was produced.Proposition 2(Scope Identifia-
bility Upper Bound) establishes φ(q) as a hard
worst-case ceiling: for any scope-indexed algo-
rithm (one with no access to the authority rela-
tion⇝),TCA@k≤φ(q)·R anchor (q), proved
by an adversarial permutation argument over
theκ(q)indistinguishable authority chains.
Supporting structure:Theorem 1(Objective
Mismatch) proves TCA and Recall@ kare
decoupled—TCA < ε while Recall@ k=
1for any ε >0 —and a corollary shows
TCA@ k= 0 when N+(q)≥k documents
outscorec∗, explaining why larger dense mod-
els are strictly worse.Theorem 2(Closure Ne-
cessity) proves any system achieving TCA = 1
must perform proactive entity-scoped search.
Theorem 3(Decomposition) proves anchor
discovery and authority resolution are each
necessary; Corollary 3 proves TCA = 1 is
achievable when φ(q) = 1 .Proposition 1
(Authority Error Factorization) organizes TCA
=R anchor·φ·R frontier ; Lemma 1 proves the
three factors are separable failure modes by
construction.Proposition 3(Frontier Recovery
Complexity) establishes Ω(|R σ|)oracle-call
lower bound and separates it from the RSSG’s
O(|R σ|2·m)implementation cost.
Three independent real-world corpora vali-
date that the proved structure is consequen-tial: security advisories (Dense TCA@5 =
0.270 , two-stage0.975), SCOTUS overrul-
ing pairs (Dense = 0.172 , two-stage0.926),
FDA drug records (Dense = 0.064 , two-
stage0.774). A GPT-4o-mini experiment
shows the downstream cost: Dense RAG pro-
duces explicit “not patched” claims for39%
of queries where a patch exists; Two-Stage
cuts this to 16%. Four benchmark datasets,
domain adapters, and a single-command scorer
are released at https://github.com/
andremir/car-retrieval.
1 Introduction
In any domain where knowledge accumulates in-
crementally, later facts can formally override ear-
lier ones. Legal precedents are overruled by later
courts. FDA drug approvals are superseded by re-
call enforcement actions. Security patches void
prior vulnerability disclosures. Compliance clear-
ances are overridden by blackout announcements.
In each case, the correct answer to a query does not
depend on the most semantically relevant document
alone—it depends on whether that document has
beensupersededby a later one. We call this family
of retrieval problemscontrolling authority retrieval,
and show that standard retrieval systems fail on it
structurally across all three authority-governed do-
mains we test.
Retrieval-augmented generation (RAG) (Lewis
et al., 2020) ranks documents by semantic rele-
vance to the query. This is silent on supersession.
A RAG system asked whether a drug approval is
still valid retrieves the approval document (highest
relevance), not the recall notice (low relevance, dif-
ferent vocabulary). It has no mechanism to detect
that the approval was voided.
Running example.A security engineer asks: “Is
the prototype pollution vulnerability in lodash still
unpatched?” The most relevant document is the
CVE disclosure—it matcheslodash,prototype pol-arXiv:2604.14488v1  [cs.IR]  15 Apr 2026

lution, andunpatchedexactly, and states no fix is
available. The correct answer requires the patch
release note published two months later, which for-
mally supersedes the disclosure for the same pack-
age and CVE. In a 2,300-document corpus mixing
CVE disclosures and release notes, this patch note
ranks 420th by dense similarity: it describes a “re-
lease” with “fixed in version 4.17.12”—vocabulary
orthogonal to the vulnerability query. Dense re-
trieval surfaces the disclosure (Recall@5 = 0for
the patch note), and TCA = 0: the system answers
that the vulnerability is unpatched and has ignored
the controlling authority document.
Contributions.
1.Formalization: CAR target front(cl(A k(q)))
over partially ordered corpora; operators cl(·),
front(·) ; quantities ∆(q) (semantic-authority
divergence), φ(q) (scope identifiability), κ(q)
(scope ambiguity ratio); metric TCA; entity-
scope axiom on the supersession relation.
2.Primary theory(the center of the paper):
•Theorem 4(Characterization): R
achieves TCA = 1 if and only ifit sat-
isfies frontier inclusion and no-ignored-
superseder. Necessary-and-sufficient
conditions on any retrieved set, indepen-
dent of retrieval method. Any future al-
gorithm is auditable against these two
conditions.
•Proposition 2(Scope Identifiability Up-
per Bound): for any scope-indexed al-
gorithm (no access to ⇝),TCA@k≤
φ(q)·R anchor (q). Proved by adversarial
permutation: the κ(q) authority chains
are informationally indistinguishable to
any scope-indexed algorithm, giving a
worst-case ceiling, not an average-case
claim.
3.Supporting theory(establishes the problem
and organizes the picture):
•Theorem 1 (Objective Mismatch): TCA
and Recall@ kdecouple when ∆(q)>0 ;
Corollary: TCA@ k= 0 when N+(q)≥
k.
•Theorem 2 (Closure Necessity): any
TCA = 1 system must perform proac-
tive entity-scoped search.
•Theorem 3 (Decomposition): anchor dis-
covery and authority resolution are eachnecessary; Corollary 3 (Achievability)
proves TCA= 1whenφ(q) = 1.
•Proposition 1 (Authority Error Factoriza-
tion) + Lemma 1 (Failure Mode Separa-
bility): TCA =R anchor·φ·R frontier ; the
three factors address separable, indepen-
dently occurring failure modes.
•Proposition 3 (Frontier Recovery Com-
plexity): Ω(|R σ|)oracle lower bound;
RSSG costs O(|R σ|2·m) without oracle
access.
•Four formal CAR generalizations (Def-
initions 8–11) as open problem state-
ments.
4.Empirical validation: Three real-world cor-
pora (GHSA, SCOTUS, FDA) confirm Dense
TCA@5 ∈ {0.064,0.172,0.270} ; Two-Stage
achieves {0.774,0.926,0.975} . Scale wors-
ens Dense. GPT-4o-mini downstream: Dense
causes “not patched” assertions for 39%; Two-
Stage reduces to 16%. Datasets and scorer
released; seven open research directions in
§8.
2 Problem Formalization
2.1 Temporally Ordered Knowledge Bases
Definition 1(Document and Supersession Rela-
tion).Atemporally ordered knowledge base K=
{d1, . . . , d n}where each document dihas a times-
tamp di.time , an entity scope di.σ⊆Σ , and con-
tentd i.text.
Asupersession rule Rinduces d1⇝Rd2(“d2
supersedes d1”) when: (d1.type, d 2.type)∈
R.type pairs ,d2.time> d 1.time , and
R.scope(d 1, d2) =True.
The supersession relation ⇝is the transitive
closure over all rules. A document disactivein K
if nod′∈Ksatisfiesd⇝d′.
Entity-scope axiom.We require every rule R
to satisfy: R.scope(d 1, d2) =True only if d1.σ∩
d2.σ̸=∅ . Consequently, d1⇝d 2implies d1.σ∩
d2.σ̸=∅ (supersession never connects documents
with disjoint entity scopes). This axiom is satisfied
by all five domain examples above and is used in
the proof of Theorem 2.
Definition 2(Controlling Authority Retrieval).A
retrieval task is acontrolling authority taskif the
correct document to return is not the most semanti-
cally relevant document in the corpus, but rather
the document thatcontrolsthe semantically rele-

vant one under a formal precedence or superses-
sion relation. Controlling authority tasks form a
proper subset of retrieval tasks, characterized by
three properties:
(i)There exists an authoritative ordering ⇝on
documents independent of semantic similarity
to the query. ( ⇝is any strict partial order;
temporal precedence is one instance, but au-
thority can be non-temporal, e.g., statutory
hierarchy over case law, or ontological sub-
sumption.)
(ii)The query vocabulary aligns with a non-
controlling document (theanchor).
(iii) The controlling document has domain-distinct
vocabulary (different event type, action terms,
or register).
Definition 3(Authority Closure and Active Fron-
tier).Let A⊆K be a set of documents and ⇝the
authority order on K. Theauthority closureof A
is:
cl(A) =A∪ {d′∈K:∃d∈A, d⇝d′}
Theactive frontierof a set S⊆K is the set of
maximal (non-superseded) elements:
front(S) ={d∈S:∄d′∈Ss.t.d⇝d′}
The operators cl(·) andfront(·) are standard con-
structions in order theory (closure under a partial
order; antichain of maximal elements); the con-
tribution is identifying that standard retrieval opti-
mizes neither, and characterizing the resulting gap.
These two operators expose the central distinc-
tion of CAR:
Standard retrieval:R∗
k(q) = arg max
|R|=kX
d∈Rs(q, d)
(1)
CAR target:C∗
k(q) = front(cl(A k(q)))
(2)
where Ak(q)is the top- ksemantic anchor set.
These are different mathematical problems: clre-
quires traversing authority edges, and front re-
quires checking domination within the retrieved
set—neither operation can be expressed as a rele-
vance ranking.
Definition 4(Semantic-Authority Divergence).
For a query q, let d1= arg max ds(q, d)
(the highest-scoring anchor) and F(q) =
front(cl({d 1}))(the active frontier of the anchor’sauthority chain, a non-empty set). Thesemantic-
authority divergenceis:
∆(q) =s(q, d 1)−max
c∈F(q)s(q, c)
∆(q)≥0 always ( d1is the highest-scoring doc-
ument; every c∈F(q) scores at most as high).
∆(q) = 0 iff the best-scoring element of the ac-
tive frontier is semantically aligned to the query,
in which case standard retrieval suffices. When
∆(q)>0 , all elements of F(q) score strictly be-
lowd1, and every standard ranker prefers d1.F(q)
may contain more than one element (e.g., a drug
subject to two simultaneous supersession types);
we write c∗∈F(q) for any element of the active
frontier.
Definition 5(Scope Identifiability, Scope Ambi-
guity, and Closure Violation).Let Dσ(q)={d∈
K:σ(q)∩σ(d)̸=∅} be the set of documents
sharing the query’s entity scope, and let κ(q) =
|Dσ(q)|/|cl(A(q))| ≥1 be thescope ambiguity ra-
tio.
Thescope identifiabilityof a queryqis:
φ(q) = min
1,1
κ(q)
∈(0,1]
φ(q) = 1 iffκ(q) = 1 (the entity scope uniquely
identifies a single authority chain, no disambigua-
tion needed). φ(q)<1 when multiple author-
ity chains share the query’s scope, requiring rule-
based disambiguation. In practice, κ(q) is esti-
mated from corpus metadata without oracle access
tocl(A(q)) ; the RSSG localizes scope by building
the supersession graph over the retrieved neighbor-
hood only.
A retrieved set Rhas aclosure violationif
∃d∈ R such that d⇝d′andd′/∈ R. Closure
violations are the retrieval-level error underlying
TCA failures: a system can achieve correct answer
extraction (Acc = 1) with a closure violation (TCA
= 0), precisely the gap TCA is designed to detect.
Standard retrieval optimizes (1); CAR requires
computing (2). This vocabulary gap is structural,
not a tuning failure (Theorem 1, §3).
Domain examples.The framework is domain-
independent:
•Legal:Dobbs(2022) supersedesRoe v.
Wade(1973); rule = (LATER RULING ⇝
PRIOR RULING) for the same constitutional
question.

•Incident response: SYSTEM RESTORED su-
persedes SYSTEM DOWN for the same host.
•Software security: CVE PATCH RELEASED
supersedes VULNERABIL-
ITY DISCLOSED for the same CVE
ID.
•Drug regulation: FDA RECALL supersedes
DRUG APPROV AL for the same compound.
•Compliance(our instantiation): BLACK-
OUT ANNOUNCED supersedes
PRE CLEARANCE APPROVED for
the same security ticker.
Definition 6(TC-MQA Instance).ATC-MQA
instanceis (K, Q, a∗)where Kis a temporally
ordered KB, Qis a natural language query, and
a∗=f({d∈K:dis active}) is the deterministic
answer from the active documents.
2.2 Temporal Compliance Accuracy
Definition 7(TCA and ProvRec).Let R=
{r1, . . . , r k}be the top- kretrieved documents. De-
fine:
• ANSWERCORRECT=1[ˆa=a∗]
• NOIGNOREDSUPERSEDER : for every super-
seded d∈ R , at least one d′withd⇝d′is
also inR
TCA=ANSWERCORRECT∧NOIGNOREDSUPERSEDER
ProvRec(Provenance Recall) is the fraction of gold
documents in R, irrespective of supersession struc-
ture. TCA and ProvRec are complementary: TCA
measures compliance-safe retrieval; ProvRec mea-
sures archival completeness.
Remark 1(Generality of TCA).TCA requires only
a supersession relation ⇝and a gold answer func-
tion. It applies to any domain where documents
have a formal validity lifecycle—legal, medical,
regulatory, or operational.
2.3 Four TC-MQA Hop Types
T0 (No supersession):One active document an-
swers the query. Equivalent to standard single-hop
retrieval.
T1 (2-hop direct supersession): d1⇝d 2; the
query refers to d1’s subject. The correct answer
requires d2, which may be semantically distant
fromQ.T2 (3-hop chain): d1⇝d 2⇝d 3; the full
chain must be available for TCA = 1.
T3 (3-hop cross-provenance):The superseding
document d3originates from a different document
source (session, jurisdiction, team) than d1and
d2, while sharing entity scope. This tests retrieval
across provenance boundaries. Query templates for
all four hop types are given in Appendix B.
2.4 Generalized CAR Variants
The theorems and Proposition 1 characterizestatic,
deterministic, single-authority CAR with recover-
able scope. The following four definitions formal-
ize the canonical extensions, each corresponding
to relaxing one component offront(cl(A k(q))):
Definition 8(Latent-Scope CAR).When the entity
scope σ(q) is not recoverable from the query text,
it is a latent variable with posterior p(σ|q) . The
latent-scope CAR targetis:
CAR latent(q) = E
σ∼p(σ|q)
front(cl(A k(q, σ)))
or the MAP approximation with ˆσ=
arg max σp(σ|q) . Latent-scope failures
are precisely the cases where φ(q) cannot be
estimated from query text alone; scope must be
inferred from an anchor-side entity linker. The
22.6% gap from the 77.4% FDA ceiling is an
empirical instance of this regime.
Definition 9(Probabilistic-Authority CAR).When
authority edges carry confidence π(d, d′)∈[0,1]
rather than being deterministic, theprobabilistic
authority closureat thresholdθis:
clπ(A) =A∪ {d′∈K:∃d∈A, π(d, d′)≥θ}
Thesafe frontierat confidence αisfront α(S) =
{d∈S: Pr[dis maximal inS]≥α} .
Probabilistic-authority CAR arises when supers-
ession edges must be inferred from text (weak legal
treatment, soft policy override, noisy recall linkage)
rather than read from structured metadata.
Definition 10(Multi-Authority CAR).When m
authority relations ⇝1, . . . ,⇝ mcoexist (chrono-
logical precedence + jurisdictional hierarchy +
exception override), thejoint authority closureis:
cl∪(A) =A∪ {d′∈K:∃i∈[m],∃d∈A, d⇝ id′}
The active frontier must satisfy all mdominance re-
lations simultaneously; the resulting structure is a
lattice of authority neighborhoods. Multi-authority

CAR captures law (jurisdiction ×chronology ×
exception), drug regulation (approval ×recall×
class-level black-box warning), and policy systems
(version×source trust×exception grant).
Definition 11(Streaming CAR).Under real-time
document arrivals {dN+1, dN+2, . . .} , the active
frontier changes with each insertion: a new doc-
ument dnewwithdold⇝d newremoves doldfrom
front(cl(A k(q))) . Thestreaming CAR problemis
to maintain front(cl(A k(q))) under insertions in
sublinear amortized time per query. Each insertion
invalidates at most |front| cached retrievals (those
superseded by the new document), giving a natural
incremental update structure.
3 Theorems
Result Role Claim (independent of pipeline)
Theorem 4 Correctness N&S conditions onany R: frontier inclusion +no-
ignored-superseder
Proposition 2 Limit Hard worst-case ceiling φ(q) on all scope-indexed algo-
rithms
Theorem 3+Cor. 3 Construction Two-stage is sufficient when φ(q) = 1 ; T1, T2 prove
each component necessary
Proposition 3 CostΩ(|R σ|)oracle lower bound; RSSG costs O(|R σ|2m)
without oracle
Theorems 1–2, P1, L1 Context / synthesis Mismatch, closure necessity, factorization, separability
Assumption 1(Standing Assumptions for Theo-
rems 1–4 and Propositions 1–3).All results in this
section assume:
(i)Static corpus: Kis fixed during retrieval (no
concurrent insertions or deletions).
(ii)Deterministic authority relation: ⇝is fixed;
each pair (d, d′)has a known, time-invariant
supersession status (no edge uncertainty).
Probabilistic-authority CAR relaxes this (Def-
inition 9).
(iii) Single authority relation: one partial order
onK. Multi-authority CAR (Definition 10)
relaxes this.
(iv)Recoverable scope: σ(q) is directly ex-
tractable from Qby a perfect extractor, or
φ(q) is computable from corpus metadata.
Latent-scope CAR (Definition 8) relaxes this.
(v)Deterministic answer function: the TC-MQA
answer function fin Definition 6 is a fixed
deterministic map from front(cl(A k(q))) to
an answer; no distributional generation.
(vi)Algorithm models: Proposition 2 ranges over
scope-indexed algorithms (Definition 14);
Proposition 3 measures cost in authority-
comparison oracle calls (Definition 15).3.1 Theorem 1 — Problem Existence:
Objective Mismatch
Definition 12(Relevance-Based Retriever).A re-
triever Fisrelevance-basedif it ranks docu-
ments by s(Q, d.text) without access to ⇝or
timestamps. Its output set satisfies Fk(Q) =
arg max |R|=kP
d∈Rs(Q, d).
Theorem 1(Objective Mismatch).For any
relevance-based retriever F, any fixed k≥1 , and
anyε >0 , there exists a CAR instance (K, Q, a∗)
with∆(Q)>0 such that TCA(F, K, Q, k)< ε
while Recall@k= 1 . Consequently, TCA and
Recall@ kare decoupled: arbitrarily high rele-
vance recall is compatible with arbitrarily low TCA
when∆(Q)>0.
Proof. Construct family Knwith: (i)target doc-
ument d1at time t1with high relevance to Q
(query refers to d1’s subject); (ii) n−1 documents
d2, . . . , d nwith the same type and similar vocab-
ulary as d1but different entity scopes, all before
t1; (iii)superseding document d∗att∗> t1with
d1⇝d∗, using domain vocabulary distinct from
Q(different event type, different action terms).
d1scores highest on s(Q,·) . As n→ ∞ ,d∗
is displaced to rank ≥n. For fixed k:Pr[d∗∈
top-k]≤k/n→0.
TCA requires retrieving d∗for both AnswerCor-
rect (only active document) and NoIgnoredSuper-
seder ( d1is retrieved and d1⇝d∗). Choosing
n > k/εgives TCA< ε.
Corollary 1(Quantitative Divergence Bound).Let
N+(q) =|{d∈K:s(Q, d)> s(Q, c∗)}|where
c∗∈front(cl(A k(q))) is the controlling document.
For anydeterministicrelevance-based ranker (As-
sumption 1):
TCA@k= 0wheneverN+(q)≥k
Proof. A deterministic relevance-based ranker out-
puts the kdocuments with highest s(Q,·) . If
N+(q)≥k , allkslots are occupied by documents
strictly outscoring c∗; hence c∗/∈ R and TCA
= 0.
Note: N+(q)≥k is a separately stated condition
stronger than ∆(q)>0 (which implies only d1
outscores c∗, not all kslots). For a corpus where
fraction pof documents score above c∗,E[N+] =
p(N−1) , so the condition holds with probability →
1asN→ ∞ , explaining the scale-harm pattern in
Table 7.

3.2 Theorem 4 — Correctness: Exact
Characterization of TCA = 1
Theorem 4(CAR-Correctness Characterization).
A retrieved set Rachieves TCA(R, q) = 1 if and
only if it simultaneously satisfies:
(i)Frontier inclusion: front(cl(A k(q)))⊆ R —
every element of the active frontier is present.
(ii)No ignored superseder: ∀d∈ R that is su-
perseded (∃d′′∈Ks.t.d⇝d′′):∃d′∈ R
withd⇝d′.
These are thenecessary and sufficientconditions on
any retrieved set, regardless of the retrieval method
used to constructR.
Proof. (⇒) Suppose TCA = 1.Condition (i):
By Definition 6, a∗=f(front(cl(A k(q)))) un-
der the deterministic answer function f(Assump-
tion 1(v)).AnswerCorrectis semantic—the derived
answer must match the ground truth—whereas
this theorem establishes the structural condition
onRthat makes that semantic condition hold, in-
dependent of how retrieval was performed. An-
swerCorrect = 1 therefore requires fto evaluate
over the complete active frontier; since fis de-
fined on {d∈K:dis active} and the active ele-
ments of cl(A k(q)) are exactly front(cl(A k(q))) ,
every element of front(cl(A k(q))) must appear in
R.Condition (ii):Directly from the definition of
NoIgnoredSuperseder (Definition 7).
(⇐) Condition (i) is precisely the structural hy-
pothesis under which the semantic answer func-
tionfevaluates over the correct active set: every
c∗∈front(cl(A k(q))) is inR, giving AnswerCor-
rect= 1. Condition (ii) is exactly NoIgnoredSuper-
seder, giving TCA= 1.
Note on strength:Condition (ii) requires only
thatat least onesuperseder of each superseded
document is retrieved, not all. The chain d1⇝
d2⇝d 3admits R={d 1, d3}as TCA-correct
(condition (i): d3∈front ; condition (ii): d3∈ R
witnesses d1⇝d 3) even without d2. A strictly
stronger condition—requiring every intermediate
chain member—would be sufficient but not neces-
sary.
3.3 Theorem 2 — Problem Structure: Closure
Necessity
Definition 13(Anchor-Based vs. Proactive En-
tity Search).A retrieval system performsanchor-
based entity searchif it promotes entity-scoped
later events only when a triggering (anchor) eventfrom the same entity scope already appears in the
initial retrieval. It performsproactive entity search
if it retrieves entity-scoped events independently of
the initial retrieval.
Theorem 2(Closure Necessity).Let Sbe a re-
trieval system. If Sachieves TCA = 1on all TC-
MQA instances, including those in Theorem 1’s
family where the superseder d∗is arbitrarily dis-
similar to Q, thenSmust perform proactive entity-
scoped search: it must retrieve d∗via a function of
d∗.σord∗.time, independent ofs(Q, d∗.text).
Proof. Suppose Sachieves TCA = 1 on all in-
stances, including those in Theorem 1’s proof
where s(Q, d∗.text)→0 asn→ ∞ . Since d∗is
never retrieved by relevance alone ( Pr[rank(d∗)≤
k]→0 ),Smust use a source other than s(Q,·) to
retrieve d∗. Since d1⇝d∗implies d1.σ∩d∗.σ̸=∅
(supersession is entity-scoped), and d1is retrieved
by relevance, Smust access d1.σandd∗.time to
locate d∗. This is proactive entity-scoped search by
definition.
Remark 2(Scope vs. Fine-Tuned Encoders).The-
orem 2 applies to all TC-MQA instances, including
adversarially constructed ones. A bi-encoder fine-
tuned on specific (Q, d∗)pairs achieves TCA >0
on training examples by implicitly learning entity-
scope conditioning: training pulls f(d∗)toward
f(Q) because d∗.σ∩Q.σ̸=∅ appears as a su-
pervision signal. This is a learned implementation
of proactive entity search, not a counterexample.
The theorem’s construction is preserved for cor-
pus instances outside the fine-tuning set: one can
always add vocabulary-disjoint superseders the
encoder has never seen, restoring the impossibil-
ity. Fine-tuning can reduce the semantic-authority
divergence ∆(q) for seen instances; Theorem 2
characterizes what a system mustcompute over the
instance class, not how it is parameterized.
3.4 Theorem 3 — Construction: Minimal
Correct Architecture
Theorem 3(Decomposition).Under identifiable
scope keys (the entity function σis injective over
query-relevant documents), the CAR problem de-
composes into three composable steps:
(i)Anchor discovery: A={d:s(Q, d)≥
θ}— retrieve the semantically-aligned non-
controlling document.

(ii)Authority resolution: cl(A) via entity-
indexed lookup — extend to the full authority
chain.
(iii)Active-status filtering: front(cl(A)) — ex-
pose only non-superseded documents to down-
stream use.
Steps (i) and (ii) arenecessary: omitting either
makes TCA = 0on T1 instances. Step (iii) is not
necessary for TCA = 1but is necessary for LLM
answer quality (returning all of cl(A) exposes su-
perseded evidence, causing confident wrong an-
swers). The two-stage pipeline implements all three
steps and achieves:
TCA@k= Pr[anchor(d 1)∈Stage-1 top-k]
which is the exact upper bound when ∆(Q)>
0: no retrieval system can exceed Stage-1 anchor
recall.
Proof. Upper bound.Under vocabulary disjoint-
ness ( ∆(Q)>0 ),s(Q, c∗.text)≈0 for all
c∗∈F(Q) . Any system achieving TCA@ k >0
on a T1 instance must place some c∗∈F(Q) in its
top-koutput. By Theorem 2, this requires proac-
tive entity-scoped search triggered by the anchor
d1; hence d1must have been retrieved first. Since
d1is the most semantically similar document to
Q, it can only enter via Stage-1 relevance retrieval.
Therefore TCA@k≤Pr[d 1∈Stage-1 top-k 1].
Attainment.The two-stage pipeline achieves
equality: if d1∈Stage-1 top-k 1, entity-indexed
lookup (step (ii)) retrieves every c∗∈F(Q)
via shared entity scope and timestamp ordering;
step (iii) promotes only F(Q) . Thus TCA@ k=
Pr[d 1∈Stage-1 top-k 1]exactly.
Necessity of step (i).Omitting anchor discovery
means no document triggers entity-indexed lookup.
Then retrieval is purely relevance-based and TCA
= 0by Theorem 1.
Necessity of step (ii).Omitting authority reso-
lution means Stage-1 results are returned without
entity-indexed extension. Since c∗∈F(Q) scores
≈0ons(Q,·) , it falls outside any relevance-based
top-k, giving TCA= 0(Theorem 1).
Non-necessity of step (iii) for TCA.Returning
cl(A) instead of front(cl(A)) includes d1(super-
seded) alongside every c∗∈F(Q) . Since d1∈ R
has superseder c∗∈ R , NoIgnoredSuperseder
holds; since F(Q)⊆ R , AnswerCorrect holds.
TCA = 1 without step (iii). However, exposing
superseded evidence to the LLM causes confidentwrong answers (Table 12), so step (iii) is necessary
for practical downstream quality.
Corollary 2(Two-Stage Identity).For the two-
stage pipeline on T1 instances with entity-
disjoint corpora ( φ(q) = 1 ):TCA@k=
Anchor Recall@k 1. Improving Stage-1 directly in-
creases TCA; improving entity lookup when Stage-1
already recalls the anchor adds no benefit.
Corollary 3(Achievability).When φ(q) = 1
andd1∈Stage-1 top-k 1, the two-stage pipeline
achievesTCA(q) = 1.
Proof.φ(q) = 1 implies κ(q) = 1 : the entity
scope σ(q) uniquely identifies a single authority
chain. Entity-indexed lookup retrieves the unique
c∗∈F(Q) (no disambiguation needed). Step (iii)
filters to {c∗}. Frontier inclusion holds ( F(Q) =
{c∗} ⊆ R ); NoIgnoredSuperseder holds since c∗
is active (no superseder exists). Both conditions of
Theorem 4 are satisfied, giving TCA = 1.
3.5 Proposition 2 — Limit: Scope-Indexed
Algorithm Ceiling
Definition 14(Scope-Indexed Algorithm).An
algorithm Aisscope-indexedif its document-
promotion function depends only on {d.σ, d.time :
d∈ R} and corpus scope statistics, with no di-
rect access to authority rules R1, . . . , R mor the
supersession relation⇝.
Proposition 2(Scope Identifiability Upper Bound).
For any scope-indexed algorithm A(Definition 14):
TCA@k(A, q)≤φ(q)·R anchor (q)
This bound holds for both deterministic and ran-
domized A, and is tight: there exist instances
achieving equality.
Proof. Fix any scope-indexed algorithm A. Sup-
pose the anchor is retrieved ( Eaholds; probability
Ranchor ). We show Pr[Aselects correct chain|
Ea]≤φ(q) = 1/κ(q).
Information argument.When φ(q)<1 ,
there exist κ(q)≥2 distinct authority chains
C1, . . . , C κ(q)inDσ(q). By Definition 14, A’s pro-
motion function sees only {d.σ, d.time:d∈ R}
and corpus scope statistics—it has no access to ⇝.
Theκ(q) chains are, by construction, indistinguish-
able under this view: each chain contains scope
documents with the same (d.σ, d.time) marginal
statistics, and no rule-based signal distinguishes
which chain contains the frontier.

Adversarial permutation argument.Consider
any fixed algorithm A(deterministic or random-
ized). For any permutation πon[κ(q)] , define cor-
pusKπasKwith chain labels permuted: what was
chain Ciis relabeled Cπ(i), and the correct chain is
always Cπ−1(1). The view of Ais identical across
allκ(q)! such permutations (scope and timestamps
are symmetric). Algorithm Amust commit to some
chain output; by symmetry, it can be correct on
at most 1/κ(q) fraction of the κ(q)! permutations.
Therefore Pr[Acorrect|E a]≤1/κ(q) =φ(q)
on the worst-case permuted instance.
Tightness.For κ(q) = 1 (φ(q) = 1 ), the bound
is 1 and is achieved by two-stage (Corollary 3). For
κ(q)>1 , a randomized algorithm that guesses
uniformly achieves 1/κ(q) =φ(q) , matching the
bound.
Combining: TCA (A, q) = Pr[E a]·
Pr[Acorrect|E a]≤R anchor (q)·φ(q).
Corollary 4(RSSG as Optimal Scope Resolver).
The RSSG isnotscope-indexed: it accesses author-
ity rules R1, . . . , R mdirectly. When those rules
correctly identify the controlling chain ( Rfrontier =
1), RSSG achieves TCA =R anchor (q), saturat-
ing the Theorem 4 conditions. The contamina-
tion recovery ( 0.652→0.776 , Table 3) mea-
sures Rfrontier rising from near-zero (timestamp-
only, which is scope-indexed) toward the Ranchor
ceiling.
3.6 Proposition 3 — Cost: Frontier Recovery
Complexity
Definition 15(Authority-Comparison Oracle).An
authority-comparison oracle C(d1, d2)∈ {d 1⇝
d2, d2⇝d 1,⊥} answers single-pair supersession
queries inO(1).
Proposition 3(Frontier Recovery Complexity).
Lethbe the maximum supersession chain depth
within entity scope σ(q), and|Rσ|the number of
retrieved documents in that scope. To exactly re-
cover front(cl(A k(q))) , measured in oracle calls
(Definition 15):
(i)Minimum entity lookups(when φ(q) = 1 ):
|Ak(q)∩D σ(q)|lookups—one per retrieved
anchor. This is achieved by Stage-2 of the
two-stage pipeline.
(ii)Problem lower bound(when φ(q)<1 ):
Ω(|R σ|)oracle calls—every retrieved scope
document must be examined at least once
to determine chain membership. The RSSG,which has no oracle and must apply mauthor-
ity rules per pair, runs in O(|R σ|2·m) total
rule checks—an upper bound on its implemen-
tation cost, not a matching lower bound.
(iii) BFS optimum: O(h·|R σ|)oracle calls suffice
when chain structure is known; BFS from the
anchor along⇝visits each chain edge once.
Proof. (i) Each anchor d∈A k(q)∩D σ(q)trig-
gers exactly one entity index lookup to extend
cl(A) ; by Theorem 3’s minimality, no lookup can
be skipped without risking a closure violation on
some instance.
(ii) Any algorithm must examine every document
inRσto determine its chain membership; no docu-
ment can be skipped without risking a frontier error.
This yields the Ω(|R σ|)problem lower bound on
oracle calls. The RSSG has no oracle: it applies
rules R1, . . . , R mto each pair (di, dj)inRσ, giv-
ing |Rσ|
2
·m=O(|R σ|2·m) rule checks—an
upper bound on the RSSG’s implementation cost,
which exceeds the problem lower bound by a factor
of|R σ| ·m.
(iii) BFS from the anchor along ⇝visits each
chain edge once; depth hbounds the number of
hops needed to reach the frontier.
Remark 3(RSSG as Robustness Layer for Con-
taminated Corpora).On an entity-disjoint corpus,
anchor-based probing alone achieves TCA = 1.000
on all hop types T0–T3: the entity bucket contains
only gold events, so no disambiguation is needed.
The RSSG contributes when the corpus iscontam-
inated: spurious same-entity distractors appear
in the entity bucket, and the RSSG applies domain
rules R1–R10 to demote them without oracle access
(+12.4pp on T3, Table 3).
Remark 4(Low φ(q) Requires Rule-Based Dis-
ambiguation).When φ(q)<1 (equivalently
κ(q)>1 ), timestamp ordering alone cannot re-
cover front(cl(A(q))) : a distractor d⊥∈D σ(q)\
cl({d 1})witht⊥> t∗is selected by any timestamp-
only (scope-indexed) algorithm, giving TCA =
0. Disambiguation requires checking reachability
fromd 1under⇝—exactly the role of the RSSG.
3.7 Proposition 1 — Synthesis: Authority
Error Factorization
Proposition 1(Authority Error Factorization).For
a two-stage retrieval system on a CAR instance,
TCA decomposes as:
TCA(q) =R anchor (q)·φ(q)·R frontier (q)

where:
•Ranchor (q) = Pr[anchord 1 ∈
Stage-1 top-k 1](anchor retrieval probability)
•φ(q)∈(0,1](scope identifiability;
Definition 5)
•Rfrontier (q) =
Pr[front(cl(A(q)))correctly resolved|
scope identified]
Define events: Ea= “anchor d1∈
Stage-1 top-k 1”; Es= “correct authority
chain identified |Ea”;Ef= “frontier correctly
resolved |Ea∩E s”. Each factor maps to one
event: Ranchor = Pr[E a];φ(q) = Pr[E s|Ea];
Rfrontier = Pr[E f|Ea∩Es].
Proof. The factorization is the chain rule of con-
ditional probability applied to the nested event se-
quenceE a⊇(E a∩Es)⊇(E a∩Es∩Ef):
Pr[TCA = 1] = Pr[E a]·Pr[E s|Ea]·Pr[E f|Ea∩Es]
φ(q) is adeterministiccorpus property; its iden-
tification with Pr[E s|Ea]follows from Propo-
sition 2’s adversarial argument: conditional on
Ea, a scope-indexed lookup cannot distinguish
among κ(q) chains, achieving Pr[E s|Ea]≤
1/κ(q) =φ(q) with equality at the worst-case
permutation.
Lemma 1(Failure Mode Separability).The three
factors Ranchor ,φ, and Rfrontier areseparable fail-
ure modes: for each factor, there exist CAR in-
stances where that factor is <1 while the other
two are= 1.
Proof.R anchor <1,φ= 1 ,Rfrontier = 1:Let K
be entity-disjoint ( κ(q) = 1 , soφ= 1 ) andd1have
moderate relevance (ranked outside Stage-1 top-
k1with positive probability). Then Ranchor <1;
when d1is retrieved, the unique c∗is promoted
correctly ( Rfrontier = 1). Example: FinSuperQA
T1 with BM25 Stage-1 (R anchor≈0.978).
Ranchor = 1,φ <1 ,Rfrontier = 1:Let d1
always be Stage-1 rank 1, but κ(q) = 2 (two chains
share scope), with a complete RSSG rule set that
always selects the correct chain. Then Ranchor = 1,
φ= 1/2<1 , butRfrontier = 1(RSSG corrects the
ambiguity). Example: contaminated FinSuperQA
where RSSG is applied.
Ranchor = 1,φ= 1 ,Rfrontier <1:Let d1
always be Stage-1 rank 1 and κ(q) = 1 (unique
scope), but the frontier F(Q) has multiple elementsT0 T1 T2 T3
Examples 250 250 250 250
KB events (avg) 1.0 2.0 3.0 3.0
Supersession hops 0 1 2 2
Cross-provenance No No No Yes
Shared retrieval corpus: 12,250 events total
Table 1: FINSUPERQA statistics.
(chain depth ≥2) and the entity lookup has partial
coverage of the authority graph. Then Ranchor = 1,
φ= 1 , but Rfrontier <1. Example: T2/T3 hop
types in FinSuperQA with shallow entity index.
Corollaries from the factorization.
(a)Entity-disjoint corpora: φ(q) = 1 and
Rfrontier = 1, soTCA(q) =R anchor (q)—
reducing to Corollary 2.
(b)Contaminated corpora: φ(q)<1 (κ(q)>
1), so TCA falls below Ranchor by Propo-
sition 2; Rfrontier rises from near-zero
(timestamp-only) to 0.776 (RSSG), explain-
ing the+12.4pp recovery in Table 3.
(c)FDA ceiling:The corpus-level average
E[φ(q)] = 0.774 is the 77.4% ceiling—the
fraction of queries with κ(q) = 1 . A prop-
erty of the corpus scope distribution (Proposi-
tion 2), not a retrieval system limitation.
4 FINSUPERQA: A TC-MQA
Benchmark
FINSUPERQA instantiates TC-MQA in SEC Rule
204A-1 employee trading compliance (U.S. Securi-
ties and Exchange Commission, 2004). The entity
scope is σ={(emp id,ticker)} ; supersession fol-
lows rules R1–R10 (Appendix A). Ground-truth
answers are algorithmically derivable from active
events, eliminating annotation ambiguity.
Each of the 1,000 examples is assigned a unique
(employee, ticker) entity pair from a pool of 6,000.
This entity-disjoint design ensures the entity probe
retrieves each example’s gold events cleanly. The
shared corpus contains 12,250 events (2,250 gold
+ 10 distractors per example, using three distractor
types: same-employee/different-ticker, different-
employee/same-ticker, and random). A contami-
nated variant relaxes the disjoint guarantee to study
disambiguation (§6.3).

5 Systems
Baselines. BM25(Robertson and Walker, 1994)
(k1= 1.5 ,b= 0.75 ) andTF-IDF(cosine) are the
lexical baselines.Dense (MiniLM)uses sentence-
transformer embeddings (Reimers and Gurevych,
2019).Oracleranks gold documents first (upper
bound). Full hyperparameters and graph construc-
tion details are in Appendix C.
DENSERERANK: dense retrieval with supers-
ession re-ranking. Promotes superseders of top-20
events that are superseded in the gold KB; restricted
to re-ranking what dense retrieval already surfaced.
This is the strongest re-ranking baseline: it has
access to ground-truth supersession edges.
TWOSTAGE(anchor probe): dense retrieval
+ entity-indexed probing. Extracts entity iden-
tifiers from the query; for every entity-matched
event in the initial top-20, looks up the full-corpus
entity index for later-timestamped events in the
same scope and promotes them. This implements
the Decomposition Theorem’s three stages: an-
chor discovery (dense top-20), authority resolu-
tion ( cl(A) via entity index), active-status deter-
mination ( front(cl(A)) by timestamp). Achieves
TCA= 1.000on entity-disjoint corpora.
TWOSTAGE+RSSG(ours): TWOSTAGEaug-
mented with aRetrieved-Set Supersession Graph
(RSSG). After the entity probe, applies domain
rules R1–R10 (Appendix A) to all retrieved entity-
scoped events—without gold KB access—to deter-
mine which events are active under the superses-
sion relation. This implements front(cl(A)) cor-
rectly when κ(q)>1 : the RSSG identifies which
retrieved documents are in the true cl(A) vs. spuri-
ous same-scope distractors, enabling accurate fron-
tier recovery (+12.4pp on T3).
6 Experiments
6.1 Setup
All systems retrieve from the shared 12,250-event
corpus with k= 5 ,ndistractors = 10 . Metrics: TCA,
Acc, R@5, ProvRec.
6.2 Main Results
Theorem 1 confirmed.BM25, TF-IDF (cosine),
and Dense (MiniLM) all achieveTCA = 0.000
on T1. The superseding blackout event is ranked
≈450 th on average. Recall@5 ≈0.45 –0.62
(the pre-clearance approval is retrieved), but TCA
= 0 because the superseder is not in the top-5.The failure of MiniLM—a 22M-parameter seman-
tic encoder that achieves R@5 = 0.450 overall—
confirms that the gap is not a TF-IDF artifact: even
state-of-the-art dense retrieval cannot surface a
vocabulary-disjoint superseder in top-5.
DENSERERANKimproves T2 but not T1 or T3.
DENSERERANKachieves TCA = 0.980 on T2 by
promoting superseders of events in the initial top-
20. It achieves TCA = 0.000 on T1 (no superseder
in top-20, no promotion possible) and TCA = 0.016
on T3 for the same reason.
TWOSTAGEand TWOSTAGE+RSSG achieve
TCA = 1.000 on all hop types.TCA = 1.000
follows directly from the entity-disjoint corpus de-
sign: each (emp, ticker) pair appears in exactly
one example, so the entity bucket retrieved by the
two-stage probe contains only the example’s gold
events, making TCA = 1 guaranteed by construc-
tion. The result confirms that the pipeline cor-
rectly implements the CAR retrieval target; the
benchmark’s purpose is to isolate the vocabulary-
mismatch failure of baselines (T1 TCA = 0.000),
not to demonstrate that TCA = 1.000 is difficult to
achieve in the entity-disjoint setting.
Accuracy vs. TCA gap.TF-IDF (cosine)
achieves Acc ≈0.491 but TCA = 0.012 on T2.
This gap reflects “right answer, wrong reason”: the
system retrieves the final-state event but ignores
the intermediate chain. TCA correctly flags this as
a compliance failure.
ProvRec vs. TCA.TWOSTAGE+RSSG achieves
TCA = 1.000 with ProvRec = 0.708. The system
derives correct answers with sound supersession
reasoning, but does not always retrieve the full
provenance chain (e.g., the original pre-clearance
for audit records). These are separate objectives:
TCA captures compliance soundness; ProvRec cap-
tures archival completeness.
6.3 Entity-Scope Contamination and the
RSSG
When the corpus contains distractors sharing the
query’s entity scope ( φ(q)<1 , equivalently
κ(q)>1 at the entity bucket level), Proposi-
tion 1 predicts TCA degrades through the φ(q) and
Rfrontier terms; Remark 4 shows timestamp order-
ing fails. Table 3 confirms this on a contaminated
corpus where entity-disjoint distractor construction
is relaxed.

Overall TCA by Hop Type
System R@5 TCA Acc ProvRec T0 T1 T2 T3
BM25 0.607 0.305 0.503 0.607 1.0000.0000.216 0.004
TF-IDF (cosine) 0.621 0.254 0.491 0.621 1.0000.0000.012 0.004
Dense (MiniLM) 0.450 0.214 0.247 0.450 0.8200.0000.036 0.000
Oracle 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
DENSERERANK0.333 0.499 0.499 0.333 1.000 0.000 0.980 0.016
TWOSTAGE(anchor only) 0.7081.000 1.0000.708 1.000 1.000 1.0001.000
TWOSTAGE+RSSG (ours)0.708 1.000 1.0000.708 1.0001.000 1.000 1.000
Table 2: Main results on FINSUPERQA ( k= 5 , 12,250-event corpus). TCA = 1.000 for TWOSTAGEand
TWOSTAGE+RSSG on T0–T3 is a guaranteed consequence of the entity-disjoint corpus design (each (emp, ticker)
pair appears in exactly one example, so the entity bucket always contains clean gold events)—not an empirical
surprise. The benchmark is designed to isolate the vocabulary-mismatch failure: BM25, TF-IDF, and Dense all
achieve TCA = 0.000 on T1 despite Recall@ k >0 , confirming Theorem 1 for both lexical and semantic retrievers.
The contamination ablation (Table 3) evaluates the harder setting where the entity-disjoint guarantee is relaxed.
System TCA T3∆T3
TWOSTAGE(anchor only) 0.911 0.652 —
TWOSTAGE+RSSG (anchor + RSSG) 0.942 0.776+12.4pp
Table 3: Results on the contaminated corpus (naive
distractor sampling with no entity-disjoint guar-
antee; enforce entity disjoint=False ).
The RSSG provides +12.4 pp on T3 by applying
domain rules to disambiguate spurious same-
entity distractors without oracle access. Full results in
data/contaminated corpus results.json .
On the contaminated corpus, the RSSG recovers
+12.4 pp on T3 and +3.1 pp overall TCA. The
RSSG is the robustness layer for production set-
tings where entity-scope disjointness cannot be
guaranteed.
6.4 Vocabulary Gap in Real-World Data
Theorem 1 requires ∆(q)>0 in natural cor-
pora. We verify this on 120 real security advisories
(GHSA + NVD): for each, we compute Jaccard
and BM25 between the user query and (a) the CVE
disclosure, (b) the patch note.
Disclosures are 1.51× more BM25-relevant to
user queries than patch notes, and the gap is
positive in 96.7% of real advisory pairs. Non-
stopword Jaccard shows a 2.6× difference (0.037
vs. 0.014). These numbers validate Theorem 1’s
construction: the vocabulary gap we engineered
in CVEPATCHQA mirrors the distribution of real
security advisories.Metric Query↔Disclosure Query↔Patch
Jaccard 0.085 0.060
Jaccard (no stopwords) 0.037 0.014
% positive gap (Jaccard) 96.7% —
BM25 score 9.73 6.45
% positive gap (BM25) 97.5% —
Table 4: V ocabulary gap between queries and (a) vul-
nerability disclosures vs. (b) patch release notes on 120
real security advisories (GHSA + NVD). Disclosures
are1.51× more BM25-relevant to the query than patch
notes; 97.5% of pairs show positive BM25 gap, 96.7%
positive Jaccard gap. The vocabulary gap is a property
of natural corpora, not an artifact of our synthetic con-
struction.
6.5 End-to-End Retrieval on Real GHSA
Advisories
The vocabulary gap analysis confirms the gapexists
in real data, but does not directly measure whether
retrieval systemsfailbecause of it. We now run
end-to-end retrieval on a real corpus of GitHub
Security Advisories.
Dataset.We collect 159 real (CVE-disclosure,
GitHub-release-note) pairs from GHSA spanning
critical, high, medium, and low severity advisories,
where each advisory (i) is reviewed by the GHSA
team and (ii) has a GitHub releases/tag/
URL we can fetch via the GitHub REST API. Key
structural property:zero of 159 release notes con-
tain the paired CVE identifier( 0/159 ,0%), con-
firming the vocabulary gap is structural across all
severity levels. The retrieval corpus consists of
all 159 disclosures, all 159 real release notes, and
2,000 synthetic CVE-disclosure distractors (2,318
documents total), simulating a realistic security ad-

visory database at scale.
Systems.We evaluate four retrieval systems and
one entity-indexed system.BM25uses a standard
term-frequency retriever.Dense (MiniLM)uses
sentence-transformers/all-MiniLM-L6-v2 (22M
params).Dense (E5-large)uses intfloat/e5-large-
v2 (335M params, MTEB top-10).HyDE(Gao
et al., 2023) prompts Llama 3.3 70B to generate
a hypothetical patch release note for each query,
then embeds the hypothetical with MiniLM and
retrieves by cosine similarity—the leading LLM-
augmented approach for vocabulary mismatch.
Regex-NER Probeextracts the package name and
CVE identifier from the query with two regular
expressions, then performs entity-indexed lookup
with no oracle labels. It matches the oracle entity-
probe upper bound exactly (TCA = 1.000) be-
cause the structured query template admits 100%-
accurate regex extraction. All systems rank the full
2,318-document corpus. TCA@ k= 1 iff the real
patch release note appears in the top-kresults.
System TCA@5 TCA@10 TCA@20 Mean rank Median rank
BM25 0.641 0.730 0.761 144.4 3
Dense (MiniLM) 0.132 0.157 0.214 1,234 1,847
Dense (E5-large) 0.069 0.120 0.170 1,113 1,246
HyDE (Llama 3.3 70B) 0.151 0.170 0.170 1,283 1,577
Regex-NER Probe1.000 1.000 1.0002.0 2
Table 5: End-to-end retrieval on 159 real GHSA ad-
visories (2,318-document corpus). 95% Wilson CIs:
Dense (E5-large) TCA@5 = 0.069 [0.038, 0.119];
Dense (MiniLM) TCA@5 = 0.132 [0.087, 0.194];
HyDE TCA@5 = 0.151 [0.103, 0.216]; Regex-NER
Probe TCA@5 = 1.000 [0.977, 1.000]. HyDE’s
LLM includes the CVE identifier in 100% of gen-
erated hypotheticals, making each hypothetical se-
mantically closer to disclosures than to release notes;
adding 70B parameters changes nothing structurally.
The Regex-NER Probe matches the oracle upper
bound (TCA = 1.000): entity-indexed retrieval is both
the correct fix and immediately practical. Artifacts:
data/ghsa realworld results.json ; HyDE:
data/ghsa hyde results.json.
Results.All four retrieval systems fail; only
the entity-indexed probe succeeds. Dense (E5-
large, 335M params) achieves TCA@5 = 0.069—
worsethan MiniLM (0.132)—because superior se-
mantic matching more strongly associates CVE-
framed queries with CVE-framed disclosures, am-
plifying the gap. HyDE with Llama 3.3 70B
achieves TCA@5 = 0.151, statistically comparable
to MiniLM (0.132): the LLM generates a patch
note that includes the CVE identifier in every sin-
gle generated hypothetical (159/159), since thequery explicitly names the CVE. The hypotheti-
cal therefore embeds closer to disclosures than to
real release notes, and retrieval fails for the same
structural reason as naive dense. Adding 70B pa-
rameters of LLM reasoning to the retrieval pipeline
cannot bridge a vocabulary gap that originates in
the query itself.
The gap between HyDE (0.151) and Regex-
NER Probe (1.000) is not a matter of model scale
or generation quality: it is an architectural gap.
Relevance-based retrieval—even when “relevance”
is computed with respect to an LLM-generated
ideal document—cannot recover the correct patch
note because the ideal document necessarily shares
vocabulary with the wrong class of documents (dis-
closures). Only entity-indexed structured retrieval
closes the gap.
The critical finding is in the Regex-NER
Probe row: a system that extracts “ nginx ” and
“CVE-2023-44487 ” from the query via two reg-
ular expressions, then looks up documents indexed
by those entities, achieves TCA = 1.000 without
any oracle supervision. This demonstrates that
entity-indexed structured retrieval is boththe cor-
rect fixandimmediately practical—it requires no
fine-tuned embeddings, no cross-encoder rerank-
ing, and no oracle labels; only the architectural
insight that supersession retrieval must be entity-
scoped.
This confirms Theorem 1 on real production
data: dense retrieval fails in 87% of real T1 se-
curity queries because it cannot bridge the vocab-
ulary gap between a user’s vulnerability question
and a vendor’s release announcement. In the lan-
guage of Definition 4, these are high- ∆(q) queries:
the anchor (CVE disclosure) scores far higher on
s(q,·) than any element of front(cl(A(q))) (the
patch note). Adding more parameters to a rele-
vance scorer cannot recover a signal that is archi-
tecturally absent; ∆(q) is a structural property of
the query-corpus pair, not a tuning parameter.
Standard benchmarks predict TCA in the
wrong direction.E5-large (intfloat/e5-large-v2,
335M params) ranks in the MTEB top-10 for
retrieval and is the dominant choice in produc-
tion RAG pipelines. On the GHSA corpus it
achieves TCA@5 =0.069—worsethan the 22M
MiniLM baseline (0.132) and 9.3 ×worse than
BM25 (0.641). MTEB rank and parameter count
areinverselycorrelated with TCA on this task: a
practitioner optimizing for MTEB would select the

model that performs worst here. The TC-MQA
metric surfaces a failure mode that MTEB, BEIR,
and standard recall@ kbenchmarks cannot detect,
because those benchmarks reward semantic simi-
larity between query and retrieved document—the
exact property that causes retrieval failure when the
target document (patch note) is vocabulary-disjoint
from the query.
6.6 Two-Stage Pipeline for Free-Form Queries
The Regex-NER Probe achieves TCA = 1.000 on
the structured query template “Is {product }still
affected by {CVE-ID }?”, but this template encodes
entity identity explicitly. In practice, developers
ask free-form questions—“Has the issue with path
parameters being substituted into URLs without
encoding been fixed?”—that describe vulnerability
behavior without the CVE identifier. The entity
probe cannot be applied to such queries.
Free-form queries.We generate one free-form
question per GHSA pair using Llama 3.3 70B
(Nebius). Each prompt provides the disclosure text
and instructs the model to write a natural-language
question (maximum 50 words) describing the vul-
nerability behavior, without including the CVE
identifier and without using the structured template.
All 159 generated queries were verified to contain
no CVE-ID string (regexCVE-\d{4}-\d+).
Key observation.BM25 on a free-form vulner-
ability question finds the CVEdisclosurereliably
(recall@1 = 91.8%, @5 = 97.5%) even though it
cannot find thepatch note(TCA@5 = 0.138). Both
the free-form question and the disclosure describe
the same vulnerability in similar vocabulary; the
gap is specifically between the query vocabulary
and thepatch-notevocabulary, not between the
query and the disclosure.
Two-stage pipeline.We exploit this asymmetry
via a two-stage architecture:
1.Stage 1(disclosure retrieval): BM25 retrieves
the top- k1documents from the full corpus
using the free-form query.
2.Stage 2(entity-indexed patch lookup): For
each Stage-1 result that is a disclosure, use
its CVE-ID metadata as a key into the entity
index to retrieve the paired patch note.
The final ranking places Stage-2 patch candidates
first (in Stage-1 order), followed by remaining doc-
uments.System Disc@5 TCA@5 TCA@10 Mean rank
BM25 0.975 0.138 0.214 609
Dense (MiniLM) 0.937 0.270 0.365 764
Dense (E5-large) 0.962 0.252 0.321 799
HyDE (Llama 3.3 70B) — 0.151 — —
Two-Stage (k 1=5) 0.9750.975— 43.2
Table 6: Free-form query results on 159 real GHSA
pairs (2,318-document corpus). Disc@5 = fraction
of free-form queries for which BM25 retrieves the
correct CVE disclosure in top-5. TCA@ kof the
two-stage pipeline equals exactly Stage-1 Disc@ k,
since the j-th retrieved disclosure’s patch note is
placed at rank jin the final ranking. Artifact:
data/ghsa twostage results.json.
Results.Table 6 shows the full results. The two-
stage pipeline achieves TCA@5 =0.975 on free-
form queries—a 3.6× improvement over the best
all-retrieval baseline (MiniLM, 0.270) and within
2.5 pp of the Regex-NER oracle (structured queries
only). The equality TCA@ k= Disc@ kholds ex-
actly: when the disclosure is the j-th document in
BM25 top- k, the paired patch note is placed at rank
jin the final ranking; TCA@5 is thus determined
entirely by whether BM25 retrieves the disclosure
in its top-5.
Increasing k1from 5 to 20 does not improve
TCA@5 (0.9748 for all values): the 4 remain-
ing failures all arise from disclosures at BM25
rank≥9—even when promoted to positions 9–
18 in the final ranking, they fall outside the top-5.
Three failures involve queries that are too generic
to distinguish their specific package (e.g. “Has the
issue with improper validation been fixed?”), and
one involves a niche npm package with no BM25
vocabulary overlap.
HyDE on free-form queries.HyDE achieves
TCA@5 = 0.151 on free-form queries—identical
to its performance on structured queries (0.151).
Without the CVE identifier in the query, the LLM
generates a patch note from the vulnerability de-
scription, which inherits the original vulnerability
vocabulary rather than the patch-note vocabulary.
LLM generation cannot predict the release-note
vocabulary of an unknown vendor.
Stage-1 ablation.BM25 achieves higher dis-
closure recall at every cutoff (Disc@1 = 91.8%,
Disc@5 = 97.5%) than dense encoders, validat-
ing BM25 as the preferred Stage-1 retriever. Ex-
act term matching outperforms embedding simi-
larity for vocabulary-aligned query-to-disclosure
retrieval, because both the free-form question and

the CVE disclosure describe the vulnerability in
domain-specific technical terms.
Cross-encoder reranker.A cross-encoder
reranker (BM25-top20 ∪Dense-top20 →cross-
encoder/ms-marco-MiniLM-L-6-v2 → top-5)
achieves TCA@5 = 0.233 on the 159 free-form
GHSA queries—below any dense baseline and
4.2× worse than Two-Stage ( 0.975 ). The failure
is architectural, not a reranking quality issue:
the union pool of BM25-top20 and Dense-top20
contains the gold patch note only when at
least one retriever already recovers it in top-20.
Since both BM25 (TCA@5 = 0.138 ) and Dense
(TCA@5 = 0.270 ) fail to retrieve the patch note for
most queries, the cross-encoder never scores it. Ap-
plying a stronger reranker to the wrong candidate
pool cannot close an architectural gap. Artifact:
data/falsification baselines results.json .
CAR versus production lookup.Production
tools such as OSV .dev are often proposed as
alternatives to CAR-style retrieval. We evaluate
the distinction on the same 159-pair GHSA
corpus under two query modes. InMode 1
(known-anchor: query = full CVE disclosure
text), BM25 achieves TCA@5 = 0.164 and
Two-Stage achieves TCA@5 = 1.000 . OSV .dev
was invoked by applying a CVE-ID regex to
each disclosure text; only 3.77% of disclo-
sures (6/159) contained a regex-extractable
CVE string, yielding TCA@5 = 0.006 (artifact:
data/falsification carvslookup.json ,
fields mode1 applicable=0.0377 ,
mode1 tca5=0.0063 ). When OSV .dev
is queried directly with ground-truth CVE
IDs (a stronger, oracle-assisted setting), it
indexes 14.5% of GHSA CVEs (23/159),
achieving TCA@5 = 0.088 overall and
0.609 on the 23 indexed pairs (artifact:
data/falsification baselines results.json ).
Both measurements confirm the same conclusion:
OSV .dev’s coverage of GHSA advisories is limited
regardless of how the CVE identifier is sourced. In
Mode 2(free-form NL: query = Llama-generated
question with no CVE identifier), OSV .dev is0%
applicable—it cannot be invoked without a CVE
identifier—and a regex entity probe is likewise 0%
applicable. Two-Stage achieves TCA@5 = 0.975
without any entity identifier in the query.
The gap between modes is categorical: produc-
tion tools arenot applicablein Mode 2, not merely
degraded. They solve a different problem (known-anchor →check supersession) from CAR (free-
form NL →retrieve controlling authority). The
Two-Stage architecture is the only system that op-
erates in both modes because BM25 reconstructs
the anchor from query vocabulary without requir-
ing the CVE identifier.
Scale amplifies error.Table 7 shows TCA@5
and disclosure recall@5 across two encoder fami-
lies at three parameter scales. Three findings stand
out:
1.Scale does not help: no dense model ex-
ceeds TCA@5 = 0.283, identical to the 22M
MiniLM baseline. A 335M model is no better
than a 22M model.
2.E5 family: monotone decline: E5-small
(33M,0.264 )>E5-base ( 109M,0.239 )>E5-
large ( 335M,0.214 )—larger encoders match
the query-to-disclosure semantic alignment
more precisely, ranking the disclosure above
the patch note with greater confidence.
3.BGE family: non-monotone, same ceil-
ing: BGE-small ( 33M) achieves 0.252 ,below
MiniLM; BGE-base and BGE-large recover
to0.283 —the same ceiling as a 22M model.
Unlike E5, larger BGE does not worsen TCA,
but it does not escape the 0.283 ceiling either.
Both failure modes converge on the same con-
clusion: scale is irrelevant when the architec-
ture is wrong.
Model Params Family Disc@5 TCA@5
MiniLM-L6-v2 22M MiniLM 0.962 0.283
E5-small-v2 33M E5 0.987 0.264
E5-base-v2 109M E5 0.950 0.239
E5-large-v2 335M E5 0.9560.214
BGE-small 33M BGE 0.981 0.252
BGE-base 109M BGE 0.975 0.283
BGE-large 335M BGE 0.987 0.283
Two-Stage (BM25 + Entity Index)0.9750.975
Table 7: Scale amplifies error in the E5 family (TCA@5
monotonically decreases with parameters), while BGE
saturates at the same ceiling as MiniLM. No dense
model exceeds TCA@5 = 0.283 regardless of scale. The
two-stage pipeline achieves 0.975 withno trained pa-
rameters, a 3.4× improvement over any dense baseline.
Artifact:data/ghsa scale results.json.
6.7 Two-Stage Pipeline and Domain
Adaptation
Theorem 3 implies a clean architecture: (1)
retrieve the semantically-aligned anchor doc-

ument in Stage 1; (2) follow entity-indexed
supersession to the controlling authority in
Stage 2. Adapting to a new domain requires
specifying three attributes— anchor type ,
superseding type ,scope keys . The three
released domain adapters and evaluation code are
described in Appendix D.
Cross-domain two-stage evaluation.Table 8
verifies this architecture holds across three domain
constructions with different query styles.
Domain Query style Stage-1 Rec@5 BM25 Two-Stage
Security (real,n=159) Generic free-form 0.975 0.1380.975
CVE (synth,n=250) Generic free-form 0.860 0.0000.860
CVE (synth,n=250) + product hint 1.000†0.0000.900
Legal (synth,n=250) Generic free-form 0.312 0.000 0.312
Legal (synth,n=250) + party-name hint 1.000 0.0001.000
Table 8: Two-stage pipeline across three domain
constructions and two query styles. BM25 di-
rect always fails (TCA@5 ≈0). Two-Stage
TCA@5 equals Stage-1 anchor Recall@5 when
entity scope keys are unambiguous (Corollary 2).
†CVE+product-hint exception: product names are
shared across CVEs, so product hints cause false anchor
retrievals; free-form queries avoid this. Artifacts:
data/cve synthetic twostage results.json ,
data/legal twostage results.json.
Stage-1 recall is the limiting factor.TCA@ k
of the two-stage pipeline equals Stage-1 anchor
recall at kexactly (Corollary 2 below). Stage-1
recall depends on the vocabulary discriminative-
ness of the free-form query relative to the anchor
corpus: CVE disclosures contain domain-specific
attack vocabulary (“buffer overflow”, “path traver-
sal”) that free-form vulnerability questions echo,
yielding 86–97.5% Stage-1 recall. Synthetic le-
gal rulings covering ≈75 distinct doctrines yield
only 31.2%; adding party surnames—one word per
party, no citation—restores recall to 100% because
surnames uniquely identify each ruling.
Corollary 2 states this identity precisely. The
practical implication:improving Stage-1 anchor
retrieval is equivalent to improving final TCA.
Any method that better retrieves anchor documents
(denser retrieval, entity hints, query expansion) di-
rectly transfers to TCA improvement with no archi-
tectural changes.
6.8 Synthetic Cross-Domain Validation
To confirm Theorem 1 is not a compliance arte-
fact, we build two additional controlled bench-
marks: CVEPATCHQA (500 examples, security)
and LEGALPRECEDENTQA (500 examples, legalprecedent), each with a T0/T1 split matching the
compliance construction. Table 9 shows BM25 and
TF-IDF achieve TCA = 0.000 on T1 in all three
synthetic domains; TWOSTAGE+RSSG achieves
TCA = 1.000 . Theorem 1 is a structural property,
not a compliance artefact.
System Finance T1 CVE T1 Legal T1
BM25 / TF-IDF 0.000 0.000 0.000
TWOSTAGE1.000 1.000 1.000
TWOSTAGE+RSSG1.000 1.000 1.000
Table 9: T1 TCA across three synthetic domains. All
relevance-based retrievers fail (TCA = 0.000 ); the two-
stage architecture solves T1 exactly in all three.
The legal vocabulary gap is structural: of 82
CourtListener overruling snippets, 88% contain
overruling vocabulary (“overrule”, “abrogate”);
0% contain query vocabulary (“good law”, “con-
trolling precedent”)—a court writes “we overrule
X,” never “Xis no longer good law” (artifact:
data/courtlistener vocab gap.json).
6.9 End-to-End Retrieval on Public SCOTUS
Overruling Pairs
We build a second real end-to-end benchmark from
122 SCOTUS overruling pairs (seeded from Cor-
nell LII, expanded via Wikipedia’s overruled deci-
sions list; 244 opinions total). Queries: “IsPlessy
v. Fergusonstill good law and controlling prece-
dent?” Each opinion is prefixed with its case name
and citation. Theorem 3’s identity applies directly
(unique case slugs): TCA@5 equals Stage-1 an-
chor recall@5.
Results.BM25 direct is competitive (0.893
[0.826, 0.936]): overruling opinions explicitly cite
the case they overrule, giving lexical retrieval a
direct signal unavailable in GHSA or FDA. Dense
direct fails (0.172 [0.115, 0.249]): semantic sim-
ilarity actively prefers theoriginalopinion over
the overruling one. Two-Stage with Dense an-
chor achieves 0.926 [0.865, 0.961]—a+3.3 pp
improvement over BM25 direct; the 95% CIs
overlap, so this gap is not statistically distinguish-
able at n = 122 . The headline comparison to
Dense ( +75.4 pp) reflects the structural failure of
semantic retrieval, not the marginal gain from the
two-stage architecture in this domain. SCOTUS is
the domain where lexical retrieval partially works
(explicit citation vocabulary), and correspondingly
where the two-stage architecture’s marginal im-

System Anchor@5 TCA@5 95% CI
BM25 direct — 0.893 [0.826, 0.936]
Dense (MiniLM) direct — 0.172 [0.115, 0.249]
Two-Stage (BM25 anchor) 0.836 0.836 [0.760, 0.891]
Two-Stage (Dense anchor) 0.9260.926[0.865, 0.961]
Table 10: End-to-end retrieval on 122 public SCOTUS
overruling pairs (244-opinion corpus). BM25 direct
achieves 0.893 [0.826, 0.936] because overruling opin-
ions explicitly cite the case they overrule; Dense direct
fails (0.172 [0.115, 0.249]) because semantic similarity
prefers the original opinion. Two-Stage (Dense anchor)
achieves 0.926 [0.865, 0.961]—the CIs of BM25 and
Two-Stage overlap (n=122 is underpowered to distin-
guish the +3.3 pp gap). SCOTUS is the easiest domain
for BM25: explicit citation vocabulary bridges the
lexical gap. The hard structural failure is Dense (0.172),
where semantic similarity actively misdirects. Artifacts:
data/legal scotus expanded pairs.json ,
data/legal scotus expanded results.json .
provement is smallest. The GHSA and FDA re-
sults, where both BM25 and Dense collapse and
Two-Stage achieves >0.774 , provide the cleaner
evidence that the architecture is necessary.
6.10 End-to-End Retrieval on FDA Drug
Recall Enforcement Actions
A third confirmation in medical/regulatory: FDA
drug recall enforcement actions formally supersede
approved drug labels with completely disjoint vo-
cabulary (clinical vs. administrative). We collect
500 real (approved-label, enforcement-notice) pairs
via OpenFDA APIs (3,000-document corpus with
2,000 distractors). Queries use indication text with-
out the drug name: “Is the drug indicated for [in-
dication] still approved?”—therapeutic vocabulary
only. Only 8.2% of recall notices contain any clini-
cal vocabulary term; 100% of labels do. This is the
largest∆(q)of any benchmark in this paper.
Results.Both direct retrieval systems collapse:
BM25 achieves TCA@5 = 0.004 (2 out of 500
pairs) and Dense (MiniLM) achieves TCA@5 =
0.064 (32 out of 500). This is the most dramatic
vocabulary gap of any benchmark in this paper. Re-
call notices use administrative language about lots,
contamination, and sterility; a query about a drug’s
therapeutic indication shares zero meaningful vo-
cabulary with a typical recall notice. Relevance-
based retrieval has no signal to rank the recall no-
tice above 499 competing documents—it is archi-
tecturally blind to the supersession relation.
The Two-Stage architecture achieves TCA@5 =
0.774 , a12×improvement over dense direct re-System Stage-1 @5 TCA@5 95% CI
BM25 direct — 0.004 [0.001, 0.014]
Dense (MiniLM) direct — 0.064 [0.046, 0.089]
Two-Stage (BM25 anchor) 0.7740.774[0.735, 0.808]
Table 11: End-to-End retrieval on 500 FDA drug re-
call pairs (3,000-document corpus). BM25 and Dense
(MiniLM) collapse (TCA@5 ≤0.064 , 95% CI up-
per bound ≤0.089 ): therapeutic indication vocabu-
lary has near-zero overlap with recall enforcement vo-
cabulary. Two-Stage TCA@5 equals Stage-1 anchor
recall@5 exactly (Theorem 3); the 95% CI [0.735,
0.808] reflects the indication ambiguity at Stage-1: mul-
tiple drugs share therapeutic categories, and Stage-1
errors cannot be remedied by entity-indexed Stage-
2. Artifact: data/fda recall pairs.json ,
data/fda recall results.json.
trieval. Theorem 3 confirms this equals Stage-1
anchor recall@5 exactly: when BM25 success-
fully retrieves the drug’s approved label (the se-
mantically aligned anchor) in Stage-1, the entity-
indexed Stage-2 lookup invariably finds the cor-
rect enforcement notice. The 77.4% ceiling is ex-
plained by scope ambiguity κ(q) at Stage-1: mul-
tiple approved drugs share therapeutic categories
(hypertension, type 2 diabetes, infection), so the
same indication vocabulary maps to several pos-
sible anchors—BM25 occasionally retrieves the
wrong drug’s label, and entity lookup then finds the
wrong recall notice. This is not a failure of Stage-2;
it is the fundamental limit of any system where
κ(q)>1 in the anchor retrieval stage. Closing it
requires indication-specific entity linking beyond
term matching.
6.11 Downstream LLM Answer Quality:
Dense vs. Two-Stage
TCA measures retrieval correctness, but the ulti-
mate risk is downstream: a language model given
superseded evidence will produce confident, incor-
rect answers that a practitioner may act on. We
test this directly by running GPT-4o-mini on top of
Dense and Two-Stage retrievers on the 159 GHSA
pairs.
Setup.Each query is issued against a 2,318-
document corpus. Dense retrieval returns MiniLM
top-5; Two-Stage returns BM25 Stage-1 top-5,
entity-indexed to promote the patch note. GPT-4o-
mini is prompted to answer:“Is the vulnerability
patched? If so, what version fixes it?”using only
the retrieved documents. The gold answer for every
pair isyes, patched(all pairs have a real patch note).

We classify each response as:correct(explicitly
states patched and names the fixing version),con-
fident wrong(explicitly asserts the vulnerability
is not patched), oruncertain(hedged or no patch
status claimed).
Results.Table 12 shows the outcome. Dense re-
trieval correctly identifies the patch for only 61.0%
of queries; for39.0%of queries (95% CI: 31.3–
47.1%), GPT-4o-mini explicitly asserts the vul-
nerability remains unpatched—a confident, action-
able wrong answer despite every gold answer be-
ingpatched. Two-stage retrieval raises the correct
patch-identification rate to 82.4% ( +21.4 pp) and
cuts the confident-wrong rate to 16.4%. The gap
is explained by retrieval: Dense retrieves the patch
note for only 28.3% of queries; when it misses
(71.7% of queries), GPT-4o-mini produces a confi-
dent “not patched” assertion at a 54% rate. When
Two-Stage correctly places the patch note in the
context window (97.5% of queries), the LLM iden-
tifies the patch at an 82% rate.
Condition Patch Recall@5 Says Patched Conf. Wrong Uncertain
Dense (MiniLM) 28.3% 61.0%39.0%0%
Two-Stage (BM25) 97.5%82.4%16.4% 1%
Table 12: GPT-4o-mini answer quality on 159 GHSA
pairs. All gold answers arepatched. “Says Patched” =
LLM asserts the vulnerability is fixed; “Conf. Wrong” =
LLM explicitly asserts it isnotpatched. Dense retrieval
causes confident wrong assertions for 39% of queries.
Two-stage retrieval cuts this to 16.4% ( −22.6 pp). Arti-
fact:data/llm downstream results.json.
This result confirms that retrieval architecture
directly determines downstream LLM reliability,
not just benchmark TCA metrics. A practitioner
relying on Dense RAG would receive explicit “this
vulnerability is unpatched” assertions for 39% of
queries, potentially deferring security patches that
have been available for months.
6.12 Real-World Benchmark Summary:
Three Authority-Governed Domains
Table 13 unifies the three real-world benchmarks.
The failure pattern of dense retrieval is consistent
across security, legal, and medical/regulatory do-
mains, while the two-stage architecture consistently
recovers. In the legal domain, BM25 is partially
competitive (see §6.9), but dense retrieval fails even
there ( 0.172 ), confirming Theorem 1 as a domain-
general law for semantic encoders, not a compli-
ance artefact.Domain n BM25 Dense Two-Stage vs. Dense
Security (GHSA) 159 0.138 0.2700.975+70.5pp
Legal (SCOTUS LII) 122 0.893 0.1720.926+75.4pp
Medical (FDA Recall) 500 0.004 0.0640.774+71.0pp
Table 13: Real-world benchmark summary (TCA@5
with 95% Wilson CIs). “Two-Stage” uses BM25 an-
chor in security and medical, Dense anchor in legal
(best per-domain). Two-Stage 95% CIs: GHSA [0.937,
0.990], SCOTUS [0.865, 0.961], FDA [0.735, 0.808].
Legal BM25 is high (0.893 [0.826, 0.936]); BM25 and
Two-Stage CIs overlap at n = 122 (see §6.9). GHSA
and FDA provide the cleaner evidence: both BM25 and
Dense collapse ( ≤0.270 ), while Two-Stage recovers to
≥0.774 with non-overlapping CIs against both base-
lines. Medical shows the most severe collapse: therapeu-
tic vocabulary shares no terms with recall enforcement
vocabulary (BM25= 0.004[0.001, 0.014]).
7 Related Work
Multi-hop QA.HotpotQA (Yang et al., 2018),
MuSiQue (Trivedi et al., 2022), and 2WikiMulti-
Hop (Ho et al., 2020) require multi-step retrieval.
IRCoT (Trivedi et al., 2023), PropRAG (Tang et al.,
2025), and HippoRAG 2 (Guti ´errez et al., 2025) im-
prove multi-hop retrieval through chain-of-thought
interleaving, graph propagation, and associative
memory. These systems assume all required doc-
uments are retrievable by relevance. TC-MQA
breaks this assumption: the superseding document
may be semantically orthogonal to the query (The-
orem 1), making relevance improvement insuffi-
cient.
Temporal RAG.T-GRAG (Li et al., 2025) re-
solves temporal redundancy in graph RAG. Ver-
sionRAG (Huwiler et al., 2025) handles version-
aware retrieval for evolving documents. Tem-
pRALM (Gade and Jetcheva, 2024) incorporates
timestamps into retrieval scoring. LedgerRAG
(Wang et al., 2025) provides governance-aware
agentic retrieval. These works addressrecency
preference—favoring newer documents—whereas
TC-MQA requires detecting that a specific later
documentformally voidsan earlier one under do-
main rules. The difference is structural: recency
preference is a soft ranking adjustment; superses-
sion is a hard validity constraint. Importantly, CAR
is not equivalent to time-aware retrieval: the au-
thority relation ⇝is not timestamp ordering—a
superseding document can predate its superseded
counterpart when effective dates differ from publi-
cation dates (common in regulatory and legal cor-
pora), so timestamp-conditioned ranking cannot

recoverfront(cl(A k(q)))in general.
Legal and regulatory temporal reasoning.
Chalkidis et al. (Chalkidis et al., 2021) provide
the earliest NLP recognition that temporal filtering
is essential for regulatory compliance IR, showing
that BM25 with chronological constraints outper-
forms pure neural re-rankers on EU/UK directive
retrieval—an empirical precursor to Theorem 1’s
mismatch result. Zhang et al. (Zhang et al., 2025)
find that LLMs backed by standard RAG consis-
tently fail to identify when legal precedents are
overruled—a direct real-world instantiation of The-
orem 1. Stammbach et al. (Stammbach et al., 2026)
document that existing legal retrieval benchmarks
fail to transfer to practitioner contexts, underscor-
ing the need for domain-specific retrieval objec-
tives. Di Florio et al. (Di Florio et al., 2024) formal-
ize legal conflict resolution with temporal ordering,
providing formal apparatus complementary to our
empirical framework. CaseFacts (Putta et al., 2026)
benchmarks precedent retrieval. RIRAG (Gokhan
et al., 2024) and FinDER (Choi et al., 2025) address
regulatory and financial QA without supersession
structure.
Supersession-aware retrieval.The closest prior
work is de Martim (de Martim, 2025), who pro-
poses SAT-Graph RAG—an ontology-driven graph
RAG that models legislative events as queryable
nodes and handles point-in-time legal retrieval.
Like CAR, SAT-Graph RAG recognizes that flat-
text retrieval is blind to the diachronic structure
of law and uses a graph to represent norm super-
session. The key distinction is theoretical: SAT-
Graph RAG is an engineering system for struc-
tured legal corpora with no formal retrieval objec-
tive, no correctness characterization (Theorem 4),
and no hardness result showing that scope-indexed
algorithms face an identifiable worst-case ceiling
(Proposition 2). CAR defines front(cl(A k(q))) as
an independent retrieval objective applicable to any
authority-governed domain, and proves necessary-
and-sufficient conditions onanyretrieved set inde-
pendent of how it was produced.
Temporal knowledge graphs.Temporal KG QA
(Su et al., 2024) and TDBench (Kim et al., 2026) ad-
dress time-sensitive graph queries over structured
KGs. TC-MQA is the unstructured-text analogue:
supersession must be inferred from document type
and timestamps rather than read from graph edges.RAG limitations.Weller et al. (Weller et al.,
2026) prove theoretical limits of embedding-based
retrieval; Theorem 1 is a domain-specific instantia-
tion with an explicit supersession construction. Bar-
nett et al. (Barnett et al., 2024) catalog seven RAG
failure modes;Controlling Authority Retrievalis
a distinct eighth mode not in their taxonomy—
one where the correct answer is provably unreach-
able by relevance ranking. Rashtchian and Juan
(Rashtchian and Juan, 2025) study retrieval suf-
ficiency; TCA formalizes a stronger sufficiency
notion that includes temporal validity.
8 Research Program
Formalizing CAR opens several research direc-
tions, each corresponding to a relaxation of one
component of front(cl(A k(q))) . The four general-
ized CAR variants are defined formally in §2.4; the
directions below elaborate their open problems.
Latent-scope CAR(Definition 8). When scope
is unobserved, the retrieval problem becomes in-
ference over p(σ|q) . The key open problem:
given a query-time NER error rate ϵ, what is the
achievable TCA ceiling? Proposition 2 gives a
φ(q) -parameterized upper bound; the latent-scope
case requires integrating over the scope posterior.
Probabilistic-authority CAR(Definition 9).
Learning π(d, d′)from text is a new structured pre-
diction problem: the target relation is “B formally
voids A under domain rules,” not “A happened
before B.” Risk-sensitive frontier recovery under
uncertain edges connects to distributionally-robust
optimization.
Multi-authority CAR(Definition 10). The lat-
tice of active frontiers under minteracting orders is
an open combinatorial problem. For m= 2 , fron-
tier recovery generalizes to finding the Pareto front
under two partial orders; for m≥3 , this is com-
putationally hard in general (multi-dimensional
Pareto dominance).
Streaming CAR(Definition 11). Maintain-
ingfront(cl(A k(q))) under document arrivals has
amortized update cost O(|front|) per insertion—
but the frontier can change discontinuously when a
new document supersedes multiple existing ones.
Batch-invalidation and lazy-update strategies are
unexplored.
Authority-preserving embeddings.Train
representations so that cl(A) is geometrically
identifiable—anchors and their controlling author-
ities cluster in authority-closure space rather than

semantic space. The new contrastive objective:
d⇝d′should bring dandd′close in embedding
space without conflating their semantic roles.
Budgeted closure recovery.Given lookup bud-
getB, maximize |front(cl(A k(q)))| subject to B.
Proposition 3 gives the lower bound on necessary
lookups; the gap between the lower bound and prac-
tical budget is the open problem. This connects to
adaptive submodular maximization.
Normative retrieval.CAR is one instance of
the broader class where the retrieval target is the
normatively correct document under a rule system,
not the most relevant one. Characterizing the com-
plexity of this class—which problems reduce to
CAR, which require stronger primitives—is the
natural theoretical generalization.
9 Limitations
Entity extraction in free-form text.The FinSu-
perQA benchmark uses structured employee IDs
that admit 100% regex extraction, and the GHSA
evaluation with structured templates admits 100%-
accurate Regex-NER. For free-form queries with-
out the CVE identifier, the two-stage pipeline (§6.6)
achieves TCA@5 = 0.975 by using BM25 to find
the CVE disclosure first, then mapping disclosure
metadata to the paired patch note via entity-indexed
lookup. The 2.5% failure rate arises from three
generic queries with insufficient vocabulary over-
lap and one package with no BM25 term match;
in these cases the pipeline degrades to the BM25
baseline. In fully free-form settings where product
names are absent or misspelled, additional NER or
entity linking would be required.
Entity collision in production corpora.On
an entity-disjoint corpus—where each (employee,
security) pair appears in exactly one example—
anchor-based probing fully solves T0–T3 (TCA =
1.000 ). In corpora where entities recur across cases
(e.g., an employee with multiple concurrent open
positions), the entity probe surfaces events from
all cases, requiring additional disambiguation. The
contamination ablation (§6.3) shows RSSG recov-
ers+12.4 pp in this regime ( 0.652→0.776 ); clos-
ing the full gap requires case-boundary metadata or
temporal windowing not always available in prac-
tice.
Synthetic supersession rules.The 10 rules cover
representative compliance scenarios. Production
deployment would require domain-expert rule spec-ification for a specific regulatory framework.
Evaluated but absent from prior work.The
cross-encoder reranker and OSV .dev (the security
production lookup tool) are now evaluated (§6.6).
The cross-encoder (TCA@5 = 0.233 ) confirms
that reranking within a relevance-ranked pool can-
not substitute for entity-indexed Stage-2 retrieval.
OSV .dev is 0% applicable under free-form queries—
it requires an explicit CVE identifier and solves a
different problem from CAR. Shepard’s Citations
(legal supersession) was not evaluated; it requires
a subscription not available during this study. The
remaining absent baseline isfine-tuned bi-encoders
trained on domain-specific (q, d∗)pairs. Remark 2
argues that such models implement proactive en-
tity search implicitly and are therefore counterex-
amples in form only; fine-tuning on 127 GHSA
pairs (3 epochs, MultipleNegativesRankingLoss)
produced a model with TCA@5 = 0.438 —below
the zero-shot MiniLM baseline ( 0.500 ) and 53 pp
below Two-Stage (0.969) . The fine-tuning run did
not improve in-domain performance, confirming
that the gap is architectural rather than a training-
data artifact.
Scope of current theory.Theorems 1–4 and
Propositions 1–3 characterizestatic, deterministic,
single-authority CAR with recoverable scope. The
four generalized variants (Definitions 8–11) for-
malize the extensions: latent-scope, probabilistic-
authority, multi-authority, and streaming CAR.
Each corresponds to relaxing exactly one compo-
nent of front(cl(A k(q))) and maps to a specific
term in Proposition 1—scope posterior for φ(q) ,
edge confidence for ⇝, order count for the lattice
structure, and corpus dynamics forR anchor .
10 Conclusion
The paper makes two primary theoretical claims
about the class of authority-governed retrieval prob-
lems, independent of any particular system.
Theorem 4gives the necessary-and-sufficient
conditions on any retrieved set Rfor TCA = 1:
frontier inclusion ( front(cl(A k(q)))⊆ R ) and
no-ignored-superseder. Any future algorithm—
learned, hybrid, or rule-based—can be audited
against these two conditions without reference to
how it was built.
Proposition 2establishes that scope-indexed al-
gorithms (those without access to the authority rela-
tion⇝) face a hard worst-case ceiling: TCA@k≤

φ(q)·R anchor (q). The proof is adversarial, not
average-case: when κ(q) authority chains share
entity scope, no algorithm that cannot distinguish
them achieves more than 1/κ(q) =φ(q) condi-
tional success probability.
These two results are supported by three theo-
rems that characterize the problem structure: Theo-
rem 1 (TCA and Recall@ kdecouple when ∆(q)>
0), Theorem 2 (authority closure is necessary for
TCA = 1), and Theorem 3 (anchor discovery and
authority resolution are each necessary; Corollary 3
proves TCA = 1 is achievable when φ(q) = 1 ).
Proposition 1 organizes the picture into the fac-
torization TCA =R anchor·φ·R frontier , whose
three terms are separable failure modes (Lemma 1).
Proposition 3 establishes the computational cost of
disambiguation under an explicit oracle model.
Three independent real-world corpora con-
firm that the proved structure is consequential:
Dense TCA@5 ∈ {0.064,0.172,0.270} across
FDA, SCOTUS, and GHSA; Two-Stage achieves
{0.774,0.926,0.975} . The 77.4% FDA ceiling
is the corpus-level average of φ(q) at Stage-1—
a property of the scope distribution, not a retrieval
limitation. A GPT-4o-mini downstream experiment
quantifies the cost of ignoring supersession: confi-
dent “not patched” claims for 39% of queries with
a patch; Two-Stage reduces this to 16%.
The four formal generalizations (Definitions 8–
11) identify the next layer of open problems: latent-
scope CAR, probabilistic-authority CAR, multi-
authority CAR, and streaming CAR. Each is a pre-
cisely stated problem class, not a vague direction,
and each inherits Theorem 4 as the correctness tar-
get.
References
Scott Barnett, Stefanus Usman, Sina Nazeri Haque,
Surya Sannasi, and Vijayaraghavan Poluri. 2024.
Seven failure points when engineering a retrieval
augmented generation system.arXiv:2401.05856.
Ilias Chalkidis, Manos Fergadiotis, Nikolaos Manginas,
Eva Katakalou, and Prodromos Malakasiotis. 2021.
Regulatory compliance through doc2doc information
retrieval: A case study in EU/UK legislation where
text similarity has limitations. InEACL.
Chanwoong Choi, Jeong-Hoon Kwon, Jinheon Ha,
and 1 others. 2025. FinDER: Financial dataset
for question answering and evaluating RAG.
arXiv:2504.15800.
Hudson de Martim. 2025. An ontology-driven graphRAG for legal norms: A structural, temporal, and
deterministic approach.arXiv:2505.00039.
Cecilia Di Florio, Huimin Dong, and Antonino Rotolo.
2024. When precedents clash: Formal case-based
reasoning with temporal and hierarchical conflict res-
olution.arXiv:2410.10567.
Anoushka Gade and Jorjeta Jetcheva. 2024. It’s about
time: Incorporating temporality in retrieval aug-
mented language models.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (ACL), pages 1762–1777.
Tuba Gokhan, Kun Wang, Iryna Gurevych, and Ted
Briscoe. 2024. RIRAG: Regulatory information re-
trieval and answer generation.arXiv:2409.05677.
Bernal Jim ´enez Guti ´errez and 1 others. 2025. From
RAG to memory: Non-parametric continual learning
for large language models.arXiv:2502.14802.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InCOLING.
Daniel Huwiler, Kurt Stockinger, and Jonathan
F¨urst. 2025. VersionRAG: Version-aware retrieval-
augmented generation for evolving documents.
arXiv:2510.08109.
Soyeon Kim, Jindong Wang, Xing Xie, and Steven Eui-
jong Whang. 2026. Harnessing temporal databases
for systematic evaluation of factual time-sensitive
question-answering in large language models. In
ICLR.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, and 1 others. 2020. Retrieval-
augmented generation for knowledge-intensive NLP
tasks. InNeurIPS.
Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing
Qi, and Jianxing Liu. 2025. T-GRAG: A dynamic
GraphRAG framework for resolving temporal con-
flicts and redundancy.arXiv:2508.01680.
Anand Rao Putta, Justin Devasier, and Cecilia Li. 2026.
CaseFacts: A benchmark for legal fact-checking and
precedent retrieval.arXiv:2601.17230.
Cyrus Rashtchian and Da-Cheng Juan. 2025. Sufficient
context: A new lens on retrieval augmented genera-
tion systems. InICLR.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empiri-
cal Methods in Natural Language Processing, pages
3982–3992.

Stephen E Robertson and Steve Walker. 1994. Some
simple effective approximations to the 2-Poisson
model for probabilistic weighted retrieval. InSIGIR.
Dominik Stammbach, Kylie Zhang, Patty Liu, Nimra
Nadeem, Inyoung Cheong, Lucia Zheng, and Peter
Henderson. 2026. Legal retrieval for public defend-
ers.arXiv:2601.14348.
Miao Su, Zixuan Li, Zhuo Chen, Long Bai, Xi-
aolong Jin, and Jiafeng Guo. 2024. Temporal
knowledge graph question answering: A survey.
arXiv:2406.14191.
Zheyuan Tang and 1 others. 2025. PropRAG: Guiding
retrieval with beam search over proposition paths.
arXiv:2504.18070.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multihop
questions via single-hop question composition. In
TACL.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InACL.
U.S. Securities and Exchange Commission. 2004. SEC
rule 204a-1: Investment adviser codes of ethics. 17
CFR Part 275.
Siwei Wang, Yangsen Zhang, Yalong Guo, and Jing
Kang. 2025. LedgerRAG: Governance-driven agen-
tic chain of retrieval for dynamic knowledge scenar-
ios.Electronics, 15(7):1376.
Orion Weller, Michael Boratko, Iftekhar Naim, and Jin-
hyuk Lee. 2026. On the theoretical limitations of
embedding-based retrieval. InICLR.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. InEMNLP.
Lena Zhang, Jakub Savelka, and Kevin Ashley. 2025.
Do LLMs truly understand when a precedent is over-
ruled? InJURIX.
A Supersession Rules R1–R10
B Query Templates
T0:“What is the pre-clearance status for {emp}’s
requested trade in{sec}?”
T1:“Is {emp}’s pre-clearance for {sec}still
valid?”
T2:“Can {emp}currently trade {sec}given all
recent compliance updates?”
T3:“What is {emp}’s current compliance status
for trading{sec}?”Rule Trigger Target Scope
R1 BlackoutAnnounced PreClearanceApproved Same ticker
R2 BlackoutLifted BlackoutAnnounced Same ticker
R3 EmergencyException BlackoutAnnounced Emp + ticker
R4 ExceptionRevoked EmergencyException Emp + ticker
R5 WatchlistAdded PreClearanceApproved Same ticker
R6 WatchlistRemoved WatchlistAdded Same ticker
R7 ConflictAmended ConflictDisclosed Same emp
R8 ConflictCleared ConflictAmended Same emp
R9 ConflictCleared ConflictDisclosed Same emp
R10 PolicyUpdate PolicyAcknowledged Global
Table 14: Supersession rules. Trigger is the later event
superseding Target.
C Implementation Details
BM25.Robertson’s BM25 (Robertson and
Walker, 1994) withk 1= 1.5,b= 0.75.
Dense.TF-IDF cosine similarity with smoothed
IDF. All events vectorized offline.
Supersession graphs.Two graphs are used.Per-
example KB graph: isolated, BFS transitive closure;
|K| ≤3 , used only for TCA ground truth.RSSG
(Retrieved-Set Supersession Graph): built at query
time from all entity-scoped events returned by the
full-corpus entity index; rules R1–R10 applied
pairwise by event type and timestamp comparison.
Used for answer derivation in TWOSTAGE+RSSG.
Entity index. idx[(emp,ticker)]→[e] precom-
puted once over the 12,250-event corpus.
Answer derivation priority.Active EMER-
GENCYEXCEPTION >active BLACKOUTAN-
NOUNCED/WATCHLISTADDED >active PRE-
CLEARANCEAPPROVED >active BLACKOUT-
LIFTED(no clearance)>REQUIRESREVIEW.
DDomain Adapters and Scorer Interface
Adapting the two-stage pipeline to a new authority
domain requires three attributes:
•anchor type : document type that is se-
mantically aligned to the query (e.g., CVE
DISCLOSURE, original SCOTUS opinion,
FDA approved label).
•superseding type : document type that
formally voids the anchor (e.g.,PATCH RE-
LEASE NOTE, overruling opinion, FDA recall
enforcement action).
•scope keys : entity fields that co-
scope anchor and superseder (e.g.,
[cve id, package] ,[case slug] ,
[ndc code]).

Three domain adapters are re-
leased alongside this paper:security
(realworld/ghsa twostage eval.py ),
legal( realworld/legal liibenchmark.py ),
andmedical( realworld/fda recall eval.py ).
Each adapter is a self-contained Python script that
fetches or loads its domain dataset, builds the
entity index, runs all baselines, and writes results
todata/.
Scorer interface. evaluate.py runs the full
synthetic FinSuperQA benchmark. The custom re-
triever class must implement retrieve(self,
query: str, corpus: list[dict],
k: int) -> list[str] , where corpus
is the list of document dicts and the return value
is a list of docidstrings. The scorer computes
TCA, Acc, ProvRec, and R@ kon all 1,000
examples with a single call:
python evaluate.py --retriever_path my_retriever.py --k 5
Output is written to
data/custom retriever results.json .
All datasets, domain adapters, evaluation
scripts, and the falsification suite are available
at https://github.com/andremir/
car-retrieval.