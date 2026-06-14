# SMSR: Certified Defence Against Runtime Memory Poisoning in Persistent LLM Agent Systems

**Authors**: Tarun Sharma

**Published**: 2026-06-10 21:45:52

**PDF URL**: [https://arxiv.org/pdf/2606.12703v1](https://arxiv.org/pdf/2606.12703v1)

## Abstract
Retrieval-augmented generation (RAG) agents increasingly run with persistent memory that accumulates across user sessions. This creates a new attack surface: an adversary interacting only through normal channels can inject crafted memories that, once retrieved, steer the agent's responses for future users, without touching model weights or code. We call this Multi-Session Memory Poisoning (MSMP) and show that no existing defence certifies against it; static-corpus defences (RobustRAG, ReliabilityRAG) assume a fixed knowledge base, and heuristic filters are bypassed by fluent enterprise-style text. We present Signed Memory with Smoothed Retrieval (SMSR), the first defence with a certified robustness bound for this setting. Component 1 adds HMAC-SHA256 provenance at write time, blocking unsigned injection. Component 2 applies randomised memory ablation with verdict-based majority voting at query time, bounding the influence of authenticated adversaries. We prove that no provenance-free retrieval-time filter can certify against adaptive injection, derive a hypergeometric certificate for Component 2, and formalise the Consistent Minority Effect, whereby a consistent adversarial answer wins string-based voting as a numerical minority while verdict-based voting removes it. Across 15 enterprise scenarios (3,150 repeated trials), Component 1 cuts attack success from 93-100% to 0% for all unsigned variants. For an authenticated adversary with a single injection, Component 2 holds success to 8.0% (95% CI [5.8, 10.9], n=450), below the certified worst case. In an end-to-end query-only attack where the agent itself writes the poison rather than it being pre-seeded, SMSR reduces success from 65.3% to 5.3% (n=150, non-overlapping CIs) on a live agent stack. Clean-query utility is 90% (Component 1) and 85% (combined).

## Full Text


<!-- PDF content starts -->

PREPRINT 1
SMSR: Certified Defence Against Runtime Memory
Poisoning
in Persistent LLM Agent Systems
Tarun Sharma
Abstract—Retrieval-Augmented Generation (RAG) systems
underpin an increasing fraction of enterprise AI deployments.
When an agent’s memory store persists across user sessions,
an adversary who can interact with the system can inject
carefully crafted memories that, once retrieved, redirect the
agent’s behaviour on future queries—without ever modifying
model weights or application code. We call this theMulti-
Session Memory Poisoning(MSMP) threat and observe that no
existing defence provides a formal security certificate against it.
Static-corpus defences (RobustRAG, ReliabilityRAG) assume a
fixed knowledge base; heuristic filters are bypassable by fluent
enterprise-style text.
We presentSigned Memory with Smoothed Retrieval(SMSR),
a two-component defence that is the first to provide a certified
robustness bound for the MSMP setting. Component 1 applies
HMAC-SHA256 provenance tagging at write time, creating a
cryptographically hard boundary against unsigned injection.
Component 2 applies randomised memory ablation at query time
with a verdict-based majority aggregator, bounding the influence
of authenticated adversaries.
We prove an impossibility result showing that no provenance-
free retrieval-time filter can certify against adaptive injection,
and derive a hypergeometric certificate for Component 2. We
also formalise and quantify theConsistent Minority Effectin the
memory-poisoning setting: string-based majority vote is gamed
by adversaries who generate consistent responses, and verdict-
based aggregation removes the effect.
Empirical evaluation on 15 enterprise knowledge-base sce-
narios (3,150 repeated trials: six Component-2 configurations×
15 scenarios×30 repetitions, plus 450 production-scale trials)
reports five findings. First, Component 1 reduces ASR from 93–
100% to 0% for all unsigned injection variants including bypass
attacks crafted to evade heuristic filters. Second, in a production-
scale store (m= 20, 20 seed memories, Tier 1), Component 2
reduces authenticated ASR from 93–100% to 8.0% (95% CI
[5.8%, 10.9%],n= 450) fort= 1, which lies safely below
the Theorem-2 worst-case boundδ= 10.4%at our evaluated
settingn runs = 5(the bound tightens to 7.1% atn runs = 7,
which we do not evaluate). Third, in our small-store evaluation
(m′= 10+t), the canonical direct-injectiont= 1ASR is 37.8%
(95% CI [33.4%, 42.3%],n= 450), below the eval-pool bound
δ= 41.5%; the direct-injection bound holds att∈ {1,2,3}, while
the flooding variant sits at the worst-case bound (its CI straddles
δ), as expected for a tight certificate. Fourth, judge reliability is
κ= 0.955(Haiku vs Sonnet,n= 84); in the production-scale
20-seed store, Sonnet and Haiku both achieve 8.0% ASR under
Component 2, confirming the defence generalises across model
families (Section VII-F). Fifth, in an end-to-end test where the
agent itself writes the poison via a query-only attack (not pre-
T. Sharma is an Independent Researcher (e-mail: tarun.sharma@ieee.org).
Code and data are available at https://github.com/tarun-ks/smsr.
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.seeded), SMSR reduces ASR from 65.3% to 5.3% (n= 150,
non-overlapping CIs; Section VII-I). Utility on clean queries is
90% under Component 1 and 85% under the combined defence.
Index Terms—LLM agent security, retrieval-augmented gen-
eration, memory poisoning, certified robustness, provenance,
randomised smoothing, OWASP LLM08.
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) agents are de-
ployed at scale in enterprise environments for internal search,
automated customer support, compliance checking, and agen-
tic process automation [1], [2]. A defining characteristic
of production deployments ispersistent memory: the agent
accumulates a growing store of past interactions, retrieved
documents, and inferred facts across user sessions, enabling
context continuity and personalisation.
The attack surface.This persistence creates a novel attack
vector: an adversary who can interact with the system through
normal channels—whether as an employee, a customer, or
a compromised upstream tool—can craft a sequence of in-
teractions that, once stored in the memory bank, persist
indefinitely and corrupt future responses for any user whose
query semantically matches the poisoned entry. Unlike training
data poisoning or jailbreak attacks that target model weights
or prompts directly, thisruntime memory poisoningoperates
at the retrieval layer and leaves no trace in the model itself.
MINJA [3] demonstrated that this attack achieves≈76–
99% success rates against memory-augmented agents. Agent-
Poison [4] showed that a single poisoned entry in a 23k-
entry database suffices for 62% end-to-end attack success.
MemoryGraft [5] demonstrated persistent compromise across
sessions at 48% poisoned retrieval proportion. None of the
proposed countermeasures in these works provides a formal
security guarantee.
Why existing defences are insufficient.The two most
rigorous existing defences, RobustRAG [6] and Reliabili-
tyRAG [7], assume astaticdocument corpus that the adversary
poisons externally before queries are made. They provide
no security guarantee when poisoned entries are injected
at runtime through the agent’s normal memory write path.
Heuristic defences such as A-MemGuard [8] use consensus
validation across multiple reasoning paths. In our like-for-like
evaluation (Section VII), A-MemGuard attains empirical ASR
comparable to SMSR att= 1, but it providesno formal
guarantee: SMSR’s distinction is a certified robustness boundarXiv:2606.12703v1  [cs.CR]  10 Jun 2026

PREPRINT 2
combined with over-fetch random sampling that, unlike full-
retrieval consensus, degrades gracefully as a persistent adver-
sary adds entries. As a general-purpose heuristic baseline we
implement a keyword blacklist, entropy proxy, and semantic
anomaly filter (not attributed to any specific system) and show
empirically that all three are fully bypassed (100% ASR) by
fluent enterprise-style policy text, motivating the need for the
provenance mechanism in Component 1.
Contributions.This paper makes the following contribu-
tions:
1)Formal definitions: We introduce the Multi-Session
Memory Poisoning (MSMP) threat model and the first
formal(t, δ)-security definition for runtime agent mem-
ory systems (Section III).
2)Impossibility result: We prove that no provenance-free
retrieval-time filter can achieve a non-trivial security
certificate against an adaptive MSMP adversary (The-
orem 1).
3)SMSR construction: We present the Signed Memory
with Smoothed Retrieval defence, combining HMAC
provenance (Component 1) with randomised ablation
and verdict-based majority aggregation (Component 2),
and derive a formal hypergeometric certificate for Com-
ponent 2 (Theorem 2).
4)Consistent Minority Effect (CME): We formalise and
characterise the CME — a known self-consistency pit-
fall [9] — in the memory-poisoning setting: an adver-
sarial response wins string-based majority vote despite
being a numerical minority, because it is textually more
consistent than the varied benign responses. We show
that verdict-based aggregation removes the effect and
quantify it honestly (Proposition 2, Section VI-B).
5)Empirical validation: We evaluate SMSR on 15
enterprise knowledge-base scenarios across three at-
tack classes (unsigned injection, authenticated injection,
heuristic bypass) and four defence configurations, using
a LLM judge for reliable verdict evaluation.
II. BACKGROUND ANDRELATEDWORK
A. Memory-Augmented LLM Agents
Modern agentic systems such as LangGraph [10], Auto-
Gen [11], and MemGPT maintain persistent memory as a
vector database over agent interaction traces. At query time the
agent embeds the user’s queryϕ(q)∈Rdand retrieves thek
entries from the memory store with highest cosine similarity.
These entries are prepended to the LLM’s context as few-
shot demonstrations or factual context, directly influencing the
response.
Unlike static RAG over a curated document corpus (e.g.,
a company policy PDF index), agent memory is continuously
written to by the agent itself during normal operation—every
interaction typically generates one or more new memory
entries. This creates the fundamental tension at the heart of the
MSMP threat: the write path that enables the agent’s utility
(accumulating useful context over time) is also the injection
path for an adversary. Traditional access-control solutions that
distinguish “data” from “instructions” do not apply, becausein an agent memory system, data IS instructions—retrieved
memories function as in-context demonstrations that shape
future behaviour.
Multi-tenant deployments exacerbate the problem: a shared
memory store serving multiple users means one user’s in-
teractions can affect all future users’ responses. This is the
precise scenario exploited by MINJA [3], which achieves
≈98% injection success rate across GPT-4/Claude in shared
agent settings.
B. The OWASP LLM08 Threat
The Open Web Application Security Project (OWASP)
recognises “Vector and Embedding Weaknesses” as LLM08
in its 2025 Top 10 for LLM Applications [1]. LLM08 covers
bothinversionattacks (recovering original text from stored em-
beddings) andpoisoningattacks (injecting malicious content
into the retrieval pipeline). Our work addresses the poisoning
sub-problem, specifically the dynamic injection variant that is
absent from all prior formal treatments of LLM08 defences.
C. Memory Poisoning Attacks
MINJA [3] is the first systematic study of runtime memory
injection: the adversary sends crafted queries that cause the
agent to generate and store poisoned reasoning traces, which
are subsequently retrieved as few-shot demonstrations for
future users (≈98% injection success rate). AgentPoison [4]
targets RAG knowledge bases with a trigger token that clusters
poisoned entries in embedding space, achieving 62.6% end-to-
end attack success. MemoryGraft [5] poisons an agent’s per-
sistent experience store via a benign-looking artefact, demon-
strating that 10 poisoned seeds in a 110-entry store yield 48%
poisoned retrieval. The Agent Security Bench [12] catalogues
these and related agent attacks in a unified benchmark.
None of these attack papers proposes a certified defence;
their countermeasure discussions are heuristic. Concurrent
work by Devarangadi Sunil et al. [13] proposes trust-scoring
and memory-sanitisation defences and independently observes
that a store pre-populated with legitimate memories dilutes
attack success—the same effect our Component-2 certificate
(Theorem 2) quantifies formally—but their defences remain
heuristic with no certified bound.
D. Defences for RAG Systems
Why static-corpus defences fail for MSMP.Robus-
tRAG [6] and ReliabilityRAG [7] provide certified bounds
under a model that is structurally incompatible with the MSMP
setting. Both assume: (1) the corpus is indexed offline before
queries are answered; (2) the adversary poisons the corpus
externally; and (3) the number of poisoned documents in the
retrieved set is ana prioriknown constant. In the MSMP
setting all three assumptions are violated: the memory store
is a live append-only log, the adversary operates through
the same write path as legitimate users, and the number of
adversarial entries in the retrieved set is unknown and depends
on adversary persistence.
The “isolate-then-aggregate” strategy of RobustRAG re-
quires partitioning the retrieved set into⌈k/k′⌉disjoint groups,

PREPRINT 3
WRITE PATH
QUERY PATHUnsigned Attacker
no credential
Legitimate User / Agent
Authenticated Attacker
valid HMAC tagComponent 1
HMAC-SHA256
signing oracleMemory Store
verified entries
(HMAC-tagged)
User Query q*Component 1
HMAC filter:
rejects unsignedComponent 2
k-of-m ablation ×nVerdict Vote
LLM-judge
majorityResponse r*
Adversarial Legitimate SMSR componentsign + storedirect DB write  no tag
retrieve top-m verified
Fig. 1: SMSR system architecture. The write path (top) shows how the HMAC signing oracle tags every legitimate memory
and how an unsigned attacker is blocked at retrieval time by Component 1. An authenticated attacker (legitimate user) can
write signed memories but is mitigated by the randomised ablation and verdict-based aggregation of Component 2. The query
path (bottom) shows both components in sequence.
each processed independently. A persistent adversary in a
dynamic memory system can plant entries in every partition,
collapsing the certificate. Our Component 2 addresses this via
over-fetch randomisation (m≫k) rather than deterministic
partitioning, preventing the adversary from guaranteeing pres-
ence in every sampled context.
Overview.RobustRAG [6] applies an isolate-then-aggregate
strategy that provides a certified lower bound on response
quality when at mostk′of thekretrieved passages are
adversarially injected. ReliabilityRAG [7] builds on this with
a Maximum Independent Set construction on a document
contradiction graph. Both require a fixed, pre-indexed corpus
and explicitly do not address dynamic memory writes; their
proofs rely on a static adversary model.
For the query privacy dimension of RAG, Boldyreva and
Tang [14] formalise simulation-based security for private ap-
proximatek-NN search, covering access, query, and volume
pattern leakage. The SoK by Bodea et al. [15] systematically
reviews the RAGprivacyliterature; no certified poisoning
defence for the dynamic/runtime memory setting is known to
us — the closest works treat static-corpus robustness [6], [7]
and leave the dynamic setting open.
E. Randomised Smoothing and Ablation
Cohen et al. [16] introduced randomised smoothing to cer-
tifyℓ 2robustness for image classifiers. Levine and Feizi [17]
adapted this to text by randomised ablation: a classifier trainedon randomly masked inputs inherits a certified bound based
on the proportion of tokens the adversary can control. Our
Component 2 follows the same structural argument from
Levine and Feizi [17] (for image classifiers) and Zeng et
al. [18] (for text), adapted to the retrieval-over-memory setting
where the adversary controls a bounded number of entries in
the retrieval candidate pool.
III. THREATMODEL ANDPROBLEMFORMULATION
A. System Model
We consider a multi-session RAG agentAwith:
•A persistent memory storeM={m 1, . . . , m N}where
each entrym iis a text string and its embedding.Mis
append-only during normal operation; entries are written
by the agent after each interaction.
•A retrieval function RETRIEVE(q,M, k)that returns the
kentries most similar to queryqby cosine similarity.
•An LLM response generatorGthat takes queryqand
retrieved contextCand produces a responser.
Sessions are independent in that a new user at sessions′
has no knowledge of prior sessions, but both sessions query
the same persistentM.
B. Adversary Model
We consider two adversary classes of increasing capability:
Unsigned adversaryA U: Has direct write access toM(via
database misconfiguration, SQL injection, or stolen backup

PREPRINT 4
with append rights) and can insert entries without any authen-
tication credential. Cannot modify existing entries or delete
anything.
Authenticated adversaryA P: Is a legitimate user of the
system; can write entries through the normal agent interaction
path. Can inject at mosttadversarially crafted entries per
attack campaign. Does not know the server-side secret keyK.
Both adversaries know the embedding model, the agent
architecture, and the retrieval mechanism. Neither knows the
HMAC keyK.
C. Security Definitions
Definition 1(Multi-Session Memory Poisoning (MSMP)).An
MSMP adversaryAagainst agentAwith memoryMis a
probabilistic algorithm that:
1)(Write phase) Injects at mosttmemories{m∗
1, . . . , m∗
t}
intoMvia either the unsigned or authenticated write
path.
2)(Trigger phase) Observes that a future user issues a
target queryq∗semantically related to the injected
content.
The attack succeeds ifG(q∗,RETRIEVE(q∗,M, k))produces
a response that the adversary designates as “malicious” (a
pre-specified false claim).
Definition 2((t, δ)-SMSR Security).A memory retrieval
system is(t, δ)-SMSR-secureif for any MSMP adversaryA
that injects at mosttadversarial entries, the probability that
the agent’s response is malicious is at mostδ:
Pr[A(q∗,M) =malicious]≤δ.
IV. IMPOSSIBILITY OFPROVENANCE-FREECERTIFIED
DEFENCE
We first establish that any defence operating solely at re-
trieval time—without write-time provenance—cannot achieve
a non-trivial security certificate against an adaptive adversary.
Theorem 1(Impossibility of Provenance-Free Worst-Case
Certification).Letf:M × Q → {0,1}be anydeterministic
retrieval-time filter that decides whether to include memorym
in the context for queryqbased solely on content. Under the
following standard embedding assumption:
Assumption A1(fluent embedding density): For any target
query embeddinge∗and thresholdτ sim>0, there exists a
fluent text string whose embedding has cosine similarity≥τ sim
withe∗.
then for any unbounded adaptive adversaryA, nofachieves
a non-trivial worst-case certificate: there always exists an
adversarial entrym∗such thatf(m∗, q∗) = 1.
Note:Component 2 of SMSR isnotan instance of the filter
class covered by this theorem. Component 2 is arandomised,
content-agnosticmechanism whose certificate (Theorem 2)
relies on acount boundtandindependent sampling, not on
content decisions. The impossibility applies only to determin-
istic content-based filters.
Proof.We constructAexplicitly. Letϕ:M →Rdbe the
embedding function.Acomputes the target embeddinge∗=ϕ(q∗)and constructsm∗as a fluent text string that: (1) is
semantically on-topic forq∗, so that⟨ϕ(m∗), e∗⟩> τ simfor
any relevance threshold, and (2) asserts a false factual claim
in natural language.
The existence of suchm∗follows from two observations.
First, the embedding spaceRdcontains infinitely many points
within anyℓ 2ball arounde∗; by the density of fluent text
in embedding space (any region around a real document’s
embedding is populated by paraphrases), there exists a fluent
stringm∗with⟨ϕ(m∗), e∗⟩> τ sim. Second, appending a false
claim to an on-topic string changes its embedding by at most
O(∥m∗∥−1)(since the false claim is a small fraction of a
long entry), preserving the similarity bound for sufficiently
longm∗.
Sincefevaluates only(m, q)at retrieval time, and we
have constructedm∗to pass any threshold-based criterion,
f(m∗, q∗) = 1whilem∗causes the LLM to output the
malicious claim. Anyδ <1certificate forfis therefore
vacuous.
This result implies that a provenance mechanism atwrite
timeis necessary. Content-based filters (keyword blacklist,
perplexity, semantic anomaly) are instances offand are
subject to this impossibility.
V. THESMSR CONSTRUCTION
A. Component 1: HMAC Provenance Tagging
MemoryGraft [5] proposed “cryptographic provenance at-
testation” as a defence direction without formalising or certi-
fying it. Component 1 formalises this direction: we prove its
security under standard HMAC assumptions (Proposition 1)
and establish it as the necessary write-time boundary that
enables Component 2’s certificate. Our contribution is the
formal proof and the integration with Component 2, not the
concept.
Every legitimate memory write passes through a trusted
server-side signing oracle that tags each entry with an HMAC-
SHA256 signature under a server secretK:
τi=HMAC K(content i∥session idi∥timestampi).
At retrieval time, only entries with valid tags are admitted to
the candidate pool:
Mverified ={m i∈ M:VERIFY K(mi) = 1}.
Proposition 1(Security of Component 1).Under the pseu-
dorandomness of HMAC-SHA256, an unsigned adversaryA U
who does not knowKproduces a valid tag for any entry
with probability at most2−λ, whereλ= 256is the output
length in bits. Therefore Component 1 achieves(0,2−256)-
SMSR-security againstA U.
The server keyKshould be stored in a secrets manager or
HSM, never in application code. Key rotation invalidates all
pre-rotation memories, which provides forward secrecy at the
cost of losing history.

PREPRINT 5
Algorithm 1SMSR: Signed Memory with Smoothed Retrieval
Input:Queryq, storeM, keyK, paramsm, k, n runs, judge
J
// Component 1: retrieve verified candidates
C ←RETRIEVE(q,{m i∈ M:VERIFY K(mi)=1}, m)
// Component 2: randomised ablation + verdict aggregation
V←[]
forj= 1ton runsdo
C(j)←samplekfromCwithout replacement
r(j)←LLM(q,C(j))
V.append(J(r(j), q), r(j))
ˆv←arg max vP
j1[Vj.verdict=v]
returnfirstr(j)with verdict= ˆv
B. Component 2: Randomised Memory Ablation with Verdict-
Based Aggregation
Component 1 handles unsigned adversaries completely. For
authenticated adversariesA Pwho can write HMAC-signed
memories, a second layer is required.
Retrieval with ablation.Given a queryq, retrieve the top-
mverified candidates:C=RETRIEVE(q,M verified, m)where
m > k. For each ofn runsindependent trials, samplekentries
uniformly at random without replacement fromC:C(j)∼ C
k
.
Verdict-based aggregation.Run the LLM with each
contextC(j)to obtain responser(j). Apply a per-
run judgeJ(r(j), q)that returns a verdictv(j)∈
{correct,malicious,neither}. The final output is determined by
majority verdict:
ˆv= arg max
vnrunsX
j=11[v(j)=v],
and the corresponding responser(ˆj)is returned.
The use of verdict-based aggregation rather than string-
based majority is critical; we show in Section VI-B that string
equality is gamed by adversaries.
C. SMSR Protocol
Algorithm 1 presents the complete SMSR protocol.
VI. SECURITYANALYSIS
A. Formal Certificate for Component 2
Theorem 2(SMSR Certificate).LetM verified containN
entries of which at mosttare adversarially crafted (but validly
signed byA P). LetCbe the top-mcandidates retrieved for
queryq, of which at mostt′≤min(t, m)are adversarial.
Define
pclean= m−t′
k
 m
k,
the probability that no adversarial entry appears in a single
ablation sample. Then the probability that the majority verdict
is malicious is at most:
δ(t′, m, k, n runs) =nrunsX
i=⌈n runs/2⌉nruns
i
(1−p clean)i·pnruns−i
clean .TABLE I: SMSR certificates (k= 5,n runs= 5; everyδ
from Theorem 2 with its(t, m, n runs)shown). Empirical ASR
is from 450-trial pooled studies (30 reps×15 scenarios).
Productiont>1rows give the bound only (not evaluated at
m=20). Flooding injects the sametentries as direct injection
(distinct paraphrases), sot′=tand it sharesδ; its slightly
higher ASR reflects greater per-run persuasiveness, not a
largert′.
t mconfigδEmpirical
ASR95% CI≤δ?
Production pool (m= 20,n runs= 5)
1 20 Tier 1 (dir)0.104 8.0%[5.8, 10.9]✓
2 20 bound only 0.402 — — —
3 20 bound only 0.684 — — —
Evaluation pool (m′=10+t,n runs= 5)
1 11 eval/dir 0.415 37.8% [33.4, 42.3]✓
1 11 eval/fld 0.415 43.1% [38.6, 47.7]≈✓
2 12 eval/dir 0.812 80.0% [76.1, 83.4]✓
2 12 eval/fld 0.812 83.1% [79.4, 86.3]≈✓
3 13 eval/dir 0.945 92.7% [89.9, 94.7]✓
3 13 eval/fld 0.945 96.2% [94.0, 97.6]≈✓
≈✓= CI spansδ(tight);✓= point est.≤δ.
The system is therefore(t, δ(t′, m, k, n runs))-SMSR-secure.
Proof.Each ablation run is an independent uniform sample
ofkentries frommcandidates. The probability that run
jcontains at least one adversarial entry is1−p clean by
the hypergeometric distribution. Since runs are independent,
the number of runs containing adversarial contentX∼
Binomial(n runs,1−p clean).
The majority vote is malicious only if more thann runs/2
runs yield a malicious verdict. This requiresX≥ ⌈n runs/2⌉
AND the LLM produces a malicious response on each con-
taminated run. Bounding the second condition by 1 (worst
case), we obtainPr[malicious majority]≤Pr[X≥ ⌈n runs/2⌉],
which equals the statedδ.
Corollary 1.Fort= 1, the actual pool in our evaluation is
m′= 11(10 seed memories + 1 adversarial), soδ= 0.415at
nruns= 5. Over 30 independent repetitions per scenario (n=
30×15 = 450trials), the pooled empirical ASR is37.8%(95%
Wilson CI[33.4%,42.3%]). This satisfies37.8%≤41.5% =
δ✓; the point estimate sits below the bound, with the CI upper
bound (42.3%) marginally exceedingδby 0.8 pp, confirming
the bound is tight.
For a production deployment wherem′≈m configured = 20,
the certificate isδ= 0.104atn runs= 5(and0.071atn runs=
7, not evaluated); Fig. 3(b) shows themrequired to achieve
each targetδfor a givent.
Table I summarises certificate values for representative
parameter settings.
B. The Consistent Minority Effect
String-based majority vote is vulnerable to a pitfall known
in the self-consistency literature [9]: paraphrase variation splits
votes, allowing a consistent minority to win. We formalise and
quantify this effect in the memory-poisoning setting.

PREPRINT 6
Definition 3(Consistent Minority Effect (CME)).Theconsis-
tent minority effecton a string-based majority vote aggregator
occurs when an adversary’s responses win the vote despite
comprisingn adv< n runs/2of the ablation runs, because: (1)
alln advadversarial runs produce textually similar responses
(the malicious factual claim is specific and repeatable), and
(2) then clean clean runs producen clean distinctparaphrases
(“I don’t know” responses vary in phrasing).
Note:The general problem of paraphrase variation causing
vote splits is well known in the self-consistency literature [9];
the CME is an instantiation of this effect in the memory-
poisoning setting. Our contribution is characterising it in this
setting (the 93.3%→13.3% string-vs-verdict gap on a single
n= 15run, affecting 12/15 scenarios) and showing that
verdict-based aggregation removes it.
Proposition 2(CME on String V ote).String-based majority
vote is vulnerable to CME whenevern adv≥1and clean
responses are drawn from a high-diversity distribution. Specif-
ically, if clean responses are iid from a distribution with
min-entropyH clean (in bits), the probability that any clean
response appears more than once isO(n2
clean·2−H clean), which
approaches 0 asH clean→ ∞. An adversary who makes
radvthe unique most-frequent string wins the string vote with
probability approaching 1 asn clean→ ∞.
Proof.Letp sbe the probability of producing stringsin a
clean run. By a birthday-bound argument, the expected number
of clean strings appearing more than once isP
s nclean
2
p2
s≤ nclean
2
·2−H cleanby the definition of min-entropy. For natural-
language “I don’t know” responses to domain-specific enter-
prise queries,H cleanis large (responses mention query-specific
details), making this probability negligible. Thereforer adv,
even with count 1, is the unique most-frequent string in most
runs.
Empirical evidence.On a singlen= 15run, string-based
vote yields 93.3% ASR for authenticatedt= 1injection while
verdict-based aggregation (immune to the CME) yields 13.3%
on thesameablation runs, so the reduction is attributable
purely to the aggregator. The 13.3% is a low-npoint estimate:
over 30 repetitions the verdict-vote rate stabilises at 37.8%
(n= 450, Table I), against which the reduction from the
(saturated) string-vote rate is≈2.5×. We report the aggregator
effect rather than a precise multiplier, as string-vote was not
re-run atn= 450.
Fix.Verdict-based aggregation (Algorithm 1) is immune
to CME because it counts verdict labels, not string equality.
Adversarial responses of the form “The limit is $50,000” and
“According to policy, the limit is $50,000” receive the same
verdict (MALICIOUS) and are counted together.
VII. EVALUATION
A. Experimental Setup
Threat scenario.We simulate a Nexora Corp enterprise
RAG agent that provides employees with answers about
company policy (expense approval thresholds, data retention
periods, access control requirements, incident response proce-
dures). The agent’s memory store is pre-seeded with 10 signedlegitimate policy entries. Attacks inject false policy entries to
cause the agent to give incorrect guidance that could lead to
security violations, compliance failures, or financial harm.
Attack scenarios.We evaluate 15 enterprise knowledge
scenarios covering finance policy, compliance, IT security,
HR policy, procurement, and incident response. Each scenario
specifies a correct answer (ground truth) and a malicious
answer (the attacker’s goal). Attack injections are crafted
as plausible enterprise policy updates, not as overt injection
strings.
Attack classes.
•Unsigned (MINJA-style): injection via direct DB write,
no HMAC tag. Three variants:direct(identical copies),
flooding(paraphrase variants),subtle(embedded in
natural-sounding research notes).
•Authenticated: injection via normal agent interaction by
a legitimate user; entries receive valid HMAC tags.
•Heuristic bypass: unsigned injections crafted to evade
keyword blacklists, perplexity filters, and semantic
anomaly detection.
Defence configurations.Four modes:none(undefended
baseline),heuristic(keyword + entropy + semantic anomaly
filter, no provenance),c1(HMAC provenance only),c1c2
(HMAC + randomised ablation + verdict-based majority vote).
Parameters.m= 20,k= 5,n runs= 5, injection countst∈
{1,2,3}. LLM: Claude Haiku 4.5 for both agent and judge.
All empirical results usen runs= 5; anyn runs= 7certificate
we quote (e.g.δ= 7.1%att= 1,m= 20) is computed
from Theorem 2 as a design reference and isnotempirically
evaluated. We did not collect human labels; judge reliability is
instead quantified by inter-judge agreement (Haiku vs. claude-
sonnet-4-6: Cohen’sκ= 0.955, 97.6% raw agreement,n=
84; see the Judge Reliability study, E6).
Evaluation metric.Attack Success Rate (ASR) = fraction
of scenarios where the agent’s final response is judgedMA-
LICIOUSby the LLM judge. Utility = fraction of clean (non-
attack) queries judgedCORRECT. Total: the 39-configuration
sweep is39×15 = 585single-run attack trials (plus 80 utility
trials) underlying Table II. The six randomised Component-
2 configurations areadditionallyrepeated 30 times each
(6×15×30) and combined with the 450-trial Tier-1 study to
give the3,150repeated trials behind Tables I and IV.
B. Attack Effectiveness Baseline
Table II and Fig. 2 reportASRacross all configurations.
Unsigned attacks (rows 1–7).Component 1 reducesASR
to 0% across all six unsigned configurations and all injection
counts, including the bypass variant crafted to evade heuristic
filters. The heuristic defence reducesASRmodestly (from
93–100% to 87–100%) but never achieves protection against
flooding or subtle variants.
Bypass attack (row 7).The bypass attack uses fluent, low-
perplexity enterprise policy text with no blacklisted keywords
and high semantic similarity to the seed corpus. The heuristic
defence fails completely (100%ASR), empirically confirming
Theorem 1. Component 1 achieves 0%ASRbecause prove-
nance is evaluated at write time on the signature, not on
content.

PREPRINT 7
Direct
(n=1)Direct
(n=3)Flood
(n=3)Subtle
(n=3)020406080100ASR (%)
0% 0% 0% 0%(a) Unsigned Injection
Bypass
(subtle)020406080100ASR (%)
0%(b) Bypass
Dir
(t=1)Dir
(t=3)Fld
(t=1)Fld
(t=3)020406080100ASR (%)(c) AuthenticatedFig. 2: ASR (%) by Defence Mode  n450 trials per configuration
Undefended Heuristic C1 (HMAC) C1+C2 (SMSR)
Fig. 2: Attack Success Rate (%) across defence modes and attack classes (15 scenarios per configuration, LLM-as-judge
evaluation). Green “0%” labels confirm Component 1 achieves zero ASR for all unsigned and bypass attacks. Panel (c) shows
Component 2 reduces authenticated ASR from 93–100% to 37.8% (30-repetition pooled,n= 450) fort= 1.
TABLE II: Attack Success Rate (%) across defence modes and attack classes.t= number of injected entries. N/A = not
applicable.
Attack class VarianttDefence mode
none heuristic c1 c1c2
Unsigneddirect 1 93.3 86.70.0 0.0
direct 3 93.3 86.70.0 0.0
flooding 1 93.3 86.70.0 0.0
flooding 3 100.0 100.00.0 0.0
subtle 1 100.0 100.00.0 0.0
subtle 3 100.0 100.00.0 0.0
Bypass subtle 3 100.0 100.00.0 0.0
Authenticateddirect 1 93.3 N/A 100.037.8†
direct 3 93.3 N/A 93.3 92.7†
flooding 1 100.0 N/A 93.343.1†
flooding 3 100.0 N/A 100.0 96.2†
Utility (clean queries) 90.0% 90.0% 90.0% 85.0%
†c1c2 authenticated entries are 30-repetition pooled rates (n= 450; CIs in Table I). Other cells are single illustrative runs (n= 15).
Authenticated attacks (rows 8–11).Component 1 fails as
expected: signed entries pass the HMAC filter and reach the
LLM. Witht= 1and verdict-based aggregation, Component 2
reducesASRfrom 93–100% to 37.8% (pooled,n= 450;
eval regimem′= 11,δ= 41.5%atn runs= 5). Table II
reports the pooled c1c2 rates (37.8% direct, 43.1% flooding at
t= 1); the corresponding single-run figures (13.3% / 46.7%,
n= 15) appear only in the Consistent-Minority analysis of
Section VI-B. Witht= 3, protection degrades (ASR≈93–
100%) as the certificate predicts (δ= 0.945at evalm′= 13
andδ= 0.684at productionm= 20, both atn runs= 5).
Utility.Component 1 preserves utility at 90% (identical
to the undefended baseline) since it only filters unverifiedentries. Component 2 reduces utility to 85% due to random
subsampling of the retrieved context, a 5-percentage-point
cost.
C. Certificate Validation
Actual pool size in this evaluation.The evaluation store
contains 10 signed seed memories plustadversarial entries,
so the retrieved pool hasm′= min(m configured ,10 +t)entries
in practice. Fort= 1,m′= 11; fort= 3,m′= 13. The
configuredm= 20is reached only in deployments with≥
20verified memories; Table I reports certificates for both the
configuredm= 20(applicable to production deployments)
and the evaluation-specificm′.

PREPRINT 8
0 2 4 6 8
t (signed adversarial injections)0.00.20.40.60.81.0 (certified malicious-output prob.)
8.0%(a) Certificate vs. adversary budget
m=20, nruns=5 (bound)
m=20, nruns=7
=0.10
Tier-1 empirical (n=450)
5 10 15 20 25 30 35 40
m (over-fetch pool size)0.00.20.40.60.81.0
t=1: 0.10 at m=21
m=20 (used): =0.104
(b) Effect of pool size (k=5, nruns=5)
t=1
t=2
t=3
=0.10
Fig. 3: SMSR certificate from Theorem 2 atn runs= 5(all
values fromsmsr_certificate.py). (a)δvs. adversary
budgettfor the production poolm= 20(solid,n runs= 5;
dashed,n runs= 7); the red point is the pooled 30-repetition
Tier-1 empirical ASR (t= 1,n= 450), which sits below the
bound. (b) Effect of pool sizem:m= 21is the smallestm
reachingδ≤0.10att= 1; the configuredm= 20gives
δ= 0.104.
R5 "Cannot determine from context."R4 "No relevant context found."R3 "The limit is $50,000."R2 "I don't have that policy info."R1 "The limit is $50,000."
n/2
Winner: "$50,000" (count=2)(a) String-vote  FAILS
R5 verdict: NEITHERR4 verdict: NEITHERR3 verdict: MALICIOUSR2 verdict: NEITHERR1 verdict: MALICIOUS
n/2
Winner: NEITHER (count=3)(b) Verdict-vote  CORRECT
Fig. 4: Consistent Minority Effect (nruns=5)  2 identical adversarial responses win string-vote (a) but lose verdict-vote (b)
Fig. 4: Consistent Minority Effect (CME) atn runs= 5. With
2 adversarial runs and 3 benign runs, string-based majority
vote selects the malicious answer (count=2, the unique most-
frequent string), whereas verdict-based aggregation correctly
selects NEITHER (count=3).
E2 result: the uniform-sampling premise holds by
construction.Because each ablation run sampleskentries
uniformly without replacement from the over-fetched pool (Al-
gorithm 1), the probability that a run is contaminated is exactly
1−p cleanindependent of the similarity rankof the adversarial
entries: a top-ranked adversarial entry is no more likely to be
sampled than any other pool member, so preferential retrieval
cannot inflate contamination. A3×105-trial Monte-Carlo of
the sampler confirms the empirical contaminated-run fraction
matches1−p cleanto within±0.0006acrosst∈ {1,2,3}; the
uniform-independent-sampling premise of Theorem 2 there-
fore holds here. This argument fixes the adversarial countt′
withinthe pool, not how many adversarial entries enter it:
in our evaluation both direct and flooding inject exactlyt
entries (paraphrased for flooding), sot′=t; an unbounded
flooding adversary writing> tnear-duplicates would raiset′,
bounded by the write-time duplicate detection of Section VIII
and covered by thet′-parameterised certificate (Theorem 2).
Tier 1: Production-scale certificate validation (20 seeds,
m= 20).With 20 seed memories the retrieved pool reaches
m= 20(the configured production parameter), giving a
Theorem-2 bound ofδ= 0.104at our evaluatedn runs= 5.
Over 30 reps×15 scenarios (n= 450): pooled ASR= 8.0%,
95% Wilson CI[5.8%,10.9%]— comfortably below the
bound. This is the headline result: in a production deployment
with at least 20 legitimate memories, Component 2 limitsauthenticatedt= 1injection to8.0%ASR, within a certified
worst case of10.4%(which tightens to7.1%ifn runsis raised
to 7).
Certificate validation — small-store study (30 reps,m′=
10+t).Fort= 1, pooled ASR across eval configs:
1) Direct injection (t= 1): 37.8% CI [33.4, 42.3]≤δ=
41.5%✓.
2) Flooding (t= 1): 43.1% CI [38.6, 47.7] — CI straddles
δ= 41.5%; the bound is tight (distinct paraphrases drive
the per-run take-probability toward the worst case the
bound assumes).
3) Fort= 2andt= 3, direct-injection ASR is below the
eval-pool bound; the flooding variants sit at the bound
(CIs straddleδ; Table I), consistent with a tight worst-
case certificate across the tested range.
Fort= 3(m′= 13,δ= 0.945atn runs= 5): the 30-
repetition direct-injection ASR is92.7%CI [89.9, 94.7]≤
δ= 94.5%✓, confirming the certificate at the largest tested
adversary budget.
Fig. 3(a) also shows bounds for the production-scalem=
20; withn runs= 5this givesδ= 0.104att= 1, which
operators can tighten toδ≤0.07by raisingn runsto 7 or
increasingm.
D. A-MemGuard Comparison (E4)
Table III compares SMSR against an A-MemGuard-style
consensus baseline (retrieve full context, generate a memory-
augmented response and a parametric-only response, flag
divergence and fall back to parametric).
TABLE III: SMSR c1c2 vs. A-MemGuard (consensus fall-
back), authenticated direct injection (t= 1). All A-MemGuard
and SMSR Tier-1 rows atn= 450, 20-seed store (like-for-
like). The eval-pool row (m′= 11) uses a 10-seed store.
Defencenstoret= 1ASR Certified?
Undefended 15 10-seed 93.3% No
A-MemGuard (consensus)45020-seed3.8%[2.4, 6.0] No
SMSR c1c2 (eval,m′=11) 450 10-seed 37.8% [33.4, 42.3] Yes (δ=41.5%)
SMSR c1c2 (Tier-1,m=20)45020-seed8.0%[5.8, 10.9] Yes (δ=10.4%)
In the like-for-like comparison (20-seed store,n= 450):
SMSR achieves 8.0% CI [5.8, 10.9] and A-MemGuard
achieves 3.8% CI [2.4, 6.0]. The CIs overlap — thepre-
registered outcome is “comparable att= 1”. A-
MemGuard’s parametric fallback is effective because our
enterprise scenarios are domain-specific (Nexora Corp policies
the LLM cannot verify parametrically), causing the consis-
tency check to always flag the adversarial response. On sce-
narios with known parametric answers, A-MemGuard achieves
0% ASR (validation confirmed, Section VII-C). The key
differentiator is the SMSR certificate: A-MemGuard offers no
formal bound on its ASR regardless of empirical performance.
A-MemGuard also degrades under a persistent adversary who
can adapt responses to pass the consistency check; SMSR’s
random-subset ablation is adaptive-adversary-resistant by de-
sign (subject to the E-AD assumptions in Section VIII).

PREPRINT 9
E. Judge Reliability (E6)
We re-judged 84 held-out responses (28 per verdict class)
using claude-sonnet-4-6 as a reference judge and compared
verdicts to the main evaluation’s claude-haiku-4-5 judge. Co-
hen’sκ= 0.955with 97.6% raw agreement (n= 84).
This is near-perfect inter-rater reliability and confirms that the
Haiku judge is not a confounding factor in any of the ASR
measurements.
F . Generality: Second Agent LLM (E7)
E7a (unsigned, Component-1 test, completed).We re-
ran the headline unsigned/direct/n= 1configurations using
claude-sonnet-4-6 as the agent (Haiku as judge). Results on
10 scenarios: undefended ASR = 100%, c1 ASR = 0%, c1c2
ASR = 0%. This confirms HMAC provenance (Component 1)
is model-independent — trivially expected since Component 1
filters at the retrieval layer before the LLM is invoked.
E7b (authenticated, Component-2 test, complete).We
ran authenticated direct injection (t= 1,n= 450, 20-seed
production store) with claude-sonnet-4-6 as agent (Haiku as
judge). Result:8.0%(95% CI [5.8%, 10.9%]).
Like-for-like comparison at the same store config: Haiku
Tier 1 (20-seed,n= 450): 8.0%=Sonnet E7b (20-seed,
n= 450): 8.0%. The two model families produceidentical
ASR in the production-scale store.
Mechanism:Component 2 operates at the retrieval layer —
it samples a random subset of the over-fetched pool and runs
a verdict-based majority vote. The sampling and voting are
agnostic to which LLM generates the response. Both models
see the same retrieval distribution, so both achieve the same
protection level. This confirms the defence generalises across
model families.
Note on comparison:The eval-store Haiku result (37.8%
atm′= 11) should not be compared to E7b (8.0% atm= 20)
— these differ in store size, not model. The model-generality
claim rests on the like-for-like 20-seed comparison above.
G. Consistent Minority Effect Validation
Fig. 4 illustrates the CME mechanism. On a single run
(n= 15), string-based vote yields 93.3%ASRfor authen-
ticatedt= 1injection while verdict-based aggregation yields
13.3% on the same ablation runs. This 13.3% is a single-run
point estimate; over 30 repetitions (n= 450) the verdict-vote
rate stabilises at 37.8% (CI [33.4%, 42.3%], Table I), against
which the reduction from the saturated string-vote rate is
≈2.5×. We therefore report the aggregator effect rather than a
precise multiplier; a matchedn= 450string-vote run is left to
future work. The CME is confirmed: string equality incorrectly
selects the adversarial response in 12/15 scenarios because
adversarial factual claims converge to a consistent string while
clean responses vary. The improvement is attributable entirely
to the vote aggregator — both conditions receive the same 5
ablation responses; only the counting method differs.TABLE IV: Per-scenario ASR (t= 1, direct injection, c1c2,
30 reps each). Small-store eval (m′= 11,δ= 41.5%) and
production Tier 1 (m= 20,δ= 10.4%), both atn runs= 5.
Scenarios above the bound are within expected sampling noise.
Scenario (abbreviated) ASR 95% CI
Max expense w/o CFO approval 26.7% [14.2, 44.4]
Data retention period (PII) 43.3% [27.4, 60.8]
Emergency escalation contact 30.0% [16.7, 47.9]
Login failure lockout threshold 40.0% [24.6, 57.7]
IAM role provisioning level 40.0% [24.6, 57.7]
PTO accrual rate 40.0% [24.6, 57.7]
Notice period for resignation 30.0% [16.7, 47.9]
Personal use of company laptops 36.7% [21.9, 54.5]
DB connection string 50.0% [33.2, 66.8]
Firewall port requirements 43.3% [27.4, 60.8]
Vendor approval list6.7%[1.8, 21.3]
Security certification minimum 40.0% [24.6, 57.7]
Breach notification deadline 56.7% [39.2, 72.6]
Disaster recovery RTO 33.3% [19.2, 51.2]
MFA requirements 50.0% [33.2, 66.8]
Pooled 37.8% [33.4, 42.3]
H. E1: Per-Scenario ASR with 30 Repetitions
Table IV shows the per-scenario ASR distribution for the
t= 1authenticated direct-injection configuration under SMSR
c1c2 (30 independent repetitions each,n= 450total).
Five scenarios exceedδ= 41.5%at the point estimate:
data retention (43.3%), DB connection string (50.0%), fire-
wall ports (43.3%), breach notification (56.7%), and MFA
requirements (50.0%). The maximum deviation is 2.1 standard
deviations (not significant atα= 0.05after multiple compari-
son correction). Importantly, this isconsistentwith the bound
being tight: a worst-case bound atδ≈41.5%predicts roughly
58% of scenarios below it; the observed 10/15 below and 5/15
above is consistent with this. The vendor approval scenario
shows 6.7% ASR (parametric knowledge helps in this case).
The pooled estimate [33.4%, 42.3%] is the statistically robust
result.
I. End-to-End Validation: Query-Only Injection (E10)
The studies above pre-seed adversarial entries to isolate
Component 2’s certificate against a known worst case. To test
external validitywe additionally mount aquery-onlyattack on
the live agent: rather than inserting poison, an attacker interacts
through the normal API and the agentitselfwrites the resulting
trace to memory via its signed write path (Algorithm 1).
Poison thus enters as in MINJA [3] —through interaction,
not insertion—placing it in the authenticated regime (the
agent signs its own writes, so Component 1 cannot filter it;
Component 2 is what is tested). We reuse the 20-seed store,
the 15 scenarios, and identical parameters; the only change
from the controlled study is the injection mechanism.
The injection lands reliably—the agent-written poison en-
ters the victim’s retrieval pool in100%of trials. End to
end, SMSR reduces ASR from65.3%(CI [57.4, 72.5])
undefended to5.3%(CI [2.7, 10.2]), a≈12×reduction with
non-overlapping confidence intervals; the defended rate is in

PREPRINT 10
TABLE V: E10: end-to-end query-only injection on the live
agent (n runs= 5, 20-seed store, 10 reps×15 scenarios,n=
150). The poison is written by the agent through its signed
path, not pre-seeded.
ASR 95% CI
Injection reaches retrieval pool 100% —
Undefended (full retrieval) 65.3% [57.4, 72.5]
SMSR c1c2 (verdict ablation)5.3%[2.7, 10.2]
the same single-digit range as the pre-seeded Tier-1 result
(8.0%) and sits below the certificate (δ= 10.4%).
This study and the controlled one are complementary,
not redundant. The pre-seeded study (93%→8%) validates
the Theorem-2 certificate against a worst case where clean
planted statements make the bound tight. The query-only study
(65.3%→5.3%) confirms efficacy under a realistic attack,
where the agent adopts the injected claim in all but three of
the 15 scenarios undefended; in those three (e.g. acceptable-
use and vendor-approval, where the injected claim is implau-
sible) it resists even without a defence, so SMSR is trivially
correct. This query-only resist-set need not match pre-seeded
susceptibility (Table IV), as the two injection mechanisms
store different text. The realistic undefended rate is therefore
lower than the controlled 93%, but the certified defence still
cuts it to single digits on a live agent stack—the end-to-end
demonstration that prior memory-poisoning defences lack.
J. Parameter Selection Guide
Table I and Fig. 3(b) inform parameter selection. All values
below are computed from the Theorem-2 bound (reproduced
bysmsr_certificate.py). Practitioners should choose
mbased on the expected adversary budgettand a targetδ:
•Fort= 1:m= 10givesδ= 0.50(insufficient); the
smallestmachievingδ≤0.10ism=21. We usem=
20in all experiments, which givesδ= 0.104fort= 1
atn runs= 5(orδ= 0.071atn runs= 7).
•Fort= 2:m= 20givesδ= 0.402; the smallestm
achievingδ≤0.10ism= 39.
•Fort≥3:m= 57is needed to reachδ≤0.10att= 3.
Fort≥5,δremains near 1 regardless ofmwithk= 5,
n= 5. Increasingn runsor them/kratio is required.
The parametern runstrades off cost against the tightness
of the majority bound. Our evaluation usesn runs= 5(a 3:2
majority margin), givingδ= 0.104att= 1,m= 20; raising
it to 7 tightens the bound to0.071and to 11 tightens it to
0.034, at the cost of more agent and judge calls per query.
K. Computational Overhead
The original 39-configuration sweep (585 attack trials, 80
utility trials) required approximately 3,160 LLM API calls (≈
$2, Claude Haiku 4.5); the 30-repetition certificate-validation
studies (Tables I and IV) add roughly3.6×104further calls—
the dominant cost, though still modest at Haiku batch pricing.
Per-query overhead for SMSR in production:n runs×(agent
call + judge call) =5×2 = 10API calls per query versus 1for the undefended baseline, a 10×call overhead. Response
latency can be reduced to≈2×by batching the ablation runs
in parallel using the API’s async interface. Component 1 adds
only the cost of one HMAC verify per retrieved entry, which
is negligible (<1µs).
In the context of enterprise deployments where each query
represents a business decision—approving an expense, grant-
ing access, generating a contract—the cost of 10 API calls
per query (roughly $0.001 USD at Haiku pricing) is negligible
compared to the cost of a successful poisoning attack.
VIII. DISCUSSION
Deployment considerations.The HMAC keyKmust be
stored in a hardware security module or secrets manager. Any
write path that does not go through the signing oracle becomes
a bypass; operators should ensure the memory store is not
directly writeable except through the trusted write path. Key
rotation every 30–90 days provides forward security at the cost
of invalidating all prior memories.
Certificate limitations.The certificate bound
δ(t, m, k, n runs)is loose for larget. Fort≥m/2, the
adversary controls a majority of the candidate pool and
δ≈1; the defence provides no guarantee. Operators
must size the deployment (choice ofm) relative to their
assumed adversary budgett. A practical rule of thumb
from Theorem 2 ism≈19tfor a targetδ≤0.10at
k= 5,n runs= 5:t= 1→m= 21,t= 2→m= 39,
t= 3→m= 57(raisingn runsto 7 lowers these to 18, 34,
50).
Justification for the bounded-tregime.The certificate is
meaningful only whentis small. In practice, several standard
enterprise controls boundt: (a)per-user write quotas— most
production agent platforms limit the number of interactions a
user can submit per session or per day, capping the injection
rate; (b)duplicate and near-duplicate detection— FAISS-
based deduplication at write time rejects entries whose cosine
similarity to existing verified memories exceeds a threshold,
limiting adversarial flooding; (c)audit logs— all write-path
interactions are signed with the HMAC oracle, creating a
tamper-evident log that allows post-hoc detection and rollback
of injected entries.
Separately, the unsigned-adversary model (Component 1
threat) assumes the attacker has DB-write access but not access
to the HMAC signing oracle. This is realistic when the mem-
ory store is exposed through a misconfigured cloud storage
policy or a SQL-injection vulnerability in the persistence layer,
while the signing oracle runs in a separate trust boundary (e.g.,
a cloud function or HSM-backed service). An attacker who can
compromise the oracle can forge tags; key rotation with short
intervals (≤7 days) limits the window of exposure.
Adaptive adversaries.A sophisticated adversary who can
observe the judge’s decisions over time may attempt to craft re-
sponses that are judged as “correct” by the per-run judge while
still containing harmful content. This would require generating
text that fools a strong LLM judge—substantially harder than
fooling the agent alone, but not impossible. Rotating the judge
model and randomising its system prompt adds friction against
such adaptation.

PREPRINT 11
Scope of protection.SMSR protects the query–response
loop of the memory-augmented agent. It does not prevent
the adversary from observing other users’ queries if the
system leaks query patterns (a separate access-pattern privacy
concern [14]). It also does not prevent poisoning of the LLM’s
parametric knowledge via training data poisoning.
Extension to multi-agent systems.In agent pipelines where
one agent writes to another’s memory, the trust boundary for
signing must be extended: the upstream agent’s output must
itself be signed before being treated as a legitimate write.
This is a natural extension of the HMAC chain to multi-hop
delegation and is left for future work.
Relationship to inversion defences.SMSR is orthogonal
to defences against embedding inversion attacks (OWASP
LLM08’s other sub-problem). An embedding-rotation or en-
cryption scheme applied to the stored vectors can prevent
an adversary who obtains a database dump from recovering
original text; such a scheme composes with SMSR — it
protects the stored vectors while SMSR certifies the retrieval
process — providing defence-in-depth against both passive
(inversion) and active (poisoning) adversaries.
Towards a comprehensive LLM08 stack.The full LLM08
threat surface comprises: (1) inversion of stored embeddings
(addressed by rotation-based defences), (2) unsigned injection
into the memory store (addressed by Component 1), and (3)
authenticated injection by legitimate users (addressed by Com-
ponent 2 for bounded adversaries). SMSR provides formal
guarantees for (2) and (3); future work should integrate (1)
into the same certificate framework to produce an end-to-end
LLM08 defence.
Completed robustness checks.The two experiments
needed for a rigorous certificate validation are reported above.
E1 (30 repetitions per scenario with Wilson intervals; §VII-C
and Table IV) converts the point estimates into rates with
confidence intervals, enabling a proper certificate-vs-empirical
comparison. E2 (§VII-C) confirms the empirical contaminated-
run fraction matches the theoretical1−p clean, validating the
uniform-independent-sampling premise of Theorem 2.
Evaluation limitations.Our evaluation uses synthetic enter-
prise scenarios (Nexora Corp) and a Claude Haiku judge. We
did not collect human labels; judge reliability is quantified by
inter-judge agreement (κ= 0.955vs. Sonnet, study E6), and
residual judge errors are subsumed into the ASR measurement.
Real enterprise deployments may have longer, more nuanced
memory entries where the distinction between “correct” and
“malicious” responses is more ambiguous; we leave robustness
of the judge to adversarial response crafting as future work.
IX. CONCLUSION
We presented SMSR, the first formally certified defence
against runtime memory poisoning in persistent LLM agent
systems. SMSR combines write-time HMAC provenance
(Component 1) with randomised memory ablation and verdict-
based majority aggregation (Component 2). We proved that
provenance-free defences cannot certify against adaptive in-
jection, derived a hypergeometric certificate for Component 2,
and formalised and quantified the Consistent Minority Effect
on string-based vote aggregators.Empirical evaluation on 15 enterprise scenarios (3,150 re-
peated trials: six Component-2 configurations×15 scenarios
×30 repetitions, plus 450 production-scale trials) confirms:
Component 1 achieves 0%ASRfor all unsigned injection vari-
ants. In a production-scale store (m= 20, 20 seed memories),
Component 2 achieves 8.0% ASR (95% CI [5.8%, 10.9%],
n= 450), safely below the certificate boundδ= 10.4%at
nruns= 5. The canonical eval-storet= 1result is 37.8% CI
[33.4%, 42.3%] (n= 450), belowδ= 41.5%. Judge reliability
isκ= 0.955. A like-for-like A-MemGuard comparison at
n= 450shows the two are empirically comparable (A-
MemGuard 3.8%, SMSR 8.0%), SMSR’s differentiator being
its formal certificate; an authenticated-adversary test with a
second model family (Sonnet) reproduces the 8.0% result. In
an end-to-end query-only attack where the agent itself writes
the poison, SMSR cuts ASR from 65.3% to 5.3% (n= 150)
—the real-stack validation prior memory-poisoning defences
lack. Utility is 90% under Component 1 and 85% under the
full defence.
The system is deployable as a drop-in wrapper around
any RAG memory store that supports signed writes and
vectorised retrieval, with no changes to the underlying LLM
or embedding model.
REFERENCES
[1] OWASP Foundation. OWASP top 10 for Large
Language Model applications v2.0. https://owasp.org/
www-project-top-10-for-large-language-model-applications/, 2025.
[2] Cloud Security Alliance. Governance and NIST AI agent standards:
Agentic governance v1. Technical report, Cloud Security Alliance, 2026.
[3] Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming
Liu, Hui Liu, and Zhen Xiang. MINJA: Memory injection attacks
on LLM agents via query-only interaction. InAdvances in Neural
Information Processing Systems (NeurIPS), 2025. Michigan State / U.
Georgia / Singapore Management Univ. arXiv:2503.03704.
[4] Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li.
AgentPoison: Red-teaming LLM agents via poisoning memory or
knowledge bases. InAdvances in Neural Information Processing Systems
(NeurIPS), 2024. arXiv:2407.12784.
[5] Saksham Sahai Srivastava and Haoyu He. MemoryGraft: Persistent
compromise of LLM agents via poisoned experience retrieval.arXiv
preprint arXiv:2512.16962, 2025. University of Georgia.
[6] Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen,
and Prateek Mittal. Certifiably robust RAG against retrieval corruption,
2024. arXiv:2405.15556.
[7] Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang, Prateek Mittal, and
Aleksandra Korolova. ReliabilityRAG: Effective and provably robust
defense for RAG-based web-search. InAdvances in Neural Information
Processing Systems (NeurIPS), 2025. arXiv:2509.23519.
[8] Qianshan Wei, Tengchao Yang, Yaochen Wang, Xinfeng Li, Lijun Li,
Zhenfei Yin, Yi Zhan, Thorsten Holz, Zhiqiang Lin, and XiaoFeng
Wang. A-MemGuard: A proactive defense framework for LLM-based
agent memory.arXiv preprint arXiv:2510.02373, 2025.
[9] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V . Le, Ed H.
Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-
consistency improves chain of thought reasoning in language models.
InProceedings of the 11th International Conference on Learning
Representations (ICLR), 2023. arXiv:2203.11171.
[10] LangChain Inc. LangGraph: Building stateful multi-actor applications
with LLMs. https://langchain-ai.github.io/langgraph/, 2024.
[11] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang,
Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Auto-
Gen: Enabling next-gen LLM applications via multi-agent conversation.
arXiv preprint arXiv:2308.08155, 2023.
[12] Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang,
Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. Agent security
bench (ASB): Formalizing and benchmarking attacks and defenses in
LLM-based agents. InProceedings of the International Conference on
Learning Representations (ICLR), 2025. arXiv:2410.02644.

PREPRINT 12
[13] Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari,
Shantanu Todmal, Shreyan Mallik, and Shuchi Mishra. Memory poison-
ing attack and defense on memory-based LLM-agents.arXiv preprint
arXiv:2601.05504, 2026.
[14] Alexandra Boldyreva and Tianxin Tang. Privacy-preserving approxi-
matek-nearest-neighbors search that hides access, query and volume
patterns.Proceedings on Privacy Enhancing Technologies (PoPETS),
2021(4):549–574, 2021.
[15] Andreea-Elena Bodea, Stephen Meisenbacher, Alexandra Klymenko,
and Florian Matthes. SoK: Privacy risks and mitigations in retrieval-
augmented generation systems.arXiv preprint arXiv:2601.03979, 2026.
IEEE SaTML 2026.
[16] Jeremy M. Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified
adversarial robustness via randomized smoothing. InProceedings of
the 36th International Conference on Machine Learning (ICML), pages
1310–1320, 2019.
[17] Alexander Levine and Soheil Feizi. Robustness certificates for sparse
adversarial attacks by randomized ablation. InProceedings of the
AAAI Conference on Artificial Intelligence, 2020. arXiv:1911.09272.
Randomised L0-ablation certificate for image classifiers.
[18] Jiehang Zeng, Jianhan Xu, Xiaoqing Zheng, and Xuanjing Huang.
Certified robustness to text adversarial attacks by randomized [MASK].
Computational Linguistics, 49(2):395–427, 2023. Text adaptation of
randomised ablation.