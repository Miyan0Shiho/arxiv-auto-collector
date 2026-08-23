# Coverage Is Not Containment: A Fundamental Limit of Admission-Time Defenses Against Coordinated Poisoning of Vector Retrieval

**Authors**: Prashant Kumar Pathak, Tarun Kumar Sharma

**Published**: 2026-08-17 03:18:51

**PDF URL**: [https://arxiv.org/pdf/2608.16044v1](https://arxiv.org/pdf/2608.16044v1)

## Abstract
Retrieval-augmented generation (RAG) answers a question by retrieving passages from a vector store and trusting them as context, so anyone who can add documents can try to steer the answer. A recent, appealing defense filters poisoning at ingestion, rejecting any document that behaves like a hub. We show it -- and every ingestion-time filter -- is defeated by a coordinated adversary that injects a handful of individually unremarkable documents which together surround one target query and seize its top-k (on BGE-large / BEIR, m=10 documents take 10/10; 9.9/10 on a live HNSW index). The attack is not theoretical. Realized as ordinary fluent text and run end-to-end through a BGE-large + HNSW + Qwen2.5-7B pipeline, it makes the generator emit the attacker's planted claim in 88% of targets, versus 0% without the injection. And no admission-time defense stops it: at ingestion an attack cone is geometrically identical to a legitimate niche upload, so -- measuring this directly -- the strongest trained classifier, given every feature and thousands of examples, separates the two no better than chance, catching 4.2% of attacks at a 1% false-positive rate. We prove this limit for the entire class of ingestion-time statistics (any decision from documents and reference queries alone), and it reproduces -- and worsens -- across two corpora and five encoders. The one signal that separates an attack from legitimate niche ingestion -- a query's demand -- is invisible before retrieval, which is also the escape: a retrieval-time detector that observes demand catches 100% of the attacks at the same 1% false-positive rate. Coverage of the query space by an admission gate is not containment of coordinated poisoning; robust defense must move past the front door, to demand.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 1
Coverage Is Not Containment: A Fundamental
Limit of Admission-Time Defenses Against
Coordinated Poisoning of Vector Retrieval
Prashant Kumar Pathak and Tarun Kumar Sharma
Abstract—Retrieval-augmented generation (RAG) answers a
question by retrieving passages from a vector store and trusting
them as context, so anyone who can add documents can try to
steer the answer. A recent, appealing defense filters poisoning
at ingestion, rejecting any document that behaves like ahub.
We show it—andeveryingestion-time filter—is defeated by a
coordinated adversary that injects a handful ofindividually un-
remarkabledocuments which together surround one target query
and seize its top-k(on BGE-large / BEIR,m=10documents take
10/10;9.9/10on a live HNSW index).
The attack is not theoretical. Realized as ordinaryfluenttext
and run end-to-end through a BGE-large+HNSW+Qwen2.5-
7B pipeline, it makes the generator emit the attacker’s planted
claim in88%of targets, versus0%without the injection. And
no admission-time defense stops it: at ingestion an attack cone
is geometrically identical to a legitimate niche upload, so—
measuring this directly—the strongest trained classifier, given
every feature and thousands of examples, separates the twono
better than chance, catching4.2%of attacks at a1%false-positive
rate. We prove this limit for the entire class of ingestion-time
statistics (any decision from documents and reference queries
alone), and it reproduces—and worsens—across two corpora
and five encoders. The one signal that separates an attack from
legitimate niche ingestion—a query’sdemand—is invisible before
retrieval, which is also the escape: a retrieval-time detector that
observes demand catches100%of the attacks at the same1%
false-positive rate. Coverage of the query space by an admission
gate is not containment of coordinated poisoning; robust defense
must move past the front door, to demand.
Index Terms—Retrieval-augmented generation, vector
database security, corpus poisoning, hubness, admission control,
adaptive adversary, embedding anisotropy.
I. INTRODUCTION
Retrieval-augmented generation (RAG) [1], [2] has become
the dominant way to ground large language models (LLMs)
in external knowledge: a user query is embedded, the nearest
documents in a vector store are retrieved, and those passages
are placed in the model’s context as trusted evidence. This
architecture makes the vector store asecurity boundary. A
document that is retrieved for a query can steer the model’s
answer to that query—through indirect prompt injection, mis-
information, or biased evidence [7], [5], [6]. An adversary who
can insert documents into the store therefore has a powerful
lever over downstream generations.
Hubness—the well-documented tendency, in high-
dimensional spaces, for a few points to appear in thek-nearest-
P. K. Pathak is with Santa Clara, CA, USA (e-mail:
prashant.pathak@ieee.org).
T. K. Sharma is with Colonia, NJ, USA (e-mail: tarun.sharma@ieee.org).neighbour lists of disproportionately many queries [3]—
sharpens this lever. A single craftedhubdocument can be
retrieved across many unrelated queries, so one injected
record influences a large fraction of interactions [4]. The
natural defense is to detect and remove such hubs. Detection
after ingestion, however, leaves an exposure window between
a hub’s insertion and the next scan, and pays the cost of
repeated corpus-wide rescans.
A recent line of work [8] moves the control toadmission.
It maintains a set of sentinel queries and, for each candidate
document, computes its reverse-kNN countκ Sagainst the
sentinels; a document is rejected ifκ Sreaches a thresholdτ
calibrated to a small benign false-positive rate, so a hub
never enters the index. That work shows, perhaps surprisingly,
that a singleglobalthreshold suffices—domain-aware (per-
topic) refinement adds nothing—and explains it geometrically:
sentence embeddings areanisotropic, occupying a narrow
cone, so a document that is hub-like within a topic is hub-
like globally. Crucially, it evaluates a fixed set ofnon-adaptive
attacks and explicitly scopestargetedandcoordinatedattacks
out of scope.
This paper takes up exactly that scoped-out threat, and
finds a fundamental limit.We ask two questions.(i) Attack:
can a coordinated adversary poison a chosentargetquery
while every injected document is individually admissible?
(ii) Defense:can anyingestion-timecontrol—per-document or
collective, seeing only documents and sentinels—stop it at an
acceptable cost? Our answers are yes, and no (Proposition 1).
The key idea is that the per-document gate reasons about
documentsone at a time. A hub is caught because it is loud
on its own. But an adversary need not build a hub: it can
inject many documents that are each individually quiet—each
retrieved by too few sentinels to be rejected—yet thattogether
dominate one target query’s retrieval. By the very anisotropy
that makes the global gate work, the documents that achieve
this must live nearperipheralqueries—queries poorly covered
by the established sentinels—which is precisely the tight-
domain residual the gate cannot close. When the defender
responds with acollectivestatistic that looks for coordinated
bursts, the adversary tunes its attack into a regime that is
geometrically indistinguishable from legitimate bulk ingestion
of related documents. This indistinguishability, which we both
measure—the strongest trained classifier separates the attack
from a location-matched legitimate upload no better than
chance (Fig. 5)—and formalize (Proposition 1), is the crux
of a limit that persists at every practical false-positive rate
arXiv:2608.16044v1  [cs.CR]  17 Aug 2026

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 2
(Fig. 4) and that no ingestion-time defense in a natural class
escapes.
Contributions.
•A coordinated attack that reaches the output (§IV).
We formalize the coordinated adversary and showm
individually-admissible documents in a tight cone around
a target query seize its top-klinearly inm(m=10seizes
10/10on BGE-large / BEIR;9.9/10on a live HNSW
index), give a feasibility law tying attackability to query
centrality, and realize it asfluenttext that evades a
perplexity filter. End-to-end through a real RAG pipeline
(BGE+HNSW+Qwen2.5-7B) it flips the generated an-
swer to the attacker’s planted claim in88%of targets,
versus0%clean.
•A measured fundamental limit (§V, §VI).Two col-
lective defenses and the adaptive game leave a4.5/10
covert residual that persists ateveryachievable false-
positive rate. We prove (Proposition 1) this holds for
the entire class of ingestion-time statistics, and—the key
evidence—measureit: the strongest trained classifier,
given every feature, separates the attack from a location-
matched legitimate uploadno better than chance(4.2%
recall at1%FPR). It reproduces across two corpora and
five encoders.
•A constructive escape, and systems (§VII, §VIII).Be-
cause the distinguishing signal is retrieval-timedemand,
a detector that observes it catches100%of the attacks at
1%FPR—where the best admission-time detector catches
4.2%. Sub-points: the collective defense costs∼10%of
insert latency, and a per-shard view is blind to a burst
split across shards (motivating global consistency).
II. BACKGROUND ANDRELATEDWORK
RAG security and corpus poisoning.Because retrieved
passages are trusted, poisoning the retrieval corpus is a di-
rect attack on RAG. PoisonedRAG [5] and corpus-poisoning
attacks [6], [5] craft passages that are retrieved for target
questions and steer generation; backdoor and trigger attacks
plant passages activated by specific queries [20], [21]; and
indirect prompt injection [7] weaponizes retrieved content.
These build on the broader data-poisoning lineage [22], [23],
[24]. Two things distinguish our setting. First, these works
typically targetspecificquestion–answer pairs or optimize a
small number of passages against anundefendedretriever,
whereas we study a defense-aware adversary that must remain
admissible under an explicit ingestion-time gate and seeks to
dominate—not merely enter—a target query’s top-k. Second,
our construction is closer in spirit to a Sybil attack [36], in
which many individually-weak entities combine, than to a sin-
gle strong poison. Adversarial attacks on neural retrieval and
ranking craft passages that are retrieved or ranked highly [31],
[32]; we differ again in operating under an admission gate.
Defenses against retrieval poisoning.Defenses span the
RAG pipeline, and it is worth situating our limit against
each stage. (i)Ingestion-time filtering—the admission gate [8]
we study, and more generally any anomaly test applied as
documents arrive—is the earliest and cheapest place to act;our results show this stage cannot succeed against coordinated
poisoning at any acceptable false-positive rate. (ii)Retrieval-
timedefenses inspect a query’s retrieved set: robust aggrega-
tion isolates each passage and combines per-passage answers
so aminorityof poisoned passages cannot dominate, with
certifiable guarantees in RobustRAG [37], which inherits from
certified poisoning defenses based on bagging and partition
aggregation [38], [39] and randomized smoothing against label
flips and backdoors [40], [41]. These bound theinfluence
of poisoned passages once retrieved but rest on a minority
assumption that our coordinated attack violates by design—it
takesall10/10retrieved slots, not a minority—so aggrega-
tion alone is not a containment; a complementarydetection
signal is needed, which we supply as retrieval-timedemand
(§VIII). (iii)Provenance and source trustadmit converging
documents only from vetted sources, shifting the problem
from geometry to identity; this escapes our limit (it is not
a function of documents and sentinels alone) at the cost of an
identity/attestation infrastructure the open ingestion channels
above typically lack. (iv)Answer-timecorroboration cross-
checks the generated answer against diverse evidence. Our
contribution is to prove that the first, most attractive stage
is a dead end for coordinated poisoning, and to give the first
quantitative wedge between an ingestion-blind detector and a
demand-aware one (4.2%vs.100%recall, §VIII).
Hubness.Hubness in high-dimensional nearest-neighbour
search is long studied [3], with reduction methods [33] and,
more recently, adversarial exploitation: adversarial hubs [4]
craft a single multi-modal document retrieved for many
queries. The admission gate we attack [8] is designed exactly
against such broad hubs.
Admission-time control and its geometry.This paper is
the adversarial counterpart to our admission-time defense [8]:
where [8] shows a single global gate suffices againstbroad
hubs, we show that it—and the entire class of ingestion-time
defenses—is defeated bycoordinatedpoisoning. Concretely,
the gate of [8] rejects hubs at ingestion via a reverse-kNN
count against sentinel queries and shows a single global
threshold suffices, attributing this to embeddinganisotropy:
dense sentence embeddings occupy a narrow cone [9], [10],
[11], coupling topic-local and global visibility. We show the
same anisotropy that makes the gate sufficient against hubs
makes itfailagainst coordinated targeted poisoning.
Adaptive adversaries.A recurring lesson in security ML is
that defenses must be evaluated against adaptive attacks [12],
[13], [14]; static evaluations overstate robustness. Our con-
tribution is to carry this discipline into admission-time RAG
defenses and to show that adaptivity is not merely a stronger
attack but reveals a geometric limit.
Encoders and infrastructure.Dense retrieval [25], [26],
[27] and general-purpose text embeddings [15], [28], [29],
[16], benchmarked by MTEB [30] and BEIR [17], underpin
RAG; production stores index them with approximate-nearest-
neighbour structures [18] in systems such as FAISS [35] and
Milvus [34]. We use BGE-large [15] as the primary encoder
and HotFlip [19] for text realizability.

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 3
III. THREATMODEL ANDPROBLEMFORMULATION
System.A vector store holds unit-normalised embeddings
E(d)∈SD−1of documents under a fixed encoderE. A
queryqretrieves the top-kdocuments by cosine similarity
⟨E(d), q⟩. Lets k(q)denote thek-th largest similarity of any
corpus document toq(the “bar” to enterq’s top-k).
Definition 1 (Admission gate):The gate maintains sentinels
S={q 1, . . . , q n}with per-sentinel thresholdsτ i(thek-th-
NN similarity ofq iover the clean corpus). For a candidated
it computes the reverse-kNN countκ S(d) =|{i:⟨E(d), q i⟩>
τi}|and the hub rateh(d) =κ S(d)/n. Itadmitsdiffh(d)< θ,
whereθis calibrated so that the benign false-positive rate
Prd∼B[h(d)≥θ] =ϕ(we useϕ= 1%).
Adversary.White-box, consistent with [8]: it knowsE, the
gate,θ, and the sentinelsS. It injects a setA={d 1, . . . , d m}
of documents through the ingestion path. Its budget is the
number of documentsmand a per-document amplitude. Given
a target queryq∗, itsobjectiveis to maximize the number of
seized slots
J(A, q∗) ={i:⟨E(d i), q∗⟩> s k(q∗)},
i.e. how many ofq∗’s top-kresults are attacker documents,
subject toevery document being admitted:h(d i)< θfor all
i. We call a documentcovertif it is admitted and seizes a slot;
the attack succeeds if it seizes a large fraction ofk.
Centrality.Letµbe the (normalised) mean query direction.
Thecentralityof a query isc(q) =⟨q, µ⟩: high for queries
aligned with the bulk of the workload, low forperipheral
queries in sparsely-covered directions.
Threat realism and cost.The write-access assumption is
mild and matches deployed RAG: production stores ingest
continuously from partially-open channels—public wikis and
forums, crawled web pages, user uploads, customer-support
tickets, and collaborative knowledge bases—so an adversary
who can post to any indexed source injects documents without
compromising the store. The cost is small and, crucially,
independent of corpus size: seizingq∗’s entire top-kneeds
onlym≈kshort passages (§VI-A; ten fork=10), and the
samemdocuments suffice as the corpus grows, because both
admission and seizure are local toq∗’s neighbourhood (h(d)
depends on the sentinels, notN;s k(q∗)is a local density).
Coordinated insertion is therefore realistic rather than exotic:
unlike a single conspicuous hub, themdocuments are indi-
vidually unremarkable and can be introduced gradually, from
distinct accounts or sources, defeating rate-limiting and burst
heuristics—the Sybil structure of the attack [36]. We assume
the store applies the admission gate but no per-sourceidentity
orprovenancecheck (exactly the class-leaving defenses of
§VIII); text-level moderation such as a fluency/perplexity filter
lies outside the gate’s classDand, as §VI shows, does not
help, because the attack realizes as fluent natural-language
text. Finally, we grant the adversary white-box knowledge
(below): the limit we prove is a property of the ingestion
channel, so aweakeradversary faces only a harder version
of the same task, and our claims are conservative.
Evaluation setup.Following [8], our primary configura-
tion is BGE-large-en-v1.5 (D=1024), a100,000-documentAlgorithm 1Coordinated cone attack on targetq∗
1:input:targetq∗, budgetm, cone widthδ, axisµ, gateθ
2:β∗←min{β:h(normalize(q∗−βµ))< θ}▷off-axis
to admit
3:qoff←normalize(q∗−β∗µ)
4:fori= 1tomdo
5:g i←random unit vector orthogonal toq off
6:d i←normalize(q off+δ g i)
7:end for
8:return{d 1, . . . , d m}▷each admissible; each in
top-k(q∗)
corpus assembled from four BEIR collections (FiQA/TREC-
COVID/SciFact/NFCorpus) with10,200grounded queries and
n=5,570sentinels;k=10;θfrozen at1%FPR on a disjoint
5,000-document benign set. §VI adds a second, composition-
ally distinct general-web corpus. Headline results are averaged
over five random seeds (which re-draw the sampled targets,
the benign calibration batches, and the cone perturbations);
we report95%confidence intervals (Student-t).
IV. THECOORDINATEDPOISONINGATTACK
A. Single-document feasibility and its geometry
The most aggressive single poison forq∗is a document
d≈q∗: it is maximally similar toq∗(hence its top-1result)
and requires no optimization. Whether the gate admits it is
governed by geometry.
Observation 1 (Feasibility law):Ford=q∗, the hub rate
h(d)equals the fraction of sentinels withinq∗’s retrieval
neighbourhood. Under anisotropy, this fraction grows with
centralityc(q∗): a central query is close to many sentinels,
sod=q∗is loud and caught; a peripheral query is close to
few, sod=q∗is quiet andadmitted. Targeted poisoning is
therefore feasible exactly where the gate’s coverage is weakest.
We confirm this over all10,200queries. A single admitted
document poisons50.6%of target queries directly. Admis-
sibility is strongly concentrated at the periphery:60.9%of
low-centrality queries versus38.4%of high-centrality queries
admit the direct poison, and the rank correlation between
centrality and hub rate is+0.21. For the remaining (central)
queries, a small push off the global axis restores admissibility:
writingd(β) = normalize(q∗−βµ), the median smallest
admittingβis0.20, at which the poison still retains cosine
0.986toq∗. Off-axis evasion is nearly free.
B. The coordinated multi-slot attack
Seizing one slot dents the retrieved context; todominate
it, the adversary seizes many. It placesmdocuments in a
tight cone around the target (Alg. 1): a base directionq off(the
smallest off-axis push making the base admissible) plus small
lateral perturbations. Because eachd ihas cosine≈0.98to
q∗—far above the top-kbars k(q∗)≈0.68—each occupies a
distinct top-kslot, while remaining as quiet asq off.
Slots seized scale linearly with the budget (Fig. 1):
m=1→1,m=3→3,m=5→5, andm=10→10/10of the
top-k(9.96±0.03over five seeds), at99%all-admissible,

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 4
2 4 6 8 10
attacker budget m (documents)246810top-k slots seized
Coordinated attack scales linearly
slots seized
poison frac (×10)
Fig. 1. Coordinated attack:mindividually-admissible cone documents seize
top-kslots linearly;m=10seizes the entire top-10.
uniform across peripheral and central queries (the tiny median
β∗=0.05handles central queries).The per-document gate
provides essentially no protection against coordinated targeted
poisoning.
C. Text realizability
The attack so far is in embedding space (d iare vectors).
We realise the cone directions astextwith HotFlip-through-
BGE, optimizing a token sequence whose embedding aligns
withq∗. Realised documents reach mean cosine0.771to
the target and, as single documents, poison92%of targets
(⟨E(d), q∗⟩> s k) and evade the gate67%of the time.
The text is non-fluent (a known HotFlip trait) but a valid,
ingestible document; the embedding-space attack survives text
constraints, and the coordinated version composesmsuch
realised documents. Realization need not be adversarial at
all: §IV-E usesfluentnatural-language documents (the query’s
phrasing plus a planted claim) that seize and admit equally
well at benign-level perplexity.
D. Poisoning a real index
To rule out an artefact of exact geometry, we inject the
coordinated attack into a live HNSW index [18] of the
100,000-document corpus and query it. The attacker holds
9.9/10of theactually retrievedtop-k(median10; at least
half in100%of targets). Coordinated poisoning is real, not a
property of brute-force search.
E. End-to-end output harm
Seizing the retrieved context is only the mecha-
nism; the harm is what the generator emits. We close
the loop with a full RAG pipeline—BGE-large re-
triever over the100,000-document HNSW index feeding
Qwen2.5-7B-Instruct—and plant, per target query,
m=10attack documents carrying a unique canary claim (“the
correct answer isZQX-77341”), written as ordinary text (the
query’s terms followed by the claim) so their embeddings
land onq∗. These natural-text documents seize10/10of
the retrieved contextandpass the per-document gate (10/10
seized
/10admitted
/10answer-flip
(poisoned)answer-flip
(clean)0.00.20.40.60.81.0fraction10 10
88%
0%End-to-end: poisoning flips the answerFig. 2. End-to-end output harm. Fluent attack documents seize10/10of
the retrieved context and pass the per-document gate (10/10admitted); the
generator then emits the attacker’s planted claim in88%of targets, versus
0%with no injection.
admitted): each is relevant to a single query, not a broad hub.
Generating an answer from the retrieved passages, the model
emits the attacker’s planted claim in88%of targets, against
a0%clean baseline (the canary never appears without the
injection). Coordinated poisoning is not a retrieval curiosity:
it propagates through the gate, into the retrieved context, and
out as the model’s answer.
V. COLLECTIVEADMISSIONCONTROLS
Since each attack document is individually admissible, any
hope of an admission-time defense must becollective: examine
the admitted stream, not each document in isolation. We
calibrate every collective statistic, as the gate is, at a1%benign
false-positive rate—against the hardest legitimate confound,
a bulk upload ofMreal same-topic documents (a routine
ingestion event).
A. Near-duplicate-burst statistic
The coordinated attack is a tight cone: its documents are
mutually very similar. For a documentdin an admitted
batch, letsim(d)be the mean cosine to its top-Cco-admitted
documents. A tight cone (δ=0.10, mutual cosine≈0.99)
scores far above a legitimate topic batch. Calibrated at1%
FPR, this statistic catches the naive attack at100%recall.
B. The adaptive game
An adaptive adversary widens the cone—raisingδlowers
mutual similarity and thussim(d). Table I and Fig. 3 show
the game against the per-document gate and the burst statistic.
Tight cones (δ≤0.30) are caught; a middle bandδ≈0.4–
0.7evadesbothwhile each document keeps cosine≈0.92to
q∗—covertly seizing the entire top-k. Very wide cones (δ=1.0)
finally fall out of the top-k. There is a clean evasion window.
C. A retrieval-side co-retrieval statistic
The wide-cone documents evade thedocument-similarity
statistic, but they still all serveq∗: they co-activate a shared
sentinel neighbourhood. We addcor(d) =P
jMd,j(Aj−

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 5
TABLE I
THE ADAPTIVE GAME AGAINST THEcombinedDEFENSE(PER-DOCUMENT+DOC-SIMILARITY+CO-RETRIEVAL):COVERT TOP-kSLOTS BY CONE
WIDTHδ(ONE REPRESENTATIVE SEED;THE BEST-δRESIDUAL IS4.5±0.4OVER FIVE SEEDS).
δpoison evade per-doc evade sim. evade co-retr. covert slots
0.05–0.40 1.00 1.00 0.00—0.0
0.55 1.00 1.00 1.00 0.42 4.1
0.70 0.99 1.00 1.00 0.474.6
1.00 0.67 1.00 1.00 0.63 3.1
0.2 0.4 0.6 0.8 1.0
cone width 
0246810top-k slots / rate×10
Adaptive game vs. combined defense
covert slots
evade sim.evade co-retr.
poison (×10)
Fig. 3. The adaptive game vs. the combined defense. Tight cones are caught
by doc-similarity; the shaded band (δ≈0.5–0.75) evades every gate while
still poisoning.
Md,j), whereMis the batch membership matrix over
sentinels andAits column sums—the number of (other-
document, shared-sentinel) incidences. Combined with a
tighter-calibrated burst statistic, thecombineddefense cuts
the attack from10/10to a4.5±0.4/10covert residual
over five seeds (Table I shows one seed): tight cones caught
by similarity, most of the wide burst by co-retrieval—but a
residual atδ≈0.55–0.70survives.
VI. A PERSISTENTFUNDAMENTALLIMIT
A. The defender frontier
Can the operator tune the residual away by tightening the
collective thresholds? Only at a benign false-positive cost on
legitimate same-topic uploads. Figure 4 maps the frontier: the
4.5±0.4/10residual (five seeds) holds atevery achievable
benign FPR up to10%. It vanishes only when the co-retrieval
threshold collapses to zero—flagginganysentinel sharing—
whose true benign FPR is100%(it flags all legitimate topic
batches). Closing the coordinated attack is impossible at any
false-positive rate a production system would accept.
Robustness tok,m,n.The residual is not tuned to the
deployed point. Sweeping the attacker budget, seizure islinear
inm—mdocuments takemin(m, k)of the top-k(m=2,4,6
seize2,4,6slots;m≥ksaturates)—and the covert residual
scales with it. Acrossk∈ {5,10,20,50}the residual holds
(per-slot fraction0.44at the deployedk=10). And the residual
does not depend on an over-provisioned sentinel set:reducing
the sentinel countnfrom5,570to1,114raisesthe residual
100101102
achieved benign FPR (\%, log)01234covert slots (residual)
degenerate
(100\% FPR)Residual persists across all practical FPRFig. 4. Defender frontier: the covert residual is flat across all practical benign
FPRs and only vanishes at a degenerate100%-FPR threshold.
(0.44→0.88of the top-k), as a coarser sentinel cover weakens
the collective statistics—more sentinels help but never close
it.
B. A scoped indistinguishability limit
The residual is measured for the specific statistics we
constructed. We now argue it is a property of a wholeclass
of defenses, not of our choices. We define the class precisely
so the claim is scoped rather than universal.
Definition 2 (Ingestion-time defense classD):A defense in
Ddecides, for each documentdin an admitted batchW,
whether to flag it, using only the co-admitted embeddings
{E(d′) :d′∈W}and the sentinelsS. It doesnotobserve
the target query or user demand for a topic—at ingestion time
no such query has been issued.
The per-document gate and both collective statistics lie in
D. The following makes precise why no member escapes.
Proposition 1 (Indistinguishability):Fix a batch sizem.
LetA δbe the law on batchesW= (E(d 1), . . . , E(d m))∈
(Sd−1)mproduced by the coordinated attack at cone width
δ, andBthe law on batches ofmlegitimate same-topic
documents. EachD∈ Dinduces a measurable, possibly
randomized batch testφ D(W, S)∈ {flag,pass}of the co-
admitted embeddings and sentinels; writerecall Aδ(D) =
PrW∼A δ[φD= flag]andFPR B(D) = Pr W∼B[φD= flag].
Then for everyD∈ Dand everyδ,
recall Aδ(D)−FPR B(D)≤TV(A δ,B).
Define the batchoverlapρ δ:= 1−TV(A δ,B)∈[0,1], so
thatρ δ≈1⇐⇒TV(A δ,B)≈0(near-identical batch laws).

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 6
An adversary choosingδ∗= arg max δρδ= arg min δTV
forces everyD∈ Dontorecall≤FPR + 
1−max δρδ
:
noingestion-time defense both catches the attack (recall→1)
and preserves legitimate ingestion (FPR→0) unless the max-
imal overlapmax δρδis bounded away from1—equivalently,
unlessmin δTVis large.
Scope.Proposition 1 boundsonlythe classDof ingestion-
blind defenses—those deciding from co-admitted documents
and sentinels. It makesnoclaim about defenses that observe
retrieval-timedemandor documentprovenance; those escape
the bound by construction (they are not functions of(W, S)),
and the demand-aware detector of §VIII is exactly such an
escape. The result is thus “admission-time defense is a dead
end for coordinated poisoning,” not “coordinated poisoning is
undetectable.”
Assumptions.(i)Dobserves only the co-admitted embeddings
and sentinels (Def. 2), not retrieval-time demand; (ii) attack
and benign batches are compared at a common sizem; (iii)φ D
is a measurable function of(W, S). Aper-documentrule
(Def. 2) lifts to such a batch test by flaggingWwhenever
it flags any co-admitted document, so the per-document gate
and the collective statistics are all special cases.
Proof sketch.φ Dis a (possibly randomized) map from(W, S)
to{flag,pass}, so its flag event is a function of that input
alone. By the data-processing inequality—equivalently, by the
Neyman–Pearson lemma, no test separates two distributions
better than their total-variation distance—the flag probability
underA δand underBdiffer by at mostTV(A δ,B), which is
the stated bound. The adversary is free to pickδ, giving the
minδTV(max δρδ).□
The frontier of §VI-A is exactly this optimization for the
strongest statistics we could construct: the covert residual stays
flat until the threshold degenerates—the empirical signature
ofmax δρδ≈1, equivalentlymin δTV(A δ,B)≈0: the best
admission-time test we could build cannot pushrecall−FPR
above≈0. The next subsection explainsgeometricallywhy the
overlap is near-total (ρ≈1) foranyD∈ D, and §VI-E shows
the effect strengthens on a second corpus. We stress the scope:
Proposition 1 concerns the classDof ingestion-blind defenses.
It doesnotpreclude defenses that leave the class—observing
retrieval-timedemandor documentprovenance(§VIII); those
are exactly where we point.
C. Why the overlap is near-total: the geometry
The bound in Proposition 1 bites only when the overlap
it leaves,max δρδ= 1−min δTV(A δ,B), is near-total; we
argueρ≈1for anyD∈ D. To a defense inD, a wide
adversarial cone near a peripheral query and a legitimate bulk
upload of same-topic documents are the same object: a group
of mutually similar documents converging on a direction the
established sentinels under-cover. This is the very anisotropy
that makes a single global gate sufficient against hubs [8]—
topic-local structure is globally coupled—and the same geom-
etry makes coordinated targeted poisoning look like benign
topic ingestion. A defense inDcannot use the one signal
that would separate them—whether the converging documents
weresolicitedby real user demand for that topic—because atingestion that demand has not yet been observed.Coverage of
the query space by a gate is not containment.
D. Can a learned detector separate them?
Proposition 1 reduces the question to whether the overlap
ρis near-total; §VI-A evidenced it for our two statistics and
the geometry argues it in general. We nowmeasureit against
the strongest learned adversary a defender could deploy. Be-
cause a binary classifier’s separating advantage lower-bounds
TV(A δ,B), the best learned member ofDupper-bounds
how wellanyingestion-time statistic can separate the two
batch laws—a classifier two-sample test. We train a gradient-
boosted classifier on a comprehensive set of class-Dbatch
features—pairwise-cosine burst moments, per-document hub
ratehand co-retrievalcor, thesimstatistic, the batch cone
geometry (centroid norm and covariance eigenspectrum), and
sentinel proximity—i.e. every signal a collective admission-
time defense could compute, on thousands of attack and
benign batches. A supervised gradient-boosted classifier over
these features subsumes off-the-shelf outlier detectors (iso-
lation forest, LOF) applied to the same batch, so it is the
strongest member ofDwe can construct.
The benign comparison must be location-matched.
Against batches ofestablishedcorpus content the classifier
separates the attack perfectly (AUC1.00), but this only dis-
tinguishesperipheralfromcentral: a coordinated cone must
occupy a sentinel-under-covered direction to clear the per-
document gate, and established content does not. The question
Proposition 1 poses is whether the attack is separable from
alegitimate niche upload at the same admissible location.
We therefore also buildadmissible uploads—tight real-topic
batches placed at admissible off-axis directions, exactly as a
legitimate uploader of niche content would produce.
Result.The classifier separates the attack from natural-
location content—random, tightest, and low-coverage niche
topics—at AUC1.00, but from location-matched admissible
uploads at AUC0.46: chance (Fig. 5). Calibrated to a1%
false-positive rate on a realistic legitimate stream thatincludes
such uploads, it catches only4.2%of attacks while flagging
the legitimate uploads at3.8%—the same rate; it cannot tell
them apart. An ablation localizes why: the attack’sgeometry
is not a usable signal—a geometry-matched cone (a real tight
topic batch rotated onto the target) has, by construction, the
same rotation-invariant geometry (pairwise cosines, covari-
ance spectrum) as its unrotated source, so a geometry-only
detector cannot separate them; the entire separation comes
from the sentinel-relative features (AUC1.0), i.e.location—a
centrality signal, not an attack signal. The result is robust to
realization: a HotFlip-realized coordinated attack (real token
sequences whose BGE embeddings land near the target, mean
cos(d, q∗) = 0.77, seizing9.2/10slots and7.0/10covertly)
is likewise indistinguishable from location-matched legitimate
content. This is the measured form ofmin δTV≈0: the
strongest learned admission-time statistic, given every feature
and thousands of examples, achieves advantage≈0against
the fair comparison—and it answers the standing objection
“a smarter collective statistic would close the residual” by
construction.

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 7
same-topic
(central)tight topic
(natural)niche topic
(natural)admissible upload
(loc.-matched)0.00.20.40.60.81.0classifier AUC (attack vs.)chance1.00 1.00 1.00
0.46Only the location-matched comparison is at chance
Fig. 5. Classifier two-sample test. The strongest learned class-Ddetector
separates the coordinated attack from natural-location content at AUC1.0,
but from alocation-matchedlegitimate niche upload at AUC0.46(chance):
at a1%-FPR threshold it flags attacks and legitimate admissible uploads at
the same∼4%rate.
single-doc
admitcoord.
slots/10combined
residual/100.00.20.40.60.81.0fractionReproduces on both corpora
finance/bio
general-web
Fig. 6. The attack and residual reproduce on both corpora and are worse on
general-web.
E. Generalisation
The attack and the persistent residual reproduce on a second,
compositionally distinct general-web corpus, and areworse
there (Fig. 6): single-document admission89.9%(vs.50.6%)
and a combined-defense residual of7.8±0.4/10(vs.4.5±
0.4/10), five seeds. A more isotropic, broadly-sampled corpus
offers the adversarymoreperipheral directions, not fewer. The
limit is not an artefact of one corpus.
F . The residual persists across encoders
Proposition 1’s strength rests on the overlap term being near-
total, a property of the embedding anisotropy the gate exploits.
To test that the residual is fundamental across the anisotropy
spectrum—not an artefact of BGE-large—we rebuild the entire
pipeline independently for five encoders spanning a≈12×
range of anisotropy (mean pairwise query cosine0.061to
0.740)1: MiniLM-L6, BGE-base, BGE-large, GTE-large, and
1We measure anisotropy as the mean pairwise cosine of query embeddings;
[8] reports the mean per-topic-centroid-to-global-centroid cosine for the same
encoders, which is numerically larger but monotone in ours (both increase
from MiniLM to E5-large).TABLE II
FIVE-ENCODER SWEEP(FIVE SEEDS EACH). THE COMBINED-DEFENSE
COVERT RESIDUAL PERSISTS ACROSS A≈12×ANISOTROPY RANGE;
“ADMIT”IS THE SINGLE-DOCUMENT ADMISSION RATE.
Encoderdaniso. admit coord. residual
MiniLM-L6 384 0.061 0.61 9.688.32±0.18
BGE-base 768 0.364 0.74 9.968.66±0.36
BGE-large 1024 0.416 0.69 9.988.80±0.25
GTE-large 1024 0.719 0.78 10.07.61±0.48
E5-large 1024 0.740 0.09 10.09.33±0.24
0.2 0.4 0.6
encoder anisotropy (mean pairwise query cosine)0246810top-k slots seized /10
minilmbge-B bge-L
gte-Le5-LResidual persists across the encoder range
coord. slots (per-doc gate)
combined residual
Fig. 7. The combined-defense covert residual stays7.6–9.3/10across five
encoders spanning a≈12×anisotropy range; it does not shrink toward the
isotropic end.
E5-large. For each we re-encode the corpus, queries, and sen-
tinels and construct a fresh1%-FPP operating point (sentinels
=held-out query half+per-encoderk-means topic centroids;
benign same-topic batches for the collective thresholds), then
measure the combined-defense covert residual over five seeds.
The residual persists on every encoder—7.6to9.3/10(Ta-
ble II, Fig. 7)—so the near-total overlap, and with it the
persistent limit, is a regularity of anisotropic retrieval geometry
rather than a single-encoder effect. It does not shrink toward
the isotropic end (MiniLM, the least anisotropic, still leaks
8.3/10), and E5-large—which gates single documentshardest,
admitting only9.1%—still leaves thelargestresidual (9.3/10):
low single-document admissibility is not containment. (Oper-
ating points are rebuilt per encoder, so absolute residuals differ
from the4.5/10headline; the claim is persistence, not the
value.)
VII. SYSTEMSCONSIDERATIONS
Deployability and overhead.The collective statistics are
cheap: on a live HNSW index, computingsimandcorper
admitted document costs∼10%of the HNSW insert time
(0.29ms vs.2.97ms per document in our measurements), so
the defense is deployable inline on the write path. Its cost is
anM-dimensional sentinel comparison and isindependent of
corpus sizeN—the property that makes the underlying gate
scale [8]—so it adds a fixed per-write overhead as the store
grows to millions of documents. The point of the paper is that
this cheap, scalable defensestilldoes not close the attack: the
limit is one of information, not cost.

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 8
2 4 6 8 10
number of shards S0.000.250.500.751.00per-shard collective recall
Per-shard defense blind to split burst
Fig. 8. A per-shard collective defense is evaded once the burst is split so≤1
attack document lands per shard; a global consistent view is required.
Operational integration and trade-offs.An admission
gate sits at the ingestion API, before the vector is written,
and returns an admit/quarantine decision synchronously. The
collectivestatistics, however, need the batch of co-admitted
documents, so a deployment either buffers a short ingestion
window (adding write latency) or evaluates asynchronously
(opening a bounded exposure window before a flagged burst is
quarantined)—a latency-versus-exposure trade-off that shard-
ing only sharpens (below). The retrieval-time detector of
§VIII, by contrast, integrates on thereadpath as a monitor: it
adds no write-path latency, but consumes the query workload
the ingestion gate is denied. The practical implication is not
to choose one stage but to layer them—a cheap ingestion
gate against broad hubs [8] and a demand-aware retrieval-time
monitor against coordinated cones—since each is blind exactly
where the other sees.
Sharded blind spot.Production stores are sharded. If
each shard runs the collective defense overonly its own
admissions, the adversary splits them-document burst across
shards; once≤1attack document lands per shard, no shard
sees a burst and the collective statistic is blind (Fig. 8). A
global view catches the tight cone; a per-shard view does
not. Restoring the collective defense under sharding therefore
requires aglobally consistentview of the admission stream—a
distributed-consistency problem we leave to future work, and
one that itself trades detection latency against an exposure
window.
VIII. DISCUSSION
What a real defense must do.Our results argue that
no ingestion-time filter—per-document or collective—contains
coordinated poisoning, because at ingestion the defender
cannot distinguish an adversarial cone from legitimate topic
demand. This suggests defense must movepastadmission:
(i)provenance / source trust, admitting converging documents
only from trusted sources, which shifts the problem from
geometry to identity; (ii)retrieval-timeanomaly detection
that flags when a query’s top-kis dominated by a single
tight, recently-ingested cluster (using demand that admission
time lacks); and (iii)answer-timecorroboration across diverse
evidence. Each moves the trust boundary and carries its own
cost; none is an admission gate.
admission-time(class )
retrieval-time
(demand)0.00.20.40.60.81.0attack recall @ 1% FPR4%100%The escape: observe demandFig. 9. The escape. Against a fair benign stream that includes legitimate
fresh uploads, the strongest ingestion-blind (classD) detector catches4.2%
of attacks at a1%false-positive rate; a retrieval-time detector that observes
demand catches100%at the same rate.
A constructive escape: retrieval-time demand.That
the limit is a property ofingestion-blindness—not of the
problem—is not just an argument: we implement option (ii)
and it works. The detector combines two signals a document-
and-sentinel test structurally lacks:recency(the fraction of a
query’s top-kingested within a recent window) anddemand
concentration(how few distinct workload queries retrieve the
same fresh cluster). The discriminator is that a legitimate fresh
topic upload is recent but retrieved bymanyorganic queries
about that topic, whereas the attack cone is recent but retrieved
essentially only by its single target—“no organic demand,” a
signal a class-Ddefense cannot see. On the same operating
point, injecting the coordinated attack against300targets
alongside300legitimatefresh topic uploads, the detector
achieves100%recall at a1%false-positive rate(Fig. 9),
correctly leaving unflagged the714benign queries that re-
trieved legitimate fresh content—versus4.2%recall for the
strongestadmission-timedetector (§VI-D). The4.2%→100%
jump is the whole thesis in one number: the attack is invisible
to any function of documents and sentinels, and visible the
moment the defender observes demand.
Where it fails, and the adaptive response.The escape is
not a finished defense; its signal weakens where the target
has genuine organic demand. An adversary can (a) aim at
popular queries and accept partial seizure, (b) spread the
payload across a small set of demanded queries so no single
cluster looks query-specific, (c) inject slowly so the cluster
ages out of the recency window, or (d) issue synthetic queries
to manufacture demand for its own cluster. Robust temporal
demand modeling—separating organic from injected demand
over adaptive windows—is the natural next problem and inher-
its its own detection game.Operationally, the detector runs as
a monitor on thereadpath: it needs read access to the recent
query workload and ingestion timestamps, a sliding window,
and per-cluster bookkeeping, and costs a periodic pass over
recent retrievals rather than any change to the write path.
This is exactly the demand signal a sharded, ingestion-only
store discards (§VII), which is why containing coordinated
poisoning is, at bottom, a question ofwho observes query
demand, not of how documents are filtered.

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 9
Responsible disclosure.The attacked defense is a research
proposal, not a deployed product; we nonetheless coordinate
with its authors. We release no turnkey exploit; the artefact re-
produces the scientific claims on public corpora. The practical
takeaway for practitioners deploying admission-style hubness
filters is defensive: such filters should not be relied upon
against targeted or coordinated poisoning, and should be paired
with provenance and retrieval-time controls.
IX. LIMITATIONS
Scope of the limit.Proposition 1 is ascopedindistinguisha-
bility result for the classDof ingestion-blind defenses; its
strength rests on the overlapmax δρδ= 1−min δTV(A δ,B)
being near-total (equivalentlymin δTV≈0), which we estab-
lish empirically (the frontier) and argue geometrically. That
this overlap stays near-total across the anisotropy spectrum
is supported by the five-encoder sweep of §VI-F (residual
7.6–9.3/10from the least- to the most-anisotropic encoder);
extending it toallencoders andallbenign-ingestion models
remains future work, though the two-corpus and five-encoder
evidence and the geometric argument point that way.Attack
realism.A fluency/perplexity pre-filter is an ingestion-time
controloutsideD(it reads token distributions, not the reverse-
kNN geometry). It does not help: while HotFlip text is high-
perplexity (GPT-2 median3.2×104, flagged100%by a1%-
FPR filter), the attack realizes just as well asfluentnatural-
language documents—the query’s phrasing plus a planted
claim—whose perplexity (median33) is indistinguishable
from benign corpus text (median41), so a perplexity filter flags
them at0%, yet each still poisons and admits. These fluent
documents are exactly the ones used in the end-to-end study
(§IV-E), in which the generator emits the planted claim in88%
of targets.Systems.A multi-node cluster deployment (beyond
the single-node sharded analysis of §VII) remains future
work.External validity.Our evaluation is a static snapshot
of a store.Dynamiccorpora—continual ingestion, deletion,
and drift—let a defender re-calibrateθand the sentinels but
also give the adversary a moving, less-monitored target; how
the limit interacts with corpus dynamics is open. We study
English text encoders:multilingualand domain-specialized
embeddings have their own anisotropy structure, and while
our five-encoder sweep spans a wide anisotropy range (0.06–
0.74) we do not test them directly. Foundation models evolve
quickly; a materially different geometry (far more isotropic,
or a non-cosine similarity) could change the constants, though
§VI-E indicatesmoreisotropy favours the attacker. Finally, we
study single-vector dense retrieval;alternative architectures—
late-interaction (ColBERT), learned-sparse, and hybrid dense–
sparse retrieval—aggregate evidence differently, and whether
coordinated admission-time poisoning transfers to them is an
important open question.
X. CONCLUSION
An admission gate that covers the query space stops broad
hubs but cannot contain a coordinated, low-amplitude adver-
sary that poisons a target query with individually-admissibledocuments. The failure is geometric, not statistical: the adver-
sarial cone and a legitimate topic batch share the embedding
anisotropy the gate relies on, so no ingestion-time observer
separates them at an acceptable false-positive rate. Defending
coordinated poisoning of vector retrieval must move beyond
admission time.
REFERENCES
[1] P. Lewis et al., “Retrieval-augmented generation for knowledge-
intensive NLP tasks,” inNeurIPS, 2020.
[2] K. Guu et al., “REALM: Retrieval-augmented language model
pre-training,” inICML, 2020.
[3] M. Radovanovi ´c, A. Nanopoulos, and M. Ivanovi ´c, “Hubs in
space: Popular nearest neighbors in high-dimensional data,”
JMLR, vol. 11, pp. 2487–2531, 2010.
[4] T. Zhang, F. Suya, R. Jha, C. Zhang, and V . Shmatikov,
“Adversarial hubness in multi-modal retrieval,” inIEEE S&P,
2026.
[5] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowl-
edge corruption attacks to retrieval-augmented generation,” in
USENIX Security, 2025.
[6] Z. Zhong, Z. Huang, A. Wettig, and D. Chen, “Poisoning
retrieval corpora by injecting adversarial passages,” inEMNLP,
2023.
[7] K. Greshake et al., “Not what you’ve signed up for: Compromis-
ing real-world LLM-integrated applications with indirect prompt
injection,” inAISec, 2023.
[8] P. K. Pathak and T. K. Sharma, “When Global Gating Is Enough:
Admission-Time Hubness Control in Anisotropic Vector Re-
trieval Systems,”Computers & Security, under revision, 2026.
[9] K. Ethayarajh, “How contextual are contextualized word repre-
sentations?” inEMNLP, 2019.
[10] J. Gao et al., “Representation degeneration problem in training
natural language generation models,” inICLR, 2019.
[11] J. Mu and P. Viswanath, “All-but-the-top: Simple and effective
postprocessing for word representations,” inICLR, 2018.
[12] N. Carlini and D. Wagner, “Towards evaluating the robustness
of neural networks,” inIEEE S&P, 2017.
[13] A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients
give a false sense of security,” inICML, 2018.
[14] F. Tram `er, N. Carlini, W. Brendel, and A. Madry, “On adaptive
attacks to adversarial example defenses,” inNeurIPS, 2020.
[15] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-
Y . Nie, “C-Pack: Packed resources for general Chinese embed-
dings” (BGE), inSIGIR, 2024.
[16] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence em-
beddings using Siamese BERT-networks,” inEMNLP, 2019.
[17] N. Thakur, N. Reimers, A. R ¨uckl´e, A. Srivastava, and
I. Gurevych, “BEIR: A heterogeneous benchmark for zero-
shot evaluation of information retrieval models,” inNeurIPS
Datasets, 2021.
[18] Yu. A. Malkov and D. A. Yashunin, “Efficient and robust ap-
proximate nearest neighbor search using hierarchical navigable
small world graphs,”IEEE TPAMI, vol. 42, no. 4, 2020.
[19] J. Ebrahimi, A. Rao, D. Lowd, and D. Dou, “HotFlip: White-
box adversarial examples for text classification,” inACL, 2018.
[20] H. Chaudhari, G. Severi, J. Abascal, M. Jagielski, C. A.
Choquette-Choo, M. Nasr, C. Nita-Rotaru, and A. Oprea, “Phan-
tom: General trigger attacks on retrieval augmented language
generation,” arXiv:2405.20485, 2024.
[21] J. Xue, M. Zheng, Y . Hu, F. Liu, X. Chen, and Q. Lou,
“BadRAG: Identifying vulnerabilities in retrieval augmented
generation of large language models,” arXiv:2406.00083, 2024.
[22] B. Biggio, B. Nelson, and P. Laskov, “Poisoning attacks against
support vector machines,” inICML, 2012.
[23] J. Steinhardt, P. W. Koh, and P. Liang, “Certified defenses for
data poisoning attacks,” inNeurIPS, 2017.

IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY 10
[24] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer,
T. Dumitras, and T. Goldstein, “Poison frogs! Targeted clean-
label poisoning attacks on neural networks,” inNeurIPS, 2018.
[25] V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu, S. Edunov,
D. Chen, and W.-t. Yih, “Dense passage retrieval for open-
domain question answering,” inEMNLP, 2020.
[26] O. Khattab and M. Zaharia, “ColBERT: Efficient and effective
passage search via contextualized late interaction over BERT,”
inSIGIR, 2020.
[27] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski,
A. Joulin, and E. Grave, “Unsupervised dense information
retrieval with contrastive learning,”TMLR, 2022.
[28] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Ma-
jumder, and F. Wei, “Text embeddings by weakly-supervised
contrastive pre-training,” arXiv:2212.03533, 2022.
[29] Z. Li, X. Zhang, Y . Zhang, D. Long, P. Xie, and M. Zhang,
“Towards general text embeddings with multi-stage contrastive
learning,” arXiv:2308.03281, 2023.
[30] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “MTEB:
Massive text embedding benchmark,” inEACL, 2023.
[31] C. Song, A. M. Rush, and V . Shmatikov, “Adversarial semantic
collisions,” inEMNLP, 2020.
[32] J. Liu, Y . Kang, D. Tang, K. Song, C. Sun, X. Wang, W. Lu,
and X. Liu, “Order-disorder: Imitation adversarial attacks for
black-box neural ranking models,” inACM CCS, 2022.
[33] D. Schnitzer, A. Flexer, M. Schedl, and G. Widmer, “Local and
global scaling reduce hubs in space,”JMLR, vol. 13, pp. 2871–
2902, 2012.
[34] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu et al., “Milvus: A purpose-
built vector data management system,” inACM SIGMOD, 2021.
[35] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity
search with GPUs,”IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–
547, 2021.
[36] J. R. Douceur, “The Sybil attack,” inIPTPS, 2002.
[37] C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, and
P. Mittal, “Certifiably robust RAG against retrieval corruption,”
arXiv:2405.15556, 2024.
[38] J. Jia, X. Cao, and N. Z. Gong, “Intrinsic certified robustness
of bagging against data poisoning attacks,” inAAAI, 2021.
[39] A. Levine and S. Feizi, “Deep partition aggregation: Provable
defenses against general poisoning attacks,” inICLR, 2021.
[40] E. Rosenfeld, E. Winston, P. Ravikumar, and J. Z. Kolter,
“Certified robustness to label-flipping attacks via randomized
smoothing,” inICML, 2020.
[41] M. Weber, X. Xu, B. Karla ˇs, C. Zhang, and B. Li, “RAB:
Provable robustness against backdoor attacks,” inIEEE S&P,
2023.