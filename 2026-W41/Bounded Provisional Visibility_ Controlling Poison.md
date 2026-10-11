# Bounded Provisional Visibility: Controlling Poisoning Exposure in Continuously Ingested RAG Vector Stores

**Authors**: Chuhong Xu, Lu Yi, Gangzhen Qian, Kainan Zhou, Zhaoyi Li, Hang Xiao

**Published**: 2026-10-05 05:28:57

**PDF URL**: [https://arxiv.org/pdf/2610.05826v1](https://arxiv.org/pdf/2610.05826v1)

## Abstract
Continuous ingestion can expose new retrieval-augmented generation (RAG) content to retrieval before vetting completes, creating a temporal attack surface that conventional admission decisions do not capture. We present a fail-closed provisional-visibility protocol that admits new content under a deadline, hides items whose verification has not committed in time, and supports lineage-scoped containment. The protocol bounds unvetted exposure by the configured visibility budget plus query-path enforcement delay; poisoning exposure remains conditional on verifier correctness. We evaluate the design on five workloads of encoded natural-language documents using an exact backend and Milvus. Under verifier backlog, the deadline reduced median poisoned retrievals from 34 (29--34) to 7 (6--7) relative to asynchronous admission without a deadline, with the same reduction in displaced clean results. Verify-before-visible avoided provisional exposure but delayed clean first visibility to 6.3 s, whereas provisional admission made content visible within 3 ms; expired clean items could nevertheless experience temporary availability gaps. Replaying decisions from a recipe-specific detector illustrated the protocol boundary: false promotions left poison visible, while false refusals excluded clean content from retrieval. Standalone Milvus tests characterized enforcement delay and concurrent HNSW retrieval on one node. These results clarify which guarantees come from the protocol and which outcomes depend on detector quality, while quantifying the freshness and availability trade-offs.

## Full Text


<!-- PDF content starts -->

Bounded Provisional Visibility: Controlling
Poisoning Exposure in Continuously Ingested RAG
Vector Stores
Chuhong Xu
Sony Corporate of America
San Jose, USA
chuhong.xu@sony.comLu Yi
Google LLC
Mountain View, USA
annabelyi@google.comGangzhen Qian
Google LLC
Mountain View, USA
irisqian@google.com
Kainan Zhou
Google LLC
Mountain View, USA
zhoumark@google.comZhaoyi Li
Independent Researcher
Mountain View, USA
lzy9776@gmail.comHang Xiao
Fortinet, Inc.
Sunnyvale, USA
xhang@fortinet.com
Abstract —Continuous ingestion can expose new retrieval-
augmented generation (RAG) content to retrieval before vetting
completes, creating a temporal attack surface that conventional
admission decisions do not capture. We present a fail-closed
provisional-visibility protocol that admits new content under a
deadline, hides items whose veriﬁcation has not committed in
time, and supports lineage-scoped containment. The protocol
bounds unvetted exposure by the conﬁgured visibility budget
plus query-path enforcement delay; poisoning exposure remains
conditional on veriﬁer correctness. We evaluate the design on
ﬁve workloads of encoded natural-language documents using an
exact backend and Milvus. Under veriﬁer backlog, the deadline
reduced median poisoned retrievals from 34 (29–34) to 7 (6–7)
relative to asynchronous admission without a deadline, with the
same reduction in displaced clean results. Verify-before-visible
avoided provisional exposure but delayed clean ﬁrst visibility to
6.3 s, whereas provisional admission made content visible within
3 ms; expired clean items could nevertheless experience tempo-
rary availability gaps. Replaying decisions from a recipe-speciﬁc
detector illustrated the protocol boundary: false promotions left
poison visible, while false refusals excluded clean content from
retrieval. Standalone Milvus tests characterized enforcement
delay and concurrent HNSW retrieval on one node. These results
clarify which guarantees come from the protocol and which
outcomes depend on detector quality, while quantifying the
freshness and availability trade-offs.
Index Terms —stream ingestion, admission control, systems
security, retrieval-augmented generation, vector databases, data
poisoning
I. INTRODUCTION
Retrieval-augmented generation (RAG) grounds a language
model’s output on documents retrieved from an external
knowledge base [1], and dense passage retrieval turns that
lookup into embedding-similarity search, commonly served by
a vector store [2]. As deployments move from static corpora
to live knowledge, that store must absorb new content while
serving queries.
Streaming RAG refreshes the knowledge base as content
arrives [3], while versioned vector lakes retain temporal views
over continuous updates [4]. Both aim to reduce freshnessdelay. Low freshness delay and trust assessment are in ten-
sion: establishing that a newly arrived item is trustworthy
takes time because it requires checking provenance, scanning
for adversarial or promotional manipulation, and comparing
against existing knowledge. A freshness objective instead
pushes the same item toward the query path as soon as
possible. Systems with aggressive freshness objectives may
therefore permit provisional content to become query-visible
before veriﬁcation completes. This is a design option, not a
requirement: keeping unveriﬁed content invisible until fully
vetted avoids unvetted exposure at the cost of veriﬁcation-
dependent freshness delay, a policy we retain as a baseline. We
study the security consequences of the low-latency operating
point and how to constrain them.
Admitting content before it is vetted creates an exposure
interval. Corpus-poisoning attacks can steer dense retrievers
with a small injected set, including for unseen queries [5],
and PoisonedRAG further shows that a few injected texts
can induce attacker-chosen answers in an end-to-end RAG
pipeline [6]. In a continuously ingested store, that exposure
is fundamentally temporal: a provisionally admitted poisoned
vector can be retrieved throughout the window between ﬁrst
visibility and containment, whether or not a detector eventually
ﬂags it.
This temporal systems view falls between existing lines:
poisoning defenses report detection effectiveness on static or
batch-constructed knowledge bases, and continuously updated
vector stores are mature but security-agnostic (Sec. VIII).
We present a fail-closed provisional-visibility protocol for
streaming vector stores. New content is query-visible but
bounded by a conﬁgurable visibility deadline Tp; only positive
multi-signal veriﬁcation promotes an item; expiry hides it
instead. A deadline scheduler decoupled from the veriﬁca-
tion queue makes that guarantee survive veriﬁer backlog,
and content later found malicious is excluded from retrieval
by lineage-scoped containment without rebuilding the store.
Author accepted manuscript. Accepted for publication at IEEE CBDCom 2026.
© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media,
including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers
or lists, or reuse of any copyrighted component of this work in other works.
arXiv:2610.05826v1  [cs.CR]  5 Oct 2026

Sweeping Tpexposes a measurable trade-off between freshness
and exposure.
This paper makes four contributions. C1 (Formalization).
We deﬁne unvetted exposure Eu, poisoning exposure Ep,
containment, and recovery as distinct intervals, separating
what a protocol can guarantee ( Eu) from what depends on
detector quality ( Ep). C2 (Protocol). We design a fail-closed
provisional-visibility protocol whose invariants I1–I7 guaran-
tee that unveriﬁed content cannot remain query-visible beyond
Tp+δhideand that each item is provisionally visible at most
once. A metadata time-to-live alone does not give this: an
expiry that queues behind the veriﬁer inherits its backlog.
C3 (Containment). We implement lineage-scoped containment
that excludes a compromised item’s descendants from the
query path before physical purging completes, and measure
its collateral revocation. C4 (Evaluation). We quantify dead-
line effectiveness and clean-content cost, evaluate a recipe-
speciﬁc feasibility detector and replay its decisions through
the promotion path, characterize eligibility completeness, and
evaluate δhideand hierarchical navigable small world (HNSW)
behavior on standalone Milvus.
II. B ACKGROUND AND THREAT MODEL
A RAG pipeline embeds a query, retrieves the top- knearest
chunks from a vector store and conditions generation on
them; under continuous ingestion an upstream stream is
chunked, embedded and inserted while queries are served.
We model the store as an insert path and a query path that
observe a per-item visibility state.
A. Threat model
We assume an adversary who can submit content into the
ingestion stream or control an already-authorized source. The
adversary may optimize against the retriever and any known
lightweight admission signal. Prior work shows high attack
success from a few retrieval-targeted texts [6]; SilentRetrieval
further demonstrates a ﬂuency-constrained evasive variant [7].
Injection is at a bounded rate (0.1–5% of the stream), steady
or bursty, with retrieval results observable. A poisoned item
succeeds if it is retrieved for a target query before it is
contained.
The adversary cannot modify admission metadata, the
deadline scheduler, per-item visibility state, or provenance
records; fully compromised veriﬁers, a malicious control-plane
operator, and embedding-model backdoors are out of scope.
We do admit trusted-source compromise: an item may carry
valid provenance yet be malicious. In RAGShield’s numerical-
claim setting, an insider with valid signing credentials reduces
provenance-only admission to the no-defense attack-success
rate [8]. We therefore do not rely on provenance alone for
promotion (Sec. IV-C).
B. Scope
We defend the ingestion-to-retrieval lifecycle of a single
vector-store deployment as observed through its supported
query path; the detector/veriﬁer is a pluggable componentthroughout. Strict multi-replica/cache consistency and fully
distributed rollback are out of scope (Sec. IX).
III. P ROBLEM FORMALIZATION
For an ingested item xwe distinguish the lifecycle times-
tamps tarrive,tvisible (ﬁrst retrievable through the supported
query path), tdetect,tcontain (no longer returned by the query
path), tpurge (physically deleted from the live store) and
taudit. The visibility deadline is anchored to ﬁrst visibility,
tdeadline =tvisible +Tp; anchoring to arrival instead shifts Eu
by the small admission-to-visibility gap. Freshness delay is
Df=tvisible−tarrive andrecovery isWrecovery =tpurge−tdetect;
physical live-store purge is treated separately from logical
containment.
We deﬁne unvetted exposure as the interval during which
unveriﬁed content is query-visible, with both endpoints on
the query path: Eu(x) =tqp-exit(x)−tvisible(x), where tqp-exit is
when the path stops returning xas unvetted, after promotion,
rejection or expiry. We deﬁne poisoning exposure as the
interval during which a malicious item can inﬂuence retrieval,
Ep(x) =tcontain (x)−tﬁrst-poison-visible (x); it depends on detector
false negatives and promotion policy, so the protocol bounds
it only when the veriﬁer never falsely promotes. We deﬁne
enforcement delay to record that a metadata state change is
not containment until the query path honors it: with texpire the
deadline instant and tqp-hide the moment the query path stops
returning x,δhide(x) =tqp-hide (x)−texpire(x).
The guarantee has three layers. (i) Operational bound.
Expiry ﬁres at tvisible +Tpwhatever the veriﬁer is doing,
because the scheduler does not share its queue, and the query
path stops returning the item δhidelater:
Eu(x)≤Tp(x)+δhide(x). (1)
Only query-path visibility counts, so exposure ends at tqp-hide
rather than at the earlier control-plane transition; anchoring
there would put δhideoutside the interval and make (1) true by
arithmetic. Equation (1) separates the conﬁgured deadline from
a deployment-speciﬁc enforcement slack. The latter comprises
scheduler ﬁring, state commit, and query-path propagation
(Sec. VI-A); each is ﬁnite only while the control plane is
live, so operators must provision and monitor δhide. The
protocol makes that term independent of veriﬁer backlog: a
separate scheduler and an O(1)ﬂag write provide it, although
shared-resource contention can still affect its magnitude. (ii)
Conditional poison bound. For a malicious item never falsely
promoted, the same bound applies to Ep. (iii) General case.
After false promotion, exposure is governed by detector error
and later containment. We therefore retain an oracle arm as a
mechanism upper bound and separately evaluate one recipe-
speciﬁc detector and its lifecycle consequences (Sec. VII).
IV. S YSTEM DESIGN
A. Architecture
An admission control plane assigns new content a risk score
r(x)and visibility budget Tp(x), then inserts it as PROVI-
SIONAL. Veriﬁer workers and the deadline scheduler consume

INGESTED
not visiblePROVISIONAL
visibleTRUSTED
visibleREVOKED
terminal
QUARANTINED
not visibleHIDDEN
not visibleadmitverify pass by Tp revoke
hard fail
faildeadline
late pass
late fail
Fig. 1. Protocol state machine. Shaded states are query-visible. Deadline
expiry only hides; a late pass enters TRUSTED directly, so each item has at
most one provisional episode.
that state independently. The query path reads an authoritative
ﬂag, admits PROVISIONAL and TRUSTED items to the same
top-kcompetition without trust-based reranking, and excludes
all other states. Physical deletion remains off the critical path.
B. Item state machine
Fig. 1 gives the atomic compare-and-set transitions; com-
peting veriﬁcation and deadline events are ordered by commit
time (Sec. IV-D).
C. Positive veriﬁcation and hard gates
Promotion requires afﬁrmative decisions from noncompen-
sating source/integrity ( S) and content/behavior ( C) families;
Cis mandatory under trusted-source compromise. D1 instan-
tiates this contract. The Sfamily checks credential validity,
provenance consistency and the pinned encoder revision. The
Cfamily sums calibration-standardized half-passage cosine
repetition and distance to the ten nearest corpus posts af-
ter excluding the item’s source group. Its threshold 1.3553
maximized poison-class F1 on 448 calibration groups, then
remained ﬁxed for a single evaluation on 512 disjoint test
groups; D1 received neither test labels nor evaluation queries.
Only the content family contributes detector-quality evidence;
the source-family controls validate interface plumbing.
D. Deadline enforcement and atomic transitions
Fail-closed enforcement rests on decoupling. A dedicated
scheduler, driven by per-item deadlines and independent of
the veriﬁer queue, performs expiration; hiding is an O(1)
visibility-state transition honored by the query path through
the authoritative ﬂag, waiting neither for veriﬁcation nor for
physical deletion. The enforcement delay δhidetherefore does
not queue behind the veriﬁer, the property on which (1) rests
under load.
Because a deadline and a veriﬁcation result may occur con-
currently, all transitions are atomic compare-and-set operations
conditioned on the current state, and the race is resolved
by commit time: a promotion succeeds only if the item is
still PROVISIONAL and the positive veriﬁcation committed
no later than the deadline; otherwise the scheduler commits
PROVISIONAL →HIDDEN and a late positive result com-
mits HIDDEN →TRUSTED idempotently. A veriﬁcation that
ﬁnishes computing before the deadline but commits after it
is thus post-deadline, so a veriﬁer cannot use its internal
completion time to extend visibility.E. Lineage-scoped containment and recovery
Each vector carries lineage metadata: document ID, source
and version, ingestion batch, embedding-model version, chunk
parent, and vector ID. A trusted item is revoked following
a later veriﬁer alert or a source/lineage revocation event,
such as a credential judged compromised or a batch ﬂagged
by an operator. Continuous per-item large language model
(LLM) monitoring is out of scope. Containment revokes
only the affected lineage rather than rebuilding the store.
Operationally, operators should choose the narrowest scope
the alert evidence justiﬁes: source scope for a compromised
publisher or credential, batch or version scope for a suspect
ingestion unit, and item scope only when the affected object
is localized and lineage is reliable. A scope narrower than
the evidence may leave correlated descendants visible, while
a broader one increases clean collateral removal. Logical
containment , an O(1)exclusion, occurs when the supported
query path no longer returns the item and ends measured poi-
soning exposure ( tcontain );physical live-store purge is deﬁned
separately through recovery latency ( Wrecovery ), so a security
metric is not conﬂated with database garbage collection. The
query-path guarantee establishes exclusion rather than erasure.
Operationally, before live-store purge completes, a hidden
vector may still occupy index storage; snapshots or exports can
preserve it afterward. Revocation tombstones must therefore be
retained and honored during restore and rebuild for as long as
such copies exist. Access control and audit remain necessary
until all retained copies are deleted.
F . Protocol invariants
Each invariant maps to a tested property. I1–I2 require
afﬁrmative SandCdecisions with no hard-gate failure; the
oracle tests the promotion outcome and D1 the concrete
decision path. I3 makes expiry hide, I4 enforces (1), I5
ends exposure at logical containment, I6 decouples expiry
from veriﬁcation and deletion, and I7 permits one provisional
episode. Lifecycle invariants I3–I7 run in both arms. A risk-
dependent Tp(x) =f(r(x))changes none of them.
V. I MPLEMENTATION
We implement the control plane on a streaming vector-
store stack. Content arrives through an asynchronous ingestion
producer and is routed by an admission service that applies
the cheap hard gate and assigns the visibility deadline, then
inserts admitted items into a single Milvus [9] collection.
PROVISIONAL and TRUSTED are eligibility states in the
control store rather than separate collections: promotion ﬂips
state and moves no vector, and the query service serves both
under one top- k(Sec. IV-A).
The authoritative item metadata (visibility state, deadline,
lineage, and audit timestamps) lives in a separate lightweight
control store, not in the vector database, which holds vectors
and serves retrieval while the query service decides eligibility.
Manu retains row-level ACID while explicitly relaxing com-
plex transaction requirements for vector workloads [10]. Our
protocol needs atomic transitions over those ﬁelds, so they

remain in the control store. Under post-ﬁltering, updating the
visibility ﬂag is an O(1)write that the query path honors im-
mediately and that is decoupled from deletion or compaction.
The timer-ordered scheduler does not share the veriﬁer pool;
together, these two forms of decoupling realize I6. Transitions
are compare-and-set updates (Sec. IV-D), and lineage columns
enable scoped revocation without a rebuild.
The veriﬁer interface has two instantiations. A deterministic
oracle isolates lifecycle timing; the detector arm replays held-
out D1 decisions and measured detector-only service times
through the same promotion path. It therefore captures ob-
served errors and latency but excludes online detector and
model contention. A single-process in-memory harness val-
idates the control plane without noise. The same interface can
be backed by a real vector store. Section VI-C compares the
resulting post-retrieval eligibility check with an in-index ﬁlter.
The harness, preregistrations and authoritative result ﬁles are
available from the authors, each recording the runtime-code
ﬁngerprint it was produced under.
VI. E VALUATION SETUP
We ask whether Euis bounded and how much the enforce-
ment delay δhidewidens it; whether that bound survives veriﬁer
overload; how freshness trades against exposure over Tp; what
top-kcompleteness cost eligibility enforcement imposes; and
how precise containment is. A separate experiment reports
HNSW index ﬁdelity [11] and descriptive single-node through-
put and tail latency across ﬁve independently restarted Milvus
Standalone processes (Sec. VII-G).
A. Instrumentation and derived metrics
Every lifecycle metric derives from one versioned event
schema of per-item timestamps, so the standalone deployment
emits the same ﬁelds without changing a deﬁnition. Three
distinctions matter. (i) tverify-end ̸=tverify-commit : a result com-
mitted after the deadline is post-deadline (Sec. IV-D). (ii)
Scheduler ﬁring, state commit and query-path propagation are
components of δhide, not extra terms in (1); observed through
probes, it is interval-censored, so we report the conservative
upper bound and never charge the probe interval to system
delay. (iii) Per-query item state is logged as observed at return
time, never reconstructed from the item’s later state, which is
what the residual exposure rate depends on.
B. Workloads, baselines, and parameters
Four workload classes stress different facets: ingestion un-
der concurrent retrieval, a poison burst, a veriﬁer slowdown
exercising I6, and post-promotion revocation. Each poison-
burst workload is generated from scratch: an eight-topic corpus
of 600 unit vectors, eight queries from the attacked topic
for crafting and eight more, disjoint, for measurement, eight
from other topics as controls, and six poison vectors perturbed
toward the crafting queries alone. Queries are issued at a ﬁxed
rate over a ﬁxed window with the poison injected partway
in, using ﬁve ﬁxed seeds, all retained. Four baselines share
one ingestion path and differ only in admission policy: B10.51
Tp
0 1 2 3 4 5 6
time since ingestion (s)poisoned top- krateB1 none B2 pre-vet
B3 no deadline B4 ours
Fig. 2. Poisoned top- krate over time on the crafted query set (Milvus Lite),
pooled over ﬁve workloads; one bin is one query rotation, below which the
rate is undersampled by construction. B3 holds at the undefended baseline’s
rate until veriﬁcation drains, B4 is cut at Tp, and B1 is never contained within
the window. Only the backlogged condition is drawn: with the veriﬁer keeping
up, B3 and B4 coincide.
immediate admission with no veriﬁcation, B2 synchronous
verify-before-visible, B3 async visible with no deadline, and
B4 the proposed protocol. B3 isolates the deadline’s own
effect, differing from B4 in nothing else. The deadline Tpis
swept over points bracketing the veriﬁer’s own latency.
C. Eligibility enforcement
A post-retrieval authoritative check (post-ﬁlter) and an in-
index metadata ﬁlter trade off along three axes: δhide, query
latency, and retrieval completeness. Post-ﬁltering gives a small
δhidebecause the control store is authoritative, but it ﬁlters after
approximate nearest-neighbor (ANN) search, so it can under-
ﬁll the top- k; the in-index ﬁlter avoids under-ﬁll but its hide
observability is left to the store’s read-consistency level [10],
which we ﬁx and report. With αover-fetch and pool size
M=αk, post-ﬁltering under-ﬁlls exactly when pool hidden
occupancy hM(q)exceeds 1 −1/α, so under-ﬁll is governed by
the upper tail of hMand not its mean. We therefore vary three
hidden distributions: uniform-random, semantic-clustered, and
query-coupled. The query-coupled set is built with knowledge
of the evaluation queries, so it is a worst case by construction;
the eligible population stays ≥k, so exact ground truth stays
well deﬁned. Completeness is reported as under-ﬁll (result
count <k) and Eligible Recall@ kagainst exact brute-force
ground truth, plus Conditional Recall@ kgiven a full- kresult,
which isolates ANN approximation error.
D. Invariant validation and reporting
Automated tests assert that timeout never promotes, that
each item enters PROVISIONAL at most once, that a post-
deadline-committed pass ﬁres HIDDEN →TRUSTED rather
than PROVISIONAL →TRUSTED, that HIDDEN and RE-
VOKED items are never returned, and that logical containment
holds while physical purge is pending. An overload test injects
heavy veriﬁer backlog and conﬁrms unvetted items still leave
the visible set within Tp+δhide.
VII. R ESULTS
The exact and Milvus Lite tiers validate lifecycle mechanics
and eligible ground truth; D1 adds measured decisions, and
Standalone adds HNSW with 1–16-client load.

A. Deadline enforcement under veriﬁcation backlog
The harness validates invariants I1–I7 and the commit-time
rule (20/20 checks). The real-time functional slice runs the full
ingest→provisional →deadline →hide→late-verify →
trusted →revoke lifecycle with 8/8 checks in both eligibility
modes. Decoupling the scheduler from veriﬁcation is what
bounds unvetted exposure under backlog: under 50 ×backlog,
decoupled scheduling gave Eu≈Tpversus 50 Tpunder coupled
scheduling. On the Milvus Lite real-time control plane, Euwas
1.00Tpdecoupled versus ≈5Tpcoupled under a ∼5 s backlog.
B. Retrieval-layer poisoning exposure and clean-content cost
Poison crafted against a designated query set is injected
into a collection held loaded throughout, with eligibility on
the post-ﬁlter path, and we measure what each baseline lets
it do (Fig. 2): ﬁve independently generated workloads whose
synthetic corpora keep exact eligible-neighbor ground truth
computable; Sec. VII-C repeats the grid on encoded natural-
language documents. Three query sets separate what one
poisoned-retrieval rate conﬂates. The set Qcraft, the queries the
poison was optimized against, is a worst case by construction;
Qtgtholds disjoint queries of the same topic, so a hit there is
transfer; Qnegis unrelated. Rates are reported over the queries
B1 compromised, a set applied unchanged to every baseline
andTp; counts are unaffected by that conditioning.
The poison reached 75–100% of crafted queries, while
transfer was partial: it reached 12–38% of disjoint same-
topic queries, with every workload showing some transfer,
and produced no poisoned retrieval on any of the 40 distinct
unrelated queries. Whether a query retrieves poison is decided
by the best poison it can see against its own top-5 cutoff, so
we report that pairing rather than an average over unrelated
pairs: for the median target query the best poison falls 0 .037–
0.061 short of the cutoff, and yet 1–3 of each workload’s eight
queries clear it. Predicted and retrieved agree for every one of
the 80 held-out queries across both workloads (Sec. VII-C),
so the transfer rate is a property of the geometry, not of the
defense or of when a query happened to run.
Under backlog, the deadline rather than veriﬁcation ended
exposure (Table I). B1 never contains the poison; its exposure
window ends only when measurement stops. B3 holds at
B1’s rate for 4 .7 s until veriﬁcation drains, and B4 is cut at
the deadline; the conditional rates are 0 .57–0 .63 for B3 and
0.125 for B4, versus 0 .875 for B1. Counts and rates are both
per-workload medians over workloads whose qualiﬁed-query
counts differ, so the rate is not the ratio of the counts. When
the veriﬁer keeps up, the two are indistinguishable, because
veriﬁcation commits inside Tpand the deadline never binds:
the deadline is what makes the bound survive backlog, not
what creates it.
This reduction came with a clean-content trade-off. Verify-
before-visible delays clean ﬁrst visibility until veriﬁcation
completes: 0 .62 s with the veriﬁer keeping up, 6 .3 s under
backlog. The deadline makes clean content provisionally visi-
ble within three milliseconds in both conditions, but does not
make availability continuous or durable: under backlog all sixTABLE I
MEDIAN POISONED RETRIEVALS ON QCRAFT ACROSS FIVE WORKLOADS ,
CUMULATIVE CLEAN -RESULT DISPLACEMENT ,AND THE TpSWEEP .
RANGES ARE SHOWN WHERE OUTCOMES VARIED ;COMPARISONS ARE
WITHIN WORKLOAD .
synthetic real text
poisoned retrievals, heavy veriﬁer backlog
B1 no admission control 49 (42–56) 49 (42–49)
B2 verify-before-visible 0 0
B3 async, no deadline 35 (29–37) 34 (29–34)
B4 this work 7 (6–8) 7 (6–7)
same, veriﬁer keeping up
B3 async, no deadline 3 3
B4 this work 3 3
clean positions lost, B1 / B3 / B4 56 / 37 / 8 49 / 34 / 7
B4 at Tp=0.15, 0 .3, 1, 2 s 2, 3, 7, 14 1, 3, 7, 14
crafted queries reached 75–100% 75–88%
held-out same-topic transfer 12–38% 12–38%
unrelated queries reached, of 40 0 0
clean items expired before veriﬁcation in every workload; ﬁve
were readmitted by the observation horizon and one was right-
censored. The median across the ﬁve per-workload median
completed-gap durations was 5 .3 s, with each workload con-
tributing ﬁve completed gaps. The appropriate policy depends
on application requirements: a security-sensitive deployment
can shorten Tp; provisioning veriﬁcation to commit before
expiry reduces gaps under expected load; and an application
that cannot tolerate gaps after ﬁrst visibility should use verify-
before-visible and accept its initial freshness delay. Lengthen-
ingTpreduces expiry risk only by widening the unvetted-
visibility budget. The deadline does not reduce a poison’s
effect while it remains visible: relative to the clean top- k
that would have been returned without the poison, the poison
displaced 20% of clean results under B1, B3, and B4 alike, and
eligible retrieval returned the exact top- kafter containment.
Per-hit displacement was therefore identical across the three;
only duration, and hence the cumulative ﬁgure, differs. Expo-
sure scales with the budget, so Tpis a calibrated dial rather
than a qualitative switch. These runs exercise a real Milvus
Lite collection, insertion path, top- kquery, and lifecycle;
the exact backend agrees on the main-grid poisoned-retrieval
counts except for one retrieval at the shortest deadline.
C. Real documents, a real encoder, and poison written as text
A planted geometry cannot establish transfer to a retriever
whose geometry did not deﬁne the attack, so we repeat
the grid of Sec. VII-B with the data replaced; a valida-
tion check conﬁrmed that all timing and protocol parame-
ters were unchanged. The corpus becomes 600 posts from
eight 20 Newsgroups categories spanning recreation, science,
computing, politics, religion and commerce, 75 each, with
headers, footers and quoted replies stripped, encoded by all-
MiniLM-L6-v2 at 384 dimensions; queries are the opening
words of held-out posts, leaving QcraftandQtgtdisjoint real
utterances. The poison is an English passage inspired by
PoisonedRAG’s black-box query-repetition construction [6]:
our implementation repeats the target query, appends ﬁxed

assertive ﬁller and tests whether the result enters the encoder’s
top-kwithout manual vector-space placement. It does not
instantiate PoisonedRAG’s target-answer generation condition.
The attack survives the substitution. Judged by the same
per-query top-5 cutoff rule, the poison falls 0 .034–0 .093 short
for the median target query, and 1–3 of each workload’s
eight clear it anyway. Poison was retrieved for 12–38% of
these queries, with at least one transfer in every workload,
but for none of the 40 distinct unrelated queries. Prediction
and retrieval agreed for all 40 real and all 40 controlled
queries. Query opportunity was ﬁxed: each query ran eight
times, seven against an unchanged candidate set. Together,
these observations support attributing transfer to the encoded
geometry rather than query opportunity.
The right-hand column of Table I shows the same baseline
ordering: B3 degrades under backlog while B4 holds at the
deadline, the two are indistinguishable when the veriﬁer keeps
up, and exposure remains monotone in the budget. Counts
are not directly compared across corpora and dimensions. The
oracle supplies the promotion outcome and nothing else: it
does not construct retrieval outcomes, alter embeddings, or
decide whether a poisoned item enters the top- k.
We next replace the oracle with controlled and measured
detector outcomes. In the controlled sweep, B4 was conﬁgured
to falsely promote k=0,1,2,3 items: B1 stayed at 49 median
poisoned retrievals, while B4 rose from 7 to 9, 13 and 15,
with every observed provisional exposure below 1 .02 s. On
512 held-out source groups, D1’s TP/FP/TN/FN counts were
129/65/255/63; recall was 125/128 for the matched recipe
but 4/64 for the predeﬁned natural-cover-sufﬁx cross-attack,
while 65/320 clean items were falsely refused. Detector-only
P50, P95, and P99 latencies were 26.1, 37.7, and 49.8 ms,
respectively. Each 300-admission HNSW replay produced 38
false promotions and 29 false refusals. All 38 false promotions
remained visible at the window end under B2–B4; B2 never
exposed false-refused clean items, whereas B3/B4 exposed
them brieﬂy and excluded them from retrieval by the window
end. The deadline bounds provisional exposure but cannot
revoke poison after promotion.
D. Eligibility enforcement and top-k completeness
Values below are means across three seeds; each seed’s
P95, under-ﬁll rate and recall are computed separately before
aggregation. At 30% global hidden occupancy with 2 ×over-
fetch, clustered hiding raised the P95 candidate-pool hidden
occupancy from 0 .47 to 0 .58 and under-ﬁll from 0 .7% to 8%
at an unchanged median hMof 0.30; the query-coupled stress
construction gave 26%. At 1 ×over-fetch, Eligible Recall@ k
was 0 .70, 0 .68 and 0 .38 under uniform, clustered and query-
coupled hiding. At 4 ×, observed under-ﬁll was eliminated
and recall reached 1 .0 in all three. Conditional Recall@ kwas
1.0 wherever a full- kresult existed and is undeﬁned for the
query-coupled 1 ×cell, which produced none, so recall loss
follows from under-ﬁll rather than selection error. The in-index
ﬁlter returned the exact eligible top- kwherever the eligible
population stayed ≥k. Over-fetch is not free. At 4 ×, eachquery requests four times as many candidates as at 1 ×, so
candidate transfer and post-ﬁlter checks increase; the backend-
speciﬁc effects on ANN search cost and latency were not
measured. Because αnever relaxes the authoritative eligibility
check, operators should use the smallest αthat meets their
completeness target for the observed upper tail of hM, or use
in-index ﬁltering when its propagation delay and consistency
semantics are acceptable. The 4 ×result demonstrates com-
pleteness for these workloads, not zero deployment cost.
E. Milvus Lite functional validation
On Milvus Lite, the adapter, all four state paths and both
eligibility modes execute against a real Milvus API and
reproduce the trends above. Conditional Recall@ kwas 1.0
because the FLAT index adds no ANN approximation.
F . Lineage-scoped containment
In the evaluated revocation workload, lineage-scoped con-
tainment caught 100% of the poison at all three granularities,
while collateral removal fell from ∼83% (by source) to ∼43%
(by batch) to 0% at exact item-lineage granularity. At equal
poison recall, these ﬁgures quantify the collateral cost of each
scope choice and the reduction in collateral removal when alert
evidence supports a narrower scope; they do not identify a
universally optimal granularity. Through the post-ﬁlter path,
no residual retrieval followed logical revocation on either
backend. Through the in-index path, it persisted immediately
after the scalar update on Milvus Lite ( ∼29–48% of probes),
so that path propagates a hide with nonzero delay (Sec. VI-C).
G. Standalone system results
We ﬁrst conduct a single-run propagation microbenchmark
on Milvus Standalone v2.4.15 (single node, Docker) [9] over a
200-item collection of synthetic unit vectors and measure the
in-index mode’s δhideend to end. The delay δhideis computed
from deadline expiry per the logged timestamps; probing be-
gins at acknowledgment. Two probes run concurrently: a scalar
point lookup and an exact (FLAT) top- ksearch. FLAT is used
because under an approximate index a vanished item cannot
be distinguished from a recall miss; concurrency keeps either
probe’s service time from being charged to the other. The
grid is four read-consistency levels ×{no, heavy} backlog,
Tp=1 s, 200 hide events per cell, with an identical control cell
at the start and end of the run; we report δhideas a conservative
upper bound derived from interval censoring (Sec. VI-A). The
hide is an override upsert because v2.4.15 has no in-place
scalar update, so the magnitudes are properties of that update
path as well as of the protocol.
Read consistency determined whether completed probes
remained stale after the hide committed. Under Strong and
Session, no completed probe returned the item in any cell: 0
stale reads out of 200 events per cell. Under Bounded and
Eventually, provisional items stayed retrievable across dozens
of completed probes.
In the Bounded and Eventually cells δhidewas 288–332 ms
at P50 and 380–431 ms at P95; the 9–12 ms censoring bracket

means that the reported bounds overstate the true delay by at
most∼12 ms. Under Strong and Session, the probe itself costs
∼420 ms at P95, so the ﬁrst completed probe already reports
the item hidden and δhideis bounded but never point-estimated.
That 420 ms characterizes the instrument, not the underlying
delay, which may be identical across modes. Adding the scalar
probe added no material cost to the top- ksearch path under
this workload (P95 421.0 ms with it versus 419.7 ms without),
so these delays are not an instrumentation artifact.
Thus δhidemust be budgeted: at Tp=1 s, a P95 enforcement
delay of ∼0.43 s widens the worst-case unvetted-exposure
window by more than 40%. That magnitude belongs to the
in-index scalar-update path, not to the control-store post-ﬁlter
conﬁguration of Sec. VII-B.
The panel cannot identify a veriﬁer-backlog effect. Its two
control cells perform identical query-only work at the run’s
start and end, yet their P50 values differ by 22.3 ms (6.5%), a
shift that exceeds the between-cell effects. We therefore report
rounded magnitudes and categorical behavior only: where δhide
resolved, the backlog contrast changed sign (Bounded +50 ms,
Eventually −23 ms), and both magnitudes were of the same
order as the control drift. With 200 events per cell, the panel
supports median estimation; P95 is tail-limited, and P99 is not
reported.
We report ﬁve Milvus process runs that passed ﬁxed validity
gates; the artifact retains all attempts. Each run built a 100,000-
vector HNSW index using cosine similarity [11], with M=16,
efConstruction =200, and Strong consistency. Each run
used the same ﬁxed set of 192 queries. At the primary ef=64
setting, run-level median Recall@5 was 0.9979 (range 0.9979–
0.9990), median P95 latency was 0.412 s, and no query errors
occurred. With 20 admissions/s, median throughput across
the four admission policies increased from 3.27–3.33 queries
per second (QPS) at concurrency 1 to 48.0–51.27 QPS at
concurrency 16, with no admission failures. Median P95 query
latency at concurrency 16 was 0.408–0.416 s.
VIII. R ELATED WORK
Corpus-poisoning attacks can steer dense retrieval with a
small injected set [5], and PoisonedRAG extends the threat to
attacker-chosen generation [6]. We propose no new attack.
SilentRetrieval studies ﬂuent, semantics-preserving variants
of such poisoning [7]. Our threat model admits this adversary
(Sec. II-A).
RAGuard ﬁlters suspicious retrieved chunks [12], and PRA-
RAG robustly aggregates retrieved evidence [13].
Orchestration and forensics complement such ﬁltering.
ADO selects defenses per query context [14], and RAGForen-
sics traces harmful outputs to responsible passages [15]. These
studies predominantly use static or batch-constructed corpora
and report method-speciﬁc outcomes rather than temporal ex-
posure, containment and recovery under continuous ingestion.
In-toto formalizes software-supply-chain attestation for how
an artifact was produced [16], and RAGShield applies prove-
nance checks to RAG ingestion and shows their insufﬁciencyagainst a valid-key insider [8]. The latter result motivates our
multi-signal rule without prescribing a detector.
Milvus provides similarity search over mutable collec-
tions [9], and Manu develops a cloud-native vector-DBMS
architecture [10]. Our control store holds only eligibility state
beside such a system (Sec. V).
FreshDiskANN supports incremental updates of a graph
index [17], while SPFresh performs in-place updates without
full rebuilds [18].
Freshness has also been pursued at the RAG layer. Stream-
ing RAG incrementally refreshes a compact prototype in-
dex [3], and SPI adapts retrieval resolution per query [19].
Temporal retrieval is a further direction. LiveVectorLake
adds versioned point-in-time retrieval [4]. These systems opti-
mize freshness, update efﬁciency or temporal retrieval but do
not model adversarial exposure windows.
A recent RAG-security survey separates admission-time
integrity, retrieval-time hardening and post-retrieval mitiga-
tion [20]. Our focus is the temporal ingestion-to-retrieval
lifecycle under concurrent updates.
IX. L IMITATIONS
The Eu≤Tp+δhidebound inherits the deployment’s en-
forcement delay. The corresponding Epbound holds only
absent false promotion; falsely promoted items require later
containment. Verify-before-visible avoids provisional exposure
at higher freshness delay, and a deadline limits exposure
duration rather than damage per retrieval. Accordingly, the
protocol controls unvetted-visibility duration; it does not solve
poisoning detection. Hard gates assume an honest control
plane; post-ﬁltering requires over-fetch; distributed clocks and
rollback remain out of scope.
D1 is one recipe-speciﬁc detector evaluated on 512 held-out
source groups. Its matched-recipe and cross-attack results do
not establish detector generality, unknown-attack robustness
or second-detector replication, and ofﬂine replay excludes
inference contention. The real-text study ends at retrieval
rather than answer generation. Physical-purge latency, storage
overhead while hidden vectors remain in the index, and restore
or reindex behavior were not evaluated.
The Standalone propagation panel is a single-run measure-
ment of the in-index update path; a second-host replication
passed its ﬁxed stability gate in only two of ﬁve runs and there-
fore contributes no latency estimate. The ﬁve HNSW runs are
single-node measurements, and 98,248 of 100,000 vectors are
deterministic pressure data. Requests are not independent sta-
tistical units; QPS and latency tails do not establish production
capacity, availability, fault tolerance, or distributed scaling.
Milvus Lite and cross-backend results validate implementation
consistency, not independent performance replication.

X. C ONCLUSION
Continuous RAG ingestion creates distinct unvetted- and
poisoning-exposure intervals. The protocol bounds the ﬁrst by
a deadline plus query-path enforcement delay and supports
lineage-scoped containment; on encoded natural-language
workloads under backlog it cut poisoned retrievals and
displaced clean results by roughly a factor of ﬁve, while
clean availability could lapse. D1’s false promotions and
refusals show why the poison bound is conditional and why
detector quality has a clean-content cost. Five single-node
HNSW runs show high-recall ANN retrieval under concurrent
ingestion for this workload.
ACKNOWLEDGMENT
Codex(OpenAI) was used for language editing, experiment
running, and LaTeX typesetting. The authors reviewed the
manuscripts, veriﬁed the reported results and citations, and
take responsibility for the ﬁnal text.
REFERENCES
[1]P. Lewis, E. Perez, A. Piktus et al. , “Retrieval-augmented generation
for knowledge-intensive NLP tasks,” in Advances in Neural Information
Processing Systems (NeurIPS) , vol. 33, 2020, pp. 9459–9474.
[2]V. Karpukhin, B. O ˘guz, S. Min et al. , “Dense passage retrieval for open-
domain question answering,” in Proc. Conference on Empirical Methods
in Natural Language Processing (EMNLP) , 2020, pp. 6769–6781.
[3]Y. Zhu, “From static to dynamic: A streaming RAG approach to real-
time knowledge base,” arXiv preprint arXiv:2508.05662 , 2025.
[4]T. Prajapati, “LiveVectorLake: A real-time versioned knowledge base
architecture for streaming vector updates and temporal retrieval,” arXiv
preprint arXiv:2601.05270 , 2026.
[5]Z. Zhong, Z. Huang, A. Wettig, and D. Chen, “Poisoning retrieval
corpora by injecting adversarial passages,” in Proc. Conference on
Empirical Methods in Natural Language Processing (EMNLP) , 2023,
pp. 13 764–13 775, arXiv:2310.19156.
[6]W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge
corruption attacks to retrieval-augmented generation of large language
models,” in Proc. 34th USENIX Security Symposium , 2025, pp. 3827–
3844, arXiv:2402.07867.[7]J. Qian, “SilentRetrieval: Hijacking retrieval-augmented generation via
semantically-preserving adversarial data poisoning,” in Proc. 32nd ACM
SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) ,
2026, pp. 4012–4023, arXiv:2605.28074.
[8]K. Patil, “RAGShield: Detecting numerical claim manipulation in gov-
ernment RAG systems,” arXiv preprint arXiv:2604.00387v2 , 2026.
[9]J. Wang, X. Yi, R. Guo et al. , “Milvus: A purpose-built vector data
management system,” in Proc. ACM SIGMOD International Conference
on Management of Data , 2021, pp. 2614–2627.
[10] R. Guo, X. Luan, L. Xiang et al. , “Manu: A cloud native vector database
management system,” Proceedings of the VLDB Endowment , vol. 15,
no. 12, pp. 3548–3561, 2022.
[11] Y. A. Malkov and D. A. Yashunin, “Efﬁcient and robust approxi-
mate nearest neighbor search using hierarchical navigable small world
graphs,” IEEE Transactions on Pattern Analysis and Machine Intelli-
gence , vol. 42, no. 4, pp. 824–836, 2020.
[12] Z. Cheng, J. Sun, A. Gao et al. , “Secure retrieval-augmented generation
against poisoning attacks,” in Proc. IEEE International Conference on
Big Data (BigData) , 2025, pp. 1799–1806, arXiv:2510.25025.
[13] X. Tan, Y. Zheng, C. Huo et al. , “PRA-RAG: Provably robust aggrega-
tion in retrieval-augmented generation against retrieval corruption,” in
Findings of the Association for Computational Linguistics: ACL 2026 ,
2026, pp. 36 002–36 017, arXiv:2607.00012.
[14] P. Pallerla, W. N. Bhukya, B. Vemula, and C. R. Kodi, “Adaptive defense
orchestration for RAG: A sentinel-strategist architecture against multi-
vector attacks,” arXiv preprint arXiv:2604.20932 , 2026.
[15] B. Zhang, H. Xin, M. Fang et al. , “Traceback of poisoning attacks to
retrieval-augmented generation,” in Proc. ACM Web Conference (WWW) ,
2025, pp. 2085–2097, arXiv:2504.21668.
[16] S. Torres-Arias, H. Afzali, T. K. Kuppusamy, R. Curtmola, and J. Cap-
pos, “in-toto: Providing farm-to-table guarantees for bits and bytes,” in
Proc. 28th USENIX Security Symposium , 2019, pp. 1393–1410.
[17] A. Singh, S. J. Subramanya, R. Krishnaswamy, and H. V. Simhadri,
“FreshDiskANN: A fast and accurate graph-based ANN index for
streaming similarity search,” arXiv preprint arXiv:2105.09613 , 2021.
[18] Y. Xu, H. Liang, J. Li et al. , “SPFresh: Incremental in-place update for
billion-scale vector search,” in Proc. 29th ACM Symposium on Operating
Systems Principles (SOSP) , 2023, pp. 545–561.
[19] D. Liu and Y. Yu, “SPI: Query-depth-adaptive indexing for streaming
RAG in vector databases,” arXiv preprint arXiv:2511.16681 , 2025.
[20] Y. Xu, M. Zhang, Z. Ge et al. , “Securing retrieval-augmented generation:
A taxonomy of attacks, defenses, and future directions,” arXiv preprint
arXiv:2604.08304 , 2026.