# Re:CAP - Auditing Retrieval Coverage in Production RAG Pipelines

**Authors**: Aviral Joshi, Hanoz Bhathena, Max Nelson, Saket Sharma

**Published**: 2026-09-21 05:19:40

**PDF URL**: [https://arxiv.org/pdf/2609.24122v2](https://arxiv.org/pdf/2609.24122v2)

## Abstract
Retrieval-augmented generation (RAG) is hard to monitor in production: exhaustive relevance labels do not exist for non-stationary multi-million-passage corpora that re-index in real time. As a result, retrieval quality is generally understudied and often deprioritised in favour of generation-oriented metrics. In this work, we propose auditing retrieval coverage by probing for evidence of missing documents rather than enumerating every relevant one. Our method Re:CAP (REtrieval Coverage Audit by iterative Probing) is a reference-free audit loop applied to a deployed RAG pipeline's initial answer and retrieved context: it identifies the topics already covered, generates probing questions for plausibly missing topics, retrieves candidate documents, and applies an LLM-as-judge to retain only those that introduce previously-unretrieved information. On four public benchmarks, Re:CAP recovers 9-29% of gold labels that flat BM25 top-500 cannot reach, rising to 48% on TREC-COVID. On MuSiQue Re:CAP beats flat hybrid top-500 by +12.9 pp on recall at less than half the document budget. An ensemble BM25, dense, and hybrid baseline (top-500 each) still leaves out 21.2% of gold docs on TREC-COVID that Re:CAP recovers; human annotators judge that 78.9% of those structurally distinct documents add new information to the baseline answer (Fleiss $κ$ = 0.79, n = 123), and 73.9% on live production traffic (n = 180). End-to-end recall is reproducible to within $\pm$1% across three independent runs, making Re:CAP a stable instrument for periodic retrieval audits.

## Full Text


<!-- PDF content starts -->

Re:CAP – Auditing Retrieval Coverage in Production RAG Pipelines
Aviral Joshi, Hanoz Bhathena, Max Nelson, Saket Sharma
Machine Learning Center of Excellence, JPMorgan Chase & Co.
{aviral.joshi, hanoz.bhathena, max.nelson, saket.sharma}@jpmchase.com
Abstract
Retrieval-augmented generation (RAG) is hard
to monitor in production: exhaustive relevance
labels do not exist for non-stationary multi-
million-passage corpora that re-index in real
time. As a result, retrieval quality is gener-
ally understudied and often deprioritised in
favour of generation-oriented metrics. In this
work, we propose auditing retrieval coverage
by probing for evidence of missing documents
rather than enumerating every relevant one.
Our method Re:CAP (REtrieval Coverage Au-
dit by iterative Probing) is a reference-free au-
dit loop applied to a deployed RAG pipeline’s
initial answer and retrieved context: it identi-
fies the topics already covered, generates prob-
ing questions for plausibly missing topics, re-
trieves candidate documents, and applies an
LLM-as-judge to retain only those that intro-
duce previously-unretrieved information. On
four public benchmarks, Re:CAP recovers 9–
29% of gold labels that flat BM25 top- 500can-
not reach, rising to 48% on TREC-COVID. On
MuSiQue Re:CAP beats flat hybrid top- 500by
+12.9 pp on recall at less than half the doc-
ument budget. An ensemble BM25, dense,
and hybrid baseline (top- 500each) still leaves
out21.2% of gold docs on TREC-COVID that
Re:CAP recovers; human annotators judge that
78.9% of those structurally distinct documents
add new information to the baseline answer
(Fleiss κ= 0.79 ,n= 123 ), and 73.9% on live
production traffic ( n= 180 ). End-to-end re-
call is reproducible to within ±1% across three
independent runs, making Re:CAP a stable in-
strument for periodic retrieval audits.
1 Introduction
Retrieval-augmented generation (RAG) is a stan-
dard component of enterprise knowledge assistants
over proprietary corpora such as news feeds, in-
ternal research, and regulatory filings (Lewis et
al., 2020; Gao et al., 2023). Popular RAG evalua-
tion frameworks — RAGAS (Es et al., 2023) andARES (Saad-Falcon et al., 2024) — score answer-
side signals (faithfulness, answer relevance) and
the relevance of the retrieved context, but cannot de-
tect what the retriever failed to surface. As a result,
an answer that is fluent and faithful to its retrieved
context can still be substantively incomplete, and
these metrics do not register the omission. For in-
stance, a research assistant asked about a recent
central-bank policy announcement may ground its
answer in the policy statement and the lead wire
story while overlooking the updated economic pro-
jections, the press-conference transcript, recorded
dissents, and post-meeting analyst commentary —
all present in the same corpus. We refer to this gap
between the documents the retriever surfaces and
the documents the corpus contains as theretrieval
coverageproblem, and we observe it most acutely
in non-stationary corpora with multiple sources of
truth. Figure 1 illustrates the problem: retrieving
more deeply extends the region a query already
reaches, so gold that lies outside it stays unreach-
able at any affordable depth.
Classical recall-based evaluation addresses re-
trieval coverage in principle but is impractical for
large production deployments. Modern production
indices contain 108or more passages (Karpukhin
et al., 2020; Bajaj et al., 2018), and the underlying
corpus is re-indexed continuously as documents
are added, edited, or withdrawn. Exhaustive per-
query relevance judgements are prohibitive at this
scale, and TREC-style pooling (V oorhees, 2000;
Buckley et al., 2007) requires multiple participat-
ing systems and fresh labels after every change
to the index, embedder, or ranker. Proprietary de-
ployments typically have no per-query relevance
labels at all, leaving operators reliant on end-to-end
answer signals that cannot distinguish a retrieval
failure from a reader failure.
We therefore reframe the problem. Rather than
enumerating every relevant document (intractable),
we probe each query for positive evidence that
arXiv:2609.24122v2  [cs.CL]  22 Sep 2026

Qtop-k top-500Raising kextends thesameregion; reranking only reorders within it.
Gold outside stays unretrieved at any affordable depth.One query reaches one region
QThe topic registry names facets the answer does not cover; each be-
comes a probe issued to thesameretriever.Re:CAP probes for what is missing
Re:CAP recovers9–29%un-
reachable gold (48%on TREC-
COVID)
corpus document gold, retrieved gold, never retrieved Re:CAP probe (gold inside was unreachable)
Figure 1: Schematic: retrieval coverage has a blind spot, and depth does not close it.Gold ranked below any affordable
top-kby the original query stays unretrieved, and answer-side evaluation cannot detect it. Re:CAP issues further queries from
the topic registry; each induces a different ranking over the same corpus and the same retriever (§2). The cone geometry is
illustrative, not a model of the retriever.
the retriever missed relevant content (tractable and
automatable). We instantiate this reframing in
Re:CAP(REtrieval Coverage Audit by iterative
Probing), a reference-free iterative loop that audits
a deployed RAG pipeline without consuming per-
query gold labels. Given a query Q, an initial set of
retrieved documents D0, and the corresponding an-
swerA0, Re:CAP (Algorithm 1) maintains a topic
registry Tof the information facets already cov-
ered, generates entity-anchored gap-probing ques-
tions targeting aspects relevant to Qbut absent
from T, expands retrieval with those questions,
and uses an LLM judge to label each new candi-
date document as novel, redundant, or off-topic.
The loop iterates until Tstops growing, yielding
a structured inventory of retrieval gaps G⊆T
together with label-free monitoring signals (gap
count, gap rate) whose stability and sensitivity to
retrieval quality we characterise empirically (§4.3,
§4.5). At 250–500LLM calls per audited query,
Re:CAP is aperiodic sampling auditorrather than
a per-query monitor: it runs over a representative
sample on an audit cycle — typically before pro-
moting an index, embedder, or ranker change —
not inline on live traffic. Cost is operator-tunable:
an all-mini pipeline is 4.6× cheaper for −0.51 pp
paired recall, at the edge of the run-to-run noise
floor (App. M, E3.5).
Our contributions are as follows:
•We propose atopic-based gap-discovery
frameworkand instantiate it as theRe:CAP
iterative protocol, whose gap-question gen-
erator combinesone dominant anchoringmechanism— the entity ledger, which ac-
counts for nearly all of the average-recall lift
— withfour supporting guards: probe-role
diversity, coverage-aware topic serialisation,
anti-collapse termination, and failure mem-
ory. The four are not recall levers. Each
suppresses a specific failure mode of a mini-
mal(Q+T) -only generator — topic-coverage
skew, low probe diversity, cold-start collapse,
and wasted iterations — that leave-one-out on
averagerecall over well-behaved benchmarks
does not surface, and their effect is concen-
trated in worst-case behaviour (§2, App. O,
Table 20).
•We provideempirical evidenceacross four
public benchmarks that Re:CAP recovers 9–
29% of gold unreachable by flat BM25 top-
500 (+12.9 pp over flat hybrid top- 500 on
MuSiQue at less than half the document bud-
get), moves predictably with retrieval quality
(§4.3), and reproduces to within 1%across
runs (§4.5).
•Wevalidate Re:CAP end-to-end on two
unrelated live production deployments
over proprietary corpora ( 400queries total;
App. E).
•Wevalidate Re:CAP-discovered gaps
against blinded human judgement: on
the ensemble-missed stratum that no single-
retriever upgrade closes, 78.9% of recovered
documents add information the baseline
answer lacks (three blinded annotators, Fleiss

κ= 0.79,n= 123; §5).
2 Method
2.1 Setting and topic-based gaps
A RAG pipeline has a corpus C, retriever R, and
LLM reader M. Given query Q, it produces an ini-
tial set of retrieved documents D0=R(Q, k) and
an answer A0=M(Q, D 0). Without exhaustive
labels for Cwe audit the pipeline by discovering
retrieval gaps: aspects of Qthat are relevant and
evidenced inCbut absent fromD 0.
We define gaps at thetopiclevel rather than the
document level. What counts as a single topic is
fixed by three rules: aparagraph test(each topic
warrants a distinct, non-overlapping paragraph), a
type-not-instancerule1, and a hard cap (default
50topics per query). A topic tis aretrieval gap
iff (i) tis relevant to Q, (ii) some d∈ C covers t,
and (iii) no d′∈D 0covers t. Topic-level counting
provides built-in deduplication across redundant ev-
idence documents, interpretable per-query reports,
and a normalised gap rate |G|/|T final|comparable
across queries of varying complexity.
2.2 The Re:CAP loop
Re:CAP runs as a six-step loop (Algorithm 1) lay-
ered on top of an unchanged host RAG pipeline.
1.Initial pass (S1).The host RAG pipeline
runs unchanged: D0=R(Q, k) ,A0=
M(Q, D 0).
2.Topic extraction and ledger init (S2).An
LLM extracts the distinct topics from A0as
short labels; each d∈D 0is then recon-
ciled against the registry Tin a batched LLM
call. Reconciliation recovers topics from D0
the reader omitted, so the rest of the loop
measures only retriever-side gaps. Anentity
ledger L(named entities, numeric and tempo-
ral anchors from QandA0) is also extracted
at this step and reused by S3 in every iteration.
3.Gap-Q generation (S3).The generator takes
Q, the registry T, and the entity ledger L,
and emits five gap questions targeting as-
pects absent from T. Each question is gen-
erated and tagged with one role from a fixed
five-role taxonomy (entity-anchored,concept-
anchored,constraint-relaxed,constraint-
tightened,inverse-negation; definitions in
1A topic describes thekindof information sought, not
a particular value of it — analogous to the class/instance
distinction in object-oriented design.Algorithm 1Re:CAP gap-discovery loop.
ColdStart(D 0, A0, i) :=i= 1∧(D 0=
∅ ∨A 0is ‘insufficient context’) vetoes early
termination (§2.2).
Require:queryQ, retrieverR, readerM, judgeJ
Require:k,m, MAX_ITER, MAX_TOPICS
1:D 0←R(Q, k);A 0←M(Q, D 0)▷S1
2:T←ExtractTopics(Q, A 0)▷S2
3:T←ReconcileDocs(Q, T, D 0)▷S2
4:L←ExtractEntityLedger(Q, A 0)▷S2
5:G← ∅;F← ∅; seen← ∅
6:fori= 1to MAX_ITERdo
7:Q g←GenGapQs(Q, T, L, F)▷S3
8:C←S
q∈QgR(q, m)\(D 0∪seen)▷S4
9: seen←seen∪C
10:N← ∅
11:for allc∈Cin parallel do▷S5
12:v, ℓ, t id← J(Q, T, A 0, c)
13:ifv∈ {NEWTOPIC,SUBTOPIC}then
14:N←N∪ {(ℓ, c)}
15:else ifv=REDUNDANTthen
16:T[t id].evid+={c}
17:end if
18:end for
19:N←Dedup(N, T)▷S5
20:T←T∪N;G←G∪N ▷S5
21:F← Q g\ {q:qyielded a new topic}▷S5
22: ifN=∅ ∧ ¬ColdStart(D 0, A0, i)then break ▷S6
23:end if
24:if|T| ≥MAX_TOPICSthen break
25:end if▷S6
26:end for
27:returnT, G,ComputeMetrics(T, G)
App. Q). Orthogonal to these five roles, one
anchoringmechanismand four supporting
guardsdrive coverage and diversity: (A) every
question must contain a literal anchor from
Lverbatim, preventing the generator from
paraphrasing away bridge entities; (B) top-
ics are serialised with their evidence count so
the generator targets under-supported topics;
(C) the role taxonomy enforces probe diver-
sity; (D) ananti-collapseguard requires ≥2
iterations when D0is empty or A0is an “in-
sufficient context” answer; (E)failure memory
passes the previous iteration’s unproductive
gap-Qs back as negative examples. As an
ablation reference we additionally evaluate a
minimal (Q+T) -only generator that drops
A–E.
4.Expanded retrieval (S4).For each gap ques-
tiongi,Rreturns the top- mdocuments, and
C=S
iR(g i, m) is deduplicated against
D0and previously seen candidate documents.
The same retriever as the host system is used,
so discovered gaps reflect that retriever’s cov-
erage limitations.

MuSiQue HotPot MHopRAG TC MS MARCO
Corpus 21K 5.2M 609 articles 171K 8.8M
Queries used 100 98 98 50 97
Evidence/q 2–4 2 2–4∼1,327∼213
Judgements binary binary binary graded (0–2) graded (0–3)
Table 1:Dataset characteristics. MHopRAG =MultiHop-
RAG; TC =TREC-COVID. Evidence/q is gold-supporting-
evidence count for binary-judgement sets, mean judged-pool
size for graded sets. MS MARCO is BM25-only (corpus size
precludes embedding); all others use hybrid BM25 + dense.
5.Novelty judging (S5).For each c∈
C, the judge assigns one of five verdicts
(NEWTOPIC, SUBTOPIC, REDUNDANT, IR-
RELEVANT, CONTRADICTORY); the first two
add the candidate to Tand to the gap inven-
toryG, REDUNDANTattaches the candidate
as additional evidence to its existing topic, and
the last two are discarded. Unproductive gap-
Qs (those yielding no new topic) are recorded
in the failure memory Fand passed back to
S3 at the next iteration. Overlapping new top-
ics across parallel judges are resolved by two-
stage post-hoc deduplication (App. C.1).
6.Termination (S6).The loop ends on natural
convergence (no novel candidates), an anti-
collapse veto on cold-start iter 1, the topic cap,
or the MAX_ITER budget (Algorithm 1).
The loop returns (i) the topic registry Twith
per-topic evidence, (ii) the gap inventory G⊆T ,
and (iii) monitoring metrics: gap count, gap rate,
and recall when labels are available.
3 Experimental Setup
3.1 Datasets
We evaluate on four primary datasets spanning
bounded-evidence multi-hop QA and pooled-
graded IR, plus MS MARCO TREC-DL 2019/2020
as an auxiliary BM25-only sensitivity ladder (Ta-
ble 1). MuSiQue and HotPotQA are the primary
bounded-evidence regime; MultiHop-RAG, on a
small 609-article corpus, serves as a saturation/-
ceiling sanity check. TREC-COVID is included to
delimit where Re:CAP applies and where it does
not (§6). Full per-dataset descriptions and rationale
are in Appendix A.
3.2 Retrieval and models
We evaluate Re:CAP with sparse BM25, dense
MiniLM-L12v2 (Reimers and Gurevych, 2019;
Wang et al., 2020), and a hybrid of the two
via reciprocal-rank fusion (Cormack et al., 2009).
On MuSiQue we additionally evaluate a strongerdense back-end, OpenAI text-embedding-3-
large . We adopt hybrid as the reference back-end;
BM25-only and dense-only act as ablations. The
QA reader is held fixed at GPT-5.2 across all con-
figurations to remove reader quality as a confound-
ing factor: the object of measurement isretrieval
gaps, not reader capability. The remaining Re:CAP
components (topic extractor, reconciler, gap-Q gen-
erator, judge) default to GPT-4.1 with temperature
0(gap-Q generator temperature = 0.3for diver-
sity). Unless noted otherwise, every main result
uses hybrid retrieval, the five-mechanism gap-Q
generator, k= 10 ,m= 50 , five gap-Qs per itera-
tion, MAX_ITER = 3, and MAX_TOPICS = 50 .
The full model ×component table and a per-knob
justification are in Appendix C.
3.3 Baselines
We compare Re:CAP against (i)flat top- Nre-
trievalat budgets matched to Re:CAP’s per-query
unique-candidate count Nq— the central budget-
matched control that isolates the audit’s contri-
bution from raw exposure to more documents —
under three retriever back-ends (BM25, MiniLM
hybrid, and OpenAI dense text-embedding-3-
large ; OpenAI on MuSiQue only); (ii)single-pass
probing(MAX_ITER = 1) to test whether iteration
adds value; (iii)RM3 pseudo-relevance feedback
(PRF)(Lavrenko and Croft, 2001; Abdul-Jaleel et
al., 2004) (App. J); and (iv) theminimal (Q+T) -
onlygenerator ablation on both BM25 and hybrid
retrievers.
Re:CAP is an audit layer, not a competing re-
triever.These comparisons are budget-matched
controls; none of them claims Re:CAP is a better
retrieval system. Re:CAP runson top ofwhatever
retriever a deployment already operates (§2), so
the evaluation question is not “does the loop beat
a stronger retriever?” but “does the loop surface
gold that the host retrieval missed?” Two standard
upgrades do not change that answer. Cross-encoder
rerankingreorders the flat top- Kalready retrieved
and cannot surface any document outside it — the
axis the headline results are anchored on (Figure 1).
One-shotquery rewritingis a strict subset of what
the loop does across iterations (§2). Applying ei-
ther to both sides raises the floor symmetrically,
which is why we report unreachable-gold share
alongside recall throughout.

3.4 Evaluation protocol
For each Re:CAP run we record per-query (i)
gold recall against dataset qrels (for validation
only, Re:CAP itself does not consume labels), (ii)
Re:CAP gap count and gap rate, (iii) unreachable-
gold share vs. flat top- 500(the fraction of Re:CAP-
recovered gold not in flat top- 500under each re-
triever), and (iv) telemetry rolled up to dollar cost
using published Azure OpenAI rates. Confidence
intervals on run-level metrics are 1,000 -sample
query-bootstrap 95%, and paired differences use
the same bootstrap on per-query deltas; the hu-
man evaluations state their interval method with
their tables. The variance protocol re-runs Re:CAP
with the default settings three times under the same
query slice and seed to estimate end-to-end stability
under inherent LLM non-determinism.
4 Results
4.1 Re:CAP vs. flat retrieval
Table 2 reports the cross-dataset main result (Pareto
visualization in App. Figure 3): Re:CAP at the
default configuration vs. three flat baselines at
matched docs-seen budget. On MuSiQue and Hot-
PotQA, Re:CAP beats flat BM25 at matched- Nq
budget by +6.1 to+29.1 pp; on MultiHop-RAG,
where flat top- 500is near saturation, the matched-
Nqgain narrows to +2.6 pp. It also beats stronger
flat baselines at less than half the document bud-
get: on MuSiQue, +12.9 pp vs. flat MiniLM hy-
brid top- 500and+4.7 pp vs. flat OpenAI dense
top-500; on HotPotQA,+11.5pp vs. flat MiniLM
dense top- 500. Across the six comparisons in Ta-
ble 2 the CIs are disjoint on the three largest mar-
gins ( +11.5 to+29.1 pp) and overlap on the three
smallest ( +2.4 to+6.1 pp): on the recall axis a
stronger baseline narrows the margin, which is why
the unreachable-gold share (§4.2) rather than recall
is the retriever-independent signal. On MultiHop-
RAG the 609-article corpus is near-saturated by
flat top- 500across retriever back-ends ( 0.88–1.00),
so it serves as a ceiling sanity check rather than
a discriminative benchmark; Re:CAP converges
within Nq≈45 –90depending on back-end. On
TREC-COVID Re:CAP shows the strongest struc-
tural complementarity: 48% of its recovered gold
is absent from flat BM25 top- 500and21.2% re-
mains absent from the ensemble of flat BM25,
dense, and hybrid top- 500combined– a 1,500 -
document, three-retriever budget – appearing in
98% of queries (49 of 50). By majority vote ofDataset Method Docs Recall [95% CI] Unr.|G|
MuSiQueFlat BM25N q 238 0.610 [.558,.664] — —
Flat hyb. top-500 500 0.772 [.723,.818] — —
Flat OpenAI top-500 500 0.854 [.815,.896] — —
RR hyb. 238 0.901 [.854,.938] 29.4% 3.6
HotPotFlat BM25N q 249 0.827 [.775,.878] — —
Flat hyb. top-500 500 0.864 [.813,.909] — —
Flat dense top-500‡500 0.773 [.717,.828] — —
RR hyb. 249 0.888 [.837,.939] 9.2% 2.3
TC† Flat BM25 top-500 500 0.241 [.211,.271] — —
RR hyb. 378 0.221 [.195,.248] 48.0%§36.9
Table 2:Re:CAP vs. flat retrieval at matched docs-seen bud-
get.RR hyb.= Re:CAP with hybrid retrieval;Unr.= share
of recovered gold not in flat BM25 top-500 (see§for the
ensemble variant); |G|= mean Re:CAP gap count per query.
MultiHop-RAG omitted (saturates at flat top- 500).†pooled-
graded scope boundary;‡MiniLM-L12 dense;§21.2% also
unreachable by the ensemble of flat BM25, dense, and hybrid
top-500. Full matrix: App. G.
three blinded annotators, 78.9% [71.5–86.2% ] of
n= 123 such documents add information beyond
theD0-only baseline answer (Fleiss κ= 0.79 ;
78.3% on MuSiQue; §5). On the recall metric
Re:CAP trails flat BM25 top- 500by a marginal
−2.0 pp (0.221 vs.0.241 ) at∼24% smaller doc-
ument budget: flat top- 500 itself reaches only
24% on this pooled-graded gold set of ∼493 rel-
evant docs per query, and binary-novelty judging
at a∼378 -doc Re:CAP budget is structurally mis-
matched with the recall ceiling (§6), thus comple-
mentarity, not recall@ k, is the audit-relevant signal
here.
4.2 Structural complementarity
Re:CAP samples a meaningfully different slice
of the relevance pool than flat retrieval — an
audit signal flat top- kcannot produce by con-
struction. Against flat BM25 top- 500,9.2–29.4%
of Re:CAP’s recovered gold is unreachable on
bounded-evidence (columnUnr.of Table 2), rising
to48% on TREC-COVID under the default hybrid
back-end ( 45–61% across alternative Re:CAP re-
triever back-ends; App. G). Against the ensemble
of BM25, dense, and hybrid top- 500, the share
tracks gold-pool diversity: 2.5% on HotPotQA
(4%of queries; 2-doc pools), 10.0% on MuSiQue
(21% of queries; 2–4 hops), and 21.2% on TREC-
COVID ( 98% of queries; ∼493 gold docs/q; Ta-
ble 17). Re:CAP’s structural contribution is largest
exactly where exhaustive recall labels are most in-
tractable to obtain — the production setting the
method is designed for.
4.3 Sensitivity to retrieval quality
A monitoring metric is only useful if itmoveswhen
the underlying retrieval changes — whether from a

corpus refresh, an embedder upgrade, or a reranker
change. Table 3 reports a within-dataset retrieval-
quality ladder: on MuSiQue, sweeping retriever
type (BM25 / hybrid) and top- k(10/20/50)
with generator fixed; on MS MARCO TREC-DL
2019/2020 ( n= 97 NIST-judged queries, rel≥2 ),
sweeping BM25 top- kover the same ladder. Two
observations. (i) Retrievertypedominates where
multiple are available: on MuSiQue, BM25 vs. hy-
brid moves Re:CAP recall by ∼6pp at matched
k. (ii) Top- kwithin a single retriever is approxi-
mately flat because Re:CAP’s expansion step sat-
urates the candidate pool regardless of D0depth
(≤1.9 pp paired across top- 10/20/50on every lad-
der;≤0.75 pp on MS MARCO BM25). Together
these are consistent with the design intent: Re:CAP
metrics arediscriminativewhere retrieval truly dif-
fers andstablewhere it does not.
The two label-free monitoring signals behave
differently along this ladder, and the distinction
matters for anyone deploying them. Mean gap
count|G|separates the two retriever types in the
expected direction — every BM25 rung sits above
every hybrid rung ( 4.00–4.20 vs.3.61–3.97) — so
a weaker host retriever leaves Re:CAP more gaps
to open. The magnitude, however, should be read
with care: the BM25–hybrid mean separation is
0.28, only 1.5× the run-to-run standard deviation
of|G|(0.19, Table 5), and at matched k=20 the
two retrievers differ by 0.03. The type separation
is consistent on this ladder, but within a retriever
top-kdoes not order |G|reliably, and the per-rung
differences are not resolvable within a single audit
cycle. |G|is therefore a directional indicator here
rather than a calibrated one, and separating adjacent
configurations needs repeated cycles or a larger
degradation than this ladder spans.
Gap rate is the more stable signal (CV 1.46%
against 5.04% ) but is near saturation here ( 0.81–
0.86 on MuSiQue, 0.96–0.98 on MS MARCO): on
MS MARCO almost every query yields a gap, so it
acts as a coveragefloorrather than a fine-grained
sensitivity signal. Neither signal is a drop-in alarm
on its own; the threshold recipe is in App. D.
4.4 Controlled gold-deletion check
We additionally validate sensitivity under con-
trolled deletion: removing K∈ {1,2,all}
gold passages from D0(n= 91 MuSiQue
queries with gold in D0), Re:CAP re-discovers
98.9%/98.4%/100% of the removed gold IDs re-
spectively. Paired recall still falls 11–13pp, at-Retriever recall∆ paired |G|gap rate $/q it
MuSiQue (n= 100, hybrid top-10 ref.)
BM25 top-10 0.842+5.924.20 0.831 0.65 1.93
BM25 top-20 0.861+4.004.00 0.859 0.65 1.88
BM25 top-50 0.857+4.424.15 0.825 0.61 1.92
Hybrid top-10 (ref) 0.901—3.61 0.845 0.59 1.79
Hybrid top-20 0.918−1.673.97 0.814 0.62 1.88
Hybrid top-50 0.915−1.013.93 0.831 0.57 1.77
MS MARCO TREC-DL (n= 97, BM25 top-10 ref.)
BM25 top-10 (ref) 0.683—12.58 0.976 0.94 2.43
BM25 top-20 0.679+0.4212.32 0.955 0.96 2.51
BM25 top-50 0.676+0.7512.03 0.956 0.87 2.34
Table 3:Retrieval-quality ladder. Columns: Re:CAP recall;
paired delta (positive ∆paired implies the cell isworsethan its
reference); mean gap count |G|per query; gap rate; cost per
query; and mean iterations per query till convergence. |G|
separates the two retriever types in the expected direction,
though by margins comparable to its own run-to-run noise;
gap rate saturates on these workloads and acts as a coverage
floor (§4.3).
Metric MuSiQue HotPot MHopRAG TC
LLM calls (total) 28.6k 25.0k 25.8k 30.8k
of which judge 28.1k 24.5k 25.3k 30.2k
Judge share 97.5% 97% 97% 97%
$ / query 0.59 0.65 0.69 2.05‡
Table 4:Cost breakdown (Re:CAP default on hybrid retrieval).
‡TREC-COVID cell is the BM25 + (Q+T) variant (only TC
configuration with full token telemetry).
tributable to secondary gold rather than to failed
recovery. Full E2.2 table and discussion: Ap-
pendix L.
4.5 Operational characteristics
Cost.The judge dominates per-query cost
(∼97% across all datasets; judge →GPT-4.1-
mini saves 75% of HotPotQA cost, the gen-
erator swap alone saves 8%). Per-query cost
varies ∼3.5× across corpora ($ 0.59–$2.05, Ta-
ble 4). A 4.6× Pareto improvement is available
for−0.51 pp paired recall on HotPotQA by swap-
ping all pipeline components to GPT-4.1-mini (at
the edge of the 0.42 pp noise floor below; full grid
in Appendix M).
Reproducibility.Three independent runs of
Re:CAP on MuSiQue (Table 5; same slice and seed;
variance is inherent LLM non-determinism) give
Re:CAP-recall a Coefficient-of-variation = 0.47%
(±0.42 pp), so Re:CAP returns a stable recall esti-
mate for a fixed (pipeline, corpus) pair. Combined
with the retriever-type sensitivity demonstrated in
§4.3, this licenses inter-run comparisons at that
granularity: recall deltas that exceed the noise floor
and accompany a change of retriever can be read
as real changes rather than measurement noise. Ad-
jacent top-ksettings are not separable this way.

Metric Mean Std CV Range
Re:CAP recall 0.89720.00430.47%0.0083
Re:CAP gap count|G|3.77 0.19 5.04% 0.37
Re:CAP gap rate 0.8320.0121.46%0.024
Nq(unique cands.) 246.5 7.4 2.99% 15
Cost / run ($) 62.47 3.18 5.09% 6.01
Table 5:Run-to-run variance, three independent runs
(MuSiQue, default, n= 100 ). Recall CV 0.47% and gap-rate
CV1.46%both underpin the deployable-monitoring claim.
Attribution of residual losses.On bounded-
evidence primaries the dominant failure is gap-
question generation, not judging ( 95.3% of miss-
ing docs never surface as candidates): early con-
vergence, bridge-entity erasure, or entity-ledger
anchoring on a hallucinated name from A0. On
pooled-graded TREC-COVID the failure is struc-
tural: binary novelty judging over-aggregates sub-
mechanisms and the Re:CAP budget is dwarfed
by the graded pool. Full taxonomy, counts, and
worked examples are in Appendix O.
Production deployment.The default configura-
tion also audits two live deployments on proprietary
corpora ( ≈123 M and ≈70M passages; 200queries
each), differing chiefly in D0width: “AI Search”
(D0fixed at 10) and “NEWS QA” ( D0mean 47.2,
max110). Both run 200/200 with no production-
specific code path: gap rate 0.991 /0.955 , gap-
iteration topic share 77.4% /78.2% , judge share
of calls ≥97.6% , median cost $ 1.68 / $1.79 per
query. Full two-cohort breakdown in App. E (Ta-
ble 10).
5 Human evaluation of recovered gap
documents
The21.2% TREC-COVID and 10.0% MuSiQue
ensemble-unreachable shares in §4.1 are qrels-
mediated. To test whether these structurally dis-
tinct documents add information the reader actu-
ally lacks — independent of the qrels — we ran a
blinded human read on the ensemble-unreachable
stratum.
Setup.We sampled n= 123 ensemble-
unreachable gap documents from the deployed de-
fault: 100TREC-COVID documents stratified by
query (target 2docs per query among the TREC-
COVID queries with ≥1 ensemble-unreachable
doc;48queries represented, 1–3docs each) plus
the23-document MuSiQue census; HotPotQA,
MultiHop-RAG, and MS MARCO are ineligible
(App. N.3). Three annotators saw the query, the
D0-only baseline answer, and the document text,blinded to judge verdict, qrels status, and dataset.
The binary rubric labels a documentnew_infoif it
adds a substantive query-relevant fact the baseline
answer lacks, andcoveredotherwise.
Result.By majority vote, 78.9% [71.5–86.2,
10,000 -resample bootstrap] of ensemble-missed
gap documents are judgednew_infoat Fleiss κ=
0.79 (Table 6); per-dataset rates are substantively
comparable. This bounds Re:CAP’s operational
value on the strictest stratum: the documents it
recovers that no 1,500 -document single-retriever
combination surfaces carry information the base-
line answer lacks.
Slicennew Rate % [95% CI]κ
TREC-COVID (stratified) 100 7979.0[71.0–87.0]0.82
MuSiQue (census) 23 1878.3[60.9–95.7]0.69
Overall 123 9778.9[71.5–86.2]0.79
Table 6:Human-evaluation verdicts on 123 ensemble-
unreachable gap documents from the deployed default.new
is the majority-votenew_infocount; 95% CIs are 10,000 -
resample bootstrap; κis Fleiss across three annotators on
binary verdicts.110/123(89.4%) unanimous.
6 Conclusion
Re:CAP reformulates the retrieval evaluation prob-
lem in Production RAG pipelines from that ofenu-
merationto one ofprobingby utilizing a topic-
aware iterative gap-discovery loop that audits a
deployed pipeline without per-query gold labels.
Across four publicly available benchmarks we
demonstrate that Re:CAP recovers gold that base-
lines cannot discover ( 9–29%, rising to 48% on
TREC-COVID) and does so at half the document
budget (on MuSiQue). Results are reproducible
to∼1% recall across runs. Human assessment
on Re:CAP recovered gold documents shows that
78.9% of the recovered documents have novel in-
formation missing from the baseline answer ( κ=
0.79). We share Re:CAP statistics and examples
from 2 live large-scale production RAG pipelines
with200-query audits each (App. E) along with a
deployment recipe (App. D).
Ethical Considerations
Computational and environmental cost.The
iterative probing loop has significant energy and
carbon implications at scale ($ 0.59–$2.05/query at
default; §4.5); we report full per-corpus cost ac-
counting to enable informed deployment decisions
and document a steep Pareto improvement (judge
→GPT-4.1-mini) for cost-sensitive deployments.

Teams running Re:CAP continuously should sam-
ple queries rather than audit every request.
Bias in gap discovery.The LLM-as-judge may
have systematic blind spots — topics it consistently
fails to recognise as novel — that could correlate
with sensitive attributes and provide false assur-
ance of completeness. Re:CAP cannot fully au-
dit itself; periodic gold-label spot-checks of judge
outputs against held-out qrels (Appendix N) are
recommended before relying on Re:CAP signals
for compliance-grade reporting. Re:CAP’s gap-
Q generator also inherits any entity biases of the
source LLM through the entity ledger; we have not
characterised this with respect to demographic or
geographic biases.
Human annotation.The human evaluations in
§5 and App. F were performed by three in-house an-
notators, not by the authors and not by crowdwork-
ers. The production cohorts are internal corpora:
annotation took place inside that organisation, and
no document text from them is reproduced here —
the worked examples in App. E report topic labels
and generated probe questions only. We collected
no personal data from annotators and report no
demographic characteristics of the annotator popu-
lation.
Privacy and access control.Expanded retrieval
issues additional probes against the corpus with
LLM-generated questions. In deployed systems
with row-level access controls, those controls must
be honoured during gap probing — otherwise the
audit can surface relevant documents the original
user would not be entitled to retrieve. Our reference
implementation passes the original retrieval ACL
context through the loop; we recommend the same
for any deployment.
Surface area for prompt injection.Step 2 (rec-
onciliation) and Step 5 (judging) ingest retrieved
document text as input to LLM calls and are there-
fore exposed to prompt-injection attempts embed-
ded in corpus documents. We use structured-output
decoding and a narrowly scoped classification task
to reduce this surface, but for adversarial corpora
additional input sanitisation is warranted.
Limitations
LLM cost.Re:CAP requires multiple LLM calls
per query (topic extraction, doc reconciliation, gap-
Q generation, novelty judging per candidate, post-hoc deduplication). At the default this totals 250–
500LLM calls per query ($ 0.59–$2.05 at GPT-
4.1 pricing; §4.5). This is materially more expen-
sive than single-pass metrics like RAGAS, and
may not suit tight monitoring budgets. Swap-
ping all pipeline components to GPT-4.1-mini cuts
this4.6× (−0.51 pp paired recall on HotPotQA;
App. M, E3.5).
Judge reliability ceiling.Re:CAP is in principle
bounded by the LLM judge’s accuracy. The gold-
label failure-mode analysis (Appendix O) bounds
this empirically: 1.4% of missing-gold docs are
judge-rejected overall ( 4.7% on the bounded-
evidence primaries), so >95% of missing-gold
losses on the regime Re:CAP is designed for are
upstream of judging. We rely on the LLM-as-judge
literature (Faggioli et al., 2023; Thomas et al., 2024;
Upadhyay et al., 2024a,b) as support for narrow,
structured-output tasks, and follow cautions against
substituting LLMs for full human qrels (Soboroff,
2025) by validating Re:CAP against held-out la-
bels.
Corpus coverage assumption.Re:CAP discov-
ers gaps only for topics that exist in the corpus. If
the corpus itself lacks coverage of an aspect, no
probing will find it. Re:CAP measuresretrieval
gaps, notcorpusgaps.
English only.All experiments are on English-
language datasets with English-language LLMs.
The approach is language-agnostic in principle but
unvalidated multilingually.
Gap-Q generator dependence.Gap-probing de-
pends on the LLM-generated questions; vague or
off-target questions under-surface relevant docu-
ments and under-count gaps. The five-mechanism
generator reduces but does not eliminate this risk.
Topic granularity.We have no formal defini-
tion of an “atomic topic”. We rely on three
prompt-engineering rules (paragraph test, type-not-
instance, different-dimension sub-topic) plus a con-
figurable hard cap. Broad or ambiguous queries can
still trigger instance enumeration, inflating topic
counts.
Instance-level coverage holes can be masked.
By design, the type-not-instance rule collapses in-
stance enumeration into a single topic. A retriever
that surfacessomeinstances of a topic but misses
others along the same dimension is scored as cov-
ering that topic, even though instance-level recall

is incomplete. The SUBTOPICverdict mitigates
this only when missed instances reveal a differ-
entdimension, not when they are missing along the
existing one. Deployments that care about instance-
level enumeration recall (e.g., fact verification, au-
dit trails) should track Re:CAP at a finer granularity,
or pair it with instance-level metrics.
Recall@ kundercounts complementarity on
pooled-graded corpora.On pooled-graded cor-
pora such as TREC-COVID — where a query
has hundreds–thousands of partially-relevant docu-
ments, flat BM25 top- 500itself reaches only 24%
recall, and any ∼378 -doc method cannot hold the
∼493 relevant docs per query — recall@ kcom-
presses complementarity into a small negative delta
(Re:CAP best cell ∆ =−2.0 pp; §4.1). The audit-
relevant signal survives: 21.2% of Re:CAP’s re-
covered gold remains absent from the ensemble of
flat BM25, dense, and hybrid top- 500combined,
present in 98% of queries (Table 17). On such cor-
pora, recall@ kshould be paired with a structural
complementarity metric to capture Re:CAP’s audit
contribution.
Initial-retriever choice can cause cell-level
regressions.The HotPotQA dense- D0cell
(−3.6 pp matched- Nq) shows the initial retriever
interacts with the dataset’s evidence structure. The
initial retriever should be selected based on dataset-
level recall-at-kdiagnostics before Re:CAP is lay-
ered on top.
Provider-side content filtering depresses recall.
All experiments run against Azure OpenAI, whose
content-management policy rejects a small share
of prompts as ResponsibleAIPolicyViolation .
Rejected candidates produce no verdict, so any that
would have been NEWTOPICare silently dropped
and reported recall is a conservative lower bound.
The effect is small per-query and well within the
variance noise floor of §4.5, but a non-filtered back-
end would likely yield slightly higher recall than
reported here.
Calibrated metrics rest on benchmarks; pro-
duction audit is a single snapshot.The recall,
paired- ∆, unreachable-gold, variance, and cost lad-
ders all rest on four English-language academic
benchmarks (§§4.1–4.5); those metrics require per-
query gold and so cannot be reported on production
traffic. We do also report end-to-end audits on two
unrelated live deployments over proprietary cor-
pora ( 400queries total, App. E), which confirmthe gap rate, topic-source distribution, convergence
behaviour, and judge-share-of-cost pattern transfer
to production traffic across deployments differing
inD 0width by∼5×.
Acknowledgments
We thank the Annotation Center of Excellence
(ACoE) at JPMorgan Chase & Co. for the human
annotation work reported in this paper. The annota-
tors are salaried employees of JPMorgan Chase &
Co. and performed this work as part of their regular
duties.
Disclaimer
This paper was prepared for informational purposes
in part by the Machine Learning Center of Excel-
lence group of JPMorgan Chase & Co. and its af-
filiates (“JP Morgan”) and is not a product of the
Research Department of JP Morgan. JP Morgan
makes no representation and warranty whatsoever
and disclaims all liability, for the completeness, ac-
curacy or reliability of the information contained
herein. This document is not intended as invest-
ment research or investment advice, or a recom-
mendation, offer or solicitation for the purchase
or sale of any security, financial instrument, finan-
cial product or service, or to be used in any way
for evaluating the merits of participating in any
transaction, and shall not constitute a solicitation
under any jurisdiction or to any person, if such so-
licitation under such jurisdiction or to such person
would be unlawful.
References
N. Abdul-Jaleel, J. Allan, W. B. Croft, F. Diaz,
L. Larkey, X. Li, M. D. Smucker, and C. Wade.
UMass at TREC 2004: Novelty and HARD. In
TREC, 2004.
G. Amati.Probability models for information retrieval
based on divergence from randomness. PhD thesis,
University of Glasgow, 2003.
A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi. Self-
RAG: Learning to retrieve, generate, and critique
through self-reflection. arXiv:2310.11511, 2023.
P. Bajaj et al. MS MARCO: A human gen-
erated machine reading comprehension dataset.
arXiv:1611.09268, 2018.
C. Buckley, D. Dimmick, I. Soboroff, and E. V oorhees.
Bias and the limits of pooling for large collections.
Information Retrieval, 10(6):491–508, 2007.

J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking
large language models in retrieval-augmented gener-
ation. InAAAI, 2024.
C. L. A. Clarke, M. Kolla, G. V . Cormack, O. Vechto-
mova, A. Ashkan, S. Büttcher, and I. MacKinnon.
Novelty and diversity in information retrieval evalua-
tion. InSIGIR, pages 659–666, 2008.
G. V . Cormack, C. L. A. Clarke, and S. Büttcher. Recip-
rocal rank fusion outperforms Condorcet and individ-
ual rank learning methods. InSIGIR, 2009.
N. Craswell, B. Mitra, E. Yilmaz, D. Campos, and
E. M. V oorhees. Overview of the TREC 2019 deep
learning track. InTREC, 2020.
S. Es, J. James, L. Espinosa Anke, and S. Schock-
aert. RAGAS: Automated evaluation of retrieval
augmented generation. arXiv:2309.15217, 2023.
G. Faggioli et al. Perspectives on large language models
for relevance judgment. InICTIR, 2023.
Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, M. Wang, and H. Wang. Retrieval-augmented
generation for large language models: A survey.
arXiv:2312.10997, 2023.
Z. Jiang et al. Active retrieval augmented generation.
InEMNLP, 2023.
J.-H. Ju, S. Verberne, M. de Rijke, and A. Yates. Con-
trolled retrieval-augmented context evaluation for
long-form RAG. InFindings of EMNLP, pages
21102–21121, 2025.
J.-H. Ju, François G. Landry, E. Yang, S. Verberne,
and A. Yates. LANCER: LLM reranking for nugget
coverage. arXiv:2601.22008, 2026.
V . Karpukhin et al. Dense passage retrieval for open-
domain question answering. InEMNLP, 2020.
V . Lavrenko and W. B. Croft. Relevance-based language
models. InSIGIR, 2001.
P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-T. Yih, T. Rock-
täschel, S. Riedel, and D. Kiela. Retrieval-augmented
generation for knowledge-intensive NLP tasks. In
NeurIPS, 2020.
R. Nogueira, W. Yang, J. Lin, and K. Cho. Document
expansion by query prediction. arXiv:1904.08375,
2019.
V . Pavlu, S. Rajput, P. B. Golbus, and J. A. Aslam. IR
system evaluation using nugget-based test collections.
InWSDM, pages 393–402, 2012.
N. Reimers and I. Gurevych. Sentence-BERT: Sen-
tence embeddings using Siamese BERT-networks. In
EMNLP, 2019.
D. Ru et al. RAGChecker: A fine-grained framework
for diagnosing retrieval-augmented generation. In
NeurIPS Datasets and Benchmarks, 2024.J. Saad-Falcon, O. Khattab, C. Potts, and M. Za-
haria. ARES: An automated evaluation frame-
work for retrieval-augmented generation systems.
arXiv:2311.09476, 2024.
S. Samuel, A. Yates, D. Lawrie, I. Soboroff, T. Adri-
aanse, B. Van Durme, and E. Yang. CoverageBench:
Evaluating information coverage across tasks and
domains. arXiv:2603.20034, 2026.
I. Soboroff. Don’t use LLMs to make relevance judg-
ments.Information Retrieval Research, 1(1), 2025.
Y . Tang and Y . Yang. MultiHop-RAG: Benchmarking
retrieval-augmented generation for multi-hop queries.
InEMNLP, 2024.
P. Thomas et al. Large language models can accurately
predict searcher preferences. InSIGIR, 2024.
H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabhar-
wal. MuSiQue: Multihop questions via single-hop
question composition.TACL, 10, 2022.
H. Trivedi, N. Balasubramanian, T. Khot, and A. Sab-
harwal. Interleaving retrieval with chain-of-thought
reasoning for knowledge-intensive multi-step ques-
tions. InACL, 2023.
S. Upadhyay, E. Kamalloo, and J. Lin. LLMs can
patch up missing relevance judgments in evaluation.
arXiv:2405.04727, 2024.
S. Upadhyay, R. Pradeep, N. Thakur, D. Campos,
N. Craswell, I. Soboroff, H. T. Dang, and J. Lin. A
large-scale study of relevance assessments with large
language models: An initial look. arXiv:2411.08275,
2024.
E. M. V oorhees. Variations in relevance judgments and
the measurement of retrieval effectiveness.Infor-
mation Processing & Management, 36(5):697–716,
2000.
E. M. V oorhees. Overview of the TREC 2003 question
answering track. InTREC, 2003.
E. M. V oorhees et al. TREC-COVID: Constructing a
pandemic information retrieval test collection.SIGIR
Forum, 54(1):1–12, 2021.
W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and
M. Zhou. MiniLM: Deep self-attention distillation
for task-agnostic compression of pre-trained trans-
formers. InNeurIPS, 2020.
P. Wang et al. Large language models are not fair evalu-
ators. arXiv:2305.17926, 2023.
K. Xie, P. Laban, P. K. Choubey, C. Xiong, and C.-S.
Wu. Do RAG systems cover what matters? Eval-
uating and optimizing responses with sub-question
coverage. InNAACL, pages 5836–5849, 2025.
Z. Yang et al. HotpotQA: A dataset for diverse, ex-
plainable multi-hop question answering. InEMNLP,
2018.

J. Zobel. How reliable are the results of large-scale
information retrieval experiments? InSIGIR, 1998.
Appendix
A Datasets and Rationale
We evaluate on four datasets, chosen to span (i)
bounded-evidence multi-hop QA — the opera-
tional regime Re:CAP is designed for — and (ii)
pooled-graded IR as a contrasting regime (Table 1
in the main body). The MS MARCO TREC-DL
2019/2020 ladder is an additional BM25-only vali-
dation surface for the sensitivity result (§4.3).
MuSiQue(Trivedi et al., 2022): 21K-paragraph
deduped corpus, 2–4-hop chained reasoning,
shortcut-resistant by design; 2,417 dev queries, of
which we sample 100with seed 42. The strongest
signal in our sensitivity ladder.
HotPotQA(Yang et al., 2018): 5.2M-passage
Wikipedia corpus, canonical multi-hop benchmark;
5,447 dev / 7,405 test queries, of which we sam-
ple98with seed 42. Provides the largest-corpus
comparison point.
MultiHop-RAG(Tang and Yang, 2024): 609-
article news corpus, 2,556 test queries, 2–4 doc-
uments per query; an EMNLP 2024 RAG-native
benchmark. We use the published test set ( n= 98
for our slice). The small corpus means flat BM25
top-500reaches recall 1.000 , so this dataset is best
read as a saturation/ceiling sanity check rather than
a discrimination result.
TREC-COVID(BEIR) (V oorhees et al., 2021):
171K biomedical passages, 50 round-3 topics with
pooled graded judgements ( ∼493 rel docs per
query at rel≥1 ;∼1,327 judged). We include it to
characterise where Re:CAP’s binary-novelty ma-
chinery breaks down (see Limitations,Recall@ k
undercounts complementarity on pooled-graded
corpora).
MS MARCOTREC-DL 2019/2020 (Bajaj et
al., 2018; Craswell et al., 2020): 8.8M passages, 97
NIST-judged queries with graded qrels thresholded
at rel≥2. We use this as a BM25-only sensitivity
ladder (Table 3, second panel) on an independently
judged, IR-standard corpus. We deliberately do not
embed MS MARCO for dense or hybrid retrieval
(project-budget choice); it is used only in BM25-
only sensitivity configurations.
B Related Work
Retrieval evaluation and coverage.Recall, pre-
cision, nDCG, and MAP require relevance judge-ments. TREC pooling (V oorhees, 2000) aggre-
gates top-ranked documents across systems and
judges only the pool, but is biased toward in-pool
systems, treats unjudged documents as irrelevant,
and does not scale to multi-million-passage cor-
pora that re-index frequently (Zobel, 1998; Buck-
ley et al., 2007). Novelty, diversity, and nugget-
based test collections move the unit of evaluation
from documents toward subtopics or information
nuggets (Clarke et al., 2008; V oorhees, 2003; Pavlu
et al., 2012). Re:CAP adopts this topic-level unit,
but does not build reusable qrels (query relevance
judgements) or estimate absolute recall: it probes
a single deployed pipeline for positive evidence of
missed topics.
RAG evaluation frameworks.RAGAS (Es et
al., 2023), ARES (Saad-Falcon et al., 2024),
RGB (Chen et al., 2024), and RAGChecker (Ru
et al., 2024) score supplied retrieval and genera-
tion for faithfulness, answer relevance, context rel-
evance/recall, and module-level diagnostics. They
are complementary to Re:CAP: they ask whether
the answer is supported by the retrieved context;
Re:CAP asks whether relevant facets existoutside
that context and returns an actionable gap inven-
tory.
Coverage-oriented RAG evaluation.Closest
to Re:CAP are recent coverage-oriented RAG
evaluations. Xie et al. (Xie et al., 2025) de-
compose open-ended questions into core, back-
ground, and follow-up sub-questions; CRUX eval-
uates whether retrieved contexts cover human-
grounded information needed for long-form gen-
eration (Ju et al., 2025); CoverageBench assem-
bles coverage-oriented test collections (Samuel et
al., 2026); and LANCER optimises reranking for
nugget coverage (Ju et al., 2026). These works
assume pre-specified sub-questions, summaries,
nuggets, or coverage-aware training/evaluation tar-
gets. Re:CAP instead induces a topic registry from
D0andA0, then actively probes for missing topics
without per-query qrels or pre-authored facets.
LLM-as-judge and iterative retrieval.LLM rel-
evance judging shows mixed results: studies re-
port strong agreement with preferences or TREC-
style assessments (Faggioli et al., 2023; Thomas et
al., 2024; Upadhyay et al., 2024a,b), while others
warn against replacing human qrels with LLM la-
bels (Wang et al., 2023; Soboroff, 2025). Re:CAP’s
judge performs novelty classification against a per-

query topic registry, not absolute relevance scor-
ing; we validate it against held-out qrels ( 1.4%
judge-rejection on missing-gold candidates). Self-
RAG (Asai et al., 2023), FLARE (Jiang et al.,
2023), and IRCoT (Trivedi et al., 2023) use itera-
tive retrieval toimprove generation; Re:CAP repur-
poses related mechanics forevaluation, auditing a
fixed pipeline after the fact. Pseudo-relevance feed-
back (Lavrenko and Croft, 2001; Abdul-Jaleel et
al., 2004) and LLM query expansion (Nogueira et
al., 2019) provide our baseline references. Table 7
below summarises the positioning.
B.1 Positioning against closest prior work
Table 7 compares Re:CAP to four families of prior
work along five capabilities.
Reading the columns.RAG metrics(RAGAS,
ARES, RGB, RAGChecker) score retrieval indi-
rectly through answer faithfulness and answer/con-
text relevance, and typically require reference an-
swers or per-query reference contexts.Coverage
evaluation(sub-question decomposition, CRUX,
CoverageBench, LANCER, nugget-based test col-
lections) audits coverage directly, but assumes pre-
specified sub-questions, summaries, or nuggets au-
thored offline.Iterative RAG(Self-RAG, FLARE,
IRCoT) issues follow-up queries toimprove the
generated answer, not to evaluate the retriever, and
does not surface a gap inventory.TREC-style pool-
ingproduces reusable qrels by aggregating top-
ranked documents across many systems, but is ex-
pensive to mount per deployment and does not
target a single pipeline.
Conceptual precedent.The closest conceptual
precedent for Re:CAP is nugget-based evalua-
tion (V oorhees, 2003): a fixed inventory of atomic
information units against which a system is scored.
Re:CAP is inspired by this style of decomposition
– measuring coverage in terms of discrete informa-
tion units rather than whole-document relevance –
but differs in two ways. First, the units (topics) are
inducedfrom (D0, A0)and grown across iterations,
rather than pre-authored as gold. Second, Re:CAP
does not score against a fixed nugget set; gap prob-
ing extends the registry by generating questions
that may surface novel topics, and novelty judg-
ing determines whether the retriever could have
reached them. The audit signal is thegrowthof
the registry under probing, not its overlap with a
held-out list. Re:CAP is the only entry in Table 7
that audits coverage on a single deployed pipeline,needs no per-query labels, and returns an actionable
per-query gap inventory.
Capability RAG metrics Cov. eval. Iter. RAG TREC poolRe:CAP
Audits retrieval coverageindir.✓—✓ ✓
Probes for missing content — offlinegen.—✓
Needs no per-query labelssome—✓—✓
Works on one deployed system✓—✓—✓
Output: gap inventory — facets — qrels✓
Table 7:Re:CAP positioning. RAG metrics include RA-
GAS/ARES/RGB/ RAGChecker; Cov. eval. includes sub-
question, CRUX, CoverageBench, and nugget-coverage work.
indir.= indirectly (via answer quality);gen.= for generation,
not for evaluation.
C Default Re:CAP Configuration
Re:CAP exposes about a dozen knobs (loop control,
generator design, LLM choice per component); the
deployed default sets each to a specific value, most
of them backed by an ablation reported elsewhere
in this appendix. Table 8 lists every default with a
pointer to its justifying ablation, and Table 9 breaks
down the model and sampling temperature used at
each pipeline step.
Choice Default Ablated in
Gap-Qs / iteration 5 App. M
MAX_ITER 3 App. M
MAX_TOPICS 50 —
Gap-Q gen. inputQ+T+LApp. M
Probe roles (Step 3) 5 App. M (LOO)
Anti-collapse guard on App. M.2 (LOO)
Failure memory last iter App. M.2 (LOO)
Shared pipeline LLM yes App. M
Expansion top-m50 App. M
Judge temperature 0 —
Gap-Q gen. temperature 0.3 —
Retriever hybrid BM25 + dense§4.1
Table 8:Default Re:CAP configuration. Hybrid retrieval
uses RRF over BM25 and MiniLM-L12 dense embeddings.
LOO= leave-one-out.
The three remaining “—” rows are not ablated.
MAX_TOPICS( = 50 ) is a safety cap on topic-
memory size that bounds prompt growth across
iterations; it sits above the per-query topic count
on every benchmark run, so on those workloads it
acts as a guardrail rather than an active parameter.
It is not inert in production: it binds on 8.0% of
AI Search and 40.5% of NEWS QA queries (Ta-
ble 10), truncating the registry and bounding |G|
and gap rate on those queries, so wide- D0deploy-
ments should raise it before reading either signal.
The twotemperaturerows are deterministic-by-
design conventions (judge / extractors at 0; gen-
erator at 0.3to give the role taxonomy room to
diversify), documented in the “Why” column of
Table 9 rather than ablated.
C.1 Post-hoc topic deduplication (Step 5b)
Parallel judging in Step 5 emits one verdict per
candidate document, so the same underlying topic

Component Model T. Why
QA reader (Step 1) GPT-5.2 0 strongest
Topic extract. (Step 2) GPT-4.1 0 deterministic
Doc reconcile (Step 2) GPT-4.1 0 deterministic
Gap-Q gen. (Step 3) GPT-4.1 0.3 diversity
Gap judge (Step 5) GPT-4.1 0 deterministic
Table 9:Per-component LLM configuration. Non-reader
components share a singlepipeline model(GPT-4.1) and are
ablated jointly in the pipeline-model swap (App. M, E3.5).
T. = sampling temperature.
is frequently surfaced by several candidates within
a single iteration under slightly different labels
(“Madonna referred to as the Queen of Pop” vs.
“Madonna’s Queen of Pop title”). Without dedu-
plication these inflate the topic count and the gap
inventory G. Re:CAP collapses them in two stages
before topics enter the registryT.
Stage A — fuzzy string clustering (determinis-
tic).The per-iteration novel verdicts (NEWTOPIC
and SUBTOPIC) are normalised (lowercased,
whitespace collapsed) and greedily clustered using
Python’sdifflib.SequenceMatcher ratio with a
threshold of 0.85. For each cluster, the first label
is taken as canonical, evidence document IDs are
merged across cluster members, and a SUBTOPIC
verdict (with its parent) dominates if any member
produced one. Stage A is deterministic and runs
without an LLM call.
Stage B — semantic merge against T(LLM).
The Stage A survivors are passed to a single LLM
call (GPT-4.1, T= 0 , batches of ≤30 labels)
together with the current registry T. This dedu-
plication call — distinct from the Step-5 novelty
judge — performs two jobs at once: (i) it clusters
semantically equivalent new labels — including
instance-of-the-same-category collapses (e.g. “In-
dia’s World Cup wins” and “Australia’s World Cup
wins” both fold under “Cricket World Cup winning
countries”) — and picks the most category-level
label as canonical; (ii) for each cluster, it marks
overlaps_existing = true together with the
existing_topic_id when the cluster duplicates
a topic already in T. Clusters with overlaps_ex-
isting attach their evidence to the existing topic
and arenotcounted as new gaps; the remaining
clusters are added to Tand to Gwith the merged
evidence set.
D Deployment Recipe
For teams operating a RAG system on a proprietary
corpus, Re:CAP plugs into the existing pipelineas an out-of-band auditor: (1) on a representa-
tive query sample, run Re:CAP against the cur-
rent production retriever to establish a baseline
(gap inventory, gap rate, Nqenvelope, per-corpus
cost); (2) before promoting an index, embedder,
or ranker change, re-run on the same sample and
compare; (3) a drop of ≥5pp in mean recall ex-
ceeds 10× the0.47% run-to-run noise floor of §4.5
and lies in the discriminative range exercised by
the BM25 sensitivity ladder of §4.3, so it can be
read as a real degradation rather than noise; we also
watch unreachable-gold share but report no thresh-
old for it, having measured no run-to-run floor for
that quantity; (4) feed the per-query gap inventory
into operations dashboards so on-call engineers see
specifically which topicswere missed rather than
only an end-to-end faithfulness score.
Why hybrid retrieval.Re:CAP’s tighter, entity-
anchored probes can under-explore when paired
with a single-channel retriever; the dense channel
in hybrid supplies the breadth that BM25 alone
cannot. Hybrid is therefore the recommended de-
fault. The choice of initial retriever is not a free pa-
rameter — on HotPotQA withdense D0, Re:CAP
regresses by 3.6pp matched- Nqbecause the gap-
Q generator inherits dense-side biases and wastes
early iterations on semantically related but non-
gold documents that BM25 would have surfaced
via bridge-entity lexical match (see Limitations,
Initial-retriever choice can cause cell-level regres-
sions). The initial retriever should be selected via
standard recall-at- kdiagnostics, with Re:CAP then
layered on top.
Attributing a metric move: retrieval drift vs.
judge or reader drift.Because Re:CAP has no
per-query gold in production, a rise in its moni-
toring signals is only actionable if a retrieval-side
cause can be separated from drift in the LLM com-
ponents themselves. Two properties make that sep-
aration possible. First, the components are pinned:
the novelty judge runs at temperature 0and the
gap-Q generator at 0.3(Table 9), which removes
deliberate sampling variance once the model ver-
sion is held fixed. Pinning does not make the com-
ponents deterministic — re-running the judge on
identical prompts reproduces its own novel/not ver-
dict on 94.0% of candidates (App. I). The opera-
tive threshold is therefore the magnitude of a move
relative to that noise, not its presence. Second,
that residual non-determinism is bounded and mea-
sured end-to-end — across three independent runs

(§4.5, Table 5) Re:CAP recall has CV 0.47% and
gap rate CV 1.46% , i.e. per-verdict disagreement
largely averages out at the level of the reported
metrics. Under a fixed pipeline, a move exceeding
those bounds cannot be produced by LLM non-
determinism alone and is therefore attributable to a
change on the retrieval side — an index refresh, an
embedder swap, or a corpus shift.
Two caveats govern how the thresholds should
be set. Gap count |G|responds to retriever quality
in the expected direction (§4.3) but is the noisiest
of the three signals, at CV 5.04% ; on the MuSiQue
ladder the BM25–hybrid separation is only 1.5×
that run-to-run standard deviation, so |G|resolves
large regressions and not the adjacent-configuration
differences that ladder spans. Gap rate is far more
stable ( 1.46% ) but saturates on high-hop work-
loads, where it is best read as a coverage-floor
breach. In practice, alert on gap rate for gradual
movement, use |G|for magnitude once a breach
fires, and treat any |G|move under roughly 2×its
noise floor as unresolved rather than as evidence
of stability. Separately, none of these bounds sur-
vive amodel versionchange: upgrading the judge
or generator re-baselines every signal, so the pre-
change sample must be re-run to re-establish the
envelope before the new version is trusted. This
is the same re-baselining discipline step (2) above
prescribes for retriever changes.
Cost-sensitive deployments.At $ 0.59–$2.05/q,
auditing a thousand queries weekly costs roughly
$600–$2,100 per week at default settings. Tighter
monitoring budgets can swap all pipeline compo-
nents to GPT-4.1-mini for a 4.6× cost reduction
at−0.51 pp paired recall on HotPotQA — at the
edge of the ±0.42 pp run-to-run noise floor of §4.5
(full ablation: App. M, E3.5).
E Production Deployment Case Studies
This appendix gives the full case-study description
for the live production audit summarised in §4.5.
We report two unrelated production deployments —
“AI Search” (the original case study) and “NEWS
QA” (a second cohort added to test transfer across
deployments with very different D0widths). Both
use the paper’s default configuration unchanged.
Table 10 summarises the two cohorts side by side;
the rest of this appendix details AI Search first and
then the NEWS QA deltas.AI Search NEWS QA
Queries (success / total)200/200 200/200
D0width (mean / median / max)10(fixed)47.2/33/110
Topics/q: mean / median25.1/20 31.2/34
MAX_TOPICS=50saturated16/200(8.0%)81/200(40.5%)
Gap rate: mean / median / @1.0 0.991/1.00/94.5% 0.955/1.00/83.5%
Topic source share: Step 1 / 2 / 521.9/0.6/77.4% 17.3/4.4/78.2%
Natural convergence59/200(29.5%)55/200(27.5%)
Mean iterations (median)2.55(3)2.13(2)
Per-q cost: median / mean $1.68/ $2.73$1.79/ $2.03
Per-q LLM calls: mean / max410/1,703 333/1,114
Judge share of calls98.0% 97.6%
Table 10:Two production cohorts, same default Re:CAP
configuration. Gap rate and judge dominance reproduce
across both deployments; NEWS QA’s ∼5× wider D0shifts
more coverage to doc-reconciliation (Step 2) and pushes more
queries to the topic cap, but the iterative loop (Step 5) still
contributes∼78%of topics on both.
Deployment.“AI Search” is an enterprise re-
search assistant deployed over a proprietary, fre-
quently re-indexed knowledge corpus ( ≈123 M pas-
sages) that mixes news feeds, internal research,
and regulatory filings — exactly the non-stationary-
with-multiple-sources-of-truth setting motivating
the paper. Live queries span short factoid lookups
(e.g. entity name disambiguation), multi-document
analyst briefs (e.g. “what’s the Street saying about
X?”), and multi-paragraph country / sector intel-
ligence summaries (the long-form tail of the dis-
tribution). Per-query gold qrels do not exist in
this setting (corpus churn precludes exhaustive of-
fline labelling), which is why Re:CAP was devel-
oped in the first place; consequently, we report the
audit’s intrinsic signals — gap rate, topic-source
provenance, convergence, cost — rather than recall
against gold.
Configuration.The audit uses the paper’s rec-
ommended default (Table 8) without modification:
hybrid retrieval combining OpenSearch BM25
and an OpenAI text-embedding-3-large 1024-
d dense index; GPT-5.2 reader; GPT-4.1 for the
topic extractor, doc-topic reconciler, gap-Q gener-
ator (at T= 0.3 ), and novelty judge (at T= 0 );
MAX_ITER = 3 ;5gap-Qs per iteration; expan-
sion top- m= 50 per gap-Q; MAX_TOPICS =
50.
Sample. 200representative production queries
drawn from the live query log without filtering
on query type, length, or expected complexity.
We deliberately included multi-paragraph briefing
queries that drive the cost long-tail rather than ex-
cluding them as outliers.
Stability.All queries executed without any no-
ticeable LLM failures. The only implementation
issue encountered were OpenSearch concurrency
related. Mean iteration count is 2.55 (median 3,

σ0.71 , range 1–3);59/200 = 29.5% of queries
converge naturally before the iteration cap, with
mean convergence iteration 2.20 implying that au-
dits for internal data might benefit from an in-
creased MAX_ITERS cap.
Coverage signal.Mean total topics per query
is25.1 (median 20,σ16.1 ,min 4 ,max 50 );16
queries saturate the MAX_TOPICS = 50 cap
(long-form briefs). The per-query gap rate (frac-
tion of topics surfaced beyond what the initial an-
swer + reconciliation already cover; equivalently,
|G|/|T final|) has mean 0.991 ;189/200 = 94.5%
of queries have gap rate= 1.0 , and the lowest
single-query gap rate is 0.60 (a short factoid query
with substantial topic overlap between the initial re-
trieval and gap expansion). In aggregate, Re:CAP
surfaces at least one previously-unretrieved topic
on effectively every production query sampled.
Where topics come from.Table 11 breaks down
the5,017 total topics across 200queries by their
source step in the loop. 77.4% of the per-query cov-
erage map is contributed by Step 5 (the gap judge’s
NEWTOPICverdicts on documents fetched in gap
iterations), with the remaining 21.9% coming from
Step 1 (the topic extractor on the initial answer)
and0.6% from Step 2 (doc-topic reconciliation on
the initial retrieval). The doc-reconciliation share
is much lower than on the benchmark runs – pro-
duction reader answers are comprehensive enough
that the reconciliation step rarely surfaces topics
the answer omitted, leaving virtually all coverage
discovery to the iterative loop.
Source step Topics (200 q) Share
Step 1 — Topic extractor onA 0 1,101 21.9%
Step 2 — Doc-topic reconciliation 32 0.6%
Step 5 — Gap judge (NEWTOPIC) 3,884 77.4%
Total 5,017 100.0%
Table 11:Topic provenance on the 200-query production au-
dit. Step 5’s 77.4% share is the iterative loop’s added value:
more than three-quarters of the per-query coverage map is sur-
faced only because Re:CAP probes beyond the initial answer.
LLM usage and cost.Mean LLM calls per query
is410 (median 294,max 1,703 ); the judge ac-
counts for 98.0% of calls ( 80,422 of82,036 across
the audit), matching the judge-dominance pattern
of Table 4. Mean input tokens per query is 1.18 M
(median 699k); mean output tokens 28.4 k (me-
dian19.8 k). Per-query cost from the LLM-trace
rollup (over the 190queries with complete teleme-
try;10queries lost their llm_trace payload dur-
ing the resume that closed the deterministic-failuretail) has median $ 1.68, mean $ 2.73 (Q1= $0.56 ,
Q3= $3.96 ,max = $24.86 ). The median sits
within the benchmark envelope of §4.5 ($ 0.59–
$2.05/q); the mean sits ∼33% above the upper
bound, driven by the long-form multi-paragraph
briefing queries in the long tail (the single $ 24.86
query issued 1,643 LLM calls across 3iterations
to map a19-topic intelligence brief).
NEWS QA cohort: deltas from AI Search.
“NEWS QA” is a second enterprise deployment
over a multi-vendor news QA corpus ( ≈70M pas-
sages, 256-dtext-embedding-3-large dense in-
dex). We re-used the production pipeline’s own
retrieved set as D0(mean 47.2 docs/q, median
33,max 110 , vs≈10 for AI Search) and its A0
as the reader output, so the audit reflects exactly
what the deployed system surfaced. All other de-
faults match Table 8. Gap rate has mean 0.955 ,
167/200 (83.5% ) atrate= 1.0 ,197/200 (98.5% )
with at least one gap. The ∼5× wider D0shifts
the topic-source mix in two ways: doc-topic rec-
onciliation (Step 2) climbs from 0.6% to4.4% of
topics, and the MAX_TOPICS=50 cap is satu-
rated on 81/200 (40.5% ) of queries (vs 8.0% on AI
Search), so reported gap counts on NEWS QA are
right-censored more often. Mean iteration count is
2.13 and natural convergence is 27.5% , both within
a percentage point of AI Search. Per-query cost
from the LLM-trace rollup has median $ 1.79, mean
$2.03 (Q1= $0.51 ,Q3= $3.16 ,max = $6.87 ).
The judge dominates calls at 97.6% (vs98.0% on
AI Search), matching the benchmark cost pattern.
Worked example (i) — AI Search ( idr news ).
Query.“idr news”. D0.10documents from the
deployed hybrid retriever (JPM Global Markets
research; April 2026 slice). A0topics.Bank In-
donesia monetary policy and hawkish stance on
IDR; FX pressures and market dynamics; ana-
lyst forecasts for 2026 ; macro/fiscal headwinds;
SRBI liquidity tightening.Iter-1 gap-Qs (sam-
ple).What recent developments have occurred
regarding SRBI and its role in stabilizing the IDR?;
What analyst forecasts contradict the prevailing
outlook for the IDR exchange rate in 2026 ?New
topics added.Iter-1; 4SUBTOPICand 1NEW-
TOPICafter dedup. SRBI-driven liquidity tighten-
ing on deposit and credit growth; IDR depreciation
on property developers’ costs and debt exposure;
Middle-East de-escalation on Indo CDS and IDR
outlook; MSCI market-classification reforms on In-
donesian equities; equity-market sentiment under

macro risk.Iter-2. 5further probes, no novel ver-
dicts; the loop converges.Interpretation.The
deployed reader returned a coherent top-line brief-
ing; the loop doubled topic coverage by surfacing
five distinct second-order dimensions the initial an-
swer left implicit, without leaving the production
retriever’s reachable candidate pool.
Worked example (ii) — NEWS QA (Tyler Tech-
nologies). Query.“Why has Tyler Technologies’
stock price been declining and what are the issues
affecting its performance?” D0.7documents
from production (Reuters and Benzinga, February
2026 ).A0topics.Government budget cuts on
revenue; Q 4earnings and revenue miss; slower
cloud migration and extended procurement cycles;
analyst downgrades and reduced price targets; fi-
nancial metrics indicating weak capital efficiency.
Iter-1 gap-Qs (sample).What evidence exists
that contradicts the claim that Tyler Technologies
has missed analyst expectations or delivered dis-
appointing financial results?;Why has Tyler Tech-
nologies’ stock price been declining, regardless of
specific economic conditions or government budget
changes?New topics added.Iter-1, all NEW-
TOPICafter dedup. Share-repurchase plan as a
capital-allocation response to perceived undervalu-
ation; acquisition ofFor The Recordexpanding the
court-technology portfolio; tail risks from cyber-
attacks, AI vulnerabilities, and regulatory changes.
Iter-2.Two additional candidates clear the judge
but fold into existing topics in post-iteration dedup;
the loop converges with new_topics_found = 0 .
Interpretation.The deployed pipeline correctly
identified the headline drivers of decline; the loop
additionally surfaced the company’s response (buy-
backs, M&A) and an unstated risk register, both
directly relevant to thewhy is the stock declining
query and neither present inA 0.
Scope of these case studies.These audits con-
firm that Re:CAP runs end-to-end on two unrelated
live production deployments with no production-
specific code path, that the gap discovery rate ob-
served on benchmarks transfers to live traffic, and
that the iterative loop — not the initial answer —
contributes the bulk of the per-query coverage map
across both cohorts despite a ∼5× difference in
D0width. They do not provide calibrated recall
numbers (no production gold labels) or variance
estimates (single run per cohort). Those roles re-
main with the four benchmark datasets reported in
§§4.1–4.5. Judge accuracy on production trafficis bounded directly by the human read of App. F,
and indirectly by the 1.4% judge-rejection rate of
App. O. That human read samples only documents
the judge called gap-filling, so it bounds the judge’s
precision on this traffic and not the gaps it missed.
F Human Evaluation on the Production
Cohorts
§5 evaluates recovered documents on public bench-
marks, where relevance labels exist. This appendix
repeats the exercise on the two production cohorts
of App. E, where no relevance labels exist and the
corpora are proprietary.
Setup.We sampled 180 documents that
Re:CAP’s novelty judge marked as gap-filling,
drawn from 180distinct queries and split evenly
between the two cohorts. Three annotators saw
the query, the deployed reader’s D0-only baseline
answer, and the document text, blinded to the
judge’s verdict and to the cohort. The rubric is the
binarynew_info/covereddecision of §5.
Result. 75.6% of judge-identified gap documents
on AI Search and 72.2% on NEWS QA add infor-
mation the baseline answer lacks ( 73.9% overall,
[67.0–79.8]; Table 12).
Cohortnnew Rate % [95% CI]κ
AI Search 90 6875.6[65.8–83.3]0.84
NEWS QA 90 6572.2[62.2–80.4]0.88
Overall 180 13373.9[67.0–79.8]0.86
Table 12:Human-evaluation verdicts on judge-identified
gap documents from the two production cohorts.newis the
majority-votenew_infocount;95%CIs are Wilson.
G Full Cross-Dataset Results (Including
TREC-COVID Matrix)
Table 13 repeats the cross-dataset compar-
ison of §4.1 and includes the full 4-cell
retriever ×generator matrix on TREC-COVID,
which we elided from the main body for com-
pactness. The TREC-COVID matrix illustrates
the structural-complementarity finding of §4.1: no
combination of retriever (BM25 / dense / hybrid)
and generator ( (Q+T) -only / five-mechanism)
closes the small negative recall delta against flat
BM25 top- 500, yet every cell maintains a ≥45%
vs-BM25-500 unreachable-gold share. The default
Re:CAP cell goes further: 21.2% of its recovered
gold remains absent even from theensembleof flat
BM25, dense, and hybrid top- 500combined (Ta-
ble 17). The negative recall delta is a regime-level
metric mismatch on pooled-graded corpora (see

Limitations,Recall@ kundercounts complementar-
ity on pooled-graded corpora), not a cell-specific
failure; the audit-relevant signal is the complemen-
tarity. Appendix H sweeps flat BM25 depth K
on HotPotQA and TREC-COVID and locates the
depth at which flat retrieval catches up.
H Recall vs. Retrieval Depth
Table 3 sweeps retrieval depth on MuSiQue and
MS MARCO TREC-DL. This appendix completes
that sweep on the two remaining benchmarks, Hot-
PotQA and TREC-COVID, over the full range of
Kfor which we hold per-query judgements (Fig-
ure 2, Table 14). The curves are recomputed from
the per-query records of the same runs that back Ta-
ble 2 rather than transcribed from it, so the two can-
not drift apart; the recomputation reproduces every
published flat-BM25 confidence interval exactly.
Deltas are paired per query and then bootstrapped
(1000 resamples, seed 42), the same estimator used
throughout the paper.
Two observations. First, Re:CAP’s advantage
is monotone decreasing in K: it is largest where
retrieval is shallow ( +29.1 pp on HotPotQA and
+20.6 pp on TREC-COVID at K=10 ) and shrinks
as the flat baseline is permitted to see more of the
corpus. This is the expected shape — Re:CAP
spends its budget onwhichregions of the corpus to
probe, an advantage that necessarily erodes once a
flat run is allowed to read the corpus exhaustively.
Second, the depth at which the flat baseline
catches up separates the two corpora, and the sep-
aration is the scope boundary reported in §4.1.
On HotPotQA, Re:CAP at 249 documents per
query still leads flat BM25 at K=200 by6.1pp
and is statistically indistinguishable from K=500
while reading half as many documents. On TREC-
COVID, Re:CAP at 378 documents matches flat
K=500 within noise and is then overtaken at
K=1000 , by10.6 pp at 2.6× its document budget.
The crossover is reported in full because on a
pooled-graded corpus Recall@ krewards depth me-
chanically, since deeper pools intersect more of the
judged set (see Limitations,Recall@ kundercounts
complementarity on pooled-graded corpora). The
audit signal Re:CAP is built to produce — gold
that no flat run surfaces at any depth we can af-
ford in production — is unchanged by it: 21.2% of
Re:CAP’s recovered gold on TREC-COVID is ab-
sent from theunionof flat BM25, dense, and hybrid
top-500(Table 17). Aggregate recall at K=1000 istherefore not the quantity the TREC-COVID result
rests on.
I Cross-model re-judgement
The gap judge and the answer generator share a
model family, so the judge’s accept/reject deci-
sions are re-run against a newer generation to test
whether they are family-specific. We draw a strati-
fied sample of n= 1000 judged candidates from
the runs whose judge calls are stored verbatim: ev-
eryNEW_TOPICandSUB_TOPICcandidate in the
pool, withIRRELEVANTandREDUNDANTsubsam-
pled to fill the remainder. We replaythe judge
promptagainst gpt-5.4-mini-2026-03-17 and gpt-
5.5-2026-04-24. Each model judges every candi-
date three times and votes; agreement is between a
challenger’s majority verdict and gpt-4.1-2025-04-
14’s own majority verdict on the same prompts, so
all three panels are measured the same way.
Models are run at temperature 0where the de-
ployment permits it. gpt-5.5-2026-04-24 serves
only the API default of 1, so that model is run
at that setting; the resulting sampling variance is
quantified by the self-agreement column. Reason-
ing is disabled on every deployment that supports it,
matching the configuration of the deployed gpt-4.1
judge, so that model generation rather than deliber-
ation budget is the variable under test.
The judge prompt was written for the deployed
model and may therefore disadvantage a newer one.
To test this, the relevance step was rewritten to ad-
mit intermediate (“bridge”) evidence, which multi-
hop queries require and which the newer models
were disproportionately rejecting; both challenger
models were then re-run unchanged in every other
respect. Agreement moved by +1.0 and+5.0
points and the multi-hop rejection rate was un-
changed, indicating that the disagreement reported
here is not an artefact of prompt wording.
Our production infrastructure is Azure-OpenAI-
only, so a truly cross-family (Anthropic / Google)
audit is feasible on benchmark artefacts but not on
production cohorts. The sample above is therefore
drawn entirely from benchmark artefacts.
J Pseudo-Relevance Feedback Baseline
(RM3)
We compare Re:CAP against RM3 pseudo-
relevance feedback (Lavrenko and Croft, 2001;
Abdul-Jaleel et al., 2004) as the closest classical
query-expansion control. RM3 uses canonical de-

Dataset Method Docs seen Recall [95% CI]∆vs. RR Notes
MuSiQueFlat BM25 matched-N q 238 0.610 [0.558, 0.664]−0.291matched-budget ref.
Flat MiniLM hybrid top-500 500 0.772 [0.723, 0.818]−0.129RR wins at 48% budget
Flat OpenAI dense top-500 500 0.854 [0.815, 0.896]−0.047strongest dense baseline
Re:CAP (hybrid) 238 0.901 [0.854, 0.938]—29.4%unreach. vs. BM25-500
Re:CAP (dense) 191 0.898 [0.862, 0.932]−0.003largest abs. lift vs. BM25 (+30.2pp)
HotPotQAFlat BM25 matched-N q 249 0.827 [0.775, 0.878]−0.061
Flat MiniLM hybrid top-500 500 0.864 [0.813, 0.909]−0.024only flat cell beating BM25-matched
Flat MiniLM dense top-500 500 0.773 [0.717, 0.828]−0.115supports+11.5pp claim (§4.1)
Re:CAP (hybrid) 249 0.888 [0.837, 0.939]—9.2%unreach. vs. BM25-500
MultiHop-RAGFlat BM25 matched-N q 90 0.948 [0.913, 0.980]−0.030ceiling test
Flat BM25 top-500 500 1.000 [1.000, 1.000]+0.022corpus saturates
Re:CAP (hybrid) 90 0.978 [0.955, 0.996]— ceiling reached at5.5×fewer documents
TREC-COVID†Flat BM25 matched-N q 493 0.233 [0.201, 0.267]+0.068
Flat BM25 top-500 500 0.241 [0.211, 0.271]+0.076
Re:CAP (BM25,(Q+T)gen.) 493 0.165 [0.138, 0.194] —52%unreach. vs. BM25-500
Re:CAP (BM25) 383 0.209 [0.190, 0.230]+0.044 45%unreach.
Re:CAP (dense) 298 0.178 [0.153, 0.208]+0.013 61%unreach.
Re:CAP (hybrid) 378 0.221 [0.195, 0.248] best cell−0.020vs. BM25-500;48%unreach.
Table 13:Full cross-dataset matrix including the 4-cell TREC-COVID retriever ×generator sweep. Default =hybrid retriever
with the five-mechanism generator (bold rows); the (Q+T) row marks the (Q+T) -only generator ablation.†Every TREC-
COVID cell maintains ≥45% vs-BM25-500 unreachable-gold share; the default cell adds 21.2% ensemble-unreachable on
98% of queries (Table 17). The small negative recall delta is a metric mismatch on pooled-graded corpora, not a cell-specific
failure (see Limitations).
10 20 50 100 200 500
Flat BM25 depth K (docs/query, log)0.60.70.80.9Recall
HotPotQA (n=98)
Flat BM25 @ K (95\% CI)
Re:CAP (at its docs-seen)
10 20 50 100 200 500 1000
Flat BM25 depth K (docs/query, log)0.000.050.100.150.200.250.300.35
TREC-COVID (n=50)
Figure 2:Recall vs. flat BM25 retrieval depth K, HotPotQA and TREC-COVID. Blue: flat BM25 recall at depth K, shaded
band is the 95% bootstrap CI. Red star: Re:CAP at its own mean docs-seen budget (dotted guide), with 95% CI. The y-axes are
independent — the two corpora sit in very different recall regimes — while the x-axes are shared; HotPotQA’s sweep stops at
K=500 . Re:CAP leads by a wide margin at shallow Kon both corpora and is caught between K=500 andK=1000 , earlier on
TREC-COVID, the scope-boundary case of §4.1. Paired per-query deltas with CIs in Table 14.
faults (no per-dataset tuning): kfb= 10 feedback
documents, nexp= 20 expansion terms, λorig=
0.5,min_df= 5 , maximum document-frequency
ratio0.5; stopwords, numerics, and original-query
terms are excluded from the expansion candidates.
On the bounded-evidence primary where flat re-
trieval leaves the most room (MuSiQue), Re:CAP
beats RM3 by +19.4 to+19.7 pp at top- 500and by
+26.5to+27.4pp at matched-N q. On HotPotQA-
hybrid Re:CAP matches PRF top- 500(both 0.888 )
using 249documents rather than 500, and recov-
ers7.5% of gold absent from PRF top- 500; on
MuSiQue this unreachable share rises to 28–30%.
The one Re:CAP regression cell (HotPot-dense) is
the dataset ×retriever combination already notedunder Limitations. We do not report Bo1 (Amati,
2003) separately: on our datasets it tracks RM3
within±1pp and the conclusions are identical.
K Unreachable-Gold Diversification
Table 17 reports per-dataset shares of gold docu-
ments Re:CAP surfaces via gap probing thatindi-
vidualflat top- 500baselines (BM25, dense, hybrid)
cannot reach, plus the strictest cell: theensemble-
unreachable share, i.e., gold absent from all three
baselines combined ( 1,500 docs total). This opera-
tionalises the diversification claim (§4.1): Re:CAP
retrieves a structurally distinct slice of the corpus
that no single-retriever upgrade recovers. The en-
semble shares are non-zero on every dataset shown,

200 300 400 500
Docs seen / query0.60.70.80.9RecallMuSiQue
200 300 400 500
Docs seen / query0.750.800.850.900.95HotPotQA
350 400 450 500 550
Docs seen / query0.160.180.200.220.240.260.280.300.32TREC-COVID
BM25
MiniLM hybrid
MiniLM dense
OpenAI dense
Re:CAPFigure 3:Recall vs. documents seen per query, three datasets. Flat baselines (blue circles) at their native top- Nbudgets;
Re:CAP at the deployed default (red star). On MuSiQue and HotPotQA, Re:CAP attains the highest recall at less than half the
document budget of the strongest flat baseline. TREC-COVID is the scope-boundary case: Re:CAP recall is within noise of flat
BM25 top- 500, but21.2% of its recovered gold remains unreachable by the ensemble of flat BM25, dense, and hybrid top- 500
combined, so the audit signal here is structural complementarity rather than aggregate recall. Error bars are 95% bootstrap CIs;
MultiHop-RAG omitted (corpus saturates at flat top-500). Numbers in Table 2.
DataKFlat BM25 [95%CI]∆pp [95%CI]
HotPotQA10 0.597 [0.526, 0.663]+29.1[+22.4,+35.7]
20 0.658 [0.592, 0.730]+23.0[+16.3,+29.6]
50 0.740 [0.673, 0.806]+14.8[+8.7,+21.4]
100 0.791 [0.735, 0.847]+9.7[+4.1,+15.8]
200 0.827 [0.776, 0.878]+6.1[+0.5,+11.7]
500 0.872 [0.827, 0.918]+1.5[−4.1,+7.7]n.s.
TREC-COVID10 0.014 [0.012, 0.017]+20.6[+18.3,+23.2]
20 0.027 [0.022, 0.032]+19.4[+17.2,+21.8]
50 0.056 [0.046, 0.067]+16.5[+14.4,+18.7]
100 0.091 [0.077, 0.106]+13.0[+11.0,+15.2]
200 0.144 [0.123, 0.167]+7.7[+5.4,+10.1]
500 0.241 [0.211, 0.271]−2.0[−5.0,+1.1]n.s.
1000 0.326 [0.289, 0.365]−10.6[−14.2,−6.6]
Table 14:Flat BM25 recall at depth K, and the paired per-
query delta ∆(Re:CAP −flat@K) in percentage points, for
the curves of Figure 2. Re:CAP reads 249 documents per
query on HotPotQA and 378 on TREC-COVID, so rows below
those depths favour Re:CAP on budget as well as on recall.
n.s.marks a delta whose CI contains zero.
peaking at 21.2% on pooled-graded TREC-COVID
where the gold pool is largest and most diverse.
L Controlled Gold Deletion (E2.2)
Protocol.For each MuSiQue query in the M0
slice ( n= 100 , hybrid retriever, Nq= 5, MAX_-
ITER = 3 ), we remove K∈ {1,2,all} gold
passages from D0(drop-all removes every gold
doc that naturally surfaced in the top- 10) and ask
whether Re:CAP re-discovers them through gap
probing. Nine queries have no gold passage in
their top- 10D 0and are excluded from the drop
test, leaving n= 91 per drop cell. The reference is
the no-drop run restricted to the same 91queries, at
recall 0.919 (0.901 over the full 100; the excluded
nine are precisely the queries whose gold never
surfaced).
Result.For drop- 1, drop- 2, and drop-all,
Re:CAP recovers 98.9% ,98.4% , and 100.0% ofJudge Self-agreement vs. gpt-4.1 95% CI
gpt-4.1(reference)94.0% — —
gpt-5.4-mini 89.5% 77.4% [74, 83]
gpt-5.5 95.7% 84.4% [81, 89]
Table 15:Novel vs. not-novel agreement ( κ= 0.410 (gpt-5.4-
mini), κ= 0.543 (gpt-5.5)). A candidate isnovelif the judge
assignsNEW_TOPICorSUB_TOPIC; the topic registry counts
both identically when forming |G|, so this is the only verdict
distinction any reported quantity depends on. On the full four-
way taxonomy, which organises the topic tree rather than pro-
ducing a number, agreement is 54.1% (gpt-5.4-mini), 59.9%
(gpt-5.5).Self-agreementis the mean pairwise agreement be-
tween replicate runs of the same model on the same prompts,
and bounds the second column: gpt-4.1 reproduces its own ver-
dicts on 94.0% of candidates, so agreement above that level
is not attainable. gpt-5.5 is the most self-consistent model
measured, exceeding the deployed judge. Candidates span
166 distinct queries; intervals are bootstrapped over queries
rather than candidates, since candidates from one query share
a topic registry and a baseline answer.
DatasetD 0 Re:CAP PRFN qPRF 500∆vs PRF 500
MuSiQue hybrid0.9010.636 0.704+0.197
MuSiQue dense0.8980.624 0.704+0.194
HotPot hybrid0.8880.832 0.8880.000
HotPot dense 0.786 0.827 0.888−0.102†
MHopRAG hybrid 0.978 0.953 1.000−0.022⋆
MHopRAG dense 0.974 0.895 1.000−0.026⋆
Table 16:Re:CAP vs. RM3 (PRF) across the six main cells
(n= 98 –100per row). PRF Nqmatches Re:CAP’s unique-
candidate budget; PRF 500is the unbounded baseline. Across
all six cells, RM3 lifts top- 500recall by at most 1.6pp over
flat BM25 top- 500.†The HotPot-dense regression is the cell
flagged under Limitations (Initial-retriever choice can cause
cell-level regressions): dense D0misses the bridge-entity
lexical signal that RM3 inherits from its BM25 backbone.
⋆MultiHop-RAG saturates at PRF top- 500on its 3.8k-doc
corpus; Re:CAP wins at matched-N qby+2.5to+7.9pp.
the deleted gold IDs respectively, with 97.8–100%
of queries achieving full recovery in each cell.
Paired recall on drop- 1, drop- 2, and drop-all drops
by11.9,13.5, and 10.9 pp respectively (vs. refer-

RR Unreach. share vs. flat top-500Qs≥1
Dataset gold BM25 dense hyb.ens.(ens.)
MuSiQue 231 29.4% 14.8% 22.6%10.0%21%
HotPotQA 174 9.2% 13.7% 3.7%2.5%4%
TREC-COVID 4 871 48.0% 48.0% 30.0%21.2%98%
Table 17:Per-baseline and ensemble unreachable shares:
Re:CAP-recovered gold absent from the indicated flat top- 500
baseline, or from theensembleof all three baselines com-
bined ( 1,500 docs total). All Re:CAP runs use the deployed
default (hybrid D0, five-mechanism generator). Dense- D0
Re:CAP gives shares within ±2pp of the hybrid rows (not
tabulated).RR gold= total Re:CAP-recovered (q,doc) pairs
summed across queries.Qs ≥1(ens.)= share of queries with
at least one ensemble-unreachable doc. MultiHop-RAG is
omitted: flat BM25 top- 500reaches recall 1.000 on that cor-
pus (App. G), so no gold is unreachable and the shares are
undefined.
ence 0.919 ). The recall loss isnotattributable to
failed recovery ( 98–100% recovery shown above);
it is thesecondarygold that no longer surfaces.
In the no-drop run, iterative expansion picks up
∼10 pp of gold beyond D0∩qrels by triangulat-
ing from the answer; when D0is poorer, the seed
answer is impoverished and downstream gap-Qs
find fewer new probe directions. The 11–13pp
paired loss is therefore a lower bound on the value
of a non-empty D0, not a failure of probe-driven
recovery.
The three cells are near-replicates.MuSiQue
rarely places more than one gold passage in D0: of
the91queries, 62have exactly one, 26have two
and3have three. All three cells therefore delete
identical documents on 62/91 queries, and drop- 2
and drop-all coincide on 88/91 . The spread across
the three cells ( 2.6pp) is also smaller than run-
to-run non-determinism: on the 88queries where
drop- 2and drop-all delete the same documents,
mean recall still differs by 2.7pp between the two
runs. E2.2 should therefore be read as one recov-
ery result replicated three times, not as a severity
ordering; the ordering of the cells is not resolvable
here.
Dropnrecovery (%) recall∆pp|G|
None (ref) 91 —0.919—3.6
K= 19198.9 0.800 +11.9 4.9
K= 29198.4 0.785 +13.5 5.1
K=all 91100.0 0.810 +10.9 5.4
Table 18:Controlled deletion (E2.2). “recovery” is per-query
mean (recovered ∩dropped)/(dropped). ∆pp=paired (ref
−cell)×100. Gap count |G|rises monotonically with K
(3.6→5.4 ,+51% ). All columns are over the 91queries that
had gold to delete.
10−1100
Per-query audit cost (USD, log scale)0.780.800.820.840.860.880.900.920.940.96Mean gold recall (MuSiQue, n=100)
default
($0.59/q, 0.901)
cheapest: N_q=1highest recall: m=100Re:CAP cost-quality trade-off (E3 ablation grid)
Re:CAP Pareto front
Recommended default
E3.2 sweep (Nq, gap-Qs per iter.)
E3.3 sweep (expansion top-m)
E3.4 sweep (MAX\_ITER)Figure 4:Re:CAP cost-quality trade-off on MuSiQue ( n=
100, hybrid retrieval + redesigned generator). Star =rec-
ommended default ( $0.59 /q, recall 0.901 ); dashed line =
Re:CAP Pareto front across the E3ablation grid; markers
indicate which hyper-parameter sweep each point comes from
(E3.2N q, E3.3 expansion top-m, E3.4 MAX_ITER).
M Ablations
We ablate Re:CAP along four dimensions:(i)gap-
Q generator design — the (Q+T) -only ablation vs.
the five-mechanism deployed default (§M.1);(ii)
per-lever leave-one-out within the five-mechanism
generator, attributing its lift to each of its five mech-
anisms (§M.2);(iii)the three primary loop hyper-
parameters — gap-Qs per iteration Nq, expansion
depth m, and iteration cap (§M.3);(iv)the pipeline-
LLM swap and a generator-vs-judge cost decom-
position (§M.4). Table 21 summarises the sweeps;
Figure 4 places each cell on the MuSiQue cost-
quality Pareto front.
Figure 4 visualises Re:CAP’s internal cost-
quality trade-off across the E3ablation grid on
MuSiQue. The recommended default ( Nq= 5,
m= 50 , MAX_ITER = 3, gpt-4.1 pipeline) sits at
the elbow of the Pareto front: spending 2×more
(theNq= 10 or top- m= 100 cells) buys +3–4pp
recall, while spending 3×less (the Nq= 1cell)
costs−10 pp. The front is monotone in recall up
to∼$1/q — no ablation simultaneously reduces
cost and lifts recall. The chart is intra-Re:CAP:
flat-retrieval baselines return ranked documents but
produce no coverage audit, so they are not plotted
on this axis; matched-budget recall comparisons
appear in Table 2.
M.1 Gap-Q generator design (E3.1)
Table 19 reports the full 2×2×3 matrix (retriever
×generator ×dataset). The five-mechanism gen-
erator beats the (Q+T) -only ablation on hybrid
by+1.1 pp on MuSiQue and +4.9 pp on Hot-
PotQA; MultiHop-RAG is at ceiling. Two cells
regress. The larger — MuSiQue with BM25 +

five-mechanism, −5.4 pp — is a retriever-coupling
effect: tighter probes under-explore on a single-
channel retriever, and the regression vanishes on
hybrid. The smaller is MultiHop-RAG on hybrid
(−1.0pp), where the corpus is already at ceiling.
Dataset BM25QTBM255m.hyb.QThyb.5m.
MuSiQue 0.888 0.834 0.8900.901
HotPotQA 0.813 0.864 0.8380.888
MultiHop-RAG 0.978 0.984 0.9880.978
Table 19:Gap-Q generator ablation (E3.1): Re:CAP mean
recall by retriever ×gap-Q generator.QT= (Q+T) -only
baseline;5m.= five-mechanism generator. Default in bold.
M.2 Per-lever leave-one-out (E3.1.2)
Table 20 attributes the five-mechanism generator’s
lift to each of its mechanisms via a 15-cell LOO
sweep (5 levers ×3 bounded-evidence datasets,
100q each on hybrid). Paired ∆is mean per-query
(reference recall −leave-one-out recall) in pp; pos-
itive∆meansdroppingthe lever hurts. Cells with
|∆|>0.42 pp (the variance noise floor of §4.5) are
bolded.
MuSiQue HotPotQA MultiHop
Lever Mechanism∆ ∆ ∆
A entity ledger+8.58 +4.17−1.47
B coverage topics−0.42−2.06−1.26
C probe roles−1.00−1.03−1.45
D anti-collapse +0.33+1.55−0.88
E failure memory+0.750.00−0.48
Table 20:Per-lever leave-one-out (E3.1.2). Paired ∆in pp;
positive =lever helps recall. Lever A (entity-anchored gap-
Q generation) accounts for nearly all of the five-mechanism
generator’s lift on the two bounded-evidence chain datasets;
on near-ceiling MultiHop-RAG it mildly hurts because entities
are already pinned by the question’s surface form.
Decomposition: the five-mechanism generator
isone dominant anchoring mechanism plus four
supporting refinements. Lever A’s +8.58 pp on
MuSiQue and +4.17 pp on HotPotQA recover the
entirety of the five-mechanism generator’s hybrid-
cell lift from Table 19. On MultiHop-RAG, drop-
ping Araisesrecall by 1.47 pp at near-ceiling ref-
erence: the entity-ledger prompt over-constrains
when the question itself names the relevant entities
(“between article X and article Y. . . ”).
M.3 Number of gap questions (E3.2),
expansion depth (E3.3), iteration depth
(E3.4)
We swept each of the three primary hyper-
parameters individually on the MuSiQue M0slice
(100q, hybrid, five-mechanism generator, other de-
faults held). All three curves confirm the default
Re:CAP config sits at the cost-recall knee.E3.2 Nq.Recall climbs monotonically:
0.802→0.883→0.901→0.936
forNq∈ {1,3,5,10} at per-query cost
$0.19/0.46/0.59/1.01 .Nq= 10 buys +3.5 pp
paired over the defaultN q=5at+71%cost.
E3.3 expansion top- m.Recall 0.875→
0.901→0.938 at cost $ 0.25/0.59/1.25 form∈
{20,50,100} .m= 100 buys +3.7 pp at +113%
cost.
E3.4 MAX_ITER.Iter =1drops 4.0pp paired
(outside the noise floor); iter ∈ {2,3,5} are within
±2pp of each other with iter = 2nominally best
(−0.83 pp vs. ref, roughly 2×the0.42 pp variance
noise floor of §4.5 and roughly five times smaller
than the iter =1drop). Mean iterations actually run
is1.00/1.75/1.79/1.91 respectively (anti-collapse
termination dominates for ≥2), so on the public
benchmarks reported here MAX_ITER >3buys
neither recall nor convergence depth. We default
to3for headroom: on internal production traffic
with broader, more open-ended queries we observe
occasional cases where the loop still surfaces novel
topics at iter =3that iter =2misses, suggesting the
public benchmarks under-sample the regime where
deeper iteration helps.
M.4 Pipeline-model swap (E3.5) and
generator-vs-judge cost decomposition
(E3.6)
Swapping everypipelinecomponent (generator,
judge, topic extractor, reconciler) from GPT-4.1
to GPT-4.1-mini on HotPotQA drops recall by
0.51 pp paired ( 0.888→0.884 , on the edge of the
0.42 pp noise floor) while cutting per-query cost
from $ 0.586 to $0.128 (−78% ). The reader (GPT-
5.2) is held fixed. Pipeline LLM quality is largely
interchangeable against a stronger reader, and this
is the largest cost lift available in the ablation grid.
To attribute the E3.5 saving, E3.6 isolates each
component on the same HotPotQA slice with read-
er/extractor/reconciler held at GPT-4.1. (i) Gen-
erator =mini, judge =4.1: recall 0.884 (paired
∆ = +0.51 pp), cost $ 0.537 /q (−8% ). (ii) Gen-
erator =4.1, judge =mini: recall 0.867 (paired
∆ = +2.06 pp, outside the noise band), cost
$0.145 /q (−75% ). The judge dominates both cost
and quality; the generator is interchangeable across
model tiers. This is the direct empirical attribu-
tion behind the ∼97% -judge claim in Table 4 and
behind the recommendation to keep the judge at
GPT-4.1 unless the deployment can tolerate ∼2pp
recall in exchange for the additional ∼70% cost

saving.
Ablation Sweep
E3.1 Generator design(Q+T)/ 5m.
E3.2 Num. gap-Qs 1 / 3 / 5 / 10
E3.3 Expansion top-m20 / 50 / 100
E3.4 Iterations 1 / 2 / 3 / 5
E3.5 Pipeline model 4.1-m / 4.1
E3.6 Cross-model gen./judge same / cross
Table 21:Ablation summary: the parameter sweeps reported
across §M. Per-cell numbers are in the corresponding subsec-
tion.
N Component Validation and Generality
N.1 Cross-dataset generality (E5.2)
Table 22 reports the default configuration applied
identically across four datasets with no per-dataset
tuning.
DatasetnRecall∆vs.N q|G|Iters
MuSiQue 1000.901 +0.2913.6 1.79
HotPotQA 980.888 +0.0612.3 1.4
MultiHop-RAG 980.978 +0.0309.7 1.0–1.3
TREC-COVID†50 0.221+0.02336.9 2.6
Table 22:Cross-dataset generality (E5.2): default config
applied with no per-dataset tuning. ∆is vs. flat BM25 at
matched-N q;|G|is mean Re:CAP gap count per query.
N.2 Judge ceiling and gap-Q quality from
gold labels
Judge ceiling.The primary component-level ques-
tion is how often the LLM judge rejects a candidate
that the gold qrels mark relevant. Across all 86
loss queries and 20,146 missing-gold documents
(§O, Table 23), 1.4% of missing gold reached
the candidate pool but was rejected by the judge;
on the bounded-evidence primaries (HotPotQA,
MuSiQue, MultiHop-RAG) the rate is 4.7% (3/64 ).
The judge is therefore an empirically tight upper
bound on the audit in the regime Re:CAP is de-
signed for: >95% of missing-gold losses are up-
stream of judging. This is consistent with LLM-as-
judge studies on narrow relevance tasks (Faggioli
et al., 2023; Thomas et al., 2024; Upadhyay et al.,
2024a,b), while respecting cautions against replac-
ing full human qrels with LLM labels (Soboroff,
2025).
Gap-Q quality (extrinsic).We evaluate gap-
Q quality by the most direct operational sig-
nal: whether the questions recover gold the host
retriever missed. Three converging pieces of
evidence. (i)Diversity:mean pairwise LLM-
embedding cosine 0.599 across batches on the
MuSiQue 100-query anchor (target ≤0.7 ; max0.758≤0.85 ). (ii)Role coverage:the five-role
taxonomy enforces≥3of5probe roles per batch.
(iii)Causal contribution:the per-lever LOO sweep
(§M, Table 20) attributes +8.58 pp on MuSiQue
and+4.17 pp on HotPotQA to the entity-anchored
gap-Q lever alone — the main recall liftisthe
gap-Q quality measurement under the audit’s own
objective.
N.3 Human-evaluation protocol on
ensemble-unreachable gap documents
This appendix expands the body §5 human evalua-
tion: sample design and item-and-rubric protocol.
Verdict counts are reported in Table 6 in the body.
Sample. n= 123 items from the deployed de-
fault (hybrid retrieval + five-mechanism genera-
tor). TREC-COVID: n= 100 stratified-random
sample from the 1,034 ensemble-unreachable docs
in the 50-query run, target 2docs/query cover-
ing all 50queries with ≥1ensemble-unreachable
doc. MuSiQue: census of all n= 23 ensemble-
unreachable docs from the 100-query run. Hot-
PotQA is omitted ( 4ensemble-unreachable docs,
insufficient for per-doc inference); MultiHop-RAG
is omitted ( 0, corpus saturates); MS MARCO is
omitted (BM25-only setup; no ensemble defined).
Item and rubric.Per item the annotator sees the
query, the baseline reader’s answer generated from
the deployed default’s D0alone, and the candidate
document text (truncated at ∼500 words). Blinded
to: which baseline(s) miss the document, qrels
status, dataset name. Binary rubric:new_infoif
the document contains a substantive query-relevant
fact not in the baseline answer (a new entity, a
numeric or temporal anchor, a correction, an unam-
biguating clarification);coveredif the relevant con-
tent is substantively present in the baseline answer.
All sampled docs are gold-relevant by construc-
tion (NIST TREC pool / MuSiQue dataset authors),
so the rubric doesnotadjudicate relevance, only
informational novelty.
Worked example.Figure 5 shows one annota-
tion item from the TREC-COVID slice. The de-
ployed default’s D0(hybrid top- 10) yields an A0
on paediatric COVID- 19outcomes. Re:CAP’s
gap-question generator produces an iteration- 3
constraint-relaxed probe“What are the health out-
comes for children who contract viral respiratory
infections, including but not limited to COVID-
19?”; expanded retrieval on this probe surfaces the
candidate“Laboratory Findings of COVID- 19In-

fection are Conflicting in Different Age Groups. . . ”,
which adds paediatric-specific lab markers (CRP,
WBC, procalcitonin) absent from A0. The candi-
date is qrels-positive yet missed by all three flat top-
500retrievers (BM25, dense, hybrid). Re:CAP’s
own novelty judge marks it REDUNDANT(collaps-
ing the paediatric lab pattern into a coarser parent
topic — the sub-mechanism over-aggregation fail-
ure mode of App. O); all three blinded annotators
independently rate itnew_info.
Figure 5:Annotation item TC-098 ( trec_covid__47__-
qo3p62m4 ) as shown to annotator A1. Top: query and D0-only
baseline answer. Bottom: candidate document surfaced by
Re:CAP’s iteration- 3constraint-relaxed gap-question (text in
main prose). All three annotators (blinded to retriever, judge
verdict, and dataset) selectednew_info, while the loop’s own
novelty judge had marked the document REDUNDANT.
O Failure-Mode Analysis
We classified 86loss queries(where Re:CAP cu-
mulative recall fell below the matched- Nqflat-
retrieval floor) across four datasets and 20,146
missing-gold documents. Tables 23 and 24 re-
port the breakdown. The aggregate 98.6% no-
surface / 1.4% judge-rejected split is dominated
by TREC-COVID, which contributes 20,082 of
20,146missing-gold documents.
Two-regime interpretation.On bounded-
evidence (HotPotQA, MuSiQue, MultiHop-RAG;
nloss= 43 ,64missing docs) the dominant failure
modes are early convergence ( 41/43 ), gap-Q
off-topic from bridge-entity erasure ( 19/43 ), and
low topic yield ( 30/43 ) — all addressable on thegenerator side. On pooled-graded (TREC-COVID,
nloss= 43 ,20,082 missing docs) the dominant
failures are judge-rejected ( 39/43 ) compounded
by iteration cap ( 24/43 ); both are structural
consequences of binary-novelty judging applied to
pooled-graded relevance with ∼493 rel docs/query
at a∼378-doc budget.
Targeted-recovery replay.We replayed the 33
unique bounded-evidence loss queries under the
deployed default (hybrid + five-mechanism gen-
erator). 22/33 (67%) flipped from loss to tie or
win; mean ∆recall +0.260 per query. Per dataset:
MuSiQue-BM25 7/11 (∆ = +0.197 ), HotPotQA-
BM2511/18(∆ = +0.278), MultiHop-RAG4/4
(∆ = +0.354 ). The residual ∼33% are largely
cases where the entity ledger anchors on a halluci-
nated entity fromA 0.
Datasetn loss Miss. % Judge % No-surf.
HotPotQA (BM25) 18 26 0.0% 100.0%
MuSiQue (BM25) 11 17 5.9% 94.1%
MuSiQue (hybrid) 10 15 6.7% 93.3%
MultiHop-RAG 4 6 16.7% 83.3%
TREC-COVID 43 20 082 1.4% 98.6%
Overall 86 20 146 1.4% 98.6%
AND only 43 64 4.7% 95.3%
Table 23:Missing-gold attribution per dataset.
Judge= surfaced as candidate but judge-rejected;No-
surf.= never reached the candidate pool. The AND-only
aggregate ( 95.3% no-surface) is the relevant number for the
primary operational regime.
Datasetn lEConv OffT ICap Jdg LowY WkD
HotPot (BM25) 18 18 6 0 0 15 10
MuSiQue (BM25) 11 11 4 0 1 6 4
MuSiQue (hyb.) 10 9 5 0 1 6 3
MHopRAG 4 3 4 1 1 3 2
TREC-COVID 43 1 0 24 39 0 0
Overall 86 42 19 25 42 30 19
Table 24:Failure-mode tags per dataset (non-exclusive).
EConv= early convergence;OffT= gap-Q off-topic;
ICap= iteration cap hit;Jdg= judge-rejected gold;
LowY= low topic yield;WkD=D 0recall = 0.
O.1 Qualitative success examples
Three worked examples in which Re:CAP recovers
gold that flat BM25 top- 500never reaches. Each
is a run from §4.1; gap-Qs and topic labels are
verbatim from the iteration trace.
(i) Hedged-answer disambiguation (HotPotQA,
hybrid). Query.“Kiley Dean was a back-up
singer for what famous singer, who was also known
as the Queen of Pop?”Cell. hotpotqa-hybrid ,
D0recall = 0.50 (Britney Spearsreturned but not
theQueen of Popbridge), matched- Nqflat= 0.50 ,

flat-500 = 0.50 ; Re:CAP recall = 1.00 in1iter-
ation. A0excerpt.“Kiley Dean sang back-up
for Britney Spears and Madonna . . . but they do
not state which of these singers was also known as
the Queen of Pop.”Unreachable gold (flat- 500).
142056 (Madonnabiographical article).Recov-
ering gap-Q.“Which singer is widely recognized
as the Queen of Pop in music history?”Topic as-
signed.“Madonna is referred to as the ‘Queen of
Pop’”.Interpretation.The entity ledger pinned
both candidates (Britney Spears,Madonna) but a
single concept-anchored probe on the ambiguous
Queen of Poprole surfaces the disambiguating arti-
cle that flat retrieval ranks below the top-500.
(ii) Triangulated multi-hop (MuSiQue, hybrid).
Query.“What county is the city that shares
a border with the state capital of the state
where Purrysburg is located?”Cell. musique-
hybrid ,D0recall = 0.25 (onlyPurrysburg
retrieved), matched- Nqflat= 0.00 , flat- 500
leaves 4of4supporting paragraphs unreachable;
Re:CAP recall = 1.00 in2iterations. A0ex-
cerpt.“The documents don’t provide enough in-
formation to determine this. Doc 2 says Purrys-
burg is in South Carolina, but none of the doc-
uments state South Carolina’s state capital. . .”
Unreachable gold. mq_charleston_south_car-
olina_9f68ae ,mq_forest_acres_south_car-
olina_35cf43 ,mq_wwnq_9c37a4 ,mq_purrys-
burg_south_carolina_254ff0 .Iter-1 gap-Qs.
Geographical location of South Carolina state cap-
ital Columbia; Which cities share borders with
Columbia, South Carolina; County jurisdiction of
West Columbia, South Carolina.Topics assigned.
“Location and county of Forest Acres, South Car-
olina”; “Geography and location of Forest Acres,
South Carolina”.Interpretation.A canonical
Re:CAP success: the loop triangulates Purrysburg
→South Carolina →Columbia →Forest Acres
in two iterations even though the initial answer ad-
mits insufficient context — the multi-hop pattern
flat retrieval cannot resolve in a single pass.
(iii) Cold-start bridge entity (HotPotQA, hy-
brid). Query.“What is the nationality of the
film director responsible for a 2008 American sci-
ence fantasy film based on a novel by Jeanne
DuPrau?”Cell. hotpotqa-hybrid ,D0recall =
0.00 (neither film nor director retrieved), matched-
Nqflat= 0.50 , flat- 500misses the gold biogra-
phy; Re:CAP recall = 1.00 in1iteration. A0
excerpt.“The documents provided do not identifythe specific 2008 American science fantasy film
based on a novel by Jeanne DuPrau, nor do they
give the director’s nationality . . .”Unreachable
gold.6167253 (Gil Kenanbiography).Recov-
ering gap-Qs.Director of City of Ember 2008
science fantasy film; Citizenship of Gil Kenan film-
maker.Topic assigned.“Nationality and biog-
raphy of Gil Kenan, director of City of Ember”.
Interpretation.The reader’s “insufficient context”
answer is recovered into the topic registry; the gap-
Q generator extracts the DuPrau anchor, names the
film (City of Ember), pivots to the director, and the
judge promotes the biography to a new topic — the
path flat retrieval cannot construct.
O.2 Qualitative failure examples
Three worked examples illustrating the three domi-
nant bounded vs pooled-graded failure modes.
(i) Bridge-entity erasure (MuSiQue, hybrid).
Query.“A line with Williamsburg, Main Street
and another station are in a state that’s next to
an ocean. When did that ocean start to open up?”
Cell.musique-hybrid , recall = 0.50 (matched-
Nqflat= 0.75 , flat-500 = 0.75 ),D0recall = 0,2
iterations, 7topics, 361candidates judged.Gold
never surfaced. Newport News, Virginia ;Vir-
ginia (the bridge state).Sample gap-Qs.Geo-
logical history of Atlantic seafloor formation near
New York;Earliest rifting events of Pangaea af-
fecting eastern North America;Age of Atlantic
Ocean crust adjacent to Brooklyn shoreline.Di-
agnosis.The bridge entityVirginiais required to
linkWilliamsburgtoAtlantic Ocean, but neither
D0norA0surface it (the loop guessesNew York/
Brooklyninstead), so the entity ledger never an-
chors a gap-Q onVirginia. DominantOffT +WkD
pattern on the BM25 cells of MuSiQue.
(ii) Hallucinated anchor (HotPotQA, BM25).
Query.“What is the nationality of the star of
The Monster who was also in Swordswallers and
Thin Men and The Savages?”Cell. hotpotqa-
bm25 , recall = 0.00 (matched- Nqflat= 0.50 , flat-
500= 0.50 ),D0recall = 0,1iteration, 1topic,
0NEWTOPICverdicts, 137 candidates judged.
Gold never surfaced. Zoe Kazan ;The Monster
(2016 film) (the correct bridge).Sample gap-
Qs.Which country is Toma Caragiu from, the ac-
tor in The Monster, Swordswallers and Thin Men,
and The Savages?;What is the citizenship of Toma
Caragiu . . .;Where was Toma Caragiu . . . born?
Diagnosis.The reader hallucinated“Toma Caragiu”

as the star in A0(Caragiu is a real Romanian actor
but was not in any of the listed films); the entity
ledger pinned on this fabricated name and every
gap-Q probed around the wrong person. The judge
had no opportunity to recoverZoe Kazanbecause
she was never returned as a candidate.
(iii) Pooled-graded saturation (TREC-COVID,
BM25 + (Q+T) -only). Query.“What is the
mechanism of inflammatory response and patho-
genesis of COVID-19 cases?”Cell. trec-covid
topic 38 , recall = 0.02 (matched- Nqflat=
0.12, flat-500 = 0.19 ),D0recall = 0.01 ,1iter-
ation (non-converged), 50topics (cap), 125NEW-
TOPICverdicts, 249 candidates judged. Gold
pool size: ∼1,000+ partially-relevant documents.
Gold judged-but-rejected (SUBTOPIC). 17β-
Estradiol. . . folded under “Estradiol and sex
hormone modulation of COVID-19 inflammation”;
Why is SARS-CoV-2 infection more severe in
obese men?. . . folded under “Gut-lung axis. . . ”.
Sample gap-Qs.Role of viral spike protein in trig-
gering lung epithelial cell injury;Interaction be-
tween SARS-CoV-2 and ACE2 receptor leading to
tissue damage;Contribution of neutrophil extracel-
lular traps to COVID-19 lung pathology.Diagno-
sis.The gap-Qs are genuinely on-topic and the loop
produces >100 NEWTOPICverdicts within the
topic cap, but the corpus has thousands of partially
relevant documents and most gold is never surfaced.
The judge over-aggregates novel sub-mechanisms
into existing topics — a structural mismatch be-
tween binary-novelty judging and pooled-graded
relevance.
P Future Work
•Robust entity grounding— cross-check ledger
entities against D0before anchoring; down-
grade anchors to soft preferences when D0
confidence is low. Targets the residual
∼33% of unrecovered bounded-evidence loss
queries.
•Judge-cost reduction(highest engineering
leverage per §4.5): batch judging across mul-
tiple candidates per LLM call; a lightweight
cross-encoder pre-filter dropping 70–80% of
candidates; topic-aware short-circuit when the
matched topic is already saturated.
•Multi-paraphrase gap-Qs per topic: a queued
lever for the remainingOffTfailures the entity
ledger does not catch.•Online gap discoveryas a production monitor-
ing tool, and the per-query gap-topic list as a
human-readable audit artifact that names the
retriever’s blind spots.
•Multilingual extensionandgraph-structured
topic registry(replacing the flat list with
a depth-3 rooted DAG, enabling multi-
resolution metrics and graph-merge dedupli-
cation).
•Gaps as training signal: discovered gaps as
hard negatives for retriever fine-tuning — clos-
ing the loop from evaluation back to improve-
ment.
•Gap inventory as synthetic qrels
for retriever benchmarking: each
(Q,gap-Q,judged-novel doc) triple is a
relevance label produced without human
annotation. The unreachable-gold result
(§4.1, Table 2: 9–29% of recovered gold
is absent from flat BM25 top- 500 on the
bounded-evidence primaries, rising to 48%
on TREC-COVID) shows these labels contain
documents that strong flat baselines cannot
surface, suggesting the inventory can serve
as a low-cost evaluation set for comparing
alternative — including inference-optimised
— retrievers without re-running the full audit.
Q Prompt Templates
Every prompt used by the Re:CAP loop is repro-
duced below with the structured-output schema it
is sent with, generated directly from the pipeline’s
definitions so that the listings cannot diverge from
the code that runs. The schemas constrain the la-
bel set the model may return and are therefore part
of the specification: a template alone does not de-
termine the reported behaviour. The information-
equivalence judge of §L is included for complete-
ness; it belongs to that controlled experiment rather
than the loop. Two presentation-only changes were
applied: long lines are hard-wrapped, and non-
ASCII glyphs are transliterated ( [+]for a check
mark,[-]for a ballot X, ->for a right arrow, –for
an em dash). Prompt wording is otherwise unmod-
ified. Prompts use {placeholder} substitution;
temperature and model settings are in Table 9.
The five probe roles tagged by the gap-question
generator are:
•entity-anchored— probe centred on a named
entity fromL;

Topic extraction (Step 1) (1 of 2)
# Topic Extraction
Identify the distinct information facets present in the answer below.
## Input
**Query:** {query}
**Answer:**
{answer}
## Instructions
List every distinct piece of information this answer provides in response to the query.
A "topic" is a specific facet of information that helps answer the query.
- If the answer states it cannot answer, has no information, or is empty/unhelpful -> return an
empty list.
- Only extract topics for **actual information provided**, not meta-statements about lack of
information.
- Do **not** invent topics -- only extract what the answer actually contains.
Topic extraction (Step 1) (2 of 2)
## Granularity rules
- A topic is an **information facet** (a type of information), not a single data point or
instance.
- **Paragraph test:** each topic should warrant its own distinct paragraph. If two topics would
produce overlapping paragraphs, merge them.
- If the answer lists multiple instances of the same type (e.g. several countries, people,
events), group them into **one** topic describing the category.
- [+]`Countries that have won the FIFA World Cup`
- [-]`Brazil won 5 times`+`Germany won 4 times`+`Italy won 4 times`
- Target: **1-5 topics** for a typical answer. More than 8 almost certainly means
over-enumeration.
## Output format
Each topic: a short descriptive label (7-12 words) describing what information is provided.
Topic extraction (Step 1) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
topics: list[str]
List of distinct topic labels extracted from the answer. Each label is a short phrase
(5-15 words) describing a specific piece of information. Empty list if the answer
contains no useful information.
•concept-anchored— probe centred on an ab-
stract noun phrase rather than a specific entity;
•constraint-relaxed— drops a constraint of Q
to broaden retrieval;
•constraint-tightened— adds a constraint to
narrow it;
•inverse-negation— probes for contradictory
or counterexample evidence.

Entity-ledger initialisation (Step 1b)
Extract LITERAL anchor strings from the query and the answer below.
These anchors will be used to ground follow-up search queries -- every
extracted anchor must appear VERBATIM in the source text (not a
paraphrase, not a normalisation).
ORIGINAL QUERY: {query}
INITIAL ANSWER:
{answer}
Two output lists:
1. QUERY ANCHORS -- literal named entities from the QUERY only.
Include: people, organisations, locations, titles of works, dates,
version numbers, named events. Do NOT include common nouns or
generic concepts.
2. ANSWER ANCHORS -- literal named entities or numerical/temporal anchors
from the ANSWER that are NOT already in the query. These are the
*bridge* anchors -- the names the query did not mention but the
answer discovered. They are typically the most valuable for finding
missing evidence.
Rules:
- Each anchor must be a short literal substring of the source text.
Use exact capitalisation and spelling as in the source.
- Do not include the same anchor in both lists -- query anchors take
precedence.
- Skip anchors that are too generic to be useful for retrieval
(e.g. "year", "country", "person").
- It is fine for either list to be empty.
- Target 1-6 anchors per list.
Entity-ledger initialisation (Step 1b) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
query_anchors: list[str]
Named entities (people, organisations, locations, works, dates) extracted verbatim from
the ORIGINAL QUERY. Each anchor is a short literal string that should appear in gap
questions.
answer_anchors: list[str]
Named entities or numerical/temporal anchors extracted verbatim from the ANSWER A_0 that
are NOT already in query_anchors. These are the bridge entities -- the items the query
did not mention but the answer discovered.

Doc–topic reconciliation (Step 2) (1 of 2)
# Document-Topic Reconciliation
Reconcile retrieved documents against a list of known topics.
## Input
**Query:** {query}
**Known topics:**
{topic_list}
**Retrieved documents:**
{documents}
## Process (for each document)
### Step 1 -- Entity check
Is this document about the **same entity/subject** as the query? If not ->`OFF_TOPIC`.
### Step 2 -- Novelty check (only if same entity)
Compare the document's information against the known topics list.
- Covers the same ground as an existing topic (even with different details) ->`REDUNDANT`--
cite the topic number(s).
- Provides a genuinely **new** type of information ->`NEW_TOPIC`-- provide a short label (7-12
words).
Doc–topic reconciliation (Step 2) (2 of 2)
## Granularity rules
- A "topic" is an **information facet** (a type/category of information), not a single data
point, instance, or example.
- **Paragraph test:** would this topic warrant its own distinct paragraph that does not overlap
with any existing topic? If not ->`REDUNDANT`.
- Another instance of a category already in the topic list ->`REDUNDANT`, not new.
- Example: "Countries that won the World Cup" exists -> a document about a specific country's
win is`REDUNDANT`.
- Sub-topics must represent a genuinely **different dimension**, not just deeper detail of the
same dimension.
- When in doubt ->`REDUNDANT`. Prefer fewer, higher-quality topics over many overlapping ones.
## Classification labels
| Label | When to use |
|---|---|
|`OFF_TOPIC`| Not about the same entity, or not relevant |
|`REDUNDANT`| Covers same ground as known topics -- cite topic number(s) |
|`NEW_TOPIC`| Genuinely new information facet -- provide a 7-12 word label |

Doc–topic reconciliation (Step 2) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
documents: list[DocReconciliationEntry]
One entry per document, in the same order as the input.
DocReconciliationEntry:
doc_number: int
1-based document number.
entity_match: bool
True if the document is about the same entity/subject as the query.
maps_to: list[int]
Topic numbers from the known list that this document covers.
classification: REDUNDANT | NEW_TOPIC | SUB_TOPIC | OFF_TOPIC
Classification of this document.
new_topic_label: str | None
If NEW_TOPIC or SUB_TOPIC: the label for the new topic. Otherwise null.
parent_topic: str | None
If SUB_TOPIC: the parent topic label or number. Otherwise null.
OFF_TOPIChere is the reconciler’s name for the outcome the novelty judge callsIRRELEVANT; the two steps use different
labels for the same decision and both count as non-novel.
Gap-probing question generation (Step 3) (1 of 2)
### Task
You are generating SEARCH QUERIES that will be used to retrieve documents
likely to answer the ORIGINAL QUERY below. The goal is to surface
documents that the original query phrasing has NOT yet retrieved.
ORIGINAL QUERY: {query}
--- ENTITY LEDGER ---
These are LITERAL anchor strings extracted from the query and the current
answer. The "answer anchors" in particular are bridge entities that the
initial query did not mention but that subsequent retrieval should
target.
{entity_ledger}
--- TOPICS WITH COVERAGE STRENGTH ---
Each topic shows its current evidence count. Topics marked (weak) have
little supporting evidence and should be prioritised. Topics marked
(strong) are well-covered already -- do not over-probe them.
{topic_list_with_coverage}
--- PREVIOUS GAP QUESTIONS THAT YIELDED NO NEW TOPICS ---
Do NOT repeat or paraphrase any of these. Generate questions that probe
DIFFERENT retrieval directions.
{failed_gap_questions}
--- PROBE ROLES (your batch must be diverse) ---
Every question must declare exactly one role. Across the batch the roles
MUST vary -- do not pick the same role + same anchor twice.

Gap-probing question generation (Step 3) (2 of 2)
- entity-anchored: Pivot on a named entity from the LEDGER and ask
a different question about it. The literal
anchor string MUST appear verbatim in the
question. Use this when bridge entities matter.
- concept-anchored: A query about a concept or mechanism in the
answer (no named entity). Use when the gap is
conceptual.
- constraint-relaxed: The original query with ONE specific constraint
(date, location, qualifier) REMOVED. Use to
widen retrieval.
- constraint-tightened: The original query with ONE specific anchor from
the LEDGER ADDED. The anchor MUST appear
verbatim. Use to narrow retrieval onto a
specific entity.
- inverse-negation: Ask the inverse / counter perspective (e.g.
"What does NOT support ...", "What contradicts
the claim that ..."). Use sparingly -- at most one
per batch.
--- HARD RULES ---
1. Generate EXACTLY {num_questions} questions.
2. Each question must seek information that helps answer the ORIGINAL
query.
3. For roles`entity-anchored`and`constraint-tightened`: the`anchor`
field MUST be one of the literal anchors from the ENTITY LEDGER, and
it MUST appear verbatim inside the question text.
4. Roles MUST NOT all be the same. Across the batch, use at least 3
different roles when {num_questions} >= 4.
5. If a topic has`(weak: 0 docs)`or`(weak: 1 doc)`, set
`targets_topic_id`to that topic's id on at least one question
(when possible without contradicting the diversity rule).
6. Do not repeat or paraphrase anything in the FAILED GAP QUESTIONS list.
Gap-probing question generation (Step 3) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
questions: list[RoleTaggedGapQuestion]
Gap questions for this iteration. Each must declare its role + (if applicable) the
literal anchor. Roles MUST vary across the batch -- repeating the same role with the
same anchor is invalid.
RoleTaggedGapQuestion:
role: entity-anchored | concept-anchored | constraint-relaxed | constraint-tightened |
inverse-negation
The structural probe role of this question. Must vary across the batch to enforce
retrieval-space diversity.
anchor: str | None
If role is entity-anchored or constraint-tightened: the literal anchor string from
the entity ledger that this question uses. MUST appear verbatim inside the question
text. Null otherwise.
targets_topic_id: int | None
If this question is designed to probe a specific topic id from the topic list
(typically a weakly-evidenced one), the topic id. Null if the question is
broad/exploratory.
question: str
The complete gap question used as a retrieval query.

Query reformulation (Step 4)
Generate alternative phrasings of a gap question to improve retrieval coverage. Invocations per query: 1_per_gap_-
question.
# Query Reformulation
Generate alternative phrasings of a search query to improve retrieval coverage.
**Original query:** {original_query}
**Gap question:** {gap_question}
## Instructions
Generate exactly **{num_reformulations}** alternative phrasings of the gap question that might
retrieve different relevant documents. Each reformulation should:
- Use different vocabulary while preserving the intent
- Be distinct enough from the others to explore diverse retrieval paths
Enforced response schema— the model is constrained to return exactly these fields:
reformulations: list[str]
List of reformulated query strings.
Novelty judge (Step 5) (1 of 2)
# Document Evaluation -- Gap Judge
> Context: This is an academic information retrieval evaluation. The document may reference
sensitive historical events, conflicts, or social issues. Classify objectively without endorsing
any viewpoint.
Evaluate whether a candidate document contains novel, relevant information for answering a
query.
## Input
**Original query:** {query}
**Topics already found:**
{topic_list}
**Original answer (for reference):**
{answer}
**Candidate document:**
{document}
## Process
### Step 1 -- Relevance check
- Is this document about the **same entity/subject** as the query?
- Does it contain information that **directly helps answer** the query?
- If no to either ->`IRRELEVANT`.

Novelty judge (Step 5) (2 of 2)
### Step 2 -- Novelty check (only if relevant)
Compare the document's information against the topics already found.
- Covers the same ground as an existing topic (even with different details) ->`REDUNDANT`--
cite the existing topic number.
- Provides a genuinely **new** type of information -> proceed to Step 3.
### Step 3 -- Topic labelling (only if novel)
Create a short label (7-12 words) for the new information facet.
## Granularity rules
- A "topic" is an **information facet** (a type/category of information), not a single data
point, instance, or example.
- **Paragraph test:** would this topic warrant its own distinct paragraph that does not overlap
with any existing topic? If not ->`REDUNDANT`.
- Another instance of a category already in the topic list ->`REDUNDANT`, not new.
- Example: "Results of Steelers vs Ravens games" exists -> a document about a specific game
result is`REDUNDANT`.
- Example: "Countries that won the World Cup" exists -> a document about a specific country's
win is`REDUNDANT`.
- Sub-topics must represent a genuinely **different dimension**, not just deeper detail of the
same dimension.
- [+] New: Parent "Hugo Weaving as V" -> "Production challenges with mask acting" (different
dimension)
- [-] Redundant: Parent "Sauron created the One Ring" -> "One Ring forged in Mount Doom" (same
dimension, more detail)
- When in doubt ->`REDUNDANT`. Prefer fewer, higher-quality topics over many overlapping ones.
Novelty judge (Step 5) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
verdict: NEW_TOPIC | SUB_TOPIC | REDUNDANT | IRRELEVANT | CONTRADICTORY
The classification verdict for this document.
topic_label: str | None
If NEW_TOPIC or SUB_TOPIC: a short descriptive label (5-15 words) for the new
information facet. Otherwise null.
parent_topic: str | None
If SUB_TOPIC: the parent topic label or number. Otherwise null.
existing_topics_covered: list[int]
Topic numbers from the known list that this document also covers.
reason: str
One sentence explaining the verdict.
The enum admits a fifth label, CONTRADICTORY , outside the four-way taxonomy used throughout: it fired once in 149,063
production verdicts and counts as non-novel wherever verdicts are collapsed.

Topic deduplication (Step 5b)
# Topic Deduplication
Deduplicate newly discovered information topics against the existing list.
## Input
**Existing known topics (already confirmed):**
{existing_topic_list}
**Newly discovered topics (need deduplication):**
{new_topic_labels}
## Instructions
1. **Cluster** new topic labels that are semantically equivalent (same concept, different
wording).
2. **Merge instances** of the same category (e.g. "India's World Cup wins" and "Australia's
World Cup wins" -> "Cricket World Cup winning countries").
3. For each cluster, pick the **best single label** as the canonical representative. Prefer
category-level labels over instance-level ones.
4. **Check overlap:** if any cluster is semantically equivalent to an already-known topic, mark
`overlaps_existing=true`and provide the existing topic ID number.
## Rules
- Merge labels that are instances/examples of the same information category into **one** cluster
with a category-level label.
- A new label overlaps an existing topic if it describes another instance of the same category,
or adds only minor detail to what that topic already covers.
- Only keep a new label as genuinely new if it represents a qualitatively **different** type of
information (would warrant its own distinct paragraph).
- If a new label adds specificity or a new angle to an existing topic, it is **not** an overlap
-- keep it as a genuine new topic.
Topic deduplication (Step 5b) — response schema
Enforced response schema— the model is constrained to return exactly these fields:
clusters: list[TopicCluster]
Clusters of semantically equivalent new topic labels. Each cluster becomes one topic.
Labels that overlap with existing topics should have overlaps_existing=True.
TopicCluster:
canonical_label: str
The best single label representing this cluster of topics.
merged_labels: list[str]
All original labels that belong to this cluster (including the canonical).
overlaps_existing: bool
True if this cluster is semantically equivalent to an already-known topic.
existing_topic_id: int | None
If overlaps_existing=True, the ID of the existing topic it overlaps with.

Reader (answer generation)
Generate an answer from retrieved documents. Invocations per query:1.
# Answer Generation
Answer the question using **only** the documents provided. Do not use prior knowledge or any
information not contained in the documents.
## Documents
{documents}
## Question
{query}
## Instructions
- Base your answer solely on evidence in the documents above.
- If the documents don't contain sufficient information, say so explicitly.
Information-equivalence judge (E2.2 only)
Used only for the controlled gold-deletion experiment (Appendix L); not part of the deployed Re:CAP loop.
REMOVED: {gold passage that was excluded}
CANDIDATE: {doc surfaced by gap probing}
Does CANDIDATE contain the same core facts as
REMOVED? Output YES / PARTIAL / NO + reason.