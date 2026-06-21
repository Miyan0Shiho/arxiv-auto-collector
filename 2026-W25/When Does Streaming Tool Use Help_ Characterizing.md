# When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation

**Authors**: Elroy Galbraith

**Published**: 2026-06-18 11:38:17

**PDF URL**: [https://arxiv.org/pdf/2606.20113v1](https://arxiv.org/pdf/2606.20113v1)

## Abstract
Streaming Retrieval-Augmented Generation (Streaming RAG) reduces user-perceived latency by issuing tool queries in parallel with ongoing user input, before the utterance is complete. Reported gains are aggregate, yet the mechanism's benefit is fundamentally query-intrinsic: speculation can only help when the correct tool query becomes determinable before the user stops speaking or typing. We isolate and measure this property -- tool-intent stabilization, the point in the input stream at which a speculative query's retrieval converges to the answer-bearing result. On the CRAG benchmark (1371 validation questions) we (i) measure the distribution of stabilization, (ii) derive a model-agnostic bound H on the portion of tool latency that can be hidden behind the user's remaining input, as a function of tool latency L and input cadence δ, (iii) validate against a working streaming pipeline that realized savings meet or exceed this bound, and (iv) identify which query properties predict early versus late stabilization. The study requires no model training and runs on commodity CPU hardware. We find that at a realistic operating point (L=600ms, δ=3w/s, θ=0.8), 73.9% of queries across the full benchmark admit substantial latency hiding -- a blended figure that mixes sufficiency stabilization on the 21.3% of questions where gold evidence is verbatim-present and BM25-retrievable (95.2% streamable on this favorable slice) with a grounding-free top-1-settling fallback on the remainder. On the favorable slice, φ_suf is bracketed to [0.26, 0.281] by exact and relaxed grounding -- both early. Question type produces a significant but coarse early/late split (Kruskal-Wallis p=0.017, epsilon^2=0.04), directly informing when a learned speculative trigger is worth its cost.

## Full Text


<!-- PDF content starts -->

When Does Streaming Tool Use Help?
Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented
Generation
Elroy Galbraith
SMG Labs
elroy.galbraith@gmail.com
Abstract
Streaming Retrieval-Augmented Generation
(Streaming RAG) reduces user-perceived la-
tency by issuing tool queries in parallel with
ongoing user input, before the utterance is com-
plete. Reported gains are aggregate, yet the
mechanism’s benefit is fundamentallyquery-
intrinsic: speculation can only help when the
correct tool query becomes determinable be-
fore the user stops speaking or typing. We
isolate and measure this property—tool-intent
stabilization, the point in the input stream at
which a speculative query’s retrieval converges
to the answer-bearing result. On the CRAG
benchmark (1371 validation questions) we (i)
measure the distribution of stabilization, (ii) de-
rive a model-agnostic bound Hon the portion
of tool latency that can be hidden behind the
user’s remaining input, as a function of tool
latency Land input cadence δ, (iii) validate
against a working streaming pipeline that real-
ized savings meet or exceed this bound, and (iv)
identify which query properties predict early
versus late stabilization. The study requires no
model training and runs on commodity CPU
hardware. We find that at a realistic operating
point ( L=600 ms,δ=3 w/s,θ=0.8 ), 73.9% of
queries across the full benchmark admit sub-
stantial latency hiding—a blended figure that
mixes sufficiency stabilization on the 21.3%
of questions where gold evidence is verbatim-
present and BM25-retrievable (95.2% stream-
able on this favorable slice) with a grounding-
free top-1-settling fallback on the remainder.
On the favorable slice, ϕsufis bracketed to
[0.26,0.281] by exact and relaxed grounding—
both early. Question type produces a signifi-
cant but coarse early/late split (Kruskal–Wallis
p= 0.017 ,ε2=0.04 ), directly informing when
a learned speculative trigger is worth its cost.
1 Introduction
Tool-augmented dialogue systems increasingly rely
on external retrieval to remain factual, but tool
calls add latency that disrupts conversational flow.Streaming RAG (Arora et al., 2025) addresses this
by predicting and issuing tool queriesin parallel
withthe user’s still-arriving input, then reflecting
on whether retrieved results suffice. The approach
reports sizable accuracy and latency improvements
and is modality-agnostic, applying equally to typed
input.
The reported latency benefit, however, is an ag-
gregate over a benchmark. It leaves a basic ques-
tion unanswered:for which queries can specula-
tion help at all, and by how much?The answer
is not a property of the model, the retriever, or the
speech stack—it is a property ofwhere, within an
utterance, the information that determines the tool
query first appears. If the decisive term arrives last
(“who makes the console called thePlayStation”),
no early speculative query can retrieve the right ev-
idence and the latency win collapses to zero regard-
less of system quality. If the decisive term arrives
early, almost the entire tool latency can be hidden
behind the remaining input. We initially expected
these two regimes to show up as abimodaldistribu-
tion of stabilization; empirically (§5.1) the picture
is better described as a large early-stabilizing mass
with a thin late tail—the early case is common
and the late case is the rarer failure mode—and
we report the distribution as found rather than as
hypothesized.
We study this directly. By characterizing tool-
intent stabilization as a measurable, query-intrinsic
quantity, we obtain (i) a model-agnostic bound on
the share of tool latency that can be hidden behind
ongoing input, computablebeforeany system is
built; (ii) a principled account of the failure mode;
(iii) actionable guidance on whether a learned trig-
ger is worth its cost; and (iv) results that trans-
fer across text and speech because the quantity is
modality-agnostic. Our contribution is measure-
ment and analysis rather than a new system.
1arXiv:2606.20113v1  [cs.CL]  18 Jun 2026

2 Background and Related Work
Retrieval-Augmented Generation (Lewis et al.,
2020) grounds generation in retrieved evidence;
dense passage retrieval (Karpukhin et al., 2020) and
the sparse BM25 model (Robertson and Zaragoza,
2009) are the standard retrieval backbones. Stream-
ing RAG (Arora et al., 2025) adds a speculative
dimension: aTriggerdecides when to fire a tool
query from a partial utterance, multiple specula-
tiveThreadsrun in parallel, and aReflectorjudges
whether results suffice. The idea is closely related
in spirit to speculative decoding (Leviathan et al.,
2023), which hides latency by doing work that may
be discarded; here the discarded work is a prema-
ture tool call. Our analysis is orthogonal to the
system: we quantify the headroom that any such
system can exploit, set by the query alone.
Incremental comprehension and anticipation.
That the referential intent of an utterance is
fixed incrementally—often well before the utter-
ance ends—is among the most robust findings in
psycholinguistics. Altmann and Kamide (1999)
showed that a verb such as “eat” directs antici-
patory attention to edible objects before they are
named, so partial input already restricts what can
follow. Kamide et al. (2003) extend this cross-
linguistically: in head-final Japanese, case-marked
pre-verbal arguments let comprehenders anticipate
a forthcoming argument even before the verb (the
head) is heard—pre-headanticipation (Kamide and
Mitchell, 1999). This is the human analogue of
firing a tool query from a prefix:wherethe disam-
biguating cue sits—varying with word order and
case morphology—governs how early intent stabi-
lizes.
Stabilization over a growing prefix.The en-
gineering counterpart—when a prediction over a
growing input prefix becomes safe to commit—is
the central question ofsimultaneousandincremen-
talNLP. Simultaneous machine translation must
decide, token by token, whether enough source has
arrived to emit output: wait- kand prefix-to-prefix
policies (Ma et al., 2019) and learned read/write
agents (Gu et al., 2017) trade latency against the
risk of committing before the disambiguating word
arrives—precisely our L-versus-residual tension—
and Grissom II et al. (2014) cast it as an agent that
waits until it can confidently predict the sentence-
final verb before committing, whose failure mode—
premature commitment forcing revision—is ex-actly what our volatility Vrecords. Closest to our
measurement is incremental spoken language un-
derstanding: DeVault et al. (2011) build a meaning
frame from partial ASR and quantify understanding
as a function of prefix length, and the incremental-
unit framework (Schlangen and Skantze, 2011) for-
malizes the same prefix-monotonicity and revision
concerns. These efforts target aclosedintent or
dialogue-act ontology. Our contribution is to port
the lens totool/retrievalintent over an open-world
knowledge benchmark: rather than learning a com-
mit policy, wemeasurethe query-intrinsic point at
which the retrieval target stabilizes, which upper-
bounds what any such policy could hide.
3 Problem Formalization
Let a query Qbe a word sequence q1:nof length
nwords, and let q1:tdenote the prefix of the first t
words (we measure stabilization in words through-
out, matching the whitespace tokenization used
by both the retriever and the input-cadence model
below). Let Rbe a retriever returning a ranked doc-
ument list; write dt= top-1(R(q 1:t))for the top
document retrieved from the prefix, dnfor the full-
query top document, and d⋆for the gold answer-
bearing document.
Self-consistency stabilization tsc.the smallest
tsuch that for all s∈ {t, . . . , n} ,ds=d n: ev-
ery prefix from tonward retrieves the same top
document as the full query.
Sufficiency stabilization tsuf.the smallest tsuch
thatd⋆∈top-k(R(q 1:t)): the prefix already sur-
faces the gold evidence. This is the operationally
meaningful notion, since the full query itself may
retrieve imperfectly.
Stabilization fraction ϕ=t⋆/n∈(0,1] ,re-
ported for both tscandtsuf. Lower ϕmeans earlier
stabilization and more headroom to hide latency.
Stabilization volatility V,the number of top-1
changes occurringafter dnis first reached. High
Vsignals early-commit risk: a confident-looking
early result that a later token would overturn.
Hidden latency.
H(Q;L, δ) = min
L,max 
0,n−t⋆
δ
,(1)
where Lis tool latency (ms) and δis the input
cadence in words per second, so that (n−t⋆)/δis
the residual input time (in seconds, scaled to ms)
2

still to arrive after the query has stabilized. His
the portion of tool latency that completes behind
the user’s remaining input. It bounds theinput-
hideablecomponent of the per-query perceived-
latency saving—not the total saving, since a real
pipeline can additionally overlap other costs (§5.4).
A slower cadence (smaller δ) lengthens the residual
and so raises Hwhen tool latency is the larger term.
Streamable. Streamable(Q;L, δ, θ) holds iff
H≥θL for a coverage threshold θ(e.g.θ= 0.8
hides at least 80% of tool latency). The streamable
fraction of a workload is the central deployment-
relevant statistic.
Research questions.We address four pre-
registered questions.RQ1asks how ϕis distributed
across queries; we expect a bimodal shape with an
early-stabilizing mass and a late-stabilizing tail.
RQ2asks what fraction of queries admit latency
hiding as L,δ, andθvary; we expect that for large
Lthe binding constraint is the residual input time
(n−t⋆)/δ, so aslowercadence paradoxicallyraises
the streamable fraction.RQ3asks whether the
Hbound matches a working pipeline.RQ4asks
which query features predict early vs. late stabiliza-
tion; we expect question type to be the dominant
signal, with simple lookups stabilizing earlier than
multi-hop or aggregation queries. We address RQ1–
RQ4 in §5 and include a top- krobustness sweep
(Table 1) as a sensitivity check. Dense-retriever
and cross-linguistic conditions are left to future
work. We report disconfirmations alongside con-
firmations: RQ2 is confirmed, but the RQ1 distri-
bution is right-skewed rather than bimodal (§5.1),
and the RQ4 ordering is governed by entity position
rather than reasoning complexity (§5.2).
4 Method
Data.We use CRAG, the Comprehensive RAG
Benchmark (Yang et al., 2024), Task 1&2 dev set:
2706 questions, of which 1371 are the validation
split we analyze. Each question carries up to five re-
trieved web pages, a gold answer (plus alternates),
and a question-type label.
Grounding d⋆.CRAG provides no gold-passage
label, only gold answer strings. We derive d⋆by
grounding the normalized gold answer(s) into pas-
sage text: substring match for multi-token answers,
and word-boundary match for single-token an-
swers, over the answer and its alternates. Questions
whose answer is never a literal span (false-premise,“I don’t know”, and many aggregation/dynamic
answers) areungroundableand excluded from
tsuf; we report the groundable rate explicitly. Be-
cause tscrequires no grounding, we report it as a
grounding-free robustness check alongsidet suf.
String grounding is a recall-oriented heuristic
and can produce false positives: a single-token gold
answer (a year, a number, “yes”/“no”) may match
a passagecoincidentally, registering sufficiency be-
fore the genuinely answer-bearing evidence arrives
and biasing tsuf—and thus ϕsuf—early. Multi-
token answers, which require a contiguous sub-
string match, are far less prone to this. We there-
fore read ϕsufas a lower bound on stabilization
time on the groundable subset, and treat the early-
stabilization finding as directionally robust rather
than exact.
To bound the population-selection bias this strict-
ness induces, we also run arelaxedgrounding arm
(§5.5): for multi-token answers it accepts a passage
if all of the answer’s content tokens (stopwords
dropped) appear anywhere in it—bag-of-content-
token containment, dropping only the contiguity
requirement. This is deliberately higher-recall and
lower-precision than the substring rule, so the two
armsbracketthe true groundable population and
theϕ sufthat goes with it.
Passages and retrieval.Page HTML is cleaned
to text and chunked into 120-word windows with a
20-word overlap (stride 100), capped at 400 chunks
per question. We index each question’s passages
with a pure-Python lexical BM25 (Robertson and
Zaragoza, 2009) ( k1=1.5 ,b=0.75 ) and retrieve
over every prefix q1:t,t= 1. . . n , recording dtand
the top- kset at each prefix. We sweep k∈ {1,3,5}
for the sufficiency definition. All retrieval is de-
terministic; the analysis uses the CRAG dev_v4
snapshot (the released web pages, so retrieval is
fixed and not subject to live drift). The two stochas-
tic or selection steps are seeded and reproducible:
the60-question RQ3 subset is simply the first 60
retrieved-gold questions in dataset order (no sam-
pling), and the bootstrap confidence intervals (§5.2)
use10,000resamples at seed0.
Latency model and harness (RQ3).We stream
each query as a uniform word sequence—
a controlled proxy for input cadence δ—and
sweep (L, δ, θ) analytically over the grid L∈
{100,300,600,1000} ms,δ∈ {2,3,4} w/s,θ∈
{0.5,0.8,1.0} to obtain the streamable surface
(RQ2). For RQ3 we replay a subset of questions
3

through a faithful asynchronous Streaming RAG
harness (fixed-interval Trigger, parallel Threads,
Reflector) calibrated with the per-stage latencies
reported by Arora et al. (2025) (Table 3): query
generation ≈590 ms and fuse/reflect ≈2520 ms,
in addition to the swept tool latency L. We compare
themeasuredperceived-latency saving (baseline
minus streaming) against theHbound.
Reproducibility.The pipeline is training-
free and CPU-only. Code and the de-
rived d⋆passage labels (the only non-
trivial artifact, since CRAG ships none)
are available at https://github.com/
elroy-galbraith/stablize_CRAG ; ev-
ery per-question metric and the full (L, δ, θ) grid
can be recomputed offline without re-running
retrieval.
5 Results
We analyze the 1371 validation questions of CRAG
Task 1&2. Under our string-grounding procedure,
35.4% of questions aregroundable(the gold an-
swer appears verbatim in the retrieved pages); the
remainder—dominated by false-premise items and
answers that never occur as a literal span (many
aggregation and dynamic questions)—are excluded
from sufficiency statistics. At k=3, BM25 surfaces
the gold passage in the top- kat some prefix for
21.3% of all questions. All reported ϕsufstatistics
are conditional on this retrieved-gold population;
ϕscis defined for every question and serves as a
grounding-free check.
This retrieved-gold subset is not a random sam-
ple of the benchmark: it is the intersection of ques-
tions whose answer is verbatim-groundableand
whose gold passage BM25 can surface from some
prefix. Both filters plausibly select for queries with
a salient, early lexical anchor, so the conditional
ϕsufstatistics should be read as characterizing this
favorable slice, not “most answerable questions”; if
anything, the selection biases the reported stabiliza-
tionearlierthan it would be on the full workload.
We therefore lean on ϕsc, defined on all 1371 ques-
tions, as the grounding-free companion to every
sufficiency claim, and we bound the magnitude
of the selection effect with a relaxed-grounding
arm below (§5.5): recovering a sixth more of the
population moves ϕsufby under 0.02, so the early-
stabilization finding survives the expansion rather
than dissolving into it.
Figure 1: Distribution of stabilization fraction ϕatk=3.
Sufficiency ( ϕsuf, dark) concentrates near zero—gold
evidence is retrievable early—while self-consistency
(ϕsc, light) is late and broad.
5.1 RQ1: How is stabilization distributed?
The two stabilization notions behave very differ-
ently (Fig. 1). Self-consistency stabilizeslate: the
lexical top-1 keeps changing until, on average, ϕsc
reaches mean 0.67 (median 0.73)— roughly three-
quarters of the way through the query. Sufficiency,
in contrast, stabilizesearly: the gold passage enters
the top- kat mean ϕsuf= 0.26 and median 0.14,
bracketed to [0.26,0.281] by a relaxed-grounding
arm (§5.5)—both early. The gap is the central
empirical observation of this study:the answer-
bearing document is retrievable from a short
prefix long before the query’s lexical top-1 set-
tles. For streaming tool use this is the favorable
regime—one need not wait for the utterance to “fin-
ish” for the right evidence to be in hand.
Theϕsufdistribution (Fig. 1) is right-skewed
with a large mass near zero and a thinner late
tail, rather than cleanly bimodal as H1 anticipated.
V olatility is modest: the post-stabilization top-1
changes at all for only 20.9% of questions (mean
V= 0.52 ), so early-commit risk is concentrated in
a minority of queries rather than pervasive.
5.2 RQ4: What predicts early vs. late
stabilization?
Question type produces a coarse early/late split in
ϕsuf, spanning a roughly threefold range across
medians (Fig. 2). A Kruskal–Wallis test across all
eight types rejects equality of the ϕsufdistributions
(H=17.0 ,df=7 ,p=0.017 ); restricting to the five
classes with n≥10 —which carry the effect—the
test is stronger ( H=13.9 ,df=4 ,p=0.008 ). The ef-
fect, though significant, issmall: the rank-based ef-
fect size is ε2=0.04 (both for all eight types and for
the five large classes), i.e. question type explains on
the order of 4%of the rank variance in ϕsuf. Con-
sistent with a small effect, the interquartile ranges
4

Figure 2: Distribution of ϕsufby question type ( k=3,
sorted by median). The 3×range across medians is real,
but box overlaps are large for adjacent types—only the
extremes (aggregation/comparisonearliest,setlatest)
are reliably ordered. Ordering tracks entity position, not
reasoning complexity.
in Fig. 2 overlap substantially for adjacent types:
the ordering is reliable only at the extremes (ag-
gregationandcomparisonearliest,setlatest— set
confirmed at n=20 under the relaxed-grounding
arm, §5.5), not as a strict cell-by-cell ranking, and
post-processing (n=5) has too few points
to place reliably. A Dunn’s post-hoc with Holm
correction makes this precise:nopairwise con-
trast survives at α=0.05 (the closest, simple
vs.aggregation , reaches pHolm=0.05 ). The
omnibus effect is therefore carried by the over-
all rank distribution, not by any single separa-
ble pair. We accordingly claim a significant but
coarse type-level effect—a robust early/lateten-
dency, not a precise per-type order. The quali-
tative ordering refines hypothesis H3 rather than
confirming it. H3 expected simple lookups to sta-
bilize early and aggregation/comparison/multi-hop
to stabilize late. We instead findaggregationand
comparisonamong theearliest-stabilizing types,
withsetquestions latest. The explanation is that
sufficiency measures when the golddocumentbe-
comes retrievable, not when the answer can be
computed: a comparison or aggregation question
often names its key entities up front, so the answer-
bearing page is retrievable early even though pro-
ducing the final answer requires post-retrieval rea-
soning. In other words, retrieval-sufficiency stabi-
lization is governed by entity position in the utter-
ance, which is not aligned with reasoning complex-
ity. This entity-position account is carried by the
well-populated classes ( simple ,comparison ,
aggregation ,simple_w_condition ), so
it does not hinge on the noisy small-ncells.5.3 RQ2: What fraction of queries admit
latency hiding?
Using t⋆=tsufwhere available and falling back to
tscotherwise, we compute the streamable fraction
over the full (L, δ, θ) grid (Fig. 3). At the cen-
tral operating point ( L=600 ms,δ=3 w/s,θ=0.8 ),
73.9% of queries admit hiding at least 80% of tool
latency.
This blended figure mixes two populations and
two notions of stabilization, and we report it de-
composed. On the 21.3% of queries with retrieved
gold—where t⋆=tsufreflects evidence actually
in hand—95.2% are streamable; on the remain-
ing78.7% , where we fall back to the grounding-
freetsc(top-1 stopped moving), 68.1% are. The
sufficiency-only number, 95.2%, is the one that
matches our framing and is the honest headline
for “evidence is retrievable in time”; the 73.9%
blend is lower because it is dominated by the tsc
majority, for which “streamable” means only that
the lexical top-1 has settled, not that gold is in
the top- k. Readers who care about evidence suffi-
ciency should weight the 95.2% figure; those who
treat top-1 stability as the deployment signal get
the blend.
The blend also counts only upside. “Streamable”
asks whether enough tool latencycanbe hidden; it
is silent on whether the speculative query wasright,
andH’smax(0,·) floor cannot represent the nega-
tive saving a mis-fire incurs (§5.4). One might ex-
pect this blind spot to bite hardest in the tsc-fallback
majority, which stabilizeslate(mean ϕ=0.68 vs.
≈0.29 on the retrieved-gold replay sample), giv-
ing a fixed-interval trigger more room to fire before
intent settles. We tested this by replaying 60 fall-
back questions (the harness previously replayed
only retrieved-gold) through the same pipeline at
L=600 ms. The expectation doesnothold: the net-
negative-saving rate is 1.7% ( 1/60 ), identical to the
1.7% on the favorable retrieved-gold population,
and mean savings are likewise comparable. The
reason is the same constant that dominates RQ3
(§5.4): the query-generation overlap is hidden re-
gardless of whether the speculativeretrievalwas
right, so late stabilization lowers Hwithout propor-
tionally raising the mis-fire rate at this operating
point. Mis-fires are thus real but rare ( ≈1.7% )
across both populations here; the streamable frac-
tion should still be read as an upper envelope on
benefit paired with this small measured downside,
and the rate may grow at larger Lor with a more ag-
5

Figure 3: Streamable fraction vs. tool latency Lat
θ=0.8 , one line per cadence δ, with exact values anno-
tated. The curves separate only once Lis large enough
for residual input time to bind; there, slower cadence
dominates (H2).
gressive trigger—a sweep we leave to future work.
Two further patterns stand out. First, for small
tool latency ( L≤300 ms) the constraint is es-
sentially never binding: 82.6% of queries are
streamable regardless of cadence—the residual in-
put time trivially covers a short tool call. Sec-
ond, and confirmingH2, when tool latency is large
(L=1000 ms) the binding constraint becomes resid-
ual input time, so aslowercadenceraisesthe
streamable fraction: from 54.5% atδ=4 w/s up to
73.9% atδ=2 w/s. Faster typists and faster talkers
leave less room to hide a slow tool.
5.4 RQ3: Does the bound match a working
pipeline?
We replayed the same 60 retrieved-gold ques-
tions through the asynchronous Streaming RAG
harness at δ=3 w/s andtwotool latencies, L∈
{600,1000} ms (Fig. 4). In the aggregate the
bound isconservativeat both: measured perceived-
latency savings (baseline minus streaming) average
1122 ms vs. the 590 ms Hbound at L=600 , and
1457 ms vs. 950 ms at L=1000 . The overshoot is
expected and informative: Hmodels only the tool
latency hidden behind input, whereas the calibrated
pipeline additionally overlaps the LLM’s query-
generation step ( ≈590 ms (Arora et al., 2025))
with the user’s input. Indeed the overshoot is al-
most exactly that step: at L=600 the bound mean
(590ms) plus the query-generation constant ( ≈
590ms) recovers the measured mean ( ≈1122 ms)
to within noise—a useful sanity check, but also
a caution that this paragraph mostly re-derives H
plus a constant rather than independently testing H
(we address that next). So a query that Hdeems
streamable will, in practice, saveat leastas much
as predicted—the property that makes Hsafe toplan with.
We are deliberately careful about the stronger
claim the bound doesnotsupport: per-query pre-
diction. At L=600 the bound saturates—59 of
the 60 queries share H=600 ms (the input resid-
ual exceeds L)—leaving no horizontal spread to
test ranking, and the rank correlation between H
and the measured saving is negligible (Spearman
ρ= + 0.14 ). We therefore re-ran theidenticalques-
tions at L=1000 specifically to break the saturation:
Hnow takes three distinct values ( 0,667,1000 ms)
with 7 of 60 queries below the cap. The spread
appears, but the correlation does not: ρ= + 0.12
over all 60, and exactly 0.00 within the unsaturated
tail. Two structural reasons explain why raising L
cannot rescue per-query prediction. First, the dom-
inant component of realized saving—the constant
≈590 ms query-generation overlap—is indepen-
dent of H, so it adds a near-uniform offset that
swamps the cross-query signal Hencodes. Second,
the failure mode isnegative, and H’smax(0,·)
floor cannot represent it. RQ3 thus confirms Has
a conservative aggregate bound butnota per-query
ranker; a predictive model would need to add the
query-generation overlap to the hideable budget
and replace the floor with a mis-fire penalty.
The negative tail makes the second point con-
crete and reconnects volatility Vto outcomes.
The single query thatlosestime—a 17-word
comparison ( t⋆=13)—costs −543 ms at L=600
and−940 ms at L=1000 , yet its Hismaximal
(1000 ms): with n−t⋆=4words of nominal resid-
ual the bound rates it fully streamable while the
fixed-interval trigger, firing before intent settles at
word 13, mis-retrieves and pays a re-fire penalty.
Notably this query has V=0 , so volatility didnot
flag it—the risk here islatestabilization under
an early-firing trigger, not post-stabilization churn.
The actionable lesson is that an early-commit safe-
guard should gate on ϕ(how late intent settles rela-
tive to the trigger cadence) at least as much as on
V; validating which signal best predicts mis-fires
at scale is the natural next step.
5.5 Robustness
Grounding (bounding the selection bias).Our
sharpest claims live on the 21.3% retrieved-gold
slice, whose selection plausibly biases stabiliza-
tion early (§5.1). To bound that bias we re-derive
d⋆with a deliberatelyrelaxedmatcher—bag-of-
content-token containment rather than contiguous
substring (§4)—which trades precision for recall
6

Figure 4: RQ3: per-question measured saving vs. the H
bound for the same 60 questions, colored by tool latency
L. Most points lie above y=x because the pipeline
also hides query-generation latency, which Hdoes
not model. At L=600 ms (light) the bound saturates—
59/60 points share H=600 ms; at L=1000 ms (dark)
Hspreads across {0,667,1000} ms, yet measured sav-
ings still do not track it (Spearman ρ= + 0.12 ). The
lone point below zero is a late-stabilizing query whose
mis-fire the floored bound cannot express—and whose
His nonetheless maximal.
and so expands the groundable population. Ground-
ability rises from 35.4% to 43.3% and the retrieved-
gold rate from 21.3% to 24.4% (a sixth more
questions). If the early-stabilization result were
largely a selection artifact, admitting these harder-
to-ground questions should pull ϕsufup sharply.
It does not: mean ϕsufmoves only from 0.26 to
0.281 (median 0.143→0.154 ). The true value is
thus bracketed in [0.26,0.281] —both early—and
the by-type ordering is preserved, now on firmer
footing: set, the latest type, grows from n=2
ton=20 and remains latest. We read the early-
sufficiency finding as signal that survives a 1.5×
population expansion, not as an artifact of the strict
matcher. (Fuzzy grounding is lower-precision, so
it if anything biases ϕsufdown; that the estimate
barely moves is the reassuring direction.)
Retriever (top- k).Varying the sufficiency top-
kshifts absolute values but preserves the qualita-
tive picture (Table 1). Larger ksurfaces the gold
passage for more questions (retrieved-gold rate
15.6%→25.0% ) and stabilizes slightly earlier
(mean ϕsuf0.327→0.248 ), as expected since a
more permissive top- kis easier to satisfy. The self-
consistency curve is unchanged by k(it depends
only on the top-1), and the central streamable frac-
tion is stable near 73–74% . The early-sufficiency /
late-self-consistency gap holds at everyk.Table 1: Top- krobustness. ϕsc(mean 0.67 / median
0.73) is independent ofkand omitted.
kretrieved-gold meanϕ suf medianϕ suf
1 15.6% 0.327 0.200
3 21.3% 0.264 0.143
5 25.0% 0.248 0.125
6 Conclusion
Tool-intent stabilization is a query-intrinsic, model-
agnostic quantity that bounds when streaming tool
use can help and by how much. On CRAG, suffi-
ciency stabilizes early on the groundable, retriev-
able subset (mean ϕsufbracketed to [0.26,0.281]
by exact and relaxed grounding, both early); ques-
tion type produces a significant but coarse early/late
split (Kruskal–Wallis p= 0.017 ,ε2=0.04 ); and
theHbound is a conservative aggregate bound
with realized savings meeting or exceeding it, at a
streamable fraction of 73.9% across the full bench-
mark. Two limitations temper these claims, and
we probed both rather than merely noting them.
First, the sufficiency results are conditional on a
favorable 21.3% slice; the relaxed-grounding arm
brackets the population effect to under 0.02 inϕsuf
(§5.5), bounding the finding as signal rather than
artifact—though a fuller recovery (dense or learned
grounding) remains open. Second, our system val-
idation establishes conservativeness in aggregate
butnotper-query prediction: at L=1000 ms, real-
ized savings do not track H(Spearman ρ=+0.12 ),
because a constant query-generation overlap domi-
nates the saving and trigger mis-fires produce neg-
ative outcomes the floored bound cannot represent
(§5.4). The downside is real but rare ( ≈1.7% net-
negative at L=600 ). Closing the prediction gap—
adding the query-generation overlap and a mis-fire
penalty—together with a dense-retriever condition
and a regression of ϕon query features are the nat-
ural next steps. Even so, the present results give
system builders a principled, training-free basis
for trigger design and a build/skip decision for a
learned trigger.
Limitations
Input timing.A uniform word stream is a proxy
for speech; real ASR pacing is variable and partial
hypotheses get revised. We bound this here by
using clean text and treat speech timing as planned
follow-up.
7

Stabilization measure.Top-1 equality is a
coarse proxy for “intent settled.” We mitigate by
also reporting sufficiency against the gold docu-
ment and by varying top-k.
Grounding. d⋆is derived by string matching, so
the groundable population excludes non-verbatim
answers; reported ϕsufstatistics are conditional on
groundability, and we report tscas a grounding-free
check.
Retriever and benchmark.BM25 is lexical and
phrasing-sensitive, and results are from a single
benchmark; a dense-retriever condition and a sec-
ond QA set are natural extensions.
References
Gerry T. M. Altmann and Yuki Kamide. 1999. Incre-
mental interpretation at verbs: Restricting the domain
of subsequent reference.Cognition, 73(3):247–264.
Siddhant Arora, Hamza Khan, Kaihang Sun, Xin Luna
Dong, Sneha Choudhary, Seungwhan Moon, Xinyao
Zhang, Aman Sagar, Sai Teja Appini, Kushal Patnaik,
Shubham Sharma, Shinji Watanabe, Anuj Kumar,
Ahmed Aly, Yuan Liu, Florian Metze, and Zhaojiang
Lin. 2025. Stream RAG: Instant and accurate spoken
dialogue systems with streaming tool usage.arXiv
preprint arXiv:2510.02044.
David DeVault, Kenji Sagae, and David Traum. 2011.
Incremental interpretation and prediction of utterance
meaning for interactive dialogue.Dialogue & Dis-
course, 2(1):143–170.
Alvin Grissom II, He He, Jordan Boyd-Graber, John
Morgan, and Hal Daumé III. 2014. Don’t until the
final verb wait: Reinforcement learning for simul-
taneous machine translation. InProceedings of the
2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 1342–1352.
Jiatao Gu, Graham Neubig, Kyunghyun Cho, and Vic-
tor O.K. Li. 2017. Learning to translate in real-time
with neural machine translation. InProceedings of
the 15th Conference of the European Chapter of the
Association for Computational Linguistics (EACL).
Yuki Kamide, Gerry T. M. Altmann, and Sarah L. Hay-
wood. 2003. The time-course of prediction in in-
cremental sentence processing: Evidence from an-
ticipatory eye movements.Journal of Memory and
Language, 49(1):133–156.
Yuki Kamide and Don C. Mitchell. 1999. Incremental
pre-head attachment in japanese parsing.Language
and Cognitive Processes, 14(5–6):631–662.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, andWen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP).
Yaniv Leviathan, Matan Kalman, and Yossi Matias.
2023. Fast inference from transformers via spec-
ulative decoding. InProceedings of the 40th Interna-
tional Conference on Machine Learning (ICML).
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng,
Kaibo Liu, Baigong Zheng, Chuanqiang Zhang,
Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and
Haifeng Wang. 2019. STACL: Simultaneous trans-
lation with implicit anticipation and controllable la-
tency using prefix-to-prefix framework. InProceed-
ings of the 57th Annual Meeting of the Association
for Computational Linguistics (ACL).
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
David Schlangen and Gabriel Skantze. 2011. A general,
abstract model of incremental dialogue processing.
Dialogue & Discourse, 2(1):83–111.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024. CRAG: Comprehensive RAG
benchmark. InAdvances in Neural Information
Processing Systems (NeurIPS), Datasets and Bench-
marks Track.
8