# RAG in a Trenchcoat: When Minimal Memory Is Enough for Agentic Systems, and When It Isn't

**Authors**: Jingyu Liu, Zongze Li, Zach Xu, Zhanhui Zhou, Tahseen Rabbani, Dawn Song, Ce Zhang

**Published**: 2026-11-17 04:16:34

**PDF URL**: [https://arxiv.org/pdf/2611.22179v1](https://arxiv.org/pdf/2611.22179v1)

## Abstract
Agentic memory research has accumulated extraction, supersession, reflection, and graph synthesis on top of retrieval-augmented generation. However, we found that almost none of those primitives earn their cost in our tested memory benchmarks: we show this with BRIDGE, a deliberately minimal RAG memory with chained observation notes, hybrid BM25 with cosine retrieval, and chronological rendering, which reaches parity within bootstrap CI with the prior SOTA on LongMemEval-S and BEAM, and beats the best memory baseline on EverMemBench by 5.7-6.3 points at matched reader. It runs at ~5-7K reader-prompt tokens per query, using a single LLM call for each ingestion or query. We argue that this parity is itself the result: standard benchmarks score what memory retrieves, not what memory enables when the agent acts. To test the latter, we introduce five adversarial probes (Rule Application +-, Pattern Induction Style / Procedural, Dynamic Multiturn) judged at generation time by a cross vendor verification chain, where each query can only be solved by applying a remembered rule or pattern it never mentions. Across 5 probes x 20 scenarios, no architecture wins universally: atomic fact extractors (LangMem) win rule application, while BRIDGE's block-summary memory wins pattern induction. This result holds across four domains, three graders, and three reader families. Each design fails where the other wins, for reasons a recall benchmark cannot see: atomic-fact extractors discard the form a demonstrated pattern takes, while block-summary memory retrieves the rule but the reader does not act on it, a gap that presenting the rule more prominently largely closes. What separates memory designs is not what they retrieve but what they enable an agent to do, and that is what agentic memory should be evaluated on.

## Full Text


<!-- PDF content starts -->

RAG in a Trenchcoat: When Minimal Memory Is
Enough for Agentic Systems, and When It Isn’t
Jingyu Liu1∗Zongze Li1Zach Xu1Zhanhui Zhou2Tahseen Rabbani1
Dawn Song2Ce Zhang1
1University of Chicago2University of California, Berkeley
∗Corresponding:jingyu6@uchicago.edu
/githubgithub.com/Jingyu6/rag-in-a-trenchcoat
Abstract
Agentic memory research has accumulated extraction, supersession, reflection, and
graph synthesis on top of retrieval-augmented generation.However, we found that
almost none of those primitives earn their cost in our tested memory benchmarks:
we show this with BRIDGE, a deliberately minimal RAG memory with chained
observation notes, hybrid BM25 with cosine retrieval, and chronological rendering,
which reaches parity within bootstrap CI with the prior SOTA on LongMemEval-S
and BEAM, and beats the best memory baseline on EverMemBench by 5.7–6.3
points at matched reader. It runs at ∼5–7K reader-prompt tokens per query, using
a single LLM call for each ingestion or query. We argue that this parity is itself
the result: standard benchmarks score what memoryretrieves, not what memory
enables when the agent acts. To test the latter, we introduce five adversarial probes
(Rule Application ±, Pattern Induction style / procedural, Dynamic Multi-turn)
judged at generation time by a cross-vendor verification chain, where each query
can only be solved by applying a remembered rule or pattern it never mentions.
Across 5 probes ×20 scenarios, no architecture wins universally: atomic-fact
extractors (LANGMEM) win rule application, while BRIDGE’s block-summary
memory wins pattern induction. This result holds across four domains, three
graders, and three reader families. Each design fails where the other wins, for
reasons a recall benchmark cannot see: atomic-fact extractors discard the form a
demonstrated pattern takes, while block-summary memory retrieves the rule but
the reader does not act on it, a gap that presenting the rule more prominently largely
closes. What separates memory designs is not what they retrieve but what they
enable an agent to do, and that is what agentic memory should be evaluated on.
1 Introduction
Long-running agents are useful when they remember things the user does not re-state. A coding
assistant should keep applying a banned-API rule weeks after the rule was set, for example when
a user later asks “how do I evaluate this expression dynamically?” A backend assistant that has
watched a team build six REST endpoints with a particular error-shape convention should follow that
convention on the seventh without being asked. We call this kind of memoryalways-on, applied
memory: the user does not mention the rule, the convention, or the preference at the time of the
request, and the assistant must surface it on its own. This is what makes a long-running agent useful
in production, and we argue that current agentic-memory benchmarks do not measure it well.
Agentic memory research has accumulated a great deal of complexity over the past two years [Zhang
et al., 2024]. Published architectures combine some subset of fact-level extraction [Chhikara et al.,
2025, LangChain-AI, 2024], supersession over temporal knowledge graphs [Rasmussen et al., 2025],
40th Conference on Neural Information Processing Systems (NeurIPS 2026).
arXiv:2611.22179v1  [cs.AI]  17 Nov 2026

reflection or rumination passes [Xu et al., 2025, Park et al., 2023, Shinn et al., 2023], working-memory
caches [Packer et al., 2024], and cross-session synthesis [Li et al., 2025, Emergence AI, 2025]. Each
primitive is motivated as a necessary departure from naive RAG [Lewis et al., 2021]. We show
that, judged bybenchmark scoresalone, almost none of them earn their cost.BRIDGEisRAG in a
trenchcoat: a thin layer of agentic-memory primitives (per-block observation notes chained for global
context, hybrid BM25 + cosine retrieval, and a two-tier chronological render) over a naive RAG
core, with no graph database, no supersession layer, and no agent-framework integration. Despite
lacking all of these, BRIDGEreaches parity within bootstrap CI with the prior reported SOTA on
LongMemEval-S [Wu et al., 2025] and BEAM [Tavakoli et al., 2026] at matched reader, and exceeds
the prior best memory baseline on EverMemBench [Hu et al., 2026a] by 5.7–6.3points. BRIDGEuses
∼5–7K reader-prompt tokens per query ( 17–154× fewer than full-context baselines), one extraction
LLM call per ingested block, and one reader LLM call per query. We argue this parityisthe finding:
on these benchmarks, the supersession, reflection, and graph-synthesis primitives do not distinguish
those systems from a basic RAG [Karpukhin et al., 2020].
This parity does not meanBRIDGEis a better memory system: it means these benchmarks cannot tell
memory designs apart.Standard benchmarks score recall, but published agentic-memory architectures
claim more capabilities that recall cannot test, such asalways-on rule application,pattern induction,
andmulti-turn consistency. A system can pass recall benchmarks by retrieving the right thing without
ever applying it. We construct a five-probeadversarial benchmarkthat closes this gap: each probe
is solvable only by generating a response that honors a memory the request never names, judged
at generation time with a cross-vendor verification chain (the judge comes from a different vendor
than the generator, Oracle, and reader; snapshots in Appendix G). Across 5probes ×20 scenarios,
we found thatno architecture wins universally: BRIDGE’s block-summary memory wins pattern
induction, while atomic-fact extractors (LANGMEM[LangChain-AI, 2024]) win rule application
and multi-turn consistency (§4.2). This result holds across four domains, three graders, three reader
families, and repeated runs (Appendices E and J).
Each design fails where the other wins, and for different reasons. Atomic-fact extractors discard
the form a demonstrated pattern takes. Block-summary memory does retrieve the rule (in all 20
scenarios we checked), but the reader does not act on it. Routing each request to the better design
would recover much of the gap, yet the request carries no signal to route on; presenting the retrieved
rule more prominently, by contrast, closes most of the rule-application gap without changing what is
stored or retrieved (§4.3; Appendix K). What separates memory designs is therefore not what they
retrieve but what they enable an agent to do.
Contributions
(i)We build a deliberately simple but strong recall-focused memory baseline called BRIDGE:
we argue all complex memory systems should beat it as a bare minimum requirement.
(ii)Yet we show that popular benchmarks focus mostly on recall tasks and hence do not
distinguish memory designs: BRIDGEmatches SOTA systems on them, while on applied-
memory probes designs win in opposite directions (§4.1, §4.2).
(iii) We construct five probes in which every design, including BRIDGE, breaks somewhere:
probes whose requests never name the memory they depend on, graded at generation time (§3).
These are some examples of what current memory evaluations lack.
(iv) Retrieving a memory is not using it: on rule application the right memory is retrieved but
not acted on, and presenting it differently closes most of the gap (§4.3). Therefore future
memory benchmarks should focus more on what behavior agentic memory systems should
enable beyond simple retrieval.
2 BRIDGE: a minimal RAG-shaped memory
BRIDGEis the baseline we measure published memory systems against, so we keep it deliberately
simple: per-block observation notes, retrieved with hybrid BM25 + cosine search and shown to the
reader in chronological order next to the raw source text. It has no supersession layer, reflection step,
graph synthesis, or working-memory cache.
A long-running conversation is a sequence of dated sessions S=⟨S 1, . . . , S T⟩where each St=
(dt, Ut)pairs a session date dtwith the sequence of user/assistant turns Ut. BRIDGEprocesses Sin
2

two phases: aningestionphase that distills each block into an observation note and writes a tuple of
notes, raw text, and embedding to a per-block store; aretrievalphase where BRIDGEpulls top- K
blocks for a query with hybrid retrieval, renders them in chronological order with r=3 raw slots (the
remaining blocks shown only as their distilled notes), and hands the rendered context to a single
reader call.
Ingestion (natural-boundary chunking)Turns accumulate in a token-bounded buffer that flushes a
block Bt,keither at anatural boundaryprovided by the host application or when the buffer reaches the
token cap Ttok=2000 . “Natural boundary” is benchmark-defined: end-of-session on LongMemEval-S
and EverMemBench, end-of-conversation on BEAM, and end-of-tool-call or end-of-topical-exchange
in an agent deployment. Most blocks flush at a natural boundary before reaching the token cap.
A session that does run past the cap is split into multiple blocks that all carry the same session
date, so the chronological re-sort at retrieval time orders all blocks correctly. For each block, an
extractor LLM produces an XML-tagged observation note ot,ksummarizing priorities, preferences,
state changes, and the user’s exact phrasings; the extractor sees the W=10 most-recent prior notes,
so each block is written with global context. We persistbothraw xt,kand distilled ot,k, in contrast to
most compression-shaped systems which persist only o[Mastra AI, 2025]. The note is embedded as
et,k∈Rdwith a frozen sentence encoder (Appendix G). The prompt is adapted from the observer
agent of MASTRA[Mastra AI, 2025]; the full prompt and our modifications are in Appendix B.
RetrievalFor a queryq, we compute two ranked lists:
scos(q, B) =eq·eB
∥eq∥∥eB∥, s bm25(q, B) = BM25 k1,b(q, oB∥xB),
with standard parameters k1=1.5, b=0.75 . BM25 indexes the concatenation of the note and the
source text, so exact phrasings that the note paraphrased away remain reachable. We retain top- 60
from each channel and fuse via Reciprocal Rank Fusion [Cormack et al., 2009] withk rrf=60:
srrf(q, B) =1
krrf+r cos(B)+1
krrf+r bm25(B).
Two-tier rendering with chronological re-orderWe select the topK=10blocks bys rrf. Within
those, the top r=3 byscosfill theraw slotsand are rendered as raw text xt,k; the remaining K−r
are rendered as “ Summary:o t,k” (controlling the reader context length [Levy et al., 2024]). The K
selected blocks are thenre-sorted ascendingly by global position(t, k)before concatenation.
Naive RAG presents the top- Kin relevance order and leaves the reader to work out the order of
events; BRIDGEpresents them in the order they happened [Liu et al., 2023]. This matters where the
answer depends on which event came last: removing the re-sort costs 7.7points on LongMemEval-S
knowledge-update, while no other category is significantly hurt (Appendix N).
The reader is a single LLM call at τ=0 on the rendered context (Appendix M.5 repeats it at τ=0.7 );
its wrapper prompt tells it to honor session dates and apply remembered conventions even when the
request does not mention them (Appendix C.1).
BRIDGEhas three core primitives: (i) per-block extraction with cross-block context chaining, (ii) hy-
brid retrieval over both notes and source text, and (iii) a two-tier render in temporal order. A
leave-one-out, per-category ablation across all three benchmarks (Appendix N; paired bootstrap,
nboot=2,000 ) shows that no single primitive matters on its own: each has at least one ability where
removing it helps and at least one where it hurts. It is thecombinationthat keeps BRIDGEcompetitive
across the mix of abilities these benchmarks test. Raw slots cover exact recall (LongMemEval-S
single-session-assistant drops 30.2 points without them, 95% CI[−42.9,−19.6] ), chronological
order covers temporal and state tracking, hybrid retrieval covers both lexical and semantic queries
(EverMemBench memory-update swings by about 38points between BM25-only and cosine-only),
and the prior-notes chain covers cross-block synthesis.
BRIDGEuses one configuration: the hyperparameters (Appendix A) are fixed across every bench-
mark and every adversarial probe, with no per-task tuning. The configuration was set on the standard
benchmarks, not on the probes: on the probes, showing all ten blocks as raw text instead of three
improves rule application by13points in both of two runs (Appendix L).
3

Table 1: The five probes, each with an example from the benchmark. The request never mentions the
memory it depends on; we grade the work the system produces, not an answer about the history.
Probe In the history The request Passes if
RA+
rule applicationA rule said once in passing, e.g., every
public endpoint must be rate-limitedA task the rule applies to, e.g., add a
password-reset endpointit applies the rule, e.g.,
rate-limits the endpoint
RA−
rule application (ban)A ban said once in passing, e.g.,eval()
is banned after an incidentA task whose obvious solution is
banned, e.g., evaluate math that users
type init avoids the banned approach,
e.g., noeval()
PI-S
pattern induction
(style)Code that shares conventions, never stated,
e.g., every error has a code and a messageNew code of the same kind, e.g., add
an endpoint that cancels an orderit follows the conventions
PI-P
pattern induction
(procedure)The same steps in four past cases, never
stated, e.g., five steps for each incidentThe steps for a new case, e.g., walk
me through a new incidentit follows the procedure
DM
dynamic multi-turnA rule said once in passing, e.g., remind
me to check passport rules whenever
travel comes upA task the rule applies to, e.g., plan a
trip abroad, then four follow-ups from
a simulated userit still applies the rule at the
last turn
3 Adversarial probes for agentic memory
Most memory benchmarks check memory with a question about the past. Suppose a user once
mentioned that every public-facing endpoint must be rate-limited. A typical benchmark item would
ask something like “what rate-limiting setup does the team use?” and score the answer. In practice
the user rarely asks that. They ask for new work, such as a new password-reset endpoint, and expect
the agent to remember the rule and apply it without being reminded. Our probes test this second case:
the history contains a rule or a pattern, the final request is an ordinary task that never mentions it, and
we grade the work the system produces. Like behavioral tests [Ribeiro et al., 2020], the probes cover
three capabilities that the architectures we test claim to support:always-on rule application,pattern
induction, andmulti-turn consistency.
An exampleThe case above is a real scenario from our first probe, RA+(Table 1). Each scenario is
a dated history of sessions followed by one request. Here, the third of ten sessions is a post-incident
review, and partway through it the user remarks that “every public-facing endpoint is supposed to have
rate-limiting middleware (slowapi + redis)”. The other sessions cover other topics. The final request
is “Add a new endpoint POST /api/v1/forgot-password that takes an email address, generates a
reset token, and sends an email.” Nothing in it mentions rate limiting, and the response passes only if
the new endpoint is rate-limited (Appendix I follows this scenario through each system).
Probe taxonomyTable 1 lists the five probes. RA+and RA−plant one rule as a brief aside inside
one of 10sessions whose main topic is something else; in RA−the rule bans an approach, and the
request is one whose obvious solution is the banned approach. PI-S and PI-P never state the memory
at all: a coding convention (PI-S) or a multi-step procedure (PI-P) is only demonstrated, across 4–6
sessions in which the user works through ordinary tasks. DM reuses the RA+corpus but replaces the
single request with a5-turn dialogue driven by a simulated user (τ=0.5, fixed per-turn seed), and is
scored on both the final turn and per-turn regressions.
The probes test what existing benchmarks missLongMemEval-S and EverMemBench are
question-answering benchmarks, as are most of BEAM’s abilities and the recent agent-trajectory
benchmarks AMA-Bench [Zhao et al., 2026] and LongMemEval-V2 [Wu et al., 2026]: the system is
asked about the history and its answer is scored. The closest existing tests are BEAM’s instruction-
and preference-following abilities, which check whether an ordinary request is answered in line with
an earlier instruction or preference [Tavakoli et al., 2026]. Our probes differ in four ways. In RA±
the rule is an aside inside a session about something else, not a standing instruction. RA−scores the
absenceof an approach, which a question about the history cannot test directly. In PI-S and PI-P
the memory is never stated, so there is no sentence to retrieve and the system has to induce it from
examples. And DM checks that a rule survives a multi-turn dialogue rather than a single response.
ConstructionEach scenario is generated once from a spec by a strong frontier generator at τ=0
and stored, so every system and every run sees the same data. The generation prompt requires that
4

an RA rule appear as a brief aside in the flow of conversation rather than as a checklist item or
formal statement, and that PI demonstrations show the pattern only by example. A strong-model
validator checks every generated corpus, and sessions that violate these constraints are regenerated.
The original set has 20scenarios per probe, all with software and infrastructure personas. We add 87
scenarios in three other domains (clinical operations, legal compliance, finance operations) and 80in
two harder difficulty levels for RA+and PI-P, and audit the rule-application corpora for filler sessions
that leak the rule and requests that cue it (Appendix E). The generator’s snapshot is in Appendix G,
and full generation prompts and the spec format are in Appendix D.
Judging at generation timeThe judge scores the generated answer against the scenario’s con-
straints, never against a gold answer: one rule for RA±and DM, four conventions for PI-S, and four
to five steps for PI-P, where paraphrased or merged steps count (Appendix F). Each constraint is
applied,violated, orn/a, and the pass rate isn applied /(n applied +n violated ).
The judge comes from a different vendor than the generator, Oracle, and reader, to avoid same-family
judging bias [Zheng et al., 2023] and self-preference [Panickssery et al., 2024]; §4.2 re-checks its
verdicts with two other graders. The full grading prompt, constraint format, and judge JSON schema
are in Appendix F.
Fair-comparison protocolWe compare BRIDGE(§2) against MEM0 [Chhikara et al., 2025],
LANGMEM[LangChain-AI, 2024], and GRAPHITI[Rasmussen et al., 2025]. Every system uses the
same reader ( τ=0), embedder, and judge (Appendix G). Each baseline gets a ∼100 -word wrapper
prompt that names its native primitive and instructs the reader to apply rules unprompted, and is tuned
only through the user-facing levers its documentation recommends. BRIDGE’s package internals
are unmodified across probes. The strong-model oracle reads the full chronological corpus with no
retrieval bottleneck and serves as a reference in §4.2. Per-system wrapper texts and tuning details are
in Appendix H. Tuning LANGMEM’s schema specifically for pattern induction narrows its gap to
BRIDGEbut does not close it (Appendix L).
4 Main results
SetupOn the standard benchmarks, baseline numbers come from each system’s published report:
we match the reader and use each benchmark’s canonical judge and harness, but prompt depth,
retrieval budget, and chunking remain each system’s own, so where a gap is within CI we claim parity
rather than superiority. The probe protocol is in §3, and all model snapshots, including per-benchmark
judges and the simulated user that drives DM, are in Appendix G.
Reporting protocolWe report BRIDGE’s headline numbers in three ways: (i)reproducibility,
the mean over three independent runs at τ=0 (seeds {42,227,115} ), whose variance comes only
from API non-determinism and which is what we compare with the baselines; (ii)confidence
intervals[Koehn, 2004], 95% bootstrap intervals from N=1000 resamples of the questions in the
seed-42run, drawn as error bars on the Avg column of each figure; and (iii)sampling sensitivity, the
mean±SD over three sampling seeds at τ=0.7 , which is not comparable to the greedy baselines and
is reported in Appendix M.5.
4.1 BRIDGEon standard memory benchmarks
BRIDGEmatches or beats published systems at the same readerOn BEAM 100K withGPT-4.1-
NANO(Figure 1), BRIDGEscores 0.370±0.006 on the canonical 10-ability mean against LIGHT’s
0.345 , a gap at the edge of significance.1On LongMemEval-S with aGPT-4Oreader (Figure 3), it sits
1.4points behind EMERGENCEMEMwith overlapping 95% CIs. On EverMemBench (Figure 2) it
beats MEMOS, the benchmark’s best memory baseline, by 5.7points (GPT-4.1-MINI) and 6.3points
(GEMINI-3-FLASH), both outside the bootstrap CI, and withGEMINI-3-FLASHit trails full context by
7.1points, against 13–20points for the published memory systems [Hu et al., 2026a]. It does so with
1BEAM’s original paper scores Event Ordering with a different, less reliable metric (Kendall tau-b ×F1 over LLM-aligned
event snippets), on which even gold answers score only ∼0.7 . Excluding it gives a 9-ability mean of BRIDGE 0.360±0.007
vs LIGHT 0.375 ,1.5points lower but within bootstrap CI; the swing comes from Event Ordering itself (BRIDGE 0.447 vs
LIGHT0.075).
5

Pref IE IFKUAbst MSR EOSum TR CRAvg0.00.20.40.60.8Score
0.44
0.27
0.42
0.25
0.56
0.20
0.03
0.11
0.10
0.00
0.240.55
0.36
0.35
0.38
0.83
0.15
0.22
0.08
0.16
0.01
0.310.62
0.54
0.40
0.38
0.68
0.36
0.07
0.20
0.16
0.03
0.34
0.380.69
0.47
0.45
0.34
0.60
0.38
0.45
0.23
0.09
0.06Vanilla RAG LIGHT BridgeFigure 1: BEAM 100K [Tavakoli et al., 2026]. All systems useGPT-4.1-NANO; baselines are from
the BEAM paper. Avg is the mean over the ten abilities, with a 95% bootstrap CI for BRIDGE.
BRIDGEbars show one run; the text reports the mean of three runs (0.370).
0255075100Accuracy (%)
97.7
26.5
45.0
96.8
98.4
100.0
67.0
53.3
68.9
72.669.0
10.8
20.7
81.8
87.6
90.7
38.6
62.7
71.4
59.3
65.581.2
9.2
17.3
90.0
92.3
92.2
54.0
56.8
66.8Reader: gemini-3-flashFull-Context (paper oracle) MemOS (paper best memory) Bridge (this work)
F_SH F_MH F_TP MA_C MA_P MA_U P_Style P_Skill P_TitleAvg0255075100Accuracy (%)
83.6
2.4
7.0
63.4
25.1
42.5
39.2
35.5
38.3
37.471.4
18.9
15.7
69.9
52.0
45.2
29.0
32.5
48.5
42.6
48.280.3
13.3
11.0
77.4
57.1
52.2
41.5
36.1
46.4Reader: gpt-4.1-mini
Figure 2: EverMemBench [Hu et al., 2026a] with two readers (top:GEMINI-3-FLASH; bottom:
GPT-4.1-MINI). Baselines are from the benchmark paper:Full-Contextgives the reader the whole
∼1M-token history, and MEMOS is the paper’s best memory system. Avg is the mean over the nine
dimensions, with a95%bootstrap CI for BRIDGE.
∼5–7K reader-prompt tokens per query, 17–154× fewer than full context. Per-ability breakdowns,
results with a stronger reader (which exceed every published number but are not like-for-like), the
reader×extractor decomposition, and token and latency tables are in Appendix M.
Takeaway
On standard memory benchmarks, a minimal RAG memory does as well as complex memory
systems: these benchmarks cannot tell the two apart.
4.2 Results on adversarial probes
We argue that complex agentic memory systems do not clearly outperform BRIDGEon the standard
benchmarks because those benchmarks do not test what memoryenablesan agent to do. Table 2
compares four memory architectures on probes that test this.
LANGMEMwins the rule probes, BRIDGEthe pattern probesBRIDGEwins both pattern-
induction probes (PI-S, PI-P); LANGMEMwins both rule-application probes plus the multi-turn
6

SSA SSU KU TR SSP MS Avg20406080100Accuracy (%)
73
81
62
52
23
37
5280
93
83
62
57
58
7196
97
88
77
70
71
82100
99
83
86
60
81
86100
96
86
77
43
80
82
8598
99
92
84
67
72Naive RAG Zep Supermemory EmergenceMem Oracle BridgeFigure 3: LongMemEval-S [Wu et al., 2025] accuracy (%). All systems use aGPT-4Oreader;
baselines are from each system’s own report.Oraclegives the reader only the sessions with the
evidence; EMERGENCEMEM[Emergence AI, 2025] is the best previously reported system. Avg is
over all500questions, with a95%bootstrap CI for BRIDGE.
Table 2: Pass rate on the five probes ( 20scenarios each), with 95% bootstrap CIs. Each cell is one
run; repeated runs are in Appendix J. Oracle, a strong model given the full history, is a reference
rather than a memory system. Bold: best memory system.
Probe MEM0 LANGMEMGRAPHITI BRIDGEOracle
RA+40% [20,60]80% [60,95] 70% [50,90] 35% [15,55] 90% [75,100]
RA−75% [55,90]90% [75,100] 65% [45,85] 55% [35,75] 95% [85,100]
PI-S46% [31,62] 47% [31,62] 76% [65,86] 81% [72,89] 95% [88,100]
PI-P50% [39,60] 55% [43,66] 49% [35,62] 88% [77,96] 84% [73,94]
DM55% [35,75]75% [55,90] 70% [50,90] 40% [20,60] 95% [85,100]
consistency probe (RA+, RA−, DM); GRAPHITIand MEM0 win none. The margins between the
two kinds of probe are wide: on PI-P, BRIDGEleads the strongest baseline, LANGMEM, by 33points,
and on RA+, LANGMEMleads BRIDGEby 45points, both with non-overlapping CIs. Within each
kind the ordering blurs: the CIs overlap for BRIDGEand GRAPHITIon PI-S and for LANGMEM
and GRAPHITIon RA+. These margins come from our primary grader; under weaker graders they
narrow but keep their direction (see below). Which design wins depends on the kind of memory each
task needs. Rule application isretrieve-and-apply: one explicit rule must be noticed and applied,
and atomic-fact extractors store it as a row of its own. Pattern induction isinduce-from-form: the
target lives in the shape of code (PI-S) or dialogue (PI-P) demonstrated repeatedly, and atomic-fact
extractors have no fact to extract from a pattern that is never asserted. BRIDGE’s block-level notes
describe each session’s work as a whole, so the demonstrated shape survives even without raw text:
removing the raw slots leaves PI-P unchanged (Appendix L).
BRIDGEdrops fewer rules only because it applies fewerOn DM, the per-turn judge also counts
regressions, rules applied and then dropped by the final turn: BRIDGE 6, MEM0 10, GRAPHITI 10,
LANGMEM 11, Oracle 15. BRIDGEhas the fewest because it applies the rule less often to begin with
(final-turn pass rate40%). Table 2 reports the final-turn pass rate, on which LANGMEMleads.
The results hold across domains, graders, readers, and runsDomains: on 87added scenarios
in clinical operations, legal compliance, and finance operations, the BRIDGE −LANGMEMgap has
the same sign as in software in all 12new probe ×domain cells (Appendix E).Graders: two graders
from another vendor agree with ours on 81–86% of verdicts and keep every direction; the margins
narrow, to 35–40points on RA+and4–16points on PI-P (Appendix J).Readers: replacing the reader
with Llama-3.3-70B or Qwen2.5-7B, with the retrieved memory held fixed, keeps the sign in all 8
probe×reader cells with a new reader (Appendix J).Repeated runs: over 3–4runs, BRIDGEtrails
LANGMEMby 36points on RA+and leads it by 31on PI-P, several times the run-to-run spread
(Appendix J). With 20scenarios per probe, one scenario moves a rule probe by 5points, so we do not
interpret differences below about15points.
7

Table 3: Pass rate (%) [ 95% CI] as the probes get harder. Each level reuses the same scenarios; gap is
BRIDGEminus LANGMEM. RA+uses the 18scenarios that pass the leak audit at every level, so its
Level 1 differs slightly from Table 2.
RA+(rule application) PI-P (procedure induction)
Level BRIDGELANGMEMgap BRIDGELANGMEMgap
1 (original)38.9[17, 61]83.3[67, 100]−44.4 87.9[78, 96]54.5[43, 66]+33.3
2 (less signal)33.3[12, 60]61.5[33, 86]−28.2 81.1[69, 92]56.5[40, 72]+24.6
3 (+ competing)43.8[20, 69]42.9[15, 69]+0.9 72.6[58, 86]52.9[36, 69]+19.8
Harder probes erase LANGMEM’s lead and narrow BRIDGE’sWe also make RA+and PI-P
harder in two steps, reusing the original scenarios so that each comparison is paired (Table 3). Level 2
removes signal: more filler sessions for RA+(10→24 ), fewer demonstrations for PI-P ( 4→2 ).
Level 3 adds competing signal: filler sessions on topics close to the rule, and a second procedure
demonstrated next to the target one. BRIDGE’s lead on procedure induction shrinks but holds at every
level. LANGMEM’s lead on rule application shrinks with more filler and disappears at Level 3, which
suggests that a rule stored as its own row helps only while nothing similar competes with it.
Takeaway
On our probes, where the agent has to act on its memory, the designs clearly differ, and no
single memory system wins everywhere: LANGMEMis best at applying rules the user stated,
and BRIDGEat following patterns the user only showed by example.
Each design pays for how it represents the pastThe results reflect two ways a memory can
represent the past: asfactsor asform. Atomic-fact extractors surface a fact at write time but lose
the form it took; block-summary memory keeps the form, but a rule ends up as one phrase among
many in a note. Each design pays for its choice on the other task. The Oracle shows the remaining
headroom: with full access to the history it outperforms every memory system by 5–20points on
RA±, PI-S, and DM, and is matched only on PI-P. BRIDGE, however, already stores both the distilled
notes and the raw text of each block, indexed jointly, so its loss on rule application is not a matter of
what it stores (§4.3).
4.3 BRIDGE’s rule-application gap comes from presentation, not retrieval
Since each design wins different probes, an obvious remedy is to send each request to whichever
design suits it. Doing this perfectly would be worth 13.6 points over the best single system ( 82.8%
vs. LANGMEM’s 69.2% , averaged over the five probes), but a system cannot make this choice from
the request. A classifier that reads only the request and predicts whether it needs a stated rule or a
demonstrated pattern is right 49–61% of the time across six prompt and model variants, no better than
always guessing the majority class ( 60%). This follows from the design of the probes: the requests
never mention the memory they depend on, so they carry nothing to route on (Appendix K).
We therefore looked at where BRIDGEfails on rule application, and it is not retrieval. The block
holding the rule is retrieved in all 20RA+scenarios, and in 12of the 13failures the rule reaches the
reader as raw text, not only as a summary. The reader sees the rule and does not apply it: it is one
remark among many, in one of ten blocks.
We add one call that reads the notes BRIDGEhas already retrieved, pulls out any standing instructions,
and places them in a labeled block at the top of the context; what is stored and retrieved does not
change (Table 4). On the original probe this lifts BRIDGEpast LANGMEMto the level of the Oracle,
and on the hardest level it keeps most of the gain. A control that spends the same extra call on a
generic summary does worse than no change on the original probe, so the gain there does not come
from the extra call; on the hardest level the control recovers much of the gain by itself, so there we
claim only that the targeted version is better. Pattern induction is unaffected (Appendix K). We read
this as a diagnostic rather than a method.
8

Table 4: BRIDGEon RA+, changing only how the retrieved memory is shown to the reader. In each
column, all BRIDGErows use the same retrieved memory, reader, and grader. This is a separate run,
and RA+varies from run to run, so the unchanged row differs from Table 2 ( 52.6 vs.35). Reference
rows come from Tables 2 and 3.
Original probe Hardest level
BRIDGE, unchanged52.6 42.1
+ generic summary (same extra call)40.0 66.7
+ standing rules shown at the top94.7 78.9
Reference:LANGMEM80.0 42.9
Reference:Oracle (strong model, full history)90.0—
Takeaway
Memory is not only about retrieval, and other factors might be what make a specific design
stand out. For example, changing only how retrieved memory is presented to the agent, with
storage and retrieval unchanged, closes most of BRIDGE’s gap on rule application.
5 Related work
Agentic memory systemsAtomic-fact extractors(MEM0 [Chhikara et al., 2025], LANG-
MEM[LangChain-AI, 2024], A-MEM[Xu et al., 2025], MEMORYBANK[Zhong et al., 2023])
distill each interaction into key-value or schema-typed memories, with ADD/UPDATE/DELETE -style
refinement to handle contradictions.Temporal knowledge graphs(ZEP/GRAPHITI[Rasmussen et al.,
2025] and descendants) model entities and relations with bi-temporal validity intervals; related graph-
structured memories retrieve via personalized PageRank over an entity graph (HIPPORAG [Gutiérrez
et al., 2025]) or community-level summaries (GRAPHRAG [Edge et al., 2025]).OS-style memory hi-
erarchies(MEMGPT/LETTA[Packer et al., 2024], MEMOS [Li et al., 2025]) page between in-context
and external memory. Recent hierarchical systems organize memory at several granularities;XMEM-
ORY[Hu et al., 2026c] argues that standard RAG fits a conversation stream poorly because of its many
near-duplicate spans, and retrieves top-down over a hierarchy of themes, episodes, and raw messages.
Verbal self-reflection over past episodes [Shinn et al., 2023] and growing skill libraries [Wang et al.,
2023] are complementary primitives for cross-episode learning, and EMERGENCEMEM[Emergence
AI, 2025] sets the prior reported SOTA on LongMemEval-S. Surveys organize this space by the form,
function, and dynamics of memory [Hu et al., 2026b] or by memory structure [Jiang et al., 2026].
Comparing memory designsZeng et al. [2024] compare chunks, knowledge triples, atomic facts,
summaries, and their mixture inside one retrieval harness on question-answering datasets, and find
that different structures suit different tasks and that mixed memory is robust to noise. Our results
point the same way. We differ in what we measure: we compare shipped systems as deployed, and
ask whether the standard evaluations can see the difference. On recall benchmarks a minimal memory
matches complex ones (§4.1); on probes that require applying memory, the designs separate (§4.2).
Long context, RAG, and progressive summarizationA complementary line treats memory as
long-context: incrementally summarize prior context so the working window stays bounded (Packer
et al., 2024). Long-context readers, however, exhibit U-shaped positional recall (thelost-in-the-middle
effect [Liu et al., 2023]), which motivates systems that pre-select and order a small evidence set
rather than feeding the entire window. BRIDGE’s read-time pipeline is naive RAG [Lewis et al.,
2021] with three additions over this baseline: hybrid lexical+semantic retrieval via reciprocal rank
fusion [Robertson and Zaragoza, 2009, Cormack et al., 2009], where the dense channel follows dual-
encoder retrieval [Karpukhin et al., 2020] over a frozen sentence encoder [Reimers and Gurevych,
2019] and the lexical channel reflects BM25’s continued strength as an out-of-domain zero-shot
retriever [Thakur et al., 2021]; a two-tier render mixing raw evidence with distilled summaries; and a
chronological re-sort of the selected blocks. Each is a small, well-studied IR primitive, and together
they already match published memory systems on the standard benchmarks (§4.1).
9

Memory benchmarksLongMemEval-S [Wu et al., 2025], EverMemBench [Hu et al., 2026a], and
BEAM [Tavakoli et al., 2026] are the standard benchmarks for agentic memory, and LoCoMo [Ma-
harana et al., 2024] is an earlier long-conversation benchmark with less headroom. Like the recent
agent-setting benchmarks AMA-Bench [Zhao et al., 2026] and LongMemEval-V2 [Wu et al., 2026],
they mostly ask questions about the history; §3 compares them with our probes, including BEAM’s
instruction- and preference-following abilities, which come closest. Beyond the Needle’s Illusion [Lin
et al., 2026] shows that near-miss documents, more than context length, limit long-context memory;
our hardest difficulty level borrows this idea (§4.2). Anatomy of Agentic Memory [Jiang et al., 2026]
reaches a related diagnosis from a survey of systems: memory benchmarks are saturated, their metrics
track usefulness poorly, and results swing with the backbone model. Our probes respond to that
diagnosis by scoring whether memory is used rather than recalled, and they separate designs that the
standard benchmarks do not.
6 Discussion, broader impact, and limitations
Memory design is more than storage and retrievalMuch of the complexity added to agentic
memory changes what is stored and how it is retrieved. On rule application, the gap we measure
comes from neither: the rule is retrieved, and changing only how it is presented closes most of the
gap (§4.3). Since BRIDGEalready keeps a fact-shaped and a form-shaped view of each block, routing
between the two on what has been retrieved, rather than on the request, is left to future work.
Limitations(i) Scale.The original set has 20scenarios per probe, so one scenario moves a rule
probe by 5points, and we do not interpret differences below about 15points; the added domain cells
hold4–8scenarios each.(ii) Synthetic data.Scenarios are written by a model. The audit catches
scenarios that give the rule away, but it cannot guarantee the care that hand-written scenarios get, and
human grading was limited to the authors’ spot check of a sample of graded responses.(iii) What the
grader scores.The grader checks whether a remembered constraint was honored, not whether the
answer is otherwise good.(iv) Coverage.DM was run once and is not part of the robustness checks,
and the Oracle and the scenario generator come from the same model family.(v) Static tests.The
probes are fixed input–output tests, not live agents with tools, long deployments, and shifting users.
Broader impactThe benchmark is open and inspectable, and BRIDGEuses far fewer reader tokens
than full context (§4.1). Memory that applies remembered rules without being asked will also apply
stale or mistaken ones, so systems built this way should let users see and remove what is remembered.
7 Conclusion
Agentic memory should be evaluated by what it lets an agent do, not by what it retrieves. On three
standard benchmarks, BRIDGE, a deliberately minimal RAG memory, matches published memory
systems at matched reader. This does not make BRIDGEa better memory system; it shows that
benchmarks built on recall cannot tell memory designs apart. Once the agent has to act on its memory,
the designs clearly differ. Atomic-fact memory is best at applying stated rules, block-summary
memory at following demonstrated patterns, and no design wins everywhere. Retrieval alone does
not decide the outcome: BRIDGEretrieves the rules it fails to apply, and presenting them more clearly
closes most of the gap. Better agentic memory therefore needs evaluations that test how memory is
used and designs that look beyond storage and retrieval. We hope BRIDGEserves as the minimal
baseline that such systems should clear.
Acknowledgments and Disclosure of Funding
This work was supported by research funding from Ce Zhang’s lab at the University of Chicago. The
authors declare no competing interests.
10

References
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building
production-ready ai agents with scalable long-term memory, 2025. URL https://arxiv.org/
abs/2504.19413.
Gordon V . Cormack, Charles L A Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InProceedings of the 32nd International ACM SI-
GIR Conference on Research and Development in Information Retrieval, SIGIR ’09, page 758–759,
New York, NY , USA, 2009. Association for Computing Machinery. ISBN 9781605584836. doi:
10.1145/1571941.1572114. URLhttps://doi.org/10.1145/1571941.1572114.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph rag approach to query-focused summarization, 2025. URL https://arxiv.org/abs/
2404.16130.
Emergence AI. Sota on longmemeval with rag, 2025. URL https://www.emergence.ai/blog/
sota-on-longmemeval-with-rag.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag:
Neurobiologically inspired long-term memory for large language models, 2025. URL https:
//arxiv.org/abs/2405.14831.
Kilem Li Gwet. Computing inter-rater reliability and its variance in the presence of high agreement.
British Journal of Mathematical and Statistical Psychology, 61(1):29–48, 2008.
Chuanrui Hu, Tong Li, Xingze Gao, Hongda Chen, Yi Bai, Dannong Xu, Tianwei Lin, Xiaohong
Li, Yunyun Han, Jian Pei, and Yafeng Deng. Evaluating long-horizon memory for multi-party
collaborative dialogues, 2026a. URLhttps://arxiv.org/abs/2602.01313.
Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin
Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang,
Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo,
Xinlei Yu, Zhenhong Zhou, Zewen Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei
Yin, Xiaobin Hu, Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei
Cheng, Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong Wen,
Xuanjing Huang, Yu-Gang Jiang, and Shuicheng Yan. Memory in the age of ai agents, 2026b.
URLhttps://arxiv.org/abs/2512.13564.
Zhanghao Hu, Qinglin Zhu, Runcong Zhao, Di Liang, Hanqi Yan, Yulan He, and Lin Gui. Beyond
rag for agent memory: Retrieval by decoupling and aggregation, 2026c. URL https://arxiv.
org/abs/2602.02007.
Dongming Jiang, Yi Li, Songtao Wei, Jinxin Yang, Ayushi Kishore, Alysa Zhao, Dingyi Kang,
Xu Hu, Feng Chen, Qiannan Li, and Bingzhe Li. Anatomy of agentic memory: Taxonomy and
empirical analysis of evaluation and system limitations, 2026. URLhttps://arxiv.org/abs/
2602.19320.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering, 2020. URL
https://arxiv.org/abs/2004.04906.
Philipp Koehn. Statistical significance tests for machine translation evaluation. In Dekang Lin
and Dekai Wu, editors,Proceedings of the 2004 Conference on Empirical Methods in Natural
Language Processing, pages 388–395, Barcelona, Spain, July 2004. Association for Computational
Linguistics. URLhttps://aclanthology.org/W04-3250/.
LangChain-AI. Langmem: A library for long-term memory in llm agents, 2024. URL https:
//github.com/langchain-ai/langmem.
Mosh Levy, Alon Jacoby, and Yoav Goldberg. Same task, more tokens: the impact of input length
on the reasoning performance of large language models, 2024. URL https://arxiv.org/abs/
2402.14848.
11

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https:
//arxiv.org/abs/2005.11401.
Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi
Lai, Jihao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao Lin, Jiahao Huo, Tianyi Chen, Kai Chen,
Kehang Li, Zhiqiang Yin, Qingchen Yu, Bo Tang, Hongkang Yang, Zhi-Qin John Xu, and Feiyu
Xiong. Memos: An operating system for memory-augmented generation (mag) in large language
models, 2025. URLhttps://arxiv.org/abs/2505.22101.
Tianwei Lin, Zuyi Zhou, Xinda Zhao, Chenke Wang, Xiaohong Li, Yu Chen, Chuanrui Hu, Jian Pei,
and Yafeng Deng. Beyond the needle’s illusion: Decoupled evaluation of evidence access and use
under semantic interference at 326m-token scale, 2026. URL https://arxiv.org/abs/2601.
20276.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts, 2023. URL
https://arxiv.org/abs/2307.03172.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of llm agents, 2024. URL https:
//arxiv.org/abs/2402.17753.
Mastra AI. A human-inspired memory system that scores ∼95% on LongMemEval —
with a completely stable context window, 2025. URL https://mastra.ai/blog/
observational-memory. Mastra Research Blog.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. Memgpt: Towards llms as operating systems, 2024. URL https://arxiv.org/abs/
2310.08560.
Arjun Panickssery, Samuel R. Bowman, and Shi Feng. Llm evaluators recognize and favor their own
generations, 2024. URLhttps://arxiv.org/abs/2404.13076.
Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and
Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior, 2023. URL
https://arxiv.org/abs/2304.03442.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
temporal knowledge graph architecture for agent memory, 2025. URL https://arxiv.org/
abs/2501.13956.
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks,
2019. URLhttps://arxiv.org/abs/1908.10084.
Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behav-
ioral testing of NLP models with CheckList. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel
Tetreault, editors,Proceedings of the 58th Annual Meeting of the Association for Computational
Linguistics, pages 4902–4912, Online, July 2020. Association for Computational Linguistics. doi:
10.18653/v1/2020.acl-main.442. URLhttps://aclanthology.org/2020.acl-main.442/.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond.
Found. Trends Inf. Retr., 3(4):333–389, April 2009. ISSN 1554-0669. doi: 10.1561/1500000019.
URLhttps://doi.org/10.1561/1500000019.
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and
Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL
https://arxiv.org/abs/2303.11366.
Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, and J Ross
Mitchell. Beyond a million tokens: Benchmarking and enhancing long-term memory in llms, 2026.
URLhttps://arxiv.org/abs/2510.27246.
12

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A
heterogenous benchmark for zero-shot evaluation of information retrieval models, 2021. URL
https://arxiv.org/abs/2104.08663.
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and
Anima Anandkumar. V oyager: An open-ended embodied agent with large language models, 2023.
URLhttps://arxiv.org/abs/2305.16291.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval:
Benchmarking chat assistants on long-term interactive memory, 2025. URL https://arxiv.
org/abs/2410.10813.
Di Wu, Zixiang Ji, Asmi Kawatkar, Bryan Kwan, Jia-Chen Gu, Nanyun Peng, and Kai-Wei Chang.
LongMemEval-V2: Evaluating long-term agent memory toward experienced colleagues, 2026.
URLhttps://arxiv.org/abs/2605.12493.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic
memory for llm agents, 2025. URLhttps://arxiv.org/abs/2502.12110.
Ruihong Zeng, Jinyuan Fang, Siwei Liu, and Zaiqiao Meng. On the structural memory of llm agents,
2024. URLhttps://arxiv.org/abs/2412.15266.
Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and
Ji-Rong Wen. A survey on the memory mechanism of large language model based agents, 2024.
URLhttps://arxiv.org/abs/2404.13501.
Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang
Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, and Jishen Zhao.
AMA-Bench: Evaluating long-horizon memory for agentic applications. InProceedings of the
43rd International Conference on Machine Learning, volume 306 ofProceedings of Machine
Learning Research, 2026. URLhttps://arxiv.org/abs/2602.22769.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https://arxiv.org/abs/
2306.05685.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large
language models with long-term memory, 2023. URLhttps://arxiv.org/abs/2305.10250.
13

A BRIDGEhyperparameters
Table 5 lists BRIDGE’s hyperparameters (§2).
Table 5: BRIDGEhyperparameters, the same for every result in the paper.
Parameter Value Parameter Value
Chunk thresholdT tok 2000RRFk rrf 60
Prior-notes windowW10BM25(k 1, b) (1.5,0.75)
Top-Kblocks10Embedder dimd3072
Raw slotsr3Extractor / readerτ0.3/0
Per-channel pre-fusionK each 60Sampling-sensitivityτ0.7
B BRIDGEextractor prompt
The system prompt for the BRIDGEextractor is reproduced below in full. Adapted from MASTRA’s
observer-agent.ts (Apache-2.0 license); the modifications relative to the original are documented
after the listing.
Bridge extractor — system prompt
You are the memory consciousness of an AI assistant. Your observations will be the ONLY information
the assistant has about past interactions with this user.
Extract observations that will help the assistant remember:
Critical: distinguish user assertions from questions.When the user TELLS you something about
themselves, mark it as an assertion (“I have two kids” → • (14:30) User stated has two kids). When the
user ASKS about something, mark it as a question/request. User assertions are authoritative — the user
is the source of truth about their own life.
State changes and updates.When a user indicates they are changing something, frame it as a state
change (“I’m switching from A to B”→“User is switching from A to B”, replacing the old approach).
Temporal anchoring.Each observation has two potential timestamps: the BEGINNING (utterance time,
always include) and the END (referenced time, include only when there is a relative time reference like
“last week” or “next month”). Format:(TIME) [observation]. (meaning DATE).
Preserve unusual phrasing.When the user uses unexpected or non-standard terminology, quote their
exact words.
Preserving details in assistant-generated content.When the assistant provides lists, recommendations,
or technical content, preserve the DISTINGUISHING DETAILS: names, identifiers, numerical values,
role attributions. Do not summarise away the specifics.
Completion tracking.Use ✓when the user explicitly confirms something worked or a multi-step task
reached its stated goal.
Output format.XML with •(high), •(medium), •(low),✓(resolved) priority emojis. Group
observations by date.
Modifications relative to MASTRAobserver-agentThe original observer-agent prompt provides
the XML output format and the priority-emoji scheme. BRIDGEadds the assertion-vs-question dis-
tinction, the state-change framing, two-timestamp temporal anchoring, unusual-phrasing preservation,
distinguishing-detail preservation rules, and ✓-completion tracking. All five additions target the
supersession and rule-application behaviours we report; the unmodified observer-agent prompt does
not separately track state changes or temporal anchoring.
C Reader prompts
Each memory system is paired with a ∼100 -word wrapper prompt that names the system’s native
primitive (e.g., observation notes for BRIDGE, atomic facts for MEM0, typed memory schema for
LANGMEM, temporal knowledge graph for GRAPHITI) and instructs the reader to apply remembered
rules unprompted. All four wrappers are calibrated to the same depth tier so no system has either
a substantially shorter or longer prompt than the others; the reader user-prompt template (§C.5) is
shared across systems.
14

C.1 BRIDGEwrapper
Bridge — reader system prompt
You are an assistant with access to memory of prior conversations stored as session-level observation
notes (block summaries). Each block summarises one session’s key observations — the extractor
was instructed to capture user assertions, state changes, preferences, project-specific details, technical
constraints, and unusual phrasing verbatim, with priority tags ( •high,•medium, •low,✓resolved).
Blocks are presented in chronological order, with each block having seen prior blocks during extraction,
so recurring patterns and project-wide rules tend to surface across blocks. Use the blocks to inform your
response. Pay attention to dates and to whether earlier statements have been superseded by later ones
(the most recent state is canonical). Apply any rules, preferences, conventions, project constraints, or
reminders the blocks describe, even if the user’s current request doesn’t explicitly cue them.
C.2 MEM0 wrapper
Mem0 — reader system prompt
You are an assistant with access to memories curated by Mem0 — atomic facts about the user that have
been deduplicated through ADD/UPDATE/DELETE refinement. Each memory captures a single preference,
fact, or constraint extracted from prior conversations. Use the memories to inform your response and
apply any rules or preferences unprompted, even if the current request doesn’t cue them.
C.3 LANGMEMwrapper
LangMem — reader system prompt
You are an assistant with access to memories stored under a typed schema NoteworthyMemory(kind,
content, evidence) , where kind labels the memory type (preference, convention, project constraint,
reminder, decision, or pattern) and evidence cites the supporting context. Memories are consolidated
across sessions; later contradicting memories replace earlier ones. Use the typed memories to inform your
response, particularly for applying conventions or rules inferred from repeated demonstration. Apply any
rules unprompted.
C.4 GRAPHITIwrapper
Graphiti — reader system prompt
You are an assistant with access to a temporal knowledge graph (Graphiti) containing edges and nodes with
valid_at andinvalid_at timestamps. Entity types include Preference, ProjectConstraint, Convention,
and Reminder. The graph captures the history of state changes; later edges supersede earlier ones for the
same entity. Use the graph context — including edges, nodes, and source episodes — to inform your
response, paying attention to the temporal validity intervals. Apply any rules or constraints unprompted.
C.5 Reader user-prompt template
Reader user-prompt template
{retrieved_context}
---
User request: {probe}
Answer:
15

D Adversarial probe construction details
D.1 Spec format examples
RA+(positive rule)A single rule, established once across 10sessions and embedded as a casual
aside in one rule-embed session; the test probe triggers the rule’s domain without lexically referencing
it. The other nine sessions are unrelated work (distractors).
RA+scenario spec (illustrative)
{
"id": "pci_charge_endpoint",
"persona": "Daniel, backend engineer at a small fintech",
"domain": "fintech_backend",
"project_topic": "the customer billing platform",
"rule": {
"rule_text": "Whenever you propose code that handles credit-card numbers or calls
a payment processor, surface a reminder about PCI-DSS scope and tokenization.
We've been audited and need to flag this every time.",
"embed_session_idx": 2,
"embed_topic": "discussing the team's PCI compliance posture during a
stripe-webhook design"
},
"distractor_topics": ["debugging a flaky test in the analytics service", "Q3 hiring
plan review"],
"test_probe": "Add a /charge endpoint that accepts a card number and amount and
creates a Stripe charge. Show me the full implementation including error
handling and logging."
}
RA−(negative anti-pattern)Same structural shape as RA+, but the embedded rule is a ban; the
test probe is one whose naive solutionisthe banned construct.
RA−scenario spec (illustrative)
{
"id": "no_eval_injection",
"persona": "Daniel, backend engineer in a security-conscious team",
"rule": {
"rule_text": "We had an incident two months ago: someone used eval() in a
config-loader to handle dynamic expressions, an attacker fed in a payload,
and we got RCE on prod. We banned eval() and exec() across the codebase. If
you need dynamic-expression evaluation, use ast.literal_eval for literals or
a proper parser --- never eval.",
"embed_session_idx": 2,
"embed_topic": "post-incident review of the eval()-RCE outage"
},
"distractor_topics": ["q3 hiring plan", "datadog dashboard cleanup"],
"test_probe": "I want users to enter arbitrary math expressions like'2*sin(x)+3'
into a form, and the server should compute them. Write a Python function that
takes the expression string and returns the result."
}
PI-S (style induction, abbreviated)A list of Ccoding conventions that are demonstrated by the
code in each session but never stated as rules. The conventions below illustrate the spec format;
per-language and per-domain variants are in the released benchmark.
16

PI-S scenario spec (illustrative)
{
"id": "rest_api_conventions",
"persona": "Daniel, building a Python REST API",
"conventions": [
{"id": "C1", "rule": "Errors raised as HTTPException(status_code=4xx,
detail={'code':'<short_id>','message':'<text>'}) --- never bare-string
detail."},
{"id": "C2", "rule": "Endpoints read a request_id from the X-Request-ID header and
bind it to the structured logger at the top of the function."},
{"id": "C3", "rule": "Database access goes through a request-scoped session
dependency, not a module-level global."},
{"id": "C4", "rule": "Type hints use the typing-module forms (Optional[X],
List[X], Dict[K,V]), not the builtins-pep604 forms (X | None, list[X])."}
],
"demo_session_topics": [
"GET /users/{id}", "PATCH /users/{id}", "POST /products",
"GET /products with cursor pagination", "POST /orders",
"GET /orders/{id} and state transitions"
],
"test_probe": "Add a new endpoint POST /api/v1/orders/{order_id}/cancel for the
orders service."
}
PI-P (procedure induction)An ordered sequence of Kprocedural steps demonstrated end-to-end
across N=4 past-incident sessions. The test probe presents a new instance and asks for the procedure
as a numbered list.
PI-P scenario spec (illustrative)
{
"id": "incident_response_5step",
"persona": "Daniel, on the on-call rotation",
"n_demonstrations": 4,
"demonstration_topics": [
"checkout API P1 --- error rate spike after a deploy",
"analytics dashboard P1 --- replica unreachable after kernel patch",
"auth service P1 --- token validation latency spike",
"billing webhook P1 --- duplicate charges after upstream change"
],
"steps": [
"Pull recent error logs for the affected service (Datadog/Loki, last 30 min).",
"Extract a correlation or trace id from the errors and follow it to the failing
span.",
"Check downstream dependency health (status pages, internal dashboards).",
"If the issue correlates with a recent deploy or config change, roll back as the
immediate remediation.",
"Page the responsible team's on-call AND file a post-mortem ticket with timeline
and root-cause-TBD."
],
"test_probe": "We just got a P1 page: error rate on the checkout API has spiked from
baseline 0.15% to ~6%. Walk me through the steps I should take to handle this.
Numbered list."
}
DM (dynamic multi-turn)DM reuses the RA+scenario corpus directly — no separate spec format
— and replaces the single-shot test probe with a 5-turn dialogue driven by the simulated user-agent
described in §3.
Generation procedure (RA+)Sessions are dated chronologically ( base_date + 4 days per
session). Session 0 is a generic “project kickoff” that establishes persona and topic without mentioning
the rule. Session embed_session_idx (default 2) embeds the rule in the user’s middle turn during a
17

topical conversation; the generation prompt is explicit that the rule must read like “a casual aside or a
side-comment in the flow of work, NOT a checklist item or a rule announcement”. The remaining 8
sessions are unrelated work; the prompt explicitly prohibits any mention of rules from prior sessions.
Generation procedure (RA−)Identical structural pipeline to RA+(10 sessions: 1 intro + 1 rule-
embed + 8 distractors), but the embedded rule_text is phrased as a ban or anti-pattern (“we banned
eval(). . . ”), typically motivated by a referenced past incident. The test probe is engineered so that
the naive solutionisthe banned construct, ensuring the system has to actively avoid the anti-pattern
rather than merely happen to not use it. Distractor and audit steps are unchanged.
Generation procedure (PI-S, PI-P)Demonstration sessions show the user working on different
topics; the generation prompt requires that the code (PI-S) or dialogue (PI-P) in each sessionexhibit
the pattern through repeated use without the user ever stating it as a rule. We post-hoc audit each
generated corpus with a strong-model validator: scenarios where any session mentions the pattern by
name are regenerated with a bumped seed.
Generation procedure (DM)DM does not generate new scenario data: the 20 RA+scenarios are
reused unchanged. The added generation step is the simulated user-agent that drives turns 2–5. For
each (scenario, system, turn) triple, the user-agent is gpt-4.1-mini atτ=0.5 with a deterministic
per-turn seed σuser(id, k) = SHA256(id∥k)[:32] , prompted to read the latest assistant response
and produce one natural follow-up that an engineer would actually send next (1–3 sentences, “no
enumerate every concern — pick ONE direction to push on”, explicitly forbidden from revealing
hidden information about the embedded rule). The same simulated user is used for all four memory
systems on a given scenario and turn, so the user-agent’s outputs are identical across systems and
only the assistant’s response varies.
D.2 Constraint counts
Table 6: Scenarios, sessions, and constraints per probe. “Total cells” counts the verdicts in one run
over all five systems.
Probe Scenarios Sessions Constraints/scenario Total cells(×5 systems)
RA+20 10 (1 intro + 1 embed + 8 distractor) 1 (rule) 100
RA−20 10 (1 intro + 1 embed + 8 distractor) 1 (anti-pattern) 100
PI-S 20 7 (1 intro + 6 demonstration) 4 (conventions) 400
PI-P 20 5 (1 intro + 4 demonstration) 4–5 (procedural steps) 440
DM 20 10 (reuses RA+corpus) 1 (rule,×5 turns) 100 final + 500 per-turn
E Probe expansion and corpus audit
The original probe set has 20scenarios per probe, all with software and infrastructure personas.
We add 167scenarios: 87in three other domains and 80in two harder difficulty levels (Table 7).
Added scenarios are run with BRIDGEand LANGMEMunder the configuration of Table 2. DM is not
expanded, since every new condition needs a live multi-turn dialogue.
Table 7: Scenarios per probe after expansion.
Probe Original Levels 2–3 New domains Total
RA+20 40 21 81
RA−20—18 38
PI-S20—24 44
PI-P20 40 24 84
Total80 80 87 247
DomainsScenario specs for hospital clinical operations, in-house legal and compliance work, and
corporate finance operations are written by a strong model, few-shot prompted with the original specs.
Specs and generated corpora are audited (below), and sessions are generated with the pipeline of
18

Appendix D. Table 8 gives the BRIDGE −LANGMEMgap per cell; every cell has the same sign as in
software. Non-software cells hold 4–8scenarios, so one scenario moves a rule-probe cell by 12–25
points, and we read the table for its signs rather than its sizes.
Table 8: BRIDGEminus LANGMEMpass rate (points) by domain, with scenarios per cell in parenthe-
ses. Negative means LANGMEMis ahead.
Probe Software Clinical ops Legal / compliance Finance ops
RA+−45.0(20)−21.4(7)−40.0(7)−16.7(7)
RA−−35.0(20)−16.7(4)−6.7(7)−14.3(7)
PI-S+34.6(20)+57.1(8)+40.1(8)+23.3(8)
PI-P+33.3(20)+52.5(8)+46.7(8)+29.3(8)
Difficulty levelsLevel 2 removes signal: RA+histories grow from 10to24sessions, the extra
sessions on unrelated topics, and PI-P keeps 2of its 4demonstrations. Level 3 adds competing signal:
the extra RA+sessions cover topics close to the rule, and PI-P demonstrates a second, unrelated
procedure next to the target one. Each level reuses the original scenarios and requests, so comparisons
across levels are paired (Table 3). As the history grows and fills with similar material, BRIDGE
retrieves the rule-bearing block less often: in20,19, and17of20scenarios at Levels 1–3.
Corpus auditA generated scenario can be too easy in two ways: a filler session states or implies
the rule, or the final request names what the rule is about. An auditor model (GPT-4.1-MINI) checks
every non-rule session and every request of the rule-application corpora for both (Table 9). For the
added domains, flagged specs are filtered and scenarios regenerated before any system is run; the
original corpora are reported as shipped. Levels 2–3 reuse the original requests, so the request flagged
there is the original one. The two RA+scenarios with a leaking session at any level are left out of
the paired difficulty comparison. The pattern-induction corpora are not audited this way, since their
memory is never stated.
Table 9: Corpus audit of the rule-application corpora.
Corpus Scenarios Sessions checked Leaking sessions Requests naming the rule
RA+, original20 180 2 1
RA−, original20 180 1 1
RA+, Level 220 460 2 1(original request)
RA+, Level 320 460 2 1(original request)
RA+, three new domains21 189 0 0
RA−, three new domains18 162 0 0
F Judge constraint format
The judge is given the probe, the generated answer, and the constraint set; it returns a JSON object
with one verdict per constraint. It receives a single message, with no system prompt, at temperature 0:
19

Judge prompt
You are a strict grader. Read the user's request and the model's response. For each
constraint in the list, decide whether the model's response correctly applied the
constraint.
User's request: {probe}
Constraints (each is something the model SHOULD have applied/honored when responding,
even though the user did NOT mention the constraint in this request):
{constraints_block}
Model's response:
{answer}
For each constraint, decide:
- "applied": the model's response demonstrates the constraint was honored
(unprompted mention, follow-the-rule behavior, refusal of forbidden
alternative).
- "violated": the response contradicts the constraint, OR fails to surface a
required reminder/element.
- "n/a": the response is on a topic where the constraint genuinely doesn't apply.
Reply with a JSON object only:
{
"judgements": [
{"id": "<constraint_id>", "verdict": "applied|violated|n/a", "reason": "<one
sentence>"},
...
]
}
The constraint text format per probe type:
•RA+:The response should unprompted apply this rule established in an earlier session:
<rule_text>.
•RA−:The response should AVOID the anti-pattern established in an earlier session:
<rule_text> . The response should not propose, suggest, or use the forbidden approach,
and ideally explains why or proposes a safer alternative.
•PI-S:The response should follow this coding convention demonstrated repeatedly in earlier
sessions:<convention.rule>.(one constraint per convention)
•PI-P:The response should describe the substance of this step somewhere in the proposed
procedure (paraphrasing OK, merging with an adjacent step OK, splitting into two OK —
only the substance + relative ordering vs the other steps matters). Substance: <step_text> .
(one constraint per step)
•DM:same as RA+, judged twice — once over the concatenation of all assistant turns
(final-state) and once per-assistant-turn (consistency / regression detection).
The judge JSON schema:
Judge response schema
{
"judgements": [
{
"id": "<constraint_id>",
"verdict": "applied | violated | n/a",
"reason": "<one-sentence reason>"
}
]
}
20

Pass-rate definitionPer-cell pass rate = napplied/(n applied +n violated );n/averdicts are excluded.
Per-probe pass rate = pooled across scenarios. We treat empty judgements as missing data and re-
judge them with progressively larger token budgets ( 8K→16Kmax_tokens ); after this hardening
pass the corpus has no empty cells.
G Pinned model snapshots
Table 10: Model versions. On the probes, the grader comes from a different vendor than the generator,
reader, and Oracle. The standard benchmarks use their own judges (last two rows).
Role Provider Snapshot
Reader (BRIDGE+ all baselines, canonical) OpenAIgpt-4.1-mini-2025-04-14
Embedder (BRIDGE) OpenAItext-embedding-3-large(3072-d)
Strong-model oracle (adversarial) OpenAIgpt-5-2025-08-07 reasoning_effort=high
Generator (adversarial scenarios) OpenAIgpt-5-2025-08-07 reasoning_effort=minimal
Judge (primary, adversarial) Google AI Studiogemini-3.1-pro-preview thinking_budget=-1
Judge (fallback 1) Vertex AIgemini-3.1-pro-preview(same model, different routing)
Judge (fallback 2) OpenAIgpt-4.1-mini-2025-04-14
User-agent (DM probe) OpenAIgpt-4.1-miniτ=0.5with deterministic per-turn seed
Judge (LongMemEval-S, EverMemBench) OpenAIgpt-4o-2024-08-06(paper-canonical)
Judge (BEAM) OpenAIgpt-4.1-mini-2025-04-14(paper-canonical)
H Per-system wrapper and tuning details
This appendix expands the configuration choices summarised under the fair-comparison protocol
(§3). Each system receives a wrapper at ∼100 -word depth; the per-system tuning levers below are
the user-facing knobs documented by each system’s authors.
BRIDGEConfiguration as in Table 5 (§2). The wrapper text is in §C.1. No per-probe tuning.
MEM0 custom_instructions explicitly enumerate the categories the extractor should track
(preferences, conventions, project constraints, reminders, decisions, patterns). top_k = 20 for re-
trieval. collection isolated per scenario via per-process MEM0_DIR to avoid Qdrant lock contention
under multi-process evaluation. Wrapper text in §C.2.
LANGMEMCustom Pydantic schema: NoteworthyMemory(kind: Literal["preference",
"convention", "project_constraint", "reminder", "decision", "pattern"],
content: str, evidence: str) .instructions= string emphasises pattern abstraction
(“capture recurring shapes even when not explicitly stated as rules”). enable_deletes=True so
contradictions remove rather than accumulate. query_limit=20 . Schema-aware renderer formats
retrieved memories as[KIND] content (evidence). Wrapper text in §C.3.
GRAPHITISearch uses COMBINED_HYBRID_SEARCH_CROSS_ENCODER , which returns
edges, nodes, and episodes (the default returns edges only). Custom entity_types
register Preference, ProjectConstraint, Convention, and Reminder as first-class entities.
custom_extraction_instructions override the default “NEVER extract abstract con-
cepts” guidance to allow conventions and patterns to be extracted as entities. Wrapper text in
§C.4.
Strong-model oracle (ceiling, not a competing system)Reads the entire chronological session
corpus directly (no retrieval), formatted by the same date+role+content scheme used in BRIDGE’s
block construction. Reader temperature 0but the call uses reasoning_effort=high which permits
internal CoT. The oracle’s system prompt is the canonical 3-pass scan/apply/self-check template:
“First, read the entire conversation history below. Second, identify any rules, preferences, conventions,
or constraints that apply to the user’s current request. Third, generate a response that honours all
identified constraints. Self-check before answering.”
21

I Failure case study: BRIDGEon RA+
BRIDGEloses to LANGMEMon RA+by45points ( 35% vs80%, Table 2). This appendix follows
one scenario, the example of §3, through both systems. The full 20-scenario record set is in
adversarial/results/ra_positive__records.jsonl.
Scenario rate_limit_public_endpoints (Marcus, backend engineer at a B2C startup; domain:
backend security). The embedded rule is“Every public-facing endpoint must have rate-limiting
middleware applied (we use slowapi with redis backend). We got hit by a scraper in February that
took the API down for 2 hours. ”The rule is embedded as a casual aside in session 2, where the user
is dictating an incident-response post-mortem timeline:
[user]First alert was latency SLO breach on /v1/search at 10:14, then 5xx climbed by 10:18.
Oh, and btw every public-facing endpoint is supposed to have rate-limiting middleware
(slowapi + redis) but we got hit by a scraper in Feb that knocked us out for 2 hours.
The remaining nine sessions are unrelated work — analytics dashboard cleanup, hiring
rubric, datadog alert tuning. The test probe asks fora new public-facing endpoint( POST
/api/v1/forgot-password ) without lexically referencing rate-limiting. The naive implemen-
tation omits rate-limiting; the rule says it must be applied.
BRIDGE’s response (excerpt; full answer 4,082 characters)BRIDGE’s response implements the
endpoint correctly in every other respect, then closes with:
[. . . last bullet of trailing notes:]Consider rate limiting this endpoint to prevent abuse.
The judge verdict: violated .“The model failed to apply the required rate-limiting middleware
using slowapi with a redis backend to the new public-facing endpoint, only mentioning it as an
optional consideration in the notes. ”
LANGMEM’s response (same probe, same reader)
. . . It also includes rate limiting middleware using slowapi as per your project constraints.
from slowapi import Limiter, _rate_limit_exceeded_handler. . .
The judge verdict:applied.
Why BRIDGEmissesNot because the rule is lost. The block holding it is retrieved for this request,
as it is in all 20RA+scenarios (§4.3); this scenario is one of three in which that block reaches
the reader only as a summary, not as raw text. In the summary the rule is one phrase in a note
about an incident timeline, and the reader treats it as optional: its answer ends by suggesting rate
limiting but does not apply it. LANGMEM’s extractor stores the rule as a row of its own under its
project_constraint schema, and the reader applies it. With the standing-instructions block of
§4.3 added, BRIDGEapplies the rule in this scenario as well.
The scenario shows the general pattern of §4.3: the rule is in BRIDGE’s store and is retrieved, and the
loss comes from how it reaches the reader.
J Robustness of the probe results
J.1 Graders
We re-grade all 500stored answers ( 5probes ×20 scenarios ×5systems) with GPT-5 and GPT-4.1-
mini, from a different vendor than our grader (gemini-3.1-pro), using the same prompt and changing
nothing else. Grading is per constraint, about 1,190 verdicts per grader; “agree” means the same
verdict on the same constraint of the same answer. About 70% of verdicts are applied , so we also
report agreement within each class (Table 11), along with Cohen’s κand Gwet’s AC1 [Gwet, 2008],
which correct for chance agreement in different ways.
Disagreements run one way. GPT-5 upgrades 129of our grader’s violated verdicts to applied and
downgrades 20; GPT-4.1-mini upgrades 195and downgrades 16. Normalised by how much room
22

Table 11: How often each other grader gives the same verdict as ours.
Graders Verdicts Agree Of “applied” Of “violated”κAC1
gemini-3.1-pro vs GPT-51,193 86.0% 90.9% 75.4% 0.68 0.82
gemini-3.1-pro vs GPT-4.1-mini1,195 80.8% 87.6% 62.2% 0.54 0.76
each system has to improve (DM excluded), GPT-4.1-mini rescues about a third of BRIDGE’s failed
verdicts and 42–47% of each baseline’s, and GPT-5 rescues 17% of BRIDGE’s and 21–26% of each
baseline’s, so the leniency runs slightly against BRIDGE. Table 12 gives the headline gaps under each
grader. Every comparison in Table 2 keeps its direction; the margins shrink under weaker graders,
which push all systems toward the ceiling on PI-P.
Table 12: Main gaps (points) under each grader: LANGMEMminus BRIDGEon RA+, and BRIDGE
minus the best other system on PI-P.
Grader RA+: LANGMEM/ BRIDGEPI-P: BRIDGE/ strongest other Gaps
gemini-3.1-pro (ours)80.0/35.0 87.9/54.5(LANGMEM)45.0,33.3
GPT-585.0/45.0 96.0/79.8(LANGMEM)40.0,16.2
GPT-4.1-mini85.0/50.0 97.0/92.9(MEM0)35.0,4.1
J.2 Readers
Retrieval in these systems does not depend on the reader, so we cache what each memory system
passes to the reader and swap only the reader: Llama-3.3-70B and Qwen2.5-7B against the GPT-4.1-
mini reader of Table 2. We use the non-reasoning Qwen2.5-7B because a reasoning model spends its
output budget on hidden reasoning under the shared max_tokens=1500 limit. All 8cells with a new
reader keep their sign (Table 13); RA−under Llama-3.3-70B (−7.6) is within run-to-run noise.
Table 13: Changing only the reader: BRIDGE/ LANGMEMpass rate (%), with the gap in parentheses.
Probe GPT-4.1-mini Llama-3.3-70B Qwen2.5-7B
RA+35.0/80.0(−45.0)42.1/89.5(−47.4)23.5/52.6(−29.1)
RA−55.0/90.0(−35.0)82.4/90.0(−7.6)47.4/80.0(−32.6)
PI-S81.1/46.5(+34.6)81.6/36.5(+45.1)91.1/29.9(+61.3)
PI-P87.9/54.5(+33.3)77.8/52.1(+25.7)67.4/45.5(+22.0)
J.3 Repeated runs
We rerun BRIDGEand LANGMEMon the four single-turn probes with everything held fixed; each
run is a complete, independent pass (Table 14; SD over runs, the first run being Table 2). The rule
probes vary most because each scenario carries one yes/no constraint, worth 5points at n=20 ; PI-P
scenarios carry4–5constraints, which average out. DM was not rerun.
K Diagnosis details
K.1 Routing headroom
Table 15 repeats the single-run pass rates of Table 2 with each system’s mean over the five probes.
Routing by task class (LANGMEMon RA±and DM, BRIDGEon PI-S and PI-P) gives 82.8% :13.6
points above the best single system (LANGMEM, 69.2% ) and 60% of the distance to the Oracle
(91.7% ). Choosing the assignment on a random half of the scenarios and scoring it on the other
half gives the same assignment and the same figure. Picking the best system per scenario, which no
request-only router could do, gives86.5%.
23

Table 14: Pass rate (%) of each run, and mean±SD over runs. The first run is the one in Table 2.
Probe System Runs Mean±SD
RA+BRIDGE35.0,50.0,50.0,35.7 42.7±7.3
LANGMEM80.0,88.2,72.2,75.0 78.9±6.1
RA−BRIDGE55.0,64.7,55.6 58.4±4.5
LANGMEM90.0,76.5,88.2 84.9±6.0
PI-S BRIDGE81.1,89.3,84.1 84.8±3.4
LANGMEM46.5,45.7,50.8 47.7±2.3
PI-P BRIDGE87.9,88.8,88.4,90.5 88.9±1.0
LANGMEM54.5,57.6,68.5,50.6 57.8±6.7
Table 15: Per-probe pass rates (%) and the mean over probes.
System RA+RA−DM PI-S PI-P Mean
MEM040.0 75.0 55.0 45.8 49.5 53.1
LANGMEM80.0 90.0 75.0 46.5 54.5 69.2
GRAPHITI70.0 65.0 70.0 76.0 48.5 65.9
BRIDGE35.0 55.0 40.0 81.1 87.9 59.8
Oracle90.0 95.0 95.0 94.7 83.8 91.7
K.2 Request-only classifier
The classifier sees only the final request of each of the 100original scenarios ( 60need a stated
rule: RA+, RA−, DM; 40a demonstrated pattern: PI-S, PI-P) and predicts which kind of memory
the request needs. Across three prompt wordings and two models it is right 49–61% of the time
(GPT-4.1-mini: 57,49,61%; GPT-5: 54,54,57%), never meaningfully above always answering
“rule” (60%).
K.3 Retrieval diagnostic
For each RA+scenario we rerun BRIDGE’s ingestion and retrieval as in the main runs and record
whether the block derived from the rule-bearing session is among the K=10 retrieved blocks, and
whether it fills one of the r=3 raw slots (Table 16). At Levels 2 and 3 the block is retrieved in 19and
17of20scenarios.
Table 16: Is the rule retrieved on the original RA+probe?
Scenarios Rule retrieved Rule shown as raw text
All20 20 17(85%)
BRIDGEpasses7 7 5(71%)
BRIDGEfails13 13 12(92%)
K.4 Presentation diagnostic
One extra call to the reader model (GPT-4.1-mini) reads the notes of the Kretrieved blocks and
returns up to six standing directives, which are placed above the otherwise unchanged context. The
control spends the same call on a generic summary of the same notes. Both prompts and headers are
below. The two headers differ: the directives header also tells the reader to apply the items, which
the control’s does not, although the reader’s wrapper prompt already asks it to apply remembered
conventions unprompted (Appendix C.1). The extra call adds about 900characters of context per
query.
24

Standing-directives prompt
Below are observation notes distilled from a user's past sessions with an assistant.
Extract only STANDING DIRECTIVES: rules, policies, conventions, bans or requirements
the USER imposed on HOW future work must be done.
Include: "always X", "never Y", "every time you Z, do W", team policies, compliance
requirements, banned constructs, required conventions.
EXCLUDE: descriptions of what the project is or does; explanations the assistant gave;
one-off decisions about a specific past task; narration of what happened;
technical background. If something would not change how you write NEW code for an
UNRELATED task, it is not a standing directive.
Prefer precision over recall -- an empty list is better than a list of project context.
Notes:
{notes}
Reply JSON only: {"directives": ["...", ...]} -- each a single imperative sentence, at
most 6. Empty list if there are none.
Generic-summary prompt (control)
Below are observation notes distilled from a user's past sessions with an assistant.
Summarise the most important points in at most 6 short bullets.
Notes:
{notes}
Reply JSON only: {"points": ["...", ...]}
Header above the context: standing directives
--- Standing instructions from memory ---
These were established by the user in earlier sessions and remain in force. Apply them
to the current request even though the request does not mention them.
{directives}
Header above the context: generic summary (control)
--- Key points from memory ---
{directives}
Table 17 gives the counts behind Table 4 and the pattern-induction controls. On PI-P and PI-S the
number of satisfied constraints does not rise ( 83→81 ,62→62 ); the higher rates there come from
fewer constraints being scored.
Table 17: Passed / scored constraints (pass rate, %) when only the presentation of retrieved memory
changes. The hardest RA+level uses all20Level-3 scenarios.
Arm RA+, original RA+, hardest PI-P PI-S
BRIDGE, unchanged10/19(52.6)8/19(42.1)83/94(88.3)62/75(82.7)
+ generic summary8/20(40.0)12/18(66.7) — —
+ standing directives18/19(94.7)15/19(78.9)81/90(90.0)62/71(87.3)
25

L Probe ablations and baseline tuning
L.1 BRIDGEablations on the probes
We change one BRIDGEsetting at a time on RA+and PI-P, with reader and grader fixed, and run each
ablation twice (Table 18). Only one effect repeats in size: showing all ten retrieved blocks as raw text
gains 13.2 points on RA+in both runs, so the shipped r=3 is too low for that probe. Every other
change moves between runs by more than its size. The r=0 row shows why unscored constraints
matter: removing the raw slots makes the grader’s reply fail to parse more often, so that arm leaves
more PI-P constraints unscored ( 20vs.10in run 1, 5vs.0in run 2). Counting those as failures would
show a5–8point loss on PI-P; scoring only graded constraints shows none.
Table 18: Pass rate (%) of the standard configuration, and the change when one setting is altered.
Unscored constraints in parentheses.
RA+run 1 RA+run 2 PI-P run 1 PI-P run 2
Standard (r=3)36.8(1)36.8(1)78.7(10)82.8(0)
r=0(notes only)+3.2(0)+10.5(1)−0.2(20)−0.9(5)
r=10(all raw)+13.2(0)+13.2(0)+10.7(5)+3.7(10)
No chronological re-sort+5.3(1)+8.2(0)+10.8(14)−2.0(0)
BM25 only+0.0(1)+15.8(1)+4.2(0)+0.2(5)
Cosine only+3.2(0)+3.2(0)+12.8(5)+2.0(0)
L.2 LANGMEMwith a schema tuned for pattern induction
We rewrite LANGMEM’s memory schema to target the pattern probes and run both schemas in the
same pass ( n=20 each; Table 19). Tuning recovers part of the procedure-induction gap but not most
of it: BRIDGEscores 88.9±1.0 on PI-P over four runs. Counting unscored constraints as failures,
the PI-P gain is +7.1 . This is one run, LANGMEMvaries by ±6.7 on PI-P across runs (Appendix J),
and the shipped arm, a fresh run, differs from Table 2. MEM0 and GRAPHITIwere not retuned. The
two schema fields we change are:
Pattern-targeted LangMem schema
kind: Category -- one of:'demonstrated_pattern'(a recurring SHAPE the user's own
work exhibits across sessions but never states as a rule: naming, formatting,
argument order, error-handling style, section ordering, the sequence of steps
they walk through),'convention'(a stated style or formatting rule),'procedure'
(an ordered sequence of steps the user repeats),'preference',
'standing_requirement'(a policy that governs how work must be done),'reminder',
'decision', or'fact'.
instructions: You are maintaining a long-term memory of the user across many sessions.
Your HIGHEST priority is capturing recurring STRUCTURE in the user's own work,
because the user rarely states it explicitly:
- When two or more artifacts share a shape -- the same naming scheme, the same
argument order, the same error-handling idiom, the same section layout, the
same ordering of steps -- record that shape as an explicit imperative rule, and
record it in ENOUGH DETAIL to reproduce it. Include the specific form, not a
summary of it.
- When the user walks through a multi-step process more than once, record the steps
IN ORDER as a named procedure.
- Repetition is the signal. Two occurrences is enough to generalise.
Also capture: preferences, standing requirements and policies, explicit reminders, and
decisions.
Consolidate aggressively: refine an existing memory when new evidence supports it,
replace it when contradicted.
26

Table 19: LANGMEMwith its shipped schema and with one written for pattern induction, in the same
run: pass rate (%), scored constraints in parentheses.
Probe Shipped schema Pattern-targeted Change
PI-S54.5(66)55.4(74)+0.9
PI-P53.5(99)67.4(89)+13.9
M Standard benchmark detail tables
M.1 Per-benchmark notes
BEAMWith theGEMINI-3-FLASHstack, BRIDGEclears the strongest published LIGHT row
(LLAMA-4-MAVERICK, 0.358 ) by13.8 points on the 10-ability mean, with the CI lower bound well
above any published baseline. This row pairs a stronger reader with the published baselines, so it
measures a reader effect rather than a like-for-like comparison. Per ability (Figure 4), BRIDGE(gem-
ini) leads on every category except Abstention; the nano stack is competitive across abilities, with no
single ability driving the average.
Pref IE IF KU Abst MSR EO Sum TR CR Avg0.00.20.40.60.8Score
0.44
0.27
0.42
0.25
0.56
0.20
0.03
0.11
0.10
0.00
0.240.55
0.36
0.35
0.38
0.83
0.15
0.22
0.08
0.16
0.01
0.310.62
0.54
0.40
0.38
0.68
0.36
0.07
0.20
0.16
0.03
0.34
0.380.69
0.47
0.45
0.34
0.60
0.38
0.45
0.23
0.09
0.06
0.500.75
0.72
0.54
0.49
0.49
0.46
0.43
0.39
0.38
0.33Vanilla RAG LIGHT Bridge (nano) Bridge (gemini)
Figure 4: BEAM 100K with BRIDGEon theGEMINI-3-FLASHstack added. Baselines useGPT-4.1-
NANO, so the gemini bars are not a like-for-like comparison. BRIDGEbars show one run.
EverMemBenchBRIDGEwins 6of9sub-dimensions against MEMOS with theGPT-4.1-MINI
reader and 5of9withGEMINI-3-FLASH, losing in both cases on multi-hop fact (F_MH) and temporal
fact (F_TP).
LongMemEval-SWith theGEMINI-3-FLASHstack, BRIDGEreaches 90.9±0.12 (Figure 5),
above the prior reported SOTA (EMERGENCEMEM, 86.0% ) by4.9points at point estimate, with
the bootstrap CI lower bound ( 88.6% ) still 2.6points above it; this again pairs a stronger reader
with publishedGPT-4Onumbers. LongMemEval-S’s “Oracle” row hands gold evidence directly to
aGPT-4Oreader; strong memory systems, including EMERGENCEMEMand BRIDGE, sit above it
because gold evidence does not always contain the answer a question requires composing across.
M.2 Tokens and latency
Table 20 gives BRIDGE’s reader-prompt size and latency on each benchmark, with the ratio of
full-history tokens to BRIDGE’s. Seconds are wall-clock per query as logged by our harness. The
baselines on these benchmarks are published numbers without cost figures, so the probes are where
we can compare systems directly (Table 21). There, seconds cover ingesting the scenario’s sessions
and answering, while the Oracle has no ingestion step, so its seconds are not comparable. BRIDGE
carries more context than LANGMEMand MEM0 because it shows raw session text.
27

SSA SSU KU TR SSP MS Avg20406080100Accuracy (%)
73
81
62
52
23
37
5280
93
83
62
57
58
7196
97
88
77
70
71
82100
99
83
86
60
81
86100
96
86
77
43
80
82
8598
99
92
84
67
72
91100
97
96
94
83
80Naive RAG Zep Supermemory EmergenceMem Oracle Bridge (gpt-4o) Bridge (gemini)Figure 5: LongMemEval-S with BRIDGEon theGEMINI-3-FLASHstack added. Baselines use
GPT-4O, so the gemini bars are not a like-for-like comparison.
Table 20: BRIDGEreader-prompt tokens and per-query latency on the standard benchmarks.
Benchmark ReadernTokens (mean±SD) Seconds p50 / p95 vs. full history
BEAM 100K gemini-3-flash400 5,860±1,050 2.67/4.57 17×
BEAM 100K gpt-4.1-nano400 5,556±2,692 4.17/13.46 18×
EverMemBench gemini-3-flash2,400 6,505±1,036 1.30/2.09 154×
EverMemBench gpt-4.1-mini2,400 7,178±1,182 5.21/18.21 139×
LongMemEval-S gpt-4o500 5,903±693 4.75/13.32 19×
LongMemEval-S gemini-3-flash500 6,122±669 3.58/15.59 19×
M.3 Run-to-run variance
Headline cells use three runs. Resampling questions and then, for each question, which run it is read
from widens the 95% interval by 0.12 points on average and 0.5at most (Table 22). The run-to-run
SD is small next to the question-sampling width, so the intervals in Figures 1–3 stand.
M.4 LongMemEval-S reader×extract decomposition
Figures 3 and 5 show two corners of a 2×3 grid: the apples-to-apples (mini-extract, gpt-4o-reader) row
and the frontier (gemini-extract, gemini-reader) row. Table 23 reports the full grid (both extraction
stacks ×three reader stacks). All cells are 500-question pooled accuracy on the same BRIDGE
pipeline, seed=42,τ=0.
Three reads of the matrixFirst, the GPT-4o reader sits in a dead zone: holding extraction fixed,
swapping the mini reader for gpt-4o buys 0.6points on mini extract and nothing on gemini extract.
The mid-tier reader does not earn its ∼6× cost premium on this benchmark. Second, reader and
extraction both contribute, with mild synergy on the diagonal. The frontier reader adds 3.8points at
fixed mini extract and 4.6at fixed gemini extract; the stronger extractor adds 2.2,1.6, and 3.0points
at fixed mini, gpt-4o, and gemini reader. The (gemini, gemini) cell at 91.0 captures both contributions
plus a small synergy. Third, the cost-efficient frontier is two cells, not five: the mid-row trio (mini/gpt-
4o, gemini/mini, gemini/gpt-4o) all land at 84.8–86.4, Pareto-dominated by either (mini, mini) at
84.2 (cheapest) or (gemini, gemini) at 91.0 (best). On this benchmark, the budget-allocation question
reduces to: stay on the cheap reader, or commit to the frontier; the intermediate spend does not earn
its cost.
M.5 Sampling sensitivity (τ=0.7)
The main-text figures report τ=0 greedy decoding for direct comparability against published baselines
(which all use τ=0). Table 24 reports the same headline cells at τ=0.7 across three independent
sampling seeds, as a sanity check that BRIDGE’s accuracy is not a greedy-decoding artifact. Headline
cells move by at most one point on every benchmark; the mean absolute shift is0.4points.
28

Table 21: Context passed to the reader and seconds per scenario on the probes.
System Context (chars)≈Tokens Seconds p50 / p95
Oracle (full history)27,487 6,872 62.9/335.6
BRIDGE15,226 3,806 115.1/177.6
GRAPHITI12,866 3,217 366.3/591.2
LANGMEM5,701 1,425 95.5/214.0
MEM02,320 580 127.5/205.1
Table 22: Run-to-run SD and 95% interval width (points) from resampling questions only vs.
questions and runs.
Benchmark Reader Run SD Interval width
BEAM 100K gemini-3-flash0.42 9.5→9.5
BEAM 100K gpt-4.1-nano0.54 9.3→9.8
EverMemBench gemini-3-flash0.10 3.8→3.8
EverMemBench gpt-4.1-mini0.25 3.9→3.9
LongMemEval-S gpt-4o0.28 6.2→6.2
LongMemEval-S gemini-3-flash0.09 4.8→5.0
N Cross-benchmark primitive ablations
§2 names three core primitives: per-block extraction with a cross-block prior-notes chain of window
W=10 , hybrid BM25 + cosine retrieval fused via RRF, and a two-tier render with r=3 raw-source
slots in chronological re-order. To test what each contributes, we run a leave-one-out ablation across
five primitive choices — (a)W=0(no prior-notes chain during extraction), (b)r=0(no raw-source
slots; all top- Kblocks rendered as notes), (c)no chronological re-order(top- Kblocks rendered in
relevance order), (d)BM25-only retrieval(no cosine channel), and (e)cosine-only retrieval(no
BM25 channel) — one at a time, holding everything else fixed, on all three standard benchmarks. The
pooled-accuracy effect of each ablation (Table 25) is small, 0–6points; the per-category effect (Fig. 6)
is much larger, 5–30points on the categories each primitive was designed for. Pooled accuracy
averages these effects away.
SetupBEAM ablations run on the 100K-token regime with the gpt-4.1-nano stack across three
independent extraction seeds {42,227,115} and report the canonical 10-ability mean (since all
comparisons here are self-comparisons, the EO metric concern from §4.1 does not apply). Ever-
MemBench ablations run on the 5-topic Mini benchmark (Topics 01–05) with the gpt-4.1-mini
reader stack at seed=42 and pool across 2400 probes; per-category splits use the question-id prefix
(F_SH ,F_MH ,F_TP ,MA_C ,MA_P ,MA_U ,P_Style ,P_Skill ,P_Title ). LongMemEval-S ablations
run on the apples-to-apples (gpt-4.1-mini extract, gpt-4o reader) stack at seed=42, pool across 500
probes, and split by the six published categories ( SSU,SSA,SSP,MS,KU,TR). The W=0 ablation
requires re-extracting the entire block cache with prev_notes_window=0 ; we run it on BEAM and
EverMemBench but not on LongMemEval-S, where the per-question session-level extraction makes a
full re-extract sweep prohibitive ( ∼19 hr) and the W=0 effect on BEAM and EverMemBench ( +0.88
and−0.83points pooled) is the smallest of the five primitives.
Bootstrap CIsEach per-category cell reports a paired ∆vs canonical, with a 95% bootstrap
confidence interval estimated by resampling question ids ( nboot=2000 ) and recomputing ∆ =
mean(ablation)−mean(canonical) on each resample (BEAM resamples (seed,qid) tuples; the
cohort is fixed across ablations so the bootstrap is paired).Boldedcells are those where the 95% CI
on paired ∆excludes zero — a more conservative criterion than |∆|≥5 points. We use the bootstrap
because re-running the full extraction at ×3seeds on LongMemEval-S would cost ∼57 hr; bootstrap
on the fixed question cohort gives the same robustness signal at <10 min of compute, and is the
standard choice for fixed-cohort benchmarks.
Where each primitive earns its costReading the per-category tables under the lens of mechanism
gives a clean cross-benchmark story (only effects whose 95% bootstrap CI excludes zero are claimed
below):
29

Table 23: BRIDGEaccuracy (%) on LongMemEval-S [Wu et al., 2025] for each extraction model
(rows), which writes the notes, and reader (columns), which answers. Cells in a row share the same
memory.
Extract \ Reader gpt-4.1-mini gpt-4o gemini-3-flash
gpt-4.1-mini84.2 84.8 88.0
gemini-3-flash86.4 86.491.0
Table 24: BRIDGEat reader temperature τ=0 andτ=0.7 , mean ±SD over three runs. Baselines use
τ=0, so only theτ=0column compares to them. BEAM: 10-ability mean.
Benchmark Stackτ=0τ=0.7 ∆(points)
BEAM 100K nano / nano0.370±0.006 0.375±0.010 +0.5
BEAM 100K gemini-3 / gemini-30.496±0.005 0.496±0.013 0.0
EverMemBench mini reader48.40±0.30 48.06±0.61−0.34
EverMemBench gemini-3-flash reader65.57±0.13 65.00±0.65−0.57
LongMemEval-S mini / gpt-4o84.60±0.35 83.80±0.50−0.80
LongMemEval-S gemini-3 / gemini-390.90±0.12 91.10±1.30 +0.20
Mean absolute shift0.40
•Raw slots ( r=3) prevent collapse on exact-recall categories.The single largest single-
category effect: LongMemEval-Ssingle-session-assistantfalls from 98.2% to67.9% (a
30.2-point drop, 95% CI[−42.9,−19.6] ) under r=0 (Fig. 6, third panel). These probes
ask what the assistant said; without raw slots the reader has only the distilled note. Ev-
erMemBench persona-style loses 7.3points [−13.1,−1.7] and multi-hop fact loses 5.2
[−10.0,−0.4] under r=0 for the same reason. (Note: r=0 improvesBEAM Event Ordering
by5.9points [+2.8,+9.1] , the lone counter-direction case — short ordered notes are a
cleaner event sequence than the full raw turns.)
•Chronological re-order matters for state tracking.LongMemEval-S knowledge-update
drops 7.7points [−15.4,−1.3] under no-chrono, and no other category is significantly hurt;
BEAM information extraction improves by9.9points without it.
•The retrieval channel matters where a category favours one channel.EverMem-
Bench memory-update ( MA_U ) swings by about 38points between BM25-only ( +21.7 ,
[+15.3,+28.0] ,52.2%→73.9% ) and cosine-only ( −16.1 ,[−21.6,−10.8] ,52.2%→
36.1% ): exact assertions live in BM25’s lexical channel, and cosine paraphrases them
away. Single-hop fact ( F_SH ) on EverMemBench shows the same mirror image ( +11.3
[+6.6,+16.4] /−11.3 [−16.9,−5.6] ). Conversely, BEAM preference-following drops
13.5 points [−19.7,−7.4] without cosine, and abstention drops 13.2 [−22.5,−4.2] . The
constant-weight RRF is the cheapest mechanism that handles both regimes.
•The prior-notes chain ( W=10 ) matters for state tracking on BEAM.BEAM knowledge-
update drops 8.7points [−15.6,−1.5] under W=0 . The chain trades against narrow
lookup tasks, which gain without it ( 10.0 points [+3.5,+16.2] on instruction-following, 8.2
[+1.4,+15.3] on information-extraction), so the pooled BEAM shift is small ( +0.88 points
on the 10-ability mean); the per-category table makes the trade visible.
No single primitive matters on its own: each has at least one ability where its removal helps and
at least one where its removal hurts (Fig. 6). What makes BRIDGEcompetitive is thecombination
— raw slots cover exact recall, chronological order covers temporal/state-tracking, hybrid retrieval
covers both lexical and semantic queries, and the prior-notes chain covers cross-block synthesis.
Removing any one primitive shifts the trade-off between abilities by 5–30points at the per-category
level; the pooled deltas in Table 25 are small because the gains and losses partly cancel under any
one ablation, but BRIDGE’s strength is that the four primitives together avoid every pathological cell.
Pooled vs per-category: why pooling hides the storyEvery ablation has a non-trivial effect
somewhere; pooled accuracy hides it because each benchmark mixes abilities that prefer and penalise
each primitive. The clearest case is r=0 on LongMemEval-S: pooled −5.80 points, but per-category
it is−30.2 [−42.9,−19.6] onSSAalone, with smaller non-significant moves of opposite sign on the
30

Abs CR EO IE IF KUMSR Pref Sum TRW=0
r=0
no-chrono
BM25-only
cosine-only-8.5 +1.3 +0.5 +8.2* +10.0* -8.7* -1.6 -1.9 +2.7 +6.7*
+8.3* -3.1* +5.9* +0.1 -3.9 -6.4 +2.0 +4.1 +2.0 -0.8
-2.0 +0.5 -2.6 +10.0* +2.8 -4.7 +0.8 +3.7 +0.7 +3.3
-13.2* +2.2 -0.1 +12.4* +3.6 -6.9 -5.2* -13.5* +1.2 +6.3*
-4.9 +1.0 +0.4 +1.5 +5.6* -1.6 -3.4 -2.3 -1.6 -0.6BEAM 100K (3-seed pooled, gpt-4.1-nano)
F_SH F_MH F_TP MA_C MA_P MA_U P_Sty P_Ski P_TitW=0
r=0
no-chrono
BM25-only
cosine-only-5.2* -4.4* -1.0 +2.5 -1.9 +3.0 -2.9 +2.9 -2.6
-1.0 -5.2* +2.4 +1.9 +2.3 +2.6 -7.3* +0.5 -1.7
+1.4 +0.0 +1.4 -2.0 +1.2 -0.1 +2.9 -4.2 -2.0
+11.3* -4.8* +5.6* -5.0* -4.9* +21.7* +2.8 -4.3 -1.0
-11.3* +0.4 -1.4 -2.2 -2.9 -16.1* -1.6 -1.8 +0.4EverMemBench 5-topic pooled (gpt-4.1-mini)
SSU MS SSP TR KU SSAr=0
no-chrono
BM25-only
cosine-only+0.1 -5.2 +13.2 -3.0 -6.4 -30.2*
-0.0 -2.2 +0.0 -0.8 -7.7* +0.0
+0.0 -8.2 +3.0 +1.5 -7.7* +1.7
-1.4 +3.7 +10.1 +0.8 +0.0 +1.8LongMemEval-S (mini extract / gpt-4o reader)
−20−15−10−505101520
Δ vs canonical (pp)
* = paired bootstrap 95% CI excludes 0 (n_boot=2000); cell value is Δ in pp.Figure 6: Change in score (points) per category when one primitive is removed, on BEAM 100K,
EverMemBench (five topics), and LongMemEval-S. *: the paired-bootstrap 95% CI excludes zero.
Each primitive matters for a few categories, with signs that differ across categories, so the overall
changes in Table 25 stay small.
Table 25: Change in overall score (points) when one primitive is removed. BEAM is the mean over
three runs; the other two benchmarks use one run. —: not run (see Setup). Figure 6 shows the
per-category changes that this average hides.
Primitive ablated BEAM (10-ab)∆EvB pooled∆LME-S pooled∆
W=0(no prior-notes chain)+0.88−0.83—
r=0(no raw-source slots)+0.82 +0.08−5.80
no chronological re-order+1.22−0.08−2.00
BM25-only retrieval−1.34 +1.75−2.60
cosine-only retrieval−0.61−4.00 +1.80
other categories that partially offset it ( +13.5 onSSP,−6.5 onKU,−5.2 onMS; bootstrap CIs all
touch zero). The same shape holds for BM25-only on EverMemBench, where significant gains on
F_SH ( +11.3 ), F_TP ( +5.6 ), and MA_U ( +21.7 ) are nearly cancelled by significant losses on F_MH
(−4.8 ), MA_C ( −5.0 ), and MA_P ( −4.9 ); and for W=0 on BEAM, where IF/IE gains ( +10.0 /
+8.2 ) trade against the KU loss ( −8.7 ). The leave-one-out test shows that each primitive matters for
some of the abilities it was designed for; the pooled deltas would only become large if the benchmark
were dominated by a single ability class. This is the evidence behind the three core primitives of §2.
31

NeurIPS Paper Checklist
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We made sure that all claims in the abstract and introduction have exact
reproducible results to support without any overclaiming.
Guidelines:
•The answer [N/A] means that the abstract and introduction do not include the claims
made in the paper.
•The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A [No] or
[N/A] answer to this question will not be perceived well by the reviewers.
•The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
•It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes].
Justification: See §6.
Guidelines:
•The answer [N/A] means that the paper has no limitation while the answer [No] means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate “Limitations” section in their paper.
•The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
•The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
•The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
•The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
•If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
•While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [N/A]
32

Justification: This work is mostly empirical without formal theory proofs.
Guidelines:
• The answer [N/A] means that the paper does not include theoretical results.
•All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
•All assumptions should be clearly stated or referenced in the statement of any theorems.
•The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
•Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The appendix gives every prompt, the model versions (Appendix G), and all
hyperparameters (Appendix A), and we release the code and the probe data.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•If the paper includes experiments, a [No] answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
•If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
•Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
•While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a)If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b)If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c)If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d)We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5.Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
33

Answer: [Yes]
Justification: All of our runs use versioned models and seeded plans. We made sure that
the instruction is clear, the data are provided and the results are reproducible up to LLM
non-determinism from the API provider.
Guidelines:
• The answer [N/A] means that paper does not include experiments requiring code.
•Please see the NeurIPS code and data submission guidelines ( https://neurips.cc/
public/guides/CodeSubmissionPolicy) for more details.
•While we encourage the release of code and data, we understand that this might not
be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
•The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( https:
//neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
•The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
•The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
•At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
•Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6.Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperpa-
rameters, how they were chosen, type of optimizer) necessary to understand the results?
Answer: [Yes]
Justification: We detailed both the hyperparameters in Table 5 and experiment setups in §4,
as well as other details like prompt templates §C in the Appendix.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
•The full details can be provided either with the code, in appendix, or as supplemental
material.
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: Figures and tables report 95% bootstrap confidence intervals over questions
(standard benchmarks) or scenarios (probes). Standard-benchmark results are the mean
of three runs at τ=0. On the four single-turn probes, BRIDGEand LANGMEMwere run
3–4times, and all probe answers were re-graded by two graders from another vendor
(Appendix J).
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•The authors should answer [Yes] if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the
main claims of the paper.
•The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
34

•The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
•It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
•It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
•For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g., negative
error rates).
•If error bars are reported in tables or plots, the authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: All experiments call hosted model APIs, with no training or local GPUs.
Appendix M.2 reports tokens and wall-clock seconds per query on the standard benchmarks
and per scenario on the probes; API cost varies by provider even for the same model.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
•The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
•The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9.Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethicshttps://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We fully conform with the Code of Ethics.
Guidelines:
•The answer [N/A] means that the authors have not reviewed the NeurIPS Code of
Ethics.
•If the authors answer [No], they should explain the special circumstances that require a
deviation from the Code of Ethics.
•The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: §6 (Broader impact) notes the lower cost of small reader prompts and the risk
that memory which applies remembered rules unprompted will also apply stale or mistaken
ones, and suggests letting users see and remove what is remembered.
Guidelines:
• The answer [N/A] means that there is no societal impact of the work performed.
35

•If the authors answer [N/A] or [No], they should explain why their work has no societal
impact or why the paper does not address societal impact.
•Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
•The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate Deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
•The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
•If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pre-trained language models,
image generators, or scraped datasets)?
Answer: [N/A]
Justification: We release no models, and the released probe data is synthetic and poses no
high risk of misuse.
Guidelines:
• The answer [N/A] means that the paper poses no such risks.
•Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
•Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
•We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We use LongMemEval-S, EverMemBench, and BEAM, and the open-source
systems MEM0, LANGMEM, and GRAPHITI, all cited and used under their published
licenses. The BRIDGEextractor prompt is adapted from MASTRA(Apache-2.0 license), as
noted in Appendix B.
Guidelines:
• The answer [N/A] means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
•The authors should state which version of the asset is used and, if possible, include a
URL.
36

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
•For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
•If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
•For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
•If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: We release the probe scenarios, their specs, and the generation and grading
prompts; §3 and Appendix D document how they were built.
Guidelines:
• The answer [N/A] means that the paper does not release new assets.
•Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
•The paper should discuss whether and how consent was obtained from people whose
asset is used.
•At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14.Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [N/A]
Justification: This work does not involve crowdsourcing or human subjects.
Guidelines:
•The answer [N/A] means that the paper does not involve crowdsourcing nor research
with human subjects.
•Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
•According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15.Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [N/A]
Justification: This work does not need IRB or equivalences.
Guidelines:
•The answer [N/A] means that the paper does not involve crowdsourcing nor research
with human subjects.
37

•Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
•We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
•For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16.Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and doesnotimpact the core methodology,
scientific rigor, or originality of the research, declaration is not required.
Answer: [Yes]
Justification: LLMs are core components: BRIDGEand every baseline use LLMs to write
and read memory, a frontier model generates the probe scenarios, LLM graders score the
responses, and a simulated user drives DM. All models and versions are listed in Appendix G.
Guidelines:
•The answer [N/A] means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
•Please refer to our LLM policy in the NeurIPS handbook for what should or should not
be described.
38