# Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge

**Authors**: Neeraj Yadav

**Published**: 2026-06-25 01:31:53

**PDF URL**: [https://arxiv.org/pdf/2606.26511v1](https://arxiv.org/pdf/2606.26511v1)

## Abstract
Retrieval-augmented generation (RAG) gives agents access to accumulated knowledge, but has no model of time. When a fact changes (e.g., a function is renamed or API restructured), RAG retrieves both the stale and current value with near-identical embedding similarity. The agent then either abstains or serves the superseded fact. We show this is a structural problem: on a calibrated dataset, cosine similarity distinguishes a contradicted fact from a duplicated one with AUROC 0.59 (near chance), as contradictions are often more embedding-similar to the original than rephrased duplicates.
  We present MemStrata, a retrieval memory maintaining temporal validity. It stores facts like RAG, preserving static recall, but when a fact's value is contradicted, a deterministic (subject, relation, object) supersession rule retires the stale value in a bi-temporal ledger - with no similarity threshold and no LLM call. Across six benchmarks run locally with a 7B model, MemStrata ties RAG on static knowledge and reaches 0.95-1.00 accuracy on evolving knowledge (where RAG reaches 0.20-0.47). The central result is the stale-fact-error rate: when required to answer, RAG serves superseded values 15-40% of the time; MemStrata drives this to ~0%, a failure class RAG cannot avoid. MemStrata achieves this at retrieval latency (~2.1s) versus ~16-18s for LLM-reranking baselines. We release the harness, datasets, and a marker-free evaluation protocol for memory under knowledge evolution.

## Full Text


<!-- PDF content starts -->

Temporal Validity in Retrieval Memory:
Eliminating Stale-Fact Errors for AI Agents over Evolving
Knowledge
A deterministic supersession layer that retrieval-augmented
generation cannot match by construction
Neeraj Yadav
MemStrata.dev — Called It Inc. (Enterprise)
memstrata@gmail.com
Draft v2 (temporal-validity framing)
For double-blind submission, anonymize the author block and the product/repository identifiers. All numbers are
from the clean re-run ( REPORT_PAPER1.md ,REPORT_PAPER1_forced.md ,calibration/REPORT_synthetic.md ),
generated with the fixed plain-text grader, local and deterministic (temperature 0, seed 0, no network). Regen-
erate every figure from those source files before submission.
Abstract
Retrieval-augmented generation (RAG) gives language-model agents access to accumulated
knowledge, but it has no model oftime. When a fact changes — a function is renamed, a
configuration value or dependency version is bumped, an API is restructured — RAG retrieves
both the stale and the current value with near-identical embedding similarity and cannot
determine which is current. The agent then either abstains or serves the superseded fact. We
show this is not a tuning problem but a structural one: on a calibrated dataset, cosine similarity
distinguishes a contradicted fact from a duplicated one with AUROC 0.59 (near chance), and
contradictions are on averagemoreembedding-similar to the original than rephrased duplicates
are. We present MemStrata, a retrieval memory that maintainstemporal validity. It stores facts
like RAG, preserving full recall on static knowledge, but when a fact’s value is contradicted by a
newer assertion, a deterministic (subject, relation, object) supersession rule retires the stale value
in a bi-temporal ledger — with no similarity threshold and no LLM call. Across six benchmarks
run entirely on consumer hardware with a 7B local model — two static (project-fact QA,
multi-session dialogue) and four marker-free evolving (code mutation, configuration migration,
dependency bumps, API evolution) — MemStrata ties RAG on static knowledge (no recall
cost) and reaches 0.95–1.00 accuracy on evolving knowledge where RAG reaches 0.20–0.47. The
central result is the stale-fact-error rate: when required to answer, RAG serves the superseded
value 15–40% of the time; MemStrata drives this to ∼0%, a failure class RAG cannot avoid by
construction. MemStrata achieves this at retrieval latency ( ∼2.1s, the embedding floor) versus
∼16–18s for LLM-reranking and LLM-verification baselines, because no language model runs on
the read path. We release the harness, prompts, datasets, and a reproducible evaluation protocol,
and we recommend a marker-free benchmark invariant for evaluating memory under knowledge
evolution.
1arXiv:2606.26511v1  [cs.CL]  25 Jun 2026

1 Introduction
Language-model agents are increasingly deployed as persistent collaborators that accumulate knowl-
edge across many sessions: a coding assistant that learns a codebase, a research assistant that tracks
a literature, an operations assistant that knows a system’s configuration. For these agents, the
binding constraint is no longer raw model capability but memory — how the agent encodes, retains,
retrieves, andkeeps currentwhat it has learned.
Retrieval-augmented generation [Lewis et al., 2020] is the dominant memory mechanism. It
stores interaction history as embedded chunks and retrieves the top- kmost similar at query time,
controlling prompt size while giving the model access to a large store. RAG handles recall well.
But it has a blind spot that becomes critical as soon as the stored knowledgeevolves: it has no
representation of time. When a fact changes, both the old and new versions remain in the store
with nearly identical embeddings — “the timeout is 1800 seconds” and “the timeout is 3600 seconds”
differ by one token and sit close together in any embedding model. Retrieval surfaces both. The
model has no principled way to tell which is current, so it either abstains (refusing a question it
could answer) or guesses (often serving the stale value with full confidence).
This is acute for code, where knowledge evolves continuously and out of band: functions
are renamed, endpoints move, configuration migrates, dependencies upgrade. An assistant that
confidently reports last month’s port number is worse than useless. But the problem is general —
any domain where facts have a validity period (organizational facts, biomedical findings, current
events) exhibits it.
A natural first instinct is to solve staleness with a better similarity rule: detect when an incoming
fact contradicts a stored one, and update rather than append. We show in Section 3 that this
instinct fails for a fundamental reason. On a calibrated dataset, cosine similarity cannot separate a
contradiction from a duplicate — contradictions are on averagemoresimilar to the original (a value-
flip is a minimal edit) than genuine rephrasings are. No threshold on similarity can distinguish “this
restates a stored fact” from “this contradicts a stored fact.” A learned classifier on top of similarity
does not reliably help either, as our experiments show. The mechanism must be deterministic and
structural, not similarity-based.
We presentMemStrata, a retrieval memory that maintains temporal validity through determin-
istic supersession. Its contributions are:
1.A structural impossibility result for similarity-based staleness detection.On 98
labeled pairs, cosine AUROC for separating contradictions from duplicates is 0.59, and the
maximum achievable precision is 0.67 — the safety floor is unreachable. Contradictions are
more embedding-similar to the original than duplicates are. (Section 3, 5.1)
2.A temporal-validity memory architecture.MemStrata stores facts like RAG (full static
recall) but applies a deterministic (subject, relation, object) supersession rule when a fact’s
value is contradicted, retiring the stale value in a bi-temporal ledger with no similarity threshold
and no LLM call. (Section 4)
3.The stale-fact-error result: a failure class RAG cannot avoid.When required to
answer, RAG serves superseded values 15–40% of the time across four evolving benchmarks;
MemStrata drives this to ∼0%. This is structural, not tuned — RAG retrieves both values
and has no mechanism to choose. (Section 5.3)
4.A marker-free evaluation protocol for memory under evolution.We construct four
evolving benchmarks where the stale and current versions of a fact are textually identical
2

except for the changed value, so the only signal of currency is the memory system’s temporal
mechanism — and we show that a contaminating textual marker silently inflates baselines.
(Section 4.5, 5)
We run all experiments locally and deterministically on consumer hardware, and we are explicit
about the limitation that bounds the claim: our evolving benchmarks are structured single-value
templates, and extraction quality — not the supersession mechanism — is the gating factor for
messier natural-language contradictions (Section 7). We frame this honestly as the subject of
follow-on work rather than papering over it.
2 Related Work
Memory for LLM agents.Recent systems give agents persistent memory of conversations and user
facts: scalablelong-termmemorypipelines[Mem0;Chhikaraetal.,2025], OS-stylememoryhierarchies
with paging and background processing [MemGPT/Letta; Packer et al., 2023], and reflective natural-
language memory for simulated agents [Park et al., 2023]. These target conversational and assistant
settings and emphasize recall depth, typically benchmarked on long-dialogue memory [LoCoMo;
Maharana et al., 2024]. MemStrata differs in mechanism — a deterministic supersession rule that
maintains validity — and in framing: the problem we attack is not recall depth but stale-fact
resistance under knowledge evolution.
Graph and hypergraph RAG.GraphRAG [Edge et al., 2024] and its successors — LightRAG
[Guo et al., 2024], NodeRAG [Xu et al., 2025], and HyperGraphRAG [Luo et al., 2025]; see Han et al.
[2025] for a survey — structure retrieval over entity-relation graphs or n-ary hyperedges, improving
multi-hop retrieval on static corpora. They enrich therepresentationof relationships but retrieve
by similarity over that representation; none introduces a notion of fact currency. Critically for our
framing, Zeng et al. [2025] re-evaluate these systems under a bias-controlled protocol and find their
advantages over naive RAG much smaller than originally reported — in some cases reversing —
confirming that representational richness alone does not address the failure we target. MemStrata is
orthogonal: it adds temporal validity, evaluated on evolving rather than static corpora.
Temporal knowledge graphs and bi-temporal data.Bi-temporal modeling — separating
valid time(when a fact is true) fromtransaction time(when it is recorded) — is long-established
in databases, formalized in the taxonomy of Snodgrass and Ahn [1985], developed into practical
application design and data management [Snodgrass, 1999, Jensen and Snodgrass, 1999], and
later standardized in SQL:2011’s system-versioned and application-period tables [ISO/IEC, 2011].
Temporal knowledge-graph reasoning, in which triples carry validity intervals, is an active area [Cai
et al., 2024]. MemStrata adapts the bi-temporal ledger to LLM-agent memory: facts are retired, not
deleted, preserving validity intervals for future as-of-time queries. Our contribution is not the ledger
primitive but its integration with deterministic extraction-time supersession in an LLM memory
system, and the empirical demonstration that this resolves a failure RAG cannot.
Hallucination and verification.Verification-augmented RAG adds self-checking to reduce
ungrounded generation; Self-RAG [Asai et al., 2023] learns reflection tokens that decide when to
retrieve and critique generated text. We include an LLM relevance-verifier baseline and show it
does not address staleness — it has no temporal signal — and costs ∼8×latency. The structurally
correct mechanism for staleness is temporal and deterministic, not a learned grounding check.
3

3 The Staleness Problem and Why Similarity Cannot Solve It
Consider an agent answering questions over a store that has accumulated, across sessions, both “the
service runs on port 8000” (recorded earlier) and “the service runs on port 8080” (recorded later,
after a migration). A query about the port retrieves both: they are near-identical in embedding
space. The agent must decide which is current. RAG provides no basis for that decision — retrieval
ranks by similarity, and both are maximally similar to the query.
The tempting fix is to detect, at write time, that the second factcontradictsthe first, and to
update rather than append. This requires distinguishing three relationships between an incoming
fact and a stored one: itduplicates(restates) it, itcontradicts(supersedes) it, or it isnovel. If
similarity could separate duplicate from contradiction, a threshold rule would suffice.
It cannot. Section 5.1 reports the calibration: contradictions are on averagemorecosine-similar
to the original than duplicates are, because a value-flip (“8000” →“8080”) is a smaller edit than
a genuine rephrasing of the same fact. The distributions overlap so heavily that the maximum
precision achievable at any threshold is 0.67, far below what a safe automatic-update rule requires. A
learned classifier over similarity features does not rescue this in practice (our v6andv6_no_verify
conditions, Section 5): the gate-judge’s contradiction calls are unreliable, and in the abstention
regime theyleakstale facts 25–60% of the time.
The conclusion is that staleness detection must bestructural: if an incoming fact and a stored
fact share a (subject, relation) key but assert different objects, the newer one supersedes the older —
independent of how similar their embeddings are. This is the mechanism MemStrata implements.
4 The MemStrata Architecture
MemStrata is a local memory layer between an agent and its language model. It maintains a store
of facts extracted from interaction, and composes a token-budgeted context block per query. We
describe the components evaluated here.
4.1 Write path: deterministic supersession over a surprise gate
Each incoming turn yields a candidate fact. The write path routes it:
1.Exact-duplicate short-circuit.A normalized text hash drops verbatim repeats at zero cost.
2.Deterministic assertion path.If the turn expresses a clean (subject, relation, object)
triple — where the object is the single mutable value — MemStrata normalizes the (subject,
relation) key and checks for an active assertion with that key. If one exists with adifferent
object, the new assertionsupersedesit: the old row’s validity interval is closed ( valid_to
set,superseded_by linked) and the new row opened. Same object →duplicate (reinforce).
No prior key→novel (store).No cosine, no LLM judge.
3.Text-gate fallback.Non-triple prose falls through to a surprise gate that classifies via
similarityplusanLLMjudge. Critically(seeSection4.3), thisfallbackretainsnon-contradictory
near-duplicates asdistinctfacts; it drops only exact duplicates.
4.2 Bi-temporal ledger
Facts are retired, not deleted. The store records valid_from ,valid_to , and superseded_by , so
superseded facts remain available for future as-of-time queries (a capability we build on but do not
evaluate here; Section 7). Active retrieval surfaces only currently-valid rows.
4

4.3 The “retain, then supersede” design
An early variant of the temporal layer compressed aggressively, merging near-duplicate facts at write
time to bound growth. The clean evaluation shows thisregresses below RAG on static recall: merging
discards detail needed to answer later questions (the temporal_v6_lossy ablation, Section 5, drops
to 0.62 on project-fact QA and 0.13 on dialogue recall). The published configuration therefore
retains like RAG— storing distinct non-contradictory facts — and bounds growthonly on the
axis that matters, by superseding contradictions. This is the design choice that makes the system
match RAG on static knowledge while dominating on evolving knowledge. We report the lossy
variant as an ablation precisely because it isolates this decision.
4.4 Read path
Embed the query, retrieve top- kby cosine over active facts and assertions, apply the deterministic
staleness filter (drop superseded rows), and pack the surviving facts. We pack each surviving
assertion’soriginal source sentence, not the terse reconstructed triple: packing the triple alone
degrades the answer model on rich facts, while packing the original recovers accuracy. No LLM runs
on the read path, so retrieval latency sits at the embedding floor ( ∼2.1s), versus ∼16–18s for the
LLM-reranking and LLM-verification baselines.
4.5 Marker-free benchmark construction
Evaluating staleness resistance is easy to contaminate. If a stale fact carries any textual marker —
“[OUTDATED]”, “(legacy)”, “deprecated” — a retrieval baseline can disambiguate byreading the label
rather than by any temporal mechanism, silently inflating its score. We enforce a strict marker-free
invariant (by test): in every evolving benchmark, the stale and current versions of a fact are textually
identical except for the changed value, with no old/new/current framing. The only available signal
of currency is ingestion order, which only a temporal mechanism can exploit. We recommend this
invariant for any evaluation of memory under evolution; Section 5 shows that removing a marker from
an earlier benchmark dropped baseline accuracy by up to 14 points, confirming the contamination is
real and measurable.
5 Experiments
All experiments are local and deterministic: temperature 0, fixed seeds, no network (enforced by test).
Answer model Qwen2.5-Coder-7B; correctness and fabrication judges Qwen2.5-Coder-3B (distinct
from the answer model and from each other to prevent self-grading); embedder nomic-embed-text
(768-d).
Conditions (8). no_memory (floor), naive_rag (cosine top- k),advanced_rag (+ LLM reranker),
v6_no_verify (surprise gate, no LLM verify), v6(gate + LLM relevance verify), temporal_v6_lossy
(deterministic supersessionwithoutthe retain/original-text fixes — ablation), temporal_v6 (the full
method),v6+infer(gate + inferability pre-filter).
Benchmarks (6).Twostatic: domain(50 project-fact QA), locomo(30 multi-session dialogue
questions over 100 turns). Fourevolving, marker-free, 20–30 paired scenarios each: code_mutation
(function renames), config_migration (configuration value changes), dependency_bump (version
upgrades), api_evolution (endpoint/signature restructuring). Evolving scenarios ingest state-A
then state-B; the question targets the current value.
Metrics.Answer accuracy;stale-fact-error rate(fraction of contradiction questions answered
with the superseded value); conditional fabrication (fabrications per attempted answer, abstentions
5

excluded); active-fact count and compression; mean and p95 retrieval latency. We additionally run
aforced-answer supplementthat disables abstention on the RAG conditions, to expose the
stale-commitment that abstention otherwise hides.
5.1 Cosine cannot separate contradictions from duplicates
On 98 labeled pairs (32 duplicate, 22 merge, 22 contradict, 22 novel), cosine AUROC for separating
duplicates from the rest is0.5926. Per-class mean cosine:
labelnmean cos min max
duplicate 32 0.7998 0.4888 1.000
contradict 220.81190.5335 1.000
merge 22 0.9381 0.8842 0.9855
novel 22 0.4773 0.4077 0.6145
Table 1: Per-class cosine similarity to the original fact.
Contradictions (0.812) are more cosine-similar to the original than duplicates (0.800). The
maximum precision achievable at any duplicate threshold is0.667; the 0.95 floor a safe automatic
rule would need is unreachable. No similarity threshold can separate these classes — the empirical
foundation for a deterministic, structural supersession rule.
5.2 Accuracy: ties RAG on static, dominates on evolving
benchmarknaive_rag advanced_rag v6 temporal_v6_lossy temporal_v6
Static
domain 0.86 0.86 0.80 0.620.82
locomo 0.30 0.30 0.13 0.130.30
Evolving
code_mutation0.43 0.47 0.57 0.801.00
config_migration0.25 0.25 0.65 1.001.00
dependency_bump0.20 0.20 0.60 1.001.00
api_evolution0.40 0.40 0.35 0.950.95
Table 2: Answer accuracy.temporal_v6ties RAG on static and dominates on evolving.
Two reads. On thestaticbenchmarks, temporal_v6 ties RAG (0.82/0.30 vs 0.86/0.30) — the
retain-like-RAG design preserves recall, while the lossy ablation collapses (0.62/0.13), isolating
the cost of aggressive compression. On theevolvingbenchmarks, temporal_v6 reaches 0.95–1.00
versus RAG’s 0.20–0.47: a 2–5 ×accuracy improvement on the task class the method targets. The
LLM-verifier condition ( v6) is inconsistent and never reaches the temporal layer’s accuracy, at ∼8×
the latency.
5.3 Stale-fact error: the structural result
The headline. Fraction of contradiction questions answered with thesupersededvalue, in both the
abstention-allowed and forced-answer regimes:
6

evolving benchmarknaive_rag(allow / forced)temporal_v6(allow / forced)
code_mutation0.10 /0.400.03 /0.03
config_migration0.05 /0.350.00 /0.00
dependency_bump0.05 /0.150.00 /0.00
api_evolution0.30 /0.350.00 /0.00
Table 3: Stale-fact-error rate, abstention-allowed / forced-answer.
Allowed to abstain, RAGhidesits failure by refusing to answer (which is why its accuracy is
low). Forced to answer, it serves the stale value15–40%of the time. (Dependency bumps are the
exception at 15%, because “higher version number is newer” is a lucky surface heuristic — the model
is guessing from the string, not reasoning about currency.) MemStrata reaches ∼0%inbothregimes,
because the stale value is removed from the store before retrieval. This is the error RAG cannot
avoid by construction: it retrieves both values and has no mechanism to choose. The surprise-gate
conditions ( v6_no_verify ,v6) areworsethan RAG in the abstention regime — they answer but leak
stale 25–60%, consistent with Section 5.1’s finding that similarity-based supersession is unreliable.
Only deterministic supersession reaches∼0.
5.4 State-bounded growth
RAG is history-bounded: it stores every turn, growing without limit. MemStrata retains distinct
static facts (0% compression on domain/locomo— which iswhyit ties RAG on recall) but caps
growth on evolving facts via supersession ( ∼48%compression: code 48%, config 47.5%, dependency
50%, API 47.5%). The honest framing is not “smaller memory” but “stale facts retired”: growth is
bounded on the axis where unbounded growth is pathological (accumulating contradictory versions),
and unbounded where it should be (distinct facts).
5.5 Latency
temporal_v6 ,naive_rag ,and v6_no_verify allsitat ∼2.1s(noLLMonthereadpath). advanced_rag ,
v6, and v6+infer sit at∼16–18s (LLM rerank/verify). The win isaccuracy and temporal valid-
ity at RAG latency— temporal_v6 matches naive RAG’s speed while eliminating the stale-fact
errors naive RAG cannot. The LLM-based conditions pay8 ×the latency for no temporal ben-
efit. The per-benchmark figures are split by regime in Tables 4–5, which show the latency is
regime-independent:temporal_v6holds∼2.1s on both static and evolving.
6 Discussion
The results delineate a clean contribution and a clean boundary. RAG is the right tool for static
knowledge and remains competitive there; MemStrata matches it. Forevolvingknowledge, RAG
has a structural failure — it cannot maintain temporal validity, serving stale facts 15–40% of the
time when forced to commit — and MemStrata eliminates that failure at the same retrieval latency,
because the mechanism is a deterministic supersession rule rather than a similarity threshold or an
LLM call.
The negative results are as informative as the positive ones. The lossy ablation shows that
compression-for-its-own-sake costs accuracy on static recall — bounded growth is aconsequenceof
retiring stale facts, not a goal to pursue by merging distinct ones. The complementary full-retention
7

ablation closes the other direction: removing supersession entirely — retaining every turn RAG-style
— collapses mean accuracy across the four evolving benchmarks from 0.99 to 0.33, statistically
indistinguishable from naive RAG (0.32), and re-raises stale-fact error from zero (Appendix D, D.1b),
so the two ablations bracket the method — over-merging forfeits static recall, no-supersession forfeits
temporal validity — isolating deterministic supersession as the single cause of the evolving-knowledge
result. The LLM-verifier condition shows that a learned grounding check does not address staleness
(it has no temporal signal) and is not worth its latency. The surprise-gate conditions show that
similarity-based supersession actively leaks stale facts, confirming Section 5.1’s impossibility result
in the end-to-end system. Each of these is a path a reasonable designer might have taken; the data
closes them.
We also note, against a backdrop of recent work showing many retrieval gains shrink under honest
evaluation, that our marker-free protocol matters: an earlier benchmark with a textual staleness
marker inflated baseline accuracy by up to 14 points. Evaluations of memory under evolution that
do not enforce marker-freeness may be measuring a model’s ability to read a label rather than a
system’s ability to track currency.
7 Limitations
We state these plainly; they scope the claim and set up follow-on work.
•Structured, single-value benchmarks.Our evolving benchmarks are marker-free templates
with a single mutable value per fact, so the triple extractor keys reliably ( ∼97%supersession).
On a messier natural-language contradiction benchmark, extraction drops to ∼44%(multi-value
sentences, malformed perturbations); we quarantined that benchmark as a flawed ruler rather
than report it as a result.Extraction quality, not the supersession mechanism, is the
gating factor for unstructured contradictions, and is the explicit subject of follow-on
work (entity canonicalization, relation typing, multi-value extraction). The temporalledger
and the supersessionruleare domain-independent; the extraction layer must travel with them.
•Ingestion order proxies time.The benchmarks use order (state-A then state-B) as the
currency signal. Real temporal benchmarks carry explicit dates; the bi-temporal ledger already
stores validity intervals, and threading real valid_from timestamps plus an “as-of-T” retrieval
mode is future work, not new storage.
•Single-judge noise.The 3B correctness judge occasionally scores a gate-condition answer
“correct” while it contains the stale value, producing a few rows where accuracy and stale-error
overlap. The temporal layer’s∼0stale-error is unaffected (no stale fact is in its context).
•Scale and models.All results use a single 7B answer model on consumer hardware. Larger
modelsorcloudinferencemayshiftbaselines; weconstraintolocal-firstdeliberately. Benchmark
sizes (tens of items each) isolate mechanisms rather than rank systems on a leaderboard; scaling
to real-world longitudinal data is the subject of follow-on work.
8 Conclusion
For agents over evolving knowledge — codebases first among them — the binding memory failure
is not recall but currency: retrieval-augmented generation has no model of time and cannot tell a
stale fact from a current one, because the two are more embedding-similar than genuine duplicates
8

are. MemStrata maintains temporal validity through a deterministic (subject, relation, object)
supersession rule over a bi-temporal ledger: it stores like RAG, preserving static recall, and retires
contradicted facts before retrieval. The result is parity with RAG on static knowledge, 0.95–
1.00 accuracy on four evolving benchmarks where RAG reaches 0.20–0.47, and — the structural
contribution — a stale-fact-error rate of ∼0%where RAG serves the superseded value 15–40%
of the time, all at retrieval latency with no language model on the read path. We release the
harness, datasets, and protocol, and recommend the marker-free invariant for evaluating memory
under evolution. Coding is the wedge; the architecture is a general temporal-context memory, and
extending it to time-stamped world knowledge is the natural next step.
Reproducibility Statement
All experiments are deterministic (temperature 0, fixed seeds, no network, enforced by test). We
release the evaluation harness, extraction and gate prompts (with content hashes), all six benchmark
datasets (with hashes), the calibration dataset, and per-run logs. The marker-free invariant is
enforced by unit tests included in the release. Source data: REPORT_PAPER1.md (8×6 main ma-
trix), REPORT_PAPER1_forced.md (forced-answer supplement), calibration/REPORT_synthetic.md
(cosine calibration).
References
Akari Asai et al. Self-rag: Learning to retrieve, generate, and critique through self-reflection.arXiv
preprint arXiv:2310.11511, 2023. URLhttps://arxiv.org/abs/2310.11511. ICLR 2024.
Li Cai et al. A survey on temporal knowledge graph: Representation learning and applications.
arXiv preprint arXiv:2403.04782, 2024. URLhttps://arxiv.org/abs/2403.04782.
Prateek Chhikara et al. Mem0: Building production-ready ai agents with scalable long-term memory.
arXiv preprint arXiv:2504.19413, 2025. URLhttps://arxiv.org/abs/2504.19413.
Darren Edge et al. From local to global: A graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130, 2024. URLhttps://arxiv.org/abs/2404.16130.
Zirui Guo et al. Lightrag: Simple and fast retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2024. URLhttps://arxiv.org/abs/2410.05779.
Xinyu Han et al. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309, 2025. URLhttps://arxiv.org/abs/2501.00309.
ISO/IEC. Iso/iec 9075:2011, information technology — database languages — sql (sql:2011):
system-versioned and application-period (bi-temporal) tables. International Standard, 2011.
Christian S. Jensen and Richard T. Snodgrass. Temporal data management.IEEE Transactions on
Knowledge and Data Engineering, 11(1):36–44, 1999. doi: 10.1109/69.755615.
Carlos E. Jimenez et al. Swe-bench: Can language models resolve real-world github issues?arXiv
preprint arXiv:2310.06770, 2023. URLhttps://arxiv.org/abs/2310.06770. ICLR 2024.
Patrick Lewis et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. InAdvances
in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2005.
11401.
9

Zhihao Luo et al. Hypergraphrag: Retrieval-augmented generation via hypergraph-structured
knowledge representation.arXiv preprint arXiv:2503.21322, 2025. URL https://arxiv.org/
abs/2503.21322.
Adyasha Maharana et al. Evaluating very long-term conversational memory of llm agents.arXiv
preprint arXiv:2402.17753, 2024. URLhttps://arxiv.org/abs/2402.17753.
Charles Packer et al. Memgpt: Towards llms as operating systems.arXiv preprint arXiv:2310.08560,
2023. URLhttps://arxiv.org/abs/2310.08560.
Joon Sung Park et al. Generative agents: Interactive simulacra of human behavior. InACM CHI
Conference on Human Factors in Computing Systems, 2023. URL https://arxiv.org/abs/2304.
03442.
Richard Snodgrass and Ilsoo Ahn. A taxonomy of time in databases. InACM SIGMOD International
Conference on Management of Data, 1985. doi: 10.1145/318898.318921.
Richard T. Snodgrass.Developing Time-Oriented Database Applications in SQL. Morgan Kaufmann,
1999. URLhttps://www.cs.arizona.edu/people/rts/tdbbook.pdf.
Yifan Xu et al. Noderag: Structuring graph-based rag with heterogeneous nodes.arXiv preprint
arXiv:2504.11544, 2025. URLhttps://arxiv.org/abs/2504.11544.
Yuxuan Zeng et al. How significant are the real performance gains? an unbiased evaluation framework
for graphrag.arXiv preprint arXiv:2506.06331, 2025. URL https://arxiv.org/abs/2506.06331 .
A Full result tables
All tables in A.1–A.3 are reproduced verbatim from the committed source reports by eval/build_appendices.py
(no hand-transcription).
A.1 Main matrix — 8 conditions×6 benchmarks (REPORT_PAPER1.md)
Answer accuracy.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 0.000 0.860 0.860 0.860 0.800 0.620 0.820 0.800
locomo 0.000 0.300 0.300 0.167 0.133 0.133 0.300 0.133
code_mutation0.000 0.433 0.467 0.533 0.567 0.800 1.000 0.567
config_migration0.000 0.250 0.250 0.700 0.650 1.000 1.000 0.650
dependency_bump0.000 0.200 0.200 0.700 0.600 1.000 1.000 0.600
api_evolution0.000 0.400 0.400 0.400 0.350 0.950 0.950 0.350
Fabricated-context rate (raw).
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 0.000 0.040 0.040 0.080 0.020 0.000 0.040 0.020
locomo 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
code_mutation0.000 0.033 0.033 0.000 0.033 0.067 0.067 0.033
config_migration0.000 0.100 0.100 0.250 0.000 0.150 0.100 0.000
dependency_bump0.000 0.100 0.100 0.000 0.000 0.050 0.000 0.000
api_evolution0.000 0.150 0.100 0.050 0.050 0.000 0.000 0.050
10

Conditional fabrication rate (per attempted answer).
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 0.000 0.045 0.045 0.089 0.024 0.000 0.048 0.024
locomo 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
code_mutation0.000 0.048 0.048 0.000 0.033 0.077 0.067 0.033
config_migration0.000 0.333 0.333 0.263 0.000 0.150 0.100 0.000
dependency_bump0.000 0.250 0.250 0.000 0.000 0.050 0.000 0.000
api_evolution0.000 0.150 0.100 0.050 0.056 0.000 0.000 0.056
Attempted-answer count.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 0 44 44 45 42 35 42 42
locomo 0 17 14 10 9 10 17 9
code_mutation0 21 21 29 30 26 30 30
config_migration0 6 6 19 18 20 20 18
dependency_bump0 8 8 20 19 20 20 19
api_evolution0 20 20 20 18 20 20 18
Memory size (active facts).
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 47 50 50 47 47 46 50 47
locomo 16 100 100 16 16 28 100 16
code_mutation31 60 60 32 32 31 31 32
config_migration24 40 40 24 24 21 21 24
dependency_bump22 40 40 22 22 20 20 22
api_evolution25 40 40 25 25 21 21 25
Mean pack tokens.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 0.0 232.0 233.4 363.2 40.7 384.1 346.7 40.7
locomo 0.0 320.1 376.5 830.6 549.2 817.8 485.8 549.2
code_mutation0.0 163.3 163.4 260.5 32.9 236.2 250.7 32.9
config_migration0.0 130.1 130.1 193.6 26.9 187.2 192.2 26.9
dependency_bump0.0 110.0 110.0 174.3 31.1 175.9 150.2 31.1
api_evolution0.0 117.6 117.9 178.6 28.1 179.8 176.8 28.1
Stale-fact-error rate (of contradiction questions).Fraction of contradiction questions an-
swered with the SUPERSEDED value. RAG retrieves both the stale and current value and cannot
tell which is current; deterministic supersession removes the stale one, driving this to ∼0. The error
RAG cannot avoid by construction.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
code_mutation0.000 0.100 0.100 0.600 0.567 0.000 0.033 0.567
config_migration0.000 0.050 0.050 0.250 0.250 0.000 0.000 0.250
dependency_bump0.000 0.050 0.050 0.300 0.300 0.000 0.000 0.300
api_evolution0.000 0.300 0.300 0.500 0.350 0.000 0.000 0.350
11

Memory compression (% of turns absorbed).Derived:100 (1 −active_facts/turns_ingested ).
Higher = more bounded growth. naive/advanced RAG ∼0%(one fact per turn); the gate and
temporal layer compress.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
domain 6.0 0.0 0.0 6.0 6.0 8.0 0.0 6.0
locomo 84.0 0.0 0.0 84.0 84.0 72.0 0.0 84.0
code_mutation48.3 0.0 0.0 46.7 46.7 48.3 48.3 46.7
config_migration40.0 0.0 0.0 40.0 40.0 47.5 47.5 40.0
dependency_bump45.0 0.0 0.0 45.0 45.0 50.0 50.0 45.0
api_evolution37.5 0.0 0.0 37.5 37.5 47.5 47.5 37.5
Retrieval latency, split by regime.The two 8 ×6 latency matrices (mean and p95) are split into
astatictable and anevolvingtable, each carrying both metrics, to make the regime-independence
oftemporal_v6 ’s latency visually explicit: it holds ∼2.1s on both static and evolving knowledge,
while the LLM-rerank/verify conditions cost∼16–24s throughout.
Table 4: Retrieval latency (ms) —staticbenchmarks.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
Mean retrieval latency
domain 0.0 2103.4 17985.8 2103.7 16997.0 2107.0 2111.4 17087.9
locomo 0.0 2131.8 18251.7 2102.7 17309.4 2099.4 2133.7 17292.4
P95 retrieval latency
domain 0.0 2120.0 18400.1 2122.3 23895.5 2125.5 2132.0 23947.3
locomo 0.0 2151.4 18653.2 2111.8 20638.9 2113.4 2150.2 20503.0
Table 5: Retrieval latency (ms) —evolvingbenchmarks.
benchmarkno_memory naive_rag advanced_rag v6_no_verify v6 temporal_v6_lossy temporal_v6 v6+infer
Mean retrieval latency
code_mutation0.0 2134.7 17907.9 2100.6 15835.9 2109.4 2110.3 15965.3
config_migration0.0 2119.4 18021.7 2116.1 15509.6 2107.4 2111.6 15573.0
dependency_bump0.0 2110.8 18088.9 2109.6 15969.5 2097.5 2099.4 15926.8
api_evolution0.0 2111.3 18220.7 2106.4 15904.4 2107.6 2107.4 16088.6
P95 retrieval latency
code_mutation0.0 2172.7 18035.2 2125.9 18765.6 2121.2 2125.7 18903.5
config_migration0.0 2124.5 18240.5 2142.5 16596.4 2118.7 2124.3 16536.8
dependency_bump0.0 2126.1 18299.3 2126.2 21181.8 2107.7 2107.8 21055.6
api_evolution0.0 2129.1 18372.9 2116.7 18910.5 2136.5 2113.0 18953.1
A.2Forced-answer supplement — no-abstention, 3 conditions ×4 evolving
benchmarks (REPORT_PAPER1_forced.md)
Answer accuracy.
benchmarknaive_rag advanced_rag temporal_v6
code_mutation0.633 0.633 0.933
config_migration0.650 0.650 1.000
dependency_bump0.850 0.850 1.000
api_evolution0.650 0.600 0.950
12

Stale-fact-error rate (of contradiction questions).
benchmarknaive_rag advanced_rag temporal_v6
code_mutation0.400 0.400 0.033
config_migration0.350 0.350 0.000
dependency_bump0.150 0.150 0.000
api_evolution0.350 0.400 0.000
Conditional fabrication rate (per attempted answer).
benchmarknaive_rag advanced_rag temporal_v6
code_mutation0.567 0.567 0.200
config_migration0.950 0.950 0.300
dependency_bump0.650 0.650 0.100
api_evolution0.450 0.550 0.000
Mean retrieval latency (ms).
benchmarknaive_rag advanced_rag temporal_v6
code_mutation2114.5 17806.5 2094.3
config_migration2093.4 17592.2 2084.4
dependency_bump2092.8 17794.0 2088.0
api_evolution2102.1 17885.8 2087.6
A.3Cosinecalibration—grader-independent( calibration/REPORT_synthetic.md )
Cosine distribution by label.
labelnmean median min max
duplicate 32 0.7998 0.7854 0.4888 1.0
merge 22 0.9381 0.9426 0.8842 0.9855
contradict 22 0.8119 0.823 0.5335 1.0
novel 22 0.4773 0.4635 0.4077 0.6145
τdupsweep (DUPLICATE auto-accept).
τdupn_predicted precision recall
0.80 47 0.2766 0.4062
0.81 46 0.2609 0.3750
0.82 45 0.2667 0.3750
0.83 43 0.2558 0.3438
0.84 41 0.2439 0.3125
0.85 41 0.2439 0.3125
0.86 40 0.2500 0.3125
0.87 40 0.2500 0.3125
13

τdupn_predicted precision recall
0.88 39 0.2564 0.3125
0.89 36 0.2778 0.3125
0.90 35 0.2857 0.3125
0.91 34 0.2941 0.3125
0.92 32 0.3125 0.3125
0.93 28 0.3571 0.3125
0.94 25 0.4000 0.3125
0.95 23 0.4348 0.3125
0.96 21 0.4762 0.3125
0.97 17 0.5882 0.3125
0.98 15 0.6667 0.3125(←rec)
τnovelsweep (skip-judge floor).
τnoveln_below false_novel_rate near_band_share
0.50 18 0.0132 0.6633
0.51 18 0.0132 0.6633
0.52 19 0.0132 0.6531
0.53 19 0.0132 0.6531
0.54 21 0.0263 0.6327
0.55 21 0.0263 0.6327
0.56 21 0.0263 0.6327
0.57 22 0.0263 0.6224
0.58 23 0.0395 0.6122
0.59 23 0.0395 0.6122(←rec)
0.60 26 0.0658 0.5816
0.61 26 0.0658 0.5816
0.62 27 0.0658 0.5714
0.63 28 0.0789 0.5612
0.64 29 0.0921 0.5510
0.65 31 0.1184 0.5306
0.66 32 0.1316 0.5204
0.67 33 0.1447 0.5102
0.68 34 0.1579 0.5000
0.69 34 0.1579 0.5000
0.70 37 0.1974 0.4694
0.71 38 0.2105 0.4592
0.72 38 0.2105 0.4592
0.73 39 0.2237 0.4490
0.74 40 0.2368 0.4388
0.75 41 0.2500 0.4286
0.76 43 0.2763 0.4082
0.77 44 0.2895 0.3980
0.78 46 0.3158 0.3776
0.79 51 0.3816 0.3265
14

τnoveln_below false_novel_rate near_band_share
0.80 51 0.3816 0.3265
0.81 52 0.3947 0.3163
0.82 53 0.4079 0.3061
0.83 55 0.4342 0.2857
0.84 57 0.4605 0.2653
0.85 57 0.4605 0.2653
B Benchmark construction
B.1 Marker-free invariant (enforced by test).In every evolving benchmark a scenario is astate-
Aturn and astate-Bturn that are textually identical except for the single mutated value, followed by
a question whose gold answer is the state-B value. The wordsold, new, current, previous, deprecated,
legacy, outdatedand synonyms never appear in either turn; the only currency signal is ingestion order.
Enforcedby tests/memory/test_evolving_benchmarks.py::test_guard_rejects_staleness_tell
andtheword-boundarytell-detector tests/memory/test_swe_longitudinal.py::test_has_tell_is_word_boundary_aware
/swe_longitudinal_benchmark.assert_marker_free.
B.2 Why marker-freeness is necessary.Removing an explicit [OUTDATED] marker from
an earlier contradiction benchmark dropped reranker-RAG accuracy by 14 points and a gate-only
baseline by 18 points while the temporal method moved only −4— the marker was a confound
baselines read off the text. We treat marker-freeness as a correctness property of the evaluation.
B.3 code_mutation (function renames / endpoint moves / config / imports / deps). Total
scenarios: 30. First 3:
state-A : The function get_user_by_id(uid) looks up a user record by primary key in the users
table.
state-B : The function fetch_user(uid) looks up a user record by primary key in the users table.
(identical except the value)
question: What function looks up a user record by primary key?
gold : fetch_user
state-A : The function load_config() returns the parsed configuration as a dict from settings.yaml
.
state-B : The function read_settings() returns the parsed configuration as a dict from settings.
yaml.
question: What function returns the parsed configuration from settings.yaml?
gold : read_settings
state-A : The function process_payment(amount) charges the customer’s card via the Stripe API.
state-B : The function charge_card(amount) charges the customer’s card via the Stripe API.
question: What function charges the customer’s card via the Stripe API?
gold : charge_card
B.4config_migration(configuration value changes). Total scenarios: 20. First 3:
state-A : The SESSION_TIMEOUT setting is 1800 seconds in config.py.
state-B : The SESSION_TIMEOUT setting is 3600 seconds in config.py.
question: What is the SESSION_TIMEOUT in seconds in config.py?
gold : 3600
state-A : The MAX_UPLOAD_SIZE is set to 10485760 bytes in settings.py.
state-B : The MAX_UPLOAD_SIZE is set to 52428800 bytes in settings.py.
15

question: What is the MAX_UPLOAD_SIZE in bytes in settings.py?
gold : 52428800
state-A : The database connection pool size is 10 in the production config.
state-B : The database connection pool size is 25 in the production config.
question: What is the database connection pool size in the production config?
gold : 25
B.5dependency_bump(version upgrades). Total scenarios: 20. First 3:
state-A : The project pins numpy==1.24.0 in requirements.txt.
state-B : The project pins numpy==1.26.4 in requirements.txt.
question: What version of numpy does the project pin in requirements.txt?
gold : 1.26.4
state-A : The project pins django==4.1.7 in requirements.txt.
state-B : The project pins django==5.0.2 in requirements.txt.
question: What version of django does the project pin in requirements.txt?
gold : 5.0.2
state-A : The setup.py sets python_requires to >=3.8.
state-B : The setup.py sets python_requires to >=3.11.
question: What python_requires minimum does setup.py set?
gold : 3.11
B.6 api_evolution (endpoint / parameter / signature restructuring). Total scenarios: 20. First
3:
state-A : The user list endpoint is GET /api/v1/users.
state-B : The user list endpoint is GET /api/v2/users.
question: What is the path of the user list endpoint?
gold : /api/v2/users
state-A : The search endpoint filters by the parameter named ’category’.
state-B : The search endpoint filters by the parameter named ’tag’.
question: What parameter does the search endpoint filter by?
gold : tag
state-A : The auth login route is POST /auth/login.
state-B : The auth login route is POST /auth/sessions.
question: What is the path of the auth login route?
gold : /auth/sessions
B.7 Static benchmarks. domain— 50 project-fact QA questions over real project facts (ports,
model names, config flags, thresholds, decision records) with no contradictions; measures recall
preservation. locomo— 30 questions over a 100-turn multi-session dialogue (capped LoCoMo sample)
with no contradictions; measures detail recall under compression pressure.
B.8 Quarantined benchmark (the flawed ruler). fair_contradiction embeds the mutated
fact in free prose. The triple extractor keys reliably on only ∼44%of scenarios (multi-value sentences
and-altstringperturbationsdefeatsingle-slotextraction—measuredby eval/diag_extract_probe.py :
97% clean supersession on code_mutation vs 44% here), so it measuresextraction quality, not the
supersession mechanism. On it, temporal_v6 scores0.62vs advanced_rag 0.74(it never engages
on the 56% of pairs it cannot extract). We exclude it from the main results and report it here as the
honest boundary of the method and the motivation for Section 7’s extraction-robustness work.
16

C Prompts and content hashes
All read-path-relevant prompts, with full SHA-256. Answer model, both judges, and the verifier
are distinct assignments (answer ̸=correctness-judge ̸=fabrication-judge ̸=verifier) to preclude
self-grading; all calls run at temperature 0, fixed seed.
prompt file SHA-256 (full)
consolidate_v1.mdd2a2f0bfd0bc27ebdb1c6676cc00e03f9cfd741edffce4f4b3faa1ee9618b23e
extract_triple_v1.md90967d66c205a6e7856bb90907cde05acad90a0ec8045b275287d8dd3cb3cec4
extract_triples_v1.md2bc9b62821d9cc3890138e04cd34594d0ca44008a04fd5ada31ed93193b3947e
extract_v1.mdd71afa291d8a05eeed039dce2b33275e021e49d84b935c31b107644ae5c0af26
gate_judge_v1.md7dd4149dc7c893145d2d025672ce4daf271e5d6f6db2457c217d3086e94acca5
infer_check_v1.mdd5f50b3447808c71194c4263a8cf73a50954df6cfeef34147e942f6c9709a5c3
rerank_v1.md18d8318d98fdd56cbb2c0c730bcead1a49b885204816acf0436480d3822c5224
verify_v1.md691456512522b15bd8cd364262ff09a3477e4dd161158c1d665c5ad5c73c198c
C.1 Deterministic triple extractor(the supersession key source).extract_triple_v1.md.
---
prompt: extract_triple
version: 1
---
You convert ONE factual statement about a codebase into a (subject, relation,
object) triple, IF and only if it states a single concrete value that could
change as the code evolves (a function name, API endpoint, config value, port,
version, import path, identifier).
Return ONLY a JSON object - no prose, no markdown fences:
{"is_triple": true,
"subject": "<the stable thing the value belongs to - MUST NOT contain the value>",
"relation": "<a short linking phrase, e.g. is / is named / is set to / is imported from>",
"object": "<the one concrete value that would change if the code evolved>"}
or, when the statement is not a single value-bearing fact (a preference, a
decision, prose, or it carries several values):
{"is_triple": false}
CRITICAL RULES:
- The ‘object‘ is the ONE value that would differ between an old and a new
version of this fact (the name / number / version / path / endpoint).
- The ‘subject‘ describes WHAT that value belongs to and MUST NOT contain the
object value. Two statements that differ only in the value must produce the
SAME subject and relation, so the system can detect the change.
- Rephrase the subject as "the <thing> that <does what>" when the changing value
is an identifier (a function/endpoint/module name).
- Keep subject and relation deterministic and minimal. Do not add commentary.
EXAMPLES:
Input: The function get_user_by_id(uid) looks up a user record by primary key in the users table.
Output: {"is_triple": true, "subject": "the function that looks up a user record by primary key in
the users table", "relation": "is named", "object": "get_user_by_id"}
Input: The SESSION_TIMEOUT setting is 1800 seconds in this project’s config.toml.
Output: {"is_triple": true, "subject": "the SESSION_TIMEOUT setting in config.toml", "relation": "
is", "object": "1800 seconds"}
17

Input: We decided to prefer Python over Go for internal tooling.
Output: {"is_triple": false}
STATEMENT:
"""
{statement}
"""
C.2 Multi-value triple extractor(P1.3; flag-gated).extract_triples_v1.md.
---
prompt: extract_triples
version: 1
---
You convert ONE factual statement into a list of (subject, relation, object)
triples - ONE triple for EACH concrete value in the statement that could change
as the system evolves. A statement may carry one value, several values, or none.
Return ONLY a JSON object - no prose, no markdown fences:
{"triples": [
{"subject": "<the stable thing the value belongs to - MUST NOT contain the value>",
"relation": "<a short linking phrase>",
"object": "<one concrete value that would change if the system evolved>"},
...
]}
When the statement carries no single value-bearing fact, return {"triples": []}.
EXAMPLES:
Input: The harness proxy listens on port 8080 and the admin API listens on port 9090.
Output: {"triples": [{"subject": "the port the harness proxy listens on", "relation": "is", "
object": "8080"}, {"subject": "the port the admin API listens on", "relation": "is", "object":
"9090"}]}
Input: We decided to prefer Python over Go for internal tooling.
Output: {"triples": []}
STATEMENT:
"""
{statement}
"""
C.3Surprise-gatejudge(text-gatefallbackonly—neverontheassertionpath). gate_judge_v1.md .
---
prompt: gate_judge
version: 1
---
You decide how a CANDIDATE memory fact relates to the EXISTING facts it most
resembles. Return ONLY a JSON object - no prose, no markdown fences:
{"verdict": "duplicate|merge|contradict|novel",
"reason": "<one short line>",
"merged_text": "<required only when verdict is ’merge’>"}
Definitions:
- duplicate - the candidate says the same thing as an existing fact.
18

- merge - the candidate adds detail; provide merged_text (lossless).
- contradict - the candidate conflicts (world changed / decision reversed);
the old fact is superseded and the candidate stored as current truth.
- novel - genuinely new; none of the existing facts cover it.
Choose the single best verdict. When unsure between merge and novel, prefer novel.
CANDIDATE:
{candidate}
TOP EXISTING MATCHES:
{matches}
C.4LLMrelevanceverifier(the v6baseline; neveron temporal_v6 ’sreadpath). verify_v1.md .
---
prompt: verify
version: 1
---
You are a strict relevance verifier. Given a user QUERY, the project’s LOCKED
RULES, and a numbered list of CANDIDATE memories, decide for EACH candidate
whether it should be shown to the assistant answering the query.
Return ONLY a JSON object - no prose, no markdown fences:
{"verdicts": [
{"n": <candidate number>,
"verdict": "SUPPORTED|IRRELEVANT|CONFLICTS",
"justification": "<= 15 words, MUST quote a verbatim span from the candidate"}
]}
Rules:
- SUPPORTED - relevant to THIS query and consistent with the locked rules.
- IRRELEVANT - does not help answer this query.
- CONFLICTS - contradicts the locked rules.
- The justification MUST contain a span copied verbatim from the candidate text.
- Judge only relevance and consistency - do not invent facts.
QUERY:
{query}
LOCKED RULES (invariant memory):
{invariant}
CANDIDATES:
{candidates}
The correctness-judge and fabrication-judge prompts are inline in eval/run_matrix.py (_correctness_fn ,
_fabrication_fn ); the answer-model prompt (with the abstention / forced-answer variants) is in
_answer_fn.
D Ablations
D.1 retain-vs-lossy (the State-Bounded Temporal Validity isolation). temporal_v6 vs
temporal_v6_lossy (deterministicsupersessionWITHvsWITHOUTtheretain-like-RAG+original-
text-packing fixes), from the A.1 accuracy table:
19

benchmarktemporal_v6_lossy temporal_v6
domain (static) 0.620.82
locomo (static) 0.130.30
code_mutation0.801.00
config_migration1.00 1.00
dependency_bump1.00 1.00
api_evolution0.95 0.95
The lossy variant merges non-contradictory near-duplicates at write time and collapses on static
recall (0.62/0.13); the full method retains them and ties RAG (0.82/0.30), at parity on the evolving
four. Bounded growth is aconsequenceof retiring stale facts, not a goal pursued by merging distinct
ones.
D.1b full-retention — the opposite bracket (supersession is the isolated cause).D.1
removes the retain/packingrefinementsbut keeps supersession; this ablation removessupersession
itself. Toggling retain_all_turns makes the write path non-lossy — every turn is stored as
a distinct fact (only exact byte-duplicates dropped), with no (S,R,O) supersession — while the
extractor, multi-value handling, and read path are otherwise unchanged. The ledger degenerates to
a retain-everything store, and the read path must choose among co-present stale and current values:
benchmark acc: supersede acc: retain RAG acc fab: supersede fab: retain
code_mutation1.000.43 0.43 0.07 0.09
config_migration1.000.35 0.25 0.10 0.56
dependency_bump1.000.15 0.20 0.00 0.25
api_evolution0.950.40 0.40 0.00 0.10
mean 0.99 0.33 0.32 0.04 0.25
Removing supersession collapses mean evolving accuracy from 0.99 to 0.33 — statistically
indistinguishable from naive_rag (0.32) — and re-raises stale-fact error from 0.00 to 0.05–0.25 (the
read path now serves the superseded value, which deterministic supersession had retired). It also
raises conditional fabrication ineverybenchmark — mean0.04 →0.25 (∼6×), peaking at 0.56
onconfig_migration — because the model, now seeing the stale and current values side by side
and unable to tell which holds, invents an answer. Retain-everything memory is thus not merely
less accurate butless safe: it manufactures a fabrication source that bounded, supersession-based
growth eliminates. The two ablations bracket the design from opposite sides: D.1 (over-merging)
forfeitsstaticrecall; D.1b (no supersession) forfeitstemporal validityandsafety; temporal_v6 sits at
the optimum between them. This isolates deterministic supersession — not retention, original-text
packing, or bounded growth, which are refinements — as the single mechanism responsible for
the evolving-knowledge result. (Same models, temperature 0, seed 0; retain_all_turns is an
ablation-only flag, default off, write path otherwise frozen.)
D.2 original-text packing.Validated jointly with D.1 (both fixes ship in temporal_v6 ; the
lossy ablation isolates their combined effect on static recall, and code_mutation 0.80→1.00reflects
packing the original sentence rather than the terse triple). A dedicated single-factor packing cell was
not run separately and is marked future work — we do not imply a measurement we did not take.
D.3 LLM-verifier (non-)contribution. v6(gate + LLM relevance verify) vs v6_no_verify
(gate only), from A.1: v6≤v6_no_verify on the static/recall tasks (domain 0.80 vs 0.86; locomo
0.13 vs 0.17) at ∼8×latency ( ≈16–18s vs ≈2.1s, A.1 latency tables). The learned relevance check
has no temporal signal and is not worth its cost.
20

D.4 +INFER (non-)contribution. v6+infer≈v6 on every benchmark (A.1) after the
single-candidate tautology-guard fix — reported for completeness; neutral everywhere.
21