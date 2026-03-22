# CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering

**Authors**: Tianyi Huang, Ying Kai Deng

**Published**: 2026-03-17 03:27:25

**PDF URL**: [https://arxiv.org/pdf/2603.16091v1](https://arxiv.org/pdf/2603.16091v1)

## Abstract
In factual question answering, many errors are not failures of access but failures of commitment: the system retrieves relevant evidence, yet still settles on the wrong answer. We present CounterRefine, a lightweight inference-time repair layer for retrieval-grounded question answering. CounterRefine first produces a short answer from retrieved evidence, then gathers additional support and conflicting evidence with follow-up queries conditioned on that draft answer, and finally applies a restricted refinement step that outputs either KEEP or REVISE, with proposed revisions accepted only if they pass deterministic validation. In effect, CounterRefine turns retrieval into a mechanism for testing a provisional answer rather than merely collecting more context. On the full SimpleQA benchmark, CounterRefine improves a matched GPT-5 Baseline-RAG by 5.8 points and reaches a 73.1 percent correct rate, while exceeding the reported one-shot GPT-5.4 score by roughly 40 points. These findings suggest a simple but important direction for knowledgeable foundation models: beyond accessing evidence, they should also be able to use that evidence to reconsider and, when necessary, repair their own answers.

## Full Text


<!-- PDF content starts -->

CounterRefine: Answer-Conditioned Counterevidence Retrieval for
Inference-Time Knowledge Repair in Factual Question Answering
Tianyi Huang*
Ryquo
tianyi@ryquo.comYing Kai Deng
App-In Club
kai.deng@appinclub.org
Abstract
In factual question answering, many errors are
not failures of access but failures of commit-
ment: the system retrieves relevant evidence,
yet still settles on the wrong answer. We present
COUNTERREFINE, a lightweight inference-
time repair layer for retrieval-grounded QA.
COUNTERREFINEfirst produces a short an-
swer from retrieved evidence, then gathers ad-
ditional support and conflicting evidence with
follow-up queries conditioned on that draft an-
swer, and finally applies a restricted refine-
ment step that outputs either KEEP orREVISE ,
with proposed revisions accepted only if they
pass deterministic validation. In effect, COUN-
TERREFINEturns retrieval into a mechanism
for testing a provisional answer rather than
merely collecting more context. On the full
SimpleQA benchmark, COUNTERREFINEim-
proves a matched GPT-5 BASELINE-RAG by
5.8 points and reaches a 73.1% correct rate,
while exceeding the reported one-shot GPT-5.4
score by roughly 40 points. These findings sug-
gest a simple but important direction for knowl-
edgeable foundation models: beyond accessing
evidence, they should also be able to use that
evidence to reconsider and, when necessary,
repair their own answers.
1 Introduction
Large language models are increasingly used as
knowledge interfaces, but a central difficulty re-
mains unresolved: a model can retrieve relevant
material and still commit to the wrong fact. In
short-form factual QA, these errors are unforgiving.
A wrong year, nearby entity, or almost-right title
is still fully incorrect. This makes factual QA a
useful setting for studying a broader question that
matters for knowledgeable foundation models: if
some knowledge is wrong, can we repair it at in-
ference time rather than only during pre-training or
parameter editing?
*Corresponding author.Retrieval-augmented generation (RAG) im-
proves factuality by grounding generation in ex-
ternal evidence rather than relying only on para-
metric memory (Lewis et al., 2021; Karpukhin
et al., 2020). Yet retrieval alone does not eliminate
candidate-selection errors. A first-pass retriever is
optimized fortopic relevance, not necessarily for
candidate discrimination. However, once a draft
answer exists, the retrieval problem changes. The
most useful next query is often not the original
question, but the question conditioned on that can-
didate answer. If the draft year is wrong, adding
that year to the query can yield a snippet that di-
rectly falsifies it. If the draft entity is too broad
or ambiguous, answer-conditioned re-querying can
reveal a more discriminative title, list entry, or sen-
tence.
We build on this observation with COUNTER-
REFINE, a retrieval-based short-answer system that
first drafts an answer from web snippets and then
performs one answer-conditioned counterevidence
pass. The second stage is intentionally narrow. It is
not tasked with solving the question from scratch.
Instead, it decides whether to KEEP the draft or
REVISE it in light of additional evidence, and a
deterministic validator blocks unsupported or ill-
formed rewrites. The result is a simple repair layer
that can sit on top of an existing retrieval pipeline.
Our empirical evidence is centered on full-
benchmark official evaluation. On the full 4,326-
question SimpleQA benchmark with the official
grader (Wei et al., 2024; OpenAI, 2025), COUN-
TERREFINEimproves a matched BASELINE-RAG
from 63.7% to 67.7% correct and from 64.1% to
68.1% F1 on Claude Sonnet 4.6. On GPT-5, cor-
rect rate improves from 67.3% to 73.1% and F1
from 58.6% to 72.1%. We also evaluate transfer
on 100- and 300-example slices from HotpotQA
distractor development (Yang et al., 2018), where
COUNTERREFINEimproves exact match on both
Claude Sonnet 4.6 and GPT-5. Qualitative exami-arXiv:2603.16091v1  [cs.CL]  17 Mar 2026

nation shows that most successful revisions repair
entity confusions, dates, and numeric values, while
the remaining failures are dominated by relation
confusion and event mismatch.
Our contributions are: (1) we formalize answer-
conditioned counterevidence retrieval as a sim-
ple keep-or-revise layer for factual QA; (2) we
show that this layer improves over a matched re-
trieval baseline under the official SimpleQA evalu-
ation pipeline and also transfers to HotpotQA exact
match; and (3) we provide a targeted qualitative
analysis of where the method helps and where its
remaining failure modes arise.
2 Related Work
Retrieval-grounded factual QA.Retrieval-
augmented generation improves knowledge-
intensive NLP by grounding generation in external
documents (Lewis et al., 2021). Dense Passage
Retrieval further improved open-domain QA with
learned neural retrieval (Karpukhin et al., 2020).
More recent work has explored adaptive retrieval
and critique, for example by teaching models when
to retrieve and how to reflect on retrieved evidence
(Asai et al., 2024). Other work has targeted
robustness to noisy retrieval (Yu et al., 2024) or
refreshed model knowledge through search-engine
augmentation (Vu et al., 2024). Our method differs
in focus: rather than redesigning the retriever or
the full RAG stack, we add a minimal second
retrieval phase to stress-test the first answer.
Verification and self-correction.A broad line of
work improves factuality by checking or revising
an initial answer. Chain-of-Verification generates
verification questions before producing a final an-
swer (Dhuliawala et al., 2024). CRITIC uses tool-
interactive critique to validate and amend outputs
(Gou et al., 2024). RARR retrieves evidence and
revises unsupported claims while trying to preserve
the original text (Gao et al., 2023). SelfCheckGPT
detects likely hallucinations through disagreement
across samples (Manakul et al., 2023). COUNTER-
REFINEis closest in spirit to this family, but it is
narrower and cheaper: one extra retrieval pass, one
extra model call, and a constrained short-answer
decision rather than broad long-form editing.
Knowledge repair and model editing.Wrong
knowledge can also be addressed by editing model
parameters directly. ROME and MEMIT show
that factual associations in language models can bemodified through targeted weight updates (Meng
et al., 2022, 2023). Those methods operate on
the model’s stored knowledge itself. COUNTER-
REFINEis complementary: it performs answer-
level repair at inference time instead of changing
model parameters, using external evidence to chal-
lenge and revise a candidate answer.
Benchmarks and conflicting evidence.Factu-
ality has been studied through claim verification
datasets such as FEVER (Thorne et al., 2018),
factual QA benchmarks such as TruthfulQA (Lin
et al., 2022), long-form factuality metrics such as
FACTSCORE (Min et al., 2023), and broader sur-
veys of LLM factuality (Wang et al., 2024). Sim-
pleQA is particularly relevant here because it iso-
lates short-form factual precision in a benchmark of
4,326 questions with a single indisputable answer
(Wei et al., 2024). Recent work has also empha-
sized that retrieval errors are not limited to irrel-
evant documents; systems often face ambiguous,
noisy, or conflicting evidence (Wang et al., 2025).
Our setting is simpler than that literature’s multi-
document conflict benchmarks, but the motivating
issue is related: even in short-answer QA, retrieved
snippets can weakly support multiple nearby can-
didates. Answer-conditioned re-querying is one
lightweight way to make those conflicts visible be-
fore finalizing an answer.
3 Method
3.1 Overview
We refer to the first-pass retrieval system as
BASELINE-RAG. Given a question q, COUNTER-
REFINEfirst runs BASELINE-RAG to produce an
initial answer a0, then runs one answer-conditioned
refinement stage:
R0= Retrieve(q),(1)
a0=f draft(q, R 0),(2)
R1= RetrieveRefine(q, a 0, R0),(3)
a⋆=f refine(q, a 0, R1).(4)
The goal is not to replace a standard retrieval
pipeline with a more complicated agent, but to add
a focused reliability layer on top of it.
3.2 Stage 1: retrieval-based drafting
The baseline stage retrieves up to kbevidence snip-
pets for the original question:
R0= Retrieve(q, k b).(5)

Figure 1: Schematic overview of COUNTERREFINE. Starting from a draft answer produced by BASELINE-RAG,
CounterRefine collects additional evidence intended to support, check, or contradict that candidate, evaluates
the draft answer against the resulting evidence set, and then passes the result to a constrained keep-vs.-revise
gate. The final answer is either the original draft or a revised candidate accepted under evidence-based validation
using the same evidence set. For readability, the figure abstracts away some implementation details from the text
(question-type-conditioned query selection and per-query snippet aggregation) but it is aligned with the three-stage
pipeline described in Section 3.
In the current implementation, kb= 5. The re-
triever returns tuples of title, URL, and evidence
text, and duplicates are removed before prompt
construction.
The drafting model answers using only the re-
trieved evidence:
a0=f draft(q, R 0).(6)
The draft prompt enforces a short answer and dis-
allows hedging or explanation. If retrieval fails
to yield a valid answer, the code falls back to a
closed-book short-answer prompt; if that also fails,
a minimal type-dependent default is emitted.
3.3 Stage 2: answer-conditioned
counterevidence retrieval
The central design choice is that refinement queries
depend on the draft answer. Let t(q) denote a
coarse question type inferred heuristically from
the question string. The refinement query set is
Q(q, a 0) ={q, q∥"a 0"} ∪I[t(q)∈ T]{"a 0"},
(7)
where∥denotes string concatenation and
T={who,where,when,year,number}.
Intuitively, the second retrieval stage asks not just
“what documents are about this question?” but
“what evidence most directly supports or contra-
dicts this candidate answer?”
For each q′∈Q(q, a 0), the system retrieves up
tokrevidence snippets, with kr= 5in the current
implementation, and merges them with the baselinesupport set:
R1= Dedupe
R0∪[
q′∈Q(q,a 0)Retrieve(q′, kr)
.
(8)
3.4 Stage 3: constrained refinement
The refiner receives the question, the baseline an-
swer, and the merged evidence set, and must output
exactly three fields:
DECISION: KEEP or REVISE
ANSWER: <short answer only>
EVIDENCE: <one short evidence snippet or
NONE>
The prompt instructs the model to KEEP when the
baseline is already the best supported answer and to
REVISE only when the additional evidence strongly
supports a different or more exact short answer.
Let(d,ˆa, e) denote the parsed refinement output.
Ifd=KEEP , the system returns a0. Ifd=REVISE ,
the proposed answer must pass deterministic vali-
dation before it is accepted.
3.5 Deterministic validation and
canonicalization
The validator is a core part of the method. It blocks
revisions that are unsupported, ill-typed, or stylisti-
cally incompatible with short-answer evaluation. A
proposed rewrite is rejected if any of the following
hold:
1.it is empty, non-responsive, or identical to the
draft after normalization;
2.for yes/no questions, it is not exactly yesorno;

3.for entity-style questions, it is clause-like, overly
long, or begins with a descriptor phrase;
4.for temporal or numeric questions, it lacks an
explicit temporal or numeric marker;
5. the refiner provides no evidence snippet;
6. lexical overlap between the revised answer and
the evidence snippet is too weak.
Accepted revisions are then canonicalized with
question-type-specific rules, such as extracting a
4-digit year, compacting number spans, stripping
leading prepositions for locations, or removing par-
enthetical descriptors from names. Only proposed
revisions are validated; KEEP decisions preserve the
baseline answer unchanged. The final output is
a⋆=(
Canon(q,ˆa),if the revision is accepted,
a0,otherwise.
(9)
4 Experimental Setup
4.1 Benchmarks and metrics
We evaluate on two benchmarks. Our primary
benchmark is SimpleQA, which contains 4,326
short, fact-seeking questions with a single in-
disputable answer (Wei et al., 2024). We use
the official SimpleQA grader from the public
simple-evals implementation (OpenAI, 2025).
Following the official setup, we report the correct
rate and F1 score.
As a second transfer setting, we evaluate on
100-example and 300-example slices from the Hot-
potQA distractor development set (Yang et al.,
2018). HotpotQA is useful here because it is more
multi-hop and span-oriented than SimpleQA. We
report the standard exact match (EM) and token-
overlap F1 metrics.
4.2 Systems
Our primary full-benchmark run uses Claude Son-
net 4.6 for both drafting and refinement (Anthropic,
2026). Retrieval is implemented through the Ope-
nAI web search API in a simple RAG pipeline that
returns web evidence for subsequent answer gen-
eration (OpenAI, 2026; Lewis et al., 2021). The
baseline system, BASELINE-RAG, uses the same
retrieval component and the same drafting back-
bone but does not run the answer-conditioned re-
finement stage. This keeps the comparison focused
Figure 2: Correct-rate comparison on the full SimpleQA
benchmark. COUNTERREFINEis compared with a
matched one-pass retrieval baseline (BASELINE-RAG)
and representative one-shot leaderboard models.
on the contribution of COUNTERREFINErather
than on backbone or retrieval differences.
We also ran matched experiments with GPT-5 on
SimpleQA and HotpotQA (OpenAI, 2025). Unless
noted otherwise, the benchmark results in the main
paper are produced with the official evaluation pro-
cedures for the corresponding task.
5 Results
5.1 Performance across benchmarks and
backbones
Table 1 reports the main quantitative results. The
central pattern is consistent: the addition of COUN-
TERREFINEon top of BASELINE-RAG improves
the primary exactness-oriented metric across every
reported setting. This holds for the full SimpleQA
benchmark under the official grader, on smaller
SimpleQA slices, and on HotpotQA across both
100-example and 300-example evaluations. The
gains are therefore not confined to a single back-
bone, a single dataset, or a single evaluation scale.
Two aspects of the result are especially impor-
tant. First, the improvement appears on top of an
already retrieval-grounded baseline rather than a
no-retrieval model. The comparison is therefore
not between a weak closed-book system and a
stronger retrieval system, but between a matched

Benchmark MetricClaude 4.6 GPT-5
Base-RAG COUNTERREFINE∆Base-RAG COUNTERREFINE∆
SimpleQA (100)Correct↑65.075.0+10.0 57.071.0+14.0
F1↑65.075.0+10.0 59.773.2+13.5
SimpleQA (4,326)Correct↑63.767.7+4.0 67.373.1+5.8
F1↑64.168.1+4.0 58.672.1+13.5
HotpotQA (100)EM↑74.079.0+5.0 71.076.0+5.0
F1↑82.284.9+2.7 83.784.5+0.8
HotpotQA (300)EM↑70.074.0+4.0 68.071.0+3.0
F1↑83.986.0+2.1 83.684.9+1.3
Table 1: Main results across backbones and benchmarks. BASELINE-RAG is the matched one-pass retrieval baseline
using the same retriever and backbone as COUNTERREFINE. SimpleQA uses the official grader. HotpotQA uses the
distractor development setting on 100-example and 300-example slices. All values are percentages.
one-pass retrieval pipeline and the same pipeline
augmented with answer-conditioned refinement.
Second, the gains are strongest on metrics that
reward final answer exactness. This is most vis-
ible in the SimpleQA correct rate and HotpotQA
EM, which matches the intended role of COUNTER-
REFINEas a repair layer for final factual precision
rather than a general-purpose reasoning scaffold.
The cross-benchmark pattern is also informa-
tive. On SimpleQA, where a near-miss entity,
date, or number is scored as fully wrong, answer-
conditioned refinement yields clear gains under
the official evaluation pipeline. On HotpotQA, the
same mechanism continues to improve exact match,
including at the larger 300-example slice. F1 gains
are positive in most reported settings, but are gener-
ally smaller and less uniform than the exact-match
gains, which again aligns with the design goal of
correcting final answer identity rather than optimiz-
ing partial lexical overlap.
5.2 Intervention profile
The full Claude SimpleQA run clarifies how COUN-
TERREFINEachieves these gains. It changes only
a small fraction of predictions, yet the changes
are overwhelmingly beneficial: the system revises
5.6% of examples, helps 180, hurts 8, and pro-
duces zero refiner failures. Beneficial outcome-
changing revisions therefore outnumber harmful
ones by 22.5 to 1. Just as importantly, the not-
attempted rate remains fixed at 1.2%, so the im-
provement is not driven by abstaining more often
or by becoming artificially conservative.
Figure 3 makes the structure of the gain explicit.
Most examples are unchanged, but among changed
outcomes, incorrect →correct transitions dominate
Figure 3: Outcome transitions on the full SimpleQA
benchmark for Claude 4.6. Of all examples, 63.5%
stay correct, 32.1% stay not-correct, 4.2% are corrected
(incorrect →correct), and 0.2% are harmed (correct →
incorrect).
correct →incorrect ones by a wide margin. This is
the behavior we want from a post-retrieval repair
layer: it should intervene selectively, and when it
does, it should usually move the answer in the right
direction.
The same qualitative picture also appears across
smaller runs. Earlier 100-example and 300-
example SimpleQA experiments showed gains in
the same direction, and during development-stage
300-example iterations we repeatedly observed
zero hurt cases. Taken together with the full-
benchmark results, this suggests that the method is
capturing a stable error mode of one-pass retrieval
QA rather than exploiting a narrow artifact of one
dataset slice.
5.3 Representative failures and successes
Table 2 illustrates the main failure and success pat-
terns. The hurt cases are not random; they are

Question (abridged) BASELINE-
RAGFinal Gold Diagnosis
Representative hurt cases
Who appointed the Chief Justice of
India, Mirza Hameedullah Beg, in
1977?Fakhruddin Ali
AhmedIndira Gandhi Fakhruddin Ali
AhmedNearby evidence named the
government behind the
elevation rather than the
appointing president
requested by the question.
What day, month, and year was the
municipality of Arboletes, Antioquia,
Colombia, founded?1920 August 1958 July 20, 1920 The refinement evidence
referred to municipal
establishment rather than the
original founding date.
On what day, month, and year did
Manny Pacquiao marry Jinkee Jamora?May 10, 1999 May 10, 2000 May 10, 1999 A noisy timeline-style
snippet overrode an already
correct baseline year.
Representative helped cases
Who was awarded the Oceanography
Society’s Jerlov Award in 2018?Collin Roesler Annick
BricaudAnnick
BricaudAnswer-conditioned retrieval
surfaced an award-list entry
explicitly naming the 2018
winner.
What were the day, month, and year of
death of Mehr Chand Mahajan?5 December
198411 December
196711 December
1967The refinement stage found a
biographical snippet
containing the exact
parenthetical death date.
In which year did Fazal Ilahi Chaudhry
join the Muslim League?1940s 1942 1942 Re-querying with the draft
answer surfaced the exact
year instead of a decade-level
answer.
Table 2: Representative failures and successes from the full Claude trace log.
mostly cases ofrelation confusionorevent mis-
match. In these examples, the refinement stage
retrieves evidence about a nearby but different re-
lation, milestone, or date, and that adjacent fact
is plausible enough to override an initially correct
answer. This suggests that the main remaining
weakness is not lack of evidence, but insufficient
discrimination between closely related factual rela-
tions.
The helped cases show the complementary pat-
tern. BASELINE-RAG is often nearly correct but
too coarse, under-specified, or anchored on the
wrong nearby candidate. Once the draft answer is
injected back into retrieval, the second-stage search
surfaces a decisive snippet that the first pass failed
to isolate. In that sense, the value of COUNTER-
REFINEis not that it “reasons more” in the abstract,
but that it changes what evidence becomes visible
after a concrete hypothesis has been formed.
6 Discussion
The results support a simple interpretation: one-
pass retrieval often fails not because the correct
evidence is globally unavailable, but because thesystem commits too early to a plausible candidate
before retrieving the evidence that would best test
that candidate. COUNTERREFINEimproves this
by shifting the role of retrieval after drafting. The
second retrieval pass is no longer only about find-
ing topic-relevant material, but about verifying or
overturning a concrete factual hypothesis.
This is precisely why the method fits the broader
agenda of knowledgeable foundation models. The
core issue is not only how much knowledge a model
contains, but whether the system can recognize
when an initially plausible answer should be up-
dated in light of new evidence. COUNTERREFINE
provides one concrete inference-time mechanism
for that update process. It neither edits model pa-
rameters nor assumes perfect retrieval from the out-
set; instead, it uses answer-conditioned evidence
gathering to make factual repair possible after an
initial answer has already been formed.
7 Conclusion
We presented COUNTERREFINE, a retrieval-based
QA system that answers first and then retrieves
specifically to challenge its own answer. Across

our evaluations, the results indicate that a meaning-
ful share of error arises not from missing evidence,
but from committing too quickly to an insufficiently
tested candidate answer. By revisiting and, when
warranted, overturning plausible but incorrect re-
sponses, COUNTERREFINEshows that a simple
inference-time intervention can turn knowledge ac-
cess into knowledge repair.
Limitations
For COUNTERREFINE, the empirical evidence is
strongest for short-form factual settings, particu-
larly SimpleQA, where exact entity, date, and value
corrections are central to evaluation. Although the
transfer results on HotpotQA are encouraging, they
cover limited slices rather than the full benchmark
and therefore should be interpreted as evidence of
generalization potential rather than as a complete
characterization of multi-hop performance.
Additionally, COUNTERREFINEis designed to
repair a draft answer after retrieval, not to replace
retrieval itself or to solve long-form generation
problems. Its effectiveness therefore depends on
whether the retrieval stage surfaces sufficiently in-
formative supporting evidence. Extending the same
repair principle to further long-form settings, more
relation-sensitive validation, and broader multi-hop
benchmarks is a natural next step. We view these
as promising directions for expanding an inference-
time knowledge repair pattern that already proves
effective in the present setting.
Broader Impact and Ethical
Considerations
COUNTERREFINEpoints to a practical and hope-
ful direction for knowledgeable foundation models:
not only to store more facts, but also to detect and
repair wrong answers before they reach the user.
Because the method is modular and compatible
with standard retrieval pipelines, it can serve as a
practical reliability layer for search assistants, edu-
cational tools, and scientific interfaces where small
factual errors can have outsized consequences.
At the same time, inference-time repair should
be understood as improving reliability rather than
guaranteeing truth. If retrieved evidence reflects
social, historical, or institutional bias, a repair layer
may reinforce those patterns rather than correct
them. In high-stakes settings, repaired answers
should therefore be treated as better-supported out-
puts rather than as definitive verification. The mostresponsible role for systems like COUNTERREFINE
is not to replace human judgment, but to raise the
standard of evidence behind model outputs, espe-
cially in domains such as medicine, law, public
policy, and education. In that sense, the broader
promise of inference-time repair is not just fewer
mistakes, but a more accountable form of knowl-
edge use.
References
Anthropic. 2026. Introducing claude sonnet 4.6.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu,
Roberta Raileanu, Xian Li, Asli Celikyilmaz, and
Jason Weston. 2024. Chain-of-verification reduces
hallucination in large language models. InFindings
of the Association for Computational Linguistics:
ACL 2024, pages 3563–3578, Bangkok, Thailand.
Association for Computational Linguistics.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony
Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent
Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
Kelvin Guu. 2023. RARR: Researching and revising
what language models say, using language models.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 16477–16508, Toronto, Canada.
Association for Computational Linguistics.
Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen,
Yujiu Yang, Nan Duan, and Weizhu Chen. 2024.
CRITIC: Large language models can self-correct
with tool-interactive critiquing. InThe Twelfth Inter-
national Conference on Learning Representations.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Preprint, arXiv:2005.11401.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
Truthfulqa: Measuring how models mimic human
falsehoods.Preprint, arXiv:2109.07958.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
SelfcheckGPT: Zero-resource black-box hallucina-
tion detection for generative large language models.
InThe 2023 Conference on Empirical Methods in
Natural Language Processing.
Kevin Meng, David Bau, Alex J Andonian, and Yonatan
Belinkov. 2022. Locating and editing factual associ-
ations in GPT. InAdvances in Neural Information
Processing Systems.
Kevin Meng, Arnab Sen Sharma, Alex J Andonian,
Yonatan Belinkov, and David Bau. 2023. Mass-
editing memory in a transformer. InThe Eleventh
International Conference on Learning Representa-
tions.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
OpenAI. 2025. Gpt-5 is here.
OpenAI. 2025. simple-evals. GitHub repository.
Archived benchmark results table and reference im-
plementation; accessed March 12, 2026.
OpenAI. 2026. Web search. https:
//developers.openai.com/api/docs/guides/
tools-web-search. OpenAI API documentation.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. InProceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry
Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and Thang Luong. 2024. Fresh-
LLMs: Refreshing large language models with search
engine augmentation. InFindings of the Association
for Computational Linguistics: ACL 2024, pages
13697–13720, Bangkok, Thailand. Association for
Computational Linguistics.
Han Wang, Archiki Prasad, Elias Stengel-Eskin,
and Mohit Bansal. 2025. Retrieval-augmented
generation with conflicting evidence.Preprint,
arXiv:2504.13079.
Yuxia Wang, Minghan Wang, Muhammad Arslan Man-
zoor, Fei Liu, Georgi Nenkov Georgiev, Rocktim Jy-
oti Das, and Preslav Nakov. 2024. Factuality of large
language models: A survey. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 19519–19529, Miami,Florida, USA. Association for Computational Lin-
guistics.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese,
John Schulman, and William Fedus. 2024. Mea-
suring short-form factuality in large language models.
Preprint, arXiv:2411.04368.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin
Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong
Yu. 2024. Chain-of-note: Enhancing robustness in
retrieval-augmented language models. InProceed-
ings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 14672–14685,
Miami, Florida, USA. Association for Computational
Linguistics.