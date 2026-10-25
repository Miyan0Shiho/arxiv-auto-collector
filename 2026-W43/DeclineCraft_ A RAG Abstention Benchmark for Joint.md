# DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge

**Authors**: Atharva Thakur, Abulhair Saparov, Bruno Ribeiro

**Published**: 2026-10-20 15:07:19

**PDF URL**: [https://arxiv.org/pdf/2610.28340v1](https://arxiv.org/pdf/2610.28340v1)

## Abstract
Reliable retrieval-augmented generation (RAG) requires recognizing when neither retrieved documents nor parametric knowledge supports an answer. We introduce DeclineCraft, a dynamic abstention benchmark whose unanswerable questions target Minecraft feature additions after a corpus cut-off, with audits of possible parametric exposure. Later cut-offs support evaluation of future models. Across 27 LLMs, we measure Correct Abstention Rate (CAR; how often models acknowledge insufficient knowledge on unanswerable questions), Answerable Attempt Rate and Answer Accuracy. We find that (1) single-shot retrieval significantly raises CAR in 38 of 40 configurations (model and thinking setting) versus no retrieval. (2) Iterative RAG, in which the model self-assesses its context and can request up to two further retrieval rounds, significantly lowers CAR versus single-shot RAG in 30 of 40 configurations (mean 46.4%), with little change in Answer Accuracy (68.5% to 69.3%). Adding query rewriting and reranking to hybrid (dense & sparse) retrieval lowers CAR for all 12 models tested. (3) Larger models answer more accurately (Spearman $ρ= +0.60$ with total parameters), but neither model size nor release date shows a significant association with CAR. (4) Without retrieval, enabling thinking on the same weights lowers CAR significantly for 8 of 13 models (by 14% points on average), and the responses shift from acknowledged uncertainty to premise denial.

## Full Text


<!-- PDF content starts -->

DECLINECRAFT: A RAG Abstention Benchmark for Jointly Measuring
Combined Retrieval and Parametric Knowledge
Atharva Thakur Abulhair Saparov Bruno Ribeiro
Purdue University
{thakur27, asaparov, ribeirob}@purdue.edu
Abstract
Reliable retrieval-augmented generation (RAG)
requires recognizing when neither retrieved
documents nor parametric knowledge supports
an answer. We introduce DECLINECRAFT, a
dynamic abstention benchmark whose unan-
swerable questions targetMinecraftfeature ad-
ditions after a corpus cut-off, with audits of
possible parametric exposure.1Later cut-offs
support evaluation of future models. Across 27
LLMs, we measure Correct Abstention Rate
(CAR; how often models acknowledge insuffi-
cient knowledge on unanswerable questions),
Answerable Attempt Rate and Answer Accu-
racy. We find that (1) single-shot retrieval sig-
nificantly raises CAR in 38 of 40 configura-
tions (model and thinking setting) versus no
retrieval. (2) Iterative RAG, in which the model
self-assesses its context and can request up to
two further retrieval rounds, significantly low-
ers CAR versus single-shot RAG in 30 of 40
configurations (mean 46.4%), with little change
in Answer Accuracy (68.5% to 69.3%). Adding
query rewriting and reranking to hybrid (dense
& sparse) retrieval lowers CAR for all 12 mod-
els tested. (3) Larger models answer more ac-
curately (Spearman ρ= +0.60 with total pa-
rameters), but neither model size nor release
date shows a significant association with CAR.
(4) Without retrieval, enabling thinking on the
same weights lowers CAR significantly for 8 of
13 models (by 14% points on average), and the
responses shift from acknowledged uncertainty
to premise denial.
1 Introduction
A player fleeing skeletons inMinecraft Java Edi-
tionasks an assistant whether copper horse armor
would let a horse survive three arrows. Answer-
ing requires combining the rules for arrow damage,
horse health points, and the protection supplied
by this particular armor in a specific game edition.
1Code and data are freely available at: https://github.
com/Thakur-Atharva/declinecraft-benchmarkSuppose copper horse armor was introduced after
both the model’s training and its retrieval corpus.
Passages about horses, skeletons and other armor
types may be relevant, yet the key fact about the
horse’s armor is missing. The assistant must ac-
knowledge this knowledge gap. For instance, when
asked for the armor’s protection, Qwen3 32B with
thinking enabled instead declares: “Since ‘copper
horse armor’ is not a recognized item in the game,
it would provide 0 armor points. If you meant
golden horse armor, it provides 7 armor points.”
(Table 1). This response treats a presupposition it
cannot verify as false (Kim et al., 2021, 2023), a
premise denial, then substitutes a different item for
the one the user asked about.
For RAG systems, recognizing such a knowl-
edge gap must remain part of the decision to an-
swer as retrieval and thinking proceed. This matters
most when the answers are consumed by AI agents,
which may not notice that a premise denial is un-
supported by the model’s available knowledge. A
human user, by contrast, may know that a new law
was just passed, or thatcopper horse armorwas
added toMinecraft, when posing their query. Eval-
uation must therefore distinguish an admission of
missing knowledge from a premise denial that the
model cannot verify.
Distinguishing abstention from premise denial
matters for retrieval-augmented generation (RAG)
wherever documentation and the world it describes
can change (Barnett et al., 2024; Xu et al., 2024).
RAG systems can combine retrieved documents
withparametric knowledge, the information stored
in a model’s weights (Lewis et al., 2020). Unless
explicitly asked about a historical version, a system
should allow for questions about a time after both
the corpus cut-off and the model’s knowledge cut-
off. An item absent from an earlier version may
exist when the user asks about it; absence from
the available sources does not establish present-day
nonexistence. A reliable system should therefore
1
arXiv:2610.28340v1  [cs.CL]  20 Oct 2026

Answerable.“I am taming a horse in the plains by feeding
it golden carrots until heart particles appear, then I equip it
with a saddle and armor. If I encounter a skeleton horse, what
item will it drop when killed?”
Reference answer: bones, documented inC pre.
Unanswerable.“I ride my horse across the plains, holding
the jump control to charge a high leap over the fence gate
blocking my path. If I equip copper horse armor, how many
armor points will it provide?”
The answer (4 points) is documented in Cpost; the retrieved
Cprepages list leather, iron, gold and diamond horse armor.
Abstention
Gemma 4 31B“The provided context does not contain the
answer. I would need to know if copper
horse armor exists in the game, as the
provided text only lists leather, iron, gold,
and diamond variants.”
Premise
denial
Qwen3 32B
(thinking)“. . . Since ‘copper horse armor’ isnot a
recognized item in the game, it would
provide0 armor points.If you meant
golden horse armor, it provides 7 armor
points.”
Confabulation
Llama 3.3
70B“. . . Copper horse armor provides8 armor
pointsfor your horse.”
Table 1: Two benchmark questions and real responses
to the unanswerable one; bold highlights unsupported
claims. Copper horse armor was documented on the
wiki only from July 2025, after the corpus cut-off. The
first response acknowledges missing knowledge; the
second is a premise denial followed by a substituted
item; the third confabulates a numerical answer. Only
the first is a correct abstention. The first two use single-
shot RAG; the third is without retrieval (with RAG,
Llama 3.3 70Babstains).
abstainwhen these sources together cannot support
an answer, identifying the knowledge gap, without
categorically rejecting a premise it cannot verify.
Evaluating this behavior is challenging: an an-
swer missing from retrieved passages may still
be available in the model’s weights. Context-
grounded benchmarks instruct models to answer
only from supplied documents (Song et al., 2025;
Ming et al., 2025; Muhamed et al., 2026). UAE-
val4RAG’s out-of-database questions include facts
from newer news absent from its database, un-
der a database-grounded answering protocol (Peng
et al., 2025). AbstentionBench evaluates, without
retrieval, questions whose answers changed after
the models’ knowledge cut-offs (Kirichenko et al.,
2025). Our study combines a time-partitioned cor-
pus, whose later part postdates the models’ knowl-
edge cut-offs, with related but insufficient retrieved
evidence, and scores premise denial separately
from acknowledged uncertainty.To evaluate this setting, we introduce DE-
CLINECRAFT, a dynamic benchmark of 700 an-
swerable and 300 unanswerable questions built
from the update history ofMinecraft(Minecraft
Wiki Contributors, 2026). The interacting game
mechanics ofMinecraftand its feature additions
over time create facts that can only be learned from
new wiki pages: e.g., knowing how existing armor
works does not establish the protection of a newly
introduced material against a pre-existing weapon.
We construct a retrieval corpus ( Cpre) from wiki
pages as they stood at a corpus cut-off and generate
unanswerable questions from pages created later
(Cpost). Answerable questions use the same gen-
eration pipeline as unanswerable ones, with their
target facts drawn from the earlier corpus (Figure 1).
We audit our unanswerable questions against the
earlier documentation. To assess possible para-
metric exposure, we use published knowledge cut-
offs and aleakage audit. The leakage audit tests
whether models answer the unanswerable questions
correctly without retrieval. The temporal design es-
tablishes a knowledge gap when the required facts
postdate both parametric knowledge and the RAG
corpus. The benchmark is dynamic: it can be re-
generated with a later corpus cut-off as new game
content becomes available.
We evaluate 27 LLMs from several families un-
der three retrieval conditions, with thinking enabled
and disabled where possible.Single-shot RAGre-
trieves five wiki passages (chunks) once, combining
keyword and embedding search.Iterative RAGlets
the model request up to two further retrieval rounds
when it judges the chunks insufficient (Section 4.2).
We measureCorrect Abstention Rate(CAR) as the
share of unanswerable questions on which a model
acknowledges that it lacks the answer. Our experi-
ments yield three main findings:
•Retrieval helps models recognize missing
knowledge, but additions to the retrieval
pipeline erode that benefit.Single-shot RAG
raises CAR for almost every configuration. Iter-
ative RAG, ten instead of five chunks per query,
and query rewriting with reranking each lower it
again (Section 4.2).
•Newer and larger models do not reliably ac-
knowledge knowledge gaps.CAR has no sig-
nificant association with release date or param-
eter count. Recent families such as Qwen3.5
and Qwen3-Next often deny the existence of the
item or feature mentioned in the question (Sec-
2

“I am taming a horse
. . . then I equip it
with a saddle and
armor. ”Horse Update
(Cpre)
“. . . skeleton horses
drop bones. . . ”Skeleton
horses drop
bones when
killed.“. . . If I encounter a
skeleton horse, what
item will it drop when
killed?”✓answer found
✓scenario does
not state the fact
“I ride my horse
across the plains . . .
over the fence gate
blocking my path. ”Copper Horse
Armor (Cpost;
page created July
2025)Copper horse
armor gives 4
armor points.“. . . If I equip copper
horse armor, how
many armor points
will it provide?”✓no target fact
found
✓no answer
found
Cpre the wiki as of the cutoff, 31
March 2025; 2,642 pages; the only corpus
evaluated systems can retrieve fromCpost pages created
after the cutoff; 170
pages; never retrievableAssemble
at most 2 questions per chunk, 10 per
page; novelty cues removed
700answerable, 300unanswerable1Scenario
3 random Cprechunks2Target chunk
sample 1 of top 5 eligible3Fact
selected from the target4Question
its answer is the fact5Verify
judge checks retrieved Cpresteps 1–4: shared generation pipeline with one generator LLMAnswerable Unanswerablegenerated text
pipeline step
reads a corpusFigure 1: Building DECLINECRAFT. A shared pipeline generates a player scenario from the pre-cut-off corpus
Cpre, samples one of the five highest-ranked eligible target chunks, selects a fact and appends a question requiring
the selected fact. The target comes from Cprefor answerable questions and the post-cut-off corpus Cpostfor
unanswerable questions. A separate verifier checks whether retrieved Cprepassages support the answer. Both
examples are real benchmark items.
tion 4.3).
•Thinking can replace uncertainty with
premise denial.Without retrieval, enabling
thinking significantly lowers CAR for 8 of the
13 models with switchable thinking and raises
it for none. The lost abstentions most often be-
come premise denials. With retrieval, the effect
of thinking differs by model (Section 4.4).
2 Related Work
Abstention and selective QA.SQuAD 2.0 eval-
uates abstention when a passage lacks the answer
(Rajpurkar et al., 2018). Selective QA studies when
to withhold answers to control error among an-
swered questions (Kamath et al., 2020). R-Tuning
trains models to acknowledge gaps in parametric
knowledge (Zhang et al., 2024), while Feng et al.
(2024) use collaboration between models to detect
knowledge gaps and decide whether to abstain.
Abstention benchmarks.AbstentionBench in-
cludes stale-knowledge questions evaluated with-
out retrieval (Kirichenko et al., 2025). Refusal-
Bench perturbs question–passage pairs to test
context-grounded abstention (Muhamed et al.,
2026). On DECLINECRAFT, its context-only an-
swer prompt makes models decline many answer-
able questions (Appendix B). Among its six cat-
egories, UAEval4RAG generates out-of-database
questions from recent news and evaluates absten-
tion when the database lacks support (Peng et al.,
2025). Our setting permits both retrieved and para-metric knowledge and distinguishes acknowledged
uncertainty from premise denial, following (QA)2’s
distinction between false and unverifiable assump-
tions (Kim et al., 2023).
Evidence sufficiency in RAG.Mallen et al.
(2023) show that retrieval helps answer questions
about less popular facts that models struggle to
recall. Joren et al. (2025) study whether the re-
trieved context alone suffices to answer, and how
models behave when it does not. Their evaluation
uses answerable questions; DECLINECRAFTadds
questions designed to expose gaps in both sources.
FLARE retrieves during generation using predicted
upcoming sentences (Jiang et al., 2023); iterative
RAG uses prompted self-assessment of context suf-
ficiency (Appendix E). Related but insufficient pas-
sages can mislead models (Cuconasu et al., 2024),
and prior work studies how further retrieval affects
abstention (Xie et al., 2026).
Appendix L discusses uncertainty estimation,
knowledge conflicts, temporal datasets, broader
RAG evaluation and Minecraft-based evaluation.
3 DECLINECRAFT: A Dynamic
Benchmark for Knowledge-Aware
Abstention
DECLINECRAFTevaluates whether a RAG sys-
tem recognizes when its parametric knowledge
and retrieved documents together cannot support
an answer. The questions use real Minecraft con-
tent, including additions documented after the cor-
3

pus cut-off. When the required facts also post-
date the model’s knowledge cut-off, the system
should acknowledge the missing information. We
distinguish such a response from premise denial,
in which the model asserts, without support, that
an item or feature named in the question does not
exist. The issue is whether the model has enough
knowledge to justify that claim, even if it happens
to be correct. Table 1 shows an example of each,
alongside a third outcome: an unsupported answer.
3.1 Temporal Corpora and Knowledge
Availability
We collect two corpora from theMinecraftWiki
(minecraft.wiki ), using 31 March 2025 as the
corpus cut-off:
•Pre-cut-off corpus ( Cpre):The latest revision
of each wiki page at or before the corpus cut-
off: 2,642 pages split into 11,612 chunks as de-
tailed in Appendix D. This is the only corpus
available to the evaluated retrieval pipelines. By
default they use ahybrid retrieverthat combines
keyword (BM25) and embedding similarity (Ap-
pendix E).
•Post-cut-off corpus ( Cpost):Pages created after
the corpus cut-off, stored at their latest revision
when scraped on 8 September 2026: 170 pages
split into 712 chunks. These supply facts for
generating unanswerable questions.
We exclude pages that existed before the corpus
cut-off from Cpost, even if they were later updated,
to focus on missing knowledge about later addi-
tions to the game. Appendix D provides details on
data collection and filtering.
Excluding a fact from retrieval does not by itself
exclude it from parametric knowledge. Temporal
separation requires the fact to postdate the model’s
knowledge cut-off as well as the corpus cut-off. We
report each model’s knowledge cut-off when avail-
able in Table 11. Our leakage audit therefore asks
models the 300 unanswerable questions without
retrieval and scores their answers against reference
answers that an oracle model writes from Cpost. A
model that often answers correctly may have seen
the post-cut-off pages in training (Figure 19).
3.2 Scenario and Question Generation
Figure 1 traces the generation of one answerable
question (top, blue) and one unanswerable ques-
tion (bottom, orange) through the same five steps.
A generator LLM, Qwen3.5 122B-A10B , performsevery writing step, each with its own prompt (Ap-
pendix J.1).
1.Scenario.We draw three random chunks from
Cpreand give them to the generator. It writes a
playerscenarioof under 30 words about one or
two related concepts in these chunks.
2.Target chunk.For an answerable question, we
embed the scenario with Qwen3-Embedding-8B
and retrieve the ten chunks with the most sim-
ilar embeddings in terms of cosine similarity
from Cpre. For an unanswerable question, we
follow the same procedure, except the chunks
are retrieved from Cpost. We drop chunks from
technical pages and, for an answerable question,
the scenario’s own three chunks. Thetarget
chunkis drawn uniformly at random from the
five highest-ranked chunks that remain.
3.Fact.The generator lists up to six facts that
appear in the target chunk but not in the com-
parison text. For an answerable question, the
comparison text is the scenario’s three chunks,
so the scenario does not already contain the fact.
For an unanswerable question, it is the three
Cprechunks most similar to the target chunk, so
the fact is new relative to Cpre. One listed fact
is chosen uniformly at random.
4.Question.The generator receives the scenario,
the fact and the target chunk. It writes one can-
didate question from the player’s perspective
whose answer is the fact, and the question is
appended to the scenario.
5.Verification.A separate LLM checks each can-
didate againstC pre(Section 3.3).
Appendix G compares Qwen3.5 122B-A10B with
two other generators on the same pipeline; we
chose to use Qwen3.5 122B-A10B since the result-
ing answerable and unanswerable questions were
the hardest to tell apart by wording alone. Ap-
pendix F details the algorithms, and Appendix C
shows a generated question of each kind.
Scenarios make the benchmark harder in a re-
alistic way, because they lead retrieval to familiar
contexts with insufficient information. For 71.3%
of the 300 unanswerable questions, the hybrid re-
triever returns at least one of the scenario’s own
source chunks among its five retrieved chunks (Fig-
ure 24). Such a chunk is related to the scenario but
says nothing about the missing fact. Removing the
scenario raises Correct Abstention Rate for all 10
models tested with single-shot RAG (Section A.4).
4

3.3 Verification and Benchmark Assembly
Generation can go wrong in three ways:
•An answerable question whose answer is not in
Cpre;
•An unanswerable question whose fact is already
inCpre, for example because a page created after
the corpus cut-off restates an older fact;
•Many questions drawn from the same few pages.
The first two mislabel a question, and the third lets
a few pages dominate the benchmark. Three steps
remove them.
Automated verification.A separate verifier
LLM, Gemma 4 31B , checks each candidate sce-
nario and question, so that no model judges its
own questions. For each query, the verifier reads
the 10 Cprechunks retrieved with hybrid retrieval
and reranking.Rerankinguses a second model to
reorder the retrieved chunks by their relevance to
the query. An answerable candidate is kept only if
both of the following hold: (1) the scenario does
not state or directly imply the target fact, so it does
not give the answer away; and (2) the chunks re-
trieved for the question contain the information
needed to answer it. An unanswerable candidate is
kept only if both of the following hold: (1) the Cpre
chunks retrieved for the target fact do not state or
directly imply it; and (2) the chunks retrieved for
the question do not answer it. Appendix J gives the
verifier’s prompts.
Manual review.We reviewed all unanswerable
questions against Cpreand dropped 21 malformed
candidates (claiming fake in-game items like “steel”
and fabricated game mechanics like “attacked by
a creeper with a diamond sword”). We found no
examples of questions which had answers in Cpre.
We keep questions that can be answered only by
assuming that an older mechanic carries over to
a new item, such as whether new copper tools ac-
cept an enchantment that iron tools already accept:
such a guess may be right, but neither Cprenor the
model’s training establishes it, so the model should
still abstain.
Assembly.The number of chunks per Cpostpage
varies widely (median 2, up to 64), so without lim-
its the questions concentrate on a few pages. We
therefore allow at most 2 questions per chunk and at
most 10 per page. We also drop candidates whose
question sentence has a novelty cue (“new,” “re-
cently”, a month or a year), which would reveal apost-cut-off subject.
The final benchmark contains 700 answerable
and 300 unanswerable questions. For the 700 an-
swerable questions, the default hybrid retriever
returns the target chunk among its five retrieved
chunks in 91.0% of cases. Answerability in the
corpus therefore does not guarantee that the LLM
receives the target passage.
4 Results
In what follows, we introduce our results over
27 LLMs evaluated on DECLINECRAFTwithout
retrieval, with single-shot RAG and with itera-
tive RAG. We assess whether models acknowl-
edge missing knowledge on unanswerable ques-
tions while attempting and correctly answering an-
swerable questions. After describing the evaluation
setup and metrics (Section 4.1), we compare re-
trieval conditions (Section 4.2), examine the associ-
ations of Correct Abstention Rate with model size
and release date (Section 4.3), and measure the ef-
fect of enabling thinking (Section 4.4). Throughout
these results, we distinguish acknowledged uncer-
tainty from premise denial. Appendix A reports the
results for every model and further analyses.
4.1 Evaluation Setup and Metrics
We use the 700 answerable and 300 unanswerable
questions constructed in Section 3. Aconfiguration
is one model with one thinking setting. 13 models
haveswitchablethinking (the same weights run
with thinking enabled or disabled); running both
gives 40 configurations per retrieval condition. Ap-
pendix H gives the API settings and statistical tests;
Appendix K lists the models and their knowledge
cut-offs, and Tables 2 and 3 report every model and
thinking setting under each condition.
Correct Abstention Rate(CAR) is the per-
centage of unanswerable questions on which the
model acknowledges insufficient knowledge with-
out a premise denial or an unsupported answer. An
LLMresponse judgelabels each response as an
abstention, a premise denial or an attempted an-
swer (Appendix I). A premise denial followed by a
guessed answer still counts as a premise denial, not
a correct abstention. Premise denials of answerable
questions count as attempts. Appendix I repeats
the analyses under a lenient rule that also counts
premise denials as abstentions: the retrieval find-
ings hold, whereas the thinking and release-date
findings change.Answerable Attempt Rateis
5

No RAG Single-shot
k = 5Single-shot
k = 10Iterative
RAG020406080100Correct Abstention Rate (%)
Command A 111B
GLM-4.5-AirGPT-5 miniGemini 3 FlashGemma 4 31B
Llama 3.3 70B
OLMo 3 32B ThinkQwen3 32B
Qwen3-Next 80B-A3BQwen3.5 122B-A10B
Qwen3.5 27B
One model Mean (95% CI over models)Figure 2: Retrieval-depth ablation on 11 models with
ak=10 run.kis the number of chunks retrieved per
query. Thinking is disabled where switchable; other
models use their defaults. Gray lines: individual mod-
els; blue: mean Correct Abstention Rate with a 95%
confidence interval over models. Appendix A.2 gives
the comparisons and their limits.
the percentage of answerable questions the model
attempts to answer.Answer Accuracyaverages
the scores of three LLMaccuracy judgesfrom dif-
ferent families on the answerable questions: 1 for
correct, 0.5 for partly correct and 0 for incorrect
answers or abstentions, againstreference answers
that anoracle modelwrites from each question’s
source pages. Outputs that exhaust the output bud-
get of 16,000 tokens, including thinking, without a
final answer (1.9% of all answers; Appendix H) are
excluded from the two rates and score 0 in Answer
Accuracy. Appendix I details the scoring.
Human validation.Three annotators indepen-
dently label 70 responses and score 30 attempted
answers for accuracy. The response judge matches
the human majority on 96% of response labels; dis-
agreements mainly concern an abstention sentence
followed by a premise denial. For accuracy, mean
judge–human agreement is κ= 0.66 , versus 0.42
between humans (quadratic-weighted Cohen’s κ).
Samples are balanced across response labels and
accuracy-score bins; Appendix I gives details and
confidence intervals.
4.2 Retrieval Raises Correct Abstention Rate;
Iterative RAG Lowers It Again
Single-shot RAG retrieves five chunks from Cpre
using the default hybrid retriever (BM25 and dense
retrieval, combined by reciprocal rank fusion; see
details in Appendix E). Figure 2 plots Correct Ab-
stention Rate (y-axis) against the retrieval condition
BM25 Dense Hybrid Hybrid
+rerankHybrid
+rewriteHybrid
+rerank
+rewrite
Retrieval pipeline (single-shot, k = 5; hybrid = dense + BM25)020406080100Correct Abstention Rate (%)
Command A 111B
GLM-4.5-AirGPT-5 miniGemini 3 FlashGemma 4 31B
Llama 3.3 70B
OLMo 3 32B Think
Qwen3 32B
Qwen3 8B
Qwen3-Next 80B-A3BQwen3.5 122B-A10B
Qwen3.5 27B
One model Mean (95% CI over models)Figure 3: Retrieval-pipeline ablation on 12 models, with
single-shot RAG and five chunks. The six pipelines are
keyword (BM25) or embedding (dense) search alone,
the default hybrid of the two, and the hybrid with rerank-
ing, query rewriting or both (Appendix E). Thinking
is disabled where switchable; other models use their
defaults. Gray lines: individual models; blue: mean
Correct Abstention Rate with a 95% confidence interval
over models. Appendix A.3 reports the pipeline com-
parisons.
(x-axis) for the subset of 11 models also evaluated
with ten retrieved chunks; gray lines show individ-
ual models, and the blue line shows their mean.
Across all 40 configurations, mean Correct Ab-
stention Rate rises from 17.1% without retrieval
to 56.4% with single-shot RAG. Relative to no re-
trieval, 38 configurations show a statistically signif-
icant increase in Correct Abstention Rate, and none
shows a statistically significant decrease ( q <0.05 ;
see Appendix H for details on statistical analysis).
Our iterative RAG implementation uses
prompted self-assessment, inspired by Self-
RAG (Asai et al., 2024) and query-level
self-evaluation (Xie et al., 2026). In our experi-
ments, it starts with the same five chunks and lets
the model request up to two additional retrieval
rounds when it judges the context insufficient
(Algorithm 4; Appendix H). Relative to single-shot
RAG, mean Correct Abstention Rate falls to 46.4%,
with significant decreases in 30 of the same 40
configurations and no significant increases. Mean
Answer Accuracy on the answerable questions
changes from 68.5% to 69.3%, compared with
24.3% without retrieval. Figure 8 in the appendix
shows both metrics for each configuration. Thus,
in this pipeline, further retrieval lowers CAR while
adding little accuracy.
The Sulfur Cube example in Appendix A.9 illus-
6

0 50 100
CAR (%)Qwen3.5 27BQwen3.5 122B-A10BGemma 4 31BGemma 4 26B-A4BGLM-4.5-AirQwen3.5 35B-A3BQwen3 32BQwen3 8BQwen3 4BQwen3 14BQwen3 235B-A22BGemini 2.5 FlashQwen3 30B-A3BNo retrieval
8 lower, 0 higher
0 50 100
CAR (%)Single-shot RAG
4 lower, 4 higherThinking
disabledLower CAR with
thinking enabledHigher CAR with
thinking enabledFigure 4: Correct Abstention Rate with thinking dis-
abled (gray dot) and enabled (arrowhead), holding
weights fixed for the 13 models with switchable think-
ing. Orange: lower with thinking enabled; blue: higher;
faded: not significant (q≥0.05).
trates this failure: GPT-OSS 120B correctly abstains
with single-shot RAG, then uses a retrieved passage
about Magma Cubes to answer about the unfamiliar
entity after further retrieval. The example shows
how relevant-looking context can support an un-
warranted substitution; the aggregate comparison
does not isolate which component of iterative RAG
causes the decrease, and the reflection prompt itself
asks whether the context suffices to answer rather
than whether to answer at all.
Appendix A.2 examines how retrieving five ver-
sus ten chunks affects Correct Abstention Rate for
11 models (Figure 2) and discusses why iterative
RAG cannot isolate the effect of follow-up queries
from that of additional context. Appendix A.3 com-
pares six retrieval pipelines (Figure 3): adding both
query rewriting and reranking to hybrid retrieval
lowers Correct Abstention Rate for all 12 tested
models. Appendix A.4 details the scenario-removal
ablation introduced in Section 3.2: removing sce-
narios from both retrieval queries and model inputs
raises Correct Abstention Rate for all 10 tested
models under single-shot RAG.
4.3 Correct Abstention Rate Has No
Significant Association with Size or
Release Date
Under single-shot RAG, neither total parameters
(Spearman ρ=−0.24 ,p= 0.31 ) nor active pa-
51020 50100200
T otal parameters (B, log)020406080100Correct Abstention Rate (%)
 = -0.24 (p = 0.31)
20 models
Jan 25 Jul 25 Jan 26
Release month
 = -0.16 (p = 0.42)
27 modelsOpen-weight Closed (API)Figure 5: Correct Abstention Rate versus total param-
eters (20 open-weight models, left) and release month
(27 models, right; closed models in orange), with single-
shot RAG. One point per model, with thinking enabled
where switchable; neither Spearman correlation is sta-
tistically significant at the 5% level. Figure 12 in the
appendix shows active parameters.
rameters ( ρ= +0.21 ,p= 0.37 ) has a significant
association with Correct Abstention Rate across
the 20 open-weight models. Active parameters are
those used for each token. A mixture-of-experts
model uses only part of its total parameters for
each token. Figure 5 (left) shows the comparison
with total parameters, with thinking enabled where
switchable; Figure 12 in the appendix shows active
parameters. Release month likewise has no signifi-
cant association across all 27 models ( ρ=−0.16 ,
p= 0.42 ; Figure 5, right), or across the 14 families
when selecting the model with the highest Correct
Abstention Rate in each ( ρ=−0.22 ,p= 0.45 ).
On the same open-weight models, Answer Accu-
racy is positively associated with total parameters
(ρ= +0.60 ,p= 0.005 ; Figure 13 in the appendix).
Thus, the observed accuracy gains with scale do not
establish better recognition of missing knowledge.
The response categories matter for interpreting
this result. Recent families such as Qwen3.5 and
Qwen3-Next frequently respond with premise de-
nial: they assert, without support, that the queried
item does not exist. Figures 17 and 18 in the ap-
pendix separate premise denials from abstentions
and attempted answers, by family and model. As
the copper horse armor responses in Table 1 illus-
trate, a premise denial can also lead to a numerical
answer or the substitution of a familiar item. Such a
response does not establish recognition of a knowl-
edge gap, even if its claim was true at the corpus
cut-off.
7

40 50 60 70 80 90 100
Answerable Attempt Rate (%)020406080100Correct Abstention Rate (%)
Qwen2.5-3B  Trust-Align 3B
Qwen2.5-7B  Trust-Align 7B
Qwen2.5-3B  Abstain-R1
Other modelsFigure 6: Answerable Attempt Rate against Correct
Abstention Rate, single-shot RAG. Gray: the 40 config-
urations of the other models. Arrows: baseline model
(hollow) to fine-tuned model (filled); Abstain-R1 also
changes the prompt template.
10 20 30 40 50 60 70 80 90 100
Answer Accuracy (%)020406080100Correct Abstention Rate (%)
Qwen2.5-3B  Trust-Align 3B
Qwen2.5-7B  Trust-Align 7B
Qwen2.5-3B  Abstain-R1
Other modelsFigure 7: Answer Accuracy against Correct Abstention
Rate with single-shot RAG, for the same models as Fig-
ure 6. Gray: the 40 configurations of the other models.
Arrows: baseline model (hollow) to fine-tuned model
(filled).
4.4 Thinking Lowers Correct Abstention Rate
Without Retrieval
Figure 4 compares thinking disabled (gray dots)
and enabled (arrowheads) for the 13 models with
switchable thinking, holding weights fixed, with-
out retrieval (left) and with single-shot RAG (right).
Orange indicates lower CAR, blue higher; faded
arrows mark changes that are not statistically sig-
nificant.
Without retrieval, we observe 8 statistically sig-
nificant decreases in CAR and no statistically sig-
nificant increases. Per question, enabling think-
ing without retrieval turns abstentions into premise
denials more often than the reverse for 9 models
(Figure 25). Appendix A.5 shows that Answer Ac-
curacy can improve while CAR falls.
Across 40 single-shot RAG configurations, An-
swer Accuracy and CAR have no statistically sig-
nificant unadjusted association (Spearman ρ=
−0.28 ,p= 0.08 ); Appendix A.6 reports a neg-
ative association after adjusting for release date.
With single-shot RAG, 33 of 40 configurations
have both a higher CAR and a higher Answer-
able Attempt Rate than without retrieval (Ap-
pendix A.7).
4.5 Models Fine-Tuned to Abstain
Some models are fine-tuned specifically for ab-
stention, such as Trust-Align 3B ,Trust-Align
7B(Song et al., 2025), and Abstain-R1 3B (Zhai
et al., 2026). We evaluate these three mod-
els against their corresponding base model (Ap-
pendix A.8). Figure 6 shows that fine-tuningraises CAR and lowers Answerable Attempt Rate.
With single-shot RAG, fine-tuning raises CAR
from 35.7% ( Qwen2.5-3B-Instruct ) to 89.3%
(Trust-Align 3B ), but Trust-Align 3B attempts
only 43.6% of answerable questions. These mod-
els lie on or near the Pareto trade-off frontier of
the other configurations, the highest CAR that they
reach at or above each Answerable Attempt Rate.
They trade a higher CAR for lower Answerable
Attempt Rate and Answer Accuracy by abstaining
on answerable questions that other models answer
correctly (Figure 7). Fine-tuning moves these three
models along the trade-off frontier rather than be-
yond it.
5 Conclusion
We introduced DECLINECRAFT, a dynamic bench-
mark built from datedMinecraftWiki pages to
evaluate recognition of missing knowledge and dis-
tinguish acknowledged uncertainty from premise
denial. Across 27 LLMs, single-shot RAG pro-
duces statistically significant increases in Correct
Abstention Rate in 38 of 40 configurations relative
to no retrieval; our prompted iterative RAG pipeline
produces statistically significant decreases in 30
of the same configurations relative to single-shot
RAG, with little change in Answer Accuracy. Un-
der single-shot RAG, Correct Abstention Rate has
no statistically significant association with model
size or release date. Enabling thinking produces
statistically significant decreases in Correct Absten-
tion Rate for 8 of 13 models without retrieval, with
no statistically significant increases; its effect with
8

single-shot RAG varies across models.
These results motivate evaluating and training
recognition of missing knowledge explicitly. A
premise denial does not establish this recognition
when the model cannot verify its claim. Future
work should improve judgments of evidence suf-
ficiency across retrieval rounds and test whether
these findings extend to other domains with dated
documentation.
6 Limitations
While DECLINECRAFTprovides a controlled envi-
ronment to evaluate the temporal adaptability and
abstention capabilities of RAG-equipped LLMs,
we identify several limitations to our work.
6.1 Domain Concentration of the Corpus
Our framework uses only theMinecraftWiki,
which may not represent the complexity of other do-
mains where RAG systems are deployed. Other do-
mains may have more erroneous chunks or poorly
maintained version histories, or may not receive
regular updates. Future work is needed to test
whether our findings on knowledge unavailability
at a corpus cut-off generalize to other domains.
6.2 Dependence on LLM Judges
We rely on LLMs throughout: Qwen3.5
122B-A10B writes the questions, Gemma 4
31B verifies them, the response judge Gemma 4
26B-A4B labels abstentions and premise denials,
the oracle model DeepSeek V4.1 Flash writes
reference answers, and three accuracy judges
score the answers. These models may impart their
own biases, such as preferring longer answers or
asking underspecified questions. A human study
on 100 shared items (Section I) finds that the
judges agree with the annotators at least as well
as the annotators agree with one another, but it
covers a small sample. The two accuracy judges
whose families we also evaluate show a small
self-preference (Section A.1).
6.3 One Chunking Scheme and Embedding
Model
We compare six retrieval pipelines, including
reranking and query rewriting, but all of them
search the same index: fixed-size chunks of about
1,500 characters split at line breaks, embedded with
Qwen3-Embedding-8B . Hierarchical or semantic
chunking, or other embedding models, may retrieve
differently.6.4 Scoring Premise Denials
Our headline Correct Abstention Rate excludes
premise denials, including premise denials fol-
lowed by a guessed answer. We report these re-
sponses separately from acknowledged uncertainty
(Appendix I). The answer prompt did not tell mod-
els that the game may contain items newer than
their knowledge. We have not tested whether mak-
ing this temporal uncertainty explicit changes the
frequency of premise denials.
6.5 Limited Evaluation Models
Because of cloud inference costs and limited local
hardware, we selected models to cover a wide range
of sizes, architectures and families. We chose the
Qwen3 family because it offers many sizes and
architectures within one family, which lets us vary
one factor while holding the others fixed.
Acknowledgments
This work was funded in part by the National Sci-
ence Foundation (NSF) award CNS-2504819 and
the Lilly Endowment. Any opinions, findings and
conclusions or recommendations expressed in this
material are those of the authors and do not neces-
sarily reflect the views of the sponsors.
References
Alibaba Cloud Community. 2025. Qwen3-Next: A
new generation of ultra-efficient model architecture
unveiled.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. InProceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering - Software Engineering for AI, CAIN
2024, pages 194–199. ACM.
Yoav Benjamini and Yosef Hochberg. 1995. Controlling
the false discovery rate: A practical and powerful
approach to multiple testing.Journal of the Royal
Statistical Society Series B: Statistical Methodology,
57(1):289–300.
Tianyu Cao, Neel Bhandari, Akhila Yerukola, Akari
Asai, and Maarten Sap. 2026. Out of style: RAG’s
fragility to linguistic variation. InProceedings of
the 19th Conference of the European Chapter of the
9

Association for Computational Linguistics (Volume
1: Long Papers), pages 280–318, Rabat, Morocco.
Association for Computational Linguistics.
Suting Chen, Peichun Hua, and Yunming Xiao. 2026.
Relevance is not sufficient evidence: Detecting ev-
idence gaps before generation in RAG.Preprint,
arXiv:2609.37469.
Jeffrey Cheng, Marc Marone, Orion Weller, Dawn
Lawrie, Daniel Khashabi, and Benjamin Van Durme.
2024. Dated data: Tracing knowledge cutoffs in large
language models. InFirst Conference on Language
Modeling.
Fedor Chernogorskii, Sergei Averkiev, Liliya Ku-
draleeva, Zaven Martirosian, Maria Tikhonova,
Valentin Malykh, and Alena Fenogenova. 2026.
DRAGOn: Designing RAG on periodically updated
corpus. InProceedings of the 19th Conference of
the European Chapter of the Association for Com-
putational Linguistics (Volume 4: Student Research
Workshop), pages 622–638, Rabat, Morocco. Associ-
ation for Computational Linguistics.
Jacob Cohen. 1960. A coefficient of agreement for
nominal scales.Educational and Psychological Mea-
surement, 20(1):37–46.
Cohere. 2025. Command A (model documentation).
Accessed 2026-10-06.
Gordon V . Cormack, Charles L A Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR
conference on Research and development in informa-
tion retrieval, SIGIR ’09, pages 758–759. ACM.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval
for RAG systems. InProceedings of the 47th In-
ternational ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024,
pages 719–729. ACM.
Gautier Dagan, Frank Keller, and Alex Lascarides. 2025.
Plancraft: an evaluation dataset for planning with
LLM agents. InSecond Conference on Language
Modeling. ArXiv:2412.21033.
DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Push-
ing the limits of KV cache compression.Preprint,
arXiv:2609.19969.
Bhuwan Dhingra, Jeremy R. Cole, Julian Martin
Eisenschlos, Daniel Gillick, Jacob Eisenstein, and
William W. Cohen. 2022. Time-aware language mod-
els as temporal knowledge bases.Transactions of the
Association for Computational Linguistics, 10:257–
273.Alden Do Rosario, Hussein Younes, and Felipe
Pires. 2026. Why RAGs hallucinate: Penalty-
aware evaluation of retrieval-augmented generation
systems with knowledge-gap canaries.Preprint,
arXiv:2608.26385.
Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Man-
dlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang,
De-An Huang, Yuke Zhu, and Anima Anandkumar.
2022. MineDojo: Building open-ended embodied
agents with internet-scale knowledge. InThirty-sixth
Conference on Neural Information Processing Sys-
tems Datasets and Benchmarks Track.
Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding,
Vidhisha Balachandran, and Yulia Tsvetkov. 2024.
Don’t hallucinate, abstain: Identifying LLM knowl-
edge gaps via multi-LLM collaboration. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 14664–14690, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Gemma Team. 2025. Gemma 3 technical report.
Preprint, arXiv:2503.19786.
Gemma Team. 2026. Gemma 4 technical report.
Preprint, arXiv:2607.02770.
GLM-4.5 Team. 2025. GLM-4.5: Agentic, reason-
ing, and coding (ARC) foundation models.Preprint,
arXiv:2508.06471.
Google. 2026a. Gemini 3 developer guide. Accessed
2026-10-13.
Google. 2026b. Gemini thinking. Accessed 2026-10-
06; table of default thinking levels per model.
Google DeepMind. 2025a. Gemini 3 Flash – model
card.
Google DeepMind. 2025b. Gemma 3 model
card. https://ai.google.dev/gemma/docs/
core/model_card_3. Accessed 5 October 2026.
Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang,
Joongbo Shin, Janghoon Han, Gyeonghun Kim, and
Minjoon Seo. 2022. TemporalWiki: A lifelong
benchmark for training and evaluating ever-evolving
language models. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6237–6250, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
10

Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-
Cheng Juan, Ankur Taly, and Cyrus Rashtchian. 2025.
Sufficient context: A new lens on retrieval augmented
generation systems. InThe Thirteenth International
Conference on Learning Representations.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna
Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and
17 others. 2022. Language models (mostly) know
what they know.Preprint, arXiv:2207.05221.
Amita Kamath, Robin Jia, and Percy Liang. 2020. Se-
lective question answering under domain shift. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 5684–
5696, Online. Association for Computational Lin-
guistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi,
Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir
Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui.
2023. RealTime QA: What’s the answer right now?
InNeural Information Processing Systems.
Najoung Kim, Phu Mon Htut, Samuel R. Bowman, and
Jackson Petty. 2023. (QA)2: Question answering
with questionable assumptions. InProceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
8466–8487, Toronto, Canada. Association for Com-
putational Linguistics.
Najoung Kim, Ellie Pavlick, Burcu Karagol Ayan, and
Deepak Ramachandran. 2021. Which linguist in-
vented the lightbulb? Presupposition verification for
question-answering. InProceedings of the 59th An-
nual Meeting of the Association for Computational
Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 1:
Long Papers), pages 3932–3945, Online. Association
for Computational Linguistics.
Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri,
and Samuel J. Bell. 2025. AbstentionBench: Rea-
soning LLMs fail on unanswerable questions. In
Advances in Neural Information Processing Systems.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Semantic uncertainty: Linguistic invariances for un-
certainty estimation in natural language generation.
InThe Eleventh International Conference on Learn-
ing Representations.Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with PagedAttention. InProceedings of the 29th
Symposium on Operating Systems Principles, SOSP
’23, pages 611–626. ACM.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InNeural Information Pro-
cessing Systems.
Qian Lin, Junyi Li, and Hwee Tou Ng. 2025. Dy-
naQuest: A dynamic question answering dataset
reflecting real-world knowledge updates. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 26918–26936, Vienna, Austria.
Association for Computational Linguistics.
Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tay-
fun Terzi, Eren Sezener, Devang Agrawal, Cyprien
De Masson D’Autume, Tim Scholtes, Manzil Zaheer,
Susannah Young, Ellen Gilsenan-Mcmahon, Sophia
Austin, Phil Blunsom, and Angeliki Lazaridou. 2022.
StreamingQA: A benchmark for adaptation to new
knowledge over time in question answering models.
InProceedings of the 39th International Conference
on Machine Learning, volume 162 ofProceedings
of Machine Learning Research, pages 13604–13622.
PMLR.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. Entity-based knowledge conflicts in question
answering. InProceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pages 7052–7063, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 5303–5315, Singa-
pore. Association for Computational Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Quinn McNemar. 1947. Note on the sampling error
of the difference between correlated proportions or
percentages.Psychometrika, 12(2):153–157.
Meta. 2024. Llama 3.3 model card.
11

Minecraft. 2025. Minecraft snapshot 25w31a. Pub-
lished 2025-07-29; accessed 2026-10-05.
Minecraft Wiki Contributors. 2026. Minecraft wiki.
Accessed 8 September 2026.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zix-
uan Ke, Xuan-Phi Nguyen, Caiming Xiong, and
Shafiq Joty. 2025. FaithEval: Can your language
model stay faithful to context, even if “The Moon Is
Made of Marshmallows”. InInternational Confer-
ence on Learning Representations.
Aashiq Muhamed, Leonardo F. R. Ribeiro, Markus
Dreyer, Virginia Smith, and Mona T. Diab. 2026.
RefusalBench: Generative evaluation of selective re-
fusal in grounded language models. InProceedings
of the 19th Conference of the European Chapter of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 6811–6856, Rabat, Mo-
rocco. Association for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage
re-ranking with BERT.Preprint, arXiv:1901.04085.
NVIDIA. 2025. llama-nemotron-rerank-1b-v2 (llama
nemotron reranking 1b). Model card.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925.
OpenAI. 2026. OpenAI API model pages: GPT-4.1
mini, GPT-4.1 nano, GPT-5 mini, GPT-5 nano. Ac-
cessed 2026-10-06.
Xiangyu Peng, Prafulla Kumar Choubey, Caiming
Xiong, and Chien-Sheng Wu. 2025. Unanswerabil-
ity evaluation for retrieval augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8452–8472, Vienna, Austria.
Association for Computational Linguistics.
Poolside Team. 2026. Introducing Laguna S 2.1.
Qwen Team. 2026a. Qwen3.5: Towards native multi-
modal agents.
Qwen Team. 2026b. Qwen3.8-Max: A new bar for
coding and cowork.
Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.
Know what you don’t know: Unanswerable ques-
tions for SQuAD. InProceedings of the 56th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers), pages 784–789,
Melbourne, Australia. Association for Computational
Linguistics.Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin
Zhao, Jing Liu, Hua Wu, Ji-Rong Wen, and Haifeng
Wang. 2025. Investigating the factual knowledge
boundary of large language models with retrieval
augmentation. InProceedings of the 31st Inter-
national Conference on Computational Linguistics,
pages 3697–3715, Abu Dhabi, UAE. Association for
Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: BM25 and beyond.
Foundations and Trends® in Information Retrieval,
4(1-2):1–174.
Yohanes Andre Setiawan. 2026. Prompt-based ab-
stention fails under misleading context: A con-
trolled study of small frozen RAG models.Preprint,
arXiv:2608.22228.
Andrew L. Smith, Felix Greaves, and Trishan Panch.
2023. Hallucination or confabulation? Neu-
roanatomy as metaphor in large language models.
PLOS Digital Health, 2(11):e0000388.
Maojia Song, Shang Hong Sim, Rishabh Bhardwaj,
Hai Leong Chieu, Navonil Majumder, and Soujanya
Poria. 2025. Measuring and enhancing trustworthi-
ness of LLMs in RAG through grounded attributions
and learning to refuse. InInternational Conference
on Learning Representations.
Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023.
Towards benchmarking and improving the temporal
reasoning capability of large language models. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 14820–14835, Toronto, Canada.
Association for Computational Linguistics.
Team Cohere. 2025. Command A: An enterprise-ready
large language model.Preprint, arXiv:2504.00698.
Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey
Kuehl, David Graham, David Heineman, Dirk Groen-
eveld, Faeze Brahman, Finbarr Timbers, Hamish
Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo,
Luca Soldaini, Matt Jordan, Mayee Chen, Michael
Noukhovitch, Nathan Lambert, Pete Walsh, and 49
others. 2026. Olmo 3.Preprint, arXiv:2512.13961.
Chris Xing Tian, Weihao Xie, Zhen Chen, Hui Liu,
Zhengyuan Yi, Haoliang Li, Shiqi Wang, and Siwei
Ma. 2026. Domain-specific data generation frame-
work for RAG adaptation. InFindings of the Associa-
tion for Computational Linguistics: ACL 2026, pages
19236–19250, San Diego, California, United States.
Association for Computational Linguistics.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry
Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and Thang Luong. 2024. Fresh-
LLMs: Refreshing large language models with search
engine augmentation. InFindings of the Association
for Computational Linguistics: ACL 2024, pages
13697–13720, Bangkok, Thailand. Association for
Computational Linguistics.
12

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-
dlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and
Anima Anandkumar. 2024. V oyager: An open-ended
embodied agent with large language models.Trans-
actions on Machine Learning Research.
Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu,
Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. 2025.
Know your limits: A survey of abstention in large
language models.Transactions of the Association for
Computational Linguistics, 13:529–556.
Frank Wilcoxon. 1945. Individual comparisons by rank-
ing methods.Biometrics Bulletin, 1(6):80–83.
Xiaobao Wu, Liangming Pan, Yuxi Xie, Ruiwen Zhou,
Shuai Zhao, Yubo Ma, Mingzhe Du, Rui Mao,
Anh Tuan Luu, and William Yang Wang. 2025. An-
tiLeakBench: Preventing data contamination by au-
tomatically constructing benchmarks with updated
real-world knowledge. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 18403–
18419, Vienna, Austria. Association for Computa-
tional Linguistics.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2024. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. InThe Twelfth International
Conference on Learning Representations.
Roy Xie, Deepak Gopinath, David Qiu, Dong Lin,
Haitian Sun, Saloni Potdar, and Bhuwan Dhingra.
2026. Over-searching in search-augmented large
language models. InProceedings of the 19th Confer-
ence of the European Chapter of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 7714–7739, Rabat, Morocco. Association for
Computational Linguistics.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for LLMs: A survey. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pages 8541–
8565, Miami, Florida, USA. Association for Compu-
tational Linguistics.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu,
Xipeng Qiu, and Xuanjing Huang. 2023. Do large
language models know what they don’t know? In
Findings of the Association for Computational Lin-
guistics: ACL 2023, pages 8653–8665, Toronto,
Canada. Association for Computational Linguistics.
Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R.
Dalal, Jennifer L. Kim, Michael Moor, Robyn Fong,Curran Phillips, Kevin Alexander, Euan Ashley, Jack
Boyd, Kathleen Boyd, Karen Hirsch, Curt Langlotz,
Rita Lee, Joanna Melia, Joanna Nelson, Karim Sal-
lam, Stacey Tullis, and 3 others. 2024. Almanac—
retrieval-augmented language models for clinical
medicine.NEJM AI, 1(2).
Skylar Zhai, Jingcheng Liang, and Dongyeop Kang.
2026. Abstain-R1: Calibrated abstention and post-
refusal clarification via verifiable RL.arXiv preprint
arXiv:2604.17073.
Hanning Zhang, Shizhe Diao, Yong Lin, Yi Fung, Qing
Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and
Tong Zhang. 2024. R-tuning: Instructing large lan-
guage models to say ‘I don’t know’. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 7113–7139, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
A Additional Results
This section collects the figures and tables support-
ing the claims in the main text.
Results per model.Table 2 reports single-shot
RAG results and Table 3 all three retrieval condi-
tions, for every configuration.
Views of the main findings per model.Figure 8
shows Correct Abstention Rate and Answer Ac-
curacy under all three retrieval conditions for ev-
ery configuration. Figure 9 shows the accuracy
changes from enabling thinking, and Figures 10
and 11 place individual configurations by their An-
swerable Attempt Rate, Correct Abstention Rate
and Answer Accuracy. Figures 12 and 13 repeat the
size comparison of Figure 5 with active parameters
and with Answer Accuracy.
Retrieval conditions.Single-shot RAG raises
CAR for most configurations, and iterative RAG
lowers it again (Figure 14); accuracy rises sharply
with retrieval and changes little with further rounds
(Figure 15). Iterative RAG lowers Correct Absten-
tion Rate at an almost unchanged Answerable At-
tempt Rate (Figure 16, against Figure 10). Every
model and thinking setting under every condition
is in Tables 2 and 3.
13

Model Attempt % CAR % Accuracy %Accuracy when
attempted % Bal. F1
Gemma 4 26B-A4BR95.4 [93.6, 96.7] 80.7 [75.8, 84.7] 69.3 [67.2, 71.4] 72.6 88.9
Gemma 4 26B-A4BR(thinking) 97.9 [96.6, 98.8] 73.7 [68.3, 78.6] 69.6 [67.5, 71.7] 73.6 87.4
Gemma 4 31BV95.7 [93.9, 97.0] 83.3 [78.7, 87.1] 69.5 [67.5, 71.6] 72.690.1
Gemma 4 31BV(thinking) 96.4 [94.8, 97.6] 65.0 [59.4, 70.2] 72.0 [70.1, 74.0] 74.7 83.3
Gemma 3 4B 92.1 [89.9, 93.9] 56.7 [51.0, 62.2] 53.5 [51.0, 56.0] 58.1 78.3
Gemma 3 12B 97.9 [96.5, 98.7] 57.7 [52.0, 63.1] 62.5 [60.2, 64.7] 63.8 81.5
Gemma 3 27B 98.4 [97.2, 99.1] 71.3 [66.0, 76.2] 64.9 [62.6, 67.0] 65.9 86.7
Llama 3.3 70B 97.4 [96.0, 98.4] 73.7 [68.4, 78.3] 68.1 [66.2, 70.3] 69.9 87.1
Command A 111B 98.1 [96.8, 98.9] 69.7 [64.2, 74.6] 68.7 [66.5, 71.0] 70.0 85.9
GLM-4.5-Air 97.6 [96.1, 98.5] 57.7 [52.0, 63.1] 64.8 [62.5, 66.9] 66.4 81.3
GLM-4.5-Air (thinking) 97.4 [95.9, 98.4] 52.8 [47.2, 58.4] 72.4 [70.1, 74.5] 74.8 79.7
OLMo 3 32B Think 95.7 [93.9, 97.0] 56.3 [50.7, 61.8] 62.8 [60.4, 65.3] 65.6 80.0
Qwen3 4B 97.3 [95.8, 98.3] 50.7 [45.0, 56.3] 60.1 [57.6, 62.5] 61.8 78.9
Qwen3 4B (thinking) 98.4 [97.2, 99.1] 62.9 [57.2, 68.2] 64.3 [61.8, 66.5] 66.5 83.6
Qwen3 8B 99.1 [98.1, 99.6] 35.0 [29.8, 40.6] 64.4 [62.3, 66.7] 64.9 75.1
Qwen3 8B (thinking) 98.4 [97.2, 99.1] 63.4 [57.8, 68.7] 68.5 [66.3, 70.7] 70.4 83.8
Qwen3 14B 98.9 [97.8, 99.4] 52.0 [46.4, 57.6] 65.0 [62.7, 67.1] 65.7 80.1
Qwen3 14B (thinking) 97.6 [96.1, 98.5] 59.2 [53.5, 64.6] 67.5 [65.3, 69.8] 69.4 81.9
Qwen3 30B-A3B 99.1 [98.1, 99.6] 44.7 [39.1, 50.3] 64.3 [62.1, 66.4] 64.9 77.9
Qwen3 30B-A3B (thinking) 98.0 [96.7, 98.8] 60.5 [54.9, 65.9] 69.3 [67.0, 71.5] 71.0 82.5
Qwen3 32B 99.0 [98.0, 99.5] 52.3 [46.7, 57.9] 66.2 [64.0, 68.4] 66.9 80.3
Qwen3 32B (thinking) 98.3 [97.0, 99.0] 52.0 [46.4, 57.6] 70.6 [68.3, 72.7] 72.1 79.8
Qwen3 235B-A22B 99.0 [98.0, 99.5] 51.7 [46.0, 57.3] 68.4 [66.2, 70.4] 69.1 80.1
Qwen3 235B-A22B (thinking) 99.0 [98.0, 99.5] 58.0 [52.3, 63.4] 71.9 [69.9, 74.0] 72.6 82.2
Qwen3-Next 80B-A3B 99.4 [98.5, 99.8] 13.0 [9.7, 17.3] 72.0 [69.6, 74.1] 72.4 69.4
Qwen3.5 27B 99.0 [98.0, 99.5] 36.0 [30.8, 41.6]77.2[75.3, 79.1] 78.0 75.3
Qwen3.5 27B (thinking)99.9[99.2, 100.0] 15.6 [11.6, 20.7] 76.3 [74.2, 78.3] 78.6 70.2
Qwen3.5 35B-A3B 98.0 [96.7, 98.8] 46.3 [40.8, 52.0] 75.0 [73.0, 77.0] 76.6 77.9
Qwen3.5 35B-A3B (thinking) 99.4 [98.4, 99.8] 38.9 [31.7, 46.6] 72.6 [70.6, 74.8] 77.8 76.3
Qwen3.5 122B-A10BG98.7 [97.6, 99.3] 41.7 [36.2, 47.3] 75.6 [73.5, 77.8] 76.6 76.8
Qwen3.5 122B-A10BG(thinking) 99.3 [98.3, 99.7] 32.2 [26.3, 38.8] 75.6 [73.7, 77.6] 77.8 74.4
GPT-4.1 mini 99.6 [98.7, 99.9] 33.0 [27.9, 38.5] 72.9 [70.8, 74.9] 73.2 74.7
GPT-4.1 nano 97.4 [96.0, 98.4] 60.0 [54.4, 65.4] 60.8 [58.4, 63.2] 62.4 82.1
GPT-5 mini 97.6 [96.1, 98.5] 70.3 [64.9, 75.2] 73.9 [71.7, 76.0] 75.7 85.9
GPT-5 nano 98.0 [96.7, 98.8] 59.0 [53.4, 64.4] 71.2 [69.0, 73.4] 72.6 82.0
GPT-OSS 120B 98.7 [97.6, 99.3] 60.3 [54.7, 65.7] 71.6 [69.4, 73.6] 72.5 82.8
Gemini 2.5 Flash 78.7 [75.5, 81.6]87.7[83.5, 90.9] 56.8 [54.0, 59.7] 72.2 82.4
Gemini 2.5 Flash (thinking) 96.0 [94.3, 97.2] 73.7 [68.4, 78.3] 69.7 [67.5, 71.7] 72.6 86.4
Gemini 2.5 Flash-Lite 93.3 [91.2, 94.9] 77.7 [72.6, 82.0] 63.2 [61.0, 65.5] 67.7 86.5
Gemini 3 Flash 95.1 [93.3, 96.5] 71.3 [66.0, 76.2] 76.2 [74.3, 78.3]80.185.0
Table 2: Results with single-shot RAG (hybrid retrieval, k=5): one row per model, with a second row, marked
“(thinking)”, for models with switchable thinking.Attempt: Answerable Attempt Rate;CAR: Correct Abstention
Rate;Accuracy when attempted: Answer Accuracy over the answerable questions the model attempted;Bal. F1:
balanced F1, defined in Section I. Brackets are 95% intervals (Wilson for rates, bootstrap over questions for
accuracy). The best value in each column is bold. Superscripts mark models with a role in building or scoring the
benchmark (G generator, V verifier and accuracy judge, R response judge).
How models decline.A model that declines ei-
ther abstains or responds with premise denial, and
the mix differs by family (Figure 17). Premise de-
nials make up 32.4% of responses to unanswerable
questions without retrieval, 13.8% with single-shot
RAG and 17.5% with iterative RAG. Without re-
trieval, 9 of the 14 families give premise denials
more often than abstentions. With single-shot RAG,
only 2 do, Qwen3.5 and Qwen3-Next (each model:
Figure 18), and the other families mostly say they
do not know. Counting premise denials as absten-
tions would therefore change some families’ rates
far more than others’ (Section I).
Leakage.Without retrieval, GPT-OSS 120B ,
whose published knowledge cut-off predates ev-
ery post-cut-off page, answers the unanswerablequestions most accurately (19.8%), followed by
GPT-4.1 (15.5%). These models attempt most of
the questions, and every other model (including
ones without a published knowledge cut-off) is less
accurate. Accuracy here is Answer Accuracy on the
unanswerable questions: the mean of the three ac-
curacy judges’ scores against the reference answers
that the oracle model writes from the Cpostpages,
with an abstention or a premise denial scoring 0.
Figure 19 and Table 4 show it for each model and
family. These answers are best explained as correct
guesses from older, general mechanics rather than
knowledge of the new pages. No model without a
published knowledge cut-off exceeds this reference,
so the audit finds no evidence that these models
learned theC postpages.
14

CAR % Answerable Attempt % Answer Accuracy %
Model No RAG RAG Iter. No RAG RAG Iter. No RAG RAG Iter.
Gemma 4 26B-A4BR15.3 80.7 67.7 88.9 95.4 96.4 26.9 69.3 69.7
Gemma 4 26B-A4BR(thinking) 10.2 73.7 71.2 90.8 97.9 98.2 26.6 69.6 70.6
Gemma 4 31BV6.0 83.3 76.0 95.0 95.7 97.3 33.4 69.5 71.2
Gemma 4 31BV(thinking) 6.7 65.0 60.7 93.2 96.4 97.7 34.0 72.0 74.0
Gemma 3 4B 25.7 56.7 38.7 90.3 92.1 92.9 14.7 53.5 53.2
Gemma 3 12B 9.0 57.7 42.3 96.3 97.9 98.9 18.1 62.5 63.2
Gemma 3 27B 14.0 71.3 61.0 94.6 98.4 98.0 20.6 64.9 64.9
Llama 3.3 70B 17.7 73.7 66.0 94.9 97.4 97.9 26.5 68.1 68.1
Command A 111B 2.0 69.7 65.7 98.6 98.1 97.9 26.0 68.7 70.0
GLM-4.5-Air 17.0 57.7 48.0 97.7 97.6 97.9 26.4 64.8 65.6
GLM-4.5-Air (thinking) 2.8 52.8 40.3 97.0 97.4 98.3 27.7 72.4 72.8
OLMo 3 32B Think 9.3 56.3 24.7 96.3 95.7 98.0 17.3 62.8 64.4
Qwen3 4B 27.3 50.7 40.0 84.3 97.3 97.4 12.3 60.1 60.6
Qwen3 4B (thinking) 13.2 62.9 51.5 89.6 98.4 99.1 12.0 64.3 65.5
Qwen3 8B 23.0 35.0 33.3 92.3 99.1 99.7 17.1 64.4 64.6
Qwen3 8B (thinking) 22.2 63.4 52.7 85.3 98.4 99.0 12.2 68.5 68.2
Qwen3 14B 40.3 52.0 49.7 83.9 98.9 99.3 17.6 65.0 66.1
Qwen3 14B (thinking) 15.6 59.2 53.4 86.5 97.6 98.6 16.5 67.5 69.0
Qwen3 30B-A3B 49.0 44.7 38.3 76.6 99.1 99.1 16.3 64.3 65.0
Qwen3 30B-A3B (thinking) 11.4 60.5 51.7 88.5 98.0 97.9 18.4 69.3 69.7
Qwen3 32B 21.0 52.3 44.0 95.3 99.0 99.0 20.6 66.2 68.3
Qwen3 32B (thinking) 10.5 52.0 45.8 96.1 98.3 98.6 20.3 70.6 70.7
Qwen3 235B-A22B 41.7 51.7 45.7 86.9 99.099.924.1 68.4 68.9
Qwen3 235B-A22B (thinking) 21.7 58.0 45.3 86.9 99.0 99.0 23.3 71.9 72.2
Qwen3-Next 80B-A3B 8.7 13.0 6.0 91.7 99.4 99.3 22.5 72.0 72.4
Qwen3.5 27B 2.3 36.0 19.7 97.3 99.0 99.6 22.477.278.5
Qwen3.5 27B (thinking) 0.5 15.6 12.8 98.299.9 99.926.3 76.379.2
Qwen3.5 35B-A3B 18.7 46.3 37.3 80.0 98.0 99.0 20.1 75.0 75.2
Qwen3.5 35B-A3B (thinking) 3.1 38.9 21.1 96.5 99.4 99.5 18.3 72.6 71.6
Qwen3.5 122B-A10BG3.0 41.7 24.0 98.1 98.7 99.1 28.4 75.6 77.1
Qwen3.5 122B-A10BG(thinking) 3.5 32.2 23.1 96.7 99.3 99.6 29.3 75.6 77.1
GPT-4.1 mini 0.0 33.0 24.799.999.6 99.7 35.0 72.9 73.3
GPT-4.1 nano 11.0 60.0 51.3 95.9 97.4 97.7 23.6 60.8 60.7
GPT-5 mini 43.0 70.3 52.7 76.0 97.6 99.1 37.7 73.9 76.9
GPT-5 nano 42.0 59.0 42.0 71.7 98.0 99.0 28.3 71.2 70.6
GPT-OSS 120B 2.0 60.3 45.0 99.4 98.7 99.4 31.4 71.6 73.7
Gemini 2.5 Flash 45.087.7 84.069.9 78.7 82.0 23.8 56.8 59.4
Gemini 2.5 Flash (thinking) 6.7 73.7 62.0 95.9 96.0 96.4 38.5 69.7 70.8
Gemini 2.5 Flash-Lite56.777.7 70.0 74.7 93.3 94.0 23.9 63.2 63.6
Gemini 3 Flash 3.7 71.3 66.7 97.6 95.1 96.754.176.2 76.4
Table 3: Each model and thinking setting under the three retrieval conditions: Correct Abstention Rate, Answerable
Attempt Rate and Answer Accuracy without retrieval, with single-shot RAG and with iterative RAG. Rows and
marks as in Table 2. The best value in each column is bold.
Family Runs Attempted % Correct % CI low CI high Accuracy when attempted %
GPT-OSS 1 97.7 19.8 16.6 23.0 20.3
GPT-4.1 2 94.5 15.5 13.0 17.9 16.4
GPT-5 2 57.5 15.4 12.7 18.3 26.7
Command A 1 98.0 13.7 11.0 16.5 14.0
GLM-4.5 2 87.2 12.9 10.6 15.1 14.8
Llama 3.3 1 82.3 12.3 9.6 15.2 14.9
Gemma 3 3 83.8 11.5 9.5 13.4 13.7
OLMo 3 1 90.7 11.1 8.7 13.7 12.3
Gemini 3 1 96.3 10.6 8.0 13.2 11.0
Qwen3.5 6 74.5 8.9 7.2 10.9 12.0
Gemma 4 4 86.9 8.6 6.6 10.6 9.9
Gemini 2.5 3 63.9 8.4 6.7 10.2 13.1
Qwen3 12 73.9 7.8 6.4 9.2 10.5
Qwen3-Next 1 91.3 7.6 5.5 9.7 8.3
Table 4: Leakage audit by model family: unanswerable questions answered correctly without retrieval, with 95%
bootstrap intervals over questions.
Accuracy judges.The three accuracy judges
rank runs almost identically but differ in leniency
(how high a judge scores the same answers): their
mean Answer Accuracy differs by up to 12.1%
points (Figure 20; Fleiss’ κ= 0.61 , Table 5), soaccuracy is reported as their mean and orderings
matter more than absolute values. Section A.1 tests
whether a judge favors its own model family.
15

Judges Rows Exact agreement % WeightedκMean diff (pts)
Gemma 4 31B (8-bit) vs Qwen3.8 27B (8-bit) 252,820 68.5 0.730 +10.9
Gemma 4 31B (8-bit) vs Laguna S 2.1 (8-bit) 252,820 85.2 0.888 -1.9
Qwen3.8 27B (8-bit) vs Laguna S 2.1 (8-bit) 252,820 69.0 0.728 -12.8
Table 5: Agreement between the three accuracy judges on the same answers (quadratic-weighted Cohen’s κ; Fleiss’
κacross all three is 0.61).
No
RAGSingle-
shotItera-
tive020406080100Abstention (%)
One model Mean
No
RAGSingle-
shotItera-
tive020406080100Accuracy (%)
Figure 8: Correct Abstention Rate (left) and Answer
Accuracy (right) without retrieval, with single-shot RAG
and with iterative RAG: one gray line per configuration,
mean in blue. Single-shot RAG raises CAR significantly
for 38 of 40 configurations and lowers it for 0; iterative
RAG lowers it for 30 and raises it for 0 ( q <0.05 ).
Accuracy rises with retrieval and barely changes with
further rounds.
0 25Gemini 2.5 FlashQwen3 4BQwen3 30B-A3BQwen3 8BGLM-4.5-AirQwen3 14BQwen3 32BQwen3 235B-A22BGemma 4 26B-A4BGemma 4 31BQwen3.5 35B-A3BQwen3.5 122B-A10BQwen3.5 27BNo retrieval
mean +1.1 points
50 75 100Single-shot RAG
mean +3.4 points
Answer Accuracy (%)Thinking
disabledLower accuracy with
thinking enabledHigher accuracy with
thinking enabled
Figure 9: Answer Accuracy with thinking disabled (gray
dot) and enabled (arrowhead) on the same weights, with-
out retrieval (left) and with single-shot RAG (right), for
the models of Figure 4. Orange: lower with thinking
enabled; blue: higher. Each panel spans 50% points.
80 85 90 95 100
Answerable Attempt Rate (%)020406080100Correct Abstention Rate (%)Gemini 2.5 Flash
Gemini 2.5 Flash-Lite
Gemma 3 4B
96 97 98 99 10020406080Gemma 4 31B
Gemma 4 26B-A4B
Qwen3.5 27B
Gemma 4
Llama 3.3
Qwen3
Qwen3-NextQwen3.5
Command A
Gemma 3
GLM-4.5OLMo 3
GPT-4.1
GPT-5
GPT-OSSGemini 2.5
Gemini 3
Thinking disabled
Thinking enabledFigure 10: Answerable Attempt Rate against Correct
Abstention Rate for every configuration, single-shot
RAG. Colors mark model families; squares are runs with
thinking enabled, circles runs with thinking disabled or
from instruct-only models. Bars are 95% Wilson inter-
vals. The inset magnifies the crowded high-coverage
region (without error bars); labels mark each family’s
highest Correct Abstention Rate and every configuration
that no other configuration exceeds on both rates. At-
tempt rates span 78.7–99.9%; CAR spans 13.0–87.7%.
The best performance is the top-right corner.
A.1 Do Accuracy Judges Have a
Self-Preference?
Two of the three accuracy judges come from model
families we also evaluate: Gemma 4 31B (Gemma)
andQwen3.8 27B (Qwen). A judge with a self-
preference would score answers from its own fam-
ily higher than the other judges do. For each of
these two judges, we take the difference between
its score and the mean score of the other two judges
on the same answer. Its self-preference is the mean
difference on answers from models of its own fam-
ily minus the mean difference on answers from all
other models, with a 95% bootstrap interval over
questions.
Both judges show a small self-preference: +1.0%
points for Gemma 4 31B (+0.5 to +1.4) and +1.8%
16

80 85 90 95 100
Answerable Attempt Rate (%)556065707580Answer Accuracy (%)Gemini 2.5 Flash-Lite
Gemini 2.5 Flash
Gemma 3 4B96 97 98 99 10060657075Qwen3.5 27B
Gemini 3 Flash
Gemma 4
Llama 3.3
Qwen3
Qwen3-NextQwen3.5
Command A
Gemma 3
GLM-4.5OLMo 3
GPT-4.1
GPT-5
GPT-OSSGemini 2.5
Gemini 3
Thinking disabled
Thinking enabledFigure 11: Answerable Attempt Rate against Answer
Accuracy (mean of the three accuracy judges), single-
shot RAG. Markers and inset as in Figure 10. Bars are
95% intervals: Wilson for the attempt rate, bootstrap
over questions for accuracy.
5B 10B 20B 50B 100B
Active parameters (log)020406080100Correct Abstention Rate (%)
  = +0.21 (p = 0.37), 20 models
Figure 12: Correct Abstention Rate versus active param-
eters across the 20 open-weight models, with single-shot
RAG; one point per model as in Figure 5. The Spearman
correlation is not significant at the 5% level.
5B 10B 20B 50B 100B 200B
T otal parameters (log)020406080100Answer Accuracy (%)
 = +0.60 (p = 0.005), 20 models
Figure 13: Answer Accuracy versus total parameters
across the 20 open-weight models, with single-shot
RAG; one point per model as in Figure 5. Unlike Cor-
rect Abstention Rate, Answer Accuracy rises with size
(Spearmanρ= +0.60,p= 0.005).
points for Qwen3.8 27B (+1.4 to +2.2). Both are
small next to the judges’ differences in leniency,
a judge’s tendency to score every answer higher
or lower than the other judges do, whatever model
wrote it. Against the mean of the three judges,
Gemma 4 31B scores +3.0, Laguna S 2.1 +4.9 and
Qwen3.8 27B -7.9% points. Because Answer Ac-
curacy averages the three judges, a self-preference
enters it at about a third of its size.
As a further check, we recompute each configu-
ration’s Answer Accuracy using only the accuracy
judges from other families. The ranking of the
40 configurations with single-shot RAG then has
a Spearman correlation of 0.84 with the original
ranking. The change comes mostly from leniency
rather than self-preference. Dropping Qwen3.8
27B, the least lenient judge, changes the accuracy
of Qwen3, Qwen3-Next and Qwen3.5 models by
+3.5, +3.0 and +3.6% points on average, and drop-
ping Gemma 4 31B , which scores above the mean,
changes Gemma 3 and Gemma 4 models by -1.2
and -2.1% points.
A.2 Ablation: Increasing Retrieval Depth
Lowers Correct Abstention Rate
To compare a larger single retrieval with iterative
RAG, we evaluate the 11 models with single-shot
runs at both k=5 andk=10 (Figure 2). We dis-
able thinking where switchable and use the other
models’ defaults. On this subset, mean Correct
Abstention Rate is 12.2% without retrieval, 56.8%
atk=5, 51.7% at k=10 and 44.9% with iterative
RAG. Iterative RAG yields a lower Correct Absten-
tion Rate than single-shot RAG at k=5 for all 11
17

0 20 40 60 80 100
Correct Abstention Rate (%)Gemini 2.5 Flash-Lite API
Gemma 4 26B-A4B 8-bit
Gemini 2.5 Flash API
Llama 3.3 70B 8-bit
Gemma 3 27B 8-bit
Gemini 3 Flash API
GPT-5 mini API
Command A 111B 8-bit
Gemma 4 31B 8-bit
Qwen3 8B 8-bit
Qwen3 4B 8-bit
Qwen3 30B-A3B 8-bit
GPT-OSS 120B API
GPT-4.1 nano API
Qwen3 14B 8-bit
GPT-5 nano API
Qwen3 235B-A22B 4-bit
Gemma 3 12B 8-bit
Gemma 3 4B 8-bit
OLMo 3 32B Think 8-bit
GLM-4.5-Air 8-bit
Qwen3 32B 8-bit
Qwen3.5 35B-A3B 8-bit
GPT-4.1 mini API
Qwen3.5 122B-A10B 8-bit
Qwen3.5 27B 8-bit
Qwen3-Next 80B-A3B 8-bit
Correct abstention by retrieval condition (T=0)
No RAG RAG Iterative RAGFigure 14: Correct Abstention Rate per model without retrieval (No RAG), with single-shot RAG and with iterative
RAG, thinking enabled or instruct-only. Across all configurations, single-shot RAG raises CAR for 38 of 40 and
lowers it for 0; iterative RAG lowers it for 30 of 40 relative to single-shot RAG ( q <0.05 ), by -10.0% points on
average.
models. Doubling kalso lowers the mean, so the
results do not support attributing the iterative de-
crease solely to follow-up query selection: chunk
count and context length change together, and it-
erative RAG additionally changes the queries and
introduces self-reflection.A separate sweep over 8 models covers k∈
{1,2,3,5,10} (Figure 21 and table 6). In that sub-
set, moving from five to ten chunks lowers mean
Correct Abstention Rate from 52.4% to 49.9%,
while mean Answer Accuracy rises from 70.4%
to 72.2%. These means describe the smaller sweep,
18

0 20 40 60 80 100
Answer Accuracy (%)Qwen3.5 27B 8-bit
Gemini 3 Flash API
Qwen3.5 122B-A10B 8-bit
GPT-5 mini API
GPT-4.1 mini API
Qwen3.5 35B-A3B 8-bit
GLM-4.5-Air 8-bit
Gemma 4 31B 8-bit
Qwen3-Next 80B-A3B 8-bit
Qwen3 235B-A22B 4-bit
GPT-OSS 120B API
GPT-5 nano API
Qwen3 32B 8-bit
Gemini 2.5 Flash API
Gemma 4 26B-A4B 8-bit
Qwen3 30B-A3B 8-bit
Command A 111B 8-bit
Qwen3 8B 8-bit
Llama 3.3 70B 8-bit
Qwen3 14B 8-bit
Gemma 3 27B 8-bit
Qwen3 4B 8-bit
Gemini 2.5 Flash-Lite API
OLMo 3 32B Think 8-bit
Gemma 3 12B 8-bit
GPT-4.1 nano API
Gemma 3 4B 8-bit
Answer accuracy by retrieval condition (T=0)
No RAG RAG Iterative RAGFigure 15: Answer Accuracy per model under the three retrieval conditions, thinking enabled or instruct-only. Mean
accuracy is 24.3% without retrieval, 68.5% with single-shot RAG and 69.3% with iterative RAG.
not the 11-model comparison above.
A.3 Ablation: Query Rewriting and
Reranking Lower Correct Abstention
Rate
We compare six pipelines under single-shot RAG
with five retrieved chunks, holding the evaluated
models and their thinking settings fixed. Ap-pendix E specifies the retrievers, query rewrites and
reranker. For the 12 models in Figure 3, mean Cor-
rect Abstention Rate is 59.4% with BM25, 52.2%
with dense retrieval, 54.7% with the default hy-
brid retriever, 52.5% with hybrid plus reranking,
52.5% with hybrid plus query rewriting and 49.8%
with both additions. The combined pipeline yields
a lower Correct Abstention Rate than the default
19

80 85 90 95 100
Answerable Attempt Rate (%)020406080100Correct Abstention Rate (%)Gemini 2.5 Flash
Gemini 2.5 Flash-Lite
Gemma 4 26B-A4B
Gemini 2.5 Flash
Gemma 3 4B
97 98 99 100204060Gemma 4 31B
Llama 3.3 70B
Gemma 4
Llama 3.3
Qwen3
Qwen3-Next
Qwen3.5Command A
Gemma 3
GLM-4.5
OLMo 3GPT-4.1
GPT-5
GPT-OSS
Gemini 2.5Gemini 3
Thinking disabled
Self-RAG attempts = 2
Thinking enabledFigure 16: As Figure 10, with iterative RAG (up to two
further retrieval rounds).
Model k CAR % Accuracy % Bal. F1
Command A 111B 178.053.8 82.2
Command A 111B 2 70.7 63.1 84.0
Command A 111B 3 72.7 66.9 86.3
Command A 111B 5 71.769.3 86.6
Command A 111B 10 70.7 68.8 86.1
GLM-4.5-Air 158.753.5 78.1
GLM-4.5-Air 2 58.0 60.1 80.3
GLM-4.5-Air 3 55.0 63.7 80.2
GLM-4.5-Air 5 54.7 64.980.5
GLM-4.5-Air 10 51.067.879.4
Gemma 4 31B 1 82.7 53.0 79.8
Gemma 4 31B 2 82.7 63.5 86.7
Gemma 4 31B 383.767.3 88.8
Gemma 4 31B 5 82.7 69.989.9
Gemma 4 31B 10 79.772.389.8
Llama 3.3 70B 181.750.2 79.6
Llama 3.3 70B 2 80.7 61.0 86.1
Llama 3.3 70B 3 78.0 65.5 87.4
Llama 3.3 70B 5 75.3 67.987.6
Llama 3.3 70B 10 69.068.085.7
Qwen3 32B 1 56.0 54.3 77.9
Qwen3 32B 2 52.7 62.3 79.7
Qwen3 32B 3 56.0 64.8 81.1
Qwen3 32B 5 53.0 67.0 80.5
Qwen3 32B 1056.7 68.7 81.6
Qwen3-Next 80B-A3B 118.357.5 68.5
Qwen3-Next 80B-A3B 2 15.7 67.269.7
Qwen3-Next 80B-A3B 3 13.7 69.8 69.2
Qwen3-Next 80B-A3B 5 12.0 71.5 69.1
Qwen3-Next 80B-A3B 10 9.373.768.5
Qwen3.5 122B-A10B 148.756.5 71.9
Qwen3.5 122B-A10B 2 43.3 68.4 75.8
Qwen3.5 122B-A10B 3 41.3 72.576.1
Qwen3.5 122B-A10B 5 36.0 75.2 74.9
Qwen3.5 122B-A10B 10 36.378.275.6
Qwen3.5 27B 1 43.7 58.5 72.2
Qwen3.5 27B 244.370.276.4
Qwen3.5 27B 3 38.3 74.8 75.3
Qwen3.5 27B 5 33.7 77.6 74.6
Qwen3.5 27B 10 26.379.972.9
Table 6: ksweep per model (hybrid retrieval, single-
shot RAG; the models of Figure 21).Bal. F1: balanced
F1 (Section I). Bold: the best kfor each model in each
column. The k=5 rows are the hybrid runs of Table 7,
not the runs in Table 2.
Gemini 2.5
Gemma 4
Llama 3.3
Gemini 3
Command A
GPT-5
Gemma 3
GPT-OSS
OLMo 3
GLM-4.5
Qwen3
GPT-4.1
Qwen3.5
Qwen3-Next36
10
18
42
16
9
10
2532
57
17
67
31
16
20
19
24
23
58
6232
33
66
30
67
42
78
90
70
71
51
71
37
29No RAG
Gemini 2.5
Gemma 4
Llama 3.3
Gemini 3
Command A
GPT-5
Gemma 3
GPT-OSS
OLMo 3
GLM-4.5
Qwen3
GPT-4.1
Qwen3.5
Qwen3-Next80
76
74
71
70
65
62
60
56
55
54
46
35
139
14
9
10
10
39
6016
15
24
15
28
29
37
36
35
35
36
45
26
27RAG
0 20 40 60 80 100
% of unanswerable questionsGemini 2.5
Gemma 4
Llama 3.3
Gemini 3
Command A
GPT-5
Gemma 3
GPT-OSS
OLMo 3
GLM-4.5
Qwen3
GPT-4.1
Qwen3.5
Qwen3-Next72
69
66
67
66
47
47
45
25
44
46
38
2312
12
11
23
16
13
10
46
6421
19
29
22
31
42
52
48
52
39
41
52
31
30Iterative RAGAbstains Denies the item exists AnswersFigure 17: What each model family does with unan-
swerable questions, by retrieval condition: abstention,
premise denial or attempted answer. Family means over
configurations.
20

0 20 40 60 80 100
Share of unanswerable answers (%)Qwen3 4B
Qwen3 4B (thinking)
Qwen3 8B
Qwen3 8B (thinking)
Qwen3 14B
Qwen3 14B (thinking)
Qwen3 30B-A3B
Qwen3 30B-A3B (thinking)
Qwen3 32B
Qwen3 32B (thinking)
Qwen3 235B-A22B
Qwen3 235B-A22B (thinking)
Qwen3-Next 80B-A3B
Qwen3.5 27B
Qwen3.5 27B (thinking)
Qwen3.5 35B-A3B
Qwen3.5 35B-A3B (thinking)
Qwen3.5 122B-A10B
Qwen3.5 122B-A10B (thinking)
Gemma 3 4B
Gemma 3 12B
Gemma 3 27B
Gemma 4 26B-A4B
Gemma 4 26B-A4B (thinking)
Gemma 4 31B
Gemma 4 31B (thinking)
Llama 3.3 70B
Command A 111B
GLM-4.5-Air
GLM-4.5-Air (thinking)
OLMo 3 32B Think (thinking)
GPT-4.1 nano
GPT-4.1 mini
GPT-5 nano (thinking)
GPT-5 mini (thinking)
GPT-OSS 120B (thinking)
Gemini 2.5 Flash-Lite
Gemini 2.5 Flash
Gemini 2.5 Flash (thinking)
Gemini 3 Flash (thinking)Abstains Denies the item exists AnswersFigure 18: What each model does with the unan-
swerable questions under single-shot RAG: abstention,
premise denial or attempted answer. Families together;
within a family, smallest model first.
hybrid retriever for all 12 models. These results
concern the tested implementations; pipeline com-
plexity does not establish that one retriever supplies
better evidence.
Table 7 gives every tested model and Figure 22
plots both Correct Abstention Rate and Answer
Accuracy. Over the 12 models in that analysis, the
mean Correct Abstention Rate falls from 54.7%
with hybrid retrieval to 49.8% with both rewrit-
ing and reranking, while mean Answer Accuracy
changes from 70.1% to 71.4%. Only 3 of the 60
paired comparisons of alternative pipelines against
hybrid retrieval are significant. The all-model direc-
tion of the combined-pipeline result above should
therefore not be read as a significant decrease for
0 10 20
Accuracy, no RAG (%)Qwen3 4BQwen3 30B-A3BQwen3.5 35B-A3BQwen3-Next 80B-A3BQwen3 14BGemini 2.5 FlashGemma 4 26B-A4BQwen3 8BQwen3 235B-A22BQwen3 32BGemma 4 31BGemini 2.5 Flash-LiteQwen3.5 27BGemma 3 4BQwen3.5 122B-A10BGemini 3 FlashOLMo 3 32B ThinkGemma 3 12BLlama 3.3 70BGPT-4.1 nanoGemma 3 27BGLM-4.5-AirCommand A 111BGPT-5 miniGPT-5 nanoGPT-4.1 miniGPT-OSS 120B
GPT-OSS 120B
(cutoff
Jun 2024)
0 50 100
Attempted (%)Open-weight ClosedFigure 19: Leakage audit per model: Answer Accuracy
on the unanswerable questions without retrieval (mean
of the three accuracy judges against the oracle model’s
reference answers from Cpost, an abstention scoring 0),
with 95% bootstrap intervals over questions, next to
each model’s attempt rate. The dashed line is GPT-OSS
120B , whose published knowledge cut-off (June 2024)
predates every post-cut-off page.
every individual model.
A.4 Ablation: Scenarios Make Abstention
Harder
Each question opens with a player scenario written
from Cprepages (Section 3.2). To test whether it
matters, we reran 10 models without it: the ques-
tion alone serves as both the retrieval query and
the model’s input. Without the scenario, Correct
Abstention Rate rises from 50.9% to 57.4%, for
all 10 models, while Answerable Attempt Rate is
almost unchanged (Figure 23). A benchmark of
bare questions would therefore overstate how often
models abstain. With hybrid retrieval, 71.3% of
unanswerable questions also retrieve a passage the
scenario was written from (Figure 24).
21

Model Thinking Retrieval CAR % Accuracy % Bal. F1
Command A 111B no Hybrid (default) 71.7 69.3 86.6
Command A 111B no Dense 68.3 69.5 85.7
Command A 111B no BM2577.364.887.0
Command A 111B no Hybrid + rerank 67.3 70.7 85.2
Command A 111B no Hybrid + rewrite 71.7 68.5 86.2
Command A 111B no Hybrid + rewrite + rerank 67.071.284.9
GLM-4.5-Air no Hybrid (default) 54.7 64.9 80.5
GLM-4.5-Air no Dense 45.7 66.7 78.0
GLM-4.5-Air no BM2560.362.081.3
GLM-4.5-Air no Hybrid + rerank 51.367.280.0
GLM-4.5-Air no Hybrid + rewrite 52.0 65.9 79.9
GLM-4.5-Air no Hybrid + rewrite + rerank 46.7 66.5 78.4
GPT-5 mini yes Hybrid (default) 72.0 74.686.3
GPT-5 mini yes Dense 67.0 74.9 84.6
GPT-5 mini yes BM2576.369.2 85.1
GPT-5 mini yes Hybrid + rerank 67.3 75.8 85.0
GPT-5 mini yes Hybrid + rewrite 66.3 74.7 84.5
GPT-5 mini yes Hybrid + rewrite + rerank 63.076.483.5
Gemini 3 Flash yes Hybrid (default) 75.3 75.0 86.5
Gemini 3 Flash yes Dense 74.377.087.2
Gemini 3 Flash yes BM2579.370.5 85.7
Gemini 3 Flash yes Hybrid + rerank 74.377.0 87.3
Gemini 3 Flash yes Hybrid + rewrite 74.3 74.1 86.4
Gemini 3 Flash yes Hybrid + rewrite + rerank 72.7 76.5 86.0
Gemma 4 31B no Hybrid (default) 82.7 69.9 89.9
Gemma 4 31B no Dense 84.3 70.5 91.1
Gemma 4 31B no BM2590.064.4 89.7
Gemma 4 31B no Hybrid + rerank 83.7 72.091.4
Gemma 4 31B no Hybrid + rewrite 82.0 69.3 89.6
Gemma 4 31B no Hybrid + rewrite + rerank 81.072.690.3
Llama 3.3 70B no Hybrid (default) 75.3 67.987.6
Llama 3.3 70B no Dense 73.068.586.8
Llama 3.3 70B no BM2579.062.0 86.4
Llama 3.3 70B no Hybrid + rerank 73.0 68.1 87.1
Llama 3.3 70B no Hybrid + rewrite 71.3 66.9 85.7
Llama 3.3 70B no Hybrid + rewrite + rerank 71.0 68.4 86.3
OLMo 3 32B Think yes Hybrid (default) 52.7 63.7 79.0
OLMo 3 32B Think yes Dense 49.7 63.6 78.8
OLMo 3 32B Think yes BM2558.759.079.4
OLMo 3 32B Think yes Hybrid + rerank 50.764.578.8
OLMo 3 32B Think yes Hybrid + rewrite 48.0 61.8 78.0
OLMo 3 32B Think yes Hybrid + rewrite + rerank 49.0 63.5 78.2
Qwen3 32B no Hybrid (default) 53.0 67.0 80.5
Qwen3 32B no Dense 49.7 68.3 79.6
Qwen3 32B no BM2557.062.980.8
Qwen3 32B no Hybrid + rerank 49.068.579.3
Qwen3 32B no Hybrid + rewrite 54.3 66.4 80.6
Qwen3 32B no Hybrid + rewrite + rerank 46.3 68.4 78.4
Qwen3 8B no Hybrid (default) 37.065.1 75.9
Qwen3 8B no Dense 37.0 64.0 75.7
Qwen3 8B no BM2538.760.3 75.4
Qwen3 8B no Hybrid + rerank 33.7 64.8 74.8
Qwen3 8B no Hybrid + rewrite 33.7 63.4 74.8
Qwen3 8B no Hybrid + rewrite + rerank 32.0 64.2 74.4
Qwen3-Next 80B-A3B no Hybrid (default) 12.0 71.5 69.1
Qwen3-Next 80B-A3B no Dense 10.7 71.7 68.9
Qwen3-Next 80B-A3B no BM2515.067.769.7
Qwen3-Next 80B-A3B no Hybrid + rerank 10.7 72.9 68.9
Qwen3-Next 80B-A3B no Hybrid + rewrite 13.3 71.7 69.3
Qwen3-Next 80B-A3B no Hybrid + rewrite + rerank 11.373.368.9
Qwen3.5 122B-A10B no Hybrid (default) 36.0 75.2 74.9
Qwen3.5 122B-A10B no Dense 33.0 75.3 74.2
Qwen3.5 122B-A10B no BM2540.771.975.3
Qwen3.5 122B-A10B no Hybrid + rerank 34.7 76.5 75.2
Qwen3.5 122B-A10B no Hybrid + rewrite 31.3 74.4 73.8
Qwen3.5 122B-A10B no Hybrid + rewrite + rerank 29.777.073.5
Qwen3.5 27B no Hybrid (default) 33.7 77.6 74.6
Qwen3.5 27B no Dense 34.3 77.5 75.1
Qwen3.5 27B no BM2540.373.575.4
Qwen3.5 27B no Hybrid + rerank 34.079.675.0
Qwen3.5 27B no Hybrid + rewrite 32.0 75.8 74.2
Qwen3.5 27B no Hybrid + rewrite + rerank 27.7 79.5 73.2
Table 7: Retrieval ablation per model (single-shot RAG, k=5).Thinking: whether thinking is enabled.Bal. F1:
balanced F1 (Section I). Bold: the best pipeline for each model in each column. TheHybrid (default)rows are a
second run of the configurations in Table 2, with a different set of responses.
22

50 100
Gemma 4 31B (%)20406080100Qwen3.8 27B (%)
 = 0.99, 381 runs
50 100
Gemma 4 31B (%)20406080100Laguna S 2.1 (%)
 = 1.00, 381 runs
50 100
Qwen3.8 27B (%)20406080100Laguna S 2.1 (%)
 = 0.98, 381 runs
Figure 20: Run-level Answer Accuracy under each pair of accuracy judges (Spearman ρabove each panel). The
judges rank runs almost identically, but their mean Answer Accuracy differs by up to 12.1% points.
20406080100Retrieved (%)
Retrieval only (hybrid)
Answerable: target chunk
Unanswerable: scenario chunk
020406080100Correct abstention (%)
Correct abstention
One model
Mean over models
1 2 3 5 10
k (chunks retrieved)50607080Answer accuracy (%)
Answer accuracy
Figure 21: Choice of k(hybrid retrieval, single-shot
RAG). Top: retrieval alone, the share of answerable
questions whose target chunk is retrieved and of unan-
swerable questions for which a chunk the scenario was
written from is retrieved. Middle and bottom: Correct
Abstention Rate and Answer Accuracy for 8 models run
at every k(gray; thinking disabled where switchable)
and their mean (blue). From k=5 tok=10 , CAR falls
from 52.4% to 49.9% while accuracy moves from 70.4%
to 72.2%.
0 25 50 75 100
Abstention (%)Hybrid (default)
Dense
BM25
Hybrid + rerank
Hybrid + rewrite
Hybrid + rewrite + rerank
One model Mean over models
60 70 80
Accuracy (%)
Figure 22: Retrieval ablation: Correct Abstention Rate
(left) and Answer Accuracy (right) for 12 models under
six retrieval pipelines (single-shot RAG, k=5; thinking
disabled where switchable, OLMo 3 32B Think and
closed models at their defaults). Gray: individual mod-
els; blue: their mean. Mean Correct Abstention Rate
is 54.7% with the default hybrid retriever, 52.5% with
reranking and 49.8% with query rewriting and rerank-
ing; 3 of 60 model ×pipeline comparisons with hybrid
are significant.
A.5 Thinking: Family Differences and
Response Categories
The paired comparison in Section 4.4 uses the 13
models with switchable thinking, with the same
weights and questions in each pair. Without re-
trieval, enabling thinking lowers mean Correct Ab-
stention Rate by 14% points (Wilcoxon signed-rank
test over per-model differences, p= 0.001 ). With
single-shot RAG, the mean change is −0.9 % points
(p= 0.68 , Wilcoxon), providing no evidence of an
overall gain. Section 4.4 reports the counts of sta-
tistically significant changes for individual models.
Under single-shot RAG, the 6 Qwen3 models
have a mean change in Correct Abstention Rate
of +11.6% points when thinking is enabled, ver-
sus -11.6% points for the other 7 models. Without
23

With
scenarioQuestion
alone20406080Correct Abstention Rate (%)
With
scenarioQuestion
alone949596979899100Answerable Attempt Rate (%)With scenario: scenario + question for retrieval and model (default)
Question alone: for retrieval and modelFigure 23: The scenario makes abstention harder. Cor-
rect Abstention Rate and Answerable Attempt Rate for
10 models (single-shot RAG, k=5) with the scenario
(blue: scenario and question for retrieval and the model;
the default) and without it (orange: the question alone
for both). Points: mean with a 95% confidence interval
over models; gray lines: one model. Each panel has its
own scale.
BM25 Dense Hybrid Hybrid
+rerankHybrid
+rewriteHybrid
+rerank
+rewrite
Retrieval pipeline (single-shot, k = 5)020406080100Unanswerable questions (%)81.7
49.771.382.7
30.357.7
Figure 24: Share of unanswerable questions whose five
retrieved chunks include a Cprepassage the scenario was
written from, by retrieval pipeline (retrieval only).
retrieval, the corresponding changes are -18.0 and
-10.5% points. These are descriptive family aver-
ages.
What happens to abstentions when thinking is
enabled.Each of the 13 models with switchable
thinking answered every unanswerable question
twice with the same weights, thinking disabled
and enabled. Each answer carries the response
judge’s label (abstain, deny, answer). For each
model and retrieval condition we cross-tabulate the
two labels of each question, leaving out questions
where either answer exhausted the token budget.
The two discordant cells abstain →deny (A →D)
and deny →abstain (D →A) are compared with anexact McNemar test: if thinking moved answers
between the two labels equally often in both di-
rections, A →D would follow a binomial distribu-
tion with n= A→D + D→A trials and probabil-
ity1
2. The same test compares abstain →answer
with answer →abstain (A →X, X→A), so that a loss
of abstention through guessing is not counted as
premise denial. Table 8 and Figure 25 report the re-
sults, with significance assessed as in Appendix H.
Without retrieval, thinking turns 372 abstentions
into premise denials and 51 premise denials into
abstentions (3,404 matched questions). The ex-
cess of A→D is significant for 9 of the 13 models,
and no model shows a significant shift the other
way; across models, the per-model net shift (A →D
−D→A, as a share of questions) averages +8.6
points (Wilcoxon signed-rank, p= 0.001 ). Of the
abstentions lost to thinking, 54.6% became premise
denials and the rest attempted answers; the A →X
excess is significant for 5 models. The Qwen3.5
models rarely abstain without retrieval in either
mode, so they contribute few pairs. With single-
shot RAG the shift towards premise denial remains
(315 against 133; significant for 6 models, the re-
verse for 1), but for 5 models, mostly Qwen3, think-
ing also turns attempted answers into abstentions,
which is why its net effect on Correct Abstention
Rate with retrieval is mixed.
A premise denial, including one that follows
the abstention sentence requested by the prompt,
does not acknowledge that the premise cannot be
verified from the model’s available knowledge (Ap-
pendix I). AbstentionBench also reports lower ab-
stention after reasoning fine-tuning, under its sepa-
rate evaluation setting (Kirichenko et al., 2025).
For the 13 models with switchable thinking,
mean Answer Accuracy changes by +1.1% points
without retrieval and +3.4% points with single-shot
RAG. Figure 9 shows these differences per model.
Thus, the change in answer quality need not have
the same direction as the change in Correct Absten-
tion Rate.
A.6 Answer Accuracy and Correct
Abstention Rate
Under single-shot RAG, the Spearman correlation
between Correct Abstention Rate and Answer Ac-
curacy is ρ=−0.28 (p= 0.08 ) across the 40
model and thinking configurations (Figure 26).
For Accuracy when attempted, the correlation is
ρ=−0.07 (p= 0.68 ). Neither unadjusted associ-
ation is significant at the 5% level. Figure 27 plots
24

No RAG Single-shot RAG
ModelnA→D / D→A A→X / X→AnA→D / D→A A→X / X→A
Gemma 4 26B-A4B 25427 / 7* 4 / 8 27825 / 2* 7 / 8
Gemma 4 31B 299 10 / 23 / 13* 30058 / 3* 11 / 11
GLM-4.5-Air 28212 / 0*33 / 6* 299 23 / 10 28 / 26
Qwen3 4B 27230 / 12*34 / 11* 291 14 / 820 / 58*
Qwen3 8B 28422 / 4* 24 / 38 2982 / 21*4 / 69*
Qwen3 14B 29555 / 4*37 / 15* 29926 / 6*4 / 45*
Qwen3 30B-A3B 29966 / 2*52 / 3* 299 14 / 214 / 44*
Qwen3 32B 29428 / 4* 22 / 14 300 26 / 12 14 / 27
Qwen3 235B-A22B 30057 / 6* 28 / 19 300 19 / 1315 / 40*
Qwen3.5 27B 194 0 / 1 1 / 0 24440 / 7* 11 / 5
Qwen3.5 35B-A3B 129 8 / 2 4 / 1 162 14 / 12 7 / 8
Qwen3.5 122B-A10B 202 1 / 2 2 / 3 21430 / 13* 3 / 9
Gemini 2.5 Flash 30056 / 5*65 / 1* 30024 / 5*28 / 5*
Table 8: Matched pairs with thinking disabled and enabled, per model: nquestions (budget-exhausted pairs left out)
and the discordant counts abstain →deny / deny →abstain and abstain →answer / answer →abstain. Bold with *: the
two counts differ significantly (q <0.05).
0 20
% of questionsQwen3 30B-A3B
Qwen3 14B
Gemini 2.5 Flash
Qwen3 235B-A22B
Qwen3 32B
Gemma 4 26B-A4B
Qwen3 4B
Qwen3 8B
Qwen3.5 35B-A3B
GLM-4.5-Air
Gemma 4 31B
Qwen3.5 122B-A10B
Qwen3.5 27B*
*
*
*
*
*
*
*
*No RAG
10
 0 10 20
% of questions*
*
*
*
*
*
*Single-shot RAGAbstain  deny
Deny  abstain
Figure 25: Matched pairs with thinking disabled and
enabled on the unanswerable questions: the share of
questions whose response changes from abstention to
premise denial when thinking is enabled (orange) and
from premise denial to abstention (blue), per model.
Faded: the two counts do not differ significantly; *:
they do ( q <0.05 ). Models sorted by the shift without
retrieval.Accuracy when attempted against Correct Absten-
tion Rate for each configuration.
60 70 80 90 100
Answer Accuracy (%)020406080100Correct Abstention Rate (%)Gemini 2.5 Flash
Gemma 4 31B
Gemma 4 26B-A4B
Gemini 2.5 Flash-Lite
Gemma 3 27B Gemini 3 Flash
GPT-5 mini Command A 111B
Gemma 4 31B
Qwen3 235B-A22B
GLM-4.5-Air
Qwen3.5 35B-A3B
Qwen3 30B-A3B Qwen3.5 122B-A10B
Qwen3.5 27BQwen3 8B
GPT-4.1 miniQwen3.5 122B-A10B
Qwen3.5 27B
Qwen3-Next 80B-A3B
Gemma 4
Llama 3.3
Qwen3
Qwen3-NextQwen3.5
Command A
Gemma 3
GLM-4.5OLMo 3
GPT-4.1
GPT-5
GPT-OSSGemini 2.5
Gemini 3
Thinking disabled
Thinking enabled
Figure 26: Correct Abstention Rate versus Answer Ac-
curacy under single-shot RAG, one point per model and
thinking setting. Colors: families; squares: thinking
enabled. Bars: 95% intervals, bootstrap over questions
for accuracy and Wilson for Correct Abstention Rate.
Appendix A.6 reports the correlations.
Across the same 40 configurations, release
month correlates with Answer Accuracy ( ρ=
+0.60 ), but not with Correct Abstention Rate ( ρ=
−0.03 ). Partialling out release month gives a nega-
tive association between Correct Abstention Rate
and Answer Accuracy ( ρ=−0.33 ,p= 0.04 ).
These cross-model comparisons do not identify the
accuracy cost of changing an individual model’s
decision to answer: they compare different systems,
25

0 20 40 60 80 100
Correct Abstention Rate (%)6065707580Accuracy when attempted (%)Gemini 3 Flash
Qwen3.5 27BQwen3.5 27B
Qwen3.5 35B-A3B
Qwen3.5 122B-A10BQwen3.5 35B-A3B
GPT-5 mini
GLM-4.5-AirGemma 4 31B
GPT-4.1 miniGPT-OSS 120BQwen3 32B
Llama 3.3 70BQwen3 14B
Qwen3 235B-A22B
Qwen3 32B
Gemma 3 27BQwen3 14B
Qwen3 8B Gemma 3 12B
GPT-4.1 nano
Qwen3 4B
Gemma 3 4B
Gemma 4
Llama 3.3
Qwen3
Qwen3-NextQwen3.5
Command A
Gemma 3
GLM-4.5OLMo 3
GPT-4.1
GPT-5
GPT-OSSGemini 2.5
Gemini 3
Thinking disabled
Thinking enabledFigure 27: Accuracy when attempted (over the an-
swerable questions a model attempted, abstentions left
out) against Correct Abstention Rate, single-shot RAG.
Spearmanρ=−0.07.
and thinking configurations of the same model are
not independent models. They therefore establish
neither an inevitable accuracy cost nor that correct
abstention is cost-free.
A.7 Answerable Attempt Rate and Correct
Abstention Rate
The trade-off frontier of a set of configurations is
the highest Correct Abstention Rate observed at or
above a given Answerable Attempt Rate (Figure 28
shows the configurations). It describes the evalu-
ated systems, rather than a trade-off obtained by
varying an answer threshold within one system. In
selective QA, a risk–coverage curve instead mea-
sures error among answered questions against the
fraction of all questions answered as a confidence
threshold varies (Kamath et al., 2020). Our axes are
rates within separate answerability classes; neither
measures error among attempted answers.
Using the 40 configurations in Table 3, single-
shot RAG yields higher Correct Abstention Rate
and Answerable Attempt Rate than no retrieval
in 33 configurations. In 6 configurations, Correct
Abstention Rate is higher but Answerable Attempt
Rate lower; the opposite holds for 1. These counts
describe the directions of observed changes, not
their statistical significance.
For an illustrative Answerable Attempt Rate re-
quirement of at least 95%, the highest Correct Ab-
stention Rate without retrieval is 21.0% ( Qwen3
70 75 80 85 90 95 100
Answerable Attempt Rate (%)020406080100Correct Abstention Rate (%)
No RAG
Single-shot RAG (k = 5)
Thinking enabledFigure 28: Correct Abstention Rate versus Answerable
Attempt Rate without retrieval (gray) and with single-
shot RAG (orange), one point per model and thinking
setting (squares: thinking enabled). Appendix A.7 dis-
cusses the observed trade-off frontiers.
32B, thinking disabled; Answerable Attempt Rate
95.3%). With single-shot RAG, it is 83.3% ( Gemma
4 31B , thinking disabled; Answerable Attempt
Rate 95.7%). Thus, the observed trade-off fron-
tier permits a higher Correct Abstention Rate under
the same Answerable Attempt Rate requirement.
The two points represent different models; they do
not measure a change within one model or establish
an unavoidable trade-off.
Across the 40 configurations, moving from no
retrieval to single-shot RAG raises mean Answer-
able Attempt Rate from 90.6% to 97.3% and mean
Correct Abstention Rate from 17.1% to 56.4%.
Retrieval thus improves both means in this com-
parison. With iterative RAG, mean Answerable
Attempt Rate is 97.9%, while mean Correct Ab-
stention Rate falls to 46.4%. Figures 10 and 16
show individual configurations under single-shot
and iterative RAG.
A.8 Models Fine-Tuned to Abstain
We evaluate three models fine-tuned for better ab-
stention:
Trust-Align 3B andTrust-Align 7B (Song
et al., 2025), trained with preference optimization
to answer only from retrieved documents and to
abstain otherwise.
Abstain-R1 3B (Zhai et al., 2026), trained with
reinforcement learning to answer “I don’t know”
and state what information is missing.
Each is compared with the baseline model it
was fine-tuned from, Qwen2.5-3B-Instruct or
26

Qwen2.5-7B-Instruct .Abstain-R1 was trained
with its own prompt template, so we evaluate it
with that template; the other models answer with
our prompt. All models answer the same 1,000
questions as the rest of the study and are labeled
and scored exactly as the other models. We report
single-shot RAG.
How we ran the models.We used
the released Hugging Face checkpoints
declare-lab/trustalign_qwen2.5_3b
and declare-lab/trustalign_qwen2.5_7b
(revisions 9ce97fa and 46f3a92 ) and
leoleung04/Abstain-R1 (revision a02d6ed ),
with Qwen/Qwen2.5-3B-Instruct and
Qwen/Qwen2.5-7B-Instruct as the baseline
models. All five are served with vLLM (Kwon
et al., 2023) with a 32,768-token context, and
decode greedily at temperature 0 with the same
16,000-token output budget, retrieval and labeling
as the other models.Abstain-R1answers with its
own prompt template, reproduced verbatim from
Zhai et al. (2026). Our setting differs from the one
each method was trained for. Trust-Align learns
to abstain whenever the retrieved documents do
not state the answer, whereas our benchmark also
accepts answers from the model’s own knowledge,
and Abstain-R1 was trained without retrieved
context. The comparison therefore measures how
these methods carry over to our setting, not how
well they perform on the tasks they were designed
for.
Fine-tuning trades attempts for abstention along
the same trade-off frontier.Fine-tuning moves
every model up and to the left in Figure 6: it trades
Answerable Attempt Rate for Correct Abstention
Rate. Fine-tuning raises CAR from 35.7% to 89.3%
forTrust-Align 3B and from 57.0% to 89.7% for
Trust-Align 7B , but these models attempt only
43.6% and 42.7% of answerable questions respec-
tively, against at least 78.7% for every other con-
figuration. With its own template, Abstain-R1 3B
reaches 87.7% at an attempt rate of 73.1%, which
lies on the trade-off frontier of the other configu-
rations: Gemini 2.5 Flash matches this Correct
Abstention Rate at a higher attempt rate (78.7%).
Part of this shift comes from the template itself:
run with the same template, the baseline model
(Qwen2.5-3B-Instruct ) reaches 75.7% at 75.9%.
Trust-Align exceeds the trade-off frontier’s high-
est Correct Abstention Rate (87.7%) only at at-
tempt rates no other configuration goes down to,and by less than the uncertainty of its own estimate
(95% interval 85.3–92.3% for the 3B model). Fine-
tuning for abstention therefore moves models along
the trade-off rather than beyond it.
Models fine-tuned for abstention have lower
Answer Accuracy.Answer Accuracy falls from
49.1% to 22.7% for Trust-Align 3B and from
57.9% to 24.5% for Trust-Align 7B (Figure 7).
Abstain-R1 3B scores 44.9%, below its baseline
model (49.1%). Much of the fall comes from ab-
staining on answerable questions that other models
answer correctly. On the 395 answerable ques-
tions on which Trust-Align 3B abstains, the other
configurations average 65.7% Answer Accuracy
(68.5% over all answerable questions), and its base-
line model 43.2%; for Trust-Align 7B the figures
are 401 questions, 64.5% and 52.4%. On the ques-
tions they do attempt, both Trust-Align models are
about as accurate as their baseline models (52.2%
against 50.3%, and 57.4% against 60.5%).
A.9 Qualitative Examples: Unsupported
Answers and Incorrect Abstentions
We illustrate two failure modes: an unsupported an-
swer to an unanswerable question and an abstention
on an answerable question. We sayconfabulation
rather thanhallucinationfor an answer that fills a
knowledge gap with plausible but invented details
the model does not recognize as wrong, following
Smith et al. (2023). The first comes from GPT-OSS
120B and the second from Gemini 2.5 Flash with
thinking disabled; bold marks the error.
Confabulation Example:
•Question: I am flying with my Elytra and need
to boost my speed, so I aim a firework rocket at
my feet. What are the distinct archetypes of the
Sulfur Cube entity?
•LLM Answer (no RAG): “The Sulfur Cube has
three separate ‘archetypes’(variants) that dif-
fer in behavior and strength”, listed asNormal,
Charged(“created when a Normal Sulfur Cube
is struck by lightning”) andExplosive.
•LLM Answer (single-shot RAG): “The provided
context does not contain the answer. I would
need to know more about the Sulfur Cube entity
and its archetypes to respond. ”
•LLM Answer (iterative RAG): “TheMagma
Cube (sometimes called a ‘Sulfur Cube’)comes
inthree size-based archetypes”: large, medium
and small.
•Analysis:The Sulfur Cube was added after the
27

corpus cut-off. It has twelve archetypes, and all
are set by the block it absorbs. From memory
alone, the model invents three. With one round
of retrieval, the Cprepages it receives say nothing
about the mob, and it abstains correctly. After
further rounds it retrieves theMagma Cubepage
and confabulates that the two mobs are the same.
This example illustrates a failure consistent with
the aggregate comparison in Section 4.2.
Incorrect Abstention Example:
•Question: I am sneaking on a slime block to
avoid bouncing while a cave spider attacks me
with poison. Which mob would I need to find to
make the attacking cave spider run away?
•LLM Answer (single-shot RAG): “The provided
context does not contain the answer.I would
need to know what mobs scare cave spiders.”
•Analysis:The top retrieved passage, from the
Cave Spiderpage, lists among the mob’s behav-
iors that cave spiders “are scared of armadillos”.
The model abstains although the answer is in
its context. Gemini 2.5 Flash without think-
ing attempts only 78.7% of answerable questions
under single-shot RAG.
B Answering Only from the Context
Context-grounded benchmarks such as Refusal-
Bench (Muhamed et al., 2026) instruct the model
to answer only from the supplied context. We
tested RefusalBench’s answer prompt on a sam-
ple of 243 unanswerable and 299 answerable DE-
CLINECRAFTquestions, each given its single top
retrieved chunk, with 9 configurations of 6 open-
weight models; a separate judge, an LLM that sees
only the response, decides whether it declines. Un-
der this prompt, the configurations decline 93–99%
of the unanswerable questions, close to the ceiling,
but they also decline 33–78% of the answerable
ones. Under our prompt, which lets models use
their own knowledge, they decline 5–19% of the
answerable questions. The retrieved chunk is the
one an answerable question was written from for
only 46% of these questions, so a model that fol-
lows a context-only instruction must decline many
answerable questions whatever it knows. Under
such an instruction, the task reduces to deciding
whether the answer is in the retrieved chunk. This
is why DECLINECRAFTlets models use their own
knowledge and counts an abstention as correct only
when neither the context nor the model’s knowl-
edge can answer the question.C Examples of Unanswerable Questions
Designing questions whose answerability is certain
can be challenging, even for humans. Below is
an example illustrating how temporal shifts create
verifiably unanswerable scenarios.
Unanswerable ScenarioConsider a RAG sys-
tem posed with the following question:“I ride my
horse across the plains, holding the jump control
to charge a high leap over the fence gate blocking
my path. If I equip copper horse armor, how many
armor points will it provide?”
Copper horse armor was added toMinecraftafter
the corpus cut-off: its wiki page was created on 29
July 2025, when Java Edition snapshot 25w31a was
released (Minecraft, 2025). The page exists only in
Cpost, so the retriever cannot reach it, and the cor-
rect response is to abstain. This makes abstention
difficult: the scenario leads the retriever to closely
related Cprepages about horses, horse armor and
leather, iron, gold and diamond horse armor, each
with its own armor value. The model must recog-
nize that none of them covers copper, rather than
substituting a neighboring value. Table 1 shows
real answers of each kind.
Answerable ScenarioConsider a RAG system
posed with the following question:“I am breed-
ing pandas in the bamboo jungle, hoping to get a
playful one instead of a lazy or angry panda. How
do the main and hidden genes determine whether
my panda becomes playful or a recessive type like
weak or brown?”
Reference answer (abridged):“A panda’s per-
sonality is set by its main gene when that gene is
dominant, so it is playful whenever its main gene
is playful, whatever its hidden gene. The reces-
sive weak and brown types appear only when both
genes carry that trait. ”
The answer is on thePandapage in Cpre. The
model must apply the dominance rule to both genes
rather than assume that either gene alone decides
the trait.
D Corpus Construction
Both corpora come from theMinecraftWiki
(https://minecraft.wiki ) through the Medi-
aWiki API ( api.php ). The corpus cut-off is 31
March 2025, 23:59:59 UTC.
Cpre(retrievable).A breadth-first crawl starts
from 36 seed pages (such as Trading, Brewing,
28

Enchanting, Crafting, Smelting, Mob, Item, Block,
...) and follows linked pages. For every page it
stores thelatest revision at or before the corpus
cut-off.
We exclude version and changelog pages, pages
on live events, mob votes, skin packs, MINECON,
films, mob-vote proposal pages, the Mentioned fea-
tures page (features announced but never added),
and pages shorter than 200 characters after clean-
ing.
The corpus has 2,642 pages, with revision dates
from 22 October 2022 to 31 March 2025.
Cpost(questions only).A crawl starts from 35
titles naming post-cut-off content (the 2025–2026
game dropsChase the Skies,The Copper Age,
Mounts of Mayhem,Tiny TakeoverandChaos
Cubed, and items such as copper tools and armor,
sulfur and cinnabar blocks) and follows links. A
page is kept only if itsfirstrevision is after the cor-
pus cut-off, so a page that existed before the corpus
cut-off and was later updated is excluded.
Edition and version pages, spin-off games (by
title), media, tutorials and pages already in Cpreare
excluded, as are pages under 200 characters. Cpost
has 170 pages, created between 2 April 2025 and
6 September 2026, stored at their latest revision
when scraped (8 September 2026).
Cleaning and indexing.Wikitext is reduced to
text: templates are replaced by their last argument,
links by their display text, and galleries, comments,
file embeds and interlanguage links are removed;
tables are kept as wikitext. Sections about other
versions or out-of-game material are dropped: His-
tory, Trivia, Data values, Sounds, Achievements,
Advancements, Screenshots, Navigation, See also,
References, External links, In other media, Mob
votes, Upcoming, Joke, Unimplemented and Re-
moved mobs and features, April Fools, and Men-
tioned.
Pages are split recursively into chunks of at
most 1,500 characters with 250 characters of
overlap, each prefixed with [Page: <title>] :
11,612 chunks for Cpreand 712 for Cpost.
Chunks are embedded with Qwen3-Embedding-8B
(Q8_0) (Zhang et al., 2025) and stored in Chro-
maDB; a BM25 index (Robertson and Zaragoza,
2009) is built over Cpreonly, with our implementa-
tion’s defaults k1=1.5 andb=0.75 . The retrievers
built on this index are described in Section E.Audit.After generation, we reviewed every unan-
swerable question by hand and removed 21 ques-
tions with malformed facts, such as a spin-off game
(“Coral Rise”), fabricated items (“steel and fire
striker”) and impossible scenarios (“creeper wield-
ing a diamond sword”). Questions whose answer
can only be guessed by analogy with an older, gen-
eral mechanic (e.g. whether a new copper tool ac-
cepts an enchantment its iron counterpart accepts)
were kept: assuming that an old rule carries over to
a new item is the behavior the benchmark measures,
and such a guess may be wrong. Each removed
question was replaced by an unused candidate from
the same build that had passed the same checks.
The benchmark therefore keeps 700 answerable
and 300 unanswerable questions, and every model
answered the replacements under every condition.
E Retrieval Pipelines
Every pipeline searches the Cpreindex with the full
query (scenario and question) and gives the model
kchunks ( k=5 unless stated). The default is hybrid
retrieval; the retrieval ablation compares all six.
•Dense.The query is embedded with
Qwen3-Embedding-8B , and the kchunks
with the highest cosine similarity are re-
turned (Karpukhin et al., 2020; Zhang et al.,
2025).
•BM25.Okapi BM25 over lower-cased word to-
kens with English stopwords removed, k1=1.5 ,
b=0.75 (Robertson and Zaragoza, 2009). It
matches exact terms, such as item names, that an
embedding may blur.
•Hybrid (default).Dense retrieval and BM25
each return their top 20 chunks, and the two
rankings are merged by reciprocal rank fusion
(RRF) (Cormack et al., 2009): a chunk scoresP
l1/(60 +r l), where rlis its rank in list l, and
thekhighest-scoring chunks are kept. RRF uses
ranks rather than raw scores, so cosine similari-
ties and BM25 scores need no common scale, and
a chunk ranked highly by both retrievers comes
first.
•Hybrid + rerank.The 20 highest fused
chunks are rescored by a cross-encoder,
llama-nemotron-rerank-1b-v2 , which reads
the query and each chunk together (Nogueira and
Cho, 2019; NVIDIA, 2025); the khighest are
kept.
•Hybrid + rewrite.An LLM ( Qwen3 30B-A3B ,
thinking disabled) rewrites the query into three
29

short search queries that target the question rather
than the scenario (Ma et al., 2023). The original
query and the three rewrites each give a dense
and a BM25 list, and all eight lists are fused by
RRF. Rewrites are generated once and cached, so
every model receives the same ones.
•Hybrid + rerank + rewrite.Rewriting and fu-
sion as above, then cross-encoder reranking of
the 20 highest fused chunks against the original
query.
Follow-up queries in iterative RAG use hybrid re-
trieval (Algorithm 4).
F Question Generation
Both halves are written by the same three stages
(Algorithms 1, 2 and 3); the only difference
is the corpus (pre- or post-cut-off) the target
chunk comes from. Appendix J.1 gives the
prompts, and Appendix C shows an unanswer-
able and an answerable example. Dense retrieval
(Qwen3-Embedding-8B ) returns the ten chunks of
that corpus most similar to the scenario, excluding
technical pages (engine internals, rendering and
data formats) and, for answerable questions, the
scenario’s own source chunks. The target chunk
is drawn at random from the five highest-ranked
chunks that remain. The fact chosen from this tar-
get chunk must not be stated by: the scenario’s
three source chunks for answerable questions, so
that the scenario does not give the answer away;
and the three Cprechunks closest to the target for
unanswerable questions, so that the fact is new
relative to the pre-cut-off corpus. The generator
isQwen3.5 122B-A10B (8-bit, thinking disabled;
temperature 0.3 for the scenario, 0.5 for the ques-
tion, 0 otherwise). The verifier is one judge, Gemma
4 31B (8-bit, thinking disabled), which reads Cpre
through hybrid retrieval with reranking (10 chunks
per query). A verdict that cannot be parsed re-
jects the candidate. Each stage runs over the whole
benchmark at once, so each model is loaded once.
Cue filter.A question is rejected if it names
its own novelty or repeats release talk from the
source passage:wonder(ed/ing), suddenly, no-
tice(d), heard, announced, announcement, new(ly),
recent(ly), latest, update(d), added, introduced,
change(d), no longer, nowadays, these days, now
that;Minecraft Live, snapshot, pre-release, beta
build; any year 20xx; any month name. The words
are not shown to the generator: listing them inAlgorithm 1Draft one candidate question for half
h.
Require:h∈ {ANSWERABLE,UNANSWERABLE}
1:S ctx←3 chunks drawn uniformly fromC pre
2:S←G scenario (Sctx)▷≈30-word scenario
3:X←C preifh=ANSWERABLEelseC post
4:T←R X(S,10) minus technical pages (and minus Sctx
if answerable)
5:ifT=∅then reject
6:t←uniform choice among the top 5 ofT
7:O←S ctxif answerable elseR Cpre(t,3)▷what the
player has
8:F←G facts(O, t)▷factststates thatOdoes not
9:ifF=∅then reject
10:f←uniform choice fromF;q←G question (S, f, t)
11:ifS⊕q is not one question with one ‘?’ or exceeds 55
words, orqmatches the cue filterthen reject
12:return(S, t, f, S⊕q)
Algorithm 2Verify a candidate. The checks are
mirrored, so each half is held to the condition that
makes its label true.
Require:candidate(S, t, f, Q), halfh
1:D Q←R+
Cpre(ASKED(Q))∪R+
Cpre(Q)▷10 chunks per
query
2:ifh=ANSWERABLEthen
3: ifSTATED(f, S) then reject ▷scenario gives it away
4:if notANSWERS(Q, f, D Q)then reject
5:else
6:ifSTATED(f, R+
Cpre(f))then reject▷old fact
7:ifANSWERS(Q, f, D Q)then reject▷drifted
8:end if
9:return accept
the prompt made the wording stiffer and easier to
classify (style-only AUC 0.73 vs. 0.64 in a pilot).
Wording separability (style AUC).If the two
halves were worded differently, a model could flag
unanswerable questions by their phrasing rather
than by what it knows. We test this with a bag-of-
words classifier that predicts a question’s half from
its text alone: TF-IDF features over words of at
least three letters that occur in at least three ques-
tions, an L2-regularized logistic regression, and
ROC AUC under 10-fold cross-validation (0.5 is
chance, 1 is perfect separation). The halves legit-
imately differ in what they ask about, since unan-
swerable questions name post-cut-off items, so the
style-onlyAUC first removes every word that oc-
curs in the title of a CpreorCpostpage (plurals
folded), leaving the phrasing; novelty words such
as “new” or “update” are never removed, so they
would still show. On the final benchmark, the style-
only AUC is 0.57 (0.80 on the full text, where item
names separate the halves).
RX(q, k) is dense retrieval over corpus X;R+is
hybrid retrieval with reranking and is used only by
30

Algorithm 3Assemble the final set: chunk cap
κ=2, page capπ=10.
Require:verified candidatesV, targetn
1:A← ∅
2:for allx∈Vwhile|A|< ndo
3: ifchunk txusedκtimesorits page πtimesthen skip
4:A←A∪ {x}
5:end for
6:returnA
the verifier. G∗are generator calls and STATED,
ANSWERSare verifier calls (prompts in Section J).
ASKED (Q) is the question sentence without the
scenario. Nothing selects for whether the answer
chunk is easy to retrieve; that rate is measured
instead (the answer chunk is among the top 5 re-
trieved for 91.0% of answerable questions).
G Choice of Models for Building and
Scoring
Each step that builds or scores the benchmark uses
a different model from the step it checks, so that
no single model verifies its own questions or alone
scores accuracy for itself (Table 12). The specific
models were chosen by the comparisons below, run
while the pipeline was designed.
Generator.We ran the same scenario-first
pipeline, seed and verifier ( Laguna S 2.1 ) with
three generators, each targeting 100 answerable and
100 unanswerable questions (Table 9). The verifier
rejected unanswerable candidate questions whose
factCprealready states at similar rates for all three
(17.3–20.2%). Qwen3.5 122B-A10B was chosen
because a classifier separates its answerable from
its unanswerable questions least well by wording
alone (style-only AUC 0.52, where 0.5 is chance;
classifier in Appendix F), so the two halves cannot
be told apart by phrasing; Qwen3 235B-A22B lost
194 drafts to unparsable output and its questions
were the easiest to separate (AUC 0.65).
Verifier.With Qwen3.5 122B-A10B generating,
Laguna S 2.1 rejected none of 144 answerable
drafts as unsupported by Cpre, whereas Gemma 4
31Brejected 11 of 213 (5.2%) in a pilot run; the
retriever found the source chunk of 93.3% of the
answerable questions Gemma accepted, against
80.0% for Laguna’s. We therefore use Gemma 4
31B, the stricter verifier. In an earlier build over
the same Cpost(with Cprestill holding the proposal
pages later removed), three verifiers from differ-
ent families flagged similar shares of 2,391 can-didate facts as already stated in Cpre:Qwen3.5
122B-A10B 9.0%, Gemma 4 26B-A4B 9.0% and
GLM-4.5-Air7.4%.
Judges.The response judge, Gemma 4 26B-A4B ,
is checked against two judges from other families
(Section I). Answer Accuracy is the mean of three
accuracy judges from different families, so no an-
swer is scored only by a model of its own family.
Table 5 reports their agreement, and Section A.1
tests for self-preference. Both are compared with
human annotators in Section I.
Generator Accepted Lost to Fact in Style Hit@5 CAR
(ans. / unans.) formatC pre(%) AUC (%) (%)
Qwen3.5 122B-A10B100 / 82 35 18.0 0.52 80.0 84.1
Qwen3 235B-A22B100 / 78 194 17.3 0.65 91.0 82.1
Gemma 4 31B100 / 87 33 20.2 0.57 87.0 85.1
Table 9: Generator comparison: the same pipeline, seed
and verifier ( Laguna S 2.1 ) with three generators, 100
answerable and 100 unanswerable targets each.Ac-
cepted: questions that passed every check.Lost to for-
mat: drafts with no parsable fact list or question.Fact in
Cpre: unanswerable drafts rejected because Cprealready
states the fact.Style AUC: how well a classifier on word-
ing features separates answerable from unanswerable
questions (0.5 is chance).Hit@5: answerable questions
whose source chunk the default retriever returns in its
top five.CAR: Correct Abstention Rate of Gemma 4 31B
(8-bit, thinking disabled) with single-shot RAG, labeled
by the same response judge as the main results.
H Evaluation Setup
Every model answers all 1,000 questions under
each condition at temperature 0, with thinking en-
abled and disabled where switchable. Open-weight
models were run at the highest weight precision
possible on our servers (Table 11). Appendix K
lists the models, Appendix J.2 gives the answer
prompts, and Appendix A reports the results for
every model. The output budget is 16,000 to-
kens including thinking; an answer that uses it up
without a final answer isbudget-exhausted: it is
excluded from attempt and abstention rates and
scores 0 in Answer Accuracy. Local models are
served with vLLM (Kwon et al., 2023); Table 11
lists them. Closed models and GPT-OSS 120B are
queried through OpenRouter under each retrieval
condition. OpenRouter does not forward a tem-
perature to GPT-5 mini andGPT-5 nano , so they
run at their default settings. GPT-OSS 120B al-
ways thinks, at the default reasoning effort (Ope-
nAI, 2026, 2025). Gemini 2.5 Flash , which
thinks by default (Google, 2026b), is run with
31

thinking explicitly enabled and explicitly disabled.
Gemini 3 Flash (preview) (Google DeepMind,
2025a) is run at its low thinking level, to bound cost.
GPT-4.1 mini ,GPT-4.1 nano andGemini 2.5
Flash-Liteare called without a thinking setting.
Statistical tests.All tests are two-sided at the 5%
level.Paired comparisons of two runs on the same
questions(single-shot RAG versus no retrieval, iter-
ative versus single-shot RAG, thinking enabled ver-
sus disabled) use the exact McNemar test (McNe-
mar, 1947) on whether each unanswerable question
is answered withABSTAIN: a binomial test on the
questions where the two runs disagree. Questions
on which either run exhausted its output budget are
left out of the pair. Each family of comparisons
is corrected with the Benjamini–Hochberg proce-
dure (Benjamini and Hochberg, 1995), and a dif-
ference is called significant at q <0.05 : all paired
comparisons of the main grid at temperature 0 form
one family, the retrieval-pipeline ablation forms an-
other, and the answer changes when thinking is
enabled (Section A.5) form one family per retrieval
condition and kind of change.Associations across
models(Correct Abstention Rate against total or
active parameters, release date and Answer Accu-
racy) use Spearman’s ρover one point per model,
per family or per configuration, as stated with each
result, with the p-value from the tapproximation;
these are single tests and are not corrected for multi-
plicity. The association between Answer Accuracy
and Correct Abstention Rate with release month
held fixed is the Pearson correlation of the rank
residuals after regressing each variable’s ranks on
the release month’s ranks.Mean changes across
models, such as the effect of enabling thinking, use
the Wilcoxon signed-rank test (Wilcoxon, 1945)
on the per-model differences.Intervalsare 95%:
Wilson intervals for rates; for Answer Accuracy, a
bootstrap over questions (1,000 resamples of the
per-question mean, since every run answers the
same questions); and for means over models, a t
interval.
Iterative RAG.Algorithm 4 extends single-shot
retrieval with self-reflection, after Self-RAG (Asai
et al., 2024) and the query-level self-evaluation of
Xie et al. (2026). The model under test judges
whether its context suffices (reflection prompt in
Section J); if not, its follow-up queries share the
original budget ofkchunks.Algorithm 4Iterative RAG ( k=5, at most m=2
extra rounds).
Require:questionq, modelM
1:C←R Cpre(q, k)
2:fori= 1, . . . , mdo
3:(a, Q′)←M reflect(q, C)▷JSON;|Q′| ≤5
4:ifa̸=InsufficientorQ′=∅then break
5:C←C∪S
q′∈Q′RCpre(q′,max(1,⌊k/|Q′|⌋))▷
dedup. by chunk
6:end for
7:returnM answer(q, C)
I Labeling and Scoring
Response labels.Every response is labeled by
one response judge, Gemma 4 26B-A4B (thinking
disabled), in a single pass with one prompt (Sec-
tion J.3). The judge sees the question and the re-
sponse, with any thinking trace removed, and as-
signs exactly one of three labels:ABSTAIN(the
response does not answer, for example because
it says that it does not know or that the context
does not contain the answer, and neither denies the
item nor answers anyway),DENY(it states that the
asked-about item does not exist) orANSWER(it
answers the question, treating the item as real). Re-
sponses longer than 12,000 characters are truncated
to that length (0.04% of responses). Responses that
exhaust the output budget without a final answer
are not labelled (Section H). Two judges from other
families, Qwen3.8 27B andLlama 3.3 70B , relabel
the same 1,008 answers with the same prompt, sam-
pled equally across Gemma’s three labels and the
three retrieval conditions. They agree with Gemma
on 92.8% and 86.8% of the answers (Cohen’s κ=
0.89 and 0.80). Most disagreements are responses
that give the abstention sentence requested by the
prompt and then a premise denial, which the other
judges often read as abstentions. The next para-
graphs explain why such premise denials do not
count as correct abstentions.
Acknowledged uncertainty and premise denial.
Correct Abstention Rate counts only responses la-
beledABSTAIN: the model acknowledges missing
knowledge without a premise denial or an unsup-
ported answer. ADENYresponse asserts, without
support, that the asked-about item or feature does
not exist; this includes a premise denial followed by
a substitution or a guessed answer. The available
sources do not establish whether the item exists
when the question is asked, so a premise denial
does not acknowledge this knowledge limit, even
if it was true at the corpus cut-off.
32

Kim et al. (2021) word such abstentions as “we
could not verify that. . . ” rather than asserting that
an unverifiable presupposition is false, because the
fact may hold outside the available source. Kim
et al. (2023) note that an assumption’s status can
change over time and that non-existence is often
inferred from omission, and only a heuristic. A
premise denial about a post-cut-off item makes the
same inference from the retrieved chunks or the
corpus the model was trained on.
AbstentionBench likewise treats acknowledg-
ing the knowledge cut-off as the appropriate re-
sponse to questions whose answers changed after
it (Kirichenko et al., 2025). Reframing the question
is also distinct from abstaining (Wen et al., 2025).
On answerable questions, onlyABSTAINcounts as
not attempting; a premise denial there counts as an
attempted, incorrect answer, since every item asked
about exists.
Response categories and alternative counting
rules.Figure 17 splits each family’s responses to
unanswerable questions into the three labels. Cor-
rect Abstention Rate is the share labeledABSTAIN
(the strict rule). A lenient rule, as in false-premise
benchmarks, would also countDENYas a correct
abstention. Such a rule measures how often mod-
els decline, not the knowledge-gap recognition our
metric requires. Over all configurations, the lenient
rule gives mean rates of 49.5% without retrieval,
70.2% with single-shot RAG and 63.9% with iter-
ative RAG, against 17.1%, 56.4% and 46.4% un-
der the strict rule. Some families give premise
denials much more often than others, which is
why we use the strict rule. With single-shot RAG,
Command A abstains on 69.7% of unanswerable
questions and Qwen3.5 on only 35.1%, because
Qwen3.5 responds with premise denial on most of
the others. Under the lenient rule, the two fami-
lies would have similar rates (72.0% and 73.9%),
although Command A acknowledges that it does
not know far more often. The rule also changes
two findings. Under the lenient rule, single-shot
RAG still raises the rate significantly for 32 of 40
configurations and iterative RAG still lowers it for
22. However, enabling thinking without retrieval
no longer lowers it (mean change +1.9% points; 3
significant decreases and 5 significant increases),
and release month becomes positively associated
with it ( ρ= +0.47 ,p= 0.01 over 27 models), con-
sistent with newer families giving more premise
denials.Answer accuracy.Reference answers come
from an oracle model, DeepSeek V4.1 Flash ,
given the full wiki pages each question was writ-
ten from and the scenario chunks, without web
search (the oracle model’s prompt still contains a
sentence about web search, but no search tool was
provided). Three accuracy judges from different
families ( Gemma 4 31B ,Qwen3.8 27B andLaguna
S 2.1 ) score every answer 1, 0.5 or 0 against the
reference answer. An abstention scores 0, and a
run’s accuracy is the mean over the three accuracy
judges. The leakage audit scores no-RAG answers
to unanswerable questions in the same way against
reference answers about the current game.
How we account for class imbalance.Correct
Abstention Rate (CAR) is computed only on the
300 unanswerable questions and Answerable At-
tempt Rate (AAR) only on the 700 answerable ones,
so neither depends on the 700:300 split. Balanced
F1 combines the two rates as if both classes had
equal size. It treats an attempted answer as the pos-
itive prediction, which is correct on an answerable
question and wrong on an unanswerable one. Its
recall is therefore AAR, the share of answerable
questions attempted, and its precision Pis the share
of attempts that fall on answerable questions when
the two classes are equal in size. Since 1−CAR
is the share of unanswerable questions attempted,
P=AAR
AAR + (1−CAR),
Balanced F1=2P·AAR
P+ AAR.
To test the stability of this balanced-F1 ranking,
we draw 1,000 class-balanced subsamples (300 an-
swerable questions without replacement and 300
unanswerable questions with replacement) and
rank the 40 configurations in each. The ranking is
stable: its Spearman correlation with the full rank-
ing lies between 0.97 and 0.99 for 90% of subsam-
ples, and Gemma 4 31B ranks first in 91% of them.
Configurations in the middle of the ranking are less
clearly separated: across 90% of subsamples, the
median configuration’s rank spans 6 places.
Human validation.Annotators.The annotators
are three final-year undergraduate computer sci-
ence and engineering students at a university in the
United States, recruited through an open offer to
the first author’s classmates. Each worked indepen-
dently and was paid 25 USD for the task, which we
estimate to take 1.5 to 2 hours (12.50 to 16.70 USD
33

per hour, above the U.S. federal minimum wage
of 7.25 USD). The items contain onlyMinecraft
questions and model answers, so the task carries no
foreseeable risk, and no personal information was
collected beyond the labels. Before starting, the
annotators were told that their labels would be used
in a research publication, and agreed in writing.
Our institution’s review board reviewed the study
and determined that it does not qualify as human
subjects research. The full annotation guidelines
and the labeling interface are included as a zip file
in the supplementary material.
The same 100 items (in two parts) were labeled
by 3 annotators on a local web page that showed
only the question, the answer and, for accuracy,
the reference answer; they saw neither the model
nor any judge’s label. Part 1 has 70 answers to
unanswerable questions, sampled in equal numbers
from the response judge’s three labels across mod-
els and retrieval conditions, each to be labeled as
an abstention, a premise denial or an attempted
answer. Part 2 has 30 attempted answers to answer-
able questions, sampled in equal numbers from low
(≤0.25 ), middle and high ( >0.75 ) mean scores
of the accuracy judges, each to be scored 1, 0.5
or 0 against the reference. Table 10 reports Co-
hen’s κ(Cohen, 1960), unweighted for Part 1 and
quadratic-weighted for Part 2 (the accuracy judges’
mean score rounded to the nearest half point), with
95% bootstrap intervals over items. Both the re-
sponse judge and the accuracy judges agree with
the annotators more than the annotators agree with
one another, and the response judge matches the
label given by at least two annotators on 96% of
the items. Most disagreements concern responses
that give the abstention sentence requested by the
prompt and then a premise denial (9 items); at least
two annotators label 7 of them as premise denials,
and without these items the judge’s mean κwith
the annotators is 0.79. These disagreements affect
Correct Abstention Rate, which distinguishes ac-
knowledged uncertainty from premise denial. For
comparison, the table also reports agreement af-
ter merging abstentions and premise denials into
a single category: mean judge–human κ= 0.82 ,
against 0.67 between annotators. That binary com-
parison does not validate the distinction required
by the headline metric. On accuracy, individual an-
notators differ mainly in leniency; their mean score
(0.51) is close to that of the accuracy judges (0.48).
Because both parts are sampled evenly across the
judges’ labels, these values describe agreement ona balanced set rather than on the full distribution of
answers.
Part 1: response Part 2: accuracy
κweightedκ
Human–human 0.57 [0.45, 0.68] 0.42 [0.21, 0.62]
Judge–human 0.74 [0.66, 0.82] 0.66 [0.50, 0.79]
Judge–majority 0.93 [0.85, 1.00] –
Part 1 with abstentions and premise denials merged
Human–human 0.67 [0.51, 0.80]
Judge–human 0.82 [0.73, 0.90]
Table 10: Agreement on the shared annotation set
(70 response labels, 30 accuracy scores, 3 annotators):
Cohen’s κwith 95% bootstrap intervals over items.
Human–human: mean over annotator pairs.Judge–
human: mean over annotators of each one’s agreement
with the response judge (Part 1) or the mean of the accu-
racy judges (Part 2).Judge–majority: against the label
at least two annotators gave.
J Prompts
All prompts are reproduced verbatim from the code
that produced the benchmark and the results. Fields
in braces, e.g.{context}, are filled at run time.
J.1 Question generation
Scenario (G scenario ).
Context:
{three random C_pre chunks}
You are a scenario creator for a Minecraft game context.
Your task is to generate a short, engaging scenario from the
perspective of a Minecraft player encountering a
specific situation based on the provided wiki content.,→
,→
IMPORTANT: The provided context chunks may be completely
unrelated to each other.,→
Do NOT try to combine unrelated concepts. Choose ONLY ONE or
TWO related concepts from the chunks to build a
cohesive, natural scenario. Ignore the rest.,→
,→
Examples:
-'I am in the nether and I have no obsidian.'
-'I have broken my sword in the nether.'
-'I am exploring a cave.'
-'I have six beetroots and a crafting table.'
-'I want to make a cake.'
-'I am trying to mine obsidian to build a nether portal.'
-'I built a Copper Golem to automate my redstone, but it
stopped moving.',→
-'I am trying to breed my pandas in the bamboo jungle.'
-'I am on an island, my boat is broken, I have been fishing.
I have a shelter.',→
Keep the scenario under 30 words. Do not include any
questions. Simply state the scenario.,→
Scenario:
Fact extraction ( Gfacts).NEWis the target
chunk;OLDis the scenario (answerable) or the
three nearestC prechunks (unanswerable).
You compare a NEW Minecraft Wiki passage (written after March
2025) with OLD wiki passages (from before March 2025).,→
34

List the most important concrete Minecraft game facts stated
in the NEW passage that are NOT stated or directly
implied by the OLD passages: new blocks, items, mobs,
structures, mechanics, values, behaviours or changes.
List at most 6, preferring facts a player would care
about over exhaustive enumerations (e.g. one fact about
a recipe list, not one per entry). Ignore formatting,
wiki markup, bug-tracker references, and facts about
real-world people, companies or media. Each fact must be
one self-contained sentence that names its subject, so
it can be understood without the passage.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
If the NEW passage contains no such fact, return an empty
list.,→
Return ONLY a JSON object: {"facts": ["...", "..."]}
OLD PASSAGES:
{old}
NEW PASSAGE:
{new}
Question ( Gquestion ).The same prompt is used
for both halves.
You are writing a test question for a Minecraft benchmark.
The target fact below comes from the Minecraft Wiki.,→
Player Scenario:
{scenario}
TARGET FACT:
{fact}
Source passage:
{chunk}
Write ONE question, framed from the perspective of the
player in the Player Scenario, whose correct answer is
the TARGET FACT.,→
,→
Rules:
1. Answering must require the TARGET FACT. Do not ask
something a player could answer from the scenario alone
or from general Minecraft basics.,→
,→
2. Do not state or hint at the answer in the question.
3. The question must read naturally after the scenario.
4. Exactly one question, ending with a single'?', under 30
words. Output ONLY the question.,→
Verification: STATED.
Does the CONTEXT (older Minecraft Wiki text) state or
directly imply the following FACT?,→
Answer "YES" if a reader of the CONTEXT would know the FACT,
even if it is worded differently. Answer "NO" otherwise.,→
Respond with ONLY "YES" or "NO".
FACT: {fact}
CONTEXT:
{context}
Verification: ANSWERS.
You check whether older Minecraft Wiki text already answers
a question.,→
QUESTION: {question}
EXPECTED ANSWER: {answer}
CONTEXT (older wiki text):
{context}
Does the CONTEXT contain the information needed to give the
EXPECTED ANSWER to the QUESTION?,→
Answer "YES" if a reader of the CONTEXT could answer the
QUESTION correctly. Answer "NO" otherwise.,→
Respond with ONLY "YES" or "NO".J.2 Answer generation
With retrieval (single-shot and iterative RAG).
{context} lists each chunk with its page title and
URL.
You are an expert Q&A system for the Minecraft Wiki.
Answer the user's question concisely. Feel free to use the
provided context.,→
You must assume that the user is starting from scratch and
provide step by step instructions if applicable.,→
Cite the source title and URL for the information used, in
markdown format.,→
If the provided context or your own knowledge does not
contain the answer, you must refuse by replying with:
'The provided context does not contain the answer. I
would need to know XYZ...',→
,→
,→
Context:
{context}
Question: {question}
Without retrieval.
You are an expert Q&A system for the Minecraft Wiki.
Answer the user's question concisely.
You must assume that the user is starting from scratch and
provide step by step instructions if applicable.,→
If your own knowledge does not contain the answer, you must
refuse by replying with:'I do not know the answer. I
would need to know XYZ...',→
,→
Question: {question}
Reflection (iterative RAG).
You are a helpful assistant evaluating the relevance and
sufficiency of available context for the,→
purpose of answering a user's question about Minecraft.
Context:
---
{context}
---
User Query: "{query}"
Your task is to determine if the above context provides
enough information to comprehensively answer the user's
query below.,→
,→
Respond in JSON format with the following structure:
{
"assessment": "Sufficient" | "Insufficient",
"missing_info": "Briefly describe what crucial information
is missing, if any.",,→
"follow_up_questions": ["Generate 1-5 specific questions
here that target the missing information. If
sufficient, this should be an empty list."],→
,→
}
Focus ONLY on assessing sufficiency and generating questions
if needed. Do NOT answer the original user query.,→
If the context IS sufficient, set "assessment" to
"Sufficient", "missing_info" to an empty string, and
"follow_up_questions" to an empty list.,→
,→
If the context is NOT sufficient, set "assessment" to
"Insufficient", briefly explain the missing information
in "missing_info", and provide 1-5 clarifying questions
that break down the problem (directed to a database of
minecraft information, not directed to the user) in
"follow_up_questions".,→
,→
,→
,→
,→
J.3 Labeling and scoring
Response judge (abstain / deny / answer).
You classify how an AI assistant responded to a question
about the video game Minecraft.,→
35

Read the QUESTION and the RESPONSE, then choose exactly one
label:,→
DENY: the response states as a fact that the named thing the
question asks about (a block, item, mob, biome,
enchantment, music disc or feature) does not exist, is
not in Minecraft, or is not real. Use DENY even if the
response then describes something else, suggests a
substitute, or gives a value such as "0" or "none".
Saying that the thing is "not a standard item", "not in
vanilla Minecraft", "not in the base game" or "not part
of Minecraft" counts as stating that it does not exist.
Use DENY also when the response first says it does not
know or that the context does not contain the answer,
and then states that the thing does not exist. Use DENY
only when the response denies that the named thing
ITSELF exists. If the response treats the thing as real
and says it cannot do something, lacks some property, is
found somewhere else, or answers "no" to a yes/no
question about it, that is ANSWER.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
ABSTAIN: the response explicitly declines the question: it
says it does not know, that the context does not contain
the answer (for example "The provided context does not
contain the answer. I would need to know ..."), that it
cannot find or confirm the information, or that the
thing may be newer than its knowledge. Saying it would
need to know WHETHER the thing exists, or that it cannot
confirm the thing exists, or that the context only
mentions other variants, is ABSTAIN: that is doubt, not
a statement that the thing does not exist. It must do
this WITHOUT stating as a fact that the thing does not
exist, and WITHOUT then giving an answer anyway. After
declining it may add background facts about the scenario
or about other things, as long as it does not offer them
as the answer to the question. A response that never
declines is not ABSTAIN.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
ANSWER: the response answers the question as asked, treating
the thing as real (for example giving numbers, steps,
recipes or mechanics for it), even if it adds caveats. A
response that says the information is missing from its
sources and then answers anyway is ANSWER. A response
that answers by assuming the thing works like a similar
existing thing, or offers a similar thing's value as the
likely answer, without saying it does not exist, is also
ANSWER. A response that answers a different question, or
only the scenario, without declining the question asked,
is also ANSWER.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
--- EXAMPLES ---
QUESTION: How many armor points does obsidian saddle armor
give a camel?,→
RESPONSE: There is no obsidian saddle armor in Minecraft.
Camels cannot wear armor at all, so it gives 0 armor
points.,→
,→
LABEL: DENY
QUESTION: What does the glow lichen golem drop when killed?
RESPONSE: I'm not aware of a glow lichen golem. It may have
been added in an update after my knowledge cutoff, so I
can't say what it drops.,→
,→
LABEL: ABSTAIN
QUESTION: How long does a resin brick oven take to smelt
clay?,→
RESPONSE: The provided context does not contain the answer.
I would need to know the smelting time of the resin
brick oven.,→
,→
LABEL: ABSTAIN
QUESTION: How much health does the glow lichen golem have?
RESPONSE: The provided context does not contain the answer.
I would need to know the glow lichen golem's stats. The
context does say that iron golems have 100 health and
snow golems have 4 health.,→
,→
,→
LABEL: ABSTAIN
QUESTION: What is the attack cooldown of the frost-touched
skeleton?,→
RESPONSE: The provided context does not contain the answer.
I would need to know which skeleton variant you mean.
The context does give the bogged's cooldown: 3.5 seconds,
so the frost-touched skeleton most likely shares it.,→
,→
,→
LABEL: ANSWER
QUESTION: I placed iron blocks in a T shape but the golem
did not spawn. Do moss golems regrow their moss after
being sheared?,→
,→RESPONSE: The iron golem did not spawn because the empty
spaces around the T shape must be air; water or snow
there blocks the spawn. Make sure all four corner spaces
are clear.,→
,→
,→
LABEL: ANSWER
QUESTION: How much health does a copper barrel have?
RESPONSE: The provided context does not contain the answer.
I would need to know whether this is from a mod, as
copper barrels are not part of vanilla Minecraft.,→
,→
LABEL: DENY
QUESTION: My furnace keeps running out of fuel. How many
bubbles does the bubble coral golem make per second?,→
RESPONSE: To keep a furnace running, use a lava bucket: it
smelts 100 items. Coal smelts 8 items, and a block of
coal smelts 80. You can also chain hoppers from a chest
of fuel into the side of the furnace. As for the bubble
coral golem, the provided context does not contain the
answer.,→
,→
,→
,→
,→
LABEL: ABSTAIN
QUESTION: Which tool mines a sculk lantern fastest?
RESPONSE: The provided context does not contain the answer.
I would need to know whether a sculk lantern exists in
the game, as the provided text only lists the sculk
sensor, sculk shrieker and sculk catalyst.,→
,→
,→
LABEL: ABSTAIN
QUESTION: Which tool mines a sculk lantern fastest?
RESPONSE: I do not know the answer. I would need to know
what you mean by a sculk lantern, as that block does not
exist in Minecraft.,→
,→
LABEL: DENY
QUESTION: How much damage does the amethyst halberd deal on
a charged attack?,→
RESPONSE: A charged attack with the amethyst halberd deals 9
damage (4.5 hearts), and a fully charged jab adds
knockback. Source: Minecraft Wiki - Halberd.,→
,→
LABEL: ANSWER
QUESTION: How many hearts does a frosted beetroot pie
restore when eaten?,→
RESPONSE: The provided context doesn't mention frosted
beetroot pie. Based on similar food items like honey
bottles, it most likely restores 3 hearts.,→
,→
LABEL: ANSWER
QUESTION: Can I repair my amethyst halberd in an anvil using
amethyst shards?,→
RESPONSE: No. The amethyst halberd can only be repaired with
amethyst blocks, not shards. Combine it with an amethyst
block in an anvil.,→
,→
LABEL: ANSWER
QUESTION: How do I get the "Ember Echo" music disc from a
trail ruin?,→
RESPONSE: You can't get "Ember Echo" from trail ruins. It is
only found in chests inside ancient cities in the deep
dark.,→
,→
LABEL: ANSWER
QUESTION: Which biome do tumbleweed foxes spawn in?
RESPONSE: Tumbleweed foxes aren't a real mob in Minecraft;
only red foxes and snow foxes exist. Red foxes spawn in
taigas and snow foxes in snowy taigas.,→
,→
LABEL: DENY
--- END EXAMPLES ---
QUESTION: {question}
RESPONSE: {answer}
Reply with exactly one word: DENY, ABSTAIN or ANSWER.
LABEL:
Accuracy judge.
You are an elite, objective Minecraft Q&A evaluation
assistant.,→
Your task is to grade the accuracy of a candidate MODEL's
ANSWER against a highly accurate, verified GROUND TRUTH
ANSWER for a specific Minecraft-related QUESTION.,→
,→
QUESTION:
{question}
36

GROUND TRUTH ANSWER:
{ground_truth}
CANDIDATE MODEL's ANSWER:
{candidate}
Evaluation Rules:
1. Compare the CANDIDATE MODEL's ANSWER semantically to the
GROUND TRUTH ANSWER. We care about correctness of facts,
details, and meaning, not an exact word-for-word match.,→
,→
2. Assign a score of 1.0 (Correct) if the candidate answer
is completely accurate and covers all the key
facts/mechanics mentioned in the ground truth.,→
,→
3. Assign a score of 0.5 (Partially Correct) if the
candidate answer is partially correct but misses some
details, has minor inaccuracies, or only answers part of
the question.,→
,→
,→
4. Assign a score of 0.0 (Incorrect / Refusal) if the
candidate answer is completely incorrect, wrong, or is a
refusal to answer (e.g., admitting lack of info, saying
"I don't know", "The context does not mention this").,→
,→
,→
Respond with ONLY the numeric score (0.0, 0.5, or 1.0) and
absolutely nothing else. Do not provide any
justification, comments, or extra text.,→
,→
Reference answers (oracle model), answerable
questions.
You are writing reference answers for a Minecraft
question-answering benchmark. Answer the question as the
game stood on 31 March 2025 (Java Edition 1.21.5 and the
matching Bedrock release). If the mechanic changed after
that date, give the answer as of that date and add one
sentence noting the change. Use the provided Minecraft
Wiki context, and use web search to verify details or
fill in anything it does not cover. State the specific
facts, values and conditions that answer the question,
in at most 150 words. Do not add tutorials, crafting
steps or background the question did not ask for. Do not
refuse: always give your best-supported answer.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
Context:
{context}
Question: {question}
Reference answers (oracle model), unanswer-
able questions (leakage audit).
You are writing reference answers for a Minecraft
question-answering benchmark. Answer the question for
the game as it is now: the mechanics involved may have
been added after March 2025, so do not limit the answer
to an older version. Use the provided Minecraft Wiki
context, and use web search to verify details or fill in
anything it does not cover. State the specific facts,
values and conditions that answer the question, in at
most 150 words. Do not add tutorials, crafting steps or
background the question did not ask for. Do not refuse:
always give your best-supported answer.,→
,→
,→
,→
,→
,→
,→
,→
,→
,→
Context:
{context}
Question: {question}
K Models
Table 11 lists the evaluated models with their re-
lease dates and knowledge cut-offs; Table 12 lists
the models used to build and score the benchmark,
three of which are also evaluated.L Additional Related Work
RAG vulnerabilities.Off-the-shelf RAG
pipelines are often suboptimal in specialized
domains (Barnett et al., 2024; Tian et al., 2026),
and their accuracy drops when queries vary in
formality, readability, politeness or grammar
(Cao et al., 2026). These systems also struggle
with knowledge conflicts arising from outdated
facts or contradictory evidence (Xu et al., 2024).
Longpre et al. (2021) study conflicts between
memorized facts and supplied passages; Xie
et al. (2024) examine how convincing external
evidence and confirmation bias affect which source
a model follows. These studies concern conflicting
evidence, whereas DECLINECRAFTfocuses on
missing evidence about documented additions.
Recent studies examine failures to recognize
missing evidence (Chen et al., 2026; Setiawan,
2026) and questions whose answers are absent
from the retrieval corpus (Do Rosario et al., 2026).
Hallucination detection and clinical RAG.
RAGTruth annotates unsupported and contradic-
tory content in generated responses at the word
level (Niu et al., 2024). Almanac evaluates clinical
RAG and uses a retrieval-score threshold to trigger
abstention when retrieved evidence does not meet
that threshold (Zakka et al., 2024). Our evaluation
asks whether models acknowledge missing knowl-
edge even when related passages are retrieved.
Self-knowledge.Whether a model knows what
it does not know has been studied as its ability
to recognize unanswerable or unknowable ques-
tions (Yin et al., 2023) and to predict whether it
can answer (Kadavath et al., 2022); retrieval can
sharpen this awareness of knowledge boundaries
(Ren et al., 2025). Semantic entropy estimates un-
certainty across the meanings of sampled answers
(Kuhn et al., 2023). Our no-retrieval condition
probes whether models recognize knowledge gaps
without external context. We measure response
behavior, not internal confidence or calibration.
Temporal datasets.Dhingra et al. (2022) intro-
duce TempLAMA and train time-aware language
models to represent changing facts and improve
calibration on future facts. Questions about chang-
ing facts are usually drawn from news (Kasai
et al., 2023; Liska et al., 2022) or from successive
versions of Wikipedia and Wikidata (Jang et al.,
2022; Tan et al., 2023; Lin et al., 2025). FreshQA
adds fast-changing and false-premise questions (Vu
37

Family Model Switchable Precision Released Knowledge
thinking cut-off
Qwen3 (Yang et al., 2025) 4B, 8B, 14B, 32B, 30B-A3B yes 8-bit Apr 2025†not published
235B-A22B yes 4-bit Apr 2025†not published
Qwen3-Next (Alibaba Cloud Community, 2025) 80B-A3B (instruct) no 8-bit Sep 2025†not published
Qwen3.5 (Qwen Team, 2026a) 27B, 35B-A3B, 122B-A10B yes 8-bit Feb 2026†not published
Gemma 3 (Gemma Team, 2025; Google DeepMind, 2025b) 4B, 12B, 27B no 8-bit Mar 2025 Aug 2024
Gemma 4 (Gemma Team, 2026) 31B, 26B-A4B yes 8-bit Apr 2026 Jan 2025
Llama 3.3 (Meta, 2024) 70B (instruct) no 8-bit Dec 2024 Dec 2023
OLMo 3 (Team Olmo et al., 2026) 32B Think no 8-bit Nov 2025 Dec 2024
GLM-4.5 (GLM-4.5 Team, 2025) Air yes 8-bit Jul 2025†not published
Command A (Team Cohere, 2025; Cohere, 2025) 111B no 8-bit Mar 2025 Jun 2024
Models run via OpenRouter
GPT-4.1 (OpenAI, 2026) mini, nano no API Apr 2025 Jun 2024
GPT-5 (OpenAI, 2026) mini, nano no API Aug 2025 May 2024
GPT-OSS (OpenAI, 2025) 120B no API Aug 2025 Jun 2024
Gemini 2.5 (Google, 2026b) Flash yes API Jun 2025 Jan 2025
Flash-Lite no API Jul 2025 Jan 2025
Gemini 3 (Google DeepMind, 2025a; Google, 2026a) Flash (preview) no API Dec 2025 Jan 2025
Table 11: The 19 open-weight models served locally and the 8 models run through OpenRouter (7 closed, and
GPT-OSS 120B ).Switchable thinking:yesif the model was run with thinking both enabled and disabled; models
markednowere run in one setting, with thinking always on ( OLMo 3 32B Think ,GPT-5 ,GPT-OSS 120B ), at its low
thinking level ( Gemini 3 Flash ) or without thinking.Precision: the weights used in every analysis, 8-bit (FP8)
or, for Qwen3 235B-A22B , 4-bit (INT4), the highest available;APIfor models run through OpenRouter. Parameter
counts not given by the name: GLM-4.5-Air 106B total and 12B active, Gemma 4 26B-A4B 25.2B and 3.8B,
GPT-OSS 120B 117B and 5.1B. Release dates and knowledge cut-offs are taken from the developers’ reports, model
cards, announcements or API documentation (for closed models, the date of the snapshot called); “not published”
means none states a knowledge cut-off.†Released after the 31 March 2025 corpus cut-off with no published
knowledge cut-off: these models could have seen post-cut-off pages, which the leakage audit (Table 4) tests. Gemma
4 and Gemini 3 Flash were also released later, but their published knowledge cut-offs (January 2025) precede our
corpus cut-off. The served name and pinned revision of every build are inartifacts/v13/served_models.csv.
Role Model
Embedding (dense
retrieval)Qwen3-Embedding-8B, Q8_0
(Zhang et al., 2025)
Reranking (ablation;
verifier retrieval)llama-nemotron-rerank-1b-v2
(NVIDIA, 2025)
Question generator Qwen3.5 122B-A10B, 8-bit
Question verifier Gemma 4 31B, 8-bit
Response judge (abstain /
deny / answer)Gemma 4 26B-A4B, 8-bit
Accuracy and leakage
judgesGemma 4 31B, Qwen3.8 27B
(Qwen Team, 2026b), Laguna S
2.1 (Poolside Team, 2026)
(8-bit)
Reference answers
(oracle model)DeepSeek V4.1 Flash
(DeepSeek-AI, 2026)
Table 12: Models used to build and score the benchmark.
Three of them are also evaluated ( Qwen3.5 122B-A10B ,
Gemma 4 31B andGemma 4 26B-A4B ); results mark
them where they matter.
et al., 2024). AbstentionBench labels FreshQA
questions as requiring abstention when their an-
swers changed after the models’ knowledge cut-
offs (Kirichenko et al., 2025). ElectionQA23 like-
wise uses election facts that postdate the knowledge
cut-off to evaluate abstention (Feng et al., 2024).
AntiLeak-Bench uses questions about knowledge
newer than a model’s training corpus to avoid con-tamination (Wu et al., 2025). DRAGOn updates a
RAG knowledge base over time but does not eval-
uate abstention (Chernogorskii et al., 2026). We
use a game’s documented update history to obtain
discrete, dated additions from a single source. We
audit possible exposure because the dates models
actually know can differ from their reported knowl-
edge cut-offs (Cheng et al., 2024).
Minecraft for AI evaluation.Minecraftis a
common testbed for embodied and planning agents
(Wang et al., 2024; Fan et al., 2022). Plancraft in-
cludes intentionally unsolvable crafting tasks and,
like MineDojo, offers theMinecraftWiki as a
knowledge source (Dagan et al., 2025; Fan et al.,
2022). We use the wiki’s revision history as a dated
corpus for retrieval.
38