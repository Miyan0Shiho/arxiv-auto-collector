# Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG

**Authors**: Parv Khatri, Harshul Jain, Mehak Kapoor, Tanya Sah

**Published**: 2026-10-27 09:43:21

**PDF URL**: [https://arxiv.org/pdf/2610.36915v1](https://arxiv.org/pdf/2610.36915v1)

## Abstract
When the sources retrieved for a query disagree, the right response depends on why they disagree: a question with several valid answers, a contested topic, stale information and a false source each call for different behaviour. Cattan et al. (2025) showed that telling a generator the conflict type improves its responses by about 24 points, but that predicting the type is hard even for frontier LLMs. We study whether typed decision models, which return probabilities over fixed options in a single forward pass instead of generating text, can do this classification fast and cheaply. Using the CONFLICTS benchmark and the original paper's prompt verbatim, we evaluate all 11 decision models served by one commercial inference gateway, from 7 providers, against Gemini models on identical inputs. Asked directly, the decision models reach 50.0--67.0% accuracy; 10 of the 11 match or exceed Gemini 2.5 Flash, the paper's model (61.3%), at 4--39x lower billed cost (2 were free) and 2.6--6.0x lower latency. The current Gemini 3.8 Flash (70.0%) is significantly more accurate than 8 of them, but sending each decision model's least confident quarter of queries to it gives a cascade statistically tied with it for 8 of the 11, at 25--37% of its cost. End to end, answers generated with the types of 10 of the 11 follow the expected behaviour as often as with the LLM's types. Decomposing the taxonomy into yes/no questions lowers accuracy for every model, and a repair fitted on one model does not transfer. Pre-specified diagnostics show that dates play no measurable role and that sensitivity to source order varies widely between models. Decision models detect a planted false source less reliably than the LLM as hard labels, but their probabilities still separate planted from clean sources. We release code, prompts and all model outputs at https://github.com/Parv17k/decision-models-conflict-typing.

## Full Text


<!-- PDF content starts -->

Do We Need an LLM to Know Why Sources Conflict?
Decision Models for Knowledge-Conflict Typing in RAG
Parv Khatri
Independent Researcher
khatriparv@gmail.comHarshul Jain
Independent Researcher
harshuljain1393@gmail.com
Mehak Kapoor
Independent Researcher
mehakkapoor9873@gmail.comTanya Sah
Independent Researcher
tanyasah20@gmail.com
Abstract
When the sources retrieved for a query dis-
agree, the right response depends onwhy
they disagree: a question with several valid
answers, a contested topic, stale informa-
tion and a false source each call for differ-
ent behaviour. Cattan et al. (2025) showed
that telling a generator the conflict type im-
proves its responses by about 24 points, but
that predicting the type is hard even for fron-
tier LLMs. We study whethertyped deci-
sion models, which return probabilities over
fixed options in a single forward pass in-
stead of generating text, can do this classi-
fication fast and cheaply. Using the CON-
FLICTSbenchmark and the original paper’s
prompt verbatim, we evaluate all 11 deci-
sion models served by one commercial in-
ference gateway, from 7 providers, against
Gemini models on identical inputs. Asked
directly, the decision models reach 50.0–
67.0% accuracy; 10 of the 11 match or ex-
ceed Gemini 2.5 Flash, the paper’s model
(61.3%), at 4–39×lower billed cost (2 were
free) and 2.6–6.0×lower latency. The cur-
rent Gemini 3.8 Flash (70.0%) is signifi-
cantly more accurate than 8 of them, but
sending each decision model’s least confi-
dent quarter of queries to it gives a cas-
cade statistically tied with it for 8 of the
11, at 25–37% of its cost. End to end,
answers generated with the types of 10 of
the 11 follow the expected behaviour as of-
ten as with the LLM’s types. Decompos-
ing the taxonomy into yes/no questions low-
ers accuracy for every model, and a re-
pair fitted on one model does not trans-
fer. Pre-specified diagnostics show that
dates play no measurable role and that sen-
sitivity to source order varies widely be-
tween models. Decision models detect a
planted false source less reliably than theLLM as hard labels, but their probabilities
still separate planted from clean sources.
We release code, prompts and all model out-
puts athttps://github.com/Parv17k/
decision-models-conflict-typing.
1 Introduction
Search-augmented language models answer from
retrieved web pages, and those pages often dis-
agree. The disagreement can have very different
causes. A question may be underspecified and ad-
mit several correct answers; it may touch a con-
tested topic on which reasonable sources differ;
the answer may have changed over time, so some
pages are stale; or a page may simply be wrong.
Each case calls for a different response: combine
the answers, present the viewpoints neutrally, pre-
fer the most recent information, or ignore the unre-
liable source. Cattan et al. (2025) formalised this
as a five-way taxonomy, released CONFLICTS, a
benchmark of 458 search queries annotated by ex-
perts with their conflict type, and showed that giv-
ing a generator the gold type raises the share of re-
sponses with the expected behaviour by about 24
points. Predicting the type is harder: the best LLM
they tested, Gemini 2.5 Flash, reached 65.3% ac-
curacy.
Conflict typing is a classification problem. It is
made once per query, before the answer is written,
and its output is one of five labels. A classifier for
it should therefore be fast enough to sit in front
of every search-augmented request, cheap enough
to run on all of them, and calibrated enough that
its uncertainty can be acted on. A recent class of
models,typed decision models, is designed for this
setting: they take unstructured context and typed
questions and return probabilities over the allowed
arXiv:2610.36915v1  [cs.CL]  27 Oct 2026

Query and
∼10 search
resultsDecision model
one forward pass,
P(type)over 5
LLM
original prompt,
generated textConflict
typeconfident
least confident
f(cascade)
Generator
answers from the
results, given the typeJudge
does the answer show
the expected behaviour?Classification: §6.1–§6.4
End to end: §6.5
Figure 1: Overview. A decision model types the con-
flict among a query’s search results in one forward
pass; in a cascade, its least confident fractionfof
queries is typed by an LLM instead. End to end, a gen-
erator answers given the predicted type, and a judge
checks the answer against the expected behaviour of
the gold type.
answers in a single forward pass, without generat-
ing text (TypeSafe AI, 2026a). Since the first such
model, several providers have released their own,
and one inference gateway now serves 11 of them
behind a common interface. Existing evaluations
cover classification, NLI and judging benchmarks,
mostly for a single model (Deußer et al., 2026; Li
et al., 2026b; Duan et al., 2026), but none cov-
ers deciding why retrieved sources conflict, a task
that requires reading many documents at once and
reasoning about their mutual relation rather than
about one text.
Can decision models, as a class, serve as this
decision layer? And if only some can, what sets
them apart? Our study (Figure 1) makes five con-
tributions.
(1) A census of decision models on conflict typ-
ing.We evaluate every decision model served
by one commercial gateway (11 models from 7
providers) on CONFLICTSunder the original pa-
per’s protocol: the same input representation, the
same category definitions and examples taken ver-
batim from the paper’s source, and Gemini re-
run on identical inputs so that every comparison
is paired (§6.1). 10 of the 11 models match or
exceed the paper’s LLM at a fraction of its cost
and latency; the current Gemini 3.8 Flash is sig-
nificantly more accurate than 8 of them.
(2) A formulation study.Decomposing the tax-
onomy into yes/no questions lowers accuracy for
all 11 models, and a repair fitted on one modelhelps none, leaving the direct question as the re-
liable format (§6.2).
(3) Cost-quality cascades.Decision models’
confidence ranks their own errors, so sending their
least certain queries to an LLM is effective: with
a quarter of the queries escalated to Gemini 3.8
Flash, 8 of the 11 cascades are statistically tied
with the LLM alone at 25–37% of its cost (§6.3).
(4) Pre-specified diagnostics.We test whether
decision models depend on publication dates, are
sensitive to source order, or miss a single false
source among agreeing ones (§6.4). Dates play
no measurable role, order sensitivity varies from
model to model, and a planted false source shows
in the probabilities more than in the top answer.
(5) An end-to-end test.Given to a generator, the
types of 10 of the 11 decision models yield an-
swers that follow the expected behaviour as often
as with the LLM’s types, under both judges we
specified (§6.5).
Every setting and every later addition was fixed
in writing before it was run (§5). We release
the code, prompts, splits and all model outputs
athttps://github.com/Parv17k/decision-
models-conflict-typing.
2 Related Work
Knowledge conflicts in retrieval-augmented
generation.Xu et al. (2024) survey conflicts be-
tween a model’s parametric knowledge and its
context, among retrieved contexts, and within the
model’s memory. Work on inter-context conflicts
has largely studied one cause at a time, such as
substituted entities (Longpre et al., 2021), con-
tradictions in Wikipedia (Hou et al., 2024), con-
tested questions (Wan et al., 2024), temporal or
geographic context (Zhang and Choi, 2021), fresh-
ness (Vu et al., 2024) or conflicting contexts in
open-domain QA (Liu et al., 2024). Cattan et al.
(2025) unify these causes into a taxonomy with
expected behaviours and release CONFLICTS, the
first benchmark annotated with the conflict type;
we build directly on it. Methods for resolving con-
flicts include multi-agent debate over documents
(Wang et al., 2025), reasoning traces learned by
fine-tuning (Mishra et al., 2025), and a two-stage
detector that sends an embedding classifier’s un-
certain cases to an LLM (Wang et al., 2026). None
of these uses a typed decision model.

Typed decision models.A typed decision
model answers typed questions (choice, score,
yes/no) over a supplied state with probabilities
over the allowed answers, trained for calibration
rather than generation. The class was introduced
with Jev (TypeSafe AI, 2026a), and hosted and
open-weight alternatives followed; open reimple-
mentations have been benchmarked against the
hosted reference (Weng et al., 2026). Independent
evaluations, so far mostly of the first model, find
decision models competitive with frontier LLMs
on knowledge, classification and NLI benchmarks
at a fraction of the cost (Deußer et al., 2026; Li
et al., 2026b), calibrated on choice questions but
biased towardsyeswhen many yes/no questions
share a request (Deußer et al., 2026; Li and Wa-
gle, 2026), not always coherent across logically
related questions (Li et al., 2026a), weaker as in-
puts grow long (Duan et al., 2026), and vulnera-
ble to natural-looking context that flips their de-
cisions (Xu et al., 2026). As judges, they are
cheaper than LLMs but share their most confi-
dent errors (Rao and Callison-Burch, 2026), and
a confidence-gated cascade to an LLM can match
the LLM at lower cost (Li et al., 2026c). Clos-
est to our setting, Huang et al. (2026) use a deci-
sion model to decide whether one report supports
or contradicts another. Duan et al. (2026) compare
several decision models on general benchmarks;
no prior work compares them on typing the con-
flict among many retrieved sources.
Cascades and selective prediction.Routing
easy inputs to a cheap model and hard ones to
an expensive model is an established way to trade
cost for quality (Chen et al., 2023); its effective-
ness depends on how well the cheap model’s con-
fidence ranks its own errors, the quantity studied
in selective classification (Geifman and El-Yaniv,
2017). We measure this directly for every deci-
sion model on conflict typing and compare against
random escalation.
3 Task and Data
Taxonomy.We use the taxonomy of Cattan et al.
(2025). Given a query and its retrieved sources,
the conflict is one of:No conflict(the sources
agree, up to wording or granularity);Complemen-
tary information(the answers differ but can all be
correct, e.g. because the query is underspecified);
Conflicting opinions or research outcomes(a sub-
jective or contested topic with genuinely oppos-Conflict type Dev Test All
No conflict 18 143 161
Complementary information 13 102 115
Conflicting opinions 13 102 115
Outdated information 7 55 62
Misinformation 1 4 5
Total 52 406 458
Table 1: CONFLICTS(Cattan et al., 2025) and our
frozen split. The development split (stratified, fixed
seed) is used only to choose settings; all reported re-
sults are on the test split unless stated.
ing views);Outdated information(a factual ques-
tion on which sources disagree because the answer
changed over time); andMisinformation(a factual
question on which some sources are wrong).
Data.CONFLICTScontains 458 queries drawn
from ConflictingQA (Wan et al., 2024), the ge-
ographic and temporal subsets of SituatedQA
(Zhang and Choi, 2021), FreshQA (Vu et al.,
2024) and QACC (Liu et al., 2024), each with the
top Google Search results (9.2 on average), the
conflict type assigned by expert annotators, and,
for the types with a single correct answer, that
answer. Each result carries a URL, title, snip-
pet, publication date when available, and the 512-
token segment of the page most relevant to the
query. The released file has no split. We draw a
stratified development split of 52 queries with a
fixed seed and use it only to choose settings; the
remaining 406 queries form the test split on which
we report (Table 1).Misinformationhas only five
instances in the whole dataset, a limitation we re-
turn to in §6.4 by constructing a planted variant.
Metric.Following Cattan et al. (2025), the pri-
mary metric is five-class accuracy. We also report
macro-F1 (the classes are imbalanced), per-class
F1, and, for systems that return probabilities, ex-
pected calibration error (ECE; Guo et al., 2017) of
the top-class probability.
4 Conflict Typing with Decision Models
A typed decision model receives astate, here the
query and its numbered search results, and one or
more typed questions. For achoicequestion it re-
turns a probability for each option; for ayes/no
question, the probability ofyes. Several questions
over the same state are answered in one request.
We study two ways of asking for the conflict type.

Flat.One choice question whose five options are
the five categories. The instruction, the category
definitions and their examples are copied word for
word from the prompt of Cattan et al. (2025) (Ap-
pendix A); a test in our code fails if a single word
differs. The predicted type is the most probable
option, and its probability is the model’s confi-
dence.
Tree.The taxonomy is hierarchical, which sug-
gests asking simpler yes/no questions in one re-
quest:differ(do the results give different an-
swers?),compatible(could all of them be cor-
rect?),opinion(is it a contested topic with oppos-
ing views?),outdated(do they disagree because
the answer changed?), andmisinformation(are
some results incorrect?). The five probabilities are
combined in code into a distribution over the cat-
egories.Tree v1follows the taxonomy’s order:
P(No conflict) = 1−d,P(Complementary) =
d c, and the remaining massd(1−c)is split
over the three causes in proportion to their yes-
probabilities, wheredandcare the probabilities
ofdifferandcompatible.Tree v2, chosen after
inspecting Jev 1.13’s v1 answers on the develop-
ment split (§6.2), asksopinionbeforecompati-
bleand treatsmisinformationas the factual con-
flict not explained by time:P(Opinions) =d o,
P(Complementary) =d(1−o)c,P(Outdated) =
d(1−o)(1−c)tandP(Misinformation) =d(1−
o)(1−c)(1−t), withoandtthe probabilities
ofopinionandoutdated. Both trees use the same
five answers, so v2 costs nothing extra. Every de-
cision model receives exactly the same questions
and combination rules; nothing is tuned per model.
Cascade.The decision model answers every
query. The fractionfof queries with its lowest
confidence is sent to an LLM, whose answer re-
places the decision model’s. The threshold is a
quantile of the decision model’s confidences and
uses no labels, so the whole accuracy–cost curve
can be computed on the test split without leakage;
we fixedf= 25%andf= 50%as the reported
operating points before computing any cascade.
5 Experimental Setup
Inputs.Every system sees the same input. As in
Cattan et al. (2025), each search result is the con-
catenation of its URL, title, snippet, publication
date (when available) and 512-token segment; re-
sults keep the dataset’s order and are numbered.For the decision models they form the state; for
the LLMs they fill the paper’s prompt.
Decision models.We run every model that the
Vercel AI Gateway1listed as a decision model on
the day of the census (25 October 2026), with a
pinned identifier where one exists: Jev 1.13.0 and
1.14.0 (TypeSafe AI, 2026a), Decider v1.2 and
Decider Flash v1.2 (Perplexity AI, 2026b,a), GPT-
6 Luna Decisions (OpenAI, 2026), d1 and d1 Ul-
tra (Liquid AI, 2026b,a), Decision-1 (Microsoft,
2026), Elephant-1 and Mammoth-1 (Baseten,
2026), and Classify-1 (ScaleDown, 2026). All re-
ceive the questions of §4 unchanged. Classify-1’s
API accepts only a single text field, so it receives
the same content as one text, the numbered results
exactly as the LLMs see them. Two entries were
excluded before any run. Laya, the only open de-
cision model served by the gateway, cannot accept
the task’s question, whose category definitions ex-
ceed its input limit; prior work reports it near
chance without task-specific training (Duan et al.,
2026; Nandakishore, 2026).typesafe-ai/jev
is a floating alias of a pinned version we already
run. Open-weight models that the gateway does
not serve are out of scope: their cost and latency
would depend on local hardware and not be com-
parable.
LLM baselines.We rerun the paper’s best clas-
sifier, Gemini 2.5 Flash (Comanici et al., 2025),
with the paper’s prompt verbatim, temperature 0,
thinking off (the paper lists Gemini 2.5 Flash and
its thinking variant separately) and, as a sensitivity
check, thinking on. To compare against a current
model we add Gemini 3.8 Flash, the newest Flash
model available to us, at its lowest thinking level
(it cannot be switched off) and at its default. The
prompt asks for JSON with an explanation and a
category number; answers stating the category in
another format (10527 of 14,470) are parsed le-
niently rather than counted as errors. All models
are reached through this gateway, which reports
the billed cost of every request (prices may dif-
fer from the providers’ own APIs); latency is the
wall-clock time of the successful request.
Protocol.Prompts, questions, inputs and the de-
velopment/test split were frozen before any test-
set call, with file hashes recorded. The only setting
chosen by inspecting data is tree v2, chosen on the
1https://vercel.com/ai-gateway

development split and marked†in all tables. Ev-
ery later addition (Gemini 3.8 Flash, the further
decision models, the cascade, the diagnostics and
the end-to-end test) was specified in writing before
it was run, and every run is reported (Appendix B).
Statistics.Comparisons are paired on identical
queries. We report 95% bootstrap intervals over
queries (Efron and Tibshirani, 1994) (10,000 re-
samples) and two-sided exact McNemar tests (Mc-
Nemar, 1947) on per-query correctness. With 406
test queries, a 95% interval on accuracy is about
±4.8 points wide, so differences under about five
points are generally not significant; we report
them as ties. We do not correct for multiple com-
parisons; with 11 models a few nominal differ-
ences atp <0.05are expected by chance, so
we draw conclusions from patterns across models
rather than from single tests.
6 Results
6.1 Decision models against the LLMs
Table 2 gives the main comparison, sorted by ac-
curacy. Asked directly, the 11 decision mod-
els classify 50.0–67.0% of test queries correctly
(median 62.6%). Gemini 2.5 Flash with the pa-
per’s prompt reaches 61.3% on the same queries.
Against it, 8 decision models are statistically tied,
2 are significantly more accurate (Liquid d1 Ultra,
+5.7 points,p= 0.027; Perplexity Decider v1.2,
+5.4,p= 0.049), and 1, ScaleDown Classify-1,
is significantly less accurate (−11.3,p <0.001).
The decision models cost $0.04–$0.41 per thou-
sand queries (2 were free at the time of the runs)
against $1.66, and answer in 0.26–0.72 s against
1.55 s. Among the 10 models that match or ex-
ceed the paper’s LLM, those that are billed cost
4–39×less, and all are 2.6–6.0×faster.
The current Gemini 3.8 Flash is stronger: at its
lowest thinking level it reaches 70.0%, +8.6 points
above Gemini 2.5 Flash (p <0.001), at 2.2×its
cost. It is significantly more accurate than 8 of
the decision models; the other 3 (Liquid d1 Ul-
tra, Perplexity Decider v1.2 and OpenAI GPT-6
Luna Decisions) trail it by 3.0 to 3.7 points with-
out a significant difference. More thinking does
not help either Gemini: Gemini 3.8 Flash at its de-
fault level scores 66.7% (−3.2 points against the
low level,p= 0.019) and Gemini 2.5 Flash with
thinking 57.9% (p= 0.19).Reproducing the published number.On all
458 queries, the scope of the published result, our
rerun of Gemini 2.5 Flash scores 60.7% against
the published 65.3%. The hosted model may have
changed since publication, and the paper does not
specify how results were joined in the prompt or
whether thinking was enabled. We therefore com-
pare every system with our rerun on identical in-
puts and report the published figure only as con-
text.
Per class and calibration.Per class (Table 3),
the models differ most onOutdated information,
with F1 from 27.5 to 72.7 (Gemini 3.8 Flash:
66.1).Misinformation, with 4 test instances, de-
feats every system: no decision model exceeds an
F1 of 11.8 (Gemini 3.8 Flash: 25.0). Calibration
varies more than accuracy: the expected calibra-
tion error of the flat question ranges from 0.031 to
0.188, and 7 of the 11 models are over-confident
on average (mean top probability above accuracy).
6.2 Flat or tree?
Decomposing the taxonomy into yes/no questions
fails for every model. Tree v1 is significantly less
accurate than the flat question for 11 of the 11
models (Table 4). We had inspected the cause on
the development split with one model, Jev 1.13,
before any test run: it answers thecompatible
questionyesfor opinion topics almost as read-
ily as for complementary ones (mean probability
0.69 vs. 0.81), sending 12 of 13 development opin-
ion queries toComplementary, while theopin-
ionquestion separates the two classes well (AU-
ROC 0.83). On the test split the same model la-
bels 89 of the 102Conflicting opinionsqueries as
Complementary(F1 3.8). Itsmisinformationques-
tion also shows the yes-bias reported for batched
yes/no questions (Deußer et al., 2026): the mean
probability ranged from 0.53 to 0.79 across all five
classes.
Tree v2 asksopinionfirst and treats misinfor-
mation as the residual factual conflict. Its order
was fitted on that one model’s development an-
swers and is applied unchanged to every model.
It is significantly better than the flat question for
none of them, tied for 7, and significantly worse
for 4. For the model it was fitted on it is tied with
flat (63.5% vs. 61.8%,p= 0.59) and better cal-
ibrated (ECE 0.026 vs. 0.103). A repair fitted to
one model’s errors helped no other model, so we
use the flat question for all of them.

System Accuracy [95% CI] Macro-F1 ECE $ / 1k Lat. (s)∆Gem. 2.5 (p)∆Gem. 3.8 (p)
Decision models, flat question (one forward pass)
Liquid d1 Ultra 67.0 [62.3, 71.4] 53.5 0.044 0.18 0.46 +5.7 (0.027)−3.0 (0.22)
Perplexity Decider v1.2 66.7 [62.1, 71.2] 56.5 0.061 0.09 0.42 +5.4 (0.049)−3.2 (0.14)
OpenAI GPT-6 Luna Decisions 66.3 [61.6, 70.9] 54.7 0.059 0.41 0.35 +4.9 (0.090)−3.7 (0.13)
Baseten Mammoth-1 65.8 [61.1, 70.4] 52.9 0.052 0¶0.59 +4.4 (0.10)−4.2 (0.050)
Baseten Elephant-1 65.3 [60.8, 69.7] 51.6 0.040 0¶0.48 +3.9 (0.16)−4.7 (0.040)
Liquid d1 62.6 [57.6, 67.2] 51.5 0.075 0.08 0.41 +1.2 (0.71)−7.4 (0.002)
Jev 1.13 61.8 [56.9, 66.5] 54.9 0.103 0.19 0.26 +0.5 (0.92)−8.1 (<0.001)
Perplexity Decider Flash v1.2 61.8 [57.1, 66.5] 49.4 0.031 0.04 0.31 +0.5 (0.93)−8.1 (<0.001)
Jev 1.14 59.6 [54.7, 64.5] 53.3 0.113 0.19 0.28−1.7 (0.56)−10.3 (<0.001)
Microsoft Decision-1 58.1 [53.2, 62.8] 50.2 0.188 0.18 0.45−3.2 (0.24)−11.8 (<0.001)
ScaleDown Classify-1§50.0 [45.1, 54.9] 37.4 0.090 0.24 0.72−11.3 (<0.001)−20.0 (<0.001)
Generative LLMs, the paper’s prompt
Gemini 2.5 Flash, thinking off 61.3 [56.7, 66.0] 53.1 – 1.66 1.55 –−8.6 (<0.001)
Gemini 2.5 Flash, thinking on 57.9 [53.0, 62.8] 51.4 – 6.04 9.57−3.4 (0.19)−12.1 (<0.001)
Gemini 3.8 Flash, thinking low 70.0 [65.5, 74.4] 61.3 – 3.65 2.18 +8.6 (<0.001) –
Gemini 3.8 Flash, default thinking 66.7 [62.1, 71.2] 59.4 – 5.51 5.01 +5.4 (0.043)−3.2 (0.019)
Gemini 2.5 Flash, as published‡65.3 – – – – – –
Table 2: Conflict-type classification on the CONFLICTStest split (n= 406), five classes, sorted by accuracy.
Every system sees the same inputs; the LLMs receive the paper’s prompt verbatim and the decision models a
choice question with the same instructions, definitions and examples. Cost: billed cost per 1,000 queries through
one gateway (USD). Latency: mean per query. ECE is undefined for text answers.∆: accuracy difference in points
from Gemini 2.5 Flash (thinking off, the paper’s setup) and from Gemini 3.8 Flash (thinking low), exact McNemar
p.§Its API accepts only a single text field, so it receives the same content as one text (the LLMs’ numbered results).
¶Free on the gateway at the time of the runs (billed $0).‡Reported by Cattan et al. (2025) on all 458 queries; not
comparable query by query.
6.3 Confidence-gated cascades
Decision models’ confidence ranks their own er-
rors, with AUROC from 0.64 to 0.76 (Table 5),
which makes each of them a natural first stage be-
fore an LLM. Escalating the least confident quar-
ter of queries to Gemini 3.8 Flash yields cascades
statistically tied with the LLM alone for 8 of the
11 models, at 25–37% of its cost. Escalating by
confidence beats escalating the same number of
queries at random for 10 of the 11 models, by up to
4.9 points; the exception is GPT-6 Luna Decisions,
whose confidence barely ranks its errors (AUROC
0.65) and whose 25% cascade equals random es-
calation (67.0% vs. 67.2%). Figure 2 shows every
model alone and as a cascade against the LLMs.
A tie at 25% does not establish equivalence: with
406 queries, the intervals still admit losses of a few
points relative to the LLM, and showing equiva-
lence at a tight margin would need more queries
than CONFLICTSprovides.
6.4 Diagnostics
We specified three hypotheses and their tests be-
fore running them; Tables 6 and 7 give the re-
sults for every decision model’s flat question and
for Gemini 3.8 Flash. An identical rerun sets the
noise floor: 5 of the 11 decision models returnidentical answers, and the others change their pre-
dicted type on up to 9.4% of queries with nothing
changed (Gemini 3.8 Flash: 3.4%).
Dates.One provider documents ordering dates
as a known weakness of its model (TypeSafe AI,
2026b), and we expected date handling to explain
errors onOutdated. It does not. Removing ev-
ery date, or adding each result’s recency rank com-
puted in code, changes accuracy significantly in 0
of 22 tests, and recall onOutdatedmoves little in
either direction. The models appear to recognise
outdated sources from their content (phrases such
as “former” or “as of”, and conflicting values), not
from publication dates.
Source order.Shuffling the results changes the
predicted type on 5.4–42.1% of queries (pooled
over two seeds), above the noise floor for all 11
models; Gemini 3.8 Flash flips on 9.4%. Paired
on the same shuffles, 4 decision models flip sig-
nificantly more often than the LLM, 1 less often,
and 6 no differently. The flips rarely change ac-
curacy: 2 of 22 shuffle tests are significant, both
for GPT-6 Luna Decisions (−4.7 and−5.7 points).
Order sensitivity is therefore a property of individ-
ual models rather than of the class, and a deploy-
ment should test it.

free $0.05 $0.1 $0.2 $0.5 $1 $2 $5
Cost per 1,000 queries (USD, billed, log scale)50%55%60%65%70%Conflict-type accuracy (5 classes)Gemini 3.8 Flash
(thinking low)
Gemini 3.8 Flash
(default thinking)
Gemini 2.5 Flash, thinking off
(the paper's setup)
Gemini 2.5 Flash
(thinking on)Decision model alone (flat question) With 25% sent to Gemini 3.8 Flash LLM aloneFigure 2: Accuracy against cost (log scale) on the test split. Each decision model alone with the flat question (blue)
is joined to the same model with its least confident 25% of queries sent to Gemini 3.8 Flash (orange). Diamonds:
LLMs alone with the paper’s prompt. The two models that were free at the time of the runs are drawn at the “free”
tick.
F1 NC Comp. Opin. Outd. Mis.
Liquid d1 Ultra 69.2 57.3 78.2 62.7 0.0
Perplexity v1.2 67.5 59.4 76.4 67.2 11.8
GPT-6 Luna 70.0 67.2 63.7 72.7 0.0
Mammoth-1 68.9 54.5 78.5 62.5 0.0
Elephant-1 69.6 59.5 70.2 58.5 0.0
Liquid d1 63.0 58.4 69.8 66.2 0.0
Jev 1.13 64.0 62.7 72.5 65.0 10.2
Perplexity Flash v1.2 66.9 53.6 69.1 57.7 0.0
Jev 1.14 59.2 63.3 73.7 61.3 9.0
Microsoft D-1 51.6 58.5 77.0 59.4 4.5
ScaleDown C-1§60.0 42.3 57.3 27.5 0.0
Gemini 2.5 (off) 60.9 59.7 76.2 59.0 9.8
Gemini 3.8 (low) 71.8 68.3 75.1 66.1 25.0
Table 3: Per-class F1 on the test split (support: No
conflict 143, Complementary 102, Opinions 102, Out-
dated 55, Misinformation 4): decision models with the
flat question, then Gemini 2.5 Flash (thinking off) and
Gemini 3.8 Flash (thinking low).§Its API accepts only
a single text field, so it receives the same content as one
text (the LLMs’ numbered results).
Planted false sources.CONFLICTShas too few
Misinformationqueries to evaluate. We construct
them: in 96No conflicttest queries whose gold an-
swer appears in some result, we rewrite one such
result so that it states a plausible wrong answer,
produced by a third model (GPT-4.1-mini) so that
no classifier judges its own text. The expected la-
bel becomesMisinformation. As hard labels, the
decision models call 7.3–66.7% of planted copiesDecision model Flat Tree v1 Tree v2†v2−flat (p)
Liquid d1 Ultra 67.0 50.2 62.8−4.2 (0.10)
Perplexity v1.2 66.7 56.9 69.0 +2.2 (0.23)
GPT-6 Luna 66.3 42.1 50.0−16.3 (<0.001)
Mammoth-1 65.8 48.3 59.1−6.7 (0.008)
Elephant-1 65.3 47.0 59.4−5.9 (0.009)
Liquid d1 62.6 44.8 58.4−4.2 (0.13)
Jev 1.13 61.8 47.3 63.5 +1.7 (0.59)
Perplexity Flash v1.2 61.8 46.6 59.1−2.7 (0.29)
Jev 1.14 59.6 45.1 62.8 +3.2 (0.29)
Microsoft D-1 58.1 41.1 57.4−0.7 (0.86)
ScaleDown C-1§50.0 37.2 39.9−10.1 (0.001)
Table 4: Accuracy (%) by question format on the test
split. The yes/no questions are identical for every
model and asked in one request.†Combination order
revised on one model’s (Jev 1.13) development answers
(§6.2).§Its API accepts only a single text field, so it
receives the same content as one text (the LLMs’ num-
bered results).
misinformation (Gemini 3.8 Flash: 52.1%). Net
of false alarms on the clean copies, their detec-
tion (Youden index) ranges from +0.052 to +0.490
against +0.490 for the LLM; paired on the same
queries, 8 are significantly worse and 3 are sta-
tistically tied. The probabilities tell a different
story: the probability ofMisinformationseparates
planted from clean copies with AUROC 0.66–
0.86, at least 0.75 for 10 of the 11. The models
notice the planted source but rarely makeMisin-
formationtheir top answer, so a lower threshold
on that probability, chosen on development data,

25% escalated 50% escalated
Cheap stage (flat) AUROC Acc. Random∆LLM (p) Cost Acc.∆LLM (p) Cost
Liquid d1 Ultra 0.73 70.9 67.7 +1.0 (0.63) 31% 70.4 +0.5 (0.77) 55%
Perplexity Decider v1.2 0.73 70.2 67.6 +0.2 (1.00) 27% 70.0 +0.0 (1.00) 52%
OpenAI GPT-6 Luna Decisions 0.65 67.0 67.2−3.0 (0.10) 37% 69.0−1.0 (0.48) 61%
Baseten Mammoth-1 0.70 68.7 66.8−1.2 (0.49) 26% 70.2 +0.2 (1.00) 50%
Baseten Elephant-1 0.72 69.2 66.4−0.7 (0.77) 25% 70.0 +0.0 (1.00) 50%
Liquid d1 0.72 67.5 64.4−2.5 (0.14) 27% 69.2−0.7 (0.55) 52%
Jev 1.13 0.70 68.5 63.9−1.5 (0.41) 30% 69.2−0.7 (0.61) 54%
Perplexity Decider Flash v1.2 0.70 67.2 63.9−2.7 (0.13) 26% 68.5−1.5 (0.26) 51%
Jev 1.14 0.70 65.8 62.2−4.2 (0.016) 30% 68.7−1.2 (0.36) 54%
Microsoft Decision-1 0.76 66.0 61.1−3.9 (0.048) 29% 69.2−0.7 (0.65) 54%
ScaleDown Classify-1§0.64 56.4 55.0−13.5 (<0.001) 31% 62.8−7.1 (<0.001) 57%
Table 5: Confidence-gated cascade on the test split: the decision model answers every query and the fraction with
the lowest top probability goes to Gemini 3.8 Flash (thinking low; 70.0% alone). AUROC: how well the decision
model’s confidence separates its correct from its wrong answers. Random: expected accuracy when the same
number of queries is escalated at random.∆LLM: difference from the LLM alone in points (exact McNemarp).
Cost: relative to the LLM alone. Thresholds are label-free quantiles, fixed before computing.§Its API accepts only
a single text field, so it receives the same content as one text (the LLMs’ numbered results).
would recover detections.
Version stability.One provider serves two
pinned versions. Jev 1.14 differs from 1.13 by
−2.2 points with the flat question (p= 0.12) and
−0.7 with tree v2 (p= 0.66), with the same fail-
ure of tree v1 (Table 4): the conclusions do not
depend on which version is served.
6.5 End to end: do cheaper types keep the
gain?
With Gemini 2.5 Flash as the generator, Cattan
et al. (2025) report that the share of answers fol-
lowing the expected behaviour rises from 65.7%
without a type (Vanilla) to 75.1% with types pre-
dicted by Gemini 2.5 Flash (Pipeline) and 88.0%
with gold types (Oracle). Can a decision model be
the Pipeline’s predictor without losing this gain?
Setup.The generator is Gemini 2.5 Flash (think-
ing off, temperature 0) on the same numbered re-
sults. Vanilla uses their answer prompt verbatim;
a type adds its definition and expected behaviour
from their paper (Appendix A). We generate one
answer per test query for Vanilla and for each
type (2,436 answers); a predictor receives the an-
swer for the type it predicts, so predictors differ
only through their predictions. As in their study,
an LLM judge (Zheng et al., 2023) (Gemini 2.5
Flash) decides whether the answer follows the
gold type’s expected behaviour, given the query,
the type’s description, an adherent and a non-
adherent example, but not the sources; a secondprompt checks whether it contains the gold an-
swer (202 queries). On 100 answers labelled blind
by two annotators, who agree on 92% (Cohen’s
κ= 0.67; Cohen, 1960), the judge agrees with
their majority on 92% (Cattan et al.: 89%;κ=
0.46, as 90% of answers follow; Appendix B).
Our protocol also specified a second judge, Gem-
ini 3.8 Flash (thinking low), which agrees less of-
ten (87%); we report both.
Results.With the primary judge the ordering of
Cattan et al. (2025) reappears (Table 8): Vanilla
87.9%, Pipeline 91.6% (+3.7 points,p= 0.032),
Oracle 95.3% (+7.4,p <0.001); our judge rates
answers higher than theirs, which compresses the
differences. Replacing Gemini 2.5 Flash as the
predictor changes little: 10 of the 11 decision
models are tied with it (−3.9 to +0.7 points),
and only the least accurate classifier, ScaleDown
Classify-1, is significantly worse (−3.9,p=
0.011). With a quarter of queries escalated to
Gemini 3.8 Flash, 11 of the 11 cascades are tied
with it (−1.0 to +0.7). Gains concentrate onCon-
flicting opinions(Vanilla 82.4%, Oracle 97.1%)
andOutdated(72.7% to 87.3%); no condition
lowers answer recall (90.6–93.1%). With the sec-
ond judge, gold types still help (+5.7,p <0.001)
but predicted types do not (Pipeline 89.7%, Vanilla
91.1%,p= 0.47), and 11 of the 11 decision mod-
els remain tied with the Gemini 2.5 Flash Pipeline.
The size of the gain from predicted types thus de-
pends on the judge; that a decision model can re-

Rerun Dates Source order
System Flip No dates Recency in code Outd. recall Flip [95% CI] Shuffle 1 Shuffle 2
Liquid d1 Ultra 5.2 +1.0 (0.45) +0.0 (1.00) 76.4 / 78.2 7.4 [5.7, 9.2] +1.7 (0.25) +2.2 (0.12)
Perplexity Decider v1.2 0.0 +0.5 (0.69) +0.7 (0.51) 78.2 / 80.0 7.0 [5.3, 8.9] +0.0 (1.00) +0.0 (1.00)
OpenAI GPT-6 Luna Decisions 0.0−0.5 (0.85) +0.0 (1.00) 72.7 / 74.5 17.9 [15.3, 20.6]−4.7 (0.011)−5.7 (0.006)
Baseten Mammoth-1 0.0−0.2 (1.00)−0.2 (1.00) 72.7 / 70.9 5.4 [3.9, 7.0] +1.2 (0.27) +2.2 (0.093)
Baseten Elephant-1 0.0 +0.5 (0.80) +0.7 (0.55) 56.4 / 65.5 7.3 [5.5, 9.1]−0.7 (0.68)−1.0 (0.48)
Liquid d1 9.4−0.2 (1.00)−0.5 (0.83) 81.8 / 81.8 15.5 [13.1, 18.0] +0.2 (1.00) +2.2 (0.25)
Jev 1.13 2.0−1.0 (0.34) +0.2 (1.00) 72.7 / 74.5 11.6 [9.4, 13.8] +1.5 (0.38) +0.2 (1.00)
Perplexity Decider Flash v1.2 0.0 +0.0 (1.00) +0.5 (0.79) 58.2 / 61.8 11.7 [9.6, 13.9] +1.0 (0.60) +0.2 (1.00)
Jev 1.14 3.0 +0.0 (1.00) +1.0 (0.39) 76.4 / 76.4 10.8 [8.7, 12.9] +1.0 (0.60) +0.0 (1.00)
Microsoft Decision-1 1.5−0.7 (0.58)−0.2 (1.00) 69.1 / 67.3 17.2 [14.7, 20.0] +0.2 (1.00) +0.7 (0.78)
ScaleDown Classify-1§8.4−2.5 (0.25)−1.5 (0.52) 20.0 / 25.5 42.1 [38.8, 45.4]−3.0 (0.32) +1.0 (0.78)
Gemini 3.8 Flash (low) 3.4−0.7 (0.51)−0.7 (0.65) 69.1 / 67.3 9.4 [7.5, 11.5]−0.5 (0.85)−2.7 (0.090)
Table 6: Diagnostics on the test split, flat question. Each perturbed run is compared with the same system’s original
run on the same queries. Flip: share of queries whose predicted type changed (%); for source order, pooled over
two shuffles. No dates, Recency in code, Shuffle: accuracy change in points (exact McNemarp). Outd. recall:
recall (%) on the 55Outdatedqueries, original / without dates. The identical rerun is the noise floor.§Its API
accepts only a single text field, so it receives the same content as one text (the LLMs’ numbered results).
System Hit F.A. Youden Any AUROC
Liquid d1 Ultra 41.7 2.1 +0.396 60.4 / 21.9 0.85
Perplexity v1.2 50.0 3.1 +0.469 70.8 / 24.0 0.86
GPT-6 Luna 24.0 1.0 +0.229 53.1 / 22.9 0.81
Mammoth-1 39.6 5.2 +0.344 61.5 / 18.8 0.81
Elephant-1 11.5 0.0 +0.115 40.6 / 16.7 0.78
Liquid d1 30.2 5.2 +0.250 64.6 / 35.4 0.81
Jev 1.13 66.7 17.7 +0.490 85.4 / 41.7 0.81
Perplexity Flash v1.2 25.0 1.0 +0.240 57.3 / 25.0 0.76
Jev 1.14 64.6 21.9 +0.427 85.4 / 51.0 0.79
Microsoft D-1 46.9 16.7 +0.302 85.4 / 58.3 0.76
ScaleDown C-1§7.3 2.1 +0.052 46.9 / 27.1 0.66
Gemini 3.8 (low) 52.1 3.1 +0.490 71.9 / 25.0 –
Table 7: One planted false source in 96No conflict
test queries (expected label:Misinformation), flat ques-
tion. Hit: planted copy labelled Misinformation (%).
F.A.: clean copy of the same query labelled Misin-
formation (%). Youden: hit−false alarm. Any:
planted / clean copy not labelled No conflict (%). AU-
ROC: P(Misinformation) separating planted from clean
copies.§Its API accepts only a single text field, so it
receives the same content as one text (the LLMs’ num-
bered results).
place the LLM as the predictor does not.
7 Discussion
What decision models offer as conflict classi-
fiers.Conflict typing sits on the critical path of
every search-augmented answer, so its latency and
cost add to every request. Most hosted decision
models classify a query with about ten retrieved
pages in under half a second, for at most $0.41 per
thousand queries, with the accuracy of the LLM
used in the original study. When the accuracy of acurrent LLM is required, a decision model’s confi-
dence is usually good enough to route only its un-
certain queries to the LLM, which recovers most
of the gap at a quarter to a third of the LLM’s cost.
End to end, the cheaper types cost no measurable
answer quality (§6.5).
Choosing among decision models.The class is
not uniform. Accuracy spans 50.0–67.0%, and the
model at the bottom is significantly worse than
the paper’s LLM. Two properties that matter in
deployment vary independently of accuracy: cal-
ibration (ECE 0.031–0.188) and how well confi-
dence ranks errors, which decides whether a cas-
cade helps at all. One of the most accurate models
has confidence that is no better than random for es-
calation. A model for this task should therefore be
chosen on all three, measured on the deployment’s
own data, not on accuracy alone.
How to ask.The direct choice question was the
reliable format for every model. The yes/no de-
composition looked natural but failed for all of
them; the repaired tree encodes choices fitted to
one model on a small development split and did
not help any other. For a new decision model we
would start from the flat question.
What the diagnostics change.Dates turned out
not to matter: outdated sources are recognised
from their content. Sensitivity to source order is
real but differs so much between models that it has
to be measured for each one. A single false source
is the subtle case, visible in the probabilities but

Type predictor Judge: Gemini 2.5 Judge: Gemini 3.8 25% to Gemini 3.8†
Type given to the generator Acc. $ / 1k Adh.∆(p) Adh.∆(p) Adh.∆(p)
Vanilla (no type) – – 87.9−3.7 (0.032) 91.1 +1.5 (0.47)
Gemini 2.5 Flash, thinking off∗61.3 1.66 91.6 – 89.7 –
Gemini 3.8 Flash, thinking low 70.0 3.65 92.1 +0.5 (0.81) 90.6 +1.0 (0.52)
Liquid d1 Ultra 67.0 0.18 91.4−0.2 (1.00) 88.4−1.2 (0.42) 92.1 +0.5 (0.81)
Perplexity v1.2 66.7 0.09 91.4−0.2 (1.00) 88.2−1.5 (0.34) 92.4 +0.7 (0.65)
GPT-6 Luna 66.3 0.41 88.9−2.7 (0.052) 89.7 +0.0 (1.00) 90.6−1.0 (0.56)
Mammoth-1 65.8 0¶91.1−0.5 (0.84) 88.9−0.7 (0.70) 91.4−0.2 (1.00)
Elephant-1 65.3 0¶90.6−1.0 (0.54) 89.9 +0.2 (1.00) 91.1−0.5 (0.83)
Liquid d1 62.6 0.08 90.6−1.0 (0.54) 89.9 +0.2 (1.00) 91.9 +0.2 (1.00)
Jev 1.13 61.8 0.19 91.4−0.2 (1.00) 90.1 +0.5 (0.84) 92.4 +0.7 (0.65)
Perplexity Flash v1.2 61.8 0.04 90.1−1.5 (0.24) 89.9 +0.2 (1.00) 90.6−1.0 (0.50)
Jev 1.14 59.6 0.19 92.4 +0.7 (0.66) 90.1 +0.5 (0.84) 92.4 +0.7 (0.63)
Microsoft D-1 58.1 0.18 91.9 +0.2 (1.00) 91.1 +1.5 (0.29) 92.4 +0.7 (0.68)
ScaleDown C-1§50.0 0.24 87.7−3.9 (0.011) 87.9−1.7 (0.36) 90.6−1.0 (0.56)
Gold type (Oracle) 100 – 95.3 +3.7 (0.011) 96.8 +7.1 (<0.001)
Table 8: End to end on the test split (n= 406): Gemini 2.5 Flash answers given the type from the first column
(the Pipeline of Cattan et al. (2025)). Acc.: type accuracy (%). $ / 1k: classification cost per 1,000 queries. Adh.:
answers following the gold type’s expected behaviour (%), per LLM judge.∆: points from the original setup∗
(exact McNemarp).†Cascade of Table 5, primary judge.§Single text field (Table 2).¶Billed $0.
rarely in the top label. With 4 naturalMisinforma-
tiontest queries, that type is the one a deployment
would most need to calibrate on its own data.
Limitations.(i) CONFLICTSis small: with 406
test queries, differences under about five points
are not significant, and the cascades can be shown
close to, but not formally equivalent to, the
LLM. (ii) The label distribution reflects how the
queries were selected, not real search traffic, and
the dataset has only five naturalMisinformation
queries; our planted variant tests one kind of mis-
information (an entity or value substituted in a sin-
gle source). (iii) The census covers the decision
models one gateway served on one day. Hosted
models change without notice: several identifiers
are unversioned, two models were free during our
runs, and our rerun of Gemini 2.5 Flash scores be-
low the published figure. Providers do not disclose
model sizes or training data, so we cannot explain
the differences between models. (iv) The end-to-
end test uses one generator and LLM judges, val-
idated on 100 answers, that do not see the search
results; as in the original study, the primary judge
is the generator’s own model, which may favour
its answers (Panickssery et al., 2024). The size
of the gain from predicted types depends on the
judge, and we do not measure factual grounding,
the third metric of Cattan et al. (2025). (v) Not
every system is deterministic: identical requestschanged up to 9.4% of a decision model’s predic-
tions, and 3.4% of Gemini 3.8 Flash’s.
8 Conclusion
We tested every typed decision model hosted on
one gateway as a classifier of knowledge-conflict
types, using the benchmark authors’ protocol and
the same inputs as the LLMs. Most match the
LLM of the original study. They do so at a small
fraction of its cost and latency. They trail the
current LLM, but for most of them a confidence-
gated cascade recovers that gap at a quarter to a
third of the LLM’s cost, and given to a generator,
their types yield answers that follow the expected
behaviour as often as the LLM’s. Future work
should measure the gain from predicted types with
judges that see the sources, and test whether a de-
cision model’s probabilities can steer a generator
between expected behaviours when the type is un-
certain.
Ethics and Broader Impact
The planted-misinformation variant rewrites one
retrieved page per query to state a wrong answer.
It uses public data and factual substitutions on
non-sensitive questions, and is released with clear
labels. All queries in CONFLICTSare safe by con-
struction (Cattan et al., 2025). Cheap conflict clas-
sification could be misused to suppress minority

viewpoints; the expected behaviour forConflict-
ing opinionsis to present all of them.
References
Baseten. 2026. The top ranked decision models
in the world: mammoth-1 and elephant-1.
https://www.baseten.co/blog/the-
best-decision-models-in-the-world-
mammoth-1-and-elephant-1/.
Arie Cattan, Alon Jacovi, Ori Ram, Jonathan
Herzig, Roee Aharoni, Sasha Goldshtein, Eran
Ofek, Idan Szpektor, and Avi Caciularu. 2025.
DRAGged into conflicts: Detecting and
addressing conflicting sources in
search-augmented LLMs.arXiv preprint
arXiv:2506.08500.
Lingjiao Chen, Matei Zaharia, and James Zou.
2023. FrugalGPT: How to use large language
models while reducing cost and improving
performance.arXiv preprint
arXiv:2305.05176.
Jacob Cohen. 1960. A coefficient of agreement
for nominal scales.Educational and
Psychological Measurement, 20(1):37–46.
Gheorghe Comanici et al. 2025. Gemini 2.5:
Pushing the frontier with advanced reasoning,
multimodality, long context, and next
generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Tobias Deußer, Lorenz Sparrenberg, and Rafet
Sifa. 2026. Evaluating and benchmarking the
System One model Jev.arXiv preprint
arXiv:2609.37647.
Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang,
Jialiang Wu, Xinnong Zhang, Hanqi Yan,
Siyuan Wang, and Zhongyu Wei. 2026.
General decision models: Benchmarking and
insights beyond Jev.arXiv preprint
arXiv:2610.03935.
Bradley Efron and Robert J. Tibshirani. 1994.An
Introduction to the Bootstrap. Chapman &
Hall/CRC.
Yonatan Geifman and Ran El-Yaniv. 2017.
Selective classification for deep neural
networks. InAdvances in Neural Information
Processing Systems.Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q.
Weinberger. 2017. On calibration of modern
neural networks. InProceedings of the 34th
International Conference on Machine
Learning, pages 1321–1330.
Yufang Hou, Alessandra Pascale, Javier
Carnerero-Cano, Tigran Tchrakian, Radu
Marinescu, Elizabeth Daly, Inkit Padhi, and
Prasanna Sattigeri. 2024. WikiContradict: A
benchmark for evaluating LLMs on real-world
knowledge conflicts from Wikipedia. In
Advances in Neural Information Processing
Systems, Datasets and Benchmarks Track.
Jiaju Huang, Hao Yang, Xinyu Ma, Xinglong
Liang, Kunyan Cai, Junqiang Ma, Shaobin
Chen, Yue Sun, and Tao Tan. 2026. Can Jev
judge radiology reports? Evaluating a System
One model for clinical factuality.arXiv
preprint arXiv:2609.27607.
Keyi Li, Yihao He, and Quanyi Li. 2026a.
Beyond calibration: Do a typed-decision
model’s probabilities obey the probability
axioms?arXiv preprint arXiv:2609.33209.
Xing Li, Qingcheng Chang, Jinzhong Ning,
Changfeng Xu, Shenlong Zhang, Yijia Zhang,
Ling Luo, and Hongfei Lin. 2026b.
Specialized decision models vs.
general-purpose LLMs: Benchmarking Jev
across knowledge, reasoning, and multilingual
tasks.arXiv preprint arXiv:2610.11978.
Yinheng Li and Justin Wagle. 2026. LLM-as-Jev:
LLMs are already Jev-style decision models –
when and how to fine-tune them.arXiv
preprint arXiv:2610.02076.
Yubo Li, Yidi Miao, Ramayya Krishnan, and
Rema Padman. 2026c. JEV-as-a-judge: Accept
when confident, escalate when unsure.arXiv
preprint arXiv:2609.26550.
Liquid AI. 2026a. d1-ultra: Introducing the
world’s best decision model.
https://www.liquid.ai/blog/d1-ultra.
Liquid AI. 2026b. Introducing d1: The most
capable decision model, now with vision.
https://www.liquid.ai/blog/d1-
decision-model.

Siyi Liu, Qiang Ning, Kishaloy Halder, Wei Xiao,
Zheng Qi, Phu Mon Htut, Yi Zhang,
Neha Anna John, Bonan Min, Yassine
Benajiba, and Dan Roth. 2024. Open domain
question answering with conflicting contexts.
arXiv preprint arXiv:2410.12311.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer
Singh. 2021. Entity-based knowledge conflicts
in question answering. InProceedings of the
2021 Conference on Empirical Methods in
Natural Language Processing, pages
7052–7063.
Quinn McNemar. 1947. Note on the sampling
error of the difference between correlated
proportions or percentages.Psychometrika,
12(2):153–157.
Microsoft. 2026. Microsoft-Decision-1.
Microsoft Foundry Labs,
https://labs.ai.azure.com/
innovations/microsoft-decision-1/.
Shubham Mishra, Samyek Jain, Gorang Mehrishi,
Shiv Tiwari, Harsh Sharma, Pratik Narang, and
Dhruv Kumar. 2025. From facts to
conclusions: Integrating deductive reasoning in
retrieval-augmented LLMs.arXiv preprint
arXiv:2512.16795.
Gowthamkumar Nandakishore. 2026. Laya as a
typed probabilistic assessor: An independent
reproduction and a preregistered study of
calibration and selective escalation.arXiv
preprint arXiv:2609.33843.
OpenAI. 2026. Decisions API guide.
https://developers.openai.com/api/
docs/guides/decisions.
Arjun Panickssery, Samuel R. Bowman, and Shi
Feng. 2024. LLM evaluators recognize and
favor their own generations. InAdvances in
Neural Information Processing Systems.
Perplexity AI. 2026a. pplx-decider-flash-v1.2
model card.
https://huggingface.co/perplexity-
ai/pplx-decider-flash-v1.2.
Perplexity AI. 2026b. pplx-decider-v1.2 model
card.https://huggingface.co/
perplexity-ai/pplx-decider-v1.2.Delip Rao and Chris Callison-Burch. 2026.
Decision models vs. LLMs as rubric judges:
Cheaper, faster, and wrong in the same places.
arXiv preprint arXiv:2609.29769.
ScaleDown. 2026. Classify overview. ScaleDown
API reference,https://docs.scaledown.
ai/api-reference/classify-overview.
TypeSafe AI. 2026a. Introducing System One
models and Jev.https:
//typesafe.ai/blog/introducing-
system-one-models-and-jev.
TypeSafe AI. 2026b. Jev 1.13 model jaggedness.
https://docs.typesafe.ai/model-
jaggedness/jev-1.13.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah
Constant, Jerry Wei, Jason Wei, Chris Tar,
Yun-Hsuan Sung, Denny Zhou, Quoc Le, and
Thang Luong. 2024. FreshLLMs: Refreshing
large language models with search engine
augmentation. InFindings of the Association
for Computational Linguistics: ACL 2024,
pages 13697–13720.
Alexander Wan, Eric Wallace, and Dan Klein.
2024. What evidence do language models find
convincing? InProceedings of the 62nd
Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers), pages 7468–7484.
Chenyu Wang, Yueyuan Li, Yingmin Liu, and
Yang Shu. 2026. ConflictRAG: Detecting and
resolving knowledge conflicts in retrieval
augmented generation.arXiv preprint
arXiv:2605.17301.
Han Wang, Archiki Prasad, Elias Stengel-Eskin,
and Mohit Bansal. 2025. Retrieval-augmented
generation with conflicting evidence. In
Conference on Language Modeling (COLM).
Yuchuan Weng, Chi Liu, Xin Chen, and
Srinivasan Manoharan. 2026. Our own Jev:
Typed decisions in one forward pass, from an
in-house judge to open LoRA models.arXiv
preprint arXiv:2610.29052.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang
Wang, Hongru Wang, Yue Zhang, and Wei Xu.
2024. Knowledge conflicts for LLMs: A
survey. InProceedings of the 2024 Conference

on Empirical Methods in Natural Language
Processing, pages 8541–8565.
Zixiang Xu, Zirui Song, Chiyu Zhang, Xiuying
Chen, Xi Liu, Xiyang Hu, and Yue Zhao. 2026.
JevOut: Natural context can flip decision
models.arXiv preprint arXiv:2609.30243.
Michael J.Q. Zhang and Eunsol Choi. 2021.
SituatedQA: Incorporating extra-linguistic
contexts into QA. InProceedings of the 2021
Conference on Empirical Methods in Natural
Language Processing, pages 7371–7387.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao
Zhuang, Zi Lin, Zhuohan Li, Dacheng Li,
Eric P. Xing, Hao Zhang, Joseph E. Gonzalez,
and Ion Stoica. 2023. Judging LLM-as-a-judge
with MT-Bench and Chatbot Arena. In
Advances in Neural Information Processing
Systems: Datasets and Benchmarks Track.
A Prompts and Questions
LLM prompt.The conflict-type prompt of Cat-
tan et al. (2025), copied from the paper’s source
with only the L ATEX bold markers removed. The
asterisks are the prompt’s own Markdown empha-
sis and were sent as written.{search_results}
is filled with the numbered results.
You are tasked with analyzing the search results
provided for a given query and classifying
the type of knowledge conflict present (if
any).
Consider the query and the search results
carefully, then classify the conflict into
*one* of the following categories, using the
descriptions and examples provided below.
1. No Conflict: The search results refer to the
same concept and are in agreement.
Differences are superficial, such as
variations in detail or granularity.
* *Example:*
* *Query:* What is the meaning of the name
Apoorva?
* *Search Results:* Unique, Quite new, Not
seen before
2. Complementary Information: The question is
underspecified or allows for multiple valid
perspectives or answers that do not
contradict each other. All provided answers
can be considered correct.
* *Example:*
* *Query:* Is public transportation faster
than driving in cities?
* *Search Results:* Depends on the city
and situation (e.g., rush hour vs.
off-peak), specific routes matter,
availability of parking is relevant.3. Conflicting Opinions or Research Outcomes: The
query addresses a subjective or contentious
topic, leading to genuinely opposing
viewpoints or contradictory research
findings.
* *Example:*
* *Query:* Is online learning as effective
as traditional classroom learning?
* *Search Results:* Some sources argue
"yes," citing flexibility and
accessibility. Others argue "no,"
emphasizing the importance of
in-person interaction.
4. Conflict Due to Outdated Information: The
question has a verifiable factual answer,
but the search results contain conflicting
information due to changes over time.
* *Example:*
* *Query:* Do Tesla and X Corp. have the
same CEO?
* *Search Results:* Some articles say yes
(dated before the change), others say
no (dated after the change).
5. Conflict Due to Misinformation: The question
has a verifiable factual answer, but some of
the search results contain factually
incorrect or misleading information.
* *Example:*
* *Query:* What is the capital of Israel?
* *Search Results:* One source correctly
states Jerusalem, while another
incorrectly states Tel Aviv.
Input:
* QUERY: {query}
* SEARCH RESULTS: {search_results}
Output:
Please provide your response as a JSON object
with the following fields:
* explanation: (String) A brief explanation of
why you chose this category.
* category: (Integer) The number (1-5)
corresponding to the category of the
knowledge conflict.
Decision models, flat.One choice question. Its
instruction is the first paragraph of the prompt
above; its options are the five category definitions
with their examples, word for word.
type: choice
instructions: You are tasked with analyzing the
search results provided for a given query
and classifying the type of knowledge
conflict present (if any).
Consider the query and the search results
carefully, then classify the conflict into
*one* of the following categories, using the
descriptions and examples provided below.
options: the five category definitions of the
prompt above, each with its example, keyed
no_conflict, complementary,
conflicting_opinions, outdated,
misinformation
Decision models, tree.Five yes/no questions
asked in one request over the same state.

differ (yes/no): Beyond superficial differences
in wording, detail or granularity, do the
search results give different answers to the
query?
compatible (yes/no): Could all the different
answers in the search results be correct at
the same time, for example because the query
is underspecified or allows several valid
answers?
opinion (yes/no): Is the query a subjective or
contentious topic on which the search
results present opposing viewpoints or
contradictory research findings?
outdated (yes/no): Does the query have a
verifiable factual answer on which the
search results disagree because the answer
changed over time, so some results are
outdated?
misinformation (yes/no): Does the query have a
verifiable factual answer on which some of
the search results are factually incorrect
or misleading?
Planted wrong answers.Prompt sent to GPT-
4.1-mini (temperature 0) to obtain a plausible
wrong answer for eachNo conflictquery whose
gold answer appears in a result. The answer is then
replaced, as whole words and case-insensitively, in
the title, snippet and text of one such result chosen
with a fixed seed. Queries whose gold answer is
yesornoare excluded.
Question: {question}
Correct answer: {answer}
Give one plausible but incorrect answer to the
question, of the same type and format as the
correct answer (a different name, place,
number or date). Reply with the incorrect
answer only.
End to end: generation.Vanilla is the answer
prompt of Cattan et al. (2025) (their Figure 3) ver-
batim. For a type, the line below is inserted before
the query; the description is the type’s definition
and expected behaviour from their §2, with cita-
tions and worked examples removed.
Write a high-quality and concise answer for the
given question using only the provided
search results.
For each sentence, cite the corresponding sources
using [1][2][3].
TYPE OF KNOWLEDGE CONFLICT IN THE SEARCH RESULTS:
{description}
QUERY: {query}
SEARCH RESULTS: {search_results}
End to end: judges.The behaviour judge,
shown forMisinformation; for the other types the
description and the two examples change. The ex-
amples are those of Cattan et al. (2025), Table 3,except forNo conflict, which has none there and
for which we wrote one pair. The answer-recall
judge follows.
You are evaluating whether a response to a search
query follows the expected behavior for the
type of knowledge conflict among the query’s
search results.
CONFLICT TYPE: Conflict due to misinformation.
The retrieved documents provide answers that
are not mutually compatible, where at least
one source contains information that is
likely false, misleading, or inaccurate. The
expected behavior for this category is to
disregard inaccurate sources and to provide
a response grounded in reliable and verified
information.
EXAMPLES:
Query: When did season 5 of prison break come out?
Response: Season 5 of Prison Break premiered on
April 4, 2017, on Fox [3]. The season was
released on May 30, 2017 [4].
Verdict: does not follow the expected behavior.
Reason: The response tries to reconcile the
sources, although source [4] is incorrect.
Query: When did season 5 of prison break come out?
Response: Season 5 of Prison Break premiered on
April 4, 2017 [2][3]. The first episode
aired on this date [2].
Verdict: follows the expected behavior. Reason:
The response provides a clear and coherent
answer to the question.
RESPONSE TO EVALUATE:
Query: {query}
Response: {response}
Does the response follow the expected behavior
for this conflict type? Judge how the
response handles the sources, as the
expected behavior describes; do not judge
its length or writing quality.
Reply with a JSON object: {"explanation": "<one
or two sentences>", "adheres": true or false}
You are checking whether a response contains the
correct answer to a question.
QUESTION: {query}
CORRECT ANSWER: {gold answer}
RESPONSE: {response}
Does the response state the correct answer? Count
paraphrases, equivalent formats and answers
with more detail as correct; the response
may also mention other information.
Reply with a JSON object: {"explanation": "<one
sentence>", "includes_answer": true or false}
B Protocol Record
The development/test split, the prompt and ques-
tion texts and the combination rules were frozen
with recorded file hashes before any test-set call.
Changes after that point were limited to the fol-
lowing, each logged with its reason before results

were computed: (1) Gemini 2.5 Flash with think-
ing added as a sensitivity check; (2) the answer
parser made lenient for category statements out-
side the requested JSON, applied to stored an-
swers without new calls (no strictly formatted an-
swer changed); (3) Gemini 3.8 Flash added at
two thinking levels; (4) the cascade, its operating
points and its controls; (5) the diagnostics, with
one amendment before any planted run: whole-
word matching and the exclusion of yes/no an-
swers, after substring matching was found to alter
unrelated numbers; (6) a second Jev version and
four further decision models, run with the frozen
questions; (7) the remaining five decision models
listed by the gateway, completing the census, with
the diagnostics extended to every decision model’s
flat question, and one amendment before any run:
Classify-1 receives the same content as a single
text field, which its API requires; (8) the end-
to-end test, specified before any generation call:
generator, prompts, judges, the 100-answer valida-
tion sample and its threshold, and a second judge
(Gemini 3.8 Flash) to be reported with the first.
Prompts were checked on the development split
for errors only; none was changed after scoring.
All runs are reported.
Judge validation.The 100 answers were drawn
with a fixed seed, 20 per gold type and about half
from Vanilla, and shown to the annotators with the
query, the gold type’s description and expected be-
haviour, and the answer, without the judge’s ver-
dict or the condition. Annotator 2’s sheet showed
another item’s answer for one item; that label is
excluded. The primary judge agrees with the an-
notators’ majority (a tie counting asfollows, as
fixed before labelling) on 92%, with annotator 1
on 91% and with annotator 2 on 87%; the sec-
ond judge on 87%, 84% and 82%. Code, splits,
prompts, cached responses and per-query outputs
are released.