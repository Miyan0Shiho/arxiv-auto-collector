# Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents

**Authors**: Lasse B. Strand, Robert Jakob, Kevin O'Sullivan, Markus Kreft

**Published**: 2026-10-06 14:38:15

**PDF URL**: [https://arxiv.org/pdf/2610.08452v1](https://arxiv.org/pdf/2610.08452v1)

## Abstract
Retrieval-augmented generation (RAG) is a widely used approach for grounding large language models (LLMs) in external knowledge. However, configuring a pipeline is an expensive hyperparameter optimization problem over many interacting choices, from chunking and embedding model to reranking and generation. Existing optimizers, from greedy search to Bayesian optimization, reduce each trial to an aggregate score and search without modeling why a configuration performed as it did, even though the retrieved chunks already provide evidence about whether each failure occurred during retrieval or after it. We introduce Agentic AutoRAG, an LLM-agent optimizer for multi-objective RAG hyperparameter optimization with retrieval-versus-generation failure attribution. It proposes configurations scored on a frozen exam from the corpus: after each trial a Diagnoser attributes each failed question to retrieval or generation, and a Proposer, grounded in a knowledge base of model rankings and pricing, selects the next configuration, weighing accuracy against cost to trace a Pareto frontier. On three multi-hop QA benchmarks it reaches higher LLM-judge accuracy than every baseline we compare, and within its first 10 trials it matches or beats the statistical baselines' full 30-trial judge accuracy. In its cost-aware mode on a real-world healthcare corpus it reaches a median exam accuracy of 77%, above the strongest baseline's 71.5%, at about 58% of that baseline's cost per query, and it matches that 71.5% at about 22% of the cost.

## Full Text


<!-- PDF content starts -->

Agentic AutoRAG: RAG Pipeline Optimization
through Reasoning-Driven Agents
Lasse B. Strand*, Robert Jakob, Kevin O’Sullivan, Markus Kreft
ETH Zurich
Abstract
Retrieval-augmented generation (RAG) is a
widely used approach for grounding large lan-
guage models (LLMs) in external knowledge.
However, configuring a pipeline is an expen-
sive hyperparameter optimization problem over
many interacting choices, from chunking and
embedding model to reranking and generation.
Existing optimizers, from greedy search to
Bayesian optimization, reduce each trial to an
aggregate score and search without modeling
why a configuration performed as it did, even
though the retrieved chunks already provide ev-
idence about whether each failure occurred dur-
ing retrieval or after it. We introduce Agentic
AutoRAG, an LLM-agent optimizer for multi-
objective RAG hyperparameter optimization
with retrieval-versus-generation failure attribu-
tion. It proposes configurations scored on a
frozen exam from the corpus: after each trial a
Diagnoser attributes each failed question to re-
trieval or generation, and a Proposer, grounded
in a knowledge base of model rankings and pric-
ing, selects the next configuration, weighing
accuracy against cost to trace a Pareto frontier.
On three multi-hop QA benchmarks it reaches
higher LLM-judge accuracy than every base-
line we compare, and within its first 10 trials it
matches or beats the statistical baselines’ full
30-trial judge accuracy. In its cost-aware mode
on a real-world healthcare corpus it reaches
a median exam accuracy of 77%, above the
strongest baseline’s 71.5%, at about 58% of
that baseline’s cost per query, and it matches
that 71.5% at about 22% of the cost.1
1 Introduction
Retrieval-augmented generation (RAG) has be-
come a standard way to ground large language
*Corresponding author:lstrand.research@gmail.com
Accepted at the Second Workshop for REsearch on Agent
Language Models (REALM) at EMNLP 2026 and at the Ma-
chine Learning for Systems Workshop at NeurIPS 2026.
1Code: https://github.com/Agentic-Systems-Lab/
Agentic-AutoRAGmodels (LLMs) in knowledge beyond their training
data (Lewis et al., 2020; Gao et al., 2023). Con-
figuring a RAG pipeline means choosing across
many interacting dimensions: chunking, embed-
ding model, index type, retrieval depth, reranker,
query expansion, passage compression, and gen-
erator. These choices are coupled and should not
be tuned in isolation (Wang et al., 2024). The best
retrieval depth, for instance, depends on chunk size:
small chunks each hold less text, so they need a
deeper retrieval to surface the same evidence that a
few large chunks already contain. The best configu-
ration also shifts from corpus to corpus: a pipeline
tuned for one can underperform on another, so the
search must be repeated for each new one (Kartal
et al., 2025; Orbach et al., 2026). Scoring a config-
uration means running the whole pipeline over a
test set, so the search must be sample-efficient.
Recent work applies general-purpose search al-
gorithms to this problem: AutoRAG’s greedy node-
wise selection (Kim et al., 2024), multi-armed ban-
dits (Fu et al., 2024), Bayesian optimization (Con-
way et al., 2025; Barker et al., 2025), and genetic
search (Kartal et al., 2025). All of these optimizers
reduce each trial to an aggregate score and search
the space without modeling why a configuration
performed as it did. When a configuration scores
low, they cannot tell whether retrieval and rerank-
ing failed to surface the relevant passages, or the
generator answered incorrectly despite having the
right context. However, any RAG pipeline already
records which chunks it retrieved. By checking
whether the passages that contain the answer (the
gold spans) appear among them, we can attribute
each failure to retrieval or generation. We are not
aware of prior RAG optimizers that use this per-
question failure evidence to drive their configura-
tion proposals.
We introduce Agentic AutoRAG, a reasoning-
driven optimizer for RAG pipelines. Like recent
LLM-agent optimizers for general hyperparameter
1
arXiv:2610.08452v1  [cs.CL]  6 Oct 2026

tuning (Liu et al., 2025; Trirat et al., 2025), it uses
a language model to propose configurations, but
specializes this approach to RAG and cost-aware
search. It contributes three ideas:
1.Adiagnose-then-propose loop.After each
trial, a Diagnoser reads the per-question fail-
ure evidence, attributes each failure to re-
trieval or generation, and compresses it into
a short diagnosis. A Proposer then reads that
diagnosis and selects the next configuration.
2.Aknowledge-base-grounded, cost-aware
agent.The Proposer prices and ranks can-
didate configurations against public model
leaderboards and per-token pricing before
spending a trial on them, so it navigates the
accuracy-cost trade-off without first learning
which models are strong or cheap from trial
outcomes.
3.Acorpus-derived exam.Adapting es-
tablished corpus-grounded exam generation
(Guinet et al., 2024; Shashidhar et al., 2025),
we draft open-ended, multi-hop questions
with verbatim gold-answer spans, filter them
through a sufficiency oracle, and keep only
questions that separate configurations span-
ning the search space. The exam is frozen
once and reused as the optimization target
across every trial.
We compare the optimizer against random
search and a multi-objective Bayesian optimizer
(MO-TPE). On three multi-hop QA benchmarks
it reaches higher LLM-judge accuracy than every
baseline we compare, and it is sample-efficient:
within its first 10 trials it already matches or beats
the statistical baselines’ full 30-trial judge accuracy.
In its cost-aware mode on a healthcare corpus it
reaches a median exam accuracy of 77%, above the
strongest baseline’s 71.5%, at about 58% of that
baseline’s cost per query, and it matches that 71.5%
at about 22% of the cost.
2 Related Work
2.1 RAG pipeline optimization
A RAG pipeline exposes many discrete choices,
making its configuration a hyperparameter opti-
mization (HPO) problem with an expensive black-
box objective.
AutoRAG (Kim et al., 2024) decomposes the
pipeline into modular nodes and greedily selectsthe best module at each node in pipeline order.
AutoRAG-HP (Fu et al., 2024) casts RAG tuning
as a multi-armed bandit problem and tunes three
hyperparameters with a hierarchical UCB1 variant.
syftr (Conway et al., 2025) applies multi-objective
Bayesian optimization (MO-TPE) to a large space
of RAG pipelines, from standard to agentic, with a
Pareto-Pruner that halts candidates unlikely to ex-
tend the frontier and a transfer step that seeds a new
search with configurations found on other datasets.
Barker et al. (2025) add more objectives, tuning
cost, latency, safety, and alignment jointly with
multi-objective Bayesian optimization. RAGSmith
(Kartal et al., 2025) searches RAG compositions
with a genetic algorithm. Orbach et al. (2026) com-
pare TPE, random search, and greedy variants and
find that about 10 trials reach the accuracy of an
exhaustive grid, though over a space of only 162
configurations, far smaller than the one we search.
All of these methods reduce a trial to its score
and do not model why a configuration failed, which
is the per-question evidence our optimizer diag-
noses after each trial to guide its next proposal.
2.2 LLM agents as optimizers
A parallel line of work uses an LLM as the opti-
mizer itself, though not yet applied to RAG pipeline
configuration. OPRO (Yang et al., 2024) gives the
model a meta-prompt that lists the solutions tried
so far with their scores and asks it to propose a
better one, shown on tasks such as linear regression
and prompt optimization.
In HPO specifically, Liu et al. (2025) introduce
AgentHPO, a two-agent framework where a Cre-
ator proposes hyperparameters and an Executor
runs experiments across 12 ML tasks. Trirat et al.
(2025) apply LLM agents to a broader range of
AutoML tasks.
These optimizers are general-purpose and single-
objective, and they carry no prior about which mod-
els are strong or cheap. We specialize the approach
to RAG pipeline configuration, and we ground the
agent in a knowledge base of public model rank-
ings and per-token pricing so it can weigh accuracy
against cost and rule out configurations before run-
ning them.
A separate line of work builds agentic RAG sys-
tems, where an agent decides what to retrieve while
answering each question (Asai et al., 2024; Yu et al.,
2024; Wu et al., 2025; Leng et al., 2025; Singh
et al., 2025). Our agent instead tunes the pipeline
configuration rather than answering questions it-
2

Corpus
Search
space
Knowledge Base
(model specs &
benchmark scores)Build
neighborhoodsComposer
AgentValidate
(filters +
oracle)Probe-based
discriminationFrozen
exam
Proposer
Agent
suggests
initial configEvaluate
RAG config
on examDiagnoser
AgentProposer
AgentRecommended
config +
Pareto frontier
+ Reasoning
traceUser ProvidedExam Generation
Optimization Loop
Until max trials reached
Figure 1: Architecture of Agentic AutoRAG. The exam generation pipeline builds a frozen exam from the corpus
once before optimization: TF-IDF neighborhoods seed the Composer, which drafts open-ended questions with
verbatim gold spans, validation filters and a sufficiency oracle enforce answerability and multi-hop dependence, and
probe-based discrimination admits only questions that differentiate across the search space. The optimization loop
alternates between a Diagnoser, which compresses each trial’s per-question failure evidence, and a Proposer, which
selects the next configuration grounded in a knowledge base of public model rankings and pricing. The optimizer
outputs a Pareto frontier over accuracy and cost, a recommended configuration, and a per-trial reasoning trace.
self.
2.3 Automated RAG evaluation
Building an evaluation set from the target corpus is
an established way to assess retrieval-augmented
systems without hand annotation (Guinet et al.,
2024; Saad-Falcon et al., 2024; Shashidhar et al.,
2025; Filice et al., 2025). Closest to us, Guinet et al.
(2024) prompt an LLM to write corpus-grounded
multiple-choice questions and use item response
theory to discard questions that do not separate sys-
tems. Our exam adapts this recipe for configuration
search: the questions are open-ended with verbatim
gold spans, and we keep only those that separate
configurations spanning our search space.
3 Method
Agentic AutoRAG is a reasoning-driven optimizer
for RAG pipelines. Each trial runs a candidate
configuration on a frozen exam of question-answer
pairs, and a two-stage reasoning loop interprets the
result. A Diagnoser interprets per-question failure
evidence from the trial that just ran, and a Proposer
selects the next configuration based on the diagno-
sis. The optimizer searches over answer accuracy
and cost per query and produces a Pareto frontier
of non-dominated configurations. Figure 1 givesan overview.
3.1 Problem Formulation
A RAG pipeline Pcmaps a question qto an an-
sweraunder a configuration cdrawn from a user-
defined search space S. The space is a product
of discrete and continuous dimensions covering
chunking, embedding, index type, retrieval, rerank-
ing, query expansion, passage compression, and
generation.
We optimize two objectives: answer accuracy
f(c)∈[0,1] , the fraction of exam questions an-
swered correctly, and mean API cost per query
g(c), in USD. A configuration c∗Pareto-dominates
cwhen f(c∗)≥f(c) andg(c∗)≤g(c) with at
least one strict inequality. The optimizer runs in
one of two modes. In its default cost-aware mode
it optimizes both objectives and returns the non-
dominated set of configurations found across trials.
In score-only mode it optimizes accuracy alone and
hides the cost and Pareto signals from the agents.
3.2 Exam Generation
Our framework builds an open-ended exam of
question-answer pairs from the target corpus once
before optimization and reuses it across every trial,
so scores stay comparable from trial to trial. Exam
3

generation is an optional component: a user can
supply a question-answer set directly, in which case
the system skips generation and scores each trial on
the supplied exam. When it does generate the exam,
an LLM agent we call the Composer drafts the
questions in three stages: it reads neighborhoods of
related chunks and drafts questions from them, val-
idation filters the drafts, and probe-based selection
keeps the questions that separate one configuration
from another. The Composer over-generates by a
configurable factor, so the candidate pool is larger
than the target exam, and validation and selection
then reduce it to the requested number of questions.
Neighborhoods.For each generation call the
Composer reads a neighborhood of related chunks
and drafts as many questions as it judges the
material supports, tagging each with a question
type from a fixed taxonomy of seven types (Ap-
pendix A.1, with an example of each in A.2). A
neighborhood pairs an anchor chunk with its same-
document siblings and a set of cross-document
chunks. The cross-document chunks are ranked
by TF-IDF cosine similarity to the anchor, a lexical
score that favors chunks sharing rare, distinctive
terms with the anchor rather than common words.
Validation.Each candidate passes a set of de-
terministic filters: a structural check that rejects
a question citing one span twice rather than two
distinct ones, a self-containment filter that rejects
questions referring to their source by name (“the
document”, “the paper”, “Chunk 1”), a non-empty-
span check, and an arithmetic verification for nu-
meric questions. Multi-hop questions then pass
a sufficiency oracle. When shown all gold spans
at once, the oracle must return the smallest sub-
set of spans that supports the answer, a verbatim
quote from each, and the answer itself. A ques-
tion is rejected when that subset is smaller than
its hop count, since it would then decompose into
fewer hops. Testing the spans one at a time would
misjudge sufficiency, because a single span shown
alongside the question often lets the oracle infer
the answer from the question itself and rate that
span sufficient on its own.
Probe-based selection.A question is useful to
the optimizer only when some of the configurations
it will evaluate solve the question and others do not.
We measure this with four probe configurations that
span the search-space extremes along several axes
(LLM, embedding model, reranker, chunk size, andretrieved top- k), and we evaluate every candidate
on all four. We discard the questions that every
probe answers, since they no longer separate one
configuration from another, favor the questions that
only one or a few probes solve, and keep a small
share that no probe solves, since a configuration
beyond the probe envelope may still answer them.
This weighting governs which questions enter the
exam, not how answers are later scored.
3.3 Optimization Loop
On each trial the system builds or loads a cached in-
dex for the proposed configuration, runs the RAG
pipeline on every exam question, scores the an-
swers, and records which gold spans appear in the
retrieved chunks. Recording the retrieved spans
is what lets the loop attribute a failure to retrieval,
when the gold spans were missed, or to generation,
when they were retrieved but the answer was still
wrong. The trial’s outcome passes to two reasoning
agents, a Diagnoser and a Proposer.
Diagnoser.The Diagnoser compresses the trial’s
per-question evidence into a compact diagnosis,
which keeps the Proposer’s context window fo-
cused on what matters. It reads a pre-computed
summary of the trial that just ran: overall accuracy
and accuracy on the questions where retrieval suc-
ceeded, how the failures divide between retrieval
and generation, and a sample of failed questions
chosen to span different failure types and question
types, including questions this configuration newly
got wrong, rather than only the lowest-scoring ones.
It also sees a compact record of how accuracy
moved across earlier trials, but not their config-
urations, their cost, the Pareto frontier, or its own
earlier diagnoses. Withholding these keeps each
diagnosis grounded in the current trial and leaves
synthesis across trials to the Proposer. The Diag-
noser emits a short narrative, findings tied to the
trial’s numbers, notable changes from earlier trials,
and a handful of illustrative questions for the Pro-
poser. It does not prescribe configuration changes.
Proposer.The Proposer chooses the next config-
uration. It reads the diagnosis, the current configu-
ration, the search space with its legal ranges, and
a knowledge base of model rankings and pricing.
A short plan, carried forward from one trial to the
next, serves as the loop’s memory. Every config-
uration evaluated so far stays in view, so it never
repeats one. It also knows which generators, em-
bedding models, and rerankers remain untried, and
4

how much of the trial budget is left. In cost-aware
mode it reads the current accuracy-cost frontier
directly, with each non-dominated member’s accu-
racy, cost, and configuration, and decides where
along the cost axis to probe next. It emits the next
configuration together with a rationale.
The knowledge base lets the Proposer rule out
configurations that public benchmarks predict will
underperform before a trial is spent on them. This
is what makes the loop tractable in a large search
space, where naive enumeration is infeasible and
Bayesian baselines spend trials rediscovering the
same priors. More importantly, it is what enables
cost-aware search, as its per-token pricing lets the
Proposer weigh accuracy against cost before each
trial and move straight to cheap-but-capable gen-
erators. Appendix C details its sources and the
per-model annotations the Proposer reasons over.
The optimizer runs for a user-specified number
of trials. In cost-aware mode it returns the Pareto
frontier together with an agent-recommended con-
figuration from it, and in score-only mode the sin-
gle best-scoring configuration. Either way it also
returns the per-trial reasoning trace.
4 Experimental Setup
We evaluate Agentic AutoRAG in two experi-
ments. The first, the Accuracy experiment, maxi-
mizes answer accuracy on three multi-hop question-
answering benchmarks and compares our optimizer
against random search, MO-TPE with and without
a warm start, and a knowledge-and-diagnosis abla-
tion of our own agent. To compare the optimizer in-
dependently of the exam-generation component of
our framework, this experiment optimizes against
a fixed exam built from each benchmark’s own
labeled questions rather than the corpus-derived
exam of Section 3.2. The second, the Pareto exper-
iment, turns on the optimizer’s cost-aware mode
and runs the full pipeline end to end, generating an
exam from a corpus of real-world healthcare doc-
uments and then tracing the accuracy-cost Pareto
frontier it finds. Every method runs under a fixed
budget of 30 trials, and we repeat each search ten
times with independent seeds.
Datasets.The Accuracy experiment uses three
multi-hop question-answering benchmarks. Hot-
potQA (Yang et al., 2018) in the distractor set-
ting asks questions that bridge or compare two
Wikipedia paragraphs. MuSiQue-Answerable
(Trivedi et al., 2022) composes two to four single-Corpus Docs. Words Words/doc.
HotpotQA 19,276 1.81M 94
MuSiQue 17,629 1.32M 75
MultiHop-RAG 609 1.07M 1,760
UniDoc 250 1.30M 5,214
Table 1: Corpus size for each benchmark.
hop questions into one, so many items need a
longer reasoning chain. MultiHop-RAG (Tang and
Yang, 2024) draws comparison, inference, and tem-
poral questions from a news corpus and includes
unanswerable questions whose correct response is
to abstain. The Pareto experiment uses the health-
care subset of UniDoc-Bench (Peng et al., 2025),
230 PDFs and 20 page images that are mostly
biomedical research articles and medical studies.
These documents are multi-page, with tables, fig-
ures, and mixed layouts, and we parse them with
Docling (Auer et al., 2024) using table-structure
extraction and OCR. We use the documents as a
corpus, generate the exam from them with our own
pipeline, and do not use UniDoc-Bench’s labeled
question-answer pairs. Table 1 reports corpus sizes.
Search space.All methods search the same space
of pipeline configurations. It spans 2 chunking
strategies over 4 chunk sizes and 4 overlaps, 13
embedding models, 2 index types with continu-
ous retrieval depth and fusion weight, 4 query-
expansion strategies, 5 reranker choices with con-
tinuous rerank depth, and 35 generator LLMs with
a per-trial reasoning toggle. Appendix B lists every
dimension with its range.
Optimizers.Every method sees the same search
space and is scored on the same fixed exam at
each trial. Random search samples configurations
uniformly and resamples when a draw violates
a dependency constraint. MO-TPE is the multi-
objective Tree-structured Parzen Estimator from
Optuna (Akiba et al., 2019) that recent RAG-tuning
work converges on (Conway et al., 2025; Orbach
et al., 2026). We follow syftr (Conway et al., 2025)
in using it as the search engine and replicate it in
our own harness with a multivariate TPE sampler.
In the Accuracy experiment it optimizes answer
accuracy alone, and in the Pareto experiment it
optimizes accuracy and cost jointly. We do not
run the full syftr system: we restrict its space to
dense and hybrid pipelines, leaving out the agentic-
RAG variants it also searches, disable its Pareto-
5

Figure 2: Best-so-far validation-exam accuracy over the 30-trial search, one panel per dataset. Each line is a
method’s running-best accuracy averaged over the 10 search seeds, and the shaded band spans ±1standard deviation
across seeds. KB Greedy, the no-search reference, is the dashed horizontal line.
frontier pruning, and run 30 trials per search rather
than the several hundred syftr uses. Reduced to
its search engine, MO-TPE differs from our agent
only in how it chooses each configuration, so the
comparison isolates the search strategy. MO-TPE
(warm) additionally seeds the study with the 30
completed trials of the paired random-search run
on the same corpus as a free, uncounted prior, so
it draws on information from 60 evaluations while
every other method sees 30. We report it at the
same 30 counted trials as a strong reference.2We
also run an ablation of our own agent, Agentic (no
KB/diag), that removes both the knowledge base
and the per-question diagnosis and leaves an LLM
optimizer that sees only the score trajectory and
its own past configurations. The gap between it
and the full agent therefore measures what the di-
agnosis loop and knowledge base add on top of the
language model. As a non-search reference, KB
Greedy takes each dimension’s top-rated option di-
rectly from the knowledge base without running
any trial. Our optimizer runs in score-only mode in
the Accuracy experiment and in cost-aware mode
in the Pareto experiment.
Evaluation and metrics.In the Accuracy exper-
iment, every method is scored at each trial on a
fixed 100-question validation exam built from the
benchmark’s labeled questions, and each returned
configuration is then re-evaluated on a separate
300-question held-out gold test set that is disjoint
from the search questions and unseen during search.
Both question sets are identical for every method
2This is a same-corpus warm start. It is not the cross-
dataset transfer seeding of syftr, which starts a new search
from Pareto-optimal flows found on different datasets.and every seed, and the seeds vary only the opti-
mizer’s own randomness, not the questions. Be-
cause each seed still returns its own configuration,
this gives ten held-out evaluations per method, and
we report the mean and standard deviation across
them. For the agentic optimizer we also take its
best configuration within the first 10 and 20 trials
as two additional returned configurations, reported
as Agentic@10 and Agentic@20. The validation
exam and the held-out set are different questions,
so per-trial validation scores and held-out scores
are not directly comparable. On the held-out set we
report exact match, token-level F1, and LLM-judge
accuracy against the gold answers. We also report
each method’s search cost and embedding compute
in Appendix D. In the Pareto experiment we report
the Pareto frontier over answer accuracy and mean
per-query API cost, scored on the corpus-derived
exam, which is generated once before any search
and identical for every method and every seed. For
each query we sum the price of its generation and
query-expansion calls from LiteLLM’s model cat-
alog and average over queries. Embedding and
reranking models run locally at no API cost, so this
axis is the generator-side cost, which also depends
on retrieval because the retrieved passages become
the generator’s input tokens.
5 Results
5.1 Accuracy experiment
Best-so-far accuracy.On all three datasets (Fig-
ure 2) the agentic optimizer’s best-so-far accuracy
climbs fastest, reaching most of its final accuracy
within the first few trials and staying at the top of
every panel through the rest of the budget. Ran-
6

HotpotQA MuSiQue MultiHop-RAG
Method EM F1 Judge EM F1 Judge EM F1 Judge
Random.506 ±.038 .664±.041 .774±.046 .165±.029 .215±.035 .270±.036 .633±.065 .651±.063 .706±.048
MO-TPE.515 ±.057 .673±.056 .784±.056 .193±.024 .254±.032 .301±.032 .664±.045 .677±.042 .724±.028
MO-TPE (warm).521 ±.044 .677±.055 .786±.057 .183±.030 .243±.032 .301±.039 .666±.051 .682±.047 .731±.045
Agentic (no KB/diag).550 ±.020 .711±.024 .819±.025 .180±.022 .230±.028 .298±.024 .729±.037 .748±.035 .780±.021
Agentic@10.558 ±.015 .715±.017 .817±.014 .177±.023 .231±.028 .303±.030 .679±.056 .697±.055 .773±.021
Agentic@20.568 ±.012 .727±.009 .835±.013 .198±.027 .261±.028 .326±.030 .704±.060 .723±.061 .783±.026
Agentic (Ours).566 ±.015 .723±.014 .825±.015 .196±.031 .262±.028 .330±.025 .729±.054 .748±.055 .790±.029
KB Greedy.480 ±.016 .611±.016 .702±.019 .150±.008 .187±.008 .254±.014 .507±.011 .518±.011 .544±.014
Table 2: Held-out accuracy on each benchmark’s gold test slice, a 300-question stratified sample disjoint from the
validation exam and unseen during search. Exact match, token-level F1, and LLM-judge accuracy are reported as
mean±SDover the 10 search seeds. Rows above the rule are search methods run under a 30-trial budget. Agentic@10
and @20 are the agentic optimizer’s best configuration within its first 10 and 20 trials. KB Greedy, below the rule,
is a no-search reference that takes each dimension’s top option from the knowledge base. Best value per column
among the search methods is in bold, with ties bold on both rows.
dom search and cold MO-TPE climb more slowly
and settle lower, and warm-started MO-TPE sits
between them and the agent. The no-KB/diag ab-
lation tracks closest to the full agent, so most of
this early efficiency comes from using a language
model to propose configurations. The added lift
from the knowledge base and per-question diagno-
sis is small in the trace and shows up more clearly
in the held-out numbers.
Held-out accuracy.On the held-out gold test
set (Table 2) the agentic optimizer achieves higher
LLM-judge accuracy than every baseline across all
three benchmarks, at 0.825 on HotpotQA, 0.330
on MuSiQue, and 0.790 on MultiHop-RAG, above
random search and with the two MO-TPE vari-
ants in between. Exact match and F1 tell the same
story on HotpotQA and MuSiQue, where an agen-
tic run holds the top score in every column. On
MultiHop-RAG the no-KB/diag ablation matches
the full agent’s exact match and F1 (0.729 and
0.748) and trails it only slightly on the judge, so
the knowledge base and per-question diagnosis add
little there over a plain LLM optimizer. Accuracy
is low for every method on MuSiQue, whose up-
to-four-hop questions need multi-step retrieval that
conditions each hop on the previous hop’s results,
a pipeline shape our single-round search space
does not support. The low ceiling there reflects
the search space rather than the optimizer, and our
agent still ranks best on MuSiQue.
5.2 Pareto experiment
Figure 3 traces each method’s median cost-
accuracy curve across the 10 seeds on the health-
care corpus. The agent’s median curve reaches
Figure 3: Cost-quality results on the UniDoc health-
care corpus, aggregated over 10 seeds. Mean API cost
per query on a log scale against accuracy on the self-
generated exam, with each method drawn as the point-
wise median across seeds of its cost-accuracy attain-
ment (line) and the interquartile (25th to 75th percentile)
spread across seeds (band). We compare our cost-aware
optimizer (Agentic) against random search, cold MO-
TPE, and warm-started MO-TPE.
77% exam accuracy at $0.000741 per query. The
strongest baseline, warm-started MO-TPE, peaks
at 71.5% and pays $0.001284, so the agent’s best
point is more accurate at 57.7% of the cost. It
matches that 71.5% at $0.000288, 22.4% of what
the baseline pays. It leads every baseline above
$0.000236 per query. Warm MO-TPE leads only
in a band from $0.000097 to $0.000236, where
neither curve reaches 60% accuracy.
These median-curve gaps could still come from
a few lucky seeds, so we test each claim across
all 10 seeds, using two-sided Mann-Whitney U
tests (Mann and Whitney, 1947) on per-seed quan-
7

tities, Holm-corrected per family of three baseline
comparisons (Holm, 1979). The higher ceiling
holds: the agent’s per-seed peak exam accuracies
span 74% to 82% and rank systematically above
each baseline’s (Holm-corrected p= 0.017 against
warm MO-TPE, 0.025 against cold MO-TPE, and
0.001 against random search). The low-cost match
holds as well: all ten agent seeds reach warm MO-
TPE’s median peak of 71.5% within their 30 trials,
against five of ten warm MO-TPE seeds, four of
ten cold MO-TPE seeds, and two of ten random-
search seeds. When seeds are ranked by the lowest
per-query cost at which they reach this accuracy,
with seeds that never reach it counted as the most
expensive, the agent leads every baseline (Holm-
correctedp≤0.001).
6 Discussion
The two experiments show the optimizer from two
angles. In the Accuracy experiment, which iso-
lates the search from cost, the agent is sample-
efficient: its best configuration within the first 10
trials already matches or beats every statistical base-
line’s full 30-trial held-out judge accuracy, at 0.817
against 0.786 on HotpotQA, 0.773 against 0.731
on MultiHop-RAG, and a tie on MuSiQue at 0.303
against 0.301, and the best-so-far trace (Figure 2)
puts it near the top of every panel within the first
few trials. The remaining 20 trials refine this region,
and the full budget is what lets the agent match the
stronger no-KB/diag ablation on MultiHop-RAG
and lead on exact match and F1 on MuSiQue. The
sample-efficiency reading holds even against warm-
started MO-TPE, which begins with a free 30-trial
same-corpus prior and so draws on information
from 60 evaluations, yet is matched or beaten by
the agent’s first 10 counted trials.
The Pareto experiment is the setting we built the
optimizer for, and it is a practical deployment sce-
nario. Deploying a RAG pipeline is rarely a search
for the most accurate configuration at any price.
A practitioner usually wants the cheapest pipeline
that clears an accuracy bar, because inference cost
is paid on every query once the system is live. Our
cost-aware experiment puts the agent in exactly
this position, running it end to end on the real-
world healthcare corpus. On this corpus the agent’s
median trade-off both undercuts the baselines on
cost and beats them on accuracy: it matches the
strongest baseline’s median accuracy at about 22%
of the cost and extends to a higher ceiling that stillcosts less per query than the baselines’ own best.
It manages this by pricing and ranking candidates
against its knowledge base before running them, so
it moves directly to small but capable generators in-
stead of spending trials to learn which cheap model
suffices. Instead of a single configuration it returns
the whole accuracy-cost trade-off and leaves the
operating point to the user.
7 Conclusion
We introduce Agentic AutoRAG, a reasoning-
driven optimizer for RAG pipelines. Instead of
reducing each trial to an opaque score, it scores ev-
ery configuration on a frozen corpus-derived exam,
and after each trial a Diagnoser attributes each
failed question to retrieval or generation while a
Proposer, grounded in a knowledge base of public
model rankings and pricing, selects the next con-
figuration. On three multi-hop QA benchmarks
it reaches higher LLM-judge accuracy than every
baseline we compare and finds most of that accu-
racy within its first 10 trials. In its cost-aware mode
on a healthcare corpus it reaches a median exam
accuracy of 77%, above the strongest baseline’s
71.5%, at about 58% of that baseline’s cost per
query, and it matches that 71.5% at about 22% of
the cost.
Limitations
Fixed parser.Parsing is outside the search space.
Every trial parses the corpus once with Docling,
caches the result, and tunes chunking and down-
stream stages on top of it. Reparsing on every trial
would dominate per-trial wall-clock time.
Knowledge-base inputs.The Proposer’s KB en-
codes LLM rankings from Artificial Analysis and
per-token pricing from LiteLLM, neither of which
is peer-reviewed. The KB is also a point-in-time
snapshot. Vendors re-rank, deprecate models, and
update prices over time, so the Proposer’s ground-
ing is accurate at snapshot time and degrades with
the gap to the run.
Agent nondeterminism.Inference temperature
is configurable, but the agent models limit repro-
ducibility. The Composer, Diagnoser, and Proposer
default to a frontier reasoning model, several rea-
soning models require a temperature of 1.0 to func-
tion, and hosted reasoning-model APIs do not guar-
antee bit-exact outputs.
8

Judge dependence.Accuracy in both experi-
ments is scored by an LLM judge, and we run
no human-agreement study, so any systematic bias
in the judge carries into the numbers we report.
Future Work
Richer cross-document bridges.Our exam
pipeline ranks bridges by lexical TF-IDF similarity,
which connects only chunks that share rare surface
terms. An entity-relation graph over the corpus,
as in graph-based RAG (Edge et al., 2024), would
also surface bridges that share an entity but not its
wording.
Expanding the search space.The diagnose-
then-propose loop is agnostic to what the search
space contains, so it can tune richer pipeline fam-
ilies than the modular flow we search here. Sup-
porting whole architectures, such as agentic RAG
or graph-based RAG, would let it search across
retrieval paradigms rather than within one.
More objectives.The optimizer weighs only ac-
curacy and cost today, and its knowledge base al-
ready records throughput and memory, so latency
or memory footprint are natural additions.
Warm starts across runs.Each run begins cold
on a new corpus. Carrying diagnoses and config-
uration priors across runs would let it start warm,
accumulating an agent memory the optimizer re-
trieves from to generalize what it learns about RAG
configuration from one corpus to the next.
Acknowledgments
This work was carried out at the ETH Agentic Sys-
tems Lab. We thank David Schaurecker for provid-
ing API credits to support the LLM usage.
References
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. 2019. Optuna: A next-
generation hyperparameter optimization framework.
InProceedings of the 25th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery & Data
Mining, pages 2623–2631.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.Christoph Auer, Maksym Lysak, Ahmed Nassar,
Michele Dolfi, Nikolaos Livathinos, Panos Vage-
nas, Cesar Berrospi Ramis, Matteo Omenetti, Fabian
Lindlbauer, Kasper Dinkla, Lokesh Mishra, Yusik
Kim, Shubham Gupta, Rafael Teixeira de Lima,
Valery Weber, Lucas Morin, Ingmar Meijer, Viktor
Kuropiatnyk, and Peter W. J. Staar. 2024. Docling
technical report.arXiv preprint arXiv:2408.09869.
Matthew Barker, Andrew Bell, Evan Thomas, James
Carr, Thomas Andrews, and Umang Bhatt. 2025.
Faster, cheaper, better: Multi-objective hyperparame-
ter optimization for LLM and RAG systems. InICLR
2025 Workshop on Foundation Models in the Wild.
Alexander Conway, Debadeepta Dey, Stefan Hackmann,
Matthew Hausknecht, Michael Douglas Schmidt,
Mark Lewis Steadman, and Nick V olynets. 2025.
syftr: Pareto-optimal generative AI. InProceedings
of the Fourth International Conference on Automated
Machine Learning, volume 293 ofProceedings of
Machine Learning Research, pages 6/1–33. PMLR.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua,
Márton Kardos, Ashwin Mathur, David Stap,
Jay Gala, Wissam Siblini, Dominik Krzemi ´nski,
Genta Indra Winata, Saba Sturua, Saiteja Utpala, et al.
2025. MMTEB: Massive multilingual text embed-
ding benchmark. InInternational Conference on
Learning Representations (ICLR).
Simone Filice, Guy Horowitz, David Carmel, Zohar
Karnin, Liane Lewin-Eytan, and Yoelle Maarek.
2025. Generating Q&A benchmarks for RAG evalua-
tion in enterprise settings. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 6: Industry Track), pages 469–
484, Vienna, Austria. Association for Computational
Linguistics.
Jia Fu, Xiaoting Qin, Fangkai Yang, Lu Wang, Jue
Zhang, Qingwei Lin, Yubo Chen, Dongmei Zhang,
Saravan Rajmohan, and Qi Zhang. 2024. AutoRAG-
HP: Automatic online hyper-parameter tuning for
retrieval-augmented generation. InFindings of the
Association for Computational Linguistics: EMNLP
2024, pages 3875–3891, Miami, Florida, USA. Asso-
ciation for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997.
Gauthier Guinet, Behrooz Omidvar-Tehrani, Anoop De-
oras, and Laurent Callot. 2024. Automated evalu-
ation of retrieval-augmented language models with
9

task-specific exam generation. InProceedings of the
41st International Conference on Machine Learning,
volume 235 ofProceedings of Machine Learning
Research, pages 16773–16801. PMLR.
Sture Holm. 1979. A simple sequentially rejective mul-
tiple test procedure.Scandinavian Journal of Statis-
tics, 6(2):65–70.
Muhammed Yusuf Kartal, Suha Kagan Kose, Ko-
rhan Sevinç, and Burak Aktas. 2025. RAGSmith:
A framework for finding the optimal composition
of retrieval-augmented generation methods across
datasets.arXiv preprint arXiv:2511.01386.
Dongkyu Kim, Byoungwook Kim, Donggeon Han, and
Matouš Eibich. 2024. AutoRAG: Automated frame-
work for optimization of retrieval augmented genera-
tion pipeline.arXiv preprint arXiv:2410.20878.
Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong,
Bojian Xiong, Yurong Zhang, Yan Gao, Yi Wu, Yao
Hu, and Deyi Xiong. 2025. DecEx-RAG: Boosting
agentic retrieval-augmented generation with decision
and execution optimization via process supervision.
InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 1412–1425, Suzhou, China. Association
for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Siyi Liu, Chen Gao, and Yong Li. 2025. AgentHPO:
Large language model agent for hyper-parameter op-
timization. InProceedings of the Conference on Par-
simony and Learning, volume 280 ofProceedings
of Machine Learning Research, pages 1146–1169.
PMLR.
Henry B. Mann and Donald R. Whitney. 1947. On
a test of whether one of two random variables is
stochastically larger than the other.The Annals of
Mathematical Statistics, 18(1):50–60.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2023. MTEB: Massive text embedding
benchmark. InProceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics, pages 2014–2037, Dubrovnik,
Croatia. Association for Computational Linguistics.
Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel
Gera, Odellia Boni, Yoav Kantor, Gal Bloch,
Omri Levy, Hadas Abraham, Nitzan Barzilay, Eyal
Shnarch, Michael E. Factor, Shila Ofek-Koifman,
Paula Ta-Shma, and Assaf Toledo. 2026. An anal-
ysis of hyper-parameter optimization methods for
retrieval augmented generation. InAAAI 2026 Work-
shop on New Frontiers in Information Retrieval.Xiangyu Peng, Can Qin, Zeyuan Chen, Ran Xu, Caim-
ing Xiong, and Chien-Sheng Wu. 2025. UniDoc-
Bench: A unified benchmark for document-centric
multimodal RAG.arXiv preprint arXiv:2510.03663.
Valentina Pyatkin, Saumya Malik, Victoria Graf,
Hamish Ivison, Shengyi Huang, Pradeep Dasigi,
Nathan Lambert, and Hannaneh Hajishirzi. 2025.
Generalizing verifiable instruction following. InAd-
vances in Neural Information Processing Systems
(NeurIPS), Datasets and Benchmarks Track.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R. Bowman. 2024. GPQA:
A graduate-level google-proof q&a benchmark. In
Conference on Language Modeling (COLM).
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Sumuk Shashidhar, Clémentine Fourrier, Alina Lo-
zovskaya, Thomas Wolf, Gokhan Tur, and Dilek
Hakkani-Tür. 2025. Yourbench: Dynamic evalua-
tion set generation with LLMs. InConference on
Language Modeling (COLM).
Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Ta-
laei Khoei, and Athanasios V . Vasilakos. 2025. Agen-
tic retrieval-augmented generation: A survey on agen-
tic RAG.arXiv preprint arXiv:2501.09136.
Yixuan Tang and Yi Yang. 2024. MultiHop-RAG:
Benchmarking retrieval-augmented generation for
multi-hop queries. InConference on Language Mod-
eling (COLM).
Patara Trirat, Wonyong Jeong, and Sung Ju Hwang.
2025. AutoML-Agent: A multi-agent LLM frame-
work for full-pipeline AutoML. InProceedings of the
42nd International Conference on Machine Learn-
ing, volume 267 ofProceedings of Machine Learning
Research, pages 60099–60146. PMLR.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Huang. 2024. Searching for best practices in
retrieval-augmented generation. InProceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pages 17716–17736, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
10

Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle
Lo, Arman Cohan, Benjamin Van Durme, Dawn
Lawrie, and Luca Soldaini. 2025. FollowIR: Eval-
uating and teaching information retrieval models to
follow instructions. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 11926–11942, Albuquerque, New Mexico. As-
sociation for Computational Linguistics.
Ruofan Wu, Youngwon Lee, Fan Shu, Danmei Xu,
Seung-won Hwang, Zhewei Yao, Yuxiong He, and
Feng Yan. 2025. ComposeRAG: A modular and com-
posable RAG for corpus-grounded multi-hop ques-
tion answering.arXiv preprint arXiv:2506.00232.
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu,
Quoc V . Le, Denny Zhou, and Xinyun Chen. 2024.
Large language models as optimizers. InInterna-
tional Conference on Learning Representations.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Tian Yu, Shaolei Zhang, and Yang Feng. 2024.
Auto-RAG: Autonomous retrieval-augmented gen-
eration for large language models.arXiv preprint
arXiv:2411.19443.
A Question taxonomy and exam samples
This appendix complements the exam-generation
pipeline in Section 3.2 by listing the seven ques-
tion types the Composer draws from and showing
one representative item per type. The examples
come from the 100-question exam the optimizer
generated for the UniDoc healthcare corpus, the
one experiment where the generated exam is the op-
timization target. The Accuracy experiment instead
optimizes against each benchmark’s own held-in
questions and generates no exam, so this appendix
documents the pipeline where it runs.
A.1 Question types
The Composer assigns one type per question from
a fixed taxonomy of seven question types defined
a priori. The taxonomy extends the bridge and
comparison split of HotpotQA (Yang et al., 2018)
with numeric and inference variants common to
multi-hop benchmarks such as MuSiQue (Trivedi
et al., 2022).
Extraction. A factoid lookup answerable from a
single span.Definitional. A short definition or description the
chunk states explicitly.
Numeric (single-span). Computes a value not
stated verbatim by combining at least two nu-
meric literals inside one span (sum, difference,
range, median, count).
Inference. Composes facts from at least two dis-
tinct spans into an answer that is not a contigu-
ous substring (causal chain, implicit-referent
resolution, qualitative direction from quantita-
tive facts).
Bridge. References an entity in one chunk via
an indirect descriptor that another chunk
uniquely identifies, then asks for an attribute
of that entity.
Comparison. Reads a comparable value from
each cited chunk and returns the relation
(which is larger, by how much).
Numeric (multi-span). Computes across cited
spans: difference, sum, ratio, duration.
Both numeric types also carry an arithmetic for-
mula. The Composer must supply it alongside the
answer, and the validator rejects the question when
the formula does not evaluate to the canonical an-
swer. This guards against arithmetic mistakes by
the Composer LLM.
Source spans.Each source span is the minimal
excerpt the answer strictly depends on, usually a
single sentence. A span carries none of the sur-
rounding context, so the example spans below can
be hard to read on their own. Each records only
the exact text a pipeline must retrieve, not the con-
text that explains the answer. Retrieval counts as
successful for a question only when the retrieved
chunks contain all of its spans, and the per-question
diagnosis uses this test to attribute a failure to weak
retrieval rather than weak generation.
Distribution on UniDoc.Table 3 reports the type
distribution of the 100 questions that survive valida-
tion and probe-based selection on UniDoc. Extrac-
tion is well represented (16): the documents state
many specific facts buried in long, multi-section
text that are not trivially retrievable. Definitional
questions do not survive selection, and only one
single-span numeric question does, because most
numeric reasoning here combines values that sit in
11

Question type Count
Extraction 16
Definitional 0
Numeric (single-span) 1
Inference 25
Bridge 19
Comparison 25
Numeric (multi-span) 14
Total 100
Table 3: Question-type distribution of the 100-question
UniDoc exam.
different spans. The exam also stays within docu-
ments: 98 questions keep all their spans within one
document, and only 2 cross into a second.
A.2 Example questions
One representative item per surviving type, drawn
verbatim from the UniDoc exam. Source spans are
reproduced as written.
Extraction
Question.What fraction of European American hap-
lotypes in the registry are represented by only one
individual?
Answer.44.65% (variant: “44.65 percent”).
Source span.“Application of our estimator to 6.59
million donors in the Be The Match Registry revealed
that haplotypes follow a heavy tail distribution across
all ethnicities: for example, 44.65% of the European
American haplotypes are represented by only 1 indi-
vidual.”
Numeric (single-span)
Question.What is the average CK-MB value across
both cattle cases with tropical theileriosis?
Answer.362.5 U/L.
Formula.(344 + 381)/2 = 362.5.
Source span.“CK-MB, Table 2. Biochemical find-
ings of two cases.Unit = (U/L). CK-MB, Case-1 =
344. CK-MB, Case-2 = 381. CK-MB, Referance
Range* = 189-235 **.”
Inference
Question.Why does the commercial Indian an-
tivenom show limited efficacy against the newly dis-
covered krait species from Southern India?
Answer.Antivenom is manufactured from B.
caeruleus venom, not from the new species (variant:
“because it is made from B. caeruleus venom”).
Source spans.
“Considering the near countrywide distribution of this
clinically important snake, B. caeruleus venoms are
used for the manufacture of commercial Indian poly-
valent antivenoms.”
“While both antivenoms recognised the venoms of B.
caeruleus (titre: 1:500; PSVPL: 1.02 Optical Den-sity (OD); Haffkine: 0.89 OD) and B. sindanus (titre:
1:500; PSVPL: 0.89 OD; Haffkine: 0.76 OD) rela-
tively well, they exhibited limited to poor recogni-
tion potential against the venom of B. romulusi (titre:
1:500; PSVPL: 0.24 OD; Haffkine: 0.25 OD).”
Bridge
Question.For the treatment comparison that showed
a statistically significant difference in adverse reac-
tions ( p <0.001 ), what was the male-to-female ratio
in each arm?
Answer.59:7 (variant: “59/7”).
Source spans.
“Male/female ratio, Group I (Budesonide 400 mcg
plus Formoterol 6 mcg) = 59/7. Male/female ratio,
Group II (Tiotropium 18 mcg plus Formoterol 12
mcg) = 59/7.”
“In group I, following adverse drug reactions were
observed; sore throat, hoarseness of voice, oral ulcers,
tremors, 2 patients showed acute exacerbation and
1 had pneumonitis. In group II, following adverse
drug reactions were observed; dryness of mouth, pal-
pitations, tremors and 2 patients developed urinary
retention. After that we have done Fisher’s exact test
and we observed that there is statistically significant
difference ( p <0.001 ) between group I and group
II.”
Comparison
Question.How did particle concentration change
when antibodies were added without magnetic beads,
compared to when uncoated beads were used?
Answer.Increased by 105% versus no change (vari-
ant: “rose by 105% versus unchanged”).
Source spans.
“We applied an identical bead procedure, except that
we abstained from coating the beads with anti-ApoB
antibodies prior to the incubation with the diluted PFP
sample. This resulted in no changes in particle con-
centration and size distribution curve.”
“we conducted a parallel experiment with a two-hour
incubation of diluted PFP with 10 µg of anti-ApoB an-
tibodies that had not been attached to magnetic beads,
which resulted in an increase in particle concentra-
tion by 105% as well as a notable shift of the size
distribution curve to the right”
Numeric (multi-span)
Question.What is the ratio of items to care-plan do-
mains in the interRAI Mental Health assessment?
Answer.19.8 items per domain (variant: “approxi-
mately 20 items per domain”).
Formula.396/20 = 19.8.
Source spans.
“The interRAI Mental Health (MH) assessment (157,
158) supports care plan development in 20 domains,
and is used in in-patient mental health settings”
“The assessment is comprised of 396 items in several
domains (see Table 1).”
12

B Search space
The search space is one of the inputs the user sup-
plies to the optimizer, alongside the corpus. It
is a single YAML file (Listing 1) that names the
corpus, the models that drive the optimizer, the
exam-generation settings, and the dimensions to
search. The optimizer reads this file directly, so the
search space is fully specified and auditable before
any trial runs.
Listing 1 shows the configuration for the Pareto
experiment on UniDoc, where the optimizer gen-
erates and scores its own exam. The Accuracy ex-
periment, on HotpotQA, MuSiQue, and MultiHop-
RAG, shares the same search_space and differs
only in its exam source, cost mode, and parsing,
each described below. Those runs supply each
benchmark’s own held-in questions as the valida-
tion exam through examiner.custom_exam_path
and generate none, so the examiner block in the
listing does not apply to them. They also run score-
only rather than cost-aware, and they parse pre-
extracted text with OCR and table-structure extrac-
tion off. UniDoc, built from real PDFs and page
images, turns both on, sets cost_aware to true,
and weights the exam-generation neighborhoods
toward same-document siblings ( 0.8 / 0.2 ) be-
cause its documents are long and multi-section.
# Project configuration: corpus, optimizer models, exam
# settings, and the search space. Provided by the user.
meta:
project_name: unidoc-pareto
corpus_path: ./corpus
# free-text hint read by the optimizer
corpus_description: |
Real-world healthcare PDFs and scanned page images:
insurance forms, claims, clinical notes. Dense,
domain-specific prose with tables and mixed layouts.
max_trials: 30 # search budget
cost_aware: true # Pareto run; the Accuracy experiment
sets false
model_aliases: # map names to your deployments
azure/gpt-5-nano: <your-deployment>
azure/gpt-5-mini: <your-deployment>
azure/gpt-4.1-nano: <your-deployment>
azure/gpt-4.1-mini: <your-deployment>
# LLMs that drive the optimizer and grade answers
agent:
optimizer_model: kimi-k2.5 # diagnose + propose
examiner_model: minimax-m2.5 # compose exam
judge_model: DeepSeek-V3.2 # grade answers
optimizer_reasoning_effort: medium
examiner_reasoning_effort: medium
examiner: # exam generation (skipped when an exam is
supplied)
exam_size: 100
# seed this x exam_size neighborhoods; the composer
# over-generates questions, then filter to exam_size
initial_question_multiplier: 1.8
probe_selection: true # keep discriminative items
neighborhood_min_chunks: 18 # context per call
neighborhood_min_words: 5000
# neighborhood mix, favoring same-document spans
neighborhood_same_doc_weight: 0.8neighborhood_cross_doc_weight: 0.2
parsing: # OCR + tables on for the UniDoc PDFs
parser: docling
ocr: true
table_structure: true
search_space: # dimensions searched
chunking: # how documents are split
strategies: [recursive, fixed]
chunk_token_size: [128, 256, 384, 512]
chunk_token_overlap: [0, 32, 48, 64]
embedding: # 13 retrievers
models:
- IEITYuan/Yuan-embedding-2.0-en
- Qwen/Qwen3-Embedding-0.6B
- ibm-granite/granite-embedding-english-r2
- ibm-granite/granite-embedding-small-english-r2
- intfloat/e5-large-v2
- intfloat/multilingual-e5-large-instruct
- BAAI/bge-large-en-v1.5
- BAAI/bge-small-en-v1.5
- BAAI/bge-m3
- Snowflake/snowflake-arctic-embed-l-v2.0
- Snowflake/snowflake-arctic-embed-xs
- mixedbread-ai/mxbai-embed-large-v1
- sentence-transformers/all-MiniLM-L6-v2
retrieval:
index_types: [vector_only, hybrid_bm25_vector]
top_k: { min: 3, max: 20 } # retrieved chunks
hybrid_alpha: { min: 0.0, max: 1.0 } # lexical/dense
bm25_vector_fusion: [alpha, rrf]
long_context_reorder: [false, true]
query_expansion:
strategies: [none, hyde, multi_query, query_decompose]
models: [azure/gpt-4o-mini] # fixed expander
reranker:
models:
- none
- BAAI/bge-reranker-v2-m3
- cross-encoder/ms-marco-MiniLM-L-6-v2
- Alibaba-NLP/gte-reranker-modernbert-base
- mixedbread-ai/mxbai-rerank-xsmall-v1
top_n: { min: 3, max: 10 } # reranked chunks
generator: # 35 answer LLMs
models:
- bedrock/us.amazon.nova-micro-v1:0
- bedrock/us.amazon.nova-lite-v1:0
- bedrock/us.amazon.nova-pro-v1:0
- bedrock/google.gemma-3-4b-it
- bedrock/google.gemma-3-12b-it
- bedrock/google.gemma-3-27b-it
- bedrock/us.meta.llama3-1-8b-instruct-v1:0
- bedrock/us.meta.llama3-3-70b-instruct-v1:0
- bedrock/us.meta.llama4-scout-17b-instruct-v1:0
- bedrock/us.meta.llama4-maverick-17b-instruct-v1:0
- bedrock/mistral.magistral-small-2509
- bedrock/mistral.ministral-3-3b-instruct
- bedrock/mistral.ministral-3-8b-instruct
- bedrock/mistral.ministral-3-14b-instruct
- bedrock/mistral.mistral-7b-instruct-v0:2
- bedrock/mistral.mistral-large-3-675b-instruct
- bedrock/mistral.mixtral-8x7b-instruct-v0:1
- bedrock/nvidia.nemotron-nano-3-30b
- bedrock/openai.gpt-oss-20b-1:0
- bedrock/openai.gpt-oss-120b-1:0
- bedrock/qwen.qwen3-32b-v1:0
- bedrock/qwen.qwen3-coder-30b-a3b-v1:0
- bedrock/zai.glm-4.7
- azure/gpt-4o
- azure/gpt-4o-mini
- azure/gpt-4.1-mini
- azure/gpt-4.1-nano
- azure/o4-mini
- azure/gpt-5-mini
- azure/gpt-5-nano
- azure/gpt-5.4-mini
- azure/gpt-5.4-nano
- vertex_ai/gemini-2.5-flash
- vertex_ai/gemini-2.5-flash-lite
- vertex_ai/gemini-3.1-flash-lite-preview
# when true, trials may use reasoning or plain models,
13

# passing reasoning params to LiteLLM where supported
reasoning: true
reasoning_effort: medium # when reasoning is on
temperature: { min: 1.0, max: 1.0 }
Listing 1: Project configuration for the Pareto
experiment on UniDoc, the file the user supplies to the
optimizer. The Accuracy experiment shares the same
search_space but supplies its own held-in exam and
runs score-only.
C Knowledge base
Appendix B names the candidate models. This ap-
pendix shows the annotations the Proposer reasons
over for each one. The knowledge base pairs every
model in the search space with public benchmark
scores, per-token pricing, and serving attributes,
which lets the Proposer rank candidates and rule
out quality-cost-dominated choices before it spends
a trial on them. LLM benchmarks and throughput
come from Artificial Analysis,3embedding and
reranker rankings from the MTEB (Muennighoff
et al., 2023), MMTEB (Enevoldsen et al., 2025),
and FollowIR (Weller et al., 2025) benchmarks,
and per-token pricing from LiteLLM.4The opti-
mizer renders the knowledge base filtered to the
configured search space and to the run’s reasoning-
effort setting, so a model that supports reasoning
can appear as two entries, one reasoning and one
non-reasoning, with different scores. The 35 gener-
ator models of Appendix B therefore render as 49
scored entries.
Tables 4, 5, and 6 report the generator, embed-
ding, and reranker annotations. The embedding
and reranker tables are complete, whereas for gen-
erators we show twelve models spanning the intel-
ligence range. The Proposer treats these scores as
comparative signal, not ground truth.
The scores are a point-in-time snapshot, taken
in May 2026 for the LLM benchmarks and pricing
and from the MTEB(eng, v2) leaderboard for the
embedders.
3https://artificialanalysis.ai/
4https://github.com/BerriAI/litellm
14

Model Creator Intel. GPQA IFBench In $/M Out $/M Tok/s Reasoning
glm-4.7 Z AI 42.1 0.859 0.679 0.60 2.20 86 Yes
gpt-5-mini OpenAI 38.9 0.803 0.712 0.25 2.00 106 Yes
gpt-5.4-mini OpenAI 37.7 0.823 0.648 0.75 4.50 172 Yes
gpt-oss-120b OpenAI 33.3 0.782 0.690 0.15 0.60 289 Yes
gemini-2.5-flash Google 27.0 0.790 0.503 0.30 2.50 225 Yes
gpt-5-nano OpenAI 25.9 0.670 0.659 0.05 0.40 151 Yes
gpt-4.1-mini OpenAI 22.9 0.664 0.383 0.40 1.60 80 No
mistral-large-3 Mistral 22.8 0.680 0.362 0.50 1.50 58 No
llama4-maverick Meta 18.4 0.671 0.430 0.24 0.97 122 No
gpt-4o OpenAI 17.3 0.543 0.343 2.50 10.00 172 No
nova-micro Amazon 10.3 0.358 0.294 0.04 0.14 318 No
gemma-3-4b Google 6.3 0.291 0.283 0.04 0.08 – No
Table 4: Generator-LLM annotations in the knowledge base, a twelve-model slice across the intelligence range (the
full set of 35 models, which render as 49 reasoning and non-reasoning entries, is named in Appendix B). Intel. is the
Artificial Analysis intelligence index. GPQA (Rein et al., 2024) and IFBench (Pyatkin et al., 2025) are accuracies in
[0,1] . Prices are USD per million input and output tokens. Tok/s is median output throughput. Reasoning marks
which variant the listed scores describe. Model names abbreviate the LiteLLM identifiers in Appendix B, and a
dash marks a value unpublished in the snapshot.
Model Dim. Max tok. Params (B) Retrieval STS Rerank. Mem. (MB)
Yuan-embedding-2.0-en 1024 2048 0.596 0.707 0.849 0.533 2272
Qwen3-Embedding-0.6B 1024 32768 0.596 0.618 0.858 0.482 1136
snowflake-arctic-embed-l-v2.0 1024 8192 0.568 0.586 0.770 0.475 2166
granite-embedding-english-r2 768 8192 0.149 0.564 0.781 0.491 284
mxbai-embed-large-v1 1024 512 0.335 0.556 0.772 0.480 639
bge-large-en-v1.5 1024 512 0.335 0.554 0.759 0.483 1242
granite-embedding-small-english-r2 384 8192 0.048 0.539 0.766 0.476 91
bge-small-en-v1.5 512 512 0.033 0.539 0.739 0.469 127
multilingual-e5-large-instruct 1024 514 0.560 0.535 0.843 0.487 1068
snowflake-arctic-embed-xs 384 512 0.023 0.526 0.728 0.453 86
bge-m3 1024 8194 0.568 0.510 0.799 – 2167
e5-large-v2 1024 514 0.335 0.493 0.749 0.457 1278
all-MiniLM-L6-v2 384 256 0.023 0.430 0.704 0.471 87
Table 5: The 13 embedding models in the search space with their MTEB(eng, v2) (Muennighoff et al., 2023)
scores. Retrieval, STS, and reranking (Rerank.) are the corresponding MTEB task scores in [0,1] . Dim. is the
embedding dimension, Max tok. the maximum input length, Params (B) the parameter count in billions, and Mem.
the memory footprint. The Proposer ranks retrievers by the Retrieval score. Names drop the organization prefix
listed in Appendix B, and a dash marks a value unpublished in the snapshot.
Model Params (B) MTEB-R MMTEB-R FollowIR
bge-reranker-v2-m3 0.6 0.570 0.584−0.01
ms-marco-MiniLM-L-6-v2 0.022 – – –
gte-reranker-modernbert-base – – – –
mxbai-rerank-xsmall-v1 – – – –
Table 6: The four reranker models in the search space (a fifth option, none , disables reranking). MTEB-R and
MMTEB-R are the MTEB (Muennighoff et al., 2023) and MMTEB (Enevoldsen et al., 2025) reranking-task scores
in[0,1] , and FollowIR (Weller et al., 2025) is the instruction-following retrieval score. Coverage is sparse, with
published scores for only one of the four rerankers in the snapshot, so the Proposer has little benchmark signal here
and relies on rerank depth and downstream accuracy. Names abbreviate the identifiers in Appendix B, and a dash
marks a value unpublished in the snapshot.
15

Figure 4: Search cost and embedding compute on the three datasets of the Accuracy experiment. Bars show the
mean over the 10 search seeds and whiskers span ±1standard deviation across seeds. In the search-cost panel
each bar splits into optimizer reasoning (the agent’s Proposer and Diagnoser calls) and trial evaluation (the RAG
generation and judge scoring at each trial). Random search and MO-TPE propose configurations without a language
model, so their cost is entirely trial evaluation. The embedding panel counts embedding-input tokens in millions,
one charge per distinct embedder and chunk-size configuration rather than per trial.
D Search cost and embedding compute
Figure 4 reports two costs of running each opti-
mizer on the three datasets of the Accuracy experi-
ment, the dollar cost of the search and the embed-
ding compute it consumes. We count the latter as
embedding-input tokens, charged once per distinct
embedder and chunk-size configuration, since each
index is built once and reused across a run’s tri-
als. Bars are the mean over the 10 search seeds
and whiskers span ±1standard deviation across
seeds. The methods differ most on embedding to-
kens used. The agent’s search consumes 11 to 16M
embedding tokens per dataset, against 46 to 79M
for random search and 34 to 63M for MO-TPE,
three to five times fewer. Guided by the knowledge
base, it settles on a few strong embedders early and
reuses their cached indexes, while random search
and MO-TPE sample embedders and chunk sizes
across the whole space and pay for a fresh index
build on each new combination.
In dollars the methods are closer, and the full
30-trial agent costs more than the statistical base-
lines. The gap is trial evaluation, not reasoning.
Its Proposer and Diagnoser calls add only $0.71 to
$0.79 per run, while random search and MO-TPEchoose configurations without a language model
and spend nothing on reasoning. What raises the
agent’s bill is the pipelines it evaluates. These are
score-only runs, so cost is not an objective and noth-
ing steers the agent toward cheaper generators. It
knows from the knowledge base and the language
model’s own priors which generators are strong,
and strong generators tend to cost more per token,
so its trials favor more capable, more expensive
configurations. However, reaching their accuracy
does not take the full budget. Its 10-trial checkpoint
recovers most of the agent’s final held-out accuracy
(Table 2) at $1.48 to $2.92 per run, comparable to
random search and MO-TPE at $1.86 to $4.45.
16