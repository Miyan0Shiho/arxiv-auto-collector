# Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law

**Authors**: Rose Cymbler, Daniel Guez, Laurent Fabre

**Published**: 2026-08-10 10:20:13

**PDF URL**: [https://arxiv.org/pdf/2608.09393v1](https://arxiv.org/pdf/2608.09393v1)

## Abstract
We identify and quantify temporal misgrounding: the systematic retrieval and citation of the currently in-force version of a legal article when the applicable version is an earlier or future one. Standard legal RAG treats the corpus as static; we argue legal question answering is a temporally-indexed retrieval problem. We introduce FiscalQA Pro, pairing a versioned corpus of 32,436 article-versions of the French tax code (93 years, 1938-2031) with an all-model-hard temporal-reasoning track: 209 scored, expert-reviewed questions across 33 CGI articles (221 released; twelve flagged out of the answerable scope). At selection time, no evaluated model recovered its date-applicable answer closed-book in any of four sampling draws, and the currently in-force text lacks the gold value for all but one of the scored questions. Answers are scored deterministically via atomic ground-truth "nuggets" (regex and numeric-with-tolerance), never LLM-as-judge: an LLM judge would inherit the temporal bias it is meant to score. Across eleven models (five frontier closed-API systems plus Gemini 2.5 Pro as a substitute entry, and five open-weight), parametric knowledge yields 3.0% mean strict accuracy and RAG over a static current-version corpus 2.7%. Static RAG retrieves the date-applicable version 0% of the time, confidently citing a real but inapplicable version. Our end-to-end retriever over a multi-version index, with no oracle, reaches 98.3% mean strict; an oracle-article ablation reaches 99.1%, locating the residual gap in first-stage recall, not version selection. We additionally release a version-aware jurisprudence dataset of 69,208 citation links, together with the corpus, benchmark, model responses, and pipeline code.

## Full Text


<!-- PDF content starts -->

Temporal Misgrounding in Legal RAG:
A Versioned-Corpus Benchmark for French Tax Law
Rose Cymbler1Daniel Guez1Laurent Fabre2
Abstract
We identify and quantifytemporal misground-
ing: the systematic retrieval and citation of the
currently in-forceversion of a legal article when
the applicable version is an earlier or future
one. Standard legal RAG treats the corpus as
static; we argue legal question answering is a
temporally-indexedretrieval problem. We intro-
duceFiscalQA Pro, pairing a versioned corpus
of 32,436 article-versions of the French tax code
(93 years, 1938–2031) with an all-model-hard
temporal-reasoning track: 209 scored, expert-
reviewed questions across 33 CGI articles (221
released; twelve flagged out of the answerable
scope). At selection time, no evaluated model
recovered its date-applicable answer closed-book
in any of four sampling draws, and the currently
in-force text lacks the gold value for all but one
of the scored questions. Answers are scoredde-
terministicallyvia atomic ground-truth “nuggets”
(regex and numeric-with-tolerance), never LLM-
as-judge: an LLM judge would inherit the tempo-
ral bias it is meant to score. Across eleven mod-
els (five frontier closed-API systems plus Gem-
ini 2.5 Pro as a substitute entry, and five open-
weight), parametric knowledge yields 3.0% mean
strict accuracy and RAG over a static current-
version corpus 2.7%. Static RAG retrieves the
date-applicable version 0% of the time, confi-
dently citing a real but inapplicable version. Our
end-to-end retriever over a multi-version index,
with no oracle, reaches98.3% mean strict; an
oracle-article ablation reaches 99.1%, locating the
Author contributions:R.C. led the research, designed the
benchmark, built the retrieval architecture, ran the controlled
experiment, and led the writing. D.G. built the data infrastruc-
ture, extraction, and versioned corpus. L.F. contributed to the
evaluation methodology, experimental design, analysis and inter-
pretation of results, and the positioning of the work.1Talia,
Paris, France2Databricks. Correspondence to: Rose Cymbler
<rose.cymbler@talia-ai.com>.
ICML 2026 Workshop on AI for Law (AI4Law), Seoul, South
Korea, 2026. Copyright 2026 by the author(s).residual gap in first-stage recall, not version se-
lection. We additionally release a version-aware
jurisprudence dataset of 69,208 citation links, to-
gether with the corpus, benchmark, model re-
sponses, and pipeline code.1
1. Introduction
Motivation.Recent progress in large language models
(LLMs) and retrieval-augmented generation (RAG) has
spurred a wave of legal-NLP benchmarks aimed at mea-
suring legal reasoning, citation extraction, and document un-
derstanding (Guha et al., 2023; Niklaus et al., 2023; Douka
et al., 2021). Yet most of these benchmarks share a critical
implicit assumption: that the legal corpus isstatic. This
assumption is most dramatically violated in tax law, where
statutes are amended yearly through finance laws (lois de
finances) and where the correct answer to a legal question
often depends critically on the date of application. A tax-
payer asking “what was the standard corporate income tax
rate in 2018?” should receive 331
3%, not the current25%.
Yet a frontier LLM trained through 2025, with or without
retrieval over a current-version corpus, will reliably return
the latter.
Phenomenon: temporal misgrounding.We give a name
to this failure mode:temporal misgrounding. It is the
systematic substitution of the currently in-force version of a
legal article for the version that should be applied given the
temporal context of the question. Temporal misgrounding
has three concurrent root causes:
1.Parametric recency bias.LLMs are trained on snap-
shots that overrepresent the most recent legal state,
biasing their priors toward current law.
2.Absence of temporal conditioning in retrieval.Stan-
dard dense retrievers index a corpus by semantic simi-
larity without conditioning on the date implicit in the
query.
1https://github.com/rosecymbler/
fiscal-fr-bench
1
arXiv:2608.09393v1  [cs.CL]  10 Aug 2026

Temporal Misgrounding in Legal RAG
3.Article-number aliasing.Article numbers persist across
versions (e.g., article 219 of the CGI exists in every ver-
sion of the code), so naive retrieval cannot distinguish
which version is intended.
Research question.We ask:On legal questions whose
date-applicable answer differs from the currently in-force
law, to what extent does temporal misgrounding bottleneck
the accuracy of state-of-the-art LLMs and RAG systems?
We isolate this regime deliberately (§5): it is where temporal
misgrounding is diagnostic, so our reported gaps are con-
ditional on it rather than an average over all legal QA. We
hypothesize that (H) temporal validity is a first-order axis
of legal grounding, and that explicitly conditioning retrieval
on the temporal context of the query closes most of the gap
between LLM-only and version-conditioned performance,
and that an end-to-end retriever, not only an oracle, realizes
this.
Contributions.We support this hypothesis with the fol-
lowing contributions:
•A characterization of temporal misgroundingas a
distinct failure mode of legal RAG, with a taxonomy
of four sub-modes (§3).
•A versioned corpusof 32,436 CGI/LPF article-
versions spanning 93 years (1938–2031), including
future-effective versions, plus an auxiliary dataset of
69,208 version-aware jurisprudence links (98–99% pre-
cision; §4, App. B).
•A benchmarkwhose R3 temporal track is 209 scored,
expert-reviewed, all-model-hard questions across 33
CGI articles (221 released) in a four-regime framework,
scored deterministically via nuggets rather than LLM-
as-judge (§5); R2/R4 released as additional tracks.
•A controlled three-condition experiment(§6) quanti-
fying the LLM-only / static-RAG / version-aware-RAG
accuracy gap on temporal questions.
Positioning.FiscalQA Pro extends the methodology of
OfficeQA Pro (Opsahl-Ong et al., 2026) (deterministic
grounded reasoning over U.S. Treasury Bulletins) to a set-
ting where the underlying documents are revisedin place
rather than appended, and where the date of application
is itself a first-class retrieval signal. Unlike SAT-Graph
RAG (de Martim, 2025), which resolves point-in-time legal
queries through an ontology-driven knowledge graph, our
method needs no ontology or graph construction, only ex-
plicit version indexing and date-conditioned retrieval over
the raw versioned corpus. We focus on tax law because
it maximizes the phenomenon under study: yearly amend-
ments through finance laws, date-sensitive answers (rates,thresholds, deduction rules), and richly cross-referenced
jurisprudence enabling version-aware linking of statutes to
case law.
2. Related Work
Legal benchmarks.LegalBench (Guha et al., 2023) pro-
vides 162 tasks covering U.S. legal reasoning across rule
application, issue spotting, and rhetorical understanding, all
in English and predominantly without temporal indexing.
LEXTREME (Niklaus et al., 2023) extends multi-lingual
coverage with 18 tasks spanning 24 languages across multi-
ple legal systems. LEXam (Fan et al., 2026b) benchmarks
legal reasoning on 340 bilingual (EN/DE) law exams across
multiple jurisdictions, with rubric-based scoring; orthog-
onal to our axis, it evaluates reasoning quality on a fixed
legal state rather than temporal grounding across versions.
JuriBERT (Douka et al., 2021) is a French legal BERT, and
ClaimRAG-LAW (Das et al., 2026) a fine-grained claim-
level legal RAG benchmark. None of these benchmarks
explicitly evaluate temporal drift in legal QA.
Enterprise grounded reasoning.OfficeQA Pro (Opsahl-
Ong et al., 2026) introduced an enterprise benchmark for
end-to-end grounded reasoning over U.S. Treasury Bulletins,
comprising 133 questions across an 89,000-page corpus
spanning nearly a century. Their evaluation is deterministic:
answers are checked against numerical values, citations, and
dates rather than via LLM-as-judge. They show that frontier
LLMs achieve below 5% accuracy on parametric knowledge
alone, 12% with web access, and 34.1% with direct corpus
access (signaling substantial headroom even at the frontier).
Agentic retrieval and nugget-based scoring.KARL
(Databricks AI Research, 2026) introduces KARLBench,
a multi-capability evaluation suite spanning six search
regimes, and proposes anugget-basedmetric: ground-truth
answers are converted into atomic “nuggets,” and a model’s
response is scored by the fraction of nuggets it covers. We
note that nugget scoring as instantiated in KARL remains
an LLM-as-judge variant: an LLM judges whether each
nugget is supported by the response, but one that anchors
the judgment in expert-curated atomic facts rather than free-
form rubrics. They also introduce OAPL, an off-policy
reinforcement-learning procedure for training enterprise
search agents. We adopt their nugget decomposition and
multi-task framing, but remove the judge entirely by restrict-
ing nuggets to deterministically checkable types: article
identifiers matched by regex and numeric values matched
with tolerance (§5.2).
Temporal grounding in NLP.Temporal question answer-
ing is surveyed by Piryani et al. (2025). TempLAMA (Dhin-
gra et al., 2022) and TimeQA (Chen et al., 2021) explore
2

Temporal Misgrounding in Legal RAG
temporal reasoning in factual QA, finding that LLMs sys-
tematically default to the most recent fact seen in train-
ing. Vu et al. (2023) extend this to dynamic open-domain
QA with their FreshQA benchmark. LexTime (Barale,
2025) studies temporal event-ordering in U.S. federal com-
plaints; FiscalQA Pro complements this work by addressing
version-conditioning of statutory retrieval (where the tempo-
ral axis is the law itself rather than the events under adjudi-
cation). Most directly, SAT-Graph RAG (de Martim, 2025)
resolves point-in-time legal queries through an ontology-
driven knowledge graph (contrasted with our graph-free
approach in §1), but evaluates it as a qualitative case study
on the Brazilian Constitution rather than a scored bench-
mark. Two works concurrent with our submission study the
same failure: Fan et al. (2026a) diagnose a training-cutoff
bias and the absence of temporal constraints in legal search
agents (our causes (1)–(2)) over a 13-task RL benchmark,
and Prior et al. (2026) release 312 expert-validated German
statutory QA pairs with version-filtering RAG conditions.
Their independent convergence underscores that temporal
misgrounding is a real, general failure of legal RAG. We
differ on three axes: our scoring is deterministic (regex /
numeric-with-tolerance), whereas Prior et al. (2026) use
an LLM-as-judge that inherits the very temporal bias it
scores (§5.2); our corpus is version-indexed at fine granu-
larity (32,436 versions over 93 years, incl. future-effective);
and our all-model-hard filter spans eleven models. We in-
tendtemporal misgroundingas an umbrella subsuming their
post-cutoff staleness and recency bias.
French legal NLP.Available resources include the open
L´egifrance API (DILA/PISTE) for legislation, Judilibre for
Cour de cassation jurisprudence, and Arianeweb for Conseil
d’´Etat decisions. We build directly on these public sources,
with full provenance preserved at the article-version level.
3. Why Static RAG Fails on Legal QA
Before describing the corpus and benchmark, we articulate
why standard RAG architectures (which dominate current
legal AI deployments) systematically fail on temporally-
grounded legal questions. We argue that the root cause
is structural, not anecdotal: it arises from the conjunction
of three independent properties of (i) the legal corpus, (ii)
typical retrieval architectures, and (iii) LLM parametric
priors.
3.1.Structural Properties of Legal Corpora That Defeat
Static Retrieval
(P1) Same identifier, different content.Article 219 of
the French CGI exists at every point in time, but its sub-
stantive content (e.g., the standard corporate income tax
rate) differs across versions: 331
3%in 2018,31%in 2019,28%in 2020,26.5%in 2021,25%from 2022 onward.
Article identifiers ( cid) are stable, but (article id,
version id) pairs vary. A retriever indexing only one
snapshot of the corpus collapses this fine-grained structure
into a single point in retrieval space.
(P2) Date-dependent correctness.Each version carries
explicit date debut /date fin fields, so a question an-
chored in 2018 has a single correct version, distinct from
2020 or 2024. Unlike encyclopedic corpora, legal correct-
ness is binary and conditional on date: a “related” or “ap-
proximately correct” version is simply wrong.
(P3) Cross-version semantic similarity.Successive ver-
sions of an article are textually near-identical, often differing
by a single rate or threshold. Their dense embeddings are
mutual nearest neighbors under any standard sentence en-
coder, making them nearly impossible to disambiguate by
similarity alone.
3.2. A Taxonomy of Temporal Misgrounding Failure
Modes
The interaction of (P1)–(P3) with current RAG architec-
tures produces four failure modes:current-law substitu-
tion(the retriever returns the in-force version regardless of
the query’s date: the dominant mode, from a current-only
index or recency-weighting);future-law leakage(a not-yet-
in-force VIGUEUR DIFF version is returned for a present-
tense question);wrong-amendment resolution(a query for
an article “in its version prior to law X” yields the current
or an arbitrary historical version; common in jurisprudence,
where courts apply a deprecated version to facts arising un-
der it); andmulti-version confusion(a before/after-reform
comparison collapses to the single highest-scoring version).
We observe the first consistently across frontier LLMs; the
other three require question types outside our single-anchor
R3 set (§9).
3.3. Why This Is Not Solved by Larger or More Recent
LLMs
One might expect newer LLMs, trained on larger legal cor-
pora, to solve temporal misgrounding parametrically. Three
observations argue otherwise. (i) Recency bias is structural:
training data skews toward recently-published, widely-cited
content, overrepresenting current law. (ii) The historical
volume is enormous (30k+ versions across CGI and LPF
alone), and memorizing it does not solve the disambiguation
problem (P3). (iii) Temporally-grounded questions are easy
torecognizebut hard toanswerwithout a version-indexed
corpus, as our parametric knowledge filter shows empiri-
cally (§5; Opsahl-Ong et al., 2026). Temporal grounding
thus requires structural changes to retrieval (version index-
ing, date-conditioning), not bigger models. We formalize
3

Temporal Misgrounding in Legal RAG
the resulting hypotheses in §6, each a retrieval condition we
evaluate.
4. Corpus Construction
We construct the FiscalQA Pro corpus in two stages: (i)
extraction of temporally-versioned tax legislation, and (ii)
version-aware linking of tax-related court decisions to the
relevant article versions. All extraction is fully reproducible
from public sources via the scripts released with this paper.
4.1. Temporally-Versioned Tax Legislation
Source.The CGI (Code g ´en´eral des imp ˆots), its four an-
nexes, and the LPF (Livre des proc ´edures fiscales) are ex-
tracted from the official PISTE L ´egifrance API (DILA): the
/consult/getArticleByCidendpoint returns an ar-
ticle’s complete version history (vs. only the current version
from/consult/getArticle ), each version carrying a
distinct identifier (LEGIARTI...) under a stablecid.
Pipeline.For each LEGITEXT code, we enumerate the
constant identifiers via the table-of-contents endpoint, then
callgetArticleByCid once per cid and upsert every
returned version with its date debut ,date fin,etat
(in force, modified, repealed, future-effective), and full text.
The pipeline is idempotent and fully reproducible from pub-
lic sources.
Output.Our final corpus comprises 32,436 article-
versions across six tax codes (per-code breakdown in
App. A, Table 2). The CGI corpus alone contains 21,040
versions over 3,696 distinct articles, an average of 5.69 his-
torical versions per article (top: 94 versions for article 81,
on tax-exempt income). The temporal span runs from 1938
(LPF) to 2031 (CGI articles whose entry into force has been
deferred by recent finance laws), totaling 93 years.
4.2. Version-Aware Jurisprudence Linking
As a secondary resource, we link tax-related court decisions
to thespecific article versionapplicable at the decision
date: a regex extractor (proximity veto + fiscal-context fil-
ter) resolves each citation against the article index and inter-
sects it with date debut /date fin ranges. This yields
69,208 version-aware links across 32,034 decisions, with
98–99% link-level precision(stratified audit) and 83–93%
decision-level recall on jurisdictional sources. Auxiliary
to the controlled experiment, it serves as weak supervision
for the R1/R4 tracks; full construction and audit are in Ap-
pendix B.5. Benchmark Design
FiscalQA Pro defines four question regimes (R1–R4) in-
spired by KARLBench’s multi-task framing. This paper’s
benchmark and controlled experiment center on theR3
temporal-reasoning track: 209 scored, expert-reviewed,
all-model-hard questions across 33 CGI articles(221 cu-
rated questions are released; twelve are flagged out of the
answerable scope and excluded from scoring: four whose
date-applicable value is an annually INSEE-indexed figure
published administratively, conservatively excluded at cu-
ration, and eight excluded in a full-set review as curation
errors or ill-posed under their date anchor), each paired
with atomic ground-truthnuggetsthat enable deterministic
scoring without LLM-as-judge. Regimes R2 (43 questions)
and R4 (209 questions) are released as additional evaluation
tracks.
5.1. Question Regimes
R1 – Citation Extraction.Given a court decision and
a legal question, identify the CGI/LPF article(s) ground-
ing the court’s reasoning; scored by exact match against
the cited cid (e.g., a 2023 Conseil d’ ´Etat decision →
LEGIARTI000006303451, art. 209 B CGI).
R2 – Deterministic Computation.Apply a tax rate or
rule to a numerical case, scored by exact match (with toler-
ance): e.g., corporate income tax on a e1.2M FY2024 profit
under the standard regime→e300,000 (1.2M×25%).
R3 – Temporal Reasoning (key contribution).Ques-
tions whose correct answer depends on the version of the
law applicable at a specific date. Evaluation: two required
nuggets identifying the article and the exact numerical value
at that date, augmented by a retrieval-side provenance check
(§5.2). This is the regime targeted by our controlled experi-
ment (Section 6).
“What was the standard rate of corporate income
tax applicable to fiscal years opening on or
after 1 January 2018, under the general regime
(excluding SMEs)?”
Expected nuggets: art219CGI (regex
\b219\b ) and 33.333% (numeric with toler-
ance); the applicable version is validated against
retrieval provenance.
We use article 219 here, and in the case study of App. C, for
clarity; thescoredset deliberately targets less-memorized
articles (Table 3, and the parametric filter below).
R4 – Multi-Document Synthesis.Combine CGI, BOFiP
(administrative doctrine), and case law to answer a complex
4

Temporal Misgrounding in Legal RAG
question, scored by nuggets (3–7 per question spanning
article, rate, doctrine reference, and a grounding decision).
5.2. Nugget-Based Scoring
Following Databricks AI Research (2026), we score a re-
sponse rby the fraction of ground-truth nuggets it covers,
score(r, N) =1
|N|P
i1[ni∈r] , where a nugget is an
atomic, deterministically-checkable claim (article identi-
fier, numerical value) and 1[·]is a regex match (identifiers),
numeric-with-tolerance match (values, after arithmetic nor-
malization: FR thousands separators, decimal comma), or
exact string match,neveran LLM judgment. We addition-
ally report astrictscore (both required nuggets hit: correct
article and correct value) and aprovenancescore for re-
trieval conditions: the date-applicable gold version is among
the retrieved versions (the single article for B and Cor, the
top-5 forC prod).
Article-nugget leakage.The article nugget is largely
given away by the question text itself: its regex fires on
thepromptfor 175 of the 209 scored questions (83.7%),
so citing the article is often not evidence of retrieval, and
coverage is dominated by the value nugget. We therefore
also reportvalue-only coverage(the value nugget alone),
which tracks strict accuracy to within 1 point pooled (A
3.0%, B 2.7%, Cor99.2%, Cprod 98.5%; per-model table in
the repository’s stats clustered report.md ); the
headline effect is carried entirely by the date-anchored value.
On the retrieval side, the same fact means Cprod receives
the article number in 83.7% of queries: the article-recall
figures of §7.2 are conditioned on that information being
present, as they would be in practice for a lawyer’s query.
Why nuggets, not LLM-as-judge?Temporal grounding
is precisely the axis on which frontier LLMs share a sys-
tematic bias (§3.3), so an LLM judge inherits the failure
mode it is asked to score: it can accept a fluent answer that
quotes the wrong-date value because that value matches its
own parametric prior. Deterministic nuggets sidestep this
circularity, at the cost of narrower expressivity (a nugget is
atomic, not free-form). We keep expressivity by fixing the
qualitative components (selecting the article, isolating the
discriminating value, choosing a tolerance) at annotation
time, verified against the primary source (the versioned arti-
cle text); at scoring time the check is a regex or a numeric
equality. Disagreement is thus resolvable by re-reading the
corpus rather than re-prompting a judge.
5.3. Parametric Knowledge Filter
Following Opsahl-Ong et al. (2026), we validate that the
benchmark requires grounding (not parametric knowledge)
via a zero-knowledge baseline: each candidate question isput to every evaluated LLMwithoutcorpus or web access,
and any question whose gold value is stated in at least one of
four sampling draws by any model is dropped. We tighten
Opsahl-Ong et al.’s “union over frontier models” criterion
to a union overall eleven evaluated models, i.e., the five
frontier systems, the substitute Gemini entry, and the five
open-weight systems used in Cprod (§6), so the retained set
is all-model-hard rather than only all-frontier-hard.2
Current-version divergence filter.A second, LLM-free
filter enforces the premise of Condition B (§6): a regex
check tests whether the gold date-anchored value still ap-
pears verbatim in thecurrently in-forceversion of the article,
and candidates whose value is unchanged are dropped. On
the final scored set the current text lacks the gold value
for208 of the 209questions (the single exception is kept
as a within-set well-formedness control, §7.2); where an
article has no currently in-force version, divergence holds
trivially and Condition B retrieves nothing. We state the
consequence explicitly:Condition B fails partly by con-
struction: the retained questions are screened so that the
current-version text does not contain the gold value. The fal-
sifiable content of the corresponding hypothesis is therefore
its provenance and ceiling components, not the low strict
score itself (see the reformulatedH 2, §6).
5.4. Question Curation
We are explicit about the manual and automatic components
of the benchmark, a common concern for benchmark papers.
Questions: corpus-surfaced candidates, author-written
questions.The questions are produced by a two-stage
pipeline, both stages released in the repository. First, can-
didate drift points are surfacedautomaticallyfrom the
versioned corpus: r3factory.py scans version histo-
ries for value transitions (a threshold or rate that changes
between consecutive versions) and r3worksheet.py
emits them as a working sheet (article, transition date, be-
fore/after values). Second, every question, date anchor,
canonical answer, and nugget iswritten by the authors, who
draft the question text, verify the value against the corpus
version, and set the tolerance; the full set was then reviewed
for correctness by a qualified French tax professional. A
subset originates from an internal pool (Fiscal-FR-Bench
v1) previously assembled by the authors.3
2On the Qwen substitution between selection and evaluation,
see note‡of Table 1.
3Fiscal-FR-Bench v1 is a proprietary benchmark held by the
authors. We release the R3 temporal track (221 questions, 209 in
scoring scope), together with the R2 (43) and R4 (209) tracks, as
Fiscal-FR-Bench v0 alongside this paper.
5

Temporal Misgrounding in Legal RAG
Selection criteria.The 209 scored questions (after the
all-model-hard parametric filter and the answerable-scope
audit) cover 33 CGI articles across seven fiscal sub-domains
(Table 3): personal income tax (IR), local business and
housing taxes (IFER, CFE, TH, TSE), betting and gaming,
indirect taxes (V AT, tobacco, TV , mining), BIC/agricultural
and cross-border, wage tax, and wealth tax (ISF). Each
references an article whose content changed measurably
over 1989–2025 (statutory drift). Consistent with the fil-
ter, we targetobscure, non-roundedparameters (indexed
allowances, thresholds, per-installation IFER tariffs) rather
than well-known headline rates (e.g., the corporate rate of
art. 219 or the income-tax scale of art. 197) that frontier
LLMs reliably memorize.
Frozen evaluation set.The 209-question scored set is
frozen prior to any retriever or reranker development
(killer qids v2.txt ); the answerable-scope audit
flags twelve released questions as out of scope by a model-
independent criterion — four (art. 199 undecies A, whose
per-square-metre ceiling is revalued annually by statutory
INSEE indexation and published via BOFiP; conservatively
excluded at curation) plus eight caught in full-set review (cu-
ration errors or items ill-posed under their date anchor; item-
ized in the repository’s excluded qids scope.txt )
— leaving k=209 scored out of 221 released. The fiscal
reranker we additionally evaluated is fit only on articles
disjointfrom those of this set; it yields no top-1 gain and is
not part of the reportedC prod pipeline.
Nuggets and answers: fully manual.Each question is
paired with two atomic required nuggets, manually anno-
tated by the authors: a regex on the applicable article num-
ber and a numeric-with-tolerance match on the exact date-
applicable value. Because the value nugget is theexact
date-applicablefigure (numeric-with-tolerance), strict ac-
curacy already requires the correct-date value, not merely
a plausible article; version-selection quality is trackedsep-
aratelyby the retrieval-side provenance score (§5.2), so
the two metrics are decoupled and neither inflates the other.
The canonical answer is cross-checked verbatim against the
version the nuggets point to (all 221 curated questions veri-
fied against the corpus) and reviewed for correctness by a
qualified French tax professional. The jurisprudence links
(§4.2) serve only as weak supervision for R1/R4, never as
ground truth for R3.
Parametric filter, in practice.At selection time, no eval-
uated model recovered its date-applicable answer in any
of four sampling draws; this is the sense in which the set
is all-model-hard. At evaluation time, mean Condition A
strict accuracy is 3.0% across the eleven models (cluster-
bootstrap 95% CI [1.4, 4.7]; per-model values in Table 1).
Residual variance stems from the Anthropic frontier models’default sampling temperature (genuinely diversified draws),
the GPT-5.x reasoning stack’s non-determinism even at tem-
perature 0, and the remaining entries’ effectively single
deterministic draws at temperature 0, which make the filter
conservative rather than lenient (a memorized answer would
appear identically in all four draws and be caught).
Residual leakage.Selection-time hardness does not
freeze evaluation-time behavior: across the eleven models,
at least one model produced the goldvaluein Condition A
for 37 of the 209 questions (17.7%; per-model counts 0–25,
highest on GPT-5.5 (25) and Opus 4.7 (16), reflected in
their higher Cond A strict in Table 1): sampling drift and
value reconstruction, not memorization (a value memorized
identically across all four draws would have been caught at
selection). The set is thus all-model-hard under the models’
evaluation configurations, and the B →C comparison is unaf-
fected, as all conditions share the configuration. Per-model
counts:stats clustered report.md(§4).
Why 209 questions?The original R3 track was deliber-
ately small ( k=35 ; cf. LegalBench’s 162 tasks, Guha et al.,
2023), designed to isolate the temporal-misgrounding phe-
nomenon at minimal annotation cost. To respond to reviewer
feedback on scale and to make room for eleven models (fron-
tier + open-weight), we extended the track to 221 curated
questions over the same annotation protocol (209 in scoring
scope after the answerable-scope audit), spanning 33 articles
instead of 10. Each additional question follows the same cu-
ration protocol (selection, version identification, canonical
answer, two nuggets, corpus cross-validation) and survives
the tightened all-model-hard filter. We now read Table 1 at
both the condition-gap level (A/B ≪C, unchanged from
the original submission) and the model-ranking level (fron-
tier vs. open-weight), which the larger kmakes statistically
well-powered (§7.2).
The per-article and per-sub-domain distribution of the
scored set is given in Table 3 (Appendix A).
6. Experiments
We run acontrolled three-condition experimentmea-
suring the accuracy gap between standard LLM/RAG ap-
proaches and our temporally-versioned retrieval on R3 ques-
tions.
6.1. Controlled Experiment: Temporal Drift
Design.We evaluate three retrieval conditions on the
frozen all-model-hard R3 subset ( k=209 scored; Table 3),
with eleven answer models: five frontier closed-API sys-
tems (Claude Opus 4.7, Claude Opus 4.8, Claude Sonnet 4.6,
GPT-5.4, GPT-5.5), plus Gemini 2.5 Pro as a substitute fron-
tier entry (Gemini 3 Pro Preview being rate-limited during
6

Temporal Misgrounding in Legal RAG
the evaluation window; note †of Table 1), and five open-
weight systems: Mistral Large 2407, Llama 4 Maverick,
Qwen 3 235B, Gemma 3 27B, and GLM 5.2. The frontier
set includes two Opus and two GPT-5 generations to test
whether within-family generation gains close the temporal-
misgrounding gap parametrically (they do not; §7). The five
open-weight systems, run via public inference endpoints
(OpenRouter and Together AI), let us test whether temporal
grounding depends on model scale or provider (it does not).
1.Condition A: LLM-only.The model answers using
parametric knowledge only: no retrieval, no web. A sin-
gle evaluation draw per question; the four-draw prob-
ing applies to the selection-time parametric filter (§5;
sampling configurations in the caption of Table 1).
2.Condition B: RAG over current corpus.The model
retrieves over the static, current-version-only CGI cor-
pus (representative of the majority of deployed legal
RAG systems today). Article retrieval uses the gold
cid; only thecurrentin-force version is exposed. Be-
cause B is handed the gold cid, it is charitable to the
static baseline, so the reported A/B ≪C gap is a lower
bound.
3.Condition C: RAG over our versioned corpus.The
model retrieves over our temporally-versioned corpus,
where the version layer returns the article version ap-
plicable at the date in the question. We evaluate two
retrievers that share this version layer:
•Cor(oracle):article retrieval uses the gold cid,
isolating the version-selection contribution (the
ceiling once the right article is found).
•Cprod (end-to-end):an end-to-end retriever
findsboththe article and the version, with
no oracle: the realistic deployment setting.
The dense channel is a domain-adapted dense
encoder over a chunked,multi-versionarti-
cle index (three versions per cid: first, me-
dian, last by date debut , excluding stillborn
MODIFIE MORT NEartefacts); the sparse chan-
nel is BM25 over the same chunks; the two are
fused by reciprocal-rank fusion (no cross-encoder
rerank in Cprod). The top-5 unique articles are
handed to the version layer, which resolves the
date-applicable version per cid before prompting
the LLM.
Metric.Per-question metrics (coverage,strict,prove-
nance) are as defined in §5.2; per-condition scores are means
over the k=209 scored questions, either per model or pooled
over the eleven models.Hypotheses.We test four falsifiable predictions, each tied
to a condition: ( H1) A is low on strict accuracy (near-absent
parametric knowledge of date-specific values), uniformly
across scales and providers, though coverage stays non-
trivial since article-id nuggets are easy; ( H2) B retrieves
the date-applicable version 0%of the time and stays below
10% strict. The near-zerostrictscore is largely aconstruc-
tion check, since the current-version divergence filter (§5)
screens retained questions so the current text does not con-
tain the gold value; thefalsifiablecontent of H2is the 0%
provenance (structural: a single-version index cannot hold
the version) and the <10% ceiling, which a model could
in principle beat by re-deriving the historical value from
the current text (e.g., un-indexing a revalorized threshold);
(H3)Corcloses most of the gap ( >80% strict, 100% prove-
nance), showing version selection (not corpus completeness
or model size) is the bottleneck; ( H4) a realistic retriever
feeding the top-5 date-applicable versions recovers essen-
tially the Corceiling, the residual gap being a recall@5
ceiling on a few niche queries, so the lever is recall, not
top-1 reranking.
7. Results
7.1. Corpus Quality
Versioning depth.Table 2 reports the distribution of
article-versions per code: a mean of 4.02 versions per cid,
with the CGI principal article 81 (tax-exempt income), re-
peatedly amended by finance laws, holding 94 distinct his-
torical versions. This depth, the very structure a static index
collapses (§3.1), is what the controlled experiment exploits.
Quality of the auxiliary jurisprudence-linking dataset (preci-
sion, recall, gold-standard audit) is reported in Appendix B.
7.2. Controlled Experiment Results
On the k=209 all-model-hard subset, parametric knowl-
edge (A) is uniformly low across all eleven models (3.0%
mean strict, cluster-bootstrap 95% CI [1.4, 4.7]; Table 1),
by construction of the filter, tightened from “no frontier
model recovers” (Opsahl-Ong et al., 2026) to “no evaluated
model recovered across four sampling draws at selection
time”, which excludes any question a model happened to
memorize (residual evaluation-time drift is quantified in
§5). Static-corpus RAG (B), the setting deployed by most
legal-AI products today, does not improve on the LLM-only
baseline (2.7% mean strict, cluster-bootstrap 95% CI [1.3,
4.8], statistically indistinguishable from A) and retrieves the
date-applicable version in 0% of cases: it fails not silently
butconfidently, grounding on a real, well-formed, but inap-
plicable version (both are construction checks rather than
surprising measurements; §5 andH 2, §6).
7

Temporal Misgrounding in Legal RAG
Coverage Strict Provenance
Model A BC orCprod A BC orCprod BC orCprod
Frontier (closed API)
Opus 4.7 53.8 55.3 100.0 99.8 7.7 10.5100.0 99.50 100 99
Opus 4.8 53.3 51.4 100.0 99.5 6.7 3.3100.0 99.00 100 99
Sonnet 4.6 49.5 50.0 100.0 99.5 1.0 1.4100.0 99.00 100 99
GPT-5.4 50.7 51.2 100.0 99.5 1.4 2.4100.0 99.00 100 99
GPT-5.5 56.0 54.1 100.0 99.8 12.0 8.1100.0 99.50 100 99
Gemini 2.5 Pro†46.2 49.5 99.0 99.0 1.4 0.599.0 98.60 100 99
Open-weight
Mistral Large 2407 50.2 50.7 99.8 99.0 0.5 1.499.5 98.10 100 99
Llama 4 Maverick 48.1 50.2 99.0 98.6 0.0 0.598.1 97.10 100 99
Qwen 3 235B‡46.7 49.3 97.4 98.6 1.0 0.594.7 97.10 100 99
Gemma 3 27B 42.8 48.6 99.8 97.4 0.0 0.599.5 95.20 100 99
GLM 5.2 49.0 50.2 99.8 99.5 1.0 1.099.5 99.00 100 99
Mean (11 models) 49.7 51.0 99.5 99.1 3.0 2.799.1 98.30 100 99
Table 1.Results on the k=209 scored all-model-hard R3 subset (questions no evaluated model, frontier or open, answered from parametric
knowledge across four sampling draws at selection time; 221 curated questions are released, twelve flagged out of the answerable scope).
A: LLM-only, a single evaluation draw per question (the three Anthropic frontier models use their default sampling temperature; GPT-5.x’s
reasoning stack is non-deterministic even at temperature 0; the remaining six entries (five open-weight and Gemini 2.5 Pro) run at
temperature 0, effectively deterministic).B: RAG over the static current-version corpus (gold article, current version). Cor: oracle version
selection (gold article, date-applicable version): the single-article ceiling. Cprod: end-to-end retriever feeding thetop-5date-applicable
versions, no oracle (retriever details in §6). Coverage = mean nugget fraction; Strict = both required nuggets hit (correct article and
exact value); Provenance = % of questions whose date-applicable gold version is among the retrieved top-5. All conditions share an
identical labeled-article prompt at an 8,000-char/article query-windowed budget (value-preserving: the date-applicable gold value lies
within the served window for all 209 scored questions), so CorandCprod differ only in the number of retrieved articles. All numbers are
percentages; k=209 ; cluster-aware 95% bootstrap CIs (articles resampled) and per-model exact McNemar/Wilcoxon tests are reported
in §7.2.†Gemini 3 Pro Preview was rate-limited during the extended evaluation window; we report Google’s next-tier available model
(Gemini 2.5 Pro) as a substitute frontier entry. Conditions A/B/ Corran on the native AI Studio endpoint; Cprod ran via OpenRouter after
the native per-day quota was exhausted. Its mandatory reasoning stack is non-configurable, so Cond A is a single reasoning-conditioned
draw. The Gemini 2.5 Pro and GLM 5.2 entries include a small number of provider-side empty responses scored as failures (22 in
total across conditions, a conservative direction; audit in the repository, parametric filter empty audit.md ).‡Qwen 2.5 72B
(used at selection time for the parametric filter) was retired from serverless inference by every accessible provider during the evaluation
window; we substitute Qwen 3 235B (Together AI). A larger, more recent model could in principle recovermorequestions parametrically;
empirically its Cond A strict is 0.0%, so the probe remains all-model-hard.
Control condition (well-formedness).Two same-set con-
trols rule out ill-posed questions or a broken pipeline as
an explanation for B’s failure. First, under theidentical
labeled-article prompt and pipeline, oracle version selection
(Cor, which changes only the servedversionof the same ar-
ticle) reaches99.1%mean strict on the same 209 questions
(Table 1): the questions are answerable and the model ex-
tracts the value correctlywhen handed the date-applicable
version, so B’s near-zero score is not ill-posedness or a read-
ing failure. Second, the current-version text lacks the gold
date-anchored value for208 of the 209scored questions
(the divergence premise of Condition B, §5); on the single
non-drifted exception, whose valueispresent in the current
text, static B retrieves it correctly. B’s failure on the drifted
set is therefore version drift, not a broken pipeline.
The operative result: end-to-end retrieval. Cprod, the
end-to-end retriever, is given no oracle (it must locate both
the article and its date-applicable version from the query
alone) and reaches98.3% mean strict(cluster-bootstrap95% CI [95.9, 99.6]); all eleven models cross 95% in point
estimates (95.2–99.5%; every per-model clustered CI stays
above 89%, §7.2), and the gold version is in the top-5 for
99% of questions (vs. 0% for B). Holding the versions
is necessary but not sufficient: an oracle-article ablation
(Cor, gold cid, leaving only version selection) reaches
99.1% mean (CI [98.4, 99.9]) at 100% provenance. The
Cor→C prod gap is only 0.8 points, concentrated onfirst-
stage recallrather than version selection: the sole prove-
nance miss is art. 1417 (2 questions whose date-applicable
version falls outside the retrieved top-5). Once the corpus is
versioned the bottleneck is article recall, not date resolution;
a cross-encoder reranker adds nothing on this niche set.
Significance (cluster-aware).The protocol is paired, with
two dependencies a pooled iid analysis would ignore:
the eleven models answer thesamequestions, and ques-
tions cluster by article (33 clusters; 34 on art. 1466 A
alone). We therefore treat themodelas the unit of in-
ference and bootstrap by resamplingarticles, not ques-
8

Temporal Misgrounding in Legal RAG
tions (10,000 iterations, seed 42). The B →C orand
B→C prod gains are significantfor every model individ-
ually(exact two-sided McNemar on per-question strict
outcomes, 11 tests, all p <10−55), so the conclusion
does not depend on pooling; per-model Wilcoxon tests on
coverage agree, while the Cor→C prod coverage differ-
ence is small and not uniformly significant, consistent with
the 1% recall@5 ceiling. Full per-model tests are in the
repository’s stats clustered report.md . Cluster-
bootstrap 95% CIs for pooled strict: A 3.0 [1.4, 4.7], B
2.7 [1.3, 4.8], Cor99.1 [98.4, 99.9], Cprod 98.3 [95.9, 99.6];
every per-model Cprod lower bound stays above 89% (the
claim holds under clustering, not only as a point estimate).
Leave-one-article-out on pooled Cprod strict spans 98.1%
(dropping art. 1519 A) to 99.2% (dropping art. 1417): no sin-
gle article carries the result. A single-question walkthrough
(art. 219 CGI) is in App. C (Fig. 1).
Windowing policy.All conditions serve avalue-
preserving, query-windowedextract at an 8,000-char/article
budget: for each retrieved article we keep the head plus the
passages most relevant to the query, capped at 8,000 char-
acters. A pre-scoring check confirms the date-applicable
gold value lies within the served window for all 209 scored
questions, so no condition is handed an extract that cannot
contain the answer, and the budget affects B, Cor, andCprod
symmetrically. This resolves the head-truncation artifact
of the original 6,000-char cap, under which the gold value
of a handful of long articles (e.g., arts. 158, 1605 bis, 261,
83) fell beyond the window. Uniform full-text prompting
is not preferable: it degrades Coron long articles through
long-context distraction (ablation scripts in the repository).
Failure-mode distribution.Condition B retrieves the cur-
rent version in 100% of cases (Table 1, provenance), so
100% of B-condition errors are current-law substitution,
the dominant mode by construction of a static index.
8. Conclusion
We identifiedtemporal misgrounding, the systematic re-
trieval and citation of the currently-in-force version of a
legal article when the question requires an earlier or fu-
ture version, as a structural failure mode of legal RAG:
stable identifiers, date-dependent correctness, and cross-
version semantic similarity meeting single-version index-
ing, no date-conditioning, and parametric recency bias. To
quantify it we builtFiscalQA Pro: a temporally-versioned
corpus of 32,436 CGI/LPF article-versions over 93 years
(plus 69,208 auxiliary version-aware jurisprudence links)
and an R3 temporal-reasoning track of 209 scored, expert-
reviewed, all-model-hard questions across 33 CGI articles
(221 released), scored deterministically via nuggets.Across eleven models, our controlled experiment shows
LLMs fail on temporally-grounded questions with or with-
out static-corpus RAG (3.0% and 2.7% pooled mean strict;
static RAG retrieves the date-applicable version 0% of the
time), while our end-to-end retriever, conditioning retrieval
on the query date with no oracle, reaches 98.3% mean strict
(all eleven models cross 95% in point estimates), an oracle-
article ablation placing the ceiling at 99.1% and the resid-
ual 0.8-point gap in first-stage recall rather than version
selection. We argue legal QA should be reframed as a
temporally-indexed retrieval problem, and release the cor-
pus, benchmark, model responses, and pipeline code (the
fine-tuned encoder weights excepted; Appendix D).
9. Limitations and Future Work
Scope and civil-law generalization.FiscalQA Pro covers
French tax law only (CGI, its four annexes, and the LPF).
We argue, however, that temporal misgrounding isstructural
to civil-law statutory retrievalrather than a French artifact:
any corpus amended in place and versioned over time ex-
hibits the same conjunction of properties (§3), and several
jurisdictions expose versioned statutory APIs with explicit
validity dates analogous to L ´egifrance (Fedlex, Gesetze-
im-Internet, Justel, L ´egilux; Swiss Federal Chancellery;
German Federal Ministry of Justice; Belgian Federal Public
Service Justice; Government of Luxembourg). Our method
layer is jurisdiction-agnostic; only the data layer is. The con-
current German study of Prior et al. (2026) finds the same
phenomenon—independent evidence of generalization; a
Swiss Fedlex replication is the next step.
Question scale.The R3 track contains 221 curated ques-
tions (209 in scoring scope), up from 35 in the original
submission (§5). The extended set is still smaller than
LegalBench (162 tasks; Guha et al., 2023) or KARLBench
(2,000+ questions): a deliberate trade against synthetic or
LLM-authored scaling, since concentrated, expert-reviewed,
all-model-hard difficulty is the right regime for measuring
temporal misgrounding (cf. OfficeQA Pro’s 133 questions).
Failure modes evaluated.Our single-anchor R3 set
exercises onlycurrent-law substitution; the other
three taxonomy modes (§3.2) require future-effective
(VIGUEUR DIFF ), “in its version prior to law X,” and two-
version-comparison questions, which we leave to v1; the
corpus’s VIGUEUR DIFF versions, extending to 2031, al-
ready enable the future-law-leakage mode via questions
aboutfuturelegal states fixed by enacted-but-deferred legis-
lation. Finally, beyond “apply the law”: R1–R3 target rule
application, notlegal interpretation; R4 is a first step toward
interpretive benchmarks.
9

Temporal Misgrounding in Legal RAG
Impact Statement
This work studies a reliability failure of legal AI, temporal
misgrounding, and provides a benchmark and methodology
to measure and mitigate it. Improving the temporal correct-
ness of legal question answering can reduce confidently-
wrong outputs in high-stakes settings (tax compliance, legal
research). The corpus and benchmark are built entirely from
public legislation and case law; no personal data is intro-
duced. As with any legal-AI tool, outputs should be verified
by a qualified professional and not treated as legal advice.
References
Barale, C. LexTime: A benchmark for temporal ordering
of legal events. InFindings of EMNLP, 2025. URL
https://arxiv.org/abs/2506.04041.
Belgian Federal Public Service Justice. Justel: Consolidated
Belgian legislation. URL https://www.ejustice.
just.fgov.be/. Accessed July 2026.
Chen, W., Wang, X., and Wang, W. Y . TimeQA: A bench-
mark for time-sensitive question answering. InNeurIPS
Datasets and Benchmarks Track, 2021.
Das, S., Abualhaija, S., and Bianculli, D. Fine-grained
claim-level RAG benchmark for law.arXiv preprint
arXiv:2605.21071, 2026.
Databricks AI Research. KARL: Knowledge agents via re-
inforcement learning.arXiv preprint arXiv:2603.05218,
2026. URL https://arxiv.org/abs/2603.
05218.
de Martim, H. An ontology-driven graph RAG for le-
gal norms: A structural, temporal, and deterministic
approach. InLegal Knowledge and Information Sys-
tems (JURIX 2025), Frontiers in Artificial Intelligence
and Applications, pp. 282–287. IOS Press, 2025. doi:
10.3233/FAIA251598.
Dhingra, B., Cole, J. R., Eisenschlos, J. M., Gillick, D.,
Eisenstein, J., and Cohen, W. W. TempLAMA: Time-
aware language models as temporal knowledge bases.
Transactions of the Association for Computational Lin-
guistics, 2022.
Douka, S., Abdine, H., Vazirgiannis, M., El Hamdani,
R., and Restrepo Amariles, D. JuriBERT: A masked-
language model adaptation for French legal text. InPro-
ceedings of the Natural Legal Language Processing Work-
shop, 2021.
Fan, W., Zhou, Y ., Zhang, M., Weng, Y ., Hu, Y ., Zheng, T.,
Xu, B., Li, C., Yang, J., Li, H., and Song, Y . Can LLMs
time travel? Enhancing temporal consistency in legalagentic search through reinforcement learning.arXiv
preprint arXiv:2605.25920, 2026a.
Fan, Y ., Ni, J., Huang, Y ., Tian, Y ., Stammbach, D., Ash,
E., Engel, C., et al. LEXam: Benchmarking legal rea-
soning on 340 law exams. InInternational Confer-
ence on Learning Representations (ICLR), 2026b. URL
https://arxiv.org/abs/2505.12864.
German Federal Ministry of Justice. Gesetze im Inter-
net: German federal law online. URL https://www.
gesetze-im-internet.de/ . Accessed July 2026.
Government of Luxembourg. L ´egilux: The official journal
of the Grand Duchy of Luxembourg. URL https://
legilux.public.lu/. Accessed July 2026.
Guha, N., Nyarko, J., Ho, D. E., R ´e, C., et al. LegalBench:
A collaboratively built benchmark for measuring legal
reasoning in large language models. InNeurIPS Datasets
and Benchmarks Track, 2023.
Niklaus, J., Matoshi, V ., Rani, P., Galassi, A., St ¨urmer,
M., and Chalkidis, I. LEXTREME: A multi-lingual and
multi-task benchmark for the legal domain. InFindings
of EMNLP, 2023.
Opsahl-Ong, K., Singhvi, A., Collins, J., Zhou, I., Wang,
C., Baheti, A., Oertell, O., Portes, J., Havens, S., Elsen,
E., Bendersky, M., Zaharia, M., and Chen, X. OfficeQA
Pro: An enterprise benchmark for end-to-end grounded
reasoning.arXiv preprint arXiv:2603.08655, 2026. URL
https://arxiv.org/abs/2603.08655.
Piryani, B., Abdallah, A., Mozafari, J., Anand, A., and
Jatowt, A. It’s high time: A survey of temporal question
answering.arXiv preprint arXiv:2505.20243, 2025.
Prior, M., Schultz, A., and Grabmair, M. Asking for an
old friend: Diagnosing and mitigating temporal failure
modes in LLM-based statutory question answering.arXiv
preprint arXiv:2605.23497, 2026.
Swiss Federal Chancellery. Fedlex: The publication plat-
form for Swiss federal law. URL https://www.
fedlex.admin.ch/. Accessed July 2026.
Vu, T., Iyyer, M., Wang, X., Constant, N., et al. FreshLLMs:
Refreshing large language models with search engine
augmentation.arXiv preprint arXiv:2310.03214, 2023.
A. Corpus and Benchmark Statistics
Table 2 breaks the corpus down by code: versions, distinct
articles ( cid), and average versioning depth. Table 3 gives
the distribution of the k=209 scored R3 questions across
CGI articles and fiscal sub-domains.
10

Temporal Misgrounding in Legal RAG
Two endpoint caveats on the 93-year span. The 1938 origin
is carried by a single LPF article-version (art. L28) whose
date debut records the entry into force of the predeces-
sor provision it codifies (d ´ecret-loi of 1 June 1938); the
LPF itself dates from 1981–82. The 2031 endpoint excludes
20 rows carrying L ´egifrance placeholder dates (2222-02-22,
2999-01-01), which the version-selection layer never serves.
Code Versions CIDs V/CID
CGI principal 21,040 3,696 5.69
CGI annexe III 3,801 1,467 2.59
LPF 3,031 1,095 2.77
CGI annexe II 2,390 1,005 2.38
CGI annexe IV 1,879 705 2.67
CGI annexe I 295 108 2.73
Total 32,436 8,076 4.02
Table 2.Tax legislation corpus statistics. V/CID = average number
of versions per article (constant identifier).
Sub-domain Principal arts. (# Q) # Span
Local business & housing
taxes (IFER, CFE, TH,
TSE)1466 A (34),
1519 A (20),
1586 nonies (17),
1519 HA (16),
1647 D (4),
1414 A (4),
1609 C (4), 1417 (2),
1414 B (1)102 1990–2024
Personal income tax (IR) 156 (16), 196 B (5),
168 (5), 158 (4),
157 bis (4),
204 H (4), 200 (2),
199 decies H (2),
83 (2),
199 sexies (1),
5 (1)46 1989–2024
Betting & gaming 302 bis ZI (16),
302 bis ZG (9)25 2012–2025
Indirect taxes (V AT, to-
bacco, TV , mining)568 (7), 1587 (7),
261 (3),
302 bis ZC (3),
1605 bis (3)23 1996–2025
BIC, agricultural & cross-
border50-0 (3), 182 A (1),
1679 A (1), 73 (1)6 1991–2025
Wage tax 231 (5) 5 2002–2019
Wealth tax (ISF) 885 H (2) 2 2010–2014
Total 33 articles, 7 sub-
domains209 1989–2025
Table 3.Distribution of the k=209 scored all-model-hard
temporal-reasoning (R3) questions across CGI articles and fiscal
sub-domains (the twelve released questions flagged out of the an-
swerable scope—art. 199 undecies A and eight review exclusions—
are not shown). The set spans seven tax sub-domains and 36 years
of statutory drift (1989–2025), including the franc-to-euro tran-
sition. Articles are listed in decreasing order of question-count
within each sub-domain (parenthetical count).B. Version-Aware Jurisprudence Linking
Alongside the versioned legislation, we release a dataset
linking tax-related court decisions to thespecific article
versionapplicable at the date of decision. This resource
is auxiliary to the temporal-misgrounding study (it is not
used by the controlled experiment of §6); we provide its full
construction and quality analysis here.
B.1. Construction
Source decisions.We work with four sources of
French tax-related case law: arianeweb decisions
(Conseil d’ ´Etat, 51,564 decisions), inca (Cour de
cassation, 300,383), judilibre decisions (Cour
de cassation with structured visa, 265,891), and
decisions unified (an aggregation of the above plus
EU and sectoral authorities, 663,748). To avoid double-
counting between the aggregation and its underlying sources,
we report metrics by individual source. We restrict
each source to a fiscal subset using either a PostgreSQL
tsvector query (for decisions unified ) or pattern
matches on tax-relevant terms (for the others); the resulting
subsets total 61,835 decisions.
Citation extraction and version selection.For each fis-
cal decision, we apply a regular-expression extractor sen-
sitive to French tax-citation conventions: numbered arti-
cles (e.g., article 209 B ), LPF prefixes ( L. 16 B ,
R. 281-1 ), and Latin ordinals ( bis,ter,quater , . . . ).
Candidates are filtered by a two-tier disambiguation: (i) a
proximity veto rejecting matches whose immediate ( ±70
chars) right-context attaches them to another source (e.g.,
du Code civil ,de la loi n° X ), and (ii) a fiscal-
context requirement in a ±100 -char window.4Each surviv-
ing candidate is resolved against an in-memory article in-
dex; we then select the articleversionvalid at the decision’s
date by intersecting against date debut anddate fin
ranges. Where structured visa fields are available (Judili-
bre), we tag the link as VISE (formally cited as legal basis);
all other links are taggedCITE.
Output.Linking produces 69,208 version-aware links
across 32,034 distinct decisions. Of these, 97% point to
a version whose validity period strictly includes the date
of decision; the rest fall back to the most recently active
version at decision time. The linked decisions span five
decades (the 2000s most represented; Table 4), enabling
longitudinal evaluation. The decade distribution and the
validity-inclusion rate are computed on the underlying deci-
sion tables, reconstructible from the public sources via the
4For ambiguous one- to three-digit numbers, we require a
strongfiscal context (explicit mention of CGI/LPF). For numbers
with suffix or LPF prefix, a weaker fiscal context suffices.
11

Temporal Misgrounding in Legal RAG
Decade Decisions Links
Before 1990 7,629 15,953
1990–1999 6,995 13,774
2000–2009 7,679 17,182
2010–2019 4,991 10,588
2020–2026 4,740 11,711
Total 32,034 69,208
Table 4.Distribution of version-aware jurisprudence links and
linked decisions by decade. Each decision is bucketed by its
decision date where available, else by the earliest applicable-
version start date (for the 25% of links from aggregated
decisions unified records whose decision-date field is un-
populated in this snapshot); totals match the 69,208 links across
32,034 decisions.
released extraction scripts; the released linking dataset ships
the resolved links themselves.
B.2. Quality Audit
Precision.We sample 100 links proportionally stratified
by source (seed=42, reproducible) and audit them with a
two-stage auto-annotator followed by manual review of
low-confidence cases. The auto-annotator first checks the
immediate right-context ( ±60 chars after the article num-
ber) for explicit CGI/LPF mention (high-confidence true
positive) or competing source attachment (high-confidence
false positive); cases with neither signal are checked in a
wider±300 -char window. We measure98–99% precision
on this stratified sample. The single residual false positive
is a header-extraction artifact (an article number in a court-
arrˆet metadata header without fiscal context). Precision is
uniform across sources (Conseil d’ ´Etat: 100%, Cour de
cassation: 92%, Judilibre: 100%, decisions unified :
99%) and across article-number types (alphabetic-suffix,
Latin-ordinal, short-numeric, LPF-prefixed: all≥97%).
Failure modes.The remaining 1–2% of false positives
fall into four patterns: (a) cross-code number collisions
(e.g., article 1382 of the Civil Code vs. the CGI), (b)
attachment to a numbered law or convention rather than a
code, (c) extraction from non-substantive metadata (decision
headers), and (d) ambiguous coreferences such as “le m ˆeme
code.” Patterns (a) and (b) are largely addressed by the
proximity veto; (c) and (d) motivate the NER-based fine-
tuning discussed below (Limitations and next step).
Cross-validation against Judilibre visa.Of the 907 fis-
cal Judilibre decisions, 843 (93%) receive at least one link
(decision-level recall). On the 485 decisions whose visa is
parseable as referring to CGI/LPF, article-level recall on
visa-matched articles is 37.6% after fuzzy stem matching.
The remaining gap is partially explained by (i) coarser visa
labels (e.g., L16) vs. finer textual citations ( L. 16 B ),and (ii) articles cited in the decision’s reasoning that are not
formally part of the visa.
Manual gold-standard recall.All CGI/LPF articles cited
in a stratified random sample of 50 fiscal decisions were ex-
haustively annotated (read on full text, not the truncated
extract), yielding 116 article references across 34 deci-
sions (16 decisions cite no nominative CGI/LPF article).
Comparing the pipeline’s links against this ground truth,
with article-identity normalization applied symmetrically
to both sides (alphabetic suffixes and Latin ordinals kept;
paragraph/alinea markers stripped), we measure88.6%
article-level precision,53.4% article-level recall, and97%
decision-level recall(33/34 decisions receive at least one
correct link). Recall is highest on structured jurisdictional
sources (Judilibre 71%, decisions unified 55%, Ar-
ianeweb 53%) and lowest on inca (48%). The residual
false-negatives concentrate in (i) citations appearing deep
in long decisions (20–45k characters, beyond the span pro-
cessed by the extractor) and (ii) LPF procedural articles
(L./R.series); both motivate the NER-based extractor be-
low. This article-level measurement complements the link-
level precision audit above: the pipeline findsat least one
grounding article in nearly every decision, and about half of
allcited articles.
Limitations and next step.Recall is uneven: 83–93%
on the three jurisdictional sources but only 39% on the
noisier aggregated decisions unified table, with 1–
2% residual false positives (cross-code collisions, header
artifacts, ambiguous coreferences). A fine-tuned legal NER
extractor, weakly supervised by the 69,208 regex links plus
the 50-decision gold standard, is the natural next step to
close most of this gap.
C. Case Study: Temporal Misgrounding on
Article 219 CGI
Figure 1 walks a single R3 question end-to-end
through the three conditions. The question asks
for the standard corporate income tax rate for fis-
cal years opening on or after 1 January 2018 (gold:
art. 219 CGI, version LEGIARTI000036431672
valid 2018-01-01, “. . . fix ´e `a 33 1/3 %”;
nuggets {art 219CGI, 33 onethird pct,
version 2018, rate general regime} ). LLM-
only (A) and static-corpus RAG (B) both answer “25%”
(2/4 nuggets): A from parametric recency bias, B by
grounding on areal but inapplicablecurrent consolidated
version (2026 consolidation; the 25% rate itself has been
unchanged since 2022): the most insidious mode, since
provenance is preserved but correctness is not. Conditioning
retrieval on the question’s date (C) returns the 2018 version
and yields the correct “ 331
3%” (4/4). Same LLM, same
12

Temporal Misgrounding in Legal RAG
Q:standard corporate income tax rate for
fiscal years opening on/after 1 Jan 2018?
(gold: 331
3%, art. 219 CGI)
ALLM-only (no retrieval)25%×
Bstatic RAG: art. 219v. 2026“25 %”25%×
confidently wrong
Cversioned (ours): art. 219v. 2018“33 1/3 %”331
3%✓
Figure 1.The three retrieval conditions on a temporal-drift question
(art. 219 CGI, fiscal year 2018).Bretrieves a real but inapplicable
version, the current consolidated version (2026 consolidation; its
25% rate has been unchanged since 2022), and isconfidently
wrong;Cconditions retrieval on the question’s date, returns the
2018 version, and grounds the correct answer. Same LLM, same
corpus; only version selection differs.
corpus, only version selection differs, and the gap moves
from 50% to 100% on this question. (We use art. 219
here only for clarity; thescoredset targets less-memorized
articles, Table 3.)
D. Reproducibility
We list explicitly what is released.5Data (CC-BY-4.0, with
L´egifrance/Judilibre–Etalab 2.0 attribution; SHA256
checksums included):the versioned corpus (32,436 article-
versions, Parquet); the linking dataset (69,208 version-aware
citations); the benchmark (R3, 221 questions with 209 in
scoring scope, plus the R2/R4 tracks, with nuggets and
ground-truth answers); the 100-link stratified audit sam-
ple and the 50-decision gold standard; all model responses
behind Table 1 (the query-windowing ablation of §7.2 is
re-runnable via the released scripts).Code (MIT):chrono
extraction, linking, audit and gold-standard sampling, the
experiment harness for the reproducible conditions (all con-
dition prompts and the 8,000-char/article query-windowed
budget are verifiable in runconditions.py ; condi-
tions A, B and Corrun with --retriever oracle ),
the deterministic scorer, and the cluster-aware statistics
(stats clustered.py).Not released:the end-to-end
Cprod retriever—its domain-adapted encoderweights, multi-
version index, and inference code, all proprietary. Cprod is
therefore not reproducible from the released artifacts; we
release its per-question model responses so that the reported
Cprod scores remain independently re-verifiable with the
deterministic scorer. Between the original submission and
this camera-ready pass, the article index was revised from
a single “longest” version per cid to three representative
versions per cid (first, median, last by date debut , ex-
cluding stillborn MODIFIE MORT NEartefacts), holding
5https://github.com/rosecymbler/
fiscal-fr-benchthe embedding model, BM25 hyper-parameters, chunker,
and RRF fusion constant. On the associated contamination
question: the encoder’s fine-tuning pairs overlap the bench-
mark on five released articles at the article level—art. 150 U,
one of the five, is flagged out of the answerable scope, leav-
ing 28 affectedscoredquestions on four of the 33 scored
articles—butzeroof those questions’ gold values appear in
any training passage (the pairs hold current-version texts;
the benchmark targets historical values); the check, with its
re-runnable script, is available from the authors on request
(the reranker’s training articles are disjoint from the bench-
mark’s, as stated in §5; the encoder needed this finer value-
level check). The benchmark files embed a BIG-bench-style
canary GUIDso that future training-set contamination is
detectable. All extraction is reproducible end-to-end from
public L ´egifrance and Judilibre data (free API access via
PISTE and the Cour de cassation portal). Random seeds
are fixed (42 for the audit sample, 2026 for the gold stan-
dard, 42 for the scorer and clustered bootstraps). Provider-
specific evaluation settings are documented in code and in
the repository README (SDK version pins; GLM 5.2’s rea-
soning disabled via OpenRouter’s extra body flag; GPT-
5.5 called with reasoning effort="none" for parity
with GPT-5.4 on closed-book probing). The repository also
contains README, METHODOLOGY , DATA SCHEMA,
and a SPEC GOLD STANDARD document specifying the
annotation protocol followed by the authors.
E. Control Condition (Well-Formedness)
The well-formedness of Condition B’s failure is established
withinthe scored set (§7.2), without a separate control split.
Two facts jointly rule out ill-posed questions or a broken
pipeline. (i) Under the identical labeled-article prompt and
retrieval pipeline, changing only the servedversionof the
same article (oracle version selection, Cor) lifts pooled strict
accuracy from 2.7% (Condition B) to99.1%on the same
209 questions (Table 1): the questions are answerable and
the model extracts the value correctlywhen handed the date-
applicable version. (ii) The current-version text lacks the
gold date-anchored value for208 of the 209scored ques-
tions (§5); on the single non-drifted exception the valueis
present and static Condition B retrieves it correctly. Condi-
tion B’s near-zero strict score is therefore version drift, not
question difficulty or a pipeline defect.
13