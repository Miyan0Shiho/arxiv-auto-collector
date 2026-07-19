# Optimizing Visibility in Generative Engines: A Critical Survey of Generative Engine Optimization (2023-2026)

**Authors**: Olivier Martinez

**Published**: 2026-07-15 17:03:08

**PDF URL**: [https://arxiv.org/pdf/2607.14035v1](https://arxiv.org/pdf/2607.14035v1)

## Abstract
Generative Engine Optimization (GEO) seeks to increase content's presence, likelihood of citation, or influence in answers produced by generative engines. Since the foundational GEO paper, the field has expanded rapidly, but terminology, metrics, and evidence standards remain heterogeneous. This critical survey reviews 45 studies selected under a November 2023-July 2026 publication window, including one earlier preprint published at EMNLP after the window opened, plus relevant RAG and evaluation work. We argue that GEO is not a single ranking task but a stochastic, partially observable pipeline spanning search activation, crawling and indexing, retrieval, reranking and context allocation, citation, prominence, factual absorption, fidelity, and user behavior. The foundational paper's widely cited gains are valid within its experimental setting but conditional on a source already being present in a fixed context; they establish neither organic discoverability nor durable traffic effects. Reviewed work indicates that topical relevance and context position are the most reproducible levers, generic heuristics transfer poorly, competition can erode individual gains, and citation-oriented rewrites can impair retrieval. Commercial audits further reveal low source overlap, substantial run-to-run variability, and persistent fidelity gaps. We contribute a multistage formal model, a visibility vector separating discoverability, citation, absorption, and economic outcomes, an evidence hierarchy, and a reproducible protocol based on repeated measurements, paraphrases, controls, human validation, and multi-actor interference. Within this corpus, the evidence is narrow: already-retrieved content can causally alter its citation or use, but no reviewed technique shows a stable, longitudinal, cross-platform causal effect on organic discoverability or downstream behavior.

## Full Text


<!-- PDF content starts -->

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
CRITICAL LITERATURE SURVEY•2023–2026
Optimizing Visibility in Generative Engines
A Critical Survey of Generative Engine Optimization
Olivier Martinez
olivier.martinez2@sciencespo.fr ORCID 0009-0009-3495-5458
Version note.This survey covers work made public or formally accepted from the first version of the foundational paper,
released on November 16, 2023, through July 14, 2026. Peer-reviewed articles, accepted work not yet presented, and preprints
are distinguished throughout.
Abstract
Generative Engine Optimization (GEO) seeks to increase
content’s presence, likelihood of citation, or influence in
answers produced by generative engines. Since the foun-
dational GEO paper, the field has expanded rapidly,
but terminology, metrics, and evidence standards re-
main heterogeneous. This critical survey reviews 45
studies selected under a November 2023–July 2026 pub-
lication window, including one earlier preprint published
at EMNLP after the window opened, plus relevant RAG
and evaluation work. We argue that GEO is not a
single ranking task but a stochastic, partially observ-
able pipeline spanning search activation, crawling and
indexing, retrieval, reranking and context allocation, ci-
tation, prominence, factual absorption, fidelity, and user
behavior. The foundational paper’s widely cited gains
are valid within its experimental setting but conditional
on a source already being present in a fixed context;
they establish neither organic discoverability nor durable
traffic effects. Reviewed work indicates that topical rel-
evance and context position are the most reproducible
levers, generic heuristics transfer poorly, competition can
erode individual gains, and citation-oriented rewrites can
impair retrieval. Commercial audits further reveal low
source overlap, substantial run-to-run variability, and
persistent fidelity gaps. We contribute a multistage for-
mal model, a visibility vector separating discoverability,
citation, absorption, and economic outcomes, an evi-
dence hierarchy, and a reproducible protocol based on
repeated measurements, paraphrases, controls, human
validation, and multi-actor interference. Within this cor-
pus, the evidence is narrow: already-retrieved content
can causally alter its citation or use, but no reviewed
technique shows a stable, longitudinal, cross-platform
causal effect on organic discoverability or downstream
behavior.
Keywords:GenerativeEngineOptimization; generative
engines; AI search; visibility; citations; attribution; RAG;
content optimization; causal measurement; algorithmic
auditing.
Contents
1 Introduction 2
2 Scope and Review Method 2
2.1 Review Type and Cutoff Date . . . . . . . . . . . . 2
2.2 Search, Selection, and Coding . . . . . . . . . . . . 22.3 Publication Status and Interpretive Caution . . . . 3
2.4 Review Limitations . . . . . . . . . . . . . . . . . . 3
3From Generative Engines to GEO: A Multistage
Formalization 3
3.1 A Stochastic and Partially Observable Pipeline . . 3
3.2 A Visibility Vector Rather Than a Single Rank . . 3
3.3 The Causal Estimand of an Intervention . . . . . . 3
4The Foundational Paper: Contributions, Results,
and Empirical Scope 3
4.1 What the Paper Actually Established . . . . . . . 3
4.2 Substantive Findings . . . . . . . . . . . . . . . . . 4
4.3 Structural Limitations . . . . . . . . . . . . . . . . 4
5 Evolution of the Field, 2023–2026 4
6 Measuring Visibility: From Mentions to Effects 4
6.1 A Hierarchy of Metrics . . . . . . . . . . . . . . . . 4
6.2 Reliability: Repetition, Paraphrase, and Time . . . 5
6.3 Denominators and Missing Outputs . . . . . . . . 5
6.4 LLM Judges and Human Validation . . . . . . . . 6
7Influence Techniques: What Holds Up and What
Depends on the Setting 6
7.1Topical Relevance and Position: The Two Most
Robust Factors . . . . . . . . . . . . . . . . . . . . 6
7.2 Extractable Evidence, Structure, and Recency . . . 6
7.3 General Heuristics Generalize Poorly . . . . . . . . 7
7.4 The End-to-End Test: SAGEO Arena . . . . . . . 7
7.5 What Can Reasonably Be Recommended . . . . . 7
8CommercialEngines: ExternalVisibility, Instability,
and Attribution 7
8.1 Surfaces Do Not Share the Same Sources . . . . . . 7
8.2 Activation and Query Profile . . . . . . . . . . . . 7
8.3 Citation Implies Neither Credibility nor Support . 7
8.4 From Recognition to Discovery . . . . . . . . . . . 8
8.5 Traffic and Conversions: The Weakest Evidence . . 8
9 Competition, Manipulation, and Defenses 8
9.1 A Technical Continuum and a Normative Boundary 8
9.2 Evidence of Manipulation in Commercial Systems . 9
9.3 Competition and Interference . . . . . . . . . . . . 9
9.4 Defenses . . . . . . . . . . . . . . . . . . . . . . . . 9
10Critical Synthesis: What the Literature Actually
Establishes 9
11Recommended Protocol for Reproducible GEO
Measurement 9
11.1 Define the Estimand Before Data Collection . . . . 9
11.2 Minimum Factorial Design . . . . . . . . . . . . . . 9
11.3 Statistical Analysis . . . . . . . . . . . . . . . . . . 10
11.4 Quality Assurance and Reproducibility . . . . . . . 10
12Governance and the Political Economy of Genera-
tive Visibility 10
13 Research Agenda 10
13.1 Connecting Crawling, Indexing, and Generation . . 10
13.2 Measuring Absorption Causally . . . . . . . . . . . 10
13.3 Observing Human Attention . . . . . . . . . . . . . 10
13.4Studying Personalization, Languages, and Geogra-
phies . . . . . . . . . . . . . . . . . . . . . . . . . . 10
1
arXiv:2607.14035v1  [cs.IR]  15 Jul 2026

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
13.5 Treating Drift as an Object of Study . . . . . . . . 10
13.6 Multi-Actor Experiments and Equilibria . . . . . . 10
13.7 Defenses and Integrity Certification . . . . . . . . . 11
13.8 Connecting Visibility to Value . . . . . . . . . . . . 11
14 Reproducibility Artifacts 11
15 Conclusion 11
A Evidence Hierarchy 14
B Condensed Matrix of Core Studies 14
1 Introduction
Searchinterfacesbuiltonlargelanguagemodelsnolonger
merely rank a list of links. They retrieve documents,
condense them, answer in natural language, and may
attachcitationstoparticularclaims. Thistransformation
changes the nature of visibility: a publisher no longer
seeks only to occupy a position on a results page, but
also to be retrieved, used, cited, featured in a prominent
passage, and, ideally, selected by the user. The term
Generative Engine Optimizationgave this problem a
name and an initial experimental protocol [Aggarwal
et al., 2024].
The scientific significance is twofold. First, genera-
tive engines are becoming intermediaries for access to
information. Their source-selection decisions distribute
attention, authority, and revenue. Second, conversational
interfaces make the allocation of visibility less transpar-
ent than conventional ranking: the same source may be
absent, cited without being used, paraphrased without a
link, or decisive in shaping the structure of an answer.
Mention frequency alone is therefore insufficient.
Commercial claims about GEO, however, have ad-
vanced faster than the evidence. Theup to 40%figure
from the foundational paper is often recast as a gen-
eral promise of ranking highly in ChatGPT, although
it describes a relative visibility gain in a simulator in
which five documents have already been placed in con-
text. Conversely, the contrary findings of Puerto et al.
[2025] are sometimes presented as a refutation of GEO,
even though they evaluate a different outcome across
several domains and multiple actors. The two bodies of
evidence can be reconciled once the stages of the pipeline
and the causal estimand are distinguished.
This survey advances four propositions.
1.GEO is multistage.Discoverability, rank within
the context, citation, prominence, absorption, and
traffic are distinct variables. An intervention may
improve one stage while impairing another.
2.Visibility is a distribution.It depends on the
engine, date, location, query formulation, whether
search is actually activated, and stochasticity in gen-
eration. A point estimate is not a stable indicator.
3.The evidence is primarily conditional on re-
trieval.Experiments provide strong evidence that a
document that has already been retrieved can alter
an answer. They establish far less often that a page
will be retrieved organically, and almost never that it
will produce a durable effect on conversions.
4.Optimization and manipulation share a chan-nel.Helpful rewrites, adversarial sequences, and indi-
rect prompt injections all modify the text consumed
by the engine. The normative distinction should rest
ontruthfulness, semanticpreservation, disclosure, and
the absence of hidden instructions, not solely on the
objective of visibility.
Our contributions are a structured review of 45 studies;
a formalization of the pipeline; a taxonomy of metrics
and interventions; a critical assessment of evidence from
commercial engines; an account linking competitive opti-
mization, security, and the creator economy; and, finally,
a minimum protocol for measurement and a research
agenda.
2 Scope and Review Method
2.1 Review Type and Cutoff Date
This article is acritical scoping reviewrather than a sys-
tematic review in the clinical sense. The field has not yet
converged on a stable vocabulary, many studies remain
available only as preprints, and the systems under study
change over the publication cycle. Our objective is there-
fore to map the research landscape, assess the strength
of the inferences, and propose a cumulative framework,
without claiming to meta-analyze incomparable effect
sizes.
The primary review window begins on November 16,
2023, the date of the first arXiv version of Aggarwal
et al. [2024], and ends on July 14, 2026. We include Liu
et al. [2023a], whose preprint predates this starting point
but whose publication in the EMNLP proceedings, in
December 2023, appeared after that date. This study
provides an important contemporaneous benchmark for
citation fidelity.
2.2 Search, Selection, and Coding
The search covered arXiv, the ACM Digital Library,
ACL Anthology, the NeurIPS proceedings, PMLR, and
OpenReview, followed by backward and forward citation
searching from the core studies. Search-term families
includedgenerative engine optimization,answer engine
optimization,conversational SEO,AI search visibility,
citation absorption,ranking manipulation,AI Overview
citations, and the names of commercial engines.
A study was included if it met at least one of the
following criteria:
•it defines or evaluates an intervention targeting the
retrieval, rank, citation, mention, or influence of a
source;
•it measures source visibility, stability, fidelity, or con-
centration in a commercial generative engine;
•it studies an attack or defense directly involving the
ranking or use of documents by a conversational en-
gine;
•it formalizes the incentives and externalities created
by the distribution of generative visibility.
Generic work on RAG was excluded unless it con-
tributed a directly necessary metric, such as subquestion
coverage [Xie et al., 2025]. Each study was coded by
2

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
publication status, pipeline stage, whether the system
was controlled or commercial, unit of measurement, prin-
cipal finding, and most consequential threat to validity.
The complete matrix and a retrospective record of the
search protocol are provided as arXiv ancillary files; a
condensed version appears in the appendix.
2.3Publication Status and Interpretive Caution
The corpus combines peer-reviewed articles, formally ac-
cepted but forthcoming papers, workshop papers, and
preprints. This distinction is substantive, not cosmetic.
As of July 14, 2026, the studies by Grossman et al. [2026]
and Vishwakarma et al. [2026] have been accepted to
SIGIR 2026, but the conference begins on July 20; they
are therefore described asforthcoming. Several promi-
nent measurement studies have not been peer-reviewed
[Schulte et al., 2026, Zhang et al., 2026b, Xu et al., 2026,
Allaham and Diakopoulos, 2026]. We use their data
where they complement published findings, but do not
assign them the same evidentiary weight.
2.4 Review Limitations
The pace of the field creates a risk of misalignment
between a manuscript, the product it audits, and the
interface available to the reader. Commercial names also
conceal different configurations: an API receiving URLs,
a model equipped with a web tool, and a consumer-
facing interface are not the same system. The original
search did not retain database-specific hit counts or a
complete exclusion ledger; the ancillary protocol records
this limitation rather than reconstructing unavailable
data. Finally, task heterogeneity precludes a meaningful
aggregation of percentage gains. Our synthesis therefore
emphasizes the direction of effects, their conditions, and
their level of evidence rather than an artificial average,
and any claim of absence is bounded to the reviewed
corpus.
3From Generative Engines to GEO: A
Multistage Formalization
3.1A Stochastic and Partially Observable
Pipeline
Letqdenote a query, ua user state (language, location,
history, account), ean engine,ta point in time, Wtthe
accessible web, and εa randomness term. An engine
may first decide whether to activate search:
A=g act(q,u,e,t,ε A)∈{0,1}.(1)
If search is activated, the engine retrieves a set Rand
then ranks or reranks it:
R=g ret(q,u,W t,e,εR),(2)
π(R) =g rank(q,R,u,e,t,ε K).(3)
Thegeneratorproducesananswer Yandasetofcitations
Cfrom a context windowπ(R) 1:k:
(Y,C) =g gen(q,π(R) 1:k,u,e,t,ε Y).(4)
The user may then read the answer, click a link, com-
plete a conversion, or ignore it. Content creators donot generally observe the complete set R, the reranking
score, or the generator’s internal states. GEO is there-
fore a black-box optimization problem under incomplete
information.
3.2A Visibility Vector Rather Than a Single
Rank
For a sources, we propose the vector
Vs= (D s,Ks,Cs,Ps,Hs,Fs,Bs),(5)
where:
•D sis retrieval probability, or discoverability;
•K sdescribes exposure in the context (rank, top- k
inclusion, allocated tokens);
•C sis the probability of a mention or citation;
•P sis observable prominence (position, repetition, at-
tributed share);
•H sis absorption, namely the source’s effective con-
tribution to the facts, language, or structure of the
answer;
•F sis fidelity, the extent to which attributed claims
are actually supported and rendered accurately;
•B sis the behavioral or economic outcome (click, re-
ferral, conversion, value).
A scalar score Mw=w⊤Vsis defensible only when
the weightswcorrespond to an explicit objective. Ag-
gregating a mention, an accurate citation, and a con-
version without a utility model merely obscures norma-
tive choices. Moreover, DsandCsshould be reported
separately: a high conditional probability of citation,
Pr(Cs= 1|s∈R ), does not compensate for a low
probability of retrieval.
3.3 The Causal Estimand of an Intervention
LetTbe a transformation of a document ds, and letm
be a metric. The average treatment effect of interest is
τT(m) =E[m{Y(T(d s))}−m{Y(d s)}],(6)
with the query, engine, corpus, context order, and timing
controlled or randomized. This notation highlights three
challenges. First, an unpaired before–after comparison
confounds the treatment with stochasticity in generation.
Second, holding Rfixed estimates an effectconditional
on retrieval, not a total effect on the pipeline. Third, the
no-interference assumption is violated: when one source
gains normalized share, another loses it, and when all
competitors optimize, one actor’s treatment changes the
outcomes of the others.
4The Foundational Paper: Contribu-
tions, Results, and Empirical Scope
4.1 What the Paper Actually Established
Aggarwal et al. [2024] introduced three elements that
continue to structure the field. They named the phe-
nomenon, modeled a generative engine as a combination
of retrieval and synthesis, and proposed metrics suited to
citations dispersed throughout a text. Their GEO-bench
3

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Activation →Crawling / indexing →Retrieval →Reranking / context →
Generation / citation
→Absorption / fidelity →Attention / click / conversion
Figure 1: The causal visibility pipeline. Most GEO studies optimize the stages between context allocation and
citation; far fewer observe crawling, organic retrieval, or user behavior.
contains 10000 queries drawn from several datasets. For
each query, the top five Google results are provided to
GPT-3.5-turbo; five answers are generated at temper-
ature 0.7. One source is modified under each of nine
alternative strategies and then compared with its original
version.
The metrics are (i)Word Count, the share of words
attributed to a source; (ii)Position-Adjusted Word Count
(pawc), which discounts passages appearing later in
the answer; and (iii)Subjective Impression, an LLM
judgment inspired by G-Eval [Liu et al., 2023b]. The
shares assigned to the five sources sum to one. Visibility
is therefore intrinsically relative and redistributive.
Theup to 40%figure derives primarily from the in-
crease inpawcfrom 19.3 to 27.2 for Quotation Addition,
or approximately 41% in relative terms. The largest gain
in Subjective Impression is lower. This result does not
mean that 40% more readers will click, nor that a page
will gain 40% in retrieval probability. It means that, in
this testbed, a source already provided to the generator
receives a larger position-weighted share of attributed
text.
4.2 Substantive Findings
Three observations have held up well in the subsequent
literature. First, keyword stuffing does not transfer ef-
fectively from conventional SEO. Second, directly ex-
tractable information—figures, definitions, quotations,
and references—can facilitate the use of a document.
Third, effects depend on the domain and initial rank.
Under theCite Sourcesstrategy, the fifth source gains
115.1% while the first loses 30.3%, illustrating the com-
petitive nature of the metric.
The Perplexity test provides useful but limited vali-
dation: it includes 200 examples, with texts uploaded
as files rather than web pages retrieved organically. The
largest reported gains reach 22% onpawcand 37% on
Subjective Impression. It is therefore a black-box test of
document use, not an experiment in crawling or ranking
in production.
4.3 Structural Limitations
The principal limitations are not errors in the paper;
rather, they define the agenda it left open:
•the source is already present in a five-document con-
text, soD sand much ofK sare fixed;
•without a user study,pawcassumes that an earlier
citation receives more attention;
•the subjective judge is an LLM from the same modelfamily, and its poorly calibrated score is renormalized
to thepawcdistribution;
•interventions that add statistics or references are not
subject to a strong truthfulness constraint;
•no clicks, referrals, traffic, or purchases are observed;
•the engine and benchmark are snapshots, whereas
deployed products change rapidly.
The paper’s enduring contribution therefore lies less
in providing a recipe than in transforming a commercial
intuition into an experimental problem. The subsequent
literature can be read as a progressive decomposition of
the variables that the paper had grouped underimpres-
sion.
5 Evolution of the Field, 2023–2026
This chronology is not strictly sequential, but it reveals
a maturing field. The first shift isolates content transfor-
mations. The second shows that the same channel can
subvert a recommendation. The third adds questions of
generalizability, domain variation, and competition. The
fourth moves upstream toward retrieval, downstream
toward traffic, and treats visibility as an unstable distri-
bution.
6Measuring Visibility: From Mentions
to Effects
6.1 A Hierarchy of Metrics
Theliteratureusestheterm“visibility”torefertoatleast
nine distinct quantities. Table 3 orders them from the
easiest signal to observe to the outcome most closely tied
to user value. Each level addresses a distinct question
and has its own denominator.
Thecitation recallmetric of Liu et al. [2023a] measures
the share of verifiable sentences that are fully supported;
citation precisionmeasures the share of citations that
correctly support their associated sentence. Across Bing
Chat, NeevaAI, Perplexity, and YouChat, only 51.5%
of sentences were fully supported, and 74.5% of cita-
tions supported the proposition with which they were
associated. These metrics evaluate engine fidelity rather
than publisher visibility, but they impose an essential
constraint: a visible citation may be incorrect.
Coverage metrics address a different limitation. A
response may cite many domains while omitting impor-
tant subquestions. Xie et al. [2025] decompose complex
information needs into subquestions; Huang et al. [2026]
distinguish the number of sources from the fraction of
4

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 1: Principal results from Aggarwal et al. [2024]. Values are mean percentage shares, normalized to sum to 100
across sources.
InterventionpawcSubjective Impression Critical interpretation
Baseline 19.3 19.3 Reference condition
Keyword stuffing 17.7 20.2 Reduces the position-adjusted metric
Fluency 24.7 21.9 Moderate, domain-dependent gain
Cite Sources 24.6 21.9 Helps within the fixed context
Quotation Addition 27.2 24.7 Highestpawc; approximately 41% relative gain
Statistics Addition 25.2 23.7 Substantial gain, factuality not guaranteed
Table 2: Four successive shifts in GEO research.
Period Central question Representative studies Conceptual shift
2023–
2024Can a source gain
visibility?Aggarwal et al. [2024]; Wan et al. [2024] From ranked links to answer share; first
controlled interventions
2024–
2025Can rankings and
recommendations
be manipulated?Pfrommer et al. [2024]; Nestaas et al.
[2025]; Kumar and Lakkaraju [2024]Retrieved content becomes an attack
surface; white-hat/adversarial distinction
2025 Do heuristics gener-
alize under compe-
tition?Puerto et al. [2025]; Qian et al. [2025];
Chen et al. [2025a]Countervailing evidence, multi-actor
congestion, and cross-engine differences
2026 How can we
measure the full
pipeline and learn
optimization poli-
cies?Kim et al. [2026]; Liu and Xu [2026];
Kirsten et al. [2026]; Schulte et al. [2026]Stage-specific optimization, absorption
metrics, longitudinal audits, and gover-
nance
cited-source content reflected in the summary’s atomic
content units. This family of metrics brings visibility
closer to informational utility.
Finally, the notion of “absorption” proposed by Zhang
et al. [2026b] distinguishes selection from contribution:
ChatGPT may cite fewer sources while relying more
heavily on each, whereas Google or Perplexity distribute
their citations more broadly. The concept is productive,
but the published score combines position, repetition,
coverage, and textual similarity. It does not reveal an
internal causal trace. A genuine measure of absorption
would ideally require a matched ablation with and with-
out the source, followed by an analysis of the facts and
formulations that change.
6.2Reliability: Repetition, Paraphrase, and
Time
A generative engine is repeatable as an experiment only
at the distributional level. Even a reported tempera-
ture of zero fixes neither the index, nor retrieval, nor
the versions of external services. Across four engines
and 45 days, Schulte et al. [2026] observe daily source-
level Jaccard scores of approximately 0.34–0.42, with
similar levels for repetitions within 24 hours. They pro-
pose seven to eight repetitions per prompt as a starting
point. This number is not a universal standard: it de-
rives from a small universe of Swiss queries and at most
ten repetitions. The appropriate practice is sequential
precision analysis: repeat the measurement until the
interval around the estimand is sufficiently narrow for
the decision at hand.Peer-reviewed audits confirm this phenomenon.
Kirsten et al. [2026] analyze 4706 queries across sev-
eral surfaces in the United States and Germany. Page
overlap across two months is 18% for AI Overviews, com-
pared with 45% for organic Google; on the surfaces for
which temperature could be controlled, repeated runs at
temperature zero change 9–28% of decisions. Grossman
et al. [2026] likewise find that minor reformulations alter
AIO sources more than those returned by conventional
search. A GEO measurement must therefore vary along
at least four dimensions: run, paraphrase, date, and
engine.
6.3 Denominators and Missing Outputs
Discarding responses without search or citations creates
selection bias. In the configuration studied by Schulte
et al. [2026], 57.8% of ChatGPT repetitions did not
activate web search. A dashboard cannot calculate a
“share of citations” only among responses that contain
citations and then interpret it as overall visibility. It
must decompose:
Pr(scited) = Pr(A= 1)
×Pr(s∈R|A= 1)
×Pr(scited|s∈R,A= 1).(7)
This simple identity explains why high conditional cita-
tion rates can coexist with low commercial visibility.
URL canonicalization is also critical. Parameters, redi-
rects, anchors, AMP versions, translated pages, and
aggregators can artificially fragment a domain. Studies
should publish their resolution rules and retain the raw
5

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 3: Taxonomy of visibility metrics.
Level Example metric What it identifies What it does not establish
ActivationPr(A= 1|q) Probability that the generative sur-
face or search is activatedVisibility of a particular source
Retrieval recall@ k,
URL/domain pres-
enceDiscoverability in the candidate
pool or contextCitation or influence in the response
Mention presence of a brand
or entityMinimal nominal exposure Tone, source, evidence, or click
Citation citation rate; rank of
first citationExplicit selection as a source Exact support or substantive contri-
bution
Prominence word count;pawc;
repetitionObservable share and position Human attention without behavioral
validation
Coverage share of subquestions
or units coveredBreadth of supported information
needsCausal effect of the source on word-
ing
Absorption similarity, ablation
with/without the
source, composite
scoreContribution to facts, language, or
structureInternal model states when the
score remains a proxy
Fidelity citation re-
call/precision; en-
tailmentAlignment between a claim and its
sourcePositive visibility or an economic
outcome
Behavior click, referral, conver-
sion, revenueObservable downstream outcome Causality without a control group or
baseline trend
URL, final URL, registrable domain, and, where possible,
a hash of the retrieved content.
6.4 LLM Judges and Human Validation
LLM judges make it possible to evaluate tens of thou-
sands of claims, but they introduce model dependence,
stylistic bias, and circularity. The risk is greatest when
the same model generates the rewrite, the response, and
the score. A robust evaluation should:
1. separate the generator and judge model families;
2. blind the judge to the experimental condition;
3. randomize the order of variants;
4.validate each dimension on a stratified human sample;
5.report agreement, sensitivity, specificity, and disagree-
ment patterns, rather than only an overall correlation;
6.retain an observable metric that does not rely on a
judge, such as a canonical citation or click.
Xu et al. [2026] report 95.6% accuracy for their veri-
fier on a manually assessed sample, which strengthens
their findings, although inaccessible video or social-media
pages remain difficult to evaluate. Vykopal et al. [2026]
likewise use an automated evaluator for part of their
groundedness analysis. These studies point in the right
direction: ajudgeisacceptableasameasuredinstrument,
not as an unexamined source of truth.
7Influence Techniques: What Holds Up
and What Depends on the Setting7.1Topical Relevance and Position: The Two
Most Robust Factors
Query–document relevance is the most reproducible fac-
tor. In ConflictingQA, Wan et al. [2024] use counter-
factual interventions to show that models strongly favor
explicit alignment with the question, often more than
human credibility cues such as scientific references or a
neutral tone. This result does not imply that credibility
is irrelevant; it indicates that, under conflicting evidence
and limited context, the model first follows what appears
to answer the question directly.
Rank within the context is equally consequential.
Puerto et al. [2025] find that moving a source higher
in the context has a greater effect than most rewrites.
The factorial experiment of Vishwakarma et al. [2026],
comprising 252000 trials across six LLMs and eighteen
factors, identifies relevance and position as the primary
determinants of the first citation. This finding redirects
GEO toward upstream stages: a perfectly formulated
page that is absent from the top-kcannot contribute.
7.2Extractable Evidence, Structure, and Re-
cency
Statistics, definitions, comparisons, prices, dates, and ref-
erences have a plausible advantage: they form units that
the generator can select and attribute. The foundational
paper, AutoGEO, and FeatGEO observe positive effects
for several of these properties [Aggarwal et al., 2024,
Wu et al., 2026c, Liu and Xu, 2026]. The controlled
experiment of Vishwakarma et al. [2026] finds effects
for explicit prices and recent dates, whereas formatting
changes alone have weak effects.
Two qualifications are necessary. First, the effect de-
pendsonintent. Arecentdatehelpswithatime-sensitive
6

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
query, but not necessarily with a stable definition. A
quotation may help answer a historical question, yet
burden a product listing. Second, adding a fabricated
statistic may increase reuse while degrading epistemic
quality. The criterion is therefore not to “add numbers,”
but to provide relevant, verifiable, dated, and properly
attributed evidence.
HTML and document structure are beginning to be
studied separately. Yu et al. [2026] report gains asso-
ciated with certain structural features, but the work’s
preprint status and reliance on automated judges war-
rant caution. SAGEO Arena provides a more instructive
result: structural fields may improve retrieval without
necessarily producing the same effect at the reranking
or citation stages [Kim et al., 2026]. Structure should
be evaluated stage by stage, not treated as a universal
talisman.
7.3 General Heuristics Generalize Poorly
C-SEO Bench provides the principal empirical correc-
tive. Across two tasks, six domains, approximately 1900
queries, and 16360 documents, only three of 54 method–
domain combinations are significantly positive in the
main experiment; none is positive in question answering
[Puerto et al., 2025]. Several transformations even reduce
rank. Gains decline as adoption increases, ultimately
producing congested dynamics that approach a zero-sum
game.
E-GEOreachesacompatibleconclusionine-commerce:
ten of the fifteen initial heuristics are neutral or nega-
tive, whereas systematically optimized prompts perform
better; the authors report a relatively stable, domain-
agnostic structure [Bagga et al., 2025]. Recent optimiza-
tion systems—AutoGEO, FeatGEO, Mind Reader, IF-
GEO, and MAGEO—replace fixed recipes with learned
rules, intent decomposition, multiple objectives, or a
memory of strategies [Wu et al., 2026c, Liu and Xu,
2026, Chen et al., 2026, Zhou et al., 2026, Wu et al.,
2026a]. They sometimes achieve substantial gains, but
the candidate document sets are generally fixed in ad-
vance and inference costs remain high.
7.4 The End-to-End Test: SAGEO Arena
SAGEO Arena is crucial because it reinstates retrieval
and reranking. Across 171003 documents and 2700
queries, body-only optimization reduces average top-20
presence by approximately 9%, top-10 presence after
reranking by 16%, and final citation by 6% [Kim et al.,
2026]. ApplyingAutoGEOtothebodyalonecanproduce
larger losses. A rewrite may therefore perform well once
injected while making the document less retrievable or
less competitive upstream.
This result resolves an apparent contradiction. Fixed-
context benchmarks estimate a direct effect on gener-
ation; SAGEO estimates the composition of several ef-
fects. IfTincreases Pr(Cs= 1|s∈R )but decreases
Pr(s∈R ), the total effect may be negative. Any op-
erational claim must specify which of these effects it
measures.7.5 What Can Reasonably Be Recommended
The best-supported recommendation is conservative: pro-
duce a relevant, comprehensive, verifiable, clearly struc-
tured, and technically retrievable page; then measure
retrieval, citation, and fidelity separately. This strategy
more closely resembles high-quality information engineer-
ing than keyword manipulation.
8Commercial Engines: External Visibil-
ity, Instability, and Attribution
8.1 Surfaces Do Not Share the Same Sources
Audits consistently find low overlap. In an audit totaling
1008 responses across three systems, the phase-specific
analysis of 672 Bing Chat and Perplexity responses iden-
tified 355 unique domains, 26% of which were cited by
both systems [Li and Sinnamon, 2024]. Kirsten et al.
[2026] observe that 53% of domains cited by Google AIO
do not appear in the organic top 10 and that 27% are
absent from the top 100. Across 11500 queries, Gross-
man et al. [2026] report URL-level Jaccard similarities
of 0.11–0.18 among organic Google, AIO, and Gemini.
These results refute the notion of a global GEO rank-
ing. Visibility is indexed by engine and surface. A site
visible in the conventional SERP may be absent from
AIO; a domain cited by Perplexity may never appear
in ChatGPT. Any strategy and metric must therefore
identify the product, search mode, and period under
study.
8.2 Activation and Query Profile
The probability of displaying a generative response de-
pends strongly on query form. Xu et al. [2026] analyze
55393 trending queries over 40 days: the overall AIO
activation rate is 13.7%, but rises to 64.7% for queries
phrased as questions. Grossman et al. [2026] observe
AIO for 51.5% of queries in their representative sample.
These figures are not contradictory: the distributions
of queries, dates, and categories differ. They illustrate
precisely why a rate reported without a description of
the sample has limited transportability.
8.3Citation Implies Neither Credibility nor Sup-
port
The foundational audit of citation fidelity had already
demonstrated the gap between appearance and support
[Liu et al., 2023a]. In 2026, Vykopal et al. [2026] ob-
serve credible-source shares of 71.4–86.3%, depending
on the assistant and topic, with more misinformation
sources in some GPT configurations than in Perplexity
or Qwen. Xu et al. [2026] classify approximately 11% of
98020 atomic claims as insufficiently supported, subject
to limitations in retrieval and adjudication. Allaham
and Diakopoulos [2026] report that approximately 16%
of the 19154 textual pages successfully retrieved and
classified were labeled AI-generated by the selected de-
tector; 27.1% of URLs were not scraped because they
were inaccessible, removed, or non-textual.
Visibility should therefore never be maximized inde-
pendently of Fs. A source may be cited for a claim
it does not support, used negatively, or embedded in
7

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 4: Level of empirical support for the main white-hat levers.
Lever Level of support Conditions Cautious interpretation
Query–document
relevanceStrong in con-
trolled settingsWell-defined intent; no fabrication Explicitly address genuine informa-
tion needs
Position in context Strong Document already retrieved Highlights the importance of re-
trieval/reranking
Extractable evidence Moderate to
strongCompatible truthfulness, attribu-
tion, and intentVerifiable figures, definitions, and
comparisons
Recency, prices,
datesModerate Time-sensitive or commercial
queriesUseful but non-universal signals
Document structure Moderate and
heterogeneousDistinct effects at each stage Test headings, tables, and fields
without assuming the direction of
effect
Fluency / simplifica-
tionWeak to moder-
ateDomain- and engine-specific Optimize for the user first
Authoritative tone Weak and unsta-
bleMay conflict with credibility Do not conflate confidence with
evidence
Keyword stuffing Null or negative Multiple benchmarks Avoid
Formatting alone /
fixed recipesPoor generaliza-
tionOccasionally local gains Requires matched, multi-engine
testing
a synthetic-content feedback loop. Audits should pair
citation rates with a matrix covering tone, attribution
accuracy, and factual support.
8.4 From Recognition to Discovery
Sharma [2026] distinguish name recognition from
category-level discovery for 112 startups. In their proto-
col, ChatGPT recognizes 99.4% of products when they
are named, but surfaces them in only 3.32% of organic
discovery queries; Perplexity falls from 94.3% to 8.29%.
These figures come from a single-study preprint using
two models, but the distinction is fundamental. Encod-
ing an entity in model weights or retrieving it by name
does not imply recommending it for a generic intent.
The observations of Chen et al. [2025a] concerning the
overrepresentation ofearned mediashould be read in this
context: third-party mentions may expand the ecosystem
of retrievable evidence. The study remains observational
and industry-supported; it does not show that securing
external coverage mechanically causes a recommendation.
It nevertheless suggests that the relevant unit of GEO
may be a network of sources rather than an isolated
page.
8.5Traffic and Conversions: The Weakest Evi-
dence
Very few studies observe Bs. Watanabe and Nakayashiki
[2026] analyze logs from a website on which some pages
received an AEO intervention. Total ChatGPT referrals
increased by a factor of 5.7, but untreated pages had
already increased by a factor of 3.5 as the platform grew.
A controlled time-series analysis estimates an additional
multiplier of 1.82, with a 95% interval of [1.31, 2.54],
while a conservative temporal placebo yields p= 0.16.
The effect is therefore suggestive rather than causally
established.
The industry study by Zhang et al. [2026a] reports a
20% production traffic lift versus control following thelarge-scale deployment of a VLM-agent framework. This
is rare production evidence, but group sizes, the unit of
assignment, statistical uncertainty, and the components
of the intervention are not described in sufficient detail to
support a general estimate. At this stage, claims about
GEO return on investment clearly outstrip the academic
evidence.
9Competition, Manipulation, and De-
fenses
9.1A Technical Continuum and a Normative
Boundary
White-hat optimization and adversarial attacks share the
mathematical objective in Equation (6). What distin-
guishes them is not whether they exert influence, but the
constraints imposed on T. We propose four cumulative
tests:
1.semantic preservation: do the facts and qualifica-
tions remain true?
2.evidentiary authenticity: are statistics, reviews,
and references verifiable?
3.content–instruction separation: does the docu-
ment inform the user rather than issue hidden com-
mands to the model?
4.disclosure and fairness: is the commercial intent
disclosed, and are competitors represented without
fabricated disparagement?
Reorganizing paragraphs or adding a verified primary
source will generally satisfy these tests. A model-directed
sequence, fabricated testimonial, or instruction to favor
a brand will violate them. This framework prevents
a rewrite from being classified as “white-hat” merely
because it is fluent.
8

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
9.2Evidence of Manipulation in Commercial Sys-
tems
Pfrommer et al. [2024] show that indirect injection into
a document can raise a target by approximately three
ranks on the Perplexity Sonar Large Online API. Because
the URLs are supplied explicitly, the evidence concerns
the post-retrieval stage. Nestaas et al. [2025] provide the
strongest published demonstration across multiple sys-
tems: for example, their preference-manipulation attacks
increase the recommendation rate of a fictitious camera
from 34.0 to 59.4%, while some plugin selections increase
by as much as a factor of 7.2. The products have since
evolved and the pages are controlled, but the existence
of the vulnerability is established.
Work on rankers reinforces this diagnosis. Qian et al.
[2025] identify a decision-making blind spot, while Xing
et al. [2026] optimize short suffixes that promote items
while remaining relatively natural. StealthRank jointly
pursues rank and stealth, but its primary status is that
of a preprint with a workshop version, not a regular
ICML publication [Tang et al., 2025]. CORE reports
high Top-1 success rates across four model families, but
passes a fixed retrieved list of ten products to the ranker
in JSON format, always places the target last, and can
generate fabricated reviews [Jin et al., 2026]. It measures
reranking manipulation, not organic discovery.
9.3 Competition and Interference
When impression shares sum to one, GEO is at least
locally redistributive. C-SEO Bench shows that gains
can erode as adoption increases [Puerto et al., 2025].
Complementary multi-attacker experiments by Nestaas
et al. [2025] demonstrate post-retrieval interference, but
do not establish an organic web equilibrium. Hu [2025]
models attack choices as an infinitely repeated prisoner’s
dilemma and studies conditions for cooperation. In a
distinct framework, Wu et al. [2026b] model competi-
tion among creators and show, under their assumptions,
that citation and compensation can sustain content-
production effort.
This interference invalidates single-actor evaluations
as predictions of equilibrium outcomes. A study should
include several saturation levels—for example, 0, 25, 50,
75, and 100% of documents treated—and estimate both
the direct effect and spillover effects. Otherwise, an
initial individual gain may fall to zero once the strategy
becomes widespread.
9.4 Defenses
Defenses cannot be limited to filtering a few suspicious
words. In GEO-BENCH, adversarial attacks remain
detectable by at least one of the two proxies in most con-
figurations; however, some black-box white-hat rewrites
evade both the lexical filter and the perplexity proxy
in certain domains [Nimase et al., 2026]. GRADA uses
inter-document relationships for reranking and reduces
thesuccessofsomeattacksbyuptoabout80%, withonly
a limited loss in accuracy [Zheng et al., 2025]. Engines
should also separate instructions from retrieved content;
limit the authority granted to documents; compare mul-
tiple independent sources; detect conflicts of interest andunverifiable evidence; retain attribution logs; and audit
effects by actor category.
Defense nevertheless creates a distributional prob-
lem. An overly strict anti-promotion filter may penalize
small publishers that legitimately describe their products,
whereas a permissive filter favors actors able to produce
large volumes of content. Effectiveness must therefore be
assessed alongside false positives, source diversity, and
effects on competition.
10Critical Synthesis: What the Litera-
ture Actually Establishes
The most important conclusion concerns scope. The
evidence is strong for a causal effect conditional on con-
text, moderate for certain informational properties, and
weak for transmission through to traffic. This gradient
explains why practitioners may observe successful cases
even though the literature does not warrant a general
promise.
The second conclusion concerns objectives. Respon-
sible optimization does not maximize only CsorPs. It
seeks a Pareto frontier among discoverability, utility, fi-
delity, cost, stability, and fairness. FeatGEO and several
agentic approaches move in this direction, but their met-
rics remain largely automated [Liu and Xu, 2026, Yuan
et al., 2026]. The future unit of evaluation should be a
multi-objective policy subject to truthfulness constraints,
not a text that wins a citation contest.
11Recommended Protocol for Repro-
ducible GEO Measurement
11.1Define the Estimand Before Data Collection
A study must state whether it targets: (i) the conditional
effect of a rewrite once the document has been injected;
(ii) the total effect in a reproducible pipeline; (iii) an
observational association with a commercial surface; or
(iv) a business outcome in production. Conflating these
estimands yields invalid conclusions. The primary metric,
denominator, exclusions, and threshold for a substan-
tively meaningful difference must be prespecified.
11.2 Minimum Factorial Design
We recommend crossing the following factors:
•multiple named engines and search modes, with ver-
sion and date;
•multiple intents and domains, analyzed separately;
•three to five paraphrases per information need;
•closely spaced repetitions and multiple time windows;
•an untreated baseline, an intervention, and, where
possible, a placebo of comparable length;
•randomized or counterbalanced context order;
•multiple adoption rates when competitors are present.
Seven to eight repetitions constitute a reasonable start-
ing point based on Schulte et al. [2026], but the pilot
study should estimate the variance. The main data col-
lection can then be powered for the desired confidence
9

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
interval. Outputs without search, without citations, or
with errors are outcomes, not data to be discarded.
11.3 Statistical Analysis
Queries, sources, and dates are clustered units. Testing
all generations as though they were independent under-
estimates uncertainty. For a binary citation outcome,
one possible hierarchical model is
logit Pr(C iqetr = 1) =β 0+β1Ti+bq+be+bt+bs+γ⊤X,
(8)
wherebq,be,bt,bscapturequery, engine, date, andsource,
andXcontains the prespecified factors. For proportions
or ranks, suitable models or a cluster bootstrap at the
query–source level are preferable. Studies should report
absolute effects, relative effects, intervals, and distribu-
tions, not merely a mean.
In production, randomization across pages or periods
is ideal. When this is infeasible, difference-in-differences,
interrupted time series, and synthetic controls can help,
but parallel trends and placebo tests must be examined.
The study by Watanabe and Nakayashiki [2026] illus-
trates the importance of a control: a large raw increase
may arise from platform growth.
11.4 Quality Assurance and Reproducibility
Subject to the applicable terms of use, the protocol
should retain prompts, raw responses, citations, search
status, timestamps, locale, account type, user agent,
and canonicalization rules. A stratified human sample
should verify citation attribution, claim support, senti-
ment, factual preservation under rewriting, and judge
errors. Sensitive or protected data can be released as
hashes, metadata, and reconstruction scripts.
12Governance and the Political Economy
of Generative Visibility
GEO redistributes a resource: attention. Three risks
follow. First, engines may concentrate citations on a
small number of domains and create barriers to entry.
Second, sponsored or optimized content may influence an
answer without disclosure equivalent to that required in
conventional advertising. Third, if generated summaries
substitute for clicks, creators may reduce investment in
producing original sources.
Wu et al. [2026b] model the trade-off among user ex-
perience, citations, content-production effort, and com-
pensation. Their conclusions depend on a theoretical
model, but they frame the question correctly: citation
is not merely a fidelity metric; it is part of an incentive
mechanism. Wen et al. [2026] argue that governance
should address concentration, disclosure, and blind spots
between academic research and industry practice.
Several operational principles follow:
•disclose commercial relationships that may affect a
recommendation;
•enable publishers to inspect or challenge incorrect
attribution;
•publish aggregate audits of source concentration and
diversity;•treat indirect instructions as untrusted content by
default;
•do not reward a citation metric without a factual-
support constraint;
•study compensation, licensing, and traffic jointly
rather than separately.
Regulation should target mechanisms and incentives
rather than freeze a list of “GEO-compliant” formats.
Engines evolve too rapidly for a purely technical taxon-
omy, whereas criteria concerning truthfulness, disclosure,
and avenues for redress are more stable.
13 Research Agenda
13.1Connecting Crawling, Indexing, and Gener-
ation
The priority is a benchmark in which a modified page
must actually be crawled, indexed, retrieved, reranked,
and then cited. SAGEO Arena provides a reproducible
step in this direction, but extending the analysis to
commercial systems requires field protocols, indexing
delays, and controlled pages. Content modifications,
internal linking, structured data, domain reputation,
and crawler accessibility must be separated.
13.2 Measuring Absorption Causally
Composite absorption scores must be validated against
interventions with and without the source, atomic factual
units, and, where available, attribution traces provided
by the engine. The question is not merely “which text
resembles the page?” but “which claim or decision would
have changed in its absence?”
13.3 Observing Human Attention
Thepawcassumes that the beginning of an answer at-
tracts more attention. Experiments using eye tracking,
clicks, recall, and choice could estimate empirical po-
sition weights. They should include mobile responses,
spoken output, conversational mode, and the effects of
trust induced by a citation.
13.4Studying Personalization, Languages, and
Geographies
Most studies use English, anonymous accounts, and a
small number of locations. The differences in overlap
between the United States and Germany observed by
Kirsten et al. [2026] motivate multilingual and multi-
region panels. Studies must also distinguish the absence
of local sources from engine bias.
13.5 Treating Drift as an Object of Study
Rather than treating an engine update as a nuisance,
research should measure drift regimes: abrupt changes,
seasonality, citation aging, and policy transfer. A living
benchmark should version prompts, snapshots, and page
content rather than publish a supposedly timeless score.
13.6 Multi-Actor Experiments and Equilibria
Studies should randomize adoption rates and examine
effects on small publishers, large brands, and public-
interest sources. Relevant metrics include individual
gain, informational welfare, concentration, optimization
10

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
cost, and post-equilibrium quality. A strategy that is
useful at 10% adoption but harmful at 100% cannot be
recommended without qualification.
13.7 Defenses and Integrity Certification
Research on attacks must be coupled with responsible-
disclosure constraints and defenses evaluated for false
positives. A standard could certify that a transformation
preserves facts, retains sources, adds no model-directed
instructions, and remains readable to users. Certification
would not guarantee a rank; it would guarantee the
integrity of the optimization process.
13.8 Connecting Visibility to Value
Partnerships with publishers and engines are necessary to
connectDs,Cs,Hs, andBs. Randomized trials should
measurenotonlyclicksbutalsotrafficquality, conversion,
satisfaction, and substitution effects. Without this link,
GEO risks optimizing decorative citations.
14 Reproducibility Artifacts
The accompanying arXiv source package contains the
LaTeX manuscript, the BibTeX bibliography, a detailed
CSV matrix of the 45 studies, and a retrospective CSV
record of the databases, query families, deduplication
rule, and review limitations. The original database-
specific hit counts and complete exclusion ledger were
not retained and are explicitly marked as unavailable.
A dated copy should also be archived in a persistent
repository because the publication status of the preprints
will change.
15 Conclusion
Between November 2023 and July 2026, GEO evolved
from a set of heuristics into a research program on source
selection, attribution, absorption, and competition. The
foundational paper correctly identified a new visibility
surface and demonstrated that the presentation of an
already retrieved document can alter its share of an an-
swer. Subsequent work has, however, narrowed the scope
of that conclusion: general heuristics transfer poorly,
context order and relevance often dominate, competi-
tion can erode gains in tested multi-actor settings, and
downstream optimization can impair retrieval.
Commercial audits likewise show that engines cite
different source ecosystems, vary across runs, and may
attribute claims to pages that do not adequately support
them. Visibility therefore cannot be reduced to “being
cited by ChatGPT.” It must be measured as a vector
across multiple engines, prompts, and periods, while
retaining fidelity measures and null outcomes.
Within the reviewed corpus, the best-supported syn-
thesis is both robust and limited:already retrieved con-
tent can causally influence an answer, while the review
identified no technique with a stable, longitudinal, cross-
platform causal effect on organic discoverability or down-
stream clicks and conversions. This limitation does not
invalidate GEO. It defines the scientific work still re-
quired to turn it into a cumulative discipline rather than
a promotional vocabulary.
11

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 5: Confidence in the field’s principal claims.
Confidence Claim Empirical basis and
caveat
High A document already placed in the context can causally alter its rank, cita-
tion, or use.Controlled replications
across multiple models;
does not address organic
retrieval.
High Query–document relevance and context position are major determinants. Counterfactual experi-
ments, benchmarks, and
a factorial study; effects
may vary with length
and task.
High Commercial engines differ from one another and vary over time. Published audits and
large preprints; hetero-
geneous products and
periods.
High Retrieved documents constitute a genuine attack surface. EMNLP 2024, ICLR
2025, and ranker studies;
many effects remain
post-retrieval.
Moderate Extractable evidence and suitable structure often facilitate use. Consistent findings, but
dependent on intent,
engine, and factuality.
Moderate Systematically optimized or learned methods often outperform fixed heuris-
tics in controlled benchmarks.AutoGEO, E-GEO,
FeatGEO, and agentic
methods; fixed contexts
and substantial costs.
Moderate Competitive adoption can erode individual gains in tested multi-actor
settings.C-SEO Bench; comple-
mentary post-retrieval
multi-attacker exper-
iments; few real-web
ecosystem studies.
Low A white-hat GEO intervention durably improves organic discoverability
across multiple engines.Very few end-to-end
tests; SAGEO even finds
adverse upstream ef-
fects.
Very low Citation scores predict clicks, conversions, or revenue. One suggestive quasi-
experiment and a few
industry claims; causal-
ity not established.
Rejected as a
general claim“GEO increases visibility by 40%.” The figure is a relative
maximum on one metric
under a specific configu-
ration.
12

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 6: Minimum checklist for a GEO study.
Component Requirement Failure avoided
System Product, mode, model, date, locale, account, and search enabled Comparing different sur-
faces under the same
name
Sample Intents, languages, paraphrases, and inclusion rule Generalizing from a hand-
ful of prompts
Treatment Before/after text, length, factuality, and randomization Confounding the effect
with added information
or order
Repetitions Closely spaced runs and multiple time windows Treating a one-off score
as a stable rank
Pipeline Retrieval, reranking, context, and generation separated A downstream gain mask-
ing an upstream loss
Metrics VectorV s, denominators, and null outcomes retained Conflating citation with
discoverability
Statistics Clustering, intervals, and absolute and relative effects Pseudoreplication and
unstable gains
Validation Human sample, blinded judge, agreement, and errors Circularity in LLM-based
judging
Competition Adoption rates and spillover effects Non-transportable single-
actor prediction
Downstream Control or randomization for clicks and traffic Attributing platform
growth to GEO
Integrity Verified sources, disclosure, and no hidden instructions Optimization that re-
wards misinformation
13

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
A Evidence Hierarchy
Table 7: Proposed framework for grading a GEO claim.
Level Design Example of a warranted conclusion Caveat
A Randomized field
trial or strong quasi-
experiment with logs
and controlsCausal effect on clicks, traffic, or conver-
sions in the setting studiedLimited transportability across products
and periods
B Live commercial en-
gines, repetitions,
paraphrases, and mul-
tiple datesDistribution of external visibility and
cross-engine differencesAPI/interface configuration and person-
alization may be incompletely character-
ized
C Commercial system
with manually sup-
plied URLs or filesBlack-box post-retrieval effect Does not establish crawling, indexing, or
organic retrieval
D Reproducible RAG
pipeline with retrieval,
reranking, and genera-
tionStage-specific causal effects and the total
effect within the testbedExternal validity for proprietary prod-
ucts
E Fixed context, syn-
thetic ranker, or LLM
judgePossible mechanism conditional on the
contextWeak evidence of real-world web visibil-
ity
A study may combine several levels. For example, Pfrommer et al. [2024] include both controlled experiments and
post-retrieval commercial validation, while Watanabe and Nakayashiki [2026] observe real logs but obtain causal
identification that is limited by the placebo test. The level grades a specific claim, not the overall value of a paper.
B Condensed Matrix of Core Studies
Table 8: Directly relevant studies included in the review.PEER= peer-reviewed;FORTHC.= accepted and
forthcoming as of the cutoff date;PREPR.= preprint;WKSP.= workshop.
Study Status Main stage Key result or limitation
Aggarwal et al.
2024PEERcitation/prominence Up to approximately 40% relative improvement in a five-
document context; no end-to-end retrieval.
Liu et al. 2023PEERfidelity 51.5% of sentences fully supported and 74.5% of citations
correct across four historical engines.
Li and Sinnamon
2024PEERcommercial sources 1008 responses overall; 26% domain overlap in the 672-
response Bing–Perplexity phase.
Wan et al. 2024PEERinfluence Topical relevance dominates in ConflictingQA; controlled
context.
Kumar and
Lakkaraju 2024PREPR.product ranking Strategic sequences effective in a fictitious catalog; no com-
mercial engine.
Pfrommer et al.
2024PEERpost-retrieval attack Injection transfers to Perplexity; URLs explicitly supplied.
Nestaas et al. 2025PEERcommercial attack Preference manipulation on Bing, Perplexity, and plugins;
controlled pages.
Narayanan Venkit
et al. 2025PEERuse/verifiability Qualitative study showing the practical limitations of cita-
tions; 3 pilot and 21 main-study participants.
Puerto et al. 2025PEERcompetition Three positive cases out of 54; none in QA; gains approach
zero under broad adoption.
Qian et al. 2025PEERadversarial ranker Exploitable decision-making blind spot in LLM ranking.
Zheng et al. 2025PEERdefense GRADA substantially reduces attack success with limited
accuracy loss.
Wu et al. 2026
(AutoGEO)PEERlearned optimization Reported average improvement of +35.99%, conditional on
five retrieved documents.
Chen et al. 2025
(CC-GSEO)PREPR.influence/quality Retrieved web contexts, but LLM-generated article-centric
queries and predominantly automated judges; no comple-
mentary human validation.
Chen et al. 2025
(audit)PREPR.commercial engines Earned media overrepresented; observational snapshot.
Lüttgenau et al.
2025PREPR.rewriting Gains on synthetic travel content; small extrinsic evaluation.
Continued on next page
14

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Table 8 (continued)
Study Status Main stage Key result or limitation
Ho et al. 2025WKSP.ad retrieval PPO improves inclusion, with a small absolute MRR gain;
offline pipeline.
Bagga et al. 2025PREPR.e-commerce Ten of fifteen heuristics neutral or negative; systemati-
cally optimized prompts perform better and converge on
a domain-agnostic structure.
Kim et al. 2026PREPR.full pipeline Downstream rewrites degrade retrieval and reranking; non-
commercial pipeline.
Tian et al. 2026PREPR.citation diagnostics Targeted repairs reported with 5% of text modified; corpus
inconsistencies.
Yuan et al. 2026PREPR.multi-objective agents Offline tests on GEO-Bench, MS MARCO, and a custom
Amazon-derived dataset using two open-weight engines and
fixed five-document lists.
Liu and Xu 2026PEER multi-objective fea-
turesInformational features more useful than lexical ones; high
cost and substantial reliance on judges.
Chen et al. 2026
(Mind Reader)PEERlatent demand Large gains, but generation and evaluation are largely per-
formed by LLMs.
Zhou et al. 2026PEERmultiple queries Formalizes conflicts and downside risk; main engine is simu-
lated.
Wu et al. 2026
(MAGEO)PEERagents/attribution Reusable strategies and a fidelity–visibility score; frozen
context.
Tang et al. 2025WKSP.stealth attack Rank–fluency trade-off on open models; no commercial
engine.
Xing et al. 2026PEERranker attack Short, naturalistic suffixes promote targets; simplified con-
text.
Jin et al. 2026PREPR.product ranking High reported Top- ksuccess; fixed retrieved lists of ten
products supplied in JSON and possible fabricated reviews.
Nimase et al. 2026PREPR.attack benchmark Compares attacks and white-hat methods on one ranker;
shares its name with the original GEO-bench.
Smirnov 2026PREPR.snippets RL exploits comparative preferences; simulated overview.
Kirsten et al. 2026PEERmulti-surface audit 4706 queries, low overlap, and 9–28% repeated-decision
changes on temperature-controllable surfaces.
Grossman et al.
2026FORTHC.Google audit 11500 queries; AIO on 51.5%; cross-surface Jaccard below
0.2.
Xu et al. 2026PREPR.longitudinal AIO 55393 queries; 13.7% activation; 11% of claims insufficiently
supported.
Allaham and Di-
akopoulos 2026PREPR.synthetic sources Approximately 16% of 19154 retrieved, classified textual
pages; single detector and 27.1% inaccessible, removed, or
non-textual URLs.
Schulte et al. 2026PREPR.stability Visibility as a distribution; small Swiss universe.
Huang et al. 2026PREPR.coverage Measures the fraction of cited-source content reflected in
summary atomic content units; dated Natural Questions
dataset.
Zhang, He, and
Yao 2026PREPR.absorption 21143 citations and 72 features; noncausal composite score.
Vishwakarma et al.
2026FORTHC.citation factors 252000 trials; relevance and position dominate; two docu-
ments injected.
Vykopal et al. 2026PEERcredibility Differences in groundedness and questionable sources; re-
stricted domains.
Watanabe and
Nakayashiki 2026PREPR.traffic Estimated ITS multiplier of 1.82, but placebo p= 0.16; a
single site.
Sharma 2026PREPR.discovery Large recognition–discovery gap for 112 startups; two mod-
els.
Zhang et al. 2026
(Pinterest)PREPR.production Reported 20% production traffic lift versus control; causal
attribution insufficiently detailed.
Yu et al. 2026PREPR.structure Structural gains reported across six engines; limited valida-
tion.
Hu 2025WKSP.theory/competition Infinitely repeated prisoner’s dilemma for attack choices;
conclusions depend on the model.
Wu et al. 2026
(ecosystem)PREPR.economics Creator competition model in which citations and compensa-
tion may sustain effort; model-dependent.
Wen et al. 2026PEERgovernance Position paper on concentration, disclosure, and academic
blind spots.
15

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
References
Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpuro-
hit, Ashwin Kalyan, Karthik Narasimhan, and Ameet
Deshpande. GEO: Generative engine optimization.
InProceedings of the 30th ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining, pages
5–16, 2024. doi: 10.1145/3637528.3671900. URL
https://doi.org/10.1145/3637528.3671900.
Mowafak Allaham and Nicholas Diakopoulos. Synthetic
sources? auditinggenerativesearchenginecitationsfor
evidence of AI-generated sources, 2026. URL https:
//arxiv.org/abs/2605.23684.
Puneet S. Bagga, Vivek F. Farias, Tamar Korkotashvili,
Tianyi Peng, and Yuhang Wu. E-GEO: A testbed for
generative engine optimization in e-commerce, 2025.
URLhttps://arxiv.org/abs/2511.20867.
Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, and Nick
Koudas. Generative engine optimization: How to
dominate AI search, 2025a. URL https://arxiv.
org/abs/2509.08919.
Qiyuan Chen, Jiahe Chen, Hongsen Huang, Qian Shao,
Jintai Chen, Renjie Hua, Hongxia Xu, Ruijia Wu,
Chuan Ren, and Jian Wu. CC-GSEO-Bench: A
content-centric benchmark for measuring source in-
fluence in generative search engines, 2025b. URL
https://arxiv.org/abs/2509.05607.
Tong Chen, JiaWei Guo, Yuxi Li, Baiming Chen, Houx-
ing Ren, Zhiwei Zhang, Yunxiang Zhang, Hanyang Xia,
Kun Liang, and Zhaoran Fan. Mind reader: Latent
user demand-guided content optimization for genera-
tive search engine. InProceedings of the 64th Annual
Meeting of the Association for Computational Linguis-
tics, pages 40832–40848, 2026. doi: 10.18653/v1/2026.
acl-long.1894. URL https://aclanthology.org/20
26.acl-long.1894/.
Riley Grossman, Songjiang Liu, Michael K. Chen, Mike
Smith, Cristian Borcea, and Yi Chen. How generative
AIdisruptssearch: Anempiricalstudyofgooglesearch,
gemini, and AI overviews.Proceedings of the 49th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, 2026. URL
https://arxiv.org/abs/2604.27790 . Forthcoming
as of July 14, 2026.
Chloe Ho, Ishneet Sukhvinder Singh, Diya Sharma,
Tanvi Reddy Anumandla, Michael Lu, Vasu Sharma,
and Kevin Zhu. Rewrite-to-rank: Optimizing ad vis-
ibility via retrieval-aware text rewriting, 2025. URL
https://arxiv.org/abs/2507.21099 . ICML 2025
workshop.
Xiyang Hu. Dynamics of adversarial attacks on large
language model-based search engines, 2025. URL
https://arxiv.org/abs/2501.00745 . ICML 2026
workshop version.Michelle Huang, Agam Goyal, Koustuv Saha, and Es-
hwar Chandrasekharan. Answer bubbles: Informa-
tion exposure in AI-mediated search, 2026. URL
https://arxiv.org/abs/2603.16138.
Haibo Jin, Ruoxi Chen, Peiyan Zhang, Yifeng Luo,
Huimin Zeng, Man Luo, and Haohan Wang. Control-
ling output rankings in generative engines for LLM-
based search, 2026. URL https://arxiv.org/abs/
2602.03608.
Sunghwan Kim, Wooseok Jeong, Serin Kim, Sangam
Lee, and Dongha Lee. SAGEO Arena: A real-
istic environment for evaluating search-augmented
generative engine optimization, 2026. URL https:
//arxiv.org/abs/2602.12187.
Elisabeth Kirsten, Jost Große Perdekamp, Qinyuan Wu,
Mihir Upadhyay, Krishna P. Gummadi, and Muham-
mad Bilal Zafar. Characterizing web search in the
age of generative AI. InFindings of the Associa-
tion for Computational Linguistics: ACL 2026, pages
10827–10848, 2026. doi: 10.18653/v1/2026.findings-a
cl.526. URL https://aclanthology.org/2026.find
ings-acl.526/.
AounonKumarandHimabinduLakkaraju. Manipulating
large language models to increase product visibility,
2024. URLhttps://arxiv.org/abs/2404.07981.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. Retrieval-
augmented generation for knowledge-intensive NLP
tasks. InAdvances in Neural Information Processing
Systems, volume 33, pages 9459–9474, 2020. URL
https://proceedings.neurips.cc/paper/2020/ha
sh/6b493230205f780e1bc26945df7481e5-Abstra
ct.html.
Alice Li and Luanne Sinnamon. Generative AI search
engines as arbiters of public knowledge: An audit of
bias and authority.Proceedings of the Association for
Information Science and Technology, 61(1):205–217,
2024. doi: 10.1002/pra2.1021. URL https://doi.
org/10.1002/pra2.1021.
Nelson F. Liu, Tianyi Zhang, and Percy Liang. Eval-
uating verifiability in generative search engines. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 7001–7025, 2023a. doi:
10.18653/v1/2023.findings-emnlp.467. URL https:
//aclanthology.org/2023.findings-emnlp.467/.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. G-Eval: NLG
evaluation using GPT-4 with better human align-
ment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522, 2023b. doi: 10.18653/v1/2023.
emnlp-main.153. URL https://aclanthology.org/
2023.emnlp-main.153/.
16

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Zikang Liu and Peilan Xu. Think before writing: Feature-
level multi-objective optimization for generative cita-
tion visibility. InProceedings of the 64th Annual Meet-
ing of the Association for Computational Linguistics,
pages 20290–20303, 2026. doi: 10.18653/v1/2026.ac
l-long.929. URL https://aclanthology.org/2026
.acl-long.929/.
Florian Lüttgenau, Imar Colic, and Gervasio Ramirez.
Beyond SEO: A transformer-based approach for rein-
venting web content optimisation, 2025. URL https:
//arxiv.org/abs/2507.03169.
Fredrik Nestaas, Edoardo Debenedetti, and Florian
Tramèr. Adversarial search engine optimization for
large language models. InInternational Conference
on Learning Representations, 2025. URL https:
//openreview.net/forum?id=hkdqxN3c7t.
Ojas Nimase, Zhe Chen, Gengpei Qi, Yue Zhao, and
Xiyang Hu. GEO-Bench: Benchmarking ranking ma-
nipulation in generative engine optimization, 2026.
URLhttps://arxiv.org/abs/2605.29107.
Samuel Pfrommer, Yatong Bai, Tanmay Gautam, and
Somayeh Sojoudi. Ranking manipulation for con-
versational search engines. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 9523–9552, 2024. doi:
10.18653/v1/2024.emnlp-main.534. URL https://
aclanthology.org/2024.emnlp-main.534/.
Haritz Puerto, Martin Gubri, Tommaso Green,
Seong Joon Oh, and Sangdoo Yun. C-SEO Bench:
Does conversational SEO work? InAdvances in
Neural Information Processing Systems 38: Datasets
and Benchmarks Track, 2025. URL https://proc
eedings.neurips.cc/paper_files/paper/2025/ha
sh/27aa3aeff0f8460a7b43d30fa6c5c032-Abstra
ct-Datasets_and_Benchmarks_Track.html.
Yaoyao Qian, Yifan Zeng, Yuchao Jiang, Chelsi Jain, and
Huazheng Wang. The ranking blind spot: Decision
hijacking in LLM-based text ranking. InProceedings of
the 2025 Conference on Empirical Methods in Natural
Language Processing, pages 21958–21968, 2025. doi:
10.18653/v1/2025.emnlp-main.1116. URL https://
aclanthology.org/2025.emnlp-main.1116/.
Julius Schulte, Malte Bleeker, and Philipp Kaufmann.
Don’t measure once: Measuring visibility in AI search
(GEO), 2026. URL https://arxiv.org/abs/2604.0
7585.
Amit Prakash Sharma. The discovery gap: How product
hunt startups vanish in LLM organic discovery queries,
2026. URLhttps://arxiv.org/abs/2601.00912.
Roman Smirnov. Exploring LLM biases to manipulate
AI search overview, 2026. URL https://arxiv.org/
abs/2605.00012.
Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue
Zhao, and Xiyang Hu. StealthRank: LLM ranking
manipulation via stealthy prompt optimization, 2025.URL https://arxiv.org/abs/2504.05804 . ICML
2026 workshop version.
Zhihua Tian, Yuhan Chen, Yao Tang, Jian Liu, and
Ruoxi Jia. Diagnosing and repairing citation failures
in generative engine optimization, 2026. URLhttps:
//arxiv.org/abs/2603.09296.
Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou,
Yixin Mao, and Chien-Sheng Wu. Search engines
in the AI era: A qualitative understanding to the
false promise of factual and verifiable source-cited
responses in LLM-based search. InProceedings of the
2025 ACM Conference on Fairness, Accountability,
and Transparency, pages 1325–1340, 2025. doi: 10.114
5/3715275.3732089. URL https://doi.org/10.114
5/3715275.3732089.
Rahul Vishwakarma, Shushant Kumar, and Ratnesh
Jamidar. What gets cited: Competitive GEO in AI
answer engines.Proceedings of the 49th International
ACM SIGIR Conference on Research and Development
in Information Retrieval, 2026. doi: 10.1145/3805712.
3808445. URL https://arxiv.org/abs/2605.25517 .
Forthcoming as of July 14, 2026.
Ivan Vykopal, Matúš Pikuliak, Simon Ostermann, and
Marián Šimko. Assessing web search credibility and
response groundedness in chat assistants. InProceed-
ings of the 19th Conference of the European Chapter of
the Association for Computational Linguistics, pages
2539–2560, 2026. doi: 10.18653/v1/2026.eacl-long.115.
URL https://aclanthology.org/2026.eacl-long.
115/.
Alexander Wan, Eric Wallace, and Dan Klein. What
evidence do language models find convincing? In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics, pages 7468–
7484, 2024. doi: 10.18653/v1/2024.acl-long.403. URL
https://aclanthology.org/2024.acl-long.403/.
Keisuke Watanabe and Kazuki Nakayashiki. Disentan-
glinganswerengineoptimizationfromplatformgrowth:
A log-based natural experiment on chatgpt referral
traffic, 2026. URL https://arxiv.org/abs/2606.0
4362.
Yizhu Wen, Nan Zhang, Haohan Yuan, Xun Chen,
Haopeng Zhang, and Hanqing Guo. Position: Genera-
tive engine optimization creates underexamined risks,
governance must target concentration, disclosure, and
academic blind spots. InProceedings of the 43rd Inter-
national Conference on Machine Learning: Position
Paper Track, 2026. URL https://arxiv.org/abs/26
06.12439.
Beining Wu, Fuyou Mao, Jiong Lin, Cheng Yang, Jiax-
uan Lu, Yifu Guo, Siyu Zhang, Yifan Wu, Ying Huang,
and Fu Li. From experience to skill: Multi-agent gen-
erative engine optimization via reusable strategy learn-
ing. InFindings of the Association for Computational
Linguistics: ACL 2026, pages 43305–43315, 2026a.
doi: 10.18653/v1/2026.findings-acl.2149. URL https:
//aclanthology.org/2026.findings-acl.2149/.
17

A Critical Survey of Generative Engine Optimization Version dated July 15, 2026
Yihang Wu, Jiajun Tang, Jinfei Liu, Haifeng Xu, and
Fan Yao. Do AI overviews benefit search engines? an
ecosystem perspective, 2026b. URL https://arxiv.
org/abs/2601.22493.
Yujiang Wu, Shanshan Zhong, Yubin Kim, and Chenyan
Xiong. What generative search engines like and how to
optimize web content cooperatively. InThe Fourteenth
International Conference on Learning Representations,
2026c. URL https://iclr.cc/virtual/2026/poste
r/10010153.
Kaige Xie, Philippe Laban, Prafulla Kumar Choubey,
Caiming Xiong, and Chien-Sheng Wu. Do RAG sys-
tems cover what matters? evaluating and optimiz-
ing responses with sub-question coverage. InPro-
ceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Com-
putational Linguistics, pages 5836–5849, 2025. doi:
10.18653/v1/2025.naacl-long.301. URL https://ac
lanthology.org/2025.naacl-long.301/.
Tiancheng Xing, Jerry Li, Yixuan Du, and Xiyang Hu.
Are LLMs reliable rankers? rank manipulation via two-
stage token optimization. InProceedings of the 64th
Annual Meeting of the Association for Computational
Linguistics, pages 9120–9132, 2026. doi: 10.18653
/v1/2026.acl-long.413. URL https://aclanthology.
org/2026.acl-long.413/.
Haofei Xu, Umar Iqbal, and Jacob M. Montgomery. Mea-
suring google AI overviews: Activation, source qual-
ity, claim fidelity, and publisher impact, 2026. URL
https://arxiv.org/abs/2605.14021.
Junwei Yu, Mufeng Yang, Yepeng Ding, and Hiroyuki
Sato. Structural feature engineering for generative
engine optimization: How content structure shapes
citation behavior, 2026. URL https://arxiv.org/
abs/2603.29979.
Jiaqi Yuan, Jialu Wang, Zihan Wang, Qingyun Sun, Rui-
jie Wang, and Jianxin Li. AgenticGEO: A self-evolving
agentic system for generative engine optimization,
2026. URLhttps://arxiv.org/abs/2603.20213.
Faye Zhang, Qianyu Cheng, Jasmine Wan, Vishwakarma
Singh, Jinfeng Rao, and Kofi Boakye. Generative
engine optimization: A VLM and agent framework
for pinterest acquisition growth, 2026a. URL https:
//arxiv.org/abs/2602.02961.
Kai Zhang, Xinyue He, and Jingang Yao. From ci-
tation selection to citation absorption: A measure-
ment framework for generative engine optimization
across AI search platforms, 2026b. URL https:
//arxiv.org/abs/2604.25707.
Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xu-
anli He, Pasquale Minervini, Youcheng Sun, and
Qiongkai Xu. GRADA: Graph-based reranking against
adversarial documents attack. InProceedings of
the 2025 Conference on Empirical Methods in Nat-
ural Language Processing, pages 22244–22266, 2025.doi: 10.18653/v1/2025.emnlp-main.1132. URL https:
//aclanthology.org/2025.emnlp-main.1132/.
Heyang Zhou, Jiajia Chen, Xiaolu Chen, Jie Bao, Zhen
Chen, and Yong Liao. IF-GEO: Conflict-aware instruc-
tion fusion for multi-query generative engine optimiza-
tion. InFindings of the Association for Computational
Linguistics: ACL 2026, pages 27576–27590, 2026.
doi: 10.18653/v1/2026.findings-acl.1373. URL https:
//aclanthology.org/2026.findings-acl.1373/.
18