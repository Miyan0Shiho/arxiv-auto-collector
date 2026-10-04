# AGO AI Quality Gate: Evidence-First Release Decisions for Retrieval-Augmented Generation

**Authors**: Giulio Zeloni, Enrico Lo Conte, Salvatore Rionero, Giuseppe Santoro, Alessandro Rastelli, Fabio Sorrentino

**Published**: 2026-10-01 07:24:04

**PDF URL**: [https://arxiv.org/pdf/2610.01218v1](https://arxiv.org/pdf/2610.01218v1)

## Abstract
Enterprises adopting retrieval-augmented generation (RAG) face a recurring operational decision: promote, revise, or block a system version. The evidence is incomplete and the metrics come from fallible LLM judges. We report on AGO AI Quality Gate (AGO), an evidence-first quality-gate framework deployed in industrial RAG assessment engagements. AGO integrates four key components: a four-state decision model that treats missing data and judge errors as explicit outcomes; layered scoring combining deterministic checks, local guardrails, and structured LLM evaluation; a stratified beta-binomial gate that quantifies regression risk probabilistically; and a mandatory meta-evaluation protocol to validate the LLM judge before it influences decisions. Since engagement data is proprietary, we evaluate the judge layer on RAGBench, a public benchmark of 100k annotated RAG traces across 12 datasets. On identical stratified test samples (N=1200 per judge), a low-cost judge (gpt-4.1-nano) detects non-adherent answers barely above chance (AUROC 0.603 [0.570, 0.634]), despite producing flawless protocol output, while gpt-4o reaches 0.783 [0.756, 0.807] -- yet its per-domain performance still ranges from 0.62 to 0.88. A fixed-seed gate study spanning regression, no change, and improvement quantifies unsafe promotion, false-alarm cost, and improvement throughput. Under regression, the decision-grade profile reduces unsafe promotion to 22.2%-35.1%, against 29.3%-41.8% for a naive gate. These results support the design choices that judge quality must be measured per engagement and that point estimates alone are not a release decision.

## Full Text


<!-- PDF content starts -->

AGO AI Quality Gate: Evidence-First Release
Decisions for Retrieval-Augmented Generation
Giulio Zeloni, Enrico Lo Conte,
Salvatore Rionero, Giuseppe Santoro,
Alessandro Rastelli, and Fabio Sorrentino
Protom Group S.p.A., Napoli, Italy
giulio.zeloni@protom.com
Abstract.Enterprises adopting retrieval-augmented generation (RAG)
face a recurring operational decision: promote, revise, or block a sys-
tem version. The evidence is incomplete and the metrics come from
fallible LLM judges. We report on AGO AI Quality Gate (AGO), an
evidence-first quality-gate framework deployed in industrial RAG as-
sessment engagements. AGO integrates four key components: a four-
state decision model that treats missing data and judge errors as ex-
plicit outcomes; layered scoring combining deterministic checks, local
guardrails, and structured LLM evaluation; a stratified beta-binomial
gate that quantifies regression risk probabilistically; and a mandatory
meta-evaluation protocol to validate the LLM judge before it influences
decisions. Since engagement data is proprietary, we evaluate the judge
layer on RAGBench, a public benchmark of 100k annotated RAG traces
across 12 datasets. On identical stratified test samples (N=1200per
judge), a low-cost judge (gpt-4.1-nano) detects non-adherent answers
barely above chance (AUROC0.603[0.570, 0.634]), despite producing
flawless protocol output, while gpt-4o reaches0.783[0.756, 0.807]—yet
its per-domain performance still ranges from0.62to0.88. A fixed-seed
gate study spanning regression, no change, and improvement quantifies
unsafe promotion, false-alarm cost, and improvement throughput. Under
regression, the decision-grade profile reduces unsafe promotion to22.2%–
35.1%, against29.3%–41.8%for a naive gate. These results support the
design choices that judge quality must be measured per engagement and
that point estimates alone are not a release decision.
Keywords:Retrieval-Augmented Generation·LLM-as-a-Judge·Qual-
ity Gates·Evaluation·Uncertainty Quantification
1 Introduction
Retrieval-augmented generation (RAG) [12] is now a standard architecture for
grounding large language models (LLMs) in private document collections; de-
ployments range from customer support and compliance to healthcare and fi-
nance [8]. Grounding mitigates but does not eliminate hallucination: answers
may contradict retrieved evidence, cite the wrong source, or fabricate content
arXiv:2610.01218v1  [cs.CL]  1 Oct 2026

2 G. Zeloni et al.
when retrieval fails [10]. In an enterprise setting this becomes a concrete, recur-
ringdecisionaboutwhetheragivenversioncanbereleased,oritneedsadditional
reviews.
This paper reports on AGO AI Quality Gate (AGO), a self-hosted quality-
gate platform built to support that decision. AGO is deployed at an IT consult-
ing firm and used in RAG assessment engagements with enterprise clients. En-
gagement constraints shaped the design: client data cannot leave the premises;
observability is often incomplete; evaluation datasets are small (tens to a few
hundred cases) and stratified over imbalanced categories; and consultants must
be able to defend every automated decision they sign.
A rich line of work addresses RAGmeasurement. Frameworks such as RA-
GAs [6] and ARES [15] decompose quality into retrieval-side and generation-side
components scored by an LLM judge [17,13]. Benchmarks such as RGB [2] and
RAGBench [7] assess the evaluators themselves. Deploying these components in-
side an accountable gate exposed three gaps between scoring and deciding. First,
evidence is routinely incomplete: traces lack contexts, providers fail, judges re-
turn malformed output; averaging over whatever is available conflates “measured
as bad” with “not measured”. Second,point estimates overstate certaintyon the
small, stratified evaluation sets that real engagements afford. Third,the measur-
ing instrument is unvalidated: LLM judges exhibit biases and task-dependent
reliability [16,1], so agreement measured on one domain does not transfer to
another.
The contributions of this work are fourfold and transferable to any organisa-
tion operating RAG gates.
First, evidence-first decision semantics deployed in production: a four-state
decision lattice (promote,manual review,block,not evaluable) with per-metric de-
cisionmodesandexplicittreatmentofmissingevidenceandjudgeprotocolerrors
(Sect. 3.3). Second, a stratified beta-binomial gate: per-stratum credible inter-
vals and a probabilistic regression riskP(p B< pA−δ)replace point estimates
(Sect. 3.4). Third, per-engagement judge validation: a meta-evaluation protocol
with balanced golden sets and threshold-free agreement statistics, mandatory
before judge scores may influence decisions (Sect. 3.5). Fourth, a public evalua-
tion at two system boundaries: judge-layer validation on RAGBench (Sect. 5.1),
operating characteristics of the production statistical gate under regression, no
change, and improvement (Sect. 5.2).
Each component builds on established work. The contribution of AGO is the
decision layer that makes missing evidence, judge protocol errors, statistical un-
certainty, and per-engagement judge validity explicit inputs to a release decision,
not hidden assumptions behind a score.
The paper is organized as follows. Sect. 2 reviews related work on RAG
evaluation and LLM-as-a-judge reliability. Sect. 3 presents the AGO framework:
evidence model, layered scoring, decision semantics, statistical gate, and judge
meta-evaluation. Sect. 4 describes the deployed platform and the engagement
workflow. Sect. 5 reports the evaluation: judge-layer validation on RAGBench
and operating characteristics of the statistical gate. Sect. 6 concludes.

AGO AI Quality Gate 3
2 Related Work
RAG evaluation frameworks.RAGAs [6] introduced reference-free, LLM-scored
metrics for RAG (faithfulness, answer relevance, context precision and recall).
ARES [15] trains lightweight judges on synthetic data and calibrates them with
prediction-powered inference over a small human-labelled set. It targets evalu-
ator accuracy on benchmarks; AGO instead embeds evaluator validation inside
an end-to-end decision pipeline with explicit missing-evidence semantics. Open-
source tooling popularised the RAG triad of context relevance, adherence, and
answer relevance, which we adopt as the judge-facing decomposition. RAGAs,
ARES, and TruLens provide scores or evaluator predictions rather than the
evidence-aware release semantics studied here; direct comparison of release deci-
sions would therefore require imposing a common decision policy. Toolkits such
as NeMo Guardrails [14] provide programmable input and output rails; AGO
incorporates comparable local guardrails as one metric provider among several,
subject to the same decision semantics.
Benchmarks for RAG and its evaluators.RGB [2] probes LLM behaviour un-
der noisy or counterfactual retrieval. RAGBench [7] releases 100k (question,
documents, response) traces from 12 datasets across five industry domains, an-
notated with the TRACe schema: continuousrelevance,utilization, andcom-
pletenessscores and a binaryadherencelabel. The labels are produced by LLM
annotators calibrated against human annotations; we therefore refer to them as
reference labelsrather than ground truth. RAGBench frames evaluator quality
as a supervised prediction problem (RMSE for continuous targets, AUROC for
adherence), letting heterogeneous evaluators be compared on equal footing; we
adopt this protocol.
LLM-as-a-judge reliability.LLM judges correlate with human preferences on
some tasks [17,13] but exhibit position and verbosity biases [16], and large-scale
studiesreportagreementvaryingsubstantiallyacrosstasksanddomains[1].This
non-transferability motivates our per-engagement validation: rather than certi-
fying a judge model once, AGO re-measures judge–human agreement for every
deployment domain, using classical agreement statistics [4,11], imbalance-robust
summaries [3], and bootstrap uncertainty [5]. Existing tooling scores answers.
Converting scores into accountable release decisions under incomplete evidence
is the gap deployed systems face, and the gap this work addresses.
3 The AGO AI Quality Gate Framework
3.1 Setting and Evidence Model
Anevaluation casex= (q, E, K∗, T, S, P)consists of a questionq, an optional
expected behaviourE(reference answer, expected termsT, expected sources
S), optional retrieval requirementsK∗, and forbidden phrasesP. A versioned
datasetD={x i}n
i=1is assembled from observed production traces; each case

4 G. Zeloni et al.
carries a categoryc(x)and a criticality flag. Invoking the system under test on
xiyields an answera iand retrieved contextsK i(possibly empty).
A set ofmetric providersmaps(x i, ai, Ki)to metric results: each resultm
carries a name, an optional numeric valuev, an optional thresholdθ, a boolean
pass flag, and the emitting provider. All thresholds, synonym policies, and judge
instructions are versioned in anevaluation profile, never hard-coded: every de-
cision is reproducible from (dataset snapshot, profile snapshot, code version).
The framework distinguishes four evidence types throughout:observed,declared,
inferred(flagged as such), andmissing. Missing evidence must surface as an
explicit gap with an impact statement, never as a default value.
3.2 Layered Scoring
Deterministic checks.The first layer is local, cheap, and exactly reproducible:
answer presence, presence of required retrieved contexts, matching of expected
sources and document versions, detection of forbidden phrases, and matching of
expected terms with normalisation of Unicode, case, and accents plus profile-
versioned synonym groups. Term matching passes when the matched fraction
reachesaprofile-definedratio.Thesechecksprovideajudge-independentbaseline
and catch structural failures early.
Local guardrails.A rule engine inspired by production rail systems [14] evalu-
ates pre- and post-conditions (PII, secrets, prompt injection, forbidden phrases,
JSON and link validity) with per-rule actions{block,warn,log}, timeouts, and
fail-open/fail-closed behaviour. Every rule execution emits a metric result, so a
guardrail timeout is visible evidence rather than a silent skip.
LLM-judge RAG triad.The third layer scores each case on the RAG triad of
context relevancec, adherenceg, and answer relevancerthrough a single con-
strained JSON call to an exchangeable, OpenAI-compatible judge endpoint at
temperature0. A metric passes if its score meets the profile thresholdθ. Two
protocol rules are central. First, scores are accepted only within a small tolerance
of the declared range: values in[−0.01,1.01]are clamped to[0,1]; non-numeric,
non-finite,orgenuinelyout-of-rangevaluesareprovider protocol errors.Themet-
ric is then marked failed with an explanatory comment and the case is routed
tomanual review. Malformed judge output is never coerced to a valid score. A
silent zero is indistinguishable from a measured catastrophic failure, and a silent
clamp could convert an invalid output into a confident pass. Second, adherence
is scored only for cases that require retrieval; for purely conversational turns the
dimension is undefined and omitted rather than imputed.
At run level, the primary grounding measure is theadherence rate: the frac-
tion of evaluated cases that passed the adherence criterion. Stakeholders thus
receive an interpretable proportion, not an average of uncalibrated judge scores;
the raw scores remain per-case diagnostics.

AGO AI Quality Gate 5
3.3 Decision Semantics
LetM(x)be the metric results for casexandF(x) ={m∈M(x) :¬m.passed}
the failed subset. The profile assigns each metric a decision modeµ(m)∈
{observe,review,block};observemetrics are reported but never influence deci-
sions. A fixed, documented setCofcritical metric identifiers(missing required
contexts, failed source or version matching, forbidden phrases, PII, secret, or in-
jection findings, output-format violations) escalates independently of mode. The
item decision is
dec(x) =

not evaluableM(x) =∅,
block∃m∈F(x) :µ(m) =block∨name(m)∈C,
manual reviewF(x)\ {m:µ(m) =observe} ̸=∅,
promoteotherwise,(1)
and run-level aggregation is conservative:
dec(D) =

block∃i: dec(x i) =block,
not evaluable∀i: dec(x i) =not evaluable,
manual review∃i: dec(x i)∈ {manual review,not evaluable},
promoteotherwise.(2)
The setCis a fixed system safety floor: membership inCtakes precedence
over a per-metric profile mode and therefore blocks even if that metric was mis-
takenly configured as observe-only. OutsideC, the versioned evaluation profile
determines whether a failed metric is blocking, review-only, or observational; this
precedence is part of the decision semantics, not a profile-dependent risk prefer-
ence. In deployment, profile-level blocking is reserved for critical violations (e.g.
PII leakage), and every non-promoteoutcome carries machine-readable failure
reasons and is routed to consultant review. Two properties follow.No implicit
pass: absent evidence can only producenot evaluableormanual review, neverpro-
mote.Human-in-the-loop by construction: the gate is an escalation mechanism,
not an autonomous verdict.
3.4 The Stratified Hierarchical Beta-Binomial Gate
AGO models release quality at the operational levels the gate acts on. Item
decisions are grouped into engagement-defined strata. Each stratum receives a
beta-binomial posterior. Run-level quality is a weighted posterior aggregate over
strata, and baseline and candidate runs are compared at the posterior level. The
hierarchy is operational—item→stratum→run→comparison—not a global
partial-pooling Bayesian model. The implementation is deliberately simple, au-
ditable, and profile-configurable.
Point pass-rates on small, stratified evaluation sets are misleading. Per cate-
goryc, we model the item outcomeX i=1[dec(x i) =promote]as Bernoulli with

6 G. Zeloni et al.
ratep cunder a conjugate priorp c∼Beta(α 0, β0)(defaultBeta(1,1)) [9]; observ-
ingk csuccesses amongn ccases gives the posteriorBeta(α 0+kc, β0+nc−kc),
reported with equal-tailed95%credible intervals per stratum. The run summary
is a weighted posterior mean¯p=P
cwcE[pc|data]with weights defaulting to
datasetcoveragew c=nc/N.Sinceevaluationsetsareoftendeliberatelyenriched
with critical cases, coverage weights reflect theevaluation design, not produc-
tion traffic; profiles may therefore supply explicit target weights (traffic shares
or business criticality) when a production-facing estimate is required, and the
report states which weighting is in effect.
For version comparison, letA(baseline) andB(candidate) be runs on the
same dataset and profile. DrawingSMonte Carlo samples of the stratified
weighted rate for each run (fixed, recorded seeds), the regression risk is
ρ=P 
pB< pA−δ|data
,(3)
withδthe minimal regression of practical relevance (default0.03); the statisti-
cal gate returnsblockifρ≥τ b,manual reviewifρ≥τ r(defaults0.8,0.5), and
promoteotherwise.ItsupportstheoperationaldecisionofEq.(2)andneverover-
rides blocking evidence. The two runs are modelled as independent. When they
share items, positive correlation means independenceoverestimatesthe variance
of the difference, so the gate errs towardmanual reviewandblock. This approxi-
mation is conservative by design: an unnecessarymanual reviewcosts consultant
time, while a silently promoted regression costs a production incident. This con-
servative treatment is intentional: when baseline and candidate evidence is too
similar, the gate should expose the uncertainty and route the comparison to
human review rather than silently promote a risky release. The uninformative
default prior is also deliberate. It confines the evidence about a candidate to
the candidate’s own run, rather than anchoring it to earlier versions, and keeps
the audit trail free of hidden information. Engagements with defensible domain
history may supply an informative prior through the versioned profile (α 0, β0
are profile parameters). Likewise,δis an engagement-level risk parameter, not a
constant; stratum-specificδis a natural profile extension. All hyperparameters
live in the versioned profile and are reported with each run. Changing the re-
lease risk appetite means changing versioned thresholds, not the evidence model
or hidden code constants. Sect. 5 evaluates judge behavior under the frozen
meta-evaluation protocol and the statistical gate’s operating characteristics un-
der three controlled data-generating scenarios.
3.5 Per-Engagement Judge Meta-Evaluation
Judge–human agreement is task- and domain-dependent [1,16]; AGO therefore
treats it as a quantity to re-measure per engagement. Before judge-based metrics
mayinfluencegatedecisionsforanewdeploymentdomain,thefollowingprotocol
must be executed and its report attached to the engagement record.
Golden set.A domain golden set with binary human labels per judge dimen-
sion: at least 30 cases as an absolute floor for a smoke-level check, and 100 or

AGO AI Quality Gate 7
more for decision-grade validation; both classes present for every dimension (mi-
nority class≥30%); a deliberate share of borderline cases; independent labelling
by two annotators with adjudication, and inter-annotator agreement reported as
the ceiling against which judge agreement is read.
Statistics.Judge scores are binarised at the operating thresholdθand com-
pared to labels: raw agreement; Cohen’sκ[4,11], reported asundefinedwhen
expected agreement is1(single-class labels) rather than coerced; the Matthews
correlation coefficient, informative under class imbalance [3]; threshold-free AU-
ROC over raw scores; a threshold sweep, sinceθis itself a profile parameter;
and95%bootstrap confidence intervals [5] throughout. A judge-dependent met-
ric may run inblockmode only when its validation report meets the profile’s
minimum lower-bound and maximum interval-width criteria. In practice an en-
gagement starts with judge metrics in review-only mode and graduates them as
the golden set matures. Validation evidence is bound to the exact judge con-
figuration (model identifier, frozen rubric, operating threshold). Hosted judge
models change without notice, so switching judge endpoint or model version
invalidates the report and re-triggers the protocol.
4 Deployment
AGO is implemented as a self-hosted platform (API backend and web console)
deployed at an IT consulting firm and in active use for RAG assessment engage-
ments with enterprise clients. Deployment constraints shaped the architecture.
Client data must not leave the premises: all storage, scoring, and reporting run
insidetheclientorconsultantperimeter.Theonlyoutboundcallisthejudgeend-
point, exchangeable between a cloud API and a self-hosted OpenAI-compatible
runtime for deployments that forbid egress. Observed traces arrive from hetero-
geneous observability stacks and are normalised into a provider-agnostic schema.
Personally identifiable information and secrets are detected and redacted before
a trace can become a dataset case. Evaluation datasets are curated before any
gate run: duplicates, weak cases, coverage gaps, and unresolved privacy findings
are surfaced, and blocking findings prevent the gate from running at all.
An engagement follows a repeatable workflow: import observed conversa-
tions; curate a versioned dataset; select or adapt an evaluation profile (thresh-
olds, guardrail policy, judge policy, statistical policy—all versioned); execute the
per-engagement judge meta-evaluation of Sect. 3.5; run the gate; and deliver a
report. Every decision in the report carries its failure reasons, evidence types,
per-stratum uncertainty, and gaps. Every run persists content-hashed snapshots
of dataset and profile, so a decision remains auditable after either evolves. Non-
promoted cases are reviewed by consultants; the gate escalates, humans decide.
5 Evaluation
We evaluate two claims at their appropriate boundaries. First, RAGBench tests
the judge layer and the frozen meta-evaluation protocol; it is not an end-to-

8 G. Zeloni et al.
end validation of AGO. Second, controlled simulation measures the statistical
gate’s unsafe-promotion risk, false-alarm cost, and improvement throughput. All
gate-level experiments are local and make no external model or benchmark calls.
5.1 Judge-Layer Validation on RAGBench
Engagement data is proprietary, so we evaluate the judge layer on a public
benchmark. RAGBench provides complete observed traces, so it plugs directly
into AGO’s evaluation contract without running any retrieval system. Each in-
stance becomes a case with question, answer, and retrieved contexts. We ask:
how well do the framework’s judge metrics agree with the benchmark’s reference
labels, how does this depend on the judge model, and what does it imply for gate
operating points?
Benchmark and Metric Mapping.The benchmark [7] is organised into 12
componentdatasetsacrossfiveindustrydomains(listedinTable3),withTRACe
reference labels produced by LLM annotators calibrated against human anno-
tations. We map our adherence score to the binaryadherencelabel, compared
by AUROC and by MCC/F1 at the deployed operating thresholdθ= 0.8. Our
context-relevance score is holistic, whereas the TRACerelevancelabel is a
fraction of relevant context spans; the definitions differ, so rank correlations
(Spearman, Kendall) are the primary comparison and RMSE is reported only as
supplementary material with this caveat. Answer relevance has no TRACe coun-
terpart and is excluded; utilization and completeness are not natively produced
by our judge and are left to future work.
Protocol.Allreportednumbersusestratifiedsamplesof100instancespercom-
ponent dataset, balanced by adherence label where possible, with fixed published
seeds.Thesesamplessupportcontrolledcomparison,notpopulation-levelperfor-
mance estimates. Sample manifests (exact instance identifiers) are persisted, so
every judge scores identical instances. Judge responses are cached by (instance,
model, rubric hash). The rubric was frozen before any test-split instance was
scored; no prompt or threshold was adjusted afterwards. Judge model selection
is part of the protocol: candidate judges (gpt-4.1-nano,gpt-4o; temperature
0) were compared on avalidation-split sample of three datasets (N=300), with
MCC at the deployed threshold as the pre-specified primary criterion. The se-
lected judge and the low-cost judge were then run once on thetestsplit of all
12 datasets (N=1200per judge). All statistics carry95%bootstrap confidence
intervals (1,000resamples, fixed seed). Judge protocol errors are excluded from
agreementstatisticsandreportedseparately;noneoccurredinthereportedruns.
Pre-registered sanity audits caught two pipeline defects before the reported
runs: a payload bug silently capping the number of documents passed to the
judge, and duplicate instance identifiers across component datasets. The latter
forcedabortingafirstfull-testattemptanddiscardingitspartialresults.Payload
auditsconfirmthatinallreportedrunseverydocumentofeveryinstancereached
the judge.

AGO AI Quality Gate 9
Table 1.Judge selection on the validation split (3 datasets,N=300per judge;95%
bootstrap CIs). MCC is computed at the deployed thresholdθ=0.8.
Judge AUROC adh. MCC @0.8Spearman rel. Cost
gpt-4.1-nano0.672[0.609, 0.732]0.284[0.176, 0.391]0.279[0.163, 0.386] $0.06
gpt-4o0.787[0.731, 0.837]0.435[0.329, 0.539]0.217[0.103, 0.330] $1.36
Table 2.Test-splitresultsoverall12RAGBenchdatasets(N=1200perjudge,identical
instances;95%bootstrap CIs; overall micro aggregation).
Judge AUROC adh. MCC @0.8Spearman rel. Cost
gpt-4.1-nano0.603[0.570, 0.634]0.170[0.110, 0.221]0.366[0.313, 0.413] $0.39
gpt-4o0.783[0.756, 0.807]0.433[0.386, 0.484]0.298[0.243, 0.351] $9.42
Results.
Judge selection.Table 1 reports the validation comparison. By the pre-specified
criterion (MCC at the operating threshold), gpt-4o was selected for the full test
run. The same protocol makes judge cost-quality trade-offs directly measurable
per engagement.
Judge quality dominates.Table 2 shows the headline result. On identical test
instances, gpt-4o exceeds the low-cost judge by+0.180AUROC and+0.263
MCC at the deployed threshold, with non-overlapping confidence intervals. The
low-cost judge detects non-adherent answers only weakly above chance (AUROC
0.603). Its per-class score distributions overlap heavily; no threshold in a0.1–
0.9sweep yields MCC above0.17, so the deficiency is not a threshold artifact.
Nothing in its operational output signals this: across all runs it produced zero
protocol errors and well-formed, plausible explanations. An earlier pilot on three
datasets (N=300) had yielded AUROC0.563[0.495, 0.631]for the same judge.
That interval includes chance, which is why the framework mandates confidence
intervals before conclusions are drawn.
Reliability is domain-dependent even for the strong judge.Table 3 disaggregates
the selected judge. AUROC ranges from0.620(cuad, legal contracts) to0.883
(expertqa), and at the deployed threshold MCC ranges from0.040(emanual)
to0.618(msmarco): on some domains the production operating point is close
to uninformative even for a frontier judge. This is the empirical core of the
central design requirement: judge quality cannot be certified once and assumed
elsewhere. It must be measured per engagement, with thresholds calibrated per
domain—exactly what the versioned profiles and the meta-evaluation protocol
implement. The relevance rank correlations are low and unstable across datasets
(including negative values), consistent with the definitional mismatch noted in
the metric mapping above; we treat this comparison as exploratory.

10 G. Zeloni et al.
0.4 0.5 0.6 0.7 0.8 0.9
AUROC (adherence / hallucination detection)expertqa
hotpotqa
msmarco
tatqa
techqa
delucionqa
finqa
covidqa
hagrid
pubmedqa
emanual
cuad
chance
GPT-3.5 (publ.)
RAGAs (publ.)TruLens (publ.)
DeBERTa ft (publ.)gpt-4o judge (ours)
Fig. 1.Per-datasetadherenceAUROCoftheselectedjudge(gpt-4o,filledcircles;strat-
ified test samples,N=100per dataset) alongside evaluator results published by the
benchmark authors on full test splits (their Table 3 [7]). Dashed line: chance. Datasets
sorted by gpt-4o AUROC; the comparison across protocols is indicative (Sect. 5.1).
Per-dataset MCC and Spearman values appear in Table 3.
Positioning against published evaluators.The benchmark authors report AU-
ROC for response-level hallucination detection (equivalently, adherence discrim-
ination) on the full test splits for a zero-shot GPT-3.5 judge, RAGAs, TruLens,
and a DeBERTa-large encoder fine-tuned on RAGBench (their Table 3 [7]):
across the 12 component datasets, GPT-3.5 ranges0.51–0.65(mean0.56), RA-
GAs0.52–0.70(mean0.60), TruLens0.40–0.70(mean0.59), and the fine-tuned
DeBERTa0.64–0.87(mean0.79). Under our protocol, the gpt-4o judge (per-
dataset mean0.77, micro0.783) falls within the range reported for the fine-
tuned encoder and above the zero-shot baselines, while our low-cost judge falls
within the range of the earlier zero-shot evaluators. Published RAGBench base-
lines are included for orientation only, not as a protocol-identical comparison:
the published numbers use full test splits and earlier GPT-3.5-backed evaluators,
whereas our judge layer is evaluated on fixed stratified samples of100instances
per dataset under a frozen judge protocol. The ordering nonetheless reinforces
the central point: evaluator quality varies widely and must be measured, not as-
sumed.Theseresultssupportthejudge-layerandmeta-evaluationanalysisunder
the tested benchmark conditions; they do not validate the full release gate.

AGO AI Quality Gate 11
Table 3.Per-dataset test results for the selected judge (gpt-4o,N=100per dataset).
Dataset AUROC (adherence) MCC @0.8Spearman (relevance)
covidqa0.769 0.388 0.009
cuad0.620 0.199 0.201
delucionqa0.800 0.166 0.236
emanual0.630 0.040 0.324
expertqa0.883 0.572 0.400
finqa0.787 0.446 0.233
hagrid0.761 0.319 0.128
hotpotqa0.860 0.530−0.159
msmarco0.859 0.618 0.363
pubmedqa0.665 0.311 0.361
tatqa0.829 0.563−0.053
techqa0.802 0.393 0.583
5.2 Operating Characteristics of the Statistical Gate
We exercise the same statistical-benchmark paths used by the product,build_
statistical_benchmarkandcompare_statistical_benchmarks. Each syn-
thetic run containsNbinary item decisions (N= 20,30,50,100, or200), split
acrossgeneral,domain_specific, andcriticalstrata in proportions0.50,
0.30, and0.20. Their baseline pass probabilities are0.90,0.85, and0.80. Candi-
date probabilities change by−0.05,0, or+0.05in every stratum, defining the
regressed,unchanged, andimprovedscenarios. For each scenario andN, we run
1,000 replicates with data seed 20260709. Baseline and candidate observations
are independent within each replicate. The beta-binomial gate uses aBeta(1,1)
prior,95%credibility, minimum relevant regressionδ= 0.03, block threshold
0.80, 121 posterior samples, and independent posterior streams seeded at 4201
and 4202. Run-level posterior draws use the empirical stratum fractions induced
by the stated allocation.
We compare a naive gate, which promotes when the observed candidate pass
rate is at least the observed baseline pass rate minusδ, with two AGO pro-
files.balanced_defaulthas review threshold0.50;decision_grade, used in
Table 4, has review threshold0.40. Both profiles receive exactly the same item
observations and posterior regression probability in each replicate, so their only
controlled difference is the review threshold. The seed schedule was fixed in an
earlier regression-only pilot of the same generator and reused unchanged when
the two additional scenarios were added; it is not the result of a seed search. It
does not create a paired Bayesian model: baseline and candidate item outcomes,
and their posterior draws, remain independent.
In these synthetic scenarios, promotion of a regressed candidate is an unsafe
promotion. Under no true change, manual review or block is a false alarm. Under
true improvement, promotion measures release throughput. We report all three
outcomes together to expose both the protection and the operational cost of
each policy.

12 G. Zeloni et al.
Table 4.Gate operating characteristics for thedecision_gradeprofile (rates in %).
NNaive promote AGO promote AGO review AGO block Avg.ρ
Regressed
20 41.8 33.0 47.3 19.70.532
30 37.4 35.1 41.3 23.60.538
50 38.1 34.5 40.8 24.70.546
100 38.4 29.9 42.5 27.60.571
200 29.3 22.2 41.6 36.20.638
Unchanged
20 59.7 51.4 38.3 10.30.419
30 56.3 52.7 38.1 9.20.408
50 67.9 60.7 31.7 7.60.356
100 76.6 70.0 25.2 4.80.286
200 82.1 76.2 21.1 2.70.249
Improved
20 76.7 71.5 24.4 4.10.297
30 82.3 78.9 19.6 1.50.245
50 89.7 85.1 13.5 1.40.176
100 97.7 95.9 4.0 0.10.081
200 99.8 99.4 0.6 0.00.028
Table 4 shows a consistent decision-grade reduction in unsafe promotion un-
der true regression. AcrossN= 20–200, the naive gate promotes29.3%–41.8%
of regressed candidates, whereas AGO promotes22.2%–35.1%; the reduction
holds at every testedN. The protection is not free. Under no true change, the
decision-grade profile promotes51.4%–76.2%and sends23.8%–48.6%to review
or block. Under true improvement, it promotes71.5%–99.4%, compared with
76.7%–99.8%for the naive rule. AsNgrows, the average posterior regression
probability separates the scenarios: atN= 200it is0.638for regression,0.249
for no change, and0.028for improvement.
The balanced profile is reported in the accompanying artifact rather than
hidden. Under regression it promotes29.8%–43.9%, which can match or exceed
the naive rate, while under no change it promotes60.6%–83.0%with a17.0%–
39.4%false-alarm rate. Under improvement it promotes77.4%–99.8%. Because
the two profiles share all simulated evidence, the stronger regression protection
ofdecision_gradeis attributable to its lower review threshold, not to different
samples or a different posterior model. These operating characteristics describe
policy trade-offs under the tested data-generating process.
6 Conclusion
AGO AI Quality Gate turns RAG evaluation from score production into an au-
ditable release decision. The RAGBench study shows that judge reliability must

AGO AI Quality Gate 13
be measured under a frozen meta-evaluation protocol, not inferred from well-
formed output. The three-scenario gate experiment exposes the release policy’s
operating characteristics: the decision-grade profile lowers unsafe promotion at
every tested sample size, at an explicit false-alarm and throughput cost.
Deployment and evaluation yielded four operational findings. First, judges
can fail silently: the low-cost judge produced flawless, plausibly explained output
while discriminating barely above chance; onlymeasurement againstlabelled ref-
erences revealed it. Second, confidence intervals change conclusions: theN=300
pilot (AUROC0.563[0.495, 0.631]) could not be distinguished from chance, and
only the full run supported a verdict. Third, judge reliability is model- and
domain-dependent (MCC0.040–0.618at a fixed threshold); thresholds must be
calibrated per engagement, not baked into code. Fourth, missing evidence must
be a first-class outcome; distinguishing “measured as bad” from “not measured”
is what made the failures above visible.
These findings support the central claim: evidence completeness, judge va-
lidity, posterior regression risk, and fixed critical precedence must be first-class
release criteria. Future work covers a paired Bayesian comparison, native eval-
uation of the remaining TRACe dimensions, and repeated-sampling analysis of
per-case variance.
Use of Generative AI.Generative AI tools were used to assist with drafting
and editing the manuscript and with implementing the evaluation harness and
simulation code. All content, code, and reported numbers were reviewed and
verified by the authors, who take full responsibility for them.
Ethical Considerations.The judge-layer study uses the public RAGBench
benchmark. The statistical operating-characteristic study uses synthetic binary
outcomes. No personal, client, or proprietary data are introduced by either gate-
level experiment, and neither experiment calls an external model service. The
deployed framework targets enterprise settings and includes local redaction of
personally identifiable information before traces are persisted; gate outcomes
are decision support with mandatory human review of non-promoted cases, not
autonomous verdicts.
References
1. Bavaresco, A., Bernardi, R., Bertolazzi, L., Elliott, D., Fernández, R., Gatt, A.,
Ghaleb, E., Giulianelli, M., Hanna, M., Koller, A., et al.: LLMs instead of hu-
man judges? A large scale empirical study across 20 NLP evaluation tasks. arXiv
preprint arXiv:2406.18403 (2024)
2. Chen, J., Lin, H., Han, X., Sun, L.: Benchmarking large language models in
retrieval-augmented generation. In: Proceedings of the 38th AAAI Conference on
Artificial Intelligence. pp. 17754–17762 (2024)
3. Chicco, D., Jurman, G.: The advantages of the Matthews correlation coefficient
(MCC) over F1 score and accuracy in binary classification evaluation. BMC Ge-
nomics21(1), 6 (2020)

14 G. Zeloni et al.
4. Cohen, J.: A coefficient of agreement for nominal scales. Educational and Psycho-
logical Measurement20(1), 37–46 (1960)
5. Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman &
Hall/CRC (1994)
6. Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: RAGAs: Automated evalua-
tion of retrieval augmented generation. In: Proceedings of the 18th Conference of
the European Chapter of the Association for Computational Linguistics: System
Demonstrations. pp. 150–158 (2024)
7. Friel, R., Belyi, M., Sanyal, A.: RAGBench: Explainable benchmark for retrieval-
augmented generation systems. arXiv preprint arXiv:2407.11005 (2024)
8. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 (2023)
9. Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., Rubin, D.B.:
Bayesian Data Analysis. Chapman & Hall/CRC, 3rd edn. (2013)
10. Ji,Z.,Lee,N.,Frieske,R.,Yu,T.,Su,D.,Xu,Y.,Ishii,E.,Bang,Y.J.,Madotto,A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Computing
Surveys55(12), 1–38 (2023)
11. Landis, J.R., Koch, G.G.: The measurement of observer agreement for categorical
data. Biometrics33(1), 159–174 (1977)
12. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Advances in Neural Information
Processing Systems 33 (NeurIPS). pp. 9459–9474 (2020)
13. Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., Zhu, C.: G-Eval: NLG evaluation using
GPT-4 with better human alignment. In: Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing. pp. 2511–2522 (2023)
14. Rebedea, T., Dinu, R., Sreedhar, M.N., Parisien, C., Cohen, J.: NeMo Guardrails:
A toolkit for controllable and safe LLM applications with programmable rails. In:
Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing: System Demonstrations. pp. 431–445 (2023)
15. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M.: ARES: An automated eval-
uation framework for retrieval-augmented generation systems. In: Proceedings of
the 2024 Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies. pp. 338–354 (2024)
16. Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Kong, L., Liu, Q., Liu,
T., Sui, Z.: Large language models are not fair evaluators. In: Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). pp. 9440–9450 (2024)
17. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li,
Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging LLM-as-a-
judge with MT-Bench and Chatbot Arena. In: Advances in Neural Information
Processing Systems 36 (NeurIPS), Datasets and Benchmarks Track (2023)