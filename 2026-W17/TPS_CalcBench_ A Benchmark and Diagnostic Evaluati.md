# TPS-CalcBench: A Benchmark and Diagnostic Evaluation Framework for LLM Analytical Calculation Competence in Hypersonic Thermal Protection System Engineering

**Authors**: Jinglai Zheng, Chuhan Qiao, Haiming Huang

**Published**: 2026-04-20 08:46:49

**PDF URL**: [https://arxiv.org/pdf/2604.17966v1](https://arxiv.org/pdf/2604.17966v1)

## Abstract
Deploying LLMs as reasoning assistants in safety-critical aerospace engineering requires stricter evaluation criteria than general scientific benchmarks. In hypersonic thermal protection system (TPS) design, inaccurate stagnation-point heat flux or boundary-layer calculations may cause catastrophic design margin violations. Models with numerically reasonable but physically invalid answers are more dangerous than those declining to respond. Current scientific benchmarks only test abstract math and basic physics, evaluate final answers solely, ignore engineering reasoning processes, and cannot detect such critical failures. We propose TPS-CalcBench, the first diagnostic benchmark for closed-form analytical calculations in hypersonic aerodynamics and high-temperature gas dynamics that experienced TPS engineers conduct without simulations. Our contributions include domain-oriented task taxonomy with 4 difficulty levels and 8 categories from Anderson's textbook, dual-track evaluation measuring result accuracy and reasoning quality via an 8-dimension rubric and calibrated judge with human audit to identify right answer wrong reasoning issues, human-AI data pipeline producing 420 high-confidence core items and 810 noise-controlled pre-gating items from 4560 raw data, noise-sensitivity analysis measuring data quality impacts on model ranking, and three diagnostic intervention methods: DFA-TPS fine-tuning, RAG-EQ retrieval grounding and PA-CoT process-aware prompting. Tests on 13 models from 7 groups show wide performance differences (KPI 12.6-87.9), hidden formula selection defects, data-driven rank changes and effective intervention improvements, establishing a complete diagnose-evaluate-intervene framework for safety-critical engineering LLM deployment assessment.

## Full Text


<!-- PDF content starts -->

TPS-CalcBench: A Benchmark and Diagnostic Evaluation Framework for
LLM Analytical Calculation Competence in Hypersonic Thermal
Protection System Engineering
ZHENG Jinglai,1QIAO Chuhan,1and HUANG Haiming1,a)
School of Civil Engineering, Beijing Jiaotong University, Beijing 100044, China
(Dated:)
Deploying large language models (LLMs) as reasoning assistants in safety-critical aerospace engineering de-
mands a qualitatively different standard of evaluation than general scientific benchmarks provide. In hyper-
sonic thermal protection system (TPS) design, a single miscalculated stagnation-point heat flux or erroneous
boundary-layer estimate can propagate silently into a catastrophic design margin violation—making a model
that produces anumerically plausible yet physically unjustifiedanswer more dangerous than one that refuses
to answer. Yet no existing benchmark probes this failure mode: current scientific reasoning suites test either
abstract mathematics or introductory physics, and evaluate only final-answer correctness, leaving the process
of engineering reasoning entirely unscrutinized.
We present TPS-CalcBench, a benchmark and diagnostic evaluation framework foranalytical(closed-
form, formula-derivable) calculation tasks in hypersonic aerodynamics and high-temperature gas dynamics.
To the best of our knowledge, this is the first benchmark specifically targeting this class of calculations
that experienced TPS engineers perform without simulation tools. Our work makes five contributions.(1)
Domain-grounded task taxonomy.We define a rigorous scope covering four difficulty levels (L1–L4)
and eight domain categories grounded in Anderson’s authoritative graduate textbook1, deliberately isolating
analytical tasks from simulation-dependent problems.(2) Dual-track evaluation framework.We design
complementary measurement tracks for outcome correctness and reasoning process quality as independently
measurable axes: a relative-error outcome scorer with unit verification, and an 8-dimension rubric scored by
a calibrated LLM judge with expanded human audit (n= 62calibration items)—enabling detection of the
“right answer, wrong reasoning” failure mode invisible to answer-only evaluators.(3) Trustworthiness-
engineered data pipeline.We develop a human–AI collaborative construction pipeline distilling 4,560
raw candidates into a 420-item high-confidence core set (v4), retaining an 810-item pre-gating set (v2) as
a noise-injection control.(4) Empirical noise-sensitivity analysis.We directly measure the effect of
benchmark data quality on model rankings and KPI estimates by running controlled experiments on both the
curated and pre-gating sets.(5) Diagnostic-informed pilot interventions.We derive three improvement
strategies from the rubric failure taxonomy as proof-of-concept mitigation pilots: Domain-Formula Alignment
fine-tuning (DFA-TPS), Retrieval-Augmented Equation Grounding (RAG-EQ), and Process-Aware Chain-of-
Thought (PA-CoT) prompting. Experiments across 13 models from 7 families reveal a wide competence
spread (KPI 12.6–87.9), systematic formula-selection deficiencies invisible to outcome-only evaluation, data-
quality-induced rank reshuffling, and promising gains from targeted diagnostic interventions—establishing
a practicaldiagnose→evaluate→interveneframework for LLM deployment assessment in safety-critical
engineering.
I. INTRODUCTION
The prospect of using large language models (LLMs)
as automated reasoning assistants in aerospace engineer-
ing is both compelling and sobering4–6. Compelling, be-
cause LLMs can potentially accelerate preliminary de-
sign, automate sanity checks, and democratize access to
specialized domain knowledge. Sobering, because the
failure modes of LLMs in high-stakes engineering set-
tings are qualitatively different—and more treacherous—
than those observed in general-purpose benchmarks. But
sobering does not mean intractable: if we can precisely
characterizewhereandwhymodels fail, we can de-
a)Electronic mail: 24120979@bjtu.edu.cnsign targeted interventions. This paper presents TPS-
CalcBench as an end-to-end framework for doing exactly
that—building a rigorous benchmark, diagnosing model
deficiencies with engineering-grounded rubrics, and de-
riving concrete improvement strategies from the diagnos-
tic evidence.
a. The evaluation gap in professional engineering cal-
culations.The benchmark landscape for scientific rea-
soning has grown rapidly7,8,10,11. Yet existing evalua-
tions share a common architectural limitation: they mea-
sure whether a model produces thecorrect final answer,
not whether it arrived there throughphysically valid rea-
soning. For introductory mathematics or general sci-
ence, this conflation is tolerable—a lucky numerical co-
incidence is unlikely and harmless. For professional engi-
neeringcalculations, itisneither. Anengineerwhotrusts
a model’s stagnation-point heat-flux estimate withoutarXiv:2604.17966v1  [cs.AI]  20 Apr 2026

2
scrutinizing the formula selection and regime assump-
tions is exposed to a systematically dangerous failure
mode: the model’s answer is numerically plausible but
physically unjustified, and no alarm bell rings.
b. Why TPS calculations are uniquely challenging.
Thermal protection system (TPS) design for hypersonic
vehicles sits at the intersection of several cognitively de-
manding disciplines. Computing stagnation-point heat
flux via the Fay–Riddell correlation, for instance, re-
quires the analyst to: (i) identify the applicable flow
regime (M ∞>5, viscous-interaction-dominated), (ii)
select the correct boundary condition formulation (cat-
alytic vs. non-catalytic wall), (iii) correctly apply the
Lewis number correction for chemically reacting bound-
ary layers, (iv) maintain SI unit consistency across a
seven-parameter expression, and (v) verify that the re-
sult is physically plausible given the altitude–velocity
combination. Missing any one of these steps produces
a wrong answer that mayappearcorrect to an evaluator
checking only the final number. The same multi-step,
regime-sensitive, unit-consistent reasoning structure ap-
plies to calculations of boundary-layer transition, oblique
shock thermodynamics, equilibrium chemical composi-
tion,vibrationalnonequilibrium,andradiativetransfer—
makingTPSdesignanaturalstresstestforLLMphysical
reasoning capability.
c. A dual-axis diagnostic imperative.We argue that
evaluating LLMs on professional engineering calcula-
tions requires twoorthogonaldiagnostic axes: (1)out-
come correctness—does the predicted numerical value
agree with the reference?—and (2)process trustwor-
thiness—does the solution path reflect physically valid,
domain-appropriate reasoning? These axes are logically
independent: a model can achieve outcome correctness
through compensating errors, memorized numerical pat-
terns, or lucky dimensional analysis, while a model with
transparent, well-reasoned derivations may make a mi-
nor arithmetic slip. An evaluation framework that col-
lapses these two axes into a single scalar conflates com-
petence with luck and conceals the failure modes that
matter most in engineering contexts.
d. The benchmark construction challenge.Building
adomain-specificbenchmarkthatissimultaneouslyhigh-
coverage, high-fidelity, and reliably scored is itself a non-
trivial research problem. Extracting calculation prob-
lems from graduate engineering textbooks introduces
a cascade of quality hazards: OCR artifacts, figure-
fragment misidentifications, incomplete given-parameter
sets, fallback target specifications, and reference solu-
tions that presuppose simulation infrastructure. Ad-
dressing these hazards with ad-hoc filtering produces
benchmarksthataretechnicallyusablebutwhosequality
characteristics are opaque, making it impossible to know
whether benchmark differences reflect model capability
or data quality variance.A. Contributions
This paper addresses the above challenges through five
principled contributions.
a. Contribution 1: TPS Analytical Calculation
Benchmark.We present TPS-CalcBench, a benchmark
targetinganalytical(closed-form, formula-derivable) cal-
culation tasks in hypersonic aerodynamics and high-
temperature gas dynamics—the class of calculations that
experienced TPS engineers perform for preliminary de-
sign, physical sanity checking, and regime identification,
without recourse to CFD or FEA simulation. To the
best of our knowledge, no prior benchmark specifically
scopes this class of problems at the professional engi-
neering level. The benchmark spans four difficulty lev-
els (L1: single-step substitution through L4: coupled it-
erative reasoning) and eight domain categories (Newto-
nian aerodynamics, shock relations, boundary-layer the-
ory, aerothermal heating, viscous interaction, chemical
equilibrium, nonequilibriumflow, andradiation), sourced
systematically from Anderson’sHypersonic and High-
Temperature Gas Dynamics1. By isolating the analytical
scope, we enable infrastructure-free evaluation of physi-
cal reasoning and multi-step calculation competence.
b. Contribution 2: Dual-Track Engineering Evalua-
tion Protocol.We design a two-track evaluation archi-
tecture that treats outcome correctness and reasoning
process quality as independently measurable quantities.
Track 1 uses relative-error banding with explicit unit ver-
ification, distinguishing four correctness levels. Track
2 employs an 8-dimension rubric aligned with the ver-
ification steps a competent TPS engineer would apply
when reviewing a junior colleague’s calculation: formula
selection, parameter identification, dimensional consis-
tency, arithmetic accuracy, physical plausibility, assump-
tion transparency, result interpretation, and presentation
quality. The rubric is scored by a LLM judge calibrated
against domain-expert human ratings, enabling scalable
yet principled process-level assessment. This dual-axis
design exposes failure modes—including “right answer
via wrong formula,” “plausible answer via undisclosed as-
sumption,” and “domain hallucination”—that are struc-
turally invisible to answer-only evaluators.
c. Contribution 3: Trustworthiness-Engineered Data
Construction Pipeline.We develop a rigorous, au-
ditable pipeline for constructing domain-specific bench-
marks from textbook sources, organized around the prin-
ciple thatbenchmark conclusions are only as credible as
their underlying data. The pipeline combines high-recall
automated extraction (dual rule-based + LLM-assisted
passes), five-stage progressive filtering (deduplication,
rule-based problem validation, automated QC, human
review and adjudication, and formal go/no-go gating),
triple-reviewer checkpoints, targeted repair sprints for
salvageable items, and version-locked dataset freezing
with full provenance metadata. This pipeline reduces
4,560 raw candidates to a 420-item high-confidence core
set verified to meet strict standards of self-containedness,

3
solution verifiability, and scoring robustness. The
pipeline design is explicitly transferable to other domain-
specific benchmarks constructed from technical litera-
ture.
d. Contribution 4: Empirical Benchmark Quality
Sensitivity Analysis.Rather than asserting that data
quality matters, wemeasureit. By maintaining the pre-
gating810-itemset(v2)asanexplicitnoise-injectioncon-
trol alongside the curated 420-item set (v4), and run-
ning identical experiments on both, we directly quan-
tify the effect of benchmark noise on model rankings,
KPI magnitudes, and evaluation conclusions. This con-
tribution provides the field with concrete empirical ev-
idence for quality-gating investment—addressing a con-
cern widely acknowledged in the benchmark construction
literature15,16butrarelyaddressedwithcontrolledexper-
imental evidence.
e. Contribution 5: Diagnostic-Informed Pilot Inter-
ventions.A benchmark is only instrumentally valu-
able if its diagnostic output translates into action-
able improvement signals. We close the diagnostic-
to-intervention loop by deriving three pilot mitigation
strategies from the rubric failure taxonomy.Strat-
egy I: DFA-TPS (Domain-Formula Alignment
Fine-Tuning)constructs a domain-specific SFT dataset
targeting G1 (formula selection) deficits that account
for the largest share of process-score failures.Strat-
egy II: RAG-EQ (Retrieval-Augmented Equation
Grounding)augments model generation with a curated
847-equation knowledge base (EKB) and a regime-aware
retrievertosuppresshallucinatedcorrelations.Strategy
III: PA-CoT (Process-Aware Chain-of-Thought)
instantiates a 7-step Engineering Calculation Protocol
as a structured reasoning scaffold, enforcing multi-step
discipline that separates expert engineering reasoning
from numerical pattern matching. These interventions
are benchmark-informed pilot studies; their transferabil-
ity beyond the present benchmark setting remains to be
validated and is discussed in Section IX.
II. RELATED WORK
a. Mathematical and Scientific Reasoning Bench-
marks.Benchmarks for LLM reasoning span a wide
capability range: from arithmetic word problems
(GSM8K7) to competition-level mathematics (MATH8,
AIME18)andmulti-domainscience(MMLU9,SciEval10).
The closest prior work to ours is SciBench11, which eval-
uates LLMs on college-level physics and chemistry prob-
lems from undergraduate textbooks and includes free-
form solution extraction. However, SciBench’s cover-
age stops at introductory thermodynamics and classi-
cal mechanics—it does not address graduate-level engi-
neering correlations, hypersonic regime-sensitivity, or the
multi-domainknowledgeintegration(e.g., couplingshock
relations with boundary-layer theory and real-gas ther-
modynamics) that characterizes TPS analytical work.Critically, SciBench, like all benchmarks in this category,
evaluates only final-answer correctness, leaving the pro-
cess of engineering reasoning entirely unmeasured.
b. Domain-Specific Engineering Benchmarks.Ef-
forts to evaluate LLMs on engineering problems have
produced benchmarks for structural analysis, circuit de-
sign, and fluid mechanics17. These benchmarks, how-
ever, share two limitations relevant to our setting: they
either require numerical simulation infrastructure (CFD
solvers, FEA packages) as part of the task, or they focus
on qualitative understanding and concept identification
rather than quantitative calculation. TPS-CalcBench
occupies an orthogonal niche: it targets the class of
problems that experienced engineers solve analytically—
without simulation tools—as part of preliminary de-
sign, physical intuition building, and sanity-checking.
This “analytical regime” is a necessary prerequisite for
simulation-augmented work and a distinct test of domain
reasoning capability that has not been benchmarked be-
fore.
c. Process-Level and Verifiable Reasoning Evalua-
tion.The inadequacy of outcome-only evaluation has
been recognized in the mathematical reasoning litera-
ture, motivatingprocessrewardmodels(PRMs)12,13that
score individual reasoning steps. Our approach differs in
two important ways. First, we do not train a step-level
verifier; instead, we definedomain-grounded rubric di-
mensionsthat map to the specific verification actions a
TPS engineer would apply. Second, our rubric captures
engineering-specific quality properties—physical plausi-
bility,regimeappropriateness,unitconsistencyacrossthe
full solution chain—that generic PRMs do not model.
The result is a process evaluation instrument calibrated
to the epistemology of the domain, not just the logic of
mathematical derivation.
d. Benchmark Construction and Data Quality.The
importance of rigorous benchmark construction has been
underscored by evidence of pervasive label errors15, con-
tamination artifacts14, and the sensitivity of evaluation
conclusionstoannotationquality16. Mostbenchmarkpa-
pers acknowledge these concerns in limitations sections
but provide no controlled evidence for their magnitude.
Our noise-sensitivity experiment is a direct response: by
maintaining both a rigorously curated set and a noisier
pre-gating set, we quantify—rather than merely assert—
the effect of data quality on model rankings.
e. Positioning Summary.Table I summarizes the
key distinctions between TPS-CalcBench and represen-
tative prior benchmarks across the five dimensions most
relevant to our contributions.
III. BENCHMARK CONSTRUCTION
Constructingadomain-specificengineeringbenchmark
from graduate textbook sources is not merely a data en-
gineering task—it is atrustworthiness engineeringchal-
lenge. A benchmark item is trustworthy if and only if

4
TABLE I. Positioning of TPS-CalcBench against representative benchmarks.✓= fully supported;∼= partial;×= absent.
Benchmark Prof. Eng. Multi-step Process Unit Quality
domain calc. eval. verif. sensitivity
GSM8K× × × × ×
MATH×✓× × ×
SciBench∼ ∼ × ∼ ×
CFD-related✓× × × ×
TPS-CalcBench (ours)✓ ✓ ✓ ✓ ✓
it satisfies three jointly necessary properties: (P1)self-
containedness(the problem is solvable from the stated
given parameters alone, without implicit reference to fig-
ures, tables, or solution context); (P2)solution verifi-
ability(a unique, numerically definite reference answer
exists and has been expert-confirmed); and (P3)scoring
robustness(the evaluation protocol can reliably distin-
guish a correct from an incorrect model response). Vi-
olating any one property makes benchmark conclusions
unreliable in a way that is invisible to downstream con-
sumers.
Thecentralchallengeisthatautomaticextractionfrom
technical documents introduces systematic violations of
all three properties: OCR artifacts corrupt numerical
parameters (P1), equation-reference misparses produce
problems with no deterministic answer (P2), and fall-
back target names produce unevaluable items (P3). Our
pipeline is specifically designed to detect and remediate
these failure modes at scale, applying progressively more
expensive quality gates as the candidate pool narrows.
Figure 1 illustrates the complete construction pipeline
with item counts at each stage.
A. Data Sources and Scope
All problems are derived from Anderson’sHypersonic
and High-Temperature Gas Dynamics(2nd Edition)1,
a standard graduate-level textbook covering hypersonic
aerodynamics, boundary-layer theory, aerothermal heat-
ing, chemical equilibrium, nonequilibrium flow, radiation
gas dynamics, and related topics. We restrict scope to
analyticaltasks: problems solvable via closed-form for-
mulas, engineering correlations, or standard thermody-
namic/aerodynamic relations, without requiring CFD,
FEA, or Monte Carlo simulation. This restriction serves
two purposes: it isolates the LLM’s reasoning and cal-
culation capability from tool-use infrastructure, and it
ensures that reference solutions are fully verifiable with-
out specialized software.
B. Extraction Strategy and Scope Isolation
a. High-recall dual extraction.We deliber-
ately begin with a permissive extraction strategy
(AUTO_EXTRACT_RELAXED) that prioritizes re-call over precision. A rule-based pass scans for segments
containing calculation-imperative verbs (“calculate,”
“determine,” “find,” and their common Chinese-language
counterparts, here referenced via the transliterations
“jisuan,” “qiu,” and “queding”), numerical parameters
with units, and equation references. An LLM-assisted
second pass identifies structurally valid calculation
tasks that lack explicit imperative markers but contain
sufficient given–target structure for analytical solution.
Crucially, both passes operate independently, and their
union forms the initial candidate pool. This aggres-
sive dual-pass strategy is intentional: false negatives
(missing genuine problems) are unrecoverable at later
stages, whereas false positives (spurious candidates) are
recoverable through downstream filtering. The result is
4,560 initial candidates with an estimated recall above
95% and a precision around 25%.
b. Analytical scope isolation.A defining design
choice of TPS-CalcBench is the explicit exclusion of
simulation-dependent problems. Table II formally de-
finestheinclusionandexclusionboundary, resolvingedge
cases that arise frequently in TPS literature.
Edge cases: equilibrium composition problems are in-
cluded if solvable by algebraic manipulation of equilib-
rium constants; they are excluded if they require nu-
merical solution of a system of nonlinear equations with-
outclosed-formreduction. Viscousinteractionparameter
calculationsareincludedbecausetheyreducetoalgebraic
expressions; full viscous/inviscid interaction solutions re-
quiring iterative field convergence are excluded. The ac-
ceptance/rejection criterion was documented for all 770
Stage 4 rejects to enable auditable review.
This isolation serves two purposes. First, it creates a
well-defined capability boundary: mastery of the analyt-
ical scope is a necessary (though not sufficient) precondi-
tion for simulation-augmented work. Second, it ensures
reference solutions are fully verifiable by domain experts
without specialized software.
C. Data Schema
Each benchmark item is represented as a struc-
tured JSON object with the following fields: a
unique identifier (id), difficulty level (level: L1–L4),
task type (task_type:numerical_calc,derivation,
comparison), domain tags (domains), the complete

5
FIG. 1. Data construction funnel for TPS-CalcBench. The pipeline proceeds from high-recall automatic extraction to increas-
ingly stringent quality gates, culminating in a triple-reviewer adjudication. The final core set (v4) contains 420 fully verified
items; the intermediate v2 set (810 items) serves as a noisier comparison baseline.
TABLE II. Analytical scope boundary for TPS-CalcBench. The “Included” column defines the accepted problem class; “Ex-
cluded” lists rejected types with representative examples.
Included Excluded
Direct formula substitu-
tion (single or multi-step)CFD/FEA solver setup or
post-processing
Textbook engineering
correlations (Fay–Riddell,
Eckert, etc.)Monte Carlo simulation re-
sults
Algebraic derivation of
classical resultsNumerical PDE integra-
tion beyond hand-iteration
Standard thermodynamic
state relations (Rankine–
Hugoniot, etc.)Machine learning or data-
driven prediction tasks
Low-order iterative conver-
gence (max 3–5 steps)Multi-physics coupling
loop execution
Equilibrium composition
via tabulated or algebraic
modelsProblems requiring figure-
reading without stated val-
ues
Radiative transfer with
closed-form approxima-
tions (optically thin/thick
limits)Problems with reference
solutions requiring simula-
tion software
problemstatement(question), alistofgivenparameters
(given, each with name, value, unit, and description), a
list of target quantities (targets, each with name, ex-
pected unit, scoring weight, and description), and meta-
datatracingprovenancetothesourcechapterandextrac-
tion method. The schema enforces self-containedness:
each item must be solvable from itsgivenfields alone,
without reference to external figures, tables, or preceding
examples.D. Multi-Stage Quality Gating Pipeline
The 4,560 raw candidates undergo a five-stage progres-
sive quality gate designed to enforce Properties P1–P3
at increasing stringency, with the most expensive (hu-
man) verification applied only to candidates that have
survived all automated gates. This design minimizes hu-
man annotation cost while maximizing the probability
that retained items are genuinely trustworthy. Table III

6
summarizes each stage’s purpose, mechanism, and yield.
a. Stage 1: Deduplication and OCR Remediation.
The first gate targets two corruption classes that are
universal in textbook extraction: near-duplicate items
arising from repeated chapter examples, and OCR arti-
factsthatcorruptnumericalparameters. Near-duplicates
are identified by Jaccard similarity on tokenized question
text (thresholdτ= 0.85) and merged with provenance
tracking. OCR artifacts—garbled Unicode sequences,
concatenated page numbers, misrecognized mathemati-
cal symbols (e.g.,βrendered as “3”)—are corrected using
a rule-based normalization dictionary derived from in-
spection of the initial candidate pool. This stage reduces
the pool from 4,560 to approximately 2,200, primarily
by eliminating duplicates that arise from the textbook’s
worked-example repetition structure.
b. Stage 2: Semantic Problem Validation.A bat-
tery of rule-based semantic checks eliminates candidates
that are structurally not calculation problems. Three
patterns account for the majority of false positives from
the extraction stage: (i)narrative passages—textbook
discussion paragraphs that contain numerical values but
no imperative calculation request; (ii)figure and table
fragments—axis labels, coordinate sequences, and leg-
endtextmisidentifiedasgiven-parameterlists,detectable
by the presence of≥6concatenated digits or axis-
label patterns (x/cm,y/m); and (iii)equation-reference
misparses—items whose “given” fields consist entirely of
equation-number tokens (e.g.,{6,85}from “Eq. (6-85)”)
rather than physical parameters. A small number of
manually identified problems missed by both extraction
passes are also added at this stage. The resulting set
of 2,250 candidates has been confirmed to contain a gen-
uine calculation task structure, though completeness and
correctness are not yet verified.
c. Stage 3: Automated Scoring-Feasibility QC.
Property P3 (scoring robustness) is verified automati-
cally before investing human review effort. Each can-
didate is checked for: (a)target non-fallback complete-
ness—at least one target must have an explicitly named
quantity with a physically resolvable unit, not a generic
fallback label; (b)given-parameter dimensional suffi-
ciency—the declared given parameters must span the di-
mensional basis required to compute the stated targets;
and (c)level–complexity consistency—the assigned diffi-
culty level must be coherent with the estimated number
of reasoning steps in a reference solution. Items failing
any check are flaggedwarn(borderline, forwarded to hu-
man review with annotations) orreject(unsuitable, re-
moved). This stage passes 1,930 candidates while surfac-
ing 320 items that would have been impossible to score
reliably.
d. Stage 4: Domain-Expert Human Review and Adju-
dication.Surviving candidates are packaged into struc-
tured review batches and evaluated by domain-expert re-
viewers against four criteria that require human physical
judgment: (i) self-containedness and absence of implicit
context dependencies; (ii) completeness of the given-parameter list for a unique analytical solution; (iii) well-
definedness and unambiguity of the target quantities;
and (iv) correctness of the reference solution. Each
item receives a disposition ofaccept,revise(action-
able repair identified, queued for the repair sprint), or
reject. This stage is the primary enforcer of Proper-
ties P1 and P2, reducing the pool to 1,160 accepted or
revision-queued items while eliminating the 770 candi-
dates that fail human-level scrutiny despite passing au-
tomated checks.
e. Stage 5: Triple-Reviewer Go/No-Go Gating and
Targeted Repair Sprint.The final gate applies a formal
triple-reviewer go/no-go protocolthat evaluates dataset
readiness across three independent dimensions:data
quality(field completeness, solution verifiability, self-
containedness rate),evaluation feasibility(answer
extractability from model outputs, unit normalization
coverage), andexperimental reproducibility(split-
manifest stability, pipeline script executability, random-
seed documentation). All three reviewers must inde-
pendently reachgostatus on all three dimensions for
the dataset version to be cleared for experimentation—a
unanimity requirement that prevents premature release
driven by any single reviewer’s optimism.
The initial go/no-go checkpoint revealed critical defi-
ciencies in the revision queue: a cluster of items with
structurally incomplete given-parameter sets that re-
quired expert reconstruction rather than minor copy-
editing. Rather than expanding the repair budget in-
definitely, we applied aminimal repair sprint—a time-
boxed, effort-bounded remediation pass targeting only
the highest-yield salvageable items. After repair and re-
gating, the final core set (v4(420)) contains 420 items
that have passed all five stages and received explicit
triple-reviewer clearance. The pre-sprint set (v2(810),
810 items) is deliberately retained as a noise-injection
control for the sensitivity analysis in Section VE.
E. Task Taxonomy
TPS-CalcBench items span four difficulty levels and
eight domain categories, as summarized in Table IV.
F. Data Split and Leakage-Controlled Evaluation Design
Astandardrandomtrain/testsplitwouldcreateadata
leakage risk specific to textbook benchmarks: items from
the same chapter share structural patterns, notation con-
ventions,andintermediateresults. Amodelthathasseen
Chapter 6 examples in its training context is advantaged
on any Chapter 6 test item, independently of its actual
physical reasoning capability.
We address this with asource-grouped stratified split:
all items originating from the same textbook chapter
are assigned to the same partition, ensuring that test
items are never from chapters represented in the training

7
TABLE III. Pipeline stage summary. Each stage enforces a subset of the trustworthiness properties P1–P3. “Cost” reflects the
dominant resource type required.
Stage Primary mechanism Properties enforced Output Cost
S1: Dedup + OCR Jaccard similarity, norm. dict. P1 (partial)∼2,200 Compute
S2: Semantic validation Rule-based pattern matching P1, P2 (partial) 2,250 Compute
S3: Automated QC Dimensional + label checks P3 1,930 Compute
S4: Human review Expert adjudication P1, P2 (full) 1,160 Human
S5: Go/no-go + repair Triple-reviewer gate P1, P2, P3 (full)420Human
TABLE IV. Task taxonomy of TPS-CalcBench. Difficulty levels are defined by the number of reasoning steps and prerequisite
knowledge depth.
Level Label Definition Count (v4)
L1 Single-step Directformulasubstitutionwithgivenparameters 170
L2 Multi-step Chained calculations requiring 2–4 intermediate
results120
L3 Cross-domain Requires combining results from multiple physical
domains80
L4 Advanced Requires iterative or coupled solution strategies 50
Total 420
Domain Scope Count (v4)
Newtonian Theory Surface pressure via Newtonian/modified Newtonian methods 100
Shock Relations Normal/oblique shock properties, Rankine–Hugoniot 50
Boundary Layer Laminar/turbulent BL thickness, skin friction, similarity solu-
tions70
Aerothermal Heating Stagnation-point heat flux, Fay–Riddell, engineering correla-
tions70
Viscous Interaction Pressure/displacement interaction parameters 30
Chemical Equilibrium Equilibrium composition, equilibrium speed of sound 30
Nonequilibrium Flow Finite-rate chemistry, vibrational relaxation 40
Radiation Radiative transfer, optical thickness, emission/absorption 30
or development sets. Within this source-grouping con-
straint, we apply level-aware stratification to maintain
theL1/L2/L3/L4difficultydistributionacrosspartitions.
The final split is recorded in a version-locked manifest
(run_manifest.json) that documents item IDs, split as-
signments, chapter provenance, and the random seed—
enabling exact reproduction of any downstream experi-
ment without access to the original extraction logs.
IV. EVALUATION PROTOCOL
a. Design philosophy.We ground our evaluation de-
sign in a simple epistemological observation: in engineer-
ing practice, the trustworthiness of a calculation depends
on twologically independentfactors. The first isnumeri-
cal accuracy—whether the predicted value is close to the
ground truth. The second isreasoning validity—whether
the solution path that produced the value is physically
sound, dimensionally consistent, and explicitly justified.
These factors are independent because (a) a model can
reach a correct answer via wrong reasoning (compensat-
ing errors, memorized patterns, accidental algebraic can-cellation), and (b) a model can produce a well-reasoned,
physically transparent derivation that contains a minor
arithmetic slip.
An evaluation framework that measures only numer-
ical accuracy cannot distinguish these cases. In the
TPS engineering context—where the consequence of mis-
placed confidence in a wrong derivation can propagate to
a thermal protection margin failure—this distinction is
not merely academic. Our dual-track protocol therefore
treats the two axes asseparately measurable and inde-
pendently informative, combining them into a composite
score only at the final aggregation stage.
A. Track 1: Outcome Correctness
For each target quantity in a benchmark item, the
model’s predicted numerical value is compared against
the reference value using relative error:
ϵrel=|ypred−yref|
|yref|(1)
We use relative error rather than absolute error be-

8
cause TPS quantities span many orders of magnitude
(stagnation pressures in kPa vs. radiative heat fluxes in
W/m2), making a uniform absolute threshold physically
meaningless. The error is mapped to a four-level cor-
rectness band that reflects engineering interpretation of
numerical proximity (Table V).
When a problem has multiple target quantities (e.g.,
both wall shear stressτ wand total drag forceD f), the
item-level outcome score is the weight-averaged sum of
individual target scores, with weights specified in the
dataset schema.
a. Unit Verification.Unit correctness is evaluated
independently as a binary indicator. The predicted unit
string is canonicalized (e.g., “Pa”≡“N/m2”, “kW/m2”≡
“1000 W/m2”) and compared against the reference. A
unit mismatch doesnotoverride the numerical score but
isreportedseparately, asitrevealsadistinctfailuremode
(dimensional reasoning vs. arithmetic accuracy).
b. Answer Extraction.Model outputs are free-form
text. Numerical answers and units are extracted by a
combination of regex patterns targeting common formats
(e.g., “The wall shear stress isτ w= 3.45×103Pa”) and a
lightweight LLM-based extractor as fallback. Extraction
failures are flagged and manually resolved; items with
unresolvable extraction are excluded from scoring with
transparent reporting.
B. Track 2: Process Quality via Engineering Rubric
Assessment
a. Rubric design rationale.The 8 rubric dimensions
were derived through a top-down analysis of the verifica-
tion workflow that an experienced TPS engineer applies
when reviewing a calculation. We asked:“If a senior en-
gineer wanted to determine whether a junior colleague’s
calculation can be trusted, which specific properties of the
solution would she check?”The answer defines a nat-
ural decomposition into verification-relevant quality di-
mensions:
•A reviewer first checks whether thecorrect govern-
ing equationswere selected for the flow regime (G1:
Formula Selection)—this is the most consequential
check, because a wrong formula yields systematically
wrong results regardless of arithmetic skill.
•She then verifies that allgiven parameters were cor-
rectly read and applied(G2: Parameter Identifica-
tion), since parameter transcription errors are the sec-
ond most common failure mode in hand calculations.
•She checks thatunits are consistent throughoutthe so-
lution chain (G3: Unit Consistency)—dimensional
inconsistency in multi-step TPS calculations often pro-
duces plausible-looking but wrong intermediate values.
•She spot-checks thearithmetic and algebra(G4: Cal-
culation Accuracy).
•She applies physical intuition to verify thatintermedi-
ate and final values are physically plausiblegiven the
stated conditions (G5: Physical Plausibility)—thiscatches regime violations (e.g., applying perfect-gas re-
lations at Mach 22) that are invisible to purely alge-
braic review.
•She assesses whether the analystexplicitly stated
and justified key assumptions(G6: Assumption
Transparency)—unstated assumptions are the pri-
marysourceofsilenterrorsincollaborativeengineering
workflows.
•She evaluates whether theresult was interpreted in
physical context(G7: Result Interpretation), dis-
tinguishingacalculationassistantfromamerenumber-
cruncher.
•Finally, she assesses whether the solution isorganized
and legiblefor downstream use (G8: Presentation
Clarity).
This mapping from engineering review practice to
rubric dimensions makes the assessment instrument
domain-grounded rather than generically academic. The
weights in Table VI reflect the relative consequence of
each failure mode: G1 (formula selection) carries the
highest weight (0.20) because an incorrect governing
equation is not patchable; G6 (assumption transparency)
carriesthelowest(0.05)becauseitaffectstrustworthiness
rather than correctness.
Each dimension is scored on a 3-point scale (0: absent
or incorrect; 1: partially correct; 2: fully correct):
The overall rubric score for an item is computed as:
Srubric =8X
i=1si
2·wi×100(2)
wheres i∈ {0,1,2}is the raw score for dimensioniand
wiis the corresponding weight (Table VI), yielding a
score in[0,100].
b. Calibrated LLM Judge.Rubric scoring at bench-
mark scale requires an automated judge. We use a
Gemini-3-Pro-Preview judge provided with four inputs:
the problem statement, the authoritative reference solu-
tion, the model’s output, and a structured rubric prompt
that specifies scoring criteria, provides anchor examples
at each score level (0/1/2) for each dimension, and in-
structs the judge to produce structured JSON output
with per-dimension scores, a brief justification for each
score, and a list of major error tags. The judge is deliber-
ately prompted to evaluaterelative to what a well-trained
TPS engineer would consider acceptable, grounding its
assessmentsindomain-appropriatestandardsratherthan
generic correctness.
c. Human calibration and audit.LLM judges intro-
duce systematic biases that must be empirically charac-
terized rather than assumed away. We conduct a two-
phase validation with an expanded calibration sample.
Phase 1: Pilot calibration.A domain ex-
pert independently scored 62 items (approximately
20–22 per task type:numerical_calc,derivation,
comparison_analysis) across all 8 dimensions, drawn
from a stratified random sample of the evaluation run.
Inter-rater agreement between human and judge was

9
TABLE V. Track 1 outcome correctness bands from relative errorϵ rel.
Band Criterion Score
Exactϵ rel<1%1.0
Acceptable1%≤ϵ rel<10%0.7
Order-Correct10%≤ϵ rel<50%0.3
Wrongϵ rel≥50%0.0
TABLE VI. Process quality rubric dimensions.
Dim. Name Assessment Criterion Weight
G1 Formula Selection Correct governing equations identified and applied 0.20
G2 Parameter Identification All given parameters correctly extracted and used 0.15
G3 Unit Consistency Consistent unit system maintained throughout 0.15
G4 Calculation Accuracy Arithmetic and algebraic operations correct 0.15
G5 Physical Plausibility Intermediate and final values physically reasonable 0.10
G6 Assumption Justification Key assumptions stated and justified 0.05
G7 Result Interpretation Final answer interpreted in physical context 0.10
G8 Presentation Clarity Solution logically organized and clearly presented 0.10
computed per dimension using weighted Cohen’sκ(Ta-
ble VII).
Agreement is highest on G1 (Formula Selection,κ w=
0.78) and G2 (Parameter Identification,κ w= 0.82)—the
dimensions that admit the most objective binary assess-
ment. Moderate discrepancies appear on G4 (Calcula-
tion Accuracy,κ w= 0.71) and G5 (Physical Plausibility,
κw= 0.68), where the judge shows a systematic positive
bias of +0.08 to +0.11—meaning the judge is slightly
more lenient than human reviewers on numeric accuracy
andphysicalregimechecking. Thisbiasisconsistentwith
the known tendency of LLM judges to reward structured,
well-organized outputs regardless of content correctness.
Phase 2: Bias analysis and sensitivity.Using
human-only scores for the 62-item sample, we compute
therankorderingofthe6modelsevaluatedonthissubset
and compare it against judge-only rankings. Spearmanρ
between human ranking and judge ranking on this sam-
ple is 0.94 (p <0.01), confirming that while absolute
KPI values may shift under improved calibration (esti-
mated shift≤3–5 KPI points based on the observed bias
magnitudes), the relative model ordering is stable. Items
where judge and human disagree by≥2points on any
dimension (n= 7, 11% of calibration items) are adju-
dicated by a third reviewer; final adjudicated scores are
used in all reported results. The calibration record is
stored as a versioned JSONL file for reproducibility and
public audit.
Limitation acknowledgment.The 62-item calibra-
tion sample, while substantially larger than the 3-item
pilot, is still limited relative to the full 420-item test
set. We recommend treating absolute KPI values as
estimates with±3–5 point systematic uncertainty at-
tributable to judge calibration; relative rankings and
intervention-induced gains are expected to be more re-
liable.C. Error Taxonomy
Beyond aggregate scores, we classify model errors into
a structured taxonomy to enable diagnostic analysis:
(i)Formula Error: Wrong governing equation se-
lected or applicable conditions violated.
(ii)Parameter Error: Incorrect extraction of given
values, use of wrong constants, or missing parame-
ters.
(iii)Unit Error: Dimensional inconsistency or incor-
rect unit conversion.
(iv)Arithmetic Error: Correct formula and parame-
ters but computational mistake.
(v)Physical Plausibility Error: Result violates
known physical bounds (e.g., negative tempera-
ture, supersonic boundary-layer thickness exceed-
ing body length).
(vi)Hallucination: Fabrication of formulas, con-
stants, or physical laws not present in established
theory.
Each scored item is tagged with its primary error
type(s), enabling analysis of error distributions across
models, difficulty levels, and domains.
D. Composite KPI Definition
To produce a single, comparable performance indica-
tor per model, we define theKPI(Key Performance In-
dicator) as the mean of per-item rubric-weighted overall

10
TABLE VII. Judge–human agreement (κ w, weighted Cohen’sκon 0–2 scale) and Spearmanρacross rubric dimensions for the
62-item calibration sample. “Bias” indicates the direction of systematic judge offset relative to human raters (+= judge rates
higher).
Dim. Nameκ wSpearmanρBias
G1 Formula Selection 0.78 0.81−0.04
G2 Parameter Identification 0.82 0.86+0.02
G3 Unit Consistency 0.80 0.83−0.01
G4 Calculation Accuracy 0.71 0.74+0.08
G5 Physical Plausibility 0.68 0.71+0.11
G6 Assumption Justification 0.72 0.76+0.06
G7 Result Interpretation 0.74 0.77+0.09
G8 Presentation Clarity 0.76 0.79+0.04
Overall 0.75 0.79+0.05
scores across all successfully judged items:
KPI(m) =1
|Jm|X
j∈JmS(j)
rubric (3)
whereJ mis the set of items for which modelmpro-
duced a valid predictionandthe rubric judge returned a
scorable response (statusokorok_partial), andS(j)
rubric
is the rubric score defined in Equation (2).
KPI is aprocess-centered summarythat reflects rubric
quality across all eight dimensions. It isnota replace-
ment for Track 1 outcome metrics: a model that con-
sistently produces correct numerical answers via well-
reasoned derivations will naturally score well on KPI
(through G4 and G1), but a model that achieves nu-
merical accuracy through compensating errors or mem-
orized patterns may earn a high Track 1 outcome score
while receiving a low KPI due to poor G1/G5/G6 rat-
ings. ThedeliberateseparationofKPI(TrackC,process-
centered) from Track A (outcome-centered) is what en-
ables the benchmark to detect the “right answer, wrong
reasoning” failure mode documented in Section VI. The
standalone Track 1 metrics (Exact/Acceptable/Order-
Correct/Wrong bands and unit correctness rate) are re-
ported separately in Table VIII and in per-item JSONL
outputs; they carry independent diagnostic information
that KPI alone does not capture.
We note that KPI is computed only oversuccessfully
judgeditems. Models with lower prediction success rates
or judge coverage are scored on fewer items, which may
introduce selection bias. We report Pred OK% and
Judge Cov% alongside KPI to enable the reader to as-
sess comparability.
V. EXPERIMENTS
A. Experimental Setup
a. Models.We evaluate 13 text-only LLMs span-
ning multiple families and capability tiers, including
GPT, Claude, Gemini, DeepSeek, GLM, Doubao, andMiniMax. All 13 models are evaluated; results for
all models appear in Tables VIII and X. For ranked
comparison and intervention experiments, models with
prediction success rate below 80% are excluded from
the ranked leaderboard, as their aggregate KPI re-
flects item-selection effects rather than capability (i.e.,
the 20%+ unanswered items skew the mean). This
exclusion is applied post-hoc and is fully transpar-
ent: the 6 excluded models (3 due to API failures:
Doubao-Seed-1.6-thinking,gpt-5,kimi-k2.5; 3 due
to low prediction coverage:deepseek-v3.2-huawei,
deepseek-r1-huawei,qwen3-32b-meituan) are identi-
fied and their exclusion reason is noted. Low prediction
coverage is itself a capability signal—models that fail to
produce a parseable answer on>20% of items have a
systematic extraction or compliance gap worth reporting
separately. All reported results are from thetext-only
inference mode under a unified zero-shot prompt and ex-
traction/scoring pipeline.
b. Tool-augmented baselines.To assess whether
TPS calculation errors stem primarily fromreasoning
gaps(wrong formula, wrong regime) orarithmetic gaps
(correct reasoning, computational mistake), we evaluate
two tool-augmented conditions on representative models
(gpt-5.2,aws.claude-sonnet-4.5):
•LLM + Calculator: The model is prompted to
output a step-by-step solution with explicit numer-
ical expressions; a Python interpreter executes each
arithmeticstepandreturnstheresult. Thisisolates
formula and reasoning failures from arithmetic fail-
ures.
•LLM + Formula Sheet (RAG-light): The
model receives a one-page domain formula refer-
ence sheet covering the top-20 TPS formulas (by
frequencyinthebenchmark), prependedtothesys-
tem prompt. This approximates lightweight RAG
without retrieval infrastructure.
Results from these conditions are reported in Table XI
to contextualize the source of errors and the headroom
available from each intervention type.
c. Prompting.All models receive a standardized
system prompt specifying the task context (TPS engi-

11
neering calculation), output format requirements (final
answer with units clearly marked, solution steps shown),
and the instruction to work in SI units unless otherwise
specified. No few-shot examples are provided, ensuring a
zero-shot evaluation.
d. Infrastructure.Experiments are ex-
ecuted via a serial evaluation pipeline
(scripts/31_run_serial_eval_with_rubric_judge.py)
and a parallel orchestrator
(scripts/33_run_parallel_models.py). The pipeline
queries each model, collects predictions, runs answer
extraction, applies scoring, invokes the rubric judge,
and aggregates outputs into per-model directories with
predictions, judgments, summary JSON, leaderboard
CSV, and run manifests.
e. Datasets.We report results on two dataset con-
figurations: the primary high-confidence core set (420
items) and the noisier comparison set (810 items). The
set is the sole basis for model ranking; the noisier com-
parison set is used exclusively for the noise-sensitivity
analysis (Section VE).
B. Main Results: Dual-Track Evaluation
We present results on three complementary tracks to
demonstrate the diagnostic value of dual-axis evaluation.
a. Track A: Outcome Correctness.Table VIII re-
ports per-model outcome scores on the core set, disaggre-
gated into Exact, Acceptable, Order-Correct, and Wrong
bands, plus unit correctness rate.
b. Track B: Process Quality.Table IX reports mean
per-dimension rubric scores (0–2 scale) for models with
≥80% judge coverage. This track reveals model-specific
failure signatures invisible to outcome-only evaluation.
Keyobservations: (1)G6(AssumptionJustification)is
systematically the weakest dimension across all models,
indicating that assumption transparency is a persistent
failuremoderegardlessofcapabilitytier. (2)Thefrontier
models (gpt-5.2, glm-5) show notably higher G5 (Phys-
ical Plausibility) scores than the strong tier, suggesting
this dimension discriminates between models with gen-
uinely internalized physical intuition vs. formula lookup.
(3)gemini-3-flash-previewshowsadistinctivefailurepat-
tern: relatively intact G2/G3/G8 scores but severely de-
graded G1, G5, and G6, consistent with a model that can
organizeacalculation structurebutfrequentlyselectsthe
wrong governing equation.
c. Track C: Composite KPI.Table X reports the
composite KPI summarizing both tracks for ranked com-
parison.
The 12 ranked models exhibit a wide KPI spread
(12.6–87.9), forming three tiers: (i)frontier(KPI
>80):gpt-5.2andgpt-5.1; (ii)strong(KPI
62–80):glm-5,deepseek-v3.1, Claude family,
Doubao-Seed-1.8, andglm-4.7; and (iii)low(KPI<
50):gemini-3-flash-previewand MiniMax variants.
Six models were excluded: three due to API failures(Doubao-Seed-1.6-thinking,gpt-5,kimi-k2.5)
and three due to low prediction coverage (<80%)
(deepseek-v3.2-huawei,deepseek-r1-huawei,
qwen3-32b-meituan).
C. Tool-Augmented Baseline Results
The calculator augmentation yields modest KPI gains
(+2.1 for gpt-5.2, +2.7 for claude-sonnet), concentrated
in G4 (Calculation Accuracy), while leaving G1 (For-
mula Selection) unchanged. This confirms that arith-
metic errors account for only a small share of the to-
tal performance gap for frontier and strong-tier mod-
els; the dominant failure mode is formula selection and
regime identification, not computational mistakes. The
formula sheet augmentation yields larger gains for the
mid-tier model (+5.0 KPI, +0.12 G1) than for the fron-
tier model (+2.3 KPI, +0.05 G1), consistent with the
hypothesis that formula knowledge gaps are more acute
in the strong tier. These results suggest that RAG-EQ
(which provides regime-conditioned retrieval rather than
a fixed formula sheet) should yield larger gains by ad-
dressing harder formula-selection failures.
D. Process Quality Analysis
Table XII summarizes the rubric-judge execution sta-
tus across representative models. For items where the
judge returns a fully parsed response (ok), all 8 dimen-
sion scores are available; for partially parsed responses
(ok_partial), only the overall score is retained.
Judgestatusisheterogeneousacrossmodels: somesys-
tems have highok_partialcounts, indicating that the
judge’s structured JSON output was truncated by token
limits and only the overall score could be salvaged. This
pattern highlights a current instrumentation limitation
of the judge pipeline and should be considered when in-
terpreting dimension-level comparability across models.
E. Noise-Sensitivity Analysis
To quantify the impact of data quality on evaluation
conclusions, we compare model performance (420 high-
confidenceitems)versusv2(810)(810itemsincludingun-
repaired candidates with potential given/target issues).
The comparison confirms that benchmark noise ma-
terially affects both absolute scores and relative order-
ing. All listed models score lower on the noisier v2(810)
set, with∆KPI ranging from +1.58 to +6.58. At the
ranking level,gpt-5.2remains stable at first place, but
thegpt-5.1/glm-5ordering flips between v2(810) and
v4(420), and several mid-tier models shift by one to two
positions.

12
TABLE VIII. Track A: Outcome correctness .Exact:ϵ rel<1%;Acc.:1–10%;OrdC.:10–50%;Wrong:≥50%or missing;
UC%: unit correctness rate. All values in %. Percentages are computed over the full 420-item core set.
Model Exact Acc. OrdC. Wrong UC% Pred OK%
gpt-5.2 51.9 29.0 9.8 9.3 87.6 95.0
gpt-5.1 48.1 27.9 10.0 14.0 85.2 95.7
glm-5 47.1 29.3 13.8 9.8 86.2 100.0
deepseek-v3.1 43.1 28.1 10.0 18.8 81.7 87.6
aws.claude-opus-4.5 41.9 24.3 14.5 19.3 85.0 100.0
Doubao-Seed-1.8 38.6 28.1 14.5 18.8 75.7 100.0
glm-4.7 37.6 24.3 13.6 24.5 76.4 100.0
aws.claude-sonnet-4.5 33.6 28.3 14.8 23.3 72.1 100.0
aws.claude-haiku-4.5 24.3 28.1 18.8 28.8 66.2 100.0
gemini-3-flash-preview 9.8 13.8 24.3 52.1 52.6 100.0
MiniMax-M2.5 0.0 5.0 14.8 80.2 24.3 97.1
MiniMax-M2.1 0.0 4.5 9.8 85.7 18.8 97.9
FIG. 2. Track A Outcome Results: Exact, Acceptable, and Order-Correct bands for all evaluated models on the TPS-CalcBench
core set.
a. Noise effects by error type and rubric dimension.
To understand the mechanism by which data noise af-
fects evaluation conclusions, Table XIV reports the per-
dimension rubric score change (∆G i) , and the change in
error type frequency, for two representative models.
Noise in v2(810) primarily degrades G2 (Parameter
Identification)andG3(UnitConsistency)scores—thedi-
mensions most sensitive to incomplete given-parameter
sets and OCR-corrupted numerical values. G1 (Formula
Selection)andG5(PhysicalPlausibility)arelessaffected,
since these dimensions are more dependent on model
knowledge than on problem statement completeness. At
the error-type level, the noisier set inflates ParameterError and Unit Error rates by 3–4%, while hallucina-
tion rates are less affected. This suggests that bench-
mark noise tends to penalize models’executioncapabil-
ities (correctly reading and applying stated parameters)
while leaving theirknowledgecapabilities (formula selec-
tion, physical intuition) relatively unchanged—an impor-
tant nuance for interpreting cross-set comparisons. This
supports the central claim: quality gating is not merely
cosmetic curation but directly changes the benchmark
conclusions a reader would draw from the leaderboard,
particularly for G2/G3 and parameter-level failure mode
analyses.

13
TABLE IX. Track B: Mean process quality scores per rubric dimension (0–2 scale).Judge Cov%: fraction of items with valid
judge output. Dimensions as in Table VI.
Model G1 G2 G3 G4 G5 G6 G7 G8 Jud.Cov%
gpt-5.2 1.86 1.91 1.89 1.84 1.79 1.63 1.77 1.91 95.4
gpt-5.1 1.75 1.88 1.80 1.71 1.67 1.48 1.67 1.85 80.1
glm-5 1.81 1.83 1.84 1.73 1.73 1.51 1.72 1.87 100.0
deepseek-v3.1 1.72 1.84 1.77 1.69 1.61 1.38 1.64 1.82 80.2
aws.claude-opus-4.5 1.72 1.82 1.80 1.65 1.59 1.33 1.54 1.78 97.7
aws.claude-sonnet-4.5 1.61 1.76 1.70 1.56 1.40 0.91 1.32 1.73 85.0
gemini-3-flash-preview 1.11 1.42 1.37 1.22 1.04 0.61 0.90 1.36 98.3
FIG. 3. Track B Process Quality: Mean scores across the 8-dimension engineering rubric for top-performing models.
TABLE X. Track C: Composite KPI .KPI: mean rubric-
weighted score (0–100), combining outcome correctness (via
G4) and process quality (via G1–G8 rubric). Best inbold,
second underlined . KPI values carry±3–5 point systematic
uncertainty from judge calibration (see Section IVB).
Model KPI↑Pred OK%↑Judge Cov%↑
gpt-5.2 87.8595.00 95.54
gpt-5.1 82.18 95.70 81.39
glm-5 79.82 100.00 100.00
deepseek-v3.1 79.22 87.60 80.65
aws.claude-opus-4.5 77.66 100.00 97.16
Doubao-Seed-1.8 75.69 100.00 86.31
glm-4.7 75.21 100.00 88.35
aws.claude-sonnet-4.5 71.75 100.00 86.37
aws.claude-haiku-4.5 62.76 100.00 94.86
gemini-3-flash-preview 42.00 100.00 97.72
MiniMax-M2.5 12.98 97.10 97.76
MiniMax-M2.1 12.85 97.90 90.32VI. ANALYSIS
A. Error Distribution
Across the items for which the rubric judge re-
turned fully parsedmajor_errorstags, we analyze
the distribution of error types. The most fre-
quent error type isformula_selection(account-
ing for∼18% of all tagged errors), followed by
derivation_missinganddomain_mismatch(∼9%
each). Notably,hallucination-class errors—including
domain_hallucinationandcontext_hallucination—
constitute∼14% of tagged errors, where models fabricate
physical laws or domain-specific constants not present in
established theory.
The overall error tag density is low because a sub-
stantial fraction of judge responses were truncated
(ok_partial) due to token limits, yielding empty
major_errorslists even when the overall score is low.

14
FIG. 4. Track C Composite KPI: Summary of model performance across both outcome and process tracks, forming clear
frontier, strong, and low-capability tiers.
TABLE XI. Tool-augmented baseline results. “Base” = zero-shot text-only; “+Calc” = LLM + Python calculator; “+Sheet” =
LLM + formula reference sheet.∆KPI and∆G1 show changes relative to the zero-shot base.
Model Condition KPI∆KPI G1↑G4↑Exact%↑
gpt-5.2 Base 87.85 — 1.88 1.83 52.4
gpt-5.2 +Calc 89.92 +2.1 1.88 1.96 57.1
gpt-5.2 +Sheet 90.18 +2.3 1.93 1.84 54.8
aws.claude-sonnet-4.5 Base 71.75 — 1.62 1.55 33.3
aws.claude-sonnet-4.5 +Calc 74.45 +2.7 1.62 1.78 38.1
aws.claude-sonnet-4.5 +Sheet 76.75 +5.0 1.74 1.58 40.5
Even under this conservative observation window, the
current pattern—formula selection as the dominant fail-
ure mode—is consistent with the hypothesis that LLMs
struggle with domain-specific equation selection, a capa-
bility critical for engineering applications but underrep-
resented in general training data.
B. Performance by Difficulty Level
Table XV reports mean KPI by difficulty level for the
top models. Performance degrades from L1 to L4 for
most models, with the sharpest drop occurring at L4
(Advanced), where iterative or coupled solution strate-
gies are required.
For most models, the L1→L4 KPI drop ranges from 15
to 25 points. An interesting exception is the GPT family,
where L4 scores are unexpectedly high; manual inspec-
tion reveals that this is partially attributable to a clusterof derivation-heavy items at L4 where the GPT models
produced well-structured reasoning chains that earned
highrubricscoresacrossalldimensions. Themostconsis-
tent degradation is observed foraws.claude-opus-4.5
(81.8→58.4), suggesting that cross-domain knowledge
integration and iterative reasoning remain significant
bottlenecks for this model family.
C. Performance by Domain
Domain-level analysis reveals systematic strengths and
weaknesses across the eight taxonomy categories (Ta-
ble IV). For the top-performing model (gpt-5.2), perfor-
mance is strongest onRadiation(mean KPI 100),Newto-
nian Theory(93.1),andShock Relations(91.1),reflecting
the wide availability of these classical formulas in train-
ing corpora. The weakest category isViscous Interac-
tion(12.0), which involves niche pressure–displacement

15
TABLE XII. Rubric-judge status summary for selected models.
Model Judge OK Judge OK-Partial Judge Scored
gpt-5.2 240 160 400
gpt-5.1 200 140 340
aws.claude-opus-4.5 220 190 410
deepseek-v3.1 190 150 340
glm-5 240 180 420
gemini-3-flash-preview 320 90 410
TABLE XIII. Noise-sensitivity comparison: performance shift from v2(810) to v4(420).∆KPI = KPI(v4)−KPI(v2). Positive
values indicate the model scores higher on the cleaner set; large|∆|indicates sensitivity to data noise.
Model KPI (v2) KPI (v4)∆KPI Rank (v2) Rank (v4)
gpt-5.2 84.20 87.85 +3.65 1 1
gpt-5.1 75.60 82.18 +6.58 4 2
glm-5 78.40 79.98 +1.58 2 3
aws.claude-opus-4.5 74.60 77.80 +3.20 6 5
glm-4.7 70.10 75.32 +5.22 8 7
gemini-3-flash-preview 37.80 42.15 +4.35 10 10
coupling phenomena that are sparsely documented out-
side specialist references. Forglm-5,Chemical Equi-
libriumis a relative strength (94.6), whileNonequilib-
rium Flowis the weakest category (10.0). Both mod-
els struggle withViscous Interaction—the only category
where gpt-5.2 scores below 50—suggesting that this sub-
domainremainsasignificantblindspotforcurrentLLMs.
In contrast,Newtonian Theory, which relies on well-
known, widely published formulas, is consistently among
the highest-scoring categories across all models (mean
86–93). This pattern supports the intuition that LLM
performance on domain-specific engineering calculations
correlates strongly with the density of relevant training
data.
D. Case Studies
We present two illustrative cases demonstrating the
value of dual-track evaluation on the fully self-contained
dataset. For each case we excerpt representative model
outputs verbatim (lightly truncated for space) and anno-
tate rubric dimension scores from the judge.
a. Case 1: Correct Mathematics, Wrong Physical
Regime (L1_0037).Item L1_0037 asks for a proof of
the Mach number independence principle forC p,CL,
andC Dw. The correct derivation must use thehyper-
sonicoblique-shockframework(AndersonCh.4), notthe
moderate-supersonic linearized theory.
Rubric analysis.gpt-5.2demonstrates sophisticated
algebraic manipulation (correctly collapsing the wave
equation) but fails G1 because the Prandtl–Glauert ap-
proachisthewrongtheoryforhypersonicflow. Thejudge
notes: “It chose the linearized small-perturbation frame-
work, which is valid in moderate supersonic flow but
breaks down in the hypersonic limit; the selected equa-tion is not applicable asM ∞→ ∞.” This failure is en-
tirely invisible to outcome-only evaluation: both deriva-
tions conclude “C pisM∞-independent”, and an answer
extractor would score both as correct.
b. Case 2: Domain Hallucination under Ambiguous
Context (L4_0015).ItemL4_0015isareactingbound-
arylayerproblemaskingtoderivethewallmass-injection
boundary condition (ρ wvw= ˙m w) from a fixed control
volume. While the problem statement is self-contained,
the token overlap with “injection” and “boundary” can
triggerhallucinationsofunrelateddisciplinesifthemodel
fails to anchor on hypersonic reacting flows.
Rubric analysis.This case illustrates two
qualitatively distinct hallucination failure modes.
aws.claude-opus-4.5exhibitscontext hallucination:
fabricated equations are physically plausible fluid
dynamics formulas but from the wrong sub-domain
(propulsion rather than ablation).glm-5exhibits
domain hallucination: triggered by domain-agnostic
physical tokens (“porous”, “injection”), it confabulates an
entirely unrelated discipline. Both types score near zero
on G2–G8.
VII. TARGETED IMPROVEMENT STRATEGIES
The diagnostic granularity of TPS-CalcBench—error
type distribution, per-domain weakness profiles, and
rubric-level failure signatures—creates a direct bridge
from evaluation to intervention. Unlike aggregate accu-
racy metrics that provide no actionable signal, our dual-
track protocol identifieswhich capability is failing, for
which domain, at which difficulty level. This section re-
ports three targeted intervention experiments, each ad-
dressing a distinct failure mode identified in Section VI.
For each strategy we describe the intervention design,

16
FIG. 5. Noise Sensitivity Analysis: Performance shift between the noisy v2 set and the high-confidence v4 core set, demon-
strating that benchmark data quality materially affects model rankings.
TABLE XIV. Noise effect disaggregated by rubric dimension and error type for two representative models.∆G i= mean
dimension score on v4 minus v2 (positive = v4 better).∆Err% = change in that error type’s frequency (negative = fewer
errors on v4).
gpt-5.2 aws.claude-sonnet-4.5
Metric∆v4−v2 Direction∆v4−v2 Direction
∆G1 (Formula Sel.) +0.06 v4 better +0.09 v4 better
∆G2 (Param. ID) +0.12 v4 better +0.14 v4 better
∆G3 (Unit Cons.) +0.08 v4 better +0.11 v4 better
∆G4 (Calc. Acc.) +0.04 slight v4 +0.05 slight v4
∆G5 (Phys. Plaus.) +0.02 neutral +0.03 neutral
∆G6 (Assumption) +0.04 slight v4 +0.05 slight v4
∆Parameter Err%−3.1% fewer on v4−4.2% fewer on v4
∆Unit Err%−2.4% fewer on v4−3.8% fewer on v4
∆Formula Err%−1.2% fewer on v4−1.9% fewer on v4
∆Hallucination%−0.8% fewer on v4−1.1% fewer on v4
summarize the experimental protocol, and report the
measured gains under the full evaluation pipeline.
A. Strategy I: Domain-Formula Alignment via Targeted
Fine-Tuning
a. Diagnostic motivation.Formula selection (G1) is
the single most frequent failure mode, accounting for
∼18% of all tagged errors across the full model set (Sec-
tion VIA). Domain-level analysis further reveals that
this failure isdomain-concentrated: Viscous Interaction(mean KPI 12.0 for the top model), Nonequilibrium Flow
(10.0 forglm-5), and Chemical Equilibrium degrade sig-
nificantly relative to Newtonian Theory and Shock Rela-
tions. The pattern is consistent with a training-data den-
sity hypothesis: formulas for well-documented domains
(Newtonian theory, oblique shocks) appear frequently in
publicly available textbooks and lecture notes, while spe-
cialist sub-domains (viscous interaction parameters, vi-
brational relaxation rates) are confined to graduate-level
references that are underrepresented in pretraining cor-
pora. This suggests that targeted supervised fine-tuning
on domain-formula alignment examples could efficiently

17
TABLE XV. Mean KPI (0–100) by difficulty level (n: item count per level, cf. Table IV).
Model L1(n=170)L2(n=120)L3(n=80)L4(n=50)
gpt-5.2 93.5 81.1 77.4 98.0
gpt-5.1 81.0 77.7 86.2 95.0
glm-5 86.8 70.6 82.8 75.0
deepseek-v3.1 82.4 81.5 80.0 67.0
aws.claude-opus-4.5 81.8 79.4 79.4 58.4
glm-4.7 78.0 71.1 82.1 64.0
TABLE XVI. Case 1 (L1_0037): Derivation strategy and formula selection.
Model KPI Derivation approach and output excerpt
glm-5 100Correct hypersonic framework.Applies hypersonic simi-
larityparameterK=M ∞τandshowsthatasM ∞→ ∞,
Cp=1
M2∞f(M∞θ)→2 sin2θ(Newtonian limit).
Output excerpt:“Using the hypersonic similarity param-
eterK=M ∞θ,C p=1
M2∞f(K), and asM ∞→ ∞,
f(K)→2 sin2θ, soC pbecomes independent ofM ∞.”
gpt-5.220Wrong regime: Prandtl–Glauert linearized theory.Ap-
plies the supersonic small-perturbation transformation
y′=βy,β=√M2∞−1, to show the wave equation
becomesM ∞-independent. Valid forM≲3but breaks
down hypersonically where the linearization assumption
fails. G1 (Formula Selection) scored 0/2.
Output excerpt:“Apply a coordinate-stretching transfor-
mation (a Prandtl–Glauert-type supersonic transforma-
tion)y′=βy... substituting yieldsϕ xx−ϕy′y′−ϕz′z′=
0, showing that the equation contains no explicitM ∞
dependence.”
aws.claude-opus-
4.590Correct framework, minor omission.Uses oblique-shock
relations and theM ∞→ ∞limit correctly, but omits
the formal derivation ofC Lindependence. G3 (Deriva-
tion Setup) scored 1/2.
Output excerpt:“AsM ∞→ ∞,C p→2 sin2θ
γ·const =
f(γ, θ), soC pis independent of Mach number and de-
pends only onγand geometry.”
close the capability gap without requiring full retraining.
b. Intervention design.We construct adomain-
formula alignment dataset(DFA-TPS) using thetrain
split of TPS-CalcBench as seed material. For each item
in the training split, we generate up to 3 augmented
examples via structured perturbation: (i)regime shift
variants—the same physical scenario moved to an adja-
cent flow regime (e.g., subsonic→transonic→super-
sonic→hypersonic), requiring selection of the correct
regime-appropriate formula; (ii)formula-swap adversar-
ials—problem statements whose surface form resembles
one formula family but whose physical conditions man-
date a different one; and (iii)assumption-flagging exam-
ples—problems where the “correct” answer includes an
explicit statement that the primary formula is inapplica-
ble and a fallback must be used. Each training example
is annotated with the correct G1 dimension score and
a brief justification, providing both the answer and the
reasoning rationale for the formula selection step.Fine-tuning is applied to a base model (selected
from the strong tier:aws.claude-sonnet-4.5or
deepseek-v3.1) using standard supervised instruction
fine-tuning on the DFA-TPS dataset, with a held-out
validation split drawn from the TPS-CalcBenchdevpar-
tition. We evaluate on the TPS-CalcBenchtestsplit
using the full dual-track protocol.
c. Results.Table XVIII reports the measured KPI
gains from targeted fine-tuning, stratified by domain and
difficulty level. The intervention yields an overall gain of
9.7 KPI points for the representative mid-tier baseline,
with the largest improvements concentrated in Viscous
Interaction (+28.2) and Nonequilibrium Flow (+21.7).
These are precisely the domains where formula-selection
failures were most concentrated in the diagnostic analy-
sis, supporting the claim that targeted domain-formula
alignment can close a substantial share of the observed
gap.

18
TABLE XVII. Case 2 (L4_0015): Domain hallucination severity.
Model KPI Behavior and output excerpt
gpt-5.2 100Correct reacting flow derivation.Constructs a control
volume at the fluid-solid interface and correctly balances
the normal mass flux, then correctly applies this to the
species continuity equation. G1–G8 all scored 2/2.
Output excerpt:“Consider a thin control volume at the
wall. The mass flux entering from the porous wall is
˙mw. The mass flux leaving into the fluid isρ wvw. By
steady mass conservation,ρ wvw= ˙m w. For speciesi,
the convective fluxρ wYi,wvwmust balance the diffusive
flux...”
aws.claude-opus-
4.520Context hallucination.Misinterprets “injection” as a
fuel-injector design problem from propulsion engineering
rather than a boundary-layer ablation/transpiration con-
dition, fabricating injector-plate pressure drop equations.
Output excerpt:“The injection mass flow rate is governed
by the orifice discharge coefficient:˙m w=C dA√2ρ∆P.
Substituting this into the boundary layer...”
glm-50Severe domain hallucination.Misidentified the problem
as structural soil mechanics (porous media seepage) due
to tokens “porous wall” and “mass injection”. G1–G8 all
scored 0/2. Judge: “It abandoned fluid dynamics entirely
and derived Darcy’s law for groundwater flow.”
Output excerpt:“For flow through a porous medium, we
apply Darcy’s Law:v w=−κ
µ∇P. The mass injection
rate into the soil foundation is therefore related to the
hydraulic conductivity...”
TABLE XVIII. Measured KPI gains from domain-formula alignment fine-tuning on the test split. Baseline = zero-shot
aws.claude-sonnet-4.5; Fine-tuned = checkpoint adapted on DFA-TPS. “∆G1” denotes the observed improvement in the
formula-selection dimension score.
Domain Baseline KPI FT KPI∆KPI∆G1
Newtonian Theory 78.2 81.4 +3.2 +0.12
Shock Relations 80.1 83.0 +2.9 +0.10
Boundary Layer 72.4 77.8 +5.4 +0.21
Aerothermal Heating 74.6 79.1 +4.5 +0.18
Viscous Interaction 18.3 46.5 +28.2 +0.74
Chemical Equilibrium 65.2 73.8 +8.6 +0.30
Nonequilibrium Flow 24.7 46.4 +21.7 +0.61
Radiation 60.4 68.9 +8.5 +0.29
Overall 71.8 81.5 +9.7 +0.26
d. Observed residual failures.Fine-tuning on
domain-formula alignment does not fully address G5
(Physical Plausibility) failures, which require the model
to internalizephysical intuition about result magnitude
and regime validityrather than formula lookup. Resid-
ual KPI gaps remain concentrated in L3/L4 items,
primarily attributable to cross-domain coupling errors
and assumption-transparency failures (G6) that are
only weakly affected by supervised formula-alignment
examples.B. Strategy II: Retrieval-Augmented Equation Grounding
(RAG-EQ)
a. Diagnostic motivation.Hallucination-
class errors—domain hallucination and context
hallucination—constitute∼14% of tagged errors
(Section VIA) and produce the most dangerous failure
mode in engineering contexts: fabricated formulas
that are internally consistent but physically incorrect.
The case studies (Section VID) demonstrate that
hallucination severity is uncorrelated with model tier:
even the top-performing model (gpt-5.2) occasionally
applies formulas from an incorrect physical regime, and

19
lower-tier models confabulate entire equation families
when context is ambiguous. This pattern suggests
that providing models withjust-in-time retrieval of
the correct governing equationsfrom an authoritative
knowledge base could suppress hallucination without
requiring parametric knowledge updates.
b. Intervention design.We implement aRetrieval-
Augmented Equation Grounding(RAG-EQ) pipeline
with three components:
1.Equation knowledge base (EKB).We construct a
structured knowledge base of 847 governing equations,
correlations, and physical bounds extracted from An-
derson1, Bertin2, and Gnoffo et al.3. Each entry con-
tains: the equation in LaTeX, the applicable regime
(e.g.,M ∞>5, Re x>106), the physical domain tag,
applicability conditions, and a brief description of fail-
ure modes when the equation is misapplied.
2.Regime-aware retriever.At inference time, the
problem statement is embedded and matched against
the EKB using a hybrid BM25 + dense retrieval
scheme. Retrievalisconditionedontheflow-regimeto-
kensextractedfromtheproblem(Machnumberrange,
temperature regime, boundary-condition type) to pri-
oritize regime-appropriate equations over superficially
similar ones.
3.Augmented prompt construction.The top-kre-
trieved equations (with applicability conditions) are
prepended to the model’s input as a “reference sheet,”
instructing the model to select from the retrieved set
and explicitly justify its selection against the applica-
bility conditions.
c. Results.RAG-EQ targets G1 (Formula Selec-
tion) and G5 (Physical Plausibility) failures simulta-
neously: by anchoring the model to retrieved equa-
tions with explicit applicability conditions, both wrong-
formula selection and regime-violation errors decrease
substantially. Table XIX reports the measured perfor-
mance gains from RAG-EQ applied to representative
models. The largest gains appear in the low and mid
tiers, where hallucination is most frequent, while frontier
models still benefit from modest but consistent improve-
ments.
d. Observed residual failures.RAG-EQ does not di-
rectly address G2 (Parameter Identification) or G4 (Cal-
culation Accuracy) failures, which involve reading com-
prehension and arithmetic rather than formula knowl-
edge. For problems where the correct equation is
multi-step—requiring intermediate derivation of a sub-
expression not present in the EKB—retrieval-augmented
grounding provides partial but incomplete support. The
remaining KPI gap is therefore dominated by parameter
and arithmetic errors rather than hallucination.C. Strategy III: Process-Aware Chain-of-Thought Prompting
(PA-CoT)
a. Diagnostic motivation.The rubric dimension
analysis reveals that G6 (Assumption Transparency) and
G7 (Result Interpretation) are systematically underper-
formed even by strong models: models that achieve high
G1 and G4 scores frequently omit explicit assumption
statementsandphysical-plausibilitychecks. Thispattern
isconsistentwiththeobservationthatstandardzero-shot
prompting incentivizes answer production over reasoning
transparency. Process-level errors of this type—silent as-
sumptions, uninterpreted results, missing regime-validity
checks—are precisely the failure modes that make LLM
outputs dangerous in engineering workflows, even when
the numerical answer is correct.
b. Intervention design.We design aProcess-Aware
Chain-of-Thought(PA-CoT) prompting framework that
explicitly scaffolds the G1–G8 rubric dimensions into the
generation process. Rather than instructing the model to
“show your work” generically, PA-CoT provides a struc-
turedEngineering Calculation Protocol(ECP) as part of
the system prompt:
Engineering Calculation Protocol (ECP):
Step 1 [Regime Identification]:State the
flow regime and verify that it falls within the ap-
plicabilityrangeofanyformulayouplantoapply.
Step 2 [Formula Selection]:Identify the gov-
erningequation(s)andexplicitlyjustifywhythey
apply to this regime.
Step 3 [Parameter Extraction]:List all
given parameters with units, confirm complete-
ness, and flag any missing or ambiguous inputs.
Step 4 [Dimensional Setup]:Perform a di-
mensional analysis to verify unit consistency be-
fore substituting values.
Step 5 [Calculation]:Substitute and compute,
showing intermediate results with units.
Step 6 [Plausibility Check]:Verify the re-
sult against known physical bounds and order-
of-magnitude expectations for this regime.
Step 7 [Interpretation]:State the physical
meaning and engineering significance of the re-
sult.
If any step cannot be completed (missing
information, inapplicable formula, ambiguous
regime), state this explicitly and describe what
additional information would be required.
This ECP maps directly onto rubric dimensions G1–
G7, providing the model with an explicit cognitive scaf-
fold that aligns its generation process with the evaluation
rubric. No fine-tuning is required; PA-CoT is applied en-
tirely at inference time.
c. Results.PA-CoT primarily targets G5 (Physical
Plausibility), G6 (Assumption Transparency), and G7
(ResultInterpretation)—thedimensionsmostaffectedby
implicit reasoning shortcuts. Table XX reports the ob-
served per-dimension score improvements and the result-
ing KPI gains. Because PA-CoT increases output length

20
TABLE XIX. Measured KPI gains from RAG-EQ on test split. “Base” = zero-shot baseline; “RAG-EQ” = retrieval-augmented
result; “Halluc.↓%” = observed reduction in hallucination-class error rate.
Model KPI (Base) KPI (RAG-EQ)∆KPI Halluc.↓%
gpt-5.2 87.85 90.2 +2.4−38%
aws.claude-opus-4.5 77.80 83.6 +5.8−52%
aws.claude-sonnet-4.5 71.81 79.4 +7.6−61%
deepseek-v3.1 79.22 84.1 +4.9−44%
gemini-3-flash-preview 42.15 57.3 +15.2−71%
MiniMax-M2.5 13.12 28.7 +15.6−68%
and structural consistency, it also improves judge cov-
erage by reducing the proportion of underspecified re-
sponses and increasing theokvs.ok_partialratio.
d. Observed residual failures.PA-CoT cannot com-
pensate for parametric knowledge gaps: a model that
does not “know” the Fay–Riddell correlation will not
spontaneously produce it under ECP scaffolding, even
when instructed to perform a regime identification step.
For G1 failures rooted in missing domain knowledge (as
opposed to reasoning shortcuts), PA-CoT often increases
G6 scores by making uncertainty explicit while leaving
G1 largely unchanged. This interaction is diagnostically
valuable: models that respond to PA-CoT with increased
refusal rates in specific domains are revealing knowledge
gaps that fine-tuning (Strategy I) can then address.
D. Combined Strategy Results
The three strategies target complementary failure
modes and can be composed with limited interference:
RAG-EQ supplies equation grounding (suppressing hal-
lucination), PA-CoT scaffolds reasoning transparency
(improving G5–G7), and fine-tuning updates paramet-
ric knowledge (closing domain formula gaps). Table XXI
reports the measured performance of a combined de-
ployment (RAG-EQ + PA-CoT, applied to a fine-tuned
checkpoint) versus the current zero-shot baseline.
The combined-strategy KPI fordeepseek-v3.1(91.4)
approaches the current zero-shot frontier (gpt-5.2: 87.9)
on this benchmark, suggesting that a substantial share of
the observed capability gap is attributable to addressable
knowledge and reasoning-process deficiencies. As a prac-
tical implication, mid-tier models can obtain meaningful
performanceimprovementsfromrelativelylow-costinter-
ventions (RAG-EQ and PA-CoT require no retraining),
while targeted fine-tuning provides additional domain-
specific gains at the cost of supervised data construction.
These gains should be interpreted in the context of the
intervention caveats below.
a. Intervention caveats.We emphasize that Ta-
bles XVIII to XXI report measured intervention results
under the present implementation of DFA-TPS, RAG-
EQ, and PA-CoT. The combined results should still be
interpreted with care: strategy interactions are not guar-
anteed to remain additive across other model families,judge behavior may shift slightly under longer structured
outputs, and EKB coverage is stronger in some domains
than others. These caveats do not change the empirical
conclusionthattargetedinterventionsmateriallyimprove
bothKPIandfailure-mode-specificbehavior, buttheydo
bound the extent to which the reported gains should be
extrapolated beyond the present benchmark setting.
VIII. DISCUSSION
a. Quality over quantity: the 420-item defense.A
natural concern is whether 420 items provide sufficient
statistical power for the granular analyses we report.
We make two arguments. First, 420 items is alower
bound sustained by the trustworthiness constraints, not
an arbitrary target: each item has cleared five pro-
gressive quality gates and received triple-reviewer clear-
ance. Expanding to 2,000 items by relaxing the gate
would introduce items that fail P1–P3 at unknown rates,
making benchmark conclusions uninterpretable with-
out knowing the contamination fraction. Second—and
more compellingly—our noise-sensitivity analysis (Sec-
tion VE) provides direct empirical evidence that the
420-item and 810-item sets produce materially different
model rankings. If noise were harmless we would ob-
serve stable rankings across both sets; we do not. The
data validates the quality-gating investment and argues
against naive expansion. We view the current core set
as ahigh-trustworthiness foundationfrom which prin-
cipled expansion—additional textbook sources, expert-
authored problems, parametric variants—can proceed
with preserved trustworthiness standards.
b. Outcome–process discrepancy: statistical evidence.
A central design claim of TPS-CalcBench is that out-
come correctness and reasoning process quality are in-
dependent diagnostic axes. To validate this empirically
beyond case studies, we perform a quadrant analysis for
allmodelswith≥80%judgecoverage: eachscoreditemis
classified asoutcome-high(Exact or Acceptable band) or
outcome-low(Order-Correct or Wrong), and asprocess-
high(S rubric≥70) orprocess-low(S rubric <70). Ta-
ble XXII reports the fraction of items falling in each
quadrant for representative models.
The HL (“right answer, wrong reasoning”) cells confirm
that 11–14% of items across most model families achieve

21
TABLE XX. Measured per-dimension rubric score improvements from PA-CoT on test split for representative models. Scores
are on the 0–2 scale per dimension;∆represents mean per-item gain. Dimensions not primarily targeted (G1, G2, G4) change
only marginally.
Model G1 G2 G3 G4 G5 G6 G7 G8∆KPI
Baseline (aws.claude-sonnet-4.5)
Base 1.62 1.78 1.71 1.55 1.41 0.92 1.33 1.74 —
PA-CoT 1.64 1.80 1.78 1.58 1.74 1.61 1.72 1.81+7.2
Baseline (deepseek-v3.1)
Base 1.71 1.82 1.74 1.66 1.48 1.03 1.41 1.77 —
PA-CoT 1.73 1.83 1.80 1.68 1.79 1.68 1.78 1.83+6.1
Baseline (gemini-3-flash-preview)
Base 1.10 1.42 1.38 1.21 1.05 0.61 0.88 1.35 —
PA-CoT 1.14 1.45 1.52 1.25 1.48 1.22 1.41 1.54+10.8
TABLE XXI. Measured KPI under combined improvement strategies on test split. “FT” = fine-tuned on DFA-TPS; “RAG-EQ”
= retrieval-augmented grounding; “PA-CoT” = process-aware chain-of-thought.
Model Zero-shot FT only RAG-EQ + PA-CoT FT + RAG-EQ + PA-CoT
aws.claude-sonnet-4.5 71.75 81.5 83.488.7
deepseek-v3.1 79.22 86.3 87.291.4
gemini-3-flash-preview 42.00 54.8 62.172.3
TABLE XXII. Outcome–process quadrant distribution (%
of scored items per model).HH: outcome-high + process-
high (correct reasoning, correct answer);HL: outcome-high
+ process-low (right answer, wrong/opaque reasoning);LH:
outcome-low + process-high (sound reasoning, minor execu-
tion error);LL: both low.
Model HH (%) HL (%) LH (%) LL (%)
gpt-5.2 68.4 12.6 6.3 12.7
glm-5 62.1 14.3 7.1 16.5
deepseek-v3.1 58.9 13.8 8.4 18.9
aws.claude-opus-4.5 55.7 11.9 9.5 22.9
aws.claude-sonnet-4.5 47.6 14.3 11.9 26.2
gemini-3-flash-preview 14.3 9.5 7.1 69.1
correctnumericaloutcomesviaphysicallyunjustifiedrea-
soning. This fraction would be invisible to outcome-only
evaluation and represents the specific failure mode most
dangerous in engineering deployment—a model that pro-
duces a defensible numerical value via an inapplicable
formula or undisclosed assumption. The LH (“sound rea-
soning, minor execution error”) cells (7–12%) show that
some models produce well-reasoned derivations that con-
tain a minor arithmetic slip; these would be penalized
incorrectly by answer-only evaluation. The co-existence
of HL and LH items across all model tiers empirically
validates the independence claim: outcome and process
quality are not redundant measurements.
c. The orthogonality of outcome and process: case
study evidence.The case studies (Section VID) pro-
vide two qualitatively distinct demonstrations: (1) Case
1 shows a model achieving correct algebraic form via
thewrong physical framework, earning high arithmeticscores but zero formula-selection credit; and (2) Case 2
shows domain hallucination scoring near zero on all pro-
cess dimensions while outcome extraction would flag only
anot-a-numbererror. A single outcome score would
conflate both failure modes. This independence also val-
idates the improvement strategy decomposition in Sec-
tion VII: because the two axes are orthogonal, each strat-
egycanbedesignedtotargetaspecificaxiswithouttrad-
ing off the other.
d. From failure taxonomy to actionable interven-
tion.The most important structural property of TPS-
CalcBench relative to prior benchmarks is that its diag-
nostic output isdirectly intervention-addressable. The
G1-dominated error distribution maps cleanly to fine-
tuning on formula-alignment data; the hallucination sig-
nature maps to retrieval-augmented equation ground-
ing; the G5–G7 transparency gap maps to process-aware
prompting. Section VII operationalizes this mapping
with concrete intervention designs and measured inter-
vention results. The combined-strategy gain observed for
mid-tier models—15–30 KPI points depending on base-
line strength—suggests that the principal bottlenecks
in TPS engineering calculation are addressable deficien-
cies in training data coverage and inference-time reason-
ing scaffolding, not fundamental model capacity ceilings.
Thishasasignificantpracticalimplication: organizations
can realistically deploy mid-tier models at near-frontier
quality by combining domain-targeted fine-tuning with
inference-time RAG and structured prompting, without
waiting for the next model generation.
e. What the 9.2% funnel ratio reveals about techni-
cal document understanding.The 4,560→420 retention
rate (9.2%) is not merely a data-efficiency observation—

22
it is a diagnostic signal about the maturity of automated
technical document parsing. Three failure categories
dominate: narrative passages misidentified as calculation
problems (≈44% of rejects), OCR figure-fragment confu-
sions (≈24%), and equation-reference misparses (≈20%).
Each points to a specific gap in current pipelines: se-
mantic intent classification for technical prose, layout-
aware figure/text disambiguation, and cross-reference
resolution in mathematical documents. We release the
rejection-annotated intermediate dataset to support re-
search on these components, which carry implications
well beyond TPS benchmark construction.
f. Deployment implications.Formula selection (G1)
accounts for≈18% of all tagged errors, meaning a sub-
stantial fraction of numerically close predictions are
reached via physically incorrect governing equations—a
failure that is not only invisible to outcome-only evalua-
tion but actively dangerous in preliminary design, where
formula choice determines the entire downstream calcu-
lation structure. Before deploying LLMs as calculation
assistantsinsafety-criticalTPSwork, process-levelverifi-
cation against domain-appropriate rubrics—not just nu-
merical spot-checking—should be considered a minimum
trust standard. The PA-CoT Engineering Calculation
Protocol (Section VIIC) provides a zero-cost path to sur-
facing these failures at inference time, making it a practi-
cal first deployment safeguard even without fine-tuning.
IX. LIMITATIONS AND FUTURE WORK
a. Sample size and statistical power.The 420-item
core set, while rigorously curated, limits the granular-
ity of per-domain and per-level statistical analyses. At
30 items per domain, confidence intervals on domain-
level KPI estimates are wide, and rank orderings within
domains may not be statistically reliable. Future work
will expand the benchmark through additional textbook
sources (Bertin2, Gnoffo et al.3), expert-authored origi-
nal problems, and carefully managed parameterized vari-
ations of existing items, targeting a minimum of 100 ver-
ified items per domain for robust sub-group analysis.
b. Judge reliability and calibration sample.The
LLM judge calibration is based on a 62-item human cal-
ibration sample (Section IVB), preceded by a 3-item pi-
lot (Section D) that revealed large initial judge–human
discrepancies and motivated the expanded calibration.
The 62-item expanded sample yields weightedκvalues
of 0.68–0.82 (acceptable-to-good agreement), with mod-
erate discrepancies on G4 and G5 suggesting system-
atic judge leniency (+0.08–+0.11 points) on those di-
mensions. Absolute KPI values carry an estimated±3–5
point systematic uncertainty from this calibration imper-
fection; relative model rankings and intervention-induced
gains are expected to be more robust.
We acknowledge three specific limitations of the cali-
bration design: (1) the 62-item sample covers∼15% of
the 420-item test set, providing sufficient power to de-tect per-dimension biases but limited ability to detect
interaction effects between dimensions; (2) the calibra-
tion sample was drawn from a stratified random selec-
tion that may not represent the full difficulty spectrum
equally; and (3) the 7 adjudicated disagreement items
(11%) in the expanded calibration are a small basis for
estimating tail-disagreement rates. Expanding ton≥30
per task type per difficulty level, and exploring multi-
judge ensembles with explicit adjudication protocols, are
priorities for the next benchmark version.
c. Intervention transferability.The three improve-
ment strategies (DFA-TPS, RAG-EQ, PA-CoT) are
benchmark-informed pilot studies: each strategy was de-
signed in response to failure modes identified on TPS-
CalcBench, which means their effectiveness may be par-
tially specific to this benchmark’s problem distribution.
External transferability to other TPS calculation sets,
other engineering domains, or real deployment scenarios
remains to be validated. We recommend interpreting the
reported KPI gains as evidence that diagnostic-informed
interventions are more efficient than generic prompting,
while noting that the absolute gain magnitudes may not
replicate in other settings.
d. Language and provenance.Thesourcetextbook’s
Chinese translation introduces potential ambiguity in
problem statements, and the benchmark’s Chinese-
language problem framing may affect models differently
depending on their multilingual training balance. Future
versionswillprovidebilingual(Chinese–English)problem
statements with verified semantic equivalence, enabling
language-controlled experiments.
e. Scope expansion toward simulation-augmented
tasks.The current exclusion of simulation-requiring
tasks is intentional but limits the coverage of the full
TPSengineeringworkflow. AnaturalextensionisaTPS-
SimBenchcompanion benchmark evaluating LLMs’ abil-
ity to set up, interpret, and validate CFD/FEA simula-
tion workflows, complementing the analytical calculation
focus of the present work. The improvement strategies in
Section VII—particularly RAG-EQ—would extend nat-
urally to simulation-augmented settings by incorporat-
ing simulation best-practice guidelines into the knowl-
edge base.
X. CONCLUSION
We have presented TPS-CalcBench, a benchmark and
diagnostic evaluation framework for LLM competence on
analytical calculations in hypersonic TPS engineering.
The work is organized around a central empirical the-
sis:outcome correctness and reasoning process quality
are logically independent diagnostic axes, and conflating
them produces systematically misleading assessments of
engineering capability.This thesis is supported empiri-
cally by the quadrant analysis (11–14% of items across
model families are “right answer, wrong reasoning”), by
three qualitatively distinct case-study demonstrations of

23
outcome-process decoupling, and by the noise-sensitivity
analysis showing that benchmark data quality reshuffles
model rankings and differentially inflates G2/G3-related
error rates.
The paper makes five contributions that form
a coherent progression. Thedomain-grounded
task taxonomyestablishes a clean, infrastructure-
free probe of physical reasoning competence. The
dual-track evaluation protocol—with expanded 62-
item human calibration, formal agreement metrics,
and bias characterization—provides independently cali-
brated measurements of numerical accuracy and reason-
ing process quality. Thetrustworthiness-engineered
pipelineensures benchmark conclusions rest on fully
verified data, with the 4,560→420 funnel providing both
a reusable methodology and a diagnostic signal about
the maturity of automated technical document pars-
ing. Theempirical noise-sensitivity analysispro-
vides controlled experimental evidence that quality gat-
ing materially affects both KPI magnitudes and model
rankings—an insight widely acknowledged but rarely
measured in the benchmark literature. Thediagnostic-
informed pilot interventions(DFA-TPS, RAG-EQ,
PA-CoT) demonstrate that rubric-diagnostic-driven im-
provements are more efficient than generic prompting,
with combined-strategy gains of 15–30 KPI points for
mid-tier models, while acknowledging that transferabil-
ity beyond this benchmark remains to be validated.
Two methodological contributions extend beyond TPS
to the broader benchmark construction community: the
P1/P2/P3 trustworthiness framework with five-stage
gating provides a transferable template for domain-
specific benchmark construction from technical litera-
ture; and the engineering-epistemology-grounded rubric
demonstrates that process evaluation instruments can be
designedwithexplicit,domain-appropriaterationale. We
release the complete dataset, evaluation scripts, DFA-
TPS fine-tuning data, RAG-EQ knowledge base, PA-
CoT prompt templates, calibration records, and human-
review audit trails to support reproducible research and
community extension.
ACKNOWLEDGMENTS
We thank the domain experts who participated in the
triple-reviewer go/no-go gate and the pilot calibration
study. Computing resources for model inference and
rubric judging were provided through internal API in-
frastructure. We acknowledge the use of Anderson’sHy-
personic and High-Temperature Gas Dynamics(2nd Edi-
tion) as the primary source for benchmark construction.
1Anderson, J. D. (2006).Hypersonic and High-Temperature Gas
Dynamics(2nd ed.). AIAA Education Series.
2Bertin, J. J. (1994).Hypersonic Aerothermodynamics. AIAA
Education Series.
3Gnoffo, P. A., Gupta, R. N., & Shinn, J. L. (1999). Conservation
equations and physical models for hypersonic air flows in thermal
and chemical nonequilibrium.NASA TP-2867.4OpenAI (2023). GPT-4 technical report.arXiv preprint
arXiv:2303.08774.
5Anthropic (2024). The Claude 3 model family. Technical report.
6Google DeepMind (2024). Gemini 1.5: Unlocking multimodal
understandingacrossmillionsoftokensofcontext.arXiv preprint
arXiv:2403.05530.
7Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., ... & Schulman, J. (2021). Training verifiers to solve
math word problems.arXiv preprint arXiv:2110.14168.
8Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S.,
Tang, E., ... & Steinhardt, J. (2021). Measuring mathematical
problem solving with the MATH dataset.NeurIPS.
9Hendrycks,D.,Burns, C.,Basart, S.,Zou, A.,Mazeika, M., Song,
D., & Steinhardt, J. (2021). Measuring massive multitask lan-
guage understanding.ICLR.
10Sun, R., et al. (2024). SciEval: A multi-level large lan-
guage model evaluation benchmark for scientific research.arXiv
preprint arXiv:2308.13149.
11Wang, X., et al. (2023). SciBench: Evaluating college-level sci-
entific problem-solving abilities of large language models.arXiv
preprint arXiv:2307.10635.
12Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B.,
Lee, T., ... & Cobbe, K. (2023). Let’s verify step by step.arXiv
preprint arXiv:2305.20050.
13Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N.,
Wang, L., ... & Kalai, A. (2022). Solving math word prob-
lems with process- and outcome-based feedback.arXiv preprint
arXiv:2211.14275.
14Yang, S., et al. (2023). Rethinking benchmark and contamina-
tion for language models with rephrased samples.arXiv preprint
arXiv:2311.04850.
15Northcutt, C. G., Athalye, A., & Mueller, J. (2021). Pervasive
label errors in test sets destabilize machine learning benchmarks.
NeurIPS.
16Biderman, S., et al. (2024). Lessons from the trenches on
reproducible evaluation of language models.arXiv preprint
arXiv:2405.14782.
17Zhang, Y., et al. (2024). Towards LLM-assisted CFD simulation:
Benchmarks and evaluation.arXiv preprint.
18AIME (2024). American Invitational Mathematics Examination.
Mathematical Association of America.
Appendix A: Dataset Schema Example
{
"id": "L2_0001",
"version": "v4.0",
"level": "L2",
"task_type": "numerical_calc",
"domains": ["boundary_layer", "aerothermal_heating"],
"solution_type": "analytical",
"requires_simulation": false,
"question": "Calculate the laminar boundary-layer
thickness at x = 1.0 m on a flat plate ...",
"given": [
{"name": "Ma_inf", "value": 8.0, "unit": "-"},
{"name": "T_inf", "value": 226.5, "unit": "K"},
{"name": "p_inf", "value": 1197, "unit": "Pa"},
{"name": "x", "value": 1.0, "unit": "m"},
{"name": "T_w", "value": 300, "unit": "K"}
],
"targets": [
{"name": "delta", "unit": "m", "weight": 0.5},
{"name": "Re_x", "unit": "-", "weight": 0.5}

24
],
"metadata": {
"source_id": "Ch6::Example_6.1",
"solution_verified": true
}
}
Appendix B: Rubric Judge Prompt
TheLLMrubricjudgereceivesthefollowingstructured
prompt (abridged):
You are an expert aerospace engineering
professor evaluating a student’s solution
to a TPS calculation problem. Score
each of the following 8 dimensions on a
0–1–2 scale based on the criteria below.
Provide a brief justification for each
score.
[Problem statement] ... [Reference
solution] ... [Student (model) output]
...
G1 - Formula Selection (0/1/2): 0 =
wrong or missing equation; 1 = partially
correct (e.g., right family but wrong
variant); 2 = fully correct governing
equation applied.
[...dimensions G2–G8 with similar
detail...]
The full prompt template is available in the supple-
mentary materials.
Appendix C: Evaluation Pipeline Commands
Allexperimentsarereproducibleviathefollowingcom-
mand sequence:
# Step 1: Single-model serial evaluation with rubric judge
python scripts/31_run_serial_eval_with_rubric_judge.py \
--input data/final/tps_calcbench_v4_core.jsonl \
--outdir data/eval/v4_baseline_parallel_runs/<model_name> \--models <model_name> \
--judge-model gemini-3-pro-preview
# Step 2: Multi-model parallel orchestrator
python scripts/33_run_parallel_models.py \
--out-root data/eval/v4_baseline_parallel_runs \
--models "gpt-5.2,gpt-5.1,glm-5,..." \
--max-parallel-models 4
# Step 3: Aggregate results into leaderboard
python scripts/35_aggregate_parallel_eval_results.py \
--eval-root data/eval/v4_baseline_parallel_runs \
--expected-n-samples 420
Therun_manifest.jsonin each model output di-
rectory records model versions, API call timestamps,
dataset checksums, and all configuration parameters.
TheaggregationscriptproducesJSON,CSV,Markdown,
and LATEX leaderboard outputs.
Appendix D: Human–Judge Agreement Report
Appendix E: Data Construction Funnel Statistics
Appendix F: Cross-Reference Verification Table
To facilitate reviewer verification, Table XXV consoli-
dates all key per-model metrics reported across the paper
into a single table. Each column references the originat-
ing table. Discrepancies of≤0.1 KPI points between ta-
bles arise from rounding at different decimal places and
do not indicate data inconsistency.
a. Statistical base note.All Track A percentages
(Exact%, UC%)usethefull420-itemsetasdenominator,
treating items without a valid prediction as “Wrong” /
unit-incorrect. TrackCKPIisthemeanrubricscoreover
|Jm|successfully judged items (Equation (3)); the effec-
tive denominator varies by model and equals Pred%×
Jud%×420/10000(rounded). The noise-sensitivity col-
umn (∆v4–v2) reports KPI(v4)−KPI(v2), where v2 is
the 810-item pre-gating set. Models marked “—” were
not included in the noise-sensitivity analysis due to miss-
ing v2 evaluation runs.

25
TABLE XXIII. Pilot calibration: human expert scores vs. LLM judge on 3 seed samples (one per task type). Human scores
are fromrubric_pilot_calibration_record_v1.jsonl; LLM judge scores are from theok-status judgements where dimension
scores are available. Due to the small pilot sample (n= 3), we report raw mean scores and signed bias (∆ =Judge−Human)
rather than Cohen’sκ.Note:this 3-item pilot informed the design of the expanded 62-item calibration (Table VII); the pilot’s
smallnmeans per-dimension∆values have large uncertainty and should not be interpreted as precise bias estimates.
Dimension Human (mean) Judge (mean)∆Direction
G1: Formula Selection 1.83 2.00+0.17 Judge higher
G2: Parameter Identification 1.01 1.79+0.78 Judge higher
G3: Unit Consistency 0.73 1.79+1.06 Judge higher
G4: Calculation Accuracy 1.08 1.83+0.75 Judge higher
G5: Physical Plausibility 0.95 1.88+0.93 Judge higher
G6: Assumption Justification 1.93 1.79−0.14 Human higher
G7: Result Interpretation 1.02 1.83+0.81 Judge higher
G8: Presentation Clarity 2.22 1.75−0.47 Human higher
Overall score (0–100) 65.1†92.4†+27.3Judge higher
†Overall scores computed via Equation (2) using the per-dimension means and weights from Table VI. The large pilot-stage judge–human gap
(∆ = +27.3) motivated the expanded 62-item calibration (Table VII), which yielded substantially smaller biases (∆≤+0.11per dimension,
overallκ w= 0.75). This reduction is attributable to improved anchor examples and structured prompting in the production judge. All reported
main-text results use the production judge calibrated against the 62-item sample.
TABLE XXIV. Detailed funnel statistics. Each row reports the count entering the stage, the count exiting, and the primary
reasons for removal.
Stage In Out Primary Rejection Reasons
0: Raw Extraction — 4560 —
1: Dedup + Format 4560 2200 Near-duplicates (48%), OCR garbage
(32%), format errors (20%)
2: Rule Filtering 2200 2250 Narrative passages removed (−31), fig-
ure fragments (−18), supplements added
(+99)
3: Auto QC 2250 1930 Fallback-only targets (41%), missing
givens (35%), level inconsistency (24%)
4: Human Review 1930 1160 Ambiguous problems (30%), incomplete
givens(28%), unscorabletargets(22%), in-
correct solutions (20%)
5: GO/NO-GO 1160 420 Triple-gate failures: 810 passed initial
gate, 420 passed final verification

26
TABLE XXV. Cross-reference verification: consolidated per-model metrics . Each column references the source table. “—”
indicates the model was not included in that analysis. All KPI values are on a 0–100 scale; Exact% and UC% are percentages
of the full 420-item set. The “Pn” column reports the total scored items for each model (from Table XV), which serves as the
effective denominator for KPI computation.
Track A Track C Coverage Noise
Model Exact% UC% KPI Tier Pred% Jud%∆v4–v2
Tab. VIII Tab. VIII Tab. X Tab. X Tab. X Tab. XIII
gpt-5.2 51.9 87.6 87.85 Frontier 95.0 95.5 +3.65
gpt-5.1 48.1 85.2 82.18 Frontier 95.7 81.4 +6.58
glm-5 47.1 86.2 79.82 Strong 100.0 100.0 +1.58
deepseek-v3.1 43.1 81.7 79.22 Strong 87.6 80.7 —
claude-opus-4.5 41.9 85.0 77.66 Strong 100.0 97.2 +3.20
Doubao-Seed-1.8 38.6 75.7 75.69 Strong 100.0 86.3 —
glm-4.7 37.6 76.4 75.21 Strong 100.0 88.4 +5.22
claude-sonnet-4.5 33.6 72.1 71.75 Strong 100.0 86.4 —
claude-haiku-4.5 24.3 66.2 62.76 Strong 100.0 94.9 —
gemini-3-flash 9.8 52.6 42.00 Low 100.0 97.7 +4.35
MiniMax-M2.5 0.0 24.3 12.98 Low 97.1 97.8 —
MiniMax-M2.1 0.0 18.8 12.85 Low 97.9 90.3 —
Excluded from ranked leaderboard (see Section VB):
API failures: Doubao-Seed-1.6-thinking, gpt-5, kimi-k2.5
Low prediction coverage (<80%): deepseek-v3.2-huawei, deepseek-r1-huawei, qwen3-32b-meituan