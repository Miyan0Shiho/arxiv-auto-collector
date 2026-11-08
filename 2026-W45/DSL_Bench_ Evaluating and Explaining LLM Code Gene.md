# DSL-Bench: Evaluating and Explaining LLM Code Generation Across Domain-Specific Languages

**Authors**: Zihan Zhou, Mingxi Ye, Yan Gao, Guipeng Wei, Zibin Zheng

**Published**: 2026-11-03 16:37:19

**PDF URL**: [https://arxiv.org/pdf/2611.05168v1](https://arxiv.org/pdf/2611.05168v1)

## Abstract
Large language models (LLMs) are increasingly used in software-engineering workflows, but code-generation benchmarks largely focus on general-purpose languages. Domain-specific languages (DSLs) have distinct syntax, semantics, and validation requirements, making results from individual DSL benchmarks difficult to compare. We present DSL-Bench, an executable benchmark and analysis framework for cross-DSL code generation. Starting from 13,045 DSL-related papers, LLM-assisted open card sorting identifies 18 representative application domains. For each domain, we select an evaluable DSL and provide a syntax checker and task-level semantic oracle. Evaluating 7 LLMs on 18 DSLs, we observe mean syntax and semantic pass rates of 31.37% and 11.64%, respectively, revealing a substantial gap between syntax acceptance and semantic correctness. A variance decomposition attributes 76.99% of semantic pass-rate variation to differences among DSLs and 7.86% to differences among models. Among four DSL-level factors, Knowledge Demand has the strongest negative association with semantic correctness (r = -0.508). We evaluate prompt engineering across all 18 DSLs and study five improvement methods on five targeted DSLs. Prompt engineering yields uneven gains; RAG-Refinement, the strongest targeted method, improves semantic correctness by 16.29 percentage points on average over matched baselines. We further derive capability profiles and DSL-Profiler summaries for interpreting data-scarce DSLs. These results highlight toolchain-specific semantic bottlenecks and the need to evaluate DSL code generation beyond syntax acceptance.

## Full Text


<!-- PDF content starts -->

DSL-Bench: Evaluating and Explaining LLM Code
Generation Across Domain-Specific Languages
Zihan Zhou, Mingxi Ye, Yan Gao, Guipeng Wei, and Zibin Zheng
Sun Yat-sen University
Abstract—Large language models (LLMs) are increasingly
used in software-engineering workflows, making reliable evalua-
tion of LLM code generation important. Existing benchmarks
largely focus on general-purpose languages, yet software sys-
tems in a wide range of real-world application domains rely
on domain-specific languages (DSLs) with their own syntax,
semantics, and validation requirements. Benchmarks designed
for individual DSLs provide useful domain-specific evidence, but
their designs and metrics are difficult to compare across DSLs.
We present DSL-Bench, an executable benchmark and analysis
framework for cross-DSL code generation. Starting from 13,045
DSL-related papers, LLM-assisted open card sorting yields 18
representative application domains. For each domain, we select
an evaluable DSL and instantiate a corresponding syntax checker
and task-level semantic oracle. Evaluating 7 LLMs on 18 DSLs,
we observe a mean syntax pass rate of 31.37% and a mean
semantic pass rate of only 11.64%, demonstrating a large gap
between syntax acceptance and semantic correctness. A variance
decomposition shows that differences among DSLs account for
76.99% of the variation in semantic pass rate, whereas differences
among models account for 7.86%. To characterize this variation,
we analyze four DSL-level factors. Knowledge Demand shows the
largest negative association with semantic correctness (r=-0.508).
To assess intervention strategies, we test prompt engineering
across all 18 DSLs and additionally evaluate five methods on
five targeted DSLs. Prompt engineering yields uneven gains. In
the five targeted DSLs, the strongest targeted method, RAG-
Refinement, improves semantic correctness by 16.29 percentage
points on average. We further derive diagnostic capability profiles
and DSL-Profiler summaries for interpreting data-scarce DSLs.
These results show that cross-DSL code generation is limited by
toolchain-specific semantic bottlenecks and should be evaluated
beyond syntax-level acceptance.
Index Terms—Domain-specific languages, code generation,
large language models, benchmark, semantic validation
I. INTRODUCTION
Existing large language model (LLM) code generation
benchmarks mainly evaluate general-purpose programming
languages (GPLs). As LLMs are increasingly used to gen-
erate, repair, and assist with code, these benchmarks have
become important for comparing models, exposing failure
modes, and guiding coding agents. Benchmarks such as Hu-
manEval, MBPP, APPS, MultiPL-E, DS-1000, and repository-
level repair suites have shaped how the community measures
LLM coding capability [1]–[8]. Subsequent work has extended
this evaluation landscape to broader domains, richer context,
different code granularities, stronger tests, and contamination
control [9]–[11].
Evaluating LLM performance on domain-specific language
(DSL) code generation is also important, since a wide rangeof real-world software-engineering domains rely on DSLs.
A DSL is a programming language tailored to a particular
application domain, trading generality for domain-specific
expressiveness [12]. DSLs appear in hardware design, formal
verification, infrastructure configuration, database and graph
querying, workflow modeling, behavior modeling, and sci-
entific computing. Examples include Verilog for hardware
design [13], Lean for theorem proving [14], and Cypher for
graph queries [15]. A DSL is not simply a low-resource
GPL or a GPL with a domain library. It encodes domain
abstractions, toolchain constraints, and semantic rules directly
into the target notation.
Evaluating DSL code generation is difficult because DSL
correctness is tied to domain-specific semantics and toolchains.
For GPL benchmarks, generated programs are often evaluated
through execution tests in relatively common programming
environments, but test strength still affects the reliability of
the conclusions [16], [17]. For DSLs, the evaluator often
needs a domain-specific checker or oracle. Verilog generation
depends on hardware simulation, Lean and Dafny depend on
proof checking, infrastructure-as-code depends on resource
and configuration constraints, and Cypher depends on graph-
query results [13]–[15], [18], [19].
Although a few benchmarks have been proposed for specific
DSLs, cross-DSL LLM behavior remains unclear. Prior work
has studied hardware description [13], theorem proving [14],
[18], infrastructure configuration [19], smart contracts [20],
graph query [15], and other DSL settings. These benchmarks
are designed around their own tasks, oracles, metrics, and
model sets. As a result, they cannot easily show how LLM per-
formance varies across DSLs, which DSL properties explain
this variation, or which improvement strategies work under
different DSL bottlenecks.
Therefore, evaluation results are needed from multiple DSLs
under the same set of LLMs and testing conditions. Improve-
ment methods also need to be tested across DSLs, so that their
cross-DSL effects can be analyzed.
To meet this need, we presentDSL-Bench, an executable
benchmark and analysis framework for LLM code generation
across DSLs. DSL-Bench is constructed through a literature-
based domain selection process and executable benchmark
design. Starting from 13,045 DSL-related papers, we identify
representative application domains and select 18 evaluable
DSLs. For each DSL, we instantiate or adapt a syntax checker
and a task-level semantic oracle, then evaluate the same set
of LLMs under the same testing conditions. DSL-Bench also
arXiv:2611.05168v1  [cs.SE]  3 Nov 2026

analyzes DSL-level factors and tests improvement methods
across DSLs, so that baseline performance and intervention
effects can be studied together.
Figure 1 summarizes the overall DSL-Bench workflow and
main outputs.
The results show that syntax acceptance substantially over-
estimates task-level correctness. Across the 18 DSLs and seven
evaluated LLMs, the mean syntax pass rate is 31.37%, while
the mean semantic pass rate is only 11.64%. LLM performance
is also strongly shaped by the target DSL. Differences among
DSLs account for 76.99% of semantic pass-rate variation,
compared with 7.86% for differences among models. Among
the four DSL-level factors analyzed in this study, the Knowl-
edge Demand factor has the strongest negative association
with semantic correctness (r=−0.508). Improvement gains
are also bottleneck-dependent. Prompt engineering gives un-
even gains across DSLs, while RAG-Refinement is the most
effective targeted method in the five targeted DSLs. These
analyses further support diagnostic model capability profiles
and aDSL-Profilersummary for data-scarce DSL triage.
In summary, this paper makes the following contributions:
•We build DSL-Bench, an executable benchmark covering
18 representative DSLs, with syntax checkers and task-
level semantic oracles for evaluating the same set of
LLMs under the same testing conditions.
•We provide cross-DSL evidence on LLM code gener-
ation, showing a large gap between syntax acceptance
and task-level semantic correctness and showing that
performance varies more by DSL than by model.
•We analyze DSL-level factors and improvement methods,
showing that Knowledge Demand is strongly associated
with lower semantic correctness and that method gains
are shaped by DSL-specific knowledge and semantic
bottlenecks.
•We derive diagnostic model capability profiles and DSL-
Profiler summaries for triaging data-scarce DSLs.
The remainder of this paper is organized as follows. Sec-
tion II gives the background needed for DSL-Bench. Sec-
tion III describes the construction of DSL-Bench. Section IV
presents the baseline evaluation setup and results. Section V
analyzes the observed performance variation, and Section VI
evaluates improvement methods under DSL bottlenecks. Sec-
tion VII discusses threats to validity, Section VIII reviews
related work, and Section IX concludes.
II. BACKGROUND
A. DSLs and Toolchains
A DSL is designed around a particular application domain
and usually trades general-purpose expressiveness for concise
domain notation [12], [21]. DSLs appear as query languages,
configuration languages, modeling languages, workflow lan-
guages, proof languages, hardware-description notations, and
scientific-computing languages. Their surface syntax may look
familiar, but their meaning is defined by domain objects
and by the toolchain that interprets them. For example, agraph query is judged by query results, a proof script by
proof checking, and a hardware description by simulation or
synthesis behavior [13]–[15].
DSL ecosystems are often smaller and more fragmented
than GPL ecosystems. A DSL may have fewer public ex-
amples, multiple active versions, tool-specific extensions, or
incomplete validation infrastructure. The same surface nota-
tion may also behave differently across related languages or
versions. These properties make DSL generation different from
generating GPL code with a domain library. The generated
program must match the domain notation, the domain seman-
tics, and the constraints of the target toolchain.
B. DSL Evaluation
LLM-based code generation for low-resource and domain-
specific programming languages has become an active research
topic [22]. Existing DSL studies provide important evidence in
individual settings, including hardware design, theorem prov-
ing, program verification, infrastructure configuration, smart
contracts, and graph querying [13]–[15], [18]–[20]. These
studies show that DSL evaluation needs executable or struc-
tural oracles, since a generated output may parse successfully
while failing the domain task.
The current evidence is still fragmented across DSLs. Ex-
isting benchmarks usually define tasks, validators, metrics,
model sets, and execution environments for one language or
one adjacent notation. This makes it difficult to tell whether
an observed failure comes from the model, the target DSL,
the validator, or the evaluation setting. A broader evaluation
therefore needs results from multiple DSLs under the same
model set and testing conditions, while still preserving the
validator that each DSL requires.
III. BENCHMARKCONSTRUCTION
A cross-DSL benchmark first needs a clear scope and a
validation design that works across heterogeneous DSLs. This
leads toRQ1. What evaluation scope is needed to obtain
a broad and representative cross-DSL view of LLM code
generation?The following subsections describe how DSL-
Bench selects application domains, maps them to evaluable
DSLs, constructs benchmark tasks, and builds syntax checkers
and task-level semantic oracles.
A. Domain Selection
DSL-Bench starts from a literature-driven corpus assembled
from arXiv, IEEE Xplore, Web of Science, the ACM Digital
Library, and from 2022 to 2025 selected top-tier venues in
software engineering, security, and artificial intelligence. The
pipeline screens DSL-related papers, applies LLM-assisted
relevance filtering, deduplicates records, then uses open card
sorting [23]–[28], sentence-transformer clustering [29], and
manual audit. The screening process yields 79,844 candidates,
14,955 relevant records, 13,045 deduplicated papers, 4,796
labels, 15 pre-audit clusters, and 18 final domains cover-
ing 10,610 papers (81.3%). The 15-to-18 transition reflects
split/merge decisions and multiple benchmarkable domains

Fig. 1. DSL-Bench construction, evaluation, and analysis workflow.
from coarse clusters. Four graduate students in software engi-
neering participated in the manual review of the LLM-assisted
clustering, domain labels, and domain-to-DSL mapping de-
cisions. Exact queries, venues, prompts, sampled decisions,
and mapping rationale are documented in the supplement. The
procedure is intended as reproducible selection, with taxonomy
claims outside its scope.
B. DSL Selection
The literature screen identifies application domains, so we
perform a criteria-guided manual mapping from each domain
to a representative DSL. Each final domain-to-DSL mapping
is documented against the same five eligibility criteria: domain
centrality, executable benchmarkability, task diversity, repro-
ducibility with public resources, and non-redundancy.
This mapping yields 18 DSLs: Nextflow DSL2, SMILES,
jq, Vega-Lite, Cypher, Terraform HCL, LaTeX, GDScript,
Verilog, IEC 61131-3 Structured Text, SysML v2, Verilog-
A, Dafny, OpenQASM 3, Behavior Tree XML, UFL, Solidity,
and Lean. The set mixes mature DSLs with newer or under-
benchmarked ones and includes analytically useful pairs such
as Verilog/Verilog-A.
C. Dataset Construction
For each DSL, we adapt an existing benchmark when
available or construct tasks from public domain resources
otherwise. The resulting datasets fall into three categories. The
first category directly inherits existing benchmark data and
evaluation methods, covering cheminformatics, data engineer-
ing, data visualization, database/query, DevOps/infrastructure,
document processing, hardware design, program verification,
quantum computing, robotics, smart contracts, and theorem
proving [13], [15], [18]–[20], [30]–[37]. The second category
uses existing data but rebuilds validators for this study, cover-
ing industrial automation and model-driven engineering. The
third category constructs both data and validators from public
ecosystem resources, covering bioinformatics, game develop-
ment, power electronics, and scientific computing [38]–[43].
The benchmark records source identifiers, sampling/adaptation
status, validators, source task counts, and train/retrieval exclu-
sions.
Table I summarizes the benchmark coverage, source task
pools, syntax validators, semantic validation bases, and con-
struction categories. For evaluation, each DSL is convertedinto an executable task set with a documented syntax validator
and task-level semantic validation basis. All reported pass
rates are computed on the evaluated task sets used by the
corresponding validators. The supplement reports source-to-
evaluation filtering, sampling, and adaptation details, together
with per-DSL and per-model denominators. Aggregate results
are reported primarily as macro-averages over DSL-model
cells to prevent large source pools from dominating the cross-
DSL comparison, with evaluated-task-weighted aggregates re-
ported as sensitivity checks.
D. Validators
DSL-Bench uses a two-level validator for every DSL. Syn-
tax pass rate means acceptance by the relevant DSL-specific
parser, compiler, schema checker, proof assistant, simulator, or
toolchain. Semantic pass rate means satisfaction of a task-level
semantic validation basis appropriate to the target DSL, such
as execution, simulation, proof acceptance, canonical equiv-
alence, query-result equivalence, policy checking, or struc-
tural control/dataflow checking. DSL-Bench therefore unifies
the generation protocol and syntax/semantic metric interface
while preserving DSL-appropriate task oracles. Cross-DSL
comparison in DSL-Bench is diagnostic: it compares model
behavior under a shared generation protocol and a common
syntax/semantic metric interface, rather than assuming a single
universal correctness notion across all DSLs. Full commands,
tool versions, audit notes, and validation boundaries are re-
ported in the supplement.
To further check validator reliability, we manually audited
20 generated outputs for each DSL, stratified where possible
across syntax failures, syntax-pass/semantic-fail cases, and
semantic-pass cases. For DSLs whose data and validators
were newly constructed, we additionally reviewed all con-
structed benchmark items and validator rules during bench-
mark construction. For each audited output, we inspected
whether the validator decision matched the task intent and
documented semantic validation basis. The audit was used to
check whether validator outcomes were consistent with the
task specification rather than to override automated results. For
each audited case, reviewers inspected the prompt, generated
output, validator output or log where available, and expected
semantic criterion. Disagreements or unclear cases are treated
as validation-boundary notes rather than as manual overrides
of the automated outcomes.

TABLE I
DSL-BENCH COVERAGE,VALIDATORS,AND CONSTRUCTION SOURCES.
Domain DSL Source tasks Syntax Semantic basis Construction
Bioinformatics Nextflow DSL2 83 Nextflow stub run module/dataflow/control-structure checks New benchmark
Cheminformatics SMILES 33,008 RDKit parser canonical molecular equivalence Reused
Data Engineering jq 4,392 jq parser JSON output equivalence Reused
Data Visualization Vega-Lite 7,247 JSON + schema chart-intent structural checks Reused
Database/Query Cypher 10,882 Neo4j execution normalized query-result equivalence Reused
DevOps/Infra. Terraform HCL 458 Terraform validate/plan task-specific policy checks Reused
Document Processing LaTeX 440pdflatexdocument-structure intent checks Reused
Game Development GDScript 30 Godot preflight runtime task harness New benchmark
Hardware Design Verilog 156 Icarus Verilog testbench simulation Reused
Industrial Automation IEC ST 31 RuSTy-compatible check interface/property checks Validator rebuilt
Model-Driven Eng. SysML v2 90 SysML v2 validator dependency and semantic diagnostics Validator rebuilt
Power Electronics Verilog-A 36 OpenV AF ngspice behavioral tests New benchmark
Program Verification Dafny 166 Dafny resolve Dafny verification Reused
Quantum Computing OpenQASM 3 397 QASM parser/verifier circuit-level checks Reused
Robotics BT XML 588 BehaviorTree.CPP tree-structure/dataflow checks Reused
Scientific Computing UFL 25 Python/UFL preflight symbolic form-signature equivalence New benchmark
Smart Contract Solidity 1,125 solc/Foundry repository tests Reused
Theorem Proving Lean 1,159 Lean/lake proof checking withoutsorry/admitReused
Total 18 DSLs 60,313 N/A N/A N/A
The project records separate benchmark construction inputs,
generation outputs, validator outputs, and analysis scripts. For
each DSL, they record task identifiers, generation prompts,
model outputs, syntax outcomes, semantic outcomes, and
validator metadata. This organization allows aggregate results
to be traced back to per-task validator decisions and supports
checks of the macro-average, evaluated-task-weighted sensi-
tivity, factor analysis, and intervention results.
Answer to RQ 1.A broad cross-DSL view of LLM code generation
requires both an explicit DSL definition and executable validation; bench-
mark size by itself is insufficient. DSL-Bench instantiates this view with
literature-driven coverage of 18 representative DSLs paired with a two-level
syntax/semantic validator per DSL, so that parser acceptance and task-level
semantic correctness become separately observable and aggregable under
a common protocol.
IV. BASELINEEVALUATION
With DSL-Bench constructed, the next step is to measure
LLM performance across the selected DSLs. This leads to
RQ2. How does LLM performance vary across DSLs when
syntax acceptance and task-level semantic correctness are
measured under the same models and testing conditions?
This section reports the model setup, evaluation metrics, base-
line pass rates, DSL/model effects, model ranking variation,
and model capability profiles.
A. Model Setup
We evaluate seven LLMs that cover different model fam-
ilies, sizes, and deployment settings: Qwen3-1.7B, Qwen3-
8B, Qwen3-32B, Qwen2.5-Coder-7B, Ministral-3-8B-Instruct,
DeepSeek-v3.2, and GLM-4.7. This set includes a controlled
Qwen size progression, a code-specialized model, compact
open models, and larger general-purpose models. Local infer-
ence for the locally deployable models—Qwen3-1.7B, Qwen3-
8B, Qwen3-32B, Qwen2.5-Coder-7B, and Ministral-3-8B-
Instruct—was performed on an NVIDIA DGX Spark environ-
ment with 128 GB GPU memory. DeepSeek-v3.2 and GLM-
4.7 were evaluated through online API endpoints becausethey were not locally deployable in our experimental setup.
For all models, we used the same prompt templates, output
extraction rules, validation pipeline, and metric computation;
for remotely served models, we recorded the endpoint model
name, query date, and decoding parameters where available.
We focus on reproducible and deployable model configurations
rather than closed-source frontier upper bounds. Evaluating
frontier APIs across all 18 DSLs and intervention settings
would substantially increase cost and introduce versioning and
reproducibility issues, so systematic frontier-model evaluation
is outside the scope of this study. All baseline generations
use zero-shot prompting with temperature 0 and deterministic
decoding where supported.
B. Evaluation Metrics
For each DSLd, modelm, and taskt, the validator returns
two binary outcomes,SY N d,m,t andSEM d,m,t .SY N d,m,t
indicates whether the generated code passes syntax validation,
andSEM d,m,t indicates whether it passes semantic validation.
The syntax pass rate for a DSL-model cell is the percentage of
tasks accepted by the DSL parser, compiler, schema checker,
proof assistant, or equivalent toolchain. The semantic pass
rate is the percentage of tasks satisfying the DSL-specific
semantic oracle. We also report the syntax-semantic gap,
defined as syntax pass rate minus semantic pass rate. Unless
otherwise stated, aggregate results are macro-averaged over
DSL-model cells so that DSLs with larger evaluated task sets,
such as Solidity, Lean, Cypher, or SMILES, have the same
cell-level weight as smaller DSLs of practical interest, such
as UFL, Verilog-A, or GDScript. The supplementary material
reports the corresponding evaluated-task-weighted aggregate
as a sensitivity check.
C. Overall Results
Figure 2 shows the baseline semantic pass rates by DSL
and deployment configuration. At the model level, DeepSeek-
v3.2 and GLM-4.7 have the highest macro semantic pass rates,

Fig. 2. Baseline semantic pass rates across DSLs and evaluated models.
while the full model-level baseline table is provided in the
supplement. Across all 126 DSL-model cells, the mean syntax
pass rate is 31.37%, while the mean semantic pass rate is only
11.64%. The mean syntax-semantic gap is 19.74 percentage
points. This gap shows that a substantial share of generated
DSL programs are accepted by the relevant toolchain and
still fail the task-level oracle. The result supports the central
motivation for DSL-Bench. Syntax-only or compilation-only
evaluation can overestimate DSL code generation competence
by a factor of about three on average. Under evaluated-task-
weighted aggregation, reported in the supplementary material,
the corresponding baseline mean is 33.13% syntax and 8.10%
semantic, preserving the same qualitative syntax-semantic gap.
D. Per-DSL Results
Performance varies far more across DSLs than the overall
average suggests. The strongest DSLs are GDScript and Ver-
ilog, with mean semantic pass rates of 47.14% and 42.40%,
respectively. In contrast, Bioinformatics/Nextflow, Scientific
Computing/UFL, and Quantum Computing/OpenQASM 3
have zero mean semantic pass rate under the baseline setting,
while SysML v2 and Behavior Tree XML are near zero.
The DSL-level results reveal two patterns. First, a low
semantic score can arise from very different failure modes.
OpenQASM 3 and UFL fail almost entirely at the syn-
tax/toolchain layer, whereas Nextflow and Behavior Tree XML
often have nontrivial syntax pass rates alongside near-zero
semantic correctness. Second, surface familiarity gives limited
guidance about semantic success. DSLs such as Vega-Lite,
SMILES, Cypher, Nextflow, and Behavior Tree XML often
produce large syntax-semantic gaps, suggesting that the modelcan generate plausible surface forms without satisfying the
domain-level intent. Several DSLs have small evaluated task
sets, including UFL (25), GDScript (30), IEC 61131-3 ST
(31), and Verilog-A (36), matching their source task pools
in Table I. Their per-DSL rates should therefore be read as
high-uncertainty estimates; a single task changes the semantic
pass rate by 2.8 to 4.0 percentage points in these DSLs. We
keep them in the macro analysis because they cover under-
benchmarked settings of practical interest, and we separately
report the leave-small-DSL-out sensitivity analysis below.
E. DSL and Model Effects
To quantify whether performance variation is associated
more with the target DSL or with the model, we decompose
DSL-model cell performance using a balanced two-way addi-
tive ANOV A over the 18 DSLs and seven models. For each
DSL-model celly d,m, whereyis either syntax or semantic
pass rate, we fit
yd,m=µ+α d+βm+ϵd,m.
Hereα dis the DSL effect,β mis the model effect, and
the residual captures DSL-model interaction and unexplained
variation. We report eta-squared proportions for DSL, model,
and residual components. We estimate uncertainty by bootstrap
resampling DSL-model cells. Because DSL source task pools
range from 25 to 33,008 tasks and evaluated task counts also
remain imbalanced after sampling/adaptation, this ANOV A
provides a macro cell-level comparison over DSL-model con-
ditions. Evaluated task counts and evaluated-task-weighted
sensitivity checks are reported in the supplement. Table II
reports the resulting eta-squared proportions. For semantic

TABLE II
ADDITIVEDSL/MODEL MAIN-EFFECT DECOMPOSITION WITH BOOTSTRAP
95%CONFIDENCE INTERVALS.
Metric DSLη2Modelη2Resid.
Semantic 76.99 [71.46, 87.27] 7.86 [4.23, 23.35] 15.15
Syntax 64.53 [58.11, 79.37] 15.94 [9.45, 32.26] 19.53
correctness, the DSL main effect accounts for 76.99% of
macro-cell pass-rate variation, while the model main effect
accounts for 7.86%. For syntax pass rate, the DSL main effect
still dominates, accounting for 64.53% of variation, while the
model main effect accounts for 15.94%. Bootstrap confidence
intervals and evaluated-task-weighted sensitivity checks show
the same ordering. Under evaluated-task weighting, the DSL
main effect accounts for 72.49% of semantic variation, com-
pared with 11.80% for the model main effect. This additive
view isolates the relative magnitude of DSL and model main
effects, while ranking instability and Kendall’sWstill show
DSL-by-model interaction.
To check whether the conclusion is driven by small DSLs
or validation-scope differences, we rerun the same cell-level
decomposition under two restrictions. Removing DSLs with
fewer than 50 evaluated tasks leaves 14 DSLs and preserves
the same ordering: the DSL main effect accounts for 71.91%
of semantic variation and the model main effect accounts for
9.57%. Restricting to the 11 DSLs with fully executable task-
level semantic validation gives a similar result. The DSL main
effect accounts for 77.51% and the model main effect accounts
for 8.17%.
To check that the macro-cell ANOV A does not drive this
conclusion, we also fit a binomial mixed-effects sensitivity
model over DSL-model semantic pass counts. The estimated
DSL random-effect variance is 15.03 times larger than the
model random-effect variance, preserving the same qualitative
ordering. Full model details are reported in the supplement.
Model choice remains relevant. Larger or stronger models
generally perform better on average. The result shows that
model improvements are strongly conditioned by the target
DSL. Moving from one DSL to another changes semantic
correctness more than moving from one evaluated model to
another in this benchmark.
F . Model Rankings
If model capability transferred uniformly across DSLs, the
model ranking would be stable from one DSL to another.
Instead, rankings shift from DSL to DSL. Kendall’s coef-
ficient of concordance over per-DSL semantic rankings is
W= 0.426, indicating limited agreement across DSLs. The
strongest average models, DeepSeek-v3.2 and GLM-4.7, lead
on multiple DSLs, and smaller or code-specialized models
remain competitive in particular settings. This reinforces the
variance-decomposition result: model choice matters, and its
effect depends on the DSL.TABLE III
DSL-LEVEL FACTOR TRENDS AGAINST BASELINE SEMANTIC PASS RATE.
Factor Sloper pInterpretation
Knowledge Demand -2.31 -0.508 .030 Largest observed negative association.
Harness Complexity -0.83 -0.150 .560 Weak semantic association; stronger for
syntax.
Syntax Familiarity +0.22 +0.034 .889 Little guidance about semantic success.
Release Year -4.49 -0.351 .144 Newer DSLs trend lower, with high uncer-
tainty.
G. Capability Profiles
We use the DSL–model matrix to derive exploratory ca-
pability profiles. These profiles normalize model performance
within DSLs and summarize relative strengths across bottle-
neck categories. They are diagnostic summaries rather than
calibrated model rankings; small differences are interpreted
with bootstrap confidence intervals and tied-within-CI com-
parisons. The full formulas, axis definitions, scorecard figure,
and results are in the supplement.
Answer to RQ 2.Across 18 DSLs and 7 LLMs, the mean semantic pass
rate is 11.64% with a 19.74-point syntax/semantic gap, and the DSL main
effect explains 76.99% of macro-cell semantic variation versus 7.86% for
the model effect; Kendall’sW= 0.426further reveals substantial DSL-
by-model interaction. Exploratory bottleneck-axis profiles can supplement,
but not replace, per-DSL results.
V. VARIATIONANALYSIS
The baseline results show that the target DSL explains
much of the semantic performance variation. This leads to
RQ3. Which DSL-level properties and failure modes help
explain cross-DSL performance variation?This section an-
alyzes DSL-level factors, presents DSL-Profiler, and examines
qualitative failure cases.
A. DSL Factors
We compute four normalized DSL-level proxies from
benchmark/reference corpora and metadata. KD captures do-
main terminology, local API/symbol surface, and domain-
term concentration. SF captures surface similarity to common
programming, query, configuration, and markup notations.
HC captures executable scaffolding and environment burden.
Release Year approximates ecosystem recency and version
instability. The supplement gives the extraction formulas and
raw component values. The factor analysis relates these DSL-
level proxies to DSL-level mean syntax and semantic pass
rates. We report Pearson and Spearman correlations. Because
there are 18 DSLs, we treat these correlations as explanatory
evidence and avoid causal claims. We check stability with
bootstrap, permutation, and leave-one-DSL-out analyses.
Table III summarizes the fitted bivariate slopes and cor-
relations. Knowledge Demand has the largest observed neg-
ative association with semantic correctness among the four
corpus-derived factors. This association is directionally stable
under leave-one-DSL-out, partial-correlation, bootstrap, and
permutation checks. The bootstrap interval for Pearsonris
[−0.770,−0.006], and permutation testing givesp= 0.030,
but this uncorrectedp-value fails a conservative Bonferroni

correction. We therefore treat KD as the strongest explanatory
factor while avoiding definitive or causal claims. Harness
Complexity is more strongly associated with syntax/toolchain
validity than with semantic correctness. Syntax Familiarity is
weakly associated with semantic correctness, suggesting that
surface resemblance to common code gives limited guidance.
Release year is negatively correlated with performance, consis-
tent with newer DSLs having less public ecosystem exposure
and stronger version instability.
B. DSL-Profiler
Practical DSLs often lack public task sets, complete valida-
tors, or enough examples for a full DSL-Bench-style evalua-
tion. DSL-Profiler is intended as a diagnostic triage aid rather
than a calibrated pass-rate predictor. It maps a target DSL’s
standardized factor profile to nearby benchmark DSLs and
returns coarse risk ranges, a bottleneck hypothesis, a candidate
intervention, and nearest-neighbor distance as a trust indicator.
Leave-one-DSL-out validation gives a semantic MAE of 11.78
percentage points and a syntax MAE of 20.29 points, so
the output should be read as a triage hint that guides which
validators, retrieval sources, or repair loops to inspect first. The
formulas and nearest-neighbor details are in the supplement.
C. Knowledge Bottlenecks
High-KD DSLs tend to require domain concepts that are
difficult to infer from surface syntax. UFL requires finite-
element and variational-form knowledge; Verilog-A requires
analog circuit modeling and continuous-time physical seman-
tics; Vega-Lite requires visualization intent and field/channel
semantics; and Cypher requires graph-schema and query-
result semantics. These DSLs require more than familiarity
with token sequences. Correct programs must bind domain
entities, APIs, or mathematical objects in ways that satisfy an
executable or structural oracle.
The correlation between Knowledge Demand and semantic
correctness is therefore consistent with the main thesis: ob-
served DSL failures often reflect weak semantic grounding.
We interpret KD as an explanatory signal and make no
causal claim from it. It captures observable properties of
the benchmark/reference corpora, and it may interact with
task design, dataset size, public documentation, and model
pretraining exposure. We therefore use KD together with case
analyses and avoid treating it as a single universal difficulty
score.
D. Misleading Syntax
Syntax Familiarity alone is a weak predictor of semantic
correctness. This leads to a recurring phenomenon that we
callsyntactic deception: a DSL appears transferable because
its surface form resembles a familiar language, while seman-
tic correctness collapses because the DSL requires different
domain-specific execution semantics.
Verilog and Verilog-A illustrate this effect. Both languages
share Verilog-family lexical and module-level conventions,
such as module declarations, port lists, parameter declarations,andendmodule-delimited structure. Verilog-A additionally
introduces analog blocks, electrical disciplines, branch dec-
larations, and contribution statements whose correctness de-
pends on continuous-time physical semantics. In DSL-Bench,
Verilog has one of the highest baseline semantic pass rates,
while Verilog-A remains much harder. Surface resemblance
alone leaves the difference unexplained. It reflects a shift from
digital RTL behavior, where simulation testbenches can often
validate discrete logic, to analog behavioral modeling, where
physical relationships, disciplines, and simulator-compatible
contributions must be correct.
Syntactic deception matters for AI coding tools because a
plausible-looking DSL output can be actively misleading. The
code may contain familiar declarations and delimiters while
violating the domain semantics that the target toolchain or
simulator enforces. This helps explain why the syntax pass
rate often runs two to three times the semantic pass rate within
a DSL.
E. Structural Semantics
Knowledge Demand explains only part of the failure pattern.
Behavior Tree XML exposes a complementary structural-
semantics bottleneck. It has relatively low Knowledge De-
mand, low Harness Complexity, and high Syntax Familiarity,
yet its baseline semantic pass rate is only 0.73%. The reason
is that correctness depends on the composition of the tree,
beyond producing parseable XML.
Behavior-tree tasks require the model to choose and place
control-flow nodes, conditions, actions, decorators, blackboard
keys, and functional attributes so that the resulting tree
matches the intended robot behavior. An output can be valid
XML and accepted by BehaviorTree.CPP while still failing
because the selector/sequence topology is wrong, a functional
node is missing, or blackboard dataflow does not match the
reference behavior. This case shows that DSL failures can
arise even without dense domain terminology. Some arise from
structural semantics: the meaning of the program is encoded
in topology, ordering, dependencies, and role-specific node
composition.
This observation has direct implications for future DSL
agents. Retrieval alone can make outputs look more like ref-
erence examples, while structural DSLs likely need topology-
aware planning and validator-guided repair. For such DSLs, the
central challenge is assembling the right executable structure
alongside retrieving the right vocabulary.
F . Version Interference
Some DSLs also fail because model priors are pulled toward
older or adjacent language versions. SysML v2 is a clear
example. It is newer than SysML v1 and uses a different
textual modeling syntax; LLMs often generate UML-like or
SysML v1-style fragments. OpenQASM 3 similarly differs
from earlier QASM conventions, and generated programs may
mix old-version syntax with newer task requirements.
Legacy version interference is distinct from low-resource
scarcity. A model may have substantial exposure to an older

language version or adjacent ecosystem, making the wrong
prior stronger than the correct one. This explains why explicit
version constraints, retrieved local examples, or fine-tuning
can help in some cases, while generic prompting may have
limited ability to override parameterized prior knowledge.
We therefore treat legacy version interference as a qual-
itative failure mode rather than estimating it as a separate
quantitative factor. The supplemental material includes rep-
resentative evaluator-log examples for these failure modes,
including Verilog-A syntactic deception, Behavior Tree XML
structural mismatch, SysML v2 legacy-version interference,
and UFL local-API failures.
Answer to RQ 3.Among four DSL-level proxies, Knowledge Demand
shows the largest negative association with semantic correctness (Pearson
r=−0.508), and case analyses surface three recurring failure modes:
syntactic deception, structural-semantics bottlenecks, and legacy-version
interference. DSL-Profiler uses these exploratory profiles to provide bottle-
neck hypotheses and candidate-intervention triage hints for unseen DSLs.
VI. IMPROVEMENTEVALUATION
The preceding analyses show that DSL performance de-
pends on DSL-specific knowledge, toolchains, and semantic
bottlenecks. This leads toRQ4. How do common LLM
improvement methods behave under different DSL bot-
tlenecks?This section evaluates prompt engineering across
all 18 DSLs and targeted retrieval, refinement, and adaptation
methods on bottleneck-selected DSLs.
A. Improvement Methods
We evaluate three intervention spaces. First, benchmark-
wide prompt engineering (PE) is applied to all 18 DSLs
using DSL-specific instructions, output constraints, and
non-benchmark examples. Second, targeted non-finetuning
pipelines are compared on five bottleneck-selected DSLs.
RAG retrieves local DSL documentation or examples before
generation. Baseline-Refinement repairs baseline output using
evaluator feedback. Retrieval-Refinement adds retrieved con-
text to repair. RAG-Refinement repairs a retrieval-generated
output using both retrieved context and evaluator feedback.
Third, a separate Qwen-family LoRA adaptation study is
reported in the supplement.
The five deep-study DSLs are SysML v2, Verilog-A, UFL,
Behavior Tree XML, and Terraform HCL. They cover ver-
sion interference, high-KD/high-HC domain semantics, struc-
tural semantics, and configuration-policy bottlenecks. Because
these interventions require retrieval corpora, evaluator feed-
back traces, or training data, the five-DSL study provides
bottleneck-aware evidence without claiming benchmark-wide
method superiority. The supplemental bottleneck-profile table
gives the detailed selection rationale.
For intervention experiments, we compute gains relative
to the matching baseline for the same DSL and model. For
methodh, the semantic gain is
∆sem
h(d, m) =Sem h(d, m)−Sem baseline (d, m).
Syntax gains are computed analogously. This paired DSL-
model comparison controls for the wide variation in baselinedifficulty across DSLs and models. Where task-level paired
evaluator outputs are available, we additionally use paired
bootstrap confidence intervals and exact McNemar tests. The
supplementary material reports these paired tests for the
fully paired targeted intervention subset and separately reports
model-domain averages for the two extension DSLs.
B. Improvement Results
Figure 3 summarizes the five-DSL semantic gains. Across
all 126 DSL-model rows, benchmark-wide PE improves mean
syntax pass rate by 6.38 percentage points and semantic pass
rate by 2.42 percentage points over the matching baseline.
This confirms that careful prompting is a useful baseline
intervention, although the average hides strong heterogeneity.
Semantic gains are positive in 12 of 18 DSLs and negative in 6,
with 65 model-domain rows improving, 23 unchanged, and 38
regressing. We therefore treat PE as a benchmark-wide output-
discipline baseline whose benefit varies by DSL. The full
per-DSL PE gains are in the supplement; the negative cases
suggest that generic persona, internal reasoning instructions,
and few-shot demonstrations can conflict with strict output
formats, validator expectations, or strong model priors.
In the five targeted DSLs, RAG-Refinement improves se-
mantic correctness by 16.29 percentage points on average
and syntax by 34.34 percentage points. Because the deep-
study DSLs include low-baseline bottleneck settings, this gain
should be read within the selected setting. Direct RAG is
second in semantic gain (+8.02 points), followed by Retrieval-
Refinement (+7.37) and Baseline-Refinement (+3.83). The
supplementary material gives the full five-DSL method table.
For the three DSLs with task-level paired evaluator JSONs
(SysML v2, UFL, and Verilog-A), RAG-Refinement improves
semantic correctness over the matching baseline by 19.11
points with a paired-bootstrap 95% CI of [16.65, 21.57] and
exact McNemarp <10−50. For Behavior Tree XML and
Terraform HCL, the evaluator outputs are aggregated at model-
domain granularity, so we report model-domain averages
rather than task-level McNemar tests.
We report PE separately from retrieval/refinement methods
because PE is evaluated over all 18 DSLs, whereas retrieval
Fig. 3. Semantic gains of targeted interventions over matched baselines; B-
Ref and Ret-Ref denote Baseline-Refinement and Retrieval-Refinement.

and refinement require additional DSL-specific infrastructure
and are evaluated in the five deep-study DSLs. The Qwen-
family LoRA study is separate from both comparisons.
Within this five-DSL setting, RAG-Refinement remains the
top method after holding out any one intervention DSL,
although dominance outside this set remains untested. The
method combines retrieval for DSL vocabulary, APIs, versions,
and examples with evaluator-guided repair of task-specific
failures. DSL-Profiler converts these intervention results into
bottleneck-aware triage hints.
C. Retrieval Limits
Direct RAG improves syntax validity, while semantic gains
are smaller because retrieval injects knowledge without guar-
anteeing task alignment. In Behavior Tree XML, retrieved ex-
amples make outputs look like valid BehaviorTree.CPP XML
without ensuring correct control-flow topology or blackboard
dataflow; in Terraform HCL, snippets can improve resource
syntax while missing policy intent. RAG-Refinement performs
better because retrieval moves the model toward the right DSL
constructs and evaluator feedback then repairs task-specific
failures.
D. Gain Patterns
Feedback-only refinement is weakest on average because it
repairs local errors without supplying missing DSL concepts.
Model capacity also modulates gains. RAG-Refinement has
the strongest scale association (r= 0.761withlog10parame-
ter count). A separate Qwen-family LoRA adaptation study is
reported in the supplement.
Answer to RQ 4.Intervention gains depend on the DSL bottleneck.
Benchmark-wide prompt engineering improves the mean unevenly (se-
mantic+2.42pp, with 12/18 DSLs gaining and 6 regressing). In the
five targeted DSLs, RAG-Refinement improves semantic correctness by
16.29 percentage points on average. These targeted results do not establish
benchmark-wide method superiority.
VII. THREATS TOVALIDITY
Construct validity.Our broad DSL definition leaves
domain-to-DSL mappings contestable, and then=18factor
analysis, capability profiles, and DSL-Profiler are explanatory
and relative, without calibrated predictions.
Benchmark construction and contamination.LLM-
assisted screening and card sorting may miss domains, and
public benchmark-derived tasks may appear in pretraining; we
exclude target solutions from retrieval and adaptation and run
a local overlap screen, while residual contamination remains
possible.
Oracle heterogeneity.Validators differ across DSL ecosys-
tems, so semantic pass rates are documented oracle outcomes
within each DSL; identical difficulty scoring across DSLs is
outside the claim.
Model coverage and deployment.The evaluated models
do not include all closed-source frontier coding models. This
is a deliberate scope choice motivated by reproducibility,
deployment control, version stability, and the cost of evaluating
many DSL-task-intervention combinations. Local inferenceused an NVIDIA DGX Spark environment, while DeepSeek-
v3.2 and GLM-4.7 used online API endpoints, so the results
characterize the selected deployment configurations rather than
the absolute upper bound of current LLM coding capability.
Interventions and profiler.Retrieval and refinement are
evaluated on five targeted DSLs, while prompt engineering
covers all 18 DSLs. DSL-Profiler offers exploratory risk
ranges and triage hints without guaranteeing improvement;
sensitivity checks, paired tests, and profiler validation are in
the supplement.
VIII. RELATEDWORK
A. GPL Benchmarks
GPL code generation benchmarks vary task scale, language
coverage, project context, contamination control, and test-
oracle design [1]–[8], [10], [11], [16], [17], [44]. Domain-
oriented benchmarks further show that code generation abil-
ity is domain-conditioned [9], [45], [46]. These benchmarks
mainly evaluate GPL code or GPL code that uses domain
libraries. Their results show that LLM coding performance
depends on task form, context, tests, and execution environ-
ment.
B. Executable Oracles
Executable benchmark infrastructure studies how bench-
mark tasks should be paired with reliable checking mech-
anisms. Defects4J and BugsInPy show the importance of
reusable benchmark tooling [47], [48], while EvalPlus and
LiveCodeBench show that weak tests and contamination can
distort LLM-code conclusions [16], [17]. These works es-
tablish oracle quality, task reproducibility, and contamination
control as central concerns in code generation evaluation.
C. DSL Benchmarks
The DSL literature characterizes DSLs as notations aligned
with domain abstractions and tool-support assumptions [12],
[21]. This literature shows that DSLs differ from GPLs not
only in syntax, but also in domain concepts, toolchains,
and semantic interpretation. A recent survey by Joel et al.
reviews LLM-based code generation for low-resource and
domain-specific programming languages, covering datasets,
models, evaluation practices, and adaptation methods [22].
Prior LLM studies evaluate individual DSLs or adjacent
executable notations in hardware and verification, theorem
proving, infrastructure as code, smart contracts, quantum and
system modeling, constrained SQL, PLC Structured Text, and
related settings [13], [14], [18]–[20], [30], [31], [43], [49]–
[57]. Related resources cover query, visualization, document,
robotics, quantum, molecule, workflow, scientific-computing,
game-scripting, analog-modeling, and JSON-transformation
settings [15], [32]–[42]. These studies establish the value of
specialized evaluation, usually with tasks, validators, metrics,
and model sets designed around one DSL or one adjacent
notation.
However, this specialization also limits the questions that
can be answered by single-DSL studies alone. When each

benchmark uses its own task distribution, validator design,
model set, and reporting convention, it becomes difficult
to determine whether an observed result is driven by the
model, the target DSL, the toolchain, or the oracle. DSL-
Bench complements these studies by preserving DSL-specific
validators while normalizing the generation protocol and the
syntax/semantic metric interface across a broader set of target
notations. Its purpose is therefore not to replace domain-
specific benchmarks, but to make their results comparable
along common axes of variation.
D. Improvement Methods
Prompting, constrained decoding, retrieval, validator-
feedback refinement, self-debugging, and LoRA-style adap-
tation target different failure modes [58]–[64]. Low-resource-
language work likewise suggests that fine-tuning, in-context
learning, and translation-style objectives operate in different
regimes [65]. These methods are often evaluated within one
language, model family, or benchmark setting, so reported
gains depend on the target environment and available feedback
or training data.
This dependence on the target setting is especially important
for DSLs. A method that improves surface form may not
improve task semantics if the main bottleneck is missing
domain knowledge, version mismatch, or toolchain setup.
Conversely, retrieval or feedback may help only when the
retrieved material and validator signal are aligned with the
DSL version, execution environment, and expected semantic
behavior. For this reason, DSL-Bench treats improvement
methods as bottleneck-sensitive interventions rather than as
benchmark-independent improvements.
IX. DISCUSSION ANDCONCLUSION
DSL-Bench suggests a practical design lesson for DSL-
aware coding agents: surface validity leaves important errors
undetected. The gap between syntax acceptance and semantic
correctness shows that parser acceptance must be paired with
task oracles such as proofs, simulations, query answers, policy
checks, or structural validators. The variance decomposition
also indicates that DSL bottlenecks are a useful organizing
principle for support: high-KD DSLs need documentation
grounding, syntactically deceptive DSLs need version- and
semantics-aware constraints, and structural DSLs need topol-
ogy and dataflow checks.
This observation is also relevant to the current shift from
general coding assistants toward domain-specific agents. For
such agents, improving the base model alone is unlikely to
be sufficient. DSL-specific support needs to cover at least
three sources of failure exposed by DSL-Bench. The first is
syntactic and version-specific knowledge, including grammars,
target-version constructs, output templates, and invalid surface
patterns that can be constrained before or during generation.
The second is domain knowledge, including concepts, APIs,
invariants, design idioms, and task semantics that cannot be
recovered from syntax alone. The third is the runtime andtoolchain environment, including parsers, compilers, simula-
tors, proof assistants, policy checkers, and diagnostic logs that
make validator-grounded repair possible.
These distinctions give a concrete basis for designing
domain-specific agents. Rather than treating all DSL tasks
as ordinary code generation, an agent can select retrieval
sources, prompting strategies, repair actions, and validation
tools according to the bottleneck profile of the target DSL.
Knowledge-heavy DSLs call for documentation- and example-
grounded generation; structurally constrained DSLs require
explicit checks over topology, dependency, and dataflow; and
toolchain-sensitive DSLs require environment-aware execution
and version control. DSL-Bench supports this line of work
by reporting syntax and semantic outcomes separately and by
exposing where surface validity fails to imply task correctness.
The improvement results reinforce this view. Prompting
improves output discipline, retrieval supplies missing concepts,
feedback repairs executable failures, and their combination
is most useful when the model can operationalize retrieved
context. Future systems should combine retrieval, executable
validation, and iterative repair while reporting syntax and
semantic metrics separately.
In conclusion, DSL-Bench provides a literature-driven ex-
ecutable benchmark across 18 DSLs and shows that DSL
code generation is largely a semantic-grounding problem in
our benchmark. Progress will require DSL-specific corpora,
validators, and agent workflows that reason over domain
semantics in addition to syntax.
REFERENCES
[1] M. Chen et al., “Evaluating large language models trained on code,”
arXiv:2107.03374, 2021, doi: 10.48550/arXiv.2107.03374.
[2] J. Austin et al., “Program synthesis with large language models,”
arXiv:2108.07732, 2021, doi: 10.48550/arXiv.2108.07732.
[3] D. Hendrycks et al., “Measuring coding challenge competence
with APPS,” inProc. NeurIPS Datasets and Benchmarks, 2021,
arXiv:2105.09938.
[4] Y . Li et al., “Competition-level code generation with AlphaCode,”
Science, vol. 378, no. 6624, pp. 1092–1097, 2022, doi: 10.1126/sci-
ence.abq1158.
[5] S. Lu et al., “CodeXGLUE: A machine learning benchmark dataset for
code understanding and generation,” arXiv:2102.04664, 2021.
[6] F. Cassano et al., “MultiPL-E: A scalable and polyglot approach to
benchmarking neural code generation,”IEEE Trans. Software Eng., vol.
49, no. 7, pp. 3675–3691, 2023, doi: 10.1109/TSE.2023.3267446.
[7] Y . Lai et al., “DS-1000: A natural and reliable benchmark for data
science code generation,” inProc. ICML, PMLR, vol. 202, pp. 18319–
18345, 2023.
[8] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and
K. Narasimhan, “SWE-bench: Can language models resolve real-world
GitHub issues?” inProc. ICLR, 2024, arXiv:2310.06770.
[9] Q. Zhu, J. Cao, Y . Lu, H. Lin, X. Han, L. Sun, and S.-C. Cheung,
“DOMAINEV AL: An auto-constructed benchmark for multi-domain
code generation,” inProc. AAAI, vol. 39, no. 24, pp. 26148–26156,
2025, doi: 10.1609/aaai.v39i24.34811.
[10] H. Yu et al., “CoderEval: A benchmark of pragmatic code generation
with generative pre-trained models,” inProc. ICSE, pp. 1–12, 2024, doi:
10.1145/3597503.3623316.
[11] X. Du et al., “Evaluating large language models in class-level code gen-
eration,” inProc. ICSE, pp. 1–13, 2024, doi: 10.1145/3597503.3639219.
[12] M. Mernik, J. Heering, and A. M. Sloane, “When and how to develop
domain-specific languages,”ACM Comput. Surv., vol. 37, no. 4, pp.
316–344, 2005, doi: 10.1145/1118890.1118892.

[13] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Invited paper:
VerilogEval: Evaluating large language models for Verilog code
generation,” inProc. ICCAD, pp. 1–8, 2023, doi: 10.1109/IC-
CAD57390.2023.10323812.
[14] K. Yang et al., “LeanDojo: Theorem proving with retrieval-augmented
language models,” inProc. NeurIPS Datasets and Benchmarks, 2023,
arXiv:2306.15626.
[15] Y . Feng, S. Papicchio, and S. Rahman, “CypherBench: Towards precise
retrieval over full-scale modern knowledge graphs in the LLM era,” in
Proc. ACL, pp. 8934–8958, 2025, doi: 10.18653/v1/2025.acl-long.438.
[16] J. Liu, C. S. Xia, Y . Wang, and L. Zhang, “Is your code generated by
ChatGPT really correct? Rigorous evaluation of large language models
for code generation,” inProc. NeurIPS, 2023, arXiv:2305.01210.
[17] N. Jain et al., “LiveCodeBench: Holistic and contamination free evalu-
ation of large language models for code,” inProc. ICLR, 2025.
[18] M. R. H. Misu, C. V . Lopes, I. Ma, and J. Noble, “Towards AI-assisted
synthesis of verified Dafny methods,”Proc. ACM Softw. Eng., vol. 1,
no. FSE, pp. 812–835, 2024, doi: 10.1145/3643763.
[19] P. T. J. Kon et al., “IaC-Eval: A code generation benchmark for
cloud infrastructure-as-code programs,” inProc. NeurIPS Datasets and
Benchmarks, 2024, doi: 10.52202/079017-4273.
[20] Z. Peng, X. Yin, R. Qian, P. Lin, Y . Liu, H. Zhang, C. Ying, and Y .
Luo, “SolEval: Benchmarking large language models for repository-level
Solidity smart contract generation,” inProc. EMNLP, pp. 4388–4411,
2025, doi: 10.18653/v1/2025.emnlp-main.218.
[21] T. Kosar, S. Bohra, and M. Mernik, “Domain-specific languages: A
systematic mapping study,”Information and Software Technology, vol.
71, pp. 77–91, 2016, doi: 10.1016/j.infsof.2015.11.001.
[22] S. Joel, J. J. W. Wu, and F. H. Fard, “A survey on LLM-based code gen-
eration for low-resource and domain-specific programming languages,”
ACM Trans. Software Eng. Methodol., 2025, doi: 10.1145/3770084.
[23] A. Begel and T. Zimmermann, “Analyze this! 145 questions for data
scientists in software engineering,” inProc. ICSE, pp. 12–23, 2014,
doi: 10.1145/2568225.2568233.
[24] N. Nurmuliani, D. Zowghi, and S. P. Williams, “Using card sort-
ing technique to classify requirements change,” inProc. IEEE
Int. Requirements Engineering Conf., pp. 240–248, 2004, doi:
10.1109/ICRE.2004.1335681.
[25] W. Hudson, “Card sorting,” inThe Encyclopedia of Human-
Computer Interaction, 2nd ed., Interaction Design Foundation, 2014.
[Online]. Available: https://www.interaction-design.org/literature/book/
the-encyclopedia-of-human-computer-interaction-2nd-ed/card-sorting.
[26] A. R. Barrett and J. S. Edwards, “Knowledge elicitation and knowledge
representation in a large domain with multiple experts,”Expert Systems
with Applications, vol. 8, no. 1, pp. 169–176, 1995, doi: 10.1016/0957-
4174(94)E0007-H.
[27] N. A. M. Maiden and G. Rugg, “ACRE: Selecting methods for require-
ments acquisition,”Software Engineering Journal, vol. 11, no. 3, pp.
183–192, 1996, doi: 10.1049/sej.1996.0024.
[28] G. Rugg and P. McGeorge, “The sorting techniques: A tutorial paper on
card sorts, picture sorts and item sorts,”Expert Systems, vol. 14, no. 2,
pp. 80–93, 1997, doi: 10.1111/1468-0394.00045.
[29] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using Siamese BERT-networks,” inProc. EMNLP-IJCNLP, pp. 3982–
3992, 2019, doi: 10.18653/v1/D19-1410.
[30] K. Zheng, J. M. Han, and S. Polu, “MiniF2F: A cross-system benchmark
for formal Olympiad-level mathematics,” inProc. ICLR, 2022, doi:
10.48550/arXiv.2109.00110.
[31] G. Tsoukalas, J. Lee, J. Jennings, J. Xin, M. Ding, M. Jennings, A.
Thakur, and S. Chaudhuri, “PutnamBench: Evaluating neural theorem-
provers on the Putnam Mathematical Competition,” inProc. NeurIPS
Datasets and Benchmarks, 2024, doi: 10.52202/079017-0368.
[32] Y . Luo, J. Tang, and G. Li, “nvBench: A large-scale synthesized
dataset for cross-domain natural language to visualization task,”
arXiv:2112.12926, 2021, doi: 10.48550/arXiv.2112.12926.
[33] S. Kale and V . Nadadur, “TeXpert: A multi-level benchmark for evaluat-
ing LaTeX code generation by LLMs,” inProc. Fifth Workshop on Schol-
arly Document Processing, pp. 7–16, 2025, doi: 10.18653/v1/2025.sdp-
1.2.
[34] R. A. Izzo, G. Bardaro, and M. Matteucci, “BTGenBot: Behavior
tree generation for robotic tasks with lightweight LLMs,” inProc.
IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS), 2024, doi:
10.1109/IROS58592.2024.10802304.[35] R. Yang, Z. Wang, Y . Gu, Y . Liang, and T. Li, “QCircuitBench:
A large-scale dataset for benchmarking quantum algorithm
design,” inAdvances in Neural Information Processing Systems
38 (NeurIPS 2025), Datasets and Benchmarks Track, 2025. [Online].
Available: https://proceedings.neurips.cc/paper files/paper/2025/
hash/3e343c7fa87656d7e88e9a83cb2a5d10-Abstract-Datasets and
Benchmarks Track.html.
[36] C. Edwards, C. Zhai, and H. Ji, “Text2Mol: Cross-modal molecule
retrieval with natural language queries,” inProc. EMNLP, pp. 595–607,
2021, doi: 10.18653/v1/2021.emnlp-main.47.
[37] G. Verbruggen, C. Parnin, V . Le, and S. Gulwani, “jqBench: A bench-
mark for reading and editing JSON from natural language and/or
examples,” inProc. ICLR, poster, 2026. [Online]. Available: https:
//openreview.net/forum?id=VStKtgGXUc.
[38] P. Di Tommaso et al., “Nextflow enables reproducible computational
workflows,”Nature Biotechnology, vol. 35, pp. 316–319, 2017, doi:
10.1038/nbt.3820.
[39] P. A. Ewels et al., “The nf-core framework for community-curated
bioinformatics pipelines,”Nature Biotechnology, vol. 38, pp. 276–278,
2020, doi: 10.1038/s41587-020-0439-x.
[40] GDQuest, “Learn GDScript From Zero,” GitHub repository, 2023.
[Online]. Available: https://github.com/GDQuest/learn-gdscript.
[41] M. S. Alnæs, A. Logg, K. B. Ølgaard, M. E. Rognes, and G. N. Wells,
“Unified Form Language: A domain-specific language for weak formu-
lations of partial differential equations,”ACM Trans. Math. Software,
vol. 40, no. 2, Art. 9, 2014, doi: 10.1145/2566630.
[42] OpenV AF Community, “OpenV AF: Next generation Verilog-A com-
piler,” 2026. [Online]. Available: https://openvaf.github.io/.
[43] Z. Liu, R. Zeng, D. Wang, G. Peng, X. Liu, Q. Liu, P. Liu, W.
Wang, and J. Wang, “Agents4PLC: Automating closed-loop PLC code
generation and verification in industrial control systems using LLM-
based agents,”IEEE Trans. Software Eng., pp. 1–16, 2026, doi:
10.1109/TSE.2026.3667895.
[44] T. Y . Zhuo et al., “BigCodeBench: Benchmarking code generation with
diverse function calls and complex instructions,” inProc. ICLR, 2025.
[45] D. Zheng, Y . Wang, E. Shi, X. Liu, Y . Ma, H. Zhang, and Z. Zheng, “Top
general performance = top domain performance? DomainCodeBench:
A multi-domain code generation benchmark,” inProc. ICSE, 2026,
arXiv:2412.18573.
[46] X. Gu, M. Chen, Y . Lin, Y . Hu, H. Zhang, C. Wan, Z. Wei, Y . Xu, and
J. Wang, “On the effectiveness of large language models in domain-
specific code generation,”ACM Trans. Software Eng. Methodol., vol.
34, no. 3, Art. 78, pp. 78:1–78:22, 2025, doi: 10.1145/3697012.
[47] R. Just, D. Jalali, and M. D. Ernst, “Defects4J: A database of existing
faults to enable controlled testing studies for Java programs,” inProc.
ISSTA, pp. 437–440, 2014, doi: 10.1145/2610384.2628055.
[48] R. Widyasari et al., “BugsInPy: A database of existing bugs in Python
programs to enable controlled testing and debugging studies,” inProc.
ESEC/FSE, pp. 1556–1560, 2020, doi: 10.1145/3368089.3417943.
[49] D. Delgado, L. Burgue ˜no, and R. Claris ´o, “A framework for assessing
the capabilities of code generation of constraint domain-specific lan-
guages with large language models,”J. Syst. Softw., Art. 112871, 2026,
doi: 10.1016/j.jss.2026.112871.
[50] P. Wu, N. Guo, J. Lv, et al., “RTLRepoCoder: Repository-level RTL
code completion through the combination of fine-tuning and retrieval
augmentation,” arXiv:2504.08862, 2025.
[51] P. Jana, S. Davidson, B. Bhasker, A. Kan, A. Deoras, and L. Cal-
lot, “TerraFormer: Automated infrastructure-as-code with LLMs fine-
tuned via policy-guided verifier feedback,” inProc. IEEE/ACM Int.
Conf. Software Engineering: Software Engineering in Practice, 2026,
arXiv:2601.08734.
[52] Z. Ye, J. Chen, Z. Shao, L. Bao, X. Yang, and Z. Liu, “SolContractEval:
A benchmark for evaluating contract-level Solidity code generation,”
arXiv:2509.23824, 2025.
[53] X. Guo, M. Wang, and J. Zhao, “QuanBench: Benchmarking quantum
code generation with large language models,” arXiv:2510.16779, 2025.
[54] D. Jin, Z. Jin, L. Li, Z. Fang, J. Li, and X. Chen, “A system
model generation benchmark from natural language requirements,”
arXiv:2508.03215, 2025.
[55] T. Scholak, N. Schucher, and D. Bahdanau, “PICARD: Parsing
incrementally for constrained auto-regressive decoding from lan-
guage models,” inProc. EMNLP, pp. 9895–9901, 2021, doi:
10.18653/v1/2021.emnlp-main.779.

[56] M. Fakih, R. Dharmaji, Y . Moghaddas, G. Q. Araya, O. Ogundare, and
M. A. Al Faruque, “LLM4PLC: Harnessing large language models for
verifiable programming of PLCs in industrial control systems,” inProc.
IEEE/ACM Int. Conf. Software Engineering: Software Engineering in
Practice, pp. 192–203, 2024, doi: 10.1145/3639477.3639743.
[57] A. Haag, B. Fuchs, A. Kacan, and O. Lohse, “Training LLMs
for generating IEC 61131-3 Structured Text with online feedback,”
arXiv:2410.22159, 2024.
[58] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inProc. NeurIPS, 2020.
[59] B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou, and W. Chen,
“CodeT: Code generation with generated tests,” inProc. ICLR, 2023.
[60] X. Chen, M. Lin, N. Sch ¨arli, and D. Zhou, “Teaching large language
models to self-debug,” inProc. ICLR, 2024, arXiv:2304.05128.
[61] A. Madaan et al., “Self-Refine: Iterative refinement with self-feedback,”
inProc. NeurIPS, 2023.
[62] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S.
Yao, “Reflexion: Language agents with verbal reinforcement learning,”
inProc. NeurIPS, 2023.
[63] S. Yao et al., “ReAct: Synergizing reasoning and acting in language
models,” inProc. ICLR, 2023, arXiv:2210.03629.
[64] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
and W. Chen, “LoRA: Low-rank adaptation of large language models,”
inProc. ICLR, 2022, arXiv:2106.09685.
[65] A. Giagnorio, A. Martin-Lopez, and G. Bavota, “Enhancing code gener-
ation for low-resource languages: No silver bullet,” arXiv:2501.19085,
2025.