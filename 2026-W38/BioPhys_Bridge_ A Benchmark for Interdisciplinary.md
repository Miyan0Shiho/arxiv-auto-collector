# BioPhys-Bridge: A Benchmark for Interdisciplinary Scientific Reasoning in Physics-Grounded Biological Research

**Authors**: Qingyang Xu

**Published**: 2026-09-15 12:53:16

**PDF URL**: [https://arxiv.org/pdf/2609.19180v1](https://arxiv.org/pdf/2609.19180v1)

## Abstract
Language models face unique challenges in analyzing interdisciplinary scientific research literature. In biophysics research, faithful answers require grounding observed data in source evidence, interpreting it through a quantitative physics model, and linking it to a biological mechanism. To address this challenge, we introduce BioPhys-Bridge, a novel benchmark dataset for evidence-grounded scientific reasoning over biophysical literature. Each case contains evidence blocks, stable evidence IDs, quantitative values, units, equations, assumptions, mechanisms, and next decisions as grounding targets for question answering (QA) and retrieval-augmented generation (RAG). The initial release contains 500 cases, 1,517 agent-facing tasks, and covers six biological domains and nine physical model families, including three sparse families reserved for future expansion. We enforce strict quality gates for all cases in schema, evidence-integrity, quantitative-grounding, source-license, duplicate, unit-normalization, with domain expert review and annotation for 81 cases. Preliminary evaluations show that DeepSeek-V4-Flash obtain the highest evidence-ID $F_1$ score (0.360), followed by Qwen3.7-Max (0.316) and GPT-4o-mini (0.294). BioPhys-Bridge is an interdisciplinary benchmark for evaluating attribution, faithfulness, hallucination reduction, and biological experiment design with complex, multi-step scientific reasoning. Future works will increase the size and complexity of the dataset and perform comprehensive evaluations. Code and data are available in the GitHub repository and on Hugging Face.

## Full Text


<!-- PDF content starts -->

BioPhys-Bridge: A Benchmark for Interdisciplinary Scientific Reasoning in
Physics-Grounded Biological Research
Qingyang Xu
Independent Researcher
Shanghai, China
qyxu1994@gmail.com
Abstract
Language models face unique challenges in
analyzing interdisciplinary scientific research
literature. In biophysics research, faithful
answers require grounding observed data in
source evidence, interpreting it through a quan-
titative physics model, and linking it to a bi-
ological mechanism. To address this chal-
lenge, we introduce BioPhys-Bridge, a novel
benchmark dataset for evidence-grounded sci-
entific reasoning over biophysical literature.
Each case contains evidence blocks, stable ev-
idence IDs, quantitative values, units, equa-
tions, assumptions, mechanisms, and next deci-
sions as grounding targets for question answer-
ing (QA) and retrieval-augmented generation
(RAG). The initial release contains 500 cases,
1,517 agent-facing tasks, and covers six bio-
logical domains and nine physical model fam-
ilies, including three sparse families reserved
for future expansion. We enforce strict quality
gates for all cases in schema, evidence-integrity,
quantitative-grounding, source-license, dupli-
cate, unit-normalization, with domain expert re-
view and annotation for 81 cases. Preliminary
evaluations show that DeepSeek-V4-Flash ob-
tain the highest evidence-ID F1score (0.360),
followed by Qwen3.7-Max (0.316) and GPT-
4o-mini (0.294). BioPhys-Bridge is an interdis-
ciplinary benchmark for evaluating attribution,
faithfulness, hallucination reduction, and bio-
logical experiment design with complex, multi-
step scientific reasoning. Future works will
increase the size and complexity of the dataset
and perform comprehensive evaluations. Code
and data are available in the GitHub repository
and on Hugging Face.
1 Introduction
Language models (LM) are increasingly used
to analyze scientific research papers in retrieval-
augmented generation (RAG) systems and other
agentic AI frameworks. However, retrieving a rele-
vant paragraph does not guarantee the faithfulnessof the final answer: the AI agent can cite loosely
related evidence, omit key assumptions, mix units,
or produce a plausible scientific interpretation that
the cited paper does not support. Scientific ground-
ing requires finding faithful connections among
source evidence, quantitative measurements, model
assumptions, and the leveraging domain expertise
to derive scientific conclusions and hypotheses.
Biophysics is an epitome of this challenge due to
its interdisciplinary nature. The LM needs to both
perform semantic text matching and draw quantita-
tive predictions from a biophysical model, in order
to connect a quantitative prediction (e.g., binding
affinity, reaction rate, phase-separation threshold)
to the model assumptions under which it is mean-
ingful, explain the underlying biological mecha-
nism, and suggest the next experiment or computa-
tion. Existing scientific QA and RAG benchmarks
often focus on gauging whether an answer is sup-
ported by textual evidence. In contrast, BioPhys-
Bridge gauges whether LM can perform evidence-
grounded reasoning across modalities (equations
and semantics) and scientific disciplines.
We propose a novel framework for grounding
LM which requires LM to identify source evidence,
use quantitative values and units consistently, re-
spect physics model assumptions, and produce an
actionable response supported by evidence IDs. Ev-
idence blocks are the atomic citation units, while
equations, assumptions, quantitative evidence, and
mechanism annotations provide intermediate struc-
ture for auditing whether a model’s answer is faith-
ful to the research question and biological reason-
ing rather than superficial semantic relations.
The initial release of BioPhys-Bridge con-
tains 500 cases extracted from open-access peer-
reviewed research papers in biophysics. This re-
lease contains 1,517 distinct tasks for biological
research. Every case passed rigorous schema val-
idation, evidence-integrity checks, quantitative-
grounding checks, source-license coverage checks,
arXiv:2609.19180v1  [cs.AI]  15 Sep 2026

unit normalization, duplicate checks, and seman-
tic content filters. To validate the correctness and
scientific value of the curated samples, we asked
domain experts to review and annotate 81 cases,
including the full held-out test set (with 50 cases)
and additional golden examples.
This works makes five important contributions
for grounded scientific LM. First, we introduce a
novel benchmark dataset for literature-grounded
scientific reasoning over biophysical papers. Sec-
ond, we define a structured case schema that treats
evidence blocks, quantitative values, equations, as-
sumptions, mechanisms, and next decisions as first-
class grounding objects. Third, we provide a task
suite for derivation, mechanism-from-evidence, dis-
crepancy explanation, and next-experiment design,
each requiring supporting evidence IDs. Fourth, we
describe a validated release pipeline with schema,
evidence-integrity, quantitative-grounding, license,
unit-normalization, duplicate, and content-quality
gates. Finally, we provide a de-leaked evaluation
harness with preliminary baselines that separate
deterministic evidence retrieval from answer gener-
ation on the same held-out tasks.
As far as we know, BioPhys-Bridge is the first
benchmark for grounding LMs in interdisciplinary
scientific reasoning in physics-grounded biological
research, where LM must connect evidence IDs to
physics equations, model assumptions, biological
mechanisms, and guide experiment design. It is
useful for studying faithfulness, attribution, and
hallucination in real-world research scenarios.
2 Related Work
2.1 Grounded Language Models and
Scientific RAG
Scientific RAG systems aim to answer research
questions by retrieving and synthesizing evidence
from papers. PaperQA and LitQA study retrieval-
augmented question answering over scientific lit-
erature (Lala et al., 2023). LitSearch evaluates
realistic scientific literature search queries over re-
cent papers (Ajith et al., 2024). LAB-Bench mea-
sures biology research capabilities, including lit-
erature reasoning, database use, figure interpreta-
tion, and sequence manipulation (Laurent et al.,
2024). SPIQA targets multimodal question answer-
ing over scientific paper figures (Pramanick et al.,
2024). These resources make scientific documents
central to evaluation, but their primary supervision
is usually an answer, paper-level evidence, retrievaltarget, or document QA context. BioPhys-Bridge
instead makes evidence blocks, evidence IDs, quan-
titative values, units, equations, model assumptions,
and mechanism links explicit fields in each case,
so attribution can be evaluated beyond passage re-
trieval.
BioPhys-Bridge also builds on scientific doc-
ument parsing and evidence extraction. MinerU
provides open-source document content extraction
for complex PDFs, including layout, table, formula,
and OCR processing (Wang et al., 2024a). In our
pipeline, MinerU is the parsing layer from which
normalized evidence blocks are derived, audited,
and linked to structured scientific reasoning fields.
2.2 Scientific Reasoning and Multimodal
Benchmarks
Scientific reasoning benchmarks increasingly test
difficult science questions. ScienceQA introduced
a multimodal science question-answering bench-
mark with explanations (Lu et al., 2022). SciBench
evaluates college-level scientific problem solving
across mathematics, chemistry, and physics (Wang
et al., 2024b). GPQA uses expert-written graduate-
level biology, physics, and chemistry questions to
stress scalable oversight (Rein et al., 2024). Math-
Vista and MMMU combine visual understanding
with expert or college-level reasoning (Lu et al.,
2024; Yue et al., 2024), while LongBench tests
long-context understanding over extended inputs
(Bai et al., 2024). These benchmarks are valuable
tests of scientific knowledge, reasoning, and mul-
timodal understanding, but they generally do not
represent a source-paper case as an evidence-linked
chain from measurement to physical model to bio-
logical mechanism and scientific decision.
2.3 AI-for-Science and Physical-Model
Benchmarks
Scientific machine-learning resources such as
PDEBench and BubbleML provide physics-
grounded data, benchmark tasks, and baselines
for simulation and surrogate modeling (Takamoto
et al., 2022; Hassan et al., 2023). Other real-
istic agent benchmarks, including SWE-bench,
WebArena, AgentBench, MINT, ToolBench, and
MLE-bench, evaluate software repair, web naviga-
tion, tool use, and machine-learning engineering
with functional, human-referenced, or environment-
based scoring (Jimenez et al., 2024; Zhou et al.,
2024; Liu et al., 2024; Wang et al., 2024c; Qin
et al., 2024; Chan et al., 2025). BioPhys-Bridge

is complementary: it does not benchmark a PDE
solver, surrogate model, or general-purpose tool
agent directly. Instead, it evaluates source-paper
grounding for quantitative and mechanistic scien-
tific interpretation. BioPhys-Bridge sits between
literature-grounded QA and AI-for-science bench-
marks: it evaluates whether grounded language
models can use source evidence to support quanti-
tatively and mechanistically meaningful scientific
interpretations.
3 Dataset Design
3.1 Evidence-Grounded Scientific Reasoning
Cases
A BioPhys-Bridge record is a structured grounding
object rather than a standalone QA item. The core
reasoning pattern is:
evidence→quantitative value→physical model
→mechanism→next decision.
Figure 1 summarizes this structure. The
schema separates the biological setting
from the physical model family. The field
domain captures the scientific area, while
biophysical_model.model_family records the
primary physical modeling strategy used by the
main equation and decision. Evidence blocks are
the atomic citation units: all supporting evidence
IDs in tasks must resolve to these blocks, and
quantitative evidence must be traceable to cited
evidence text.
Appendix A gives a concrete expert-annotated
gold-case schematic for the same evidence-to-
decision structure.
3.2 Record Fields
Each case includes paper provenance, evidence
blocks, quantitative evidence, model annotation,
mechanism annotation, a staged trajectory, agent
tasks, and quality metadata. Table 2 gives the main
fields. The source-of-truth schema is implemented
in Pydantic and exported as JSON Schema; un-
known fields are forbidden and cross-field invari-
ants are checked during validation. The expected
model output for each task includes both an answer
and supporting evidence IDs, making evidence at-
tribution part of the task rather than post-hoc expla-
nation.3.3 Task Types
The release contains four task types, each de-
signed to test a different grounding challenge.
derivation tasks ask a model to use a physical
relation or equation while citing the relevant
evidence IDs. mechanism_from_evidence
tasks require a biological mechanism in-
ferred from cited quantitative or textual
evidence. discrepancy_explanation tasks
require an explanation of tension between
expected physical behavior and observed evi-
dence, grounded in the case’s evidence blocks.
next_experiment_design tasks require a follow-
up experiment or computation grounded in cited
evidence and model assumptions.
4 Data Curation Pipeline
The data curation pipeline converts scientific re-
search papers into grounding benchmark records.
It is designed to be traceable, resumable, and com-
pliant with source licensing. Figure 2 sketches the
main stages; the archived release audit funnel is
reported in Appendix B.
Source acquisition and batching.Candidate pa-
pers are drawn from open-access scientific sources
with DOI and/or PMCID provenance and release-
compatible licenses. Candidate metadata includes
a title, source URL, license, abstract, domain guess,
model-family guess, and model or quantitative key-
words. Papers are selected in batches while track-
ing coverage across domains and physical model
families.
Document parsing and evidence block construc-
tion.Each PDF is parsed with MinerU. The raw
PDF files and MinerU intermediate payloads are re-
tained outside the public release, while normalized
evidence artifacts are used to build release records.
Normalization converts parsed outputs into canon-
ical document folders with markdown, structured
JSON, evidence blocks, and parse metadata. Evi-
dence blocks are preserved as the atomic citation
units used by model prompts and evidence-ID scor-
ing.
Grounded candidate extraction and structur-
ing.A regex-first pass extracts candidate numeric
values, equations, units, model keywords, and
evidence-rich snippets from prose, tables, formu-
las, and figure captions. An evidence-only LLM
pass may then structure fields such as quantita-
tive evidence, physical interpretation, mechanism

Benchmark family Core unit Primary strength Gap addressed by BioPhys-Bridge
PaperQA, LitQA (Lala
et al., 2023); LAB-
Bench (Laurent et al.,
2024)Scientific papers and
research questionsLiterature-grounded QA, biol-
ogy research skills, RAG eval-
uationEquations, units, assumptions, and
mechanism links are not first-class
case fields.
ScienceQA (Lu et al.,
2022); SciBench
(Wang et al., 2024b);
GPQA (Rein et al.,
2024)Science questions or
problemsScientific knowledge, explana-
tions, expert difficultyNo explicit evidence-to-equation-to-
mechanism case object.
MathVista (Lu et al.,
2024); MMMU (Yue
et al., 2024); SPIQA
(Pramanick et al.,
2024)Multimodal questions
and paper figuresVisual reasoning and scientific
document QAPhysical models and biological mech-
anisms are not first-class grounding
targets.
PDEBench (Takamoto
et al., 2022); Bub-
bleML (Hassan et al.,
2023)Simulation data and
physical benchmark
tasksPhysics-grounded prediction
with controlled ground truthFocuses on numerical modeling
rather than literature-grounded mech-
anism decisions.
BioPhys-Bridge (ours) Evidence-linked scien-
tific reasoning casesEvidence IDs, quantitative val-
ues, equations, assumptions,
mechanisms, and next decisionsDesigned for evidence-grounded
equation-to-mechanism reasoning
over biophysical literature.
Table 1: Positioning BioPhys-Bridge against recent benchmark families. The distinguishing unit is not a question
alone, but a structured case linking source evidence, physical model, quantitative evidence, biological mechanism,
and scientific decision.
text, and agent tasks. The shipped release used
OpenAI gpt-4o for this evidence-only structuring
pass. The prompt instructs the model to use only
supplied evidence and to leave unsupported fields
empty rather than hallucinate measurements, equa-
tions, assumptions, or mechanisms. Unsupported
or fabricated evidence IDs are removed by release
validation.
Release export.Final export writes JSONL
records, deterministic splits, metadata, schema, a
dataset card, gold samples, and a Sci-Evo view.
Source PDFs, raw MinerU payloads, and LLM re-
sponses are not redistributed.
Expert annotation layer.Expert annotation is
tracked separately from release-gate review. The
full held-out test split, 10 contest gold sam-
ples, 30 extended-gold samples, and one legacy
reviewed record include physics reasoning, bi-
ological reasoning, uncertainty, and reviewer
notes. This separation prevents the release-gate
field quality.manual_review_status from be-
ing mistaken for a claim that all 500 cases have full
expert biological and physical review.
5 Release Statistics
Table 3 summarizes the current release. The dataset
contains 500 cases and 1,517 agent tasks. Thesplit is deterministic and stratified by domain, with
400 training cases, 50 validation cases, and 50 test
cases. It is also grouped by source paper under
DOI/PMCID/title keys: the train, validation, and
test splits share zero source papers. The release in-
cludes 107 cases with an explicit failure or revision
stage, and 81 cases with expert annotation. The
expert annotation count covers the 50 held-out test
cases, 10 contest gold samples, 30 extended-gold
samples, and one legacy reviewed record.
Table 4 shows the weighted domain coverage.
The release is not intended to be perfectly balanced.
Instead, it emphasizes coverage across biophys-
ical discovery settings while preserving enough
examples in the larger protein-ligand and systems-
biology groups to support model training and eval-
uation.
The primary physical model families include
binding thermodynamics (185), systems stochas-
tic dynamics (128), conformational/allosteric en-
ergy landscapes (93), polymer phase-separation sta-
tistical mechanics (41), enzyme reaction kinetics
(23), folding stability thermodynamics (20), spatial
transport and electrostatics (6), evolutionary fitness
landscapes (2), and mechanical force response (2).
The last three families are included for coverage
and future expansion, not for standalone per-family
evaluation in this release.

Figure 1: Each case as an evidence-linked bridge from source-paper evidence to quantitative values, physical-model
assumptions, mechanistic interpretation, and an agent-facing scientific decision task.
Field Purpose
case_idStable identifier for an evidence-grounded case.
domain,bridge_typeBiological setting and domain-specific physics-to-mechanism bridge.
sourceDOI/PMCID, title, license, source URL, and MinerU parse identifier.
evidence[]Text, table, formula, and figure/caption blocks with stable evidence IDs.
quantitative_evidence[]Metric/value/unit records citing evidence IDs.
biophysical_modelModel name, family, equation, variables, assumptions, validity conditions.
physical_interpretationDerived quantity, directionality, consistency notes, and caveats.
biological_mechanismMechanism type and text linking physical result to biology.
sci_evo_trajectory[]Stages from research question through observation, interpretation, and next step.
agent_tasks[]Evaluation tasks with gold answers and supporting evidence IDs.
expert_annotationExpert review notes when present.
qualityValidation status, review status, score, provenance, and audit metadata.
Table 2: Dataset schema of BioPhys-Bridge. Evidence IDs, quantitative values, equations, assumptions, mechanisms,
and decisions are explicit grounding targets.
Release item Value
Cases 500
Train / validation / test 400 / 50 / 50
Agent tasks 1,517
Failure/revision cases 107
Expert-annotation cases 81
Mean modalities per case 2.944
Cases with 3+ modalities 80.4%
Schema-valid rate 100%
Evidence coverage rate 100%
Quantitative-evidence rate 100%
Source-license coverage 100%
Duplicate rate 0%
Table 3: BioPhys-Bridge v1 release at a glance.
6 Quality Control and Validation
Quality control is central to the release. Table 6
summarizes the main gates. A shipped case must
validate against the schema, cite valid evidence
IDs, ground quantitative values in cited evidence
text, carry source license metadata, pass duplicate
checks, and avoid unresolved templates or weak
task prompts. Unit normalization retains both raw
and normalized units when normalization is sup-Biological domain Cases
protein_ligand_binding188
systems_biology_dynamics126
conformational_dynamics_allostery91
biomolecular_phase_separation52
enzyme_kinetics22
protein_stability_thermodynamics21
Table 4: Biological-domain coverage in the 500-case
release.
ported. The content gate blocks examples with
unsupported measurements, fabricated evidence
IDs, evidence-ID-only prompts, unresolved tem-
plate strings, malformed tool vocabularies, and
missing next-step stages.
The field quality.manual_review_status =
reviewed should be read as a release-gate sta-
tus, not as a claim that all 500 cases have full
expert physics and biology annotation. Full ex-
pert annotations are tracked separately through the
expert_annotationfield.

Figure 2: Curation pipeline. Open-access papers are parsed with MinerU, normalized into evidence blocks,
structured into case records, and filtered by validation and quality gates before release.
Task type Count
mechanism_from_evidence480
next_experiment_design403
derivation355
discrepancy_explanation279
Table 5: Agent-task distribution.
7 Evaluation Protocol and Baselines
Evaluation setting.Given a task prompt and can-
didate evidence blocks, the model must produce
a JSON answer and a set of supporting evidence
IDs. The scorer computes evidence-ID F1 against
the gold supporting evidence IDs and answer to-
ken F1 against the gold answer. Evidence-ID F1
evaluates whether the model grounds its answer in
the correct evidence blocks. Answer token F1 is
a rough sanity-check metric for this initial release,
not a final measure of scientific correctness. The
released harness also computes overall_score as
0.7answer token F1 plus 0.3evidence-ID F1 for
diagnostics, but we do not use that composite met-
ric for the main paper because token F1 is weak for
open-ended scientific answers.
De-leaked prompt construction.The evaluation
prompt hides gold answers, gold supporting evi-
dence IDs, and expert annotations. The main de-
leaked prompt exposes the task domain, task type,
question, and ranked candidate evidence blocks,
but not the structured equation, physical direc-
tionality, or biological mechanism fields. We use
this no-scaffold prompt for the main results be-
cause those structured fields can be answer-bearing
for derivation or mechanism tasks. Candidate ev-
idence selection uses a lexical ranker over only
the task type, task question, research question, do-main, and source evidence text; it does not ac-
cess task.supporting_evidence_ids or struc-
tured target annotations such as equations, physical
directionality, biological mechanisms, or quanti-
tative value records. Gold IDs are used only by
the scorer after prediction. In the released har-
ness, prompt construction and candidate selection
operate on public case/task views that exclude
gold_answer ,supporting_evidence_ids , and
expert_annotation ; regression tests fail if these
fields reach the model-visible evaluation path. Each
test task is shown 48 candidate evidence blocks se-
lected from a median 206 evidence blocks per case;
the deterministic lexical baseline cites the top 3
ranked blocks. This candidate generator contains
234 of 267 gold supporting evidence IDs on the
test set (0.876 gold-ID recall), with all gold IDs
present for 127 of 154 tasks; the mean maximum
achievable evidence-ID F1 under the candidate set
is 0.916. Thus the reported scores evaluate attribu-
tion and re-ranking within a lexically pre-filtered
candidate set, not unconstrained retrieval over ev-
ery evidence block. Regression tests verify that
prompts and the lexical baseline do not copy gold
evidence IDs. All LLM rows use temperature 0
and JSON-mode decoding. Confidence intervals
are nonparametric bootstrap 95% intervals over
tasks.
Paired bootstrap over the same 154 tasks
gives evidence-ID F1 deltas over lexical re-
trieval of +0.172 for DeepSeek v4 Flash (95%
CI[0.117,0.227] ),+0.127 for Qwen 3.7 Max
(95% CI [0.076,0.179] ), and +0.106 for GPT-
4o-mini (95% CI [0.061,0.152] ). Claude Opus
4.8 shows a smaller positive delta ( +0.050 ,
95% CI [0.005,0.094] ), while Claude Sonnet
5 is not significantly above lexical retrieval

Gate Check Release result
Schema validation Pydantic and generated JSON Schema 500 / 500 valid
Evidence integrity Cited IDs resolve to evidence blocks 100% coverage
Quantitative grounding Values appear in cited evidence text 100% coverage
Source compliance Release-compatible source licenses 490 CC-BY-4.0, 10 CC0-1.0
Unit normalization Raw and normalized units retained 100% success
Duplicate check Stable case IDs are unique 0 duplicates
Content gate No unresolved templates or weak tasks 100% pass
Expert annotation Selected gold and test records 81 cases
Audit notes Deterministic notes for shipped cases 100% note coverage
Table 6: Release quality gates. The deterministic audit notes are a supporting traceability aid; only about 2% of
cases have an implemented closed-form relation-level check, so it is not used as a global correctness metric.
Model Provider Evidence-ID F1 Answer token F1 Empty EID outputs
Lexical retrieval deterministic 0.188 [0.151, 0.226] 0.058 [0.053, 0.065] 0/154
DeepSeek v4 Flash DeepSeek 0.360 [0.309, 0.412] 0.193 [0.173, 0.214] 0/154
Qwen 3.7 Max OpenRouter 0.316 [0.273, 0.359] 0.129 [0.115, 0.143] 1/154
GPT-4o-mini OpenRouter 0.294 [0.251, 0.339] 0.163 [0.149, 0.177] 0/154
Claude Opus 4.8 OpenRouter 0.238 [0.205, 0.272] 0.095 [0.083, 0.107] 18/154
Claude Sonnet 5 OpenRouter 0.224 [0.190, 0.258] 0.077 [0.068, 0.086] 21/154
DeepSeek v4 Pro DeepSeek 0.111 [0.070, 0.156] 0.046 [0.029, 0.065] 127/154
Gemini 2.5 Pro OpenRouter 0.010 [0.000, 0.023] 0.099 [0.084, 0.115] 151/154
Table 7: Preliminary de-leaked no-scaffold evaluation on the identical 154-task held-out test set. Values in
brackets are bootstrap 95% confidence intervals over tasks. Empty EID outputs counts predictions with no parsed
supporting_evidence_ids; this is an output-compliance measure as well as an attribution failure mode.
under this paired bootstrap ( +0.035 , 95% CI
[−0.010,0.081] ). DeepSeek v4 Flash also outper-
forms Qwen 3.7 Max by +0.045 evidence-ID F1
(95% CI [0.012,0.079] ). We retain a scaffolded
prompt mode for diagnostic audits, but exclude it
from the main baseline because structured fields
can be answer-bearing, including equations that
match derivation gold answers.
These baselines are intentionally preliminary.
The same-test comparison shows that several
LLMs improve over lexical retrieval in aggregate
evidence attribution, with the largest gains on
derivation , discrepancy_explanation ,
and mechanism_from_evidence tasks.
next_experiment_design remains harder:
even the strongest model reaches only 0.216
evidence-ID F1, consistent with the task’s open-
ended decision structure. The panel also exposes a
grounded-generation failure mode: some models
produce plausible prose but fail to emit usable
evidence IDs. Gemini 2.5 Pro has nontrivial
answer-token overlap (0.099) but empty parsed
evidence IDs for 151 of 154 tasks, making its
attribution score near zero. DeepSeek v4 Pro
similarly leaves 127 of 154 outputs without parsed
evidence IDs. Answer token F1 therefore remains
a weak proxy for grounded scientific answer
quality, while evidence-ID F1 captures whether
the answer is actually attributable to the suppliedevidence blocks.
Because scientific answers are open-ended, fu-
ture releases should add rubric-based expert scor-
ing for physical-model use, quantitative consis-
tency, mechanism correctness, uncertainty han-
dling, and experimental feasibility. This is es-
pecially important for grounded scientific agents,
where a cited but mechanistically wrong answer
can still receive partial evidence overlap. The 50
test cases have expert annotation notes, but they are
not independent parallel task-level evidence-ID la-
bels, so we do not report inter-annotator agreement
for supporting evidence IDs in this release.
8 Dataset Access and Reproducibility
For peer review, the dataset, schema, aggregate re-
ports, and evaluation outputs are supplied in the
supplementary materials. The public dataset and
code repository will be released after peer review.
The repository will contain code, schema, reports,
validation tests, small samples, aggregate sum-
maries, and release files. Code will be released
under the MIT License, while curated dataset meta-
data, reports, and public release files will be re-
leased under CC-BY-4.0. Upstream source papers
retain their original licenses, recorded per case.
Raw PDFs, raw MinerU payloads, and LLM re-
sponses are not redistributed.
The public release will expose validation and

Task type N Lex. Flash Qwen GPT-mini
derivation39 0.192 0.475 0.380 0.346
discrepancy31 0.201 0.404 0.379 0.350
mechanism46 0.188 0.352 0.311 0.306
next experiment38 0.173 0.216 0.203 0.181
Table 8: Held-out evidence-ID F1 by task type for the lexical floor and the three strongest LLM rows in Table 7.
Each task-type cell contains only 31–46 tasks, so these point estimates should be read as noisier diagnostics rather
than stable per-type rankings.
evaluation commands through the biophysevo
Python package. The source-of-truth schema lives
insrc/biophysevo/schemas/case_schema.py ,
and the generated JSON Schema is included in
the supplementary materials.
9 Limitations and Responsible Use
BioPhys-Bridge is an initial benchmark release, not
a substitute for reading the original papers and not
a complete solution to biological reasoning. Users
should verify important claims against the cited
source literature before using them in research, en-
gineering, clinical, or safety-critical settings. Ex-
pert annotation covers 81 cases, not all 500. Cur-
rent baselines are preliminary: under the fully de-
leaked public-query setting, they include a lexical
retrieval floor and seven LLMs, but not a human
ceiling, hidden test server, repeated stochastic runs,
or full rubric-based expert grading. Evidence-ID
F1 captures attribution to evidence blocks but not
full scientific correctness, and token F1 is a weak
proxy for open-ended answers. The expert layer
checks physical and biological reasoning for the
test cases, but it does not provide independent par-
allel task-level labels for supporting evidence IDs;
future versions should add inter-annotator agree-
ment and expert scoring for mechanism correctness.
Because the release structuring pass used GPT-
4o, shared-provider bias remains a caveat for the
GPT-4o-mini row, although the strongest reported
evidence-ID scores come from non-OpenAI sys-
tems. Model routing and output-format behavior
also affect measured attribution: systems that omit
parseable supporting_evidence_ids can receive
very low evidence-ID F1 even when their prose an-
swer overlaps the reference answer.
The test set is public in this release, which sup-
ports reuse and workshop discussion but limits
long-term contamination control; future versions
should include hidden or contamination-aware eval-
uation. The source pool is open-access and release-
compatible, which induces domain and source bias.
Domain coverage is weighted rather than balanced.OCR and table parsing artifacts can remain in
evidence text, because preserving traceability to
parsed source artifacts is prioritized over silently
rewriting source evidence. The archived release
metadata begins at the reviewed candidate set rather
than the full exploratory source pool, so future ver-
sions should report complete rejection and parser-
failure counts.
10 Conclusion
We introduce BioPhys-Bridge, an initial bench-
mark for evidence-grounded scientific reasoning
over biophysical literature. The current 500-case
release is reusable and validated enough for com-
munity exploration: it represents evidence blocks,
quantitative values, equations, assumptions, mecha-
nisms, and next scientific decisions in a shared case
schema, and it provides de-leaked baselines for
evidence-ID attribution. Preliminary same-test re-
sults show that several LLMs improve over lexical
retrieval for evidence attribution, with DeepSeek
v4 Flash, Qwen 3.7 Max, and GPT-4o-mini produc-
ing the strongest evidence-ID F1 in our initial panel.
The results also show that grounded answering is
not only answer generation: models can produce
plausible prose while failing to emit usable evi-
dence IDs. This workshop version is intended to
invite feedback on evaluation design, expert-rubric
scoring, contamination-aware protocols, and future
grounded scientific-agent benchmarks that test not
only what a paper says, but how source evidence
supports quantitative and mechanistic scientific de-
cisions.
Acknowledgments
Acknowledgments are omitted for anonymous re-
view.
Declaration on Generative AI
During the preparation of this work, the author
used ChatGPT for brainstorming, paper drafting
and editing, and code scaffolding, and used Claude

Code for code implementation. The author re-
viewed and edited the entire content and takes full
responsibility for the publication’s content.
References
Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya
Goyal, Danqi Chen, and Tianyu Gao. 2024. Lit-
search: A retrieval benchmark for scientific literature
search. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 15068–15083, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. Longbench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.
Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James
Aung, Dane Sherburn, Evan Mays, Giulio Starace,
Kevin Liu, Leon Maksin, Tejal Patwardhan, Alek-
sander Madry, and Lilian Weng. 2025. Mle-bench:
Evaluating machine learning agents on machine
learning engineering. InInternational Conference on
Learning Representations.
Sheikh Md Shakeel Hassan, Arthur Feeney, Akash
Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu,
Yoonjin Won, and Aparna Chandramowlishwaran.
2023. Bubbleml: A multiphase multiphysics dataset
and benchmarks for machine learning. InAdvances
in Neural Information Processing Systems, Datasets
and Benchmarks Track.
Carlos E. Jimenez, John Yang, Alexander Wettig,
Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. 2024. Swe-bench: Can language mod-
els resolve real-world github issues? InInternational
Conference on Learning Representations.
Jakub Lala, Odhran O’Donoghue, Aleksandar Sht-
edritski, Sam Cox, Samuel G. Rodriques, and An-
drew D. White. 2023. Paperqa: Retrieval-augmented
generative agent for scientific research.Preprint,
arXiv:2312.07559.
Jon M. Laurent, Joseph D. Janizek, Michael Ruzo,
Michaela M. Hinks, Michael J. Hammerling, Sid-
dharth Narayanan, Manvitha Ponnapati, Andrew D.
White, and Samuel G. Rodriques. 2024. Lab-bench:
Measuring capabilities of language models for biol-
ogy research.Preprint, arXiv:2407.10362.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu
Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen
Men, Kejuan Yang, et al. 2024. Agentbench: Evalu-
ating llms as agents. InInternational Conference on
Learning Representations.Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun-
yuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-
Wei Chang, Michel Galley, and Jianfeng Gao. 2024.
Mathvista: Evaluating mathematical reasoning of
foundation models in visual contexts. InInterna-
tional Conference on Learning Representations.
Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei
Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark,
and Ashwin Kalyan. 2022. Learn to explain: Multi-
modal reasoning via thought chains for science ques-
tion answering. InAdvances in Neural Information
Processing Systems, volume 35, pages 2507–2521.
Shraman Pramanick, Rama Chellappa, and Subhashini
Venugopalan. 2024. Spiqa: A dataset for multimodal
question answering on scientific papers. InAdvances
in Neural Information Processing Systems, Datasets
and Benchmarks Track.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan
Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang,
Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian,
Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li,
Zhiyuan Liu, and Maosong Sun. 2024. Toolllm: Fa-
cilitating large language models to master 16000+
real-world apis. InInternational Conference on
Learning Representations.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R. Bowman. 2024. Gpqa:
A graduate-level google-proof q&a benchmark. In
First Conference on Language Modeling.
Makoto Takamoto, Timothy Praditia, Raphael Leiteritz,
Dan MacKinlay, Francesco Alesiani, Dirk Pflüger,
and Mathias Niepert. 2022. Pdebench: An exten-
sive benchmark for scientific machine learning. In
Advances in Neural Information Processing Systems,
Datasets and Benchmarks Track.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, et al. 2024a. Mineru: An open-
source solution for precise document content extrac-
tion.Preprint, arXiv:2409.18839.
Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu
Zhang, Satyen Subramaniam, Arjun R. Loomba,
Shichang Zhang, Yizhou Sun, and Wei Wang.
2024b. Scibench: Evaluating college-level scientific
problem-solving abilities of large language models.
InProceedings of the 41st International Conference
on Machine Learning, pages 50622–50649. PMLR.
Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen,
Lifan Yuan, Hao Peng, and Heng Ji. 2024c. Mint:
Evaluating llms in multi-turn interaction with tools
and language feedback. InInternational Conference
on Learning Representations.
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng,
Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang,
Weiming Ren, Yuxuan Sun, et al. 2024. Mmmu: A
massive multi-discipline multimodal understanding

and reasoning benchmark for expert agi. InPro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition.
Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou,
Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue
Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Gra-
ham Neubig. 2024. Webarena: A realistic web envi-
ronment for building autonomous agents. InInterna-
tional Conference on Learning Representations.
A Gold Case Example
B Release Audit Funnel
Archived release stage Count
Reviewed source records 500
Schema-valid source records 500
Exported case records 500
Deterministic train split 400
Deterministic validation split 50
Deterministic test split 50
Expert-annotated records 81
Table 9: Release audit funnel recorded in the pub-
lic metadata. Future dataset versions should addition-
ally archive exploratory source-pool and parser-failure
counts to support a full rejection funnel.
C Prompt Template
The evaluation prompt is generated by the released
harness as follows:
You are answering an evidence-grounded
scientific question for biological
research. Use only the evidence
below. Return JSON with keys answer
and supporting_evidence_ids. Choose
supporting_evidence_ids from the
candidate evidence IDs below; gold IDs
are not provided. Domain: <domain>.
Task type: <task_type>. Evidence:
<ranked evidence blocks>. Question:
<task input>.
For evaluation, gold answers, gold supporting ev-
idence IDs, and expert annotations are hidden
from the prompt; gold labels are used only by the
scorer. The harness also retains a scaffolded diag-
nostic mode that adds Physical model ,Equation ,
Physical directionality , and Biological
mechanism lines; it is not used for the main de-
leaked results because those fields can be answer-
bearing.

Figure 3: A real BioPhys-Bridge case illustrates the benchmark object: source provenance, evidence-linked
quantitative measurements, a physical model, mechanism interpretation, caveats, and an agent-facing task are stored
together rather than flattened into a standalone question.