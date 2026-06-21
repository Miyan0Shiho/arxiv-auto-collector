# Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio

**Authors**: Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai

**Published**: 2026-06-15 17:56:41

**PDF URL**: [https://arxiv.org/pdf/2606.17041v3](https://arxiv.org/pdf/2606.17041v3)

## Abstract
Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds.
  Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not.

## Full Text


<!-- PDF content starts -->

Benchmarking LLM Agents on Meta-Analysis Articles from
Nature Portfolio
Anzhe Xie
xaz0314@gmail.com
Tsinghua University
Beijing, ChinaWeihang Su
swh22@mails.tsinghua.edu.cn
Tsinghua University
Beijing, ChinaYujia Zhou
zhouyujia@mail.tsinghua.edu.cn
Tsinghua University
Beijing, China
Yiqun Liu
yiqunliu@tsinghua.edu.cn
Tsinghua University
Beijing, ChinaQingyao Ai∗
aiqingyao@gmail.com
Tsinghua University
Beijing, China
Abstract
Meta-analysis is a demanding form of evidence synthesis that com-
bines literature retrieval, PI/ECO-guided study selection, and sta-
tistical aggregation. Its structured, verifiable workflow makes it
an ideal substrate for evaluating systematic scientific reasoning,
yet existing benchmarks lack ground truth across the full retrieval-
screening-synthesis pipeline. We introduce MetaSyn, a dataset of
442 expert-curated meta-analyses from Nature Portfolio journals.
Each entry pairs a research question with PI/ECO criteria, a re-
trieval corpus of 140k PubMed articles, verified positive studies,
hard negatives that are topically similar but PI/ECO-ineligible, and
complete search strategies and date bounds.
Benchmarking twelve pipeline configurations (nine RAG vari-
ants and a protocol-driven agent) reveals a critical screening bot-
tleneck: despite a retrieval ceiling of 90.9% recall at 𝐾= 200, no
system recovers more than 52.7% of ground-truth included litera-
ture. Current LLMs fail to reliably separate eligible studies from
PI/ECO-failing distractors in pools of comparable topical relevance.
Stage-attributed metrics capture where systems succeed and fail; a
single end-to-end score does not.
CCS Concepts
•Computing methodologies→Information retrieval;Natu-
ral language processing;Evaluation of retrieval results;•Informa-
tion systems→Digital libraries and archives;•Human-centered
computing→Interactive systems and tools.
Keywords
Meta-analysis, Systematic review, Dataset, Benchmark, Evidence
synthesis, Large language model, Retrieval-augmented generation
ACM Reference Format:
Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗. 2026.
Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio.
InProceedings of arXiv Preprint.ACM, New York, NY, USA, 13 pages. https:
//doi.org/XXXXXXX.XXXXXXX
* Corresponding author.
Code: https://github.com/BFTree/MetaSyn; Dataset: https://huggingface.co/datasets/
BFTree/MetaSyn.
arXiv Preprint,
2026. ACM ISBN 978-1-4503-XXXX-X/26/07
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Scientific evidence is growing faster than any individual reader can
keep up with. In evidence-based medicine, this pressure has been
described as the appearance of dozens of trials and systematic re-
views each day, with unreliable synthesis contributing to avoidable
research waste and delayed translation [ 1,2,9]. Meta-analysis is
one of the main instruments for addressing this challenge, but it
is not merely a way to summarize a large body of literature. It is
a protocol-governed procedure for determining which studies are
eligible for synthesis and how their results should be combined.
Under an explicit protocol, meta-analysis turns a set of primary
studies into a reproducible estimate through documented decisions
on search, screening, extraction, and synthesis [ 4,12,16,30]. Across
theNaturePortfolio, recent meta-analyses synthesize evidence on
questions ranging from clinical safety to environmental interven-
tion [ 5,17,55]. In this sense, meta-analyses do more than summarize
a topic: they define the evidentiary boundary of a research ques-
tion, document the inclusion and exclusion of candidate studies,
and make an aggregate claim that can be audited.
This combination of scale, structure, and protocol dependence
makes meta-analysis an attractive but difficult target for AI as-
sistance. Much of the workflow is labor-intensive: constructing
search strategies, screening titles and abstracts, assessing full texts
against eligibility criteria, extracting outcomes, and drafting evi-
dence summaries. Earlier work on systematic-review automation
has shown that text mining and machine learning can reduce the
burden of study identification and screening [27, 29, 53, 56]. More
recent LLMs and retrieval-augmented generation systems further
expand this possibility through stronger language understanding,
tool use, and long-form synthesis [ 3,11,24,26]. Yet the opportu-
nity is not simply to generate a fluent review; it is to accelerate
the workflow while preserving the eligibility reasoning that makes
the synthesis trustworthy. Meta-analyses are governed by criteria
often organized around Population, Intervention or Exposure, Com-
parison, and Outcome (PI/ECO) components [ 32,35]. A candidate
article may appear topically relevant but still be excluded because
it studies an ineligible population, comparator, outcome, follow-up
window, intervention or exposure definition, or study design. This
criterion-based exclusion separates meta-analysis from ordinary
scientific search and narrative survey generation, and it is precisely
where current end-to-end AI systems struggle to reproduce expert
protocol decisions reliably.arXiv:2606.17041v3  [cs.CL]  17 Jun 2026

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
Evaluating this capability remains difficult. Existing benchmarks
and systems cover important pieces of the landscape, including
automated survey writing, scientific agents, live research synthesis,
and clinical evidence review [ 31,50,61,62], but most either evalu-
ate the final text, focus on one stage, or use relevance labels that
do not encode the full protocol. Therefore, the exclusion logic that
determines trustworthiness goes largely unmeasured. Stage-level
resources for the complete workflow remain scarce in part because
published meta-analyses vary in how they report search strings,
date bounds, included studies, and supplementary tables, and be-
cause coverage across bibliographic databases is incomplete. As a
result, existing resources rarely provide verifiable targets for evalu-
ating whether agents can reproduce protocol-grounded decisions
across the full meta-analysis workflow.
To this end, we introduceMetaSyn, a dataset and benchmark
for evaluating LLM agents on the complete meta-analysis pipeline.
MetaSyn contains 442 meta-analyses from journals in theNature
Portfolio, distilled from an initial pool of over 34,000 candidate
articles. Each paper is paired with a PubMed-anchored corpus of
140,585 articles, including 8,674 corpus-matched positive studies
and 131,911 hard negatives. Each instance also includes expert-
verified PI/ECO research questions, search strategies, date bounds,
and inclusion and exclusion criteria. The positive studies are drawn
from the analyzed-study lists in the source publications, while the
hard negatives are topically close to positives but fail at least one
eligibility requirement. This design makes screening the central
challenge by construction, and it gives each stage a verifiable target:
retrieval is judged against the corpus-matched included studies,
screening against the original eligibility decisions, and synthesis
against the reported conclusion and key findings.
We benchmark a range of standard RAG pipelines and a protocol-
faithful agent on MetaSyn, and the main result is a sharp screening
bottleneck. MA-Retriever reaches 90.9% Recall@200, yet on the
same retrieved pool no end-to-end system recovers more than 52.7%
of the ground-truth included literature. The missing 38 points fall
almost entirely after retrieval, where systems must separate a small
number of eligible studies from hundreds of PI/ECO-failing distrac-
tors. This is the criterion-based judgment identified above, now
quantified. Beyond this gap, the pipelines settle into distinct oper-
ating regimes rather than a single ranking, so one aggregate score
would obscure where each succeeds. MetaSyn therefore reports
stage-attributed metrics and validates the evaluator-dependent ones
against 8-annotator judgments.
Our contributions are:
(1)MetaSyn: a dataset for benchmarking LLM agents on
meta-analysis.442 expert-curated meta-analyses from jour-
nals in theNaturePortfolio, spanning clinical and non-clinical
domains (Section 3.5), with verified positive studies, topically
matched hard negatives, and PI/ECO-structured research ques-
tions, search strategies, and inclusion and exclusion criteria.
Because meta-analysis follows a codified publication protocol,
the included-study list, eligibility criteria, and conclusion direc-
tion at each pipeline stage are derived from the source paper
without separate relabeling. This gives a structural guarantee
of cross-stage consistency that distinguishes MetaSyn from
benchmarks built by independent annotation.(2)Benchmark.End-to-end meta-analysis generation scored by
nine stage-attributed metrics (inclusion, screening, criteria ad-
herence, synthesis), with retrieval additionally measured in
isolation via Recall@K against corpus-matched ground truth so
that upstream and downstream failures can be diagnosed sepa-
rately rather than averaged into a single score. Five evaluator-
dependent metrics are validated against 8-annotator pairwise
judgments and assigned to reliability tiers that govern how they
are used in the experimental analysis.
(3)Baselines and analysis.MA-Retriever (a dense retriever fine-
tuned on MetaSyn), nine RAG configurations across three LLM
backbones and three retrievers, and three retrieval variants
of a protocol-driven meta-analysis agent, establishing stage-
attributable performance levels for current systems and show-
ing that the four pipeline families settle into four distinct oper-
ating profiles that a single aggregate score would collapse.
2 Preliminaries
2.1 The Meta-analysis Workflow
A meta-analysis proceeds in four ordered stages [ 4,30]. The first
stage is a comprehensive literature retrieval, conducted against
one or more academic databases using pre-specified search queries,
with the aim of maximizing recall so that no eligible study is missed.
The retrieved records are then screened against explicit eligibility
criteria, typically in two phases (title/abstract screening followed
by full-text screening), to progressively remove ineligible studies.
In the third stage, data are extracted from each included study:
effect sizes, outcome measures, sample sizes, and study-level char-
acteristics such as population and comparator. These extracted
values are then combined in the fourth stage through a formal
statistical model (most commonly fixed-effect or random-effects
meta-analysis), producing an overall estimate of the effect together
with an assessment of between-study heterogeneity. The final re-
port interprets this estimate in light of the pooled evidence and
identifies remaining sources of bias.
2.2 PI/ECO Criteria
PI/ECO criteria give the meta-analysis protocol its structured for-
mulation of the research question and its principled basis for study
selection [ 32,35]. PI/ECO specifies the Population under investiga-
tion, the Intervention or Exposure being evaluated, the Comparison
condition, and the Outcome of interest. Together, these components
define the scope of the evidence to be synthesized and constrain
which studies qualify. By grounding inclusion and exclusion deci-
sions in PI/ECO criteria, meta-analysis enforces that all selected
studies address a consistent research question and are comparable
along clinically or scientifically meaningful dimensions. Narrative
reviews leave inclusion decisions implicit; PI/ECO-grounded eligi-
bility makes screening reproducible and conclusions auditable. The
PI/ECO framework also shapes the retrieval problem in a specific
way. A study can be topically indistinguishable from a ground-truth
positive (same disease, same population, same outcome) yet fail
a single eligibility criterion such as a different control condition
or a sample size below a protocol-specified threshold. Retrievers
that match only on topical similarity are therefore insufficient for

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
meta-analysis screening, regardless of how well they perform on
generic scientific-search benchmarks.
2.3 An Illustrative Example
Consider:Does continuing antidepressant treatment after remission
prevent relapse in adults with major depressive disorder?[ 22] The
meta-analysis pre-specified four eligibility criteria:
•Population: adults with major depressive disorder who achieved
remission during acute antidepressant treatment
•Intervention: continuation of the same antidepressant (mainte-
nance therapy)
•Comparison: switch to placebo (antidepressant discontinuation)
•Outcome: relapse rate from end of acute treatment through
follow-up
The original authors identified 40 qualifying double-blind ran-
domized trials; 33 are recoverable in MetaSyn’s PubMed-anchored
corpus and constitute the retrieval ground truth for this query. Sur-
rounding these 33 are studies on antidepressants and depression
that each fail exactly one criterion: trials enrolling patients with
active (non-remitted) depression (P), trials comparing two antide-
pressants with no placebo arm (C), trials for bipolar or treatment-
resistant depression (P), and open-label or non-randomized studies
(design criterion). Recovering the correct 33 requires applying all
four PI/ECO gates simultaneously across a pool of hundreds.
3 Dataset Construction
MetaSyn is built from journal meta-analyses whose included-study
lists are directly recoverable from published supplementary mate-
rial, paired with a retrieval corpus sized to reflect realistic search
conditions.
3.1 Source Selection and Initial Retrieval
We drew source papers from journals in theNaturePortfolio. These
journals span diverse scientific domains, so the resulting dataset
is not specialized to a single field (a per-domain breakdown is
given in Section 3.5); their meta-analyses frequently publish the
list of analyzed studies as supplementary material, making stage-
level ground-truth extraction feasible; and their editorial process
imposes a consistent baseline for reporting completeness, keeping
the annotation task tractable at scale.
We searched for articles containing “meta-analysis” or “system-
atic review” across all journals in theNaturePortfolio, yielding
over 60,000 candidates. BecauseScientific Reportshas a publication
model and reviewer pool distinct from the rest of the family, we re-
stricted the pool to the remaining 34,375 articles (from journals such
asNature Medicine,Nature Communications, andNature Human
Behaviour) to keep the editorial baseline uniform.
3.2 Human Annotation and Filtering
We recruited approximately 50 annotators to manually review all
34,375 candidates. The annotation protocol comprised four stages:
(1)Accessibility verification: Annotators identified articles with
open access to enable full-text review.(2)Meta-analysis confirmation: Annotators read full texts to
confirm each article constitutes a genuine meta-analysis con-
ducting quantitative synthesis, rather than merely mentioning
the term or presenting a narrative review.
(3)Ground truth extraction: This critical step required careful
distinction between reference lists and analyzed studies. We
examined whether each article provides a complete list of stud-
ies included in the analysis (the specific primary studies whose
data were pooled), rather than the full reference list. A meta-
analysis may cite over 100 papers while analyzing only 20 to
30 studies meeting inclusion criteria. Ground truth extraction
required identifying supplementary tables, forest plot sources,
or explicit study enumeration in methods sections.
(4)Metadata extraction: For qualifying articles, annotators recorded
research questions, search databases and queries, search date
ranges, inclusion/exclusion criteria, and the number of included
studies.
The protocol was highly selective: many meta-analyses report
only aggregate statistics without listing individual studies; others
provide partial lists or use proprietary identifiers. After filtering,
442 papers remained with complete, extractable ground truth.
Annotation quality.Ground-truth quality rests on human
annotation throughout, with two-tier review at each stage. The
core decisions (confirming genuine meta-analysis status, extract-
ing included-study lists, and capturing search metadata) were per-
formed by human annotators with full document access: supple-
mentary tables, forest plots, and methods sections were all con-
sulted to distinguish analyzed studies from the broader reference
list. Ambiguous cases (papers with partial supplementary lists, non-
standard study identifiers, or format discrepancies between supple-
mentary tables and forest plots) were flagged and resolved by the
annotation team against the source publication. Stage-4 metadata
elements (search strategies, date bounds, inclusion and exclusion
criteria) were cross-checked by a second team member for all 442
entries, with disagreements adjudicated against the source text.
Structured research-question elements (the PI/ECO components)
are the one field class where LLM extraction assists in the initial
pass; human annotators reviewed and corrected all LLM outputs
before they entered the corpus. This two-stage process is described
in Section 3.3; human judgment remained the final authority at
every stage of the pipeline.
3.3 Research Question Structuring
Research questions vary considerably in how they are stated across
source papers, so we structured each one into PI/ECO components
via a two-stage pipeline: GLM-4.6 [ 13] parsed article abstracts and
introductions to produce an initial extraction, and human annota-
tors then reviewed and corrected every field before it entered the
corpus.
3.4 Corpus Construction
To support the retrieval task, we constructed a corpus of 140,585
articles:
Positive samples (8,674 articles): These represent studies ac-
tually included in at least one meta-analysis. For each paper’s list

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
of analyzed studies, we queried PubMed using titles to retrieve full
content elements.
Hard negative samples (131,911 articles): For each positive
study, we queried PubMed with its title to retrieve topically sim-
ilar but non-included articles, producing the approximately 1:15
positive-to-negative ratio described in Section 2.3. These are studies
a retrieval system would plausibly surface but that fail the meta-
analysis eligibility criteria on at least one PI/ECO criterion.
Content representation: 74.5% of corpus articles carry at least
one structured section beyond the abstract (introduction, methods,
results, or conclusions) recovered from PubMed records where
available. Systems that reason over full-section content rather than
abstracts alone can be evaluated on that subset without a separate
data collection pass.
3.5 Dataset Statistics
Table 1 summarizes MetaSyn statistics. MetaSyn contains 8,674
ground-truth positive studies across 442 meta-analyses; the test
split averages 19.2 per paper. The shortfall from reported totals re-
flects the multi-database nature of systematic search: meta-analyses
routinely draw from EMBASE, the Cochrane Central Register, and
trial registries alongside PubMed, so a PubMed-anchored corpus
captures the largest single indexed source rather than the union.
Each entry also carries its original search end date (99.3% coverage),
enabling time-bounded evaluation.
Domain coverage.We group the 442 papers into eleven domains
by source journal; the distribution is shown in Figure 1. The largest
groups are oncology (16.5%), mental health and neuroscience (14.3%),
other clinical specialties (11.1%), and metabolism and nutrition
(10.4%); together with cardiovascular medicine (4.8%), dental and
oral health (7.2%), and reproductive and perinatal medicine (3.4%),
clinical specialties account for 67.6% of the dataset. The remain-
ing 32.4% comes from non-clinical domains: general biology and
multidisciplinary topics (9.5%), digital health and AI in medicine
(9.3%), social sciences and humanities (9.0%), and environmental
and planetary health (4.5%). MetaSyn is therefore not a clinical-only
benchmark; the same evaluation pipeline applies to behavioural
and environmental meta-analyses, which differ markedly in PI/ECO
structure and reporting conventions.
The domain distribution also implies heterogeneous retrieval
conditions. Clinical meta-analyses, oncology in particular, draw
more extensively from EMBASE, specialty trial registries, and con-
ference proceedings than from PubMed alone, which depresses
their corpus match rate. Non-clinical meta-analyses in environ-
mental and social-science domains rely more heavily on PubMed-
indexed journals, and their match rates are correspondingly higher.
This database-coverage gap means that aggregate retrieval num-
bers in Section 5 likely understate domain-level variation; domain-
stratified evaluation is a natural direction for future analysis.
4 Benchmark Design
The meta-analysis workflow has a verifiable ground truth at every
stage: included studies are enumerated in the original paper, eli-
gibility criteria are stated explicitly, and the conclusion direction
is reported rather than inferred. MetaSyn uses this structure to
support an end-to-end task, where a system receives a researchTable 1: MetaSyn Dataset Statistics. The training split sup-
ports model fine-tuning; all reported evaluations use the
held-out test set.
Statistic Full Test
Meta-Analysis Papers
Number of papers 442 88
Avg. included studies (reported) 38.1 28.9
Avg. corpus-matched studies 17.4 19.2
Match rate (matched / reported) 45.7% 66.4%
PI/ECO structure 100% 100%
Search strategies 99.5% 98.9%
Search end date 99.3% 98.9%
Inclusion criteria 96.6% 97.7%
Retrieval Corpus
Total articles 140,585
Positive (ground truth) 8,674 (6.2%)
Hard negatives 131,911 (93.8%)
Avg. abstract length 258 words
Multi-section content 74.5%
Note: “Included studies (reported)” is the count stated in the original paper;
“corpus-matched” is the subset localized in our PubMed-based corpus and used as
retrieval ground truth. The gap reflects the multi-database nature of systematic search
(see text). “Multi-section content” indicates articles with at least one structured
section beyond the abstract recovered from PubMed.
question with PI/ECO criteria and must search, screen, extract,
and synthesize the full meta-analysis, and, separately, a retrieval
task that isolates the initial search phase alone. Running retrieval
on its own does more than match a standard IR measurement: it
supplies the retrieval ceiling against which end-to-end inclusion
recall is read, and it gives a direct check of whether the dataset
carries domain signal learnable by fine-tuning, independent of any
downstream system’s behavior.
4.1 End-to-End Meta-Analysis Generation
A system receives the research question and PI/ECO elements and
must conduct the full meta-analysis pipeline, from query formula-
tion through screening and data extraction to final synthesis.
Input:Research question and structured PI/ECO elements.
Output:A structured report containing (1) a list of included
studies, (2) a 2–3 sentence conclusion, and (3) key insights in bullet-
point form.
Ground truth:The corpus-matched studies, the expert-written
conclusion, the key findings, and the effect direction from the orig-
inal meta-analysis.
Evaluation dimensions.
•Inclusion:Recall, Precision, and F1 of the system’s cited study set
against the ground-truth corpus-matched studies, plus Screening
Accuracy (per-paper binary include/exclude).
•Criteria adherence:Semantic consistency of the system’s stated
inclusion and exclusion criteria against the reference criteria,
capturing whether the system reasons explicitly about eligibility.

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
0 20 40 60 80 100
Number of meta-analyses (out of 442)Oncology
Mental health & neuroscience
Other clinical specialties
Metabolism & nutrition
General biology / multidisciplinary
Digital health & AI in medicine
Social sciences & humanities
Dental & oral health
Cardiovascular
Environmental & planetary health
Reproductive & perinatal73  (16.5%)
63  (14.3%)
49  (11.1%)
46  (10.4%)
42  (9.5%)
41  (9.3%)
40  (9.0%)
32  (7.2%)
21  (4.8%)
20  (4.5%)
15  (3.4%)Clinical specialties (67.6%)
Digital health & AI in medicine
Environmental & planetary health
Social sciences & humanities
General biology / multidisciplinary
Figure 1: Domain distribution of the 442 MetaSyn meta-analyses, grouped by source journal. Clinical specialties (purple)
account for 67.6% of the dataset; the remaining 32.4% spans digital health & AI in medicine, social sciences & humanities,
environmental & planetary health, and general biology / multidisciplinary topics.
•Synthesis:Conclusion direction accuracy (positive/negative/mixed
effect), key insights consistency (LLM-assessed), and report struc-
ture quality (LLM-rated 1–5).
These nine metrics span an objectivity gradient. Set-level met-
rics (Inc.R, Inc.P, Inc.F1, Scr.A) are computed by exact comparison
against ground-truth IDs and carry no evaluator uncertainty. Cri-
teria and synthesis metrics (Inc.C, Exc.C, Dir.A, Insights, SQ) re-
quire an evaluator model to judge semantic similarity or structural
quality, so Section 6 validates them against 8-annotator pairwise
judgments and characterizes the resolution at which each reliably
discriminates systems.
4.2 Retrieval in Isolation
This sub-evaluation targets the initial search phase alone, the entry
point of the full pipeline: any study missed here cannot be recovered
during screening or synthesis, so retrieval quality in isolation is an
upper bound on end-to-end inclusion recall.
Input:Research question (free text) and structured PI/ECO ele-
ments. Search strategies and date ranges are withheld, so that the
system must construct its own queries.
Output:A ranked list of candidate articles from the corpus.
Ground truth:The set of corpus-matched articles for the orig-
inal meta-analysis. Critically, this ground truth is defined by the
original paper’s PI/ECO protocol, not by topical similarity: a re-
trieval system that surfaces on-topic but methodologically ineligible
studies will score low here, even if every retrieved paper looks su-
perficially relevant.
Recall emphasis.In meta-analysis, missing a relevant study
is worse than retrieving irrelevant ones: the latter can be filtered
by screening, but the former introduces uncorrectable selection
bias [30]. Evaluation therefore focuses on Recall@K.To prevent leakage, for each query we exclude articles published
after the meta-analysis’s stated search end date, mirroring the time-
bounded nature of real systematic searches.
We place no constraints on system architecture. Because the
original meta-analysis supplies ground truth at every intermediate
stage, a single run of any system yields separate measurements of
retrieval coverage, screening decisions, criteria articulation, and
synthesis fidelity. A system that reaches the wrong conclusion
can therefore be diagnosed as having retrieved the wrong papers,
selected from the right papers poorly, or synthesized the right
papers incorrectly. Figure 2 summarizes this structure: each pipeline
stage is paired with its own ground-truth object extracted from the
source meta-analysis, and each evaluation metric is attached to the
stage whose decisions it scores.
5 Experiments
5.1 Experimental Setup
Data split:The 88 test queries are held out for evaluation; the
remaining 354 meta-analyses form the training split used for model
fine-tuning and few-shot example selection.
Retrieval infrastructure:BM25 uses BM25Okapi ( 𝑘1=1.5,𝑏=0.75).
Dense retrieval uses BGE-large-en-v1.5 [ 63] indexed with FAISS [ 20].
MA-Retriever(Meta Analysis Retriever) is BGE-large-en-v1.5 fine-
tuned on the MetaSyn training split using MultipleNegativesRank-
ingLoss (in-batch negatives, up to 10,000 pairs, 10 epochs); the best
checkpoint is selected based on Recall@10 on a held-out 10% of the
training split.
LLM infrastructure:End-to-end experiments use three frontier
synthesis backbones.DeepSeek-R1is a reasoning model in the
DeepSeek series, trained to exhibit stronger reasoning behavior
through reinforcement learning [ 15].GLM-5is an agentic and
instruction-following model from the GLM family [ 14].GPT-5is

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
Research Question
+ PI/ECORetrieval
Query→top-𝐾from
140,585-article corpusScreening
Apply PI/ECO criteria
Include / excludeSynthesis
Aggregate effect
direction + insightsStructured
Meta-analysis ReportGT: expert-verified
included study IDsGT: PI/ECO inclusion /
exclusion criteriaGT: effect direction
+ key insightsVerifiable
Ground Truth
Inclusion Recall,
Precision & F1Screening Accuracy,
Criteria ConsistencyDirection Accuracy,
Insights, Structure QualityEvaluation
Metrics
Figure 2: The meta-analysis workflow and MetaSyn’s stage-level supervision. The middle row (Retrieval →Screening→
Synthesis) is the standard meta-analysis pipeline followed by any systematic reviewer, independent of MetaSyn. What MetaSyn
adds, for each stage, is a verifiable ground-truth object extracted from the source paper (top row) and a set of evaluation metrics
tied to that stage (bottom row), so a system’s failures can be read at the stage where they occur rather than averaged across the
end-to-end output.
OpenAI’s general-purpose frontier language model [ 36]. All three
are accessed via their official APIs at default sampling settings with
no meta-analysis-specific system prompts; differences in inclusion
behavior across backbones therefore reflect intrinsic generation
strategies rather than prompt engineering.
5.2 Retrieval in Isolation
5.2.1 Baselines.We compare three retrieval methods:
•BM25: Sparse term-frequency retrieval [ 33] applied to the struc-
tured PI/ECO query. A lexical baseline.
•Dense (BGE): Off-the-shelf BGE-large-en-v1.5 embeddings, no
domain adaptation. Measures what generic semantic matching
achieves on this corpus.
•MA-Retriever: BGE-large-en-v1.5 fine-tuned on the MetaSyn
training split. If MetaSyn carries domain signal beyond generic
pretraining, fine-tuning should produce clear recall gains.
Queries are formatted as structured PI/ECO text (“Research: {Ti-
tle}. Population: {P}. Intervention: {I/E}. Comparison: {C}. Outcome:
{O}”), matching the format used during fine-tuning.
5.2.2 Results.Figure 3 shows three patterns.
Three levels of retrieval signal, stacking.BM25 matches
terms. Off-the-shelf dense retrieval adds generic semantic similarity
and improves R@100 by +12.8 points (78.2% vs. 65.4%; paired 𝑡=6.29,
𝑝<10−9). MA-Retriever adds a further +5.4 points to 83.7% ( 𝑝<10−5)
and raises Recall@200 from 86.8% to 90.9% ( 𝑝<10−5); every pairwise
comparison at 𝐾≥10is significant at 𝑝<0.001. The two gains stack
rather than substitute. Semantic matching recovers papers that
use different terminology for the same intervention; MA-Retriever
then pushes studies satisfying the original paper’s PI/ECO protocol
above topically adjacent studies that fail a specific criterion. The
two signals are non-redundant, which is why the training split adds
recall over what BGE already encodes rather than saturating it.
MA-Retriever consistently improves recall, and helps large
meta-analyses the most.At K=200, MA-Retriever achieves per-
fect recall on 53 of 88 test queries, against 46 for the off-the-shelf
model; at R@100 it improves on 33 queries and regresses on only
4. Bucketing by the number of included studies reveals a sharper
@5 @10 @20 @50 @100 @200
K020406080100Recall@K (%)657884BM25
Dense (BGE)
MA-RetrieverFigure 3: Recall@ 𝐾on the 88 MetaSyn test queries across the
full𝐾grid. BM25 is the sparse baseline; Dense (BGE) adds
generic semantic matching (+12.8 pts at 𝐾=100); MA-Retriever,
fine-tuned on the MetaSyn training split, contributes a fur-
ther +5.5 pts. Both gains stack rather than substitute and
hold consistently across all𝐾values.
pattern: the mean R@100 gain of MA-Retriever over Dense (BGE)
is +4.6 pts for small meta-analyses (fewer than 10 included studies),
+4.8 pts for medium ones (10–20), +6.6 pts for large ones (20–50),
and +7.8 pts for the largest (50+). The gain is therefore largest when
the target study set is topically diverse yet methodologically coher-
ent, which is the regime where a narrow-topic retriever struggles
most.
Where retrieval still fails.Four queries drop below R@100
=0.3even with MA-Retriever, and in all four cases all three meth-
ods fail in parallel. The four cases span distinct failure modes: one
(a network meta-analysis of antidepressants for treatment-resistant
depression) fails because network meta-analyses impose eligibility

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
constraints on study comparators that topical retrieval cannot infer
from a single query; another (a meta-analysis of postmortem neu-
roinflammation in schizophrenia) pulls from a tissue-specific litera-
ture whose vocabulary is absent from most meta-analysis queries; a
third (biomarkers of major depressive disorder, 51 included studies)
spans heterogeneous study designs. The residual failures indicate
a limit of single-query retrieval for meta-analyses whose PI/ECO
protocol cuts across study-design or tissue boundaries.
What this implies for downstream pipelines.At K=200,
MA-Retriever covers 90.9% of ground-truth studies on average:
the residual 9.1% is uncoverable downstream, however capable the
synthesis model. Within this pool, each retrieved set contains on
average 16 true positives among 184 distractors per query, a 1:11
ratio that screening must resolve. Both numbers are ceilings on
what any pipeline using this retriever can possibly achieve.
5.3 End-to-End Pipeline Evaluation
5.3.1 Compared Systems.We evaluate two categories of end-to-
end systems: General-purpose RAG pipelines and ProtoMA (Proto-
col Meta-Analysis). For general RAG pipelines, we test a standard
retrieve-then-generate pipeline [ 11,24] across three reasoning back-
bones (DeepSeek-R1, GLM-5, GPT-5) and three retrieval modes:
•BM25: standard sparse retrieval.
•Dense (raw): off-the-shelf BGE embeddings.
•MA-Retriever: MetaSyn fine-tuned BGE model.
Each configuration retrieves the top-200 candidate articles. Each
article is represented by its title and abstract; all 200 representations
are concatenated into a single context window alongside the PI/ECO
question, and the LLM generates a synthesis report in one forward
pass without additional screening or batching. Included studies are
identified by matching paper titles cited in the generated report
against the corpus; inclusions are implicit in which papers the
synthesis references rather than from an explicit include/exclude
decision.
For ProtoMA (Protocol Meta-Analysis), we use a protocol-faithful
reference implementation to characterize what strict adherence to
the systematic-review workflow produces in MetaSyn’s measure-
ment space. The pipeline uses GPT-5 and executes the protocol
steps in order: it generates structured MeSH-style Boolean queries;
screens the 200 retrieved candidates for eligibility in sequential
batches of 25 articles per LLM call, with each call receiving the
PI/ECO criteria alongside each article’s title and abstract (truncated
to 500 characters per abstract); extracts structured data from each
included article by processing its full available sections (introduc-
tion, methods, results, and conclusions, up to 3,000 characters per
article) in a dedicated per-article extraction call; and produces a
structured report with data-extraction tables. No domain-specific
tuning, few-shot prompting, or augmentation is added. We include
ProtoMA because it instantiates the high-precision end of the recall-
precision tradeoff that all screening-based systems face, and be-
cause its output distribution (detailed in Section 5) reveals how
strict eligibility enforcement interacts with retrieval quality. It is
evaluated with all three retrieval modes for the same reason as the
RAG pipelines, to isolate the effect of retrieval quality from the
effect of the synthesis-side design.5.3.2 Metrics.For each configuration we report nine metrics across
three evaluation dimensions:
•Inclusion quality: Inc.R(Inclusion Recall, fraction of GT stud-
ies cited in the output),Inc.P(Inclusion Precision, fraction of
cited studies that are GT positives), andInc.F1.
•Screening and criteria adherence: Scr.A(Screening Accuracy,
per-article binary include/exclude accuracy over the retrieved
pool),Inc.C(Inclusion Criteria Consistency, semantic similarity
between the system’s stated inclusion criteria and the reference),
andExc.C(Exclusion Criteria Consistency, same for exclusion
criteria). Because the retrieved pool is dominated by hard nega-
tives (on average 184 negatives out of 200 retrieved candidates), a
system that excludes all candidates would achieve approximately
92% Scr.A; the informative signal is therefore thechangein Scr.A
as the retrieved pool composition shifts across retrieval condi-
tions, and the contrast between systems, rather than the absolute
value.
•Synthesis quality: Dir.A(Conclusion Direction Accuracy, whether
the system correctly identifies positive/negative/mixed effect),
Insights(semantic consistency of key insights against the refer-
ence, assessed by LLM), andSQ(Structure Quality, LLM-rated
on a 1–5 scale).
The first four metrics are ID-based and require no evaluator
model: Inc.R, Inc.P, and Inc.F1 are computed by comparing the
predicted included-study set with ground-truth study IDs, while
Scr.A is computed from per-article include/exclude labels over the
retrieved pool. The remaining five metrics rely on an evaluator
model and are validated against 8-annotator human judgments
in Section 6. Among them, Dir.A shows strong agreement with
human raters ( 𝜌=+ 0.82, 91.2% annotator-LLM label agreement).
Inc.C, Exc.C, and Insights support coarse-grained cross-system com-
parisons, while SQ is used only as a diagnostic measure of report
organization. Accordingly, our per-stage claims below rely primar-
ily on the ID-based metrics and Dir.A. The evaluator-dependent
metrics are used for directional cross-regime analysis rather than
fine-grained system ranking.
5.3.3 Results.Table 2 reports the full pipeline results. Figure 4
visualizes the key metrics for one configuration per system family
and makes the cross-system comparison direct. We report paired
𝑡-tests over the 88 test queries for the comparisons we interpret;
sample standard deviations for every reported mean are available
in the supplement.
Overall difficulty.The headline result is that no pipeline re-
covers more than 52.7% of ground-truth included literature, despite
a retrieval ceiling of 90.9% Recall@200. The gap between what is
retrievable and what pipelines actually include is not a benchmark
artifact; it reflects the genuine difficulty of the screening stage. At
𝐾=200retrieval, a system must identify on average 16 true posi-
tives from a pool of 200 candidates in which 184 are PI/ECO-failing
distractors (the 1:11 ratio noted in Section 2.3). That discrimination
task is the central challenge MetaSyn is built to probe. The 12.5%
to 52.7% spread in Inc.R confirms that the benchmark resolves gen-
uine differences in pipeline capability; four distinct patterns emerge
from this spread, each tied to a different pipeline stage.
Retrieval quality propagates to inclusion recall for some
backbones but not all.Retrieval ceilings at R@200 are 77.0%,

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
Table 2: End-to-End Pipeline Evaluation on 88 test queries. All values are percentages (%) except Structure Quality (SQ, scale 1–5).
MA-Retriever = MetaSyn fine-tuned retriever; Inc.R = Inclusion Recall; Inc.P = Inclusion Precision; Scr.A = Screening Accuracy;
Inc.C = Inclusion Criteria Consistency; Exc.C = Exclusion Criteria Consistency; Dir.A = Conclusion Direction Accuracy; Insights
= Insights Consistency; SQ = Structure Quality. Best per-column value in bold.
System Retrieval Inc.R Inc.P Inc.F1 Scr.A Inc.C Exc.C Dir.A Insights SQ
RAG
(DeepSeek-R1)BM25 12.5 18.7 11.5 89.2 13.1 6.3 55.7 37.5 2.40
Dense (raw) 13.2 16.5 10.8 86.4 12.1 4.8 52.3 36.4 2.47
MA-Retriever 15.6 16.1 11.6 84.1 10.0 3.9 56.8 37.8 2.40
RAG
(GLM-5)BM25 44.1 25.1 28.5 83.0 10.6 1.3 59.1 44.6 3.22
Dense (raw) 50.3 26.9 31.6 77.6 9.9 1.7 56.8 42.9 3.12
MA-Retriever52.726.6 31.0 73.9 13.9 3.161.443.1 3.20
RAG
(GPT-5)BM25 42.5 36.1 35.0 87.4 50.5 28.9 50.0 45.84.24
Dense (raw) 31.7 25.9 25.4 84.9 45.1 33.4 51.1 44.6 4.08
MA-Retriever 32.2 25.9 26.1 83.5 48.3 37.6 52.347.74.01
ProtoMA
(GPT-5)BM25 30.4 53.0 34.793.7 51.6 47.420.5 22.1 3.99
Dense (raw) 34.5 54.3 38.2 89.9 49.5 46.7 26.1 22.2 3.98
MA-Retriever 35.655.5 39.888.9 49.2 46.9 31.8 22.1 4.00
0.250.500.751.0Inc.R
Scr.A
Inc.C
Dir.AInsightsSQ
DeepSeek-R1
GLM-5GPT-5
ProtoMA
Figure 4: Per-system performance profiles on the MetaSyn
end-to-end benchmark (best retrieval configuration per sys-
tem; values normalized to [0,1], SQ divided by 5). No system
leads on all four dimensions: the winners on inclusion recall,
screening accuracy, criteria adherence, and synthesis quality
are four different systems.
86.8%, and 90.9% for BM25, Dense (BGE), and MA-Retriever respec-
tively. GLM-5’s inclusion recall tracks the retrieval ceilings closely,
rising from 44.1% (BM25) to 52.7% (MA-Retriever), a gain of 8.6
points significant at 𝑝<0.001. ProtoMA follows the same direction
but within a tighter range: 5.2 points ( 𝑝=0.017), consistent with its
higher-precision screening leaving less room for recall to expand.
BM25 Dense (BGE) MA-Retriever20406080%
Pool R@100 Inc.R Inc.PFigure 5: GPT-5 only: as pool quality rises (BM25 →BGE→
MA-Retriever), both Inc.R (blue) and Inc.P (orange) decline
while Pool R@100 (gray dashed) increases. The joint drop
rules out selective screening and points to pool composition:
denser retrieval surfaces topically proximate but PI/ECO-
failing candidates that GPT-5 includes.
DeepSeek-R1 sits outside this pattern entirely. Its Inc.R moves from
12.5% to 15.6%, within noise at every pairwise comparison, because
a system that cites fewer than five studies per query has almost no
channel through which retrieval improvements can propagate.
GPT-5 is an exception (Figure 5). Its Inc.R is highest under BM25
at 42.5% and falls to 31.7–32.2% under dense retrieval, with Inc.P
dropping in parallel from 36.1% to 25.9%, both significant at 𝑝<0.001.
A selective-screening account would predict precision to rise as
recall falls, so the joint drop points instead to pool composition.
The dense retrievers substitute some lexically obvious positives
with topically proximate studies that fail a specific methodological
criterion, and GPT-5’s inclusion decisions are sensitive to this sub-
stitution in a way the other backbones’ are not. Stage-attributed
metrics make the interaction visible; the aggregate end-to-end score
would average it away.
Screening Accuracy declines mildly across all four systems as
retrieval improves, but decomposing the confusion matrix shows

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
the decline has different origins. For GLM-5 and ProtoMA the ef-
fect is mechanical. Fine-tuned retrieval concentrates positives in
a smaller pool, so true-negative counts drop while true-positive
counts rise, Inc.P holds steady, and the Scr.A shift is an accounting
consequence of the tighter pool. For GPT-5 the Scr.A decline tracks
the inclusion-set expansion described above: the included-set size
triples and the false-positive count quadruples as dense retrieval
injects methodologically near-duplicate candidates. Reading these
two mechanisms apart requires the per-stage confusion counts; the
aggregate Scr.A alone would not distinguish them.
Four systems, four operating profiles.Each system fam-
ily concentrates in a different corner of the performance space.
DeepSeek-R1 produces concise analytical reports citing fewer than
five studies per query; its chain-of-thought mode favors depth over
breadth, and the Exc.C below 7% reflects minimal explicit exclu-
sion reasoning. GLM-5 takes the opposite approach, accepting a
large share of the retrieved pool to reach the highest Inc.R (up to
52.7%), at the cost of accumulating false inclusions and lower Scr.A.
GPT-5 occupies a middle range of Inc.R (31.7–42.5%) with the high-
est structure-quality scores and the strongest criteria consistency
among the RAG configurations. ProtoMA is the precision end of the
spectrum: highest Inc.P, highest Scr.A, and strongest criteria adher-
ence, all following from its explicit protocol-driven screening. No
system ranks first on all four dimensions: the highest-recall system
scores lowest on screening accuracy, the most structured reports sit
mid-range on recall, and the tightest criteria adherence comes with
the most hedged synthesis. The separations are sharpest on exact
metrics and Dir.A (validated in Section 6); the evaluator-dependent
metrics show the same ordering at coarser resolution.
This spread across independent dimensions is itself a finding
about benchmark design. A single aggregate score would fold all
four failure modes into an arbitrary ranking that reflects metric
weighting more than system capability. Stage-attributed measure-
ment makes the pattern legible, and each profile points to a con-
crete improvement target: GLM-5 needs stronger PI/ECO-gated
screening, ProtoMA needs a direction-aware synthesis step, GPT-5
needs pool-composition-robust screening, and DeepSeek-R1 needs
a generation prompt that produces explicit, comprehensive study
enumeration.
A note on Dir.A.ProtoMA’s Dir.A of 20.5–31.8% is substantially
below the RAG range of 50–61%. Two design properties explain the
gap: inclusion recall of 30–35% means the synthesis step draws from
a partial evidence base, and the single-pass prompt applies no cross-
study aggregation, producing hedged outputs such as “the direction
appears to favor 𝑋, but certainty is very low” when the included
set is small or contradictory. The categorical extractor maps these
outputs to Mixed on 80.7% of ProtoMA reports (versus 23–35% for
RAG). A joint (direction, confidence) extraction would capture this
behavior more precisely; under the established categorical metric,
ProtoMA’s Dir.A is best read as a lower bound.
Scope and headroom.The twelve configurations span the
standard RAG pipeline (three backbones ×three retrievers) and
ProtoMA (three retrieval variants), the latter representing what a
protocol-faithful implementation achieves without domain-specific
augmentation. The best Inc.R is 52.7%, from GLM-5 with MA-
Retriever, and every pipeline misses roughly half of the ground-
truth literature. Retrieval alone already reaches 90.9% at 𝐾=200,
GPT-5 GLM-5 DeepSeek-R1 ProtoMA0255075100Share of test papers (%)60%55%
11% 10%20%
17%15%
18%
70%10%
80%18%
Positive Negative Mixed None / unclearFigure 6: Distribution of extracted conclusion direction
across 88 test papers (best retrieval mode per system). Pro-
toMA assigns “Mixed” on 80.7% of papers; DeepSeek-R1 pro-
duces no directional label on 80%.
so the gap between what is retrievable and what pipelines actu-
ally cite is a screening gap, not a retrieval gap. Two directions for
closing it follow from the stage-attributed results: stronger use of
MetaSyn’s labeled inclusion pairs during retriever training beyond
a single fine-tuning pass, and screening that applies PI/ECO eligibil-
ity to semantically diverse pools rather than only lexically aligned
candidates.
6 Metric Validation Against Human Judgments
Four of the nine metrics (Inc.R, Inc.P, Inc.F1, Scr.A) are computed
by direct set comparison against ground-truth corpus IDs and carry
no evaluator uncertainty. The remaining five (Inc.C, Exc.C, Dir.A,
Insights, SQ) depend on an evaluator model’s interpretation of
semantic or structural quality. This section validates these five
metrics against human pairwise judgments and characterizes the
resolution at which each reliably discriminates system outputs.
Study design.We sampled 30 papers from the test set and con-
structed pairwise comparisons across three system pairs: RAG (GPT-
5) under BM25 versus MA-Retriever, ProtoMA under BM25 versus
MA-Retriever, and RAG (GPT-5) versus ProtoMA both under MA-
Retriever. We collected judgments on four dimensions for every pair
(inclusion, exclusion, conclusion direction, and key insights), with
an additional report-structure judgment on one pair, giving 390
pairwise tasks in total. Each task was rated by all 8 annotators on a
five-point scale from strongly A through tie to strongly B, a fully
overlapping design that supports direct computation of pairwise
agreement.
Agreement protocol.All ratings (human and LLM) are col-
lapsed to{𝐴,tie,𝐵} (A1/A2→A; B1/B2→B; LLM label =sign(𝑀 𝐴−
𝑀𝐵), zero→tie). H–H denotes pairwise agreement between two
annotators on the same task; L–H denotes agreement between one
annotator and the LLM. For conclusion direction, a tie is informative
(it means both reports agree, or neither does), so tie-vs-tie counts as
agreement. For all other dimensions, uninformative ties are dropped

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
Dir.A Exc.C Insights Inc.C SQ507090Agreement (%)94
76
5676
5691
68666177H-H (human-human)
L-H (LLM-human)
0 .5 1Spearman ρSQ
Inc.C
Insights
Exc.C
Dir.A+0.19
+0.24
+0.43
+0.59
+0.82
Figure 7: Five evaluator-dependent metrics validated against 8-annotator pairwise judgments (390 pairs). H–H pairwise
agreement (blue) and LLM-vs-annotator agreement (L–H, gray) appear alongside the Spearman correlation between human-
mean ratings and LLM-metric differences. Dir.A achieves the strongest correlation ( 𝜌=+0.82), the only dimension supporting
fine-grained system ranking; Exc.C and Insights track human orderings at coarser resolution; Inc.C and SQ characterize system
profiles at the regime level rather than fine-grained rank.
before computing agreement. Spearman 𝜌is computed over the
continuous 5-point mean ratings and the LLM-metric differences
across all tasks.
Figure 7 presents agreement rates and rank correlations across all
five dimensions. The correlations are computed over 30 comparison
papers per system pair; the tier assignments below are supported by
the consistency of the pattern across all three system pairs rather
than by any single correlation value. Conclusion direction proves
the most reliable signal throughout: annotator pairs agree 93.5% of
the time, the LLM metric matches annotator labels on 91.2% of tasks,
and the rank correlation reaches 𝜌=+ 0.82, confirming that Dir.A
orders system outputs the same way human evaluators do. Inclusion
and exclusion criteria both fall in the 76% range for annotator
pairwise agreement; exclusion is somewhat more consistent than
inclusion on the human-versus-LLM comparison (68.1% versus
61.3%), a gap that carries through to the rank correlations ( 𝜌=+
0.59versus+0.24). Even where individual pairwise preferences
are borderline, the relative ordering that the LLM metric produces
tracks human preferences well enough to support the cross-regime
contrasts reported in Section 5. Key insights and report structure
show positive but lower correlations throughout, reflecting that
annotators and the LLM converge on broad preferences without
consistently agreeing on fine-grained pairwise orderings.
Metric reliability tiers.The evidence places Dir.A in its own
tier: it is the only dimension where both agreement rates and rank
correlation support fine-grained system ranking. Inc.C, Exc.C, and
Insights are reliable for directional contrasts when regimes are
clearly separated, with Exc.C the most stable of the three. SQ is
reported for completeness and serves as a descriptor of broad report
organization rather than a precision ranking instrument.7 Related Work
7.1 Systematic Review Automation
Automating systematic reviews predates the current wave of LLM
agents. Surveys and practical guides identify search formulation,
citation screening, data extraction, and report updating as the main
automation targets [ 27,29,53], and systems such as ASReview
emphasize efficient, transparent screening with humans in the
loop [ 56]. Recent studies examine whether LLMs can assist with
screening and extraction, but consistently find that current models
are unreliable for unsupervised use throughout the full review pro-
cess [ 23,25]. A parallel line applies LLMs to broader scientific work:
The AI Scientist drives end-to-end workflows from ideation to writ-
ing [ 26], ScienceAgentBench evaluates tool use and reasoning in
data-driven discovery [ 3], and AutoSurvey targets coherent survey
generation rather than protocol-bound inclusion decisions [ 61]. Ear-
lier literature-based discovery [ 37] and long-context generation [ 64]
extend this direction. These systems, however, are evaluated on
open-ended text production or scientific problem solving, and do
not test whether a model preserves the inclusion and exclusion logic
of a review protocol. MetaSyn follows this line but uses published
meta-analyses as stage-level supervision for the full pipeline.
7.2 Retrieval-Augmented Generation for
Scientific Evidence Synthesis
Retrieval-augmented generation (RAG) equips LLMs with informa-
tion retrieved from external knowledge sources, providing a non-
parametric interface between generation and large-scale evidence
repositories [ 6,40,54]. Prior work shows that this paradigm can im-
prove factual grounding and reduce hallucinations [45, 49, 57, 59],
support knowledge updates [ 58,60], and adapt LLMs to specialized

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
domains without extensive retraining [ 39,41,43,51]. A standard
implementation follows a retrieval-then-read pipeline: a query is
matched against a large corpus using sparse, probabilistic, or neu-
ral retrievers [ 10,21,34,38], and the retrieved evidence is then
provided as context for generation [ 7,11]. Recent extensions make
this pipeline more adaptive and structured, including iterative and
dynamic retrieval [ 18,46,47], graph-based retrieval [ 8], paramet-
ric RAG [ 41,42,48,52], multi-hop evidence aggregation [ 65], and
agentic search [19, 44].
Scientific literature retrieval is a central setting for RAG be-
cause most AI-assisted research workflows depend on finding, rank-
ing, and synthesizing papers. BM25 remains a strong sparse base-
line [ 33], dense retrieval improves semantic matching over large
corpora [ 21], and embedding models such as BGE serve as general-
purpose retrievers [ 63]. Cross-encoder rerankers further capture
fine-grained relevance among topically similar documents [ 28],
which is especially important in scientific domains where a paper
can be topically close to a review question but methodologically
ineligible. Existing RAG systems for scientific literature primarily
evaluate whether retrieved evidence improves answer quality, sur-
vey generation, or long-form synthesis. Much less attention is paid
to whether the retrieved evidence supports reliable downstream
decisions under explicit eligibility criteria. This distinction is crucial
for meta-analysis: the model must not only retrieve relevant papers,
but also preserve the inclusion and exclusion logic specified by the
review protocol. MetaSyn therefore evaluates RAG-style evidence
use at the levels of criterion-based retrieval, screening, and synthe-
sis, rather than treating scientific RAG as a final text-generation
problem alone.
7.3 Benchmarks for Scientific Writing and
Synthesis
Several benchmarks target scientific writing and synthesis. SurGE [ 50]
scores multi-domain survey generation on document-level narra-
tive quality, DeepScholar-Bench [ 31] evaluates research synthesis
under live retrieval, and TrialReviewBench [ 62] covers screening
and synthesis across clinical-trial review stages with stage-level
annotations. These benchmarks, however, measure a model’s ability
to generate coherent narratives or synthesize topically relevant pa-
pers. Meta-analysis demands a different kind of judgment: eligibility
is governed by PI/ECO criteria (Population, Intervention, Compari-
son, Outcome), and the core difficulty, separating methodologically
compliant studies from those that merely appear relevant, is not
captured by narrative or synthesis benchmarks. In short, existing
benchmarks evaluate topical matching and narrative coherence,
but provide no testbed for the criterion-based, multi-stage retrieval
and exclusion process that defines a meta-analysis.
8 Limitations
The current release has two scope limitations. First, the retrieval
corpus is anchored to PubMed, the largest single indexed source
that systematic reviewers consult. Because meta-analyses also draw
from EMBASE, Cochrane Central, clinical-trial registries, and grey
literature, on average 17.4 of a paper’s 38.1 reported included stud-
ies are recovered as corpus-matched ground truth (Section 3). Allevaluation metrics are therefore computed against this PubMed-
matched subset. The design is conservative: a system that retrieves
additional non-PubMed evidence cannot receive credit under the
current setup, so the reported numbers are lower bounds on what
a multi-source pipeline could achieve. Extending the corpus to ad-
ditional databases is a natural direction for future work. Second,
as documented in Section 5, the categorical Dir.A metric penalizes
reports whose conclusions are phrased with explicit uncertainty, so
ProtoMA’s protocol-driven synthesis is systematically scored lower
on this dimension than less-hedged RAG outputs; we report Dir.A
as-is because it is the established direction metric in comparable
benchmarks, and treat a certainty-aware variant as the target for
future metric work.
9 Conclusion
MetaSyn provides 442 expert-curated meta-analyses from theNa-
turePortfolio, each paired with a 140,585-article retrieval corpus,
structured PI/ECO annotations, and ground-truth study lists. Be-
cause meta-analysis follows a codified protocol, stage-level ground
truth at retrieval, screening, and synthesis is inherited from the
source publications without additional annotation, and the same
data supports both pipeline benchmarking and retriever training.
Benchmarking twelve configurations reveals screening as the
binding constraint: despite a retrieval ceiling of 90.9% at 𝐾=200,
no pipeline recovers more than 52.7% of ground-truth included
studies. The 38-point gap traces to pools averaging 16 positives
among 184 PI/ECO-failing distractors, a ratio current LLMs cannot
reliably resolve. Three directions follow from the stage-attributed
results: retrievers that exploit MetaSyn’s eligibility-labeled pairs
beyond a single fine-tuning pass; screening components that apply
PI/ECO criteria to semantically diverse candidate pools; and syn-
thesis metrics that accommodate certainty-aware direction claims
PRISMA [30] requires.
Acknowledgments
Anonymous for review.
References
[1]Hilda Bastian, Paul Glasziou, and Iain Chalmers. 2010. Seventy-Five Trials and
Eleven Systematic Reviews a Day: How Will We Ever Keep Up?PLoS Medicine7,
9 (Sept 2010), e1000326. doi:10.1371/journal.pmed.1000326
[2]Iain Chalmers and Paul Glasziou. 2009. Avoidable waste in the production
and reporting of research evidence.The Lancet374, 9683 (July 2009), 86–89.
doi:10.1016/s0140-6736(09)60329-9
[3]Ziru Chen, Shijie Chen, Yuting Ning, et al .2025. ScienceAgentBench: Toward
Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery.
arXiv:2410.05080 [cs.CL] https://arxiv.org/abs/2410.05080
[4]2019.Cochrane Handbook for Systematic Reviews of Interventions. Wiley. doi:10.
1002/9781119536604
[5]David M. Cordas dos Santos, Tobias Tix, Roni Shouval, et al .2024. A systematic
review and meta-analysis of nonrelapse mortality after CAR T cell therapy.Nature
Medicine30, 9 (July 2024), 2667–2678. doi:10.1038/s41591-024-03084-6
[6]Qian Dong, Qingyao Ai, Hongning Wang, Yiding Liu, Haitao Li, Weihang Su,
Yiqun Liu, Tat-Seng Chua, and Shaoping Ma. 2025. Decoupling Knowledge and
Context: An Efficient and Effective Retrieval Augmented Generation Framework
via Cross Attention. InProceedings of the ACM on Web Conference 2025. 4386–
4395.
[7]Qian Dong, Qingyao Ai, Hongning Wang, Yiding Liu, Haitao Li, Weihang Su,
Yiqun Liu, Tat-Seng Chua, and Shaoping Ma. 2025. Decoupling Knowledge and
Context: An Efficient and Effective Retrieval Augmented Generation Framework
via Cross Attention. InProceedings of the ACM on Web Conference 2025(Sydney
NSW, Australia)(WWW ’25). Association for Computing Machinery, New York,
NY, USA, 4386–4395. doi:10.1145/3696410.3714608

arXiv Preprint, , Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai∗
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130
(2024).
[9]Julian H. Elliott, Tari Turner, Ornella Clavisi, James Thomas, Julian P. T. Higgins,
Chris Mavergames, and Russell L. Gruen. 2014. Living Systematic Reviews: An
Emerging Opportunity to Narrow the Evidence-Practice Gap.PLoS Medicine11,
2 (Feb 2014), e1001603. doi:10.1371/journal.pmed.1001603
[10] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun
Liu. 2024. Scaling laws for dense retrieval. InProceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
1339–1349.
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[12] Gene V. Glass. 1976. Primary, Secondary, and Meta-Analysis of Research.Educa-
tional Researcher5, 10 (Nov 1976), 3–8. doi:10.3102/0013189X005010003
[13] Team GLM, :, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, et al .2024.
ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All
Tools. arXiv:2406.12793 [cs.CL] https://arxiv.org/abs/2406.12793
[14] GLM-5-Team, Aohan Zeng, Xin Lv, et al .2026. GLM-5: from Vibe Coding to
Agentic Engineering. arXiv:2602.15763 [cs.LG] https://arxiv.org/abs/2602.15763
[15] Daya Guo, Dejian Yang, Haowei Zhang, et al .2025. DeepSeek-R1 incentivizes
reasoning in LLMs through reinforcement learning.Nature645, 8081 (Sept 2025),
633–638. doi:10.1038/s41586-025-09422-z
[16] Jessica Gurevitch, Julia Koricheva, Shinichi Nakagawa, and Gavin Stewart. 2018.
Meta-analysis and the science of research synthesis.Nature555, 7695 (Mar 2018),
175–182. doi:10.1038/nature25753
[17] Lam Thi Mai Huynh, Jie Su, Quanli Wang, Lindsay C. Stringer, Adam D. Switzer,
and Alexandros Gasparatos. 2024. Meta-analysis indicates better climate adapta-
tion and mitigation performance of hybrid engineering-natural coastal defence
measures.Nature Communications15, 1 (Apr 2024), 2871. doi:10.1038/s41467-
024-46970-w
[18] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. 7969–7992.
[19] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason
and leverage search engines with reinforcement learning.arXiv preprint
arXiv:2503.09516(2025).
[20] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-Scale Similarity
Search with GPUs.IEEE Transactions on Big Data7, 3 (2019), 535–547. doi:10.
1109/TBDATA.2019.2921572
[21] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. arXiv:2004.04906 [cs.CL] https://arxiv.org/abs/
2004.04906
[22] Masaki Kato, Hikaru Hori, Takeshi Inoue, Junichi Iga, Masaaki Iwata, Takahiko
Inagaki, Kiyomi Shinohara, Hissei Imai, Atsunobu Murata, Kazuo Mishima, and
Aran Tajika. 2021. Discontinuation of antidepressants after remission with
antidepressant medication in major depressive disorder: a systematic review and
meta-analysis.Molecular Psychiatry26 (2021), 118–133. doi:10.1038/s41380-020-
0843-0
[23] Qusai Khraisha, Sophie Put, Johanna Kappenberg, Azza Warraitch, and Kristin
Hadfield. 2024. Can large language models replace humans in systematic reviews?
Evaluating GPT-4’s efficacy in screening and extracting data from peer-reviewed
and grey literature in multiple languages.Research Synthesis Methods15, 4 (Mar
2024), 616–626. doi:10.1002/jrsm.1715
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al .2021. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] https:
//arxiv.org/abs/2005.11401
[25] Judith-Lisa Lieberum, Markus Toews, Maria-Inti Metzendorf, et al .2025. Large
language models for conducting systematic reviews: on the rise, but not yet
ready for use—a scoping review.Journal of Clinical Epidemiology181 (May 2025),
111746. doi:10.1016/j.jclinepi.2025.111746
[26] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David
Ha. 2024. The AI Scientist: Towards Fully Automated Open-Ended Scientific
Discovery. arXiv:2408.06292 [cs.AI] https://arxiv.org/abs/2408.06292
[27] Iain J. Marshall and Byron C. Wallace. 2019. Toward systematic review automa-
tion: a practical guide to using machine learning tools in research synthesis.
Systematic Reviews8, 1 (July 2019), 93. doi:10.1186/s13643-019-1074-9
[28] Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage Re-ranking with BERT.
arXiv:1901.04085 [cs.IR] https://arxiv.org/abs/1901.04085
[29] Alison O’Mara-Eves, James Thomas, John McNaught, Makoto Miwa, and Sophia
Ananiadou. 2015. Using text mining for study identification in systematic reviews:
a systematic review of current approaches.Systematic Reviews4, 1 (Jan 2015), 5.doi:10.1186/2046-4053-4-5
[30] Matthew J Page, Joanne E McKenzie, Patrick M Bossuyt, et al .2021. The PRISMA
2020 statement: an updated guideline for reporting systematic reviews.BMJ
(Mar 2021), n71. doi:10.1136/bmj.n71
[31] Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Za-
haria, and Carlos Guestrin. 2026. DeepScholar-Bench: A Live Benchmark and Au-
tomated Evaluation for Generative Research Synthesis. arXiv:2508.20033 [cs.CL]
https://arxiv.org/abs/2508.20033
[32] Melissa L. Rethlefsen, Shona Kirtley, Siw Waffenschmidt, et al .2021. PRISMA-S:
an extension to the PRISMA Statement for Reporting Literature Searches in
Systematic Reviews.Systematic Reviews10, 1 (Jan 2021), 39. doi:10.1186/s13643-
020-01542-z
[33] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Found. Trends Inf. Retr.3, 4 (Apr 2009), 333–389.
doi:10.1561/1500000019
[34] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3, 4 (2009), 333–389.
[35] Connie Schardt, Martha B Adams, Thomas Owens, Sheri Keitz, and Paul Fontelo.
2007. Utilization of the PICO framework to improve searching PubMed for
clinical questions.BMC Medical Informatics and Decision Making7, 1 (June 2007),
16. doi:10.1186/1472-6947-7-16
[36] Aaditya Singh, Adam Fry, Adam Perelman, et al .2026. OpenAI GPT-5 System
Card. arXiv:2601.03267 [cs.CL] https://arxiv.org/abs/2601.03267
[37] Neil R. Smalheiser. 2017. Rediscovering Don Swanson: the Past, Present and
Future of Literature-Based Discovery.Journal of Data and Information Science2,
4 (2017), 43–64. doi:10.1515/jdis-2017-0019
[38] Weihang Su, Qingyao Ai, Xiangsheng Li, Jia Chen, Yiqun Liu, Xiaolong Wu, and
Shengluan Hou. 2024. Wikiformer: Pre-training with Structured Information of
Wikipedia for Ad-hoc Retrieval. arXiv:2312.10661 [cs.IR] https://arxiv.org/abs/
2312.10661
[39] Weihang Su, Qingyao Ai, Yueyue Wu, Anzhe Xie, Changyue Wang, Yixiao Ma,
Haitao Li, Zhijing Wu, Yiqun Liu, and Min Zhang. 2025. Pre-training for legal
case retrieval based on inter-case distinctions.ACM Transactions on Information
Systems43, 5 (2025), 1–27.
[40] Weihang Su, Qingyao Ai, Jingtao Zhan, Qian Dong, and Yiqun Liu. 2025. Dynamic
and Parametric Retrieval-Augmented Generation. arXiv:2506.06704 [cs.CL] https:
//arxiv.org/abs/2506.06704
[41] Weihang Su, Xuanyi Chen, Yueyue Wu, Qingyao Ai, and Yiqun Liu. 2026. Enhanc-
ing Judgment Document Generation via Agentic Legal Information Collection
and Rubric-Guided Optimization.arXiv preprint arXiv:2605.02011(2026).
[42] Weihang Su, Qian Dong, Qingyao Ai, and Yiqun Liu. 2025. SIGIR-AP 2025
Tutorial Proposal: Dynamic and Parametric Retrieval-Augmented Generation.
In3rd International ACM SIGIR Conference on Information Retrieval in the Asia
Pacific.
[43] Weihang Su, Yiran Hu, Anzhe Xie, Qingyao Ai, Quezi Bing, Ning Zheng, Yun
Liu, Weixing Shen, and Yiqun Liu. 2024. STARD: A Chinese Statute Retrieval
Dataset Derived from Real-life Queries by Non-professionals. InFindings of the
Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit
Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics,
Miami, Florida, USA, 10658–10671. doi:10.18653/v1/2024.findings-emnlp.625
[44] Weihang Su, Jianming Long, Qingyao Ai, Yichen Tang, Changyue Wang, Yiteng
Tu, and Yiqun Liu. 2026. Skill Retrieval Augmentation for Agentic AI.arXiv
preprint arXiv:2604.24594(2026).
[45] Weihang Su, Jianming Long, Changyue Wang, Shiyu Lin, Jingyan Xu, Ziyi Ye,
Qingyao Ai, and Yiqun Liu. 2025. Towards Unification of Hallucination Detection
and Fact Verification for Large Language Models.arXiv preprint arXiv:2512.02772
(2025).
[46] Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Zhijing Wu, and Yiqun
Liu. 2024. Mitigating entity-level hallucination in large language models. In
Proceedings of the 2024 Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval in the Asia Pacific Region. 23–31.
[47] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic Retrieval Augmented Generation based on the Information Needs of
Large Language Models. arXiv:2403.10081 [cs.CL] https://arxiv.org/abs/2403.
10081
[48] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 1240–1250.
[49] Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou,
and Yiqun Liu. 2024. Unsupervised real-time hallucination detection based on
the internal states of large language models. InFindings of the Association for
Computational Linguistics: ACL 2024. 14379–14391.
[50] Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Xuanyi Chen, Jiaxin Mao,
Ziyi Ye, and Yiqun Liu. 2026. SurGE: A Benchmark and Evaluation Framework
for Scientific Survey Generation. arXiv:2508.15658 [cs.CL] https://arxiv.org/abs/
2508.15658

Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio arXiv Preprint, ,
[51] Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang,
Kaiyuan Zhang, Yueyue Wu, and Yiqun Liu. 2025. JuDGE: Benchmarking Judg-
ment Document Generation for Chinese Legal System. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’25), July 13–18, 2025, Padua, Italy. doi:10.1145/3726302.3730295
[52] Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. 2025. Dynamic
parametric retrieval augmented generation for test-time knowledge enhancement.
arXiv preprint arXiv:2503.23895(2025).
[53] Guy Tsafnat, Paul Glasziou, Miew Keen Choong, Adam Dunn, Filippo Galgani,
and Enrico Coiera. 2014. Systematic review automation technologies.Systematic
Reviews3, 1 (July 2014), 74. doi:10.1186/2046-4053-3-74
[54] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. 2025. Robust
Fine-tuning for Retrieval Augmented Generation against Retrieval Defects. In
Proceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 1272–1282.
[55] Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone. 2024. When com-
binations of humans and AI are useful: A systematic review and meta-analysis.
Nature Human Behaviour8, 12 (Oct 2024), 2293–2303. doi:10.1038/s41562-024-
02024-1
[56] Rens van de Schoot, Jonathan de Bruin, Raoul Schram, et al .2021. An open source
machine learning framework for efficient and transparent systematic reviews.
Nature Machine Intelligence3, 2 (Feb 2021), 125–133. doi:10.1038/s42256-020-
00287-7
[57] Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun Liu. 2026. Joint evalua-
tion of answer and reasoning consistency for hallucination detection in large
reasoning models. InProceedings of the AAAI Conference on Artificial Intelligence,
Vol. 40. 33377–33385.[58] Changyue Wang, Weihang Su, Qingyao Ai, Yichen Tang, and Yiqun Liu. 2025.
Knowledge editing through chain-of-thought. InProceedings of the 2025 Confer-
ence on Empirical Methods in Natural Language Processing. 10684–10704.
[59] Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, and Yiqun Liu. 2025.
Decoupling reasoning and knowledge injection for in-context knowledge editing.
InFindings of the Association for Computational Linguistics: ACL 2025. 24543–
24562.
[60] Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, and Yiqun Liu. 2025. De-
coupling Reasoning and Knowledge Injection for In-Context Knowledge Editing.
arXiv:2506.00536 [cs.CL] https://arxiv.org/abs/2506.00536
[61] Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan
Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue
Zhang. 2024. AutoSurvey: Large Language Models Can Automatically Write
Surveys. arXiv:2406.10252 [cs.IR] https://arxiv.org/abs/2406.10252
[62] Zifeng Wang, Lang Cao, Benjamin Danek, Qiao Jin, Zhiyong Lu, and Jimeng Sun.
2025. Accelerating clinical evidence synthesis with large language models.npj
Digital Medicine8, 1 (Aug 2025), 509. doi:10.1038/s41746-025-01840-7
[63] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and
Jian-Yun Nie. 2024. C-Pack: Packed Resources For General Chinese Embeddings.
arXiv:2309.07597 [cs.CL] https://arxiv.org/abs/2309.07597
[64] Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Long-Context Language
Modeling with Parallel Context Encoding. arXiv:2402.16617 [cs.CL] https://
arxiv.org/abs/2402.16617
[65] Liwen Zheng, Chaozhuo Li, Litian Zhang, Haoran Jia, Senzhang Wang, Zheng
Liu, and Xi Zhang. 2025. MRR-FV: Unlocking Complex Fact Verification with
Multi-Hop Retrieval and Reasoning. InProceedings of the Thirty-Ninth AAAI
Conference on Artificial Intelligence (AAAI ’25). AAAI Press, 26066–26074. doi:10.
1609/aaai.v39i24.34802