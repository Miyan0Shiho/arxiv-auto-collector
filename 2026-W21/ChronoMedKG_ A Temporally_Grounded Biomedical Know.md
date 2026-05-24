# ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning

**Authors**: Md Shamim Ahmed, Farzaneh Firoozbakht, Lukas Galke Poech, Jan Baumbach, Richard Röttger

**Published**: 2026-05-21 17:04:28

**PDF URL**: [https://arxiv.org/pdf/2605.22734v1](https://arxiv.org/pdf/2605.22734v1)

## Abstract
Biomedical knowledge graphs (KGs) treat disease associations as static facts, but temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of one disease at age 3 may imply a different disease at age 13. Existing KGs such as PrimeKG, Hetionet, and iKraph do not encode when a finding becomes clinically relevant over the course of a disease. This limits their usefulness for longitudinal clinical reasoning and retrieval augmentation.
  We introduce ChronoMedKG, a temporal biomedical knowledge graph that contains 460,497 evidence-linked triples (filtered from 13M raw extractions) covering 13,431 diseases. Each association is tied to temporal components like onset window or progression stage, which are backed by PMID-traceable evidence and a multi-signal credibility score. The graph is constructed through a disease-autonomous multi-agent pipeline in which multiple frontier LLMs independently extract knowledge from PubMed and PMC literature. Only those relations are kept that are supported by multi-model consensus, survive credibility filtering, as well as ontology alignment.
  ChronoMedKG scored 92.7% agreement against Orphadata and adds temporal grounding for 6,250 diseases absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-coded rare diseases. We further introduce ChronoTQA, a benchmark of 3,341 questions across eight task types (six temporal plus two static controls), with a 12-question supplementary probe. Frontier LLMs lose roughly 30 points moving from static to temporal questions; ChronoMedKG retrieval rescues 47-65% of their long-tail failures, against 17-29% for HPOA-RAG. As such, ChronoMedKG provides a crucial temporal axis for retrieval-augmented clinical systems that was previously absent.

## Full Text


<!-- PDF content starts -->

ChronoMedKG: A Temporally-Grounded Biomedical
Knowledge Graph and Benchmark for Clinical
Reasoning
Md Shamim Ahmed∗
University of Southern DenmarkFarzaneh Firoozbakht
University of HamburgLukas Galke Poech
University of Southern Denmark
Jan Baumbach
University of HamburgRichard Röttger
University of Southern Denmark
Abstract
Biomedical knowledge graphs (KGs) treat disease associations as static facts, but
temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of
one disease at age 3 may imply a different disease at age 13. Existing KGs such as
PrimeKG, Hetionet, and iKraph do not encodewhena finding becomes clinically
relevant over the course of a disease. This limits their usefulness for longitudinal
clinical reasoning and retrieval augmentation. We introduceChronoMedKG, a
temporal biomedical knowledge graph that contains 460,497 evidence-linked triples
(filtered from 13M raw extractions) covering 13,431 diseases. Each association is
tied to temporal components like onset window or progression stage, which are
backed by PMID-traceable evidence and a multi-signal credibility score. The graph
is constructed through a disease-autonomous multi-agent pipeline in which multiple
frontier LLMs independently extract knowledge from PubMed and PMC literature.
Only those relations are kept that are supported by multi-model consensus, survive
credibility filtering, as well as ontology alignment. ChronoMedKG scored 92.7%
agreement against Orphadata and adds temporal grounding for 6,250 diseases
absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-
coded rare diseases. We further introduceChronoTQA, a benchmark of 3,341
questions across eight task types (six temporal plus two static controls), with a
12-question supplementary probe. Frontier LLMs lose roughly 30 points moving
from static to temporal questions; ChronoMedKG retrieval rescues 47–65% of
their long-tail failures, against 17–29% for HPOA-RAG. As such, ChronoMedKG
provides a crucial temporal axis for retrieval-augmented clinical systems that was
previously absent.
Dataset:https://doi.org/10.5281/zenodo.19697542
Code:https://gitlab.sdu.dk/screen4care/chronomedkg
1 Introduction
Biomedical knowledge graphs (KGs) are widely used in applications such as drug repurposing and
computational phenotyping, and are increasingly explored for supporting AI-driven clinical reasoning
[4,9,23]. For example, PrimeKG [ 4] integrates 20 curated databases into 4.05M edges across
17,080 diseases; iKraph [ 23], Hetionet [ 9], and KARMA [ 15] take alternative integration approaches.
However, these KGs lack temporal dimension of clinical knowledge. For example, associations
such as "DMD is associated with cardiomyopathy" do not convey when manifestations occur or
how evidence evolves over time. Static edges do not capture such temporal progression and do not
distinguish whether the association comes from a 1995 case series or a 2023 cohort study. Yet clinical
decisions depend on this temporal context: when to start screening, which diagnosis to consider at a
given age, and whether intervention is still early enough to matter.
Existing resources provide limited support for temporal clinical information. The most temporally-
aware structured resource, HPOA [ 13], provides coarse onset categories (e.g., “childhood”) for 1,429
∗Corresponding author:shamim@imada.sdu.dk
Preprint.arXiv:2605.22734v1  [cs.CL]  21 May 2026

Figure 1:ChronoMedKG: fromwhatis associated towhenit emerges. (A)Across 17,080 PrimeKG diseases, zero edges
carry temporal annotations.(B)4-agent pipeline ingests ∼2.5M public documents and produces literature-grounded temporal
triples with PMID provenance, multi-LLM consensus, and six-signal credibility scoring.(C)Downstream outputs unique to
ChronoMedKG.(D)Top-line numbers. Constructedde novofrom public literature; no patient-level records.
diseases, while Orphadata covers onset for 5,796 rare diseases, both at the disease level rather than
phenotype level. In addition, Phenopackets [ 10] provides patient-level case data for 518 diseases.
Even when combined, these sources cover at most 7,743 diseases out of PrimeKG’s 17,080 and do not
provide per-phenotype temporal annotations at scale. No biomedical QA benchmark (e.g., MedQA,
PubMedQA, BioASQ, MMLU-Medical [ 11,12,21,8]) evaluates temporal clinical reasoning either.
The temporal dimension of disease progression is therefore both unstructured and untested.
Contributions.
1.ChronoMedKG: a temporally-grounded biomedical KG of 460,497 validated consensus triples
derived from 13,431 diseases, built by a disease-autonomous four-agent pipeline with multi-LLM
consensus (Section 3). Each triple includes onset ages, progression stages, clinical milestones,
and six-signal evidence grading with PMID provenance. We validate against Orphadata at92.7%
effective accuracy on the 2,563-disease overlap (Section 4, Table 4) and via an independent
three-LLM judge-panel audit at87.9%verified accuracy on a 100-disease novel-coverage sample
(Section 4.1, Table 3); neither figure is a claim about the full 13,431-disease resource, and clinician
validation at scale across the 6,250 novel-coverage diseases is the remaining step (Limitation (ii),
Section 7).
2.ChronoTQA(Section 5). To the best of our knowledge, ChronoTQA is the first temporal
biomedical QA benchmark, with 3,329 questions across eight reported task types plus a 12-
question supplementary diagnostic set; Tier 1 questions are grounded in Orphadata, HPOA, and
GA4GH Phenopackets, and Tier 2 questions trace to PMID-verified ChronoMedKG triples.
3.Three claims that static KGs cannot support(Section 6). A coverage-gap analysis shows 6,250
diseases with onset data absent from every curated resource, of which 1,657 are Orphanet-coded
rare diseases. Retrieval against ChronoMedKG rescues 47–65% of long-tail onset queries that
three frontier LLMs answer incorrectly without retrieval. A frontier-LLM baseline records a
+30 pp gap between static- and temporal-question accuracy.
Scope of claims.ChronoMedKG is built from peer-reviewed literature, not patient records, and any
clinical use requires clinician oversight and regulatory evaluation beyond the scope of this work. The
validation harness, judge-panel code, and error-taxonomy classifier are released so the community
can audit and extend these numbers directly.
2

Table 1: ChronoMedKG vs. existing biomedical KGs. Only ChronoMedKG provides per-phenotype temporal metadata,
edge-level PMID + verbatim quotes, and a dedicated temporal QA benchmark.*MedKGent disease count estimated from a
156K total-entity release without node-type breakdown.
Resource Diseases Edges PMID per Evidence Temporal Temporal
edge quote profiles QA benchmark
iKraph∼48K 30.8M Partial No No No
PrimeKG 17,080 4.05M No No No No
MedKGent*∼14K 2.97M Partial No Pub-date only No
Hetionet∼4,500 2.25M No No No No
HPOA∼8,600 283K Yes (ref col) No Coarse No
KARMA∼200∼150K Partial No No No
ChronoMedKG 13,431 460K 100% 99.9% Per-phenotype ChronoTQA (3,341 Qs)
2 Related Work
Biomedical knowledge graphs.Table 1 summarizes existing biomedical KGs. PrimeKG [ 4] inte-
grates 20 curated databases into a 4.05M-edge graph across 17,080 diseases. iKraph [ 23] extracts
over 30M relations from PubMed and reports strong performance in drug repurposing; however,
it provides only document-level timestamps (publication dates) rather than clinically meaningful
temporal information. Hetionet [ 9] covers 136 diseases with 2.25M edges. KARMA [ 15] (NeurIPS
2025 Spotlight) introduces a multi-agent collaborative pipeline to extract 150K triples from 1,200
articles with multi-agent conflict resolution; it validates at 83.1% LLM-verified accuracy but does
not incorporate temporal metadata or provide a downstream temporal evaluation.None of these
resources captureswhenclinical associations occur during disease progression.
Concurrent agentic-extraction work.Two recent systems extend biomedical KG construction along
axes orthogonal to ChronoMedKG. MedKGent [ 22] (Aug 2025) processes 10M PubMed abstracts as
a 1975–2023 daily time series, producing a 2.97M-triple medical KG with self-consistency confidence
scoring. Its temporal axis is thepublication dateof source abstracts: when knowledge appeared in
the literature. ChronoMedKG instead grounds facts in patient age and disease stage. AutoBioKG
[24] (bioRxiv Jan 2026) takes a different angle: each edge carries composite contextual conditions,
so a relation can flip under different physiological states (Ca2+-gated activation/inhibition is the
running example). ChronoMedKG addresses an analogous expressivity gap by encoding context as
structured ontology fields ( onset_age_min /onset_age_max , namedprogression_stage , named
milestone ), not free-text strings. Publication-time, contextual-state, and clinical-time are three
independent axes that could be unified in future work.
Temporal knowledge graphs.Prior work on temporal KGs [ 20,6] models time-stamped facts (e.g.,
“Company X acquired Company Y in 2019”) where the temporal dimension is the calendar date of an
event. Clinical temporal grounding differs in two ways: the temporal dimension ispatient ageor
disease stageat which a phenotype manifests, and the underlying source is population-level literature
rather than per-event records. T-Phenotype [ 16] discovers temporal phenotypes from patient-level
EHR data using deep clustering, which is complementary to our literature-derived approach. To our
knowledge, no prior work has built a population-level temporally-grounded biomedical KG from
literature at scale.
Biomedical QA and evaluation benchmarks.MedQA [ 11], PubMedQA [ 12], BioASQ [ 21], and
MMLU-Medical [ 8] collectively contain over 480K questions assessing clinical knowledge, diagnostic
reasoning, and literature comprehension. The rare-disease-focused RareBench [ 5] adds differential-
diagnosis coverage across four dimensions (phenotype extraction, screening, differential diagnosis,
single-shot diagnosis). None of these benchmarks evaluatestemporalclinical reasoning. For example,
questions such as “at what age does phenotype X emerge in disease Y?” or “which disease has
an earlier onset?” are generally not represented in these benchmarks. GA4GH Phenopackets [ 10]
provides 6,668 patient-level cases with phenotype onset ages, which can serve as ground truth
but is not structured as a QA benchmark. ChronoTQA addresses this gap with 3,329 questions
across eight reported task types (plus a 12-question supplementary diagnostic set), including temporal
window identification, temporal differential diagnosis, cross-disease onset comparison, and phenotype
ordering. These tasks require temporal knowledge that is not systematically captured by existing
KGs.
3

3 ChronoMedKG Construction
ChronoMedKG is built end-to-end from a disease identifier through four cooperating agents, with
no per-disease code, manual curation, or prompt engineering. Two design choices distinguish the
pipeline: temporal information is collected at extraction time (not as a post-hoc enrichment), and
every triple carries its evidence (PMID, verbatim quote, six-signal credibility score) into the released
file.
3.1 Four-Agent Autonomous Pipeline
ChronoMedKG is built using a multi-stage pipeline (Figure 1) that processes each disease indepen-
dently. Given a disease identifier (e.g., MONDO, OMIM, ORPHA), four agents execute sequentially.
(1)Disease Profilerqueries biomedical ontologies (MONDO, OMIM, HPO) to generate disease-
specific metadata, including synonyms, differential diagnoses, associated genes, PubMed counts,
and an estimate of literature coverage (Standard ≥100 articles, Light 20–99, Minimal <20). (2)Ev-
idence Harvesterretrieves abstracts and full-text articles from PubMed and PMC using NCBI
E-utilities [ 17]; Standard-tier diseases get up to 150 documents, light/minimal tiers get all available
literature. (3)Knowledge Extractorprocesses documents in parallel with 2–3 LLMs from different
families (DeepSeek V3, GPT-4o-mini or GPT-4.1-nano, Claude 3 Haiku as conditional tiebreaker).
Candidate triples are retained only when independently extracted by at least ≥2 models from the
same document with entity fuzzy matching ≥80% and same canonical relation. From 13,045,687
raw extractions, 460,497 pass consensus (3.53%; retention-rate decomposition in Appendix C.3).
(4)Quality Controlleraligns extracted entities to the PrimeKG schema, applies temporal plausibility
constraints (e.g., age within ∈[0,120] ), detects conflicts, and computes six-signal credibility scores
(journal tier, citation velocity, study type, replication, retraction, and LLM consensus). Full per-agent
prompts, model parameters, and tier definitions are documented in Appendix G.
3.2 Schema and Full Evidence Grounding
Each ChronoMedKG edge carries three classes of metadata.Temporal metadatarecords when
phenotypes manifest, in one of three forms: onset age range (present on 24.5% of validated triples),
progression stage (16.4%), or clinical milestone (45.5%); a single edge may carry more than one
form.Evidence provenancerecords PMIDs, the verbatim source quote ( ≤300 chars), publication
year, study type, and a six-signal credibility score.Consensus confidencerecords the fraction of
extracting models that agreed (1.00 = all, 0.67 = 2/3).
Full evidence grounding at scale.In contrast to prior resources, such as PrimeKG (zero edge-level
PMIDs) or iKraph (partial document traces, no quotes), ChronoMedKG providesboththe raw pool
and the validated output: Among the raw extractions, 90.6% of 13M raw triples are associated with
a PMID and 99.9% include a supporting verbatim quote; for the validated subset, all triples are
linked to a PMID and 99.9% include a corresponding quote. Multi-source triples (1.6%, cross-paper
validated) have up to 81 supporting PMIDs, enabling direct verification of the underlying evidence.
A complete example record (DMD →dilated cardiomyopathy, onset 10–18y, non-ambulatory stage)
and a visualisation of DMD’s full staged temporal profile are in Appendix C.1.
3.3 Resource Statistics
Evidence age.Among 455,519 validated triples, 98.9% carry PMID-traceable publication dates, with
a median year of 2015; of these, 27.4% are supported by evidence from the past five years, while
24.1% rely on studies >20 years ago. In contrast, static KGs do not provide edge-level publication
dates. The full distribution and evidence-decay analysis are presented in Appendix H.8.
4 Validation
We validate the resource in three steps that match the structure of the contribution. Step one re-verifies
the novel-coverage subset (the diseases where no external gold standard exists) using a three-LLM
judge panel against the underlying evidence text. Step two cross-checks against three external onset
databases (Orphadata, HPOA, GeneReviews) on diseases that overlap with each. Step three reports
the HEG-TKG clinician anchor on the underlying extraction methodology.
4

Table 2: ChronoMedKG resource (code-verified). Both raw and validated tiers are evidence-grounded with PMID + verbatim
quote.
Metric Value Notes
Scale
Diseases processed 13,431 78.6% of 17,080 PrimeKG diseases
Diseases with validated output 10,852 Validated consensus triples retained
Documents processed∼2.5M 43K full-text PMC + 2.4M abstracts
Raw extraction pool (pre-consensus) - evidence-grounded at scale
Raw triples extracted 13,045,687 From multi-LLM extraction
Raw triples with PMID 11,819,56190.6%of all raw triples
Raw triples with evidence quote 13,035,19599.9%of all raw triples
Fully grounded raw triples (PMID + quote) 11,809,56590.5%of all raw triples
Consensus filtering (denominator: 10,852 diseases with validated output)
Validated consensus triples 460,497 3.53% of raw triples
3-model consensus (diseases) 6,551 60.4% of 10,852
2-model consensus (diseases) 4,246 39.1% of 10,852
Other configurations (4 models or 1+unlabelled) 55 0.5% of 10,852
Full agreement (triples) 87.1% All models that processed doc agree
Partial agreement (triples) 12.9% 2/3 models agree
Validated triples - temporal metadata and full grounding
Triples with onset ages 112,932 24.5%
Triples with progression stages 75,723 16.4%
Triples with milestones 209,485 45.5%
Diseases with temporal data 8,935 82.3% of validated output
Triples with PMID citation 460,497100%(full provenance)
Triples with evidence text 459,98899.9%(verbatim quote≤300 chars)
Multi-source triples (≥2 PMIDs) 7,563 1.6% (cross-paper validation)
4.1 Novel-Coverage Verification via Multi-LLM Judge Panel
For 6,250 diseases ChronoMedKG provides onset ranges absent from any curated gold standard
(Section 6.1). With no external reference available, we audit the evidence-to-claim fidelity of a
stratified sample: do the verbatim quotes attached to each triple actually support the claimed numeric
onset range? This is a text-grounding check, not an independent re-validation of the underlying
biology, and we report it as such.
We sampled n=100 diseases (seed =42), stratified by literature tier (Standard / Light / Minimal,
defined in Section 3) and onset bucket (clinical era: prenatal, neonatal, infantile, childhood, adolescent,
adult), selecting one (claim, evidence) pair per disease. Three judges (DeepSeek V3, GPT-4o-mini,
Claude Haiku 4.5, temperature 0.0) independently rated each pair under a chain-of-thought prompt
that required verbatim-quote →numeric-range translation via a fixed clinical-era lookup and returned
SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED/UNVERIFIABLE. Full sampling, prompt,
and error-taxonomy detail in Appendix H.5.
Table 3: Three-LLM judge panel on n=100 novel-coverage diseases.Verifiable majorityexcludesUNVERIFIABLEand
three-way splits.
Verdict DeepSeek V3 GPT-4o-mini Claude H-4.5 Majority
SUPPORTED80 70 75 76
PARTIALLY_SUPPORTED2 11 10 4
NOT_SUPPORTED14 7 12 11
UNVERIFIABLE4 12 3 3
Three-way split— — — 6
Inter-judge agreement: 66/100 unanimous, 28/100 two-of-three, 6/100 three-way split.
Verified accuracy((SUPPORTED+PARTIALLY)/verifiable majority):80/91 =87.9%.
All 11 majority-NOT_SUPPORTEDcases share one failure mode: the extraction preserved
a qualitative qualifier (e.g., “elderly onset”, “mid-trimester”) but did not populate numeric
onset_age_{min,max} ; a qualifier-to-range fallback would reclassify 10 of 11 (genuine noise
rate 1/100 = 1%). The 87.9% figure applies to the novel-coverage subset only and is not a re-estimate
of the 92.7% Orphadata accuracy below.
4.2 Cross-Gold-Standard Accuracy
Where ChronoMedKG diseases overlap with external onset databases, we run a direct comparison:
for each matched disease, we compare ChronoMedKG’s median per-phenotype onset range against
5

the gold standard’s disease-level range, and a ChronoMedKG range iscontainedif it falls within the
gold standard. We test this on three sources (Orphadata, HPOA, and GeneReviews) with results in
Table 4. The novel-coverage audit above complements this by handling diseases without any external
ground truth.
Table 4: Validation against external gold standards. Effective accuracy = strict + clinically defensible (§4.3).
Gold Standard Matched Diseases Strict Precision Effective Accuracy
Orphadata 2,563 50.1%92.7%
HPOA 365 18.6%92.6%
GeneReviews 116 56.9%86.2%
4.3 Strict Error Taxonomy: Granularity vs. Genuine Errors
The gap between strict precision and effective accuracy reflects granularity mismatches rather than
errors. HPOA stores onset as broad categorical bins (infantile, juvenile, adult), while ChronoMedKG
reports per-phenotype age ranges in years. Strict containment between these formats fails by con-
struction in most cases, even when both sources agree on the clinical timeline; the 18.6% strict /
92.6% effective gap on HPOA reflects this mismatch, not extraction error. Strict precision across the
three gold standards lines up with gold-standard resolution: HPOA (18.6%) <Orphadata (50.1%)
<GeneReviews (56.9%). We manually classified all 2,563 ChronoMedKG–Orphadata disease
matches by the relationship between the ChronoMedKG onset range and the Orphadata reference
range (Table 5), and find that most discrepancies trace to ChronoMedKG’s finer granularity, namely
per-phenotype onset ranges where gold standards report a single disease-level bin, rather than to
extraction errors. Section 6.1 quantifies the granularity gap across the full resource; a stacked-bar
visualisation of the error categories is in Appendix H.4.
Table 5: Error taxonomy (Orphadata, code-verified). Only 7.3% are genuine errors.
Category Description % Verdict
Contained (correct) Range within gold standard 50.1% Correct
Adjacent stage Differs from gold by one clinical era 15.6% Not an error
Granularity mismatch Per-phenotype vs gold disease-level bin 13.8% Not an error
Wider but overlaps Captures early+late phenotypes; core overlaps 6.7% Not an error
Single-triple noise Outlier triple pulls range; median correct 5.7% Not an error
Genuinely wrong No overlap,>10y gap 7.3% Real error
4.4 Clinician Anchor (via HEG-TKG)
On three diagnostic pairs of rare neuromuscular diseases (DMD/BMD, MG/LEMS, CIDP/GBS),
correct diagnosis hinges on temporal cues such as age of onset, progression rate, or symptom duration.
A three-rater clinician panel evaluated our system against vanilla GPT-4.1. The senior neurologist (C1,
35 blinded cases) rated ChronoMedKG markedly more verifiable ( ∆=+1.64/5 ,d=1.79 ,p<0.001 ,
BH-corrected); a second senior neurologist replicated this on the 11-case CIDP/GBS subset with the
largest effect in the panel ( d=2.57 ,p<0.003 ); a trainee returned a smaller but still BH-significant
advantage ( d=0.67 ,q=0.006 ), noting that our system cites consensus knowledge specialists already
know but non-specialists need.
This evaluation covers a small subset of diseases under a simplified extraction protocol (titles and
abstracts only); clinician review at ChronoMedKG scale across the 6,250 novel-coverage diseases is
the remaining step (Limitation (ii), Section 7). Full Likert breakdowns by rater and dimension in
Appendix H.6.
5 ChronoTQA Benchmark
On the basis of ChronoMedKG, we introduceChronoTQA, the first temporal biomedical QA
benchmark. The benchmark contains 3,341 questions: 3,329 across eight reported task types
plus a 12-question HPOA-grounded negative-temporal MCQ probe held back as a supplementary
diagnostic set (because n=12 is too small for per-type leaderboard reporting). ChronoTQA tests
whenphenotypes emerge in the course of a disease; it complements rare-disease differential-diagnosis
benchmarks like RareBench [ 5] that testwhichdisease a phenotype set indicates. None of the existing
biomedical QA benchmarks discussed in §2 (MedQA, PubMedQA, BioASQ, MMLU-Medical,
RareBench) evaluates age- or stage-conditioned reasoning; a side-by-side landscape comparison is in
supplementary Appendix D.1. Table 6 reports the ChronoTQA composition.
6

ChronoTQA grows with its sources.Both tiers expand with their underlying data. Tier 1 questions
regenerate when Orphadata, HPOA, or Phenopackets release new versions, and the generation scripts
ship with the benchmark. Tier 2 items trace to versioned ChronoMedKG triples and grow with each
release, with full PMID provenance preserved. The benchmark scales with the underlying clinical
literature; v1.0 is the first release in a versioned series, not a fixed snapshot.
Table 6: ChronoTQA composition. Tier 1 grounds in external databases; Tier 2 traces to PMID-verified ChronoMedKG triples;
static controls sanity-check basic biomedical recall. The 12-question HPOA negative-temporal MCQ probe is included in the
release as a supplementary diagnostic set; n=12 is too small to support per-type leaderboard claims, so we exclude it from the
per-type reporting.
Tier Type N Probe Gold Source
Tier 1 (external)Temporal window 800 Phenotype-onset age Orphadata
Temporal differential Dx 687 Disease from onset profile Orphadata
Cross-disease comparison 600 Earlier-onset disease Orphadata
Phenopackets onset 147 Precise age from case data Phenopackets
Tier 2 (KG-derived)Phenotype ordering 395 DoesAprecedeBinD? ChronoMedKG
Stage-conditional 200 Stage-specific features ChronoMedKG
Static controlsDrug–disease indication 250 Indication (control) PrimeKG
Gene–disease association 250 Association (control) PrimeKG
Subtotal (8 task types) 3,329
Supplementary HPOA negative-temporal MCQ probe 12 Rule-out by age HPOA
Total release 3,341
Quality assurance.Four rounds of multi-LLM independent evaluation (540 assessments across
180 questions, 3 evaluator LLMs per question) verified well-formedness and answer correctness.
Automated QC removed 30 logically-inconsistent questions. All 1,447 external-source answers
verified against source data with 0 mismatches.
Frontier-LLM baseline: where the difficulty concentrates.We drew a stratified 120-question
subsample from the 3,329 reported items and posed it to four frontier LLMs via web chat (no API, no
system prompts, no retrieval): GPT-4o-mini, Gemini, Claude, and DeepSeek V3. Per-model answer
logs ship with the benchmark release (Appendix H.2). Table 7 shows the per-task breakdown.
The aggregate gap is +30.1 pp (static minus temporal, mean across models); the breakdown shows
where it concentrates. Binary cross-disease comparison and the static controls saturate near ceiling.
Bounded MCQ subtasks (temporal window, temporal differential diagnosis) sit in the 36–83% range
and split unevenly across models. The free-text Phenopackets onset subtask, which asks for a precise
age range without multiple-choice scaffolding, collapses to a 4.5% mean across models (per-model
rates 0–12%). Frontier LLMs cannot produce calibrated age windows from parametric memory.
Section 6.2 reports retrieval against ChronoMedKG that recovers 47–65% of the questions each
model fails without retrieval on the Phenopackets-grounded onset subset; PrimeKG-RAG recovers
40–51%, HPOA-RAG 17–29%. Full per-model breakdowns are in Appendix H.2.
Table 7: Per-task ChronoTQA accuracy on a 120-question stratified subsample (4 frontier LLMs, web chat, no retrieval).
Free-text precise-age collapses near zero; static controls and binary comparisons saturate. Tier 2 types reported in §6.2 and
Appendix H.2.
Task GPT-4o-mini Gemini Claude DeepSeek Mean
Cross-disease comparison 100% 100% 100% 91% 98%
Temporal window 58% 83% 75% 67% 71%
Temporal differential Dx 82% 75% 55% 36% 62%
Phenopackets onset (free-text) 0% 12% 0% 6% 4.5%
Static drug (control) 80% 100% 80% 80% 85%
Static gene (control) 75% 100% 75% 75% 81%
Per-model temporal mean 52.9% 64.0% 51.0% 45.1% -
Per-model static mean 77.8% 100.0% 77.8% 77.8% -
Gap (static−temporal, pp)+24.8+36.0+26.8+32.7 +30.1
6 Novel Analyses Enabled by Temporal Grounding
Two analyses use the temporal annotations directly: a coverage-gap study and a retrieval experiment
on long-tail questions. Link prediction, evidence decay, trajectory clustering, and bin-granularity
sensitivity are in Supplementary Appendices H.7, H.8, I.1, and I.4.
7

6.1 Coverage Gap: What Temporal Data Exists Where
We evaluate coverage by testing whether onset age information is available in existing structured
biomedical resources for each of the 17,080 diseases in PrimeKG. Table 8 shows the result. PrimeKG
does not contain temporal metadata and therefore provides no onset information. HPOA covers 8.4%
of diseases, Orphadata 33.9%, and Phenopackets 3.0%. In comparison, ChronoMedKG provides
onset information for 52.3% of diseases.6,250 diseases(36.6% of all PrimeKG diseases) have onset
data that is not present in any reference resource.
Table 8: Onset-age coverage vs. PrimeKG’s 17,080 diseases. Granularity = per-phenotype or disease-level.
Resource Diseases with onset Coverage Granularity
PrimeKG 0 0.0% None
HPOA 1,429 8.4% Coarse bins (disease-level)
Phenopackets 518 3.0% Per-patient (case data)
Orphadata 5,796 33.9% Coarse bins (disease-level)
ChronoMedKG 8,935 52.3% Per-phenotype, numeric
ChronoMedKG novel coverage 6,250 36.6%Not in any gold standard
Beyond coverage, ChronoMedKG provides substantially finer granularity. Orphadata reports a
single range per disease (e.g., “DMD: 1–5 years”); ChronoMedKG provides a median of 5 distinct
phenotype-onset pairs per disease, with 6,480 diseases having ≥3 such pairs. In DMD, for example,
ChronoMedKG separately records walking delay (2–5y), Gowers sign (5–8y), loss of ambulation
(8–12y), and cardiomyopathy (10–18y). This information is absent from every existing structured
resource. Figure 2 visualizes this gap. Among the 6,250 novel-coverage diseases, 1,657 are Orphanet-
coded rare diseases gaining first-time temporal grounding; 4,593 are PrimeKG diseases unmatched
by any curated onset resource.
Figure 2:Coverage and granularity of disease onset data. (A)ChronoMedKG covers 8,935 of 17,080 PrimeKG diseases
(52.3%);6,250have onset data absent from every curated gold standard ( +18.4 pp over Orphadata’s 33.9%).(B)DMD
granularity contrast: Orphadata gives a single 1–5y range; ChronoMedKG records five per-phenotype onsets (walking delay
2–5y, Gowers sign 5–8y, loss of ambulation 8–12y, cardiomyopathy 10–18y, respiratory insufficiency 12–20y). Across the
resource: median 5 phenotype-onset pairs per disease (n=6,480with≥3),∼5×more detail than any gold standard.
6.2 Retrieval-Augmented Generation: Rescue on Long-Tail Questions
We evaluated whether retrieval from ChronoMedKG improves LLM performance on 147 externally-
grounded GA4GH Phenopackets questions drawn from the ChronoTQA benchmark (Tier 1). These
147 questions cover the 518 diseases that Phenopackets describes, where ground truth exists. They are
not the 6,250 novel-coverage diseases of §6.1; in that pool, no external resource has data, so there is
no way to score one retrieval source against another. The rescue experiment therefore runs on shared
8

territory, where every retrieval source has a fair shot. Three LLMs (Claude 3 Haiku, DeepSeek V3,
GPT-4o-mini) answered each question under four conditions: no retrieval (NR), PrimeKG-RAG,
HPOA-RAG, and ChronoMedKG-RAG, scored with a calibrated rubric. The Phenopackets corpus
has been public since 2019, so LLMs reach 76–86% accuracy from parametric memory alone (NR).
This is the ceiling set by what these models already know, and all four retrieval sources operate
against it.
A better test under this ceiling islong-tail rescue: on the subset where each model fails without
retrieval, how often does each retrieval source recover the correct answer? Table 9 reports the
result, with 95% bootstrap confidence intervals from 10,000 resamples (seed=42). ChronoMedKG
rescues47–65%of NR-failed questions across the three models. The improvement over HPOA-RAG
(17–29%) is robust under bootstrap resampling for two of three models (Claude, DeepSeek), where
the CIs do not overlap. The improvement over PrimeKG-RAG (40–51%) holds at the point estimate;
CIs overlap given the n=20–35 NR-fail subset per model. ChronoMedKG is the best rescue source at
the point estimate for all three models tested.
Pooled accuracy is bounded by the parametric ceiling: ChronoMedKG-RAG reaches 82.1% vs.
79.8% NR ( +2.3 pp), with Claude 3 Haiku the only per-model significance (McNemar p=0.008 ,
+12.9 pp). Full leaderboard and per-model significance are in Appendix H.1. Retrieval is one strategy
for KG-grounded LLM use; runtime triple verification (KGARevion [ 18]) is an orthogonal one,
enabled by grounded temporal KGs like ChronoMedKG.
Table 9: Long-tail rescue on 147 Phenopackets-grounded onset questions (ChronoTQA Tier 1). Fraction of NR-failed questions
each retrieval source recovers per model. Brackets are 95% bootstrap CIs over 10,000 resamples (seed=42). ChronoMedKG is
the best rescue source at the point estimate for all three models; the gain over HPOA-RAG is statistically robust for Claude
and DeepSeek.
Model NR-failnPrimeKG rescues HPOA rescues ChronoMedKG rescues
Claude 3 Haiku 35 51.4% [34, 69] 17.1% [6, 31]60.0% [43, 77]
DeepSeek V3 20 40.0% [20, 60] 20.0% [5, 40]65.0% [45, 85]
GPT-4o-mini 34 41.2% [26, 59] 29.4% [15, 44]47.1% [29, 65]
Clinical-case illustration.On 31 open-access PMC diagnostic-odyssey case reports (delays 1–
50 y, median 14, 11 disease categories), we matched observed phenotype sequences against
ChronoMedKG’s staged profiles. In PMC5688900, 26 years of schizophrenia misdiagnosis preceded
a Wilson-disease diagnosis; ChronoMedKG records neuropsychiatric onset at 20–40 y alongside
hepatic involvement at 15–45 y, the timing pattern that separates Wilson from primary psychiatric
illness. Full 31-case table, timeline, and per-case temporal-discriminator analysis are in Appendix F.
7 Discussion
ChronoMedKG complements three concurrent agentic biomedical KG systems along an axis the
others do not touch. KARMA [ 15] resolves conflicts between multi-agent extractors. MedKGent [ 22]
tracks how knowledge enters the literature over publication time. AutoBioKG [ 24] encodes contextual
state. None of these encodes clinical time, the patient-age and disease-stage axis on which clinical
decisions actually get made. ChronoMedKG fills that gap at PrimeKG scale, with edge-level onset
ages, progression stages, and disease-course milestones grounded to PMIDs.
Two findings carry the resource, and the populations behind them are different. First, 6,250 diseases
gain temporal grounding that no other resource provides (§6.1); 1,657 of those are Orphanet-coded
rare diseases getting structured onset for the first time. Second, on long-tail onset questions where
Phenopackets supplies the ground truth, ChronoMedKG retrieval rescues 47–65% of the queries each
frontier LLM answers incorrectly without retrieval, against 40–51% for static PrimeKG and 17–29%
for HPOA (§6.2). These are not the same diseases. The first finding is that ChronoMedKG has data
that no other source does; the second is that ChronoMedKG is more precise in the territory all four
sources share.
Limitations.Six gaps that ChronoMedKG has, in descending order of how much they constrain
near-term use. (i)Entity canonicalisation: LLM-extracted phenotype names give ChronoMedKG a
3×larger entity space than HPOA, suppressing link-prediction MRR; SapBERT [ 14]. (ii)Clinician
validation at scale: the six-disease HEG-TKG panel (§4.4) anchors the methodology, not the full
13,431-disease resource; the 6,250 novel-coverage diseases need expert review at scale. (iii)Extrac-
tion error: ≥2-model consensus still leaves 7.3% genuinely wrong (Table 5), which limits single-triple
9

reliance and favours aggregate or median-based use. (iv)Deep tier never triggered: all triples are
PubMed-extracted; GeneReviews, OMIM, and other databases are not yet integrated. (v)Credibility
score: two of six signals ( citation_count ,is_retracted ) are unpopulated. (vi)Research scope:
literature-derived; direct clinical use requires clinician oversight (Ethics, Appendix K).
References
[1]Md Shamim Ahmed, Maja Dusanic, Moritz Nikolai Kirschner, Elisabeth Nyoungui, Jana
Zschüntzsch, Lukas Galke Poech, and Richard Röttger. The provenance gap in clinical AI:
Evidence-traceable temporal knowledge graphs for rare disease reasoning.arXiv preprint
arXiv:2604.17114, Apr 2026. doi: 10.48550/arXiv.2604.17114. URL https://arxiv.org/
abs/2604.17114.
[2]Mehdi Ali, Max Berrendorf, Charles Tapley Hoyt, Laurent Vermue, Sahand Sharifzadeh,
V olker Tresp, and Jens Lehmann. PyKEEN 1.0: A Python library for training and evaluating
knowledge graph embeddings.Journal of Machine Learning Research, 22(82):1–6, 2021.
arXiv:2007.14175.
[3]Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko.
Translating embeddings for modeling multi-relational data. InAdvances in Neural Information
Processing Systems, volume 26, 2013.
[4]Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable
precision medicine.Scientific Data, 10(1):67, 2023. doi: 10.1038/s41597-023-01960-3.
[5]Xuanzhong Chen, Xiaohao Mao, Qihan Guo, Lun Wang, Shuyang Zhang, and Ting Chen.
RareBench: Can LLMs serve as rare diseases specialists? InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24). ACM, 2024. doi:
10.1145/3637528.3671576. URLhttps://doi.org/10.1145/3637528.3671576.
[6]Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. HyTE: Hyperplane-based
temporally aware knowledge graph embedding. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing (EMNLP), pages 2001–2011, 2018. doi:
10.18653/v1/D18-1225.
[7]Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna
Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets.Communications of the
ACM, 64(12):86–92, 2021. doi: 10.1145/3458723.
[8]Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding. InInternational
Conference on Learning Representations (ICLR), 2021. arXiv:2009.03300.
[9]Daniel Scott Himmelstein, Antoine Lizee, Christine Hessler, Leo Brueggeman, Sabrina L
Chen, Dexter Hadley, Ari Green, Pouya Khankhanian, and Sergio E Baranzini. Systematic
integration of biomedical knowledge prioritizes drugs for repurposing.eLife, 6:e26726, 2017.
doi: 10.7554/eLife.26726.
[10] Julius O B Jacobsen, Michael Baudis, Gareth S Baynam, et al. The GA4GH Phenopacket
schema defines a computable representation of clinical data.Nature Biotechnology, 40(6):
817–820, 2022. doi: 10.1038/s41587-022-01357-4.
[11] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What
disease does this patient have? A large-scale open domain question answering dataset from
medical exams.Applied Sciences, 11(14):6421, 2021. doi: 10.3390/app11146421.
[12] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. PubMedQA:
A dataset for biomedical research question answering. InProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing (EMNLP), pages 2567–2577, 2019. doi:
10.18653/v1/D19-1259.
[13] Sebastian Köhler, Michael Gargano, Nicolas Matentzoglu, et al. The Human Phenotype
Ontology in 2024: phenotypes around the world.Nucleic Acids Research, 52(D1):D1333–
D1346, 2024. doi: 10.1093/nar/gkad1005.
10

[14] Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. Self-alignment
pretraining for biomedical entity representations. InProceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies (NAACL-HLT), pages 4228–4238, 2021. doi: 10.18653/v1/2021.naacl-main.334.
[15] Yuxing Lu, Wei Wu, Xukai Zhao, Rui Peng, and Jinzhuo Wang. KARMA: Leveraging
multi-agent LLMs for automated knowledge graph enrichment. InAdvances in Neural In-
formation Processing Systems (NeurIPS), volume 38, 2025. NeurIPS 2025 Spotlight Poster.
arXiv:2502.06472.
[16] Yuchao Qin, Mihaela van der Schaar, and Changhee Lee. T-Phenotype: Discovering phenotypes
of predictive temporal patterns in disease progression. InProceedings of the 26th International
Conference on Artificial Intelligence and Statistics (AISTATS), volume 206 ofProceedings of
Machine Learning Research, pages 3466–3492, 2023.
[17] Eric W Sayers, Evan E Bolton, J Rodney Brister, Kathi Canese, Jessica Chan, Donald C Comeau,
Ryan Connor, Kathryn Funk, Chris Kelly, Sunghwan Kim, et al. Database resources of the
national center for biotechnology information.Nucleic Acids Research, 50(D1):D20–D26, 2022.
doi: 10.1093/nar/gkab1112.
[18] Xiaorui Su, Yibo Wang, Shanghua Gao, Xiaolong Liu, Valentina Giunchiglia, Djork-Arné
Clevert, and Marinka Zitnik. KGARevion: An AI agent for knowledge-intensive biomedical
QA. InThe Thirteenth International Conference on Learning Representations (ICLR), 2025.
URLhttps://openreview.net/forum?id=tnB94WQGrn.
[19] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. RotatE: Knowledge graph em-
bedding by relational rotation in complex space. InInternational Conference on Learning
Representations (ICLR), 2019. arXiv:1902.10197.
[20] Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-Evolve: Deep temporal
reasoning for dynamic knowledge graphs. InProceedings of the 34th International Conference
on Machine Learning (ICML), pages 3462–3471, 2017.
[21] George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, et al. An overview of the
BIOASQ large-scale biomedical semantic indexing and question answering competition.BMC
Bioinformatics, 16:138, 2015. doi: 10.1186/s12859-015-0564-6.
[22] Duzhen Zhang, Zixiao Wang, Zhong-Zhi Li, Yahan Yu, Shuncheng Jia, Jiahua Dong, Haotian
Xu, Xing Wu, Yingying Zhang, Tielin Zhang, Jie Yang, Xiuying Chen, and Le Song. Med-
KGent: A large language model agent framework for constructing temporally evolving medical
knowledge graph.arXiv preprint arXiv:2508.12393, 2025. doi: 10.48550/arXiv.2508.12393.
URLhttps://arxiv.org/abs/2508.12393.
[23] Yuan Zhang, Xin Sui, Feng Pan, Kaixian Yu, Keqiao Li, Shubo Tian, Arslan Erdengasileng, Qing
Han, Wanjing Wang, Jianan Wang, Jian Wang, Donghu Sun, Henry Chung, Jun Zhou, Eric Zhou,
Ben Lee, Peili Zhang, Xing Qiu, Tingting Zhao, and Jinfeng Zhang. A comprehensive large-
scale biomedical knowledge graph for AI-powered data-driven biomedical research.Nature
Machine Intelligence, 7:602–614, 2025. doi: 10.1038/s42256-025-01014-w.
[24] Yikai Zheng, Wanquan Liu, Bi Zeng, Yichun Feng, Xiawei Du, Lu Zhou, and Yixue Li. Automat-
ing biomedical knowledge graph construction for context-aware scientific inference.bioRxiv,
2026. doi: 10.64898/2026.01.14.699420. URL https://www.biorxiv.org/content/10.
64898/2026.01.14.699420v1. Preprint, posted 14 January 2026.
Data, Code, and AI Usage
ChronoMedKG (CC BY 4.0) and ChronoTQA are archived on Zenodo at https://doi.org/
10.5281/zenodo.19697542 ; pipeline and experiment code (MIT) are on GitLab at https://
gitlab.sdu.dk/screen4care/chronomedkg , with a Hugging Face mirror planned for camera-
ready release. Construction used DeepSeek V3 and OpenAI models (GPT-4.1-nano, then GPT-4o-
mini), with Claude 3 Haiku as a conditional tiebreaker. The novelty-judge panel used DeepSeek V3,
GPT-4o-mini, and Claude Haiku 4.5; model IDs, settings, and prompts are in Appendix G. Claude
11

(Anthropic) was used for limited editorial assistance, including language refinement and structural
revision under author supervision; Grammarly and Overleaf handled copyediting. Figure 1, the
four-agent pipeline schematic, was created with assistance from an AI diagram-generation tool
and then reviewed and refined by the authors for technical accuracy. All quantitative figures were
generated directly from benchmark data using the released Python scripts. No data-bearing element in
any figure was produced or altered by AI. All scientific content, experimental design, data collection,
analysis, and interpretation were performed and verified by the authors, who take responsibility for
the manuscript content.
A NeurIPS Paper Checklist
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract makes four headline claims, each supported by a numbered section
in the main text: (1) 460,497 validated triples derived from 13,431 PrimeKG diseases; (2)
92.7% effective accuracy against Orphadata with a strict 7.3% genuine-error rate (main
paper Section 4.3); (3) 87.9% verified accuracy on the three-LLM novel-coverage audit
(main paper Section 4.1); (4) selective retrieval against ChronoMedKG rescues 47–65%
of LLM long-tail failures across three frontier models, alongside a TransE link-prediction
ablation that confirms the temporal annotations carry signal under standard KG evaluation
(Appendix I.1). All claims are cross-referenced in main paper Sections 3–6.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Main paper Section 7 (Discussion) lists six explicit limitations, in descend-
ing severity: (i) entity canonicalisation (3 ×larger phenotype entity space than HPOA,
suppresses link-prediction MRR; SapBERT slated for v1.1), (ii) clinician validation at
scale (six-disease HEG-TKG anchor only; 13,431-disease scale unvalidated by clinicians),
(iii) 7.3% extraction error after ≥2-model consensus (Table 5), (iv) Deep tier never triggered
(GeneReviews/OMIM not yet integrated), (v) credibility score uncalibrated in v1.0 (two
of six signals unpopulated; recalibration is v1.1), (vi) research scope (literature-derived;
direct clinical use requires clinician oversight). The LLM judge calibration in Appendix H.3
further quantifies the limitation of automated quality assessment.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [N/A]
Justification: This is a dataset/benchmark paper; no theoretical results are claimed.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper?
Answer: [Yes]
Justification: Appendix C describes the 4-agent pipeline, consensus algorithm, credibility
scoring formula, and experimental setup with hyperparameters (link prediction: TransE,
100 epochs, 100-dim, lr=0.01, 3 seeds). All experiment scripts are in the released code
repository.
5.Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results?
Answer: [Yes]
Justification: Section J provides URLs to the dataset (Zenodo DOI) and code repository
(GitLab). Instructions include exact Python environment (Python 3.12), dependencies
(requirements.txt), and random seeds for all experiments.
6.Experimental setting/details
Question: Does the paper specify all the training and test details necessary to understand the
results?
Answer: [Yes]
12

Justification: Section C.5 details train/val/test splits (80/10/10), hyperparameters (TransE
embedding dim 100, batch size 1024, Adam lr=0.01, 100 epochs), optimizer, and random
seeds (42, 7, 123).
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: Link prediction results (Table 17) report mean ±std across 3 seeds with paired
t-testp-values. Validation accuracy uses exact counts (n=2,563 for Orphadata, n=365 for
HPOA, n=116 for GeneReviews).
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources needed to reproduce the experiments?
Answer: [Yes]
Justification: Pipeline cost: $2,400 total, $0.18/disease (LLM API calls). Link prediction
trained on Apple M4 MPS (no GPU required); each 100-epoch training takes ∼15 seconds.
Full 12-run seed experiment completes in∼3 minutes.
9.Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics?
Answer: [Yes]
Justification: The resource is built entirely from open-access biomedical literature (PubMed,
PMC Open Access Subset). No personally-identifiable information is included. Case study
PMC references are to published open-access clinical case reports with patient anonymity
preserved in the source publications. The resource explicitly warns against use for direct
patient care without clinical oversight.
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: Main paper Section 7 (Discussion, Broader Impact) discusses potential positive
impacts (clinical decision support, education, ML research) and negative impacts (risks of
direct clinical use without expert oversight; limitations as literature summaries vs. peer-
reviewed clinical references).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse?
Answer: [Yes]
Justification: Dataset release includes prominent disclaimer that ChronoMedKG is a
literature-summarized resource and must not be used for patient care without clinical
expert oversight. All triples include PMID provenance for verification.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: PrimeKG (CC BY 4.0), HPOA (CC BY 4.0), Orphadata (CC BY-NC 3.0),
GeneReviews (institutional license), PMC Open Access (various CC licenses) are all used
within their terms. Source databases are cited in main paper Section 2 (Related Work).
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: ChronoMedKG dataset is documented via this supplementary material includ-
ing a full Datasheet for Datasets (Appendix B), ChronoTQA benchmark is documented
with construction methodology (main paper Section 5) and full question-type breakdown
(Appendix D).
14.Crowdsourcing and research with human subjects
13

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [N/A]
Justification: No crowdsourcing or human-subject research was conducted. Clinician
evaluation (if included) follows standard academic consultation and is described in methods.
15.Institutional review board (IRB) approvals or equivalent
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [N/A]
Justification: The resource is built from public-domain biomedical literature; no human
subjects were recruited. PMC case reports are published clinical cases with patient privacy
handled by the original publishing journals.
16.Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research?
Answer: [Yes]
Justification: LLMs are central to the pipeline. Main paper Section 3 (Construction)
describes extraction by DeepSeek V3 and an OpenAI primary (GPT-4.1-nano in early runs,
GPT-4o-mini in later runs), with Claude 3 Haiku as a conditional tiebreaker. The three-
LLM novelty-judge panel (main paper Section 4.1) uses DeepSeek V3, GPT-4o-mini, and
Claude Haiku 4.5. Model versions, parameters (temperature=0), and prompt templates are
in Appendix G.
14

B Datasheet for ChronoMedKG
Following the Datasheets for Datasets template [7].
B.1 Motivation
For what purpose was the dataset created?ChronoMedKG was created to fill a structural gap in
biomedical knowledge graphs: no existing resource captureswhenin the disease course clinical facts
are true (onset ages, progression stages, milestone timing). Existing KGs treat all associations as
static and timeless, limiting their utility for temporal clinical reasoning tasks.
Who created the dataset?The author team listed on the main paper title page (University of
Southern Denmark and University of Hamburg).
Who funded the creation of the dataset?Screen4Care has received funding from the Innovative
Medicines Initiative 2 Joint Undertaking (JU) under grant agreement No 101034427. The JU
receives support from the European Union’s Horizon 2020 research and innovation programme and
EFPIA. The funders had no role in study design, data collection and analysis, decision to publish, or
preparation of the manuscript. Total direct costs: ∼$2,400 (LLM API calls, all-in across construction
including retries and failed extractions; $0.18/disease average) plus researcher time.
B.2 Composition
What do the instances that comprise the dataset represent?Each instance is atemporally-
grounded biomedical knowledge triple: (subject, relation, object) with associated temporal metadata
(onset age range, progression stage, milestone, evidence year), evidence provenance (PMID, study
type, credibility score), and multi-LLM consensus confidence.
How many instances are there in total?460,497 validated consensus triples derived from 13,431
of PrimeKG’s 17,080 diseases processed by the pipeline (10,852 of those diseases retain surviving
triples after multi-LLM consensus and Quality Controller filtering); 253,094 unique entities (diseases,
phenotypes, drugs, genes, anatomical terms, biological processes) across 10 relation types.
Does the dataset contain all possible instances or is it a sample?It is a sample of biomedical
knowledge. Coverage is limited by (a) PubMed’s indexed literature, (b) the 4-agent pipeline’s
extraction scope (13,431 of 17,080 PrimeKG diseases processed), and (c) the consensus filter (only
triples with≥2-model agreement retained).
What data does each instance consist of?See main paper Section 3.2. Each triple has: sub-
ject, relation, object, temporal metadata ( onset_age_min ,onset_age_max ,progression_stage ,
milestone ), evidence ( source_ids list of PMIDs, credibility_score ,study_type ,
consensus_confidence), and quality grade (A = PrimeKG-confirmed, B = novel).
Is there a label or target associated with each instance?Triples are not labeled instances for
supervised learning; the resource is a knowledge graph. For downstream tasks (e.g., link prediction),
triples serve as observed positive examples in the graph.
Is any information missing from individual instances?NA
Are relationships between individual instances made explicit?Yes: triples share entities, forming
a graph. Edges with the same (head, relation) but different (tail) targets represent alternative related
entities.
Are there recommended data splits?For link prediction experiments, we use 80/10/10 random
splits with random seeds 42, 7, 123 (Appendix C.5).
Are there any errors, sources of noise, or redundancies in the dataset?Yes. Error taxonomy
(main paper Section 4.3): 7.3% genuine errors. Additional sources of noise: (a) entity name variants
for synonymous concepts (e.g., “proximal weakness” vs. “proximal muscle weakness”), (b) onset age
ranges from individual papers that may not represent the general disease population, (c) single-triple
noise pulling aggregated disease-level ranges.
Is the dataset self-contained, or does it link to or otherwise rely on external resources?Partially
self-contained. Each triple links to one or more PubMed PMIDs (external) and entity IDs that align
with PrimeKG (external). The full KG is downloadable as a JSONL file; PMID verification requires
the PubMed API.
15

Does the dataset contain data that might be considered confidential?No. All source material is
public-domain biomedical literature.
Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening,
or might otherwise cause anxiety?No.
Does the dataset identify any subpopulations (e.g., by age, gender)?No. The dataset describes
diseases, not individuals. Age-specific onset ranges apply to disease populations, not to individuals.
Is it possible to identify individuals, either directly or indirectly (i.e., in combination with other
data) from the dataset?No.
Does the dataset contain data that might be considered sensitive in any way?No.
B.3 Collection Process
How was the data associated with each instance acquired?Via a 4-agent automated pipeline
(main paper Section 3) that: (1) profiles each disease against ontologies (MONDO, OMIM, HPO),
(2) retrieves PubMed abstracts and PMC full-text articles via NCBI E-utilities, (3) extracts candidate
triples using 2–3 LLMs in parallel, (4) applies consensus filtering and PrimeKG schema alignment.
What mechanisms or procedures were used to collect the data?Python scripts using OpenAI,
DeepSeek, Anthropic, and Google Gemini APIs. NCBI E-utilities (PubMed, PMC) with registered
API key.
If the dataset is a sample from a larger set, what was the sampling strategy?Literature-adaptive
tiering: diseases with ≥100 PubMed articles received up to 150 documents; diseases with 20–99
articles received all available; diseases with <20 articles received all available (exhaustive coverage
for rare diseases).
Who was involved in the data collection process and how were they compensated?The authors
(primary investigators). No external annotators were used (all labels are from multi-LLM consensus).
LLM API costs:∼$2,400 total (paid out of research budget).
Over what timeframe was the data collected?PubMed literature spanning 1960–2026 (median
publication year 2015). Extraction pipeline ran February–April 2026.
Were any ethical review processes conducted?Not required: the dataset is built from public-domain
literature with no human subjects.
B.4 Preprocessing/cleaning/labeling
Was any preprocessing/cleaning/labeling of the data done?Yes. (1) Entity normalization:
lowercase, strip parentheticals, slash-split multi-term entities. (2) Consensus filtering: only triples
with≥2 LLMs agreeing (fuzzy match ≥80%) pass. (3) Temporal plausibility: age ranges outside [0,
120] filtered. (4) PrimeKG alignment: entity types mapped to PrimeKG’s schema.
Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data?Yes. 13,045,687
raw triples (pre-consensus) are retained in per-disease raw_triples.jsonl files, enabling re-
analysis with different consensus thresholds.
Is the software used to preprocess/clean/label the instances available?Yes: full pipeline code at
the project repository (Appendix J).
B.5 Uses
Has the dataset been used for any tasks already?In this paper: (1) validation against 3 gold
standards, (2) ChronoTQA benchmark construction, (3) coverage gap analysis, (4) evidence decay
audit, (5) unsupervised trajectory clustering, (6) KG link prediction with temporal features.
Is there a repository that links to any or all papers or systems that use the dataset?Yes:
works citing the dataset DOI are tracked automatically via the “Cited by” panel on the Zenodo
record (https://zenodo.org/records/19697543 ), populated through Crossref Event Data and
OpenAIRE. A Hugging Face dataset page mirroring the same tracking will be added at camera-ready.
What (other) tasks could the dataset be used for?Temporal differential diagnosis, phenotype
progression prediction, stage-specific drug repurposing, clinical decision support systems, educational
16

tools, ML benchmarks for temporal biomedical reasoning, computational phenotyping with temporal
features.
Is there anything about the composition of the dataset or the way it was collected and prepro-
cessed/cleaned/labeled that might impact future uses?Yes. (1) ChronoMedKG phenotype entities
use LLM-extracted names, not HPO-controlled vocabulary; future users should consider canonical-
ization (e.g., via SapBERT). (2) The resource should not be used for direct clinical decision-making
without expert oversight. (3) 7.3% of onset ranges are genuinely wrong; aggregate/median-based
usage is more robust than single-triple reliance.
Are there tasks for which the dataset should not be used?Direct patient care decisions, diagnostic
certainty claims, or any clinical use without peer-reviewed validation and expert oversight.
B.6 Distribution
Will the dataset be distributed to third parties outside of the entity on behalf of which the
dataset was created?Yes: ChronoMedKG v0.0.1 is already publicly released on Zenodo under
CC BY 4.0 ( https://zenodo.org/records/19697543 ); a Hugging Face mirror exposing the
datasetsload_dataset()API will be added at camera-ready.
How will the dataset be distributed?Zenodo DOI, HuggingFace Datasets, and GitLab repository.
Format: JSONL for triples, CSV for tabular summaries, Parquet for large-scale queries.
When will the dataset be distributed?v0.0.1 is already distributed on Zenodo (CC BY 4.0, pub-
lic); subsequent versioned releases follow the timeline in rai:dataReleaseTimeline (Croissant
metadata).
Will the dataset be distributed under a copyright or other intellectual property (IP) license?
Dataset: CC BY 4.0. Code: MIT License.
Have any third parties imposed IP-based or other restrictions on the data associated with the
instances?No.
Do any export controls or other regulatory restrictions apply to the dataset or to individual
instances?No.
B.7 Maintenance
Who will be supporting/hosting/maintaining the dataset?M. S. Ahmed (University of South-
ern Denmark, shamim@imada.sdu.dk ) maintains the dataset on Zenodo under the concept DOI
https://doi.org/10.5281/zenodo.19697542 , which always resolves to the latest version. New
versions and errata are published as fresh Zenodo deposits; a Hugging Face mirror surfacing the same
release timeline will be added at camera-ready.
How can the owner/curator/manager of the dataset be contacted?By email at
shamim@imada.sdu.dk , or via the GitLab issue tracker at https://gitlab.sdu.dk/
screen4care/chronomedkg/-/issues.
Is there an erratum?The decision log ( docs/decision_log.md ) in the public code repository
tracks corrections made during development. Version-specific errata for released artefacts are
published on Zenodo as new deposits under the concept DOI; the Hugging Face mirror added at
camera-ready will surface the same release history.
Will the dataset be updated?Planned updates: (1) entity canonicalisation via SapBERT and
citation_count /is_retracted credibility-signal backfill (v1.1), (2) activation of the GeneRe-
views / OMIM Deep tier (v1.2), (3) clinician-validated subset (v2.0). All released versions are
preserved as frozen Zenodo DOIs.
If the dataset relates to people, are there applicable limits on the retention of the data?N/A (no
personal data).
Will older versions of the dataset continue to be supported/hosted/maintained?Yes: all released
versions are preserved on Zenodo with unique DOIs.
If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for
them to do so?Yes: GitLab merge requests and issue tracker.
17

C Extended Methods
C.1 Full Schema Example
A real ChronoMedKG record (Becker muscular dystrophy →cardiomyopathy, edge ef58608a735b )
reproduced verbatim from the releasedvalidated_triples.jsonl, illustrating every field:
{"edge_id": "ef58608a735b",
"source_id": "10311", "source_type": "disease",
"source_name": "Becker muscular dystrophy",
"relation": "disease_phenotype_positive",
"target_id": "1638", "target_type": "phenotype",
"target_name": "cardiomyopathy",
"temporal": {"onset_age_min": 20, "onset_age_max": 40,
"progression_stage": "adult",
"milestone": "cardiac involvement",
"temporal_qualifier": null,
"discovery_date": null, "validity_start": null,
"validity_end": null, "superseded_by": null,
"temporal_resolution": "unknown",
"duration": null, "treatment_start_age": null},
"evidence": {"tier": 2,
"source_ids": ["PMID:38224155"],
"evidence_text": "Cardiac involvement in BMD often
manifests in the third to fourth decade",
"study_type": "review",
"credibility_score": 0.395,
"consensus_confidence": 1.0,
"extraction_models": ["claude-haiku"],
"extraction_method": "tier2_llm_consensus",
"citation_count": null, "is_retracted": false},
"conditions": null,
"extraction_date": "2026-04-03",
"pipeline_version": "1.0.0",
"disease_profile_id": "MONDO:10311",
"quality_grade": "A"}
Two notes on this v1.0 record. (a) extraction_models stores the model whose surviving ex-
traction was kept as the representative row; the canonical record of multi-LLM agreement is
consensus_confidence (1.0 = full agreement across the models that processed the document),
with the relationship between the two fields detailed below. (b) Two of six credibility signals
(citation_count ,is_retracted ) are not populated in v1.0 and await v1.1 backfill, as flagged in
Limitation (v) of the main paper.
C.2 4-Agent Pipeline Details
Agent 1 - Disease Profiler.For each disease identifier, queries MONDO, OMIM, and HPO ontologies
to generate a YAML configuration containing: disease name, synonyms, differential diagnoses,
known genes, key phenotypes, PubMed article count, PMC full-text availability, and coverage flag
(rich/moderate/sparse). No LLM calls.
Agent 2 - Evidence Harvester.Retrieves abstracts via NCBI E-utilities ( esearch +efetch ). For
diseases with sufficient literature, also retrieves PMC full-text via the PMC Open Access Subset
API. Documents are ranked by credibility score (journal tier, citation velocity) and capped per tier:
standard tier gets up to 150 documents, light and minimal tiers get all available.
Agent 3 - Knowledge Extractor.Each document is sent in parallel to 2–3 LLMs: DeepSeek V3
(primary, low cost), GPT-4.1-nano or GPT-4o-mini (primary), Claude 3 Haiku (conditional tiebreaker,
invoked when ≥1 primary model returns 0 triples). All LLMs use temperature 0 and 120-second
timeout. Extraction prompts are in Appendix G.
Agent 4 - Quality Controller.Applies per-triple validation: (1) non-empty entities, (2) age bounds
[0, 120], (3) self-reference prevention, (4) PrimeKG schema alignment (Grade A if confirmed, B if
novel), (5) credibility scoring.
18

Duchenne muscular dystrophy
Temporal phenotype profile
Presymptomatic
(0-2y)Ambulatory
(2-8y)Non-ambulatory
(8-14y)Respiratory
(14-20y+)
0 5 10 15 20 years
Phenotypes emerging at each stage:
Elevated CK (0-2y) detectable at birth
Walking delay (2-5y)
Calf pseudohyp. (2-6y)
Gowers sign (5-8y)
Loss of ambulation (8-12y)
Cardiomyopathy (10-18y)
Respiratory insuf. (12-20y)
Motor Cardiac Respiratory
Static KG:
"DMD is associated with cardiomyopathy"vs.ChronoMedKG:
"cardiomyopathy appears at age 10-18 in DMD"
Temporal grounding transforms "what" into "what + when" for clinical decision supportFigure 3:Example temporal phenotype profile for Duchenne muscular dystrophy.Stages (presymptomatic, ambulatory,
non-ambulatory, respiratory) partition the age axis; phenotype bars show the onset window for each clinical feature, colour-
coded by system (motor, cardiac, respiratory). The static-KG statement “DMD is associated with cardiomyopathy” is replaced
in ChronoMedKG by a stage-aware statement: “cardiomyopathy appears at age 10–18 in DMD”, which supports temporal
differential diagnosis and age-specific screening decisions that static KGs cannot.
C.3 Multi-LLM Consensus Algorithm
function compute_consensus ( per_model_triples , threshold =2):
# Step 1: collect candidates
candidates = [(s, r, o, model ) for model , triples in
per_model_triples
for s, r, o in triples ]
# Step 2: normalize
normalized = [( normalize (s), canonicalize (r), normalize (o), model )
for s, r, o, model in candidates ]
# Step 3: Union - Find clustering by entity similarity
clusters = UnionFind ()
for i, j in pairs ( normalized ):
if relation [i] == relation [j] and
same_model (i, j) is False and
fuzz . ratio (s[i], s[j]) >= 80 and
fuzz . ratio (o[i], o[j]) >= 80:
clusters . union (i, j)
# Step 4: filter by consensus threshold
consensus = []
for cluster in clusters . groups ():
models_agreeing = set( model for _, _, _, model in cluster )
if len( models_agreeing ) >= threshold :
representative = max( cluster , key = lambda t: t. confidence )
representative . confidence = len( models_agreeing ) /
total_models
consensus . append ( representative )
return consensus
Default threshold is 2 (at least 2 distinct LLMs must agree). Fuzzy match threshold of 80% is based
on preliminary calibration: stricter thresholds (90%) exclude legitimate synonym variants; looser
19

(70%) admit false positives. The 3.53% consensus retention rate (460,497 of 13.05M raw triples,
main paper §3) reflects per-model duplicate-fact collapse under fuzzy match plus single-model triples
filtered by the≥2-model threshold, not signal loss.
C.4 Six-Signal Credibility Scoring
For each source document, credibility = weighted combination:
credibility= 0.15·journal_tier+ 0.15·citation_velocity
+ 0.25·study_type_weight+ 0.15·replication_signal
+ 0.15·retraction_check+ 0.15·llm_consensus
Study type weights: meta-analysis 1.0 >guideline 0.95 >RCT 0.9 >database 0.85 >cohort 0.7 >
case-control 0.6>review 0.5>case-series 0.4>case-report 0.3>expert-opinion 0.2>other 0.1.
C.5 Link Prediction Experimental Setup
Data:1,302 diseases present in both HPOA and ChronoMedKG (disease name normalized: lowercase,
common suffixes removed). HPOA edges: 31,061 structure / 31,077 with onset bins. ChronoMedKG
edges: 26,184 structure / 28,161 with onset bins.
Onset bins (aligned between HPOA and ChronoMedKG):congenital, neonatal, infantile,
early_childhood (ChronoMedKG only), childhood, juvenile, young_adult, adult, middle_age (HPOA
only), late_onset.
Split:80% train / 10% validation / 10% test; one fixed random split per seed (seeds listed below).
Model:TransE (PyKEEN implementation), embedding dimension 100, margin 1.0.
Training:100 epochs, batch size 1024, Adam optimizer at lr=0.01, negative sampling ratio 1:1.
Evaluation:Hits@1, Hits@3, Hits@10, Mean Reciprocal Rank (MRR). Computed on both head
and tail prediction, averaged.
Seeds:42, 7, 123 (3 independent runs per condition).
Statistical testing:Paired t-test on MRR across seeds, treating HPOA-struct →HPOA-temporal and
ChronoMedKG-struct→ChronoMedKG-temporal as paired observations.
Hardware:Apple M4 (PyTorch MPS backend). Each 100-epoch training takes ∼15 seconds; the
full 12-run experiment (2sources×2conditions×3seeds) completes in∼3 minutes.
20

D Extended ChronoTQA Benchmark Statistics
D.1 Biomedical QA Benchmark Landscape
ChronoTQA situated alongside the other biomedical QA benchmarks discussed in main paper §2:
ChronoTQA is the only benchmark with age- or stage-conditioned reasoning and dual Tier 1 (external
gold standard) + Tier 2 (PMID-traceable KG) grounding.
Table 10: Biomedical QA benchmarks. ChronoTQA is the only one with age/stage-conditioned reasoning and Tier 1 external +
Tier 2 PMID-traceable grounding.
Benchmark N Format Domain Source / Q Temporal
MedQA [11] 12,723 MCQ USMLE clinical — No
PubMedQA [12] 1,000*Yes/No/Maybe Biomedical literature Per-abstract No
BioASQ [21]∼5k Mixed Biomedical Per-snippet No
MMLU-Medical [8]∼1k MCQ Medical knowledge — No
RareBench [5]∼5.4k Multi-task Rare-disease Dx Per-case No
ChronoTQA 3,341 MCQ + free-text Temporal clinical Tier 1 ext / Tier 2 PMID Yes
*PQA-L expert-labeled subset; the artificially-generated PQA-A pool adds∼211k.
Head-to-head with concurrent agentic KG systems.Direct comparison of ChronoMedKG against
KARMA, MedKGent, or AutoBioKG on ChronoTQA is not feasible: none of the three releases
exposes a temporal-query interface, and each encodes a different temporal or contextual axis (main
paper §2). Reproducing the systems against ChronoTQA’s temporal demands would require re-
implementing their pipelines, which is out of scope for the v1 release.
D.2 Question Difficulty Distribution
Each ChronoTQA question is labeled with a difficulty level based on automated heuristics:
•Easy:Common diseases, standard onset ranges (e.g., DMD age 3–5 years)
•Medium:Moderate-frequency diseases, some variability in onset literature
•Hard:Rare diseases, limited literature, or ambiguous onset ranges
Table 11: Difficulty distribution across ChronoTQA question types. Difficulty labels are assigned per question type based on
the reasoning demand: exact onset-window and ordering subtasks arehard(require precise age ranges or sequences), coarse
comparisons aremedium, and single-fact static lookups areeasy.
Type Easy Medium Hard Total
Temporal window 0 800 0 800
Temporal differential Dx 0 0 687 687
Cross-disease comparison 0 600 0 600
Phenotype ordering 0 0 395 395
Stage-conditional 0 0 200 200
Phenopackets onset 0 0 147 147
Negative temporal MCQ 0 0 12 12
Static control drug 250 0 0 250
Static control gene 250 0 0 250
Total 500 1,400 1,441 3,341
D.3 Disease Coverage in ChronoTQA
Of the 13,431 PrimeKG diseases the pipeline processed, 2,315 appear in at least one ChronoTQA
question. Coverage per disease averages 1.44 questions.
D.4 Example Questions per Type
Temporal window:“Is age 8 years within the typical onset window for Kleefstra Syndrome Due To
9q34 Microdeletion?” Gold: No (0–2 years).
Temporal differential Dx:“A patient presents with symptoms during the prenatal period. Based
on typical age of onset, which of the following diseases is most consistent: (A) Acute Zonal Occult
Outer Retinopathy, (B) Lipodystrophy Due To Peptidic Growth Factors Deficiency, (C) Isolated
Congenital Microcephaly, (D) Trichofolliculoma?” Gold: C.
Cross-disease comparison:“Which disease typically has an earlier age of onset: Maffucci Syndrome
or Congenital Hydrocephalus?” Gold: Congenital Hydrocephalus.
21

Phenopackets onset:“At what age does ‘proptosis’ typically present in Chitayat Syndrome? (Based
on patient case data)” Gold: 0.0–2.7 years (from 2 cases).
Stage-conditional:“What phenotypes are characteristic of the non-ambulatory stage of Duchenne
Muscular Dystrophy?” Gold: Loss of independent ambulation, joint contractures, Gowers sign.
Phenotype ordering:“Rank the following clinical milestones for Duchenne Muscular Dystrophy by
typical age of occurrence: diagnosis, loss of ambulation, cardiomyopathy onset.” Gold: diagnosis →
loss of ambulation→cardiomyopathy.
22

E Error Analysis: Concrete Examples
Below are representative examples from each error taxonomy category (main paper Table 5). Exam-
ples are drawn from the 2,563 ChronoMedKG–Orphadata matched diseases.
E.1 Correct (contained): 50.1%
Example: Duchenne muscular dystrophy. ChronoMedKG median range: 2–5y. Orphadata: 1–5y.
Within gold standard range.
E.2 Adjacent stage: 15.6%
Example: Disease X. ChronoMedKG median: 2–12y (childhood). Orphadata: 5–15y (juvenile).
Ranges overlap at 5–12y but ChronoMedKG extends slightly earlier and gold slightly later. Clinically
borderline, not a real error.
E.3 Granularity mismatch: 13.8%
Example: Duchenne muscular dystrophy. Orphadata gives one range (1–5y) for the disease.
ChronoMedKG gives per-phenotype ranges: walking delay 2–5y, cardiomyopathy 10–18y. Ag-
gregated ChronoMedKG range (2–18y) does not fit Orphadata’s 1–5y, but neither is wrong:
ChronoMedKG just tracks more detail.
E.4 ChronoMedKG wider but overlaps: 6.7%
Example: Marfan syndrome. Orphadata: childhood (5–12y). ChronoMedKG: 0–40y because
ChronoMedKG captures congenital features (aortic dilation detected prenatally) and adult features
(progressive aortic root dilation). Core childhood period still overlaps.
E.5 Single-triple noise: 5.7%
Example: Disease Y . ChronoMedKG has 50 triples about this disease; 49 have onset in 2–8y, but
one outlier triple has onset at 60y (possibly misattributed from a comorbid condition). Median
aggregation still correct; min/max includes the noise.
E.6 Genuinely wrong: 7.3%
Example: Disease Z. ChronoMedKG median range: 30–60y (adult onset). Orphadata: 0–1y (neona-
tal). No overlap, >10-year gap. Likely due to extraction from papers where patient age at report was
conflated with disease onset age.
23

F PMC Clinical Case Studies (All 31 Cases)
PMC clinical case diagnostic odysseys
Selected cases showing years from initial misdiagnosis to correct rare disease diagnosis
0 5 10 15 20 25 30 35 40 45
Diagnostic delay (years)SMA
PMC11154026
Muscular dystrophy dx (age 5) 42 years SMA dx (age 47)
Wilson disease
PMC5688900
Schizophrenia dx (age 14) 26 years Wilson dx (age 40)
X-ALD
PMC8634231
Bipolar dx (age 16)X-ALD dx (age 24)
8-year delay
DMD
PMC5417281
CP dxDMD dx (age 4)
3.6-year delay
(+27 more cases in dataset)
Misdiagnosis period Correct diagnosis
Temporal phenotype profiles could flag earlier:
"Wilson disease: neuropsychiatric onset 20-40y, hepatic involvement 15-45y"
Clinician prompted to investigate hepatolenticular degeneration at first psychiatric presentation
Figure 4: Diagnostic odysseys in four representative PMC cases. Pink bars show the misdiagnosis period (from initial
presentation to correct diagnosis); green check marks the correct diagnosis. Delays range from 3.6 to 42 years across 31 cases
(median 14y; full list in Table 12). The green callout illustrates how a ChronoMedKG phenotype profile could prompt earlier
investigation, using Wilson disease as the example.
Table 12: All 31 PMC open-access clinical case reports used as qualitative case studies. Diagnostic de-
lay is the number of years between initial symptom onset and correct diagnosis. All cases are verifiable at
https://pmc.ncbi.nlm.nih.gov/articles/<PMC_ID>/.
PMC ID Correct Diagnosis Delay Misdiagnosis Category
PMC5417281 Duchenne muscular dystrophy 3.6y Cerebral palsy + epilepsy Neuromuscular
PMC6243516 LGMD2A (calpainopathy) 8y Duchenne muscular dystrophy Neuromuscular
PMC5526785 Oculopharyngeal MD 8y Myasthenia gravis Neuromuscular
PMC5787973 Duchenne muscular dystrophy — None (reference) Neuromuscular
PMC3454578 Becker MD w/ cardiomyopathy — None (reference) Neuromuscular
PMC8597609 Myhre syndrome (SMAD4) 1.5y Marfan syndrome Connective tissue
PMC4639334 Late-onset Pompe disease 14y Muscular dystrophy Lysosomal
PMC11154026 Spinal muscular atrophy 42y Muscular dystrophy Neuromuscular
PMC9468738 Vascular EDS 22.5y Hemophilia (negative workup) Connective tissue
PMC5688900 Wilson disease 26y Schizophrenia Neurogenetic
PMC12131641 Late-onset Fabry disease 16y Type 2 diabetic nephropathy Lysosomal
PMC3756019 Fabry disease (heterozygous) 5y Multiple sclerosis Lysosomal
PMC10519872 Late-onset MELAS 2y Coronary artery disease Mitochondrial
PMC9549677 MELAS syndrome 4y Autoimmune encephalitis Mitochondrial
PMC11938598 Adult-onset Leigh 19.5y Hypertensive encephalopathy Mitochondrial
PMC3669838 MPS IIIB (Sanfilippo) 21y Attention deficit disorder Lysosomal
PMC12831942 Alstrom syndrome 24y None (classical) Ciliopathy
PMC10619069 X-ALD (cerebral) 20y Tumefactive MS Peroxisomal
PMC8634231 X-ALD (adult cerebral) 8y Bipolar affective disorder Peroxisomal
PMC10319965 Bardet-Biedl syndrome 14y None (heterogeneous) Ciliopathy
PMC11196773 Bardet-Biedl syndrome 7y None (heterogeneous) Ciliopathy
PMC11468024 Long QT syndrome type 2 1y Epilepsy Channelopathy
PMC8418736 Long QT syndrome type 2 22y Refractory epilepsy Channelopathy
PMC7132987 Danon disease (LAMP2) 2y Sarcomeric HCM Cardiomyopathy
PMC11583564 Gaucher disease type 1 34y Chronic asthma Lysosomal
PMC11346882 Gaucher disease 20y Idiopathic thrombocytopenia Lysosomal
PMC5994559 Chronic granulomatous disease 36y Tuberculosis / vasculitis Immunodeficiency
PMC3326000 Chronic granulomatous disease 50y Fungal allergy / ABPA Immunodeficiency
24

Table 12 – continued
PMC ID Correct Diagnosis Delay Misdiagnosis Category
PMC11101243 Common variable immunodefici. 9y Immune thrombocytopenia Immunodeficiency
PMC3535999 Osteogenesis imperfecta I 1.33y Non-accidental injury Skeletal dysplasia
PMC8047228 Wiskott-Aldrich syndrome 3.9y Pulmonary tuberculosis Immunodeficiency
Summary:26/31 cases involved misdiagnosis. Diagnostic delays range 1–50 years (median 14y).
159 total phenotype-timepoint annotations across 11 disease categories.
25

G LLM Extraction Prompts
G.1 Primary Extraction Prompt
The following prompt is sent to each LLM per document. Variables {disease_name} ,
{known_genes} , etc. are substituted per disease. The PMID and document publication year arenot
part of what the LLM returns: they are attached to every extracted triple by the orchestrator, copied
from the source document’s NCBI E-utilities metadata. This ensures that PMID provenance is deter-
ministic and cannot be hallucinated by the LLM. Reviewers can verify any quoted evidence_text
against the PMID-linked article directly.
You are a temporal biomedical knowledge extraction system. Your PRIMARY
task is to extract relationships WITH TEMPORAL GROUNDING from the text
about {disease_name}.
CRITICAL: Every relationship you extract MUST include temporal information
when available.
Extraction priorities (highest to lowest):
1. TEMPORAL FACTS: onset ages, disease milestones, progression timelines,
treatment timing, discovery dates
2. EVIDENCE-DATED FACTS: relationships anchored by publication year
3. CONDITIONAL FACTS: relationships that depend on age, stage, genetic
subtype
4. STATIC FACTS: general relationships without temporal context
Output format (JSON):
{
"triples": [
{
"subject": "entity name",
"subject_type": "disease|gene/protein|drug|phenotype|anatomy|...",
"relation": "disease_protein|indication|disease_phenotype_positive|...",
"object": "entity name",
"object_type": "same vocabulary as subject_type",
"confidence": "high|medium|low",
"evidence_text": "exact quote from source (max 200 chars)",
"temporal_context": {
"onset_age_min": 3.0,
"onset_age_max": 12.0,
"progression_stage": "ambulatory",
"milestone": "loss of ambulation",
"discovery_year": 2015,
"temporal_qualifier": "by age 12"
},
"conditions": {
"age_group": "pediatric",
"genetic_subtype": "exon deletion"
}
}
]
}
Disease context:
- Name: {disease_name}
- Category: {disease_category}
- Inheritance: {inheritance_pattern}
- Known genes: {known_genes}
- Key phenotypes: {known_phenotypes}
- Differential diagnoses: {differential_diseases}
26

Source text:
{text}
Extract ALL temporally-grounded relationships. Return valid JSON only.
G.2 Temporal Re-extraction Prompt (Second Pass)
For documents that yielded few temporal triples on the first pass, a second focused prompt is used:
SECOND PASS — TEMPORAL ONLY: Find temporal information that was missed.
Extract ONLY relationships with temporal grounding: ages, stages, durations,
progression, milestones. Skip any fact without temporal content.
Output format (JSON): same as primary extraction, but ’temporal_context’
field is REQUIRED (non-null).
27

H Extended Evaluation Experiments
H.1 RAG Experiment on External Gold-Standard Questions (Tier 1)
Three-LLM calibrated-scoring RAG leaderboard (147 Phenopackets questions).We tested RAG
performance on the 147 Phenopackets-grounded onset questions in ChronoTQA across three LLMs
(Claude 3 Haiku, DeepSeek V3, GPT-4o-mini) under four retrieval conditions (NR, PrimeKG, HPOA,
ChronoMedKG). Answers were scored with a calibrated rubric (overlap tolerance = max(0.5,0.5·
wgold), capped at ±2y, with category-keyword matching admitted only when keyword width is
compatible with the gold range), which corrected two bugs in an earlier scorer that had inflated
aggregate accuracy by∼24%.
Table 13: RAG leaderboard on 147 Phenopackets-grounded onset questions (calibrated scoring, n=147 per cell). Pooled
accuracy improvements are bounded by high no-retrieval ceilings; the stronger result is long-tail rescue (main paper Table 9).
Model NR PrimeKG HPOA ChronoMedKG ChronoMedKG−NR
Claude 3 Haiku 76.2% 68.7% 75.5%81.6%+5.4
DeepSeek V3 86.4% 81.6% 83.7% 85.0%−1.4
GPT-4o-mini 76.9% 81.6% 81.0% 79.6%+2.7
Pooled(n=441) 79.8% 77.3% 80.1%82.1%+2.3
Claude 3 Haiku is the only model where the per-model ChronoMedKG-vs-PrimeKG gain reaches
significance (McNemar exact p=0.008 ,+12.9pp); pooled, ChronoMedKG-vs-PrimeKG trends
positive but is not significant ( p=0.06 ). PrimeKG-RAGhurtsaccuracy for two of three models,
consistent with the observation that retrieving non-temporal associations distracts from temporally-
framed questions.
Training-contamination ceiling.Three LLMs score 76–86% on Phenopackets-sourced questions
withno retrieval, a ceiling consistent with the GA4GH Phenopackets corpus having been public since
2019 and likely represented in LLM pretraining. Pooled RAG gains are upper-bounded: there is little
headroom for any retrieval source to improve an LLM that already answers 80%+ from parametric
memory. The better test under this ceiling is the long-tail rescue measure (main paper Table 9).
On the questions each model cannot answer from pretraining, ChronoMedKG improves outcomes
by 47–65% across three LLM families. We therefore frame ChronoMedKG’s RAG contribution
as selective: it delivers the largest gain where static KGs and coarse-onset resources fail, not as a
pooled-accuracy claim against the ceiling set by what these models already know.
The underlying capability gap this RAG experiment targets is quantified independently in §H.2: across
four frontier LLMs (ChatGPT, Gemini, Claude, DeepSeek) on 120 ChronoTQA items posed without
retrieval, temporal accuracy trails static accuracy by a mean +30.1 pp, and free-text Phenopackets
onset questions collapse to a 4.5% mean across models, the exact subtask ChronoMedKG is designed
to serve.
H.2 Parametric-Knowledge Gap Across Four LLMs
The headline static-vs-temporal gap (main paper Table 7 in Section 5) was measured by sampling
120 questions from ChronoTQA (stratified across temporal-window, temporal differential diagnosis,
cross-disease onset comparison, Phenopackets-grounded onset, and static controls) and posing them
without retrieval to four LLMs via their web chat interfaces (no API calls, no system prompts
beyond the benchmark’s formatting instructions): OpenAI GPT-4o-mini (ChatGPT), Google Gemini,
Anthropic Claude, and DeepSeek-V3. Answers were scored against the gold label using exact match
for multiple-choice and Yes/No items and bidirectional substring match for free-text disease names
and onset bins.
The pattern is especially stark on free-text Phenopackets onset questions (“At what age does phenotype
Xtypically present in disease Y?”): the four models score 0%, 12%, 0%, and 6% (mean 4.5%),
against an 80%+ static-control ceiling (main paper Table 7). This is precisely the task where
parametric knowledge is insufficient and where ChronoMedKG provides directly queryable answers
with PMID-grounded provenance. Full per-model breakdowns and answer logs are released in
data/benchmark/chat_eval_summary.json.
H.3 LLM-as-Judge Calibration
A DeepSeek V3 judge tested on 100 diseases with known ground truth (50 where ChronoMedKG is
objectively correct, 50 where ChronoMedKG is objectively wrong) produces: 77.1% “supported” on
28

correct cases (22.9% false negative rate) and 46.8% “supported” on wrong cases. The single-judge
baseline is too lenient on errors and too strict on correct answers to be a primary validation tool;
the three-LLM panel used for the novel-coverage audit (§H.5) reduces both directions of bias by
requiring verifiable-majority agreement, and the 87.9% reported in main paper Section 4.1 is taken
only after that filter. The 92.7% Orphadata accuracy uses direct gold-standard comparison rather than
LLM judging.
H.4 Error Taxonomy Visualisation
Error taxonomy: 2,563 disagreements with Orphadata
Classification of temporal range mismatches
100%
80%
60%
40%
20%
0%50.1%
Contained
15.6%
13.8%
6.7%
5.7%
7.3%Contained (correct)
ChronoMedKG range falls within Orphadata range
Adjacent stage
Off by one clinical stage boundary
Granularity mismatch
Different temporal resolution ("childhood" vs "3-8y")
ChronoMedKG wider, overlaps
Broader range but still overlapping with reference
Single-triple noise
Isolated outlier triple from one sourceGenuinely wrong
Factually incorrect temporal range
Correct (50.1%) Boundary / granularity (41.8%) Genuine error (7.3%)
92.7% of disagreements reflect boundary effects or correct containment, not factual errors
Figure 5:Classification of 2,563 ChronoMedKG–Orphadata onset-range disagreements.Only 7.3% are factually incorrect
temporal ranges (no overlap, >10y gap from gold). 50.1% are strictly contained within the gold range, and a further 41.8%
reflect boundary effects or granularity mismatches (e.g., ChronoMedKG’s “3–8y” vs. Orphadata’s “childhood”), not extraction
errors. Percentages sum to 99.2% due to rounding.
H.5 Novel-Coverage Verification: Full Protocol, Error Taxonomy, Scope
Full sampling and protocol.We drew n=100 diseases from the 6,250 novel-coverage set (seed =42),
stratified by literature tier (Standard/Light/Minimal) and onset bucket (prenatal, infantile, childhood,
adolescent, adult, late-adult). For each disease we selected the single triple whose evidence text
contained the most informative timing keyword (longest keyword-bearing span), producing one
(claim, evidence) pair per disease. Three judges (DeepSeek V3, GPT-4o-mini, Claude Haiku 4.5,
all temperature 0.0) independently rated each pair under a citation-aware chain-of-thought prompt
that required, in order: (1) quote the timing clause from the evidence verbatim, (2) translate the
clause to a numeric age range using a fixed clinical-era lookup (prenatal 0–0 y, infancy 0–1, early
childhood 1–5, childhood 1–11, adolescence 10–18, adulthood 18–65, older adulthood ≥65), (3)
compare per triple, not per disease, against the claimed range, and (4) return one ofSUPPORTED/
PARTIALLY_SUPPORTED/NOT_SUPPORTED/UNVERIFIABLE. Per-triple comparison corrects an
aggregation bug in a prior single-judge audit that compared the disease-level aggregated median to
one triple’s evidence.
Full error taxonomy.Every one of the 11 NOT_SUPPORTED cases failed in the same way. The
extractor picked up a qualitative phrase like "elderly onset" or "mid-trimester" and wrote it to the
temporal_qualifier field, but never translated it into the numeric onset_age_min/onset_age_max fields
that the novel-coverage count actually uses. So the evidence text was fine - the pipeline just dropped
the numeric range on the floor. This is a normalisation bug downstream of extraction, not an LLM
hallucination. A simple fallback mapping from qualifier strings to clinical-era ranges would reclassify
10 of 11 failures; the one remaining case (MONDO:402) has neither a qualifier nor a numeric range
and represents genuine extraction noise (1/100 = 1%). The 6 three-way-split cases are dominated
by disagreement over whether broad qualitative phrases (“adult”, “late-onset”) sufficiently support
29

narrower numeric claims, which is a semantic question about claim width rather than evidence
presence.
Scope of this audit.The 87.9% figure applies to the novel-coverage subset only; it is not a re-estimate
of the 92.7% Orphadata accuracy (different denominator, different evidence). Scaling the audit to all
6,250 novel-coverage diseases, and to the full 13,431-disease resource, is open work; the automation
built here ( scripts/llm_judge_novelty_v2_multi.py , 3 providers async, ∼$2 per 100 triples)
is released with the resource to let reviewers and downstream users replicate or extend the audit on
their own samples.
H.6 Clinician Validation via HEG-TKG: Full Likert Breakdown
The extraction methodology that ChronoMedKG scales was evaluated by a three-clinician panel in
HEG-TKG [ 1] on six rare neuromuscular diseases (three pairs: DMD/BMD, MG/LEMS, CIDP/GBS)
that are also covered in ChronoMedKG. Each clinician independently rated single system-generated
outputs (either vanilla GPT-4.1 or HEG-TKG, blinded to system identity, shuffled with a deterministic
seed) on five Likert dimensions: Verifiability (D1), Actionability (D2), Temporal Precision (D3),
Non-Expert Safety (D4), and Clinical Completeness (D5). Statistical inference uses Mann–Whitney U
with Benjamini–Hochberg correction across five comparisons and bootstrap 95% confidence intervals
from 10,000 resamples.
Table 14: Clinician ratings of HEG-TKG vs. vanilla GPT-4.1 on six shared diseases. C1 = senior neurologist (36-case reduced
pack: 18 vanilla, 18 HEG-TKG); C2 = neurologist (72-case full pack: 36 per arm). Scores on 5-point Likert scale; statistics as
described in the preceding paragraph.
Rater Dimension Vanilla HEG-TKG∆(Cohen’sd,p BH)
C1 (senior neurologist) D1 Verifiability 2.36 4.00+1.64 (d=1.81,p<0.001)
C1 D2 Actionability 2.75 3.61+0.86 (d=1.00,p=0.009)
C1 D3 Temporal 3.78 4.17+0.39 (d=0.45,p=0.17)
C1 D4 Safety 3.17 3.61+0.44 (d=0.57,p=0.12)
C1 D5 Completeness 3.25 3.47+0.22 (d=0.35,p=0.17)
C2 (neurologist) D1–D5 No significant effect on any dimension (allp BH>0.79,|d|<0.31)
The senior neurologist (C1) found HEG-TKG markedly more verifiable ( +1.64/5, d=1.81 ,p<0.001 )
and more actionable ( +0.86/5, d=1.00 ,p=0.009 ) than the vanilla-GPT-4.1 baseline; trends were
positive on the remaining three dimensions but did not reach significance after BH correction.
The second neurologist (C2) found no significant differences on any dimension (all pBH>0.79 ,
|d|<0.31 ); qualitative notes reported in HEG-TKG attribute this to an evaluation philosophy that
treats citations of consensus knowledge as redundant rather than useful (e.g., penalising HEG-TKG
for citing standard-of-care treatments with PMIDs). This rater-by-rater divergence is itself the case
for the multi-rater clinician-validated subset queued for v2.0 (Limitation (ii) in main paper Section 7).
Follow-up independent replication on CIDP/GBS.Subsequent to the 2-clinician panel above,
HEG-TKG co-author J. Zschüntzsch (senior neurologist, 20 years’ experience, University Medical
Center Göttingen) performed an independent blinded evaluation of the same 12-case CIDP/GBS
reduced pack that C1 scored. Following the same protocol (single output per case, blinded to system,
shuffled), she rated 11 of 12 cases on all five dimensions (one case left unrated). Her per-dimension
stratified means are reported in Table 15, with C1’s matched CIDP/GBS subset for direct comparison.
Direction of effect replicates across raters on D1–D4: both raters find HEG-TKG higher than vanilla
GPT-4.1 on Verifiability ( ∆=+1.30 vs. C1’s ∆=+1.58 on the same 12 cases), with consistent
positive deltas on Actionability, Temporal Precision, and Safety. D5 Completeness diverges in
sign between raters (Jana ∆=+1.00 , C1 ∆=−0.08 ), reflecting small-sample variance. Sample
sizes are too small for per-rater significance testing at this subset ( n=5–6per arm); the finding is a
direction-of-effect replication, not an independent significance claim.
Scope of clinician evidence.This validation applies to the HEG-TKG pipeline on six diseases;
ChronoMedKG has not been clinician-validated at its 13,431-disease scope. The panel result is an
anchor for the underlying methodology, not a claim about the resource as a whole. Clinician rating
of novel-disease triples, in particular the 6,250 diseases where ChronoMedKG provides onset data
absent from any gold standard, is the critical remaining validation step and is in progress.
30

Table 15: Follow-up independent blinded evaluation by HEG-TKG co-author J. Zschüntzsch (UMG Göttingen) on the same
12-case CIDP/GBS reduced pack that C1 scored. n=6 vanilla, n=5 HEG-TKG for Jana (one case unrated); n=6 per arm for
C1. Values are per-dimension Likert means (1–5 scale).
Jana (follow-up) C1 (original, CIDP/GBS subset)
Dimension Vanilla HEG-TKG∆ Vanilla HEG-TKG∆
D1 Verifiability 2.50 3.80+1.30 2.08 3.67+1.58
D2 Actionability 3.00 3.80+0.80 2.58 3.50+0.92
D3 Temporal 3.83 4.40+0.57 4.00 4.50+0.50
D4 Safety 2.83 3.40+0.57 3.00 3.67+0.67
D5 Completeness 3.00 4.00+1.00 3.42 3.33−0.08
H.7 Disease Trajectory Clustering: Archetype Table and Sensitivity
We clustered 8,935 diseases with onset data by temporal signature (median onset, onset spread,
number of stages, milestone density, fraction of triples with onset). K-means identified four clinically-
coherent archetypes (silhouette 0.362, best ofk∈ {4..8}):
Table 16: Disease trajectory archetypes discovered by unsupervised clustering.Median onsetis the per-disease median of the
earliest-onset triple;median spreadis the per-disease maximum minus minimum onset across all phenotype triples for that
disease. The “broad-onset progressive” archetype has a childhood-centred median but a wide spread because these diseases
accumulate phenotypes spanning the full life course (e.g., muscular dystrophies with both paediatric motor signs and late-adult
cardiorespiratory manifestations).
Archetype N Median onset Median spread Example diseases
Congenital/neonatal 3,573 0.5y 12y Pierre–Robin, gastroschisis
Broad-onset progressive 4,334 11y 78y Duchenne MD, neurodevelopmental disorders
Late-onset 900 54.5y 20y ALS, prostate cancer, acute coronary syndrome
Early childhood 128 2y 10y BPES, O’Donnell–Luria–Rodan syndrome
Figure 6: Disease trajectory archetypes discovered by unsupervised clustering of 8,935 diseases with temporal features. Four
clinically-coherent clusters emerge: congenital/neonatal, broad-onset progressive, early childhood, and late-onset.
Sensitivity to clustering choice.We also ran HDBSCAN on a 4-feature subset (median onset, onset
spread, number of stages, total triples) with min_cluster_size∈ {50,100,200,500,1000} . Small
values recover 7–11 clusters but assign 74–79% of diseases to noise (silhouette of non-noise points
0.41–0.47); large values degenerate to 2 clusters (silhouette 0.71 but structurally uninformative) or
return no clusters at all. K-means k=4 recovers a comparable non-noise silhouette (0.42 on the
4-feature subset) while assigning every disease to an archetype, which is the behaviour required for
the coverage-gap and trajectory analyses. The K-means solution is therefore a robustness-preserving
reporting choice, not a cherry-picked algorithm.
H.8 Evidence Decay: Full Age Distribution
PrimeKG has 4.05M edges andzeroevidence dates at the edge level. Each edge is a database snapshot
with no way to determine when a fact was established, what evidence supports it, or whether newer
findings have refined or contradicted it. A clinician querying PrimeKG for “DMD→corticosteroid
31

treatment” cannot distinguish a 1995 consensus from a 2024 guideline update. ChronoMedKG, by
contrast, has 455,519 triples (98.9% of validated) with PMID-traceable publication dates.
The evidence age distribution reveals that ChronoMedKG integrates knowledge spanning six decades
of biomedical research: 27.4% of triples cite evidence from the last 5 years (2021–2026), reflecting
recent discoveries and guideline updates, while 24.1% cite evidence older than 20 years, capturing
established clinical knowledge. The median evidence year is 2015. This temporal provenance enables
evaluative questions that static KGs cannot support: “Is this association based on recent evidence?”
“Has this treatment recommendation been updated?” “What proportion of knowledge about disease X
predates the genomic era?”
Figure 7: Evidence publication year distribution across 455K ChronoMedKG triples with PMID-traceable dates. Median year:
2015. PrimeKG, by contrast, has zero evidence dates at the edge level, leaving no way to assess recency, supersession, or
evidence evolution.
I Extended Ablations
I.1 Link Prediction with Temporal Features: Setup and Full Result
To test whether ChronoMedKG’s temporal annotations provide useful signal for standard KG
evaluation, we trained TransE [ 3] via PyKEEN [ 2] on disease–phenotype link prediction un-
der four conditions, restricted to the 1,302 diseases present in both HPOA and ChronoMedKG
for fair comparison: (1)HPOA-struct: plain (d,has_phenotype, p) triples; (2)HPOA-
temporal: (d,has_phenotype_onset<bin>, p) with HPOA’s coarse onset categories (disease-level);
(3)ChronoMedKG-struct: plain ChronoMedKG disease–phenotype triples; (4)ChronoMedKG-
temporal: ChronoMedKG triples with fine-grained per-phenotype onset bins. Each condition was
trained with 3 random seeds (embedding dim 100, 100 epochs, Adam lr=0.01); RotatE [ 19] and
ComplEx were tested but failed on Apple Silicon MPS, see Appendix I.4.
Table 17: KG link prediction on disease–phenotype edges (TransE, 3 seeds). Temporal features significantly improve MRR in
both HPOA and ChronoMedKG, with ChronoMedKG’s fine-grained annotations providing larger relative gains than HPOA’s
coarse bins.
Condition MRR (mean±std) Hits@1 Hits@10 Gain
HPOA-struct0.0224±0.00150.0040.0563±0.006—
HPOA-temporal (coarse)0.0389±0.00010.0060.0961±0.002+73.6%(p=0.003)
ChronoMedKG-struct0.0101±0.00030.0010.0244±0.002—
ChronoMedKG-temporal (fine)0.0192±0.00150.0020.0473±0.004+89.8%(p=0.015)
Temporal features significantly improve MRR in both systems (paired t-test across 3 random seeds
per condition). The remaining subsections cover scope of claims (§I.2), the MPS-related model
coverage (§I.4), and the bin-granularity sensitivity sweep (§I.5).
I.2 Scope of the Link-Prediction Experiment: Controlled Ablation, Not SOTA Claim
The link-prediction results in Appendix I.1 are a controlled ablation of temporal features, not a
claim that ChronoMedKG outperforms state-of-the-art biomedical KG embeddings. The absolute
MRR values (ChronoMedKG-temporal: 0.0192; HPOA-temporal: 0.0389) are low in absolute terms
because the experiment deliberately restricts to the intersection of HPOA and ChronoMedKG’s
disease sets (1,302 diseases) and uses the raw LLM-extracted phenotype vocabulary without entity
canonicalisation, making ChronoMedKG’s entity space roughly 3×larger than HPOA’s for the same
disease set. A larger entity space makes link prediction harder because the denominator of the rank
computation is larger.
32

struct temporal
(coarse)struct temporal
(fine)0.000.010.020.030.040.050.06Mean reciprocal rank (MRR)0.02240.0389
0.01010.0192+73.6%  (p=0.003)+89.8%  (p=0.015)
HPOA ChronoMedKGFigure 8: Mean reciprocal rank for TransE link prediction on disease–phenotype edges (3 random seeds; error bars are std).
Temporal features lift MRR over the matched structural baseline in both HPOA ( +73.6%, p=0.003 ) and ChronoMedKG
(+89.8%,p=0.015); the larger relative gain in ChronoMedKG sits on a 3×harder entity space.
Entity-space expansion is a handicap, not an excuse.The 3×entity space makes ChronoMedKG’s
TransE task strictly harder than HPOA’s on the same disease set. It makes the +89.8% relative gain a
harder threshold to clear, not an easier one. ChronoMedKG-temporal must beat ChronoMedKG-struct
on a larger, noisier vocabulary than HPOA does.
Absolute MRR is within the biomedical KG range.Absolute MRR in the 0.01–0.05 range is
consistent with other biomedical KG link-prediction studies at similar entity-space sizes (e.g., Hetionet
drug-repurposing benchmarks; default PyKEEN TransE training on PrimeKG). A reader unfamiliar
with the biomedical KG literature may find 0.0192 surprisingly low; it is not anomalous for this
vocabulary scale.
Relative gain replicates across seeds and metrics.The +89.8% gain is reproducible across 3
random seeds (paired t-test,p=0.015 ) and holds across MRR, Hits@1, and Hits@10 independently
(Table 17). A finding that replicates across three metrics and three seeds is robust regardless of
absolute magnitude.
Entity canonicalisation via SapBERT/UMLS (Limitation (i) in main paper Section 7) would compress
the ChronoMedKG vocabulary and likely raise absolute MRR substantially; that work is out of scope
for v1 and deferred to future work.
I.3 Consensus Threshold Sensitivity
The default consensus threshold is 2 (at least 2 distinct LLMs must agree). We also analyzed results
at thresholds 1 (any single model), 2 (default), and 3 (strict).
Note on multi-model agreement: 99.4% of validated triples have an extraction_models field
containing one model name, while 87.1% have consensus_confidence = 1.00 (full agree-
ment across all models that processed the document). The two fields play different roles:
extraction_models records the model whose surviving extraction was kept as the representa-
tive row, and consensus_confidence records how many independent models agreed on it. The
consensus filter uses the latter.
I.4 Link Prediction: Multiple Models (TransE)
We tested RotatE and ComplEx as additional baselines but both failed due to PyKEEN/MPS com-
patibility issues on Apple Silicon (“norm ops not supported for complex y”). CPU-only training for
complex-valued models is slower and was not pursued at scale. TransE (real-valued) ran successfully
on MPS with the hardware at hand.
33

I.5 Bin-Granularity Sensitivity for Link Prediction
To disentangle “temporal bins help” from “fine-grained temporal bins help”, we collapsed
ChronoMedKG’s eight numeric-derived bins (neonatal, infantile, early childhood, childhood, juvenile,
young-adult, adult, late-onset) into five HPOA-style categories by pairwise merge (antenatal-infantile,
childhood, juvenile, adult, late-onset) and retrained under identical conditions.
Table 18: Bin-granularity sensitivity on ChronoMedKG disease–phenotype link prediction (3 seeds per cell).Coarse= 5
HPOA-style bins;fine= 8 numeric-derived bins. ∆= relative MRR gain over the no-temporal ta_struct baseline (TransE
0.0105±0.0005) for TransE rows; pairedt-test rightmost column compares fine vs coarse within-scorer.
Scorer Bins MRR Hits@10∆struct fine vs coarse
TransE coarse (5)0.0177±0.0003 0.0445±0.0006 +68.4% —
TransE fine (8)0.0182±0.0007 0.0440±0.0033 +72.9%+2.7%,p=0.42
DistMult coarse (5)0.0133±0.0011 0.0258±0.0024— —
DistMult fine (8)0.0157±0.0022 0.0327±0.0043—+18.2%
The sensitivity result is a negative finding forfinegranularity specifically but a strongly positive one
for temporal metadata overall. On TransE, coarse 5-bin ChronoMedKG and fine 8-bin ChronoMedKG
are indistinguishable ( +2.7% mean, paired t=1.02 ,p=0.42 ); the +72.9% lift over ta_struct is
already captured by five bins. On DistMult, fine 8-bin shows a +18.2% advantage over 5-bin,
suggesting that the utility of finer bins depends on scorer expressivity. The load-bearing property
is that ChronoMedKG encodes per-phenotype onset windows at all. Entity-name canonicalisation
(Limitation (i) in main text Section 7) is the more pressing remaining gap: with a 3 ×smaller entity
space, fine-grained bins may recover a larger signal than we observed here.
34

J Data, Code, and Reproducibility
J.1 Dataset Availability
•ChronoMedKG dataset(460,497 validated triples derived from 13,431 PrimeKG diseases),
organised as three tiers matching the construction pipeline:
– Concept DOI(always latest):https://doi.org/10.5281/zenodo.19697542
–Version DOI(v0.0.1 reviewed here): https://doi.org/10.5281/zenodo.
19697543
–Gold tier:validated_triples.jsonl(502 MB, 460,497 post-QC triples)
–Silver tier: consensus_triples.jsonl.gz (30 MB, 443,114 pre-QC multi-LLM
consensus rows)
–Bronze tier: raw_triples.jsonl.gz (644 MB, 13,001,651 raw LLM extractions
across four models)
–Silver and Bronze rows carry adisease_mondofield so each row is self-locating
–Hugging Face mirror with datasets.load_dataset() API: deferred to camera-
ready; during review, reviewers download directly from Zenodo
–License: CC BY 4.0
–Format: JSONL (one triple per line); Silver/Bronze tiers gzipped
–Croissant 1.0 metadata (8 RAI fields) shipped alongside the data for mlcommons
tooling
–README includes aKnown Discrepanciessection documenting audit findings, all
under 1% of their respective denominators: 4,331 validated rows in the flat file that do
not appear in the per-disease tree (0.94% of 460,497), 54 diseases with validated rows
that bypassed the consensus intermediate (0.50% of 10,852), 44,036-row drift between
pre- and post-merge raw counts (0.34% of 13M)
•ChronoTQA benchmark(3,341 questions across 9 task types):
–Co-released at the same Zenodo deposit (tqa_benchmark.json, 3 MB)
–JSON format with answers and gold-standard metadata (Tier 1 = external source, Tier
2 = ChronoMedKG-derived with PMID trace)
•PMC clinical cases(31 curated cases):
–Shipped in the Zenodo deposit aspmc_clinical_cases.json(63 KB)
–Each case is a JSON record linked to its PMC URL (all open-access)
J.2 Code Availability
•Pipeline code:https://gitlab.sdu.dk/screen4care/chronomedkg
•License:MIT
•Language:Python 3.12
•Key dependencies:OpenAI, Anthropic, Google-GenerativeAI, DeepSeek SDKs; PyKEEN
1.11; scikit-learn; rapidfuzz; pandas
•Reproduce link prediction: .venv-sapbert/bin/python
scripts/experiment_link_prediction_v3.py
•Reproduce all experiments:seescripts/experiment_*.py
J.3 Compute Requirements
•Pipeline extraction(one-time):∼$2,400 LLM API costs,∼100 CPU-hours
•Link prediction experiment: 3 minutes on Apple M4 MPS, no GPU required
•Trajectory clustering: 2 minutes on CPU (sklearn K-Means + TSNE)
•Evidence decay analysis: 5 minutes (single-pass over JSONL files)
J.4 Reproducibility Statement
All experiments use fixed random seeds (42, 7, 123 for link prediction; 42 for clustering). Exact
package versions are in requirements.txt . The decision log at docs/decision_log.md tracks
all methodological changes during development.
K Ethics Statement
Data provenance.ChronoMedKG is constructed entirely from (i) public biomedical literature
(PubMed abstracts and PMC Open Access full-text, via NCBI E-utilities with an API key),
35

(ii) publicly-licensed ontologies and knowledge bases (PrimeKG, Orphadata, HPOA, GA4GH
Phenopackets), and (iii) LLM-generated extractions from those sources via commercial APIs (Ope-
nAI, Anthropic, Google, DeepSeek). No patient-level records, electronic health records, images, or
identifiers are used at any stage of the pipeline or included in the released resource.
Clinician evaluation.The three-clinician panel results we cite (Appendix H.6) were collected
in a prior study (HEG-TKG [ 1]) on rare neuromuscular diseases. Ratings were gathered from
three physicians (one senior neurologist, one physician-scientist, one medical-informatics specialist)
who consented to their anonymised Likert ratings being used for research purposes, including the
comparative analysis reported here. The rating task assessed system-generated literature summaries,
not patient data; no individual patient was the subject of any rating. Institutional ethics review was not
required for the clinician-rating component under the rules of the collaborating institutions because
the activity did not involve human-subjects research beyond expert opinion on synthetic outputs.
Per-rater identifiers are anonymised to C1/C2/MI and only aggregate statistics (means, effect sizes)
are reported.
Clinical case studies.The 31 diagnostic-odyssey cases cited in the main text (and listed in full in
Appendix F) are all taken from PubMed Central open-access case reports. Each case is referenced only
by its PMC identifier; no re-identification is performed, no patient-contact information is collected,
and the illustrative description in the paper consists of disease-course summaries already publicly
disclosed in the source report. No new patient recruitment occurred.
LLM use and disclosure.LLMs are used in two distinct roles in this work: as components inside
the construction pipeline, and as writing assistants during manuscript preparation. Both are disclosed
below.
Pipeline LLMs (extraction and judging).Extractions are produced by commercial LLM APIs:
DeepSeek V3 and an OpenAI primary (GPT-4.1-nano or GPT-4o-mini), with Claude 3 Haiku as a
conditional tiebreaker; the three-LLM novelty-judge panel uses DeepSeek V3, GPT-4o-mini, and
Claude Haiku 4.5. We record the model identifier, prompt, and timestamp for every extracted
triple; the representative-model field documented in these supplementary methods is paired with
consensus_confidence , which records full multi-LLM agreement. LLM output is treated as a
literature summary, not an authoritative clinical reference; 7.3% of outputs are demonstrably wrong
(main text Section 4.3), and released triples carry per-source PMIDs so users can verify every claim.
Writing-assistant LLMs (manuscript preparation).Claude (Anthropic) assisted the authors with
prose drafting and revision against author-prepared outlines and analyses, structural restructuring of
sections, and cross-checking references against their sources. Grammarly and Overleaf were used
for copyediting. AI-assisted prose was reviewed by the authors against the underlying experimental
results before inclusion. No LLM was used to generate numerical results, to perform statistical
analysis, or to make experimental-design decisions. All scientific content, experimental design, data
collection, analysis, and interpretation were performed by the authors, who take full responsibility
for the manuscript content.
Image-generation AI (figures).Figure 1 in the main paper (the four-agent pipeline schematic)
was produced with assistance from a diagram-generation AI tool; the resulting schematic was
reviewed and refined by the authors for technical accuracy before inclusion. Any schematic styling or
layout assistance in other figures has likewise been reviewed by the authors against the underlying
experimental setup. No data-bearing element in any figure was generated or modified by AI: every
quantitative figure in this paper (Figure 3 in the main text and all data plots in the supplementary) is
rendered directly from the released benchmark data via Python scripts shipped in the code repository,
with no AI image-generation tool in the rendering pipeline.
Risks and safeguards.ChronoMedKG is a literature knowledge graph, not a clinical-decision tool.
Direct downstream use in patient-facing systems without clinician oversight is explicitly not supported.
The resource has not been regulated as a medical device and must not be used as one. Downstream
users building clinical tools that incorporate ChronoMedKG are responsible for their own regulatory
clearance, clinician review, and patient-safety evaluation. The release includes the main-text error-rate
table and the novel-coverage verification protocol so users can bound expected-error budgets in their
own deployments.
Dual use.Biomedical knowledge graphs could in principle be used to target patients in harmful
ways, such as risk-based insurance discrimination by age-of-onset. ChronoMedKG contains only
36

population-level onset ranges drawn from published literature, information that is already public, and
adds no private or patient-linked information. We do not identify any dual-use risk beyond those
already present in the underlying literature.
L Reviewer Access and Release Artefacts
Data access.ChronoMedKG v0.0.1 is released open-access under CC BY 4.0 on Zenodo. The
concept DOI https://doi.org/10.5281/zenodo.19697542 always resolves to the latest ver-
sion; theversion DOI https://doi.org/10.5281/zenodo.19697543 resolves specifically to
v0.0.1 (the bytes reviewed for this submission). A Hugging Face mirror exposing the datasets
load_dataset() API will be made available at camera-ready; during review, reviewers should
download directly from Zenodo. Code (the four-agent pipeline and experiment scripts) is released
separately under the MIT License.
Release artefacts.The Zenodo deposit contains nine files, organised as three tiers match-
ing the construction pipeline: (1) validated_triples.jsonl (502 MB, 460,497 Gold-
tier post-QC triples); (2) consensus_triples.jsonl.gz (30 MB, 443,114 Silver-tier
pre-QC consensus rows); (3) raw_triples.jsonl.gz (644 MB, 13,001,651 Bronze-
tier raw LLM extractions across four models); (4) tqa_benchmark.json (3 MB, 3,341
ChronoTQA questions); (5) pmc_clinical_cases.json (63 KB, 31 diagnostic-odyssey cases);
(6)croissant.json (Croissant 1.0 metadata with the eight RAI fields: dataCollection ,
dataPreprocessingImputation ,dataUseCases ,dataLimitations ,dataSocialImpact ,
annotationsPerItem ,personalSensitiveInformation ,dataReleaseTimeline );
(7)README.md (data card with full schema, tier descriptions, and a Known Discrepancies
section); (8) LICENSE-DATA (CC BY 4.0); (9) NOTICE (attribution to upstream sources including
PrimeKG, PubMed, PMC, MONDO, OMIM, HPO).
Code and environment.Code is released under the MIT License (see LICENSE-CODE ). A pinned
requirements.txt fixes the versions of 15 load-bearing Python packages used to produce this
release (openai==2.9.0 ,anthropic==0.75.0 ,rapidfuzz==3.14.3 ,torch==2.7.0 , etc.); a
companion .venv-sapbert environment is reused unchanged from the HEG-TKG project for
SapBERT-based entity normalisation. Random seeds are fixed where applicable (link prediction:
{42,7,123}; novelty sampling:42).
37