# IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark

**Authors**: Tirana Noor Fatyanosa, Ummu Ditya Erliana, Mahendra Data, Muhammad Nurwegiono, Eleazar Tadeo Eman, Alham Fikri Aji

**Published**: 2026-10-22 13:34:29

**PDF URL**: [https://arxiv.org/pdf/2610.31256v1](https://arxiv.org/pdf/2610.31256v1)

## Abstract
Clinical Question Answering (QA) underpins Clinical Intelligent Decision Support Systems, but progress is severely limited for low-resource languages such as Indonesian by the absence of public domain-specific data. We introduce IndoNutriQA, an expert-curated Indonesian clinical QA benchmark of 911 question-answer pairs built from real clinical-nutrition reports. Rather than a large-scale training corpus, it is positioned as a high-quality evaluation resource: every pair is grounded in a clinically relevant context, enriched with structured metadata (anthropometric assessment, medical diagnosis, document provenance, chunk segmentation). Eight standardised question types following the Nutrition Care Process (NCP), including the Problem, Etiology, Sign and Symptom (PES) diagnosis statement, are answered for every report, which was written by a graduate-level dietetic intern, by one of three nutritionists. Answers are free-form clinical prose rather than extractive spans, and an expert validation study finds 6-10% of reference answers contestable. The dataset supports generative, extractive and retrieval-augmented QA as well as zero- and few-shot LLM evaluation. We report fine-tuned baselines (IndoBERT variants, mT5, RAG) with 5-fold cross-validation, learning curves and bootstrap 95% confidence intervals, and benchmark eight contemporary LLMs zero- and five-shot. We further calibrate our LLM-as-judge rubric against three expert raters (one RD, two nutritionists), finding it human-comparable on factuality and completeness but unreliable on clinical safety, and a human error analysis shows it misses unsupported content that the raters identify. IndoNutriQA is released to foster research in Indonesian clinical NLP and nutritional decision support.

## Full Text


<!-- PDF content starts -->

IndoNutriQA: An Indonesian Clinical Nutrition
Question Answering Benchmark
Tirana Noor Fatyanosa•Ummu Ditya Erliana•Mahendra Data•
Muhammad Nurwegiono◦Eleazar Tadeo Eman•Alham Fikri Aji⊗
•Universitas Brawijaya,◦Universitas Ma Chung,⊗MBZUAI
{fatyanosa, ummuditya.fk, mahendra.data}@ub.ac.id
eleazartadeo@student.ub.ac.id
muhammad.nurwegiono@machung.ac.id, alham.fikri@mbzuai.ac.ae
Abstract
Clinical Question Answering (QA) under-
pins Clinical Intelligent Decision Support Sys-
tems, but progress is severely limited for low-
resource languages such as Indonesian by
the absence of public domain-specific data.
We introduce IndoNutriQA, an expert-curated
Indonesian clinical QA benchmark of 911
question–answer pairs built from real clinical-
nutrition reports. Rather than a large-scale
training corpus, it is positioned as a high-
quality evaluation resource: every pair is
grounded in a clinically relevant context, en-
riched with structured metadata (anthropomet-
ric assessment, medical diagnosis, document
provenance, chunk segmentation). Eight stan-
dardised question types following the Nutrition
Care Process (NCP), including the Problem,
Etiology, Sign and Symptom (PES) diagnosis
statement, are answered for every report, which
was written by a graduate-level dietetic intern,
by one of three nutritionists. Answers are free-
form clinical prose rather than extractive spans,
and an expert validation study finds 6–10%
of reference answers contestable. The dataset
supports generative, extractive and retrieval-
augmented QA as well as zero- and few-shot
LLM evaluation. We report fine-tuned base-
lines (IndoBERT variants, mT5, RAG) with 5-
fold cross-validation, learning curves and boot-
strap95% confidence intervals, and benchmark
eight contemporary LLMs zero- and five-shot.
We further calibrate our LLM-as-judge rubric
against three expert raters (one RD, two nutri-
tionists), finding it human-comparable on fac-
tuality and completeness but unreliable on clin-
ical safety, and a human error analysis shows it
misses unsupported content that the raters iden-
tify. IndoNutriQA is released to foster research
in Indonesian clinical NLP and nutritional de-
cision support.1
1Data and code: https://github.com/NLP-UB/
IndoNutriQA1 Introduction
Clinical Question Answering (QA) is a core com-
ponent of Clinical Intelligent Decision Support Sys-
tems (CIDSS), enabling efficient access to relevant
medical information from unstructured clinical text
(Demner-Fushman and Lin, 2007; Mutabazi et al.,
2021). Recent progress in QA has been driven by
large-scale benchmarks in both open-domain and
biomedical settings, including SQuAD (Rajpurkar
et al., 2016), Natural Questions (Kwiatkowski et al.,
2019), TyDiQA (Clark et al., 2020), as well as
domain-specific resources such as BioASQ (Tsat-
saronis et al., 2015), PubMedQA (Jin et al., 2019),
and MedQA (Jin et al., 2021). However, existing
clinical QA benchmarks degrade when applied to
low-resource languages, nutrition subdomains, and
localized clinical practices (Clark et al., 2020; Jin
et al., 2019).
In Indonesian healthcare contexts, clinical nutri-
tion reports play an important role in medical de-
cision making, integrating anthropometric assess-
ments, medical diagnoses, and narrative clinical
information. Supporting QA over such documents
requires datasets that capture domain-specific ter-
minology and reasoning patterns. However, to
date, no publicly available Indonesian clinical QA
dataset has been specifically designed for nutri-
tional and medical decision support.
To address this gap, IndoNutriQA is introduced,
a novel Indonesian clinical QA dataset constructed
from real-world clinical nutrition reports. In-
doNutriQA consists of expert-authored question-
answer pairs grounded in clinically relevant con-
texts and enriched with structured metadata, in-
cluding document provenance, chunk indices, an-
thropometric assessments, and medical diagnoses.
The dataset supports extractive, generative, and
retrieval-augmented QA paradigms.
Baseline experiments with state-of-the-art neural
QA models establish reference performance and
arXiv:2610.31256v1  [cs.CL]  22 Oct 2026

expose challenges specific to Indonesian nutrition-
focused clinical QA, such as domain terminology
and numerical reasoning over anthropometric data.
We deliberately position IndoNutriQA as an eval-
uation benchmark rather than a training corpus:
annotating clinical text with experts is intrinsically
expensive, so a small, expert-authored test set is
the operative bottleneck for measuring progress
here. The protocol of Section 6 makes this concrete
through cross-validation with bootstrap confidence
intervals, a learning curve over training-split frac-
tions, and zero- and few-shot LLM evaluation that
needs no in-domain training data. Our contribu-
tions are: (i)the first public Indonesian clinical-
nutrition QA benchmark, 911pairs over 120real
multi-hospital reports, anonymised at patient and
institution level; (ii)an expert-authored anno-
tation designapplying eight standardised ques-
tions from the Nutrition Care Process (NCP) and its
Problem, Etiology, Sign and Symptom (PES) state-
ment format uniformly to every report, each answer
written by one of three nutritionists from reports
authored by graduate-level dietetic interns; (iii)
reference results for four model families (one
RAG configuration); (iv)a five-axis evaluation
protocol(lexical, semantic, LLM-judge, RAGAS,
stratified) with cross-validation, learning curves
and bootstrap CIs; and (v)a validation study by
three expert raters (one Registered Dietitian,
two nutritionists)covering the LLM-judge rubric,
reference-answer errors and a human error analysis
(Section 6.8, Appendix E). IndoNutriQA supports
future research in Indonesian clinical NLP, nutri-
tion informatics, and AI-driven clinical decision
support.
2 Related Work
General-purpose QA benchmarks such as
SQuAD (Rajpurkar et al., 2016), Natural
Questions (Kwiatkowski et al., 2019), and the
multilingual TyDi QA (Clark et al., 2020) (which
includes Indonesian) have driven progress on
machine reading comprehension but do not target
clinical reasoning. Clinical/biomedical QA re-
sources include BioASQ (Tsatsaronis et al., 2015)
(biomedical semantic indexing over PubMed), Pub-
MedQA (Jin et al., 2019) (yes/no/maybe questions
from biomedical literature), and MedQA (Jin et al.,
2021) (medical licensing-exam multiple choice),
all of which are English-language and do not
target nutrition-focused reasoning over real clinicaldocumentation. Pretrained English biomedical
and clinical encoders such as BioBERT (Lee
et al., 2020) and ClinicalBERT (Alsentzer et al.,
2019) support such tasks, but we are not aware
of comparable Indonesian clinical encoders.
Indonesian NLP has matured rapidly through
resources such as IndoBERT (Wilie et al., 2020),
IndoBART (Cahyawijaya et al., 2021), and the
IndoLEM benchmark (Koto et al., 2020), but no
publicly available Indonesian clinical or nutrition
QA dataset existed prior to IndoNutriQA; prior
Indonesian medical-NLP work (Aji et al., 2021;
Vihikan and Trisna, 2024) has been limited to
general medical-term extraction and classification
rather than question answering.
These resources differ in how answers are
produced, which matters more than raw size.
BioASQ (Tsatsaronis et al., 2015) answers are
written by biomedical experts; PubMedQA (Jin
et al., 2019) derives yes/no/maybe labels from ab-
stract conclusions, only 1K of its 273K instances
being expert-labelled; MedQA (Jin et al., 2021)
and MedMCQA (Pal et al., 2022) inherit certi-
fied answer keys from licensing exams, but the
reasoning is exam-style; MedQuAD (Abacha and
Demner-Fushman, 2019) harvests answers from
curated NIH pages; and emrQA (Pampari et al.,
2018) template-generates 400K+ pairs over expert-
annotated electronic health records (EHRs), mak-
ing it large but not individually expert-authored.
IndoNutriQA sits at the small-and-fully-curated
end: each of its 911answers is written by one of
three nutritionists from real reports authored by
dietitians-in-training.
QA resources aimed at nutritional or dietary de-
cision support remain scarce: nutrition appears
in general medical benchmarks only incidentally,
and we are aware of no public QA dataset built
around the NCP, anthropometric and biochemical
assessment, or diet-prescription reasoning in any
language. Outside English the largest resources
are Chinese (the MCMLE and TWMLE portions
of MedQA (Jin et al., 2021) contribute roughly
34K and 14K licensing items), while multilingual
open-domain QA such as TyDi QA (Clark et al.,
2020) includes Indonesian but not clinical text. In-
donesian medical NLP has so far been limited to
term extraction and classification (Aji et al., 2021;
Vihikan and Trisna, 2024).
Existing benchmarks each satisfy some of the
properties IndoNutriQA targets, but none satisfies
all three at once. TyDi QA (Clark et al., 2020)

covers Indonesian but is open-domain, while ev-
ery clinical resource above is English or Chinese.
No benchmark in any language targets clinical
nutrition, despite nutrition being the locus of a
distinct reasoning pattern (anthropometric assess-
ment, biochemical interpretation, and prescription
arithmetic) that general medical QA does not ex-
ercise. MedQA (Jin et al., 2021) and MedM-
CQA (Pal et al., 2022) are drawn from licensing
exams and PubMedQA (Jin et al., 2019) from liter-
ature abstracts, so neither requires reasoning over
the messy, semi-structured documentation clini-
cians actually produce; emrQA (Pampari et al.,
2018) is grounded in real EHRs but is English
and template-generated rather than expert-authored.
IndoNutriQA is the first resource to occupy the
intersection: Indonesian, nutrition-focused, and
grounded in authentic clinical reports with every an-
swer written by a nutritionist following NCP/PES
documentation guidelines (Table 6).
A second distinction concerns the unit of an-
notation. The benchmarks above pair free-form
questions with answers, so question phrasing and
answer content vary together. IndoNutriQA in-
stead applies eight standardised questions, drawn
directly from the NCP and PES reporting frame-
works, uniformly to every patient report. This fixes
the question side and confines annotator variation
to answer text, which is what makes a 911-pair
corpus usable as a controlled evaluation instrument
rather than merely a small training set.
3 Dataset Construction
3.1 Data Sources
The IndoNutriQA dataset comprises clinical
reports and nutritional assessment documents
sourced from diverse healthcare facilities across
Indonesia. This multi-institution sampling ensures
broad coverage of varied hospital settings, anthro-
pometric profiles, medical diagnoses, and nutri-
tional conditions.
All data were de-identified at both patient and
institution level before annotation. Indonesian hos-
pitals are not subject to the US Health Insurance
Portability and Accountability Act (HIPAA), but
we adopt its Safe Harbor identifier list as a vol-
untary de-identification standard: every category
of Protected Health Information (PHI) it enumer-
ates, namely names, medical-record and registra-
tion numbers, geographic detail, and all dates, is
removed and replaced with deterministic place-holders, while the clinically and nutritionally rel-
evant content required for question answering is
preserved. The procedure and placeholder scheme
are detailed in the Ethics Statement. The source
materials are patient assessment reports, nutritional
consultation records, and clinical follow-up notes.
The dataset development was carried out in close
collaboration with clinical nutrition experts, who
provided guidance throughout the data collection,
annotation, and validation processes. The docu-
ments were derived from complete clinical rotation
reports prepared by graduate-level dietetic interns
during clinical nutrition rotations. These reports
include detailed case descriptions, anthropomet-
ric measurements, biochemical parameters, and
dietary management plans, which served as the
foundation for generating clinically grounded ques-
tion–answer pairs. The question–answer pairs were
written by three nutritionists, and a stratified sam-
ple was later validated by one Registered Dietitian
(RD) and two of those nutritionists (Section 4).
Unlike fully synthetic datasets, IndoNutriQA is
grounded in authentic clinical documentation, with
answers written by nutritionists. This approach
ensures that the dataset reflects the complexity,
variability, and contextual reasoning of real-world
clinical nutrition practice, providing a reliable foun-
dation for research in Indonesian clinical question
answering and medical decision support.
3.2 Dataset Schema and Pipeline
Each entry in IndoNutriQA contains: a unique
id, the source report_title andfile ,
the page_number and chunk_index
that locate the answer in the report, struc-
tured anthropometric_assessment and
medical_diagnosis metadata, the context
(the chunk text), the question , and the ground-
truth answer . An example instance appears in
Appendix J.
Although IndoNutriQA uses a SQuAD-style
container with an answer_start offset, its an-
swers are abstractive: annotators authored clinical
prose rather than selecting text from the chunk.
Only 8.7% of answers occur verbatim in their con-
text (31.2% ignoring all whitespace); the median
answer is 30words and 38% exceed 50. The
answer_start field marks the best-matching
passage (Appendix A), not a verbatim span. This
is the main reason the extractive baselines trail the
prompted LLMs (Section 6.2): for most items no
exactly correct span exists.

1. Screening method 5. Dietary goals
2. Nutritional status 6. Dietary principles
3. Abnormal biochemistry 7. Dietary requirements
4. PES nutrition diagnosis 8. Dietary guidelines
Table 1: The eight standardised questions, applied uni-
formly to every source report; each maps onto one NCP /
PES section, which bounds the acceptable answer space.
Verbatim Indonesian wording is in Appendix I.
The construction pipeline has five stages:
(i) collection of clinical-nutrition reports in
PDF/plain-text Indonesian from collaborating hos-
pitals; (ii) personally identifiable information (PII)
anonymization and chunking; (iii) expert anno-
tation of anthropometric and diagnostic meta-
data; (iv) QA construction by authoring answers
to a fixed set of eight standardized clinical-
nutrition questions (Appendix I); and (v) algorith-
mic context-pairing that maps each authored an-
swer to its source chunk (full annotation, valida-
tion, and matching details are in Appendix A). The
eight standardized questions are applied uniformly
to every patient report, so the unit of annotation
is (report, answer-to-fixed-question) rather than
free-form (question, answer); this constrains inter-
annotator variation to answer text rather than ques-
tion phrasing. The fixed question set measures com-
petence on eight NCP/PES steps, not open-ended
clinical QA (see Limitations).
4 Annotation Protocol and Agreement
IndoNutriQA’s reference answers are written by
three nutritionists, one per report, rather than by a
larger panel with parallel annotation. That design
decision governs how the benchmark should be
read, so we report the protocol and its agreement
properties in the body; Appendix A adds procedu-
ral detail.
The120 reports were written by 89distinct
graduate-level dietetic interns ( 67contributed one
report, 13two, and 9three) as clinical rotation de-
liverables of a dietetic internship, each supervised
by a lecturer of the Faculty of Health Sciences, Uni-
versitas Brawijaya. The eight answers of Table 1
for every report were then written by three nutri-
tionists, each covering a different set of reports, so
every (report, question) pair has exactly one au-
thor. Answers are free text (median 30words) and
follow the NCP/PES documentation standard as
annotation guideline. A stratified 50-unit sample
of model outputs was later re-scored by three ex-pert raters, one RD who took no part in writing the
answers and two of the three nutritionists, for the
calibration study of Appendix E.
Three mechanisms limit variance, and one lim-
itation remains. (i) The question side is fixed: all
three nutritionists answer the same eight questions
(Table 1) for every report, so no question-authoring
variance exists. (ii) The rubric is external to the
dataset: each question maps one-to-one onto an
NCP section, and Question 4 must be written in
PES form with its three components in the pre-
scribed order, so annotators work to a national
clinical-documentation standard that predates this
dataset rather than to a guideline we authored. (iii)
The author pool is small, so there are only three
writing styles to account for. The limitation is that
each nutritionist wrote the answers for different
reports, so author effects cannot be separated from
report effects.
Each report was answered by a single nutrition-
ist, so the same (report, question) pair is never anno-
tated in parallel and a conventional inter-annotator
κover the references is not defined for this release.
We state this plainly because the alternative, report-
ing the rater agreement figures below as though
they were dataset κ, would misrepresent what was
measured. Three agreement-relevant quantities are
available.
(a) Gold-answer error rate. The three raters
could flag any audited unit whose reference an-
swer they judged incorrect or contestable. Three of
50units were flagged ( 6%), one by two raters inde-
pendently; counting units where a rater criticised
the reference in free text without ticking the flag
raises this to five ( 10%). The disputes are concrete
rather than matters of clinical taste: macronutrient
percentages not summing to 100% , a mid-upper-
arm-circumference cut-off, a neonatal protein pre-
scription. This is a direct estimate of reference
error and the figure users of the benchmark should
carry: on this sample roughly one item in ten is
contestable. The flagged items are released so they
can be excluded or re-adjudicated.
(b) Expert-to-expert agreement on identical
items. The same three raters re-scored all 50units
on a1–5rubric. Quadratic-weighted Cohen’s κ
for the reference rater pair is 0.780 on factuality
and0.752 on completeness (Krippendorff ordinal
α= 0.592 and0.712 across all three raters), but
only0.160 on clinical safety ( α= 0.250 ), where
the three raters’ mean scores span 1.72 rubric
points. Expert agreement in this domain is there-

fore substantial on verifiable content and effectively
absent on safety judgement. That is a ceiling on
what any annotation protocol or automatic metric
can be expected to reproduce, and it is why we re-
port the safety dimension as unvalidated throughout
(Section 6.8, Appendix E).
(c) Independence. Two of the three raters also
wrote references, so the audit of reference answers
is not fully independent: only the RD took no part
in writing them. We therefore read the 6–10% as
an estimate by raters who are not all external to the
dataset.
Thecontext field for each (report, ques-
tion, answer) triple is algorithmically located
from the source report after answer author-
ing using a six-stage match (exact, case-
insensitive, whitespace-normalized, punctuation-
stripped, fragment-matched) on the chunked
report. We obtain a verified character-level
answer_start for823of911pairs ( 90.3% );
the remaining 88pairs are released with a chunk-
level pointer but without a character offset and are
flagged in the released metadata. Full details (per-
pair validation procedure, subjectivity mitigation,
and the six-stage matching algorithm) are in Ap-
pendix A.
5 Dataset Characteristics
5.1 Dataset Statistics and Domain Coverage
IndoNutriQA contains 911question–answer pairs
from 120clinical-nutrition reports, covering 128
unique medical diagnoses. Questions average
8.32 tokens, answers 70.66 tokens, and contexts
131.28 tokens, with 2.60 chunks per report (Ta-
ble 3, Appendix A.1). The reports span the dis-
ease categories common in Indonesian inpatient
nutrition practice. Each report is assigned one
dominant category, so the distribution partitions
the corpus: respiratory is largest at 28.33% of the
120reports, followed by infectious ( 17.50% ), gas-
trointestinal ( 12.50% ), nutrition/growth ( 10.83% ),
neurologic ( 10.00% ), and cardiac and hemato-
logic/oncologic ( 8.33% each), with renal/urologic
and ENT each below 1%(Table 18). The per-pair
distribution tracks the per-report one closely (respi-
ratory 29.97% , infectious 17.23% ), so no category
is inflated by reports contributing unusually many
pairs. Respiratory prominence matches Indonesian
hospital epidemiology and its known co-occurrence
with nutritional risk (Kementerian Kesehatan Re-
publik Indonesia, 2019; Oktaria et al., 2017). Themix is naturally imbalanced rather than syntheti-
cally balanced, but no category exceeds 30%; Sec-
tion 6.7 tests whether the imbalance harms general-
isation, and Appendix H gives the treemap.
5.2 Demographics, Anthropometrics, and
Syntactic Divergence
The population profile is clinically realistic: age
concentrates in early childhood but extends into
adolescence and adulthood, the sex split is 58.4%
male / 41.6% female, weight-vs-height shows the
expected positive correlation with wider dispersion
at older ages, and mid-upper arm circumference
clusters around screening thresholds. Syntactic di-
vergence between predicted and reference answers
(token-level normalised edit distance) is bimodal,
with one mode near zero for near-verbatim cor-
rect predictions and a second spike at maximum
divergence for empty or unrelated outputs, which
motivates the multi-metric protocol of Section 6.
Per-axis figures are in Appendix H.
5.3 Comparison with Existing Medical QA
Datasets
IndoNutriQA differs from existing medical QA
benchmarks on three axes at once: Indonesian
language, clinical-nutrition domain, and ground-
ing in authentic clinical reports rather than exam
questions or literature abstracts (Section 2, Ta-
ble 6). Appendix Table 7 compares it side by
side with MedQA, MedMCQA, PubMedQA, em-
rQA, MedQuAD, and MedExQA across language,
domain, sample count, question and answer type,
source, and length. Its 911pairs are smaller than
the largest English benchmarks (emrQA exceeds
400K) but are written by nutritionists from authen-
tic Indonesian inpatient nutrition reports.
6 Experiments
We evaluate IndoNutriQA along five axes, each
with its own subsection below: aggregate model
performance (Section 6.2), a dedicated LLM eval-
uation (Section 6.3), statistical robustness (Sec-
tion 6.4), semantic and clinical-rubric metrics be-
yond lexical overlap (Section 6.5), and stratified
performance (Section 6.7), with a RAG-specific
diagnostic in Section 6.6. Bulky tables and method
details are deferred to Appendices B, D, E and G.
6.1 Experimental Setup
We evaluate four categories of models in our bench-
mark experiments:

•Extractive QA models based on IndoBERT
variants, which predict answer spans directly
from the given clinical context.
•Generative QA models based on mT5, which
generate free-form answers conditioned on
the input question and contextual information.
•Retrieval-Augmented QA (RAG) models,
which first retrieve relevant clinical context
from a pool of document chunks and then gen-
erate answers using a generative model, en-
abling reasoning over broader and potentially
incomplete contexts.
•Large Language Models (LLMs), evaluated
in zero-shot and five-shot settings without
task-specific fine-tuning, covering closed-
source instruction-tuned models (GPT-4o-
mini, GPT-4o), open-source multilingual
models (Llama-3.1-8B-Instruct, Qwen2.5-7B-
Instruct), Indonesian-tuned models (Sahabat-
AI-Llama3-8B, Merak-7B-v4), and English-
medical LLMs (BioMistral-7B, MedGemma-
4B-IT) included to quantify cross-lingual
transfer to Indonesian clinical text.
The dataset is split deterministically into
70/15/15 train/validation/test ( 637/137/137pairs);
pairs are ordered by source report so the train/test
boundary has zero cross-report leakage, confirmed
by the GroupKFold variant in Section 6.4. The
validation set is used exclusively for model selec-
tion during fine-tuning, while the test set is reserved
for final evaluation. Fine-tuning is enabled for the
extractive, generative, and RAG models using iden-
tical data splits to ensure a fair comparison. LLMs
are evaluated on the same test split without any
parameter updates, isolating their off-the-shelf clin-
ical reasoning ability on Indonesian text.
For extractive models, the maximum input se-
quence length is set to 512 tokens with a sliding
window strategy to handle long clinical contexts.
For the generative model, the input is formatted as
a structured prompt consisting of the question fol-
lowed by its clinical context. For LLMs, all models
are queried with an identical Indonesian-language
system prompt instructing the model to act as a clin-
ical nutrition assistant and to answer strictly from
the provided context. Decoding uses temperature
0with a maximum of 256new tokens. Few-shot
exemplars are sampled deterministically from the
training split with a fixed seed, ensuring that all
LLMs see identical demonstration examples.Because the dataset contains 911expert-curated
QA pairs, the held-out test split alone ( n=
137) is small enough that a single point esti-
mate could be misleading. We therefore comple-
ment the headline test-split numbers with three
robustness procedures. First, the four fine-tuned
models of Table 5 (Rifky_IndoBERT_QA, In-
doLEM_IndoBERT, Cahya_IndoBERT, and mT5)
are additionally evaluated under 5-fold cross-
validation stratified by source report, so that no
chunk from a given clinical document appears in
both training and evaluation folds; we report the
per-fold mean and standard deviation alongside the
headline test scores. Second, a learning-curve anal-
ysis re-trains the same four models on 10%,25%,
50%, and 100% of the training split and evaluates
each model on the same fixed test set; this quanti-
fies how much additional expert annotation would
be expected to improve performance. Third, we
report bootstrap 95% confidence intervals for every
metric in every results table, computed with 1,000
resamples drawn with replacement from the test set.
LLM zero-shot and few-shot evaluations are point
estimates over the same test split (no parameter
updates), but bootstrap confidence intervals over
the same 1,000 resamples are reported alongside
their scores so that all rows in our results tables are
directly comparable in terms of statistical uncer-
tainty.
6.2 Model Performance
Table 2 reports all evaluated models on the test
set, as percentages: extractive QA over IndoBERT
variants, generative QA over mT5, and a retrieval-
augmented (RAG) system.
The extractive baselines are span-prediction
heads fine-tuned on our training split.
Rifky_IndoBERT_QA( Rifky/Indobert-QA )
is an IndoBERT-base checkpoint already fine-tuned
for extractive QA on Indonesian SQuAD-style
data before we fine-tune it further, which likely
explains why it leads the extractive group: it
starts from in-language QA supervision rather
than a plain LM initialisation.Cahya_IndoBERT
andIndoLEM_IndoBERTare general-purpose
Indonesian BERT checkpoints without prior QA
tuning;XLM-R-largeandmDeBERTa-v3-baseare
multilingual encoders testing whether multilingual
scale substitutes for Indonesian pretraining. The
generative baseline is mt5-base , reused as the
RAG generator. Among the extractive approaches,
Rifky_IndoBERT_QAachieves the best overall

Model Family Shots EM F1 BLEU R-1 R-2 R-L
Fine-tuned baselines (this work)
Rifky_IndoBERT_QA IndoBERT (extr.) – 12.41 36.34 12.93 37.28 34.31 36.72
IndoLEM_IndoBERT IndoBERT (extr.) – 5.84 23.78 10.60 24.44 22.87 24.17
Cahya_IndoBERT IndoBERT (extr.) – 4.38 16.56 6.14 17.02 16.01 16.92
XLM-R-large XLM-R (extr.) – 1.46 12.60 8.62 12.62 11.45 12.45
mDeBERTa-v3-base mDeBERTa (extr.) – 5.84 19.35 14.03 19.51 18.28 19.19
mT5 (Generative) mT5 (gen.) – 0.00 7.84 1.30 7.22 2.78 6.48
RAG (retr.+gen.) Retrieval + mT5 – – – 0.15 2.75 0.05 2.55
Closed-source instruction-tuned (LLM, no fine-tuning)
GPT-4o-mini OpenAI 0 0.00 36.06 9.87 36.84 22.77 30.15
GPT-4o-mini OpenAI 5 1.46 43.07 16.40 43.48 30.80 39.46
GPT-4o OpenAI 0 1.46 37.49 13.70 38.20 26.34 33.47
GPT-4o OpenAI 5 7.30 50.33 23.32 51.37 40.75 48.95
Open-source multilingual (LLM, no fine-tuning)
Llama-3.1-8B-Instruct Meta 0 0.73 37.82 14.48 38.51 26.83 33.89
Llama-3.1-8B-Instruct Meta 516.79 55.5526.5956.65 46.71 54.03
Qwen2.5-7B-Instruct Alibaba 0 0.00 31.03 9.04 31.81 19.47 26.43
Qwen2.5-7B-Instruct Alibaba 5 2.19 42.62 19.16 43.52 33.98 40.72
Indonesian-tuned (LLM, no fine-tuning)
Sahabat-AI-Llama3-8B GoToCompany 0 0.00 26.78 10.08 27.15 18.96 23.66
Sahabat-AI-Llama3-8B GoToCompany 5 5.84 43.32 25.32 44.15 36.40 41.95
Merak-7B-v4 Ichsan2895 0 0.00 32.44 11.43 33.30 23.02 28.95
Merak-7B-v4 Ichsan2895 5 3.65 41.83 20.31 42.89 32.45 40.60
English medical, cross-lingual (LLM, no fine-tuning)
BioMistral-7B BioMistral 0 2.92 29.83 12.07 30.38 24.22 29.21
BioMistral-7B BioMistral 5 8.03 40.08 17.03 40.73 30.42 38.40
MedGemma-4B-IT Google 0 2.92 42.60 16.88 43.44 31.42 39.01
MedGemma-4B-IT Google 5 14.60 54.2830.0755.47 45.79 53.63
Table 2: All evaluated systems on the IndoNutriQA test split ( n= 137 ). The top block reports fine-tuned baselines
(extractive, generative, retrieval-augmented); the rest report LLMs in zero- and five-shot settings under an identical
Indonesian system prompt, temperature 0, fixed few-shot sampling seed, and no parameter updates.Boldmarks the
best score per column.
performance (F1 = 36.34 , EM = 12.41 ). Because
the references are free-form prose and only 8.7%
occur verbatim in the context, an extractive model
cannot reproduce most of them: its EM and F1
are bounded by construction, and the EM–F1 gap
reflects partial overlap with a passage rather than
boundary errors on a single correct span. These
scores measure locating reference-relevant text,
not clinical correctness. Among the extractive
models, the lower scores ofIndoLEM_IndoBERT,
Cahya_IndoBERT, and the multilingual encoders
(mDeBERTa-v3-base 19.35 ,XLM-R-large 12.60 )
are consistent with in-language QA pretraining
mattering more than multilingual scale.
The mT5-based generative model performs sub-
stantially worse than extractive models, achieving
an F1 score below 8% and an EM score of 0%.
This result suggests that, for this model and data
size, the fine-tuned generative baseline struggles
to produce concise and clinically precise answers.
Error analysis (the LLM-judged taxonomy of Sec-
tion 6.7 and Table 17 in Appendix G) shows that
the model frequently generates generic or incom-
plete responses, which negatively affects both exact
matching and lexical overlap metrics. The retrieval-
augmented system yields near-zero lexical scores(ROUGE-1 and ROUGE-L below 3%; Table 2),
which Section 6.6 localizes to the generator rather
than the retriever.
Among the fine-tuned systems we evaluated, the
extractive models score highest on these overlap
metrics, subject to the caveat above; Section 6.3
shows that five-shot LLMs nonetheless surpass all
of them without any in-domain training.
6.3 Large Language Model Evaluation
We evaluate a suite of LLMs in zero- and five-
shot settings on the same test split ( n= 137 )
as the fine-tuned baselines, spanning four fam-
ilies (closed-source, open-source multilingual,
Indonesian-tuned, English-language medical). The
LLM rows of Table 2 enable head-to-head compar-
ison; this evaluation quantifies off-the-shelf clini-
cal reasoning on Indonesian text and tests whether
in-context demonstrations recover the gap to fine-
tuned models. Figure 2 (Appendix G) summarizes
the three headline findings across its panels.
Three patterns emerge clearly from Table 2.
First, every evaluated LLM benefits substantially
from five-shot prompting: gains range from +7.0
F1 (GPT-4o-mini) to +17.7 F1 (Llama-3.1-8B-
Instruct), with Sahabat-AI-Llama3-8B a close
second at +16.5 and frontier GPT-4o at +12.8

(37.49→50.33 ). The asymmetry is striking:
smaller and language-tuned models benefit more
than twice as much as the closed-source mini
model, consistent with the hypothesis that few-shot
exemplars primarily calibrate answer style (formal,
terminology-rich, span-like) rather than imparting
clinical knowledge, and that closed-source mod-
els already approximate the target style in zero-
shot. Second, model scale matters less than prompt-
formatting transfer: frontier GPT-4o (F1 = 50.33 )
is matched by the smaller Llama-3.1-8B-Instruct
(55.55 ) under five-shot prompting (overlapping
CIs; Section 6.4), mirroring the same upset for
GPT-4o-mini. Third, exact-match scores remain
low across all LLMs even when token-overlap met-
rics are strong, confirming that LLMs produce para-
phrased rather than verbatim spans: this is a metric
artifact rather than a quality issue, and reinforces
the case for semantic and clinician-judged evalua-
tions discussed below.
A second comparison is between off-the-shelf
LLMs and the best fine-tuned extractive baseline,
Rifky_IndoBERT_QA (F1 = 36.34 ). Even in
zero-shot, GPT-4o-mini ( 36.06 ), GPT-4o ( 37.49 ),
Llama-3.1-8B-Instruct ( 37.82 ), and MedGemma-
4B-IT ( 42.60 ) match or exceed this baseline despite
using no IndoNutriQA training data. With five-shot
prompting the gap widens decisively: Llama-3.1-
8B-Instruct reaches F1 = 55.55 and MedGemma-
4B-IT 54.28 , a statistical tie that also covers GPT-
4o (50.33 ; see Section 6.4), +19.21 above the best
fine-tuned model. At the data scale realistic for
expert-annotated clinical resources in low-resource
languages, IndoNutriQA’s primary value therefore
lies in evaluation rather than supervised training:
the strongest number on the benchmark is achieved
by a model that saw only five in-context training
exemplars and was never fine-tuned.
In zero-shot, Indonesian-tuned models do
not outperform multilingual peers (Sahabat-AI-
Llama3-8B F1 = 26.78 vs. Llama-3.1-8B-Instruct
37.82 ; Merak-7B-v4 32.44 is on par with Qwen2.5-
7B-Instruct 31.03 ), whereas the smallest English-
medical model, MedGemma-4B-IT at 4B parame-
ters, is the strongest zero-shot LLM overall (F1
= 42.60 , ahead of both GPT-4o variants and
Llama-3.1-8B-Instruct), with BioMistral-7B at
29.83 . With five-shot prompting MedGemma-4B-
IT climbs to F1 = 54.28 and the column-best
BLEU ( 30.07 ), statistically tied with Llama-3.1-
8B-Instruct ( 55.55 ) despite half the parameters.
For these models the pattern is consistent witha domain-transfer account, in which medical En-
glish pretraining encodes clinical-reasoning struc-
tures (PES diagnoses, anthropometric thresholds)
that general-domain Indonesian pretraining does
not. The evidence is limited, however: it rests on
two English-medical models, and BioMistral-7B
(29.83) does not follow the pattern.
6.4 Statistical Robustness Analysis
To address the statistical-power concern from n=
137 we run 5-fold cross-validation stratified by
source report, a learning curve over training frac-
tions, and bootstrap 95% CIs on every metric.
Rankings are stable: cross-validated F1 for the best
extractive baseline ( 32.62±1.29 ) matches its held-
out estimate ( 36.34 ), fold-level standard deviations
stay≤2.66 F1, and the top three 5-shot LLMs
cluster within overlapping intervals well above the
best fine-tuned baseline. The learning curve sepa-
rates data-bound from architecture-bound systems.
Full tables are in Appendix B.
6.5 Beyond Lexical Overlap: Semantic and
LLM-as-Judge Evaluation
Lexical-overlap metrics (EM, F1, BLEU, ROUGE)
underestimate paraphrased LLM outputs.2We add
two semantic metrics (BERTScore F1 (Zhang et al.,
2020) with an XLM-R backbone (Conneau et al.,
2020) and multilingual sentence-embedding co-
sine) and one clinical-rubric signal (GPT-4o-mini
LLM-as-judge on a 1-5scale: factuality, complete-
ness, safety), all with bootstrap 95% CIs (Table 9,
Appendix E). Two patterns stand out. (i) The LLM-
vs-baseline gap widens on the semantic axes: top 5-
shot LLMs reach BERTScore F1 ≈91 and cosine
≈74 versus 46/45for the best fine-tuned baseline,
expanding the lexical-F1 +19 gap to ≈+45 . (ii)
Closed-source models lead the clinical rubric (GPT-
4o-mini overall 4.37, GPT-4o safety 4.98) while
Llama and MedGemma win the similarity axes;
a second judge (Claude Haiku 4.5) preserves this
ranking (pooled κ= 0.652 , Spearman ρ= 0.916 ;
Table 10), though the top rubric position itself is
judge-dependent (GPT-4o leads under Claude). Be-
cause GPT-4o-mini is both judge and evaluated
system, we test explicitly for self-preference and
find no detectable effect (difference-in-differences
+0.125 rubric points, p= 0.34 ; Appendix E).
2Tokenisation and normalisation conventions differ across
these metrics, and the SQuAD-derived normaliser has one
Indonesian-specific side effect on exact match; see Ap-
pendix E.

Three expert raters independently re-scored a 50-
unit stratified sample: judge-vs-rater agreement ap-
proaches the inter-rater ceiling on factuality ( 0.705
vs0.780 ) and completeness ( 0.623 vs0.752 ) but
collapses on clinical safety ( 0.133 ), where both
judges score well above the rater consensus (Ap-
pendix E). We therefore treat the safety column as
unvalidated.
6.6 RAG Diagnostic Evaluation (RAGAS)
The near-zero lexical numbers for the RAG base-
line cannot localize the failure to retrieval or gener-
ation, so we compute the four RAGAS metrics (Es
et al., 2024) on the same test split. In this sin-
gle configuration (one dense retriever, recursive
chunks, un-fine-tuned mt5-base ), retrieval is the
stronger component (context precision 73.43 , re-
call35.98 ) while generation collapses (faithfulness
12.86 over the 105examples on which it is defined),
pointing to the generator as the binding constraint.
We draw no broader conclusion about RAG for
Indonesian clinical QA. Full metrics, the 32un-
scorable outputs, and the scope of this result are in
Appendix D.
6.7 Stratified Performance Analysis
We stratify F1 along three axes (full tables in Ap-
pendix G).Question type:factual (short-span ex-
traction) is uniformly the hardest, inverting the
SQuAD-style intuition; on longer-answer cate-
gories the winners diverge: Llama-3.1-8B wins rec-
ommendation (69.36), GPT-4o inferential (62.39),
MedGemma factual ( 33.39 ).Disease category:
the dataset’s respiratory prominence does not trans-
late to biased generalization; Llama-5-shot ranges
50.08 –64.17 F1 across the seven categories present
in the test split, with respiratory ( 57.55 ) within
the spread.Error taxonomy:across three fine-
tuned baselines and eight 5-shot LLMs (Table 17),
the baselines’ dominant failure is noise ( 52–76%)
and the LLMs’ is partial answer ( 52-72%, GPT-4o
leads on correct at 44%); the judge’s hallucination
label is not a reliable detector, since raters inde-
pendently find unsupported additions in 10% of
audited units (Section 6.8).
6.8 Human Error Analysis
The taxonomy above is assigned by an LLM judge,
which makes it circular as evidence about that
judge and blind to what the judge cannot see. We
therefore built an independent human counterpart:
during the calibration study the three raters couldleave a free-text note on any unit with no taxon-
omy shown to them, and 53notes were written
over37of the 50units. Error identification is thus
human and unprompted; we coded the notes into
nine categories (Table 14 and Appendix F, which
also records the method and its caveats, chiefly
that notes were optional, so every count is a lower
bound).
Two findings do not follow from the LLM-
judged taxonomy. First, the raters name a failure
class the judge’s codebook does not contain: viola-
tion of the NCP/PES structure ( 6units, 12%), most
often expanding the “S” of PES as Solusi (solution)
rather than Sign/Symptom, or collapsing several di-
agnoses into one undifferentiated statement. Every
instance falls on the nutrition-diagnosis question,
and no lexical, semantic, or judge metric here de-
tects it.
Second, the raters flagged unsupported added
content in 5units ( 10%) that the judge’s halluci-
nation label did not flag. Three concern figures
that cannot be checked against the reference, the
weaker reading; the remaining two are substantive:
the candidate asserts a weight-loss trend for a pa-
tient whose record contains no weight history, and
one of the two was flagged independently by two
raters. GPT-4o-mini scored those units 4on factu-
ality and 5on safety. We therefore make no claim
about the absence of hallucination, and the judge’s
hallucination label should not be used as a safety
indicator: a reference-only comparison cannot see
a claim the patient record silently fails to license.
7 Conclusion
IndoNutriQA fills a gap in Indonesian-language
clinical QA with 911pairs written by three nu-
tritionists and grounded in real clinical-nutrition
reports written by graduate-level dietetic interns.
The strongest results form a statistically tied top
tier of 5-shot LLMs (Llama-3.1-8B-Instruct F1
= 55.55 , MedGemma-4B-IT 54.28 ), above the
best fine-tuned IndoBERT baseline by+19.21F1,
supporting IndoNutriQA as an evaluation resource.
These conclusions are limited to the eight question
types, the clinical reports, and the configurations
tested. An expert-rater audit found 6–10% of ref-
erences contestable, safety scores unvalidated, and
unsupported content the judge missed, so we claim
no hallucination-free output. Future work will add
cases from other facilities, extend the expert audit,
and test stronger Indonesian generators for RAG.

Limitations
This work has several limitations that should be
considered when interpreting the results and using
the IndoNutriQA dataset.
First, the size of the dataset is modest at 911
question-answer pairs from 120 source reports.
This reflects the realistic constraints of collect-
ing and annotating clinical data with clinical nu-
trition experts, and we therefore frame IndoNu-
triQA primarily as an evaluation benchmark rather
than a large-scale training corpus (Sections 6.3–
6.4). To address the statistical-power concern
that follows from this scale, we report 5-fold
cross-validation stratified by source report (with
GroupKFold forbidding any cross-report con-
tamination), a learning-curve analysis over training-
set fractions, and bootstrap 95% confidence in-
tervals on every headline metric, in addition to
zero-shot and five-shot LLM evaluations that re-
quire no in-domain training data. Nevertheless,
the modest scale limits the benchmark’s ability
to expose tail behaviours of data-hungry gener-
ative and retrieval-augmented systems; reported
numbers should be interpreted as benchmark ref-
erences rather than upper-bound performance esti-
mates achievable with substantially more data.
A related concern is the scope of the question set.
IndoNutriQA is built from eight fixed questions
rather than free-form ones, which bounds what
the benchmark can measure: it exercises the eight
NCP/PES reasoning steps that structure Indone-
sian clinical-nutrition practice, and nothing outside
them. It therefore does not test open-ended clinical
dialogue, patient-initiated questions, or reasoning
about sections of the report the eight questions do
not touch (for instance pharmacological interac-
tions or longitudinal monitoring, both present in
the source documents but unqueried). We regard
the fixed question set as a deliberate trade of cov-
erage for control, since it makes answers compara-
ble across 120patients and bounds inter-annotator
variation, but it is a real ceiling on scope, and ex-
tending the question inventory is the most direct
way to raise it.
A further concern is the number of source doc-
uments rather than the number of pairs: 911pairs
drawn from 120reports average roughly 7.5pairs
per report, so case diversity is bounded by the re-
port count regardless of how many pairs are ex-
tracted. Three considerations bear on this. First,
the eight questions are standardised and applieduniformly to every report, so the pairs within a
report are not near-duplicates of one another but
probe eight distinct sections of the NCP; the effec-
tive diversity of the benchmark is therefore closer
to120cases×8reasoning types than to 911inter-
changeable items. Second, because the question
side is fixed, annotator variation is confined to an-
swer text, which is what makes a corpus of this
size usable as a controlled evaluation instrument
rather than merely a small training set. Third, the
120reports are not drawn from a single site or a
single condition: they span multiple healthcare fa-
cilities and ten disease categories with no category
exceeding 30% of reports (Table 18). None of this
removes the ceiling that 120documents impose on
rare-case coverage, which we return to below; it
does mean the constraint is on case breadth rather
than on redundancy among the pairs.
Second, the dataset exhibits domain and popula-
tion skew. A substantial portion of the cases is de-
rived from pediatric and respiratory-related clinical
contexts, reflecting real-world hospital data avail-
ability, although no single disease category exceeds
30% of source reports (Table 18). We explicitly
stratify performance by disease category in Sec-
tion 6.7 (Table 16) and find that the dataset-level
respiratory prominence does not translate to bi-
ased generalisation: Llama-3.1-8B (5-shot) ranges
50.08 –64.17 F1 across the seven categories repre-
sented in the test split, with respiratory ( 57.55 ) sit-
ting within the spread. Two caveats remain. First,
the held-out test split is not stratified by disease
category: it over-represents neurologic ( 20.0% vs.
10.0% of the corpus) and hematologic/oncologic
(15.0% vs.8.3% ) while under-representing infec-
tious ( 5.0% vs.17.5% ), and contains no cardiac or
renal/urologic reports at all, so generalisation to
those conditions remains untested on the held-out
split. Second, the 5-fold cross-validation of Sec-
tion 6.4 uses GroupKFold over all 120reports
and therefore does exercise every category, which
is the stronger evidence for category-level robust-
ness.
This bears directly on how far the reported re-
sults should be expected to generalise. Across insti-
tutions: the reports are authored at multiple Indone-
sian healthcare facilities spanning public regional
hospitals, private medical centres, and a military
hospital, rather than at a single site. This matters
because clinical-nutrition documentation is shaped
by institutional templates, and a benchmark drawn
from one hospital would reward models that learn

that hospital’s report layout. Two design choices
limit that risk: the eight standardised questions de-
rive from the NCP and PES frameworks, which are
national practice standards rather than site conven-
tions, and the cross-validation groups by source
report so that no institutional template appears
in both training and evaluation folds, with fold-
level standard deviations remaining small ( ≤2.66
F1). What remains untested is transfer to facilities
outside the contributing set, particularly primary-
care puskesmas settings, where documentation is
briefer and less standardised than the inpatient re-
ports collected here. Across rare conditions: re-
nal/urologic and ENT cases each account for un-
der1%of source reports, and for conditions rep-
resented by one or two reports no partition can
support a reliable per-category estimate. We would
expect degradation on conditions whose nutritional
management diverges sharply from the modal pedi-
atric case, for instance renal diets with protein and
electrolyte restriction, which invert the high-protein
prescriptions dominating this corpus. Extending
coverage of these conditions is the highest-value
direction for a second data-collection round, and
we flag it explicitly so that users of the benchmark
do not over-read aggregate scores as evidence of
broad clinical competence.
Third, single-annotator pairs do not guarantee
a unique gold answer per question, particularly
for the recommendation-style questions. We mit-
igate this on two axes: the NCP/PES guideline
structurally constrains acceptable answers (Ap-
pendix A.1), and our evaluation protocol uses
paraphrase-tolerant semantic and rubric metrics
(Section 6.5) alongside lexical scores. We also
report inter-rater agreement directly: three ex-
pert raters independently re-scored 50stratified
test-split units (Appendix E), reaching quadratic-
weighted κ= 0.780 on factuality and 0.752 on
completeness but only 0.160 on clinical safety. The
same exercise flagged 3of50reference answers
(6%) as incorrect or contestable by explicit flag,
rising to 5(10%) if free-text criticism of the ref-
erence is counted (Section 4); we take this as a
direct estimate of gold-answer error in the test split,
and those items are released with the benchmark
so they can be excluded or re-adjudicated. Because
two of the three raters also wrote references, this
estimate is not fully independent. Extending this
re-annotation beyond 50units, and to the training
split, remains future work.Fourth, all data are sourced from Indonesian
healthcare facilities and written in Indonesian.
While this is a deliberate design choice to sup-
port low-resource clinical NLP, it restricts direct
applicability to other languages and healthcare sys-
tems without additional adaptation. The bench-
mark does provide partial cross-lingual signals,
since two large multilingual encoders (XLM-R-
large, mDeBERTa-v3-base) are fine-tuned on the
Indonesian training data, and two English medi-
cal LLMs (BioMistral-7B, MedGemma-4B-IT) are
evaluated zero- and five-shot on Indonesian clinical
text, but a parallel English-translated test split for
fully aligned cross-lingual experiments is left for
future work.
Fifth, although the evaluation protocol goes be-
yond lexical-overlap metrics (Section 6.5 adds
BERTScore, sentence-embedding cosine, and an
LLM-as-judge clinical rubric, with a second-judge
cross-validation using Claude Haiku 4.5 yielding
pooled κ= 0.652 ), per-prediction human eval-
uation by expert raters was conducted on a 50-
unit stratified sample rather than at full scale (Ap-
pendix E). That calibration validates the rubric un-
evenly. On factuality and completeness, judge-vs-
rater agreement ( κ= 0.705 and0.623 for GPT-4o-
mini) approaches the inter-rater ceiling ( 0.780 and
0.752 ), so on those dimensions the judge is about
as close to a rater as raters are to each other. On
clinical safety it does not: agreement is κ= 0.133
(GPT-4o-mini) and 0.480 (Claude Haiku 4.5), and
both judges score safety above the rater consen-
sus, by +1.14 and+0.58 rubric points respec-
tively. The raters also disagree among themselves
on safety (Krippendorff α= 0.250 ), so this is not
solely a judge failure: safety as operationalised
here lacks a stable human target. We therefore
report the safety column as unvalidated, and no
safety score in this paper should be read as a clini-
cal guarantee. At 50units the per-stratum intervals
remain wide, and agreement on a stratified sample
need not transfer to the full test split. The human
error analysis derived from the same exercise (Sec-
tion 6.8) carries its own limits: the raters’ notes
were optional and unevenly distributed across the
three raters, so its category counts are lower bounds
on prevalence rather than error rates, and the cod-
ing of those notes into categories is ours rather than
the raters’ (Appendix F). It is nonetheless the find-
ing that most changes how our own results should
be read, since it shows the judge’s hallucination
label to be a detection ceiling rather than a mea-

sure of unsupported clinical content. The primary
judge (GPT-4o-mini) also provides two evaluated
systems; a difference-in-differences test against a
control judge from a different provider finds no
detectable self-preference (Appendix E), but with
four same-provider systems that test is underpow-
ered, and it cannot rule out a stylistic prior shared
by both judges.
Finally, the retrieval-augmented generation
(RAG) baseline exhibits low lexical scores. The
RAGAS diagnostic in Section 6.6 localises this
to the mt5-base generator (faithfulness 12.86 on
the105of137examples where it is defined, answer
relevancy 0.24) while the retriever is the stronger
component (context precision 73.43 ). Because we
evaluate a single retriever–generator–chunking con-
figuration, these numbers should be read as a lo-
calisation of failure within that configuration and
not as evidence about the effectiveness of RAG
for Indonesian clinical QA in general. A stronger
Indonesian generator with grounding-aware train-
ing is the most direct next step, and retriever-side
changes (recall is only 35.98 ) were not tested; both
are left for future work.
Despite these limitations, IndoNutriQA provides
a clinically grounded and ethically curated re-
source that establishes a foundation for future re-
search in Indonesian clinical question answering
and nutrition-focused decision support.
Ethics Statement
This work involves the construction and release
of a clinical question answering dataset derived
from real-world clinical nutrition reports. Ethical
considerations related to data privacy, consent, po-
tential harm, and responsible use were carefully
addressed throughout the dataset development and
release process.
Patient-level anonymisation.All source docu-
ments were fully anonymised prior to annotation
and release. Personally identifiable information
(PII), including patient names, medical record num-
bers, registration numbers, addresses, and dates of
birth, was removed using rule-based procedures
tailored to Indonesian clinical documents and re-
placed with deterministic placeholders ( <NAMA> ,
<NO_RM> ,<NO_REG> ,<TGL> ,<LOKASI> ). As
a result, the released dataset does not contain in-
formation that can be used to identify individual
patients. No attempts were made to re-identifyindividuals, and the dataset is intended solely for
research and educational purposes.
Institution-level anonymisation.A second
anonymisation pass strips identifying institution
mentions that the per-chunk NER list would
not catch but that nevertheless identify the host
site: hospital names ( <RS> , e.g., regional public
hospitals, private medical centers, military naval
hospitals, and other named or acronym hospital
references) and publisher names attached to
academic citations ( <PENERBIT> , e.g., medical
textbook publishers, university presses). The two
passes together leave the released dataset stripped
of both individual-patient and host-institution
identifiers.
Annotation and consent.The dataset was con-
structed from retrospective clinical-nutrition re-
ports authored by graduate-level dietetic interns as
part of supervised clinical nutrition rotations. No
direct patient contact or intervention was involved
in this study. Each (report, question, answer) triple
has exactly one author, one of three nutritionists
who wrote the answers to reports that graduate-
level dietetic interns had written under lecturer su-
pervision (Appendix A). The study does not in-
clude vulnerable-population targeting beyond what
is inherent in clinical nutrition practice (e.g., pedi-
atric cases), and no demographic attributes were
collected or inferred beyond those explicitly docu-
mented in anonymised clinical contexts.
Intended use.IndoNutriQA is released with the
explicit intention of supporting research in clini-
cal natural language processing and low-resource
language question answering. The dataset is not
intended for direct clinical deployment or auto-
mated medical decision making. Any predictions
or recommendations generated by models trained
or evaluated on this dataset should be interpreted
as decision-support signals and must not replace
professional medical judgement.
Risks and mitigations.Potential risks include the
misuse of trained models for unsupported clini-
cal inference or deployment without appropriate
validation. To mitigate this: (i) the dataset docu-
mentation states usage limitations explicitly; (ii)
benchmark results are reported as baseline refer-
ences with bootstrap confidence intervals rather
than performance guarantees (Section 6.4); (iii)
we make no claim that the evaluated systems are
free of hallucination: the LLM judge’s hallucina-
tion label is unvalidated, and expert raters audit-
ing the same kind of output independently identi-

fied unsupported clinical additions in 10% of units,
two of them asserting findings the patient record
does not license (Section 6.8); and (iv) the LLM-as-
judge clinical rubric is cross-validated across two
judge families (GPT-4o-mini and Claude Haiku
4.5, pooled Cohen’s κ= 0.652 ; Section 6.5). Re-
searchers using IndoNutriQA are nevertheless en-
couraged to conduct additional validation, includ-
ing expert review by qualified clinicians, before
applying derived models in real-world clinical set-
tings.
Overall, this work adheres to established ethi-
cal guidelines for clinical NLP research, prioritis-
ing patient privacy, institutional anonymity, trans-
parency, and responsible use while contributing a
valuable resource for advancing medical and nutri-
tional decision support in low-resource language
contexts.
Data and Code Availability.The anonymized In-
doNutriQA dataset and the full evaluation pipeline
(including the prediction caches needed to re-
produce every table and figure in this paper
without re-querying LLM APIs) are publicly re-
leased at https://github.com/NLP-UB/
IndoNutriQA . The dataset is released under CC
BY-NC 4.0, and the code is released under the MIT
license.
Acknowledgements
This research was funded by Universitas Brawi-
jaya. We thank the lecturers who supervised the re-
ports and the graduate-level dietetic interns whose
clinical-nutrition reports made this dataset possi-
ble. We also thank Al Fila Miftahul Jannah, Firda
Aqilatul ’Izzah and Reynata Olivia Kristabel, the
nutritionists who wrote the question answers, two
of whom also took part in the validation.
References
Asma Ben Abacha and Dina Demner-Fushman. 2019. A
question-entailment approach to question answering.
BMC Bioinformatics, 20:511.
Alham Fikri Aji, Haryo Akbarianto Wibowo,
Made Nindyatama Nityasya, Radityo Eko Prasojo,
and Tirana Noor Fatyanosa. 2021. BERT goes brrr:
A venture towards the lesser error in classifying
medical self-reporters on twitter. InThe Social
Media Mining for Health Applications (SMM4H)
workshop.
Emily Alsentzer, John Murphy, William Boag, Wei-
Hung Weng, Di Jindi, Tristan Naumann, andMatthew McDermott. 2019. Publicly available clin-
ical BERT embeddings. InProceedings of the 2nd
Clinical Natural Language Processing Workshop,
pages 72–78. Association for Computational Linguis-
tics.
Samuel Cahyawijaya, Genta Indra Winata, Bryan Wilie,
Karissa Vincentio, Xiaohong Li, Adhiguna Kun-
coro, Sebastian Ruder, Zhi Yuan Lim, Syafri Ba-
har, Masayu Khodra, Ayu Purwarianti, and Pascale
Fung. 2021. IndoNLG: Benchmark and resources for
evaluating Indonesian natural language generation.
InProceedings of the 2021 Conference on Empiri-
cal Methods in Natural Language Processing, pages
8875–8898, Online and Punta Cana, Dominican Re-
public. Association for Computational Linguistics.
Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan
Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and
Jennimaria Palomaki. 2020. TyDi QA: A benchmark
for information-seeking question answering in typo-
logically diverse languages.Transactions of the As-
sociation for Computational Linguistics, 8:454–470.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. 2020. Unsupervised
cross-lingual representation learning at scale. InPro-
ceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 8440–
8451.
Dina Demner-Fushman and Jimmy Lin. 2007. An-
swering clinical questions with knowledge-based and
statistical techniques.Computational Linguistics,
33(1):63–103.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAS: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2021. What disease
does this patient have? a large-scale open domain
question answering dataset from medical exams.Ap-
plied Sciences, 11:6421.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Kementerian Kesehatan Republik Indonesia. 2019. La-
poran nasional riset kesehatan dasar (riskesdas) 2018
(national report on basic health research). Technical
report, Badan Penelitian dan Pengembangan Kese-
hatan, Jakarta, Indonesia.

Yunsoo Kim, Jinge Wu, Yusuf Abdulle, and Honghan
Wu. 2024. MedExQA: Medical question answer-
ing benchmark with multiple explanations. InPro-
ceedings of the 23rd Workshop on Biomedical Natu-
ral Language Processing, pages 167–181, Bangkok,
Thailand. Association for Computational Linguistics.
Fajri Koto, Afshin Rahimi, Jey Han Lau, and Timo-
thy Baldwin. 2020. IndoLEM and IndoBERT: A
benchmark dataset and pre-trained language model
for Indonesian NLP. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 757–770, Barcelona, Spain (Online). Interna-
tional Committee on Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:453–466.
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon
Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang.
2020. BioBERT: a pre-trained biomedical language
representation model for biomedical text mining.
Bioinformatics, 36(4):1234–1240.
Emmanuel Mutabazi, Jianjun Ni, Guangyi Tang, and
Weidong Cao. 2021. A review on medical textual
question answering systems based on deep learning
approaches.Applied Sciences, 11(12).
V . Oktaria, K. J. Lee, J. E. Bines, E. Watts, C. D. Satria,
J. Atthobari, H. Nirwati, C. D. Kirkwood, Y . Soe-
narto, and M. H. Danchin. 2017. Nutritional status,
exclusive breastfeeding and management of acute
respiratory illness and diarrhea in the first 6 months
of life in infants from two regions of indonesia.BMC
Pediatrics, 17:211.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan
Sankarasubbu. 2022. Medmcqa: A large-scale multi-
subject multi-choice dataset for medical domain ques-
tion answering. InProceedings of the Conference
on Health, Inference, and Learning, volume 174 of
Proceedings of Machine Learning Research, pages
248–260. PMLR.
Anusri Pampari, Preethi Raghavan, Jennifer Liang, and
Jian Peng. 2018. emrQA: A large corpus for question
answering on electronic medical records. InProceed-
ings of the 2018 Conference on Empirical Methods
in Natural Language Processing, pages 2357–2368,
Brussels, Belgium. Association for Computational
Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.George Tsatsaronis, Georgios Balikas, Prodromos
Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R Alvers, Dirk Weissenborn, Anastasia
Krithara, Sergios Petridis, Dimitris Polychronopou-
los, Yannis Almirantis, John Pavlopoulos, Nico-
las Baskiotis, Patrick Gallinari, Thierry Artiéres,
Axel-Cyrille Ngonga Ngomo, Norman Heino, Eric
Gaussier, Liliana Barrio-Alvers, and 3 others. 2015.
An overview of the bioasq large-scale biomedical se-
mantic indexing and question answering competition.
BMC Bioinformatics, 16:138.
Wayan Oger Vihikan and I Nyoman Prayana Trisna.
2024. Indonesian health question multi-class classifi-
cation based on deep learning.Journal of Informa-
tion Systems and Informatics, 6:1931–1944.
Bryan Wilie, Karissa Vincentio, Genta Indra Winata,
Samuel Cahyawijaya, Xiaohong Li, Zhi Yuan Lim,
Sidik Soleman, Rahmad Mahendra, Pascale Fung,
Syafri Bahar, and Ayu Purwarianti. 2020. IndoNLU:
Benchmark and resources for evaluating Indonesian
natural language understanding. InProceedings of
the 1st Conference of the Asia-Pacific Chapter of the
Association for Computational Linguistics and the
10th International Joint Conference on Natural Lan-
guage Processing, pages 843–857, Suzhou, China.
Association for Computational Linguistics.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. InInterna-
tional Conference on Learning Representations.
A Annotation and Context-Pairing
Details
A.1 Question Types and Complexity
The questions in IndoNutriQA are designed to re-
flect real-world clinical nutrition workflows and
span multiple levels of reasoning complexity. Over-
all, the dataset includes four major categories of
questions. Factual questions target explicit clinical
information such as anthropometric measurements,
laboratory values, or diagnostic outcomes. Infer-
ential questions require clinical reasoning by inte-
grating multiple pieces of contextual information,
for example combining anthropometric findings
with medical diagnoses. Comparative questions
involve assessing differences across parameters or
time points, such as changes in nutritional status
or laboratory indicators. Finally, recommendation
questions focus on clinical decision support, asking
for appropriate dietary goals, principles, or guide-
lines based on the patient’s condition.
To ensure consistency and comprehensive cov-
erage across cases, each patient record is associ-
ated with a standardized set of NCP/PES questions
that operationalize these categories. The complete

Metric Value
Total QA Pairs 911
Total Source Reports 120
Average Question Length 8.32 tokens
Average Answer Length 70.66 tokens
Average Context Length 131.28 tokens
Unique Medical Diagnoses 128
Average Chunks per Report 2.60
Table 3: Dataset Statistics Overview
list of standardized questions applied uniformly to
all cases is provided in Appendix I. These ques-
tions collectively cover factual assessment, clinical
reasoning, and recommendation-oriented decision
making, ensuring that models trained on IndoNu-
triQA are evaluated across a broad spectrum of
clinically relevant clinical question answering sce-
narios. Table 3 gives the main dataset statistics,
including the average question, answer, and con-
text lengths.
Annotator Pool
The120source reports were written by graduate-
level dietetic interns at the Faculty of Health Sci-
ences, Universitas Brawijaya, as inpatient or sub-
inpatient clinical rotation deliverables of a dietetic
internship. Each report was supervised by a lec-
turer of the faculty. The reports are the source
material, not the annotation. The question answers
were written by three nutritionists, each assigned a
different set of reports. Each nutritionist wrote the
eight answers for every report in their set.
Per-pair Annotation Count and Validation
Each (report, question) pair is answered by exactly
one nutritionist. We do not collect multiple in-
dependent annotations of the same pair, because
adding parallel annotators would duplicate the en-
tire chart-review workflow per report and was out-
side the scope of this dataset release. A stratified
sample of 50units of model output was later re-
scored by one Registered Dietitian, who took no
part in writing the answers, and two of the three
nutritionists (Appendix E). That study provides the
only validation of the reference answers; we do not
claim that all pairs were reviewed.
Annotation Guideline
The nutritionists followed the clinical-nutrition doc-
umentation guideline used in the clinical nutrition
rotations, which is grounded in two standardizedclinical-nutrition reference frameworks: the Nu-
trition Care Process (NCP) for clinical-nutrition
documentation, and the PES (Problem–Etiology–
Sign/Symptom) format for nutrition diagnosis state-
ments. The eight standardized questions corre-
spond directly to NCP/PES sections (screening, an-
thropometric assessment, biochemical assessment,
nutrition diagnosis, dietary goals, dietary princi-
ples, dietary requirements, dietary guidelines), so
the answer space for each question is bounded by
the NCP/PES rubric rather than by free-form clini-
cal opinion.
Subjectivity and Inter-Annotator Agreement
We do not claim that a single nutritionist’s annota-
tion produces a unique gold answer for every case:
clinically valid alternatives exist, particularly for
the recommendation-style questions. Two design
choices mitigate this. First, the NCP/PES guide-
line structurally constrains acceptable answers: a
PES diagnosis (Question 4) must contain the three
required components in the prescribed order, and
a dietary-goals statement (Question 5) is bounded
by the patient’s clinical condition and standard-of-
care guidelines. Second, our experimental protocol
uses metrics that are robust to clinically valid lex-
ical variation: BERTScore, sentence-embedding
cosine similarity, and the LLM-as-judge rubric of
Section 6.5 explicitly do not penalize paraphrase,
and the LLM judge scores factuality and clinical
safety separately. Three caveats remain, the first
now partly resolved by measurement rather than
argument. (i) A conventional inter-annotator κover
the references is undefined here, because each re-
port was answered by a single nutritionist; in its
place we report the three-rater re-scoring study and
the gold-answer audit of Section 4 and Appendix E,
which give an expert-agreement ceiling (quadratic
κ= 0.780 factuality, 0.752 completeness, 0.160
safety) and a direct reference-error estimate ( 6%
by explicit flag, 10% including free-text criticism).
Extending that audit beyond 50units, and to the
training split, remains future work. (ii) Two of the
three raters also wrote references, so the audit is
not fully independent: only the RD took no part
in writing the answers. (iii) The three-nutritionist
pipeline is one specific protocol; reproducing this
dataset under a parallel multi-annotator protocol
would establish a strict upper bound on what single-
annotator authoring can recover, and because each
nutritionist wrote answers for different reports, au-
thor effects cannot be separated from report effects.

Question-Context Pairing: Six-Stage Matching
Unlike SQuAD-style datasets in which an anno-
tator writes both the question and a span within
a fixed passage, IndoNutriQA’s context field is
algorithmically located from the source clinical
report after the answer has been authored. For
each (report, question, answer) triple we assign
thecontext field to the chunk in which the au-
thored answer is best located, using a six-stage
matching strategy applied in order: (1) exact
substring match; (2) case-insensitive match; (3)
whitespace-normalized match (collapsing multi-
space and newline variants); (4) case-insensitive
whitespace-normalized match; (5) punctuation-
stripped match; (6) sliding-window fragment
matching with progressively shorter prefixes of the
answer ( {150,100,80,60,40,30,20} characters).
The character-level start position of the answer
within the chunk is recorded asanswer_start,
supporting extractive-QA training under a SQuAD-
compatible schema. Across the full 911-pair re-
lease, the procedure locates a verified character-
level position in 823pairs ( 90.3% ); the remain-
ing88pairs ( 9.7% ) are released with a non-
negative chunk_index pointing to the most sim-
ilar chunk but without a verified character-level
answer_start , and are flagged in the released
metadata so downstream users requiring strictly
extractive-style supervision can filter on this col-
umn.
B Statistical Robustness Details
Statistical Robustness: full discussion
To address the statistical-power concern from
n= 137 , we run three robustness procedures: 5-
fold cross-validation stratified by source report,
a learning-curve analysis over training-set frac-
tions{10,25,50,100}% , and bootstrap 95% CIs
over 1,000 resamples on every reported metric.
Cross-validated F1 for the best extractive base-
line ( 32.62±1.29 ) is consistent with its held-
out estimate ( 36.34 ,95% CI[29.03,43.59] ), with
uniformly small fold-level standard deviations
(≤2.66 F1), so rankings are stable across parti-
tions. The learning curve localizes bottlenecks:
Rifky_IndoBERT_QA still rises at 100% training
data (data-bound), while mT5 plateaus near zero
(architecture-bound). Bootstrap CIs show the top
three 5-shot LLMs (Llama-3.1-8B, MedGemma-
4B-IT, GPT-4o) cluster within overlapping inter-
vals, so the headline ranking is a top tier rather thanModel F1 % [95% CI]
Fine-tuned baselines
Rifky_IndoBERT_QA 36.34 [29.03, 43.59]
IndoLEM_IndoBERT 23.78 [17.48, 29.75]
Cahya_IndoBERT 16.56 [11.20, 21.96]
XLM-R-large 12.60 [ 8.05, 17.26]
mDeBERTa-v3-base 19.35 [13.34, 25.42]
mT5 (Generative) 7.84 [ 5.82, 9.92]
Closed-source LLM
GPT-4o-mini (0-shot) 36.06 [32.22, 40.16]
GPT-4o-mini (5-shot) 43.07 [38.75, 47.46]
GPT-4o (0-shot) 37.49 [32.60, 42.22]
GPT-4o (5-shot) 50.33 [45.14, 55.58]
Open-source multilingual LLM
Llama-3.1-8B (0-shot) 37.82 [33.76, 42.27]
Llama-3.1-8B (5-shot)55.55[49.96, 60.78]
Qwen2.5-7B (0-shot) 31.03 [27.21, 34.79]
Qwen2.5-7B (5-shot) 42.62 [37.72, 47.21]
Indonesian-tuned LLM
Sahabat-AI-Llama3-8B (0-shot) 26.78 [23.22, 30.35]
Sahabat-AI-Llama3-8B (5-shot) 43.32 [37.44, 48.79]
Merak-7B-v4 (0-shot) 32.44 [28.40, 36.66]
Merak-7B-v4 (5-shot) 41.83 [37.00, 46.75]
English-medical LLM
BioMistral-7B (0-shot) 29.83 [25.13, 34.79]
BioMistral-7B (5-shot) 40.08 [35.42, 45.15]
MedGemma-4B-IT (0-shot) 42.60 [38.14, 46.85]
MedGemma-4B-IT (5-shot) 54.28 [49.02, 59.42]
Table 4: F1 with bootstrap 95% confidence intervals
over1,000 resamples of the test set ( n= 137 ). LLM
rows distinguish 0-shot and 5-shot. Bold marks the best
point estimate.
a strict ordering, sitting well above the best fine-
tuned baseline (non-overlapping CIs; Appendix B).
Bootstrap95%Confidence Intervals
Table 4 reports F1 with bootstrap 95% confidence
intervals over 1,000 resamples of the test set ( n=
137) for every model in Table 2. CIs overlap
among the top three 5-shot LLMs, supporting the
tier-rather-than-ordering interpretation discussed in
Section 6.4.
5-Fold Cross-Validation (Stratified by Source
Report)
Each fine-tuned baseline is re-trained under
GroupKFold cross-validation so that chunks
from a single clinical document never appear in
both training and evaluation folds. Table 5 reports
per-fold mean and sample standard deviation. The
cross-validated mean for the best extractive base-
line (32.62±1.29 F1) is consistent with the held-
out test estimate of36.34(Table 2).
Learning Curve
To quantify the marginal value of additional
expert annotation, four of the fine-tuned mod-
els (Rifky_IndoBERT_QA, Cahya_IndoBERT, In-
doLEM_IndoBERT, and mT5) are re-trained on

Model EM (%) F1 (%) ROUGE-L (%)
Rifky_IndoBERT_QA 10.20±1.37 32.62±1.29 33.23±1.62
IndoLEM_IndoBERT 5.59±1.62 21.69±2.37 22.25±2.53
Cahya_IndoBERT 5.70±0.94 23.75±2.65 24.40±3.04
mT5 (Generative) 0.00±0.00 6.60±2.66 5.66±2.30
Table 5: 5-fold cross-validation, folds stratified by
source report. Mean ±standard deviation across folds.
10 25 50 100
Training-set fraction (%)05101520253035F1 (%)
Rifky_IndoBERT_QA
Cahya_IndoBERTIndoLEM_IndoBERT
mT5
Figure 1: Learning curve: F1 on the held-out test set as
a function of training-set fraction.
{10,25,50,100}% of the training split (validation
and test splits held fixed). Figure 1 plots F1 on
the held-out test set. The strongest extractive base-
line rises overall (with a small dip at 50%) and is
still climbing at 100% , indicating additional anno-
tation would likely yield gains; mT5 plateaus near
zero across all fractions, indicating its bottleneck
is architectural, not data.
CComparison with Existing Medical QA
Datasets
This appendix compares IndoNutriQA with the
medical and Indonesian QA resources discussed
in Section 2. Table 6 summarises the positioning
on the three properties the benchmark targets: In-
donesian language, nutrition domain, and ground-
ing in authentic clinical documentation. TyDi QA
covers Indonesian but is not clinical, and emrQA
is grounded in clinical records but is English and
template-generated. No other resource in the table
combines all three properties.
Table 7 extends this comparison to the full set
of benchmarks, with language, domain, sample
count, question and answer types, source, and av-
erage question and answer length. Three differ-
ences stand out. First, at 911pairs IndoNutriQA is
the smallest resource listed, which is why the pa-
per positions it as an evaluation benchmark rather
than a training corpus (Section 6). Second, its an-
swers are free text grounded in a clinical report.Benchmark Indo. Nutr. Clin. docs
SQuAD – – –
TyDi QA√– –
PubMedQA – – –
MedQA – – –
MedMCQA – – –
MedQuAD – – –
emrQA – –√
IndoNutriQA√ √ √
Table 6: Positioning of IndoNutriQA against repre-
sentative QA benchmarks. “Clinical docs” marks re-
sources grounded in authentic clinical documentation
rather than exam items or literature abstracts. The full
nine-column comparison, including sample counts and
question/answer lengths, is in Appendix C.
The exam-based resources (MedQA, MedMCQA,
MedExQA) provide a selected option, and Pub-
MedQA questions are answered yes, no or maybe,
whereas emrQA answers are spans in an electronic
health record. Third, its mean answer length of
70.66 tokens is closer to long-answer resources
such as MedQuAD ( 197words) and MedExQA
(82.5 words) than to the exam options ( 2.7–20.6
tokens or characters) or the emrQA spans ( 18.7
tokens). Length units are those reported by each
source (tokens, words or characters), and tokenisa-
tion differs, so these comparisons are approximate.
D RAGAS Metrics for the RAG Baseline
RAG Diagnostic Evaluation: full discussion
The near-zero lexical numbers for the RAG
baseline (Table 2) cannot localize the failure
to retrieval or generation. We compute the
four RAGAS metrics (Es et al., 2024) on the
same test split, scored by GPT-4o-mini with
text-embedding-3-small (Table 8, Ap-
pendix D). In this configuration retrieval is the
stronger component (context precision 73.43 , re-
call35.98 , which is only moderate) while gener-
ation collapses (faithfulness 12.86 over the 105
examples on which it is defined, answer relevancy
0.24), pointing to mt5-base rather than the re-
triever as the binding constraint here. The 32re-
maining examples are themselves evidence: RA-
GAS cannot extract a single atomic claim from
those generated answers, so faithfulness is unde-
fined: roughly one generated answer in four is too
degenerate to score at all.
Scope of this result.We draw no conclusion about
the effectiveness of RAG for Indonesian clinical
QA. This is a diagnostic of one configuration (a

Dataset Language Domain # Samples Q Type A Type SourceAvg Q
LengthAvg A
Length
MedQA (USMLE) (Jin et al., 2021) EN General Med 12,723 MCQ Single Exam 116.6 tokens 3.5 tokens
MedQA (MCMLE) (Jin et al., 2021) ZH General Med 34,251 MCQ Single Exam 45.7 chars 7.3 chars
MedQA (TWMLE) (Jin et al., 2021) ZH General Med 14,123 MCQ Single Exam 61.0 chars 20.6 chars
MedMCQA (Pal et al., 2022) EN General Med 194,000+ MCQ Single Exam 12.71 tokens 2.70 tokens
PubMedQA (PQA-L) (Jin et al., 2019) EN Biomedical 1,000+ Y/N/Maybe Short Literature 14.4 tokens 43.2 tokens
PubMedQA (PQA-U) (Jin et al., 2019) EN Biomedical 61,200+ Y/N/Maybe Short Literature 15.0 tokens 45.9 tokens
PubMedQA (PQA-A) (Jin et al., 2019) EN Biomedical 211,300+ Y/N/Maybe Short Literature 16.3 tokens 41.0 tokens
emrQA (Pampari et al., 2018) EN Clinical 400,000+ Extractive Span EHR 8.6 tokens 18.7 tokens
MedQuAD (Abacha and Demner-Fushman, 2019) EN General Med 47,457 Open Long NIH Web 7 words 197 words
MedExQA (Kim et al., 2024) EN 5 Specialties 3,500+ MCQ Single+Exp Exam 19.1 tokens 82.5 words
IndoNutriQA ID Nutrition+Med 911 Diverse Contextual Clin. Rep. 8.32 tokens 70.66 tokens
Table 7: Combined overview and statistical comparison of medical QA datasets.
Metric Score % [95% CI]
Generator-side
Faithfulness (n= 105) 12.86 [–, –]
Answer relevancy 0.24 [ 0.00, 0.60]
Retriever-side
Context precision 73.43 [66.85, 80.23]
Context recall 35.98 [29.61, 42.86]
Table 8: RAGAS metrics for the RAG baseline on the
test split ( n= 137 ). Scores in percent ( ×100 ) with
bootstrap 95% CIs. Faithfulness reported on the 105
examples for which RAGAS could parse atomic claims;
32examples produced near-empty outputs on which
faithfulness is undefined.
single dense retriever over recursive chunks feed-
ing an un-fine-tuned mt5-base generator), and
it localises the binding constraint to the generator,
not retrieval. Establishing what RAG can achieve
here needs a stronger Indonesian generator with
grounding-aware training and a sweep over retriev-
ers and chunking; we leave both to future work.
Table 8 reports the four RAGAS metrics for the
RAG baseline on the test split, decomposing the
near-zero lexical score into generator- and retriever-
side components. The diagnosis (Section 6.6) is
that, in this configuration, retrieval is the stronger
component and mt5-base generation the weaker.
E Semantic and LLM-Judge Metric
Details
Metric normalisation conventions.Exact match
and token-F1 are computed after SQuAD-style
normalisation (lowercasing, punctuation stripping,
article removal, whitespace collapsing), whereas
BLEU uses raw whitespace tokenisation and
ROUGE uses rouge_score ’s internal tokenisa-
tion and stemming. BLEU is therefore punctuation-
sensitive and reads systematically lower than F1
on the same predictions; because the convention is
applied uniformly across systems, model rankings
are unaffected. The SQuAD-derived normaliseralso removes the English articles a, an, and the,
which on Indonesian clinical text deletes the token
A from nutrient names such as vitamin A; 67of
the911reference answers contain a standalone a.
Since the same normaliser is applied to predictions
and references alike F1 is essentially unaffected,
but exact match cannot distinguish vitamin from
vitamin A. Future releases of the evaluation script
will drop the article-removal step.
Table 9 reports per-model BERTScore F1, mul-
tilingual sentence-embedding cosine, and the four
LLM-as-judge dimensions (factuality, complete-
ness, safety, overall) with bootstrap 95% CIs over
1,000 resamples. BERTScore F1 and embedding
cosine are reported as percentages; judge scores
are on a 1-5Likert scale. Lexical F1 is reproduced
from Table 2 for direct comparison.
Second-Judge Cross-Validation (Claude Haiku
4.5)
To verify that the LLM-as-judge scores are not arte-
facts of a single judge family, we re-score the same
137-example test split with Claude Haiku 4.5 (An-
thropic) under identical prompts and decoding set-
tings. Per-example inter-judge agreement between
GPT-4o-mini and Claude Haiku 4.5 across the three
rubric dimensions ( n= 2,879 paired judgements,
pooled across all 22model rows) is summarised
in Table 10; Claude-judge model-level scores are
reported in Table 11.
Pooled κ= 0.652 indicates substantial agree-
ment under the Landis–Koch rubric, with the two
judges within 1Likert point on 82.8% of paired
judgements. The dimension-level breakdown is
intuitive: completeness has the highest agreement
(κ= 0.711 ) because both judges look for miss-
ing reference content; factuality is moderate ( κ=
0.584 ); safety is the lowest ( κ= 0.251 ) because
both judges concentrate scores near the ceiling
(mean ≈4.8 /5across all models), which com-

Model Lexical F1 BERTScore F1 Embed. cos Judge fact. Judge compl. Judge safety Judge overall
Fine-tuned baselines
Rifky_IndoBERT_QA 36.34 45.94 [38.41, 54.07] 45.27 [39.31, 51.74] 3.14 [2.84, 3.45] 2.84 [2.53, 3.13] 4.89 [4.80, 4.96] 3.62 [3.43, 3.83]
IndoLEM_IndoBERT 23.78 35.42 [27.87, 42.57] 36.23 [30.51, 41.84] 2.61 [2.34, 2.91] 2.33 [2.07, 2.60] 4.85 [4.77, 4.92] 3.26 [3.09, 3.45]
Cahya_IndoBERT 16.56 25.78 [19.24, 32.26] 28.61 [23.64, 33.79] 2.46 [2.18, 2.75] 2.19 [1.93, 2.45] 4.85 [4.77, 4.92] 3.17 [2.99, 3.36]
XLM-R-large 12.60 23.16 [16.74, 29.64] 27.61 [23.02, 32.62] 2.34 [2.05, 2.64] 2.12 [1.88, 2.38] 4.84 [4.75, 4.92] 3.10 [2.92, 3.29]
mDeBERTa-v3-base 19.35 28.47 [21.53, 35.90] 32.18 [26.71, 38.05] 2.54 [2.28, 2.83] 2.32 [2.07, 2.59] 4.89 [4.82, 4.94] 3.25 [3.08, 3.44]
mT5 (Generative) 7.84 79.11 [78.54, 79.70] 18.19 [15.09, 21.40] 1.08 [1.02, 1.15] 1.04 [1.01, 1.08] 4.62 [4.43, 4.80] 2.25 [2.18, 2.32]
Closed-source LLM
GPT-4o-mini (0-shot) 36.06 88.08 [87.43, 88.74] 61.13 [58.11, 64.08] 4.19 [4.01, 4.36]3.86[3.71, 4.02] 4.93 [4.89, 4.97] 4.33 [4.22, 4.44]
GPT-4o-mini (5-shot) 43.07 89.40 [88.67, 90.17] 66.51 [63.18, 69.77]4.36[4.20, 4.52] 3.80 [3.61, 3.96] 4.96 [4.93, 4.99]4.37[4.26, 4.47]
GPT-4o (0-shot) 37.49 88.07 [87.24, 88.82] 57.38 [52.96, 61.48] 3.94 [3.69, 4.19] 3.67 [3.43, 3.88] 4.96 [4.91, 4.99] 4.19 [4.03, 4.34]
GPT-4o (5-shot) 50.33 90.62 [89.75, 91.51] 69.66 [65.42, 74.14] 4.21 [4.01, 4.41] 3.58 [3.37, 3.80]4.98[4.95, 5.00] 4.26 [4.12, 4.38]
Open-source multilingual LLM
Llama-3.1-8B (0-shot) 37.82 88.32 [87.61, 89.04] 60.95 [57.01, 64.50] 3.94 [3.72, 4.15] 3.72 [3.51, 3.90] 4.87 [4.81, 4.93] 4.18 [4.03, 4.31]
Llama-3.1-8B (5-shot)55.5591.09 [90.20, 91.94]74.37[70.58, 78.24] 3.95 [3.72, 4.17] 3.43 [3.20, 3.65] 4.88 [4.82, 4.93] 4.09 [3.93, 4.24]
Qwen2.5-7B (0-shot) 31.03 87.35 [86.69, 87.99] 58.28 [55.21, 61.30] 3.73 [3.53, 3.93] 3.62 [3.47, 3.77] 4.82 [4.74, 4.88] 4.06 [3.94, 4.18]
Qwen2.5-7B (5-shot) 42.62 89.48 [88.65, 90.26] 66.50 [62.98, 70.08] 4.12 [3.93, 4.28] 3.64 [3.46, 3.80] 4.91 [4.85, 4.96] 4.22 [4.10, 4.33]
Indonesian-tuned LLM
Sahabat-AI-Llama3-8B (0-shot) 26.78 85.89 [85.24, 86.56] 64.09 [61.15, 67.06] 4.05 [3.84, 4.26] 3.58 [3.36, 3.81] 4.92 [4.88, 4.96] 4.18 [4.04, 4.32]
Sahabat-AI-Llama3-8B (5-shot) 43.32 89.97 [88.97, 90.99] 71.69 [67.99, 75.59] 3.54 [3.27, 3.82] 3.01 [2.77, 3.25] 4.67 [4.54, 4.78] 3.74 [3.56, 3.93]
Merak-7B-v4 (0-shot) 32.44 88.00 [87.35, 88.69] 61.48 [58.01, 64.88] 3.69 [3.45, 3.91] 3.43 [3.23, 3.62] 4.74 [4.64, 4.82] 3.95 [3.80, 4.10]
Merak-7B-v4 (5-shot) 41.83 89.95 [89.17, 90.81] 69.43 [65.98, 73.06] 3.82 [3.61, 4.04] 3.18 [2.99, 3.37] 4.85 [4.77, 4.91] 3.95 [3.82, 4.09]
English-medical LLM
BioMistral-7B (0-shot) 29.83 74.76 [69.61, 79.76] 53.05 [48.38, 57.77] 2.84 [2.57, 3.13] 2.40 [2.18, 2.65] 4.90 [4.82, 4.96] 3.38 [3.22, 3.55]
BioMistral-7B (5-shot) 40.08 89.38 [88.55, 90.31] 66.14 [62.14, 70.35] 3.28 [3.03, 3.55] 2.66 [2.46, 2.88] 4.74 [4.64, 4.82] 3.56 [3.40, 3.73]
MedGemma-4B-IT (0-shot) 42.60 88.79 [88.04, 89.57] 63.50 [60.08, 66.87] 3.97 [3.77, 4.17] 3.66 [3.46, 3.83] 4.92 [4.86, 4.96] 4.18 [4.05, 4.31]
MedGemma-4B-IT (5-shot) 54.2891.12[90.20, 92.03] 73.56 [69.91, 77.52] 3.82 [3.59, 4.05] 3.37 [3.15, 3.61] 4.88 [4.81, 4.94] 4.02 [3.87, 4.18]
Table 9: Semantic and LLM-judge metrics on the IndoNutriQA test split (n= 137), with bootstrap95%CIs.
Dimension QuadraticκWithin-1 pt. Mean abs. diff.
Factuality 0.584 83.3% 0.77
Completeness 0.711 84.6% 0.71
Safety 0.251 80.7% 0.66
Pooled (all dims) 0.652 82.8% 0.71
Table 10: Inter-judge agreement between GPT-4o-mini
and Claude Haiku 4.5 across the three rubric dimensions.
Quadratic-weighted Cohen’s κover the 1–5Likert or-
dinal scale; “within-1” = fraction of paired judgements
that differ by≤1Likert point.
presses the kappa denominator; the within-1-point
agreement on safety remains high ( 80.7% ). Im-
portantly, the top-tier ranking on the overall-rubric
score is preserved across both judges: GPT-4o (5-
shot) wins safety under both, and GPT-4o-mini /
GPT-4o / Llama-3.1-8B / MedGemma form a tight
top tier on the overall score under both judges. The
full Claude-judge model-level scores are listed in
Table 11.
Judge Self-Preference Analysis
GPT-4o-mini serves as the primary judge while
GPT-4o-mini and GPT-4o also appear as evalu-
ated systems. We therefore test directly for self-
preference. A raw comparison of the judge’s scores
on OpenAI systems against its scores on other sys-
tems would confound self-preference with the fact
that the OpenAI systems are genuinely stronger.
We instead use a difference-in-differences design.
For each of the 22evaluated systems we compute
the mean overall-rubric score under GPT-4o-miniminus the mean under Claude Haiku 4.5, restricted
to the examples both judges scored, so the per-
system judge gap is a paired quantity from which
true system quality, visible to both judges, has been
differenced out. We then ask whether that gap is
larger for systems sharing the judge’s provider than
for the rest. Claude Haiku 4.5is a valid control be-
cause no evaluated system comes from its provider,
so it has no self to prefer.
Table 13 reports the result. GPT-4o-mini is
uniformly more lenient than Claude Haiku 4.5
(+0.186 rubric points on non-OpenAI systems
alone), but the additional leniency it extends to its
own provider is +0.125 points on the overall score
and is not statistically distinguishable from zero
(permutation p= 0.34 ). The per-dimension break-
down argues further against a systematic mecha-
nism: the estimated effect is positive on factual-
ity (+0.456 ,p= 0.103 ) but negative on safety
(−0.259 ,p= 0.092 ), i.e. the judge is if anything
harsher on its own provider’s safety. A genuine
self-preference bias would push consistently in one
direction. Across the eight tests reported, none
is significant at a Bonferroni-corrected threshold
(α= 0.00625).
Two caveats. First, with only four same-provider
systems the test is underpowered; we can report
that no self-preference effect is detectable, not that
none exists. Second, the design detects provider-
level favouritism, not a stylistic prior that both
judges might share; the expert-rater calibration of

Model Judge fact. Judge compl. Judge safety Judge overall
Fine-tuned baselines
Rifky_IndoBERT_QA 3.98 [3.68, 4.29] 2.93 [2.60, 3.26] 4.28 [4.02, 4.53] 3.73 [3.46, 3.98]
IndoLEM_IndoBERT 3.80 [3.49, 4.10] 2.35 [2.05, 2.66] 4.17 [3.93, 4.41] 3.44 [3.20, 3.68]
Cahya_IndoBERT 3.85 [3.52, 4.18] 2.23 [1.90, 2.57] 4.21 [3.93, 4.45] 3.43 [3.14, 3.69]
XLM-R-large 3.81 [3.51, 4.12] 1.96 [1.69, 2.29] 4.12 [3.86, 4.38] 3.30 [3.07, 3.55]
mDeBERTa-v3-base 3.87 [3.58, 4.17] 2.20 [1.88, 2.54] 4.18 [3.93, 4.42] 3.42 [3.18, 3.65]
mT5 (Generative) 1.60 [1.42, 1.78] 1.06 [1.02, 1.09] 3.03 [2.81, 3.26] 1.90 [1.78, 2.01]
Closed-source LLM
GPT-4o-mini (0-shot) 3.99 [3.82, 4.16] 3.23 [3.04, 3.42] 4.34 [4.18, 4.48] 3.86 [3.70, 4.00]
GPT-4o-mini (5-shot) 4.15 [3.98, 4.31]3.32[3.09, 3.54] 4.50 [4.37, 4.64] 3.99 [3.83, 4.14]
GPT-4o (0-shot) 4.09 [3.89, 4.28] 3.10 [2.86, 3.33] 4.53 [4.39, 4.65] 3.91 [3.76, 4.05]
GPT-4o (5-shot)4.39[4.22, 4.56] 3.29 [3.06, 3.51]4.75[4.64, 4.85]4.15[4.00, 4.28]
Open-source multilingual LLM
Llama-3.1-8B (0-shot) 3.92 [3.73, 4.11] 3.09 [2.87, 3.31] 4.30 [4.14, 4.44] 3.77 [3.60, 3.93]
Llama-3.1-8B (5-shot) 4.12 [3.91, 4.30] 3.20 [2.97, 3.44] 4.42 [4.26, 4.57] 3.91 [3.74, 4.08]
Qwen2.5-7B (0-shot) 3.69 [3.51, 3.88] 2.96 [2.75, 3.18] 4.15 [4.00, 4.31] 3.60 [3.45, 3.76]
Qwen2.5-7B (5-shot) 4.17 [3.99, 4.34] 3.16 [2.93, 3.38] 4.53 [4.38, 4.67] 3.95 [3.80, 4.10]
Indonesian-tuned LLM
Sahabat-AI-Llama3-8B (0-shot) 3.94 [3.73, 4.13] 2.80 [2.61, 3.00] 4.26 [4.09, 4.42] 3.66 [3.51, 3.82]
Sahabat-AI-Llama3-8B (5-shot) 3.74 [3.50, 3.96] 2.85 [2.58, 3.12] 4.16 [3.96, 4.35] 3.58 [3.37, 3.78]
Merak-7B-v4 (0-shot) 3.66 [3.45, 3.87] 2.69 [2.47, 2.92] 3.99 [3.77, 4.18] 3.45 [3.26, 3.63]
Merak-7B-v4 (5-shot) 3.88 [3.67, 4.08] 2.77 [2.55, 3.00] 4.23 [4.04, 4.40] 3.62 [3.45, 3.80]
English-medical LLM
BioMistral-7B (0-shot) 3.68 [3.41, 3.93] 2.20 [1.98, 2.43] 4.11 [3.89, 4.31] 3.33 [3.14, 3.53]
BioMistral-7B (5-shot) 3.65 [3.42, 3.87] 2.63 [2.39, 2.87] 3.96 [3.76, 4.16] 3.41 [3.21, 3.61]
MedGemma-4B-IT (0-shot) 3.95 [3.75, 4.14] 3.04 [2.83, 3.26] 4.38 [4.23, 4.51] 3.79 [3.64, 3.95]
MedGemma-4B-IT (5-shot) 4.00 [3.80, 4.20] 3.09 [2.85, 3.32] 4.29 [4.12, 4.47] 3.79 [3.62, 3.96]
Table 11: Claude Haiku 4.5 LLM-as-judge scores (the second-judge cross-validation of Table 9). Scale: 1–5Likert
with bootstrap95%CIs over1,000resamples. Bold marks the strongest model per dimension.
Appendix E addresses the latter by anchoring both
judges to human clinical judgement.
Expert-Rater Calibration of the LLM Judge
To anchor the LLM-as-judge rubric to human clin-
ical judgement, three expert raters (one Regis-
tered Dietitian, two nutritionists) independently re-
scored a stratified sample of model outputs. We
drew 50(test example, system output) units strat-
ified by question category (factual / inferential /
recommendation) crossed with system quality tier,
so that the sample spans the full answer-quality
range; the 50units cover 50distinct test exam-
ples and all 20source reports in the test split, and
were produced by 20of the 22evaluated systems.
Each rater scored every unit on the same three-
dimension, 1–5rubric the LLM judge uses (factual-
ity, completeness, clinical safety), blind to system
identity and to the judges’ scores, working from
the question, the reference answer, the candidate
answer, and the patient’s diagnosis and anthropo-
metric assessment. Stratification was on question
category and aggregate system quality, never on
the per-unit judge score, so the sample does not
condition on the quantity being estimated.
Table 12 (top) reports quadratic-weighted Co-
hen’s κfor the reference rater pair and Krippen-dorff’s ordinal αacross all three. Factuality ( κ=
0.780 ,α= 0.592 ) and completeness ( κ= 0.752 ,
α= 0.712 ) reach substantial agreement. Clini-
cal safety does not ( κ= 0.160 ,α= 0.250 ): the
three raters’ mean safety scores are 3.88,4.66, and
2.94, a spread of 1.72 rubric points between the
most and least lenient. Safety, as operationalised
by this rubric, is therefore not a dimension on
which raters themselves converge, and any judge-
vs-human safety comparison inherits that instabil-
ity.
Table 12 (middle) compares each judge against
the rater consensus (per-dimension mean, rounded
half up). The key comparison is vertical: on factu-
ality, GPT-4o-mini agrees with the rater consensus
atκ= 0.705 against an inter-rater ceiling of 0.780 ,
and on completeness at 0.623 against a ceiling of
0.752 ; the confidence intervals overlap substan-
tially in both cases. On these two dimensions the
judge is therefore about as close to a rater as raters
are to each other, which is the strongest validation
available for a rubric of this kind. Agreement is
highest on the recommendation questions (factu-
alityκ= 0.918 ) and weakest on the short-span
factual ones ( 0.478 ), mirroring the difficulty order-
ing in Section 6.7.

Fact. Compl. Safety
Inter-rater (n= 50)
Quadraticκ0.780 0.752 0.160
Krippendorffα0.592 0.712 0.250
Judge vs rater consensus (quadraticκ)
GPT-4o-mini0.705 0.623 0.133
Claude Haiku 4.50.674 0.585 0.480
Judge leniency (judge−consensus, rubric points)
GPT-4o-mini−0.14−0.20 +1.14
Claude Haiku 4.50.00−0.78 +0.58
Table 12: Expert-rater calibration of the LLM-as-judge
rubric on 50stratified units. Inter-rater κis the ref-
erence pair; Krippendorff’s ordinal αpools all three
raters. Judge agreement on factuality and completeness
approaches the inter-rater ceiling, whereas safety agree-
ment is low and both judges score safety substantially
above the rater consensus.
Safety is the exception on both axes. Judge-vs-
consensus κis0.133 for GPT-4o-mini and 0.480
for Claude Haiku 4.5, and the leniency column
shows why: GPT-4o-mini scores safety +1.14
rubric points above the rater consensus (mean 4.92
against 3.78), essentially saturating the scale, and
Claude +0.58 . Both judges systematically over-
state the clinical safety of model output. Two fur-
ther asymmetries are worth recording. GPT-4o-
mini is not uniformly lenient: on the factual ques-
tions it is stricter than the raters ( −0.82 on factual-
ity). Claude Haiku 4.5 systematically under-scores
completeness ( −0.78 pooled). We therefore rec-
ommend that the safety dimension of this rubric be
treated as unvalidated, that Claude Haiku 4.5 be pre-
ferred where a safety signal is needed ( κ= 0.480
vs0.133 ), and that neither judge’s absolute safety
score be read as a clinical guarantee.
Raters could additionally flag units whose refer-
ence answer they judged incorrect or contestable.
Three of 50units were flagged ( 6%), one of them
by two raters independently; the flagged cases con-
cern macronutrient percentage distributions that do
not sum to 100% , a mid-upper-arm-circumference
cut-off, and a neonatal protein prescription. This
is a direct estimate of gold-answer error in the test
split and is consistent with the single-annotator
caveat recorded in the Limitations; the flagged
items are released with the benchmark so that users
can exclude or re-adjudicate them.
Finally, the choice of judge does not materially
reorder the leaderboard (Spearman ρ= 0.916 be-
tween the two judges’ system rankings), and the
substantive conclusion, that closed-source modelsMean judge gap
Dimension Same prov. Other DiDp
(n=4) (n=18)
Factuality+0.018−0.438 +0.456 0.103
Completeness+0.491 +0.313 +0.178 0.225
Safety+0.425 +0.684−0.259 0.092
Overall+0.311 +0.186 +0.125 0.340
Table 13: Judge self-preference test. “Judge gap” is
the per-system mean score under GPT-4o-mini minus
the mean under Claude Haiku 4.5on commonly scored
examples; positive values mean GPT-4o-mini is more
lenient. “Same prov.” groups the four systems sharing
the judge’s provider (GPT-4o-mini and GPT-4o, zero-
and five-shot). DiD is the difference-in-differences es-
timate of self-preference; pis a two-sided permutation
test over 20,000 relabellings. No dimension shows a
significant effect, and the factuality and safety estimates
carry opposite signs.
lead the clinical rubric while open-source models
lead the similarity axes, holds under both judges.
The single rank that does move is the top position
itself: GPT-4o-mini leads under its own family’s
judge ( 4.37) whereas GPT-4o leads under Claude
(4.15). We therefore report the rubric leader as a
top tier rather than a single winner.
F Human Error Analysis Details
The nine categories of Table 14 were obtained
by open-coding the 53free-text notes the three
expert raters left during the calibration study of
Appendix E, then re-applying the finalised code-
book to every note from scratch. No taxonomy was
shown to the annotators, so error identification is
theirs and unprompted; the coding is ours. A note
may carry several codes (a rater often names an
omission and a format violation in one sentence),
so codes sum to more than53.
Table 14 reports, for each category, the number
and percentage of the 50units in which it occurs;
a unit may carry several categories, so the percent-
ages sum to more than 100% . “Mean rater score”
is the raters’ own 1–5rating (factuality, complete-
ness, safety) on the units carrying that category.
Two categories have no counterpart in the LLM-
judged taxonomy of Table 17: NCP/PES violation,
and unsupported addition, for which the judge’s
hallucination label did not flag the units.
Notes were optional, and the rate differs sharply
by rater ( 36,9, and 8notes), so each count is a
lower bound on prevalence and not an error rate;
the category ordering is more reliable than the abso-

Units Mean rater score
Human error categoryn% Fact. Compl. Safe.
Omission18 36 3.32 2.58 2.53
Non-answer / truncated8 16 1.75 1.50 1.50
NCP/PES violation6 12 3.20 2.60 2.40
Reference itself wrong5 10 2.83 3.67 2.50
Unsupported addition5 10 4.17 3.83 4.00
Degenerate text4 8 4.40 4.20 4.40
Contradicts patient data3 6 2.00 2.00 1.80
Numeric error3 6 2.00 3.00 1.75
Non-standard language3 6 4.00 4.00 3.67
Table 14: Human error analysis: 53free-text notes from
three expert raters over 37of50audited units, coded
into nine categories. % is of the 50units; counts are
lower bounds. Details are in Appendix F.
lute percentages. The 50units span 20systems, so
the analysis characterises failure modes of the eval-
uated pool rather than of any one system. Because
the raters saw the patient’s diagnosis and anthropo-
metric assessment while the judge saw only refer-
ence and candidate, part of the judge-vs-human gap
on unsupported content is an information gap rather
than a reasoning gap, which is itself an argument
for giving judges the record.
Numeric errors concentrate on the recommenda-
tion questions, which carry prescription arithmetic.
And degenerate output (repetition, or report boiler-
plate and table markup leaking into the answer) still
draws high rubric scores ( 4.40/4.20/4.40 ): the
raters noted the defect in prose but did not treat
it as a clinical error, which is a rubric-design gap
rather than a model behaviour.
G Stratified Performance Tables
Figure 2 summarizes the headline findings of the
experiments: the gap between lexical and semantic
scores (panel a), the error taxonomy of the LLM
judge (panel b), and the effect of five-shot prompt-
ing (panel c). The tables that follow give the strati-
fied results in detail.
Question-Category Breakdown
Table 15 reports per-model F1 broken down by
the question categories defined in Appendix A.1
(factual, inferential, and recommendation; the com-
parative category is absent because the standard-
ized questions each target a single time point per
patient). This stratification isolates which reason-
ing types drive the aggregate scores in Section 6.2:
recommendation and inferential questions, whose
answers are longer and more structured, generally
attract higher F1 than the shorter factual questions,indicating that headline performance is not uniform
across reasoning complexity.
Disease-Category Breakdown
Figure 3 visualizes the disease-category F1 matrix
as a heatmap, making the absence of a systematic
advantage for the largest category (respiratory) vis-
ible at a glance. Table 16 gives the full numeric
values.
Error Taxonomy
Table 17 classifies each model’s outputs on a sam-
pled subset of 50test examples into six categories
(correct, partial, wrong span, terminology mis-
match, hallucination, and noise), assigned by an
LLM judge (GPT-4o-mini). This qualitative view
complements the aggregate metrics in Section 6.2
by revealing how models fail rather than only how
much. Two patterns stand out. First, the fine-tuned
extractive baselines are dominated by noise ( 52–
76%), reflecting empty or malformed span predic-
tions, whereas the 5-shot LLMs convert almost all
of these into partial answers that capture some but
not all of the reference content. Second, and most
importantly, the human error analysis of Section 6.8
shows that the judge’s hallucination label cannot
be relied on: raters reading the same class of output
flagged unsupported additions in 10% of audited
units, including two that assert a clinical trend the
patient record does not support and that the judge
scored 4on factuality. We therefore draw no con-
clusion about fabrication rates; under-answering is
the dominant failure on both the judge’s and the
raters’ accounts.
H Demographic, Anthropometric, and
Syntactic-Divergence Figures
Table 18 gives the number and share of source
reports and QA pairs for each disease category.
Figure 4 presents the full distribution of disease
categories in IndoNutriQA. Figure 5 shows the age
in months (panel (a)), sex (panel (b)), weight-vs-
height colored by age (panel (c)), and MUAC/LILA
(panel (d)) distributions of patient cases, alongside
the bimodal distribution of token-level normalized
edit distance between model predictions and refer-
ence answers (panel (e)).

0 20 40 60
Lexical F1 (%)020406080100BERTScore F1 (%)Rifky
IndoLEM
CahyamT55-shot LLMs(a) Lexical vs. semantic
metrics agree
Fine-tuned
Closed LLM
Open LLM
Indonesian LLM
Medical LLM
0 20 40 60 80 100
% of sampled outputs (n=50)Rifky
(fine-tuned)
GPT-4o-mini
(5-shot)
Llama-3.1-8B
(5-shot)32%
30%
36%12%
70%
62%54%(b) Error taxonomy
correct
partial answerwrong span
terminology mismatchhallucination
noiseGPT-4o-mini Llama-3.1-8B01020304050607080Lexical F1 (%)36.143.1+7.0
37.855.6+17.7(c) Few-shot lift
0-shot
5-shotFigure 2: Headline findings on the IndoNutriQA test split ( n= 137 ).(a)Lexical vs. BERTScore F1, markers shaped
by model family: LLMs cluster above the “metrics agree” diagonal and mT5 falls far off it (high BERTScore from
generic-token alignment, near-zero lexical), so no single metric suffices.(b)LLM-judge error taxonomy over 50
sampled outputs per model (three representative rows): the extractive baseline is dominated by noise ([CLS]-token
collapse), the LLMs by partial answer, and the judge’s hallucination label is unreliable, as the human error analysis
of Section 6.8 shows.(c)Five-shot prompting lifts Llama-3.1-8B by +17.7 F1 versus +7.0 for GPT-4o-mini,
consistent with demonstrations mainly calibrating answer style.
Model Factual (n=38) Inferential (n=39) Recommendation (n=60)
Fine-tuned baselines
Rifky_IndoBERT_QA 12.78 29.80 55.51
IndoLEM_IndoBERT 5.07 17.18 39.92
Cahya_IndoBERT 2.55 3.55 33.89
XLM-R-large 2.09 4.43 24.56
mDeBERTa-v3-base 3.07 6.05 38.30
mT5 (Generative) 5.19 12.10 6.75
Closed-source LLM
GPT-4o-mini (0-shot) 21.83 40.46 42.20
GPT-4o-mini (5-shot) 31.24 42.82 50.71
GPT-4o (0-shot) 18.47 51.91 40.17
GPT-4o (5-shot) 25.8862.3957.97
Open-source multilingual LLM
Llama-3.1-8B (0-shot) 21.14 43.31 44.82
Llama-3.1-8B (5-shot) 29.64 59.5569.36
Qwen2.5-7B (0-shot) 18.46 33.98 37.07
Qwen2.5-7B (5-shot) 29.23 52.12 44.91
Indonesian-tuned LLM
Sahabat-AI-Llama3-8B (0-shot) 19.91 29.32 29.48
Sahabat-AI-Llama3-8B (5-shot) 21.38 49.61 53.13
Merak-7B-v4 (0-shot) 22.55 38.33 34.88
Merak-7B-v4 (5-shot) 28.96 47.12 46.55
English-medical LLM
BioMistral-7B (0-shot) 14.58 30.41 39.10
BioMistral-7B (5-shot) 32.30 32.24 50.10
MedGemma-4B-IT (0-shot) 23.90 52.61 47.92
MedGemma-4B-IT (5-shot)33.3961.52 62.80
Table 15: F1 by question category on the IndoNutriQA test split. Categories from Appendix I: factual ( n= 38 ),
inferential ( n= 39 ), recommendation ( n= 60 ). The fourth category proposed in Appendix A.1 (comparative)
is not represented because the standardized questions all target a single time point per patient. Bold marks the
strongest model per category.
I Standardized Questions and Example
Instance
Standardized Clinical Nutrition Questions
1.Apa metode skrining status gizi yang di-
rekomendasikan?
What nutritional screening method is recom-
mended?2. Bagaimana status gizi pada pasien ini?
What is the nutritional status of this patient?
3.Bagaimana data biokimia/parameter lab pada
pasien (yang abnormal)?
What are the abnormal biochemical or labo-
ratory parameters of this patient?

Resp. Gastro. Neuro. Hem./
Onc.Nutr./
Gr.Infect. ENTRifky_IndoBERT
IndoLEM_IndoBERT
Cahya_IndoBERT
XLM-R-large
mDeBERTa-v3-base
mT5
GPT-4o-mini (0)
GPT-4o-mini (5)
GPT-4o (0)
GPT-4o (5)
Llama-3.1-8B (0)
Llama-3.1-8B (5)
MedGemma-4B-IT (0)
MedGemma-4B-IT (5)35.7 41.1 31.8 33.7 23.8 47.5 48.9
24.5 26.8 27.5 18.8 5.7 13.7 38.6
18.1 18.7 9.4 14.3 8.4 27.3 29.0
13.8 19.8 4.9 10.7 2.1 16.5 17.9
20.2 19.4 23.3 13.3 0.7 16.5 40.2
7.4 5.4 6.3 5.2 13.9 16.5 14.8
36.2 34.3 38.3 33.7 35.3 42.8 35.2
44.6 38.7 46.5 39.7 40.8 45.8 48.4
37.1 35.2 38.9 34.8 40.6 44.1 42.2
53.4 51.0 48.2 43.0 45.9 59.4 55.2
41.1 36.6 37.2 31.9 34.7 47.3 37.0
57.5 52.2 57.5 50.1 51.8 61.6 64.2
43.6 40.7 41.5 42.1 44.7 51.7 37.9
57.6 51.2 54.4 48.0 57.3 66.4 50.5F1 by disease category
010203040506070
F1 (%)Figure 3: F1 by disease category for each evaluated model. Rows banded by family (fine-tuned baselines above the
solid line, LLMs below); cells colored on a common scale. The respiratory column (the largest category, 28.33% of
source reports) is not systematically darker than the columns for categories with far fewer test-split reports (ENT,
infectious, nutrition/growth), supporting the Section 5.1 claim that the dataset imbalance does not translate to biased
generalization.
ModelResp.
(n=42)Gastro.
(n=27)Neuro.
(n=25)Hem./Onc.
(n=22)Nutr./Gr.
(n=7)Infect.
(n=7)ENT
(n=7)
Fine-tuned baselines
Rifky_IndoBERT_QA 35.71 41.96 31.83 34.23 23.77 41.15 48.91
IndoLEM_IndoBERT 24.51 26.76 27.48 18.78 5.71 13.65 38.57
Cahya_IndoBERT 18.37 18.72 9.35 13.89 8.37 27.27 29.00
XLM-R-large 13.77 19.83 4.87 10.67 2.09 16.54 17.85
mDeBERTa-v3-base 20.22 19.40 23.29 13.32 0.68 16.53 40.24
mT5 (Generative) 7.36 6.53 6.33 5.19 13.86 16.48 14.83
Closed-source LLM
GPT-4o-mini (0-shot) 36.20 34.35 38.28 33.73 35.34 42.75 35.16
GPT-4o-mini (5-shot) 44.57 38.73 46.53 39.74 40.76 45.80 48.36
GPT-4o (0-shot) 37.10 35.20 38.93 34.83 40.55 44.13 42.23
GPT-4o (5-shot) 53.40 51.03 48.19 42.98 45.91 59.41 55.24
Open-source multilingual LLM
Llama-3.1-8B (0-shot) 41.14 36.58 37.23 31.95 34.71 47.28 36.95
Llama-3.1-8B (5-shot) 57.5552.23 57.54 50.0851.83 61.5664.17
Qwen2.5-7B (0-shot) 30.49 29.90 31.92 31.18 35.21 30.98 30.83
Qwen2.5-7B (5-shot) 44.23 44.85 40.72 33.29 41.95 41.50 62.17
Indonesian-tuned LLM
Sahabat-AI-Llama3-8B (0-shot) 27.88 26.59 27.02 25.18 29.35 23.57 25.61
Sahabat-AI-Llama3-8B (5-shot) 44.64 39.63 48.81 43.22 40.86 38.05 38.12
Merak-7B-v4 (0-shot) 31.57 33.67 33.31 30.91 33.30 35.12 31.16
Merak-7B-v4 (5-shot) 41.79 45.76 42.58 35.52 42.31 42.74 42.74
English-medical LLM
BioMistral-7B (0-shot) 27.30 32.18 30.97 23.01 27.11 33.19 52.58
BioMistral-7B (5-shot) 42.32 32.29 43.73 37.67 35.04 53.04 43.28
MedGemma-4B-IT (0-shot) 43.64 40.65 41.54 42.10 44.71 51.71 37.93
MedGemma-4B-IT (5-shot)57.5851.25 54.41 47.9657.30 66.3650.48
Table 16: F1 by disease category on the IndoNutriQA test split. Categories follow the Section 5.1 treemap; the test
split contains the seven categories shown. Cardiac and renal/urologic are not represented in this test split. Bold
marks the strongest model per category.

Model Correct Partial Wrong span Term. mismatch Hallucination Noise
Fine-tuned baselines
Rifky_IndoBERT_QA 32.0 14.0 0.0 2.0 0.0 52.0
XLM-R-large 10.0 14.0 0.0 0.0 0.0 76.0
mDeBERTa-v3-base 12.0 16.0 0.0 0.0 0.0 72.0
Closed-source LLM (5-shot)
GPT-4o-mini 30.0 70.0 0.0 0.0 0.0 0.0
GPT-4o44.052.0 0.0 0.0 0.0 4.0
Open-source multilingual LLM (5-shot)
Llama-3.1-8B-Instruct 36.0 62.0 0.0 0.0 0.0 2.0
Qwen2.5-7B-Instruct 30.0 70.0 0.0 0.0 0.0 0.0
Indonesian-tuned LLM (5-shot)
Sahabat-AI-Llama3-8B 28.0 58.0 4.0 8.0 0.0 2.0
Merak-7B-v4 26.0 68.0 6.0 0.0 0.0 0.0
English-medical LLM (5-shot)
BioMistral-7B 16.0 72.0 6.0 0.0 0.0 6.0
MedGemma-4B-IT 38.0 62.0 0.0 0.0 0.0 0.0
Table 17: Error taxonomy over a sampled subset of n= 50 test examples (seed 42), classified by GPT-4o-mini.
Cells show % of the 50sampled outputs per row; each row sums to 100% . Bold marks the strongest model on
correct. The Hallucination column is a judge label that human audit shows to under-detect unsupported content
(Section 6.8); it is not evidence of absence.
Disease categorySource reports QA pairs
n%n%
Respiratory34 28.33 273 29.97
Infectious21 17.50 157 17.23
Gastrointestinal15 12.50 104 11.42
Nutrition/growth13 10.83 92 10.10
Neurologic12 10.00 86 9.44
Cardiac10 8.33 74 8.12
Hematologic/oncologic10 8.33 85 9.33
Other3 2.50 22 2.41
Renal/urologic1 0.83 11 1.21
ENT1 0.83 7 0.77
Total120 100.00 911 100.00
Table 18: Disease-category distribution of IndoNutriQA.
Each source report is assigned a single dominant cate-
gory, so the columns partition the corpus; percentages
are rounded and may not sum exactly to 100. The same
categorisation backs the stratified results in Section 6.7.
Respiratory
34 (28.3%)Infectious
21 (17.5%)
Gastro-
intestinal
15 (12.5%)Nutrition/
Growth
13 (10.8%)Neurologic
12 (10.0%)Cardiac
10 (8.3%)Hematologic/
Oncologic
10 (8.3%)Other
3 (2.5%)Renal/Urologic
1 (0.8%)ENT
1 (0.8%)Disease-category distribution across 120 source reports (911 QA pairs)
Figure 4: Distribution of disease categories in IndoNu-
triQA.4.Bagaimana menyusun diagnosis gizi dengan
format PES (Problem, Etiology, Sign and
Symptom) pada pasien ini?
How is the nutrition diagnosis constructed in
PES (Problem, Etiology, Sign and Symptom)
format for this patient?
5. Apa tujuan diet pada pasien ini?
What are the dietary goals for this patient?
6.Bagaimana prinsip diet yang tepat sesuai den-
gan kondisi pasien ini?
What are the appropriate dietary principles
according to this patient’s condition?
7.Bagaimana syarat diet yang harus dipenuhi
dalam pengaturan diet pasien ini?
What dietary requirements must be fulfilled in
this patient’s diet management?
8.Apa pedoman diet yang digunakan dalam
perencanaan diet pasien ini?
What dietary guidelines are used in planning
the patient’s diet?
J Example Question-Answer Instance
This subsection presents representative example of
question-answer instance from the IndoNutriQA
dataset. The example includes the original Indone-
sian text accompanied by an English translation to
facilitate understanding for non-Indonesian readers.
All personally identifiable information has been
anonymized.

0 50 100 150 200
Age (months)01020304050Frequency43.8%
10.9%
7.0%7.8%
3.9%7.0%9.4%
4.7%3.9%
1.6%(a) Age (months)
Male
Female
Sex010203040506070Count58.4%
41.6%
(b) Sex
40 60 80 100 120 140 160
Height_cm010203040506070Weight_kgAge_months
0
40
80
120
160
200
(c) Weight vs height
5 10 15 20 25 30 35
MUAC (cm)0.02.55.07.510.012.515.017.5Count
6.0%26.9%
20.9%25.4%
10.4%
3.0% 3.0%
1.5% 1.5% 1.5%
0.000.020.040.060.08
Density
(d) MUAC (LILA)
0.0 0.2 0.4 0.6 0.8 1.0
Syntactic divergence01020304050Percentage
(e) Syntactic divergence
Figure 5: Demographic and anthropometric character-
istics of IndoNutriQA cases, and distribution of token-
level syntactic divergence between model predictions
and reference answers.Clinical Context (Excerpt).
Report Title:Laporan Praktik Kerja Profesi Pro-
gram Studi Profesi Dietisien Rotasi Gizi Klinik
Sub Rotasi Rawat Inap (Stase PICU/NICU)
Professional Clinical Internship Report for the
Dietitian Profession Program Clinical Nutrition
Rotation – Inpatient Sub-Rotation (PICU/NICU
Stage)
Anthropometric Assessment:Jenis kelamin:
Laki-laki. Usia: 1 tahun 1 bulan. Panjang badan
(PB): 65 cm. BBI/U menurut PB: 8,8 kg.
Sex: Male. Age: 1 year 1 month. Body length
(PB): 65 cm. Ideal body weight for length: 8.8
kg.
Medical Diagnosis:Severe Obstructive Hydro-
cephalus post repair VP shunt + Labiopalatoschi-
sis + Cerebral Palsy (CP) + Pneumonia.
Context:Skrining gizi bertujuan untuk menge-
tahui apakah pasien berisiko malnutrisi atau tidak.
Beberapa alat skrining gizi yang dapat digu-
nakan pada pasien anak-anak adalah Strong Kids,
PYMS, SGNA, dan SPNRS.
Nutritional screening aims to determine whether
the patient is at risk of malnutrition. Several
screening tools that can be used for pediatric
patients include Strong Kids, PYMS, SGNA, and
SPNRS.
Question.Apa metode skrining status gizi yang
direkomendasikan?
What nutritional screening method is recom-
mended?
Answer (Extractive).Beberapa alat skrining
gizi yang dapat digunakan pada pasien anak-anak
adalah Strong Kids, PYMS, SGNA, dan SPNRS.
Several nutritional screening tools that can be
used for pediatric patients are Strong Kids, PYMS,
SGNA, and SPNRS.