# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar

**Published**: 2025-08-20 16:09:37

**PDF URL**: [http://arxiv.org/pdf/2508.14817v1](http://arxiv.org/pdf/2508.14817v1)

## Abstract
Electronic health records (EHRs) are long, noisy, and often redundant, posing
a major challenge for the clinicians who must navigate them. Large language
models (LLMs) offer a promising solution for extracting and reasoning over this
unstructured text, but the length of clinical notes often exceeds even
state-of-the-art models' extended context windows. Retrieval-augmented
generation (RAG) offers an alternative by retrieving task-relevant passages
from across the entire EHR, potentially reducing the amount of required input
tokens. In this work, we propose three clinical tasks designed to be replicable
across health systems with minimal effort: 1) extracting imaging procedures, 2)
generating timelines of antibiotic use, and 3) identifying key diagnoses. Using
EHRs from actual hospitalized patients, we test three state-of-the-art LLMs
with varying amounts of provided context, using either targeted text retrieval
or the most recent clinical notes. We find that RAG closely matches or exceeds
the performance of using recent notes, and approaches the performance of using
the models' full context while requiring drastically fewer input tokens. Our
results suggest that RAG remains a competitive and efficient approach even as
newer models become capable of handling increasingly longer amounts of text.

## Full Text


<!-- PDF content starts -->

Evaluating Retrieval-Augmented Generation vs. Long-Context Input for
Clinical Reasoning over EHRs
Skatje Myers1, Dmitriy Dligach2, Timothy A. Miller3,4, Samantha Barr1,
Yanjun Gao5,Matthew Churpek1,Anoop Mayampurath1,Majid Afshar1,
1University of Wisconsin-Madison,2Loyola University Chicago,3Boston Children’s Hospital,
4Harvard Medical School,5University of Colorado-Anschutz
Correspondence: skatje.myers@wisc.edu
Abstract
Electronic health records (EHRs) are long,
noisy, and often redundant, posing a major
challenge for the clinicians who must navi-
gate them. Large language models (LLMs)
offer a promising solution for extracting and
reasoning over this unstructured text, but the
length of clinical notes often exceeds even state-
of-the-art models’ extended context windows.
Retrieval-augmented generation (RAG) offers
an alternative by retrieving task-relevant pas-
sages from across the entire EHR, potentially
reducing the amount of required input tokens.
In this work, we propose three clinical tasks
designed to be replicable across health systems
with minimal effort: 1) extracting imaging pro-
cedures, 2) generating timelines of antibiotic
use, and 3) identifying key diagnoses. Using
EHRs from actual hospitalized patients, we
test three state-of-the-art LLMs with varying
amounts of provided context, using either tar-
geted text retrieval or the most recent clinical
notes. We find that RAG closely matches or
exceeds the performance of using recent notes,
and approaches the performance of using the
models’ full context while requiring drastically
fewer input tokens. Our results suggest that
RAG remains a competitive and efficient ap-
proach even as newer models become capa-
ble of handling increasingly longer amounts of
text.
1 Introduction
Electronic health records (EHRs) contain compre-
hensive documentation of patient care, including
critical information for diagnosis and treatment
planning. However, the volume of clinical notes
has exploded in recent years, driven in part by copy-
paste practices, templated documentation, and reg-
ulatory pressures—a phenomenon often referred to
as “note bloat”. For example, nearly 1 in 5 patients
arrive at the emergency department with a chart the
size of Moby Dick (over 200K words) (Pattersonet al., 2024). As a result of this, clinicians must nav-
igate increasingly lengthy and redundant records
to locate key information. Large language models
(LLMs) can potentially alleviate this burden by as-
sisting clinicians in quickly extracting information
and reasoning over EHR, and have demonstrated
promising capabilities in clinical summarization
(Van Veen et al., 2024) and question answering
(Singhal et al., 2025). However, the sheer volume
of clinical documentation can exceed most LLMs’
context window size. A practical approach is to
provide the most recent notes, which may suffice
for some tasks but risks omitting crucial informa-
tion buried in earlier documentation.
Retrieval-augmented generation (RAG) has
emerged as a prominent solution to using LLMs
on long documents by retrieving only the most rel-
evant text passages for a given task. Rather than
processing entire patient charts, RAG systems can
selectively extract pertinent clinical information
to answer specific questions. This approach can
potentially reduce computational costs, improve ac-
curacy through elimination of noise, and mitigate
the “lost-in-the-middle” effect (Liu et al., 2024),
where model performance degrades when relevant
information is buried within lengthy contexts.
However, there has been limited empirical evalu-
ation on the accuracy and token efficiency of this re-
trieval approach for tasks that require longitudinal
reasoning over real-world EHR data. One barrier
is the scarcity of large, annotated clinical datasets
due to legal and ethical constraints regarding pa-
tient privacy. While the MIMIC datasets (Johnson
et al., 2016) have been further annotated for bench-
marking a variety of natural language processing
tasks, including question-answering, this data is
restricted to the patients’ ICU stay, as opposed to
the full hospital course, limiting their potential for
testing realistic use cases that stretch the token lim-
itations of LLMs’ processing abilities.
To address these gaps, we define three tasks that
1arXiv:2508.14817v1  [cs.CL]  20 Aug 2025

Figure 1: Retrieval-augmented generation pipeline for clinical question answering over EHR.
reflect different clinical reasoning demands and
can be replicated in other health systems without
labor-intensive manual annotation:
•Imaging Procedures : Produce a list of imag-
ing procedures (including modality, date, and
anatomical location) that occurred during a
hospitalization from the raw clinical notes.
•Antibiotic Timelines : Generate the timelines
of therapeutic antibiotic use for patients with
a severe infection.
•Diagnosis generation : Identify the key diag-
noses relevant to the hospitalization.
The Imaging Procedures task is a straightfor-
ward extractive task, requiring the model to identify
the imaging procedures that occurred on different
days across the course of the hospital stay. The An-
tibiotic Timelines task requires not only identifying
the antibiotics the patient was on and when they
were discontinued, but also incorporating medical
reasoning to determine what those antibiotics were
administered for. The final task, Diagnosis Gen-
eration requires the most medical reasoning—the
model is asked not just to list the diagnoses that
were mentioned, but determine which required ac-
tive management and impacted the care plan.
These tasks allow us to investigate the following
questions: Given a limited token budget, to what
extent can targeted retrieval of information from
the full hospital stay improve efficiency and per-
formance over simply providing an LLM with the
most recent notes? Does utilizing the extremely
long context windows of state-of-the-art models
provide any further benefit?
Using EHR data from an academically-affiliated
US hospital, we evaluate three LLMs on these tasksusing varying amounts of clinical context, includ-
ing up to the models’ full context window of 128K
tokens.
Our findings suggest that while RAG can provide
substantial efficiency improvements over compa-
rable amounts of recent clinical note tokens, this
effect is highly task-dependent. However, in all
three tasks, we found RAG to achieve near-parity
to using the full context window with a fraction
of the tokens, indicating that retrieval remains a
competitive approach even as newer model archi-
tectures continue to extend context windows.
2 Related Work
Our goal in proposing these tasks is to 1) re-
quire synthesis of information distributed across
the EHR, rather than in a single location, and 2)
provide evaluation methods tailored to the tasks,
and 3) allow use on any EHR systems, rather than
a single publicly released dataset such as MIMIC
(Johnson et al., 2016), which some models train
on.
Many of the existing question-answering
datasets for EHR focus on fact extraction. Datasets
such as EmrQA (Pampari et al., 2018) and
DrugEHRQA (Bardhan et al., 2022) are semi-
automatically constructed by leveraging previous
annotations from National NLP Clinical Chal-
lenges (n2c2) to transform them into question-
answer pairs. For this type of data, template ques-
tions are constructed where the annotation can fill a
slot, such as “What is the dosage of |medication|?”.
The MedAlign dataset (Fleming et al., 2024)
is comprised of clinician-generated instruction-
answer pairs and longitudinal EHR. Many of the
instructions are yes/no questions that can be an-
swered by retrieving a single piece of evidence
(e.g. “Does she smoke?”), but some instructions
2

Imaging Antibiotics Diagnosis
Hospitalizations 200 200 200
Mean notes per hospitalization 110 145 111
Tokens per hospitalization:
mean 74k 108k 75k
range 17k-401k 16k-1.4m 20k-389k
Table 1: Dataset statistics for each task.
require the model to synthesize information across
the EHR (e.g. “Summarize this patient’s cardiac
history.”). However, evaluation on open-ended
responses poses an ongoing challenge in NLP,
with popular automatic metrics such as BLEU
and ROUGE showing poor correlation with human
judgment on natural language generation tasks in
healthcare (Croxford et al., 2024).
Retrieval-augmented generation has been used
for a variety of tasks within the medical domain,
including answering open-ended medical questions
by retrieving from medical guidelines and journal
articles (Zakka et al., 2024) and assessing surgical
fitness by retrieving from perioperative guidelines
(Ke et al., 2025).
Alkhalaf et al. (2024) used RAG to generate
structured summaries by retrieving from EHR,
querying for relevant text using the names of the
summary fields (such as “age” and “weight”).
3 Data and Models
We constructed datasets of 200 inpatient hospital-
izations for each of our three tasks using data from
a US hospital system, comprised of clinical notes
from admission to discharge (daily progress notes,
specialist consultations, imaging reports, etc.). Ta-
ble 1 provides summary statistics. All hospitaliza-
tions were at least 7 days long and were comprised
of at least 15,000 tokens of clinical notes. For
theImaging Procedures andDiagnosis Generation
tasks, only the clinical notes prior to the discharge
summary are used to provide information to the
LLM, to avoid leaking information from the hos-
pital course or diagnosis sections of the discharge
summary. For the Antibiotic Timelines task, all
included hospitalizations involved a consultation
with Infectious Diseases and only the notes prior to
the consultation note are included in the data that
may be presented to the LLM.
The only structured EHR data provided to the
system are the notes’ timestamp and type (e.g.
progress note, handoff, etc.).
We evaluated three state-of-the-art LLMs capa-Task Retrieval query
Imaging ProceduresX-ray, CT, MRI, ultrasound, NM imaging,
echocardiogram, fluoroscopy
Antibiotic Timelines What antibiotics is the patient taking?
Diagnosis Generation What are the patient’s diagnoses?
Table 2: Queries used for retrieving relevant text pas-
sages. Queries were prepended with “Represent this
sentence for searching relevant passages:”, in accor-
dance with recommended usage with the BGE embed-
ding model.
ble of processing up to 128K tokens:
•o4-mini (OpenAI, 2025)
•GPT-4o-mini (OpenAI, 2024)
•DeepSeek-R1 (Guo et al., 2025)
4 RAG System
For each patient hospitalization, clinical notes were
segmented into overlapping 128-token chunks,
with a sliding window of 20. These chunks were
embedded using BGE-en-large-v1.5 (Xiao et al.,
2023), a popular general-purpose BERT-based em-
bedding model trained through contrastive learn-
ing. We selected this model based on findings from
Myers et al. (2025), who conducted an ablation
study of embedding models and pooling strategies
for EHR retrieval and found BGE-en-large-v1.5
to significantly outperform general-domain and
biomedical-domain alternatives on several retrieval
tasks over EHR.
For each task, we manually crafted a simple
query for retrieving relevant passages (Table 2). We
used cosine similarity between the query and each
chunk to retrieve the top-N most relevant passages
(N = 20, 40, 60). These chunks were inserted into
the instruction prompt (Appendix B) and passed to
the LLM.
We compared this retrieval configuration to a
baseline approach of providing the most recent
clinical notes in comparable amounts, up to 3K,
5.5K, or 8K tokens (including prompt) and long-
context inputs with up to 64K or 128K tokens. Ref-
erences to these token amounts throughout this
study should be understood as an upper bound , as
some encounters consist of fewer tokens, reflective
of the underlying hospitalization distribution.
Performance on the tasks was evaluated using
either F1 or Jaccard index, as described in the fol-
lowing sections, and we assessed the comparative
3

performance between the RAG and non-RAG ap-
proaches over the increasing number of tokens by
calculating the area under the curves and reporting
the normalized area difference.
5 Task 1: Imaging Procedures
5.1 Methods
TheImaging Procedures task involves extracting
structured information about diagnostic imaging
procedures from unstructured clinical notes. We
focused on five common imaging modalities: Mag-
netic Resonance Imaging (MRI), Computed To-
mography (CT), Ultrasound, X-ray, and Nuclear
Medicine (NM) Imaging. The model was prompted
to produce a list of imaging procedures that oc-
curred during the hospitalization, giving the modal-
ity, anatomical location, and date.
As a gold standard, we used tabular procedure
records from the EHR. We mapped these procedure
descriptions to imaging modality and anatomical
site using simple rules and regular expressions. For
example:
X-RAY CHEST 2 VIEWS
modality: "X-ray"
location: "chest"
CT LUMBAR SPINE W/O IV CONTRAST
modality: "CT"
location: "lumbar spine"
Evaluation metrics are reported for three levels
of strictness:
• M ODALITY +DATE+LOCATION
• M ODALITY +DATE
• M ODALITY +DATE(±1 DAY)
In the lattermost case, we allow for a reasonable
tolerance in the predicted date, due to observed
variation in the reported metadata times for the
procedure and note. For example, the timestamp
for the note may reflect the date it was filed into the
system, rather than the date it was actually written.
It should also be noted that the anatomical lo-
cation is not normalized, other than for capitaliza-
tion—under the strictest metric, predicting simply
“spine” for the above example would not be deemed
a positive match.
5.2 Results
Across all three models and evaluation methods,
RAG yielded dramatic performance improvements.GPT-4o-mini o4-mini DeepSeek R1
MODALITY +DATE+LOCATION 552.3% 425.3% 430.6%
MODALITY +DATE 432.0% 375.0% 364.3%
MODALITY +DATE(±1 DAY) 406.9% 382.8% 378.0%
Table 3: Normalized area difference between the RAG
and Recent Notes curves for the Imaging Procedures
task.
In Figure 2, we show the classification performance
forMODALITY +DATE(±1DAY)across varying
amounts of provided EHR context. We calculated
the normalized area difference between the curves
for the overlapping token amounts, presented in
Table 3. We found at minimum a 3.75-fold perfor-
mance gain against using similar amounts of the
most recent notes. These results also demonstrate
that targeted retrieval of passages can closely ap-
proach the performance of utilizing the full context
window with only a fraction of the tokens: Using
only 60 retrieved chunks, GPT 4o-mini, o4-mini,
and DeepSeek R1 only fell short by 2.43, 5.56, and
1.72, respectively.
These findings are similarly reflected under the
stricter evaluation conditions. A complete listing
of precision, recall, and F1 can be found in the
Appendix in Table 6.
6 Task 2: Antibiotic Timelines
6.1 Methods
This task emulates the work performed by Infec-
tious Diseases (ID) physicians to document the
antibiotic regimen for an active infection. When
these specialists are consulted, they document the
history of the present illness, including lab results
and medications, as well as outline a treatment
plan. This note typically contains a “History of
Anti-Infectives” section, where they list the antibi-
otics that have been used to treat the infection of
concern, omitting prophylactic or non-relevant anti-
infectives. For example:
Vancomycin: 1/16-present
Ceftriaxone: 1/17-present
These medication names and date ranges are
manually annotated by the specialist after review-
ing the patient’s chart and serve as our ground truth
for this task, after extraction using regular expres-
sions. No notes authored by ID physicians were
included in the data that was presented to the model,
and only notes that were written prior to the ground
truth note were made available.
4

Figure 2: Imaging Procedures : F1 scores for the three models using the MODALITY +DATE(±1DAY)evaluation
method, across varying amounts of provided EHR tokens.
We evaluated system accuracy with the follow-
ing metrics:
•MEDICATIONS (NAME ONLY ): Classification
accuracy of only the medications, disregard-
ing timespans.
•TIMESPAN OVERLAP : The overlap between
the predicted and gold date ranges for each an-
tibiotic, reported using Jaccard index. A value
of 1 indicates an exact match; 0 indicates no
overlap, missing a medication entirely, or in-
cluding a medication not present in the gold
standard. Values are averaged over the dataset.
The medications in both the generated predic-
tions and the gold data are normalized to their in-
gredients using the RxNorm (Nelson et al., 2011)
API provided by the National Library of Medicine
and a handful of manual rules for edge cases,
such as typos. This allows for accurate matching
of generic and brand name medications, such as
Zosyn (piperacillin and tazobactam).
As a baseline, we used a rule-based approach
of directly extracting the time ranges for all med-
ications of the “anti-infective” therapeutic class
from the list of administered medications using the
EHR’s medication administration record (MAR),
a tabular form ubiquitous to EHRs for tracking
all medications and infusions. However, this list
of medications includes those used to treat otherGPT-4o-mini o4-mini DeepSeek R1
MEDICATIONS (NAME ONLY ) 39.35% 41.4% 43.1%
TIMESPAN OVERLAP 34.7% 30.3% 32.9%
Table 4: Normalized area difference between the RAG
and Recent Notes curves for the Antibiotic Timelines
task.
conditions that were not the focus of the ID consul-
tation. By formulating this task to replicate the ID
specialists’ work, rather than on replicating struc-
tured data as the Imaging Procedures did, this task
requires an additional level of medical reasoning to
accurately conform to inclusion criteria.
6.2 Results
Figure 3 shows the performance of the models
forTIMESPAN OVERLAP . The RAG approach
consistently exceeds the rule-based baseline and
demonstrates close performance to the peak perfor-
mance using large amounts of recent notes (using
60 chunks, GPT 4o-min: -0.020, o4-mini: -0.075,
DeepSeek R1: -0.012). The performance of the
RAG approach only sees slight gains from increas-
ing the amount of retrieved text.
For two of the models, the average Jaccard index
drops slightly when increasing the maximum pro-
vided context from 64K to 128K tokens (GPT 4o-
mini: -0.032, o4-mini: -0.049), while for DeepSeek
R1, the additional data provides a negligible in-
crease of 0.006.
5

Figure 3: Antibiotic Timelines : Average Jaccard index for the three models using the TIMESPAN OVERLAP
evaluation method, across varying amounts of provided EHR tokens.
On the task of predicting MEDICATIONS (NAME
ONLY ), the RAG approach slightly outperforms us-
ing the full context window with only 60 retrieved
chunks (GPT 4o-mini: +2.14, o4-mini: +2.55,
DeepSeek R1: +4.75). A complete listing of preci-
sion, recall, and F1 can be found in the Appendix
in Table 7.
7 Task 3: Diagnosis Generation
7.1 Methods
The goal of this task is to generate a list of diag-
noses for a given hospitalization that is of primary
relevance to the clinician.
We drew from two EHR sources to construct our
gold labels for each hospitalization:
•DISCHARGE SUMMARY : The free text from
the discharge summary that lists the primary
and secondary diagnoses.
•BILLING CODES : The lists of International
Classification of Diseases (ICD-10) codes
from the structured EHR for the hospitaliza-
tion, manually annotated by billing coders.
The diagnoses annotated by the coders are guar-
anteed to be documented within the notes, provide
a high degree of specificity, and are normalized to
a standard vocabulary, but these lengthy lists often
include diagnoses that are not necessarily consid-
ered to be of key importance to clinicians, such asa history of smoking, obesity, or minor issues such
as a contusion. On the other hand, the text in the
discharge summary does not necessarily list spe-
cific diagnoses, such as noting post-surgical status
(e.g. “S/p kidney transplant”) or leaving some diag-
noses undocumented because they can be inferred.
For example, documenting that the patient is post-
kidney transplant and has complications, but not
that they have kidney disease.
While discharge summaries reflect the informa-
tion of most clinical relevance to the clinicians
providing treatment, the billing codes lend them-
selves better to validation due to their standard-
ization. To produce a more balanced representa-
tion, we instructed Gemma-3-32B (Google, 2005)
to filter the billing ICD code lists to only the en-
tries that reflect the clinician’s primary foci for the
hospital stay based on the text from the discharge
summary. This FILTERED list of ICD-10 codes
serves as our primary evaluation target and the in-
struction prompt was designed to elicit this list by
outlining inclusion and exclusion criteria for diag-
noses (e.g. include acute conditions requiring ICU
care, exclude stable chronic conditions or irrelevant
historical diagnoses).
To enable classification evaluation, the free text
generated by the LLM and from the discharge sum-
mary need to be normalized to ICD-10 codes. For
this process, we trained SNOBERT (Kulyabin et al.,
2025) to extract Systematized Nomenclature of
6

GPT-4o-mini o4-mini DeepSeek R1
BILLING CODES 4.32% -0.94% 4.83%
DISCHARGE SUMMARY -6.18% -7.31% -4.99%
FILTERED -4.08% -4.05% -1.47%
Table 5: Normalized area difference between the RAG
and Recent Notes curves for the Diagnosis Generation
task.
Medicine (SNOMED) concepts from the text1and
used the mappings provided by the SNOMED Clin-
ical Terms data release to convert them to ICD-10.
ICD-10 is a hierarchical vocabulary, ranging
from broad concepts (e.g. “Anemia, unspeci-
fied” [D64.9]) to highly specific (e.g. “Age-
related osteoporosis with current pathological frac-
ture, right shoulder, initial encounter for fracture”
[M80.011A]). Due to this high granularity, eval-
uating this task requires a more fuzzy matching
technique, rather than evaluating classification ac-
curacy on the ICD codes themselves. For this pur-
pose, we employed the Healthcare Cost and Uti-
lization Project’s Clinical Classifications Software
Refined (CCSR) (Agency for Healthcare Research
and Quality, 2025). The CCSR provides a mapping
from ICD-10 codes to about 530 clinically relevant
categories.
CCSR is a many-to-many mapping, which en-
ables mapping very fine-grained ICD codes such as
“Hypertensive chronic kidney disease” to multiple
CCSR categories: “Hypertension with complica-
tions and secondary hypertension" and “Chronic
kidney disease”. This allowed us to consider pre-
dicted diagnoses to be a match even if the LLM
split them into “Hypertension” and “Chronic kid-
ney disease”.
Some broader non-billable ICD codes are not
included in the CCSR mapping (e.g. “Hypotension”
[I95]). In these cases, we used the set intersection
of the CCSR categories that the ICD code’s sub-
categories (e.g. I95.3, I95.89, etc.) are mapped
to.
7.2 Results
Unlike the previous two tasks, we do not see a
consistent improvement in performance for us-
ing RAG compared to comparable amounts of re-
cent notes, shown in Figure 4 and Table 5, but
actually a slight decrease, other than evaluating
against the BILLING CODES using GPT 4o-mini
and DeepSeek R1. However, the performance us-
1Training details are provided in Appendix Cing very long contexts is not substantially higher
than that of using fewer tokens. Overall, perfor-
mance is relatively flat across models and data se-
lection approaches and does not reach any higher
than an F1 score of 44.41. For the FILTERED list
that serves as our primary target, scores across all
context selection methods for GPT 4o-mini, o4-
mini, and DeepSeek R1 all fell within the small
ranges of 5.18, 4.91, and 4.86, respectively.
For both FILTERED and DISCHARGE SUM-
MARY targets, performance using the most recent
notes is detrimented by using very large context
amounts, while performance on BILLING CODES
demonstrates additional benefit from the additional
text (though only up to 64K for o4-mini and
DeepSeek R1). A complete listing of precision,
recall, and F1 can be found in the Appendix in
Table 8.
8 Discussion
TheImaging Procedures task, which requires rela-
tively shallow extraction of information data from
the clinical notes, demonstrated the clearest benefit
from RAG, and the performance gains from re-
trieval were both substantial and consistent across
models.
TheAntibiotic Timelines task introduces greater
complexity, requiring both temporal reasoning and
clinical understanding to distinguish therapeutic an-
tibiotics from incidental medications. While RAG
also provided a significant improvement over us-
ing only recent notes, performance gains plateaued
quickly—suggesting that only a limited number of
passages are needed to reconstruct the key temporal
history when performing targeted retrieval.
Error analysis for this task draws attention to one
of the limitations to be encountered when designing
tasks on longitudinal EHR. In 22.4% of the gold
medications analyzed, the information needed to
generate the gold standard medication and precise
timespan is not present in the full clinical notes.
Most often this occurs due to the patient initially
being admitted to another hospital and then trans-
ferred to our health system. This incomplete pic-
ture of a patient’s history is hard to avoid when
constructing datasets to capture longitudinal EHR,
as patients don’t exclusively visit a single health-
care system and healthcare data governance creates
barriers to accessing to this external information.
Additionally, when retrieving only 20 chunks, rel-
evant information that could’ve improved perfor-
7

Figure 4: Diagnosis Generation : F1 scores for the three models using the FILTERED evaluation method, across
varying amounts of provided EHR tokens.
mance was missed for 32% of gold medications,
contributing to further performance degradation.
The Diagnosis Generation task presented the
greatest challenge, as it is also a very subjective
task. Physicians can vary in documentation prac-
tices and what is chosen to be included in the dis-
charge summary – an inherent limitation in auto-
matic evaluation of this task. Performance varied
by the evaluation target, with the BILLING CODES
list benefiting the most from additional text, likely
due to this being a more exhaustive list of diag-
noses to capture everything that can be billed. The
fact that all scores, regardless of retrieval method
and consistently across the models, fell within low,
narrow ranges points towards performance reach-
ing a ceiling caused by limitations of the task or
evaluation method.
Across all tasks and models, we observed a
consistent trend: retrieval-augmented generation
was able to closely match the performance of full-
context inputs with far fewer tokens.
9 Conclusion
In this work, we introduced three clinically rele-
vant tasks designed to evaluate the effectiveness
of retrieval-augmented generation across varying
information demands in electronic health records.
Each task was selected for its clinical relevance,
reproducibility across health systems, and varying
degrees of reasoning complexity.Our results demonstrate that a targeted retrieval
approach can reach near parity with using up to
128K tokens of recent clinical notes on these three
tasks, while requiring significantly fewer input to-
kens. These findings show RAG’s continued value
even as LLMs grow more capable of processing
long sequences. Further tuning the retrieval ap-
proach (queries, embedding model, retrieving more
than 60 chunks, etc.), may close the remaining gap.
Future work should explore additional tasks that
can be devised without extensive manual effort and
informed by clinical workflows and documenta-
tion practices in order to provide a more robust
assessment of models and retrieval methods over
longitudinal EHR tasks.
Limitations
Due to legal and ethical restrictions, we cannot
release the datasets used in this study. However,
we have designed the tasks to be reproducible on
other EHR systems using structured metadata and
simple regex-based extraction of text from standard
clinical note types.
Our evaluated RAG implementation uses a fixed
chunking strategy, query formulation, and embed-
ding model. Retrieval performance is sensitive to
these parameters, and alternative configurations
may yield different results.
Additionally, our evaluation of Diagnosis Gen-
eration depends on normalizing free text to ICD
8

codes, which we do through a trained model identi-
fying SNOMED codes before using manually writ-
ten mappings. Less-than-perfect performance of
this model may have introduced some noise to this
evaluation.
References
Agency for Healthcare Research and Quality. 2025.
Clinical Classifications Software Refined (CCSR)
for ICD-10-CM Diagnoses — hcup-us.ahrq.gov.
https://hcup-us.ahrq.gov/toolssoftware/
ccsr/dxccsr.jsp . [Accessed 17-07-2025].
Mohammad Alkhalaf, Ping Yu, Mengyang Yin, and
Chao Deng. 2024. Applying generative ai with
retrieval augmented generation to summarize and
extract key clinical information from electronic
health records. Journal of biomedical informatics ,
156:104662.
Jayetri Bardhan, Anthony Colas, Kirk Roberts, and
Daisy Zhe Wang. 2022. DrugEHRQA: A question an-
swering dataset on structured and unstructured elec-
tronic health records for medicine related queries. In
Proceedings of the Thirteenth Language Resources
and Evaluation Conference , pages 1083–1097, Mar-
seille, France. European Language Resources Asso-
ciation.
Emma Croxford, Yanjun Gao, Brian Patterson, Daniel
To, Samuel Tesch, Dmitriy Dligach, Anoop Mayam-
purath, Matthew M. Churpek, and Majid Afshar.
2024. Development of a Human Evaluation Frame-
work and Correlation with Automated Metrics for
Natural Language Generation of Medical Diagnoses.
AMIA ... Annual Symposium proceedings. AMIA Sym-
posium , 2024:309–318.
Scott L Fleming, Alejandro Lozano, William J
Haberkorn, Jenelle A Jindal, Eduardo Reis, Rahul
Thapa, Louis Blankemeier, Julian Z Genkins, Ethan
Steinberg, Ashwin Nayak, and 1 others. 2024.
Medalign: A clinician-generated dataset for instruc-
tion following with electronic medical records. In
Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 22021–22030.
Google. 2005. Introducing Gemma 3: The most capa-
ble model you can run on a single GPU or TPU —
blog.google. https://blog.google/technology/
developers/gemma-3/ .
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-R1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 .
Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H
Lehman, Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits, Leo Anthony Celi,and Roger G Mark. 2016. Mimic-iii, a freely accessi-
ble critical care database. Scientific data , 3(1):1–9.
Yu He Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal
Abdullah, Nan Liu, Alex Tiong Heng Sia, Chai Rick
Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong,
Chang-Fu Kuo, and 1 others. 2025. Retrieval aug-
mented generation for 10 large language models and
its generalizability in assessing medical fitness. npj
Digital Medicine , 8(1):187.
Mikhail Kulyabin, Gleb Sokolov, Aleksandr Galaida,
Andreas Maier, and Tomas Arias-Vergara. 2025.
SNOBERT: A benchmark for clinical notes entity
linking in the SNOMED CT clinical terminology. In
Pattern Recognition , pages 154–163, Cham. Springer
Nature Switzerland.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Skatje Myers, Timothy A Miller, Yanjun Gao,
Matthew M Churpek, Anoop Mayampurath, Dmitriy
Dligach, and Majid Afshar. 2025. Lessons learned on
information retrieval in electronic health records: a
comparison of embedding models and pooling strate-
gies. Journal of the American Medical Informatics
Association , 32(2):357–364.
Stuart J Nelson, Kelly Zeng, John Kilbourne, Tammy
Powell, and Robin Moore. 2011. Normalized names
for clinical drugs: Rxnorm at 6 years. Journal
of the American Medical Informatics Association ,
18(4):441–448.
OpenAI. 2024. GPT-4o mini: advanc-
ing cost-efficient intelligence — ope-
nai.com. https://openai.com/index/
gpt-4o-mini-advancing-cost-efficient-intelligence/ .
[Accessed 24-07-2025].
OpenAI. 2025. Introducing OpenAI o3 and o4-
mini — openai.com. https://openai.com/index/
introducing-o3-and-o4-mini/ . [Accessed 24-
07-2025].
Anusri Pampari, Preethi Raghavan, Jennifer Liang, and
Jian Peng. 2018. emrQA: A large corpus for question
answering on electronic medical records. In Proceed-
ings of the 2018 Conference on Empirical Methods
in Natural Language Processing , pages 2357–2368,
Brussels, Belgium. Association for Computational
Linguistics.
Brian W Patterson, Daniel J Hekman, Frank J Liao,
Azita G Hamedani, Manish N Shah, and Majid Af-
shar. 2024. Call me dr ishmael: trends in elec-
tronic health record notes available at emergency
department visits and admissions. JAMIA open ,
7(2):ooae039.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres,
Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin
9

Clark, Stephen R Pfohl, Heather Cole-Lewis, and
1 others. 2025. Toward expert-level medical ques-
tion answering with large language models. Nature
Medicine , 31(3):943–950.
Dave Van Veen, Cara Van Uden, Louis Blanke-
meier, Jean-Benoit Delbrouck, Asad Aali, Christian
Bluethgen, Anuj Pareek, Malgorzata Polacin, Ed-
uardo Pontes Reis, Anna Seehofnerová, and 1 others.
2024. Adapted large language models can outper-
form medical experts in clinical text summarization.
Nature medicine , 30(4):1134–1142.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding. Preprint ,
arXiv:2309.07597.
Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R
Dalal, Jennifer L Kim, Michael Moor, Robyn Fong,
Curran Phillips, Kevin Alexander, Euan Ashley,
and 1 others. 2024. Almanac—retrieval-augmented
language models for clinical medicine. Nejm ai ,
1(2):AIoa2300068.
A Additional Results
B Prompts
B.1 Imaging Procedures
# Task: Identification of Imaging Procedures
from Electronic Health Records
You are an AI assistant tasked with identifying
imaging procedures performed on a patient
during their hospital stay. You will be
provided with relevant passages from the
patients Electronic Health Record.
## Instructions
1. Carefully read through all provided EHR
passages and identify the imaging procedures
performed and the location imaged.
2. Create a bulleted list of the imaging
procedures performed during this current
hospitalization, DO NOT include procedures
from their history that occurred prior to
this stay and DO NOT include imaging that
was only performed for guidance during
another procedure.
3. ONLY include procedures from these primary
categories: MRI (Magnetic Resonance Imaging)
, CT (Computed Tomography), Ultrasound (US),
X-ray (Radiograph), NM Imaging (Nuclear
Medicine)
4. For each procedure found, you must identify:
- The primary imaging modality (from the list
above)
- Whether there is a more specific subtype of
the modality (e.g., Fluoroscopy, Mammography
, Echocardiography, PET, Angiography)
- The time of the imaging (as MM/DD format or "
unknown" if this cannot be determined)
- The specific body location that was imaged (e.
g., brain, chest, left ankle)
5. Ignore all other medical information such as
tests, medications, treatment plans,
assessments, other non-imaging procedures.## Output Format
- Use EXACTLY this format for each item: "- (MM/
DD or "unknown") [Primary Imaging Modality]
- [Subtype or "None"]: [Body Location]"
- Use "None" in place of subtype when no
specific subtype is necessary and "unknown"
in place of the '
date if the time of the imaging cannot be
determined.
- Present as a clean bulleted list
- Include no explanatory text, introductions, or
conclusions
- Do not number the items
- If multiple imaging procedures of the same
type were performed on different locations,
list each separately
- The same imaging occurence may be mentioned
multiple times throughout the EHR, only
include one entry per occurence.
- If no imaging procedures exist, output only: "
No imaging procedures identified."
## Example
2018-03-12 15:43:00 H&P
Patient admitted with chest pain. Cardiac
enzymes were elevated. A chest X-ray was
performed on 3/10 showing cardiomegaly. CT
scan of the chest was completed to rule out
aortic dissection. The patient also
underwent a transthoracic echocardiogram
today.
Example output:
- (03/10) X-ray - None: Chest
- (unknown) CT - None: Chest
- (03/12) Ultrasound - Echocardiogram: Heart
## Begin Task
EHR passages:
[INSERT TEXT]
Your response as a list of imaging types and
locations:
B.2 Antibiotic Timelines
# Task: Identification of Administered
Antibiotics and Date Ranges from Electronic
Health Records\n\n
You are an AI assistant tasked with identifying
antibiotics administered to a patient during
their hospital stay and the date ranges for each
antibiotic 's use. You will be
provided with relevant passages from the patient
's Electronic Health Record (EHR), each with
an
associated timestamp.\n\n
## Instructions\n
1. Carefully read through all provided EHR
passages, noting their timestamps.\n
2. Create a list of antibiotics being
administered, prescribed, or continued.\n
3. Do not include antibiotics given for
prophylaxis or minor conditions. Only
include antibiotics being used for the
treatment of the major acute condition of
the ICU patient.\n
4. For each antibiotic, determine the start and
end dates of its use by inferring from the
10

MODALITY +DATE+LOCATION MODALITY +DATE MODALITY +DATE(±1)
# chunks/tokens P R F P R F P R F
GPT 4o-miniRAG20 chunks 49.38 29.61 37.02 61.62 36.94 46.19 72.50 43.47 54.35
40 chunks 50.00 36.00 41.86 60.84 43.81 50.94 72.15 51.95 60.41
60 chunks 50.59 40.11 44.74 61.46 48.72 54.35 72.33 57.34 63.96
Recent Notes3K 23.08 1.21 2.30 35.90 1.88 3.58 46.15 2.42 4.60
5.5K 27.88 3.92 6.87 42.31 5.95 10.43 52.88 7.43 13.03
8K 35.94 6.19 10.56 51.95 8.95 15.27 62.89 10.83 18.48
64K 52.74 32.44 40.17 65.86 40.51 50.17 76.15 46.84 58.00
128K 49.54 39.43 43.91 63.91 50.87 56.65 74.89 59.62 66.39
o4-miniRAG20 chunks 54.00 31.76 40.00 69.22 40.71 51.27 82.49 48.52 61.10
40 chunks 54.55 39.17 45.59 69.54 49.93 58.13 82.29 59.08 68.78
60 chunks 55.46 42.40 48.05 71.65 54.78 62.09 84.77 64.80 73.46
Recent Notes3K 36.49 1.82 3.46 51.35 2.56 4.87 63.51 3.16 6.03
5.5K 45.73 5.05 9.09 67.07 7.40 13.33 76.83 8.48 15.27
8K 50.41 8.21 14.12 67.36 10.97 18.87 79.34 12.92 22.22
64K 58.94 36.61 45.16 76.92 47.78 58.95 89.06 55.32 68.24
128K 58.60 42.66 49.38 78.84 57.40 66.43 91.40 66.55 77.02
DeepSeek R1RAG20 chunks 52.68 31.76 39.63 69.64 41.99 52.39 83.26 50.20 62.64
40 chunks 55.87 39.70 46.42 73.20 52.02 60.82 85.89 61.04 71.36
60 chunks 54.29 42.60 47.74 73.50 57.67 64.63 85.68 67.23 75.34
Recent Notes3K 14.58 1.88 3.34 23.44 3.03 5.36 27.60 3.57 6.32
5.5K 44.91 5.05 9.07 70.06 7.87 14.16 78.44 8.82 15.85
8K 49.40 8.28 14.18 69.48 11.64 19.94 81.12 13.59 23.29
64K 55.17 38.09 45.06 73.20 50.54 59.79 84.31 58.21 68.87
128K 54.21 45.02 49.19 73.50 61.04 66.69 84.93 70.52 77.06
Table 6: Scores for the Imaging Procedures task, across different models and differing amounts of provided clinical
notes.
11

TIMESPAN OVERLAP MEDICATIONS (NAME ONLY )
# chunks/tokens Jaccard index P R F1
GPT 4o-miniRAG20 chunks 0.5030 75.44 79.32 77.33
40 chunks 0.5092 74.60 79.94 77.18
60 chunks 0.5182 75.15 80.40 77.69
Recent Notes3K 0.3573 67.94 30.33 41.94
5.5K 0.3712 75.00 47.59 58.23
8K 0.4275 77.78 57.70 66.25
64K 0.5386 76.21 76.21 76.21
128K 0.5068 73.60 77.60 75.55
o4-miniRAG20 chunks 0.5671 78.78 76.21 77.47
40 chunks 0.566 80.22 80.72 80.47
60 chunks 0.5858 79.78 80.40 80.09
Recent Notes3K 0.4024 76.95 29.17 42.31
5.5K 0.4132 80.21 47.28 59.49
8K 0.5529 80.43 57.54 67.09
64K 0.6604 80.55 72.78 76.47
128K 0.6111 80.67 74.65 77.54
DeepSeek R1RAG20 chunks 0.5814 74.81 76.21 75.50
40 chunks 0.5974 76.86 76.98 76.92
60 chunks 0.6112 77.42 78.38 77.90
Recent Notes3K 0.3489 65.22 30.33 41.40
5.5K 0.4610 72.97 46.19 56.57
8K 0.5547 71.98 55.52 62.69
64K 0.6176 71.37 75.58 73.41
128K 0.6232 69.93 76.67 73.15
Structured EHR baseline 0.4650
Table 7: Scores for the Antibiotic Timelines task, across different models and differing amounts of provided clinical
notes.
timestamps of the passages and any date
information within the text.\n
5. Use the format MM/DD-MM/DD for date ranges.
If the antibiotic use is ongoing, use \"
present\" for the end date.\n
6. Don 't include dosages or administration
routes.\n
7. If a date range can 't be determined
whatsoever, list the antibiotic with \"(
dates unclear)\" after it.\n
\n
## Output Format\n
Provide your response as a list of antibiotics
with their date ranges in the following
format:\n
- Antibiotic 1 (MM/DD - MM/DD)\n
- Antibiotic 2 (MM/DD - present)\n
## Example\n
Right now it is 2019-09-15 14:51:00.\nEHR passages:\n\n
2019-09-12 10:15:00\n
\"Patient admitted with suspected pneumonia.
Started on IV ceftriaxone 1g daily.\"\n\n
2019-09-14 14:30:00\n
\"Blood cultures positive for MRSA. Ceftriaxone
discontinued. Started on IV vancomycin 1g
q12h.\"\n\n
Output:\n
- Ceftriaxone (09/12-09/14)\n
- Vancomycin (09/14-ongoing)\n\n
## Begin Task\n
Right now it is [TIMESTAMP].\n\nEHR passages:\n\
n[INSERT TEXT]
\n\nYour response as a list of antibiotic names
and date ranges:\n
B.3 Diagnosis Generation
12

BILLING CODES DISCHARGE SUMMARY FILTERED
# chunks/tokens P R F1 P R F1 P R F1
GPT 4o-miniRAG20 chunks 57.08 24.71 34.49 29.19 44.51 35.25 31.10 39.55 34.82
40 chunks 58.17 27.32 37.18 30.19 49.95 37.64 31.90 44.01 36.99
60 chunks 57.95 28.54 38.25 30.52 52.96 38.72 32.25 46.67 38.14
Recent Notes3K 55.70 21.23 30.74 31.50 42.22 36.09 33.47 37.45 35.35
5.5K 58.96 26.53 36.60 33.59 53.16 41.17 35.14 46.42 40.00
8K 58.96 28.68 38.59 33.15 56.81 41.87 32.77 46.83 38.56
64K 58.11 31.80 41.10 31.27 60.28 41.18 32.00 51.45 39.46
128K 58.78 32.46 41.83 30.84 60.00 40.74 31.71 51.45 39.24
o4-miniRAG20 chunks 63.75 20.03 30.49 36.08 39.79 37.84 39.50 36.27 37.82
40 chunks 63.89 22.87 33.68 35.44 44.69 39.53 38.12 40.09 39.08
60 chunks 64.99 25.24 36.36 33.70 46.10 38.94 37.35 42.62 39.81
Recent Notes3K 65.55 17.74 27.92 39.09 37.25 38.15 42.05 33.44 37.25
5.5K 65.81 24.99 36.23 38.73 51.61 44.25 39.94 44.64 42.16
8K 65.16 25.67 36.83 35.99 49.95 41.84 38.57 44.64 41.38
64K 64.72 28.58 39.65 34.00 52.93 41.40 36.98 48.22 41.86
128K 64.73 27.56 38.66 35.39 52.97 42.44 37.16 46.45 41.29
DeepSeek R1RAG20 chunks 62.90 22.87 33.54 32.77 41.97 36.81 36.58 39.08 37.79
40 chunks 61.65 28.28 38.77 31.90 51.55 39.41 35.04 47.22 40.23
60 chunks 61.18 30.70 40.89 30.54 53.99 39.01 33.40 49.26 39.81
Recent Notes3K 60.23 20.95 31.09 34.02 41.69 37.47 36.55 37.35 36.95
5.5K 60.80 27.24 37.62 34.32 54.18 42.02 36.35 47.85 41.31
8K 61.59 30.94 41.19 33.05 58.50 42.24 35.07 51.76 41.81
64K 56.99 36.38 44.41 26.60 59.81 36.82 29.39 55.13 38.34
128K 56.67 33.98 42.49 27.56 58.22 37.41 30.18 53.17 38.50
Table 8: Scores for the Diagnosis Generation task, across different models and differing amounts of provided
clinical notes.
# Task: Identification of Clinically Important
Diagnoses
You are an AI assistant tasked with creating a
clinically relevant problem list for an ICU
patient 's stay. You will analyze passages of
clinical notes from their hospitalization
and identify diagnoses that required active
management or monitoring during their stay.
# Task
Review the provided clinical note passages and
generate a structured list of diagnoses that
:
1. Required active management during the
hospitalization
2. Were clinically significant to their critical
care course
3. Impacted their ICU care plan or outcomes
# Inclusion Criteria
Include diagnoses that meet ANY of these
criteria:
- Acute conditions requiring ICU-level care (e.g
., NSTEMI, septic shock, acute respiratory
failure)
- Chronic conditions requiring active management
or affecting ICU care (e.g., atrialfibrillation, COPD exacerbation)
- New diagnoses made during the encounter
- Complications that developed during the stay
- Conditions requiring monitoring or
intervention (e.g., acute kidney injury,
severe electrolyte disorders)
- Neurologic/cognitive conditions affecting ICU
care (e.g., delirium, acute stroke)
- Conditions directly related to the reason for
ICU admission
# Exclusion Criteria
Do NOT include:
- Stable chronic conditions not requiring active
management (e.g., well-controlled diabetes,
stable hypothyroidism)
- Historical diagnoses not affecting current
care (e.g., "history of appendectomy")
- Social history items (e.g., "former smoker")
- Procedural or post-surgical statuses (e.g., "s
/p CABG", "post-cholecystectomy")
- Symptoms without clear diagnoses
- Conditions that resolved prior to admission
- Incidental findings not requiring intervention
# Output Format
Present the diagnoses as a numbered list,
13

ordered by clinical priority :
1. [Primary diagnosis]
2. [Secondary diagnosis]
3. [Tertiary diagnosis]
...
# Note
- Every listed item must be a specific medical
diagnosis - Use standard medical terminology
for diagnoses
# EHR passages:\n[INSERT TEXT]\n
# Output the diagnoses that required active
management or monitoring during their ICU
stay, as instructed. Every listed item must
be a specific medical diagnosis:
C SNOBERT Training
We trained SNOBERT using the configuration pro-
vided in the authors’ Github Repository as-is with
the same training data from the SNOMED CT En-
tity Linking Challenge, but using the International
SNOMED vocabulary files from 2025, since we
did not have access to the version used for the
challenge. We trained a single model, as opposed
to their approach of ensembling six with varying
class weights and data splits for the competition,
but through expert review, we determined perfor-
mance on the downstream ICD-10 code extraction
step to be acceptable.
14