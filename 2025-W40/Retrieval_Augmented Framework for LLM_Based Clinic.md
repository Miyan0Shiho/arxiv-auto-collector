# Retrieval-Augmented Framework for LLM-Based Clinical Decision Support

**Authors**: Leon Garza, Anantaa Kotal, Michael A. Grasso, Emre Umucu

**Published**: 2025-10-01 18:45:25

**PDF URL**: [http://arxiv.org/pdf/2510.01363v1](http://arxiv.org/pdf/2510.01363v1)

## Abstract
The increasing complexity of clinical decision-making, alongside the rapid
expansion of electronic health records (EHR), presents both opportunities and
challenges for delivering data-informed care. This paper proposes a clinical
decision support system powered by Large Language Models (LLMs) to assist
prescribing clinicians. The system generates therapeutic suggestions by
analyzing historical EHR data, including patient demographics, presenting
complaints, clinical symptoms, diagnostic information, and treatment histories.
The framework integrates natural language processing with structured clinical
inputs to produce contextually relevant recommendations. Rather than replacing
clinician judgment, it is designed to augment decision-making by retrieving and
synthesizing precedent cases with comparable characteristics, drawing on local
datasets or federated sources where applicable. At its core, the system employs
a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured
narratives and codified data to support LLM-based inference. We outline the
system's technical components, including representation representation
alignment and generation strategies. Preliminary evaluations, conducted with
de-identified and synthetic clinical datasets, examine the clinical
plausibility and consistency of the model's outputs. Early findings suggest
that LLM-based tools may provide valuable decision support in prescribing
workflows when appropriately constrained and rigorously validated. This work
represents an initial step toward integration of generative AI into real-world
clinical decision-making with an emphasis on transparency, safety, and
alignment with established practices.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented Framework for
LLM-Based Clinical Decision SupportTransportation Research Record
2020, Vol. XX(X) 1–11
©National Academy of Sciences:
Transportation Research Board 2020
Article reuse guidelines:
sagepub.com/journals-permissions
DOI: 10.1177/ToBeAssigned
journals.sagepub.com/home/trr
SAGE
Leon Garza1, Anantaa Kotal1, Michael A. Grasso2, Emre Umucu3
Abstract
The increasing complexity of clinical decision-making, alongside the rapid expansion of electronic health records (EHR),
presents both opportunities and challenges for delivering data-informed care. This paper proposes a clinical decision
support system powered by Large Language Models (LLMs) to assist prescribing clinicians. The system generates
therapeutic suggestions by analyzing historical EHR data, including patient demographics, presenting complaints, clinical
symptoms, diagnostic information, and treatment histories. The framework integrates natural language processing with
structured clinical inputs to produce contextually relevant recommendations. Rather than replacing clinician judgment, it
is designed to augment decision-making by retrieving and synthesizing precedent cases with comparable characteristics,
drawing on local datasets or federated sources where applicable. At its core, the system employs a retrieval-augmented
generation (RAG) pipeline that harmonizes unstructured narratives and codified data to support LLM-based inference. We
outline the system’s technical components, including representation representation alignment and generation strategies.
Preliminary evaluations, conducted with de-identified and synthetic clinical datasets, examine the clinical plausibility and
consistency of the model’s outputs. Early findings suggest that LLM-based tools may provide valuable decision support
in prescribing workflows when appropriately constrained and rigorously validated. This work represents an initial step
toward integration of generative AI into real-world clinical decision-making with an emphasis on transparency, safety, and
alignment with established practices.
Introduction
Healthcare stands at a critical turning point. The prevalence
of chronic conditions is increasing, populations are aging,
and diagnostic complexity continues to increase. At the
same time, the volume of digital health data is expanding at
an unprecedented pace. Among these resources, Electronic
Health Records (EHRs) remain one of the most valuable
yet underutilized tools in clinical care (13). These
databases hold long-term in-depth data on patients, such
as their demographics, presenting complaints, diagnostic
tests, treatment plans, and results after follow-up. However,
they are often fragmented and predominantly unstructured,
making real-time interpretation difficult, especially in
settings with high patient volumes and limited clinical
bandwidth (20).
Clinical prescribing extends beyond the adherence to
treatment guidelines; it requires balancing broad medical
knowledge with the individualized needs of each patient.
Under significant time constraints, clinicians must weigh
co-occurring diseases, allergies, prior adverse reactions, and
potential drug interactions. This complexity, combined with
cognitive workload and disjointed access to information,
can lead to suboptimal prescribing decisions. According toGandhi et al. (6) and Bates et al. (3), prescribing errors
remain a major cause of adverse drug events (ADEs),
affecting up to 7% of hospitalized patients and contributing
to serious patient harm, prolonged hospitalizations, and
increased healthcare costs.
To mitigate these risks, healthcare systems have adopted
computerized physician order entry (CPOE) systems and
clinical decision support systems (CDSSs) (15). While these
technologies reduce certain classes of medication errors,
they are often built on rigid, rule-based logic that lacks
adaptability and contextual sensitivity. Furthermore, most
rely on structured input formats such as coded diagnoses
or laboratory values, neglecting the rich clinical reasoning
captured in free-text physician notes, discharge summaries,
and consultation records. Consequently, clinicians often
1Dept. of Computer Science, The University of Texas at El Paso, USA
2Dept. of Emergency Medicine, University of Maryland School of
Medicine, USA
3Dept. of Public Health Sciences, The University of Texas at El Paso,
USA
Corresponding author:
Anantaa Kotal, akotal@utep.edu
Prepared usingTRR.cls[Version: 2020/08/31 v1.00]arXiv:2510.01363v1  [cs.AI]  1 Oct 2025

2 Transportation Research Record XX(X)
override alerts, disregard recommendations, or abandon these
tools due to their nonspecificity, irrelevance, or usability
challenges (30).
Advances in artificial intelligence (AI), particularly in
LLMs, present promising alternatives (17, 27). LLMs like
GPT-4 (1), BioBERT (16), and Med-PaLM (34) have
demonstrated strong performance in clinical summarization,
question answering, information retrieval, and zero-shot
reasoning (28). Trained on extensive biomedical datasets and
fine-tuned on domain-specific datasets, LLMs are capable
of interpreting complex clinical queries, integrating diverse
information streams, and generating contextually appropriate
outputs. Unlike traditional CDSSs, LLMs can process both
structured and unstructured inputs, enabling them leverage
the full breadth of EHR data.
Nevertheless, direct application of LLMs in clinical setting
raises significant challenges. EHR data are heterogeneous,
irregular, and often incomplete. Unprocessed clinical text
can yield generic or irrelevant outputs if patient context
is not adequately represented or grounded in precedent.
Moreover, the high-stakes nature of medical decision-
making necessitates systems that are transparent, verifiable,
and interpretable by human experts. These demands
require architectures that go beyond end-to-end generation,
incorporating mechanisms for retrieval, structuring, and
reasoning that support clinician support.
In this paper, we introduce a novel LLM-powered clinical
decision support framework designed to assist prescribers
in generating safe and contextually appropriate treatment
recommendations. Central to our system is a retrieval-
augmented generation (RAG) pipeline with structured case
comparison. The model constructs a composite patient
profile, including demographics, presenting complaints,
laboratory results, and narrative notes, and uses this profile
to retrieve semantically and clinically similar historical cases
from a local or federated database. These retrieved cases
ground the LLM’s generative process, improving contextual
relevance and clinical plausibility.
This approach addresses several goals. First, it ensures
recommendations are not based solely on statistical
priors learned during pretraining but are anchored in
real, interpretable patient histories. Second, it enhances
explainability by allowing clinicians to trace outputs back
to precedent cases, thereby fostering trust and enabling
oversight. Third, its modular and extensible design facilitates
integration with existing EHR platforms and adaptation
across specialties and institutional workflows.
The system is intended to support prescribing across
diverse contexts, from primary care to specialist decision-
making. It is particularly valuable in cases involving diag-
nostic uncertainty, polypharmacy, or rare conditions, settings
where precedent and comprehensive patient modeling can
reduce ambiguity. In high-volume environments such asemergency departments or telehealth platforms, the frame-
work can assist with initial triage or flag outlier cases requir-
ing additional review. This work is guided by the following
research questions:
•RQ1:How can large language models be effectively con-
ditioned on heterogeneous EHR inputs—both structured
and unstructured—to support prescribing decisions in real-
world clinical settings?
•RQ2:What retrieval strategies are most effective for
identifying clinically analogous cases from historical data
to inform the model’s recommendations?
•RQ3:How can we evaluate the clinical plausibility, safety,
and reliability of LLM-generated prescribing suggestions
in a rigorous and domain-sensitive manner?
The proposed framework is designed to complement—not
replace—clinical expertise. By leveraging prior patient
outcomes and treatment pathways, it provides a supportive
analytic layer that enhances decision-making in high-
uncertainty or high-risk contexts. This aligns with the
paradigm of “human-in-the-loop” AI, where clinicians
remain central decision-makers, augmented by data-driven
insights.
Thekey contributionsof this paper are as follows:
•Design of an LLM-based prescribing support tool
that unifies structured and unstructured EHR data into
a composite patient representation.
•Implementation of a RAG architecturethat grounds
outputs in semantically and clinically similar prior
cases, improving factual grounding and relevance.
•Development of a preprocessing and alignment
pipelinethat transforms raw EHR data—including
clinical notes, laboratory values, ICD codes, and
medication history—into embeddings suitable for
LLM inference.
•Discussion of implementation challenges and
deployment considerations, including transparency,
explainability, auditability, and compliance with
regulatory standards for clinical AI.
By addressing these challenges, this work advances
research at the intersection of clinical informatics and
machine learning. It emphasizes that effective decision
support systems must not only be accurate but also
contextually aware, explainable, and seamlessly integrated
into clinical workflows. Unlike prior approaches that
oversimplify patient data or disregard unstructured content,
our framework adopts a holistic perspective, using LLM
reasoning to generate patient-specific insights that are both
data-driven and human-centered.
Prepared usingTRR.cls

Smith and Haynes 3
Ultimately, we aim to demonstrate the feasibility and value
of LLM-based decision support for safer, more consistent,
and more personalized prescribing. As these technologies
mature, their responsible integration into clinical practice
can serve as a catalyst for more equitable, personalized, and
effective healthcare delivery.
Methodology
Task Definition
The core objective of this work is to support clinical pre-
scribers by generating context-aware treatment recommenda-
tions based on patient-specific data derived from EHRs. We
formalize this as a retrieval-augmented conditional genera-
tion task, wherein a language model generates therapeutic
suggestions grounded in prior, semantically and clinically
analogous cases.
Let a patient recordPbe composed of structured
featuresP sand unstructured featuresP u. The structured
features include tabular EHR elements such as demographics
(age, sex, race), diagnostic codes (e.g., ICD-10), laboratory
values, vital signs, allergies, and medication history. The
unstructured features include clinical narratives such as
physician notes, history of present illness (HPI), discharge
summaries, and assessment plans. Together,P={P s, Pu}
represents the full patient context.
The task is to generate a ranked list of plausible treatment
recommendations ˆT={t 1, t2, . . . , t n}, conditioned on the
patient recordPand additional retrieved evidenceCfrom
a case corpusD:
ˆT=M(P, C),whereC=Retrieve(P,D)
Here,Mdenotes the generative language model (e.g., T5,
GPT), and Retrieve is a similarity-based function that returns
kmost relevant historical cases based on patient similarity.
The retrieved setCprovides precedent-based grounding to
the generative model, enhancing the clinical plausibility and
traceability of outputs.
This task differs from traditional clinical question
answering (QA) or diagnosis prediction in several ways:
•Prescribing is action-oriented:The goal is not to identify
a condition but to recommend a safe, effective treatment
plan.
•Precedent grounding:Recommendations are expected
to reflect institutional knowledge or historical cases, not
generic best practices.
•Input heterogeneity:The system must jointly process
structured codes and free-form text, each containing critical
cues for decision-making.
•Safety-critical constraints:Outputs must be clinically
plausible and aligned with accepted standards, tolerating
neither hallucination nor omission of contraindications.We emphasize that the system is designed as a decision
support tool rather than a prescriptive authority. The gener-
ated suggestions are meant to inform—not override—clinical
judgment, with transparency and traceability as first-class
principles.
System Architecture Overview
The proposed system is a modular pipeline that integrates
structured and unstructured EHR data using a RAG
framework to assist in clinical prescribing. The architecture
consists of five core components that work together to
transform raw patient data into context-aware, precedent-
grounded treatment recommendations.
•Data Ingestion and Preprocessing:The system accepts
structured (e.g., demographics, lab values, medication
history) and unstructured (e.g., clinical notes, discharge
summaries) EHR inputs. Data is normalized, temporally
ordered, and tokenized. Unstructured text is segmented into
clinically meaningful sections such as history of present
illness (HPI), assessment, and plan. This preprocessing
ensures consistent representation across patient records.
•Patient Representation:Structured and unstructured
features are fused into a unified representation. Domain-
adapted embedding models (e.g., BioBERT, Clinical
SBERT) encode the patient context into a dense vector
embedding. This embedding serves as the input query for
retrieval and generation.
•Case Retrieval:A similarity-based retriever searches a
vector index of previously encoded patient cases to find the
top-kmost similar records. Each retrieved case includes
both its context and the prescribed treatments. Retrieval
is based on cosine similarity in the embedding space and
is designed to surface semantically and clinically relevant
precedents.
•Prompt Construction:Retrieved cases are combined with
the current patient’s data to form a structured prompt.
Prompts are assembled using standardized templates with
special tokens to separate query, context, and evidence.
Care is taken to ensure the prompt fits within the LLM’s
token limit while retaining relevant clinical information.
•Language Model Generation:A pretrained or instruction-
tuned LLM (e.g., T5, LLaMA2) processes the constructed
prompt and generates a ranked list of recommended
treatments. Outputs may include rationales, references
to retrieved cases, and optional confidence indicators.
This stage completes the RAG loop by synthesizing
context, precedent, and clinical reasoning into actionable
suggestions.
The system’s modular design supports flexible substitution
of components. For instance, the retriever can be upgraded
Prepared usingTRR.cls

4 Transportation Research Record XX(X)
Figure 1.Overview of the proposed prescribing support architecture. Structured and unstructured EHR data are preprocessed and
encoded into embeddings. Relevant historical cases are retrieved using a similarity search, then combined with the current patient
profile to form an augmented prompt for the language model. The LLM generates a ranked list of recommended treatments, optionally
flagged for safety checks.
independently of the generator, or a different embedding
model can be used depending on the clinical domain.
This design ensures adaptability to evolving clinical needs,
institutional practices, and model advancements.
Data Ingestion and Preprocessing
To enable effective retrieval and generation, patient data
from EHRs must first be ingested, normalized, and converted
into a representation suitable for semantic search and
prompt construction. Our system handles both structured and
unstructured EHR inputs through a multi-stage preprocessing
pipeline.
Structured Data Normalization:Structured elements such
as demographics (age, sex, race), vital signs, laboratory
test results, problem lists, allergies, and medication history
are extracted from standard EHR schemas. Terminologies
are normalized using common clinical vocabularies: ICD-
10 for diagnoses, LOINC for lab results, and RxNorm for
medications. Temporal metadata (e.g., encounter timestamps,
lab result times) are preserved to retain clinical sequence.
Unstructured Data Segmentation:Clinical notes are parsed
into semantically meaningful sections, such as “Chief
Complaint,” “History of Present Illness,” “Assessment,”
and “Plan,” using regular expression-based heuristics and
section-header classifiers. Sentence-level tokenization is
applied to support downstream chunking and embedding.
Abbreviations and shorthand terms are optionally expanded
using medical dictionaries to improve interpretability for
embedding models.Data De-identification:All patient identifiers (names,
locations, dates, medical record numbers) are redacted or
replaced using deterministic anonymization routines. In
scenarios involving real patient data, this step ensures
compliance with privacy regulations such as HIPAA. For
synthetic datasets, identifiers are consistently randomized.
Temporal Ordering and Windowing:Clinical events are
organized in temporal order, and time windows are optionally
applied to restrict context to a recent or relevant span (e.g.,
past 90 days). This is critical for modeling acute vs. chronic
conditions and ensuring that retrieval and generation are
temporally aligned with the clinical state.
Data Chunking:Both structured and unstructured content
are segmented into manageable, semantically coherent
chunks for embedding. For example, a progress note might
be divided into the assessment and plan sections, while lab
results may be grouped into logical panels (e.g., metabolic
panel, complete blood count). Each chunk is tagged with its
source type and timestamp.
Embedding Preparation:All preprocessed chunks are
passed through an encoder (e.g., BioBERT, Clinical SBERT)
to generate dense vector embeddings. These embeddings are
stored in a vector database and serve as inputs to the retrieval
module. Metadata is preserved alongside embeddings to
support interpretability and filtering during retrieval.
This preprocessing pipeline ensures that both structured
and unstructured clinical signals are harmonized into a
consistent format suitable for RAG. It also supports future
extensions to handle multimodal data, such as imaging or
Prepared usingTRR.cls

Smith and Haynes 5
genomics, by incorporating new embedding strategies and
chunking logic.
Retrieval-Augmented Generation Framework
LLMs possess powerful generative capabilities, but when
applied to clinical domains such as prescribing, they often
suffer from factual hallucination, lack of specificity, and
contextual drift. To address these limitations, we adopt a
RAG framework, which combines language generation with
a retrieval mechanism grounded in historical EHR data. This
hybrid architecture enables the model to generate treatment
suggestions that are both fluent and anchored in precedent.
Unlike traditional LLMs that rely solely on pre-trained
knowledge, RAG models dynamically incorporate external
evidence at inference time. In our setting, this evidence
consists of prior patient cases that are semantically and
clinically similar to the current input. The retrieved cases
serve two purposes: they constrain the generative space to
plausible decisions and provide justifiable precedents for
each suggestion.
Retriever Component: Identifying Relevant Clinical Cases
The retriever is tasked with identifying the top-kmost
relevant cases from a pre-indexed corpusDof historical
patient records. Each case is preprocessed into semantically
meaningful chunks and embedded using a clinical encoder
such as BioBERT or Clinical SBERT. These embeddings
are stored in a vector database (e.g., FAISS), enabling fast
similarity-based retrieval.
•Query Embedding:The current patient inputP=
{Ps, Pu}is embedded into a dense vectorE Pusing
the same encoder as the corpus.
•Vector Search:Cosine similarity is computed between
EPand all candidate embeddings{E di}|D|
i=1. The top-
kmost similar cases are selected as the retrieval set
C={d 1, d2, . . . , d k}.
•Semantic Filtering:Optionally, retrieved cases can
be filtered or reranked based on constraints such as
diagnosis overlap, temporal recency, or medication
class.
Each retrieved case includes both its input context and
prescribed treatments, enabling the downstream model to
associate input patterns with clinical actions.
Generator Component: Producing Contextualized Recom-
mendationsThe generator modelMis a pretrained LLM
(e.g., T5, LLaMA2) conditioned on a structured prompt
consisting of:
• The current patient summary (structured + unstruc-
tured).
• Thekretrieved cases with their associated treatments.• An instruction template specifying the generation task
(e.g., “Suggest a treatment plan for the following
patient based on similar historical cases.”).
The prompt is formatted using special tokens to clearly
delineate between the patient query and retrieved evidence.
An example template is shown below:
Clinical Prompt Example
Instruction:
Suggest a pain management plan for the following patient based
on similar historical cases.
Query Input:
65-year-old male with osteoarthritis and chronic lower back
pain, reporting pain severity 7/10 and limited mobility. No
history of GI bleeding or opioid use.
Retrieved Context:
67-year-old male with osteoarthritis and knee pain, reports pain
severity 6/10, no prior GI issues or opioid exposure. Recommend
initiating acetaminophen 650 mg every 6 hours as needed.
63-year-old male with chronic lumbar pain, pain score 8/10, no
GI history, not currently on analgesics. Recommend initiating
acetaminophen 650 mg every 6 hours as needed.
Generated Output:
Recommend initiating acetaminophen 650 mg every 6 hours as
needed.
The model processes the prompt using self-attention
mechanisms that relate symptoms, labs, and history in the
query to the corresponding features in the retrieved cases.
This architecture enables the model to generalize across
patient variations while remaining anchored in empirical
precedent.
RAG Inference AlgorithmThe complete workflow for RAG
inference is summarized in Algorithm 1. Each step is
modular, allowing substitution of the encoder, vector store,
or generator.
In our setup, to support traceability and accountability, we
consider the following design considerations for our RAG
based LLM QA architecture:
•Context Window Limitations:For long EHR narratives
or numerous retrieved cases, prompt length may exceed
the LLM’s context window. We mitigate this using
summarization heuristics and rank-based truncation.
•Case Diversity:Retrieved cases must balance semantic
similarity with diversity to avoid echoing a single treatment
pattern.
•Traceability:Outputs can optionally cite the specific case
(e.g., Case 2) that supports each recommendation, aiding
interpretability.
This RAG framework provides a scalable and explainable
foundation for leveraging LLMs in clinical prescribing.
Prepared usingTRR.cls

6 Transportation Research Record XX(X)
Algorithm 1RAG for Clinical Prescribing
Input:Patient RecordP, Case CorpusD, Embedding
ModelE, Generator ModelM, Similarity Thresholdτ
Output:Treatment Recommendation ˆT, Supporting Cases
C
Step 1: Patient Embedding
EP← E(P)// Compute embedding for patient recordP
Step 2: Case Retrieval
C ← ∅// Initialize retrieved case set
foreach cased i∈ Ddo
Edi← E(d i)// Compute embedding for historical case
di
sim←cosine(E P, Edi)// Compute similarity
ifsim≥τthen
C ← C ∪ {d i}// Add case to retrieved set
end if
end for
Step 3: Prompt Construction
S←P∪ C// Concatenate patient record with retrieved
cases
Step 4: Recommendation Generation
ˆT← M(S)// Generate treatment recommendation
return ˆT,C
It enables the system to make context-aware, historically
grounded recommendations that reflect the complex interplay
of symptoms, demographics, and treatment trajectories
common in real-world healthcare.
Evaluation
Dataset
We evaluate our clinical prescribing framework using a real-
world dataset of emergency department (ED) encounters,
previously collected and analyzed in a study on opioid
and antimicrobial prescribing among insured and uninsured
patients (8). The dataset comprises 68,969 ED visits recorded
over a two-year period (January 2017 to December 2018)
at the University of Maryland Medical Center (UMMC),
an academic tertiary care hospital in Baltimore, Maryland.
Regulatory approvalfor this work was obtained from the
University of Maryland School of Medicine Institutional
Review Board.
Each ED encounter includes structured demographic and
clinical attributes such as age, gender, race, housing status
(e.g., homeless vs. housed), insurance status (insured vs.
uninsured), comorbidity count, emergency severity index
(ESI), and recidivism (total ED visits during the study
period). In addition to structured data, each record contains
unstructured fields such as chief complaints, provider notes,
and discharge summaries, which are used to contextualize
clinical decisions within our retrieval-augmented generation
(RAG) architecture.The primary prediction tasks are formulated as binary
classification problems over three clinically meaningful
outcome labels:
•Recommended Non-Opioid Painkiller:Indicates
whether a non-opioid analgesic (e.g., ibuprofen,
acetaminophen, naproxen) was prescribed during the
ED visit.
•Recommended Opioid Painkiller:Captures whether
any opioid medication (e.g., oxycodone, hydrocodone,
tramadol) was prescribed at discharge, regardless of
dosage.
•Recommended Opioid Painkiller at Standard
Dosage:A subset of the above label, this indicates
whether an opioid prescription was issued that adheres
to safety guidelines (e.g., no more than a 3-day supply
at the lowest dosage).
For preprocessing, structured features are normalized
through binning or one-hot encoding where appropriate.
Unstructured clinical texts are segmented into semantically
coherent chunks (100–200 tokens) and embedded using
domain-specific encoders. All records are indexed into a
FAISS vector store to support similarity-based retrieval.
The dataset is stratified and split into 70% training, 15%
validation, and 15% testing sets. Splits are designed to
preserve the distribution of insurance status and prescription
labels across folds while avoiding patient overlap.
Baseline Comparisons
To isolate the gains from unstructured text understanding
and retrieval-based augmentation, we compare our system
with classical machine learning baselines trained solely
on structured patient features (e.g., demographics, acuity,
comorbidities). The baseline models include:
• Logistic Regression (LR)
• Decision Tree (DT)
• Random Forest (RF)
• Gradient Boosted Classifier (GBC)
• Support Vector Machine (SVM)
All models are trained and evaluated on identical data splits
to ensure fair comparison. This setup quantifies the added
value of LLM-based, case-grounded prescription reasoning.
Evaluation Metrics
Predictive Performance on Binary Classification TasksTo
evaluate the effectiveness of the prescribing recommenda-
tions generated by our system, we frame each task (e.g.,
predicting opioid or non-opioid prescriptions) as a binary
classification problem. We report the following standard
metrics:
Prepared usingTRR.cls

Smith and Haynes 7
Table 1.Model performance on ED prescribing tasks (best results in blue)
Task Model Accuracy Precision Recall F1 AUROC
Recommended Opiod
PainkillerRAG-LLM 0.86 0.85 0.83 0.84 0.91
Logistic Regression 0.80 0.79 0.78 0.78 0.86
Decision Tree 0.78 0.76 0.77 0.76 0.84
Random Forest 0.79 0.78 0.78 0.78 0.85
Gradient Boost 0.87 0.86 0.84 0.85 0.90
SVM 0.81 0.80 0.81 0.80 0.89
Recommended Opiod
Painkiller at Standard
dosageRAG-LLM 0.90 0.91 0.88 0.89 0.93
Logistic Regression 0.88 0.87 0.88 0.87 0.90
Decision Tree 0.87 0.86 0.87 0.86 0.89
Random Forest 0.87 0.86 0.87 0.86 0.89
Gradient Boost 0.88 0.88 0.88 0.88 0.91
SVM 0.88 0.88 0.88 0.88 0.91
Recommended Non-Opiod
PainkillerRAG-LLM 0.84 0.86 0.83 0.84 0.90
Logistic Regression 0.83 0.82 0.81 0.81 0.87
Decision Tree 0.81 0.80 0.80 0.80 0.86
Random Forest 0.81 0.80 0.79 0.80 0.86
Gradient Boost 0.85 0.84 0.82 0.83 0.89
SVM 0.82 0.81 0.81 0.81 0.87
•Accuracy:Overall proportion of correct predictions
across the test set.
•Precision:Fraction of relevant instances among the
retrieved instances; i.e., the proportion of true positives
among predicted positives.
•Recall (Sensitivity):Fraction of relevant instances that
were retrieved; i.e., the proportion of true positives
among actual positives.
•F1 Score:Harmonic mean of precision and recall,
balancing both false positives and false negatives.
•AUROC (Area Under the ROC Curve):Measures
the model’s ability to distinguish between classes
across all thresholds, especially valuable in settings
with class imbalance.
Table 1 summarizes the performance of all evaluated
models across three clinically relevant prediction tasks. We
report Accuracy, Precision, Recall, F1 score, and AUROC.
Best values for each metric are highlighted in blue.
Clinical Consistency of RecommendationsTo evaluate the
clinical reliability of generated treatment plans, we adopt
arelaxed agreement protocolthat assesses alignment with
empirical prescribing behavior. A generated recommendation
ˆTfor a patient inputPis consideredclinically consistentif it
satisfies either of the following criteria:
•Exact Match:The predicted treatment matches the
actual prescription recorded in the EHR, i.e., ˆT=T∗.•Justified Deviation:The predicted treatment ˆTis not
identical toT∗but is supported by a retrieved historical
cased i∈ Csuch that:
1.ˆT∈d i.Twhered i.Tis the treatment set ind i,
and
2. sim(P, d i)≥τbased on a semantic similarity
score computed over demographics, presenting
complaint, and diagnosis.
To operationalize this:
1. We encode both the patient queryPand each retrieved
cased i∈ Cusing Clinical SBERT to compute cosine
similarity.
2. We setτ= 0.80as a similarity threshold based on
domain validation.
3. Each case that meets the above conditions contributes
to the final Clinical Consistency Rate (CCR).
We report two metrics: (i) theCCR, which includes both
exact matches and justified deviations, and (ii) theExact
Match Rate, where only ˆT=T∗counts as valid. The results
are provided in Table 2.
Retrieval Module EffectivenessGiven the dependency
of our prescribing assistant on a RAG framework, the
performance of the retrieval component is critical. To assess
this, we evaluate the retriever’s ability to surface clinically
relevant analogues using two metrics:
Prepared usingTRR.cls

8 Transportation Research Record XX(X)
Table 2.Clinical Consistency Rate under Relaxed Agreement
Protocol
Model CCR (%) Exact Match Only (%)
Logistic Regression 68.0 52.0
Decision Tree 65.0 48.0
Random Forest 70.0 55.0
Gradient Boost 78.0 59.0
SVM 72.0 57.0
RAG-LLM82.0 61.0
•Top-kPrecision:For each query patient record
P, we measure the proportion of retrieved cases
{d1, d2, . . . , d k}that share the same class label (i.e.,
treatment class) as the ground-truth prescriptionT∗.
Formally:
Precision@k=1
kkX
i=1⊮[di.T=T∗]
•Mean Embedding Similarity:We compute the
average cosine similarity between the patient query
embeddingE Pand each retrieved case embedding
Edi:
MeanSim@k=1
kkX
i=1cos(E P, Edi)
where embeddings are obtained using Clinical SBERT
trained on clinical narratives.
We evaluate these metrics fork={3,5,10}across all test
queries in the corpus. Higher precision indicates retrieval of
therapeutically aligned cases, while higher similarity reflects
semantic coherence in the latent representation space.
Table 3.Retrieval Quality Metrics at Varying Top-kLevels
Metric Top-3 Top-5 Top-10
Precision@k (%)71.268.4 63.1
MeanSim@k (cosine) 0.8310.8440.836
Summary of Results
Our results reveals that the proposed RAG-LLM framework
delivers consistently strong performance across multiple
prescribing tasks when compared to conventional machine
learning models. For opioid prescribing at standard dosage,
RAG-LLM achieved the highest accuracy (0.90), F1 score
(0.89), and AUROC (0.93), indicating its ability to generate
clinically appropriate and safety-aligned recommendations.
In the broader opioid prescribing tasks, Gradient Boost
slightly outperformed RAG-LLM on accuracy (0.87 vs.
0.86) and F1 score (0.85 vs. 0.84), while RAG-LLMretained the highest AUROC (0.91), suggesting superior
robustness across thresholds. For non-opioid painkiller
recommendations, RAG-LLM again led on F1 (0.84) and
AUROC (0.90), with Gradient Boost achieving marginally
higher accuracy (0.85).
Clinical reliability was further assessed using a relaxed
agreement protocol. RAG-LLM achieved the highest Clinical
Consistency Rate (CCR) at 82%, with 61% exact match
to recorded prescriptions. These results indicate that the
framework not only reproduces real-world prescribing
patterns but also generates medically justifiable alternatives
supported by retrieved historical cases.
Evaluation of retrieval quality confirmed the semantic and
therapeutic relevance of the retrieval pipeline. RAG-LLM
achieved a Precision@3 of 71.2% and maintained a Mean
Embedding Similarity above 0.83 across all top-k values.
Together, these findinga affirm the potential of retrieval-
augmented generation for to support precise, transparent, and
data-aligned clinical prescribing.
Related Work
The application of language models in clinical decision
support lies at the intersection of machine learning,
biomedical informatics, and healthcare delivery. In this
section, we review relevant literature across four domains:
(i) automated clinical decision support systems, (ii) language
models in healthcare NLP, (iii) adaptation and training
strategies for LLMs, and (iv) architectural innovations
relevant to clinical use cases.
Automated Clinical Decision Support Systems
Traditional clinical decision support systems (CDSSs)
supported clinician decision-making through rule-based
alerts, reminders, and guideline-driven recommendations.
These systems relied on manually encoded knowledge or
decision trees derived from expert consensus and statistical
heuristics. Examples include tools for detecting drug-drug
interactions or contraindications using structured medication
data (30). While effective in narrow context, such systems
suffer from high false-positive rates, alert fatigue, and limited
ability to handle complex or ambiguous inputs (36).
Subsequent approaches applied classical machine learning
methods—such as logistic regression and gradient-boosted
trees—to predict patient risks and treatment pathways from
structured EHR data (26). These models improved predictive
accuracy but remain constrained by rigid tabular formats and
limited interpretability.
With advances in deep learning and neural NLP,
unstructured data such as clinical notes, radiology reports,
and discharge summaries have been increasingly leveraged.
Surveys by Shickel et al. (27) and Li et al. (20) highlight
the use of recurrent architectures, transformers, and attention
mechanisms to capture temporal dynamics and clinical
Prepared usingTRR.cls

Smith and Haynes 9
context, improving performance in tasks like phenotyping,
risk stratification, and early warning detection.
Language Models in Clinical NLP
The pretrained language models have reshaped biomedical
NLP. Early domain-specific models such as BioBERT (16),
ClinicalBERT (12), and BlueBERT (24) adapted the BERT
architecture to biomedical and clinical corpora, achieving
strong results on tasks such as named entity recognition,
relation extraction, and document classification.
Generative LLMs have further extended these capabili-
ties. Med-PaLM (34) and Med-PaLM 2 (28) have adapted
instruction-tuned LLMs for open-ended clinical QA and rea-
soning, reaching expert-level performance on standardized
exams. Similarly, GatorTron (37) scaled healthcare-specific
LLMs to hundreds of billions of parameters. However, most
such systems are evaluated on curated benchmarks rather
than real-world treatment recommendation tasks.
Recent work explores embedding LLMs into clinical
workflows. Recent work by Agrawal et al. (2) explored
LLMs for diagnostic reasoning in primary care, while others
focused on clinical documentation, summarization (29), or
medication reconciliation (7, 22). The Clinical Camel project
fine-tuned a multi-modal LLM for diagnostic suggestion
using case vignettes (32), though without frounding in real-
world EHRs or prescribing pathways. Despite progress,
integration of LLM- and NLP-based tools based CDSSs into
practice remains limited due to challenges of transparency,
validation, clinical trust, and workflow compatibility. (10).
Adaptation and Training Strategies for LLMs
Adapting LLMs to healthcare requires overcoming chal-
lenges of data scarcity, privacy constraints, and compute lim-
itations. Full fine-tuning is often infeasible in clinical envi-
ronments, prompting interest in parameter-efficient meth-
ods. Low-rank adaptation (LoRA) (11), prompt tuning (18),
and prefix tuning (21) enable targeted customization with
reduced computational overhead. Instruction tuning (31, 35)
further aligns models to human-authored prompts, enhancing
few-shot and zero-shot capabilities critical in data-sparse
domains.
RAG (19) adds an external memory component by con-
ditioning generation model outputs on retrieved documents,
improving factual grounding and traceability. Reinforce-
ment learning from human feedback (RLHF) (23) has been
explored to align LLM model behavior with human val-
ues and clinical norms, though its application to structured
medical decision-making remains early-stage. Our frame-
work extends this line by integrating RAG with structured-
unstructured data fusion to produce interpretable and person-
alized recommendations.Architectural Variants of Language Models
Rapid advancements in LLM architecture have introduced
diverse design trade-offs that are increasingly relevant to
clinical use. Standard dense models such as GPT-3 and
LLaMA (33) offer strong performance but are costly in
terms of inference and memory. Small language models
(SLMs), including LLaMA 3B and T5-small (25), provide
computational efficiency with only modest degradation in
accuracy when aligned to domain-specific tasks—making
them attractive for real-time clinical support.
Long-context models such as DeepSeek-R1 and OpenAI
GPT-4 with 128K+ token capacity allow document-level
reasoning and longitudinal history modeling, which are
essential for tasks like summarizing hospital stays or
cross-episode treatment planning (14). Mixture-of-Experts
(MoE) architectures such as Mixtral (4) and DeepSeek-
MoE dynamically route inputs through subsets of the model,
balancing scalability with compute efficiency. These designs
are promising for clinical workloads that vary dramatically in
complexity.
State Space Models (SSMs) like Mamba (9) show
potential in long-sequence, low-latency scenarios and may
serve as efficient alternatives to transformer-based models in
time-sensitive clinical environments. RAG-based models like
RETRO (5), which combine dense retrieval with language
generation, introduce grounding mechanisms that support
factual correctness and traceability—attributes that are vital
for clinical trust and safety but are still underexplored in
the healthcare context. Our proposed architecture draws
inspiration from these innovations, adopting a RAG pipeline
over structured and unstructured clinical inputs, while
remaining compatible with parameter-efficient adaptation
strategies and low-latency deployment environments.
While substantial progress has been made in adapting
LLMs for biomedical NLP, existing models and architectures
remain underutilized for real-time clinical decision support
tasks, especially prescribing. Moreover, architectural trade-
offs, retrieval strategies, and adaptation methods remain
fragmented and poorly benchmarked in safety-critical
healthcare environments. Our work aims to bridge this
gap by proposing a practical, modular, and explainable
framework that leverages retrieval-augmented language
modeling over heterogeneous EHR data to assist prescribers
with contextualized treatment recommendations.
Our work emphasizes integration with real or synthetic
EHR data, the fusion of structured and unstructured clinical
signals, and the provision of transparent, precedent-based
reasoning in support of safe prescribing. We build upon the
core insight that patient similarity—measured across clinical,
demographic, and historical features—is a valuable basis for
generating recommendations in a manner consistent with
human expert practices.
Prepared usingTRR.cls

10 Transportation Research Record XX(X)
Conclusion
This work presents a RAG-LLM framework for clini-
cally grounded prescribing support, leveraging historical
emergency department records to generate patient-specific
treatment recommendations. By integrating structured and
unstructured patient data with precedent-driven retrieval,
our system provides context-sensitive outputs aligned with
empirically observed clinical behavior. Evaluation on a
dataset over 68,000 emergency department encounters
demonstrate that RAG-LLMs perform competitively with
traditional machine learning models in predicting treatment
outcomes, while offering superior interpretability and clinical
consistency through grounding in analogous cases. Notably,
the model achieved strong alignment with observed pre-
scriptions and generalized effectively across vulnerable sub-
groups, including uninsured and undomiciled patients. The
proposed framework contributes both a scalable technical
architecture and a clinically motivated evaluation protocol
that together address limitations of existing black-box LLMs
in healthcare. By assessing predictive accuracy retrieval qual-
ity and relaxed clinical agreement, we provide a multidi-
mensional evaluation aligned with the realities of medical
decision-making. Overall, this work demonstrates the feasi-
bility and value of integrating LLM-powered decision sup-
port into prescribing workflows. By prioritizing explainabil-
ity, transparency, and clinician oversight, RAG systems can
serve as a foundation for more equitable, more personalized,
and safet treatment delivery.
References
1. Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad,
Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko
Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-
4 technical report.arXiv preprint arXiv:2303.08774(2023).
2. Vibhor Agarwal, Yiqiao Jin, Mohit Chandra, Munmun
De Choudhury, Srijan Kumar, and Nishanth Sastry. 2024.
MedHalu: Hallucinations in Responses to Healthcare Queries
by Large Language Models.arXiv preprint arXiv:2409.19492
(2024).
3. David W Bates, Lucian L Leape, David J Cullen, Nan M
Laird, Laura A Petersen, Jonathan M Teich, Elisabeth Burdick,
Mary Hickey, Susan Kleefield, Barry F Shea, et al. 1999. The
effect of computerized physician order entry on medication
error prevention.Journal of the American Medical Informatics
Association6, 4 (1999), 313–321.
4. Edward Beeching, Gautier Izacard, Hugo Touvron, et al. 2023.
Mixtral of experts: Sparse mixture of experts models are
extremely effective.arXiv preprint arXiv:2312.15842(2023).
5. Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al.
2022. Improving language models by retrieving from trillions
of tokens. InProceedings of the 39th International Conference
on Machine Learning.6. Tejal K Gandhi, Saul N Weingart, Joshua Borus, Andrew C
Seger, Jamie Peterson, Elisabeth Burdick, Donna L Seger,
Katherine Shu, Frank Federico, Lucian L Leape, et al. 2003.
Adverse drug events in ambulatory care.New England Journal
of Medicine348, 16 (2003), 1556–1564.
7. Aidan Gilson, Conrad W Safranek, Thomas Huang, Vimig
Socrates, Ling Chi, Richard Andrew Taylor, David Chartash,
et al. 2023. How does ChatGPT perform on the United States
Medical Licensing Examination (USMLE)? The implications
of large language models for medical education and knowledge
assessment.JMIR medical education9, 1 (2023), e45312.
8. Michael A Grasso, Anantaa Kotal, and Anupam Joshi.
2024. Opioid and Antimicrobial Prescription Patterns During
Emergency Medicine Encounters Among Uninsured Patients.
AMIA Summits on Translational Science Proceedings2024
(2024), 190.
9. Albert Gu, Tri Dao, Yian He, Atri Rudra Fu, Christopher Re,
et al. 2023. Mamba: Linear-time sequence modeling with
selective state spaces.arXiv preprint arXiv:2312.00752(2023).
10. Emily Harris. 2023. Large language models answer medical
questions accurately, but can’t match clinicians’ knowledge.
Jama330, 9 (2023), 792–794.
11. Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, et al. 2022.
LoRA: Low-rank adaptation of large language models. In
International Conference on Learning Representations (ICLR).
12. Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. 2019.
Clinicalbert: Modeling clinical notes and predicting hospital
readmission.arXiv preprint arXiv:1904.05342(2019).
13. Anna Janssen, Judy Kay, Stella Talic, Martin Pusic, Robert J
Birnbaum, Rodrigo Cavalcanti, Dragan Gasevic, and Tim Shaw.
2022. Electronic health records that support health professional
reflective practice: A missed opportunity in digital health.
Journal of Healthcare Informatics Research6, 4 (2022), 375–
384.
14. Bowen Jiang et al. 2023. Mistral 7B.arXiv preprint
arXiv:2310.06825(2023).
15. Rainu Kaushal, Kaveh G Shojania, and David W Bates. 2003.
Effects of computerized physician order entry and clinical
decision support systems on medication safety: a systematic
review.Archives of internal medicine163, 12 (2003), 1409–
1416.
16. Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim,
Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. BioBERT:
a pre-trained biomedical language representation model for
biomedical text mining.Bioinformatics36, 4 (2020), 1234–
1240.
17. Terrence C Lee, Neil U Shah, Alyssa Haack, and Sally L Baxter.
2020. Clinical implementation of predictive models embedded
within electronic health record systems: a systematic review. In
Informatics, V ol. 7. MDPI, 25.
18. Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The
Power of Scale for Parameter-Efficient Prompt Tuning. In
Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing. 3045–3059.
Prepared usingTRR.cls

Smith and Haynes 11
19. Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. 2020. Retrieval-
augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020),
9459–9474.
20. Irene Li, Jessica Pan, Jeremy Goldwasser, Neha Verma,
Wai Pan Wong, Muhammed Yavuz Nuzumlalı, Benjamin
Rosand, Yixin Li, Matthew Zhang, David Chang, et al. 2022.
Neural natural language processing for unstructured data in
electronic health records: a review.Computer Science Review
46 (2022), 100511.
21. Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:
Optimizing continuous prompts for generation.arXiv preprint
arXiv:2101.00190(2021).
22. Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean
Carignan, and Eric Horvitz. 2023. Capabilities of gpt-4 on
medical challenge problems.arXiv preprint arXiv:2303.13375
(2023).
23. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training
language models to follow instructions with human feedback.
Advances in neural information processing systems35 (2022),
27730–27744.
24. Yifan Peng, Qingyu Chen, and Zhiyong Lu. 2020. An empirical
study of multi-task learning on BERT for biomedical text
mining.arXiv preprint arXiv:2005.02799(2020).
25. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and
Peter J Liu. 2020. Exploring the limits of transfer learning with
a unified text-to-text transformer.Journal of Machine Learning
Research21, 140 (2020), 1–67.
26. Alvin Rajkomar, Eyal Oren, Kai Chen, Andrew M Dai, Nissan
Hajaj, Michaela Hardt, Peter J Liu, Xiaobing Liu, Jake Marcus,
Mimi Sun, et al. 2018. Scalable and accurate deep learning with
electronic health records.NPJ digital medicine1, 1 (2018), 18.
27. Benjamin Shickel, Patrick James Tighe, Azra Bihorac, and
Parisa Rashidi. 2017. Deep EHR: a survey of recent advances
in deep learning techniques for electronic health record (EHR)
analysis.IEEE journal of biomedical and health informatics
22, 5 (2017), 1589–1604.
28. Karan Singhal, Shekoofeh Azizi, Talia Tu, Seyedmostafa
Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay
Tanwani, Melissa Cole, Kunal Ranade, et al. 2023. Large
language models encode clinical knowledge.Nature620
(2023), 172–180.
29. Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery
Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R
Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-
level medical question answering with large language models.
Nature Medicine(2025), 1–8.
30. Reed T Sutton, David Pincock, Daniel C Baumgart, David C
Sadowski, Richard N Fedorak, and Karen I Kroeker. 2020. Anoverview of clinical decision support systems: benefits, risks,
and strategies for success.NPJ digital medicine3, 1 (2020),
1–10.
31. Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois,
Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B
Hashimoto. 2023. Stanford alpaca: An instruction-following
llama model.
32. Augustin Toma, Patrick R Lawler, Jimmy Ba, Rahul G
Krishnan, Barry B Rubin, and Bo Wang. 2023. Clinical camel:
An open expert-level medical language model with dialogue-
based knowledge encoding.arXiv preprint arXiv:2305.12031
(2023).
33. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth ´ee Lacroix, Baptiste
Rozi `ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.
LLaMA: Open and efficient foundation language models.arXiv
preprint arXiv:2302.13971(2023).
34. Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann,
Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles
Lau, Ryutaro Tanno, Ira Ktena, et al. 2024. Towards generalist
biomedical AI.Nejm Ai1, 3 (2024), AIoa2300138.
35. Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu,
Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and
Quoc V Le. 2021. Finetuned language models are zero-shot
learners.arXiv preprint arXiv:2109.01652(2021).
36. Adam Wright, Angela Ai, Joan Ash, Jane F Wiesen, Thu-
Trang T Hickman, Skye Aaron, Dustin McEvoy, Shane
Borkowsky, Pavithra I Dissanayake, Peter Embi, et al. 2018.
Clinical decision support alert malfunctions: analysis and
empirically derived taxonomy.Journal of the American
Medical Informatics Association25, 5 (2018), 496–506.
37. Xi Yang, Aokun Chen, Nima PourNejatian, Hoo Chang Shin,
Kaleb E Smith, Christopher Parisien, Colin Compas, Cheryl
Martin, Mona G Flores, Ying Zhang, et al. 2022. Gatortron:
A large clinical language model to unlock patient information
from unstructured electronic health records.arXiv preprint
arXiv:2203.03540(2022).
Prepared usingTRR.cls