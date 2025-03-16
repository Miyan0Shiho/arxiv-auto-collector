# Towards Scalable and Cross-Lingual Specialist Language Models for Oncology

**Authors**: Morteza Rohanian, Tarun Mehra, Nicola Miglino, Farhad Nooralahzadeh, Michael Krauthammer, Andreas Wicki

**Published**: 2025-03-11 11:34:57

**PDF URL**: [http://arxiv.org/pdf/2503.08323v1](http://arxiv.org/pdf/2503.08323v1)

## Abstract
Clinical oncology generates vast, unstructured data that often contain
inconsistencies, missing information, and ambiguities, making it difficult to
extract reliable insights for data-driven decision-making. General-purpose
large language models (LLMs) struggle with these challenges due to their lack
of domain-specific reasoning, including specialized clinical terminology,
context-dependent interpretations, and multi-modal data integration. We address
these issues with an oncology-specialized, efficient, and adaptable NLP
framework that combines instruction tuning, retrieval-augmented generation
(RAG), and graph-based knowledge integration. Our lightweight models prove
effective at oncology-specific tasks, such as named entity recognition (e.g.,
identifying cancer diagnoses), entity linking (e.g., linking entities to
standardized ontologies), TNM staging, document classification (e.g., cancer
subtype classification from pathology reports), and treatment response
prediction. Our framework emphasizes adaptability and resource efficiency. We
include minimal German instructions, collected at the University Hospital
Zurich (USZ), to test whether small amounts of non-English language data can
effectively transfer knowledge across languages. This approach mirrors our
motivation for lightweight models, which balance strong performance with
reduced computational costs, making them suitable for resource-limited
healthcare settings. We validated our models on oncology datasets,
demonstrating strong results in named entity recognition, relation extraction,
and document classification.

## Full Text


<!-- PDF content starts -->

Towards Scalable and Cross-Lingual Specialist Language Models for
Oncology
Morteza Rohanian Tarun Mehra Nicola Miglino Farhad Nooralahzadeh
Michael Krauthammer Andreas Wicki
University of Zurich and University Hospital Zurich, Switzerland
morteza.rohanian@uzh.ch
Abstract
Clinical oncology generates vast, unstructured
data that often contain inconsistencies, missing
information, and ambiguities, making it diffi-
cult to extract reliable insights for data-driven
decision-making. General-purpose large lan-
guage models (LLMs) struggle with these chal-
lenges due to their lack of domain-specific
reasoning, including specialized clinical termi-
nology, context-dependent interpretations, and
multi-modal data integration. We address these
issues with an oncology-specialized, efficient,
and adaptable NLP framework that combines
instruction tuning, retrieval-augmented gener-
ation (RAG), and graph-based knowledge in-
tegration. Our lightweight models prove effec-
tive at oncology-specific tasks, such as named
entity recognition (e.g., identifying cancer di-
agnoses), entity linking (e.g., linking entities
to standardized ontologies), TNM staging, doc-
ument classification (e.g., cancer subtype clas-
sification from pathology reports), and treat-
ment response prediction. Our framework em-
phasizes adaptability and resource efficiency.
We include minimal German instructions, col-
lected at the University Hospital Zurich (USZ),
to test whether small amounts of non-English
language data can effectively transfer knowl-
edge across languages. This approach mir-
rors our motivation for lightweight models,
which balance strong performance with re-
duced computational costs, making them suit-
able for resource-limited healthcare settings.
We validated our models on oncology datasets,
demonstrating strong results in named entity
recognition, relation extraction, and document
classification.
1 Introduction
Clinical oncology and related disciplines such as ra-
diology or pathology often capture patient-related
information in an unstructured or semi-structured
way. At the same time, there is an increasing
need to use real-world data to enable data-driventherapy decisions as a strategy that complements
standardized evidence-based (study-informed) de-
cision making.
In the typical healthcare setting, oncologists
must gather vast amounts of information from dif-
ferent data sources, including radiology images
and reports, pathology reports, molecular analyses,
clinical notes, and patient histories. They rely on
these diverse sources to guide diagnosis, the as-
sessment of prognosis and stage, and the decision
on therapy. However, much of this data is in free
text format within electronic health records (EHR)
(Pardoll, 2012; Topol, 2019). Clinicians waste time
and resources as they parse these notes by hand.
This leads to slow, inconsistent, and error-prone
decision-making, especially in resource-limited en-
vironments (Bedogni, 2009).
Natural language processing (NLP) offers tools
to extract insights from free-text clinical records.
Rule-based systems and machine learning methods
with hand-engineered features have successfully
identified entities such as diseases and treatments
(Alawad et al., 2020). These methods fail to han-
dle the nuanced language and variability of clinical
data. Pretrained language models (LMs), such as
BERT (Devlin, 2018), BioBERT (Lee et al., 2020),
and ClinicalBERT (Alsentzer et al., 2019), improve
performance on tasks like entity recognition and lit-
erature mining by leveraging large biomedical cor-
pora (Gu et al., 2021; Huang et al., 2020; Rohanian
et al., 2023, 2024a). Despite these advances, such
models focus primarily on classification, lack flex-
ible reasoning capabilities, and are limited in their
ability to generate coherent text for summarization
or prediction (Ruder, 2017). Moreover, these mod-
els predominantly support English, overlooking
the multilingual requirements of many healthcare
systems.
Large language models (LLMs), such as GPTs
(Brown et al., 2020) and LLaMA (Touvron et al.,
2023), overcome some of these limitations by han-arXiv:2503.08323v1  [cs.CL]  11 Mar 2025

dling diverse tasks and adapting to new domains
with minimal labeled data. Researchers have used
them to summarize medical records, answer ques-
tions, and support clinical decisions (Singhal et al.,
2023; Saab et al., 2024). General-purpose LLMs
often fail in specialized fields like oncology. They
lack domain-specific knowledge, produce inconsis-
tent reasoning (Wu et al., 2024; Hu et al., 2024),
and require substantial computational resources,
which many healthcare institutions cannot afford.
Lightweight models provide a practical alternative
by delivering strong performance with significantly
reduced resource requirements.
Recent research has adapted LLMs for oncology-
specific applications, often addressing single tasks
such as named entity recognition (NER) or relation
extraction (Alawad et al., 2021; Zhou et al., 2022;
Nishio et al., 2023; Fujimoto et al., 2023). How-
ever, these approaches lack scalability and multilin-
gual flexibility. Newer methods integrate biomedi-
cal corpora, retrieval mechanisms, and parameter-
efficient fine-tuning to handle complex tasks. Some
studies have curated large corpora (e.g., from the
TCGA dataset) to build prognostic models or clas-
sify cancer subtypes, but these often rely on manual
feature engineering or rule-based systems (Alawad
et al., 2021). Other works used transformer-based
models for TNM extraction, disease coding, or
limited classification tasks (Kefeli et al., 2024).
We propose an oncology-specialized NLP frame-
work that combines lightweight models, bilingual
adaptability, and advanced reasoning techniques.
Given the Swiss healthcare system’s nature, in-
corporating German alongside English ensures
the framework can address the linguistic diver-
sity encountered in clinical practice at institutions
like USZ. We curated minimal German instruc-
tions from clinical queries at the University Hos-
pital Zurich (USZ) and systematically varied their
number (100, 200, 400) to test whether small
amounts of bilingual data can transfer domain-
specific knowledge effectively across languages.
Both bilingual adaptability and lightweight mod-
els align with our overarching goal of creating
scalable NLP systems that can adapt to diverse
healthcare environments, from large hospitals to
resource-limited clinics.
Our framework tries to solve key challenges in
oncology NLP by integrating instruction tuning,
retrieval-augmented generation, and graph-based
reasoning. Each component targets specific issuesin processing clinical data.
Instruction tuning improves the accuracy of
lightweight models for oncology-specific tasks.
Models handle named entity recognition, relation
extraction, TNM staging, and treatment response
prediction with precision. Bilingual instructions
in English and German align with real clinical use
cases such as ICD-10 coding and treatment classi-
fication. Testing this variation reveals how small
bilingual datasets transfer knowledge across lan-
guages and strengthen cross-lingual adaptability.
RAG improves outputs by retrieving relevant
clinical data from trusted sources. External
datasets such as MIMIC-IV and curated German
oncology reports add real-time context to the
model’s responses. The retrieval process con-
nects queries with factual information from on-
cology corpora. Using hierarchical methods, RAG
retrieves critical details efficiently without over-
whelming the input with unnecessary context.
Graph-based reasoning ensures outputs are reli-
able and factually grounded. A knowledge graph
integrates resources like UMLS, linking extracted
entities to verified medical facts. Relationships
between entities, such as treatments and stages,
are organized as nodes and edges. Triple graph
construction connects entities to authoritative refer-
ences, reducing ambiguity and improving reason-
ing. This process strengthens the clinical reliability
of model-generated outputs.
Lightweight LLaMA variants (LLaMA-2-7B,
LLaMA-3.1-8B, LLaMA-3.2-1B, and LLaMA-
3.2-3B) combine these methods to balance effi-
ciency and performance. The framework adapts
to resource-limited clinical environments while
maintaining high accuracy and flexibility across
oncology-specific applications.
Our contributions are as follows:
1.Oncology-Specialized Modeling: Lightweight
models fine-tuned for oncology tasks like TNM
staging, named entity recognition, relation ex-
traction, document classification, and treatment
prediction. Benchmarks include datasets like
NCBI-Disease, i2b2-2010, and labeled subsets
of TCGA.
2.Multilingual Adaptability: Minimal German
instructions collected from USZ improve cross-
lingual performance on ICD-10 coding and
TNM staging. The bilingual framework sup-
ports diverse healthcare systems by addressing
multilingual requirements.

3.Model Efficiency: Lightweight models such as
LLaMA-2-7B deliver high accuracy with lower
computational costs. This ensures advanced
NLP tools remain accessible to institutions with
limited resources.
4.Task Adaptability: The framework applies
to diverse tasks, including relation extraction,
document classification, and multilingual ICD-
10 coding. Models adapt to new domains and
tasks.
The integration of instruction tuning, RAG, and
graph-based reasoning provides oncology NLP sys-
tems that deliver accurate, efficient, and context-
aware solutions for multilingual and resource-
limited settings.
2 Data Sources
We use a combination of bilingual clinical datasets
and diverse public benchmarks to fine-tune and
evaluate our oncology NLP framework. These
datasets enable the exploration of bilingual adapt-
ability, cross-lingual generalization, and task scala-
bility.
2.1 DUP-127 Clinical Dataset
We include the DUP-127 dataset, a German on-
cology dataset containing structured and unstruc-
tured clinical data. This dataset aligns with our
goal of creating a bilingual NLP framework to ad-
dress the linguistic diversity in clinical practice,
particularly in Swiss healthcare. The dataset was
established within the framework of the Swiss Per-
sonalized Health Network (SPHN), an initiative
supported by the Swiss State Secretariat for Edu-
cation, Research and Innovation (SERI). It encom-
passes around 110 distinct structured datapoints
such as ICD diagnoses, TNM annotations, and
medications extracted from the electronic health
care records of patients with cancer. Data were col-
lected using the SPHN reference dataset (version
2021.1) . The dataset was represented with a Re-
source Description Framework (RDF) schema, en-
crypted, and securely transferred to the data repos-
itories of the participating universities (BioMedIT
network). To ensure semantic harmonization, a
set of semantic rules written in shapes constraint
language (SHACL) was defined and distributed
together with the ontology. To prevent erroneous
content and ensure interoperability, we integrated a
set of queries in a data validation pipeline. Integrity
checks were carried out to investigate possiblemissing data points and inconsistencies in patient
timelines. Metastatic treatment lines were reconsti-
tuted according to progression dates. Drugs were
grouped into regimens according to their adminis-
tration dates and manually corrected by medical
experts Unstructured elements, such as treatment
histories, radiology reports as well as histology
reports and genomic profiles, were manually an-
notated by expert physicians during initial data
preparation. Diagnoses and genomic information
link directly to corresponding free-text records us-
ing patient IDs, ensuring integration between struc-
tured and unstructured data. The SPO protocol (No.
2020-00347) was approved by the Northwest and
Central Swiss Ethics Committee (EKNZ) and rati-
fied by the local ethics committees (CCER, CER-
VD, Kantonale Ethikkommission Bern, Kantonale
Ethikkommission Zürich).
2.2 Public Datasets
We fine-tune and evaluate our model on a range of
tasks that capture the complexity of oncology prac-
tice. This multi-dataset approach reflects the adapt-
ability of our framework across oncology-specific
tasks and supports its scalability to different clini-
cal challenges.
NER: We use NCBI-Disease (Do ˘gan et al.,
2014), BC5CDR (Disease/Chem) (Li et al., 2016),
BC2GM (Ando, 2007), JNLPBA (Collier et al.,
2004), and i2b2-2012 (Uzuner et al., 2011) to test
how well the model extracts biomedical entities
such as diseases, chemicals, or genes from text.
These datasets focus on biomedical literature and
primarily employ the standard BIO (Beginning-
Inside-Outside) labeling scheme.
Relation Extraction: i2b2-2010 (Uzuner et al.,
2011) and GAD (Bravo Serrano et al., 2015) mea-
sure how well the model links genes, diseases, and
treatments. This step tests the model’s ability to
identify relations, for example, a gene-disease as-
sociation or a drug-disease treatment link. The
i2b2-2010 dataset centers on clinical narratives,
where relationships are defined between problems,
test results, and treatments.
NLI: MedNLI (Romanov and Shivade, 2018)
tests logical reasoning about clinical statements,
requiring the model to determine whether a conclu-
sion follows logically from given premises. This
task is particularly relevant in oncology, where clin-
icians must reconcile conflicting findings from re-
ports, pathology notes, or imaging summaries. For

Figure 1: (A) Fine-tuning and evaluation workflow: This panel shows the process of data collection, instruction
building, fine-tuning, and evaluation against clinician annotations. (B) Document and graph retrieval: This panel
highlights the integration of document retrieval and graph-based reasoning for query-based inference.
instance, determining whether a pathology report
implies disease progression based on an imaging
report involves reasoning over subtle textual cues.
Document Classification: Document classifi-
cation addresses the task of assigning labels to
entire texts, such as clinical reports, based on their
content. We use the Hallmarks of Cancer (HoC)
(Baker et al., 2016) dataset and the TCGA Pathol-
ogy Report Dataset (Kefeli et al., 2024)cite for
these experiments.
The Hallmarks of Cancer dataset provides multi-
class labels aligned with ten canonical hallmarks of
cancer, including sustained proliferative signaling,
immune evasion, and genomic instability. These
categories represent critical biological processes
that drive cancer progression. By applying these la-
bels, the model learns to classify biomedical litera-
ture according to underlying cancer-related themes.
The TCGA Pathology Report dataset grounds
this classification in clinical practice. It includes
9,523 pathology reports spanning 32 distinct can-
cer types, each processed through OCR and careful
post-processing. Beyond cancer-type classification,
the TCGA reports include TNM staging annota-
tions (T1–T4, N0–N3, M0–M1). TNM staging pro-
vides essential prognostic information and guides
treatment decisions. We split this dataset into 70%
training, 15% validation, and 15% test, ensuringa balanced approach to model development and
performance evaluation.
We also incorporate the MSK-IMPACT (Ze-
hir et al., 2017) dataset, a curated resource from
Memorial Sloan Kettering Cancer Center. It in-
cludes 1,479 patients treated with systemic im-
mune checkpoint blockade (ICB). This dataset pro-
vides binary labels for treatment response, where
patients are categorized as responders or non-
responders based on clinical response criteria, such
as the RECIST v1.1 guidelines. Responders in-
clude both complete responders (CR), defined as
the disappearance of all target lesions, and partial
responders (PR), defined as at least a 30% decrease
in the sum of the diameters of target lesions. Non-
responders encompass patients with stable disease
(SD) or progressive disease (PD).
3 Methodology
Our methodology transforms pretrained language
models into specialized oncology tools by integrat-
ing instruction tuning, retrieval-augmented gen-
eration (RAG), and graph-based knowledge inte-
gration. In Figure 1, we illustrate the fine-tuning
process (Panel A) and the document and graph
retrieval mechanisms (Panel B). Panel A demon-
strates the end-to-end workflow for building la-

Table 1: Instruction Tuning Examples for Oncology Tasks
Task Instruction Input Text Output
Hallmarks of Can-
cer (HoC)As a medical expert, assess the
clinical text for cancer hallmarks.
Assign one or more labels from
the list: Sustaining proliferative
signaling (PS), Enabling replica-
tive immortality (RI), Inducing
angiogenesis (A), Genome insta-
bility & mutation (GI), Tumor-
promoting inflammation (TPI), ...Taken together, the present study clearly
shows the synergistic anti-inflammatory
as well as anti-oxidative stress effects
of CUR and PUFA.Tumor-promoting
inflammation (TPI)
Natural Lan-
guage Inference
(MedNLI)Evaluate the connection between
two clinical sentences and clas-
sify them into one of these cate-
gories: Contradiction (if the sen-
tences conflict), Neutral (if no
logical association), or Entail-
ment (if one sentence logically
implies the other)...Sentence 1: Lung cancer as above s/p
pneumonectomy
Sentence 2: History of smoking.Neutral
Relationship
Extraction (i2b2-
2010)In the clinical text, your objec-
tive is to identify relationships
between medical problems, treat-
ments, and tests. Medical prob-
lems are tagged as @problem$,
medical tests as @test$, and treat-
ments as @treatment$. Classify
the relationship as: Treatment is
administered for medical prob-
lem (TrAP)...His past medical history is significant
for prostate cancer, benign prostatic hy-
pertrophy, hypothyroidism, status post
@treatment$ for @problem$, chronic
painless hematuria, degenerative joint
disease, and history of a murmur.TrAP
Named Entity
Recognition
(NER)Your mission is to tag disease-
related Named Entities in the text
using the BIO labeling scheme.
When you encounter a disease-
related phrase, mark the start
with B (Begin) and continue with
I (Inner) ...Its role in the therapy of glomeru-
lonephritis, autoimmunity, cystic renal
diseases and renal cancer is under in-
vestigation.... cystic:
B, renal: I,
diseases: I,
and: O, renal:
B, cancer: I...
beled datasets, constructing instructions, and fine-
tuning lightweight models. Panel B highlights how
the system integrates document retrieval, graph-
based reasoning, and query embeddings to generate
clinically relevant responses. Together, these steps
form the core of our methodology for transforming
general-purpose LLMs into oncology-specialized
tools.
These components enable the models to pro-
cess complex oncology data, reason about medical
facts, and generate precise predictions for clinical
workflows. By emphasizing bilingual adaptability
through minimal German instructions and resource-
efficient lightweight models, we ensure our ap-
proach scales across multilingual and resource-
limited healthcare environments.3.1 Instruction Tuning Across Languages
To fine-tune our lightweight generative language
models (LLaMA-2-7B, LLaMA-3.1-8B, LLaMA-
3.2-1B, and LLaMA-3.2-3B), we use curated
instruction-response pairs in English and German.
These instructions simulate real-world oncology
queries, such as identifying cancer-related entities,
TNM staging annotations, or extracting treatment
protocols. Each instruction-response pair provides
structured outputs, such as JSON-formatted anno-
tations specifying entity types, attributes, and their
spans within the text. For instance, a tumor-related
entity recognition query might yield outputs cate-
gorizing “lung cancer” or “EGFR-positive adeno-
carcinoma” with attributes like diagnosis date or
molecular markers. Table 1 provides examples of

instructions used across different oncology tasks,
highlighting their diversity and task-specific objec-
tives. These examples demonstrate how instruc-
tions align with tasks like named entity recognition,
natural language inference, and relation extraction,
ensuring task relevance and improving model gen-
eralization (Rohanian et al., 2024b).
To evaluate cross-lingual adaptability, we aug-
ment public datasets with minimal German instruc-
tions, ranging from 100 to 400 examples. These
instructions cover tasks such as ICD coding, TNM
staging, and treatment annotation. Training mini-
mizes the instruction tuning loss:
Ltuning =−1
NNX
i=1logPθ(yi|xi,instruction ),
where xirepresents the input text, “instruction”
specifies the task, and yiis the expected response.
Cross-validation splits are applied to ensure gener-
alization to unseen instructions and languages.
3.2 Retrieval-Augmented Generation (RAG)
Oncology workflows often require reasoning over
large, diverse, and evolving datasets. To address
this complexity, we integrate retrieval-augmented
generation (RAG), which grounds model responses
in external knowledge. We use a sentence embed-
ding model, fine-tuned for oncology-specific tasks,
to encode user queries ( Q) and candidate docu-
ments Dinto dense vector representations. These
embeddings capture semantic similarity between
clinical terms and contexts. To store and index
these embeddings efficiently, we use the FAISS
(Facebook AI Similarity Search) library (Johnson
et al., 2019). FAISS provides high-speed similar-
ity searches across large document collections, en-
abling real-time retrieval and processing of oncol-
ogy data. User queries Qand candidate documents
Dare encoded into dense vector representations,
with cosine similarity determining their relevance:
sim(Q, D ) =ϕ(Q)·ϕ(D)
∥ϕ(Q)∥∥ϕ(D)∥.
The top-k most relevant documents, selected based
on similarity scores, are appended to the model’s
input. These documents are drawn from external
datasets MIMIC-IV and curated German oncology
discharge reports, ensuring that model responses re-
main accurate, evidence-based, and context-aware.
We incorporate semantic document chunking to
improve retrieval efficiency. Oncology documents,often lengthy and complex, are segmented into
smaller, contextually coherent chunks. We encode
each chunk using a sentence embedding model
fine-tuned for oncology-specific tasks. The result-
ing dense vector representations are indexed in the
FAISS, enabling fast and scalable similarity-based
searches. This hybrid approach uses paragraph-
based splitting combined with semantic similar-
ity analysis, ensuring that each chunk retains top-
ical coherence. By storing these chunks indepen-
dently in the FAISS index, the system ensures that
even detailed oncology data is processed and re-
trieved with high granularity and contextual rele-
vance. Semantic chunking also aligns with graph-
based knowledge integration by mapping extracted
entities to corresponding graph nodes.
We optimize retrieval further using a hierarchi-
cal U-Retrieval strategy. High-level clinical tags,
such as tumor stage, disease type, or treatment
categories, guide the initial retrieval, reducing the
document pool to a manageable size. The system
then iteratively integrates broader contextual sum-
maries, balancing precision with global context
awareness. This multi-layered retrieval enables
comprehensive reasoning over complex oncology-
specific scenarios.
3.3 Graph-Based Knowledge Integration
To enhance factual reliability and interpretability,
we integrate a domain-specific knowledge graph G,
constructed from standardized resources UMLS,
SNOMED-CT, and ICD-10. This graph encodes
entities as nodes and their relationships as edges:
G={(vi, eij, vj)|vi, vj∈V, e ij∈E},
where viandvjrepresent medical entities (e.g.,
“adenocarcinoma” or “Osimertinib”), and eijrep-
resents relationships (e.g., “treated_with”).
Graph enrichment occurs through triple graph
construction, linking retrieved entities to authorita-
tive references and professional definitions:
Triple = [entity ,source ,definition ].
For instance, a TNM stage extracted from text is
mapped to corresponding UMLS nodes and linked
to oncology treatment guidelines, ensuring outputs
remain grounded in verified medical knowledge.
To encode the graph, we employ a two-step pro-
cess:

Table 2: Performance of Models Across Biomedical Tasks with Different Configurations
Model Configuration NCBI-Disease BC5CDR-Disease BC5CDR-Chem BC2GM JNLPBA i2b2-2012 i2b2-2010 MedNLI
Type Model NER NER NER NER NER NER RE NLI
Base LLM
LLaMA-2-7B 85.69 83.12 93.77 77.40 79.67 79.66 90.01 88.76
LLaMA-3.1-8B 86.33 83.86 93.45 79.95 79.78 80.58 90.83 88.09
LLaMA-3.2-1B 85.58 82.48 92.41 77.30 79.63 79.67 89.59 86.63
LLaMA-3.2-3B 83.56 82.89 92.27 78.97 79.12 79.97 89.84 86.63
Instruction-Tuned
LLaMA-2-7B 88.37 86.48 94.12 82.70 82.77 81.62 93.70 90.57
LLaMA-3.1-8B 89.50 87.64 94.82 84.41 83.60 81.92 93.26 90.56
LLaMA-3.2-1B 85.70 86.08 93.08 81.34 81.96 80.83 92.08 90.61
LLaMA-3.2-3B 85.43 86.08 93.27 81.70 81.99 80.88 92.57 89.88
+RAG
LLaMA-2-7B 88.17 86.61 94.38 82.76 82.54 81.92 92.22 91.89
LLaMA-3.1-8B 88.85 87.50 94.77 84.71 83.01 81.21 92.95 91.08
LLaMA-3.2-1B 85.74 86.45 93.29 82.27 81.24 80.39 91.82 90.20
LLaMA-3.2-3B 85.47 86.03 93.18 82.37 81.90 80.77 91.07 90.61
+Graph-RAG
LLaMA-2-7B 88.26 86.42 94.67 84.06 82.29 81.80 93.63 91.19
LLaMA-3.1-8B 88.79 87.32 94.40 84.84 83.56 81.92 93.50 91.87
LLaMA-3.2-1B 87.53 86.49 93.22 82.64 81.41 80.39 92.62 90.72
LLaMA-3.2-3B 87.37 86.53 93.90 83.59 82.09 80.26 92.57 90.58
1.Node Encoding: Each node is represented
as a dense vector embedding using a pre-
trained graph embedding model TransE (Bor-
des et al., 2013). These embeddings capture
the semantic meaning of entities based on
their attributes and the structure of the graph.
For example, the embedding for “adenocarci-
noma” encodes its connections to treatments,
symptoms, and associated genes.
2.Edge Encoding: Relationships (edges) be-
tween nodes are represented as directional
vectors. These are computed by applying
transformation functions to the embeddings
of the connected nodes. For instance, the edge
“treated_with” between a disease node and a
medication node reflects the nature and direc-
tion of the relationship.
Hierarchical tagging further improves graph ef-
ficiency and interpretability. Each graph node is
tagged with categories such as “Symptoms,” “Med-
ications,” or “Patient History,” creating a multi-
level abstraction. During inference, the model
accesses relevant graph layers, ensuring fast and
precise retrieval for tasks that require high-level
summaries and fine-grained details.
The combined encoding of nodes and edges
enables efficient traversal and reasoning over
the graph. By embedding the graph in a high-
dimensional space, the model can retrieve semanti-
cally similar nodes and relations, supporting robustand context-aware clinical predictions.
3.4 Model Implementation and Evaluation
Metrics
The instruction tuning, RAG, and graph-based rea-
soning components are integrated into lightweight
LLaMA variants, creating a unified inference
pipeline. Scalability is evaluated by varying the
number of German instructions and the model size.
Minimal German instructions (100–400 examples)
are used to test cross-lingual adaptability, highlight-
ing how small bilingual datasets influence perfor-
mance. During training, we systematically vary
the instructions to improve the model’s adaptabil-
ity. Lightweight models are compared with larger
variants to assess their performance in resource-
constrained environments.
We evaluate the framework’s performance using
metrics tailored to specific tasks. For entity recog-
nition, relation extraction, and document classifi-
cation, we report the F1 score. For imbalanced
datasets like TCGA-C, we use the area under the
precision-recall curve (AU-PRC) to emphasize per-
formance in uneven class distributions. Binary
tasks, such as TNM staging and treatment response
prediction, are evaluated using the area under the
curve (AUC).
4 Results
We evaluated our instruction-tuned LLMs across
biomedical and oncology tasks. Each model vari-

Table 3: Performance of Models on English and Multilingual Tasks
Model Configuration HoC TCGA-C TCGA-T TCGA-N TCGA-M MSK-IMPACT ICD-10 DUP-T DUP-N DUP-M SNOMED
Type Model EN EN EN EN EN EN DE DE DE DE DE
Base LLM
LLaMA-2-7B 79.32 0.89 0.92 0.91 0.73 0.78 77.02 0.81 0.78 0.73 0.78
LLaMA-3.1-8B 80.12 0.89 0.92 0.92 0.73 0.78 78.58 0.81 0.78 0.74 0.78
LLaMA-3.2-1B 80.01 0.88 0.91 0.90 0.71 0.77 76.16 0.77 0.75 0.71 0.70
LLaMA-3.2-3B 80.09 0.88 0.91 0.90 0.71 0.77 76.53 0.75 0.77 0.71 0.72
Instruction-Tuned
LLaMA-2-7B 82.34 0.89 0.92 0.92 0.74 0.79 82.84 0.82 0.80 0.73 0.82
LLaMA-3.1-8B 83.12 0.90 0.93 0.92 0.74 0.78 83.45 0.83 0.80 0.75 0.81
LLaMA-3.2-1B 82.03 0.88 0.91 0.90 0.73 0.77 80.04 0.78 0.77 0.72 0.72
LLaMA-3.2-3B 82.11 0.88 0.91 0.90 0.73 0.77 80.08 0.75 0.77 0.73 0.72
+RAG
LLaMA-2-7B 82.32 0.89 0.93 0.93 0.74 0.80 82.07 0.83 0.81 0.75 80.00
LLaMA-3.1-8B 83.12 0.89 0.94 0.93 0.75 0.80 82.17 0.83 0.80 0.74 81.00
LLaMA-3.2-1B 82.12 0.88 0.92 0.91 0.73 0.78 79.87 0.80 0.77 0.72 0.72
LLaMA-3.2-3B 82.71 0.89 0.91 0.91 0.73 0.77 80.05 0.80 0.77 0.73 0.72
+Graph-RAG
LLaMA-2-7B 83.11 0.91 0.94 0.93 0.75 0.79 84.86 0.82 0.81 0.74 0.84
LLaMA-3.1-8B 83.84 0.91 0.94 0.93 0.75 0.80 85.75 0.82 0.80 0.74 0.85
LLaMA-3.2-1B 82.76 0.90 0.92 0.92 0.73 0.77 83.00 0.80 0.77 0.72 0.77
LLaMA-3.2-3B 83.04 0.90 0.92 0.91 0.73 0.78 84.18 0.80 0.78 0.73 0.79
ant (LLaMA-2-7B ,LLaMA-3.1-8B ,LLaMA-3.2-1B ,
and LLaMA-3.2-3B ) underwent a progression:
base configuration, instruction tuning, RAG inte-
gration, and Graph-RAG enhancement. We ob-
served general performance boosts at every stage
especially with the oncology tasks. Larger mod-
els like LLaMA-3.1-8B achieved higher accuracy,
but smaller models like LLaMA-3.1-1B remained
competitive as required fewer resources.
Instruction tuning substantially increased F1
scores on standard NER benchmarks. For example,
LLaMA-3.1-8B improved from 86.33% to 89.50%
onNCBI-Disease after tuning, while LLaMA-2-7B
jumped from 83.12% to 86.48% on BC5CDR-
Disease . This tuning step aligned the models
with domain-specific tasks and helped them recog-
nize disease names, biomarkers, and chemical en-
tities. Adding retrieval further refined results. On
BC5CDR-Chem , contextual information reduced
confusion about similar chemical mentions. Graph-
RAG then linked terms to the ontologies, which
resolved ambiguities and improved NER perfor-
mance on more complex datasets like JNLPBA .
Instruction tuning also helped relation extraction.
The models learned to link diseases with treatments
or genetic variants. LLaMA-2-7B reached 93.70%
oni2b2-2010 , while LLaMA-3.1-8B achieved sim-
ilar results. Using RAG and Graph-RAG, the mod-
els matched gene-disease pairs more precisely.
Natural language inference tasks like MedNLI
tested logical reasoning. Instruction-tuned
LLaMA-2-7B improved from 88.76% to 90.57%.
On oncology-specific tasks, the graph-basedmodels excelled. For example, in TNM staging,
Graph-RAG improved entity linking by referenc-
ing established oncology guidelines, boosting F1
scores by 2.5%. This structured reasoning al-
lowed the models to generate consistent, verifi-
able outputs even in complex staging scenarios.
LLaMA-3.1-8B classified biomedical literature into
canonical cancer hallmarks with an F1 of 83.84%.
Linking TNM attributes to known ontologies sup-
ported the model in assigning the correct cate-
gory, raising F1 scores on T, N, and M labels.
LLaMA-2-7B , even though smaller, benefited from
Graph-RAG and produced high AUC values.
Cross-lingual tests revealed the value of even
minimal German instructions on tasks with the
DUP-127 dataset (Figure 2). LLaMA-2-7B im-
proved from 77.02% to 82.84% on ICD-10 cod-
ing by learning from a few hundred German in-
structions. The multilingual tuning also helped
theSNOMED classification and TNM staging in
German. Figure 2 highlights that gains peaked
with 200 instructions, illustrating that even small
bilingual datasets can enable cross-lingual general-
ization.
Scaling the number of German instructions from
100 to 400 led to incremental gains. Perfor-
mance improvements peaked around 200 instruc-
tions, but complex tasks still benefited from 400.
LLaMA-3.1-8B , with its larger capacity, took bet-
ter advantage of these extra instructions. Smaller
models also gained but reached a plateau sooner.
LLaMA-2-7B maintained a balance between effi-
ciency and accuracy, making it attractive for clin-

ical environments with limited computational re-
sources.
Overall, instruction tuning, retrieval, and graph-
based reasoning worked together to produce a flex-
ible model family. The models adapted to new
tasks, integrated domain knowledge at inference
time, and reasoned about complex medical con-
cepts. They achieved these gains without over-
hauling internal parameters whenever new data
emerged and instead leaned on retrieval and graphs
to fetch needed facts on the fly.
5 Discussion
Our findings show the promise of combining in-
struction tuning, retrieval augmentation, and graph-
based knowledge integration for oncology NLP.
The incorporation of a few instructions in an-
other language demonstrated the potential of cross-
lingual capabilities. By using minimal bilingual
training data, our approach bypassed the usual
costs associated with large-scale multilingual train-
ing, offering a practical and scalable solution for
global healthcare systems with diverse linguistic
needs.
Retrieval augmentation added critical agility to
the system, allowing the model to dynamically
access up-to-date information at inference time in-
stead of relying solely on static, parameter-encoded
knowledge. This design enables models to adapt
to evolving oncology guidelines and clinical prac-
tices, which often change multiple times a year. For
example, retrieval mechanisms can help the model
navigate newly introduced therapies or updated
TNM staging criteria without requiring expensive
retraining. The integration of retrieval from trusted
clinical sources highlights its potential in dealing
with incomplete or ambiguous clinical data.
Graph-based knowledge integration improved
the model’s reasoning by structuring relationships
between clinical entities. Rather than merely re-
trieving relevant concepts, the knowledge graph
enabled the model to place these concepts into
a structured context, improving logical reasoning
and reducing errors due to ambiguous terms. This
structured reasoning aligns closely with clinical
workflows, where decisions depend on clear rela-
tionships between diagnoses, treatments, and out-
comes. By linking predictions to specific nodes
in the graph, the model can help with traceability
and explainability, which are crucial for building
clinician trust.Model size played a role in performance. Larger
models, like LLaMA-3.1-8B , excelled in extracting
biomedical entities. However, smaller models like
LLaMA-2-7B achieved comparable results on many
tasks, particularly when supported by retrieval and
graph integration. This trade-off between perfor-
mance and computational cost is especially rel-
evant for resource-constrained settings. Smaller
models, paired with efficient retrieval and graph-
based reasoning, offer a viable pathway for de-
ploying advanced NLP tools in clinics with limited
hardware capabilities.
Our experiments showed diminishing returns
with high instruction counts. After approximately
200 German instructions, improvements plateaued
for simpler tasks, such as ICD-10 coding. Complex
tasks, like TNM staging, showed marginal gains
up to 400 instructions. This finding shows the
importance of tailoring instruction counts to task
complexity and resource availability. Future explo-
ration of instruction prioritization or curriculum
learning could optimize the cost-benefit balance,
ensuring that effort is directed where it yields the
most significant gains.
The cross-lingual modeling approach demon-
strated real-world applicability. Bilingual instruc-
tion tuning, combined with retrieval and knowl-
edge graphs, empowered the model to navigate
clinical texts in another language, even with min-
imal supervision. This adaptability can address
challenges faced by rural or underserved regions
where linguistic diversity often limits access to
advanced clinical technologies. Adding a mod-
est number of domain-specific glossaries or syn-
thetic training examples may further enhance per-
formance on rare or compound medical terminol-
ogy.
Qualitative analyses showed model limitations.
On NER tasks, confusion between biomarkers like
EGFR and HER2 highlighted the need for more
robust contextual disambiguation. Graph-based
reasoning mitigated these issues in part by link-
ing terms to authoritative definitions, yet uncom-
mon or rare entities continued to pose challenges.
Similarly, for TNM staging extraction, the model
excelled with standard terminology but struggled
with vague or non-standard formulations. Retrieval
partially addressed these gaps by surfacing canoni-
cal TNM definitions, while graph integration pro-
vided structured connections between terms and
staging guidelines. However, cases where clinical

Figure 2: Performance scores for Instruction-Tuned Models with 100, 200, and 400 German instructions.
texts themselves lacked clarity remained problem-
atic, underscoring the dependence of NLP systems
on the quality of source data.
Cross-lingual coding introduced unique chal-
lenges. While minimal German instructions helped
the model perform ICD-10 coding and SNOMED
classification tasks, the model occasionally failed
with long compound German words or uncommon
clinical expressions. Refining multilingual embed-
dings and incorporating domain-specific lexicons
could improve outcomes, particularly in handling
rare terminology and idiomatic language usage.
A deeper look at the model’s performance on the
MSK-IMPACT dataset revealed its ability to cor-
rectly match common mutations, such as EGFR, to
appropriate therapies. However, the model strug-
gled with rare genetic variants due to sparse re-
trieval references. In such cases, indirect reasoning
and inference from related mutations proved insuf-ficient. Future work could address this limitation
by integrating curated genomic knowledge bases or
using generative retrieval strategies to synthesize
knowledge from related contexts.
Future directions could refine these methods
further. Expanding multimodal capabilities by
integrating text-based NLP with imaging data,
such as radiology scans or histopathology images,
could create a more comprehensive oncology as-
sistant. Generative retrieval strategies and graph
embedding techniques may raise the performance
ceiling by improving the depth and scope of re-
trieved knowledge. Extending cross-lingual inte-
gration to low-resource languages could address
global disparities in healthcare technology access.
Testing the framework in clinical trials with real-
world practitioners will provide critical insights
into its usability, reliability, and impact on decision-
making.

References
Mohammed Alawad, Shang Gao, John X Qiu, Hong Jun
Yoon, J Blair Christian, Lynne Penberthy, Brent
Mumphrey, Xiao-Cheng Wu, Linda Coyle, and Geor-
gia Tourassi. 2020. Automatic extraction of cancer
registry reportable information from free-text pathol-
ogy reports using multitask convolutional neural net-
works. Journal of the American Medical Informatics
Association , 27(1):89–98.
Mohammed Alawad, Shang Gao, Mayanka Chandra
Shekar, SM Hasan, J Blair Christian, Xiao-Cheng
Wu, Eric B Durbin, Jennifer Doherty, Antoinette
Stroup, Linda Coyle, et al. 2021. Integration of
domain knowledge using medical knowledge graph
deep learning for cancer phenotyping. arXiv preprint
arXiv:2101.01337 .
Emily Alsentzer, John R Murphy, Willie Boag, Wei-
Hung Weng, Di Jin, Tristan Naumann, and Matthew
McDermott. 2019. Publicly available clinical bert
embeddings. arXiv preprint arXiv:1904.03323 .
Rie Kubota Ando. 2007. Biocreative ii gene mention
tagging system at ibm watson. In Proceedings of the
second biocreative challenge evaluation workshop ,
volume 23, pages 101–103. Centro Nacional de In-
vestigaciones Oncologicas (CNIO) Madrid, Spain.
Simon Baker, Ilona Silins, Yufan Guo, Imran Ali, Johan
Högberg, Ulla Stenius, and Anna Korhonen. 2016.
Automatic semantic classification of scientific litera-
ture according to the hallmarks of cancer. Bioinfor-
matics , 32(3):432–440.
Giorgio Bedogni. 2009. Clinical prediction models—a
practical approach to development, validation and
updating.
Antoine Bordes, Nicolas Usunier, Alberto Garcia-
Duran, Jason Weston, and Oksana Yakhnenko.
2013. Translating embeddings for modeling multi-
relational data. Advances in neural information pro-
cessing systems , 26.
Àlex Bravo Serrano, Janet Piñero González, Núria Quer-
alt Rosinach, Michael Rautschka, and Laura I Fur-
long. 2015. Extraction of relations between genes
and diseases from text and large-scale data analy-
sis: implications for translational research. BMC
Bioinformatics. 2015 Feb 21; 16 (1): 55 .
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems , 33:1877–1901.
Nigel Collier, Tomoko Ohta, Yoshimasa Tsuruoka,
Yuka Tateisi, and Jin-Dong Kim. 2004. Introduc-
tion to the bio-entity recognition task at jnlpba. In
Proceedings of the International Joint Workshop on
Natural Language Processing in Biomedicine and its
Applications (NLPBA/BioNLP) , pages 73–78.Jacob Devlin. 2018. Bert: Pre-training of deep bidi-
rectional transformers for language understanding.
arXiv preprint arXiv:1810.04805 .
Rezarta Islamaj Do ˘gan, Robert Leaman, and Zhiyong
Lu. 2014. Ncbi disease corpus: a resource for dis-
ease name recognition and concept normalization.
Journal of biomedical informatics , 47:1–10.
Koji Fujimoto, Morteza Rohanian, Fabio Rinaldi,
Mizuho Nishio, Farhad Nooralahzadeh, Chikako
Tanaka, and Michael Krauthammer. 2023. Classifi-
cation of cancer tnm stage from japanese radiology
report using on-premise llm at ntcir-17 mednlp-sc
rr-tnm subtask.
Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto
Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng
Gao, and Hoifung Poon. 2021. Domain-specific lan-
guage model pretraining for biomedical natural lan-
guage processing. ACM Transactions on Computing
for Healthcare (HEALTH) , 3(1):1–23.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation. arXiv preprint
arXiv:2405.16506 .
Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar
Karnin. 2020. Tabtransformer: Tabular data mod-
eling using contextual embeddings. arXiv preprint
arXiv:2012.06678 .
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with gpus. IEEE
Transactions on Big Data , 7(3):535–547.
Jenna Kefeli, Jacob Berkowitz, Jose M Acitores Cortina,
Kevin K Tsang, and Nicholas P Tatonetti. 2024. Gen-
eralizable and automated classification of tnm stage
from pathology reports with external validation. Na-
ture Communications , 15(1):8916.
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon
Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang.
2020. Biobert: a pre-trained biomedical language
representation model for biomedical text mining.
Bioinformatics , 36(4):1234–1240.
Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sci-
aky, Chih-Hsuan Wei, Robert Leaman, Allan Peter
Davis, Carolyn J Mattingly, Thomas C Wiegers, and
Zhiyong Lu. 2016. Biocreative v cdr task corpus:
a resource for chemical disease relation extraction.
Database , 2016.
Mizuho Nishio, Koji Fujimoto, Fabio Rinaldi, Hidetoshi
Matsuo, Morteza Rohanian, Michael Krautham-
mer, Takaaki Matsunaga, and Farhad Nooralahzadeh.
2023. Zero-shot classification of tnm staging for
japanese radiology report using chatgpt at rr-tnm
subtask of ntcir-17 mednlp-sc.
Drew M Pardoll. 2012. The blockade of immune check-
points in cancer immunotherapy. Nature reviews
cancer , 12(4):252–264.

Omid Rohanian, Mohammadmahdi Nouriborji, Hannah
Jauncey, Samaneh Kouchaki, Farhad Nooralahzadeh,
Lei Clifton, Laura Merson, David A Clifton, IS-
ARIC Clinical Characterisation Group, et al. 2024a.
Lightweight transformers for clinical natural lan-
guage processing. Natural language engineering ,
30(5):887–914.
Omid Rohanian, Mohammadmahdi Nouriborji,
Samaneh Kouchaki, and David A Clifton. 2023.
On the effectiveness of compact biomedical
transformers. Bioinformatics , 39(3):btad103.
Omid Rohanian, Mohammadmahdi Nouriborji,
Samaneh Kouchaki, Farhad Nooralahzadeh, Lei
Clifton, and David A Clifton. 2024b. Exploring the
effectiveness of instruction tuning in biomedical
language processing. Artificial intelligence in
medicine , 158:103007.
Alexey Romanov and Chaitanya Shivade. 2018.
Lessons from natural language inference in the clini-
cal domain. arXiv preprint arXiv:1808.06752 .
S Ruder. 2017. An overview of multi-task learn-
ing in deep neural networks. arXiv preprint
arXiv:1706.05098 .
Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno,
David Stutz, Ellery Wulczyn, Fan Zhang, Tim
Strother, Chunjong Park, Elahe Vedadi, et al. 2024.
Capabilities of gemini models in medicine. arXiv
preprint arXiv:2404.18416 .
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl,
et al. 2023. Large language models encode clinical
knowledge. Nature , 620(7972):172–180.
Eric J Topol. 2019. High-performance medicine: the
convergence of human and artificial intelligence. Na-
ture medicine , 25(1):44–56.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, et al. 2023. Llama: Open and effi-
cient foundation language models. arXiv preprint
arXiv:2302.13971 .
Özlem Uzuner, Brett R South, Shuying Shen, and
Scott L DuVall. 2011. 2010 i2b2/va challenge on
concepts, assertions, and relations in clinical text.
Journal of the American Medical Informatics Associ-
ation , 18(5):552–556.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, and Vicente Grau. 2024.
Medical graph rag: Towards safe medical large lan-
guage model via graph retrieval-augmented genera-
tion. arXiv preprint arXiv:2408.04187 .
Ahmet Zehir, Ryma Benayed, Ronak H Shah, Aijazud-
din Syed, Sumit Middha, Hyunjae R Kim, Preethi
Srinivasan, Jianjiong Gao, Debyani Chakravarty,Sean M Devlin, et al. 2017. Mutational landscape
of metastatic cancer revealed from prospective clini-
cal sequencing of 10,000 patients. Nature medicine ,
23(6):703–713.
Sicheng Zhou, Nan Wang, Liwei Wang, Hongfang Liu,
and Rui Zhang. 2022. Cancerbert: a cancer domain-
specific language model for extracting breast cancer
phenotypes from electronic health records. Journal
of the American Medical Informatics Association ,
29(7):1208–1216.