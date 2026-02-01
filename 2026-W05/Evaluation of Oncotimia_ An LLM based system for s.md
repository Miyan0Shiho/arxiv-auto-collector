# Evaluation of Oncotimia: An LLM based system for supporting tumour boards

**Authors**: Luis Lorenzo, Marcos Montana-Mendez, Sergio Figueiras, Miguel Boubeta, Cristobal Bernardo-Castineira

**Published**: 2026-01-27 18:59:38

**PDF URL**: [https://arxiv.org/pdf/2601.19899v1](https://arxiv.org/pdf/2601.19899v1)

## Abstract
Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality.

## Full Text


<!-- PDF content starts -->

EVALUATION OFONCOTIMIA:ANLLM-BASED SYSTEM FOR
SUPPORTING TUMOURBOARDS
TECHNICALREPORT
Luis Lorenzo1, Marcos Montaña-Méndez1, Sergio Figueiras1, Miguel Boubeta1,∗, Cristóbal Bernardo-Castiñeira1,∗
1Innovation Department, Bahía Software SLU, Ames (A Coruña), Spain
∗Corresponding author(s):miguel.boubeta@bahiasoftware.es, cristobal.bernardo@bahiasoftware.es
January 28, 2026
ABSTRACT
Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require
manual processes and structuring large volumes of heterogeneous clinical information, resulting in
a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure
clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows
and evaluate its application to the automatic completion of lung cancer tumour board forms using large
language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector
storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform
unstructured clinical documentation into structured and standardised tumour board records. We assess
the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring
both completion form accuracy and end-to-end latency. The results demonstrate high performance
across models, with the best performing configuration achieving an 80% of correct field completion
and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best
accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM-
assisted autocompletion form is technically feasible and operationally viable in multidisciplinary
lung cancer workflows and support its potential to significantly reduce documentation burden while
preserving data quality.
KeywordsGenAI ·LLMs ·Vector database ·Embeddings ·RAG ·Tumour boards ·Lung cancer form autocompletion
1 Introduction
In recent years, advances in transformer-based architectures have firmly established LLMs as a foundational technology
in biomedical informatics. Early developments in general-purpose LLMs (e.g., GPT-3 and its successors) revealed
emergent capabilities in clinical summarisation, question answering, report generation, coding support and contextual
reasoning (Brown et al., 2020). Subsequent work has shown that domain-adapted models, such as BioGPT (Luo
et al., 2022), BioMedLM, PubMedBERT (Gu et al., 2021) and Med-PaLM (Singhal et al., 2023), achieve expert-
level performance on diverse medical reasoning benchmarks. An expanding evidence base further demonstrates
that LLMs can reliably extract salient clinical information and support guideline-informed recommendations when
deployed with appropriate safeguards. Recent prospective evaluations in hospital settings indicate that LLM-assisted
clinical documentation can meaningfully reduce clinician workload while maintaining high linguistic quality (Bracken
et al., 2025; Nori et al., 2023). Nevertheless, these systems continue to require rigorous oversight owing to risks of
hallucinations, incomplete contextualisation and occasional misinterpretation of clinical guidelines.
In medicine, multidisciplinary management (MDM) offers cancer patients the advantage of having specialists from
different medical fields collaboratively involved in treatment planning. This approach is usually implemented through
multidisciplinary clinics, such as breast units, where various experts assess patients, perform physical examinations,
request and conduct diagnostic tests efficiently, and jointly evaluate potential treatment options. MDM is also conducted
through multidisciplinary tumour board (MDTB) meetings, which are structured sessions in which all relevant patientarXiv:2601.19899v1  [cs.CL]  27 Jan 2026

information is collected, and key specialists convene to discuss the diagnosis and management of cancer patients
(El Saghir et al., 2014). However, studies show that clinicians spend significant effort managing and analysing
information within electronic health records (EHRs), reducing the time for direct patient care, especially in high-
complexity fields such as oncology (Arndt et al., 2017). MDTBs are particularly affected because each case requires the
preparation of standardised summaries, structured staging information, pathology details, radiological interpretations,
biomarker data, allergy profiles and records of prior treatment, demands that are specially challenging in settings with
limited staff resources. The fragmentation of data across narrative notes, laboratory systems, pathology platforms and
PACs often results in manual information retrieval and redundant re-entry of variables into MDTB case forms.
Automating pieces of this workflow is therefore both operationally compelling and clinically relevant. Early efforts
using traditional natural language processing (NLP) methods demonstrated benefit in extracting structured information
from radiology or pathology reports (Wang et al., 2018). The transition from rule-based NLP to LLM-enabled generative
systems, in addition to extracting information, also allows it to be synthesised into coherent drafts aligned with medical
guidelines.
Autocompletion in clinical documentation has emerged as a promising application of LLMs. Initial experiments in
general EHR contexts have shown that LLMs can assist with automatic drafting of the main complaints, suggesting
phrasing for assessment and plan sections, and generating templated texts conditioned on structured inputs Ayers et al.
(2023). These systems have demonstrated that LLM-driven autocompletion improves efficiency and reduces repetitive
typing. However, studies explicitly examining autocompletion for oncology MDTB forms remain extremely limited.
To date, most published work in oncology has focused on information extraction (e.g., stage from clinical notes) or
summarisation of radiology reports. Some recent pilot studies have explored the use of LLMs to generate oncology
case summaries or harmonise staging descriptions Chen et al. (2025), but standardised autocompletion of MDTB forms,
particularly in lung cancer, has not yet been rigorously evaluated in prospective settings. This represents a critical
evidence gap given the structured, repetitive and data-dense nature of these forms and their importance in treatment
decision making.
Lung cancer has been one of the earliest and most active oncology domains for AI research due to the abundance of
imaging, molecular data and clinical texts. NLP and deep learning models have been applied to staging extraction,
biomarker result interpretation, automatic radiology summarisation and automated assessment of eligibility for targeted
therapy or clinical trials (Esteva et al., 2019; Hu et al., 2021; Aldea et al., 2025).
In this work, our objective is to describe and technically evaluate the performance of ONCOTIMIA, a modular and
secure LLM-based tool that integrates RAG and a rule -driven adaptive form model to automate the completion of lung
cancer tumour board forms. We assess its performance in a realistic tumour board setting, using synthetic but clinically
representative cases. Six state -of-the-art LLMs are evaluated in terms of form -completion accuracy and end -to-end
latency. Through this study, we aim to provide empirical evidence on the technical and operational feasibility of
GenAI -assisted autocompletion within oncology workflows, and to demonstrate its potential to reduce documentation
burden while maintaining data quality.
The following Section 2 introduces the ONCOTIMIA platform, outlining its architecture, data ingestion pipeline, and
the design of lung cancer data schema. Section 3 summarises the methodology for generating medical data records, the
RAG workflow and the selection criteria for LLMs. Section 4 reports the performance evaluation results, and Section 5
concludes by highlighting key findings, limitations, and directions for future work.
2 ONCOTIMIA tool description
ONCOTIMIA is a modular system that integrates generative AI to support tumour board workflows and reduce
documentation burden. Its core functionality focuses on the automatic autocompletion of standardised tumour board
forms and the generation of structured patient summaries from heterogeneous clinical sources. The system also
incorporates information retrieval and RAG -assisted reasoning modules to facilitate case preparation; these features are
intended to support review and do not replace clinical judgment. In this section, we present the system architecture and
clinical data ingestion process and the definition of lung cancer form used in tumour boards committee workflows.
2.1 System architecture
The architecture of ONCOTIMIA has been conceived as a modular, scalable, and secure infrastructure designed to enable
the seamless integration of GenAI into oncology workflows. Its design adheres to the principles of interoperability,
traceability, maintainability, and controlled evolution, thereby ensuring long-term sustainability in complex clinical
environments undergoing continuous technological transformation. From a conceptual standpoint, the ONCOTIMIA
architecture (see Figure 1 for more details) is structured around a set of interconnected yet decoupled modules that
2

operate in a coordinated manner through standardised interfaces and secure communication protocols. This modular
arrangement enables the independent evolution of system components and the incremental incorporation of new
functionalities without compromising overall system stability.
Figure 1: ONCOTIMIA architecture. (i) Data ingestion layer, (ii) storage subsystem, (iii) backend services, (iv) LLM
abstraction layer and (v) reverse proxy.
Its design principles emphasise interoperability, traceability, maintainability and controlled evolution, ensuring robust-
ness in complex clinical environments undergoing continuous technological change. The ONCOTIMIA architecture
is structured into five core modules: (i) data ingestion layer, (ii) storage subsystem, (iii) backend services, (iv) LLM
abstraction layer and (v) reverse proxy.
The ingestion layer serves as the system’s entry point, acquiring and validating data from heterogeneous clinical sources,
including electronic health records, structured and unstructured reports, laboratory results and administrative datasets.
It implements automated pipelines for extraction, cleaning and normalisation to guarantee data quality and consistency
prior to downstream processing. The use of mature data-engineering ecosystems enables reproducible, auditable and
standard-compliant ingestion workflows, with native support for healthcare interoperability standards. The storage
subsystem provides persistent, secure and traceable management of clinical data. ONCOTIMIA adopts a multilayered
data lake structure comprising: (1) a landing layer for preserving source data in native formats, (2) a staging layer where
data are transformed into analytically consistent representations, and (3) a refined layer optimised for high-performance
queries and GenAI-driven retrieval tasks. The refined layer integrates both relational storages, suited for structured
clinical variables, and vector databases supporting semantic search and embedding-based retrieval. The backend, built
around a microservices paradigm, encapsulates the business logic required to support core tumour-board use cases. Key
functionalities include:
• Automatic summarisation of clinical histories, combining structured and unstructured inputs.
•Autocompletion of tumour-specific forms, converting narrative text into semantically normalised representa-
tions aligned with oncology terminologies.
•A clinical assistant module, enabling question answering, hypothesis exploration and guideline-informed
decision support.
An intermediate abstraction layer mediates interactions between backend services and LLMs. It translates clinical
requests into model-compliant queries, enforces safety and audit constraints and standardises output formatting. This
layer enables model interchangeability and facilitates the integration of retrieval components and domain-specific
knowledge bases. A reverse proxy manages traffic routing across system components. It enforces security policies, load
balancing and rate limiting, while supporting real-time monitoring and audit logging. This layer ensures controlled and
secure exposure of services to external and internal clients.
3

2.2 Data ingestion and ETL processes
The clinical data ingestion process constitutes the entry point of the data processing pipeline and relies on a data
lake architecture designed to efficiently and securely manage the heterogeneity of hospital information sources.
This infrastructure supports the integration of both structured data (e.g., administrative records, laboratory results,
demographic variables) and unstructured data (e.g., radiological reports, medical notes, clinical guidelines, oncology
protocols, and supplementary documentation). The primary objective of this ingestion layer is to ensure complete
preservation of the original content while enforcing quality-control mechanisms, format validation, and metadata
generation to maintain full traceability of the information flow.
The data lake is organised into three functional layers (landing, staging and refined) which reflect the progressive
transformation of data from raw inputs to curated, analysis-ready outputs (see Figure 1 for more details). Documents
received from hospital information systems or authorised external sources are stored in the landing layer, maintaining
their original formats (e.g., .pdf, .docx or .txt) to preserve auditability and end-to-end traceability.
Data stored in the landing layer feeds a set of sequential ETL (Extract, Transform and Load) processes implemented
in Python, which constitute the operational backbone of the pipeline. First ETL process validates formats, applies
integrity checks, and generates technical metadata during initial ingestion. The second ETL process extracts content
and metadata from documents into the staging layer using specialised LangChain loaders (e.g., Docx2txtLoader and
PyPDFLoader), followed by text cleaning (e.g., removal of line breaks or non-informative characters), tokenisation,
lemmatisation, and stemming. The third and fourth ETL processes populate the refined layer by loading curated
unstructured data into a vector storage system and structured patient information into a relational database. Unstructured
text is encoded as semantic embeddings using the Nomic model and stored in a Qdrant vector store, enabling contextual
RAG-based pipelines, and AI-assisted reasoning. On the other hand, structured and normalised clinical variables (e.g.,
demographics, coded diagnoses, tumour staging, and treatments) are stored in a PostgreSQL database, which serves
as the analytical source of truth. The coexistence of relational and vectorial storage provides a hybrid integration of
explicit clinical knowledge and contextual information derived from language models.
2.3 Lung cancer form schema
The MDTB lung cancer data form is organised into seven blocks, defined as logical units that group multiple questions.
Transitions between blocks follow a non-linear, rule-based logic, whereby responses to specific questions determine the
activation, omission, or redirection of subsequent sections. This adaptive structure allows the form to dynamically adjust
to individual patient characteristics and the clinical context under evaluation. The form captures key clinical domains,
including demographic information, smoking status and other risk factors, radiological and pathological findings,
molecular biomarkers relevant to precision oncology, and prior treatments with their therapeutic intent. The overall
structure is anchored by Block 1, which functions as the central node of the form. Block 1 consolidates core clinical
variables (risk factors, comorbidities, ongoing medication, diagnostic test results, and detailed tumour profiling), and
conditionally activates the remaining blocks based on pivotal responses. The main components of this block include:
•Patient characteristics and medical history: smoking status, comorbidities, allergies and prior malignancies.
•Functional status and previous therapies: ECOG performance score, radiotherapy or chemotherapy, and
documented treatment refusals.
•Imaging and endoscopic assessment: free-text summaries of CT, PET-CT, bronchoscopy and other relevant
procedures.
•Histopathological and molecular analysis: tumour histology, molecular biomarkers, PD-L1 expression,
tumour mutational burden (TMB) and microsatellite status.
•Tumour staging: standardised categories capturing local, locoregional and systemic disease spread.
Responses collected in Block 1 conditionally determine the activation of subsequent blocks collected in Table 1,
enabling an adaptive and patient-specific data collection workflow. A reported history of malignancy activates Block 2
to capture details of earlier cancer diagnoses. Documentation of treatment refusal triggers Block 3 while indication of
disease recurrence activates Block 4 to record affected sites. When a rebiopsy has been performed, Block 5 is enabled
to document updated histology and molecular markers. A history of radiotherapy activates Block 6, which captures
treatment intent, target lesions and timelines, whereas prior systemic therapy triggers Block 7 to document administered
agents, therapeutic intent and treatment dates. This modular, conditionally driven design ensures that only clinically
relevant information is collected for each case, resulting in a structured yet flexible representation of patient data. Such
a context-aware data model supports downstream tasks including LLM-assisted summarisation, retrieval-augmented
reasoning, and clinical decision support within multidisciplinary tumour board workflows.
4

Table 1: Description of the content of Blocks 2 to 7 derived from Block 1.
Block Description
2 Previous neoplasms
3 Treatment refusal
4 Recurrence
5 Rebiopsy and new biomarkers
6 Radiotherapy
7 Chemotherapy
Table 2 summarises the core clinical variables included in the lung cancer form, organised into thematic sections that
reflect the logical structure of the oncological assessment process. Each section groups related fields according to their
clinical meaning and functional role within the data model, while explicitly specifying the corresponding data type to
ensure consistency, interpretability and suitability for downstream computational processing.
The medical history section includes key baseline variables that characterise the patient’s background and prior clinical
context. This section records smoking status encoded as a categorical variable with three possible states (smoker,
non-smoker and ex-smoker), alongside binary indicators of previous neoplasia and documented treatment refusal. In
addition, patient medication is represented as a categorical field, allowing for the structured documentation of the
current medication (e.g., oral anticoagulation, antiplatelet agents, etc.).
The performance status section captures the patient’s functional condition through the ECOG performance status core,
encoded as an integer ranging from 0 (fully active) to 5 (dead). This variable is widely used in oncology to assess a
patient’s ability to tolerate systemic treatments.
Table 2: Subset of core clinical fields for the lung cancer form.
Section Field Data type
Medical historySmoking status Categorical
Previous neoplasia Boolean
Treatment refusal Boolean
Patient medication Categorical
Performance status ECOG value Integer
DiagnosisLocal location Categorical
Locoregional location Categorical
Systemic location Categorical
Hystology type Categorical
Molecular marker Categorical
PD-L1 value Float
Recurrence Boolean
Rebiopsy Boolean
TreatmentRadiotherapy Boolean
Chemotherapy Boolean
The diagnosis section constitutes the most extensive component of the table and encompasses variables describing
disease localisation, pathological characterisation and molecular profiling. Tumour extent is represented through cate-
gorical fields capturing local, locoregional and systemic involvement (e.g., lung, bone, liver, etc.). Histological tumour
type is recorded as a categorical variable, enabling standardised classification of lung cancer subtypes (adenocarcinoma,
squamous cell carcinoma, large cell carcinoma and small cell carcinoma). Molecular characterisation is incorporated
through a categorical molecular marker field complemented by a continuous PD-L1 expression value that records
5

quantitative immunohistochemical values, generally reported as the percentage of tumour cells expressing the marker.
Molecular biomarkers are encoded as categorical variables that explicitly represent both the presence and absence of
clinically actionable genomic alterations (e.g., EGFR, ALK, KRAS, BRAF, ROS1, etc.). The section also includes
Boolean indicators for tumour recurrence and rebiopsy, which are essential for documenting disease evolution and the
availability of updated pathological or molecular information.
Finally, the treatment section records prior oncological interventions, specifically radiotherapy and chemotherapy, both
encoded as Boolean variables. These fields provide a concise representation of previous treatment exposure and serve
as key triggers for conditional workflows and more detailed treatment-specific documentation elsewhere in the system.
3 Materials and methods
Due to the highly sensitive nature of patient health data and the strict regulatory constraints governing its use, no
real patient records were directly employed in the experimental evaluation of the proposed system. Instead, a fully
synthetic dataset was constructed, using as a starting point a real clinical history that had been previously and irreversible
anonymised in accordance with applicable data protection regulations. This reference case was used exclusively as a
structural and narrative template, without retaining any real patient-identifiable or clinically traceable information.
A total of ten Spanish synthetic clinical histories were generated, reflecting the real operational language of the clinical
environment. Given the exploratory nature of this study and the need to assess the performance of different LLMs under
controlled conditions, this initial experiment was intentionally limited in scope as a proof of concept to future large-scale
deployment in real clinical settings. The synthetic cohort was generated using the Qwen3-14b LLM executed locally via
the Ollama framework. This choice was motivated by the need to ensure full data governance and prevent any external
data leakage. The model was prompted to produce multiple clinically plausible, internally consistent, and representative
Spanish lung cancer patient histories that reflected the diversity of cases typically discussed in MDTBs, including
variations in staging, molecular profiles, prior treatments, and clinical evolution. Following this initial generation phase,
a two-step validation and refinement process was applied to ensure medical coherence and internal consistency. First,
an automated reflection-based validation step was performed using GPT-OSS-120b model, which was tasked with
critically reviewing each synthetic clinical history to detect logical inconsistencies, missing information, temporal
contradictions, or medically implausible statements. The model was instructed to either confirm the coherence of the
case or propose corrective revisions, which were then applied to the dataset. Second, the resulting synthetic cases were
subjected to a final manual review by an expert in oncology, who assessed their clinical plausibility, internal consistency
and suitability for use in a simulated tumour board setting. Only cases that passed this expert review were included in
the final evaluation dataset.
This multi-stage process ensures that the resulting dataset, preserves a high degree of clinical realism and complexity,
making it suitable for a meaningful and rigorous assessment of the proposed system, based on a RAG architecture
specifically designed for the automatic completion of structured lung cancer tumour board forms from unstructured clin-
ical narratives in Spanish. The architecture integrates three main components: (i) document ingestion and preprocessing
pipeline, (ii) a hybrid storage and retrieval layer, and (iii) an LLM-based generation layer.
Clinical narratives are first segmented, normalised and embedded into a dense vector space using the Nomic embedding
model. These embeddings are stored in a Qdrant database. At inference time, for each target form block, a query is
constructed and use to retrieve the most relevant textual fragments from the vector storage. These retrieval contexts are
then injected into a structured prompt template together with explicit instructions and the schema of the target form
fields. This RAG-based approach also enables traceability, as each generated field can be linked back to the specific
source fragments that supported it.
Six LLMs models were selected for the experimental evaluation: GPT-OSS-20b, GPT-OSS-120b, Mistral-large-2402-
v1, Pixtral-large-2502-v1, Qwen3-32b and Qwen3-next-80b. This set was designed to provide a representative and
methodologically sound benchmark across different architectural families, parameter scales, and deployment profiles.
GPT-OSS-20b and GPT-OSS-120b enable a controlled analysis of the impact of model scale within a single architectural
lineage, isolating the effect of parameter count on extraction accuracy and reasoning stability. Mistral-Large-2402-v1
was included as a strong general-purpose model optimised for long-context understanding and complex reasoning,
which is essential given the length and heterogeneity of the clinical narratives. Pixtral-Large-2502-v1 was selected
for its strengths in structured generation and schema-constrained reasoning, which closely match the requirements
of mapping free text to a predefined clinical form. Finally, Qwen3-32b and Qwen3-Next-80b were selected as high-
performing open-weight models that combine strong predictive performance with operational feasibility, enabling
reliable deployment in on-premise or tightly controlled infrastructures, as required in regulated clinical environments.
6

Each synthetic clinical case was processed independently by the system, and each of the six models was used as the
generation component within the same RAG pipeline, ensuring that all other components of the system remained strictly
identical. This design isolates the effect of the language model itself on the quality and latency of the generated outputs.
The automatically completed forms were then compared against the ground-truth structured information associated with
each synthetic case. Field-level accuracy metrics were computed, and latency measurements were recorded end-to-end
for each inference.
4 Results
The performance of the proposed form autocompletion tool was evaluated on the 10 synthetically generated lung cancer
histories using the 6 selected LLMs available through AWS Bedrock (GPT-OSS-20b, GPT-OSS-120b, Mistral-large-
2402-v1, Pixtral-large-2502-v1, Qwen3-32b and Qwen3-next-80b). For each LLM, we have assessed two dimensions:
(i) accuracy, quantified as the percentage of correctly completed fields and (ii) model latency, measured as end-to-end
response time (in seconds) per form.
Figure 2: Boxplots of accuracies in %(A) and latencies in seconds (B) computed over N= 10 clinical cases per model.
Boxes represent the interquartile range (IQR), the central line indicates the median, and whiskers extend to the most
extreme values within 1.5×IQR . Dots denote outliers. The dotted line connects the mean values obtained for each
model.
Figure 2 (A) presents the boxplots of accuracy and the corresponding mean values across the ten lung cancer cases
discussed by the tumour board for each evaluated model. As can be observed, the automatic lung cancer form completion
system achieves consistently high accuracy levels, thereby demonstrating the practical feasibility of LLM-assisted
form autocompletion in the context of multidisciplinary lung cancer tumour committees. Overall, larger models tend
to exhibit superior performance. The highest mean accuracies were obtained with the Pixtral-large-2502-v1 model
(80%) and with the GPT-OSS-120b, Qwen3-32b, and Qwen3-120b models, all of which reached a mean accuracy of
79.3% . In contrast, the lowest mean accuracy was observed for the GPT-OSS-20b model ( 72.1% ). Regarding result
stability, the lowest standard deviations were achieved by Qwen3-32b (5.3), Qwen3-80b ( 6.3), Pixtral-large-2502-v1
(6.6), and GPT-OSS-120b ( 7.1), indicating more consistent performance across cases, whereas the highest variability
was observed for GPT-OSS-20b, with a standard deviation of10.4.
7

Latency distributions for the evaluated LLMs are shown in Figure 2 (B). Two clearly distinct behaviours can be
identified. Most models (GPT-OSS-120b, Mistral-large-2402-v1, Pixtral-large-2502-v1, Qwen3-32b and Qwen3-120b)
operate within a narrow and homogeneous latency range with mean values concentrated around 20−21 seconds.
In contrast, GPT-OSS-20b exhibits a markedly higher latency, with a mean of 54seconds and substantially greater
variability. GPT-OSS-120b achieves a latency measure comparable to smaller models, indicating that inference time is
driven more by development and serving optimisations than by model size. Mistral family models show the lowest
and most stable latency, while Qwen3 variants display similar performance. By contrast, GPT-OSS-20b constitutes an
operational outlier and is poorly suited for time-sensitive workflows due to its excessive and unstable response times.
5 Conclusions
This work demonstrates the feasibility, robustness and practical relevance of using LLMs to automate the completion of
lung cancer tumour board forms within a clinical infrastructure. By integrating a modular data ingestion pipeline, a
hybrid relational-vector storage layers and a rule-driven adaptive form model under a RAG architecture, ONCOTIMIA
provides a comprehensive and extensible tool for structured clinical documentation powered by GenAI.
The experimental evaluation across six LLMs provided by AWS Bedrock represents a promising step toward scalable
AI-assisted clinical documentation in precision oncology and shows that LLM-assisted autocompletion can achieve
high and stable accuracy, approaching 80% correct field completion in tumour board cases. The results also reveal that
inference latency is not strictly correlated with model size, since the largest and modern LLMs can deliver response
times comparable to smaller systems, making them suitable for clinical use in asynchronous or semi-interactive
workflows. From a care perspective, the proposed approach directly addresses one of the main operational bottlenecks in
multidisciplinary oncology, and particularly in the implementation of MDTB in hospitals with limited staff resources for
the preparation of cases. ONCOTIMIA reduces the manual, repetitive, and error-prone transcription of heterogeneous
clinical information into structured forms frequently assigned to clinical staff. By automating a substantial portion of
this process, the system has the potential to reduce clinician workload, improve data consistency and accelerate case
preparation without altering existing clinical decision workflows.
Nevertheless, several limitations remain. The current evaluation is based on a limited number of cases and focuses
primarily on technical performance metrics. Future work will include larger prospective studies, fine-grained error
analysis (e.g., by clinical category) and formal assessment of time savings and user acceptance in real tumour board
settings. Finally, further research is needed to strengthen safety guarantees, traceability and explainability, especially in
the presence of model hallucinations or incomplete source documentation.
Funding sources
This work is part of project ONCOTIMIA (BAHIA SOFTWARE), that was supported by the IG408M-IA360 program
under grant number IG408M-2025-000-000021, funded by the Instituto Galego de Promoción Económica (IGAPE) and
cofinanced by the Autonomous Community of Galicia ( 25%) and the European Union through the Recovery and Re-
silience Facility ( 75%), within the framework of the Recovery, Transformation and Resilience Plan-NextGenerationEU,
Component 16: National Artificial Intelligence Strategy.
References
Aldea, M., Rotow, J. K., Arcila, M., Hatton, M., Sholl, L., Rolfo, C., Tagliamento, M., Radonic, T., Schalper, K. A.,
Subbiah, V ., Malapelle, U., Roden, A. C., Manochakian, R., Tsao, M.-S., Linardou, H., Hui, R., Novello, S.,
Greystoke, A., Saqi, A., Lantuejoul, S., Hwang, D. M., Nevins, K., Wynes, M., Waqar, S., Han, Y ., Yatabe, Y .,
Chang, W.-C., Hayashi, T., Kim, T.-J., Hofman, P., Tavora, F., Hirsch, F. R., Denninghoff, V ., Leighl, N. B., Drilon,
A., Cooper, W. A., Dacic, S., Mohindra, P., Pavlakis, N., and Lopez-Rios, F. (2025). Molecular tumor boards: A
consensus statement from the international association for the study of lung cancer.Journal of Thoracic Oncology,
20(11):1594–1614.
Arndt, B. G., Beasley, J. W., Watkinson, M. D., Temte, J. L., Tuan, W. J., Sinsky, C. A., and Gilchrist, V . J. (2017). Teth-
ered to the ehr: Primary care physician workload assessment using ehr event log data and time-motion observations.
Annals of family medicine, 15(5):419–426.
Ayers, J. W., Poliak, A., Dredze, M., Leas, E. C., Zhu, Z., Kelley, J. B., Faix, D. J., Goodman, A. M., Longhurst, C. A.,
Hogarth, M., and Smith, D. M. (2023). Comparing physician and artificial intelligence chatbot responses to patient
questions posted to a public social media forum.JAMA Internal Medicine, 183(6):589–596.
8

Bracken, A., Reilly, C., Feeley, A., Sheehan, E., Merghani, K., and Feeley, I. (2025). Artificial intelligence (ai) –
powered documentation systems in healthcare: A systematic review.J Med Syst, 49:28.
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., Agarwal, S., Herbert-V oss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J.,
Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S.,
Radford, A., Sutskever, I., and Amodei, D. (2020). Language models are few-shot learners.CoRR, abs/2005.14165.
Chen, D., Rod, P., Karl, S., John-Jose, N., Andrew, C., S, B. D., Liu, F.-F., and Raman, S. (2025). Large language
models in oncology: a review.BMJ Oncology, 4:e000759.
El Saghir, N. S., Keating, N. L., Carlson, R. W., Khoury, K. E., and Fallowfield, L. (2014). Tumor boards: optimizing
the structure and improving efficiency of multidisciplinary management of patients with cancer worldwide.American
Society of Clinical Oncology educational book. American Society of Clinical Oncology, Annual Meeting:e461–e466.
Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V ., DePristo, M., Chou, K., Cui, C., Corrado, G., Thrun, S., and
Dean, J. (2019). A guide to deep learning in healthcare.Nat Med, 25:24–29.
Gu, Y ., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., and Poon, H. (2021). Domain-
specific language model pretraining for biomedical natural language processing.ACM Trans. Comput. Healthcare,
3(1).
Hu, D., Zhang, H., Li, S., Wang, Y ., Wu, N., and Lu, X. (2021). Automatic extraction of lung cancer staging information
from computed tomography reports: Deep learning approach.JMIR medical informatics, 9(7):e27955.
Luo, R., Sun, L., Xia, Y ., Qin, T., Zhang, S., Poon, H., and Liu, T.-Y . (2022). Biogpt: generative pre-trained transformer
for biomedical text generation and mining.Briefings in Bioinformatics, 23(6):bbac409.
Nori, H., King, N., McKinney, S. M., Carignan, D., and Horvitz, E. (2023). Capabilities of gpt-4 on medical challenge
problems.ArXiv, abs/2303.13375.
Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S.,
Payne, P., Seneviratne, M., Gamble, P., Kelly, C., Scharli, N., Chowdhery, A., Mansfield, P., y Arcas, B. A., Webster,
D., Corrado, G. S., Matias, Y ., Chou, K., Gottweis, J., Tomasev, N., Liu, Y ., Rajkomar, A., Barral, J., Semturs,
C., Karthikesalingam, A., and Natarajan, V . (2023). Large language models encode clinical knowledge.Nature,
620:172–180.
Wang, Y ., Wang, L., Rastegar-Mojarad, M., Moon, S., Shen, F., Afzal, N., Liu, S., Zeng, Y ., Mehrabi, S., Sohn, S., and
Liu, H. (2018). Clinical information extraction applications: A literature review.Journal of biomedical informatics,
77:34–49.
9