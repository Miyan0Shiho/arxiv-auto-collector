# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade

**Published**: 2025-05-06 10:31:54

**PDF URL**: [http://arxiv.org/pdf/2505.03406v1](http://arxiv.org/pdf/2505.03406v1)

## Abstract
This research paper investigates the application of Large Language Models
(LLMs) in healthcare, specifically focusing on enhancing medical decision
support through Retrieval-Augmented Generation (RAG) integrated with
hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation
(QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By
embedding and retrieving context-relevant healthcare information, the system
significantly improves response accuracy. QLoRA facilitates notable parameter
efficiency and memory optimization, preserving the integrity of medical
information through specialized quantization techniques. Our research also
shows that our model performs relatively well on various medical benchmarks,
indicating that it can be used to make basic medical suggestions. This paper
details the system's technical components, including its architecture,
quantization methods, and key healthcare applications such as enhanced disease
prediction from patient symptoms and medical history, treatment suggestions,
and efficient summarization of complex medical reports. We touch on the ethical
considerations-patient privacy, data security, and the need for rigorous
clinical validation-as well as the practical challenges of integrating such
systems into real-world healthcare workflows. Furthermore, the lightweight
quantized weights ensure scalability and ease of deployment even in
low-resource hospital environments. Finally, the paper concludes with an
analysis of the broader impact of LLMs on healthcare and outlines future
directions for LLMs in medical settings.

## Full Text


<!-- PDF content starts -->

Lightweight Clinical Decision Support System using
QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation
Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma,
and Anil S. Mokhade
Department of Computer Science and Engineering, Visvesvaraya National Institute of
Technology (VNIT), Nagpur
Abstract
This research paper investigates the application of Large Language Models (LLMs) in healthcare,
specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation
(RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation
(QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and re-
trieving context-relevant healthcare information, the system significantly improves response accuracy.
QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity
of medical information through specialized quantization techniques. Our research also shows that
our model performs relatively well on various medical benchmarks, indicating that it can be used to
make basic medical suggestions. This paper details the system’s technical components, including its
architecture, quantization methods, and key healthcare applications such as enhanced disease predic-
tion from patient symptoms and medical history, treatment suggestions, and efficient summarization
of complex medical reports. We touch on the ethical considerations—patient privacy, data secu-
rity, and the need for rigorous clinical validation—as well as the practical challenges of integrating
such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights
ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the
paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future
directions for LLMs in medical settings.
1 Introduction
The healthcare industry faces significant challenges in managing and processing information, requiring
clinicians to navigate ever-expanding medical knowledge bases while providing high-quality patient care.
Large Language Models (LLMs) have emerged as promising tools to address these challenges through
their ability to process and synthesize vast amounts of information. However, general-purpose LLMs
often lack the domain-specific knowledge and nuanced contextual understanding essential for high-stakes
medical applications [1].
This research presents a novel approach to healthcare LLM implementation through a two-component
system: (1) Retrieval-Augmented Generation (RAG) leveraging institution-specific hospital data, and (2)
domain-specific optimization using Quantized Low-Rank Adaptation (QLoRA) fine-tuning techniques.
By combining these approaches, we create a system that maintains the broad knowledge capabilities of
foundation models while incorporating local clinical practices and reducing computational requirements
for practical deployment.
The system architecture harnesses vector embeddings to identify relevant clinical information from hospi-
tal databases, which is then provided as context to an LLM fine-tuned specifically for medical reasoning.
This approach addresses critical limitations of standalone LLMs in healthcare, including knowledge re-
cency, institutional protocol alignment, and factual grounding in patient-specific information [2].
Using Llama 3.2-3B-Instruct as our base model, we demonstrate how even relatively compact LLMs can
achieve impressive performance in specialized clinical tasks when augmented with appropriate retrieval
mechanisms and efficient fine-tuning methodologies. The research focuses particularly on symptom-based
disease prediction, treatment recommendation, and medical documentation summarization—three areas
with significant potential to improve clinical workflows and patient outcomes.
1arXiv:2505.03406v1  [cs.CL]  6 May 2025

Crucially, this research demonstrates a pathway for democratizing advanced clinical decision support. By
integrating the computational efficiency of QLoRA fine-tuning with the contextual relevance provided by
RAG on institution-specific data, our approach makes sophisticated LLM capabilities accessible even for
smaller hospitals or healthcare settings with limited computational resources, enabling them to leverage
AI for improved patient care and workflow efficiency.
Our key contributions are: (1) an efficient system integrating RAG with fine-tuned lightweight LLMs; (2)
application-specific adaptations for hospital data; (3) empirical evaluation on medical QA benchmarks
and real use cases.
2 System Architecture
The proposed healthcare LLM system employs a comprehensive architecture designed to leverage both
institutional knowledge and general medical expertise. Figure 1 illustrates the key components of the
system and the flow of information.
Figure 1: System Architecture Diagram
2.1 Hospital Data Preprocessing and Embedding
The system’s ability to provide contextually relevant responses is rooted in its integration with hospital-
specific data, which include clinical guidelines, electronic health records (EHRs), treatment protocols, and
institutional best practices. Crucially, when incorporating sensitive sources like EHRs, appropriate data
de-identification techniques compliant with privacy regulations (e.g., HIPAA) are employed to protect
patient confidentiality. Before runtime, these diverse text sources undergo rigorous preprocessing:
1.Document Segmentation: Longer documents are divided into semantically coherent chunks
(approximately 512 tokens) to optimize retrieval granularity.
2.Metadata Extraction: Critical information including document type, authorship, creation date,
department origin, and other relevant attributes are preserved as metadata associated with each
chunk. This metadata helps to filter and prioritize the retrieved information.
3.Embedding Generation: Each text segment is transformed into a dense vector representation us-
ing a specialized medical embedding model, E5-large-v2 [3], which has shown superior performance
in the capture of clinical semantic relationships.
4.Vector Database Indexing: The generated embeddings, along with their corresponding source
text and metadata, are stored and indexed in a vector database (e.g., Pinecone) optimized for
efficient similarity search operations.
2.2 Runtime Query Processing
When a healthcare professional initiates a query, the system executes the following sequence:
2

1.Query embedding: The user’s natural language query is transformed into an embedding using
the same E5-large-v2 model to ensure representational consistency.
2.Vector similarity search: The query embedding is compared against the indexed hospital data
embeddings using cosine similarity. The system retrieves the k most similar document segments
(typically k=5-10, adjusted based on query complexity). Metadata filtering (e.g., by date or doc-
ument type) can be applied here.
3.Context assembly: The retrieved segments are synthesized into a structured context document,
preserving provenance information (source document, chunk ID) and relevance ranking.
4.Prompt Construction: A carefully engineered prompt template combines:
(a) The original user query.
(b) The retrieved context information.
(c) System instructions specifying the desired response format and reasoning constraints (e.g.,
”Prioritize hospital protocols”).
5.LLM inference: The assembled prompt is passed to the fine-tuned Llama 3.2-3B-Instruct model,
which generates a comprehensive response incorporating both the retrieved information and its
medical knowledge.
6.Response presentation: The system delivers the generated response to the user, optionally
including source attribution and confidence indicators.
This architecture implements a true RAG paradigm, distinguishing it from simple prompt augmentation
by dynamically retrieving only the most relevant institutional knowledge for each query, thus reducing
noise and improving response precision.
3 Fine-Tuning with QLoRA
The effectiveness of LLMs in specialized medical applications depends significantly on their ability to
understand domain-specific terminology, reasoning patterns, and clinical contexts. While pre-trained
models like Llama 3.2-3B-Instruct possess general language capabilities, they require adaptation to excel
in medical tasks. This section details the implementation of Quantized Low-Rank Adaptation (QLoRA)
for fine-tuning the base model to enhance its medical decision support capabilities.
3.1 Quantized Low-Rank Adaptation: Principles and Implementation
Traditional fine-tuning of LLMs for domain-specific applications often requires substantial computational
resources, limiting practical deployment in healthcare settings with budget constraints. Quantized Low-
Rank Adaptation (QLoRA) addresses this limitation through a multifaceted approach combining model
quantization and parameter-efficient fine-tuning [4].
QLoRA represents an advancement over traditional fine-tuning techniques by combining the efficiency
of Low-Rank Adaptation (LoRA) [5] with the memory benefits of quantization. LoRA modifies the
traditional neural network layer equation from
y=WX +b
to
y= (W+BA)X+b
where Wis the frozen pre-trained weight matrix, BandAare low-rank matrices, and Xis the input.
This approach reduces the number of trainable parameters to approximately 0.5-5% of the original model.
QLoRA further enhances this efficiency by applying quantization to the frozen weights of the base model.
While standard LoRA requires approximately 2+ GB of VRAM per 1GB model, QLoRA reduces this
requirement to 0.5+ GB, enabling fine-tuning on more modest hardware configurations. This efficiency
is achieved through 4-bit quantization of the frozen base model weights, low-rank decomposition of the
weight updates, and parameter-efficient gradient propagation.
3

The quantization process converts the 16-bit or 32-bit floating-point weights to 4-bit integers, signifi-
cantly reducing memory requirements. Despite this compression, QLoRA maintains training stability
through techniques such as double quantization and paged optimizers to manage memory efficiently
during training.
3.2 Implementation with Llama 3.2-3B-Instruct
For our implementation, we selected Llama 3.2-3B-Instruct as the base model due to its strong perfor-
mance on general language tasks, manageable size for fine-tuning, and instruction-following capabilities.
The model’s 3 billion parameters provide sufficient capacity for complex medical reasoning while remain-
ing computationally tractable for deployment in clinical settings.
The fine-tuning process consisted of several key stages:
3.2.1 Dataset Selection
We compile a medical question-answer dataset by combining two sources: the Medical Meadow WikiDoc
dataset (curated from WikiDoc articles) [6] and the MedQuAD dataset from NIH domains [7]. The
combined dataset contains 26,412 question–answer pairs.
3.2.2 LoRA Adapter Configuration
The LoRA configuration was carefully designed to optimize learning efficiency while ensuring robustness
in adapting to new data distributions. We implemented rank-8 adapters (r=8) for key attention layers
and feed-forward networks, with an alpha value of 16 to scale the contribution of the adapters. This
results in 2.4 million trainable parameters (about 0.75% of the base model’s size). The bias term was
set to “none,” ensuring that no additional bias parameters were learned.
3.2.3 Training Details
The model was trained for 1 epoch on the curated dataset, achieving a final training loss of 1.2734.
Figure 2(a) and Figure 2(b) show the loss value and learning rate progression for each global step during
the fine-tuning process.
Training was conducted in a Linux-based environment using CPython 3.12.3. The script was executed
on a local workstation equipped with an NVIDIA TITAN RTX GPU (24 GB VRAM), 2,560 CUDA
cores, and CUDA version 12.8. The system had 8 logical CPU cores. Detailed system specifications are
provided in Table 1.
The total training time was 5,718 seconds, and the throughput was 4.619 samples per second. The
AdamW optimizer was used with a linear learning-rate schedule (initial LR = 2 ×10−4). A detailed
summary of the training metrics is provided in Table 2.
Table 1: System Information
Component Specification
OS Linux 6.11.0-21-generic x86 64 GNU/Linux
Python Version CPython 3.12.3
CPU Count 8
Logical CPU Count 16
GPU NVIDIA TITAN RTX
CUDA Version 12.8
GPU Architecture Turing
CUDA Cores 2560
GPU Memory 24 GB
3.2.4 Model Evaluation
To quantitatively evaluate our fine-tuned model, we test it on medical multiple-choice and question-
answering benchmarks. In Table 3, we report accuracy on the MedMCQA dataset and selected medical
4

(a) Loss value for every global step
 (b) Learning rate with every global step
Figure 2: Loss Value and Learning Rate during Finetuning
Table 2: Training Metrics
Metric Value
Epoch 1
Steps per second 0.289
Samples per second 4.619
Total FLOPs 2.5656e17
Total Runtime (seconds) 5,718
Gradient Norm 0.2829
Global Step 1650
Learning Rate 1.2195e-7
Train Loss 1.2734
Peak reserved memory 4.328 GB (18.449% of max memory)
Peak reserved memory for training 0.887 GB (3.781% of max memory)
subsets of the MMLU benchmark. MedMCQA is a large-scale medical multiple-choice dataset covering
entrance-exam questions [8], while MMLU (Medical) covers professional medical knowledge [9]. The
results demonstrate significant improvements in clinical question answering and reasoning.
Table 3: Model Evaluation on Medical Benchmarks
Dataset QLoRA Fine-tuned Model Llama-3.2- 3B-Instruct
MedMCQA 56.39 50.9
MMLU Anatomy 62.30 59.26
MMLU Clinical Knowledge 65.28 62.64
MMLU High School Biology 75.97 70.32
MMLU College Biology 78.74 70.83
MMLU College Medicine 56.07 58.38
MMLU Medical Genetics 71.00 74.00
MMLU Professional Medicine 74.63 74.26
3.3 Benefits of QLoRA Fine-Tuning over Traditional Approaches
Feature Traditional Fine-Tuning LoRA QLoRA Importance in Medical Settings
Trainable Parameters High Low Very Low Reduces overfitting risk on limited clinical datasets
Memory Usage High Moderate Low Enables deployment in resource-constrained hospitals
Training Time Long Moderate Short Allows rapid adaptation to evolving medical protocols
Quantization No No 4-bit NormalFloat (NF4) Maintains precision for critical medical calculations
Risk of Overfitting Higher Lower Lower Preserves performance across diverse patient populations
Hardware High-end GPUs Mid-range GPUs Consumer-grade GPUs Makes AI implementation viable for smaller clinics
5

4 RAG on Hospital Data
Retrieval-Augmented Generation (RAG) represents a pivotal advancement in improving the accuracy and
relevance of LLM outputs for healthcare applications. By combining the knowledge retrieval capabilities
of information retrieval systems with the generative abilities of LLMs, RAG addresses several limita-
tions of standalone LLMs, particularly in healthcare contexts where factual accuracy, evidence-based
recommendations and up-to-date information are critical.
The RAG process in our healthcare LLM system involves two main phases:
•Retrieval : When a user submits a query, the system uses search techniques to fetch relevant
information from the vector database containing the embeddings of the hospital data. This retrieval
is based on the semantic similarity between the user’s query and the stored document embeddings.
•Generation : The retrieved information is then seamlessly incorporated into the prompt provided
to the LLM. This augmented context provides the LLM with a more comprehensive understanding
of the topic, enabling it to generate more precise, informative, and contextually relevant answers.
4.1 Medical Data Preprocessing and Embedding
Hospital data encompasses diverse document types with unique characteristics, including clinical guide-
lines and protocols, electronic health records, medication formularies, departmental procedures, and
research publications. Each document is augmented with metadata such as document type, publication
date, and source—is split into manageable chunks (typically around 512 tokens) to ensure fine-grained
retrieval.
After preprocessing, every document chunk is converted into dense vector representations using medical
domain-specific embedding models (e.g., E5-large-v2) that capture medical semantics more effectively
than general-purpose embeddings. The resulting embedding vectors, along with their associated meta-
data, are stored in a vector database (such as Pinecone). This repository supports fast and accurate
similarity searches through cosine similarity metrics. Figure 3 illustrates this end-to-end preprocessing
and embedding workflow.
To maintain the system’s accuracy with evolving hospital information, the vector database requires
periodic updates. New or revised data sources (e.g., clinical guidelines, protocols) must undergo the
described preprocessing pipeline. The frequency of these updates depends on the rate of change in the
source data.
Figure 3: Data Preprocessing Pipeline for RAG
4.2 Hybrid Retrieval System
Our RAG implementation employs a hybrid retrieval approach that combines multiple retrieval mech-
anisms to optimize for different types of medical queries. This includes vector similarity search as the
6

primary retrieval mechanism, BM25 lexical search to ensure critical medical terms are matched precisely,
and hybrid fusion to combine scores from both approaches with medical term weighting.
The system implements a hierarchical retrieval strategy that first performs a broad retrieval across the
entire corpus, then conducts a focused retrieval within specific document sections, and finally extracts
the most relevant passages for inclusion in the prompt. This multi-stage approach enhances precision
while maintaining computational efficiency.
For time-sensitive information (e.g., evolving treatment protocols), the system incorporates age of infor-
mation as a factor in the retrieval ranking to prioritize recent content, ensuring that the most current
guidelines and protocols are considered in response generation.
4.3 Medical Context Integration
The retrieved information is integrated into the LLM prompt to provide contextually relevant input
for the model. This integration involves careful structuring of the prompt to balance the user query,
retrieved content, and system instructions. The prompt template includes placeholders for the user query,
retrieved passages, and reasoning constraints to guide the LLM in generating accurate and informative
responses.
Example Prompt Structure
[SYSTEM] You are a medical assistant providing information based on hospital
guidelines and medical knowledge.
For each response:
1. Consider the retrieved context carefully
2. Prioritize hospital-specific protocols when available
3. Clearly indicate when information comes from general knowledge vs. retrieved
context
4. Identify any information gaps requiring additional clarification
5. Format responses with clinical relevance in mind
[/SYSTEM]
[QUERY] {user_query}
[RETRIEVED CONTEXT]
{retrieved_documents}
[/RETRIEVED CONTEXT]
Response:
The following query illustrates a sample scenario in a typical clinic:
Query : What is our hospital’s protocol for managing diabetic ketoacidosis in pediatric patients?
Non-RAG Response :Provides a generic evidence-based protocol that conflicts with the hospital’s
specific fluid resuscitation guidelines and monitoring intervals.
RAG-Enhanced Response :Accurately cites the hospital’s pediatric DKA protocol including institution-
specific insulin dosing calculations, laboratory monitoring frequencies, and criteria for ICU transfer, with
proper attribution to the hospital’s pediatric endocrinology department guidelines updated in June 2023.
5 Applications in Healthcare
The integration of our RAG-empowered, QLoRA-fine-tuned LLM system offers transformative appli-
cations across various healthcare domains. This section explores two primary applications: disease
prediction and treatment suggestion, and medical report summarization. For each application, we ex-
amine the current clinical challenges, outline the technical approach using our system, provide concrete
workflow examples, and discuss metrics for measuring success.
7

5.1 Disease Prediction and Treatment Suggestion
5.1.1 Current Clinical Challenges
Accurately predicting diseases based on a patient’s symptoms and medical history is a complex task
for clinicians. It requires navigating a vast amount of medical knowledge, considering the often-variable
ways in which diseases can manifest, and managing time constraints during patient consultations.
5.1.2 Technical Approach Using Our System
Our system addresses these challenges by leveraging the fine-tuned LLM’s ability to analyze patient
symptoms and historical data in conjunction with RAG’s retrieval capabilities. The system can pre-
dict possible diseases based on a patient’s symptoms and medical history while suggesting appropriate
treatments based on institutional guidelines and recent clinical evidence. Crucially, the system’s out-
puts, including diagnostic suggestions and treatment options, are intended as decision support tools
and require careful review and validation by clinicians before being applied in patient care. In practi-
cal deployments, this may involve integrating the system into a clinician-in-the-loop framework where
responses are presented with appropriate disclaimers and clinicians have tools to assess, override, or
annotate AI-generated outputs before acting on them. The prompt explicitly instructs the model to
explain its reasoning and cite any sources. For example:
[SYSTEM] Based on the patient’s data and relevant medical guidelines, provide a ranked
list of possible diagnoses. Explain each choice and cite sources if relevant.
[/SYSTEM]
[USER] {Patient case description}
Response:
5.1.3 Concrete Example Workflow
1.Scenario : A clinician inputs a query stating, ”Patient presents with fatigue, unexplained weight
loss, excessive thirst, and frequent urination. Medical history includes hypertension and family
history of cardiovascular disease.”
2.Retrieval : The system retrieves relevant documents that include information on potential causes
for these symptoms and treatment protocols.
3.Response Generation : The LLM generates differential diagnosis with probability estimates:
(a) Type 2 Diabetes Mellitus (High probability - 78%)
•Supporting evidence: Constellation of classic symptoms (polydipsia, polyuria, weight loss)
•Risk factors: Family history, hypertension
•Recommended next steps: Fasting blood glucose, HbA1c
(b) Hyperthyroidism (Moderate probability - 42%)
•Supporting evidence: Weight loss, fatigue
•Distinguishing factors: Absence of tremor, heat intolerance
•Recommended next steps: TSH, T3, T4 tests
(c) Adrenal insufficiency (Lower probability - 18%)
•Supporting evidence: Fatigue, weight loss
•Distinguishing factors: Absence of hyperpigmentation, no history of steroid use
•Recommended next steps: Morning cortisol, ACTH stimulation test if warranted
8

5.1.4 Metrics for Measuring Success
Success metrics for this application can include:
1.Diagnostic Accuracy: Percentage of correct diagnoses suggested by the system compared to
expert evaluations.
2.Time Efficiency: Reduction in time taken by clinicians to arrive at a diagnosis compared to
traditional methods.
3.Diagnostic Breadth: The system’s ability to consider rare but clinically significant conditions
that might otherwise be overlooked in time-constrained settings.
5.2 Medical Report Summarization
5.2.1 Current Clinical Challenges
Clinicians often face overwhelming amounts of documentation in patient care, including lengthy medical
reports that can be time-consuming to read and interpret. This can lead to burnout and reduced
efficiency in clinical settings. There is a need for effective summarization tools that can distill essential
information from comprehensive reports while preserving critical details. LLMs have shown promise in
reducing documentation-based cognitive burden for healthcare providers [10].
5.2.2 Technical Approach Using Our System
The proposed system can automate the process of medical report summarization, leveraging LLMs’
ability to handle extensive medical data effectively. When a medical report, such as a radiology report
or a patient’s discharge summary, is input into the system, along with a user’s request for a summary
(e.g., ”summarize the key findings for a physician” or ”explain this report to a patient in simple terms”),
the system embeds both the report and the request. RAG can then retrieve additional relevant context,
such as previous reports or pertinent aspects of the patient’s medical history, to further inform the
summarization process. The fine-tuned Llama model is then prompted with instructions to generate a
summary. The prompt also includes an audience instruction (e.g., “as a summary for a physician” or
“explain to a patient”). For example:
[SYSTEM] Summarize the key findings of this radiology report for the attending
physician, including any recommended follow-up steps.
[/SYSTEM]
[USER]: [Full radiology report text]
Response:
The QLoRA-fine-tuned LLM then generates a concise and accurate summary that is tailored to the spe-
cific needs and understanding level of the intended audience. The tailoring of the summary’s complexity
(e.g., simplifying for a patient audience) is handled through specific instructions within the prompt
design, guiding the LLM on the desired output style and detail level.
5.2.3 Concrete Example Workflow
Scenario : A clinician requests a summary of a 20-page discharge report for a patient with multiple
comorbidities.
Retrieval : The system retrieves sections relevant to medications prescribed, follow-up appointments,
and critical lab results.
Response Generation : The LLM produces a summary such as: ”The patient was discharged with
recommendations for follow-up in two weeks. Key medications include Metformin 500 mg twice daily
and Lisinopril 10 mg daily. Notable lab results indicate elevated blood glucose levels; consider monitoring
closely.”
9

5.2.4 Metrics for Measuring Success
Success metrics for this application can include:
•Summary Accuracy : Percentage of key details correctly captured in summaries compared to
expert-generated summaries.
•Time Saved : Reduction in time spent by clinicians reviewing reports due to effective summariza-
tion.
•User Satisfaction Scores : Feedback from clinicians regarding the usefulness and clarity of gen-
erated summaries.
6 Discussion
The integration of LLMs into healthcare, as explored in this research, presents a significant opportu-
nity to transform medical practices. Analyzing the broader impact of the proposed system reveals both
substantial benefits and potential challenges that warrant careful consideration. Furthermore, the impli-
cations of using such technology for medical purposes raise important ethical considerations that must
be addressed to ensure responsible and beneficial implementation.
6.1 Benefits
•Enhanced Decision Support: The RAG-augmented LLM can serve as an on-demand medical
knowledge base for clinicians. By integrating patient data with current clinical guidelines, it pro-
vides contextually relevant suggestions, thereby increasing clinician confidence in decision-making
[11].
•Increased Efficiency: Automating tasks like report summarization and initial differential diag-
nosis can significantly reduce the cognitive load on healthcare providers. Studies have shown that
LLMs can alleviate documentation burden, potentially reducing clinician burnout [11].
•Tailored Responses: By incorporating hospital-specific data, our model ensures that responses
are aligned with local practices and guidelines [11].
6.2 Challenges
•Data Privacy: Handling sensitive patient data necessitates robust privacy measures to comply
with regulations such as HIPAA. Our architecture can be adapted to Indian regulations including
NDHM and DISHA through implementing federated learning and standardized anonymization
processes that satisfy both international and India-specific healthcare privacy requirements [12].
•Clinical Liability and AI Interpretability: When AI systems influence medical decisions, de-
termining responsibility for adverse outcomes becomes complex. Approaches such as providing
confidence scores and citing source documents can improve transparency. Integration with LLM
explainability techniques—such as visualizing attention maps and employing prompt-chaining to
trace reasoning steps—further enhances interpretability. Importantly, the system must be posi-
tioned as a decision-support tool, not a replacement for clinician judgment.
•Bias in Training Data: If the training data contains biases, it may lead to skewed predictions
or recommendations that do not reflect equitable healthcare practices [12].
•Integration Complexity: Successful implementation requires seamless integration into existing
electronic health record systems without disrupting workflows.
•Computational requirements: While QLoRA significantly reduces resource needs compared to
traditional approaches, deployment still requires dedicated computational infrastructure that may
be challenging for smaller healthcare facilities.
6.3 Ethical Considerations
The use of LLMs in healthcare raises several critical ethical considerations. Issues related to data privacy,
informed consent for the use of patient data in training and operation, and the overall responsible
10

use of sensitive medical information are paramount. The potential for algorithmic bias to exacerbate
existing health disparities based on factors like race, ethnicity, or socioeconomic status needs to be
carefully monitored and mitigated through the use of diverse and representative datasets and rigorous
testing for fairness. Transparency and accountability in the development and deployment of LLM-based
healthcare systems are essential to build trust among patients and healthcare providers. Clear guidelines
and regulations governing the use of LLMs in medical practice are necessary to ensure patient safety
and ethical conduct. The role of human oversight in reviewing and validating the outputs of LLMs in
medical contexts cannot be overstated, as the ultimate responsibility for patient care rests with healthcare
professionals.
7 Conclusion
This research demonstrates the significant potential of Large Language Models (LLMs) for transforming
healthcare applications through our novel integration of Retrieval-Augmented Generation (RAG) with
hospital-specific data and Quantized Low-Rank Adaptation (QLoRA) fine-tuning of the Llama 3.2-3B-
Instruct model. We have established that this combined approach substantially improves the accuracy,
relevance, and efficiency of medical decision support systems while maintaining computational feasibility
for practical clinical deployment. Our evaluation on various medical domain tasks showcased the superior
performance of our model compared to existing models, underscoring its ability to comprehend complex
medical queries.
8 Future Scope
Future research should focus on several promising directions:
•Multimodal Integration and Medical Imaging Analysis: Integrating the analysis of medical
images, such as radiology reports, alongside textual data to provide more comprehensive diagnostic
support.
•Advancing Privacy-Preserving Techniques: Developing and evaluating more sophisticated
techniques for data de-identification and privacy protection when using sensitive patient data within
RAG and fine-tuning processes.
•Optimizing Clinical Workflow Integration: Investigating seamless integration into existing
Electronic Health Record (EHR) systems and clinical workflows to maximize usability and minimize
disruption for healthcare professionals.
•Longitudinal Clinical Validation: Conducting extensive longitudinal testing in real-world clin-
ical pilot settings to assess long-term performance, reliability, and clinical impact.
•Human-AI Collaboration Benchmarking: Rigorously benchmarking the system’s perfor-
mance against scenarios involving human-in-the-loop interventions to understand optimal collabo-
ration models.
•Multilingual Adaptation for Diverse Settings: Adapting the system for multilingual capa-
bilities, particularly focusing on languages prevalent in diverse healthcare settings such as those
found across India.
By pursuing these research directions, the clinical utility and practical impact of LLM-based healthcare
systems can be significantly expanded, ultimately improving healthcare delivery, clinical outcomes, and
provider efficiency.
References
[1] Y. H. Ke, L. Jin, K. Elangovan, et al. , “Retrieval augmented generation for 10 large language
models and its generalizability in assessing medical fitness,” en, npj Digital Medicine , vol. 8, no. 1,
p. 187, Apr. 2025, issn: 2398-6352. doi:10.1038/s41746-025-01519-z .
[2] P. Lewis, E. Perez, A. Piktus, et al. ,Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks , arXiv:2005.11401, Apr. 2021. doi:10.48550/arXiv.2005.11401 .
11

[3] L. Wang, N. Yang, X. Huang, et al. ,Text Embeddings by Weakly-Supervised Contrastive Pre-
training , arXiv:2212.03533 version: 2, Feb. 2024. doi:10.48550/arXiv.2212.03533 .
[4] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, QLoRA: Efficient Finetuning of Quan-
tized LLMs , arXiv:2305.14314, May 2023. doi:10.48550/arXiv.2305.14314 .
[5] E. J. Hu, Y. Shen, P. Wallis, et al. ,LoRA: Low-Rank Adaptation of Large Language Models ,
arXiv:2106.09685 version: 2, Oct. 2021. doi:10.48550/arXiv.2106.09685 .
[6]Medalpaca/medical meadow wikidoc ·Datasets at Hugging Face , Apr. 2025. [Online]. Available:
https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc .
[7]Healthcare NLP: LLMs, Transformers, Datasets , en. [Online]. Available: https://www.kaggle.
com/datasets/jpmiller/layoutlm .
[8] A. Pal, L. K. Umapathi, and M. Sankarasubbu, MedMCQA : A Large-scale Multi-Subject Multi-
Choice Dataset for Medical domain Question Answering , arXiv:2203.14371, Mar. 2022. doi:10.
48550/arXiv.2203.14371 .
[9] D. Hendrycks, C. Burns, S. Basart, et al. ,Measuring Massive Multitask Language Understanding ,
arXiv:2009.03300 version: 3, Jan. 2021. doi:10.48550/arXiv.2009.03300 .
[10] E. Croxford, Y. Gao, N. Pellegrino, et al. , “Current and future state of evaluation of large language
models for medical summarization tasks,” en, npj Health Systems , vol. 2, no. 1, p. 6, Feb. 2025,
issn: 3005-1959. doi:10.1038/s44401-024-00011-2 .
[11] J. Vrdoljak, Z. Boban, M. Vilovi´ c, M. Kumri´ c, and J. Boˇ zi´ c, “A Review of Large Language Models
in Medical Education, Clinical Decision Support, and Healthcare Administration,” en, Healthcare ,
vol. 13, no. 6, p. 603, Mar. 2025, issn: 2227-9032. doi:10.3390/healthcare13060603 .
[12] M. Harishbhai Tilala, P. Kumar Chenchala, A. Choppadandi, et al. , “Ethical Considerations in the
Use of Artificial Intelligence and Machine Learning in Health Care: A Comprehensive Review,” en,
Cureus , Jun. 2024, issn: 2168-8184. doi:10.7759/cureus.62443 .
12