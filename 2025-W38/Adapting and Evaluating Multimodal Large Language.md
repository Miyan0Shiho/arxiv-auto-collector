# Adapting and Evaluating Multimodal Large Language Models for Adolescent Idiopathic Scoliosis Self-Management: A Divide and Conquer Framework

**Authors**: Zhaolong Wu, Pu Luo, Jason Pui Yin Cheung, Teng Zhang

**Published**: 2025-09-15 07:34:12

**PDF URL**: [http://arxiv.org/pdf/2509.11645v1](http://arxiv.org/pdf/2509.11645v1)

## Abstract
This study presents the first comprehensive evaluation of Multimodal Large
Language Models (MLLMs) for Adolescent Idiopathic Scoliosis (AIS)
self-management. We constructed a database of approximately 3,000
anteroposterior X-rays with diagnostic texts and evaluated five MLLMs through a
`Divide and Conquer' framework consisting of a visual question-answering task,
a domain knowledge assessment task, and a patient education counseling
assessment task. Our investigation revealed limitations of MLLMs' ability in
interpreting complex spinal radiographs and comprehending AIS care knowledge.
To address these, we pioneered enhancing MLLMs with spinal keypoint prompting
and compiled an AIS knowledge base for retrieval augmented generation (RAG),
respectively. Results showed varying effectiveness of visual prompting across
different architectures, while RAG substantially improved models' performances
on the knowledge assessment task. Our findings indicate current MLLMs are far
from capable in realizing personalized assistant in AIS care. The greatest
challenge lies in their abilities to obtain accurate detections of spinal
deformity locations (best accuracy: 0.55) and directions (best accuracy: 0.13).

## Full Text


<!-- PDF content starts -->

Adapting and Evaluating Multimodal Large
Language Models for Adolescent Idiopathic
Scoliosis Self-Management: A Divide and
Conquer Framework
Zhaolong Wu, Pu Luo, Jason Pui Yin Cheung, and Teng Zhang
Department of Orthopaedics and Traumatology, The University of Hong Kong, Hong
Kong SAR, China{wuzl01,luopudent}@connect.hku.hk,
{cheungjp,tgzhang}@hku.hk
Abstract.This study presents the first comprehensive evaluation of
Multimodal Large Language Models (MLLMs) for Adolescent Idiopathic
Scoliosis (AIS) self-management. We constructed a database of approxi-
mately 3,000 anteroposterior X-rays with diagnostic texts and evaluated
five MLLMs through a ‘Divide and Conquer’ framework consisting of
a visual question-answering task, a domain knowledge assessment task,
and a patient education counseling assessment task. Our investigation
revealed limitations of MLLMs’ ability in interpreting complex spinal ra-
diographs and comprehending AIS care knowledge. To address these, we
pioneered enhancing MLLMs with spinal keypoint prompting and com-
piled an AIS knowledge base for retrieval augmented generation (RAG),
respectively. Results showed varying effectiveness of visual prompting
across different architectures, while RAG substantially improved models’
performances on the knowledge assessment task. Our findings indicate
current MLLMs are far from capable in realizing personalized assistant
in AIS care. The greatest challenge lies in their abilities to obtain accu-
rate detections of spinal deformity locations (best accuracy: 0.55) and
directions (best accuracy: 0.13).
Keywords:Multimodal Large Language Models·Scoliosis·X-ray
1 Introduction
Adolescent Idiopathic Scoliosis (AIS) is the most common spinal deformity in
pediatric populations, affecting up to 4.8% of adolescents[9,5]. This condition
predominantly occurs during growth spurts between ages 11 and 14. Without
timely intervention, spinal deformities may progressively worsen, leading to seri-
ous consequences such as back pain and impaired pulmonary function, reducing
patients’ quality of life [3].
While treatments, surgical-interventions and clinical management are key in
AIS care, patient self-management plays an essential role in the recovery from
AIS, which could have paramount effects on outcomes, quality of life and long-
term well-being [7,13]. This includes exercise and physical therapy, compliancearXiv:2509.11645v1  [cs.AI]  15 Sep 2025

2 Z. Wu et al.
with treatment, emotion and mental health as well as monitoring and reporting
[19,10,17].
MLLMs such as LLaVA-Med have demonstrated powerful capabilities in
medical image analysis, detecting lesions, analyzing conditions, and providing
medical advice [16,26,21]. Although these models excel in chest radiographs
and fundus image analysis, research targeting spinal diseases, particularly AIS,
remains scarce due to the limited availability of specialized spinal data. Further-
more, it is unclear whether and how well current state-of-the-art open-source /
weight MLLMs are at realizing a personal assistant for AIS self-management,
which requires capabilities beyond medical imaging comprehension.
To address this gap, we constructed a database of approximately 3,000 an-
teroposterior X-rays with corresponding diagnostic texts. We evaluated leading
MLLMs through a comprehensive framework that breaks down the AIS self-
management requirement into three downstream tasks: a visual spinal assess-
ment task (the ability of x-ray analysis for disease progression), a domain knowl-
edge assessment task (the understanding of the disease and managements), and
a patient education counseling assessment task (the ability to do personalized
assistant given patient’s x-ray and open-ended queries).
The main contributions of this paper can be summarized as follows.
1. We conducted the first comprehensive study on using MLLMs for AIS self-
management through a divide-and-conquer framework, which is aligned well
with clinical practice, separating complex AIS care into specialized tasks.
2. For adapting MLLMs for AIS care, we integrated a spinal keypoint detection
model with MLLMs for improving their ability on analyzing spinal deformi-
ties for the image modality. To improve MMLMs in comprehending AIS
knowledge, we compiled an AIS knowledge base and implemented a knowl-
edge augumented generation approach.
3. We constructed a large-scale specialized image-text database for AIS, the
largesttodatetothebestofourknowledge,providingfoundationalresources
for future related research.
2 Related Work
Recent years have witnessed significant advances in applying MLLMs to medical
tasks. These models successfully integrate various modalities to perform complex
medical tasks, such as medical image-text generation, disease diagnosis, and
automated report generation [8,18,14,4]. However, current research exhibits a
notable data bias toward certain anatomical regions.
CurrentMLLMs,includingmedical-specificmodelslikeLLaVA-Medandgen-
eralmodelslikeQwen-VL2.5andInternVL-3[16,1,27],havedevelopedexpertise
primarily in chest radiograph analysis due to the predominance of chest X-ray
datasets [12,11,22]. This has resulted in limited capabilities for spinal disorders
like AIS, which requires precise assessment of curve patterns and specialized
knowledge of classification systems and progression risk factors.

Divide and Conquer Framework for MLLMs in AIS Self-Management 3
Fig. 1.Divide and Conquer Framework for Adapting and Evaluating MLLMs for AIS
Care.
To address this gap, we introduce a novel framework that decomposes the
complex AIS care process into distinct, manageable components. This "Divide
and Conquer" approach allows us to systematically evaluate and adapt MLLMs
for specific aspects of AIS management, from visual diagnosis to patient educa-
tion. We construct a specialized spinal database and develop targeted assessment
methodsforeachcomponent,whilealsoemployingakeypointdetectionmodelto
supply critical anatomical landmarks and investigate what missing information
leads to diagnostic failures.
3 The ‘Divide and Conquer’ Evaluation Framework
Figure 1 shows our multi-granularity clinical assessment framework for AIS anal-
ysis. We designed three comprehensive evaluation tasks to assess MLLMs’ ef-
fectiveness in analyzing spinal deformities from AP spine X-rays across multi-
ple clinical perspectives. Additionally, we integrated a spinal keypoint detection
model to investigate whether diagnostic errors stem from MLLMs’ inability to
identify critical spinal landmarks.
Visual Spinal Assessment (VSA)When diagnosing AIS in clinical prac-
tice,cliniciansprimarilyfocusonexaminingimagestoanalyzespinaldeformities,
their location, and curve patterns. We decomposed this complex visual reasoning
process into three sequential tasks of increasing granularity:
AIS Diagnosis (AD): This binary classification task requires MLLMs to gen-
erate a potential diagnosis based on spinal X-ray images and textual prompts,
determining whether AIS is present or absent.
Spinal Deformity Location Detection (SDLD): This multi-class classification
task focuses on localizing spinal deformities. Building on the diagnosis from
AD, MLLMs must identify whether the spinal curvature occurs in the thoracic,
thoracolumbar, or lumbar segments based on X-ray images and textual prompts.
This localization helps determine the type of spinal deformity curve.

4 Z. Wu et al.
Spinal Deformity Direction Detection (SDDD): This multi-class classifica-
tion task determines the direction of spinal deformity. Following AD and SDLD
tasks, MLLMs must assess whether the spinal curvature is directed leftward or
rightward, distinguishing between left-convex and right-convex curvatures.
Domain Knowledge Assessment Task (DKA)The second major com-
ponent of our framework, DKA, addresses the professional knowledge dimension
of AIS care. This multiple-choice task evaluates whether MLLMs possess suffi-
cient domain knowledge of AIS. We designed questions across six categories: ba-
sic knowledge, etiology and pathophysiology, clinical presentation and diagnosis,
assessment and monitoring, treatment options, and complications and progno-
sis. This comprehensive assessment examines the models’ understanding of AIS
from fundamental concepts to diagnosis, patient management, and treatment
strategies.
PatientEducationandCounselingAssessment(PECA)Thispatient-
oriented question-answering task evaluates MLLMs’ ability to provide accurate,
accessible information to patients with varying degrees of spinal deformity. We
developed161questionsacrossfivecategories:diseaseexplanation,treatmentop-
tions,dailylifemanagement,long-termprognosis,andfollow-upandmonitoring.
Each question was stratified by three severity levels (mild, moderate, severe) to
assess how well models adapt their responses to different clinical scenarios. This
task specifically measures the models’ capabilities in translating complex med-
ical knowledge into patient-appropriate explanations while maintaining clinical
accuracy.
4 Method
4.1 Datasets
In this study, we collected 3,683 AP spinal X-rays with corresponding radiologi-
cal reports from 3,022 patients at X Hospital between December 2019 and July
2023, with all data collection protocols approved by the Institutional Review
Board (IRB). The dataset was randomly partitioned into training, validation,
and testing sets following an 8:1:1 ratio, resulting in 2,946 samples for training,
368 samples for validation, and 369 samples for testing. To address the inher-
ent class imbalance common in medical datasets, we implemented a stratified
sampling approach that maintained consistent distribution of scoliosis severity
across all partitions, ensuring that the proportion of each scoliosis category (nor-
mal, mild, moderate, and severe) remained constant across all sets. To minimise
input image variance, all X-rays were standardized by cropping to a uniform size
of 896×448 pixels, ensuring that the entire spinal column was captured within
each image.
4.2 Visual Prompting Strategies
We established MLLMs zero-shot predictions as baselines and investigated how
performance is affected by visual prompts provided by spine keypoint detection

Divide and Conquer Framework for MLLMs in AIS Self-Management 5
models. As shown in Figure 1, we designed three visual prompting strategies:
Curved Spine Midline (CSM), Vertebral Connection Line (VCL), and Segmented
Vertebrae Marks (SVM). These visual prompts were designed to provide models
with critical information about spinal curvature and structure. For SDLD and
SDDDtasks,wedifferentiatedthoracic,thoracolumbar,andlumbarregionsusing
distinct colors to provide anatomical localization information.
4.3 Retrieval-Augmented Generation (RAG) Approach
To enhance model performance on knowledge-intensive tasks, we implemented a
specialized RAG framework utilizing an AIS-specific knowledge database. This
database was constructed by integrating authoritative sources including clinical
practice guidelines, research publications from PubMed, and patient education
resources from organizations such as the Scoliosis Research Society (SRS) [15,
24,23,2,6]. To optimize information retrieval, we employed Gemini to generate
structured knowledge graphs that capture key relationships between AIS con-
cepts, treatments, and outcomes [25].
4.4 DKA and PECA Data Population
Our evaluation datasets were designed to assess different aspects of AIS under-
standing. The DKA dataset consists of multiple-choice questions targeting pro-
fessional medical knowledge, while the PECA dataset simulates patient-centered
scenarios requiring both clinical accuracy and appropriate communication. To
ensurecomprehensivecoverage,wedevelopedquestionsspanningdiagnosis,treat-
ment options, management strategies, and daily living accommodations for pa-
tientswithvaryingdegreesofspinaldeformity.Arigorousqualitycontrolprocess
involving two junior doctors and verification by a senior physician was imple-
mented to ensure clinical relevance, accuracy, and appropriate difficulty levels
across all questions.
4.5 Evaluation
For the VSA task, we addressed dataset class imbalance by employing a compre-
hensive evaluation approach combining F1 score, AUC, and accuracy metrics.
This multi-metric strategy ensures balanced assessment of model performance
across both majority and minority classes. For the DKA task, we used accuracy
as our primary performance metric, as it directly measures the models’ profi-
ciency in selecting correct answers within the multiple-choice format. In evalu-
ating the PECA task, we implemented a structured human evaluation protocol
wherein three junior physicians independently assessed model responses using a
five-dimensional framework on a 5-point Likert scale. The detailed assessment
criteria, outlined in Table 1, were specifically designed to evaluate both clini-
cal accuracy and communication effectiveness, providing comprehensive insight
into models’ capabilities in addressing patient concerns across varying degrees
of spinal deformity.

6 Z. Wu et al.
5 Experiments
5.1 Experimental Setup
All experiments were conducted using 4 NVIDIA GeForce RTX 3090 GPUs.
Table 1 presents the models used for comparison. We primarily utilized open-
source general-domain models, selecting five multimodal large language models
fromfourdifferentcompanies,withparametersizesrangingfrom4.2Bto14B.To
ensure experimental reproducibility and consistency, we standardized all MLLM
configurations with a temperature parameter of 0, bfloat16 quantization, and
disabled flash attention. For the PECA Task, we set the maximum response
length to 300 tokens. These settings minimized model generation randomness
and ensured result reliability. For keypoint detection, we employed SpineHR-
Net+, a model specifically trained on spine data based on HRNet and UNet
architectures [20].
Table 1.Multimodal Large Language Models Description.
Model Vision Encoder Backbone LLM Connector Release dates
Qwen2.5-VL-7B Redesigned ViT Qwen2.5-7B MLP 2025.01
InternVL3-8B InternViT-300M-448px-V2_5 Qwen2.5-7B MLP 2024.04
InternVL3-14B InternViT-300M-448px-V2_5 Qwen2.5-14B MLP 2025.04
Llama 3.2-Vision CLIP ViT-H/14 Llama 3.1-8B MLP 2024.07
Phi-3-Vision CLIP ViT-L/14 Phi-3 Mini Projection 2024.05
5.2 Results and Discussion
Table 2.Visual Spinal Assessment Results. Thor. = Thoracic, TL = Thoracolum-
bar, Lum. = Lumbar, OA = Overall accuracy. Baseline shows results without visual
prompts. "Color" indicates region-specific color encoding in visual prompts.
Model MethodTask 1: AIS Task 2: SDLD (OA-Acc, Region-F1,AUC) Task 3: SDDD(OA-Acc, Region-Acc)
F1 AUCNo Color With Color No Color With Color
OA Thor./TL/Lum. OA Thor./TL/Lum. OA Thor./TL/Lum. OA Thor./TL/Lum.
Qwen2.5-VL-7BBaseline 0.83 0.74 0.15 0.19,0.52/0.22,0.52/0.51,0.58 - - 0.09 0.37/0.45/0.43 - -
CSM 0.93 0.79 0.33 0.68,0.64/0.25,0.57/0.61,0.62 0.31 0.79,0.69/0.17,0.47/0.71,0.57 0.09 0.49/0.51/0.36 0.13 0.58/0.53/0.30
VCL 0.96 0.71 0.43 0.81,0.67/0.15,0.49/0.71,0.58 0.44 0.81,0.77/0.23,0.55/0.80,0.64 0.08 0.58/0.66/0.24 0.08 0.64/0.64/0.22
SVM 0.92 0.52 0.35 0.81,0.55/0.19,0.51/0.75,0.48 0.40 0.78,0.57/0.11,0.47/0.77,0.54 0.03 0.52/0.67/0.17 0.12 0.51/0.69/0.32
InternVL3-8BBaseline 0.94 0.50 0.10 0.38,0.59/0.24,0.54/0.20,0.50 - - 0.03 0.42/0.22/0.32 - -
CSM 0.94 0.50 0.07 0.82,0.56/0.23,0.51/0.02,0.50 0.06 0.80,0.50/0.23,0.50/0.00,0.50 0.01 0.53/0.07/0.36 0.00 0.49/0.05/0.36
VCL 0.94 0.50 0.07 0.82,0.56/0.23,0.51/0.02,0.50 0.05 0.81,0.53/0.23,0.50/0.00,0.50 0.01 0.53/0.07/0.36 0.00 0.51/0.05/0.36
SVM 0.94 0.50 0.07 0.83,0.62/0.23,0.50/0.02,0.50 0.06 0.80,0.50/0.23,0.50/0.00,0.50 0.01 0.56/0.05/0.36 0.00 0.49/0.05/0.36
InternVL3-14BBaseline 0.60 0.57 0.26 0.57,0.56/0.00,0.50/0.48,0.53 - - 0.10 0.46/0.87/0.30 - -
CSM 0.95 0.65 0.17 0.82,0.55/0.00,0.50/0.02,0.50 0.28 0.79,0.77/0.00,0.50/0.43,0.54 0.11 0.52/0.87/0.36 0.16 0.61/0.87/0.34
VCL 0.96 0.83 0.21 0.84,0.63/0.00,0.50/0.02,0.50 0.32 0.83,0.80/0.04,0.51/0.49,0.53 0.14 0.57/0.87/0.36 0.21 0.65/0.87/0.35
SVM 0.92 0.84 0.22 0.83,0.67/0.00,0.50/0.25,0.52 0.50 0.82,0.74/0.00,0.50/0.79,0.66 0.16 0.60/0.87/0.37 0.38 0.62/0.87/0.57
Llama 3.2-VisioBaseline 0.94 0.50 0.05 0.81,0.53/0.24,0.55/0.80,0.58 - - 0.02 0.50/0.17/0.17 - -
CSM 0.94 0.50 0.03 0.80,0.50/0.23,0.52/0.74,0.54 0.01 0.81,0.52/0.23,0.50/0.79,0.51 0.01 0.49/0.09/0.17 0.00 0.50/0.05/0.10
VCL 0.80 0.77 0.10 0.75,0.67/0.23,0.54/0.70,0.63 0.10 0.74,0.66/0.23,0.54/0.73,0.64 0.09 0.57/0.38/0.27 0.09 0.56/0.38/0.27
SVM 0.94 0.50 0.06 0.80,0.50/0.23,0.50/0.34,0.44 0.01 0.80,0.50/0.23,0.50/0.78,0.50 0.00 0.49/0.05/0.26 0.00 0.49/0.05/0.09
Phi-3-VisionBaseline 0.84 0.57 0.11 0.00,0.50/0.00,0.50/0.00,0.50 - - 0.11 0.33/0.87/0.36 - -
CSM 0.94 0.50 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.33/0.87/0.36 0.11 0.33/0.87/0.36
VCL 0.67 0.74 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.33/0.87/0.36 0.11 0.33/0.87/0.36
SVM 0.94 0.60 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.00,0.50/0.00,0.50/0.00,0.50 0.11 0.33/0.87/0.36 0.11 0.33/0.87/0.36

Divide and Conquer Framework for MLLMs in AIS Self-Management 7
Impact of Different Visual Prompts on AIS Diagnosis.Table 2 shows
significant variations in the performance of AIS diagnosis between models with
different visual prompts. Structured visual cues generally improved diagnostic
accuracy, with VCL achieving high F1 scores for Qwen2.5 (0.96) and InternVL3-
14B (0.96). CSM boosted InternVL3-14B’s F1 score from 0.60 to 0.95. Notably,
InternVL3-8B and most Llama-3.2 configurations showed AUC=0.50, indicating
they function as constant positive predictors rather than discriminative classi-
fiers. Only VCL enabled meaningful discrimination in Llama-3.2 (AUC=0.77),
while Phi-3.5 showed improved discrimination with VCL (AUC=0.74) and SVM
(0.60). These findings demonstrate that visual prompts’ effectiveness varies by
model architecture, with certain prompting strategies uniquely enabling discrim-
inative capabilities absent in baseline conditions.
Impact of Different Visual Prompts on SDLD.Qwen2.5’s accuracy im-
proved significantly from a baseline of 0.15 to 0.43 with VCL prompting, while
InternVL3-14B initially declined from 0.26 to 0.17 (CSM) and 0.22 (SVM). Color
encoding produced mixed overall effects but dramatically enhanced InternVL3-
14B with SVM (0.22→0.50). Regional performance was significantly impacted
by color enhancement: for thoracic detection, Qwen2.5 maintained strong F1
scores (CSM: 0.68→0.79, VCL: 0.81→0.81, SVM: 0.81→0.78), while InternVL3-
14B showed impressive gains (CSM: 0.52→0.79, SVM: 0.63→0.82). Thoracolum-
bar detection remained challenging with modest improvements for Qwen2.5 (F1
scores 0.55-0.67), while lumbar region detection showed no significant improve-
ment across models with color enhancement. Results indicate that despite in-
terventions successfully improving some models, others like InternVL3-8B, Phi-
3.5, and certain Llama-3.2 configurations remained unimproved (OA 0.10-0.11,
F1=0, AUC=0.50), likely due to insufficient domain knowledge in these models.
This highlights the importance of foundational model capabilities.
Impact of Different Visual Prompts on SDDD.The best overall perfor-
mance without color comes from InternVL3-14B+SVM (0.16), while with color
enhancement, this same configuration dramatically improves to 0.38. Qwen2.5
shows moderate performance, with its best configuration being VCL without
color for regional detection (0.58/0.66/0.24) but lower overall accuracy (0.08),
suggesting it can identify individual region directions but struggles to inte-
grate these into correct overall bending states. Color enhancement generally im-
proves Qwen2.5’s performance, particularly with CSM prompting (0.10→0.13).
In stark contrast, several models demonstrate consistently poor performance
regardless of prompting or color enhancement: InternVL3-8B achieves near-zero
overallaccuracywithcolor(0.0000acrossallpromptingstrategies)andverypoor
thoracolumbar detection (0.04-0.06); Llama3.2 with SVM prompting similarly
achieves 0.00 overall accuracy with both color versions; and multiple configura-
tions show minimal response to different prompting strategies. Regional analysis
reveals an important pattern: the seemingly high thoracolumbar accuracy (0.87)
for InternVL3-14B and Phi-3.5 is misleading, as Task 2 results indicate these

8 Z. Wu et al.
models are consistently outputting negative results rather than actually detect-
ing deformities.
Enhancement Models Through RAG.DKA and PECA results consistently
demonstrate significant improvements through RAG implementation across all
models. In DKA, InternVL 2.5-14B achieved the highest accuracy (0.97 with
RAG), while Phi 3.5-Vision showed the largest improvement (+0.20). Similarly,
for PECA, RAG substantially enhanced Medical Accuracy (+1.06 to +1.09)
and Safety (+0.94 to +1.02) across all models, though Communication Clarity
saw more modest gains (+0.49 to +0.68). This suggests that while RAG effec-
tively addresses knowledge limitations in specialized domains like AIS, extensive
retrieved content may occasionally impact narrative coherence. Notably, perfor-
mance gaps between models narrowed with RAG implementation, with smaller
models showing proportionally greater improvements. These findings confirm
that retrieval augmentation effectively compensates for limited parameters in
specialized medical applications, enabling smaller models to approach the per-
formance of larger counterparts.
Table 3.MLLMsPerformanceComparisononDKAandPECATasksWithandWith-
out RAG. "w/o RAG" = Without RAG, "w/ RAG" = With RAG, "Imp" = Improve-
ment.
ModelDKA (Acc) PECA (Acc)
w/o RAG w/ RAG ImpMedical Accuracy Response Completeness Communication Clarity Response Personalization Safety
w/o RAG w/ RAG Imp w/o RAG w/ RAG Imp w/o RAG w/ RAG Imp w/o RAG w/ RAG Imp w/o RAG w/ RAG Imp
InternVL 2.5-14B 0.82 0.97 +0.16 2.96 3.95 +0.98 3.20 3.93 +0.73 3.38 3.88 +0.50 3.13 3.84 +0.71 2.87 3.88 +1.02
InternVL 2.5-8B 0.77 0.91 +0.14 2.88 3.84 +0.96 3.11 3.95 +0.84 3.30 3.86 +0.56 3.08 3.75 +0.67 2.87 3.84 +0.97
Llama 3.2-Vision 0.77 0.94 +0.17 2.84 3.90 +1.06 3.15 3.89 +0.74 3.30 3.80 +0.50 3.20 3.79 +0.59 2.84 3.78 +0.94
Phi 3.5-Vision 0.70 0.90 +0.20 2.80 3.88 +1.08 2.89 3.88 +0.99 3.20 3.88 +0.68 2.88 3.67 +0.79 2.84 3.82 +0.98
Qwen 2.5VL-7B 0.78 0.94 +0.16 2.86 3.95 +1.09 3.09 3.91 +0.82 3.27 3.76 +0.49 2.89 3.72 +0.83 2.88 3.84 +0.96
6 Conclusion
This research introduces a novel Divide and Conquer framework that breaks
down complex AIS analysis into distinct, evaluable stages. Findings demonstrate
thatcurrentMLLMsremaininsufficientforimplementingautomatedAISpatient
self-management systems, despite showing promise in specialized tasks. While
anatomical guidance improved quantification performance for models with ade-
quate baseline capabilities, RAG significantly enhanced models’ capabilities in
specialized AIS knowledge domains, particularly overcoming knowledge limita-
tions in patient education and domain knowledge tasks. This systematic evalua-
tion approach provides a roadmap for targeted improvements, suggesting that as
MLLMs advance in both foundational capabilities and specialized medical un-
derstanding,theywillincreasinglysupportclinicalpracticewithoutyetreplacing
human expertise in AIS care.
Acknowledgments.This research was supported by the Health and Medical
Research Fund (HMRF) [Grant No. 19200911 and 21223141] and the National

Divide and Conquer Framework for MLLMs in AIS Self-Management 9
Natural Science Foundation of China (NSFC) Young Scientists Fund [Grant No.
82303957]. We sincerely thank all funding agencies for their generous support.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.
References
1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang,
S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923
(2025)
2. Berdishevsky, H., Lebel, V.A., Bettany-Saltikov, J., Rigo, M., Lebel, A., Hennes,
A., Romano, M., Białek, M., M’hango, A., Betts, T., et al.: Physiotherapy scoliosis-
specific exercises–a comprehensive review of seven major schools. Scoliosis and
spinal disorders11, 1–52 (2016)
3. Cheung, J.P.Y., Cheung, P.W.H., Samartzis, D., Luk, K.D.K.: Curve progres-
sion in adolescent idiopathic scoliosis does not match skeletal growth. Clinical
Orthopaedics and Related Research®476(2), 429–436 (2018)
4. Davis, A., Souza, R., Lim, J.H.: Knowledge-augmented language models interpret-
ing structured chest x-ray findings. arXiv preprint arXiv:2505.01711 (2025)
5. De Sèze, M., Cugy, E.: Pathogenesis of idiopathic scoliosis: a review. Annals of
physical and rehabilitation medicine55(2), 128–138 (2012)
6. Dimitrijević, V., Rašković, B., Popović, M., Viduka, D., Nikolić, S., Drid, P.,
Obradović, B.: Treatment of idiopathic scoliosis with conservative methods based
on exercises: a systematic review and meta-analysis. Frontiers in Sports and Active
Living6, 1492241 (2024)
7. Dufvenberg, M., Diarbakerli, E., Charalampidis, A., Öberg, B., Tropp, H., Asp-
berg Ahl, A., Möller, H., Gerdhem, P., Abbott, A.: Six-month results on treatment
adherence, physical activity, spinal appearance, spinal deformity, and quality of life
in an ongoing randomised trial on conservative treatment for adolescent idiopathic
scoliosis (contrais). Journal of Clinical Medicine10(21), 4967 (2021)
8. Fan,Z.,Liang,C.,Wu,C.,Zhang,Y.,Wang,Y.,Xie,W.:Chestx-reasoner:Advanc-
ing radiology foundation models with reasoning through step-by-step verification.
arXiv preprint arXiv:2504.20930 (2025)
9. Fong, D.Y., Cheung, K.M., Wong, Y.W., Wan, Y.Y., Lee, C.F., Lam, T.P., Cheng,
J.C., Ng, B.K., Luk, K.D.: A population-based cohort study of 394,401 children
followed for 10 years exhibits sustained effectiveness of scoliosis screening. The
Spine Journal15(5), 825–833 (2015)
10. Holt, C.J., McKay, C.D., Truong, L.K., Le, C.Y., Gross, D.P., Whittaker, J.L.:
Sticking to it: a scoping review of adherence to exercise therapy interventions in
children and adolescents with musculoskeletal conditions. journal of orthopaedic &
sports physical therapy50(9), 503–515 (2020)
11. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H.,
Haghgoo, B., Ball, R., Shpanskaya, K., et al.: Chexpert: A large chest radiograph
dataset with uncertainty labels and expert comparison. In: Proceedings of the
AAAI conference on artificial intelligence. vol. 33, pp. 590–597 (2019)
12. Johnson, A.E., Pollard, T.J., Berkowitz, S.J., Greenbaum, N.R., Lungren, M.P.,
Deng, C.y., Mark, R.G., Horng, S.: Mimic-cxr, a de-identified publicly available
database of chest radiographs with free-text reports. Scientific data6(1), 317
(2019)

10 Z. Wu et al.
13. Karol, L.A., Virostek, D., Felton, K., Wheeler, L.: Effect of compliance counseling
on brace use and success in patients with adolescent idiopathic scoliosis. JBJS
98(1), 9–14 (2016)
14. Kim, Y., Wu, J., Abdulle, Y., Gao, Y., Wu, H.: Enhancing human-computer inter-
action in chest x-ray analysis using vision and language model with eye gaze pat-
terns. In: International Conference on Medical Image Computing and Computer-
Assisted Intervention. pp. 184–194. Springer (2024)
15. Kuznia, A.L., Hernandez, A.K., Lee, L.U.: Adolescent idiopathic scoliosis: common
questions and answers. American family physician101(1), 19–23 (2020)
16. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T.,
Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for
biomedicine in one day. Advances in Neural Information Processing Systems36,
28541–28564 (2023)
17. Li, J., Chan, E.A., Li, M., Lam, Y.P., Wong, A.Y., Cheung, J.P.Y., Li, Y.: “am i
different?” coping and mental health among teenagers with adolescent idiopathic
scoliosis: A qualitative study. Journal of pediatric nursing75, e135–e141 (2024)
18. Li, Q., Cui, Z., Bae, S., Xu, J., Yuan, R., Zhang, Y., Feng, R., Shen, Q., Zhang,
X., He, J., et al.: Aor: Anatomical ontology-guided reasoning for medical large
multimodal model in chest x-ray interpretation. arXiv preprint arXiv:2505.02830
(2025)
19. Marchese, R.: World-wide variation in Schroth therapists’ clinical reasoning and
exercise prescription for adolescents with idiopathic scoliosis. Ph.D. thesis, Mac-
quarie University (2023)
20. Meng, N., Cheung, J.P., Wong, K.Y.K., Dokos, S., Li, S., Choy, R.W., To, S., Li,
R.J., Zhang, T.: An artificial intelligence powered platform for auto-analyses of
spine alignment irrespective of image quality with prospective validation. EClini-
calMedicine43(2022)
21. Moor, M., Huang, Q., Wu, S., Yasunaga, M., Dalmia, Y., Leskovec, J., Zakka, C.,
Reis, E.P., Rajpurkar, P.: Med-flamingo: a multimodal medical few-shot learner.
In: Machine Learning for Health (ML4H). pp. 353–367. PMLR (2023)
22. Nguyen, H.Q., Lam, K., Le, L.T., Pham, H.H., Tran, D.Q., Nguyen, D.B., Le,
D.D., Pham, C.M., Tong, H.T., Dinh, D.H., et al.: Vindr-cxr: An open dataset of
chest x-rays with radiologist’s annotations. Scientific Data9(1), 429 (2022)
23. Roye,B.D.,Simhon,M.E.,Matsumoto,H.,Bakarania,P.,Berdishevsky,H.,Dolan,
L.A., Grimes, K., Grivas, T.B., Hresko, M.T., Karol, L.A., et al.: Establishing con-
sensus on the best practice guidelines for the use of bracing in adolescent idiopathic
scoliosis. Spine deformity8, 597–604 (2020)
24. Seifert, J., Thielemann, F., Bernstein, P.: Adolescent idiopathic scoliosis: guideline
for practical application. Der Orthopäde45, 509–517 (2016)
25. Team, G., Georgiev, P., Lei, V.I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vin-
cent,D.,Pan,Z.,Wang,S.,etal.:Gemini1.5:Unlockingmultimodalunderstanding
across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024)
26. Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.C., Carroll,
A., Lau, C., Tanno, R., Ktena, I., et al.: Towards generalist biomedical ai. Nejm
Ai1(3), AIoa2300138 (2024)
27. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W.,
Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for
open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025)