# MMIR-TCM: Memory-Integrated Multimodal Inference and Retrieval for TCM Clinical Decision Support

**Authors**: Lihui Luo, Joongwon Chae, Ziyan Chen, Yang Liu, Siyi Cheng, Weihan Gao, Zelin Zeng, Xiaoming Yin, Samaneh Beheshti Kashi, Dongmei Yu, Lian Zhang, Jing Sui, Zeming Liang, Jiansong Ji, Peter E. Lobie, Peiwu Qin

**Published**: 2026-07-02 07:30:19

**PDF URL**: [https://arxiv.org/pdf/2607.01814v1](https://arxiv.org/pdf/2607.01814v1)

## Abstract
Traditional Chinese Medicine (TCM) diagnosis, particularly through tongue inspection, faces persistent challenges in subjectivity and reproducibility. The application of multimodal artificial intelligence to TCM clinical tasks, such as syndrome differentiation and prescription generation, is significantly hampered by the semantic gap between visual tongue features and textual reasoning, as well as the lack of large-scale, standardized datasets. To address these challenges, we introduce MMIR-TCM, a novel framework that emulates the diagnostic process of TCM experts by integrating multimodal large language model(MLLM) with memory-augmented segmentation and retrieval-augmented generation (RAG). Employing a three-stage architecture, MMIR-TCM integrates a training-free Memory-SAM module for robust tongue extraction, a fine-tuned Qwen3-VL model for structured tongue diagnosis generation, and a Qwen3-based RAG component for evidence-grounded clinical decision support generation. The framework was developed and validated using MedTCM, a new large-scale multimodal dataset that we introduce specifically for advanced TCM research. To properly evaluate our framework's clinical accuracy, which existing metrics fail to capture, we also developed TDEU, a domain-specific evaluation metric incorporating semantic understanding and diagnostic importance. Our comprehensive experiments demonstrate that MMIR-TCM significantly outperforms leading models, including GPT-4o and Gemini 2.5 Flash.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS AND JOURNALS TEMPLA TE 1
MMIR-TCM: Memory-Integrated Multimodal
Inference and Retrieval for TCM Clinical Decision
Support
Lihui Luo, Joongwon Chae, Ziyan Chen, Y ang Liu, Siyi Cheng, W eihan Gao, Zelin Zeng, Xiaoming Yin,
Samaneh Beheshti Kashi, Dongmei Y u, Lian Zhang, Jing Sui, Zeming Liang, Jiansong Ji, Peter E. Lobie,
and Peiwu Qin
Abstract— T raditional Chinese Medicine (TCM) diagnosis,
particularly through tongue inspection, faces persistent chal-
lenges in subjectivity and reproducibility . The application of
multimodal artificial intelligence to TCM clinical tasks, such
as syndrome differentiation and prescription generation, is
significantly hampered by the semantic gap between visual
tongue features and textual reasoning, as well as the lack
of large-scale, standardized datasets. T o address these chal-
lenges, we introduce MMIR-TCM, a novel framework that
emulates the diagnostic process of TCM experts by integrat-
ing multimodal large language model(MLLM) with memory-
augmented segmentation and retrieval-augmented generation
(RAG). Employing a three-stage architecture, MMIR-TCM in-
tegrates a training-free Memory-SAM module for robust tongue
extraction, a fine-tuned Qwen3-VL model for structured tongue
diagnosis generation, and a Qwen3-based RAG component for
evidence-grounded clinical decision support generation. The
framework was developed and validated using MedTCM, a
new large-scale multimodal dataset that we introduce specif-
ically for advanced TCM research. T o properly evaluate our
framework’s clinical accuracy , which existing metrics fail to
capture, we also developed TDEU, a domain-specific evaluation
metric incorporating semantic understanding and diagnostic
importance. Our comprehensive experiments demonstrate that
MMIR-TCM significantly outperforms leading models, includ-
Lihui Luo, Joongwon Chae, Ziyan Chen, Y ang Liu, Siyi
Cheng, W eihan Gao and Peter E. Lobie are with the In-
stitute of Biopharmaceutics and Health Engineering, T singhua
Shenzhen International Graduate School, Shenzhen, 518000,
China, also with the Chinese Medicine Guangdong Labora-
tory , Zhuhai, 519000, China (e-mail: luolh23@mails.tsinghua.edu.cn;
chi-zy24@mails.tsinghua.edu.cn; chenziya23@mails.tsinghua.edu.cn;
lyang22@mails.tsinghua.edu.cn; chengsiy25@mails.tsinghua.edu.cn;
gwh25@mails.tsinghua.edu.cn; pelobie@sz.tsinghua.edu.cn).
Jing Sui is with the Beijing Normal University , Beijing, 100000,
China (e-mail:Xi0218@hotmail.com).
Zeming Liang and Peiwu Qin are with the Chinese Medicine
Guangdong Laboratory , Zhuhai, 519000, China (e-mail:
305823980@qq.com, pwqin1979@gmail.com).
Lian Zhang is with the First Hospital of Hebei Medical University ,
Shijiazhuang, 050000, China (e-mail: pwqin@sz.tsinghua.edu.cn).
Samaneh Beheshti Kashi, Dongmei Y u and Jiansong Ji
are with the Lishui Hospital of Zhejiang University , Lishui,
323000, China (e-mail:lolisky1@163.com, qaydm1979@gmail.com,
llzzmm0218@163.com).
Zelin Zeng and Xiaoming Yin are with XiaoMing TCM Hos-
pital, Shenzhen, 518000 China. (e-mail: 13074519056@163.com;
15864642233@163.com).
The code is available at https://github.com/jw-chae/MMIR-
TCM .
Lihui Luo and Joongwon Chae contributed equally to this work.
Corresponding author: Peiwu Qin.ing GPT-4o and Gemini 2.5 Flash.
Index T erms— Artificial Intelligence, Large Language Mod-
els, T raditional Chinese Medicine, T ongue Image Analysis,
Clinical Decision Support Systems.
I. INTRODUCTION
MUL TIMODAL Artificial Intelligence (AI) have re-
cently achieved radiologist-level performance on
several W estern medical imaging tasks, such as chest X
ray interpretation and abnormality detection [1]. How-
ever, T raditional Chinese Medicine (TCM), a diagnostic
paradigm built on holistic pattern recognition and cen-
turies of clinical practice, remains relatively underexplored
by modern AI. TCM diagnosis tightly couples visual
observations (tongue, facial complexion), auditory , tactile
cues (auscultation and pulse), and textual knowledge
(syndrome typologies, herbal properties, and classical case
precedents). This inherent multimodality and the need for
evidence grounded clinical decisions make TCM a natural,
yet challenging, target for multimodal model integration.
Within the F our Diagnostic Methods of TCM (inspec-
tion, auscultation and olfaction, inquiry , and palpation),
tongue inspection holds a special diagnostic role. Mor-
phological features of the tongue body (e.g., color, shape,
moisture), and features of the coating (e.g., color, thick-
ness, texture, distribution) are systematically associated
with core pathophysiological patterns [2]. F or example,
a pale tongue with a thin white coating may indicate
qi deficiency and internal cold, suggesting the use of
tonifying formulas such as Si Junzi T ang, whereas a red
tongue with a yellow greasy coating implies damp–heat,
prompting clearing and draining strategies like Longdan
Xiegan T ang. Clinicians synthesize tongue visual cues,
patient history , and canonical case precedents to choose
personalized herbal prescriptions [3].
Despite its clinical significance, tongue diagnosis suf-
fers from several challenges that limit its reproducibility
and computational integration. Practitioner subjectivity
introduces considerable inter- and intra-rater variability
[4]. In addition, variations in image acquisition conditions,
such as illumination (e.g., changes in color temperature),arXiv:2607.01814v1  [cs.AI]  2 Jul 2026

2 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
background clutter (e.g., clothing, facial features), and
framing inconsistencies (e.g., tongue positioning, camera
angle) can substantially degrade the reliability of auto-
mated analysis pipelines. Moreover, existing approaches to
tongue image analysis [5] typically address segmentation
or attribute classification in isolation, without bridging
the gap to clinical decision-making or grounding outputs
in verifiable knowledge sources.
The emergence of multimodal large language models
(MLLMs) offers a promising new approach. Leveraging
strong image understanding capabilities, MLLMs have
shown competitive performance in a variety of medical
imaging tasks, including disease classification and visual
report generation. F or instance, Zhao et al. [6] fine
tuned MLLMs by integrating visual data with structured
medical knowledge, enabling clinical reasoning within
diagnostic workflows. However, the application of large
language models (LLMs) to TCM tasks is constrained by
the scarcity of large scale, publicly available multimodal
datasets. Existing resources tend to be fragmented, pro-
viding isolated elements such as herbal prescriptions or
tongue images rather than complete clinical records that
pair images with diagnoses and treatment strategies [7].
Motivated by these considerations, we propose MMIR-
TCM, an end-to-end multimodal framework that emulates
the staged reasoning workflow of experienced TCM prac-
titioners: memory-integrated tongue region extraction,
structured tongue diagnosis generation, and retrieval-
augmented-generation (RAG) [8] for prescription gener-
ation that grounds clinical decisions in precedent and evi-
dence. T o support rigorous evaluation and reproducibility ,
we assemble MedTCM, a multi center multimodal corpus
reflecting realistic clinical diversity , and introduce TDEU,
a domain aware metric designed to evaluate tongue diag-
nostic outputs with attention to both semantic fidelity and
clinical importance. T ogether, these components enable a
system that not only predicts tongue diagnosis accurately
but also links those predictions to verifiable and evidence-
anchored clinical recommendations. By grounding gen-
eration in retrieved clinical precedents and documented
formula rationales, the system improves factuality , inter-
pretability , and clinician auditability , which are essential
for safe clinical deployment.
In summary , the main contributions of this work are as
follows:
1) W e propose MMIR-TCM, a memory-integrated mul-
timodal pipeline for TCM clinical decision sup-
port that integrates training free tongue extrac-
tion, attribute-level tongue diagnosis generation,
and retrieval-augmented prescription generation to
produce evidence-grounded clinical decision support.
2) W e construct MedTCM, a large scale multimodal
TCM dataset with tongue images, diagnostic re-
ports, and clinical prescriptions collected from mul-
tiple hospitals to ensure diversity and clinical rele-
vance. The dataset will be made publicly available
and maintained.
3) W e develop TDEU, a domain aware evaluation met-ric for tongue diagnosis that incorporates semantic
similarity and clinical importance, addressing the
limitations of traditional text matching metrics.
II. R ELATED WORK
A. TCM-Specific Large Language Models
Recent efforts have adapted large language models
(LLMs) to TCM through domain-specific pretraining and
alignment. TCMChat [9] demonstrated that continued
pretraining on TCM corpora followed by supervised fine-
tuning improves knowledge recall and question-answering
over general-purpose LLMs. Complementary benchmark
efforts have emerged to standardize evaluation: TCM-
Bench [10] curated 5,473 licensing exam questions with
the TCMScore metric to assess TCM semantic consis-
tency beyond surface accuracy , revealing that current
models still have considerable room for improvement.
TCMD [11] provided large-scale annotated QA with ro-
bustness analysis, exposing inconsistencies under random
perturbations. BianCang [12] implemented a two-stage
learning pipeline (knowledge injection followed by targeted
alignment) using the Chinese Pharmacopoeia and hospital
records, demonstrating superior performance in syndrome
differentiation tasks. These efforts underscore the value
of TCM-specific alignment, though excessive fine-tuning
may degrade general language capabilities [10].
F or clinical prescription recommendation, recent work
has moved beyond pure LLM generation toward
knowledge-augmented approaches. TCM-FTP [13] fine-
tuned LLMs for herbal prescription prediction and intro-
duced normalized mean squared error (NMSE) metrics
for dosage prediction, directly addressing the quantifica-
tion challenge in formula generation. TCM-KLLaMA [14]
combined knowledge graphs encoding herb-symptom-
contraindication relationships with LLM generation, using
structured knowledge constraints to suppress interaction
risks and improve groundedness.
B. Multimodal T ongue Diagnosis
T ongue diagnosis occupies a central role in TCM’s F our
Examinations (inspection, auscultation, inquiry , palpa-
tion), as properties of the tongue body (color, shape) and
coating (color, thickness, texture, distribution) directly
inform syndrome differentiation. Early computational ap-
proaches focused on segmentation and attribute classifica-
tion using traditional computer vision or deep learning [5],
but recent work has shifted toward multimodal fusion
that jointly reasons over images and textual clinical
information. T ongueNet [15] introduced image-text fusion
with consistency and complementarity constraints in the
representation space, enabling simultaneous multi-label
prediction of pathological attributes and anatomical lo-
cations. Cross-modal attention architectures [16] have ex-
tended this to organ-level pathology classification, bridg-
ing the gap between visual features and high-level clinical

AUTHOR et al.: TITLE 3
Fig. 1 : Overall architecture of MMIR-TCM. A Memory-SAM–based T ongue Extractor precedes a Qwen3-VL tongue
diagnosis generator, followed by a Qwen3-based RAG prescription generator, emulating expert TCM clinical workflow.
reasoning. However, robustness to acquisition variations—
illumination shifts, color bias, and partial occlusion—
remains underexplored in these frameworks.
Dataset standardization has emerged as a critical
enabler for reproducible evaluation. TCM-T ongue [17]
provides 6,719 high-quality images with 20 pathological
annotations under controlled acquisition conditions, while
TCMEval-SDT [18] offers a benchmark for syndrome
diagnosis capability . These resources facilitate systematic
comparison of multimodal models, though comprehensive
robustness analysis across illumination, compression, and
occlusion stressors is still limited.
a) Impact of Segmentation on Downstream Diagnosis. :
Accurate tongue–background separation is widely re-
ported to improve downstream performance in classical
vision pipelines. F or example, Xian et al. [19] showed
that using segmentation as an auxiliary loss can enhance
both quality assessment and subsequent diagnostic stages.
F or fine-grained structures, segment-based approaches [20]
enable stable crack detection, and improved T ransUNet
variants [21] achieve precise coating segmentation that
stabilizes feature extraction. Moreover, integrating quality
control with segmentation can filter low-quality shots
prior to analysis, thereby reducing variance and improving
reproducibility . Recent zero-shot methods such as T ongue-
SAM [22] leverage Segment Anything to generalize acrossdiverse acquisition conditions without retraining.
C. Retrieval-Augmented Generation for Clinical Decision
Support
Retrieval-augmented generation (RAG) has become the
standard approach for grounding medical LLM outputs
in external knowledge sources. General RAG frame-
works [23] retrieve relevant documents to augment gen-
eration context, improving factual accuracy and reducing
hallucination. In medical domains, specialized architec-
tures have emerged: ClinicalRAG [8] implements multi-
agent workflows that dynamically integrate structured
codes and unstructured clinical notes, while LINS [24]
enhances evidence-based medicine with citation consis-
tency metrics. GraphRAG [25] extends retrieval to graph-
structured knowledge, enabling multi-hop reasoning and
query-focused summarization over large private corpora.
Recent evaluations [26] confirm that RAG workflows
outperform pure LLMs in clinical appropriateness and
urgency assessment tasks.
In TCM, knowledge sources bifurcate into empirical
case knowledge (modern EMRs) and normative princi-
ple knowledge (classical texts such as Huangdi Neijing,
Shanghan Lun, Bencao Gangmu) [27]. These sources
exhibit representational mismatches—modern vernacu-
lar versus classical Chinese, symptom-syndrome-formula

4 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
taxonomies—that complicate unified indexing. PreGen-
erator [6] addressed this through a retrieval→generation
hybrid that first recalls similar prescriptions and cases
before combining them with generation rules, achiev-
ing consistent improvements over pure neural models.
OpenTCM [25] constructed a TCM knowledge graph
from 68 classical obstetrics texts (3.73M characters) with
48k entities and 152k relations, integrating GraphRAG
for symptom-based retrieval and demonstrating expert-
evaluated superiority in diagnostic QA. These approaches
highlight the complementarity of case-based specificity
and principle-based normativity , motivating our hybrid
indexing strategy . F urthermore, integration with clinical
practice guidelines [28] enables injection of normative
constraints (contraindications, interaction rules) into the
reasoning chain, which is critical for safety-critical pre-
scription generation in TCM.
D. Evaluation, Robustness, and Safety in Medical AI
T raditional metrics such as BLEU and ROUGE capture
surface-level similarity but fail to reflect clinical semantic
equivalence or groundedness. TCM-specific benchmarks
have introduced domain-aware evaluation: TCMBench’s
TCMScore [10] incorporates semantic consistency , while
MTCMB [29] decomposes evaluation across 12 subtasks
including knowledge QA, clinical reasoning, prescription
generation, and safety compliance, revealing that current
LLMs perform adequately on knowledge recall but struggle
with clinical reasoning and safety adherence. Multimodal
evaluation has expanded to include tongue and herb
images, audio, and video through TCM-Ladder [30],
measuring cross-modal grounding capabilities beyond text
accuracy .
F or RAG systems, evaluation must extend beyond
answer quality to retrieval and augmentation stages. RAG
frameworks such as RAGAS [31] propose metrics for faith-
fulness (evidence-output alignment), answer relevance,
and context relevance, with faithfulness showing strong
correlation with human judgments. GroUSE [32] bench-
marks judge LLMs themselves, exposing failure modes
such as partial citation and context overfitting. Medical
applications demand additional rigor: MedCite [33] tar-
gets ”verifiable text” in medical QA through multi-pass
retrieval and citation, elevating citation consistency to a
first-class output criterion.
Robustness, explainability , and safety form the opera-
tional pillars for clinical deployment. Robustness encom-
passes resilience to input perturbations (noise, illumina-
tion shift, occlusion), distribution shift (out-of-distribution
detection [34]), and reasoning chain vulnerabilities [35].
Uncertainty quantification techniques—temperature scal-
ing, deep ensembles, conformal prediction [36], [37]—
enable selective prediction with rejection thresholds for
high-risk cases. Explainability in medical contexts requires
attribute-level grounding: attention maps, segmentation
masks, and counterfactual interventions [38] to validate
feature-decision pathways. Safety enforcement in prescrip-
tion generation necessitates multi-layered defenses: (1)pretraining safety alignment with contraindication rules,
(2) constrained decoding with normative scoring [39],
(3) post-hoc validators using rule-based and knowledge-
graph checking, and (4) human-in-the-loop gates for high-
risk decisions [40]. Systematic reviews emphasize that
accuracy alone is insuﬀicient; trustworthy medical AI must
co-optimize reliability , interpretability , and accountabil-
ity [41].
III. M ETHODS
W e present MMIR-TCM, an end-to-end multimodal
framework that mirrors expert TCM workflows by stan-
dardizing visual inputs, generating structured tongue de-
scriptions, and producing evidence-grounded prescriptions
via retrieval-augmented reasoning. As illustrated in Fig. 1,
the system processes tongue images and clinical metadata
jointly to yield accurate and interpretable outputs. Given
a patient’s tongue image Itongue and clinical metadata
M patient = (Q, H, P ), where Q,H, and P denote chief
complaint, history of present illness, and pulse diagnosis,
respectively , the pipeline consists of three tightly coupled
stages:
1) T ongue Extractor. A training-free Memory-SAM
segmenter retrieves visual exemplars and converts
them into structured foreground/background point
prompts for SAM2 [42], yielding a precise tongue
mask and a background-removed region-of-interest
(ROI) IROI
tongue .
2) T ongue Diagnosis Generator. A fine-tuned Qwen3-
VL model analyzes IROI
tongue and outputs a one-
sentence, attribute-level report D that consistently
enumerates tongue body color/shape and coating
color/thickness/texture with anatomical locations.
3) Prescription Generator. A Qwen3-based generator
fuses D with M patient , retrieves similar cases from
a de-identified EMR knowledge base via vector
search, and synthesizes syndrome differentiation and
an herbal prescription with concise evidence-based
reasoning.
This modular design enables independent optimization
of each stage while preserving end-to-end traceability from
raw images to clinically actionable, evidence-grounded
outputs.
A. T raining-F ree T ongue Extractor via Memory-SAM
Supervised segmentation models require extensive pixel-
level annotations and exhibit poor generalization to out-
of-distribution acquisition conditions—varying lighting,
background clutter, camera specifications, and patient po-
sitioning. Bounding-box-based pipelines that feed detected
regions to SAM suffer from detector failures under these
conditions, often leaking into surrounding facial regions.
T o address these limitations without incurring annotation
costs or retraining overhead, we used Memory-SAM [43],
a training-free architecture that converts visual exemplar
retrieval into explicit point prompts for SAM2.

AUTHOR et al.: TITLE 5
Tongue Diagnosis Generator: 
You are an expert in Chinese medicine tongue diagnosis. Please analyze the tongue photos provided based on 
Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example: 
(The tip of the tongue is red, with a thin and greasy fur, The tongue is pale -red, covered by a thin and white 
coating, Swollen tongue with a thin, white coating etc.})
Prescription Generator: 
You are a senior traditional Chinese medicine expert with rich clinical experience. Please  provide professional 
TCM diagnosis, dialectical analysis, prescription recommendations, and diagnostic reasons based on the 
patient's symptom information.
Patient information: 
Tongue diagnosis: {shezhen}       Pulse diagnosis: {maizhen}
Chief complaint: {zhusu}         Present medical history: {xianbingshi}
Please output strictly in the following format and do not output any other content: 
Diagnosis: [Disease Name]
Dialectical: [Dialectical result]
Prescription: [Medication 1] [Dosage] [Medication 2] [Dosage] [Medication 3] [Dosage] [Usage and 
Dosage] [Instructions]
Reason for diagnosis: [Detailed diagnosis analysis and reasons, including symptom analysis, tongue and 
pulse analysis, and pathogenesis explanation]
Requirement:
1. The above four lines must be output, each starting with "Diagnosis:", "Dialectics:", "Prescription:", 
"Reason for Diagnosis:“
2. The diagnosis is concise and clear, consisting of 2 -8 words
3. Dialectically accurate, 4-8 words
4. Separate each medication in the prescription with "" (three spaces) to avoid duplicate medication
5. Must include the section on usage and dosage
6. The diagnostic reasons should be detailed, including symptom analysis, tongue and pulse analysis, and 
pathogenesis explanation
7. Do not output thought processes, explanations, or any other contentCOT Prompt for MMIR -TCM
(a)
(b)
Fig. 2 : The detailed prompt template used in MMIR-
TCM.
a) Dense F eature Extraction. :Given a tongue image
Itongue , we extract patch embeddings using a pretrained
DINOv3 (ViT-L/16) encoder [44]. Each patch embedding
isℓ2-normalized to form a dense feature map Eq∈
RH×W×D, where H, W are spatial dimensions and D=
1024 is the embedding dimension. W e also compute a
global image descriptor ¯eq∈RDby average-pooling patch
embeddings.
The memory bank M stores N reference exemplars,
where each entry m= (Im, Mm, Em,¯em) consists of
the reference image, ground-truth binary mask, patch
embeddings, and global descriptor.
b) Memory Retrieval. :W e retrieve the most similar
exemplar via cosine similarity over global descriptors:
m∗= arg max
m∈Msim(¯eq,¯em), (1)
where sim (·,·)denotes cosine similarity . This top-1 re-
trieval is implemented using F AISS [45] with a nearest-
neighbor search.
c) Mask-Constrained Dense Matching. :W e resize Mm∗
to the feature grid resolution and define foreground and
background index sets JfgandJbg based on mask values.
F or each query patch i, we compute pairwise cosine
similarities with all reference patches:
Sij=E(i)
q·E(j)
m∗
∥E(i)
q∥∥E(j)
m∗∥. (2)
W e then identify the best-matching foreground and
background patches:
j∗
fg(i) = arg max
j∈J fgSij, (3)
j∗
bg(i) = arg max
j∈J bgSij. (4)B. Attribute-Level T ongue Diagnosis Generator with
Qwen3-VL
F ree-text tongue descriptions from practitioners vary
widely in granularity , terminology , and structure, creat-
ing a semantic gap that obstructs downstream retrieval
and reasoning. Inconsistent linguistic representations—
ranging from telegraphic notes (”red tip, greasy coat”) to
verbose narratives—reduce the effectiveness of similarity-
based retrieval and complicate evidence integration. T o
standardize the perception-reasoning interface, we fine-
tune Qwen3-VL to generate consistent, attribute-level one-
sentence tongue reports that enumerate key diagnostic
features in a fixed format.
1) Model and T raining Objective :W e employ Qwen3-VL-
30B [46], a vision-language model with a vision encoder
(SigLIP), projection layer, and autoregressive language
decoder. Given a tongue image Itongue and target report
T, we minimize the cross-entropy loss:
L(θ) =−∑
(I,T)∈DlogPθ(T|I), (5)
whereD is our training dataset of 2,805 image-report pairs
from the MedTCM corpus, split 90/10 for training and
validation.
T o reduce computational overhead, we apply Low-Rank
Adaptation (LoRA) [47] to the vision projection layer
and language decoder’s query , key , value, and output
projection matrices, as well as feed-forward network layers.
W e use LoRA rank r∈ {64} with α= 2r. T raining
employs Adam W optimizer with learning rate 5×10−5,
weight decay 0.01, and fp16 mixed precision over 3–20
epochs.
2) Structured Output F ormat :W e enforce a standardized
output schema through system prompts (See Fig. 2(a)).
Example output: ”T ongue body pale red, shape normal,
coating white, thin, slightly greasy , distributed evenly with
mild redness at tip. ”
This format ensures that every report covers the same
diagnostic dimensions, enabling reliable parsing and struc-
tured input for downstream RAG retrieval.
3) Implementation Details :W e use Qwen3-VL-30B with
LoRA applied to both vision and language components
(rank 64, 10 epochs, dropout = 0); the optimizer uses a
learning rate of 5×10−5. Input images are resized to 768×
768 pixels. During inference, we use nucleus sampling with
p= 0.9and temperature T= 0.7to balance consistency
and natural language fluency .
C. RAG-Enhanced Prescription Generator with Qwen3
The diagnostic reasoning module is built upon a RAG
framework grounded in Qwen3, designed to emulate the
comprehensive and evidence-driven reasoning process of
experienced TCM physicians. As shown in Fig. 3, this
module integrates multimodal diagnostic information, in-
cluding the generated tongue diagnosis Ttongue and the pa-
tient’s structured metadata M patient , to retrieve clinically
similar cases and synthesize context-aware prescriptions
through LLM inference.

6 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
Fig. 3 : RAG-Enhanced Prescription Generator. Retrieved similar cases and Liujing theory are fused with current
patient metadata and tongue diagnosis to produce evidence-based clinical decision support.
1) Knowledge Base Construction and Indexing :The foun-
dation of our RAG system is a comprehensive Memory
Base, consisting of two key components: the LiuJing
Theory and the Clinical Case Bank. The case bank is built
from 124,593 anonymized patient records collected from
four major TCM hospitals. Each record contains complete
clinical information—including chief complaints, medical
history , tongue and pulse diagnoses, syndrome differentia-
tion, and herbal prescriptions—providing rich contextual
knowledge for retrieval. In parallel, the LiuJing Theory
Memory incorporates canonical TCM theoretical texts,
particularly those based on the LiuJing (Six-Meridian)
differentiation, enabling the model to perform higher-level
reasoning grounded in traditional diagnostic principles.
T o enable eﬀicient similarity-based retrieval, we em-
ploy Langchain, a data framework for LLM applications,
combined with F AISS for vector indexing. The indexing
process involves:
1) Document Processing: Each record in memory base
is structured as a document containing multiple
fields (chief complaint, history , pulse, tongue, syn-
drome, prescription).
2) Semantic Embedding: Documents are encoded into
768-dimensional dense vectors using a domain-
adapted Chinese medical language model.
3) Hierarchical Indexing: F AISS constructs hierarchi-
cal navigable small world (HNSW) graphs [48] for
approximate nearest neighbor search, enabling sub-
linear query time complexity .
This indexing strategy ensures both retrieval accuracyand computational eﬀiciency , critical for real-time clinical
decision support applications.
2) Query Construction and Retrieval :Given the tongue
diagnosis output Ttongue from the visual analysis module
and patient metadata M patient = (Q, H, P ), we construct
a unified query vector:
q= Embed (Ttongue⊕M patient ) (6)
where⊕denotes information concatenation, and Embed (·)
represents the semantic embedding function mapping text
to a dense vector space.
The retrieval module then identifies the kmost clinically
relevant cases using cosine similarity:
R= Retrieve (q, k) ={C1, C2, . . . , C k} (7)
where each retrieved case Cicontains comprehensive
diagnostic and treatment information:
Ci=(
T(i)
tongue , Q(i), H(i), P(i), S(i), Rx(i))
(8)
Here, S(i)represents the syndrome differentiation and
Rx(i)denotes the prescribed herbal formula for case i.
3) Context-A ware Diagnostic Generation :The final stage
of the prescription generator synthesizes the retrieved
clinical cases with the current patient’s information to
generate comprehensive diagnostic outputs. F ollowing the
prompt shown in Fig. 2(b), the Qwen3-based reasoning
engine processes an evidence-augmented context formed
by combining the retrieved cases R with the patient’s
metadata M patient and tongue diagnosis Ttongue :
Output = Qwen3 (M patient , T tongue , R) (9)

AUTHOR et al.: TITLE 7
Algorithm 1 TDEU Score Computation
Require: Predicted report ˆT and reference report T∗
Ensure: Overall similarity score sand per-category simi-
larity dictionary
1: function Compute T ongue Eval( ˆT,T∗)
2: cfg← LoadConfig(”token_config.json”)
3: ts_pred← cfg.extract( ˆT)
4: ts_lab← cfg.extract( T∗)
5: cats_pred← ts_pred.to_categories(cfg.category_map)
6: cats_lab← ts_lab.to_categories(cfg.category_map)
7: total← 0.0, total_w← 0.0
8:C←{ TONGUE ,COA T ,LOCA TION ,OTHER}▷
Define category set
9: forc∈C do
10: sim← PairwiseSimilarity(cats_pred[ c],
cats_lab[ c], cfg)
11: weight← cfg.weights[ c]
12: total← total + sim× weight
13: total_w← total_w + weight
14: details[ c]← sim
15: end for
16: overall← total / total_w
17: return (overall, details)
18: end function
The system generates four key clinical outputs: (1)
syndrome differentiation identifying the underlying TCM
patterns, (2) primary diagnosis in biomedical terms, (3)
herbal prescription with specific herb combinations and
dosages, and (4) clinical reasoning explaining the diag-
nostic logic and treatment rationale. This comprehensive
output format ensures both clinical actionability and
interpretability , facilitating physician review and patient
understanding.
D. TDEU: TCM Diagnosis Evaluation with Semantic
Understanding
Standard text similarity metrics such as BLEU and
ROUGE operate on surface-form n-gram overlap, failing
to capture clinically meaningful equivalences in TCM
tongue diagnosis. F or example, white coating and (greasy
coating) are lexically dissimilar but represent distinct
pathological states (dampness-phlegm vs. interior cold),
while pale red and slightly red are near-synonyms. Sim-
ilarly , omitting critical attributes such as tongue body
color carries greater clinical consequence than missing
location descriptors. T o address these limitations, we in-
troduce TDEU (TCM Diagnosis Evaluation with Semantic
Understanding), a domain-aware metric that incorporates
synonym recognition, cross-attribute semantic similarity ,
and clinical importance weighting. The overall procedure is
summarized in Algorithm 1, which consists of the following
components:
1) T oken Decomposition and Categorization :Given a pre-
dicted tongue report ˆT and reference report T∗, we
parse both into atomic attribute tokens using predefinedpattern-matching rules. Each token is assigned to one of
four categories:
•TONGUE: tongue body color and shape
•COA T: coating color, thickness, and texture
•LOCA TION: anatomical regions
•OTHER: supplementary descriptors
LetPcandLcdenote the predicted and reference token
lists for category c∈{ TONGUE ,COA T ,LOCA TION}.
2) Semantic Similarity and Optimal Matching :F or each
category c, we compute pairwise semantic similarities be-
tween tokens in PcandLc. Each token follows the format
PREFIX_V ALUE (e.g., COLOR_red, THICK_thin).
Similarity S(p, l)is defined through a hierarchical match-
ing scheme:
S(p, l) =

1.0 if exact match
Synπ(p)(ν(p), ν(l)) ifπ(p) =π(l)
Cross (p, l) ifπ(p)̸=π(l)
0.0 otherwise(10)
where Sync(v1, v2) denotes the expert-curated syn-
onym similarity score for values v1, v2 within cate-
gory c. These scores range from 0.6 to 0.9 based on
clinical equivalence (e.g., SynCOLOR (pale,pale_white ) =
0.9, SynCOLOR (red,dark_red ) = 0 .6). The cross-
category similarity Cross (p, l) captures semantic cor-
relations between attributes from different categories
(e.g., Cross (SHAPE_swollen ,THICK_thick ) = 0 .3), with
scores ranging from 0.2 to 0.9 based on TCM diagnostic
principles.
W e apply the Hungarian algorithm to find the optimal
one-to-one matching Mc⊆Pc×Lcthat maximizes total
similarity:
Mc= arg max
M∑
(p,l)∈MS(p, l). (11)
The category-level similarity score incorporates a partial
matching bonus to reward correctly identified attributes
even when the prediction is incomplete:
T okenListSim c= max ( BaseScore c,BonusScore c) (12)
where
BaseScore c=∑
(p,l)∈McS(p, l)
max(|Pc|,|Lc|)(13)
and
BonusScore c= 0.3×|{(p, l)∈Mc:S(p, l)>0}|
max(|Pc|,|Lc|). (14)
This ensures that predictions with partial correctness
receive a minimum score proportional to the number of
matched tokens, preventing over-penalization of incom-
plete but clinically relevant predictions.

8 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
3) Clinical Importance W eighting :W e evaluate four cat-
egories: TONGUE, COA T, LOCA TION, and OTHER
(auxiliary descriptors). Through consultation with senior
TCM practitioners, we let wTONGUE =2.0,wCOA T =1.5,
wLOCA TION =1.0, and wOTHER =0.5. Define the category
setC={TONGUE ,COAT ,LOCATION ,OTHER}. The
overall score is a weighted average:
TDEU =∑
c∈Cwc·TokenListSim c∑
c∈Cwc. (15)
In T able II, we report the sub-scores for TONGUE,
COA T, and LOCA TION to improve readability . The
Overall score follows the four-category weighted aggrega-
tion scheme described above, which also includes OTHER.
In our implementation, the OTHER category represents
auxiliary descriptors such as teeth-mark impressions, sur-
face cracks, and moisture-related adjectives that are not
covered by the other three categories.
IV. R ESULTS
A. Dataset Construction: MedTCM
T o address the critical scarcity of large-scale multimodal
TCM datasets, we constructed MedTCM through collab-
oration with four leading TCM hospitals in China. The
study protocol received ethical approval from the Insti-
tutional Review Boards of all participating institutions
(ethical approval number: 2025(S) No. 029). All patient
data were rigorously de-identified to protect privacy , with
all personally identifiable information (PII) systematically
removed.
1) Multi-Center Data Collection Strategy :The data col-
lection framework, illustrated in Fig. 4, was structured
around a primary institution leading data standardization,
complemented by three partner hospitals contributing di-
verse clinical cases. This multi-center strategy was crucial
for ensuring data diversity and mitigating biases inherent
in single-center collections [49], [50].
The collaborative effort yielded an extensive corpus of
124,593 anonymized patient records spanning the period
from April 2021 to June 2025. Each record contains
comprehensive clinical information including:
•Chief complaint and present illness history
•T ongue diagnosis (visual and textual description)
•Pulse diagnosis findings
•TCM syndrome differentiation
•Herbal prescription with specific formulations
F rom this corpus, we curated a high-resolution multi-
modal subset of 2,805 tongue images, each meticulously
paired with its corresponding expert-annotated diagnostic
description and complete medical record. T o the best of
our knowledge, MedTCM represents the first large-scale,
open-source dataset integrating visual tongue imagery
with comprehensive clinical records and expert annota-
tions.T ABLE I : Detailed Characteristics and Statistical Distri-
bution of the MedTCM Dataset.
Category Characteristic Metric / Sub-
categoryV alue /
Distribution
Overall
StatisticsCases Count Records 124,593 cases
T ongue Subset Image–Diagnosis 2,805 pairs
Time period – 04.2021 – 06.2025
Patient
DemographicsAge (years)Mean± SD 37.6± 15.2
Range 1–96
GenderMale 37369 (30%)
F emale 87,224 (70%)
Prescriptions Categoriessyndrome diff. 855 types
T ongue Diagno-
sis7294 types
Fig. 4 : MedTCM multi-center data construction and
curation pipeline. High-quality tongue images are paired
with de-identified clinical records and expert annotations.
2) Dataset Statistics and Characteristics :As shown in
T able I, the dataset exhibits rich demographic diver-
sity and comprehensive syndrome coverage. The tongue
image subset (2,805 pairs) was selected to ensure high
image quality . Expert TCM physicians verified all image-
diagnosis pairs for consistency and clinical accuracy . The
dataset will be continuously updated and released as open
source to facilitate future research in AI-assisted TCM
diagnosis.
B. T ongue Extractor Performance
Rationale and prior validation. W e adopt Memory-SAM
as a training-free inference pipeline that avoids task-
specific model fine-tuning. It requires only a small gallery
of annotated exemplars (binary masks) to convert exem-
plar correspondences into explicit foreground/background
point prompts for SAM2, rather than large-scale pixel-
level labeling of the entire dataset. This “few-annotation”
setup drastically reduces annotation cost while retaining
robustness under acquisition shifts. [43]
Operational yield on corpus. Across the full dataset of
2,805 images, Memory-SAM successfully generated clini-
cally usable masks for 2,780 cases as shown in Fig. 5(99.1%
success rate). The remaining 25 cases were manually
corrected before they were entered into the training data,

AUTHOR et al.: TITLE 9
T ABLE II : Combined results: baselines, ViTCM-LLM ablations, and MMIR-TCM (ours). Missing values are shown
as “–” .
Model / Setting Params LoRA r EpochsBLEU / ROUGE TDEU
BLEU-4 R-1 R-2 R-L Overall T ongue Coat Location
Baselines (zero-shot)
Grok–2–Vision–1212 – – – 26.40 22.12 22.08 26.43 0.338 0.262 0.404 0.367
LLaMA4–scout 109B 109B – – 25.68 22.45 22.43 25.70 0.336 0.254 0.412 0.356
Gemini–2.5–Flash – – – 24.43 21.37 21.33 24.46 0.353 0.271 0.421 0.372
ViTCM–LLM (Qwen2.5–VL 32B)
E1 Zero–shot 32B – – 24.08 – – – 0.3538 – – –
E2 Language–only 32B 16 3 26.74 – – – 0.2698 – – –
E3 Vision+Projector 32B 16 3 35.82 – – – 0.3610 – – –
E4 F ull ( r=16 , 3ep) 32B 16 3 37.94 – – – 0.3648 – – –
E4 F ull ( r=64 , 3ep) 32B 64 3 39.57 – – – 0.3915 – – –
E4 F ull ( r=64 , 10ep) 32B 64 10 43.57 – – – 0.5858 – – –
E4 F ull ( r=64 , 20ep) 32B 64 20 44.07 – – – 0.6150 – – –
MMIR–TCM (ours; Qwen3–VL 30B + LoRA)
Original Qwen3–VL 30B (untuned) 30B – – 33.98 28.42 28.23 34.05 0.289 0.187 0.288 0.141
Raw train + Raw 30B 64 10 83.20 76.24 76.06 83.62 0.601 0.442 0.673 0.611
Raw train + Mask 30B 64 10 83.11 75.96 75.57 83.34 0.612 0.472 0.683 0.590
Mask train + Raw 30B 64 10 83.22 76.06 75.69 83.45 0.617 0.478 0.693 0.592
Mask train + Mask 30B 64 10 83.57 77.30 76.77 84.30 0.627 0.473 0.693 0.649
Raw train + Raw 30B 64 3 81.72 74.42 73.88 82.20 0.580 0.385 0.682 0.607
Mask train + Mask 30B 64 3 83.09 76.50 76.28 83.73 0.611 0.439 0.698 0.632
Fig. 5 : Background removal and ROI standardization with
Memory-SAM. F or diverse in-the-wild inputs, extracted
tongue masks produce clean foreground crops that reduce
illumination and background variance, stabilizing the
interface for report and prescription generation.
mainly due to low contrast between the tongue and
background or under-exposed images.
C. T ongue Diagnosis Generator Performance
Setup. W e evaluate a tongue report generator built on
Qwen3–VL–30B under six settings that vary the input
type for training and evaluation (raw vs. segmented) and
the number of fine–tuning epochs (3, 10). All in–house
runs use the same LoRA configuration (rank r=64 ,α=128 ,
dropout = 0 ) and learning rate 5×10−5.
The Original Qwen3–VL 30B model without domain
fine–tuning serves as an internal baseline. F or external
reference, we additionally report three public MLLMs
(Grok–2–Vision–1212, LLaMA4–scout 109B, Gemini–2.5–
Flash) evaluated zero–shot. Metrics include BLEU–4,ROUGE, and domain–aware clinical fidelity TDEU (Over-
all, T ongue, Coat, Location), summarized in T able II.
Results. Using segmented inputs consistently at both
training and evaluation (Mask →Mask, 10 epochs) yields
the best text metrics (BLEU–4 83.57, ROUGE–1/2/L
= 77.30/76.77/84.30). T raining and evaluating on raw
images (Raw→Raw, 10 epochs) is competitive (BLEU–
4 83.20; ROUGE–L 83.62). On the domain–aware TDEU,
Mask→Mask (10 epochs) attains the highest Overall
0.627, with sub–category scores indicating that Coat and
Location are comparatively easier to match than T ongue.
Zero–shot public MLLMs remain far below the domain–
adapted models (BLEU–4 mid–20s; TDEU Overall ≈
0.34–0.35). F or a detailed analysis of how input standard-
ization with a tongue extractor affects performance across
training/evaluation stages, see the ablation in Sec. IV-E.1 .
D. Prescription Generator Performance
Setup. F or our quantitative evaluation, we constructed
a held-out test set by randomly sampling 1,151 cases from
MedTCM. T o ensure comprehensive coverage of diagnostic
scenarios, this sample was stratified to include representa-
tive cases from all major syndrome differentiation types.
F or each case, we extracted the patient’s visual tongue di-
agnosis results along with other clinical metadata M patient
to serve as the input for the models. All experiments were
conducted with a fixed random seed of 42. The publicly
released model Qwen/Qwen3-8B, serving as the LLM
backbone, was deployed on 8 NVIDIA R TX 3090 GPUs
(24 GB). Our implementation integrates a sophisticated
retrieval and indexing method.The retrieval component
utilizes the LangChain framework with F AISS as the
underlying vector database for eﬀicient similarity search.
Case documents are indexed using HNSW graphs within

10 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
F AISS, optimized for fast approximate nearest neighbor
search. Generation uses nucleus sampling with p=0.9,
temperature=0.7, and top-k=50 to balance creativity and
coherence.
Results. A qualitative examination of Fig. 6(a) reveals
that the performance polygon of MMIR-TCM consistently
encloses those of all baseline models, underscoring its
robust superiority across the entire evaluation spectrum.
T o further quantify this superiority , Fig. 6(b) presents
the averaged results across five evaluation metrics and
three representative clinical tasks. MMIR-TCM achieves
the highest overall performance, reaching the 5 scores
of 0.142, 0.343, 0.220, 0.369, and 0.288, respectively—
corresponding to relative gains of +209.6%, +45.2%,
+84.6%, +34.1%, and +41.3% over the ablated variant
without tongue diagnosis (MMIR-TCM w/o tongue).
These improvements highlight the critical contribution of
tongue diagnosis in enhancing the semantic fidelity and
diagnostic precision of prescription generation. Fig. 6(c)
further illustrates the distribution of aggregated per-
formance scores, where MMIR-TCM exhibits a higher
median and a more compact, right-skewed density , in-
dicating stronger stability and generalization in clinical
prescription reasoning.
E. Ablation Study
1) Ablation on T ongue Extractor : In deep learning-
based tongue diagnosis and medical imaging, it is well-
established that mask segmentation (background removal)
or ROI standardization enhances model performance and
stability . This is generally attributed to two factors:
(i) suppressing non-medical sources of variance such as
background clutter, illumination, and skin tone, and (ii)
guiding downstream modules to focus on the semantically
meaningful region (in this case, the tongue). However,
while these benefits are well-documented for traditional
computer vision tasks like classification, segmentation,
and detection, or in multimodal contexts where text
is secondary , their effects have not been systematically
validated in scenarios centered on language generation,
such as generating tongue diagnosis reports with MLLMs.
This ablation study is designed to address precisely
this gap. Specifically , we evaluate the extent to which
input standardization via mask segmentation improves
text quality and clinical fidelity in MLLM-based tongue
report generation. F urthermore, we assess whether the
magnitude of this improvement is suﬀicient to justify its
adoption from a practical and operational standpoint.
Evaluation relied on BLEU-4 and ROUGE for text
quality , and on the clinically aware fidelity met-
ric TDEU (Overall, T ongue, Coat, Location) (T a-
ble II). Using segmented inputs consistently for both
training and evaluation (Mask Mask) achieved the
best performance at 10 epochs (BLEU-4 83.57; ROUGE-
1/2/L 77.30/76.77/84.30; TDEU Overall 0.627). The fully
raw pipeline (Raw Raw) was competitive (BLEU-
4 83.20; ROUGE-L 83.62; TDEU Overall 0.601), but theT ABLE III : LoRA fine-tuning scenarios. ”LoRA” indicates
that new low-rank adapters are trained while the original
weights remain frozen.
Scenario Vision T ower Projector Language LM
Baseline (E1) F rozen F rozen F rozen
Language-only (E2) F rozen F rozen LoRA
Vision+Projector (E3) LoRA LoRA F rozen
F ull Model (E4) LoRA LoRA LoRA
Mask Mask approach remained consistently superior in
both lexical overlap and clinical semantics.
Crossed conditions, such as Raw Mask and Mask Raw,
where segmentation was applied at only one stage,
produced nearly identical BLEU-4 scores at 10 epochs
(≈83.22). However, they yielded modestly higher TDEU
Overall scores (0.612 and 0.617, respectively) than the
Raw Raw pipeline, indicating that standardizing inputs
at either the training or evaluation stage improves se-
mantic stability . Even with only 3 epochs of training,
the Mask Mask setup (BLEU-4 83.09; TDEU 0.611) out-
performed the 3-epoch Raw Raw model (BLEU-4 81.72;
TDEU 0.580), demonstrating that background suppression
and shape normalization yield immediate performance
gains.
In conclusion, mask segmentation provides a mod-
est benefit by reducing variance from background and
illumination, slightly improving the stability of report
generation. While its impact in a MLLM-based tongue
diagnosis model is limited and it cannot be considered an
essential component, it is retained in the system to ensure
consistent input formatting for downstream modules. In
other words, it serves as a helpful preprocessing step
that supports the pipeline’s robustness. Achieving higher
diagnostic accuracy likely requires strategies beyond input
standardization, such as expanding dataset size or explor-
ing more advanced fine-tuning techniques beyond LoRA
(e.g., full/partial fine-tuning, vision tower alignment, color
calibration, and robust data augmentation).
2) Ablation on Fine-T uning Strategy and Hyperparameters :
The model architecture is composed of three main compo-
nents: (i) a vision tower, (ii) a multimodal projector, and
(iii) a LLaMA-style language transformer. LoRA adapters
are inserted into the linear layers of both the self-attention
modules ( q, k, v, o ) and the feed-forward modules (gate,
up, down). The training configuration employs a batch
size of 4 with two-step gradient accumulation for 3 epochs.
Early stopping is applied when the validation loss does not
improve for three consecutive epochs.
As summarized in T able III , to separate component-
wise contributions and locate an eﬀiciency–performance
trade-off, we evaluate three controlled configurations:
•Language-only (L-only): LoRA applied exclusively to
the language transformer.
•Vision + Projector (V+Proj): LoRA applied to the
vision tower and multimodal projector.
•F ull: LoRA applied jointly to all three components.
T o further investigate the balance between representa-

AUTHOR et al.: TITLE 11
(a)
(c)
Fig. 6 : Overall performance comparison. (a) Radar chart shows MMIR-TCM’s consistent superiority . (b) A verage
results across tasks. (c) Score distributions highlight higher median and stability .
tional capacity and overfitting, we vary both the LoRA
rank ( r) and the training epochs. The results, presented
in T able II, show the following key observations:
•Module contributions. V+Proj yields larger gains
than L-only on both BLEU/TDEU, and F ull achieves
the best overall performance, indicating that visual
domain grounding is more influential than language-
only adaptation.
•Capacity & schedule. Increasing rfrom 16 to64 and
extending training from 3 to 10 epochs jointly improve
BLEU-4 and TDEU. However, moving from 10 to
20 epochs leads to only modest BLEU-4 gains while
TDEU shows diminishing returns, suggesting early
signs of overfitting (language-dominated drift).
3) Ablation on Prescription Generator :T o validate the
effectiveness and contribution of each key component of
our proposed MMIR-TCM’s RAG framework, we con-
ducted a series of ablation studies. W e systematically re-
moved specific modules from the full model and evaluated
the performance degradation. The results, visualized as
heatmaps in Fig. 7, compare the full MMIR-TCM model
against four ablated variants across three core TCM tasks
(Diagnosis, Syndrome Differentiation, and Prescription)
using five evaluation metrics. F or clarity in visualization,performance improvements exceeding 500% are capped at
500% in the heatmaps.
The first comparison reveals a substantial performance
gap against the zero-shot baseline (Fig. 7a), with improve-
ments reaching 484.8% (BLEU for Diagnosis) and 500.0%
(ROUGE-2 for Syndrome Differentiation). Component-
specific ablations further highlight their individual con-
tributions. Removing the clinical cases memory (w/o
Cases, Fig. 7d) caused the most significant degradation,
with performance dropping by 500.0% on both Diagnosis
and Syndrome Differentiation. Similarly , ablating the
LiuJing theory memory (w/o LiuJing, Fig. 7c) severely
compromised Syndrome Differentiation (500.0% drop in
BLEU score). The removal of tongue diagnosis (w/o
tongue, Fig. 7b) also led to a notable decline, particularly
impacting the Prescription task with a 213.7% drop in
BLEU.
These findings demonstrate that the clinical cases mem-
ory , LiuJing theory memory , and tongue diagnosis are all
integral yet complementary components, each providing
distinct and non-redundant contributions to the model’s
overall eﬀicacy in complex TCM reasoning tasks. This
validates the architectural design choices of the proposed
MMIR-TCM framework.

12 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
(a) (b)
(c) (d)
Fig. 7 : Heatmap Visualization of Performance Gains from MMIR-TCM’s Prescription Generator F ramework.
Fig. 8 : Comparison of MMIR-TCM and GPT-4o prescriptions. The left panels show two representative cases: (a) a case
where MMIR-TCM outperforms GPT-4o, and (b) a case favoring GPT-4o. The radar plot on the right summarizes
physicians’ average evaluation scores across all cases, highlighting overall model performance.

AUTHOR et al.: TITLE 13
F. User Preference Study .
1) Overall Ratings :W e conducted a blinded preference
study with 12 experienced TCM clinicians comparing
MMIR-TCM and GPT-4o on 20 random cases (240
evaluations). Outputs were scored from 1 to 5 on five
dimensions: Acc_SN, Acc_Diag, V alidity , Safety , and
Rationality . Aggregated ratings (Fig. 8) show a consis-
tent clinician preference for MMIR-TCM, with feedback
indicating better alignment with TCM reasoning. Case 1
strongly favored MMIR-TCM. In Case 2, GPT-4o scored
slightly higher on the five metrics, but clinicians still
preferred MMIR-TCM due to stronger perceived safety
and reliability , emphasizing that diagnostic correctness
and prescription safety are primary priorities.
2) Challenges and Limitations :T o quantify real-world
accuracy , we performed manual error analysis against
formal TCM hospital records. Five experts reviewed 2,362
generated cases and re-evaluated a random 50% subset
(1,181) for disease diagnosis, meridian attribution, and
syndrome consistency . Inconsistencies were marked as
incomplete or incorrect, and 50 representative error cases
were analyzed in six categories: SIE, DDE, ME, IPI, IMI,
and IDD.
F requency analysis (Fig. 9(a)) showed IMI as the
dominant error (76.0%, 38/50), followed by IPI (52.0%,
26/50), indicating a priority to improve meridian and
syndrome completeness. High-frequency terms (Fig. 9(b))
were concentrated in Shaoyin (23) and Jueyin (12), and
in complex syndrome combinations (e.g., Liver Depression
and Kidney Deficiency; Liver–Kidney Yin Deficiency),
suggesting limited integration of multifactorial TCM cues.
Co-occurrence analysis (Fig. 9(c)) showed strong IPI–
IMI coupling ( >20 times, Jaccard = 0.52, co-occurrence
>50%), moderate SIE–IMI coupling (16), and near-
independence of ME–SIE (Jaccard = 0.07). This suggests
”Error” and ”Incomplete” types may arise from different
reasoning biases and should be optimized separately .
The word cloud (Fig. 9(d)) confirms concentration in
meridian–syndrome differentiation and complex combi-
nations; clinically this indicates weakness in disease–
meridian linking and deficiency/viscera reasoning, while
technically it likely reflects terminology bias and limited
contextual TCM semantic understanding.
G. Clinical Relevance Study
T o assess translational potential, we examined prac-
tical applicability in real-world clinical settings. The
framework’s core value is improved diagnostic objectiv-
ity , eﬀiciency , and safety . Standardized tongue-pattern
analysis provides quantitative visual references, while the
RAG mechanism enables rapid retrieval of similar cases
and preliminary diagnostic reports. These outputs are
grounded in an authentic clinical knowledge base, reducing
“hallucinations” and improving reliability .
Nonetheless, error analysis in Fig. 9 highlights key
challenges. Among erroneous cases, 76% of IMI and 52%of IPI suggest the model may miss comorbid or context-
dependent syndromes, potentially affecting treatment ac-
curacy . MMIR-TCM should therefore be positioned as
an assistive system under professional supervision rather
than an autonomous agent. Its clinical relevance lies in
a human–machine collaborative paradigm that reduces
documentation and retrieval burden and allows clinicians
to focus on judgment and communication. F eedback from
experienced TCM physicians supports this collaborative
mode as a feasible and responsible path to implementa-
tion.
V. C ONCLUSION
W e present MMIR-TCM, an end-to-end multimodal
framework that combines robust tongue region extraction,
structured tongue diagnosis generation, and RAG for
evidence-anchored prescription recommendations. Eval-
uated on the multi-center MedTCM corpus with the
TDEU metric, the proposed pipeline improves attribute-
level diagnostic fidelity and reduces unsupported recom-
mendations by grounding prescriptions in deidentified
clinical precedents and documented formula rationale,
thereby enhancing factuality and clinician auditability .
Limitations include reliance on retrospective, site biased
records and sensitivity to image quality and metadata
completeness; future work will focus on prospective vali-
dation, broader multi center data collection, integration
of additional diagnostic modalities, and interface designs
that present provenance and uncertainty to clinicians.
Overall, MMIR-TCM offers a reproducible path toward
multimodal clinical decision support in TCM practice.
VI. REFERENCES
[1] J. S. Ryu, H. Kang, Y. Chu, and S. Y ang, “Vision-language
foundation models for medical imaging: a review of current
practices and innovations,” Biomedical Engineering Letters, pp.
1–22, 2025.
[2] T.-C. W u, C.-N. Lu, W.-L. Hu, K.-L. W u, J. Y. Chiang, J.-
M. Sheen, and Y.-C. Hung, “T ongue diagnosis indices for gas-
troesophageal reflux disease: a cross-sectional, case-controlled
observational study ,” Medicine, vol. 99, no. 29, p. e20471, 2020.
[3] M. Segawa, N. Iizuka, H. Ogihara, K. T anaka, H. Nakae,
K. Usuku, K. Y amaguchi, K. W ada, A. Uchizono, Y. Nakamura
et al., “Objective evaluation of tongue diagnosis ability using
a tongue diagnosis e-learning/e-assessment system based on
a standardized tongue image database,” F rontiers in Medical
T echnology , vol. 5, p. 1050909, 2023.
[4] G. Arji, R. Safdari, H. Rezaeizadeh, A. Abbassian,
M. Mokhtaran, and M. H. A yati, “A systematic literature
review and classification of knowledge discovery in traditional
medicine,” Computer methods and programs in biomedicine,
vol. 168, pp. 39–57, 2019.
[5] Q. Liu, Y. Li, P . Y ang, Q. Liu, C. W ang, K. Chen, and Z. W u,
“A survey of artificial intelligence in tongue image for disease
diagnosis and syndrome differentiation,” Digital Health, vol. 9,
p. 20552076231191044, 2023.
[6] Z. Zhao, X. Ren, K. Song, Y. Qiang, J. Zhao, J. Zhang,
and P . Han, “Pregenerator: T cm prescription recommendation
model based on retrieval and generation method,” IEEE Access,
vol. 11, pp. 103 679–103 692, 2023.
[7] R. Zhang, D. Tian, and Y. W ang, “Large language models in
traditional chinese medicine: A short survey and outlook,” AI
Medicine, vol. 2, no. 1, p. 3, 2025.

14 IEEE TRANSACTIONS AND JOURNALS TEMPLA TE
1.00 | 24 0.30 | 9 0.07 | 2 0.00 | 0 0.35 | 16 0.04 | 1 0.30 | 9 1.00 | 15 0.25 | 4 0.17 | 6 0.15 | 7 0.00 | 0 0.07 | 2 0.25 | 4 1.00 | 5 0.11 | 3 0.00 | 0 0.00 | 0 0.00 | 0 0.17 | 6 0.11 | 3 1.00 | 26 0.52 | 22 0.00 | 0 0.35 | 16 0.15 | 7 0.00 | 0 0.52 | 22 1.00 | 38 0.03 | 1 0.04 | 1 0.00 | 0 0.00 | 0 0.00 | 0 0.03 | 1 1.00 | 1 
SIE DDE ME IPI IMI IDD 
SIE DDE ME IPI IM I
ID D
Error T ype Error T ype Jaccard coefficien t
0.00 0.25 0.50 0.75 1.00 Er ror Type Co-occur rence Jaccard Coefficient Heatmap 
6LPLODULW\0HWULF-$% _$ŀ%__$ B| 
(a) (c) 
(b) (d) 
Fig. 9 : Statistical analysis of representative erroneous cases in MMIR-TCM
[8] Y. Lu, X. Zhao, and J. W ang, “Clinicalrag: enhancing clinical
decision support through heterogeneous knowledge retrieval,”
in Proceedings of the 1st W orkshop on T owards Knowledgeable
Language Models (KnowLLM 2024), 2024, pp. 64–68.
[9] Y. Dai, X. Shao, J. Zhang, Y. Chen, Q. Chen, J. Liao, F. Chi,
J. Zhang, and X. F an, “T cmchat: A generative large language
model for traditional chinese medicine,” Pharmacological Re-
search, vol. 210, p. 107530, 2024.
[10] W. Y ue, X. W ang, W. Zhu, M. Guan, H. Zheng, P . W ang,
C. Sun, and X. Ma, “T cmbench: A comprehensive benchmark
for evaluating large language models in traditional chinese
medicine,” arXiv preprint arXiv:2406.01126, 2024.
[11] P . Y u, K. Song, F. He, M. Chen, and J. Lu, “T cmd: A
traditional chinese medicine qa dataset for evaluating large
language models,” arXiv preprint arXiv:2406.04941, 2024.
[12] S. W ei, X. Peng, Y. W ang, T. Shen, J. Si, W. Zhang, F. Zhu,
A. V. V asilakos, W. Lu, X. W u et al., “Biancang: a tradi-
tional chinese medicine large language model,” IEEE Journal
of Biomedical and Health Informatics, 2025.
[13] X. Zhou, X. Dong, C. Li, Y. Bai, Y. Xu, K. C. Cheung,
S. Se, X. Song, R. Zhang, X. Zho et al., “T cm-ftp: fine-tuning
large language models for herbal prescription prediction,” in
2024 IEEE International Conference on Bioinformatics and
Biomedicine (BIBM). IEEE, 2024, pp. 4092–4097.
[14] Y. Zhuang, L. Y u, N. Jiang, and Y. Ge, “T cm-kllama: Intelligent
generation model for traditional chinese medicine prescriptions
based on knowledge graph and large language model,” Comput-
ers in Biology and Medicine, vol. 189, p. 109887, 2025.
[15] C. Zhou, H. F an, and Z. Li, “T onguenet: accurate localization
and segmentation for tongue images using deep neural net-
works,” IEEE access, vol. 7, pp. 148 779–148 789, 2019.
[16] Q. Gan, C. W ang, Z. Zhong, J. W u, Q. Ge, L. Shi, J. Shang,
and C. Liu, “Cross-modal attention model integrating tongue
images and descriptions: a novel intelligent tcm approach for
pathological organ diagnosis,” F rontiers in Physiology , vol. 16,
p. 1580985, 2025.
[17] X. Jin, L. Gao, A. T ong, Z. Chen, J. Kong, N. Sun, H. Ma,
Q. W ang, Y. Bai, and T. Su, “T cm-tongue: A standardizedtongue image dataset with pathological annotations for ai-
assisted tcm diagnosis,” arXiv preprint arXiv:2507.18288, 2025.
[18] Z. W ang, M. Hao, S. Peng, Y. Huang, Y. Lu, K. Y ao, X. Y ang,
and Y. Zhu, “T cmeval-sdt: a benchmark dataset for syndrome
differentiation thought of traditional chinese medicine,” Scien-
tific Data, vol. 12, no. 1, p. 437, 2025.
[19] H. Xian, Y. Xie, Z. Y ang, L. Zhang, S. Li, H. Shang, W. Zhou,
and H. Zhang, “Automatic tongue image quality assessment
using a multi-task deep learning model,” F rontiers in Physiology ,
vol. 13, p. 966214, 2022.
[20] J. Y an, J. Cai, Z. Xu, R. Guo, W. Zhou, H. Y an, Z. Xu, and
Y. W ang, “T ongue crack recognition using segmentation based
deep learning,” Scientific Reports, vol. 13, no. 1, p. 511, 2023.
[21] J. W u, Z. Li, Y. Cai, H. Liang, L. Zhou, M. Chen, and
J. Guan, “A novel tongue coating segmentation method based
on improved transunet,” Sensors, vol. 24, no. 14, p. 4455, 2024.
[22] S. Cao, Q. W u, and L. Ma, “T onguesam: An universal tongue
segmentation model based on sam with zero-shot,” in 2023 IEEE
international conference on bioinformatics and biomedicine
(BIBM). IEEE, 2023, pp. 4520–4526.
[23] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai,
J. Sun, H. W ang, and H. W ang, “Retrieval-augmented gen-
eration for large language models: A survey ,” arXiv preprint
arXiv:2312.10997, vol. 2, no. 1, 2023.
[24] S. W ang, F. Zhao, D. Bu, Y. Lu, M. Gong, H. Liu, Z. Y ang,
X. Zeng, Z. Y uan, B. W an et al., “Lins: A general medical
q&a framework for enhancing the quality and credibility of llm-
generated responses,” Nature Communications, vol. 16, no. 1,
p. 9076, 2025.
[25] J. He, Y. Guo, L. K. Lam, W. Leung, L. He, Y. Jiang, C. C.
W ang, G. Xing, and H. Chen, “Opentcm: a graphrag-empowered
llm-based system for traditional chinese medicine knowledge
retrieval and diagnosis,” arXiv preprint arXiv:2504.20118, 2025.
[26] F. Gaber, M. Shaik, F. Allega, A. J. Bilecz, F. Busch, K. Goon,
V. F ranke, and A. Akalin, “Evaluating large language model
workflows in clinical decision support for triage and referral
and diagnosis,” npj Digital Medicine, vol. 8, no. 1, p. 263, 2025.
[27] P . U. Unschuld and H. T essenow, Huang Di Nei Jing Su W en:

AUTHOR et al.: TITLE 15
An annotated translation of Huang Di’s inner classic–basic
questions: 2 volumes. Univ of California Press, 2011.
[28] D. Oniani, X. W u, S. Visweswaran, S. Kapoor, S. Kooragayalu,
K. Polanska, and Y. W ang, “Enhancing large language models
for clinical decision support by incorporating clinical practice
guidelines,” in 2024 IEEE 12th International Conference on
Healthcare Informatics (ICHI). IEEE, 2024, pp. 694–702.
[29] S. Kong, X. Y ang, Y. W ei, Z. W ang, H. T ang, J. Qin, S. Lan,
Y. W ang, J. Bai, Z. Chen et al., “Mtcmb: A multi-task bench-
mark framework for evaluating llms on knowledge, reasoning,
and safety in traditional chinese medicine,” arXiv preprint
arXiv:2506.01252, 2025.
[30] J. Xie, Y. Y u, Z. Zhang, S. Zeng, J. He, A. V asireddy ,
X. T ang, C. Guo, L. Zhao, C. Jing et al., “T cm-ladder: A
benchmark for multimodal question answering on traditional
chinese medicine,” arXiv preprint arXiv:2505.24063, 2025.
[31] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas:
Automated evaluation of retrieval augmented generation,” in
Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demon-
strations, 2024, pp. 150–158.
[32] S. Muller, A. Loison, B. Omrani, and G. Viaud, “Grouse:
A benchmark to evaluate evaluators in grounded question
answering,” arXiv preprint arXiv:2409.06595, 2024.
[33] X. W ang, M. T an, Q. Jin, G. Xiong, Y. Hu, A. Zhang, Z. Lu, and
M. Zhang, “Medcite: Can language models generate verifiable
text for medicine?” arXiv preprint arXiv:2506.06605, 2025.
[34] Z. Hong, Y. Y ue, Y. Chen, L. Cong, H. Lin, Y. Luo, M. H.
W ang, W. W ang, J. Xu, X. Y ang et al., “Out-of-distribution
detection in medical image analysis: A survey ,” arXiv preprint
arXiv:2404.18279, 2024.
[35] A. Balendran, C. Beji, F. Bouvier, O. Khalifa, T. Evgeniou,
P . Ravaud, and R. Porcher, “A scoping review of robustness con-
cepts for machine learning in healthcare,” npj Digital Medicine,
vol. 8, no. 1, p. 38, 2025.
[36] B. Lambert, F. F orbes, S. Doyle, H. Dehaene, and M. Dojat,
“T rustworthy clinical ai solutions: A unified review of uncer-
tainty quantification in deep learning models for medical image
analysis. ” Artif. Intell. Medicine, vol. 150, p. 102830, 2024.
[37] J. F ayyad, S. Alijani, and H. Najjaran, “Empirical validation of
conformal prediction for trustworthy skin lesions classification,”
Computer Methods and Programs in Biomedicine, vol. 253, p.
108231, 2024.
[38] S. Singla, M. Eslami, B. Pollack, S. W allace, and K. Batmanghe-
lich, “Explaining the black-box smoothly—a counterfactual
approach,” Medical Image Analysis, vol. 84, p. 102721, 2023.
[39] Y. F u, E. Baker, Y. Ding, and Y. Chen, “Constrained decoding
for secure code generation,” arXiv preprint arXiv:2405.00218,
2024.
[40] E. Asgari, N. Montaña-Brown, M. Dubois, S. Khalil, J. Balloch,
J. A. Y eung, and D. Pimenta, “A framework to assess clinical
safety and hallucination rates of llms for medical text summari-
sation,” npj Digital Medicine, vol. 8, no. 1, p. 274, 2025.
[41] T. Y. C. T am, S. Sivarajkumar, S. Kapoor, A. V. Stol-
yar, K. Polanska, K. R. McCarthy , H. Osterhoudt, X. W u,
S. Visweswaran, S. F u et al., “A framework for human evaluation
of large language models in healthcare derived from literature
review,” NPJ digital medicine, vol. 7, no. 1, p. 258, 2024.
[42] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma,
H. Khedr, R. Rädle, C. Rolland, L. Gustafson et al., “Sam
2: Segment anything in images and videos,” arXiv preprint
arXiv:2408.00714, 2024.
[43] J. Chae, L. Luo, X. Y uan, D. Y u, Z. Chen, L. Zhang, and P . Qin,
“Memory-sam: Human-prompt-free tongue segmentation via
retrieval-to-prompt,” arXiv preprint arXiv:2510.15849, 2025.
[44] O. Siméoni, H. V. V o, M. Seitzer, F. Baldassarre, M. Oquab,
C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa
et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.
[45] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy , P .-
E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou, “The faiss
library ,” IEEE T ransactions on Big Data, 2025.
[46] A. Y ang, A. Li, B. Y ang, B. Zhang, B. Hui, B. Zheng, B. Y u,
C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv
preprint arXiv:2505.09388, 2025.
[47] E. J. Hu, Y. Shen, P . W allis, Z. Allen-Zhu, Y. Li, S. W ang,
L. W ang, W. Chen et al., “Lora: Low-rank adaptation of large
language models. ” ICLR, vol. 1, no. 2, p. 3, 2022.[48] Y. A. Malkov and D. A. Y ashunin, “Eﬀicient and robust ap-
proximate nearest neighbor search using hierarchical navigable
small world graphs,” IEEE transactions on pattern analysis and
machine intelligence, vol. 42, no. 4, pp. 824–836, 2018.
[49] Y. Liu, K. Zhao, L. Luo, Z. Zhang, Z. Qian, C. Jiang, Z. Du,
S. Deng, C. Y ang, D. W u et al., “Diagnosing pathologic myopia
by identifying morphologic patterns using ultra widefield images
with deep learning,” npj Digital Medicine, vol. 8, no. 1, p. 435,
2025.
[50] G. Lin, W. Chen, Y. F an, Y. Zhou, X. Li, X. Hu, X. Cheng,
M. Chen, C. Kong, M. Chen et al., “Machine learning radiomics-
based prediction of non-sentinel lymph node metastasis in
chinese breast cancer patients with 1-2 positive sentinel lymph
nodes: a multicenter study ,” Academic Radiology , vol. 31, no. 8,
pp. 3081–3095, 2024.