# MammoWise: Multi-Model Local RAG Pipeline for Mammography Report Generation

**Authors**: Raiyan Jahangir, Nafiz Imtiaz Khan, Amritanand Sudheerkumar, Vladimir Filkov

**Published**: 2026-02-25 22:51:31

**PDF URL**: [https://arxiv.org/pdf/2602.22462v1](https://arxiv.org/pdf/2602.22462v1)

## Abstract
Screening mammography is high volume, time sensitive, and documentation heavy. Radiologists must translate subtle visual findings into consistent BI-RADS assessments, breast density categories, and structured narrative reports. While recent Vision Language Models (VLMs) enable image-to-text reporting, many rely on closed cloud systems or tightly coupled architectures that limit privacy, reproducibility, and adaptability. We present MammoWise, a local multi-model pipeline that transforms open source VLMs into mammogram report generators and multi-task classifiers. MammoWise supports any Ollama-hosted VLM and mammography dataset, and enables zero-shot, few-shot, and Chain-of-Thought prompting, with optional multimodal Retrieval Augmented Generation (RAG) using a vector database for case-specific context. We evaluate MedGemma, LLaVA-Med, and Qwen2.5-VL on VinDr-Mammo and DMID datasets, assessing report quality (BERTScore, ROUGE-L), BI-RADS classification, breast density, and key findings. Report generation is consistently strong and improves with few-shot prompting and RAG. Classification is feasible but sensitive to model and dataset choice. Parameter-efficient fine-tuning (QLoRA) of MedGemma improves reliability, achieving BI-RADS accuracy of 0.7545, density accuracy of 0.8840, and calcification accuracy of 0.9341 while preserving report quality. MammoWise provides a practical and extensible framework for deploying local VLMs for mammography reporting within a unified and reproducible workflow.

## Full Text


<!-- PDF content starts -->

MammoWise: Multi-Model Local RAG Pipeline for
Mammography Report Generation
Raiyan Jahangir, Nafiz Imtiaz Khan, Amritanand Sudheerkumar, Vladimir Filkov
University of California, Davis, CA, USA
Abstract
Screening mammography is high-volume, time-
sensitive, and documentation-heavy. Radiologists
must translate subtle visual findings into consis-
tentBI-RADSassessments, breast-densitycategories,
and narrative reports. Although recent Vision Lan-
guage Models (VLMs) make image-to-text report-
ing plausible, many demonstrations rely on closed,
cloud-hosted systems (raising privacy, cost, and re-
producibility concerns) or on tightly coupled, task-
specific architectures that are hard to adapt across
models, datasets, and workflows. We presentMam-
moWise, a local, multi-model pipeline that turns
open-sourceVLMsintoclinically-styledmammogram
report generators and multi-task classifiers.Mam-
moWiseaccepts any Ollama-hosted VLM and any
mammography dataset, supports zero-shot, few-shot,
and Chain-of-Thought prompting, and includes an
optional multimodal Retrieval Augmented Genera-
tion (RAG) mode that retrieves image-text exem-
plars from a vector database to provide case-specific
context. To illustrate the utility ofMammoWisein
complex use cases, we evaluate three open VLMs
(MedGemma,LLaVA-Med,Qwen2.5-VL) on VinDr-
Mammo and DMID datasets, and assess report qual-
ity (BERTScore, ROUGE-L), BI-RADS classifica-
tion (5-class), breast density (ACR A–D), and key
findings (mass, calcification, asymmetry). Across
models and datasets, report generation is consis-
tently strong. It generally improves with few-shot
prompting and RAG, whereas prompting-only classi-
ficationisfeasiblebuthighlysensitivetothe(dataset,
model, prompt) combination. To improve reliability,
we added parameter-efficient fine-tuning (QLoRA) to
MedGemmaon VinDr-Mammo, yielding substantial
gains, including BI-RADS accuracy of 0.7545, den-
sity accuracy of 0.8840, and calcification accuracy
of 0.9341, while preserving report quality. Overall,
MammoWiseprovides a practical, extensible scaffold
for deploying and studying local VLMs for mammog-
raphyreporting, spanningprompting, RAG,andfine-
tuning within a unified, reproducible workflow.1 Introduction
Screening mammography is among the most widely
used imaging modalities in modern preventive care
[1]. Every study requires careful inspection of bilat-
eral views, such as Cranio-Caudal (CC) and Medio-
Lateral Oblique (MLO), identification of subtle find-
ings (masses, asymmetries, calcifications), and con-
version of visual evidence into standardized clinical
language: BI-RADSevaluationandrecommendation,
breast-density assessment, and a structured narrative
report [2]. This translation step is essential, as re-
ports drive follow-up imaging, biopsies, and longitu-
dinal follow-up [3]. Still, it is also labor-intensive and
vulnerable to variability in phrasing, completeness,
and coding-relevant details. As imaging volumes con-
tinue to grow, tools that can help draft consistent,
clinically styled mammogram reports have immedi-
ate practical value, without compromising privacy or
introducing unsafe hallucinations [4].
VLMs offer a promising direction because they
can jointly reason over images and text and gen-
erate natural-language output [5]. In principle, a
VLM could ingest mammogram views and produce
(i) a narrative report that follows radiology conven-
tions and (ii) structured labels such as BI-RADS and
density. However, in practice, two friction points
limit the usefulness of the real world. First, many
of the strongest demonstrations of report genera-
tion use large, closed, cloud-hosted models [6]. For
clinical imaging, this creates deployment barriers.
For example, protected health information may be
transmitted off-premises, operational costs can be
high, and results are difficult to reproduce or audit.
Second, open-source VLMs can be run locally, but
"out of the box,” they are often not specialized for
mammography and may produce clinically implausi-
ble details unless carefully guided [7]. Bridging this
gap typically requires either prompt engineering [8]
(lightweight but unreliable for fine-grained classifica-
tion) or fine-tuning [9, 10] (more reliable but model-
and dataset-dependent and often expensive).
Existing multimodal mammography-focused sys-
1arXiv:2602.22462v1  [cs.CV]  25 Feb 2026

tems illustrate these trade-offs. Several lines of
work emphasize specialized architectures or training
recipes optimized for specific tasks (e.g., malignancy,
breast density, or lesion detection), while others fo-
cus on reporting using proprietary models [11]. As a
result, practitioners and researchers who want to (a)
run locally, (b) compare multiple open VLMs, and
(c) evaluate multiple adaptation strategies (prompt-
ing, retrieval-augmented prompting, and fine-tuning)
often face fragmented codebases, inconsistent exper-
imental protocols, and limited support for producing
complete, clinically styled reports rather than only
labels.
This paper takes a different stance. Instead of
proposing yet another bespoke model, we intro-
duceMammoWise, a multi-model, locally runnable
pipeline that makes open VLMs usable for mam-
mogram report generation and associated classifica-
tion tasks. Conceptually, MammoWise is a modular
one-stop shop, comparable to middleware/IDE, that
provides a unified interface to launch local VLMs,
configure multimodal RAG, and run standardized
evaluations across multiple mammography use cases
and metrics. The design goal is to enable prac-
tical, reproducible experimentation under realistic
constraints. The pipeline can plug in any Ollama-
hosted VLM [12], ingest standard mammography
datasets, and run a consistent end-to-end workflow
that includes preprocessing, prompt templating, out-
put parsing, and evaluation.MammoWisesupports
zero-shot [13], few-shot [14], and Chain-of-Thought
(CoT) prompting [15]. It also adds an optional RAG
[16] mode that selects image-text exemplars from a
multimodal vector database to provide a customized
context to the input mammogram. When prompt-
ing alone is insufficient, especially for stable, high-
fidelity classification, we also support Parameter-
Efficient Fine-Tuning (PEFT) [17] (QLoRA), which
adapts an open VLM to the target label space with-
out full model retraining.
We evaluateMammoWiseacross two datasets
(VinDr-Mammo [18] and DMID [19]) and three open
VLMs (MedGemma[20],LLaVA-Med[21], Qwen2.5-
VL [22]). The results highlight a consistent pat-
tern. Report generation is robust and improves fur-
ther with few-shot prompting and RAG. At the same
time, prompting-only classification is highly variable
across model/dataset/prompt choices. To reduce
this sensitivity, we fine-tuneMedGemmaon VinDr-
Mammo using QLoRA [23] and observe substan-
tial improvements in classification quality, reaching
0.7545 BI-RADS accuracy, 0.8840 density accuracy,
and 0.9341 calcification accuracy (F1-score 0.9313)
while maintaining strong report-generation behavior.These findings suggest a pragmatic division of labor:
prompting and retrieval are often sufficient to draft
clinically styled reports, but dependable structured
classification benefits from lightweight fine-tuning.
Contributions
This paper makes the following contributions:
•A local, multi-model pipeline (MammoWise)
that turns open VLMs into mammogram re-
port generators and multi-task classifiers across
datasets and prompting regimes.
•A multimodal RAG workflow for case-specific
few-shot context using an image-text vector
database.
•A systematic comparison of prompting, RAG,
and QLoRA fine-tuning across models and
datasets, with unified evaluation for both nar-
rative report quality and clinical labels.
•Evidence that parameter-efficient fine-tuning
can substantially stabilize and improve classifi-
cation performance in this setting.
Based on these contributions, we formulate the fol-
lowing research questions:
RQ 1:How well do local medical VLMs generate
structured reports under prompting vs RAG?
RQ 2: When does lightweight fine-tuning out-
perform prompting/RAG for classification?
The remainder of the paper reviews related work,
describes theMammoWisearchitecture and experi-
mental design, presents results and analyses across
adaptation strategies, and closes with practical im-
plications and limitations.
2 Literature Background
Several recent studies have explored how VLMs can
assist radiologists with mammogram interpretation.
Haver et al. [24] investigated the potential ofChat-
GPTto determine BI-RADS assessments from mam-
mogram images. Pesapane et al. [25] usedGPT-
4to generate mammogram reports and identify ab-
normalities, such as masses and microcalcifications.
While these works demonstrated that large, closed-
source models can describe mammogram findings,
they also reported hallucinations, low sensitivity and
specificity, and the inherent privacy and cost issues
associated with cloud-based models.
2

DMID
BI-RADS: 
1
Breast 
Density:D
Findings:....
Multimodal 
Embeddings
Merged
Image
JSON 
REport
CSV 
Files
4 
Images 
per 
patient
VINDR-MAmmo
Reports
Image
Data 
Collection
BI-RADS: 
1
Breast 
Density:D
Findings:....
JSON 
REport
Image
Multimodal 
Embeddings
Medgemma
Zero-shot
llava-med
qwen2.5 
VL
few 
shot
cot
Rag 
fewshot
data 
preprocessing
prompt 
design
model 
selection
findings
type 
of 
output
breast 
density
birads
mass
calcification
asymmetry
Suspicion
finetune
Precision
Recall
f1-score
BERtscore
ROuge-l
Accuracy
Specificity
evaluationFigure 1: Overall methodology of the study
Other work has investigated CLIP-style models as
backbones for mammography. Moura et al. [26] eval-
uated several Contrastive Language-Image Pretrain-
ing (CLIP) variants-CLIP[27],BiomedCLIP[28],
andPubMedCLIP[29] for breast-density and BI-
RADS classification. Khan et al. [30] usedMed-
CLIP[31] to construct diverse teaching cases for ra-
diology education via image-text retrieval, exploring
zero-shot, few-shot, and supervised prompting sce-
narios.
Several specialized VLM architectures have also
been proposed. Chen et al. introducedLLaVA-
Mammo[32], a fine-tuned variant ofLLaVA[33] tai-
lored for BI-RADS, breast density, and malignancy
classification. Increasing the language model size
from 7B to 13B parameters improved density accu-
racy from 72.1 to 76.6 and malignancy AUC from
0.687 to 0.723. Their laterLLaVA-MultiMammo
model [34] extended capabilities to breast cancer risk
prediction while retaining core classification tasks.
Ghosh et al. [35] proposedMammoCLIP, a VLM
combining CNN-based vision [36] andBioBERT[37]
language encoders trained on mammogram-report
pairs. Their model supports the classification of ma-
lignancy, density, mass, and calcification, and pro-
vides visual explanations via a novel feature attribu-
tion method (Mammo-Factor). Jain et al. [38] devel-opedMMBCD, a multimodal model that integrates
mammogram images and clinical histories using ViT
andRoBERTa-basedembeddings[39]. Caoetal. [40]
introducedMammoVLM, aGLM-4-9B-based VLM
with a dedicated visual encoder, fine-tuned to pro-
vide diagnostic assistance at a level comparable to a
junior radiologist.
Very few works provide a single reusable pipeline
that supports both report generation and multi-task
labels across models/datasets. Our work addresses
this gap by framingMammoWiseas a reusable
pipelineratherthan asinglefixedmodel, andbyeval-
uating its behavior across multiple VLMs, datasets,
and adaptation strategies.
3 Methodology
The study’s overall methodology is shown in Figure
1 and discussed in detail in the following subsections.
3.1 Data Collection
We used two mammography datasets for our work.
The primary dataset, VinDr-Mammo, was prepared
by Nguyen et al. [18] and is used for most experi-
ments, including fine-tuning. The second dataset is
3

the DMID dataset prepared by Oza et al. [19].
The VinDr-Mammo dataset consists of 20,000
images from 5,000 Vietnamese patients, with four
images per patient (two per breast): Cranio-Caudal
(CC) and Medio-Lateral Oblique (MLO) for both
left and right breasts. The dataset also includes
metadata describing acquisition devices, study num-
bers, and image identifiers, as well as breast-level
information like tissue density (ACR A/B/C/D) and
BI-RADS score (1–5), and abnormal findings such as
masses, calcifications, and other abnormalities. The
dataset is publicly available via PhysioNet
(https://physionet.org/content/vindr-
mammo/1.0.0/).
DMID is a relatively small but lesion-enriched
dataset that contains 510 mammogram images from
India, of which 300 include abnormalities. Each
image is paired with a corresponding text report
describing breast density, BI-RADS score, and any
abnormal findings. This image-report pairing makes
DMID particularly suitable for evaluating report-
generation behavior and prompting strategies. The
dataset is accessible via Figshare
(https://figshare.com/authors/Parita_Oza/17353984).
3.2 Data Preprocessing
To prevent memory overload and ensure compatibil-
ity with the VLMs, we resize all images from both
datasetsto afixedsize of512×512pixels. For VinDr-
Mammo, we then merge the four views (right/left
CC and right/left MLO) of each patient into a sin-
gle composite image, as shown in Figure 1, arranged
symmetrically: CC views are side by side at the top,
and MLO views are side by side at the bottom. This
layout mimics the way radiologists routinely inspect
mammograms, comparing the right and left breasts
on both projections to assess asymmetries and subtle
patterns [41, 42].
From the 20,000 single-view images, this merging
process yields 5,000 composite images, one per pa-
tient. For each merged image, we generate a cor-
responding report, yielding 5,000 composite image-
report pairs. In the report:
•Breast density is taken directly from the breast-
level labels.
•For BI-RADS, when different views have differ-
ent categories, we take the highest BI-RADS
across the four views as the patient-level label,
reflecting clinical caution.
•The findings of all views are combined into a sin-
gle sentence summarizing the presence and loca-tion of masses, calcifications, asymmetries, and
other abnormalities.
•We define a field of derivedsuspicionthat indi-
cates whether the case is labeled “healthy” (BI-
RADS 1), “benign” (BI-RADS 2–3), or “suspi-
cious” (BI-RADS 4-5). Importantly, this la-
bel reflectsradiologic suspicionimplied by BI-
RADS, not biopsy-confirmed cancer.
For the DMID dataset, there were already 510 cor-
responding reports for each image. We extract key
information, such as “breast density”, “BI-RADS”,
and “findings”, from those reports and store them in
JavaScript Object Notation (JSON) format.
For RAG implementation, we convert these image-
report pairs into multimodal embeddings. We di-
vide both the data into an 80:20 train-test split. For
VinDr-Mammo, it is a patient-level split, whereas in
DMID, it is an image-level split. The image and
text embeddings are generated using a multimodal
OpenCLIPEmbedding function, which applies a Vi-
sion Transformer model [43] for image encoding and
a text encoder for report encoding. The resulting
embeddings are stored as semantic indexes in Chro-
maDB [44], a vector database used for RAG retrieval
in our experiments.
3.3 Model Selection
We selected three open-source VLMs available in Ol-
lama [12], a framework that supports local hosting
of multimodal language models. Each of these mod-
els can take an image and a text prompt as input
and generate text as output. Because we work with
medical data, we prioritize models that have been
instruction-tuned on medical or biomedical content
when possible. The three models areMedGemma,
LLaVA-Med, andQwen2.5-VL, described below.
MedGemma:MedGemmais a VLM created by
Google [20], derived from theGemma 3family [45].
It is specifically trained on medical data from radi-
ology, pathology, and dermatology to generate med-
ically grounded text based on multimodal input. It
comes in 4B and 27B parameter variants; we use the
4B model to keep hardware requirements reasonable.
LLaVA-Med:LLaVA-Med[21] is a fine-tuned
version of the general-purposeLLaVAmodel [33],
trained on biomedical datasets. The baseLLaVAar-
chitecture combines a vision encoder with a Vicuna-
based language backbone [46]. We use the 7B-
parameterLLaVA-Medvariant.
Qwen2.5-VL:Qwen2.5-VL[22] is an improved
version of theQwen2multimodal model [47]. It also
has 7B parameters. Although it is not specifically
4

ChromaDB
Reports
Image
Embeddings
Indexing
User
Multimodal 
prompt
Embeddings
Semantic 
search
Prompt 
with 
similar 
examples
Response
ModelFigure 2: RAG pipeline used in this study
tuned on medical data, it is trained to extract and
reason about text and objects from images, making
it a plausible candidate for detecting abnormalities in
mammogram images.
3.4 Prompt Design
Prompt engineering is the process of creating cus-
tomized instructions for LLMs and VLMs to guide
them toward appropriate, well-structured responses
[48]. In our setting, the prompts include textual in-
structions and one or more mammogram images. To
answer RQ1, we focus on three widely used prompt-
ing styles: zero-shot learning [13, 49], few-shot learn-
ing [14], and Chain-of-Thought (CoT) prompting
[15].
In thezero-shotsetting, the model is given only
high-level instructions about its role (“You are a
board-certified breast radiologist...”), the image lay-
out(4-viewarrangement), andthedesiredoutputfor-
mat (JSON with fields for breast density, findings,
BI-RADS, and suspicion). Forfew-shotprompting,
we augment the zero-shot template with five fixed
examples of image-report pairs that illustrate the de-
sired JSON structure and reporting style. These ex-
amples demonstrate how to map inputs to outputs.
ForCoT prompting, we modify the prompt so that
the model emulates a radiologist’s reasoning process
step by step: first assess breast density, then identify
findings, then determine whether the breast has sus-
picion of malignant tumors or not, and finally assign
a BI-RADS category.All prompt templates used in this study are pro-
vided in the Appendix.
3.5 Experimental Design
All experiments are conducted on a local machine,
and each VLM is served via Ollama. We set the tem-
perature parameter to 0 throughout, because mam-
mogram reporting is a high-stakes task where we pri-
oritize deterministic, low-variance outputs over cre-
ative variation [50]. Our experiments fall into three
main configurations.
3.5.1 Base Configuration
In the base setting, we provide the model only with
the prompt (zero-shot, few-shot, or CoT) and the
images. The model’s responses are captured and
stored locally. This configuration isolates the effect
of prompting alone, without additional retrieval or
fine-tuning.
3.5.2 RAG Configuration
Hallucination-plausible but incorrect or fabricated
content is a significant concern for VLMs [51]. For
mammogram reports, hallucinations are unaccept-
able. Explainability is another key requirement: clin-
icians want to understandwhya model produced a
particular conclusion [52, 16]. One practical way
to improve grounding and interpretability is to use
Retrieval-AugmentedGeneration(RAG) [53,54,55],
5

Table 1: BI-RADS class counts before and after rebalancing for fine-tuning.
BI-RADS Original After Action
1 3331 500 Downsample (random)
2 1167 500 Downsample (random)
3 242 500 Augment (flip + scale)
4 205 500 Augment (flip + scale)
5 55 200 Augment (flip + scale + translation; capped)
Total 5000 2200
where the model is given relevant supporting exam-
plesordocumentsascontextalongsidetheuserquery.
A recent Nature article [56] also highlights RAG as a
promising mechanism for improving transparency in
medical generative models.
To enable the RAG, we design a pipeline that dy-
namically retrieves examples and feeds them into the
prompt (Figure 2). When a new image and prompt
are provided, they are embedded in a multimodal
space and sent to the ChromaDB vector database.
The retrieval index is built only from training data
with patient-level separation to avoid leakage. We
then retrieve the five most similar image–report pairs
using cosine similarity. These retrieved examples are
inserted into the zero-shot prompt template, yielding
a dynamic, context-aware few-shot prompt whose ex-
amples are semantically close to the current query.
This is theRAGconfiguration.
3.5.3 Fine-Tune Configuration
Prompting alone leaves the model weights un-
changed. The model does not gain new knowledge,
which can limit performance. Fine-tuning can im-
prove accuracy and robustness, but fine-tuning large
VLMs to their full extent is often impractical on
commodityhardware. Parameter-efficientfine-tuning
[17, 57] addresses this by training only a small set of
adapter parameters while freezing the base model.
We fine-tuneMedGemma, the smallest of our
candidate medical VLMs, on VinDr-Mammo using
a class-rebalanced training subset constructed via
data augmentation [58]. The original cohort ex-
hibits a strong BI-RADS imbalance (Table 1). To
mitigate this, we downsampled the majority classes
(BI-RADS 1 and 2) to 500 images each by random
sampling, and augmented the minority classes using
simple geometric transforms commonly used in
previous mammography work (flipping, scaling, and
translation) [59, 60]. When applying horizontal flips,
we update any laterality-sensitive report fields to
preserve semantic consistency. We increased the
number of BI-RADS 3 and 4 cases to 500 each. ForBI-RADS 5, we increased the number of images from
55 to 200 (rather than 500) to respect a conservative
augmentation cap reported in previous work (not
exceeding 300% augmentation) [61]. This yields a
final fine-tuning set of 2,200 images. We loaded the
pre-trainedMedGemmamodel from HuggingFace
[62] and applied Quantized Low-Rank Adaptation
(QLoRA) [23], an efficient variant of LoRA [63].
QLoRA freezes the base weights and inserts low-rank
adapter matrices into selected layers; only these
adapters are trained, and at inference, their outputs
are combined with the frozen weights to produce
predictions. We fine-tune under two output formats:
multi-task generation, where the model outputs the
full JSON report containing all target fields in a
single response, andsingle-task generation, where the
model is prompted to output only one target field at
a time. Unless stated otherwise, we report results for
multiple checkpoints (multi-task: 3/6/10/15 epochs;
single-task: 6/10 epochs) to capture task-dependent
and potentially non-monotonic training effects.
We trained with different numbers of epochs to
observe how performance scales with them. The
hyperparameters are as follows:
Temperature: 0
Lora Alpha: 16
Lora Dropout: 0.05
Bias: None
Learning Rate: 2e-4
Batch size: 1
Optimizer: Adamw_bnb_8bit
Gradient Accumulation Steps: 8
Optimization: Qlora
3.6 Evaluating Model Response
We evaluate models along two main axes: classifica-
tion performance and generation performance.
Forclassification, we consider three multi-class
tasks (BI-RADS, breast density, and suspicion) and
6

three binary tasks (presence or absence of mass, cal-
cification, and asymmetry), reflecting abnormalities
that radiologists routinely prioritize in mammogram
interpretation. We report macro-averaged accuracy,
precision, recall, F1-score, andspecificity. Givenclass
imbalances in both datasets, macro averaging ensures
that minority classes receive equal weight [64, 65].
This choice aligns with our study goal of compar-
ing reliability across rare and common labels under
prompting, RAG, and lightweight fine-tuning. How-
ever, macro-averaged improvements do not necessar-
ily imply well-calibrated performance under the im-
balanced distribution. Rather, calibration-oriented
evaluation would be required for deployment-facing
conclusions.
Forgeneration, we assess whether the model’s
JSON output resembles human-written mammogram
reports. We focus on BI-RADS descriptions, breast
density descriptions, and findings. We compute
BERTScore [66] and ROUGE-L [67], since both cap-
ture semantic similarity rather than exact n-gram
matches. BERTScore measures similarity in embed-
ding space, while ROUGE-L evaluates the longest
common subsequence between generated and refer-
ence text.
4 Results & Discussion
Detailed results are provided in the Appendix. Here,
we focus on the key findings relevant to our research
questions.
4.1 Report generation (textual simi-
larity)
We first evaluate report-generation similarity us-
ing only text-similarity metrics (BERTScore and
ROUGE-L) on narrative fields, BI-RADS, den-
sity, and findings directly read by clinicians. On
VinDr-Mammo (Table 2), retrieval generally im-
proves narrative similarity beyond prompting alone.
The best BI-RADS-text BERTScore increases from
0.9313 (few-shot,Qwen2.5-VL) to 0.9401 (RAG,
MedGemma), while ROUGE-L improves from 0.3593
to 0.4347 (RAG,MedGemma), indicating better lex-
ical alignment and phrasing consistency for the clin-
ically salient BI-RADS narrative. For density text,
ROUGE-Limprovesfrom0.8643(few-shot,Qwen2.5-
VL) to 0.8740 (RAG,Qwen2.5-VL), and for the find-
ings text, ROUGE-L increases from 0.4090 (few-shot,
LLaVA-Med) to 0.4652 (RAG,Qwen2.5-VL). Over-
all, on VinDr-Mammo, RAG yields small but consis-
tent gains in narrative similarity, suggesting that re-trieved context helps the model better anchor phras-
ing and content for structured report fields expressed
in free text.
On DMID (Table 3), narrative trends are more
mixed and highlight that retrieval quality and corpus
fit can strongly affect textual similarity. BI-RADS,
BERTScore, and ROUGE-L remain the same. Den-
sity text improves more clearly, with BERTScore in-
creasing from 0.8860 to 0.8991 and ROUGE-L from
0.3964 to 0.4939. In contrast, the similarity of
the findings’ text drops substantially under RAG
(BERTScore from 0.9017 to 0.8615 and ROUGE-L
from 0.5423 to 0.2706). These results suggest that
retrieval can reinforce narrative similarity (as in den-
sity) or introduce distracting or mismatched context
(as in findings), depending on how well the retrieved
examples align with the target distribution and the
constraints of the prompt. These DMID results illus-
trate that RAG is not universally beneficial for nar-
rative similarity. The DMID reports are short and al-
ready tightly paired with each image. In this setting,
retrievalcanintroduceredundantormismatchedcon-
text, leading to lexical and semantic noise and reduc-
ing overlap-based metrics even when the model re-
mains clinically plausible. In particular, retrieved ex-
amples may differ in the type, distribution, or phras-
ing conventions of the abnormality, causing the gen-
erated findings sentence to drift toward the retrieved
style rather than the ground-truth wording. More-
over, when multiple retrieved examples are injected,
the model may over-condition on the example text,
effectively “crowding out” direct image-based reason-
ing. Overall, RAG effectiveness depends on the im-
age and report, on the retrieval, and on how strongly
prompts constrain the model to prioritize the input
image over retrieved narratives.
4.2 Prompting vs RAG for structured
labels
We next compare prompting and RAG for structured
label prediction without fine-tuning, using classifi-
cation metrics for multi-class tasks (BI-RADS, den-
sity) and, where available, binary findings (calcifi-
cation, mass, asymmetry, suspicion). On VinDr-
Mammo (Table 2), RAG tends to provide modest
improvements in structured prediction compared to
best-prompt baselines. For BI-RADS classification,
accuracy improves from 0.2909 (few-shot,LLaVA-
Med) to 0.3190 (RAG,MedGemma), with corre-
sponding improvements in F1-score from 0.2217 to
0.2756 (RAG,Qwen2.5-VL). Breast density accu-
racy improves from 0.7832 to 0.8068 (both best with
Qwen2.5-VL), and calcification shows stronger gains
7

Table 2: Performance of the models on VinDr-Mammo dataset without and with RAG
Task Evaluation ParametersWithout RAG With RAG
Best Result Prompt Model Result Model
BI-RADSAccuracy 0.2909 Fewshot LLAVA-Med 0.319 MedGemma
Precision 0.4121 Fewshot Qwen2.5VL 0.4508 Qwen2.5VL
Recall 0.2909 Fewshot LLAVA-Med 0.319 MedGemma
F1-score 0.2217 Fewshot LLAVA-Med 0.2756 Qwen2.5VL
Specificity 0.8165 Fewshot LLAVA-Med 0.8265 MedGemma
BERTScore 0.9313 Fewshot Qwen2.5VL 0.9401 MedGemma
ROUGE-L 0.3593 Fewshot Qwen2.5VL 0.4347 MedGemma
Breast DensityAccuracy 0.7832 Fewshot Qwen2.5VL 0.8068 Qwen2.5VL
Precision 0.7923 Fewshot LLAVA-Med 0.8185 Qwen2.5VL
Recall 0.7832 Fewshot Qwen2.5VL 0.8068 Qwen2.5VL
F1-score 0.69 Fewshot MedGemma 0.7537 Qwen2.5VL
Specificity 0.7754 Zeroshot MedGemma 0.8082 LLAVA-Med
BERTScore 0.9788 Fewshot Qwen2.5VL 0.9808 Qwen2.5VL
ROUGE-L 0.8643 Fewshot Qwen2.5VL 0.874 Qwen2.5VL
FindingsBERTScore 0.9053 Fewshot Qwen2.5VL 0.9103 MedGemma
ROUGE-L 0.409 Fewshot LLAVA-Med 0.4652 Qwen2.5VL
CalcificationAccuracy 0.8391 Fewshot Qwen2.5VL 0.8591 Qwen2.5VL
Precision 0.865 Fewshot MedGemma 0.8409 Qwen2.5VL
Recall 0.8391 Fewshot Qwen2.5VL 0.8591 Qwen2.5VL
F1-score 0.7679 Fewshot Qwen2.5VL 0.8432 Qwen2.5VL
Specificity 0.5477 Zeroshot Qwen2.5VL 0.6879 MedGemma
MassAccuracy 0.6218 CoT LLAVA-Med 0.6582 Qwen2.5VL
Precision 0.7647 Fewshot Qwen2.5VL 0.7041 Qwen2.5VL
Recall 0.6218 CoT LLAVA-Med 0.6582 Qwen2.5VL
F1-score 0.5016 Zeroshot MedGemma 0.5726 Qwen2.5VL
Specificity 0.5211 Fewshot MedGemma 0.5845 MedGemma
AsymmetryAccuracy 0.8915 Zeroshot LLAVA-Med 0.8664 Qwen2.5VL
Precision 0.7611 Fewshot LLAVA-Med 0.848 LLAVA-Med
Recall 0.8915 Zeroshot LLAVA-Med 0.8664 Qwen2.5VL
F1-score 0.7875 Zeroshot MedGemma 0.8373 Qwen2.5VL
Specificity 0.5215 CoT LLAVA-Med 0.5951 Qwen2.5VL
SuspicionAccuracy 0.6841 Fewshot Qwen2.5VL 0.735 Qwen2.5VL
Precision 0.7841 Fewshot Qwen2.5VL 0.7282 Qwen2.5VL
Recall 0.6841 Fewshot Qwen2.5VL 0.735 Qwen2.5VL
F1-score 0.5897 Fewshot LLAVA-Med 0.6961 Qwen2.5VL
Specificity 0.5347 CoT MedGemma 0.617 Qwen2.5VL
in several metrics (accuracy 0.8391 to 0.8591 and F1-
score 0.7679 to 0.8432). The Mass prediction im-
proves moderately (accuracy 0.6218 to 0.6582 and
F1-score 0.5016 to 0.5726). The asymmetry illus-
trates the metric-dependent behavior. That is, accu-
racy decreases from 0.8915 (zero-shot,LLaVA-Med)
to 0.8664 (RAG,Qwen2.5-VL), yet the F1-score im-
proves from 0.7875 to 0.8373, consistent with RAG
shifting the operating point rather than uniformly
improving all metrics. Suspicion prediction improves
(accuracy 0.6841 to 0.7350 and F1-score 0.5897 to0.6961). In aggregate, RAG typically yields incre-
mental gains for structured labels, but improvements
are not uniformly monotonic across tasks or metrics.
On DMID (Table 3), the effect of RAG on struc-
tured labels is dataset-dependent and can be much
larger. For BI-RADS classification, RAG produces a
substantial increase in accuracy (0.57 to 0.8950) and
F1-score (0.59 to 0.9300), suggesting that retrieved
examples can dramatically stabilize label inference
in this setting. However, density classification shows
onlyslightchangesinaccuracy(0.3589to0.3739)and
8

Table 3: Performance of the models on DMID dataset without and with RAG
Task Evaluation ParametersWithout RAG With RAG
Best Result Prompt Model Best Result Model
BI-RADSAccuracy 0.57 CoT Qwen2.5VL0.895MedGemma
Precision 0.8139 Fewshot LLAVA-Med0.9757MedGemma
Recall 0.57 CoT Qwen2.5VL0.895MedGemma
F1-score 0.59 CoT Qwen2.5VL0.93002MedGemma
Specificity 0.8602 Fewshot LLAVA-Med0.9807MedGemma
BERTScore0.9999Fewshot Qwen2.5VL0.9999MedGemma
ROUGE-L0.4906Fewshot MedGemma0.4906MedGemma
Breast DensityAccuracy 0.3589 Fewshot MedGemma0.3739MedGemma
Precision 0.4585 CoT Qwen2.5VL0.65MedGemma
Recall 0.3589 Fewshot MedGemma0.3739MedGemma
F1-score0.3715Fewshot MedGemma 0.3605 MedGemma
Specificity0.7886Fewshot MedGemma 0.7877 MedGemma
BERTScore 0.886 Fewshot MedGemma0.8991MedGemma
ROUGE-L 0.3964 Fewshot MedGemma0.4939MedGemma
FindingsBERTScore0.9017Fewshot MedGemma 0.8615 LLAVA-Med
ROUGE-L0.5423Fewshot MedGemma 0.2706 LLAVA-Med
decreases in F1-score (0.3715 to 0.3605), suggesting
that retrieval gains are not guaranteed and may de-
pend on label distribution, report style, and the de-
gree to which retrieval provides discriminative cues
for the target label.
RQ1: How well do local medical VLMs
generate structured reports under
prompting vs RAG?
RQ1Findings:Local medical VLMs can
generate high-quality structured reports under
prompting alone, particularly with few-shot
prompting. Adding RAG generally improves re-
port similarity and often improves downstream
label quality, with noticeable narrative gains.
However, the benefit is not uniform across all
text fields, indicating sensitivity to dataset
characteristics and retrieval fit.
4.3 Effect of QLoRA fine-tuning
Finally, we show the impact of PEFT on MedGemma
using QLoRA (Table 4). In addition to standard
prompting and RAG, Table 4 separates two fine-
tuning output formats:multi-task generation, where
the model produces the full JSON report with all
fields in one response, andsingle-task generation,
where the model is prompted to generate only one
task/field at a time. This distinction matters be-
cause single-task decoding can reduce output cou-
pling across fields, while multi-task decoding pre-
serves the intended end-to-end report setting.In general, QLoRA produces substantially higher
gainsthanpromptingorRAGinstructuredlabelpre-
diction, supporting an "escalating reliability ladder”
for classification: prompting provides a usable base-
line, RAG can stabilize performance in some tasks,
and fine-tuning produces the most consistent im-
provements when reliable label accuracy is required.
For example, BI-RADS accuracy improves from the
best prompting setting (0.2493) to 0.3190 with RAG,
and further to 0.6355 with multi-task QLoRA at 10
epochs; notably, single-task QLoRA achieves an even
higher BI-RADS accuracy of 0.7545 at 10 epochs.
Breast density prediction improves similarly from
0.7420 (best prompt) to 0.7862 (RAG) and to 0.8840
with multi-task QLoRA at 10 epochs. Calcification
prediction improves from 0.6641 (best prompt) to
0.7441 (RAG) and to 0.9341 with multi-task QLoRA
at 10 epochs. Mass accuracy increases from 0.5441
(RAG) to 0.7791 under multi-task QLoRA at 6
epochs, and rises further under single-task QLoRA
to 0.8740 at 10 epochs. Suspicion (derived from BI-
RADS)improvesfrom0.3982(bestprompt)to0.4134
(RAG) and to 0.7981 with multi-task QLoRA at 10
epochs.
Fine-tuning effects are not always monotonic with
additional epochs, and the optimal checkpoint can
differ across tasks and output formats. For instance,
in multi-task mode, BI-RADS peaks at 10 epochs
(0.6355) but drops at 15 epochs (0.5139), and mass
peaks at 6 epochs (0.7791) with only minor varia-
tion thereafter. Asymmetry highlights an additional
nuance: RAG reduces accuracy relative to the best
prompt (0.8545 to 0.7868), but multi-task QLoRA
9

Table 4: Performance comparison of MedGemma model
Task Evaluation Parameter Zeroshot Fewshot CoT RAG-FewshotFinetune
Generate All Tasks Together Generate 1 Task at a Time
3 Epochs 6 Epochs 10 Epochs 15 Epochs 6 Epochs 10 Epochs
BI-RADSAccuracy 0.2218 0.1019 0.2493 0.319 0.3564 0.4698 0.6355 0.5139 0.49760.7545
Precision 0.2722 0.1515 0.1846 0.3663 0.3608 0.5734 0.6544 0.5444 0.4840.7356
Recall 0.2218 0.1019 0.2493 0.319 0.3564 0.4698 0.6355 0.5139 0.45910.7516
F1-score 0.1237 0.0376 0.1412 0.2698 0.3489 0.4186 0.6261 0.4911 0.41520.7404
Specificity 0.8006 0.8004 0.8059 0.8265 0.8353 0.8629 0.906 0.875 0.86120.9387
Breast DensityAccuracy 0.7173 0.5921 0.742 0.7862 0.7736 0.83140.8840.7922 0.5606 0.8262
Precision 0.6507 0.6818 0.6631 0.7472 0.7606 0.58260.89670.8771 0.6018 0.8831
Recall 0.7173 0.5921 0.742 0.7862 0.7736 0.84850.8840.7922 0.5606 0.8677
F1-score 0.6819 0.6061 0.69 0.7529 0.7654 0.61470.88940.8145 0.5172 0.873
Specificity 0.7754 0.7649 0.7667 0.7998 0.8224 0.94630.93120.9294 0.8764 0.9271
CalcificationAccuracy 0.4009 0.2055 0.6641 0.7441 0.8164 0.86230.93410.9186 0.8388 0.7621
Precision 0.7461 0.7725 0.7466 0.8162 0.8421 0.55340.93170.9179 0.9425 0.9257
Recall 0.4009 0.2055 0.6641 0.7441 0.8164 0.80780.93410.9186 0.9423 0.9209
F1-score 0.4535 0.1368 0.6973 0.7694 0.8266 0.65690.93130.9182 0.9384 0.9107
Specificity 0.5254 0.5107 0.5414 0.6879 0.73220.87290.8418 0.846 0.8388 0.7621
MassAccuracy 0.4945 0.4127 0.4914 0.5441 0.595 0.7791 0.7541 0.7586 0.84550.874
Precision 0.5272 0.6556 0.5305 0.6221 0.6173 0.6964 0.7682 0.7555 0.85740.8787
Recall 0.4945 0.4127 0.4914 0.5441 0.595 0.7446 0.7541 0.7586 0.85820.8773
F1-score 0.5016 0.2885 0.4976 0.5404 0.6007 0.7197 0.7571 0.7519 0.85760.8777
Specificity 0.4988 0.5211 0.5022 0.5845 0.5951 0.8003 0.7578 0.724 0.84550.874
AsymmetryAccuracy 0.8545 0.8259 0.8545 0.7868 0.7882 0.88230.86050.8423 0.8536 0.8536
Precision 0.7302 0.7359 0.7302 0.7713 0.7854 0.56310.82920.8042 0.7287 0.7287
Recall 0.8545 0.8259 0.8545 0.7868 0.7882 0.850.86050.8423 0.8536 0.8536
F1-score 0.7875 0.7759 0.7875 0.7787 0.7868 0.67750.82610.8152 0.7862 0.7862
Specificity 0.5 0.4884 0.5 0.5369 0.5675 0.88780.57220.5667 0.5 0.5
SuspicionAccuracy 0.3464 0.3291 0.3982 0.4134 0.6735 0.74950.79810.789 0.8536 0.7146
Precision 0.6325 0.7579 0.6464 0.5853 0.7126 0.88420.79390.7555 0.6307 0.7179
Recall 0.3464 0.3291 0.3982 0.4134 0.6735 0.72230.79810.789 0.6173 0.6868
F1-score 0.2277 0.1771 0.3369 0.3271 0.6836 0.79510.78650.7519 0.6179 0.6874
Specificity 0.5111 0.5076 0.5347 0.5174 0.6741 0.80560.72630.724 0.8061 0.8478
recovers and peaks at 0.8823 at 6 epochs; in con-
trast, single-task asymmetry accuracy remains the
same (0.8536 for both 6 and 10 epochs), suggest-
ing that single-task fine-tuning may be ineffective for
certain labels in this setup. Taken together, these
results indicate that QLoRA generally increases reli-
ability for structured labels, but optimal training du-
ration and even the preferred output format (multi-
task vs single-task) are task-dependent and may re-
quire checkpoint selection or early stopping.
Summary of macro F1 gains from RAG and fine-
tuning. Table5summarizestheF1gainsfromadding
RAG and QLoRA fine-tuning for theMedGemma
model on the VinDr-Mammo dataset. We use the
macro F1-score, which better reflects minority-class
performance in imbalanced settings. Best Prompt
is the maximum over zero-shot, few-shot, and CoT
prompting. Best RAG is the maximum over the
corresponding retrieval-augmented settings. Best
QLoRA selects the best-performing fine-tuned check-
point across both output formats, multi-task full-
JSON generation, and single-task one-field genera-
tion, because the optimal format/checkpoint can be
task-dependent. The negative gains (asymmetry and
suspicion) indicate that retrieval can occasionally in-
troduce context mismatches, whereas QLoRA yields
the strongest and most consistent gains across tasks.
Only the F1-score is compared here, as it is the har-
monic mean of precision and recall [68, 69].RQ2: When does lightweight fine-tuning
outperform prompting/RAG for classifi-
cation?
RQ2Findings:Lightweight fine-tuning
(QLoRA) outperforms prompting/RAG most
strongly for hard structured classification.
Overall, the evidence supports a practical split:
prompting/RAG for strong report drafting and
grounding, and fine-tuning when dependable
label accuracy is required.
To compare to existing approaches, we use Mam-
moWise to compare our fine-tunedMedGemma
model against reported SOTA results in the litera-
ture. Because papers often report results across dif-
ferent task subsets and with diverse metrics, a one-
to-one comparison for every task is not always possi-
ble. Table6summarizeskeycomparisonsofaccuracy,
where available.
On VinDr-Mammo,PubMedCLIPachieves 0.5325
accuracy for BI-RADS classification, while our fine-
tunedMedGemmaachieves 0.7545 with single-task
training, an improvement on the same dataset and
metric. For breast-density classification,LLaVA-
MammoandLLaVA-MultiMammoreport accuracies
of 0.766 and 0.806, respectively; our model reaches
0.884. For mass and calcification detection, our
fine-tuned model (0.8740) outperformsMammoCLIP
(0.8) on mass classification but falls short on calcifi-
10

Table 5: Gain summary of adding RAG and fine-tuning for MedGemma on VinDr-Mammo (macro F1-score).
Task Best Prompt Best RAG∆(RAG–Prompt) Best QLoRA∆(QLoRA–RAG)
BI-RADS 0.1412 0.2698+0.1286 0.7404+0.4706
Density 0.6900 0.7529+0.0629 0.8894+0.1365
Calcification 0.6973 0.7694+0.0721 0.9384+0.1690
Mass 0.5016 0.5404+0.0388 0.8777+0.3373
Asymmetry 0.7875 0.7787−0.0088 0.8261+0.0474
Suspicion 0.3369 0.3271−0.0098 0.7951+0.4680
Table 6: Comparison of classification accuracy on several tasks ofMammoWisemodels with other SOTA
Task Our Work MammoCLIP LLaVA-Mammo LLaVA-MultiMammo PubMedCLIP
BI-RADS0.7545– – – 0.5325
Breast Density0.8840.88 0.766 0.806 –
Mass0.87400.8 – – –
Calcification 0.93410.98– – –
cation (0.9341 vs. 0.98).
For report generation, we found no peer-
reviewed mammography papers that report end-to-
end BERTScore and ROUGE-L metrics for complete
narrative mammogram reports, making direct exter-
nalcomparisonimpossible. MostcontemporaryVLM
work focuses on classification, localization, or visual
question answering rather than full, structured re-
porting. Nonetheless,MedGemmaachieves strong
generation metrics and produces clinically structured
JSON reports, offering capabilities that go beyond
pure classification.
Overall,MammoWisedemonstrates that a multi-
model pipeline powered by open-source VLMs can
be used to effectively iterate over models and to con-
vergeonamodelthatmatchesorexceedsSOTAbase-
lines on key classification tasks, while also enabling
high-quality report generation that has been under-
explored in prior work.
Our study has several limitations, mainly due to
computational and scope constraints. First, we did
not explore more advanced prompting strategies such
as Tree-of-Thought [70], or Reason+Act prompting
[71]. Second, wetestedonlythreeopen-sourceVLMs;
other models may offer different trade-offs in per-
formance and efficiency. Third, we did not fine-
tunelargermodelsthatmightfurtherimproveperfor-
mance but require more powerful hardware. Fourth,
we did not investigate hybrid architectures that com-
bine specialist mammography encoders with general-
purpose decoders. Fifth, because only two datasets
were used, there may be a domain shift (e.g., scale,
label noise), leading to different results between the
VinDr-Mammo and DMID datasets. Additionally,
our QLoRA fine-tuning uses a class-rebalanced sub-set (Table 1) constructed via downsampling and aug-
mentation, which differs from the original test dis-
tribution. This distribution shift can inflate macro-
averaged metrics while potentially harming calibra-
tion or operating characteristics under real-world
prevalence. We therefore interpret fine-tuning gains
primarily as evidence of improvedtask reliability un-
der balanced emphasis, rather than a definitive state-
ment about screening-time calibration. Finally, we
evaluated only single-exam images. Thus, extend-
ing MammoWise to longitudinal mammograms and
richer multimodal clinical data remains an important
direction for future work.
5 Conclusion
In this study, we presentMammoWise, a novel multi-
model pipeline that combines open-source medical
VLMs, tailored prompting techniques, and retrieval-
augmented generation to produce structured mam-
mogram reports and perform key classification tasks.
Our experiments showed that while base prompt-
ing strategies yield variable classification perfor-
mance, they already support strong report gener-
ation, particularly when combined with few-shot
prompting. RAG-based prompting further im-
proves text-generation quality and provides context-
driven grounding. Parameter-efficient fine-tuning of
MedGemmasignificantly increases classification per-
formance, especially for BI-RADS, breast density,
and calcification, demonstrating that local, resource-
aware adaptation can close much of the gap to spe-
cialized SOTA models.
Overall,MammoWiseoffers a practical blueprint
11

for deploying local VLMs as flexible, privacy-
preserving assistants in breast cancer screening. By
decoupling the pipeline from any single model and
supporting prompting, RAG, and fine-tuning within
one framework, it creates a reusable platform for fu-
ture research and clinical prototyping in mammogra-
phy and beyond.
Data & Code Availability State-
ment
The code for this research and the pipeline tool are
available on GitHub
(https://github.com/RaiyanJahangir/MammoWise).
Declaration of Interests
Theauthorsdeclarethattheyhavenoknowncompet-
ing financial interests or personal relationships that
could have appeared to influence the work reported
in this paper.
References
[1] Joyce C Lashof, I Craig Henderson, and Sharyl J
Nass. Mammography and beyond: developing
technologiesfortheearlydetectionofbreastcan-
cer. 2001.
[2] Raiyan Jahangir and Vladimir Filkov. Mammo-
find: An llm-based multi-channel tool for recom-
mending public mammogram datasets. InInter-
national Conference on Software Engineering of
Emerging Technology, pages 446–463. Springer,
2025.
[3] Thusitha Mabotuwana, Christopher S Hall,
Vadiraj Hombal, Prashanth Pai, Usha Nandini
Raghavan, ShawnRegis, BradyMcKee, Sandeep
Dalal, ChristophWald, andMartinLGunn. Au-
tomated tracking of follow-up imaging recom-
mendations.American Journal of Roentgenol-
ogy, 212(6):1287–1294, 2019.
[4] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu,
Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, An-
drea Madotto, and Pascale Fung. Survey of hal-
lucination in natural language generation.ACM
computing surveys, 55(12):1–38, 2023.
[5] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and
Shijian Lu. Vision-language models for vision
tasks: A survey.IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2024.[6] Mohaimenul Azam Khan Raiaan, Md Sad-
dam Hossain Mukta, Kaniz Fatema, Nur Mo-
hammad Fahad, Sadman Sakib, Most Mar-
ufatul Jannat Mim, Jubaer Ahmad, Mo-
hammed Eunus Ali, and Sami Azam. A review
on large language models: Architectures, appli-
cations, taxonomies, open issues and challenges.
IEEE access, 12:26839–26874, 2024.
[7] Menglin Jia, Luming Tang, Bor-Chun Chen,
Claire Cardie, Serge Belongie, Bharath Hariha-
ran, and Ser-Nam Lim. Visual prompt tuning.
InEuropean Conference on Computer Vision,
pages 709–727. Springer, 2022.
[8] LouieGiray. Promptengineeringwithchatgpt: a
guide for academic writers.Annals of biomedical
engineering, 51(12):2629–2633, 2023.
[9] Raiyan Jahangir, Tanjim Sakib, Ramisha Baki,
andMdMushfiqueHossain. Acomparativeanal-
ysis of potato leaf disease classification with big
transfer (bit) and vision transformer (vit) mod-
els. In2023 IEEE 9th International Women in
Engineering (WIE) Conference on Electrical and
Computer Engineering (WIECON-ECE), pages
58–63. IEEE, 2023.
[10] Mohammad Shahjahan Majib, Md Mahbubur
Rahman, TM Shahriar Sazzad, Nafiz Imtiaz
Khan, and Samrat Kumar Dey. Vgg-scnet: A
vgg net-based deep learning framework for brain
tumor detection on mri images.IEEE Access,
9:116942–116952, 2021.
[11] Raphael Sexauer, Patryk Hejduk, Karol
Borkowski, Carlotta Ruppert, Thomas Weikert,
Sophie Dellas, and Noemi Schmidt. Diagnostic
accuracy of automated acr bi-rads breast density
classification using deep convolutional neural
networks.European Radiology, 33(7):4589–4596,
2023.
[12] Francisco S Marcondes, Adelino Gala, Renata
Magalhães, Fernando Perez de Britto, Dalila
Durães, andPauloNovais. Usingollama. InNat-
ural Language Analytics with Generative Large-
Language Models: A Practical Approach with
Ollama and Open-Source LLMs, pages 23–35.
Springer, 2025.
[13] Farhad Pourpanah, Moloud Abdar, Yuxuan
Luo, Xinlei Zhou, Ran Wang, Chee Peng Lim,
Xi-Zhao Wang, and QM Jonathan Wu. A re-
view of generalized zero-shot learning methods.
IEEE transactions on pattern analysis and ma-
chine intelligence, 45(4):4051–4070, 2022.
12

[14] Yaqing Wang, Quanming Yao, James T Kwok,
and Lionel M Ni. Generalizing from a few ex-
amples: A survey on few-shot learning.ACM
computing surveys (csur), 53(3):1–34, 2020.
[15] Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting
elicits reasoning in large language models.Ad-
vances in neural information processing systems,
35:24824–24837, 2022.
[16] Nandkishore Patidar, Sejal Mishra, Rahul Jain,
Dhiren Prajapati, Amit Solanki, Rajul Suthar,
Kavindra Patel, and Hiral Patel. Transparency
in ai decision making: A survey of explainable ai
methods and applications.Advances of Robotic
Technology, 2(1), 2024.
[17] Zihao Fu, Haoran Yang, Anthony Man-Cho So,
WaiLam,LidongBing,andNigelCollier. Onthe
effectiveness of parameter-efficient fine-tuning.
InProceedings of the AAAI conference on artifi-
cial intelligence, volume 37, pages 12799–12807,
2023.
[18] Hieu T Nguyen, Ha Q Nguyen, Hieu H Pham,
Khanh Lam, Linh T Le, Minh Dao, and Van Vu.
Vindr-mammo: Alarge-scalebenchmarkdataset
for computer-aided diagnosis in full-field digital
mammography.Scientific Data, 10(1):277, 2023.
[19] Parita Oza, Urvi Oza, Rajiv Oza, Paawan
Sharma, Samir Patel, Pankaj Kumar, and Bakul
Gohel. Digital mammography dataset for breast
cancer diagnosis research (dmid) with breast
mass segmentation analysis.Biomedical Engi-
neering Letters, 14(2):317–330, 2024.
[20] Andrew Sellergren, Sahar Kazemzadeh, Tiam
Jaroensri, Atilla Kiraly, Madeleine Traverse,
Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían
Hughes, Charles Lau, et al. Medgemma tech-
nical report.arXiv preprint arXiv:2507.05201,
2025.
[21] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto
Usuyama, Haotian Liu, Jianwei Yang, Tristan
Naumann, Hoifung Poon, and Jianfeng Gao.
Llava-med: Training a large language-and-vision
assistant for biomedicine in one day.Ad-
vances in Neural Information Processing Sys-
tems, 36:28541–28564, 2023.
[22] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, Sibo Song, Kai Dang,
Peng Wang, Shijie Wang, Jun Tang, et al.Qwen2. 5-vl technical report.arXiv preprint
arXiv:2502.13923, 2025.
[23] TimDettmers, ArtidoroPagnoni, AriHoltzman,
and Luke Zettlemoyer. Qlora: Efficient finetun-
ing of quantized llms.Advances in neural in-
formation processing systems, 36:10088–10115,
2023.
[24] Hana L Haver, Paul H Yi, Jean Jeudy, and
Manisha Bahl. Use of chatgpt to assign bi-
rads assessment categories to breast imaging
reports.American Journal of Roentgenology,
223(3):e2431093, 2024.
[25] Filippo Pesapane, Luca Nicosia, Anna Rotili,
Silvia Penco, Valeria Dominelli, Chiara Trentin,
Federica Ferrari, Giulia Signorelli, Serena Car-
riero, and Enrico Cassano. A preliminary inves-
tigation into the potential, pitfalls, and limita-
tionsoflargelanguagemodelsformammography
interpretation.Discover Oncology, 16(1):233,
2025.
[26] Luís Vinícius de Moura, Rafaela Ravazio, Chris-
tian Mattjie, Lucas Silveira Kupssinskü, Carla
Maria Dal Sasso Freitas, and Rodrigo C Barros.
Unlocking the potential of vision-language mod-
els for mammography analysis. In2024 IEEE
International Symposium on Biomedical Imag-
ing (ISBI), pages 1–4. IEEE, 2024.
[27] Yi Li, Hualiang Wang, Yiqun Duan, Jiheng
Zhang, and Xiaomeng Li. A closer look at the
explainability of contrastive language-image pre-
training.Pattern Recognition, 162:111409, 2025.
[28] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Han-
wen Xu, Jaspreet Bagga, Robert Tinn, Sam
Preston, Rajesh Rao, Mu Wei, Naveen Valluri,
et al. Biomedclip: a multimodal biomedical
foundation model pretrained from fifteen mil-
lion scientific image-text pairs.arXiv preprint
arXiv:2303.00915, 2023.
[29] Sedigheh Eslami, Christoph Meinel, and Gerard
De Melo. Pubmedclip: How much does clip ben-
efit visual question answering in the medical do-
main? InFindings of the Association for Com-
putational Linguistics: EACL 2023, pages 1181–
1193, 2023.
[30] Aisha Urooj Khan, John Garrett, Tyler Brad-
shaw, Lonie Salkowski, Jiwoong Jeong, Amara
Tariq, and Imon Banerjee. Knowledge-grounded
adaptation strategy for vision-language models:
13

Building a unique case-set for screening mam-
mograms for residents training. InInternational
Conference on Medical Image Computing and
Computer-Assisted Intervention, pages 587–598.
Springer, 2024.
[31] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal,
and Jimeng Sun. Medclip: Contrastive learning
from unpaired medical images and text. InPro-
ceedings of the Conference on Empirical Methods
in Natural Language Processing. Conference on
Empirical Methods in Natural Language Process-
ing, volume 2022, page 3876, 2022.
[32] Xuxin Chen, Jingchu Chen, Xiaoqian Chen,
Judy Gichoya, Hari Trivedi, and Xiaofeng Yang.
Llava-mammo: adapting llava for interactive
and interpretable breast cancer assessment. In
Medical Imaging 2025: Imaging Informatics,
volume 13411, pages 80–88. SPIE, 2025.
[33] Haotian Liu, Chunyuan Li, Qingyang Wu, and
Yong Jae Lee. Visual instruction tuning.Ad-
vances in neural information processing systems,
36:34892–34916, 2023.
[34] Xuxin Chen, Jingchu Chen, Xiaoqian Chen,
Judy Gichoya, Hari Trivedi, and Xiaofeng Yang.
Llava-multimammo: adapting vision-language
models for explainable and comprehensive mul-
tiview mammogram analysis in breast cancer as-
sessment. InMedical Imaging 2025: Computer-
Aided Diagnosis, volume 13407, pages 165–173.
SPIE, 2025.
[35] Shantanu Ghosh, Clare B Poynton, Shyam
Visweswaran, and Kayhan Batmanghelich.
Mammo-clip: A vision language foundation
model to enhance data efficiency and robustness
in mammography. InInternational Conference
on Medical Image Computing and Computer-
Assisted Intervention, pages 632–642. Springer,
2024.
[36] Syed Taha Yeasin Ramadan, Tanjim Sakib,
Md Ahsan Rahat, Shakil Mosharrof,
Fatin Ishrak Rakin, and Raiyan Jahangir.
Enhancing mango leaf disease classification:
vit, bit, and cnn-based models evaluated on
cyclegan-augmented data. In2023 26th interna-
tional conference on computer and information
technology (ICCIT), pages 1–6. IEEE, 2023.
[37] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim,
Donghyeon Kim, Sunkyu Kim, Chan Ho So, and
Jaewoo Kang. Biobert: a pre-trained biomedi-
callanguagerepresentationmodelforbiomedicaltext mining.Bioinformatics, 36(4):1234–1240,
2020.
[38] Kshitiz Jain, Aditya Bansal, Krithika Rangara-
jan, and Chetan Arora. Mmbcd: Multimodal
breast cancer detection from mammograms with
clinical history. InInternational Conference
on Medical Image Computing and Computer-
Assisted Intervention, pages 144–154. Springer,
2024.
[39] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei
Du, Mandar Joshi, Danqi Chen, Omer Levy,
Mike Lewis, Luke Zettlemoyer, and Veselin
Stoyanov. Roberta: A robustly optimized
bert pretraining approach.arXiv preprint
arXiv:1907.11692, 2019.
[40] Zhenjie Cao, Zhuo Deng, Jie Ma, Jintao Hu, and
Lan Ma. Mammovlm: A generative large vision-
language model for mammography-related di-
agnostic assistance.Information Fusion, page
102998, 2025.
[41] S Sasikala, M Bharathi, M Ezhilarasi, and
S Arunkumar. Breast cancer detection based
on medio-lateral obliqueview and cranio-caudal
view mammograms: an overview. In2019
IEEE 10th International Conference on Aware-
ness Science and Technology (iCAST), pages 1–
6. IEEE, 2019.
[42] Niketa Chotai and Supriya Kulkarni.Breast
Imaging Essentials. Springer, 2020.
[43] Raiyan Jahangir, Tanjim Sakib, Riasat Haque,
and Mahedi Kamal. A performance analysis
of brain tumor classification from mri images
using vision transformers and cnn-based classi-
fiers. In2023 26th International Conference on
Computer and Information Technology (ICCIT),
pages 1–6. IEEE, 2023.
[44] K Lavanya, K Aravind, Vishal Dixit, et al. Ad-
vanced video transcription and summarization
a synergy of langchain, language models, and
vectordb with mozilla deep speech. In2024
Second International Conference on Emerging
Trends in Information Technology and Engineer-
ing (ICETITE), pages 1–9. IEEE, 2024.
[45] Gemma Team, Aishwarya Kamath, Johan Fer-
ret, Shreya Pathak, Nino Vieillard, Ramona
Merhej, Sarah Perrin, Tatiana Matejovicova,
Alexandre Ramé, Morgane Rivière, et al.
Gemma 3 technical report.arXiv preprint
arXiv:2503.19786, 2025.
14

[46] Hugo Touvron, Thibaut Lavril, Gautier Izacard,
Xavier Martinet, Marie-Anne Lachaux, Timo-
thée Lacroix, Baptiste Rozière, Naman Goyal,
Eric Hambro, Faisal Azhar, et al. Llama: Open
and efficient foundation language models.arXiv
preprint arXiv:2302.13971, 2023.
[47] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang,
Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing
Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl:
Enhancing vision-language model’s perception
of the world at any resolution.arXiv preprint
arXiv:2409.12191, 2024.
[48] J Diego Zamfirescu-Pereira, Richmond Y Wong,
Bjoern Hartmann, and Qian Yang. Why johnny
can’t prompt: how non-ai experts try (and fail)
todesignllmprompts. InProceedings of the 2023
CHI conference on human factors in computing
systems, pages 1–21, 2023.
[49] Nafiz Imtiaz Khan, Kylie Cleland, Vladimir
Filkov, and Roger Eric Goldman. Intelligent
documentation in medical education: Can ai
replace manual case logging?arXiv preprint
arXiv:2601.12648, 2026.
[50] Matthew Renze. The effect of sampling temper-
ature on problem solving in large language mod-
els. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2024, pages 7346–
7356, 2024.
[51] Vipula Rawte, Amit Sheth, and Amitava Das. A
survey of hallucination in large foundation mod-
els.arXiv preprint arXiv:2309.05922, 2023.
[52] Julia Amann, Alessandro Blasimme, Effy
Vayena, Dietmar Frey, Vince I Madai, and Pre-
cise4Q Consortium. Explainability for artificial
intelligence in healthcare: a multidisciplinary
perspective.BMC medical informatics and deci-
sion making, 20(1):310, 2020.
[53] Nafiz Imtiaz Khan and Vladimir Filkov. Lever-
aging language models to discover evidence-
based actions for oss sustainability.arXiv
preprint arXiv:2602.11746, 2026.
[54] Zeerak Babar, Nafiz Imtiaz Khan, Muhammad
Hassnain, and Vladimir Filkov. Open-source
llms for technical q&a: Lessons from stackex-
change. InInternational Conference on Software
Engineering of Emerging Technology, pages 615–
626. Springer, 2025.[55] Nafiz Imtiaz Khan and Vladimir Filkov. Ev-
idencebot: a privacy-preserving, customizable
rag-based tool for enhancing large language
model interactions. InProceedings of the 33rd
ACM International Conference on the Founda-
tions of Software Engineering, pages 1188–1192,
2025.
[56] Munib Mesinovic, Peter Watkinson, and Tingt-
ing Zhu. Explainability in the age of large lan-
guage models for healthcare.Communications
Engineering, 4(1):128, 2025.
[57] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal.
Vl-adapter: Parameter-efficienttransferlearning
for vision-and-language tasks. InProceedings of
the IEEE/CVF conference on computer vision
and pattern recognition, pages 5227–5237, 2022.
[58] Evgin Goceri. Medical image data aug-
mentation: techniques, comparisons and in-
terpretations.Artificial intelligence review,
56(11):12561–12605, 2023.
[59] Linda Blahová, Jozef Kostoln` y, and Ivan Cim-
rák. Neural network-based mammography anal-
ysis: Augmentation techniques for enhanced
cancer diagnosis—a review.Bioengineering,
12(3):232, 2025.
[60] Parita Oza, Paawan Sharma, Samir Patel, Fes-
tus Adedoyin, and Alessandro Bruno. Image
augmentation techniques for mammogram anal-
ysis.Journal of Imaging, 8(5):141, 2022.
[61] Yuliana Jiménez-Gaona, Diana Carrión-
Figueroa, Vasudevan Lakshminarayanan, and
María José Rodríguez-Álvarez. Gan-based data
augmentation to improve breast ultrasound and
mammography mass classification.Biomedical
Signal Processing and Control, 94:106255, 2024.
[62] Shashank Mohan Jain. Hugging face. InIntro-
duction to transformers for NLP: With the hug-
ging face library and models to solve problems,
pages 51–67. Springer, 2022.
[63] Edward J Hu, Yelong Shen, Phillip Wallis,
Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-
rankadaptationoflargelanguagemodels.ICLR,
1(2):3, 2022.
[64] Raiyan Jahangir, Tanjim Sakib, Md Fahmid-
Ul-Alam Juboraj, Shejuti Binte Feroz, and
MdMunkaserIslamSharar. Braintumorclassifi-
cationonmriimageswithbigtransferandvision
15

transformer: Comparative study. In2023 IEEE
9th International Women in Engineering (WIE)
Conference on Electrical and Computer Engi-
neering (WIECON-ECE), pages 46–51. IEEE,
2023.
[65] Nafiz Imtiaz Khan, Tahasin Mahmud, Muham-
mad Nazrul Islam, and Sumaiya Nuha Musta-
fina. Prediction of cesarean childbirth using en-
semble machine learning methods. InProceed-
ings of the 22nd international conference on in-
formation integration and web-based applications
& services, pages 331–339, 2020.
[66] Tianyi Zhang, Varsha Kishore, Felix Wu, Kil-
ian Q Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert.arXiv
preprint arXiv:1904.09675, 2019.
[67] Chin-Yew Lin. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81, 2004.
[68] Raiyan Jahangir, Muhammad Nazrul Islam,
Md Shofiqul Islam, and Md Motaharul Islam.
Ecg-based heart arrhythmia classification using
feature engineering and a hybrid stacked ma-
chine learning.BMC Cardiovascular Disorders,
25(1):260, 2025.
[69] Raiyan Jahangir, Nasif Shahriar Mohim,
Nafiz Imtiaz Khan, Md Akhtaruzzaman, and
Muhammad Nazrul Islam. Proposing novel
recurrent neural network architectures for infant
cry detection in domestic context. In2023
IEEE 11th Region 10 Humanitarian Technology
Conference (R10-HTC), pages 7–12. IEEE,
2023.
[70] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak
Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate prob-
lem solving with large language models.Ad-
vances in neural information processing systems,
36:11809–11822, 2023.
[71] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du,
Izhak Shafran, Karthik Narasimhan, and Yuan
Cao. React: Synergizing reasoning and acting
in language models. InInternational Conference
on Learning Representations (ICLR), 2023.
16

Appendix
Zero Shot Prompt:
You are a board-certified breast radiologist with lots of experience in interpreting
,→screening and diagnostic mammograms. You are meticulous, up to date with the latest BI
,→-RADS guidelines, and always provide clear, concise, and clinically actionable reports
,→.
I am providing you with a mammogram image. The image has all 4 breast views of a patient
,→shown together. The upper two views are the craniocaudal (CC) views of each breast,
,→right and left, and the lower two views are the mediolateral oblique (MLO) views of
,→each breast, right and left. Your task is to analyze the image and provide a
,→structured report in JSON format.
For analyzing the image, at first glance, you should identify the breast density using
,→the ACR classification, which includes:
- ACR A: Almost entirely fatty
- ACR B: Scattered fibroglandular densities
- ACR C: Heterogeneously dense
- ACR D: Extremely dense
Then, you should write the abnormalities you notice from the images in 1 sentence.
,→Mention in which view the findings are present, and say "Healthy Breast. No Findings"
,→for normal breasts. Findings include Mass, Suspicious Calcification, Architectural
,→Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry, Suspicious Lymph Nodes,
,→Nipple Retraction, Skin Retraction, Skin Thickening. There may be multiple findings in
,→a single image.
Assign a BI-RADS category based on the findings:
- BI-RADS 1: Negative (no abnormalities)
- BI-RADS 2: Benign (no suspicion of cancer)
- BI-RADS 3: Probably benign (short-term follow-up recommended)
- BI-RADS 4: Suspicious abnormality (biopsy needed)
- BI-RADS 5: Highly suggestive of malignancy (high probability of cancer)
Finally, indicate whether the image is healthy, benign, or suspicious.
Here is the JSON format you should follow for your response:
{
"breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
"findings": "<Summary of any abnormalities as described above in one sentence>",
"BI-RADS": "<1|2|3|4|5> followed by a brief description of the BI-RADS category>",
"suspicion": "<healthy|benign|suspicious>"
}
Few-shot Prompt:
<Prompt from Zero-Shot> +
Here are some examples of doctor-annotated reports to guide you:
Example 1:
{
"image_id": "image_file_path_1",
17

"breast_density": "Density C - Heterogeneously Dense. More of the breast is made of
,→dense glandular and fibrous tissue. This can make it hard to see small masses in or
,→around the dense tissue, which also appear as white areas.",
"BI-RADS": "BI-RADS 1 - Negative. Healthy Breast.",
"findings": "Healthy Breast. No Findings",
"suspicion": "healthy"
}
Example 2:
{
"image_id": "image_file_path_2",
"breast_density": "Density C - Heterogeneously Dense. More of the breast is made of
,→dense glandular and fibrous tissue. This can make it hard to see small masses in or
,→around the dense tissue, which also appear as white areas.",
"BI-RADS": "BI-RADS 2 - Benign (non-cancerous) finding",
"findings": "Healthy Breast. No Findings",
"suspicion": "healthy"
}
....
Chain-Of-Thought Prompt:
<Prompt from Zero-Shot> +
Step 1: Identify breast density
Let’s check the breast tissue density. Breast tissue is usually composed of fatty tissue,
,→which appears lighter, and fibro-glandular tissue, which appears darker.
If there is a presence of small white wisps of fibro-glandular strands near the nipple (
,→right and left end in the CC view, lower right and lower left end in the MLO view)
,→against a background of fat, that only covers a few places of the breast. The density
,→is "DENSITY A \- Almost all fatty tissue." Such images are easier to use for
,→diagnosing abnormalities.
If a few scattered pockets of white fibro-glandular tissue start near the nipple and go
,→more outwards, but most of the breast remains of lighter white fat. Those isolated
,→denser areas occupy roughly one-quarter to one-half of the breast, then the density is
,→"DENSITY B - Scattered areas of dense glandular and fibrous tissue". Such images are
,→harder to diagnose any abnormalities.
If there are large swaths of the breast composed of dense, white tissue that start near
,→the nipple, are heterogeneously distributed and patchy, and cover over half the breast
,→, interspersed with lighter fatty areas. Then the density is "DENSITY C \-
,→Heterogeneously Dense. More of the breast is made of dense glandular and fibrous
,→tissue." So, some small findings could be masked and difficult to diagnose.
If the entire breast appears almost uniformly dense white, with very little fat visible,
,→then the density is "DENSITY D - Extremely Dense. Hard to see masses or other
,→findings that may appear as white areas on the mammogram." This "extremely dense"
,→pattern of fibroglandular tissue makes it most challenging to detect subtle lesions
,→and diagnose.
Step 2: Now it’s time to analyze the breast and identify any abnormalities.
18

Step 2a: Finding masses or lesions
Let’s analyze all views one by one, starting from the right CC view, right MLO view, left
,→CC view, and left MLO view. I scan from top to bottom in a breast view. A mass or
,→lesion should appear as a discrete area of dense white opacity, shaped either round,
,→oval, or irregular, that is visible from both the CC and MLO views of the breast. If
,→it is not visible in one, it may be an overlap or summation artifact. If the mass is
,→well-defined, round, or oval, it is usually benign and should be assigned a BI-RADS 3.
,→If the shape is indistinct and obscured, it raises suspicion of malignancy and should
,→be assigned a BI-RADS 4. If the shape is spiculated or has radiating lines, it is
,→highly suspicious for malignancy and should be assigned a BI-RADS 5. If nothing is
,→visible, no mass is there. Otherwise, I should specify that mass is found in which
,→view of which breast.
Step 2b: Finding calcification
Again, let’s analyze each view of the breast from top to bottom. Calcifications appear as
,→tiny white spots on mammograms. Locate all areas of increased density (tiny white
,→specks), punctate, micronodular bright spots on the CC and MLO views. Next, count how
,→many specks are clustered within approximately 1 cubic cm of tissue. Then the
,→morphology of those specks is to be observed. That is, if the specks are round and
,→uniform, or pleomorphic, or have fine-linear branching. Then their distribution has to
,→be ascertained. Are they scattered or grouped in clusters or segmented along ductal
,→anatomy? Then compare with the contralateral breast to assess the asymmetry.
Finally, if the specks are found to be round, "milk-of-calcium," or vascular, they are
,→benign. They are to be assigned BI-RADS 3. If they are pleomorphic or clustered, then
,→assign BI-RADS 4. If the specks are fine-linear or branching, then assign BI-RADS 5.
Step 2c: Finding asymmetry
For determining asymmetry, first line up the CC views of the right and left breasts (and
,→then the MLOs). The target will be to look for areas of density or architectural
,→patterns that appear on one side but not the other. If such a pattern is found in one
,→breast, note its exact position (quadrant, depth) and check the same spot on the other
,→breast and the other views.
If the asymmetry is visible upon only one projection (either CC or MLO), then it is an
,→asymmetry. An additional image might be taken from a different angle to ascertain. If
,→the asymmetry is persistent but benign, such as a fat lobule, assign BI-RADS 3.
,→However, if suspicious margins/architecture are observed, assign BI-RADS 4.
If the asymmetry is observed in two projections and lacks convex borders or conspicuity
,→of a true mass, and the area covered by the asymmetry is less than or equal to one
,→quadrant in size, it could be a focal asymmetry. It could be assigned BI-RADS 3 if no
,→other abnormalities are available. Based on the presence of mass and suspicious
,→calcification, and other abnormalities, BI-RADS 4 or BI-RADS 5 could be assigned.
If the asymmetry is observed in two projections and spans more than one quadrant, it is a
,→global asymmetry. It could be assigned BI-RADS 3 if no other abnormalities are
,→available. Based on the presence of mass and suspicious calcification, and other
,→abnormalities, BI-RADS 4 or BI-RADS 5 could be assigned.
Step 2d: Finding architectural distortion
Architectural distortion is one of the more subtle but highly important mammographic
,→signs of malignancy. It refers to the disruption of the normal fibro-glandular
19

,→framework without a discrete mass. To detect them, first identify any suspicious
,→patterns like spiculations or radiating lines with no central mass or focal retraction
,→or pulling of Cooper’s ligaments toward a point, or a distortion of parenchymal lines
,→, that is, tissue planes appear bent, kinked, or tethered. If the distortion is stable
,→, assign BI-RADS 3. If the distortion is mild, with sparse striations or dense, thick
,→spicules, assign BI-RADS 4. Else if it has a classic starburst pattern, assign BI-RADS
,→5.
Step 2e: Check for suspicious lymph nodes
The axillary lymph nodes lie in the upper outer quadrant of each breast. It is normally
,→visible in the MLO view. In the given image, it is in the upper central of the MLO
,→view. If nodes are visible and reniform (bean-shaped) with a long axis parallel to the
,→skin, then it is benign. Assign BI-RADS 2. If mild, diffuse cortical thickening
,→without other worrisome features is observed, it is likely benign. Assign BI-RADS 3.
,→If it is round-shaped with a focal cortical bulge or loss of hilum, it is highly
,→suspicious of malignancy. Assign BI-RADS 4. If the cortex is markedly thickened with
,→spiculated margins or clustered microcalcifications, biopsy is urgent. Assign BI-RADS
,→5.
Step 2f: Finding other observations
First, check the nipple retraction. It is an inward pulling or inversion of the nipple.
,→At first, in the CC view, look at the nipple border. Normally, it projects forward as
,→a small convex contour.
Retraction shows as an inward indentation or loss of the convex silhouette. Then check
,→the MLO view and confirm that the nipple tip lies posterior (toward the chest wall)
,→relative to the skin line rather than anterior.
Then check skin retraction. It is a focal pulling in of the skin surface, often from an
,→underlying desmoplastic (fibrotic) reaction. Search for a subtle focal area where skin
,→thickness suddenly narrows or a dimple forms, often overlying a spiculated mass. Also
,→, look for converging parenchymal lines (ligamentous strands) that course from the
,→lesion toward the skin. Then, verify if it is on two projections.
Then check skin thickening. It is a diffuse or focal increase in the thickness of the
,→subcutaneous fat layer, which can be due to edema, an inflammatory cancer (e.g., Paget
,→’s disease, inflammatory carcinoma), or prior surgery/radiation. At first, identify
,→the skin line on both CC and MLO views. Normally, the skin appears as a thin
,→radiopaque line ~1.5 to 2mm thick. If it is a Diffuse thickening >2.5mm (some texts
,→use >3mm) across multiple quadrants, it is abnormal. If it is a focal thickening >3mm
,→in a localized area, it warrants further work-up. Bilateral and symmetric thickening
,→often reflects systemic edema (e.g., heart failure). Unilateral or focal suggests
,→underlying inflammatory malignancy or local process. Trabecular thickening (edema) in
,→the subcutaneous fat or Cooper’s ligaments. Usually, such features do not indicate
,→malignancy and may be considered BI-RADS 1 or BI-RADS 2 in the presence of multiple
,→observations. If observed with other abnormalities, they may be assigned a higher BI-
,→RADS score.
Step 2g: If nothing is found,
If nothing is found, write "Healthy Breast. No Findings" and assign BI-RADS 1.
Step 2h: Write the findings
Write which abnormalities are found in which view of which breast. If there are multiple
,→abnormalities, write them all.
20

Step 3: Assign a BI-RADS Score
Based on the above explanation of findings, assign a BI-RADS score.
- BI-RADS 0: Incomplete (needs additional imaging)
- BI-RADS 1: Negative (no abnormalities)
- BI-RADS 2: Benign (no suspicion of cancer)
- BI-RADS 3: Probably benign (short-term follow-up recommended)
- BI-RADS 4: Suspicious abnormality (biopsy needed)
- BI-RADS 5: Highly suggestive of malignancy (high probability of cancer)
- BI-RADS 6: Known malignancy (biopsy-proven cancer)
Step 4: Finally, indicate whether the case is healthy, benign, or suspicious (radiologic
,→suspicion, not biopsy-confirmed cancer).
Step 5: Here is the JSON format you should follow for your response:
{
"breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
"findings": "<Summary of any abnormalities as described above in one sentence>",
"BI-RADS": "<1|2|3|4|5> followed by a brief description of the BI-RADS category>",
"suspicion": "<healthy|benign|suspicious>"
}
21

Table 7: Classification performance of different models on VinDr-Mammo dataset on zero-shot
Task Evaluation ParameterModels
MedGemma Qwen2.5VL LLAVA-Med
BI-RADSAccuracy 0.2218 0.1957 0.2273
Precision 0.2722 0.0855 0.0517
Recall 0.2218 0.1957 0.2273
F1-score 0.1237 0.0913 0.0842
Specificity 0.8006 0.8008 0.8
Breast DensityAccuracy 0.7173 0.4336 0.1041
Precision 0.6507 0.6223 0.7925
Recall 0.7173 0.4336 0.1041
F1-score 0.6819 0.5105 0.0211
Specificity 0.7754 0.7587 0.7503
CalcificationAccuracy 0.4009 0.5282 0.8368
Precision 0.7461 0.7528 0.7003
Recall 0.4009 0.5282 0.8368
F1-score 0.4535 0.5887 0.7625
Specificity 0.5254 0.5477 0.5
MassAccuracy 0.4945 0.6186 0.6191
Precision 0.5272 0.3832 0.3833
Recall 0.4945 0.6186 0.6191
F1-score 0.5016 0.4732 0.4734
Specificity 0.4988 0.4996 0.5
AsymmetryAccuracy 0.8545 0.5095 0.8915
Precision 0.7302 0.7587 0
Recall 0.8545 0.5095 0
F1-score 0.7875 0.5802 0
Specificity 0.5 0.5147 1
SuspicionAccuracy 0.3464 0.6396 0.6818
Precision 0.6325 0 0.4649
Recall 0.3464 0 0.6818
F1-score 0.2277 0 0.5528
Specificity 0.5111 1 0.5
22

Table 8: Classification performance of different models on the VinDr-Mammo dataset on few-shot
Task Evaluation ParameterModels
MedGemma Qwen2.5VL LLAVA-Med
BI-RADSAccuracy 0.1019 0.2582 0.2909
Precision 0.1515 0.4121 0.3493
Recall 0.1019 0.2582 0.2909
F1-score 0.0376 0.1876 0.2217
Specificity 0.8004 0.808 0.8165
Breast DensityAccuracy 0.5921 0.7832 0.1141
Precision 0.6818 0.7153 0.6446
Recall 0.5921 0.7832 0.1141
F1-score 0.6061 0.6893 0.046
Specificity 0.7649 0.7516 0.7506
CalcificationAccuracy 0.2055 0.8391 0.8377
Precision 0.7725 0.865 0.8235
Recall 0.2055 0.8391 0.8377
F1-score 0.1368 0.7679 0.7655
Specificity 0.5107 0.507 0.5039
MassAccuracy 0.4127 0.6205 0.6195
Precision 0.6556 0.7647 0.7644
Recall 0.4127 0.6205 0.6195
F1-score 0.2885 0.4766 0.4745
Specificity 0.5211 0.5018 0.5006
AsymmetryAccuracy 0.8259 0.7159 0.7514
Precision 0.7359 0.7608 0.761
Recall 0.8259 0.7159 0.7514
F1-score 0.7759 0.7362 0.7561
Specificity 0.4884 0.5213 0.52
SuspicionAccuracy 0.3291 0.6841 0.5914
Precision 0.7579 0.7841 0.5881
Recall 0.3291 0.6841 0.5914
F1-score 0.1771 0.5581 0.5897
Specificity 0.5076 0.5036 0.5251
23

Table 9: Classification performance of different models on VinDr-Mammo dataset on chain-of-thought
Task Evaluation ParameterModels
MedGemma Qwen2.5VL LLAVA-Med
BI-RADSAccuracy 0.2493 0.2291 0.2488
Precision 0.1846 0.0526 0.1898
Recall 0.2493 0.2291 0.2488
F1-score 0.1412 0.0855 0.2059
Specificity 0.8059 0.8 0.8046
Breast DensityAccuracy 0.742 0.0168 0.0386
Precision 0.6631 0.7923 0.0086
Recall 0.742 0.0168 0.0386
F1-score 0.69 0.0155 0.014
Specificity 0.7667 0.7506 0.7475
CalcificationAccuracy 0.6641 0.1695 0.525
Precision 0.7466 0.689 0.7164
Recall 0.6641 0.1695 0.525
F1-score 0.6973 0.0626 0.587
Specificity 0.5414 0.4982 0.4796
MassAccuracy 0.4914 0.6191 0.6218
Precision 0.5305 0.3833 0.6273
Recall 0.4914 0.6191 0.6218
F1-score 0.4976 0.4734 0.486
Specificity 0.5022 0.5 0.5054
AsymmetryAccuracy 0.8545 0.1505 0.5723
Precision 0.7302 0.6362 0.7611
Recall 0.8545 0.1505 0.5723
F1-score 0.7875 0.0526 0.635
Specificity 0.5 0.4938 0.5215
SuspicionAccuracy 0.3982 0.6832 0.3279
Precision 0.6464 0.4668 0.4435
Recall 0.3982 0.6832 0.3279
F1-score 0.3369 0.5546 0.1629
Specificity 0.5347 0.5 0.4995
24

Table 10: Performance of different models on DMID dataset with zero-shot prompting
Task Evaluation Parameters MedGemma LLAVA-Med Qwen2.5VL
BI-RADSAccuracy 0.2878 0.2 0.2918
Precision 0.2861 0.0824 0.6562
Recall 0.2878 0.2 0.2918
F1-score 0.2602 0.1167 0.247
Specificity 0.8371 0.8 0.8268
BERTScore 0.8662 0.8322 0.8557
ROUGE-L 0.1766 0.1373 0.0492
Breast DensityAccuracy 0.2649 0.25 0.268
Precision 0.3077 0.0995 0.1424
Recall 0.2649 0.25 0.268
F1-score 0.2476 0.1424 0.1782
Specificity 0.7554 0.75 0.7585
BERTScore 0.8408 0.8408 0.8217
ROUGE-L 0.1984 0.2228 0.216
FindingsBERTScore 0.843 0.8293 0.8252
ROUGE-L 0.1437 0.0861 0.1298
Table 11: Performance of different models on DMID dataset with few-shot prompting
Task Evaluation Parameters MedGemma LLAVA-Med Qwen2.5VL
BI-RADSAccuracy 0.3531 0.3998 0.201
Precision 0.3717 0.8139 0.2483
Recall 0.3531 0.3998 0.201
F1-score 0.3392 0.3419 0.0797
Specificity 0.8542 0.8602 0.8005
BERTScore 0.9983 0.9508 0.9999
ROUGE-L 0.4906 0.0392 0.2431
Breast DensityAccuracy 0.3739 0.2226 0.2512
Precision 0.3685 0.3664 0.3423
Recall 0.3739 0.2226 0.2512
F1-score 0.3605 0.155 0.1373
Specificity 0.7877 0.7429 0.7508
BERTScore 0.8991 0.8093 0.8862
ROUGE-L 0.4939 0.1726 0.3036
FindingsBERTScore 0.8615 0.8141 0.8492
ROUGE-L 0.2706 0.1076 0.2303
25

Table 12: Performance of different models on DMID dataset with Chain-of-Thought prompting
Task Evaluation Parameters MedGemma LLAVA-Med Qwen2.5VL
BI-RADSAccuracy 0.2277 0.183 0.57
Precision 0.3108 0.336 0.65
Recall 0.2277 0.183 0.57
F1-score 0.1462 0.1288 0.59
Specificity 0.81 0.8358 0.77
BERTScore 0.8657 0.9939 0.8448
ROUGE-L 0.0373 0.3982 0.0382
Breast DensityAccuracy 0.2823 0.2494 0.35
Precision 0.3361 0.1168 0.65
Recall 0.2823 0.2494 0.35
F1-score 0.2711 0.1082 0.3
Specificity 0.7674 0.7477 0.7316
BERTScore 0.8508 0.7696 0.72
ROUGE-L 0.1582 0.0632 0.2
FindingsBERTScore 0.8478 0.7702 0.55
ROUGE-L 0.1388 0.0405 0.12
26