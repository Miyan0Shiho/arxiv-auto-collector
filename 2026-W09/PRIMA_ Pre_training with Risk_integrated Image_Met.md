# PRIMA: Pre-training with Risk-integrated Image-Metadata Alignment for Medical Diagnosis via LLM

**Authors**: Yiqing Wang, Chunming He, Ming-Chen Lu, Mercy Pawar, Leslie Niziol, Maria Woodward, Sina Farsiu

**Published**: 2026-02-26 18:07:52

**PDF URL**: [https://arxiv.org/pdf/2602.23297v1](https://arxiv.org/pdf/2602.23297v1)

## Abstract
Medical diagnosis requires the effective synthesis of visual manifestations and clinical metadata. However, existing methods often treat metadata as isolated tags, failing to exploit the rich semantic knowledge embedded in clinical descriptions. We propose PRIMA (Pre-training with Risk-integrated Image-Metadata Alignment), a framework that integrates domain-specific knowledge into multi-modal representation learning. We first curate an expert corpus of risk-disease correlations via Retrieval-Augmented Generation (RAG) to refine Clinical ModernBERT, embedding diagnostic priors into the text encoder. To bridge the modality gap, we introduce a dual-encoder pre-training strategy utilizing DINOv3 and our refined BERT, optimized by a suite of four complementary loss functions. These losses are designed to capture multi-granular semantic alignment and handle the ambiguity of clinical correlations through soft labels. Finally, we leverage Qwen-3 to fuse these aligned features for precise disease classification. Extensive experiments demonstrate that PRIMA effectively harmonizes pixel-level features with abstract clinical expertise, significantly outperforming other state-of-the-art methods. Notably, our framework achieves superior robustness without the need for massive data collection or exhaustive computational resources. Our code will be made public upon acceptance.

## Full Text


<!-- PDF content starts -->

PRIMA: Pre-training with Risk-integrated
Image-Metadata Alignment for Medical Diagnosis
via LLM
Yiqing Wang1, Chunming He1, Ming-Chen Lu2, Mercy Pawar2, Leslie Niziol2,
Maria Woodward2, and Sina Farsiu1
1Duke University, Durham NC 27708, USA
{yq.wang,chunming.he,sina.farsiu}@duke.edu
2University of Michigan, Ann Arbor MI 48109, USA
{mingchlu,mpawar,lmniziol,mariawoo}@med.umich.edu
Abstract.Medical diagnosis requires the effective synthesis of visual
manifestations and clinical metadata. However, existing methods often
treatmetadataasisolatedtags,failingtoexploittherichsemanticknowl-
edgeembeddedinclinicaldescriptions.WeproposePRIMA(Pre-training
with Risk-integrated Image-Metadata Alignment), a framework that in-
tegratesdomain-specificknowledgeintomulti-modalrepresentationlearn-
ing. We first curate an expert corpus of risk-disease correlations via
Retrieval-AugmentedGeneration(RAG)torefineClinicalModernBERT,
embedding diagnostic priors into the text encoder. To bridge the modal-
ity gap, we introduce a dual-encoder pre-training strategy utilizing DI-
NOv3andourrefinedBERT,optimizedbyasuiteoffourcomplementary
lossfunctions.Theselossesaredesignedtocapturemulti-granularseman-
tic alignment and handle the ambiguity of clinical correlations through
soft labels. Finally, we leverage Qwen-3 to fuse these aligned features for
precise disease classification. Extensive experiments demonstrate that
PRIMA effectively harmonizes pixel-level features with abstract clini-
cal expertise, significantly outperforming other state-of-the-art methods.
Notably, our framework achieves superior robustness without the need
for massive data collection or exhaustive computational resources. Our
code will be made public upon acceptance.
Keywords:Vision-LanguagePre-training·ClinicalKnowledgeIntegra-
tion·Medical Image Diagnosis.
1 Introduction
Medical imaging is central to clinical diagnosis, where experts synthesize infor-
mation from multi-view scans and complementary metadata, such as patient
risk factors. While deep learning has seen success, most methods remain lim-
ited to single-image analysis, overlooking the heterogeneous nature of real-world
clinical data. This creates a significant gap between current algorithms and ac-
tual diagnostic protocols involving diverse imaging and structured risk profiles.arXiv:2602.23297v1  [cs.CV]  26 Feb 2026

2 Y. Wang et al.
Traditional Approach
(Single Image, No Prior Knowledge)
Single Image
Neural Network
(e.g., ResNet , ViT)
Incorrect Diagnosis: 
Nevus
Limited context leads 
to misclassification.Our PRIMA
(Multi -modal, Knowledge -Enhanced, LLM -Integrated)
Multi -modal Images 
Feature Alignment
PRIMA PipelinePatient Risk Factors: Age, 
Gender, Family History, Sun 
Exposure, Diameter, etc.
External 
Knowledge 
Corpus
𝓛𝒊𝒎𝒈𝓛𝒈𝒍𝒐
𝓛𝒍𝒐𝒄𝓛𝒔𝒐𝒇𝒕
LLM 
IntegrationCorrect Diagnosis: 
Melanoma
Comprehensive analysis with 
knowledge and LLM reasoning.
Fig.1: Traditional vs. Our PRIMA Approach.
Furthermore, data scarcity remains a persistent barrier; despite recent efforts to
curate large-scale medical datasets [21,10,19], relying on massive data is often in-
feasible for specialized tasks or rare diseases where patient cohorts are inherently
limited.
Previous studies [13,17] have explored metadata fusion, yet these often rely
on ad-hoc designs that impede generalizability across different clinical formats.
RecentshiftstowardLargeLanguageModels(LLMs)andCLIP-basedparadigms
[18] offer robust representation capabilities but are typically data-intensive and
sensitive to quality. Even with specialized medical adaptations [22,8,5], current
methods remain heavily dependent on large-scale pre-training or fine-grained
textual reports. Consequently, the rich semantic potential of structured clinical
metadata remains largely underexplored, presenting a critical opportunity for
more efficient, attribute-aware diagnostic frameworks.
In this paper, we propose PRIMA (Pre-training with Risk-integrated Image-
MetadataAlignment),aframeworkdesignedtosynergizedomain-specificclinical
knowledge with visual features (Fig. 1). To capture high-quality priors, we first
construct a specialized knowledge bank of risk-disease relationships by leverag-
ing GPT [15] and Gemini [3] via Retrieval-Augmented Generation (RAG) [1] on
clinical literature. Subsequently, we fine-tune Clinical ModernBERT [7] on this
corpus, circumventing the need for massive paired datasets by exploiting cost-
effective expert literature to enhance semantic understanding. To bridge modal-
ity gaps, a dual-encoder pre-training stage utilizes DINOv3 [20] alongside our
refined text encoder. We introduce four complementary objective functions to
orchestrate the integration of diverse imaging modalities and textual knowledge
at multiple scales. By harmonizing global context with fine-grained local fea-
tures, this hierarchical alignment ensures robust adaptation to complex clinical
scenarios. Finally, Qwen3 [24] aggregates these multi-modal features for precise
disease diagnosis. Evaluations on PAD-UFES-20 [16] and AQUA show PRIMA
significantly outperforms SOTA baselines without requiring massive data or ex-
haustive compute. Our contributions are summarized as follows:
–Knowledge-EnhancedEncoding:Weelevatemetadatatosemanticknowledge
by fine-tuning ClinicalBERT on RAG-derived corpora, explicitly injecting
domain priors without requiring massive paired datasets.

PRIMA 3
–Multi-GranularAlignment:Weproposeaversatilestrategywithfourcomple-
mentary losses to orchestrate global-local integration across diverse modali-
ties, ensuring flexibility for heterogeneous clinical data.
–LLM-Driven Diagnosis: We introduce a unified pipeline leveraging Qwen3
to synthesize aligned features, achieving state-of-the-art performance and
robust generalization on PAD-UFES-20 and AQUA datasets.
2 Method
Fig. 2 illustrates the overall architecture of PRIMA, which comprises three pro-
gressive training stages. First, we curate a domain-specific knowledge bank via
Retrieval-Augmented Generation (RAG) using GPT [15] and Gemini [3] based
on public literature. This corpus serves as the foundation for fine-tuning our
text encoder, ClinicalBERT [7], to inject medical priors (Section 2.1). Next, we
introduce a set of four complementary objective functions to orchestrate feature
alignment across hierarchical semantic levels and diverse visual-textual perspec-
tives (Section 2.2). Finally, a large language model is employed to synthesize
the robust features extracted by the pre-trained encoders for precise diagnostic
predictions (Section 2.3).
Stage 1: Corpus Curation and 
Knowledge Prior Injection
Public 
Literature
RAG Process
Knowledge 
Bank (Physician 
Vetted)Prompt: You are a 
medical researcher 
specializing in 
dermatology and skin 
oncology. You are given 
a specific risk factor:
{RISK_FACTOR}.
Your task is to generate 
a structured description 
of the relationship 
between this risk factor 
and six skin lesion 
diagnoses (Basal Cell 
Carcinoma, Squamous 
Cell Carcinoma, Actinic 
Keratosis, Seborrheic 
Keratosis, Melanoma, 
and Nevus), based on 
the clinical dermatology 
and oncology literature I 
provided above…….
Frozen Backbone 
(Clinical ModernBERT )LoRA
Module
MLM TrainingSummary + 
Detailed 
DescriptionsStage 2: Risk -integrated Image -Metadata Alignment
Multi -modal 
Scans { vi,j}Clinical 
Metadata { ti}
Knowledge -Enhanced 
Text Encoder 
(Parameter 
Transferred 
from Stage 1) Image Encoder
Frozen 
Backbone 
(DINO v3)
LoRA
Module
Projection 
Head
Projection 
Head
Classification 
Head
Feature Alignment
Image Consistency 𝓛𝒊𝒎𝒈
Align 𝑝𝑖,𝑗𝑐𝑙𝑠
Global Semantic Consistency 𝓛𝒈𝒍𝒐
Align 𝑝𝑖,𝑗𝑐𝑙𝑠𝑞𝑖𝑐𝑙𝑠
Local Semantic Consistency 𝓛𝒍𝒐𝒄
Align 𝑝𝑖,𝑗𝑠𝑒𝑞𝑞𝑖𝑠𝑒𝑞
Soft Semantic Consistency 𝓛𝒔𝒐𝒇𝒕
Align 𝑝𝑖,𝑗𝑐𝑙𝑠𝑞𝑖𝑐𝑙𝑠 
by metadata similaritySupervised 
Fine -tuning
Cross Entropy 
Loss 
Aligned 
Features
𝒑𝒊,𝒋𝒄𝒍𝒔, 𝒒𝒊𝒄𝒍𝒔,
𝒑𝒊,𝒋𝒔𝒆𝒒, 𝒒𝒊𝒔𝒆𝒒Stage 3: Feature Integration via 
Large Language Model 
Global 
Image 
Tokens
𝑝𝑖,𝑗𝑐𝑙𝑠Local 
Image 
Tokens
𝑝𝑖,𝑗𝑠𝑒𝑞Global 
Text 
Tokens
𝑞𝑖𝑐𝑙𝑠Local 
Image 
Tokens
𝑞𝑖𝑠𝑒𝑞
MLP 
Projector
Large Language Model
Frozen Backbone 
(Qwen3 -1.7B)LoRA
Module
<Prompt> <img_start > <txt_start >
Vocabulary -Restricted Output 
(Logits for Class Subset C)
Diagnostic Prediction2D 
Conv 
Block
MLP 
Projector
1D 
Conv 
Block
Fig.2: Overview of our PRIMA.
2.1 Corpus Curation and Knowledge Prior Injection
WhileLLMsexcelgenerally,theyoftenfalterinmedicalscenariosduetocomplex
knowledge barriers. Clinical ModernBERT [7] partially mitigates this, yet it
still struggles to capture fine-grained attribute correlations and rare pathologies.

4 Y. Wang et al.
Since data scarcity often hinders downstream fine-tuning, we leverage PubMed
[11] literature to enhance semantic understanding. By retrieving task-specific
articles (e.g., systematic reviews and case reports), we extract expert knowledge
on risk-disease relationships, allowing our encoder to capture diagnostic priors
without relying on scarce clinical data.
However, as clinical literature often presents correlations implicitly or con-
tainsconflictingevidence,weemployGPT-5.1[15]andGemini-2.5[3]viaRetrieval-
Augmented Generation (RAG) [1] to synthesize these findings. RAG effectively
mitigates hallucination by grounding model responses in the retrieved sources.
As shown in Fig. 2, the LLMs generate structured descriptions—comprising a
global summary and detailed paragraphs on specific risk factors—which are vet-
ted by senior physicians to ensure accuracy. Following the Clinical ModernBERT
protocol, we utilize Masked Language Modeling (MLM) [4] for knowledge injec-
tion. To maintain computational efficiency, we adopt LoRA [6], updating only
1% of the parameters while preserving pre-trained features.
2.2 Risk-integrated Image-Metadata Alignment
ImageandTextEncoderWeemployDINOv3[20]andourknowledge-enhanced
Clinical ModernBERT [7] as vision and text backbones. For thei-th patient, in-
put data consists of multi-modal scans{v i,j}jand structured clinical metadata
{ti,k}k. Guided by domain priors (Section 2.1), we filter for visually relevant
attributes and concatenate them into a sequencet i. The encoders extract latent
featuresq i∈R(n+1)×dfrom text andp i,j∈R(m+1)×dfrom each image, where
n, mdenote sequence lengths anddis the latent dimension. Both modalities are
mapped to a shared space via projection heads. At this stage, only the projection
heads and LoRA [6] adapters in DINOv3 are trained (Fig. 2).
Alignment StrategyWe decomposep i,jandq iinto global class tokens
(pcls
i,j, qcls
i)andlocalsequencetokens(pseq
i,j, qseq
i)tofacilitatemulti-granularalign-
ment through four complementary objectives (Fig. 3). Image Consistency Loss
(Limg)enforcesintra-patientconsistencybyaligningglobalvisualfeaturesacross
scans. Global Semantic Loss (L glo) synchronizes visual and textual class tokens
forhigh-levelsemanticalignment,whileLocalSemanticLoss(L loc)capturesfine-
grainedcorrelationsbetweenimagepatchesandtextualtokens.Tohandleclinical
ambiguity, Soft Semantic Loss (L soft) provides soft supervision via metadata-
based similarity matrices. The final objective is a weighted sum of these losses.
Following alignment, the image encoder undergoes supervised fine-tuning with
ground-truth labels to sharpen its discriminative power.
– Image Consistency LossL img:To enforce intra-patient visual consis-
tency, we construct positive pairs by sampling either two distinct scans or a
duplicated single scan from the same patient, applying independent random
augmentations to each. The vision encoder extracts global class tokens from
these views, and their similarity is measured via a temperature-scaled dot

PRIMA 5
Batch of N Patients
Multi -modal 
Scans vi1, vi2Clinical 
Metadata ti…
Clinical 
ModernBERT
Text EncoderDINO v3
Image Encoder
Frozen 
Backbone 
LoRA
Module
𝑞𝑖𝑐𝑙𝑠𝑞𝑖𝑠𝑒𝑞𝑝𝑖,𝑗𝑐𝑙𝑠𝑝𝑖,𝑗𝑠𝑒𝑞Image Consistency Loss 𝓛𝒊𝒎𝒈
Intra -Patient Visual Alignment
Batch Similarity Matrix (2N ×2N)
Negatives
…
Negatives𝑝1,1𝑐𝑙𝑠𝑝1,2𝑐𝑙𝑠𝑝2,1𝑐𝑙𝑠𝑝2,2𝑐𝑙𝑠…    𝑝𝑁,1𝑐𝑙𝑠𝑝𝑁,2𝑐𝑙𝑠
𝑝1,1𝑐𝑙𝑠
𝑝1,2𝑐𝑙𝑠
𝑝2,1𝑐𝑙𝑠
𝑝2,2𝑐𝑙𝑠
…    
𝑝𝑁,1𝑐𝑙𝑠
𝑝𝑁,2𝑐𝑙𝑠𝑝𝑖,1𝑐𝑙𝑠
𝑝𝑖,2𝑐𝑙𝑠Global Semantic Consistency Loss𝓛𝒈𝒍𝒐
Image -Text High -Level Alignment
Batch Similarity Matrices (N ×N)
Negatives
…
Negatives𝑝1,𝑗𝑐𝑙𝑠𝑝2,𝑗𝑐𝑙𝑠𝑝3,𝑗𝑐𝑙𝑠𝑝4,𝑗𝑐𝑙𝑠…  𝑝𝑁−1,𝑗𝑐𝑙𝑠𝑝𝑁,𝑗𝑐𝑙𝑠
𝑞1𝑐𝑙𝑠
𝑞2𝑐𝑙𝑠
𝑞3𝑐𝑙𝑠
𝑞4𝑐𝑙𝑠
…    
𝑞𝑁−1𝑐𝑙𝑠
𝑞𝑁𝑐𝑙𝑠𝑞𝑖𝑐𝑙𝑠𝑝𝑖,𝑗𝑐𝑙𝑠
Local Semantic Consistency Loss𝓛𝒍𝒐𝒄
Fine-Grained Image -Text Alignment
𝑝𝑖,𝑗𝑠𝑒𝑞
𝑞𝑖𝑠𝑒𝑞
𝑞𝑖𝑙𝑝𝑖,𝑗𝑘
Attention෢𝑝𝑖,𝑗𝑙Negatives
…
Negatives(j = 1, 2)
Batch Similarity Matrices (L×L) 
෢𝑝𝑖,𝑗1   ෢𝑝𝑖,𝑗2     …෢𝑝𝑖,𝑗𝐿-1  ෢𝑝𝑖,𝑗𝐿
𝑞𝑖1
𝑞𝑖2
…    
𝑞𝑖𝐿-1
𝑞𝑖𝐿
(i=1,…N,j = 1, 2)Soft Semantic Consistency Loss𝓛𝒔𝒐𝒇𝒕
Metadata -Driven Soft Targets
𝑞𝑖𝑐𝑙𝑠𝑝𝑖,𝑗𝑐𝑙𝑠
𝑦𝑖Meta data Similarity 
CalculationNegatives
…
Negatives𝑝1,𝑗𝑐𝑙𝑠𝑝2,𝑗𝑐𝑙𝑠…   𝑝𝑁−1,𝑗𝑐𝑙𝑠𝑝𝑁,𝑗𝑐𝑙𝑠
𝑞1𝑐𝑙𝑠
𝑞2𝑐𝑙𝑠
…    
𝑞𝑁−1𝑐𝑙𝑠
𝑞𝑁𝑐𝑙𝑠Batch Similarity Matrices (N×N) 
(j = 1, 2)Soft Cross -
Entropy 
SupervisionPatient 
Metadata
Fig.3: Training Objectives of Our PRIMA.
product (Eq. 1) with hyperparameterτ.
sim(x 1, x2) =x⊤
1x2
∥x1∥∥x2∥/τ(1)
For a batch ofNpatients, letpcls
i,1andpcls
i,2be global visual tokens from
two views (distinct scans or augmentations) of patienti. Defined in Eq. 2,
Limgpromotes patient-invariant representation learning by aligning latent
features belonging to the same subject, where⊮is the indication function.
Limg=−1
NNX
i=1logesim(pcls
i,1,pcls
i,2)
PN
k=1(esim(pcls
i,1,pcls
k,2)+⊮[k̸=i]esim(pcls
i,1,pcls
k,1))(2)
– GlobalSemanticLossL glo:Toestablishasharedfeaturespace,theGlobal
Semantic Loss (L glo) employs a symmetric cross-entropy objective (Eq. 3)
overNmatched global pairs{(pcls
i,j, qcls
i)}N,2
i=1,j=1. This synchronizes high-
level semantic context, ensuring global visual embeddings capture the ab-
stract clinical concepts encoded in the metadata.
Lglo=−1
4NNX
i=12X
j=1 
logesim(pcls
i,j,qcls
i)
PN
k=1esim(pcls
i,j,qcls
k)+ logesim(qcls
i,pcls
i,j)
PN
k=1esim(qcls
i,pcls
k,j)!
(3)
– LocalSemanticLossL loc:To capturefine-grainedcorrelations, weemploy
an attention-guided mechanism to synthesize a visual-word representation
ˆpl
i,jfor each of theLtext tokensqseq
i={ql
i}L
l=1. By usingql
ias a query to
attend to allKimage patchespseq
i,j={pk
i,j}K
k=1, we aggregate the patches
based on their relevance scores per Eq. 4.
αl,k=exp(ql⊤
ipk
i,j/√
d)
PK
r=1exp(ql⊤
ipr
i,j/√
d),ˆpi,jl=KX
k=1αl,kpk
i,j (4)
Subsequently, we maximize the similarity between the original tokenql
iand
its corresponding visual contextˆpl
i,jwhile minimizing its similarity to other

6 Y. Wang et al.
tokens in the sequence, as formulated in Eq. 5. This mechanism effectively
grounds abstract clinical attributes (e.g., irregular borders) onto their cor-
responding visual manifestations in the scan.
Lloc=−1
4LNLX
l=1NX
i=12X
j=1 
logesim(ql
i,ˆpl
i,j)
PL
m=1esim(ql
i,ˆpm
i,j)+ logesim(ˆpl
i,j,ql
i)
PL
m=1esim(ˆpm
i,j,ql
i)!
(5)
– Soft Semantic LossL soft:To address the limitations of strict one-to-
one mapping in standard contrastive losses, which neglect shared clinical
attributes across patients, we introduce a soft-target alignment mechanism.
We aggregate metadata into one-hot vectorsy, upweighting disease-related
dimensions (by a factor of 3) to prioritize diagnostic discriminability. A dy-
namic target distribution is then constructed via label sharpening:s i,j=
Softmax(⟨y i, yj⟩/τlabel),whereτ labelcontrolssignalsparsity.Finally,themodel
minimizes the soft cross-entropy between this distribution and the predicted
image-text similarities (Eq. 6).
Lsoft=−1
4N2NX
i=1NX
j=12X
k=1si,j 
logesim(pcls
i,k,qcls
j)
PN
l=1esim(pcls
i,k,qcls
l)+ logesim(qcls
j,pcls
i,k)
PN
l=1esim(qcls
j,pcls
l,k)!
(6)
The final alignment loss is as shown in Eq. 7. The weight coefficientsβ 1,β2,β3
andβ 4are set asβ 1= 0.2, β 2= 0.3, β 3= 0.2, β 4= 0.3.
Lalign =β1Limg+β2Lglo+β3Lloc+β4Lsoft.(7)
Following feature alignment, the image backbone undergoes supervised refine-
ment using ground-truth labels and Cross-Entropy loss [14]. This stage sharpens
visual representations by bridging the gap between general cross-modal align-
ment and task-specific diagnostic requirements.
2.3 Feature Integration via Large Language Model
To leverage Large Language Model reasoning, we bridge the gap between en-
coders and the LLM backbone via a multi-modal projection strategy. Global
tokens are projected using MLP layers, while local sequence tokens are aligned
through1D/2Dconvolutionblocksthatperformspatial/temporaldownsampling
to reduce overhead. These features are concatenated into an input sequence
demarcated by learnable special tokens (e.g., <|img_start|>), which are ini-
tialized with semantic embeddings to accelerate convergence. To maintain effi-
ciency and prevent overfitting, we utilize LoRA [6], updating only the projec-
tors and∼1% of total parameters. We further employ a vocabulary-restricted
strategy to eliminate hallucinations: instead of free-form generation, we extract
logitsz kexclusively from a token subsetCcorresponding to pre-defined clin-
ical classes. Probabilities are computed via Softmax over this constrained set:
P(y=k|x) =exp(z k)P
j∈Cexp(z j). The model is optimized by Cross Entropy loss [14].

PRIMA 7
3 Experiment and Result
Dataset DetailsWe evaluated PRIMA on two benchmarks. PAD-UFES-20
[16] contains 2,298 images of 1,891 lesions from 1,373 patients across six cate-
gories: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic
Keratosis (ACK), Seborrheic Keratosis (SEK), Melanoma (MEL), and Nevus
(NEV). AQUA, a private dataset collected at anonymized medical centers under
an approved IRB protocol ([Anonymized for Review]), comprises 19,567 slit-
lamp photography (SLP) images from 1,827 subjects. It focuses on bacterial and
fungal keratitis using three modalities: diffuse white light, diffuse blue light, and
sclerotic scatter illumination. We adopted a lesion-level split for PAD-UFES-20
and a patient-level split for AQUA. We employed five-fold cross-validation and
evaluated performance using F1-score, accuracy, and balanced accuracy (BAcc).
Implementation DetailsPRIMA is trained on two NVIDIA RTX 4090
GPUs (24GB) using AdamW and a cosine scheduler with 10% warm-up. Hyper-
parameters across the three stages are: Stage 1 (LR 5e-5, WD 1e-3, 2500 epochs),
Stage 2 (LR 3e-5, WD 1e-3, 150 epochs), and Stage 3 (LR 1e-5, WD 3e-2, 80
epochs). LoRA ranks are set to 8, 32, and 16 for our text encoder, our image
encoder, and the LLM, respectively. Our code will be public upon acceptance.
3.1 Main Results
We evaluated PRIMA against several prominent baselines across two datasets.
To establish a rigorous reference, we first fine-tuned DINOv3 [20] via LoRA as a
pure-imagebaseline.WethencombinedthesevisualfeatureswithvanillaClinical
ModernBERT[7]embeddingsusinganMLPtoassesstheimpactofsimplemulti-
modal fusion. Furthermore, we compared PRIMA against specialized diagnostic
networks: DeepIK [9], which employs a CNN-based architecture specifically for
keratitis diagnosis; MedKLIP [23], which proposes a Transformer-based fusion
module to integrate multi-modal visual and textual information; and KnoBo
[25], which first projects image features into a symbolic concept space to facili-
tate clinical decision-making. Finally, we included MedBLIP [2] and MLRG [12],
which leverage LLMs for feature aggregation through diverse training strategies.
Results in Tables 1–2 show that while DINOv3 provides a high performance
floor, PRIMA consistently yields a>5%accuracy boost via multi-granular
alignment, confirming that expert prior injection is vital where generic visual
strength is insufficient. PRIMA achieves state-of-the-art results on PAD-UFES-
20 (F1: 73.75%, Acc: 78.27%) and AQUA (F1: 85.22%, Acc: 86.04%), surpass-
ing all competitors. The performance drop of specialized methods (MedKLIP,
KnoBo) on PAD-UFES-20 underscores their limited transferability compared to
our holistic strategy. While LLM-based baselines (MedBLIP, MLRG) perform
strongly, our optimized training and knowledge injection consistently yield supe-
rior feature integration. Although PAD-UFES-20 metadata could theoretically
appear in some foundation models’ pre-training corpora, we mitigate this by us-
ingLLMssolelyforofflineknowledgeextractionwithoutpatientdatainteraction.

8 Y. Wang et al.
Crucially, PRIMA’s significant gains on the private AQUA dataset—entirely in-
accessible to foundation models—validate that our performance stems from the
proposed alignment strategy rather than data memorization.
Table 1: Quantitative results of PAD-UFES-20 dataset (%). SD in parentheses.
Methods BCC SCC MEL ACK SEK NEV Avg F1-score Acc BAcc
DINOv3 (ArXiv-25) [20] LoRA 72.23 (1.76) 32.55 (5.80) 70.41 (13.35) 79.38 (1.36) 76.72 (3.35) 76.90 (4.58) 68.03 (2.48) 71.07 (0.83) 69.66 (3.53)
+ Metadata via MLP 75.68 (3.08) 34.90 (6.57) 73.12 (12.72) 83.00 (2.95) 76.72 (5.15) 79.50 (4.74) 70.49 (3.14) 73.93 (2.50) 71.96 (4.20)
MedKLIP (ICCV-23) [23] 54.10 (6.87) 16.08 (3.59) 21.85 (18.08) 61.99 (3.53) 47.81 (3.14) 55.24 (7.17) 42.84 (4.77) 51.29 (3.59) 43.81 (5.10)
KnoBo (NIPS-24) [25] 63.33 (3.40) 23.04 (6.47) 33.32 (22.43) 70.80 (1.99) 32.25 (9.48) 63.75 (8.56) 47.75 (4.91) 58.80 (2.63) 50.89 (5.77)
MedBLIP (ACCV-24) [2] 77.24 (4.29) 38.37 (5.00) 77.60 (13.21) 81.18 (2.40) 76.08 (4.31) 81.61 (5.53) 72.01 (2.36) 75.41 (2.58) 72.32 (3.01)
MLRG (CVPR-25) [12] 73.70 (8.11) 23.45 (8.56) 62.36 (13.17) 81.39 (2.12) 59.34 (1.79) 69.00 (4.56) 61.54 (4.11) 70.65 (4.55) 60.28 (4.79)
PRIMA (Ours) 80.21 (1.19) 37.33 (7.86) 85.84 (8.35) 85.73 (2.23) 73.10 (5.77) 80.26 (6.07) 73.75 (3.40) 78.27 (1.78) 72.76 (2.86)
Table 2: Quantitative results of AQUA
dataset (%). SD in parentheses.
Methods Fungal Bacterial Avg F1-score Acc BAcc
DINOv3 (ArXiv-25) [20] LoRA 82.54 (1.44) 71.42 (2.74) 76.98 (1.78) 78.37 (1.60) 78.25 (1.42)
+ Metadata via MLP 86.36 (2.48) 79.10 (2.96) 82.73 (2.63) 83.52 (2.66) 83.15 (1.93)
DeepIK (NPJ Dig. Med.-24) [9] 80.84 (2.58) 76.54 (3.56) 78.69 (3.00) 78.93 (2.94) 81.11 (2.38)
KnoBo (NIPS-24) [25] 83.54 (4.62) 72.42 (7.71) 77.98 (6.09) 79.42 (5.72) 77.76 (5.75)
MedBLIP (ACCV-24) [2] 80.96 (6.41) 75.55 (4.67) 78.26 (5.13) 78.82 (5.37) 79.86 (4.34)
MLRG (CVPR-25) [12] 83.35 (2.36) 74.28 (4.32) 78.81 (2.73) 79.91 (2.44) 79.15 (2.81)
PRIMA (Ours) 88.66 (1.22) 81.78 (3.21) 85.22 (2.17) 86.04 (1.82) 85.40 (2.27)Table 3: Ablation study of PAD-
UFES-20 dataset (%).
Knowledge PretrainingL imgLgloLlocLsoftLloc_dir Lsup_conAvg F1-score Acc BAcc
67.94 (3.88) 75.25 (3.85) 66.20 (3.50)
✓ 68.41 (4.19) 74.78 (3.82) 67.70 (3.54)
✓ 68.49 (3.34) 75.99 (2.97) 67.79 (3.87)
✓ ✓ 70.36 (2.48) 76.63 (1.90) 69.69 (3.05)
✓ ✓ ✓ 72.42 (2.35) 76.42 (3.19) 72.09 (3.45)
✓ ✓ ✓ ✓ 72.38 (3.23) 77.84 (3.54) 71.80 (2.19)
✓ ✓ ✓ ✓ ✓ 70.02 (3.80) 77.47 (3.35) 68.53 (4.25)
✓ ✓ ✓ ✓ ✓ 71.42 (2.96) 76.15 (3.03) 72.05 (2.43)
✓ ✓✓✓✓ 73.75 (3.40) 78.27 (1.78) 72.76 (2.86)
3.2 Ablation Study
Ablation results (Table 3) justify each component’s role. While LLMs possess
inherent representational power, they struggle under limited medical supervi-
sion, evidenced by the performance drop when trained solely on class labels.
Integrating Image Consistency Loss (L img) enhances visual robustness, though
cross-modal alignment remains unaddressed. Global Semantic Loss (L glo) pro-
vides essential high-level synchronization, whereas Local Semantic Loss (L loc)
establishes fine-grained correspondences between local image regions and textual
tokens. Furthermore, Soft Semantic Loss (L soft) leverages clinical metadata to
incorporate complex relationships during feature aggregation.Finally, injecting
domain-specific priors significantly sharpens the model’s grasp of clinical nu-
ances, with their synergy achieving PRIMA’s optimal performance. To further
validate our designs, we evaluated simplified variants: removing the attention
mechanism fromL loc(denotedL loc_dir) and restrictingL softto basic class la-
bels (denotedL sup_con). The resulting performance degradation underscores the
necessity of our multi-granular and metadata-aware formulations.

PRIMA 9
4 Conclusion
We introduce PRIMA, which integrates Clinical ModernBERT, pre-trained on
an expert medical corpus, with DINOv3 to align diagnostic priors with robust
visual representations. Four multi-granular losses synchronize these components
to capture precise clinical correspondences. With Qwen-3 orchestrating the fi-
nal multi-modal fusion, PRIMA achieves state-of-the-art accuracy and high effi-
ciency without requiring massive compute. We acknowledge two limitations: the
need for backbone-controlled experiments to isolate alignment gains from en-
coder strength, and potential generational biases among LLMs of similar scales.
These aspects will be addressed in a subsequent journal extension.
References
1. Amugongo, L.M., Mascheroni, P., Brooks, S., Doering, S., Seidel, J.: Retrieval aug-
mented generation for large language models in healthcare: A systematic review.
PLOS Digital Health4(6), e0000877 (2025)
2. Chen, Q., Hong, Y.: Medblip: Bootstrapping language-image pre-training from 3d
medical images and texts. In: Proceedings of the Asian conference on computer
vision. pp. 2404–2420 (2024)
3. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,
I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long context, and next generation
agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)
4. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidi-
rectional transformers for language understanding. In: Proceedings of the 2019
conference of the North American chapter of the association for computational
linguistics: human language technologies, volume 1 (long and short papers). pp.
4171–4186 (2019)
5. Du, J., Guo, J., Zhang, W., Yang, S., Liu, H., Li, H., Wang, N.: Ret-clip: A retinal
image foundation model pre-trained with clinical diagnostic reports. In: Interna-
tional conference on medical image computing and computer-assisted intervention.
pp. 709–719. Springer (2024)
6. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen,
W., et al.: Lora: Low-rank adaptation of large language models. Iclr1(2), 3 (2022)
7. Lee, S.A., Wu, A., Chiang, J.N.: Clinical modernbert: An efficient and long context
encoder for biomedical text. arXiv preprint arXiv:2504.03964 (2025)
8. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T.,
Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for
biomedicine in one day. Advances in Neural Information Processing Systems36,
28541–28564 (2023)
9. Li, Z., Xie, H., Wang, Z., Li, D., Chen, K., Zong, X., Qiang, W., Wen, F., Deng,
Z., Chen, L., et al.: Deep learning for multi-type infectious keratitis diagnosis: a
nationwide, cross-sectional, multicenter study. NPJ Digital Medicine7(1), 181
(2024)
10. Li, Z., Wang, Y., Farsiu, S., Kinahan, P.: Boosting medical visual understanding
from multi-granular language learning. arXiv preprint arXiv:2511.15943 (2025)
11. Lindberg, D.: Internet access to the national library of medicine. Effective clinical
practice3(5) (2000)

10 Y. Wang et al.
12. Liu, K., Ma, Z., Kang, X., Li, Y., Xie, K., Jiao, Z., Miao, Q.: Enhanced con-
trastive learning with multi-view longitudinal data for chest x-ray report genera-
tion. In: Proceedings of the Computer Vision and Pattern Recognition Conference.
pp. 10348–10359 (2025)
13. Lu,S.,Liu,J.,Wang,X.,Zhou,Y.:Collaborativemulti-metadatafusiontoimprove
the classification of lumbar disc herniation. IEEE Transactions on Medical Imaging
42(12), 3590–3601 (2023)
14. Mao, A., Mohri, M., Zhong, Y.: Cross-entropy loss functions: Theoretical analysis
and applications. In: International conference on Machine learning. pp. 23803–
23828. pmlr (2023)
15. OpenAI: Openai 5.1 system card.https://cdn.openai.com/pdf/
4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf(2026), ac-
cessed: 2026-02-23
16. Pacheco, A.G.C., Lima, G.R., da Silva Salomão, A., Krohling, B., Biral, I.P.,
de Angelo, G.G., Jr, F.C.A., Esgario, J.G.M., Simora, A.C., Castro, P.B.C., Ro-
drigues, F.B., Frasson, P.H.L., Krohling, R.A., Knidel, H., Santos, M.C.S., do Es-
pírito Santo, R.B., Macedo, T.L., Canuto, T.R.P., de Barros, L.F.: Pad-ufes-20:
A skin lesion dataset composed of patient data and clinical images collected from
smartphones. Data in Brief32(2020)
17. Pacheco, A.G., Krohling, R.A.: An attention-based mechanism to combine images
and metadata in deep learning models applied to skin cancer classification. IEEE
journal of biomedical and health informatics25(9), 3554–3563 (2021)
18. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International conference on machine learning. pp.
8748–8763. PmLR (2021)
19. Silva-Rodriguez, J., Chakor, H., Kobbi, R., Dolz, J., Ayed, I.B.: A foundation
language-image model of the retina (flair): Encoding expert knowledge in text
supervision. Medical Image Analysis99, 103357 (2025)
20. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khali-
dov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint
arXiv:2508.10104 (2025)
21. Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z.,
Shen, Y., et al.: Sam-med3d: a vision foundation model for general-purpose seg-
mentation on volumetric medical images. IEEE Transactions on Neural Networks
and Learning Systems (2025)
22. Wang, Z., Wu, Z., Agarwal, D., Sun, J.: Medclip: Contrastive learning from un-
paired medical images and text. In: Proceedings of the 2022 Conference on Empir-
ical Methods in Natural Language Processing. pp. 3876–3887 (2022)
23. Wu, C., Zhang, X., Zhang, Y., Wang, Y., Xie, W.: Medklip: Medical knowledge
enhanced language-image pre-training for x-ray diagnosis. In: Proceedings of the
IEEE/CVF international conference on computer vision. pp. 21372–21383 (2023)
24. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang,
C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin,
H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin,
J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang,
P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin,
W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y.,
Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical
report. arXiv preprint arXiv:2505.09388 (2025)

PRIMA 11
25. Yang, Y., Gandhi, M., Wang, Y., Wu, Y., Yao, M., Callison-Burch, C., Gee, J.,
Yatskar, M.: A textbook remedy for domain shifts: Knowledge priors for medical
image analysis. Advances in neural information processing systems37, 90683–
90713 (2024)