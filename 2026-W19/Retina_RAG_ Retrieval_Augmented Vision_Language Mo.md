# Retina-RAG: Retrieval-Augmented Vision-Language Modeling for Joint Retinal Diagnosis and Clinical Report Generation

**Authors**: Abdelrahman Zaian, Sheethal Bhat, Mohamed Abdalkader, Andreas Maier

**Published**: 2026-05-07 12:54:53

**PDF URL**: [https://arxiv.org/pdf/2605.06173v1](https://arxiv.org/pdf/2605.06173v1)

## Abstract
Diabetic Retinopathy (DR) is a leading cause of preventable blindness among working-age adults worldwide, yet most automated screening systems are limited to image-level classification and lack clinically structured reporting. We propose Retina-RAG, a low-cost modular framework that jointly performs DR severity grading, macular edema (ME) detection, and report generation. The architecture decouples a high-performance retinal classifier and a parameter-efficient vision-language model (Qwen2.5-VL-7B-Instruct) adapted via Low-Rank Adaptation (LoRA), enabling flexible component integration. A retrieval-augmented generation (RAG) module injects curated ophthalmic knowledge together with structured classifier outputs at inference time to improve diagnostic consistency and reduce hallucinations. Retina-RAG achieves an F1-score of 0.731 for DR grading and 0.948 for ME detection, substantially outperforming zero-shot Qwen (0.096, 0.732) and MMed-RAG (0.541, 0.641) on a retinal disease detection dataset with captions. For report generation, Retina-RAG attains ROUGE-L 0.429 and SBERT similarity 0.884, exceeding all baselines. The full framework operates on a single consumer-grade GPU, demonstrating that clinically structured retinal AI can be achieved with modest computational resources.

## Full Text


<!-- PDF content starts -->

Retina-RAG: Retrieval-Augmented
Vision–Language Modeling for Joint Retinal
Diagnosis and Clinical Report Generation
Abdelrahman Zaian1, Sheethal Bhat1, Mohamed Abdalkader2, and Andreas
Maier1
1Friedrich-Alexander-Universität, Germany
{abdelrahman.zaian, sheethal.bhat, andreas.maier}@fau.de
2Mohameed.Abdalkadeer@gmail.com
Abstract.Diabetic Retinopathy (DR) is a leading cause of preventable
blindness among working-age adults worldwide, yet most automated
screening systems are limited to image-level classification and lack clini-
cally structured reporting. We propose Retina-RAG, a low-cost modular
framework that jointly performs DR severity grading, macular edema
(ME) detection, and report generation. The architecture decouples a
high-performanceretinalclassifierandaparameter-efficientvision–langu-
agemodel(Qwen2.5-VL-7B-Instruct)adaptedviaLow-RankAdaptation
(LoRA),enablingflexiblecomponentintegration.Aretrieval-augmented
generation(RAG)moduleinjectscuratedophthalmicknowledgetogether
with structured classifier outputs at inference time to improve diagnos-
tic consistency and reduce hallucinations. Retina-RAG achieves an F1-
score of 0.731 for DR grading and 0.948 for ME detection, substantially
outperforming zero-shot Qwen (0.096, 0.732) and MMed-RAG (0.541,
0.641) on a retinal disease detection dataset with captions. For report
generation, Retina-RAG attains ROUGE-L 0.429 and SBERT similarity
0.884, exceeding all baselines. The full framework operates on a single
consumer-grade GPU, demonstrating that clinically structured retinal
AI can be achieved with modest computational resources. Source code
at: https://xx/xxx.git
Keywords:Diabetic Retinopathy·Retrieval-Augmented Generation·
LoRA Vision-Language Models·Medical Report Generation
1 Introduction
Diabetic retinopathy (DR) is a leading cause of preventable blindness among
working-age adults worldwide [1]. According to the Global Burden of Disease
Study 2021, the prevalence of vision impairment attributable to DR is projected
to rise by 55.6% to 160.50 million cases by 2045 [2]. This increase is driven
by urbanisation, an ageing population, and the continued expansion of the di-
abetes epidemic in low- and middle-income countries [1]. Despite this growingarXiv:2605.06173v1  [cs.CV]  7 May 2026

2
burden, access to trained ophthalmologists is still severely limited in many high-
prevalence regions [2], creating an urgent need for scalable, automated screening
that delivers accurate and clinically interpretable results.
Although deep learning systems have achieved strong performance in auto-
mated DR severity grading, most existing approaches remain limited to image-
level classification without producing clinically interpretable reports suitable for
real-world deployment. CNN-based architectures such as EfficientNet [3] and
ResNet [4] variants have demonstrated competitive grading accuracy even un-
der significant class imbalance [5,6]. Hybrid CNN–Transformer and ensemble
methods have further advanced classification performance on standard bench-
marks [7]. This limitation persists even at the commercial level: LumineticsCore
(formerly IDx-DR), the first FDA-cleared autonomous AI system for DR detec-
tion deployed across primary care settings in the United States, outputs only
a binary referral decision without any structured clinical description of retinal
findings [8]. These systems output discrete severity labels without clinical justi-
fication, limiting their translational value in ophthalmic workflows.
Vision–Language models (VLM)s such as LLaVA-Med extend instruction
tuning to medical imaging tasks [9]. However, systematic evaluations show that
both general-purpose and medically adapted VLMs frequently generate factu-
ally incorrect outputs in medical settings, raising serious concerns about their
reliability for clinical use [10]. Dedicated hallucination benchmarks further doc-
ument this, revealing consistent failure patterns across multiple medical imaging
modalities [11]. Moreover, adapting large-scale VLMs often demands substantial
computational resources, limiting deployment in resource-constrained clinical
settings [12], motivating the need for parameter-efficient and modular alterna-
tives.
Retrieval-Augmented Generation (RAG) has been proposed to improve fac-
tual reliability by conditioning report generation on externally retrieved medical
knowledge[13]. RULE demonstrates meaningful gains in medical visual ques-
tion answering through preference-based retrieval alignment [14]. MMed-RAG
further extends this by introducing a domain-aware retrieval mechanism that
improves factual accuracy across diverse medical tasks [15]. However, current
RAG frameworks are not designed for DR screening’s diagnostic setup, which
combines imbalanced multi-class severity grading with binary Macular Edema
(ME) detection.
To address this gap, we introduceRetina-RAG, a modular framework for
joint DR severity grading, ME detection, and structured report generation.
Retina-RAG, as seen in Fig. 1 separates a high-performance retinal classifier
from a parameter-efficient VL backbone and integrates classifier-guided retrieval
prompting, injecting structured diagnostic predictions alongside curated oph-
thalmic knowledge at inference. This design aligns language generation with
task-specificclassificationoutputs,improvingdiagnosticconsistencywhileavoid-
ing computationally expensive end-to-end (e2e) multimodal training.
Main contributions:We 1) propose Retina-RAG, a diagnostic-aware frame-
work that separates DR severity grading and ME detection from report genera-

3
tion, enabling modular optimization while preserving clinical task alignment. 2)
introduce a classifier-guided retrieval prompting mechanism that conditions lan-
guage generation on structured diagnostic predictions and curated ophthalmic
knowledge, enforcing consistency between classification and reporting without
costly e2e multimodal training. 3) provide comprehensive evaluation on a retinal
disease and caption dataset [16], demonstrating improved grading (F1: 0.721),
ME detection (F1: 0.948), and report quality (ROUGE-L 0.419; SBERT 0.874)
using Retina-RAG in comparison to zero-shot (ZS) and medical RAG baselines.
2 Materials and methods
Fig. 1.Overview of Retina-RAG for both training and inference. We use a dual-branch
retinal classifier to process the fundus image. The classifier predicts either DR severity
and ME, producing structured text outputsP. These predictions are serialized to query
an ophthalmic knowledge base, retrievingktask-relevant snippetsK k. The fundus
image, classifier outputs, and retrieved knowledge are composed into a classifier-guided
promptTfor a LoRA-adapted VLM (Qwen2.5-VL), which generates the final clinical
report.
2.1 Dataset details and preparation
WeevaluateRetina-RAGonDatasetA,theRetinalDiseaseDetectiondataset[16],
comprising 2,254 colour fundus images (1,577 train / 339 val / 338 test) at
≈2,000×1,333 pixels, each annotated with a five-class DR grade and binary ME
label. DR grades are severely imbalanced: No DR dominates at 52.6%, followed
by Moderate (22.7%), Mild (13.0%), Severe (7.4%), and Proliferative (4.3%).
ME labels are 19.6% positive. DatasetB, MESSIDOR [17], provides 1,200 addi-
tional fundus images at 2,240×1,488 with retinopathy grade and ME risk anno-
tations, used solely for supplementary ME classifier training. No patient overlap

4
exists between splits. To address class imbalance, inverse-frequency weighting
is applied during training. Images are resized to 512×512 for DR classification,
224×224 for ME classification, and 448×448 for the LoRA-adapted VLM, all
normalised with ImageNet statistics (µ=[0.485, 0.456, 0.406],σ=[0.229, 0.224,
0.225]). VLM inputs use LANCZOS resampling; ME classifier inputs apply mor-
phological top-hat and black-hat transforms (15×15 elliptical kernel) on the
green channel for lesion contrast enhancement.
2.2 Methodology
Retina-RAGis implemented as a modular three-stage pipeline. As illustrated
inFig.1,thesystemcomprises:(i)adual-branchretinalclassifier,(ii)aclassifier-
guided RAG module, and (iii) a LoRA-adapted VLM for report generation. The
classifier and VLM are trained independently, enabling low-resource adaptation
and flexible component replacement. The Classifier processes a fundus image
I, producing a DR severity gradeg∈ {0, . . . ,4}, a ME labele∈ {0,1}, with
confidence scoress gands erespectively.P={g, e, s g, se}is a structured text in-
struction to the RAG module to retrievektask snippetsK k. The VLM processes
T= {P,K}, to generate the final reportr.
(i) Retinal classifiercomprises of two independently fine-tuned classifier mod-
ules. These are selected based on their performance on DR and ME detection.
DR grading branch:We adopt FLAIR [18], an ophthalmology foundation
model pretrained on 288,307 fundus images across 101 ophthalmic labels, as
the DR grading backbone. FLAIR combines a ResNet-50 visual encoder with
a BioBERT text head, aligned via contrastive pretraining on expert-curated
clinical descriptions. We adapt FLAIR for five-class DR severity grading (g∈
{0,1,2,3,4}corresponding to No DR, Mild, Moderate, Severe, and Proliferative
DR, respectively. using linear probing: the FLAIR backbone is frozen and a
single linear classification head is trained with a standard CE loss.
ME branch:Binary ME detection is processed by an ImageNet-pretrained
FunSwin Transformer classifier [33], fine-tuned on our training set using a stan-
dard BCE loss. Decoupling the two branches preserves the domain-specific repre-
sentations of each pretrained backbone and allows independent optimization per
task. At inference, both branches run in parallel and their outputs are serialised
intoP={g, e, s g, se}, which is passed downstream to both the RAG module
and the VLM prompt.
(ii) RAG modulegrounds report generation from structured classifier outputs
and retrieves ophthalmic knowledge. This reduces hallucinations and enforces
consistency between predicted grades and generated text [14].

5
Knowledge Base Construction:An offline knowledge base of curated DR
clinical report descriptions, grounded in established severity criteria [19,20], is
embedded withMedEmbed-small-v0.1and indexed via FAISS [21] for nearest-
neighbour retrieval.
Structured Query Formation and Class Matching:At inference,Pis
serialized from natural-language (e.g.,“Moderate DR, ME detected, confidence
0.87”) to query embeddingsq∈Rd. We performclass matchingbased on the
gradesgande, thereby ensuring retrieved snippets are semantically aligned with
the predicted diagnostic class, rather than with superficial visual features of the
query image.
Retrieval and prompt construction:The top-k(k=3) knowledge snippets
Kk={d 1, . . . , d k}are retrieved by cosine similarity betweenqandd i∈Rd, the
embedding of thei-th knowledge entry. The fundus imageI, classifier outputsP,
and retrieved snippetsK kare composed into a structuredT:Prompt(I,P,K k).
The VLM is instructed to treatPas the primary diagnostic anchor andK kas
supporting clinical context as below,
“You are an expert ophthalmologist. The classifier predicts:{P}. Relevant clin-
ical context:{K k}. Generate a concise diagnostic report for this fundus image.”
(iii)LoRA-AdaptedVLMReportgenerationusesQwen2.5-VL-7B-Instruct[22],
chosenforitsstronginstruction-followingperformanceunderresource-constrained
deployment[12].Fine-tuningisperformedusingUnsloth’sFastVisionModel[23]
with LoRA [24], applied to attention and MLP projections in both the vision
and language blocks. The adapted weight update is:
W′=W+∆W=W+α
rBA,(1)
whereWis frozen, andA∈Rr×d,B∈Rd×rare trainable low-rank matrices
withr≪d. The model is trained via supervised fine-tuning (SFT) [34,23] using
causal language modelling loss on the assistant response tokensr, wherer tis
thet-th token of the reference report andTis the structured prompt.
3 Experiments
Evaluationprotocol:Classificationperformanceunderclassimbalanceiseval-
uated using F1-score, precision, and recall. Accuracy is not reported due to its
sensitivity to skewed label distributions. Report generation quality is assessed
using BLEU-4 [35], ROUGE-1 and ROUGE-L [36], and semantic similarity com-
puted with SBERT [37] and ClinicalBERT [38] embeddings.

6
Implementation details:FLAIR is fine-tuned for 50 epochs using Adam
(LR1×10−4for encoder–decoder,1×10−5for visual extractor, weight decay
1×10−5) [24], while FunSwin uses AdamW (LR1×10−4) [33]. Qwen2.5-VL-
7B-Instruct is adapted via LoRA (r=64,α=128) applied to attention and MLP
layers in both vision and language branches; the 4-bit base model is frozen and
only LoRA parameters are updated. Training employs 8-bit AdamW with cosine
decay (warmup 0.15), weight decay 0.01, batch size 2 with gradient accumulation
(×4), and gradient clipping 0.5; loss is computed only over assistant-response to-
kens (SFT). Minority DR grades are upsampled with±15◦rotation and bright-
ness augmentation. The RAG module retrievesk=3snippets via FAISS using
MedEmbed-small-v0.1. Experiments run on a single RTX 1650 (4GB), and re-
sults report the mean of three seeds.
Baselines:We compare Retina-RAG against (1) ZS Qwen, the unmodified
Qwen2.5-VL-7B-Instructpromptedwiththefundusimageandagenericophthal-
mology instruction, without fine-tuning or retrieval; and (2) MMed-RAG [15], a
state-of-the-art (SOTA) as well as FFA-IR [30].
3.1 Results
Table 1 shows that Retina-RAG achieves the best overall classification perfor-
mance, with macro-F1 of 0.731 for DR severity and 0.948 for ME. For DR
grading, Retina-RAG substantially outperforms ZS Qwen (0.096) and MMed-
RAG (0.541), and improves over FFA-IR (0.660), while remaining competitive
in AUC (0.774 vs. 0.837). For ME detection, Retina-RAG achieves the highest
F1 (0.948), slightly surpassing FFA-IR (0.947), and clearly outperforming ZS
Qwen (0.732) and MMed-RAG (0.641). These results highlight the importance
of domain adaptation and retrieval grounding for fine-grained retinal classifica-
tion.
Table2comparesreportgenerationquality.Retina-RAGachievesaROUGE-
L of 0.419 and SBERT similarity of 0.874, substantially outperforming MMed-
RAG (0.353/0.786) and the ZS baseline (0.163/0.764). ClinicalBERT similarity
(0.972) further confirms strong clinical coherence and semantic alignment with
reference reports. While FFA-IR shows competitive classification performance,
it is limited to short reports (≤60 words), whereas Retina-RAG supports long-
form generation (approx. 1,000 words) while maintaining high diagnostic and
linguistic quality.
Qualitative Results:Fig. 2 denotes an example of a fundus image from the
RDD test set with the ground truth and predicted caption. This example shows
how Retina-RAG leverages retrieved Grade 2 clinical context to ground the
predicted caption, correctly identifying microaneurysms, cotton-wool spots, and
dot-and-blot hemorrhages consistent with Moderate Non-Proliferative DR. The
macular assessment correctly indicates no signs of diabetic ME, and no halluci-
nated severe features present in the output.

7
Table 1.Classification comparison between Retina-RAG and various SOTA methods
on RDD test dataset. Precision, Recall, F1 are weighted-averaged and AUC is macro-
averaged across DR categories. Results are mean of 3 runs.
DR Severity (5-class) ME (Binary)
Method Prec. Rec. F1 AUC Prec. Rec. F1 AUC
ZS Qwen [29] 0.076 0.204 0.096 0.511 0.743 0.722 0.732 0.598
MMed-RAG [15] 0.679 0.450 0.541 0.603 0.682 0.621 0.641 0.494
FFA-IR [30] 0.668 0.650 0.6600.837 0.942 0.945 0.9470.934
Retina-RAG (Ours) 0.778 0.795 0.7310.774 0.949 0.947 0.9480.897
Table 2.Report generation quality on the RDD test set denoted by NLP metrics for
Retina-RAG and other SOTA methods.
Method BLEU-4 ROUGE-1 ROUGE-L SBERT Clin-BERT
[35] [36] [36] [37] [38]
ZS Qwen [29] 0.014 0.175 0.163 0.764 0.921
MMed-RAG [15] 0.082 0.365 0.353 0.786 0.923
FFA-IR [30]0.235 0.5290.383 0.8690.982
Retina-RAG (Ours) 0.194 0.4610.438 0.8840.975
Cost impact:Retina-RAG is designed for resource-constrained clinical deploy-
ment. The dual-branch classifiers are fine-tuned via linear probing on either
single GPU or Google Collab TPU. Qwen2.5-VL-7B is adapted via LoRA (Un-
sloth), training only 1.23% of parameters (103M/8.4B) in 367 minutes with
10.5GB peak memory. The full pipeline runs on a single GPU at<$0.0002per
report, [28], and to the best of our knowledge, orders of magnitude cheaper than
proprietary models(∼$0.01−$0.05per query), supporting scalable screening
in low-resource settings.
Ablation Studies:We ablate the system into five components: (1) frozen VLM
baseline, (2) fine-tuned VLM only, (3) classifier+frozen VLM, (4) classifier+
fine-tuned VLM, and (5) the full Retina-RAG model with retrieval. Adding the
classifier substantially improves DR grading performance, fine-tuning the VLM
improves report quality, and incorporating retrieval provides the final gain in
diagnostic consistency. We evaluate additional classifier in place of FLAIR, in-
cluding a single VGG-16 [32] and a dual GoogleNet+ResNet-18 [31] configura-
tion with an RBF-SVM [25] layer at the end for classification. The GoogleNet
and ResNet-18 are trained similar to the Retinal classifier.
4 Conclusion
We present Retina-RAG, a modular framework for joint retinal diagnosis and
long-form report generation, demonstrating strong performance at low computa-
tional cost at<2.23% parameter tuning. A primary limitation is that evaluation

8
Fig. 2.QualitativeexamplefromRDDtestdataset(ImageID00804)withgroundtruth
and Retina-RAG predicted report. Image indicates DR Grade 2 (Moderate); ME Risk
0 (No Risk), which is reflected in the predicted report.
Table 3.Component-wise ablation on DR grading and edema detection. ZS= Zero
shot, FT=Fine-tuned.
Config Classifier DR Severity (5-class) ME (Binary)
Prec. Rec. F1 Prec. Rec. F1
ZS Qwen (Frozen VLM) – 0.076 0.204 0.096 0.743 0.722 0.732
FT VLM – 0.395 0.252 0.160 0.731 0.479 0.525
Classifier + ZS Qwen FLAIR 0.667 0.462 0.470 0.910 0.911 0.910
VGG16 0.537 0.396 0.411 0.941 0.941 0.938
RBF-SVM 0.691 0.459 0.470 0.899 0.899 0.899
Classifier + FT VLM FLAIR 0.782 0.752 0.698 0.881 0.876 0.878
VGG16 0.444 0.568 0.487 0.850 0.861 0.846
RBF-SVM 0.682 0.701 0.674 0.860 0.861 0.861
Retina-RAG (Ours) FLAIR0.778 0.795 0.731 0.949 0.947 0.948
is conducted on a single benchmark dataset, which may not fully reflect variabil-
ity across imaging devices, patient populations, and grading protocols. Future
work will investigate cross-dataset generalization and prospective clinical vali-
dation.

9
Table 4.Component-wise ablation on report generation using standard NLP metrics.
ZS=Zero shot, FT=Fine-tuned.
Config Classifier BLEU-4 ROUGE-1 ROUGE-L SBERT Clin-BERT
[35] [36] [36] [37] [38]
ZS Qwen (Frozen VLM) – 0.014 0.175 0.163 0.764 0.921
FT VLM – 0.114 0.342 0.333 0.840 0.960
Classifier + ZS Qwen FLAIR 0.016 0.166 0.158 0.695 0.895
VGG16 0.019 0.176 0.167 0.699 0.898
RBF-SVM 0.016 0.170 0.162 0.700 0.896
Classifier + FT VLM FLAIR 0.186 0.439 0.422 0.881 0.975
VGG16 0.165 0.410 0.395 0.863 0.970
RBF-SVM 0.186 0.419 0.412 0.871 0.971
Retina-RAG (Ours) FLAIR0.194 0.461 0.438 0.884 0.975
References
1. Teo, Z.L. et al.: Global prevalence of diabetic retinopathy and projection of burden
through 2045: systematic review and meta-analysis.Ophthalmology128(11), 1580–
1591 (2021)
2. Pan, J. et al.: Global, regional and national burden of blindness and vision loss
attributable to diabetic retinopathy, 1990–2021: a systematic analysis for the
Global Burden of Disease Study 2021.Diabetes, Obesity and Metabolism(2025).
https://doi.org/10.1111/dom.16588
3. Tan, M., Le, Q.V.: EfficientNet: Rethinking model scaling for convolutional neural
networks. In:Proceedings of the 36th International Conference on Machine Learning
(ICML), pp. 6105–6114 (2019)
4. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In:
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 770–778 (2016)
5. Humayun, M. et al.: Enhancing diabetic retinopathy classification using deep learn-
ing.Digital Health9(2023). https://doi.org/10.1177/20552076231203676
6. Arora, L. et al.: Ensemble deep learning and EfficientNet for accurate diagnosis of
diabetic retinopathy.Scientific Reports14, 30554 (2024)
7. Xu, Y. et al.: A hybrid neural network approach for classifying diabetic retinopathy
subtypes.Frontiers in Medicine10, 1293019 (2024)
8. Abràmoff, M.D., Lavin, P.T., Birch, M., Shah, N., Folk, J.C.: Pivotal
trial of an autonomous AI-based diagnostic system for detection of dia-
betic retinopathy in primary care offices. npj Digital Medicine1, 39 (2018).
https://doi.org/10.1038/s41746-018-0040-6
9. Li, C. et al.: LLaVA-Med: Training a large language-and-vision assistant for
biomedicine in one day. In:NeurIPS Datasets and Benchmarks Track(2023)
10. Gu, J. et al.: MedVH: Toward systematic evaluation of hallucination for large
vision language models in the medical context.Advanced Intelligent Systems(2025).
https://doi.org/10.1002/aisy.202500255
11. Chen, J. et al.: Detecting and evaluating medical hallucinations in large vision
language models. In:ICLR(2025)

10
12. He,J.etal.:PeFoMed:Parameterefficientfine-tuningofmultimodallargelanguage
models for medical imaging.arXiv preprint arXiv:2401.02797(2024)
13. Lewis, P. et al.: Retrieval-augmented generation for knowledge-intensive NLP
tasks.In:Advances in Neural Information Processing Systems, vol. 33,pp. 9459–9474
(2020)
14. Xia, P. et al.: RULE: Reliable multimodal RAG for factuality in medical vision
language models. In:EMNLP, pp. 1081–1093 (2024)
15. Xia, P. et al.: MMed-RAG: Versatile multimodal RAG system for medical vision
language models. In:ICLR(2025). arXiv:2410.13085
16. Abdalkader, M.: Retinal Disease Detection Dataset. Kaggle (2023).
https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection
17. Decencière, E. et al.: Feedback on a publicly distributed image database:
the Messidor database.Image Analysis and Stereology33(3), 231–234 (2014).
https://doi.org/10.5566/ias.1155
18. Porras, A.R. et al.: FLAIR: A foundation model for retinal image analysis.arXiv
preprintarXiv:2501.09706 (2025)
19. Wilkinson, C.P., Ferris, F.L., Klein, R.E., et al.: Proposed international clinical
diabetic retinopathy and diabetic macular edema disease severity scales. Ophthal-
mology110(9), 1677–1682 (2003). https://doi.org/10.1016/S0161-6420(03)00475-5
20. American Academy of Ophthalmology: Diabetic Retinopathy Pre-
ferred Practice Pattern®. Ophthalmology127(1), P66–P145 (2020).
https://doi.org/10.1016/j.ophtha.2019.09.025
21. Johnson, J. et al.: Billion-scale similarity search with GPUs.IEEE Transactions
on Big Data7(3), 535–547 (2019)
22. Wang, P. et al.: Qwen2-VL: Enhancing vision-language model’s perception of the
world at any resolution.arXiv preprintarXiv:2409.12191 (2024)
23. Han, D., Han, M.: Unsloth: Efficient LLM fine-tuning (2024).
https://github.com/unslothai/unsloth
24. Hu, E. et al.: LoRA: Low-rank adaptation of large language models. In:ICLR
(2022)
25. Behera, M.K., Mishra, R., Ransingh, A., Chakravarty, S.: Prediction of different
stages in diabetic retinopathy from retinal fundus images using radial basis function
based SVM. Indian Journal of Science and Technology13(20), 2030–2040 (2020).
https://doi.org/10.17485/IJST/v13i20.322
26. Hartsock, I., Rasool, G.: Vision-language models for medical report generation and
visual question answering: a review.Frontiers in Artificial Intelligence7(2024)
27. Dettmers, T. et al.: QLoRA: Efficient finetuning of quantized LLMs. In:NeurIPS
(2023)
28. Hyperbolic Labs: Serverless inference pricing (2025).
https://docs.hyperbolic.xyz/docs/hyperbolic-ai-inference-pricing
29. Bai,S.etal.:Qwen2.5-VLtechnicalreport.arXiv preprintarXiv:2502.13923(2025)
30. Li, M. et al.: FFA-IR: Towards an explainable and reliable medical report genera-
tion benchmark. In:Thirty-fifth Conference on NeurIPS Datasets and Benchmarks
Track(2021)
31. Butt,M.M.,Iskandar,D.N.F.A.,Abdelhamid,S.E.,Latif,G.,Alghazo,R.:Diabetic
retinopathy detection from fundus images of the eye using hybrid deep learning fea-
tures. Diagnostics12(7), 1731 (2022). https://doi.org/10.3390/diagnostics12071731
32. da Rocha, D.A., Ferreira, F.M.F., Peixoto, Z.M.A.: Diabetic retinopathy classifica-
tion using VGG16 neural network. Research on Biomedical Engineering38, 761–772
(2022). https://doi.org/10.1007/s42600-022-00200-8

11
33. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo,
B.: Swin transformer: Hierarchical vision transformer using shifted win-
dows. In: Proceedings of the IEEE/CVF ICCV, pp. 10012–10022 (2021).
https://doi.org/10.1109/ICCV48922.2021.00986
34. Ouyang, L., Wu, J., Jiang, X., et al.: Training language models to follow instruc-
tions with human feedback. In: NeurIPS, vol. 35, pp. 27730–27744 (2022)
35. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: BLEU: A method for
automatic evaluation of machine translation. In: Proceedings of the 40th
Annual Meeting of the ACL, pp. 311–318. ACL, Philadelphia (2002).
https://doi.org/10.3115/1073083.1073135
36. Lin, C.Y.: ROUGE: A package for automatic evaluation of summaries. In: Text
Summarization Branches Out: Proceedings of the ACL-04 Workshop, pp. 74–81.
ACL, Barcelona (2004)
37. Reimers, N., Gurevych, I.: Sentence-BERT: Sentence embeddings using siamese
BERT-networks. In: Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992. ACL, Hong Kong
(2019). https://doi.org/10.18653/v1/D19-1410
38. Huang, K., Altosaar, J., Ranganath, R.: ClinicalBERT: Modeling clinical notes
and predicting hospital readmission. arXiv preprint arXiv:1904.05342 (2019)