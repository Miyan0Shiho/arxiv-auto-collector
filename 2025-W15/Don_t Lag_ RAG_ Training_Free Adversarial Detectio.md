# Don't Lag, RAG: Training-Free Adversarial Detection Using RAG

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Published**: 2025-04-07 09:14:47

**PDF URL**: [http://arxiv.org/pdf/2504.04858v1](http://arxiv.org/pdf/2504.04858v1)

## Abstract
Adversarial patch attacks pose a major threat to vision systems by embedding
localized perturbations that mislead deep models. Traditional defense methods
often require retraining or fine-tuning, making them impractical for real-world
deployment. We propose a training-free Visual Retrieval-Augmented Generation
(VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial
patch detection. By retrieving visually similar patches and images that
resemble stored attacks in a continuously expanding database, VRAG performs
generative reasoning to identify diverse attack types, all without additional
training or fine-tuning. We extensively evaluate open-source large-scale VLMs,
including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside
Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO
model achieves up to 95 percent classification accuracy, setting a new
state-of-the-art for open-source adversarial patch detection. Gemini-2.0
attains the highest overall accuracy, 98 percent, but remains closed-source.
Experimental results demonstrate VRAG's effectiveness in identifying a variety
of adversarial patches with minimal human annotation, paving the way for
robust, practical defenses against evolving adversarial patch attacks.

## Full Text


<!-- PDF content starts -->

Don’t Lag, RAG:
Training-Free Adversarial Detection Using RAG
Roie Kazoom
Electrical and Computers Engineering
Ben Gurion University
Beer Sheba 84105, Israel
roieka@post.bgu.ac.ilRaz Lapid
DeepKeep
Tel-Aviv, Israel
raz.lapid@deepkeep.ai
Moshe Sipper
Computer Science
Ben Gurion University
Beer Sheba 84105, Israel
sipper@bgu.ac.ilOfer Hadar
Electrical and Computers Engineering
Ben Gurion University
Beer Sheba 84105, Israel
hadar@bgu.ac.il
Abstract
Adversarial patch attacks pose a major threat to vision systems by embedding lo-
calized perturbations that mislead deep models. Traditional defense methods often
require retraining or fine-tuning, making them impractical for real-world deploy-
ment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG)
framework that integrates Vision-Language Models (VLMs) for adversarial patch
detection. By retrieving visually similar patches and images that resemble stored at-
tacks in a continuously expanding database, VRAG performs generative reasoning
to identify diverse attack types—all without additional training or fine-tuning. We
extensively evaluate open-source large-scale VLMs—including Qwen-VL-Plus,
Qwen2.5-VL-72B, and UI-TARS-72B-DPO—alongside Gemini-2.0, a closed-
source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to
95% classification accuracy, setting a new state-of-the-art for open-source adver-
sarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98%, but
remains closed-source. Experimental results demonstrate VRAG’s effectiveness in
identifying a variety of adversarial patches with minimal human annotation, paving
the way for robust, practical defenses against evolving adversarial patch attacks.
1 Introduction
Deep learning models, particularly convolutional neural networks (CNNs) [ 23,18,47] and vision
transformers (ViTs) [ 12], have demonstrated remarkable success in computer vision tasks such as
object detection [ 15,43,42], image classification [ 23,12], and segmentation [ 34,46]. However,
despite advances, these models remain highly vulnerable to adversarial attacks [ 49,24,25,2,16,35,
50, 27], where small perturbations or carefully crafted patches manipulate predictions.
Adversarial patch attacks [ 6,28,33,54] introduce localized perturbations that persist across different
transformations, making them significantly more challenging to mitigate using conventional defense
mechanisms [ 53]. Unlike traditional adversarial perturbations that introduce subtle noise across an
image, adversarial patches are structured, high-magnitude perturbations, which are often physically
realizable. These patches can be printed, placed in real-world environments, and still cause misclassi-
fication or mislocalization in deployed deep learning models. Their adversarial effect remains robust
under different lighting conditions, transformations, and occlusions, allowing them to be successfully
Preprint. Under review.arXiv:2504.04858v1  [cs.AI]  7 Apr 2025

Is there an adversarial 
patch in this image?
No.
VLMIs there an adversarial 
patch in this image?
Retrieving similar 
images and editing 
prompt…VLM
   <Image>                                <Patch>
   <Image>                                <Patch>
Is there an adversarial 
patch in this image?
Yes.
VLM
Visual 
RAG DBIs there an adversarial 
patch in this image?
Retrieving similar 
images and editing 
prompt…VLM
No.
VLM
Visual 
RAG DB
Is there an adversarial 
patch in this image?
Figure 1: Illustration of three different settings for detecting adversarial patches. (Left) The zero-
shot baseline, in which the model is directly prompted to determine if the image is adversarial but
incorrectly concludes it is benign. (Center) Our VRAG-based approach on a benign image; as the
database does not contain benign exemplars, no relevant references are retrieved. Consequently, the
classification relies solely on the prompt content and remains accurate. (Right) Our VRAG-based
approach on an adversarial image, which leverages relevant references from the database to enhance
the prompt, ultimately yielding a correct detection of the adversarial patch.
deployed in real world scenarios [ 32,11]. Furthermore, retraining-based defenses require extensive,
labeled adversarial data, which is expensive to obtain and generalizes poorly to novel attack strategies
[53].
Traditional adversarial detection methods typically fall into one of three categories, (1) supervised
learning-based defenses, (2) unsupervised defenses and (3) adversarial training. Supervised learning-
based defenses [ 39,38] use deep learning classifiers trained on labeled adversarial and non-adversarial
samples. These methods are data-dependent and do not adapt well to adversarial attacks outside
the training distribution. Unsupervised defenses [ 58,37,48,36], typically rely on analyzing the
intrinsic structure or distribution of unlabeled data to detect anomalous inputs. For example, Feature
Squeezing [ 58] reduces input dimensionality (e.g., through bit-depth reduction or smoothing) to
reveal suspicious high-frequency artifacts; [ 37] use deep generative models to flag inputs with high
reconstruction error as potential adversarial samples. Although these methods can detect novel or
previously unseen attack strategies without relying on adversarial labels, they often require carefully
chosen hyperparameters and remain vulnerable to adaptive attacks that mimic the statistics of benign
inputs. In contrast to the supervised detection methods, which separately classify inputs as adversarial
or benign, adversarial training [35,26] augments the training data with adversarial examples to
directly improve model robustness. Rather than solely learning to detect adversarial inputs, this
approach modifies the model parameters and decision boundaries to make correct classification more
likely under attack. However, adversarial training is computationally expensive and risks overfitting
to specific attack types, leading to weaker defenses against unseen attacks [29].
In this paper, we introduce a retrieval-augmented adversarial patch detection framework that dy-
namically adapts to evolving threats without necessitating retraining. The method integrates visual
retrieval-augmented generation (VRAG) with a vision-language model (VLM) for context-aware
2

detection. As illustrated in Figure 1, visually similar patches are retrieved from a precomputed
database using semantic embeddings from grid-based image regions, and structured natural language
prompts guide the VLM to classify suspicious patches.
This paper makes the following contributions:
1.A training-free retrieval-based pipeline that dynamically matches adversarial patches against
a precomputed (and expandable) database.
2.The integration of existing VLMs with generative reasoning for context-aware patch detec-
tion through structured prompts.
3.A comprehensive evaluation demonstrating robust detection across diverse adversarial patch
scenarios, all without additional training or fine-tuning.
Experimental results confirm that our retrieval-augmented detection approach not only outperforms
traditional classifiers, but also achieves state-of-the-art detection across a variety of threat scenarios.
This method offers higher accuracy and reduces dependence on labeled adversarial datasets, under-
scoring the practicality of incorporating retrieval-based strategies alongside generative reasoning to
develop scalable, adaptable defenses for real-world security applications [45].
2 Related Work
Adversarial attacks exploit neural network vulnerabilities through carefully crafted perturbations.
Early works focused on small, imperceptible ℓp-bounded perturbations such as FGSM [ 16] and
PGD [ 35]. In contrast, adversarial patch attacks apply localized, high-magnitude changes that remain
effective under transformations and pose a threat in real-world scenarios [19, 32, 11].
Defenses fall into reactive andproactive categories. Reactive methods like JPEG compression [ 13]
and spatial smoothing [ 57] attempt to remove adversarial patterns at inference time but struggle against
adaptive attacks. Diffusion-based methods, such as DIFFender [ 20] and purification models [ 30],
leverage generative models to restore clean content but are often computationally intensive.
Another line of work focuses on patch localization and segmentation , e.g., SAC [ 31], which detects
and removes patches using segmentation networks. These approaches are limited by their reliance
on training and struggle with irregular or camouflaged patches. PatchCleanser [ 55] offers certifiable
robustness but assumes geometrically simple patches.
Proactive defenses like adversarial training [ 53] aim to increase robustness through exposure to
adversarial examples. While effective against known attacks, they generalize poorly and are resource-
intensive.
We propose a retrieval-augmented framework that detects a wide range of patch types—including
irregular and naturalistic ones—without degrading input quality or relying on segmentation or
geometric assumptions. Our method leverages a diverse patch database and vision-language reasoning
to dynamically adapt to unseen attacks.
3 Preliminaries
We briefly review core paradigms relevant to our defense framework: vision-language foundation
models, zero- and few-shot learning, adversarial attacks and defenses, and RAG.
3.1 Vision-Language Foundation Models and Zero- and Few-Shot Learning
Foundation models leverage large-scale transformer architectures and self-attention to learn general-
purpose representations from massive image-text data. A typical VLM consists of two encoders, fθ
for images Iandgϕfor text T, projecting them into a shared embedding space:
EI=fθ(I), E T=gϕ(T), S (I, T) =EI·ET
∥EI∥∥ET∥. (1)
Models like CLIP [ 41] and Flamingo [ 1] align image-text pairs via contrastive objectives, enabling
flexible zero-shot capabilities:
g(I, Q)→A, (2)
3

where Qis a textual query and Ais the inferred label without explicit task-specific training. Few-shot
learning refines zero-shot by supplying a small support set {(I1, y1), . . . , (Ik, yk)}:
g 
I, Q{(Ii, yi)}k
i=1
→A, (3)
allowing adaptation to novel tasks with limited labeled data.
3.2 Adversarial Attacks and Defense Strategies
Adversarial Attacks. Formally, an adversary seeks a perturbation δsubject to ∥δ∥p≤ϵthat
maximizes a loss function ℓfor a model fθwith true label y:
δ∗= arg max
∥δ∥p≤ϵℓ 
fθ(I+δ), y
. (4)
Patch-based attacks instead replace a localized region using a binary mask M∈ {0,1}H×W:
I′=I⊙(1−M) +P⊙M, (5)
where Pis a high-magnitude patch. Such localized perturbations remain visually inconspicuous in
many practical settings [19, 32, 11].
Preprocessing and Detection. A common defense strategy is to apply a transformation g(·)to
I′, yielding g(I′), with the goal of suppressing adversarial noise (e.g., blurring, smoothing [ 22]).
Detection can be formulated by a function D 
g(I′)
∈ {0,1}that flags anomalous inputs based on
statistical or uncertainty-based criteria [7].
Generative Reconstruction. Diffusion-based defenses [ 20] iteratively denoise adversarial inputs by
reversing a noisy forward process:
xt=√αtxt−1+√
1−αtϵt, ϵ t∼ N(0, I), (6)
often guided by patch localization [ 31]. Although effective, these approaches can falter against
unseen attacks or large patch perturbations, making robust generalization challenging in practice.
3.3 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) integrates external knowledge into a generative model to
improve both its generative capacity and semantic coherence. Formally, given a query Q, the model
retrieves the top- kmost relevant documents or embeddings Rkfrom a database D:
Rk= arg max
Ri∈DS(Q, R i), (7)
where S(·,·)is a similarity function. The query Qis then combined with Rkwithin a generative
function:
A=G(Q, R k). (8)
In our approach, this retrieval phase facilitates access to known adversarial patches, thereby enabling a
more robust generative reasoning process. By incorporating historical data on diverse attack patterns,
RAG-based defenses can dynamically adapt to novel threats while sustaining high efficacy against
existing adversaries.
4 Methodology
This section details our VRAG-based approach for adversarial patch detection using a vision-language
model. We first describe the construction of a comprehensive adversarial patch database (§4.1), then
present our end-to-end detection pipeline (§4.2), and finally discuss how the framework generalizes to
diverse patch shapes (§A.6). To enable scalability, we parallelize patch embedding and augmentation—
see Appendix A.1 for runtime benchmarks across varying numbers of workers.
4

4.1 Database Creation
To handle a wide variety of adversarial patch attacks, we build a large-scale database of patched
images and their corresponding patch embeddings. We aggregate patches generated by SAC [ 31],
BBNP [ 28], and standard adversarial patch attacks [ 6], placing each patch onto diverse natural images
at random positions and scales. This process, summarized in 1, ensures that the database spans
different patch configurations and visual contexts.
Algorithm 1 Adversarial Patch Database Creation with Positional Augmentation
1:Input: Set of patches {Pi}m
i=1, set of natural images {Ij}q
j=1, embedding model f, grid size
n×n, number of placement variations A
2:Output: Database D
3:Initialize database D ← ∅
4:fori= 1tomdo
5: Compute patch embedding EPi=f(Pi)
6: Store (Pi, EPi)inD
7: forj= 1toqdo
8: fora= 1toAdo
9: Randomly select position (xa, ya)in image Ij
10: Apply patch Piat(xa, ya)to obtain patched image I(a)
j
11: Divide I(a)
jinto grid cells {C(a)
j,k}n2
k=1
12: fork= 1ton2do
13: Compute embedding E(a)
j,k=f(C(a)
j,k)
14: ifC(a)
j,koverlaps with Pithen
15: Store (C(a)
j,k, E(a)
j,k)inD
16: end if
17: end for
18: end for
19: end for
20:end for
21:return D
Concretely, each patched image is subdivided into an n×ngrid, yielding localized regions
{C1, . . . , C n2}that spatially partition the image. For each region Ci, we compute a dense visual
embedding using a pre-trained vision encoder f(·):
ECi=f(Ci),
which captures high-level semantic and structural features of the corresponding image patch. In
parallel, we encode each adversarial patch Pjinto its own latent representation EPj=f(Pj)to
ensure embeddings are in the same feature space. These patch embeddings act as keys, while the
embeddings of overlapping regions serve as their corresponding values in a key-value database. This
design enables efficient and scalable nearest-neighbor retrieval at inference time, allowing the system
to match visual evidence in test images with known adversarial patterns from the database.
4.2 VRAG-Based Detection Pipeline
System Overview. Our detection system (illustrated in Figure 2) identifies adversarial patches in a
query image by leveraging the patch database as retrieval context for a vision-language model. The
process involves four main steps:
1.Image Preprocessing: Divide the input image Iinto an n×ngrid of regions {C1, . . . , C n2}
to enable localized inspection of each part of the image.
2.Feature Extraction: Encode each region Ciinto an embedding Ei=f(Ci)using a
pre-trained vision encoder (e.g., CLIP). These embeddings capture high-level semantic
features.
5

3.Retrieval Step: For each Ei, perform a nearest-neighbor search in the patch database
D. Retrieve the top- kmost similar patch embeddings to form a context set Ri=
Top-k({d(Ei, EPj)}). Appendix A.3 presents an ablation study comparing cosine sim-
ilarity with alternative distance metrics for this retrieval step.
4.Generative Reasoning with a VLM: Combine each region Ciwith its retrieved examples
Riand short textual cues to construct a multimodal prompt. This prompt is passed to a
vision-language model g(·)to answer:
g(Ci)→“Does this region contain an adversarial patch?”
We summarize the overall detection procedure in 2.
Image 
Encoder
…
Visual 
RAG DB
VLM
“Based on the 
given context, 
this image was 
attacked”
Top−k  
⊕
Few-shot examples
Corresponding  
Patchx xContext
Figure 2: Overview of our VRAG framework for adversarial patch detection. Given a query image,
we extract grid-based embeddings and retrieve the top- kvisually similar adversarial patches from
our database. These patches and their associated attacked images form a few-shot context for a
vision-language model that decides whether the query contains an adversarial patch.
Algorithm 2 Adversarial Patch Detection via VRAG
1:Input: Image I, VLM V, Database D, Embedding function f, threshold τ, top-mpatches, top- k
images
2:Output: Decision: Attacked orNot Attacked
3:Divide Iinto grid cells {Ci}n2
i=1, compute embeddings Ei=f(Ci)
4:foreachEido
5: Compute max similarity Si= max Ed∈DEi·Ed
∥Ei∥∥Ed∥
6:end for
7:Select candidates C={Ci|Si≥τ}, choose top- mpatches
8:Retrieve top- ksimilar attacked images from D
9:Build context Twith top- mpatches and top- kimages as examples
10:Query VLM with T:R=V(T, I)
11:return R∈ {Attacked ,Not Attacked }
Decision Mechanism (Zero-Shot and Few-Shot). After retrieving similar patches and attacked
images, the VLM is prompted to judge the query image under zero-shot or few-shot conditions:
•Zero-Shot Detection: The model relies on pre-trained knowledge and textual prompts to
classify each region Cias adversarial or benign, without additional fine-tuning.
•Few-Shot Adaptation: A small, labeled set of adversarial examples, denoted as {Ai}, along
with their corresponding patches {Pi}, is incorporated into the retrieved context to refine
the model’s decision-making process. This integration enhances the model’s robustness to
previously unseen attacks by explicitly exposing the VLM to representative instances of
patch-induced behavior.
6

A sample query prompt for the VLM might be:
“Here are examples of adversarial patches: [Patch 1], [Patch 2].
Here are images that contain these patches: [Image 1], [Image
2]. Based on this context, does the following image contain an
adversarial patch? Answer ’yes’ or ’no’.”
The model’s answer is then used to decide whether the image is Attacked orNot Attacked .
Optimal Threshold Selection. We determine the optimal threshold based on ROC-AUC analysis of
cosine similarity scores computed from embedding vectors. Specifically, the optimal cosine similarity
threshold identified was 0.77, providing the best trade-off between sensitivity and specificity. We
observed that for thresholds approaching 1.0, the similarity criterion becomes overly permissive,
resulting in nearly every image retrieving similar images, thereby substantially increasing the false-
positive rate.
5 Experimental Evaluation
We conduct extensive experiments to assess the robustness and efficiency of our adversarial patch
detection framework across diverse datasets, models, attack types, and defenses, simulating realistic
deployment scenarios.
Vision Language Models. For generative reasoning, we use several VLMs g(·), including Qwen-VL-
Plus [8],Qwen2.5-VL-Instruct [9],UI-TARS-72B-DPO [44], and Gemini [10]. These were chosen
for their strong multimodal reasoning in zero- and few-shot settings. While Gemini 2.0 yields the
highest accuracy, it is proprietary. UI-TARS-72B-DPO , meanwhile, offers competitive performance
and sets a strong benchmark among open-source models.
Classification Models. To evaluate the impact of adversarial patches across diverse architectures, we
consider four representative image classification models: (1) ResNet-50 [18], (2) ResNeXt-50 [56], (3)
EfficientNet-B0 [51], and (4) ViT-B/16 [12]. These models span both convolutional and transformer-
based paradigms and offer a clear comparison across varying robustness profiles and architectural
biases. For all models, we report clean and attacked accuracies under each defense method, using the
same attack configuration and patch size distribution.
Datasets and Attacks. We evaluate on both synthetic and real-world patch benchmarks: (1) ImageNet-
Patch [40], a 50/50 balanced dataset of attacked and clean ImageNet samples, comprising 400
test images, where attacks are applied to exactly 50% of the data to ensure balanced evaluation;
and (2) APRICOT [31], a real-world dataset of 873 images, each containing a physically applied
adversarial patch. We test two strong attacks: the classical adversarial patch [ 6] targeting CNNs, and
PatchFool [ 14] targeting vision transformers. Patches are randomly placed and vary in size from
25×25to65×65.
Defense Mechanisms. We compare against several baselines: (1) JPEG compression [ 13], (2) Spatial
smoothing [ 57], (3) SAC [ 31], and (4) DIFFender [ 20], a recent diffusion-based approach. We also
evaluate a retrieval-only baseline that flags regions as adversarial based on visual similarity, without
using VLM reasoning.
Evaluation Protocol. OnImageNet-Patch , we report classification accuracy over a balanced 50/50
clean/attacked split. On APRICOT , we report binary accuracy (presence vs. absence of a patch) across
three settings: (1) Clean , (2) Undefended , and (3) Defended . Candidate regions are retrieved using
top-k= 2cosine similarity and verified via VLM prompts. Thresholds are calibrated on a held-out
validation set to ensure fair comparisons across all methods.
6 Results
Table 1 reports defense accuracy on the APRICOT dataset [ 5] under adversarial patches of varying
sizes ( 25×25to65×65). Traditional defenses like JPEG compression [ 13], spatial smoothing [ 57],
and SAC [ 31] yield limited robustness, especially as patch size increases. DIFFender [ 20] improves
performance but still falls short. Our method consistently outperforms all 0-shot baselines and scales
better with patch size. Even in the 0-shot setting, it achieves competitive accuracy, while the 4-shot
7

configuration delivers strong gains, outperforming all baselines. Confusion matrices in Appendix 7
further highlight its robustness and real-world applicability.
Table 1: Accuracy (%) on APRICOT [ 5] with adversarial patches of varying sizes. Methods are
evaluated in 0-shot (0S), 2-shot (2S), and 4-shot (4S) settings; methods without few-shot use show
“–”. Gray indicates the best 0S result, underline the second-best overall, and bold the best overall.
Method25×25 50 ×50 55 ×55 65 ×65
0S 2S 4S 0S 2S 4S 0S 2S 4S 0S 2S 4S
Undefended 34.59 – – 32.18 – – 30.24 – – 28.55 – –
JPEG [13] 29.35 – – 32.53 – – 35.28 – – 41.11 – –
Spatial Smoothing [57] 33.56 – – 36.19 – – 39.17 – – 42.26 – –
SAC [31] 45.93 – – 48.22 – – 49.14 – – 52.80 – –
DIFFender [20] 65.06 – – 66.32 – – 68.61 – – 70.90 – –
Baseline 56.81 – – 59.56 – – 60.59 – – 69.64 – –
Ours (Qwen-VL-Plus) 45.37 76.18 87.64 46.40 77.90 88.78 47.55 79.62 90.50 50.98 81.91 92.22
Ours (Qwen2.5-VL-72B) 47.37 78.18 89.64 48.40 79.90 90.78 49.55 81.62 92.50 52.98 83.91 94.22
Ours (UI-TARS-72B-DPO) 49.37 80.18 91.64 50.40 81.90 92.78 51.55 83.62 94.50 54.98 85.91 96.22
Ours (Gemini) 56.24 82.59 93.92 57.16 85.11 96.33 58.76 86.94 96.79 63.12 90.26 97.93
Table 2 shows defense accuracy under adversarial patches of varying sizes. Accuracy drops sharply
without defense. Traditional methods like JPEG compression [ 13], spatial smoothing [ 58], and
SAC [ 31] offer limited robustness, while DIFFender [ 20] performs better via generative reconstruction.
Our retrieval-only baseline surpasses these, underscoring the value of visual similarity. Combining
retrieval with VLM reasoning yields the best results—especially in the 4-shot setting, which nearly
restores clean accuracy under large patches.
Table 2: Accuracy (%) of four models under adversarial patch attacks of varying sizes. Each method
is evaluated under three configurations: 0-shot (0S), 2-shot (2S), and 4-shot (4S), reflecting increasing
levels of visual context provided to the vision-language model. For methods that do not support
few-shot adaptation, results for 2S and 4S are omitted and marked with “–”. Gray indicates the
best-performing method in the 0-shot setting, underline highlights the second-best overall result
across all configurations, and bold denotes the highest overall accuracy. This presentation enables a
clear comparison of zero- and few-shot performance across varying patch sizes and models.
Model Method Clean25×25 50 ×50 55 ×55 65 ×65
0S 2S 4S 0S 2S 4S 0S 2S 4S 0S 2S 4S
ResNet-50 [18]Undefended
97.507.50 – – 9.25 – – 8.75 – – 6.95 – –
JPEG [13] 50.75 – – 51.75 – – 49.25 – – 49.00 – –
Spatial Smoothing [57] 55.50 – – 58.25 – – 55.25 – – 50.75 – –
SAC [31] 64.75 – – 66.75 – – 68.00 – – 69.50 – –
Baseline 58.50 – – 59.75 – – 62.00 – – 62.50 – –
Ours (Qwen-VL-Plus) 49.75 70.00 85.25 54.00 73.00 86.50 62.50 75.00 87.25 79.00 79.50 88.00
Ours (Qwen2.5-VL-72B) 55.25 82.00 88.25 60.00 84.00 89.25 79.75 86.00 90.50 91.00 91.25 91.50
Ours (UI-TARS-72B-DPO) 54.50 83.00 89.50 55.50 87.75 90.50 57.50 86.25 89.75 57.50 87.50 94.00
Ours (Gemini) 56.25 87.25 93.25 58.50 89.75 93.75 59.75 90.25 96.25 60.25 91.25 99.25
ResNeXt-50 [56]Undefended
97.509.25 – – 11.00 – – 10.75 – – 8.95 – –
JPEG [13] 48.75 – – 50.75 – – 47.75 – – 46.50 – –
Spatial Smoothing [57] 55.75 – – 57.50 – – 55.75 – – 50.25 – –
SAC [31] 64.75 – – 66.25 – – 68.00 – – 66.75 – –
Baseline 56.50 – – 58.50 – – 60.25 – – 61.75 – –
Ours (Qwen-VL-Plus) 48.25 68.50 83.00 52.00 71.25 84.50 58.00 72.75 85.25 74.25 77.00 86.25
Ours (Qwen2.5-VL-72B) 53.25 78.25 85.75 58.50 80.75 87.00 76.00 84.00 88.25 89.25 90.00 90.75
Ours (UI-TARS-72B-DPO) 52.50 80.75 85.75 55.25 85.00 89.25 55.25 86.25 91.00 59.25 84.75 93.25
Ours (Gemini) 55.50 85.00 91.25 57.75 87.50 92.75 58.75 88.50 94.75 60.75 89.75 98.50
EfficientNet [51]Undefended
95.5024.25 – – 25.75 – – 24.00 – – 21.50 – –
JPEG [13] 51.00 – – 53.75 – – 50.75 – – 49.25 – –
Spatial Smoothing [57] 60.50 – – 63.25 – – 61.75 – – 57.50 – –
SAC [31] 58.25 – – 60.75 – – 63.25 – – 67.25 – –
Baseline 54.75 – – 56.75 – – 58.25 – – 61.00 – –
Ours (Qwen-VL-Plus) 50.25 69.25 84.00 53.00 72.25 85.50 59.50 74.00 86.25 76.00 78.75 87.75
Ours (Qwen2.5-VL-72B) 54.50 79.50 87.00 59.25 82.00 89.00 78.25 85.00 90.50 90.50 91.00 92.00
Ours (UI-TARS-72B-DPO) 49.75 80.50 85.25 52.25 83.00 88.75 54.75 82.75 91.00 57.75 86.00 95.00
Ours (Gemini) 53.00 84.25 91.25 55.50 85.75 93.50 57.00 88.00 95.75 59.75 89.75 97.50
ViT-B-16 [14]Undefended
97.7527.75 – – 29.25 – – 27.00 – – 24.25 – –
JPEG [13] 57.75 – – 58.75 – – 55.50 – – 51.00 – –
Spatial Smoothing [57] 66.75 – – 67.25 – – 64.00 – – 61.25 – –
SAC [31] 63.25 – – 64.75 – – 65.75 – – 69.25 – –
Baseline 59.50 – – 61.50 – – 62.75 – – 64.00 – –
Ours (Qwen-VL-Plus) 51.25 69.50 84.25 55.00 72.50 85.75 60.50 76.00 86.75 74.00 79.00 87.25
Ours (Qwen2.5-VL-72B) 56.75 78.75 87.00 60.75 81.00 88.75 78.00 84.50 90.50 90.25 91.00 91.75
Ours (UI-TARS-72B-DPO) 53.25 82.00 89.75 54.75 84.25 91.00 56.75 85.50 93.25 59.50 88.75 95.25
Ours (Gemini) 58.75 86.75 93.50 60.75 89.00 95.25 61.25 90.75 98.75 63.00 93.00 99.00
8

We further analyze the effect of prompt design on detection performance. As shown in Appendix 6,
incorporating visual examples of both adversarial patches and attacked images into the prompt
significantly improves detection accuracy, with the combined prompt format achieving the best
results across multiple models and patch sizes. This finding highlights the importance of structured,
context-rich prompts in maximizing the reasoning capabilities of vision-language models (VLMs).
Specifically, prompts that present both the cause (adversarial patch) and the effect (altered image
behavior) enable the VLM to better associate visual cues with adversarial intent, even in zero-shot
settings. This insight suggests that prompt engineering is not merely a cosmetic component but
a critical design factor in VLM-driven adversarial detection pipelines. It also opens the door to
automated or learned prompt optimization strategies that could further boost performance under
different deployment scenarios.
Additionally, Appendix A.8 presents a comprehensive ablation study that quantifies the impact of
key system components. We analyze the trade-offs introduced by retrieval strategy choices (e.g.,
key/value formulation, embedding granularity), prompt formulations (e.g., descriptive vs. direct),
few-shot context sizes (0-shot, 2-shot, 4-shot), and inference-time efficiency. These experiments
offer actionable insights into which design choices yield the best accuracy-performance trade-off and
help identify bottlenecks in system scalability. Together, these findings reinforce the critical role of
retrieval and prompt design in enabling robust, generalizable adversarial patch detection without the
need for retraining.
7 Discussion and Conclusion
We introduced a training-free framework for adversarial patch detection that integrates visual retrieval-
augmented generation (VRAG) with vision-language models (VLMs). By leveraging a precomputed
and expandable database of diverse adversarial patches, our method enables dynamic retrieval and
context-aware reasoning without any model retraining or fine-tuning. This makes our approach both
scalable and deployment-ready in dynamic or resource-constrained environments. In contrast to many
prior defenses that rely on task-specific training regimes or assumptions about patch geometry, our
method generalizes effectively to a broad range of patch types—including naturalistic, camouflaged,
and physically realizable attacks.
Extensive evaluations on two complementary datasets— ImageNet-Patch , a synthetic benchmark
with clean/attacked image pairs, and APRICOT , a real-world dataset with 873 physically attacked
images—demonstrate the robustness of our framework. Across varying patch sizes and attack
methods, our method consistently outperforms traditional defenses such as JPEG compression [13],
spatial smoothing [ 57], SAC [ 31], and DIFFender [ 20], as well as a retrieval-only baseline that lacks
the reasoning capabilities of VLMs. Our full system achieves detection rates of up to 98%, and
crucially, maintains performance as the threat severity increases.
Beyond raw accuracy, we conducted thorough ablation studies (Appendix A.2) to isolate the con-
tributions of retrieval strategies, similarity metrics, prompt engineering, and few-shot context size.
These experiments highlight the importance of structured prompts and representative visual context
in enabling reliable VLM-based reasoning. We also report inference-time performance and paral-
lelization trade-offs to assess real-world feasibility. Appendix 6 and Appendix A.6 provide qualitative
comparisons, confusion matrices, and generalization analysis to diverse patch shapes and designs,
further reinforcing the robustness of our method.
Limitations and Future Work. While effective, our method currently assumes access to a represen-
tative patch database. Future work will focus on automatically identifying and augmenting missed
or novel adversarial patterns using generative models and self-supervised learning. We also aim to
incorporate uncertainty quantification into VLM outputs to better handle ambiguous or borderline
cases. Furthermore, improving inference speed—particularly for high-resolution images and real-time
applications—remains an important direction for deployment at scale.
Conclusion. Our VRAG-based framework combines retrieval-based search with generative vision-
language reasoning to offer a robust, adaptive, and training-free solution to adversarial patch detection.
It achieves high accuracy, generalizes across patch types, and requires minimal supervision—making
it a practical and scalable defense strategy for modern vision systems.
9

References
[1]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson,
Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual
language model for few-shot learning. Advances in neural information processing systems , 35:
23716–23736, 2022.
[2]Tal Alter, Raz Lapid, and Moshe Sipper. On the robustness of kolmogorov-arnold networks: An
adversarial perspective. Transactions on Machine Learning Research , 2025. ISSN 2835-8856.
URLhttps://openreview.net/forum?id=uafxqhImPM .
[3]Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang
Zhou, and Jingren Zhou. Qwen-VL: A versatile vision-language model for understanding,
localization, text reading, and beyond, 2024. URL https://openreview.net/forum?id=
qrGjFJVl3m .
[4]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
[5]A. Braunegg, A. Chakraborty, M. Krumdick, N. Lape, S. Leary, K. Manville, E. Merkhofer,
L. Strickhart, and M. Walmer. Apricot: A dataset of physical adversarial attacks on object
detection. https://arxiv.org/abs/1912.08166 , 2020.
[6]Tom B Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. Adversarial
patch. arXiv preprint arXiv:1712.09665 , 2017.
[7]T. J. Chua, W. Yu, C. Liu, and J. Zhao. Detection of uncertainty in exceedance of threshold
(duet): An adversarial patch localizer. In IEEE/ACM International Conference on Big Data
Computing, Applications and Technologies (BDCAT) , 2022.
[8]Alibaba Cloud. Qwen-vl: A vision-language model from alibaba cloud, 2023. URL https:
//huggingface.co/Qwen/Qwen-VL .
[9]Alibaba Cloud. Qwen2.5-vl-72b-instruct: A large multimodal model, 2024. URL https:
//huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct .
[10] Google DeepMind. Gemini: Google’s most capable ai model, 2024. URL https://deepmind.
google/technologies/gemini/ .
[11] B. Deng, D. Zhang, F. Dong, J. Zhang, M. Shafiq, and Z. Gu. Rust-style patch: A physical and
naturalistic camouflage attacks on object detector for remote sensing images. Remote Sensing ,
2023.
[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image
recognition at scale. arXiv preprint arXiv:2010.11929 , 2020.
[13] Gintare Karolina Dziugaite, Zoubin Ghahramani, and Daniel M. Roy. A study of the effect of
jpg compression on adversarial images. arXiv preprint , arXiv:1608.00853, 2016.
[14] Yonggan Fu. Patch-fool: Are vision transformers always robust against adversarial perturba-
tions?. arXiv , 2022.
[15] Ross Girshick. Fast r-cnn. arXiv preprint arXiv:1504.08083 , 2015.
[16] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversar-
ial examples. arXiv preprint arXiv:1412.6572 , 2014.
[17] Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong
Liao, Yao Qin, V olker Tresp, and Philip Torr. A systematic survey of prompt engineering on
vision-language foundation models. arXiv preprint arXiv:2307.12980 , 2023.
10

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition ,
pages 770–778, 2016.
[19] R. H. Hwang, J. Y . Lin, S. Y . Hsieh, H. Y . Lin, and C. L. Lin. Adversarial patch attacks on
deep-learning-based face recognition systems using generative adversarial networks. Sensors ,
2023.
[20] C. Kang, Y . Dong, Z. Wang, S. Ruan, H. Su, and X. Wei. Diffender: Diffusion-based adversarial
defense against patch attacks. arXiv preprint arXiv:2306.09124 , 2024.
[21] Roie Kazoom, Raz Birman, and Ofer Hadar. Meta classification model of surface appearance
for small dataset using parallel processing. Electronics , 11(21):3426, 2022.
[22] T. Kim, Y . Yu, and Y . M. Ro. Defending physical adversarial attack on object detection via
adversarial patch-feature energy. In Proceedings of the 30th ACM International Conference on
Multimedia , 2022.
[23] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep
convolutional neural networks. Advances in neural information processing systems , 25, 2012.
[24] Raz Lapid and Moshe Sipper. I see dead people: Gray-box adversarial attack on image-to-text
models. In Joint European Conference on Machine Learning and Knowledge Discovery in
Databases , pages 277–289. Springer, 2023.
[25] Raz Lapid, Zvika Haramaty, and Moshe Sipper. An evolutionary, gradient-free, query-efficient,
black-box algorithm for generating adversarial instances in deep convolutional neural networks.
Algorithms , 15(11):407, 2022.
[26] Raz Lapid, Almog Dubin, and Moshe Sipper. Fortify the guardian, not the treasure: Resilient
adversarial detectors. Mathematics , 12(22):3451, 2024.
[27] Raz Lapid, Ron Langberg, and Moshe Sipper. Open sesame! universal black-box jailbreaking
of large language models. In ICLR 2024 Workshop on Secure and Trustworthy Large Language
Models , 2024. URL https://openreview.net/forum?id=0SuyNOncxX .
[28] Raz Lapid, Eylon Mizrahi, and Moshe Sipper. Patch of invisibility: Naturalistic black-box
adversarial attacks on object detectors. In 6th Workshop on Machine Learning for Cybersecurity,
part of ECMLPKDD 2024 , 2024.
[29] J. Liang, R. Yi, J. Chen, Y . Nie, and H. Zhang. Securing autonomous vehicles visual perception:
Adversarial patch attack and defense schemes with experimental validations. IEEE Transactions
on Intelligent Vehicles , 2024.
[30] S. Y . Lin, E. Chu, C. H. Lin, J. C. Chen, and J. C. Wang. Diffusion to confusion: Naturalistic
adversarial patch generation based on diffusion model for object detector. arXiv preprint
arXiv:2307.08076 , 2023.
[31] J. Liu, A. Levine, C. H. L. Lau, R. Chellappa, and S. Feizi. Segment and complete: Defending
object detectors against adversarial patch attacks with robust patch detection. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) , pages 14973–14982, 2022.
[32] T. Liu, C. Yang, X. Liu, R. Han, and J. Ma. Rpau: Fooling the eyes of uavs via physical
adversarial patches. IEEE Transactions on Intelligent Transportation Systems , 2024.
[33] Xin Liu, Huanrui Yang, Ziwei Liu, Linghao Song, Hai Li, and Yiran Chen. Dpatch: An
adversarial patch attack on object detectors. arXiv preprint arXiv:1806.02299 , 2018.
[34] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for se-
mantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern
recognition , pages 3431–3440, 2015.
[35] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083 ,
2017.
11

[36] Eylon Mizrahi, Raz Lapid, and Moshe Sipper. Pulling back the curtain: Unsupervised adversarial
detection via contrastive auxiliary networks. arXiv preprint arXiv:2502.09110 , 2025.
[37] Nicolas Papernot and Patrick McDaniel. Deep k-nearest neighbors: Towards confident, inter-
pretable and robust deep learning. arXiv preprint arXiv:1803.04765 , 2018.
[38] Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation
as a defense to adversarial perturbations against deep neural networks. In 2016 IEEE symposium
on security and privacy (SP) , pages 582–597. IEEE, 2016.
[39] Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, and Yehudit Aperstein. XAI-based
detection of adversarial attacks on deepfake detectors. Transactions on Machine Learning Re-
search , 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=7pBKrcn199 .
[40] Maura Pintor. Imagenet-patch: A dataset for benchmarking machine learning robustness against
adversarial patches. Pattern Recognition , 2023.
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning ,
pages 8748–8763. PMLR, 2021.
[42] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified,
real-time object detection. In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 779–788, 2016.
[43] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time
object detection with region proposal networks. IEEE transactions on pattern analysis and
machine intelligence , 39(6):1137–1149, 2016.
[44] ByteDance Research. Ui-tars-72b-dpo: A 72b parameter vision-language model for ui under-
standing. https://huggingface.co/bytedance-research/UI-TARS-72B-DPO , 2024.
Accessed: 2024-04-01.
[45] Kazoom Roie, Raz Birman, and Ofer Hadar. Improving the robustness of object detection and
classification ai models against adversarial patch attacks. arXiv preprint arXiv:2403.12988 ,
2024.
[46] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
for biomedical image segmentation. In Medical image computing and computer-assisted
intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9,
2015, proceedings, part III 18 , pages 234–241. Springer, 2015.
[47] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
image recognition. arXiv preprint arXiv:1409.1556 , 2014.
[48] Angelo Sotgiu, Ambra Demontis, Marco Melis, Battista Biggio, Giorgio Fumera, Xiaoyi Feng,
and Fabio Roli. Deep neural rejection against adversarial examples. EURASIP Journal on
Information Security , 2020:1–10, 2020.
[49] C Szegedy. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 , 2013.
[50] Snir Vitrack Tamam, Raz Lapid, and Moshe Sipper. Foiling explanations in deep neural
networks. Transactions on Machine Learning Research , 2023. ISSN 2835-8856. URL
https://openreview.net/forum?id=wvLQMHtyLk .
[51] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural
networks. In International conference on machine learning , pages 6105–6114. PMLR, 2019.
[52] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly
capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
12

[53] H. Wei, H. Tang, X. Jia, Z. Wang, H. Yu, Z. Li, S. Satoh, L. Van Gool, and Z. Wang. Physical
adversarial attack meets computer vision: A decade survey. arXiv preprint arXiv:2209.15179 ,
2022.
[54] Xingxing Wei, Yao Huang, Yitong Sun, and Jie Yu. Unified adversarial patch for cross-modal
attacks in the physical world. In Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 4445–4454, 2023.
[55] Chong Xiang, Saeed Mahloujifar, and Prateek Mittal. {PatchCleanser }: Certifiably robust
defense against adversarial patches for any image classifier. In 31st USENIX security symposium
(USENIX Security 22) , pages 2065–2082, 2022.
[56] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual
transformations for deep neural networks. In Proceedings of the IEEE conference on computer
vision and pattern recognition , pages 1492–1500, 2017.
[57] Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in
deep neural networks. arXiv preprint , arXiv:1704.01155, 2017.
[58] Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples
in deep neural networks. In Proceedings 2018 Network and Distributed System Security
Symposium . Internet Society, 2018.
A Appendix: Ablation Study
We perform all evaluations on the ImageNet-Patch [40] dataset.
A.1 Effect of Parallelization
Parallelization significantly improves the efficiency of adversarial patch database creation. Since
the application of each patch to each image—and the subsequent embedding computation—are
independent operations, the process can be parallelized across multiple workers [ 21]. This enables
rapid generation and encoding of large-scale patched image datasets.
In our setup, we applied adversarial patches to a collection of clean images, using a key-value
approach where each image was divided into a 5×5grid. Patch embeddings served as keys, while
embeddings of image regions acted as values for retrieval. The end result was a database of 3,500
patch-image pairs with corresponding embeddings. To evaluate scalability, we measured execution
time with varying levels of parallelism, confirming substantial speedups as the number of workers
increased.
Table 3: Execution time for adversarial patch detection with different numbers of workers. Results
are reported as mean ±standard deviation, in minutes.
Number of Workers Execution Time (min)
1 24.57 ±0.07
2 12.12 ±0.10
3 8.11 ±0.16
4 6.14 ±0.26
5 4.59 ±0.40
6 3.58 ±0.54
As shown in Table 3, using a single worker resulted in an average execution time of 24.57 minutes,
whereas increasing the number of workers to six reduced the execution time to 3.58 minutes, demon-
strating a 6.86 ×speedup. The results indicate that distributing the workload across multiple processes
significantly reduces execution time while maintaining detection accuracy.
These findings validate the effectiveness of parallelization in our method, allowing it to scale efficiently
for larger datasets. The speedup enables the rapid processing of extensive adversarial patch collections,
making real-time detection feasible.
13

A.2 Embedding Distance Analysis
We evaluate the effectiveness of our retrieval mechanism through an ablation study comparing several
distance metrics for nearest-neighbor retrieval, including cosine similarity, L1 distance, L2 distance,
and Wasserstein distance. All experiments in this subsection were conducted on the ImageNet-Patch
dataset. Rather than relying solely on cosine similarity for retrieving stored adversarial patches, we
also assess alternative metrics using embeddings extracted via CLIP [41].
Given an input image I, we partition it into grid-based regions and extract feature embeddings using
CLIP’s image encoder:
EI=f(I), E D={f(Di)|Di∈ D} , (9)
where f(·)denotes the CLIP embedding function and Drepresents the precomputed adversarial
patch database.
For cosine similarity-based retrieval, the similarity score is computed as:
S(EI, ED) =EI·ED
∥EI∥∥ED∥, (10)
with a stored adversarial patch retrieved if S(EI, ED)exceeds a similarity threshold τs.
We also evaluate L1 and L2 distances. The L1 distance is defined as:
dL1(EI, ED) =X
|EI−ED|, (11)
and the L2 distance is given by:
dL2(EI, ED) =∥EI−ED∥2. (12)
For both L1 and L2 distances, retrieval is triggered when the computed distance falls below a threshold
(τL1orτL2, respectively).
Additionally, we examine the Wasserstein distance, which measures the optimal transport cost
between distributions. For two distributions PandQover the embedding space, the Wasserstein
distance is defined as:
W(EI, ED) = inf
γE(x,y)∼γ[∥x−y∥], (13)
where γis a joint distribution with marginals PandQ. This metric quantifies the minimal effort
required to transport mass between the two embedding distributions.
We compare the retrieval effectiveness of these four metrics using Gemini-2.0 [ 10] for final classifica-
tion. The cosine similarity-based approach achieves the highest classification accuracy at 98.00%,
followed by L2 distance ( 89.75%), L1 distance ( 86.25%), and Wasserstein distance ( 84.25%). These
results are visualized in Figure 3a.
Cosine SimilarityL2 Distance L1 DistanceWasserstein Distance80.082.585.087.590.092.595.097.5100.0Accuracy (%)98.00
89.75
86.25
84.2598.50
88.2588.00
82.50Comparison of Retrieval Performance on Different Databases
ImageNet-Patch
APRICOT
(a) Retrieval performance using different distance met-
rics on CLIP embeddings.
1 2 3 4 5 6 7 8
Inference Time (Seconds)5060708090100Accuracy (%)
JPEGSpatial SmoothingSACDIFFenderOurs (Qwen-VL-Plus)Ours (Qwen2.5-VL-72B-Instruct)Ours (UI-TARS-72B-DPO)Ours (Gemini-2.0)Inference Time vs Accuracy for Different Defense Mechanisms(b) Inference time vs. accuracy for different defense
mechanisms.
These results indicate that cosine similarity most effectively captures the high-dimensional semantic
relationships essential for robust adversarial patch retrieval, while the alternative metrics, although
reasonable, perform less effectively—particularly the Wasserstein distance, which struggles to model
distributional similarity from limited embedding samples.
14

A.3 Inference Time Analysis
All experiments in this subsection were conducted on the ImageNet-Patch dataset. In addition to
detection performance, we assess the inference time required for each defense mechanism. For an
input image I, the processing time for a defense mechanism Dis defined as:
TD=1
NNX
i=1ti, (14)
where tiis the processing time for the i-th image and Nis the total number of test images.
We analyze the trade-off between inference time TDand classification accuracy AD, which is
calculated as:
AD=Ccorrect
Ctotal×100, (15)
withCcorrect representing the number of correctly classified images and Ctotalthe total number of
images.
As shown in Figure 3b, JPEG compression [ 13] and Spatial Smoothing [ 57] offer the fastest inference
times ( 0.92s and 0.97s, respectively), albeit with limited accuracy improvements ( 49.25% and
55.25%). SAC [ 31] requires 1.58s while achieving an accuracy of 68.00%, and DIFFender [ 20]
attains an accuracy of 70.90% with an inference time of 7.98s.
Our method, leveraging Qwen-VL-Plus [ 8], Qwen2.5-VL-72B-Instruct [ 9], UI-TARS-72B-DPO
[44], and Gemini-2.0 [ 10], achieves superior classification accuracy ( 79.00%,91.00%,95.00%, and
98.00%, respectively) with inference times of 1.97s,2.09s,2.17s, and 2.25s.
These findings highlight a clear performance–efficiency trade-off: higher detection accuracy generally
demands increased computational cost. Our approach effectively balances these aspects by leveraging
retrieval-augmented detection while maintaining inference times that remain competitive with existing
defense mechanisms.
A.4 Prompt Engineering Analysis
All experiments in this subsection were conducted on the ImageNet-Patch dataset. To investigate
the impact of prompt design [ 17] on adversarial patch detection, we conducted an ablation study
evaluating five distinct prompting strategies. Each strategy aims to guide the VLM in classifying
whether an image contains an adversarial patch. Given an input image I, the VLM is provided with a
textual prompt Tand returns a classification response:
R=V(T, I), (16)
where Vrepresents the VLM inference function.
To enhance context, we leverage a retrieved set of adversarial patch examples {P1, . . . , P m}, where
eachPiis an adversarial patch stored in the database, and a set of attacked images {I1, . . . , I k},
where each Ijis a full image containing an applied adversarial patch. These elements provide
additional visual references during inference.
The prompting strategies evaluated are as follows, along with the specific examples used:
1.Instruction-only: A generic instruction without examples:
“Adversarial physical attacks involve placing random patches
on images. You are an expert in identifying such patches.
Is the following image attacked? Answer ’yes’ or ’no’.”
2.Attacked Images: The instruction followed by examples of attacked images {I1, . . . , I k}:
“Here are examples of images that have been attacked: [Image
1], [Image 2], [Image 3]. Given the next image, is it
attacked? Answer ’yes’ or ’no’.”
3.Patch Examples: The instruction followed by examples of extracted adversarial patches
{P1, . . . , P m}:
15

“Here are examples of adversarial patches: [Patch 1], [Patch
2], [Patch 3]. Given the next image, is it attacked? Answer
’yes’ or ’no’.”
4.Chain-of-Thought (CoT): The instruction augmented with reasoning:
“Adversarial attacks often involve adding suspicious patches.
First, analyze if there are irregular regions. Then, decide
if an attack is present. Is the following image attacked?
Answer ’yes’ or ’no’.”
5.Combined (Final, Without CoT): The instruction with both attacked images and patch
examples:
“Adversarial physical attacks involve random patches on
images. You are an expert at detecting them. Here are
examples of adversarial patches: [Patch 1], [Patch 2].
Here are examples of attacked images: [Image 1], [Image
2]. Given the above context, is this image attacked? Please
answer ’yes’ or ’no’.”
To quantify the effectiveness of each prompt type, we measured the detection accuracy ATobtained
under each configuration. The final selected prompt, as presented in 2, corresponds to the Combined
(Final) strategy, which achieved the highest detection accuracy of 98.00%. The complete results are
summarized in Figure 4b, where we observe that simple instructional prompts result in low accuracy
(58.00%), while adding contextual examples (patches and attacked images) significantly improves
performance. The CoT-based prompt further enhances accuracy to 91.25%, whereas the combined
strategy achieves the highest overall detection rate.
This ablation study highlights that careful prompt engineering, particularly including few-shot visual
examples and reasoning, is critical for maximizing VLM-based adversarial patch detection.
A.5 Impact of Few-Shot Context Size on Classification Accuracy
All experiments in this subsection were conducted on the ImageNet-Patch dataset. To evaluate the
effect of context size on adversarial patch detection, we conducted an ablation study by varying
the number of few-shot examples provided to the VLM during inference. Let k∈ {0,1, . . . , 6}
denote the number of retrieved examples (i.e., the few-shot shots). For each k-shot configuration,
we measured the classification accuracy Akof the VLM in detecting adversarial patches across four
different models: Qwen-VL-Plus, Qwen2.5-VL-Instruct, UI-TARS-72B-DPO, and Gemini-2.0.
Figure 4a illustrates the trend of Akas a function of k. Across all models, we observe a consistent
improvement in detection accuracy with increasing values of k, indicating that providing more
contextual examples strengthens the model’s ability to generalize and distinguish adversarial patterns.
Notably, UI-TARS-72B-DPO consistently achieves intermediate performance, surpassing Qwen-
based models and closely approaching Gemini-2.0 accuracy.
0 1 2 3 4 5 6
K-Shots Used5060708090100Accuracy (%)
Few-Shot Accuracy vs. Shots per Model
Qwen-VL-Plus
Qwen2.5-VL-Instruct
UI-TARS-72B-DPO
Gemini-2.0
(a) Few-shot detection accuracy across varying context
sizes k.
Instruction-only Attacked Images Patch Examples Chain-of-Thought Combined (Final)020406080100Accuracy (%)58.00%69.50%86.75%91.25%98.00%Effect of Prompt Engineering on Accuracy(b) Effect of prompt engineering on adversarial patch
classification accuracy.
16

These results suggest that larger few-shot contexts allow the VLM to better align the input query with
prior adversarial patterns stored in the retrieval database. However, the performance gains tend to
plateau beyond k= 4, highlighting a saturation effect where additional examples yield diminishing
returns. The comparison also reveals that more capable VLMs (e.g., Gemini-2.0 and UI-TARS-72B-
DPO) benefit more rapidly from few-shot conditioning than smaller models such as Qwen-VL-Plus
and Qwen2.5-VL-Instruct, although Gemini-2.0 still demonstrates superior performance overall.
A.6 Generalization to Diverse Patch Shapes
Real-world adversarial patches appear in many shapes and textures, from geometric (square, round,
triangular) to naturalistic or camouflage-like forms. To ensure robustness against these diverse
patterns, we incorporate a range of patch types in the database creation phase. Concretely, each patch
Pi∈ P may be:
square ,round ,triangle ,realistic , . . .
Since detection relies on embedding-based similarity rather than geometric assumptions, unusual or
irregular patch shapes remain identifiable as long as their embeddings lie above a retrieval threshold τ.
In practice, this approach allows our VRAG-based framework to detect both canonical patches and
highly unobtrusive, adaptive adversarial artifacts designed to evade simpler defenses.
By collectively leveraging a rich database of patch embeddings, a retrieval-augmented paradigm, and
a capable vision-language model, our method achieves robust generalization in adversarial patch
detection across a wide spectrum of attack strategies.
Realistic Patches
 Round
 Square
 Triangle
Figure 5: Examples of adversarial patch masks used in our dataset. We consider four types: realistic,
round, square, and triangle. This diversity improves robustness across patch shapes.
A.7 Qualitative Results
In addition to quantitative evaluations, we present qualitative results highlighting the effectiveness
of our proposed framework compared to existing defenses. Figure 6 illustrates visual comparisons
across various defense mechanisms: Undefended, JPEG compression [ 13], Spatial Smoothing [ 57],
SAC [31], DIFFender [20], and our method.
Adversarial patches remain clearly visible and disruptive in both Undefended and JPEG-compressed
images, indicating that these methods fail to mitigate patch attacks effectively. SAC partially reduces
the visibility of adversarial patches but does not consistently eliminate them, often leaving residual
disruptions. DIFFender [ 20] demonstrates improved effectiveness compared to SAC by further
reducing patch visibility, though residual disturbances remain apparent.
In contrast, our method reliably identifies and neutralizes adversarial patches, effectively mitigating
their influence while preserving image integrity. However, our approach also has specific failure
17

Undefended JPEGSpatial 
SmoothingSAC Ours
 DIFFender
Figure 6: Qualitative comparison of different defense mechanisms. From left to right: Undefended,
JPEG compression [13], Spatial Smoothing [57], SAC [31], DIFFender [20] and our method.
modes, particularly evident when the adversarial patch blends seamlessly into the noisy background
of an image, matching its distribution. In such challenging cases (e.g., the last row of the right-hand
table in Figure 6), the model may struggle to accurately differentiate between patch and background
noise, highlighting a limitation to be addressed in future research.
A.8 Impact of Few-Shot Retrieval on VLM Accuracy
To further understand performance across different vision-language models (VLMs), Figure 7 shows
confusion matrices for Qwen-VL-Plus [ 3], Qwen2.5-VL-72B [ 4], UI-TARS-72B-DPO [ 44], and
Gemini-2.0 [ 52] under 0-shot, 2-shot, and 4-shot configurations. Increasing the number of retrieved
examples consistently improves both true-positive and true-negative rates. Notably, the 4-shot config-
uration with Gemini-2.0 yields near-perfect separation between adversarial and clean samples. While
18

Gemini-2.0 remains the top-performing model, UI-TARS-72B-DPO achieves highly competitive
results, outperforming all other open-source VLMs by a significant margin.
These findings highlight the power of retrieval-augmented prompting for adversarial patch detection—
especially when representative visual-textual context is injected via advanced VLMs.
Attack Not AttackAttack Not Attack190 210
192 2080-shot, Qwen-VL-Plus
Attack Not AttackAttack Not Attack224 176
182 2180-shot, Qwen2.5-VL-72B
Attack Not AttackAttack Not Attack240 160
150 2300-shot, UI-TARS-72B-DPO
Attack Not AttackAttack Not Attack210 190
124 2760-shot, Gemini-2.0-Pro
Attack Not AttackAttack Not Attack252 148
152 2482-shot, Qwen-VL-Plus
Attack Not AttackAttack Not Attack320 80
82 3182-shot, Qwen2.5-VL-72B
Attack Not AttackAttack Not Attack342 58
54 3462-shot, UI-TARS-72B-DPO
Attack Not AttackAttack Not Attack358 42
40 3602-shot, Gemini-2.0-Pro
Attack Not AttackAttack Not Attack312 88
80 3204-shot, Qwen-VL-Plus
Attack Not AttackAttack Not Attack368 32
40 3604-shot, Qwen2.5-VL-72B
Attack Not AttackAttack Not Attack372 28
24 3764-shot, UI-TARS-72B-DPO
Attack Not AttackAttack Not Attack390 10
2 3984-shot, Gemini-2.0-Pro
Predicted ClassActual Class
Figure 7: Confusion matrices across three few-shot configurations (rows) and four VLMs (columns).
Axes represent predicted and actual classes (“Attack” vs. “Not Attack”). Gemini-2.0 achieves the
best overall accuracy, while UI-TARS-72B-DPO offers the strongest open-source performance.
19