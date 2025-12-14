# Structure-Aware Feature Rectification with Region Adjacency Graphs for Training-Free Open-Vocabulary Semantic Segmentation

**Authors**: Qiming Huang, Hao Ai, Jianbo Jiao

**Published**: 2025-12-08 10:00:36

**PDF URL**: [https://arxiv.org/pdf/2512.07360v1](https://arxiv.org/pdf/2512.07360v1)

## Abstract
Benefiting from the inductive biases learned from large-scale datasets, open-vocabulary semantic segmentation (OVSS) leverages the power of vision-language models, such as CLIP, to achieve remarkable progress without requiring task-specific training. However, due to CLIP's pre-training nature on image-text pairs, it tends to focus on global semantic alignment, resulting in suboptimal performance when associating fine-grained visual regions with text. This leads to noisy and inconsistent predictions, particularly in local areas. We attribute this to a dispersed bias stemming from its contrastive training paradigm, which is difficult to alleviate using CLIP features alone. To address this, we propose a structure-aware feature rectification approach that incorporates instance-specific priors derived directly from the image. Specifically, we construct a region adjacency graph (RAG) based on low-level features (e.g., colour and texture) to capture local structural relationships and use it to refine CLIP features by enhancing local discrimination. Extensive experiments show that our method effectively suppresses segmentation noise, improves region-level consistency, and achieves strong performance on multiple open-vocabulary segmentation benchmarks.

## Full Text


<!-- PDF content starts -->

Structure-Aware Feature Rectification with Region Adjacency Graphs for
Training-Free Open-Vocabulary Semantic Segmentation
Qiming Huang, Hao Ai, Jianbo Jiao
The MIx Group, School of Computer Science
University of Birmingham
{qxh366, hxa456}@student.bham.ac.uk, j.jiao@bham.ac.uk
Abstract
Benefiting from the inductive biases learned from large-
scale datasets, open-vocabulary semantic segmentation
(OVSS) leverages the power of vision-language models,
such as CLIP , to achieve remarkable progress without re-
quiring task-specific training. However, due to CLIP’s pre-
training nature on image-text pairs, it tends to focus on
global semantic alignment, resulting in suboptimal perfor-
mance when associating fine-grained visual regions with
text. This leads to noisy and inconsistent predictions, par-
ticularly in local areas. We attribute this to a dispersed bias
stemming from its contrastive training paradigm, which is
difficult to alleviate using CLIP features alone. To address
this, we propose a structure-aware feature rectification ap-
proach that incorporates instance-specific priors derived
directly from the image. Specifically, we construct a region
adjacency graph (RAG) based on low-level features (e.g.
colour and texture) to capture local structural relationships
and use it to refine CLIP features by enhancing local dis-
crimination. Extensive experiments show that our method
effectively suppresses segmentation noise, improves region-
level consistency, and achieves strong performance on mul-
tiple open-vocabulary segmentation benchmarks. Project
page: https://qiming-huang.github.io/RAG-OVS/.
1. Introduction
Pretrained vision-language models such as CLIP [27] have
demonstrated remarkable performance in zero-shot and
open-vocabulary recognition tasks. Despite its effective-
ness in capturing global image-text alignment, CLIP suffers
from notable inductive biases at the local image level, lim-
iting its applicability to fine-grained visual understanding.
Specifically, CLIP was trained on image-text pairs without
explicit supervision, enforcing implicit alignment between
visual regions and textual descriptions. Consequently, it
captures coarse semantic correspondences rather than fine-
Figure 1.Illustration of the main idea and performance.High-
level feature region adjacency graphs (RAGs) introduce local
noise, while low-level colour-based RAGs maintain clean struc-
ture. The RAGs built on CLIP [27] and DINO [4] pretrained fea-
tures exhibit noisy and inconsistent connectivity in local regions
(see zoomed-in areas), when compared to the low-level based one.
This highlights the potential of low-level cues for tasks requiring
fine-grained local modelling,e.g. image segmentation.Bottom:
Comparison of average performance across multiple datasets us-
ing different features for RAG construction. C.-only: colour-based
features, and C. + G.: colour and texture features.
grained regional details, making it less effective for tasks re-
quiring high spatial granularity, such as training-free open-
vocabulary semantic segmentation (OVSS). This limitation
manifests as noisy and inconsistent region-level predictions
in training-free OVSS. As shown in Fig. 1, we observe that
features extracted from CLIP [27] and DINO [4] lack clear
discrimination across local (superpixel) regions, with higharXiv:2512.07360v1  [cs.CV]  8 Dec 2025

levels of noise and blurred boundaries. In contrast, simple
cues such as average colour differences are able to reflect lo-
cal structural differences more clearly. This motivates us to
ask:Can low-level region-adjacency information be lever-
aged to guide CLIP toward more localised attention?
These observations align with the contrastive training
paradigm of CLIP, which encourages semantic alignment
based on paired image-text data. Since high-resolution im-
ages naturally contain more discriminative details, they im-
prove the model’s ability to form stable and structured sim-
ilarity matrices. However, the inductive bias introduced
by CLIP’s global training procedure cannot be easily mit-
igated using its own representations. As shown in Fig. 1,
CLIP and DINO features are scattered and unstructured in
local regions. Fortunately, the image itself inherently pro-
vides instance-specific priors that are largely immune to
such global alignment biases. Specifically, the Region Ad-
jacency Graph (RAG), constructed purely from low-level
cues such as colour and texture, effectively captures spa-
tial relationships between regions without being affected
by CLIP’s global feature behaviour. Motivated by this,
we propose a structure-aware feature rectification approach
that incorporates RAG-based guidance into attention mech-
anisms. By constructing RAGs from low-level features
(e.g. colour, GLCM texture statistics), we introduce local
structure-aware biases to guide patch-level attention and
similarity computation.
Extensive experiments validate that the proposed method
enhances training-free open-vocabulary semantic segmen-
tation performance. It improves regional consistency, re-
duces noise in segmentation outputs, and better preserves
fine-grained structures, which are clearly visible in qualita-
tive results (see Fig. 5).
2. Related Work
2.1. Contrastive Language-Image Pre-training
Contrastive Language-Image Pre-training (CLIP) [27] is a
large-scale multi-modal foundation model that leverages
contrastive learning to align visual and textual features,
enhancing generalisation on unseen samples. Due to its
strong zero-shot capabilities, CLIP has been widely adopted
in Few-Shot/Zero-Shot Learning (FSL/ZSL) [17, 20, 24,
45, 46], Prompt Learning [17, 20, 45, 46], and Out-of-
Distribution (OoD) detection tasks [33].
More recently, researchers have extended CLIP to dense
prediction tasks [30, 38, 41, 42], such as semantic segmen-
tation [24, 34]. However, a major challenge in utilising
CLIP is the inherent noise in its features. Liet al. [21] anal-
yse this issue from an explainability perspective and pro-
pose self-attention improvements to enhance CLIP’s perfor-
mance in open-vocabulary tasks.
Unlike conventional pipelines that fine-tune pre-trainedmodels on additional datasets, CLIP’s encoder is typically
kept frozen to maintain its alignment with the text feature
space [44]. As a result, researchers tend to use CLIP di-
rectly as an encoder to extract preliminary features while fo-
cusing on designing sophisticated decoders [7, 9, 13, 28, 39]
to refine image-level representations for dense prediction
tasks.
2.2. Open-Vocabulary Semantic Segmentation
Open-vocabulary semantic segmentation (OVSS) extends
segmentation [26, 35, 36] and refers to segmenting seman-
tic regions via textual names or descriptions for the open
world without any mask annotations. Early works [44] ver-
ify the importance of modal alignment in CLIP, and com-
mon downstream fine-tuning may destroy its generalisa-
tion ability. MaskCLIP [44] attempts to improve the Vi-
sion Transformer (ViT) [10] structure of CLIP to allow
the model to obtain coarse feature localisation, and com-
bines transductive learning to improve performance. CLIP-
Surgery [22] analyses the difficulty of the current semantic
segmentation task introduced by CLIP from the perspec-
tive of image-text noise, and makes certain improvements
to the model using the idea of self-attention. SCLIP [37]
inherits the idea of self-attention from MaskCLIP and di-
rectly adapts the improved CLIP structure to the semantic
segmentation task.
Both CLIP-Surgery and SCLIP utilise the idea of self-
attention to improve CLIP, while only CLIP-Surgery men-
tions the noise problem caused by the open category of text.
None of them explores and analyses why CLIP lacks the
semantic correlation between patches. Our work comple-
ments this point that it is the global patch formed during the
attention interaction between [CLS] token and patches that
leads to this.
Beyond architectural improvements, recent work scruti-
nizes OVS evaluation protocols regarding task ambiguity.
Huang et al. [15] argue that rigid pixel-wise metrics con-
tradict the open-world premise by penalizing plausible syn-
onyms (e.g., ‘sofa’ vs. ‘couch’). They propose a mask-
wise evaluation protocol, demonstrating that mitigating cat-
egory ambiguity significantly enhances model capabilities
and suggesting a need for evolved benchmarks.
Regarding methodology, recent approaches leverage
CLIP as an encoder within a ”mask generation and
classification” pipeline, inspired by MaskFormer [5] and
Mask2Former [6]. These methods utilise pixel and query
decoders to refine features and generate masks via query
embeddings. By calculating the similarity between these
embeddings and text prompts, the model weights query
masks to produce final object boundaries and categories.

2.3. Training-free OVSS
Trident [31] proposes a training-free framework that ad-
dresses CLIP’s resolution limitation in semantic segmen-
tation through a splice-then-segment approach. Trident
first splices features extracted by CLIP and DINO from
sub-images, then leverages the Segment Anything Model
(SAM) for global aggregation, expanding the receptive
field and improving segmentation performance. The kNN-
CLIP [12] proposes a training-free approach for open-
vocabulary continual segmentation that mitigates catas-
trophic forgetting. Instead of traditional continual training,
kNN-CLIP augments the model with a database of instance
embeddings, enabling segmentation methods to adapt to
growing vocabularies without retraining or high memory
costs. These methods primarily modify the internal atten-
tion structure of CLIP-like models to better capture rela-
tionships between image regions and textual descriptions.
In contrast, our approach takes a different direction by
directly modifying the visual patch embeddings instead of
adjusting attention maps. Specifically, we improve the ac-
curacy of the visual patch-text embedding similarity matrix,
ensuring a more precise alignment between visual and tex-
tual representations. By refining the embedding space at
the patch level, our method enhances feature interaction and
boosts segmentation performance, complementing and sur-
passing attention-based optimisation strategies.
3. Preliminaries of Training-free OVSS
Training-free OVSS aims to segment an image into mean-
ingful regions by assigning semantic labels given arbitrary
vocabulary, without requiring extra training. Instead of
learning a segmentation model with annotated data, this ap-
proach leverages large pretrained vision-language models,
such as CLIP, to directly match visual features with text
embeddings through similarity computations. The visual
patches embedding{v i}N
i=1, where each patchv iis repre-
sented by a feature embedding of dimension1×RD, ex-
tracted from the vision encoder of CLIP. The text embed-
ding{t j}M
j=1is obtained from the text encoder, where each
tjcorresponds to a text and is also represented as a feature
embedding of dimension1×RD. The core idea is to com-
pute the cosine similarity between each visual patch feature
and all text embeddings:
si,j=⟨vi,tj⟩
∥vi∥∥tj∥,(1)
wheres i,jis the similarity score between visual patchv i
and text embeddingv j. The semantic label for each visual
patch is assigned based on the highest similarity score:
ˆyi= arg max
jsi,j,(2)
whereˆy idenotes the predicted semantic label for patchv i.4. Structure-Aware Feature Rectification
Due to CLIP’s global training paradigm on image-text
pairs, it lacks the capability for fine-grained local align-
ment [2, 14, 19, 29], resulting in structural inconsistency
and noisy predictions when directly applied to segmenta-
tion. This issue is especially pronounced in training-free
open-vocabulary semantic segmentation, where no addi-
tional data is available for model adaptation. To mitigate
this, our method leverages a region adjacency graph (RAG)
constructed from image low-level features to enhance struc-
tural awareness. It comprises two key modules: RAG-
guided Attention, which introduces a structure-aware bias
into CLIP’s attention mechanism to encourage local seman-
tic consistency; and Similarity Fusion, which refines cross-
modal similarity computation to suppress noisy matches.
4.1. RAG-guided Attention
Region adjacency graph (RAG) is a graph-based represen-
tation that captures the spatial relationships between image
regions. Formally, a RAG is defined as an undirected graph
G= (V, E), where each nodev i∈Vcorresponds to a
low-level regionR i(e.g. a superpixel), and an edgee ij∈E
exists if regionsR iandR jare spatially adjacent in the im-
age. By encoding both the appearance and structural prox-
imity of regions, the RAG provides a compact yet informa-
tive structure that reflects the local layout of the image.
RAG construction.However, traditional RAGs are typ-
ically constructed based on the average colour differences
between adjacent superpixel regions. The weight of each
edgee ijis defined as:
wcolor
ij=∥µ i−µj∥2,(3)
whereµ iandµ jdenote the mean RGB colour vectors of re-
gionsR iandR j, respectively. While this formulation pro-
vides clear local structural cues—as illustrated in Fig. 1—
colour differences alone are insufficient for robust region
discrimination. In real-world scenarios, colour ambiguity
often arises, such as between a “white toilet” and a “white
wall”. To build a more robust RAG, we incorporate not
only colour differences but also texture information. Specif-
ically, for each regionR i, we compute theGrey-Level Co-
occurrence Matrix (GLCM)P iand extract several statistical
features from it. The edge weight is redefined as a combi-
nation of colour and texture similarities:
wij=wcolor
ij+wtexture
ij,(4)
wherewtexture
ij is computed using the GLCM-based feature
difference:
wtexture
ij =X
kf(k)
i−f(k)
j,(5)

Figure 2.Illustration of superpixel-to-patch encoding.The dis-
tance between two patches is first represented as a list of all pair-
wise superpixel regions(□,□), then the patch distance is com-
puted from this list using Eq. 8.
in whichf(k)
irepresents thek-th texture feature (e.g.
contrast, homogeneity, energy, correlation) extracted from
regionR i’s GLCM. These features are defined as fol-
lows: contrast: Contrast=P
m,n(m−n)2Pi(m, n),
homogeneity: Homogeneity=P
m,nPi(m,n)
1+|m−n|, energy:
Energy=P
m,nPi(m, n)2, and correlation: Correlation=P
m,n(m−µ m)(n−µ n)Pi(m,n)
σmσn, whereP i(m, n)denotes the
normalized co-occurrence probability at position(m, n),
andµ m, µn, σm, σnare the means and standard deviations
of the marginal distributions ofP i.
Superpixel-aligned patch encoding.Another challenge
lies in the mismatch between the superpixel-based RAG and
the patch-based tokenisation used in transformers, where
inputs are typically divided into fixed-size square patches.
To address this, we design a mechanism that preserves the
structural advantages of superpixels—such as their abil-
ity to align flexibly with object boundaries—while en-
abling compatibility with patch-wise representations re-
quired by standard transformer attention. As illustrated
in Fig. 2, for two adjacent patches, denoted as□and□,
the computation of their edge weight is based on all pair-
wise distances between the superpixel regions contained
within each patch. Specifically, let patchicontain su-
perpixels{si
1, si
2, . . . , si
m}and patchjcontain superpixels
{sj
1, sj
2, . . . , sj
n}. We compute the pairwise distances:
Dij=
d(si
p, sj
q)|si
p∈i, sj
q∈j	
,(6)
whered(·,·)is the distance function defined in Eq. 5.
Therefore the computed edge weightw ijis a list rather than
a scalar,i.e.
wij=h
d(si
1, sj
1), d(si
1, sj
2), . . . , d(si
m, sj
n)i
.(7)
To preserve the structural variations within each patch,
we compute the mean and variance ofD ij, and use them
ImageGaussian KernelRAG-biasBilateralFigure 3.Illustration of different attention bias mechanisms.
The first column shows the input images. The second column visu-
alises the traditional Gaussian kernel, which models spatial prox-
imity in a local window. The third column shows the RAG-bias
computed from the Region Adjacency Graph (RAG), capturing
structural relationships between neighbouring regions. The fourth
column combines both the Gaussian kernel and the RAG-bias to
form a bilateral attention bias, which accounts for both spatial dis-
tance and local structure.
as the final edge weight representation between patchiand
patchj:
µij=1
|Dij|X
d∈Dijd, σ2
ij=1
|Dij|X
d∈Dij(d−µ ij)2.(8)
We then define the final edge weight list as (we use the
standard deviationσ i,j):
wfinal
ij= [µ ij, σij].(9)
RAG-guided attention via RAG bias.We leverage the
constructed Region Adjacency Graph (RAG) to compute a
structure-aware prior, referred to as theRAG bias, which
serves as a local structural constraint in the attention mech-
anism. As illustrated in Fig. 3 (third column), the RAG bias
is calculated for each token (patch) based on the topology
of its local neighbourhood in the RAG.
Specifically, for a node (patch)i, we consider its adjacent
neighboursN(i)—defined either as 4-connected (cross-
shaped) or 8-connected neighbours. For each neighbour
j∈ N(i), we use the final edge weightwfinal
ij= [µ ij, σ2
ij].
The RAG biasb ijis then computed by averaging the struc-
tural affinities from nodei’s neighbourhood:

MatMul𝑸𝑲𝑽ScaleSoftMax
∗
⨁MatMul
RAG-guided Attention
Gaussian Attention Map
Bilateral Attention MapFigure 4.Overview of the proposed RAG-guided attention
mechanism.The bilateral attention bias is computed by combin-
ing a spatial Gaussian kernel with a structure-aware RAG-bias.
This combined bias is integrated into the attention weights to en-
hance structural sensitivity.Right: visualisation of the Gaussian
and bilateral attention maps.
bij=1
|N(i)|X
k∈N(i)(µik+σik).(10)
Although the RAG biasb i,jeffectively encodes struc-
tural context by aggregating information from a node’s lo-
cal neighbourhood, it is fixed across all positions within the
same image. This static nature makes it insufficient for cap-
turing the pairwise relationships required in self-attention,
where different attention weights are computed between ev-
ery pair of tokens. To address this limitation, we draw inspi-
ration from bilateral filtering and introduce a more flexible
bias mechanism, Bilateral Bias, that combines both spatial
proximity and structural similarity. Specifically, we com-
pute a spatial Gaussian kernelg(i, j)between any two po-
sitionsiandj:
g(i, j) = exp
−∥pi−pj∥2
2σ2
,(11)
wherep iandp jdenote the 2D coordinates of patchesiand
j, andσcontrols the spatial range.
We then define the bilateral biasB ijas the product of the
spatial kernel and the structural RAG bias:
Bij=g(i, j)·exp (b i,j)(12)
as shown in Fig. 3 (fourth column). Then the final biased
attention is computed as:
attenbiased
ij =softmax j(qi·kj√
d+Bij).(13)The overall of this RAG-guided attention process is
shown in Fig. 4.
4.2. Similarity Fusion Module
Although introducing a bias that emphasises local structures
helps the model attend to fine-grained region boundaries
and maintain local consistency, it may also cause the model
to respond to irrelevant local noise, such as background tex-
tures, illumination variations, or boundary artefacts. To mit-
igate this, we propose similarity fusion.
Specifically, given the original visual-textual similarity
matrixS i,jdefined in Eq. 1, we compute a refined similar-
ity matrix ˜Si,jby first applying a Gaussian kernel to smooth
the visual features. Let ˆvidenote the smoothed visual fea-
ture at positioni; we can then compute the cosine similarity
between the smoothed visual features and the text featuret:
˜Si,j=ˆvi·tj
∥ˆvi∥∥tj∥.(14)
Finally, we fuse the original and smoothed similarities
using geometric mean fusion:
Sfused
i,j=
˜Si,jα
·(Si,j)1−α.(15)
5. Experiment
5.1. Implementation Details
Datasets.We evaluate our method on the following seg-
mentation benchmarks, whose names are abbreviated (in
parentheses) to conserve table space: PASCAL VOC 2012
(V21) [11], ADE20K-150 (ADE) [43], PASCAL Context
(PC60) [25], COCO-Stuff (C-Stf) [3], Cityscapes (City)
[8], COCO-Object (C-Obj) [23]. Additionally, alongside
the original benchmarks on these datasets, we follow [37]
and evaluate on variants of PASCAL VOC 2012 (V20) and
PASCAL Context (PC59) in which the background class is
removed from the evaluation.
Baselines.We compare our method to a set of relevant
works in OVSS, including: MaskCLIP [44], ReCo [32],
GroupVit [40], SCLIP [37], OVDiff [16], CLIPtrace [29],
NACLIP [14], and ProxyCLIP [19]. It is worth noting
that, for the sake of fair comparison, none of the methods,
including our baselines, involve any post-processing dur-
ing evaluation. This includes commonly used techniques
such as Conditional Random Fields (CRF), multi-scale test-
ing, mask refinement, or other enhancement strategies. All
methods are evaluated based on their raw model outputs to
ensure a fair and consistent comparison. Specifically, we
adopt SCLIP, CLIPtrace, NACLIP, and ProxyCLIP as our
baselines and integrate our proposed module into their orig-
inal frameworks. All other settings strictly follow those de-
scribed in the respective original papers.

Table 1.Quantitative results on various OVSS benchmarks.Our method consistently improves different CLIP-based baselines across
all datasets, showing its generality and effectiveness. Best performance inbold.
Method Venue V21 PC60 C-Obj V20 PC59 Stuff City ADE Avg
CLIP [27] ICML’21 16.4 8.4 5.6 41.9 9.2 4.4 5.0 2.9 11.7
GroupViT [40] CVPR’22 52.3 18.7 27.5 79.7 18.5 23.4 10.4 15.3 30.7
MaskCLIP [44] ECCV’22 43.4 23.2 20.6 74.9 26.4 16.7 24.9 11.9 30.3
Reco [32] NeurIPS’22 25.1 19.9 15.7 57.7 21.6 22.3 11.2 14.8 23.5
OVDiff [16] ECCV’24 66.3 29.7 34.6 80.9 32.9 20.3 23.4 14.1 37.8
CLIP-Surgery [21] Pattern Recognition 59.0 30.1 30.2 80.1 33.9 22.1 31.8 15.8 37.9
,→+ Ours - 61.1 32.2 31.8 81.3 34.8 23.6 33.5 17.4 39.4
SCLIP [37] ECCV’24 59.1 30.4 30.5 80.4 34.2 22.4 32.2 16.1 38.2
,→+ Ours - 61.9 32.9 32.3 81.8 35.0 24.1 33.9 18.1 39.9
CLIPtrace [29] ECCV’24 53.0 30.8 33.8 81.2 35.0 24.1 35.0 17.0 38.7
,→+ Ours - 56.2 32.1 35.2 83.8 36.4 25.1 36.6 18.5 40.5
NACLIP [14] WACV’25 58.9 32.2 33.2 79.7 35.2 23.3 35.5 17.4 39.4
,→+ Ours - 60.3 33.5 34.6 81.2 36.0 25.7 36.8 19.1 40.9
ProxyCLIP [19] ECCV’24 61.3 35.3 37.5 80.3 39.1 26.5 38.1 20.2 42.3
,→+ Ours - 62.9 36.6 38.9 82.1 39.8 27.7 40.1 21.1 43.6
Implementation details.All experiments are conducted
on a single NVIDIA RTX 4090 GPU. We adopt mean
Intersection-over-Union (mIoU) as the evaluation metric
across all experiments. For the Similarity Fusion Module,
we set the weighting parameterαto 0.6, selected based on
the best performance on cocostuff171-val. The Gaussian
kernel used in the module has a kernel size of 3 and a stan-
dard deviationσof 3. SLIC [1] is used as our default super-
pixel method, with n segments=300 and compactness=10.
More details about the hypaparameter sensitivity analysis
can be found in Supplementary Material Section S1.
5.2. Results
Table 1 presents quantitative results on various OVSS
benchmarks, comparing our method with several state-of-
the-art baselines. As shown, integrating our proposed mod-
ule into different CLIP-based models consistently improves
performance across all datasets. Notably, our approach
yields gains on challenging datasets such as ADE20K,
Cityscapes, and PC60, regardless of the baseline model.
These improvements validate the effectiveness and gener-
ality of our method in enhancing open-vocabulary semantic
segmentation performance. Specifically, our method boosts
the average mIoU by +1.8 on SCLIP (38.2→40.0), +1.8
on CLIPtrace (38.7→40.5), +1.5 on NACLIP (39.4→
40.9), and +1.4 on ProxyCLIP (42.3→43.7). These results
highlight the generality and effectiveness of our approach in
enhancing training-free CLIP-based open-vocabulary seg-
mentation models.Table 2. Component ablation results based on the NACLIP model.
Method V21 PC60 C-Obj V20 PC59 Stuff City ADE
w/o 58.9 32.2 33.2 79.7 35.2 23.3 35.5 17.4
SimFusion RAG-bias
✔ ✗ 59.4 32.6 33.5 80.5 35.9 24.2 36.0 18.1
✗ ✔ 60.0 33.1 34.2 81.0 36.1 25.5 36.7 19.0
✔ ✔ 60.2 33.4 34.4 81.0 36.2 25.8 36.9 19.2
Table 3.Performance comparison using different feature types
to construct RAG edges.The top half shows results under stan-
dard input conditions, while the bottom half (marked with❀) rep-
resents experiments with colour perturbations to evaluate robust-
ness.C.-onlydenotes colour-only input, andC. + G.indicates
colour with additional GLCM texture statistics.Boldnumbers
highlight the best performance per column and per setting.
RAG edge V21 PC60 C-Obj V20 PC59 Stuff City ADE
CLIP feat 55.5 28.4 23.9 74.3 33.2 19.2 28.9 14.9
DINO feat 55.2 27.9 25.2 75.3 32.9 20.2 29.2 15.4
C.-only 58.6 32.1 32.2 80.0 35.8 24.0 35.2 17.2
C. + G.60.0 33.1 34.2 81.0 36.1 25.5 36.7 19.0
CLIP feat❀ 53.0 26.7 22.4 72.0 30.2 16.4 25.3 12.8
DINO feat❀ 53.5 25.4 21.9 71.8 31.7 18.9 26.0 13.0
C.-only❀ 50.4 27.6 25.3 74.2 29.9 18.0 30.4 10.2
C. + G.❀ 58.5 32.0 32.9 79.9 35.1 23.9 35.8 18.2
5.3. Ablation Study
In this section, we conduct extensive ablation studies to
analyse our model. Unless otherwise specified, the baseline
used in these ablation studies is NACLIP [14].
Ablation on proposed components.Table 2 presents
a component-level ablation study based on the NACLIP
model, evaluating the individual and combined contribu-

tions of the Similarity Fusion module (SimFusion) and the
RAG-bias mechanism. The baseline model without either
component achieves an average mIoU of 39.4. Introducing
SimFusion or RAG-bias individually improves the perfor-
mance to 40.0 (+0.6) and 40.9 (+1.5), respectively. When
both components are enabled, the model reaches the highest
performance with an average mIoU of 41.2, showing con-
sistent gains across all datasets.
These results suggest that the two components are com-
plementary and jointly contribute to performance improve-
ments. SimFusion enhances cross-region similarity integra-
tion, while RAG-bias introduces semantically meaningful
structural bias to the attention. Notably, RAG-bias yields a
greater standalone improvement than SimFusion, highlight-
ing its stronger impact on the model’s effectiveness.
Ablation on RAG edge.Table 3 presents an ablation
study comparing different feature types for constructing
RAG edges. We evaluate four configurations: CLIP fea-
tures, DINO features, colour-only input (C.-only), and
colour with additional GLCM texture statistics (C. + G.).
The top half reports results under standard conditions, while
the bottom half (grey-shaded rows, marked with❀) in-
cludes colour perturbations to assess robustness. Since our
RAG is primarily constructed using low-level features, it
may be susceptible to common image perturbations. To
analyse model behaviour under such conditions and eval-
uate robustness, we apply random colour jitter using the
ColorJitter function with the following parameters: bright-
ness=0.2, contrast=0.3, saturation=0.3, and hue=0.1. This
augmentation introduces appearance shifts while preserving
semantic content.
We find that combining colour with GLCM texture
(C.+G.) consistently outperforms other settings under both
clean and perturbed conditions. Under colour jitter,C.-only
degrades noticeably, whileC.+G.remains strong (e.g. 35.8
on City, 18.2 on ADE), surpassing even CLIP and DINO.
This confirms the effectiveness of integrating texture fea-
tures for robust RAG edge construction.
Ablation on the number of neighbours.When comput-
ing the RAG-bias (seeN(i)in Eq. 10), we can aggregate
information from a varying number of neighbouring nodes.
Table 4 reports the results using 4 and 8 neighbours. While
using 8 neighbours yields slightly better performance, the
differences are marginal. This suggests that, once the RAG
is constructed, our method is relatively insensitive to the
number of aggregated neighbours.
Ablation on patch size and image size.Since our pro-
posed superpixel-aligned patch encoding method computes
representations based on the superpixel regions within each
patch, both the patch size and input image resolution mayTable 4. Performance comparison with different neighbour con-
figurations. “#neigh.” is the number of neighbours.
#neigh. V21 PC60 C-Obj V20 PC59 Stuff City ADE
4 60.1 33.0 34.2 81.0 35.8 25.6 36.5 19.0
8 60.2 33.4 34.4 81.0 36.2 25.8 36.9 19.2
Table 5. Effect of patch size and image resolution on performance.
The results are reported on the NACLIP model.
Patch Img V21 PC60 C-Obj V20 PC59 Stuff City ADE
B/16 33660.2 33.4 34.4 81.0 36.2 25.8 36.9 19.2
B/32 336 57.2 30.0 31.2 78.7 34.3 22.7 32.5 16.4
B/16 224 58.5 31.2 30.9 79.2 35.0 23.6 34.0 16.1
B/32 224 55.2 28.0 29.6 75.0 31.7 20.0 30.6 13.6
Table 6. Comparison of different superpixel segmentation meth-
ods for RAG construction. Best performance inbold. A discus-
sion on using masks generated by the Segment Anything Model
(SAM) [18] is provided in Supplementary Material Section S5.
Method V21 PC60 C-Obj V20 PC59 Stuff City ADE
SLIC60.2 33.4 34.4 81.0 36.2 25.8 36.9 19.2
Watershed 58.2 32.8 33.2 80.2 35.3 23.2 34.9 18.2
Felzenszwalb 55.2 30.1 28.9 76.9 33.0 22.5 32.9 14.5
affect the model performance. As shown in Table 5, us-
ing smaller patch sizes (e.g. B/16vs.B/32) and higher
image resolutions (e.g. 336vs.224) consistently leads to
better performance across all benchmarks. In particular, the
best results (an average mIoU of 40.9) are achieved with the
B/16 patch size and 336 image resolution setting. These re-
sults suggest that finer spatial granularity in the patch-level
representation helps better capture region-boundary align-
ment with superpixels, enhancing segmentation quality.
Ablation on different superpixel methods.In Table 6,
we compare different superpixel segmentation methods for
RAG construction, including SLIC, Watershed, and Felzen-
szwalb. Among them, SLIC achieves the best overall per-
formance, while Felzenszwalb performs the worst across
all benchmarks. We attribute the weaker performance of
Felzenszwalb to its irregular and region-driven segmenta-
tion outputs, which often deviate from the grid-like patch
structure used in our superpixel-to-patch encoding. In con-
trast, SLIC produces more compact and uniformly shaped
superpixels that align better with patch boundaries, making
it more compatible with our proposed encoding strategy.
Ablation onαin similarity fusion.Table 8 shows the
performance of our model under different weighting pa-
rameterαin the Similarity Fusion module. We observe
that the module with anyα >0consistently improves
performance over the baseline (w/o), suggesting the effec-

Table 7. Efficiency analysis of our method. Comparison of in-
ference speed (FPS) and computational cost (FLOPs) on different
baseline models.
Method FPS (∆) FLOPs (∆)
CLIP / + Ours 72.5→71.1 (-1.4) 41.7→ ≈41.7 (+0.0)
SCLIP / + Ours 68.8→66.9 (-1.9) 44.2→ ≈44.2 (+0.0)
ProxyCLIP / + Ours 52.9→51.5 (-1.4) 81.1→ ≈81.1 (+0.0)
Table 8. Performance comparison under differentαvalues on
SimFusion module based on the NACLIP model.
α V21 PC60 C-Obj V20 PC59 Stuff City ADE
w/o 58.9 32.2 33.2 79.7 35.2 23.3 35.5 17.4
0.1 58.9 32.2 33.2 79.7 35.2 23.3 35.5 17.6
0.2 59.0 32.5 33.4 79.9 35.3 23.5 35.6 17.7
0.5 59.2 32.533.680.3 35.824.535.9 18.0
0.6 59.4 32.633.580.5 35.924.236.0 18.1
0.7 58.9 32.0 33.1 80.3 35.8 24.1 35.9 18.0
tiveness of combining the original similarity matrix with its
smoothed counterpart. This fusion helps suppress noisy or
unreliable similarity signals, leading to more robust region
aggregation. The best performance is achieved atα= 0.6,
which is used as the default setting in all experiments.
Generalisation analysis.We provide an extensive analy-
sis of our model’s generalisation capabilities in Section S2
of the Supplementary Material, evaluating its performance
on challenging transformations such as overexposure, un-
derexposure, grayscale, style transfer, and texture destruc-
tion. In addition, we assess its zero-shot performance on a
remote sensing dataset, with results reported in the Supple-
mentary Material in Table S2 and Fig. S5.
Failure case analysis.Our method’s primary failure
cases occur in underexposed conditions, where the loss of
fine-grained local details causes our attention bias to be mis-
weighted. For a detailed analysis of these failure cases,
please refer to Section S3 in the Supplementary Material.
Qualitative results.In Fig. 5 we present qualitative com-
parisons between our method and CLIPtrace. As illus-
trated, our approach consistently yields more coherent and
accurate segmentation results, especially in challenging re-
gions such as object boundaries, fine-grained structures, and
texture-rich areas. The highlighted areas (in red boxes) em-
phasise our model’s ability to preserve local consistency
and object integrity, reducing the fragmented or noisy pre-
dictions that are often observed in the CLIPtrace outputs.
These qualitative observations underscore the effectiveness
of our proposed components in enhancing performance at
finer levels of granularity.Computational cost.As shown in Table 7, our method
is computationally efficient. When integrated with vari-
ous baselines, it introduces no additional FLOPs while only
causing a negligible decrease in inference speed (FPS).
ImageGTCIIPtraceOurs
Figure 5. The qualitative results of our method. For more chal-
lenging cases, such as grayscale and stylised images (e.g. oil paint-
ings), please refer to Figs. S4, S6, and S7 in the Supplementary
Materials.
6. Conclusion
In this work, we proposed a new feature rectification ap-
proach for training-free open-vocabulary semantic segmen-
tation. Our method leverages a Region Adjacency Graph
(RAG) to refine visual patch embeddings and address local
inconsistency in CLIP-based models. Specifically, we intro-
duced a RAG-bias to guide attention toward semantically
relevant regions, and a Similarity Fusion module to better
align visual patches with textual categories. Extensive ex-
perimental analysis showed the effectiveness and generalis-
ability of our approach, presenting consistent improvements
across multiple datasets without additional training. Ab-
lation studies further highlighted the importance of neigh-
bourhood design and RAG construction, providing insights
into utilising low-level priors for semantic refinement.

Acknowledgements
This project is partially supported by an Amazon Research
Award. Qiming Huang is supported by the China Schol-
arship Council (Grant No. 202408060321). The computa-
tions in this research were performed using the Baskerville
Tier 2 HPC service. Baskerville was funded by the EP-
SRC and UKRI through the World Class Labs scheme
(EP\T022221\1) and the Digital Research Infrastructure
programme (EP\W032244\1) and is operated by Advanced
Research Computing at the University of Birmingham.
Appendix
S1. More ablation study for hyperparameters
To investigate the impact of key hyperparameters on our
model’s performance, we conducted a series of ablation
studies. The experiments focused on the parameters of the
Simple Linear Iterative Clustering (SLIC) algorithm and the
selection of features from the Grey-Level Co-occurrence
Matrix (GLCM).
5 10 20 40
Compactness1003005001000Number of Segments
22.92 23.26 22.13 16.4424.65 25.47 23.39 17.3722.76 22.99 22.40 17.0017.78 17.95 16.75 13.76
141618202224
mIoU
Figure S1. Ablation study on the number of segments
(nsegments) and compactness for the SLIC algorithm. Results
are reported as mIoU on the COCO-Stuff-171 validation set. The
baseline model is NACLIP.
We performed a grid search to optimise the number of
segments and the compactness for the SLIC algorithm, with
the quantitative results shown in Fig. S1. Different hyperpa-
rameters of SLIC produce different region proposal results,
as visualised in Fig. S3. The number of segments directly
controls the scale of the superpixels; increasing this value
results in finer, more numerous regions. The compactness
parameter manages the trade-off between spatial proximity
24.624.825.025.225.425.625.8mIoU25.78 25.78
25.1725.15
25.0825.03 25.02
24.97 24.95
24.90
24.79
24.68 24.68
24.6324.61F1 (Correlation)
F2 (Contrast)
F3 (Energy)
F4 (Homogeneity)
F2, F4
F1, F2, F4 F2, F3, F4
F1, F2, F3, F4F2
F2, F3
F1, F3, F4F3
F3, F4F4 F1
F1, F2 F1, F3 F1, F4
F1, F2, F3
Feature Combination0.00.20.40.60.81.0Figure S2. The impact of different SLIC feature combinations on
the performance of the NACLIP baseline. The experiment was
conducted on the COCO-Stuff-171 validation set with baseline
NACLIP.
and colour similarity; a lower value allows superpixels to
conform more closely to image textures and edges, while a
higher value produces more uniform, regularly shaped re-
gions. Our analysis indicates that the optimal balance for
this task was achieved with 300 segments and a compact-
ness of 10, yielding a peak mIoU of 25.47.
Furthermore, we analysed the contribution of different
combinations of four GLCM texture features: Correlation
(F1), Contrast (F2), Energy (F3), and Homogeneity (F4).
The results, presented in Fig. S2, reveal that the combi-
nation of Contrast (F2) and Homogeneity (F4) yielded the
highest mIoU of 25.78. Notably, this two-feature subset
outperformed the combination of all four features, high-
lighting that an appropriate selection of features is more ef-
fective than using them all.
S2. Generalisation analysis
To further evaluate the generalisation capabilities of our
proposed method, we conduct a series of analyses under
various challenging conditions, including common image
corruptions, domain shifts, and zero-shot segmentation on
extra domain-related remote sensing datasets postdam1.
Robustness to common image corruptions.We first as-
sess the model’s resilience to common visual perturbations
that degrade image quality. Table S1 quantitatively mea-
sures the performance changes of our method built upon
NACLIP baseline under four conditions: overexposure, un-
derexposure, grayscale conversion, and texture destruction
(via Gaussian blur). For overexposure, we used a brightness
factor of 1.8 to significantly increase the luminosity, caus-
1Isprs potsdam dataset on kaggle. Available online:
https://www.kaggle.com/datasets/jahidhasan66/isprs-potsdam

Figure S3. Comparison of SLIC segmentation results across different combinations of Segments (number of superpixels) and Compactness.
ing highlights to become washed out. For underexposure,
we used a brightness factor of 0.4 to decrease the luminos-
ity, obscuring details in shadows. For texture destruction,
we used a kernel size of(9,9)pixels andσ= 5to create
a significant and noticeable blurring effect that effectively
destroys surface textures. Our method maintains reasonable
performance in most cases, but significantly reduces when
facing strong low exposure.
To better understand this, we provide a visualisation of
the segmentation results under these conditions.Effect of
Lighting.As shown in Fig. S4, our model performs well
under extreme lighting changes. Despite significant infor-
mation loss in the bright, washed-out areas of overexposed
images or the dark, detail-lacking regions of underexposed
images, our model consistently generates reasonable seg-
mentation masks for objects like ‘doughnuts’, ‘zebras’.
Effect of texture destruction.To simulate the loss of
fine-grained details and high-frequency textures, we appliedTable S1. The performance drops of our method under different
cases. We use NACLIP as the baseline.
V20 Stuff PC59 ADE
overexposure -2.1 -1.8 -2.5 -1.5
underexposure -8.5 -10.8 -7.5 -4.5
grayscale -1.1 -1.5 -2.1 -1.2
texture destruction -1.8 -2.3 -2.2 -1.8
a Gaussian blur filter. This process involves convolving the
image with a Gaussian kernel. The intensity of the blur is
controlled by the kernel size and the standard deviation (σ).
In our experiments, we used a kernel size of(9,9)pixels
andσ= 5to create a significant and noticeable blurring
effect that effectively destroys surface textures. The visual-
isation results are shown in Fig. S6.
Analysis of domain shift.We further investigate the

donutschool buszebraroad signfoodcalculatororiginaloverexposureunderexposureoriginaloverexposureunderexposureFigure S4. The effect of overexposure and underexposure on segmentation performance.
ImagePredGTImagePredGT
Figure S5. Visualisation results on the Potsdam dataset of our
method built upon NACLIP.
Table S2. Comparison of Zero-Shot Performance on the Potsdam
Remote Sensing Datasets. Results are reported on mIoU
Potsdam
NACLIP 28.6
NACLIP + Ours 30.4
ClipSurgery 30.2
ClipSurgery + Ours 32.1
model’s ability to generalise across different visual do-
mains, a critical aspect of real-world applications. Fig. S7
showcases the segmentation performance on images that
have undergone significant style and domain shifts. We
test on artistic renderings (e.g. oil painting), images with
altered colour schemes (grayscale vs. coloured), and var-
ious other style-transferred examples. The model con-sistently produces precise segmentations for objects like
’dogs’, ’pineapples’, and ’boats’ across these diverse visual
styles.
Zero-Shot generalisation to Postdam remote sensing
dataset.Additionally, we evaluate our method’s zero-shot
performance on a completely unseen and specialised do-
main: the Potsdam remote sensing dataset. As reported in
Table S2, when our module is integrated with existing base-
lines (NACLIP and ClipSurgery), it yields substantial im-
provements in mean Intersection over Union (mIoU). The
Visualisation result is shown in Fig. S5.
S3. Analysis of failure cases
We analysed the failure cases of our method to better under-
stand its limitations. Fig. S8 illustrates five representative
scenarios where the segmentation performance is compro-
mised:Small Object Insensitivity.Our approach relies on
an initial superpixel segmentation. Consequently, objects
that are exceptionally small, such as the ‘faucet’, may be
smaller than the generated superpixels and are incorrectly
absorbed into larger background regions. This prevents
them from being represented as distinct nodes in the region
adjacency graph.Extreme Lighting Conditions.In cases
of severe underexposure, the lack of sufficient colour and
brightness information cripples the feature extraction pro-
cess. Both SLIC and GLCM features become unreliable,
leading to a near-complete failure to identify any objects.
Ambiguous Boundaries and Camouflage.The model’s
performance degrades when there is no clear distinction be-
tween foreground and background. This occurs in scenes
with chaotic colour and texture, where boundaries are in-

originalblurredsegmentation_b
segmentation_o
computer
grass
grasscake
toweroriginalblurredsegmentation_bsegmentation_o
flagsuitcaseFigure S6. The effect of texture destruction (Gaussian blur) on segmentation performance.
oil paintinggrayscalecolouredpineapplepineapplehot air balloonhot air balloonelephantelephantbusdogstree
style transferredstyletransferredstyletransferredcatcatcatdogdogdog
boatboatboat
Figure S7. The effect of domain shift on segmentation performance.
herently ambiguous, and in cases of camouflage, where the
object’s texture features are nearly identical to the back-
ground’s.Excessive Scene Complexity.Our method can
be challenged by scenes containing an overwhelming den-
sity of small, intricate details. The high frequency of colour
and texture changes results in an overly fragmented super-
pixel map and a highly complex region graph, which hin-
ders the effective propagation and feature rectification.
S4. Visualisation comparison with ClipSurgery
Fig. S9 visualises the qualitative impact of our method
when applied to ClipSurgery. It is evident that our ap-
proach refines the model’s attention mechanism. The base-line ClipSurgery model, while effective, often produces
coarse and noisy attention maps that fail to precisely lo-
calise the target object (e.g.’bus’, ’grass’). By incorporating
our structure-aware feature rectification using region adja-
cency graphs, the resulting attention becomes more focused
and clean. Our method successfully prunes background
noise and sharpens the activation to align with true object
boundaries.
S5. Combining SAM with our method
The construction of the Region Adjacency Graph (RAG) is
critical to our method’s success. A natural consideration
was to leverage powerful segmentation models like SAM to

small object: faucet 
underexposure
colourtexture disorder
camouflaged objects
excessive colourcomplexityFigure S8. Illustration of challenging scenarios leading to segmentation failures. The cases, from left to right, include: (a) an object that is
too small to be accurately detected (faucet); (b) severe underexposure resulting in loss of detail; (c) disordered color and texture, making
boundaries ambiguous; (d) an object camouflaged against a similar background; and (e) a scene with excessive colour complexity and
numerous small details.
dogsImageClipSurgery+OursClipSurgery
bus
TV
shadowImageClipSurgery+OursClipSurgery
sign
grass
Figure S9. Comparison of model attention visualisation results with ClipSurgery and ClipSurgery + Ours
generate the graph’s nodes. However, as Fig. S10 reveals,
this approach introduces a significant topological problem.
SAM masks, while semantically meaningful, vary dramat-
ically in size. This leads to the formation of a highly cen-
tralised RAG, where massive background regions become
“hub nodes” that connect to a vast number of smaller re-
gions (column 3). Such a structure is unstable for fea-
ture propagation, as these hubs can wash out important lo-
cal details. To solve this, our method instead employs a
superpixel-based tessellation of the image (column 4). This
approach guarantees a granular and uniform partitioning,
resulting in a balanced and regular RAG. Each node main-
tains local connectivity with a consistent number of neigh-
bours, providing a stable and unbiased foundation for the
structure-aware feature rectification.References
[1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien
Lucchi, Pascal Fua, and Sabine S ¨usstrunk. Slic superpix-
els compared to state-of-the-art superpixel methods.IEEE
transactions on pattern analysis and machine intelligence,
34(11):2274–2282, 2012. 6
[2] Sule Bai, Yong Liu, Yifei Han, Haoji Zhang, and Yansong
Tang. Self-calibrated clip for training-free open-vocabulary
segmentation.arXiv preprint arXiv:2411.15869, 2024. 3
[3] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-
stuff: Thing and stuff classes in context. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2018. 5
[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv ´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers. InPro-
ceedings of the IEEE/CVF international conference on com-

ImageSAM masksSAM RAGOurs RAGFigure S10. Comparison of RAG construction methods. While SAM masks (column 2) provide semantic regions, their imbalanced sizes
create a problematic RAG (column 3). Large masks become hub nodes with numerous connections, dominating inter-region calculations;
in column 3, larger circles indicate a greater number of neighbours. Our approach uses superpixels (column 4) to create a RAG with
uniformly sized regions and a consistent number of neighbours, ensuring a more stable and reliable graph structure.
puter vision, pages 9650–9660, 2021. 1
[5] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-
pixel classification is not all you need for semantic segmen-
tation.Advances in Neural Information Processing Systems,
34:17864–17875, 2021. 2
[6] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-
der Kirillov, and Rohit Girdhar. Masked-attention mask
transformer for universal image segmentation. InProceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 1290–1299, 2022. 2
[7] Seokju Cho, Heeseong Shin, Sunghwan Hong, Seungjun
An, Seungjun Lee, Anurag Arnab, Paul Hongsuck Seo,
and Seungryong Kim. Cat-seg: Cost aggregation for
open-vocabulary semantic segmentation.arXiv preprint
arXiv:2303.11797, 2023. 2
[8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo
Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe
Franke, Stefan Roth, and Bernt Schiele. The cityscapes
dataset for semantic urban scene understanding. InProceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 3213–3223, 2016. 5
[9] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. De-
coupling zero-shot semantic segmentation. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 11583–11592, 2022. 2
[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale.arXiv preprint
arXiv:2010.11929, 2020. 2
[11] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christo-
pher KI Williams, John Winn, and Andrew Zisserman. The
pascal visual object classes challenge: A retrospective.In-
ternational Journal of Computer Vision, 2015. 5
[12] Zhongrui Gui, Shuyang Sun, Runjia Li, Jianhao Yuan, Zhao-
chong An, Karsten Roth, Ameya Prabhu, and Philip Torr.
knn-clip: Retrieval enables training-free segmentation on
continually expanding large vocabularies.arXiv preprint
arXiv:2404.09447, 2024. 3
[13] Jie Guo, Qimeng Wang, Yan Gao, Xiaolong Jiang, Shaohui
Lin, and Baochang Zhang. Mvp-seg: Multi-view prompt
learning for open-vocabulary semantic segmentation. InPat-
tern Recognition and Computer Vision: 6th Chinese Confer-
ence, page 158–171. Springer-Verlag, 2023. 2
[14] Sina Hajimiri, Ismail Ben Ayed, and Jose Dolz. Pay attention
to your neighbours: Training-free open-vocabulary semantic
segmentation. In2025 IEEE/CVF Winter Conference on Ap-
plications of Computer Vision (WACV), pages 5061–5071.
IEEE, 2025. 3, 5, 6
[15] Qiming Huang, Han Hu, and Jianbo Jiao. Revisit the open
nature of open vocabulary segmentation. InThirteenth In-

ternational Conference on Learning Representations, 2025.
2
[16] Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian
Rupprecht. Diffusion models for zero-shot open-vocabulary
segmentation.arXiv preprint arXiv:2306.09316, 2023. 5, 6
[17] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad
Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple:
Multi-modal prompt learning. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19113–19122, 2023. 2
[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. InProceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 7
[19] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang
Wang, Litong Feng, and Wayne Zhang. Proxyclip: Proxy
attention improves clip for open-vocabulary segmentation.
InEuropean Conference on Computer Vision, pages 70–88.
Springer, 2024. 3, 5, 6
[20] Dongjun Lee, Seokwon Song, Jihee Suh, Joonmyeong Choi,
Sanghyeok Lee, and Hyunwoo J Kim. Read-only prompt
optimization for vision-language few-shot learning. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 1401–1411, 2023. 2
[21] Yi Li, Hualiang Wang, Yiqun Duan, and Xiaomeng Li. Clip
surgery for better explainability with enhancement in open-
vocabulary tasks.arXiv preprint arXiv:2304.05653, 2023. 2,
6
[22] Yi Li, Hualiang Wang, Yiqun Duan, Jiheng Zhang, and Xi-
aomeng Li. A closer look at the explainability of con-
trastive language-image pre-training.Pattern Recognition,
162:111409, 2025. 2
[23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll ´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
European Conference on Computer Vision, pages 740–755.
Springer, 2014. 5
[24] Yuqi Lin, Minghao Chen, Wenxiao Wang, Boxi Wu, Ke
Li, Binbin Lin, Haifeng Liu, and Xiaofei He. Clip is also
an efficient segmenter: A text-driven approach for weakly
supervised semantic segmentation. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 15305–15314, 2023. 2
[25] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu
Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and
Alan Yuille. The role of context for object detection and
semantic segmentation in the wild. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2014. 5
[26] Bohao Peng, Zhuotao Tian, Xiaoyang Wu, Chengyao Wang,
Shu Liu, Jingyong Su, and Jiaya Jia. Hierarchical dense cor-
relation distillation for few-shot segmentation. InProceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 23641–23651, 2023. 2
[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. InInternational Conference on Machine Learning,
pages 8748–8763. PMLR, 2021. 1, 2, 6
[28] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong
Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu.
Denseclip: Language-guided dense prediction with context-
aware prompting. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
18082–18091, 2022. 2
[29] Tong Shao, Zhuotao Tian, Hang Zhao, and Jingyong Su. Ex-
plore the potential of clip for training-free open vocabulary
semantic segmentation. InEuropean Conference on Com-
puter Vision, pages 139–156. Springer, 2024. 3, 5, 6
[30] Hengcan Shi, Munawar Hayat, Yicheng Wu, and Jianfei
Cai. Proposalclip: Unsupervised open-category object pro-
posal generation via exploiting clip cues. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9611–9620, 2022. 2
[31] Yuheng Shi, Minjing Dong, and Chang Xu. Har-
nessing vision foundation models for high-performance,
training-free open vocabulary segmentation.arXiv preprint
arXiv:2411.09219, 2024. 3
[32] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Re-
trieve and co-segment for zero-shot transfer.Advances in
Neural Information Processing Systems, 2022. 5, 6
[33] Yang Shu, Xingzhuo Guo, Jialong Wu, Ximei Wang, Jianmin
Wang, and Mingsheng Long. Clipood: Generalizing clip to
out-of-distributions.arXiv preprint arXiv:2302.00864, 2023.
2
[34] Jiajin Tang, Ge Zheng, Cheng Shi, and Sibei Yang. Con-
trastive grouping with transformer for referring image seg-
mentation. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 23570–
23580, 2023. 2
[35] Zhuotao Tian, Hengshuang Zhao, Michelle Shu, Zhicheng
Yang, Ruiyu Li, and Jiaya Jia. Prior guided feature enrich-
ment network for few-shot segmentation.IEEE transactions
on pattern analysis and machine intelligence, 44(2):1050–
1065, 2020. 2
[36] Zhuotao Tian, Xin Lai, Li Jiang, Shu Liu, Michelle Shu,
Hengshuang Zhao, and Jiaya Jia. Generalized few-shot se-
mantic segmentation. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
11563–11572, 2022. 2
[37] Feng Wang, Jieru Mei, and Alan Yuille. Sclip: Rethink-
ing self-attention for dense vision-language inference.arXiv
preprint arXiv:2312.01597, 2023. 2, 5, 6
[38] Yixuan Wei, Yue Cao, Zheng Zhang, Zhuliang Yao, Zhenda
Xie, Han Hu, and Baining Guo. icar: Bridging image clas-
sification and image-text alignment for visual recognition.
arXiv preprint arXiv:2204.10760, 2022. 2
[39] Letian Wu, Wenyao Zhang, Tengping Jiang, Wankou Yang,
Xin Jin, and Wenjun Zeng. [cls] token is all you
need for zero-shot semantic segmentation.arXiv preprint
arXiv:2304.06212, 2023. 2
[40] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon,
Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit:

Semantic segmentation emerges from text supervision. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 18134–18144, 2022. 5,
6
[41] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan
Li, Jianwei Yang, and Lei Zhang. A simple framework for
open-vocabulary segmentation and detection. InProceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 1020–1031, 2023. 2
[42] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chun-
yuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou,
Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-
based language-image pretraining. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16793–16803, 2022. 2
[43] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fi-
dler, Adela Barriuso, and Antonio Torralba. Semantic under-
standing of scenes through the ade20k dataset.International
Journal of Computer Vision, 127:302–321, 2019. 5
[44] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free
dense labels from clip. InEuropean Conference on Com-
puter Vision, pages 696–712. Springer, 2022. 2, 5, 6
[45] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei
Liu. Conditional prompt learning for vision-language mod-
els. InProceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 16816–16825,
2022. 2
[46] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei
Liu. Learning to prompt for vision-language models.In-
ternational Journal of Computer Vision, 130(9):2337–2348,
2022. 2