# MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding

**Authors**: Fan Yang, Kaihao Zhang

**Published**: 2025-12-02 16:22:01

**PDF URL**: [https://arxiv.org/pdf/2512.02906v2](https://arxiv.org/pdf/2512.02906v2)

## Abstract
Understanding high-resolution images remains a significant challenge for multimodal large language models (MLLMs). Recent study address this issue by dividing the image into smaller crops and computing the semantic similarity between each crop and a query using a pretrained retrieval-augmented generation (RAG) model. The most relevant crops are then selected to localize the target object and suppress irrelevant information. However, such crop-based processing can fragment complete objects across multiple crops, thereby disrupting the computation of semantic similarity. In our experiments, we find that image crops of objects with different sizes are better handled at different resolutions. Based on this observation, we propose Multi-resolution Retrieval-Detection (MRD), a training-free framework for high-resolution image understanding. To address the issue of semantic similarity bias caused by objects being split across different image crops, we propose a multi-resolution semantic fusion method, which integrates semantic similarity maps obtained at different resolutions to produce more accurate semantic information and preserve the integrity of target objects. Furthermore, to achieve direct localization of target objects at a global scale, we introduce an open-vocalbulary object detection (OVD) model that identifies object regions using a sliding-window approach.Experiments on high-resolution image understanding benchmarks using different MLLMs demonstrate the effectiveness of our approach.

## Full Text


<!-- PDF content starts -->

MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image
Understanding
Fan Yang
Harbin Institute of Technology (Shenzhen)
25b951055@stu.hit.edu.cnKaihao Zhang
Harbin Institute of Technology (Shenzhen)
super.khzhang@gmail.com
Abstract
Understanding high-resolution images remains a signif-
icant challenge for multimodal large language models
(MLLMs). Recent study address this issue by dividing
the image into smaller crops and computing the seman-
tic similarity between each crop and a query using a pre-
trained retrieval-augmented generation (RAG) model. The
most relevant crops are then selected to localize the tar-
get object and suppress irrelevant information. However,
such crop-based processing can fragment complete objects
across multiple crops, thereby disrupting the computation
of semantic similarity. In our experiments, we find that im-
age crops of objects with different sizes are better handled at
different resolutions. Based on this observation, we propose
Multi-resolution Retrieval-Detection (MRD), a training-
free framework for high-resolution image understanding.
To address the issue of semantic similarity bias caused by
objects being split across different image crops, we propose
a multi-resolution semantic fusion method, which integrates
semantic similarity maps obtained at different resolutions to
produce more accurate semantic information and preserve
the integrity of target objects. Furthermore, to achieve di-
rect localization of target objects at a global scale, we in-
troduce an open-vocalbulary object detection (OVD) model
that identifies object regions using a sliding-window ap-
proach.Experiments on high-resolution image understand-
ing benchmarks using different MLLMs demonstrate the ef-
fectiveness of our approach.
1. Introduction
Multimodal Large Language Models (MLLMs) have
demonstrated significant advancements in integrating and
interpreting visual and linguistic information, enabling ro-
bust capabilities in vision-language understanding, reason-
ing, and interactive tasks [25]. By leveraging visual signals,
these models can process and decipher complex visual in-
formation, forming a bridge between pixel-level data and
Figure 1. Overview of the proposed Multi-resolution Retrieval-
Detection framework, which uses RAG and OVD to obtain se-
mantic similarity map and detection confidence map respectively.
By integrating the two, the target objects can be localized more
accurately.
semantic interpretation [17, 20]. However, a common prac-
tice among most MLLMs is to process input images at fixed
and pre-defined resolutions [1, 13, 14]. While this uniform
input pipeline simplifies model architecture and reduces
computational overhead, it introduces substantial limita-
tions. Specifically, resizing high-resolution (HR) real-world
images to a fixed low resolution often leads to shape distor-
tion and blurring, which degrades the quality of fine-grained
visual details. Recent studies [11, 19, 22, 24, 29, 30] indi-
cate that existing methods remain unsatisfactory for high-
resolution image tasks. This is clearly demonstrated by their
suboptimal results on dedicated high-resolution image un-
derstanding benchmarks [21, 24].
To address this limitation, improving the high-resolution
image perception capability of MLLMs has become an
emerging research focus. A common ”locate-and-zoom-
in” strategy is widely adopted to enhance detail percep-
tion in models. Although training-based approaches such
as Supervised Fine-Tuning (SFT) [18] and ReinforcementarXiv:2512.02906v2  [cs.CV]  3 Dec 2025

Learning (RL) [30] can effectively identify relevant re-
gions, they are hampered by critical limitations, including
high computational costs, long training cycles, and poor
cross-architecture transferability, which curtails their scal-
ability and practical application. In contrast, training-free
methods [11, 19, 29] automatically locate regions by using
attention mechanisms or tree-based search, without requir-
ing the construction of the dataset or the fine-tuning of the
model. Despite employing a top-down search strategy from
high to low resolution, these methods face significant limi-
tations. A primary issue is the model’s inadequate percep-
tion of small objects during the initial search stage [19, 21],
which frequently leads to the generation of erroneous search
paths.
Recently, Inspired by the success of Retrieval-
Augmented Generation (RAG) for enabling long-context
understanding in general LLMs [9], Wang et al. [22]
introduced Retrieval-Augmented Perception (RAP) — a
training-free framework designed to enhance MLLMs’ per-
ception of high-resolution images. RAP extends this
paradigm to the visual domain, achieving significant perfor-
mance improvement on high-resolution benchmarks. The
RAP framework consists of three key components: First,
the Visual Retrieval module employs a pre-trained vision
RAG model, VisRAG, to compute semantic similarity be-
tween the query and different image regions (crops), re-
trieving the most relevant ones to reduce noise. Next,
the Spatial-Awareness Layout module preserves the origi-
nal relative spatial relationships among the retrieved crops
when composing them into the model input, maintain-
ing spatial coherence. Finally, the Adaptive Retrieval-
Exploration Search (RE-Search) module dynamically deter-
mines the optimal number of retrieved crops by constructing
a Retrieval-Exploration Tree (RE-Tree), balancing informa-
tion sufficiency with computational efficiency.
Despite its promise, the RAP method suffers from sev-
eral inherent limitations.First, the patching operation can
fragment large objects across multiple disjointed crops, dis-
rupting their holistic semantics and leading to biased simi-
larity calculations. Our empirical observations confirm that
some patches semantically irrelevant to the query can ob-
tain abnormally high similarity scores.Second, the patch
resolution is a critical yet difficult-to-tune hyperparameter:
overly large patches introduce redundant background infor-
mation, while overly small ones exacerbate object fragmen-
tation. Experiments show that the choice of resolution sig-
nificantly impacts performance.Third, in high-resolution
images with cluttered backgrounds, the similarity measure
is prone to false positives, where background regions may
attain higher similarity than those containing the actual tar-
get objects, severely hampering recognition.
To tackle these challenges, we propose a novelMulti-
resolution Retrieval-Detection (MRD)framework, basedon RAP to improve retrieval quality and localization accu-
racy through two key techniques:
•Multi-resolution Semantic Fusion: To mitigate the bias
inherent in single-resolution patching, we design a sim-
ple yet effective fusion strategy. It computes semantic
similarities across multiple proportional resolutions and
performs consistency-based fusion to calibrate the results,
yielding a more robust and accurate relevance estimation
that alleviates semantic deviations caused by object frag-
mentation.
•Open-vocabulary Detector Enhancement: For more
precise target localization, we incorporate an advanced
open-vocabulary object detector, LLMDet [7]. First,
we leverage the in-context learning capability of LLMs
to extract target concepts from the query, defining the
categories for the detector. Subsequently, a sliding win-
dow mechanism is employed to traverse the entire high-
resolution image, detecting target objects within each
window to generate a confidence map indicating target
presence.
Finally, the calibrated multi-resolution semantic similar-
ity is augmented by the object detection confidence. This
synergistic fusion effectively amplifies the response in true
target regions, enabling faster and more accurate localiza-
tion of critical areas during subsequent retrieval, thereby
guiding the MLLM toward more reliable inference.
We conduct extensive experiments on several high-
resolution benchmarks, including V* [24], HRBench-4K,
and HRBench-8K [21], utilizing various MLLMs such
as LLaV A-ov and LLaV A-v1.5. The results demonstrate
that our MRD framework surpasses all existing training-
free methods and achieves state-of-the-art performance on
both single-object and multi-object retrieval and recognition
tasks, with particularly notable gains on single-object tasks.
Our contributions are summarized as follows:
• To the best of our knowledge, this is the first work
that systematically leverages an open-vocabulary object
detector to enhance MLLMs’ understanding of high-
resolution images. Experiments validate that the detector
provides precise target localization, effectively suppress-
ing interference from irrelevant regions.
• We proposeMRD, a training-free and generic frame-
work. It innovatively corrects semantic similarity via a
multi-resolution fusion strategy and integrates open-set
detection results to enhance target regions, creating a syn-
ergistic effect.
• Comprehensive experiments validate the effectiveness
and generalization of our method. It achieves lead-
ing performance on both single-object and multi-object
tasks across different MLLMs and high-resolution bench-
marks.

2. Related Work
2.1. Multimodal Large Language Models
MLLMs have rapidly advanced as powerful foundation
models capable of understanding and generating multi-
modal content across diverse vision–language tasks [25,
28]. Early MLLM architectures generally adopt fixed-
resolution vision encoders—such as224×224or448×448
ViTs [2, 12–14]. While this design simplifies training
and computation, it inevitably requires resizing or crop-
ping high-resolution (HR) images, thereby discarding fine-
grained visual details crucial for tasks such as fine-grained
recognition, dense reasoning, or detecting small objects.
To enhance high-resolution image understanding with-
out proportionally increasing the computational burden
from visual tokens, several studies have integrated high-
resolution visual encoders into MLLMs. For example, Vary
[23] and Deepseek-VL [15] incorporate the SAM encoder
[10] to improve model performance on HR images.
Alternatively, another line of work introduced
Native/Dynamic-Resolution MLLMs that processes
images at their native resolution. The core idea is to gener-
ate a variable-length sequence of visual tokens that adapts
to the original dimensions of the input image, thereby
preserving spatial fidelity and high-frequency details.
These models employ various mechanisms to handle the
resulting long sequences and computational complexity,
including sliding window attention, dynamic masking, and
patch-based encoding strategies. Representative works in
this category have demonstrated significant progress. For
instance, the InternVL series [4, 6, 31] adopts a strategy
of splitting a high-resolution image into multiple fixed-size
patches. In contrast, models like Qwen2.5-VL [2, 3]
take an end-to-end approach by training a ViT directly on
native-resolution images. This allows the model to embed
the entire image into a single, coherent token sequence
in one forward pass, potentially leading to better global
context understanding.
2.2. High-Resolution Image Understanding
Multimodal large language models (MLLMs) have made
substantial progress in recent years; however, they con-
tinue to face challenges in accurately recognizing and in-
terpreting fine-grained details within high-resolution (HR)
images [21?]. To enhance the capability of MLLMs
in high-resolution image understanding, existing studies
generally follow two main directions. Training-based ap-
proaches rely on supervised fine-tuning (SFT) [18, 24] or
reinforcement learning (RL) [30], but such methods often
compromise the model’s generalization ability on broad vi-
sion–language tasks. In contrast, training-free approaches
[11, 19, 21, 22]typically perform hierarchical or tree-based
search to localize target regions. However, these methodstend to suffer from low efficiency and may fail to retain
all target objects during the search process, particularly in
multi-object scenarios.
3. Preliminary
In this section, We first conduct an analysis of the relation-
ship between the resolution of image crops and the perfor-
mance of MLLMs in subsection 3.2. The experimental re-
sults indicate that using different resolution has a significant
impact on MLLMs to analyze HR images. Objects of dif-
ferent sizes are suitable for different resolutions. Inspired
by this we propose the MRD framework.
3.1. Semantic Similarity
This section presents the pipeline for integrating Retrieval-
Augmented Generation (RAG) into Multimodal Large Lan-
guage Models (MLLMs) to calculate the semantic sim-
ilarity scores between the query embedding and images
crops from HR images. Given an HR image, we first
partition it collection of image patches, denoted asP=
{p1, p2, . . . , p n}, wherenis the total number of image
crops. Following the approach of Yu et al. [27], the tex-
tual query and each image crop are encoded independently
using the text and image encoders of a Vision-Language
Model (VLM), producing a sequence of hidden represen-
tations. The semantic similarity score between the query
embedding and each image crop embedding is then com-
puted. Specifically, the similarity scores(q, p i)for thei-th
crop is calculated by the cosine similarity of the query and
image crop embeddings:
s(q, p i) =1
2·(1 +q·pi
∥q∥ · ∥p i∥)(1)
Based on these scores, the topKmost relevant image
crops are selected and provided to the MLLM to support
detailed understanding of the high-resolution input.
3.2. Impact of the Resolution of Image Crops
In this section, we conduct an analysis to investigates the
relationship between the resolution of image crops and the
performance of MLLMs in HR image understanding.
Experimental setting.We analyze the relation between
performance and the resolution of image crops, using
LLaV A-ov and LLaV A-v1.5 on V * benmark.
Observations.We visualize the relationship between the
resolution of image crops and performance of MLLMs. As
shown in Figure 3, from the overall accuracy obtained by
using different resolutions, when the resolution of image
crops is set to 112, using different MLLM achieved the
highest accuracy rates in both the single-object task and the
multi-object task. This indicates that setting the resolution
to 112 might be the optimal choice. However, when we take

Figure 2. Setting the resolution of image crops to 112 causes complete objects to be split across different regions, which disrupts the
semantic information of the target objects.
Figure 3. The effect of the resolution of retrieved image crops on
model performance. Attribute and Spatial represent the attribute
recognition and spatial reasoning inV∗Bench.
a closer look at the results of each sample, we find that in
some cases, choosing a different resolution actually leads
to more accurate results compared to when the resolution is
112, as shown in Figure 3. The results of the image crops
selected based on different resolutions and the visualization
of the semantic similarity map can provide a very intuitive
analysis of the reasons: Since the complete object is divided
into different crops, some parts of it have a higher semantic
similarity calculated by VisRAG, while other parts have a
lower semantic similarity. After screening, only the parts
with higher similarity are retained. However, this will dam-
age the integrity of the target object and cause interference
to the judgment of MLLM.
4. Method
In this section, we propose a novel framework named
Multi-resolution Retrieval Detection (MRD). The core
design of MRD lies in its multi-resolution approach at dif-
ferent scales to better localize regions containing target ob-jects. This enables subsequent search processes to more
easily identify image crops corresponding to the target ob-
jects, eliminating irrelevant distractions and enhancing the
perceptual understanding of HR images by MLLMs. Based
on the findings in subsection 3.2, we argue that using dif-
ferent resolutions for semantic similarity computation is
more suitable for objects of varying sizes and locations. In-
spired by this idea, we first introduce a simple yet effective
Multi-resolution Semantic Fusion method, which computes
semantic similarity maps at different resolutions on a lo-
cal scale and performs consistency-based fusion to refine
the semantic similarity and improve its accuracy. To more
directly localize target objects, we incorporate an Open-
vocabulary object detection model that traverses the en-
tire HR image globally using a sliding window approach,
generating confidence scores for regions containing target
objects. Finally, by integrating the detection confidence
scores with the multi-resolution semantic similarity maps,
our method not only improves localization of target regions
but also distinguishes fine-grained differences among crops
in these regions, thereby assisting subsequent search pro-
cesses in more accurately identifying key areas. The fol-
lowing sections will provide detailed explanations of each
component.
4.1. Multi-resolution Semantic Fusion
In subsection 3.2, we observe that image crops of different
resolutions are suitable for objects of varying sizes and loca-
tions in different cases. Compared to the semantic similarity
map obtained using a single resolution, those derived from
multiple resolutions exhibit respective advantages. There-
fore, we first propose a Multi-Resolution Semantic Fusion
method. As shown in the top part of Figure 4, we partition
the HR image using proportional resolutions, with the low
resolution set toland the high resolution set to ˆl, where

Figure 4. Detailed information of our propsoedMRD. First, We use VisRAG with different resolution of image crops to obtain multi-
resolution semantic similarity map. We then employ an open-set object detection model, LLMDet, to localize the target objects extracted
from the query within the high-resolution image using a sliding-window approach, yielding a global detection confidence map. Finally,
the obtained multi-resolution semantic similarity map is linearly fused with the detection confidence map, and the fused scores are used to
guide the subsequent search to select image crops containing the target objects.
ˆl=k·l. The set of image patches at high resolution is
denoted as ˆP={ˆp 1,ˆp2, . . . ,ˆp m}, and at low resolution as
P={p 1, p2, . . . , p n}. Due to the proportional relationship
between high and low resolutions, we haven=k2·m,
and each high-resolution patchˆp icorresponds tok2low-
resolution patches:
ˆpi=
˜pi,1 ˜pi,2 ···˜p i,k
˜pi,(k+1) ˜pi,(k+2) ···˜p i,(2k)
............
˜pi,(k(k−1)+1) ˜pi,(k(k−1)+2) ···˜p i,k2
(2)
where˜p ij∈P, i∈ {1,2, . . . , m}, j∈ {1,2, . . . , k2}.
Then, using Equation 1, we compute the cosine similar-
ity between the query and image crop embeddings to ob-
tain the semantic similarity scores for high and low res-
olutions, respectively: ˆS={ˆs 1,ˆs2, . . . ,ˆs m}andS=
{s1, s2, . . . , s n}:
ˆsi=s(f(q), g(ˆp i)), s j=s(f(q), g(p j))(3)
wheref(·)andg(·)denote the embedding operations
for the query and image crop, respectively, withi∈
{1,2, . . . , m}andj∈ {1,2, . . . , n}. According to the map-
ping fromˆp itopjin (2), we can map the semantic similarity
ˆsobtained from each high-resolutionˆpand the query to thecorrespondingk2low-resolutionppositions. The mapping
operation can be expressed as:
˜S=H( ˆS)(4)
where ˜S={˜s 1,˜s2, . . . ,˜s n}. After obtaining ˜SandS,
we perform consistency fusion on the semantic similarities
at corresponding positions to obtain the multi-resolution se-
mantic similaritySf={sf
1, sf
2, . . . , sf
n}, which can be ex-
pressed as:
sf
t=p
˜st·st, t∈1,2, . . . , n(5)
Finally, we can transform the semantic similarity scores into
a two-dimensional semantic similarity mapsf(i, j)withi∈
{1,2, . . . , H}andj∈ {1,2, . . . , W}. The total number of
low resolution image cropn=H×W.
Fusing the multi-resolution semantic similarity scores
from high and low resolutions enables correction of the low-
resolution similarities when a complete object is split across
different patches in the low-resolution view. This enhances
the similarity of various parts of the object, thereby preserv-
ing the integrity of the object as much as possible during
subsequent search processes and improving the recognition
accuracy of the MLLM.
4.2. Open-vocabulary Detector Enhancement
VisRAG divides the HR image into patches and computes
semantic similarity between the query and each image crop,

enabling localized object retrieval at a small scale. How-
ever, it struggles to accurately localize larger objects. To
address this limitation and achieve more direct and large-
scale object localization, we introduce an advanced open-
vocabulary object detection model—LLMDet—to directly
locate regions containing target objects. First, we employ
in-context learning with a Large Language Model (LLM)
to extract the primary target objects from the query, which
serve as the target categories for LLMDet. Due to the ex-
tremely high resolution of HR images in datasets such as
HR-Bench, we adopt a sliding window strategy for object
localization. To align with the semantic similarity map de-
rived from image crops, we assign the detection confidence
scores of the target bounding boxes to their corresponding
image patches, thereby generating a detection map that re-
flects the confidence of target object presence in each patch.
This detection map offers a more intuitive localization rep-
resentation compared to the semantic similarity map. In the
following, we provide a detailed introduction to the pro-
posed method.
Object Extraction.To enable open-vocabulary object de-
tection, we first leverages Large Language Models (LLMs)
to dynamically identify target objects from textual queries.
Given an input queryQ, we employ in-context learning to
extract the primary object entities that serve as detection tar-
gets for our LLMDet framework.
Formally, we define the object extraction process as:
O=LLM(Psystem,Eexamples, Q)(6)
whereOrepresents the set of extracted objects,Psystem
denotes the system prompt containing extraction guidelines,
andEexamples constitutes the demonstration examples.
Sliding-window Object Detection.In order to get a global
detection confidence map align with the previously seman-
tic similarity map, we similarly partition the HR image into
a grid ofH×Wnon-overlapping patches where total num-
ber of patchesn=H×W. A sliding window of sizeh×w
patches (whereh < Handw < W) traverses the entire
image with a predefined stride. In this way, We can obtain
Tsliding windowsW={W 1, W 2, ..., W T}.
After obtaining multiple sliding windows using the slid-
ing window method, we use LLMDet to detect objects
within each sliding window. The detector generates a set
of bounding boxesB t={b 1, b2, . . . , b Kt}and the corre-
sponding confidence scoress kindicating the likelihood of
containing a target object wheretdenotes thet-th sliding
window.
We apply a confidence thresholdτto filter out low-
quality detections:
Bfilter
t={b k∈ Bt|sk> τ}(7)
Subsequently, we generate a window detection confi-
dence mapcw
t∈Rh×wfor the current window. The valueat patch coordinate(p, q)within this local window is as-
signed the maximum confidence score among all bounding
boxes inBfiltered
t that contain this patch. If no box covers the
patch, the confidence is set to 0:
cw
t(p, q) = max
bk∈Bfilter
t{sk·I[(m, n)∈b k]}(8)
whereI[·]is the indicator function that equals 1 if the
patch(p, q)is inside the bounding boxb k, and 0 otherwise.
To aggregate information from all sliding windows and
form a global, unified detection confidence mapcg∈
RH×Wfor the entire high-resolution image, we employ an
averaging fusion strategy. For a global patch at coordinate
(p, q), its final confidence score is computed as the average
of all confidence scores assigned to it from every sliding
window that contained it.
For the patchI(i, j)at position(i, j)in the HR image,
if it is contained in thet-th sliding window, we denote its
position in thet-th sliding windowW tas(t i, tj), which
can be expressed as
I(i, j) =W t(ti, tj), t∈ T i,j (9)
whereT ijdenotes the set of sliding windows that contain
I(i, j). Now, we can obtain the global detection confidence
map of the whole HR image, which can be computed as:
cg(i, j) =1
|Ti,j|X
t∈Ti,jcw
t(ti, tj)(10)
wherei∈ {1,2, . . . , H}andj∈ {1,2, . . . , W}.
The detection confidence map provides effective local-
ization of target regions on a global scale, offering direct
spatial guidance but lacking the ability to distinguish fine-
grained differences within the target object. To address this
limitation, we integrate the detection confidence with multi-
resolution semantic similarity through linear combination,
which can be expressed as:
sF(i, j) = (1−w)·sf(i, j) +w·cg(i, j)(11)
This synergistic fusion enables precise target localiza-
tion while effectively highlighting intra-object variations,
thereby facilitating more accurate extraction of key regions
in subsequent search processes. The details of the subse-
quent Retrieved-Exploration Search process can be found
in paper [22].
5. Experiments
Evaluated benchmark.We evaluate ourMRDon two
high-resolution benchmarks. The first isV∗Bench[24],
with an average resolution of2246×1582, consists of
two sub-tasks: attribute recognition and spatial reasoning.
The Second is HRBench which includes two sub-task Fine-
grained Single-instance Perception (FSP) and Fine-grained
Cross-instance Perception (FCP).

Table 1. Comparison ofMRDwith existing works on high-resolution benchmarks
MethodV* Bench HR-Bench 4K HR-Bench 8K
Attribute Spatial Overall FSP FCP Overall FSP FCP Overall
Open-source MLLMs
LLaV A-v1.6-7B [14] 60.9 63.2 61.8 49.0 46.8 47.9 37.3 44.3 40.8
LLaV A-v1.6-13B [14] 60.0 64.5 61.8 49.8 41.3 45.5 38.0 38.3 38.1
LLaV A-v1.6-34B [14] - - - 55.3 50.5 52.9 44.5 50.3 47.4
LLaV A-HR-X-13B [16] - - - 61.3 46.0 53.6 49.5 44.3 46.9
LLaV A-HR-X-7B [16] 51.3 64.5 56.5 57.8 46.3 52.0 42.0 41.3 41.6
InternVl-1.5-26B [5] - - - 69.5 51.8 60.6 69.3 48.5 57.9
Yi-VL-34B [26] - - - 46.0 42.8 44.4 39.5 38.5 39.0
Closed-source MLLMs
GPT-4o [8] - - 66.0 70.0 48.0 59.0 62.0 49.0 55.5
Qwen-VL-max [2] - - - 65.052.058.5 54.051.052.5
Baselines and MRD
LLaV A-v1.5-7B [14] 43.5 56.6 48.7 38.5 33.8 36.1 33.0 31.3 32.1
LLaV A-v1.5-7B-Zoom Eye [19] 83.5 82.9 83.3 67.8 38.8 53.3 65.5 36.0 50.8
LLaV A-v1.5-7B-RAP [22] 90.496.191.1 73.8 40.5 57.1 72.3 35.3 53.8
LLaV A-v1.5-7B-MRD(ours) 97.4 96.1 95.6 76.8 42.7 59.7 72.6 37.2 54.9
LLaV A-ov-0.5B [14] 63.5 64.5 63.9 63.5 39.5 51.5 47.3 38.3 42.8
LLaV A-ov-0.5B-Zoom Eye[19] 85.2 73.7 80.6 75.5 39.8 57.6 68.5 38.3 53.4
LLaV A-ov-0.5B-RAP [22] 80.0 84.2 83.6 80.3 42.3 61.381.845.3 63.5
LLaV A-ov-0.5B-MRD(ours) 89.6 82.9 88.0 84.0 45.2 64.6 81.8 47.3 64.5
Figure 5. Visualization of the Effects of Different Modules in MRD. Upper: Visualization of the Effects of the Multi-resolution Semantic
Fusion Method. Lower: Visualization of the Effects of the Multi-resolution Semantic Fusion Method
5.1. Main Results
As shown in Table 1, compared with both the baseline
MLLMs and previous baseline approaches, our proposed

MRD framework consistently delivers substantial perfor-
mance gains across all sub-tasks, datasets, and model con-
figurations. The improvement is most pronounced on the
V∗dataset using the LaV A-v1.5-7B model, where MRD
achieves a remarkable 46.9% absolute increase in accu-
racy—nearly doubling the original performance. Signif-
icant gains are also observed on HR-Bench 4K and HR-
Bench 8K, with maximum improvements of 23.6% and
22.8%, respectively.
In comparison to the state-of-the-art baseline RAP, MRD
achieves superior performance across all datasets and model
settings, yielding an average improvement of 2.8%. When
examining results across sub-task categories, MRD demon-
strates particularly strong performance on single-object
tasks. We attribute this advantage to the integration of a de-
tection module, which provides more accurate localization
for isolated objects.
Overall, these results indicate that MRD markedly en-
hances the perception and understanding capabilities of
MLLMs when operating on high-resolution images.
5.2. Effect of the Multi-resolution Semantic Fusion
Multi-resolution Semantic Fusion can obtain more accu-
rate information by integrating semantic similarity maps
from different resolutions. From the two cases shown in
the upper part of Figure 5, we can clearly observe that
incorporating multi-resolution semantic fusion allows the
high-resolution semantic similarity map to correct the low-
resolution map, alleviating semantic deviations caused by
different parts of the target object being split across multi-
ple patches. This helps better preserve the integrity of the
target object. The results in the cases demonstrate that the
approach is effective for both single-object and multi-object
tasks. Overall, the experimental results indicate that Multi-
resolution Semantic Fusion provides better adaptability to
objects of different sizes compared to using a single resolu-
tion.
5.3. Effect of Open-vocabulary Object Detection
To achieve more accurate and direct localization of the tar-
get object at a global scale, we introduce an open-set ob-
ject detection model. As shown in lower part of Figure 5,
sliding-window detection results effectively identify the tar-
get object’s location. By combining the detection results
with semantic similarity scores, MRD amplifies the scores
of patches that contain the target object while suppressing
false-positive patches that also exhibit high semantic simi-
larity. This integration facilitates a more efficient and accu-
rate patch retrieval process in subsequent searching.
5.4. Ablation Study
To better understand the contributions of different mod-
ules in ourMRDframework, we conduct ablation studiesTable 2. Ablation study of different module inMRD.
V* Bench∆↑
Attribute Spatial Overall
RAP 80.0 84.2 83.6 -
OVD 84.3 81.6 84.9 +1.3
RAP+Multi-res 82.9 85.2 85.8 +2.2
RAP+OVD 85.2 84.2 86.2 +2.6
RAP+OVD+Multi-Res90.4 85.5 89.3+5.7
on theV∗dataset using the LLaV A-ov-0.5B model. As
shown in Table 2, using the OVD model alone (second
row) yields higher localization accuracy for single-object
tasks, but its performance on multi-object tasks is inferior to
RAP. When RAP employs multi-resolution semantic fusion
(third row), performance improves on both single-object
and multi-object tasks, indicating that multi-resolution se-
mantic fusion can better handle objects of varying sizes
across different scenarios.
Fusing the semantic similarity map obtained from RAP
with the detection confidence map from OVD (fourth row)
significantly improves performance on single-object tasks;
however, the performance on multi-object tasks is even
worse than using OVD alone, suggesting that some target
objects may be lost during the search. By further incor-
porating multi-resolution semantic fusion, performance im-
proves on both single-object and multi-object tasks, demon-
strating the effectiveness of this fusion strategy.
In summary, introducing OVD helps localize single ob-
jects more accurately but may result in missed objects in
multi-object scenarios. Multi-resolution semantic fusion
corrects semantic similarity scores and preserves object
completeness under different conditions, enhancing MLLM
performance on both single- and multi-object tasks. The fi-
nal model, which integrates all modules, achieves a 5.7%
higher accuracy than RAP, demonstrating the effectiveness
of MRD’s design in improving high-resolution image un-
derstanding for MLLMs.
6. Conclusion
In this work, we propose a novel training-free method,
Multi-resolution Retrieval-Detection (MRD), to enhance
the understanding of high-resolution images by MLLMs.
MRD employs multi-resolution semantic similarity to cor-
rect single-resolution similarity maps, ensuring the integrity
of target objects. Moreover, to localize target objects more
accurately and directly, we introduce an OVD model that
identifies object regions using a sliding-window approach.
We demonstrate the effectiveness of MRD across multiple
high-resolution benchmarks with different MLLMs, show-
ing its superior performance in HR image understanding.

References
[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, et al. Qwen technical report.arXiv preprint
arXiv:2309.16609, 2023. 1
[2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A versatile vision-language model for un-
derstanding, localization, text reading, and beyond, 2023. 3,
7
[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report.arXiv preprint
arXiv:2502.13923, 2025. 3
[4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhang-
wei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,
Zhaoyang Liu, et al. Expanding performance boundaries of
open-source multimodal models with model, data, and test-
time scaling.arXiv preprint arXiv:2412.05271, 2024. 3
[5] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhang-
wei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng
Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang
Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin,
Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang,
Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min
Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao,
Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v?
closing the gap to commercial multimodal models with open-
source suites.Sci. China Inf. Sci., 67(12), 2024. 7
[6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen,
Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu,
Lewei Lu, et al. Internvl: Scaling up vision foundation mod-
els and aligning for generic visual-linguistic tasks. InPro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 24185–24198, 2024. 3
[7] Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei,
Jingke Meng, Xiaohua Xie, and Wei-Shi Zheng. Llmdet:
Learning strong open-vocabulary object detectors under the
supervision of large language models. InProceedings of the
Computer Vision and Pattern Recognition Conference, pages
14987–14997, 2025. 2
[8] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perel-
man, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli-
hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.
arXiv preprint arXiv:2410.21276, 2024. 7
[9] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik.
Long-context llms meet rag: Overcoming challenges for
long inputs in rag.arXiv preprint arXiv:2410.05983, 2024.
2
[10] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. InProceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 3
[11] Geng Li, Jinglin Xu, Yunzhen Zhao, and Yuxin Peng. Dyfo:
A training-free dynamic focus visual search for enhancing
lmms in fine-grained visual understanding. InProceedingsof the Computer Vision and Pattern Recognition Conference,
pages 9098–9108, 2025. 1, 2, 3
[12] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. InIn-
ternational conference on machine learning, pages 19730–
19742. PMLR, 2023. 3
[13] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning. InPro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 26296–26306, 2024. 1
[14] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved
reasoning, ocr, and world knowledge, 2024. 1, 3, 7
[15] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai
Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li,
Hao Yang, et al. Deepseek-vl: towards real-world vision-
language understanding.arXiv preprint arXiv:2403.05525,
2024. 3
[16] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xi-
aoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-
of-resolution adaptation for multimodal large language mod-
els. InThe Thirteenth International Conference on Learning
Representations, ICLR 2025, Singapore, April 24-28, 2025.
OpenReview.net, 2025. 7
[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. InInternational conference on machine learning, pages
8748–8763. PmLR, 2021. 1
[18] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuo-
fan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual
cot: Advancing multi-modal language models with a com-
prehensive dataset and benchmark for chain-of-thought rea-
soning.Advances in Neural Information Processing Systems,
37:8612–8642, 2024. 1, 3
[19] Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen
Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoom-
eye: Enhancing multimodal llms with human-like zooming
capabilities through tree-based image exploration. InPro-
ceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 6613–6629, 2025. 1,
2, 3, 7
[20] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue
Cao. Eva-clip: Improved training techniques for clip at scale.
arXiv preprint arXiv:2303.15389, 2023. 1
[21] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li
Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer
and combine: A training-free framework for high-resolution
image perception in multimodal large language models. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence, pages 7907–7915, 2025. 1, 2, 3
[22] Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang,
Li Shen, Yong Luo, Bo Du, and Dacheng Tao. Retrieval-
augmented perception: High-resolution image perception
meets visual rag.arXiv preprint arXiv:2503.01222, 2025.
1, 2, 3, 6, 7

[23] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng
Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu
Zhang. Vary: Scaling up the vision vocabulary for large
vision-language model. InEuropean Conference on Com-
puter Vision, pages 408–424. Springer, 2024. 3
[24] Penghao Wu and Saining Xie. V?: Guided visual search as
a core mechanism in multimodal llms. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 13084–13094, 2024. 1, 2, 3, 6
[25] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun,
Tong Xu, and Enhong Chen. A survey on multimodal
large language models.National Science Review, 11(12):
nwae403, 2024. 1, 3
[26] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang,
Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen,
Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue,
Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao
Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng
Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu
Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation
models by 01.ai.CoRR, abs/2403.04652, 2024. 7
[27] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao
Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han,
Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented
generation on multi-modality documents.arXiv preprint
arXiv:2410.10594, 2024. 3
[28] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan
Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances
in multimodal large language models. InFindings of the
Association for Computational Linguistics ACL 2024, pages
12401–12430, 2024. 3
[29] Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and
Filip Ilievski. Mllms know where to look: Training-free per-
ception of small visual details with multimodal llms.arXiv
preprint arXiv:2502.17422, 2025. 1, 2
[30] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao,
Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deep-
eyes: Incentivizing” thinking with images” via reinforce-
ment learning.arXiv preprint arXiv:2505.14362, 2025. 1,
2, 3
[31] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shen-
glong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su,
Jie Shao, et al. Internvl3: Exploring advanced training and
test-time recipes for open-source multimodal models.arXiv
preprint arXiv:2504.10479, 2025. 3

MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image
Understanding
Supplementary Material
A. Implement Details ofMRD
Following [22], given an input HR imageI, it is first parti-
tioned into smaller image crops based on a predefined crop
size, which corresponds to the preferred resolution of the
retriver. According to the Resolution of HR image, we set
the crop resolutions as 112, 224 and 448 forV∗Bench,
HR-Bench-4KandHR-Bench-8Krespectively. For multi-
resolution semantic fusion, the ratio between the high and
low resolutions is set tok= 2in all experiments. For
Sliding Window detection, we set window size and step
as 1232 and 896 forV∗Bench, 2240 and 1792 forHR-
Bench-4K, 3136 and 2688 for forHR-Bench-8Kto balance
efficiency and accuracy. The weightwof the detection con-
fidence map is 0.4 by default in semantic detection map fu-
sion. For the followingRetrieved-Exploration Search (RE-
Search)process, we adopt the same hyperparameters as the
baseline methodRAP[22]. In all experiments, the maxi-
mum search steps are set to 200, and the answering confi-
dence thresholdτis set to 0.6. In the following hyperpa-
rameter studies, all other hyperparameters of ourMRDuse
their default settings unless otherwise specified.
B. More Experiment Result
To analyze the impact of different hyperparameters on per-
formance, we conduct experiments onMRDusing various
hyperparameter settings, includingCrop Resolution,Max-
imum Search Steps,Detection Weight, andDetection
Window Size. We perform these experiments on theV∗
Benchusing both the LLaV A-ov-0.5B and LLaV A-v1.5-7B
models. Unless otherwise specified, all other hyperparame-
ters follow the default settings mentioned in section A.
B.1. Effect of Crop Resolution
In our experiments, we evaluate the effect of different crop
resolutions on performance. The results are shown in Fig-
ure 6. For the Single Instance Task (Figure 6 (a)), we ob-
serve thatMRDremains highly stable across different reso-
lutions for both models, with only minor performance fluc-
tuations, whereas RAP exhibits much larger variations, es-
pecially when using LLaV A-ov-0.5B.
For the Cross Instance Task, the performance gap be-
tweenMRDandRAPis relatively small when using
LLaV A-v1.5-7B. However, with LLaV A-ov-0.5B,MRDis
largely unaffected by resolution changes, further demon-
strating its robustness. Overall,MRDconsistently outper-
formsRAPacross different resolutions and model settings,highlighting the advantages of our approach.
In summary, the Multi-resolution Semantic Fusion and
Detector Enhancement modules inMRDeffectively miti-
gate the interference caused by fragmenting complete ob-
jects across multiple crops when using different crop reso-
lutions. As a result, ourMRDperformance is only weakly
influenced by crop resolution and achieves notably better
results in the Single Instance Task.
B.2. Effect of Maximum Search Steps
The performance ofMRDandRAPunder different max-
imum search steps is shown in Figure 7. In Figure 7 (a),
for the single-instance task,MRDconsistently outperforms
RAP across different max step settings on both the LLaV A-
ov-0.5B and LLaV A-v1.5-7B models.
In Figure 7 (b), for the Cross Instance Task,MRDis
slightly inferior to RAP only when using LLaV A-v1.5-
7B with small max steps. However, as the max step in-
creases,MRDsurpassesRAPand maintains better perfor-
mance. Overall,MRDachieves superior results compared
toRAP. Notably,MRDwith LLaV A-ov-0.5B performs only
marginally lower thanRAPwith the powerful LLaV A-v1.5-
7B model.
Most importantly,MRDreaches its peak performance
with a significantly smaller number of maximum search
steps (Max Step = 30). This means that in practical ap-
plications,MRDcan operate effectively with fewer steps,
achieving high accuracy while reducing search time and im-
proving efficiency.”
B.3. Effect of Detection Weight
The results of using different detection weights are shown
in Figure 8. We observe that relying solely on the seman-
tic similarity map (weight = 0) or solely on the detection
map (weight = 1) does not yield optimal performance for
either task. In contrast, fusing the two maps leads to better
results, demonstrating that the semantic similarity map and
detection map provide complementary information.
Overall (Figure 8 (c)), the optimal detection weight
varies slightly across models: LLaV A-ov-0.5B achieves its
best performance at weight = 0.4, while LLaV A-v1.5-7B
performs best at weight = 0.2.
B.4. Effect of Window Size
As shown in Figure 9, adopting different sliding-window
sizes for object detection also affects the results. Except
for the Cross Instance Task with LLaV A-ov-0.5B, using a

Figure 6. The effect of the resolution of image crops on model performance. Single and Cross represent the attribute recognition and
spatial reasoning inV∗Bench. (a) Single-instance Task. (b) Cross-instance Task. (c) Overall Performance.
.
Figure 7. The effect of the maximum search steps ofMRDandRAP.
smaller sliding-window size (Window Size = 896) gener-
ally yields better performance. This is because a smaller
window reduces background interference unrelated to the
target object, leading to more accurate detection results.
However, a smaller window size also means that more
windows are required to scan the entire high-resolution im-
age, resulting in increased computational complexity and
longer processing time. Therefore, to balance accuracy and
efficiency, we select a larger sliding-window size, Window
Size = 1232, as the default setting.
B.5. Compared with Other HR Methods
We compare ourMRDapproach with three high-resolution
processing baselinesRAP, DC2and Zoom Eye. DC2is atraining-free framework that improves MLLM comprehen-
sion of HR images by dividing them into crops, generat-
ing textual descriptions for each region, and aggregating
these descriptions to obtain a more complete understand-
ing. Zoom Eye, on the other hand, uses a tree-search strat-
egy to traverse the hierarchical visual structure of an image,
enabling efficient identification and extraction of relevant
information.
As shown in Table 3, all HR processing methods yield
overall performance improvements compared with the base-
line. Among them, ourMRDachieves consistently stronger
results across most tasks, demonstrating its clear advantage
over existing approaches.

Figure 8. The effect of the detection weight inMRD.
Figure 9. The effect of the detection window size inMRD.
Table 3. Comparison ofMRDwith existing works on high-resolution benchmarks. We conduct experiments onV∗BenchandHR-Bench
using LLaV A-v1.5 7B.
MethodV* Bench HR-Bench 4K HR-Bench 8K∆(↑)
Attribute Spatial Overall FSP FCP Overall FSP FCP Overall
LLaV A-v1.5-7B 43.5 56.6 48.7 38.5 33.8 36.1 33.0 31.3 32.1 -
-w/ DC249.6 59.2 51.6 45.3 37.0 41.1 36.5 33.3 34.9 +3.5
-w/ Zoom Eye83.5 82.9 83.3 67.8 38.8 53.3 65.5 36.0 50.8 +23.5
-w/ RAP90.496.191.1 73.8 40.5 57.1 72.3 35.3 53.8 +28.4
-w/ MRD (ours) 97.4 96.1 95.6 76.8 42.7 59.7 72.6 37.2 54.9 +31.1
C. Case Study
C.1. Single-instance Perception Task Examples
Figure 10 shows two single-instance perception cases from
each HR benchmarks usingRAPandMRDon LLaV A-v1.5-7B. From the first to the last column, we show: the
HR image, theRAPsemantic similarity map, the object de-
tection confidence map, theMRDsemantic–detection fu-
sion map, theRAPresult and theMRDresult. From the

visualization ofRAPsemantic similarity maps, we can ob-
serve that due to the crop partitioning, a complete object
may be divided across multiple crops, leading to inconsis-
tencies in semantic similarity among different parts of the
object. This inconsistency interferes with the subsequent
retrieval process. For example, in the second case ofHR-
Bench4K,RAPonly retrieves the right half of the speed-
limit sign, resulting in an incorrect final prediction. In addi-
tion, the semantic similarity maps contain many false posi-
tives; for instance, in the first case ofHR-Bench8K, the sky
region—irrelevant to the query—shows undesirably high
similarity scores.
MRDaddresses these issues by using multi-resolution
semantic fusion to correct the semantic inconsistencies
across different parts of the object, ensuring its complete-
ness. Moreover, by incorporating an object detection model
to directly localize the target,MRDreinforces the similarity
of the true target region while suppressing false positives.
As shown in the figure, theMRDsemantic–detection fu-
sion map exhibits much clearer contrast between the target
and irrelevant regions compared withRAP, significantly re-
ducing false positives and enabling more accurate retrieval
of the target-related crops during the search process.
C.2. Cross-instance Perception Task Examples
Figure 11 shows two cross-instance perception cases from
each HR benchmarks usingRAPandMRDon LLaV A-
v1.5-7B. In the cross-instance task, the retrieval results
show thatRAPoften retains only a subset of the target ob-
jects while ignoring others when multiple objects need to be
localized. For example, in the first case ofV∗Bench,RAP
completely misses the pink umbrella. This omission be-
comes more pronounced when there is a large size discrep-
ancy between different target objects. As seen in the sec-
ond case ofV∗Benchand the two cases fromHRBench-
8K, RAP tends to keep only the larger primary object while
neglecting the smaller ones. Similar issues also appear in
counting scenarios (e.g., the two cases inHRBench-4K),
whereRAPidentifies only a few among multiple instances.
In contrast,MRDleverages object detection to simul-
taneously detect all target objects, ensuring that even small
objects are preserved to the greatest extent. This givesMRD
a clear advantage in cross-instance perception tasks.

Figure 10. Qualitative examples ofSingle-instance Perception task. We conduct experiments using LLaV A-v1.5-7B on three HR Bench-
marks.

Figure 11. Qualitative examples ofCross-instance Perceptiontask.We conduct experiments using LLaV A-v1.5-7B on three HR Bench-
marks.