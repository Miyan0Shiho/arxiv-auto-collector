# First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection

**Authors**: Wutao Liu, YiDan Wang, Pan Gao

**Published**: 2025-08-21 07:14:18

**PDF URL**: [http://arxiv.org/pdf/2508.15313v1](http://arxiv.org/pdf/2508.15313v1)

## Abstract
Camouflaged object detection (COD) poses a significant challenge in computer
vision due to the high similarity between objects and their backgrounds.
Existing approaches often rely on heavy training and large computational
resources. While foundation models such as the Segment Anything Model (SAM)
offer strong generalization, they still struggle to handle COD tasks without
fine-tuning and require high-quality prompts to yield good performance.
However, generating such prompts manually is costly and inefficient. To address
these challenges, we propose \textbf{First RAG, Second SEG (RAG-SEG)}, a
training-free paradigm that decouples COD into two stages: Retrieval-Augmented
Generation (RAG) for generating coarse masks as prompts, followed by SAM-based
segmentation (SEG) for refinement. RAG-SEG constructs a compact retrieval
database via unsupervised clustering, enabling fast and effective feature
retrieval. During inference, the retrieved features produce pseudo-labels that
guide precise mask generation using SAM2. Our method eliminates the need for
conventional training while maintaining competitive performance. Extensive
experiments on benchmark COD datasets demonstrate that RAG-SEG performs on par
with or surpasses state-of-the-art methods. Notably, all experiments are
conducted on a \textbf{personal laptop}, highlighting the computational
efficiency and practicality of our approach. We present further analysis in the
Appendix, covering limitations, salient object detection extension, and
possible improvements.

## Full Text


<!-- PDF content starts -->

First RAG, Second SEG: A Training-Free Paradigm for
Camouflaged Object Detection
Wutao Liu
Nanjing University of Aeronautics
and Astronautics
Nanjing, China
wutaoliu@nuaa.edu.cnYiDan Wang
Nanjing University of Aeronautics
and Astronautics
Nanjing, China
wangyidan@nuaa.edu.cnPan Gao
Nanjing University of Aeronautics
and Astronautics
Nanjing, China
pan.gao@nuaa.edu.cn
Abstract
Camouflaged object detection (COD) poses a significant challenge
in computer vision due to the high similarity between objects and
their backgrounds. Existing approaches often rely on heavy train-
ing and large computational resources. While foundation models
such as the Segment Anything Model (SAM) offer strong general-
ization, they still struggle to handle COD tasks without fine-tuning
and require high-quality prompts to yield good performance. How-
ever, generating such prompts manually is costly and inefficient.
To address these challenges, we propose First RAG, Second SEG
(RAG-SEG) , a training-free paradigm that decouples COD into
two stages: Retrieval-Augmented Generation (RAG) for generating
coarse masks as prompts, followed by SAM-based segmentation
(SEG) for refinement. RAG-SEG constructs a compact retrieval data-
base via unsupervised clustering, enabling fast and effective feature
retrieval. During inference, the retrieved features produce pseudo-
labels that guide precise mask generation using SAM2. Our method
eliminates the need for conventional training while maintaining
competitive performance. Extensive experiments on benchmark
COD datasets demonstrate that RAG-SEG performs on par with
or surpasses state-of-the-art methods. Notably, all experiments are
conducted on a personal laptop , highlighting the computational
efficiency and practicality of our approach. We present further anal-
ysis in the Appendix, covering limitations, salient object detection
extension, and possible improvements.
CCS Concepts
•Computing methodologies →Interest point and salient region
detections ;Image segmentation ;•Information systems →Re-
trieval models and ranking.
Keywords
Camouflaged object detection, Segment Anything Model, Retrieval-
Augmented Generation
ACM Reference Format:
Wutao Liu, YiDan Wang, and Pan Gao. 2018. First RAG, Second SEG: A
Training-Free Paradigm for Camouflaged Object Detection. In Proceedings
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXof Make sure to enter the correct conference title from your rights confirmation
email (Conference acronym ’XX). ACM, New York, NY, USA, 18 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Camouflaged Object Detection (COD) has emerged as a crucial field
in both academic research and industrial applications, with signifi-
cant implications for medical imaging analysis, wildlife protection,
and industrial defect detection. The fundamental challenge in COD
stems from the high visual similarity between target objects and
their environmental context [ 12]. The methodological evolution
in COD has progressed from traditional image processing tech-
niques to modern deep learning approaches. Convolutional Neural
Networks (CNNs), particularly SINet [ 10], marked a significant ad-
vancement, leading to various CNN-based methods [ 9,35,40,46].
More recently, attention mechanisms through Vision Transformers
(ViT) [ 5] and transformer-based architectures [ 19,19,20,49] have
further enhanced performance.
Table 1: Comparison of computational costs across COD
methods. The “Resource” column lists the GPU model and
the corresponding training time (in hours). Bold denotes our
method. A hyphen (‘–’) indicates unavailable information.
(*) For RAG-SEG, the time refers to one-time unsupervised
KMeans clustering using FAISS, rather than model training.
Method Ep. Resource (Time [h]) Arch.
SINet 20[10] 30 TITAN RTX ×1 (1.17) ResNet50 [16]
SINetv2 22[9] 100 TITAN RTX ×1 (4.00) ResNet50 [16]
SAM-Adapter 23[2] 20 A100 ×4 (– ) SAM-ViT-H [24]
DSAM 24[50] 100 3080Ti ×1 (7.5) SAM-ViT-H [24]
SAM2-Adapter 24[1] 20 A100 ×3 (– ) SAM2 [38]
CamoDiff 23[3] 150 A100 ×1 (– ) PVTv2-B4 [44]
RAG-SEG (Ours) 0 Optional (0.13*) SAM2 [38]
Despite these advances, current COD methods still face critical
limitations. Most notably, their reliance on extensive computational
resources—often demanding training for over 100 epochs-raises ma-
jor concerns about environmental sustainability and resource
efficiency (Table 1). In addition, they depend on high-end GPUs
to achieve state-of-the-art performance. Recent developments, such
as [1,2,31], have pushed performance boundaries by implementing
a large-scale model with an extremely long training phase. How-
ever, the trend toward increasingly complex models and prolonged
training raises concerns about the sustainability and scalability of
current approaches.arXiv:2508.15313v1  [cs.CV]  21 Aug 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Train ImageModel
(+Adapter)Prediction Ground Truth
Test ImageModel
(+Adapter)Prediction
Train ImagesFeature 
ExtractorVector Database
(token vector , mask value)Ground Truths
Test Image RAG SystemInitial
PredictionSegmentation
ModelPredictionTraining Phase
Infer ence Phase
Build RAG System
Infer ence PhaseTraditional Pipeline
RAG-SEG Pipeline
❄❄❄
Data Flow Process Flow Supervision Flow🔥
❄🔥 Tunable Frozen
Figure 1: Comparison between conventional and RAG-SEG
approaches for camouflaged object detection. Unlike conven-
tional methods that rely on supervised fine-tuning, RAG-SEG
offers a training-free alternative via feature-based retrieval
and SAM2-guided refinement.
To address these challenges, we propose First RAG, Second
SEG (RAG-SEG) , a novel training-free framework that harnesses
off-the-shelf foundation models without requiring extensive train-
ing. RAG-SEG runs efficiently on standard laptops and eliminates
SAM’s dependence on specialized adapters for satisfactory COD
performance. Recognizing that SAM’s performance hinges
critically on prompt quality—and that crafting such prompts
manually is both time-consuming and labor-intensive—we
integrate Retrieval-Augmented Generation (RAG) to extract
key camouflage cues automatically, which then guide SAM’s
segmentation. Figure 1 compares the conventional and RAG-SEG
methods for COD. We make the following contributions:
(1)Introduction of RAG-SEG. We present RAG-SEG, the first
of its kind retrieval-augmented generation (RAG) paradigm
for object segmentation. By combining RAG’s ability to mine
domain-specific cues with the promptable power of SAM,
RAG-SEG inaugurates a training-free segmentation pipeline
that efficiently harnesses foundation models.
(2)Training-free COD on a laptop. We propose a zero-training
framework for camouflaged object detection (COD) that runs
entirely on an affordable laptop, eliminating the need for
GPUs or any traditional training. Extensive experiments on
real-world COD benchmarks demonstrate that our method—–
despite its zero-training design—matches the performance
of existing supervised training approaches.
(3)Comprehensive empirical validation. We carry out thor-
ough ablation studies and comparisons against state-of-the-
art COD methods, confirming that (i) RAG-SEG’s retrieval
components supply SAM with high-quality prompts and
(ii) the overall pipeline sustains competitive accuracy with
orders-of-magnitude lower resource consumption.
In addition, we give in-depth analysis of the RAG-SEG pipeline
in Appendix, which provides more analysis, extends RAG-SEG to
salient object detection, and outlines future research directions.2 Related Works
2.1 Camouflaged Object Detection
Camouflaged Object Detection (COD) addresses the challenge of
identifying objects that visually assimilate into their surroundings.
Traditional approaches relied on handcrafted features [ 12], which,
while pioneering, showed limited effectiveness in complex scenar-
ios where objects and backgrounds exhibited subtle distinctions.
The advent of deep learning, particularly Convolutional Neural Net-
works (CNNs), marked a significant advancement in COD. While
CNNs enhanced spatial feature extraction capabilities, they faced
limitations in capturing long-range dependencies crucial for com-
plex camouflage scenarios. This led to the adoption of transformer-
based architectures, which excel at capturing global context but gen-
erally demand substantial computational resources. Recent progress
has been driven by comprehensive datasets like COD10K [ 10] and
NC4K [ 32]. While many methods [ 9,10,39] rely on conventional
supervised learning approaches, researchers have explored various
complementary features to enhance detection precision. These in-
clude depth information [ 42,45], frequency domain features [ 4],
edge cues [ 15,22], and gradient features [ 21]. A comprehensive
review can be found in [ 11]. While these improvements have led to
better detection accuracy, they also come with notable trade-offs in
terms of computational resources.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ 14] has emerged as an
effective approach for mitigating model hallucination in Natural
Language Processing (NLP) and multimodal tasks. While RAG has
demonstrated success in multimodal document retrieval, its
potential in segmentation tasks remains largely unexplored.
Our work aims to bridge this gap by introducing a novel RAG-based
approach for generating reliable segmentation pseudo-labels.
2.3 Foundation Models
Foundation Models (FMs) have transformed the machine learn-
ing landscape through their extensive pre-training on large-scale
datasets. While recent COD methods have explored Parameter-
Efficient Fine-Tuning (PEFT [ 33]) approaches using models like
DINOv2 [ 36], SAM [ 24], and SAM2 [ 38], these adaptations often in-
cur significant computational costs and environmental impact due
to gradient storage and training requirements. In contrast, our pro-
posed RAG-SEG leverages existing FMs without additional training
overhead, enabling efficient deployment even on a laptop.
3 Method
The challenge of COD segmentation stems from the scarcity of
labeled datasets and the high resemblance between foreground and
background objects. To achieve competitive performance in this
task, existing approaches rely on large backbones, extensive param-
eterization, and long training durations. However, these methods
come at the cost of high computational overheads andsignifi-
cant environmental impacts due to the large amount of energy
required for training deep learning models.
Due to the high visual similarity between camouflaged fore-
grounds and their backgrounds, these off-the-shelf models (SAM [ 24],

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
SAM2 [ 38]) often fail to produce reliable masks in COD scenarios
[1,2]. Rather than designing a separate prompt-tuning network or
fine-tuning SAM itself, we propose leveraging Retrieval-Augmented
Generation (RAG) [ 14] to supply SAM with targeted, memory-
driven prompts. By maintaining a database of prototype camouflage
patterns and using RAG’s retrieve-and-generate to extract salient
cue fragments, we can craft context-aware prompts that guide SAM
toward accurately delineating camouflaged objects without any
additional model training. Our method, RAG-SEG framework, com-
bines the strengths of RAG and SAM, offering a more efficient and
carbon-neutral solution for COD segmentation.
3.1 Preliminary
Retrieval-Augmented Generation (RAG) [ 14] enhances neural gen-
eration by integrating external knowledge retrieval. It comprises a
retriever and a generator : the retriever, often based on dense passage
retrieval [ 23], maps queries into a dense vector space to retrieve rel-
evant context, which the generator then conditions on to produce
informed outputs.
In this work, we adapt RAG to improve segmentation perfor-
mance by utilizing external knowledge stored in a vector database.
This approach eliminates the need of expensive retraining, provid-
ing a more efficient and sustainable solution. The overall process
follows these key steps: (1). Vector Database Construction : An
embedding model (can be dubbed feature extractor) extracts fea-
ture vectors to store in a vector database. In our case, we store the
feacture vectors and their corresponding ground truth mask values
from training images with DINOv2 [ 36].(2). Query Embedding
Generation : Given a test image, the embedding model (DINOv2)
extracts its feature vectors. (3). Retrieval : The system searches
the vector database using nearest-neighbor algorithms to identify
the most relevant entries based on the features. (4). Generation :
After obtaining the retrieved vectors and their corresponding mask
values, we reshape and upsample them to generate the initial pre-
diction. (5). Optimization : Although RAG can yield more accurate
results, the initial prediction is still imperfect. It is necessary to opti-
mize the result using some techniques, like normalization, filtering,
and prompt engineering.
3.2 Overview of Framework
Directly applying SAM to COD is often suboptimal [ 1,2], since SAM
has not been trained on camouflage data and therefore struggles to
locate objects that blend into their surroundings. However, SAM
possesses a strong ability to interpret well-crafted prompts. We
hypothesize that, by supplying SAM with prompts that encode the
subtle but essential cues distinguishing camouflaged objects, we can
enable it to detect them without any additional training. To realize
this training-free paradigm, we employ a retrieval-augmented gen-
eration (RAG) framework to automatically generate such prompts.
To the best of our knowledge, this is the first work to design an en-
tirely training-free method that leverages RAG-generated prompts
to adapt SAM for COD.
Our proposed RAG-SEG framework, shown in Figure 2, intro-
duces a two-stage approach for COD that eliminates the need for
traditional training while maintaining competitive segmentation
accuracy. After building the vector database, the final segmentationprediction is obtained through two stages: 1. First Stage—RAG :
In this stage, a feature extractor is used to generate a query from
the test image, which is then used to retrieve relevant information
from the vector database. This retrieval process generates initial
predictions with low computational and time complexity. 2. Second
Stage—SEG : The pseudo-label generated in the previous stage is
used as a prompt for the segmentation model, SAM2, after undergo-
ing post-processing steps such as thresholding. The segmentation
mask is then finally produced.
3.3 RAG-based Pseudo-label Generation
This section presents RAG-based pseudo-label generation in three
steps: feature extraction, storage optimization, and retrieval-based
generation.
3.3.1 Feature Extraction. To construct the vector database, we use
the DINOv2 [ 36] Small model as the feature extractor ( FE). DINOv2
Small is chosen for its efficient performance and relatively small
parameter count, offering results comparable to ResNet50 [ 16], but
with superior image representations due to its self-supervised train-
ing approach. An alternative, such as using a ResNet-based feature
extractor, would generate a 7×7grid of tokens, resulting in only
49 token-mask pairs per image, which has less image representa-
tion. Moreover, using ResNet could yield inaccurate downsampled
masks, negatively impacting performance. This is why we avoided
using a pyramid-style ViT backbone [30, 43, 44].
For each image in the training set, we extract feature vectors
from the final layer of the feature extractor and pair them with
corresponding downsampled mask regions, yielding vector-mask
pairs. The input images are resized to 224×224, and with a patch size
of 14, the extracted feature vectors form a 16×16grid. As a result,
each image generates 256 feature vectors, each corresponding to
a resized 16×16mask. For simplicity and resource efficiency, we
omit the use of class tokens. The feature vectors and corresponding
mask values are formally represented as follows:
D={(𝑣𝑖,𝑚𝑖)|𝑣𝑖=FE(I,𝑡𝑖),𝑚𝑖∈[0,1],𝑖∈{1,2,...,𝑁}}.
Here, Idenotes the input image, and 𝑡𝑖represents the 𝑖-th token
(patch) of the image. Each 𝑣𝑖is the feature vector corresponding
to the token 𝑡𝑖, extracted by the feature extractor. The value 𝑚𝑖is
the corresponding mask value for the token 𝑡𝑖, which lies within
the range[0,1]due to the downsampling process. Finally, 𝑁is the
total number of tokens (or patches) in the image.
3.3.2 Optimizing Storage. Despite the resource efficiency of DI-
NOv2 Small, the size of the resulting vector database still remains
substantial. With 4040 images, and each image generating 256
vector-mask pairs, the total number of vector-mask pairs can be
calculated as: 4040×256≈10.3424×107≈103.424 million . The
scale of this dataset presents challenges, particularly in terms of
retrieval time and the computational resources required for subse-
quent processing. To mitigate these issues, we apply unsupervised
clustering—specifically the KMeans algorithm to compress the data-
base while retaining its representative capacity for segmentation
tasks. The clustering process is formalized as follows:
C=KMeans(D,𝑘=𝐾),

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Feature Extractor
(DINOv2-small )
Segmentation
Model
(SAM2)Build RAG System
Infer ence PhaseRAG-SEG Pipeline
❄
Data Flow Process Flow Post process & Prompt Flow ❄ Frozen
Train ImagesVector Database
(token vector , mask value)
Downsample
Ground Truth
Test Image
Prediction
Feature Extractor
(DINOv2-small )
RetrievalGenerationReshape
UpsampleVector Database
Initial PredictionFirst RAG ❄
token vectormask valueSecond SEG❄
Figure 2: Architectural overview of the proposed RAG-SEG, consisting of two stages: (1) RAG system construction via feature
extractor and vector database indexing; and (2) inference through retrieval-based mask generation and refinement with SAM2.
whereDdenotes the original set of vector-mask pairs, Crepresents
the clustered set, and 𝐾is the number of clusters. Experimental
results indicate that setting 𝐾=4096 achieves a favorable trade-off
between storage reduction and segmentation accuracy.
This clustering approach significantly reduces storage require-
ments and enhances retrieval efficiency, enabling faster access to
relevant information (vector-mask pairs). The theoretical basis for
this method lies in the observed similarity between adjacent patches
and their corresponding masks within the same image, as well as
the similarity across different images within the same task.
3.3.3 Retrieval and Generation. Given a query image, we extract
token-wise features and retrieve the top- 𝑘most similar vector-mask
pairs from the database using a similarity function 𝑓(e.g., L2, Inner
Product (IP), or Cosine). In practice, IP yields the best results.
Each stored vector v𝑗is assigned a mask value 𝑚𝑗∈[0,1]. For
each query token q𝑖, its pseudo-label ˆ𝑚𝑖is computed by averaging
the mask values of the top- 𝑘retrieved vectors from the database:
ˆ𝑚𝑖=1
𝑘∑︁𝑘
𝑗=1𝑚𝑗.
Empirically, using 𝑘=1(i.e., nearest neighbor) achieves the best
segmentation performance while simplifying computation.
The final segmentation mask ˆ𝑀𝑞is formed by aggregating pseudo-
labels across all token positions:
ˆ𝑀𝑞={ˆ𝑚𝑖|𝑖∈[1,𝑁]}.
This retrieval-based generation produces high-quality pseudo-labels
without requiring model fine-tuning, leveraging external knowl-
edge encoded in the feature database.
3.4 SAM-based Refinement
While RAG-based segmentation provides robust coarse localization
by exploiting recurring patterns such as occlusion and texture (see
Figure 4), its output masks often lack saturation and fine structuraldetail. Traditional post-processing methods, e.g., Conditional Ran-
dom Fields (CRF) [ 25], are ineffective due to the similar appearance
statistics of foreground and background in camouflaged scenes.
Conversely, off-the-shelf SAM shows limited efficacy in camou-
flaged object detection without task-specific training. Nonetheless,
SAM’s prompt-driven design enables integration of detailed infor-
mation into segmentation masks. Leveraging these complementary
strengths, we employ SAM2 as a refinement module, guided by
preliminary RAG-generated masks.
Post-Processing Optimization. To enhance mask prompts for
SAM2, we evaluate various thresholding strategies on initial RAG
outputs. Experiments indicate a threshold of 0.3 optimally balances
segmentation quality and computational efficiency, effectively sup-
pressing noise while preserving structural details to boost SAM2
refinement. Additionally, we explore point prompts to further im-
prove refinement (see Appendix).
4 Experiments
4.1 Datasets and Evaluation
Following Fan et al. [10], we employ the training datasets for camou-
flaged object detection to validate our method. The training dataset
comprises 4,040 images, with 3,040 images from the COD10K [ 10]
dataset and 1,000 images from the CAMO [ 26] dataset. To eval-
uate our method, we select the widely-used datasets: COD10K,
CAMO, and CHAMELEON [ 37]. To comprehensively evaluate gen-
eralization, we adopt multiple metrics capturing complementary
aspects. Structure-measure ( 𝑆𝑚) [7] assesses structural similarity
via gradient-based signal comparison. Enhanced alignment with
human perception is provided by mean E-measure ( 𝐸𝜉) [8], which
emphasizes fine structures and boundary quality. The weighted
F-measure (𝐹𝑤
𝛽) [34] balances precision and recall through the 𝛽pa-
rameter. Finally, Mean Absolute Error (MAE) measures pixel-wise
prediction error, with lower values indicating better accuracy.

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
4.2 Implementation Details
Implemented in PyTorch with FAISS [ 6], our framework constructs
a feature database from 224×224images, clustering features into
4096 centroids via FAISS K-Means (200 iterations; convergence
observed at about 150). Although we do not rely on high-end
GPUs, we use an Intel i5-11400H CPU and an NVIDIA GeForce
RTX 3050Ti (4GB) on a personal laptop to accelerate feature
extraction. For inference, test images are initially processed at
784×784resolution for mask prediction ( 𝐾=4096), then downsam-
pled to 256×256following the SAM2 pipeline for final segmentation.
4.3 Comparison with State-of-the-Art
To our knowledge, we are the first to propose a training-free RAG-
based method tailored for camouflaged object detection (COD). We
compare against several SAM-based COD approaches, including the
original SAM [ 24,38] and fine-tuned variants [ 1,2,17,18,50,52],
as well as training-based COD methods [ 9,10,29,35]. Table 2 sum-
marizes the quantitative comparison across multiple metrics, where
arrows indicate whether higher ( ↑) or lower (↓) values denote better
performance; best results are bolded. As illustrated in Figure 3, our
method achieves competitive segmentation results, demonstrat-
ing robustness in completeness and challenging scenarios. Our
qualitative results in Appendix further highlight its robustness in
challenging cases such as occlusion ,fine-grained structures ,
andmulti-object camouflage —scenarios that often degrade the
performance of even fine-tuned models. This success is attributed
to the effectiveness of our initial RAG step, which generates high-
quality segmentations capable of capturing occlusions.
4.4 Ablation Studies
We conduct comprehensive ablation studies to validate the effec-
tiveness of our RAG-SEG framework and identify key factors in-
fluencing the final segmentation performance. Unless otherwise
specified, experiments are performed with 𝐾=1024, uti-
lizing cosine similarity to retrieve the top-1 most similar
token for generating the initial prediction on CAMO, with
no post-processing applied. Additional evaluations—including
feature extractor selection, top- 𝑘choice, comparisons with seg-
mentation models such as MobileSAM [ 51] and SAM [ 24], the
effect of CRF [ 25], the impact of point prompts and feature extrac-
tors, visual comparisons, and extension to salient object detection
(SOD) [13, 27, 28, 41, 47, 48]—are detailed in the Appendix.
4.4.1 Impact of Query Image Input Size. We employ DINOv2-small
for feature extraction, where image resolution significantly impacts
both query generation and processing speed in our RAG-SEG frame-
work. Experiments on CAMO dataset reveal optimal performance
at 784-896 resolution, with larger sizes offering diminishing returns
or even degradation, particularly at 1120-1680 due to feature dis-
crepancies with our 224-resolution training database (Table 3). The
qualitative comparison can be seen in Appendix.
4.4.2 Impact of Thresholding Strategies. We investigate the effect
of different thresholding strategies on the masks generated by RAG.
Let the initial probability map be 𝑃initial∈[0,1]𝐻×𝑊. We evaluate:
(1) no thresholding ( 𝑇0, baseline), (2) fixed thresholding at 𝑛×10−1(𝑇𝑛), and (3) normalized thresholding ( 𝑇𝑁), computed as
𝑃norm=𝑃initial−min(𝑃initial)
max(𝑃initial)−min(𝑃initial)+𝜖,where𝜖=10−9.
The experimental results presented in Table 4 demonstrate sev-
eral key findings regarding thresholding strategies. First, moderate
thresholding ( 𝑇3) exhibits superior performance across all evalu-
ation metrics, suggesting an optimal balance in feature selection.
Second, aggressive thresholding ( 𝑇9) leads to a substantial degra-
dation in segmentation performance, particularly evident in the
decreased accuracy metrics. Third, normalized thresholding ( 𝑇𝑁)
demonstrates comparable effectiveness to moderate thresholding
while offering the additional advantage of adaptive scaling capabili-
ties. These findings indicate that moderate thresholding effectively
balances feature preservation and noise reduction, while excessive
thresholding degrades performance.
4.4.3 Impact of Clustering Size. Clustering feature vectors by simi-
larity reduces storage demands and accelerates the RAG process.
The number of clusters 𝐾in KMeans directly impacts both storage
and retrieval efficiency. We evaluate clustering with FAISS under
𝐾∈[512,8192]using strategy 𝑇3. As shown in Table 5, moderate
values (2048-4096) yield the best performance: 𝐾=4096 achieves
optimal𝑀𝐴𝐸 (0.0858),𝐹𝑤
𝛽(0.6776), and 𝐸𝜉(0.8006), while 𝐾=2048
leads in𝑆𝑚(0.7591). To assess retrieval efficiency, we measure the
time to find top-1 similar vectors under different 𝐾. For 784×784
images, the number of tokens per query is 784
142. We adopt the
IP metric and include a warm-up phase to avoid cold-start effects.
Table 5 also reports total latency over 1,000 queries and average
time per query.
5 Conclusion
The RAG-SEG framework demonstrates that effective COD can be
achieved without the need of expensive training or huge computa-
tional resources. By leveraging RAG to generate initial prompts, our
method aids SAM in overcoming the challenges of identifying cam-
ouflaged objects. This approach achieves competitive performance
while significantly reducing environmental impact and computa-
tional requirements. This work lays the foundation for efficient,
eco-friendly computer vision in the era of foundation models. Fu-
ture directions include adapting RAG-SEG to other segmentation
tasks (e.g., salient object detection and semantic segmentation),
refining the RAG pipeline, exploring adaptive RAG architectures,
and developing lightweight SAM alternatives for edge devices.
References
[1]Tianrun Chen, Ankang Lu, Lanyun Zhu, Chaotao Ding, Chunan Yu, Deyi Ji, Zejian
Li, Lingyun Sun, Papa Mao, and Ying Zang. 2024. Sam2-adapter: Evaluating &
adapting segment anything 2 in downstream tasks: Camouflage, shadow, medical
image segmentation, and more. arXiv preprint arXiv:2408.04579 (2024).
[2]Tianrun Chen, Lanyun Zhu, Chaotao Deng, Runlong Cao, Yan Wang, Shangzhan
Zhang, Zejian Li, Lingyun Sun, Ying Zang, and Papa Mao. 2023. Sam-adapter:
Adapting segment anything in underperformed scenes. In Proceedings of the
IEEE/CVF International Conference on Computer Vision . 3367–3375.
[3]Zhongxi Chen, Ke Sun, Xianming Lin, and Rongrong Ji. 2023. CamoDiffusion:
Camouflaged Object Detection via Conditional Diffusion Models. arXiv preprint
arXiv:2305.17932 (2023).
[4]Runmin Cong, Mengyao Sun, Sanyi Zhang, Xiaofei Zhou, Wei Zhang, and Yao
Zhao. 2023. Frequency perception network for camouflaged object detection. In
Proceedings of the 31st ACM International Conference on Multimedia . 1179–1189.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Image Ground Truth RAG-SEG DSAM COMPr ompter SINetv2 SINet Image Ground Truth RAG-SEG DSAM COMPr ompter SINetv2 SINet
Figure 3: Visual comparison of results between our proposed method and other SOTA models.
Table 2: Quantitative performance comparison of COD methods.
Method CHAMELEON [10] CAMO [11] COD10K [9]
𝑆𝛼↑𝐸𝜉↑𝐹𝜔
𝛽↑MAE↓𝑆𝛼↑𝐸𝜉↑𝐹𝜔
𝛽↑MAE↓𝑆𝛼↑𝐸𝜉↑𝐹𝜔
𝛽↑MAE↓
SINet 2020[10] 0.869 0.891 0.740 0.044 0.751 0.771 0.606 0.100 0.771 0.806 0.551 0.051
SINetv2 2021[9] 0.888 0.941 0.816 0.030 0.820 0.882 0.743 0.070 0.815 0.887 0.680 0.037
RankNet 2021[32] 0.846 0.913 0.767 0.045 0.712 0.791 0.583 0.104 0.767 0.861 0.611 0.045
SAM-Adapter2023[2] 0.896 0.919 0.824 0.033 0.847 0.873 0.765 0.070 0.883 0.918 0.801 0.025
SAM2-adapter2024[1] 0.915 0.955 0.889 0.018 0.855 0.909 0.810 0.051 0.899 0.950 0.850 0.018
COMPrompte2024[52] 0.906 0.955 0.857 0.026 0.882 0.942 0.858 0.044 0.889 0.949 0.821 0.023
DSAM 2024[50] - - - - 0.832 0.913 0.794 0.061 0.846 0.921 0.760 0.033
MDSAM 2024[13] - - - - 0.852 0.903 0.834 0.053 0.862 0.921 0.803 0.025
SAM 2023[24] 0.727 0.734 0.639 0.081 0.684 0.687 0.606 0.132 0.783 0.798 0.701 0.050
SAM2 2024[38] 0.359 0.375 0.115 0.357 0.350 0.411 0.079 0.311 0.429 0.505 0.115 0.218
GenSAM 2024[17] 0.774 0.806 0.696 0.073 0.729 0.798 0.669 0.106 0.783 0.843 0.695 0.058
ProMaC 2024[18] 0.833 0.899 0.790 0.044 0.767 0.846 0.725 0.090 0.805 0.876 0.716 0.042
RAG-SEG 0.880 0.915 0.838 0.024 0.831 0.883 0.795 0.064 0.854 0.902 0.783 0.027
(Note) The five rows on the bottom are training-free methods.
786
 1024
 Image 224
 448
 GT Prediction
Figure 4: Impact of input resolution on RAG-based detection
performance. The two rightmost columns show RAG-SEG
results at 784 resolution and ground truth (GT), respectively.Table 3: Performance comparison across different query im-
age resolutions on CAMO.
Resolution 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
112 0.5775 0.3642 0.1375 0.5940
224 0.6952 0.5585 0.1004 0.7297
448 0.7242 0.6163 0.0980 0.7694
784 0.7369 0.6350 0.0934 0.7791
896 0.7378 0.6384 0.0942 0.7802
1008 0.7286 0.6224 0.0969 0.7677
1024 0.7311 0.6272 0.0968 0.7699
1680 0.6974 0.5743 0.1067 0.7301
[5]Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi-
aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image
is Worth 16x16 Words: Transformers for Image Recognition at Scale. (2021).
https://openreview.net/forum?id=YicbFdNTTy

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 4: Impact of thresholding strategies on segmentation
performance on CAMO.
Strategy 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
𝑇0 0.7369 0.6350 0.0934 0.7791
𝑇3 0.7534 0.6643 0.0866 0.7909
𝑇5 0.7295 0.6252 0.0947 0.7682
𝑇7 0.7158 0.6000 0.1003 0.7538
𝑇9 0.6883 0.5557 0.1077 0.7244
𝑇𝑁 0.7371 0.6354 0.0933 0.7789
Table 5: Comparison of clustering and retrieval times, along
with performance metrics, across different cluster sizes. “C.
Time” is the clustering duration measured on CPU, and “R.
Time” is the retrieval latency per 784×784image.
𝐾 C. Time (s) R. Time (s) 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
512 4.12 0.028731 0.7518 0.6671 0.0893 0.7934
1024 14.60 0.034090 0.7534 0.6643 0.0866 0.7909
2048 57.13 0.041342 0.7591 0.6755 0.0863 0.7975
4096 223.00 0.055203 0.7587 0.6776 0.0858 0.8006
8192 465.42 0.088668 0.7548 0.6709 0.0879 0.7944
[6]Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The Faiss library. (2024). arXiv:2401.08281 [cs.LG]
[7]Deng-Ping Fan, Ming-Ming Cheng, Yun Liu, Tao Li, and Ali Borji. 2017. Structure-
measure: A New Way to Evaluate Foreground Maps. In IEEE International Con-
ference on Computer Vision .
[8]Deng-Ping Fan, Cheng Gong, Yang Cao, Bo Ren, Ming-Ming Cheng, and Ali Borji.
2018. Enhanced-alignment Measure for Binary Foreground Map Evaluation.
InProceedings of the Twenty-Seventh International Joint Conference on Artificial
Intelligence . AAAI Press.
[9]Deng-Ping Fan, Ge-Peng Ji, Ming-Ming Cheng, and Ling Shao. 2022. Concealed
Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence
44, 10 (2022), 6024–6042. doi:10.1109/TPAMI.2021.3085766
[10] Deng-Ping Fan, Ge-Peng Ji, Guolei Sun, Ming-Ming Cheng, Jianbing Shen, and
Ling Shao. 2020. Camouflaged Object Detection. In IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) .
[11] Deng-Ping Fan, Ge-Peng Ji, Peng Xu, Ming-Ming Cheng, Christos Sakaridis, and
Luc Van Gool. 2023. Advances in Deep Concealed Scene Understanding. Visual
Intelligence (VI) (2023).
[12] Meirav Galun, E. Sharon, Ronen Basri, and Achi Brandt. 2003. Texture Seg-
mentation by Multiscale Aggregation of Filter Responses and Shape Elements.
International Conference on Computer Vision (Oct 2003).
[13] Shixuan Gao, Pingping Zhang, Tianyu Yan, and Huchuan Lu. 2024. Multi-Scale
and Detail-Enhanced Segment Anything Model for Salient Object Detection.
arXiv preprint arXiv:2408.04326 (2024).
[14] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. A Comprehensive
Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape
and Future Directions. arXiv preprint arXiv:2410.12837 (2024).
[15] Chunming He, Kai Li, Yachao Zhang, Longxiang Tang, Yulun Zhang, Zhenhua
Guo, and Xiu Li. 2023. Camouflaged Object Detection with Feature Decomposition
and Edge Reconstruction. In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) .
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual
learning for image recognition. In Proceedings of the IEEE conference on computer
vision and pattern recognition . 770–778.
[17] Jian Hu, Jiayi Lin, Shaogang Gong, and Weitong Cai. 2024. Relax Image-Specific
Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camou-
flaged Objects. In Proceedings of the AAAI Conference on Artificial Intelligence ,
Vol. 38. 12511–12518.
[18] Jian Hu, Jiayi Lin, Junchi Yan, and Shaogang Gong. 2024. Leveraging Hallucina-
tions to Reduce Manual Prompt Dependency in Promptable Segmentation. arXiv
preprint arXiv:2408.15205 (2024).[19] Xiaobin Hu, Shuo Wang, Xuebin Qin, Hang Dai, Wenqi Ren, Ying Tai, Chengjie
Wang, and Ling Shao. 2022. High-resolution Iterative Feedback Network for
Camouflaged Object Detection. arXiv preprint arXiv:2203.11624 (2022).
[20] Zhou Huang, Hang Dai, Tian-Zhu Xiang, Shuo Wang, Huai-Xin Chen, Jie Qin,
and Huan Xiong. 2023. Feature Shrinkage Pyramid for Camouflaged Object
Detection with Transformers. (2023).
[21] Ge-Peng Ji, Deng-Ping Fan, Yu-Cheng Chou, Dengxin Dai, Alexander Liniger,
and Luc Van Gool. 2023. Deep Gradient Learning for Efficient Camouflaged
Object Detection. Machine Intelligence Research 20 (2023), 92–108. Issue 1.
[22] Ge-Peng Ji, Lei Zhu, Mingchen Zhuge, and Keren Fu. 2022. Fast camouflaged
object detection via edge-based reversible re-calibration network. Pattern Recog-
nition 123 (2022), 108414.
[23] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. arXiv preprint arXiv:2004.04906 (2020).
[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura
Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr
Dollár, and Ross Girshick. 2023. Segment Anything. arXiv:2304.02643 (2023).
[25] Philipp Krähenbühl and Vladlen Koltun. 2011. Efficient inference in fully con-
nected crfs with gaussian edge potentials. Advances in neural information pro-
cessing systems 24 (2011).
[26] Trung-Nghia Le, Tam V. Nguyen, Zhongliang Nie, Minh-Triet Tran, and Akihiro
Sugimoto. 2019. Anabranch network for camouflaged object segmentation. Com-
puter Vision and Image Understanding (Jul 2019), 45–56. doi:10.1016/j.cviu.2019.
04.006
[27] Guanbin Li and Yizhou Yu. 2015. Visual saliency based on multiscale deep features.
InProceedings of the IEEE conference on computer vision and pattern recognition .
5455–5463.
[28] Yin Li, Xiaodi Hou, Christof Koch, James M Rehg, and Alan L Yuille. 2014. The
secrets of salient object segmentation. In Proceedings of the IEEE conference on
computer vision and pattern recognition . 280–287.
[29] Jiaying Lin, Xin Tan, Ke Xu, Lizhuang Ma, and Rynson WH Lau. 2023. Frequency-
aware camouflaged object detection. ACM Transactions on Multimedia Computing,
Communications and Applications 19, 2 (2023), 1–16.
[30] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer us-
ing shifted windows. In Proceedings of the IEEE/CVF international conference on
computer vision . 10012–10022.
[31] Zhengyi Liu, Zhili Zhang, Yacheng Tan, and Wei Wu. 2022. Boosting Camouflaged
Object Detection with Dual-Task Interactive Transformer. (2022), 140–146.
[32] Yunqiu Lyu, Jing Zhang, Yuchao Dai, Aixuan Li, Bowen Liu, Nick Barnes, and
Deng-Ping Fan. 2021. Simultaneously Localize, Segment and Rank the Camou-
flaged Objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) .
[33] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak
Paul, and Benjamin Bossan. 2022. PEFT: State-of-the-art Parameter-Efficient
Fine-Tuning methods. https://github.com/huggingface/peft.
[34] Ran Margolin, Lihi Zelnik-Manor, and Ayellet Tal. 2014. How to Evaluate Fore-
ground Maps. In 2014 IEEE Conference on Computer Vision and Pattern Recognition .
doi:10.1109/cvpr.2014.39
[35] Haiyang Mei, Ge-Peng Ji, Ziqi Wei, Xin Yang, Xiaopeng Wei, and Deng-Ping Fan.
2021. Camouflaged Object Segmentation with Distraction Mining. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) .
[36] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec,
Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-
Nouby, et al .2023. Dinov2: Learning robust visual features without supervision.
[37] Jakub Błaszczyk Tomasz Depta Przemysław Skurowski, Hassan Abdulameer
and Adam Kornacki. 2018. Animal camouflage analysis: Chameleon database.
Unpublished manuscript 2, 6 (2018), 7.
[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali,
Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson,
Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan
Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2:
Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714 (2024).
https://arxiv.org/abs/2408.00714
[39] Yujia Sun, Geng Chen, Tao Zhou, Yi Zhang, and Nian Liu. 2021. Context-aware
Cross-level Fusion Network for Camouflaged Object Detection. In Proceedings of
the 30th International Joint Conference on Artificial Intelligence . 1025–1031.
[40] Kang Wang, Hongbo Bi, Yi Zhang, Cong Zhang, Ziqi Liu, and Shuang Zheng.
2022. D2C-Net: A Dual-Branch, Dual-Guidance and Cross-Refine Network for
Camouflaged Object Detection. IEEE Transactions on Industrial Electronics 69, 5
(2022), 5364–5374. doi:10.1109/TIE.2021.3078379
[41] Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai
Yin, and Xiang Ruan. 2017. Learning to detect salient objects with image-level
supervision. In Proceedings of the IEEE conference on computer vision and pattern
recognition . 136–145.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
[42] Qingwei Wang, Jinyu Yang, Xiaosheng Yu, Fangyi Wang, Peng Chen, and Feng
Zheng. 2023. Depth-Aided Camouflaged Object Detection. In Proceedings of
the 31st ACM International Conference on Multimedia (MM ’23) . Association for
Computing Machinery.
[43] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong
Lu, Ping Luo, and Ling Shao. 2021. Pyramid vision transformer: A versatile back-
bone for dense prediction without convolutions. In Proceedings of the IEEE/CVF
International Conference on Computer Vision . 568–578.
[44] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong
Lu, Ping Luo, and Ling Shao. 2022. Pvtv2: Improved baselines with pyramid
vision transformer. Computational Visual Media 8, 3 (2022), 1–10.
[45] Zongwei Wu, Danda Pani Paudel, Deng-Ping Fan, Jingjing Wang, Shuo Wang,
Cédric Demonceaux, Radu Timofte, and Luc Van Gool. 2023. Source-free depth
for object pop-out. In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) .
[46] Jinnan Yan, Trung-Nghia Le, Khanh-Duy Nguyen, Minh-Triet Tran, Thanh-Toan
Do, and Tam V Nguyen. 2021. Mirrornet: Bio-inspired camouflaged object seg-
mentation. IEEE Access 9 (2021), 43290–43300.
[47] Qiong Yan, Li Xu, Jianping Shi, and Jiaya Jia. 2013. Hierarchical saliency detection.
InProceedings of the IEEE conference on computer vision and pattern recognition .
1155–1162.
[48] Chuan Yang, Lihe Zhang, Huchuan Lu, Xiang Ruan, and Ming-Hsuan Yang. 2013.
Saliency detection via graph-based manifold ranking. In Proceedings of the IEEE
conference on computer vision and pattern recognition . 3166–3173.
[49] Fan Yang, Qiang Zhai, Xin Li, Rui Huang, Ao Luo, Hong Cheng, and Deng-Ping
Fan. 2021. Uncertainty-Guided Transformer Reasoning for Camouflaged Object
Detection. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV) .
doi:10.1109/iccv48922.2021.00411
[50] Zhenni Yu, Xiaoqin Zhang, Li Zhao, Yi Bin, and Guobao Xiao. 2024. Exploring
Deeper! Segment Anything Model with Depth Perception for Camouflaged Object
Detection. In Proceedings of the 32nd ACM International Conference on Multimedia
(ACM MM 2024) . Association for Computing Machinery, 123–132.
[51] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu
Lee, and Choong Seon Hong. 2023. Faster Segment Anything: Towards Light-
weight SAM for Mobile Applications. arXiv preprint arXiv:2306.14289 (2023).
[52] Xiaoqin Zhang, Zhenni Yu, Li Zhao, Deng-Ping Fan, and Guobao Xiao. 2024. COM-
Prompter: Rethink SAM in Camouflaged Object Detection with Multi-Prompt
Network. SCIENCE CHINA Information Sciences (SCIS) 1 (2024), 1–14.

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Motivation
Camouflaged object detection (COD) often requires large-scale
training and substantial computational resources. Although the
Segment Anything Model (SAM) exhibits strong generalization, its
performance on COD is limited by the high visual similarity be-
tween foreground and background. We observe that SAM benefits
greatly from informative prompts, but manual prompt design is nei-
ther scalable nor efficient. To overcome this, we propose RAG-SEG,
which leverages retrieval-augmented generation (RAG) to auto-
matically extract prototypical features from the training set and
use them as prompts for SAM. By fusing RAG’s ability to retrieve
relevant patterns with SAM’s powerful segmentation backbone,
RAG-SEG delivers accurate COD with minimal resource overhead.
Our objective in this paper is to demonstrate that RAG-SEG
achieves high-quality segmentation using minimal compu-
tational resources, thereby highlighting the promise of RAG
in the computer vision community. In future work, we will
incorporate supervised training to further advance performance
toward state-of-the-art levels.
B Experiment Settings
All experiments were carried out on an affordable personal com-
puter with the following specifications:
•Processor: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70
GHz (2.69 GHz)
•RAM: 32.0 GB (31.8 GB available)
•Operating System: Windows 11, 64-bit architecture
•GPU: 4 GB NVIDIA GeForce 3050 Ti
The speed of RAG-SEG is remarkably fast. Although our experi-
ments could be conducted without relying entirely on the CPU, we
utilized an affordable GPU to accelerate the processes of feature
extraction and segmentation refinement in implementation. The
experiments were conducted on a personal device, where multiple
applications, such as the Edge browser with over 200 open tabs,
were running concurrently, utilizing both system RAM and GPU
memory. Therefore, the reported time may not be entirely accurate.
We strictly followed the default settings of the COD bench-
mark without applying any modifications to the dataset or
employing any data augmentation techniques.
C Availability of Some Qualitative Results
In the literature on training-free camouflaged object detection
(COD), full visual examples are often difficult to obtain and quan-
titative results must be taken from the original publications. For
example, two recent test-time adaptation frameworks—ProMaC
and GenSAM—both rely on large foundation models and high-end
GPUs, rendering them impractical on standard affordable laptop
hardware.
•ProMaC employs LLAVA-1.5-13B (GPT-4V), CLIP-CS ViT-
B/16, and SAM, augmented by Stable Diffusion 2 for inpaint-
ing, and performs 4 iterations of iterative prompt refinement
strategy on a single NVIDIA A100 GPU.
•GenSAM integrates BLIP-2 (ViT-g OPT-6.7B), CLIP-CS ViT-
B/16, and SAM, and conducts 6 iterations of iterative prompt
refinement strategy on a single NVIDIA A100 GPU.Both methods iteratively refine the segmentation by using the
previous iteration’s mask as a prompt for the next. By contrast,
our RAG-SEG requires only a single segmentation pass, achieves
superior performance with minimal resources, and does not rely on
test-time adaptation. Moreover, neither ProMaC nor GenSAM cur-
rently release their qualitative segmentation outputs, and—owing
to our hardware constraints—we were unable to reproduce their
results. Upon acceptance, we will make our code and visualizations
publicly available on GitHub.
D Additional Comparison for COD
D.1 Additional Quantitative Comparison on
NC4K
As shown in Table 6, our method requires the least computational
resources yet outperforms MDSAM on the NC4K dataset, delivers
a substantial improvement over DSAM, and achieves performance
metrics comparable to those of COMPrompter. Note that NC4K re-
sults for some methods are not available and are therefore omitted.
Table 6: Quantitative comparison of various methods on the
NC4K dataset.
Method 𝑆𝛼↑𝐸𝜉↑𝐹𝜔
𝛽↑ MAE↓
SINet 2020 0.8080 0.7227 0.8713 0.0576
SINetv2 2021 0.8472 0.7698 0.9027 0.0476
COMPrompter 2024 0.9070 0.8760 0.9550 0.0300
MDSAM 2024 0.8750 0.9210 0.8500 0.0370
DSAM 2024 0.8710 0.8260 0.9320 0.0400
RAG-SEG 0.8824 0.8507 0.9263 0.0308
D.2 Additional Visual Comparison Results
As shown in Figure 5, our results demonstrate superior perfor-
mance in scenarios involving occlusion, fine details, and multiple
objects. This qualitative advantage helps explain why, despite not
achieving state-of-the-art numerical scores compared to fine-tuned
SAM methods, our approach produces more visually accurate and
perceptually compelling segmentations.
E More Results about ablation studies
E.0.1 Impact of Similarity Metric. In our experimental framework,
we conducted evaluations based on a top-1 criterion, wherein the
system selects the most similar feature vector and its associated
mask value. We examined three widely adopted similarity metrics:
Inner Product (IP), Cosine Similarity, and L2 distance. The com-
parative performance of these metrics is shown in Table 7. The
experimental results demonstrate that Inner Product and Cosine
Similarity exhibit comparable performance characteristics, with IP
showing marginally superior results across all evaluation metrics.
E.1 Impact of Feature Extractor
To validate our choice of DINOv2-S as the feature extractor in
RAG-SEG, we conduct an ablation study under identical settings
(𝐾=4096, top-𝑘=1), comparing four backbones: DeiT, CLIP-ViT,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Image Ground Truth RAG-SEG DSAM COMPrompter SINetv2 SINet
Figure 5: Additional visual comparison of the RAG-SEG framework against other SOTA methods.

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 7: Performance comparison of different similarity met-
rics.
Metric 𝑆𝛼↑ 𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
Cosine 0.7534 0.6643 0.0866 0.7909
IP 0.7570 0.6714 0.0856 0.7986
L2 0.7468 0.6549 0.0901 0.7849
Table 8: Performance comparison of different Feature Extrac-
tors (FEs) of RAG-SEG ( 𝐾=4096, topk = 1), where "B" and
"S" refer to the Base and Small versions, respectively, and
ResNet50-32/16 correspond to the features from the final and
penultimate stages.
FE 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
CLIP-VIT-B 0.4601 0.1226 0.1746 0.3567
DeiT-B 0.6402 0.5525 0.1981 0.7011
DeiT-S 0.5904 0.5171 0.2621 0.6496
DINOv2-S 0.8334 0.8004 0.0579 0.8866
ResNet50-16 0.4199 0.0640 0.2057 0.3178
ResNet50-32 0.3441 0.1843 0.3972 0.4516
Swin-Base 0.6324 0.5423 0.1986 0.6970
ResNet-50, and DINOv2-S. The quantitative results are presented
in Table 8 while the visual results are in Figure 6. Among these,
DINOv2-S achieves the highest segmentation accuracy and the
most precise localization of camouflaged targets, significantly out-
performing DeiT, CLIP-ViT, and ResNet-50. This demonstrates that
DINOv2-S provides the most discriminative and robust features for
our RAG-SEG framework.
E.2 Impact of Clustering Method
We chose FAISS KMeans for its optimized memory footprint and
speed on large-scale, high-dimensional vectors. In contrast, Spectral
Clustering, Agglomerative Clustering, and KMedoids all ran out of
memory -first on a standard laptop and then even on a workstation
with an NVIDIA RTX 4090D (24 GB VRAM), 16 vCPUs, and 80,GB
RAM—due to their high eigen-decomposition or quadratic/cubic
complexity. FAISS KMeans, however, completed clustering into
4 096 clusters in just about 433.85 seconds, demonstrating its supe-
rior scalability and efficiency.
E.3 Impact of top- 𝑘Retrieval
We evaluated the effect of varying Top- 𝑘retrievals (𝑘= 1 to 10) on
model performance, maintaining constant parameters of 𝐾=1024,
Cosine metric, and 𝑇3strategy. Table 9 presents our findings. The
results demonstrate that lower k values consistently yield better
performance across all metrics. Top-3 retrieval achieved the highest
𝐹𝑤
𝛽(0.6692) and 𝐸𝜉(0.7938), while Top-1 performed best for 𝑆𝛼
(0.7534) and 𝑀𝐴𝐸 (0.0866). Performance consistently declined for
k values above 3. These results suggest that focusing on the most
similar features ( 𝑘≤3) provides optimal mask prediction while
maintaining efficiency. Figure 7 illustrates the initial masks gener-
ated by RAG for different top- 𝑘settings. As 𝑘increases, the masksexhibit reduced contrast and their pixel values converge toward 0.5,
resulting in progressively blurrier segmentations. This degradation
occurs because larger 𝑘injects more irrelevant features into the
retrieval, amplifying noise and smoothing the mask values.
Table 9: Performance comparison across different top- 𝑘re-
trievals.
Top-k𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
1 0.7534 0.6643 0.0866 0.7909
3 0.7541 0.6692 0.0890 0.7938
5 0.7518 0.6652 0.0903 0.7884
10 0.7444 0.6543 0.0937 0.7817
20 0.7429 0.6521 0.0951 0.7799
50 0.7368 0.6402 0.0970 0.7726
100 0.7270 0.6243 0.1011 0.7609
E.4 Analysis of Post-Processing Strategies
To enhance segmentation quality, we evaluated various post-processing
strategies, with a specific focus on CRF. The experiments examined
CRF application at two key stages: after retrieval-based prediction
(Initial-CRF, I-CRF) and after SAM-based refinement (Final-CRF,
F-CRF). The results are summarized in Table 10. The “CRF-ONLY"
strategy applies CRF directly to the retrieval-based prediction for
the final output, while the “Baseline" strategy relies solely on SAM2
to generate predictions without additional processing. The results
Table 10: Comparison of CRF strategies on the CAMO dataset.
Strategy 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
CRF-ONLY 0.5366 0.3482 0.1383 0.4995
Baseline 0.7534 0.6643 0.0866 0.7909
+I-CRF 0.6526 0.4909 0.1178 0.6757
+F-CRF 0.7536 0.6642 0.0865 0.7902
+I-CRF + F-CRF 0.6529 0.4904 0.1177 0.6730
reveal that CRF post-processing provides limited improvements in
segmentation accuracy for camouflaged object detection. Apply-
ing CRF independently ( CRF-ONLY ) results in significantly lower
performance ( 𝑆𝑚=0.5366 ), while combined strategies, such as
Initial-CRF + Final-CRF, fail to outperform the baseline. These ob-
servations highlight the challenges of camouflaged object detection,
where object boundaries often lack distinct intensity or texture
features. The Final-CRF strategy achieves results comparable to
the baseline, indicating that the benefit of CRF post-processing in
this context is minimal. Consequently, CRF is excluded from our
pipeline.
E.5 Comparison with Other Segmentation
Models
To evaluate the segmentation component of our RAG-SEG frame-
work, we conducted experiments on COD using MobileSAM, SAM,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
GT
 DINOv2-S DeiT B
 DeiT S
 ResNet50-32
 ResNet50-16
 Image
 CLIP-ViT-B
Figure 6: Additional visual comparison of RAG outputs produced by various feature extractors.
Top-1 Top-3 Top-10 Top-50 Top-100
 Image
 GT
Figure 7: Visual comparison about initial mask across different top- 𝑘retrievals.
and SAM2 under the same condition. Despite achieving favorable re-
sults on some images, MobileSAM and SAM showed suboptimal per-
formance on camouflaged objects when compared to SAM2. This dif-
ference in performance can be attributed to SAM2’s stronger abilityto generalize across different camouflage patterns and backgrounds.
This performance gap underscores the superiority of SAM2, which
we chose as the secondary segmentation model in our study. The
results of these models are summarized in Table 11, where SAM2

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
significantly outperforms both MobileSAM and SAM. All models
were evaluated using the same post-processing strategy, specifically
the𝑇3thresholding technique, and no additional techniques were
applied. The input resolution for all models was set to 1024.
Table 11: Segmentation performance of different models.
Model 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
MobileSAM 0.4072 0.0098 0.1854 0.2696
SAM 0.4085 0.0038 0.1821 0.2570
SAM2 0.7534 0.6643 0.0866 0.7909
E.6 Impact of Positive and Negative Point
Prompts
We conducted an ablation study to analyze the effect of varying
positive and negative point thresholds and mask prompts on model
performance. The results are presented in Table 12, where we eval-
uate the segmentation performance using S-measure ( 𝑆𝑚). The ex-
perimental results demonstrate that integrating both positive and
negative point prompts, with appropriately chosen thresholds, sig-
nificantly improves performance compared to using mask prompts
alone.
When using only mask prompts (first row in the table), the model
achieves an 𝑆𝑚score of 0.7534. In comparison, incorporating point
prompts with a positive threshold of 𝑇𝑝=0.95and a negative
threshold of 𝑇𝑛=0.005yields a substantial improvement with an
𝑆𝑚score of 0.8263. Similar performance is achieved with 𝑇𝑝=0.99
and𝑇𝑛=0.05, resulting in an 𝑆𝑚score of 0.8254.
The importance of combining both mask prompts and point
prompts is further highlighted by the significant performance drop
(𝑆𝑚=0.7300 ) observed when the mask prompt is removed while
maintaining point prompts ( 𝑇𝑝=0.95,𝑇𝑛=0.005). Throughout our
experiments, we maintained a consistent mask prompt threshold of
0.3, corresponding to the 𝑇3strategy. These findings demonstrate
that a well-balanced integration of both point prompts and mask
prompts is crucial for optimal segmentation performance.
F Scalability to Large-Scale Datasets
Although standard camouflaged object detection (COD) bench-
marks contain only 4,040 images, related tasks—such as salient
object detection (SOD)—offer large-scale datasets with over 10,000
samples. To evaluate the scalability of RAG-SEG, we apply our
framework to the large-scale DUTS-TR dataset, which contains
10,533 training images, and evaluate performance on the standard
SOD benchmark DUTS-TE (5,019 images). The settings follow those
used in the main paper.
Table 13 reports clustering time, average per-image search time
(for 784×784 resolution), and the corresponding file size for storing
vector–mask pairs under different numbers of clusters 𝐾. As seen,
both clustering and search time scale approximately linearly with
𝐾, reflecting the trade-off between efficiency and granularity of
representation. Moreover, the modest file size underscores the effi-
ciency of our approach and suggests its suitability for deployment
on resource-constrained platforms such as mobile devices.Table 12: Ablation study on positive and negative point
thresholds and mask prompts. The mask prompt threshold
is set to 0.3 (corresponding to the 𝑇3strategy). Best results
are highlighted in bold.
Positive T pNegative T nMask𝑆𝛼↑
- - 0.3 0.7534
0.75 0.005 0.3 0.8251
0.85 0.005 0.3 0.8174
0.95 0.005 0.3 0.8263
0.95 0.005 - 0.7300
0.95 0.01 0.3 0.8216
0.95 0.05 0.3 0.8215
0.95 0.1 0.3 0.8235
0.99 0.005 0.3 0.8240
0.99 0.01 0.3 0.8228
0.99 0.05 0.3 0.8254
0.99 0.1 0.3 0.8202
Table 13: Impact of centroid count 𝐾on clustering time, per-
image 784×784 search latency, and storage footprint.
K Clustering Time (s) Search Time per Image (s) File Size (MB)
1024 37.4264 0.032836 1.62
4096 289.9503 0.061045 6.49
8192 925.0247 0.092390 12.97
10240 1422.8707 0.107845 16.21
16384 2333.6137 0.176441 25.94
32768 4682.4752 0.284777 51.88
65536 9377.0623 0.575370 103.75
Table 14: Quantitative comparison on the DUTS-TE dataset
for varying centroid counts 𝐾.
K𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
1024 0.8828 0.8478 0.0370 0.9213
4096 0.8863 0.8526 0.0354 0.9248
8192 0.8865 0.8520 0.0353 0.9224
10240 0.8906 0.8592 0.0331 0.9275
16384 0.8893 0.8571 0.0338 0.9269
32768 0.8936 0.8639 0.0328 0.9311
65536 0.8974 0.8701 0.0315 0.9331
Table 14 presents the performance on DUTS-TE across different
cluster sizes. Interestingly, increasing 𝐾does not consistently lead
to improved results, although 𝐾=65536 yields the best overall per-
formance across metrics. Figure 8 further supports this observation,
showing that larger values of 𝐾do not significantly enhance seg-
mentation quality. This phenomenon likely stems from the relative
simplicity of the SOD task compared to COD: SAM’s learned priors
already enable effective localization of salient objects, but for SOD,
SAM still requires appropriate prompts to suppress noisy or irrele-
vant masks. As a result, increasing 𝐾beyond a certain threshold
offers diminishing returns.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
GT
 K=32768
 K=8192
 K=4096
 K=1024
 Image
 K=16384
 K=65536
Figure 8: Visual comparison of the RAG-SEG framework for SOD with various 𝐾.
In the SOD field, training-free approaches are basically rare. We
compare RAG-SEG with MDSAM, a SAM-based method fine-tuned
for SOD. MDSAM is trained for 80 epochs on DUTS-TR using a
SAM model fine-tuned on an NVIDIA A100 GPU. Additionally,
to further validate the effectiveness of RAG-SEG, we include a com-
parison with SAM2’s built-in AutomaticMaskGenerator (AMG)
functionality. Since AMG produces multiple masks per image, we
employ ground-truth filtering to retain only the target object
masks.
As indicated in the official SAM2 repository, using mul-
tiple prompt types in AMG significantly increases memory
consumption and inference latency. Therefore, we test differ-
ent prompt counts by uniformly sampling points along the image
borders: 5, 8, 10 and 16 points per side, resulting in total prompts
𝑃=25,64,100, and 256respectively. These variants are denoted as
SAM2-AMG-P-GT , where𝑃indicates the total number of prompts.
Table 15 demonstrates that RAG-SEG substantially outperforms
all AMG-based methods. Notably, our approach utilizing only 20
point and mask prompts surpasses the performance of AMG with
GT filtering even when employing 256 points. These results clearlyTable 15: Comparison of different methods on DUTS-TE
dataset.
Method 𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
MDSAM 2024(Trained) 0.9198 0.8928 0.0245 0.9494
SAM2-AMG-25-GT 0.7250 0.6442 0.1179 0.7634
SAM2-AMG-64-GT 0.7827 0.7290 0.0813 0.8181
SAM2-AMG-100-GT 0.8036 0.7605 0.0683 0.8375
SAM2-AMG-256-GT 0.8247 0.7930 0.0564 0.8589
RAG-SEG (𝐾=65536) 0.8974 0.8701 0.0315 0.9331
illustrate SAM2’s dependency on prompt quality and quan-
tity. When compared to MDSAM, which requires extensive com-
putational resources for training specifically on SOD tasks, our
training-free RAG-SEG approach achieves remarkably compara-
ble performance across multiple evaluation metrics. As shown in
Figure 9, our method achieves better visual results than both the
AMG-based methods and the fine-tuned MDSAM, demonstrating
its effectiveness without any model adaptation.

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
GT AMG-256-GT MDSAM RAG-SEG Image AMG-100-GT AMG-25-GT
Figure 9: Qualitative comparison of RAG-SEG with existing SOD methods. We further include AMG-P-GT results, obtained by
applying SAM2’s AMG module with P point prompts (P = 25, 100, 256).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
In contrast to conventional SAM-based SOD pipelines—which
typically require hundreds of ground-truth points to retrieve the
best-matching mask and consequently suffer from substantial mem-
ory and computational costs—RAG-SEG generates compact prompt
masks directly through retrieval, without any ground-truth su-
pervision. As shown in Table 15, RAG-SEG not only maintains
high segmentation quality but also significantly reduces both mem-
ory footprint and inference time on large-scale datasets exceeding
10,000 images.
Based on the above analysis, we argue that using the AMG
method is highly inefficient and relies on ground truth masks, which
are impractical to obtain in real-world scenarios. Considering our
hardware constraints and the poor performance of AMG, we ex-
clude it from further experiments. Instead, we compare our method
with MDSAM on additional SOD datasets including DUT-OMRON,
ECSSD, HKU-IS, and PASCAL-S, as shown in Table 16. The results
demonstrate that our approach RAG-SEG achieves competitive seg-
mentation performance, comparable to MDSAM, despite requiring
no large-scale training.
G Visualization of Clustering Results
Although t-SNE is a common method for visualizing clustering
outcomes, it is impractical in our case due to the massive number of
feature vectors involved. While dimensionality reduction methods
like PCA can be applied, they still cannot fully represent the cluster
structure in a compact form. Instead, we propose a quantitative
method to indirectly verify clustering quality by analyzing the mask
scores associated with the cluster centers.
Each cluster center is assigned a mask score ranging from 0
to 1, reflecting its semantic saliency. To analyze the clustering
quality, we partition this range into 10 uniform intervals and count
the number of cluster centers within each bin. Figure 10 presents
the score distributions for different values of 𝐾, highlighting how
the number of high-score cluster centers increases with larger 𝐾,
indicating improved coverage of salient regions.
As shown, most cluster centers are concentrated in the lowest
bin[0.0,0.1), particularly for large 𝐾values such as 32,768 or 65,536.
This reflects that a majority of centers represent background or
non-salient regions. However, while the number of high-score cen-
ters (e.g.,[0.9,1.0)) does increase with larger 𝐾, the improvement
in clustering quality for foreground objects is relatively modest.
This suggests that finer granularity provides some enhancement in
the representation of salient areas, but the benefit to downstream
segmentation tasks may not be as significant as expected.
Figure 11 presents the same score distribution as Figure 10. No-
tably, COD contains a much smaller proportion of cluster centers in
the 0.9–1.0 score interval, indicating the scarcity of highly confident
samples. This reflects the intrinsic difficulty of camouflaged object
detection and also implies that camouflaged objects are generally
smaller and less distinguishable than salient ones.
H Joint COD–SOD Segmentation
The proposed RAG-SEG framework requires no task-specific train-
ing and can be applied directly to both camouflaged object detection
(COD) and salient object detection (SOD). This raises the question of
whether these two tasks can be addressed simultaneously within asingle, training-free pipeline. Although most prior work treats COD
and SOD as distinct problems, there is little exploration of unified,
training-free solutions. To investigate this, we set 𝐾COD=4096 and
𝐾SOD=65536 , and merge these into a combined dictionary size
𝐾𝑈=𝐾COD+𝐾SOD. We then evaluate segmentation performance
using each of 𝐾COD,𝐾SOD, and𝐾𝑈. Notably, the unified dictio-
nary shows negligible performance degradation and occasionally
achieves slight improvements compared to the task-specific dictio-
naries (see Table 17). We attribute this robustness to the intrinsic
similarity of COD and SOD as binary segmentation tasks: COD can
be viewed as a more challenging variant of SOD, and SOD as its
simpler counterpart. Moreover, since our retrieval uses only the
single most similar feature (top-1), the presence of features from
the alternate task does not significantly disrupt matching. These
findings demonstrate that RAG-SEG can serve as a truly uni-
fied COD–SOD segmentation method without any additional
training . This outstanding property gives our method exceptional
scalability: by iteratively generating segmentation masks on new
images, we can extract features and progressively enhance the rep-
resentations in our vector database. We will further explore and
quantify this scalability in future work.
I Limitations
I.1 RAG Process
•Fixed embeddings : We rely on a pre-trained embedding
model without fine-tuning, which may limit adaptability to
new COD scenarios. Future work could explore fine-tuning
or self-supervised refinement of the embedding space.
•Basic RAG pipeline : Our design employs a straightforward
retrieval mechanism; advanced variants (e.g., GraphRAG)
remain unexamined and could improve recall and precision.
•Scalability : Experiments were conducted on datasets of
moderate scale. In Appendix E, we further evaluate RAG-
SEG on large-scale salient object detection benchmarks. For
ultra-large collections , the adoption of inverted file (IVF)
indexing or product quantization can preserve computa-
tional efficiency.
I.2 SEG Process
•SAM dependency : The framework performs two rounds of
feature extraction with SAM and DINOv2, increasing com-
putational load. Developing or fine-tuning a lightweight
segmentation head could reduce redundancy and resource
usage.
J Future Work
This work introduces a novel exploration of training-free segmenta-
tion approaches for COD. While our current implementation shows
promising results, we acknowledge certain limitations due to com-
putational constraints and the absence of model training in our
approach. These limitations present several promising directions
for future research, which we outline below:
Extension of the RAG-SEG framework to diverse segmen-
tation tasks. Future research could expand the RAG-SEG paradigm
beyond COD to encompass tasks such as semantic segmentation,
panoptic segmentation, and open-vocabulary segmentation.

First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 16: Quantitative comparison with MDSAM on four standard SOD datasets.
Method ECSSD PASCAL-S
𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
MDSAM 2024 0.9483 0.9463 0.0215 0.9671 0.8820 0.8510 0.0518 0.9167
RAG-SEG 0.9267 0.9275 0.0283 0.9546 0.8784 0.8586 0.0447 0.9269
Method HKU-IS DUT-OMRON
𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑𝑆𝛼↑𝐹𝜔
𝛽↑ MAE↓𝐸𝜉↑
MDSAM 2024 0.9414 0.9348 0.0193 0.9691 0.8783 0.8235 0.0387 0.9099
RAG-SEG 0.9169 0.9175 0.0259 0.9573 0.8043 0.7185 0.0654 0.8343
0-0.10.1-0.2 0.2-0.3 0.3-0.4 0.4-0.5 0.5-0.6 0.6-0.7 0.7-0.8 0.8-0.9 0.9-1.0020,00040,000
Score IntervalNumber of Samples
𝐾=1024𝐾=2048𝐾=4096𝐾=8192
𝐾=16384𝐾=32768𝐾=65536
Figure 10: Distribution of SOD sample scores across 10 score intervals under different numbers of cluster centers..
0-0.10.1-0.2 0.2-0.3 0.3-0.4 0.4-0.5 0.5-0.6 0.6-0.7 0.7-0.8 0.8-0.9 0.9-1.002,0004,0006,000
Score IntervalNumber of Samples
512 centers 1024 centers 2048 centers
4096 centers 8192 centers
Figure 11: Distribution of COD sample scores across 10 score intervals under different numbers of cluster centers.
Improvement of the RAG mechanism. The current imple-
mentation relies on a static retrieval system. Developing a dynamicRAG mechanism with an end-to-end optimized retriever could sig-
nificantly enhance retrieval accuracy and efficiency, particularly
for complex segmentation tasks.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 17: Impact of joint clustering size 𝐾𝑈on COD and SOD performance on CAMO and ECSSD.
Dataset COD (K=4096) SOD (K=65536) 𝑆𝛼↑𝐸𝜉↑𝐹𝜔
𝛽↑ MAE↓
CAMO✓ 0.8305 0.7950 0.0637 0.8834
✓ ✓ 0.8223 0.7886 0.0760 0.8802
ECSSD✓ 0.9267 0.9275 0.0283 0.9546
✓ ✓ 0.9267 0.9280 0.0281 0.9553
Revisiting vector storage architecture. The list-like vector
storage structure used in this work is efficient but simple. Exploring
graph-based vector databases could better handle complex data
relationships, improving scalability and retrieval performance.
Optimization of segmentation models. Leveraging light-
weight network architectures or pre-trained models could enhancesegmentation efficiency and reduce reliance on resource-intensive
models such as SAM. This would accelerate the segmentation
pipeline and enable deployment on edge devices with limited com-
putational capacity.