# RAG-6DPose: Retrieval-Augmented 6D Pose Estimation via Leveraging CAD as Knowledge Base

**Authors**: Kuanning Wang, Yuqian Fu, Tianyu Wang, Yanwei Fu, Longfei Liang, Yu-Gang Jiang, Xiangyang Xue

**Published**: 2025-06-23 17:19:41

**PDF URL**: [http://arxiv.org/pdf/2506.18856v1](http://arxiv.org/pdf/2506.18856v1)

## Abstract
Accurate 6D pose estimation is key for robotic manipulation, enabling precise
object localization for tasks like grasping. We present RAG-6DPose, a
retrieval-augmented approach that leverages 3D CAD models as a knowledge base
by integrating both visual and geometric cues. Our RAG-6DPose roughly contains
three stages: 1) Building a Multi-Modal CAD Knowledge Base by extracting 2D
visual features from multi-view CAD rendered images and also attaching 3D
points; 2) Retrieving relevant CAD features from the knowledge base based on
the current query image via our ReSPC module; and 3) Incorporating retrieved
CAD information to refine pose predictions via retrieval-augmented decoding.
Experimental results on standard benchmarks and real-world robotic tasks
demonstrate the effectiveness and robustness of our approach, particularly in
handling occlusions and novel viewpoints. Supplementary material is available
on our project website: https://sressers.github.io/RAG-6DPose .

## Full Text


<!-- PDF content starts -->

arXiv:2506.18856v1  [cs.CV]  23 Jun 2025RAG-6DPose: Retrieval-Augmented 6D Pose Estimation
via Leveraging CAD as Knowledge Base
Kuanning Wang1, Yuqian Fu2,†, Tianyu Wang1, Yanwei Fu1, Longfei Liang3, Yu-Gang Jiang1, Xiangyang Xue1
Abstract — Accurate 6D pose estimation is key for robotic
manipulation, enabling precise object localization for tasks
like grasping. We present RAG-6DPose , a retrieval-augmented
approach that leverages 3D CAD models as a knowledge base
by integrating both visual and geometric cues. Our RAG-
6DPose roughly contains three stages: 1) Building a Multi-
Modal CAD Knowledge Base by extracting 2D visual features
from multi-view CAD rendered images and also attaching
3D points; 2) Retrieving relevant CAD features from the
knowledge base based on the current query image via our
ReSPC module; and 3) Incorporating retrieved CAD informa-
tion to refine pose predictions via retrieval-augmented decoding.
Experimental results on standard benchmarks and real-world
robotic tasks demonstrate the effectiveness and robustness of
our approach, particularly in handling occlusions and novel
viewpoints. Supplementary material is available on our project
website: https://sressers.github.io/RAG-6DPose .
I. I NTRODUCTION
Monocular 6D pose estimation aims to accurately predict
an object’s 3D position and orientation from a single RGB
image, making it crucial for tasks such as robotic grasping
and interaction. However, achieving robust 6D pose estima-
tion remains challenging due to factors such as occlusions
(including self-occlusions), lack of object textures, and the
domain gap between synthetic and real-world data.
Given one query RGB image and its corresponding 3D
CAD model, many methods [1], [2], [3] obtain the object
pose using the image as input while treating the CAD
model solely as a supervision signal. We believe the valuable
information, e.g., spatial relationships and visual appearance
in CAD, should be actively explored. Some works [4], [5]
with depth input have attempted to take CAD as direct
input to the model, bringing geometric information into the
learning process via techniques like point-based encoding,
etc. Inspired by those works but going beyond them, we
propose to explore the full usage of CAD, not only by
leveraging its geometric properties but also by incorporating
the often-overlooked appearance as input to the model. A
summary of CAD utilization approaches is shown in Fig. 1,
highlighting our early exploration of integrating visual and
geometric information into the model for pose estimation .
Technically, achieving this integration poses two key chal-
lenges: 1) Cross-modal discrepancy: Visual feature extrac-
tion is typically conducted in 2D, while geometric informa-
tion (e.g., coordinates) naturally resides in 3D, leading to
1Fudan University, China.2INSAIT, Sofia University “St. Kliment
Ohridski”, Bulgaria.3NeuhHelium Co.,Ltd., China.
†indicates the corresponding author: Yuqian Fu (yuqian.fu@insait.ai)
(a)modelpredicted pose
Image3D CADℒ!" ##CAD as supervision signals(b)
3D CAD
model
CAD geometry as model input(c)
render
modelCAD geometry and visual information as inputs (Ours)Fig. 1. Comparison of methods: Prior works (a) and (b) use CAD for
solely supervision or geometry input, while ours (c) explores both CAD
geometry and visual appearance for enhanced pose estimation.
a cross-modal gap. 2) Efficient information retrieval: Not
all CAD features are equally relevant—only the regions
corresponding to the query image contribute effectively to
pose estimation. Thus, how to dynamically retrieve the most
useful features for the query image is also crucial.
To address these challenges, we introduce RAG-6DPose ,
a retrieval-augmented pose estimation method that lever-
ages CAD as the knowledge base. RAG-6DPose follows
a structured three-stage pipeline: a) Building a Multi-
Modal Knowledge Base: Given that current models, e.g.,
DINOv2 [6], excel at extracting 2D features compared to
3D, we represent CAD visual features in 2D while preserving
their 3D geometric properties. Specifically, we use DINOv2
to extract visual features from multi-view images rendered
from CAD. These features are then mapped back to 3D
points using depth. Finally, each point integrates visual fea-
tures, coordinates, and color, forming a rich multimodal CAD
knowledge base. b) Retrieving CAD Information: This step
involves retrieving relevant features from the knowledge base
based on the RGB input. To accomplish this, we introduce
the ReSPC module, which performs a retrieval operation that
facilitates the extraction and fusion of both geometric and
visual appearance information, thereby effectively retrieving
the CAD information that best corresponds to the input im-
age. c) Incorporating Retrieved CAD for Pose Estimation:
Finally, we integrate the retrieved CAD features into the final
output through retrieval-augmented decoding.
Extensive results show state-of-the-art (SOTA) perfor-
mance, confirming the importance of integrating CAD’s
visual and geometric information, especially the visual as-
pect. Real-world robotic experiments further demonstrate the
application of our method in the case of object grasping.

II. R ELATED WORK
6D Pose Estimation. Current pose estimation approaches
could be broadly categorized into direct regression-
based methods and 2D-3D correspondence-based methods,
each with distinct core concepts and approaches. Direct
regression-based methods use end-to-end models to directly
regress 6D pose parameters. For instance, PoseCNN [1] uses
a CNN to regress 3D translation and rotation from the input
RGB image. Other notable methods include PoseNet [7],
GDRNet [8], and MRCNet [2]. Typically, these methods
involve a single stage that extracts image features and
predicts 6D poses, though some, like MRCNet, incorporate
an additional stage for refined pose estimation through multi-
scale residual correlation.
Different from the direct regression-based methods, 2D-
3D correspondence-based methods [9], [10], [11] propose
to establish correspondences between 2D image pixels and
3D model points, solving pose with algorithms like PnP-
RANSAC [12]. This type could be further categorized into
sparse and dense correspondence approaches: Sparse meth-
ods, such as PVNet [9], use keypoints to link 2D and 3D
features and are efficient but sensitive to keypoint detection.
Flagship examples of dense methods include Pix2Pose [10],
SurfEmb [11], DPODv2 [13]. Typically, SurfEmb [11] es-
tablishes dense correspondences between 2D pixels and 3D
CAD points in a self-supervised fashion, which achieves
significant improvements over previous supervised 2D-3D
correspondence methods.
According to how CAD is being used, we can also catego-
rize those methods into: a) those use CAD solely for super-
vision, most of the regression-based methods [1], [8] fall in
this group; b) those explicitly incorporate CAD geometry as
input to learn the spatial geometric relationships, examples
include well-designed 2D-3D correspondence methods [11],
3D-3D correspondence [4], [14], etc. Compared with the
prior works, our RAG-6DPose features integrate both CAD
geometry and visual appearance, which aligns more closely
with human intuitive visual perception.
Retrieval Augmented Generation. Retrieval Augmented
Generation (RAG) [15] has been shown to enhance the
performance of large language models (LLMs), particularly
in knowledge-intensive tasks, by mitigating issues such as
opaque reasoning. In this framework, a pre-trained model
serves as parametric memory, while external non-parametric
memory, e.g., a dense vector index of Wikipedia, pro-
vides searchable knowledge, achieving state-of-the-art per-
formance over traditional models. RAG techniques are now
expanding beyond text [16]. For example, RA-CLIP [17]
enhances contrastive language-image training by retriev-
ing relevant image-text pairs, RealRAG [18] and Domain-
RAG [19] leverage retrieved images to improve the text-to-
image generation result. Applications are also extending into
multimodal industrial tasks [20], showing RAG’s versatility
beyond conventional NLP. In this paper, we explore an
efficient retrieval-augmented method in 6D pose estimation,
particularly for leveraging 3D CAD models.III. M ETHODOLOGY
Task Formulation. Given an RGB image Iand a 3D CAD
model Mof a target object, 6D pose estimation requires the
method to estimate the pose pof the object instance in the
image Iusing the CAD model M.
A. Framework Overview
Preliminary. Considering the prior SOTA performance
achieved by SurfEmb [11], a typical 2D-3D correspondence-
based pose estimator, we establish our method following a
similar pipeline with SurfEmb. Specifically, SurfEmb con-
tains an encoder-decoder architecture to extract the feature
from image ( query ,Fq), and a Siren MLP module [21] to
extract geometry i.e., point cloud features from CAD ( key,
Fk). During the training stage, the query image feature Fq
and the key CAD geometry feature Fkare used to perform
the contrastive learning – maximizing the similarity between
corresponding 2D pixels and 3D points. When it comes to
deployment, the Fq,Fkare used to predict the final pose.
We inherit the basic modules and contrastive learning
from SurfEmb, while making several clear improvements: 1)
We construct a rich multimodal CAD knowledge base and
incorporate it into the vanilla encoder-decoder architecture
via the idea of retrieval augmentation; 2) We also combine
the 3D CAD multimodal feature with the CAD points making
the final generated key CAD feature more comprehensive; 3)
Besides, instead of using only CNN for extracting key image
feature, we also introduce the DINOv2 into the encoder.
Overall of Our RAG-6DPose. The illustration of our
method is provided in Fig. 2, including mainly three stages.
1)Building a Multi-Modal CAD Knowledge Base: Given
the CAD model M, this step aims to extract both the visual
and geometry information from M. Typically, the visual
information is obtained in 2D space via extracting features
from multiple rendered images, i.e., offline onboarding,
while the geometry information is conveyed in 3D, i.e.,
the coordinate and color to each point. This step results
in the multimodal CAD knowledge base Fb. 2)Retrieving
CAD Information: This step takes a cropped image as input
based on 2D object detection, using encoders (CNNs and
DINOv2) to generate features Fc
g,Fi. The Fc
gdenotes the
feature generated by one CNN solely, while Fidenotes the
combined feature from CNN and DINOv2. Then our pro-
posed retrieval module, namely ReSPC, takes the 2D image
features Fc
g,Fi, and the CAD knowledge base Fbas input,
generating the retrieved CAD features Fr. 3)Incorporating
Retrieved CAD for Pose Estimation : This step decodes
the concatenated FrandFigenerating the decoded query
feature Fq. The key feature Fkis obtained by incorporating
and embedding the multimodal CAD knowledge feature Fb
and CAD points. With the query image feature Fqand key
CAD feature Fkgenerated, following SurfEmb, contrastive
learning is performed during training, resulting in Lcon,
while the deployment achieves pose prediction.

CNNs
Conv2d
ReLU
UpSample×2Incorporating 
Retrieved CADRetrieving CAD 
Information
C𝐹𝑒𝑎𝑡𝑢𝑟𝑒 𝑃𝑜𝑠𝑖𝑡𝑖𝑜𝑛 𝐶𝑜𝑙𝑜𝑟
DecodersMulti -View Rendering of CAD Model
×4Learnable 𝑭𝒃′
CAD Points
CMLPs
𝑭𝒃
CConcatenation
FrozenDINOv2
Predicted P osePnP-RANSACContrastive 
Learning
𝓛conTraining DeploymentInput Image
DINOv2
Copy
𝐹𝑟𝐹𝑔𝑐𝐹𝑖
𝐹𝑞𝐹𝑘Projection
CAD Knowledge Base Construction𝑁𝑝 points𝑭𝒃
𝐹𝑏
ReSPC  Module
CMLPs
𝐹𝑖
Fig. 2. Method Architecture . The left part provides an overview of “Building a Multi-Modal CAD Knowledge Base.” Particularly, we have two processes,
“Retrieving CAD Information” and “Incorporating Retrieved CAD for Pose Estimation” to leverage the knowledge base Fb. The diagonal dashed line
separates the two processes, highlighting their distinct functions in our method.
B. Offline Multi-Modal Knowledge Base Construction
To efficiently extract visual and geometric information
from the given CAD M, we propose tackling them differ-
ently. Our approach, shown in Fig. 2, rather than directly
extracting features from the CAD model, we first render it
from multiple viewpoints forming multi-view RGBD images
{Ii}m
i=1, we then adopt DINOv2 [6] to extract visual appear-
ance features for each view of that specific object forming
F(i)
v∈RHv×Wv×Dv, where Hv,WvandDvare the height
and width of the feature map, respectively, and Dvis the
feature dimension. Each feature map is then upsampled via
interpolation to match the original image resolution:
F(i)
v→eF(i)
v.
After that, these multi-view image features are mapped
back to the 3D CAD space with the rendered depth map.
This reprojection from 2D to 3D is inspired by recent
methods [22], [23], [24]. To align the rendered views and
3D CAD, we convert the rendered depth map I(i)
dof each
view into a point cloud P(i)
d, with each point corresponding
to a pixel at 2D space. For each sampled point pkin the
CAD model’s 3D point cloud with Nppoints, we identify
the nearest point pdjinP(i)
d:
pdj= arg minpd∈P(i)
d∥pk−pd∥,
allowing the feature vector of each pixel in eF(i)
vto
be assigned directly to the corresponding CAD point. To
ensure comprehensive coverage, we sample from multiple
viewpoints eF(i)
vand average the feature vectors for each
CAD point across all views resulting in a visual feature set
Fp={Fp(pk)}Np
k=1.
To finally integrate the 2D visual and the 3D geometry, for
each point pk, we concatenate its positions, colors, and its
visual feature. Note that the 3D coordinates are represented
by 3D positional encoding. This leads to the multimodal
knowledge base Fb.C. Retrieving CAD Information
Given the multimodal CAD knowledge base Fband the
input image I, the retrieval step is mainly achieved by our
ReSPC module. As in Fig. 3, the ReSPC module is composed
by a Self-attention, PointNet [25] and Cross-attention with
the corresponding details as follow,
Self-Attention. We feed the specific object CAD knowledge
baseFbinto our designed Multi-head Self-Attention module.
It computes relationships among all feature elements, captur-
ing both global context and local details. To further process
the data, we apply a linear layer in the attention module as,
Fsa=SelfAttn (Fb),
to reduce point dimensionality for efficient computation
while preserving key information.
PointNet. We extract global appearance features Fg∈
R1×Dgwith dimension Dgfrom the query image Iusing
ResNet34 and replicate these features to form Fc
g, which we
treat as guidance for PointNet to process the multi-modal
knowledge base effectively. We then concatenate Fc
gwith
Fsaand feed them into our PointNet [25], which consists
of multiple layers of Conv1d, normalization, and activation
functions,
Fpn=PointNet (
Fsa, Fc
g
).
It helps extract geometric information from the knowledge
base, while also capturing local dependencies in the input
feature sequence and ensuring more stable training through
normalization. Through Self-Attention and PointNet, Fpn
is enriched with fine-grained features capturing both the
appearance and geometry of the CAD model.
Cross-Attention. To extract features from RGB images, we
use both DINOv2 [6] and ResNeXt [26]. The output features
from both encoders are then concatenated and used as Fi.
During training, we freeze DINOv2’s parameters to save on
computing resources. Both the construction of the knowledge
base and the extraction of image features utilize DINOv2.

Cross -Attention
Self-AttentionPointNet
C
Cconcatenation𝐹𝑏𝐹𝑔
𝐹𝑔𝑐×𝐷𝑓𝐹𝑝𝑛𝐹𝑟
𝐹𝑖
𝐹𝑝𝑛𝐹𝑖
Attention Weight𝐹𝑟
Reshape
Appearance GeometryCADReshapeFig. 3. ReSPC Module Architecture . This module takes the CAD
knowledge base Fband image features as input and outputs the retrieved
features Fr. The left part illustrates how we use encoders to extract image
features. The middle part demonstrates how attention mechanisms and
PointNet process Fb. The right part details the function of Cross-Attention,
illustrating the core retrieval process from the feature Fpnextracted from
the appearance and geometry of the CAD model.
Therefore, they naturally share consistent feature representa-
tions, facilitating more effective information interaction. To
efficiently retrieve the CAD knowledge base using image
features, we introduce a multihead cross-attention module,
Fr=CrossAttn (Fi, Fpn, Fpn),
where Fiserves as the query, and Fpnacts as the key and
value for the attention mechanism. The process is shown
in Fig. 3. This module measures similarity between Fiand
Fpnthrough attention. Additionally, it enhances information
interaction by incorporating geometric information in Fpn
into the retrieval process, leading to more accurate and robust
matching. With DINOv2’s robust features for both the query
image and CAD knowledge base, Frcaptures the effective
visual appearance and geometric attributes of CAD models
related to the query image through our ReSPC module.
D. Incorporating Retrieved CAD for Pose Estimation
Query Feature Extraction. The retrieved features Fr, ob-
tained through the ReSPC module, and together with the im-
age features Fiare concatenated and fed into both decoders.
As shown in Figure 2, we use two decoders: one to decode
the query features Fq, structured as a U-Net [27], and the
other to decode the mask probabilities.
Key Features Extraction. This module extracts key
features Fkfrom the CAD model Min point cloud form
and multi-modal knowledge base. Given the learnable
features F′
bcopied from the CAD knowledge base (with
Fbfixed), for each coordinate in the pose-specific rendered
coordinates image or each 3D point pjinM, we locate the
nearest point among the corresponding 3D coordinates in F′
b
(based on 3D spatial distance) and assign it the associated
feature vector fjfrom F′
b. Using Siren [21] layers Sg,Sv,
andSi, we then compute key feature as follows:
fk=Si(concat (Sg(pj), Sv(fj)).
The collection of all such fkforms Fk.
Training: Contrastive Learning. The query consists of
the decoded features Fq. During training, the key featuresFk, corresponding to the rendered visible object coordinates
of the ground truth, are treated as positives, while those
uniformly sampled from the surface of the object model are
considered negatives. The mask probabilities by mask de-
coder are used to isolate pixels belonging to the target object
within the input image, allowing these pixels to be sampled in
the rendered visible object coordinates map and the decoded
query features Fq, thereby forming positive query-key pairs.
Finally, Fkis optimized in contrastive learning against the
decoder output Fq. This process improves the consistency
between 2D and 3D features.
Training: Loss Function . In the mask decoder, the final
layer outputs a single channel, and a Sigmoid function is
applied to obtain pixel-wise segmentation results. We utilize
L1 loss Lmfor the segmentation task. For the feature decoder
output, we adopt InfoNCE [28] loss Lconfor contrastive
learning. The point picorresponds to pixel coordinates ci.
Lcon=−logexp(qik+
i)
exp(qik+
i) +PN
j=1,j̸=iexp(qik−
j),
where qi=Fq(ci),k+
i=Fk(pi),k−
j=Fk(pj),j̸=i. The
combined loss function is defined as:
L=Lcon+α· Lm.
where αis a weighting factor to balance the contributions
of segmentation and contrastive learning.
Deployment: Pose Estimation and Refinement. During
inference, we obtain the decoded query features Fqand
generate key features Fkfrom the CAD. For pose estimation,
query features are combined with masks and compared to key
features to form a similarity matrix. 2D-3D correspondences
are sampled based on similarity, and pose is estimated
using PnP-RANSAC [29], [12]. The training loss score helps
evaluate and select the best pose hypothesis, which is then
refined by maximizing the correspondence score to generate
the final prediction.
IV. E XPERIMENTS
A. Experimental Setup
Datasets. We evaluate the effectiveness of our method across
a wide range of datasets, including the challenging LM-
O [30], YCB-V [1], IC-BIN [31], HB [32] and TUD-L [33]
datasets. These datasets feature multiple densely cluttered
rigid objects with limited texture and significant occlusion,
providing a comprehensive and rigorous evaluation that
sufficiently validates the competitiveness of our approach.
Model Details. We use the pre-trained DINOv2-Base and
ResNeXt-101 32x8d as the backbone models. Our method
is tested with both RGB-only input and RGB-D input, with
the CAD model of the target object available in both cases.
For the detection of the 2D image, the default detection
method of BOP [34] Challenge 2023 is used. In the Building
a Multi-Modal CAD Knowledge Base stage, we follow
the perspective selection strategy of CNOS [35], choosing
perspectives for each object and rendering corresponding
images for each view. During the encoder-decoder stage, the

input image is first cropped around the object, resized to
224×224, and normalized.
Training Details. The RAG-6DPose model is trained to
convergence on the LM-O, TUD-L, IC-BIN, HB, and YCB-
V datasets. Synthetic images are used for LM-O, IC-BIN,
HB, and YCB-V , while both synthetic and real images are
used for TUD-L to evaluate the model’s performance in
mixed-domain scenarios. We use the Adam optimizer with
a fixed learning rate of 3e-5. Experiments are performed on
RTX A6000 GPUs and RTX 3090 GPUs.
Evaluation. For evaluation, we used the Average Recall
metric, which is commonly used in the BOP Challenge. The
Average Recall is the average of three indicators based on
different error functions: VSD (Visible Surface Discrepancy),
MSSD (Maximum Symmetry-aware Surface Distance), and
MSPD (Maximum Symmetry-aware Projection Distance), as
mentioned in work [36].
B. Comparison Results
To comprehensively show the advantage of our proposed
method, we compare our RAG-6DPose model with various
prior state-of-the-art methods. Note that for our model, we
only train one model for all objects on one dataset, while
some competitors, e.g., DPODv2 [13] and NCF [37] require
training several different models for different objects.
TABLE I
COMPARISON WITH STATE -OF-THE-ART RGB BASED METHODS ON
BOP BENCHMARKS .WE REPORT AVERAGE RECALL IN %ONLM-O,
IC-BIN, TUD-L, HB AND YCB-V DATASETS . P.E. MEANS THE
NUMBER OF POSE ESTIMATORS FOR AN N-OBJECTS DATASET .
Method LM-O IC-BIN TUD-L YCB-V HB P.E.
CDPNv2 [3] 62.4 47.3 77.2 53.2 72.2 N
DPODv2 [13] 58.4 - - - 72.5 N
NCF [37] 63.2 - - 67.3 - N
GDRNet [8] 67.2 - - - - N
SurfEmb [11] 65.6 58.5 80.5 65.3 79.3 N
CosyPose [38] 63.3 58.3 82.3 57.4 65.6 1
SO-Pose [39] 61.3 - - 66.4 - 1
CRT-6D [40] 66.0 53.7 78.9 - 60.3 1
PFA [41] 67.4 - - 61.5 71.2 1
YOLO6DPose [42] 62.9 - - - - 1
MRCNet [2] 68.5 - - 68.1 - 1
RAG-6DPose 70.0 60.1 83.3 68.6 85.3 1
Results on RGB based Pose Estimation. The main results
are shown in Tab. I. Overall, our method outperforms all
the competitors across five datasets, with especially strong
results on datasets with heavy occlusion like LM-O. Unlike
approaches that require separate models for each object,
RAG-6DPose solely needs one model, sharing parameters
across all objects while retaining a small set of object-specific
parameters. Despite this setup, it still outperforms others.
Compared to SurfEmb [11], we achieve a 3.6% improve-
ment in average recall across 5 datasets, demonstrating a
significant and consistent performance gain. Our method also
outperforms the recent competitive state-of-the-art method
MRCNet [2], which uses the Render-and-Compare strategy,
on both LM-O and YCB-V datasets, with a clear margin
of 1.5% on LM-O. This significant improvement validatesthe effective integration of the multi-modal CAD knowledge
base in our model and highlights the effectiveness of the
novel modules we introduced.
Results on RGB-D based Pose Estimation. We adopt the
ICP [43] algorithm to refine pose estimation. With RGB-
D input (see Tab. II), our RAG-6DPose shows significant
improvements over previous results (Tab. I). We also outper-
form other RGB-based methods that use depth refinement.
TABLE II
COMPARISON WITH METHODS ON BOP BENCHMARKS BASED ON
RGBD AND DEPTH REFINEMENT .WE REPORT AVERAGE RECALL IN %
ONLM-O, IC-BIN AND TUD-L DATASETS .
Method Domain LM-O IC-BIN TUD-L
Pix2Pose [10] RGB-D 58.8 39.0 82.0
CosyPose [38] RGB-D 71.4 64.7 93.9
SurfEmb [11] RGB-D 75.8 65.6 93.3
RAG-6DPose RGB-D 76.8 68.7 93.9
Comparison with 2D-3D Correspondence Methods. As
shown in Tables I and II, our method outperforms 2D-3D
correspondence-based approaches, such as DPOPv2 [13] and
SurfEmb [11]. This improvement is due to our approach’s
ability to retrieve appearance features in CAD knowledge
features. By leveraging this retrieval process, our approach
effectively enhances the precision of pose estimation, espe-
cially in challenging scenarios with complex occlusions on
the LM-O dataset.
Results on RGB-based Methods For Each Object. To
further investigate the performance of our method, we also
analyzed the results for each object in the LM-O dataset.
Specifically, we use eas the pose estimation error and
set the pose error thresholds θeto 5 and 10, which are
the smallest two thresholds for fine-grained evaluation. A
pose is considered correct if the calculated pose error e
satisfies e<θe. We use ARe
MSPD to denote recall based on
MSPD as a pose-error function. The final recall rate for each
object is reported in Tab. III. Comparing our RAG-6DPose
with CDPNv2, SurfEmb, and MRCNet, we observe that our
method achieves the best results on most objects.
TABLE III
ARe
MSPDCOMPARISON RESULTS WITH RGB METHODS ON BOP
BENCHMARKS USING POSE ERROR THRESHOLD θe=5ANDθe=10. “ DRI.”
MEANS THE “DRILLER ”CLASS , “E.B.”AND “H.P.”DENOTE “EGGBOX ”
AND “HELICOPTERS ”CLASS RESPECTIVELY .
Method θeape can cat dri. duck e.b. glue h.p.
CDPNv2 5 37 44 46 22 30 7 19 22
SurfEmb 5 52 46 50 28 36 27 23 19
MRCNet 5 44 56 53 40 39 22 25 22
RAG-6DPose 5 51 52 53 41 39 27 25 26
CDPNv2 10 70 85 68 66 77 42 53 83
SurfEmb 10 77 92 78 82 78 70 74 89
MRCNet 10 77 92 81 82 82 64 74 90
RAG-6DPose 10 85 95 81 90 82 59 78 87
C. Qualitative Results
As shown in Fig. 4, we tested SurfEmb and our RAG-
6DPose on challenging scenes with heavy occlusion from the

(a)
(b)Ground Truth RAG -6DPose Surfemb
Fig. 4. Qualitative Comparison . Parts (a) and (b) illustrate different
scenes from the LM-O and YCB-V datasets. Each part displays three images
(from left to right): the original RGB image (with the object’s ground
truth pose annotated), the SurfEmb result, and the RAG-6DPose result.
This comparison underscores the effectiveness of RAG-6DPose in handling
highly occluded scenes.
LM-O and YCB-V datasets. In Fig. 4(a), SurfEmb struggles
with complex occlusions involving both the foreground and
the background, leading to poor pose estimation. In Fig. 4(b),
SurfEmb shows some improvement when the object can
be inferred from visible edges, but it still faces difficulties
with severe occlusion. By contrast, RAG-6DPose delivers
more accurate pose estimation and maintains robustness in
complex environments.
D. Ablation Studies
As shown in Tab. IV, we conducted ablation studies on
LM-O to evaluate the effectiveness of our proposed modules.
We categorize the experiments into full-scale and reduced-
scale ablations. In the full-scale ablations, the model is
trained on the entire dataset, while in the reduced-scale
ablations, the model is trained on a subset comprising one-
tenth of the full dataset until the error stabilizes.
TABLE IV
ABLATION EXPERIMENTS . USING RGB IMAGES FROM THE LM-O
DATASET AS INPUT . “AR” MEANS AVERAGE RECALL .
Method AR↑ARV SD ARMSSD ARMSPD
Full-scale Ablation
RAG-6DPose 70.0 53.4 69.1 87.3
−3D CAD Features 66.5 50.1 64.8 84.7
−DINOMLP 68.8 52.5 67.7 86.4
C.A.Fusion →ConvFusion 68.6 52.1 67.7 86.0
C.A.Fusion →Avg 67.7 51.4 66.5 85.2
ResNet34 →Avg 68.4 52.5 67.9 86.6
Reduced-scale Ablation
RAG-6DPose 59.3 43.2 55.5 79.3
PointNet →MLP 55.6 40.1 50.8 75.8
−ResNeXt 47.4 29.9 39.0 73.2
−Fp 57.8 42.0 53.6 77.9
Settings. For full-scale ablations, we conduct 5 experiments,
as shown in Tab. IV. In the “ −3D CAD Features” (row 2),
we removed the CAD knowledge base Fband the ReSPC
module. In the “ −DINOMLP” (row 3), we removed the
learnable copy of the CAD knowledge base F′
b. This leaves
only the 3D coordinates processed by MLPs for key features,
KINOV A GEN 2RealSense D435
CupCan
EggboxBrickPen
Ape
Setup Level 2 Tasks
Ⅰ
Ⅱ
Fig. 5. Robotic Experimental Setup. Left: a third-person view of the
setup. A first-person camera captures RGB image observations. We show
different 3D printed objects from the LM-O and YCB-V datasets and a pen,
all of which are used in our experiments. Right : we show “Level 2” (i.e.,
more challenging) tasks in our experiments. We present the “Pick and Place
Pen” task (I) and the “Move Ape” task (II). The bottom left corner shows
the predicted object poses obtained from the observed image input.
similar to SurfEmb, relying solely on geometric information.
In the “C.A. Fusion →ConvFusion” (row 4), we replaced the
cross-attention with a convolution-based module, as in [44],
which simply fuses concatenated inputs without retrieval. In
the “C.A. Fusion →Avg”(row 5), we replaced the cross-
attention with a simple fusion method that computes the
average of FiandFpn. In the “ResNet34 →Avg”(row 6), we
replaced the global feature extracted by ResNet34 with the
average of the ResNeXt feature. For reduced-scale ablations,
we conduct 3 experiments. In the “PointNet →MLP,” we
replaced PointNet with an MLP. In the “ −ResNeXt”, we
removed the ResNeXt. Finally, in the “ −Fp”, we removed
the visual features Fpin knowledge base Fb.
Results. Tab. IV presents our results. The ablation exper-
iments validate the effectiveness of our designs. Notably,
the “−3D CAD Features” shows a significant drop in
the average recall, emphasizing the importance of retrieving
detailed CAD visual and geometric information. Both “-
DINOMLP” and “ −Fp” demonstrate that integrating CAD
visual appearance features enhances the model’s ability to
learn robust 2D-3D correspondences. Furthermore, “C.A.
Fusion →ConvFusion”, “C.A. Fusion →Avg”, “PointNet
→MLP”, and “ResNet34 →Avg” show the effectiveness
of the individual components within our ReSPC module.
Finally, “ −ResNeXt” demonstrates that incorporating the
features retrieved using a frozen-parameter DINOv2 into the
decoding process produces favorable results, which suggests
that our retrieval method is effective.
E. Robotic Experiment
Setup. Our experimental setup is illustrated in Fig. 5. Our
model is deployed on an NVIDIA RTX 6000 Ada Gener-
ation GPU to estimate the object’s pose from RGB images
captured by a RealSense D435 camera. We use a Kinova
Gen2 robot arm equipped with three fingers. The experiments
utilize objects from the LM-O and YCB-V datasets, inspired
by home service and industrial applications.

TABLE V
ROBOTIC EXPERIMENT RESULTS . W E REPORT THE SUCCESS RATE OF
RAG-6DP OSE OVER 10TRIALS FOR EACH OF 5TASKS .
Level 1 Level 2
Grasp Brick Grasp Cup Grasp Eggbox Pick and Place Pen Move Ape
10/10 10/10 10/10 9/10 9/10
Task Description. We categorize our tasks into two difficulty
levels. Level 1 tasks involve directly grasping specific objects
on the table—namely, “Grasp Brick”, “Grasp Cup,” and
“Grasp Eggbox.” Level 2 tasks are more challenging due
to occlusions and the high precision required, and they
necessitate the sequential estimation of two objects’ poses
to complete the task, as illustrated in Fig. 5. The first Level
2 task, named “Pick and Place Pen,” involves grasping a
pen (not included in the datasets) inserted into a hole in
the center of a brick and placing it into an adjacent cup,
which requires accurate pose estimation for both the brick
and the cup to guide the pen’s pick-and-place actions. The
second task, called “Move Ape,” entails grasping an ape that
is partially occluded by a watering can and moving it in front
of the can’s spout, thus requiring precise pose estimation for
both the ape and the watering can.
We use predefined grasp policies for the object model and,
by combining them with the estimated object pose, transform
these grasp poses into the robot frame for motion planning.
Our evaluation metric focuses on task success rate. For Level
1 tasks, success is defined as a successful grasp without
dropping the object. For Level 2 tasks, success is determined
by whether the task is completed perfectly according to the
specified requirements. We repeat the experiment 10 times
for each task. In repeated experiments, we randomize object
placements and other aspects of the scene setup to increase
the diversity of the settings.
Results. The results for Level 1 and Level 2 tasks are
shown in Tab. V. Our approach achieves highly accurate pose
estimation, particularly obtaining precise grasping poses for
Level 2 tasks. The results demonstrate the effectiveness of
our approach in real-world deployments.
V. C ONCLUSION
In this paper, we introduced a novel RAG-inspired ap-
proach, RAG-6DPose , for 6D pose estimation from input
RGB images. Our method aims to leverage the multimodal
information in 3D CAD models, particularly the visual
appearances, which have been less explored in prior works.
Technically, our approach consists of three key steps: build-
ing a multimodal CAD knowledge base, retrieving relevant
CAD information, and incorporating the retrieved informa-
tion into pose estimation. Extensive experiments, conducted
with both RGB and RGB-D inputs, along with quantitative
evaluations, ablation studies, visualizations, and detailed
analysis, demonstrate the effectiveness and robustness of our
method in tackling the challenging task of 6D pose esti-
mation. Furthermore, real-world robotic experiments validate
the successful application of our method for object grasping,
further highlighting its practical utility.REFERENCES
[1] Y . Xiang, T. Schmidt, V . Narayanan, and D. Fox, “Posecnn: A
convolutional neural network for 6d object pose estimation in cluttered
scenes,” ArXiv , 2017.
[2] Y . Li, Y . Mao, R. Bala, and S. Hadap, “Mrc-net: 6-dof pose estimation
with multiscale residual correlation,” ArXiv , 2024.
[3] Z. Li, G. Wang, and X. Ji, “Cdpn: Coordinates-based disentangled pose
network for real-time rgb-based 6-dof object pose estimation,” 2019
IEEE/CVF International Conference on Computer Vision (ICCV) ,
2019.
[4] H. Li, J. Lin, and K. Jia, “Dcl-net: Deep correspondence learning
network for 6d pose estimation,” in European Conference on Computer
Vision , 2022.
[5] H. Jiang, Z. Dang, S. Gu, J. Xie, M. Salzmann, and J. Yang,
“Center-based decoupled point cloud registration for 6d object pose
estimation,” 2023 IEEE/CVF International Conference on Computer
Vision (ICCV) , 2023.
[6] M. Oquab, T. Darcet, T. Moutakanni, H. Q. V o, M. Szafraniec,
V . Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. As-
sran, N. Ballas, W. Galuba, R. Howes, P.-Y . B. Huang, S.-W. Li,
I. Misra, M. G. Rabbat, V . Sharma, G. Synnaeve, H. Xu, H. J ´egou,
J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning
robust visual features without supervision,” ArXiv , 2023.
[7] A. Kendall, M. K. Grimes, and R. Cipolla, “Posenet: A convolutional
network for real-time 6-dof camera relocalization,” 2015 IEEE Inter-
national Conference on Computer Vision (ICCV) , 2015.
[8] G. Wang, F. Manhardt, F. Tombari, and X. Ji, “Gdr-net: Geometry-
guided direct regression network for monocular 6d object pose esti-
mation,” 2021 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , 2021.
[9] S. Peng, Y . Liu, Q.-X. Huang, H. Bao, and X. Zhou, “Pvnet: Pixel-wise
voting network for 6dof pose estimation,” 2019 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR) , 2018.
[10] K. Park, T. Patten, and M. Vincze, “Pix2pose: Pixel-wise coordinate
regression of objects for 6d pose estimation,” 2019 IEEE/CVF Inter-
national Conference on Computer Vision (ICCV) , 2019.
[11] R. L. Haugaard and A. G. Buch, “Surfemb: Dense and continuous
correspondence distributions for object pose estimation with learnt
surface embeddings,” 2022 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , 2021.
[12] M. A. Fischler and R. C. Bolles, “Random sample consensus: a
paradigm for model fitting with applications to image analysis and
automated cartography,” Communications of the ACM , 1981.
[13] I. S. Shugurov, S. Zakharov, and S. Ilic, “Dpodv2: Dense
correspondence-based 6 dof pose estimation,” IEEE Transactions on
Pattern Analysis and Machine Intelligence , 2021.
[14] H. Gan, L. Wang, Y . Su, W. Ruan, and X. Jiao, “Prior-information-
guided corresponding point regression network for 6d pose estima-
tion,” Comput. Graph. , 2024.
[15] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Kuttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
nlp tasks,” ArXiv , 2020.
[16] X. Zheng, Z. Weng, Y . Lyu, L. Jiang, H. Xue, B. Ren, D. Paudel,
N. Sebe, L. Van Gool, and X. Hu, “Retrieval augmented generation and
understanding in vision: A survey and new outlook,” arXiv preprint ,
2025.
[17] C.-W. Xie, S. Sun, X. Xiong, Y . Zheng, D. Zhao, and J. Zhou, “Ra-clip:
Retrieval augmented contrastive language-image pre-training,” 2023
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) , 2023.
[18] Y . Lyu, X. Zheng, L. Jiang, Y . Yan, X. Zou, H. Zhou, L. Zhang, and
X. Hu, “Realrag: Retrieval-augmented realistic image generation via
self-reflective contrastive learning,” arXiv preprint , 2025.
[19] Y . Li, X. Qiu, Y . Fu, J. Chen, T. Qian, X. Zheng, D. P. Paudel,
Y . Fu, X. Huang, L. Van Gool et al. , “Domain-rag: Retrieval-guided
compositional image generation for cross-domain few-shot object
detection,” arXiv preprint , 2025.
[20] M. Riedler and S. Langer, “Beyond text: Optimizing rag with multi-
modal inputs for industrial applications,” 2024.
[21] V . Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and
G. Wetzstein, “Implicit neural representations with periodic activation
functions,” ArXiv , 2020.

[22] Y . Wang, Z. Li, M. Zhang, K. R. Driggs-Campbell, J. Wu, F.-F.
Li, and Y . Li, “D3fields: Dynamic 3d descriptor fields for zero-shot
generalizable robotic manipulation,” ArXiv , vol. abs/2309.16118, 2023.
[23] A. Caraffa, D. Boscaini, A. Hamza, and F. Poiesi, “Freeze: Training-
free zero-shot 6d pose estimation with geometric and vision foundation
models,” 2023.
[24] W. Huang, C. Wang, R. Zhang, Y . Li, J. Wu, and L. Fei-Fei, “V oxposer:
Composable 3d value maps for robotic manipulation with language
models,” ArXiv , 2023.
[25] C. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning
on point sets for 3d classification and segmentation,” 2017 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) ,
2016.
[26] S. Xie, R. B. Girshick, P. Doll ´ar, Z. Tu, and K. He, “Aggregated resid-
ual transformations for deep neural networks,” 2017 IEEE Conference
on Computer Vision and Pattern Recognition (CVPR) , 2016.
[27] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional
networks for biomedical image segmentation,” ArXiv , 2015.
[28] A. van den Oord, Y . Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” ArXiv , 2018.
[29] V . Lepetit, F. Moreno-Noguer, and P. Fua, “Ep n p: An accurate o
(n) solution to the p n p problem,” International journal of computer
vision , 2009.
[30] E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, and
C. Rother, “Learning 6d object pose estimation using 3d object
coordinates,” in European Conference on Computer Vision , 2014.
[31] A. Doumanoglou, R. Kouskouridas, S. Malassiotis, and T.-K. Kim,
“Recovering 6d object pose and predicting next-best-view in the
crowd,” 2016 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , 2015.
[32] R. Kaskman, S. Zakharov, I. S. Shugurov, and S. Ilic, “Homebreweddb:
Rgb-d dataset for 6d pose estimation of 3d objects,” 2019 IEEE/CVF
International Conference on Computer Vision Workshop (ICCVW) ,
2019.
[33] T. Hodan, F. Michel, E. Brachmann, W. Kehl, A. G. Buch, D. Kraft,
B. Drost, J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt,
F. Tombari, T.-K. Kim, J. Matas, and C. Rother, “Bop: Benchmark
for 6d object pose estimation,” ArXiv , 2018.
[34] T. Hodan, M. Sundermeyer, Y . Labbe, V . N. Nguyen, G. Wang,
E. Brachmann, B. Drost, V . Lepetit, C. Rother, and J. Matas, “Bop
challenge 2023 on detection, segmentation and pose estimation of seen
and unseen rigid objects,” 2024.
[35] V . N. Nguyen, T. Hodan, G. Ponimatkin, T. Groueix, and V . Lepetit,
“Cnos: A strong baseline for cad-based novel object segmentation,”
2023 IEEE/CVF International Conference on Computer Vision Work-
shops (ICCVW) , 2023.
[36] T. Hodan, M. Sundermeyer, B. Drost, Y . Labb ´e, E. Brachmann,
F. Michel, C. Rother, and J. Matas, “Bop challenge 2020 on 6d object
localization,” ArXiv , 2020.
[37] L. Huang, T. Hodan, L. Ma, L. Zhang, L. Tran, C. D. Twigg, P.-C. Wu,
J. Yuan, C. Keskin, and R. Wang, “Neural correspondence field for
object pose estimation,” in European Conference on Computer Vision ,
2022.
[38] Y . Labb’e, J. Carpentier, M. Aubry, and J. Sivic, “Cosypose: Consistent
multi-view multi-object 6d pose estimation,” in European Conference
on Computer Vision , 2020.
[39] Y . Di, F. Manhardt, G. Wang, X. Ji, N. Navab, and F. Tombari, “So-
pose: Exploiting self-occlusion for direct 6d pose estimation,” 2021
IEEE/CVF International Conference on Computer Vision (ICCV) ,
2021.
[40] P. Castro and T.-K. Kim, “Crt-6d: Fast 6d object pose estimation with
cascaded refinement transformers,” 2023 IEEE/CVF Winter Confer-
ence on Applications of Computer Vision (WACV) , 2022.
[41] Y . Hu, P. Fua, and M. Salzmann, “Perspective flow aggregation for
data-limited 6d object pose estimation,” in European Conference on
Computer Vision , 2022.
[42] D. Maji, S. Nagori, M. Mathew, and D. Poddar, “Yolo-6d-pose:
Enhancing yolo for single-stage monocular multi-object 6d pose
estimation,” 2024 International Conference on 3D Vision (3DV) , 2024.
[43] S. Rusinkiewicz and M. Levoy, “Efficient variants of the icp algo-
rithm,” Proceedings Third International Conference on 3-D Digital
Imaging and Modeling , 2001.
[44] S. Ren, Y . Zeng, J. Hou, and X. Chen, “Corri2p: Deep image-to-point
cloud registration via dense correspondence,” IEEE Transactions on
Circuits and Systems for Video Technology , 2022.