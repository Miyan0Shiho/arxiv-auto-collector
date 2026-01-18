# RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation

**Authors**: Yue Chang, Rufeng Chen, Zhaofan Zhang, Yi Chen, Sihong Xie

**Published**: 2026-01-15 08:15:01

**PDF URL**: [https://arxiv.org/pdf/2601.10168v1](https://arxiv.org/pdf/2601.10168v1)

## Abstract
Open-vocabulary 3D Scene Graph (3DSG) generation can enhance various downstream tasks in robotics, such as manipulation and navigation, by leveraging structured semantic representations. A 3DSG is constructed from multiple images of a scene, where objects are represented as nodes and relationships as edges. However, existing works for open-vocabulary 3DSG generation suffer from both low object-level recognition accuracy and speed, mainly due to constrained viewpoints, occlusions, and redundant surface density. To address these challenges, we propose RAG-3DSG to mitigate aggregation noise through re-shot guided uncertainty estimation and support object-level Retrieval-Augmented Generation (RAG) via reliable low-uncertainty objects. Furthermore, we propose a dynamic downsample-mapping strategy to accelerate cross-image object aggregation with adaptive granularity. Experiments on Replica dataset demonstrate that RAG-3DSG significantly improves node captioning accuracy in 3DSG generation while reducing the mapping time by two-thirds compared to the vanilla version.

## Full Text


<!-- PDF content starts -->

RAG-3DSG: ENHANCING3D SCENEGRAPHS WITH
RE-SHOTGUIDEDRETRIEVAL-AUGMENTEDGENER-
ATION
Yue Chang, Rufeng Chen, Zhaofan Zhang, Yi Chen, Sihong Xie
AI Thrust, HKUST(GZ)
ABSTRACT
Open-vocabulary 3D Scene Graph (3DSG) generation can enhance various down-
stream tasks in robotics, such as manipulation and navigation, by leveraging
structured semantic representations. A 3DSG is constructed from multiple im-
ages of a scene, where objects are represented as nodes and relationships as
edges. However, existing works for open-vocabulary 3DSG generation suffer
from both low object-level recognition accuracy and speed, mainly due to con-
strained viewpoints, occlusions, and redundant surface density. To address these
challenges, we proposeRAG-3DSGto mitigate aggregation noise through re-
shot guided uncertainty estimation and support object-level Retrieval-Augmented
Generation (RAG) via reliable low-uncertainty objects. Furthermore, we propose
a dynamic downsample-mapping strategy to accelerate cross-image object aggre-
gation with adaptive granularity. Experiments on Replica dataset demonstrate that
RAG-3DSG significantly improves node captioning accuracy in 3DSG generation
while reducing the mapping time by two-thirds compared to the vanilla version.
1 INTRODUCTION
Compact and expressive representation of complex and semantic-rich 3D scenes has long been a
fundamental challenge in robotics, with direct impact on downstream tasks such as robot manip-
ulation (Shridhar et al., 2022; Rashid et al., 2023) and navigation (Gadre et al., 2022; Shah et al.,
2023). A promising solution is 3D scene graphs (3DSGs) (Armeni et al., 2019; Gay et al., 2018),
which encode a scene into a graph where nodes denote objects and edges capture their pairwise
relationships. Early efforts focus on building 3DSGs by detecting objects and relationships from a
closed vocabulary (Hughes et al., 2022; Rosinol et al., 2021; Wu et al., 2021). While these methods
perform well and are efficient in fixed environments, their reliance on a closed vocabulary restricts
their generalization to unseen scenes in the real-world. To mitigate this limitation, recent approaches
(Gu et al., 2024; Werby et al., 2024; Koch et al., 2024; Maggio et al., 2024; Jatavallabhula et al.,
2023) leverage foundation models to provide open-vocabulary 3DSG generation, which produces
more expressive representations for more diverse scenes.
Despite this progress, open-vocabulary approaches largely adhere to the pre-defined one-way
pipeline of per-image object-level information extraction followed by cross-image aggregation.
However, as illustrated in the upper part of Figure 1, constrained viewpoints, occlusions, and other
poor imaging conditions can introduce significant level of noise to object-level information extrac-
tion and reduces the accuracy of cross-image aggregation. For example, as shown in Figure 1(b),
although the object of interest is the table, the presence of an occluding vase leads to multiple crop
captions being mistakenly recognized as vase. When aggregating multiple captions or embeddings
across images, such misleading semantics can compromise the accuracy of the resulting 3DSGs.
Such noise-induced inaccuracies in 3DSGs are unacceptable for downstream robotic tasks, partic-
ularly in safety-critical scenarios. For example, if a medication bottle is incorrectly described as
a beverage container, a service robot could deliver dangerous substances to humans. Therefore, it
is crucial to assess the uncertainty in per-image object-level information and mitigate the noise ac-
cordingly. At the same time, we observe that existing methods use object crops only and miss the
comprehensive descriptors of the target object and also the broader contextual information, which
provide valuable clues for foundation models’ captioning.
1arXiv:2601.10168v1  [cs.CV]  15 Jan 2026

Ceramic VaseCoffee TableDecorative VaseWoodenTableCropCaptionsRe-shot Caption0.230.430.46Table……DoorWindow
Object RetrieverBased on Position…There is a table near the object.Briefly describe the object…VLMVaseLow-uncertainty Object Document(b) Re-shot Guided Uncertainty Estimation(c) Object-level RAG(a) Object Mapping
Best View……
……Uncertainty=1 -Similarity
(d) Edge Generation
LLMRe-shot ImagesMotivated RAG-3DSG Pipeline
Object Occlusion Frames
Constrained Viewpoint FramesMistakenCrop Captions:WoodenBoard
MistakenCrop Captions:PillowNoise in Cross-image Aggregation
……Object-level Cross-image Aggregation
RetireveAugmentGenerate
Slow Fixed Downsample-Mapping
……Good Viewpoint FramesObject Mapping
Fast Dynamic Downsample-MappingFigure 1: An overview of the RAG-3DSG framework. The upper part illustrates common challenges
in multi-view 3D scene graph generation. Our pipeline addresses these issues through (a) Multi-view
RGB-D frames are segmented and fused into a global object list with point clouds and semantic em-
beddings. (b) Re-shot images are used to select the best-view caption, which is compared with crop
captions to estimate uncertainty; clustering is applied and the top-1 cluster is retained for fusion.
(c) Low-uncertainty objects form a retrieval document, while high-uncertainty objects leverage re-
trieved context for caption refinement via a VLM. (d) Finally, spatial and semantic relationships
among objects are inferred by an LLM to construct the 3D scene graph.
To address these challenges, we proposeRAG-3DSG, a 3DSG generation framework that mitigates
noise in aggregation through re-shot guided uncertainty estimation. It treats the 3DSG under con-
struction as a database, and use the principle of Retrieval-Augmented Generation (RAG) to leverage
surrounding context to enhance the representation of objects with high uncertainty. Specifically, we
first adopt a dynamic downsample-mapping strategy to construct a global 3D object list with low
computational cost (Figure 1a). Inspired by the concept of optimal viewpoint (V ´azquez et al., 2001),
our method then renders the object point clouds reconstructed from multiple images to obtain the
best-view re-shot images. Next, we perform uncertainty estimation by comparing object captions
generated from these re-shot images with those from the crop images (Figure 1b). Low-uncertainty
objects are directly used to construct an object document, while high-uncertainty objects will trigger
the retrieval of surrounding high certainty object documents on the 3DSG to guide caption refine-
ment (Figure 1c), iteratively improve the accuracy of 3DSG.
Experiments on Replica (Straub et al., 2019) dataset demonstrate that our method consistently out-
performs existing baselines, achieving an average node precision of 0.82 (vs. 0.68 for Concept-
Graphs (Gu et al., 2024)) and edge precision of 0.91 (vs. 0.85 for ConceptGraphs-Detector (Gu et al.,
2024)), while maintaining more valid objects and edges. In addition, our dynamic downsample-
mapping strategy reduces the time of 3D object mapping by nearly two-thirds compared to the fixed
downsample-mapping strategy in ConceptGraphs (Gu et al., 2024) (from 6.65s/iter to 2.49s/iter
when the base voxel size is set to 0.01 on Replica Room 0), demonstrating improved accuracy,
robustness, and efficiency in 3DSG construction.
2

2 RELATED WORK
Scene Graph Generation (SGG)Scene graphs were initially introduced in the 2D domain to ex-
tract objects and their relationships from images, providing a structured representation that supports
a highly abstract understanding of scenes for intelligent agents (Sun et al., 2023; Liu et al., 2021; Yin
et al., 2018; Krishna et al., 2017; Lu et al., 2016; Johnson et al., 2015). A scene graph is typically
composed of three fundamental elements—objects, attributes, and relations—and can be expressed
as a set of visual relationship triplets in the form of<subject, relation, object>(Li et al., 2024). In the
field of 3D scene representations, several works (Wald et al., 2020; Kim et al., 2019; Armeni et al.,
2019) have drawn inspiration from 2D scene graphs and extended the concept into the 3D domain.
Compared with original 3D scene representations such as point clouds with per-point semantic vec-
tor, which are often overly dense and difficult to interpret, 3D scene graphs (3DSGs) provide a more
compact and structured abstraction of the scene. By organizing objects and their relationships into
a graph representation, they enable more efficient reasoning and facilitate downstream tasks such as
robotic navigation (Gadre et al., 2022; Shah et al., 2023) and scene understanding (Rana et al., 2023;
Agia et al., 2022). Early efforts in 3D scene graph generation (3DSGG) enabled the construction of
real-time systems capable of dynamically building hierarchical 3D scene representations (Hughes
et al., 2022; Rosinol et al., 2021; Wu et al., 2021). However, these methods were confined to the
closed-vocabulary setting, which restricted their applicability to a limited range of tasks. More re-
cently, several works (Gu et al., 2024; Werby et al., 2024; Koch et al., 2024; Maggio et al., 2024;
Jatavallabhula et al., 2023) have begun to explore open-vocabulary approaches for 3D scene graph
generation. Using Vision Language Models (VLMs) and Large Language Models (LLMs), these
methods sacrifice part of the real-time capability but significantly expand the range of object cate-
gories and relations that can be recognized, thus broadening the applicability to a wider variety of
downstream tasks.
Open-vocabulary 3DSGGRecent advances have extended 3DSGG to the open-vocabulary setting
by leveraging VLMs and LLMs. The existing approaches can be divided into two main paradigms.
The first paradigm follows a caption-first strategy, where objects in each image are independently
described and later aggregated into scene-level semantics. In practice, multiple masked views of the
same object are captioned separately, and a LLM is then used to aggregate these descriptions into a
final caption (e.g., Gu et al. (2024)). The second paradigm adopts an embedding-first strategy, where
semantic embeddings of multiple masked views of the same object are first extracted and aggregated
(through like weighted fusion) without explicit captioning. The aggregated embeddings are then
converted into captions or directly aligned with user queries in a joint vision–language space using
models such as CLIP (Werby et al., 2024; Koch et al., 2024; Jatavallabhula et al., 2023). In addition,
some works assign captions to object embeddings by matching them against an arbitrary vocabulary
with CLIP, a strategy commonly referred to as open-set 3DSG (Maggio et al., 2024). Despite their
differences, both paradigms share the same underlying pipeline of per-image information extraction
followed by cross-image information aggregation. However, the information extraction process is
often affected by factors such as constrained viewpoints and object occlusion, which introduce noise
into the representations. As a result, aggregated information may be inaccurate, leading to reduced
precision in the node and edge captions of the 3DSGs. This limitation cannot be simply resolved
by adopting more powerful captioning models, as the noise originates from inherent challenges in
multi-view perception such as occlusion and viewpoint constraints.
Time Considerations in 3DSGGClosed-vocabulary 3D scene graph generation methods are typ-
ically lightweight and efficient, since they operate on a fixed set of semantic categories. This ef-
ficiency enables some approaches to support online (Wu et al., 2023) or even faster-than-real-time
construction (e.g., Hou et al. (2025) achieves 7 ms end-to-end latency). In contrast, open-vocabulary
scene graph generation relies on VLMs or LLMs to provide flexible semantics, which introduces
significant runtime overhead. As a result, existing open-vocabulary methods (Gu et al., 2024; Koch
et al., 2024) are usually performed offline, making real-time deployment challenging. Unlike strictly
open-vocabulary methods, Maggio et al. (2024) adopts an open-set formulation, which relaxes cat-
egory constraints without relying on heavy VLM/LLM inference. This design makes real-time
3DSGG feasible, whereas open-vocabulary approaches typically remain offline. Even in the of-
fline setting, construction time remains a critical bottleneck: in addition to the heavy time cost of
vision-language reasoning, cross-image aggregation also incurs significant latency, further limiting
the practicality of open-vocabulary 3DSGG.
3

3 METHOD
In this section, we present our proposed framework RAG-3DSG for open-vocabulary 3D scene
graph generation. The overall pipeline is illustrated in Figure 1. Our pipeline consists of three
main stages: (1)Cross-image Object Mapping(Section 3.1), where we perform 2D segmentation,
dynamic downsampling, and 3D object fusion to construct a global object list with object-level
point clouds and semantic embeddings; (2)Node Caption Generation(Section 3.2), where we first
obtain initial captions, then perform re-shot guided uncertainty estimation, and finally refine high-
uncertainty objects using object-level RAG; (3)Edge Caption Generation(Section 3.3), where we
use a structured LLM prompt to produce interpretable relationship captions.
3.1 CROSS-IMAGEOBJECTMAPPING
Given an RGB-D image sequenceI={I 1, I2, . . . , I t}, each image is represented asI i=
⟨IRGB
i, ID
i, Pi⟩, whereIRGB
i,ID
i, andP idenote the color image, depth image, and camera pose
respectively. Each imageI iyields a set of detected objectsOlocal
i={olocal
i,1, . . . , olocal
i,M}obtained
from segmentation. Our objective is to construct a global object listOglobal
t ={oglobal
t,1, . . . , oglobal
t,N}
by incrementally fusing the per-image objectsOlocal
tinto the accumulated setOglobal
t−1. Following
Gu et al. (2024), we adopt incremental object-level cross-image mapping and further introduce a
dynamic downsample-mapping strategy to improve efficiency and effectiveness.
3.1.1 2D SEGMENTATION
Fort-th input imageIRGB
t, we first extract local object-level information. A class-agnostic segmen-
tation model SAM(·)(Kirillov et al., 2023) produces a set of 2D object masks{m t,i}i=1...M =
SAM(IRGB
t). Next, a visual semantic encoder CLIP(·)(Radford et al., 2021) is used to obtain se-
mantic embeddings{f t,i}i=1...M =CLIP(IRGB
t,{mt,i}i=1...M )for each masked region. To prepare
for 3D information aggregation, the camera poseP tis used to project each 2D mask into 3D space,
producing a set of point clouds{p t,i}i=1...M . With these extracted information, we define the local
object list ofI tasOlocal
t={olocal
t,1, . . . , olocal
t,M}, whereolocal
t,i=⟨flocal
t,i, plocal
t,i⟩. This local object list
Olocal
tserves as the basis for subsequent 3D object mapping and fusion.
3.1.2 DYNAMICDOWNSAMPLING
Before performing object mapping forolocal
t,iwith point cloudsp t,iand semantic embeddingsf t,i,
we downsample the point clouds to reduce computational cost. Existing approaches typically adopt
a fixed voxel sizeδsample, which is determined by the size of smaller objects within the scene. This
strategy has a clear drawback that large objects remain lack-sampled, resulting in unnecessarily
dense point clouds. To address this issue, we propose a dynamic downsampling strategy that adapts
the voxel size according to the scale of each object. This not only improves efficiency but also fa-
cilitates the subsequent re-shot guided uncertainty estimation (Section 3.2.2) by ensuring that object
pixels in re-shot images are dense enough to faithfully capture the underlying object. Formally, the
voxel sizeδvoxel
t,ifor the point cloudp t,iof objectolocal
t,iis defined as:
δvoxel
t,i=δsample· ∥Bbox(p t,i)∥1/2
2,(1)
whereδsampledenotes the fixed base voxel size, and∥Bbox(p t,i)∥2corresponds to the Euclidean
norm of the bounding box size vector, i.e., the diagonal length of the 3D bounding box, which
reflects its overall spatial extent. For simplicity, we continue to usep t,ito denote the downsampled
point cloud in the following sections. Compared with fixed voxel-size downsampling, our strategy
yields denser point clouds for smaller objects and sparser ones for larger objects within the scene,
thereby simultaneously achieving both finer granularity and higher efficiency, without incurring the
usual trade-off between the two. Concrete examples of dynamic voxel sizes for common indoor
objects are provided in Appendix A.2.
4

3.1.3 3D OBJECTFUSION
Incremental object mapping and fusion begins after dynamic downsampling. We use the local object
listOlocal
1in the first image to initialize the global object list asOglobal
1 =Olocal
1. Later, for each
objectolocal
t,i=⟨flocal
t,i, plocal
t,i⟩in thet-th image input, we follow Gu et al. (2024) to construct a fusion
similarityθ(i, j)as follows,
θ(i, j) =θ semantic (i, j) +θ spatial(i, j),(2)
θsemantic (i, j)is the semantic similarity betweenolocal
t,iandoglobal
t−1,j as follows,
θsemantic (i, j) = (flocal
t,i)Tfglobal
t−1,j/2 + 1/2,(3)
andθ spatial(i, j)is the spatial similarity betweenolocal
t,iandoglobal
t−1,j as follows,
θspatial(i, j) =dnnratio(plocal
t,i, pglobal
t−1,j),(4)
where dnnratio(·)is the proposed dynamic nearest neighbor ratio, equal to the proportion of points
in point cloudplocal
t,ithat have nearest neighbors in point cloudpglobal
t−1,j , within a dynamic distance
thresholdδnnratio
i,j =δsample(∥Bbox(plocal
t,i)∥1/2
2+∥Bbox(pglobal
t−1,j)∥1/2
2)/2.
By calculating fusion similarity, each new local object is matched with a global object which has
the highest similarity score. If no match is found with a similarity higher thanδsim, the local object
will be treated directly as a new global object. For the two matching objectsolocal
t,iandoglobal
t−1,j ,
the fused objectoglobal
t,j =⟨fglobal
t,j, pglobal
t,j⟩. The fused semantic embeddingfglobal
t,j is calculated as
fglobal
t,j = (nfglobal
t−1,j+flocal
t,i)/(n+1), wherenrepresents the mapping times offglobal
t−1,j . The fused point
cloud is directly taken as the union aspglobal
t,j =pglobal
t−1,j∪plocal
t,i. Aftertiterations, we construct global
object listOglobal
t ={oglobal
t,1, . . . , oglobal
t,N}, whereoglobal
t,i =⟨fglobal
t,i, pglobal
t,i⟩. The detailed algorithm is
provided in Appendix A.3.
3.2 NODECAPTIONGENERATION
Given the global object list, our goal is to derive node captions from the information aggregated
in Section 3.1. Object masks are first fed into the Vision-Language Model (VLM) to obtain initial
captions (Section 3.2.1). We then render multi-view reconstructed point clouds to produce best-view
re-shot images and perform re-shot guided uncertainty estimation by comparing captions from re-
shot and original images (Section 3.2.2). Low-uncertainty objects directly form the object document,
while high-uncertainty objects retrieve this document for caption refinement, yielding more accurate
and robust 3D scene graphs (Section 3.2.3).
3.2.1 INITIALCAPTIONGENERATION
For each object in the global object list, we maintain the top-kviews with the highest segmentation
confidence. Object-level crops from these top-kviews are fed into a VLM (Hurst et al., 2024) to
obtain initial crop captions using the prompt “briefly describe the central object in the image in a few
words.” The initial crop captions for objectoglobal
t,i are denoted asc t,i={c 1, . . . , c k}, which may be
incorrect due to constrained viewpoints or occlusion as illustrated in the upper part of Figure 1.
3.2.2 RE-SHOTGUIDEDUNCERTAINTYESTIMATION
The initial crop captions for objectoglobal
t,i may be unreliable due to constrained viewpoints and se-
vere object occlusion. Relying solely on these captions could propagate noise into the subsequent
aggregation of the object-level information. To mitigate this issue, we introduce a re-shot strategy
that generates best-view re-shot images from object point clouds. Unlike multi-view crops from the
original scene images, the reconstructed object-level point cloudpcontains only the object of inter-
est, free from occlusion and background clutter. Since the point cloud can be observed from arbitrary
viewpoints, we can render a 2D image from a perspective that maximally represents the object’s ge-
ometry and appearance. This ensures that the resulting re-shot captions capture the most informative
5

object features. Conceptually, this approach is analogous to the viewpoint entropy (V ´azquez et al.,
2001) or best-view selection problem in computer vision, where the goal is to choose a view that
maximizes information content.
Given an object point cloudpwith a maintained average camera positionvavgfrom the images used
to construct the point cloud, our goal is to render an optimal 2D view that best represents the object.
To this end, we uniformly sample multiple candidate camera positions on a hemisphere centered at
the object centero, and render the corresponding 2D re-shot images. To select the most informative
view, we define a view quality score for each candidate positionc iwith three complementary terms:
Svis=|pvisible|
|p|, S up= 1− |v i·g|, S prior=1
2(1 +v i·f),(5)
whereS vismeasures the visible ratio of points under hidden-point removal,S upevaluates the align-
ment of the view directionv i=o−c iwith the gravity vectorg, andS priorenforces consis-
tency with the prior directionf=vavg−o. The overall score is then computed asS view=
(1−α−β)S vis+αS up+βS prior, withαandβcontrolling the trade-off between uprightness
and prior alignment. Based on the view quality score, we select the candidate view with the highest
Sviewand render the corresponding 2D re-shot imageIreshot(see Appendix A.4 for examples). The
re-shot captioncreshotis then obtained from the VLM using the same prompt as in Section 3.2.1.
To quantify uncertainty, we compute the cosine similarity between the CLIP embeddings of the
re-shot caption and the initial crop captions:
{s1, . . . , s k}=
cos 
CLIP(creshot),CLIP(c i)	k
i=1.(6)
We then perform clustering on the similarity scores and select the top-1 cluster
{c1, . . . , c l},{s 1, . . . , s l}, l≤k, which is considered the most reliable subset of initial
captions for further refinement. The captions in this subset are aggregated into a single caption
ˆcusing a Large Language Model (LLM) (Achiam et al., 2023) with a designed prompt. The
corresponding similarity scores are averaged to obtainˆs=1
lPl
i=1si,where a higherˆsindicates
stronger agreement among the captions, and thus lower uncertainty inˆc.
3.2.3 OBJECT-LEVELRAG
Based on the re-shot guided uncertainty estimation introduced in the previous section, we first rank
all objects by their uncertainty scores1−ˆs. We additionally apply a prompt to the VLM (Hurst et al.,
2024) to filter out background objects via crops. The top-50%low-uncertainty objects are directly
included in the object document for RAG, where each final captioncis set toˆc. For the remaining
high-uncertainty objects, we perform refinement with the aid of contextual information. Specifically,
a 3D position-based retriever retrieves the nearest object in the document, whose captioncenvserves
as augmented auxiliary context. In addition, we construct a composite image by concatenating the
re-shot image (providing global context) with the crop image that yields the highest similarity score.
This composite image, together with a text prompt containingcenv, is fed into a VLM (Hurst et al.,
2024) to generate the refined captionc. The prompt is designed as: “The picture is stitched from
the point cloud image and the RGB image of the same indoor object. There is acenvnear the object.
Briefly describe the object in the picture.” Through the refinement process, we obtain a precise
object listO={o 1, . . . , o N}, where each objecto iis represented aso i=⟨f i, pi, ci⟩, consisting of
its semantic embeddingf i, point cloudp i, and precise node captionc i.
3.3 EDGECAPTIONGENERATION
Following Gu et al. (2024), we estimate the spatial relationships among 3D objects to complete
the scene graph. Given the set of 3D objectsO={o 1, . . . , o N}, we first compute their potential
connectivity. Unlike Gu et al. (2024), which adopts a fixed NN-ratio threshold to build a dense
graph and then prunes it with a minimum spanning tree (MST), we introduce a dynamic threshold
similar to Equation 4, consistent with our dynamic downsample-mapping strategy, thus adapting the
edge construction to varying point cloud densities.
For relationship captioning, Gu et al. (2024) employ a structured prompt that restricts the output to
five predefined relation types and uses object captions and 3D locations as inputs. In contrast, we
6

extend this design by (i) expanding the relation space to eight categories (“a on b, ” “b on a, ” “a
in b, ” “b in a, ” “a part of b, ” “b part of a, ” “near, ” “none of these”), and (ii) providing few-shot
in-context examples to guide the model. This design ensures that the generated relationships are
more expressive.
3.4 SCENEGRAPHREPRESENTATION
With the refined object list and edge captions, we formally define the final 3D scene graph asG=
(O, E), whereO={o 1, . . . , o N}denotes the set of objects andE={e ij}the set of edges. Each
objecto iis represented aso i=⟨f i, pi, ci⟩, consisting of its semantic embeddingf i, point cloudp i,
and precise captionc i. Each edgee ijis represented ase ij=⟨o i, oj, rij⟩, wherer ijis the discrete
relation label selected from the predefined set of eight categories. This formulation yields a complete
and interpretable 3D scene graph with both node- and edge-level semantic annotations (Figure 2).
Figure 2: Visualization of our 3DSGs for Replica (Straub et al., 2019) Room 0 (left) and 1 (right).
The blue points represent the objects and the red lines indicate the relationships between them.
4 EXPERIMENTS
4.1 IMPLEMENTATIONDETAILS
For object segmentation, we use SAM(·)with the pretrained checkpointsam vith4b8939.
For encoding semantic embeddings, we use CLIP(·)with theViT-H-14backbone pretrained on
laion2b s32b b79k. For all purely text-based tasks, we employgpt-4-0613as the LLM,
while for vision-language tasks involving both images and text, we adoptgpt-4o-minias the
VLM. Regarding hyperparameters, we set the similarity threshold for object mapping toδsim= 0.45,
and the base voxel size for dynamic downsampling toδsample= 0.01meters. For the scoring function
in re-shot guided uncertainty estimation, we setα=β= 0.2. Other non-critical hyperparameters
and prompts will be released with our code in the accompanying GitHub repository.
4.2 SCENEGRAPHCONSTRUCTIONEVALUATION
We first evaluate the accuracy of the generated 3D scene graphs on Replica (Straub et al., 2019)
dataset in Table 1. We compare our method against two baselines: ConceptGraphs (CG) and its vari-
ant ConceptGraphs-Detector (CG-D) (Gu et al., 2024). The open-vocabulary nature of our method
makes automatic evaluation challenging. Therefore, following the protocol of ConceptGraphs (Gu
et al., 2024), we resort to human evaluation. We recruited knowledgeable university students as an-
notators and randomly shuffled the basic evaluation units before distribution. For node evaluation,
each unit consists of a point cloud, its mask images, and the predicted node caption, and annotators
are asked to judge whether the caption is accurate. For edge evaluation, each unit includes two such
node units along with the predicted edge caption and the whole scene point clouds, and annotators
are similarly asked to assess the correctness of the relationship description.
7

Table 1: Performance comparison of 3D scene graph generation methods on Replica (Straub et al.,
2019) dataset. Node precision, edge precision, duplicate objects and object/edge counts are evalu-
ated through human annotation across multiple indoor scenes (room0-office4).
scene node prec. valid objects duplicates edge prec. valid edges
Ours CG CG-D Ours CG CG-D Ours CG CG-D Ours CG CG-D Ours CG CG-D
room0 0.870.77 0.53 61 57 60 1 4 3 0.930.87 0.88 27 15 16
room1 0.880.73 0.71 51 45 42 0 5 3 0.970.92 0.91 30 12 11
room2 0.850.63 0.50 47 48 50 0 3 2 0.940.91 0.92 35 11 12
office0 0.730.61 0.61 48 44 41 1 1 1 0.930.78 0.82 27 9 11
office1 0.730.64 0.46 44 25 24 0 1 3 0.930.80 0.86 28 5 7
office2 0.870.77 0.68 67 48 44 1 3 2 0.880.79 0.86 34 14 14
office3 0.850.69 0.60 65 59 57 2 4 2 0.840.78 0.77 32 9 13
office4 0.790.61 0.57 53 41 46 1 5 4 0.860.67 0.80 22 3 5
Average 0.82 0.68 0.58 - - - - - - 0.91 0.82 0.85 - - -
From Table 1, our method consistently outperforms both ConceptGraphs (CG) and ConceptGraphs-
Detector (CG-D) across most evaluation metrics. In terms of node precision, our method achieves an
average score of0.82, which is notably higher than CG (0.68) and CG-D (0.58), demonstrating the
effectiveness of our re-shot guided uncertainty estimation in reducing noise during object caption ag-
gregation. For edge precision, our method also attains the highest average score (0.91), surpassing
CG (0.82) and CG-D (0.85), indicating that our structured prompt and refined relationship cate-
gories lead to more accurate and interpretable edge captions. In addition, our method substantially
reduces duplicate predictions while maintaining a higher number of valid objects and edges, further
confirming its robustness.
Overall, these results validate that our approach not only improves caption accuracy at both the node
and edge levels but also enhances the reliability of the entire 3D scene graph construction pipeline.
4.3 SEMANTICSEGMENTATIONEVALUATION
We evaluate our dynamic downsample-mapping strategy on the closed-set Replica dataset (Straub
et al., 2019) following the ground-truth construction and evaluation protocol of Gu et al. (2024);
Jatavallabhula et al. (2023). Concretely, the ground-truth (GT) point clouds with per-point semantic
labels are obtained as in ConceptGraph Gu et al. (2024): SemanticNeRF (Zhi et al., 2021) provides
rendered RGB-D frames and 2D semantic masks, the masks are converted to one-hot per-pixel
embeddings and fused into 3D via GradSLAM (Krishna Murthy et al., 2020), yielding the reference
GT point cloud with per-point semantic annotations.
After performing the cross-image object mapping described in Section 3.1, our method produces
object-level point clouds with fused semantic embeddings. To align predictions with GT categories,
we map each GT object label to the predicted object whose fused semantic embedding has the high-
est cosine similarity with the CLIP text embedding of that GT label. Following the same protocol,
for each point in the GT point cloud we compute its 1-NN in the predicted point cloud, compare
the GT class with the predicted class of that 1-NN to build the confusion matrix, and compute the
class-mean recall (mAcc) and the frequency-weighted mean intersection-overunion (f-mIOU).
We report our result combined with the results reported in Maggio et al. (2024); Gu et al. (2024) in
Table 2. Our method shares the same overall matching pipeline with ConceptGraph, but significantly
reduces the computational cost. Under the same base voxel size, our dynamic downsample-mapping
strategy shortens the processing time by nearly two-thirds (e.g., from 6.65s/iter to 2.49s/iter when
the voxel size is set to 0.01 in Replica (Straub et al., 2019) Room 0). Moreover, as shown in Table
2, our method achieves comparable or even superior accuracy, reaching the best mAcc score (40.67)
while maintaining a competitive f-mIoU (35.65). This demonstrates that our approach not only
accelerates the object mapping process but also preserves segmentation quality.
4.4 ABLATIONSTUDY
We conduct ablation studies to quantify the contribution of each component in our pipeline. Our
full model (Ours) consists of dynamic downsample-mapping & fusion, re-shot guided uncertainty
estimation, and node-level RAG with concatenated re-shot image prompts. We compare against
8

Table 2: Semantic segmentation experiments
on Replica (Straub et al., 2019) dataset for ob-
ject mapping and time evaluation. Baseline re-
sults are reported from Maggio et al. (2024); Gu
et al. (2024). mAcc denotes class-mean recall
and f-mIOU denotes frequency-weighted mean
intersection-over-union reported from (Jataval-
labhula et al., 2023).
Method mAcc F-mIOU
MaskCLIP 4.53 0.94
Mask2former + Global CLIP feat 10.42 13.11
ConceptFusion 24.16 31.31
ConceptFusion + SAM 31.53 38.70
ConceptGraphs 40.63 35.95
ConceptGraphs-Detector 38.72 35.82
OpenMask3D 39.5449.26
Clio-batch 37.95 36.98
Ours40.6735.65
mIoU mRecall mPrecision mF1 f-mIoUOurs
w/o Concat
w/o RAG
Random RAG
w/o Reshotmethod23.60 37.37 34.87 30.78 54.26
21.41 33.61 30.04 28.09 51.16
21.55 32.36 31.40 27.96 49.08
18.90 31.59 26.19 25.39 49.06
10.48 31.09 22.28 14.66 35.56
 1520253035404550Figure 3: Ablation study quantifying the con-
tribution of each pipeline component. Perfor-
mance degradation is observed when removing re-
shot guided uncertainty estimation (w/o Reshot),
RAG (w/o RAG), concatenated re-shot images
(w/o Concat), or using random retrieval (Random
RAG), confirming the complementary roles of all
proposed components.
several variants: (i) removing re-shot guided uncertainty estimation (w/o Reshot), (ii) removing
RAG (w/o RAG), (iii) applying random retrieval for RAG (random-RAG), and (iv) removing the
concatenated re-shot image prompts (w/o Concat).
Experiments are conducted on Replica (Straub et al., 2019) dataset, following a protocol similar to
Section 4.3. The ground-truth (GT) point clouds with per-point semantic labels are obtained as in
ConceptGraphs (Gu et al., 2024). Different from Section 4.3, where fused semantic embeddings are
directly compared with GT embeddings via cosine similarity, here we leverage the final node cap-
tions of predicted objects as semantic representations and employ GPT-4o as a semantic assigner.
Concretely, for each GT object label, GPT-4o is prompted to determine the most likely correspond-
ing predicted node based on the node captions, thereby enabling a more faithful evaluation under the
open-vocabulary setting. After establishing this assignment, we compute 1-NN matching between
GT and predicted point clouds, construct the confusion matrix, and report quantitative results in
Figure 3. Detailed heatmaps for each individual scene are provided in Appendix A.5.
As shown in Figure 3, removing any component leads to a noticeable performance drop, confirm-
ing their complementary roles. In particular, removing re-shot guided uncertainty estimation (w/o
Reshot) causes a drastic degradation in mF1 (14.66 vs. 30.78) and f-mIoU (35.56 vs. 54.26), un-
derscoring its importance in filtering unreliable captions. Eliminating RAG (w/o RAG) reduces both
precision and overall accuracy, while replacing it with random retrieval (random-RAG) further dete-
riorates performance, highlighting the necessity of semantic-aware retrieval. Removing the concate-
nated re-shot image (w/o Concat) also leads to lower recall and f-mIoU, suggesting that multi-view
prompts alleviate viewpoint bias and enrich object descriptions. These results collectively demon-
strate that all three proposed components contribute significantly to the robustness and accuracy of
our framework.
5 CONCLUSION
In this work, we propose a 3DSG generation method named RAG-3DSG for more accurate and
robust 3DSGs. We are the first to specifically address noise in cross-image information aggregation
and incorporate an object-level RAG into 3DSGs for caption refinement. To evaluate our approach,
we conduct experiments on Replica (Straub et al., 2019) dataset, which shows that RAG-3DSG
significantly improves node captioning accuracy in 3DSG generation.
9

ETHICSSTATEMENT
This work focuses on developing methods for open-vocabulary 3D scene graph generation using
public dataset (Replica (Straub et al., 2019)). It does not involve human subjects, personal data,
or sensitive information. Our work is intended solely for advancing robotics and embodied AI
research in safe and beneficial contexts, such as robot navigation and manipulation. We encourage
responsible use aligned with ethical research practices.
REPRODUCIBILITYSTATEMENT
We have made extensive efforts to ensure reproducibility. All implementation details, including
pipline architectures, foundation models, and hyperparameters, are provided in the main text and
appendix. The datasets we used (Replica (Straub et al., 2019)) are publicly available. Our code,
along with evaluation scripts, will be released on GitHub to facilitate replication. We also describe
the evaluation procedure in detail, so that results can be reproduced independently.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vi-
neet, Mustafa Mukadam, Liam Paull, and Florian Shkurti. Taskography: Evaluating robot task
planning over large 3d scene graphs. InConference on Robot Learning, pp. 46–58. PMLR, 2022.
Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Zamir, Martin Fischer, Jitendra Malik, and
Silvio Savarese. 3d scene graph: A structure for unified semantics, 3d space, and camera. In
Proceedings of the IEEE/CVF international conference on computer vision, pp. 5664–5673, 2019.
Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Ilharco, Ludwig Schmidt, and Shuran Song. Clip
on wheels: Zero-shot object navigation as object localization and exploration.arXiv preprint
arXiv:2203.10421, 3(4):7, 2022.
Paul Gay, James Stuart, and Alessio Del Bue. Visual graphs from motion (vgfm): Scene under-
standing with object geometry reasoning. InAsian Conference on Computer Vision, pp. 330–346.
Springer, 2018.
Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya
Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs:
Open-vocabulary 3d scene graphs for perception and planning. In2024 IEEE International Con-
ference on Robotics and Automation (ICRA), pp. 5021–5028. IEEE, 2024.
Hao-Yu Hou, Chun-Yi Lee, Motoharu Sonogashira, and Yasutomo Kawanishi. Fross: Faster-than-
real-time online 3d semantic scene graph generation from rgb-d images.ArXiv, abs/2507.19993,
2025. URLhttps://api.semanticscholar.org/CorpusID:280322834.
Nathan Hughes, Yun Chang, and Luca Carlone. Hydra: A real-time spatial perception system for
3d scene graph construction and optimization.arXiv preprint arXiv:2201.13360, 2022.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Os-
trow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv preprint
arXiv:2410.21276, 2024.
Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa
Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-
set multimodal 3d mapping.arXiv preprint arXiv:2302.07241, 2023.
Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and
Li Fei-Fei. Image retrieval using scene graphs. InProceedings of the IEEE conference on com-
puter vision and pattern recognition, pp. 3668–3678, 2015.
10

Ue-Hwan Kim, Jin-Man Park, Taek-Jin Song, and Jong-Hwan Kim. 3-d scene graph: A sparse and
semantic representation of physical environments for intelligent agents.IEEE transactions on
cybernetics, 50(12):4921–4933, 2019.
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. InProceed-
ings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.
Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski.
Open3dsg: Open-vocabulary 3d scene graphs from point clouds with queryable objects and open-
set relationships. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 14183–14193, 2024.
Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie
Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting lan-
guage and vision using crowdsourced dense image annotations.International journal of computer
vision, 123(1):32–73, 2017.
Jatavallabhula Krishna Murthy, Soroush Saryazdi, Ganesh Iyer, and Liam Paull. gradslam: Dense
slam meets automatic differentiation. InarXiv, 2020.
Hongsheng Li, Guangming Zhu, Liang Zhang, Youliang Jiang, Yixuan Dang, Haoran Hou, Peiyi
Shen, Xia Zhao, Syed Afaq Ali Shah, and Mohammed Bennamoun. Scene graph generation: A
comprehensive survey.Neurocomputing, 566:127052, 2024.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances
in neural information processing systems, 36:34892–34916, 2023.
Hengyue Liu, Ning Yan, Masood Mortazavi, and Bir Bhanu. Fully convolutional scene graph gen-
eration. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 11546–11556, 2021.
Cewu Lu, Ranjay Krishna, Michael Bernstein, and Li Fei-Fei. Visual relationship detection with
language priors. InEuropean conference on computer vision, pp. 852–869. Springer, 2016.
Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, Dan Griffith, Carlyn Dougherty,
Eric Cristofalo, Lukas Schmid, and Luca Carlone. Clio: Real-time task-driven open-set 3d scene
graphs.IEEE Robotics and Automation Letters, 2024.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. InInternational conference on machine learning, pp.
8748–8763. PmLR, 2021.
Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf.
Sayplan: Grounding large language models using 3d scene graphs for scalable robot task plan-
ning.arXiv preprint arXiv:2307.06135, 2023.
Adam Rashid, Satvik Sharma, Chung Min Kim, Justin Kerr, Lawrence Yunliang Chen, Angjoo
Kanazawa, and Ken Goldberg. Language embedded radiance fields for zero-shot task-oriented
grasping. In7th Annual Conference on Robot Learning, 2023.
Antoni Rosinol, Andrew Violette, Marcus Abate, Nathan Hughes, Yun Chang, Jingnan Shi, Arjun
Gupta, and Luca Carlone. Kimera: From slam to spatial perception with 3d dynamic scene graphs.
The International Journal of Robotics Research, 40(12-14):1510–1546, 2021.
Dhruv Shah, Bła ˙zej Osi ´nski, Sergey Levine, et al. Lm-nav: Robotic navigation with large pre-
trained models of language, vision, and action. InConference on robot learning, pp. 492–504.
PMLR, 2023.
Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic
manipulation. InConference on robot learning, pp. 894–906. PMLR, 2022.
11

Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel,
Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor
spaces.arXiv preprint arXiv:1906.05797, 2019.
Shuzhou Sun, Shuaifeng Zhi, Janne Heikkil ¨a, and Li Liu. Evidential uncertainty and diversity guided
active learning for scene graph generation. InThe Eleventh International Conference on Learning
Representations, 2023. URLhttps://openreview.net/forum?id=xI1ZTtVOtlz.
Pere-Pau V ´azquez, Miquel Feixas, Mateu Sbert, and Wolfgang Heidrich. Viewpoint selection using
viewpoint entropy. InVMV, volume 1, pp. 273–280, 2001.
Johanna Wald, Helisa Dhamo, Nassir Navab, and Federico Tombari. Learning 3d semantic scene
graphs from 3d indoor reconstructions. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 3961–3970, 2020.
Abdelrhman Werby, Chenguang Huang, Martin B ¨uchner, Abhinav Valada, and Wolfram Burgard.
Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. InFirst
Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.
Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab, and Federico Tombari. Scene-
graphfusion: Incremental 3d scene graph prediction from rgb-d sequences. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7515–7525, 2021.
Shun-Cheng Wu, Keisuke Tateno, Nassir Navab, and Federico Tombari. Incremental 3d semantic
scene graph prediction from rgb sequences. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 5064–5074, 2023.
Guojun Yin, Lu Sheng, Bin Liu, Nenghai Yu, Xiaogang Wang, Jing Shao, and Chen Change Loy.
Zoom-net: Mining deep feature interactions for visual relationship recognition. InProceedings
of the European conference on computer vision (ECCV), pp. 322–338, 2018.
Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene la-
belling and understanding with implicit scene representation. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 15838–15847, 2021.
12

A APPENDIX
A.1 LARGELANGUAGEMODELUSAGE
Large Language Models are utilized in this work to aid and polish writing, specifically to improve
clarity, grammatical accuracy, and overall readability of the manuscript. All technical content, sci-
entific contributions, and research findings remain entirely the work of the authors, who take full
responsibility for the entire content of this paper.
A.2 DYNAMICDOWNSAMPLEEXAMPLES
We provide detailed examples of our dynamic downsampling strategy for indoor 3D pointclouds in
Table 3. The voxel size for each object is computed based on its spatial extent using Equation 1 in
Section 3.1.2.
A.3 PSEUDOCODE FOROBJECTMAPPING
For clarity and reproducibility, we provide the pseudo code of our incremental 3D object mapping
algorithm in Algorithm 1. This algorithm describes the step-by-step procedure of updating global
object list from local object lists of incoming frames, including point cloud integration and semantic
refinement. It serves as a concise summary of the mapping pipeline presented in the main paper.
A.4 RE-SHOTIMAGES
Figure 4 shows some examples of re-shot images automatically selected by our method. As dis-
cussed in Section 3.2.2, the initial image crops of an object may suffer from occlusions or con-
strained viewpoints, leading to unreliable captions and noisy semantics. To address this issue, we
propose a re-shot strategy that leverages object-level point clouds to render new views from arbi-
trary perspectives, ensuring that the essential geometry and appearance of the object are faithfully
captured. This approach effectively eliminates occlusion and viewpoint constraints inherent in the
original images.
A.5 ALLHEATMAPS OFABLATIONSTUDY ONREPLICA
As shown in Figure 5, we present the heatmaps of semantic segmentation metrics across different
scenes and ablation settings.
A.6 CASESTUDY
Figure 6 demonstrates a concrete example of our re-shot guided uncertainty estimation when
LLaV A (Liu et al., 2023) is the VLM. Multiple crop images incorrectly identify the object as“vase”
due to occlusions and constrained viewing angles. In contrast, the re-shot image (top-left) provides
a correct caption from an optimal viewpoint selected from our algorithm, avoiding the viewpoint
limitations and occlusions that plague the original crop images. The low similarity scores flag these
crop captions as unreliable, triggering our object-level RAG caption refinement process.
13

Top-1
Top-1Top-1Top-1Top-1Top-2
Top-2Top-2Top-2Top-2Top-3Top-3Top-3Top-3Top-3Top-4Top-4Top-4Top-4Top-4
Top-1Top-2Top-3Top-4Figure 4: Examples of re-shot images automatically selected by our method. For each object cat-
egory, we present the top-4 rendered views ranked by our re-shot scoring strategy. From top to
bottom: armchair, vase, cushion, sofa, artwork, and TV stand.
14

full_pipeline w/o concatw/o rag
random ragw/o reshot
Methodroom0
room1
room2
office0
office1
office2
office3
office4Scene28.21 25.32 22.17 22.37 11.24
26.79 25.17 25.49 24.69 9.14
16.57 20.77 16.83 20.57 12.12
19.82 21.64 20.41 19.06 15.10
11.01 9.20 6.02 12.58 5.26
20.74 24.16 24.21 15.54 11.95
21.80 24.05 22.71 18.73 9.80
20.12 21.74 21.70 19.75 10.44mIoU
full_pipeline w/o concatw/o rag
random ragw/o reshot
Methodroom0
room1
room2
office0
office1
office2
office3
office4Scene38.97 39.80 38.47 38.85 32.05
33.15 32.12 33.18 33.07 19.41
31.61 30.88 27.16 31.46 22.32
30.19 29.37 27.99 32.66 23.78
14.58 14.28 11.10 17.24 11.78
26.40 28.65 28.00 22.44 17.76
27.69 28.81 28.01 24.37 18.70
26.29 27.78 27.43 34.39 18.99mRecall
full_pipeline w/o concatw/o rag
random ragw/o reshot
Methodroom0
room1
room2
office0
office1
office2
office3
office4Scene37.37 32.57 29.72 30.04 19.72
34.46 32.32 32.78 30.64 11.29
20.89 26.83 21.70 26.30 13.75
26.82 30.92 26.27 23.68 19.66
16.03 14.72 11.37 17.68 5.39
33.92 32.68 34.21 25.95 22.11
28.09 31.34 27.65 25.87 10.04
24.25 28.03 30.40 25.95 16.66mPrecision
full_pipeline w/o concatw/o rag
random ragw/o reshot
Methodroom0
room1
room2
office0
office1
office2
office3
office4Scene34.45 30.41 28.34 28.10 14.86
31.83 29.61 30.40 29.36 11.37
20.66 25.05 20.66 24.88 15.00
24.68 26.06 24.37 23.50 18.18
13.87 11.68 8.85 14.96 5.63
25.43 28.19 28.98 19.89 14.66
26.26 28.21 26.80 22.70 11.49
24.09 26.04 26.52 24.15 12.17mF1score
full_pipeline w/o concatw/o rag
random ragw/o reshot
Methodroom0
room1
room2
office0
office1
office2
office3
office4Scene57.36 57.70 57.52 57.46 45.80
65.81 65.09 67.14 66.42 22.82
31.77 38.90 32.22 38.72 31.57
48.08 30.10 30.17 30.17 23.48
20.94 16.68 18.60 21.23 16.83
48.98 55.59 54.49 48.26 42.51
59.52 59.78 59.64 59.23 59.40
66.94 68.11 68.13 67.79 41.89fMiou
10152025
1520253035
101520253035
1015202530
2030405060Figure 5: Heatmaps of semantic segmentation performance across scenes for different ablation
settings. Metrics include mIoU, mRecall, mPrecision, mF1score, and frequency-weighted mIoU
(fMiou). The full pipeline serves as the baseline, while “w/o concat”, “w/o rag”, “random rag”, and
“w/o reshot” show the impact of removing or modifying individual components.
15

Figure 6: Case study of re-shot guided uncertainty estimation using LLaV A (Liu et al., 2023) as
VLM. Top panel: Re-shot image with caption (top-left) and crop images from original viewpoints
with their captions and similarity scores to the re-shot image. Bottom panel: Complete scene re-
construction via GradSLAM (Krishna Murthy et al., 2020) with the target object highlighted in red
bounding box. The crop images consistently misidentify the object as “vase” due to occlusions and
constrained viewpoints, resulting in low similarity scores that indicate high uncertainty.
16

Table 3: Examples of dynamic voxel sizes for different objects with base voxel sizeδsample= 0.01m.
Object Category Typical Size (L×W×H) Diagonal Length Voxel Size Reduction Factor
Small Objects
Coffee cup 0.08×0.08×0.12 0.165 m 0.004 m 2.5×
Smartphone 0.15×0.07×0.008 0.166 m 0.004 m 2.5×
Computer mouse 0.12×0.06×0.04 0.14 m 0.0037 m 2.7×
Medium Objects
Monitor 0.55×0.32×0.05 0.638 m 0.008 m 1.25×
Chair 0.60×0.55×0.85 1.177 m 0.011 m 0.9×
Desk 1.20×0.60×0.75 1.537 m 0.012 m 0.8×
Large Objects
Sofa 2.00×0.90×0.85 2.352 m 0.015 m 0.67×
Dining table 2.50×1.20×0.75 2.873 m 0.017 m 0.59×
Bookshelf 0.80×0.30×2.20 2.36 m 0.015 m 0.67×
Extra Large Objects
Wall 10.0×0.20×3.0 10.44 m 0.032 m 0.03×
Floor 10.0×10.0×0.05 10.00 m 0.0316 m 0.03×
Algorithm 13D Object Fusion with dynamic threshold
Require:Local object listOlocal
tfor framet, similarity thresholdδsim, sample thresholdδsample
Ensure:Global object listOglobal
t
1:Initialize global object list:Oglobal
1 =Olocal
1
2:fort= 2toTdo
3:foreach local objectolocal
t,i= (flocal
t,i, plocal
t,i)∈Olocal
tdo
4:best match=None
5:max similarity= 0
6:foreach global objectoglobal
t−1,j = (fglobal
t−1,j, pglobal
t−1,j)∈Oglobal
t−1 do
7:// Calculate semantic similarity
8:θ semantic (i, j) = (flocal
t,i)Tfglobal
t−1,j/2 + 1/2
9:// Calculate spatial similarity
10:δnnratio
i,j =δsample·(∥Bbox(plocal
t,i)∥1/2
2+∥Bbox(pglobal
t−1,j)∥1/2
2)/2
11:θ spatial(i, j) =dnnratio(plocal
t,i, pglobal
t−1,j){Using thresholdδnnratio
i,j}
12:// Calculate fusion similarity
13:θ(i, j) =θ semantic (i, j) +θ spatial(i, j)
14:ifθ(i, j)>max similaritythen
15:max similarity=θ(i, j)
16:best match=j
17:end if
18:end for
19:ifmax similarity> δsimand best match̸=Nonethen
20:// Fuse with matched global object
21:j=best match
22:n=mapping times(fglobal
t−1,j)
23:fglobal
t,j = (n·fglobal
t−1,j +flocal
t,i)/(n+ 1)
24:pglobal
t,j =pglobal
t−1,j∪plocal
t,i
25:oglobal
t,j = (fglobal
t,j, pglobal
t,j)
26:else
27:// Create new global object
28:Addolocal
t,itoOglobal
t as a new global object
29:end if
30:end for
31:end for
32:returnOglobal
t ={oglobal
t,1, . . . , oglobal
t,N}whereoglobal
t,i = (fglobal
t,i, pglobal
t,i)
17