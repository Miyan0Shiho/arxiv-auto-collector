# KeySG: Hierarchical Keyframe-Based 3D Scene Graphs

**Authors**: Abdelrhman Werby, Dennis Rotondi, Fabio Scaparro, Kai O. Arras

**Published**: 2025-10-01 15:53:27

**PDF URL**: [http://arxiv.org/pdf/2510.01049v1](http://arxiv.org/pdf/2510.01049v1)

## Abstract
In recent years, 3D scene graphs have emerged as a powerful world
representation, offering both geometric accuracy and semantic richness.
Combining 3D scene graphs with large language models enables robots to reason,
plan, and navigate in complex human-centered environments. However, current
approaches for constructing 3D scene graphs are semantically limited to a
predefined set of relationships, and their serialization in large environments
can easily exceed an LLM's context window. We introduce KeySG, a framework that
represents 3D scenes as a hierarchical graph consisting of floors, rooms,
objects, and functional elements, where nodes are augmented with multi-modal
information extracted from keyframes selected to optimize geometric and visual
coverage. The keyframes allow us to efficiently leverage VLM to extract scene
information, alleviating the need to explicitly model relationship edges
between objects, enabling more general, task-agnostic reasoning and planning.
Our approach can process complex and ambiguous queries while mitigating the
scalability issues associated with large scene graphs by utilizing a
hierarchical retrieval-augmented generation (RAG) pipeline to extract relevant
context from the graph. Evaluated across four distinct benchmarks -- including
3D object segmentation and complex query retrieval -- KeySG outperforms prior
approaches on most metrics, demonstrating its superior semantic richness and
efficiency.

## Full Text


<!-- PDF content starts -->

KeySG: Hierarchical Keyframe-Based 3D Scene Graphs
Abdelrhman Werby Dennis Rotondi Fabio Scaparro Kai O. Arras
Abstract— In recent years, 3D scene graphs have emerged as a
powerful world representation, offering both geometric accuracy
and semantic richness. Combining 3D scene graphs with large
language models enables robots to reason, plan, and navigate
in complex human-centered environments. However, current
approaches for constructing 3D scene graphs are semantically
limited to a predefined set of relationships, and their serialization
in large environments can easily exceed an LLM’s context
window. We introduce KeySG, a framework that represents 3D
scenes as a hierarchical graph consisting of floors, rooms, objects,
and functional elements, where nodes are augmented with multi-
modal information extracted from keyframes selected to optimize
geometric and visual coverage. The keyframes allow us to
efficiently leverage VLM to extract scene information, alleviating
the need to explicitly model relationship edges between objects,
enabling more general, task-agnostic reasoning and planning.
Our approach can process complex and ambiguous queries
while mitigating the scalability issues associated with large
scene graphs by utilizing a hierarchical retrieval-augmented
generation (RAG) pipeline to extract relevant context from the
graph. Evaluated across four distinct benchmarks –including
3D object segmentation and complex query retrieval– KeySG
outperforms prior approaches on most metrics, demonstrating
its superior semantic richness and efficiency.
I. INTRODUCTION
A long-standing goal in robotics is to create autonomous
agents that can operate effectively in human-centered en-
vironments such as homes or offices. These environments
are characterized by high object density, semantic richness,
and a variety of potential tasks. A key challenge for this
goal is the development of a 3D world representation that is
simultaneously detailed for precise manipulation and abstract
enough for high-level reasoning and long-horizon planning.
3D scene graphs (3DSGs) [1]–[5] have gained significant
attention as a powerful representation to address the limita-
tions of purely geometric maps. By modeling the world as a
graph where nodes represent entities and edges represent their
relationships, 3DSGs impose a structure on raw perception,
explicitly linking geometry to semantics.
However, current 3D scene graph approaches have two
main limitations: first, they are restricted to a predefined set of
geometric or semantic relationships, reducing the diversity of
tasks and queries they can support. For instance, a 3DSG [6]
with edges representing spatial relationships between objects
and places would excel in locating objects in large buildings.
In contrast, a 3DSG [7] encoding functional relationships
would be well suited for tasks that require understanding how
functional elements control their objects (e.g, “turn off the
oven," where knowing the relationship between the knob and
All the authors are with the Socially Intelligent Robotics Lab, Insti-
tute for Artificial Intelligence University of Stuttgart, Germany. Email:
{first.last}@ki.uni-stuttgart.de
Fig. 1: As illustrated (top), KeySG is a hierarchical, keyframe-based 3D scene
graph comprising floors, rooms, objects, and functional elements (bottom
right). Each node is augmented with contextual information efficiently
extracted from scene keyframes via adaptive keyframe sampling (bottom
left). Leveraging a multimodal RAG pipeline, KeySG enables users to ask
complex natural language queries and receive answers grounded in the 3D
scene (bottom middle).
the oven is necessary). A 3D scene graph designed with a
predefined set of relationships for a specific task is inherently
suboptimal for others.
Second, scalability is a major bottleneck. 3D scene graphs
are often paired with large language models (LLMs), serving
as a persistent world model that the LLM uses for planning
and high-level reasoning. However, providing a complete,
detailed scene graph of a large-scale environment, such as a
multi-story office building, directly to an LLM can exceed
the context window limits of even the most advanced models.
Even if the graph fits within the context window, LLMs suffer
from attentional biases and a “lost in the middle" problem
[8], where performance degrades as the model is distracted by
the vast amount of task-irrelevant information present in the
prompt. This makes it challenging for the LLM to identify
the crucial entities and environmental states necessary for
robust reasoning and planning.
To this end, we present the Hierarchical KeyFrame-Based
3D Scene Graphs (KeySG), a novel framework that resolves
the semantic and scalability dilemma by augmenting the
3D scene graph with multi-modal contextual information,
implicitly capturing the geometry, semantics, affordances,
and states of objects. Our key idea is to sample keyframes
that ensure comprehensive visual coverage of each room
in the environment. A Vision-Language Model (VLM) then
generates a detailed description for each keyframe. To address
scalability, these descriptions are recursively summarized intoarXiv:2510.01049v1  [cs.CV]  1 Oct 2025

concise textual overviews for rooms and, subsequently, entire
floors. This hierarchy is queried using a multi-modal retrieval-
augmented generation (RAG) pipeline, which ensures that
only the most relevant information is retrieved and provided
to the LLM planner, enabling efficient and accurate reasoning.
In summary, we make the following contributions:
•We introduce KeyFrame-Based 3DSGs (KeySG), the
first 3D Scene Graph framework to model environments
across five hierarchical levels of abstraction: buildings,
floors, rooms, objects, and functional elements.
•We propose a new pipeline to augment 3DSGs with
multi-modal context from keyframes, featuring hierar-
chical scene summarization, and a RAG-based retrieval
mechanism that efficiently provides task-relevant context
to LLMs.
•We conduct a comprehensive evaluation of KeySG across
diverse benchmarks, including open-vocabulary 3D seg-
mentation (Replica [9]), functional element segmentation
(FunGraph3D [10]), and 3D object grounding from
natural language queries (Habitat Matterport [11], Nr3D
[12]).
Our code will be made publicly available upon acceptance
at:https://github.com/anonymous/keysg.
II. RELATEDWORK
A. 3D Scene Graphs
The idea of 3D Scene Graphs (3DSGs) [13], [14] is to
represent a scene as a graph G= (V,E) , where the nodes V
correspond to objects and spatial entities (e.g., cameras, rooms,
floors, buildings), while the edges Eencode relationships such
ashierarchical (e.g., A “is part of” B), spatial (e.g.,
A “is next to” B), and comparative (e.g., A “is larger
than” B).
Unlike geometric maps fused with language features [15]–
[20], 3DSGs provide higher-level abstraction, scale naturally
to large environments [21], and, thanks to detailed node
captions and semantic relationships, have proved useful in
robotics tasks such as planning [5], [22], [23], manipula-
tion [24], [25], and navigation [1], [6].
3DSGs are typically constructed either from dense se-
quences of RGB-D images [3], [21], [26]–[28] or from class-
agnostic segmented point clouds [4], [29]–[32].
In general, all available input is used to construct the graph,
and in some approaches, images that meet specific criteria
–for example, those focusing on a particular pair of objects [1],
[32], [33]– are used to establish spatial edges, while others
are selected to extract functional interactive relationships [7],
[10] (e.g., a button “turns on" a monitor).
In contrast to our approach, all these solutions explicitly
model edges and disregard the RGB input once the graph is
constructed, thereby constraining the 3DSG to the specific
application for which it was built.
To address this problem, the works of [34], [35] attempt
to reason about object primitives at the resolution required
for a specific task by mitigating the information bottleneck.
However, a key limitation of these methods is that even a
slight change in the task requires reconstructing the entiregraph from scratch. In contrast, [36] generates a 3DSG from
a NeRF [37], capturing the necessary relationships between
pairs of objects, but these relationships are restricted to
those modeled by the NeRF at training time. Our method
overcomes both issues by simply rendering new edges from
our keyframes to answer specific queries.
B. Object Localization with 3D Scene Graphs
3D object localization [38], also known as object grounding,
is the task of predicting a target 3D object given a point cloud
and a natural language expression as input. Contrary to other
approaches [39]–[41], 3DSGs do not need to be trained on
ground-truth point clouds, nor do they require fine-tuning of
the LLM itself for object grounding.
In practice, object localization with 3DSGs is often realized
by serializing the graph into JSON [1], [5], [7] and prompting
an LLM to find the target object using the graph structure
and node features. Different works have tried to optimize the
JSON by adding hierarchical structure [24] or by applying
a deductive scene reasoning algorithm [33] that prunes the
search space using spatial relationships. All these methods
are limited to the information contained in the 3DSG, which
is often insufficient to disambiguate complex queries that
include specific edges.
III. TECHNICALAPPROACH
This work aims to build a hierarchical 3D scene graph for
a large-scale environment, consisting of floors, rooms, objects,
and functional elements, where each node is augmented
with multi-modal knowledge extracted from keyframes in
the environment. Given a posed RGB-D image sequence, we
first reconstruct a dense 3D point cloud of the environment
and segment it into floors and rooms (Sec. III-A). Within
each room, we select keyframes that maximize both geometric
coverage and visual informativeness (Sec. III-B). A VLM then
extracts textual descriptions, object categories, and functional
element labels, which guide an open-vocabulary segmentation
pipeline to produce 3D object and functional element seg-
ments (Sec. III-C). An LLM subsequently condenses keyframe
descriptions into room summaries and aggregates them into
floor-level summaries, thereby building contextual information
across multiple levels of abstraction within the 3DSG (Sec. III-
D). Finally, we introduce a multi-modal hierarchical Retrieval-
augmented generation (RAG) pipeline inspired by [42], which
exploits the graph’s topology to support top-down querying,
ensuring that LLMs receive task-relevant information without
exceeding their context window (Sec. III-E). Fig. 2 provides
an overview of our approach.
A. Hierarchical Scene Segmentation
Given a posed RGB-D sequence I={P t, It, Dt}T
t=1of
the environment, where Pt= [R t|tt]∈SE(3) is the camera
pose, Itis the RGB image, and Dtis the depth map, we
first reconstruct a global 3D point cloud Pscene∈RN×3
of the entire scene. Second, we segment the full scene
point cloud Pscene into a set of NFfloor point clouds,
{Fi}NF
i=1. Then we segment each floor point cloud Fiinto a
set of NRiroom point clouds {Rij}NRi
j=1. This hierarchical

A. Hierarchical Scene Segmentation 
 B. Keyframe Sampling 
Dense Poses Sparse Poses 
C. Objects and Functional Elements Segmentation 
VLM 
D. Scene Description Generation 
HEllo 
VLM 
E. Scene Querying and Hierarchical RAG 
👤
 Where is the coffee mug in the kitchen? DBSCAN 
LLM 
LLM 
Posed RGB-D sequence 3D Reconstruction Floor and Room Segmentation 
HEllo 
Frame Descriptions Room Descriptions Floor Descriptions HEllo 
Keyframes 
Keyframes Keyframes Object and Functional 
element Segments 
,
Text Descriptions Text Vector Database 
LLM 
Keyframes Image Vector Database Context 3D Object Masks 
🔍
 id: obj_32 , bbox: [0.23, 7.35, 1.10] 
Merging 
Best view 
selection 
CLIP O.V. 
Detector 
RAG Fig. 2: Overview of KeySG: (A) we first reconstruct the full point cloud of the 3D scene and segment it into floors and rooms; (B) for each room, we
select keyframes that provide geometric coverage of the entire space while maximizing visual information; (C) we leverage VLMs to extract descriptions,
object tags, and functional element tags from the selected keyframes (D) we combine these tags with an open-vocabulary segmentation pipeline to obtain
3D segments of objects and their associated functional elements (E) we employ LLMs to summarize the extracted keyframe descriptions into a dense,
informative room summary, and subsequently aggregate room summaries into a floor-level summary, thereby generating contextual information at increasing
levels of abstraction within the 3DSG. To enable efficient querying of the 3DSG, we introduce a hierarchical retrieval mechanism grounded in RAG. This
mechanism exploits the graph’s structure to perform a top-down search: starting from global, high-level concepts and progressively narrowing to local object
nodes, ensuring that LLMs receive rich task-relevant content without exceeding their context window.
segmentation follows the approach from [6], segmenting
floors by computing a height histogram of the point cloud,
then selecting dominant peaks. For room segmentation, we
compute a 2D histogram from the bird’s-eye view of the floor
and apply the Watershed algorithm.
B. Keyframe Sampling
A key principle of KeySG is to augment graph nodes with
both raw sensory data and semantic context using VLMs.
However, storing and processing the entire data sequence for
a large-scale environment is computationally impractical. To
address this, KeySG employs a down-sampling procedure that
balances computational efficiency with the preservation of
critical spatial and visual information, a problem known from
keyframe-based visual SLAM [43], but we select frames based
on visual room coverage rather than geometric reconstruction
accuracy.
We start by associating each frame from the original
sequence with a specific segmented room Ri. A frame’s
camera pose is assigned to a room if its associated 3D
camera center falls within the room’s volumetric boundary,
i.e.tt∈Vol(R i). This process yields a dense set of poses for
each room Ri, denoted by D′
i={P t∈ I |t t∈Vol(R i)}.
To avoid the corner cases where the frame’s camera pose
lies in one room but is viewing a different room or corridor,
we further filter D′
iby requiring that a significant fractionηof the frame’s back-projected 3D points also fall within
the room’s 2D polygon. The resulting refined set is denoted
byDi. Next, we want to extract a subset Si⊆ Disuch that
|Si| ≪ |D i|. For each pose matrix Pt∈ Diwe extract pose
features as 7D vectors ft= (tt;w·q t)where qt∈R4is the
quaternion derived from its rotation matrix Rottandwis a
scalar rotation weight. The features are then standardized:
˜ft=ft−µ
σ
where µandσare the mean and standard deviation computed
across all pose features. We apply DBSCAN clustering to
the set of standardized features, which yields a set of clusters
Ci. For each cluster ck∈ Ci, we compute the medoid, i.e.
the point f∗
k∈ckthat minimizes the sum of distances to all
other points within that cluster:
f∗
k= arg min
fj∈ckX
ft∈ck∥˜fj−˜ft∥2.(1)
The final set of selected keyframes, Si={f∗
k|k=
0, . . . ,|C i|}, is formed by collecting the medoid from
each cluster. For example, this method subsamples the
scene0011_00 , a kitchen scene from ScanNet [44] dataset,
from 2374 frames down to 23keyframes, with 96.26%
geometric coverage compared to the full dense point cloud.

C. Objects and Functional Elements Segmentation
After obtaining the representative keyframes for each room,
the pipeline proceeds with the 3D segmentation of objects and
their corresponding functional elements. For the keyframes
Siextracted from each room Ri, we first utilize a VLM [45]
to generate sets of object tags Oframe and functional element
tagsFframe for each keyframe. These are then aggregated
across all frames to form the comprehensive sets OiandFi
for the entire room. Next, we use open-vocabulary object
detection and segmentation models [46], [47], guided by the
object tags Oi, to perform 3D segmentation on the dense set
of room frames Di. This yields a set of redundant point clouds
in global coordinates for each object. To consolidate these,
we incrementally merge objects with significant geometric
overlap, following the approach in [6]. Each merged object
incorporates 2D masks from multiple viewpoints. For deriving
a canonical semantic feature, we apply a best-view selection
strategy: each 2D mask is scored based on its size and distance
from the image boundaries, prioritizing large, centrally
located views. The highest-scoring view is then used to
compute a CLIP [48] embedding, ensuring a high-quality
feature representation. Finally, we extract the 3D segments
of associated functional elements using the best view from
each object, leveraging open-vocabulary models [46], [47]
with the functional element tagsF i.
D. Scene Description Generation
In KeySG, we argue that scene details, such as layout,
semantics, object relationships, state, and affordance, are
implicitly stored within the keyframes and their corresponding
text descriptions, alleviating the need to explicitly model
these specific relations as edges in the 3D scene graph. In
this step, we utilize the sparse room keyframes and a vision
language model (VLM) to generate a detailed description of
each frame. To force the VLM to generate a geometrically-
grounded frame description, we determine if an object point
cloud is visible from a keyframe’s perspective, and we
perform a multi-step visibility check. First, object points
are transformed into the camera’s coordinate system. These
points are then projected onto the keyframe 2D image plane,
and any points falling outside the image are culled. For
the remaining points, we perform an occlusion check by
comparing each point’s depth to the corresponding value in
the keyframe depth map. The object is considered visible if
the fraction of its visible points exceeds a minimum threshold
θvis. We feed the VLM with a keyframe image and the list
of objects that are visible in it, resulting in a description that
is geometrically grounded to the 3D objects we found in the
scene. We then aggregate all keyframe descriptions to generate
a comprehensive summary for each room. Subsequently, we
aggregate all room summaries to generate the overall floor
summary.
To construct the final hierarchical 3D scene graph, we
assign each extracted context entity to its appropriate level
based on spatial and semantic containment. Floors form the
top-level nodes, encapsulating aggregated floor summaries.
Within each floor node, room nodes are nested, each contain-ing the segmented room’s point cloud, keyframe multi-modal
data, and room summary. Objects are assigned as child nodes
under their respective rooms, incorporating their 3D merged
point clouds, CLIP embeddings, and associated functional
elements as sub-nodes. The KeySG hierarchy reflects the
environment’s physical organization, and it is semantically
rich, making it applicable for a wide range of tasks.
E. Scene Querying and Hierarchical RAG
In general, 3D scene graphs are often paired with LLMs [1],
[6], [7] to answer user queries. However, serializing the entire
scene graph into a single prompt can overwhelm an LLM’s
context window. Moreover, LLMs’ performance degrades
when processing long contexts, making it harder for the
model to retrieve relevant information. These issues hinder
scalability to large environments. To address this, we design
a hierarchical retrieval-augmented generation (RAG) pipeline
aligned with the structure of KeySG.
This pipeline enables users to query the environment
through KeySG (e.g., “Where is the coffee mug in the
kitchen?") and receive answers grounded in the recovered
3D geometry and semantic data. It also supports indirect
references via attributes, status, or spatial relations to other
objects in the scene.
Our RAG pipeline consists of three main steps.
•We create text chunks from all information extracted
from KeySG keyframes and group them by their graph
level, yielding four chunk types: floor, room, frame,
and object. Each chunk contains text as described in
(Sec. III-D).
•We compute vector embeddings for all chunks and index
them by type, using [45], [49].
•We build a visual vector database over all keyframes in
the scene, as well as a separate database of object-level
CLIP embeddings.
Grounding a target can be challenging for complex queries,
e.g., those that refer indirectly via nearby objects, describe
attributes such as shape or color, or omit the object’s name
entirely. To handle this, we use an LLM to parse the
query into a <target object> and a set of <anchor
objects> , following [33]. We then compute embeddings for
the<target object> and<anchor objects> . Then,
hierarchically retrieve related text chunks, visual keyframes,
and objects via cosine similarity. The resulting multi-modal
context is passed to the LLM to generate the answer.
IV. EXPERIMENTALEVALUATION
In this section, we evaluate KeySG on four different
benchmarks to demonstrate both the quality of the geometric
and semantic data stored in the graph nodes and its abil-
ity to accurately handle complex queries without explicit
relationship modeling.
First, we assess its open-vocabulary 3D semantic segmenta-
tion capabilities against recent methods on the Replica dataset
(Sec. IV-A). Second, we compare it to recent approaches in
functional element segmentation on the FunGraph3D dataset
(Sec. IV-B). Third, we evaluate its ability to retrieve objects

MethodmAcc F-mIoU
MaskCLIP [48] 4.53 0.94
Mask2former [50] + CLIP [51] 10.42 13.11
ConceptFusion [18] 24.16 31.31
ConceptFusion [18] + SAM [47] 31.53 38.70
ConceptGraphs [1] 40.63 35.95
ConceptGraphs-Detector [1] 38.72 35.82
Clio [34] 37.95 36.26
HOV-SG [6] 38.07 40.16
KeySG (ours) 45.81 46.16
TABLE I: Results for open-vocabulary 3D semantic segmentation. We report
the mAcc and F-mIoU metrics (%).
from hierarchical queries in large-scale indoor environments
using the Habitat Matterport 3D Semantic Dataset (Sec. IV-C).
Finally, we examine its capabilities to ground objects from
complex natural language queries that require understanding
of the scene and its objects’ shape, location, color, and
affordance, using the Nr3D dataset (Sec. IV-D).
A. Open-vocabulary 3D Semantic Segmentation
To evaluate the objects’ visual semantic embeddings gen-
erated by KeySG, we utilized scenes office0-office4
androom0-room2 from the Replica dataset [9] to facilitate
comparison with other researchers. We followed the evaluation
protocol from [1]: first, we extracted all semantic class
names and modified them to “an image of {class name}".
Then, we computed the CLIP text embeddings for each class.
Finally, we calculated the cosine similarity between these text
embeddings and the object embeddings in the scene, assigning
each object the class with the maximum similarity. We report
mAcc as the class-mean recall and f-mIOU as frequency-
weighted mean intersection over union (IoU). Tab. I shows
that KeySG surpasses all recent approaches with a notable
margin, demonstrating the effectiveness of extracting the
visual CLIP embedding from the best object view.
B. Functional Elements 3D Segmentation
To evaluate the segmentation performance of functional
elements in KeySG, we use the FunGraph3D dataset [10],
a collection of annotated functional interactive elements
within 3D scenes. We compare KeySG against FunGraph [7]
and OpenFunGraph [10], the only two 3DSG methods
specifically designed for this task. Our primary evaluation
metric is Recall@K (R@K) for K= {1,5,10} across all
scenes in the dataset. A prediction is considered a true
positive if the open-vocabulary class assigned to a segmented
functional element has its embedding ranked within the top-k
closest to the ground-truth class among the available labels,
and if the IoU with the ground-truth segment exceeds a
specified threshold. We report results under different IoU
thresholds (0.0,0.10,0.25) in Tab. II. KeySG outperforms
both FunGraph and OpenFunGraph across all metrics except
Recall@10 with IoU ≥0.0, highlighting that while OpenFun-
Graph tends to detect more functional elements, it is far less
accurate in segmenting their geometry.
C. Object Retrieval From Large-Scale Environment
To evaluate KeySG’s capabilities for retrieving objects
in large-scale multi-floor environments, we used the Habi-tat Matterport 3D Semantic Dataset [11], which contains
scenes with multiple floors and rooms. We used RGB-D se-
quences from scenes 00824, 00829, 00843, 00861,
00862, 00873, 00877, 00890 , as collected in [6].
Following [6], we assessed performance on two query types:
floor-room-object (e.g., “the toilet in the bathroom on the
ground floor,“) and room-object (e.g., “oven in the kitchen,“).
As shown in Tab. III, we compared KeySG against HOV-
SG [6] using two evaluation variants. The metrics reported
are R@K, K= {1,5,10} , where a retrieval is successful if the
target object appears within the top-K results and its IoU
with the ground truth meets or exceeds the specified threshold
(0.0,0.10,0.50).
In the first evaluation variant (without RAG), we re-
lied on an LLM to decompose the hierarchical query
into[<floor>, <room>, <object>] , mirroring HOV-
SG [6]. We then computed CLIP text embeddings for each
parsed concept and hierarchically compared their cosine
similarities with corresponding levels in the scene graph. In
the second variant (with RAG), we forwent the explicit LLM
parsing. Instead, we compared the raw query embedding
directly to scene graph chunks in a hierarchical manner:
first computing cosine similarities with all floor summary
embeddings to select the maximum-scoring floor, then pro-
ceeding to the room level by comparing with room summary
embeddings within that floor to select the room, and finally to
the object level by comparing with object embeddings within
that room to identify the target object. KeySG outperformed
HOV-SG [6] in both variants, with the RAG-based approach
achieving the highest scores despite forgoing explicit query
parsing.
D. Object Grounding from Language Queries on Nr3D
Finally, to evaluate our scene graph and RAG pipeline
on grounding objects from language queries in cluttered
environments, we used the Nr3D dataset [12]. It provides a
diverse set of natural language queries, which are categorized
into eight classes based on how the target object is referenced.
Following the evaluation protocol of [33], we used the scenes
0011_00 ,0030_00 ,0046_00 ,0086_00 ,0222_00 ,
0378_00 ,0389_00 ,0435_00 . In Tab. IV, we report
grounding accuracy at an IoU threshold of 0.1. The results are
shown both overall and categorized by query characteristics,
including the presence of spatial, color, or shape language,
as well as explicit target mentions. KeySG achieves the best
overall score (30.4%) compared to recent 3D scene graphs,
and leads on most subsets, particularly those without color
(38.0%), without shape (38.6%), and with a target mention
(40.1%). This performance highlights KeySG’s strong recall
and robustness to relational cues. Thus, we conclude that
KeySG representation, by implicitly storing rich scene and
object information, can answer a diverse set of user queries on
demand without requiring explicitly modeled relationships.
V. LIMITATIONS
Our framework, while effective, has several limitations.
First, its reliance on computationally expensive large lan-
guage and vision-language models makes graph construction

R@3 R@5 R@10
MethodIoU ≥0.0 IoU≥0.10 IoU≥0.25 IoU≥0.0 IoU≥0.10 IoU≥0.25 IoU≥0.0 IoU≥0.10 IoU≥0.25
OpenFunGraph [10] 45.34 5.39 0.31 47.74 6.89 1.5060.409.30 1.50
FunGraph [7] 33.56 22.03 13.04 35.79 22.93 13.6439.98 24.28 14.30
KeySG (ours) 46.44 24.23 13.33 53.06 25.19 13.6457.12 27.57 14.53
TABLE II: Results for 3D functional elements segmentation. Recall (R) grouped by Top-k and IoU thresholds metrics (%) are reported.
R@1 R@5 R@10
Method Query TypeIoU ≥0.0 IoU≥0.10 IoU≥0.5 IoU≥0.0 IoU≥0.10 IoU≥0.5 IoU≥0.0 IoU≥0.10 IoU≥0.5
HOV-SG [6]Simple (r, o) 23.30 0.60 0.00 44.30 2.00 0.00 55.90 4.50 0.00
Simple (f, r, o) 22.80 0.60 0.00 44.90 0.20 0.00 56.60 4.30 0.00
KeySG w/o RAGSimple (r, o) 32.62 26.50 15.5070.7561.25 40.0583.25 75.5053.00
Simple (f, r, o)35.30 30.3715.80 69.60 61.90 40.3083.1075.00 51.50
KeySG w/ RAGSimple (r, o) 34.00 30.40 20.6068.1062.00 45.9080.80 75.10 58.30
Simple (f, r, o) 32.90 28.50 18.40 68.00 61.00 43.40 79.80 73.10 55.50
TABLE III: Results for hierarchical 3D object retrieval on large-scale environment. We report Recall (R) grouped by Top-k and IoU thresholds metrics (%).
Overall w Spatial Lang. w/o Spatial Lang. w Color Lang. w/o Color Lang. w Shape Lang. w/o Shape Lang. w Target Mention w/o Target Mention
MethodIoU ≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10 IoU≥0.10
OpenFusion [20] 10.7 8.9 22.3 11.8 10.5 9.8 10.9 11.3 4.9
ConceptGraphs [1] 16.0 15.0 22.3 17.6 15.7 10.8 16.9 16.9 6.6
BBQ [33] 28.3 28.1 29.8 25.2 29.0 34.327.3 29.6 14.8
KeySG (ours) 30.4 37.7 37.9 36.2 38.033.4 38.6 40.112.2
TABLE IV: Results for 3D object grounding. We report accuracy at IoU threshold of 0.1 (%) for different query characteristics.
an offline process that requires a pre-reconstructed scene.
However, once the graph is built, it can be deployed on a
robot as a persistent knowledge base that can be efficiently
queried in real time thanks to the RAG pipeline. Second, our
method currently assumes a static environment and does not
handle dynamic objects or changes in object states. These
areas offer clear directions for future research.
VI. CONCLUSION
In this work, we addressed a fundamental limitation of
existing 3D Scene Graphs: relying on a predefined set of
relationship edges that constrain their applicability to tasks
that require a different set of relationships. While 3DSGs
provide a valuable bridge between geometry and semantics,
their rigidity to predefined sets of relationships restricts their
usefulness for reasoning and complex queries in human-
centered environments. We introduced KeyFrame-Based
3DSGs (KeySG), the first framework to extend 3DSGs across
multiple levels of resolution, from buildings to functional
elements. By combining adaptive down-sampling, hierarchical
scene summarization, and a retrieval-augmented generation
mechanism, KeySG efficiently provides task-relevant context
to large language models without exceeding scalability limits.
This integration enables 3D scene graphs to process a
wide range of tasks and queries. Our experiments across
benchmarks in open-vocabulary 3D segmentation, functional
element segmentation, and 3D object grounding demonstrate
that KeySG outperforms prior approaches on most metrics
while maintaining scalability. These results highlight the
promise of augmenting static 3D representations with adaptive,
LLM-driven reasoning mechanisms, paving the way towardpersistent and general-purpose world models for robotics.
REFERENCES
[1]Q. Gu, A. Kuwajerwala, S. Morin, K. Jatavallabhula, B. Sen, A. Agar-
wal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. de Melo,
J. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “Conceptgraphs:
Open-vocabulary 3d scene graphs for perception and planning,”arXiv,
2023.
[2]A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, “3D dynamic
scene graphs: Actionable spatial perception with places, objects, and
humans,”Robotics: Science and Systems, 2020.
[3]N. Hughes, Y . Chang, and L. Carlone, “Hydra: A real-time spatial
perception system for 3D scene graph construction and optimization,”
inRobotics: Science and Systems, 2022.
[4]J. Wald, H. Dhamo, N. Navab, and F. Tombari, “Learning 3D Semantic
Scene Graphs from 3D Indoor Reconstructions,” inCVPR, 2020.
[5]K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. Reid, and N. Suen-
derhauf, “Sayplan: Grounding large language models using 3d scene
graphs for scalable task planning,” inConf. on Robot Learning, 2023.
[6]A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, “Hier-
archical Open-V ocabulary 3D Scene Graphs for Language-Grounded
Robot Navigation,” inProceedings of Robotics: Science and Systems,
Delft, Netherlands, 2024.
[7]D. Rotondi, F. Scaparro, H. Blum, and K. O. Arras, “Fungraph:
Functionality aware 3d scene graphs for language-prompted scene
interaction,”arXiv preprint arXiv:2503.07909, 2025.
[8]C.-Y . Hsieh, Y .-S. Chuang, C.-L. Li, Z. Wang, L. T. Le, A. Kumar,
J. Glass, A. Ratner, C.-Y . Lee, R. Krishnaet al., “Found in the middle:
Calibrating positional attention bias improves long context utilization,”
arXiv preprint arXiv:2406.16008, 2024.
[9]J. Straub, T. Whelan, L. Ma, Y . Chen, E. Wijmans, S. Green, J. J.
Engel, R. Mur-Artal, C. Ren, S. Vermaet al., “The replica dataset:
A digital replica of indoor spaces,”arXiv preprint arXiv:1906.05797,
2019.
[10] C. Zhang, A. Delitzas, F. Wang, R. Zhang, X. Ji, M. Pollefeys, and
F. Engelmann, “Open-V ocabulary Functional 3D Scene Graphs for
Real-World Indoor Spaces,” inIEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2025.
[11] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner,
A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savvaet al.,

“Habitat-matterport 3d semantics dataset,” inProc. of the IEEE Conf. on
Computer Vision and Pattern Recognition, 2023.
[12] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. J. Guibas,
“ReferIt3D: Neural listeners for fine-grained 3d object identification in
real-world scenes,” in16th European Conference on Computer Vision
(ECCV), 2020.
[13] I. Armeni, Z.-Y . He, A. Zamir, J. Gwak, J. Malik, M. Fischer, and
S. Savarese, “3D scene graph: A structure for unified semantics, 3D
space, and camera,” inInt. Conf. on Computer Vision, 2019.
[14] U.-H. Kim, J.-M. Park, T.-J. Song, and J.-H. Kim, “3-d scene graph:
A sparse and semantic representation of physical environments for
intelligent agents,”TOC, 2019.
[15] D. Shah, B. Osi ´nski, S. Levineet al., “Lm-nav: Robotic navigation with
large pre-trained models of language, vision, and action,” inConf. on
Robot Learning. PMLR, 2023, pp. 492–504.
[16] C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language
maps for robot navigation,” inInt. Conf. on Robotics and Automation,
London, UK, 2023.
[17] N. M. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, and A. Szlam,
“Clip-fields: Weakly supervised semantic fields for robotic memory,”
arXiv preprint arXiv: Arxiv-2210.05663, 2022.
[18] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen,
S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewariet al., “Conceptfusion:
Open-set multimodal 3d mapping,”RSS, 2023.
[19] O. Mees, J. Borja-Diaz, and W. Burgard, “Grounding language with
visual affordances over unstructured data,” inICRA, 2023.
[20] K. Yamazaki, T. Hanyu, K. V o, T. Pham, M. Tran, G. Doretto,
A. Nguyen, and N. Le, “Open-fusion: Real-time open-vocabulary 3d
mapping and queryable scene representation,” inICRA, 2024.
[21] N. Hughes, Y . Chang, S. Hu, R. Talak, R. Abdulhai, J. Strader, and
L. Carlone, “Foundations of spatial perception for robotics: Hierarchical
representations and real-time systems,”IJRR, 2024.
[22] C. Agia, K. M. Jatavallabhula, M. Khodeir, O. Miksik, V . Vineet,
M. Mukadam, L. Paull, and F. Shkurti, “Taskography: Evaluating robot
task planning over large 3d scene graphs,”CoRL, 2022.
[23] Y . Liu, L. Palmieri, S. Koch, I. Georgievski, and M. Aiello, “Delta:
Decomposed efficient long-term robot task planning using large
language models,”ICRA, 2025.
[24] D. Honerkamp, M. Büchner, F. Despinoy, T. Welschehold, and
A. Valada, “Language-grounded dynamic scene graphs for interactive
object search with mobile manipulation,”RA-L, 2024.
[25] Z. Yan, S. Li, Z. Wang, L. Wu, H. Wang, J. Zhu, L. Chen, and J. Liu,
“Dynamic open-vocabulary 3d scene graphs for long-term language-
guided mobile manipulation,”RAL, 2025.
[26] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, “3d dynamic
scene graphs: Actionable spatial perception with places, objects, and
humans,”RSS, 2020.
[27] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y . Chang, J. Shi,
A. Gupta, and L. Carlone, “Kimera: From slam to spatial perception
with 3d dynamic scene graphs,”IJRR, 2021.
[28] Y . Chang, N. Hughes, A. Ray, and L. Carlone, “Hydra-multi: Collabo-
rative online construction of 3d scene graphs with multi-robot teams,”
IROS, 2023.
[29] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, “Scenegraph-
fusion: Incremental 3d scene graph prediction from rgb-d sequences,”
inCVPR, 2021.
[30] S. Zhang, A. Hao, H. Qinet al., “Knowledge-inspired 3d scene graph
prediction in point cloud,”NeurIPS, 2021.
[31] Z. Wang, B. Cheng, L. Zhao, D. Xu, Y . Tang, and L. Sheng, “Vl-
sat: Visual-linguistic semantics assisted training for 3d semantic scene
graph prediction in point cloud,” inCVPR, 2023.
[32] S. Koch, N. Vaskevicius, M. Colosi, P. Hermosilla, and T. Ropinski,
“Open3dsg: Open-vocabulary 3d scene graphs from point clouds with
queryable objects and open-set relationships,” inCVPR, 2024.
[33] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, D. Yudin,
M. Monastyrny, and A. Valenkov, “Beyond bare queries: Open-
vocabulary object grounding with 3d scene graph,”ICRA, 2025.
[34] D. Maggio, Y . Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty,
E. Cristofalo, L. Schmid, and L. Carlone, “Clio: Real-time task-driven
open-set 3d scene graphs,”RA-L, 2024.
[35] Y . Chang, L. Fermoselle, D. Ta, B. Bucher, L. Carlone, and J. Wang,
“vs-graphs: Integrating visual slam and situational graphs through multi-
level scene understanding,” inProc. of the IEEE Conf. on Computer
Vision and Pattern Recognition, 2025.
[36] S. Koch, J. Wald, M. Colosi, N. Vaskevicius, P. Hermosilla, F. Tombari,
and T. Ropinski, “Relationfield: Relate anything in radiance fields,” inIEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2025.
[37] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng, “Nerf: Representing scenes as neural radiance fields
for view synthesis,” inECCV, 2020.
[38] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object
localization in rgb-d scans using natural language,”16th European
Conference on Computer Vision (ECCV), 2020.
[39] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Language
conditioned spatial relation reasoning for 3d object grounding,” in
NeurIPS, 2022.
[40] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff,
S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud, K. Yadav, Q. Li, B. New-
man, M. Sharma, V . Berges, S. Zhang, P. Agrawal, Y . Bisk, D. Batra,
M. Kalakrishnan, F. Meier, C. Paxton, S. Sax, and A. Rajeswaran,
“Openeqa: Embodied question answering in the era of foundation
models,” inConference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[41] H. Huang, Y . Chen, Z. Wang, R. Huang, R. Xu, T. Wang, L. Liu,
X. Cheng, Y . Zhao, J. Panget al., “Chat-scene: Bridging 3d scene
and large language models with object identifiers,”Proceedings of the
Advances in Neural Information Processing Systems, Vancouver, BC,
Canada, 2024.
[42] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[43] G. Younes, D. Asmar, E. Shammas, and J. Zelek, “Keyframe-based
monocular slam: design, survey, and future directions,”Robotics and
Autonomous Systems, 2017.
[44] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor
scenes,”CVPR, 2017.
[45] OpenAI, “GPT-4 technical report,”CoRR, vol. abs/2303.08774, 2023.
[46] S. Fu, Q. Yang, Q. Mo, J. Yan, X. Wei, J. Meng, X. Xie, and W.-S.
Zheng, “Llmdet: Learning strong open-vocabulary object detectors
under the supervision of large language models,”arXiv preprint
arXiv:2501.18954, 2025.
[47] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y . Loet al., “Segment anything,”
inCVPR, 2023.
[48] X. Dong, J. Bao, Y . Zheng, T. Zhang, D. Chen, H. Yang, M. Zeng,
W. Zhang, L. Yuan, D. Chenet al., “Maskclip: Masked self-distillation
advances contrastive language-image pretraining,” inProceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2023, pp. 10 995–11 005.
[49] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré,
M. Lomeli, L. Hosseini, and H. Jégou, “The faiss library,”arXiv, 2024.
[50] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar,
“Masked-attention mask transformer for universal image segmentation,”
CVPR, 2022.
[51] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clarket al., “Learning transferable
visual models from natural language supervision,” inICML, 2021.