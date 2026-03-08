# SGR3 Model: Scene Graph Retrieval-Reasoning Model in 3D

**Authors**: Zirui Wang, Ruiping Liu, Yufan Chen, Junwei Zheng, Weijia Fan, Kunyu Peng, Di Wen, Jiale Wei, Jiaming Zhang, Rainer Stiefelhagen

**Published**: 2026-03-04 21:19:54

**PDF URL**: [https://arxiv.org/pdf/2603.04614v1](https://arxiv.org/pdf/2603.04614v1)

## Abstract
3D scene graphs provide a structured representation of object entities and their relationships, enabling high-level interpretation and reasoning for robots while remaining intuitively understandable to humans. Existing approaches for 3D scene graph generation typically combine scene reconstruction with graph neural networks (GNNs). However, such pipelines require multi-modal data that may not always be available, and their reliance on heuristic graph construction can constrain the prediction of relationship triplets. In this work, we introduce a Scene Graph Retrieval-Reasoning Model in 3D (SGR3 Model), a training-free framework that leverages multi-modal large language models (MLLMs) with retrieval-augmented generation (RAG) for semantic scene graph generation. SGR3 Model bypasses the need for explicit 3D reconstruction. Instead, it enhances relational reasoning by incorporating semantically aligned scene graphs retrieved via a ColPali-style cross-modal framework. To improve retrieval robustness, we further introduce a weighted patch-level similarity selection mechanism that mitigates the negative impact of blurry or semantically uninformative regions. Experiments demonstrate that SGR3 Model achieves competitive performance compared to training-free baselines and on par with GNN-based expert models. Moreover, an ablation study on the retrieval module and knowledge base scale reveals that retrieved external information is explicitly integrated into the token generation process, rather than being implicitly internalized through abstraction.

## Full Text


<!-- PDF content starts -->

SGR3Model: Scene Graph Retrieval-Reasoning Model in 3D
Zirui Wang1∗, Ruiping Liu1∗†, Yufan Chen1‡, Junwei Zheng1, Weijia Fan2,
Kunyu Peng1, Di Wen1, Jiale Wei1, Jiaming Zhang3and Rainer Stiefelhagen1
Abstract— 3D scene graphs provide a structured representa-
tion of object entities and their relationships, enabling high-level
interpretation and reasoning for robots while remaining intu-
itively understandable to humans. Existing approaches for 3D
scene graph generation typically combine scene reconstruction
with graph neural networks (GNNs). However, such pipelines
require multi-modal data that may not always be available, and
their reliance on heuristic graph construction can constrain the
prediction of relationship triplets. In this work, we introduce a
Scene Graph Retrieval-Reasoning Model in 3D (SGR3Model),
a training-free framework that leverages multi-modal large
language models (MLLMs) with retrieval-augmented genera-
tion (RAG) for semantic scene graph generation.SGR3Model
bypasses the need for explicit 3D reconstruction. Instead, it
enhances relational reasoning by incorporating semantically
aligned scene graphs retrieved via a ColPali-style cross-modal
framework. To improve retrieval robustness, we further in-
troduce a weighted patch-level similarity selection mechanism
that mitigates the negative impact of blurry or semantically
uninformative regions. Experiments demonstrate thatSGR3
Modelachieves competitive performance compared to training-
free baselines and on par with GNN-based expert models.
Moreover, an ablation study on the retrieval module and
knowledge base scale reveals that retrieved external information
is explicitly integrated into the token generation process, rather
than being implicitly internalized through abstraction.
I. INTRODUCTION
3D scene understanding requires the extraction of object
attributes and relationships and their organization into an
abstract, graph-based representation. Such representations
support a wide range of downstream tasks, including robot
manipulation and navigation [1], [2], [3], [4], and provide an
accessible bridge between visual perception and symbolic
reasoning. In a 3D scene graph, objects are represented
as nodes and semantic relationships are represented as
edges; this relational structure is often the key to strong
performance in the aforementioned tasks, especially when
an agent must provide human-understandable spatial de-
scriptions. Such representations also enhance interpretability
and facilitate modular reasoning. A common form of 3D
scene graphs is hierarchical, typically following a top-down
taxonomy from buildings to rooms, zones, and objects [5],
[6], [7], [1], [8]. These graphs primarily encode geometric
*indicates equal contribution.
†project lead.
‡corresponding author.
1Zirui Wang, Ruiping Liu, Yufan Chen, Junwei Zheng, Kunyu Peng,
Di Wen, Jiale Wei and Rainer Stiefelhagen are with Karlsruhe Institute of
Technology, 76131 Karlsruhe, Germany.first.last@kit.edu
2Weijia Fan is with Shenzhen University, Shenzhen, China.
wakinghours.szu@outlook.com
3Jiaming Zhang is with Hunan University, Changsha, China.
jiamingzhang@hnu.edu.cn
Fig. 1:Comparison of SGR3Model and reconstruction-based models.
The requirements of the reconstruction-based models include RGB images,
depth information, camera poses, extrinsics, and intrinsics, whereas SGR3
Model requires only RGB images with information from an external
knowledge base. Reconstruction-based pipelines often depend on geometric
proximity heuristics to define candidate edges, thereby constraining relation
modeling to spatially local interactions.
properties and spatial organization. The multi-level taxonomy
facilitates cross-scale contextual reasoning by combining
global context with local geometry, but it does not always
provide explicit semantic relations. In contrast, a second line
of work focuses on predicting semantic relationship triplets
(subject-predicate-object) on top of reconstructed geometry
[9], [10], [11], [12], [13]. These methods typically rely on
object proposals and heuristic graph construction to define
candidate edges, most of which depend on spatial distance.
A GNN then refines node and edge representations for
relationship classification.
Despite strong progress, two limitations remain. First,
reconstruction-centric pipelines impose heavy requirements
on sensor data (e.g., RGB-D sequences, accurate camera
poses, and clean meshes), which may not be available
in practical deployments. Second, relationship prediction
is challenging under long-tailed predicate distributions and
ambiguous geometry; heuristic candidate generation and
purely geometric cues can further amplify these issues.
Recently, vision-language models and large language models
have shown strong semantic priors, motivating training-free
alternatives that bypass explicit 3D reasoning and instead
condition relation prediction on image evidence and language
context [14], [15].
In this paper, we study training-free 3D scene graph gen-
eration using an MLLM equipped with retrieval-augmented
generation (RAG) [16]. The retrieval module ofSGR3Model
follows the design principles of ColPali [17], operating onarXiv:2603.04614v1  [cs.CV]  4 Mar 2026

patch-level embeddings to retrieve structurally aligned scene
graphs from an external knowledge base. The retrieved scene
graphs are then used as structured prompts to guide the gen-
eration of relationship triplets. To ensure a stable inference,
we introduce a ColQwen-based key-frame filtering mecha-
nism that suppresses visually redundant observations before
retrieval. To make retrieval robust to low-quality regions, we
introduce weighted patch-level voting, which emphasizes se-
mantically informative patches while down-weighting blurry
or visually uninformative areas. We evaluate our approach
on 3RScan and perform ablations with respect to retrieval
granularity and knowledge-base scale to characterize how
retrieved information influences generation. A comparison
betweenSGR3Modeland traditional reconstruction-based
metrics is shown in Fig. 1.
The main contributions of our work are as follows:
•We propose a training-free 3D scene graph generation
framework via MLLM without explicit reconstruction
with camera poses.
•We introduce a ColPali-style retrieval pipeline with
weighted voting for robust reference selection.
•Our experiments show thatSGR3Model outperforms
other training-free frameworks and performs on par with
GNN-based models.
II. RELATED WORK
A. 3D Scene Graph Generation
The concept of 3D scene graphs was first introduced in
a hierarchical representation built upon reconstructed 3D
meshes and scene panoramas [5]. Since then, substantial
progress has been made in 3D scene graph generation. In par-
ticular, the 3RScan dataset [18] has become a foundational
benchmark, as it provides explicit semantic annotations for
relationship triplets. Building upon this dataset, several works
have explored incremental 3D scene and graph reconstruction
[9], [11], [10]. These approaches extract primitive entities
from RGB-D or RGB sequences and construct dynamic
neighbor graphs to encode spatial proximity. An online
fusion framework incrementally integrates local sub-map
predictions into a globally consistent semantic graph, where
GNN-based local updates are asynchronously merged into a
persistent global model. Despite these advances, reasoning
over ambiguous or long-tailed relationships remains chal-
lenging due to the limited semantic richness of 3D geometry.
To alleviate this issue, VL-SAT [19] introduces a visual-
linguistic oracle model to mitigate the long-tail distribu-
tion of relationship triplets. Furthermore, HE-3DSGR [20]
enhances incremental reasoning by incorporating historical
context . Specifically, it employs a recurrent mechanism
with a one-hot candidate matrix to capture both global and
local historical dependencies, thereby improving robustness
in relationship prediction.
To accommodate the shift toward more flexible architec-
tures, Open3DSG [14] leverages 2D vision-language models
and LLMs to enable zero-shot 3D scene graph generation,
thereby extending label-constrained relationship triplets toopen-vocabulary generation. In a similar vein, Concept-
Graphs [15] adopts a modular design by employing LLM to
process object captions and geometric relationships, build-
ing an open-vocabulary map for robotic planning. Build-
ing upon these ideas, 3DGraphLLM [21] further advances
the paradigm by directly feeding flattened scene graphs
into LLMs, enabling complex spatio-temporal reasoning and
dialogue-driven interaction.
In our work, theSGR3Modelleverages MLLMs as the pri-
mary reasoning backbone, rather than relying on specialized
modular components, in a manner similar to OpenWorldSG
[22]. However, unlike OpenWorldSG, our framework assigns
a more central role to the MLLM, which is responsible
for both semantic reasoning and graph-structure generation.
Since our objective is to generate a semantic 3D scene
graph and conduct analysis entirely through the MLLM,
the overall architecture omits explicit 3D reconstruction
modules and heuristic-constrained graph neural networks.
This design enables a more flexible definition of object pairs
and relationship triplets during inference.
B. Retrieval Augmented Generation
LLMs face inherent limitations due to their reliance on
static, non-real-time training data. As a promising solution
to this challenge, retrieval-augmented generation (RAG) en-
hances the relevance and timeliness of model responses by
incorporating external knowledge during inference [16]. To
support effective retrieval, numerous techniques have been
proposed for document indexing, query generation, and re-
trieval optimization [23], [24], [25]. These methods compute
alignment or similarity between a given query and entries in
a knowledge base, using the retrieved results to refine and
condition the generated answers. Extending this paradigm
beyond textual data, Video-RAG [26] has recently been
introduced to selectively retrieve relevant video frames for
query responses, demonstrating the adaptability of retrieval-
augmented frameworks across modalities.
ColPali [17] introduces a vision encoder to enable efficient
retrieval over visually rich documents. Instead of relying on
computationally expensive text chunking, ColPali directly
encodes document pages from their image representations,
and document indexing is performed purely based on visual
features. Similarly, VisRAG [27] represents each document
page as a single 2304-dimensional embedding vector, making
it well-suited for large-scale retrieval across millions of doc-
uments. To extend retrieval beyond purely textual or visual
modalities, fused-modal retrieval pipelines have been pro-
posed to support cross-modal generation. For example, after
image–caption contrastive training, the text-retrieval model
T5-ANCE, enabling a unified encoder to produce modality-
agnostic embeddings [28], and ViDoRAG handles multi-
modal retrieval based on a Gaussian Mixture Model [29].
The integration of RAG broadens the applicability of scene
graph generation to a wider range of tasks. INHerit-SG
[30] treats scene graphs as RAG-ready knowledge bases by
anchoring natural-language descriptions to hierarchical graph
structures, enabling a dual process of graph construction

Fig. 2: (Top) Traditional scene graph generation based on 3D reconstruction and GNN. (Bottom) Our training-free 3D scene graph generation pipeline. We
split the complete RGB sequence into RGB windows containing several consecutive frames. Through ColQwen retrieval, we determine whether a single
frame is a key frame whose visual content is not in the processed buffer. The RAG component performs patch embedding and knowledge base search.
and high-level description generation and retrieval. Similarly,
the SGG-RAG framework [31] formalizes 3D scene graphs
as explicit external memory for LLMs. Its graph-guided
retrieval mechanism provides concise structural evidence to
support open-world reasoning in complex environments.
Unlike existing RAG-based frameworks that primarily
treat scene graphs as external knowledge bases for down-
stream task planning,SGR3Modelintroduces a graph-to-
graph retrieval paradigm. Specifically, we leverage a library
of completed scene graphs to retrieve structurally relevant
relationship triplets via RAG, which support the generation
of the current scene graph. Furthermore, we systematically
analyze how the retrieved triplets influence the accuracy and
semantic coherence of the predicted relationship predicates.
III. METHODOLOGY
The overall pipeline of our framework is illustrated in
Fig. 2 bottom with a traditional GNN-based approach shown
in Fig. 2 top for comparison. We first describe the con-
struction of the external knowledge base and introduce the
key-frame filtering module. During inference, consecutive
RGB frames are grouped into sliding windows to balance
temporal context and input token limitations of the MLLMs.
For each window, we perform RAG to identify structurally
relevant triplets from visually similar scenes in the knowl-
edge base. These retrieved triplets are incorporated into the
prompt alongside the window’s frames, enabling the model to
generate the corresponding scene graph in a single inference
step. This unified inference simultaneously handles object
recognition and relationship prediction for the entire window.
A. External Knowledge Base Building
We build the external knowledge base from the 3RScan
dataset [18]. Each annotated 3D scene graph is decomposedinto frame-level subgraphs, establishing a direct correspon-
dence between individual RGB frames and their seman-
tic structures. We sample image patches from the training
scenes, as defined in 3DSSG [10], and embed each patch by
the SigLip2 model [32] into a 768-dimensional dense feature
vector. All patch embeddings are aggregated into a large-
scale repository, which serves as our retrieval database. We
index these vectors using FAISS [33] to enable efficient ap-
proximate nearest-neighbor search. During inference, query
frames are likewise decomposed into patches and embedded.
For each query patch, we retrieve its top-knearest neighbors
from the indexed base patches.
B. Key Frame Images Filtering
Recent studies [34], [35] have demonstrated that MLLMs
exhibit limited spatial reasoning capabilities. In particular,
when processing consecutive frames, the model often fails
to recognize that an object seen in one frame has already
been processed [34], leading to repeated detections of the
same physical object. This results in duplicate object nodes
and local inconsistencies in the generated graph. To address
this, the SGR3model includes a retrieval-based key-frame
filtering module withColQwen, a Qwen-based variant of
ColPaLi [36], as a similarity evaluation module.
ColQwencompares each incoming frame to a continuously
maintained image buffer. Redundancy is therefore evaluated
with respect to the entire accumulated context rather than
only temporally adjacent frames. In addition, skipping visu-
ally redundant frames reduces unnecessary repeated genera-
tion and accelerates the overall inference process. To quantify
visual similarity, we compareqagainst every buffered frame
b∈ Band compute a token-wise matching score rather than
using a single global embedding. LetV q={vq
i}Tq
i=1and
Vb={vb
j}Tb
j=1denote the ColQwen token embeddings ofq

Fig. 3:Retrieval process for reference edge selection.
andb, respectively. Following ColPali-style late interaction,
the similarity between two frames is defined as
Sim(q, b) =1
TqTqX
i=1max
jvq
i·vb
j.
This score aggregates maximum token-to-token similari-
ties, capturing fine-grained local overlap between frames and
remaining robust to partial viewpoint changes.
Empirically, frames with substantial visual overlap tend
to produce higher similarity scores than frames with distinct
viewpoints within the same scene. This observation allows
a fixed thresholdσ= 0.5to serve as a practical decision
boundary for redundancy filtering.
For each incoming frameq, we compute its maxi-
mum similarity to the buffered frames, defined ass(q) =
max b∈BSim(q, b). Ifs(q)> σ, the frame is regarded as
visually redundant and discarded; otherwise, it is retained as
a new key frame. All retained key frames are appended to
the bufferB, which is incrementally updated throughout the
entire scan.
C. Retrieval for Reference Edges
Given a set of retained frames within the current win-
dowW, each frame is decomposed into normalized patch
embeddings. As illustrated in Fig. 3, we adopt a scene-
level aggregation strategy to obtain a coherent reference. Let
{qi}P
i=1be thePpatch embeddings of a query frameI q. For
each patchq i, we retrieve itsknearest neighbor embeddings
{ki,r}k
r=1from the knowledge base, with cosine similarity
scores
si,r=⟨q i,ki,r⟩,(1)where bothq iandk i,rareℓ 2-normalized embeddings, and
⟨·,·⟩therefore equals cosine similarity. For each candidate
base framefin scenes, we define
as,f[i] = max
k∈K(s,f)⟨qi,k⟩,(2)
whereK(s, f)is the set of retrieved patch embeddings
associated with frame(s, f)in the database. To mitigate the
effects of motion blur and repetitive structure, we weight
each patch by its uniqueness. We compute the patch self-
similarity matrixS∈RP×PwithS ij=⟨q i,qj⟩. Let
µi=1
P−1X
j̸=iSij.(3)
be the mean correlation of patchiwith all other patches. We
then define the normalized weight
wi=exp(−µ i/τ)PP
t=1exp(−µ t/τ),(4)
whereτis a scaling parameter, set to0.1in our experiments.
Intuitively, patches with higher average similarity,i.e., less
unique content, receive lower weight.
With such weights, the similarity between the query frame
Iqand a candidate frame(s, f)is as follows:
Score(I q, Is,f) =X
i∈Ωs,fwias,f[i],(5)
whereΩ s,fdenotes the set of query patches that retrieve
at least one match in frame(s, f). We then aggregate scores
over all query frames to compute a scene-level score and
select the scenes∗with the highest score:
s∗= arg max
sX
q∈Wmax
f∈F(s)Score(I q, Is,f),(6)
whereF(s)is the set of frames in scenes. Within scenes∗,
we pick the top-ranked frames and merge their scene graphs
to obtain the reference edge setE ref, removing duplicate
edges. This set of edges is provided as a structured relational
prior for the subsequent scene graph generation.
D. Window-level 3D Scene Graph Generation
In the final step, the MLLM is prompted with the key-
frame images, the retrieved reference edgesE refand the
current global scene graph. The prompt instructs the model
to match object instances across frames, detect emergent
objects, and infer relationships among all objects. These
instructions are formatted sequentially in the input. The
MLLM then outputs the scene graph for the window, which
is merged into the global scene graph. If no frames remain
after filtering, inference for that window is skipped.
IV. EXPERIMENT
A. Implementation Details
We evaluateSGR3Modelon the 3RScan dataset [18],
which provides 3D scene graphs aligned with reconstructed
3D scenes and relationship triplet annotations. Quantitative
results are reported on this dataset. In addition, we use the

ScanNet dataset [37], an indoor dataset with object labels as
ground truth, for qualitative analysis and visualization. We
employ Qwen3-VL 32B [38] for inference. All experiments
were conducted on four NVIDIA H100 GPUs, each equipped
with 80GB of memory.
B. Evaluation Metrics
The evaluation protocol for scene graph generation follows
the standard metrics defined in [39] and adopted in [40].
Specifically, recall metrics are reported for both predicate
detection and relationship detection. Since our framework
focuses on semantic-only generation, ensuring object-node
consistency is a prerequisite for reliable evaluation.
In practice, we implement the consistency assessment
using Qwen3-VL 32B [38]. Ground-truth bounding boxes are
provided to extract sufficient visual context, while predicted
object candidates whose textual descriptions are generated
during inference are incorporated as input to the MLLM-
based consistency judge. Object occurrences across frames
are jointly considered to determine matching recall.
In the scene graph setting, only object pairs that are
determined to have a relationship are included in the eval-
uation. The predicate detection evaluation is defined as
follows: given the object classes and boxes, we predict only
the predicate. Relationship evaluation requires the correct
identification of object nodes, object pairs, and predicate
labels. Following the official evaluation protocol of 3DSSG
[10], two variants of relationship recall are defined. The first
uses all object pairs determined by heuristic rules before
GNN inference as the denominator (referred to as the old
recall), while the second uses all ground-truth object pairs
as the denominator (referred to as the new recall). In Tab. I,
we report both variants for completeness. For the subsequent
ablation studies, we report only the new recall and refer to
it simply as relationship recall.
C. Comparison with Other Methods
In this work, we compare our proposed 3D scene graph
generation with representative supervised RNN- or GNN-
based expert models and training-free frameworks. The su-
pervised baselines include VGfM [41], 3DSSG [10], SGFN
[9], MonoSSG [11] and VLSAT [19]. The training-free
methods include ConceptGraph [15] and OpenWorld [22].
For OpenWorld, we employ GroundingDINO [42] for entity
detection. The quantitative results are shown in Tab. I. It
is worth noting that the overall values of the new-type
relationship recall remain relatively low across all methods.
This phenomenon is consistent for both geometry-based
pipelines and MLLM-based approaches. In general, GNN-
based methods achieve stronger performance, as geometric
point cloud representations and entity fusion enable more
precise node construction, thereby providing an inherent
advantage over training-free methods.
SGR3Modeldemonstrates competitive performance in
relationship triplet prediction under both evaluation settings,
although it slightly lags behind MonoSSG [11]. Its overallTABLE I. Evaluation on 3RScan. We report object recall@10 (R@10),
predicate recall@3 (R@3), and relationship triplet recall@1 (R@1) under
old and new types of denominators.
MethodObject Predicate Relationship
R@10 R@3 Old R@1 New R@1
Fully supervised
VGfM [41] 0.77 0.36 0.63 0.06
3DSSG [10] 0.74 0.94 0.59 0.070
SGFN [9] 0.80 0.82 0.59 0.074
MonoSSG [11]0.890.870.62 0.131
VLSAT [19] 0.860.980.54 0.087
Training-free
ConceptGraph [15] 0.750.960.55 0.084
OpenWorld [22] 0.46 0.10 0.27 0.043
Only Qwen0.780.56 0.57 0.064
Abstraction 0.65 0.59 0.59 0.096
SGR3Model (Ours) 0.67 0.780.62 0.125
TABLE II. Ablation on ColQwen. Obj Rec means object recall. Rel Rec
means relationship triplet recall.
Method Obj Rec Rel Rec Inference Time Redundancy
filter 0.67 0.1252.73s 1.42
w/o filter0.80 0.1316.18s 4.18
capability in semantic relationship reasoning is strong. With-
out predefined object pairs constrained by heuristics, triplet
prediction becomes more flexible. At the same time, object
detection and grounding remain challenging when relying
solely on an MLLM.
It is important to emphasize that our objective is not to
compete for state-of-the-art performance against expert or
other training-free models. Instead, this work aims to sys-
tematically investigate whether RAG can enhance semantic
reasoning in 3D scene graph generation under a training-free
setting. From this perspective, the experimental results are
encouraging, indicating that incorporating structured external
knowledge through RAG is a feasible and effective strategy
for improving semantic relationship prediction.
D. Ablation Study
In Tab. I, we additionally report an end-to-end MLLM-
based scene graph generation under theOnly Qwensetting.
The result indicates that, except for object detection, our
RAG-enhanced method,SGR3Model, achieves better overall
performance in relationship prediction. To further analyze
the contribution of individual components, we ablate the
ColQwen key-frame filtering module, vary the size of the
external knowledge base, and change the retrieval granularity
for RAG.
Tab. II shows the impact of employing ColQwen during
inference. Redundancy quantifies the degree of duplication
of object nodes in the generated scene graph. In practice,
the same physical object may appear in multiple frames.
If the system repeatedly instantiates these occurrences as
separate nodes instead of merging them into a single entity,

Fig. 4:Visualization of a 3D scene graph generated by theSGR3Model on ScanNet.Red dotted lines indicate incorrect predictions.
TABLE III. Ablation on the scale of the external knowledge base.
Scale Obj Rec Obj mRec Rel Rec Rel mRec
100% 0.67 0.750.125 0.239
75% 0.67 0.76 0.121 0.207
50% 0.64 0.76 0.117 0.238
25% 0.66 0.76 0.110 0.176
0% 0.66 0.75 0.061 0.089
redundancy increases. At the expense of slightly lower object
and relationship recall, it achieves much faster and cleaner
scene graph generation with ColQwen.
Tab. III presents the effects of varying the scale of the
external knowledge base. A gradual reduction in relationship
prediction performance is observed as the knowledge base
shrinks. However, this decline becomes substantial only
when the knowledge base is entirely removed, indicating
that retrieval provides essential relational priors beyond what
the MLLM can reliably infer from visual inputs alone.
The relatively stable performance between 25% and 100%
suggests that once sufficient structured reference information
is available, most relational reasoning capability is recovered.
Further increasing the scale of the knowledge base yields
only marginal improvements. This pattern implies that RAG
primarily contributes useful relational priors rather than
depending on exhaustive coverage. We further examine this
hypothesis in the subsequent analysis of reference absorption
and predicate-level gains.
To investigate the effectiveness of weighted score during
retrieval, we compare it with standard MaxSim patch-level
voting and image-level voting. The results in Tab. IV in-
dicate that finer-grained patch-level retrieval yields better
performance on relationship prediction, and incorporating
uniqueness-aware patch weighting further improves gener-
ation quality.
E. Inference on ScanNet
We select several scenes from ScanNet [37] for qualitative
evaluation and visualization.SGR3Modelis applied to these
scenes to generate semantic scene graphs without additional
supervision. One representative scene, including the gener-
ated object nodes and relationship edges produced by ourTABLE IV. Performance under different retrieval granularities.
Granularities Obj Rec Rel Rec Redundancy
Weighted patch-level0.67 0.125 1.42
Patch-level 0.62 0.117 1.44
Image-level 0.63 0.095 1.49
training-free pipeline, is visualized in Fig. 4.
F . Research on the Mechanism of MLLM with RAG
We first investigated whether summarizing the retrieved
reference triplets into high-level predicate usage instruc-
tions could better guide the generation model. Specifically,
we used an additional LLM step to abstract the retrieved
relationships into generalized patterns of predicate usage
and then provide these abstractions rather than raw triplets.
However, as shown in Tab. I (rowAbstraction), this strategy
does not improve performance. Relationship recall decreases
from 0.125 to 0.096, suggesting that the generation model
benefits more from concrete structural examples than from
abstracted predicate instructions.
Raw triplets preserve explicit object-pair configurations
and spatial co-occurrence patterns, which can serve as im-
plicit structural templates during generation. In contrast,
high-level abstractions compress such structural information
into linguistic summaries, potentially weakening their guid-
ance effect. These observations suggest that MLLMs with
RAG tend to function more as providers of structural priors
than as semantic rule learners.
To further explore the mechanism of processing aug-
mented information in MLLM, we analyze the relationship
triplet gain brought by RAG qualitatively. Three relationship
triplet sets are defined as follows: triplets correctly predicted
under RAGHw
s=H s(RAG), triplets correctly predicted
without RAGHwo
s=H s(NoRAG)and triplets appear in
retrieved reference edgesEref
s= Ref s, we define the gained
triplets brought by RAG as
Hgain
s=Hw
s\ Hwo
s.
We then calculate the overlap between the gained triplets and
the retrieved reference edgesEref
s= Ref sas
Hcopy
s=Hgain
s∩ Eref
s.

Fig. 5:Attention Distribution when generating two predicates ‘sup-
ported by’ for triplets.Red vertical lines indicate the reference triplets
span. Colors indicate relative attention strength. Several tokens within
the span receive noticeable attention, top-2 corresponding tokens for each
predicate are ‘supported’,‘:’ and ‘supported’, ‘supported’.
The copy ratio, which can be seen as explicit usage of
reference triplets, is defined as
ρs=|Hcopy
s|
|Hgain
s|.
Analogously, we compute an object-pair copy ratio by mea-
suring the overlap between object pairs inHgain
s and those
appearing in retrieved reference edges. The result shows that
ρs= 64.7%, and the object pair copy ratio is 71%, indicating
that a substantial portion of newly gained triplets under RAG
can be directly associated with reference triplets, suggesting
that performance improvements largely stem from explicit
structural information provided by RAG rather than implicit
generalization.
For interpretability, we analyze the attention distribution
during predicate generation by extracting the last-layer cross-
attention between generated predicate tokens and the refer-
ence relationship part in the prompt. Although the most high-
lighted tokens do not necessarily correspond to an identical
predicate, the concentration of attention within the reference
token span suggests that retrieved information influences the
generation process. An illustrative example of this attention
behavior is shown in Fig. 5.
V. CONCLUSION
In this work, we presentedSGR3Model, a training-
free framework that integrates RAG with MLLM for 3D
scene graph generation. We investigated the application and
underlying mechanism of RAG in scene graph inference
without relying on additional geometric cues such as camera
poses or depth images. Unlike GNN-based methods that
depend on heuristic graph construction,SGR3Modeldoes not
constrain the MLLM with structural assumptions, enabling
more flexible triplet prediction. Experimental results prove
thatSGR3Modelimproves the prediction of both object
pairs and relationship triplets compared to other training-free
methods and on par with GNN-based expert models. Further
ablation studies reveal that structural reference information
retrieved from an external knowledge base is explicitly used
during generation. This utilization is primarily reflected in
the alignment and reuse of specific relationship structures,
rather than in deep semantic fusion or intrinsic structural
reasoning. Overall, theSGR3Modeldemonstrates the fea-
sibility of incorporating RAG with external knowledge intosemantic scene graph generation and lays the foundation for
future research on more advanced structural modeling and
knowledge integration mechanisms.
REFERENCES
[1] H. Yin, X. Xu, Z. Wu, J. Zhou, and J. Lu, “Sg-nav: Online 3d scene
graph prompting for llm-based zero-shot object navigation,”Advances
in neural information processing systems, vol. 37, pp. 5285–5307,
2024.
[2] R. Liu, X. Wang, W. Wang, and Y . Yang, “Bird’s-eye-view scene graph
for vision-language navigation,” inProceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 10 968–
10 980.
[3] Z. Seymour, N. C. Mithun, H.-P. Chiu, S. Samarasekera, and R. Kumar,
“Graphmapper: Efficient visual navigation by scene graph generation,”
in2022 26th International Conference on Pattern Recognition (ICPR).
IEEE, 2022, pp. 4146–4153.
[4] Y . Chang, L. Ballotta, and L. Carlone, “D-lite: Navigation-oriented
compression of 3d scene graphs for multi-robot collaboration,”IEEE
Robotics and Automation Letters, vol. 8, no. 11, pp. 7527–7534, 2023.
[5] I. Armeni, Z.-Y . He, J. Gwak, A. R. Zamir, M. Fischer, J. Malik,
and S. Savarese, “3d scene graph: A structure for unified semantics,
3d space, and camera,” inProceedings of the IEEE International
Conference on Computer Vision, 2019, pp. 5664–5673.
[6] A. Werby, C. Huang, M. B ¨uchner, A. Valada, and W. Burgard,
“Hierarchical open-vocabulary 3d scene graphs for language-grounded
robot navigation,” inRobotics: Science and Systems XX, ser. RSS2024.
Robotics: Science and Systems Foundation, Jul. 2024.
[7] W. Xu, V . Ila, L. Zhou, and C. T. Jin, “Tb-hsu: Hierarchical 3d scene
understanding with contextual affordances,” 2025.
[8] R. Korekata, Q. Xie, Y . Bisk, and K. Sugiura, “Affordance rag:
Hierarchical multimodal retrieval with affordance-aware embodied
memory for mobile manipulation,”IEEE Robotics and Automation
Letters, vol. 11, no. 3, pp. 2706–2713, 2026.
[9] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, “Scenegraph-
fusion: Incremental 3d scene graph prediction from rgb-d sequences,”
2021.
[10] J. Wald, H. Dhamo, N. Navab, and F. Tombari, “Learning 3d semantic
scene graphs from 3d indoor reconstructions,” inConference on
Computer Vision and Pattern Recognition (CVPR), 2020.
[11] S.-C. Wu, K. Tateno, N. Navab, and F. Tombari, “Incremental 3d
semantic scene graph prediction from rgb sequences,” 2023.
[12] Y . Qiu and H. I. Christensen, “3d scene graph prediction on point
clouds using knowledge graphs,” 2023.
[13] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, D. Yudin,
M. Monastyrny, and A. Valenkov, “Beyond bare queries: Open-
vocabulary object grounding with 3d scene graph,” in2025 IEEE
International Conference on Robotics and Automation (ICRA). IEEE,
2025, pp. 13 582–13 589.
[14] S. Koch, N. Vaskevicius, M. Colosi, P. Hermosilla, and T. Ropinski,
“Open3dsg: Open-vocabulary 3d scene graphs from point clouds with
queryable objects and open-set relationships,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 14 183–14 193.
[15] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen,
A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappaet al.,
“Conceptgraphs: Open-vocabulary 3d scene graphs for perception and
planning,” in2024 IEEE International Conference on Robotics and
Automation (ICRA). IEEE, 2024, pp. 5021–5028.
[16] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances
in neural information processing systems, vol. 33, pp. 9459–9474,
2020.
[17] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot,
and P. Colombo, “Colpali: Efficient document retrieval with vision
language models,” 2025.
[18] J. Wald, A. Avetisyan, N. Navab, F. Tombari, and M. Nießner, “Rio:
3d object instance re-localization in changing indoor environments,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2019, pp. 7658–7667.

[19] Z. Wang, B. Cheng, L. Zhao, D. Xu, Y . Tang, and L. Sheng, “Vl-
sat: Visual-linguistic semantics assisted training for 3d semantic scene
graph prediction in point cloud,” inProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2023, pp.
21 560–21 569.
[20] M. Feng, C. Yan, Z. Wu, W. Dong, Y . Wang, and A. Mian, “History-
enhanced 3d scene graph reasoning from rgb-d sequences,”IEEE
Transactions on Circuits and Systems for Video Technology, 2025.
[21] T. Zemskova and D. Yudin, “3dgraphllm: Combining semantic graphs
and large language models for 3d scene understanding,” inProceedings
of the IEEE/CVF International Conference on Computer Vision, 2025,
pp. 8885–8895.
[22] A. Dutta, K. S. Mehrab, M. Sawhney, A. Neog, M. Khurana, S. Fatemi,
A. Pradhan, M. Maruf, I. Lourentzou, A. Dawet al., “Open world
scene graph generation using vision language models,”arXiv preprint
arXiv:2506.08189, 2025.
[23] H. Koo, M. Kim, and S. J. Hwang, “Optimizing query generation for
enhanced document retrieval in rag,” 2024.
[24] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettle-
moyer, and W. tau Yih, “Replug: Retrieval-augmented black-box
language models,” 2023.
[25] Z. Rackauckas, “Rag-fusion: A new take on retrieval augmented
generation,”International Journal on Natural Language Computing,
vol. 13, no. 1, p. 37–47, Feb. 2024.
[26] S. Jeong, K. Kim, J. Baek, and S. J. Hwang, “Videorag: Retrieval-
augmented generation over video corpus,” inFindings of the Asso-
ciation for Computational Linguistics: ACL 2025, 2025, pp. 21 278–
21 298.
[27] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang,
X. Han, Z. Liu, and M. Sun, “Visrag: Vision-based retrieval-augmented
generation on multi-modality documents,” 2025.
[28] T. Zhou, S. Mei, X. Li, Z. Liu, C. Xiong, Z. Liu, Y . Gu, and G. Yu,
“Marvel: Unlocking the multi-modal capability of dense retrieval via
visual module plugin,” 2024.
[29] Q. Wang, R. Ding, Z. Chen, W. Wu, S. Wang, P. Xie, and F. Zhao, “Vi-
dorag: Visual document retrieval-augmented generation via dynamic
iterative reasoning agents,” inProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, 2025, pp. 9124–
9145.
[30] Y . Fang, Z. Shi, J. Qiu, Z. Chen, J. Shi, H. Xu, J. Huo, and Y . Gao,
“Inherit-sg: Incremental hierarchical semantic scene graphs with rag-
style retrieval,”arXiv preprint arXiv:2602.12971, 2026.
[31] F. Yu, Q. Deng, S. Tang, Y . Li, and L. Cheng, “Open-world 3d scene
graph generation for retrieval-augmented reasoning,”arXiv preprint
arXiv:2511.05894, 2025.
[32] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdul-
mohsin, N. Parthasarathy, T. Evans, L. Beyer, Y . Xia, B. Mustafaet al.,
“Siglip 2: Multilingual vision-language encoders with improved se-
mantic understanding, localization, and dense features,”arXiv preprint
arXiv:2502.14786, 2025.
[33] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,”IEEE transactions on big data, vol. 7, no. 3, pp. 535–547,
2019.
[34] P. Xu, S. Wang, Y . Zhu, J. Li, and Y . Zhang, “Spatialbench: Bench-
marking multimodal large language models for spatial cognition,”
arXiv preprint arXiv:2511.21471, 2025.
[35] I. Stogiannidis, S. McDonagh, and S. A. Tsaftaris, “Mind the gap:
Benchmarking spatial reasoning in vision-language models. 2025,”
URL https://arxiv. org/abs/2503.19707, 2025.
[36] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen,
X. Liu, J. Wang, W. Geet al., “Qwen2-vl: Enhancing vision-language
model’s perception of the world at any resolution,”arXiv preprint
arXiv:2409.12191, 2024.
[37] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and
M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor
scenes,” inProceedings of the IEEE conference on computer vision
and pattern recognition, 2017, pp. 5828–5839.
[38] S. Bai, Y . Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng,W. Ding, C. Gao, C. Geet al., “Qwen3-vl technical report,”arXiv
preprint arXiv:2511.21631, 2025.
[39] C. Lu, R. Krishna, M. Bernstein, and L. Fei-Fei, “Visual relationship
detection with language priors,” inEuropean conference on computer
vision. Springer, 2016, pp. 852–869.
[40] D. Xu, Y . Zhu, C. B. Choy, and L. Fei-Fei, “Scene graph generation
by iterative message passing,” inProceedings of the IEEE conference
on computer vision and pattern recognition, 2017, pp. 5410–5419.
[41] P. Gay, J. Stuart, and A. Del Bue, “Visual graphs from motion
(vgfm): Scene understanding with object geometry reasoning,” in
Asian Conference on Computer Vision. Springer, 2018, pp. 330–346.
[42] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li,
J. Yang, H. Suet al., “Grounding dino: Marrying dino with grounded
pre-training for open-set object detection,” inEuropean conference on
computer vision. Springer, 2024, pp. 38–55.