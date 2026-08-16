# RAGMesh with FaME-G2E: Long-Form Text-Driven 3D Face Generation and Editing

**Authors**: Hao Li, Ju Dai, Feng Zhou, Mengting Shi, Haofei Wang, Zhen Song, Wei Zhou, Lei Li, Junjun Pan

**Published**: 2026-08-10 06:55:35

**PDF URL**: [https://arxiv.org/pdf/2608.09186v1](https://arxiv.org/pdf/2608.09186v1)

## Abstract
Text-driven 3D face generation and editing remains challenging due to the difficulty of translating long-form descriptions into fine-grained facial geometry. Existing methods primarily align global textual semantics with facial structures but often struggle to capture subtle local deformations, such as eyebrow tension, cheek contraction, and asymmetric mouth motions, resulting in limited geometric fidelity and editing precision. To facilitate fine-grained text-driven facial modeling, we first construct FaME-G2E, a large-scale multimodal dataset containing detailed text--mesh annotations and paired text--blendshape samples for unified 3D facial generation and editing. Based on this dataset, we propose RAGMesh, a retrieval-augmented framework that leverages text-correlated geometric priors to improve high-fidelity facial synthesis and editing. Specifically, the Multi-Scale Retrieval Fusion (MSRF) module retrieves semantically consistent global and regional facial priors and fuses them in the blendshape space, suppressing conflicting local deformations while preserving coherent deformation patterns. Furthermore, we introduce Adaptive RAG-guided Supervision (AdaRAGS), a region-aware constraint that explicitly aligns textual semantics with corresponding facial regions, enhancing regional controllability and editing accuracy. Extensive experiments on FaME-G2E demonstrate that RAGMesh achieves superior performance over state-of-the-art methods in local geometric accuracy, text-guided controllability, regional editing precision, and inference efficiency. Video demo is available at https://youtu.be/Yr0_XkpWcNk, and the source code and dataset will be released upon paper acceptance.

## Full Text


<!-- PDF content starts -->

1
RAGMesh with FaME-G2E: Long-Form
Text-Driven 3D Face Generation and Editing
Hao Li, Ju Dai∗, Feng Zhou, Mengting Shi, Haofei Wang, Zhen Song, Wei Zhou,Senior Member IEEE, Lei Li,
Junjun Pan∗
Abstract—Text-driven 3D face generation and editing remains
challenging due to the difficulty of translating long-form de-
scriptions into fine-grained facial geometry. Existing methods
primarily align global textual semantics with facial structures but
often struggle to capture subtle local deformations, such as eye-
brow tension, cheek contraction, and asymmetric mouth motions,
resulting in limited geometric fidelity and editing precision. To fa-
cilitate fine-grained text-driven facial modeling, we first construct
FaME-G2E, a large-scale multimodal dataset containing detailed
text–mesh annotations and paired text–blendshape samples for
unified 3D facial generation and editing. Based on this dataset,
we propose RAGMesh, a retrieval-augmented framework that
leverages text-correlated geometric priors to improve high-fidelity
facial synthesis and editing. Specifically, the Multi-Scale Retrieval
Fusion (MSRF) module retrieves semantically consistent global
and regional facial priors and fuses them in the blendshape space,
suppressing conflicting local deformations while preserving coher-
ent deformation patterns. Furthermore, we introduce Adaptive
RAG-guided Supervision (AdaRAGS), a region-aware constraint
that explicitly aligns textual semantics with corresponding facial
regions, enhancing regional controllability and editing accu-
racy. Extensive experiments on FaME-G2E demonstrate that
RAGMesh achieves superior performance over state-of-the-art
methods in local geometric accuracy, text-guided controllability,
regional editing precision, and inference efficiency. Video demo is
available at https://youtu.be/Yr0 XkpWcNk, and the source code
and dataset will be released upon paper acceptance.
Index Terms—3D Facial animation, Fine-grained edit, AIGC.
I. INTRODUCTION
TExt-driven 3D facial generation and editing aim to
achieve controllable creation and manipulation of human
faces via natural language descriptions [1]–[5]. These tasks
enable intuitive and flexible expression modeling for virtual
humans and character animation [6]–[10]. However, existing
models struggle to align complex and structured long tex-
tual descriptions with local facial geometry, often confusing
direction-sensitive regions such as the left and right eyes or
mouth corners, resulting in ambiguous or mirrored expressions
that deviate from the intended textual semantics [11]–[15].
The generation results of early end-to-end modeling frame-
works often focus on text tokens with the most substantial
gradients while neglecting region-specific or subtle descriptive
Corresponding authors: Ju Dai (daij@pcl.ac.cn) and Junjun Pan
(pan junjun@buaa.edu.cn).
Hao Li and Junjun Pan are with the State Key Laboratory of Virtual Reality
Technology and Systems, Beihang University, Beijing, China. Hao Li, Ju Dai,
Haofei Wang, and Zhen Song are with Pengcheng Laboratory, Shenzhen,
China. Mengting Shi is with Shanxi University of Finance and Economics,
Shanxi, China. Feng Zhou is with North China University of Technology,
Beijing, China. Wei Zhou is with Cardiff University, Cardiff, UK, Lei Li is
with Beijing Institute of Technology, Beijing, China.cues [16]–[19]. Methods optimized via gradient supervision
derived from CLIP-based 2D priors (e.g., SDS [13]) effectively
capture global semantics, but their performance is constrained
by the resolution and expressiveness of the pretrained feature
space, making it difficult to handle fine-grained descriptions in
long textual inputs [11], [13], [20], [21]. Recently, LLM-based
parsing methods, such as ICE [12], map parsed textual se-
mantics onto predefined BlendShape weights. Although these
approaches provide interpretability and controllability, their
overall coherence remains suboptimal. In summary, existing
methods [13], [22], [23] struggle to achieve precise alignment
between complex, structured long textual descriptions and
localized facial geometric deformations.
Recently, several works [24], [25] such as RetDream [26]
for text-to-3D reconstruction and AMD [27] for text-to-motion
generation have demonstrated that introducing external priors
through retrieval-augmented generation (RAG) effectively en-
hances semantic richness and diversity across various AIGC
tasks [28]–[31]. By retrieving task-relevant knowledge or ex-
amples, RAG can provide additional prior knowledge, enhance
contextual grounding and enhance the fidelity of generated
content. Inspired by these successes, we argue that retrieving
semantically similar local geometries from the retrieval set
as external priors can effectively constrain and guide the
generation process, leading to more stable and fine-grained
alignment between linguistic semantics and facial geometry.
In this paper, we address the challenge that existing text-
driven 3D facial mesh generation and editing methods often
fail to accurately capture fine-grained local facial deformations
described in long-form text, especially when the descriptions
contain multiple localized and direction-sensitive cues. To
support fine-grained generation and editing, we first construct
FaME-G2E, a multimodal dataset built using a visual LLM and
a monocular 3D reconstruction model. FaME-G2E contains
fine-grained text–mesh pairs and text–blendshape pairs, pro-
viding unified data support for both 3D facial generation and
editing tasks. Based on this dataset, we propose RAGMesh,
a retrieval-augmented framework that incorporates external
facial geometry priors into the generation process. Specifically,
we design a Multi-Scale Retrieval Fusion (MSRF) module
to retrieve complementary facial priors from both global and
regional perspectives and fuse them into a reference mesh
within a shared blendshape space. The fused reference mesh
provides structured geometric guidance, enabling the model to
better infer subtle local deformations that are difficult to learn
from text alone. Beyond using retrieval results as additional
inputs, RAGMesh further exploits them to guide optimization.
arXiv:2608.09186v1  [cs.CV]  10 Aug 2026

2
We introduce Adaptive RAG-Guided Supervision (AdaRAGS),
which identifies text-relevant facial regions according to the
retrieved local priors and applies region-aware constraints
during training. This design encourages the model to focus on
subtle local changes, such as asymmetric mouth movements
or eye-region deformations, rather than being dominated by
large static facial areas. As a geometry-centric framework,
RAGMesh can be seamlessly combined with existing tex-
ture synthesis and rendering pipelines. Extensive experiments
demonstrate its effectiveness in fine-grained 3D facial mesh
generation and editing.
•We construct FaME-G2E, a large-scale multimodal
dataset containing fine-grained long-form text–mesh an-
notations and paired text–blendshape samples, which
enables supervised learning for both 3D face generation
and editing and alleviates the reliance on SDS-based
optimization.
•We present RAGMesh, a retrieval-augmented framework
for controllable 3D facial mesh generation and edit-
ing from long-form textual descriptions. By introducing
topology-consistent facial geometry priors, RAGMesh
bridges complex language descriptions and localized
facial deformations, enabling more precise text-driven
control over fine-grained facial geometry.
•We design a geometry-aware retrieval and supervision
mechanism for local facial editing. The proposed MSRF
module retrieves and fuses global and region-level facial
priors into a reference mesh, while AdaRAGS converts
retrieved local deformation cues into region-aware super-
vision, improving localized editing accuracy and reducing
unintended changes in non-target facial regions.
II. RELATEDWORKS
A. Text-driven Facial Mesh Generation
Text-driven facial mesh generation aims to synthesize 3D
meshes directly from natural language descriptions. A straight-
forward approach is to learn a direct mapping from text to a
3D mesh using neural networks [32], [33]. However, when
processing long and structurally complex textual inputs, the
model tends to focus on dominant tokens while neglecting
fine-grained semantic details. Due to the scarcity of 3D
data, many studies leverage 2D priors for optimization [23],
[34]. For example, T2P [35] employs a CLIP-based semantic
alignment mechanism to jointly optimize facial parameters and
textual features, enabling text-driven mesh generation. Score
Distillation Sampling (SDS) [36] optimizes 3D representa-
tions by acquiring gradient signals from a pretrained text-
to-image diffusion model. Methods such as DreamFace [11]
and FaceG2e [13] adopt the SDS framework to achieve
text-driven 3D face generation. Although SDS leverages 2D
image gradients for supervision and eliminates the need for
large-scale 3D datasets, it primarily captures global semantics
and is computationally expensive. To balance efficiency and
reliability, ICE [12] employs an LLM-based instruction parser
and a semantic-guided low-dimensional solver for predefined
blendshape bases. However, its generated results still exhibit
limited global coherence.B. Datasets for Text-driven Mesh Generation
Text-driven 3D tasks encompass both generation and edit-
ing, yet existing datasets are insufficient to support such
tasks effectively. Some 3D facial scan datasets, such as BU-
3DFE [37] and BP4D [38], provide high-quality geometry but
lack detailed textual descriptions. Although Describe3D [39]
provides fine-grained textual annotations for 3D meshes, its
predefined description categories, limited sample size, and
task-specific design make it unsuitable for editing. Due to the
scarcity of large-scale 3D datasets, most existing methods rely
on 2D priors within SDS-based frameworks [13], [25], [36].
However, these SDS-based approaches can handle only short
text inputs, struggle with long, structurally complex textual
descriptions, and are nearly incapable of performing con-
tinuous text-driven editing. By integrating Vision–Language
Large Models [40], [41] with monocular reconstruction tech-
niques [42], we generate high-quality paired Text-Mesh pair
data from large-scale 2D facial images, substantially enriching
both data diversity and scale. Moreover, the incorporation
of diverse blendshape representations enables FaME-G2E to
seamlessly handle both generation and editing tasks.
C. RAG for 3D Content Generation
Retrieval-Augmented Generation (RAG) has been widely
adopted across domains such as natural language understand-
ing, knowledge-grounded text generation, and image synthe-
sis [43]–[45]. It enhances generative models by integrating
external knowledge retrieved from large-scale databases. Re-
cently, RAG has been extended to 3D content generation,
demonstrating that incorporating external priors can improve
both geometric consistency and overall generation quality [26],
[27], [46]. ReMoDiffuse [46] dynamically integrates retrieved
motion samples with textual semantics for text-to-motion
generation, while AMD [27] decomposes textual descriptions
and fuses language and motion priors within a diffusion
framework. However, these methods primarily operate on
temporal motion priors and global semantic structures, with-
out explicitly modeling conflicts among fine-grained local
geometric cues under long-form descriptions. Similarly, Ret-
Dream [26] retrieves semantically relevant 3D assets as geo-
metric priors for SDS-based text-to-3D optimization, where
the retrieved assets mainly provide global-level geometric
guidance. In contrast, facial geometry generation introduces a
more challenging setting, where long textual descriptions often
involve region-specific and direction-aware constraints that
require fine-grained geometric consistency. To address this,
we move beyond single-reference retrieval or naive feature
aggregation and construct a more expressive geometric prior
by jointly leveraging multiple retrieved facial meshes. We
perform blendshape-level fusion within a shared deformation
space, yielding a structured and descriptive reference mesh.
This fusion strategy preserves deformation patterns consis-
tently supported across semantically relevant retrieved sam-
ples, while suppressing incompatible local deformations across
different facial regions. As a result, the fused representation
avoids over-smoothing artifacts commonly observed in vertex-
space averaging and provides a more stable and structured

3
Visual Language 
Large Model
Gender is female, age is in 
the youth stage; The width of 
the nose bridge is moderate;
...... The emotional state pre-
sents a joyful state.
Monocular 3D 
Reconstruction ModelManual InspectionPrompt
Input ImageBlendshape
Turn the corner of the 
left mouth upwardsGeneration Task data acquisition process
Prompt
Text DescriptionEditing Task data acquisition process
Language 
Large Model1.The left corner of the 
mouth slightly tilts upwards.
2. The left corner of the 
mouth …… , with a naturally 
cheerful expression.
3. .…….
3D Mesh3D 
Blendshape
Manual InspectionDescribe the characteristics 
of the characters in the 
picture, including gender, 
age, .....Using the tone of user 
editing, expand the text 
description to XXX.....
Fig. 1. Pipeline of FaME-G2E dataset construction. The left presents the generation task, and the right depicts the editing task.
geometric prior. Furthermore, the aggregated reference mesh
naturally enables region-aware supervision, allowing fine-
grained alignment between textual semantics and localized
facial deformations, thereby significantly improving both 3D
facial mesh generation and editing.
III. METHODOLOGY
A. FaME-G2E
1) Motivation:Most existing text-driven 3D generation
methods rely on 2D prior supervision, such as CLIP-based
semantic alignment or SDS-based optimization [13], [25], to
circumvent the lack of large-scale text-annotated 3D datasets.
However, these 2D priors mainly emphasize image-level se-
mantic consistency and provide only indirect supervision for
3D geometry. In particular, CLIP-based representations have
limited capability in preserving multiple fine-grained and
direction-sensitive constraints contained in long, structurally
complex descriptions. Similarly, SDS derives optimization
signals from pretrained 2D diffusion models through rendered
images, rather than directly constraining localized 3D facial
geometry. Consequently, these approaches often struggle to
accurately model subtle and asymmetric facial deformations,
such as unilateral mouth movements, eyebrow variations,
and local cheek contractions. These limitations hinder the
establishment of precise correspondence between long-form
textual semantics and fine-grained facial geometry.
To further clarify the limitations of existing data resources
and supervision paradigms, Table I provides two complemen-
tary comparisons. As summarized in Table I, existing 3D facial
datasets primarily provide geometric scans, blendshape bases,
or predefined expression labels, but generally lack paired free-
form textual descriptions. Describe3D introduces text–mesh
annotations, yet its predefined categories, limited scale, and
absence of editing-oriented samples restrict its applicability to
long-form generation and localized editing. Meanwhile, CLIP-
and SDS-based paradigms avoid paired 3D annotations by ex-
ploiting pretrained 2D priors, but their image-level supervision
provides only indirect constraints on localized facial geometry.
These limitations motivate FaME-G2E, which provides large-
scale text–mesh pairs for generation and text–blendshape pairs
for fine-grained editing.
2) Dataset Construction:As shown in Figure 1, FaME-
G2E comprises two complementary subsets. For the gener-
ation task, we use images from AffectNet as visual inputs
and employ the vision-language model, doubao-1-5-thinking-
vision-pro, to generate fine-grained facial descriptions using
predefined visual question-answering prompts. In parallel, weapply EMOCA [42] to reconstruct the corresponding 3D
facial meshes in the FLAME topology with 5023 vertices,
producing paired Text–Mesh samples. After manual validation,
only samples with high semantic consistency between the
generated text and reconstructed mesh are retained. For the
editing task, we further construct a Text–Blendshape subset
from shape and motion blendshape control bases. For each
blendshape basis, we generate and expand user-oriented textual
descriptions using LLM-augmented prompts that cover diverse
local deformation types, directional attributes, and expression
intensities. To simulate different editing intensities, we assign
different activation values to the corresponding blend shape
controls to represent weak, natural, and strong expressions,
respectively. The generated descriptions are manually verified
to ensure consistency with the target blendshape semantics.
The detailed prompts for both data generation and data editing
are provided in the supplementary material (SM).
3) Dataset Statistics:In summary, FaME-G2E is a large-
scale and comprehensive multimodal dataset designed for
text-driven 3D facial generation and editing. As shown in
Table II, it contains 500K Text–Mesh pairs with fine-grained
multi-dimensional annotations for generation, and 30K Text-
Blendshape pairs curated for editing. The Text-Blendshape
subset includes 50 FLAME shape blendshapes, 51 ARKit
motion blendshapes, 14 AU-based blendshapes, and 7 stan-
dard emotion blendshapes. Each blendshape is further divided
into three intensity levels and is associated with 50 to 100
distinct textual descriptions. This subset provides diverse
textual descriptions and corresponding blendshape control
bases, enabling interpretable and controllable expression mod-
eling and manipulation. Compared with existing 3D facial
datasets, FaME-G2E is the first large-scale multimodal paired
Text–Mesh dataset that simultaneously supports text-driven
generation and editing tasks.
B. RAGMesh
As illustrated in Figure 2, our framework addresses both
generation and editing tasks through text-driven 3D facial
synthesis. The generation task takes a template as input to
produce text-guided meshes, whereas the editing task deforms
existing 3D meshes according to the text. The dual-stream
architecture leverages the reference mesh generated by Multi-
Scale Retrieval Fusion (MSRF) to compensate for the missing
local details in long textual descriptions, thereby enhanc-
ing the fine-grained geometric representation and semantic
consistency. In addition, Adaptive RAG-Guided Supervision

4
TABLE I
COMPARISON OF EXISTING3DFACIAL DATASETS AND SUPERVISION PARADIGMS FOR TEXT-DRIVEN GENERATION AND EDITING.
Dataset Comparison
Dataset Geometry Representation Text Description Annotation Style
BU-3DFE [37] 3D Mesh✗Emotion
FaceWarehouse [47] Blendshape✗Expression Parameters
Describe3D [39] 3D Mesh Limited Template-based Text
FaME-G2E3D Mesh and Blendshape✓Free-form Text
Supervision Paradigm Comparison
Paradigm Supervision Optimization Granularity Local Control
CLIP-based Image–Text Alignment Global-oriented Semantic Alignment✗
SDS-based Diffusion Prior Global-oriented Semantic Optimization Limited
LLM-based Parsing Semantic Mapping Region-level Parameter Mapping Limited
Paired Text–MeshText–Mesh Pairs Fine-grained Geometric Optimization✓
✓: supported;✗: unsupported; Limited: partially supported.
TABLE II
DATA CHARACTERISTICS INFAME-G2EFOR DIFFERENT TASKS.
Attribute Generation Task Editing task
Data Scale500K 30K
Text LengthAvg. 63 words Avg. 11 words
Annotation TypeEmotion, AU, Motion,
Shape, AttributeEmotion, AU, Motion,
Shape
IntensityNo intensity Weak/Natural/Strong
Annotation StrategyType Mixing Type Independent
(AdaRAGS) explicitly aligns textual semantics with localized
geometric structures, improving consistency and robustness.
C. 3-Part RVQVAE for Representation
Build a compact, structured latent representation using a
discrete codebook, enabling the model to generate in a control-
lable, continuous space. Traditional compression models [32],
[48] often fail to preserve subtle local deformations. To address
the issue, as shown in Figure 2 (c), we construct a 3-Part
RVQV AE network to encode facial priors and divide the facial
mesh into upper face, lower face, and head regions for region-
specific compression and reconstruction. Our 3-Part RVQV AE
enables the model to focus on local geometric changes and
subtle facial deformation modeling. The encoding process can
be described as:
z(0)
r=Er(Vr),(1)
z(l)
q,r= Quantization(z(l−1)
r),(2)
wherer∈ {upper,lower,head},V rdenotes the vertices of
the facial regionr,E r(·)is the mesh encoder of the 3-
part RVQV AE for regionr, andz(l)
q,rdenotes the quantized
representation from thel-th quantized layer. As shown in
Figure 2 (d), the residual update allows each quantization
layer to encode the remaining error of the previous stage,
progressively refining the latent representation:
z(l)
r=z(l−1)
r−z(l)
q,r,(3)
wherez(l)
ris the residual feature atl-th quantization layer.The decoder takes the quantized features of different layers
as input and predicts the target facial vertices:
ˆVr=Dr(LX
l=1z(l)
q,r),(4)
whereD r(·)denotes the decoder for facial regionr, ˆVr
represents the predicted vertices of that region.
D. Multi-Scale Retrieval Fusion
MSRF aims to generate a reference mesh that semantically
aligns with the description. It consists of two stages: multi-
scale retrieval and mesh fusion. In the retrieval stage, global
and local retrieval are performed simultaneously to capture
multi-scale semantic and geometric correlations within a pre-
built retrieval set. The global retrieval captures overall facial
structure and emotional consistency, while the local retrieval
focuses on regional geometry and subtle facial deformations.
In the fusion stage, the retrieved meshes are aggregated using
a weighted strategy to produce the reference mesh.
To ensure retrieval efficiency and semantic representative-
ness, we construct a retrieval setSfrom FaME-G2E, which
includes paired text and mesh. We extract hybrid repre-
sentations of geometric features and textual semantics, and
apply hierarchical clustering to select the most representative
samples for the retrieval candidates. Given a text description
T, a Text2Vec [49] encoderE tis leveraged to obtain the text
embedding, which serves as a query vector to retrieve the Top-
Kfacial meshes from the retrieval setS.
We perform both global and local retrieval. Global retrieval
operates on the entire text sequence to capture overall semantic
alignment. In contrast, local retrieval dividesTintoNsmaller
segments based on punctuation marks and performs stepwise
retrieval to capture fine-grained, region-specific semantic cues.
The retrieval process can be formulated as:
Qg,Rg= TopK(Sim(E t(T),E t(Tk)), Vk), k∈ |S|,(5)
Ql,Rl=N[
n=1TopK(Sim(E t(Tn),Et(Tk)), Vk),(6)

5
3-Part
Transformer
Decoder
Input Mesh
A middle-aged 
woman with a 
leftward shift the 
mouth, raised 
cheeks, open 
eyes, natural 
eyebrows, and 
raised nose.
Text
Multmodal Fusion Network
Adaptive RAG-Guided Supervision(b) Geometric Stream(a) Textual Stream
BERT
Latent CodeMesh 
Encoder
Output Mesh
(c) 3-Part RVQVAE
Output Mesh
Reference 
MeshMesh 
Encoder
3-Part
Transformer
DecoderMesh 
Decoder
...
Latent
Feature
Completes fine-grained alignment of regions baesd on retrieval information
Retrieval SetGlobal Retrieval
      A middle-aged 
woman with a leftward 
shift the mouth 
......
Local Retrieval
   A middle-aged woman.
   Leftward shift the 
mouth. 
   ....
Mesh Fusion...
...
Multi-Scale Retrieval Fusion 
Input Mesh
Upper-Quantized
Head-QuantizedLower-QuantizedMesh 
EncoderMesh 
DecoderFrozen Modules
Retrieval 
Alignment
(d) Residual Quantization
Quantized Code
0
1
…
N
0
1
…
N0
1
…
NCode Book 1 Code Book 2 Code Book 3
Fig. 2. Overview of RAGMesh. (a) (b) show the textual stream and geometric stream in the dual-stream architecture. Given a text description and an input
mesh, MSRF retrieves meshes from a predefined retrieval set via global and local semantic retrieval. The retrieved meshes are fused to generate a reference
mesh serving as a geometric prior. Next, two 3-Part Transformer decoders process the input mesh with textual and geometric features. Finally, the multimodal
fusion network fuses textual and geometric features to generate latent features, which are decoded into the target mesh. AdaRAGS completes fine-grained
alignment of regions based on retrieval information. (c) (d) show the architecture of 3-Part RVQV AE and Residual Quantization.
where(T k, Vk)∈ Sis thek-th text and mesh pair in the
retrieval set,Sim(·)is the similarity operation, andTnis then-
th text segment.R gandR lare the global and local retrieved
meshes.Q gandQ lare the semantic similarity scores with the
corresponding text query global and local retrieved mesh. The
fusion weight of thek-th retrieved mesh is computed as:
wk
b=exp  
qk
b+ηk
b
/τ
PK
j=1exp
qj
b+ηj
b
/τ, b∈ {g, l},(7)
whereqk
bis the similarity of the k-th geometric text.ηk
b∼
N(0, σ2)is a small stochastic perturbation, andτis a tem-
perature parameter controlling the sharpness of the weight
distribution.
Finally, the reference mesh is obtained by merging the
global and local meshes in the blendshape space:
Vr
ref=αw gRg+βw lRl,(8)
whereαandβcontrol the contributions of the global and local
retrieval branches.
E. Textual Geometric Dual-Stream Architecture
To better leverage text and input mesh information, we
design a dual-stream architecture that jointly processes textual
semantics and geometric priors for 3D facial generation and
editing. Specifically, the textual stream extracts semantic rep-
resentations from the textual description, and the geometric
stream integrates multi-scale information from the retrieved
reference meshes:
ft,r= Φt,r([Er(Vinp,r) ; BERT(T)]),(9)
fg,r= Φg,r([Er(Vinp,r) ;Er(Vref,r)]),(10)whereV inp,r andV ref,r denotes vertices of facial regionrfor
the input mesh and reference mesh, respectively.Φ t,randΦ g,r
represent 3-part Transformer decoder for processing textual
and geometric references, andf t,randf g,rdenote the textual
and geometric encoding features of the facial region.f t,rand
fg,rare integrated via a Multimodal Fusion Network (MFN),
a three-layer MLP with GELU activations designed to fuse the
concatenated text-guided and geometry-guided latent features,
to produce a fused latent representation that is subsequently
decoded into the target mesh:
zf,r= MFN([f t,r;fg,r]),(11)
ˆVr=Dr(zf,r),(12)
wherez f,rdenotes the fused representation of regionr, ˆVr
is the predicted mesh vertex of facial regionr.
F . Adaptive RAG-Guided Supervision
Text descriptions usually convey localized and directional
facial semantics. Directly applying global supervision fails to
capture fine-grained regional cues and deformation directions.
To address this issue, we propose AdaRAGS to enhance
supervision on the vertices of specific regions. It leverages
fine-grained, semantically relevant meshes retrieved by the
MSRF, enabling directionally consistent, semantically aligned
local supervision. Firstly, we define the facial vertex mask:
M=M[
m=1n
iRm
l,i−V 0,i
2> τ,o
,(13)
whereMdenotes a mask set that selects vertices with signif-
icant local motion,Mrepresents the total number of locally
retrieved reference meshes,idenotes the index of a vertex,

6
Rm
lis them-th retrieved mesh,V 0is the template mesh,τis
the displacement threshold. We apply the L1 loss only to the
masked vertices to guide the model in learning semantically
relevant local deformations:
LM=1
MMX
m=1P
i∈M(m)Vi−ˆVi
1
|M(m)|,(14)
whereM(m)denotes the vertex mask corresponding to the
m-th locally retrieved mesh,V iandˆVidenote the i-th vertex
of ground truth and predicted vertex, encouraging the network
to focus on local motion patterns.
G. Loss functions
We first train the 3-Part RVQV AE to reconstruct different
regions of the facial mesh. Each model is optimized using a
facial mesh reconstruction loss, along with two intermediate
latent-level losses.
Lrvq=X
r∥Vr−ˆVr∥1+X
r,l
∥SG( ˆzl−1
r)−zl
q,r∥2
2
+δ∥ ˆzl−1
r−SG(zl
q,r)∥2
2
,(15)
where the first term is a mesh reconstruction loss,SGstands
for a stop-gradient operation, andδrefers to the weighting
factor controlling the update rate.
For both generation and editing tasks, we train RAGMesh
using the corresponding data from FaME-G2E. We optimize
our model through facial mesh reconstruction loss, intermedi-
ate latent space loss, and AdaRAGS:
L=X
r∥Vr−ˆVr∥2
2+X
r,l∥ˆzl−1
r−SG(zl
q,r)∥2
2+γL M,(16)
whereγis a weight parameter for the AdaRAGS.
IV. EXPERMENTS
A. Implementation details
Our framework is built on the PyTorch platform and trained
on RTX 6000 Ada. All experiments use a random 8:1:1 train,
val, and test split on the proposed dataset. First, we optimize
the 3-part RVQV AE, with each RVQV AE setting the layer
numbers to 8 and latent spatial dimensions to 1024. The
hyperparameterδequals 0.1. Adam optimizer is used to train
RVQV AE with a learning rate of 1×10−5and a batch size of
8 for 200 epochs. Subsequently, we optimize the dual-stream
architecture by setting the hidden dimension to 1024. Both the
number of heads and the number of layers are set to 8. In the
retrieval fusion module, the retrieval size is set to TopK=5,
and the weighting factors for the reference meshes areα=0.3
andβ=0.7. In the Adaptive RAG-Guided Supervision, the
threshold parameter is set toτ=0.0001. The model is trained
for 20 epochs with a batch size of 8, a learning rate of 1×
10−5, and a loss weightγ=4.
B. Quantitative Evaluation
To verify RAGMesh, we evaluate all models on both gen-
eration and editing tasks using LVE (Loss of Vertex Error),
MRE (Mask-based Reconstruction Error), VS (Vendi Score),
and TIAS (Text–Image Alignment Score) metrics.1) LVE:Since RAGMesh performs long-context, fine-
grained text-to-3D facial generation, where the input de-
scription explicitly specifies local facial regions, deformation
directions, and deformation strength, the generation process
becomes strongly constrained and exhibits reduced semantic
ambiguity. Accordingly, we apply Loss of Vertex Error(LVE)
to evaluate the geometric fidelity of generated meshes with
respect to text-specified facial deformations:
LVE =||V− ˆV||2,(17)
whereVand ˆVdenote ground truth and predicted vertex.
2) MRE:To evaluate region-specific generation quality un-
der textual instructions, we adopt Mask-based Reconstruction
Error (MRE), which focuses on local structural fidelity. MRE
measures the geometric discrepancy between the generated
mesh and ground-truth mesh within the masked facial regions:
MRE =1
MMX
m=1P
i∈M(m)Vi−ˆVi
1
|M(m)|.(18)
3) VS:Vendi score(VS) measures the effective number of
distinct samples based on their pairwise similarity structure,
providing a distribution-level assessment of diversity. Unlike
geometry accuracy metrics, VS captures the global variability
of the generated mesh distribution and penalizes mode col-
lapse. We generate 10 meshes under the same input condition
and compute VS over these samples to evaluate geometric
diversity. Formally, given a set of generated meshes{V i}10
i=1,
we construct a similarity matrix:
Kij= Sim(V i, Vj), i, j= 1, . . . ,10,(19)
which is then normalized as:
˜K=K
tr(K).(20)
Let{λ i}denote the eigenvalues of ˜K. VS is defined as:
VS = exp 
−10X
i=1λilogλ i!
, λ i∈eig( ˜K),(21)
whereSim(·)denotes a pairwise mesh similarity function, ˜K
is the normalized similarity matrix,tr(·)is the matrix trace,
andeig(·)denotes eigenvalue decomposition.
Visual Language 
Large ModelTIAS Score Describe prompt:
Gender is female, age is in the middle and 
young stages; The width of the nose bridge is 
relatively small; ....... The emotional state 
presents a natural state of relaxation.
Evaluation Prompt：
Please evaluate the consistency 
between the given image and its 
textual description. .....
Render
Fig. 3. Overview of the TIAS assessment process. Given a generated 3D
face, we first render it into a 2D image and then use a visual-language large
model to evaluate the consistency between the rendered image and the textual
description, producing the final TIAS score.
4) TIAS:Due to the limited capability of CLIP-based simi-
larity models in capturing long-context semantic dependencies,
we introduce the Text–Image Alignment Score (TIAS) to
evaluate fine-grained text–geometry consistency in long-text
scenarios. Formally, TIAS is computed by a vision-language

7
Lip pucker.Editing TaskThe expression 
turned sad and 
the mouth 
closed.A middle-aged woman’s face, with moder-
ately high cheekbones; eyes tightly closed, 
corners of the mouth raised upward, and 
lips slightly parted.A man with a moderately wide nose bridge, 
moderately high cheekbones, a distinct soft 
jawline, his mouth slightly pouted and tilted a 
little to the left, with the left corner of his 
mouth raised, and his right eye tightly closed.GT T2P[31] FaceG2e [8]ICE [7] Ours TextGeneration TaskTADA[18]
Render Result
Text Source Editor GT Ours T2P[31] FaceG2e [8]ICE [7] TADA[18]
Source Image
Render Result
Render Result
Render Result
Source Image Source Image Source Image
Fig. 4. Qualitative comparisons of generation and editing results. The upper four rows present generation results and their renderings, while the lower four
rows present editing results and their renderings. Failure cases are highlighted with red boxes.
model-based evaluator. We use condition-driven 3DGS [50] as
our rendering module:
TIAS =f VLLM 
I, T, p e
,(22)
whereI= Render( ˆV,I s)denotes the rendered image of the
generated mesh ˆV,I sis source image,Tis the description
text, andp eis the evaluation prompt. Specifically, each gener-
ated 3D mesh is rendered into 2D images using 3D Gaussian
Splatting (3DGS) [50], conditioned on the original AffectNet
image as the source image. A VLLM-based evaluator is then
employed to assess the consistency between the rendered
images and the corresponding textual descriptions. Figure 3
illustrates the TIAS computation pipeline.C. Comparison with State-of-the-Art Methods
We compare RAGMesh with representative methods for
text-driven 3D facial generation and editing. These baselines
include FaceG2E [13] and ICE [12], which are specifically
designed for 3D facial modeling, as well as T2P [35] and
TADA [23], two general SDS-based text-to-3D approaches
adapted to the facial domain. For T2P and TADA, we re-
tain their original text-guided optimization strategies while
reformulating their geometric representations in the FLAME
parameter space to enable facial mesh generation and edit-
ing. Due to the lack of large-scale 3D facial mesh datasets
with long-form textual descriptions, all methods are evaluated
on the FaME-G2E dataset using the same data split and
experimental settings. This unified protocol facilitates a fair
comparison between dedicated facial modeling approaches and

8
adapted general-purpose text-to-3D methods.
Experimental results are reported in Table III. As illustrated,
RAGMesh consistently outperforms FaceG2E [13], T2P [35],
TADA [23], and ICE [12] across all evaluation metrics.
Specifically, the improvement in LVE validates enhanced
stability in geometric reconstruction, while the lowest MRE
demonstrates superior local geometric accuracy. To evaluate
generation diversity, we generate ten meshes for each input text
and quantify diversity using the Vendi Score. Each generated
mesh is rendered into a 2D image via GAGAvatar [50],
and semantic consistency is assessed using a VLLM-based
evaluator. RAGMesh achieves clear improvements in both
Vendi Score and TIAS, indicating that the generated facial
meshes better preserve fine-grained textual semantics while
producing richer and more plausible geometric variations.
For editing tasks, the diversity is relatively lower than that
of generation tasks, since editing instructions usually provide
more deterministic constraints on facial regions and deforma-
tion directions. Nevertheless, the retrieval-augmented geomet-
ric priors introduced by RAGMesh improve the accuracy of
text-specified regions, leading to better local fidelity and more
controllable editing results. Additionally, inference efficiency
is analyzed. FaceG2E [13], T2P [35], and TADA [23] require
iterative optimization, resulting in slower inference speed.
ICE [12] achieves faster inference but still relies on multi-
stage parsing and refinement. In contrast, RAGMesh achieves
high-quality generation with the highest inference efficiency.
These advantages across all metrics collectively demonstrate
that RAGMesh outperforms existing methods in fine-grained
modeling, semantic consistency, generation diversity, local
editing accuracy, and inference efficiency.
Visual comparisons of different methods for facial genera-
tion and editing are presented in Figure 4. For the generation
task, FaceG2e [13], T2P [35], and TADA [23] utilize CILP
to provide optimized gradients that only capture the global
facial structure but fail to model fine-grained details. The
LLM-parsing-based ICE [12] method generates meshes with
locally accurate textual alignment yet suffers from compro-
mised global coherence. Leveraging the proposed MSRF and
AdaRAGS, our RAGMesh effectively focuses on semantically
relevant regions via geometric priors and dynamic region-level
alignment, enabling more precise modeling of local directional
deformations.
Regarding the editing task, since textual descriptions are
typically shorter than in generation, ICE [12] effectively
captures local semantic-geometric correspondences, yielding
notable reconstruction gains. While FaceG2e [13] improves
in this task, it still struggles to capture fine-grained facial
deformations. In addition, we integrate the generated meshes
with an existing rendering framework to render them into 2D
images, and employ a VLLM-based evaluator to assess the
consistency between the input text and the generated results.
As reported in Table III, the results demonstrate that our
generated meshes are compatible with downstream rendering
pipelines. Meanwhile, the higher text-alignment scores further
indicate that RAGMesh achieves more semantically consistent
generation and editing results compared with existing methods.
Lip pucker.Editing TaskThe expression 
turned sad and 
the mouth 
closed.A middle-aged woman’s face, with moderately 
high cheekbones; eyes tightly closed, corners of 
the mouth raised upward, and lips slightly 
parted.A man with a moderately wide nose bridge, 
moderately high cheekbones, a distinct soft 
jawline, his mouth slightly pouted and tilted a 
little to the left, with the left corner of his 
mouth raised, and his right eye tightly closed.GT Baseline Ours TextGeneration TaskBaseline+MSRF
Text Template GT Baseline Ours Baseline+MSRFFig. 5. Visualization results of ablation experiments with different modules.
Our method better captures fine-grained textual semantics in both global
generation and local editing tasks, especially around the eyes, mouth, and
expression-related regions highlighted by red boxes.
D. Ablation Study for RAGMesh
To systematically evaluate the contribution of each design
choice, we conduct comprehensive ablation studies on model
key components, retrieval fusion strategies, MFN input con-
figurations, retrieval robustness, and RVQV AE partitioning
settings. We adopt an 8-layer Transformer with 8 attention
heads as the baseline model, and leverage BERT to provide
the textual representations.
Model key Component.MSRF and AdaRAGS are two
core components of RAGMesh for reducing ambiguity and
preserving local details in long-form text-to-3D facial gen-
eration. Notably, our text-only Transformer baseline already
outperforms existing state-of-the-art methods, mainly due to
the paradigm shift from CLIP-based image-level gradient
optimization to direct geometry-level optimization in the 3D
mesh/FLAME space. Compared with CLIP-guided supervision
obtained from rendered images, direct 3D losses provide more
stable and explicit constraints for facial geometry. However,
this baseline still struggles to reconstruct subtle facial details
and lacks the ability to explicitly distinguish region-specific
and direction-aware textual descriptions, as shown in Figure 5.
To address this limitation, MSRF retrieves complemen-
tary global and local geometric cues and fuses them in the
blendshape deformation space rather than directly averaging
vertices. Unlike single-reference retrieval methods that tend
to capture only the dominant mode of facial configuration
under long-form ambiguous descriptions, multi-scale retrieval
provides a richer and more complete support set for geometry
generation. The fusion is performed in a structured deforma-
tion space, which preserves consistent geometric components
while suppressing conflicting deformations, thereby avoiding
the over-smoothing effect of direct geometric averaging. This
design is grounded in the observation that text-to-3D facial
editing is an inherently ill-posed problem, where multiple
geometrically valid solutions may correspond to the same
textual instruction. In this setting, the retrieved reference mesh
should not be interpreted as a deterministic target, but rather
as a data-driven geometric prior that constrains the feasible

9
TABLE III
COMPARISON WITHSOTA METHODS.↓MEANS THE LOWER THE BETTER AND↑MEANS THE HIGHER THE BETTER.
MethodGeneration Editing
LVE↓MRE↓VS↑TIAS↑ LVE↓MRE↓VS↑TIAS↑ Time↓
(×10−6) (×10−4) (×10−1) (×10−1)(×10−7) (×10−4) (×10−1) (×10−1) s
T2P [35] 9.46 13.94 0.31 5.51 4.62 9.60 0.18 6.11 176
TADA [23] 8.32 13.48 0.38 5.17 4.75 9.54 0.16 5.97 294
FaceG2e [13] 7.32 12.71 0.33 6.47 4.15 9.36 0.15 6.15 361
ICE [12] 10.12 12.24 0.36 7.51 3.65 3.64 0.17 7.71 32
Ours1.07 7.88 0.43 8.31 1.72 3.41 0.21 8.59 2
TABLE IV
ABLATION EXPERIMENTS REGARDING MODEL COMPONENT,RETRIEVAL FUSION STRATEGY, MFNINPUT SETTING,AND RETRIEVAL ROBUSTNESS.
MethodGeneration Editing
LVE↓MRE↓VS↑TIAS↑ LVE↓MRE↓VS↑TIAS↑
(×10−6) (×10−4) (×10−1) (×10−1) (×10−7) (×10−4) (×10−1) (×10−1)
Model Component
Baseline 1.35 8.94 0.35 6.76 1.87 3.71 0.15 6.52
Baseline+MSRF 1.19 8.890.448.15 1.83 3.53 0.21 8.02
Baseline+MSRF+AdaRAGS1.07 7.880.438.31 1.72 3.41 0.21 8.59
Retrieval Fusion Strategy
Seq-GLF 1.42 9.72 0.40 7.64 1.91 3.73 0.19 6.98
Single-GF 1.24 8.91 0.41 7.95 1.82 3.69 0.20 7.12
Single-LF 1.12 8.64 0.41 8.01 1.76 3.52 0.20 7.84
Single-GLF1.07 7.88 0.43 8.31 1.72 3.41 0.21 8.59
MFN Input Setting
Textual Stream 1.35 8.94 0.39 6.76 1.87 3.71 0.17 6.52
Geometric Stream 1.29 7.92 0.41 7.86 1.77 3.61 0.19 7.91
Textual+Geometric Streams1.07 7.88 0.43 8.31 1.72 3.41 0.21 8.59
Retrieval Robustness
OOD Prompt 1.32 8.16 0.40 6.18 2.05 3.77 0.21 7.52
Random Retrieval 3.51 9.640.654.72 2.91 3.940.266.51
Ours1.07 7.880.438.31 1.72 3.410.218.59
deformation space. By injecting structural facial cues from
the data distribution into the editing process, MSRF reduces
ambiguity in the text-to-geometry mapping and improves
controllability while preserving multi-modal facial variations.
AdaRAGS further performs mask-guided semantic-
geometry alignment, which enforces region-aware consistency
between textual semantics and localized deformation fields.
Unlike global vertex-level supervision that is dominated by
large facial regions, this design explicitly reallocates learning
signals to semantically critical regions such as the eyes,
eyebrows, and mouth corners. This is not a simple weighting
heuristic, but a structured alignment constraint that improves
both reconstruction fidelity and editing controllability.
As summarized in Table IV, incorporating MSRF con-
sistently improves LVE, VS, and TIAS, confirming the ef-
fectiveness of retrieval-enhanced geometric priors. Adding
AdaRAGS further improves MRE, demonstrating stronger
local reconstruction fidelity in text-relevant masked regions.
The visual comparisons in Figure 5 further show that, in
both generation and editing tasks, the combination of MSRF
and AdaRAGS substantially reduces generation ambiguity and
better preserves fine-grained facial deformations specified by
long textual descriptions.
Retrieval Fusion Strategy.We design four fusion strate-
gies, as shown in Figure 6, to investigate how globally andlocally retrieved meshes should be integrated. Among them,
Seq-GLF follows a ReDream [26] and AMD [27]-style re-
trieval paradigm, where globally and locally retrieved meshes
are provided as sequential reference priors during generation.
In contrast, Single-GF, Single-LF, and Single-GLF aggregate
the retrieved priors into a single reference mesh using global-
only, local-only, and joint global-local fusion, respectively.
As reported in Table IV, using a single fused reference mesh
achieves more stable performance on fine-grained generation
and editing tasks, as it avoids the inter-reference inconsistency
and noise introduced by sequentially using multiple retrieved
meshes. Moreover, Single-GLF outperforms both Single-GF
and Single-LF, demonstrating that joint global-local fusion
effectively captures both holistic facial structure and text-
specified local details. These results validate the necessity of
our multi-scale fusion design: rather than directly relying on
sequentially retrieved references as in ReDream-style retrieval,
RAGMesh resolves conflicts among multi-scale priors and pro-
duces a compact, coherent geometric reference for controllable
facial generation and editing.
MFN Input Setting.Multi-scale retrieved meshes provide
geometric priors that complement textual semantics. To exam-
ine the effectiveness of the MFN input design, we evaluate
different input configurations, including text-only, retrieval-
only, and joint text-retrieval inputs. As shown in Table IV,

10
...
...
Global Retrieval
Local Retrieval
(a)
 (b)...
Mesh Fusion
(c) (d)...
...
Mesh Fusion
...Mesh 
EncoderMesh 
EncoderMesh 
EncoderMesh Fusion
...
Mesh 
EncoderLocal Retrieval Global RetrievalGlobal Retrieval
Local RetrievalSequence Output Single Frame Output
Fig. 6. Retrieval fusion. (a) Seq-GLF: Global-Local Fusion. (b) Single-GF: Global Fusion. (c) Single-LF: Local Fusion. (d) Single-GLF: Global-Local Fusion.
jointly using textual semantics and retrieval-based geomet-
ric priors consistently achieves the best performance across
LVE, MRE, VS, and TIAS. This demonstrates that textual
features provide semantic guidance, while retrieved meshes
offer explicit geometric constraints. Their combination enables
MFN to produce more accurate and semantically aligned facial
generation and editing results.
GT OOD PromptRandom RetrievalCorrect RetrievalEditing Task Generation TaskReference Mesh
Reference Mesh
GT OOD PromptRandom RetrievalCorrect RetrievalGenerate a young male 
with his mouth moving 
to the left, two corners 
of his mouth slightly 
raised, and his right eye 
closed.
Raise the corner of 
the character's right 
mouth.
Fig. 7. Ablation study on OOD prompts and retrieval quality.
Retrieval Robustness.We further evaluate the robustness
of RAGMesh during the retrieval process by using LLM to
rewrite the test text as out-of-distribution (OOD) samples and
comparing the performance with that of randomly generated
samples. As shown in Fig. 7, RAGMesh produces plausible
approximations under OOD prompts, indicating its robustness
to linguistic variations. When randomly retrieved meshes
are used, the editing accuracy decreases, but the generation
does not collapse, suggesting that the model is not overly
dependent on retrieval quality. In contrast, using correctly
retrieved meshes achieves the best performance, demonstrating
the importance of accurate retrieval for fine-grained text-
guided editing. We further quantitatively evaluate these effects
in Table IV. Results indicate that random retrieval improves
output diversity, whereas correct retrieval prioritizes fidelity,
highlighting a clear trade-off between diversity and accuracy
in retrieval-conditioned generation.
E. User Study Evaluation
User evaluation is critical for assessing both generation
quality and interaction performance of text-driven 3D face
editing models. For a comprehensive comparative analysis,
Fig. 8. User study for 3D facial generation and editing result.
32 participants were recruited to subjectively evaluate the
outputs of T2P [35], TADA [23], FaceG2e [13], ICE [12],
and our RAGMesh. Participants were instructed to complete
generation and editing tasks using each method, and rated the
results against four evaluation criteria: (1) semantic alignment,
(2) visual naturalness, (3) identity consistency, and (4) edit-
ing locality. As illustrated in Figure 8, RAGMesh achieves
consistently higher subjective scores across all metrics, which
validates its superior semantic controllability, geometric natu-
ralness, and identity preservation, as well as a more intuitive
and fluent user interaction experience. (Refer to SM for user
study questionnaire).
F . Feature-Level Analysis of RAG
Figure 9 visualizes text and geometric feature representa-
tions with MSRF and AdaRAGS. The baseline focuses on
dominant tokens in long text descriptions, which correspond
to regions with large geometric variations. While capturing
global deformations, it neglects semantically critical but subtle
local details. By contrast, MSRF introduces extra geometric
priors and strengthens the model’s representation capability.
Furthermore, AdaRAGS provides region-level guidance from
locally retrieved meshes, which reallocates attention in the
feature space and enables the model to better capture fine-
grained details ignored by the baseline.
G. Progressive Continuous Editing
For continuous facial editing, we first employ the gener-
ation model to produce an initial facial mesh. Since both
the generation and editing models operate in the FLAME
geometric space, the generated mesh is directly used as the

11
The face is that of a mid-dle 
aged woman, with the corners 
of her mouth rai-sed, cheeks 
bulging, eyes twit-ching, and 
eyebrows natu-rally raised. 
Her em-otions are expressed 
as h-appiness.
Baseline
Textual Stream Geometric StreamBaseline+MSRF
 Baseline+MSRF+AdaRAGS
Fig. 9. Visualization of intermediate feature representations as the MSRF and
AdaRAGS modules are introduced into the model.
Generation Editing 
Generate a young male with a 
round chin, thin and tall nose, 
and calm emotions.
Generate a male with facial weight 
gain, slightly upward corners of the 
mouth, open eyes, and maintain 
calm emotions.Move mouth to 
the left.Close left eye 
tightly.Chin contraction.
Lower lip inward 
curl.mouth shrug upper Facial slimming
Fig. 10. Progressive text-guided 3D face editing. Our method sequentially
applies multiple local editing instructions to the generated faces while pre-
serving identity consistency and overall facial geometry.
input for subsequent editing. Given a new editing instruction
in the text domain, RAGMesh retrieves the most relevant facial
geometries from the database as geometric priors to guide
the editing process. As illustrated in Figure 10, the retrieval-
conditioned editing pipeline enables progressive and consistent
facial modifications. Throughout the editing process, the iden-
tity of the character is well preserved, while fine-grained and
region-specific facial attributes can be precisely controlled.
H. Discussion and Limitation
RAGMesh adopts a geometry-centric design that does not
explicitly model texture or appearance. This design isolates
fine-grained geometric controllability and text–geometry align-
ment, while maintaining compatibility with existing rendering
pipelines, serving as a controllable geometric backbone that
complements full appearance-generation systems. However,
the separate training of generation and editing tasks restricts
parameter sharing and compromises cross-task consistency.
Consequently, generated meshes may not always fully adhere
to fine-grained textual descriptions, necessitating iterative re-
finement via the editing module (see SM for failure cases).
A promising direction is to unify generation and editing in
a single architecture with shared representations to enable
mutual supervision and enhance controllability and fidelity.V. CONCLUSION
In this paper, we first construct FaME-G2E, a large-scale
multimodal 3D facial dataset containing text–mesh pairs for
generation and text–blendshape pairs for localized editing.
Based on this dataset, we propose RAGMesh, a retrieval-
augmented framework that leverages geometric priors for
text-driven 3D facial generation and editing. Specifically,
MSRF retrieves semantically relevant reference meshes to
enhance geometric fidelity, while AdaRAGS leverages region-
aware supervision for fine-grained text-to-geometry alignment.
Extensive experiments demonstrate that RAGMesh achieves
superior performance over state-of-the-art methods in semantic
consistency, geometric fidelity, diversity, editing locality, and
inference efficiency. Future work will explore more diverse
facial assets, unified generation-editing models, and extensions
to broader 3D domains and multimodal interactions.
ACKNOWLEDGMENTS
Ethics Statement.This work involved human subjects in its
research. Approval of all ethical and experimental procedures
and protocols was granted by Biological and Medical Ethics
Committee of Beihang University (IF PROVIDED under
Application N0.BM20230165, and performed in line with
the Approval Letter from the Biological and Medical Ethics
Committee of Beihang University)
REFERENCES
[1] Z. Jiang, G. Lu, X. Liang, J. Zhu, W. Zhang, X. Chang, and H. Xu,
“3d-togo: Towards text-guided cross-category 3d object generation,” in
Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp.
1051–1059.
[2] C. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis,
S. Fidler, M. Liu, and T. Lin, “Magic3d: High-resolution text-to-3d
content creation,” inProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2023, pp. 300–309.
[3] Z. Liu, Y . Wang, X. Qi, and C. Fu, “Towards implicit text-guided
3d shape generation,” inProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2022, pp. 17 875–17 885.
[4] Y . Wang, Y . Zhuang, J. Zhang, L. Wang, Y . Zeng, X. Cao, X. Zuo, and
H. Zhu, “Tera: Rethinking text-guided realistic 3d avatar generation,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2025, pp. 10 686–10 697.
[5] R. Dey and V . N. Boddeti, “Generating diverse 3d reconstructions
from a single occluded face image,” inProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
1547–1557.
[6] Y . Liang, C. Zhang, J. Zhao, W. Wang, and X. Li, “Skull-to-face:
Anatomy-guided 3d facial reconstruction and editing,”IEEE Transac-
tions on Visualization and Computer Graphics, vol. 31, no. 9, pp. 6425–
6436, 2024.
[7] W. Song, X. Wang, Y . Jiang, S. Li, A. Hao, X. Hou, and H. Qin,
“Expressive 3d facial animation generation based on local-to-global
latent diffusion,”IEEE Transactions on Visualization and Computer
Graphics, vol. 30, no. 11, pp. 7397–7407, 2024.
[8] J. Zhang, K. Zhou, Y . Luximon, T.-Y . Lee, and P. Li, “Meshwgan: Mesh-
to-mesh wasserstein gan with multi-task gradient penalty for 3d facial
geometric age transformation,”IEEE Transactions on Visualization and
Computer Graphics, vol. 30, no. 8, pp. 4927–4940, 2023.
[9] Y . Zhuang, C. Ma, Y . Cheng, X. Cheng, J. Liao, and J. Lin, “Talkingeyes:
Pluralistic speech-driven 3d eye gaze animation,”IEEE Transactions on
Visualization and Computer Graphics, 2026.
[10] J. Ling, Z. Wang, M. Lu, Q. Wang, C. Qian, and F. Xu, “Semantically
disentangled variational autoencoder for modeling 3d facial details,”
IEEE Transactions on Visualization and Computer Graphics, vol. 29,
no. 8, pp. 3630–3641, 2022.

12
[11] L. Zhang, Q. Qiu, H. Lin, Q. Zhang, C. Shi, W. Yang, Y . Shi, S. Yang,
L. Xu, and J. Yu, “Dreamface: Progressive generation of animatable 3d
faces under text guidance,”ACM Transactions on Graphics, vol. 42,
no. 4, pp. 138:1–138:16, 2023.
[12] H. Wu, M. Zhao, Z. Hu, C. Fan, L. Li, W. Chen, R. Zhao, and X. Yu,
“ICE: interactive 3d game character facial editing via dialogue,” in
Proceedings of the 32nd ACM International Conference on Multimedia,
2025, pp. 3210–3223.
[13] Y . Wu, Y . Meng, Z. Hu, L. Li, H. Wu, K. Zhou, W. Xu, and X. Yu, “Text-
guided 3d face synthesis - from generation to editing,” inProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 1260–1269.
[14] E. Wood, T. Baltrusaitis, C. Hewitt, M. Johnson, J. Shen, N. Milosavl-
jevic, D. Wilde, S. J. Garbin, T. Sharp, I. Stojiljkovic, T. Cashman,
and J. P. C. Valentin, “3d face reconstruction with dense landmarks,”
inProceedings of the European Conference on Computer Vision, 2022,
pp. 160–177.
[15] K. Youwang, J. Kim, and T. Oh, “Clip-actor: Text-driven recommenda-
tion and stylization for animating human meshes,” inProceedings of the
European Conference on Computer Vision, 2022, pp. 173–191.
[16] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clarket al., “Learning transferable
visual models from natural language supervision,” inProceedings of the
International Conference on Machine Learning, 2021, pp. 8748–8763.
[17] T. Wang, B. Zhang, T. Zhang, S. Gu, J. Bao, T. Baltrusaitis, J. Shen,
D. Chen, F. Wen, Q. Chen, and B. Guo, “RODIN: A generative model
for sculpting 3d digital avatars using diffusion,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 4563–4573.
[18] S. Aneja, J. Thies, A. Dai, and M. Nießner, “Clipface: Text-guided
editing of textured 3d morphable models,” inACM SIGGRAPH 2023
Conference Proceedings, 2023, pp. 70:1–70:11.
[19] C. Ge, C. Xu, Y . Ji, C. Peng, M. Tomizuka, P. Luo, M. Ding,
V . Jampani, and W. Zhan, “Compgs: Unleashing 2d compositionality
for compositional text-to-3d via dynamically optimizing 3d gaussians,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2025, pp. 18 509–18 520.
[20] J. Zhu, Z. Chen, G. Wang, X. Xie, and Y . Zhou, “Segmentdreamer:
Towards high-fidelity text-to-3d synthesis with segmented consistency
trajectory distillation,” inProceedings of the IEEE/CVF International
Conference on Computer Vision, 2025, pp. 15 864–15 874.
[21] C. Zheng, Y . Lin, B. Liu, X. Xu, Y . Nie, and S. He, “Recdreamer:
Consistent text-to-3d generation via uniform score distillation,” inInter-
national Conference on Learning Representations, 2025.
[22] X. Han, Y . Cao, K. Han, X. Zhu, J. Deng, Y . Song, T. Xiang, and K. K.
Wong, “Headsculpt: Crafting 3d head avatars with text,” inAdvances in
Neural Information Processing Systems, 2023, pp. 4915–4936.
[23] T. Liao, H. Yi, Y . Xiu, J. Tang, Y . Huang, J. Thies, and M. J. Black,
“Tada! text to animatable digital avatars,” inInternational Conference
on 3D Vision. IEEE, 2024, pp. 1508–1519.
[24] J. Seo, S. Hong, W. Jang, I. H. Kim, M. Kwak, D. Lee, and S. Kim,
“Retrieval-augmented score distillation for text-to-3d generation,” in
Proceedings of the International Conference on Machine Learning, vol.
235, 2024, pp. 44 190–44 211.
[25] Z. Wang, C. Lu, Y . Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolific-
dreamer: High-fidelity and diverse text-to-3d generation with variational
score distillation,” inAdvances in Neural Information Processing Sys-
tems, 2023, pp. 8406–8441.
[26] J. Seo, S. Hong, W. Jang, I. H. Kim, M. Kwak, D. Lee, and S. Kim,
“Retrieval-augmented score distillation for text-to-3d generation,” in
Proceedings of the International Conference on Machine Learning,
2024, pp. 44 190–44 211.
[27] B. Jing, Y . Zhang, Z. Song, J. Yu, and W. Yang, “AMD: anatomical
motion diffusion with interpretable motion decomposition and fusion,”
inProceedings of the AAAI Conference on Artificial Intelligence, 2024,
pp. 2643–2651.
[28] M. Komeili, K. Shuster, and J. Weston, “Internet-augmented dialogue
generation,” inProceedings of the Annual Meeting of the Association
for Computational Linguistics, 2022, pp. 8460–8478.
[29] S. Zhou, U. Alon, F. F. Xu, Z. Jiang, and G. Neubig, “Docprompting:
Generating code by retrieving the docs,” inInternational Conference on
Learning Representations, 2023.
[30] Z. Wang, W. Nie, Z. Qiao, C. Xiao, R. G. Baraniuk, and A. Anandkumar,
“Retrieval-based controllable molecule generation,” inInternational
Conference on Learning Representations, 2023.[31] B. Yang, M. Cao, and Y . Zou, “Concept-aware video captioning:
Describing videos with effective prior information,”IEEE Transactions
on Image Processing, vol. 32, pp. 5366–5378, 2023.
[32] J. Xing, M. Xia, Y . Zhang, X. Cun, J. Wang, and T. Wong, “Codetalker:
Speech-driven 3d facial animation with discrete motion prior,” inPro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 12 780–12 790.
[33] S. Stan, K. I. Haque, and Z. Yumak, “Facediffuser: Speech-driven 3d
facial animation synthesis using diffusion,” inACM SIGGRAPH 2023
Conference Proceedings, 2023, pp. 13:1–13:11.
[34] O. Michel, R. Bar-On, R. Liu, S. Benaim, and R. Hanocka, “Text2mesh:
Text-driven neural stylization for meshes,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 13 482–13 492.
[35] R. Zhao, W. Li, Z. Hu, L. Li, Z. Zou, Z. Shi, and C. Fan, “Zero-
shot text-to-parameter translation for game character auto-creation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 21 013–21 023.
[36] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion:
Text-to-3d using 2d diffusion,” inInternational Conference on Learning
Representations, 2023.
[37] L. Yin, X. Wei, Y . Sun, J. Wang, and M. J. Rosato, “A 3d facial
expression database for facial behavior research,” inIEEE International
Conference on Automatic Face and Gesture Recognition, 2006, pp. 211–
216.
[38] X. Zhang, L. Yin, J. F. Cohn, S. J. Canavan, M. Reale, A. Horowitz,
P. Liu, and J. M. Girard, “Bp4d-spontaneous: a high-resolution spon-
taneous 3d dynamic facial expression database,”Image and Vision
Computing, vol. 32, no. 10, pp. 692–706, 2014.
[39] M. Wu, H. Zhu, L. Huang, Y . Zhuang, Y . Lu, and X. Cao, “High-
fidelity 3d face generation from natural language descriptions,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 4521–4530.
[40] J. Lin, H. Yin, W. Ping, P. Molchanov, M. Shoeybi, and S. Han, “VILA:
on pre-training for visual language models,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 26 679–26 689.
[41] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y . Wang, Z. Wang,
Y . Dong, M. Ding, and J. Tang, “Cogagent: A visual language model for
GUI agents,” inProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 14 281–14 290.
[42] R. Danecek, M. J. Black, and T. Bolkart, “EMOCA: emotion driven
monocular face capture and animation,” inProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
20 279–20 290.
[43] Y . Wang, P. Li, M. Sun, and Y . Liu, “Self-knowledge guided retrieval
augmentation for large language models,” inProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing,
2023, pp. 10 303–10 315.
[44] X. Huang, J. Kim, and B. Zou, “Unseen entity handling in complex
question answering over knowledge base via language generation,” in
Proceedings of theProceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing, 2021, pp. 547–557.
[45] H. Zhang, Z. Liu, C. Xiong, and Z. Liu, “Grounded conversation
generation as guided traverses in commonsense knowledge graphs,” in
Annual Meeting of the Association for Computational Linguistics, 2020,
pp. 2031–2043.
[46] M. Zhang, X. Guo, L. Pan, Z. Cai, F. Hong, H. Li, L. Yang, and
Z. Liu, “Remodiffuse: Retrieval-augmented motion diffusion model,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 364–373.
[47] C. Cao, Y . Weng, S. Zhou, Y . Tong, and K. Zhou, “Facewarehouse: A
3d facial expression database for visual computing,”IEEE Transactions
on Visualization and Computer Graphics, vol. 20, no. 3, pp. 413–425,
2014.
[48] C. Guo, Y . Mu, M. G. Javed, S. Wang, and L. Cheng, “Momask:
Generative masked modeling of 3d human motions,” inProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 1900–1910.
[49] X. Huang, H. Peng, D. Zou, Z. Liu, J. Li, K. Liu, J. Wu, J. Su, and P. S.
Yu, “Cosent: Consistent sentence embedding via similarity ranking,”
IEEE/ACM Transactions on Audio, Speech, and Language Processing,
vol. 32, pp. 2800–2813, 2024.
[50] X. Chu and T. Harada, “GAGAvatar: Generalizable and animatable
gaussian head avatar,” inAdvances in Neural Information Processing
Systems, 2024, pp. 57 642–57 670.