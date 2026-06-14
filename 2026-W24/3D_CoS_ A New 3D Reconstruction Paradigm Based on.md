# 3D-CoS: A New 3D Reconstruction Paradigm Based on VLM Code Synthesis

**Authors**: Yuhao Wang, Puyi Wang, Linjie Li, Zhengyuan Yang, Kevin Qinghong Lin, Yu Cheng

**Published**: 2026-06-09 06:46:29

**PDF URL**: [https://arxiv.org/pdf/2606.10478v1](https://arxiv.org/pdf/2606.10478v1)

## Abstract
Most recent 3D reconstruction and editing systems operate on implicit and explicit representations such as NeRF, point clouds, or meshes. While these representations enable high-fidelity rendering, they are fundamentally low-level and hard to control programmatically. In contrast, we propose and systematically evaluate a new 3D reconstruction paradigm, 3D Code Synthesis (3D-CoS), where 3D assets are constructed as executable Blender code, a programmatic and interpretable medium. To assess how well current VLMs can use code to represent 3D objects, we evaluate representative open-source and closed-source VLMs in code-based reconstruction under a unified protocol. We further introduce a suite of structured code-synthesis workflows, including blueprint-based planning, Retrieval-Augmented Generation (RAG) over Blender API documentation, few-shot geometric demonstrations, and a component-level Agent workflow for part-wise code generation. To demonstrate the unique advantages of this representation, we further evaluate localized text-driven modifications and compare our code-based edits with a point-cloud-based 3D editing baseline. Our study shows that code as a 3D representation offers strong controllability and locality, yielding stronger edit fidelity and better preservation of unedited regions in our targeted editing evaluation. Our work also analyzes the potential of this paradigm, delineates the current capability frontier of VLMs for programmatic 3D modeling, and highlights code synthesis as a promising direction for editable 3D reconstruction.

## Full Text


<!-- PDF content starts -->

3D-CoS: A New 3D Reconstruction Paradigm
Based on VLM Code Synthesis
Yuhao Wang*1, Puyi Wang*2, Linjie Li3, Zhengyuan Yang3
Kevin Qinghong Lin4, Yu Cheng2
1Shanghai Jiao Tong University
2The Chinese University of Hong Kong
3Microsoft4University of Oxford
*Equal contribution.
Abstract
Most recent 3D reconstruction and editing systems operate on implicit and explicit
representations such as NeRF, point clouds, or meshes. While these representations
enable high-fidelity rendering, they are fundamentally low-level and hard to control
programmatically. In contrast, we propose and systematically evaluate a new
3D reconstruction paradigm, 3D Code Synthesis (3D-CoS), where 3D assets are
constructed as executable Blender code, a programmatic and interpretable medium.
To assess how well current VLMs can use code to represent 3D objects, we evaluate
representative open-source and closed-source VLMs in code-based reconstruction
under a unified protocol. We further introduce a suite of structured code-synthesis
workflows, including blueprint-based planning, Retrieval-Augmented Generation
(RAG) over Blender API documentation, few-shot geometric demonstrations, and
a component-level Agent workflow for part-wise code generation. To demonstrate
the unique advantages of this representation, we further evaluate localized text-
driven modifications and compare our code-based edits with a point-cloud-based
3D editing baseline. Our study shows that code as a 3D representation offers strong
controllability and locality, yielding stronger edit fidelity and better preservation
of unedited regions in our targeted editing evaluation. Our work also analyzes the
potential of this paradigm, delineates the current capability frontier of VLMs for
programmatic 3D modeling, and highlights code synthesis as a promising direction
for editable 3D reconstruction.
1 Introduction
Recent breakthroughs in large vision-language models (VLMs) have led to remarkable progress in
multimodal understanding, logical reasoning, and tool usage. These models have shown the ability to
operate within a “perception, reasoning, planning, and execution" loop, and automatically generate
executable code to accomplish complex tasks [ 14,26,28]. This capability suggests a new path for
3D: instead of recovering geometry purely as meshes, point clouds, or implicit fields, we can generate
executable programs that reconstruct 3D assets inside 3D engines (e.g., Blender [ 6], Unity [ 39],
Unreal [ 13]). Code as a 3D representation brings interpretability, editability, and compositional
control. 3D components are constructed in an explicit and parameterized manner, and are verifiable
by execution. Moreover, the mature APIs of 3D engines [ 5,12,40] enable full programmatic control
over creation and rendering, providing a robust substrate for automation [1].
Several recent works have explored the feasibility of using code to generate and edit 3D assets, and
two recent lines of work motivate our study. LL3M [ 29] demonstrates text-driven 3D asset creation
by coordinating agents that write Blender scripts, evidencing that code can serve as a powerful repre-
sentation for modeling geometry, layout, and appearance. BlenderGym [ 16] introduces a benchmark
Preprint.arXiv:2606.10478v1  [cs.CV]  9 Jun 2026

VLMs
 Executable Code
 3D Assets & Evaluation
Reasoning
& PlanningCompilation
& Rendering
High-Fidelity Reconstruction
（Single Image → 3D Code）Precise & Controllable Editing
Code-Generated
CodeParameter
UpdateInstruction:
Lift the lid slightly above 
the cup.
Instruction:
Remove the small part from 
the top & Make this lamp taller. 
Ours (Code-based)Better Instruction 
Following
Superior 
Edit Fidelity
Point-cloud-based BaselineFigure 1:An overview of our 3D code modeling paradigm.The top workflow summarizes our core
process: code synthesis via VLMs, and its subsequent evaluation. Our work treats code as a unified
representation for 3D assets. (Left) We demonstrate its capability inreconstruction, generating
high-fidelity objects from a singleimage. (Right) We highlight its advantages inediting, where
code-driven edits achieve superior fidelity compared to a point-cloud-based 3D editing baseline.
that tasks VLM systems with code-based 3D scene editing and shows that state-of-the-art models
can comprehend programmatic code and further make targeted code-level modifications. Together,
these works validate the feasibility of “code for 3D" while revealing a gap in image-conditioned
reconstruction and in systematic evaluation specific to 3D reconstruction. In reconstruction, the
image input is essential: it supplies silhouette constraints, object pose, and disambiguates topology
and fine details that text alone cannot specify. Equally important is a standardized pipeline to eval-
uate code-as-representation under image conditioning. We fill this gap with a systematic study of
image-conditioned, code-based 3D reconstruction, accompanied by a unified evaluation protocol
across multiple VLM families.
We focus on the image →code→3D setting and address two fundamental questions: Why use
code as the representation? and What is the reconstruction ceiling? First, unlike purely implicit
(e.g., NeRF, Gaussian Splatting) or low-level explicit representations, code enables fine-grained
control and reliable iteration through high-level logic. Second, regarding the ceiling, we observe that
the vast majority of human-made 3D assets originate from programmatic workflows. Datasets like
Fusion 360 [ 44] capture these parametric design histories, and even large-scale mesh collections like
ModelNet [ 48] and ShapeNet [ 7] are derived from CAD models. Therefore, the theoretical ceiling of
this paradigm is the recovery of the original executable design program used by human modelers.
In this work, we propose a novel paradigm for 3D reconstruction using programmatic code on the
Blender platform (Figure 1). This code-based representation makes results easier to edit, offers finer-
grained control, and is more interpretable than standard mesh formats. To systematically evaluate the
capabilities of modern VLMs in this setting, we introduce a code-based reconstruction benchmark
that evaluates representative open- and closed-source VLMs on single-image reconstruction under
a unified evaluation protocol, analyzes four direct prompting paradigms together with an Agent
workflow evaluated on closed-source models, and compares their results to representative external 3D
reconstruction baselines. We report standard 3D geometric metrics and complementary 2D metrics
computed from the same camera view used to generate the input condition image. These 2D metrics
evaluate how well a model reconstructs the observed view when it receives only a single-view input.
Beyond reconstruction, we include a code-based editing protocol to expose the unique strengths
of programmatic control (e.g., targeted parameter changes, retention of unedited areas) relative to
point-cloud-based 3D editing pipelines, further highlighting the potential of code as an editable 3D
representation.
Our main contributions are threefold:
•We introduce a benchmark protocol and metrics suite for Blender-code reconstruction, and
use it to evaluate representative open- and closed-source VLMs in terms of their code-based
3D reconstruction capabilities.
2

•We evaluate four direct prompting paradigms and a component-level Agent workflow for
VLM-based code synthesis, showing that structured workflows achieve promising recon-
struction quality and that the best-performing Gemini configurations approach InstantMesh
on specific metrics.
•We demonstrate that the proposed code-based representation offers substantial advantages
for 3D editing, and empirically validate its benefits against a point-cloud-based 3D editing
baseline through targeted editing evaluations.
2 Related Work
Traditional 3D Reconstruction Representations.Existing methods mainly develop along two lines:
(i)Neural or continuous rendering representationssuch as neural-radiance-field-based methods [ 30,
33,23,42,41], point-based 3D Gaussian Splatting methods [ 22,9,51,47], and other approaches that
learn a latent space and decode it into neural shape representations [ 53,21,24]. This family excels
in visual fidelity, but typically offers limited precise control, lacks interoperability with standard
graphics pipelines, and often relies on heavy optimization or bespoke training. (ii)Explicit shape
representations(point clouds, voxels, meshes) are more amenable to geometric measurement and
integration with existing engines, and have been extensively studied [ 8,27,19,52]. However, they
lack shared high-level primitives and constraints, making automated control and cross-category,
generalizable editing challenging.
Code Based 3D Representations.Regarding Domain-Specific Language (DSL) approaches, 3D
Shape Programs [ 38] encode repeated and symmetric structures as programs. ShapeAssembly [ 20]
designs DSLs specifically for 3D shape structures, constructing hierarchical and reconfigurable
shape programs. These approaches often rely on custom DSLs whose primitive vocabularies and
domain-specific grammars constrain free-form geometry. In the CAD-oriented line of code-based
representations, DeepCAD [ 46] treats CAD operation sequences (sketch, extrude, etc.) as program se-
quences. Later text- and point-to-CAD methods [ 50] similarly exploit parametric operation sequences
to obtain editable designs. These methods typically rely on custom DSLs or commercial CAD envi-
ronments, which are best suited to constructing regular geometries for downstream manufacturing
rather than the free-form, animation-oriented content creation enabled by tools like Blender.
Large Models for 3D Generation and Editing via Code.The success of Large Models (LMs) in
leveraging code to solve problems [ 14] has inspired exploration into using LMs to generate code for
manipulating 3D objects. BlenderAlchemy [ 18] generates materials in Blender for existing geometry.
SceneCraft [ 17] retrieves 3D assets and employs an LLM to organize them into a coherent spatial
layout. LL3M [ 29] generates 3D assets from text guidance, incorporating geometry and appearance
attributes and BlenderMCP [ 1] uses a single LLM calling Blender functions via the Model Context
Protocol [ 2]. BlenderGym [ 16] utilizes VLMs for 3D scene reconstruction through code editing.
MeshCoder [ 10] fine-tunes an LLM to translate 3D point clouds into editable Blender scripts.
However, these methods either do not directly compare code with traditional 3D representations in
terms of reconstruction and editing behavior, or do not systematically evaluate image-conditioned 3D
reconstruction through executable code.
In contrast, we demonstrate the benefits of code compared to traditional representations and provide
a comprehensive evaluation of the capability of current VLMs to reconstruct and edit 3D objects
directly from image inputs.
3 Method
3.1 3D Reconstruction Paradigms for Code Synthesis
Problem setup.Given a single image I, we generate an executable Blender Python script whose
rendered 3D object matches the target object as closely as possible. To study how different code
synthesis paradigms affect reconstruction, we compare four direct prompting paradigms,Single-call,
Planning,RAG, andFew-shot, together with anAgent Paradigmfor component-level generation. We
also design a text-conditional reconstruction variant described below.
Single-call Paradigm.We prompt the VLM once to reconstruct the object and output a complete
script, requiring the script to encapsulate all reconstruction logic in one callable function with explicit
3

Input
Vision -Language
Models
In-context Visual Reference & Code
Blueprint
(JSON  format)
 Output 3D Model
Agent Workflow
Part-wise Generation
RAG
Qdrant  →Retrieved APIsFigure 2:Overview of our VLM-based 3D reconstruction pipeline.Solid arrows show the image-
to-code-to-3D workflow; dashed arrows show optional context from visual-code examples, blueprints,
RAG-based API retrieval, and part-wise Agent generation. Appendix A.4 provides example artifacts.
parameters as input. This is the fastest setting, but it often misses fine details or mis-specifies
parameters for complex shapes.
Planning Paradigm.We decouplewhat to buildfromhow to implement it: Stage 1 infers a
quantitative blueprint fromI, and Stage 2 generates code conditioned on that blueprint.
Stage 1: quantitative blueprint.The VLM defines an object-level base dimension (e.g., overall width
or height) and sets it to 1.0, then specifies the dimensions of all parts in Bas ratios to this base. We
also require a feasibility check: under occlusion or perspective ambiguity, the model may propose
minimal adjustments to ensure structural stability. See Appendix A.4 for a blueprint example.
Stage 2: code generation.Conditioned on (I,B) , the VLM generates a script that preserves the
numeric structure in B. The image is used mainly for non-parametric details (e.g., complex curves),
with only minimal edits allowed to resolve infeasible geometry. Compared to Single-call, Planning
improves interpretability via an explicit blueprint and a clean separation between geometry and API
usage.
RAG Paradigm.Relying solely on internal knowledge can lead to hallucinated API calls and make
the model overlook appropriate but long-tail functions, so we retrieve Blender 4.4 documentation as
external context during code synthesis. We crawl the Blender 4.4 docs using the Sphinx inventory [ 37]
and sphobjinv [ 36], extract functions, signatures, and parameter semantics from bpy,bmesh , and
mathutils , and store each entry as a standardized JSON item D(Listing 2). The resulting database
covers 1,683 pages and 21,102 functions, which we embed and index in Qdrant [ 34] for hybrid
retrieval.
We first infer the same quantitative blueprint Bas in the Planning paradigm. For each component in B,
the VLM generates API-oriented queries with module preferences and keyword constraints, mapping
semantic parts to primitives and operations such as bevel or solidify (Listing 3). We retrieve Top- k
(k= 8 ) documentation chunks, consolidate them into background knowledge Kin acomponent →
candidate-API listschema (Listing 4), and generate the final script conditioned on(I,B,K).
Few-shot Paradigm.To probe in-context learning capabilities of VLMs, we prepend a small set of
exemplary pairs (Iex,Sex)to the task instruction. We generate the exemplar scripts with Gemini 3 Pro
on samples disjoint from ModelNet10 and manually curate them to ensure consistent blueprint-style
reasoning and correct API usage. By observing exemplary pairs, the model is guided to implicitly
learn the underlying geometric reasoning patterns and syntactic standards, applying them to the target
reconstruction task.
Agent Paradigm.Besides the four direct prompting paradigms above, we study an agent paradigm
that introduces a component-level reconstruction workflow. Given an input image I, the agent first
constructs a structured object plan P={p i}N
i=1, where each part pidescribes a major component
together with its approximate geometry and spatial placement. Conditioned on P, the model
synthesizes bpycode for each part separately, so local structure is specified explicitly rather than
being produced in a single global pass. The generated code is executed during construction, and when
a part program fails, the agent revises that part and executes the revised program before continuing.
4

After all parts are completed, the part-level programs are assembled into a full reconstruction, yielding
the final 3D model. This design is particularly suited to improving local geometric fidelity and more
balanced recovery of fine object components.
Variant: From Reconstruction to Editing.This variant extends reconstruction to incorporate an
edit intent Tedit. Reusing the Planning workflow, Stage 1 predicts an edited blueprint Beditfrom
(I,T edit), and Stage 2 generates the final script conditioned on (I,B edit). This variant demonstrates
the flexibility of our paradigm by unifying reconstruction and editing into a single generation process.
3.2 Reconstruction Benchmark
We introduce a unified benchmark for code-based 3D reconstruction that combines explicit geometric
registration with complementary 3D and 2D evaluation. The protocol is motivated by a characteristic
ambiguity of executable 3D code: a generated asset may preserve the target structure while differing in
global scale or pose convention. The benchmark therefore factors out such nuisance transformations
while retaining sensitivity to geometric fidelity and view-consistent reconstruction quality.
Registration protocol.
Before metric computation, we apply an upright-preserving similarity alignment. Global scale,
translation, and yaw are optimized, whereas pitch and roll are fixed. We first enumerate candidate
yaw angles and score them with a robust symmetric surface-distance objective, then optimize the
best candidates with iterative closest point updates under the same yaw-only constraint. The selected
transform is applied to the generated mesh for all quantitative evaluation.
Benchmark scope.The benchmark is instantiated on a controlled single-view reconstruction setting
derived from ModelNet10 and includes an easy/hard split for analyzing sensitivity to structural
complexity. We further extend this setting with a text-conditional reconstruction variant, denoted
ModelNet10-V , to qualitatively assess generation conditioned jointly on an input image and an
edit instruction. For ModelNet10 reconstruction, quality is assessed through complementary 3D
metrics of geometric accuracy and structural coverage together with 2D metrics computed from the
input-condition camera view. Detailed dataset construction and metric definitions are provided in the
experimental setup.
3.3 3D Editing Paradigm with Code Modification
A key reason we choose code as the representation for 3D shapes is the programmatic controllability it
provides for subsequent editing operations. When a 3D object generated by code needs to be modified,
we can make adjustments directly at the code level, leveraging the VLM’s powerful comprehension
and reasoning capabilities.
Code-based 3D Editing Paradigm.We edit programmatic 3D assets by taking a source script Ssrc
and a text instruction Tedit. The VLM parses Ssrc, localizes the code relevant to Tedit, and outputs
an edited scriptS dstwith targeted changes.
Construction of Localized 3D Editing Assets.We build on theBlendNetdataset [ 11] by selecting 55
representative objects and manually writing a high-quality and specific edit instruction for each. The
resulting dataset,BlendNet-E, contains triplets (source scripti,source descriptioni,edit instruction i).
The quantitative evaluation protocol for this task is presented in Section 4.2.
4 Experiments
4.1 3D Reconstruction Evaluation
Our goal is to synthesize executable bpycode from a single input image and reconstruct a 3D object
that matches the target as closely as possible. We evaluate this setting under the reconstruction
benchmark described in Section 3.2 and report results on ModelNet10 for four direct prompting
paradigms, namely Single-call, Planning, RAG, and Few-shot, across both open-source and closed-
source VLMs. We additionally study an Agent paradigm on the three closed-source models.
5

GT
Qwen
Bed Desk Table Sofa
 Bathtub
Geminio3
Dresser
Figure 3:Code-based reconstruction on ModelNet10.All inputs come from the ModelNet10
dataset and contain only geometry; the cyan shading is for visualization only.
4.1.1 Experimental Setup
Datasets and Benchmark Setting.We use ModelNet10 [ 48] under a controlled rendering pro-
tocol. Our reconstruction benchmark contains 100 objects sampled from ModelNet10 across its
10 categories. Each object is normalized and rendered from eight evenly spaced viewpoints on a
sphere of radius 1.76, with depth and normal maps also generated for analysis. A human annotator
selects the most informative RGB view as the single-image input. This selected view is fixed as the
input condition image and as the camera view for 2D evaluation. To analyze sensitivity to structural
complexity, we additionally split ModelNet10 intoeasyandhardsubsets with one annotator and one
verifier; details are provided in Section A.1.
Models and Baselines.We evaluate code-based reconstruction on three open-source families,
InternVL3.5 38B [ 43], LLaV A OneVision Qwen2 72B [ 25], and Qwen2.5 VL 72B Instruct [ 4],
and three closed-source families, Claude Sonnet 4 [ 3], o3 [ 32], and Gemini 3 Pro [ 15]. For all six
models, we report Single-call, Planning, RAG, and Few-shot. For the three closed-source models, we
additionally report Agent. We restrict Agent evaluation to the three closed-source models, where the
multi-stage generation and self-correction workflow can be executed reliably in our current setup. We
compare against Unique3D [ 45], InstantMesh [ 49], and MeshCoder [ 10]. All VLM-based methods
take a single RGB view as input, whereas MeshCoder uses ground-truth 3D point clouds.
Registration and Metrics.We follow the reconstruction benchmark in Section 3.2 and apply the
registration protocol in Section 3.2 before metric computation. We report four 3D metrics, Chamfer
Distance (CD), F-score at 5% (F@5%), Spatial Balanced Recall at 2% (SBR@2%), and Normal
Consistency at 5% (NC@5%), together with three input-view 2D metrics: Silhouette IoU (Sil IoU),
depth NRMSE, and normal MAE. Detailed metric definitions are provided in Appendix A.1. For the
external baselines and the closed-source VLM configurations emphasized in the main comparisons,
we further quantify sampling uncertainty over object identities with 1,000-sample nonparametric
bootstrap 95% confidence intervals. These intervals are computed from evaluated object-level metric
records and reported in Appendix Table 6.
Reconstruction Variant.To demonstrate the text-conditional reconstruction variant, we derive a
companion test set from the same ModelNet10 assets and use the same human-selected informative
views as inputs. For each image, we use GPT-4o [ 31] to generate a high-level edit instruction tailored
to the depicted object, yielding 100 triplets of (image, edit instruction, source .blend ). We denote
this set byModelNet10-V. Since the set does not contain edited ground-truth shapes, we use it for
qualitative analysis rather than quantitative reconstruction metrics.
6

Table 1:Reconstruction on ModelNet10.We report 3D metrics (CD, F@5%, SBR@2%, NC@5%)
and 2D metrics (Sil IoU, NRMSE, MAE); arrows indicate preferred directions. Metrics average
successful outputs; open-source failure rates are reported in Appendix Table 4. Best and second-best
values are highlighted in green and blue, within each comparison block.
Model ParadigmModelNet10
3D Metrics 2D Metrics
CD↓F@5%↑SBR@2%↑NC@5%↑Sil IoU↑NRMSE↓MAE↓
Traditional mesh baselines
Unique3D [45] – 0.0580 0.5679 0.2304 0.6236 0.5231 0.1388 0.4999
InstantMesh [49] –0.0191 0.9060 0.6443 0.8095 0.8068 0.0537 0.2110
Code-based baseline
MeshCoder [10] –0.0442 0.7053 0.4529 0.6544 0.6091 0.1250 0.3893
Open-source VLM families
LLaV A-OneVision-Qwen2-72B [25]Sin. 0.0674 0.5186 0.2603 0.56970.60180.1963 0.5460
Pla. 0.0704 0.5032 0.2757 0.5378 0.4753 0.1715 0.5188
RAG 0.0698 0.5137 0.2750 0.5838 0.4243 0.1504 0.4919
Few. 0.0699 0.4909 0.2756 0.5442 0.5093 0.1733 0.5423
InternVL3.5-38B [43]Sin. 0.0673 0.5267 0.2614 0.5800 0.5236 0.1618 0.4984
Pla. 0.0642 0.5584 0.2956 0.5683 0.4983 0.1606 0.5048
RAG 0.0663 0.5458 0.2770 0.5721 0.4855 0.1492 0.4832
Few. 0.0648 0.5668 0.3152 0.5760 0.5326 0.1656 0.5158
Qwen2.5-VL-72B-Instruct [4]Sin.0.0593 0.6038 0.34820.5849 0.5920 0.1558 0.4817
Pla. 0.0635 0.5995 0.33830.60240.5457 0.14580.4310
RAG 0.0665 0.5571 0.3004 0.5952 0.50670.14410.4421
Few.0.0492 0.6702 0.4104 0.6197 0.6167 0.1289 0.4269
Closed-source VLM families
Claude Sonnet 4 [3]Sin. 0.0405 0.7178 0.4435 0.6864 0.6333 0.1057 0.3307
Pla. 0.0419 0.7069 0.4541 0.6526 0.6264 0.1208 0.3733
RAG 0.0368 0.7394 0.4814 0.6641 0.6660 0.1171 0.3773
Few. 0.0374 0.7497 0.4818 0.6745 0.6561 0.1060 0.3491
Agt. 0.0435 0.7140 0.4990 0.6807 0.6262 0.1188 0.3652
o3 [32]Sin. 0.0608 0.5887 0.3185 0.6091 0.5469 0.1410 0.4340
Pla. 0.0328 0.7879 0.5312 0.6964 0.6921 0.0983 0.3085
RAG 0.0466 0.6642 0.3732 0.6536 0.5656 0.1221 0.3891
Few. 0.0318 0.7949 0.5306 0.7174 0.6934 0.0969 0.2939
Agt. 0.0510 0.6556 0.4058 0.6614 0.5540 0.1390 0.4006
Gemini 3 Pro [15]Sin. 0.0356 0.7624 0.4986 0.7134 0.6591 0.0998 0.3032
Pla. 0.0280 0.8277 0.5820 0.7190 0.7275 0.0900 0.2861
RAG 0.02510.8573 0.6199 0.7374 0.75850.08290.2681
Few.0.0249 0.85320.61060.74270.75170.0748 0.2550
Agt.0.02480.85130.62630.73320.7535 0.08260.2693
4.1.2 Main Results
Table 1 evaluates image-conditioned Blender-code reconstruction on ModelNet10 under four direct
prompting paradigms and the Agent workflow, together with three external 3D reconstruction
baselines. Representative reconstructions are shown in Figure 3, illustrating the geometry-only
ModelNet10 setting used throughout this evaluation.
Code-based reconstruction is competitive with several reference baselines under different input
conditions.Gemini 3 Pro with Few-shot already outperforms Unique3D and MeshCoder on all
reported metrics. For MeshCoder, it reduces CD from 0.0442 to 0.0249, and improves F@5%,
SBR@2%, and NC@5% from 0.7053/0.4529/0.6544 to 0.8532/0.6106/0.7427. This comparison
is notable as all VLM-based methods use a single RGB view, while MeshCoder is conditioned on
ground-truth point clouds. Compared with InstantMesh, Gemini 3 Pro with Agent narrows the gap
on geometry-centric metrics, reaching CD 0.0248 and SBR@2% 0.6263, compared with 0.0191 and
0.6443 for InstantMesh.
Closed-source VLMs define the current frontier among code-based VLM reconstructions.
Across the shared paradigms, closed-source VLMs define the frontier of code-based VLM perfor-
7

Make it more square  
and a dd a flat base…
Add a second 
drawer below …
Change the base to 
single leg…
Source Variant
Ours BlendedPC InputConnect X -shaped 
strips of the legs…Change to a 
triangular coneFigure 4:Reconstruction variant & Code editing results.(Left) Examples of our text-conditional
reconstruction variant, which modifies an object based on a source image and a textual instruction.
For the reconstruction variant, Source denotes the original object rather than an edited ground truth.
(Right) A direct comparison of our code editing method against the BlendedPC [ 35] baseline. Code-
driven editing shows stronger edit fidelity while better preserving overall visual quality. A brief
instruction is provided for each example in the figure; the full instructions are listed in Section A.3.
mance. Under Few-shot, Gemini 3 Pro achieves CD 0.0249 and F@5% 0.8532, while the best
open-source counterpart, Qwen2.5-VL-72B-Instruct, reaches 0.0492 and 0.6702. The same separa-
tion is visible in SBR@2% and 2D metrics. This pattern suggests that current reconstruction quality
is more constrained by VLM capability than by the expressivity of executable code alone.
Auxiliary structure interacts with model capability.The prompting paradigms do not form a
universal ranking. Among open-source models, Qwen2.5-VL benefits most clearly from Few-shot
prompting, reducing CD from 0.0593 to 0.0492 and improving F@5% from 0.6038 to 0.6702 over
Single-call, whereas LLaV A-OneVision and InternVL show less consistent gains. Closed-source
models show different preferences: o3 benefits most from Planning and Few-shot, Claude Sonnet 4 is
stronger with RAG/Few-shot than with Planning on most metrics, and Gemini 3 Pro remains strong
under all structured paradigms. These comparisons suggest that plans, retrieved APIs, and exemplars
provide complementary sources of structure rather than a fixed hierarchy.
Agent favors part-balanced recovery.Agent adds explicit part-wise generation, intermediate
execution, and final assembly. This emphasizes component coverage, which is directly reflected
by SBR@2%. Agent gives Claude Sonnet 4 its best SBR@2%, and gives Gemini 3 Pro its best
CD and SBR@2%. It is not uniformly best because separately generated parts must still agree in
scale, relative placement, and inter-part consistency. Such coordination errors can affect global
surface agreement or visible-view alignment, explaining why Gemini 3 Pro with RAG obtains the
best F@5% and Sil IoU, Gemini 3 Pro with Few-shot obtains the best NC@5%, NRMSE, and MAE,
and o3-Agent remains below o3-Planning and o3-Few-shot on most metrics. Agent is therefore most
useful when better part coverage outweighs the cost of multi-stage assembly.
The easy/hard split supports the role of structural complexity.The easy/hard split captures
structural complexity: easy objects have fewer parts, more regular structures, and milder curvature
transitions, while hard objects contain more parts, irregular topology, or high-curvature transitions.
The complete split in Appendix Table 5 includes the external 3D baselines and all evaluated closed-
source VLM paradigms. Within that table, all five Gemini 3 Pro paradigms show higher Sil IoU
and lower NRMSE/MAE on easy shapes, suggesting that current code-synthesis systems are most
reliable when visual structure maps cleanly to parts, parameters, and transformations.
Reconstruction variants as a bridge to editing.To demonstrate paradigm flexibility, we perform
text-conditional reconstruction (Figure 4, left), where models modify the source object based on
textual instructions during generation. Using the o3 model onModelNet10-V, we qualitatively observe
examples in which our approach interprets edit intents and applies targeted geometric modifications
(additional examples in Figure 7), suggesting the potential of code-based representations for semantic
editing and motivating the following editing evaluation.
8

Method CLIP sim↑CLIP dir↑
BlendedPC 0.0142 0.2017
Ours P 0.0578 0.2499
Ours A 0.0408 0.2469
Table 2: CLIP-based similarity and direction scores.
Ours Pis tested on the lamp and table categories com-
pared with BlendedPC, while Ours Ais tested on the
entireBlendNet-E.Method Inst.↑Pres.↑
BlendedPC 1.90 2.45
Ours 4.37 4.30
Table 3: User study from 12 volun-
teers, scores (1–5) for BlendedPC vs.
ours. Inst. = instruction following;
Pres. = unedited preservation.
Overall, these results position executable code as a competitive representation for single-image 3D
reconstruction, with the current frontier shaped by VLM capability and by structural workflows that
turn visual evidence into reliable, editable programs.
4.2 3D Code Editing Evaluation
We evaluate code editing onBlendNet-E: given a source bpyscript and a text instruction, the model
edits the script to modify the asset. We compare with BlendedPC [ 35] on its suggested lamp/table
subset and evaluate our o3-based method on the full dataset. Following BlendedPC’s Edit Fidelity
protocol, we render four orthogonal views and report average CLIP simfor text alignment and
CLIP dir, computed as in BlendedPC, for edit correctness.
4.2.1 Main Results
Our method obtains stronger editing scores on the matched subset.On the lamp/table subset, our
code-based editing achieves CLIP sim= 0.0578 andCLIP dir= 0.2499 , while BlendedPC records
0.0142 and 0.2017 , respectively, as shown in Table 2 and in Figure 4. This corresponds to a 4.1 ×
higher score in CLIP simand a +23.9% gain in CLIP dir, indicating both stronger text-image alignment
and more faithful execution of the intended edit. In addition, a user study further favors our method
in “Instruction following" and “Preservation of unedited regions", as shown in Table 3.
Generalization to other categories.On the fullBlendNet-Eset, Ours Aretains a comparable CLIP dir
but shows a lower CLIP simthan Ours P, indicating consistent semantic editing across a broader set
of shapes with weaker text-image similarity. We use Ours Pfor the matched lamp/table comparison
against BlendedPC, while Ours Aevaluates generalization beyond that subset. Additional examples
in Appendix Figures 10 and 11 further show that code-based edits implement targeted geometric
changes while preserving unmodified parts.
4.3 Limitations
Despite promising results, executable code is not yet a drop-in replacement for dedicated mesh
reconstruction models: as shown in Table 1, our best Agent result narrows but does not close the gap
to InstantMesh [ 49] on aggregate reconstruction metrics. Our experiments also show that part-wise
Agent generation improves coverage but can introduce scale, placement, or inter-part inconsistencies;
complex assemblies may still lead to missing dependencies, broken object relations, or non-executable
code. Improving code-centric VLMs and their self-verification mechanisms is therefore important for
making executable 3D representations both accurate and editable.
5 Conclusion
This work systematically evaluates executable Blender code as a representation for single-image
3D reconstruction. Across open- and closed-source VLMs, we study how prompting structure,
retrieval, few-shot examples, and a component-level Agent workflow affect reconstruction quality.
Results show that VLM-driven code synthesis recovers structured geometry, is competitive with the
point-cloud-conditioned code baseline, and approaches representative mesh reconstruction baselines
on selected metrics. We also show that code-based editing enables targeted semantic and geometric
changes with stronger preservation than a point-cloud editing baseline, supporting executable code as
a promising representation for editable 3D reconstruction.
9

References
[1]Siddharth Ahuja and BlenderMCP Contributors. Blendermcp - blender model context protocol
integration, 2025. URLhttps://github.com/ahujasid/blender-mcp.
[2]Anthropic PBC. Introduction to model context protocol, 2024. URL https://www.
anthropic.com/news/model-context-protocol.
[3]Anthropic PBC. Claude Sonnet 4, 2025. URL https://www.anthropic.com/claude/
sonnet. Product page.
[4]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang
Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen
Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report.
arXiv preprint arXiv:2502.13923, 2025.
[5]Blender Foundation. Blender Python API reference, 2025. URL https://docs.blender.
org/api/current/index.html. Online; accessed 2025-09-18.
[6]Blender Online Community. Blender: Open-source 3d creation suite. https://www.blender.
org, 2025. Version 4.4 as used in this work.
[7]Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo
Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu.
Shapenet: An information-rich 3d model repository, 2015. URL https://arxiv.org/abs/
1512.03012.
[8] Zhiqin Chen, Vladimir G Kim, Matthew Fisher, Noam Aigerman, Hao Zhang, and Siddhartha
Chaudhuri. Decor-gan: 3d shape detailization by conditional refinement. InProceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 15740–15749, 2021.
[9]Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting.
InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
21401–21412, 2024.
[10] Bingquan Dai, Li Ray Luo, Qihong Tang, Jie Wang, Xinyu Lian, Hao Xu, Minghan Qin,
Xudong Xu, Bo Dai, Haoqian Wang, et al. Meshcoder: Llm-powered structured mesh code
generation from point clouds.arXiv preprint arXiv:2508.14879, 2025.
[11] Yuhao Du, Shunian Chen, Wenbo Zan, Peizhao Li, Mingxuan Wang, Dingjie Song, Bo Li, Yan
Hu, and Benyou Wang. Blenderllm: Training large language models for computer-aided design
with self-improvement, 2024. URLhttps://arxiv.org/abs/2412.14203.
[12] Epic. Unreal engine c++ api reference, 2025. URL https://dev.epicgames.com/
documentation/en-us/unreal-engine/API.
[13] Epic. Unreal engine, 2025. URLhttps://www.unrealengine.com/.
[14] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan,
and Graham Neubig. Pal: Program-aided language models, 2023. URL https://arxiv.org/
abs/2211.10435.
[15] Google DeepMind. Gemini 3 Pro, 2025. URL https://blog.google/products/gemini/
gemini-3/. Official blog post (Nov 18, 2025).
[16] Yunqi Gu, Ian Huang, Jihyeon Je, Guandao Yang, and Leonidas Guibas. Blendergym: Bench-
marking foundational model systems for graphics editing. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages 18574–18583, 2025.
[17] Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid,
and Alireza Fathi. Scenecraft: An llm agent for synthesizing 3d scenes as blender code. In
Forty-first International Conference on Machine Learning, 2024.
10

[18] Ian Huang, Guandao Yang, and Leonidas Guibas. Blenderalchemy: Editing 3d graphics with
vision-language models. InEuropean Conference on Computer Vision, pages 297–314. Springer,
2024.
[19] Moritz Ibing, Isaak Lim, and Leif Kobbelt. 3d shape generation with grid-based implicit
functions. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 13559–13568, 2021.
[20] R. Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy
Mitra, and Daniel Ritchie. Shapeassembly: Learning to generate programs for 3d shape structure
synthesis.ACM Transactions on Graphics (TOG), Siggraph Asia 2020, 39(6):Article 234, 2020.
[21] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions.arXiv
preprint arXiv:2305.02463, 2023.
[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian
splatting for real-time radiance field rendering.ACM Trans. Graph., 42(4):139–1, 2023.
[23] Adam R Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Sona
Mokrá, and Danilo Jimenez Rezende. Nerf-vae: A geometry aware 3d scene generative model.
InInternational conference on machine learning, pages 5742–5752. PMLR, 2021.
[24] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan,
and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation.
InEuropean Conference on Computer Vision, pages 112–130. Springer, 2024.
[25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei
Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. URL
https://arxiv.org/abs/2408.03326.
[26] Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey
Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language
model-augmented code emulator, 2024. URLhttps://arxiv.org/abs/2312.04474.
[27] Ruihui Li, Xianzhi Li, Ka-Hei Hui, and Chi-Wing Fu. Sp-gan: Sphere-guided 3d shape
generation and manipulation.ACM Transactions on Graphics (TOG), 40(4):1–12, 2021.
[28] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence,
and Andy Zeng. Code as policies: Language model programs for embodied control, 2023. URL
https://arxiv.org/abs/2209.07753.
[29] Sining Lu, Guan Chen, Nam Anh Dinh, Itai Lang, Ari Holtzman, and Rana Hanocka. Ll3m:
Large language 3d modelers.arXiv preprint arXiv:2508.08228, 2025.
[30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoor-
thi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis.
Communications of the ACM, 65(1):99–106, 2021.
[31] OpenAI. GPT-4o Technical Report.https://openai.com/index/hello-gpt-4o/, 2024.
[32] OpenAI. Introducing OpenAI o3 and o4-mini, 2025. URL https://openai.com/index/
introducing-o3-and-o4-mini/. Model announcement (Apr 16, 2025).
[33] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using
2d diffusion.arXiv preprint arXiv:2209.14988, 2022.
[34] Qdrant Team. Qdrant: Vector database and vector search engine, 2025. URL https://github.
com/qdrant/qdrant. GitHub repository, Version v1.15.4.
[35] Etai Sella, Noam Atia, Ron Mokady, and Hadar Averbuch-Elor. Blended point cloud diffusion
for localized text-guided shape editing, 2025. URL https://arxiv.org/abs/2507.15399 .
[36] Brian Skinn. sphobjinv: A practical tool for manipulating sphinx objects.inv files, 2024. URL
https://github.com/bskinn/sphobjinv.
11

[37] The Sphinx Project. Sphinx documentation.https://www.sphinx-doc.org/, 2025.
[38] Yonglong Tian, Andrew Luo, Xingyuan Sun, Kevin Ellis, William T. Freeman, Joshua B.
Tenenbaum, and Jiajun Wu. Learning to infer and execute 3d shape programs. InInternational
Conference on Learning Representations, 2019.
[39] Unity. Unity real-time development platform, 2025. URLhttps://unity.com/.
[40] Unity. Unity scripting api, 2025. URL https://docs.unity3d.com/6000.2/
Documentation/ScriptReference/index.html.
[41] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-
and-image driven manipulation of neural radiance fields. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 3835–3844, 2022.
[42] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing
Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d
digital avatars using diffusion. InProceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 4563–4573, 2023.
[43] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu,
Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal
models in versatility, reasoning, and efficiency.arXiv preprint arXiv:2508.18265, 2025.
[44] Karl D. D. Willis, Yewen Pu, Jieliang Luo, Hang Chu, Tao Du, Joseph G. Lambourne, Armando
Solar-Lezama, and Wojciech Matusik. Fusion 360 gallery: A dataset and environment for
programmatic cad construction from human design sequences.ACM Transactions on Graphics
(TOG), 40(4), 2021.
[45] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and
Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image,
2024.
[46] Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computer-
aided design models. InProceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), pages 6772–6782, October 2021.
[47] Zhicong Wu, Hongbin Xu, Gang Xu, Ping Nie, Zhixin Yan, Jinkai Zheng, Liangqiong Qu, Ming
Li, and Liqiang Nie. Textsplat: Text-guided semantic fusion for generalizable gaussian splatting.
arXiv preprint arXiv:2504.09588, 2025.
[48] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and
Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes, 2015. URL
https://arxiv.org/abs/1406.5670.
[49] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. In-
stantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruc-
tion models.arXiv preprint arXiv:2404.07191, 2024.
[50] Jingwei Xu, Chenyu Wang, Zibo Zhao, Wen Liu, Yi Ma, and Shenghua Gao. Cad-mllm:
Unifying multimodality-conditioned cad generation with mllm, 2024.
[51] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu,
Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by
bridging 2d and 3d diffusion models. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 6796–6807, 2024.
[52] xiaohui zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and
Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. In S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors,Advances in Neu-
ral Information Processing Systems, volume 35, pages 10021–10039. Curran Associates,
Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/
40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf.
12

[53] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape
representation for neural fields and generative diffusion models.ACM Transactions On Graphics
(TOG), 42(4):1–16, 2023.
13

A Appendix
A.1 Reconstruction Pipeline Details
ModelNet10 Easy/Hard Split.We partition ModelNet10 intoeasyandhardsubsets per category:
objects with fewer parts, regular structures, and mild curvature transitions are labeled easy; objects
with more parts, irregular topology, or pronounced/high-curvature transitions are labeled hard.
1.bathtub:
Easy: bathtub_0111, bathtub_0119, bathtub_0139, bathtub_0154, bathtub_0155;
Hard: bathtub_0141, bathtub_0153, bathtub_0124, bathtub_0115, bathtub_0150.
2.Bed:
Easy: bed_0555, bed_0557, bed_0561, bed_0572, bed_0595;
Hard: bed_0548, bed_0566, bed_0571, bed_0614, bed_0598.
3.Chair:
Easy: chair_0894, chair_0897, chair_0950, chair_0901, chair_0896;
Hard: chair_0893, chair_0891, chair_0941, chair_0898, chair_0943.
4.Desk:
Easy: desk_0217, desk_0262, desk_0246, desk_0236, desk_0220;
Hard: desk_0263, desk_0253, desk_0231, desk_0209, desk_0226.
5.Dresser:
Easy: dresser_0248, dresser_0254, dresser_0266, dresser_0232, dresser_0205;
Hard: dresser_0209, dresser_0257, dresser_0243, dresser_0217, dresser_0233.
6.Monitor:
Easy: monitor_0503, monitor_0545, monitor_0535, monitor_0531, monitor_0528;
Hard: monitor_0483, monitor_0522, monitor_0529, monitor_0511, monitor_0539.
7.night_stand:
Easy: night_stand_0207, night_stand_0232, night_stand_0263, night_stand_0231,
night_stand_0283;
Hard: night_stand_0225, night_stand_0208, night_stand_0278, night_stand_0270,
night_stand_0262.
8.Sofa:
Easy: sofa_0692, sofa_0687, sofa_0770, sofa_0756, sofa_0683;
Hard: sofa_0761, sofa_0777, sofa_0745, sofa_0746, sofa_0743.
9.Table:
Easy: table_0443, table_0439, table_0399, table_0422, table_0447;
Hard: table_0436, table_0405, table_0430, table_0470, table_0423.
10.Toilet:
Easy: toilet_0393, toilet_0438, toilet_0408, toilet_0355, toilet_0439;
Hard: toilet_0419, toilet_0409, toilet_0367, toilet_0436, toilet_0401.
Table 4:Open-source VLM failure rateon ModelNet10. Values arefail/total.
Strategy InternVL3.5-38B Qwen2.5-VL-72B LLaV A-OneVision-72B
Single-call 4/100 (4%) 4/100 (4%) 48/100 (48%)
Planning 27/100 (27%) 3/100 (3%) 42/100 (42%)
RAG 24/100 (24%) 7/100 (7%) 46/100 (46%)
Few-shot 28/100 (28%) 1/100 (1%) 76/100 (76%)
Failure rate of VLMs.We report the proportion of prompts that producedunsuccessfulruns on
ModelNet10 for open-source VLMs. For each task, when the first generated code runs into an
error, the model has 5 chances to correct it. If the code still reports an error after the chances are
exhausted, the generation is considered to have failed. The failure rates are shown as fail/total in
Table 4. For open-source models, we compute reconstruction metrics only on correctly generated
samples. For closed-source models, failed initial generations are handled by the same automatic
14

retry/self-correction protocol until an executable script is obtained; no manual code or mesh repair is
used before evaluation.
Reconstruction Metrics.All 3D metrics are computed after the upright-preserving similarity
alignment described in Section 3.2. The evaluation samples surface points from the aligned generated
mesh and the ground-truth mesh, and uses the ground-truth robust scale for normalization.
3D metrics.
•Chamfer Distance (CD).CD is the bidirectional nearest-neighbor surface distance, averaged
over generated-to-ground-truth and ground-truth-to-generated directions and normalized by the
ground-truth robust scale. Lower CD indicates better geometric accuracy.
•F-score at 5% (F@5%).F@5% uses a threshold of 5% of the ground-truth scale. Precision is the
fraction of generated surface samples close to the ground truth, recall is the fraction of ground-truth
samples recovered by the generated surface, and F@5% is their harmonic mean. Higher values
indicate better surface matching under this tolerance.
•Spatial Balanced Recall at 2% (SBR@2%).SBR@2% measures ground-truth coverage under a
2%-of-scale threshold after partitioning the ground-truth surface into spatial cells and averaging
recall across occupied cells. Higher values indicate more spatially distributed recovery rather than
recovery concentrated on large easy surfaces.
•Normal Consistency at 5% (NC@5%).NC@5% averages the absolute dot product between
matched surface normals for bidirectional nearest-neighbor pairs whose distance is within 5% of
the ground-truth scale. Higher values indicate better local surface orientation among matched
regions.
For 2D evaluation, we use the same selected camera view as the single-image input and ray-cast the
aligned generated mesh and the ground-truth mesh to obtain foreground masks, depths, and normals.
Input-view 2D metrics.
•Silhouette IoU (Sil IoU).Sil IoU is the intersection-over-union of the two foreground masks.
Higher values indicate better projected shape agreement.
•Depth NRMSE.Depth NRMSE is the root mean squared depth error over the foreground inter-
section, normalized by the ground-truth robust scale. Lower values indicate better visible-depth
recovery.
•Normal MAE.Normal MAE is the mean angular error between visible normals over the foreground
intersection, normalized by π/2and computed with an absolute normal dot product. Lower values
indicate better visible surface-orientation agreement. If the foreground intersection is empty, depth
NRMSE and normal MAE are set to 1.0 by the evaluation script.
Examples.The blueprint example is shown in Listing 1, the Blender API example in Listing 2, an
example query generated by Gemini in Listing 3, and a retrieved RAG example in Listing 4.
Bootstrap uncertainty for selected reconstruction comparisons.We estimate sampling uncer-
tainty over object identities using nonparametric bootstrap resampling. To keep the analysis focused,
we report confidence intervals for the reconstruction comparisons most directly tied to the main
conclusions: the external baselines used for reference comparisons and the closed-source VLM
configurations emphasized in the main discussion. For each selected method and metric, we resample
evaluated object-level metric records with replacement 1,000 times and recompute the mean. The
95% confidence intervals are the 2.5th and 97.5th percentiles of the bootstrap distribution.
A.2 More Experimental Results
Complete Results on ModelNet10-easyandhard.We evaluate the three closed-source VLMs on
ModelNet10-easyandhardunder theSingle-call,Planning,RAG,Few-shot, andAgentparadigms;
results are summarized in Table 5. The table also includes the external 3D baselines used in the main
reconstruction comparison: Unique3D, InstantMesh, and MeshCoder. Overall, the split results are
consistent with the main paper: code-based reconstruction generally performs better on easy shapes
than on hard shapes, especially for the stronger Gemini variants, whereas the external 3D baselines
15

Table 5:Quantitative reconstruction results on ModelNet10easy/hardsplit.“Sin." stands for
“single-call", “Pla." for “planning", “Few." for “Few-shot", and “Agt." for “agent". Metrics: CD =
Chamfer Distance, F@5% = F-score at 5% threshold, SBR@2% = Spatial Balanced Recall at 2%
threshold, NC@5% = Normal Consistency at 5% threshold, Sil IoU = Silhouette IoU, NRMSE =
Normalized RMSE, and MAE = Mean Angular Error.
Model ParadigmModelNet10-easyModelNet10-hard
CD↓F@5%↑SBR@2%↑NC@5%↑Sil IoU↑NRMSE↓MAE↓CD↓F@5%↑SBR@2%↑NC@5%↑Sil IoU↑NRMSE↓MAE↓
External 3D baselines
Unique3D – 0.0589 0.5526 0.2236 0.6321 0.5225 0.1367 0.4952 0.0571 0.5832 0.2372 0.6151 0.5238 0.1408 0.5046
InstantMesh – 0.0187 0.9066 0.6716 0.8253 0.8119 0.0484 0.1897 0.0194 0.9054 0.6169 0.7936 0.8017 0.0590 0.2324
MeshCoder – 0.0464 0.6764 0.4438 0.6410 0.5868 0.1214 0.3806 0.0420 0.7354 0.4624 0.6685 0.6324 0.1288 0.3983
Closed-source VLM families
Claude Sonnet 4Sin. 0.0394 0.7321 0.4952 0.7052 0.6653 0.1026 0.3055 0.0416 0.7036 0.3917 0.6675 0.6014 0.1088 0.3558
Pla. 0.0436 0.6940 0.4738 0.6500 0.6400 0.1216 0.3689 0.0403 0.7198 0.4345 0.6552 0.6129 0.1201 0.3777
RAG 0.0364 0.7380 0.4999 0.6706 0.6642 0.1088 0.3448 0.0372 0.7408 0.4629 0.6575 0.6678 0.1253 0.4099
Few. 0.0336 0.7734 0.5340 0.6977 0.6840 0.0929 0.3054 0.0413 0.7259 0.4296 0.6513 0.6283 0.1190 0.3929
Agt. 0.0422 0.7185 0.5059 0.6918 0.6412 0.1167 0.3451 0.0452 0.7085 0.4908 0.6673 0.6082 0.1214 0.3891
o3Sin. 0.0596 0.5906 0.3513 0.6253 0.5622 0.1363 0.4141 0.0620 0.5869 0.2857 0.5928 0.5317 0.1458 0.4538
Pla. 0.0317 0.7855 0.5455 0.7018 0.7049 0.0988 0.2950 0.0339 0.7902 0.5170 0.6911 0.6794 0.0978 0.3220
RAG 0.0432 0.6876 0.4157 0.6726 0.6055 0.1168 0.3580 0.0499 0.6402 0.3298 0.6341 0.5249 0.1274 0.4208
Few. 0.0308 0.7991 0.5517 0.7277 0.7082 0.0865 0.2654 0.0329 0.7907 0.5096 0.7071 0.6786 0.1072 0.3224
Agt. 0.0557 0.6245 0.3921 0.6371 0.5353 0.1394 0.4022 0.0460 0.6883 0.4201 0.6868 0.5736 0.1386 0.3990
Gemini 3 ProSin. 0.0346 0.7720 0.5350 0.7150 0.6926 0.0974 0.2991 0.0366 0.7528 0.4623 0.7117 0.6256 0.1023 0.3074
Pla. 0.0263 0.8444 0.6105 0.7273 0.7630 0.0820 0.2691 0.0297 0.8106 0.5530 0.7105 0.6912 0.0982 0.3035
RAG 0.0227 0.8710 0.6496 0.7478 0.7778 0.0735 0.2513 0.0275 0.8439 0.5908 0.7271 0.7395 0.0920 0.2845
Few. 0.0256 0.8502 0.6335 0.7500 0.7657 0.0678 0.2329 0.0241 0.8560 0.5881 0.7355 0.7380 0.0817 0.2766
Agt. 0.0239 0.8534 0.6318 0.7380 0.7701 0.0779 0.2519 0.0257 0.8492 0.6208 0.7285 0.7369 0.0874 0.2868
Table 6:Object-level bootstrap 95% confidence intervals for selected ModelNet10 reconstruction
comparisons.We report mean [2.5th, 97.5th percentile] over 1,000 bootstrap resamples of evaluated
object-level metric records.
3D Metrics
MethodCD↓F@5%↑SBR@2%↑NC@5%↑
InstantMesh 0.0191 [0.0176, 0.0205] 0.9060 [0.8898, 0.9213] 0.6443 [0.6127, 0.6746] 0.8095 [0.7950, 0.8242]
MeshCoder 0.0442 [0.0380, 0.0507] 0.7053 [0.6598, 0.7460] 0.4529 [0.4049, 0.4999] 0.6544 [0.6238, 0.6836]
Claude RAG 0.0368 [0.0338, 0.0400] 0.7394 [0.7105, 0.7668] 0.4814 [0.4483, 0.5178] 0.6641 [0.6429, 0.6855]
Claude Few-shot 0.0374 [0.0340, 0.0410] 0.7497 [0.7193, 0.7781] 0.4818 [0.4418, 0.5219] 0.6745 [0.6435, 0.7030]
o3 Few-shot 0.0318 [0.0290, 0.0347] 0.7949 [0.7652, 0.8222] 0.5306 [0.4905, 0.5720] 0.7174 [0.6970, 0.7378]
Gemini RAG 0.0251 [0.0227, 0.0278] 0.8573 [0.8336, 0.8797] 0.6199 [0.5855, 0.6600] 0.7374 [0.7173, 0.7559]
Gemini Few-shot 0.0249 [0.0223, 0.0280] 0.8532 [0.8327, 0.8740] 0.6106 [0.5746, 0.6475] 0.7427 [0.7257, 0.7605]
Gemini Agent 0.0248 [0.0228, 0.0269] 0.8513 [0.8301, 0.8706] 0.6263 [0.5915, 0.6608] 0.7332 [0.7162, 0.7508]
2D Metrics
MethodSil IoU↑NRMSE↓MAE↓
InstantMesh 0.8068 [0.7875, 0.8254] 0.0537 [0.0479, 0.0598] 0.2110 [0.1930, 0.2283]
MeshCoder 0.6091 [0.5669, 0.6497] 0.1250 [0.1109, 0.1401] 0.3893 [0.3485, 0.4273]
Claude RAG 0.6660 [0.6362, 0.6994] 0.1171 [0.1067, 0.1270] 0.3773 [0.3437, 0.4093]
Claude Few-shot 0.6561 [0.6240, 0.6901] 0.1060 [0.0952, 0.1172] 0.3491 [0.3108, 0.3905]
o3 Few-shot 0.6934 [0.6609, 0.7247] 0.0969 [0.0872, 0.1075] 0.2939 [0.2676, 0.3206]
Gemini RAG 0.7585 [0.7310, 0.7823] 0.0829 [0.0723, 0.0929] 0.2681 [0.2406, 0.2938]
Gemini Few-shot 0.7517 [0.7255, 0.7784] 0.0748 [0.0671, 0.0825] 0.2550 [0.2300, 0.2821]
Gemini Agent 0.7535 [0.7261, 0.7784] 0.0826 [0.0745, 0.0914] 0.2693 [0.2470, 0.2936]
show weaker or less consistent easy/hard trends. This suggests that current code-based reconstruction
remains sensitive to structural complexity: easy objects map more cleanly to parts, parameters, and
transformations, while hard objects require finer visual parsing and more precise spatial assembly.
Reconstruction Bad Cases Analysis.Figure 6 demonstrates several failure cases of Gemini
within the RAG paradigm. For Chair 0898, while the generated object exhibits a plausible shape
and successfully produces chair legs with complex intersecting lines, it does not conform to the
specifications of the ground truth. The reconstruction of Chair 0941 captures the general structure;
however, the size of the "Y"-shaped backrest is incorrect, and the interconnecting components
between the legs are missing. In Chair 0950, the individual components are generated approximately
correctly, but their spatial arrangement is inaccurate, resulting in an overall structure that deviates
significantly from the ground truth. At first glance, the object Table 0423 appears somewhat similar,
but a detailed inspection reveals that the orientation of the legs and the angles of the connecting bars
are rotated by 90 degrees. Furthermore, while the ground truth features four A-shaped leg structures,
16

Multi -view 
RenderingDepth NormalFigure 5: Rendered multi-view images of 3D object Chair 0891 in ModelNet10, with depth and
surface normals.
Ground Truth GeneratedChair 0898 Chair 0941 Chair 0950 Table 0423 Desk 0217 Desk 0226
Figure 6: Failure cases demonstration. The generated objects preserve category-level semantics but
deviate in fine geometric details.
the generated object exhibits only two. For Desk 0217, the model misjudges the relative spacing,
placing a horizontal bar at the midpoint instead of the correct position at one-quarter of the height.
Desk 0226 possesses a complex structure with numerous curved elements and components. Although
the final generated result bears a rough resemblance, the details differ substantially.
The primary failure modes can be summarized as follows:
• Incomplete comprehension of the input image, leading to missing components.
•Difficulty in accurately interpreting complex images, resulting in structures that are only
coarsely similar to the ground truth.
•Insufficient spatial reasoning capability, causing failures in the correct assembly of compo-
nents even when they are generated accurately.
Depth and Normal rendering examples.We render each object from multiple candidate view-
points, as shown in Figure 5. A human-selected RGB view is used as the input condition image. For
2D metrics, we render both the prediction and the ground truth from this same camera view, so the
evaluation quantifies recovery of the observed input view under the single-view condition.
Token cost.To better understand the practical overhead of different interaction paradigms, we report
the average token usage per item for closed-source model runs in Table 7. The single-call paradigm
is the most economical. Planning, RAG, and few-shot prompting add structured context through
blueprints, retrieved APIs, or exemplars, increasing the token footprint. The Agent paradigm incurs
the largest token usage because it uses multi-stage planning, checking, part-wise code generation, and
error correction; its prompt-token count includes cached-read tokens. These results highlight a clear
17

Table 7: Average token usage per item for closed-source model runs under different paradigms.
single-call planning RAG few-shot Agent
Prompt tokens 1,222 9,093 11,867 9,800 118,000
Completion tokens 10,885 16,989 21,752 10,900 52,000
Total tokens 12,107 26,082 33,619 20,700 170,000
Source
Variant
Source
Variant
Figure 7:Additional examples of the text-conditional reconstruction variant.The model generates
a modified 3D asset based on a source image from ModelNet10 and a corresponding text instruction.
Here Source denotes the original object rather than an edited ground truth.
trade-off between efficiency and performance. In practice, this suggests that single-call prompting
may be preferable in resource-sensitive scenarios, whereas structured paradigms are more suitable
when accuracy or part-level fidelity is prioritized and additional token overhead is acceptable.
Additional Qualitative Results.We provide additional qualitative examples for thetext-
conditional reconstruction variantin Figure 7 and forcode editingin Figures 10 and 11. The
complete editing instructions are listed in Section A.3.
Articulation Results.We evaluate the effectiveness of our Blender-based articulation pipeline on
samples from three object categories: Cabinet, Monitor, and Toilet. Specifically, we utilize shape keys
to implement the translational motion of cabinet drawers, while applying rotational transformations
to achieve the horizontal and vertical swiveling of monitor screens and the axial rotation of toilet lids.
The generated motions are designed to follow plausible physical motion patterns. The qualitative
results are illustrated in Figure 8.
A.3 Editing pipeline details
User study interface.Figure 9 shows the interface used in our user study for comparing anonymous
edited results against the same reference image and edit prompt.
Complete Instructions Used for Editing.For visual clarity, we only show the most critical text
instruction when showing the edit figures, and omit the long part with .... Here we list the complete
instructions.
In Figure 10, the instructions we use are:
18

Cabinet Monitor ToiletFigure 8:Visualization of articulated objects.We demonstrate the articulation effects generated by
our Blender script across three categories. These include the sliding translation of cabinet drawers,
the multi-axis rotation of monitor screens, and the hinge-based rotation of toilet lids, all simulating
real-world usage scenarios.
Figure 9:User study interface for image editing evaluation.Each item presents one reference
render, two anonymous edited variants in randomized left/right order, the edit prompt, and rating
controls for instruction following and preservation of unedited regions.
1. Let all the rings on the pillar sink with gravity and fit together.
2.The ring handle on the side of this cup is too big and does not match the cup body. Make it
smaller.
3.The candle on this cake is too thick and short. Make it thinner and longer and reduce the
number to 1 and insert it in the middle of the top of the cake.
4.Change the frustum-shaped lampshade of the upper part of the table lamp into a cylindrical
shape.
5. Position the top layer of the burger off-center so people can see the insides.
6. This oil-paper umbrella is open, with a cone on top. Turn it to the closed position.
In Figure 4, the instructions we use are:
1. Make the bathtub more square and add a flat base for stability.
19

Input
EditedAll the rings sink 
with gravity…Make the ring handle smaller…Make the candle one and taller…Make lampshade  into cylindrical…Close the oil -
paper umbrella…
Position the top layer off-center…Figure 10:Qualitative results for code editing.Each example shows an input and edited object
fromBlendNet-Ewith a shortened text instruction; complete instructions are listed below.
Input
Edited
Figure 11:Complementary code editing examples on BlendNet-E.These examples cover additional
objects and edit intents beyond Figure 10.
2. Add a second drawer below the existing one.
3. Change the base legs to a single centered pedestal.
4. Replace the cylindrical lampshade above this desk lamp with a triangular cone.
5.Make this table lamp taller. The column mistakenly passes through the lampshade and
protrudes a little from the top. Remove this small part.
6.Lengthen the four cylindrical legs of this table and connect the legs at opposite corners at
the bottom with X-shaped wooden strips to make its structure more stable.
In Figure 7, the instructions we use are:
• Upper part:
1. Add a lower shelf between the two legs.
2. Convert the corner bath to an oval shape.
3. Convert one of the crib’s sides into a removable panel.
4. Cut a large opening in the middle of the backrest.
5. Extend the basin to double its current length.
• Lower part:
1. Add a central open shelf in the knee space area for additional storage.
2. Add a headboard to the bed.
3. Add a fifth drawer at the bottom.
4. Add a second, smaller screen on top to create a dual-monitor setup.
5. Add a lower central support beam between the sofa legs.
In Figure 11, the instructions we use are:
1.This sofa has armrests on only one side and the modification makes it have armrests on both
sides.
20

2. The keychain circle on this cup is too big; make it smaller.
3.The cylindrical portion of this cup was incorrectly generated as a solid shape. Make it
hollow.
4. Add a handguard in the middle of this sofa to give it two separate seats.
5. Separate the spherical part of this bulb from the base.
6. This bucket has an ugly ring around the cylindrical waist. Remove it.
21

A.4 Detailed Pipeline Artifacts
Listing 1: Blueprint JSON for object in Figure 2. The blueprint normalizes to a base dimension
(overall_height=1.0) and encodes part-wise parametrics used by the code generator.
{
"object_category": "bar_stool",
"base_dimension": "overall_height",
"dimensional_profile": {
"overall_height": 1.0,
"overall_width_at_base_ratio_to_height": 0.42,
"overall_depth_at_base_ratio_to_height": 0.4,
"seat_height_from_ground_ratio_to_overall_height": 0.65
},
"geometric_components": {
"legs": {
"count": 4,
"profile_shape": "cylindrical",
"diameter_ratio_to_overall_height": 0.02,
"splay_angle_from_vertical_degrees": 6.0
},
"footrest": {
"structure_type": "continuous_four_sided_brace",
"height_from_ground_ratio_to_overall_height": 0.18,
"cross_section_diameter_ratio_to_leg_diameter": 1.0,
"front_bar_outward_curve_depth_ratio_to_overall_depth": 0.15
},
"seat": {
"plan_shape": "rounded_square",
"width_ratio_to_overall_width_at_base": 0.86,
"depth_ratio_to_overall_depth_at_base": 0.85,
"thickness_ratio_to_overall_height": 0.03,
"ergonomic_concave_dip_ratio_to_seat_depth": 0.05,
"front_edge_waterfall_radius_ratio_to_seat_thickness": 1.0,
"cutouts": {
"count": 2,
"type": "slot",
"slot_length_ratio_to_seat_depth": 0.7,
"slot_width_ratio_to_seat_width": 0.08,
"slot_spacing_from_centerline_ratio_to_seat_width": 0.25
}
},
"backrest": {
"height_above_seat_ratio_to_overall_height": 0.35,
"width_ratio_to_seat_width": 0.95,
"tilt_angle_from_vertical_degrees": 12.0,
"horizontal_lumbar_curve_depth_ratio_to_width": 0.08,
"structure": {
"type": "slatted_frame",
"frame_thickness_ratio_to_leg_diameter": 1.2,
"slat_count": 3,
"slat_height_ratio_to_backrest_height": 0.12,
"slat_vertical_gap_ratio_to_slat_height": 1.1,
"vertical_support_count": 2,
"vertical_support_width_ratio_to_frame_thickness": 1.0
}
},
"armrests": {
"count": 2,
"structure_type": "continuous_loop_from_backrest_to_seat",
"height_above_seat_at_rear_ratio_to_overall_height": 0.15,
"length_ratio_to_seat_depth": 0.8,
"cross_section_diameter_ratio_to_leg_diameter": 1.0,
"downward_slope_angle_degrees": 3.0,
"outward_bow_distance_ratio_to_seat_width": 0.05
}
}
}
22

Listing 2: Blender 4.4 Python API database entry (structured JSON).
{
"symbol": "bpy.ops.curves.add_bezier",
"language": "python",
"module": "bpy.ops.curves",
"signature": ".ops.curves.add_bezier(*, radius=1.0, enter_editmode=False, align=’WORLD’,
location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))",
"parameters": [
{"name": "radius", "description": "radius (float in [0, inf], optional) - Radius"},
{"name": "enter_editmode", "description": "enter_editmode (boolean, optional) - Enter Edit Mode
when adding this object"},
{"name": "align", "description": "align (enum in [’WORLD’, ’VIEW’, ’CURSOR’], optional) -
Alignment of the new object. WORLD: align to world. VIEW: align to view. CURSOR: use 3D cursor
orientation."},
{"name": "location", "description": "location (mathutils.Vector of 3 items in [-inf, inf],
optional) - Location for the newly added object"},
{"name": "rotation", "description": "rotation (mathutils.Euler of 3 items in [-inf, inf],
optional) - Rotation for the newly added object"},
{"name": "scale", "description": "scale (mathutils.Vector of 3 items in [-inf, inf], optional) -
Scale for the newly added object"}
],
"doc_text": "Add new bezier curve",
"version": "4.4"
}
Listing 3: VLM-generated intent queries per blueprint component.
{
"intent_queries": {
"queries": {
"legs": {
"prefer_modules": ["bpy.ops.mesh", "bpy.ops.object"],
"keywords": ["primitive_cylinder_add", "cylinder", "duplicate_move", "rotate", "splay"],
"query": "Create four cylindrical legs for a bar stool. Start by adding a cylinder
primitive, then duplicate it and rotate the legs to create a splayed angle from the vertical."
},
"footrest": {
"prefer_modules": ["bpy.ops.curve", "bpy.ops.object"],
"keywords": ["curve", "bezier", "bevel", "extrude", "depth", "join"],
"query": "Model a continuous four-sided footrest brace with a curved front. Use a bezier
curve with bevel depth to form the cylindrical rails and join the segments."
},
"seat": {
"prefer_modules": ["bpy.ops.mesh", "bpy.ops.object"],
"keywords": ["primitive_cube_add", "subdivision_set", "bevel", "boolean",
"proportional_edit", "loop_cut"],
"query": "Create a rounded square seat with an ergonomic dip and two slot cutouts. Start
with a cube, use subdivision and proportional editing for the dip, bevel the edges, and apply a
boolean difference modifier for the slots."
},
"backrest": {
"prefer_modules": ["bpy.ops.mesh", "bpy.ops.object"],
"keywords": ["plane", "extrude", "inset", "boolean", "loop_cut", "curve_deform", "modifier"],
"query": "Create a slatted backrest frame that is tilted and curved. Model the basic shape
from a plane using extrude and inset, use a boolean modifier to create the slats, and bend the
result with a curve deform modifier."
},
"armrests": {
"prefer_modules": ["bpy.ops.curve", "bpy.ops.object"],
"keywords": ["curve", "bezier", "extrude", "bevel", "depth", "mirror"],
"query": "Create two continuous loop armrests extending from the backrest to the seat. Model
one armrest using a bezier curve with a circular bevel depth, then use a mirror modifier to
create the second one."
}
}
}
}
23

Listing 4: Retrieved APIs organized by component (post-retrieval structuring).
{
"components": [
{
"name": "legs",
"apis": [
{
"symbol": "bpy.ops.mesh.primitive_cylinder_add",
"signature": "primitive_cylinder_add(vertices=32, radius=1.0, depth=2.0,
end_fill_type=’NGON’, calc_uvs=True, enter_editmode=False, align=’WORLD’, location=(0.0, 0.0,
0.0), rotation=(0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))",
"parameters": {
"vertices": "int in [3, 10000000], optional - Number of vertices for the circular caps.",
"radius": "float in [0, inf], optional - Radius of the cylinder.",
"depth": "float in [0, inf], optional - Height of the cylinder.",
"end_fill_type": "enum in [’NOTHING’, ’NGON’, ’TRIFAN’], optional - How to fill the
cylinder ends.",
"location": "3D vector, optional - Location for the newly added object.",
"rotation": "3D Euler rotation, optional - Rotation for the newly added object.",
"scale": "3D vector, optional - Scale for the newly added object."
}
}
]
},
{ "name": "footrest", "apis": [] },
{ "name": "seat", "apis": [] },
{ "name": "backrest", "apis": [] },
{
"name": "armrests",
"apis": [
{
"symbol": "bpy.ops.curves.add_bezier",
"signature": "add_bezier(radius=1.0, enter_editmode=False, align=’WORLD’, location=(0.0,
0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))",
"parameters": {
"radius": "float in [0, inf], optional - Radius to set for the curve points.",
"enter_editmode": "boolean, optional - Enter Edit Mode after creating the object.",
"align": "enum in [’WORLD’, ’VIEW’, ’CURSOR’], optional - Alignment of the new object.
WORLD: align to world. VIEW: align to view. CURSOR: use 3D cursor orientation."
}
}
]
}
]
}
24