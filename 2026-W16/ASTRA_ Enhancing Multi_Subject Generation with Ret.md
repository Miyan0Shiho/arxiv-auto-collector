# ASTRA: Enhancing Multi-Subject Generation with Retrieval-Augmented Pose Guidance and Disentangled Position Embedding

**Authors**: Tianze Xia, Zijian Ning, Zonglin Zhao, Mingjia Wang

**Published**: 2026-04-15 14:50:05

**PDF URL**: [https://arxiv.org/pdf/2604.13938v1](https://arxiv.org/pdf/2604.13938v1)

## Abstract
Subject-driven image generation has shown great success in creating personalized content, but its capabilities are largely confined to single subjects in common poses. Current approaches face a fundamental conflict when handling multiple subjects with complex, distinct actions: preserving individual identities while enforcing precise pose structures. This challenge often leads to identity fusion and pose distortion, as appearance and structure signals become entangled within the model's architecture. To resolve this conflict, we introduce ASTRA(Adaptive Synthesis through Targeted Retrieval Augmentation), a novel framework that architecturally disentangles subject appearance from pose structure within a unified Diffusion Transformer. ASTRA achieves this through a dual-pronged strategy. It first employs a Retrieval-Augmented Pose (RAG-Pose) pipeline to provide a clean, explicit structural prior from a curated database. Then, its core generative model learns to process these dual visual conditions using our Enhanced Universal Rotary Position Embedding (EURoPE), an asymmetric encoding mechanism that decouples identity tokens from spatial locations while binding pose tokens to the canvas. Concurrently, a Disentangled Semantic Modulation (DSM) adapter offloads the identity preservation task into the text conditioning stream. Extensive experiments demonstrate that our integrated approach achieves superior disentanglement. On our designed COCO-based complex pose benchmark, ASTRA achieves a new state-of-the-art in pose adherence, while maintaining high identity fidelity and text alignment in DreamBench.

## Full Text


<!-- PDF content starts -->

ASTRA: Enhancing Multi-Subject Generation with Retrieval-Augmented Pose
Guidance and Disentangled Position Embedding
Tianze Xia1∗Zijian Ning1∗Zonglin Zhao1∗Mingjia Wang1
1Huazhong University of Science and Technology
Figure 1. Representative outputs showcase the capabilities ofASTRAin multi-subject generation, demonstrating both high subject fidelity
and precise pose control.In each example, the pose maps presented are retrieved from our meticulously constructed database based on the
prompts, and these pose maps provide clear structural guidance for the image synthesis process.
Abstract
Subject-driven image generation has shown great success
in creating personalized content, but its capabilities are
largely confined to single subjects in common poses. Cur-
rent approaches face a fundamental conflict when handling
multiple subjects with complex, distinct actions: preserv-ing individual identities while enforcing precise pose struc-
tures. This challenge often leads to identity fusion and pose
distortion, as appearance and structure signals become en-
tangled within the model’s architecture. To resolve this
conflict, we introduceASTRA(AdaptiveSynthesis through
TargetedRetrievalAugmentation), a novel framework that
architecturally disentangles subject appearance from pose
1arXiv:2604.13938v1  [cs.CV]  15 Apr 2026

structure within a unified Diffusion Transformer. ASTRA
achieves this through a dual-pronged strategy. It first em-
ploys a Retrieval-Augmented Pose (RAG-Pose) pipeline to
provide a clean, explicit structural prior from a curated
database. Then, its core generative model learns to pro-
cess these dual visual conditions using our Enhanced Uni-
versal Rotary Position Embedding (EURoPE), an asymmet-
ric encoding mechanism that decouples identity tokens from
spatial locations while binding pose tokens to the canvas.
Concurrently, a Disentangled Semantic Modulation (DSM)
adapter offloads the identity preservation task into the text
conditioning stream. Extensive experiments demonstrate
that our integrated approach achieves superior disentangle-
ment. On our designed COCO-based complex pose bench-
mark, ASTRA achieves a new state-of-the-art in pose adher-
ence, while maintaining high identity fidelity and text align-
ment in DreamBench.
1. Introduction
Recent text-to-image models [7, 10, 16, 35] have enabled
powerful personalized image generation. While early meth-
ods achieved remarkable single-subject control [8, 30, 46],
generating coherent scenes with multiple subjects in com-
plex interactions remains a major challenge. This task
exposes a core conflict in existing paradigms: preserving
multi-subject identity while ensuring complex pose fidelity.
This difficulty stems from two fundamental bottlenecks.
First, a data bottleneck: High-quality training data for
multi-subject interactions is scarce and impractical to col-
lect, limiting model generalization. Second, an architec-
tural bottleneck: State-of-the-art methods [5, 23, 37, 43]
that inject identity via attention mechanisms in Diffusion
Transformers (DiTs) [25] often suffer from attention confu-
sion. This leads to artifacts like “identity fusion” and “pose
distortion” , highlighting the need for a new approach to
decouple identity and pose control.
To address these bottlenecks, we draw inspiration from
Retrieval-Augmented Generation (RAG) [9, 17] in LLMs
and propose a “retrieve-then-generate” framework. In-
stead of generating complex poses from scratch a pro-
cess hindered by data scarcity and prone to hallucination
[15]—we first retrieve a suitable pose structure from a pre-
built “pose knowledge database.” This strategy circumvents
the data problem and decouples the entangled appearance-
structure signals at the input level. We introduce AS-
TRA, a framework that uses dual visual inputs: (1) a
Retrieval-Augmented Pose Pipeline provides explicit struc-
tural priors from a high-quality text-pose database, and (2)
a model framework with our proposedEnhancedUniversal
RotaryPositionEmbedding (EURoPE) andDisentangled
SemanticModulation(DSM) achieves precise control over
subject identities and poses.This decoupling of appearance and structure provides
fine-grained, assignable control that current methods lack.
While tools like ControlNet [47] master structural guidance,
they cannot customize subject identity. Conversely, IP-
Adapter [46] preserves a single subject’s identity but strug-
gles to assign distinct poses to multiple individuals. Our
framework ASTRA is the first to allow users to explicitly
assign a unique, complex pose to multiple customized sub-
jects simultaneously(as shown in Figure 1). This capability
is crucial for complex storytelling and personalized content
creation, overcoming the limitations of text-only and single-
subject-only descriptions for such scenes [2].
Our contributions are summarized as follows:
• We introduce a RAG-inspired paradigm that addresses the
data scarcity of complex actions by retrieving structural
priors from a high-quality, curated text-pose database.
• We develop ASTRA, a generative architecture with an
asymmetric encoding mechanism and Disentangled Se-
mantic Modulation, enabling precise and assignable con-
trol over multiple subjects’ identities and poses.
• On established benchmarks including DreamBench[30]
and multi-subject driven generation benchmarks, our AS-
TRA achieves highest DINO and CLIP-I scores. Further-
more, in our proposed Controlled Composition on COCO
evaluation, our method attains the highest Object Key-
point Similarity (OKS) and CLIP-I score, while secur-
ing the second-highest CLIP-T score. These demonstrate
its strong subject fidelity, text and pose control ability,
also showcasing its capability to deliver state-of-the-art
(SOTA) results.
2. Related Work
Subject-Driven and Controllable Generation.The pur-
suit of personalized visual content has led to signifi-
cant progress in subject-driven generation. This field is
broadly divided into two main paradigms: Fine-tuning-
based methods, including Textual Inversion [8] and Dream-
Booth [30], which perform efficient per-subject tuning to
embed novel concepts; and Tuning-free methods, such as
IP-Adapter [46], which utilize a dedicated image encoder
to inject subject features. Concurrently, a separate line of
work on controllable generation, championed by Control-
Net [47], has achieved precise spatial control by condition-
ing on structural maps such as human poses.However, ex-
tension of these methods to multiple subjects introduces sig-
nificant challenges, such as identity bleeding and attribute
confusion. To overcome these complexities, recent works
have proposed several novel strategies. For instance, MS-
Diffusion [40] introduces a layout-guided framework, us-
ing a ”Grounding Resampler” and masked cross-attention to
mitigate subject conflicts. UNO [43] proposes a model-data
co-evolution paradigm, leveraging a progressive pipeline
to synthesize high-consistency multi-subject training data.
2

OmniGen2 [42] addresses this as part of a unified model,
using a decoupled architecture and low-level V AE features
to preserve fine-grained identity during its in-context gen-
eration tasks. While these methods advance multi-subject
identity preservation, a critical challenge lies at the inter-
section of these domains:Existing methods struggle to
simultaneously preserve multiple, specific subject iden-
tities while enforcing complex, structured poses. This
issue is particularly acute in modern Diffusion Transform-
ers (DiTs) [25], highlighting the need for a new architecture
that can fundamentally decouple appearance from structure.
Retrieval-Augmented Generation (RAG).Inspired by
its success in Large Language Models (LLMs), Retrieval-
Augmented Generation (RAG) [9, 17] is an emerging
paradigm in visual synthesis. Although prior work has ex-
plored retrieval to improve rare concept generation or style
transfer [1, 12, 31, 32], our work introduces a novel appli-
cation of RAG philosophy. We leverage it to address the
structural data scarcity bottleneck in complex action syn-
thesis. By building and retrieving from a curated “text-
pose” database, we provide an explicit, clean structural
prior to the model. This strategic, pre-generation retrieval
not only bypasses the difficulty of synthesizing complex
poses from scratch but is also the cornerstone of our frame-
work’s ability to cleanly separate structural information (the
pose) from appearance information (the subject’s identity),
thereby preventing the feature entanglement that plagues
existing methods.
3. Method
Modern diffusion transformers (DiT) excel at text-to-
image generation, but struggle when simultaneously con-
ditioned on multiple subject identities and precise struc-
tural guidance (pose maps). Shared self-attention treats
all conditional inputs uniformly, causing feature entangle-
ment—either compromising pose to preserve identity or
blending identities incorrectly. ASTRA addresses this by
separating structural and appearance signals at both the ar-
chitectural and representational levels (Figure 2).
3.1. Pose Retrieval-Augmented Framework
The paucity of high-quality, subject-consistent datasets has
long presented a formidable obstacle for subject-driven gen-
eration. This challenge is particularly acute in the domain of
pose estimation and synthesis, where no large-scale, high-
fidelity dataset of pose maps currently exists to support
robust model training and inference. Compounding this
issue, the field of vision-language retrieval—particularly
for image-based Retrieval-Augmented Generation (RAG)
systems—lacks standardized, end-to-end methodologies for
constructing high-quality databases and implementing effi-cient retrieval mechanisms. Our proposed framework di-
rectly addresses these critical gaps by delivering a compre-
hensive solution that includes: (1) a large-scale, curated
pose database with diverse and anatomically accurate 2D
skeleton maps; (2) a simple but efficient curation pipeline
ensuring semantic fidelity and data quality; and (3) an ef-
ficient, lightweight retrieval system that seamlessly maps
ambiguous textual descriptions to the most relevant pose
representations.The entire process is visualized in Figure 3.
Systematic and Diverse Prompt Engineering.Our pro-
cess began with a foundation of over 300 distinct human
actions, covering a wide spectrum of complexity from sim-
ple gestures to intricate multi-person interactions. To en-
sure semantic richness and diversity, we then algorithmi-
cally generated approximately 30 textual variations for each
action. This was achieved by systematically manipulat-
ing a set of descriptive axes, including subject attributes
(e.g., age, body type), action specifics (e.g., ”stretching”
vs. ”stretching lazily”), environmental context, and camera
perspective. This systematic generation resulted in a com-
prehensive set of over 9,000 unique prompts, forming the
basis for our image generation phase.
High-Fidelity Generation and Curation.We utilized
the state-of-the-art text-to-image model, FLUX.1-pro[16],
to generate an initial image pool from our prompt set. To
distill this raw collection into a high-quality dataset, we de-
veloped a scalable, VLM-driven curation pipeline. The core
of our pipeline is an automated semantic verification stage
using the multi-modal capabilities of GPT-4o[14]. This
stage systematically measures the alignment between a gen-
erated image and its prompt by evaluating three key dimen-
sions: subject consistency (s 1), interaction logic (s 2), and
detail fidelity (s 3). An overall semantic alignment scoreS
is then computed as a weighted sum:
S=ω 1s1+ω2s2+ω3s3 (1)
These components are precisely calibrated. Scoress iare
generated on a continuous scales∈[0,1]by instructing the
VLM, via a structured prompt, to provide a final numerical
rating after its qualitative analysis. The weights (ω 1, ω2, ω3)
were optimized on a held-out validation set against human
preference scores, using regression to find the best fit. This
resulted in a hierarchy that prioritizes subject and interac-
tion correctness (ω 1> ω 2> ω 3). The final acceptance
thresholdθwas set to maximize the F1-score on the same
validation set. To validate the effectiveness of this auto-
mated pipeline, we conducted a limited human evaluation
on the VLM-filtered dataset, checking for subtle errors in
pose, composition, and aesthetics. The results showed that
fewer than 3% of the filtered images were flagged as low-
3

Figure 2. The ASTRA framework for multi-subject pose-controllable generation. The left panel shows the overall architecture, which
uniquely conditions a DiT on reference identities (I ref), a text prompt, and an explicit pose structure (C pose). Its effectiveness stems from
our Enhanced Universal Rotary Position Embedding (EURoPE), detailed on the right. EURoPE employs an asymmetric strategy: it
decouples identity from spatial location using universal rotary position embedding (UnoPE) forI reftokens, while enforcing strict structural
constraints by bindingC posetokens to the latent canvas with native RoPE. Concurrently, Disentangled Semantic Modulation (DSM) offloads
identity preservation by modulating text embeddings with visual features.
quality, confirming the VLM filter’s effectiveness in pro-
ducing high-quality training data.
Pose Extraction and Data Indexing.For each image that
passed the curation stages, we used OpenPose[3] to ex-
tract 2D pose skeletons and added a two-stage pose quality
check, with only 3.2% of pose maps flagged as low-quality,
validating the effectiveness of our pipeline.For any prompt
whose corresponding pose map was rejected at this stage,
the entire generation and two-stage curation cycle was re-
initiated until a valid pose representation was successfully
produced, ensuring complete and high-quality coverage for
our prompt set. Finally, for each verified pose map, the as-
sociated text prompt was encoded into a 384-dimensional
dense vector using the lightweight all-MiniLM-L6-v2[29]
sentence-transformer model. These vectors were then L2-
normalized[6] and stored in an efficient vector index, link-
ing each textual representation to its corresponding high-
fidelity pose map.
3.1.1. Retrieval at Inference Time.
During inference, a retriever is employed to source an
optimal structural prior from our pre-compiled database
for any given user prompt. To maximize retrieval pre-
cision, we introduce a prompt normalization stage. Raw
user prompts often entangle core pose-defining semantics
with non-structural information. Therefore, we utilize alightweight LLM, Qwen2.5-1.5B-Instruct[39], to parse user
inputP userand distill it into a canonical queryP canonical .
The objective of this normalization is to generate a query
that aligns with the multi-axial descriptive schema used for
our database construction. Specifically, the LLM is in-
structed to preserve semantically rich descriptors pertinent
to the pose—such as subject-action nuance, demographic
attributes, dynamic features, and compositional directives.
This normalized query,P canonical , is then encoded into a
fixed-size semantic vector,v query, using an all-MiniLM-L6-
v2 sentence-transformer followed by mean pooling and L2
normalization. This vector facilitates a Maximum Inner
Product Search (MIPS)[34] against the indexed database
vectors (v j), equivalent to a cosine similarity search, to
identify the entry (j∗) with the highest score:
j∗= arg max
j(vquery·vj)(2)
To ensure relevance, we apply a confidence gating mecha-
nism: the corresponding pose map,C pose, is selected only if
the scores j∗exceeds a predefined threshold (α u= 0.55).
This value was empirically determined on a validation set
to optimally balance precision and recall. If no score meets
this threshold—a scenario that typically arises for out-of-
distribution prompts—the retrieval step is bypassed. In this
case, the model defaults to relying solely on the text prompt
for structural guidance. This two-step process—semantic
4

Figure 3. The inference pipeline of ASTRA. A user prompt is first
refined into a canonical query by a VLM-based optimizer to isolate
pose-defining semantics. This query facilitates a vector search on
our curated database to find an optimal pose map (C pose). The
retrieved pose, acting as explicit structural guidance, is then passed
to the ASTRA model along with the subject identities (I ref) and
original text to synthesize the final image.
normalization followed by vector search—provides the gen-
eration model with an explicit structural condition (C pose),
establishing a robust foundation for high-fidelity synthesis.
3.1.2. Training Data Construction Pipeline
To construct training data pairs{prompt,ref imgs,pose},
we design a pipeline based on the curated pose database.
Target images match specific pose structures. Descriptive
prompts are extracted using Florence-2[44]’s captioning
module, while its object detection module identifies sub-
ject labels and bounding boxes from target images. These
are processed by SAM2[28] to segment individual subjects,
which are then re-edited by Flux.1 Kontext to generate
diverse reference images. Pose maps are extracted from
the target images using OpenPose. Finally, reference im-
ages are filtered by CLIP-I scores to ensure alignment with
prompts. The resulting data pairs provide a robust founda-
tion for training, integrating textual, structural, and visual
information.
3.2. ASTRA: The Generation Model Framework
ASTRA trains a DiT-based generator to synthesize images
faithful to multiple identities and precise poses. All train-
ing data originates from the curated retrieval-augmented
database.
Unified Multi-Modal Conditioning and Training.In
subject-driven generation tasks, a key challenge lies in in-
tegrating diverse input modalities—text prompts, noisy la-
tents, and multimodal visual references—while maintain-
ing training stability and generation quality. Traditional
methods, such as DreamBooth [30] and LoRA [11], typ-
ically rely on cross-attention mechanisms to inject iden-
tity features. However, when scaled to multi-subject sce-
narios, these approaches encounter issues such as compu-
tational complexity and feature entanglement, where each
additional subject requires an independent attention path.Recent methods propose two main improvements: convert-
ing visual features into token sequences for cross-attention
interactions with text [19], and directly concatenating all
input conditions into a unified sequence processed by the
model’s self-attention mechanism [43]. We adopt the lat-
ter approach for its efficiency, though directly introducing
noise-free reference tokens into a model pretrained on noisy
data can lead to instability. While previous methods address
this through multi-stage training, ASTRA resolves this by
explicitly defining the roles of visual conditions at the archi-
tectural level, leveraging an asymmetric positional encoding
mechanism to enable direct and efficient single-stage train-
ing. We formalize the input sequence construction process
as follows. First, we tokenize the reference imagesI refand
the retrieved pose mapC pose:
trefi=E img(Irefi), t pose=E pose(Cpose)(3)
These tokens are then assembled into an ordered visual con-
ditioning sequenceC visual, expressed as:
Cvisual=⟨t ref1, . . . , t refN−1, tpose⟩(4)
Finally, we concatenate the text condition tokensc, visual
conditioning sequenceC visual, and noisy latentz tinto the
complete input streamz in:
zin=c⊕C visual⊕zt (5)
Enhanced Universal Rotary Position Embedding (EU-
RoPE).As shown in the right panel of Figure 2, EURoPE
is an asymmetric positional encoding mechanism that re-
solves the conflict between preserving subject appearance
and enforcing precise pose geometry. It applies different en-
coding strategies to identity reference tokens and pose map
tokens within the DiT attention mechanism.
Forreference images, the goal is to extract identity fea-
tures while avoiding fixed spatial layout signals that may
conflict with text or pose constraints. We adopt Universal
Rotary Position Embedding (UnoPE) [43], which sequen-
tially re-indexes reference tokens:
(i′, j′) = (i+w N−1, j+h N−1)(6)
wherew, hdenote cumulative dimensions of preceding ref-
erences. This enforces layout control via text/pose while
keeping subject identity independent of original image ar-
rangement.
Forpose maps, strict spatial binding is desirable. We ap-
ply native RoPE [36] with positional indices(i, j)aligned
exactly to corresponding patches in the noisy latentz t, forc-
ing strong attention at skeleton locations. Because the pose
maps are clean skeletons on neutral backgrounds, positional
binding enforces geometry without introducing unwanted
details—turning a common positional conflict into a precise
structural constraint.
5

Figure 4. Qualitative comparison with different methods on single and multi subject driven generation
Disentangled Semantic Modulation.While the EU-
RoPE encoding mechanism effectively separates structural
and appearance signals at the input level, representational
entanglement within the self-attention layers of DiT re-
mains a critical challenge. The model must simultane-
ously process global text semantics, preserve multiple high-
fidelity identities, and enforce precise structural constraints,
all within a unified feature space. To address this, we pro-
pose the Disentangled Semantic Modulation (DSM) stream,
a novel module designed to offload identity preservation
from the primary denoising pathway. By recontextualizing
visual identity as a learnable modulation of the textual con-
dition, DSM explicitly encodes identity information into the
semantic instructions guiding the generation process. The
core premise behind DSM is that a subject’s unique iden-
tity can be conceptualized as a distinct point in a semantic
subspace of the model’s multimodal feature space. DSM
distills this identity, extracted from a reference image, into
a semantic offset vector∆E t. This offset modifies the text
embeddingsE tby introducing subject-specific biases. For-
mally, the DSM moduleΦmaps the visual featuresF vandtext embeddingsE tas:
∆Et= Φ(F v, Et;θΦ),(7)
whereθ Φrepresents the trainable parameters ofΦ. The
modulated embeddings are obtained as:
E′
t=E t+ ∆E t.(8)
This mechanism embeds the subject’s identity directly into
the textual instructions, eliminating the need for the primary
DiT blocks to learn implicit correlations between reference
tokens and image outputs. Instead, identity is seamlessly
integrated into the semantic control signals. The DSM
module is implemented as a lightweight, parameter-efficient
adapter with a cross-attention architecture. Text embed-
dingsE tare used as learnable queries to attend to high-
dimensional visual featuresF v, extracted from intermediate
layers of a CLIP vision encoder. This design ensures effi-
cient distillation of identity-critical features, with minimal
overhead relative to the base model. To enhance flexibility,
DSM applies modulation hierarchically: a global offset ad-
justs the main textual condition, while finer-grained offsets
6

are injected into individual layers of the DiT for dynamic
identity reinforcement over the denoising process.
4. Experiments
4.1. Experiments Setting
Implementation Details.We build our framework upon
the pre-trained FLUX.1 dev [16] model, with a learning rate
of10−5and a batch size of 1. Using a single-stage train-
ing strategy, we employ 32,000 high-quality text-pose pairs
(128,000 images) and train the model for 100,000 steps.
The experiments are conducted on 8 NVIDIA H200 GPUs,
with LoRA [11] of rank 512 applied throughout training.
Comparative Methods.ASTRA is capable of handling
both single-subject and multi-subject driven generation. For
general subject-driven tasks, we compare it with several
leading methods, including OmniGen [45], Ominicontrol
[37], FLUX IPAdapter v2 [16], Ms-diffusion [40], MIP-
Adapter [13], RealCustom++ [22], SSR-Encoder [48] and
UNO[43]. For the more demanding task of multi-subject
generation with complex pose control, where many of the
aforementioned methods are not applicable, we bench-
mark against recent state-of-the-art models: OmniGen[45],
DreamO[23] ,OmniGen2 [42], and UNO [43].
Evaluation Metrics.We adopt standard automatic met-
rics for our evaluation. Subject similarity is measured by
CLIP-I [27] and DINO [24] scores, while text fidelity is as-
sessed with CLIP-T. Crucially, to quantify structural adher-
ence, we employ Object Keypoint Similarity (OKS) [20].
For subject-driven generation, our evaluation covers both
single and multi-subject tasks. We assess single-subject
performance on the standard DreamBench [30]. For multi-
subject scenarios, following established protocols [13, 33],
we use 30 subject combinations from DreamBench with 25
text prompts each, resulting in a comprehensive test set of
4,500 image groups.
To evaluate our core contribution in multi-subject com-
plex pose control, we introduce a new challenging bench-
mark curated from the COCO Keypoints dataset [20], con-
taining 1,000 images with up to three subjects. For each im-
age, every annotated person is treated as a distinct subject:
we use the COCO person bounding boxes to crop identity
reference images, and construct a multi-person pose map
by rasterizing all ground-truth keypoints into an OpenPose-
style skeleton image. This skeleton map is then used as the
pose condition for all pose-aware methods on this bench-
mark, while the same identity crops and text prompts are
shared across methods. Concretely, for methods that offi-
cially support OpenPose-style or structural control inputs,
we follow their recommended usage and feed our skeleton
map through the corresponding pose-control branch. Formethods that do not officially provide a dedicated pose in-
terface, we implement our own support. In this way, every
compared model receives an equivalent pose prior, ensur-
ing that differences in performance reflect how identity and
pose are fused rather than whether a method has access to
pose information.
4.2. Qualitative Analyses
We provide qualitative comparisons against state-of-the-art
methods to visually verify the effectiveness of ASTRA in
Figure 4.
4.3. Quantitative Evaluations
4.3.1. Automatic Scores.
Table 1 compares our methods on DreamBench [30] against
both tuning-based and tuning-free methods. As a tuning-
free method, ASTRA has a leading scores over previous
methods with the highest CLIP-I and CLIP-T scores of
0.847 and 0.330, and a competitive DINO score of 0.699.
We also compare our method in the multi-image condi-
tion scenario in Table 2. ASTRA achieves the CLIP-I and
DINO scores of 0.745 and 0.554, and has competitive CLIP-
T scores compared to existing leading methods. These
show that ASTRA has a robust ability to preserve identity
while adhering to text descriptions in both single and multi-
subject contexts.
To evaluate the core challenge of multi-subject complex
pose generation, we present the results on our benchmark
derived from COCO Keypoints in Table 3. ASTRA demon-
strates a significant lead in OKS score of 0.0452, confirming
its superior ability to adhere to precise structural guidance.
Critically, this top-tier pose accuracy is achieved without
compromising identity, as evidenced by our state-of-the-art
CLIP-I and CLIP-T scores of 0.7087 and 0.3194, and com-
petitive DINO scores. This quantitatively validates that our
framework effectively decouples appearance and structure,
overcoming the trade-off that plagues existing methods.
4.4. Ablation Study
We conducted an ablation study on our COCO-based bench-
mark to validate the effectiveness of each key component in
ASTRA. The results are shown in Table 4.
Effect of RAG-Pose.As shown in row (a), removing the
RAG-Pose module causes a severe degradation in pose ac-
curacy, with the OKS score plummeting from 0.0452 to
0.0095. This result demonstrates that RAG-Pose is essen-
tial for providing the clean and explicit structural priors re-
quired for high-fidelity pose control.
Effect of Asymmetric Positional Encoding.Rows (b)
and (c) demonstrate the critical impact of our asymmetric
7

design.Applying a uniform RoPE to all concatenated to-
kens (b) leads to a complete performance collapse in both
identity (CLIP-I: 0.7087→0.3813) and pose (OKS: 0.0452
→0.0041), proving it is incompatible with our direct to-
ken concatenation. While a uniform UNOPE (c) also fails,
it particularly struggles to learn the pose (OKS drops to
0.0076), with other metrics degrading due to this inappro-
priate encoding.
Effect of Disentangled Semantic Modulation (DSM).
Removing the DSM stream (row d) results in a clear drop
in identity preservation (CLIP-I: 0.7087→0.6754) while
pose accuracy remains largely stable. This indicates that
DSM successfully offloads the identity-learning task to a
dedicated module, leading to more robust identity genera-
tion without interfering with the model’s primary structural
control task.
Method CLIP-I↑DINO↑CLIP-T↑
Oracle (reference images) 0.885 0.774 N/A
Textual Inversion[8] 0.780 0.569 0.255
DreamBooth[30] 0.803 0.668 0.305
BLIP-Diffusion[18] 0.805 0.670 0.302
ELITE[41] 0.772 0.647 0.296
Re-Imagen [4] 0.740 0.600 0.270
BootPIG [26] 0.797 0.674 0.311
SSR-Encoder[48] 0.821 0.612 0.308
RealCustom++ [22] 0.794 0.702 0.318
OmniGen[45] 0.801 0.693 0.315
OminiControl[37] 0.799 0.684 0.312
FLUX.1 IP-Adapter[38] 0.820 0.582 0.288
UNO[43] 0.835 0.7600.304
ASTRA(Ours) 0.8470.6990.330
Table 1. Quantitative results for single-subject driven generation
on DreamBench. We present the oracle results in the first row and
compare both tuning and tuning-free methods. Thebestresults
are shown in bold and the second-best result is underlined.
Method CLIP-I↑DINO↑CLIP-T↑
DreamBooth [30] 0.695 0.430 0.308
BLIP-Diffusion [18] 0.698 0.464 0.300
SubjectDiffusion [21] 0.696 0.506 0.310
MIP-Adapter [13] 0.726 0.482 0.311
MS-Diffusion [40] 0.726 0.525 0.319
OmniGen [45] 0.722 0.5110.331
UNO [43] 0.733 0.542 0.322
ASTRA (Ours) 0.745 0.5540.326
Table 2. Quantitative results for multi-subject driven generation
based on DreamBench. Our method achieves state-of-the-art per-
formance among both tuning and tuning-free methods.Method OKS↑CLIP-I↑DINO↑CLIP-T↑
DreamO[23] 0.0015 0.6736 0.4665 0.3084
OmniGen[45] 0.0219 0.6865 0.4284 0.2999
OmniGen2[42] 0.0270 0.69380.46700.3075
UNO[43] 0.0277 0.6857 0.4283 0.2970
IP-Adapter*[38] 0.0314 0.6942 0.4310 0.3050
PhotoMaker*[19] 0.0308 0.6920 0.4305 0.3042
ASTRA (Ours) 0.0452 0.70870.44570.3194
Table 3. Evaluation on the COCO-based Complex Pose Bench-
mark. All compared methods receive identical multi-person pose
maps and identity crops. Methods marked with * useControlNet
and OpenPosefor explicit pose conditioning. ASTRA achieves
the highest OKS while maintaining strong identity fidelity.
Configuration OKS↑CLIP-I↑DINO↑CLIP-T↑
(a) w/o RAG-Pose 0.0095 0.6815 0.4173 0.3061
(b) w/ Symmetric RoPE 0.0041 0.3813 0.2255 0.1705
(c) w/ Symmetric UNOPE 0.0076 0.6028 0.3591 0.2876
(d) w/o DSM 0.0345 0.6754 0.4208 0.3072
(e)ASTRA (Ours) 0.0452 0.7087 0.4457 0.3194
Table 4. Ablation study of ASTRA’s key components on the
COCO-based complex pose benchmark. The performance of the
full model (e) is the main result, while rows (a-d) show the impact
of removing or altering each component.
5. Conclusion
In this work, we address the challenge of generating multi-
subject images with both high identity fidelity and precise
pose control by identifying and resolving a core “positional
encoding conflict” in existing models. We introduce AS-
TRA, a novel framework that architecturally disentangles
appearance and structure by integrating three key compo-
nents: a Retrieval-Augmented Pose (RAG-Pose) pipeline
for clean structural priors, an asymmetric positional encod-
ing strategy to separate identity and pose signals, and a
Disentangled Semantic Modulation (DSM) stream to en-
hance identity robustness. Extensive experiments and ab-
lation studies validate that our method effectively decou-
ples the representation of “who” from “where,” achieving
state-of-the-art results on complex pose and multi-subject
benchmarks and paving the way for more controllable and
reliable generative models. Beyond the benchmark tasks,
the proposed design principles can be readily extended
to other controllable generation scenarios involving struc-
tured or multi-modal conditions. As an additional contribu-
tion, we will release the curated pose database and the re-
trieval library used in our framework, providing valuable re-
sources for future research in pose-conditioned and identity-
preserving generation.
8

References
[1] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas
M¨uller, and Bj ¨orn Ommer. Retrieval-augmented diffusion
models.Advances in Neural Information Processing Sys-
tems, 35:15309–15324, 2022. 3
[2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. In-
structpix2pix: Learning to follow image editing instructions.
InProceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 18392–18402, 2023. 2
[3] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh.
Realtime multi-person 2d pose estimation using part affinity
fields. InCVPR, 2017. 4
[4] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W
Cohen. Re-imagen: Retrieval-augmented text-to-image gen-
erator.arXiv preprint arXiv:2209.14491, 2022. 8
[5] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye
Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao,
Yilin Wang, et al. Unireal: Universal image generation and
editing via learning real-world dynamics. InProceedings of
the Computer Vision and Pattern Recognition Conference,
pages 12501–12511, 2025. 2
[6] Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh.
L2 regularization for learning kernels.arXiv preprint
arXiv:1205.2653, 2012. 4
[7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M ¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
InForty-first international conference on machine learning,
2024. 2
[8] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patash-
nik, Amit H Bermano, Gal Chechik, and Daniel Cohen-
Or. An image is worth one word: Personalizing text-to-
image generation using textual inversion.arXiv preprint
arXiv:2208.01618, 2022. 2, 8
[9] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and
Haofen Wang. Retrieval-augmented generation for large lan-
guage models: A survey.arXiv preprint arXiv:2312.10997,
2(1), 2023. 2, 3
[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models.Advances in neural information
processing systems, 33:6840–6851, 2020. 2
[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models.ICLR,
1(2):3, 2022. 5, 7
[12] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen,
Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing
Gong, William Cohen, et al. Instruct-imagen: Image gen-
eration with multi-modal instruction. InProceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 4754–4763, 2024. 3
[13] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng
Yu, and Jie Song. Resolving multi-condition confusion for
finetuning-free personalized image generation. InProceed-ings of the AAAI Conference on Artificial Intelligence, pages
3707–3714, 2025. 7, 8
[14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perel-
man, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli-
hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.
arXiv preprint arXiv:2410.21276, 2024. 3
[15] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su,
Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and
Pascale Fung. Survey of hallucination in natural language
generation.ACM computing surveys, 55(12):1–38, 2023. 2
[16] Black Forest Labs. Flux, 2024. 2, 3, 7
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems,
33:9459–9474, 2020. 2, 3
[18] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-
trained subject representation for controllable text-to-image
generation and editing.Advances in Neural Information Pro-
cessing Systems, 36:30146–30166, 2023. 8
[19] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-
Ming Cheng, and Ying Shan. Photomaker: Customizing re-
alistic human photos via stacked id embedding. InProceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 8640–8650, 2024. 5, 8
[20] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll ´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
European conference on computer vision, pages 740–755.
Springer, 2014. 7
[21] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-
diffusion: Open domain personalized text-to-image genera-
tion without test-time fine-tuning. InACM SIGGRAPH 2024
Conference Papers, pages 1–12, 2024. 8
[22] Zhendong Mao, Mengqi Huang, Fei Ding, Mingcong Liu,
Qian He, and Yongdong Zhang. Realcustom++: Represent-
ing images as real-word for real-time customization.arXiv
preprint arXiv:2408.09744, 2024. 7, 8
[23] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze
Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen
Zhang, Xinghui Li, et al. Dreamo: A unified framework
for image customization.arXiv preprint arXiv:2504.16915,
2025. 2, 7, 8
[24] Maxime Oquab, Timoth ´ee Darcet, Th ´eo Moutakanni, Huy
V o, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193, 2023. 7
[25] William Peebles and Saining Xie. Scalable diffusion models
with transformers. InProceedings of the IEEE/CVF inter-
national conference on computer vision, pages 4195–4205,
2023. 2, 3
[26] Senthil Purushwalkam, Akash Gokul, Shafiq Joty, and Nikhil
Naik. Bootpig: Bootstrapping zero-shot personalized im-
age generation capabilities in pretrained diffusion models. In
European Conference on Computer Vision, pages 252–269.
Springer, 2024. 8
9

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. InInternational conference on machine learning, pages
8748–8763. PmLR, 2021. 7
[28] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-
ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-
Yuan Wu, Ross Girshick, Piotr Doll ´ar, and Christoph Feicht-
enhofer. Sam 2: Segment anything in images and videos.
arXiv preprint arXiv:2408.00714, 2024. 5
[29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks. InProceedings of
the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing. Association for Computational Linguis-
tics, 2019. 4
[30] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch,
Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine
tuning text-to-image diffusion models for subject-driven
generation. InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 22500–
22510, 2023. 2, 5, 7, 8
[31] Rotem Shalev-Arkushin, Rinon Gal, Amit H Bermano,
and Ohad Fried. Imagerag: Dynamic image retrieval
for reference-guided image generation.arXiv preprint
arXiv:2502.09411, 2025. 3
[32] Shelly Sheynin, Oron Ashual, Adam Polyak, Uriel Singer,
Oran Gafni, Eliya Nachmani, and Yaniv Taigman. Knn-
diffusion: Image generation via large-scale retrieval.arXiv
preprint arXiv:2204.02849, 2022. 3
[33] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. In-
stantbooth: Personalized text-to-image generation without
test-time finetuning. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
8543–8552, 2024. 7
[34] Anshumali Shrivastava and Ping Li. Asymmetric lsh (alsh)
for sublinear time maximum inner product search (mips).
Advances in neural information processing systems, 27,
2014. 4
[35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan,
and Surya Ganguli. Deep unsupervised learning using
nonequilibrium thermodynamics. InInternational confer-
ence on machine learning, pages 2256–2265. pmlr, 2015. 2
[36] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen
Bo, and Yunfeng Liu. Roformer: Enhanced transformer with
rotary position embedding.Neurocomputing, 568:127063,
2024. 5
[37] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue,
and Xinchao Wang. Ominicontrol: Minimal and uni-
versal control for diffusion transformer.arXiv preprint
arXiv:2411.15098, 2024. 2, 7, 8
[38] InstantX Team. Instantx flux.1-dev ip-adapter page, 2024. 8
[39] Qwen Team. Qwen2.5: A party of foundation models, 2024.
4[40] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and
Hao Jiang. Ms-diffusion: Multi-subject zero-shot im-
age personalization with layout guidance.arXiv preprint
arXiv:2406.07209, 2024. 2, 7, 8
[41] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei
Zhang, and Wangmeng Zuo. Elite: Encoding visual con-
cepts into textual embeddings for customized text-to-image
generation. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 15943–15953, 2023.
8
[42] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin
Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie
Zhou, et al. Omnigen2: Exploration to advanced multimodal
generation.arXiv preprint arXiv:2506.18871, 2025. 3, 7, 8
[43] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei
Ding, and Qian He. Less-to-more generalization: Unlocking
more controllability by in-context generation.arXiv preprint
arXiv:2504.02160, 2025. 2, 5, 7, 8
[44] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong
Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan.
Florence-2: Advancing a unified representation for a vari-
ety of vision tasks.arXiv preprint arXiv:2311.06242, 2023.
5
[45] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xin-
grun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun
Huang, and Zheng Liu. Omnigen: Unified image genera-
tion. InProceedings of the Computer Vision and Pattern
Recognition Conference, pages 13294–13304, 2025. 7, 8
[46] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-
adapter: Text compatible image prompt adapter for text-to-
image diffusion models.arXiv preprint arXiv:2308.06721,
2023. 2
[47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding
conditional control to text-to-image diffusion models. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 3836–3847, 2023. 2
[48] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng
Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al.
Ssr-encoder: Encoding selective subject representation for
subject-driven generation. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 8069–8078, 2024. 7, 8
10