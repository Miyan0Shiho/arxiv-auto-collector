# RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation

**Authors**: Hao Li, Yuhao Wang, Wenning Hao, Pingping Zhang, Dong Wang, Huchuan Lu

**Published**: 2026-03-04 01:02:04

**PDF URL**: [https://arxiv.org/pdf/2603.03617v1](https://arxiv.org/pdf/2603.03617v1)

## Abstract
RGB-Thermal (RGBT) tracking aims to achieve robust object localization across diverse environmental conditions by fusing visible and thermal infrared modalities. However, existing RGBT trackers rely solely on initial-frame visual information for target modeling, failing to adapt to appearance variations due to the absence of language guidance. Furthermore, current methods suffer from redundant search regions and heterogeneous modality gaps, causing background distraction. To address these issues, we first introduce textual descriptions into RGBT tracking benchmarks. This is accomplished through a pipeline that leverages Multi-modal Large Language Models (MLLMs) to automatically produce texual annotations. Afterwards, we propose RAGTrack, a novel Retrieval-Augmented Generation framework for robust RGBT tracking. To this end, we introduce a Multi-modal Transformer Encoder (MTE) for unified visual-language modeling. Then, we design an Adaptive Token Fusion (ATF) to select target-relevant tokens and perform channel exchanges based on cross-modal correlations, mitigating search redundancies and modality gaps. Finally, we propose a Context-aware Reasoning Module (CRM) to maintain a dynamic knowledge base and employ a Retrieval-Augmented Generation (RAG) to enable temporal linguistic reasoning for robust target modeling. Extensive experiments on four RGBT benchmarks demonstrate that our framework achieves state-of-the-art performance across various challenging scenarios. The source code is available https://github.com/IdolLab/RAGTrack.

## Full Text


<!-- PDF content starts -->

RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented
Generation
Hao Li1,2, Yuhao Wang2, Wenning Hao1∗, Pingping Zhang2∗, Dong Wang3, Huchuan Lu2,3
1College of Command and Control Engineering, Army Engineering University of PLA
2School of Future Technology, Dalian University of Technology
3School of Information and Communication Engineering, Dalian University of Technology
lihao@aeu.edu.cn, 924973292@mail.dlut.edu.cn, hwnbox@aeu.edu.cn,
{zhpp,wdice,lhchuan}@dlut.edu.cn
Abstract
RGB-Thermal (RGBT) tracking aims to achieve robust ob-
ject localization across diverse environmental conditions by
fusing visible and thermal infrared modalities. However,
existing RGBT trackers rely solely on initial-frame visual
information for target modeling, failing to adapt to appear-
ance variations due to the absence of language guidance.
Furthermore, current methods suffer from redundant search
regions and heterogeneous modality gaps, causing back-
ground distraction. To address these issues, we first intro-
duce textual descriptions into RGBT tracking benchmarks.
This is accomplished through a pipeline that leverages
Multi-modal Large Language Models (MLLMs) to auto-
matically produce texual annotations. Afterwards, we pro-
poseRAGTrack, a novelRetrieval-AugmentedGeneration
framework for robust RGBT tracking. To this end, we intro-
duce a Multi-modal Transformer Encoder (MTE) for uni-
fied visual-language modeling. Then, we design an Adap-
tive Token Fusion (ATF) to select target-relevant tokens and
perform channel exchanges based on cross-modal correla-
tions, mitigating search redundancies and modality gaps.
Finally, we propose a Context-aware Reasoning Module
(CRM) to maintain a dynamic knowledge base and employ
a Retrieval-Augmented Generation (RAG) to enable tempo-
ral linguistic reasoning for robust target modeling. Exten-
sive experiments on four RGBT benchmarks demonstrate
that our framework achieves state-of-the-art performance
across various challenging scenarios. The source code is
available at https://github.com/IdolLab/RAGTrack.
1. Introduction
Visual object tracking [6–10, 30–36, 48, 52–59, 78–83]
aims to localize objects of interest across video sequences.
∗Corresponding author.
Feature Extraction
&
Relation ModelingPrediction
RAG
Template
TemplateMLLM
BroomFeature Extraction
&
Relation ModelingPredictionSearch Redundancy
&
Modality Gap
Template
TemplateBroom ?
Dustpan ?
Leg?
Token Selection
&
Channel Exchange
(a) Existing RGBT Tracker
(b) Our RAGTrackInadequate 
Appearance Modeling
Linguistic Reasoning
Figure 1. Comparison with different RGBT tracking paradigms.
(a) Existing RGBT trackers suffer from inadequate appearance
modeling, search redundancy, and modality gap. (b) Our RAG-
Track introduces linguistic reasoning, dynamic token selection,
and adaptive channel exchange.
Despite its wide applications in autonomous driving, intel-
ligent surveillance, and human-computer interaction, track-
ing relying solely on the visible (RGB) modality often fails
under low illumination. In contrast, the thermal infrared
(TIR) imaging has the advantage of all-weather capability,
providing complementary information of the RGB modal-
ity. To integrate these advantages, RGB-Thermal (RGBT)
tracking combines both modalities for robust performance.
However, current RGBT trackers [29, 39–44] model the
target using visual information from the initial frame, which
often leads to drift under severe appearance variations. This
limitation stems from two main factors. First, a single im-
age template offers limited information of the target, failing
to capture full appearance changes across different view-
points. Second, the inherent ambiguity of targets often
causes trackers to overemphasize unstable features, result-
ing in background distraction. As shown in Fig. 1, the
tracker may become confused among the broom, dustpan,
or the lower body of the pedestrian. Moreover, boundingarXiv:2603.03617v1  [cs.CV]  4 Mar 2026

box-based initialization is inconvenient in practical scenar-
ios, consequently hindering its widespread deployment.
On the other hand, recent advances in multi-modal ob-
ject tracking have focused on integrating features from di-
verse modalities. In previous works, multi-modal features
are first extracted independently, then fused through dedi-
cated fusion modules [65]. They establish connections of
multiple feature sources, including RGB and TIR templates
and search regions, operating at token-level. However, a
significant portion of search regions is redundant, contain-
ing substantial background regions and distractors that neg-
atively impact the tracking accuracy. Furthermore, hetero-
geneous modality gaps hinder the establishment of effec-
tive cross-modality correspondences. This raises a basic
question:How can we construct a target model with rich
semantic representations while mitigating the impact of
search redundancies and modality gaps?
To address these challenges, we introduce language as a
high-level representation to enhance RGBT tracking. Lan-
guage provides a more abstract understanding of objects
than images, addressing semantic limitations in visual rep-
resentation. Language-based target representation can of-
fer clear semantic information, including object categories,
appearance attributes and motion states, enabling effective
target-background separation. Recent advances in RGB-
Language (RGBL) tracking have demonstrated the potential
of language descriptions for robust tracking [9]. However,
these methods face the challenge of visual-language mis-
alignment as targets evolve across video frames. Recently,
Multi-modal Large Language Models (MLLMs) have ex-
hibited strong reasoning capabilities, delivering opportuni-
ties for incorporating textual descriptions into RGBT track-
ing. Despite these advancements, there is no RGBT track-
ing benchmarks with text annotations. To bridge this gap,
we extend current RGBT tracking benchmarks by introduc-
ing textual descriptions through MLLMs.
Building upon the textual descriptions, we propose a
novel RGBT tracking framework, named RAGTrack. As
depicted in Fig. 2, our framework consists of three key
components: Multi-modal Transformer Encoder (MTE),
Adaptive Token Fusion (ATF) and Context-aware Reason-
ing Module (CRM). The MTE utilizes text features for uni-
fied modeling with visual patch tokens. Meanwhile, ATF
utilizes linguistic cues to obtain text-guided attention scores
for identifying target-relevant search tokens, reducing noisy
interference. This module further employs adaptive chan-
nel exchange to dynamically bridge heterogeneous features
across modalities. As a result, it maintains semantic con-
sistency and mitigates modality gaps. The CRM imple-
ments a Retrieval-Augmented Generation (RAG) paradigm
that establishes a continuous reasoning cycle across frames.
This module maintains a dynamic knowledge base, retrieves
historically relevant features, and generates adaptive tar-get descriptions via MLLMs. With the above components,
our RAGTrack enables comprehensive temporal reasoning,
leveraging both visual evidence and linguistic context to
maintain target identities under complex scenarios. Exten-
sive experiments on four benchmarks demonstrate the ef-
fectiveness of our method, achieving state-of-the-art perfor-
mance. In summary, our main contributions are as follows:
• To the best of our knowledge, we are the first to intro-
duce textual descriptions into RGBT tracking, extending
existing benchmarks with semantic annotations.
• We propose RAGTrack, a novel framework that leverages
Retrieval-Augmented Generation (RAG) to enhance lin-
guistic reasoning for robust target modeling.
• We design the Adaptive Token Fusion (ATF) to address
search redundancies and modality gaps through dynamic
token selection and adaptive channel exchange.
• Extensive experiments on four RGBT tracking bench-
marks demonstrate that RAGTrack outperforms existing
methods in both accuracy and robustness.
2. Related Work
2.1. RGB-Thermal Tracking
RGBT tracking [38] combines RGB and TIR modalities to
overcome single-modality limitations in complex scenarios.
Early works [87] use MDNet [47] for feature extraction,
while CMPP [63] employs affinity-based pattern propaga-
tion. Subsequent methods advance fusion strategies, such
as template-bridged interactions [17] and Mamba-based
fusion [43]. The Mixture of Experts (MoE) paradigms,
exemplified by XTrack [59] and FlexTrack [58], further
enhance robustness through expert routing and heteroge-
neous fusion. Meanwhile, temporal modeling is refined by
TATrack [64], CSTrack [8] and STTrack [15]. Recently,
ViPT [85], BAT [2], SDSTrack [14], OneTracker [13] and
RDTTrack [86] leverage visual prompt learning to boost
tracking performance. However, current methods solely de-
pend on visual information, which becomes unstable dur-
ing tracking. Different from existing methods, our method
leverages textual descriptions to complement visual features
and distinguish similar-appearance targets.
2.2. RGB-Language Tracking
RGBL tracking [7] utilizes both natural language descrip-
tions and visual references to localize the target across
video sequences. Li et al. [33] pioneer this field by ex-
tending tracking datasets with linguistic annotations. Wang
et al. [67] and Zhou et al. [84] advance visual grounding
techniques to integrate language semantics with tracking.
Subsequently, SNLT [6], All-in-One [73] and OVLM [74]
develop multi-modal fusion approaches through dynamic
aggregation, unified Transformers and memory mecha-
nism, respectively. CiteTracker [30] and MMTrack [80]

Template
Search Region
Template
Search Region
RGB Branch TIR Branch···
···
···
C
C ·········Cross Attention
Cross AttentionQCross Attention
Q
Cross Attention
QK
V
K
VK V
K V
Context -aware Reasoning Module
Multi -modal 
Transformer EncoderMulti -modal 
Transformer Encoder
Patch Embedding

Hadamard ProductC Concatenation
RGB/TIR Reasoning TokenRGB/TIR Template Token
RGB/TIR Search TokenRGB/TIR Text TokenInitial Text Token
··· ······ ···
··· ···
Text
 EncoderA sequence of a [  ] 
object: { description }Extract···
Load
Retrieval
······LoadRetrieval······
Augmentation Q···GuidanceShuffle
··· ··· ··· ······ ··· ···
···
Update
Generation
··· ··· ··· ······
··· ··· ······
ShuffleGuidance
···Augmentation 
Update
Patch Embedding
1. Awhite catisstanding onthe
concrete road .
    ******
440. A white  cat sits on the edge  
of a stone  platform , ears perked  
up alertly,  eyes fixated  with keen  
attention .
    ******
ATF
Qwen-VL
······C
Positional Embedding

Prediction  HeadConstruction
Construction
Figure 2. Overall framework. Our method begins by tokenizing input texts and images with reasoning tokens. MTE then performs
unified visual-language modeling, while ATF utilizes text-guided attention to dynamically select target-relevant tokens and enables adaptive
channel exchange. Subsequently, CRM retrieves relevant contexts from a dynamic knowledge base for context-aware reasoning. Finally,
the prediction head outputs tracking results, which are used by MLLMs to generate updated textual descriptions for following frames.
adopt generative frameworks for text-based target estima-
tion, while VLTTT [12] and UVLTrack [45] leverage con-
trastive learning for vision-language alignment. Recently,
LLM-driven methods, such as DTLLM-VLT [31], Chat-
Tracker [57], DUTrack [32] and ReasoningTrack [68], en-
hance robustness through multimodal reasoning and dy-
namic language generation. However, current methods suf-
fer from insufficient visual-language feature association and
ineffective background interference suppression. Different
from existing methods, our method utilizes text features to
address search redundancies and modality gaps through dy-
namic token selection and adaptive channel exchange.
2.3. Retrieval-Augmented Generation
RAG [22] can enhance LLMs by retrieving external knowl-
edge during inference to mitigate hallucinations and im-
prove the factual accuracy in domain-specific tasks. While
widely adopted in applications [11, 60], its integration into
visual tracking remains limited. As an outstanding work,
TrackingMiM [34] proposes a query-based retrieval module
for UA V tracking. However, current methods merely reuse
pre-stored features. Different from existing methods, our
method introduces RAG into RGBT tracking for the first
time, enabling context-aware linguistic reasoning.
3. Methodology
In this paper, we propose RAGTrack, which includes
three key components: Multi-modal Transformer Encoder
(MTE), Adaptive Token Fusion (ATF), and Context-aware
Reasoning Module (CRM). As shown in Fig. 2, MTE servesas the foundation for processing heterogeneous inputs. ATF
dynamically captures target-relevant tokens and fuses cross-
modal features. CRM leverages a RAG mechanism to main-
tain target identities through temporal semantic reasoning.
Details are described as follows.
3.1. Overall Framework
Our framework localizes targets across video sequences by
fusing complementary information from RGB, TIR, and
language modalities. At time stept, the input consists of
search imagesXt
m∈R3×H x×Wxand multi-modal refer-
ences comprising template imagesZt
m∈R3×H z×Wzand
language descriptionsLt. Here,HandWdenote the height
and width of the images, andm∈ {B, R}represents the
RGB or TIR modality. The MTE employs parameter-shared
branches for processing both RGB and TIR modalities. The
encoder processes these inputs via patch embedding and to-
kenization to generate tokens, which are concatenated and
fed into the visual-language unified modeling. These to-
kens then undergo ATF to mitigate search redundancies and
modality gaps, followed by CRM for linguistic refinement.
Finally, the refined features are fed into the prediction head
to produce tracking results.
3.2. Multi-modal Transformer Encoder
To extract multi-modal features, we introduce MTE for uni-
fied visual-language modeling. We process template and
search images through a three-stage downsampling [77]. It
transforms the template and search images into patch tokens
ˆZt
m∈RNz×CandˆXt
m∈RNx×C, whereNrepresents the

Channel AttentionTransposeAdaptive 
Channel Exchange··· ··· ··· ··· ··· ··· ··· ···
··· ··· ··· ··· ··· ··· ··· ···Dynamic Token Selection Dynamic Token SelectionSelf-Attention Self-AttentionCross -Modal Fusion +
··· ··· ··· ······ ··· ··· ···
··· ··· ··· ······ ··· ··· ······
························ ··· ··· ······ ··· ··· ······
·····················
2xr
BA
2xx
BA
2xh
BA
2xz
BA
2xr
RA
2xx
RA
2xh
RA
2xz
RA
BA
RAFigure 3. Details of our proposed ATF.
number of tokens, andCrepresents the feature dimension.
Afterwards, we introduce a sequence prefixEtto en-
hance the temporal awareness and address the potential mis-
alignment between visual contents and language descrip-
tions across frames. This prefix combines a fixed textual
prompt with learnable tokens. Specifically,Etis formulated
as“A sequence of a [∗] object:”, where[∗]is replaced by
learnable tokens. We concatenateEtandLtto form the
text inputHt= [Et,Lt]. Then,Htis encoded by the text
encoderTto obtain semantic features ˆHt∈RNh×C:
ˆHt=T(Ht).(1)
Subsequently, we concatenate the reasoning tokenRt
m∈
RNr×C, the text token ˆHt, the template token ˆZt
mand the
search token ˆXt
minto a unified token sequence:
F0
m=h
Rt
m;ˆHt;ˆZt
m;ˆXt
mi
,(2)
where[; ]denotes the concatenation operation. The unified
visual-language modeling is formulated as:
ˆFl−1
m= MHSA(Fl−1
m,Fl−1
m,Fl−1
m),
˜Fl−1
m=Fl−1
m+ LN(δ 1·ˆFl−1
m),
Fl
m=˜Fl−1
m+ LN(δ 2·MLP( ˜Fl−1
m)),(3)
whereMHSAstands for multi-head self-attention [5],LN
represents the layer normalization [1] andMLPdenotes the
multilayer perceptron.l∈ {1,2, . . . , L}indicates the layer
index.δ 1andδ 2are two learnable parameters. By unified
visual-language modeling, semantic information from text
is effectively leveraged to enhance feature discrimination.
3.3. Adaptive Token Fusion
To address search redundancies and modality gaps, we pro-
pose ATF for dynamic token selection and adaptive chan-
nel exchange. Traditional fusion methods suffer from in-
efficient token utilization and significant feature discrepan-
cies, leading to degraded tracking performance. In contrast,
ATF identifies target-relevant tokens and adapts channel ex-
changes based on cross-modality correlations.
In fact, attention scoresA mfrom the self-attention mod-
ule serve as an effective indicator for token significance,
while simultaneously guiding the model to excavate con-
textual information. This fact motivates our use of attentionscores as metrics for token selection. As shown in Fig. 3,
the calculation for each term is as follows:
Ax2o
m=SoftmaxQx
m(Ko
m)T
√
d
, (4)
whereAx2r
m,Ax2h
m,Ax2z
m, andAx2x
mrepresent correlations
between search and reasoning, text, template, and search to-
kens, respectively. Specially, to mitigate template noise, we
extract the center region of the template image, as it con-
tains sufficient target information [72]. We aggregate these
scores to identify key tokens as follows:
Atotal
m=Ax2r
m+Ax2h
m+Ax2z
m+Ax2x
m. (5)
Tokens with higherAtotal
mare retained using a retention ra-
tioγ. This parameter-free token selection avoids redundant
computations by directly reusing the attention scores.
For channel exchange, we address the challenge of
modality gaps by establishing channel-level associations
between modalities. More specifically, we first compute
cross-modal relevanceSbetween features along the chan-
nel dimension:
S= 
Fl
BTWl
B 
Fl
RTWl
RT
,(6)
whereWl
BandWl
Rare learnable parameters. Then, we av-
erageSalong the channel dimension to determine channel
importance, then select corresponding channels for adaptive
exchange based on the exchange ratioσ. Finally, we con-
catenate the features of RGB and TIR modalities along the
token dimension and apply the cross-modal fusionM:
Fl
B,Fl
R=M 
Fl
B;Fl
R
,(7)
whereMis a MLP layer. This channel exchange bridges
the independently processed RGB and TIR modalities dur-
ing feature extraction, enabling the model to learn cross-
modal correlations and enhance feature discrimination.
Our ATF mitigates search redundancies and modality
gaps through dynamic token selection and adaptive channel
exchange. It facilitates information transmission between
RGB and TIR modalities, enabling the model to focus on
the most relevant features for enhanced robustness.
3.4. Context-aware Reasoning Module
To enhance temporal modeling, we propose the CRM for
linguistic reasoning. As illustrated in Fig. 2, it includes a
RAG mechanism that operates through four stages: con-
struction, retrieval, augmentation, and generation.
Construction.We construct a local knowledge base
from historical text features. This knowledge base is
maintained as a set ofnfeature embeddingsD m=n
ˆH1
m,···, ˆHn
mo
, which is dynamically updated during in-
ference to suppress redundancy. Specifically, a new text

feature ˆHt
mis added only if its maximum cosine similar-
ity with existing entries falls below a thresholdλ:
max
ˆHim∈DmˆHt
m·ˆHi
mˆHtm
2ˆHim
2< λ.(8)
Retrieval.Given a query feature ˆHt
m, the retrieverO
selects the top-kmost relevant features fromD m:
Vm=O
ˆHt
m,Dm
,(9)
whereV m⊆D mand|V m|=k. To refine the search
features, we then apply intra-modal cross-attentionΦ:
¯Xt
m=ˆXt
m+ Φ
ˆXt
m,Vm
.(10)
Augmentation.For temporal propagation across video
frames, we enhance the reasoning tokens to capture both lo-
cal and global context. First, we perform an average pool-
ingPon the reasoning features, text features, and template
features along the token dimension:
¯Rt
m,¯Ht
m,¯Zt
m=P(Rt
m,ˆHt
m,ˆZt
m).(11)
Subsequently, we concatenate these pooled features
along the channel dimension and apply the guidanceGto
inject current frame cues:
Rt+1
m=G( ¯Rt
m,¯Ht
m,¯Zt
m),(12)
whereGis a MLP layer. Then, the updated reasoning token
Rt+1
mis propagated to the next frame.
Finally, we enhance the feature representation through a
three-step temporal augmentation:
ˆRt+1
m=Rt+1
m+ Φ(Rt+1
m,¯Xt
m),
˜Rt+1
m=ˆRt+1
m+MLP( ˆRt+1
m),
˜Xt
m=¯Xt
m⊗(˜Rt+1
m)T⊙¯Xt
m,(13)
where⊗and⊙denotes the matrix multiplication and
hadamard product.
Generation.To overcome the limitations of static lan-
guage annotations, we employ MLLMs to dynamically gen-
erate context-aware target descriptions during inference.
The input consists of the search image and a structured
prompt:“Describe the object located in the image at
<box>(x,y,x+w,y+h)</box>. Focus on distinctive visual
features, motion patterns, and key identifiers to distinguish
it from background elements and distractors. ”The result-
ing descriptions continuously refresh the multi-modal ref-
erence, improving cross-frame appearance reasoning.
Our CRM maintains a dynamic knowledge base through
iterative retrieval and update mechanisms, and enhances
temporal coherence via reasoning token propagation. It
further generates adaptive target descriptions to overcome
static annotation limitations and adapt to scene variations.3.5. Prediction Head and Loss Function
The refined multi-modal features ˜Xt
mare first reshaped into
a 2D spatial representation. These features are then con-
catenated, followed by channel compression with convolu-
tional layers. Finally, they are transformed through a Fully
Convolutional Network (FCN) consisting of stacked Conv-
BN-ReLU layers [18, 46]. The prediction head generates
three outputs: classification scores, spatial offsets, and nor-
malized sizes. The final bounding box is constructed at the
position with maximum classification score by combining
the corresponding predictions.
For model optimization, we adopt a multi-task loss func-
tion combining the focal lossL cls[21] for classification,
along withL 1loss and generalized IoU lossL iou[50] for
regression. The overall loss function is formulated as:
L=L cls+λ iouLiou+λL1L1,(14)
whereλ iouandλ L1are the hyper-parameters.
4. Experiment
4.1. Datasets and Evaluation Metrics
Our experiments are conducted on four RGBT track-
ing benchmarks. GTOT [24] contains 50 RGB-T se-
quences covering seven primary challenges. RGBT210 [25]
contains 210 sequences and 12 challenging attributes.
RGBT234 [26] includes 234 sequences with increased di-
versities. LasHeR [28] comprises 1,224 sequences anno-
tated with 19 fine-grained attributes. To enable language-
aware RGBT tracking, we extend these benchmarks with
textual descriptions through a two-step generation pipeline.
In the first step, we generate informative text descriptions
by inputting the image and bounding box into MLLMs,
overcoming the time-consuming and labor-intensive nature
of manual annotations. Then, we refine these generated
descriptions through MLLMs and human experts to miti-
gate hallucinations, ensuring high-quality semantic annota-
tions. For the training set of LasHeR, we annotate all frames
across 979 sequences with 514,081 textual descriptions to
support model training. The test sets of LasHeR and the
other three benchmarks are annotated with first-frame de-
scriptions to evaluate the textual reasoning capability dur-
ing inference. Following standard protocols, we adopt the
Precision Rate (PR) and Success Rate (SR) as primary eval-
uation metrics. For LasHeR, we additionally use the Nor-
malized Precision Rate (NPR) to account for target scale
variations. On GTOT and RGBT234 with modality mis-
alignment, we report the Maximum Precision Rate (MPR)
and Maximum Success Rate (MSR) for fair comparison.
4.2. Implementation Details
We implement RAGTrack using PyTorch on 4 NVIDIA
V100 GPUs with a batch size of 16. The text encoder em-

Table 1. Performance comparison on four RGBT tracking benchmarks. The best results are in bold.
Method Publication ResolutionGTOT RGBT210 RGBT234 LasHeR
MPR↑MSR↑ PR↑SR↑ MPR↑MSR↑ PR↑NPR↑SR↑
CMPP [63] CVPR 2020 256×256 92.6 73.8 − − 82.3 57.5 − − −
CAT [27] ECCV 2020 256×256 88.9 71.7 79.2 53.3 80.4 56.1 45.0 39.5 31.4
APFNet [70] AAAI 2022 256×256 90.5 73.7 − − 82.7 57.9 50.0 43.9 36.2
CMD [75] CVPR 2023 256×256 89.2 73.4 − − 82.4 58.4 59.0 54.6 46.4
ViPT [85] CVPR 2023 256×256 − − − − 83.5 61.7 65.1−52.5
TBSI [17] CVPR 2023 256×256 − − 85.3 62.5 87.1 63.7 69.2 65.7 55.6
TATrack [64] AAAI 2024 256×256 − − 85.3 61.8 87.2 64.4 70.2 66.7 56.1
BAT [2] AAAI 2024 256×256 − − − − 86.8 64.1 70.2−56.3
GMMT [61] AAAI 2024 256×256 − − − − 87.9 64.7 70.7 67.0 56.6
Un-Track [69] CVPR 2024 256×256 − − − − 83.7 61.8 66.7−53.6
SDSTrack [14] CVPR 2024 256×256 − − − − 84.8 62.5 66.5−53.1
OneTracker [13] CVPR 2024 384×384 − − − − 85.7 64.2 67.2−53.8
CKD [39] ACM MM 2024 256×256 93.2 77.2 88.4 65.2 90.0 67.4 73.2 69.3 58.1
PTrMA [44] TIM 2025 256×256 − − 86.0 63.4 87.6 65.2 71.5 67.5 56.8
TVTracker [10] IoTJ 2025 256×256 − − 87.6 63.7 88.6 64.8 72.6 68.4 57.5
AINet [43] AAAI 2025 384×384 − − 87.5 64.8 89.2 67.3 74.2 70.1 59.1
SUTrack [4] AAAI 2025 384×384 − − − − 92.1 69.2 75.8−60.9
STTrack [15] AAAI 2025 256×256 − − − − 89.8 66.7 76.0−60.3
CAFormer [71] AAAI 2025 256×256 91.8 76.9 85.6 63.2 88.3 66.4 70.0 66.1 55.6
TBSI-Ext [23] TPAMI 2025 256×256 − − − − 91.0 67.0 75.5 71.5 59.6
CMDTrack [76] TPAMI 2025 256×256 − − − − 85.9 61.8 68.8−56.6
UM-ODTrack [83] TPAMI 2025 384×384 − − − − 91.5 69.2 74.4−58.8
XTrack [59] ICCV 2025 256×256 − − − − 87.4 64.9 69.1−55.7
SMSTracker [3] ICCV 2025 256×256 − − − − 86.9 64.5 70.3−56.0
AETrack [88] TCSVT 2025 256×256 − − 90.4 66.3 91.6 68.8 74.7 71.0 59.6
MambaVT [20] TCSVT 2025 256×256 94.1 75.3 88.0 63.7 88.9 65.8 73.0 69.5 57.9
MoETrack [62] TIP 2025 256×256 93.6 78.4 − − 88.1 65.1 72.1−57.8
DMD [16] TIP 2025 256×256 94.2 78.6 89.6 65.8 90.7 67.8 73.6 69.7 58.2
RAGTrack Ours 256×256 95.1 79.3 93.2 67.1 93.8 69.5 76.8 73.0 61.1
LI
(50.5, 68.1)MB
(56.6, 70.5)
HO
(44.3, 61.5)
TO
(57.4, 68.1)
PO
(60.7, 74.8)
NO
(83.7, 92.0)
ARC
(58.8, 71.3)
SV
(63.1, 77.1)
FM
(61.6, 75.2)
OV
(71.9, 78.3)FL
(51.9, 63.8)TC
(55.4, 69.8)CM
(61.0, 74.7)SA
(54.7, 68.9)BC
(62.5, 76.2)DEF
(65.4, 76.6)LR
(54.8, 68.5)AIV
(39.6, 57.2)HI
(67.2, 83.2)LI
(40.9, 54.0)MB
(45.3, 55.3)HO
(41.7, 53.7)
TO
(45.8, 53.5)
PO
(48.9, 59.2)
NO
(67.1, 75.4)
ARC
(49.0, 57.5)
SV
(50.9, 61.3)
FM
(50.1, 60.1)
OV
(62.2, 67.7)FL
(41.2, 49.6)TC
(44.3, 55.3)CM
(48.9, 59.2)SA
(44.4, 54.7)BC
(49.8, 60.3)DEF
(54.1, 62.1)LR
(40.5, 50.5)AIV
(36.4, 49.6)HI
(53.6, 65.9)Precision Rate of Challeng ingAttributes Success Rate of Challeng ingAttributes
Figure 4. Attribute-based evaluations on the LasHeR dataset.
ploys CLIP [49], while the visual backbone uses HiViT-B
initialized from SOT [32]. Training employs the AdamW
optimizer [37] with a learning rate10−4and a weight de-
cay10−4. Template images and search regions are pro-
cessed at fixed resolutions of 128×128 and 256×256, re-
spectively. We configure feature dimensionC= 512, text
tokenN h= 1, and reasoning tokenN r= 1. Our ATF is
deployed at the 6-th, 12-th, 18-th and 24-th layers of HiViT-
B with retention ratioγ= 85%. We select 256 channels for
exchange per modality. In CRM, the local knowledge base
maintainsn= 4entries, and the retrieval selectsk= 2fea-tures with thresholdλ= 1.0. The MLLM adopts QWen2.5-
VL-3B [66]. We train RAGTrack on the LasHeR training
set with loss weightsλ iou= 2andλ L1= 5to balance the
regression and classification terms.
4.3. Comparison with State-of-the-Art Trackers
We evaluate RAGTrack on four RGBT benchmarks against
state-of-the-art methods. Quantitative results are summa-
rized in Tab. 1, with detailed analysis provided below.
GTOT.Our method delivers state-of-the-art perfor-
mance with MPR 95.1% in Tab. 1. These results demon-
strate significant improvements: +1.5% MPR over MoE-
Track [62] and +4.0% MSR over MambaVT [20]. The per-
formance gain highlights the effectiveness of our method in
handling complex scenarios with modality discrepancies.
RGBT210.Our method shows outstanding perfor-
mance, achieving 93.2% PR and 67.1% SR in Tab. 1. It
exceeds AETrack [88] by a margin of +2.8% PR and shows
improved robustness than AINet [43] with a margin of
+2.3% SR. These results indicate the robustness of our
language-aware representation in improving tracking accu-
racy across diverse environmental conditions.

Table 2. Ablation study of key components on RGBT234.
Method MPR↑MSR↑
Baseline 87.9 64.5
w/ CRM* 89.1 65.0
w/ MTE+CRM* 91.1 66.7
w/ MTE+CRM 91.8 67.4
w/ MTE+CRM+ATF 93.8 69.5
Table 3. Comparison of fusion positions on RGBT234.
6 12 18 24 MPR↑MSR↑
✓92.2 67.9
✓ ✓92.5 68.5
✓ ✓ ✓92.9 68.8
✓✓✓✓ 93.8 69.5
Table 4. Comparison of fusion paradigms on RGBT234.
Module MPR↑MSR↑Params↓
TBSI [17] 92.8 67.6 145.9M
BSI [15] 93.1 68.2 103.6M
DFM [43] 92.7 67.8 110.3M
ATF (Ours) 93.8 69.5 101.8M
RGBT234.Our method achieves 93.8% MPR and
69.5% MSR in Tab. 1, outperforming SMSTracker [3] by
6.9% MPR and STTrack [15] by 2.8% MSR. This improve-
ment validates the superiority of our RAGTrack in leverag-
ing semantic information for accurate target localization.
LasHeR.Our method achieves leading performance
with 76.8% PR and 61.1% SR in Tab. 1, surpassing re-
cent trackers: +4.2% PR over TVTracker [10] and +5.4%
SR over XTrack [59]. Fig. 4 further reveals the attribute-
specific strengths: +10.7% PR in Total Occlusion (TO) and
+5.5% SR in Out-of-View (OV). Notably, these attribute-
specific advantages underscore the ability of our CRM in
maintaining target identities under appearance variations.
4.4. Ablation Studies
We conduct thorough ablation studies to evaluate key com-
ponents and parameters in RAGTrack, with results summa-
rized in the corresponding tables and figures below.
Component Analysis.We conduct ablation studies to
evaluate the contribution of key components. As detailed
in Tab. 2, the baseline with a backbone and convolutional
fusion achieves 87.9% MPR. The text-free CRM* brings
moderate improvement to 89.1% MPR, while incorporating
MTE with CRM* further increases performance to 91.1%
MPR. The complete CRM yields more clear gains at 91.8%
MPR, confirming the value of language information. The
full integration of MTE, CRM and ATF achieves superior
performance at 93.8% MPR, validating their complemen-
tary roles in semantic reasoning and feature interaction.
These results indicate that both linguistic enhancement and
cross-modal fusion are essential for robust RGBT tracking.
Effect of Fusion Positions.As shown in Tab. 3, fu-
sions at Layers 6, 12, 18 and 24 achieve the optimal perfor-Table 5. Comparison of augmentation mechanisms on RGBT234.
Module MPR↑MSR↑Params↓
Add 92.5 68.5 101.8M
Transformer 93.3 69.0 104.8M
Mamba 92.7 68.1 103.5M
Ours 93.8 69.5 101.8M
Table 6. Comparison of token configurations on RGBT234.
Reasoning Token Learnable Token
Nr MPR↑MSR↑ Length MPR↑MSR↑
1 93.8 69.5 0 92.8 68.4
2 93.5 69.4 2 93.8 69.5
4 93.2 68.9 4 92.6 68.7
Table 7. Comparison of attention scores on RGBT234.
Ax2rAx2h Ax2z Ax2x MPR↑MSR↑
✓92.0 67.8
✓ ✓92.9 68.7
✓ ✓ ✓93.7 69.0
✓ ✓ ✓ ✓ 93.8 69.5
mance. Early fusions capture low-level visual features but
lack semantics, while deeper fusions extract high-level con-
textual information but miss fine-grained details. The pro-
gressive fusion achieves complementary advantages, show-
ing the importance of cross-layer semantic fusion.
Effect of Different Fusion Paradigms.To validate the
effectiveness of ATF, we compare it with several represen-
tative fusion paradigms: TBSI [17] establishes template-
bridged interaction, BSI [15] introduces template-temporal
similarity for background suppression and DFM [43] mod-
els inter-modal differences for complementary fusion. In
contrast, ATF selects target-relevant tokens and exchanges
informative channels. As shown in Tab. 4, ATF achieves the
best performance (69.5% MSR) with the fewest trainable
parameters (101.8M). The results confirm that ATF enables
more discriminative and efficient cross-modal fusion.
Effect of Different Augmentation Mechanisms.We
evaluate different designs for the augmentation in CRM. As
shown in Tab. 5, our method achieves superior performance.
The addition operation yields a limited accuracy due to in-
sufficient representational capacity. Transformer increases
trainable parameters by 3.0%, whereas Mamba shows lower
performance. These results confirm that our design pro-
vides the balance for temporal propagation.
Effect of Reasoning Token Configurations.We evalu-
ate the effect of reasoning token quantityN r. As shown in
Tab. 6, the best result is achieved atN r= 1, outperform-
ing multi-token configurations. This improvement shows
that a single reasoning token can capture essential temporal
context, while additional tokens may introduce redundant
parameters without substantial performance benefits. In ad-
dition, our design effectively maintains temporal coherence
while minimizing computational overhead.

Performance Performance
Performance Performance
(d) Retrieval Count (c) Knowledge Base Size(a) Retention Ratio/% (b) Exchange Ratio/%Figure 5. Comparison with different hyper-parameters.
Effect of Learnable Token Lengths.We analyze the
effect of learnable token lengths in the sequence prefix. As
shown in Tab. 6, the 2-token configuration achieves the opti-
mal performance. The 0-token variant suffers a 1.0% MPR
drop, highlighting its limited capacity to sequence-specific
semantic variations. The 4-token variant exhibits a 0.8%
MSR decline, indicating that excessive tokens introduce re-
dundant information. These results confirm the necessity of
contextual tokens for accurate target representation.
Effect of Attention Scores in ATF.We investigate the
contribution of different attention scores in ATF. As pre-
sented in Tab. 7, performance improves progressively as
more attention scores are incorporated. The complete con-
figuration utilizing all four attention scores achieves the op-
timal results. Our experiments confirm that our multi-score
fusion strategy enables effective token selection.
Effect of the Retention Ratio in ATF.The retention ra-
tio determines the proportion of search tokens selected for
ATF. As shown in Fig. 5 (a),γ= 85%achieves the leading
results with 93.8% MPR. Lower values result in insufficient
target information retention, impairing the localization ac-
curacy. In contrast, higher values retain noisy or redundant
tokens, degrading feature discrimination. These findings
demonstrate that our setting preserves critical target infor-
mation while filtering noise.
Effect of the Exchange Ratio in ATF.The exchange ra-
tioσplays a key role in cross-modal interaction. According
to Fig. 5 (b), it yields peak performance withσ= 50%. At
σ= 0, poor modality fusion results in suboptimal perfor-
mance. Asσincreases to 50%, the improved feature rep-
resentation enhances target modeling. However, further in-
creasingσto 100% introduces excessive cross-modal inter-
ference, leading to performance degradation. This validates
σ= 50%as the optimal configuration, achieving a balance
between modality specificity and shared representation.
Effect of the Knowledge Base Size in CRM.We eval-
uate how the knowledge base sizenaffects the tracking ac-
(b) Search Region (c) Baseline (d) RAGTrack (a) Template Image 
The person stands near a
parked car, wearing apink
coat and dark pants, distinct
against thelight pavement
andsunny backdrop .A person  is squatting  down  
on the sidewalk,  wearing  a 
pink coat and black  pants . 
They  appear  to be interacting  
with something  on the ground .Language RGB TIRReasoning
RAGTrackFigure 6. Visualization of attention maps.
curacy. As evidenced by Fig. 5 (c), settingn= 4achieves
69.5% MSR. Whenn= 2, the model lacks sufficient histor-
ical context, resulting in a limited reasoning ability. While
expanding ton= 8maintains competitive results, further
increasing ton= 16causes performance degradation. This
decline is likely due to the introduction of redundant or less
relevant features. Thus,n= 4strikes the trade-off between
context richness and feature quality.
Effect of the Retrieval Count in CRM.As shown in
Fig. 5 (d), our method achieves outstanding results with
k= 2. Smaller counts compromise performance due to
limited contextual information, while larger counts perform
worse as excessive features introduce noise. By retrieving
two relevant features, our method ensures sufficient contex-
tual guidance while achieving effective noise suppression.
4.5. Visualization Analysis
Fig. 6 visualizes attention maps across the search region,
illustrating how our RAGTrack improves target localiza-
tion. The baseline model shows a noticeable discrepancy
between the attentive region and the actual target location.
In contrast, our proposed method eliminates this misalign-
ment, enhancing the tracking accuracy.
5. Conclusion
In this work, we propose a novel language-aware frame-
work called RAGTrack for RGBT tracking. To estab-
lish effective feature representation, we introduce a Multi-
modal Transformer Encoder (MTE) that aligns visual and
linguistic features through unified modeling. To reduce
search redundancies and modality gaps, we design an Adap-
tive Token Fusion (ATF), which dynamically selects target-
relevant tokens and enables adaptive channel exchange.
Furthermore, we develop a Context-aware Reasoning Mod-
ule (CRM) that maintains a dynamic knowledge base and
performs context-aware linguistic reasoning to enhance
temporal modeling. Extensive experiments on four track-
ing benchmarks validate the effectiveness of our method.

Acknowledgements
This work was supported in part by the Na-
tional Natural Science Foundation of China
(No.62441231, 62576069), Dalian Science and
Technology Innovation Fund (No.2023JJ11CG001)
and Natural Science Foundation of Liaoning
Province (No.2025-MS-025, 2023JH26/10200016).
References
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hin-
ton. Layer normalization.arXiv preprint arXiv:1607.06450,
2016. 4
[2] Bing Cao, Junliang Guo, Pengfei Zhu, and Qinghua Hu. Bi-
directional adapter for multimodal tracking. InAAAI, pages
927–935, 2024. 2, 6
[3] Sixian Chan, Zedong Li, Wenhao Li, Shijian Lu, Chun-
hua Shen, and Xiaoqin Zhang. Smstracker: Tri-path score
mask sigma fusion for multi-modal tracking. InICCV, pages
4766–4775, 2025. 6, 7
[4] Xin Chen, Ben Kang, Wanting Geng, Jiawen Zhu, Yi Liu,
Dong Wang, and Huchuan Lu. Sutrack: Towards simple and
unified single object tracking. InAAAI, pages 2239–2247,
2025. 6
[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at
scale. InICLR, 2021. 4
[6] Qi Feng, Vitaly Ablavsky, Qinxun Bai, and Stan Sclaroff.
Siamese natural language tracker: Tracking by natural lan-
guage descriptions with siamese trackers. InCVPR, pages
5851–5860, 2021. 1, 2
[7] Xiaokun Feng, Shiyu Hu, Xuchen Li, Dailing Zhang, Meiqi
Wu, Jing Zhang, Xiaotang Chen, and Kaiqi Huang. Atctrack:
Aligning target-context cues with dynamic target states for
robust vision-language tracking. InICCV, pages 19850–
19861, 2025. 2
[8] Xiaokun Feng, Dailing Zhang, Shiyu Hu, Xuchen Li, Meiqi
Wu, Jing Zhang, Xiaotang Chen, and Kaiqi Huang. CSTrack:
Enhancing RGB-X tracking via compact spatiotemporal fea-
tures. InICML, pages 17030–17048, 2025. 2
[9] Xiaokun Feng, Dailing Zhang, Shiyu Hu, Xuchen Li, Meiqi
Wu, Jing Zhang, Xiaotang Chen, and Kaiqi Huang. Enhanc-
ing vision-language tracking by effectively converting tex-
tual cues into visual cues. InICASSP, pages 1–5, 2025. 2
[10] Fang Gao, Wenjie Wu, Yan Jin, Jingfeng Tang, Hanbo
Zheng, Shengheng Ma, and Jun Yu. Tvtracker: Target-
adaptive text-guided visual fusion for multi-modal rgb-t
tracking.IEEE IoTJ, 12(13):24819–24833, 2025. 1, 6, 7
[11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025. 3[12] Mingzhe Guo, Zhipeng Zhang, Liping Jing, Haibin Ling, and
Heng Fan. Divert more attention to vision-language object
tracking.IEEE TPAMI, 46(12):8600–8618, 2024. 3
[13] Lingyi Hong, Shilin Yan, Renrui Zhang, Wanyun Li, Xinyu
Zhou, Pinxue Guo, Kaixun Jiang, Yiting Chen, Jinglun Li,
Zhaoyu Chen, et al. OneTracker: Unifying visual object
tracking with foundation models and efficient tuning. In
CVPR, pages 19079–19091, 2024. 2, 6
[14] Xiaojun Hou, Jiazheng Xing, Yijie Qian, Yaowei Guo, Shuo
Xin, Junhao Chen, Kai Tang, Mengmeng Wang, Zhengkai
Jiang, Liang Liu, et al. SDSTrack: Self-distillation symmet-
ric adapter learning for multi-modal visual object tracking.
InCVPR, pages 26551–26561, 2024. 2, 6
[15] Xiantao Hu, Ying Tai, Xu Zhao, Chen Zhao, Zhenyu Zhang,
Jun Li, Bineng Zhong, and Jian Yang. Exploiting multimodal
spatial-temporal patterns for video object tracking. InAAAI,
pages 3581–3589, 2025. 2, 6, 7
[16] Yufan Hu, Zekai Shao, Bin Fan, and Hongmin Liu. Dual-
level modality de-biasing for RGBT tracking.IEEE TIP, 34:
2667–2679, 2025. 6
[17] Tianrui Hui, Zizheng Xun, Fengguang Peng, Junshi Huang,
Xiaoming Wei, Xiaolin Wei, Jiao Dai, Jizhong Han, and Si
Liu. Bridging search region interaction with template for
RGBT tracking. InCVPR, pages 13630–13639, 2023. 2, 6,
7
[18] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal co-
variate shift. InICML, pages 448–456, 2015. 5
[19] Zong Ke, Yuqing Cao, Zhenrui Chen, Yuchen Yin, Shouchao
He, and Yu Cheng. Early warning of cryptocurrency reversal
risks via multi-source data.FRL, page 107890, 2025. 14
[20] Simiao Lai, Chang Liu, Jiawen Zhu, Ben Kang, Yang Liu,
Dong Wang, and Huchuan Lu. MambaVT: Spatio-temporal
contextual modeling for robust RGBT tracking.IEEE
TCSVT, 35(9):9312–9323, 2025. 6
[21] Hei Law and Jia Deng. Cornernet: Detecting objects as
paired keypoints. InECCV, pages 734–750, 2018. 5
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks. InNeurIPS, pages 9459–9474, 2020. 3
[23] Bo Li, Fengguang Peng, Tianrui Hui, Xiaoming Wei, Xi-
aolin Wei, Lijun Zhang, Hang Shi, and Si Liu. Rgb-t
tracking with template-bridged search interaction and target-
preserved template updating.IEEE TPAMI, 47(1):634–649,
2025. 6
[24] Chenglong Li, Hui Cheng, Shiyi Hu, Xiaobai Liu, Jin Tang,
and Liang Lin. Learning collaborative sparse representa-
tion for grayscale-thermal tracking.IEEE TIP, 25(12):5743–
5756, 2016. 5
[25] Chenglong Li, Nan Zhao, Yijuan Lu, Chengli Zhu, and
Jin Tang. Weighted sparse representation regularized graph
learning for RGBT object tracking. InACM MM, pages
1856–1864, 2017. 5
[26] Chenglong Li, Xinyan Liang, Yijuan Lu, Nan Zhao, and Jin
Tang. Rgbt object tracking: Benchmark and baseline.PR,
96:106977, 2019. 5

[27] Chenglong Li, Lei Liu, Andong Lu, Qing Ji, and Jin Tang.
Challenge-aware rgbt tracking. InECCV, pages 222–237,
2020. 6
[28] Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin
Luo, Jin Tang, and Dengdi Sun. LasHeR: A large-scale high-
diversity benchmark for RGBT tracking.IEEE TIP, 31:392–
404, 2021. 5
[29] Hao Li, Yuhao Wang, Xiantao Hu, Wenning Hao, Pingping
Zhang, Dong Wang, and Huchuan Lu. Cadtrack: Learning
contextual aggregation with deformable alignment for robust
rgbt tracking.arXiv preprint arXiv:2511.17967, 2025. 1, 14
[30] Xin Li, Yuqing Huang, Zhenyu He, Yaowei Wang, Huchuan
Lu, and Ming-Hsuan Yang. Citetracker: Correlating image
and text for visual tracking. InICCV, pages 9974–9983,
2023. 1, 2
[31] Xuchen Li, Xiaokun Feng, Shiyu Hu, Meiqi Wu, Dailing
Zhang, Jing Zhang, and Kaiqi Huang. Dtllm-vlt: Diverse
text generation for visual language tracking based on llm. In
CVPRW, pages 7283–7292, 2024. 3
[32] Xiaohai Li, Bineng Zhong, Qihua Liang, Zhiyi Mo, Jian
Nong, and Shuxiang Song. Dynamic updates for language
adaptation in visual-language tracking. InCVPR, pages
19165–19174, 2025. 3, 6
[33] Zhenyang Li, Ran Tao, Efstratios Gavves, Cees GM Snoek,
and Arnold WM Smeulders. Tracking by natural language
specification. InCVPR, pages 6495–6503, 2017. 2
[34] Bingxi Liu, Calvin Chen, Junhao Li, Guyang Yu, Haoqian
Song, Xuchen Liu, Jinqiang Cui, and Hong Zhang. Track-
ingmim: Efficient mamba-in-mamba serialization for real-
time uav object tracking.arXiv preprint arXiv:2507.01535,
2025. 3
[35] Chang Liu, Xiao-Fan Chen, Chun-Juan Bo, and Dong Wang.
Long-term visual tracking: review and experimental compar-
ison.MIR, 19(6):512–530, 2022.
[36] Chang Liu, Yongsheng Yuan, Xin Chen, Huchuan Lu, and
Dong Wang. Spatial-temporal initialization dilemma: to-
wards realistic visual tracking.VI, 2(1):35, 2024. 1
[37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. InICLR, 2019. 6
[38] Andong Lu, Chenglong Li, Yuqing Yan, Jin Tang, and Bin
Luo. Rgbt tracking via multi-adapter network with hierar-
chical divergence loss.IEEE TIP, 30:5613–5625, 2021. 2
[39] Andong Lu, Jiacong Zhao, Chenglong Li, Yun Xiao, and
Bin Luo. Breaking modality gap in RGBT tracking: Cou-
pled knowledge distillation. InACM MM, pages 9291–9300,
2024. 1, 6
[40] Andong Lu, Chenglong Li, Jiacong Zhao, Jin Tang, and
Bin Luo. Modality-missing rgbt tracking: Invertible prompt
learning and high-quality benchmarks.IJCV, 133(5):2599–
2619, 2025.
[41] Andong Lu, Cun Qian, Chenglong Li, Jin Tang, and Liang
Wang. Duality-gated mutual condition network for rgbt
tracking.IEEE TNNLS, 36(3):4118–4131, 2025.
[42] Andong Lu, Wanyu Wang, Chenglong Li, Jin Tang, and Bin
Luo. After: Attention-based fusion router for rgbt tracking.
IEEE TIP, 34:4386–4401, 2025.[43] Andong Lu, Wanyu Wang, Chenglong Li, Jin Tang, and Bin
Luo. RGBT tracking via all-layer multimodal interactions
with progressive fusion Mamba. InAAAI, pages 5793–5801,
2025. 2, 6, 7, 14
[44] Binxin Luo, Dongxu Liu, Xianrong Peng, Haorui Zuo, Jian-
lin Zhang, Meihui Li, and Yuxing Wei. Progressive trans-
former with multi-modality adaptation for rgb-t tracking.
IEEE TIM, 74:1–17, 2025. 1, 6
[45] Yinchao Ma, Yuyang Tang, Wenfei Yang, Tianzhu Zhang,
Jinpeng Zhang, and Mengxue Kang. Unifying visual and
vision-language tracking via contrastive learning. InAAAI,
pages 4107–4116, 2024. 3
[46] Vinod Nair and Geoffrey E Hinton. Rectified linear units im-
prove restricted boltzmann machines. InICML, pages 807–
814, 2010. 5
[47] Hyeonseob Nam and Bohyung Han. Learning multi-domain
convolutional neural networks for visual tracking. InCVPR,
pages 4293–4302, 2016. 2
[48] Haolin Qin, Tingfa Xu, Tianhao Li, Zhenxiang Chen, Tao
Feng, and Jianan Li. Must: The first dataset and unified
framework for multispectral uav single object tracking. In
CVPR, pages 16882–16891, 2025. 1
[49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. InICML, pages 8748–8763, 2021. 6
[50] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir
Sadeghian, Ian Reid, and Silvio Savarese. Generalized in-
tersection over union: A metric and a loss for bounding box
regression. InCVPR, pages 658–666, 2019. 5
[51] Yongqi Shan, Yunzhi Zhuge, and Huchuan Lu. Hdvs: semi-
supervised semantic segmentation via heterogeneous dual-
branch voting supervision.VI, 4(1):4, 2026. 14
[52] Zekai Shao, Yufan Hu, Bin Fan, and Hongmin Liu. PURA:
Parameter update-recovery test-time adaption for RGBT
tracking. InCVPR, pages 22089–22098, 2025. 1
[53] Zikai Song, Junqing Yu, Yi-Ping Phoebe Chen, and Wei
Yang. Transformer tracking with cyclic shifting window at-
tention. InCVPR, pages 8791–8800, 2022.
[54] Zikai Song, Run Luo, Junqing Yu, Yi-Ping Phoebe Chen,
and Wei Yang. Compact transformer tracker with correlative
masked modeling. InAAAI, pages 2321–2329, 2023.
[55] Zikai Song, Ying Tang, Run Luo, Lintao Ma, Junqing Yu,
Yi-Ping Phoebe Chen, and Wei Yang. Autogenic language
embedding for coherent point tracking. InACM MM, pages
2021–2030, 2024.
[56] Zikai Song, Run Luo, Lintao Ma, Ying Tang, Yi-Ping Phoebe
Chen, Junqing Yu, and Wei Yang. Temporal coherent object
flow for multi-object tracking. InAAAI, pages 6978–6986,
2025.
[57] Yiming Sun, Fan Yu, Shaoxiang Chen, Yu Zhang, Junwei
Huang, Yang Li, Chenhui Li, and Changbo Wang. Chat-
tracker: Enhancing visual tracking performance via chatting
with multimodal large language model. InNeurIPS, pages
39303–39324, 2024. 3

[58] Yuedong Tan, Jiawei Shao, Eduard Zamfir, Ruanjun Li,
Zhaochong An, Chao Ma, Danda Pani Paudel, Luc Van Gool,
Radu Timofte, and Zongwei Wu. What you have is what you
track: Adaptive and robust multimodal tracking. InICCV,
pages 3455–3465, 2025. 2
[59] Yuedong Tan, Zongwei Wu, Yuqian Fu, Zhuyun Zhou,
Guolei Sun, Eduard Zamfi, Chao Ma, Danda Pani Paudel,
Luc Van Gool, and Radu Timofte. XTrack: Multimodal
training boosts RGB-X video object trackers. InICCV, pages
5734–5744, 2025. 1, 2, 6, 7
[60] Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and
Sihong Xie. MBA-RAG: a bandit approach for adaptive
retrieval-augmented generation through question complex-
ity. InCOLING, pages 3248–3254, 2025. 3
[61] Zhangyong Tang, Tianyang Xu, Xiaojun Wu, Xue-Feng Zhu,
and Josef Kittler. Generative-based fusion mechanism for
multi-modal tracking. InAAAI, pages 5189–5197, 2024. 6
[62] Zhangyong Tang, Tianyang Xu, Xiao-Jun Wu, Xuefeng Zhu,
Chunyang Cheng, Zhenhua Feng, and Josef Kittler. Revisit-
ing rgbt tracking benchmarks from the perspective of modal-
ity validity: A new benchmark, problem, and solution.IEEE
TIP, 34:7235–7249, 2025. 6
[63] Chaoqun Wang, Chunyan Xu, Zhen Cui, Ling Zhou, Tong
Zhang, Xiaoya Zhang, and Jian Yang. Cross-modal pattern-
propagation for rgb-t tracking. InCVPR, pages 7064–7073,
2020. 2, 6
[64] Hongyu Wang, Xiaotao Liu, Yifan Li, Meng Sun, Dian Yuan,
and Jing Liu. Temporal adaptive rgbt tracking with modality
prompt. InAAAI, pages 5436–5444, 2024. 2, 6
[65] Mengmeng Wang, Teli Ma, Shuo Xin, Xiaojun Hou, Ji-
azheng Xing, Guang Dai, Jingdong Wang, and Yong Liu.
Visual object tracking across diverse data modalities: A re-
view.arXiv preprint arXiv:2412.09991, 2024. 2
[66] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui
Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-
yang Lin. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution.arXiv preprint
arXiv:2409.12191, 2024. 6
[67] Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei
Wang, Yonghong Tian, and Feng Wu. Towards more flexible
and accurate object tracking with natural language: Algo-
rithms and benchmark. InCVPR, pages 13763–13773, 2021.
2
[68] Xiao Wang, Liye Jin, Xufeng Lou, Shiao Wang, Lan Chen,
Bo Jiang, and Zhipeng Zhang. Reasoningtrack: Chain-of-
thought reasoning for long-term vision-language tracking.
arXiv preprint arXiv:2508.05221, 2025. 3
[69] Zongwei Wu, Jilai Zheng, Xiangxuan Ren, Florin-Alexandru
Vasluianu, Chao Ma, Danda Pani Paudel, Luc Van Gool, and
Radu Timofte. Single-model and any-modality for video ob-
ject tracking. InCVPR, pages 19156–19166, 2024. 6
[70] Yun Xiao, Mengmeng Yang, Chenglong Li, Lei Liu, and Jin
Tang. Attribute-based progressive fusion network for rgbt
tracking. InAAAI, pages 2831–2838, 2022. 6
[71] Yun Xiao, Jiacong Zhao, Andong Lu, Chenglong Li, Bing
Yin, Yin Lin, and Cong Liu. Cross-modulated attentiontransformer for RGBT tracking. InAAAI, pages 8682–8690,
2025. 6
[72] Botao Ye, Hong Chang, Bingpeng Ma, Shiguang Shan, and
Xilin Chen. Joint feature learning and relation modeling for
tracking: A one-stream framework. InECCV, pages 341–
357, 2022. 4
[73] Chunhui Zhang, Xin Sun, Yiqian Yang, Li Liu, Qiong Liu,
Xi Zhou, and Yanfeng Wang. All in one: Exploring uni-
fied vision-language tracking with multi-modal alignment.
InACM MM, pages 5552–5561, 2023. 2
[74] Huanlong Zhang, Jingchao Wang, Jianwei Zhang, Tianzhu
Zhang, and Bineng Zhong. One-stream vision-language
memory network for object tracking.IEEE TMM, 26:1720–
1730, 2023. 2
[75] Tianlu Zhang, Hongyuan Guo, Qiang Jiao, Qiang Zhang, and
Jungong Han. Efficient rgb-t tracking via cross-modality dis-
tillation. InCVPR, pages 5404–5413, 2023. 6
[76] Tianlu Zhang, Qiang Zhang, Kurt Debattista, and Jungong
Han. Cross-modality distillation for multi-modal tracking.
IEEE TPAMI, 47(7):5847–5865, 2025. 6
[77] Xiaosong Zhang, Yunjie Tian, Lingxi Xie, Wei Huang, Qi
Dai, Qixiang Ye, and Qi Tian. Hivit: A simpler and more
efficient design of hierarchical vision transformer. InICLR,
2023. 3
[78] Jie Zhao, Jingshu Zhang, Dongdong Li, and Dong Wang.
Vision-based anti-uav detection and tracking.IEEE TITS, 23
(12):25323–25334, 2022. 1
[79] Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhenjun Tang,
Rongrong Ji, and Xianxian Li. Leveraging local and global
cues for visual tracking via parallel interaction network.
IEEE TCSVT, 33(4):1671–1683, 2022.
[80] Yaozong Zheng, Bineng Zhong, Qihua Liang, Guorong Li,
Rongrong Ji, and Xianxian Li. Toward unified token learn-
ing for vision-language tracking.IEEE TCSVT, 34(4):2125–
2135, 2023. 2
[81] Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhiyi Mo,
Shengping Zhang, and Xianxian Li. Odtrack: Online dense
temporal token learning for visual tracking. InAAAI, pages
7588–7596, 2024.
[82] Yaozong Zheng, Bineng Zhong, Qihua Liang, Ning Li, and
Shuxiang Song. Decoupled spatio-temporal consistency
learning for self-supervised tracking. InAAAI, pages 10635–
10643, 2025.
[83] Yaozong Zheng, Bineng Zhong, Qihua Liang, Shengping
Zhang, Guorong Li, Xianxian Li, and Rongrong Ji. Towards
universal modal tracking with online dense temporal token
learning.IEEE TPAMI, 47(11):10192–10209, 2025. 1, 6
[84] Li Zhou, Zikun Zhou, Kaige Mao, and Zhenyu He. Joint
visual grounding and tracking with natural language specifi-
cation. InCVPR, pages 23151–23160, 2023. 2
[85] Jiawen Zhu, Simiao Lai, Xin Chen, Dong Wang, and
Huchuan Lu. Visual prompt multi-modal tracking. InCVPR,
pages 9516–9526, 2023. 2, 6
[86] Xue-Feng Zhu, Tianyang Xu, Yifan Pan, Jinjie Gu, Xi Li,
Jiwen Lu, Xiao-Jun Wu, and Josef Kittler. Collaborating
vision, depth, and thermal signals for multi-modal tracking:
Dataset and algorithm. InNeurIPS, 2025. 2

[87] Yabin Zhu, Chenglong Li, Bin Luo, Jin Tang, and Xiao
Wang. Dense feature aggregation and pruning for rgbt track-
ing. InACM MM, pages 465–472, 2019. 2
[88] Zhiruo Zhu, Bineng Zhong, Qihua Liang, Hongtao Yang,
Yaozong Zheng, and Ning Li. Adaptive expert decision for
rgb-t tracking.IEEE TCSVT, 35(10):10330–10338, 2025. 6

RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented
Generation
Supplementary Material
This supplementary material provides additional imple-
mentation details, experimental analyses, and visualiza-
tions for our proposed method. We begin by detailing
the pipeline for generating high-quality textual descriptions
in RGB-Thermal (RGBT) tracking benchmarks (§A). We
then present the architecture of our prediction head (§B)
and elaborate on the evaluation metrics used in our ex-
periments (§C). Next, we outline implementation details
including data augmentation and parameter settings (§D).
Furthermore, we conduct extensive ablation studies to ana-
lyze key components of our method, including fusion posi-
tion selection and robustness to missing text (§E). Finally,
we show additional visualizations that demonstrate the ef-
fectiveness of our dynamic token selection mechanism and
present qualitative results under various challenging scenar-
ios (§F). Collectively, these contents offer deeper insights
into the design choices of RAGTrack, further validating the
overall performance of our framework.
A. Textual Description Generation
We generate high-quality textual descriptions for RGBT
tracking through a two-stage pipeline, ensuring semantic
consistency and minimal hallucinations.
Step 1: Initial Description Generation.We utilize
Multimodal Large Language Models (MLLMs) to automat-
ically generate initial descriptions for the object of tracking
in each frame. The specific prompt used is:
“Describe the object located in the image at
<box>(x, y, x+w, y+h)</box>. Focus on dis-
tinctive visual features, motion patterns, and key
identifiers to distinguish it from background ele-
ments and distractors. Keep the description in a
continuous sentence under 20 words. Avoid men-
tioning bounding boxes or coordinates. Do not
use parentheses for explanations. ”
This approach efficiently produces informative descrip-
tions at scale, overcoming the cost of manual annotation.
Step 2: Description Refinement.The initial descrip-
tions from MLLMs may contain inaccuracies or hallucina-
tions. To ensure quality, we perform a refinement step using
the following prompt:
“Correcting the textual description of the track-
ing object. Ensure the final output is a continu-
ous sentence under 20 words, logically coherent,
does not mention bounding boxes, or coordinatesterms, and does not use parentheses for explana-
tions. Do not introduce new details. Output only
the integrated description without any additional
text. Textual description: [Initial description]”
Finally, human experts review the refined descriptions
to correct any remaining issues. This includes rectifying
hallucinations, fixing grammatical errors, removing mixed-
language content or garbled text, and ensuring descriptions
accurately reflect the visual target.
Annotation Statistics.Using this pipeline, we anno-
tate the entire LasHeR training set, which comprises 979
sequences. This process yields a total of 514,081 textual
descriptions, with one description provided for each frame,
to support model training. For evaluation, we annotate
only the first frame across the test set of LasHeR and all
sequences of the GTOT, RGBT210 and RGBT234 bench-
marks. This results in a collection of 739 high-quality tex-
tual descriptions, which are used to assess the textual rea-
soning capability of trackers during inference.
B. Prediction Head
The enhanced features of search region are fed into the
prediction head to produce tracking results in the form of
bounding boxes[x, y, w, h]. The prediction head gener-
ates three outputs: (1) a target classification score map
I∈[0,1]HF×WFindicating presence probabilities, (2) a
spatial offset mapG∈[0,1]2×H F×WFcompensating for
discretization errors, and (3) a normalized bounding box
size mapJ∈[0,1]2×H F×WFrepresenting target width and
height. Here,H FandW Fdenote the height and width of
the feature map. The final bounding box is constructed at
the position(i∗, j∗)with maximum classification score by
combining the corresponding predictions:
x=i∗+G(0, i∗, j∗), y=j∗+G(1, i∗, j∗),
w=J(0, i∗, j∗), h=J(1, i∗, j∗).(15)
C. Evaluation Metrics
We employ Precision Rate (PR) and Success Rate (SR) as
our primary evaluation metrics. On the LasHeR benchmark,
we additionally use the Normalized Precision Rate (NPR)
to address scale variations. For GTOT and RGBT234, we
instead report Maximum Precision Rate (MPR) and Maxi-
mum Success Rate (MSR) due to modality annotation mis-
alignment. PR quantifies the accuracy of target localization

Table 8. Comparison of different fusion positions.
Fusion Positions MPR↑MSR↑
[1,2,3,4]92.4 67.5
[11,12,13,14]92.8 67.9
[21,22,23,24]93.1 68.7
[6,12,18,24] 93.8 69.5
as the proportion of frames where the predicted center po-
sition lies within a predefined distance threshold from the
ground truth. SR measures the bounding box overlap, com-
puted as the percentage of frames where the Intersection
over Union (IoU) between the predicted and ground truth
boxes surpasses a given threshold. NPR extends PR by nor-
malizing the precision based on target size, providing fair
comparison across scale variations. MPR and MSR address
annotation inconsistencies by reporting the maximum per-
formance across modalities, ensuring equitable evaluation
when RGB and TIR annotations are misaligned.
D. Additional Implementation Details
During training, we apply standard data augmentation tech-
niques [19, 51] to the training samples from each sequence,
including rotation, translation, and grayscale transforma-
tion. During inference, the update threshold for multi-
modal references is set to 0.65, with an update interval of
5 frames. Following common practice [29, 43], the back-
bone achieves a tracking speed of 24.3 FPS on a NVIDIA
V100 GPU, with a computational cost of 62.7G FLOPs.
E. More Ablation Studies
This section presents further ablation studies to examine the
impact of individual components in our method. Detailed
analysis and discussions are provided as follows.
Selection of Fusion Positions in ATF.To evaluate the
impact of fusion locations in Adaptive Token Fusion (ATF),
we conduct an ablation study by applying cross-modal fu-
sion at different layers of the Multi-modal Transformer En-
coder (MTE). As shown in Tab. 8, fusing at Layers 6,
12, 18, and 24 achieves the best performance with 93.8%
MPR and 69.5% MSR on RGBT234. Shallow-layer fu-
sion (1-4) captures low-level features but yields subopti-
mal performance due to limited semantic information. Mid-
layer fusion (11–14) improves semantic understanding but
still lacks comprehensive representation. While deep-layer
fusion (21–24) retains high-level context, it misses fine-
grained spatial details. Our design progressively integrates
features across multiple stages, effectively combining spa-
tial details with semantic abstractions. The results confirm
that cross-layer fusion is essential for robust tracking.
Robustness to Missing Text.To evaluate the robust-
Fraction of Missing Text/%Performance
Impact of Missing Text on PerformanceFigure 7. Comparison with different fractions of missing text.
ness of our method under incomplete language guidance,
we conduct an ablation study by randomly masking the in-
put text during inference. As shown in the Fig. 7, RAG-
Track maintains strong performance even when the first-
frame text is partially or fully absent. The performance re-
mains nearly unchanged with 0% to 60% of text missing.
This demonstrates that our model effectively addresses ab-
sent language cues through its Retrieval-Augmented Gen-
eration (RAG) mechanism and context-aware reasoning.
Even when all text is unavailable, our method still achieves
competitive results of 92.9% MPR and 68.8% MSR on
RGBT234, surpassing leading methods [43]. This high-
lights the capacity of RAGTrack to robustly utilize visual
features when textual input is unavailable.
F. More Visualizations
This section provides additional visual analysis to further
validate the effectiveness of our method through the follow-
ing examples and discussions.
More Discussions of Dynamic Token Selection in
ATF.To better understand the behavior of our dynamic to-
ken selection mechanism in ATF, we provide visualizations
of the selected tokens on the LasHeR benchmark. The visu-
alization shows that the mechanism effectively focuses on
target regions while suppressing background distractions.
As shown in Fig. 8, the retained tokens mainly cover the tar-
get area, while the discarded tokens correspond primarily to
background regions and distractors. This demonstrates that
our attention-based selection identifies semantically impor-
tant regions guided by the textual descriptions. Compared
to processing all tokens equally, our method reduces unnec-
essary token processing while maintaining critical target in-
formation. This selective processing allows the model to
concentrate its reasoning capacity on the most informative
image regions. These results provide clear evidence that our
dynamic token selection addresses search redundancies, en-
abling more efficient and accurate RGBT tracking.

10runoneRGB #001 TIR #001
 RGB #106 TIR #106
 RGB #267 TIR #2671blackteacherRGB #001 TIR #001 RGB #222 TIR #222 RGB #399 TIR #3993thmotoRGB #001 TIR #001 RGB #047 TIR #047 RGB #081 TIR #081AQtruck2northRGB #001 TIR #001 RGB #033 TIR #033 RGB #091 TIR #091
Figure 8. Visualization of dynamic token selection in ATF.
More Qualitative Results.Fig. 9 and Fig. 10 present
comprehensive qualitative comparisons of RAGTrack on
challenging sequences from the RGBT234 benchmark. The
visualization highlights the capacity of our method to main-
tain precise tracking through dynamic language reason-
ing across diverse scenarios. Each sequence illustrates the
tracking results, evolving textual descriptions and corre-
sponding per-frame IoU curves. These results demonstrate
how our framework effectively handles several challenging
situations. Through context-aware reasoning and historical
knowledge retrieval, RAGTrack successfully distinguishes
similar-appearance targets in Fig. 9 (a) and Fig. 10 (a). The
method resolves ambiguous target references in Fig. 9 (b)
by leveraging linguistic guidance to maintain tracking con-
sistency. During occlusion shown in Fig. 9 (c) and Fig. 10
(b), the visual-language unified modeling preserves target
identity despite severe appearance changes. Additionally,
as evidenced in Fig. 10 (c), our method overcomes insuffi-
cient visual cues through adaptive fusion of complementarymulti-modal features. The stable IoU curves across chal-
lenging sequences confirm the robustness of our method in
addressing complex tracking difficulties.

(b) tree5(a) walkingman1Ground Truth RAGTrack XTrack CAFormer GMMT
(c) whitesuv
The figure  on the left wears  dark clothing  
and appears  motionless  against  a backdrop  
of grassy  areas  and scattered  trees .A pedestrian  in dark top and pants  actively  
slipping  on a light jacket , their distinct  attire  
distinguishing  them  from  the white  path and 
surrounding  distractors .A pedestrian  in dark pants  and light jacket , 
distinct  from  the white  cement  path and 
similar  distractors  by their contrasting  attire .
IoU
0 500 1000 1500 2000 Frame
The person  walking,  wearing  dark clothing , 
stands  out against  the bright  road,  trees,  and 
buildings  with their distinct  figure  and 
movement  against  the static  urban  backdrop .A person  in dark attire  carrying  a black  
bag walking  steadily  along  the sidewalk , 
their distinct  clothing  and bag clearly  
differentiating  them  from  the background .The object  is a person  wearing  a dark top, 
walking  near a white  SUV  parked  by the 
roadside .
IoU
0 200 300 400 600 500 Frame 100
The target  is a white  SUV  partially  
obscured  by a white  sedan  in the 
foreground .The target  object  is a white  SUV  with a roof 
rack,  distinguishable  from  other  vehicles  by 
its color,  SUV  body  style, and the roof rack,  
as it moves  along  the road.This white  SUV  is positioned  across  the road 
lanes  at an angle  perpendicular  to the 
surrounding  traffic , differing  from  nearby  
vehicles  that include  sedans  in colors .
IoU
0 200 300 400 600 500 Frame 100
Figure 9. Qualitative results on the RGBT234 benchmark.

(b) walkingman41
(c) walking40Ground Truth RAGTrack XTrack CAFormer GMMT
(a) baketballwaliking
The target  is a woman  with long hair 
tied in a ponytail , partially  obscured  by a 
bus, with a visible  white  collar  area.A pedestrian  in a dynamic  pose with 
crossed  legs and forward  lean,  partially  
obscured  by a person  in pink.The person  wearing  a gray coat and black  
pants  is walking  with purpose,  their dark boots  
and visible  legs stand  out against  the light-
colored  pavement  and muted  autumn  backdrop .
IoU
0 100 200 300 400 Frame 500
Aperson dressed indark clothing is
walking away ,their silhouette contrasting
against thelight ofday, creating asense
ofdepth against thebackdrop oftrees .Aperson walking  steadily  with only 
black  shoes  visible  through  the foliage ,  
distinct  from  the tree-covered  background  
and surrounding  distractors  by footwear .Aperson walking  with black  pants  featuring  
white  stripes  and jacket  draped  over the arm, 
distinct  from  the green  foliage  and 
surrounding  distractors  by clothing  pattern .
IoU
0 50 100 150 200 Frame
A cyclist  in dark clothing  rides  across  the 
street , discernible  by their bicycle's  front  
wheel  in motion  and the distinct  shape  of 
the bike against  the pedestrian  backdrop .The target  is a cyclist  dressed  in dark 
clothing,  partially  obscured  by surrounding  
pedestrians,  with only the upper  body  visible  as 
they navigate  through  the crowd  on a bicycle .The target  is a bicycle  with a rider  in dark 
clothing,  and the bicycle  is partially  obscured  
by a person  wearing  a white  coat in the scene .
IoU
0 100 200 250 350 150 300 Frame 50
Figure 10. More qualitative results on the RGBT234 benchmark.