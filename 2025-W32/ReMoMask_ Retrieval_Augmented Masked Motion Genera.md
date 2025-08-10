# ReMoMask: Retrieval-Augmented Masked Motion Generation

**Authors**: Zhengdao Li, Siheng Wang, Zeyu Zhang, Hao Tang

**Published**: 2025-08-04 16:56:35

**PDF URL**: [http://arxiv.org/pdf/2508.02605v1](http://arxiv.org/pdf/2508.02605v1)

## Abstract
Text-to-Motion (T2M) generation aims to synthesize realistic and semantically
aligned human motion sequences from natural language descriptions. However,
current approaches face dual challenges: Generative models (e.g., diffusion
models) suffer from limited diversity, error accumulation, and physical
implausibility, while Retrieval-Augmented Generation (RAG) methods exhibit
diffusion inertia, partial-mode collapse, and asynchronous artifacts. To
address these limitations, we propose ReMoMask, a unified framework integrating
three key innovations: 1) A Bidirectional Momentum Text-Motion Model decouples
negative sample scale from batch size via momentum queues, substantially
improving cross-modal retrieval precision; 2) A Semantic Spatio-temporal
Attention mechanism enforces biomechanical constraints during part-level fusion
to eliminate asynchronous artifacts; 3) RAG-Classier-Free Guidance incorporates
minor unconditional generation to enhance generalization. Built upon MoMask's
RVQ-VAE, ReMoMask efficiently generates temporally coherent motions in minimal
steps. Extensive experiments on standard benchmarks demonstrate the
state-of-the-art performance of ReMoMask, achieving a 3.88% and 10.97%
improvement in FID scores on HumanML3D and KIT-ML, respectively, compared to
the previous SOTA method RAG-T2M. Code:
https://github.com/AIGeeksGroup/ReMoMask. Website:
https://aigeeksgroup.github.io/ReMoMask.

## Full Text


<!-- PDF content starts -->

ReMoMask: Retrieval-Augmented Masked Motion Generation
Zhengdao Li1∗Siheng Wang2∗Zeyu Zhang1∗†Hao Tang1‡
1Peking University2Jiangsu University
∗Equal contribution.†Project lead.‡Corresponding author: bjdxtanghao@gmail.com.
Abstract
Text-to-Motion (T2M) generation aims to synthesize realis-
tic and semantically aligned human motion sequences from
natural language descriptions. However, current approaches
face dual challenges: Generative models (e.g., diffusion mod-
els) suffer from limited diversity, error accumulation, and
physical implausibility, while Retrieval-Augmented Gener-
ation (RAG) methods exhibit diffusion inertia, partial-mode
collapse, and asynchronous artifacts. To address these lim-
itations, we propose ReMoMask, a unified framework inte-
grating three key innovations: 1) A Bidirectional Momen-
tum Text-Motion Model decouples negative sample scale
from batch size via momentum queues, substantially im-
proving cross-modal retrieval precision; 2) A Semantic Spa-
tiotemporal Attention mechanism enforces biomechanical
constraints during part-level fusion to eliminate asynchronous
artifacts; 3) RAG-Classier-Free Guidance incorporates mi-
nor unconditional generation to enhance generalization. Built
upon MoMask’s RVQ-V AE, ReMoMask efficiently gener-
ates temporally coherent motions in minimal steps. Ex-
tensive experiments on standard benchmarks demonstrate
the state-of-the-art performance of ReMoMask, achieving
a 3.88% and 10.97% improvement in FID scores on Hu-
manML3D and KIT-ML, respectively, compared to the pre-
vious SOTA method RAG-T2M. Code: https://github.com/
AIGeeksGroup/ReMoMask. Website: https://aigeeksgroup.
github.io/ReMoMask.
Introduction
Human motion generation has attracted growing attention
due to its broad applicability in domains such as gam-
ing, film production (Zhang et al. 2024e), virtual real-
ity, and robotics. Recent advancements aim to synthesize
diverse and realistic motions to reduce manual anima-
tion costs and enhance content creation efficiency. Among
these efforts, text-to-motion (T2M) generation (Zhang et al.
2024d,c, 2025, 2024b) has emerged as a particularly intu-
itive paradigm, where the objective is to generate a sequence
of human joint positions based on a textual description of the
motion.
Previous research on text-to-motion generation could be
categorized into two directions. The first is conventional
t2m models , which have explored various generative models
such as generative adversarial networks (GANS) (Goodfel-
low et al. 2014), variational autoencoders (V AEs)(Kingma
Part-Level Motion feature
Momentum Part -Level Motion featureText feature Momentum Text feature Motion feature
Base -layer RVQ tokenQ
K V Cross Attention
Masked code Prompt Embedding(b) RAG t2m models
(c) ReMoMask  (Ours)Generative model
TextContrastive 
Loss
ConditionPromptRetrieved 
motionRetrieved
textconcat
Q
K VQ
K VQ
K VQ
K V
Generative model
ConditionPromptRetrieved
textQ
K VQ
K V
TextEMA EMAMomentum
Contrastive LossCoarse 
Motion
Part-level 
Motion
FramesJointsFrames
2D Token 2D Motion Decoder1D Motion Decoder
Q
K VQ
K V(a) Conventional t2m models
1D Motion DecoderGenerative model
Q
K VQ
K VQ
K VQ
K VCondition
Promptconcat
1D Token Frames
“A person is 
walking in a circle ”
1D Token
Retrieved 
motion
Figure 1: Comparison between t2m models. (a) The con-
ventional t2m models. (b) The Existing RAG-t2m models.
(c) The framework of our proposed ReMoMask.
and Welling 2014), diffusion models (Ho, Jain, and Abbeel
2020), motion language models (Zhang et al. 2023a), or gen-
erative masked models (Li et al. 2024). Among them, gen-
erative mask models such as MoMask and MMM use a mo-
tion quantizer to transform a motion seuqnce into discrete
tokens and train the model through randomly masked to-
ken prediction, resulting in high-fidelity motion synthesis.
The second, termed RAG-t2m models , leverages retrieved
text and motion knowledge from an external database to
guide the motion generation, complementing the generative
model, which performs well in handling uncommon text in-
puts. ReMoDiffuse and ReMoGPT represent two represen-
tative approaches: the former relies on the text-to-text sim-
ilarity between captions using CLIP for the retrieval, while
the latter adopts a cross-modal retriever for improved motion
alignment.arXiv:2508.02605v1  [cs.CV]  4 Aug 2025

As shown in Figure 1, although conventional t2m models
provide precise motion synthesis and RAG-t2m models en-
hance generation versatility, these approaches still face two
key challenges. First, the motion retriever is trained using
mini-batch, which suffers from a limited number of nega-
tive samples, thus limiting the learning of robust represen-
tations. Second, concatenating the text condition with a 1D
motion token is insufficient for modeling the relationships
among the text condition, motion spatio-temporal informa-
tion, and retrieved knowledge. These limitations highlight
the need for a new retriever training paradigm that supports
larger negative samples, as well as a more powerful infor-
mation fusion mechanism.
To address the first challenge, we propose a bidirectional
momentum text-motion modeling (BMM) algorithm, which
provides a mechanism of using two momentum encoders
and maintaining dual queues that hold text and motion neg-
ative samples, respectively. At each step, the negative sam-
ples encoded by momentum encoders of the current mini-
batch are enqueued, while the oldest are dequeued. This de-
sign decouples the negative pool size from the mini-batch
size, allowing a larger negative set for contrastive learning.
Furthermore, the two momentum encoders are updated via
an exponential moving average of their online counterparts,
ensuring temporal consistency across negatives.
Moreover, to address the second challenge, we introduce
a semantics spatial-temporal attention (SSTA) mechanism.
Unlike previous motion VQ quantizations that produce a 1D
token map and neglect spatial relationships between individ-
ual joints, SSTA tokenizes the motion into a 2D token map
via a 2D RVQ-V AE, which not only captures the temporal
dynamics but also aggregates the spatial information. During
the later generation, the 2D token map is flattened and fused
with text embedding, retrieved text features, and retrieved
motion features by redefining the Q, K, V matrix in the trans-
former layer. Compared to simply concatenating conditions,
our powerful information fusion mechanism enables com-
prehensive alignment across text guidance, retrieved knowl-
edge, motion temporal dynamics, and even motion spatial
structure, facilitating precision and generalization simulta-
neously.
Together, these components constitute ReMoMask, an
early retrieval augmented text-to-motion masked model,
which outperforms prior text-to-motion approaches on
HumanML-3D and KIT-ML benchmarks.
The contributions of our paper can be summarized as fol-
lows:
• We propose ReMoMask, an innovative RAG-t2m
masked model, equipped with a powerful information fu-
sion mechanism, SSTA, which enables effective fusion
of conditions with both temporal dynamics and spatial
structure of motion.
• To enlarge the negative sample pool in text-motion con-
trastive learning, we proposed a bidirectional momentum
text-motion modeling algorithm (BMM), which decou-
ples the number of negative samples from the mini-batch
size and achieves state-of-the-art performance on text-
motion retrieval.• ReMoMask generates motion sequences with better gen-
eralization and precision than MoMask (Guo et al. 2024),
achieving a 3.14% improvement in MM Dist on Hu-
manML3D and 32.35% in FID on KIT-ML.
Related Work
Text-to-Motion Generation
In the field of text-driven 3D human motion generation, nu-
merous research achievements have been made. Initially,
Text2Motion pioneered the establishment of the mapping
between text and motion through adversarial learning. Sub-
sequently, TM2T (Guo et al. 2022c) first introduced vec-
tor quantization (VQ), and T2M-GPT (Zhang et al. 2023b)
employed autoregressive transformers for semantic control.
However, T2M-GPT suffered from error accumulation dur-
ing unidirectional decoding. MoMask (Guo et al. 2024) pro-
posed a hierarchical residual quantization framework, de-
composing the motion into base tokens and residual to-
kens, and combined with bidirectional masked transformers
for parallel decoding, achieving remarkable results on the
HumanML3D dataset. Regarding masked modeling, Mo-
tionCLIP (Tevet et al. 2022) achieved unsupervised cross-
modal alignment using CLIP but was limited by continuous
representations.The diffusion models have significantly ad-
vanced this field. MotionGPT (Biao et al. 2023) discretizes
motion and leverages autoregressive transformers to unify
generation tasks, mitigating error accumulation. The de-
noising diffusion models (DDPMs) proposed by Song et
al. (Song and Ermon 2020) enable high-quality parallel gen-
eration via non-autoregressive paradigms. Building on lan-
guage model pretraining techniques by Radford et al. (Rad-
ford et al. 2019), researchers further enhance text semantic
control. Current methods integrate discrete representations
with diffusion frameworks to balance efficiency and gener-
ation quality. Moreover, for fine-grained part-level control,
ParCo introduced a breakthrough approach, it discretizes
whole-body motion into six part motions (limbs, backbone,
root) using lightweight VQ-V AEs to establish part priors.
Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) has become a pow-
erful approach for enhancing large language models (LLMs)
by incorporating external knowledge retrieved during infer-
ence (Guo et al. 2025; Qian et al. 2024; Gao et al. 2023). Ini-
tially developed for natural language processing tasks, RAG
helps models generate factually grounded, contextually rele-
vant, and domain-specific responses, addressing common is-
sues such as hallucinations, outdated knowledge, and limited
expertise in closed models. A typical RAG system consists
of three key components: indexing, retrieval, and generation.
Data is first encoded and stored in a vector database; at in-
ference, the most relevant information is retrieved based on
the input query and used to guide generation.
The development of RAG builds on the evolution of in-
formation retrieval (IR) methods. Early IR systems relied
on sparse vector representations, with techniques like TF-
IDF (Sparck Jones 1972) and BM25 (Robertson et al. 1995)
ranking documents based on term frequency and inverse

document frequency. These methods, however, struggled to
capture semantic similarity due to their reliance on exact
term matches. With the rise of deep learning, neural IR mod-
els began representing queries and documents as dense vec-
tors, typically using a bi-encoder (Karpukhin et al. 2020;
Izacard et al. 2021) model or cross-encoder (Nogueira and
Cho 2019) model. These representations enabled more ef-
fective semantic matching through similarity computations,
laying the foundation for modern retrieval-based generation.
Beyond text, RAG has been extended to multimodal do-
mains such as image (Qi et al. 2025), video (Ren et al. 2025),
and motion generation (Zhang et al. 2023c; Kalakonda, Ma-
heshwari, and Sarvadevabhatla 2024; Yu, Tanaka, and Fuji-
wara 2025), where external visual or motion references are
retrieved to guide the generation process. These advances
highlight the flexibility and broad applicability of the RAG
framework across diverse tasks and modalities.
The Proposed Method
Framework Overview
Figure 2 shows the overall architecture of ReMoMask. To
ensure the quality of the motion in both temporal dynamics
and spatial structure, we quantize a motion sequence into
a 2D spatial-temporal map via a 2D RVQ-V AE encoder.
During generation, starting from an all masked 2D token
map, ReMoMask first retrieves text and motion features us-
ing a Part-Level Bidirectional Momentum Text-Motion Re-
triever , which is trained with the Bidirectional Momentum
text-motion modeling (BMM) algorithm to enable a large
negative samples pool. These retrieved features are then
fed into the Masked Transformer and fused by Semantics
Spatial-Temporal Attention (SSTA), providing strong se-
mantic alignment and guidance for reconstructing the core
motion structure. Finally, a Residual Transformer refines
motion details, and the latent motion vector is decoded
through a 2D RVQ-V AE decoder.
Bidirectional Momentum Text-Motion Modeling
As shown in Figure 2(a), we adopt a dual momentum en-
coder architecture equipped with corresponding memory
queues. Let fmandftdenote the motion and text encoders,
parameterized by θmandθt, respectively. To ensure tempo-
ral consistency across negative samples, we introduce two
momentum counterparts ˆfmandˆft, with parameters ˆθmand
ˆθt, which are updated using exponential moving averages
with momentum coefficient ˜m:
ˆθm= ˜m·ˆθm+ (1−˜m)·θm, (1)
ˆθt= ˜m·ˆθt+ (1−˜m)·θt, (2)
To decouple the size of negative samples from the
mini-batch size, we employ two negative queues: Qm=
{km
j}Nq
j=1andQt={kt
j}Nq
j=1, where each km
jandkt
jis a
momentum feature vector extracted via:
km
j=ˆfm(mj), kt
j=ˆft(tj). (3)
Given a training mini-batch B={(mi, ti)}Nb
i=1⊆ D
(Nb=|B| ≪ Nq), we compute the momentum featuresfor each sample pair and enqueue them into their respec-
tive queues, while simultaneously dequeuing the oldest Nb
entries to maintain a fixed queue size. For contrastive learn-
ing, each motion sample takes its paired text as the positive
example, and all entries in Qtare treated as negatives. The
motion-to-text contrastive loss is formulated as:
LM2T=−NbX
i=1logexp(qm
i·kt
i/τ)
exp(qm
i·kt
i/τ)+neg(qm
i,Qt, τ),(4)
where qm
i=fm(mi)andkt
i=ˆft(ti). The negative term is
defined by:
neg(qm
i, Qt, τ) =X
kt
j∈Qtexp(qm
i·kt
j/τ). (5)
Analogously, we reverse the roles of motion and text to com-
pute the text-to-motion contrastive loss:
LT2M=−NbX
i=1logexp(qt
i·km
i/τ)
exp(qt
i·km
i/τ)+neg(qt
i, Qm, τ),(6)
where qt
i=ft(ti)andkm
i=ˆfm(mi). The final bidirec-
tional momentum contrastive loss is the sum of both direc-
tions:
LBMM=LM2T+LT2M. (7)
Algorithm 1: Bidirectional Momentum Text-Motion Model-
ing
Require: Training set D
Require: Online encoders ft,fmwith parameters θt,θm
Require: Momentum encoders ˆft,ˆfmwith ˆθt,ˆθm
Require: Queues: Qt,Qm
1:Initialize: ˆθt←θt,ˆθm←θm
2:while not converged do
3: Sample a mini-batch B={(mi, ti)}Nb
i=1⊆ D
4: // Feature encoding
5: foreach (mi, ti)∈ B do
6: qt
i←ft(ti),qm
i←fm(mi)
7: kt
i←ˆft(ti),km
i←ˆfm(mi)
8: end for
9: // Contrastive loss
10: Compute LM2TandLT2Musing Eq.(4) and Eq.(6)
11: LBMM← L M2T+LT2M
12: // Optimization
13: Update θt, θmby minimizing LBMM
14: // Momentum update
15: ˆθt←˜m·ˆθt+ (1−˜m)·θt
16: ˆθm←˜m·ˆθm+ (1−˜m)·θm
17: // Queue update
18: Enqueue {kt
i}Nb
i=1intoQt,{km
i}Nb
i=1intoQm
19: Dequeue earliest Nbentries from each queue
20:end while
Semantics Spatial-Temporal Attention
As demonstrated in Figure 2(c), for a 2D token map gener-
ated by a 2D RVQ-V AE (discussed in later sections), we first

(a) Bidirectional Momentum Text -Motion Modeling (BMM) (b) ReMoMask  Pipeline (c) Semantics Spatial -Temporal Attention (SSTA)1D Multi -Head Attention 
FramesJointsFramesJointsFlaten 1D
Reshape 2DQ K V
A person is jogging 
back and force 
Part-level Motion Paired Text𝐿𝑀2𝑇 𝐿𝑇2𝑀𝐿𝐵𝑀𝑀
Motion Encoder Text EncoderEMA EMAMomentum
Motion EncoderMomentum
Text EncoderA person is walking in a circle
SSTA
Part-Level BTM Retrieverquery
STA
2D RVQ -V AE Decoder2D RVQ -V AE Encoder
FramesJointsPart-level Motion feature Momentum Part -level Motion feature Text feature Momentum Text feature Prompt Embedding Base -layer RVQ token Masked code
Figure 2: Overview of ReMoMask . (a) Bidirectional Momentum Contrastive Retrieval (BMM) uses two momentum queues,
enabling a large pool of negative samples for contrastive learning. (b) ReMoMask quantizes a motion sequence into a 2D token
map, capturing not only temporal dynamics but also spatial structure. After that, a Part-Level BMM Retriever retrieves relevant
text and motion features based on the prompt embedding. All these conditions are fused via an SSTA module in a 2D RAG-
Mask-Transformer together with the latent motion representaion. (c) Semantic Spatial-temporal Attention (SSTA) first flattens
the masked 2D token map into a 1D structure, then redefines the Q, K, V matrix utilizing the conditions above, providing
effective semantic alignment between the conditions and the spatial-temporal information of motion
add a 2D position encoding (Wang and Liu 2019). The 2D
position encoding P∈RT×Jis obtained by independently
applying sinusoidal functions along the temporal and spatial
axes. After that, we flatten the 2D token map to a 1D struc-
ture, resulting in latent motion vector z∈R(TJ)×d. Then we
extract the text embedding t∈R1×dusing the text encoder
of Clip (Radford et al. 2021), and get the retrieved text fea-
tureRt∈R1×dand retrieved motion feature Rm∈R1×d
using part-level bmm retriever (as will be explained later).
Here, Jis the number of joints, Tis the number of frames,
anddis the feature dimension. Then we perform the adapted
attention in the flattened spatial-temporal dimension, as:
Assta(Q, K, V ) =softmaxQKT
√
d+P
V, (8)
Q=m, (9)
K=concat (z, t,[Rm;Rt]), (10)
V=concat (z, t, R m). (11)
where [·;·]denotes the concatenation of both terms, Pis
the 2D Position embedding, and Q, K, V are the query, key,
and value matrices. We refer to this attention mechanism as
Semantics Spatial-Temporal Attention (SSTA). A simplified
variant that only concatenates the text embedding twith the
motion vector zand excludes retrieval features, is termed
Spatial-Temporal Attention (STA). As shown in Figure 2(b),both attention mechanisms will be used in the subsequent
motion generation module.
Training: Part-Level BMM Retriever
To model fine-grained motion details, we also implement
a part-level motion encoder inspired by (Yu, Tanaka, and
Fujiwara 2025; Zou et al. 2024). Concretely, we divide the
whole-body motion into six parts and embed each part, re-
spectively. These part-level features are then concatenated
and reprojected into a latent dimension to produce a fine-
grained motion feature, enabling more precise retrieval of
part-specific features. For the retriever, we replacing the mo-
tion encoder fmand its corresponding momentum model
ˆfmwith part-level motion encoder fpland ˆfplin Equa-
tion 4 and Equation 6. The full training objective of Part-
level BMM Retriever is:
Lpl
BMM=Lpl
M2T+Lpl
T2M. (12)
Training: Retrieval-Augmented MoMask
Network Architecture As illustrated in Figure 2(b), Re-
MoMask includes three key components: a 2D RVQ-
V AE encode-decoder quantizes motion into discrete 2D to-
kens and reconstructs motion from them. a 2D retrieval-
augmented masked transformer generates base-layer tokens
conditioned on text and retrieved features. a 2D residual
transformer refines the remaining token layers to capture
fine-grained details.

Table 1: Performance comparison on HumanML3D dataset.
Method R-Precision ↑ FID↓MM Dist ↓Diversity → MultiModality ↑
Top1 Top2 Top3
Real Motions 0.511 0.703 0.797 0.002 2.974 9.503 –
MoCoGAN (Tulyakov et al. 2018) 0.037 0.072 0.106 94.41 9.643 0.462 0.019
Dance2Music (Lee et al. 2019) 0.033 0.065 0.097 66.98 8.116 0.725 0.043
Language2Pose (Ahuja and Morency 2019) 0.246 0.387 0.486 11.02 5.296 7.676 –
Text2Gesture (Bhattacharya et al. 2021) 0.165 0.267 0.345 7.664 6.030 6.409 –
T2M (Guo et al. 2022a) 0.457 0.639 0.740 1.067 3.340 9.188 2.090
T2M-GPT (Zhang et al. 2023b) 0.491 0.680 0.775 0.116 3.118 9.761 1.856
FineMoGen (zhang et al. 2023) 0.504 0.690 0.784 0.151 2.998 9.263 2.696
MDM (Tevet et al. 2023) – – 0.611 0.544 5.566 9.559 2.799
MotionDiffuse (Zhang et al. 2024a) 0.491 0.681 0.782 0.630 3.113 9.410 1.553
MoMask (Guo et al. 2024) 0.521 0.713 0.807 0.045 2.958 – 1.241
MoGenTS (Yuan et al. 2024) 0.529 0.719 0.812 0.033 2.867 9.570 –
ReMoDiffuse (Zhang et al. 2023d) 0.510 0.698 0.795 0.103 2.974 9.018 1.795
ReMoGPT (Yu, Tanaka, and Fujiwara 2024) 0.501 0.688 0.792 0.205 2.929 9.763 2.816
RMD (Liao et al. 2024) 0.524 0.715 0.811 0.111 2.879 9.527 2.604
MoRAG-Diffuse (Sai Shashank Kalakonda 2024) 0.511 0.699 0.792 0.270 2.950 9.536 2.773
ReMoMask (Ours) 0.531 0.722 0.813 0.099 2.865 9.535 2.823
2D Residual VQ-V AE Given a motion m∈RT×J, we
first extract 2D latent features ˆy∈RT×Jusing a 2D convo-
lutional encoder E2d. We then apply residual vector quanti-
zation (RVQ) (Guo et al. 2024) with L+1levels:
yl=Ql(rl),rl+1=rl−yl, (13)
starting from r0=ˆy, where Ql(·)denotes the vector quan-
tization operation at level l, mapping each latent vector to
its nearest code in a learnable codebook. The final summed
quantized representationPL
l=0ylis then fed into a 2D con-
volutional decoder D2dto reconstruct motion, resulting in
ˆ m∈RT×J.
We train the model by minimizing reconstruction and em-
bedding losses using the straight-through gradient estimator,
as
L2D-RVQ =∥m−ˆm∥1+γLX
l=1∥rl−sg[yl]∥2
2, (14)
where sg [·]denotes the stop-gradient operation, and γcon-
trols the strength of the embedding loss.
2D Retrieval-Augmented Masked-Transforemr Under
the hypothesis of hierarchical RVQ-V AE, the base quanti-
zation layer—generated by a masked transformer—captures
the coarse motion structure, while the residual transformer
layers refine fine-grained details. Since the base layer cap-
tures the main semantics of motion, we introduce retrieval-
augmented context to this stage only, aiming to enhance
structural reconstruction without overburdening the refine-
ment stages.
We design our 2D retrieval-augmented masked trans-
former to generate the base-layer motion tokens y0∈
RT×J, conditioned on the text embedding t, retrieved text
feature Rt, and retrieved motion feature Rm, all of which are
fused using SSTA. To train the model, we randomly mask asubset of tokens in y0, replacing them with a [MASK] to-
ken to obtain a corrupted sequence y0
msk. We perform a 2D
Mask strategy (Yuan et al. 2024) by first randomly masking
along the temporal dimension and then randomly masking
along the spatial dimension on those unmasked frames. The
model is then trained to reconstruct the original tokens by
minimizing the negative log-likelihood:
Lrag
mask=X
[MASK]−logp(y0|y0
msk, t, R t, Rm). (15)
We also employ a masking ratio schedule and a BERT-style
remasking strategy, extending the previous method (Chang
et al. 2022, 2023; Devlin et al. 2019; Guo et al. 2024).
2D Residual Transformer The architecture of our 2D
residual transformer mirrors that of the 2D retrieval aug-
mented masked transformer, except that we adopt STA in-
stead of SSTA. During training, we randomly select a quan-
tization layer l∈[1, L]to predict. All tokens from previ-
ous layers y0:l−1are summed to form the latent input, to-
gether with the quantizer layer index land the text condition
t. The model is optimized by minimizing the negative log-
likelihood:
Lres=LX
l=1−logp 
yl|y0:l−1, l, t
, (16)
Inference
In the inference, we start from an empty 2D token map that
all tokens are masked, defined as y∈RT×J. Then the 2D
retrieval-augmented masked transformer repeatedly predicts
the masked tokens as a base quantization layer by N itera-
tions, conditioned on the caption embedding t, the retrieved
motion feature Rm, and the retrieved text features Rt. Once

Table 2: Performance comparison on KIT-ML dataset.
Method R-Precision ↑ FID↓MM Dist ↓Diversity → MultiModality ↑
Top1 Top2 Top3
Real Motions 0.424 0.649 0.779 0.031 2.788 11.08 –
MoCoGAN (Tulyakov et al. 2018) 0.022 0.042 0.063 82.69 10.47 3.091 0.250
Language2Pose (Ahuja and Morency 2019) 0.221 0.373 0.483 6.545 5.147 9.073 –
Dance2Music (Lee et al. 2019) 0.031 0.058 0.086 115.4 10.40 0.241 0.062
Text2Gesture (Bhattacharya et al. 2021) 0.156 0.255 0.338 12.12 6.964 9.334 –
T2M (Guo et al. 2022a) 0.370 0.569 0.693 2.770 3.401 10.91 1.482
MotionDiffuse (Zhang et al. 2024a) 0.417 0.621 0.739 1.954 2.958 11.10 0.730
T2M-GPT (Zhang et al. 2023b) 0.416 0.627 0.745 0.514 3.007 10.92 1.570
MDM (Tevet et al. 2023) – – 0.396 0.497 9.191 10.85 1.907
MoMask (Guo et al. 2024) 0.433 0.656 0.781 0.204 2.779 – 1.131
MoGenTS (Yuan et al. 2024) 0.445 0.671 0.797 0.143 2.711 10.918 –
ReMoDiffuse (Zhang et al. 2023d) 0.427 0.641 0.765 0.155 2.814 10.80 1.239
ReMoMask (Ours) 0.453 0.682 0.805 0.138 2.682 10.83 2.017
the prediction is completed, the 2D residual transformer pro-
gressively predicts the residual tokens of the rest quantiza-
tion layers. As a final stage, all tokens are decoded and pro-
jected back to motion sequences through the 2D RVQ-V AE
decoder.
RAG Classifier Free Guidance We extend the classifier-
free guidance (CFG) to incorporate the text embedding t, re-
trieved text feature Rt, and retrieved motion feature Rmas
conditional inputs. Explicitly, we define the guidance condi-
tion as {t, Rt, Rm}, the final logits are computed as:
logits = (1 + s)·logitscon−s·logitsun, (17)
con={t, Rt, Rm} (18)
where logits is the output of the final linear projection layer
andsis the guidance scale (set to 4).
During training, unconditional sampling is applied with a
probability of 10% to enable guidance-free learning. At in-
ference, R-CFG is applied to the final projection layer prior
to softmax.
Experiment
Dataset and Evaluation Metrics
We evaluate our model on HumanML3D (Guo et al. 2022b)
and KIT-ML (Plappert, Mandery, and Asfour 2016) datasets.
The HumanML3D dataset (Guo et al. 2022b) stands as the
largest available dataset focused solely on 3D body motion
and associated textual descriptions. It consists of 14616 mo-
tion sequences and 44970 text descriptions, and KIT-ML
consists of 3911 motions and 6278 texts. The motion pose is
extracted into the motion feature with dimensions of 263 and
251 for HumanML3D and KIT-ML respectively. Following
previous methods (Guo et al. 2022b), the datasets are aug-
mented by mirroring,and divided into training, testing,and
validation sets with the ratio of 0.8:0.15:0.05.Evaluation Metrics We adapt standard evaluation metrics
to assess various aspects of our experiments: For overall mo-
tion quality, we propose Fr ´echet Inception Distance (FID) to
measure the distributional difference between high-level fea-
tures of generated and real motions. For semantic alignment
between input text and generated motions, we propose R-
Precision and multimodal distance. For diversity of motions
generated from the same text, we propose Multimodality.
Implementation Details
Our framework is trained on four NVIDIA H20 GPUs us-
ing PyTorch with a batch size of 256 and a learning rate of
2×10−4. For motion quantization, we employ a 2D RVQ-
V AE structure following (Yuan et al. 2024). The pose data
is restructured into a joint-based format of size 12×Jand
quantized into 2D latent representations. This is achieved
using two codebooks: (1) a joint VQ codebook containing
256 codes (dimension 1024), and (2) a global VQ code-
book with 256 codes (dimension 1024) to capture holis-
tic motion information.For motion generation, we imple-
ment two transformer architectures:M-trans (Motion Trans-
former): 6 layers, 8 attention heads, and 512 latent dimen-
sions, R-trans (Residual Transformer): 6 layers, 6 attention
heads, and 384 latent dimensions. The 2D Residual Trans-
former structure follows (Guo et al. 2024) with 5 residual
layers.For retrieval enhancement, we design a 2D Retrieval-
augmented Masked Transformer using momentum con-
trastive learning. The motion encoder adopts RemoGPT’s
4-layer Transformer architecture (Yu, Tanaka, and Fujiwara
2024) with 512 latent dimensions. Text embeddings are gen-
erated using DistilBERT (Victor et al. 2019). Key hyperpa-
rameters include: momentum coefficient m= 0.99, temper-
ature τ= 0.07, and dynamic queue size 65536. This module
is trained on eight NVIDIA A800 GPUs with batch size 128
for 200 epochs.

Main Results
Evaluation of Motion Generation we compare our
model with previous text-to-motion works, including comb-
ing RAG method and without RAG method. From the re-
sults reported in Table 1, 2, our method outperforms all
previous methods on both the HumanML3D and the KIT-
ML datasets, which demonstrates the effectiveness of our
method. Crucially, the FID is decreased by 0.093 on Hu-
manML3D compared and is decreased by 0.066 on KIT-ML
compared to MoMask (Guo et al. 2024). Moreover, the R-
precision even significantly surpasses the ground truth.
Evaluation of Retriever As shown in Table 3, in the text-
to-motion retrieval task, BMM achieves state-of-the-art per-
formance with significant improvements across key metrics.
It obtains the highest scores in R1 (13.76), R2 (21.03), R3
(25.63), and R5 (32.40), outperforming PL-TMR with abso-
lute gains ranging from 2.76% to 2.92%. Although its R10
score (43.27) is slightly lower than that of PL-TMR (43.43),
the overall superiority of BMM is evident. Similarly, BMM
demonstrates strong performance in the motion-to-text re-
trieval task, achieving the highest results in R1 (14.80) and
R3 (25.60), with a notable 2.55% absolute improvement in
R1 over PL-TMR. However, it shows limitations in higher-
recall metrics: its R5 (25.75) and R10 (34.61) lag behind PL-
TMR by 2.59% and 4.50%, respectively. Moreover, BMM’s
MedR (25.00) also underperforms, suggesting reduced ef-
fectiveness in retrieving multiple relevant texts per motion.
Ablation Study
This section conducts systematic ablation experiments
to validate the contributions of ReMoMask’s core mod-
ules: Bidirectional Momentum Model (BMM), Semantic
Spatio-Temporal Attention (SSTA), Retrieval-Augmented
Classifier-Free Guidance (RAG-CFG), and local retrieval
mechanism. Results conclusively demonstrate the necessity
of each innovative component.
Core Module Contribution Analysis As shown in Ta-
ble 4, the full model (ReMoMask) achieves comprehensive
SOTA performance on HumanML3D: BMM Module : Re-
moval causes 16.2% Top1 R-Precision drop (0.531 →0.445)
and 50.18% FID degradation (0.411 →0.825), proving its ir-
replaceability in cross-modal alignment. SSTA Module : Re-
placement with feature concatenation leads to 61.2% mul-
timodality collapse (2.823 →1.094) and 6.1% MM Dist in-
crease (2.865 →3.04), highlighting its critical role in mo-
tion diversity. RAG-CFG : Deactivation reduces Top1 R-
Precision by 22.6% (0.531 →0.411), confirming its effi-
cacy in enhancing text-motion consistency. Local Retrieval :
Global retrieval substitution decreases Top3 R-Precision by
9.8% (0.813 →0.733) and diversity by 4.8% (9.535 →9.08),
demonstrating the superiority of local context retrieval.
Bidirectional Momentum Queue Optimization Table 5
reveals the impact of momentum queue design on cross-
modal retrieval: Text→Motion Retrieval : Bidirectional
queues improve R1 by 31.3% (10.48 →13.76) and reduce
MedR by 15.8% (19 →16) compared to no queues. Mo-
tion→Text Retrieval : Unidirectional queues cause catas-Table 3: Performance comparison of text-motion retrieval
tasks on HumanML3D dataset.
Method Params R1 ↑ R2↑ R3↑ R5↑ R10↑MedR ↓
Text-to-motion retrieval
TMR 82M 8.92 12.04 16.33 22.06 33.37 25.00
MotionPatches 152M 10.80 14.98 20.00 26.72 38.02 19.00
PL-TMR 118M 11.00 17.02 22.18 29.48 43.43 14.00
BMM 238M 13.76 21.03 25.63 32.40 43.27 16.00
Motion-to-text retrieval
TMR 82M 9.44 11.84 16.90 22.92 32.21 26.00
MotionPatches 152M 11.25 13.86 19.98 26.86 37.40 20.00
PL-TMR 118M 12.25 14.95 21.45 28.34 39.11 19.00
BMM 238M 14.80 15.63 25.60 25.75 34.61 25.00
trophic failure (R1=0.70), while bidirectional queues boost
R1 by 41.0% (10.50 →14.80), proving symmetric negative
sample queues are indispensable for bidirectional retrieval.
The optimal configuration significantly outperforms base-
lines in both tasks, validating the robustness of our bidirec-
tional momentum design.
Table 4: Ablation study 1 on HumanML3D dataset. We test
ReMoMask’s core modules: BMM, SSTA, RAG-CFG, and
local retrieval mechanism.
Method R-Precision ↑ FID↓MM Dist ↓Diversity → MultiModality ↑
Top1 Top2 Top3
w/o BMM 0.445 0.639 0.751 0.825 3.44 8.80 1.017
w/o SSTA 0.495 0.652 0.789 0.714 3.04 9.39 1.094
w/o RAG-CFG 0.411 0.612 0.741 0.798 3.16 9.12 1.088
w/o Coarse-Level Retrieval 0.402 0.644 0.733 0.722 3.32 9.08 1.044
ReMoMask (Ours) 0.531 0.722 0.813 0.411 2.865 9.535 2.823
Table 5: Ablation study 2 on HumanML3D.We explored the
importance of BMM
text queue motion queue R1↑ R2↑ R3↑ R5↑ R10↑MedR ↓
Text-to-motion retrieval
✗ ✗ 10.48 15.80 20.90 28.59 41.25 19.00
✗ ✓ 12.44 18.98 22.79 29.72 43.02 17.00
✓ ✓ 13.76 21.03 25.63 32.40 43.27 16.00
Motion-to-text retrieval
✗ ✗ 10.50 13.36 18.64 24.98 36.83 20.00
✗ ✓ 0.70 1.02 1.22 1.87 3.18 552.00
✓ ✓ 14.80 15.63 25.60 25.75 34.61 25.00
Conclusion
In this paper, we introduce an innovative retrieval-
augmented masked model, ReMoMask, for text-driven mo-
tion generation. The proposed bidirectional momentum text-
motion relation modeling enlarges the set of negative sam-
ples across modalities, facilitating more effective contrastive
learning for the part-level retriever. Quantizing the motion
sequence into a 2D token map and applying well-designed
cross-attention with textual and retrieved conditions en-
ables a more expressive fusion of conditional semantics and
spatio-temporal motion dynamics. Extensive experiments on
HumanML3D and KIT datasets demonstrate that our model
achieves SOTA performance.

References
Ahuja, C.; and Morency, L.-P. 2019. Language2Pose: Nat-
ural Language Grounded Pose Forecasting. In International
Conference on 3D Vision (3DV) , 719–728.
Bhattacharya, U.; Rewkowski, N.; Banerjee, A.; Guhan,
P.; Bera, A.; and Manocha, D. 2021. Text2Gestures: A
Transformer-Based Network for Generating Emotive Body
Gestures. In IEEE Virtual Reality and 3D User Interfaces
(VR) , 1–10.
Biao, J.; Xin, C.; Wen, L.; Jingyi, Y .; Gang, Y .; and Tao, C.
2023. MotionGPT: Unified Human Motion Generation via
Autoregressive Transformers. In NeurIPS .
Chang, H.; Zhang, H.; Barber, J.; Maschinot, A.; Lezama,
J.; Jiang, L.; Yang, M.-H.; Murphy, K.; Freeman, W. T.; Ru-
binstein, M.; Li, Y .; and Krishnan, D. 2023. Muse: Text-
To-Image Generation via Masked Generative Transformers.
arXiv:2301.00704.
Chang, H.; Zhang, H.; Jiang, L.; Liu, C.; and Freeman, W. T.
2022. MaskGIT: Masked Generative Image Transformer.
arXiv:2202.04200.
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019.
BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding. arXiv:1810.04805.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2(1).
Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.;
Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y .
2014. Generative adversarial nets. In NIPS , 2672–2680.
Guo, C.; Mu, Y .; Javed, M. G.; Wang, S.; and Cheng, L.
2024. MoMask: Generative Masked Modeling of 3D Human
Motions. In CVPR .
Guo, C.; Zou, S.; Zuo, X.; Wang, S.; Ji, W.; Li, X.; and
Cheng, L. 2022a. Generating Diverse and Natural 3D Hu-
man Motions from Text. In IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) , 5152–5161.
Guo, C.; Zou, S.; Zuo, X.; Wang, S.; Ji, W.; Li, X.; and
Cheng, L. 2022b. Generating Diverse and Natural 3D Hu-
man Motions From Text. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR) , 5152–5161.
Guo, C.; Zuo, X.; Wang, S.; and Cheng, L. 2022c. Tm2t:
Stochastic and tokenized modeling for the reciprocal gener-
ation of 3d human motions and texts. In ECCV .
Guo, Z.; Xia, L.; Yu, Y .; Ao, T.; and Huang, C. 2025. Ligh-
tRAG: Simple and Fast Retrieval-Augmented Generation.
arXiv:2410.05779.
Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion
probabilistic models. In NIPS , volume 33, 6840–6851.
Izacard, G.; Caron, M.; Hosseini, L.; Riedel, S.; Bojanowski,
P.; Joulin, A.; and Grave, E. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv preprint
arXiv:2112.09118 .Kalakonda, S. S.; Maheshwari, S.; and Sarvadevabhatla,
R. K. 2024. MoRAG – Multi-Fusion Retrieval Augmented
Generation for Human Motion. arXiv:2409.12140.
Karpukhin, V .; Oguz, B.; Min, S.; Lewis, P. S.; Wu, L.;
Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Pas-
sage Retrieval for Open-Domain Question Answering. In
EMNLP , 6769–6781.
Kingma, D. P.; and Welling, M. 2014. Auto-encoding varia-
tional bayes. In ICLR .
Lee, H.-Y .; Yang, X.; Liu, M.-Y .; Wang, T.-C.; Lu, Y .-
D.; Yang, M.-H.; and Kautz, J. 2019. Dancing to Mu-
sic. In Advances in Neural Information Processing Systems
(NeurIPS) , volume 32.
Li, J.; Wang, J.; Xu, H.; Yang, J.; Xu, M.; Yu, J.; Chen, W.;
et al. 2024. Generative Masked Autoencoders for 3D Human
Motion Synthesis. arXiv preprint arXiv:2403.12096 .
Liao, Z.; Zhang, M.; Wang, W.; Yang, L.; and Komura, T.
2024. RMD: A Simple Baseline for More General Human
Motion Generation via Training-free Retrieval-Augmented
Motion Diffuse. arXiv preprint arXiv:2412.04343 .
Nogueira, R.; and Cho, K. 2019. Passage Re-ranking with
BERT. arXiv preprint arXiv:1901.04085 .
Petrovich, M.; Black, M. J.; and Varol, G. 2023. TMR: Text-
to-Motion Retrieval Using Contrastive 3D Human Motion
Synthesis. arXiv:2305.00976.
Plappert, M.; Mandery, C.; and Asfour, T. 2016. The KIT
Motion-Language Dataset. Big Data , 4(4): 236–252.
Qi, J.; Xu, Z.; Wang, Q.; and Huang, L. 2025. AR-RAG: Au-
toregressive Retrieval Augmentation for Image Generation.
arXiv preprint arXiv:2506.06962 .
Qian, H.; Zhang, P.; Liu, Z.; Mao, K.; and Dou, Z.
2024. Memorag: Moving towards next-gen rag via
memory-inspired knowledge discovery. arXiv preprint
arXiv:2409.05591 , 1.
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.;
Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.;
Krueger, G.; and Sutskever, I. 2021. Learning Transfer-
able Visual Models From Natural Language Supervision.
arXiv:2103.00020.
Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; and
Sutskever, I. 2019. Language Models are Unsupervised
Multitask Learners. OpenAI blog .
Ren, X.; Xu, L.; Xia, L.; Wang, S.; Yin, D.; and Huang,
C. 2025. VideoRAG: Retrieval-Augmented Generation
with Extreme Long-Context Videos. arXiv preprint
arXiv:2502.01549 .
Robertson, S. E.; Walker, S.; Jones, S.; Hancock-Beaulieu,
M. M.; Gatford, M.; et al. 1995. Okapi at TREC-3. Nist
Special Publication Sp , 109: 109.
Sai Shashank Kalakonda, R. K. S., Shubh Maheshwari.
2024. MoRAG – Multi-Fusion Retrieval Augmented Gener-
ation for Human Motion. arXiv preprint arXiv:2409.12140 .
Song, Y .; and Ermon, S. 2020. Denoising Diffusion Proba-
bilistic Models. arXiv:2006.11239 .

Sparck Jones, K. 1972. A statistical interpretation of term
specificity and its application in retrieval. Journal of docu-
mentation , 28(1): 11–21.
Tevet, G.; Gordon, B.; Hertz, A.; Bermano, A. H.; and
Cohen-Or, D. 2022. Motionclip: Exposing human motion
generation to clip space. ECCV .
Tevet, G.; Raab, S.; Gordon, B.; Shafir, Y .; Cohen-Or, D.;
and Bermano, A. H. 2023. Human Motion Diffusion Model.
InInternational Conference on Learning Representations
(ICLR) .
Tulyakov, S.; Liu, M.-Y .; Yang, X.; and Kautz, J. 2018.
MoCoGAN: Decomposing Motion and Content for Video
Generation. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) , 1526–1535.
Victor, S.; Lysandre, D.; Julien, C.; and Thomas, W. 2019.
DistilBERT, a distilled version of BERT: smaller, faster,
cheaper and lighter. arXiv .
Wang, Z.; and Liu, J.-C. 2019. Translating Math Formula
Images to LaTeX Sequences Using Deep Neural Networks
with Sequence-level Training. arXiv:1908.11415.
Yu, Q.; Tanaka, M.; and Fujiwara, K. 2024. ReMoGPT: Part-
Level Retrieval-Augmented Motion-Language Models. In
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR) , 1–12.
Yu, Q.; Tanaka, M.; and Fujiwara, K. 2025. ReMoGPT: Part-
Level Retrieval-Augmented Motion-Language Models. Pro-
ceedings of the AAAI Conference on Artificial Intelligence ,
39(9): 9635–9643.
Yuan, W.; Shen, W.; He, Y .; Dong, Y .; Gu, X.; Dong,
Z.; Bo, L.; and Huang, Q. 2024. MoGenTS: Motion
Generation based on Spatial-Temporal Joint Modeling.
arXiv:2409.17686.
Zhang, J.; Li, X.; Xu, S.; Yu, L.; Xu, J.; and Dai, Q. 2023a.
MotionGPT: Human Motion as a Foreign Language. arXiv
preprint arXiv:2306.10995 .
Zhang, J.; Zhang, Y .; Cun, X.; Huang, S.; Zhang, Y .; Zhao,
H.; Lu, H.; and Shen, X. 2023b. T2m-gpt: Generating hu-
man motion from textual descriptions with discrete repre-
sentations. arXiv:2301.06052 .
Zhang, M.; Cai, Z.; Pan, L.; Hong, F.; Guo, X.; Yang, L.; and
Liu, Z. 2024a. MotionDiffuse: Text-Driven Human Motion
Generation with Diffusion Model. IEEE Transactions on
Pattern Analysis and Machine Intelligence (TPAMI) .
Zhang, M.; Guo, X.; Pan, L.; Cai, Z.; Hong, F.; Li, H.; Yang,
L.; and Liu, Z. 2023c. ReMoDiffuse: Retrieval-Augmented
Motion Diffusion Model. arXiv:2304.01116.
Zhang, M.; Guo, X.; Pan, L.; Cai, Z.; Hong, F.; Li, H.; Yang,
L.; and Liu, Z. 2023d. ReMoDiffuse: Retrieval-Augmented
Motion Diffusion Model. In IEEE International Conference
on Computer Vision (ICCV) , 1–10.
zhang, M.; Li, H.; Cai, Z.; Ren, J.; Yang, L.; and Liu, Z.
2023. FineMoGen: Fine-Grained Spatio-Temporal Motion
Generation and Editing. In NIPS .
Zhang, Z.; Gao, H.; Liu, A.; Chen, Q.; Chen, F.; Wang, Y .;
Li, D.; Zhao, R.; Li, Z.; Zhou, Z.; et al. 2024b. Kmm: Keyframe mask mamba for extended motion generation. arXiv
preprint arXiv:2411.06481 .
Zhang, Z.; Liu, A.; Chen, Q.; Chen, F.; Reid, I.; Hartley,
R.; Zhuang, B.; and Tang, H. 2024c. Infinimotion: Mamba
boosts memory in transformer for arbitrary long motion gen-
eration. arXiv preprint arXiv:2407.10061 .
Zhang, Z.; Liu, A.; Reid, I.; Hartley, R.; Zhuang, B.; and
Tang, H. 2024d. Motion mamba: Efficient and long se-
quence motion generation. In European Conference on
Computer Vision , 265–282. Springer.
Zhang, Z.; Wang, Y .; Mao, W.; Li, D.; Zhao, R.; Wu, B.;
Song, Z.; Zhuang, B.; Reid, I.; and Hartley, R. 2025. Mo-
tion anything: Any to motion generation. arXiv preprint
arXiv:2503.06955 .
Zhang, Z.; Wang, Y .; Wu, B.; Chen, S.; Zhang, Z.; Huang,
S.; Zhang, W.; Fang, M.; Chen, L.; and Zhao, Y . 2024e. Mo-
tion avatar: Generate human and animal avatars with arbi-
trary motion. arXiv preprint arXiv:2405.11286 .
Zou, Q.; Yuan, S.; Du, S.; Wang, Y .; Liu, C.; Xu, Y .; Chen, J.;
and Ji, X. 2024. ParCo: Part-Coordinating Text-to-Motion
Synthesis. arXiv:2403.18512.

User Study
To comprehensively evaluate the generation capability of
ReMoMask , we conducted a comparative user study. We
randomly selected 20 text prompts from the HumanML3D
test set and generated motion sequences using ReMo-
Mask , current state-of-the-art retrieval-augmented method
(ReMoDiffuse), generative model (MoMask), and ground
truth motions.
We employ a forced-choice paradigm in our user study,
asking participants two key questions: “Which of the two
motions is more realistic?” and “Which of the two motions
corresponds better to the text prompt?”. The study is con-
ducted via a Google Forms interface, as illustrated in Fig-
ure 7. To ensure fairness and reduce potential bias, the names
of the generative models are hidden, and the order of pre-
sentation is randomized for each question. In total, over 50
participants took part in the evaluation.
Empirical results, depicted in Figure 3 and Figure 4, un-
derscore ReMoMask’s strong capability to generate motions
that are not only realistic but also closely aligned with tex-
tual descriptions. Specifically, as shown in Figure 3, ReMo-
Mask achieves a 42% preference rate over ground truth (GT)
in terms of realism. Although GT motions are derived from
real human data, this result indicates that ReMoMask is per-
ceived as comparably realistic by human evaluators. More-
over, the model significantly outperforms both baselines: it
achieves 67% preference over MoMask and 75% over Re-
MoDiffuse, demonstrating its strength in producing high-
quality, lifelike motion sequences.
In terms of text correspondence (reported in Figure 4),
ReMoMask attains a 47% preference rate over GT, suggest-
ing that its generated motions exhibit nearly human-level
alignment with text prompts. Compared to the baselines, Re-
MoMask again shows substantial improvements, with 72%
preference over MoMask and 86% over ReMoDiffuse.
Figure 3: Motion Quality User Study
Limitation and Future Work
Current Limitations : BMM’s dual queues and SSTA’s
2D attention significantly increase the model parameters
(238M), hindering real-time deployment. Furthermore, ex-
periments conducted primarily on short sequences (¡100
Figure 4: Text-Motion Correspondence User Study
frames) lack validation for complex motions requiring
strong spatiotemporal coherence, such as dance. Part-level
retrieval also struggles with abstract textual descriptions
(e.g., ”jumping joyfully”) due to reliance on predefined mo-
tion partitions. Additionally, generated motions may violate
biomechanical constraints (e.g., joint rotation limits) due to
the lack of physics-based verification.
Proposed Future Work : To address these limitations, we
propose: (1) Adopting knowledge distillation or sparse at-
tention mechanisms to reduce model size; (2) Decomposing
long motions into sub-actions and applying phased SSTA
to enhance temporal consistency; (3) Integrating Large Lan-
guage Models (LLMs, e.g., GPT-4) to parse abstract texts
and dynamically adapt part-level retrieval; (4) Incorporating
physical constraint losses during RVQ-V AE decoding to en-
sure biomechanically valid motions.
Visualization
Figure 5 demonstrates our model’s capability in generating
diverse human motions. The 16 randomly inferred samples
exhibit complex motion patterns such as directional transi-
tions (”walks toward the front, turns to the right”), rhythmic
actions (”raises arms three times”), and semantically rich be-
haviors (”pretending to be a chicken”). This showcases our
model’s proficiency in capturing nuanced motion dynamics
and temporal transitions.
Figure 6 provides a comparative analysis against
MeGenTS, TMR, and ReMoDiffuse. While baseline models
generate basic motions like walking or balancing, our ap-
proach consistently produces more natural transitions (e.g.,
”walks forward then turns” vs. simple linear motion) and
physically plausible sequences (e.g., multi-step ”jumps for-
ward three times”). The visual comparison highlights our
model’s superior handling of motion complexity and behav-
ioral expressiveness.

Figure 5: We randomly sample and visualize 16 motions generated by the proposed ReMoMask framework. These examples are
conditioned on diverse prompts randomly selected from the HumanML3D (Guo et al. 2022b), providing qualitative evidence
of the model’s ability to synthesize a wide range of realistic and semantically coherent motions.

Figure 6: Comparison of the proposed ReMoMask with three state-of-the-art methods: MoGenTS (Yuan et al. 2024),
TMR (Petrovich, Black, and Varol 2023), and ReMoDiffuse (Zhang et al. 2023d) We visualize motion sequences generated
in response to three distinct text prompts. Each row corresponds to a specific prompt, and each column represents the output
of a different method. The results demonstrate that ReMoMask produces more realistic and semantically aligned motions com-
pared to existing approaches.

Figure 7: This figure illustrates the User Interface (UI) used in the ReMoMask User Study. Participants are presented with
two motion videos, labeled as Motion A and Motion B, alongside a shared textual prompt. The motion clips are sampled from
outputs generated by different models or the ground truth (GT), with model identities anonymized and video order randomized.
Participants are asked to answer two evaluative questions: (1) “Which of the two motions is more realistic?”, assessing the
visual plausibility and motion quality; and (2) “Which of the two motions corresponds better to the text prompt?”, evaluating
the semantic alignment between the motion and the given description. This dual-question design enables a comprehensive
human assessment of both motion realism and text-motion correspondence.