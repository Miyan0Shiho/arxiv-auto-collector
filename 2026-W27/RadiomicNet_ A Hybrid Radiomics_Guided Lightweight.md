# RadiomicNet: A Hybrid Radiomics-Guided Lightweight Architecture for Interpretable Medical Image Segmentation

**Authors**: Mohammad Amanour Rahman

**Published**: 2026-07-02 13:54:33

**PDF URL**: [https://arxiv.org/pdf/2607.02185v1](https://arxiv.org/pdf/2607.02185v1)

## Abstract
Deep learning has achieved remarkable performance in medical image segmentation, yet it suffers from critical limitations: mathematical intractability, substantial parameter requirements, and lack of clinical interpretability. We propose RadiomicNet, a novel two-stream hybrid architecture that enhances standard deep learning by integrating handcrafted radiomics features directly into the segmentation learning process. The key contribution is the Radiomics Attention Gate (RAG), which leverages Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Pattern (LBP) features to modulate skip-connection attention in a lightweight MobileNetV2-based encoder-decoder, providing ante-hoc interpretability without post-hoc approximations. A novel Radiomics Consistency Loss further enforces alignment between texture complexity and prediction uncertainty, reducing Expected Calibration Error (ECE) from 0.142 to 0.118. RadiomicNet achieves a Dice Similarity Coefficient (DSC) of 0.763 +/- 0.231 on the Breast Ultrasound Images (BUSI) dataset and 0.854 +/- 0.112 on Kvasir-SEG, outperforming U-KAN by 1.2% and 1.8%, respectively (p < 0.05, Wilcoxon signed-rank test), with only 3.27M parameters, 9.5x fewer than standard U-Net and 4.3x fewer than U-KAN. Gradient-based feature importance analysis reveals that GLCM dissimilarity (15.24%), GLCM energy (14.56%), and LBP entropy (11.49%) are the dominant radiomics cues, providing clinically meaningful explanations for segmentation decisions. The proposed approach demonstrates that compact, interpretable models grounded in domain knowledge can deliver state-of-the-art segmentation performance with substantially reduced computational overhead.

## Full Text


<!-- PDF content starts -->

RadiomicNet: A Hybrid Radiomics-Guided
Lightweight Architecture for Interpretable Medical
Image Segmentation
Mohammad Amanour Rahman
Department of Computer Science and Engineering
Ahsanullah University of Science and Technology (AUST)
Dhaka, Bangladesh
Email: amanourrahman609@gmail.com
Abstract—Deep learning has achieved remarkable perfor-
mance in medical image segmentation, yet it suffers from critical
limitations: mathematical intractability, substantial parameter
requirements, and lack of clinical interpretability. We propose
RadiomicNet, a novel two-stream hybrid architecture that en-
hances standard deep learning by integrating handcrafted ra-
diomics features directly into the segmentation learning process.
The key contribution is the Radiomics Attention Gate (RAG),
which leverages Gray-Level Co-occurrence Matrix (GLCM) and
Local Binary Pattern (LBP) features to modulate skip-connection
attention in a lightweight MobileNetV2-based encoder–decoder,
providingante-hocinterpretability without post-hoc approxi-
mations. A novel Radiomics Consistency Loss further enforces
alignment between texture complexity and prediction uncertainty,
reducing Expected Calibration Error (ECE) from 0.142 to 0.118.
RadiomicNet achieves a Dice Similarity Coefficient (DSC) of
0.763±0.231on the Breast Ultrasound Images (BUSI) dataset
and0.854±0.112on Kvasir-SEG, outperforming U-KAN by1.2%
and1.8%respectively (p <0.05, Wilcoxon signed-rank test),
with only 3.27M parameters —9.5×fewer than standard U-Net
and4.3×fewer than U-KAN. Gradient-based feature importance
analysis reveals that GLCM dissimilarity (15.24%), GLCM
energy (14.56%), and LBP entropy (11.49%) are the dominant
radiomics cues, providing clinically meaningful explanations for
segmentation decisions. The proposed approach demonstrates
that compact, interpretable models grounded in domain knowl-
edge can deliver state-of-the-art segmentation performance with
substantially reduced computational overhead.
Index Terms—Medical image segmentation, radiomics, inter-
pretable AI, attention mechanism, beyond deep learning, breast
ultrasound, polyp segmentation, lightweight model
I. INTRODUCTION
Deep convolutional neural networks have transformed med-
ical image segmentation, with architectures such as U-Net [1]
and its variants [2], [3] establishing strong benchmarks across
modalities. However, these models are characterized by three
fundamental limitations that impede clinical deployment: (1)
mathematical intractability— predictions are not traceable to
human-understandable features; (2)parameter inefficiency—
standard U-Net requires 31M parameters, posing challenges
for edge deployment; and (3)data hunger— performance
degrades significantly under limited annotation regimes.
The LBDL workshop calls for learning paradigms that prior-
itize interpretability, smaller model sizes, lower computationalcomplexity, and high performance [14]. Radiomics — the ex-
traction of high-throughput quantitative features from medical
images — offers precisely this complementary perspective [7].
Features such as GLCM texture statistics and Local Binary
Patterns are mathematically defined, clinically validated, and
dataset-agnostic.
Despite their complementary strengths, the integration of
radiomics featuresintothe segmentation learning process —
rather than as a post-hoc analysis layer — remains largely
unexplored. Existing hybrid approaches concatenate deep fea-
tures with handcrafted ones only for classification tasks, not
for pixel-level segmentation attention [10]. Recent KAN-based
methods [4], [5] improve interpretability through learnable
activation functions but do not incorporate clinically grounded
texture priors.
We address this gap withRadiomicNet, whose primary
contributions are:
•A novelRadiomics Attention Gate (RAG)that uses
handcrafted texture features to guide skip-connection
attention, providing ante-hoc interpretability traceable to
specific radiomics features.
•ARadiomics Consistency Lossthat enforces alignment
between GLCM-derived texture complexity and predic-
tion uncertainty, with demonstrated calibration improve-
ment.
•Empirical validation on two distinct medical imaging
domains — breast ultrasound (BUSI) and colonoscopy
(Kvasir-SEG) — with statistically significant improve-
ments over all baselines, using only 3.27M parameters.
II. RELATEDWORK
A. Deep Learning for Medical Image Segmentation
U-Net [1] introduced the encoder-decoder architecture with
skip connections that remains the dominant paradigm. Tran-
sUNet [3] incorporated Transformers for global context mod-
eling, while UNet++ [2] introduced dense nested connections.
These approaches achieve strong performance but at the cost
of large parameter counts (9M–93M) and opacity.arXiv:2607.02185v1  [cs.CV]  2 Jul 2026

Fig. 1. RadiomicNet architecture. The Radiomics Attention Gate (RAG) fuses
handcrafted texture features with deep skip connections at each decoder stage,
providing ante-hoc interpretability.
B. KAN and Interpretable Architectures
Kolmogorov-Arnold Networks (KAN) [6] propose learnable
univariate activation functions as a mathematically grounded
alternative to MLPs. U-KAN [4] and IKANet [5] extend this
to medical segmentation with improved interpretability at the
activation level. However, these methods do not incorporate
domain-specific clinical priors, and their interpretability re-
mains at the network architecture level rather than at the
clinically meaningful feature level.
C. Radiomics in Medical Imaging
Radiomics extracts quantitative imaging biomarkers —
shape, intensity, and texture — that are clinically validated
and physiologically meaningful [8]. GLCM-MAE [9] exploits
GLCM-based reconstruction loss for self-supervised pretrain-
ing, demonstrating that texture-aware objectives improve med-
ical image representation. To our knowledge, no prior work
uses radiomics features asspatial attention modulatorswithin
a segmentation decoder.
III. METHODOLOGY
A. Overview
RadiomicNet is a two-stream architecture (Fig. 1). Stream
1 is a MobileNetV2 encoder-decoder [13] operating on RGB
images. Stream 2 is a lightweight MLP processing 13 hand-
crafted radiomics features. The two streams are fused at each
decoder level through the Radiomics Attention Gate (RAG).
B. Radiomics Feature Extraction
We extractR= 13features from each input image prior
to any augmentation, preserving their physical interpretability.
The feature set was designed to span three clinically orthog-
onal axes established in validated radiomics literature [7],
[8]: (i)GLCM texture statistics, capturing tissue heterogeneity
and co-occurrence patterns that are well-validated as imagingbiomarkers for lesion characterisation; (ii)LBP structural
descriptors, encoding local micro-texture patterns sensitive to
boundary irregularity; and (iii)global intensity statistics, cap-
turing first-order tissue density distributions. This principled,
literature-driven selection avoids dataset-specific overfitting
and ensures cross-domain applicability, as both GLCM and
LBP features are modality-agnostic descriptors defined on
pixel intensity relationships rather than modality semantics.
GLCM features(d={1,3},θ={0,45,90,135}, 64
quantization levels): contrast (f 1), dissimilarity (f 2), homo-
geneity (f 3), energy (f 4), correlation (f 5).
LBP features(radius=3,P= 24points, uniform method):
mean (f 6), std (f 7), entropy (f 8), uniformity (f 9).
Intensity statistics: mean (f 10), std (f 11), skewness (f 12),
kurtosis (f 13).
These features arer∈R13passed through a Radiomics
Stream MLP:
ˆr=LayerNorm(ReLU(W 2·ReLU(W 1r+b 1) +b 2))(1)
producing a 32-dimensional representation ˆr∈R32.
C. Radiomics Attention Gate (RAG)
At each decoder levelℓ, the RAG receives skip features
Fℓ∈RB×C ℓ×H ℓ×W ℓand the radiomics embedding ˆr. It
produces an attention-modulated feature map:
Channel attentionfrom radiomics:
αc=σ(W c2·ReLU(W c1ˆr))∈RCℓ(2)
Spatial attentionfrom feature statistics:
αs=σ(Conv 7×7(Fℓ))∈RHℓ×W ℓ(3)
Gated fusionvia learnable scalarγ∈[0,1]:
F′
ℓ=Fℓ⊙[γ·αc·αs+ (1−γ)·αs](4)
The scalar gateγis learned during training and reveals
the relative contribution of radiomics-driven vs. feature-driven
attention — itself an ante-hoc, interpretable quantity. The total
parameter count of 3.27M is detailed as follows: the Mo-
bileNetV2 backbone,withoutits original 1,280-dim classifica-
tion head (which contributes≈1.28M parameters), contributes
≈2.20M parameters; the lightweight decoder, RAG modules
across four decoder levels, and the Radiomics Stream MLP
together contribute≈1.07M parameters, yielding 3.27M in
total.
D. Composite Loss Function
The total loss combines four terms:
L=λ 1LBCE+λ 2LDice+λ 3Lbnd+λ 4LRC (5)
whereλ 1=0.5,λ 2=0.5,λ 3=0.1,λ 4=0.05.
Lbndis a boundary-weighted BCE with dilation factorα=
3, penalizing errors near lesion boundaries3×more heavily.
The novelRadiomics Consistency LossL RCenforces the
clinically grounded principle that regions with high texture
complexity (high GLCM contrast) correspond to diagnostically

uncertain regions where the model should produce calibrated,
lower-confidence predictions:
LRC=MSE
¯H(p),ef1
(6)
where ¯H(p)is mean prediction entropy and ef1is normalized
GLCM contrast. This constraint embeds a radiological induc-
tive bias directly into the training objective: texture complexity
is a clinically validated correlate of diagnostic uncertainty in
breast ultrasound [8]. As shown in Section IV-G,L RCreduces
ECE from 0.142 to 0.118, confirming that this soft constraint
produces meaningfully better-calibrated predictions.
E. Implementation Details
The MobileNetV2 encoder is initialized with ImageNet
pretrained weights, with differential learning rates: encoder
1×10−5, decoder1×10−4, AdamW optimizer with weight
decay10−4, linear warmup (5 epochs) followed by cosine
annealing, batch size 8, gradient clipping at norm 1.0, early
stopping with patience 20. All experiments run on a single
NVIDIA GPU.
IV. EXPERIMENTS
A. Datasets
BUSI[11]: 647 breast ultrasound images (437 benign,
210 malignant) with pixel-level lesion masks. Multiple masks
per image are merged via logical OR. Split: 452/97/98
(train/val/test), stratified by class.
Kvasir-SEG[12]: 1,000 colonoscopy images with polyp
segmentation masks. Official train/val splits used where avail-
able, with held-out test set for evaluation.
B. Evaluation Metrics
We report DSC, IoU, Precision, Recall, and Specificity
(mean±std), computed per-sample and averaged over
the test set. Statistical significance is assessed via paired
Wilcoxon signed-rank tests on per-image DSC scores against
the strongest baseline (U-KAN).
C. Comparison Baselines
We compare against U-Net [1], UNet++ [2], TransUNet [3],
and U-KAN [4] — covering the CNN, Transformer, and KAN
paradigms. All baselines use their original architectures with
pretrained encoders where applicable, trained under identical
data splits and augmentation protocols.
D. Results
As shown in Tables I and II, RadiomicNet achieves state-
of-the-art DSC on both datasets while requiring only 3.27M
parameters —9.5×fewer than U-Net and4.3×fewer than U-
KAN. On BUSI, RadiomicNet surpasses U-KAN by+1.2%
DSC and+1.7%IoU; on Kvasir-SEG the margin widens to
+1.8%DSC and+3.8%IoU, demonstrating stronger cross-
domain generalisation. Paired Wilcoxon signed-rank tests con-
firm that both improvements are statistically significant (p <
0.05) on per-image DSC scores. The high recall (0.845 / 0.886)
alongside competitive precision (0.762 / 0.865) indicates thatTABLE I
SEGMENTATION PERFORMANCE ON THEBUSIDATASET. BEST RESULTS
INBOLD.†DENOTES THE PROPOSED METHOD.∗p <0.05VS. U-KAN
(WILCOXON SIGNED-RANK TEST,PER-IMAGEDSC). 95% CIFOR
RADIOMICNETDSC: [0.716, 0.810].
Method Params (M) DSC IoU Precision Recall
U-Net [1] 31.0 0.726 0.618 – –
UNet++ [2] 9.0 0.740 0.634 – –
TransUNet [3] 93.0 0.748 0.641 – –
U-KAN [4] 14.0 0.751 0.645 – –
RadiomicNet†(Ours)3.270.763∗0.662 0.762 0.845
TABLE II
SEGMENTATION PERFORMANCE ON THEKVASIR-SEGDATASET. BEST
RESULTS INBOLD.†DENOTES THE PROPOSED METHOD.∗p <0.05VS.
U-KAN (WILCOXON SIGNED-RANK TEST,PER-IMAGEDSC). 95% CIFOR
RADIOMICNETDSC: [0.829, 0.879].
Method Params (M) DSC IoU Precision Recall
U-Net [1] 31.0 0.818 0.721 – –
UNet++ [2] 9.0 0.825 0.730 – –
TransUNet [3] 93.0 0.833 0.742 – –
U-KAN [4] 14.0 0.836 0.745 – –
RadiomicNet†(Ours)3.270.854∗0.783 0.865 0.886
the RAG mechanism effectively suppresses false negatives
— a clinically critical property for lesion detection tasks
where missed detections carry greater clinical risk than false
positives.
E. Ablation Study
Table III isolates the contribution of each architectural com-
ponent. Removing the RAG entirely (reducing the model to the
plain MobileNetV2 baseline) causes a 3.2% DSC drop, con-
firming that radiomics-guided attention meaningfully improves
segmentation beyond mere parameter addition. RemovingL RC
causes a 1.5% drop, demonstrating the benefit of consistency-
enforced calibration. The RAG stream decomposition (bottom
block) further isolates whether gains arise from the radiomics-
derived channel attentionαcspecifically, or whether compa-
rable gains could be achieved by spatial attention alone.
F . Interpretability Analysis
Ante-hoc interpretability.RadiomicNet’s interpretability
operates at two distinct levels. First, the architecture isstruc-
turally ante-hoc: the radiomics features (r∈R13) are mathe-
matically defined, clinically validated inputs whose semantic
meaning is fixed prior to training. The learned gate scalarγ
and channel attention weightsαcare direct linear functions
of these named features, making the model’s radiomics-driven
modulation traceable by construction — not through approx-
imation. Second, the Radiomics Consistency Loss further
encodes a clinical prior (texture complexity→prediction
uncertainty) as a structural training constraint, embedding
domain knowledge into the learning objective itself.
Post-hoc diagnostic analysis.Gradient-based feature im-
portance analysis (Fig. 2) — computed via input gradient
magnitude w.r.t. the loss — is used as anadditional diagnostic

TABLE III
EXTENDED ABLATION STUDY ONBUSITEST SET.TOP BLOCK:COMPONENT-LEVEL ABLATION.BOTTOM BLOCK: RAGATTENTION STREAM
DECOMPOSITION. ALL VARIANTS USE IDENTICAL HYPER-PARAMETERS.
Variant DSC IoU
Component ablation
w/o RAG (MobileNetV2 baseline)0.731±0.24 0.628±0.25
w/oL RC 0.748±0.23 0.645±0.24
w/oL bnd 0.756±0.23 0.658±0.24
RAG stream decomposition
RAG: spatial attention only (no radiomics channelαc)0.743±0.18 0.642±0.21
RAG: channel attention only (no spatialαs)0.741±0.11 0.648±0.22
RAG: full (channelαc+ spatialαs+ gateγ)0.763±0.23 0.662±0.25
RadiomicNet (full, all components)0.763±0.23 0.662±0.25
Fig. 2. Radiomics feature importance via input gradient magnitude (averaged
over BUSI test set). GLCM features dominate, consistent with clinical knowl-
edge of texture heterogeneity in breast lesions. Note: this gradient analysis is
a post-hoc diagnostic confirming the ante-hoc design; RadiomicNet’s primary
interpretability derives from its structural radiomics integration.
TABLE IV
CALIBRATION ANALYSIS ONBUSITEST SET. LOWERECEINDICATES
BETTER-CALIBRATED PREDICTIONS.
Variant ECE↓DSC↑
w/oL RC 0.142 0.748
RadiomicNet (withL RC)0.118 0.763
toolto verify that the model’s learned weights align with
known radiological priors. This analysis reveals that GLCM
dissimilarity (15.24%) and GLCM energy (14.56%) are the
dominant texture cues, consistent with radiological knowledge
that malignant lesions exhibit higher textural heterogeneity [8].
LBP entropy (11.49%) captures local structural irregularity.
This gradient analysis is secondary to, not the basis of,
RadiomicNet’s interpretability claim: it serves as empirical
verification that the ante-hoc design intent is realised in
practice.
G. Calibration Analysis
To quantify the calibration benefit ofL RC, we report Ex-
pected Calibration Error (ECE) computed with 15 equal-width
probability bins on the BUSI test set. ECE measures the mean
gap between predicted confidence and actual accuracy across
confidence intervals, providing a model-calibration metric
complementary to DSC.
As shown in Table IV,L RCreduces ECE from 0.142 to
0.118 (a 16.9% relative improvement), confirming that en-
forcing alignment between texture complexity and prediction
entropy produces meaningfully better-calibrated predictions,
not merely a marginal DSC gain.
Fig. 3. Qualitative segmentation results on BUSI test set. Columns: input
image, ground truth, prediction, RAG attention map, overlay. Green=TP,
Red=FP, Blue=FN. High attention regions correlate with textural lesion
boundaries, demonstrating clinically coherent focus.
H. Qualitative Results
Fig. 3 shows representative segmentation results together
with the corresponding RAG attention maps on the BUSI
test set. High-attention regions consistently align with lesion
boundaries and heterogeneous texture patterns, indicating that
the proposed radiomics-guided attention mechanism focuses
on clinically relevant structures while suppressing background
tissue. The overlay visualization further demonstrates accurate
boundary delineation with minimal false-positive and false-
negative regions.
I. Computational Efficiency
RadiomicNet achieves 8.85 ms per-image inference on a
single GPU, enabling real-time clinical deployment at 113
images/sec throughput.
V. DISCUSSION
RadiomicNet demonstrates that incorporating domain
knowledge in the form of radiomics features as structural
priors — rather than learned from data alone — yields a

TABLE V
MODEL COMPLEXITY COMPARISON.
Model Params (M) Size (MB) Infer. (ms)
U-Net 31.0 124.0 –
UNet++ 9.0 36.0 –
TransUNet 93.0 372.0 –
U-KAN 14.0 56.0 –
RadiomicNet 3.27 13.1 8.85
learning paradigm that is simultaneously more parameter-
efficient, more interpretable, and competitive in performance.
This aligns directly with the LBDL vision: the model’s deci-
sion process is traceable to mathematically defined, clinically
validated features rather than abstract neural activations.
The Radiomics Consistency Loss introduces a soft inductive
bias derived from clinical knowledge (texture complexity cor-
relates with diagnostic uncertainty), embedding radiological
reasoning into the training objective rather than appending it
post-hoc. The demonstrated ECE reduction (0.142 to 0.118)
confirms that this constraint is not a marginal regulariser but
a meaningful calibration mechanism.
Limitations.The current radiomics feature set is fixed at
13 features; future work could learn an optimal radiomics
subset via differentiable feature selection. The model currently
processes 2D slices; extension to 3D volumes for CT/MRI
modalities is planned. Evaluation on additional modalities (CT,
MRI) remains a direction for future work.
VI. CONCLUSION
We presented RadiomicNet, a two-stream hybrid archi-
tecture that integrates handcrafted radiomics texture features
into a lightweight segmentation network through the novel
Radiomics Attention Gate. By embedding clinical knowledge
directly into the attention mechanism and loss function, Ra-
diomicNet achieves ante-hoc interpretability — not as an
afterthought, but as a structural property. With 3.27M param-
eters, 13.1 MB model size, statistically significant DSC im-
provements (p <0.05) across two medical imaging modalities,
and demonstrated calibration improvement (ECE: 0.142→
0.118), RadiomicNet offers a viable path toward interpretable,
edge-deployable clinical AI that goesbeyond deep learning.
REFERENCES
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional net-
works for biomedical image segmentation,” inProc. MICCAI, 2015, pp.
234–241.
[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “UNet++:
A nested U-Net architecture for medical image segmentation,” inDeep
Learning in Medical Image Analysis, 2018, pp. 3–11.
[3] J. Chenet al., “TransUNet: Rethinking the U-Net architecture design for
medical image segmentation through the lens of transformers,”Medical
Image Analysis, vol. 97, 2024.
[4] C. Li, X. Liu, W. Li, C. Wang, H. Liu, and Y . Yuan, “U-KAN makes
strong backbone for medical image segmentation and generation,” in
Proc. AAAI, 2025, pp. 4652–4660.
[5] M. Chen, J. Zhang, L. Jin, X. Xuan, and Y . Cheng, “IKANet: A
Kolmogorov-Arnold network for interpretable medical image segmenta-
tion,” inProc. PRCV, 2025.
[6] Z. Liuet al., “KAN: Kolmogorov-Arnold networks,” inProc. ICLR,
2025.[7] P. Lambinet al., “Radiomics: Extracting more information from medical
images using advanced feature analysis,”Eur. J. Cancer, vol. 48, no. 4,
pp. 441–446, 2012.
[8] R. J. Gillies, P. E. Kinahan, and H. Hricak, “Radiomics: Images are more
than pictures, they are data,”Radiology, vol. 278, no. 2, pp. 563–577,
2016.
[9] S. Basuet al., “Focus on texture: Rethinking pre-training in masked
autoencoders for medical image classification,” inProc. MICCAI, 2025.
[10] B. Baidya Kayalet al., “Advances in medical image segmentation:
A comprehensive review of traditional, deep learning and hybrid ap-
proaches,”Bioengineering, vol. 11, no. 10, 2024.
[11] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, “Dataset of
breast ultrasound images,”Data in Brief, vol. 28, 2020.
[12] D. Jhaet al., “Kvasir-SEG: A segmented polyp dataset,” inProc. MMM,
2020, pp. 451–462.
[13] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen,
“MobileNetV2: Inverted residuals and linear bottlenecks,” inProc.
CVPR, 2018.
[14] C.-C. J. Kuo and L. Guan, “The second workshop on learning beyond
deep learning (LBDL II),” inProc. IEEE ICIP Workshop, 2026.