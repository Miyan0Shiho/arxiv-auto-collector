# UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

**Authors**: Qianfeng Yang, Qiyuan Guan, Xiang Chen, Jiyu Jin, Guiyue Jin, Jiangxin Dong

**Published**: 2026-03-04 11:58:51

**PDF URL**: [https://arxiv.org/pdf/2603.03967v1](https://arxiv.org/pdf/2603.03967v1)

## Abstract
Despite significant progress has been made in image deraining, we note that most existing methods are often developed for only specific types of rain degradation and fail to generalize across diverse real-world rainy scenes. How to effectively model different rain degradations within a universal framework is important for real-world image deraining. In this paper, we propose UniRain, an effective unified image deraining framework capable of restoring images degraded by rain streak and raindrop under both daytime and nighttime conditions. To better enhance unified model generalization, we construct an intelligent retrieval augmented generation (RAG)-based dataset distillation pipeline that selects high-quality training samples from all public deraining datasets for better mixed training. Furthermore, we incorporate a simple yet effective multi-objective reweighted optimization strategy into the asymmetric mixture-of-experts (MoE) architecture to facilitate consistent performance and improve robustness across diverse scenes. Extensive experiments show that our framework performs favorably against the state-of-the-art models on our proposed benchmarks and multiple public datasets.

## Full Text


<!-- PDF content starts -->

UniRain: Unified Image Deraining with RAG-based Dataset Distillation and
Multi-objective Reweighted Optimization
Qianfeng Yang1Qiyuan Guan1Xiang Chen2Jiyu Jin1∗Guiyue Jin1Jiangxin Dong2
1Dalian Polytechnic University2Nanjing University of Science and Technology
https://lowlevelcv.com/ https://github.com/QianfengY/UniRain
Abstract
Despite significant progress has been made in image de-
raining, we note that most existing methods are often devel-
oped for only specific types of rain degradation and fail to
generalize across diverse real-world rainy scenes. How to
effectively model different rain degradations within a uni-
versal framework is important for real-world image de-
raining. In this paper, we propose UniRain, an effective
unified image deraining framework capable of restoring
images degraded by rain streak and raindrop under both
daytime and nighttime conditions. To better enhance uni-
fied model generalization, we construct an intelligent re-
trieval augmented generation (RAG)-based dataset distil-
lation pipeline that selects high-quality training samples
from all public deraining datasets for better mixed training.
Furthermore, we incorporate a simple yet effective multi-
objective reweighted optimization strategy into the asym-
metric mixture-of-experts (MoE) architecture to facilitate
consistent performance and improve robustness across di-
verse scenes. Extensive experiments show that our frame-
work performs favorably against the state-of-the-art models
on our proposed benchmarks and multiple public datasets.
1. Introduction
Recent years have witnessed the emergence of diverse rain
datasets [1, 16, 17] and the development of numerous deep
learning–based deraining methods [4, 18, 35]. As the field
has evolved, researchers have increasingly focused on ad-
dressing different rain degradation patterns, such as rain
streaks [48], raindrops [30], nighttime rain [13], and so
on. Such clear settings enable researchers to build models
adapted to the specific characteristics of each rainy scenario.
However, these models are typically designed for spe-
cific rain degradation patterns, and their performance of-
ten drops significantly when applied to other types of rain.
Real-world rainy scenes are highly complex and dynamic.
For example, daytime rain is mainly characterized by streak
degradations, while nighttime scenes, affected by low illu-
High Realism
Diverse S cene
Complex RainOur Distilled Data
High Realism
Limited S cene
Subtle Rain
Mixed Synthetic Data
Low Realism
Diverse S cene
Simplified Rain
Mixed Real -world Data
(a) Visualization of rainy images from different mixed datasets
2Data V olume / million1.5
1
0.5≈2,000k
≈50k
All data Distilled dataPSNRPSNR30
29
28
27
(b) Dataset
 (c) Training imbalance
 (d) Performance difference
Figure 1. Overview ofmotivation. (a) Rainy image samples from
public datasets, illustrating the noticeable differences in data qual-
ity. (b) Directly merging existing synthetic and real datasets en-
larges data volume, yet quality disparity hinders performance, as
shown by PSNR results. (c) The loss curves of DRS, DRD, NRS,
and NRD (denoting daytime/nighttime rain streaks and raindrops)
show different convergence rates, leading to imbalance in unified
training. (d) The PSNR curves indicate that the model tends to fa-
vor simpler degradations but struggles with complex ones.
mination and varying light sources, amplify the complexity
of rain artifacts. Intelligent systems need to handle multiple
rain degradation types, but relying on separate models for
different conditions makes switching cumbersome and sig-
nificantly reduces deployment efficiency [19, 23, 45]. How-
ever, few effort has been made in unified image deraining
due to the absence of mixed datasets that integrate multiple
rain degradation types. Thus, it is of great interest to unlock
a unified image deraining method that can simultaneously
handle multiple rain degradations within a single model.
To achieve better unified image deraining, two key fac-
tors should be considered: constructing mixed dataset and
designing unified model. Regarding themixed dataset,
the most direct strategy is to merge all synthetic and real-
world rain datasets (i.e.,>2,000k pairs) [17, 30, 31, 48, 52].
1arXiv:2603.03967v1  [cs.CV]  4 Mar 2026

However, we note that these public datasets vary noticeably
in quality, as some visual samples illustrated in Figure 1a.
Such uneven data quality may lead to introduce inaccurate
supervisory signals during training, which hinders model
convergence [27]. This instability further causes undesir-
able interference and weakens the model’s ability to gener-
alize across diverse rain conditions, as shown in Figure 1b.
These insights motivate us to distill reliable data from large-
scale public datasets for unified image deraining.
For theunified model, how to equip it with consistent
capability across diverse rain conditions is of vital impor-
tance. As depicted in Figure 1c, we find that different
types of rain degradations exhibit varying levels of diffi-
culty and distinct convergence behaviors during training. If
we train mixed rain-type data with the same optimization
objective, it would lead to imbalance across different degra-
dation types [11]. In this case, the model tends to overfit
easier rain types (e.g., nighttime rain streaks) while strug-
gling with more complex rain degradations (e.g., daytime
raindrops), resulting in uneven restoration quality, as illus-
trated in Figure 1d. This motivates us to introduce a multi-
objective optimization strategy to balance the deraining ca-
pability of the unified model across different rain types.
To address these issues, we propose UniRain, an effec-
tive unified image deraining framework capable of restor-
ing images degraded by rain streak and raindrop under
both daytime and nighttime conditions. Specifically, we
first construct an intelligent retrieval augmented generation
(RAG)-based dataset distillation pipeline that distills reli-
able training samples from all public deraining datasets for
better mixed training. We build a database of real-world
rainy images to guide vision-language models (VLMs) in
assessing data quality, thereby enabling large-scale dataset
evaluation and distillation of high-quality samples for uni-
fied image deraining. Furthermore, we develop an asym-
metric mixture-of-experts (MoE) architecture, where the
encoder integrates the soft-MoE to collaboratively preserve
diverse degradation cues, while the decoder employs the
hard-MoE with top-k routing to enhance fine texture recon-
struction. To achieve better training balance, we incorporate
a simple yet effective multi-objective reweighted optimiza-
tion strategy into the framework to harmonize the learning
of different rain degradation types. Extensive experiments
demonstrate that our framework performs consistently fa-
vorably against the state-of-the-art models on both our pro-
posed benchmarks and multiple public datasets.
We summarize our main contributions as follows:
• We propose UniRain, an effective unified image de-
raining model capable of handling rain streaks and
raindrops under daytime and nighttime conditions.
• We develop an intelligent RAG-based data distillation
pipeline that filters reliable training samples from pub-
lic datasets to improve training on mixed data.• We introduce a simple yet effective multi-objective
reweighted optimization strategy to harmonize the
learning of multiple types and alleviate task imbalance.
• Extensive experiments demonstrate that our Uni-
Rain achieves consistently competitive performance in
terms of both quantitative metrics and visual quality.
2. Related Work
Deep image deraining. In recent years, the field of sin-
gle image deraining has witnessed the emergence of numer-
ous approaches and benchmark datasets [7]. With respect to
deraining datasets, numerous synthetic datasets featuring
different types of rain degradation have been proposed, in-
cluding rain streaks (e.g., Rain200L/H [48], Rain13k [17]),
raindrops (e.g., RainDrop [30], RainDS [31]), and nighttime
rain (e.g., GTA V-NightRain [52], HQ-NightRain [13]). To
boost real-world deraining performance, several real-world
datasets have been proposed, including SPA-Data [40], GT-
Rain [1], and LHP-Rain [16]. It is worth noting that signifi-
cant differences exist among these datasets in terms of back-
ground quality, image resolution, and the formation patterns
of rain. In this work, we propose a dataset distillation strat-
egy to select high-quality training samples from all public
datasets, aiming to better facilitate unified image deraining.
Regarding thederaining models, we note that most ex-
isting approaches remain limited to handling only specific
types of rain degradation, without considering a unified so-
lution that generalizes across diverse real-world rainy sce-
narios. Inspired by the concept of all-in-one image restora-
tion, Yan first contributed a universal deraining network de-
veloped for multiple rain degradation types in driving sce-
narios [45]. To break through the generalization bottleneck
of image deraining, our goal is to develop a unified derain-
ing framework capable of simultaneously handling multiple
rain degradation types across diverse real-world scenarios.
Retrieval augmented generation (RAG). RAG refers to
the process of retrieving relevant information from external
knowledge bases prior to generating outputs with large lan-
guage models (LLMs) [21]. This allows models to leverage
domain-specific information and generate more informed
outputs without requiring retraining [10]. As a cutting-edge
technique, it has been applied in high-level vision tasks,
e.g., image analysis [25, 54], image synthesis [2, 28], and
3D modeling [41, 47]. Recent studies have begun explor-
ing the potential of RAG in low-level vision tasks. For
example, Guoet al.[15] proposed ReFIR, a training-free
retrieval-augmented image restoration framework that re-
trieves content-relevant high-quality references and injects
them into large restoration models to boost fidelity. Inspired
by this popular technique, we first leverage RAG in the pro-
cess of dataset distillation to better facilitate model training.
Multi-objective learning. Multi-objective learning jointly
optimizes models across multiple tasks, leveraging shared
2

Relation
Triplet
All Public 
Datasets
Contextual 
Encoder
Image Caption
Image Feature
Data Prototype
Contextual 
Encoder
Database 𝒟𝑟
Visual 
Encoder
Feature 
SimilaritySemantic 
SimilarityStructural 
Similarity
Prompt Template
VLMs
True
Fake
Distilled 
DataUnlabeled 
Real Rainy
DataVisual 
Encoder
NRD
 DRD
 NRS
 DRS
Expert 2
 Expert 1
 Expert  3
Expert 2
 Expert 1
 Expert 3Asymmetric MoE 
Architecture𝜔1
Reweighting𝓁𝑟𝑎𝑖𝑛=෍
𝑖𝐾
𝜔𝑖𝓁𝑖
Update𝑦=𝛼𝑖𝑥+β𝑖
𝑺𝟏𝑺𝟑𝑺𝟒 𝑺𝟐 𝑩𝟏𝑩𝟑𝑩𝟒 𝑩𝟐
EpochsLoss
t t-N+1Windows
Forward
TypeBalance Score
TypeStability Score
Adaptivity FactorDaytime Rain Streaks (DRS)
Daytime Raindrops (DRD)
Nighttime Rain Streaks (NRS)
Nighttime Raindrops (NRD)TBS
TSS
AFTBS TSS
Hard-MoE DecoderSoft-MoE Encoder
Unified Deraining  Framework 
Labeled Data
(𝑇𝑟, 𝑓𝑟,  𝐼𝑟)
𝐼𝑞
𝑇𝑞
𝑓𝑞Query Triplet
𝒮(𝑞)
𝒮(𝑞)
𝒞2(𝑞)𝒞1(𝑞)𝑃
𝐼𝑞෠𝑅𝑞RAG-based Dataset Distillation
𝑦𝑒𝑛
𝑦𝑑𝑒ℛℎ𝑎𝑟𝑑ℛ𝑠𝑜𝑓𝑡𝑥𝑒𝑛
𝑥𝑑𝑒Multi-objective Reweighted Optimization
Slope Estimation
𝜔2
𝜔3𝜔4
+(1-       )•       AFTBS AF
TSS𝜔𝑖    =       •       
Retrieved
Set
Model
Data 
PyramidFigure 2. Overall framework ofUniRain. (Left) The RAG-based dataset distillation pipeline retrieves real rainy references consistent with
the query image via multi-level similarity search and employs vision language models to evaluate its quality, thereby distilling reliable
samples from public datasets. (Right) The asymmetric MoE architecture consists of soft-MoE encoder and hard-MoE decoder, optimized
via the multi-objective reweighted strategy to achieve balanced learning and robust performance across multiple rain degradation types.
representations to capture common underlying structures
among related tasks [9]. In the context of optimizing multi-
ple objectives that may exhibit conflicting gradients, multi-
objective learning has been extensively investigated to im-
prove both optimization stability and overall task perfor-
mance. To this end, numerous techniques have been pro-
posed, including multi-armed bandit [34], gradient manip-
ulation approaches [49] and self-paced learning [20]. For
low-level vision tasks, Sunet al.[36] proposed a hybrid
evolutionary gradient optimizer to balance perceptual qual-
ity and reconstruction fidelity through multi-objective opti-
mization. In this work, we propose a dynamic reweighted
optimization mechanism to adaptively balance multiple rain
degradation removal objectives for unified image deraining.
3. Methodology
Our goal is to formulate an effective unified image deraining
model that can address multiple types of rain degradations.
Towards this goal, we develop an intelligent RAG-based
dataset distillation pipeline that evaluates large-scale public
datasets and selects high-quality samples for more balanced
mixed training, as illustrated in Figure 2 (left). Furthermore,
to achieve consistent performance and improved robust-
ness, we incorporate a simple yet effective multi-objective
reweighted optimization strategy into the asymmetric MoE
architecture, as shown in Figure 2 (right). In what follows,
we present the details of the proposed approach.
3.1. RAG-based dataset distillation
If all synthetic and real datasets are directly mixed for train-
ing, the uneven data quality may introduce interference and
consequently degrade the model performance. To maximize
the effectiveness of mixed training dataset, we constructthe RAG-based dataset distillation pipeline. The pipeline
mainly consists of retrieval stage, which retrieves relevant
real-world reference, and generation stage, which assesses
data quality and distills reliable samples for training.
Retrieval stage. This stage start from a large-scale knowl-
edge corpus of millions of synthetic and real rainy images,
partitioned into unlabeled real data for database construc-
tion and labeled data for distillation. For each real image
Ir, a textual captionT ris generated using the contextual
encoder BLIP [24], and a visual encoder CLIP [32] is used
to extract its feature embeddingf r. The resulting relation
triplet(T r, fr, Ir)is then stored in the databaseD r. For
each query image, we perform a similar process [12, 24, 46]
to obtain its relation triplet(T q, fq, Iq). Given the query
triplet, we adopt a hierarchical similarity matching process
that considers semantic, feature, and structural similarities
to retrieve the most relevant samples fromD r. Specifically,
we first compute the semantic similarity scores txt(q, r):
stxt(q, r) =∥ϕ T(Tq)−ϕ T(Tr)∥2, (1)
whereϕ T(·)denotes the CLIP text encoder and∥ · ∥ 2de-
notes theL 2norm. The topK 1samples with the highest se-
mantic similarity scores form the initial candidate setC 1(q).
Within this candidate set, we further compute the visual fea-
ture similarity scoress vis(q, r′)using cosine similarity to
assess appearance closeness, as defined below:
svis(q, r′) =f⊤
qfr′
∥fq∥2∥fr′∥2, r′∈ C1(q). (2)
The candidate setC 2(q)is constructed, retaining the top
K2most similar samples. Finally, we employ theSSIM(·)
3

function to evaluate structural similarity and capture fine-
grained perceptual consistency, as defined below:
sperc(q, r′′) =SSIM(I q, Ir′′), r′′∈ C2(q). (3)
The topK 3samples are selected based on the structural
similarity scoress perc(q, r′′)to constructS(q). Through
this hierarchical similarity matching process, the pipeline
ensures thatS(q)contains the most semantically and visu-
ally consistent real rainy scenes, which serve as authentic
references for data distillation.
Generation stage. Once the retrieved setS(q)is obtained,
it is combined with the query imageI qand a predefined
prompt templatePas input to the VLMs. The overall gen-
eration process is expressed as follows:
Rq=V(I q,S(q), P), (4)
whereV(·)is the VLM, andR qrepresents the VLM’s re-
sponse. To enhance reliability, we adopt an ensemble vot-
ing strategy across three independent VLMs (InternVL2.5-
8B [8], LLaV A-NeXT-7B [22], and MobileVLM-3B [43]).
LetRi
q∈ {0,1}denote the binary response from thei-th
model, where 1 indicates True and 0 indicates False. The
final decision is determined by majority voting:
ˆRq=(
1ifP3
i=1I(Ri
q= 1)≥2,
0otherwise,(5)
whereI(·)is the indicator function, and ˆRq= 1indi-
cates the query rainy image is considered reliable. Finally,
this process forms a data-quality pyramid, where the top
layer represents real rainy data, the middle layer consists
of distilled high-quality and reliable data close to real dis-
tributions, and the bottom layer contains low-fidelity sam-
ples. Through this hierarchical refinement, the system dis-
tills millions of images into a reliable dataset.
3.2. Multi-objective reweighted optimization
Training a single model to jointly handle different rain types
introduces optimization imbalance. The shared optimiza-
tion objective tends to favor easier rain type with faster loss
decay, while more difficult ones remain under-optimized.
Consequently, the learned representations become biased,
limiting the model’s overall generalization across diverse
degradations. To address the above imbalance, we pro-
pose multi-objective reweighted optimization, an adaptive
strategy inspired by [11], which dynamically balances con-
vergence rates and harmonizes multi-objective optimization
during training. This strategy consists of convergence slope
estimation, which estimates each type’s convergence rate
through linear regression of loss, and adaptive reweighting,
which integrates type balance score (TBS), type stability
score (TSS), and adaptivity factor (AF), to dynamically bal-
ance type contributions and maintain stable optimization.Convergence slope estimation. We first estimate the con-
vergence tendency of each type by analyzing the evolution
of its loss. Let{(k, y k)}t
k=t−N+1 denote the training-step
indices and their corresponding normalized losses within a
sliding window[t−N+ 1, t], whereNis the window size
andtis the current iteration. To capture short-term con-
vergence dynamics, we apply a least-squares linear fitting
over this window and define the regression coefficientαas
the convergence slope, which measures the local rate of loss
change. The slope and intercept are calculated as:
α=PN
k=1(k−¯k)(y k−¯y)
PN
k=1(k−¯k)2, β= ¯y−α ¯k,(6)
where ¯k=1
NPt
k=t−N+1kand¯y=1
NPt
k=t−N+1ykare
the mean values within the window. In this definition,α <
0indicates a decreasing loss (converging),α >0implies
divergence, and a larger|α|denotes a faster change rate.
Adaptive reweighting. This estimated slopeαserves as the
foundation for computing the subsequent adaptive weight-
ing indicators, namely type balance score (TBS), type sta-
bility score (TSS), and adaptivity factor (AF), which jointly
determine how type weights evolve during training. The
goal of TBS is to assign smaller weights to types that
converge faster and larger weights to those that converge
more slowly, ensuring synchronized convergence among all
types. For each typeiat iterationt, we define the TBS as
TBS i(t) =softmaxi 
Kαi(t)PK
i=1|αi(t)|!
,(7)
whereα i(t)is the convergence slope of typei,softmaxi
operates across the type dimension, andKdenotes the total
number of degradation types. This normalization removes
scale sensitivity and ensures that slower types obtain higher
TBS values, thus receiving greater optimization emphasis.
However, relying solely on TBS is insufficient, since a
diverging type with a large positive slope could still obtain
an undesirably high score. To mitigate this issue, we pro-
pose the TSS, which evaluates each type’s internal stability
by considering its historical convergence behavior:
TSS i(t) =softmaxi 
−Nαi(t)Pt
k=t−N+1|αi(k)|!
,(8)
where normalization is performed along the temporal di-
mensionkfrom stept−N+1tot, but subsequently applies
the softmax function across the type dimensioni. A higher
TSS indicates stable and consistent convergence, whereas
diverging types receive lower scores.
During early training, all types generally exhibit con-
vergent behavior, in which case TBS should dominate the
weighting process. As training proceeds, some types may
4

begin to diverge and thus require greater influence from
TSS. To achieve a transition between these two regimes, we
define the AF that reflects the global divergence state:
AF(t) =min 
t·softmaxt 
−τt·α max(t)Pt
i=1αmax(i)!
,1!
,
(9)
whereα max(t)is the largest convergence slope across all
types andsoftmaxtis applied along the temporal dimen-
sion. The termtprevents AF from monotonically decreas-
ing as training progresses, whileτcontrols its sensitivity.
Finally, the type loss weights are defined as follows:
ωi(t) = AF(t) TBS(t)+ 
1−AF(t)
TSS(t), i∈[1, K]
(10)
whereω i(t)is the dynamic weight vector acrossKrain
types. This approach enables adaptive focus between inter-
type balance and intra-type stability, ensuring steady con-
vergence and improved generalization across all rain types.
3.3. Asymmetric MoE architecture
To address diverse rain degradation behaviors, we adopt an
asymmetric MoE with the soft-MoE encoder and the hard-
MoE decoder for adaptive and efficient processing.
Soft-MoE encoder. Given an input featurex en, the soft
router functionR softgenerates continuous routing weights
to adaptively combine multiple experts:
Rsoft=σ(W(φ(x en+ϵ))),(11)
whereW(·)is a learnable projection,φ(·)is global average
pooling,ϵ∼ N(0, σ2)is the Gaussian noise for load bal-
ancing, andσ(·)is the softmax. The outputy encombines
all experts contributions through weighted aggregation:
yen=NX
i=1Ri
soft⊗yi
en,(12)
where⊗is the element-wise multiplication,yi
enis the fea-
ture produced by thei-th expert,Ri
softis the routing weight
of thei-th expert, andNis the total number of experts.
Hard-MoE decoder. For the decoder inputx de, the hard
routerR hard function selectively activates the most relevant
experts via Top-krouting. The process is defined as:
Rhard=Tk(σ(W(φ(x de+ϵ)))),(13)
yde=NX
i=1Ri
hard·yi
de,(14)
whereT k(·)denotes the Top-kselection operator andyi
de
is the output of thei-th decoder expert. This design en-
ables the hard-MoE to efficiently allocate computation to
the most relevant experts, focusing on structural and fine-
grained detail reconstruction, and complementing the soft-
MoE encoder for balanced expressiveness and efficiency.4. Experiments
4.1. Experimental settings
Training dataset. Through the RAG-based dataset distilla-
tion pipeline, we construct RainRAG, a high-quality dataset
distilled from large-scale (i.e.,>2,000k pairs) public syn-
thetic and real-world deraining datasets [1, 3, 13, 16, 17, 30,
31, 40, 42, 45, 48, 52]. Our pipeline retains approximately
2.6%of the original data, ensuring only high-quality sam-
ples for model training. The training set consists of 52,869
pairs and divided into four representative subsets: daytime
rain streaks (DRS), daytime raindrops (DRD), nighttime
rain streaks (NRS), and nighttime raindrops (NRD).
Test datasets. To comprehensively evaluate our model’s ca-
pability in unified image deraining, we obtain 400 test im-
age pairs using the same filtering process and evenly divide
them into four subsets: DRS, DRD, NRS, and NRD. In ad-
dition, we evaluate our method alongside other comparison
methods on several real-world benchmarks (i.e., RealRain-
1k [26], RainDS-real [31], and WeatherBench [14]).
Implementation details. We train our proposed UniRain
on 4 NVIDIA GeForce RTX 4090 GPUs using the AdamW
optimizer with default parameters. During training, images
are randomly cropped into128×128patches with a batch
size of 8. The initial learning rate is set to2×10−4and grad-
ually reduces to1×10−6using a cosine annealing sched-
uler. For the asymmetric MoE architecture, each expert is
designed with a distinct self-attention capacity, enabling di-
verse feature modeling across varying attention complexi-
ties. We train the network for a total of 300,000 iterations.
For multi-objective reweighted optimization, we set the step
window size to 10 and the sensitivity parameterτto 5.
4.2. Comparison with the state-of-the-art
We compare our proposed UniRain with 10 image derain-
ing technologies, including PReNet [33], RCDNet [39],
MPRNet [50], Restormer [51], IDT [44], DRSformer [5],
RLP [53], MSDT [4], NeRD-Rain [6], and URIR [45].
Evaluation on the RainRAG dataset. Table 1 shows that
the proposed UniRain achieves the best average perfor-
mance across four rain degradation types. Compared with
Restormer [51] and NeRD-Rain [6], it achieves improve-
ments of 1.35 dB and 1.41 dB in PSNR on the DRD subset,
respectively. As shown in Figure 3, we further present the
visual comparisons under various rain degradations, where
our method delivers superior deraining performance. In
contrast, other methods still exhibit rain streak residuals and
fail to recover background details affected by raindrops.
Evaluation on real-world public benchmarks. For further
general verification in practical applications, we conduct
comparisons with other methods on the real-world rainy
datasets. As shown in Table 2, our method achieves the best
5

Table 1. Quantitative evaluations on the proposed RainRAG dataset, where DRS, DRD, NRS, and NRD denote daytime rain streaks,
daytime raindrops, nighttime rain streaks, and nighttime raindrops, respectively. The best and second-best values areboldand underlined .
DatasetsRainRAG Dataset
DRS DRD NRS NRD Average
Methods Venue PSNR↑SSIM↑LPIPS↓ PSNR↑SSIM↑LPIPS↓ PSNR↑SSIM↑LPIPS↓ PSNR↑SSIM↑LPIPS↓ PSNR↑SSIM↑LPIPS↓
LQ - 23.79 0.7514 0.3207 20.79 0.7293 0.2433 26.89 0.8480 0.3583 21.05 0.5590 0.6130 23.13 0.7220 0.3839
PReNet [33] CVPR 2019 26.73 0.8233 0.1885 21.45 0.7358 0.2697 32.85 0.9520 0.1957 24.29 0.8058 0.3751 26.33 0.8292 0.2572
RCDNet [39] CVPR 2020 25.27 0.7949 0.2777 21.46 0.7295 0.2778 28.44 0.9070 0.2996 20.62 0.5832 0.5931 23.95 0.7537 0.3620
MPRNet [50] CVPR 2021 27.99 0.8242 0.1701 22.50 0.7264 0.3308 33.75 0.9541 0.1834 24.46 0.8118 0.3634 27.17 0.8291 0.2619
Restormer [51] CVPR 2022 28.45 0.8298 0.1725 23.36 0.7547 0.2323 33.92 0.9541 0.1830 25.85 0.8234 0.3452 27.89 0.8405 0.2332
IDT [44] TPAMI 2022 26.97 0.8279 0.1960 23.17 0.7648 0.2300 33.05 0.9470 0.2132 25.20 0.7963 0.3718 27.10 0.8340 0.2527
DRSformer [5] CVPR 2023 28.08 0.8253 0.1960 22.93 0.7499 0.2407 32.86 0.9457 0.2202 24.99 0.7953 0.3702 27.22 0.8291 0.2568
RLP [53] ICCV 2023 28.01 0.8290 0.2339 22.42 0.7353 0.3029 32.06 0.9438 0.1971 23.48 0.7638 0.4282 26.49 0.8180 0.2905
MSDT [4] AAAI 2024 28.60 0.8414 0.1576 23.31 0.7487 0.2587 34.56 0.9591 0.1677 25.28 0.8148 0.3555 27.94 0.8410 0.2349
NeRD-Rain [6] CVPR 2024 28.11 0.8309 0.1720 23.30 0.7513 0.2318 33.88 0.9511 0.1964 25.31 0.8027 0.3571 27.65 0.8340 0.2393
URIR [45] AAAI 2025 28.29 0.8386 0.2001 23.19 0.7483 0.2815 34.32 0.9570 0.1860 25.82 0.8263 0.3390 27.91 0.8425 0.2516
UniRain (Ours) - 29.580.8400 0.1814 24.71 0.76870.2408 35.23 0.9592 0.1617 26.21 0.8380 0.3275 28.93 0.8515 0.2279
LQ patch PReNet [33] RCDNet [39] IDT [44] RLP [53]
Rainy image from RainRAG-NRS MSDT [4] NeRD-Rain [6] URIR [45]UniRainGT patch
LQ patch PReNet [33] MPRNet [50] Restormer [51] DRSformer [5]
Rainy image from RainDS-real-RD [31] MSDT [4] NeRD-Rain [6] URIR [45]UniRainGT patch
Figure 3. Visual comparison of image deraining results on the RainRAG-NRS and RainDS-real-RD datasets. Zoom in for a better view.
overall performance, outperforming URIR [45] by 1.73 dB
in PSNR on average. Figure 4 shows that our method not
only removes real and complex rain streaks but also restores
clear details, even achieving visually better results than the
GT by eliminating residual raindrops contained in it.
Generalization to multiple application scenarios. We
further evaluate the generalization ability of our method
across multiple application scenarios, including au-
tonomous driving, unmanned aerial vehicle (UA V), and
maritime scenes. As presented in Figure 5, other methods
struggle to preserve pixel-level structural fidelity, whereas
our method produces faithful and visually pleasing results,
demonstrating generalization across multiple scenarios.
Model complexity. We evaluate the complexity of our
method and state-of-the-art ones in terms of FLOPs andParams. As shown in Table 3, UniRain achieves compet-
itive performance with lower FLOPs and fewer parameters.
4.3. Analysis and discussion
Effectiveness of RAG-based dataset distillation. We first
analyze the impact of different mixed training datasets on
the performance of the unified deraining model, including
(1) synthetic dataset combinations, (2) real-world dataset
combinations, and (3) synthetic&real dataset combina-
tions. As shown in Figure 6a, we present the quantitative re-
sults of the same model trained on these equal-sized dataset
combinations across four rain types. It can be observed that
the model trained with our distilled dataset achieves the best
overall performance, surpassing all other dataset combina-
tions. Figure 6b visualizes the corresponding feature distri-
6

Table 2. Quantitative evaluations on the real-world public benchmarks. The best and second-best values areboldand underlined .
Datasets Metrics PReNet [33] RCDNet [39] MPRNet [50] IDT [44] Restormer [51] DRSformer [5] RLP [53] MSDT [4] NeRD-Rain [6] URIR [45] UniRain (Ours)
RealRain-1k-L [26]PSNR↑ 28.80 28.58 33.89 33.71 32.44 33.65 32.02 33.56 33.98 33.5436.51
SSIM↑ 0.9307 0.9202 0.9701 0.9664 0.9644 0.9625 0.9558 0.9640 0.9710 0.96860.9778
RealRain-1k-H [26]PSNR↑ 25.49 25.46 30.83 30.37 28.80 30.35 29.01 30.91 31.37 31.1033.74
SSIM↑ 0.8820 0.8613 0.9542 0.9441 0.9389 0.9360 0.9309 0.9451 0.9583 0.95500.9672
RainDS-real-RD [31]PSNR↑ 19.08 19.37 19.48 20.35 20.74 20.15 20.86 20.72 21.01 20.4422.07
SSIM↑ 0.6020 0.5937 0.5957 0.6237 0.6346 0.6133 0.6136 0.6205 0.6155 0.59450.6436
RainDS-real-RDS [31]PSNR↑ 18.21 18.57 18.74 18.98 19.13 18.79 19.85 19.36 19.93 19.8720.61
SSIM↑ 0.5665 0.5572 0.5681 0.5833 0.5876 0.5712 0.5762 0.5830 0.5839 0.56890.5881
WeatherBench-rain [14]PSNR↑ 31.18 27.91 32.35 32.05 32.91 31.90 31.48 33.56 32.77 33.5034.25
SSIM↑ 0.9364 0.8880 0.9352 0.9355 0.9372 0.9281 0.93300.94530.9372 0.9435 0.9448
AveragePSNR↑ 24.55 23.98 27.06 27.09 26.80 26.97 26.64 27.62 27.81 27.6929.42
SSIM↑ 0.7835 0.7641 0.8047 0.8106 0.8125 0.8022 0.8019 0.8116 0.8132 0.80610.8222
LQ Restormer [51] MSDT [4] NeRD-Rain [6] URIR [45]UniRainGT
Figure 4. Visual comparison of image restoration results on the real-world benchmark (i.e., WeatherBench [14]). Compared to the state-of-
the-art methods, our UniRain restores a high-quality image with clear details, even outperforming the GT by removing residual raindrops.
Table 3. Comparisons of model complexity against state-of-the-art
methods. The size of the test image is256×256pixels.
Methods Restormer [51] DRSformer [5] NeRD-Rain [6] UniRain (Ours)
FLOPs (G) 140.990 220.378 147.978 126.541
Params (M) 26.097 33.627 22.856 24.388
LQ
 NeRD-Rain [6]
 UniRain
Figure 5. Generalization results across multiple scenarios. Our
method achieves the best restoration results in driving scenes (first
row), UA V scenes (second row), and maritime scenes (third row).
butions, revealing that our distilled dataset covers a wider
and more diverse feature space. This richer representation
contributes to stronger generalization ability and ensures
more consistent deraining performance across diverse rain
conditions. Besides, we compare the averaged results be-
tween directly mixing all synthetic and real-world datasets
and using our distilled dataset on the test set of Table 2.
As shown in Figure 1b, directly combining all datasets for
training leads to suboptimal results due to inconsistent data
quality across different sources. This finding indicates that
simply enlarging the dataset does not necessarily lead to
better generalization. Rather, the reliability and quality of
Avg. PSNR
28.93
27.91
27.76
27.19
(a) Different combination strategies
 (b) Feature distributions
Figure 6. Statistical ablation analysis of different combination
strategies and their feature distributions across combinations.
the data are the key factors that dictate model performance.
Finally, to further validate the effectiveness of our pro-
posed RAG-based dataset distillation pipeline, we conduct
a comparison with several ablated variants, including (i) us-
ing only VLMs without RAG, (ii) without retrieval stage,
and (iii) without generation stage. As shown in Table 4, our
designed pipeline obtains higher performance as it lever-
ages real-scene images to filter training data, ensuring that
high-quality samples are distilled for model training.
Effectiveness of multi-objective reweighted optimiza-
tion. We compare our multi-objective reweighted optimiza-
tion strategy with alternative schemes, including contin-
ual learning (e.g., PIGWM [55]) and prompt learning (e.g.,
PromptIR [29]). We conduct experiments on the RainDS-
real-RDS [31] dataset, with the quantitative results shown in
Table 5. It can be observed that applying our optimization
strategy to the PromptIR [29] architecture yields a 0.97 dB
improvement in PSNR. Compared with continual learning,
our optimization strategy improves UniRain by 1.69 dB in
PSNR, highlighting its effectiveness.
7

Table 4. Ablation analysis for RAG-based dataset distillation.
Variants only VLM w/o Retrieval stage w/o Generation stage UniRain (Ours)
PSNR↑ 27.73 28.39 28.3628.93
SSIM↑ 0.8358 0.8453 0.84250.8515
Table 5. Ablation analysis on various models using different opti-
mizers, including our adaptive reweighting optimization strategy.
Base ModelsOptimization Strategy Metrics
Continual learning Prompt learning Ours PSNR↑SSIM↑
PIGWM [55]" % % 18.38 0.5758
% % " 18.68 0.5880
PromptIR [29]% " % 18.58 0.5639
% % " 19.55 0.5759
UniRain (Ours)" % % 18.92 0.5657
% " % 19.64 0.5737
% % " 20.61 0.5881
(a) Average loss per variants
 (b) Average PSNR per variants
Figure 7. Loss and PSNR of the optimization strategy variants.
To disentangle the interplay between the effectiveness of
optimization strategies and that of models, we further an-
alyze the effect of different models under the same opti-
mization strategy. As shown in Table 5, under the same
prompt-learning optimization setting, UniRain achieves a
1.06 dB gain in PSNR over PromptIR [29]. Moreover, un-
der the same multi-objective reweighted optimization strat-
egy, it surpasses PIGWM [55] by 2.23 dB in PSNR metric.
Finally, we investigate the influence of different variants
of the optimization strategy, including (i) a fixed adaptive
factor (AF = 0.5), (ii) removing the task stability score
(w/o TSS), and (iii) removing the task balance score (w/o
TBS). As illustrated in Figure 7a, the average loss curves
indicate that our full strategy achieves more stable optimiza-
tion throughout training. Meanwhile, the average PSNR re-
sults in Figure 7b confirm that our complete formulation
enables faster convergence and higher overall performance
compared with the ablated variants.
Effectiveness of asymmetric MoE architecture. To vali-
date the effectiveness of asymmetric MoE architecture, we
conduct ablation studies as shown in Table 6. The baseline
model without any MoE modules yields the lowest perfor-
mance, indicating that a plain backbone cannot effectively
model complex and diverse rain degradations. Introducing
hard-MoE in both the encoder and decoder significantly en-
hances performance. In contrast, using soft-MoE in both
stages leads to performance degradation, suggesting that
soft routing alone is insufficient to capture heterogeneous
degradations. Our design adopts soft-MoE in the encoder
for exhaustive expert exploration and hard-MoE in the de-
coder for selective expert activation, and their integrationTable 6. Ablation analysis of asymmetric MoE architecture. Ex-
periments are conducted on the proposed RainRAG dataset.
VariantsEncoder Decoder Metrics
hard-MoE soft-MoE hard-MoE soft-MoE PSNR↑SSIM↑
Baseline % % % % 27.54 0.8442
Variant 1 " % " % 28.39 0.8471
Variant 2 " % % " 28.15 0.8485
Variant 3 % " % " 27.91 0.8465
UniRain (Ours) % " " % 28.93 0.8515
Table 7. Extension to all-in-one weather restoration on the Weath-
erBench dataset [14], which contains real-world multi-weather
degradations (rain, snow, and haze). Average results are reported.
Methods TransWeather [38] WGWS-Net [56] Histoformer [37] UniRain (Ours)
PSNR↑ 24.70 23.89 24.5926.01
SSIM↑ 0.7931 0.7811 0.79760.8032
LQ
 TransWeather
 Histoformer
 UniRain
Figure 8. Visual comparison of all-in-one weather restoration re-
sults (e.g., hazy input (first row) and snowy input (second row)).
enables the model to achieve the best overall performance.
Extension to all-in-one weather restoration. We extend
our network to all-in-one weather restoration and com-
pare it with adverse weather restoration method (i.e., Tran-
sWeather [38]) and all-in-one image restoration approaches
(i.e., WGWS-Net [56], and Histoformer [37]). Table 7 and
Figure 8 show that our framework achieves a clear quantita-
tive advantage and better visual restoration results, demon-
strating its effectiveness in all-in-one weather restoration.
5. Conclusion
We have presented an effective unified framework for im-
age deraining, called UniRain, which addresses rain streak
and raindrop degradations under both daytime and night-
time conditions. We construct an intelligent retrieval aug-
mented generation-based dataset distillation pipeline to bet-
ter improve model generalization, enabling mixed training
with high-quality samples distilled from existing deraining
datasets. To achieve consistent performance and improved
robustness across diverse scenarios, we integrate a simple
yet effective multi-objective reweighted optimization strat-
egy into the asymmetric union-of-experts architecture. Ex-
tensive experiments show that our method achieves compet-
itive performance compared with state-of-the-art models on
both our proposed benchmarks and multiple public datasets.
8

References
[1] Yunhao Ba, Howard Zhang, Ethan Yang, Akira Suzuki,
Arnold Pfahnl, Chethan Chinder Chandrappa, Celso M
De Melo, Suya You, Stefano Soatto, Alex Wong, et al. Not
just streaks: Towards ground truth for single image derain-
ing. InECCV, 2022. 1, 2, 5
[2] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas
M¨uller, and Bj ¨orn Ommer. Semi-parametric neural image
synthesis.arXiv preprint arXiv:2204.11824, 2022. 2
[3] Wenhui Chang, Hongming Chen, Xin He, Xiang Chen, and
Liangduo Shen. Uav-rain1k: A benchmark for raindrop re-
moval from uav aerial imagery. InCVPRW, 2024. 5
[4] Hongming Chen, Xiang Chen, Jiyang Lu, and Yufeng Li. Re-
thinking multi-scale representations in deep deraining trans-
former. InAAAI, 2024. 1, 5, 6, 7
[5] Xiang Chen, Hao Li, Mingqiang Li, and Jinshan Pan. Learn-
ing a sparse transformer network for effective image derain-
ing. InCVPR, 2023. 5, 6, 7
[6] Xiang Chen, Jinshan Pan, and Jiangxin Dong. Bidirectional
multi-scale implicit neural representations for image derain-
ing. InCVPR, 2024. 5, 6, 7
[7] Xiang Chen, Jinshan Pan, Jiangxin Dong, and Jinhui Tang.
Towards unified deep image deraining: A survey and a new
benchmark.IEEE TPAMI, 2025. 2
[8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhang-
wei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,
Zhaoyang Liu, et al. Expanding performance boundaries of
open-source multimodal models with model, data, and test-
time scaling.arXiv preprint arXiv:2412.05271, 2024. 4
[9] Michael Crawshaw. Multi-task learning with deep neural
networks: A survey.arXiv preprint arXiv:2009.09796, 2020.
3
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and
Haofen Wang. Retrieval-augmented generation for large lan-
guage models: A survey.arXiv preprint arXiv:2312.10997,
2023. 2
[11] Zi Gong, Hang Yu, Cong Liao, Bingchang Liu, Chaoyu
Chen, and Jianguo Li. Coba: convergence balancer for mul-
titask finetuning of large language models.arXiv preprint
arXiv:2410.06741, 2024. 2, 4
[12] Maarten Grootendorst. Keybert: Minimal keyword extrac-
tion with bert., 2020. 3
[13] Qiyuan Guan, Xiang Chen, Guiyue Jin, Jiyu Jin, Shumin
Fan, Tianyu Song, and Jinshan Pan. Rethinking nighttime
image deraining via learnable color space transformation.
NeurIPS, 2025. 1, 2, 5
[14] Qiyuan Guan, Qianfeng Yang, Xiang Chen, Tianyu Song,
Guiyue Jin, and Jiyu Jin. Weatherbench: A real-world bench-
mark dataset for all-in-one adverse weather image restora-
tion. InACM MM, 2025. 5, 7, 8
[15] Hang Guo, Tao Dai, Zhihao Ouyang, Taolin Zhang, Yaohua
Zha, Bin Chen, and Shu-tao Xia. Refir: Grounding large
restoration models with retrieval augmentation.NeurIPS,
2024. 2
[16] Yun Guo, Xueyao Xiao, Yi Chang, Shumin Deng, and Luxin
Yan. From sky to the ground: A large-scale benchmark andsimple baseline towards real rain removal. InICCV, 2023. 1,
2, 5
[17] Kui Jiang, Zhongyuan Wang, Peng Yi, Chen Chen, Baojin
Huang, Yimin Luo, Jiayi Ma, and Junjun Jiang. Multi-scale
progressive fusion network for single image deraining. In
CVPR, 2020. 1, 2, 5
[18] Kui Jiang, Zhongyuan Wang, Peng Yi, Chen Chen, Zheng
Wang, Xiao Wang, Junjun Jiang, and Chia-Wen Lin. Rain-
free and residue hand-in-hand: A progressive coupled net-
work for real-time image deraining.IEEE TIP, 2021. 1
[19] Zhiqiang Kou, Jing Wang, Yuheng Jia, and Xin Geng. Pro-
gressive label enhancement.PR, 2025. 1
[20] M Kumar, Benjamin Packer, and Daphne Koller. Self-paced
learning for latent variable models.NeurIPS, 2010. 3
[21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks.NeurIPS, 2020. 2
[22] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li,
Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave:
Tackling multi-image, video, and 3d in large multimodal
models.arXiv preprint arXiv:2407.07895, 2024. 4
[23] Hao Li, Xiang Chen, Jiangxin Dong, Jinhui Tang, and Jin-
shan Pan. Foundir: Unleashing million-scale training data to
advance foundation models for image restoration. InICCV,
2025. 1
[24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
Blip: Bootstrapping language-image pre-training for unified
vision-language understanding and generation. InICML,
2022. 3
[25] Pengpeng Li, Xiangbo Shu, Chun-Mei Feng, Yifei Feng,
Wangmeng Zuo, and Jinhui Tang. Surgical video workflow
analysis via visual-language learning.npj Health Systems,
2025. 2
[26] Wei Li, Qiming Zhang, Jing Zhang, Zhen Huang, Xinmei
Tian, and Dacheng Tao. Toward real-world single image
deraining: A new benchmark and beyond.arXiv preprint
arXiv:2206.05514, 2022. 5, 7
[27] Shih-Jui Liang, Zihao Chen, Jun-Cheng Chen, and Yan-
Tsung Peng. Enhancing image deraining through vlm-based
data refinement and classification. InICIP, 2025. 2
[28] Avideep Mukherjee, Soumya Banerjee, Piyush Rai, and
Vinay P Namboodiri. Rissole: Parameter-efficient diffusion
models via block-wise generation and retrieval-guidance.
arXiv preprint arXiv:2408.17095, 2024. 2
[29] Vaishnav Potlapalli, Syed Waqas Zamir, Salman H Khan,
and Fahad Shahbaz Khan. Promptir: Prompting for all-in-
one image restoration.NeurIPS, 2023. 7, 8
[30] Rui Qian, Robby T Tan, Wenhan Yang, Jiajun Su, and Jiay-
ing Liu. Attentive generative adversarial network for rain-
drop removal from a single image. InCVPR, 2018. 1, 2,
5
[31] Ruijie Quan, Xin Yu, Yuanzhi Liang, and Yi Yang. Remov-
ing raindrops and rain streaks in one go. InCVPR, 2021. 1,
2, 5, 6, 7
9

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. InICML, 2021. 3
[33] Dongwei Ren, Wangmeng Zuo, Qinghua Hu, Pengfei Zhu,
and Deyu Meng. Progressive image deraining networks: A
better and simpler baseline. InCVPR, 2019. 5, 6, 7
[34] Aleksandrs Slivkins et al. Introduction to multi-armed ban-
dits.FTML, 2019. 3
[35] Tianyu Song, Guiyue Jin, Pengpeng Li, Kui Jiang, Xiang
Chen, and Jiyu Jin. Learning a spiking neural network for
efficient image deraining.arXiv preprint arXiv:2405.06277,
2024. 1
[36] Lingchen Sun, Jie Liang, Shuaizheng Liu, Hongwei Yong,
and Lei Zhang. Perception-distortion balanced super-
resolution: A multi-objective optimization perspective.
IEEE TIP, 2024. 3
[37] Shangquan Sun, Wenqi Ren, Xinwei Gao, Rui Wang, and
Xiaochun Cao. Restoring images in adverse weather condi-
tions via histogram transformer. InECCV, 2024. 8
[38] Jeya Maria Jose Valanarasu, Rajeev Yasarla, and Vishal M
Patel. Transweather: Transformer-based restoration of im-
ages degraded by adverse weather conditions. InCVPR,
2022. 8
[39] Hong Wang, Qi Xie, Qian Zhao, and Deyu Meng. A model-
driven deep neural network for single image rain removal. In
CVPR, 2020. 5, 6, 7
[40] Tianyu Wang, Xin Yang, Ke Xu, Shaozhe Chen, Qiang
Zhang, and Rynson WH Lau. Spatial attentive single-image
deraining with a high quality real rain dataset. InCVPR,
2019. 2, 5
[41] Zhenwei Wang, Tengfei Wang, Zexin He, Gerhard Hancke,
Ziwei Liu, and Rynson WH Lau. Phidias: A generative
model for creating 3d content from text, image, and 3d con-
ditions with reference-augmented diffusion.arXiv preprint
arXiv:2409.11406, 2024. 2
[42] Qiang Wen, Yue Wu, and Qifeng Chen. Video waterdrop re-
moval via spatio-temporal fusion in driving scenes. InICRA,
2023. 5
[43] Qinzhuo Wu, Weikai Xu, Wei Liu, Tao Tan, Jianfeng Liu,
Ang Li, Jian Luan, Bin Wang, and Shuo Shang. Mobilevlm:
A vision-language model for better intra-and inter-ui under-
standing.arXiv preprint arXiv:2409.14818, 2024. 4
[44] Jie Xiao, Xueyang Fu, Aiping Liu, Feng Wu, and Zheng-Jun
Zha. Image de-raining transformer.IEEE TPAMI, 2022. 5,
6, 7
[45] Hujie Yan. Towards universal rainy image restoration:
Benchmark and baseline. InAAAI, 2025. 1, 2, 5, 6, 7
[46] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo
Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang,
Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei
Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin,
Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei
Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin,
Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren,
Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical
report.arXiv preprint arXiv:2412.15115, 2024. 3
[47] Qianfeng Yang, Xiang Chen, Pengpeng Li, Qiyuan Guan,
Guiyue Jin, and Jiyu Jin. Rethinking rainy 3d scene recon-
struction via perspective transforming and brightness tuning.
arXiv preprint arXiv:2511.06734, 2025. 2
[48] Wenhan Yang, Robby T Tan, Jiashi Feng, Jiaying Liu, Zong-
ming Guo, and Shuicheng Yan. Deep joint rain detection and
removal from a single image. InCVPR, 2017. 1, 2, 5
[49] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine,
Karol Hausman, and Chelsea Finn. Gradient surgery for
multi-task learning.NeurIPS, 2020. 3
[50] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar
Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling
Shao. Multi-stage progressive image restoration. InCVPR,
2021. 5, 6, 7
[51] Syed Waqas Zamir, Aditya Arora, Salman Khan, Mu-
nawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang.
Restormer: Efficient transformer for high-resolution image
restoration. InCVPR, 2022. 5, 6, 7
[52] Fan Zhang, Shaodi You, Yu Li, and Ying Fu. Gtav-nightrain:
Photometric realistic large-scale dataset for night-time rain
streak removal.arXiv preprint arXiv:2210.04708, 2022. 1,
2, 5
[53] Fan Zhang, Shaodi You, Yu Li, and Ying Fu. Learning rain
location prior for nighttime deraining. InICCV, 2023. 5, 6,
7
[54] Zilun Zhang, Haozhan Shen, Tiancheng Zhao, Zian Guan,
Bin Chen, Yuhao Wang, Xu Jia, Yuxiang Cai, Yongheng
Shang, and Jianwei Yin. Imagerag: Enhancing ultra high
resolution remote sensing imagery analysis with imagerag.
arXiv preprint arXiv:2411.07688, 2024. 2
[55] Man Zhou, Jie Xiao, Yifan Chang, Xueyang Fu, Aiping Liu,
Jinshan Pan, and Zheng-Jun Zha. Image de-raining via con-
tinual learning. InCVPR, 2021. 7, 8
[56] Yurui Zhu, Tianyu Wang, Xueyang Fu, Xuanyu Yang, Xin
Guo, Jifeng Dai, Yu Qiao, and Xiaowei Hu. Learn-
ing weather-general and weather-specific features for image
restoration under multiple adverse weather conditions. In
CVPR, 2023. 8
10