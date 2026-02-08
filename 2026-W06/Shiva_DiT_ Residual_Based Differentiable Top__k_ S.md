# Shiva-DiT: Residual-Based Differentiable Top-$k$ Selection for Efficient Diffusion Transformers

**Authors**: Jiaji Zhang, Hailiang Zhao, Guoxuan Zhu, Ruichao Sun, Jiaju Wu, Xinkui Zhao, Hanlin Tang, Weiyi Lu, Kan Liu, Tao Lan, Lin Qu, Shuiguang Deng

**Published**: 2026-02-05 12:42:22

**PDF URL**: [https://arxiv.org/pdf/2602.05605v1](https://arxiv.org/pdf/2602.05605v1)

## Abstract
Diffusion Transformers (DiTs) incur prohibitive computational costs due to the quadratic scaling of self-attention. Existing pruning methods fail to simultaneously satisfy differentiability, efficiency, and the strict static budgets required for hardware overhead. To address this, we propose Shiva-DiT, which effectively reconciles these conflicting requirements via Residual-Based Differentiable Top-$k$ Selection. By leveraging a residual-aware straight-through estimator, our method enforces deterministic token counts for static compilation while preserving end-to-end learnability through residual gradient estimation. Furthermore, we introduce a Context-Aware Router and Adaptive Ratio Policy to autonomously learn an adaptive pruning schedule. Experiments on mainstream models, including SD3.5, demonstrate that Shiva-DiT establishes a new Pareto frontier, achieving a 1.54$\times$ wall-clock speedup with superior fidelity compared to existing baselines, effectively eliminating ragged tensor overheads.

## Full Text


<!-- PDF content starts -->

Shiva-DiT: Residual-Based Differentiable Top-kSelection
for Efficient Diffusion Transformers
Jiaji Zhang1Hailiang Zhao1Guoxuan Zhu2Ruichao Sun1Jiaju Wu3Xinkui Zhao1Hanlin Tang2
Weiyi Lu2Kan Liu2Tao Lan2Lin Qu2Shuiguang Deng1
Abstract
Diffusion Transformers (DiTs) incur prohibitive
computational costs due to the quadratic scaling
of self-attention. Existing pruning methods fail to
simultaneously satisfy differentiability, efficiency,
and the strict static budgets required for hardware
overhead. To address this, we proposeShiva-
DiT, which effectively reconciles these conflicting
requirements viaResidual-Based Differentiable
Top-kSelection. By leveraging a residual-aware
straight-through estimator, our method enforces
deterministic token counts for static compilation
while preserving end-to-end learnability through
residual gradient estimation. Furthermore, we
introduce a Context-Aware Router and Adaptive
Ratio Policy to autonomously learn an adaptive
pruning schedule. Experiments on mainstream
models, including SD3.5, demonstrate that Shiva-
DiT establishes a new Pareto frontier, achieving a
1.54×wall-clock speedup with superior fidelity
compared to existing baselines, effectively elimi-
nating ragged tensor overheads.
1. Introduction
The advent of Diffusion Transformers (DiTs) marks a
paradigm shift in generative modeling (Peebles & Xie, 2023;
Chen et al., 2025; Esser et al., 2024; Black Forest Labs,
2024). By scaling standard architectures, DiTs are tran-
scending visual fidelity to emerge as general-purpose world
simulators that model physical dynamics (Brooks et al.,
2024; Gemini Team, Google, 2025). However, this perfor-
mance comes at a prohibitive computational cost. Unlike
U-Nets (Ronneberger et al., 2015), which operate on com-
pressed feature maps, DiTs process flattened sequences of
spatial tokens, where the self-attention mechanism scales
1College of Computer Science and Technology, Zhejiang Uni-
versity2AIOS, Alibaba Group3Nanyang Technological University.
Correspondence to: Hailiang Zhao <hliangzhao@zju.edu.cn >,
Shuiguang Deng<dengsg@zju.edu.cn>.
Preprint. February 6, 2026.quadratically with resolution. For a standard 1024 ×1024
image, processing 4096 tokens per layer across dozens of
timesteps creates a massive latency bottleneck, hindering
real-time deployment.
To mitigate this, dynamic token pruning has emerged as a
critical direction. However, designing an efficient pruning
mechanism for DiTs requires balancing three conflicting
objectives, which we term theTrilemma of Sparse Learning
(see Figure 1a):1) Differentiability, to ensure the pol-
icy is end-to-end learnable for capturing complex diffusion
semantics;2) Efficiency, requiring minimal training over-
head to ensure scalability on high-resolution inputs; and
3) Strict Budget, enforcing a deterministic top- kcount to
guarantee static tensor shapes. This final constraint is essen-
tial for hardware-friendly static compilation (e.g., CUDA
Graphs) and efficient gather /scatter primitives, effec-
tively avoiding the latency overhead of ragged tensors.
Existing approaches typically compromise on at least one
of these vertices. Heuristic methods (Bolya & Hoffman,
2023; Wu et al., 2025; Lu et al., 2025; Fang et al., 2025)
ensure hardware compatibility but lack learnability, rely-
ing on static rules that fail to capture semantic nuances.
Conversely, learnable masking methods inspired by Gumbel-
Softmax (Zhao et al., 2025) achieve differentiability but treat
pruning as a thresholding problem. This results inragged
tensorswith variable sequence lengths, which preclude the
use of static graph compilation and introduce significant
kernel launch overheads. While DiffCR (You et al., 2025)
attempts to bridge this gap by learning a fixed budget via
interpolation, it incurs prohibitive training overhead, neces-
sitating dual forward passes to estimate gradients, which
effectively doubles the training cost. Similarly, soft-ranking
operators like NeuralSort (Grover et al., 2019) offer dif-
ferentiable top- k, but constructing the required N×N
permutation matrix during training incurs a quadratic mem-
ory footprint, rendering them intractable for high-resolution
vision tasks.
In this paper, we introduceShiva-DiT, a framework that
resolves the efficiency-quality trilemma viaResidual-Based
Differentiable Sorting. By leveraging the backbone’s resid-
ual nature, we formulate a gradient estimator that models
1arXiv:2602.05605v1  [cs.LG]  5 Feb 2026

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
(a)The Trilemma of Sparse Learning.
 (b)Shiva-DiT Workflow.
Figure 1.Overview of Shiva-DiT.(a) Shiva simultaneously resolves the sparse learning trilemma: Differentiability, Efficiency, and Strict
Budget. (b) We inject a lightweightImportance RouterandAdaptive Ratio Policyinto the frozen backbone. Utilizing differentiable
sorting, Shiva ensures learnability while maintaining static tensor shapes for hardware efficiency.
pruning as a learnable selection between the active block
and the identity skip connection. This design enables the
end-to-end optimization of a deterministic token budget k,
enforcing hardware-friendly static tensor shapes. To fully
exploit this, we incorporate aContext-Aware Routerusing a
pairwise layer-sharing strategy and anAdaptive Ratio Policy
that learns a decoupled spatiotemporal schedule, thereby
automatically allocating computation to information-dense
generative stages without manual heuristics.
Our contributions are summarized as follows:(1) We
propose aResidual-Based Differentiable Sortingmecha-
nism that formulates token pruning as a differentiable top-
kselection within residual blocks. By deriving a novel
gradient estimator, this design enables the end-to-end op-
timization of a learnable budget k, providing the mathe-
matical foundation for adaptive pruning while maintaining
a hardware-friendly deterministic execution flow. (2) We
introduce aContext-Aware Routerand anAdaptive Ratio
Policyto achieve intelligent computation allocation. This ar-
chitecture autonomously learns a spatiotemporally adaptive
pruning schedule by capturing local feature consistency and
global diffusion dynamics, eliminating the reliance on man-
ual heuristics. (3) Extensive experiments on mainstream
DiT models, including SD3.5, demonstrate that Shiva-DiT
establishes a new Pareto frontier. Our method achieves a
1.54×wall-clock speedup while maintaining superior gen-
erative fidelity compared to state-of-the-art baselines (Zhao
et al., 2025; Wu et al., 2025).
2. Related Work
2.1. Token Reduction in Vision Transformers
The imperative to mitigate the quadratic complexity of self-
attention originated in discriminative Vision Transformers
(ViTs) (Dosovitskiy et al., 2021). Pioneering works (Rao
et al., 2021; Marin et al., 2023; Bolya et al., 2023; Zhang
et al., 2024b; Wang et al., 2024a; Lee et al., 2024) exploredvarious strategies such as dynamic pruning, token pool-
ing, and bipartite matching to eliminate redundant tokens.
Notably, Token Fusion (Kim et al., 2024) bridges the gap
between pruning and merging, selecting operations based
on functional linearity and introducing MLERP to preserve
feature norms. While these methods established the ground-
work for efficiency, their design assumptions do not fully
align with the generative nature of DiTs (Peebles & Xie,
2023). However, these methods are tailored for classifica-
tion, where discarding background is harmless. In contrast,
diffusion models require dense prediction for pixel-level
generation. Therefore, applying aggressive pruning naively
to DiTs compromises spatial integrity and image fidelity.
2.2. Heuristic and Training-Free Acceleration
To adapt reduction techniques to diffusion, ToMeSD (Bolya
& Hoffman, 2023) extended token merging to Stable Dif-
fusion, introducing unmerge operations to preserve spatial
consistency. To better preserve semantic details, subse-
quent works incorporated advanced saliency metrics. AT-
EDM (Wang et al., 2024b) evaluates token importance us-
ing attention maps, while IBTM (Wu et al., 2025) lever-
ages classifier-free guidance (CFG) scores to protect high-
information tokens. Other approaches explore spectral or
optimization perspectives: FreqTS (Yang et al., 2025) priori-
tizes high-frequency components encoding structural details,
and ToMA (Lu et al., 2025) reformulates token selection as a
submodular optimization problem. Leveraging the temporal
dynamics, SDTM (Fang et al., 2025) utilizes the “structure-
then-detail” denoising prior, applying a hand-crafted sched-
ule to merge tokens aggressively in early structural stages.
While effective as post-training optimizations, such heuris-
tic policies are inherently static and may not optimally adapt
to complex redundancy patterns.
Kernel Incompatibility.A practical bottleneck for meth-
ods relying on explicit attention maps (e.g., ranking via
QKT(Wang et al., 2024b;a; Fang et al., 2025)) is their
2

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
conflict with hardware-efficient attention algorithms. State-
of-the-art implementations like FlashAttention (Dao et al.,
2022) employ tiling-based parallel algorithms that com-
pute attention scores block-by-block in fast on-chip SRAM.
Crucially, these algorithmsnever materializethe complete
N×N attention map in High-Bandwidth Memory (HBM).
Pruning algorithms that mandate access to the global atten-
tion map inevitably force a fallback to quadratic-memory
implementations, thereby negating the scalability and speed
benefits of modern attention kernels.
2.3. Temporal Feature Caching
Accelerating inference by exploiting feature correlation be-
tween adjacent timesteps has also gained traction. DiT-
FastAttn (Yuan et al., 2024) identifies redundancies across
spatial and temporal dimensions, proposing cross-timestep
attention sharing. ToCa (Zou et al., 2025) and DaTo (Zhang
et al., 2024a) introduce token-wise caching based on dy-
namics, updating only critical tokens while reusing cached
features for others. Similarly, CA-ToMe (Saghatchian et al.,
2025) and AsymRnR (Sun et al., 2025) extend merging re-
sults across steps. While effective for standard sampling
regimes, these strategies fundamentally rely on the assump-
tion oftemporal continuity. This dependency may limit
their applicability in accelerated inference scenarios, such as
few-step distillation, where the feature distribution evolves
rapidly between discrete timesteps. By avoiding reliance
on historical feature alignment, our approach maintains a
generalized design suitable for diverse sampling schedules.
2.4. Learning-Based Dynamic Architectures
To surpass heuristic limitations, recent research focuses on
learning-based dynamic pruning. DyDiT (Zhao et al., 2025)
trains lightweight routers but relies on threshold-based
masking, leading to unpredictable computational budgets.
SparseDiT (Chang et al., 2025) re-architects DiT with sparse
modules, though it requires invasive modifications. Ad-
dressing ratio learning, DiffCR (You et al., 2025) proposes
timestep-adaptive ratios via linear interpolation. However,
this approach necessitates computationally expensive dual
forward passes for differentiable ratio. In contrast, Shiva-
DiT integrates pruning decisions directly into a single pass,
reducing training cost by ∼35% compared to consistency-
based baselines, as detailed in Appendix B.4. By employing
a fully differentiable top- kmechanism, our framework en-
sures adeterministicbudget and providesfine-grainedtoken
gradients without the overhead or architectural invasiveness
of prior arts.3. Method
3.1. Preliminaries
Unified Perspective on Generative Models.We unify
Denoising Diffusion Probabilistic Models (DDPMs) (Ho
et al., 2020) and Flow Matching frameworks (Lipman et al.,
2023) under a common objective: learning a time-dependent
backboneF θ(xt, t, c)to minimize the regression error
L=E t,x0,x1
∥Fθ(xt, t, c)−y t∥2
.(1)
Here, the regression target ytvaries by architecture,
representing either the noise ϵin legacy models (e.g.,
SD1.5 (Rombach et al., 2022), SDXL (Podell et al., 2024))
or the velocity field vtin modern flow matching approaches
(e.g., Flux (Black Forest Labs, 2024), SD3 (Esser et al.,
2024)). SinceShiva-DiToptimizes the internal token pro-
cessing efficiency of Fθ, it is fundamentally agnostic to the
specific prediction target (noise vs. velocity) and compatible
with off-the-shelf samplers (e.g., DDIM (Song et al., 2021),
Euler (Karras et al., 2022) and DPM (Lu et al., 2022)).
Backbone Architecture.We focus on models where the
backbone Fθis instantiated as a Diffusion Transformer
(DiT) (Peebles & Xie, 2023). The input latent is flattened
into a sequence of Nspatial tokens X∈RN×D, where D
is the hidden dimension. These tokens are processed by L
transformer layers. A generic block at layer ℓupdates the
sequence via:
Yℓ=X ℓ+Attention(Norm(X ℓ, c, t)),
Xℓ+1=Y ℓ+FFN(Norm(Y ℓ, c, t)).(2)
Problem Formulation.Regardless of the prediction tar-
get, the computational bottleneck lies in the sequence length
N, as Attention (Vaswani et al., 2017) scales with O(N2)
and FFN with O(N) . Our goal is to introduce a differen-
tiable selection mechanism within these blocks to dynam-
ically identify a sparse subset Xsel∈Rk×D(k≪N ) for
computation, thereby accelerating the entire iterative gener-
ation process.
3.2. Residual-Based Differentiable Sorting
To overcome the non-differentiability of top- kselection
without the overhead of soft masking (Jang et al., 2017) or
the coarseness of binning (You et al., 2025), we propose
Residual-Based Differentiable Sorting. This mechanism
enables efficient hard selection during inference while main-
taining accurate gradient estimation for both token scores
and the budgetkduring training.
Forward Pass: Hard Selection.To strictly enforce com-
putational acceleration, the forward pass executes a deter-
ministic hard selection. Let X∈RN×Ddenote the input
3

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
tokens and s∈RNbe the corresponding importance scores
predicted by the Router. Given a target budget k, we identify
the index set of the top-kscores:
Itopk= argtopk(s, k).(3)
The token set is physically partitioned into selected ( Xsel)
and rejected (X rej) subsets via a gather operation:
Xsel={x i|i∈ I topk},X rej=X\X sel.(4)
Crucially, subsequent Transformer layers process only Xsel,
reducing FLOPs strictly proportional to the reduction ratio.
Backward Pass: Gradient Estimation.Since the dis-
crete indicator function I(i∈ I topk)has zero gradients,
we construct a differentiable surrogate graph. Inspired by
stochastic relaxation (Burges et al., 2005), we inject Gaus-
sian noise ϵ∼ N(0, σ2)into the scores ( ˜s=s+ϵ ) and
compute adescending soft rank ˜rivia pairwise sigmoid
comparisons (detailed in Algorithm 1):
˜ri(˜s) = 1 +X
j̸=iσ˜sj−˜si
τrank
,(5)
where τrankis the temperature. Crucially, while Eq. (5)
entails O(N2)comparisons, this operation is restrictedex-
clusivelyto the scalar scores ˜s∈RN×1. This explicitly
avoids the prohibitive O(N2D)overhead of feature-wise
soft-sorting methods like NeuralSort (Grover et al., 2019),
ensuring the memory footprint remains negligible compared
to the backbone’s attention maps.
To enable gradient flow to the learnable budget k, we formu-
late the inclusion score πias a continuous relaxation (Jang
et al., 2017) of the binary selection indicator:
πi(˜s, k) =σk−˜r i(˜s)
τsel
,(6)
where τselis a temperature parameter. This formulation
creates a fully differentiable path:∂πi
∂sand∂πi
∂kare non-
zero. Note that in this relaxed view, kacts as a continuous
threshold against the soft ranks.
Residual-Based Gradient Estimator.To propagate gra-
dients through the non-differentiable bottleneck, we employ
aResidual-Based Straight-Through Estimator(STE). Since
the deterministic hard selection restricts observation to a
single path per token, we define the gradient of the selection
probability πibased on the feature sensitivity of the actually
executed path. A detailed discussion on this estimator and
its gradient formulation is provided in Appendix A.2.
∂L
∂πi≈(
∇sel
xiL,x i
ifi∈ I topk
−D
∇rej
xiL,x iE
ifi /∈ I topk,(7)Algorithm 1Differentiable Soft Rank Calculation
1:Input:Perturbed Scores ˜S∈RB×N, Tempτ
2:Output:Soft RanksR∈RB×N
3:D←( ˜S:,1,:−˜S:,:,1)/τ ▷Broadcast differences
4:P←σ(D)·(1−I)▷Exclude self-comparison
5:R←1 +P
kP:,:,k ▷Sum over the last dimension
6:returnR
where ∇seland∇rejdenote the gradients of the loss Lwith
respect to the effective input features under the selected
and rejected paths, respectively. In practice, this piecewise
formulation is implemented efficiently via vectorized ten-
sor operations, where the negative sign for rejected tokens
acts as a corrective signal to penalize erroneous routing
decisions.
Comparison with Existing Paradigms.Our mechanism
addresses critical limitations in prior arts. First, regarding
Budget Controllability, Gumbel-based methods (Jang et al.,
2017) like DyDiT (Zhao et al., 2025) rely on prediction-
then-threshold masking. This paradigm suffers from bud-
get instability and results in ragged tensors that are incom-
patible with efficient batching or gather operations. In
contrast, our explicit top- kselection decouples budgeting
from ranking, guaranteeing exact adherence to budget k
and enabling efficient, hardware-friendly inference with
regular tensors. Second, regardingGradient Granularity,
interpolation-based methods like DiffCR (You et al., 2025)
necessitate expensive dual forward passes and derive coarse
gradients solely from bin slopes. Conversely, our differen-
tiable sorting enables efficientsingle-passestimation with
fine-grainedvisibility into individual token contributions
via Eq. 6, allowing for end-to-end optimization without
computational redundancy.
3.3. Context-Aware Token Importance Router
Token importance in diffusion is inherently contextual: a
spatial token’s redundancy depends on the denoising stage,
its semantic alignment with the text prompt, and its ab-
straction level within the network. Unlike prior heuristics
that rely on local metrics like L2 norms or attention scores,
we propose theShivaRouter. This lightweight, context-
conditioned scoring network efficiently captures these multi-
dimensional dependencies to predict token informativeness.
Architecture.To efficiently capture multi-dimensional
dependencies, the router predicts a scalar importance score
siby conditioning the local token feature xi∈RDon three
global signals: global semantics cand temporal embedding
t(both reused from the pre-trained backbone), alongside
a learnable layer identity embedding l. We project these
inputs into a low-dimensional bottleneck space ( d′= 64 )
4

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
and apply an additive fusion to minimize overhead:
hctx=Projt(t) +Projp(c) +Projl(l),(8)
si=wT·LayerNorm(σ(Projx(xi) +h ctx)),(9)
where Proj∗denote lightweight linear projections and σ
is the SiLU activation. This design acts as a conditional
shift, modulating the token’s latent representation based on
context. The layer embedding lis particularly crucial, as
it allows the router to distinguish nuances between specific
layers within a shared parameter group.
Locally Shared Parameters.To balance representation
capacity with parameter efficiency, we evaluate different
router sharing configurations across network depths. While
independent routers for each layer maximize flexibility, our
empirical results show thatPairwise Sharing(grouping ev-
ery two adjacent layers to share one router) provides the
optimal trade-off. This strategy effectively captures the local
feature consistency between consecutive blocks while sig-
nificantly reducing parameter overhead. Experiments con-
firm that this localized sharing maintains generative quality
more effectively than either layer-specific or globally-shared
routers.
3.4. Adaptive Ratio Policy
Determining optimal pruning ratios across layers and
timesteps is challenging. Prior methods often rely on static
heuristics (Chang et al., 2025) or discrete timestep buck-
ets (You et al., 2025), failing to capture the fine-grained
dynamics of the diffusion process. To address this, we in-
troduce aRatio Policy Network, a lightweight controller
that predicts a continuous, context-aware retention ratio
rt,l∈(0,1] . This ratio determines the discrete budget
k=⌊N·r t,l⌋(as discussed in Sec. 3.2), enabling precise
and continuous control over hardware-constrained token
counts throughout the generation.
Continuous Decoupled Modulation.We hypothesize
that the optimal sparsity profile can be decomposed into
a superposition of a static spatial prior and a global tem-
poral trend. Unlike multiplicative mechanisms such as
FiLM (Perez et al., 2018), which inherently assume that
temporal dynamics exert complex, layer-specific scaling
effects (i.e., different timesteps modulate different layers
non-uniformly), we argue that the redundancy trend across
timesteps is largely consistent across network depths. There-
fore, we adopt aDecoupled Additivearchitecture to explic-
itly treat spatial and temporal redundancies as independent,
additive factors.
Leteℓ∈Rdbe a learnable embedding for layer ℓ, and
et=MLP(t) be the projected timestep embedding. Thepolicy predicts the logit of the retention ratio via:
logit(r t,ℓ) = Φ time(et) + Φ layer(eℓ) +b anchor,(10)
rt,ℓ=σ(logit(r t,ℓ)),(11)
where Φdenotes lightweight MLPs and banchor is a learnable
scalar initialized to σ−1(Rtarget). This additive design not
only stabilizes optimization but also guarantees inference
efficiency. Crucially, since the policy network Φdepends
solely on the timestep tand layer index ℓ, and is independent
of dynamic input tokens or text prompts, the predicted ratios
are deterministic for a given checkpoint. Consequently,
during inference, the entire policy network is compiled into
a lightweightLook-Up Table(LUT), introducing negligible
computational overhead to the generation process.
Stabilized Budget Constraint.Optimizing a dynamic
policy to satisfy a global budget Rtarget involves a trade-
off between stability and flexibility. Prior methods like
DyDiT (Zhao et al., 2025) produce unpredictable costs via
thresholding, while DiffCR (You et al., 2025) penalizes mini-
batch deviations from the target ( ∥¯rbatch−R target∥2). While
effective for spatially static policies, this rigid constraint is
ill-suited for time-dependent pruning schedules. By forcing
every mini-batch to strictly adhere to the target, it precludes
the flexibility required to allocate varying computational
budgets across different diffusion timesteps.
To support time-dependent pruning, we introduce anEx-
ponential Moving Average (EMA) µglobal to decouple the
budget constraint from the sparse mini-batch sampling. At
iterationk, it tracks the long-term retention ratio via:
µ(k)
global=β·¯r(k)
batch+ (1−β)·µ(k−1)
global,(12)
where ¯rbatchis the globally synchronized batch mean. We
then formulate a proxy linear budget loss:
Lbudget =λ·¯r batch·sg[2·(µ global−R target)],(13)
where sg[·] denotes the stop-gradient operator, and µglobal
utilizes the updated moving average (from Eq. 12) to strictly
enforce the budget constraint based on the latest global statis-
tics. This formulation constructs a dynamic gradient field:
the term in brackets acts as a directional signal, pushing
¯rbatchdownwards only when the global average exceeds the
budget (and vice versa). This anchors the global constraint
while allowing the policy to explore optimal spatiotempo-
ral allocations. The stability of this mechanism is further
ensured byStratified Timestep Sampling(Sec. 3.5), which
minimizes the variance of ¯rbatchand provides a stable signal
for the EMA.
3.5. Training Objectives and Strategy
To effectively optimize the router’s discrete decisions and
the policy’s continuous constraints alongside the backbone,
5

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Vanilla
 ToMeSD
 IBTM
 DiffCR
 DyDiT
 Shiva-DiT
 Finetuned
Figure 2.Qualitative Comparison. Prompt:God Zeus wearing a golden oak leaves crown on his head, grey beard.By directing token
reduction to the background, Shiva-DiT preserves high-fidelity subject details (e.g., beard, weaving) comparable to Vanilla, whereas
competitors degrade the primary figure.
we employ a composite objective function and a multi-stage
training process.
Composite Loss with Distillation.The total objective
integrates diffusion reconstruction Ldiff, budget constraint
Lbudget , and distillation terms:
Ltotal=L diff+λbLbudget +λdLdistill.(14)
To ensure semantic consistency, we incorporate multi-scale
distillation from the frozen teacher model. Specifically,
we apply feature alignment losses every 4 transformer
blocks (Zhao et al., 2025) alongside a final output distil-
lation loss, defined as Normalized Feature Distillation (see
Eq.(31) in Appendix B.2), providing dense supervision to
guide the router’s optimization.
Staged Training Protocol.Directly optimizing all compo-
nents from a cold start can lead to instability. We therefore
adopt a progressive three-stage strategy, consisting ofRouter
Warmup,Policy Learning, andJoint Tuning, to sequentially
establish ranking capability, optimize the budget schedule,
and recover generation fidelity. Detailed configurations for
each stage are provided in Appendix B.2.
Stratified Sampling Strategy.To stabilize the joint opti-
mization of the modules, we introduce aStratified Sampling
mechanism to rectify the high variance of standard uniform
sampling.
1) Stratified Timestep Sampling.Standard sampling t∼
U(0, T) often leads to temporal clustering within a mini-
batch, destabilizing the budget controller. We instead en-
force uniform coverage by partitioning the domain [0, T]
intoBequi-width intervals for a global batch size B, draw-
ing one samplet ifrom each:
ti∼ Ui·T
B,(i+ 1)·T
B
, i∈ {0, . . . , B−1}.(15)
As detailed in Appendix A.4, this ensures ¯rbatchacts as a
low-variance estimator of the global expectation, providing
a stable gradient signal for optimization.2) Stratified Ratio Sampling.Primarily utilized during the
router warmup stage, this strategy forces the router to learn
a robust global ranking capability across the full sparsity
spectrum before the policy network is activated. Specifically,
we partition the batch into Kgroups, assigning each a target
rksampled from disjoint intervals spanning [rmin, rmax].
This prevents the router from overfitting to local decision
boundaries at fixed ratios, ensuring compatibility with the
subsequent dynamic schedule.
4. Experiments
4.1. Main Results
Experimental Settings.We evaluate Shiva-DiT on SD3.5-
Medium (Esser et al., 2024), Flux.1-dev (Black Forest Labs,
2024), and PixArt- Σ(Chen et al., 2025) using PEFT (Man-
grulkar et al., 2022) on the MJHQ-30K (Li et al., 2024)
dataset. Training is conducted on 8 ×NVIDIA H200 GPUs
(30–50 hours), while inference metrics are measured on
an NVIDIA RTX 4090. We compare against two cate-
gories of baselines: (1)Training-free methods, includ-
ing ToMeSD (Bolya & Hoffman, 2023), ToFu (Kim et al.,
2024), SDTM (Fang et al., 2025), ToMA (Lu et al., 2025)
and IBTM (Wu et al., 2025); (2)Training-based methods,
including SparseDiT (Chang et al., 2025), DiffCR (You
et al., 2025) and DyDiT (Zhao et al., 2025). Note that we
exclude NeuralSort (Grover et al., 2019) because it requires
explicit multiplication with an N×N permutation matrix,
incurring a computational complexity of O(N2D)and pro-
hibitive memory costs. This heavy overhead effectively
negates the potential speedup from token reduction, making
it unsuitable for our acceleration target.
For fairness, all training-based baselines are re-implemented
within our unified framework using the same dataset. Cru-
cially, we utilize their official training objectives and con-
figurations (e.g., distillation for DyDiT (Zhao et al., 2025),
standard MSE for DiffCR (You et al., 2025)) to ensure their
optimal performance, while Shiva-DiT follows the proposed
staged protocol. Detailed configurations and generalization
results on Flux.1-dev (Black Forest Labs, 2024) and PixArt-
Σ(Chen et al., 2025) are provided in Appendix B.2 and B.6.
6

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Table 1.Quantitative results on SD3.5 ( 1024×1024 , 50 steps, measured on RTX 4090). Metrics include IR (ImageReward), CLIP (CLIP
Score), and IQA (CLIP-IQA).Finetuneddenotes the LoRA-based model serving as the quality upper bound (sharing the same inference
cost asVanilla). Our methods achieve the superior trade-off between efficiency and fidelity.
MethodsImage Quality Metrics Efficiency Metrics
FID↓IR↑CLIP↑IQA↑FLOPs (T) Latency (ms) Speedup∆FLOPs (%)
Vanilla 16.86 0.9385 30.89 0.456711.80 7659.9 1.00×0.0Finetuned 13.22 1.0179 31.18 0.4743
ToMeSD (2023) 40.22 0.6727 30.88 0.4632 8.98 7352.9 1.04×23.9
ToFu (2024) 47.43 0.4157 30.36 0.4655 8.98 7296.7 1.05×23.9
SDTM (2025) 47.08 0.3257 29.96 0.4705 8.83 7180.6 1.07×25.2
ToMA (2025) 30.06 0.7033 30.83 0.4633 8.23 7406.5 1.03×30.3
IBTM (2025) 19.96 0.8610 30.88 0.4695 10.18 7526.6 1.02×13.7
DiffCR (2025) 17.51 0.7884 30.81 0.4941 9.93 6709.8 1.14×15.8
DyDiT (2025) 15.29 0.7844 30.65 0.4824 9.42 6553.9 1.17×20.2
SparseDiT (2025) 25.71 0.7751 30.56 0.4795 8.95 6211.3 1.24×24.2
Shiva-80% 13.83 0.8915 31.10 0.5051 8.92 6112.6 1.25× 24.4
Shiva-60% 16.42 0.9974 31.35 0.4952 6.96 4989.0 1.54× 41.0
Main Results on SD3.5.Table 1 and Figure 2 demon-
strate Shiva-DiT’s superior trade-off between fidelity and
latency. Specifically, Shiva-80% achieves the best FID
(13.83) among all acceleration methods and the highest
IQA (0.5051), effectively establishing a new Pareto frontier
closely matching the finetuned upper bound. Notably, Shiva-
60% secures a 1.54× wall-clock speedup while maintain-
ing a CLIP score ( 31.35 ) that surpasses even the finetuned
baseline. We attribute this qualitative gain to the selective
removal of redundant background tokens, which effectively
acts as a spatial attention mechanism, forcing the model to
allocate generation capacity to salient regions.
4500 5000 5500 6000 6500 7000 7500 8000
Inference Latency (ms)1020304050FID
Vanilla
FinetunedToMeSDToFu
SDTM
ToMA
IBTM DiffCR
DyDiTSparseDiT
Shiva-80%Shiva-60%
Figure 3.Efficiency-Quality Trade-off. Shiva-DiT pushes the
Pareto frontier towards the bottom-left, achieving a superior bal-
ance between inference latency and FID compared to baselines.
Hardware-Aware Efficiency.We investigate the gap be-
tween theoretical FLOPs reduction and practical accelera-
tion. As shown in Figure 3, optimization-heavy methods like
ToMA (Lu et al., 2025) suffer from high runtime overhead
due to iterative solving processes. Similarly, DyDiT (Zhao
et al., 2025) relies on ragged tensors and online mask com-putation, incurring significant memory re-allocation costs.
In contrast, Shiva-DiT enforces strictly static tensor shapes,
incurring negligible architectural overhead during the hard
forward pass beyond a lightweight top- kselection. This
design minimizes allocator churn and ensures seamless com-
patibility with static compilation pipelines (e.g., TensorRT).
Table 2.Router Configurations. We evaluate sharing strategies at
a fixed 0.6pruning ratio. Fidelity is measured via MSE (Mean
Squared Error of reconstructed features) and Sim (Cosine Simi-
larity between student and teacher features).Group-12achieves
the best alignment, outperforming the unstableMaxand layer-
independentSlimbaselines.
Config GroupingG d′Params MSE↓Sim↑
Tiny Global 1 64 0.3M 2.102 0.829
Shared Global 1 512 2.6M 1.254 0.862
Max Independent 24 512 64M 1.831 0.805
Slim Independent 24 64 8.0M 1.072 0.862
Group-4 6-Layers 4 128 2.7M 1.074 0.858
Group-8 3-Layers 8 64 2.7M 0.906 0.879
Group-12Pairwise 12 64 4.1M0.894 0.881
4.2. Ablation Study
Router Architecture Analysis.Table 2 examines the
trade-off between parameter sharing and generalization.
Global sharing ( G= 1 ) yields suboptimal reconstruction,
indicating that significant depth-wise feature heterogeneity
precludes a single unified policy. Conversely, the indepen-
dent baseline (Max) suffers from overfitting due to over-
parameterization. By exploiting local feature consistency,
our pairwise strategy achieves the lowest MSE ( 0.894 ). This
adjacent-layer sharing acts as a regularizer, mitigating do-
main shifts while preserving discriminative power. Figure 5
confirms this qualitatively, showing accurate isolation of
salient regions despite the reduced parameter count.
7

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
1000 2000 3000 4000 5000
Training Steps0.300.350.400.450.50Validation MSE LossFixed Ratio (r=0.6)
Ours (Adaptive, r=0.6)
(a)Validation Loss during training
0.51.0Spatial Ratio
1 3 5 7 9 11 13 15 17 19 21 23
Layer Depth1.0
0.5
0.0Timestep t/T
0.4 0.6 0.8
Temporal RatioKeep Ratio
0.2 1.0 (b)Learned pruning ratio schedule
Figure 4.Adaptive Policy Analysis. (a) The adaptive policy achieves consistently lower validation MSE, indicating superior convergence.
(b) The heatmap illustrates the pruning ratios across network layers ( y-axis) and diffusion timesteps ( x-axis), revealing a clear “structure-
first, detail-later” preference.
Impact of Adaptive Ratio Policy.We evaluate our pol-
icy against a Fixed Ratio baseline under a strict budget
(¯r≈0.6 ). As shown in Figure 4(a) and Table 3, our adap-
tive approach achieves lower validation MSE and superior
perceptual metrics by dynamically prioritizing information-
dense regions. Visualizing the learned schedule (Figure 4)
reveals a convergence to astructure-firstspatial strategy
(preserving shallow layers) and a progressive temporal pat-
tern: distinct from the uniform baseline, it allocates maxi-
mum budget ( ≈0.8 ) to high-frequency refinement stages for
detail recovery while aggressively pruning ( ≈0.4 ) during
early chaotic structural phases.
Table 3.Ratio Policy Analysis on MJHQ-30K. We compare our
Adaptive Policy with the Fixed baseline under an identical token
budget (¯r≈0.6) to ensure fair comparison.
Policy FID↓ImageReward↑CLIP Score↑CLIP IQA↑
Fixed Ratio 21.76 0.835 30.75 0.497
Adaptive 17.36 0.880 30.94 0.500
Analysis of Gradient Estimator and Budget Learning.
We highlight that a standard Straight-Through Estimator
(STE) is mathematically insufficient for our framework, as
it cannot propagate gradients to the budget parameter k
(since ∂I/∂k= 0 almost everywhere). Our Residual-Based
Estimator (Eq. (7)) is specifically designed to bridge this gap.
Therefore, the “Fixed Ratio” baseline (Table 3) effectively
serves as the ablation study for our learnable budget mech-
anism: without our specific gradient estimator, the model
effectively degenerates to a static pruning schedule. We
further validate the directional correctness of our estimator
via synthetic tasks in Appendix A.3, proving it successfully
guideskto the optimal sparsity (Figure 6a).
Limitations.Shiva-DiT relies on the assumption of spatial
redundancy, which weakens in high-density scenarios such
as dense text rendering or crowded scenes. As our current
implementation discards rejected tokens to strictly enforce
t  900
Input
 Layer 1
 Layer 8
 Layer 15
 Layer 23
t  700
 t  500
 t  100
0.0 0.2 0.4 0.6 0.8 1.0
Router Importance ScoreFigure 5.Visualization of Learned Masks (Group-12). The pair-
wise router effectively identifies semantic regions (e.g., foreground
objects) and texture-rich areas, demonstrating that locally shared
policies can accurately capture token importance.
the hardware budget, detail loss may occur when redundancy
is minimal. Future work could investigate fusion strategies
to aggregate information from rejected tokens into the active
set, rather than discarding them entirely.
5. Conclusion
Shiva-DiT bridges the gap between theoretical FLOPs reduc-
tion and practical hardware acceleration. By reformulating
token pruning as differentiable sorting within a residual
framework, we enforce static tensor constraints without
compromising generative capability. This work highlights
the value of hardware-algorithm co-design, offering a scal-
able and efficient path for deploying large-scale diffusion
backbones in real-time scenarios.
8

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
References
Black Forest Labs. Flux.1, 2024. URL https://blac
kforestlabs.ai/. Accessed: 2025-05-05.
Bolya, D. and Hoffman, J. Token merging for fast stable
diffusion.CVPR Workshop on Efficient Deep Learning
for Computer Vision, 2023.
Bolya, D., Fu, C.-Y ., Dai, X., Zhang, P., Feichtenhofer, C.,
and Hoffman, J. Token merging: Your ViT but faster. In
International Conference on Learning Representations,
2023.
Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y .,
Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman,
E., Ng, C., Wang, R., and Ramesh, A. Video generation
models as world simulators. 2024. URL https://op
enai.com/research/video-generation-m
odels-as-world-simulators.
Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds,
M., Hamilton, N., and Hullender, G. Learning to rank
using gradient descent. InProceedings of the 22nd In-
ternational Conference on Machine Learning, ICML
’05, pp. 89–96, New York, NY , USA, 2005. Associ-
ation for Computing Machinery. ISBN 1595931805.
doi: 10.1145/1102351.1102363. URL https:
//doi.org/10.1145/1102351.1102363.
Chang, S., WANG, P., Tang, J., Wang, F., and Yang, Y .
Sparsedit: Token sparsification for efficient diffusion
transformer. InThe Thirty-ninth Annual Conference
on Neural Information Processing Systems, 2025. URL
https://openreview.net/forum?id=jTBx
yQempF.
Chen, J., Ge, C., Xie, E., Wu, Y ., Yao, L., Ren, X., Wang,
Z., Luo, P., Lu, H., and Li, Z. Pixart- σ: Weak-to-strong
training of diffusion transformer for 4k text-to-image
generation. In Leonardis, A., Ricci, E., Roth, S., Rus-
sakovsky, O., Sattler, T., and Varol, G. (eds.),Computer
Vision – ECCV 2024, pp. 74–91, Cham, 2025. Springer
Nature Switzerland. ISBN 978-3-031-73411-3.
Dao, T., Fu, D. Y ., Ermon, S., Rudra, A., and R ´e, C. FlashAt-
tention: Fast and memory-efficient exact attention with
IO-awareness. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS), 2022.Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,
M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N.
An image is worth 16x16 words: Transformers for image
recognition at scale.ICLR, 2021.
Esser, P., Kulal, S., Blattmann, A., Entezari, R., M ¨uller,
J., Saini, H., Levi, Y ., Lorenz, D., Sauer, A., Boesel, F.,
Podell, D., Dockhorn, T., English, Z., and Rombach, R.
Scaling rectified flow transformers for high-resolution
image synthesis. InProceedings of the 41st International
Conference on Machine Learning, ICML’24. JMLR.org,
2024.
Fang, H., Tang, S., Cao, J., Zhang, E., Tang, F., and Lee,
T. Attend to not attended: Structure-then-detail token
merging for post-training dit acceleration. InIEEE/CVF
Conference on Computer Vision and Pattern Recognition,
CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp.
18083–18092. Computer Vision Foundation / IEEE, 2025.
doi: 10.1109/CVPR52734.2025.01685.
Gemini Team, Google. Gemini 3 technical report. Technical
report, Google DeepMind, 2025. URL https://de
epmind.google/technologies/gemini/.
Grover, A., Wang, E., Zweig, A., and Ermon, S. Stochastic
optimization of sorting networks via continuous relax-
ations. InInternational Conference on Learning Repre-
sentations, 2019. URL https://openreview.net
/forum?id=H1eSS3CcKX.
Ho, J. and Salimans, T. Classifier-free diffusion guidance,
2022. URL https://arxiv.org/abs/2207.1
2598.
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-
bilistic models. InProceedings of the 34th International
Conference on Neural Information Processing Systems,
NIPS ’20, Red Hook, NY , USA, 2020. Curran Associates
Inc. ISBN 9781713829546.
Jang, E., Gu, S., and Poole, B. Categorical reparameter-
ization with gumbel-softmax. InInternational Confer-
ence on Learning Representations, 2017. URL https:
//openreview.net/forum?id=rkE3y85ee.
Karras, T., Aittala, M., Laine, S., and Aila, T. Elucidating
the design space of diffusion-based generative models.
InProceedings of the 36th International Conference on
Neural Information Processing Systems, NIPS ’22, Red
Hook, NY , USA, 2022. Curran Associates Inc. ISBN
9781713871088.
Kim, M., Gao, S., Hsu, Y .-C., Shen, Y ., and Jin, H. Token
fusion: Bridging the gap between token pruning and
token merging. In2024 IEEE/CVF Winter Conference on
9

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Applications of Computer Vision (WACV), pp. 1372–1381,
2024. doi: 10.1109/WACV57701.2024.00141.
Kingma, D. P. and Ba, J. Adam: A method for stochastic
optimization. In Bengio, Y . and LeCun, Y . (eds.),3rd
International Conference on Learning Representations,
ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Confer-
ence Track Proceedings, 2015.
Lee, S.-H., Wang, J., Zhang, Z., Fan, D., and Li, X. Video
token merging for long-form video understanding. In
Proceedings of the 38th International Conference on
Neural Information Processing Systems, NIPS ’24, Red
Hook, NY , USA, 2024. Curran Associates Inc. ISBN
9798331314385.
Li, D., Kamko, A., Akhgari, E., Sabet, A., Xu, L., and Doshi,
S. Playground v2.5: Three insights towards enhancing
aesthetic quality in text-to-image generation, 2024. URL
https://arxiv.org/abs/2402.17245.
Lipman, Y ., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and
Le, M. Flow matching for generative modeling. InThe
Eleventh International Conference on Learning Repre-
sentations, 2023. URL https://openreview.net
/forum?id=PqvMRDCJT9t.
Liu, S.-Y ., Wang, C.-Y ., Yin, H., Molchanov, P., Wang,
Y .-C. F., Cheng, K.-T., and Chen, M.-H. Dora: weight-
decomposed low-rank adaptation. InProceedings of
the 41st International Conference on Machine Learning,
ICML’24. JMLR.org, 2024.
Loshchilov, I. and Hutter, F. Decoupled weight decay reg-
ularization. InInternational Conference on Learning
Representations, 2019. URL https://openreview
.net/forum?id=Bkg6RiCqY7.
Lu, C., Zhou, Y ., Bao, F., Chen, J., Li, C., and Zhu, J.
Dpm-solver: a fast ode solver for diffusion probabilistic
model sampling in around 10 steps. InProceedings of
the 36th International Conference on Neural Information
Processing Systems, NIPS ’22, Red Hook, NY , USA,
2022. Curran Associates Inc. ISBN 9781713871088.
Lu, W., Zheng, S., Xia, Y ., and Wang, S. ToMA: Token
merge with attention for diffusion models. InForty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum
?id=51l8tvuIxo.
Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y ., Paul,
S., Bossan, B., and Tietz, M. PEFT: State-of-the-art
parameter-efficient fine-tuning methods. https://gi
thub.com/huggingface/peft, 2022.Marin, D., Chang, J.-H. R., Ranjan, A., Prabhu, A., Raste-
gari, M., and Tuzel, O. Token pooling in vision transform-
ers for image classification. In2023 IEEE/CVF Winter
Conference on Applications of Computer Vision (WACV),
pp. 12–21, 2023. doi: 10.1109/WACV56688.2023.00010.
Peebles, W. and Xie, S. Scalable diffusion models with
transformers. In2023 IEEE/CVF International Confer-
ence on Computer Vision (ICCV), pp. 4172–4182, 2023.
doi: 10.1109/ICCV51070.2023.00387.
Perez, E., Strub, F., de Vries, H., Dumoulin, V ., and
Courville, A. Film: visual reasoning with a general condi-
tioning layer. InProceedings of the Thirty-Second AAAI
Conference on Artificial Intelligence and Thirtieth Inno-
vative Applications of Artificial Intelligence Conference
and Eighth AAAI Symposium on Educational Advances
in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18.
AAAI Press, 2018. ISBN 978-1-57735-800-8.
Podell, D., English, Z., Lacey, K., Blattmann, A., Dock-
horn, T., M ¨uller, J., Penna, J., and Rombach, R. SDXL:
Improving latent diffusion models for high-resolution
image synthesis. InThe Twelfth International Confer-
ence on Learning Representations, 2024. URL https:
//openreview.net/forum?id=di52zR8xgf.
Rao, Y ., Zhao, W., Liu, B., Lu, J., Zhou, J., and Hsieh, C.-J.
Dynamicvit: Efficient vision transformers with dynamic
token sparsification. InAdvances in Neural Information
Processing Systems (NeurIPS), 2021.
Robbins, H. and Monro, S. A stochastic approximation
method.The Annals of Mathematical Statistics, 22(3):
400–407, 1951. ISSN 00034851.
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and
Ommer, B. High-resolution image synthesis with la-
tent diffusion models. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 10684–10695, June 2022.
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolu-
tional networks for biomedical image segmentation. In
Navab, N., Hornegger, J., Wells, W. M., and Frangi, A. F.
(eds.),Medical Image Computing and Computer-Assisted
Intervention – MICCAI 2015, pp. 234–241, Cham, 2015.
Springer International Publishing. ISBN 978-3-319-
24574-4.
Saghatchian, O., Moghadam, A. G., and Nickabadi, A.
Cached adaptive token merging: Dynamic token reduc-
tion and redundant computation elimination in diffusion
model. 2025.
Song, J., Meng, C., and Ermon, S. Denoising diffusion
implicit models. InInternational Conference on Learning
10

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Representations, 2021. URL https://openreview
.net/forum?id=St1giarCHLP.
Sun, W., Tu, R.-C., Liao, J., Jin, Z., and Tao, D. Asymrnr:
Video diffusion transformers acceleration with asymmet-
ric reduction and restoration. InForty-second Interna-
tional Conference on Machine Learning, 2025. URL
https://openreview.net/forum?id=5PiZ
evq9fY.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention
is all you need. InProceedings of the 31st International
Conference on Neural Information Processing Systems,
NIPS’17, pp. 6000–6010, Red Hook, NY , USA, 2017.
Curran Associates Inc. ISBN 9781510860964.
von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert,
N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman,
W., Xu, Y ., Liu, S., and Wolf, T. Diffusers: State-of-the-
art diffusion models. https://github.com/hug
gingface/diffusers, 2022.
Wang, H., Dedhia, B., and Jha, N. K. Zero-tprune: Zero-
shot token pruning through leveraging of the attention
graph in pre-trained transformers. In2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 16070–16079, 2024a. doi: 10.1109/CVPR
52733.2024.01521.
Wang, H., Liu, D., Kang, Y ., Li, Y ., Lin, Z., Jha, N. K., and
Liu, Y . Attention-driven training-free efficiency enhance-
ment of diffusion models. In2024 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pp.
16080–16089, 2024b. doi: 10.1109/CVPR52733.2024.0
1522.
Wu, H., Xu, J., Le, H., and Samaras, D. Importance-based
token merging for efficient image and video generation,
2025. URL https://arxiv.org/abs/2411.1
6720.
Yang, X., Yang, Y ., Pang, H., Tian, A. X., and Li, L. Freqts:
frequency-aware token selection for accelerating diffu-
sion models. InProceedings of the Thirty-Ninth AAAI
Conference on Artificial Intelligence and Thirty-Seventh
Conference on Innovative Applications of Artificial Intelli-
gence and Fifteenth Symposium on Educational Advances
in Artificial Intelligence, AAAI’25/IAAI’25/EAAI’25.
AAAI Press, 2025. ISBN 978-1-57735-897-8. doi:
10.1609/aaai.v39i9.33008. URL https://doi.
org/10.1609/aaai.v39i9.33008.
You, H., Barnes, C., Zhou, Y ., Kang, Y ., Du, Z., Zhou,
W., Zhang, L., Nitzan, Y ., Liu, X., Lin, Z., Shecht-
man, E., Amirghodsi, S., and Lin, Y . C. Layer- andtimestep-adaptive differentiable token compression ra-
tios for efficient diffusion transformers. InIEEE/CVF
Conference on Computer Vision and Pattern Recognition,
CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp.
18072–18082. Computer Vision Foundation / IEEE, 2025.
doi: 10.1109/CVPR52734.2025.01684.
Yuan, Z., Zhang, H., Pu, L., Ning, X., Zhang, L., Zhao, T.,
Yan, S., Dai, G., and Wang, Y . DiTFastattn: Attention
compression for diffusion transformer models. InThe
Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. URL https://openrevi
ew.net/forum?id=51HQpkQy3t.
Zhang, E., Xiao, B., Tang, J., Ma, Q., Zou, C., Ning, X.,
Hu, X., and Zhang, L. Token pruning for caching better:
9 times acceleration on stable diffusion for free, 2024a.
URLhttps://arxiv.org/abs/2501.00375.
Zhang, Y ., Wei, L., and Freris, N. Synergistic patch pruning
for vision transformer: Unifying intra- & inter-layer patch
importance. InThe Twelfth International Conference on
Learning Representations, 2024b. URL https://op
enreview.net/forum?id=COO51g41Q4.
Zhao, W., Han, Y ., Tang, J., Wang, K., Song, Y ., Huang, G.,
Wang, F., and You, Y . Dynamic diffusion transformer.
InThe Thirteenth International Conference on Learning
Representations, 2025. URL https://openreview
.net/forum?id=taHwqSrbrb.
Zou, C., Liu, X., Liu, T., Huang, S., and Zhang, L. Ac-
celerating diffusion transformers with token-wise fea-
ture caching. InThe Thirteenth International Confer-
ence on Learning Representations, 2025. URL https:
//openreview.net/forum?id=yYZbZGo4ei.
11

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
A. Theoretical Derivations and Synthetic Validation
In this appendix, we provide formal derivations for the gradient estimators and validate the core mechanisms through
synthetic experiments. We detail the Jacobian of the soft ranking operation (Sec. A.1), the formulation of the residual-based
gradient estimator (Sec. A.2), and empirical validation on synthetic tasks (Sec. A.3).
A.1. Jacobian and Consistency of the Soft Ranking Matrix
The differentiability of our method hinges on the continuous mapping from scores to ranks. Here, we derive the Jacobian of
the soft ranking operation to elucidate how gradients propagate through the sorting mechanism.
Recall that thedescending soft rank˜r iof tokeniis defined based on pairwise comparisons with all other tokensj:
˜ri(s) = 1 +X
j̸=iσ(D ji),whereD ji=sj−si
τrank.(16)
In this descending formulation, Dji>0 implies sj> si, contributing to a larger (worse) rank index for token i. Let
πi=σ((k−˜r i)/τ sel)be the selection probability. To update the router scores s, we must compute the gradient of the rank
vector ˜rwith respect to the score vectors.
Letσ′(z) =σ(z)(1−σ(z)) denote the derivative of the sigmoid function. We define the pairwise gradient scalar as
δji≜σ′(Dji). The partial derivative of the rank˜r iwith respect to an arbitrary scores mis:
∂˜ri
∂sm=X
j̸=i∂σ(D ji)
∂sm=1
τrankX
j̸=iδji·∂(sj−si)
∂sm.(17)
We analyze the two distinct cases for the indexm:
Case 1: Self-Sensitivity ( m=i ).We consider the gradient of a token’s rank with respect to its own score. In the
summation overj,s iappears in every term as−s i:
∂˜ri
∂si=1
τrankX
j̸=iδji·(−1) =−1
τrankX
j̸=iσ′sj−si
τrank
.(18)
Since σ′>0, this diagonal term of the Jacobian is strictly negative. This mathematically confirms that increasing a token’s
scores iconsistently decreases (improves) its rank index˜r i.
Case 2: Cross-Sensitivity ( m̸=i ).We consider the gradient of token i’s rank with respect to another token m’s score.
The summation overjvanishes for all terms except wherej=m:
∂˜ri
∂sm=1
τrank·δmi·(1) =1
τrankσ′sm−si
τrank
.(19)
This off-diagonal term is strictly positive, indicating that increasing the score of a competitor token mincreases (worsens)
the rank of tokeni.
Global Gradient Flow.By combining these terms via the chain rule, the gradient of the loss Lwith respect to a specific
scores maggregates feedback from the entire sequence:
∂L
∂sm=NX
i=1∂L
∂πi·∂πi
∂˜ri|{z}
Selection·∂˜ri
∂sm|{z}
Ranking.(20)
This formulation highlights thecontext-awarenature of our gradient estimation: updating smnot only affects the probability
of token mbeing selected (via∂˜rm
∂sm) but also impacts the selection probabilities of all other tokens i(via∂˜ri
∂sm), thereby
encouraging the router to learn a global sorting policy rather than independent scoring.
12

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
A.2. Derivation of the Residual-Based Gradient Estimator
To enable end-to-end optimization of the router parameters despite the discrete hard selection, we employ a Straight-Through
Estimator (STE) based on a differentiable surrogate graph.
Surrogate Input-Gating Model.We define the surrogate effective inputs for the selected path ( xsel
i) and the rejected path
(xrej
i) as functions of the inclusion scoreπ i. This modelsπ ias a continuous gate modulating the input flow:
xsel
i=πi·xi,
xrej
i= (1−π i)·xi.(21)
Gradient Derivation.We derive the gradient of the loss Lwith respect to πivia the chain rule. By differentiating
Eq. 21, we observe that∂xsel
i
∂πi=xiand∂xrej
i
∂πi=−x i. Let∇seland∇rejdenote the backpropagated gradients∂L
∂xseland∂L
∂xrej,
respectively. The exact gradient for the surrogate model is derived as:
∂L
∂πi=∂L
∂xsel
i,∂xsel
i
∂πi
+*
∂L
∂xrej
i,∂xrej
i
∂πi+
=
∇sel
xiL,x i
+
∇rej
xiL,−x i
=
∇sel
xiL − ∇rej
xiL,x i
.(22)
Stochastic Single-Path Estimator.In the actual training computation, the deterministic top- kselection activates only one
path per token (either xselorxrej). Consequently, the simultaneous evaluation of both gradient terms in Eq. 22 is impossible.
To address this, we construct a stochastic estimatorˆg ithat utilizes only the gradient from the executed path:
ˆgi=∂L
∂πi≈(
∇sel
xiL,x i
ifi∈ I topk
−D
∇rej
xiL,x iE
ifi /∈ I topk(23)
The validity of this single-sample estimator relies on the stochastic nature of our training protocol. Specifically, we
incorporatestratified ratio sampling(varying kdynamically) andscore perturbation(injecting Gaussian noise ϵ) to prevent
tokens from locking into a fixed state. These mechanisms ensure that over the course of training, a token visits both selected
and rejected states, allowing the estimator to approximate the expected gradient direction via Monte Carlo sampling.
Gradient Flow to the Budget k.A key advantage of our differentiable sorting formulation is the explicit gradient flow to
the learnable budgetk. Applying the chain rule, the gradient forkaggregates the sensitivities of all tokens:
∂L
∂k=NX
i=1∂L
∂πi·∂πi
∂k.(24)
Since πiis a sigmoid function of k(Eq. 6), the term∂πi
∂kis strictly positive. Consequently, the update direction for kis
determined by the sum of∂L
∂πi. Intuitively, if the aggregate marginal gain of selecting tokens (positive ˆgi) outweighs the
penalty of rejecting them, the gradient ∇kLwill be positive, driving the model to increase the budget k. This allows the
model to dynamically learn the optimal number of active tokens required to minimize the task loss.
Differentiable Cost Control via k-Relaxation.Unlike traditional continuous relaxations (Jang et al., 2017) that primarily
focus on the differentiability of individual categorical selections, our formulation explicitly treats the budget kas a learnable
parameter embedded within the soft inclusion function πi(k). In standard Straight-Through Estimators (STE) or categorical
samplers, the number of selected elements is typically either a fixed hyperparameter or a result of independent thresholding.
By contrast, our method enables the model to end-to-end optimize the computational cost by propagating gradients directly
tokthrough the sorting-based relaxation. This mechanism effectively bridges the gap between differentiable sorting and
dynamic resource allocation, allowing the model to autonomously seek an optimal equilibrium between task performance
and sparsity penalty during training.
13

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Analysis of Bias and Variance.We analyze the properties of our estimator ˆgicompared to the ideal surrogate gradient g∗
i
(Eq. 22).
•Bias and Sign Consistency:While ˆgiis a single-sample approximation, it maintainssign consistencywith the ideal
gradient. If the active path significantly reduces loss (large ⟨∇sel,x⟩),ˆgiis positive, encouraging selection. Conversely,
if the residual path induces high error sensitivity (large ⟨∇rej,x⟩),ˆgibecomes negative (due to the sign inversion),
penalizing rejection. Thus, the estimator consistently drives the parameters towards the lower-loss path.
•Variance and Exploration:The estimator introduces high variance due to the binary nature of the observation.
However, in the context of discrete optimization, this variance acts as a regularization term, promoting exploration of
the combinatorial search space and preventing premature convergence to suboptimal subsets.
A.3. Validation on Synthetic Data
To empirically validate the theoretical properties of our gradient estimator and the dynamic budgeting mechanism, we
conduct controlled experiments on synthetic tasks.
Joint Learning of Scoring and Budgeting.To verify the capability of the Shiva router to simultaneously learn feature-
aware ranking and converge to the optimal budget, we constructed a synthetic dataset ( N= 100, D= 16 ) comprising 20
high-magnitude signal tokens (mean 10.0) and 80 noise tokens ( k∗= 20 ). We initialized the budget at k= 50 and employed
a dual-optimizer strategy, using Adam (Kingma & Ba, 2015) ( lr= 0.1 ) for the router and SGD (Robbins & Monro, 1951)
(lr= 2.0 ) for the budget k. The training process consisted of a 100-step warmup followed by a 700-step adaptation phase.
As illustrated in Figure 6(a), the router rapidly learns to identify signal tokens during the warmup. Subsequently, the budget
kautomatically decays and stabilizes around k≈18 . This equilibrium point reflects the optimization trade-off determined
by the sparsity penalty weight ( λ= 0.1 ), where the model retains approximately 95% of the signal tokens while effectively
filtering out the noise.
Gradient Consistency.We further investigated the quality of our residual-based gradient estimator by conducting a Monte
Carlo simulation with 1,000 independent trials. In each trial, we sampled random input features Xand logits to measure the
alignment between our estimated gradients and the ideal feature-aligned direction. Figure 6(b) reveals that the resulting
cosine similarity distribution is strictly positive with a mean of 0.825±0.043 . This strong alignment confirms that our
estimator provides reliable and consistent directional signals for optimization, despite the discrete nature of the selection
operation.
0 100 200 300 400 500 600 700 800
Training Steps0102030405060Budget Size (k)
Warmup AdaptationLearned Budget k
Score Accuracy
True Sparsity k*
0.00.20.40.60.81.0
Score Accuracy
(a)Evolution of budget and accuracy.
0.70 0.75 0.80 0.85 0.90 0.95
Cosine Similarity0246810Probability DensityTrials: 1000
Mean: 0.825
Std:  0.043 (b)Distribution of gradient similarity.
Figure 6.Validation on Synthetic Data. (a) Training dynamics showing the learnable budget k(blue) decaying from 50 and stabilizing
near the ground truth sparsity k∗= 20 (red dotted line), while the sorting accuracy (orange) remains robust ( ≈95% ). (b) Monte Carlo
analysis confirms a strong positive alignment (cosine similarity >0) between our estimated gradients and the ideal optimization direction.
14

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
A.4. Variance Reduction Analysis of Stratified Sampling
In the main text (Section 3.5), we employ Stratified Timestep Sampling to stabilize the budget estimation. Here, we provide
a theoretical justification for why this strategy yields a strictly lower variance estimator compared to standard Uniform
Sampling, specifically in the context of diffusion models.
Problem Formulation.Let r(t) : [0, T]→[0,1] denote the optimal pruning ratio at timestep t. In diffusion models, r(t)
exhibits significant temporal variation across the generation process. Our objective is to estimate the global expected pruning
ratioµover the entire diffusion process:µ=E t∼U[0,T] [r(t)].
Uniform Sampling Estimator.A standard mini-batch Buniconsists of Bindependent samples drawn uniformly, i.e.,
ti∼ U[0, T]. The Monte Carlo estimator isˆµ uni=1
BPB
i=1r(ti). The variance of this estimator is:
Var(ˆµ uni) =σ2
B,(25)
whereσ2=Var(r(t))represents the total variance of the pruning ratio across the entire timestep domain[0, T].
Stratified Sampling Estimator.We partition the domain [0, T] intoBdisjoint strata intervals S1, . . . , S B, each of length
T/B . We draw exactly one sample tjuniformly from each stratum j. The stratified estimator is ˆµstrat=1
BPB
j=1r(tj).
According to theLaw of Total Variance, the total variance σ2can be decomposed into the averagewithin-stratumvariance
and thebetween-stratumvariance:
σ2=1
BBX
j=1σ2
j
|{z}
Average Within-Stratum Var+1
BBX
j=1(µj−µ)2
|{z }
Between-Stratum Var,(26)
where σ2
jis the variance within stratum j, and µjis the expected value within stratum j. Crucially, the variance of the
Stratified Estimator is determined only by the within-stratum components:
Var(ˆµ strat) =1
B2BX
j=1σ2
j=1
B
1
BBX
j=1σ2
j
.(27)
The reduction in variance achieved by Stratified Sampling is explicitly given by the difference between the two estimators:
∆Var=Var(ˆµ uni)−Var(ˆµ strat) =1
B
1
BBX
j=1(µj−µ)2

| {z }
Between-Stratum Variance.(28)
Since the term inside the square brackets is a sum of squares, it is strictly non-negative. In DiTs, the pruning ratio r(t)
varies significantly across different timesteps, causing the stratum means µjto differ substantially from the global mean µ.
This makes theBetween-Stratum Varianceterm large. By eliminating this term, Stratified Sampling provides a significantly
lower-variance gradient estimate, effectively preventing the learnable budget from oscillating due to sampling noise.
15

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
B. More Experimental Results
B.1. Impact of Training Strategy on Ratio Generalization
To validate the robustness of our routing mechanism, we visualize the learned policies and score distributions under different
training strategies in Figure 7. A critical requirement for our framework is that the router must function effectively across a
wide dynamic range of sparsity ratios, as the subsequent Ratio Policy is designed to predict depth-specific keeping rates (e.g.,
retaining high ratios for shallow texture layers vs. low ratios for deep semantic layers). A router trained with a fixed ratio
(e.g., a constant 0.6) typically overfits to a specific sorting threshold, failing to generalize when the Ratio Policy demands a
different sparsity level. As shown in the histograms (Figure 7c vs. d), our Stratified Sampling strategy forces the router to
learn a more dispersed and discriminative score distribution. This ensures that the rankings remain robust and meaningful
regardless of the varying reduction targets imposed by the Ratio Policy, whereas the fixed baseline produces clustered scores
with limited differentiability.
t  900
Input
 Layer 1
 Layer 8
 Layer 15
 Layer 23
t  700
 t  500
 t  100
0.0 0.2 0.4 0.6 0.8 1.0
Router Importance Score
(a)Heatmap (Ours / Stratified)
t  900
Input
 Layer 1
 Layer 8
 Layer 15
 Layer 23
t  700
 t  500
 t  100
0.0 0.2 0.4 0.6 0.8 1.0
Router Importance Score (b)Heatmap (Fixed)
0123t  900
Density=0.39
Layer 1
=0.58
Layer 8
=0.41
Layer 15
=0.59
Layer 23
0.00.51.01.52.02.5t  700
Density=0.47
 =0.55
 =0.39
 =0.58
01234t  500
Density=0.55
 =0.56
 =0.36
 =0.57
0.0 0.2 0.4 0.6 0.8 1.0
Score0246810t  100
Density=0.68
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.64
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.35
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.48
(c)Histogram (Ours / Stratified)
051015t  900
Density=0.63
Layer 1
=0.79
Layer 8
=0.88
Layer 15
=0.68
Layer 23
051015t  700
Density=0.65
 =0.83
 =0.91
 =0.63
051015t  500
Density=0.65
 =0.76
 =0.92
 =0.62
0.0 0.2 0.4 0.6 0.8 1.0
Score051015t  100
Density=0.62
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.50
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.93
0.0 0.2 0.4 0.6 0.8 1.0
Score=0.61
 (d)Histogram (Fixed)
Figure 7.Comparison of Router Training Strategies: Top row (a, b) displays spatial importance heatmaps; Bottom row (c, d) shows the
corresponding score distributions. Training with our proposed Stratified Sampling (Left column) exposes the router to varying sparsity
levels, resulting in a more dispersed score distribution with a broader dynamic range. This indicates stronger discriminative power between
informative and non-informative tokens. In contrast, the Fixed Ratio baseline (Right column) optimizes for a single sorting threshold,
leading to a narrow, clustered distribution with limited differentiation.
16

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
B.2. Training Details
To ensure the stability of the coupled optimization problem involving the discrete router ranking, the continuous ratio
policy, and the generative backbone, we employ a staged training process. This curriculum learning strategy decouples the
optimization targets to allow each component to initialize properly before joint optimization.
Stage 1: Router Warmup.In the initial phase, theRatio Policynetwork is bypassed. To prevent the router from overfitting
to a single fixed sparsity level and to encourage it to learn a global ranking capability, we sample the target retention ratio
rfrom a stratified uniform distribution r∼ U(r min, rmax)as described in Sec. 3.5. The router is forced to adapt to these
varying random budgets, thereby learning to discriminate token importance across the entire sparsity spectrum.
Stage 2: Policy Learning.Once the router has converged, we activate theRatio Policynetwork. In this stage, the training
is driven by theUnbiased Budget Constraint Loss(Eq. (13)) to align the global expected FLOPs with the target. We employ
a global EMA estimator for the budget constraint with a momentum coefficient of β= 0.2 to ensure stability against
batch-wise variance.
Crucially, to prevent the policy from collapsing into a trivial solution (e.g., a flat uniform ratio) due to the strict budget
penalty, we introduce aLogit Noise Injectionmechanism. We add zero-mean Gaussian noise ϵ∼ N(0,1) to the output
logits of the policy network before the sigmoid activation:
rt,l=σ(logit(r t,l) +ϵ).(29)
This high-variance noise forces the policy to explore the boundaries of the ratio landscape (i.e., testing extreme sparsity
or density) during training. Importantly, this noise injection does not compromise convergence. Since the injected noise
is zero-mean, the perturbed gradient serves as an unbiased estimator of the true gradient. Optimization algorithms like
AdamW (Loshchilov & Hutter, 2019), which rely on moment estimation, effectively filter out the stochastic noise over time,
driving the deterministic policy parameters towards the optimal expected logits. The noise merely smooths the optimization
landscape to escape local minima without biasing the final solution.
Stage 3: Joint Tuning.In the final stage, the exploration noise is removed. We unfreeze the backbone adaptation
parameters, such as LoRA, and jointly fine-tune the Router, Policy, and Backbone. The objective is to minimize the diffusion
reconstruction loss while maintaining the established budget constraints. This stage allows the backbone to adapt to the
sparse token flow and recover any fine-grained details lost due to pruning.
Hyperparameters and Optimization.All models are trained using the AdamW (Loshchilov & Hutter, 2019) optimizer
with mixed-precision ( bfloat16 ) to ensure memory efficiency. We utilize a global effective batch size of 16 (micro-batch
size 1 per device). The learning rates are set to 3×10−4for the Router and 1×10−4for the Ratio Policy. For µ(k)
global in
Eq.(12), we set βto0.2. For efficient backbone adaptation, we employ DoRA (Liu et al., 2024) with a rank of r= 32 ,
applied to all linear projections within the Attention and FFN blocks. A linear warmup of 200 steps is applied at the
beginning of each training stage. To support Classifier-Free Guidance (Ho & Salimans, 2022) (CFG), we randomly drop
text captions with a probability of p= 0.1 . During inference, we use 50 sampling steps with CFG scales of 7.0, 3.5, and 4.5
for SD3.5, Flux, and PixArt, respectively.
Temperature Annealing with Normalized Sorting.In our implementation, we introduce a normalization factor N
(sequence length) to Eq. 6 to decouple the gradient scale from the resolution. The modified selection probability is formulated
as:
πi=σk−˜r i
τsel·N
.(30)
To bridge the gap between soft training and hard inference, we employ a deterministic annealing schedule: the ranking
temperature τrankdecays from 0.2, and the selection temperature τseldecays from 0.1, gradually hardening the probability
distribution over the course of training.
Normalized Feature Distillation.For intermediate layer distillation, we observed that standard MSE loss is sensitive to
the varying feature magnitudes across depths, often necessitating heuristically small loss weights in prior works (Zhao et al.,
17

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
2025). To stabilize optimization, we proposeNormalized Feature Distillation, which applies Layer Normalization before the
distance calculation:
Ldistill=X
lLayerNorm(hStudent
l )−LayerNorm(hTeacher
l )2
2.(31)
This standardization ensures that the distillation gradient is driven by semantic misalignment rather than numerical scale
differences. For comprehensive configuration files, please refer to the attached source code.
B.3. Implementation Details and Baseline Configurations
To ensure reproducibility and facilitate future research, we developed a unified and modular codebase based on the
diffusers library (von Platen et al., 2022). Below we detail the engineering framework, model-specific adaptations, and
baseline configurations.
Adaptive Ratio Policy Network.The policy network employs a decoupled additive architecture comprising two parallel
branches. TheLayer Branchmaps the layer index to a hidden embedding ( R256) via a learnable lookup table, followed by a
2-layer MLP (Linear →SiLU→Linear) to predict the spatial logit. Simultaneously, theTime Branchprocesses the sinusoidal
timestep embedding through a mirrored MLP structure with a hidden dimension of 256. The final retention ratio rt,lis
obtained by summing the logits from both branches with a learnable global bias parameter banchor . For training stability,
banchor is initialized to the inverse sigmoid of the target initial ratio (e.g., σ−1(0.6) ) to ensure the training starts from a valid
operating point.
Unified Hook-Based Framework.We implement a flexible,plug-and-playacceleration framework that abstracts the token
reduction logic from the backbone definition. Instead of invasively modifying the model source code, we utilize a centralized
Hook Managerto inject custom operations at critical computation stages of a DiT block, including pre/post-Attention,
pre/post-FFN, and block boundaries. This design allows our method (and baselines) to be seamlessly applied to diverse
architectures such as Stable Diffusion 3 (Esser et al., 2024) (SD3), PixArt- Σ(Chen et al., 2025), and Flux.1-dev (Black
Forest Labs, 2024), without specific adaptation code. Furthermore, this modular infrastructure decouples the pruning policy
from the model execution, enabling rapid prototyping and iteration of efficient diffusion algorithms in future work.
Model-Specific Adaptations.We address two critical implementation details regarding positional embeddings and
guidance strategies:
•Rotary Positional Embeddings (RoPE):Handling spatial information varies by architecture. For models like SD3
and PixArt, where absolute positional embeddings are added to the tokens prior to the transformer blocks, no additional
handling is required as the spatial information is carried within the token features. However, for models like Flux
which apply RoPE at every attention layer, simply discarding tokens breaks the geometric correspondence. In our
implementation, we gather the corresponding cosine and sine components of the RoPE cache based on the selected
token indices ( Itopk). We note that exploring more sophisticated, topology-aware RoPE selection strategies remains a
promising direction for future work.
•Guidance Consistency Strategy:During inference with Classifier-Free Guidance (CFG), the batch dimension is
typically doubled to process conditional ( c) and unconditional ( ∅) inputs simultaneously. Independent scoring for
these two batches can lead to index mismatch, where different spatial tokens are retained for the cond/uncond branches,
causing severe artifacts during the latents combination step. To resolve this, we enforce aUnified Scoring Policy. We
compute the logits for both branches but perform selection based on the element-wise maximum score:
sunified = max(s cond,suncond).(32)
This ensures that if a token is salient ineithercontext (e.g., an object mentioned in the prompt but present in the
empty-prompt background), it is preserved in both branches, guaranteeing alignment for the subsequent arithmetic
operations.
Baseline Reproduction and Configuration.To ensure a fair comparison, we integrated all baseline methods into our
unified framework. For open-source methods, we strictly followed their official implementations; for others, we reproduced
them based on the algorithm details provided in their respective papers. All hyperparameters were kept consistent with
18

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
the original literature unless otherwise specified. Crucially, for our method (Shiva-DiT), we implement aFirst-Block Skip
strategy: we explicitly bypass the token reduction in the very first DiT block. Empirically, we observed that the initial
layer is responsible for establishing the fundamental low-level feature space; pruning at this nascent stage leads to training
instability and degrades generation quality. Processing the full sequence in the first block stabilizes the importance scoring
for subsequent layers with negligible computational overhead.
B.4. Training Efficiency Verification
To substantiate the claims regarding training efficiency, we conduct a rigorous comparison of computational overhead
(throughput and latency) between Shiva-DiT, DiffCR (You et al., 2025), and the Vanilla baseline. All models are trained on
the MJHQ-30K dataset using 8×NVIDIA H200 GPUs with a global batch size of 16.
Table 4.Training cost comparison measured on 8 ×H200 GPUs with SD3.5. Shiva-DiT significantly outperforms DiffCR in training
speed, maintaining near-baseline efficiency. DiffCR suffers from high latency due to its dual-pass consistency requirement.
Method Throughput (it/s)↑Time/Iter (ms)↓Relative Cost
Vanilla 1.86 5381.00×
DiffCR (You et al., 2025) 1.13 8851.65×
Shiva (Ours) 1.73 5781.07×
As presented in Table 4, DiffCR incurs a severe training penalty ( 1.65× cost,∼50 min/epoch). This overhead primarily
stems from its ratio mechanism, which necessitates dual forward passes to linearly interpolate between different ratio bins.
In contrast, although Shiva-DiT introduces additional parameters for the Context-Aware Router and Adaptive Ratio Policy,
it maintains high throughput ( 1.73 it/s,∼35 min/epoch). The negligible overhead ( ∼7%) validates that our residual-based
single-pass design effectively bypasses the computational bottleneck of prior consistent pruning methods.
B.5. Additional Qualitative Results
We provide additional qualitative comparisons on SD3.5-Medium in Figures 8, 9, and 10. While existing baselines often
introduce structural distortions or lose fine-grained textures, our method maintains superior visual fidelity and semantic
alignment. Compared to both training-free and training-based approaches, Shiva-DiT preserves the structural integrity and
intricate details (e.g., textures and reflections) across diverse and complex prompts, closely matching the quality of the
Vanilla and Finetuned upper bounds.
Vanilla
 ToMeSD
 ToFu
 SDTM
 IBTM
 ToMA
DiffCR
 SparseDiT
 DyDiT
 Shiva-80
 Shiva-60
 Finetuned
Figure 8.Prompt:a delicate silver link bracelet with a small red seastar attached to it.
19

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Vanilla
 ToMeSD
 ToFu
 SDTM
 IBTM
 ToMA
DiffCR
 SparseDiT
 DyDiT
 Shiva-80
 Shiva-60
 Finetuned
Figure 9.Prompt:a ragdoll cat dressed in magnificent court attire, worn a crown, sparkling necklace, seated on a throne, surrounded by
plates of sweets, realistic direct sun lighting, ultra detailed, super realistic, digital photography.
Vanilla
 ToMeSD
 ToFu
 SDTM
 IBTM
 ToMA
DiffCR
 SparseDiT
 DyDiT
 Shiva-80
 Shiva-60
 Finetuned
Figure 10.Prompt:Design a superrealistic digital painting of a French Bulldog in a rainy urban scene, wearing a rain jacket. Use dim
and moody lighting with street lights and reflections on the pavement to create the perfect atmosphere. Make the rain jacket bright blue
and the background dark blue and grey to contrast. Emphasize the urban environment and the rain with a midshot of the dog walking
towards the camera.
20

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
B.6. Results on Additional Models
To validate the architectural generalization of Shiva-DiT, we extend our evaluation to Flux.1-dev(Black Forest Labs, 2024)
and PixArt- Σ(Chen et al., 2025), with quantitative results detailed in Table 5. We observe substantial acceleration ( 1.24× )
on the compute-dense Flux backbone. Conversely, gains on the lightweight PixArt- Σare marginal ( 1.04× ), as the reduced
FLOPs do not translate linearly to wall-clock speedup due to GPU under-saturation. Additional qualitative comparisons are
presented in Figures 11, 12, 13, 14, 15, 16, and 17.
Table 5.Generalization results on Flux.1-dev and PixArt- Σ(1024×1024 , 50 steps, RTX 4090). Note that while Shiva achieves a
significant speedup ( 1.25× ) on the compute-dense Flux.1-dev, the acceleration on the lightweight PixArt- Σis more modest ( 1.08× ). This
is primarily due to the compact backbone (0.6B) limiting GPU saturation; even scaling to batch size 8 only marginally improves speedup
to1.11×. Additionally, Shiva adopts a conservative pruning rate (∆FLOPs≈17%) for PixArt to strictly preserve generative quality.
(a) Flux-Dev
MethodQuality Efficiency
FID↓IR↑CLIP↑IQA↑FLOPs (T) Latency (ms) Speedup∆FLOPs (%)
Vanilla 14.25 0.93 30.56 0.5044.63 33691 1.00×0.0Finetuned 9.29 0.33 29.19 0.56
Shiva15.03 0.92 30.29 0.53 33.93 26891 1.25×24.0
(b) PixArt-Σ
MethodQuality Efficiency
FID↓IR↑CLIP↑IQA↑FLOPs (T) Latency (ms) Speedup∆FLOPs (%)
Vanilla 11.08 0.93 31.37 0.4612.97 5910 1.00×0.0Finetuned 9.63 0.97 31.32 0.48
Shiva11.84 0.72 30.63 0.47 10.74 5471 1.08×17.2
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 11.Prompt:Wolf in fantasy forest, moonlight with rain, inked style.
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 12.Prompt:Hyperdetailed photography, Mount Fuji behind cherry blossoms overlooking Saiko lake.
21

Shiva-DiT: Residual-Based Differentiable Top-kSelection for Efficient Diffusion Transformers
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 13.Prompt:Young woman in an openback summer dress, looking longingly, visible shoulders, by Henri Gervex, ´Edouard Manet,
JeanHonor ´e Fragonard, Alfons Mucha, high fantasy, cinematic lighting, romantic atmosphere, ultrarealistic, hyperrealistic, intricate
detail.
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 14.Prompt:1989 Studio Ghibli anime movie, mythical ocean beach san francisco.
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 15.Prompt:A Fox druid wearing blue colorful robes casting thunder Wave.
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 16.Prompt:Country road with mountains in the background at sunrise. realistic footage, ultra details, 4k.
Vanilla-Flux
 Shiva-Flux
 Finetuned-Flux
 Vanilla-Pixart
 Shiva-Pixart
 Finetuned-Pixart
Figure 17.Prompt:Plein air painting, palette knife, loose brushwork, slightly abstract, a thin creek spilling over rocks, drying grass field,
soft lighting, soft colors, beige, white, brown, serene, vintage.
22