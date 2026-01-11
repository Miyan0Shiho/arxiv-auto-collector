# RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion

**Authors**: Yu Huo, Siyu Zhang, Kun Zeng, Yuquan Lu, Cheng Yang, Yifu Guo, Xiaoying Tang

**Published**: 2026-01-06 19:27:32

**PDF URL**: [https://arxiv.org/pdf/2601.03378v1](https://arxiv.org/pdf/2601.03378v1)

## Abstract
Repository-level code completion benefits from retrieval-augmented generation (RAG). However, controlling cross-file evidence is difficult because chunk utility is often interaction-dependent: some snippets help only when paired with complementary context, while others harm decoding when they conflict. We propose RepoShapley, a coalition-aware context filtering framework supervised by Shapley-style marginal contributions. Our module ChunkShapley constructs offline labels by (i) single-chunk probing with teacher-forced likelihood to estimate signed, weighted effects, (ii) a surrogate game that captures saturation and interference, (iii) exact Shapley computation for small retrieval sets, and (iv) bounded post-verification that selects a decoding-optimal coalition using the frozen generator. We distill verified $KEEP$ or $DROP$ decisions and retrieval triggering into a single model via discrete control tokens. Experiments across benchmarks and backbones show that RepoShapley improves completion quality while reducing harmful context and unnecessary retrieval. Code: https://anonymous.4open.science/r/a7f3c9.

## Full Text


<!-- PDF content starts -->

REPOSHAPLEY: Shapley-Enhanced Context Filtering for Repository-Level
Code Completion
Yu Huo1,2,3 *, Siyu Zhang1,3*, Kun Zeng4, Yuquan Lu3, Cheng Yang5,
Yifu Guo4, Xiaoying Tang1,2,3†
1School of Science and Engineering, The Chinese University of Hong Kong (Shenzhen),
Longgang, Shenzhen, Guangdong, 518172, P.R. China
2The Shenzhen Institute of ArtificialIntelligence and Robotics for Society
3Guangdong Provincial Key Laboratory of Future Networks of Intelligence
4Sun Yat-sen University,5Hangzhou Dianzi University
yuhuo@link.cuhk.edu.cn
Abstract
Repository-level code completion benefits from
retrieval-augmented generation (RAG). How-
ever, controlling cross-file evidence is diffi-
cult because chunk utility is often interaction-
dependent: some snippets help only when
paired with complementary context, while oth-
ers harm decoding when they conflict. We pro-
pose REPOSHAPLEY, a coalition-aware con-
text filtering framework supervised by Shapley-
style marginal contributions. Our module
ChunkShapleyconstructs offline labels by (i)
single-chunk probing with teacher-forced like-
lihood to estimate signed, weighted effects, (ii)
a surrogate game that captures saturation and
interference, (iii) exact Shapley computation
for small retrieval sets, and (iv) bounded post-
verification that selects a decoding-optimal
coalition using the frozen generator. We dis-
till verified KEEP orDROP decisions and
retrieval triggering into a single model via dis-
crete control tokens. Experiments across bench-
marks and backbones show that REPOSHAP-
LEYimproves completion quality while re-
ducing harmful context and unnecessary re-
trieval. Code: https://anonymous.4open.
science/r/a7f3c9.
1 Introduction
Repository-level code completion requires resolv-
ing non-local dependencies across files, including
project-specific APIs, shared contracts, and invari-
ants (Jimenez et al., 2024; Ding et al., 2024b).
Retrieval-Augmented Generation (RAG) addresses
this setting by injecting cross-file evidence into
Code LMs (Lewis et al., 2020; Kang et al., 2024;
Shrivastava et al., 2023; Bairi et al., 2023). How-
ever, effective retrieval control remains a bottle-
neck. Under a fixed context budget, the model must
identify truly useful evidence from a noisy candi-
date pool, where many chunks are redundant and
*Equal contribution
†Corresponding author
Figure 1: Performance radar charts on StarCoder-Base-
7B and CodeLlama-13B. The plots display relative im-
provements over the No-Retrieve baseline (center). RE-
POSHAPLEYachieves SOTA performance across 11
tested metrics (As shown in table 1).
some are actively misleading (Ding et al., 2024a;
Zhang et al., 2023; Wei et al., 2025; Liu et al.,
2024a; Yoran et al.).
The core difficulty is that chunk utility is often
interaction-dependent. A snippet may appear un-
informative in isolation yet become decisive when
paired with complementary context, such as an
interface declaration together with its implemen-
tation. Conversely, a plausible chunk can degrade
generation when it co-occurs with conflicting evi-
dence, such as deprecated versus updated APIs (Shi
et al., 2023; Xu et al., 2024). Therefore, methods
that score candidates independently can misesti-
mate the utility of the multi-chunk context that is
actually consumed at test time (Khandelwal et al.,
2020; Yan et al., 2024; Bertsch et al., 2025).
To address this, we adopted a coalition-first ap-
proach. Retrieval control should be supervised by
signals that reflect how a chunk behaves within
a set, rather than in isolation. We introduce RE-
POSHAPLEY, a framework that learns to filter con-
text using Shapley-style marginal contributions.
Our approach has two stages. First, we propose
ChunkShapley, an offline labeling pipeline for
1arXiv:2601.03378v1  [cs.SE]  6 Jan 2026

Figure 2: Under the same input context and the exact same retrieved candidate chunks, CODEFILTER makes decisions
from independent per-chunk signals and can break under interaction effects, whereas REPOSHAPLEYperforms
coalition-aware filtering that more reliably removes high-score noise while preserving complementary evidence.
interaction-aware supervision. Considering that
computing Shapley values directly with the genera-
tor is prohibitive, we introduce a structured logistic
surrogate that can capture saturation and conflict
efficiently. We then apply a verification step to
ground the selected coalitions in the generator’s
actual decoding behavior. Second, we distill the
resulting coalition-derived labels into a single gen-
erator via discrete control tokens, which we call
REPOSHAPLEY. This distillation enables efficient,
interaction-aware retrieval control at inference time.
As shown in Figure 1, REPOSHAPLEYachieves
SOTA performance across benchmarks and met-
rics, supporting our motivation that coalition-aware
supervision is crucial for difficult cross-file com-
pletion. Our contributions are as follows:
•Coalition-aware supervision for context fil-
tering.We formulate context selection as a
cooperative game and use Shapley marginal
contributions to capture complementarity and
conflict beyond independent scoring.
•ChunkShapley: Practical shapley labeling
for chunk filtering.We combine single-
chunk probing with a structured surrogate util-
ity to compute exact Shapley values on small
retrieval sets ( K=10 ). We further select a ver-
ified coalition from a bounded candidate pool
under decoding-time metrics.•REPOSHAPLEY: distillation for online re-
trieval control.We distill verified keep and
drop decisions into discrete control tokens,
enabling a single model to decide when to
retrieve and which chunks to keep.
2 Related Work
Repository-Level RAG and Retrieval Control.
RAG mitigates non-local dependencies in code
completion by retrieving cross-file evidence (Lewis
et al., 2020; Izacard and Grave, 2021; Parvez et al.,
2021; Guu et al., 2020; Jiang et al., 2023; Mallen
et al., 2023; Yao and Fujita, 2024). Recent work
improves context quality via iterative retrieval (Gao
et al., 2023; Zhang et al., 2023; Shrivastava et al.,
2023; Zhang et al., 2025), structure-aware index-
ing, including dataflow or call graphs (Cheng et al.,
2024; Liu et al., 2024c), and dedicated bench-
marks (Ding et al., 2023; Liu et al., 2024b; Li
et al., 2025; Generation, 2024; Yang et al., 2025).
In parallel, retrieval control has received increas-
ing attention, focusing on when to retrieve and
what to retain under a fixed context budget. Repo-
Former (Wu et al., 2024) triggers retrieval through
self-evaluation, while CODEFILTER (Li et al., 2025)
filters chunks using independent likelihood-based
signals. However, these controllers largely assess
chunks in isolation. As a result, they do not ex-
plicitly account for combinatorial interactions such
as complementarity between interfaces and imple-
2

mentation. In contrast, we cast context filtering as
a coalition scoring problem to model such inter-
dependencies.
Shapley Values in RAG and Supervision.Shap-
ley values (Shapley et al., 1953) provide an
axiomatic notion of marginal contribution and
have been widely used in interpretability, includ-
ing SHAP-style formulations (Lundberg and Lee,
2017; Ghorbani and Zou, 2019; Sundararajan et al.,
2017). In RAG, prior work applies Shapley-
style analysis to attribute outputs to retrieved
documents (Nematov et al., 2025; Ye and Yoga-
narasimhan, 2025) or to estimate token-level im-
portance (Asai et al., 2024; Xiao et al., 2025). Our
use differs in its role in the pipeline. Instead of post-
hoc analysis of a frozen system, we use Shapley-
style marginalization to construct supervision for
retrieval control. We then distill the resulting coali-
tion reasoning into a token-level policy, enabling
practical retrieval decisions during generation.
3 Methodology
3.1 Repository-level Retrieval-Augmented
Code Completion
Repository-level code completion requires ground-
ing generation in cross-file information such as
project-specific APIs, shared utilities, and type
or contract conventions. RAG addresses this by
retrieving candidate snippets from the repository.
However, retrieved evidence is often interaction-
heavy: a snippet may be useful only when paired
with complementary context, and seemingly rel-
evant snippets can degrade generation when they
introduce conflicting implementations.
Problem setup.Given a repository Rand
a target file, each instance is represented as
(Xin, Xout, Y). Here Xin= (X p, Xs)is the in-
file context in fill-in-the-middle (FIM) format with
prefix Xpand suffix Xs,Xoutdenotes a cross-file
pool constructed from other files in R, and Yis
the ground-truth missing span between XpandXs
(Zhang et al., 2023; Wu et al., 2024).
Retrieval and generation.A retriever Rqueries
XoutwithXinand returns top- Kcandidate chunks
Xcc=R(X in, Xout) ={cc 1, . . . , cc K}. A genera-
torGθthen predicts the completion ˆYconditioned
onXinand a selected subset XS⊆X cc. Hence,
the key problem is to estimate chunk utility and
retain the subset that best supports generatingY.3.2 Interaction-aware Chunk Attribution via
Shapley Values
Why independent chunk scoring is insufficient.
Retrieved code snippets rarely contribute indepen-
dently. A chunk can be uninformative on its own
but become essential when paired with complemen-
tary context such as an interface and its implementa-
tion. Conversely, a seemingly relevant snippet may
reduce generation quality when it conflicts with
other retrieved evidence. As a result, per-chunk
scores computed in isolation can be a poor proxy
for the utility of the multi-chunk context used at
test time.
Subset utility as a cooperative game.We there-
fore evaluate chunks at the set level. Given top- K
candidates, we treat each chunk as a player and
any subset as a coalition. Let D= 1, . . . , K index
candidates and S⊆D denote a coalition, with
XS=cc i:i∈S . We define the coalition value as
the normalized teacher-forced log-likelihood gain
on the ground-truth completion:
v(S|X in, Y) =ℓ(X in, XS)−ℓ(X in)
ℓ(C) =1
|Y|logp θ(Y|C).
where logp θ(Y|C) =P|Y|
t=1logp θ(yt|
y<t, C). By construction, v(∅ |X in, Y) = 0 , and
v(S) can be negative when retrieved context de-
creases model likelihood. Appendix C.4 compares
log-likelihood with metric-based utilities (EM/ES)
and shows log-likelihood yields the best down-
stream performance.
Shapley attribution.We quantify interaction-
aware chunk contributions using the Shapley
value (Shapley et al., 1953), which is defined as
the average marginal gain of chunk iover all coali-
tions:
ϕi=X
S⊆D\{i}|S|! (K− |S| −1)!
K!∆vi(S)
∆vi(S) =v(S∪ {i} |X in, Y)−v(S|X in, Y).
Intuitively, ϕi>0 indicates that chunk iis
helpful on average across different co-occurring
contexts, while ϕi≤0 suggests redundancy or
harm under interactions. Shapley values satisfy
efficiency:P
i∈Dϕi=v(D|X in, Y), allowing
negative attributions when some chunks reduce
coalition utility.
3

Figure 3: The overall framework of REPOSHAPLEY. The pipeline consists of two phases: (2) An offline ChunkShap-
ley module that estimates the interaction-aware contribution of each chunk; and (3) An online Shapley-supervised
Generator trained to control retrieval and filter contexts based on the estimated Shapley values.
3.3 ChunkShapley: Practical Shapley
Labeling for Chunk Filtering
Exact Shapley computation under the true coalition
utility v(·)is impractical, as it would require evalu-
ating the generator on exponentially many subsets.
We therefore proposeChunkShapley, anoffline
labeling pipeline that (a) probes each chunk once
to obtain a signed effect, (b) defines a lightweight
surrogate game to approximate interaction patterns,
(c) computes exact Shapley values under the sur-
rogate by enumerating all 2Kcoalitions, which is
inexpensive since vsur(·)is closed-form, and (d)
performs bounded post-verification with the frozen
generator to ground final keep and drop labels in
decoding-time behavior. Algorithmic details are
deferred to Appendix Alg. 2.
(a) Single-chunk probing.We first compute a
per-instance baseline score using teacher forcing
and probe each candidate in isolation. Let ℓ(C) de-
note the normalized teacher-forced log-likelihood.
For each retrieved chunk cci, we define its single-
chunk effect
∆i=ℓ(X in,{cc i})−ℓ(X in)
yi= sign(∆ i), ω i=|∆ i|.
To ensure consistent likelihood estimation under
a limited context window, we preserve the full tar-get span Yand applyleft-truncationonly to the
input context (i.e.,X inand retrieved chunks).
(b) Logistic surrogate game.While ranking
by∆icaptures individual relevance, it ignores
coalition dynamics. To model interactions effi-
ciently, we define a one-dimensional surrogate util-
ity. Given (yi, ωi), we aggregate coalition Svia a
weighted vote:
g(S) =X
i∈Sωiyi, v sur(S) =σ(β g(S))−σ(0).
where σ(·) is the sigmoid and β >0 controls
the saturation scale. This surrogate is not meant
to match the full combinatorial utility; it targets
two dominant effects for filtering. The sigmoid
yields diminishing returns: when |g(S)| is large,
σ′(βg(S))≈0 , so additional similarly-signed evi-
dence contributes little, capturing redundancy un-
der a fixed budget. Conflicts are expressed by neg-
ative votes ( yi=−1 ), which reduce g(S) and can
suppress vsur(S)even when some chunks are in-
dividually helpful. By the way, subtracting σ(0)
ensures vsur(∅) = 0 and keeps utilities centered.
The surrogate remains lightweight for exhaustive
subset evaluation, while any residual mismatch to
decoding-time behavior is addressed by verifica-
tion.
4

(c) Exact Shapley values under the surrogate.
We compute Shapley values using the subset form
under the surrogate utility:
ϕi=1
KX
S⊆D\{i}vsur(S∪ {i})−v sur(S) K−1
|S| .
Since our vsur(S)is closed-form, evaluating all 2K
subsets is computationally negligible for small re-
trieval sizes ( K≤10 ). This allows us to obtain
exactShapley values under vsur, avoiding the vari-
ance of sampling approximations.
In contrast, computing interactions using the
heavy generator Gθwould require exponentially
many coalition evaluations and is intractable.
Therefore, we use ϕiunder the surrogate as a pro-
posal signal and rely on post-verification to finalize
the decision.
(d) Post-verification via a bounded candi-
date pool.Because decoding quality is non-
monotonic in context, so the positive attributions
alone do not guarantee improved greedy decoding.
Since the surrogate is only a proxy, we verify a
small candidate pool with the frozen generator and
select the coalition that maximizes decoding-time
quality. This step is used only for offline label con-
struction with access to Y;the inference never uses
Y.
Letπϕandπ∆be indices sorted by ϕiand∆i.
We build a de-duplicated set Ccontaining: (i) Shap-
ley prefixes {πϕ[1:n]}Nv
n=1, (ii) short ∆prefixes as
a strong single-chunk baseline, and (iii) size-2/3
combinations among top- Lchunks by ∆to explic-
itly probe local synergies. For each S∈ C , we
decode with the frozen generator and choose
S⋆= arg max
S∈C 
ES(ˆYS, Y),EM( ˆYS, Y)
using lexicographic maximization (ES first, EM as
tie-break). We then treat S⋆as the teacher keep/-
drop labels for distillation.
Verified labels for retrieval triggering.The
post-verification step also yields an oracle deci-
sion on whether retrieval is necessary. Let ˆY∅
be the decoding result using only in-file context
Xin, and let ˆYS⋆be the decoding result using the
verification-selected coalition S⋆. We define the
retrieval-control label as
r⋆=(
⟨DONE⟩,ifES( ˆYS⋆, Y)−ES( ˆY∅, Y)≤ϵ
⟨NEED⟩,otherwise.where ϵis a small margin tuned on the validation
set (default ϵ= 0 unless stated otherwise). This
label is used only for offline supervision; inference
never accessesY.
3.4 REPOSHAPLEY: Distilling ChunkShapley
into Signal Tokens
While ChunkShapley provides robust coalition-
aware supervision, the pipeline is too computation-
ally intensive for online use. We therefore propose
REPOSHAPLEY, whichdistillsverified coalition
decisions into discrete control tokens, enabling a
single generator to efficiently decidewhento re-
trieve andwhichchunks to retain at inference time.
Signal tokens and verified labels.We introduce
retrieval-control tokens TR={⟨NEED⟩,⟨DONE⟩} to
decide whether cross-file evidence is required, and
candidate-selection tokens TS={⟨KEEP⟩,⟨DROP⟩}
to indicate which retrieved chunks should be re-
tained.
Step (d) outputs averification-selected coalition
S⋆by evaluating a small set of Shapley-proposed
candidate coalitions Cusing the frozen gener-
ator under decoding-time constraints, to match
decoding-time behavior. We treat S⋆as the teacher
keep/drop label set and distill it into token-level
supervision by assigning, for each retrieved chunk
cci,
Q(cc i) =(
⟨KEEP⟩ifi∈S⋆
⟨DROP⟩otherwise.
In this way, surrogate Shapley signals are used only
to propose promising coalitions, while the student
model learns to imitate theverifiedcoalition-level
behavior encoded by S⋆, turning combinatorial sub-
set selection into a single-shot, controllable genera-
tion policy at inference time.
Training: two-format verbalized supervision.
Following the standard separation of evidence
selection and completion generation in retrieval-
augmented code modeling, we train a single model
with two serialized views of each instance. Format-
1 supervisesselection: given the in-file context and
retrieved candidates, the model emits a keep/drop
decision token for each chunk. Format-2 super-
visesgeneration: the model produces the missing
span conditioned only on the kept evidence. Both
formats reuse the same control tokens and share
all parameters, enabling the model to learn selec-
tion and generation within a unified autoregressive
interface.
5

Format-1: Selection.Given the in-file con-
text and the retrieved candidate list, the model
predicts a length- Kdecision sequence q1:K∈
{⟨KEEP⟩,⟨DROP⟩}Kunder a dedicated ⟨SELECT⟩
marker. Let [Xp]and[Xs]denote tokenized FIM
prefix and suffix, and let Pack(X cc)be the de-
terministic serialization of retrieved candidates
Xcc={cc 1, . . . , cc K}:
Pack(X cc) =⟨C 1⟩[cc1]⟨/C1⟩ ··· ⟨C K⟩[ccK]⟨/CK⟩.
The Format-1 sequence is
F1:⟨PFX⟩[X p]⟨SFX⟩[X s]⟨NEED⟩
Pack(X cc)⟨SELECT⟩q 1q2···q K⟨DONE⟩.
We supervise qiusing theverified teacher coali-
tionS⋆:q⋆
i=⟨KEEP⟩ ifi∈S⋆and⟨DROP⟩ other-
wise.
Format-2: Generation.To teach the model how to
complete codegiven filtered evidence, we construct
a generation format that includes only the chunks in
S⋆and then decodes the target span in FIM mode:
F2:⟨PFX⟩[X p]⟨SFX⟩[X s]⟨NEED⟩Pack(C S⋆)
⟨DONE⟩ ⟨MID⟩[Y].
No-retrieval format.If retrieval is unnecessary
(r⋆=⟨DONE⟩ ), we drop the cross-file block and the
selection head:
⟨PFX⟩[X p]⟨SFX⟩[X s]⟨DONE⟩ ⟨MID⟩[Y].
This indicates that in-file context suffices.
Remark.We reuse ⟨DONE⟩ both as the retrieval
decision token and as a block delimiter; the two
usages are unambiguous from their fixed posi-
tions in the sequence. The retrieval-control token
(⟨NEED⟩/⟨DONE⟩ ) is learned with teacher forcing as
a next-token target (counted in LR), rather than
provided as an oracle input.
Objectives with masked contexts.We mask all
in-file and cross-filecontenttokens in the loss
and compute gradients only ongenerated targets
(control tokens, selection tokens, and the comple-
tionY). Let r⋆∈ T R={⟨NEED⟩,⟨DONE⟩} be
the retrieval-control label. For retrieval-needed in-
stances (Formats F1/F2 ),r⋆=⟨NEED⟩ ; for no-
retrieval instances, r⋆=⟨DONE⟩ . For Format-1, we
optimize retrieval triggering and selection:
LF1
R=−logP G 
r⋆|X in
LF1
S=−X
i∈JlogP G 
q⋆
i|X in, Xcc, r⋆, q⋆
<i
LF1=λRLF1
R+λSLF1
SAlgorithm 1:REPOSHAPLEYInference
Process
Input:GeneratorG, RetrieverR,
Cross-file poolX out, In-file context
Xin= (X p, Xs);
Token setsT R={⟨NEED⟩,⟨DONE⟩},
TS={⟨KEEP⟩,⟨DROP⟩}; thresholdt c.
Output:Completed code ˆY.
1X←(⟨PFX⟩, X p,⟨SFX⟩, X s)
2r←Select(Softmax TR(G(· |X)), t c)
3ifr=⟨DONE⟩then
4X←append(X,⟨MID⟩)
5return ˆY←G(X)
6end
7X cc←R(X in, Xout)
8X sel←X⊕⟨NEED⟩⊕Pack(X cc)⊕⟨SELECT⟩
9(q1, . . . , q K)←G(X sel)
10ˆS← {i∈ {1, . . . , K}:q i=⟨KEEP⟩}
11X←X⊕ ⟨NEED⟩ ⊕Pack(C ˆS)⊕ ⟨DONE⟩
12X←append(X,⟨MID⟩)
13return ˆY←G(X)
where J={1, . . . , K} for retrieval-needed in-
stances andJ=∅for no-retrieval instances.
For Format-2, we optimize retrieval triggering
and generation conditioned on the verified filtered
context. Let XS⋆= Pack(C S⋆)denote the serial-
ized filtered evidence.
LF2
R=−logP G 
r⋆|X in
LF2
Y=−TX
t=1logP G 
yt|y<t, Xin, XS⋆, r⋆
LF2=λRLF2
R+LF2
Y.
HereLRis implemented as the cross-entropy on
the next-token prediction at the retrieval-control
position (i.e., immediately after ⟨SFX⟩[X s]), rather
than a separate classifier.
During training, we either (i) include both for-
mats for each instance, or (ii) sample one format
per instance with a fixed mixing ratio. The final
objective is the expectation over the chosen format:
L=E F∼π
LF
,F∈ {F1,F2}.
Inference.At inference time, REPOSHAPLEY
makes retrieval decisions in one autoregressive
rollout. Given the in-file context, the model first
predicts a retrieval-control token r∈ T R=
⟨NEED⟩,⟨DONE⟩ . Ifr=⟨DONE⟩ , it directly performs
FIM decoding to generate the completion.
6

Table 1: Code completion performance in the Infilling setting.
Model StrategyRepoEval CCLongEval CCEval
Line API Function Chunk Func Line
EM (M1) ES (M2) EM (M3) ES (M4) UT (M5) ES (M6) EM (M7) ES (M8) ES (M9) EM (M10) ES (M11)
SC-Base-1BNo-Retrieve 43.14 67.39 38.03 66.81 21.67 47.29 30.62 60.54 47.16 18.72 42.85
Full-Retrieve 52.27 73.13 44.18 69.09 25.61 55.93 37.49 64.04 50.72 22.38 47.26
RepoFormer 54.71 76.52 45.73 72.41 28.46 57.69 41.93 70.21 54.37 25.42 49.18
CODEFILTER 57.19 78.84 48.37 75.66 31.13 59.91 44.52 72.48 56.59 27.81 52.03
REPOSHAPLEY 61.34+4.15 82.78+3.94 53.62+5.25 79.53+3.87 35.84+4.71 64.39+4.48 48.57+4.05 77.52+5.04 61.18+4.59 32.26+4.45 56.37+4.34
SC-Base-3BNo-Retrieve 48.12 72.38 40.17 68.91 24.93 51.52 36.16 65.19 49.63 21.82 45.58
Full-Retrieve 57.84 77.21 48.83 72.68 30.58 58.16 42.61 68.29 53.84 25.92 50.31
RepoFormer 58.59 79.16 49.82 74.63 32.89 60.62 46.38 72.11 56.39 28.85 52.16
CODEFILTER 61.21 81.09 51.97 77.62 35.18 63.26 49.62 74.58 58.51 30.84 55.29
REPOSHAPLEY 64.93+3.72 85.27+4.18 56.38+4.41 81.72+4.10 39.91+4.73 68.16+4.90 53.52+3.90 78.83+4.25 62.84+4.33 35.79+4.95 59.41+4.12
SC-Base-7BNo-Retrieve 51.62 75.51 43.83 71.29 25.62 52.71 38.91 66.62 52.84 23.37 48.01
Full-Retrieve 58.26 77.79 50.38 75.01 32.26 60.21 44.62 69.19 55.16 28.51 52.91
RepoFormer 59.83 79.26 51.31 77.46 35.71 61.19 46.84 74.16 57.11 29.62 55.49
CODEFILTER 61.49 81.41 53.62 79.29 37.79 63.41 49.16 77.26 59.84 32.11 57.84
REPOSHAPLEY 65.81+4.32 86.59+5.18 58.79+5.17 84.11+4.82 41.84+4.05 68.16+4.75 54.73+5.57 81.29+4.03 64.62+4.78 36.59+4.48 62.91+5.07
Llama-7BNo-Retrieve 51.89 73.42 41.53 66.98 24.81 44.56 37.21 65.16 50.37 18.16 43.34
Full-Retrieve 60.18 78.91 48.76 73.16 29.93 52.21 45.41 69.37 52.11 23.41 47.46
RepoFormer 60.52 79.36 49.31 75.91 33.19 52.64 46.84 69.56 52.16 24.26 48.31
CODEFILTER 63.76 82.31 52.62 78.54 32.84 54.49 50.76 74.91 54.74 27.16 51.68
REPOSHAPLEY 68.31+4.55 86.76+4.45 57.28+4.66 83.11+4.57 37.56+4.72 59.14+4.65 55.21+4.45 79.19+4.28 59.41+4.67 31.23+4.07 55.24+3.56
Llama-13BNo-Retrieve 53.81 74.84 42.19 67.96 26.31 47.14 41.91 67.46 52.71 20.97 45.88
Full-Retrieve 61.41 79.29 49.81 77.41 31.69 54.21 47.36 70.61 55.24 25.53 50.20
RepoFormer 62.19 81.51 50.46 79.21 34.39 54.44 48.96 71.11 55.41 27.20 52.20
CODEFILTER 64.16 82.71 53.01 78.99 35.41 57.76 51.31 74.19 58.81 29.91 54.69
REPOSHAPLEY 68.89+4.73 87.11+4.40 57.66+4.65 83.41+4.42 40.11+4.70 62.39+4.63 55.91+4.60 78.59+4.40 63.51+4.70 35.05+5.14 59.37+4.68
Ifr=⟨NEED⟩ , we retrieve Kcross-file can-
didates Xcc=cc 1, . . . , cc Kand serialize them
asPack(X cc). Conditioned on this packed block,
the model outputs a length- Kselection sequence
under⟨SELECT⟩ , where (q1, . . . , q K)∈ TK
Sand
TS=⟨KEEP⟩,⟨DROP⟩ . We then keep only chunks
withqi=⟨KEEP⟩ , append them to the prompt, and
generate ˆYvia FIM decoding after emitting ⟨MID⟩ .
Alg. 1 provides the full procedure. We use ⊕to
denote token sequence concatenation.
4 Experiments
4.1 Experimental Setup
Dataset.We curate 290k Python repositories
from The Stack (Kocetkov et al., 2023) after strict
quality filtering (LOC constraints, AST parsing,
and deduplication; Appendix A.2). Following (Wu
et al., 2024), we sample 7.5k repositories to con-
struct 50k labeled training instances: for each in-
stance, we retrieve top-10 cross-file chunks using
Jaccard similarity (Jaccard, 1912) and assign super-
vision derived from ChunkShapley. During data
labeling, we discard instances whose verification-
selected coalition S⋆fails to reach a minimum com-
pletion quality, so that ES(ˆYS⋆, Y)< τ es, to en-
sure supervision reliability. We split the data into
95%/5% for training and validation.
Models and Training.We fine-tune StarCoder-
Base (SCB-1B/3B/7B) (Li et al., 2023) and CodeL-
lama (Llama-7B/13B) (Roziere et al., 2023) for 2epochs using a learning rate of 2×10−5with linear
decay and 5% warm-up. We set λR=λS= 2.0 ,
max sequence length to 4096. With a global batch
size of 512 on 8 NVIDIA H100 (80GB), train-
ing takes on average 2.2/6.5/15.4 hours for SCB-
1B/3B/7B and 15.8/28.6 hours for Llama-7B/13B,
respectively. (Details are shown in Appendix. B)
Benchmarks and Metrics.We evaluate on three
repository-level code completion benchmarks: Re-
poEval (Zhang et al., 2023), CrossCodeEval (Ding
et al., 2023), and CrossCodeLongEval (Wu et al.,
2024). Together they cover line, API, chunk,
and function-level completion tasks under realistic
cross-file dependencies. We consider two prompt-
ing settings:Infilling(FIM with Xin= (X p, Xs))
andLeft-to-right(prefix-only with Xin=X p).
Following prior work (Wu et al., 2024), we report
Exact Match (EM) and Edit Similarity (ES) for
non-function tasks, and unit-test pass rate (UT)
for function tasks (Formulations are shown in Ap-
pendix. A.1).
Baselines.We compare REPOSHAPLEYagainst:
(1)No-Retrieve(in-file only); (2)Full-Retrieve
(Zhang et al., 2023) (top-10 sparse retrieval); (3)
RepoFormer(Wu et al., 2024) (selective retrieval);
and (4) CODEFILTER (Li et al., 2025) (likelihood-
based filtering). CODEFILTER serves as the pri-
mary baseline to highlight the benefit of interaction-
aware supervision.
7

Table 2: Component Ablation of REPOSHAPLEYon
RepoEval. We investigate components in (A) Labeling
and (B) Distillation. Baseline is SC-Base-1B.
Method / VariantRepoEval-Line RepoEval-API Latency
EM ES EM ES (ms/req)
RepoFormer 54.71 79.26 45.73 72.41 661
CODEFILTER 57.19 78.84 48.37 75.66 947
REPOSHAPLEY 61.34 82.78 53.62 79.53 1053
A. Labeling Strategy
1. w/o Post-verification 38.50 54.44 36.15 55.81 –
2.∆-only labeling 58.45 77.12 48.46 75.26 –
3. Linear Surrogate 59.92 76.41 50.73 77.09 –
4. Uniform Weights 60.18 80.97 51.82 77.38 –
B. Distillation
5. Format-1 only 5.56 8.27 2.34 5.66 523
6. Format-2 only 59.88 79.11 52.12 79.49 830
7. No Trigger 61.26 81.33 52.15 78.81 1462
4.2 Main Results
Tables 1 and 3 show that REPOSHAPLEYconsis-
tently improves repository-level infilling across
benchmarks and backbones, validating our core
hypothesis that supervision derived from evidence
coalitions better reflects interaction-heavy retrieval.
First, interaction-blind filtering remains brittle.
While adaptive controllers generally outperform
Full-Retrieve, methods trained from per-chunk la-
bels ( CODEFILTER ) can still overfit to isolated simi-
larity and fail to account for complementarity and
conflict that only appear when multiple chunks are
concatenated. This gap is most visible on harder
settings that require resolving non-local dependen-
cies such as Function, where selecting the right
combination of evidence matters more.
Second, coalition-aware supervision yields the
strongest gains on difficult tasks. On SC-Base-
7B, REPOSHAPLEYimproves RepoEval API from
53.62/79.29 to58.79/84.11(EM/ES) and raises Re-
poEval Function unit-test pass rate from 37.79 to
41.84, outperforming CODEFILTER by clear mar-
gins. These improvements align with our motiva-
tion: modeling evidence interactions helps retain
complementary context while suppressing conflict-
ing or redundant chunks.
Finally, the gains generalize beyond RepoE-
val. REPOSHAPLEYalso delivers consistent im-
provements on long-context and chunk-level bench-
marks, on CCLongEval Chunk it improves ES from
77.26 to81.29on SC-Base-7B and from 74.19 to
78.59on Llama-13B, indicating that the learned
keep and drop policy transfers across evaluation
granularities and context regimes.
Although REPOSHAPLEYintroduces additionalcomputation during offline labeling, its inference-
time overhead remains modest. As shown in Ta-
ble 2, REPOSHAPLEYruns at 1053 ms/req, which
is comparable to CODEFILTER (947 ms) and within
the same runtime scale as RepoFormer (661 ms).
4.3 Ablation Study & Analysis
We study how each component of REPOSHAPLEY
affects performance by ablating (A) the offline la-
beling pipeline and (B) the online distillation strat-
egy on StarCoderBase-1B with RepoEval (Table 2).
Coalition-aware labeling matters.Ablations in
Part A show that modeling interactions is necessary
for reliable filtering. Using Shapley signs without
post-verification (Row 1) causes a large drop, in-
dicating that signed marginal effects alone are not
stable under prerequisite dependencies. Replac-
ing coalition-based attribution with single-chunk
probing (Row 2) also hurts performance, suggest-
ing that independent scores miss synergy among
chunks. Simplifying the surrogate utility by remov-
ing the sigmoid (Row 3) or using uniform weights
(Row 4) further degrades results, supporting our
design for capturing saturation and conflict effects.
Distillation and triggering improve inference.
Part B shows that the training design is essen-
tial. Training with selection-only signals (Row 5)
fails to produce usable code, while generation-only
training (Row 6) lags behind the full model due
to residual noise. Removing the trigger (Row 7)
yields similar accuracy but increases latency, con-
firming that the learned trigger reduces unnecessary
retrieval while maintaining generation quality.
5 Conclusion
In this work, we study repository-level retrieval
control under strong evidence interaction effects.
We propose ChunkShapley, a practical approxima-
tion that combines Ksingle-chunk probes with
a continuous logistic surrogate game to compute
Shapley-style attributions efficiently, and further
introduces a compact post-verification step that
grounds selection in the true generator’s behavior.
This design bridges the gap between interaction-
aware attribution and deployable retrieval con-
trol: the surrogate Shapley stage provides a princi-
pled ranking that accounts for complementary or
conflicting evidence, while post-verification miti-
gates non-monotonicity and metric-specific failure
modes by selecting the best coalition from a small,
structured candidate pool.
8

Limitations
Our method has several limitations. First, the surro-
gate utility is constructed from single-chunk probes,
which provides a tractable approximation but can
miss higher-order interactions where multiple indi-
vidually weak chunks become useful only jointly.
Our bounded post-verification mitigates this issue,
yet it cannot guarantee recovering such cases out-
side the candidate pool. Second, our offline label-
ing requires enumerating all 2Ksubsets under the
surrogate game; although we use small retrieval
sizes ( K=10 ), the cost grows exponentially and
may limit scalability to larger candidate sets or
higher budgets. Third, the verification stage is tied
to a particular decoding setup and evaluation cri-
teria (greedy decoding with ES/EM); the selected
coalition may vary under different decoding algo-
rithms, stochastic sampling, or task-specific objec-
tives. Finally, the labeling pipeline involves multi-
ple teacher-forced forward passes for probing and
additional decoding runs for verification, which in-
creases offline computation and can be costly when
constructing large-scale supervision.
References
Anthropic. 2025. System card: Claude opus 4 & claude
sonnet 4. Technical report, Anthropic. Accessed:
2026-01-03.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Ramakrishna Bairi, Atharv Sonwane, Aditya Kanade,
Vageesh D C, Arun Iyer, Suresh Parthasarathy, Sri-
ram Rajamani, B. Ashok, and Shashank Shet. 2023.
Codeplan: Repository-level coding using LLMs and
planning. InNeurIPS 2023 Foundation Models for
Decision Making Workshop.
Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon,
Jonathan Berant, Matthew R Gormley, and Graham
Neubig. 2025. In-context learning with long-context
models: An in-depth exploration. InProceedings of
the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 12119–12149.
Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-
guided retrieval augmentation for repository-level
code completion. InProceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 7957–7977.
Yangruibo Ding, Jinjun Peng, Marcus J. Min, Gail
Kaiser, Junfeng Yang, and Baishakhi Ray. 2024a.Semcoder: Training code language models with com-
prehensive semantics reasoning. InAdvances in
Neural Information Processing Systems, volume 37,
pages 60275–60308. Curran Associates, Inc.
Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian
Ding, Ming Tan, Nihal Jain, Murali Krishna Ra-
manathan, Ramesh Nallapati, Parminder Bhatia, Dan
Roth, and 1 others. 2023. Crosscodeeval: A diverse
and multilingual benchmark for cross-file code com-
pletion.Advances in Neural Information Processing
Systems, 36:46701–46723.
Yangruibo Ding, Zijian Wang, Wasi Ahmad, Murali Kr-
ishna Ramanathan, Ramesh Nallapati, Parminder
Bhatia, Dan Roth, and Bing Xiang. 2024b. Cocomic:
Code completion by jointly modeling in-file and
cross-file context. InProceedings of the 2024 Joint
International Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-
COLING 2024), pages 3433–3445.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
CRAC Generation. 2024. Coderag-bench: Can retrieval
augment code generation.
Amirata Ghorbani and James Zou. 2019. Data shapley:
Equitable valuation of data for machine learning. In
International conference on machine learning, pages
2242–2251. PMLR.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-
r1 incentivizes reasoning in llms through reinforce-
ment learning.Nature, 645(8081):633–638.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Day-
iheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang,
Bowen Yu, Kai Dang, and 1 others. 2024. Qwen2.
5-coder technical report.CoRR.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics: main volume, pages
874–880.
Paul Jaccard. 1912. The distribution of the flora in the
alpine zone. 1.New phytologist, 11(2):37–50.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
9

Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Carlos E Jimenez, John Yang, Alexander Wettig,
Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. 2024. Swe-bench: Can language mod-
els resolve real-world github issues? In12th Inter-
national Conference on Learning Representations,
ICLR 2024.
Mintong Kang, Nezihe Merve Gürel, Ning Yu, Dawn
Song, and Bo Li. 2024. C-RAG: Certified genera-
tion risks for retrieval-augmented language models.
InForty-first International Conference on Machine
Learning.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2020. Generalization
through memorization: Nearest neighbor language
models. InInternational Conference on Learning
Representations.
Denis Kocetkov, Raymond Li, Loubna Ben allal, Jia LI,
Chenghao Mou, Yacine Jernite, Margaret Mitchell,
Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf,
Dzmitry Bahdanau, Leandro V on Werra, and Harm
de Vries. 2023. The stack: 3 TB of permissively li-
censed source code.Transactions on Machine Learn-
ing Research.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
R Li, LB Allal, Y Zi, N Muennighoff, D Kocetkov,
C Mou, M Marone, C Akiki, J Li, J Chim, and 1
others. 2023. Starcoder: May the source be with
you!Transactions on machine learning research.
Yanzhou Li, Shangqing Liu, Kangjie Chen, Tianwei
Zhang, and Yang Liu. 2025. Impact-driven context
filtering for cross-file code completion. InSecond
Conference on Language Modeling.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Tianyang Liu, Canwen Xu, and Julian McAuley. 2024b.
Repobench: Benchmarking repository-level code
auto-completion systems. InThe Twelfth Interna-
tional Conference on Learning Representations.
Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang,
Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024c.
Graphcoder: Enhancing repository-level code com-
pletion via coarse-to-fine retrieval based on code con-
text graph. InProceedings of the 39th IEEE/ACM
International Conference on Automated Software En-
gineering, pages 570–581.Scott M. Lundberg and Su-In Lee. 2017. A unified
approach to interpreting model predictions. InPro-
ceedings of the 31st International Conference on Neu-
ral Information Processing Systems, NIPS’17, page
4768–4777, Red Hook, NY , USA. Curran Associates
Inc.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822.
Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko,
Gabriele Fugagnoli, Dimitris Sacharidis, Katja Hose,
and Tomer Sagi. 2025. Source attribution in
retrieval-augmented generation.arXiv preprint
arXiv:2507.04480.
OpenAI. 2025. Gpt-5. https://openai.com/gpt-5 .
Accessed: 2026-01-03.
Md Rizwan Parvez, Wasi Ahmad, Saikat Chakraborty,
Baishakhi Ray, and Kai-Wei Chang. 2021. Retrieval
augmented code generation and summarization. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2021, pages 2719–2734.
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi,
Jingyu Liu, Romain Sauvestre, Tal Remez, and 1
others. 2023. Code llama: Open foundation models
for code.arXiv preprint arXiv:2308.12950.
Lloyd S Shapley and 1 others. 1953. A value for n-
person games.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. InInter-
national Conference on Machine Learning, pages
31210–31227. PMLR.
Disha Shrivastava, Denis Kocetkov, Harm De Vries,
Dzmitry Bahdanau, and Torsten Scholak. 2023. Re-
pofusion: Training code models to understand your
repository.arXiv preprint arXiv:2306.10998.
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017.
Axiomatic attribution for deep networks. InInterna-
tional conference on machine learning, pages 3319–
3328. PMLR.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. In-
structRAG: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. InThe Thirteenth
International Conference on Learning Representa-
tions.
Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Kr-
ishna Ramanathan, and Xiaofei Ma. 2024. Repo-
former: selective retrieval for repository-level code
10

completion. InProceedings of the 41st Interna-
tional Conference on Machine Learning, ICML’24.
JMLR.org.
Yingtai Xiao, Yuqing Zhu, Sirat Samyoun, Wanrong
Zhang, Jiachen T Wang, and Jian Du. 2025. Token-
shapley: Token level context attribution with shapley
value. InFindings of the Association for Computa-
tional Linguistics: ACL 2025, pages 3882–3894.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RE-
COMP: Improving retrieval-augmented LMs with
context compression and selective augmentation. In
The Twelfth International Conference on Learning
Representations.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884.
Zezhou Yang, Ting Peng, Cuiyun Gao, Chaozheng
Wang, Hailiang Huang, and Yuetang Deng. 2025.
A deep dive into retrieval-augmented generation for
code completion: Experience on wechat.arXiv
preprint arXiv:2507.18515.
Chengyuan Yao and Satoshi Fujita. 2024. Adaptive
control of retrieval-augmented generation for large
language models through reflective tags.Electronics,
13(23):4643.
Zikun Ye and Hema Yoganarasimhan. 2025. Fair docu-
ment valuation in llm summaries via shapley values.
arXiv preprint arXiv:2505.23842.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. Making retrieval-augmented language models
robust to irrelevant context. InICLR 2024 Workshop
on Large Language Model (LLM) Agents.
Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin
Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao
Zeng, Jiajie Zhang, and 1 others. 2025. Glm-4.5:
Agentic, reasoning, and coding (arc) foundation mod-
els.arXiv preprint arXiv:2508.06471.
Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin
Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and
Weizhu Chen. 2023. Repocoder: Repository-level
code completion through iterative retrieval and gener-
ation. InThe 2023 Conference on Empirical Methods
in Natural Language Processing.
Sheng Zhang, Yifan Ding, Shuquan Lian, Shun Song,
and Hui Li. 2025. Coderag: Finding relevant and nec-
essary knowledge for retrieval-augmented repository-
level code completion. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 23289–23299.
11

A Details of dataset construction
A.1 Metrics Formulation
We evaluate code completion quality using Exact
Match (EM), Edit Similarity (ES), and Unit Tests
(UT). Let ˆYbe the generated code and Ybe the
ground truth:
EM=1( ˆY=Y)
ES= [1−D(ˆY , Y)
max(| ˆY||Y|)]×100%
UT=1(Pass( ˆY))
where 1(·) is the indicator function, D(·) de-
notes the Levenshtein distance, and Pass(·) returns
true if the code passes all unit tests.
A.2 Data Collection and Preprocessing
File-level filtering.We begin with conservative
file hygiene to reduce retrieval noise and stabilize
likelihood-based labeling. We keep only .pyfiles
and discard files with fewer than 10non-empty
lines. To remove minified/generated blobs that dis-
tort sparse retrieval, we drop files whose maximum
line length exceeds 300characters or whose aver-
age line length exceeds 120characters (computed
after trimming trailing whitespace). We further
filter out non-code payloads by requiring alphanu-
meric density ≥0.35 (ratio of letters/digits over
all characters). Finally, we exclude vendored or
generated directories by path keywords, includ-
ingvendor/ ,third_party/ ,site-packages/ ,
dist/ ,build/ ,.venv/ , and migrations/ . All
statistics are computed on UTF-8 decoded text
(with a permissive fallback that drops undecodable
bytes).
Repository-level filtering.We retain reposito-
ries with sufficient structure for cross-file inter-
actions by requiring at least 8remaining Python
files and total non-empty LOC between 300and
50,000 after file-level filtering. To avoid duplicate-
heavy projects where top- Kretrieval collapses to
repeated copies, we estimate the near-duplicate
file ratio usingSimHash. Specifically, for each
repository we compute SimHash fingerprints over
a normalized UTF-8 text representation of each
file (whitespace-collapsed, with trailing whitespace
removed), and perform pairwise checks on up to
the first 200files ( max_files_for_dup_check=
200). We mark two files as near-duplicates iftheir SimHash Hamming distance is at most 3
(simhash_hamming_threshold= 3 ). Reposito-
ries with more than 30% near-duplicate files are
discarded (max_dup_ratio= 0.3).
We also enforce syntactic integrity by parsing
a sampled subset of files with Python ast.parse .
Concretely, we uniformly sample up to 20files per
repository ( ast_sample_k= 20 ) from the remain-
ing Python files after file-level filtering, and com-
pute the parse success rate on this sample. Reposi-
tories with parse success rate <70% are removed
(min_ast_parse_rate= 0.7 ). For reproducibility,
both the duplicate-check subsampling (when appli-
cable) and AST sampling use a fixed random seed
of13(seed= 13).
A.3 Data labeling.
We chunk the cross-file pool and construct a re-
trieval query from the in-file context for each target
spanY. For each query, we retrieve the top- Kcan-
didate chunks and distill coalition-aware decisions
into chunk-wise KEEP /DROP labels and a retrieval-
control token. Specifically, given Xin= (X p, Xs)
and retrieved candidates Xcc= (cc 1, . . . , cc K), we
run our offlineChunkShapleypipeline to obtain a
verification-selected coalition S⋆⊆ {1, . . . , K} .
We first compute a teacher-forced baseline log-
likelihood ℓ(∅) and probe each chunk in isola-
tion, ∆i=ℓ({i})−ℓ(∅) , yielding a signed vote
yi= sign(∆ i)and weight ωi=|∆ i|. We then
define a lightweight surrogate game vsur(S) =
σ(βP
i∈Sωiyi)−σ(0) , where σ(·) is the sigmoid
function, and compute exact surrogate Shapley val-
ues by enumerating all 2Kcoalitions (tractable for
small Kand performed offline). Finally, we ver-
ify a bounded set of Shapley-proposed coalitions
using the frozen generator under decoding-time
constraints and select S⋆that maximizes comple-
tion quality (lexicographically by ES then EM).
We treat S⋆as the teacher subset and assign labels:
Q(cc i) =⟨KEEP⟩ifi∈S⋆and⟨DROP⟩otherwise.
To supervise retrieval triggering, we assign the
retrieval-control token r⋆∈ {⟨NEED⟩,⟨DONE⟩} by
comparing the completion quality with and with-
out cross-file evidence: if the in-file context alone
already achieves ES above a threshold τdone(or the
verified coalition provides negligible gain), we set
r⋆=⟨DONE⟩ ; otherwise r⋆=⟨NEED⟩ . To ensure
that retained evidence is meaningful, we filter in-
stances by requiring the verification-selected coali-
tion to achieve ES ≥τ es. Alg. 2 summarizes the
labeling procedure.
12

Table 3: Code completion performance in the Left-to-Right setting.
Model StrategyRepoEval CCLongEval CCEval
Line API Function Chunk Func Line
EM ES EM ES UT ES EM ES ES EM ES
SC-Base-1BNo-Retrieve 33.42 57.88 28.54 57.36 16.55 40.21 22.45 53.05 39.88 16.54 54.47
Full-Retrieve 44.52 66.21 36.95 64.77 21.30 48.55 31.12 63.49 45.36 20.12 58.21
RepoFormer 46.12 68.33 37.44 66.12 23.45 50.12 32.55 65.12 46.88 22.45 60.33
CODEFILTER 48.88 70.15 39.85 69.11 24.12 51.55 34.15 66.88 48.22 24.88 62.55
REPOSHAPLEY 54.21+5.33 76.45+6.30 45.66+5.81 74.88+5.77 29.85+5.73 57.22+5.67 40.55+6.40 72.44+5.56 54.12+5.90 30.12+5.24 68.95+6.40
SC-Base-3BNo-Retrieve 35.82 60.12 29.55 58.45 20.05 38.95 25.44 53.45 44.82 18.22 57.51
Full-Retrieve 50.45 70.88 40.66 67.89 26.12 48.66 36.15 59.88 45.75 23.45 62.12
RepoFormer 50.11 71.95 41.02 69.88 27.55 50.12 36.75 61.95 47.12 25.66 63.88
CODEFILTER 53.12 73.66 43.15 73.12 27.88 51.22 38.05 61.55 48.88 27.45 65.12
REPOSHAPLEY 59.45+6.33 79.11+5.45 48.88+5.73 79.55+6.43 33.45+5.57 57.88+6.66 44.22+6.17 67.12+5.57 54.66+5.78 33.15+5.70 70.44+5.32
SC-Base-7BNo-Retrieve 38.15 62.12 31.45 59.88 21.88 39.95 29.45 58.55 53.45 19.68 59.00
Full-Retrieve 51.22 71.55 42.45 68.33 28.15 50.45 41.55 64.88 48.75 24.55 63.45
RepoFormer 50.88 70.45 40.88 72.66 28.05 48.55 41.05 64.75 48.15 26.88 65.12
CODEFILTER 53.95 74.22 44.55 72.15 29.05 52.33 42.12 66.88 57.88 28.55 67.55
REPOSHAPLEY 59.88+5.93 80.55+6.33 50.12+5.57 78.45+6.30 34.66+5.61 58.12+5.79 48.45+6.33 72.15+5.27 63.45+5.57 34.12+5.57 73.22+5.67
Llama-7BNo-Retrieve 39.55 64.12 30.88 60.22 22.45 42.55 30.12 58.12 45.45 20.88 60.12
Full-Retrieve 52.45 70.88 43.15 68.75 26.88 50.12 41.22 63.88 52.66 25.44 64.55
RepoFormer 51.12 71.45 40.88 70.88 28.66 50.05 39.45 62.95 50.45 27.12 65.88
CODEFILTER 53.66 73.12 43.88 73.55 29.88 50.88 41.88 65.45 53.55 29.45 67.88
REPOSHAPLEY 59.12+5.46 78.66+5.54 49.55+5.67 79.12+5.57 35.12+5.24 56.45+5.57 47.22+5.34 71.05+5.60 59.12+5.57 35.66+6.21 73.45+5.57
Llama-13BNo-Retrieve 41.55 65.12 31.22 60.66 24.12 43.55 31.55 57.66 46.12 21.88 61.45
Full-Retrieve 54.22 74.45 44.66 71.88 28.95 51.45 43.22 68.12 50.22 26.55 66.12
RepoFormer 52.45 71.55 43.95 71.66 28.66 51.12 43.55 67.88 52.45 28.12 67.45
CODEFILTER 55.33 75.12 45.12 74.95 30.12 52.45 44.22 67.95 57.12 30.88 69.55
REPOSHAPLEY 61.12+5.79 81.45+6.33 51.55+6.43 80.66+5.71 36.45+6.33 58.22+5.77 50.12+5.90 73.45+5.50 62.88+5.76 36.95+6.07 75.12+5.57
A.4 Labeling Algorithm Details
Algorithm 3 outlines the process of deriving
supervision signals from raw code repositories.
First, top- Kcandidate chunks Xccare retrieved
based on the query window Q. We then utilize
CHUNKSHAPLEYto identify the optimal chunk
subset S⋆that maximizes generation quality rela-
tive to the ground truthY.
The labeling logic follows three specific criteria:
1.Quality Control:Instances are discarded if
the optimal subset’s performance falls below a
minimum threshold τes, ensuring training data
quality.
2.Retrieval Label ( r⋆):We measure the per-
formance gain of using external context ( S⋆)
versus the closed-book baseline ( ∅). If the
gain is negligible ( ≤ϵ), the retrieval label is
set to⟨DONE⟩; otherwise, it is⟨NEED⟩.
3.Selection Label ( q⋆
i):Individual chunks are
labeled as ⟨KEEP⟩ if they belong to the optimal
subsetS⋆, and⟨DROP⟩otherwise.
B Hyperparameter Optimization
We tune training hyperparameters using
StarCoderBase-1Bas a proxy model to re-
duce search cost. Unless otherwise specified, all
other settings follow the main experimental setup
(e.g., data split, prompt formats, max sequence
length, and batching).Search space.We conduct a grid search on
the following space: learning rate ∈ {1×
10−5,2×10−5,5×10−5}, loss weight λ∈
{0.2,1.0,2.0,5.0} , training epochs ∈ {1,2,5} ,
and warmup steps ∈ {50,100} . Here λis applied
to the retrieval-control and selection losses, i.e.,
λR=λS=λ (while the generation loss uses unit
weight).
Selection criterion.For each configuration, we
evaluate code completion performance on the vali-
dation split using the same metrics as in the main
experiments. We select the best hyperparameters
by maximizing the validation completion quality
(with ES as the primary criterion and EM as a tie-
breaker).
Final configuration.The selected hyperparame-
ters are: learning rate 2×10−5,λR=λS= 2.0 ,
epochs = 2, warmup steps = 50 . We reuse this
configuration for all backbones in our experiments
for consistency.
C Detailed Experiments
C.1 Ablation on Verification Scope (L)
Table 5 shows that expanding the verification scope
fromL= 0 toL= 3 brings the largest gains: mov-
ing beyond prefix-only verification substantially
improves both EM and ES, and performance in-
creases steadily up to the default L= 3 . In contrast,
further enlarging the scope ( L >3 ) yields only
marginal improvements, despite a rapidly growing
13

Algorithm 2: ChunkShapley: Surrogate
Shapley Attribution with Bounded Verifica-
tion
Input:In-file contextX in; ground-truth completion
Y; retrieved chunksX cc= (cc 1, . . . , cc K);
Frozen generatorG θ; surrogate scaleβ; verification
params(N v, L).
Output:Verification-selected coalition
S⋆⊆ {1, . . . , K} ; surrogate Shapley scores
{ϕi}K
i=1.
1ℓ(∅)←1
|Y|logp θ(Y|X in)
2fori←1toKdo
3ℓ({i})←1
|Y|logp θ(Y|X in,{cc i})
4∆ i←ℓ({i})−ℓ(∅)
5y i←sign(∆ i);ω i← |∆ i|
6end
7foreachS⊆ {1, . . . , K}do
8g(S)←P
j∈Sωjyj
9v sur(S)←σ(β g(S))−σ(0)
10end
11fori←1toKdo
12ϕ i←0
13foreachS⊆ {1, . . . , K} \ {i}do
14w(S)←|S|! (K−|S|−1)!
K!
15ϕ i←ϕ i+w(S) 
vsur(S∪ {i})−v sur(S)
16end
17end
18π ϕ←argsort({ϕ i},desc);
π∆←argsort({∆ i},desc)
19C ←BuildPool(π ϕ, π∆;Nv, L)
20foreachS∈ Cdo
21 ˆYS←Decode 
Gθ|X in, XS
22ComputeES( ˆYS, Y)andEM( ˆYS, Y)
23end
24S⋆←arg max S∈C 
ES(ˆYS, Y),EM( ˆYS, Y)
25returnS⋆,{ϕi}K
i=1
candidate pool and offline labeling cost. Therefore,
we adopt L= 3 as a practical default that captures
most of the benefit of combinatorial probing.
C.2 Oracle Analysis
To validate the theoretical superiority of Shapley-
based valuation over independent likelihood prob-
ing (as used in CODEFILTER ), we conducted an Or-
acle study. We calculated the best possible Edit
Similarity (ES) achievable if the model perfectly
selected chunks according to the respective valua-
tion methods (selecting top- Kchunks with score
>0).
As shown in Table 6, the Shapley-based oracle
outperforms the CODEFILTER oracle by10.45per-
centage points. This confirms that modelling chunk
interactions, like synergy and conflict, is critical for
repository-level code completion, as independent
probing fails to identify chunks that are only usefulAlgorithm 3:REPOSHAPLEYCross-file
Labeling via ChunkShapley
Input:Repository cross-file poolX out; in-file
contextX in= (X p, Xs); target spanY;
RetrieverR; frozen generatorG; chunk windoww;
strides; retrieve budgetK;
verification params (Nv, L)(as in Alg. 2); thresholds
τesandϵ.
Output:Labeled instance: retrieval label
r⋆∈ {⟨NEED⟩,⟨DONE⟩}and selection labels
(q⋆
1, . . . , q⋆
K)withq⋆
i∈ {⟨KEEP⟩,⟨DROP⟩}
1Q←X p[−w:]
2eXout←chunkize(X out;w, s)
3X cc←R(Q, eXout)[1:K]
4(S⋆,ˆYS⋆)←
ChunkShapley(X in, Y, X cc, G;N v, L)
5ˆY∅←G(X in)
6ifES( ˆYS⋆, Y)< τ esthen
7return discard instance
8end
9ifES( ˆYS⋆, Y)−ES( ˆY∅, Y)≤ϵthen
10r⋆← ⟨DONE⟩
11end
12r⋆← ⟨NEED⟩
13fori←1toKdo
14q⋆
i← ⟨KEEP⟩ifi /∈S⋆then
15q⋆
i← ⟨DROP⟩
16end
17
18end
19returnr⋆,(q⋆
1, . . . , q⋆
K)
Table 4: Hyperparameter search space and selected val-
ues (tuned on StarCoderBase-1B).
Hyperparameter Search space Selected
Learning rate{1e−5,2e−5,5e−5}2e−5
λ(λR=λS){0.2,1.0,2.0,5.0}2.0
Epochs{1,2,5}2
Warmup steps{50,100}50
when combined, for instance interface definitions
and implementations.
C.3 Sensitivity Analysis on Retrieval Budget
K
Table 7 investigates the trade-off between comple-
tion performance and inference latency by varying
the retrieval budget K(i.e., the number of candi-
date chunks processed by ChunkShapley) on SC-
Base-1B. Increasing Kexpands the search space
for complementary evidence, potentially captur-
ing more synergistic interactions. However, since
our method involves exact Shapley estimation via
subset enumeration, the computational cost grows
exponentially with K. Specifically, a small budget
(K= 7 ) yields low latency but fails to retrieve suffi-
14

Table 5:Ablation on Verification Scope ( L).Impact of the Top- Lrange used for combinatorial probing in the
post-verification stage. We report the average size of the candidate pool |C|, offline labeling latency per instance,
and performance on SC-Base-1B.
Top-LLabeling Cost Performance
Avg. Pool Size(|C|)Train Time per Sample(s) EM(%)ES(%)
0 (Prefix Only) 8.41 33 45.15 70.86
1 10.78 94 57.73 76.27
2 12.56 187 59.15 78.40
3(Default) 18.29 348 61.34 82.78
4 25.07 671 61.70 83.22
5 35.42 869 61.94 83.24
7 68.94 1528 62.13 83.59
10 (All) 225 6823 - -
Table 6:Oracle Performance Comparison.We report
the mean Best ES score achievable by selecting contexts
based on oracle labels. REPOSHAPLEY(Oracle) demon-
strates a significantly higher theoretical upper bound.
Method Oracle Best ES (%)
Full-Retrieve 71.52
CODEFILTER(Oracle) 85.23
REPOSHAPLEY(ORACLE) 95.68
cient complementary pairs, resulting in suboptimal
accuracy (52.16% EM). Conversely, increasingK
beyond 10 yieldsdiminishing returns; for instance,
expanding to K= 13 marginally improves ES by
0.51% but causes latency to explode to over 3.5
seconds due to the combinatorial complexity of
the surrogate game ( 213subsets), rendering it im-
practical. Crucially, K= 10 achieves the optimal
balance, we adopt K= 10 as the default setting
to balance interaction coverage with inference effi-
ciency.
C.4 Ablation Study on Coalition Utility
Functions
InChunkShapley, the choice of the characteristic
function v(S) is critical, as it defines the "value"
distributed among retrieved chunks. We hypoth-
esize that while task-specific metrics (like Exact
Match) align perfectly with the final objective, they
provide sparse and noisy signals for attribution. To
validate the effectiveness of our Log-likelihood-
based utility, we conduct an ablation study compar-
ing it against task-metric-based utilities.Table 7: Sensitivity analysis of the retrieval budget ( K)
on SC-Base-1B in the Infilling setting. We report Line
Completion accuracy (EM/ES) and average inference
latency. K= 10 achieves the best trade-off between
context coverage and computational cost.
Retrieval SizeRepoEval-Line Efficiency
EM ES Latency (ms)
7 52.16 70.44 513
9 58.33 75.12 833
10 (Ours) 61.34 82.78 1053
11 61.37 82.40 1924
13 61.88 81.62 3539
20 61.76 79.92 18825
Experimental Setup.We compare three defini-
tions of coalition utilityv(S):
Log-likelihood Utility (Ours):We use the
normalized token-level log-probability gain under
teacher forcing. This provides a continuous, dense
signal reflecting the model’s confidence:
vlog(S) =1
|Y||Y|X
t=1
logp θ(yt|y<t, Xin, XS)
−logp θ(yt|y<t, Xin,∅)
.
Exact Match (EM) Utility:We define utility
as the binary gain in obtaining a perfect prediction.
This signal is discrete ( ∈ {−1,0,1} ) and highly
sparse:
vEM(S) =1[EM( ˆYS, Y) = 1]−1[EM( ˆY∅, Y) = 1]
15

Table 8:Ablation study on Utility Functions.We re-
port the performance ofREPOSHAPLEYwhen trained
with Shapley labels derived from different utility defini-
tions.Log-likelihood(Ours) significantly outperforms
metric-based utilities due to the density and stability of
the teacher-forcing signal.
Utility Function Signal Type EM ES
w/v EM Binary 58.12 77.45
w/v ES Discrete-Step 59.03 78.10
w/v log(Ours) Continuous 60.50 79.07
where ˆYSis the greedy decoding result given con-
textS.
Edit Similarity (ES) Utility:We define utility
based on the improvement in surface-level similar-
ity. While continuous (0-100), ES is derived from
discrete decoding steps and is non-differentiable:
vES(S) =ES( ˆYS, Y)−ES( ˆY∅, Y)(1)
For all variants, we compute the exact Shapley
values using the respective v(S) , select the optimal
subset S⋆using the verification strategy, and train
the corresponding REPOSHAPLEYmodel.
Results and Analysis.As shown in Table 8, us-
ing log-likelihood as the utility function yields the
best performance. We attribute the inferiority of
metric-based utilities ( vEM, vES) to thesparsity
and high varianceof the signal. In code com-
pletion, a chunk might significantly improve the
model’s understanding (providing the correct vari-
able type) without immediately flipping the final
prediction to an exact match. vlogcaptures this "par-
tial credit," whereas vEMassigns zero value, lead-
ing to false negatives in attribution. Furthermore,
generation-based metrics are sensitive to decoding
dynamics, where a small change in context might
drastically alter the greedy path, causing vmetric(S)
to fluctuate wildly. In contrast, teacher-forced log-
probabilities provide a smoother and more robust
estimation of marginal contribution.
C.5 Filtering effectiveness via selective drop
and counterfactual inverse.
Following CODEFILTER (Li et al., 2025), we study
whether attribution signals can reliably separate
helpful from harmful retrieved context on RepoE-
val under the same Jaccard-based retriever. Ta-
ble 9 reports two complementary interventions:
Selective (Drop), which removes chunks labeledTable 9: Impact of the filtering policy based on dif-
ferent attribution signals.Selectivedenotes removing
chunks labeled as ⟨DROP⟩ , whileInverseretains only
those chunks (to verify the toxicity of dropped content).
StrategyCODEFILTERREPOSHAPLEY
EM ES EM ES
Selective(Drop) 49.31 57.50 54.79↑ 78.62↑
Inverse (Keep) 33.17 40.26 34.72 41.15
Figure 4: Distribution: CODEFILTER positive chunks vs.
REPOSHAPLEYselected chunks
as⟨DROP⟩ and keeps the remaining context, and
Inverse (Keep), which retains only the dropped
chunks as a counterfactual diagnostic.
REPOSHAPLEYgains markedly from selective
filtering, improving both EM and ES compared to
theCODEFILTER counterpart. In contrast, the in-
verse setting substantially degrades performance
for both methods, confirming that the dropped
chunks are predominantly low-utility (e.g., re-
dundant or misleading) rather than accidentally
filtered-out evidence. Notably, Figure 4 shows
that CODEFILTER ’s decisions are prone to brittle
single-chunk thresholds: when individual signals
are weak, it tends to label very few chunks as pos-
itive, effectively collapsing the available context.
REPOSHAPLEYinstead maintains a stable selec-
tion set, consistent with interaction-aware supervi-
sion that removes toxic context while preserving
the evidence required for accurate repository-level
completion.
C.6 Interaction-Aware Proposal.
We examine whether gains stem solely from ver-
ification by testingDelta+Verify, which replaces
Shapley candidates with single-chunk rankings
(∆i) under the same verifier. Table 10 shows RE-
POSHAPLEYachieves a higher net gain. This con-
firms that ∆scores misssynergistic chunks(weak
in isolation), creating a hard performance ceiling.
In contrast, Shapley effectively captures these high-
16

Table 10:Impact of Proposal Mechanism.Compari-
son between using Single-Chunk ∆vs. Coalition Shap-
leyϕto generate candidates for the verification step.
Proposal Method Metric Gain over Full
Delta (∆) + Verify ES 74.21
REPOSHAPLEY(ϕ) + Verify ES 82.78
potential interactive subsets, providing the verifier
with a superior candidate pool.
C.7 Abstention and Selection Analysis
To further understand RepoShapley’s decision be-
havior, we conduct an abstention analysis across
datasets and task granularities. We partition re-
trieved candidates into three outcome types by
combining the retriever score (high vs. low) with
RepoShapley’s keep or drop decision:High re-
tained(high-score kept),High discarded(high-
score dropped), andLow captured(low-score
kept). Figure 5 reports the proportions of these
categories.
Across relatively local settings, RepoShapley
stays consistent with the dense retriever while mak-
ing selective corrections. OnRepoEval Lineand
CCLEval Chunk, most high-scoring chunks are
retained (93% and 90%), yet the policy still rejects
a small fraction of high-score candidates (3–4%)
and recovers a small but non-trivial set of low-score
evidence (4–6%). This indicates that RepoShapley
does not simply follow similarity ranking; instead,
it can stably abstain from a few high-score but inef-
fective chunks and preserve occasional low-score
signals even when surface similarity is largely reli-
able.
As task difficulty increases, the policy increas-
ingly captures low-score but necessary evidence.
ForRepoEval FunctionandCCLEval Function,
the low-captured portion rises to 12% and 11%,
while the high-discarded portion remains modest
(6% and 5%). The shift is most pronounced on
RepoEval API, where RepoShapley retains only
69% of high-score chunks, discards 7% as redun-
dant/noisy, and captures 24% from the low-score
tail. This trend supports our core claim: when
cross-file interactions become more complex, Re-
poShapley can recover individually under-ranked
chunks whose utility emerges through strong in-
teractions with other context, effectively filtering
“high-score noise” while identifying “low-score but
interaction-critical” evidence.
Figure 5: Breakdown of chunk selection decisions by
RepoShapley on different benchmarks.High retained
indicates consensus between retriever and policy.High
discardedandLow capturedhighlight cases where
RepoShapley corrects the retriever’s judgment.
Figure 6: Comparison of retained cross-file con-
text lengths. RepoFormer keeps the most tokens,
CODEFILTER is the most aggressive, and RepoShapley
balances pruning and coverage.
C.8 Context Length Distribution Analysis
Beyond accuracy, we analyze efficiency by measur-
ing how many cross-file tokens are retained after
filtering. Figure 6 shows the distribution of re-
tained context lengths (in tokens) for RepoFormer,
CODEFILTER , and RepoShapley across five bench-
marks.
RepoFormer consistently retains the longest con-
texts and can exceed 12k tokens on function-level
tasks, indicating a high-recall behavior that carries
substantial redundancy. This increases both com-
putational cost and the risk of distracting evidence
during generation.
CODEFILTER sits at the other extreme: it retains
the fewest tokens by pruning aggressively, which
reduces latency but can remove weak yet necessary
dependencies.
RepoShapley lies between these two regimes.
It substantially shortens contexts relative to Re-
poFormer, suggesting effective removal of redun-
17

Table 11: Accuracy of state-of-the-art code LMs as the generation model and with REPOSHAPLEYas the policy
model for selective RAG in Infilling setting. We compare DeepSeek (Guo et al., 2025), Qwen (Hui et al., 2024),
GLM (Zeng et al., 2025), GPT-5 (OpenAI, 2025), and Claude Opus (Anthropic, 2025).For closed-source models,
we use the official APIs for generation.
Model StrategyRepoEval-Line RepoEval-API
EM ES EM ES
StarCoderBase-7BFull-Retrieve 58.26 77.79 50.38 75.01
RepoShapley 65.81 86.59 58.79 84.11
CodeLlama-13BFull-Retrieve 52.73 71.91 42.28 69.57
RepoShapley 68.89 87.11 57.66 83.41
DeepSeek R1Full-Retrieve 52.94 70.09 42.61 70.93
RepoShapley 68.79 87.92 58.96 84.07
Qwen 2.5 MaxFull-Retrieve 53.38 71.91 44.72 72.39
RepoShapley 69.51 88.17 59.14 84.58
GLM-4.5Full-Retrieve 55.93 73.04 45.27 73.21
RepoShapley 69.96 88.49 60.32 84.94
ChatGPT-5Full-Retrieve 56.51 73.73 45.94 73.82
RepoShapley 70.37 89.08 60.19 84.41
Claude Opus 4.1Full-Retrieve 58.23 74.38 47.56 75.61
RepoShapley 71.09 89.13 61.74 85.92
dant or conflicting chunks, while remaining slightly
longer than CODEFILTER . This gap is expected: Re-
poShapley tends to keep complementary chunks
that may appear low-signal in isolation but im-
prove the coalition, consistent with the abstention
analysis in Appendix C.7. Overall, RepoShapley
achieves a favorable trade-off by reducing token
overhead without sacrificing semantic coverage.
C.9 SOTA Generation Models with
REPOSHAPLEYPolicy in Infilling
Table 11 reports results when we pair REPOSHAP-
LEYwith a wide range of state-of-the-art code LMs
under the infilling setting on RepoEval-Line and
RepoEval-API. For each backbone, we compare a
standardFull-Retrievestrategy against using RE-
POSHAPLEYas a selective RAG policy while keep-
ing the same generation model.
Across all backbones, REPOSHAPLEYconsis-
tently improves both EM and ES on both bench-
marks, indicating that coalition-aware evidence fil-
tering is complementary to model scaling and re-
mains effective for both open-source and closed-
source generators. For closed-source models, we
use the official APIs for generation, while RE-POSHAPLEYis applied as an external policy to
decide whether to retrieve and which chunks to
keep.
C.10 Latency-Accuracy Trade-off Analysis
Following RepoFormer (Wu et al., 2024), we
visualize the latency-accuracy trade-off to eval-
uate the efficiency of REPOSHAPLEYacross
StarCoderBase-1B, 3B, and 7B as shown in Figure
7-9. By varying the retrieval triggering threshold tc
during inference, we control the model’s sensitivity
to external evidence.
The results demonstrate that REPOSHAPLEYes-
tablishes a superior Pareto frontier compared to
static retrieval strategies. We find that our model
can improve accuracy while also reducing latency
by skipping retrieval when the in-file context is al-
ready sufficient, and focusing retrieval on harder
cases that truly need cross-file information. Consis-
tent with prior observations, this efficiency gain is
particularly pronounced in Line and API comple-
tion tasks, where avoiding the overhead of unneces-
sary retrieval significantly lowers average latency
without compromising generation quality.
18

D Case Study
In this section, we present a case study to illus-
trate how REPOSHAPLEYperforms interaction-
aware chunk selection for repository-level code
completion in the FIM setting. The target file de-
fines utilities for extracting event start/end mark-
ers from log search results and computing event
durations. In this instance, the missing span
lies inside LogEventStats.run : after parsing
each end-marker timestamp end from end_tag
results, the code should immediately register it
into EventCollection viaadd_event_end . Af-
ter registering each end marker, the routine pro-
ceeds to handle start markers and finally calls
calculate_event_deltas . The repository con-
tains many timestamp-related helpers; however,
most retrieved evidence is only partially relevant
or unrelated (e.g., YAML formatting, tests, Ceph
helpers). Naively appending all retrieved contexts
can distract the model into rewriting timestamp
parsing rather than emitting the required event-
collection logic.
Instance (FIM).Given the in-file prefix and suf-
fix (Figure 10), the model must generate the miss-
ing span at <MID> . Concretely, the correct comple-
tion should insert an add_event_end call that uses
the current result ’sevent_id and the parsed end
timestamp. For compact presentation, Figure 10
splits the code across two columns. Therefore, the
<SFX> panel shows the subsequent lines after the
insertion point (not necessarily the immediate next
line in the source file), while preserving the original
indentation and control flow.
Retrieved top-10 cross-file chunks.We retrieve
the top-10 candidates {c1, . . . , c 10}from other files
in the same repository (Figure 11). Most candi-
dates are either unrelated or only partially relevant.
Importantly, the most helpful timestamp utilities
are split across multiple chunks ( c1, c8, c9), so that
no single chunk alone fully specifies the needed
behavior, making the evidenceinteraction-heavy.
Why the kept coalition matters.Chunks c1,c8,
andc9form a coherent timestamp-handling sub-
routine. c1andc8provide compatible datetime
parsing formats, and c9implements temporal fil-
tering logic used by the surrounding utilities.
Full-Retrieve is distracted by irrelevant utilities
(e.g., c2, c3) and hallucinates redundant timestamp-
parsing logic inside run, instead of emitting the
requiredadd_event_endregistration.Generation comparison.Figure 12 compares
the generations. REPOSHAPLEYcorrectly in-
serts the add_event_end registration and then fol-
lows the existing start-marker logic, whereas Full-
Retrieve is distracted and produces redundant pars-
ing code.
19

Figure 7: Latency-accuracy trade-off on SC-Base-1B.
Figure 8: Latency-accuracy trade-off on SC-Base-3B.
Figure 9: Latency-accuracy trade-off on SC-Base-7B.
20

<PFX>
importstatistics
fromdatetimeimportdatetime
classEventCollection(object):
"""Used to collect events found in logfiles..."""
def__init__(self):
self._events = {}
defmost_recent(self, items):
return sorted(items, key=lambdae: e["end"],
reverse=True)[0]
@property
defcomplete_events(self):
# ... (omitted for brevity if needed) ...
returncomplete
@property
defincomplete_events(self):
# ... (omitted for brevity) ...
returnincomplete
deffind_most_recent_start(self, event_id, end_ts):
""" For a given event end marker, find the most recent
start marker. """
last = None
foriteminself._events[event_id].get("heads", []):
start_ts = item["start"]
ifstart_ts <= end_ts:
if notlastorstart_ts > last["start"]:
last = item
returnlast
defadd_event_end(self, event_id, end_ts):
ifevent_idnot inself._events:
self._events[event_id] = {}
if"tails"not inself._events[event_id]:
self._events[event_id]["tails"] = [end_ts]
else:
self._events[event_id]["tails"].append(end_ts)
defadd_event_start(self, event_id, start_ts,
metadata=None,
metadata_key=None):
# ... logic to add start markers ...
pass
defcalculate_event_deltas(self):
# ... logic to calc deltas ...
passclassSearchResultIndices(object):
# ... index definitions ...
pass
classLogEventStats(object):
"""Used to identify events within logs..."""
def__init__(self, results, results_tag_prefix,
custom_idxs=None):
self.data = EventCollection()
self.results = results
self.results_tag_prefix = results_tag_prefix
# ... init logic ...
defrun(self):
""" Collect event start/end markers... """
seq_idxs = self.log_seq_idxs
end_tag = "{}-end".format(self.results_tag_prefix)
forresultinself.results.find_by_tag(end_tag):
day = result.get(seq_idxs.day)
secs = result.get(seq_idxs.secs)
end = "{} {}".format(day, secs)
end = datetime.strptime(end, "%Y-%m-%d
%H:%M:%S.%f")
<SFX>
start = "{} {}".format(day, secs)
start = datetime.strptime(start, "%Y-%m-%d
%H:%M:%S.%f")
metadata = result.get(seq_idxs.metadata)
meta_key = seq_idxs.metadata_key
event_id = result.get(seq_idxs.event_id)
self.data.add_event_start(event_id, start,
metadata=metadata,
metadata_key=meta_key)
self.data.calculate_event_deltas()
defget_top_n_events_sorted(self,max, reverse=True):
# ... sorting logic ...
returntop_n_sorted
defget_event_stats(self):
# ... stats logic ...
returnstats
<NEED>
Figure 10:FIM instance.The missing span inserts the add_event_end registration after parsing each end marker;
the subsequent start-marker handling logic continues in the suffix.
21

Retrieved Candidates Pool & Selection Decisions
<C_1>[KEEP]
ts_formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
# ... (timestamp parsing logic) ...
deffilter_by_age(cls, results, result_age_hours):
# ...
current = datetime.strptime(current, "%Y-%m-%d %H:%M:%S")
# ...
<C_2>[DROP]
ifmessageis notNone:
message =str(message).format(**fdict)
# ... message formatting utilities ...
@cached_yproperty_attr
def type(self):
""" Name of core.issues.IssueTypeBase object ... """
# ...
<C_3>[DROP]
data_file = os.path.join(dtmp,'data.txt')
# ... YAML generation for testing ...
classMyEventHandler(events.YEventCheckerBase):
def__init__(self):
super().__init__(yaml_defs_group='mygroup'...)
<C_4>[DROP]
def_override_keys(cls):return['raises']
# ... YPropertyOverride logic ...
defmessage_with_format_dict_applied(self,property=None):
# ...
<C_5>[DROP]
s = FileSearcher()
# ... Unit tests for SequenceSearchDef ...
deftest_sequence_searcher_eof(self):
# ...<C_6>[DROP]
# ... Duplicate or similar unit tests ...
self.assertEqual(len(sections), 2)
# ...
<C_7>[DROP]
""" Returns a dict of ceph versions info ... """
out = self.cli_cache['ceph_versions']
# ... Ceph daemon logic ...
<C_8>[KEEP]
classYPropertySearchBase(YPropertyMappedOverrideBase):
@classmethod
defget_datetime_from_result(cls, result):
""" This attempts to create a datetime object... """
ts = result.get(1)
# ...
ts_formats = ["%Y-%m-%d %H:%M:%S.%f", ...]
# ...
<C_9>[KEEP]
forrin sorted(_results, key=lambdai: i[0], reverse=True):
# ... filtering logic ...
defapply_constraints(self, results):
# ...
results = self.filter_by_age(results, result_age_hours)
# ...
<C_10>[DROP]
relname ='unknown'
# ... package version check ...
def_get_bind_interfaces(self,type):
""" For the given config network type ... """
# ...
Figure 11:10-chunk retrieved pool and selection.REPOSHAPLEYidentifies and keeps {c1, c8, c9}(timestamp
utilities split across chunks) while dropping unrelated evidence. (Content abbreviated for display).
Ground Truth
self.data.add_event_end(
result.get(seq_idxs.event_id),
end)
start_tag = "{}-start".format(
self.results_tag_prefix)
forresultinself.results.find_by_tag(
start_tag):
day = result.get(seq_idxs.day)
secs = result.get(seq_idxs.secs)REPOSHAPLEY(Ours)
self.data.add_event_end(
result.get(seq_idxs.event_id),
end)
start_tag = "{}-start".format(
self.results_tag_prefix)
forresultinself.results.find_by_tag(
start_tag):
day = result.get(seq_idxs.day)
secs = result.get(seq_idxs.secs)Full-Retrieve Baseline
ts = "{} {}".format(day, secs)
ts_formats = ["%Y-%m-%d..."]
for format ints_formats:
try:
end = datetime.strptime(ts,
format)
break
exceptValueError:
continue
start_tag = "{}-start".format(
self.results_tag_prefix)
forresultinself.results.find_by_tag(
start_tag):
day = result.get(seq_idxs.day)
secs = result.get(seq_idxs.secs)
Figure 12:Completion comparison.REPOSHAPLEYinserts the correct control-flow logic. Full-Retrieve is
distracted and hallucinates redundant parsing logic.
22