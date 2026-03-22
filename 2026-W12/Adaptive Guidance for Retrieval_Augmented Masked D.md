# Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models

**Authors**: Jaemin Kim, Jong Chul Ye

**Published**: 2026-03-18 12:54:50

**PDF URL**: [https://arxiv.org/pdf/2603.17677v1](https://arxiv.org/pdf/2603.17677v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves factual grounding by incorporating external knowledge into language model generation. However, when retrieved context is noisy, unreliable, or inconsistent with the model's parametric knowledge, it introduces retrieval-prior conflicts that can degrade generation quality. While this problem has been studied in autoregressive language models, it remains largely unexplored in diffusion-based language models, where the iterative denoising process introduces unique challenges for integrating retrieved context. In this work, we propose Adaptive Retrieval-Augmented Masked Diffusion (ARAM), a training-free adaptive guidance framework for Masked Diffusion Models (MDMs) in RAG settings. ARAM dynamically calibrates the guidance scale during denoising according to the Signal-to-Noise Ratio (SNR) of the distributional shift induced by retrieved context. Intuitively, the model strengthens guidance when the retrieved context provides reliable corrective evidence and suppresses it when the contextual signal is noisy or non-supportive. Extensive experiments on multiple knowledge-intensive QA benchmarks show that ARAM improves overall QA performance over competitive RAG baselines.

## Full Text


<!-- PDF content starts -->

Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models
Jaemin Kim
Graduate School of AI, KAIST
Republic of Korea
kjm981995@kaist.ac.krJong Chul Ye
Graduate School of AI, KAIST
Republic of Korea
jong.ye@kaist.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) im-
proves factual grounding by incorporating ex-
ternal knowledge into language model genera-
tion. However, when retrieved context is noisy,
unreliable, or inconsistent with the model’s
parametric knowledge, it introduces retrieval-
prior conflicts that can degrade generation qual-
ity. While this problem has been studied in
autoregressive language models, it remains
largely unexplored in diffusion-based language
models, where the iterative denoising process
introduces unique challenges for integrating
retrieved context. In this work, we propose
Adaptive Retrieval-Augmented Masked Dif-
fusion (ARAM), a training-free adaptive guid-
ance framework for Masked Diffusion Models
(MDMs) in RAG settings. ARAM dynamically
calibrates the guidance scale during denoising
according to the Signal-to-Noise Ratio (SNR)
of the distributional shift induced by retrieved
context. Intuitively, the model strengthens guid-
ance when the retrieved context provides reli-
able corrective evidence and suppresses it when
the contextual signal is noisy or non-supportive.
Extensive experiments on multiple knowledge-
intensive QA benchmarks show that ARAM
improves overall QA performance over com-
petitive RAG baselines.
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities across various nat-
ural language processing tasks (Grattafiori et al.,
2024; Hui et al., 2024). However, their reliance on
static parametric knowledge often leads to halluci-
nations and outdated responses when dealing with
dynamic or knowledge-intensive queries. Retrieval-
Augmented Generation (RAG) (Lewis et al., 2020;
Zhao et al., 2026) addresses this limitation by
grounding generation in external knowledge re-
trieved at inference time.
Recently, diffusion-based language generation
Masked Diffusion Model
Query
How many times has 
South Korea held the 
Winter Olympics?
Answer Once.
Retrieved Context
ConflictingReliable
Irrelevantor
or
Adaptive Guidance[MASK]  [MASK]  [MASK]
“Once”      “.”      [EOS][MASK]  [MASK]   [EOS]
“Once”  [MASK]   [EOS]
ARAM
Prior Logit
Conditioned Logit (Reliable)
Conditioned Logit 
 (Conflict)
Masked Diffusion Model
Query
How many times has 
South Korea held the 
Winter Olympics?
Answer Once.
Retrieved Context
ConflictingReliable
Irrelevantor
or
Adaptive Guidance[MASK]  [MASK]  [MASK]
“Once”      “.”      [EOS][MASK]  [MASK]   [EOS]
“Once”  [MASK]   [EOS]
ARAM
ARAM
Prior Logit
Conditioned Logit (Reliable)
Conditioned Logit 
 (Conflict)
Retrieved Context 𝑪 Query 𝒒
𝒍𝒐𝒈𝒊 𝒕𝒄𝒐𝒏𝒅 =𝐥𝐨𝐠𝒑𝒄𝒐𝒏𝒅 (𝒙|𝒒,𝑪) 𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 =𝐥𝐨𝐠𝒑𝒑𝒓𝒊𝒐𝒓 (𝒙|𝒒) 
(b) Ours (a) RAG
Guidance 𝝀𝒕 [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
Guidance 𝝀𝒕 Retrieved Context 𝑪 Query 𝒒
𝒍𝒐𝒈𝒊 𝒕𝒈𝒖𝒊𝒅𝒆𝒅  =𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 +𝝀𝒕(𝒍𝒐𝒈𝒊 𝒕𝒄𝒐𝒏𝒅  −𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 )Retrieved Context 𝑪 Query 𝒒
𝒍𝒐𝒈𝒊 𝒕𝒄𝒐𝒏𝒅 =𝐥𝐨𝐠𝒑𝒄𝒐𝒏𝒅 (𝒙|𝒒,𝑪) 𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 =𝐥𝐨𝐠𝒑𝒑𝒓𝒊𝒐𝒓 (𝒙|𝒒) Retrieved Context 𝑪 Query 𝒒
𝒍𝒐𝒈𝒊 𝒕𝒈𝒖𝒊𝒅𝒆𝒅  =𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 +𝝀𝒕(𝒍𝒐𝒈𝒊 𝒕𝒄𝒐𝒏𝒅  −𝒍𝒐𝒈𝒊 𝒕𝒑𝒓𝒊𝒐𝒓 )
𝑆𝑖𝑔𝑛𝑎𝑙 =𝐷𝐾𝐿(𝑝𝑐𝑜𝑛𝑑 ||𝑝𝑝𝑟𝑖𝑜𝑟 )
𝑁𝑜𝑖𝑠𝑒  =ℋ(𝑝𝑐𝑜𝑛𝑑 )+𝐷𝐾𝐿(𝑝𝑝𝑟𝑖𝑜𝑟 ||𝑝𝑐𝑜𝑛𝑑 )𝜆𝑡∝𝑆𝑖𝑔𝑛𝑎𝑙
𝑁𝑜𝑖𝑠𝑒
Signal -to-Noise  (SNR) Based Step -wise  and Token -wise
Figure 1:Overview of ARAM.ARAM mitigates
knowledge conflict via step-wise and token-wise adap-
tive guidance driven by the Signal-to-Noise Ratio.
has emerged as a promising alternative to autore-
gressive models. In particular, Masked Diffusion
Models (MDMs) generate text through an iterative
denoising process that progressively fills masked
tokens while leveraging bidirectional attention (Nie
et al., 2025; Ye et al., 2025). Compared to autore-
gressive models, this generation paradigm enables
parallel token updates and iterative refinement of
intermediate predictions.
While recent research has focused on improving
the generative capabilities of MDMs to narrow the
gap with autoregressive models (Lou et al., 2023),
applying RAG to MDMs remains notably under-
explored. Preliminary studies indicate that MDM-
based RAG demonstrates remarkable in-context
learning capabilities that often surpass those of au-
toregressive models (Dai et al., 2026; Yu et al.,
2026). Despite this potential, existing approaches
predominantly assume that the retrieved documents
are reliable. In realistic RAG settings, however, the
retrieved context is often noisy, partially irrelevant,
or even contradictory to the model’s internal knowl-
edge.
A fundamental challenge is how to effectively
ground the iterative denoising process in retrieved
knowledge, especially when this external informa-
tion conflicts with the model’s parametric prior.arXiv:2603.17677v1  [cs.CL]  18 Mar 2026

When such heterogeneous context is injected into
the generation process, the model must dynami-
cally decide how much it should trust the retrieved
context versus its parametric prior. We refer to this
tension between retrieved context and parametric
knowledge asknowledge conflict(Shi et al., 2024;
Wang et al., 2025b; Yuan et al., 2024). This raises
a key question:
How should a diffusion language model de-
termine when to trust retrieved context and
when to rely on its parametric knowledge?
To address this question, we proposeAdap-
tive Retrieval-Augmented Masked Diffusion
(ARAM), an adaptive guidance framework that
self-calibrates the guidance scale at each denois-
ing step and token position. To effectively inject
conditional information and control the generation
quality in MDMs, we adopt Classifier-Free Guid-
ance (CFG) (Ho and Salimans, 2022) for RAG.
However, applying a static guidance scale under
retrieved contexts leads to a critical dilemma: (1)
over-guidance on noisy context, which causes hallu-
cinations, and (2) under-guidance on valid context,
which fails to correct the flawed prior.
Our key insight is that the influence of retrieved
context can be interpreted as a distributional shift
from a retrieval-free prior to a retrieval-conditioned
posterior. Specifically, we analyze the interaction
through an information-theoretic lens, and derive a
Signal-to-Noise Ratio (SNR) of this distributional
shift to adapt the guidance strength. Consequently,
when retrieved context consistently supports the
conditional distribution, guidance can be strength-
ened; when the retrieved context is uncertain or
non-supportive, guidance can be attenuated to pre-
vent the over-amplification of noise. We illustrate
the overview of our proposed method in Fig. 1.
We evaluate our adaptive guidance mechanism
on knowledge-intensive QA benchmarks. Empir-
ical results demonstrate that our method outper-
forms both static CFG and baseline MDMs, achiev-
ing significant improvements in exact matching
(EM) and F1 scores for MDM-based RAG. Our
main contributions are as follows:
•We propose ARAM, an adaptive CFG frame-
work for MDMs in RAG, which dynamically
modulates guidance scale based on the SNR.
•We analyze the interaction between retrievalcontext and parametric prior in diffusion-
based generation through an information-
theoretic lens and derive a guidance rule moti-
vated by signal and uncertainty.
•Extensive experiments across multiple QA
benchmarks validate that our method im-
proves EM and F1 over other baselines and
mitigates retrieval-prior conflicts.
2 Related Work
2.1 Knowledge Conflict in RAG
While Retrieval-Augmented Generation (RAG) im-
proves factual grounding by incorporating exter-
nal context, it can also introduce knowledge con-
flict (Longpre et al., 2021) between the model’s
internal knowledge and the retrieved external con-
text, leading to degraded performance and hallu-
cinations (Cheng et al., 2024). To mitigate these
conflicts, current approaches rely on external con-
text assessors (Choi et al., 2025), multi-agent de-
bates (Wang et al., 2025b), or contrastive decod-
ing (Shi et al., 2024; Wang et al., 2025a). Among
these, CAD (Shi et al., 2024) is most closely related
to our work, as it contrasts output distributions with
and without external context to penalize reliance
on parametric priors. However, applying a static
contrastive penalty often overcorrects when the re-
trieved context already aligns with the model’s para-
metric knowledge, inadvertently degrading perfor-
mance on non-conflicting data (Yuan et al., 2024).
To address this overcorrection, dynamic decoding
strategies like AdaCAD (Wang et al., 2025a) and
COIECD (Yuan et al., 2024) have emerged. While
effective, these approaches are fundamentally de-
signed for autoregressive language models and can-
not be trivially extended to the non-causal, iterative
denoising dynamics of diffusion-based language
models. Motivated by the need for dynamic adjust-
ment (Wang et al., 2025a; Yuan et al., 2024), we
introduce a discrete-diffusion-specific framework
that modulates the guidance scale at each denoising
step using a Signal-to-Noise Ratio (SNR).
2.2 Classifier-Free Guidance in MDMs
Classifier-Free Guidance (CFG) (Ho and Salimans,
2022) is widely used to strengthen conditional gen-
eration by combining conditional and prior score
functions. Since CFG was originally formulated
for continuous spaces, defining an analogous pro-
cedure for discrete diffusion is nontrivial, as the
discrete state space is inherently non-differentiable.

Recent theoretical works (Schiff et al., 2024; Ro-
jas et al., 2025) have formalized the validity of
CFG at the logit level in discrete diffusion models,
but standard CFG applies a static guidance scale
throughout the entire denoising process. Recog-
nizing the dynamic uncertainty during generation,
recent methods like A-CFG (Li et al., 2025) dynam-
ically alter the unconditional input by re-masking
low-confidence tokens. Distinct from modifying
the input formulation, our approach directly cali-
brates the guidance itself, preventing over-guidance
on noisy context and under-guidance on valid cor-
rections.
2.3 RAG in Masked Diffusion Models
Recent studies demonstrate the potential of inte-
grating RAG with MDMs to enhance knowledge
utilization beyond autoregressive (AR) models. For
example, recent analysis on Attention Floating (Dai
et al., 2026) reveals that MDMs exhibit distinc-
tive attention dynamics, where dominant attention
tokens are dispersed across positions and denois-
ing steps. This behavior allows MDMs to effec-
tively track relevant context, resulting in stronger
utilization of external knowledge than AR mod-
els. SPREAD (Yu et al., 2026) further shows that
MDMs exhibit a strong reliance on contextual in-
formation and improve answer precision by enforc-
ing query-level semantic coherence. Despite this
potential, integrating RAG into MDMs remains un-
explored. To the best of our knowledge, our work
is the first to address knowledge conflict in MDM-
based RAG through an intrinsic step-wise adaptive
guidance mechanism.
3 Preliminaries
Masked Diffusion Language Models.Masked
Diffusion Models (MDMs) generate text through
an iterative denoising process over discrete token
sequences. Let x0= (x 1, . . . , x L)denote the tar-
get token sequence of length L, where each token
xibelongs to a vocabulary V. The forward pro-
cessq(xt|xt−1)gradually corrupts the sequence
over discrete timesteps t∈ {1, . . . , T} by replac-
ing a subset of tokens with a special mask token
[MASK] . The reverse process is parameterized by
a neural network pθ, which predicts token distribu-
tionspθ(x0|xt)for the masked positions. The state
transition to the previous step xt−1is then sampledusing the posterior distribution:
pθ(xt−1|xt) =X
x0q(xt−1|xt,x0)pθ(x0|xt).(1)
In MDMs, this unmasking process is typically exe-
cuted in parallel across multiple masked positions
based on token-level confidence. For notational
simplicity, the following definitions are written for
a single masked position, where xtdenotes the
noised token at timestep tandxdenotes the token
at the target position.
Retrieval-Augmented Generation.In Retrieval-
Augmented Generation (RAG), the model condi-
tions generation on a user query qand a set of K
retrieved documents C={d 1, . . . , d K}. The re-
trieved context Cis concatenated with the query
and provided as input to the language model. We
denote this conditional token distribution as:
pcond(x) :=p θ(x|x t, q,C).(2)
Conversely, when the model relies on its parametric
prior without retrieval context, the distribution is:
pprior(x) :=p θ(x|x t, q).(3)
Classifier-Free Guidance in Discrete Diffusion.
While Classifier-Free Guidance (CFG) was origi-
nally designed for continuous diffusion by extrap-
olating score differences, CFG in discrete state
spaces operates directly on the prediction logits.
Letℓcondandℓpriordenote the logits corresponding
topcondandpprior, respectively. The guided log-
its are computed as follows, where λ≥0 is the
guidance scale:
ℓλ=ℓ prior+λ(ℓ cond−ℓ prior).(4)
4 Method
In this section, we present the Adaptive Retrieval-
Augmented Masked Diffusion (ARAM) frame-
work. We first mathematically define the knowl-
edge conflict and the limitations of static guidance
(Sec. 4.1), and then interpret CFG as an exponen-
tial tilting family (Sec. 4.2). Finally, we derive an
adaptive guidance rule based on an information-
theoretic Signal-to-Noise Ratio (SNR) (Sec. 4.3
and Sec. 4.4).
4.1 Dilemma of Static Guidance
In standard discrete CFG (Eq. (4)), the guidance
scale λuniformly amplifies the logit difference

across all tokens and diffusion steps. However,
in realistic RAG settings, the retrieved context C
introduces varying degrees of reliability, creating
a knowledge conflict between the parametric prior
and external evidence (Longpre et al., 2021). We
can categorize these generation dynamics into three
distinct scenarios:
1.Reliable Context:When Ccontains highly
relevant and accurate supporting facts, pcond
confidently shifts toward the correct tokens.
This produces a large divergence between
pcondandppriorwith low uncertainty. In this
scenario, a larger λis required to fully incor-
porate the retrieved facts and update missing
or outdated parametric knowledge.
2.Irrelevant Context:When Cis topically un-
related to the query, the model largely ignores
the external context, resulting in minimal dis-
tributional divergence ( pcond≈p prior). The
guidance scale λshould remain close to zero
to prevent unnecessary perturbation of the gen-
eration process.
3.Conflicting Context:When Ccontains mis-
leading, unreliable, or contradictory informa-
tion relative to the query or the model prior,
pcond can deviate substantially from pprior.
However, this deviation is typically accompa-
nied by high uncertainty or internal inconsis-
tency. If a large λis applied, it overwrites the
parametric prior ppriorand exacerbates halluci-
nations. Therefore, λmust be adaptively sup-
pressed considering the uncertainty ofp cond.
To resolve these varying dynamics, the model re-
quires an intrinsic, token-level mechanism to evalu-
ate the trustworthiness of the condition Cand adap-
tively modulateλ.
4.2 Information-Theoretic View of CFG
We consider an MDM generating a discrete token
sequence. To quantify the influence of the retrieved
context Cat diffusion step t, we define thecontext
scoreas a log-likelihood ratio:
s(x;C) := logpcond(x)
pprior(x).(5)
Using this term with Eq. (4), the guided distribution
pλ(x)∝p cond(x)λpprior(x)1−λcan then be written
as:
pλ(x) =pprior(x) exp(λs(x;C))
Zλ,(6)whereZ λ=P
x∈Vpprior(x) exp(λs(x;C)).
To quantify the influence of retrieved context,
we refer to the retrieval information gain as:
IGt:=D KL(pcond∥pprior) =E pcond[s(x;C)].(7)
This quantity measures how strongly the retrieved
context alters the model’s posterior belief. If the
retrieved context Cis treated as a random variable
produced by the retriever, the Conditional Mutual
Information (CMI) satisfies
I(x0;C |x t) =E C[DKL(pcond∥pprior)] =E C[IGt].
(8)
Thus, IGtserves as an information-theoretic signal
measuring the usefulness of retrieved context.
4.3 Lower Bound and the Ideal SNR
To understand the optimal choice of λ, we relate
the information gain to the Donsker-Varadhan (DV)
variational representation (Donsker and Varadhan,
1983). All proofs are provided in Appendix A.
Theorem 1(DV Variational Lower Bound).For
anyλ, the retrieval information gain satisfies:
IGt≥ L(λ) :=λE pcond[s(x;C)]−logZ λ.(9)
Corollary 1.1(Global Maximizer of the DV
Bound).Let L(λ) be defined in Eq. (9). Then
L(1) =D KL(pcond∥pprior). Hence, λ= 1 maxi-
mizes the variational lower boundL(λ).
If the retrieved context Cwere a perfect oracle,
Corollary 1.1 suggests that setting λ= 1 would
be globally optimal. However, in practical RAG
settings (Sec. 4.1), the retriever may return noisy
or conflicting documents in C. Therefore, setting
λ= 1 assumes full trust in pcondand leaves the
generation highly vulnerable to hallucinations.
To mitigate this risk, we treat the prior distri-
bution pprior(where λ= 0 ) as a safe parametric
anchor. Instead of jumping to the global maximum
atλ= 1 , we analyze the local geometry around
our anchor ( λ= 0 ) by taking a second-order Taylor
expansion as:
L(λ)≈λ 
Epcond[s]−E pprior[s]
| {z }
Signal−λ2
2Varpprior(s)
|{z}
Noise.
(10)
using the derivatives L′(0) =E pcond[s]−E pprior[s]
andL′′(0) =−Var pprior(s). Maximizing this con-
cave approximation yields
λ∗=Signal
Noise=Epcond[s]−E pprior[s]
Varpprior(s).(11)

(a)Var pprior(s)
 (b)H(p cond)
Figure 2:Comparison of noise proxies during the de-
noising process. (a)The variance of the log-likelihood
ratiosattenuates the guidance scale even when correc-
tion is required.(b)In contrast, conditional entropy
provides a stable measure of uncertainty, more clearly
distinguishing between informative and noisy contexts.
This equation naturally forms the Signal-to-Noise
Ratio (SNR): an expected information gain (Sig-
nal) penalized by the dispersion introduced by the
context (Noise). Detailed proofs are provided in
Appendix A.
4.4 Practical Adaptive Guidance
Although Eq. (11) provides a theoretically optimal
λ∗, directly using Varpprior(s)as the noise term is
unstable in masked diffusion decoding. As shown
in Fig. 2a, the variance often becomes larger for
Gold Doc(documents containing the correct an-
swer) than forNoise Doc(documents not contain-
ing the answer), which undesirably suppresses λt
even when the retrieved context is informative. In
addition, the variance exhibits substantial step-wise
fluctuations, making the resulting guidance scale
unstable across denoising steps. Experimental de-
tails are provided in Sec. 5.4.
To obtain a more stable estimate of the disper-
sion introduced by the context, we instead use the
conditional entropy H(p cond)as the noise proxy.
This choice is motivated by prior observations that
natural language generation typically lies within
a narrow entropy region, and deviations from this
region indicate conflicting or uncertain contextual
signals (Arora et al., 2023; Yuan et al., 2024). As
shown in Fig. 2b, when the retrieved context pro-
vides clear and consistent information, the condi-
tional distribution becomes concentrated and its
entropy decreases. Conversely, when the retrieved
context introduces ambiguity or contradiction, the
conditional distribution becomes more diffuse, re-
sulting in higher entropy. Accordingly, we use
H(p cond)as a practical surrogate for contextual
noise, which empirically preserves the desired be-
havior and SNR structure of the optimal guidance.
In practice, for each denoising step t, we com-Algorithm 1Adaptive Retrieval-Augmented
Masked Diffusion (ARAM)
Require: Query q, Retrieved context C, Masked Diffusion
Model pθ, Timesteps T, Maximum Guidance Scale
λmax, Sensitivityβ, A Small Positiveϵ
1: Initializex T←([MASK], . . . ,[MASK])
2:fort=Tto1do
3:p prior, pcond←pθ(·|xt, q), p θ(·|xt, q,C)
4:forEach masked positiondo
5: Signal←D KL(pcond∥pprior)+D KL(pprior∥pcond)
6: Noise← H(p cond)
7:λ t←λ max·tanh
β·Signal
Noise+ϵ
8: Computeℓ guided using Eq. (13)
9:end for
10: Sample xt−1using the guided logits ℓguided according
to the unmasking policy
11:end for
12:returnGenerated sequencex 0
pute the following signal and noise proxies:
Signal :=E pcond[s]−E pprior[s]
=D KL(pcond∥pprior) +D KL(pprior∥pcond),
Noise :=H(p cond) =−X
x∈Vpcond(x) logp cond(x).
To ensure stability and controllability, we scale the
SNR:
λt=λ max·tanh
β·Signal
Noise +ϵ
,(12)
where λmaxdefines the maximum guidance scale,
βcontrols the sensitivity to the SNR, the tanh
function and ϵ >0 ensure numerical stability. Fi-
nally, we apply this token-wise adaptive λtto the
logit-space CFG interpolation:
ℓguided (x) =ℓ prior(x) +λ t 
ℓcond(x)−ℓ prior(x)
.
(13)
This adaptive guidance is applied independently
to each token at every denoising step. The com-
plete procedure for our proposed ARAM method
is detailed in Algorithm 1.
5 Experiments and Results
5.1 Experimental Protocol
Datasets and evaluation metrics.Following
Dai et al. (2026), we evaluate our method on
five diverse knowledge-intensive QA datasets:
NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi
et al., 2017), MARCO QA (Bajaj et al., 2016a),
HotpotQA (Yang et al., 2015), and T-REx (Elsahar
et al., 2018). These datasets cover a wide range of

Table 1: Performance comparison across QA benchmarks. Bold indicates the best score within each backbone.
MethodOpen-Domain QA Multi-hop QA Slot Filling
NQ TriviaQA MARCO QA HotpotQA T-REx
EM F1 EM F1 EM F1 EM F1 EM F1
Autoregressive Models
LLaMA 20.5 30.9 64.4 69.3 1.4 14.7 15.0 22.8 34.2 39.6
+ RAG (Lewis et al., 2020) 31.9 48.2 71.7 78.3 2.4 19.0 21.2 31.4 30.3 35.3
Qwen 14.9 25.2 48.1 52.7 1.1 11.8 21.2 29.0 32.6 36.5
+ RAG (Lewis et al., 2020) 31.3 46.6 70.8 76.6 2.3 19.0 24.2 33.7 30.1 34.7
Masked Diffusion Models - LLaDA
LLaDA 9.1 15.9 27.7 32.0 1.1 9.7 15.7 21.5 23.9 26.4
+ RAG (Lewis et al., 2020) 17.5 31.9 44.8 55.1 1.425.66.5 15.6 17.8 23.8
+ CAD (Shi et al., 2024) 1.4 16.6 7.1 23.9 0.4 24.7 0.3 9.0 1.3 7.4
+ ADACAD (Wang et al., 2025a) 9.3 18.2 27.1 33.6 1.2 14.3 15.0 21.4 21.7 24.7
+ COIECD (Yuan et al., 2024) 13.0 28.4 37.4 49.2 0.9 25.4 7.1 15.7 11.1 17.1
+ SPREAD (Yu et al., 2026) 19.3 34.4 51.7 61.0 1.7 25.2 12.6 21.2 18.0 24.0
+ A-CFG (Li et al., 2025) 19.1 33.9 51.8 61.2 1.4 24.5 12.6 21.5 18.235.9
+Ours 30.3 43.2 67.7 73.1 2.5 20.0 20.7 28.9 31.4 34.8
Masked Diffusion Models - Dream
Dream 14.9 24.9 36.7 43.5 1.9 11.0 16.9 26.2 30.4 35.1
+ RAG (Lewis et al., 2020) 35.9 50.5 72.7 78.4 2.8 17.5 25.0 35.2 29.7 35.0
+ CAD (Shi et al., 2024) 23.9 40.9 55.5 67.0 1.7 20.9 14.1 24.8 19.3 26.8
+ ADACAD (Wang et al., 2025a) 37.4 50.7 72.5 77.4 2.921.724.3 34.5 31.4 36.7
+ COIECD (Yuan et al., 2024) 30.2 45.9 68.5 75.4 2.721.719.1 29.4 24.4 30.6
+ SPREAD (Yu et al., 2026) 35.9 50.1 73.7 78.5 2.9 17.8 25.0 34.734.5 38.0
+ A-CFG (Li et al., 2025) 34.0 49.2 70.6 77.8 2.6 17.4 24.2 34.6 28.1 34.1
+Ours 37.7 51.4 74.4 79.5 3.3 18.0 25.8 36.1 30.0 35.5
scenarios, including open-domain QA (NQ, Triv-
iaQA, MARCO QA), multi-hop reasoning (Hot-
potQA), and slot filling (T-REx), making them suit-
able for evaluating knowledge utilization in lan-
guage models. For all datasets, we adopt Exact
Match (EM) and F1 score as our primary evalua-
tion metrics. We use bge-large (Xiao et al., 2024) to
retrieve documents from the MS MARCO 2.1 (Ba-
jaj et al., 2016b) following Li et al., 2024.
Models and baselines.We compare our method
across Autoregressive (AR) models and Masked
Diffusion Models (MDMs). For AR models, we
utilize LLaMA-3.1-8B-Instruct (Grattafiori et al.,
2024) and Qwen2.5-7B-Instruct (Hui et al., 2024).
For MDMs, we evaluate LLaDA-8B-Instruct (Nie
et al., 2025) and Dream-v0-Instruct-7B (Ye et al.,
2025). We benchmark ARAM against the follow-
ing decoding strategies, which can be broadly cate-
gorized into AR-adapted contrastive methods and
MDM-specific generation mechanisms:
•Standard RAG:The conventional generation
method for MDMs utilizing the logit condi-
tioned on the retrieved contextℓ cond(x).
•AR-Based Baselines:We adapt AR-based
methods to the diffusion generation pro-
cess. These include CAD (Shi et al., 2024),COIECD (Yuan et al., 2024), and ADA-
CAD (Wang et al., 2025a).
•MDM-Specific Baselines:We compare
against guidance mechanisms tailored for dis-
crete diffusion. These include SPREAD (Yu
et al., 2026), which guides token selection to
maintain semantic coherence, and A-CFG (Li
et al., 2025), which dynamically re-masks low-
confidence tokens to construct a localized un-
conditional input for more effective guidance.
5.2 Benchmark Results
Table 1 presents the overall EM and F1 scores
across the five QA benchmarks. The results demon-
strate the strong capability of our proposed method
in enhancing MDM-based RAG. Compared to the
standard MDM-based RAG (which uses a fixed
guidance scale λ= 1 ), our adaptive guidance im-
proves overall EM and F1 trends.
Applying AR-specific dynamic decoding meth-
ods (CAD, COIECD, and ADACAD) to MDMs
results in suboptimal performance. Notably, CAD
exhibits consistent degradation in our experiments,
suggesting that uniform contrastive guidance can
over-correct in MDM-RAG settings. While ADA-
CAD dynamically adjusts the guidance scale
across generation steps, it applies a uniform scale

Table 2: Categorization rates of retrieval-prior interac-
tion outcomes.
Method Positive (↑) Negative (↓) Cons. Correct Cons. Wrong
LLaDA
RAG 0.135 0.051 0.040 0.774
Ours 0.234 0.022 0.069 0.675
Dream
RAG 0.259 0.049 0.100 0.592
Ours 0.266 0.038 0.111 0.585
Table 3: Performance comparison across subsets strati-
fied by retrieval context quality.
MethodGold Non-Answering Irrelevant
LLaDA Dream LLaDA Dream LLaDA Dream
RAG 24.2 49.7 0.77 1.54 0.00 0.00
Ours 41.7 52.0 1.92 1.92 0.00 0.00
across tokens. This limits performance in MDMs,
where token-wise uncertainty varies across posi-
tions and denoising steps.
We also compare our method against SPREAD
and A-CFG tailored for MDMs. While SPREAD
and A-CFG generally improve upon static RAG
by intervening in the token unmasking process,
our method achieves higher overall performance
across most benchmarks. Although SPREAD
shows higher scores in the T-REx slot-filling task
on Dream, our method outperforms the baselines in
open-domain and multi-hop reasoning tasks. These
results show that calibrating logit-space guidance
via SNR provides a strong alternative to re-masking
strategies, achieving better overall performance.
5.3 Analysis of Retrieval-Prior Conflict
To validate how ARAM resolves the knowledge
conflict between the parametric prior and retrieved
context, we analyze both model behavior and re-
trieval context quality. Detailed settings are pro-
vided in Appendix B.
First, we analyze the interaction between re-
trieval context and prior knowledge by compar-
ing retrieval-based generation with the retrieval-
free baseline. We categorize the outcomes into
four cases:Positive(baseline incorrect →retrieval-
based correct),Negative(baseline correct →
retrieval-based incorrect),Consistently Correct,
andConsistently Wrong. As shown in Table 2,
ARAM increases thePositiverate while reducing
theNegativerate for both models compared to
static RAG. This indicates that ARAM strengthens
guidance when retrieval provides useful corrective
evidence while suppressing it when the retrieved
(a) Average guidance trajectory.
(b) Token-level guidance heatmap.
Figure 3:Dynamics of adaptive guidance.(a)Average
guidance scale ( λt) per step. ARAM maintains high
guidance for gold documents but suppresses λtfor noisy
ones.(b)Token-wise guidance heatmap. Black squares
mark unmasking positions, demonstrating a right-to-left
generation tendency. ARAM assigns larger guidance to
positions that align with the unmasking pattern.
context conflicts with the parametric prior.
Second, we stratify the evaluation samples by re-
trieval context quality. We first defineGoldas cases
where the retrieved context contains the answer.
For the remaining samples (equivalent toNoise Doc
in Sec. 4.4), we further distinguishNon-Answering
as cases where the retrieved context is related to the
query but does not provide sufficient evidence for
the correct answer, andIrrelevantas cases where
the retrieved context is off-topic. Table 3 shows that
ARAM improves over static RAG in both theGold
andNon-Answeringsettings. OnGoldsamples,
ARAM more effectively incorporates valid exter-
nal evidence, mitigating the under-guidance prob-
lem of static RAG. On theNon-Answeringsubset,
ARAM avoids over-amplifying unreliable contex-
tual signals. On theIrrelevantsubset, ARAM does
not improve over static RAG, indicating that adap-
tive guidance cannot recover performance when
the retrieved context is entirely uninformative.
Taken together, these results show that adap-
tive SNR-based guidance better resolves the
dilemma of static guidance under knowledge con-
flict. ARAM reduces harmful retrieval-induced
changes while preserving beneficial corrections and

Table 4: Ablation study on λmaxandβ. Bold denotes
the best score and underlined represents second-best.
λmax βLLaDA Dream
EM F1 EM F1
1.0
1.024.9 36.1 33.9 43.6
2.0 13.9 27.3 31.4 42.4
3.0 9.1 22.8 26.3 38.9
1.00.1 30.4 39.3 33.5 42.0
0.3 28.4 38.7 34.343.4
0.5 26.9 37.7 34.2 43.5
0.7 26.1 37.0 34.0 43.5
1.0 24.9 36.1 33.943.6
2.0 22.3 34.1 33.6 43.3
remains effective when the retrieved context is se-
mantically related but non-supportive.
5.4 Dynamics of Adaptive Guidance
To analyze the behavior of the proposed adaptive
guidance, we monitor the step-wise and token-wise
guidance scales during the denoising process, as
shown in Fig. 3. We sampled 200 instances from
the NQ dataset for bothGold Doc(documents con-
taining the answer) andNoise Doc(documents not
containing the answer) settings using the LLaDA
model. At each timestep t, we recorded the token-
wise averageλ tvalues.
As illustrated in Fig. 3a, the step-wise average
guidance trajectory aligns with the SNR formula-
tion. When provided with aGold Doc, the model
maintains a larger guidance scale. In contrast, when
confronted with aNoise Doc, the guidance scale
is decreased, effectively mitigating the risk of the
model hallucinating based on irrelevant context.
Furthermore, we visualize the token-level guid-
ance heatmap to examine the spatial distribution of
λt. In Fig. 3b, LLaDA exhibits a right-to-left un-
masking pattern. Specifically, the highest guidance
values cluster adjacent to recently unmasked to-
kens. Since newly generated tokens provide strong
local context for predicting nearby masked tokens,
this spatial distribution suggests that ARAM allo-
cates stronger guidance to positions that remain
uncertain near recently resolved tokens.
5.5 Ablation Study
Effect of guidance hyperparameters.We inves-
tigate the impact of the maximum guidance scale
λmaxand the sensitivity parameter βon generation
quality. As shown in Table 4, varying βwhile fix-
ingλmax= 1.0 reveals that the two models exhibit
different sensitivities to the context signal. Specif-
ically, Dream achieves balanced performance atTable 5: Comparison of noise proxies. Average EM and
F1 scores are reported.
Noise ProxyLLaDA Dream
EM F1 EM F1
Varpprior(s) 20.3 32.5 33.5 43.2
H(p cond) 24.9 36.1 33.9 43.6
Table 6: Performance on LLaDA 1.5.
Method EM F1
LLaDA 1.5 15.8 22.0
LLaDA 1.5 + RAG 17.6 30.6
LLaDA 1.5 + Ours 24.2 35.9
β= 0.5 , while LLaDA achieves optimal results at
β= 0.1.
Choice of noise proxy.To validate the use of
conditional entropy H(p cond)as a noise proxy, we
compare it against the score variance Varpprior(s)
derived in Eq. (11). For a controlled comparison,
we set λmax= 1.0 andβ= 1.0 for both prox-
ies. As shown in Table 5, the entropy proxy yields
higher generation quality for both models. This
performance gap stems from the behavior of noise
proxies during the unmasking process (Fig. 2a),
demonstrating that the conditional entropy provides
a more stable measure of uncertainty, resulting in
robust guidance. Additional ablation studies on
guidance scale design are provided in Appendix C.
Scalability across model configurations.To ver-
ify the generalizability of ARAM, we evaluate the
framework on a more recent architecture, LLaDA
1.5 (Zhu et al., 2025). As shown in Table 6, ARAM
improves the EM and F1 scores over the standard
RAG baseline using λmax= 1.0, β= 1.0 . This
outcome indicates that our SNR-based adaptive
guidance mechanism functions consistently across
different model variants.
6 Conclusion
In this work, we introduce Adaptive Retrieval-
Augmented Masked Diffusion (ARAM), a decod-
ing framework designed to address knowledge con-
flict in MDM-based RAG. We propose an adaptive
guidance rule based on a Signal-to-Noise Ratio,
which suppresses guidance when the retrieved con-
text is conflicting or uninformative. Extensive ex-
periments across QA benchmarks demonstrate that
ARAM improves robustness under retrieval-prior
conflict and demonstrates strong improvements
over competitive MDM-based RAG baselines.

Limitations
While our findings demonstrate that ARAM effec-
tively mitigates knowledge conflict in MDM-based
RAG, we acknowledge several limitations. Our
empirical claims are scoped to English-language,
knowledge-intensive QA tasks using open-source
MDMs. It remains an open question whether our
method translates effectively to open-ended gen-
eration tasks or other decoding strategies. From a
practical perspective, computing the guidance scale
across the entire vocabulary at every denoising step
introduces a computational overhead during infer-
ence. As quantified in Appendix C, ARAM incurs
additional inference latency compared to standard
CFG due to these intermediate calculations. Due
to this overhead, it remains an important consid-
eration for deploying ARAM in highly latency-
sensitive production environments.
References
Kushal Arora, Timothy J O’Donnell, Doina Precup, Ja-
son Weston, and Jackie CK Cheung. 2023. The stable
entropy hypothesis and entropy-aware decoding: An
analysis and algorithm for robust natural language
generation.arXiv preprint arXiv:2302.06784.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, and 1
others. 2016a. Ms marco: A human generated ma-
chine reading comprehension dataset.arXiv preprint
arXiv:1611.09268.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, and 1
others. 2016b. Ms marco: A human generated ma-
chine reading comprehension dataset.arXiv preprint
arXiv:1611.09268.
Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang,
and William Yang Wang. 2024. Understanding the
interplay between parametric and contextual knowl-
edge for large language models.arXiv preprint
arXiv:2410.08414.
Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee.
2025. Conflict-aware soft prompting for retrieval-
augmented generation. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 26969–26983.
Xin Dai, Pengcheng Huang, Zhenghao Liu, Shuo Wang,
Yukun Yan, Chaojun Xiao, Yu Gu, Ge Yu, and
Maosong Sun. 2026. Revealing the attention float-
ing mechanism in masked diffusion models.arXiv
preprint arXiv:2601.07894.Monroe D Donsker and SR Srinivasa Varadhan. 1983.
Asymptotic evaluation of certain markov process ex-
pectations for large time. iv.Communications on
pure and applied mathematics, 36(2):183–212.
Hady Elsahar, Pavlos V ougiouklis, Arslen Remaci,
Christophe Gravier, Jonathon Hare, Frederique Lafor-
est, and Elena Simperl. 2018. T-rex: A large scale
alignment of natural language with knowledge base
triples. InProceedings of the Eleventh International
Conference on Language Resources and Evaluation
(LREC 2018).
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Jonathan Ho and Tim Salimans. 2022. Classifier-
free diffusion guidance.arXiv preprint
arXiv:2207.12598.
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang,
Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun
Zhang, Bowen Yu, Keming Lu, and 1 others. 2024.
Qwen2. 5-coder technical report.arXiv preprint
arXiv:2409.12186.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Pengxiang Li, Shilin Yan, Joey Tsai, Renrui Zhang,
Ruichuan An, Ziyu Guo, and Xiaowei Gao.
2025. Adaptive classifier-free guidance via dy-
namic low-confidence masking.arXiv preprint
arXiv:2505.20199.
Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan,
Shuo Wang, Shi Yu, Zheni Zeng, Hao Chen,
Ge Yu, Zhiyuan Liu, and 1 others. 2024. Rag-
ddr: Optimizing retrieval-augmented generation us-
ing differentiable data rewards.arXiv preprint
arXiv:2410.13509.

Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. Entity-based knowledge conflicts in question
answering. InProceedings of the 2021 conference on
empirical methods in natural language processing,
pages 7052–7063.
Aaron Lou, Chenlin Meng, and Stefano Ermon.
2023. Discrete diffusion modeling by estimating
the ratios of the data distribution.arXiv preprint
arXiv:2310.16834.
Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang,
Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong
Wen, and Chongxuan Li. 2025. Large language dif-
fusion models.arXiv preprint arXiv:2502.09992.
Kevin Rojas, Ye He, Chieh-Hsin Lai, Yuta Takida, Yuki
Mitsufuji, and Molei Tao. 2025. Theory-informed
improvements to classifier-free guidance for discrete
diffusion models.arXiv preprint arXiv:2507.08965.
Yair Schiff, Subham Sekhar Sahoo, Hao Phung,
Guanghan Wang, Sam Boshar, Hugo Dalla-torre,
Bernardo P de Almeida, Alexander Rush, Thomas
Pierrot, and V olodymyr Kuleshov. 2024. Simple
guidance mechanisms for discrete diffusion models.
arXiv preprint arXiv:2412.10193.
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia
Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024.
Trusting your evidence: Hallucinate less with context-
aware decoding. InProceedings of the 2024 Confer-
ence of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (Volume 2: Short Papers), pages
783–791.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025a. Adacad: Adaptively decoding
to balance conflicts between contextual and paramet-
ric knowledge. InProceedings of the 2025 Confer-
ence of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 11636–11652.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025b. Retrieval-augmented gen-
eration with conflicting evidence.arXiv preprint
arXiv:2504.13079.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th international ACM SIGIR
conference on research and development in informa-
tion retrieval, pages 641–649.
Yi Yang, Wen-tau Yih, and Christopher Meek. 2015.
Wikiqa: A challenge dataset for open-domain ques-
tion answering. InProceedings of the 2015 con-
ference on empirical methods in natural language
processing, pages 2013–2018.Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui
Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong.
2025. Dream 7b: Diffusion large language models.
arXiv preprint arXiv:2508.15487.
Chuanyue Yu, Jiahui Wang, Yuhan Li, Heng Chang,
Ge Lan, Qingyun Sun, Jia Li, Jianxin Li, and Ziwei
Zhang. 2026. Unlocking the potentials of retrieval-
augmented generation for diffusion language models.
arXiv preprint arXiv:2601.11342.
Xiaowei Yuan, Zhao Yang, Yequan Wang, Shengping
Liu, Jun Zhao, and Kang Liu. 2024. Discerning
and resolving knowledge conflicts through adaptive
decoding with contextual information-entropy con-
straint. InFindings of the Association for Computa-
tional Linguistics: ACL 2024, pages 3903–3922.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2026.
Retrieval-augmented generation for ai-generated con-
tent: A survey.Data Science and Engineering, pages
1–29.
Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang,
Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen,
Yankai Lin, Ji-Rong Wen, and 1 others. 2025. Llada
1.5: Variance-reduced preference optimization for
large language diffusion models.arXiv preprint
arXiv:2505.19223.

A Proofs and Derivations
A.1 Proof of Theorem 1
The Donsker-Varadhan (DV) variational representa-
tion (Donsker and Varadhan, 1983) of the Kullback-
Leibler divergence for two distributions PandQ
states for the function classf:X →Rthat:
DKL(P∥Q) = sup
f(EP[f(x)]−logE Q[exp (f(x))]).
We set P=p cond,Q=p prior, and f(x) =
λs(x;C) =λlogpcond(x)
pprior(x)forλ∈R . Then, sub-
stituting these into the DV representation yields:
DKL≥E pcond[λs(x;C)]−logE pprior[exp(λs(x;C))].
By definition, IGt=D KL(pcond∥pprior)andZλ=
Epprior[exp(λs(x;C))] . Therefore, we obtain the
lower bound:
IGt≥λE pcond[s(x;C)]−logZ λ=:L(λ).
A.2 Proof of Corollary 1.1
To evaluate L(λ) atλ= 1 , we first expand the
partition termZ 1:
Z1=E pprior[exp(s(x;C))]
=X
x∈Vpprior(x) exp
logpcond(x)
pprior(x)
=X
x∈Vpcond(x)
= 1.
Consequently, the log-partition function is
logZ 1= 0. Substituting this intoL(1)gives:
L(1) = 1·E pcond[s(x;C)]−log(1)
=E pcond[s(x;C)]
=X
x∈Vpcondlogpcond
pprior
=D KL(pcond∥pprior).
SinceL(λ) is bounded above by DKL(pcond∥pprior)
according to Theorem 1, λ= 1 is a global maxi-
mizer.
A.3 Derivation of Eq. (11)
We analyze the local geometry of L(λ) around the
parametric anchor λ= 0 . The objective function
is:
L(λ) =λE pcond[s(x;C)]−logZ λ,where the partition functionZ λis:
Zλ=E pprior[exp(λs(x;C))]
=X
x∈Vppriorexp(λs(x;C)).
ands(x;C) = logpcond(x)
pprior(x).
The first derivative with respect toλis:
L′(λ) =E pcond[s(x;C)]−∂
∂λlogZ λ.
Using the property of the log-partition function,
its derivative is the expectation under the tilted
distributionp λ(x) :=pprior(x) exp(λs(x;C))
Zλ:
∂
∂λlogZ λ=1
ZλX
x∈Vpprior(x)s(x;C) exp(λs(x;C))
=X
x∈Vpλ(x)s(x;C)
=E pλ[s(x;C)].
Thus,L′(λ) =E pcond[s(x;C)]−E pλ[s(x;C)]. At
λ= 0,p 0(x) =p prior(x), which yields:
L′(0) =E pcond[s(x;C)]−E pprior[s(x;C)].
Using the first derivative, the second derivative with
respect toλis:
L′′(λ) =∂
∂λL′
=∂
∂λ(Epcond[s(x;C)]−E pλ[s(x;C)])
=−∂
∂λEpλ[s(x;C)]
=−X
x∈Vs(x;C)∂pλ(x)
∂λ.
To get the derivative of pλ(x), we first calculate
derivative of log-probability as:
∂
∂λlogp λ(x) =∂
∂λ
logp prior(x) +λs(x;C)−logZ λ
=s(x;C)−∂
∂λlogZ λ
=s(x;C)−E pλ[s(x;C)].
Then, we obtain
∂
∂λpλ(x) =p λ(x)∂
∂λlogp λ(x)
=pλ(x)
s(x;C)−E pλ[s(x;C)]
.

Therefore, we calculate
L′′(λ) =−X
x∈Vs(x;C)p λ(x)
s(x;C)−E pλ[s(x;C)]
=−
Epλ[s(x;C)2]−(E pλ[s(x;C)])2
=−Var pλ(s(x;C)).
Evaluating atλ= 0gives:
L′′(0) =−Var pprior(s(x;C))
Since Z0= 1, we have L(0) = 0 . The second-
order Taylor expansion ofL(λ)aroundλ= 0is:
L(λ)≈ L(0) +λL′(0) +λ2
2L′′(0)
=λ 
Epcond[s]−E pprior[s]
−λ2
2Varpprior(s).
To find the local maximum of this concave
quadratic approximation, we set the derivative with
respect toλto zero:
Epcond[s]−E pprior[s]−λVar pprior(s) = 0.
Solving for λyields the optimal scale λ∗formu-
lated as the Signal-to-Noise Ratio:
λ∗=Epcond[s]−E pprior[s]
Varpprior(s).
B Implementation Details
Hyperparameters.In the retrieval component,
we fetch the top-3 relevant documents per query.
All benchmark evaluations are conducted on a
randomly sampled subset of 1,000 instances per
dataset. For all baseline methods, we adopt the
default hyperparameters specified in their original
papers. For the autoregressive baselines adapted
to MDMs, we set the guidance scale to λ= 2
for CAD, and use α= 1 with a threshold of
0.25 for COIECD. In AdaCAD, we substitute
the static λwith the token-level Jensen-Shannon
Divergence (JSD). For MDM-specific baselines,
SPREAD modifies the re-masking strategy without
altering the pre-softmax logits. For A-CFG, we use
a re-masking proportion of ρ= 0.7 and a guidance
scale of λ= 1.5 (corresponding to w= 0.5 in the
original formulation). For the proposed ARAM,
we calculate λtat each denoising step, setting the
sensitivity parameter to β= 0.1 and the maximum
guidance scale to λmax= 1.0 for both LLaDA and
Dream.Prompt and Decoding Settings.We follow the
decoding configuration and prompt design of Atten-
tion Floating (Dai et al., 2026). We utilize a temper-
ature of 0, nucleus sampling with p= 0.9 , a maxi-
mum generation length of 32 tokens, and 32 denois-
ing steps. For the unmasking strategy, we adopt the
default policies of the respective backbone mod-
els: low-confidence for LLaDA and entropy for
Dream. We employ a zero-shot instruction prompt
for LLaDA (Figure 5) and a few-shot prompt with
in-context examples for Dream and other AR base-
lines (Figure 6). When retrieved documents are
available, they are concatenated sequentially with
index labels (e.g., “Passage 1: [Text]”). In the no-
retrieval setting, the context field is populated with
the string “No relevant context available.”
Experiment Details for Robustness Analysis.
For the analysis in Sec. 5.3, we utilize the outputs
from the main experiments: the retrieval-free base-
line, static RAG, and ARAM. For each backbone
model, we align predictions across methods on the
NQ benchmark and determine correctness using
the exact-match criterion.
For the retrieval-prior interaction analysis (Ta-
ble 2), we compare each retrieval-based method
against the retrieval-free baseline on an instance-
by-instance basis. Each sample is assigned to one
of four categories:Positiveif the no-retrieval pre-
diction is incorrect but the retrieval-based method
is correct,Negativeif the no-retrieval prediction is
correct but the retrieval-based method is incorrect,
Consistently Correctif both are correct, andCon-
sistently Wrongif both are incorrect. We report the
proportion of instances in each category.
For the analysis of the retrieval context qual-
ity (Table 3), we stratify samples according to the
quality of the retrieved context. We first define a
sample asGoldif at least one retrieved document
contains the answer. This is determined by normal-
ized string matching, including lowercasing, article
removal, and punctuation stripping, between the
reference answer set and the retrieved documents.
For the remaining non-gold (equivalent toNoise
Docin Sec. 5.4) samples, we further distinguish
betweenNon-AnsweringandIrrelevantcases using
an LLM-based annotation procedure that utilizes
the prompt template detailed in Figure 4. Given
the question, reference answer, and retrieved docu-
ments, the LLM is prompted to output one of two
labels:irrelevant, if the documents are off-topic
and provide no useful evidence for answering the

Table 7: Comparison of noise proxies. Average EM and
F1 scores are reported.
Noise ProxyLLaDA Dream
EM F1 EM F1
Varpprior(s) 20.3 32.5 33.5 43.2
H(p prior) 20.5 32.4 33.6 42.8
|H(p prior)− H(p cond)| 23.7 35.3 33.4 42.8
H(p cond) 24.9 36.1 33.9 43.6
Table 8: Comparison of stability function. Average EM
and F1 scores are reported.
Signal ProxyLLaDA Dream
EM F1 EM F1
- (Raw SNR) 21.5 33.5 33.5 43.4
Tanh 24.9 36.1 33.9 43.6
question, orNon-Answering, if the documents are
topically related but fail to provide sufficient sup-
port for the correct answer. For stable annotation,
we use GPT-4.1-mini with deterministic decoding.
C Additional Ablation Studies
Extended Comparison of Noise Proxies.To fur-
ther investigate the formulation of the noise proxy,
we compare the conditional entropy H(p cond)
against three alternative metrics: score variance
Varpprior(s), prior entropy H(p prior), and the abso-
lute entropy difference |H(p prior)− H(p cond)|. As
shown in Table 7, H(p cond)yields the highest EM
and F1 scores across both Dream and LLaDA mod-
els. The prior entropy H(p prior)omits the uncer-
tainty introduced by the external context, leading to
lower guidance accuracy. The absolute difference
formulation also underperforms compared to direct
conditional entropy. This indicates that H(p cond)
alone provides a sufficient and stable measure for
the noise introduced by the retrieved context.
Effect of the Stability Function.Table 8
presents the ablation on the stability function used
to bound the adaptive guidance scale. We compare
the raw SNR scaling against the Tanh-scaled formu-
lation. The results demonstrate that applying the
Tanh function improves generation performance
for both models. This improvement is attributed
to the bounding property of Tanh, which prevents
numerical explosion of λtwhen the denomina-
tor (noise) approaches zero during the unmasking
steps. Bounding the guidance scale ensures stable
generation without over-amplifying the contextual
logits.Table 9: Average inference time per query (seconds).
Method NFE / Step LLaDA Dream
No-RAG 1 0.93 0.94
RAG 1 1.55 1.56
CFG 2 2.02 3.06
ARAM (Ours) 2 2.07 3.29
Computational Cost Analysis.We measure in-
ference latency to analyze the computational over-
head introduced by retrieval and adaptive guidance.
Table 9 reports the average execution time over
three runs for both the LLaDA and Dream back-
bones. The latency gap between No-RAG and
static RAG mainly comes from the longer input
sequence induced by concatenating retrieved con-
text. In contrast, methods based on classifier-free
guidance, including standard CFG and ARAM, re-
quire both conditional and unconditional forward
passes at each step (2 NFE), leading to higher over-
all latency. Compared with standard CFG, ARAM
introduces additional overhead from the computa-
tion of the adaptive guidance scale, including the
entropy term and the Signal-to-Noise Ratio.
LLM Annotation Prompt
You are labeling retrieval context quality for a QA
example.
Question:
{query}
Reference answers:
{answers}
Retrieved passages:
{passages}
Task:
Return exactly one label:
- irrelevant: the passages are largely unrelated to the
question and do not provide useful evidence.
- related_non_answering: the passages are related to
the question but do not clearly contain the correct
answer.
Rules:
- If the passages are about the same en-
tity/topic/question but fail to answer it clearly, return
non answering.
- Use irrelevant only when the passages are genuinely
off-topic or retrieval junk.
- Return only the label, with no explanation.
Figure 4: Prompt template for LLM-based retrieval
context quality annotation.

Prompt for LLaDA
Use the following passages as context if they are
helpful, but provide a SHORT and DIRECT answer.
RULES:
- First, check the context passages for the answer
- If the context contains the answer, use it
- If not, use your general knowledge
- Answer must be 1-10 words maximum
- Do NOT include any explanations or extra text
- Do NOT repeat or summarize the passages
Context:
{context}
Question: {query}
Short Answer:
Figure 5: Prompt template used for LLaDA.Prompt for Dream
You are a helpful assistant that answers questions.
INSTRUCTIONS:
1. First, check if the answer is in the provided context
passages
2. If the answer is in the context, use it
3. If the context doesn’t contain the answer, use your
general knowledge
4. Always provide a direct, concise answer (typically
1-10 words)
5. Do NOT include explanations, reasoning, or
phrases like "based on" or "according to"
6. Never say "no answer found" - always attempt to
answer using available information
Context:
{context}
Here are examples showing the expected answer
format:
Example 1:
Context:
Passage 1: The Eiffel Tower is located in Paris,
France. It was completed in 1889.
Question: Where is the Eiffel Tower located?
Answer: Paris, France
Example 2:
Context:
Passage 1: Albert Einstein was born on March 14,
1879 in Ulm, Germany.
Question: When was Albert Einstein born?
Answer: March 14, 1879
Example 3:
Context:
Passage 1: The Great Wall of China stretches over
13,000 miles.
Question: What is the capital of Japan?
Answer: Tokyo
(Note: This answer uses general knowledge since the
context doesn’t contain it)
—
Now answer the following question in the same con-
cise format:
Question: {query}
Answer:
Figure 6: Few-shot prompt template used for the Dream
and other AR backbones.