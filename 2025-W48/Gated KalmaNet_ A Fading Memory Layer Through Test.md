# Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression

**Authors**: Liangzu Peng, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Wei Xia, Stefano Soatto

**Published**: 2025-11-26 03:26:37

**PDF URL**: [https://arxiv.org/pdf/2511.21016v1](https://arxiv.org/pdf/2511.21016v1)

## Abstract
As efficient alternatives to softmax Attention, linear state-space models (SSMs) achieve constant memory and linear compute, but maintain only a lossy, fading summary of the past, often leading to inferior performance in recall oriented tasks. We propose Gated KalmaNet (GKA), a layer that reduces this gap by accounting for the full past when predicting the next token, while maintaining SSM-style efficiency. GKA achieves this by solving an online ridge regression problem at test time, with constant memory and linear compute cost in the sequence length. Drawing inspiration from the Kalman Filter, we iteratively solve the online ridge regression problem. However, a critical insight is that standard Kalman filter equations are numerically unstable in low-precision environments (like bfloat16) and difficult to parallelize in modern hardware. We address both challenges through two key innovations: (1) an adaptive regularization strategy with input-dependent gating that controls the condition number of the ridge regression, ensuring numerical stability while balancing memory retention. And (2) the use of Chebyshev Iteration instead of other conventional iterative solvers, which we demonstrate to be more stable in low-precision settings. To further improve scalability, we develop a hardware-aware chunk-wise implementation of Chebyshev Iteration along with custom kernels for backpropagating through our adaptive regularization and gating mechanisms. Empirically, GKA shows strong language understanding capabilites on short-context tasks outperforming existing SSM layers (like Mamba2, GLA and Gated DeltaNet). On long-context, GKA excels at real-world RAG and LongQA tasks up to 128k tokens, achieving more than $10$% relative improvement over other fading memory baselines.

## Full Text


<!-- PDF content starts -->

Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression
Liangzu Peng1,*Aditya Chattopadhyay2,†Luca Zancato2Elvis Nunez2Wei Xia2Stefano Soatto2
University of Pennsylvania1AWS Agentic AI2
lpenn@seas.upenn.edu
{achatto,zancato,elvisnun,wxia,soattos}@amazon.com
Abstract
As efficient alternatives to softmax Attention, linear state-
space models (SSMs) achieve constant memory and linear
compute, but maintain only a lossy, fading summary of the
past, often leading to inferior performance in recall oriented
tasks. We propose Gated KalmaNet (GKA), a layer that re-
duces this gap by accounting for the full past when predicting
the next token, while maintaining SSM-style efficiency. GKA
achieves this by solving an online ridge regression problem
at test time, with constant memory and linear compute cost in
the sequence length. Drawing inspiration from the Kalman
Filter, we iteratively solve the online ridge regression prob-
lem. However, a critical insight is that standard Kalman
filter equations are numerically unstable in low-precision
environments (like bfloat16) and difficult to parallelize in
modern hardware. We address both challenges through two
key innovations: (1) an adaptive regularization strategy with
input-dependent gating that controls the condition number
of the ridge regression, ensuring numerical stability while
balancing memory retention. And (2) the use of Chebyshev It-
eration instead of other conventional iterative solvers, which
we demonstrate to be more stable in low-precision settings.
To further improve scalability, we develop a hardware-aware
chunk-wise implementation of Chebyshev Iteration along
with custom kernels for backpropagating through our adap-
tive regularization and gating mechanisms. Empirically,
GKA shows strong language understanding capabilites on
short-context tasks outperforming existing SSM layers (like
Mamba2, GLA and Gated DeltaNet). On long-context, GKA
excels at real-world RAG and LongQA tasks up to 128k to-
kens, achieving more than 10% relative improvement over
other fading memory baselines.
*Work done during an internship at AWS Agentic AI.
†Correspondence to achatto@amazon.com1. Introduction
Large Language Models (LLMs) powered by (softmax) At-
tention mechanisms [ 57] have revolutionized sequence mod-
eling through their ability to form rich associations within
their context window. However, a fundamental challenge
that LLMs face is that their time complexity scales quadrati-
cally and storage grows linearly with their input length.
Recent years have seen intense efforts to develop Atten-
tion alternatives. Among them, memory layers based on
linear State-Space models (SSMs) have grown popular for
their linear-time computation and constant storage cost in the
sequence length [ 9,64]. These SSMs find inspirations from
classic techniques in adaptive signal processing, and integrat-
ing them into modern SSMs leads to principled layer design
and enhanced performance [ 34,66,71]. However, pure SSM
models still underperform Attention in many settings, espe-
cially on long-context tasks. This gap is a consequence of
their different memory mechanisms: SSMs have afading
fixed dimensionallossystate of the past, while Attention has
aneideticever increasing KV-cache state [71].
To bridge this gap, we aim at designing a memory layer
that enjoys the efficiency of linear SSMs while performing
computation conditioned on the exact past. Towards this
goal, we first draw insights from theKalman filter(KF) [ 28].
In signal processing terms, KF computes the most recent
state conditioned on all data seen thus far, and, under mild
assumptions, KF is optimal in theMaximum A-Posteriori
(MAP) sense. In the LLM context, we use KF to update
the state of an SSM layer and predict its output based on
all past inputs. However, integrating KF into such a layer is
non-trivial and faces two challenges:
•Parallelizable Training.KF is an online algorithm and
needs to be parallelized to fully utilize modern hardware
that is highly optimized for large-scale LLM training.
•Numerical Stability.KF involves matrix inversion, which
can be numerically unstable in low precision arithmetic.
In this work, we proposeGated KalmaNet(GKA), a memory
layer that incorporates KF into its design and is both numeri-
cally stable and trainable on highly parallelizable hardware.
We start by observing that the KF recursion solves a test-timearXiv:2511.21016v1  [cs.LG]  26 Nov 2025

ridge regression problem. Then, to solve such a regularized
problem stably, we make the following choices:
•At the modeling level, we adaptively choose the regular-
ization strength of our test-time objective function based
on the Frobenius norm of the regularized data covariance.
With this choice we can easily upper bound the condition
number of the optimization problem.
•At the algorithmic level, we note that exact solvers (e.g.,
torch.linalg.solve ) are hard to parallelize (in a
chunk-wise manner), so we resort to the classicChebyshev
Iteration(CH), which we show has high numerical accu-
racy and fast convergence compared with alternatives such
as (accelerated) gradient descent and conjugate gradient.
To make GKA scalable and efficient, we implement CH
with adaptive regularization in Triton in a hardware-aware,
chunk-wise manner. Our technical novelty here includes
deriving a chunk-wise implementation that back-propagates
through the Frobenius norm, for which the difficulty is the
presence of anestedrecurrence. Furthermore, we combine
CH with agating mechanismthat decides the regression
residual weights in an input-aware and time-varying fashion,
enhancing the contribution of recent inputs and smoothly
fading out distant contexts. Overall, to the best of our knowl-
edge, this is a first adoption of the CH method for training
sequence modeling layers in LLMs stably at scale.
Finally, we demonstrate the efficacy of GKA on numerous
LLM benchmarks. For example, on synthetic recall tasks
(MQAR) [ 1], our method achieves the highest recall accuracy
among other state-of-the-art linear SSMs including Mamba2
[9] and (Gated) DeltaNet [ 66,67]. Also, GKA outperforms
existing SSMs on several (short-context tasks from LM-
Harness [ 15]) and long-context tasks (from RULER [ 25] and
HELMET [ 68]). Specifically, GKA improves upon SSM
baselines by at least 10% on real-world long-context tasks
like Retrieval-Augmented Generation and Long Question-
Answering tasks up to 128k tokens.
2. Prior Work and Preliminaries
In this section we briefly review prior work and preliminaries
that will set the stage for motivating our choice of design-
ing an SSM layer based on the Kalman Filter. For a more
detailed exposition of related work refer Section A.
(Softmax) Attention.At each time t, Attention [ 57] linearly
projects the t-th input token to obtain three vectors, named
query qt,key kt,value vtrespectively. Then, it outputs a
vector yt∈RDas a convex combination of all values seen
so far, with coefficients c1, . . . , c tgiven by inner products
of the current query qtwith all seen keys and a softmax
mapping:
yt=tX
i=1civi, c i:=exp(k⊤
iqt√
D)
Pt
i=1exp(k⊤
iqt√
D).(Attn)From an optimization perspective, (Attn) can be viewed as
solving the followingregressionobjective1,
yt= argmin
vtX
i=1expk⊤
iqt√
D
· ∥v−v i∥2
2.(1)
The success of (Attn) is often attributed to its ability to per-
form verbatim retrieval of relevant context from the entire
past. Here, the past refers to the entire key-value pairs ob-
served thus far, also known as theKV-cache, which grows
linearly with time t. Moreover, the computation is also linear
at each time t, and doing so for all tresults in a quadratic
time complexity. This high computation and storage cost of
Attention makes its use prohibitive in long context scenarios.
Linear State-Space Models (SSMs).The high computation
cost of (Attn) has motivated a flurry of work developing new
LLM layers, like SSMs, with linear rather than quadratic
cost. Most SSMs maintain astate matrix St∈RD×Dand
update it at each time step via a linear recursion of the form
St=γt·St−1+βt·vtk⊤
t, y t=Stqt,(Linear-SSM)
where γt, βtare typically in [0,1] . Unlike the verbatim
lookup of (Attn), here (Linear-SSM) essentially compresses
the entire KV-cache into a fixed-dimensional representation
St. Subsequent computation of the output ytrelies on St
and no longer on the exact past. This results in a constant
cost of storage and computation at every timestep.
In many linear SSMs (e.g., RetNet [ 54], Mamba2 [ 9]),
the use of γtandβtis often heuristic and finds inspirations
from nonlinear recurrent neural networks [ 23]; in that light,
γtandβtare calledforgettingandinputgates, respectively.
This basic form of (Linear-SSM) has been generalized by
replacing γtwith a diagonal matrix (GLA [ 64], RWKV-6
[44], Longhorn [ 34]) orlow-rank-plus-identitymatrix (Gated
DeltaNet [51, 66, 67], DeltaProduct [53], RWKV-7 [45]).
Similarly to that of (Attn) the case withlow-rank-plus-
identitymatrices can often be justified from an optimization
perspective. For example, Gated DeltaNet [ 51,66] updates
the state via (Iis theD×Didentity matrix)
St=γt·St−1 
I−β tktk⊤
t
+βt·vtk⊤
t,(GDN)
which can be viewed as applying one gradient descent step
with stepsizeβ tand initializationγ tSt−1to the objective
min
S∥Skt−vt∥2
2.(2)
The objectives of (GDN) (2) and (Attn) (1) are prime exam-
ples that expose a general distinction between linear SSMs
and (Attn): The former updates its state based on a regres-
sion objective that considers only the previouslossystate
1Concretely, for keys and queries of unit norm, Attention is precisely
theNadaraya-Watson estimator[ 39,62] with the Gaussian kernel to ap-
proximate the conditional expectation of the value given a query; cf. [ 4,58].

and the current time step, whereas the latter uses the entire,
exact KV-cache to solve its regression objective (1).
We hypothesize this myopic view of SSM objectives re-
sults in their lower performance and limited long-context
abilities. We then ask:What is an objective or, equivalently,
a recursion that considers the entire past as (Attn) while still
being solvable in linear time as in (Linear-SSM)?
3. A Linear SSM Inspired by the Kalman Filter
In Section 3.1 we show how the Kalman Filter (KF) gives
insights into a new linear SSM layer that takes all past time
instants into account. In Section 3.2 we explain the numeri-
cal and efficiency challenges of building such a layer.
3.1. Motivation from Kalman Filter
KF is an established online approach that takes the exact past
into account to optimally solve aweighted ridge regression
objective (e.g., see [ 46, Proposition 2 & Lemma 3]). In our
context, this means that theoptimalstate
St= argmin
S∈RD×Dλ· ∥S∥2
F+tX
i=1ηi· ∥Sk i−vi∥2
2 (3)
can be computed by the KF recursion
St=St−1−(St−1kt−vt)k⊤
tΦt−1
1/ηt+k⊤
tΦt−1kt,(KF)
where ηtis the weight for the t-th key-value pair, and Φt−1
is the Hessian inverse of (3) at time t−1 (Φt−1itself can be
continually updated via theWoodbury matrix identity). It is
now clear that objective (3) takes the entire KV-cache into
account, similarly to (Attn). It is also clear that (KF) is an
efficient update scheme similarly to (Linear-SSM); indeed,
(KF) is also alow-rank-plus-identityform (cf. (GDN)).
A key difference from (Linear-SSM) is that (KF) lever-
ages second-order information from Φt−1to solve (3) opti-
mally, whereas (Linear-SSM) relies on instantaneous objec-
tives akin to (2) (cf. [ 66, Table 2]). It is in this sense that
we say (KF) is more expressive than other (Linear-SSM) or
(GDN). We now detail the differences in the objectives of
(KF) and (Attn):
•(KF) computes a parametric linear estimator that en-
ables a constant-sized memory, while (1) computes a non-
parametric point estimate that entails storing the full cache.
•In (1), the weights of the same residual vary over time as
the queries differ at each time, while in (3) i-th weight
ηiis constant once observed at time i. The former re-
sults in quadratically many weights—thus a quadratic time
complexity—and the latter linearly many.
•In (3), the regularizer λ·∥S∥2
Fprevents overfitting our state
to key-value pairs, as only a finite amount of “information"
can be stored in a constant-sized memory beyond which
will result in “fuzzy" recall. In this light, λcan be thought
of as controlling the memorization “capacity" of the state.3.2. Hurdles Towards Scalable Kalman Filter SSMs
Despite its optimality and (sequential) computational effi-
ciency the (KF) recursion lacks a hardware-aware implemen-
tation that leverages parallelism in modern Tensors Cores.
Moreover, for long sequences it can lose numerical precision
due to division (and more significantly due to how the Hes-
sian inverse Φtis updated). The final hurdle is conceptual:
Fixing weights ηiand regularization λover time as in (3)
might make a layer less expressive.
We are aware of the use of (KF) or (3) in neural network
training three decades ago [ 52] or in deep continual learning
recently [ 36,47,72]. We are also aware of the recent men-
tioning of (3) or efforts towards solving it, which go by the
nametest-time optimization[ 59,61]. However, to the best of
our knowledge, none of the prior work has fully addressed
the above hurdles that need to be solved to design an SSM
layer that is trainable in parallel, numerical well-behaved,
and sufficiently expressive. In particular, both [ 59] and [ 61]
have overlooked a basic numerical concern: The worst-case
numerical error in solving (3) can be ϵ·κ [19], where κis
the condition number of the Hessian in (3) and ϵthe machine
precision; since ϵ≈0.007 (bf16), (3) has to be regularized
stronglyfor κand the worst-case error to be small, regardless
of algorithmic choices to solve (3). Indeed, the regulariza-
tion enforced in [ 59] sets λto be lower bounded by 0.25,
but this is not sufficient: Their κis as large as 500[59, Fig.
13], implying a worst-case error of 3.5(The implementation
of [59] available on GitHub is numerically vulnerable; we
failed to train it without NaNs in various settings.). Also,
the regression objective in [ 61] has no regularization, which
makes it numerically ill-posed for low-precision training.
4. Gated KalmaNet (GKA)
We proposeGated KalmaNet(GKA) to address the above
hurdles: We enhance numerical stability via adaptive regu-
larization and the classicChebyshev Iteration(CH), increase
expressivity of KF via a standard gating mechanism, and
improve parallelism via a hardware-friendly implementation.
4.1. CH with Adaptive Regularization & Weighting
Motivation.As alluded earlier, solving (3) via (KF) is
sequential in nature, and here we consider alternatives
amenable to parallelizable training. Our first step towards
this is to write down a closed form solution to (3) and com-
pute the output
yt=Stqt= tX
i=1ηivik⊤
i! tX
i=1ηikik⊤
i+λI!−1
qt.
With the weighted covariances Ut:=Pt
i=1ηivik⊤
iand
Ht:=Pt
i=1ηikik⊤
i, we note that ytcan be computed via
first solving (Ht+λI)x=q tforxand then left-multiplying

Algorithm 1:Chebyshev Iteration to solve Hξ=q
1Input:H∈RD×D, q∈RD, eigenvalue bounds
L, µwithL≥µ >0, number of iterationsr;
2Initialize:ρ←L−µ
L+µ;ω0= 0; the first two
iteratesξ −1←0, ξ 0←2q
L+µ;
3For Loop (i= 1, . . . , r):
ωi←4
4−ρ2ωi−1(weight schedule)
ξi←ξi−1−2·ω i
L+µ(Hξi−1−q)(grad descent)
ξi←ξi+ (ω i−1)(ξ i−1−ξi−2)(momentum)
4Output:ξ r
Ut. An exact solver (e.g., torch.linalg.solve ) can
do so with high accuracy, by parallelizing over the batch
dimension. However, it is inefficient here for two reasons.
First, it takes O(D3)time for every t. Second, it requires
explicitly forming and materializing all Ht’s, which would
entail a large I/O cost. In light of this, we resort to first-
order iterative methods that admit chunk-wise implementa-
tion without materializing all Ht’s, enabling parallelism over
chunks and batches. Furthermore, they often take O(D2)
time complexity per iteration and can converge quickly in a
few iterations. The iterative method we choose is theCheby-
shev Iteration(CH); we proceed to describe its basic idea,
with a justification of using CH deferred to Section 4.2.3.
Chebyshev Iteration (CH).CH can be seen as anaccel-
erated gradient descentmethod (AGD) that applies (grad
descent) and (momentum) to the strongly convex objective
1
2ξ⊤Hξ−ξ⊤q, that is to solve the optimality condition
Hξ=q (Algorithm 1). Different from AGD, CH incor-
porates a (weight schedule) and makes specific choices of
different parameters; these choices makes CH optimal with
the fastest convergence among first-order methods [43].
We now replace the above exact solver with CH:
ˆxt=CH(H t+λI, q t, r), y t=Utˆxt.
Here, CH(H t+λI, q t, r)means riterations of CH to ap-
proximately solve (Ht+λtI)x=q t. To improve stability
and expressivity, next we allow regularization λand weight
ηito be time-varying and chosen adaptively. We write λt
andηt,ito make their dependency in time texplicit, with
ηt,ibeing the weight of thei-th token at timet.
Adaptive Regularization.As mentioned, the condition
number κtofHt+λtIhas to be controlled for any method
to be numerically stable. We choose λtto be proportional to
the Frobenius norm ∥Ht∥F, that is to set λt=a· ∥H t∥Fforsome constanta >0. An upper bound onκ tnow ensures:
κt=λmax(Ht) +λ t
λmin(Ht) +λ t≤∥Ht∥F+λt
λt=a+ 1
a.(4)
Here λmax(Ht), λ min(Ht)are the maximum and minimum
eigenvalues of Ht, respectively. Given this choice of λt, we
setL=∥H t∥F+λtandµ=λ tfor Algorithm 1.
Adaptive Weighting (Gating).We use weights ηt,ithat are
exponentially decaying in time: For all t≥i , we parameter-
izeηt,i=Qt
j=i+1γj, with each γj∈[0,1] learnable. The
fading weights encode the “prior" ofrecency biasthat has
been shown to exist in LLMs [ 13] without even explicitly
computing the weights from the query-key dot products as in
(Attn). Similarly to (Attn), the weights on the residuals are
now time-varying, but differently, the exponentially decay
parameterization allows for linear-time implementation.
Forward Recurrence.We now summarize our recurrence
which arms CH with adaptive regularization and weighting:
Ht=γt·Ht−1+ktk⊤
t, Ut=γt·Ut−1+vtk⊤
t,
yt=Utˆxt,ˆx t=CH(H t+λtI, qt, r).(CH)
4.2. Chunk-wise Implementation
In this subsection, we describe our hardware-aware imple-
mentation for the forward + backward passes for (CH). More
details can be found in Section B.
4.2.1. Forward Pass
Similarly to prior work [ 9,64,66], we now describe achunk-
wiseimplementation for (CH). In (CH), given Utandˆxt,
computing yt=U tˆxtin a chunk-wise fashion is similar
to that of (Linear-SSM); also similar is the calculation of
Htξi−1as needed in (grad descent). For these we refer the
reader to [ 64,66] for details. Our algorithmic novelty here
is a chunk-wise computational formula for ∥Ht∥F, presented
next.
LetTbe the sequence length and Cthe chunk size such
thatN:=T/C is an integer. For t= 0, . . . , N−1 , write
[t] :=tC . The core idea of a chunk-wise implementation
is as follows. First, we compute and store theinitial state
H[t]of every chunk. This gives us implicit access to H[t]+c
via unrolling the recurrence of Htforcsteps and therefore
allows us to carry out computation with H[t]+c; for exam-
ple, we can compute the matrix-vector product H[t]+1ξvia
H[t]ξ+γ 1k[t]+1k⊤
[t]+1ξ. This is without forming H[t]+1 ex-
plicitly, thereby reducing the number of states to materialize
on chip. To implement such a scheme, we need to precom-
pute all H[t]’s sequentially, and then do the computation with
parallelism over chunks and within each chunk.
We now make this idea precise for computing all ∥Ht∥F’s
within a chunk. Since the computation of each chunk is
the same, we simplify by working with the first one where
we have access to initial state H0, gates γ1, . . . , γ C, keys

KC= [k 1, . . . , k C]∈RD×C,and we aim to compute
∥H1∥F,∥H 2∥F, . . . ,∥H C∥F. With these notations, we first
compute the C-dimensional vector ζ= [ζ 1, . . . , ζ C]⊤of
cumulative products of γi’s, with ζc=Qc
i=1γi. Then, form
theC×C upper triangular matrix Mwhose ( i, j)-th entry
Mj,cisζc/ζj(∀c≥j). Now, unroll the recurrence ofH c:
Hc=ζcH0+cX
j=1Mj,ckjk⊤
j=ζcH0+CX
j=1Mj,ckjk⊤
j.
Expanding∥H c∥2
Fgives the following sum of three terms:
ζ2
c∥H0∥2
F+ 2ζ cCX
j=1Mj,ck⊤
jH0kj+CX
j=1Mj,ckjk⊤
j2
F.
Withζ, the first term ζ2
c·∥H0∥2
Fis easily computed in parallel
for all c. For the second term, we first compute the vector of
quadratic forms k⊤
jH0kjfor all jin parallel, broadcast it and
multiply it with Melement-wise, sum over each column, and
multiply the result with 2ζelement-wise. Finally, with Gram
matrix GC:=K⊤
CKC, one verifies the third term can be
computed in parallel for all cvia the following pseudocode:
column-sum(((G C⊙G C)M)⊙M).(5)
Here⊙denotes element-wise multiplication and the sum is
over each column. Summing the three terms and taking the
square root, we obtain∥H 1∥F, . . . ,∥H C∥F, as desired.
4.2.2. Backward Pass
Motivation.Typically, the backward pass is done automati-
cally via torch.autograd . However, for iterative meth-
ods such as CH (Algorithm 1), torch.autograd would
store someactivationsor intermediate iterates, entailing
large storage cost. While in principle we can back-propagate
through CH without storing any intermediate activations or
iterates (by our trick ofreverting the CH iterations, cf. Sec-
tion C.1), under this trick it is difficult to compute all the
gradients in a chunk-wise fashion. Therefore, we resort to
theimplicit differentiationtrick, which is practically effi-
cient and chunk-wise implementable, for backpropagation
through the linear equations that CH approximately solves.
Implicit Differentiation.We derive the backward pass for
our method with the standard implicit differentiation trick.
It assumes we find an exact solution x∗
tto the equations
(Ht+λtI)x=q t. In the backward pass, we are given the
gradient dx∗
t:=dL
dx∗
tof some loss function L, and need
to compute the corresponding gradients at qt, kt, γt. For
example, via the chain rule we obtaindq t:=dL
dqtvia
dqt= (H t+λtI)−1dx∗
t,(6)that is to solve linear equations similarly to the forward pass.
Since the forward pass computes an approximate solution ˆxt
via CH, we receive an approximate up stream gradient dˆxt
(not exactly dx∗
t). Thus we employ CH to obtain an approxi-
mate gradientdˆq t=CH(H t+λtI, dˆx t, r); cf. Table 1.
Backward Recurrence.Besides dqt, we need to compute
dHtfrom which we obtain dktanddγtvia the chain rule.
We describedγ tin the Appendix. Here we analyzedk t:
Lemma 1.Withλ t=a∥H t∥F,wt=2a·(x∗
t)⊤dqt
∥Ht∥F, we have
dki=X
t≥iMi,t 
−dqt(x∗
t)⊤−x∗
tdq⊤
t−wtHt
ki.(7)
With Ai:=P
t≥iMi,t·dqt(x∗
t)⊤, we can compute the
first two terms −Aikiand−A⊤
ikiin (7), similarly to (Linear-
SSM). Specifically,A isatisfies the recursion
Ai=γi+1Ai+1+dq i(x∗
i)⊤,(8)
thus calculating Aikiamounts to calculating Utqtin (Linear-
SSM); a difference is that the recursion here runs backwards.
Similarly, with Bi=P
t≥iMi,twtHt, the third term in
(7) can be written recursively as
Bi=γi+1Bi+1+wiHi, o i=B iki.(9)
Chunk-wise Recurrence.As indicated, a chunk-wise im-
plementation for computing Aikiis known. On the other
hand, computing Bikiis more challenging than Aiki, as the
additive term wiHiin the backward recursion (9) is not nec-
essarily rank- 1; rather, Hiitself is defined via the forward
recursion in (CH). Our contribution here is a derivation for
computingB ikiefficiently in a chunk-wise manner.
We begin by unrollingB itoB C+1:
Bi=Mi,C·γC+1BC+1+Bintra
i
Bintra
i:=CX
c=iMi,c·wcHc(10)
We next discuss theintra-chunkterm Bintra
ikiandcross-
chunktermM i,C·γC+1BC+1 in succession.
Intra-chunk Computation.We now unroll Hcand obtain
an expressionBintra
imore amenable to parallelism:
Bintra
i=CX
c=iMi,c·wcHc
=CX
c=1Mi,cwc
ζcH0+CX
j=1Mj,ckjk⊤
j
=H 0CX
c=1Mi,cwcζc+CX
j=1kjk⊤
jCX
c=1Mi,cwcMj,c.

Table 1.Implicit differentiation for computingdq t.
forward pass backward pass
exactx∗
t= (H t+λtI)−1qt dq∗
t= (H t+λtI)−1dx∗
t
CGˆx t=CG(H t+λtI, qt, r)dˆq t=CG(H t+λtI, dˆx t, r)
CHˆx t=CH(H t+λtI, qt, r)dˆq t=CH(H t+λtI, dˆx t, r)
The coefficients of H0, written as bi, are easily computed in
parallel for all ivia element-wise operations, broadcasting,
and summing. The coefficient of kjk⊤
jis precisely the (i, j) -
th entry of the matrix Mw:=Mdiag(w 1, . . . , w C)M⊤.
Thus[Bintra
1k1, . . . , Bintra
CkC]is equal to
H0(diag(b 1, . . . , b C)KC) +K C 
(K⊤
CKC)⊙M w
.
Here themask Mwis in general a full matrix with no zero
entries, as opposed to the triangular matrix in the case of
(Linear-SSM). While the triangular mask in the backward
pass allows the error feedback from future tokens to be
leveraged for learning past tokens, here our full mask Mw
allows all tokens to interact with all other tokens in the
backward pass, which facilitates the information flow and
learning.
Cross-chunk Computation.In (10), both γC+1 andBC+1
are from the future chunks, thus we revise (10) into the cross
chunk recursion of eBC+1:=γ C+1BC+1 which allows us
to maintain a single term eBC+1 from the future:
eB1=ζC·eBC+1+eBintra
1,eBintra
1:=CX
c=iζc·wcHc.
In our intra-chunk computation, we store the intra-chunk
termeBintra
1of all chunks, implement the above with a simple
for loop, and collect the termsζ C·eBC+1ki.
4.2.3. Comparison to Other Iterative Solvers
Here we validate our choice of Chebyshev Iteration (CH) by
benchmarking it against other iterative methods.
Convergence in the Forward Pass.We generate random
regression problems, which we solve via CH and 3 other
baselines: gradient descent (GD), accelerated GD with Nes-
terov’s momentum (AGD), conjugate gradient (CG). GD and
AGD are run with stepsizes that are optimal for regression
problems. Fig. 1a shows CG converges the fastest within a
few iterations, while CH reaches the same accuracy as CG
at iteration 10 and eventually attains the smallest errors.
Stability of the Backward Pass.We then proceed and mea-
sure the gradient stability of CG and CH, whose backward
passes are implemented either via implicit differentiation as
per Table 1 (impl), or viatorch.autograd(auto).
In Fig. 1b, CG (impl) as a standalone layer has its gra-
dient close to that of the exact solver up to a 10−3relative
difference. In Fig. 1c, this difference is amplified to almost 1in a 5-layer LLAMA where (Attn) is replaced with (3). This
indicates CG (impl) completely deviates from the reference
gradient (exact), defeating its purpose of training the net-
work from the regression feedback. In contrast, the gradients
of CH (impl) and CH (auto) are eventually close to that of
the exact solver either as a single layer (Fig. 1c) or within
multiple layers (Fig. 1d), up to a 10−6difference. More-
over, the curves for CH (impl) and CH (auto) nearly overlap,
suggesting that their gradients may be close. The following
lemma confirms and formalizes this intuition (see Section C
for a proof), thereby justifying our choice of CH over the
alternatives:
Lemma 2.Let dqtbe the exact gradient of qtfor CH, e.g.,
computed by CH (auto). Let dˆqtbe the gradient of CH (impl),
computed as per Table 1. We havedq t=dˆq t.
4.3. Architectural Consideration
Our GKA layer in Fig. 2 includes two components (in green)
on top of established practices (in blue). The CH component
is described in Section 4.1, thus here we introduce the α-
connection. First, the sigmoid activation ensures αt∈[0,1] ,
so the output of the α-connection is a convex combination
of the original query qtand the output ˆxtof CH. Second, it
plays a similar role to residual connection, which establishes
a direct path that facilitates the gradient flow and improves
training; we show this is indeed the case in Section F.3.
Finally, the full architecture for GKA is the standard Trans-
former, with its attention layer replaced by the GKA layer.
5. Experiments
In this section, we empirically validate the efficacy of our ap-
proach. We first evaluate memorization ability on synthetic
associative recall tasks (Section 5.1). We then report training
throughput of GKA (Section 5.2). Finally, we examine per-
formance on short-context language understanding bench-
marks such as commonsense reasoning and long-context
modeling abilities in Section 5.3. The Appendix details our
experimental settings (Section D) and ablations of various
modeling choices (Section F, Section G).
Baselines.All experiments consider the following state-of-
the-art linear SSM-based fading memory layers as baselines:
Mamba2 [ 9], DeltaNet [ 66], Gated DeltaNet (GDN) [ 67],
and Gated Linear Attention (GLA) [ 64]. Each of these layers
rely on instantaneous objectives that depend on the previous
lossystate and current tokens (e.g., (2)), as opposed to the
entire history of tokens observed so far as in GKA. Finally,
we contrast our results with (Softmax) Attention, which
serves as our paragon. For our Attention-based model, we
adopt the architecture proposed in Qwen3 models [63].

(a) Empirical convergence (b) CG as a single layer (c) CG in 5-layer LLAMA (d) CH as a single layer (e) CH in 5-layer LLAMA
Figure 1.CH converges with smaller errors than CG and is more numerically stable.Convergence of different methods in residual
norms during the forward pass with batch size 8, sequence length 2048, 8 heads, head dimension 128 (a), and relative gradient differences
from the exact solver ( torch.linalg.solve ) to CG (b, c) or CH (d, e). The backward pass is viaimplicit differentiation(impl) or
torch.autograd(auto); cf. Table 1. In (b, d) the gradients are those of[q t, kt]; in (c, e) the gradients are those of network weights.
Figure 2.Our GKA block. Blue refers to established practices in
the literature with the solid circles denote ℓ2normalization. Green
components (CH andα-connection) are our proposals.
5.1. GKA on Synthetic Associative Recall Tasks
We first assess the capability of our models to recall infor-
mation on the multi-Query Associative Recall (MQAR) task,
a synthetic task introduced by Arora et al. [ 1]. This task
presents the model with a sequence of key-value pairs to
memorize, followed by a sequence of queries. For each
query, the model must retrieve the corresponding key from
memory and accurately recall its associated value. Attention
based layers perform the best in this task, while SSM-based
memory layers are known to struggle as their memory fades
away as the context length grows.
We compare GKA with Attention and other linear SSM
baselines on this task. For each memory layer type, we train
2-layer models on MQAR training data and evaluate on a
held-out test set. We repeat this experiment for 4differ-
ent learning rates spanning from 10−4to10−2. As shown
in Fig. 3a, GKA improves upon every other linear SSM
baseline at all sequence lengths and model dimensions con-
sidered. Note, the complexity of the task increases with
increasing sequence length and number of key-value pairs,
while larger model dimensions improve memorization ca-
pacity through increased state size. The success of our layer
can be attributed to our modeling choice: unlike other fading
memory designs (like GDN or Mamba2), we construct states
based on the optimal MAP estimate conditioned on the entire
history, enabling better retention of remote information.5.2. Training Throughput of GKA
In Fig. 3b we measure the running time (forward + back-
ward) of a single GKA layer and compare it with FlashAt-
tention [ 18], DeltaNet, and GDN. Our layer achieves com-
parable running time to GDN, a state-of-the-art SSM layer,
despite having a more computationally expensive state up-
date equation (CH) than (GDN). This demonstrates that our
chunk-wise parallelization strategy effectively compensates
for the additional computational cost.
5.3. GKA on Language Modeling
5.3.1. Short-context Tasks
Setup.For this set of experiments, we construct 2.8B LLM
models for each memory layers (GKA and baselines de-
scribed in Section 5) by cascading blocks of mem + Multi-
Layer Perceptron (MLP) blocks.2Hereby, we refer to the
2.8B models with the same name as the layer used to con-
struct them. We then train each model on DCLM [ 32], a
generic pre-training dataset for 100B tokens at 4K context
length using the AdamW optimizer with a peak Learning
Rate (LR) of 10−3and gradient clipping of 1.0. We used
the cosine LR scheduler with a warmup period of 5B tokens
with a global batch size of 2M tokens. All models employ
the GPT2 tokenizer with a vocabulary size of50K tokens.
Tasks.Following prior works [ 64,67,71], to consider lan-
guage modeling capabilities of our model we perform zero-
shot evaluation on the following eight common-sense rea-
soning tasks from LM-Harness [ 15]: Arc-E, Arc-C, BoolQ,
COPA, HellaSWAG, PIQA, SciQ, Winogrande. We also eval-
uate models on FDA and SWDE, real-world recall-intensive
tasks which focus on extracting structured information like
tagged content from raw text (for example, HTML files). All
these tasks are relatively short (<2K tokens).
Results.We report our results in Table 2. GKA outperforms
all fading memory baselines on average across all tasks ow-
ing to its ability to better manage its state via solving (3).
In particular, GKA outperforms both GDN and Mamba2
on recall-intensive tasks (FDA and SWDE) by about 10%
2For Mamba2 baseline, we consider cascading blocks of Mamba2 layer
alone since a single Mamba2 layer has the Mamba2 SSM and MLP.

(a) Accuracy vs. model dimension for different fading memory layers on MQAR. (b) Runtime of a single memory layer
Figure 3.MQAR results(a) Each plot corresponds to a particular sequence length and number of key-value pairs for the model to memorize.
Runtime(b) Runtimes are for a single forward + backward pass (8 heads, head dim128, batch size4, averaged over 20 runs).
Table 2.On average GKA improves upon all fading memory baselines across all tasks.We report results for zero-shot evaluation of
2.8B language models for short-context tasks. For each task, bold indicates highest value followed by underlined.
Model ARC-C ARC-E BoolQ COPA HellaSWAG PIQA SciQ Winogrande FDA SWDE Avg
acc_n↑acc_n↑acc↑acc↑acc_n↑acc_n↑acc_n↑acc↑contains↑contains↑
Transformer 32.25 56.1064.2880.00 60.96 73.56 79.50 61.7258.53 72.28 63.92
Gated Linear Attention 27.82 50.80 52.57 78.00 48.83 70.13 69.60 54.54 2.81 20.43 47.55
DeltaNet 32.8558.16 42.51 81.00 61.13 73.78 43.90 61.72 11.80 46.08 51.29
Mamba2 32.24 59.64 58.72 82.00 62.23 73.78 79.80 62.19 7.71 41.13 55.94
Gated DeltaNet 32.59 60.0262.75 82.00 62.80 74.32 80.60 62.35 8.26 44.28 57.00
Gated KalmaNet (Ours) 32.51 59.89 61.6885.00 63.84 74.81 83.20 64.1712.89 50.95 58.89
(rel. improvement). We note that although GKA improves
upon existing SSM layers there is still a gap with Attention-
based Transformer especially on recall-tasks owing to the
eidetic capabilities of Attention. Nevertheless, as discussed
in Section 1 this improvement comes at a quadratic cost at
training time, whereas our layer’s computational complexity
is still comparable to existing SSM layers (cf. Section 5.2).
In Section I we extend our results to Hybrid models (stack
of SSM and Attention layers) and show that the gap with
full Transformer models becomes negligible (while still ben-
efiting the SSM’s computational advantages). Finally, in
Section E we show that GKA exhibits stronger scaling with
compute than other SSM baseline models.
5.3.2. Long-context Tasks
Setup.To enable long-context capabilities of our models,
as is common practice, we perform continued pre-training
of our 2.8B models obtained in Section 5.3.1 on 25B tokens
of long documents at 128K context length (cf. Appendix).
To the best of our knowledge we are the first to train and
evaluate SSM models up to 128K context (e.g., previous
work [67] only considered up to4K/8K context).
Tasks.For long-context, we refrain from using perplexity
as it is known to have limitations at assessing long-context
performance of LLMs [ 14,16,40]. Instead, we turn to
recently proposed benchmarks that mix synthetic and real
datasets comprising several long-context tasks: synthetic
Recall, Retrieval-Augmented Generation (RAG), Many shot
In-Context Learning (ICL) and Long Question-Answering
(LongQA). For synthetic Recall and LongQA we consider
tasks from the RULER benchmark [ 25]. For RAG and ICLwe consider tasks from HELMET [68].
Results.Fig. 4 reports our results. GKA shows strong RAG
and LongQA capabilities, outperforming all fading memory
baselines by at least 10% (rel. improvement). Interestingly,
on synthetic Recall tasks from RULER, GKA is competitive
only at 4K context length and starts to fall behind afterwards.
We attribute this divergence to the fundamental differences
between these task types. While both RAG and LongQA
can be thought of as finding relevant information in long
streams of text, they involve more realistic linguistic pat-
terns and semantic relationships that align with natural text
distributions seen during pretraining. In contrast, synthetic
Recall tasks require models to find specific words, numbers,
or UUIDs verbatim from long contexts filled with random
distractors. This artificial setting does not reflect natural
text distributions. Since GKA computes MAP estimates of
the latent state based on learned representations of observed
tokens, it relies on its pretrained weights to determine which
information is important to retain. The synthetic and un-
natural structure of Recall tasks makes it difficult for the
model to identify what should be retained, as pretrained
knowledge provides little signal about importance in these
artificial contexts. This suggests that our approach excels in
realistic scenarios where pretrained knowledge about natural
language structure can guide information selection, but strug-
gles when the signal-to-noise distinction is purely artificial.

8k 16k 32k 64k 128k
Sequence Length12.515.017.520.022.525.027.530.0Average Score
RAG
8k 16k 32k 64k 128k
Sequence Length456789Average Score
ICL
4k 8k 16k 32k 64k 128k
Sequence Length05101520253035Average Score
Synthetic Recall
4k 8k 16k 32k 64k 128k
Sequence Length5.07.510.012.515.017.520.022.525.0Average Score
Long-QA
Gated KalmaNet (Ours) Gated DeltaNet DeltaNet Mamba2 Gated Linear AttentionFigure 4.Long Context Performance up to 128k tokens. GKA achieves strong RAG and LongQA capabilities, outperforming all baselines
by 10% in relative improvement. Interestingly, we observe that there is no clear winner Synthetic Recall. All models struggle to perform
better than random chance on ICL.
6. Kalman Filter forOptimallyModelling Fad-
ing Memory
In this section, we show how the Kalman Filter (KF) pro-
vides a principled solution for constructing an optimal fading
memory that accounts for the entire history. We begin by de-
scribing the standard Kalman Filter recurrence in the context
of memory modeling. However, the KF has a fundamental
limitation: its inherently sequential nature makes it impracti-
cal for large-scale training on modern hardware accelerators
Section 3.2. To address this, we make simplifying assump-
tions that makes KF amenable to parallelization on modern
hardware accelerators. We then demonstrate that several
recent state-space models (DeltaNet, Gated DeltaNet, and
Kimi Delta Attention) can be viewed as approximations to
the KF recurrence. Specifically, these methods approximate
the “optimal" Kalman gain matrix while ignoring depen-
dencies on the past. In contrast, GKA computes the exact
Kalman gain by considering the full history. This theoretical
advantage translates to improved empirical performance, as
we demonstrate in Section 5.
6.1. A Dynamical System for Fading Memory
The Kalman filter is a classical algorithm for online optimal
inference in linear Gaussian State-Space Models. It gives
a principled way to maintain and update a compact state
as new noisy observations arrive. The latent state serves as
a compressed "memory" of the past. More formally, it is
a minimal sufficient statistic that makes past observations
conditionally independent of future ones given the state.
We begin by describing a linear Gaussian model for fad-
ing memory.
st=A tst−1+Btut+wt, w t∼ N(0, Q t)
vt=k⊤
tst+µt, µ t∼ N(0, r t),(LGM)
where st∈Rnis a latent state that summarizes the past,
ut∈Rnis the control input that updates the state and vtis
the scalar measurement observed at time t.At, Bt∈Rn×n
are the state transition and input selection matrices, andkt∈Rnis the emission (readout) vector. Finally, wtandµt
are Gaussian process and measurement noise, respectively.
Parameter interpretation. AtandBtcontrol the forget-
ting (fading of the remote past) and input selectivity rates
respectively, determining how the state evolves over time.
The measurement noise µtnaturally gives rise to gating
mechanisms commonly used in modern SSM layers, as we
will show in Section 6.4.
Extension to multi-channel measurements.In attention
mechanisms, the memory consists of verbatim key-value
pairs that can be queried to retrieve past information [ 71].
Similarly, we want our state to reconstruct past values from
their corresponding keys. To achieve this, we extend to a
matrix-valued state St∈Rn×d, where each column inde-
pendently follows the dynamics in (LGM).
Specifically, for theithchannel:
st,i=A t,ist−1,i+Bt,iut,i+wt,i, w t∼ N(0, Q t,i)
vt,i=k⊤
tst,i+µt,i µt∼ N(0, r t,i),
where (kt, vt)is the key-value pair at time tandvt,iis the ith
element of vt. In what follows, we focus on a single channel
and drop the subscriptifrom the state for notational clarity.
6.2. Kalman Filter for Optimal Inference
Given the model in (LGM) and a sequence of measurements
{v1, v2, . . . , v t}, the Kalman Filter computes theMaximum
A-Posteriori(MAP) estimate of the latent state at timet:
ˆst= arg max
sp(s|v 1, v2, . . . , v t),(11)
where pis the probability density function. The MAP esti-
mate is optimal in the sense that it minimizes the expected
squared error between the true state and its estimation given
all measurements up to timet.
The KF recursion.The Kalman Filter updates the state
estimate recursively as new measurements arrive. At time t,

the update is:
ˆst=A tˆst−1+Btut|{z }
Predicted state+G t(Innovationz }| {
vt,i−k⊤
th
Atˆst−1+Btuti
| {z }
Predicted state),
(12)
where theinnovationmeasures the discrepancy between the
actual measurement vtand the predicted measurement based
on the predicted state estimate.
TheKalman gain Gtdetermines how much to trust the
new measurement versus the predicted state. It is computed
as follows:
Gt=h
AtΣt−1AT
t+Q ti
kt
k⊤
th
AtΣt−1AT
t+Q ti
kt+rt.(13)
Theerror covariance Σtquantifies the uncertainty in the
state estimate. It represents the covariance of the estimation
error (st−ˆst)conditioned on all measurements up to time
t. The covariance is updated as:
Σt=
I−G tk⊤
t
AtΣt−1AT
t+Q t
(14)
Equations (12),(13) and(14) constitute the KF recursion.
We initialize with ˆs0= 0 andΣ0=σI n, where Inis the
n×n identity matrix and σrepresents our prior uncertainty
about the state before observing any measurements.
6.3. Gated KalmaNet: A Steady-State Dynamical
System for Large-Scale Training
Despite its optimality, the KF recursion in its most general
form is inherently sequential; each update depends on the
previous state estimate. This sequential dependency prevents
the parallelization necessary for efficient large-scale training
on modern hardware.
To enable parallelization, we make a key simplifying
assumption: the underlying state remains static over time.
This reduces the problem fromtrackinga dynamic state to
estimatinga fixed but unknown parameter from sequential
noisy measurements. Formally, we assume a steady-state
model:
st=st−1
vt,i=k⊤
tst+µt, µ t∼ N(0, r t),(15)
where At=In,Bt= 0, andwt= 0(i.e., no state evolution,
no control input, and no process noise).
Adapting to evolving context.While the steady-state
assumption may initially seem restrictive, contexts naturally
evolve as topics change, GKA addresses this through adap-
tive weighting (Section 4.1). By assigning higher weights to
recent measurements, older observations are naturally fadedout over time, allowing the model to track shifting context
despite the static formulation.
Under this simplification, the KF recursion reduces to:
ˆst= ˆst−1+G t(vt,i−k⊤
tˆst−1).
Gt=Σt−1kt
k⊤
tΣt−1kt+rt.
Σt=
I−G tk⊤
t
Σt−1.(16)
Collecting all channels, these equations can be written com-
pactly in matrix form as shown in (KF) .3. A key insight
of this work is that the KF recursion for the steady-state
model admits an efficient parallel implementation via chun-
ked processing (detailed in Section 4) that results in Gated
KalmaNet.
Critically, the KF recursion accounts for the entire history
when computing state estimates. The Kalman gain Gtat
each step depends on all previous measurements through
Σt−1. This contrasts with most existing SSMs, which we
show next can be viewed as approximations that ignore his-
torical dependencies when computing their gain matrices.
This principled treatment of the full history is a key advan-
tage of our approach.
6.4. Connection with Existing SSM Layers
DeltaNet[ 65] approximates the KF recursion in (16) by
assuming fixed error covariance: Σt=Infor all t. This
simplifies the Kalman gain to:
Gt=kt
k⊤
tkt+rt=kt
1 +r t,(17)
where the second equality assumes unit-normalized keys, a
common assumption in practical instantiations of DeltaNet.
Substituting (17) into the state update (16) and defining
βt= (1 +r t)−1yields:
ˆst= (I−β tktk⊤
t)ˆst−1+βtktvt,i,(DeltaNet)
which is the DeltaNet recurrence. By fixing Σt, DeltaNet
avoids tracking the evolving uncertainty in the state estimate,
a key simplification that sacrifices optimality for computa-
tional efficiency. In contrast, GKA maintains the full error
covariance Σt, allowing it to optimally weight measurements
based on the entire history.
Gated DeltaNet (GDN)[ 67] extends DeltaNet by incorpo-
rating explicit forgetting through a time-dependent decay
factor αt. Like DeltaNet, GDN can be viewed as fixing
Σt=In, but applying this approximation to the KF recur-
sion for a fading dynamical system where the state decays
over time.
3with columns of Sttransposed to being rows of Stto be consistent
with the notation in (KF) and taking the noise variancer t=1
ηt.

Specifically, GDN assumes
st=αtst−1+wt wt∼ N(0, I n)
vt,i=k⊤
tst+µt, µ t∼ N(0, r t),(18)
where αt∈[0,1] is a learned decay factor controlling how
much past information to retain. This corresponds to setting
At=αtInin(LGM) . When αt→0, the state "forgets" the
past completely; whenα t→1, the state is fully retained.
Under the identity covariance assumption Σt=In, the
Kalman gain becomes:
Gt=(α2
t+ 1)k t
(α2
t+ 1)k⊤
tkt+rt=kt
1 +r t/(α2
t+ 1),(19)
where the second equality again assumed unit-normalized
keys (as in DeltaNet). Defining βt= (1 +rt
α2
t+1)−1and
substituting into the state update (12) yields:
ˆst=αtˆst−1+βtkt(vt,i−k⊤
th
αtˆst−1i
),
=h
In−βtktk⊤
ti
αtˆst−1+βtktvt,i,(GDN)
which recovers the GDN recurrence. In practice, βtis an
input-dependent learnable parameter.
Like DeltaNet, GDN avoids tracking the evolving uncer-
tainty Σt, trading optimality for computational simplicity.
The key difference is that GDN’s explicit forgetting factor
αtprovides additional control over the memory horizon.
However, by fixing Σt=In, GDN still ignores how mea-
surement history should optimally influence the Kalman
gain, leading to suboptimal performance compared to GKA
(see Section 5).
Kimi Delta Attention (KDA)[ 56] further extends GDN
by using channel-specific decay factors αt,iin place of the
global αt. This allows different channels to have indepen-
dent memory horizons. In the KF framework, this corre-
sponds to:
st,i=αt,ist−1,i+wt,iwt,i∼ N(0, I n),(20)
for each channel i. While this added flexibility can improve
expressiveness, KDA still assumes Σt=Inand therefore
does not optimally consider the entire past when computing
its state update. Like DeltaNet and GDN, KDA sacrifices
optimality for computational simplicity.
7. Discussions and Limitations
Thanks to its expressive test-time ridge regression objective,
Gated KalmaNet extends previous fading memory layers
like Mamba2, LongHorn and Gated DeltaNet, all of which
only depend on an instantaneous test-time objective. How-
ever, GKA is only optimal among linear memory layers,
solving our test-time objective leveraging non-linear updateswhile still maintaining hardware efficiency and numerical
stability is an interesting area for future research. Despite
the efficient kernels we implemented, we believe even faster
implementations of our idea are possible, e.g., viasketching
(see Section H for preliminary results). Finally, while we
have showed promising results in combining GKA with At-
tention layers into Hybrid models (Section I), further scaling
beyond 3B parameters models is required to validate GKA
on more challenging real world problems.

References
[1]Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys John-
son, Michael Poli, James Zou, Atri Rudra, and Christopher
Ré. Zoology: Measuring and improving recall in efficient
language models.arXiv preprint arXiv:2312.04927, 2023. 2,
7, 15, 25
[2]Sangmin Bae, Bilge Acun, Haroun Habeeb, Seungyeon Kim,
Chien-Yu Lin, Liang Luo, Junjie Wang, and Carole-Jean Wu.
Hybrid architectures for language models: Systematic anal-
ysis and design insights.arXiv preprint arXiv:2510.04800,
2025. 15
[3]Maximilian Beck, Korbinian Pöppel, Markus Spanring, An-
dreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter
Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xl-
stm: Extended long short-term memory.Advances in Neural
Information Processing Systems, 37:107547–107603, 2024.
15
[4]Sneha Chaudhari, Varun Mithal, Gungor Polatkan, and Rohan
Ramanath. An attentive survey of attention models.ACM
Transactions on Intelligent Systems and Technology, 12(5):
1–32, 2021. 2
[5]Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei,
Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu,
et al. Minimax-m1: Scaling test-time compute efficiently
with lightning attention.arXiv preprint arXiv:2506.13585,
2025. 15
[6]Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi
Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu,
et al. Scienceagentbench: Toward rigorous assessment of
language agents for data-driven scientific discovery.arXiv
preprint arXiv:2410.05080, 2024. 15
[7]Hao Cui, Zahra Shamsi, Gowoon Cheon, Xuejian Ma, Shu-
tong Li, Maria Tikhanovskaya, Peter Norgaard, Nayantara
Mudur, Martyna Plomecka, Paul Raccuglia, et al. Curie: Eval-
uating llms on multitask scientific long context understanding
and reasoning.arXiv preprint arXiv:2503.13517, 2025. 15
[8]Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell,
Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: At-
tentive language models beyond a fixed-length context. In
Proceedings of the 57th annual meeting of the association for
computational linguistics, pages 2978–2988, 2019. 15
[9]Tri Dao and Albert Gu. Transformers are SSMs: Generalized
models and efficient algorithms through structured state space
duality. InInternational Conference on Machine Learning,
2024. 1, 2, 4, 6, 15
[10] Tri Dao and Albert Gu. Transformers are ssms: Generalized
models and efficient algorithms through structured state space
duality.arXiv preprint arXiv:2405.21060, 2024. 15
[11] Soham De, Samuel L Smith, Anushan Fernando, Alek-
sandar Botev, George Cristian-Muraru, Albert Gu, Ruba
Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan,
et al. Griffin: Mixing gated linear recurrences with lo-
cal attention for efficient language models.arXiv preprint
arXiv:2402.19427, 2024. 15
[12] Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zi-
jia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu,
Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, et al.Hymba: A hybrid-head architecture for small language mod-
els.arXiv preprint arXiv:2411.13676, 2024. 15
[13] Hanpei Fang, Sijie Tao, Nuo Chen, Kai-Xin Chang, and Tet-
suya Sakai. Do large language models favor recent content? a
study on recency bias in llm-based reranking.arXiv preprint
arXiv:2509.11353, 2025. 4
[14] Lizhe Fang, Yifei Wang, Zhaoyang Liu, Chenheng Zhang,
Stefanie Jegelka, Jinyang Gao, Bolin Ding, and Yisen Wang.
What is wrong with perplexity for long-context language
modeling?arXiv preprint arXiv:2410.23771, 2024. 8
[15] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence Gold-
ing, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle Mc-
Donell, Niklas Muennighoff, Chris Ociepa, Jason Phang,
Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang
Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang,
and Andy Zou. The language model evaluation harness, 2024.
2, 7
[16] Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen.
How to train long-context language models (effectively). In
Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages
7376–7399, 2025. 8
[17] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James
Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Mil-
lidge. Zamba: A compact 7b ssm hybrid model.arXiv
preprint arXiv:2405.16712, 2024. 15
[18] Alicia Golden, Samuel Hsia, Fei Sun, Bilge Acun, Basil
Hosmer, Yejin Lee, Zachary DeVito, Jeff Johnson, Gu-Yeon
Wei, David Brooks, et al. Is flash attention stable?arXiv
preprint arXiv:2405.02803, 2024. 7
[19] Gene H Golub and Charles F Van Loan.Matrix Computations
(4th ed.). The Johns Hopkins University Press, 2013. 3
[20] Albert Gu and Tri Dao. Mamba: Linear-time sequence
modeling with selective state spaces.arXiv preprint
arXiv:2312.00752, 2023. 15
[21] Albert Gu, Karan Goel, and Christopher Ré. Efficiently mod-
eling long sequences with structured state spaces.CoRR,
abs/2111.00396, 2021. 15
[22] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao,
Atri Rudra, and Christopher Ré. Combining recurrent, convo-
lutional, and continuous-time models with linear state space
layers.Advances in neural information processing systems,
34:572–585, 2021. 15
[23] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term
memory.Neural Computation, 9(8):1735–1780, 1997. 2
[24] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch,
Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de
Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan
Clark, et al. An empirical analysis of compute-optimal large
language model training.Advances in neural information
processing systems, 35:30016–30030, 2022. 26
[25] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu
Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Gins-
burg. Ruler: What’s the real context size of your long-context
language models?arXiv preprint arXiv:2404.06654, 2024. 2,
8, 25

[26] Samy Jelassi, David Brandfonbrener, Sham M Kakade, and
Eran Malach. Repeat after me: Transformers are bet-
ter than state space models at copying.arXiv preprint
arXiv:2402.01032, 2024. 15
[27] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao,
Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench:
Can language models resolve real-world github issues?arXiv
preprint arXiv:2310.06770, 2023. 15
[28] R. E. Kalman. A new approach to linear filtering and predic-
tion problems.Journal of Basic Engineering, 82(1):35–45,
1960. 1, 15
[29] Gregory Kamradt. Needle in a haystack - pressure testing
llms. https://github.com/gkamradt/LLMTest_
NeedleInAHaystack, 2023. 25
[30] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya.
Reformer: The efficient transformer.arXiv preprint
arXiv:2001.04451, 2020. 15
[31] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng,
Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao
Zhang, and Ion Stoica. Efficient memory management for
large language model serving with pagedattention. InPro-
ceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles, 2023. 15
[32] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt
Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Guha,
Sedrick Scott Keh, Kushal Arora, et al. Datacomp-lm: In
search of the next generation of training sets for language
models.Advances in Neural Information Processing Systems,
37:14200–14282, 2024. 7
[33] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan
Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan
Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hy-
brid transformer-mamba language model.arXiv preprint
arXiv:2403.19887, 2024. 15
[34] Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone,
and Qiang Liu. Longhorn: State space models are amortized
online learners. InInternational Conference on Learning
Representations, 2025. 1, 2
[35] Stefan Ljung and Lennart Ljung. Error propagation properties
of recursive least-squares adaptation algorithms.Automatica,
21(2):157–167, 1985. 15
[36] Mark D McDonnell, Dong Gong, Amin Parvaneh, Ehsan
Abbasnejad, and Anton van den Hengel. RanPAC: Random
projections and pre-trained models for continual learning.
Advances in Neural Information Processing Systems, 2023. 3
[37] Amirkeivan Mohtashami and Martin Jaggi. Landmark atten-
tion: Random-access infinite context length for transformers.
arXiv preprint arXiv:2305.16300, 2023. 15
[38] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth
Gopal. Leave no context behind: Efficient infinite con-
text transformers with infini-attention.arXiv preprint
arXiv:2404.07143, 101, 2024. 15
[39] Elizbar A Nadaraya. On estimating regression.Theory of
Probability & Its Applications, 9(1):141–142, 1964. 2
[40] Elvis Nunez, Luca Zancato, Benjamin Bowman, Aditya Go-
latkar, Wei Xia, and Stefano Soatto. Expansion span: Combin-
ing fading memory and retrieval in hybrid state space models.
arXiv preprint arXiv:2412.13328, 2024. 8, 15[41] Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fer-
nando, Caglar Gulcehre, Razvan Pascanu, and Soham De.
Resurrecting recurrent neural networks for long sequences.
InInternational Conference on Machine Learning, pages
26670–26698. PMLR, 2023. 15
[42] Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zan-
cato, Tri Dao, Yida Wang, and Ravi Netravali. Marconi:
Prefix caching for the era of hybrid llms.arXiv preprint
arXiv:2411.19379, 2024. 15
[43] Fabian Pedregosa. Residual polynomials and the Cheby-
shev method. http://fa.bianp.net/blog/2020/
polyopt/, 2020. 4
[44] Bo Peng, Daniel Goldstein, Quentin Gregory Anthony, Alon
Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy
Ferdinan, Kranthi Kiran GV , Haowen Hou, Satyapriya Kr-
ishna, Ronald McClelland Jr., Niklas Muennighoff, Fares
Obeid, Atsushi Saito, Guangyu Song, Haoqin Tu, Ruichong
Zhang, Bingchen Zhao, Qihang Zhao, Jian Zhu, and Rui-Jie
Zhu. Eagle and finch: RWKV with matrix-valued states and
dynamic recurrence. InConference on Language Modeling,
2024. 2
[45] Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide,
Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna
Lu, William Merrill, et al. Rwkv-7" goose" with expressive
dynamic state evolution. Technical report, arXiv preprint
arXiv:2503.14456, 2025. 2
[46] Liangzu Peng and René Vidal. Mathematics of continual
learning. Technical report, arXiv:2504.17963 [cs.LG], 2025.
3
[47] Liangzu Peng, Juan Elenter, Joshua Agterberg, Alejandro
Ribeiro, and Rene Vidal. TSVD: Bridging theory and practice
in continual learning with pre-trained models. InInternational
Conference on Learning Representations, 2025. 3
[48] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis,
Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite,
Vladimir Karpukhin, Jean Maillard, et al. Kilt: a benchmark
for knowledge intensive language tasks. InProceedings of
the 2021 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language
Technologies, pages 2523–2544, 2021. 25
[49] A.H. Sayed.Fundamentals of Adaptive Filtering. Wiley,
2003. 15
[50] A.H. Sayed.Adaptive Filters. Wiley, 2011. 15
[51] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Lin-
ear transformers are secretly fast weight programmers. In
International conference on machine learning, 2021. 2
[52] Samir Shah, Francesco Palmieri, and Michael Datum. Opti-
mal filtering algorithms for fast learning in feedforward neural
networks.Neural Networks, 5(5):779–787, 1992. 3
[53] Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter,
Massimiliano Pontil, and Riccardo Grazzi. Deltaproduct: Im-
proving state-tracking in linear rnns via householder products.
Technical report, arXiv:2502.10297v6 [cs.LG], 2025. 2
[54] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing
Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive net-
work: A successor to transformer for large language models.
Technical report, arXiv:2307.08621v4 [cs.CL], 2023. 2

[55] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing
Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive net-
work: A successor to transformer for large language models.
arXiv preprint arXiv:2307.08621, 2023. 15
[56] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi
Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang,
Zhiyuan Li, et al. Kimi linear: An expressive, efficient at-
tention architecture.arXiv preprint arXiv:2510.26692, 2025.
11
[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need.Advances in Neural
Information Processing Systems, 2017. 1, 2, 15
[58] René Vidal. Attention: Self-expression is all you need, 2022.
2
[59] Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca
Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile,
Yanick Schimpf, Oliver Sieberling, Alexander Meulemans,
et al. Mesanet: Sequence modeling by locally optimal test-
time training. Technical report, arXiv:2506.05233 [cs.LG],
2025. 3, 23, 26
[60] Roger Waleffe, Wonmin Byeon, Duncan Riach, Bran-
don Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali
Hatamizadeh, Sudhakar Singh, Deepak Narayanan, et al. An
empirical study of mamba-based language models.arXiv
preprint arXiv:2406.07887, 2024. 15
[61] Ke Alexander Wang, Jiaxin Shi, and Emily B Fox. Test-
time regression: a unifying framework for designing se-
quence models with associative memory. Technical report,
arXiv:2501.12352v3 [cs.LG], 2025. 3
[62] Geoffrey S Watson. Smooth regression analysis.Sankhy ¯a:
The Indian Journal of Statistics, Series A, pages 359–372,
1964. 2
[63] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025. 6, 29
[64] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda,
and Yoon Kim. Gated linear attention transformers with
hardware-efficient training. InInternational Conference on
Machine Learning, 2024. 1, 2, 4, 6, 7, 15
[65] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and
Yoon Kim. Parallelizing linear transformers with the delta
rule over sequence length. InAdvances in Neural Information
Processing Systems, pages 115491–115522. Curran Asso-
ciates, Inc., 2024. 10, 15
[66] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and
Yoon Kim. Parallelizing linear transformers with the delta
rule over sequence length. InNeural Information Processing
Systems, 2024. 1, 2, 3, 4, 6
[67] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta
networks: Improving mamba2 with delta rule. InInterna-
tional Conference on Learning Representations, 2025. 2, 6,
7, 8, 10, 15
[68] Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel
Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen.
HELMET: How to evaluate long-context language modelseffectively and thoroughly. InInternational Conference on
Learning Representations (ICLR), 2025. 2, 8, 25
[69] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang
Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang,
Zhiping Xiao, et al. Native sparse attention: Hardware-
aligned and natively trainable sparse attention. InProceedings
of the 63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages 23078–
23097, 2025. 15
[70] Luca Zancato, Alessandro Achille, Giovanni Paolini, Alessan-
dro Chiuso, and Stefano Soatto. Stacked residuals of dy-
namic layers for time series anomaly detection.arXiv preprint
arXiv:2202.12457, 2022. 15
[71] Luca Zancato, Arjun Seshadri, Yonatan Dukler, Aditya Go-
latkar, Yantao Shen, Benjamin Bowman, Matthew Trager,
Alessandro Achille, and Stefano Soatto. B'mojo: Hybrid
state space realizations of foundation models with eidetic and
fading memory. InAdvances in Neural Information Process-
ing Systems, pages 130433–130462. Curran Associates, Inc.,
2024. 1, 7, 9, 15
[72] Guanxiong Zeng, Yang Chen, Bo Cui, and Shan Yu. Continual
learning of context-dependent processing in neural networks.
Nature Machine Intelligence, 1(8):364–372, 2019. 3

A. Related Work
Since the introduction of Self-Attention [ 57], significant research has been conducted to reduce its quadratic cost in processing
long input sequences. As models and systems scale to million-token contexts, Attention’s bottlenecks have become critical
blockers to frontier agentic applications in coding, information gathering, and scientific discovery [ 6,7,27]. Prior works have
proposed various approximation schemes to overcome these limitations. For example, Reformer [ 30] uses locality-sensitive
hashing to group tokens with similar embeddings. This enables the model to attend only to a subset of tokens rather than the
entire sequence. Other works equip Transformer models with "compressed" memory tokens that are updated dynamically and
causally over sliding windows on entire sequence chunks [ 8,37,38]. While a lot of prior work have focused on reducing
the quadratic complexity of Attention with sparse approximations [ 40,69], this work focuses on linear approximations of
Attention.
A.1. Linear Attention
Linear Attention methods approximate the Attention mechanism with constant-size recurrent dynamical systems [ 3,9,64,67].
Numerous State-Space Model (SSM) variations have been proposed, ranging from those closely resembling Linear Attention
[55] or Linear Time-Invariant dynamical systems [ 22,70], to those introducing novel adaptive or gated state updates [ 9,41,64].
Despite their differences, all SSMs follow the same basic working principle inspired by classical state-space models
[28]: they process the input sequence by maintaining afixed-sizestate that acts as a compressed (lossy) representation of all
processed tokens. Moreover, when implemented in hardware, the state must have finite precision and “fades away the past"
as more samples are processed. Successful SSM layers typically employ hardware-aware implementations that efficiently
utilize modern matrix multiplication accelerators through highly parallelizable and scalable primitives, including associative
scans [ 11,20], chunking mechanisms [ 9,64], and techniques that avoid materializing the entire state in slow high-bandwidth
memory [20].
From a modeling perspective, most Linear Attention implementations introduce data-dependent gating factors to control
the speed of their “fading” memory, balancing expressivity with scalability. For example, the transition from Mamba to
Mamba2 replaced channel-wise data-dependent gating with head-wise gating for better scalability and Tensor Cores utilization.
Input-dependent Gating has been shown to empirically improve training stability [ 1,67] and has driven the development of
Linear Attention models (e.g., from S4 [ 21] to Mamba [ 20] and from DeltaNet [ 65] to Gated DeltaNet [ 67]). In our work, we
demonstrate that gating emerges naturally as a consequence of solving a weighted least squares objective function, establishing
a connection to the favorable numerical properties classically described in the adaptive filtering literature [35, 49, 50].
A.2. Hybrid State Space Attention Models
While extending the recurrent state in SSM layers has yielded performant models, they typically underperform on tasks requiring
recall of information from the distant past [ 26,60]. Hybrid State-Space Models address this limitation by complementing
SSMs’ “fading" state with Attention layers [ 10,11,17,33]. Early architectures simply stacked SSMs and Attention layers with
different blending ratios [ 9,20,60] or replaced full Attention layers with Sliding Window Attention [ 11]. More sophisticated
designs have recently emerged [17, 71].
Notably, B’MOJO [ 71] complements SSMs’ fading state with "eidetic" memory by combining SSMs with Sliding Window
Attention (SWA) in a single layer. Within the window, tokens can attend to a selected set of past tokens that were deemed
difficult to predict using an asynchronous causal selection mechanism. B’MOJO was the first hybrid model to propose a
parallel fusion of SSM and SWA at the layer level. Subsequent works [ 2,12] have shown this parallel fusion approach to be
more performant (at equivalent compute) than the stacked approach of earlier works.
Thanks to their lower memory footprint and test-time scalability over long sequences, Hybrid architectures are expanding
into long-range agentic tasks and have recently been trained with Reinforcement Learning at scale [ 5]. When coupled with
system-level optimizations like prefix caching [ 42] and specialized inference engines [ 31], Hybrid models can increase the
number of rollouts (exploration), thereby improving end-to-end performance in Reinforcement Learning loops.

B. Forward and Backward Passes of Chebyshev Iteration (Details)
In Section 4.2 we described our chunk-wise implementation of the CH method with adaptive regularization and gating. We
now give full details omitted there.
B.1. Forward Pass
CH in Detail.We begin with describing the CH method (Algorithm 1) in more detail. Assume we have a linear system of
equations Hξ=q where His aD×D positive definite matrix. We assume Hhas its all eigenvalues lie in the interval [µ, L]
and the values ofµandLis known. Note that solving this system is equivalent to solving the following quadratic problem:
min
ξ∈RD1
2ξ⊤Hξ−ξ⊤q.(21)
The classic Chebyshev Iteration in its standard form is presented in Algorithm 1. In the initialization phase, we set ρ=L−µ
L+µ,
which is the typical convergence rate of gradient descent applied to the above quadratic problem with stepsize2
L+µ; vaguely
speaking, in this setting, this stepsize choice is optimal (e.g., that allows gradient descent to converge the fastest possible).
Algorithm 1 initializes two points, ξ−1andξ0. Here ξ−1is zero, and ξ0is a gradient step for (21) starting at ξ−1and with
stepsize2
L+µ. The final component in initialization is the weight ω0= 2. This is the starting point for the weight schedule
recursion of ωiin (weight schedule). Similarly, the initialization of ξ−1, ξ0is where we start to compute ξi, whose update
consists of (grad descent) and (momentum). Note that (grad descent) is with stepsize 2·ω i/(L+µ) . Since ωi>1, this
stepsize is strictly larger than 2/(L+µ) , the latter being the optimal stepsize for vanilla gradient descent. Such a large stepsize
alone might not guarantee convergence, but it is balanced by the (momentum) term ξi−1−ξi−2with positive weight ωi−1so
that the convergence of the Chebyshev iterative method is ensured.
Numerical Stability Considerations.Now we analyze the numerical properties of the Chebyshev Iteration. The major
computation consists of matrix-vector multiplication; in a batched parallel implementation, this turns out to be matrix-matrix
multiplication. For this, the numerical accuracy is well controlled (e.g., in Triton we could specify the accuracy in tl.dot ).
The update of ωiin (weight schedule) might raise numerical concerns as it involves division. That said, we show this division
operates in a numerically well-behaved range asω iis decreasing withiyet lower bounded by1:
Lemma 3.For anyr, we have2 =ω 0≥ ··· ≥ω r≥ω∗
1>1, whereω∗
1is defined as
ω∗
1:=2(1−p
1−ρ2)
ρ2.
As a consequence, we have4−ρ2ωi∈[2,4]for alli= 0, . . . , r.
Proof. IfL=µ , then His a scaled identity matrix, and the algorithm is simplified a lot. So we assume L > µ in what
follows. With L > µ >0 we have ρ∈(0,1) . Since ω0= 2, we have 4−ρ2ω0≥2and therefore 0< ω 1≤2. Repeating this
argument and we seeω i∈(0,2]for alli. By the definition ofω i, to showω i≤ωi−1is to show
4
4−ρ2ωi−1≤ωi−1⇔g(ω i)≤0
where gis defined as g(ω) =ρ2ω2−4ω+ 4 . Note that g(ω) has two roots, ω∗
1, as defined earlier, and ω∗
2=2(1+√
1−ρ2)
ρ2 ;
ω∗
1, ω∗
2are the two fixed points of the update (weight schedule). Observing that ω0= 2 lies in the interval (ω∗
1, ω∗
2), and
moreover, for anyi≥1, ifω i−1> ω∗
1we must have
ωi=4
4−ρ2ωi−1>4
4−ρ2ω∗
1=ω∗
1.
This proves ωi> ω∗
1for all i= 1, . . . , r . Next, since ω0= 2lies in the interval (ω∗
1, ω∗
2)where g(ω) decreases, therefore
we have ω1≤ω 0. Thus ω1lies in (ω∗
1, ω∗
2)again. We could then conclude inductively that ω∗
1< ω i≤ω i−1for all
i= 1, . . . , r.
From Lemma 3 we know that the update of ωiin (weight schedule) would not create much numerical concern in a forward
pass, as we haveω i∈[1,2]for alli. Furthermore, we can bound the rate at whichω iconverges toω∗
1:

Lemma 4.Defineκ:=L
µ. For anyi= 1, . . . , r, we have
(ωi−ω∗
1)≤Ri·(ω0−ω∗
1),
whereRis defined as
R:=κ−1
κ+ 1·√κ−1√κ+ 1.
Proof.From the update rule ofω iin (weight schedule) and the fixed point property ofω∗
1, we have
ωi−ω∗
1=4
4−ρ2ωi−1−4
4−ρ2ω∗
1
=4ρ2
(4−ρ2ωi−1)(4−ρ2ω∗
1)·(ωi−1−ω∗
1)
(i)=ρ2ω∗
1
4−ρ2ωi−1·(ωi−1−ω∗
1)
(ii)
≤ρ2wi−1ω∗
1
4·(ωi−1−ω∗
1)
(iii)
≤
1−p
1−ρ2
·(ωi−1−ω∗
1)
(iv)=κ−1
κ+ 1·√κ−1√κ+ 1
·(ωi−1−ω∗
1)
Here, (i) follows from the fact that ω∗
1is a fixed point, (ii) follows from Lemma 3 that ωi≤ωi−1, (iii) follows from the
definition ofω∗
1and the factw i−1≤2, and (iv) follows from the definitions ofκandρ. The proof is concluded by unrolling
the above recurrence.
Remark1.Here, we call Rthe linear convergence rate (orcontraction factor) of ωitoω∗
1. First-order methods for solving
Hξ=q converge at most at a rate Ra:=√κ−1√κ+1, and we see ωiconverges at an even faster rate. Numerically, assuming
κ=L
µ=1.02
0.02= 51, we then have:
R≈0.7253, R5≈0.2, R10≈0.04, R20≈0.0016, R30≈6×10−5
Ra≈0.7543, R5
a≈0.244, R10
a≈0.0597, R20
a≈0.0036, R30
a≈0.0002.
Thus, withκ= 51, the update ofω iin (weight schedule) converges in at most 20 iterations up to the bfloat16 precision.
B.2. Backward Pass
We now give details for backpropagation through the Chebyshev Iteration (Algorithm 1) via implicit differentiation.
ComputingdL
dqtanddL
dkt.First, we follow Table 1 and Lemma 2, and compute dqt. Then, given the equation (Ht+λtI)dq t=
dx∗
t, we have that
d(Ht+λtI) =−dq t(x∗
t)⊤.(22)
Thereforedλ t=tr(d(H t+λtI)) =−(x∗
t)⊤dqt. Since we setλ t=a· ∥H t∥F, this indicates
dHt=−dq t(x∗
t)⊤−a·Ht
∥Ht∥F· 
(x∗
t)⊤dqt
.(23)
Note that this expression of dHtispartial: It accounts for the upstream gradient from dqtonly and one might think of the
subsequent states all depend onH t. We will accumulate the gradients later when needed.
Now, the recursion ofH tin (CH) implies
dki=X
t≥i 
dHt+ (dH t)⊤
ki·ζt
ζi(24)
=X
t≥iζt
ζi 
−dqt⊗(x∗
t)⊤−x∗
t⊗(dq t)⊤+wtHt
ki,(25)

which proves Lemma 1. We refer the reader to Section B.2.1 for more detailed derivations ofdq tanddk t.
Derivatives for Gating.In practice we often parameterize γtin the log space to ensure numerical stability. Thus, let us first
revise our notations for this case. Letg t= logγ tandG t:=Pt
i=1gi= logQt
i=1γi
. Then the mask matrixMis
Mi,j=(
exp(G j−G i)j≥i;
0otherwise.(26)
Now, since for anyc= 1, . . . , Cwe have
Hc= exp(G c)·H 0+cX
j=1exp(G c−G j)·kjk⊤
j,(27)
for anyG iwe have the following basic derivatives:
dHc
dGi=

0c < i;
exp(G i)·H 0+Pi−1
j=1exp(G i−G j)·kjk⊤
jc=i;
−exp(G c−G i)kik⊤
i c > i,i= 1, . . . , C, c= 1, . . . , C;(28)
dHC+1
dGi=−exp(G C+1−G i)kik⊤
i i= 1, . . . , C(29)
WithdH C+1 being the aggregated gradient from the future, we have fori= 1, . . . , Cthat
dGi=C+1X
c=i⟨dHc,dHc
dGi⟩ (30)
=eGi⟨dHi, H0⟩+i−1X
j=1eGi−Gj⟨dHi, kjk⊤
j⟩ −CX
c=i+1eGc−Gi⟨dHc, kik⊤
i⟩ −eGC+1−Gi⟨dHC+1, kik⊤
i⟩(31)
=eGi⟨dHi, H0⟩+iX
j=1eGi−Gj⟨dHi, kjk⊤
j⟩ −CX
c=ieGc−Gi⟨dHc, kik⊤
i⟩ −eGC+1−Gi⟨dHC+1, kik⊤
i⟩(32)
dGC+1=eGC+1⟨dHC+1, H0⟩+CX
j=1eGC+1−Gj⟨dHC+1, kjk⊤
j⟩(33)
Note that in one of the above equations we add and subtract the term⟨dH i, kik⊤
i⟩, which will simplify the implementation.
Recall that dHt=−dq t(x∗
t)⊤−1
2·wtHtwithwt=2a·(x∗
t)⊤dqt
∥Ht∥F. In computing the derivatives of Githe first term
dqt(x∗
t)⊤is the standard term that arises in that of (Linear-SSM), which we omit here. We now focus on the second term
1
2·wtHt. This implies the gradients dGianddGC+1 are partly given respectively by (using the notations in Section 4.2.2 and
omitting some algebraic operations)
1
2· ⟨wiHi, Hi⟩ −1
2·k⊤
iBiki and1
2·eGC⟨eBC+1, H0⟩+1
2·CX
j=1eGC−Gj·k⊤
jeBC+1kj.(34)
Computing the first term ⟨wiHi, Hi⟩in parallel is easy by invoking the definition of wiand the Frobenius norm of Hiwe
stored during the forward pass. Computing the quadratic terms k⊤
iBikiandk⊤
jeBC+1kjin parallel is easy and follows from
our computation of BikiandeBC+1kifordkiin Section 4.2.2. Computing ⟨eBC+1, H0⟩is easy since we recompute the initial
statesH 0of each chunk and have them available during the backward pass, while eBC+1 is updated backwards in a for loop.
B.2.1. ComputingdL
dqtanddL
dkt.
In forward pass we solve
(Ht+λtI)xt=qt

xt= (H t+λtI)−1qt
=⇒dx t= (H t+λtI)−1
|{z}
Jqt→xtdqt (35)
Recall that the gradient is transpose of the Jacobian, thus we obtain
dL
dqt= (H t+λtI)−1dL
dxt.(36)
Thus, we can obtaindL
dqtby running a Chebyshev iteration to solve (forz) the linear system of equations
(Ht+λtI)z=dL
dxt.
Now we have
dxt=d(H t+λtI)−1qt
dxt=−(H t+λtI)−1d(Ht+λtI)(H t+λtI)−1qt
dxt=−(H t+λtI)−1d(Ht+λtI)xt
= (x⊤
t⊗ −(H t+λtI)−1)vec(d(H t+λtI))(37)
In the last equality we have used the identity vec(ABC) = (C⊤⊗A)vec(B).
Now we will compute the Jacobian ofλwith respect toH t:
λt=a||H t||F
=aq
Tr(H⊤
tHt)
=⇒dλ=adq
Tr(H⊤
tHt)
=a1
2||H t||FTr((dH t)⊤Ht+H⊤
t(dHt))
=a1
||Ht||Fvec(H t)⊤dvec(H t)(38)
Substituting (38) in (37).
dxt= (x⊤
t⊗ −(H t+λtI)−1)
vec(dH t) +a
||Ht||Fvec(I)vec(H t)⊤vec(dH t))
(39)
Thus, we can obtain vec(dL
dHt)as,
vec(dL
dHt) = (x t⊗ −(H t+λtI)−1)dL
dxt+a
||Ht||Fvec(H t)vec(I)⊤(xt⊗ −(H t+λtI)−1)dL
dxt(40)
Substituting from (36),
vec(dL
dHt) =−(x t⊗dL
dqt)−a
||Ht||Fvec(H t)vec(I)⊤(xt⊗dL
dqt)
=−(x t⊗dL
dqt)−a
||Ht||Fvec(H t)⟨dL
dqt, xt⟩(41)
Now, with gating, we haveH t=γtHt−1+ktk⊤
t. Which can be unrolled as
Hl=lX
i=0l−1Y
k=iγk
kik⊤
i (42)
We will computedL
dklfor somel≤t,
dL
dkl=X
t≥ldvec(H t)
dklvec(dL
dHt) (43)

Computingdvec(H t)
dklfor somet≥l
Ht=t−1Y
i=lγiklk⊤
l+terms indep. ofk l. (44)
Taking differentials on both sides,
dHt=t−1Y
i=lγih
dklk⊤
l+kl(dkl)⊤i
d(vec(H t)) =t−1Y
i=lγih
vec(dk lk⊤
l) +vec(k l(dkl)⊤)i
=t−1Y
i=lγih
(kl⊗I) + (I⊗k l)i
| {z }
Jkl→vec(H)dkl(45)
where in the last equality we used the identity vec(k ldk⊤
l) =dk l⊗kl= (I⊗k l)dkl.
Subsituting the Jacobian (transposed for gradients) from (45) to (43) we obtain.
dL
dkl=X
t≥lt−1Y
i=lγih
(k⊤
l⊗I) + (I⊗k⊤
l)i
vec(dL
dHt) (46)
Substituting the expression for vec(dL
dHt)into equation (46) we get:
dL
dkl=−X
t≥l t−1Y
i=lγi!h
(k⊤
l⊗I) + (I⊗k⊤
l)ih
(xt⊗dL
dqt) +a
||Ht||Fvec(H t)⟨dL
dqt, xt⟩i
=
=−X
t≥l t−1Y
i=lγi!h
(k⊤
l⊗I)(x t⊗dL
dqt) +a
||Ht||F(k⊤
l⊗I)vec(H t)⟨dL
dqt, xt⟩
+
(I⊗k⊤
l)(xt⊗dL
dqt) +a
||Ht||F(I⊗k⊤
l)vec(H t)⟨dL
dqt, xt⟩i(47)
Note that the following equations hold:
(k⊤
l⊗I)(x t⊗dL
dqt) = (k⊤
lxt⊗dL
dqt) =⟨k l, xt⟩dL
dqt
(I⊗k⊤
l)(xt⊗dL
dqt) =x t⊗k⊤
ldL
dqt=⟨k l,dL
dqt⟩xt(48)
since (A⊗B)(C⊗D) =AC⊗BD and the fact that the Kronecker products after the simplification is a scalar times a vector.
For the other terms is holds:
(k⊤
l⊗I)vec(H t) =vec(H tkl)
(I⊗k⊤
l)vec(H t) =vec(k⊤
lHt) =vec(H⊤
tkl)(49)
where we used the fact vec(AXB) = (B⊤⊗A)vec(X) and the fact that the vecoperator applied to a row vector returns
the same result as applying it on its transpose (so we go from k⊤
lHttoH⊤
tkl). Since Htis symmetric we can sum both
contributions and get twice that amount.
Eventually we get:
dL
dkl=−X
t≥l t−1Y
i=lγi!"
⟨kl, xt⟩dL
dqt+⟨k l,dL
dqt⟩xt+2a
||Ht||F⟨xt,dL
dqt⟩Htkl#
(50)

Or equivalently, collecting the terms that are linear in the gradient:
dL
dkl=−X
t≥l t−1Y
i=lγi!"
⟨kl, xt⟩dL
dqt+⟨k l,dL
dqt⟩xt+2a⟨x t,dL
dqt⟩
||Ht||FHtkl#
(51)
Note: The last term creates a dependence on klthrough Htkl, which is expected since the regularization λtcouples the
gradient computation.

C. Proof of Lemma 2
Lemma 2 describes an interesting phenomenon where, for the CH method (Algorithm 1), the gradient dˆqobtained from
implicit differentiation coincides with the exact gradient dqobtained via backpropagation (chain rule). To prove this result, one
way is to derive an analytic expression for dq(Section C.1) and then inspect the recursions. However, this can be algebraically
involved. Here, we present a clear proof based on some simple observations.
First, note that the outputξ ris linear inqand moreover there is a matrix functionp r(H)∈RD×Dsuch that
ξr=pr(H)·q.(52)
Here pr(H) is a polynomial function of Hthat encodes the Chebyshev iteration (Algorithm 1). Conversely, we understand
thatpr(H)·q can be computed by applying the Chebyshev iteration with H, q forriterations (together with other parameters
such asµ, L). Then, given the output gradientdξ r, we have
dq=p r(H)⊤·dξr=pr(H)·dξ r,(53)
where the last equality follows, since His symmetric, which implies pr(H) is symmetric. The proof is finished by observing
thatp r(H)·dξ rcan be computed via Algorithm 1 withH=H, q=dξ rand other parameters, which gives usdq.
C.1. The Exact Backward Pass fordqanddH
Here we show how to obtain the exact gradients ofdHanddqin Algorithm 1 given the output gradientdξ r, which might be
of independent interests. The key insight here is that the Chebyshev iteration can bereversed.
Backward Pass for dq.Let Ibe the identity matrix of suitable size. To derive a backward pass of Algorithm 1, we first write
down the update ofξ iconcisely in the following recursion
ξi=A iξi−1+biξi−2+ciq,(54)
whereA i, bi, ciare defined as
Ai=ωiI−2·ω i
L+µH, b i=−(ω i−1), c i=2·ω i
L+µ.(55)
Note that Aiis symmetric. Define dξi:=dL
dξifor every i. With some loss function L, assume we are now given dξr, and our
goal is to compute dq:=dL
dq. Since qappears in (54) for every i, we know ξ0, ξ1, . . . , ξ rall depend on q. Therefore, with
c0:=2
L+µ, we have
dq=rX
i=0ci·dξi.
It remains to computedξ ifor everyi. Applying the chain rule to (54), we obtain
dξr−1=A r·dξr
dξi−2=A i−1·dξi−1+bi·dξi,∀i=r, . . . ,2.(56)
Note that Ai, bi, cidepend on some constant terms and ωi. Thus, to compute them backward we assume access to ωrand
these constants. By reversing (weight schedule) we derive the following recursion:
νr←ω r
νi−1←4
ρ2
1−1
νi
,∀i=r, . . . ,1.(57)
Similarly to how ωidecreases with iand converges to ω∗
1, we may prove νiis convergent to the other fixed point, ω∗
2, asi
decreases (and the iterate does not stop ati= 1).
Backward Pass for dA.From (54) and (55) we see that
dAi=dξ i⊗ξ⊤
i−1, dH=−2·ω i
L+µ·dA i (58)

(a) (b)
Figure 5. (a) The theoretical lower and upper bounds for the values of the divisor bithat arise in reversing Chebyshev (59); (b) The empirical
lower and upper bounds for the divisor that arises in reversing CG.
where⊗denotes the Kronecker product; this is the out product ofdξ iandξ⊤
i−1, asdξ iandξ i−1are vectors.
Reverse Chebyshev Iteration.At first glance, computing dAi=dξ i⊗ξ⊤
i−1requires storing ξi−1in the forward pass, and
the actual calculation of dAiis done after we run the backward pass for dξiin (56). However, storing all ξi’s would be
memory-inefficient. To address this issue, a main insight here is that we can reverse (54) and write
ξi−2=1
bi(ξi−Aiξi−1+ciq).(59)
This implies that we can recover all the iterates ξr, . . . , ξ 0as soon as we have access to the last two, ξr, ξr−1. Therefore, to
obtaindA i, we can run two iteration schemes in (56) and (59) simultaneously.
Remark2.We find that being able to run the iterative updatebackwardin a numerically stable fashion is a main feature of
the Chebyshev iterative method (or more generally, gradient descent variants with momentum). Vanilla gradient descent can
not efficiently reverse its iterate ξi=ξi−1−γi(Hξi−1−q) with stepsize γi, as it requires inverting (I−γ iH). Moreover,
reversing (59) can be done stably, as biis often in a good numerical range, which means division by biin (59) is not an issue.
To see this, first note that by Lemma 3 we have
1≥ −b i=ωi−1≥ω∗
1−1.
Note that ω∗
1defined in Lemma 3 is an increasing function of ρand therefore of κ. We then have that −bi∈[0.25,1] for any
κ≥10 (we will not consider the case κ <10 as this means we need to add a very large regularization strength which might
harm the minimization of the regression loss). In comparison, if we were to reverse the CG iteration, we would need to divide
a quantity that is often numerically as small as 10−3or as large as 1010(see Fig. 5). This is why it is numerically unstable to
reverse CG.
D. Experimental Setup
Model Configurations.We consider models of 3 different sizes: 440M, 1B, and 2.8B. This is summarized in Table 3. All
models are with the GPT2 tokenizer similarly to [59].
Training Configurations.All models are trained with the AdamW optimizer with initial learning rate 10−3,5%warm-up
steps, cosine schedule, gradient clipping with maximum norm1.
Models of the same scale use the same training configurations. Specifically (see also Table 4):
• For 440M models, we use sequence length 2048 and 8B DCLM tokens.
• For 1B models, we use sequence length 2048 and 20B DCLM tokens.
• For 2.8B models, we use sequence length 4096 and 100B DCLM tokens.

Algorithm 2:Backward Pass of Chebyshev Iteration
1Input:H, dξ r,L, µ, , number of iterationsr, the final weightω r;
2Initializeρ←L−µ
L+µ, dξr+1←0,ν r←ω r, νr+1←0, ν 0←1,dq←2νr
L+µ·dξr,dH← −2νr
L+µdξr⊗ξ⊤
r−1;
3Fori=r, . . . ,1:
ξi−2← −1
νi−1
ξi−
νiI−2·νi
L+µH
ξi−1+2·νi
L+µq
(60)
dξi−1←
νi·dξi−2·νi
L+µ·H·dξ i
−(ν i+1−1)·dξ i+1 (61)
νi−1←4
ρ2
1−1
νi
(62)
dq←dq+2νi−1
L+µ·dξi−1 (63)
dH←dH−2νi−1
L+µ·dξi−1⊗ξ⊤
i−2 (64)
Output:dq,dH;
Table 3. Model sizes and the corresponding architectural configurations.
Model Size Number of Layers Number of Heads Hidden Dimension
440M 28 8 1024
1B 28 12 1536
2.8B 32 20 2560
Table 4. Model sizes and the corresponding architectural configurations.
Model Size Global Batch Size Total Number of Training Tokens Sequence Length
440M 1M 8B 2048
1B 2M 20B 2048
2.8B 2M 100B 4096
Model Hyperparameters.We use default parameters for all other models as given in the Flash-Linear-Attention v0.4.0 library
(except the ones mentioned in Table 3). For our approach, we use λt= 0.02· ∥H t∥F, with gating and α-connection enabled
by default, unless otherwise specified. We also run CH for30iterations for all experiments.
Individual Experiments.We now describe the setups for each individual experiment.
In Fig. 1a, we randomly generate tensorsk∈RB×T×H×Dandqand normalize them along the last dimension (D). Here
B, T, H, D simulate the batch size, sequence length, number of heads, and head dimension, respectively. Then we compute
the covariance matrices H∈RB×T×H×D×Dofk, normalize its every D×D slice by its Frobenius norm. The code to
generate data is shown below.
k = torch.randn(B, T, H, D).to(dtype).to(’cuda’)
q = torch.randn(B, T, H, D).to(dtype).to(’cuda’)
q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).to(q)
k = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True).to(k)
kk = torch.einsum(’...i,...j->...ij’, k, k).cumsum(1)
kk = kk / torch.linalg.matrix_norm(kk, ord=’fro’)[..., None, None]

kk.diagonal(dim1=-2, dim2=-1).add_(ridge_strength)
For Fig. 1c and Fig. 1e, we generate random input ids with vocabulary size 5000, sequence length 2048 within a 5-layer
LLAMA; we set 2 heads and head dimension 128 for this architecture.
In the MQAR experiments of Fig. 3a, we follow the standard experimental setting but consider a strictly harder setting with
smaller model dimension (or hidden dimension). Indeed, in the setting of [ 1], the model dimension is always larger than or
equal to the number of KV pairs, while in the setting here, in some cases the model dimension is smaller than the number of
KV pairs, in which case linear SSMs could not perfectly memorize all KV pairs.
In the main paper, Fig. 3a is without any gating orα-connection.
In Fig. 4 we considered the following tasks for long context evaluations. Reported results for each task is average over the
score obtained for individual datasets in that task.
•Retrieval-Augmented Generation (RAG): These tasks consist of open-domain question answer where the model is given
a gold passage (passage containing the answer) interspersed between many other retrieved passages from a corpus [ 48,
Wikipedia dump split into 100-word passages]. The model is tasked with answering the question based on the obtained
passages. We consider the following datasets from HELMET [ 68] for this task: Natural Questions, TriviaQA, PopQA,
HotpotQA.
•Many-shot In-Context Learning (ICL): ICL tests LLMs ability to learn new skills from a few examples. Here the task is
to learn to classify between different concepts based on several in-context examples of the said concept. We consider the
following datasets from HELMET [68] for this task: TREC Coarse, TREC Fine, NLU, BANKING77, CLINIC150.
•Synthetic Recall: These tasks are variations of the “Needle-in-a-Haystack" task [ 29] where the goal is to retrieve an important
piece of information, the “needle" from a long context of distractor tokens, the “haystack". These variations also test
multi-hop tracing and aggregation capabilities of the model. We consider the following datasets from RULER [ 25] for this
task: S-NIAH-1/2/3, MK-NIAH-1,2,3, MV-NIAH, MQ-NIAH, VT, CWE, FWE.
•LongQA: These are long document based question-answering tasks. The documents are typically made long by randomly
sampling different paragraphs from the same dataset along with the paragraph that contains the answer. We consider the
following datasets from RULER [25] for this task: SQuAD, HotpotQA.
E. How Does The Performance GKA Scale with Compute?
We consider models at three different scales: 440M, 1B and 2.8B. For training configurations and architecture refer to
Section D. We use prototypical tasks from LM-Harness (see Section 5.3.1 for list of tasks) to evaluate language modeling
capabilities of GKA and compare with baseline SSM/fading memory layers. Table 5 shows that at 440M scale, GKA is
competitive with GDN and Deltanet. However, differences emerge at larger scales, with GKA showing increasing benefits. In
particular, the retrieval capabilities of our model, as measured by FDA and SWDE consistently outperform all SSM baselines
at 1B and 2.8B scale. We also report the results of equal-sized Transformer for completeness, which serves as a performance
ceiling at each scale.

Table 5.GKA shows stronger scaling with compute that other SSM baseline models.LM-Harness results for models at different
scales: 440M, 1B and 2.8B. All models were trained from scratch. 440M and 1B models were trained on 8B and 20B tokens respectively in
accordance to the Chinchila scaling laws [24]. For the 2.8B model we trained on 100B tokens.
Model ARC-C ARC-E BoolQ COPA HellaSWAG PIQA SciQ Winogrande FDA SWDE Avg
acc_n↑acc_n↑acc↑acc↑acc_n↑acc_n↑acc_n↑acc↑contains↑contains↑
440M Models
Transformer 24.40 42.26 59.88 70.00 36.19 64.15 61.5051.70 5.17 35.64 45.09
Gated Linear Attention 24.06 40.28 56.57 71.00 32.70 62.24 57.80 50.67 1.00 9.18 40.55
Gated DeltaNet 25.1741.96 58.2372.0036.9664.6963.6 51.71.91 11.88 42.81
DeltaNet 25.09 41.9261.1365.00 37.20 64.47 64.0049.49 2.81 14.31 42.54
Gated KalmaNet (Ours) 24.5743.2256.94 71.00 37.2264.47 62.8 50.83 1.45 14.04 42.65
1B Models
Transformer 26.62 46.42 59.9477.0044.01 67.1468.3054.068.35 45.18 49.70
Mamba2 28.0746.63 60.21 70.00 44.57 67.57 65.50 54.30 1.45 15.75 45.40
Gated Linear Attention 25.94 42.00 58.84 70.00 36.34 63.60 58.20 51.85 1.45 10.53 41.88
Gated DeltaNet 27.0547.9859.54 74.00 44.27 67.36 66.2 53.83 2.18 17.82 46.02
DeltaNet 27.56 46.25 59.97 71.00 43.18 67.74 65.9055.413.09 20.61 46.07
Gated KalmaNet (Ours) 25.43 46.5560.7374.00 44.59 68.8867.60 52.41 6.17 21.87 46.82
2.8B Models
Transformer 32.25 56.1064.2880.00 60.96 73.56 79.50 61.7258.53 72.28 63.92
Mamba2 32.24 59.64 58.72 82.00 62.23 73.78 79.80 62.19 7.71 41.13 55.94
Gated Linear Attention 27.82 50.80 52.57 78.00 48.83 70.13 69.60 54.54 2.81 20.43 47.55
Gated DeltaNet 32.59 60.0262.75 82.00 62.8 74.32 80.6 62.35 8.26 44.28 57.00
DeltaNet 32.8558.16 42.51 81.00 61.13 73.78 43.90 61.72 11.80 46.08 51.29
Gated KalmaNet (Ours) 32.51 59.89 61.6885.00 63.84 74.81 83.2 64.1712.89 50.95 58.89
F. Ablations
In this section we consider ablations for various modeling choices made in arriving at our final GKA model. For all ablations,
we consider 2.8B models trained on 100B tokens on DCLM at 4K context length (unless mentioned otherwise). We use the
same architecture and training configurations for these ablations as mentioned in Section D.
F.1. Does Adaptive Regularization Help?
As discussed in Section 4.1, we introduced adaptive regularization to control the condition number of HT+λtIfor numerical
stability. Here we ablate this choice, specifically we compare the following runs.
1.Adaptive regularization. We train a model withλ t=a||H t||F. We report results fora= 0.02for this run.
2.Constant regularizationWe train same model architecture (as above) with λt= 0.25 (a constant). This choice of 0.25 is
motivated from concurrent work [59] which explored a similar ridge regression objective for LLM training.
As shown in Fig. 6, without strict condition number control, gradient norms spike during training, leading to increased
cross entropy loss (compared to the run with adaptive regularization).
0 10000 20000 30000 40000 50000
Training Iteration0246810Cross Entropy Lossconstant regularization
adaptive regularization
0 10000 20000 30000 40000 50000
Training Step0246810Gradient Normconstant regularization
adaptive regularization
Figure 6.Adaptive regularization results in smoother and better training curves.(a) Plots the training curve for 2.8B models on 100B
tokens from DCLM. (b) Plots the corresponding gradient norm. The model with constant regularization (red curve) results in a higher loss
that can be attributed to its non-smooth trajectory over the course of its training run (spiky gradient norms).

F.2. Does Adaptive Weighting Help?
In Section 4.1, we discussed increasing the expressivity of our layer by introducing adaptive weights ηt,iwhich re-weigh the
past to be exponentially decaying in time. Given constant-sized memory, we hypothesize this adaptive weighting (gating)
allows GKA to learn an effective representation by incorporating recency bias into its computation. In this subsection we test
this hypothesis. We carry out the following runs.
1.Adaptive weighting (gating). We train a model with adaptive weights. Specifically, for all t≥i , we parameterize the weight
for theithsample at time-steptasη t,i=Qt
j=i+1γj, with eachγ j∈[0,1]learnable.
2.No weighting. We train the same model architecture as above, but with no weights. This essentially results in an unweighted
ridge regression objective obtained by settingη i= 1in (3).
Table 6 shows clear benefits of adapting weighting with improvements across the board on all LM-Harness tasks considered,
thereby validating our hypothesis.
Table 6.Adaptive weighting outperforms across the board on LM-Harness tasks.Results for 2.8B models trained on 100B tokens from
DCLM with and without adaptive weights as introduced in Section 4.1.
Adaptive Weights ARC-C ARC-E BoolQ COPA HellaSWAG PIQA SciQ Winogrande FDA SWDE Avg
acc_n↑acc_n↑acc↑acc↑acc_n↑acc_n↑acc_n↑acc↑contains↑contains↑
✗ 28.24 51.73 57.68 76 53.87 71.87 71.6 54.38 6.08 33.03 50.45
✓ 32.51 59.89 61.68 85 63.84 74.81 83.2 64.17 12.89 50.95 58.89
F.3. Doesα-connection Improve Training of GKA?
In Section 4.3, we introduce the α-connection as a residual connection that establishes a direct path for gradient flow through
the GLA solution, improving training stability. This allows the model to fall back on the GLA solution when CH produces
poor-quality results due to non-convergence of the iterative solver within the fixed iteration budget. To validate this design
choice, we perform two runs.
R1.withα-connection. We train a model with theα-connection as shown in our GKA block in Fig. 2.
R2. without α-connection. We train the same model architecture as above, but with no αconnection. This can be simply
understood as settingα t= 1for all time-stepstin Fig. 2.
On LM-Harness, both models perform similarly, with R1 and R2 achieving aggregate scores of 58.89 and 58.39, respectively.
However, clear differences emerge under long-context evaluation, where we trained both models on an additional 25B tokens
from long documents at 128K context length. Fig. 7 shows that GKA without the α-connection exhibits inferior long-context
performance on average, with Synthetic Recall and LongQA showing major degradation.
8k 16k 32k 64k 128k
Sequence Length1416182022Average Score
RAG
8k 16k 32k 64k 128k
Sequence Length5.65.86.06.26.46.66.87.0Average Score
ICL
4k 8k 16k 32k 64k 128k
Sequence Length05101520253035Average Score
Synthetic Recall
4k 8k 16k 32k 64k 128k
Sequence Length7.510.012.515.017.520.022.525.0Average Score
Long-QA
8k 16k 32k 64k 128k
Sequence Length81012141618Average Score
Average
R1: with -connection
R2: without -connection
Figure 7.GKA without the α- connection severaly underperforms on Synthetic Recall and LongQA.On ICL all SSMs struggle to
perform better than random chance (see Fig. 4). Interestingly, although R2 exhibits poorer long-context abilities in aggregate, it outperforms
R1 on RAG by a few points.

G. Effects of Different Regularization Strengths
Recall that we proposed setting adaptive regularizationλ t=a· ∥H t∥F. We now present experiments validating this choice.
Synthetic Experiments.First, we generate data as per Fig. 1a, where the covariance matrix is normalized by its Frobenius
norm. In this case we set λt=aforavarying in {0.01,0.02,0.05,0.1} . Fig. 8 shows that themaximum regularized residual
norm(computed as the maximum of ∥(Ht+λtI)ξi−q∥ 2over all dimensions where ξiis the estimate of CH at iteration i)
decreases as we enlarge λt. This is because having a large λtreduces the condition number. The downside, though, with a
largeλ tis that it reduces the memorization capacity, namely, it might enlarge∥H tξi−q∥ 2, the true residual of interest.
(a)λt= 0.01(b)λ t= 0.02(c)λ t= 0.05(d)λ t= 0.1
Figure 8. Convergence for varying regularization strengths (batch size8, sequence length 2048, 8 heads, and head dimension 128).
GKA with different regularization strengths.We train several 2.8B models with varying regularization strength by choosing
a∈[0.01,0.02,0.05,0.1] . While performance on LM-Harness (Table 7) shows little discrepancy, we observe noticeable
differences in long-context performance—where memorization capacity matters most—(Fig. 9). Specifically, the long-context
performance of GKA improves initially as we decrease afrom 0.1→0.05 . This is expected since this increases the
memorization capacity of the model. However, decreasing further from 0.05→0.02→0.01 causes performance to decrease.
This can be attributed to the increasing condition number of the problem, which reduces the quality of the solution computed
by CH (Fig. 8).
8k 16k 32k 64k 128k
Sequence Length12.515.017.520.022.525.027.530.0Average Score
RAG
8k 16k 32k 64k 128k
Sequence Length5.255.505.756.006.256.506.757.00Average Score
ICL
4k 8k 16k 32k 64k 128k
Sequence Length05101520253035Average Score
Synthetic Recall
4k 8k 16k 32k 64k 128k
Sequence Length7.510.012.515.017.520.022.525.0Average Score
Long-QA
8k 16k 32k 64k 128k
Sequence Length81012141618Average Score
Average
GKA [0.02] GKA [0.05] GKA [0.1] GKA [0.01]
Figure 9.Long context performance GKA for different regularization strengths.The long-context performance of GKA improves
initially as we decrease afrom0.1→0.05 . This is expected since this increases the memorization capacity of the model. However,
decreasing further from 0.05→0.02→0.01 causes performance to decrease. This can be attributed to the increasing condition number of
the problem, which reduces the quality of the solution computed by CH (Fig. 8)
Table 7.Ablation over different choices of regularization strength λt=a· ∥H t∥F.Short-context performance on LM-Harness shows
little discrepancy with different regularization strengths.
a ARC-C ARC-E BoolQ COPA HellaSWAG PIQA SciQ Winogrande FDA SWDE Avg
acc_n↑acc_n↑acc↑acc↑acc_n↑acc_n↑acc_n↑acc↑contains↑contains↑
0.01 33.4558.63 62.6385.0063.36 73.99 81.40 63.14 11.1651.49 58.43
0.02 32.51 59.89 61.6885.0063.84 74.8183.20 64.17 12.8950.95 58.89
0.05 32.6861.6653.57 79.00 63.4674.8482.60 63.77 11.98 49.68 57.32
0.1 32.76 59.8563.5284.0063.95 75.08 83.2063.54 11.43 51.22 58.86

Table 8.Latent sketching increases training throughput (by up to 10%) while marginally reducing accuracy (< 1%).Training
throughput is reported in # Billion tokens/day/node. It is measured on a single H200 GPU with a batch size of 1M tokens. Our results
indicate minimal regression on LM-harness tasks but up to 10% improvement in training throughput (going from no-sketch to sketch dim
32). However, long context performance is adversely affected with sketching with up to 60% relative drop in performance. Future work will
address this by exploring the use of sketching adaptively depending on the "complexity" of the task.
Sketch dimension LM-Harness avg. Training throughput
32 57.57 8.37
no-sketch 58.89 7.65
H. Latent Sketching for Approximate Solutions
We introduce the idea of sketching from random matrix theory to further control the amount of FLOPs vs accuracy in
GKA. Sketching involves down projecting the normal equations into a low-dimensional subspace, solving the equations in
this subspace and finally up-projecting the solution back to the original space. This reduces the worst-case computational
complexity of our approach from O(D2r)toO(d2r), where d≪D andris the number of iterations in Algorithm 1. To
the best of our knowledge our work is the first one introducing sketching as a viable solution to increase efficiency of neural
network layers that are defined implicitly by the solution to an optimization problem. Sketching can be thought of as an
analogous to the Multi Latent Attention idea introduced by DeepSeek but applied to fading memory layers. Table 8 shows
preliminary results of this idea applied to GKA. Both models (no-sketch and sketch dim 32) are trained from scratch at 2.8B
scale on 100B tokens.
I. Hybrid Gated KalmaNet
As discussed in Section A.2, augmenting SSM models with Attention layers has proven to be an effective way of improving
performance on tasks that require recalling information from the distant past. In this section, we show that our Gated KalmaNet
layer can be interleaved with Attention layers to yield even stronger models. Our Hybrid GKA model is based on the Qwen3
architecture [ 63]. Namely, our Hybrid model consists of a stack of “decoder” blocks, each of which contains a sequence
mixer—either Attention or GKA—followed by an MLP. Similar to Qwen3, our Attention layers use QK normalization layers.
Our Hybrid model consists of 30 decoder blocks, 26 of which use GKA as the sequence mixer, and 4 that use Attention.
The Attention decoder blocks are at indices 6, 14, 22, and 29. Our Hybrid models follow the same training procedure as
our non-Hybrid models. Specifically, we pretrain our Hybrid model on 100B tokens with a 4K context size, followed by
fine-tuning on 25B tokens at a 128K context size.
When evaluating our pretrained Hybrid model standard NLP benchmarks, we observe that it improves substantially on
recall-oriented tasks (FDA & SWDE) compared to the non-Hybrid model4, as shown in Table 9. Further, when evaluating our
fine-tuned long-context model on tasks that require effective modeling of long-range dependencies, we observe a significant
improvement across all context lengths, as shown in Fig. 10.
Table 9.Our Hybrid GKA + Attention model improves language modeling performance.When interleaving Attention layers into
our GKA models, we observe a significant improvement on recall-oriented tasks, such as FDA and SWDE, while preserving a similar
performance on short-context tasks.
Model ARC-C ARC-E BoolQ COPA HellaSWAG PIQA SciQ Winogrande FDA SWDE Avg
acc_n↑acc_n↑acc↑acc↑acc_n↑acc_n↑acc_n↑acc↑contains↑contains↑
Gated KalmaNet (Hybrid) 33.0259.4764.0780.00 62.74 74.59 81.4064.64 53.18 72.46 64.56
Gated KalmaNet 32.5159.8961.6885.00 63.84 74.81 83.2064.17 12.89 50.95 58.89
4Note, our non-hybrid model shares the same architecture as the hybrid with the distinction that all 4 Attention layers are replaced with GKA layers.

8k 16k 32k 64k 128k
Sequence Length1520253035Average Score
RAG
8k 16k 32k 64k 128k
Sequence Length102030405060Average Score
ICL
4k 8k 16k 32k 64k 128k
Sequence Length1020304050Average Score
Synthetic Recall
4k 8k 16k 32k 64k 128k
Sequence Length10152025303540Average Score
Long-QA
8k 16k 32k 64k 128k
Sequence Length101520253035Average Score
Average
Gated KalmaNet Gated KalmaNet (Hybrid)Figure 10.Our Hybrid GKA + Attention model significantly improves performance across all long-context benchmarks compared
to our non-Hybrid model.Adding a few Attention layers to our GKA model improves long-range dependency modeling, improving
performance across all sequence lengths on RAG, ICL, Synthetic Recall, and Long-QA.