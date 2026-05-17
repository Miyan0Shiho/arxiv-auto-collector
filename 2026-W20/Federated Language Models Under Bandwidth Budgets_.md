# Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage

**Authors**: Prasanjit Dubey, Xiaoming Huo

**Published**: 2026-05-11 05:01:43

**PDF URL**: [https://arxiv.org/pdf/2605.09986v1](https://arxiv.org/pdf/2605.09986v1)

## Abstract
Training a language model on data scattered across bandwidth-limited nodes that cannot be centralized is a setting that arises in clinical networks, enterprise knowledge bases, and scientific consortia. We study the regime in which data must remain distributed across nodes, and ask what statistical guarantees are in principle achievable under explicit bandwidth budgets; we aim to characterize what is provably possible, not to demonstrate a deployment-ready system. Existing theory treats either training-time consistency or inference-time calibration in isolation, and none makes bandwidth a first-class statistical parameter. We analyze two protocols, Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first main result is an explicit high-probability KL-consistency rate for FPLD with simultaneous dependence on node count $K$, per-node sample size $n$, quantization budget $B$, probe-set size $m$, and vocabulary size $V$; bandwidth enters only through an exponentially vanishing quantization term. Our second main result is a distribution-free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack $Δ_{\mathrm{RAG}} = f_{\max}\sqrt{K^{-2}\sum_i v(B_i)}$ makes per-node retrieval bandwidth a first-class statistical parameter, with arithmetic aggregation across $K$ nodes shrinking the slack as $K^{-1/2}$ in the per-node-uniform regime. A Pinsker-type corollary composes the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify the predicted scaling along the bounds' parameters; small-scale experiments on a GPT-2 testbed illustrate that the qualitative bandwidth-accuracy tradeoff survives on a real language model. A deployment-scale empirical evaluation is out of scope.

## Full Text


<!-- PDF content starts -->

Federated Language Models Under Bandwidth
Budgets: Distillation Rates and Conformal Coverage
Prasanjit Dubey Xiaoming Huo
H. Milton Stewart School of Industrial and Systems Engineering,
Georgia Institute of Technology, Atlanta, GA 30332, U.S.A.
Abstract
Training a language model on data scattered across bandwidth-limited nodes
that cannot be centralized is a setting that arises in clinical networks, enterprise
knowledge bases, and scientific consortia. We study the regime in which data
must remain distributed across nodes, and ask what statistical guarantees arein
principle achievableunder explicit bandwidth budgets; we aim to characterize what
is provably possible, not to demonstrate a deployment-ready system. Existing theory
treats either training-time consistency or inference-time calibration in isolation, and
none makes bandwidth a first-class statistical parameter. We analyze two protocols,
Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal
RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first
main result is an explicit high-probability KL-consistency rate for FPLD with
simultaneous dependence on node count K, per-node sample size n, quantization
budget B, probe-set size m, and vocabulary size V; bandwidth enters only through an
exponentially vanishing quantization term. Our second main result is a distribution-
free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack
∆RAG =fmaxp
K−2P
iv(Bi)makes per-node retrieval bandwidth a first-class
statistical parameter, with arithmetic aggregation across Knodes shrinking the
slack as K−1/2in the per-node-uniform regime. A Pinsker-type corollary composes
the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify
the predicted scaling along the bounds’ parameters; small-scale experiments on a
GPT-2 testbed illustrate that the qualitative bandwidth–accuracy tradeoff survives
on a real language model. A deployment-scale empirical evaluation is out of scope.
1 Introduction
In several application domains, training data isscattered across locations that cannot be
centralizedfor reasons of regulation, consent, or institutional policy: hospitals, enterprise
knowledge bases, scientific consortia, and edge fleets. In each location, local resources are
limited; no single node has enough data or compute to train a domain expert on its own.
Worse, the communication link between nodes isbandwidth-constrained: a small handful of
megabits per round, not unlimited gradient exchange. Consider, as a running example, a
multi-hospital consortium where each hospital fine-tunes a local clinical language model on
its private patient records, exchanges only bandwidth-limited summaries over an existing
low-rate wide-area network (WAN), and at query time produces conformal answer sets
with provable coverage on patient questions. The hospital language models (LMs) are
1arXiv:2605.09986v1  [stat.ML]  11 May 2026

individually weak (small local datasets, modest compute, narrow domain coverage), and
gradient or weight exchange is infeasible (full-precision weight updates exceed the available
uplink and may leak patient information). We therefore ask whether such a swarm canin
principleadmit distribution-free statistical guarantees on bothwhat the trained model
predictsandwhat the deployed system says it does not know, under explicit per-uplink
budgets.
Prior work.Federated language-model distillation [Lin et al., 2020, Li and Wang, 2019,
Yao et al., 2025, Li et al., 2025] exchanges soft outputs on a shared probe set, but treats
distillation as a black-box fusion step: no statistical rate, no quantization budget, no
inference-time guarantee. Single-site conformal retrieval-augmented generation (RAG) [Li
et al., 2024, Feng et al., 2025, Chakraborty et al., 2026] delivers calibrated answer sets
but assumes one model, one corpus, one calibration set; per-node retrieval bandwidth
does not appear in their model. Federated conformal prediction [Wen et al., 2026, Xu
et al., 2025] generalizes coverage to the federated setting but is not RAG-aware and has
no retrieval-bandwidth term. No prior result spans the training-time and inference-time
halves of a bandwidth-limited federated large language model (LLM) pipeline with both
ends made statistical.
Constraints and goal.We adopt three operational constraints: (i) no gradient or
weight exchange; (ii) no data pooling; (iii) per-uplink budgets B(training-time probe),
Bi(inference-time per-node summary), and Bcal(calibration summary) are first-class.
Our goal is to characterize whether a federated LLM pipeline can admit training-time
conditional-density convergence and inference-time conformal coverage that degrade by
quantitatively explicit, vanishing functions of these budgets. Both halves are statistical,
not systems-level.
Three obstacles make a naive extension of prior work fail.
•(C1) Federated training without gradient or weight exchange.Black-box
distillation gives no rate, but quantizinglogitson a shared probe set raises a non-
trivial statistical question: do scalar-quantized logits, averaged across nodes and
distilled into a parametric student, still inherit the pooled maximum-likelihood-
estimator (MLE) rate, and how does a per-coordinate bandwidth budget enter the
bound?
•(C2) Inference-time coverage with heterogeneous retrieval and partitioned
calibration.Per-node retrieval injects a node-specific score perturbation that prior
federated conformal analyses do not account for; simultaneously, the calibration set
is itself partitioned across nodes and summarized at a separate bandwidth budget
Bcal. The two bandwidth axes pull on coverage in different ways and need separate
slack terms.
•(C3) Composing training-time error into inference-time guarantees.The
student trained at training time becomes the nonconformity scorer at inference.
Training KL error must propagate into a coverage gap, but a naive Pinsker bound
ignores the density factor of the score cumulative distribution function (CDF) that
converts a score perturbation into a probability perturbation.
2

Our key methodological move is to transmitoutputs on a shared public probe setrather
than weights or gradients. We instantiate this principle in two protocols, each addressing
one of the three challenges above, with a propagation corollary composing them.
•Federated Probe-Logit Distillation(FPLD; Algorithm 1) addresses (C1). Each node
transmits a scalar-quantized logit vector per probe context; the aggregator averages
in logit space and distills a global student. Theorem 4.1 gives a high-probability KL
rate with explicitK,n,m,B,Vdependence.
•Federated Conformal RAG(FC-RAG; Algorithm 2) addresses (C2). Each node
retrieves locally and uploads a Bi-bit summary of its score; a one-shot federated
split-conformal calibration assembles a ˆq1−αfrom Bcal-bit per-node summaries.
Theorem 5.4 gives a distribution-free coverage bound with explicit ∆ FL(calibration)
and ∆ RAG(retrieval) slacks.
•Pinsker-type propagation(Corollary 6.7) addresses (C3). Training KL error converts
to total-variation distance via Pinsker, then to a coverage gap via the score-CDF
density factor of (B5); the result is a clean ∆ train=O(√
EKL ) slack that adds to
Theorem 5.4’s coverage bound.
The multi-hospital consortium of the running example is one motivating instance our
theory covers: each hospital scalar-quantizes its per-probe logit vector at B/V bits per
coordinate, and the aggregator produces a single global student that every hospital would
use behind FC-RAG at inference. Section 7 reports small-scale numerical experiments
illustrating the predicted scaling on synthetic n-gram ground truth, on GPT-2-small
fine-tuning over WikiText-2, and on multi-domain FC-RAG over DBpedia, AG News, and
MMLU.
Contributions.
1.Two new protocols.FPLDfor federated training andFC-RAGfor federated
inference, both bandwidth-explicit and protocol-level reproducible (Section 3).
2.Theorem 1 (training-time rate).A high-probability upper bound on EX[KL(P⋆∥ˆP)]
under FPLD,
c1d
Kn+c2ρr
Vlog(V/δ)
m+c3V
K2−2B/V+εopt+εfit
(Section 4).
3.Theorem 2 (inference-time coverage).A distribution-free marginal-coverage
bound for FC-RAG of the form Pr[Y∈ C α(X)]≥1−α− 1/(ncal+ 1)−∆FL−∆RAG,
with ∆ RAG=fmaxp
(1/K2)P
iv(B i)a novel retrieval-bandwidth slack not present
in prior federated conformal analyses (Section 5).
4.Propagation corollary and numerical illustrations.A Pinsker-type corollary
(Corollary 6.7) that propagates training-time KL error into coverage gap (Sec-
tion 6); six small-scale experiments map one-to-one onto the theoretical results,
with synthetic and real-LM experiments playing distinct roles. Three synthetic
n-gram experiments (KL training rate, heterogeneous-data extension, coverage-
bound) directly verify the predicted scaling along the bound’s parameters, since
3

closed-form distillation makes the predictions exactly checkable; three small-scale
real-LM experiments (a bandwidth-tax measurement on GPT-2-small + WikiText-2,
a multi-domain FC-RAG study on DBpedia / AG News / MMLU, and an end-to-end
Pinsker propagation chain) illustrate that the qualitative tradeoffs predicted by the
theory survive on a real language model (Section 7).
Scope and what is not claimed.This paper is a theoretical feasibility study: our
contributions are the bandwidth-explicit rate and coverage bounds, and the corollary
that composes them. The experiments are reported as numerical illustrations, not as a
deployment-scale empirical evaluation. In particular, we do not claim state-of-the-art end-
task accuracy, do not benchmark systems-level performance (latency, memory footprint,
communication overhead in real networks), do not provide a privacy-accounting analysis
of probe-logit exchange, and do not study scaling beyond GPT-2-small. These extensions
are deliberately left to follow-up work.
2 Problem Setup
We study conditional next-token prediction in a federated swarm of weak language models
constrained by a bandwidth budget. This section fixes the data, communication, and
estimand formalism that Theorems 4.1, 5.4, and Corollary 6.7 will refer to; the concrete
protocols sit in Section 3.
Data and nodes.Let Vbe a finite vocabulary of size Vand let X=V≤Lbe the space
of contexts of length at most L. A predictor is any map X → ∆(V), where ∆( V) is the
simplex over tokens. We assume Knodes; node i∈ { 1, . . . , K} holds a local dataset
Di={(x(i)
j, y(i)
j)}n
j=1, drawn i.i.d. from a local distribution PioverX × V , and raw data
never leaves its node. We present the main results under thehomogeneousregime Pi=P⋆
for all i, with a heterogeneous corollary accounting for an additional1
KP
iKL(P⋆∥Pi)
drift term.
Probe set.A public unlabeled probe set Xprobe={x(l)}m
l=1⊆ X is common knowledge,
with contexts drawn i.i.d. from a probe marginal Qthat covers the target context marginal
P⋆
Xin the bounded-Radon–Nikodym sense ρ=ess supdP⋆
X/dQ <∞ . The probe set is
the only object the training-time protocol ever transmits logits on.
Retrieval corpora.For inference, node iadditionally owns a passage corpus Ciwith
|Ci|=Niand retrieves top- kipassages Zi⊆C iper query via a local retriever. The corpora
Ciare not shared, and only bandwidth-limited summaries of the per-node predictions
travel to the hub.
Communication budget.Bandwidth is the only resource we charge, and only on
uplink; downlink broadcast of the aggregated model or of a query is free, matching the
standard convention in federated theory.Training:in each of Trounds, each node
transmits a logit vector ˜ℓ(t,l)
i∈RVper probe context, at most Bbits per vector total; i.e.,
B/V bits per coordinate after clipping to [ −Lℓ, Lℓ] and scalar quantization.Inference:
per query, node iuploads a Bi-bit summary of its local prediction to the hub.Calibration:
one-shot, with node itransmitting a Bcal-bit summary of its local calibration scores
4

from which the hub reconstructs a global conformal quantile. Total training uplink is
O(KTmB); per-query inference uplink isP
iBi.
Estimands.At training time, the aggregator produces a global student ˆP:X → ∆(V)
approximating P⋆(· |x) in expected KL: Theorem 4.1 bounds EX∼P⋆
X[KL(P⋆(· |X)∥ˆP(· |
X))]. At inference time, the answer space Yis finite and discrete throughout (multiple-
choice tasks, or a bounded candidate set extracted from a top- ptruncation of the student’s
next-token distribution); the nonconformity score is s(X, y) =−log ˆPswarm(y|X )∈
[0, Smax] with Smaxenforced by the same top- ptruncation. The predictor emits a set
Cα(X)⊆ Y for which we seek Pr[Y∈ C α(X)]≥1−α− ∆ with ∆ an explicit function of
bandwidth: Theorem 5.4 makes ∆ explicit.
Adversarial nodes, differential privacy, architecture-heterogeneous clients, streaming
calibration, and open-ended generation conformal prediction are all out of scope; each is a
natural extension but none changes the statistical object we analyze.
3 Protocols: FPLD and FC-RAG
This section specifies the two protocols our theorems analyze. Both are synchronous,
single-aggregator, and charge bandwidth on uplink only.
3.1 Training: Federated Probe-Logit Distillation (FPLD)
FPLD replaces gradient or weight exchange with the exchange ofquantized logits on a
public probe set. All nodes initialize from a shared pretrained base ˆP(0). In each round,
a node fine-tunes locally, then evaluates its model on the probe set and transmits a
quantized logit vector per probe context. The aggregator averages in logit space, distills a
student on the averaged logits, and broadcasts the student back to all nodes. Algorithm 1
states the protocol.
Algorithm 1Federated Probe-Logit Distillation (FPLD)
Require: Rounds T; local epochs E; per-vector uplink budget Bbits; probe set Xprobe=
{x(l)}m
l=1; shared base ˆP(0).
1:fort= 1, . . . , Tdo
2:foreach nodei= 1, . . . , Kin parallel do
3: ˆP(t)
i←FineTune( ˆP(t−1),Di, E)▷local SGD
4:forl= 1, . . . , mdo
5:ℓ(t,l)
i←Logits( ˆP(t)
i, x(l))∈RV
6: ˜ℓ(t,l)
i←ScalarQuantize(ℓ(t,l)
i;B/Vbits/coord,[−L ℓ, Lℓ])
7:end for
8:Uplink{ ˜ℓ(t,l)
i}m
l=1to aggregator▷ Bbits per probe context
9:end for
10: ¯ℓ(t,l)←1
KPK
i=1˜ℓ(t,l)
ifor alll
11: ˆP(t)←arg min P∈F ΘPm
l=1KL 
softmax( ¯ℓ(t,l))∥P(· |x(l))
12:Broadcast ˆP(t)to all nodes
13:end for
14:return ˆP← ˆP(T)
5

Bandwidth.Total uplink is O(KTmB ). Downlink broadcast of ˆP(t)is not charged,
following standard federated convention.
Remarks.Quantizing in logit space (rather than in probability space) preserves the
classical parametric-MLE structure under the softmax link, which is what makes the KL
rate in Theorem 4.1 tractable. A round tconsumes no new samples; Tenters only by
shrinking the optimization slackε optand distillation slackε fit, not the statistical term.
3.2 Inference: Federated Conformal RAG (FC-RAG)
At inference time, each node holds its own retrieval corpus Ciand a copy of the distilled
student ˆP. Per query, every node retrieves locally, conditions on its retrieved passages, and
uploads a Bi-bit summary of its top candidates to the hub. The hub averages available
scores coordinate-wise and emits a conformal prediction set at level α. Algorithm 2 states
the per-query protocol; Section 3.3 specifies how the conformal quantile ˆq1−αis calibrated
from node summaries.
Algorithm 2Federated Conformal RAG (FC-RAG): per-query inference
Require: Query X; student ˆP; per-node corpora {Ci}and retrievers {Ri}; uplink budgets
{Bi}; conformal quantile ˆq 1−α.
1:Hub broadcastsXto all nodes (untracked)
2:foreach nodei= 1, . . . , Kin parallel do
3:Z i←R i(X, C i;ki)▷top-k iretrieval
4:A i(X)←TopCandidates( ˆP(· |X, Z i);k′)
5:s i(X, y)← −log ˆP(y|X, Z i) fory∈ A i(X)
6:˜s i(X, y)←QuantizeScore(s i(X, y);B ibits)
7:Uplink{(y,˜s i(X, y)) :y∈ A i(X)}to hub
8:end for
9:foreach candidatey∈S
iAi(X)do
10:K y← |{i:y∈ A i(X)}|
11:s swarm(X, y)←1
KyP
i:y∈A i(X)˜si(X, y)
12:end for
13:s swarm(X, y)←+∞fory /∈S
iAi(X)
14:returnC α(X) ={y∈ Y:s swarm(X, y)≤ˆq 1−α}
Bandwidth.Per-query uplink isP
iBi. Downlink ofXis not charged.
Why average scores and not probabilities.Averaging negative log-probabilities
corresponds to geometric averaging of the underlying conditional distributions, which (a)
is the canonical score aggregation rule for a product-of-experts swarm, and (b) inherits,
via the bounded score-CDF density of (B5), a stability bound that Theorem 5.4 exploits
to control the retrieval-bandwidth slack ∆ RAG.
3.3 Calibration: one-shot federated split-conformal
A calibration set Dcalof size ncalis partitioned across the Knodes. Each node computes
local scores {s(Xj, Yj)}on its share using the frozen student ˆPand the federated retrieval
6

pipeline, and summarizes them in Bcalbits, either via equispaced quantized order statistics
(our default) or via a GC-FCP / Fed-CCP coreset. The hub reconstructs an empirical
quantile ˆq1−αfrom the Ksummaries; the quantile reconstruction error enters Theorem 5.4
as the ∆ FLterm through an additiveϕ(B cal) =O(2−Bcal/bq) slack.
This one-shot design sidesteps streaming / online conformal questions; extending
FC-RAG to a streaming calibration set is a natural but out-of-scope follow-up.
4 Training-Time KL Convergence
In this section we analyze the training-time statistical behavior of Federated Probe-Logit
Distillation (FPLD). Our goal is a bound on the expected KL divergence between the target
conditional distribution P⋆(· |X) and the globally distilled student ˆP(· |X) that exposes
the interaction of the number of nodes K, per-node sample size n, quantization budget B,
probe-set size m, and vocabulary size V. The result shows that the pooled Knsamples
drive parametric MLE convergence, probe generalization contributes a ρp
Vlog(V/δ)/m
term, and the bandwidth budget enters only through an exponentially vanishingV
K2−2B/V
distortion.
We isolate the statistical structure of FPLD with six assumptions covering the para-
metric family (A1), the probe distribution (A2), the data-homogeneity regime (A3),
the local-optimization slack (A4), the quantization channel (A5), and the aggregator’s
distillation optimization (A6). Together these assumptions let the KL bound decompose
into separate sample-complexity, probe-generalization, quantization, and optimization
terms.
Assumption A1(Parametric well-specification).The target conditional distribution lies
in a parametric family: P⋆∈ F Θ={Pθ:θ∈Θ}withΘ ⊂Rdcompact, there exists
θ⋆∈int (Θ)with Pθ⋆=P⋆, and the Fisher information I(θ⋆)≻0is positive definite.
Standard smoothness (twice continuous differentiability of logp θ) holds so that the quadratic
KL expansion is valid.
Assumption A2(Probe coverage).The public probe marginal Qcovers the target context
marginalP⋆
Xwith bounded Radon–Nikodym derivative
ρ= ess supxdP⋆
X
dQ(x)<∞.
This is the change-of-measure factor we will pay when transferring probe-empirical risk to
target-distribution risk.
Assumption A3(Homogeneous data). Pi=P⋆for all nodes i= 1, . . . , K . The
heterogeneous case is treated as Corollary 4.3 below.
Assumption A4(Local optimization slack).Every node’s local optimizer terminates
within mean-squared distance εoptof the local MLE: E∥ˆθ(t)
i−ˆθMLE
i∥2≤εopt. This absorbs
the effect of finite local epochs Eand finite communication rounds T; both enter only
through the shrinkage ofε opt.
Assumption A5(Scalar quantization).Each node clips its per-probe logit vector to
[−Lℓ, Lℓ]Vand scalar-quantizes each coordinate to B/V bits using a standard uniform
quantizer. The resulting per-coordinate distortion is bounded:
E
(˜ℓi,v−ℓi,v)2
≤C q2−2B/V, C q=L2
ℓ/3,
7

and we assume the quantization errors across nodes are independent mean-zero (dithered).
Both properties are standard for subtractively dithered uniform quantization Gersho and
Gray [1992]. The dithering randomness and the probe-set sampling in (A2) are drawn
from independent random sources, both independent of the per-node training data; this
independence is what makes the parameter-space cross term in the Theorem 4.1 union
bound vanish in expectation.
Assumption A6(Distillation fit).The aggregator’s distillation step produces a student
ˆP∈ F Θwhose empirical probe-KL against the aggregated teacher is within εfitof the
infimum over FΘ. This is a controllable algorithmic nuisance that shrinks as the aggregator
trains longer.
Theorem 4.1 below pins down the dependence of the expected KL on each of the five
protocol parameters, with constants that are explicit in the boundedness levers Lℓand
the parametric Fisher curvatureλ min(I(θ⋆)).
Theorem 4.1(Training-time KL rate for FPLD).Under assumptions (A1)–(A6), there
exist absolute constants c1, c2, c3>0such that, for every δ∈(0,1), with probability at
least1−δover the local datasets and the probe draw,
EX∼P⋆
Xh
KL 
P⋆(· |X)ˆP(· |X)i
≤c1d
Kn+c2ρr
Vlog(V/δ)
m+c3V
K2−2B/V+εopt+εfit.
Here c1= Θ(1 /λmin(I(θ⋆))),c2depends only on Lℓand the Rademacher constant of
softmax-linear classes, and c3depends only on Lℓand the dithered-quantizer distortion
constant Cqof (A5). The V/K prefactor of the quantization term tracks two structural
facts: each of the Vcoordinates contributes an independent dithered error of MSE
Cq2−2B/V(giving the factor of V), and averaging Kindependent dithered errors at
the aggregator divides by K. The conversion from L2logit error to softmax KL is via a
Csisz´ ar-type Fisher-quadratic identity (no1 /qminpickup, hence no extra Vore2Lℓfactor).
The proof is provided in Appendix A.2.
Interpretation.The four terms have distinct interpretations. The first term is apooled
effective sample size: the Knlocal samples behave as a single centralized dataset from
the statistical point of view; Kandnappear symmetrically and only their product
matters, which mirrors the classical distributed-estimation regime of Shamir–Srebro and
Zhang–Duchi–Wainwright. The second term is theprobe-generalization term: it is what
you pay for never seeing the target marginal P⋆
Xduring aggregation, being forced instead
to distill on the probe marginal Q; it scales as 1 /√mwith a slowly growing logarithmic
vocabulary factor and a ρpenalty for how badly Qunder-weights rare contexts. The third
term shows thatquantization vanishes exponentiallyin the bandwidth budget: doubling
Bsquares the error, so moderate budgets already push this contribution below the other
terms. Finally, εopt+εfitareoptimization slacks controllableby the protocol: they shrink
with more local epochs E, more rounds T, and more aggregator distillation steps, and do
not interact with the statistical terms.
The proof decomposes the target KL using the aggregated teacher ¯P(· |x ) :=
softmax (¯ℓ(x)), where ¯ℓ(x) =1
KP
i˜ℓi(x) is the aggregator’s quantized average; the first
summand decomposes into an ideal-teacher piece and a quantization piece, and the second
is controlled by probe generalization plus the distillation slack. Theorem 4.1 relies on the
auxiliary lemma below.
8

Lemma 4.2(Softmax Lipschitzness in L1).For any logit vectors a, b∈RV,∥softmax (a)−
softmax(b)∥ 1≤1
2∥a−b∥ 1. (Proof: Appendix A.1.)
4.1 Heterogeneous data extension
Assumption (A3) is the cleanest setting for the rate analysis but is restrictive in practice:
real federated deployments place each node on a slightly different conditional distribution
Pi. The next corollary drops (A3) and quantifies the resulting drift.
Corollary 4.3(Heterogeneous extension).Drop (A3). In the small-drift regime where
the parametric quadratic approximation of KL near θ⋆holds (i.e., KL(P⋆∥Pi)is small
for all i), and under (A1), (A2), (A4)–(A6), the conclusion of Theorem 4.1 holds with an
extra additive bias term:
EX∼P⋆
Xh
KL(P⋆∥ˆP)i
≤c1d
Kn+c2ρr
Vlog(V/δ)
m+c3V
K2−2B/V+1
KKX
i=1KL(P⋆∥Pi)+ε opt+εfit.
The proof is provided in Appendix A.3.
4.2 Position relative to prior work
Neither FedFD Li et al. [2025] nor FedDF Lin et al. [2020] proves a rate with simultaneous
K, n, B, m, V dependence specialized to next-token conditional density estimation; both
treat distillation as a black-box fusion step with no statistical rate. Classical distributed-
estimation work such as Shamir and Srebro Shamir and Srebro [2014], Zhang, Duchi
and Wainwright Zhang et al. [2013], and Huang and Huo Huang and Huo [2019] gives
information-theoretic lower bounds, communication-efficient algorithms, and a one-step
distributed estimator, but lacks both the softmax-vocabulary structure (the Vdependence
and the quantization-through-softmax coupling) and the probe-distillation primitive.
Theorem 4.1 appears to be the first KL-consistency rate for federated logit distillation
with explicit bandwidth accounting in the LLM-flavored (K, n, B, m, V) regime.
5 Inference-Time Coverage Guarantees
In this section we analyze the inference-time statistical behavior of Federated Conformal
RAG (FC-RAG). The goal is a distribution-free marginal coverage bound for the swarm’s
prediction set Cα(X) that makes explicit how two distinct bandwidth budgets (per-
node retrieval bandwidth Biand calibration bandwidth Bcal) degrade coverage. Unlike
Theorem 4.1, which controls the student ˆPon the training side, Theorem 5.4 treats ˆPas a
black-box nonconformity scorer and asks what coverage a federated split-conformal wrapper
can deliver on top of it when per-node retrieval and calibration are both communication-
constrained.
We isolate the inference-time conformal structure with six assumptions covering
data exchangeability (B1), bounded nonconformity scores plus full candidate-set inclu-
sion (B2), the per-node retrieval-bandwidth quantization channel (B3), the federated
quantile-reconstruction primitive (B4), the score-density regularity needed to convert score
perturbations into coverage perturbations (B5), and a score-informativeness condition (B6)
used by the cardinality corollary. The two bandwidth axes (per-node retrieval bandwidth
9

Bivia (B3), and federated calibration bandwidth Bcalvia (B4)) enter the coverage bound
through distinct slack terms; (B6) is needed only for Corollary 5.6 (set size), not for the
coverage Theorem 5.4.
Assumption B1(Exchangeability).The calibration set Dcal={(Xj, Yj)}ncal
j=1and the test
point( X, Y)are i.i.d. from a joint distribution P. In particular, the nonconformity scores
s1, . . . , s ncaland the test score stest=s(X, Y)are exchangeable. The heterogeneous case
Pi̸=Pjis handled by a weighted-conformal patch in the manner of Tibshirani et al. as a
corollary and is elided in the main statement.
Assumption B2(Bounded score and full candidate-set inclusion).The nonconformity
score is uniformly bounded: s(X, y)∈[0, Smax]for all( X, y), enforced by composing the
raw score −log ˆPswarm(y|X )with top- ptruncation. We additionally assume that every
yin the test/calibration support lies inTK
i=1Ai(X), i.e. Ky=Kuniformly, so that
the swarm score is finite and computed from all Kper-node contributions. This is the
conformal analogue of the logit-clipping assumption (A5) of Theorem 4.1.
Assumption B3(Mean-zero dithered score quantization).The QuantizeScore primitive
is a subtractively dithered scalar quantizer, so that for each node ithe per-node quantized
score decomposes additively as
˜si(X, y) =s⋆
i(X, y) +ξ i(X, y),
where, conditional on X, the noise terms {ξi(X, y)}K
i=1are independent across nodes with
mean zero and bounded second moment:
E[ξi|X] = 0,E[ξ2
i|X]≤v(B i) =O 
2−2B i/bs
,
withbs>0a protocol-specific bits-per-score constant. This is the inference-time analogue
of (A5) at training: A5 specifies the same dithered scalar quantizer for the per-coordinate
logits, and the variance bound O(2−2B i/bs)is the standard distortion of subtractively dithered
scalar quantization Gersho and Gray [1992].
Assumption B4(Federated quantile reconstruction).The hub’s estimate ˆqof the pop-
ulation(1 −α)-quantile q⋆:=q⋆
1−αof the oracle calibration scores satisfies the hybrid
deviation bound
Pr
|ˆq−q⋆|> t
≤2 exp 
−c n calt2
+ϕ(B cal), ϕ(B cal) =O 
2−Bcal/bq
,
for an absolute constant c >0and a bits-per-quantile constant bq>0. The sub-Gaussian
term is the usual statistical deviation of the empirical(1 −α)-quantile about its population
counterpart; the additive ϕ(Bcal)is the federated-summary reconstruction error. Both
pieces are provided off the shelf by GC-FCP Wen et al. [2026] and Fed-CCP Xu et al.
[2025].
Assumption B5(Score-density regularity).The cumulative distribution function Fof
the oracle score s⋆admits a density fwith f(u)≤f maxforuin a fixed deterministic
neighborhood[ q⋆−r, q⋆+r]for some r > 0; the analysis verifies a posteriori that ˆq
lies inside with probability at least1 −δvia (B4)’s deviation bound. Equivalently, Fis
fmax-Lipschitz on[q⋆−r, q⋆+r].
10

Assumption B6(Score informativeness).Let ˜y∼Unif (Y)be independent of X. For all
thresholdsuin the (B5) neighborhood ofq⋆,
Pr
X[sswarm(X,˜y)≤u]≤Pr
X,Y∼P⋆
Y|X[sswarm(X, Y)≤u].
That is, scoring uniformly random candidates yields an acceptance probability no larger than
scoring true labels. The inequality holds with equality at the chance-level limit (uniform ˆP,
or any scorer independent of Ywith uniform true marginal P⋆
Y) and strictly whenever the
model is informative about the truth. It is the minimal “no worse than random guessing”
content the cardinality corollary requires; it does not enter Theorem 5.4’s coverage bound.
Theorem 5.4 below decomposes the marginal coverage gap into a finite-sample split-
conformal correction 1 /(ncal+ 1), a federated-calibration slack ∆ FLthat mixes statistical
quantile deviation with the calibration-bandwidth reconstruction cost, and a retrieval-
bandwidth slack ∆ RAGthat is the central new ingredient of the result.
Theorem 5.4(Inference-time coverage for FC-RAG).Under assumptions (B1)–(B5),
the FC-RAG prediction set Cα(X) ={y:sswarm(X, y)≤ˆq} satisfies, with probability at
least1−δover the draw of the calibration set,
Pr
Y∈ C α(X)
≥1−α−1
ncal+ 1−∆ FL−∆ RAG,
with explicit slacks
∆FL=f maxs
log(2/δ)
c ncal+f maxϕ(B cal),∆ RAG =f maxvuut1
K2KX
i=1v(B i).
The proof is provided in Appendix A.4.
Interpretation.The four subtractive terms map cleanly onto four distinct phenomena.
The−αterm is the target miscoverage level, as in standard split conformal. The
1/(ncal+ 1) term is the usual finite-sample split-conformal overshoot, inherited from Vovk–
Gammerman–Shafer Vovk et al. [2005] and Lei and Wasserman Lei and Wasserman [2014].
The ∆ FLterm is thestatistical-quantile plus federated-summary-reconstructionslack: its
first piece is the usual sub-Gaussian deviation of the empirical quantile and shrinks like
n−1/2
cal, while its second piece is the bandwidth-limited reconstruction cost ϕ(Bcal) that does
not shrink with ncal. The ∆ RAGterm is the new piece produced by bandwidth-limited
retrieval; because the hub aggregates per-node contributions arithmetically and (B3)’s
mean-zero noise structure causes independent per-node errors to partially cancel under
averaging, the penalty enters as fmaxq
1
K2P
iv(B i), scaling as Θ( K−1/2) in the per-node-
uniform regime rather than the Θ(1) of a worst-case max iψbound. This√
Kimprovement
is the quantitative payoff of arithmetic aggregation, and it depends crucially on (B3)’s
mean-zero / independent structure; without it, only the conservative data-processing
boundP
iψ(B i) is available.
The proof chains four perturbations between the actually-observed coverage Pr[Y∈
Cα(X)] and the oracle baseline Pr[s⋆
test≤q⋆]: oracle exchangeability, quantile perturbation
q⋆→ˆq, score perturbation s⋆→s swarm, and a union bound. Theorem 5.4 relies on the
auxiliary lemma below.
11

Lemma 5.5(Quantile stability under small perturbations).Let Fbe a CDF with density
fsatisfying f(u)≤f maxfor all uin an open interval Icontaining the true quantile q⋆.
Then, for anyˆq∈Iwith|ˆq−q⋆| ≤t,|F(ˆq)−F(q⋆)| ≤f max·t. (Proof: Appendix A.1.)
5.1 Set-size efficiency
The coverage bound of Theorem 5.4 can be inverted to read out the cost of bandwidth
shortfall in set-size units. The next corollary makes this operational reading explicit.
Corollary 5.6(Expected set size).Under (B1)–(B6), and assuming Yis a finite discrete
answer space,
ECα(X)≤ |Y| ·
1−α+1
ncal+1+ ∆ FL+ ∆ RAG
.
The proof is provided in Appendix A.5.
Corollary 5.6 is the operational reading of the bandwidth penalty: bandwidth-starved
swarms produce looser sets, with slack proportional to the coverage gap. In the Bi→ ∞
andBcal→ ∞ limit, ∆ FLand ∆ RAGboth vanish (up to the n−1/2
calstatistical term) and
the set size approaches the standard split-conformal baseline|Y| ·(1−α).
5.2 Position relative to prior work
We briefly contrast Theorem 5.4 with the closest prior work. TRAQ Li et al. [2024],
Conformal-RAG Feng et al. [2025], and Principled Context Engineering Chakraborty
et al. [2026] are single-site RAG protocols: they do not consider multiple nodes, cannot
accommodate federated calibration, and in particular have no ∆ RAGterm because retrieval
bandwidth does not appear in their model. At the other end of the spectrum, GC-FCP Wen
et al. [2026] and Fed-CCP Xu et al. [2025] give federated conformal guarantees but are
not RAG-specific: they assume scores are computed end-to-end on each node and are
therefore silent on per-node retrieval bandwidth Bi. Theorem 5.4 occupies the intersection.
The ∆ RAGterm is new, and the simultaneously federated-calibration-aware and retrieval-
bandwidth-aware coverage bound appears to be the first of its kind.
6 End-to-End Coverage via Pinsker Propagation
Theorems 4.1 and 5.4 stand independently: the former controls the swarm-student ˆPat
training time, the latter treats ˆPas a black-box nonconformity scorer at inference time.
This section shows that the two halves compose cleanly: the training-time expected KL
error of Theorem 4.1 translates into an additional coverage gap for Theorem 5.4 via a
Pinsker-type√
KLfactor. The result is stated as a corollary because both inputs are
already proved theorems and Pinsker is the only new analytic ingredient; a practitioner
who cares only about end-to-end coverage can read a single unified bound.
Corollary 6.7(Propagation bound).Assume (A1)–(A6) of Theorem 4.1 and (B1)–
(B5) of Theorem 5.4, with (B5) strengthened so the conditional density of s⋆(Y)given
∆(Y) :=s(Y)−s⋆(Y)is also bounded by fmaxon the same neighborhood. Suppose the
nonconformity score used in Theorem 5.4 is the truncated log-loss s(X, y) =−log ˆP(y|X )
(bounded by (B2)), with oracle counterpart s⋆(X, y) =−logP⋆(y|X ). Then the FC-RAG
12

prediction set Cα(X)satisfies, with probability at least1 −δover the calibration and
training draws,
Pr
Y∈ C α(X)
≥1−α−1
ncal+ 1−∆ FL−∆ RAG−∆ train,
where∆ FLand∆ RAG are as in Theorem 5.4 and, writing KL:=EX∼P⋆
X
KL(P⋆(· |
X)∥ ˆP(· |X))
,
∆train=f max
KL +p
2KL
.
Thefmaxmultiplier arises because the score-perturbation argument (Step 2 of the proof be-
low) converts a score-distance into a coverage perturbation via the bounded score-CDF den-
sity of (B5). The KL+√
2KLshape comes from the elementary bound EP⋆|log(P⋆/ˆP)| ≤
KL(P⋆∥ˆP) + 2 dTV(P⋆,ˆP), combined with Pinsker dTV≤p
KL/2 and Jensen on√·;
in the small- KLregime the square-root term dominates and∆ train≍f max√
2KL. The
expected KL is bounded by the right-hand side of Theorem 4.1, so explicitly
∆train≤f max
R+√
2R
, R:=c1d
Kn+c2ρr
Vlog(V/δ)
m+c3V
K2−2B/V+εopt+εfit.
The proof is provided in Appendix A.6.
The proof chain (Appendix A.6) starts from the elementary identity EP⋆|log(P⋆/ˆP)|=
KL(P⋆∥ˆP) + 2EP⋆[(log(ˆP/P⋆))+], bounds the negative-part integral by dTV(P⋆,ˆP) via
log(1 +t)≤t, then applies Pinsker pointwise and Jensen to the concave√·overXto get
EX
EP⋆|log(P⋆/ˆP)|
≤KL+√
2KL. The score-perturbation step of Theorem 5.4 (the
FTC + (B5) bound) then converts this into the additional ∆ trainslack on top of ∆ RAG.
Interpretation.If training bandwidth grows (larger B, larger probe size m, larger node
count K), the right-hand side of Theorem 4.1 shrinks, and so ∆ trainshrinks as thesquare
rootof the training KL. In particular, the three training-side levers enter coverage as
O(1/√
Kn),O(m−1/4), and O(2−B/V) respectively; the square-root slows mandKnbut
leaves the bandwidth term exponentially small. The propagation is tight up to constants
whenever the training KL is small.
Tightness.When KL(P⋆∥ˆP) is small, Pinsker’s inequality is known to be tight up to
constants (Gilardoni’s refinement improves the constant but preserves the√·scaling), and
the score-perturbation step inherits the same tightness profile on smooth one-dimensional
score families. An explicit matching-lower-bound construction at the score-CDF level is
outside the scope of this submission and is deferred to an extended journal version. For
adversarial / spiky models that violate the conditional-density clause (e.g., ∆ concentrated
on a measure-zero set in Y), only the weaker rate ∆ train=O(KL1/4) is recoverable via
Markov truncation in place of the FTC step.
7 Experiments
7.1 Overview
Our six experiments map one-to-one onto the theoretical results, and split into two
qualitatively distinct roles. The synthetic and real-LM experiments are not redundant
13

trials of the same claim: the synthetic experimentsverifythe predicted scaling, and
the real-LM experimentsillustratethat the same qualitative tradeoffs survive on a real
language model.
Synthetic experiments (verification of scaling).Three synthetic n-gram experi-
ments verify Theorem 4.1’s training rate along five axes, Corollary 4.3’s data-heterogeneity
extension, and Theorem 5.4’s coverage bound across three calibration sweeps. E1 (Sec-
tion 7.2) is reported in full in the main body; E1.5 and E2, summarized in Section 7.4,
have full results in Appendix B.9 and Appendix B.10. These are the experiments where
verification is meaningful: the closed-form ground truth lets the bounds’ predictions be
checked exactly along the parameters (K, n, m, B, V, n cal, Bi) the theory exposes.
Real-LM experiments (illustration of feasibility).Three small-scale real-LM
experiments lift the predictions onto GPT-2-small: end-to-end FC-RAG coverage on
DBpedia, AG News, and MMLU (illustrating Corollary 5.6’s set-size efficiency along
theBiaxis), a bandwidth-tax measurement on WikiText-2, and an end-to-end Pinsker-
propagation chain that connects the two theorems via Corollary 6.7. E4 (Section 7.3) is
reported in full in the main body; E3 bandwidth-decay and E5, summarized in Section 7.4,
have full results in Appendix B.11 and Appendix B.13. We frame these asfeasibility
illustrationsrather than statistical confirmation: they use a single 124M-parameter
model and a limited seed budget, and are reported to show that the theory’s qualitative
predictions are recognizable on a real LM, not to make deployment-scale claims. Large- K
and non-i.i.d. extensions fall outside the homogeneous local-data assumption (A3) and
are reported in Appendix B.8. Code, configs, and seed-lists are released with the paper.
7.2 Verifying the KL training rate
This experiment tests Theorem 4.1’s ( K, n, m, n bits, V)-dependence directly on synthetic
n-gram ground truth, where the closed-form distillation lets the predictions be checked
exactly. The ground-truth model has vocabulary V= 256 and context length k= 1;
each node fits a local MLE table with Laplace smoothing β= 0.5 and exchanges probe-
logits at B-bit quantization with clip 20, and the aggregator distills by direct logit
averaging. We sweep ( K, n, m, n bits, V) one parameter at a time, holding the others fixed
at (4,3,000,3,000,8,256), and report mean expected KL across 40 seeds. Per-point values
are tabulated in Appendix B.2.
100101
K102
101
100[KL]
1/x
emp.
102103104
n
1/x
emp.
102103104
m
1/x
emp.
2 4 8
nb
22B/V
emp.
1026×1012×102
V
V2
emp.
Figure 1: Empirical KL stays below Theorem 4.1’s additive bound across all five sweep
axes, with the predicted slopes recovered in the rate regime of each panel. Panels from
left to right sweep K,n,m,nbits, and V; dashed lines are the predicted slopes from
Theorem 4.1.
14

Empirical KL stays below the additive prediction at every point (Figure 1). The
n-sweep is the cleanest rate panel: between n= 3·104andn= 105the empirical log–log
slope is ≈ −0.81, approaching the theoretical −1 asymptote of d/(Kn). The m-sweep
saturates at 0 .4253 across all six values because the probe-generalization termp
VlogV/m
is dominated by the Laplace-smoothing-bias floor at V= 256; additional probes cannot
lower the empirical KL below this floor, which is exactly the additive-saturation pattern
Theorem 4.1 predicts.
7.3 End-to-end FC-RAG on multi-domain benchmarks
This experiment lifts Theorem 5.4’s coverage bound and Corollary 5.6’s set-size efficiency
direction onto an end-to-end real-LM FC-RAG pipeline across three benchmarks spanning
different difficulty levels: retrieval-friendly entity classification (DBpedia 4-class), news-
topic classification (AG News 4-class), and an academic-subject benchmark on which
GPT-2-small operates near chance (MMLU 4-subject). Four topic-specialized nodes each
host a GPT-2-small scoring model and a MiniLM retrieval index over a domain corpus
and score four-class multiple-choice question (MCQ) queries. Each benchmark has ∼912
balanced questions, split 50 /50 into calibration and test pools. We score MCQ options
using per-token-averaged fullname-NLL (Appendix B.3 compares letter-token vs. fullname;
the letter-token alternative reduces to chance on all three benchmarks). We sweep Biat
α= 0.1 and 3 seeds, and additionally run an α-sweep α∈ { 0.05,0.10,0.20}atBi= 32.
Per-topic lists, the exact Bigrid, retrieval-corpus construction, and the Ky=K= 4
candidate-set-inclusion argument are in Appendix B.4.
On DBpedia, coverage reaches the 0 .9 target at Bi= 32 (0 .909±0.021) with mean
set size 2 .23 out of 4 and acc@1 = 0 .604 (substantively above chance level for 4-class
classification), and stays flat through Bi= 512 (with acc@1 stabilizing at 0 .610 by Bi= 64).
AG News reproduces the same pattern at the same Bi= 32 asymptote (0 .915±0.010
coverage, set size 2 .42,acc@1 = 0 .452). Both trace the elbow at Bi= 16 (cov 0 .94, set 2 .37
on DBpedia; cov 0 .92, set 2 .49 on AG News), with set size monotonically decreasing as Bi
grows: the efficiency direction of Theorem 5.4 quantified by Corollary 5.6. A finer-grained
Bigrid (Appendix B.5) puts the DBpedia elbow at Bi∈[14,16] and the AG News elbow
earlier at [10,12].
MMLU is the hardest benchmark in our suite: GPT-2-small’s acc@1 stays at 0 .27,
only marginally above the 0 .25 chance level for 4-option MCQ. The conformal procedure
gracefully widensrather than under-covers: mean set size 3 .46 at Bi= 32, near-but-not-
equal to the trivial set |Y|= 4 (full MMLU panel in Appendix B.6). This is exactly the
desired behavior under a weak scorer: the conformal procedure absorbs scorer uncertainty
into a wider prediction set rather than into under-coverage.
7.4 Additional empirical verifications
Five supporting verifications (E1.5, E2, E3 bandwidth-decay, α-sweep across miscoverage
levels, and E5) supplement the flagship experiments E1 and E4 above. Each entry below
gives the brief takeaway with a pointer to the corresponding appendix subsection where
the full methodology, figure or table, and discussion appear.
E1.5 (heterogeneous-data extension, Corollary 4.3).On synthetic n-gram ground
truth with per-node distributions Pi=softmax (ℓ⋆+drift·ε i), sweeping K∈ { 2,4,8}and
15

2123252729
per-query bandwidth Bi (bits)0.00.20.40.60.81.0Coverage
DBpedia (4-class, fullname): coverage vs Bi
target 1 =0.90
empirical
2123252729
per-query bandwidth Bi (bits)0.00.51.01.52.02.53.03.54.0Mean set size (out of 4)
DBpedia (4-class, fullname): set size vs Bi
2123252729
per-query bandwidth Bi (bits)0.00.20.40.60.81.0Coverage
AG News (4-class, fullname): coverage vs Bi
target 1 =0.90
empirical
2123252729
per-query bandwidth Bi (bits)0.00.51.01.52.02.53.03.54.0Mean set size (out of 4)
AG News (4-class, fullname): set size vs BiFigure 2: End-to-end FC-RAG empirical coverage (left columns) and mean conformal set
size (right columns) versus per-query bandwidth Bion DBpedia 4-class (top) and AG
News 4-class (bottom), 3 seeds, α= 0.1. Target coverage is 1 −α= 0.9. Both benchmarks
reach the target byB i= 32 and remain there throughB i= 512.
drift∈ { 0,0.1,0.2,0.3,0.5,0.75,1.0}, the additive prediction is an upper bound at every
(K,drift ) point. The bound holds, the rate-regime K-axis collapses to Theorem 4.1’s
homogeneous floor at drift 0, and adding nodes counteracts drift through statistical
pooling. Full results in Appendix B.9.
E2 (coverage-bound verification, Theorem 5.4).Sweeping FC-RAG calibration
parameters ( ncal, Bi, Bcal) independently on synthetic n-gram ground truth at target
α= 0.1, Theorem 5.4’s coverage bound holds across all three axes, the predicted-LB
curves climb monotonically with bandwidth as the slack schedule predicts, and in the
operating regime Bcal≥6 coverage is indistinguishable from unquantized split-conformal.
Full results in Appendix B.10.
E3 (bandwidth tax on GPT-2-small, Theorem 4.1).Sweeping the per-node
uplink nbitsonK= 4 federated GPT-2 students on WikiText-2 reproduces the predicted
exponentially-decaying quantization tax on perplexity: the empirical curve decreases
monotonically from 93 .5 ppl at nbits= 2 through 54 .7 at nbits= 8 to the no-quant
floor of 43 .1, replicating Theorem 4.1’sV
K2−2n bitsenvelope on a real LM; the multi-seed
FPLD-vs-FedDF gap at nbits= 8 is 15 .6 ppl (FPLD 61 .01±1.19 vs. FedDF 45 .41±0.76,
16

3 seeds) with >10σstatistical separation. Bandwidth-decay figure in Appendix B.11;
multi-seed verification in Appendix B.7.
E4 (α-sweep verification across miscoverage levels, Theorem 5.4).At Bi= 32
and 3 seeds, sweeping α∈ { 0.05,0.10,0.20}on DBpedia, AG News, and MMLU directly
verifies Theorem 5.4’s coverage bound across miscoverage levels: empirical coverage tracks
1−αin every cell of the 3 ×3 grid, within ±0.015 on DBpedia and AG News and within
±0.022 on MMLU. Full table in Appendix B.12.
E5 (end-to-end propagation chain, Corollary 6.7).Varying training bandwidth
nbits∈ {2,4,6,8,12}plus an FPLD-no-quant reference and deploying the trained student
behind FC-RAG on DBpedia and AG News tests Corollary 6.7’s Pinsker propagation
slack end-to-end on a real LM. The full Theorem 4.1–Theorem 5.4–Corollary 6.7 chain
holds: training-time bandwidth determines scorer quality, which determines set size and
acc@1, but distribution-free coverage survives the chain (coverage stays in [0 .900,0.925]
on DBpedia and [0 .903,0.927] on AG News across all training bandwidths), with the
bandwidth axis manifesting in set size (AG News grows from 2 .30 at no-quant to 2 .71 at
nbits= 2) rather than coverage. Full results in Appendix B.13.
The load-bearing empirical takeaways from all six experiments are consolidated in
Appendix B.1.
8 Related Work
Federated distillation and federated LLM training.Federated distillation since
FedMD [Li and Wang, 2019] and FedDF [Lin et al., 2020] averages soft predictions on a
shared proxy set; the FedKD line [Wu et al., 2022] quantifies systems-level savings, and
FedFD [Li et al., 2025] argues for feature distillation under model heterogeneity. None
derives a rate error =f(K, n, B, m, V ) for an LLM-style conditional density problem. We
reuse the probe-set logit-distillation primitive of FedMD/FedDF and the dithered scalar-
quantization primitive of Gersho and Gray [1992], and borrow pooled-MLE intuition from
classical distributed estimation [Shamir and Srebro, 2014, Zhang et al., 2013, Huang and
Huo, 2019]; what we add is, to our knowledge, the previously absent KL-consistency rate
for federated LLM training with simultaneous ( K, n, B, m, V ) dependence (Theorem 4.1).
Conformal prediction for RAG.Conformal prediction [Vovk et al., 2005, Lei and
Wasserman, 2014, Romano et al., 2019] turns any black-box predictor into a distribution-
free set predictor. The RAG specialization is recent: TRAQ [Li et al., 2024], Conformal-
RAG [Feng et al., 2025], and Principled Context Engineering [Chakraborty et al., 2026]
all assumesingle-sitedeployment (one model, one corpus, one calibration set). We reuse
the score-stability machinery (our (B3) generalizes their retriever-stability condition) and,
to our knowledge, charge retrieval itself for happening across bandwidth-limited nodes for
the first time in this RAG line; the explicit ∆ RAGslack in Theorem 5.4 is novel.
Federated conformal prediction.GC-FCP [Wen et al., 2026] and Fed-CCP [Xu et al.,
2025] calibrate a conformal quantile from federated score summaries; their reconstruction-
error slack matches our (B4). Neither specializes to RAG or charges retrieval bandwidth.
Theorem 5.4 couples federated calibration with retrieval-bandwidth-aware score stability,
17

producing, to our knowledge, the previously absent coverage bound for federated RAG with
bandwidth-limited retrieval; Corollary 6.7 ties our training and inference contributions
together via Pinsker, which to our knowledge has no precedent in either line.
Table 1: Position vs. closest prior lines. “Train rate” = explicit f(K, n, B, m, V ) KL-
consistency rate; “ Btr” = training uplink charged; “Coverage” = distribution-free inference
coverage guarantee; “ Bi” = per-node retrieval bandwidth as first-class statistical parameter;
“Compose” = explicit Train→Infer composition.
Train rateB tr CoverageB iCompose
FedMD / FedDF / FedFD [2019, 2020, 2025] — partial — — —
Distrib. MLE [2014, 2013, 2019]✓(K, n) — — — —
TRAQ / Conf-RAG / PCE [2024, 2025, 2026] — —✓— —
GC-FCP / Fed-CCP [2026, 2025] — —✓— —
This work (Theorems 4.1, 5.4, Corollary 6.7)✓(K, n, B, m, V)✓ ✓ ✓ ✓
9 Conclusion
We studied a federated swarm of weak language models under explicit bandwidth budgets,
asking what statistical guarantees are in principle achievable. Theorem 4.1 gives a high-
probability KL-consistency rate for FPLD with simultaneous ( K, n, B, m, V ) dependence,
in which bandwidth enters only through an exponentially vanishing quantization term.
Theorem 5.4 gives a distribution-free marginal-coverage bound for FC-RAG with a novel
retrieval-bandwidth slack ∆ RAGscaling as Θ( K−1/2). Corollary 6.7 composes the two via
Pinsker, closing the loop. Synthetic experiments verify the predicted scaling along the
bound’s parameters; small-scale GPT-2 experiments illustrate the qualitative tradeoffs
on a real LM. The work’s broader societal context, reproducibility commitments, and
compute footprint are reported in Appendix C.
Limitations.The setup is deliberately narrow: finite discrete answer spaces only (open-
ended generation conformal not covered); homogeneous data in the main statements with
heterogeneity as a corollary; architecture-heterogeneity not empirically validated, though
output-space aggregation is structurally compatible with mixed local architectures sharing
a vocabulary; no adversarial-node or differential-privacy modeling; the density-ratio ρis
assumed bounded, relaxable via truncation or importance-weighted estimators. The nu-
merical experiments are illustrative, not deployment-scale: a single 124M-parameter model,
limited seed budget, three benchmarks. A clinical-domain end-to-end demonstration is
left to follow-up.
Future work.Four directions are natural. (i) Streaming federated conformal calibration,
for online deployment where the calibration set evolves. (ii) Open-ended generation
via sub-claim conformal prediction, combining our federated calibration with TRAQ-
style decomposition. (iii) Adversarial robustness: replace the homogeneity assumption
with a contamination model and quantify coverage under Byzantine nodes. (iv) Fully
decentralized aggregation without a hub, using gossip or decentralized-SGD substitutes
for the probe-logit averaging step of FPLD.
18

References
Peter L. Bartlett and Shahar Mendelson. Rademacher and Gaussian complexities: Risk
bounds and structural results.Journal of Machine Learning Research, 3:463–482, 2002.
URLhttps://www.jmlr.org/papers/v3/bartlett02a.html.
Debashish Chakraborty, Eugene Yang, Daniel Khashabi, Dawn Lawrie, and Kevin Duh.
Principled context engineering for RAG: Statistical guarantees via conformal prediction.
InAdvances in Information Retrieval, pages 537–546. Springer Nature Switzerland,
2026. ISBN 9783032213006. doi: 10.1007/978-3-032-21300-6 45.
Naihe Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, and Ga Wu. Response quality
assessment for retrieval-augmented generation via conditional conformal factuality.
InProceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR ’25, pages 2832–2836. ACM, 2025. doi:
10.1145/3726302.3730244.
Allen Gersho and Robert M. Gray.Vector Quantization and Signal Compression. Springer
US, 1992. ISBN 9781461536260. doi: 10.1007/978-1-4615-3626-0.
Cheng Huang and Xiaoming Huo. A distributed one-step estimator.Mathematical
Programming, 174(1-2):41–76, 2019. ISSN 1436-4646. doi: 10.1007/s10107-019-01369-0.
Michel Ledoux and Michel Talagrand.Probability in Banach Spaces: Isoperimetry and
Processes. Springer Berlin Heidelberg, 1991. ISBN 9783642202124. doi: 10.1007/
978-3-642-20212-4.
Jing Lei and Larry Wasserman. Distribution-free prediction bands for non-parametric
regression.Journal of the Royal Statistical Society Series B: Statistical Methodology, 76
(1):71–96, 2014. ISSN 1467-9868. doi: 10.1111/rssb.12021.
Daliang Li and Junpu Wang. FedMD: Heterogenous federated learning via model distilla-
tion, 2019. URLhttps://arxiv.org/abs/1910.03581.
Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. TRAQ: Trustworthy retrieval
augmented question answering via conformal prediction. In Kevin Duh, Helena Gomez,
and Steven Bethard, editors,Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 3799–3821, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.210. URL https:
//aclanthology.org/2024.naacl-long.210/.
Yichen Li, Xiuying Wang, Wenchao Xu, Haozhao Wang, Yining Qi, Jiahua Dong, and
Ruixuan Li. Feature distillation is the better choice for model-heterogeneous federated
learning. InAdvances in Neural Information Processing Systems, 2025. URL https:
//openreview.net/forum?id=xYik0sKYVo. arXiv:2507.10348.
Tao Lin, Lingjing Kong, Sebastian U Stich, and Martin Jaggi. Ensemble distilla-
tion for robust model fusion in federated learning. In H. Larochelle, M. Ran-
zato, R. Hadsell, M.F. Balcan, and H. Lin, editors,Advances in Neural Infor-
mation Processing Systems, volume 33, pages 2351–2363. Curran Associates, Inc.,
19

2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
18df51b97ccd68128e994804f3eccc87-Paper.pdf.
Yaniv Romano, Evan Patterson, and Emmanuel J. Candes. Conformalized quantile
regression. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch´ e Buc, E. Fox,
and R. Garnett, editors,Advances in Neural Information Processing Systems, vol-
ume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/
paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf.
Ohad Shamir and Nathan Srebro. Distributed stochastic optimization and learning. In
2014 52nd Annual Allerton Conference on Communication, Control, and Computing
(Allerton), pages 850–857. IEEE, 2014. doi: 10.1109/ALLERTON.2014.7028543.
A. W. van der Vaart.Asymptotic Statistics, volume 3 ofCambridge Series in Statistical
and Probabilistic Mathematics. Cambridge University Press, 2000. ISBN 9780521784504.
doi: 10.1017/CBO9780511802256.
Vladimir Vovk, Alexander Gammerman, and Glenn Shafer.Algorithmic Learning in a
Random World. Springer, 2005. ISBN 0387001522. doi: 10.1007/b106715.
Haifeng Wen, Osvaldo Simeone, and Hong Xing. Efficient federated conformal prediction
with group-conditional guarantees, 2026. URL https://arxiv.org/abs/2603.14198 .
Chuhan Wu, Fangzhao Wu, Lingjuan Lyu, Yongfeng Huang, and Xing Xie. Communication-
efficient federated learning via knowledge distillation.Nature Communications, 13(2032),
2022. ISSN 2041-1723. doi: 10.1038/s41467-022-29763-x.
Rui Xu, Xingyuan Chen, Wenxing Huang, Minxuan Huang, Yun Xie, Weiyan Chen, and
Sihong Xie. Federated conditional conformal prediction via generative models, 2025.
URLhttps://arxiv.org/abs/2510.13297.
Yuhang Yao, Jianyi Zhang, Junda Wu, Chengkai Huang, Yu Xia, Tong Yu, Ruiyi Zhang,
Sungchul Kim, Ryan Rossi, Ang Li, Lina Yao, Julian McAuley, Yiran Chen, and Carlee
Joe-Wong. Federated large language models: Current progress and future directions,
2025. URLhttps://arxiv.org/abs/2409.15723.
Yuchen Zhang, John C. Duchi, and Martin J. Wainwright. Communication-efficient
algorithms for statistical optimization.Journal of Machine Learning Research, 14(104):
3321–3363, 2013. URLhttp://jmlr.org/papers/v14/zhang13b.html.
20

A Proofs and auxiliary lemmas
A.1 Auxiliary lemmas
Proof of Lemma 4.2 (softmax Lipschitzness in L1).Let u=b−a . By the
fundamental theorem of calculus, softmax (b)−softmax (a) =R1
0J(a+tu)u dt, where
J(z) =diag(p)−pp⊤is the softmax Jacobian at logit z(with p=softmax (z)). The
inducedL1→L1operator norm is the maximum column-sum of|J|:
∥J∥ 1→1= max
jX
i|Jij|= max
j
pj(1−p j) +X
i̸=jpipj
= max
j2pj(1−p j)≤1/2,
with the maximum at pj= 1/2. Hence ∥J(z)u∥1≤1
2∥u∥ 1uniformly in z, and integrating
gives∥softmax(b)−softmax(a)∥ 1≤1
2∥b−a∥ 1.□
Proof of Lemma 5.5 (quantile stability).By the fundamental theorem of calculus,
F(ˆq)−F(q⋆) =Rˆq
q⋆f(u)du. Taking absolute values and using f≤f maxon the interval of
integration (which lies in Iby hypothesis) gives |F(ˆq)−F(q⋆)| ≤f max· |ˆq−q⋆| ≤f max·t.
□
A.2 Proof of Theorem 4.1
The proof decomposes the target KL using the aggregated teacher ¯P(· |x ) :=
softmax( ¯ℓ(x)), where ¯ℓ(x) =1
KP
i˜ℓi(x) is the aggregator’s quantized average:
KL 
P⋆∥ˆP
= KL 
P⋆∥¯P
+E P⋆
log¯P
ˆP
.
We control the two summands separately. The first decomposes further into an ideal-
teacher piece and a quantization piece; the second is controlled by probe generalization
plus the distillation slack.
Step 1: aggregate MLE error.Let θ⋆be the population parameter and let ˆθMLE
denote the MLE on the pooled dataset of size N:=Kn(this is the estimator a hypothetical
centralized oracle would compute from the union of the local datasets). Under (A1) and
(A3), classical parametric MLE theory (van der Vaart van der Vaart [2000], Chapter 5)
gives the Fisher expansion
ˆθMLE−θ⋆=I(θ⋆)−1·1
NNX
i=1∇θlogp(x i, yi;θ⋆) +o P
1√
N
,
and the local quadratic expansion of KL aroundθ⋆,
KL 
Pθ⋆∥PˆθMLE
=1
2(ˆθMLE−θ⋆)⊤I(θ⋆)(ˆθMLE−θ⋆) +o P 1
N
.
Taking expectations and usingE∥ ˆθMLE−θ⋆∥2
I=d/N+o(1/N),
EKL 
P⋆∥PˆθMLE
≤d
2N+o(1/N) =d
2Kn+o(1/(Kn)).
21

Writing ¯P⋆(· |x) :=softmax (¯ℓ⋆(x)) for the aggregator output in the absence of quanti-
zation (so that each node contributes its un-quantized logits from a local MLE fit), we
extend the pooled-MLE rate from ˆθMLEto¯P⋆via a Taylor expansion of the logit map.
By smoothness of θ7→ℓ θunder (A1), the per-node MLEs satisfy ˆθMLE
i−θ⋆=OP(1/√n)
independently across i(van der Vaart van der Vaart [2000], Theorem 5.39), so the
unweighted parameter average satisfies ¯θδ:=1
KP
i(ˆθMLE
i−θ⋆) = OP(1/√
Kn) with
E∥¯θδ∥2
I(θ⋆)=d/(Kn) +o(1/(Kn)). A second-order Taylor expansion of θ7→ℓ θ(x) atθ⋆
gives
¯ℓ⋆(x) =1
KKX
i=1ℓˆθMLE
i(x) =ℓ θ⋆(x) +∇ θℓθ⋆(x)· ¯θδ+O P(1/Kn),
so¯P⋆=softmax (¯ℓ⋆) =Pθ⋆+¯θδ+OP(1/Kn) inL1by softmax1
2-Lipschitzness (Lemma 4.2).
Plugging into the local KL quadratic expansion and absorbing the higher-order terms into
a constant, we obtain
EKL 
P⋆∥¯P⋆
≤c1d
Kn+ε opt,
with c1≍1/λmin(I(θ⋆)) and where the εoptabsorbs (A4) and the T-round/ E-epoch
optimization drift. For softmax-linear families, the Taylor expansion is exact (the linear
part captures ¯ℓ⋆exactly), and the intermediate model Pθ⋆+¯θδcoincides with the parameter-
averaged model.
Step 2: quantization bias.Fix a probe point x(l)and write ℓi:=ℓ(t,l)
i,˜ℓi:=˜ℓ(t,l)
i
for brevity. Let ¯ℓ=1
KP
i˜ℓiand ¯ℓ⋆=1
KP
iℓibe the quantized and un-quantized
averages. By (A5), each coordinate has E[(˜ℓi,v−ℓi,v)2]≤C q2−2B/V. The aggregator
averages Kindependent (mean-zero dithered) quantization errors, so by independence
E[(¯ℓv−¯ℓ⋆
v)2]≤C q2−2B/V/K. Summing overVcoordinates,
E∥¯ℓ−¯ℓ⋆∥2
2≤CqV
K2−2B/V.
To convert this L2logit deviation into a KL bound, we use a Csisz´ ar-type Fisher-
quadratic identity for softmax. Define f(ℓ′) := KL 
softmax (¯ℓ⋆)∥softmax (ℓ′)
. Direct
calculation gives ∇f(ℓ′) =softmax (ℓ′)−softmax (¯ℓ⋆) and Hessian ∇2f(ℓ′) =J(ℓ′) :=
diag(softmax (ℓ′))−softmax (ℓ′)softmax (ℓ′)⊤, the softmax Jacobian (equivalently, the Fisher
information of the categorical likelihood at logit ℓ′). Note f(¯ℓ⋆) = 0 and ∇f(¯ℓ⋆) = 0, so
the integral form of Taylor’s remainder applied atℓ′=¯ℓ⋆+ηyields
KL 
softmax( ¯ℓ⋆)∥softmax( ¯ℓ⋆+η)
=Z1
0(1−t)η⊤J(¯ℓ⋆+tη)η dt≤1
4∥η∥2
2,
where the inequality uses ∥J∥ 2≤1/2 uniformly: for any unit vector v,v⊤Jv=Varq(v)≤
1
4(maxv−minv )2by Popoviciu’s variance inequality, and maxv−minv≤√
2over unit
vectors (saturated at two-atom equal-mass v= (1/√
2,−1/√
2,0, . . . , 0)). Apply with
η=¯ℓ−¯ℓ⋆:
EKL ¯P⋆∥¯P
≤1
4E∥¯ℓ−¯ℓ⋆∥2
2≤CqV
4K2−2B/V=c 3V
K2−2B/V,
with c3=Cq/4 =L2
ℓ/12 depending only on the clip level Lℓ, with no 1 /qminpickup and
hence no V- ore2Lℓ-blowup as in the chi-square reverse-Pinsker route. The V/K prefactor
tracks the per-coordinate quantization MSE summed over Vcoordinates and the 1 /K
variance reduction from averagingKindependent dithered errors.
22

Step 3: probe-to-target generalization.The aggregator distills on the probe set,
hence its distillation loss estimates EX∼QKL(¯P(· |X)∥ˆP(· |X)), not the target-marginal
quantity we want. Let G={x7→KL (¯P(· |x)∥Pθ(· |x)) :θ∈Θ}. We tighten the
Rademacher complexity ofGvia the Ledoux–Talagrand contraction principle.
The KL functional g(θ) := KL(¯P∥P θ) has gradient ∇θg=Pθ−¯P(a difference of
two probability vectors), so ∥∇θg∥2≤√
2uniformly in θ, i.e. gis√
2-Lipschitz in θ
with respect to the ℓ2norm. Under (A1) and (A5), the parameter set Θ embeds in
{θ∈RV:∥θ∥∞≤L ℓ}, hence ∥θ∥2≤L ℓ√
V. The Rademacher complexity of the ℓ2-
bounded linear class is the classical ≤L ℓX2p
V/m bound Bartlett and Mendelson [2002],
whereX 2is theℓ2envelope of the input features. Composing with the√
2-Lipschitz KL
functional via Ledoux–Talagrand contraction Bartlett and Mendelson [2002], Ledoux and
Talagrand [1991],
Rm(G)≤√
2LℓX2p
V/m.
Each g∈ G is bounded pointwise by G:= 2Lℓ+logV (the standard bound on KL between
two clipped-softmax distributions). Standard Rademacher symmetrization (e.g. Bartlett
and Mendelson [2002], Theorem 8) then gives, with probability at least 1 −δ/3, for every
θ,
EX∼Qgθ(X)−1
mX
lgθ(x(l))≤2R m(G) +Gr
log(3/δ)
m≤c 2r
Vlog(V/δ)
m.
The change-of-measure from QtoP⋆
Xcosts the density-ratio factor from (A2): EX∼P⋆
Xgθ(X)≤
ρ·E X∼Qgθ(X). Combining,
EX∼P⋆
XKL(¯P∥ˆP)≤ρ· cKLQ(¯P∥ˆP) +c 2ρr
Vlog(V/δ)
m,
where cKLQdenotes the empirical probe KL.
Step 4: distillation fit.By (A6), the aggregator produces ˆPsatisfying
cKLQ(¯P∥ˆP)≤inf
θ∈ΘcKLQ(¯P∥P θ) +ε fit.
Since ¯Pis itself an element of the softmax-linear family (up to the aggregation noise
controlled in Step 2) and ¯P⋆∈ F Θby (A1), the infimum is at most the bias from Step 2,
which is already counted. So Step 4 contributes only the slackε fit.
Union bound.We combine the pieces via the local parametric expansion of (A1). For
softmax-linear FΘthe distilled student is ˆP=Pˆθwith ˆθ∈Θ, and the second-order Taylor
expansion of the KL functional atθ⋆gives
KL(P⋆∥Pˆθ) =1
2∥ˆθ−θ⋆∥2
I(θ⋆)+o 
∥ˆθ−θ⋆∥2
.
Decompose ˆθ−θ⋆=u+v, with u:=¯θ−θ⋆the per-node-averaged MLE deviation of
Step 1 and v:=ˆθ−¯θthe distillation-stage deviation. By the first-order optimality of (A6)
and the implicit function theorem applied to the empirical probe-KL functional under
softmax-linear, vis, to leading order, a Fisher-weighted empirical mean of the per-context
dithered quantization noise of (A5) plus the empirical-vs-population probe-sampling noise
23

of (A2); both sources are mean-zero conditional on uand independent of the data partition
that producesu. Consequently
E⟨u, v⟩ I(θ⋆)= 0,E∥ ˆθ−θ⋆∥2
I(θ⋆)=E∥u∥2
I+E∥v∥2
I+o(·),
i.e. the parameter-space cross term that the naive identity KL(P⋆∥ˆP) =KL(P⋆∥¯P⋆) +
EP⋆[log(¯P⋆/¯P)] +EP⋆[log(¯P/ˆP)] would generate vanishes in expectation. Step 1 controls
E∥u∥2
I≤c′
1d/(Kn) +εopt. Steps 2–4 control E∥v∥2
I: the Csisz´ ar-type Fisher-quadratic
bound of Step 2, lifted to parameter space through the empirical-distillation first-order
conditions, contributes c′
3V/K· 2−2B/V; the Rademacher complexity of Step 3 contributes
c′
2ρp
Vlog(V/δ)/m ; and (A6) contributes εfit. Substituting back into the quadratic
expansion and absorbing the factor of 1/2 into the constantsc 1, c2, c3,
EX∼P⋆
Xh
KL 
P⋆∥ˆPi
≤c1d
Kn+c 3V
K2−2B/V+c 2ρq
Vlog(V/δ)
m+ε opt+εfit.
Splitting δacross the two probabilistic events (Step 1’s MLE tail and Step 3’s Rademacher
tail) and taking a union bound completes the proof.□
A.3 Proof of Corollary 4.3 (heterogeneous data)
Proof sketch. Per-node MLEs ˆθiconverge to θproj
i:=arg min θ∈ΘKL(Pi∥Pθ), the natural
parameter for the I-projection of PiontoFΘ. For softmax-linear FΘ, the FPLD aggregator
outputs ˆP=P¯θwith parameter average ¯θ=1
KP
iθproj
i(logit averaging coincides with
parameter averaging for softmax-linear families). In finite samples, ˆθi=θproj
i+ξiwith per-
node MLE noise ξiof covariance I−1/nindependent across nodes; averaging contributes
thec1d/(Kn) statistical term as in Theorem 4.1’s Step 1, while the heterogeneity drift is
bounded by the chain below. By the parametric quadratic approximation of KL nearθ⋆
and Jensen’s inequality on∥ · ∥2
I(θ⋆),
KL(P⋆∥P¯θ)≈1
2∥¯θ−θ⋆∥2
I≤1
KX
i1
2∥θproj
i−θ⋆∥2
I≈1
KX
iKL(P⋆∥Pθproj
i).
For each i, Csisz´ ar’s Pythagorean identity for I-projection onto exponential families
(using P⋆∈ F Θfrom (A1)) gives KL(Pθproj
i∥P⋆)≤KL (Pi∥P⋆). In the small-drift regime
where the parametric quadratic approximation holds, local symmetry of KL ( KL(P∥Q) =
KL(Q∥P)+o(KL) forP, Q close) yields KL(P⋆∥Pθproj
i)≤KL (P⋆∥Pi)+o(KL). Combining,
KL(P⋆∥P¯θ)≤1
KX
iKL(P⋆∥Pi) +o(KL),
which becomes the additive drift term once the o(KL) correction is absorbed into εfit.
Steps 2–4 are unchanged.□
A.4 Proof of Theorem 5.4
Setup.Write s⋆(X, y) :=1
KPK
i=1s⋆
i(X, y) for the un-quantized swarm-mean noncon-
formity score, i.e., the score the algorithm would compute at Bi=Bcal=∞(no
quantization noise but still federated retrieval through the same per-node corpora {Ci}).
Lets⋆
j:=s⋆(Xj, Yj) on calibration point jands⋆
test:=s⋆(X, Y), and let q⋆:=q⋆
1−αbe
24

the empirical (1 −α)-quantile of {s⋆
1, . . . , s⋆
ncal,+∞}. By contrast, the actually-computed
swarm score sswarm(X, y) =1
KP
i˜si(X, y) uses the bandwidth-limited quantized per-
node summaries, and ˆqis the actually-reconstructed hub quantile. Coverage occurs iff
sswarm(X, Y)≤ˆq.
Step 1 (oracle coverage).Suppose, hypothetically, that (i) scores were un-quantized
(s=s⋆) and (ii) the centralized empirical quantile q⋆were used in place of the federated
estimate ˆq. Under (B1), the calibration scores {s⋆
j}and the test score s⋆
testare exchangeable,
so the standard split-conformal argument (cf. Vovk, Gammerman, Shafer Vovk et al.
[2005], and Lei and Wasserman Lei and Wasserman [2014]) gives
Pr
s⋆
test≤q⋆
≥1−α−1
ncal+ 1.(1)
The 1 /(ncal+ 1) term is the classical discrete-order-statistic correction: the calibration
quantile is the ⌈(1−α)(ncal+ 1)⌉-th order statistic, which slightly under-covers at the
target level in finite samples.
Step 2 (quantile perturbation q⋆→ˆq).Let Fdenote the CDF of s⋆under the joint
law of (B1). Under (B5), Fisfmax-Lipschitz in a neighborhood of q⋆. Apply Lemma 5.5
to the pair ( q⋆,ˆq):|F(ˆq)−F(q⋆)| ≤f max|ˆq−q⋆|. Now (B4) provides the deviation bound
for|ˆq−q⋆|: with probability at least 1−δover the calibration draw,
|ˆq−q⋆| ≤s
log(2/δ)
c ncal+ϕ(B cal).
Combining with Lemma 5.5,
Pr
s⋆
test≤ˆq
≥Pr
s⋆
test≤q⋆
−f maxs
log(2/δ)
c ncal−f maxϕ(B cal) = Pr
s⋆
test≤q⋆
−∆ FL,
(2)
with probability at least 1−δover the calibration draw.
Step 3 (score perturbation s⋆→s swarm).By the candidate-set inclusion hypothesis
Ky=Kof (B2) (cf. Section 3.2), both s⋆(un-quantized) and sswarm (quantized) sum
over all Knodes. By (B3)’s additive decomposition ˜si=s⋆
i+ξi, we get sswarm(X, y) =
s⋆(X, y) +¯ξ(X, y), where the aggregated noise ¯ξ:=1
KP
iξihas, by independence +
mean-zero of the per-nodeξ i,
E[¯ξ|X] = 0,E[ ¯ξ2|X] =1
K2KX
i=1E[ξ2
i|X]≤1
K2KX
i=1v(B i).
For any fixed threshold u(in particular u=ˆq, which is calibration-set-measurable and
independent of the test point under (B1)), the probability gap is controlled by the noise
via the “Lipschitz CDF” bound |Pr[sswarm≤u|X ]−Pr[s⋆≤u|X ]| ≤f maxE[|¯ξ||X], which
follows from the FTC applied to the CDF of s⋆on the fmax-bounded-density interval
guaranteed by (B5). Cauchy–Schwarz then converts the first absolute moment of ¯ξinto
25

its second moment: E[|¯ξ||X]≤p
E[¯ξ2|X]≤q
(1/K2)PK
i=1v(B i). Combining and taking
expectation overX,
|Pr[s swarm,test ≤ˆq]−Pr[s⋆
test≤ˆq]| ≤f maxvuut1
K2KX
i=1v(B i) = ∆ RAG.(3)
Hence
Pr
sswarm,test ≤ˆq
≥Pr
s⋆
test≤ˆq
−∆ RAG.(4)
Step 4 (union).Chain (1), (2), (4):
Pr
Y∈ C α(X)
= Pr
sswarm,test ≤ˆq
≥Pr
s⋆
test≤ˆq
−∆ RAG
≥Pr
s⋆
test≤q⋆
−∆ FL−∆ RAG
≥1−α−1
ncal+ 1−∆ FL−∆ RAG.
The first inequality is (4), holding deterministically given the realization of ˆqand the
score distributions. The second is (2), holding with probability at least 1 −δover the
calibration draw. The third is (1).□
A.5 Proof of Corollary 5.6 (expected set size)
Proof. Expand the cardinality as a sum of acceptance indicators and take expectation,
then convert the average to a probability under a uniformly random candidate:
E|Cα(X)|=X
y∈YPr[s swarm(X, y)≤ˆq] =|Y| ·Pr
X,˜y[sswarm(X,˜y)≤ˆq],
where ˜y∼Unif (Y) is independent of Xand the calibration draw. By (B6) applied at
u= ˆq, which lies in the (B5) neighborhood by construction,
Pr
X,˜y[sswarm(X,˜y)≤ˆq]≤Pr
X,Y[sswarm(X, Y)≤ˆq] = Pr[Y∈ C α(X)].
The right-hand side is the standard split-conformal acceptance probability. The The-
orem 5.4 chain admits a symmetric upper-tail: under (B5)’s no-ties / continuous-
density assumption, Step 1’s split-conformal lower bound is matched by Pr[s⋆
test≤q⋆]≤
1−α+ 1/(ncal+ 1), and the quantile-perturbation (Lemma 5.5) and score-perturbation
(3) bounds in Steps 2–3 are stated as absolute-value inequalities, so the upper sides match
the lower sides used in Theorem 5.4’s proof; chaining gives
Pr[Y∈ C α(X)]≤1−α+1
ncal+ 1+ ∆ FL+ ∆ RAG.
Multiplying through by|Y|gives the corollary.
26

A.6 Proof of Corollary 6.7 (training-time propagation)
Proof. The strengthening of (B5) (conditional density of s⋆(Y) given ∆( Y) bounded by
fmax) is automatic for smooth parametric models where ∆ = log(P⋆/ˆP) is locally smooth
in the parameter perturbation; both the synthetic n-gram setup of Appendix B.10 and the
softmax-linear scoring of Section 7.3 satisfy it. The Tightness paragraph in§6 discusses
the failure mode for adversarial ∆.
Step 1 (training-time TV via Pinsker + Jensen).By Pinsker’s inequality applied
pointwise, dTV(P⋆(·|X),ˆP(·|X))≤q
(1/2) KL(P⋆(·|X)∥ ˆP(·|X)) . Taking expectation
overX∼P⋆
Xand applying Jensen’s inequality to the concave function√·,
EXdTV 
P⋆(·|X), ˆP(·|X)
≤r
1
2EXh
KL 
P⋆(·|X)∥ ˆP(·|X)i
.(5)
Step 2 (score-perturbation via indicator-difference + density bound).Fix
a calibration-set-measurable threshold u(e.g., u=ˆq). Comparing the P⋆-pushforward
of the plug-in score s=−log ˆPto that of the oracle score s⋆=−logP⋆: since both
indicators1[ s≤u ] and1[ s⋆≤u] are functions of the same ( X, Y)∼P⋆
X,Y, the difference
satisfies1[s≤u]−1[s⋆≤u]≤1[ min(s, s⋆)≤u≤max(s, s⋆)].
By Corollary 6.7’s strengthening of (B5), the conditional density of s⋆(Y) given ∆( Y) =
s(Y)−s⋆(Y) =δis bounded by fmax. Conditioning on ∆ = δ, the indicator-difference
event sits in an interval of length |δ|adjacent to s⋆, soPr[min(s, s⋆)≤u≤max (s, s⋆)|
∆ =δ]≤f max|δ|by the FTC. Taking expectation over ∆,
|Pr
P⋆[s≤u]−Pr
P⋆[s⋆≤u]| ≤f maxEP⋆|s(X, Y)−s⋆(X, Y)|.
To bound EP⋆|s−s⋆|in terms of KL, write f:=log(P⋆/ˆP) so that |s−s⋆|=|f|on the
support ofP⋆. Splittingfinto positive and negative parts,
EP⋆|f|=E P⋆[f] + 2E P⋆[(−f) +] = KL 
P⋆∥ˆP
+ 2Z
ˆP>P⋆P⋆log(ˆP/P⋆)dy.
Apply log(1 + t)≤tto the integrand: P⋆log(ˆP/P⋆)≤P⋆(ˆP/P⋆−1) = ˆP−P⋆on
{ˆP > P⋆}, so
Z
ˆP>P⋆P⋆log(ˆP/P⋆)dy≤Z
ˆP>P⋆(ˆP−P⋆)dy=d TV(P⋆,ˆP).
Hence pointwise in X,EP⋆
Y|X|f| ≤KL (P⋆∥ˆP)+2dTV(P⋆,ˆP). Taking EX, applying Pinsker
dTV≤p
KL/2 pointwise, then Jensen’s inequality on the concave√·,
EP⋆
XY|s−s⋆| ≤E XKL 
P⋆∥ˆP
+q
2EXKL 
P⋆∥ˆP
.
Combining with the indicator-difference bound,
|Pr
P⋆[s≤u]−Pr
P⋆[s⋆≤u]| ≤f max
EXKL 
P⋆∥ˆP
+q
2EXKL 
P⋆∥ˆP
=: ∆ train.
27

In the small- KLregime the second summand dominates and ∆ train≍f max√2EXKL,
preserving the Pinsker√·shape of (5).
Step 3 (chaining).Inserting this score perturbation into the four-step argument
of Theorem 5.4 as an additional subtractive slack on top of ∆ RAGgives the displayed
inequality. The explicit bound on ∆ trainin the corollary statement follows by substituting
Theorem 4.1’s right-hand side for the expected KL, and taking a union bound over the
training event (probability 1 −δin Theorem 4.1) and the calibration event (probability
1−δin Theorem 5.4); splittingδacross the two is absorbed into the constants.□
B Per-experiment empirical details
B.1 Summary of empirical takeaways
The six experiments verify each load-bearing theoretical claim of the paper.
•Synthetic KL training rate matches the additive prediction.The empirical
KL tracks Theorem 4.1’s five-axis envelope; the n-sweep recovers an empirical log–log
slope of −0.81 between n= 3·104andn= 105, approaching the d/(Kn) asymptote
of−1, and the nbitssweep traces the exponential-decay envelope of the quantization
term (Section 7.2).
•Data heterogeneity is upper-bounded by the analytical drift slack.Em-
pirical KL stays below KLhomo+1
KP
iKL(P⋆∥Pi) at every ( K,drift ) point; the
drift = 0 row reduces to Theorem 4.1’s homogeneous setting, and the K-axis shows
additional nodes counteracting drift via statistical pooling (Appendix B.9).
•Synthetic coverage hugs the target across calibration sweeps.Empirical
coverage stays at the 1 −α= 0.9 target across the ncal,Bi, and Bcalaxes; the
predicted lower bound climbs monotonically with bandwidth, as the ∆ FLand ∆ RAG
slack schedule predicts (Appendix B.10).
•Bandwidth tax decays exponentially on GPT-2-small.FPLD perplexity on
WikiText-2 falls along the nbitsaxis from 93 .5 atnbits= 2 to 54 .7 atnbits= 8 to
the no-quant floor at 43 .1, replicating Theorem 4.1’sV
K2−2n bitsenvelope on a real
LM; the multi-seed FPLD-vs-FedDF gap of 15 .6 ppl at nbits= 8 is the operational
bandwidth tax at the operating point (Appendix B.11).
•End-to-end coverage holds across three domains and three miscoverage
levels.FC-RAG empirical coverage tracks the 1 −αtarget across DBpedia, AG
News, and MMLU at α∈ { 0.05,0.10,0.20}with gap ≤0.022 in every cell of the 3 ×3
verification table; set size decreases monotonically in Bi(Corollary 5.6’s efficiency
direction). DBpedia reaches acc@1 = 0 .604 at Bi= 32; MMLU stays at chance
under graceful widening (Section 7.3).
•Pinsker chain holds end-to-end on real LMs.Training FPLD at nbits∈
{2,4,6,8,12,no-quant} and deploying through FC-RAG maintains coverage in
[0.900,0.927] on DBpedia and AG News across the entire training-bandwidth range;
the bandwidth axis manifests in efficiency (set size, accuracy) rather than in coverage,
validating Corollary 6.7’s Pinsker propagation chain (Appendix B.13).
28

B.2 E1 per-point n-gram sweep tables
We tabulate the per-axis grid and full empirical mean expected KL (40 seeds, ±1σ) for the
five one-at-a-time sweeps of Section 7.2, with defaults ( K, n, m, n bits) = (4 ,3,000,3,000,8)
andV= 256 when held fixed. Tables 2–6 contain every empirical value plotted in
Figure 1; each table’s caption relates the observed pattern to the corresponding term in
Theorem 4.1’s additive bound. The nbits= 4 point dips ≈15σbelow its neighbors due to
a quantizer-step-vs-smoothing-bias accidental cancellation at clip = 20.
Table 2: Synthetic n-gram, K-sweep ( n= 3,000, m= 3,000, nbits= 8, V= 256).
Empirical KL plateaus at the εfitfloor; the d/(Kn) statistical term is dominated by the
smoothing-bias floor forKn≪d=V2= 65,536.
KMean KL 1σ
1 0.4557 0.0038
2 0.4341 0.0033
4 0.4253 0.0030
8 0.4214 0.0029
16 0.4197 0.0029
32 0.4189 0.0029
64 0.4184 0.0030
128 0.4183 0.0030
256 0.4182 0.0030
Table 3: Synthetic n-gram, n-sweep ( K= 4,m= 3,000,nbits= 8,V= 256). The cleanest
rate panel: 40 ×KL reduction across the sweep, log–log slope ≈ −0.81 between n= 3·104
andn= 105, approaching the−1 asymptote ofd/(Kn).
nMean KL 1σ
1020.4897 0.0045
3·1020.4848 0.0043
1030.4684 0.0039
3·1030.4253 0.0030
1040.3212 0.0018
3·1040.1823 0.0009
1050.0684 0.0004
B.3 Letter-token vs. fullname scoring
The MCQ scoring choice affects empirical acc@1 on every benchmark while leaving
conformal coverage at the 1 −α target. We compare two scoring methods on the
four-class MCQ setup. Letter scoring uses si(X, y) =−logp θi(“Y”|prompt ) where
Y∈ {A,B,C,D} is the single-letter token; the inter-letter score gap is then dominated by
the GPT-2-small per-letter token bias rather than by the answer content. Fullname scoring
uses the per-token-averaged NLL of the full class-name string (e.g. “company”, “athlete”,
29

Table 4: Synthetic n-gram, m-sweep ( K= 4, n= 3,000, nbits= 8, V= 256). KL
is essentially constant at 0 .4253: the probe-generalization termp
VlogV/m is already
dominated by the smoothing-bias floor at everymtested.
mMean KL 1σ
1020.4255 0.0057
3·1020.4253 0.0037
1030.4253 0.0029
3·1030.4253 0.0030
1040.4253 0.0030
3·1040.4253 0.0030
Table 5: Synthetic n-gram, nbits-sweep ( K= 4, n= 3,000, m= 3,000, V= 256).
Exponential decay in the small- Bregime ( nbits∈ {2,3,4}); the nbits= 4 point sits ≈15σ
below its neighbors, attributed to a quantizer-step ↔Laplace-smoothing-bias accidental
cancellation (clip = 20, step ∆ = 2.5 aligns with theβ= 0.5 smoothing-bias scale).
nbits Mean KL 1σ
2 0.4921 0.0045
3 0.4573 0.0057
4 0.3855 0.0027
5 0.4226 0.0030
6 0.4267 0.0030
7 0.4229 0.0029
8 0.4253 0.0030
10 0.4262 0.0030
12 0.4262 0.0030
“world”, “biology”) as the score, which removes the per-letter bias and exposes the answer-
content score gap. Both methods are valid nonconformity scores under Theorem 5.4’s
assumptions.
The takeaway is that the conformal coverage guarantee is robust to the choice of
scorer ((B6)’s monotonicity holds for both), but the prediction-set efficiency is not: a
scorer better aligned to the answer-content distribution gives smaller prediction sets at the
same coverage. Section 7.3’s fullname-scoring main-text headline reflects this efficiency
advantage on benchmarks where GPT-2-small is competent (DBpedia, AG News); MMLU
sits near chance under both scorers because the underlying scorer genuinely lacks academic-
subject competence at this model scale.
B.4 E4 benchmark construction details
The three benchmarks of Section 7.3 are constructed as follows. The DBpedia node-topics
are{company,athlete,animal,plant} ; AG News uses {world,sports,business,scitech} ;
MMLU uses high-school {statistics ,physics ,biology ,world history} . Per-topic
retrieval corpora are built from non-test splits of the same domain so a query never
retrieves its own row as context. Each query is split 50 /50 into calibration and test
pools ( ncal=ntest≈456). We sweep Bi∈ {1,2,4,8,16,32,64,128,256,512}atα= 0.1
30

Table 6: Synthetic n-gram, V-sweep ( K= 4,n= 3,000,m= 3,000,nbits= 8). Monotone
increase, well below the V2/(Kn) statistical envelope ( d=V(V−1)≈V2for the bigram
model); the empirical curve traces a sub- V2shape and saturates as the smoothing-bias
floor takes over at largeV.
VMean KL 1σ
64 0.1256 0.0027
128 0.2969 0.0033
256 0.4253 0.0030
512 0.4772 0.0025
1024 0.4932 0.0012
Table 7: Letter vs. fullname scoring at Bi= 32, α= 0.10, 3 seeds. Coverage tracks
the 0 .9 target under both scoring methods on all three benchmarks; fullname scoring
substantively lifts acc@1 (2 .3×on DBpedia, 1 .9×on AG News, 1 .3×on MMLU) and
shrinks the mean prediction-set size on the easier benchmarks (DBpedia, AG News).
Dataset Scoring Coverage Set size (out of 4) acc@1
DBpedia letter 0.904±0.024 3.56±0.06 0.261±0.013
DBpedia fullname 0.909±0.021 2.23±0.04 0.604±0.003
AG News letter 0.917±0.025 3.81±0.04 0.238±0.007
AG News fullname 0.915±0.010 2.42±0.01 0.452±0.006
MMLU letter 0.916±0.028 3.74±0.04 0.203±0.007
MMLU fullname 0.888±0.012 3.46±0.06 0.268±0.010
and 3 seeds, and additionally run an α-sweep α∈ { 0.05,0.10,0.20}atBi= 32 to verify
coverage tracks the target across miscoverage levels. Each MCQ option is scored by all
four topic-nodes, so Ky=K= 4 uniformly and (B2)’s candidate-set inclusion clause
holds by construction.
B.5 E4 fine-grainedB isweep
To locate the bandwidth elbow more precisely than the power-of-2 grid in Figure 2, we
additionally sweep Bi∈ {4,6,8,10,12,14,16,20,24,28,32,40,48}at 5 seeds on DBpedia
and AG News under fullname scoring. The two benchmarks elbow at different bandwidth
thresholds. DBpedia saturates at set size 4 .0 through Bi= 14 and drops to 2 .37 at
Bi= 16. AG News elbows earlier: it saturates through Bi= 10 and drops to 2 .79 at
Bi= 12. AG News’s earlier elbow reflects its slightly easier inter-class score gap (the four
AG News topics are more lexically separable than the four DBpedia entity types in the
GPT-2-small score space), so a coarser quantization preserves the inter-class ordering at
lower bandwidth.
B.6 E4 MMLU coverage check
Figure 4 reports the MMLU 4-subject coverage and set-size panels under fullname scoring
across the canonical power-of-2 Bigrid. Coverage reaches the 0 .9 target at Bi≥16 (0 .92
atB i= 16, 0.89 atB i≥32); set size sits at 3.46 atB i= 32, larger than DBpedia (2.23)
31

10 20 30 40 50
per-query bandwidth Bi (bits)0.00.51.01.52.02.53.03.54.0Mean set size (out of 4)
DBpedia (fullname, fine Bi): Corollary 2 set-size efficiency
10 20 30 40 50
per-query bandwidth Bi (bits)0.00.51.01.52.02.53.03.54.0Mean set size (out of 4)
AG News (fullname, fine Bi): Corollary 2 set-size efficiencyFigure 3: Fine- Bigrid: mean set size on DBpedia (left) and AG News (right) under
fullname scoring, 5 seeds. DBpedia elbows between Bi= 14 and Bi= 16; AG News
elbows earlier between Bi= 10 and Bi= 12. Corollary 5.6’s set-size-vs- Bicurve resolved
near the transition.
or AG News (2 .42) at the same bandwidth, because GPT-2-small’s MMLU score gap
between the correct and incorrect choices is small.
2123252729
per-query bandwidth Bi (bits)0.00.20.40.60.81.0Coverage
MMLU 4-subject (fullname): coverage vs Bi
target 1 =0.90
empirical
2123252729
per-query bandwidth Bi (bits)01234Mean set size (out of 4)
MMLU 4-subject (fullname): set size vs Bi
Figure 4: MMLU 4-subject coverage and set-size panels under fullname scoring, 3 seeds,
α= 0.10. Conformal coverage remains at the 0 .9 target despite the underlying scorer
operating near chance level on academic-subject reasoning, with the prediction set widening
to 3.46 to absorb scorer uncertainty.
B.7 E3 multi-seed verification atn bits= 8
We rerun FPLD, FedDF [Lin et al., 2020] (the no-quantization probability-space distillation
analogue), FedAvg, and the pretrained checkpoint at 3 training seeds at the operating point
nbits= 8 (Table 8). The bandwidth-tax measurement (the FPLD-vs-FedDF perplexity gap)
is 15.60 ppl ( σFPLD = 1.19,σFedDF = 0.76), exceeding both methods’ per-seed standard
deviations by an order of magnitude. This is the real-LM analogue of FedDF being
FPLD’s no-bandwidth-budget upper bound: the gap reflects Theorem 4.1’s quantization
32

term plus the asymmetry between logit-space (FPLD) and probability-space (FedDF)
aggregation rules.
Table 8: Multi-seed at K= 4, Trounds = 5, Elocal= 1, m= 2,048,nbits= 8, clip ±40, 3
training seeds. Lower is better. The FPLD–FedDF gap is the empirical 8-bit bandwidth
tax on a real LM.
Method Perplexity (mean±1σ) ∆ vs. FedDF
Pretrained (no fine-tune) 59.15 +13.74
FedAvg 37.05±0.10−8.36
FedDF (no quantization) 45.41±0.76 0.00
FPLD (n bits= 8) 61.01±1.19+15.60
Centralized and single-shard fine-tuning rows are omitted because at this lr (5 ·10−5)
and epoch budget (5) vanilla AdamW is past the overfitting elbow and shifts by tens of ppl
across hardware (the federated methods reproduce within ∼6 ppl); a properly-regularized
centralized baseline is left to future work.
B.8 E3 large-Kand non-i.i.d. extensions
For completeness we report two extensions to the GPT-2-small bandwidth-tax experiment
that fall outside Theorem 4.1’s homogeneous-data regime ((A3)). A large- Ksweep at
K∈ { 4,8,16,32}with the total training-data budget held constant at 16 ,000 WikiText-2
sequences shrinks each shard to N/K blocks; FPLD perplexity grows with Kin this regime
as the per-shard statistical estimator becomes data-starved (the per-shard local-MLE bias
grows even as the total-pooled d/(Kn) stays fixed by total data). A non-i.i.d. sweep using
a topic-clustered partition (capacitated K-means on TF-IDF-weighted token-presence
features) tests robustness under data heterogeneity; FPLD perplexity increases over the
i.i.d. baseline by an amount comparable to the synthetic data-heterogeneity experiment’s
drift-induced KL increase, consistent with Corollary 4.3. Both regimes are outside the
assumptions of Theorem 4.1 and we accordingly do not promote their results to the main
text; the JSON results and configs are released with the paper.
B.9 E1.5 heterogeneous-data extension
This experiment tests Corollary 4.3’s data-heterogeneity slack on synthetic n-gram ground
truth: does empirical KL track the analytical drift term1
KP
iKL(P⋆∥Pi)? We construct
Kper-node distributions Pi=softmax (ℓ⋆+drift·ε i) with εiiid∼ N(0, I), perturbing a com-
mon base ℓ⋆∈RV×V(V= 256, k= 1); each node samples its training data from its own
Pirather than from P⋆. We sweep K∈ { 2,4,8}anddrift∈ { 0,0.1,0.2,0.3,0.5,0.75,1.0}
holding the homogeneous-experiment hyperparameters fixed ( n= 30 ,000, m= 3,000,
nbits= 8, clip 20,β= 0.5), and report empirical KL(P⋆∥ˆP) across 20 seeds per point.
The additive prediction is an upper bound at every ( K,drift ) point (Figure 5). The
drift = 0 row reduces by construction to Theorem 4.1’s homogeneous setting, with
empirical KL dropping from 0 .223 at K= 2 to 0 .182 at K= 4 and 0 .162 at K= 8,
consistent with the d/(Kn) statistical term decreasing as Kgrows. The empirical KL
grows sub-linearly in the drift term: at K= 4, KL grows from 0 .182 at drift 0 to 0 .247
at drift 1 .0 (drift term 0 .492), an empirical slope below the corollary’s slope-1 upper
33

0.0 0.1 0.2 0.3 0.4 0.5
drift term 1
K
iKL(PPi)
0.20.30.40.50.60.7empirical KL(PP)
Heterogeneous-data extension: empirical KL vs drift term
Cor 1 prediction:
KL_homo + drift_term (homo=0.189)
K=2
K=4
K=8Figure 5: Empirical KL(P⋆∥ˆP) versus the analytical drift term1
KP
iKL(P⋆∥Pi), 20
seeds per point. Solid curves: empirical means ±1σacross K∈ { 2,4,8}. Dashed line:
Corollary 4.3’s additive prediction KL homo+ drift term with KL homoaveraged acrossK.
bound. The sub-unit slope is consistent with the small-drift quadratic approximation in
Corollary 4.3’s proof, where the o(KL) correction absorbs the KL-functional curvature in
the high-drift tail.
The bound holds, the rate-regime K-axis collapses to Theorem 4.1’s homogeneous
floor at drift 0, and adding nodes counteracts drift through statistical pooling, so the
corollary’s structure is recovered empirically with room to spare.
B.10 E2 coverage-bound verification
This experiment tests whether Theorem 5.4’s distribution-free coverage bound holds
across all three calibration axes ( ncal,Bi,Bcal) on synthetic n-gram ground truth where
the predicted lower bound has a closed form. We reuse the n-gram ground truth from
Section 7.2 as the per-node scoring model and run FC-RAG calibration at target mis-
coverage α= 0.1, with each axis swept independently while the others are fixed at
(ncal, Bi, Bcal) = (3000 ,8,8). We expand the Bcalgrid to {1,2,4,6,8,10,12,14,16}to
expose both the noise-floor and saturation regimes. Every node scores every candidate
token so Ky=Kuniformly, satisfying (B2)’s candidate-set clause by construction; the
Laplace-smoothed n-gram MLE satisfies (B6)’s informativeness clause strictly. Mean
empirical coverage and the theoretical lower bound are reported across 20 ,000 test queries
and 20 seeds.
Empirical coverage tracks the 0 .9 target across all three sweeps. The ncalsweep
gives 0 .891±0.012,0.898±0.005,0.900±0.005,0.900±0.003,0.900±0.002 at ncal∈
{102,3·102,103,3·103,104}, approaching the target from below as ncalgrows, with the
per-seed 1 σtightening from 0 .012 to 0 .002 as expected from the 1 /√ncalstatistical-quantile
term. The predicted lower bound is loose at low bandwidth (e.g. Bi= 1: empirical 0 .900
vs predicted LB= 0.322) and tightens monotonically as bandwidth grows: at Bi= 8 the
LB is 0 .493 and at Bi= 32 it tightens to 0 .564. The Bcalsweep best illustrates the regime
transition: at Bcal= 1 the discretized quantile causes severe undercoverage (0 .674±0.006);
recovery to the target is monotone, reaching 0 .887 at Bcal= 4, 0 .902 at Bcal= 6, and
saturating at 0 .899 for Bcal≥8, while the predicted LB climbs from −0.973 at Bcal= 1
34

102103104
n_cal_per_node0.700.750.800.850.900.951.00coverage
E2: Theorem 2 coverage vs n_cal_per_node
predicted lower bound
1
empirical coverage
0 5 10 15 20 25 30
B_i0.700.750.800.850.900.951.00coverage
E2: Theorem 2 coverage vs B_i
predicted lower bound
1
empirical coverage
2 4 6 8 10 12 14 16
B_cal0.700.750.800.850.900.951.00coverage
E2: Theorem 2 coverage vs B_cal
predicted lower bound
1
empirical coverageFigure 6: Empirical conformal coverage (solid) versus Theorem 5.4’s predicted lower
bound (dashed) as a function of the per-node calibration size ncal, the per-score bandwidth
Bi, and the per-node calibration bandwidth Bcal. Empirical coverage hugs the 1 −α= 0.9
target across all three sweeps.
to 0.760 atB cal= 16.
Theorem 5.4’s coverage bound holds across all three axes, the predicted-LB curves
climb monotonically with bandwidth as the slack schedule predicts, and in the operating
regime Bcal≥6 coverage is indistinguishable from unquantized split-conformal, consistent
with Lemma 5.5.
B.11 E3 bandwidth-decay on GPT-2-small
This experiment lifts Theorem 4.1’sV
K2−2n bitsquantization-decay envelope onto a real
LM: does FPLD’s perplexity (ppl) on GPT-2-small + WikiText-2 follow the predicted
exponential decay as the per-coordinate quantization budget grows? GPT-2-small (124M
parameters, V≈ 50,257) is fine-tuned on a K= 4 random partition of WikiText-2 (4 ,000
length-128 blocks per shard, 16,000 total). Each round, every node fine-tunes locally for
one epoch (Adam, lr 5 ·10−5, batch 8), then evaluates next-token logits on a fixed m= 2,048
probe set drawn from the WikiText-2 test split (disjoint from train and validation). Each
node’s probe logits are scalar-quantized to nbitsbits per coordinate with clip ±40 and
averaged in logit space; the student is distilled against the softmax of the average for
3 epochs of KL minimization, repeated for Trounds = 5 rounds. Held-out perplexity is
reported on the WikiText-2 validation split. We sweep nbits∈ {2,4,6,8,10,12}at a
single training seed with an FPLD-no-quant reference (the nbits→ ∞ limit), and run a
multi-seed verification at the operating point nbits= 8 over 3 seeds against FedDF [Lin
et al., 2020] (the no-quantization probability-space distillation analogue), FedAvg, and
the pretrained checkpoint.
The bandwidth sweep (Figure 7) decreases monotonically from 93 .5 ppl at nbits= 2
through 54 .7 atnbits= 8 to the no-quant floor of 43 .1 asnbits→ ∞ . The 2−2n bitsfactor
in Theorem 4.1’s quantization envelope tightens by two orders of magnitude between
nbits= 4 (2−8≈4·10−3) and nbits= 8 (2−16≈1.5·10−5); the empirical curve replicates
this exponential-decay shape, with the universal constant c3=L2
ℓ/12≈133 at clip 40
making the bound qualitatively useful but quantitatively loose, and the empirical curve
sitting well below the bound at every nbitstested. The multi-seed verification at nbits= 8
gives an FPLD-vs-FedDF perplexity gap of 15 .60 ppl ( σFPLD = 1.19,σFedDF = 0.76),
exceeding both methods’ per-seed standard deviations by an order of magnitude (full
table in Appendix B.7). FedDF is the chosen comparator because the FPLD-FedDF gap
isolates the quantization channel from the protocol family; FedAvg requires |θ|-bit weight
35

2 4 6 8 10 12
nbits per coordinate5060708090Held-out WikiT ext-2 perplexity
quantization taxat nb=8:54.7343.08=11.65 ppl
FPLD bandwidth sweep on GPT-2-small (K=4, V50,257)
FPLD (K=4)
FPLD-no-quant (43.08)Figure 7: Held-out WikiText-2 perplexity of FPLD on GPT-2-small as a function of the
per-coordinate quantization budget nbits, atK= 4 (single training seed). Dashed line:
the FPLD-no-quant floor at 43 .08 ppl. FPLD perplexity decreases monotonically toward
this floor as nbitsgrows, consistent with Theorem 4.1’sV
K2−2n bitsquantization-term decay.
averaging and violates the paper’s bandwidth budget (Section 2), so it is reported in the
appendix table for parameter-space reference only. The single-seed bandwidth sweep and
the multi-seed verification were run on different hardware configurations; both confirm a
substantial bandwidth tax of order 10–15 ppl at the operating point.
GPT-2-small reproduces the qualitative nbitsaxis of Theorem 4.1 on a real LM: the
empirical curve traces the exponential-decay envelope, and the multi-seed FPLD-FedDF
gap quantifies the bandwidth tax with >10σstatistical separation. The K-axis is not
load-bearing here by design: the synthetic rate experiment (Section 7.2) covers it, the
heterogeneous-data regime is covered by Appendix B.9, and the real-LM contribution
here is the bandwidth axis.
B.12 E4α-sweep verification across miscoverage levels
Theα-sweep at Bi= 32 directly verifies Theorem 5.4’s Pr[Y∈ C α(X)]≥1−α− ···
inequality across miscoverage levels: empirical coverage tracks 1 −αin every cell of the
3×3 grid (Table 9), within ±0.015 on DBpedia and AG News and within ±0.022 on
MMLU.
Table 9: α-sweep at Bi= 32, 3 seeds. Each cell reports empirical coverage (mean ±1σ)
against the target 1 −α. The empirical-vs-target gap is ≤0.015 on DBpedia and AG
News and ≤0.022 on MMLU, verifying Theorem 5.4’s coverage bound across miscoverage
levels.
Datasetα= 0.05 (1−α=0.95)α= 0.10 (1−α=0.90)α= 0.20 (1−α=0.80)
DBpedia (fullname) 0.950±0.021 0.909±0.021 0.813±0.029
AG News (fullname) 0.952±0.007 0.915±0.010 0.814±0.019
MMLU (fullname) 0.937±0.012 0.888±0.012 0.778±0.023
36

B.13 E5 end-to-end propagation chain
This experiment tests Corollary 6.7’s Pinsker propagation slack ∆ trainend to end on a
real LM: when FPLD’s training-time KL varies, does FC-RAG coverage remain at the
1−αtarget as the corollary predicts? For each training bandwidth nbits∈ {2,4,6,8,12}
plus an FPLD-no-quant reference, we (i) train FPLD on WikiText-2 at this bandwidth,
reusing the configuration from Appendix B.11 ( K= 4,Trounds = 5,Elocal= 1,m= 2,048,
clip= 40); (ii) record the trained student’s held-out WikiText-2 validation perplexity as
a proxy for EXKL(P⋆∥ˆP) (log ppl tracks KL up to the data-marginal entropy, which is
constant across runs); and (iii) run the student behind FC-RAG on DBpedia and AG
News (fullname, K= 4,α= 0.1,Bi= 32, Bcal= 8), averaging coverage and mean set size
across 3 FC-RAG calibration seeds at fixed FPLD training seed. Corollary 6.7 predicts
∆train=fmax(EKL +√
2EKL ), so the empirical signature should be (a) coverage holding
at 1−αacross the entire training-bandwidth range whenever the parametric scorer’s fmax
stays bounded, and (b) the training-bandwidth axis manifesting in efficiency (set size,
acc@1) rather than in coverage.
40 60 80 100 120 140
FPLD held-out ppl (KLtrain proxy)0.50.60.70.80.91.0Coverage
DBpedia (fullname): coverage vs FPLD training KL
target 1 =0.90
FPLD-no-quant
empirical (FPLD-quantized)
40 60 80 100 120 140
FPLD held-out ppl (KLtrain proxy)1.92.02.12.22.32.4Mean set size (out of 4)
DBpedia (fullname): set-size growth with KL_train
FPLD-no-quant
empirical (FPLD-quantized)
40 60 80 100 120 140
FPLD held-out ppl (KLtrain proxy)0.50.60.70.80.91.0Coverage
AG News (fullname): coverage vs FPLD training KL
target 1 =0.90
FPLD-no-quant
empirical (FPLD-quantized)
40 60 80 100 120 140
FPLD held-out ppl (KLtrain proxy)2.22.32.42.52.62.72.8Mean set size (out of 4)
AG News (fullname): set-size growth with KL_train
FPLD-no-quant
empirical (FPLD-quantized)
Figure 8: End-to-end FC-RAG coverage (left columns) and mean prediction-set size (right
columns) for FPLD-trained students at varying training bandwidth nbits∈ {2,4,6,8,12},
with the FPLD-no-quant reference shown as a star. Top row: DBpedia fullname; bottom
row: AG News fullname. Coverage stays at the 1 −α= 0.9 target across the entire
training-bandwidth range on both benchmarks, validating Corollary 6.7’s propagation
chain end to end.
37

On DBpedia, coverage stays in [0 .900,0.925] across all five training bandwidths plus
the no-quant reference, while the underlying FPLD validation perplexity ranges from
139.0 at nbits= 2 (worst-trained student) to 43 .2 at no-quant (best-trained student).
Specifically, nbits= 2 gives empirical coverage 0 .900±0.021 with mean set size 2 .09 and
acc@1 = 0 .326; the FPLD-no-quant reference gives 0 .925±0.020 with set size 2 .10 and
acc@1 = 0 .662. AG News tells the same story: coverage in [0 .903,0.927], with nbits= 2 at
0.927±0.003 (set size 2 .71,acc@1 = 0 .563) and no-quant at 0 .903±0.013 (set size 2 .30,
acc@1 = 0 .613). The training-bandwidth axis manifests in acc@1 (DBpedia drops from
0.66 at no-quant to 0 .33 at nbits= 2) and set size (AG News grows from 2 .30 at no-quant
to 2.71 atn bits= 2), exactly as Corollary 5.6 predicts.
The full Theorem 4.1–Theorem 5.4–Corollary 6.7 chain holds end-to-end on a real
LM: training-time bandwidth determines scorer quality, which determines set size and
acc@1, but distribution-free coverage survives the chain. Corollary 6.7’s ∆ trainslack is not
bound-binding for parametric LM students whose log-density is locally smooth in their
parameters: the strengthened (B5)’s fmaxstays bounded and the Pinsker√
KLfactor is
loose, so the conformal procedure absorbs scorer degradation into the calibration quantile
rather than into under-coverage.
C Broader impact and reproducibility
Broader impact.Our work is motivated by settings (clinical research networks, en-
terprise knowledge bases, scientific consortia) in which data cannot be centralized for
legitimate reasons (regulation, consent, institutional policy). A federated LLM pipeline
that preserves data locality while still delivering statistically calibrated predictions is a
strictly more deployable tool for such users than a centralized model. That said, two
second-order concerns deserve explicit mention. First, the conformal coverage guarantee is
marginal: it holds on average over the query distribution, not conditionally per input, so
in safety-critical applications operators should pair FC-RAG with conditional diagnostics
(group-wise coverage, per-domain set size) rather than relying on the marginal bound alone.
Second, we do not claim differential privacy; an adversary with access to the logit stream
could in principle mount membership-inference-style attacks against node-local data,
and deploying FPLD in sensitive settings should compose it with a differentially private
mechanism at the logit stage. Neither concern invalidates the statistical contribution, but
both are operational caveats a practitioner must confront. Reproducibility details and
compute footprint are described below.
Reproducibility.All theorems are stated with explicit constants and proved from the
listed assumptions; the proofs of Theorems 4.1, 5.4, and Corollary 6.7 (Appendix A)
make no deferred claims. The synthetic experiments (E1, E1.5, E2) run on CPU in
minutes and fully reproduce the scaling predictions of Theorem 4.1, Corollary 4.3, and
Theorem 5.4 respectively on n-gram ground truth, with no GPU or external data required.
The real-LM experiments (E3, E4, E5) use publicly available models (GPT-2-small from
HuggingFace, MiniLM for retrieval) and publicly available benchmarks (WikiText-2 for
FPLD pretraining and bandwidth-sweep splits; DBpedia 4-class, AG News 4-class, and
MMLU 4-subject for the FC-RAG end-to-end demonstrations and the Pinsker propagation
chain). Source code, configs, cached experiment result JSONs, and non-interactive figure-
regeneration scripts accompany this paper; each experiment emits a single JSON summary
38

that the plot scripts read to produce the figures used in Section 7. Commit hashes for the
exact model and data snapshots used in each figure, and a public repository link, will be
added in a future revision.
Compute footprint.Total compute is ≈60 GPU-hours on an academic GPU cluster
(mostly RTX 6000; the original single-seed bandwidth-sweep was run on A100) plus <10
CPU-hours for the synthetic sweeps; per-run breakdowns are read from the released
JSONs.
39