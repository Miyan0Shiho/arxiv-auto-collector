# Anytime-Valid Federated Conformal RAG for LLM Swarms

**Authors**: Prasanjit Dubey, Xiaoming Huo

**Published**: 2026-05-27 22:06:57

**PDF URL**: [https://arxiv.org/pdf/2605.29139v1](https://arxiv.org/pdf/2605.29139v1)

## Abstract
Federated Conformal RAG (FC-RAG) provides distribution-free coverage for a bandwidth-limited swarm of weak language models, but only at a fixed horizon. We extend it to anytime-valid sequential coverage: validity at every stopping time, preserved under predictable adaptive control (recalibration, per-node bandwidth escalation, distilled-student refresh), at no extra cost in assumptions over fixed-horizon FC-RAG. Naive composition fails because FC-RAG's marginal coverage bound makes the betting e-process a non-supermartingale on adverse calibration draws, and Ville's inequality cannot be invoked. We give Anytime-FC-RAG, a sequential extension built on a summable per-step calibration-deviation budget that converts the marginal bound into a strict conditional bound on a calibration-good event, paired with a truncated betting e-process that is a nonnegative supermartingale on the entire probability space. From these two ingredients, we obtain four guarantees: time-uniform alarm validity $\mathbb{P}(\sup_t E_t \ge 1/δ_e) \le δ_e + δ_{\mathrm{cal}}$, a Hoeffding-stitched cumulative-miscoverage envelope at the same total budget, safety under any predictable controller (recalibration, bandwidth escalation, student refresh), and training-side error propagation across an unbounded sequence of Federated Probe-Logit Distillation (FPLD) refreshes via a summable training budget. As a practical consequence, an adaptive controller that escalates retrieval bandwidth only when the e-process crosses a warning threshold matches the alarm rate of a fixed-high-bandwidth schedule at substantially lower communication cost. Experiments on a GPT-2-small + MiniLM swarm across MMLU, DBpedia, and AG News verify the predicted alarm rate, detection delay, envelope coverage, and $14$-$57\%$ bandwidth savings; the alarm fires when and only when coverage genuinely breaks.

## Full Text


<!-- PDF content starts -->

Anytime-Valid Federated Conformal RAG for LLM
Swarms
Prasanjit Dubey Xiaoming Huo
H. Milton Stewart School of Industrial and Systems Engineering,
Georgia Institute of Technology, Atlanta, GA 30332, U.S.A.
Abstract
Federated Conformal RAG (FC-RAG) [Dubey and Huo, 2026] provides
distribution-free coverage for a bandwidth-limited swarm of weak language models,
but only at a fixed horizon. We extend it to anytime-valid sequential coverage:
validity at every stopping time, preserved under predictable adaptive control
(recalibration, per-node bandwidth escalation, distilled-student refresh), at no extra
cost in assumptions over fixed-horizon FC-RAG. Naive composition of fixed-horizon
FC-RAG with off-the-shelf sequential testing fails because FC-RAG’s marginal
coverage bound makes the natural betting e-process a non-supermartingale on
adverse calibration draws, and Ville’s inequality cannot be invoked. We give
Anytime-FC-RAG, a sequential extension built on asummable per-step calibration-
deviation budgetthat converts the marginal bound into a strict conditional bound on
a calibration-good event, paired with atruncated betting e-processthat is a nonnega-
tive supermartingale on the entire probability space. From these two ingredients, we
obtain four guarantees: time-uniform alarm validity P(suptEt≥1/δe)≤δe+δcal,
a Hoeffding-stitched cumulative-miscoverage envelope at the same total budget,
safety under any predictable controller (recalibration, bandwidth escalation, student
refresh), and training-side error propagation across an unbounded sequence of
Federated Probe-Logit Distillation (FPLD) refreshes via a summable training
budget. As a practical consequence, an adaptive controller that escalates retrieval
bandwidth only when the e-process crosses a warning threshold matches the alarm
rate of a fixed-high-bandwidth schedule at substantially lower communication cost.
Synthetic and end-to-end experiments on a GPT-2-small + MiniLM swarm across
MMLU, DBpedia, and AG News verify the predicted alarm rate, detection delay,
envelope coverage, and 14–57% bandwidth savings; the alarm fires when and only
when coverage genuinely breaks, not on every drift.
1 Introduction
Federated Conformal RAG (FC-RAG) [Dubey and Huo, 2026] provides a distribution-free
coverage guarantee for the answer set produced by a bandwidth-limited swarm of weak
language models, but only at a fixed horizon. Real deployments are sequential: operators
inspect coverage across the answered-query stream, recalibrate when the rolling buffer
freshens, and may escalate per-node retrieval bandwidth or refresh a distilled student
in response to observed drift. The fixed-horizon guarantee does not apply once any of
these actions is taken. This paper asks how to deliver time-uniform coverage and safe
1arXiv:2605.29139v1  [stat.ML]  27 May 2026

adaptive control on top of FC-RAG without strengthening the i.i.d.-deployment-data
assumption that fixed-horizon FC-RAG already relies on, and how to propagate the
underlying federated training rate across an unbounded sequence of student refreshes
while preserving the sequential guarantee.
As a running example throughout this paper, consider the K= 4 topic-specialized
GPT-2-small + MiniLM retrieval swarm of [Dubey and Huo, 2026,§7.5]: four nodes serving
MMLU subjects high school statistics ,high school physics ,high school biology ,
and high school world history , each retrieving from its own subject-specific corpus
and uploading bandwidth-limited score summaries to a hub that emits a conformal answer
set at level 1 −α= 0.9. A reasonable operational goal is to monitor coverage across
the answered-query stream and, if the corpus on high school biology drifts after an
update, fire an alarm and selectively escalate that node’s bandwidth without forfeiting
the statistical guarantee already accumulated on the other three subjects.
This operational goal is harder than a naive composition of fixed-horizon FC-RAG and
off-the-shelf sequential testing would suggest. The natural betting e-process built on FC-
RAG’s marginal coverage bound fails the supermartingale property on adverse calibration
draws, so Ville’s inequality cannot be applied off-the-shelf. Adaptive controller actions
further change the test centering mid-stream in a way that compounds the difficulty, and
the underlying student model may be refreshed multiple times during deployment, so
training-side error must propagate cleanly across an unbounded sequence of refreshes.
The construction we introduce closes all three gaps via a summable per-step calibration-
deviation budget paired with a truncated supermartingale, and the resulting guarantees
apply to any sequential conformal protocol with a predictable slack decomposition, beyond
the RAG instance considered here (Section 6).
Limitations of prior work.Three lines of work each cover a piece of this question but
leave a gap.
•Fixed-horizon FC-RAG.The base FC-RAG protocol [Dubey and Huo, 2026] controls
a one-shot marginal coverage guarantee; sequential monitoring with optional stopping
or adaptive intervention breaks the guarantee.
•Single-site conformal RAG.TRAQ [Li et al., 2024] and conformal-RAG-style
work [Chakraborty et al., 2026] apply split-conformal to RAG QA pipelines but assume
one model, one corpus, and one calibration set, with no federation, no bandwidth
charging, and no sequential validity.
•Conformal test martingales.Conformal test martingales [Vovk et al., 2021] give
a time-uniform changepoint-detection construction over exchangeable data, but in a
single-site setting with no slack decomposition (no ∆ FL, ∆RAG, ∆train) and no model
for federated calibration with bandwidth budgets.
Centralized sequential testing tools [Ramdas et al., 2023, Howard et al., 2020, 2021,
Waudby-Smith and Ramdas, 2024, Gauthier et al., 2025, Hultberg et al., 2026, Gibbs
and Candes, 2021, Angelopoulos et al., 2024b] and federated conformal methods [Lu
et al., 2023, Plassier et al., 2023, Wen et al., 2026, Xu et al., 2025] cover related ground
but not this composition; no prior result gives a time-uniform reliability guarantee for
a federated LLM swarm whose retrieval and calibration messages are simultaneously
communication-constrained. Table 1 summarizes the joint capability gap.
2

Table 1: Capabilities of related approaches. The combination of time-uniform validity, a
bandwidth-charged slack decomposition, federated calibration with summary compression,
training-side propagation, and validity under predictable controller actions is jointly absent
from prior work.
ApproachTime-
uniformBandwidth-
charged slackFederated
calibrationTraining
propagationPredictable
controller
FC-RAG
[Dubey and Huo, 2026]×✓ ✓ ✓×
Conformal test martingales
[Vovk et al., 2021]✓× × × ×
Online conformal
[Gibbs and Candes, 2021, Angelopoulos et al.,
2024b]✓∗× × ×—
Federated conformal
[Lu et al., 2023, Plassier et al., 2023, Wen et al.,
2026, Xu et al., 2025]×partial✓× ×
Anytime-FC-RAG (ours)✓ ✓ ✓ ✓ ✓
∗Long-run average coverage via adaptive step size; not strictly per-step time-uniform.
Constraints and goal.We adopt three operational constraints inherited from FC-
RAG [Dubey and Huo, 2026]: (i) no gradient or weight exchange; (ii) no data pooling;
(iii) per-uplink budgets Bi,t(per-query inference) and Bcal
t(per-refresh calibration) are
first-class. The goal is to characterize whether a sequential extension can admit a strict
conditional bound E[Mt| Ft−1]≤bt=α+ 1/(ncal,t+ 1) + ∆ FL,t+ ∆ RAG,t + ∆ train,t on
a calibration-good event Gt, with the same slack decomposition as fixed-horizon FC-
RAG. We further require that this bound be preserved under any predictable controller
(recalibration, bandwidth escalation, student refresh). Our aim is to characterize what is
provably achievable, not to demonstrate a deployment-ready system.
Contributions.
1.Anytime-FC-RAG protocol(Section 3): a sequential extension of FC-RAG
in which a swarm answers a stream of queries, updates compressed calibration
summaries on a rolling buffer, and maintains a betting e-process for alarm-triggered
intervention.
2.Cal-deviation budget and alarm validity(Lemmas 4.5, 4.6, Theorem 4.7):
a summable per-step budget {δcal
t}defines an Ft−1-measurable calibration-good
event Gton which the per-step miscoverage admits astrictconditional bound; the
truncated betting e-process eEt=Et1T
s≤tGsis then a nonnegative supermartingale
on the entire probability space, and Ville plus splitting on GtgivesP(suptEt≥
1/δe)≤δ e+δcal.
3.Cumulative-miscoverage envelope and safe adaptive control(Theo-
rems 4.9, 4.11): a time-uniform Hoeffding boundary ut(δ) controls the empirical
miscoverage rate against the predictable slack at probability ≥1−δ, and any
predictable controller preserves both this envelope and the alarm guarantee.
4.Training-to-deployment propagation(Theorem 4.12): an FTC chain inheriting
the (B5’) clause of base FC-RAG [Dubey and Huo, 2026, Cor. 3] gives ∆ train,t≤
3

fmax,t(¯Kt+p
2¯Kt) simultaneously over ton an event of probability ≥1−δtrain, for
a summable budgetP
rδr≤δtrain.
Scope.This work establishes a time-uniform reliability guarantee for federated LLM
swarms with bandwidth-constrained retrieval and calibration, paired with end-to-end
empirical validation on a K= 4 GPT-2-small + MiniLM swarm. Privacy accounting,
delayed-label handling, adversarial-node and architecture-heterogeneous regimes, and
deployment-scale benchmarking are natural extensions of the same machinery and are
deferred to follow-up work.
Paper roadmap.Section 2 specifies the sequential deployment model. Section 3 states
the Anytime-FC-RAG protocol. Section 4 proves the four theorems (alarm validity,
envelope, safe control, training propagation). Section 5 reports synthetic, real-world,
and comparative experiments validating the four theorems on a GPT-2-small + MiniLM
swarm across three benchmarks. Section 6 situates the contribution against prior work
and discusses limitations.
2 Problem setup
We study sequential discrete-answer prediction in a federated swarm of weak language
models constrained by a bandwidth budget. This section fixes the data, communication,
filtration, and estimand formalism that the planned theorems will refer to; the concrete
protocol sits in Section 3.
2.1 Data, swarm, and communication
LetXdenote the query or context space and let Ybe a finite answer space (multiple-
choice-style QA, label prediction, or a bounded candidate set extracted from a top- p
truncation of a language model). Let Vbe the token vocabulary of the underlying language
model. There are Knodes; node i∈ {1, . . . , K} holds a local retrieval corpus Ci,tand a
local retrieval mechanism, and raw node-local data and corpora never leave their node.
A global student model bP(0)is available at deployment start, typically obtained from a
federated training stage; we use theFederated Probe-Logit Distillation(FPLD) protocol
of [Dubey and Huo, 2026] as the running training stage, where each node fine-tunes locally
and exchanges B-bit quantized logits on a shared probe set rather than gradients or
weights. Theorem 1 of [Dubey and Huo, 2026] gives an explicit high-probability KL rate
for FPLD in ( K, n, m, B, V ), which we reuse as a black-box rate input for Theorem 4.12.
During deployment, the active student is denoted bPtand may be refreshed at selected
intervention times.
Per-query protocol.At each time t∈N , a query-answer pair ( Xt, Yt)∈ X × Y is
realized; the operator does not commit to a terminal time in advance, and may inspect,
recalibrate, or escalate after every query. We assumeimmediate label feedbackin the
main formulation: the true answer Ytbecomes available before the next query arrives, so
revealed labels drive the monitoring process. Arbitrarily delayed labels are deferred to
future work. At time t, node iretrieves a top- ki,tpassage set Zi,t=Ri,t(Xt, Ci,t;ki,t)⊆C i,t;
corpora may evolve slowly and are not shared across nodes. Bandwidth is the only resource
4

we charge, and we charge uplink only: at time t, node iuploads a Bi,t-bit summary of
its local scores; at calibration-refresh times t∈ T cal, node ialso uploads a compressed
calibration summary within a total budget Bcal
t; retraining inherits its cost from the
underlying training protocol (e.g., FPLD). The per-query inference plus calibration cost
is Γcomm
t =PK
i=1Bi,t+Bcal
t1{t∈ T cal}.
Given query Xt, node iforms a candidate list Ai,t(Xt)⊆ Y and local nonconformity
scores si,t(y) =−logbPt(y|X t, Zi,t) for y∈A i,t(Xt), clipped to [0 , Smax]. Node iuploads
a compressed messageU i,t=Q Bi,t 
{(y, s i,t(y)) :y∈A i,t(Xt)}
, the hub decodesU i,tinto
approximate scoreses i,t(y), and aggregates them. WithK t,y=|{i:y∈A i,t(Xt)}|,
s⋆
t(y) =1
Kt,yX
i:y∈A i,t(Xt)si,t(y), sswarm
t(y) =1
Kt,yX
i:y∈A i,t(Xt)esi,t(y),
where sswarm
t(y) = +∞ifKt,y= 0. The first is theoracle uncompressedswarm score; the
second is what the hub actually sees. Inheriting candidate-set inclusion (Assumption B2
of [Dubey and Huo, 2026]), every yin the test or calibration support lies inTK
i=1Ai,t(Xt),
i.e.,Kt,y=Kuniformly; the analysis lifts to general Kt,yat the cost of y-dependent
variance constants.
2.2 Calibration and filtration
We maintain a rolling labeled calibration buffer Dcal
t={(Xs, Ys) :s∈Ical
t}with ncal,t:=
|Ical
t|, where Ical
tis a predictable index set (e.g. a sliding window of the most recent answered
queries or a batched refresh buffer). At a refresh time t∈ T cal, node irecomputes its local
scores on Dcal
t, compresses the resulting score summary into its allotted share of Bcal
tbits,
and uploads it. Let q⋆
tdenote theoracleconformal threshold from the full uncompressed
calibration scores, and bqtthe threshold reconstructed from compressed node summaries;
the swarm outputs the implemented prediction set Ct(Xt) ={y∈ Y :sswarm
t(y)≤bq t}
with miscoverage indicatorM t=1{Y t/∈Ct(Xt)}.
LetFt=σ 
X1:t, Y1:t,bP0:t,{C i,1:t}i,{B i,1:t}i,{U i,1:t}i,bq1:t,A1:t
be the observable his-
tory, where Atis the intervention action at time t. A stopping time τis any Ft-adapted
random time. The core modeling restriction ispredictability: any bandwidth schedule,
recalibration schedule, or retraining trigger used at time tisFt−1-measurable. This is
what lets monitoring and intervention coexist without breaking validity.
2.3 Slack decomposition:∆ FL,t,∆ RAG,t,∆ train,t
Three slack terms enter the per-step coverage bound: federated calibration, retrieval
bandwidth, and training-side approximation. Each is Ft−1-measurable by predictability
of the associated schedule. We restate each at the level of detail required to follow the
analysis of Section 4; full constructions and constants are in [Dubey and Huo, 2026].
Retrieval-bandwidth distortion (∆ RAG,t).Inheriting Assumption (B3) of [Dubey
and Huo, 2026], each node’sQuantizeScoreprimitive is a subtractively dithered scalar
quantizer, so that the per-node quantized score decomposes additively as esi,t(Xt, y) =
si,t(Xt, y) +ξi,t(Xt, y), with {ξi,t}K
i=1conditionally independent across nodes given
(Ft−1, Xt), mean zero, and bounded conditional second moment E[ξ2
i,t| Ft−1, Xt]≤v(Bi,t)
where v(B) =O(2−2B/b s) for a protocol-specific scale bs. By the average-aggregation
5

rule, the implemented swarm score satisfies sswarm
t(Xt, y) =s⋆
t(Xt, y) +¯ξt(Xt, y) with
¯ξt:= (1 /K)P
iξi,t, and cross-node independence yields E[¯ξ2
t| F t−1, Xt]≤V K,t:=
(1/K2)P
iv(Bi,t). Combined with the fmax,t-Lipschitz score CDF (Assumption 4.3, (B5’))
and Cauchy–Schwarz onE| ¯ξt| ≤p
VK,t, the per-step retrieval-bandwidth slack is
∆RAG,t =f max,tvuut1
K2KX
i=1v(B i,t).
The 1 /K2averaging (rather than the worst-case 1 /K) is the variance gain from independent
cross-node dithering.
Federated-calibration distortion (∆ FL,t).The threshold bqtdeviates from the popu-
lation (1 −α)-quantile qpop
tvia a deterministic compression piece |bqt−q⋆
t| ≤ϕ (Bcal
t) =
O(2−Bcal
t/bq) (e.g., a quantized order-statistics summary or a GC-FCP / Fed-CCP core-
set [Wen et al., 2026, Xu et al., 2025]) and a statistical piece |q⋆
t−qpop
t|from the i.i.d.
buffer. To control the latter pointwise rather than only on average, we fix a summable
per-stepcalibration budget {δcal
t}t≥1withP
t≥1δcal
t≤δcal∈(0,1); the canonical choice is
δcal
t= 6δcal/(π2t2). Sub-Gaussian deviation of the empirical (1 −α)-quantile gives, with
probability at least 1−δcal
t,
bqt−qpop
t≤c qq
log(2/δcal
t)
ncal,t+ϕ(Bcal
t),(1)
for an absolute constantc q. Pushing (1) through the score CDF yields
∆FL,t=f max,t
cqq
log(2/δcal
t)
ncal,t+ϕ(Bcal
t)
.(2)
The event Gtthat (1)holds at time tisFt−1-measurable and has P(Gt)≥1−δcal
t; the
cumulativecal-goodevent Ω cal:=T
t≥1Gtsatisfies P(Ωcal)≥1−δcal. The conditional
theorems below take the form “on Gt” or “on Ω cal” and absorb δcalinto the final probability
budget. The conditional bound (2)is strictly weaker than the marginal-over-calibration
form of [Dubey and Huo, 2026] Theorem 2 by a factorp
log(2/δcal
t)/log(2/δ cal), which
grows from ≈1.07×att= 1 to ≈2.48×att= 104forδcal= 0.05, a slow O(√logt) price
for converting marginal validity into pointwise-conditional validity.
Training-side distortion (∆ train,t).Let εtrain,t denote the current training-side ap-
proximation level, an upper bound on EX[KL(P⋆(· |X)∥bPt(· |X))], and let ∆ t(x, y) :=
s⋆
t(x, y)−s⋆
ideal(x, y) =log(P⋆(y|x)/bPt(y|x)) be the score-level training residual. The
indicator-difference + FTC + Pinsker chain of [Dubey and Huo, 2026, Corollary 3], under
the (B5’) conditional-density clause inherited in Assumption 4.3, yields the two-term
bound
∆train,t =f max,t 
εtrain,t +p
2εtrain,t
.
In the small- εtrain,t regime (typical post-FPLD-training), the second summand dominates
and recovers the Pinsker√·shape. Theorem 4.12 bounds εtrain,t via the FPLD rate at the
most recent training event. Under adversarial models violating the conditional-density
clause, only the weaker rate ∆ train,t =O(fmax,tε1/4
train,t) is recoverable via Markov truncation;
we adopt the (B5’) regime as the operating assumption.
The four slack objects entering the per-step coverage bound are summarized below.
6

Symbol Source of slack Form
1/(n cal,t+ 1) Discrete split-conformal overshoot Vanishes asn cal,t→ ∞
∆FL,t Federated calibration: quantile + compressionf max,t
cqq
log(2/δcal
t)/ncal,t+ϕ(Bcal
t)
∆RAG,t Retrieval-bandwidth quantizationf max,tp
(1/K2)P
iv(Bi,t)
∆train,t Distilled-student approximation (FTC + B5’)f max,t 
εtrain,t +p2εtrain,t
2.4 The deployment null and scope
The base-paper FC-RAG inequality, lifted to a one-step conditional bound on the per-query
miscoverage, is
E[M t| Ft−1]1Gt≤b t1Gt, b t:=α+1
ncal,t+ 1+ ∆ FL,t+ ∆ RAG,t + ∆ train,t.(3)
We call (3)thedeployment null H0. Each ∆ •,tisFt−1-measurable by predictability,
hence so is bt. The bound holds pointwise on the calibration-good event Gt; offGtthe
bound is vacuous and the residual probability is absorbed into the final δcalbudget. The
theorems in Section 4 test whether the realized stream is consistent with H0on Ω cal. The
classical 1 /(ncal,t+ 1) split-conformal overshoot is included for compatibility with the
marginal-coverage bound of [Dubey and Huo, 2026, Theorem 2] and is dominated by ∆ FL,t
for typicaln cal,t.
Adversarial nodes, differential privacy, open-ended generation, arbitrarily delayed
labels, and architecture-heterogeneous clients are excluded from the first version of the
theory. Each is a natural extension, but none is necessary to formulate the sequential
object above.
3 The Anytime-FC-RAG protocol
The protocol is synchronous, single-aggregator, and charges uplink communication only.
Figure 1 shows the two coupled loops: a fast per-query inference loop in whichKnodes
serve a query under their bandwidth budgets, and a slow sequential-testing loop in which
observed miscoverage drives a betting e-process and a predictable controller feeds back to
budgets, threshold, and student.
Once the answer Ytis observed, the hub computes Mt=1{Yt/∈Ct(Xt)}and updates
the betting e-process
E0= 1, E t=E t−1 
1 +λ tZt
, Z t:=M t−bt,(4)
where λt∈[0,1/bt] is predictable. The bound λt≤1/btensures 1 + λtZt≥0 on every
realization, sinceZ t≥ −b t. The alarm time is
τalarm = inf{t≥1 :E t≥1/δ}.
Note the algorithmdoes not reset Etafter an alarm or refresh: Etcontinues to accumulate
evidence across interventions, which is what preserves the time-uniform Ville bound across
the entire post-deployment trajectory. (A reset variant with budgeted δacross epochs is
also valid; we discuss this in Section 4.5.) The ordering inside Algorithm 1 is essential: bqt
is set from Ft−1before Xtis served, Mtis recorded, Etis updated using Ft−1-measurable
7

Node 1
C1,t, R1,t,bPt
score→Q B1,t
Node 2···
NodeK
CK,t, RK,t,bPt
score→Q BK,tHub
decode{U i,t}; formsswarm
t
thresholdbq t
Ct(Xt) ={y:sswarm
t (y)≤bq t}queryX t
prediction setC t(Xt)
labelY tU1,t(B1,tbits)
UK,t(BK,tbits)Per-query inference loop
Betting e-process(Lem. 4.6, Thm. 4.7)
Et=Et−1(1 +λ tZt), Z t=M t−bt
bt=α+1
ncal,t+1+ ∆ FL,t+ ∆ RAG,t + ∆ train,t
alarm ifE t≥1/δPredictable controllerΠ (Thm. 4.11)
Ft−1-measurable action:
•recalibratebq t+1•escalateB i,t+1,Bcal
t+1
•refresh student bPt+1(FPLD)Mt=1{Y t/∈Ct(Xt)}
triggerSequential monitoring & adaptive controlpredictable
(Ft-meas.)
Figure 1: Anytime-FC-RAG architecture.Top (blue): per-query inference loop: K
nodes retrieve, score, and uplink Bi,t-bit summaries; the hub assembles the conformal set
Ct(Xt).Bottom (green): sequential monitoring loop: the betting e-process Ettests the
deployment null, and the predictable controller Π recalibrates, escalates bandwidth, or
refreshes the student. All controller actions are Ft-measurable (red dashed), preserving
validity (Theorems 4.11, 4.9).
btandλt, and onlythendoes the controller act, executing Algorithm 2 to recompute
the threshold and choose the next round’s budgets. Each downstream object is therefore
predictable with respect to the next round.
Predictable interventions.The controller may react to the monitoring state in three
ways: (i)recalibration(request fresh compressed calibration summaries and update bqt);
(ii)bandwidth escalation(increase selected Bi,tor the calibration-refresh budget Bcal
t); (iii)
retraining refresh(replace the current student with a refreshed model). All such actions
must be Ft−1-measurable. That predictability condition is what lets the same e-process
both guide interventions and remain valid after them (Theorem 4.11).
Relationship to FC-RAG.Fixed-horizon FC-RAG [Dubey and Huo, 2026] sits inside
our framework as the degenerate case in which Bi,t≡B i, the threshold bqtis computed
once from a one-shot calibration set, the student is frozen throughout deployment, no
intervention is triggered, and only the terminal-time miscoverage matters. The bounds
differ in conditioning structure: FC-RAG’s marginal-coverage Theorem 2 holds with high
probability over the calibration draw, whereas the present paper’s per-step bound holds
conditional on Ft−1on the cal-good event Gt(with the residual probability collected
intoδcalvia union bound). The new ingredients of Anytime-FC-RAG are exactly three:
arollingcalibration state, a(truncated) e-processfor time-uniform monitoring, and
predictablecontrol actions that change bandwidth or refresh the model only when justified
by accumulated evidence.
8

Algorithm 1Anytime-FC-RAG: query-time inference and monitoring
Require: Initial student bP(0); initial threshold bq0; target level α; alarm level δ; predictable
controller Π; per-node budgets{B i,1}K
i=1; betting cap ¯λ.
1:E 0←1
2:fort= 1,2, . . .do
3:Hub broadcasts queryX tto all nodes
4:foreach nodei= 1, . . . , Kin paralleldo
5:Z i,t←R i,t(Xt, Ci,t;ki,t)
6:A i,t(Xt)←TopCandidates( bPt(· |X t, Zi,t))
7:s i,t(y)← −log bPt(y|X t, Zi,t) fory∈A i,t(Xt)
8:U i,t←Q Bi,t({(y, s i,t(y)) :y∈A i,t(Xt)})
9:UplinkU i,tto hub
10:end for
11:Hub decodes{U i,t}K
i=1, formssswarm
t, and setsC t(Xt)← {y∈ Y:sswarm
t(y)≤bq t}
12:Observe labelY t, setM t←1{Y t/∈Ct(Xt)},Z t←M t−bt
13:Choose predictableλ t∈[0,min( ¯λ,1/b t)] fromF t−1
14:E t←E t−1(1 +λ tZt)
15:Update rolling bufferDcal
t←BufferUpdate(Dcal
t−1; (X t, Yt))
16:ifE t≥1/δorΠ(F t) requests refreshthen
17:Execute Algorithm 2▷predictable intervention;E tcontinues, not reset
18:end if
19:end for
Algorithm 2RefreshThresholdAndControl
Require:HistoryF t; rolling calibration bufferDcal
t; calibration budgetBcal
t
1:foreach nodei= 1, . . . , Kin paralleldo
2:Recompute local scores onDcal
tusing current bPtand local retrieval
3:Scal
i,t←CompressCalSummary i(Dcal
t;Bcal
t)
4:UplinkScal
i,tto hub
5:end for
6:Hub reconstructs updated thresholdbq t+1from{Scal
i,t}K
i=1
7:Predictably choose next budgets{B i,t+1}K
i=1and retrieval settings fromF t
8:ifΠ(F t) declares model-side driftthen
9:Refresh student bPt+1via FPLD or another approved training stage
10:else
11: bPt+1←bPt
12:end if
9

4 Anytime-valid guarantees
Off-the-shelf sequential testing supplies Ville’s inequality and the Hoeffding-stitched
envelope, and base FC-RAG [Dubey and Huo, 2026] supplies the slack decomposition
∆FL,t,∆RAG,t,∆train,t in marginal-over-calibration form. Neither composes into a sound
conditional supermartingale on its own, because base FC-RAG’s coverage bound is a
marginal claim (it fails conditionally on adverse calibration realizations) and the betting
e-process needs a strict conditional centering. Our load-bearing construction is thecal-
deviation budget {δcal
t}together with the resulting calibration-good event Gtand the
truncated supermartingale eEt=Et1T
s≤tGs: this is what converts the marginal slack form
into a strict conditional bound (Lemma 4.5) and turns the obstructed e-process into a
bona fide supermartingale on the entire probability space (Lemma 4.6). Once those two
pieces are in place, the rest is reuse: Ville’s inequality (Theorem 4.7), Hoeffding-stitching
(Theorem 4.9), predictability of the controller (Theorem 4.11), and the FTC + (B5’) chain
of [Dubey and Huo, 2026, Corollary 3] applied to the FPLD KL rate (Theorem 4.12).
The analysis is organized in six results. Lemma 4.5 lifts the base-paper Theorem 2
to a one-step conditional bound on Gt. Lemma 4.6 establishes that eEtis a nonnegative
supermartingale. Theorem 4.7 applies Ville’s inequality and a union bound on Gtto
recover a time-uniform alarm guarantee. Theorem 4.9 converts the same construction into
a time-uniform envelope on cumulative miscoverage. Theorem 4.11 shows that predictable
interventions preserve both guarantees. Theorem 4.12 bounds ∆ train,t using the FPLD
training rate at the most recent refresh.
4.1 Assumptions
Assumption 4.1(Predictable schedules).For every t, the calibration index set Ical
t, the
budgets {Bi,t}K
i=1andBcal
t, the active student bPt, the threshold bqt, the slack quantities bt,
and the betting fractionλ tare allF t−1-measurable.
Assumption 4.2(I.i.d. data and predictable buffer).The deployment sequence( Xt, Yt)t≥1
is i.i.d. from a fixed joint law PonX ×Y , and for every tthe calibration index set satisfies
Ical
t⊆ {1, . . . , t−1}and isF t−1-measurable.
This is the analogue of Assumption (B1) in [Dubey and Huo, 2026], adapted to the
sequential setting. It is the cleanest sufficient condition for the per-step rank exchange-
ability between ( Xt, Yt) and the buffer points to drive the conditional coverage analysis
below; the buffer, being Ft−1-measurable, is held fixed when conditioning, and randomness
reduces to the test point and the buffer realization (the latter shared between Ft−1and
the implicit Gtevent over the buffer draw). The assumption fails under genuine drift, and
rejection of the deployment null is exactly the alarm event the e-process detects.
Assumption 4.3(Bounded score and strengthened density regularity).The score s
is bounded in[0 , Smax](clipping). The cumulative distribution function Ftof the oracle
uncompressed score s⋆
t(Xt, Yt)admits a density ft≤f max,t on afixed deterministic
neighborhood[ qpop
t−rt, qpop
t+rt]for some rt>0; the analysis verifies a posteriori via
(1)thatbq tlies inside this neighborhood with probability at least1−δcal
t. Inheriting (B5’)
of [Dubey and Huo, 2026], the conditional density (on the same neighborhood) of the
relevant “before-perturbation” score given the perturbation is also bounded by fmax,t in two
specific cases:
10

•Retrieval-bandwidth dither(Step 3 of Lemma 4.5): the conditional density of
s⋆
t(Xt, Yt)given the dither average ¯ξt(Xt, Yt) := (1 /K)P
iξi,t(Xt, Yt), where ¯ξthas
E[¯ξt| Ft−1, Xt] = 0and bounded conditional second moment by Assumption (B3)
of [Dubey and Huo, 2026].
•Training residual(FTC chain of Section 2 and Theorem 4.12): the conditional density
ofs⋆
ideal(Xt, Yt) :=−logP⋆(Yt|Xt)given∆ t(Xt, Yt) :=log(P⋆(Yt|Xt)/bPt(Yt|Xt)) =
s⋆
t−s⋆
ideal.
Assumption 4.4(Slack admissibility).For every t,bt∈(0,1−η]for some fixed η >0,
and the betting cap ¯λsatisfies ¯λ≤1/b tuniformly int.
Assumption 4.4 ensures the e-process stays nonnegative; we typically take ¯λ≤1/(α+
∆max) for a slack upper bound ∆ maxchosen by the operator.
4.2 One-step deployment null
Lemma 4.5(One-step deployment null on the cal-good event).Under Assumptions 4.1–
4.3, on the calibration-good event Gtdefined in Section 2 (which satisfies P(Gt)≥1−δcal
t),
E[M t| Ft−1]1Gt≤b t1Gt.
Equivalently, on the eventG t,E[M t| Ft−1]≤b t.
Proof.The proof is provided in Appendix S1.1.
This is the conditional bound the betting e-process needs as a supermartingale on top
of FC-RAG. The pointwise form on Gt(not FC-RAG’s marginal Theorem 2 form) is the
centering Ville’s inequality requires; without it, no anytime-valid alarm is possible. The
price is ap
log(2/δcal
t)inflation of ∆ FL,t, anO(√logt) factor amounting to 1 .07×–2.72×
across t∈[1,105]. The cal-deviation budget is not optional: without it, the marginal
centering fails the supermartingale property on adverse calibration draws and Ville’s
inequality cannot be applied (Appendix S1).
4.3 Alarm validity
Define the per-step centered residual Zt:=Mt−bt∈[−bt,1−bt], anFt-measurable
bounded random variable, and the cumulative cal-good event Ω cal,t:=T
s≤tGs(Ft−1-
measurable since eachG sisF s−1-measurable).
Lemma 4.6(Truncated e-process is a supermartingale).Under Lemma 4.5 and Assump-
tions 4.1, 4.4, define the truncated e-process
eEt:=E t·1Ωcal,t,eE0:= 1,
where Etis the betting e-process (4)with predictable λt∈[0,1/bt]. Then( eEt)t≥0is a
nonnegative supermartingale withE eE0= 1:
E[eEt| Ft−1]≤eEt−1 for everyt≥1.
Proof.The proof is provided in Appendix S1.2.
11

Theorem 4.7(Time-uniform alarm validity).Under the assumptions of Lemma 4.6, for
everyδ e∈(0,1),
P
sup
t≥1Et≥1/δ e
≤δ e+δcal.
In particular, with the canonical split δe=δcal=δ/2, the alarm time τalarm =inf{t :Et≥
2/δ}satisfiesP(τ alarm<∞ |H 0)≤δ.
Proof.The proof is provided in Appendix S1.3.
Remark 4.8(Total probability budget).Theorem 4.7’s budget δe+δcalholds on the
training-good event of Theorem 4.12, which has probability ≥1−δtrainover the training
draws. Combining by union bound, the unconditional alarm guarantee is P(suptEt≥
1/δe)≤δe+δcal+δtrain, matching the user-facing total stated in Section 1. The canonical
equal splitδ e=δcal=δtrain=δ/3recovers a singleδ-level guarantee.
4.4 Cumulative-miscoverage envelope
The alarm at suptEt≥1/δcontrols the probability ofeverflagging a violation. To get a
numerically interpretable coverage statement at any predictable stopping time τwe use
the cumulative residual.
Theorem 4.9(Time-uniform Hoeffding envelope).Under Lemma 4.5 and Assumption 4.1,
define St:=Pt
s=1(Ms−bs) =Pt
s=1Zs. Then |Zs| ≤1deterministically, and for every
δe∈(0,1)there is an explicit boundaryu t(δe)with
P(∃t≥1 :S t> u t(δe))≤δ e+δcal.
A closed-form admissible choice (polynomial-stitching variant of [Howard et al., 2021]) is
ut(δe) =c Hr
1
2t
log(1/δ e) + log 
1 + log2t
with absolute constant cH≤1.7. In particular, with probability at least1 −δe−δcal, for
every stopping timeτadapted to(F t),
1
ττX
s=1Ms≤α+1
ττX
s=1
1
ncal,s+1+ ∆ FL,s+ ∆ RAG,s + ∆ train,s
+uτ(δe)
τ.
The envelope widthu τ(δe)/τ=O(p
log logτ/τ)vanishes withτ.
Proof.The proof is provided in Appendix S1.4.
Remark 4.10.The envelope and the betting e-process are complementary: Etaccumulates
evidence multiplicatively and is best when violations are persistent (high power for sustained
drift), while ut(δ)controls the cumulative deviation at any finite horizon and is best for
reporting an interpretable upper bound on the realized miscoverage rate. We track both
and report whichever is tighter at the requestedδ.
12

4.5 Safe adaptive control
Theorem 4.11(Safe adaptive control).LetΠbe any controller that maps Ft−1to a
(possibly randomized) action in At, where Atranges over: recalibration refreshes (changing
bqt), per-node bandwidth changes (changing Bi,torBcal
t), and student refreshes (changing
bPt). If every action is Ft−1-measurable, then under Assumptions 4.1–4.4 the supermartin-
gale property of the truncated e-process( eEt)and the time-uniform Hoeffding envelope of
Theorem 4.9 both continue to hold. Hence the alarm bound P(suptEt≥1/δe)≤δe+δcal
of Theorem 4.7 and the envelope bound of Theorem 4.9 apply to the controlled trajectory.
Proof.The proof is provided in Appendix S1.5.
Sticky vs. resetting alarms.Algorithm 1 does not reset Etafter an alarm (sticky
mode). A reset variant with per-epoch budgets is also valid; both variants preserve
Theorem 4.7 (Appendix S1).
4.6 Training propagation
Theorem 4.12(Training propagation under (B5’)).Fix a summable per-training-event
budget {δr}r≥1withP
r≥1δr≤δ train(canonical choice δr= 6δtrain/(π2r2)). Suppose the
student bPtused at deployment time tcomes from FPLD training event r(t)[Dubey and
Huo, 2026, Theorem 1] with parameters(K, n r(t), mr(t), Br(t), V), and let
Rr(t):=c1d
K n r(t)+c2ρVlog(V/δ r(t))
√mr(t)+c32−2B r(t)/V+εopt+εfit
denote the corresponding training-rate bound. Under the strengthened conditional-density
clause of Assumption 4.3, with probability at least1 −δtrainover the training draws,
simultaneously over allt≥1,
∆train,t≤f max,t 
Rr(t)+p
2Rr(t)
.
Proof.The proof is provided in Appendix S1.6.
In the small- Rregime thep2Rr(t)summand dominates, recovering the Pinsker shape.
Absent (B5’), only the weaker O(R1/4) rate is recoverable; the sequential-testing layer is
unaffected either way (Appendix S1).
Open direction.A communication-optimality oracle inequality of the form EPT
t=1ΓΠ
t≤
infΠ′∈CvalidEPΓΠ′
t+overhead (T), where Cvalidis the class of validity-preserving controllers,
would be substantially stronger than Theorem 4.11. We do not attempt it here and flag it
as future work.
5 Experiments
The experiments split into two qualitatively distinct roles.Synthetic experimentsprobe the
sequential-testing layer in isolation on Bernoulli streams whose conditional miscoverage is
set by hand, decoupled from the FC-RAG / FPLD pipeline.Real-world experimentsdeploy
the GPT-2-small + MiniLM swarm of [Dubey and Huo, 2026,§7.5] on three benchmarks
13

— MMLU [Hendrycks et al., 2021] (in-cal-set redistribution), DBpedia ontology [Lehmann
et al., 2015] (drift to a class the swarm has no node for), and AG News [Zhang et al.,
2015] (drift to a target recoverable from GPT-2 priors) — and compare against conformal
test martingales [Vovk et al., 2021], CUSUM, Shiryaev–Roberts, and online conformal
prediction [Angelopoulos et al., 2024b]. Real-world runs use 3 calibration splits ×5
deployment seeds (15 trajectories per benchmark); synthetic runs use 200–2000 seeds. All
runs use α= 0.10,δe= 0.05,δcal= 0.05; per-experiment details and expanded ablations
are in Appendix S2.
5.1 Sequential-testing layer validation
On synthetic Bernoulli streams, Type-I rate is 0 .0105 at the boundary of H0and 0 .0025
in the interior, both an order of magnitude below the δe+δcal= 0.10 budget (Table 2).
Detection power saturates at ≥99.6% from drift +0 .04 onward, with median delay
3047→687 steps as drift grows from +0 .04 to +0 .15 (Figure 2, Table 3), matching
the predicted Ω( log logT/drift2) rate. The Hoeffding-stitched envelope holds across all
4000 null trajectories with 0 .0000 breach rate, and a Monte-Carlo sanity check confirms
Lemma 4.5 pointwise (Appendix S2.1).
Table 2: Type-I rate is an order of magnitude below the δe+δcal= 0.10 budget on
synthetic Bernoulli streams.T= 5000, 2000 seeds per regime.
Regime alarm rate median suptEtp95suptEtp99suptEt
Boundary (E[Z t| Ft−1] = 0) 0.0105 1.131 6.381 21.230
Interior (E[Z t| Ft−1]<0) 0.0025 1.000 3.063 6.622
0.025 0.050 0.075 0.100 0.125 0.150
Drift size ppostb
0.00.20.40.60.81.0Fraction detected
0.025 0.050 0.075 0.100 0.125 0.150
Drift size ppostb
1032×1033×1034×103Median post-drift detection delay
Figure 2: Detection saturates at ≥99.6% by drift +0 .04, with delay decaying log-linearly
from∼5000 steps at +0.02 to∼700 at +0.15.
5.2 Slack decomposition and cost overhead
Empirical ∆ RAGtracks the variance form on a log-log axis with slope exactly −0.5000
forK∈ { 1, . . . , 128}(Figure 3, left). The training slack’s two-term form holds tightly:
empirical-to-theory ratio ≤0.91 (mean 0 .65) across 20 Bernoulli pairs with Beta(2,2)
residuals. The cost-overhead factor R(t) grows from 1 .07 at t= 1 to only 2 .72 at t= 105
(Figure 3, right), so the conditional construction is essentially free at any practical horizon.
14

Table 3: Delay decays inversely with drift size, matching the nonparametric
Ω(log logT/drift2) lower bound. Post-drift steps until Et≥1/δe;T= 8000, drift onset
t= 2000, 500 seeds.
Drift size fraction detected median delayp 95delay
+0.02 0.310 5076 5938
+0.04 0.996 3047 4464
+0.06 0.998 1904 2685
+0.08 0.998 1361 1864
+0.10 0.998 1057 1480
+0.15 0.998 687 931
100101102
K (number of nodes)103
RAG
Empirical (slope =0.500)
Predicted (variance form)
K1/2 (theory)
100101102103104105
t1.01.52.02.53.0R(t)=log(2/cal
t)/log(2/cal)
cal=0.01
cal=0.05
cal=0.1
Figure 3: The slack decomposition is numerically tight and the conditional construction
is essentially free.Left: ∆ RAGvs.Kmatches the theoretical −1/2 slope exactly.Right:
cost-overheadR(t)≤2×at all practical horizons, growing only logarithmically.
Additional necessity checks (cal-deviation budget, predictability of the controller layer)
and robustness studies (aGRAPA vs. constant- λbettors, hyperparameter sensitivity) are
reported in Appendix S2.4.
5.3 Adaptive bandwidth controller
On a synthetic stream with +0 .20 drift at t= 2500, all three bandwidth regimes (low-only,
high-only, adaptive) reach 100% alarm rate, but the adaptive regime pays only 1 .708 on
average — a 57% cost saving (Table 4). On the GPT-2-small swarm ( Bi= 8 low, Bi= 12
high), the same pattern holds: on DBpedia all three regimes alarm at 100% and adaptive
saves 14% (41 .5 vs. 48 .0); on MMLU and AG News (no genuine drift) the controller
correctly does not escalate (Figure 4). Predictable escalation is a real operational lever:
alarm validity is preserved, and high bandwidth is paid only where needed.
5.4 End-to-end real-world deployment and comparisons
The end-to-end result (Figure 5, Table 5) on the GPT-2-small + MiniLM swarm with
sudden drift at t= 500 over T= 2000 (15 trajectories): the e-process is silent on MMLU
in-cal redistribution (alarm 0 .00, post-miscov 0 .149) and AG News priors-recoverable drift
(0.00, 0.127), and fires on DBpedia genuine drift (0 .33, rising to 1 .00 in the head-to-head;
15

Table 4: Adaptive controller saves 57% cost at identical 100% alarm rate. Drift +0 .20
injected att= 2500;T= 5000, 200 seeds.
Regime mean cost alarm rate
Low only (∆ RAG= 0.04) 1.000 1.000
High only (∆ RAG= 0.005) 4.000 1.000
Adaptive (0.5/δ ewarning trigger) 1.708 1.000
Low only High only Adaptive0.00.20.40.60.81.0Alarm rate0.33 0.33 0.331.00 1.00 1.00
0.00 0.00 0.00(a) Alarm rate
MMLU
DBpedia
AG News
Low only High only Adaptive01020304050Mean cost per query3248
343248
41
2436
24(b) Communication cost
Figure 4: Adaptive controller (Theorem 4.11). All three bandwidth regimes match in
alarm rate; the adaptive regime saves 14% of communication cost on DBpedia (41 .5 vs.
48.0) without sacrificing detection.
post-miscov 0 .370> b≈ 0.21). Three robustness ablations (drift schedule, heterogeneous
bandwidth, FPLD multi-refresh) preserve this discriminative behavior (Appendix S2.3).
0 500 1000 1500 2000
t25
0255075100logEtMMLU (in-cal drift)
alarm=0.00, post miscov=0.15
mean log Et ± 1
alarm threshold log(1/e)
drift onset
0 500 1000 1500 2000
tDBpedia (out-of-cal alarm)
alarm=0.33, post miscov=0.37
0 500 1000 1500 2000
tAG News (priors absorb)
alarm=0.00, post miscov=0.13
Figure 5: The alarm fires when and only when coverage genuinely breaks. Trajectory of
logE ton the GPT-2-small + MiniLM swarm under sudden drift at t= 500: silent on
MMLU and AG News, fires on DBpedia.
Against prior monitoring methods, our e-process matches conformal test martin-
gales [Vovk et al., 2021] in null validity (0 .009 vs. 0 .000) and dominates on drift (1 .000 vs.
0.000 on large drift), because CTM does not exploit the slack-decomposed null bound.
Parametric CUSUM and Shiryaev–Roberts match or exceed at every drift by assuming
the alternative is known; our nonparametric e-process is slower only at borderline drift,
the expected price of distribution-free testing [Howard et al., 2021,§6]. Online confor-
mal [Angelopoulos et al., 2024b] adapts αtto maintain coverage rather than alarming, so
the two are complementary.
16

The real-world head-to-head makes this concrete (Figure 6): on DBpedia our alarm
fires in every trajectory while OC compresses αtfrom 0 .10 to 0 .031; on MMLU and AG
News both heads correctly stay quiet. Aggregated (Table 5), the alarm fires when and
only when coverage genuinely breaks.
MMLU DBpedia AG News0.00.20.40.60.81.0Alarm rate(a) Anytime-FC-RAG alarm rate
MMLU DBpedia AG News0.00.10.20.30.40.5Post-drift miscoveragetarget =0.10
(b) Coverage outcome
Ours
Online conformal
MMLU DBpedia AG News0.000.020.040.060.080.100.12Final t (OC)
0=0.10
(c) OC adaptive compression
Figure 6: Head-to-head with online conformal [Angelopoulos et al., 2024b]. On DBpedia
our alarm fires in every trajectory while OC compresses αtfrom 0 .10 to 0 .031 to maintain
coverage; on MMLU and AG News both methods correctly stay quiet. The two heads are
complementary: ours surfaces events, OC adjusts set sizes.
Table 5: The alarm fires when and only when coverage genuinely breaks. Each cell reports
alarm rate / mean post-drift miscoverage (or alarm rate / mean cost-per-query for the
adaptive controller). The end-to-end row uses 15 trajectories; the head-to-head uses a
paired configuration with higher power, hence DBpedia 1.00 vs. 0.33.
Experiment MMLU DBpedia AG News
End-to-end 0.00 / 0.149 0.33 / 0.370 0.00 / 0.127
Sudden-drift ablation 0.33 1.00 0.00
Adaptive controller 0.33 / 34.4 1.00 / 41.5 0.00 / 24.0
Online-conformal head-to-head 0.27 / 0.184 1.00 / 0.445 0.00 / 0.168
6 Discussion and outlook
Related work and what is new.Anytime-valid testing via e-processes goes back
to [Ville, 1939] and was modernized for nonparametric settings in [Howard et al., 2020,
2021]; the betting interpretation we use [Waudby-Smith and Ramdas, 2024, Ramdas
et al., 2023] and recent conformal extensions [Gauthier et al., 2025, Hultberg et al.,
2026, Angelopoulos et al., 2024a, Gibbs and Candes, 2021, Angelopoulos et al., 2024b]
cover the sequential-testing layer. The alarm half of our construction is a federated,
bandwidth-aware analogue of conformal test martingales [Vovk et al., 2021]; classical
CUSUM and Shiryaev–Roberts changepoint detection are recovered as the special case
bt=αof our construction. Single-site conformal-RAG [Li et al., 2024, Chakraborty
17

et al., 2026] and federated conformal prediction [Lu et al., 2023, Plassier et al., 2023, Wen
et al., 2026, Xu et al., 2025] are proper subsets of our setting in distinct dimensions: the
former assumes one model, one corpus, one calibration set; the latter takes the score as
given and is silent on per-node retrieval bandwidth. Closest in spirit to our deployment
null is the non-exchangeable coverage analysis of [Barber et al., 2023]. The construction
is not RAG-specific: any sequential conformal protocol with i.i.d. deployment data, an
Ft−1-measurable threshold satisfying a high-probability quantile bound, and a predictable
slack decomposition admits the same alarm and envelope guarantees at budgetδ e+δcal.
Summary.We extended FC-RAG from a fixed-horizon coverage guarantee to an anytime-
valid deployment-time reliability framework via the cal-deviation budget and the truncated
supermartingale; the four theorems cover alarm validity, cumulative-miscoverage envelope,
safe adaptive control, and training-to-deployment propagation, and the empirical Type-I
rate, detection power, envelope coverage, controller-cost saving, and discriminative real-LM
behavior all match the predicted regimes (Section 5).
Limitations and broader impact.The main formulation assumes immediate label
feedback, and switching to an empirical-Bernstein boundary [Howard et al., 2021] would
tighten the envelope by typically 1 .5×. Under adversarial models that violate the (B5’)
clause, only the weaker ∆ train,t =O(fmax,tR1/4
r(t)) rate is recoverable; the sequential-testing
layer is unaffected. The construction is not specific to RAG and can be attached to any
sequential conformal protocol with a predictable slack decomposition. Two operational
caveats: the conformal coverage guarantee holds in expectation across queries, not
conditionally per input; and the protocol does not provide differential privacy on its
own. Adversarial nodes, open-ended generation, architecture-heterogeneous clients, and a
communication-optimality oracle inequality are out of scope.
18

References
Anastasios Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster.
Conformal risk control. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan,
and Y. Sun, editors,International Conference on Learning Representations, volume
2024, pages 55198–55218, 2024a. URL https://proceedings.iclr.cc/paper_files/
paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf.
Anastasios Nikolas Angelopoulos, Rina Barber, and Stephen Bates. Online conformal
prediction with decaying step sizes. In Ruslan Salakhutdinov, Zico Kolter, Katherine
Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors,
Proceedings of the 41st International Conference on Machine Learning, volume 235 of
Proceedings of Machine Learning Research, pages 1616–1630. PMLR, 21–27 Jul 2024b.
URLhttps://proceedings.mlr.press/v235/angelopoulos24a.html.
Rina Foygel Barber, Emmanuel J. Cand` es, Aaditya Ramdas, and Ryan J. Tibshirani.
Conformal prediction beyond exchangeability.The Annals of Statistics, 51(2):816 – 845,
2023. doi: 10.1214/23-AOS2276. URLhttps://doi.org/10.1214/23-AOS2276.
Debashish Chakraborty, Eugene Yang, Daniel Khashabi, Dawn Lawrie, and Kevin
Duh. Principled context engineering for RAG: Statistical guarantees via confor-
mal prediction. InAdvances in Information Retrieval: 48th European Conference
on Information Retrieval, ECIR 2026, Delft, The Netherlands, March 29–April
2, 2026, Proceedings, Part II, pages 537–546, Berlin, Heidelberg, 2026. Springer-
Verlag. ISBN 978-3-032-21299-3. doi: 10.1007/978-3-032-21300-6 \45. URL https:
//doi.org/10.1007/978-3-032-21300-6_45.
Prasanjit Dubey and Xiaoming Huo. Federated language models under bandwidth budgets:
Distillation rates and conformal coverage, 2026. URL https://arxiv.org/abs/2605.
09986.
Etienne Gauthier, Francis Bach, and Michael I. Jordan. E-values expand the scope of
conformal prediction, 2025. URLhttps://arxiv.org/abs/2503.13050.
Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift.
In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan,
editors,Advances in Neural Information Processing Systems, volume 34, pages 1660–
1672. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_
files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding. InInternational
Conference on Learning Representations, 2021. URL https://openreview.net/forum?
id=d7KBjmI3GmQ.
Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform
Chernoff bounds via nonnegative supermartingales.Probability Surveys, 17:257 – 317,
2020. doi: 10.1214/18-PS321. URLhttps://doi.org/10.1214/18-PS321.
Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform,
nonparametric, nonasymptotic confidence sequences.The Annals of Statistics, 49(2):1055
– 1080, 2021. doi: 10.1214/20-AOS1991. URL https://doi.org/10.1214/20-AOS1991 .
19

Bror Hultberg, Dave Zachariah, and Antˆ onio H. Ribeiro. Anytime-valid conformal risk
control, 2026. URLhttps://arxiv.org/abs/2602.04364.
Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N.
Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, S¨ oren Auer, and
Christian Bizer. DBpedia – a large-scale, multilingual knowledge base extracted from
Wikipedia.Semantic Web, 6(2):167–195, 2015. doi: 10.3233/SW-140134.
Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. TRAQ: Trustworthy retrieval
augmented question answering via conformal prediction. In Kevin Duh, Helena Gomez,
and Steven Bethard, editors,Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 3799–3821, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.210. URL https:
//aclanthology.org/2024.naacl-long.210/.
Charles Lu, Yaodong Yu, Sai Praneeth Karimireddy, Michael Jordan, and Ramesh Raskar.
Federated conformal predictors for distributed uncertainty quantification. In Andreas
Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and
Jonathan Scarlett, editors,Proceedings of the 40th International Conference on Machine
Learning, volume 202 ofProceedings of Machine Learning Research, pages 22942–22964.
PMLR, 23–29 Jul 2023. URLhttps://proceedings.mlr.press/v202/lu23i.html.
Vincent Plassier, Mehdi Makni, Aleksandr Rubashevskii, Eric Moulines, and Maxim
Panov. Conformal prediction for federated uncertainty quantification under label shift.
In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan
Sabato, and Jonathan Scarlett, editors,Proceedings of the 40th International Conference
on Machine Learning, volume 202 ofProceedings of Machine Learning Research, pages
27907–27947. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/
plassier23a.html.
Aaditya Ramdas, Peter Gr¨ unwald, Vladimir Vovk, and Glenn Shafer. Game-Theoretic
Statistics and Safe Anytime-Valid Inference.Statistical Science, 38(4):576 – 601, 2023.
doi: 10.1214/23-STS894. URLhttps://doi.org/10.1214/23-STS894.
Jean Ville. ´Etude critique de la notion de collectif. Number 3 in Monographies des proba-
bilit´ es. Gauthier-Villars, Paris, 1939. URL http://eudml.org/doc/192893 . Fascicule
III.
Vladimir Vovk, Ivan Petej, Ilia Nouretdinov, Ernst Ahlberg, Lars Carlsson, and Alex
Gammerman. Retrain or not retrain: conformal test martingales for change-point
detection. In Lars Carlsson, Zhiyuan Luo, Giovanni Cherubin, and Khuong An Nguyen,
editors,Proceedings of the Tenth Symposium on Conformal and Probabilistic Prediction
and Applications, volume 152 ofProceedings of Machine Learning Research, pages 191–
210. PMLR, 08–10 Sep 2021. URL https://proceedings.mlr.press/v152/vovk21b.
html.
Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables
by betting.Journal of the Royal Statistical Society Series B: Statistical Methodology,
86(1):1–27, 02 2024. ISSN 1369-7412. doi: 10.1093/jrsssb/qkad009. URL https:
//doi.org/10.1093/jrsssb/qkad009.
20

Haifeng Wen, Osvaldo Simeone, and Hong Xing. Efficient federated conformal prediction
with group-conditional guarantees, 2026. URL https://arxiv.org/abs/2603.14198 .
Rui Xu, Xingyuan Chen, Wenxing Huang, Minxuan Huang, Yun Xie, Weiyan Chen, and
Sihong Xie. Federated conditional conformal prediction via generative models, 2025.
URLhttps://arxiv.org/abs/2510.13297.
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional net-
works for text classification. In Corinna Cortes, Neil D. Lawrence, Daniel D.
Lee, Masashi Sugiyama, and Roman Garnett, editors,Advances in Neural Infor-
mation Processing Systems, volume 28, pages 649–657. Curran Associates, Inc.,
2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/hash/
250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html.
21

Supplementary Material
This supplement contains complete proofs of all six results stated in the main paper
(Lemmas 4.5–4.6 and Theorems 4.7–4.12), with extended remarks on the cal-deviation
budget, sticky vs. resetting alarms, and the tightness of the conditional-density clause
placed after the corresponding proofs; and extended experimental results including the
Lemma 4.5 sanity check, the ∆ traintwo-term form verification, real-world ablations (A2–
A4), and necessity and robustness studies. We use the notation of the main paper
throughout.
S1 Proofs
S1.1 Proof of Lemma 4.5 (one-step deployment null)
The proof proceeds in three steps. We first bound the conditional miscoverage relative to
the population quantile (Step 1), then absorb the empirical-vs-population deviation on
the cal-good event Gt(Step 2), then absorb retrieval-bandwidth quantization (Step 3).
Throughout, the test pair ( Xt, Yt) is independent of Ft−1by Assumption 4.2, and any
quantity defined from past observations isF t−1-measurable.
Step 1 (population coverage).Recall the deployed-student score s⋆
tfrom Section 2, and
letqpop
tdenote the (1 −α)-quantile of s⋆
t(X, Y) under ( X, Y)∼P⋆(the score function is
Ft−1-measurable, so qpop
tis too). Since ( Xt, Yt)| Ft−1has the same conditional law as the
deployment-law marginal,
P
s⋆
t(Xt, Yt)≤qpop
tFt−1
≥1−α,
with equality under continuity. (The classical 1 /(ncal,t+ 1) split-conformal overshoot
is a strictly weaker bound on the same quantity; we keep it inside btfor compatibility
with [Dubey and Huo, 2026] but the high-probability conditional analysis below does not
need it.)
Step 2 (quantile reconstruction on Gt).On the event Gt, by the construction of Gtin
(1),bqt−qpop
t≤c qq
log(2/δcal
t)/ncal,t+ϕ(Bcal
t).
By Assumption 4.3 the score CDF is fmax,t-Lipschitz in a neighborhood of qpop
t[Dubey
and Huo, 2026, Lemma 2], so onG t,
P(s⋆
t≤bqt| Ft−1)−P(s⋆
t≤qpop
t| Ft−1)≤∆ FL,t.
Step 3 (score perturbation, variance-based).Under the dithered-quantization As-
sumption (B3) of [Dubey and Huo, 2026] the swarm score decomposes as sswarm
t(Xt, y) =
s⋆
t(Xt, y)+¯ξt(Xt, y) with the dither average ¯ξt= (1/K)P
iξi,tsatisfying E[¯ξt| Ft−1, Xt] = 0
and, by independence of the per-node noise,
E[¯ξ2
t| Ft−1, Xt]≤V K,t:=1
K2KX
i=1v(B i,t).
22

For any Ft−1-measurable threshold uin the score-density neighborhood of Assumption 4.3,
writeF⋆
|¯ξ(u) :=Fs⋆
t|¯ξt(u| F t−1, Xt). Then
P(sswarm
t≤u| F t−1, Xt)−P(s⋆
t≤u| F t−1, Xt) =E
F⋆
|¯ξ(u− ¯ξt)−F⋆
|¯ξ(u)Ft−1, Xt
.
The strengthened conditional-density clause of Assumption 4.3 (the conditional density of
s⋆
tgiven ¯ξtis bounded by fmax,t on the same neighborhood) makes u7→F⋆
|¯ξ(u) uniformly
fmax,t-Lipschitz, so by Cauchy–Schwarz,
P(sswarm
t≤u| F t−1, Xt)−P(s⋆
t≤u| F t−1, Xt)≤f max,tE[|¯ξt| | F t−1, Xt]≤f max,tp
VK,t.
Taking expectation overX t| Ft−1and recalling the definition of ∆ RAG,t,
P(sswarm
t(Xt, Yt)≤u| F t−1)−P(s⋆
t(Xt, Yt)≤u| F t−1)≤∆ RAG,t.
Combine.OnG t, chaining Steps 1–3,
P(sswarm
t(Xt, Yt)≤bq t| Ft−1)≥P(s⋆
t(Xt, Yt)≤bq t| Ft−1)−∆ RAG,t (Step 3)
≥P(s⋆
t(Xt, Yt)≤qpop
t| Ft−1)−∆ RAG,t−∆ FL,t (Step 2)
≥1−α−∆ RAG,t−∆ FL,t (Step 1)
≥1−α−∆ RAG,t−∆ FL,t−∆ train,t (∆train,t≥0).
Hence E[Mt| Ft−1]≤α+ ∆ FL,t+ ∆ RAG,t + ∆ train,t≤b tonGt. The ∆ train,t slack in
btabsorbs training-side conservatism: it is bounded explicitly via the FTC chain of
Section 2 (training-side distortion paragraph), inheriting the (B5’) conditional-density
clause of [Dubey and Huo, 2026, Corollary 3], and propagated across FPLD refresh events
in Theorem 4.12.
Why the cal-deviation budget is not optional.Suppose one tried to skip the
cal-deviation budget and instead use the marginal-over-calibration high-probability bound
of [Dubey and Huo, 2026, Theorem 2] with a single fixedδ, centering the e-process at
bmarg
t:=α+1
ncal,t+1+fmax,t
cqq
log(2/δ)/n cal,t+ϕ(Bcal
t)
+ ∆ RAG,t + ∆ train,t.
The marginal bound holds with probability ≥1−δat any single t, but for sequential
validity simultaneously over all ta union bound is needed and a fixed δdoes not deliver
one. On buffer realizations whose quantile deviation exceeds the centering at sufficiently
many t,E[Mt| Ft−1] exceeds bmarg
tpointwise and the supermartingale property fails;
Ville’s inequality cannot be applied. The summable budget {δcal
t}remedies this at the
cost of anO(√logt) inflation of ∆ FL,t.
S1.2 Proof of Lemma 4.6 (truncated supermartingale)
Nonnegativity. Zt≥ −b t, so 1 + λtZt≥1−λ tbt≥0 when λt≤1/bt; inductively Et≥0
and hence eEt=E t1Ωcal,t≥0.
Conditional mean.Since Et−1,λt, and1 Ωcal,t−1 areFt−1-measurable, and1 Gtis
Ft−1-measurable by construction (Section 2), the indicator1 Ωcal,t=1 Ωcal,t−11GtisFt−1-
measurable. Hence
E[eEt| Ft−1] =Eh
Et−1(1 +λ tZt)1Ωcal,t−11GtFt−1i
=E t−11Ωcal,t−11Gt 
1 +λ tE[Z t| Ft−1]
.
23

OnG t,E[Z t| Ft−1]≤0 by Lemma 4.5, so1 Gt 
1 +λ tE[Z t| Ft−1]
≤1 Gt≤1. Therefore
E[eEt| Ft−1]≤E t−11Ωcal,t−1 =eEt−1.
S1.3 Proof of Theorem 4.7 (alarm validity)
(eEt) is a nonnegative supermartingale with EeE0= 1 by Lemma 4.6, so Ville’s inequal-
ity [Ville, 1939, Howard et al., 2020] gives P(supteEt≥1/δe)≤δe. On Ω cal,eEt=Etfor
allt. Splitting on Ω cal,
P
sup
tEt≥1/δ e
≤P
sup
teEt≥1/δ e
+P(Ωc
cal)≤δ e+δcal.
S1.4 Proof of Theorem 4.9 (Hoeffding envelope)
Zs∈[−b s,1−b s] has range 1, so by Hoeffding’s lemma applied conditionally onF s−1,
E[exp(λZ s)| F s−1]≤exp
λE[Z s| Fs−1] +λ2
8
.
By Lemma 4.5, on Gswe have E[Zs| F s−1]≤0, so1 GsE[exp(λZs)| F s−1]≤
1Gsexp(λ2/8). For each fixedλ≥0 the truncated exponential process
fWλ
t:= exp(λS t−λ2t/8)1 Ωcal,t
is therefore a nonnegative supermartingale with EfWλ
0= 1 (the same indicator-truncation
argument as Lemma 4.6). Ville’s inequality gives P(∃t:fWλ
t≥1/δλ)≤δλ, equivalently
P(∃t:St≥λ−1log(1/δλ) +λt/8onΩcal,t)≤δλ. Stitching over a discrete grid λk= 2−k
(k= 0,1,2, . . .) with budgets δk=δecζ/(k+1)2(cζ= 6/π2) by the union-bound argument
of [Howard et al., 2021] produces the displayed boundary on the event Ω cal; the constant
cH≤1.7 is from their Eq. (14). Splitting on Ω calthen yields P(∃t:St> u t(δe))≤δe+δcal.
The stopping-time corollary follows by evaluating at t=τinside the joint high-probability
event.
S1.5 Proof of Theorem 4.11 (safe adaptive control)
The post-action quantities ( bqt, B•,t,bPt) at time tareFt−1-measurable by hypothesis. Hence
bt=α+ 1/(ncal,t+ 1) + ∆ FL,t+ ∆ RAG,t + ∆ train,t and the calibration-good event Gtboth
remain Ft−1-measurable, with the same per-step bound P(Gt)≥1−δcal
tapplying since
δcal
tis fixed by the predictable budget schedule. The conclusion of Lemma 4.5 applies as
written. The supermartingale calculation in Lemma 4.6 proceeds without modification,
as does the truncated Hoeffding bound underlying Theorem 4.9. Validity is therefore
preserved under the entire intervention path.
Sticky vs. resetting alarms.Algorithm 1 does not reset Etafter an alarm. The sticky
version preserves P(suptEt≥1/δe)≤δ e+δcalover the entire trajectory but does not
give post-intervention re-tests. A reset variant divides δeinto per-epoch budgets δe,kwithP
kδe,k≤δeand starts a fresh e-process at each reset; Theorem 4.7 applies to each epoch
independently and a union bound recovers the global guarantee. Both variants are safe;
we default to sticky.
24

S1.6 Proof of Theorem 4.12 (training propagation)
Theorem 1 of [Dubey and Huo, 2026] gives, for each training event r,¯K(r):=
EX∼P⋆
X[KL(P⋆(·|X)∥bP(r)(·|X))]≤ R rwith probability ≥1−δr. The chain of the
training-side distortion paragraph in Section 2 ((B5’) of Assumption 4.3 + indicator-
difference + FTC + the splitting EP⋆|∆t| ≤ ¯Kt+p
2¯Ktvialog(1 + t)≤t and
Pinsker; [Dubey and Huo, 2026, Corollary 3]) gives, on each training-good event with
¯Kt:=¯K(r(t)),
∆train,t≤f max,tEX,Y|∆t(X, Y)| ≤f max,t ¯Kt+p
2¯Kt
≤f max,t 
Rr(t)+p
2Rr(t)
.
Union bound over training events withP
rδr≤δtrainyields the simultaneous-in- tstatement
on an event of probability≥1−δ train.
Tightness and the conditional-density clause.The two-term form ∆ train,t =
fmax,t(Rr(t)+p2Rr(t)) follows [Dubey and Huo, 2026, Corollary 3] under the clause
(B5’) of Assumption 4.3. In the small- Rregime the second summand dominates, re-
covering the Pinsker-√·shape; the linear summand is the price paid for matching base
FC-RAG’s exact form. Absent the clause (B5’), the FTC step fails and only the weaker
O(fmax,tR1/4) rate is recoverable via Markov truncation. Both the alarm validity and the
cumulative envelope are insensitive to this choice; only the absolute scale ofb tshifts.
S2 Extended experimental results
S2.1 E10: conditional bound on the cal-good event
We sample 1000 buffer realizations of size ncal= 100, scores Uniform [0,1],α= 0.10,
δcal= 0.05,fmax= 1.0. For each buffer we compute the empirical (1 −α)-quantile bq,
the cal-good bound btatt∈ { 1,10,102,103,104}(with δcal
t= 6δcal/(π2t2)), and check
whether the conditional miscoverage rate P(s >bq|bq ) exceeds bt. The violation rate
is 0.0000 across all 5000 buffer/horizon pairs, confirming Lemma 4.5 pointwise. The
bound widens with tas expected ( b1= 0.280,b104= 0.471, via thep
log(2/δcal
t)inflation
in ∆ FL,t), while the realized conditional miscoverage stays at α= 0.10 regardless of t.
The horizon-dependent inflation is the price of the conditional (vs. marginal) form; E8
quantifies it as a 1.07×–2.72×factor over the cost-relevant range.
S2.2 E5:∆ traintwo-term form verification
For Bernoulli pairs ( p, q) on a 5 ×5 grid with FPLD-distillation residuals drawn from
Beta(2,2), the empirical ratio of |p−q| tofmax(R+√
2R) stays below 1 in 100% of 20
sampled pairs (max ratio 0 .91, mean ratio 0 .65). The two-term form from [Dubey and
Huo, 2026, Corollary 3] holds tightly under the clause (B5’): this is a numerical witness
for Theorem 4.12.
S2.3 Real-world ablations: A2, A3, A4 in detail
Drift types (A2).On DBpedia (the genuine-drift benchmark), alarm rates rank by
drift severity as expected: nodrift 0.00,sudden 1.00,gradual 0.67,periodic 0.33.
25

Sudden onset gives the strongest signal; gradual interpolation slows evidence accumulation;
periodic schedules return to cal subjects every period and dilute the signal. On MMLU
(in-cal) and AG News (priors-recoverable), all schedules including sudden stay near zero
alarm, mirroring A1’s discriminative behavior.
Heterogeneous bandwidth (A3).On MMLU across four bandwidth configurations,
alarm rate is monotone in average bandwidth: uniform high (Bi= 10, alarm 0 .27);
mixed oneweak (Bi= (3,10,10,10), alarm 0 .00); mixed twoweak (Bi= (3,3,10,10),
alarm 0 .00);uniform low(Bi= 4, alarm 0 .00). Higher bandwidth shrinks ∆ RAG, tightens
bt, and lifts detection power, exactly the dependence the slack decomposition predicts
(Theorem 4.7).
Multi-refresh (A4).On MMLU with two FPLD-style refresh events (perturbation
noise schedule 0 .5→0.25→0 with transitions at t∈ { 500,1500}),btshrinks at each
refresh as the training residual Rr(t)drops, and the e-process growth rate recalibrates
accordingly, the predicted behavior under Theorem 4.12. Final miscoverage is 0 .120±0.027
across 15 trajectories.
S2.4 Necessity and robustness of the construction
Two construction choices in§3 are not optional. The cal-deviation budget {δcal
t}is needed
because adversarial buffer realizations make a marginal-bound centering bmarg
tfail the
supermartingale property; on uniform-Bernoulli null streams both our construction and the
marginal variant alarm at 0 .0000 (the inflation only kicks in on the exponentially-decaying
tail event the budget is designed to control), so the budget functions as a theoretical safety
net rather than an empirical-power booster. Predictability is needed at the controller
layer: a non-predictable controller that switches bandwidth based on Mtrather than
Ft−1-measurable evidence inflates Type-I from 0 .0070 (predictable, within δe) to 0 .1010
(non-predictable, above δe+δcal= 0.10), a 14 .4×violation of Theorem 4.7 confirming the
predictability assumption of Theorem 4.11 is empirically load-bearing.
The bettor and hyperparameters are robust. Against three constant- λbaselines
(λ∈ { 0.32,1.61,¯λ= 3.23}, the slack-admissibility cap), the predictable-plug-in aGRAPA
bettor matches all three under the null and on large drift but dominates by 4 .9×on
small drift (0 .348 alarm rate vs. best constant- λ’s 0.071, Figure 7); aGRAPA is the
only choice that is uniformly competitive across drift sizes. One-at-a-time sweeps
over α∈ { 0.05,0.10,0.15,0.20},δe∈ {0.01,0.025,0.05,0.10}, and the λ-cap factor
∈ {0.25,0.5,0.75,1.0}(12 configurations in total) leave Type-I within the corresponding δe
in every configuration and detection power ≥0.81 throughout (Figure 8), so the canonical
choices used elsewhere in the paper are not load-bearing in their specific values.
26

Null (boundary) Null (interior) Small drift Large drift0.00.20.40.60.81.0Alarm rate
e=0.05
aGRAPA (predictable plug-in)
constant =0.323
constant =1.613
constant =3.226
Figure 7: Alarm rate of aGRAPA vs. three constant- λbaselines across four regimes (two
null, two drift). Constant- λmatches aGRAPA on large drift but undercovers small drift
by 4.9×; aGRAPA is the only bettor that is uniformly competitive.
0.05 0.10 0.15 0.20
0.00.20.40.60.81.0Rate
Type-I rate
Fraction detected
0.02 0.04 0.06 0.08 0.10
e
0.00.20.40.60.81.0
Type-I rate
Fraction detected
0.4 0.6 0.8 1.0
 cap factor
0.00.20.40.60.81.0
Type-I rate
Fraction detected
Figure 8: Type-I (red) and detection power (blue) under one-at-a-time sweeps over α,
δe, and the λ-cap factor. Type-I tracks δein every configuration; power stays ≥0.81
throughout.
27