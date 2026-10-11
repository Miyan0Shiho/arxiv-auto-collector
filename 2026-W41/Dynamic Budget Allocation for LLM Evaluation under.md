# Dynamic Budget Allocation for LLM Evaluation under Hard Resource Constraints

**Authors**: Shai Feldman, Yaniv Romano

**Published**: 2026-10-05 20:31:22

**PDF URL**: [https://arxiv.org/pdf/2610.07362v1](https://arxiv.org/pdf/2610.07362v1)

## Abstract
We evaluate large language models (LLMs) in multi-turn interactions through their time-to-event: the number of interaction steps required to produce an event of interest, such as a successful jailbreak or agentic task completion. Under limited compute, interactions may be terminated before the event occurs, so that event times are only partially observed (censored). Existing allocation methods for calibrating time-to-event bounds satisfy the budget only in expectation and can exceed the available budget on a particular evaluation run. Enforcing a hard constraint is particularly challenging as the cost of a trajectory is initially unknown. We introduce Hard-budget Allocation with Reflow for Predictive calibration (HARP), a budget allocation that satisfies hard resource constraints and adaptively reallocates unused budget. We show how to use HARP to construct lower predictive bounds (LPBs) on the time-to-event and to estimate evaluation metrics such as the jailbreak rate on a fixed benchmark. Although HARP induces dependence in acquisition decisions across different trajectories, we prove that HARP never exceeds the target budget, that its LPBs have finite-sample coverage guarantees, and that its metric estimates are unbiased. Experiments on agentic task success, LLM jailbreaks, toxic content generation, and RAG hallucinations show that HARP achieves coverage close to the nominal level with low variance, while never exceeding the given budget.

## Full Text


<!-- PDF content starts -->

Dynamic Budget Allocation for LLM Evaluation under Hard
Resource Constraints
Shai Feldman1and Yaniv Romano1,2
1Department of Computer Science, Technion IIT, Israel
2Department of Electrical and Computer Engineering, Technion IIT, Israel
October 7, 2026
Abstract
We evaluate large language models (LLMs) in multi-turn interactions through theirtime-to-event: the number of
interaction steps required to produce an event of interest, such as a successful jailbreak or agentic task completion.
Under limited compute, interactions may be terminated before the event occurs, so that event times are only
partially observed (censored). Existing allocation methods for calibrating time-to-event bounds satisfy the budget
only in expectation and can exceed the available budget on a particular evaluation run. Enforcing a hard constraint
is particularly challenging as the cost of a trajectory is initially unknown. We introduceHard-budgetAllocation
withReflow forPredictive calibration( HARP ), a budget allocation that satisfies hard resource constraints and
adaptively reallocates unused budget. We show how to use HARP to construct lower predictive bounds (LPBs) on
the time-to-event and to estimate evaluation metrics such as the jailbreak rate on a fixed benchmark. Although
HARP induces dependence in acquisition decisions across different trajectories, we prove that HARP never exceeds
the target budget, that its LPBs have finite-sample coverage guarantees, and that its metric estimates are unbiased.
Experiments on agentic task success, LLM jailbreaks, toxic content generation, and RAG hallucinations show that
HARPachieves coverage close to the nominal level with low variance, while never exceeding the given budget.
1 Introduction
Evaluating LLMs and agents through repeated interaction can require substantially more computation than assessing
a single response. For example, a model may initially refuse a harmful request but comply after several revisions,
or initially fail a coding task while succeeding with further assistance. Time to such an event describes behavior
that a one-turn success rate does not capture. Lower and upper predictive bounds (LPBs and UPBs) on this
time-to-event can support evaluating the safety and utility of agents and LLMs, but constructing reliable bounds
requires observations from trajectories of unknown length.
In this work, we consider ahardbudget constraint for running LLM interactions, as required in many practical
scenarios. A policy satisfying the budget in expectation may exceed it on specific runs. Reserving the maximal
trajectory length for each sampled prompt may satisfy the deterministic constraint, but it can underutilize the
budget when the event occurs early; conversely, acquiring until exhaustion and treating the observed sample as
representative may induce a distribution shift which invalidates the theoretical guarantees.
Designing this budget allocation under hard constraints raises three challenges. First, the budget cost of a trajectory
is stochastic and unknown before acquisition. Second, the budget allocation must incorporate a dependence of
the budget spent on different trajectories. Thus, the theoretical coverage guarantee must account for dependence
1
arXiv:2610.07362v1  [cs.LG]  5 Oct 2026

between trajectories induced by the allocation. Prior works [ 1–4] require that the censoring times are independent
across different trajectories. Third, samples in which an event occurs early save budget units, which we aim to
reflow to other samples. This reallocation must preserve the feasibility of the remaining trajectories, so that the
deterministic constraint holds throughout execution.
Davidov et al. [3]proposed a budget allocation and a calibration scheme for constructing calibrated LPBs with
theoretical coverage guarantees. DAPRO [4] extended this approach to a dynamic budget allocation for multi-turn
LLM evaluation that leverages the trajectory history to allocate the budget more efficiently. Specifically, it can
dynamically choose to stop acquiring a trajectory at any turn, and the remaining budget will be exploited by other
samples. While both approaches provide informative predictive bounds and metric estimates with theoretical
guarantees, they satisfy the budget constraint only in expectation rather than deterministically.
Beyond controlling the budget deterministically, we seek to derive low-variance estimators because evaluation is
typically performed on a single realized dataset. Specifically, lower variability in the coverage rate of predictive
bounds means that the coverage attained on a given dataset is more likely to be close to the nominal level 1−α . For
metric estimation, even if two estimators are unbiased, the lower-variance estimator is preferable as, on any single
dataset, it is expected to yield an estimate closer to the true marginal value. Thus, the estimator with the lower
variance provides more reliable evaluations.
The work of Yamin et al. [5]studies budget allocation under hard constraints. In particular, they use dependent
rounding [ 6–8] for budget-constrained experimental design. Nonetheless, these works do not show how to construct
calibrated predictive bounds with finite-sample distribution-free guarantees under hard constraints where the
acquisition cost of each sample is arbitrary, stochastic, and initially unknown.
1.1 Our contribution
In this work, we introduceHard-budgetAllocation withReflow forPredictive calibration( HARP ), a novel budget
allocation with deterministic guarantees to form a statistically valid bound for the time-to-event or a general
evaluation metric. Our method has both static [ 3] and dynamic [ 4] variants. We outline our main contributions as
follows:
•Coverage and unbiasedness guarantees under dependent acquisition:We show that HARP constructs
predictive bounds with finite-sample coverage guarantees and produces unbiased metric estimates even
though its acquisition decisions are dependent across samples. Existing guarantees [ 1,3,4] assume that the
acquisition is independent across samples to provide such guarantees. HARP violates this assumption by
design: decisions are coupled within groups. Our proof exploits the specific allocation structure ofHARP.
•Budget validity: We formally prove that HARP satisfies the deterministic budget constraint. Empirically, we
observe thatHARPhas a higher budget utilization compared to prior methods [4].
•Depth and execution variants:We develop a depth rule that, as observed empirically, leads to reduced
coverage variance of the constructed calibrated predictive bounds and was not considered in prior works.
•Empirical Evaluation:We demonstrate HARP across diverse tasks, including agentic task success, toxic
content, LLM jailbreaks, and RAG-based hallucinations. Our experiments show that our method satisfies
the deterministic budget constraint and achieves low variance despite our method having a more restrictive
deterministic resource constraint.
The software package implementing our method is available athttps://github.com/Shai128/harp
2

2 Background
Multi-turn evaluation.Let Xdenote an initial prompt and H(t) its trajectory history after tinteractions. We
suppose we are givenNinitial promptsD cal={X i}N
i=1as the calibration data.
We assume a model Ainteracts sequentially with a target model Gfrom an initial prompt Xi∈ D cal. The model A
aims to elicit a key event. The interaction begins with Ai(1) =X i. At each step t, the target model produces a
response Ri(t) =G(A i(t)), which is then evaluated by an audit function Jto obtain an indicator for the event of
interest Yi(t) =J(A i(t), R i(t))∈ {0,1}, where Yi(t) = 1 indicates that the event has occurred. If Yi(t) = 1 , the
interaction terminates. Otherwise, Auses the interaction history Hi(t), which contains previous prompts, responses,
and audit annotations, to generate the next prompt,
Ai(t+ 1) =A(H i(t)).
The process continues until the event occurs or the interaction is otherwise terminated.
The corresponding time-to-event is defined as Ti= min{t≥1 :Y i(t) = 1} . To make this procedure tractable, we
set the maximal sequence length to M. For sample i, the censoring time Ci≤M determines the maximal observed
length min(T i, Ci). If the event has not occurred by turn Ci, the trajectory is right-censored and its event time is
unobserved.
We use a pre-trained survival model that generates predictions bfτ(X) for the τ-th quantile of the time-to-event. Any
model that estimates conditional quantiles can be used, e.g., quantile regression [ 9–11]. The quantile predictions are
nondecreasing inτ, as quantiles of a single predicted distribution, and the candidate setTcontains a levelτ 0with
bfτ0≡0. We trim the predictions at M, so that bfτ(X)≤M , to correspond to the maximal length we allow. We
define
Ai(τ) =1{T i<bfτ(Xi)}, µ(τ) =P{T < bfτ(X)}.(1)
Given α, δ∈(0,1) , our goal is to construct a predictive bound bfbτwith miscoverage at most αup to a deviation ∆
which arises from the uncertainty due to the finite calibration sample. Specifically, we seek the PAC guarantee
P{µ(bτ)> α+ ∆} ≤δ.(2)
The outer probability is taken over calibration trajectories and acquisition randomization, while the inner one
concerns an independent test point sampled from the same population. Thus, this guarantee differs from marginal
coverage averaged over the multiple runs of the calibration procedure, common in classic conformal prediction [ 12].
Throughout this work, the randomizations are taken conditionally on the pre-trained predictive model, when used,
and an independent reference dataset.
Random acquisition and calibration.Let Ziindicate that sample iisresolved, i.e., that its observed prefix
determines Ai(τ)for every candidate τ. Letπibe the probability that Zi= 1, conditional on the complete trajectory
of sample iand on the history of the previously acquired groups (Appendix B.3). The Horvitz–Thompson estimator
[13] then gives
bµ(τ) =1
NNX
i=1ZiAi(τ)
πi,bτ= max
τ∈ T: max
u∈T:u≤τbµ(u)≤α
.(3)
The denominator considers every sample in the calibration set, including prompts with unobserved event times. In
practice, we only need access to the probabilityπ ifor samples withZ iAi(τ) = 1.
From static allocation to DAPRO .Davidov et al. [3]propose optimizing the sampling probabilities to minimize
the variance of the empirical coverage rate. DAPRO [4] extends this approach by updating continuation decisions
dynamically from the trajectory history. It splits the calibration data into two: the first is used to learn an optimized
3

allocation policy deployed on the second split. It further includes an internal budget calibration step based on
conformal risk control (CRC) [14] to provide an expected budget guarantee.
For UPBs, the coverage error of the bound 1{T > bfτ(X)} decreases with τ, and an infinite bound has zero error.
Population metrics instead target an expectation, without selecting a quantile level. These changes require different
acquisition depths and estimators. We therefore develop them separately in Appendices D and E.
3 Proposed Method and Theory
We split the calibration data into groups and allocate the budget iteratively over groups, based on the remaining
budget each iteration. For each group, we set randomized censoring times so that the worst-case cost does not
exceed the remaining budget, acquire the interactions based on them, and update the remaining budget according to
the turns actually taken.
3.1 Reservations, commitments, and reflow
We rely on a reference set Rof size |R|=R to compute estimated predictions, without spending budget on it. We
exclude the reference samples from the calibration estimator. In the experiments, the total cap is B=N 0¯B, where
N0=N+R and¯Bis the budget per parent prompt. The reference therefore saves interactions at the cost of a
smaller estimation sample.
We randomly divide the calibration samples into K=⌈N/m⌉ groups Ik, each of size mk≤m . We assign sample
ia maximal execution horizon di∈ {0, . . . , M} , determined by the reference design and the sample’s own initial
features. LetL k−1be the remaining budget before groupk, initialized atL 0=B. We chooseη >0and require
B≥K(1 +η)M.(4)
The budget allowance and the pacing budget of the current group are
κk=Lk−1−(K−k)(1 +η)M, β k=Lk−1
K−k+ 1.(5)
The allowance reserves (1 +η)M turns for each later group, while the pacing budget distributes the remaining
budget over the remaining groups. Consequently, an inaccurate estimate of the consumed budget cost might not
attainβ k, but it will not violate the allowanceκ k.
We denote byˆr j(τ)the estimated probability for miscoverage event1{T j<bfτ(Xj)}. Next, we define
τ⋆:= min arg min
τ∈T1
RX
j∈Rˆrj(τ)−α.(6)
The average predicted event probability is formulated as ¯r:=R−1P
j∈Rˆrj(τ⋆). We denote the estimated expected
cost bybc i=bE[min(T i, di)|X i], and the average estimated cost function over the reference data is thus
cR(λ) :=R−1X
j∈Rπj(λ)c j πj(λ) = min
1,maxq
νj/(λc j), pmin
.(7)
We write νi= (1−ω) +ωbr i(τ⋆)/¯r, with ω= 1/2 . We choose the multiplier λkto satisfy cR(λk)≈β k/mkand,
ford i>0, set
pprop
i= cliprνi
λkbci, pmin,1
, pc
i= min
pprop
i,1,κk−M
mkdi
.(8)
4

The specific formulation of pprop
i is chosen as a heuristic, based on the closed-form minimizer derived by Davidov
et al. [3]. Our guarantees hold for any definition of pprop
i that depends only on the sample’s own features and
previous groups.
We next apply weighted pivotal rounding [ 15] to sample a commitment indicator ζiwith probability pc
i. The
commitment indicator ζidetermines whether we commit to allocating budget to sample ito observe its event time
Ti; ifζi= 0, we set Ci= 0. This pivotal rounding algorithm begins by initializing ζi=pc
i, and runs iteratively,
where at each iteration, it randomly rounds at least one index jintoζj∈ {0,1} , while preserving the identityP
idiζithroughout the process. The rounding procedure terminates once at most one coordinate remains unrounded.
This last coordinate is then rounded with a Bernoulli draw that changes the weighted sum by at most max idi≤M .
Combining this property with the cap in (8) gives
E[ζi|group data, past] =pc
i,X
i∈Ikdiζi≤X
i∈Ikdipc
i+M≤κ k.(9)
In Appendix B.1 we outline the full algorithm we use and prove the properties of this algorithm. After executing the
group, we subtract its actual paid cost Skfrom the ledger, Lk=Lk−1−Sk. Thus, turns saved by early events or
randomized stopping are available to the next group.
3.2 Completion, continuation, and depth
HARP -Commit( HARP -C) runs each sample with ζi= 1until resolving it, i.e., reaching its event time or horizon di,
costing min(T i, di)units for the i-th sample, and the resolution probability is pc
i. This version adapts the budget
between groupsthrough the ledger. HARP -Dynamic( HARP -D) additionally adapts the budget dynamicallywithin a
trajectory. Before spending budget for turn t, it uses only Hi(t−1) to compute a score νi(t), which is a smoothed
miscoverage probability at step t, and estimates the remaining cost bei(t). We denote by ai(t)the probability of
reaching timet, initialized atpc
i. We set
ui(t) = min(
1,s
νi(t)
λkbei(t))
, P i(t) = clipui(t)
ai(t−1), pmin,1
, a i(t) =a i(t−1)P i(t).(10)
So, at each stept, we continue to the next step with probabilityP i(t), and this random draw is independent across
different samples and of the randomization of ζi. Importantly, we do not continue running a trajectory that was
stopped by a continuation coin. For bi= min(T i, di), sample iis resolved, i.e., Zi=1{C i≥bi}, when acquisition
reaches its event time Tior its horizon di. Conditional on its trajectory and the previous groups, this occurs with
probability
πi=(
pc
i,modeC,
pc
iQbi
t=1Pi(t),modeD(11)
TheseZ iandπ iare used in the estimator (3).
Depth.The vanilla version of HARP sets the acquisition horizon as di=bfτprior(Xi), where τprior is a prior
quantile level chosen to reflect the belief that the calibrated level satisfies ( bτ≤τ prior). Prior work provides examples
of such choices: for a target coverage level of1−α= 90%, [3, 4] consider(τ prior∈ {0.3,0.56}).
Thedepthversion of HARP chooses a quantile level ℓ≤τ prior based on the initial reference predictions and restricts
the candidate set to τ≤ℓ . It then sets the resolution horizon to di= (bfℓ(Xi)−1) +, which is sufficient to determine
all retained events Ti<bfτ(Xi). By avoiding acquisition at later interaction steps, we save budget at the cost of
restricting the largest LPB that can be selected.
We choose the depth ℓusing only the initial predictions for the independent reference prompts R. For each candidate
τ, these predictions estimate the average resolution cost, LPB miscoverage, and LPB size. We consider five risk
5

Algorithm 1CoreHARPacquisition and LPB calibration
Require:Pre-trained model, independent reference, calibration samples, parametersB, M, m, η, p min
1:Fix reference design and retained family; setL 0=B; form random groups.
2:fork= 1, . . . , Kdo
3:Compute reserve, allowance, and pacing target by (5).
4:Chooseλ kfrom the reference curve; cap commitments by (8).
5:Draw weighted pivotal commitments with horizonsd ias costs (Algorithm 2).
6:Execute C or D; record paid turns, resolution indicators, and propensities (11).
7:UpdateL k=Lk−1−Skusing actual group spend.
8:end for
9:Compute (3) over allNworking samples; return bfbτ.
scenarios, multiplying the predicted miscoverage curve by γ∈ {.6, .8,1,1.25,1.6} while keeping predicted costs
and sizes fixed. For each τand scenario γ, we estimate the largest mean LPB size among its retained candidates
whose modeled risk plus a planning uncertainty penalty is at most αdesign . The regret is defined as the signed
difference of the LPB size compared to the best observed in the same configuration of risk γ. We choose the level
ℓwith the smallest worst-case regret, then increase it if a deeper endpoint still allows the planner’s hypothetical
acquisition fraction to remain one. Appendix C.2 provides the full derivations.
Truncating the candidate family would change the UPB and metric estimation tasks. For these tasks, we instead
independently assign each sample either a shallow or a full horizon, choosing the full-horizon branch with probability
.25and correcting for this draw in the weights; see Appendices D–E for more details.
3.3 Theoretical coverage and budget guarantees
We begin with the budget guarantee, which holds for every realization of the acquisition procedure.
Theorem 1(Deterministic budget control).Under(4),HARPsatisfiesPK
k=1Sk≤Balmost surely.
Crucially, this guarantee holds even when the predictions are inaccurate or when the data is not i.i.d. The full proof
is provided in Appendix B.2.
For the proof, we define the filtration Fk−1to contain the fixed design objects, all complete trajectories in previous
groups, and all random coins used in those groups. For a fixed candidate, define its predictable second-moment
average
v(τ) =1
NKX
k=1X
i∈IkEAi(τ)
πiFk−1
.(12)
The average is random but predictable. We assume an upper bound v⋆that is deterministic conditional on the fixed
reference set and holds almost surely for every retained candidate. This is a population envelope, rather than an
estimated sample variance.
Theorem 2(LPB PAC bound).Assuming the calibration and test data are i.i.d., suppose v(τ)≤v ⋆almost surely
for every candidateτ. SetL= log(1/δ). For any∆≥0satisfying
∆≥L
3N+r
L2
9N2+2(v⋆−α2)+L
N,(13)
the LPB chosen by(3)satisfies(2).
With probability at least 1−δ , the selected LPB has population miscoverage at most α+ ∆ , with dependent
acquisition entering the correction only through v⋆. See Appendix B.4 for the proof. Notice that the coverage
6

guarantee holds despite the dependence between the censoring times Ci.HARP inherently violates independence
across samples through both the pivotal selection and the ledger.
UPBs require a separate argument.The UPB estimator updates its predictions at every reached prefix:
Vi(τ) =m i0(τ) +X
tRit
ρit[mit(τ)−m i,t−1(τ)].(14)
Here mit∈[0,1] predicts the finite-bound loss. We replace it with the true loss upon resolution and keep it fixed
thereafter. The indicator Ritdenotes whether turn tis reached, and ρitaccounts for commitment, continuation, and
the insured-depth correction when used. Since the prediction increments can be signed, this estimator requires a
different argument from the nonnegative LPB product bound.
Theorem 3(UPB PAC bound for signed updates).Assuming the calibration and test data are i.i.d., use the
depth/propensity and prediction-path construction of Appendix D. Suppose the effective reach weights ρitare
nonincreasing along each prefix (equivalently, their inverses are nondecreasing) and ρit≥1/W >0 on every
prefix that can contribute, with Wfixed conditional on the reference. Then the UPB suffix-envelope selector with an
infinite fallback satisfies(2)for UPB miscoverage with
∆U= (2W−1)r
1
Nlog1
δ.(15)
Acquisition enters this correction only through W. Summation by parts bounds each sample contribution in
[1−W, W] , and conditioning on sample-local counterfactual continuations makes it affine in the commitment
indicator. We apply the two pivotal cylinder inequalities to positive and negative coefficients separately, then
use Cauchy–Schwarz to compare exponential moments with independent sample marginals at a factor-of-two
cost. Combining these conditional bounds across groups handles reflow. Appendix D.4 proves the result and a
variance-sensitive version. Small propensities can still make the range-only bound vacuous.
4 Experiments
We evaluate HARP using the same datasets and setup as in [ 4]: toxicity, red teaming with Qwen and Llama Guard
judges, retrieval-grounded hallucination, and AutoIF instruction following. Each setup uses Qwen 2.5 14B, Llama
3.1 8B, Phi-4 Mini, and Gemma 3 4B as target models. We use a survival model pre-trained on 4000 independent
samples to predict event time through M= 200 . We remark that the budget constraint is for the calibration step
only; the training phase has a separate budget.
Protocol and budget choice.We fix one target budget per dataset and task. For LPBs, we follow the configuration
of Feldman and Romano [4]and set target ¯B= 20 budget units per sample for the toxicity data and ¯B= 10 for
other datasets. For constructing UPBs and estimating metrics, we set ¯B= 20 throughout. Appendix F provides the
full evaluation protocol.
Each experiment is repeated twenty times for random splits of calibration (2000) and test samples (1000). HARP
uses 100 reference prompts for which no budget is consumed, and the budget is allocated over the remaining
1,900. For HARP , we set m= 25 ,η=.05 , and pmin=.02 . For DAPRO , we use 25 data points for learning an
optimized allocation policy and a different 25 samples for the CRC calibration, as in [ 4]. We compare HARP -C and
HARP -D with and without depth, then choose one variant per task using coverage error and bound size for predictive
bounds, and normalized estimation MSE for metrics (Appendix F.4). The resulting choices are HARP -D (+depth)
for LPBs, HARP -D (+depth) for UPBs, and HARP -D (no depth) for metrics, applied to every dataset and target. Full
comparison of the four variants is given in Appendix F.6.
7

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0481216Coverage deviation (pp)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model16182022Paid turns per promptUncalibrated DAPRO HARP-D (+depth)Figure 1:Toxicity LPBs.Coverage comparison: absolute coverage deviation and paid turns per parent prompt, all
four target models, 90% coverage, ¯B= 20 turns per sample, twenty runs. HARP -D (+depth); chosen once for the
LPB task by the task-level criterion in Appendix F.4. Uncalibrated uses raw quantiles and has no cost box. The
dashed red line marks the budget.
Predictive bounds.On toxicity (Figure 1), DAPRO andHARP attain mean absolute coverage deviations of 1.08
and 1.05 percentage points (pp), averaged across four targets. For Qwen 2.5 on AutoIF (Figure 2), DAPRO has
smaller deviations: 1.025 versus 1.245 pp for LPBs and 1.160 versus 1.225 pp for UPBs. We remark that HARP
cannot obtain performance substantially better than DAPRO for LPB construction tasks, as the results in [ 4] show
thatDAPROperforms almost as well as an oracle alternative that has infinite budget on these tasks.
Population metrics.We estimate the marginal event rate P(T≤M) for the red teaming data with Llama Guard
as a judge and report the performance of the methods in Figure 3. While both calibration methods recover the
correct metric in expectation,HARPachieves 60.65% lower variance across the four target models.
Budget.The mean budget utilized, i.e., consumed budget divided by available budget, by HARP for LPB, UPB,
and metrics is 94.51%, 95.70%, and 97.03% compared to 91.58%, 69.14%, and 82.80% of DAPRO .HARP never
exceeds the target budget across all experiments, while DAPRO ’s overrun frequencies are 13.00%, 3.00%, and
12.75%, respectively. This is not a surprise, as Theorem 1 guarantees a deterministic budget control of HARP , while
DAPROprovides only an expected budget guarantee.
8

LPB UPB
Bound type05101520Coverage deviation
(pp)
LPB UPB
Bound type10.012.515.017.520.0Budget used
per sampleUncalibrated DAPRO HARP-D (+depth)
T arget budgetFigure 2:AutoIF LPBs and UPBs, Qwen 2.5 14B.Coverage deviation (left) and budget used per sample (right),
over twenty runs. Target coverage: 90% for LPB and 80% for UPB; target budgets: 10 and 20 turns per sample,
respectively (red dashed marks). HARP uses D with depth for both bounds, selected by the criterion in Appendix F.4.
Uncalibrated has no acquisition-cost box.
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model75808590Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model1.53.04.5Event-rate
variance (pp2)Uncalibrated DAPRO HARP-D (no depth)
Reference means
Figure 3:Red-team event rates, Llama Guard judge.Estimates and across-seed variances, all four targets,
¯B= 20 turns per sample, twenty runs. HARP -D (no depth); chosen once for metric estimation by the task-level
criterion in Appendix F.4. Red marks denote benchmark-cohort means. Acquired methods use the same estimator.
9

5 Discussion
We introduced HARP for multi-turn allocation and calibration under hard budget constraints. Unlike DAPRO [4],
HARPdoes not require a policy-learning phase or a CRC budget-calibration step. Instead, it couples the censoring
times of different samples. Despite this dependence, we show that the predictive bounds and estimated metrics of
HARP satisfy finite-sample guarantees. Our experiments support our theoretical claims and show that HARP utilizes
the budget effectively and yields estimates that have a low variance.
We remark that our guarantees require that the calibration and test data are i.i.d. with correct audit labels; in practice,
labels may be noisy or partially missing, motivating extensions of HARP that are robust to such corruptions [ 16–19].
Furthermore, HARP constructs bounds for a single event time. In many applications, it is necessary to jointly evaluate
the safety of modern AI systems such as agents, which calls for developing bounds for multiple event times [ 20].
Additionally, under smaller budgets, many event times may be censored. Model-generated or synthetic outcomes
could complement censored events [21–24]. Finally,HARPrequires a minimal total budgetB≥K(1 +η)M.
Based on our experiments, we recommend HARP -D with depth for constructing predictive bounds and HARP -D
without depth for estimating metrics. Depth reduces the median LPB coverage variance but increases the variance
of the metric estimates (see Appendix F.6).
Acknowledgments
Y . R. and S. F. were supported by the European Union (ERC, SafetyBounds, 101163414). Views and opinions
expressed are however those of the authors only and do not necessarily reflect those of the European Union or the
European Research Council Executive Agency. Neither the European Union nor the granting authority can be held
responsible for them. Y .R. thanks the Career Advancement Fellowship, Technion.
10

References
[1]Yu Gui, Rohan Hore, Zhimei Ren, and Rina Foygel Barber. Conformalized survival analysis with adaptive
cut-offs.Biometrika, 111(2):459–477, 2024. doi: 10.1093/biomet/asad076.
[2]Hen Davidov, Shai Feldman, Gil Shamai, Ron Kimmel, and Yaniv Romano. Conformalized Survival Analysis
for General Right-Censored Data. InInternational Conference on Learning Representations, 2025. URL
https://openreview.net/forum?id=JQtuCumAFD.
[3]Hen Davidov, Shai Feldman, Gilad Freidkin, and Yaniv Romano. Calibrated predictive lower bounds on time-
to-unsafe-sampling in LLMs. InThe 29th International Conference on Artificial Intelligence and Statistics,
2026.
[4]Shai Feldman and Yaniv Romano. How many iterations to jailbreak? dynamic budget allocation for multi-turn
llm evaluation.arXiv preprint arXiv:2605.06605, 2026.
[5]Khurram Yamin, Edward Kennedy, and Bryan Wilder. Dependent randomized rounding for budget constrained
experimental design. In Silvia Chiappa and Sara Magliacane, editors,Proceedings of the Forty-first Conference
on Uncertainty in Artificial Intelligence, volume 286 ofProceedings of Machine Learning Research, pages
4681–4700. PMLR, 21–25 Jul 2025.
[6]William G Madow and Lillian H Madow. On the theory of systematic sampling, i.The Annals of Mathematical
Statistics, 15(1):1–24, 1944.
[7]Jean-Claude Deville and Yves Tillé. Efficient balanced sampling: the cube method.Biometrika, 91(4):
893–912, 2004.
[8]Rajiv Gandhi, Samir Khuller, Srinivasan Parthasarathy, and Aravind Srinivasan. Dependent rounding and its
applications to approximation algorithms.Journal of the ACM (JACM), 53(3):324–360, 2006.
[9]Roger Koenker and Gilbert Bassett. Regression quantiles.Econometrica, 46(1):33–50, 1978. ISSN 00129682,
14680262. URLhttp://www.jstor.org/stable/1913643.
[10] Youngseog Chung, Willie Neiswanger, Ian Char, and Jeff Schneider. Beyond pinball loss: Quantile methods
for calibrated uncertainty quantification.arXiv preprint arXiv:2011.09588, 2020.
[11] Shai Feldman, Stephen Bates, and Yaniv Romano. Improving conditional coverage via orthogonal quantile
regression.Advances in neural information processing systems, 34:2060–2071, 2021.
[12] Vladimir V ovk, Alexander Gammerman, and Glenn Shafer.Algorithmic Learning in a Random World,
volume 29. Springer, 2005. doi: 10.1007/b106715.
[13] Daniel G. Horvitz and Donovan J. Thompson. A generalization of sampling without replacement from a finite
universe.Journal of the American Statistical Association, 47(260):663–685, 1952.
[14] Anastasios Nikolas Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal
risk control. InThe Twelfth International Conference on Learning Representations, 2024. URL https:
//openreview.net/forum?id=33XGfHLtZg.
[15] Jarosław Byrka, Thomas Pensyl, Bartosz Rybicki, Aravind Srinivasan, and Khoa Trinh. An improved approx-
imation for k-median and positive correlation in budgeted optimization.ACM Transactions on Algorithms
(TALG), 13(2):1–31, 2017.
[16] Bat-Sheva Einbinder, Shai Feldman, Stephen Bates, Anastasios N Angelopoulos, Asaf Gendler, and Yaniv
Romano. Label noise robustness of conformal prediction.Journal of Machine Learning Research, 25(328):
1–66, 2024.
[17] Coby Penso and Jacob Goldberger. A conformal prediction score that is robust to label noise.arXiv preprint
arXiv:2405.02648, 2024.
11

[18] Shai Feldman and Yaniv Romano. Robust conformal prediction using privileged information.Advances in
Neural Information Processing Systems, 37:117813–117852, 2024.
[19] Shai Feldman, Stephen Bates, and Yaniv Romano. Conformal prediction with corrupted labels: Uncertain
imputation and robust re-weighting. InInternational Conference on Learning Representations, volume 2026,
pages 30901–30942, 2026.
[20] Shai Feldman, Stephen Bates, and Yaniv Romano. Calibrated multiple-output quantile regression with
representation learning.Journal of Machine Learning Research, 24(24):1–48, 2023.
[21] Anastasios N Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I Jordan, and Tijana Zrnic. Prediction-
powered inference.Science, 382(6671):669–674, 2023.
[22] Anastasios N Angelopoulos, John C Duchi, and Tijana Zrnic. Ppi++: Efficient prediction-powered inference.
arXiv preprint arXiv:2311.01453, 2023.
[23] Meshi Bashari, Yonghoon Lee, Roy Maor Lotan, Edgar Dobriban, and Yaniv Romano. General synthetic-
powered inference. InForty-third International Conference on Machine Learning, 2026. URL https:
//openreview.net/forum?id=sxLncu2Fhx.
[24] Meshi Bashari, Roy Maor Lotan, Yonghoon Lee, Edgar Dobriban, and Yaniv Romano. Synthetic-powered
predictive inference.Advances in Neural Information Processing Systems, 38:117704–117764, 2026.
[25] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. RealToxicityPrompts:
Evaluating Neural Toxic Degeneration in Language Models. InFindings of the Association for Computational
Linguistics: EMNLP, pages 3356–3369, 2020.
[26] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann,
Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms:
Methods, scaling behaviors, and lessons learned.arXiv preprint arXiv:2209.07858, 2022.
[27] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev,
Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai
conversations.arXiv preprint arXiv:2312.06674, 2023.
[28] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for
SQuAD. In Iryna Gurevych and Yusuke Miyao, editors,Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia,
July 2018. Association for Computational Linguistics.
[29] Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play
with execution feedback: Improving instruction-following capabilities of large language models. InThe
Thirteenth International Conference on Learning Representations, 2025.
[30] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreak-
ing black box large language models in twenty queries. In2025 IEEE Conference on Secure and Trustworthy
Machine Learning (SaTML), pages 23–42. IEEE, 2025.
[31] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multilingual e5
text embeddings: A technical report.arXiv preprint arXiv:2402.05672, 2024.
12

A Additional Related Work
Inverse-probability estimation for unequal-probability sampling is classical [ 13]. Calibrated event-time prediction
under censoring also predates this work: for example, Gui et al. [1]develop adaptive cut-offs for conformalized
survival analysis. These results motivate controlling the resolution event and candidate family. Model-assisted
correction has a substantial history as well, with prediction-powered inference providing a recent general framework
for combining predictions and observed outcomes [21]. We use the sequential augmented estimator from [4].
Yamin et al. [5]study experimental assignment with a fixed treatment count. Their dependent-rounding construction
preserves first-order probabilities and supports unbiasedness, variance comparisons under specified conditions,
and asymptotic inference. The distinction from our setting is therefore the constraint and subsequent inference
problem. HARP reserves heterogeneous maximum trajectory lengths, incurs random event-stopped costs, updates
later policies from released resources, and calibrates a selected nested family of predictive bounds. We account for
these features through the reservation invariant and predictable-group analysis below.
The weighted pairwise rounding properties we use are established in the budgeted-rounding literature, including
Byrka et al. [15]. We provide a self-contained proof to specify the properties needed for calibration. In particular,
the proof uses cylinder inequalities for products of selected or unselected coordinates. Our analysis applies these
inequalities within a sequential ledger, together with sample-local continuation and subsequent predictive calibration.
DAPRO [4] is our direct baseline. Its history-dependent score, inverse-probability correction, and augmented UPB
construction remain relevant to our setting. HARP additionally connects worst-case reservations to realized costs
and justifies inference under the resulting dependence.
B Complete Budget and LPB Proofs
B.1 Weighted pivotal rounding
Fix a group, positive costs di≤M and starting probabilities yi=pc
i. Coordinates initially at zero or one need no
update. At each step, choose two fractional coordinatesi, jand set
a+= min{d i(1−y i), djyj}, a −= min{d iyi, dj(1−y j)}.(16)
Draw s=a +with probability a−/(a++a−)ands=−a −otherwise, and replace (yi, yj)by(yi+s/d i, yj−s/d j).
At least one coordinate becomes integral at each step. If one fractional coordinate remains, round it by a Bernoulli
draw with its current value as success probability. Denote the resulting vector byζ.
Lemma 1(Properties of the weighted pivotal rounding).Conditional on the input group, weighted pivotal rounding
satisfies
Eζi=pc
i,X
idiζi≤X
idipc
i+ max
idi,(17)
EY
i∈Jζi≤Y
i∈Jpc
i,EY
i∈J(1−ζ i)≤Y
i∈J(1−pc
i)for everyJ.(18)
In particular, for fixedh i∈[0,1],
EY
i(1−h iζi)≤Y
i(1−h ipc
i).(19)
Proof. Each pairwise step has E[s|y] = 0 , so every coordinate is a martingale and the weighted sum is unchanged.
The final Bernoulli step changes the weighted sum by at mostmax idi≤M. This proves (17).
13

For a fixed subset J, considerQ
i∈Jyi. If neither updated coordinate is in J, the product does not change. If exactly
one is in J, its conditional expectation is unchanged. If both are in J, their product changes by a linear term in s
minus s2/(didj). Multiplication by the other nonnegative coordinates preserves the nonpositive conditional drift.
The same calculation for (1−y i)(1−y j)gives a nonpositive quadratic term. Both subset products are therefore
supermartingales until the last fractional coordinate. That final step is linear in its single coordinate and preserves
their conditional means. This proves (18) without requiring any independence of the output coordinates.
Finally, write 1−h iζi= (1−h i) +h i(1−ζ i)and expand the product. Every coefficient in this expansion is
nonnegative. Applying the second inequality in (18) term by term proves (19). Zero-cost samples can be set to one
outside the pivot operation; they do not affect the cost bound or these inequalities.
B.2 The pathwise ledger invariant
Proof of Theorem 1. At the beginning of the first group, feasibility gives L0≥K(1 +η)M . Suppose that before
groupk,
Lk−1≥(K−k+ 1)(1 +η)M.(20)
Then κk≥(1 +η)M , and in particular κk−M≥ηM >0 . Summing the cap in (8)over the positive-cost samples
givesX
i∈Ikdipc
i≤#{i∈I k:di>0}
mk(κk−M)≤κ k−M.
Lemma 1 implies that the selected horizons cost at most κk. Each selected sample executes no more than diturns.
Event stopping and D’s continuation stopping can only lower that cost. ThusS k≤κkand
Lk=Lk−1−Sk≥Lk−1−κk= (K−k)(1 +η)M.
This is the invariant for the next group. At k=K it gives LK≥0, provingP
kSk=B−L K≤B . All
inequalities are pointwise. Randomized depth is covered by conditioning on its chosen horizons, all of which remain
in[0, M].
Lemma 2(Positive reach).Under the same conditions, pc
i≥p0= min{p min, η/m} for every positive-cost sample.
Proof. The cap before clipping at one is at least ηM/(m kdi)≥η/m . The proposal is at least pmin, so their
minimum with one is at least p0. C introduces no further thinning. D has at most Mcontinuation factors, each at
leastp min≤1. Free samples have probability one.
The specified continuation rule admits a stronger bound when ω <1 . Suppose λk≤Λ<∞ for every group and
the predicted remaining cost lies in [0, M] . The model probabilities give νi(t)≥1−ω , hence ui(t)≥u 0:=
min{1,p
(1−ω)/(ΛM)} . Equation (10) implies ai(t)≥min{a i(t−1), u i(t)}, whether ui(t)is above or below
the previous reach. Induction therefore yields
πi≥ep0:= min{p 0, u0}for the specified D policy.(21)
The interpolation in Appendix C.1 remains within the reference multiplier grid, whose largest value therefore
supplies Λ. The reported configuration uses Λ = 109andω= 1/2 . For M= 200 , this gives ep0≥1.58×10−6.
Although considerably stronger than p0pM
min, it remains too small for informative coarse uniform PAC corrections.
For arbitrary continuations with onlyP i(t)≥p min, the product-of-floors bound still applies.
14

B.3 A filtration that permits reflow
Condition on the training data, fitted model, independent reference design, and independent group permutation.
After this conditioning, write the iid full working trajectories as W1, . . . , W N, with XiandTimeasurable functions
ofWi. LetFk−1contain those fixed design objects, all complete trajectories in previous groups, and all random
coins used in those groups. The observed past is a sub- σ-field, so the actual ledger and λkareFk−1-measurable.
Complete trajectories in groupkremain independent ofF k−1and of each other.
Within the group, pc
iis a function of (Wi,Fk−1), in fact only of the initially available portion of Wi. Fix independent
counterfactual continuation streams for each sample, whether or not that sample is ultimately committed. The
stream and its own trajectory determine Ui, the indicator that continuation would resolve the sample if committed,
and the productq i=Q
t≤biPi(t). For C setU i=qi= 1. WithZ i=ζiUiandπ i=pc
iqi,
E[Zi|WIk,Fk−1] =π i,EZiAi(τ)
πiWIk,Fk−1
=A i(τ).(22)
The expectation here averages over pivot and continuation randomization; it does not imply independence of the Zi.
Each qiis a counterfactual product along that sample’s full history, rather than the probability of reaching only a
prematurely censored prefix. In practice, we need the full product only for samples withZ i= 1.
B.4 Concentration and selection for LPBs
For a fixedτ, abbreviateXτ
i=ZiAi(τ)/π iandµ=µ(τ). Conditional onF k−1,
EXτ
i=µ, σ2
ik:= Var(Xτ
i| Fk−1) =E[A i(τ)/π i| Fk−1]−µ2.(23)
For any s≥0 , condition first on the current full trajectories and their counterfactual continuation streams. Apply (19)
withhi= 1−exp{−sU iAi(τ)/π i} ∈[0,1] . Then integrate the independent continuation streams and current
trajectories. Sample locality permits this final factorization, giving
Eh
e−sP
i∈IkXτ
i| Fk−1i
≤Y
i∈IkE[e−sXτ
i| Fk−1].(24)
For completeness, if a centered random variable Y≤1 has variance σ2, Taylor’s formula with integral remainder
implies
esy≤1 +sy+ (es−1−s)y2(y≤1, s≥0).
Indeed, the remainder divided by s2y2isR1
0(1−t)etsydt, at most its value at y= 1 . For 0< s <3 , the series
boundes−1−s≤s2/[2(1−s/3)]therefore yields
EesY≤exps2σ2
2(1−s/3)
.(25)
Apply this withY=µ−Xτ
i≤µ≤1. By (24), the process
exp

sX
j≤kX
i∈Ij(µ−Xτ
i)−s2
2(1−s/3)X
j≤kX
i∈Ijσ2
ij

(26)
is a nonnegative supermartingale in the enlarged group filtration. Adaptive changes of the ledger are allowed because
the group bound is conditional on its current past.
15

Ifv(τ)≤v ⋆almost surely, the compensator sum is at most N(v ⋆−µ2). The exponential Markov inequality and
optimization oversgive, forx >0,
P{µ−bµ(τ)≥x} ≤exp
−Nx2
2(v⋆−µ2+x/3)
.(27)
The usual zero-variance case follows by a limit or directly from the deterministic centered sum. This is a lower-tail
bound for a nonnegative HT estimator; no bounded upper weight is needed beyond existence of the stated second
moments.
Proof of Theorem 2. If no retained candidate has population risk greater than α+ ∆ , the assertion is automatic.
Otherwise let τ†be the first such candidate. It is fixed after conditioning on the reference and model, even though
it is unknown to the procedure. Since the population losses are nested, µ(bτ)> α+ ∆ implies bτ≥τ†. The
inclusive-prefix rule then forcesbµ(τ†)≤α, and consequently
µ(τ†)−bµ(τ†)>∆.
Use the exponential argument above at this single candidate, bounding its variance compensator by N(v ⋆−α2)
since µ(τ†)> α . The resulting upper bound is exp{−N∆2/[2((v ⋆−α2)++ ∆/3)]} , at most δunder (13). If
α+∆≥1 , the assertion is already trivial. Integrating over the conditioned reference/model preserves the probability
bound, including when the declared envelope and∆are functions of that reference.
B.5 Restricted moment envelopes and jumps
Order the retained finite grid asτ 0= 0<···< τ J. Foru≥0, define
T+(u) ={τ 0} ∪ {τ j:j≥1, µ(τ j−1)≤u}.
Suppose v♯(u)bounds (12) almost surely at every candidate in T+(u). Theorem 2 remains valid if (13) uses
v♯(α+ ∆)in place ofv ⋆. Its proof is identical because the predecessor ofτ†is safe and henceτ†∈ T+(α+ ∆).
An envelope restricted to {τ:µ(τ)≤u} would miss the crossing candidate at a risk jump and would not justify
this proof. Such jumps arise naturally with discrete event times and quantile grids.
B.6 Expected unused budget
We next quantify the effect of conditional pacing on unused budget.
Proposition 1(Pacing-defect identity).LetS kdenote actual group spend and define the predictable defect
ek=Lk−1
K−k+ 1−E[S k| Fk−1].
ForK≥2,
ELK≤EL K−1=B
K+K−1X
k=1Eek
K−k≤B
K+K−1X
k=1E(ek)+
K−k.(28)
In particular, exact conditional pacing in every nonfinal group implies ELK≤B/K . For K= 1 the bound
EL1≤Bis immediate.
16

Proof.Puth=K−k+ 1. FromL k=Lk−1−Sk,
E[Lk| Fk−1] =h−1
hLk−1+ek.
Taking expectations and iterating for k= 1, . . . , K−1 telescopes the coefficient of Bto1/K . The coefficient of
EekisQK−1
j=k+1(K−j)/(K−j+1) = 1/(K−k) . This proves the equality. Nonnegative spend gives LK≤LK−1,
ande k≤(e k)+gives the last inequality.
Recycling savings does not ensure that the reference-based λkeliminates the pacing defect. Commitment caps,
inaccurate cost forecasts, continuation thinning, or a shortage of useful remaining turns can all affect realized spend.
For example, suppose all Ti= 1andB > N . Any policy that stops at an event can spend at most Nturns on these
Nsamples, regardless of its ledger accuracy. This rules out an unconditional utilization guarantee.
C Reference Planning and Complete Algorithms
C.1 The predictions from the reference set
For LPBs, choose the initial design index closest to the target risk:
τfull
⋆∈arg min
τ≤τ priorR−1X
r∈RbP(Tr<bfτ(Xr)|X r)−α design.
We resolve ties using the first grid index. Once LPB depth ℓis chosen, we replace the design index by min{τfull
⋆, ℓ}
in grid order. For constructing LPBs, we setα design =.1.
We construct the reference set using only initial PMFs. For C at multiplier λ, the expected surrogate cost is the
mean of pprop
r(λ)bc r. For D, we condition each initial PMF on survival to the current prefix, without accessing
future reference histories. We apply the continuation rule along this synthetic survival-conditioned path and sum the
predicted reached-turn costs:
cR(λ) =1
RX
r∈RdrX
t=1bP(Tr≥t|X r)pprop
r(λ)tY
s=1Pref
r(s;λ).(29)
This expression includes commitment and the first continuation draw. It forecasts the uncapped proposal. We use
192 logarithmically spaced multipliers from 10−9to109, make the cost curve nonincreasing, and interpolate logλ
against reversed cost to match βk/mk. Targets outside the tabulated cost range use the corresponding endpoint
multiplier. Both reference model error and the subsequent commitment cap can therefore prevent exact conditional
pacing.
For a working LPB sample that has survived throught−1, the design risk in (10) is
νi(t) = (1−ω) +ωbP{Ti<bfτ⋆(Xi)|H i(t−1), T i≥t}
max(¯r,10−12).
We predict remaining cost as bei(t) =bE[min(T i−t+ 1, d i−t+ 1)|H i(t−1), T i≥t]. Nonpositive remaining
lengths have zero cost, and the policy is called only for required positive turns. We safeguard against denominators
and square-root inputs by adding10−12.
17

Algorithm 2Weighted pivotal commitments (positive-cost samples)
Require:Probabilitiespc
i∈[0,1], positive costsd i≤M
1:Sety←pc.
2:whileat least two coordinates are fractionaldo
3:Pick two fractional coordinatesi, j; computea +, a−by (16).
4:Draws=a +with probabilitya −/(a++a−), otherwises=−a −.
5:Sety i←y i+s/d i,yj←y j−s/d j.
6:end while
7:ifone fractional coordinateiremainsthen
8:Sety i←Bernoulli(y i).
9:end if
10:returnζ=y.
C.2 The LPB depth planner
Let the grid indices run from 0toJ, with 0the zero bound. From the initial reference distribution, compute
nondecreasing curves
cj=R−1X
rbE[min(T r,(bfτj(Xr)−1) +)|X r],(30)
bµj=R−1X
rbP(Tr<bfτj(Xr)|X r), s j=R−1X
rbfτj(Xr).(31)
UseΓ ={.6, .8,1,1.25,1.6} ,δplan=.1,ϵplan=.02 ,B′= (1−ϵ plan)B/N , andgj= min{1, B′/max(c j,10−12)}.
Define the surrogate width and attainable value by
Uj,r(γ) =γbµ r+s
2 log(5/δ plan)γbµr
gjN+log(5/δ plan)
3gjN,(32)
Vj(γ) = max 
{sr:r≤j, U j,r(γ)≤α design} ∪ {0}
.(33)
Amongj= 1, . . . , J, choose the first minimizer
ej= arg min
jmax
γ∈Γ{max
hVh(γ)−V j(γ)}.
Letjaff= max{j:c j≤B′}, with zero allowed, and set bj= min{J,max(1, ej, jaff)}. The retained endpoint is
ℓ=τbj. The quantities gjandUj,rare only planning surrogates: the actual resolution probability is determined by
the ledger, pivot, and, for D, continuation.
We implement continuation with independent random uniforms; C can omit draws with probability one. An
unresolved sample contributes zero to the LPB estimator. For sequential AHT, each reached turn uses its own
recorded cumulative probability, including commitment at the first turn. Although a replay cache may store full
histories, the policy must not access future predictions.
D Upper Predictive Bounds
D.1 Candidate family, acquisition, and selection
For each finite candidate fi=bfτ(Xi)≤M , define AU
i(τ) =1{T i> fi}. When fi=∞ , we set AU
i= 0
deterministically. We represent infinity by 201 when M= 200 ; this value is not an observed event on turn 201. We
18

Algorithm 3HARP-C/D execution with exact acquisition records
Require:Working initial features, fixed reference set, horizonsd i, capB
1:Independently permute samples; form groups of size at mostm; verify (4).
2:SetL=Band initialize all observed-turn counts to zero.
3:fork= 1, . . . , Kdo
4:Computeκ k, βkfrom (5); choose reference multiplierλ k.
5:Computepc
iby (8); setpc
i=ζi= 1ford i= 0.
6:Round positive-cost samples with Algorithm 2; setS k= 0.
7:foreach samplei∈I kdo
8:Seta i=pc
i; record commitment and horizon; ifd i= 0, mark resolved.
9:ifζ i= 1andd i>0then
10:fort= 1, . . . , d iuntil an event or a continuation stopdo
11:For C setP i(t) = 1; for D use (10) from the acquired prefix.
12:RecordP i(t)and updatea i←a iPi(t).
13:Draw continuation; if it fails, stop this sample without paying for turnt.
14:Otherwise acquire turnt, increment the sample’s count andS k, and update history.
15:If the event occurs ort=d i, mark resolved and storeπ i=ai.
16:end for
17:end if
18:end for
19:SetL←L−S k; retain the records needed for the chosen estimator.
20:end for
21:returnAcquired prefixes, commitment/continuation probabilities, resolution flags, total spend.
retain the grid through τprior, together with the last, all-infinite candidate, excluding finite candidates beyond the
acquisition prior. Every retained finite candidate therefore satisfiesf i≤di= min{ bfτprior(Xi), M}.
We choose the reference anchor τa≤τprior as the first candidate for which the entire subsequent reference-predicted
coverage suffix reaches the target 1−α . If no candidate satisfies this condition, we use τprior. Unlike the LPB
closest-risk anchor, this rule uses a coverage suffix. The design score also changes: we predictsurvival beyond
bfτa(Xi)from the reference and observed history, rather than an event strictly before an LPB endpoint. An infinite
anchor has zero initial risk. We then apply the same smoothing and cost-based continuation formula.
Without depth, each sample has acquisition horizon di. ForHARP with depth version, let ai= min{ bfτa(Xi), di},
clipped to[1, M], and draw independently
Di=(
di,with probabilityq,
ai,with probability1−q,q=.25.(34)
We use Dias both commitment cost and horizon in Algorithm 3, and form the reference cost curve from the
corresponding q/(1−q) mixture of full and shallow forecasts. This reduces late acquisition while retaining every
finite target and the reported UPB family. Define the depth-tail function
Gi(t) =P(D i≥t|X i) =(
1, t≤a i,
q, a i< t≤d i.(35)
The commitment and continuation probabilities can depend on the sampled Di. Conditional on that depth and the
complete trajectory, write their product through turn tasrDi
i(t)when t≤D i. Beyond Di, extend the product
positively, for example, with continuation factors of one; those terms are never observed. The effective weighting
quantity is
ρDi
i(t) =G i(t)rDi
i(t).(36)
19

Since rDi
ican differ between the two horizons, this quantity is generallynotthe unconditional marginal reach
probability. Validity instead follows from the weighting identity below. To represent Gi(t), we multiply the recorded
conditional probability at turntbyG i(t)/G i(t−1), usingG i(0) = 1.
For a finite candidate, we initialize mi0(τ) =bP(Ti> fi|Xi). If an event occurs by fi, we set the post-event
prediction to zero; if the sample survives through fi, we set it to one. Before resolution, we use the pre-trained
model at the reached history, and after resolution, we keep the value fixed. For an infinite candidate, all predictions
and the loss remain zero. We apply (14) until resolution or acquisition stopping and average sample contributions to
obtainbµ U(τ).
Although population miscoverage decreases with τ, the signed empirical curve need not be decreasing. We therefore
select thesmallestcandidate satisfying
max
u≥τbµU(u)≤α,(37)
where the maximum includes the retained finite candidates and the infinite fallback. Equivalently, we require the
minimum empirical coverage over the suffix to be at least 1−α . The selector uses this ordering, rather than an
isolated crossing of the raw empirical curve.
D.2 Unbiasedness with depth-dependent acquisition
Lemma 3(Sequential weighting identity).Fix the past groups and full current trajectories. Assume the independent
depth draw has a positive tail Gi(t)on each required prefix, and the model prediction path is a causal function of
the trajectory and fixed design, not of the sampled depth. Then the estimator in(14), with(36), satisfies
E[Vi(τ)|W Ik,Fk−1] =AU
i(τ).
Proof. Consider a turn t≤min(T i, fi)where a prediction increment can be nonzero. Conditional on all current
depths and full trajectories, the marginal commitment property and the independent continuation stream give
E[Rit|D, W Ik,Fk−1] =1{D i≥t}rDi
i(t).
After division by Gi(t)rDi
i(t)and averaging over the independent depth draw, the expectation is one. Depth-
dependent commitment probabilities cancelinsidethat expectation. The prediction increment is fixed by the
trajectory and is independent of the depth draw, so every term has expectation mit−m i,t−1 . Their sum telescopes
from mi0to the terminal value AU
i. Later increments are zero. Infinite candidates are deterministically zero. This
proof requires correct design probabilities and correct resolution.
For the superpopulation theorem, we reveal the current group’s independent depth coins only when processing
that group in the proof filtration. Practically, these coin tosses can be pre-sampled without requiring the proof to
condition on all future draws.
D.3 A valid range bound for signed updates
Lemma 4(Range of a sequential augmented contribution).Let m0, . . . , m J∈[0,1] and1≤w 1≤ ··· ≤w J≤W .
Then
1−W≤m 0+JX
t=1wt(mt−m t−1)≤W.
The same interval containsm 0whenJ= 0.
20

Proof.ForJ≥1, summation by parts gives
(1−w 1)m0+J−1X
t=1(wt−wt+1)mt+wJmJ.
All coefficients except the last are nonpositive and sum to 1−w J. Their minimum contribution is 1−w J, while
the positive coefficient contributes at most wJ. Hence the expression lies in [1−w J, wJ]⊆[1−W, W] . This
argument is sharper than separately bounding allJincrements byW.
Every observed sample consists of a prefix. Along that prefix, (36) is nonincreasing because both Gi(t)and the
conditional-depth reach are nonincreasing, so the range lemma applies to each acquired sample. A sufficient uniform
choice is
W=(
(qp0)−1,C with insured depth,
(qp0pM
min)−1,D with insured depth,
withq= 1 when depth is not used. For the specified D policy, (21) improves the sufficient choice to W= (qep 0)−1.
D.4 Signed pivotal concentration and adaptive reflow
To analyze signed updates, we use the structure of the UPB estimator rather than treating each group as an arbitrary
bounded sum. Fix a candidate and the group past. For each sample, introduce a latent variable Uicontaining
its complete trajectory, depth draw, and independent counterfactual continuation stream. These variables are
conditionally independent across samples. The commitment probabilities pc
iand horizons are functions of Uiand
the past, computed using only the information available to the procedure. We can then write
Vi=m i0(Ui) +ζ ihi(Ui),(38)
where hisums the weighted prediction increments that would be acquired if sample iwere committed. The
coefficient himay have either sign, but is independent of the pivot coins conditional on Uiand the past. This
representation applies to C and D because within-sample stopping never depends on another current sample’s history
and the group multiplier is fixed. It also applies to insured depth: the draw belongs to Uiand does not change the
model prediction path.
Lemma 5(A signed exponential comparison).Under this sample-local representation and the two cylinder
inequalities of Lemma 1, for every realsfor which the marginal moments exist,
Eh
e−sP
i∈IkVi| Fk−1i
≤Y
i∈Ik 
E[e−2sV i| Fk−1]1/2.(39)
Proof. Condition on every Uiand the group past, so all coefficients and probabilities are fixed. For s≥0 , split
indices intoJ +={i:h i≥0}andJ −={i:h i<0}. Cauchy–Schwarz gives
Eζe−sP
iζihi≤
Eζe−2sP
i∈J+ζihi1/2
Eζe−2sP
i∈J−ζihi1/2
.
The first expectation is bounded byQ
i∈J+(1−pc
i+pc
ie−2sh i)using the down-cylinder expansion in (19). In the
second, each factor is 1 + (e−2sh i−1)ζ i, with a nonnegative coefficient. Expansion and the up-cylinder inequality
give the same product form for J−. Including the fixed baselines therefore bounds the full conditional expectation
byY
ih
(1−pc
i)e−2sm i0+pc
ie−2s(m i0+hi)i1/2
.
21

Each bracket is a function of its own Uiand the past. Conditional independence of the Uifactors the expectation of
their product. Jensen’s inequality for the square root bounds each factor by the square root of its expectation. That
expectation is the actual marginal E[e−2sV i| Fk−1], because the pivot preserves the conditional marginal pc
i. This
proves (39). Fors <0, interchange the two sign sets.
Proof of Theorem 3. By Lemma 3, E[Vi| Fk−1] =µ U(τ) =µ for a fixed candidate. Its range has length
a= 2W−1 by Lemma 4. Conditional Hoeffding’s lemma gives Eet(µ−V i)≤et2a2/8; this follows by bounding the
exponential on the range interval by its endpoint chord and maximizing the centered two-point moment generating
function. Apply (39) and bound each centered marginal mgf att= 2sto obtain
Eh
esP
i∈Ik(µ−V i)| Fk−1i
≤es2mka2/4.
Iterated conditional expectations over the ledger filtration give Ees(Nµ−P
iVi)≤es2Na2/4. Exponential Markov
withs= 2x/a2yields
P{µ−N−1X
iVi≥x} ≤e−Nx2/a2.(40)
Reverse the candidate order, starting at infinity. Population loss is now nondecreasing and (37) is an inclusive-prefix
rule. Unsafe selection requires underestimating the fixed first population-unsafe candidate by more than ∆U.
Apply (40) there and integrate over the independent reference, as in the LPB proof.
Corollary 1(Variance-sensitive signed-UPB bound).Under the same assumptions, suppose a reference-measurable
envelopev Usatisfies, almost surely at every retained candidate,
1
NX
kX
i∈IkE[Vi(τ)2| Fk−1]≤v U.
Then(2)also holds for any correction satisfying
∆≥2WL
3N+r
4W2L2
9N2+4(vU−α2)+L
N, L= log(1/δ).(41)
Proof. Since µ−V i≤W , the scaled one-sided Bernstein bound (25) applies. Combining it with (39) gives a
conditional group mgf bounded by
exp(
s2P
i∈IkVar(V i| Fk−1)
1−2sW/3)
,0< s <3/(2W).
The sum of marginal conditional variances over groups is at most N(v U−µ2). The exponential-supermartingale ar-
gument used for LPBs, now with variance factor two and scale 2W/3 , yields a lower-tail bound exp{−Nx2/[4(v U−
µ2+Wx/3)]} . At the first unsafe candidate, replace µ2byα2and solve the resulting quadratic inequality to
obtain (41).
We may take the minimum of the range-only and variance-sensitive corrections, since each bounds the same fixed-
candidate failure event deterministically conditional on the reference. Treating each group as an arbitrary dependent
block would instead give (2W−1)pP
km2
kL/(2N2). That argument requires only conditional unbiasedness and
the sample range. The sharper signed comparison also requires the sample-local affine representation and both
cylinder inequalities, which hold forHARP.
The signed comparison does not imply negative covariance between the Vi. For example, selecting exactly one of
two equally likely items and multiplying the indicators by opposite signed coefficients makes contribution covariance
positive, even though selection covariance is negative. The resulting variance is twice the sum of independent
marginal variances. Thus, reusing the LPB product argument for signed updates without an additional cost would
be incorrect. An independent-sample sequential-AHT variance diagnostic may provide marginal terms. Both valid
UPB bounds can still be vacuous with small propensities.
22

Algorithm 4UPB construction under theHARPacquisition design
Require:UPB quantiles including infinity, target1−α, reference initial PMFs
1:Retain finite candidates through the prior and the all-infinite fallback.
2:Select reference anchor by the predicted coverage suffix rule; set full horizonsd i.
3:If using depth, drawD iby (34); otherwise setD i=di,Gi= 1.
4:Build the reference cost set and run Algorithm 3 with horizonsD i.
5:foreach working sample and retained finite candidatedo
6:Start fromm i0; for each reached required turn add(m it−m i,t−1)/ρDi
i(t).
7:Impose loss zero after an event or one on survival to the finite bound; freeze thereafter.
8:end for
9:Set infinite-candidate contributions to zero; average over allNworking samples.
10:returnThe smallest candidate satisfying (37).
E Population-Level Metric Estimation
FollowingDAPRO’s evaluation setup, we consider full-horizon event probability and restricted mean event time:
pM=P(T≤M), r M=E[min(T, M)].(42)
For harmful events, these quantities describe failure probability and time to failure. In AutoIF, the event is successful
completion, so the interpretation is reversed. Neither target depends on the selected LPB endpoint. The mean event
time conditional onT≤Mis a separate, third estimand.
For metric acquisition, we use design event T≤M and full horizon M, retaining the same commitment
and continuation rules for C and D. With depth, the LPB reference planner supplies a shallow horizon ai=
clip((bfℓ(Xi)−1) +,1, M) , with probability .25of acquiring to M. Applying the same Gi(t)correction as in (35)
preserves the full-horizon target. The planner optimizes a surrogate for attainable LPB size, however, rather than
metric variance.
LetRiTindicate that an event at Ti≤M was observed and let ρi(Ti)be its effective reach weight. Ordinary
unbiased estimators are
bpHT
M=1
NX
iRiT1{Ti≤M}
ρi(Ti),br M=M−1
NX
iRiT(M−T i)1{T i≤M}
ρi(Ti).(43)
We assign zero contribution to unobserved events, so no unknown event propensity is needed for them. The
complement form for brMalso avoids requiring a fully observed event-free trajectory, using the identity min(T, M) =
M−(M−T)1{T≤M} . We use this same terminal weighted construction for restricted mean under both acquired
methods.
For event rate, both acquired methods use sequential augmentation. In (14), we let mitpredict 1{Ti≤M} , set it to
one upon observing an event and to zero upon event-free survival through M, and average Vi. The depth-averaged
telescoping proof in Lemma 3 applies unchanged. We apply this estimator to both acquired methods.
For a fixed finite population, the weighting identity gives design unbiasedness for the working-sample mean,
conditional on the complete benchmark and reference split. Averaging over a uniform reference split recovers the
parent-cohort mean. With iid working trajectories, it also gives population unbiasedness for (42); the budget theorem
applies in either setting. For a bounded sequential augmented metric, the proof of Theorem 3 gives a two-sided
population bound with log(2/δ) replacing log(1/δ) , without a candidate-selection argument. A normalized metric
in[0,1]uses the same range lemma.
23

Algorithm 5Population metrics from oneHARPacquisition
1:Freeze the full target horizonMand an independent initial-prediction reference design.
2:Used i=M; optionally draw insured shallow/full horizons with positive full-depth mass.
3:Run Algorithm 3; retain reached prefixes and their effective weights.
4:Estimate the event probability by the sequential event-target version of (14).
5:Estimate restricted mean by (43).
For conditional event time, form the weighted ratio
btcond=P
iRiTTi1{Ti≤M}/ρ i(Ti)P
iRiT1{Ti≤M}/ρ i(Ti)
when its denominator is positive. Although both numerator and denominator are unbiased totals, their ratio is not
generally unbiased. We report a zero denominator as undefined, rather than assigning a value of zero.
F Experimental Details
F.1 Benchmarks, event definitions, and data generation
We use the datasets, generated conversations, event labels, and prediction caches from [ 4] without modification.
The five settings are RealToxicityPrompts toxicity [ 25], Anthropic red teaming [ 26] with a Qwen judge or Llama
Guard [ 27], SQuAD-based retrieval-grounded hallucination [ 28], and AutoIF instruction following [ 29]. In safety
settings, the event is the first response judged harmful or ungrounded; in AutoIF, it is the first response passing all
executable checks. The two red-team judges give distinct evaluation configurations. We use one-based event times,
with 201 representing no event within theM= 200paid-turn horizon.
Qwen 2.5 14B is the attacker/helper in the inherited PAIR-style conversation setup [ 30]; Section 4 lists the four
targets. We retain DAPRO ’s generation hyperparameters, judge parsing, and retry rules without retuning them for
HARP . Each configuration has 10,000 initial prompts, split into 4,000 for model training and 6,000 for acquisition
and calibration evaluation. Conversation generation and model fitting are unchanged.
F.2 Prediction model
We embed text histories with Multilingual-E5-large [ 31]. The model projects these embeddings to 258 dimensions
with ReLU and dropout .2, then applies three pre-normalized Transformer layers with six attention heads, feed-
forward width 1,032, GELU, and a causal attention mask. A multilayer perceptron maps the concatenated
representations to survival logits. The conditional output is an M×(M+ 1) PMF grid. Its final outcome class
represents no event throughM, and its support is restricted to times after the observed prefix.
Training uses Adam with learning rate 10−4, weight decay .001, and batch size 64 for at most 500 epochs. Ten
percent of the training set is reserved for validation, with early-stopping patience of 50. We fit a time-dependent
temperature using L-BFGS on the validation survival negative log-likelihood. During HARP runs, row tof the
conditional PMF cache becomes available only after acquiring the first tinteractions. Reference planning uses row
zero alone, conditioning this initial PMF on survival to construct surrogate later predictions. These proxies might
not be the same as the model’s predictions after observing a real future response.
24

Table 1: Target budgets in paid interaction turns per parent calibration sample. Each setting uses the same budget
for all four target models; both acquired methods include their planning and control costs.
Dataset/judge LPB UPB Metrics
Toxicity 20 20 20
Red team, Qwen 10 20 20
Red team, Llama Guard 10 20 20
Hallucination 10 20 20
AutoIF 10 20 20
F.3 Acquisition settings and comparison accounting
We use the LPB budgets from the corresponding DAPRO comparison. For full-horizon metrics and UPBs, we use
twenty turns per prompt throughout, retaining access to late outcomes while staying well below full-observation cost.
Budget choice does not depend on the selected HARP variant. With N0= 2,000 ,R= 100 ,m= 25 ,M= 200 and
η=.05 , the reserve is 76·210 = 15,960 turns, or 7.98 per parent prompt, so both prescribed budgets are feasible.
We use a continuation/proposal floor of .02, risk smoothing ω=.5 , and full-depth probability .25 for insured depth.
ForDAPRO , we use the two-bin, current-hazard soft-prefix policy with global regularization .001 and sample-cost
cap2¯Bin the independent CRC controller. Its first stage has fifty paid samples: twenty-five for fitting the policy and
twenty-five for CRC. HARP ’s independent reference set uses initial predictions, incurs no interaction cost, and is
excluded from estimation. We normalize every cost by the same parent size; HARP ’s estimator uses N0−R= 1,900
working samples.
For LPBs, the grid contains zero, a 256-point logarithmic low tail below .001, and 1,000 logarithmically spaced
points from .001 to 10−.01, with prior .56. For UPBs, we use 3,001 equally spaced levels between zero and one,
prior .98, and the semantic infinite fallback. We design acquisition separately for 70%, 80%, and 90% coverage.
F.4 Choice ofHARPvariant
We compare HARP -C,HARP -C+depth, HARP -D, and HARP -D+depth using matched calibration/test splits. For each
experimental configuration, we partition 6,000 nontraining prompts into two disjoint cohorts of 3,000 prompts and
perform ten random splits within each cohort, each with 2,000 calibration prompts and 1,000 test prompts. Variant
selection uses only the ten runs from the first cohort.
We select one variant per task across all datasets and target models. For LPBs and UPBs, we rank the variants
within each configuration by their mean squared deviation from the target coverage, then average these ranks equally
across configurations. Ties are resolved using bound size, favoring larger LPBs and smaller UPBs. For metric
estimation, we apply the same ranking procedure to the average squared error of the event-rate and restricted-mean
estimates, with restricted-mean error normalized by the evaluation horizon. This procedure selects HARP -D+depth
for constructing LPBs and UPBs, and HARP -D without depth for estimating metrics. The displayed comparisons
include all twenty runs, including the ten used for selection, and therefore provide descriptive comparisons rather
than an independent held-out evaluation of the selected variant.
F.5 Statistics, figures, and provenance
Each boxplot contains twenty run-level values and shows the median, quartiles, 1.5-IQR whiskers, and all outliers.
Variance bars use sample variance with denominator nineteen. We compute ratios within configurations before
25

aggregation; a zero baseline variance gives no finite ratio. Bound comparisons report nominal-coverage deviation,
size, and UPB infinity fraction. Metric errors use the full mean of the corresponding evaluation cohort. Raw estimate
variance can include differences between cohort means.
Both acquired methods use sequential AHT for event rate and terminal HT for restricted mean. The metric grids
show absolute event-rate error, variance, and estimates with the benchmark-cohort means. Uncalibrated uses raw
initial predictions and has no acquisition-cost box. We display weighted estimates without clipping them to the
target’s natural range. Actual spend can differ across methods.
F.6 Completion and depth ablation
We restrict the method ablation to the four C/D and depth variants. Table 2 compares them over all twenty
runs, without selecting a winner for each dataset. D uses a closed-form ratio of a model-based risk/cost target to
accumulated reach, with a per-turn floor; it does not fitDAPRO’s time/score-bin policy.
Table 2: Four-variant comparison over twenty runs. Entries are median within-configuration variance ratios to
DAPRO : LPB/UPB coverage and the two population metrics. All four variants are retained to assess C/D and depth.
Variant LPB UPB Event rate Restricted mean
HARP-C (no depth) 1.935 0.622 0.632 0.466
HARP-C (+depth) 1.017 0.678 1.081 0.720
HARP-D (no depth) 1.410 0.615 0.671 0.457
HARP-D (+depth) 0.948 0.585 1.200 0.819
F.7 Interpreting the contribution of continuation and depth
Table 2 shows that depth reduces the median LPB coverage-variance ratio for both C (1.935 to 1.017) and D (1.410
to 0.948). For UPBs, it reduces D’s ratio from 0.615 to 0.585 but increases C’s from 0.622 to 0.678. D with depth
has the smallest median coverage-variance ratio among the four variants for both bound tasks, consistent with its
saved selection by coverage-MSE ranks.
For metrics, depth increases the event-rate and restricted-mean variance ratios under both C and D. In D, the ratios
rise from 0.671 to 1.200 for event rate and from 0.457 to 0.819 for restricted mean. The task-level normalized-MSE
criterion selects D without depth, whose mean rank is 1.45 versus C’s 1.90. C nevertheless has the smaller median
event-variance ratio, 0.632 versus 0.671, illustrating why event variance alone does not determine the two-metric
choice. The LPB depth objective does not optimize metric MSE, and insured metric depth preserves the target at the
cost of larger late-prefix weights. This provides a possible explanation for the observed tradeoff.
G All LPB Benchmark Comparisons
We show each dataset at its sole primary budget, with all four targets and twenty runs. The three rows display
coverage and size, absolute coverage deviation and paid cost, and coverage and size variance, respectively. HARP -D
(+depth); chosen once for the LPB task by the task-level criterion in Appendix F.4.
26

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model75808590Coverage (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0255075100LPB size (turns)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0481216Coverage deviation (pp)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model16182022Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.61.21.82.4Coverage variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model20406080LPB-size variance
(turns2)Uncalibrated DAPRO HARP-D (+depth)Figure 4:Toxicity LPBs.90% coverage, ¯B= 20 turns per sample, twenty runs per target. HARP -D (+depth); chosen
once for the LPB task by the task-level criterion in Appendix F.4. Black/red dashed lines denote coverage/budget.
All methods use matched splits.
27

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model8184879093Coverage (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model612182430LPB size (turns)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model02468Coverage deviation (pp)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model8.49.09.610.210.8Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.81.62.43.2Coverage variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model2.55.07.510.0LPB-size variance
(turns2)Uncalibrated DAPRO HARP-D (+depth)Figure 5:Red team, Qwen judge LPBs.90% coverage, ¯B= 10 turns per sample, twenty runs per target. HARP -D
(+depth); chosen once for the LPB task by the task-level criterion in Appendix F.4. Black/red dashed lines denote
coverage/budget. All methods use matched splits.
28

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model80848892Coverage (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model1.62.43.24.04.8LPB size (turns)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model036912Coverage deviation (pp)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model891011Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.51.01.52.0Coverage variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.0250.0500.0750.100LPB-size variance
(turns2)Uncalibrated DAPRO HARP-D (+depth)Figure 6:Red team, Llama Guard LPBs.90% coverage, ¯B= 10 turns per sample, twenty runs per target.
HARP -D (+depth); chosen once for the LPB task by the task-level criterion in Appendix F.4. Black/red dashed lines
denote coverage/budget. All methods use matched splits.
29

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model72808896Coverage (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model1.53.04.56.0LPB size (turns)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model05101520Coverage deviation (pp)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model7.28.08.89.610.4Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.61.21.82.4Coverage variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.0150.0300.0450.060LPB-size variance
(turns2)Uncalibrated DAPRO HARP-D (+depth)Figure 7:Hallucination LPBs.90% coverage, ¯B= 10 turns per sample, twenty runs per target. HARP -D
(+depth); chosen once for the LPB task by the task-level criterion in Appendix F.4. Black/red dashed lines denote
coverage/budget. All methods use matched splits.
30

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model80848892Coverage (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model2.55.07.510.012.5LPB size (turns)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model036912Coverage deviation (pp)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model8.49.09.610.210.8Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.61.21.82.4Coverage variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.050.100.150.20LPB-size variance
(turns2)Uncalibrated DAPRO HARP-D (+depth)Figure 8:AutoIF LPBs.90% coverage, ¯B= 10 turns per sample, twenty runs per target. HARP -D (+depth); chosen
once for the LPB task by the task-level criterion in Appendix F.4. Black/red dashed lines denote coverage/budget.
All methods use matched splits.
31

H All Population-Metric Benchmark Comparisons
For each dataset, we show one grid at ¯B= 20 , with all four targets and twenty runs. The rows display event-rate
estimates and variance, restricted-mean estimates and variance, and paid cost and absolute event-rate error. Red
dashed marks show the benchmark-cohort means; metric errors use the mean corresponding to each run. Both
acquired methods use the same estimators. HARP -D (no depth); chosen once for metric estimation by the task-level
criterion in Appendix F.4.
32

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model3040506070Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model2468Event-rate
variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model100120140160Restricted mean (turns)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model8162432Restricted-mean
variance (turns2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model15.017.520.022.5Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model036912Absolute event error (pp)
Uncalibrated DAPRO HARP-D (no depth)
Reference meansFigure 9:Toxicity population metrics.Budget ¯B= 20 turns per sample, twenty runs per target. HARP -D (no
depth); chosen once for metric estimation by the task-level criterion in Appendix F.4. The event is harmful/un-
grounded in safety settings and successful completion for AutoIF; restricted mean is E[min(T,200)] . Red cost lines
mark the cap.
33

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model56647280Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model2468Event-rate
variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model90105120135Restricted mean (turns)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model6121824Restricted-mean
variance (turns2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model15.017.520.022.525.0Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0246Absolute event error (pp)
Uncalibrated DAPRO HARP-D (no depth)
Reference meansFigure 10:Red team, Qwen judge population metrics.Budget ¯B= 20 turns per sample, twenty runs per target.
HARP -D (no depth); chosen once for metric estimation by the task-level criterion in Appendix F.4. The event is
harmful/ungrounded in safety settings and successful completion for AutoIF; restricted mean is E[min(T,200)] .
Red cost lines mark the cap.
34

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model75808590Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model1.53.04.5Event-rate
variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model456075Restricted mean (turns) Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model6121824Restricted-mean
variance (turns2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model1214161820Paid turns per promptLlama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.01.53.04.56.0Absolute event error (pp)
Uncalibrated DAPRO HARP-D (no depth)
Reference meansFigure 11:Red team, Llama Guard population metrics.Budget ¯B= 20 turns per sample, twenty runs per target.
HARP -D (no depth); chosen once for metric estimation by the task-level criterion in Appendix F.4. The event is
harmful/ungrounded in safety settings and successful completion for AutoIF; restricted mean is E[min(T,200)] .
Red cost lines mark the cap.
35

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model9095100105Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model2.55.07.510.0Event-rate
variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0153045Restricted mean (turns)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model20406080Restricted-mean
variance (turns2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model9121518Paid turns per promptLlama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model0.02.55.07.510.0Absolute event error (pp)
Uncalibrated DAPRO HARP-D (no depth)
Reference meansFigure 12:Hallucination population metrics.Budget ¯B= 20 turns per sample, twenty runs per target. HARP -D
(no depth); chosen once for metric estimation by the task-level criterion in Appendix F.4. The event is harmful/un-
grounded in safety settings and successful completion for AutoIF; restricted mean is E[min(T,200)] . Red cost lines
mark the cap.
36

Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model44485256Event rate (%)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model36912Event-rate
variance (pp2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model104112120128136Restricted mean (turns)
 Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model10203040Restricted-mean
variance (turns2)
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model16182022Paid turns per prompt
Llama
3.1Phi-4
MiniQwen
2.5Gemma
3
T arget model036912Absolute event error (pp)
Uncalibrated DAPRO HARP-D (no depth)
Reference meansFigure 13:AutoIF population metrics.Budget ¯B= 20 turns per sample, twenty runs per target. HARP -D (no
depth); chosen once for metric estimation by the task-level criterion in Appendix F.4. The event is harmful/un-
grounded in safety settings and successful completion for AutoIF; restricted mean is E[min(T,200)] . Red cost lines
mark the cap.
37