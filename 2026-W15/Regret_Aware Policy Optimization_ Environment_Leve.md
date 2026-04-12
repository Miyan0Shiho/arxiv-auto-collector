# Regret-Aware Policy Optimization: Environment-Level Memory for Replay Suppression under Delayed Harm

**Authors**: Prakul Sunil Hiremath

**Published**: 2026-04-08 17:45:45

**PDF URL**: [https://arxiv.org/pdf/2604.07428v1](https://arxiv.org/pdf/2604.07428v1)

## Abstract
Safety in reinforcement learning (RL) is typically enforced through objective shaping while keeping environment dynamics stationary with respect to observable state-action pairs. Under delayed harm, this can lead to replay: after a washout period, reintroducing the same stimulus under matched observable conditions reproduces a similar harmful cascade.
  We introduce the Replay Suppression Diagnostic (RSD), a controlled exposure-decay-replay protocol that isolates this failure mode under frozen-policy evaluation. We show that, under stationary observable transition kernels, replay cannot be structurally suppressed without inducing a persistent shift in replay-time action distributions.
  Motivated by platform-mediated systems, we propose Regret-Aware Policy Optimization (RAPO), which augments the environment with persistent harm-trace and scar fields and applies a bounded, mass-preserving transition reweighting to reduce reachability of historically harmful regions.
  On graph diffusion tasks (50-1000 nodes), RAPO suppresses replay, reducing re-amplification gain (RAG) from 0.98 to 0.33 on 250-node graphs while retaining 82\% of task return. Disabling transition deformation only during replay restores re-amplification (RAG 0.91), isolating environment-level deformation as the causal mechanism.

## Full Text


<!-- PDF content starts -->

Regret-Aware Policy Optimization: Environment-Level
Memory
for Replay Suppression under Delayed Harm
Prakul Sunil Hiremath
Department of Computer Science and Engineering, VTU, Belagavi, India
Aliens on Earth (AoE) Autonomous Research Group, Belagavi, India
prakulhiremath@vtu.ac.in
Abstract
Safety in reinforcement learning (RL) is often enforced by objective shaping (e.g.,
Lagrangian penalties) while keeping the environment response to observable state–
action pairs stationary. With delayed harms, this can createreplay: after a washout
period, reintroducing the same stimulus under a matched observable configuration
reproduces a similar harmful cascade because the observable transition law is unchanged.
We introduce theReplay Suppression Diagnostic (RSD), a controlled exposure–
washout–replay protocol that fixes stimulus identity and resets observable state and
agent memory, while freezing the policy; only environment-side memory is allowed to
persist. We prove a no-go result: under stationary observable kernels, replay-phase
re-amplification cannot be structurally reduced without a persistent shift in replay-time
action distributions. Motivated by platform-mediated systems, we proposeRegret-
Aware Policy Optimization (RAPO), which adds persistent harm-trace and scar
fields and uses them to apply a bounded, mass-preserving transition reweighting that
reduces reachability of historically harmful regions. On graph diffusion tasks (50–1000
nodes), RAPO consistently suppresses replay; on 250-node graphs it reduces RAG from
0.98 (PM-ST control) to 0.33 while retaining 82% task return. A counterfactual that
disables deformation only during replay restores re-amplification (RAG: 0.91), isolating
transition deformation as the causal mechanism.
1 Introduction
Content recommendation systems face a challenging safety failure mode: harmful content
(misinformation, extremism) generates short-term engagement but causes delayed negative
outcomes—user churn, reputation damage, regulatory penalties. Standard safe RL applies
transient penalties when harm signals arrive, temporarily suppressing similar content. How-
ever, once penalties decay and similar contexts recur, systems with stationary observable
transitions can reproduce the same harmful cascade under the same policy. We call this
replay.
1arXiv:2604.07428v1  [cs.LG]  8 Apr 2026

Replay arises from a structural mismatch: transient penalties modify objectives while
environment responses to observable inputs remain history-invariant. This differs from
exploration failures or distribution shift—it occurs undercontrolled conditions(same stimulus,
same observable state) after penalty decay.
Why existing approaches are insufficient.Constrained RL (CPO [Achiam et al.,
2017], Lagrangian methods) shapes objectives but leaves observable transitions stationary.
Policy-side memory (recurrent policies, history features) changes action selection but doesn’t
alter environment responses to given ( x, a) pairs. Shields [Alshiekh et al., 2018] prevent replay
via persistent global avoidance, which can be overly conservative.
Our approach: Platform-mediated transition deformation.We targetplatform-
mediated systems—recommendation engines, network routers, warehouse controllers, digital
twins—where a centralized component controls routing or exposure. In these settings, we
can deform transition kernels based on accumulated harm history while agents act on the
same observable inputs. This is not “changing physics”; it’s implementing safety layers that
gate transitions into regions with delayed-harm histories.
Contributions.We introduce theReplay Suppression Diagnostic (RSD), an exposure-
decay-replay protocol that isolates re-amplification under observable-matched conditions
and frozen-policy evaluation (Section 3). We prove stationary observable kernels cannot
structurally suppress replay without changing action distributions (Theorem1), motivating
environment-level intervention. We proposeRAPO, which uses persistent harm-trace ( G,
decaying) and scar ( H, irreversible) fields to deform transitions (Section 4), with theoretical
guarantees on odds contraction (Lemma 1) and utility preservation (Lemma 2). On graph
diffusion (50-1000 nodes), RAPO reduces replay 67–83% (RAG: 0.33 vs 1.02 for stationary
baselines) while retaining 78–91% utility. A policy-memory control (PM-ST) with matched
observations shows no suppression (RAG: 0.98), and disabling deformation only during replay
restores re-amplification (RAG: 0.91), providing causal evidence (Section 6).
2 Problem Setup: Replay under Delayed Harm
2.1 Observable Dynamics with Delayed Harm
The nominal environment follows stationary observable dynamics xt+1∼P 0(· |x t, at) with
reward r(xt, at), where xt∈ X is the observable state and at∈ Ais the action. We allow the
environment to maintain latent memory ξt(e.g., logs, throttling state, or safety filters) that
is not included inx t.
Stationary-observable baseline vs. environment memory.In thestationary-observable
baseline, ξtmay evolve but does not affect the conditional law of the next observable state:
for any fixed stimulus identityzand allξ t,
P(x t+1∈ · |x t, ξt, at;z) =P 0(· |x t, at).(1)
2

RAPOviolates (1)by making the observable kernel depend on persistent environment-side
fields (e.g.,G t, Ht), while keeping the agent’s observation interfacex tunchanged.
Formally, for a fixed stimulus identity z∈ Z during Exposure/Replay, the environment
evolves as
(xt+1, ξt+1)∼˜P(· |x t, ξt, at;z),
while the agent observes onlyx t.
Harm signals ˜ct≥0 arrive with delay D:˜ct=g(xt−D:t, at−D:t) for t≥D and˜ct= 0
otherwise. Policies may have arbitrary memory: at∼π(· |x t, mt) with internal state
mt+1=u(mt, xt, at, xt+1), covering recurrent networks and history features.Critically, the
baseline assumption is thatP 0remains stationary in(x, a)regardless of policy memory.
Many safety mechanisms maintain a transient penalty variable pt≥0 (dual variable,
cost accumulator) that decays when ˜ct= 0: there exist β∈(0,1) and pmin≥0 such that
pt+1−p min≤β(p t−p min).
2.2 Replay: Re-amplification under Matched Observables
We studyreplayas a property of the closed-loop system that can occur even when the policy
is held fixed. Intuitively, replay captures whether reintroducing thesamestimulus under a
matched observable configuration reproduces a similar propagation cascade.
Definition 1(Replay episode and replay suppression).Fix an evaluation policy πand an
environment (which may carry internal memory not revealed in x). Areplay episodeconsists
of two rollouts indexed by ϕ∈ {exp,rep} with a shared stimulus identity z∈ Z and a shared
observable reset statex⋆∈ X:
1.Exposure rollout ( ϕ=exp).Initialize x0=x⋆and agent memory m0=0, set stimulus
zactive, and roll out forTsteps.
2.Replay rollout ( ϕ=rep).Reinitialize the observable state to the same x⋆and reset
agent memory tom 0=0, reactivate the same stimulusz, and roll out forTsteps.
Across the two rollouts, thepolicy parameters are frozenand theagent memory is reset; any
environment-side internal variables (e.g., logs, safety filters, throttling state) arenotreset
unless explicitly stated.
LetM( τ) be a nonnegative propagation functional of a trajectory τ(e.g., peak reach,
sensitive mass, or area-under-reach). Define µexp:=E[M(τexp)] and µrep:=E[M(τrep)] under
the same (π, z, x⋆). We say replay issuppressedifµ rep< µ expandfull replayifµ rep≈µ exp.
What RSD isolates.Because the policy is frozen and the agent memory is reset between
Exposure and Replay, differences in propagation cannot be attributed to learning, explo-
ration, or internal policy state. Under matched ( z, x⋆), any change in E[M] must arise from
environment-side mechanisms that persist across rollouts (e.g., platform gating, throttling,
routing constraints, or other transition-level interventions).
3

Motivating mechanism: transient penalties with delayed harm.In many safe RL
pipelines, delayed harm signals are converted into transient penalties or dual variables that
decay when harm is absent. This can temporarily reduce harmful exposure during training,
but it does not, by itself, change how the environment responds to matched observable inputs
at evaluation time. RSD therefore evaluates replay under a frozen policy to test whether
suppression is structural (environment-mediated) rather than a byproduct of time-varying
penalties.
3 Replay Suppression Diagnostic (RSD)
RSD is a controlled evaluation protocol that isolates replay by fixing stimulus identity,
resetting observable state, and measuring re-amplification under frozen-policy evaluation.
3.1 Protocol
Each RSD episode has three phases:(1) Exposure( t= 0, . . . , T exp−1): sample stimulus
z∼ D stimonce and activate it; delayed harm may arrive.(2) Decay( t=Texp, . . . , t rep−1
where trep=Texp+Tdecay): disable the stimulus and run the system for a fixed, pre-registered
Tdecaysteps to separate Exposure and Replay.(3) Replay( t=trep, . . . , t rep+Trep−1): reset
the observable state ( xtrep←x⋆) and reset agent memory, then reintroduce thesamestimulus
z; environment-side memory persists. Inpolicy-frozen RSD, all learnable parameters are
frozen across phases; only environment state evolves.
3.2 Metrics and Baselines
We report replay-phase metrics that are not peak-only.
Replay metrics.Let Reach(t) =|A t|and Sens(t) =|A t∩V sens|.
Re-amplification Gain (RAG):
RAG :=max t∈repReach(t)
max t∈expReach(t) +ϵ.
Replay AUC ratio (AUC-R):
AUC-R :=P
t∈repReach(t)P
t∈expReach(t) +ϵ.
Sensitive-mass ratio (SM-R):
SM-R :=P
t∈repSens(t)P
t∈expSens(t) +ϵ.
Replay return (ReplayRet):normalized replay-phase return
ReplayRet :=EhP
t∈repγt−treprti
EhP
t∈repγt−treprti
GE.
4

Mechanism / auxiliary metrics (reported in figures/appendix).We reportOdd-
sRatio(stepwise harmful-entry odds contraction) to validate Lemma 1 and, for graphs, a
containment radius Rc; these are not required to interpret replay suppression and are
deferred to Figure 1 and Appendix.
Baselines. GE: reward-only.SS: stationary Lagrangian penalty shaping under P0.DR:
delayed-cost variant under P0.Shield-UM: reachability-based action blocking under P0with
a threshold tuned on held-out episodes to match RAPO’s replay return (utility-matched);
tuning and compute are in Appendix.PM-ST: policy observes ( x, G, H ) and uses RAPO
costs but samples transitions from P0(no deformation).PM-RNN: GRU policy trained under
P0with the same delayed-cost signals as DR/SS; evaluated under policy-frozen RSD.RAPO:
environment-side transition deformation using ( G, H).RAPO (off@rep): deformation
disabled only during Replay (sampling fromP 0), holding the trained policy and fields fixed.
Stimulus and observable matching in platform systems.In recommenders, zcan
denote a fixed content item or cluster (fixed metadata), and x⋆a standardized serving-time
context snapshot (coarse profile/session features). RSD matches only the observable features
used at serving time, not unobserved user/platform latents.
4Method: Regret-Aware Policy Optimization (RAPO)
4.1 Platform-Mediated Transition Deformation
RAPO targetsplatform-mediatedsettings where a centralized layer can modify theeffective
next-state distribution while leaving the agent’s observation interface unchanged. We write
P0(· |x, a ) for the nominal observable kernel and P(· |x, a, G, H ) for the gated kernel induced
by persistent environment-side memory. RAPO targets platform-mediated settings where a
centralized layer can modify the effective next-state distribution while leaving the agent’s
observation interface unchanged. We write P0(· |x, a ) for the nominal observable kernel and
P(· |x, a, G, H ) for the gated kernel induced by persistent environment-side memory (cf.
Section 2) by making the conditional law ofx t+1depend on (G, H).
4.2 Augmented State: Persistent Harm Memory
The environment maintains region-indexed fields Gt, Ht∈RR
≥0where ρ:X → { 1, . . . , R}
maps observable states to regions. Gr
tis a decaying harm-trace, and Hr
tis a persistent scar
that increases when trace exceeds a threshold. The augmented state st:= (xt, Gt, Ht) is
Markov (with a finite delay buffer for delayed harm).
4.3 Bounded, Mass-Preserving Deformation
RAPO implements gating by reweighting the nominal kernel toward safer destinations. Define
a destination conductance
ψt(x′) := clip
exp(−w GGρ(x′)
t−w HHρ(x′)
t), ψ min,1
,(2)
5

and renormalize:
P(x′|x, a, G t, Ht) =P0(x′|x, a)ψ t(x′)
Zt(x, a), Z t(x, a) :=X
yP0(y|x, a)ψ t(y).(3)
This deformation is mass-preserving (valid kernel) and bounded ( ψt∈[ψmin,1]) to avoid
degenerate shutdown. WhenP 0has local support, normalization is local.
4.4 Field Dynamics and Delayed Harm
Fields evolve via analytic environment updates:
Gr
t+1= (1−λ)Gr
t+α˜c t1{ρ(x t) =r},(4)
Hr
t+1=Hr
t+ηmax(0, Gr
t−τ).(5)
We also consider a slow-decay variant
Hr
t+1=δHr
t+ηmax(0, Gr
t−τ), δ∈[0.95,0.999],(6)
to permit gradual recovery under distribution shift. Delayed credit assignment (mapping
delayed harm to regions) is implemented via a finite delay buffer.
4.5 Training Objective and PM-ST Control
We train πθ(a|s ) on s= (x, G, H ) using PPO with Lagrangian costs that discourage
harm-trace mass and new scar formation:
L(θ, λ G, λH) =EhX
tγt 
r(xt, at)−λ GP
rGr
t−λ HP
r(Hr
t+1−Hr
t)i
,(7)
with dual variables updated by projected gradient ascent.
PM-ST control.PM-ST observes the same ( x, G, H ) and uses the same costs as RAPO,
but samples transitions from P0(i.e., disables (3)). Under policy-frozen RSD, the difference
between RAPO and PM-ST isolates transition deformation as the suppression mechanism.
5 Theoretical Guarantees
We establish two results. First, we formalize a no-go statement for replay suppression under
RSD: if the observable transition kernel is stationary in ( x, a), then Exposure and Replay
rollouts coincide under a frozen policy with reset agent memory, so replay metrics cannot
change without either (i) an action-distribution shift at replay time or (ii) a history-dependent
change in the observable kernel (Theorem 1, Corollary 1). Second, we show that RAPO’s
scar field yields a quantitativeodds contractioninto harmful regions under the deformed
kernel, providing a mechanism-level guarantee that persists across RSD phases (Lemma 1).
6

5.1 A No-Go for Replay Suppression with Stationary Observable
Kernels
We model the environment as potentially maintaining latent memory ξtthat is not included
in the observable xt. RSD resets the observable state and the agent’s internal memory, but
does not reset ξtunless explicitly stated. The stationary-observable baseline corresponds to
environments whose observable next-state law is history-invariant:
P(x t+1∈ · |x t, ξt, at;z) =P 0(· |x t, at) for allξ tand fixedz.(8)
In words, latent environment memory may evolve, but it does not affect the conditional law
ofx t+1given (x t, at).
The next theorem shows that, under (8), RSD cannot exhibit structural replay suppression
under a frozen policy with reset agent memory.
Theorem 1(No-go for replay suppression under stationary observable kernels).Consider
RSD with fixed stimulus identity zand reset observable initial state x⋆. Let πbe a frozen
evaluation policy whose internal memory is reset at the start of both Exposure and Replay.
Assume the observable dynamics satisfy (8). Then the Exposure and Replay rollout laws
coincide:
(xrep
0:T, arep
0:T−1)d= (xexp
0:T, aexp
0:T−1),(9)
and consequently, for any measurable trajectory functionalM,
E[M(τ rep)] =E[M(τ exp)].(10)
Remark1 (When action laws coincide).The conclusion follows because, under a frozen policy
πwith reset internal memory and matched observable state, the conditional action law during
Replay matches that during Exposure. This is automatic when πis Markov in xt(memoryless
in observables), and it also holds for recurrent policies when the internal memory is reset at
the start of each phase, as in policy-frozen RSD.
Proof. We show equality of finite-dimensional distributions by induction. At t= 0, both
phases start from the same observable x⋆and the agent memory is reset. Because πis frozen
and initialized identically, the conditional distribution of a0given x0is identical across phases.
Assume the joint law of the observable histories and actions ( x0:t, a0:t−1) matches across
Exposure and Replay. Given the same observable history and the same policy initialization,
the conditional law of atis identical in both phases. By (8), the conditional distribution
ofxt+1given ( xt, at) isP0(· |x t, at) in both phases, independent of latent ξt. Thus the
joint law of ( x0:t+1, a0:t) matches, completing the induction. Equality of expectations in (10)
follows.
Corollary 1(Observable suppression requires action shift or transition deformation).Under
policy-frozen RSD with reset agent memory, if E[M(τrep)]̸=E[M(τexp)]for some RSD metric
M, then at least one of the following must hold: (i) the replay-time action distribution differs
from that induced by the frozen policy under matched observables (i.e., a persistent action
shift); or (ii) the observable transition law differs from P0(· |x, a )(violation of (8)), i.e., the
environment implements history-dependent transition deformation relative to observables.
7

Interpretation and link to controls.Corollary 1 makes RSD a mechanism test. Stationary-
transition methods can suppress replay only via persistent action shifts at replay time (global
avoidance). Our PM-ST control isolates this: it provides the policy with the same history
fields and costs as RAPO but samples next states from P0, enforcing (8)and predicting no
suppression under policy-frozen RSD. RAPO violates (8)by construction via environment-side
transition deformation; disabling deformation only during Replay restores the stationary
condition and therefore restores replay.
5.2 RAPO Mechanism: Odds Contraction and Safe-Mass Preser-
vation
RAPO augments the observable state with environment-maintained fields Gt, Htand deforms
the nominal kernelP 0via destination conductance:
P(x′|x, a, G, H) =P0(x′|x, a) exp(−w GGρ(x′)−w HHρ(x′))P
yP0(y|x, a) exp(−w GGρ(y)−w HHρ(y)).(11)
This defines a Markov process on the augmented state ( x, G, H ) (and any finite delay buffer
used to implement delayed updates), with a stationary kernel on the augmented space.
Lemma 1(Harmful-entry odds contraction).Let H ⊆ X denote harmful destinations.
Assume a scar gap between harmful and non-harmful destinations:
min
y∈HHρ(y)≥h ⋆andmax
y/∈HHρ(y)≤h 0withh ⋆> h 0.(12)
For any(x, a, G, H)define
p:=X
y∈HP(y|x, a, G, H), q:= 1−p,
and the corresponding nominal probabilities underP 0:
p0:=X
y∈HP0(y|x, a), q 0:= 1−p 0.
Then the harmful-entry odds contract by the scar gap:
p
q≤exp(−w H(h⋆−h 0))·p0
q0.(13)
Proof sketch. Under (11), harmful destinations receive multiplicative weight at most e−wHh⋆
while non-harmful destinations receive weight at least e−wHh0(absorbing any common Gterms
into the same bound). In the ratio p/q, the normalization cancels, yielding the multiplicative
bound (13).
8

Multi-step compounding.In diffusion-like propagation where reaching a large harmful
mass requires repeated transitions into H, the stepwise contraction (13)compounds across
steps, leading to exponential attenuation of harmful reach as the scar gap grows.
Lemma 2(Safe-region probability lower bound).Assume the nominal kernel retains at least
δprobability of transitioning outside the harmful set, i.e., q0≥δfor all encountered( x, a).
Under the scar gap condition of Lemma 1,
q≥δe−wHh0
δe−wHh0+ (1−δ)e−wHh⋆h⋆−h0→∞− − − − − − →1.(14)
Mechanism metric.Our OddsRatio metric in RSD estimates the left-hand side of (13)
relative to the nominal kernel, and therefore directly tests whether replay suppression arises
from the predicted contraction mechanism rather than from policy-side action shifts.
6 Experiments
6.1 Setup
Why graph diffusion models recommendation replay.Graph diffusion abstracts
repeated exposure cascades: a stimulus zseeds activation that propagates via local interactions.
This captures the failure mode we target: after a washout period, reintroducing the same
stimulus under matched observable features can reproduce a similar cascade unless the
platform’s effective routing/exposure mechanism has changed. In recommendation systems,
Atcan be interpreted as reached users (or a proxy for exposure mass), and Vsensas a high-risk
community or topic cluster.
Environment.We generate directed graphs with |V| ∈ { 50,100,250,500,1000}nodes,
out-degree dout∼Uniform{ 3,5}, and edge activation probabilities puv∼Beta (2,5) (rescaled
for comparable cascade sizes across |V|). The observable state summarizes the active set
At⊆V via graph statistics (e.g., |At|, centroid, spread) and time. Actions choose an injection
strategy in {aggressive, moderate, conservative }controlling seed selection. Under the nominal
kernel, each active nodeuindependently activates neighborvwith probabilityp uv.
Stimuli, harm, and delayed credit assignment.Stimulus z∈ { 1, . . . , 20}indexes fixed
seed distributions. We designate a connected sensitive subgraph Vsens(15–25% of nodes).
Delayed harm is computed from thecausalactive setDsteps in the past:
˜ct= min(0.1· |A t−D∩V sens|,1.0), D= 50.
To implement delayed credit assignment, we inject harm-trace into the regions implicated at
timet−Dusing a normalized attribution weightw t(r) supported onA t−D∩V sens:
Gr
t+1= (1−λ)Gr
t+α˜c twt(r),X
rwt(r) = 1, w t(r)≥0.
9

For node-level graphs ( R=|V|andρ(x) =x), we use uniform attribution over affected nodes:
wt(r)∝1{r∈A t−D∩V sens}. RSD horizons are Texp= 500, Tdecay= 200, Trep= 500. Results
average over 10 graph seeds×20 RSD episodes.
Training.Unless stated otherwise, policies and value functions are 2-layer MLPs (256 units,
ReLU). We train with PPO (lr=3 ×10−4, clip=0.2, GAE- λ=0.95, batch=2048) for 2 ×106
steps. RAPO uses λ= 0.1,α= 0.5,η= 0.05,τ= 0.3,wG= 1.0,wH= 2.0; dual lr=10−2.
All RSD evaluations arepolicy-frozenand reset agent memory between Exposure and Replay.
Baselines. GEis reward-only.SSis stationary Lagrangian penalty shaping under P0.
DRpropagates delayed costs via eligibility-style traces under P0.Shieldis reachability-
based action blocking under P0using Monte-Carlo rollouts (100-step horizon) to estimate
expected sensitive mass; actions are blocked if expected sensitive mass exceeds a threshold.
To avoid overstating Shield by over-blocking, we report (i) a fixed-threshold Shield and (ii)
a utility-matched variant (Shield-UM) whose threshold is selected on held-out episodes
to match RAPO’s Replay return within a small tolerance. We also report Shield compute
as simulated transitions per environment step.PM-STis the critical control: the policy
observes ( x, G, H ) and uses the same cost terms as RAPO, but transitions are sampled
from P0(no deformation).PM-RNNreplaces the MLP policy with a recurrent policy
(GRU) over the last Hobservation steps (we use H= 50), trained with the same costs
as DR/SS under P0. This tests whether richer policy memory alone can suppress replay
under stationary observable transitions.RAPOuses environment-side deformation.RAPO
(deformation-off at Replay)disables deformation only during the Replay phase (sampling
fromP 0), holding the trained policy and fields fixed.
Partial deployment ablations.To model limited gating capacity, we evaluate:RAPO-
top-k, which applies deformation only to the top- kmost probable destinations under P0(· |
x, a) and leaves the remainder unchanged, renormalizing locally; andRAPO-local, which
applies deformation only within a designated region subset (e.g., regions overlapping Vsensand
a small neighborhood), leaving other destinations unmodified. These ablations test whether
replay suppression persists under constrained intervention.
RSD metrics.We report multiple replay-phase measures to avoid peak-only artifacts. Let
Reach(t) =|A t|and let Sens(t) =|A t∩V sens|.
Re-amplification Gain (RAG):RAG :=maxt∈rep Reach(t)
maxt∈exp Reach(t)+ϵ.
Replay AUC ratio (AUC-R): AUC-R :=P
t∈repReach(t)P
t∈expReach(t)+ϵ, which captures overall
replay-phase mass, not just the peak.
Sensitive mass ratio (SM-R): SM-R :=P
t∈repSens(t)P
t∈expSens(t)+ϵ, which directly targets harm-
relevant exposure.
Action-shift distance (ASD):to test Corollary 1, we measure ... ASD :=1
TPT−1
t=0E[D TV(π(· |x t)exp, π(· |x t)rep)].
Action-shift distance (ASD):to test Corollary 1, we measure how much the replay-time
10

Table 1: Policy-frozen RSD results (250-node graphs). Mean ±std over 10 seeds ×20
episodes.†: significant vs. PM-ST (p <0.01).
Method RAG↓AUC-R↓SM-R↓ReplayRet↑
GE 1.08±0.14 1.05±0.12 1.03±0.11 1.00±0.03
DR 1.01±0.11 0.99±0.10 0.97±0.09 0.96±0.04
Shield-UM 0.78±0.15 0.81±0.14 0.79±0.13 0.82±0.03
PM-ST 0.98±0.10 0.99±0.09 0.98±0.08 0.96±0.03
PM-RNN 1.00±0.10 1.00±0.09 0.99±0.08 0.95±0.04
RAPO0.33±0.08†0.36±0.07†0.31±0.06†0.82±0.03
off@rep 0.91±0.13 0.93±0.12 0.92±0.11 0.82±0.03
action distribution shifts under the same observable reset:
ASD :=1
TT−1X
t=0E[D TV(πexp(· |x t), π rep(· |x t))].
where DTVis total variation distance and the expectation is over the RSD rollouts. Stationary
baselines can only suppress replay by increasing ASD (persistent avoidance); RAPO targets
suppression with low ASD by changing transitions instead.
6.2 Results: Replay Suppression
Table 1 reports policy-frozen RSD metrics (mean ±std). RAPO achieves substantial replay
suppression while retaining task utility.
R1: Stationary baselines exhibit full replay.GE, DR, PM-ST, and PM-RNN show
RAG≈1.0 under policy-frozen RSD (Table 1), consistent with Theorem 1 and Corollary 1.
Critically, PM-ST and PM-RNN have access to history information and are trained against
delayed harm, yet do not exhibit structural replay suppression when transitions remain at P0.
Across these stationary baselines, ASD remains near zero under policy-frozen RSD; without
a persistent replay-time action shift, replay metrics (RAG, AUC-R, SM-R) remain near 1.
Results for SS are similar and reported in Appendix Table A.1.
R2: RAPO achieves large replay reduction.RAPO reduces RAG to 0.33 (67%
reduction vs PM-ST baseline of 0.98), with containment radius dropping from 16.9 to 8.4
hops. Effect size: ∆ RAG=−0.65, 95% CI: [−0.73,−0.57],p <10−8(Welch’st-test).
R3: Deformation-off-at-Replay restores replay (causal evidence).Disabling defor-
mation only during Replay increases RAG to 0.91, recovering most of the baseline replay.
This isolates transition deformation as the causal mechanism: the trained policy and memory
fields remain unchanged, yet suppression largely disappears when sampling fromP 0.
11

0 100 200 300 400 500
Timestep (Replay Phase)0.40.60.81.0Stepwise Odds RatioPersistent
contractionNo suppressionRAPO
PM-STRAPO (deform-off)
Baselines (GE/SS/DR)Shield
No contractionFigure 1:Replay-phase odds contraction.Stepwise odds ratio during Replay (mean ±
1 s.d.). Under stationary transitions (PM-ST and other stationary baselines), contraction
remains near 1. RAPO maintains persistent contraction; turning deformation off only during
Replay restores odds ratio near 1, supporting a causal role for transition deformation.
R4: Slow-decay scars maintain suppression.The slow-decay scar variant achieves
RAG = 0.38, showing that gradual recovery is compatible with replay suppression over RSD
timescales.
R5: Shield is less effective under utility matching and incurs higher compute.
Shield achieves moderate suppression but can be highly conservative. We report a utility-
matched variant (Shield-UM) to control for over-blocking; even under utility matching, Shield
remains less effective than RAPO and requires substantial online Monte-Carlo simulation.
R6: Partial deployment remains effective.RAPO-top- kand RAPO-local retain replay
suppression while restricting gating capacity, indicating that full kernel deformation is not
required for practical gains (details in Appendix).
6.3 Mechanism Validation: Odds Contraction
Figure 1 plots the stepwise odds ratio during Replay. RAPO exhibits persistent odds
contraction (mean: 0.41), whereas stationary baselines and deformation-off remain near 1.0,
validating Lemma 1. Across runs, measured OddsRatio correlates with RAG ( ρ= 0.87,
p <10−5), linking the mechanism metric (harmful-entry attenuation) to outcome suppression.
12

0 100 200 300 400
Timestep0
10
20
30
40Region (sorted: sensitive first)Sensitive
regions
Non-sensitive
regionsScar Field Evolution
0 1 2 3
Final scar
Hr
T0
10
20
30
40
0.00.51.01.52.02.5
Scar intensity Hr
tFigure 2:Scar persistence across phases.Total scar mass rises during Exposure due
to delayed-harm injection and persists through Decay into Replay, enabling replay-time
transition deformation and odds contraction.
6.4 Utility–Safety Trade-off
We evaluate whether RAPO suppresses replay by trivial shutdown. Figure 3 plots Replay
return vs. RAG across methods and RAPO parameter sweeps ( wH∈[0.5,4.0],η∈[0.01,0.1]).
Key findings.
•RAPO achieves a favorable trade-off: at RAG = 0.33, it retains 82% of baseline return (vs.
Shield at 65%).
•Increasing wHreduces RAG with diminishing returns beyond wH≈2.5, at which point
utility losses dominate.
•PM-ST achieves high return but no replay suppression, confirming that RAPO’s utility cost
is attributable to localized transition deformation rather than merely observing history.
Stagnation defense.RAPO suppression is localized: task activity (injection rate, ex-
ploration breadth) remains at 78–91% of baseline levels, and reach curves show sustained
propagation in non-sensitive regions (Appendix Figure A.3), distinguishing RAPO from
global shutdown.
7 Related Work
Safe RL in CMDPs (objective shaping).Constrained MDP methods enforce safety by
modifying the objective under fixed dynamics, including Lagrangian/primal–dual approaches
and CPO [Altman, 1999, Chow et al., 2017, Achiam et al., 2017]. These can learn to avoid
harm, including delayed costs, but under a stationary observable kernel they do not change
how the system responds to matched observable inputs at evaluation time. RAPO targets
the complementary issue captured by RSD: under observable-matched replay, structural
13

0.2 0.4 0.6 0.8 1.0 1.2
Re-amplification Gain (RAG) 
0.60.70.80.91.0Replay Return (normalized) 
82% utility
67% replay reduction96% utility
no suppression
65% utility
moderate suppressionFavorable
region
RAPO
RAPO (main)
Baselines
PM-ST
Shield
Figure 3:Utility–safety trade-off.Replay return (normalized) vs. re-amplification gain
(RAG). RAPO traces a Pareto-like curve as deformation strength varies, improving replay
suppression while retaining substantial utility, in contrast to stationary-transition baselines
and hard shielding.
suppression requires either a persistent replay-time action shift or a change in the observable
transition law (Theorem 1, Corollary 1).
Policy memory and partial observability.Recurrence, belief-state control, and history
encoders address hidden state and delay by changing the policy class [Kaelbling et al., 1998,
Hausknecht and Stone, 2015, Mnih et al., 2016]. However, if the observable kernel remains
stationary in ( x, a), policy memory alone cannot make the environment response history-
dependent under matched observables. Our PM-ST and PM-RNN controls separate these
effects by giving the policy the same history inputs/costs as RAPO while keeping transitions
atP 0.
Intervention-based safety: shielding and action restriction.Shields and reachability/control-
barrier style methods enforce safety by restricting actions [Alshiekh et al., 2018, Ames et al.,
2017, Berkenkamp et al., 2017]. They can prevent replay via persistent avoidance, but
may be conservative or require expensive online checks. RAPO instead implements a soft,
mass-preserving gating of next-state outcomes (transition reweighting), which can be localized
and bounded.
14

Delayed feedback and non-stationarity.Delayed credit assignment is typically handled
through eligibility traces and related methods (e.g., RUDDER) or auxiliary predictors [Sutton
and Barto, 2018, Arjona-Medina et al., 2019, Jaderberg et al., 2017], while non-stationary
RL often studies exogenous drift or adversarial change [Kirk et al., 2023, Pinto et al.,
2017]. RAPO introduces endogenous history-dependence relative to observables via persistent
harm memory, while remaining Markov on the augmented state; RSD isolates whether this
history-dependence yields replay suppression under frozen-policy evaluation.
Platform-mediated decision systems.Deployed systems commonly include mediation
layers that throttle exposure, reweight routes, or gate access based on incident logs [Chen
et al., 2019, Mao et al., 2016, Wang, 2020, Tao et al., 2019]. RAPO formalizes such mediation
as bounded, local, mass-preserving transition deformation driven by persistent harm traces.
8 Discussion and Limitations
8.1 Deployment Scope
When RAPO is applicable.RAPO assumes a platform can mediate the effective transi-
tion mechanism (routing, exposure, access) via a gating layer. This is natural in recommenders
(exposure throttling and eligibility filters), network routing (path reweighting), warehouses
(zone access control), and digital twins (runtime safety constraints). In unmediated physical
systems, RAPO should be interpreted as an external safety controller that can only act
through allowable intervention channels (e.g., action constraints or supervisory overrides).
Choosing the region map ρ.The partition ρ:X →{ 1, . . . , R} controls the bias–variance
trade-off of persistence. Fine partitions yield sparse, localized scars but require more data to
avoid noise; coarse partitions improve statistical stability but can cause spillover suppression
beyond the truly harmful region. Graphs admit node-level or community-level ρ; continuous
domains require discretization, learned clustering, or kernelized representations [Rasmussen
and Williams, 2006].
Delayed harm attribution.RAPO relies on an attribution rule that maps delayed
harm to regions (Section 4). If the attribution is misaligned (wrong region blamed), scars
can suppress the wrong transitions. Mitigations include multi-region attribution weights,
conservative thresholds, and logging/auditing of which regions received harm credit.
Proxy quality and persistence.Persistence amplifies proxy errors: if ˜ctis biased or noisy,
scars can entrench mistakes. We mitigate with (i) thresholded scarring ( τ), (ii) multi-signal
confirmation before increasing H, (iii) bounded injection, and (iv) audit trails that support
human review and rollback.
15

8.2 Key Trade-offs
Utility vs. safety.Stronger deformation (larger wHor faster scar growth η) reduces replay
metrics but can reduce task return by rerouting away from high-utility regions. We report
utility–safety curves (Replay return vs. RAG/AUC-R/SM-R) across ( wH, η, τ) to show that
suppression is localized rather than a trivial shutdown.
Irreversibility, recovery, and distribution shift.Irreversible scars capture deployments
where repeated incidents create lasting restrictions (e.g., persistent throttles). However,
under distribution shift, permanent scarring can cause over-suppression long after the system
has changed. The slow-decay variant Hr
t+1=δHr
t+ηmax (0, Gr
t−τ) with δ∈[0.95,0.999]
provides gradual recovery; operationally, this corresponds to time-limited throttles and
periodic re-evaluation.
Scaling beyond discrete graphs.Dense R-dimensional fields are intractable when |X|is
large. Practical approximations include (i) kernelized scars H(x) =P
iαik(x, x i) with sparse
dictionaries, (ii) learned scar networks x7→H (x) with capacity control and calibration, and
(iii) density models over harmful regions to produce a compact penalty field. To remain
deployable, deformation normalization Ztmust be computed locally (e.g., nearest neighbors
or constrained candidate sets), consistent with top-kand region-restricted ablations.
8.3 Broader Impact
Potential benefits.RAPO provides a mechanism for localized, persistent suppression in
platform-mediated systems with delayed harm, reducing repeated re-entry into historically
harmful pathways without requiring blanket avoidance.
Risks and misuse.Persistent gating can be used for censorship or exclusion, and biased
proxies can encode discrimination through scarring. Because scars persist, errors can be
harder to reverse than transient penalties.
Guardrails.Deployments should (i) audit harm proxies for demographic and topical bias,
(ii) provide recovery mechanisms (slow decay, manual override, or time-bounded scars), (iii)
log region-level attributions and gating decisions for review, and (iv) monitor both outcome
metrics (RAG/AUC-R/SM-R, return) and mechanism metrics (scar distributions and gating
rates), with explicit escalation procedures when anomalies appear.
9 Conclusion
We formalized replay—re-amplification when delayed harms, transient penalties, and sta-
tionary transitions combine—and introduced RSD to isolate it under controlled conditions.
RAPO suppresses replay via transition deformation: persistent harm-trace and scar fields
reduce reachability of historically harmful regions. Under policy-frozen RSD, RAPO achieves
67–83% replay reduction while retaining 78–91% task utility, with PM-ST and deformation-off
16

controls confirming that suppression requires environment-level memory. Theorem1 proves
stationary kernels cannot structurally suppress replay without action changes; Lemma 1
predicts and experiments validate odds contraction as the mechanism. Future work includes
scaling to continuous spaces, utility-safety optimization, and real deployment validation in
routing or allocation systems.
References
Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization.
InProceedings of the 34th International Conference on Machine Learning, 2017.
Mohammad Alshiekh, Roderick Bloem, R¨ udiger Ehlers, Robert K¨ onighofer, Scott Niekum,
and Ufuk Topcu. Safe reinforcement learning via shielding. InProceedings of the Thirty-
Second AAAI Conference on Artificial Intelligence, 2018.
Eitan Altman.Constrained Markov Decision Processes. Chapman and Hall/CRC, 1999.
Aaron D. Ames, Xingkang Xu, Jessy W. Grizzle, and Paulo Tabuada. Control barrier function
based quadratic programs for safety critical systems.IEEE Transactions on Automatic
Control, 62(8):3861–3876, 2017.
Jose Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes
Brandstetter, and Sepp Hochreiter. Rudder: Return decomposition for delayed rewards.
InAdvances in Neural Information Processing Systems, 2019.
Felix Berkenkamp, Matteo Turchetta, Angela P. Schoellig, and Andreas Krause. Safe model-
based reinforcement learning with stability guarantees. InAdvances in Neural Information
Processing Systems, 2017.
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H. Chi.
Top-K off-policy correction for a reinforce recommender system. InProceedings of the
Twelfth ACM International Conference on Web Search and Data Mining, 2019.
Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone. Risk-constrained
reinforcement learning with percentile risk criteria.Journal of Machine Learning Research,
18(167):1–51, 2017.
Matthew Hausknecht and Peter Stone. Deep recurrent Q-learning for partially observable
MDPs. InProceedings of the AAAI Fall Symposium on Sequential Decision Making for
Intelligent Agents, 2015.
Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z. Leibo,
David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary
tasks. InInternational Conference on Learning Representations, 2017.
Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting
in partially observable stochastic domains.Artificial Intelligence, 101(1–2):99–134, 1998.
17

Robert Kirk, Amy Zhang, Edward Grefenstette, and Tim Rockt¨ aschel. A survey of deep
reinforcement learning in non-stationary environments.arXiv preprint arXiv:2301.02804,
2023.
Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. Resource manage-
ment with deep reinforcement learning. InProceedings of the 15th ACM Workshop on Hot
Topics in Networks, 2016.
Volodymyr Mnih, Adri` a Puigdom` enech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap,
Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep
reinforcement learning. InProceedings of the 33rd International Conference on Machine
Learning, 2016.
Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial
reinforcement learning. InProceedings of the 34th International Conference on Machine
Learning, 2017.
Carl Edward Rasmussen and Christopher K. I. Williams.Gaussian Processes for Machine
Learning. MIT Press, 2006.
Richard S. Sutton and Andrew G. Barto.Reinforcement Learning: An Introduction. MIT
Press, 2 edition, 2018.
Fei Tao, Meng Zhang, Yu Liu, and Andrew Y. C. Nee. Digital twin driven smart manufacturing.
Journal of Manufacturing Systems, 54:1–12, 2019. Often cited as 2018 online/early access;
use journal year as final.
TODO: verify authors/title/venue Wang. Adaptive control for warehouse operations with
reinforcement learning. InTODO, 2020. Placeholder: please verify this citation (ti-
tle/authors/venue) before submission.
18