# Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning

**Authors**: Edward Y. Chang, Ethan Y. Chang

**Published**: 2025-10-06 04:52:17

**PDF URL**: [http://arxiv.org/pdf/2510.04488v1](http://arxiv.org/pdf/2510.04488v1)

## Abstract
Multi-agent debate often wastes compute by using a fixed adversarial stance,
aggregating without deliberation, or stopping on heuristics. We introduce MACI,
an active controller with two independent dials that decouple information from
behavior: an information dial that gates evidence by quality, and a behavior
dial that schedules contentiousness from exploration to consolidation. A
moderator tracks disagreement, overlap, evidence quality, and argument quality,
and halts when gains plateau. We provide theory-lite guarantees for
nonincreasing dispersion and provable termination, with a budget-feasible
scheduler. Across clinical diagnosis and news-bias tasks, MACI improves
accuracy and calibration while reducing tokens, and converts residual
uncertainty into precision RAG plans that specify what to retrieve next. We use
a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal,
validated for order invariance and judge-swap stability; stability depends on
using high-capability judges. MACI turns debate into a budget-aware,
measurable, and provably terminating controller.

## Full Text


<!-- PDF content starts -->

Multi-Agent Collaborative Intelligence: Dual-Dial Control for
Reliable LLM Reasoning
Edward Y. Chang, Ethan Y. Chang
Stanford University, UIUC
Abstract
Multi-agent debate often wastes compute by
using a fixed adversarial stance, aggregating
without deliberation, or stopping on heuris-
tics. We introduce MACI, an active controller
with two independent dials that decouple in-
formation from behavior: an information dial
that gates evidence by quality, and a behav-
ior dial that schedules contentiousness from
exploration to consolidation. A moderator
tracks disagreement, overlap, evidence qual-
ity, and argument quality, and halts when
gains plateau. We provide theory-lite guar-
antees for nonincreasing dispersion and prov-
abletermination, withabudget-feasiblesched-
uler. Across clinical diagnosis and news-bias
tasks, MACIimproves accuracy and calibra-
tion while reducing tokens, and converts resid-
ual uncertainty into precision RAG plans that
specify what to retrieve next. We use a cross-
family LLM judge ( CRIT) as a conservative
soft weight and stop signal, validated for or-
der invariance and judge-swap stability; sta-
bility depends on using high-capability judges.
MACIturns debate into a budget-aware, mea-
surable, and provably terminating controller.
1 Introduction
Despite intense interest, multi-agent debate has seen
limited progress in theory or practice. Unmoderated
or ad hoc collaboration lacks provable termination,
calibrated uncertainty, and cost control [Cemri et al.,
2025]. Common approaches use a fixed adversarial
stance [Liang et al., 2024], aggregate without delibera-
tion [Wang et al., 2023], or stop on heuristics, which
wastes compute and locks in early errors.
The control problem.Reliable multi-agent reason-
ing requiresactive modulation. Early rounds shouldexplore alternatives, and later rounds should consol-
idate on well-supported conclusions. This demands
two independent controls—what information enters
(evidence gating) and how agents interact (behavioral
stance)—together with principled stopping that detects
convergence.
Beyond single-dial modulation.Prior work intro-
duced contentiousness modulation [Chang, 2023a,b],
showing that behavior matters, but a single dial is
insufficient. Systems need concurrent control over in-
formation quality and interaction intensity, and they
must stop when additional debate yields diminishing
returns. The CRITframework [Chang, 2023a] evalu-
ates arguments but has not been coupled to systematic
stopping.
MACI: Dual dials with measured stopping.We
present MACI(Multi-AgentCollaborativeIntelligence),
an active controller with two orthogonal dials. The
informationdial ( τ) gates evidence by quality, and
thebehaviordial ( CL) schedules contentiousness from
exploration to consolidation. A moderator tracks four
signals—disagreement ( DJS), overlap (O)1, evidence
quality (Q), and argument quality ( CRIT)—and halts
when gains plateau. Beyond prediction, MACItrans-
lates residual uncertainty into precision RAG: targeted
acquisition plans that specify what to retrieve next.
Guarantees and scheduler.Under gated averaging
with smoothing, dispersion is nonincreasing and the
plateau test terminates in O(1/ε)rounds, improving
toO(log(1/ε))under a contraction condition (Lem. 1,
Prop. 2, Appx. M). A budget-feasible UCB scheduler
over(∆ α,∆γ)achieves ˜O(√
KT)no-regret with zero
expected budget violation (Prop. 1, Appx. L).
Empirical highlights.On clinical diagnosis with
1,500 cases spanning100 +diseases, MACIimproves
Acc@1 by+3.9ppover majority vote and+3.7pp
1Overlap is not synonymous with agreement; it measures
convergence ofevidence, whereas disagreement measures
convergence ofbeliefs. We require both.arXiv:2510.04488v1  [cs.AI]  6 Oct 2025

Table 1: Multi-agent orchestration methods: architectural comparison.
Method Info Gate Behavior Ctrl Stopping Signals Notes
Aggregation methods
Majority Vote Choi et al. [2025a]✗ ✗Fixed (N) – Passive aggregation
Self-Consistency Wang et al. [2023]✗ ✗Fixed (N) – Fixed sampling
Mixture-of-Agents Wang et al. [2024b]✗ ✗Heuristic – Layered ensemble
Fixed-stance debate
Liang et al. Liang et al. [2024]✗Fixed Heuristic – Fixed adversarial roles
Diverse MAD Liu et al. [2025]✗Varied Heuristic – Prompting style variation
Multi-LLM Debate Estornell and Liu [2024]✗Fixed Heuristic – Ad hoc termination
Adaptive orchestration
MACI (ours)✓Scheduled Plateau 4 Dual-dial control
over fixed-contentiousness debate, reduces calibration
error (ECE0.081vs. 0.103), and uses19% fewer
generation tokens. A cross-domain study on news bias
(619 articles) shows portability: the same controls nar-
row the partisan gap by68%without domain tuning.
MACIalso translates residual uncertainty intopreci-
sion RAGplans that specify what to retrieve next to
improve decision quality.
Contributions.We operationalize these ideas into
five contributions:
1.Dual control.Two-dial orchestration with
information-theoretic stopping, where D(t)is nonin-
creasing and plateau tests terminate within bounded
rounds (Lem. 1, Prop. 2).
2.Judge robustness. CRITis order invariant and stable
under cross-family judge swaps (Appx. E.7; n=100,
2–3%winnerflips; α=0.68); weuseitasasoftweight
and part of a compound stop rule, not a hard oracle.
3.Targeted acquisition.Systematic collaboration yields
predictions together with precision RAG plans for
underspecified problems.
4.Empirical validation.Improvements in accuracy,
calibration, and efficiency across two domains, with
complete ablations and transcripts.
Evaluator reliability.We rely on CRIT, a cross-
family judge that scores evidence-, logic-, and relevance-
grounded argument quality. We validate order in-
variance and judge-swap stability ( n=100; 2–3% win-
ner flips) and report tentatively acceptable agreement
(Krippendorff’s α=0.68). We therefore use CRITas a
soft aggregation weight and as part of a compound stop
rule, rather than as a hard oracle. Stability holds with
high-capability judges; weaker or poorly aligned judges
may require larger K, tighter thresholds, or periodic
human audits (Appx. E.7).
2 Related Work
Multi-agent debate and orchestration.Debate-
style prompting aims to surface errors through agentopposition [Irving et al., 2018], yet implementations
often collapse to majority opinions or underperform
strong single models [Wang et al., 2024b, Smit et al.,
2024, Estornell and Liu, 2024, Choi et al., 2025b]. For
example, Liang et al. [Liang et al., 2024] maintain fixed
adversarial intensity throughout, wasting tokens when
contention becomes unproductive. Diverse Multi-Agent
Debate [Liu et al., 2025] varies agent prompting styles
but lacks behavioral adaptation within debates. Recent
audits reveal that multi-agent systems frequently fail
without systematic orchestration [Cemri et al., 2025],
motivating principled control mechanisms.
Table 1 reveals a critical gap: existing methods pro-
vide either no control or single-axis control (fixed/var-
ied debate intensity), but none combine independent
information gating with scheduled behavioral transi-
tions.Aggregation methodssample diverse responses
but aggregate without deliberation, causing perfor-
mance degradation as answer spaces grow [Choi et al.,
2025b].Fixed-stance debatemethods maintain constant
intensity, unable to transition from exploratory dis-
agreement to consolidative synthesis. MACIaddresses
both limitations: an information dial filters evidence by
quality while a behavior dial schedules contentiousness
from high to low, with information-theoretic stopping
that detects convergence plateaus rather than relying
on heuristic round limits.To our knowledge, no
prior work provides independent controls over
information quality and behavioral stance.
Computational paradigms and capability trade-
offs.Transformers [Vaswani et al., 2017] required
orders of magnitude more computation than RNNs
but unlocked scaling laws [Kaplan et al., 2020]; large
languagemodels[Brownetal.,2020]continuethistrend.
MACIfollows this paradigm: systematic orchestration
requires overhead, but provides measurable gains in
accuracy, calibration, and epistemic humility. For high-
stakes applications, reliability justifies cost.
Ensemble methods and aggregation.Self-
Consistency [Wang et al., 2023] and majority voting
aggregate sampled chains with fixed behavior. Mixture-2

of-Agents [Wang et al., 2024a, Li et al., 2025] improves
performance via layered ensembles but under-specifies
convergence beyond heuristic rounds. These meth-
ods assume weak error correlation and rarely specify
stopping criteria. On short-horizon multiple-choice
tasks, voting can dominate when answer spaces are
small [Choi et al., 2025b], but performance degrades
as dimensionality grows (Appendix K). MACI differs
by using information-theoretic signals (disagreement,
overlap, evidence quality) to drive active deliberation
with plateau-based stopping.
Control and stopping in AI systems.Control-
theoretic approaches in reinforcement learning use mea-
sured feedback to adjust policy parameters [Recht,
2019]. Active learning frameworks gate information
acquisition based on uncertainty [Settles, 2009]. Our
work extends these ideas to multi-agent reasoning: the
moderator tracks four signals and adjusts behavioral
stance when progress plateaus, analogous to adaptive
control in dynamical systems. Prior debate work lacks
this closed-loop structure, relying instead on fixed in-
teraction patterns or ad hoc stopping rules.
Initialization and few-shot learning.Bayesian
[Xie et al., 2022], meta-learning [Dai et al., 2023], and
optimization-as-inference [von Oswald et al., 2023] per-
spectives explain in-context learning. Selection and
weighting strategies adapt which demonstrations mat-
ter [Hong et al., 2025, Wang et al., 2024c]. We use
lightweight information-theoretic metrics (pattern den-
sity, prior-target mismatch) to initialize debate param-
eters. Appendix F validates these heuristics on con-
trolled tasks; broader ICL theory appears in concurrent
work.
Evaluation with LLM judges.Independent LLM
judges are appealing, but evaluators can exhibit self-
preference bias, especially when sharing a family with
the debaters [Panickssery et al., 2024]. Moreover, re-
cent scalable-oversight studies report that weak-judge
regimes yield inconsistent gains [Kenton et al., 2024].
Compounding this,LLMs lack robust self-validation for
constraint adherence and reasoning consistency [Bom-
masani et al., 2022, Chang, 2025], a limitation loosely
analogous to Gödel-style barriers [Gödel, 1967] and
supported empirically [Hong et al., 2024]. Hence we
use cross-family judges, mask agent identity, require
span-grounded justifications, and aggregate small judge
panels ( K≥3) with reliability weighting; judge-swap
and order-swap checks, plus a CRIT-free controller vari-
ant, indicate that stopping and weights are not driven
by a single evaluator.3 MACI: Information and Behavior
Dials with Measurable Stopping
Goal.Coordinate multiple LLM agents to explore
meaningfully and then consolidate on well-supported
conclusions with calibrated confidence under a fixed
token budget. All notation is defined in Appx. D.
Failures we address.
•Prior capture:Single models fixate on high-
frequency patterns and confirm early impressions.
•Majority illusion:Voting appears strong in multiple
choice, but does not test the depth of reasoning or
the quality of evidence.
•Unregulated debate:Contentiousness drifts, evidence
quality is not gated, and stopping is ad hoc.
•Cost sprawl:Long chains consume budget without
proportional gains.
Design summary. MACIexposes two independent
controls. Theinformation dial( τQ)gates which evi-
dence agents may cite by enforcing a quality threshold.
Thebehavior dial( CL) sets contentiousness, instruct-
ing agents how aggressively to challenge opponents. A
moderator tracks four signals—target-aligned evidence
quality (Q), inter-agent disagreement ( DJS), support
overlap (O), and argument quality ( CRIT)—and stops
when they plateau jointly. The construction of the
target prototype θused byQis detailed in Appx. D.1.
3.1 The Control Loop
Algorithm 1 (in Appx. C) shows the three-phase cy-
cle. Agents receive a composite anchor A(t)= 
ψ(t), CL(t), τ(t)
Q, τ(t)
CRIT
that contains the history of
curated debates, current contentiousness instruction,
and evidence quality thresholds. Each agent pro-
duces claim-justification pairs with probability distri-
bution p(t)
uover the outcomes and evidence cited spans
ˆE(t)
u. The moderator scores arguments withCRIT
(Appendix E), computes evidence quality Qfor the
cited spans, and admits only those satisfying both
gates, Q≥τQ(t)and CRIT≥τ CRIT(t)(defaults in
Appx. B.1); it then computes all signals and updates
both dials.
3.2 Four Measurement Signals
Notation.Let Ybe finite outcome space,∆( Y)the
probability simplex. Two agents u∈ {A, B} alternate
over rounds t= 1,2, . . .. Each outputs p(t)
u∈∆(Y)and
cites evidence spans ˆE(t)
u.3

Signal 1: Evidence quality (Q).Measures align-
ment between cited evidence and target distribution
using encoder embeddings:
Q(t)= cos( ¯e(t),θ), ¯e(t)=1
|E(t)|X
e∈E(t)ϕ(e)
∥ϕ(e)∥,(1)
where E(t)collects the spans fromadmittedarguments
in round t, andθis the target prototype. RisingQ(t)
indicates convergence on task-relevant evidence.
Admission rule.For eachcandidateargument iat
round t, we compute a per-argument evidence score
q(t)
i(analogous to (1), using only the spans cited by i).
The moderator admitsiiff it passes thedual gate:
q(t)
i≥τQ(t)andCRIT(t)
i≥τCRIT(t).
After admission, E(t)aggregates all admitted spans and
Q(t)summarizes the evidence quality of that batch.
Default settings for τQandτCRITappear in Appx. B.1.
Signal 2: Disagreement ( DJS).Jensen-Shannon
divergence quantifies distributional disagreement:
D(t)
JS=1
2KL(p(t)
A∥m(t)) +1
2KL(p(t)
B∥m(t)),
m(t)=1
2(p(t)
A+p(t)
B).(2)
DJSis symmetric and bounded in[0 ,1]when using
log base2, and remains finite under disjoint support.
Falling DJStracks convergence; plateaus indicate ex-
hausted productive disagreement (thresholds are se-
lected on the development set; Appendix D).
Signal 3: Support overlap (O).Jaccard similarity
of cited evidence sets:
O(t)=|ˆE(t)
A∩ˆE(t)
B|
|ˆE(t)
A∪ˆE(t)
B|.(3)
Rising overlap indicates agents grounding arguments
in shared evidence rather than talking past each other.
Combined with falling DJS, this signals genuine con-
vergence rather than forced agreement.
Signal 4: Argument quality ( CRIT).Cross-
family LLM evaluator scores each argument on [0,1]
for logical coherence, evidence support, and reason-
ing depth (Appendix E). The moderator maintains
exponential moving average of per-agent quality:
Γ(t)
u=λΓ(t−1)
u + (1−λ) CRIT u,t,
ω(t)
u=Γ(t)
u+ϵ
Γ(t)
A+ Γ(t)
B+ 2ϵ,(4)
where ω(t)
uweights agent u’s beliefs. Rising CRITin-
dicates improving argument quality; agents producing
consistently weak arguments receive less weight in the
mixture.3.3Dual-Dial Control with Plateau Detection
Information gain.Normalized entropy reduction of
reliability-weighted mixture:
ˆI(t)=H(p(t−1))−H(p(t))
log|Y|
+, p(t)=X
uω(t)
up(t)
u.(5)
Falling ˆIindicates lower returns from continued debate.
Plateau detection via relative progress ratios.
Rather than fixed thresholds, we measure progress
relative to remaining capacity:
r(t)
I=MAw(ˆI(t))
MAw(H(p(t−1))/log|Y|) +η,
r(t)
D=MAw([D(t−1)
JS−D(t)
JS]+)
MAw(D(t−1)
JS) +η,(6)
whereMAwis length- wmoving average, η= 10−3pre-
vents division by zero. Binary flags trigger when ratios
fall below thresholds forτ stopconsecutive rounds:
π(t)
I=⊮[r(t)
I< εrel
I], π(t)
D=⊮[r(t)
D< εrel
D].(7)
Behavior dial: scheduled contentiousness.
When plateaus detected, reduce contentiousness:
CL(t+1)= [CL(t)−αIπ(t)
I−αDπ(t)
D]+.(8)
This transitions agents from exploratory ( CL= 0.9:
“challenge aggressively”) to consolidative ( CL= 0.5:
“build on agreements”).
Information dial: raising evidence standards.
As consensus forms, admit increasingly higher-quality
arguments:
τ(t+1)= min
τ(t)+γ π(t)
I, τmax1	
,τ(t)= 
τ(t)
Q, τ(t)
CRIT
,
(9)
where the minimum is applied element-wise, and1 ∈
Rmis the all-ones vector formgates (herem=2).
Stoppingcriterion.Haltwhenbothplateauspersist
and evidence quality is sufficient:
τstop−1X
s=0π(t−s)
I =τstop,τstop−1X
s=0π(t−s)
D =τstop,
Q(t)≥τ(t)
Q,O(t)≥τO.(10)
This ensures that the gains have genuinely plateaued,
not just temporarily stalled.
Learned scheduler (summary).We replace the
fixed( α, γ)schedule with a tiny bandit policy over
a discrete action set of(∆ α,∆γ)moves, optimizing
per-round information gain and dispersion drop un-
der a token budget. Proposition 1 (Appx. L) shows
˜O(√
KT)no-regret versus the best fixed schedule with
zero expected budget violation.4

Plateau guarantees (summary).Under a gated
averaging update with bounded noise, dispersion D(t)
is non-increasing and the plateau controller terminates
inO(1/ε)expected rounds. With a contraction con-
dition the bound becomes O(log(1/ε)). See Lemma 1
and Proposition 2 (Appx. M).
On dispersion.Our guarantees analyze dispersion as
the average KLto the running mixture, DKL(t) =
1
mP
iKL(p(i)
t∥¯pt), which yields monotonicity under
gated averaging by convexity; in plots we addition-
ally report JSD, which is locally Lipschitz-equivalent
on theδ-smoothed simplex (Appx. M).
3.4 Initialization Strategy
Before debate, we compute initial evidence qualityQ
and disagreement DJSfrom the query and retrieved
context. We initialize contentiousness as
CL(1)=

0.7ifQ≥τhi
QandD JS≤τlo
DJS,
0.9ifQ≤τlo
QandD JS≥τhi
DJS,
0.8otherwise.
The thresholds τlo
Q, τhi
Q, τlo
DJS, τhi
DJSare set by percentiles
on the development set (Appendix F). This reduces
wasted exploration on straightforward cases while pre-
serving search capacity on ambiguous ones.
3.5 Complete Algorithm
Algorithm 1 integrates all components (see Appx. C
for complete pseudocode).
3.6 Problem regime classification
MACItargets open-ended reasoning where (1) the an-
swer space is large ( |Y| ≥ 20), (2) evidence synthesis is
required rather than pattern matching, and (3) uncer-
tainty quantification matters. For small multiple-choice
tasks ( |Y| ≤ 5), majority voting typically suffices. Ap-
pendix K.1 formalizes this distinction.
4 Empirical Validation
We evaluate MACIin two domains, clinical diagnosis
(G1) and news bias detection (G2), measuring accuracy,
calibration, and cost, with ablations and case studies.
4.1 G1: Clinical Diagnosis
Dataset and protocol. Kaggle Symptom–
Disease[Tharmalingam, 2023]: 1,500 cases across
30 diseases. The corpus provides only current symp-
toms (no labs or history). We use theunconstrained
setting: models output free-text diagnoses from100 +Table 2:Pairwise MACIvs. single models on 1,500 cases.
Stars: paired Wilcoxon vs. best single (BH FDR q=0.05):
∗∗∗p<0.001,∗∗p<0.01.
Model Acc@1 Acc@3 MRR
GPT-4o0.734 0.847 0.781
Claude 3 Opus0.720 0.832 0.768
Gemini 1.5 Pro0.693 0.825 0.756
GPT+Claude0.786∗∗∗0.874∗∗∗0.823∗∗∗
GPT+Gemini0.751∗∗0.861∗∗0.807∗∗
Claude+Gemini0.745∗∗∗0.858∗∗∗0.798∗∗∗
possibilities (no candidate list), matching clinical triage
with broad differentials.
Why voting fails here.When predictions are con-
strained to 30 labels, accuracy reaches 95–97%. In the
unconstrained regime, agreement collapses and Acc@1
falls to 73–78% because many cases are under-specified
and require evidence synthesis rather than pattern
matching. Majority voting is unreliable in large output
spaces (Appx. K).
Setup. Backbones:GPT-4o, Claude 3 Opus, Gem-
ini 1.5 Pro; all pairs ( M=2).Retrieval:BM25
+ dense reranking ( kret=32).Baselines:best sin-
gle; majority vote (20); self-consistency (20); fixed-
contentiousness debate [Liang et al., 2024].Statistics:
BCa 95% CIs from B=20bootstraps; paired Wilcoxon
with Benjamini–Hochberg FDR (q=0.05).
4.1.1 Main Results, Three Parts
Results #1: MACIvs. single models.Every
MACIpair outperforms its best single (Table 2); for ex-
ample, GPT–4o+Claude improves Acc@1 by +5.2pp.
Results #2: MACIablation. MACIrelies on three
active controls: (i) behavioral scheduling (high CL
for exploration, low CLfor consolidation), (ii)dual
gating( Q≥τQand CRIT≥τ CRIT), and (iii) infor-
mation–theoretic stopping (plateaus in DJSand IG).
Fixed–contentiousness debate wastes tokens; passive
aggregation uses 5–7 ×more tokens and underperforms.
Table 3 shows orthogonal effects: removing theQgate
costs−5.2pp Acc@1 and raises ECE by+0 .040; uni-
form weights (no reliability) cost −3.0pp and+0 .013
ECE; disabling scheduling (holding CLhigh/low) costs
−3.9to−6.0pp with larger ECE.5

Table 3:Ablations on clinical diagnosis.∆is the drop
relative to full MACI(lower is better). (Uniform weights
= no reliability weighting)
Variant Acc@1∆ECE∆
MACI(full) 77.8 – 0.081 –
No schedule (high) 73.9 3.9 0.112 0.031
No schedule (low) 71.8 6.0 0.099 0.018
NoQgate 72.6 5.2 0.121 0.040
Uniform weights 74.8 3.0 0.094 0.013
Results #3: MACIvs. competing schemes.
Against debate/aggregation baselines, MACIis more
accurate and better calibrated with fewer tokens (Ta-
ble 4): +3.9pp over majority vote and +3.7pp over
Liang et al. (both p<0.001), ECE0 .081vs.0 .103, and
19%fewer generation tokens. These gains align with
the ablation evidence.
Computational overhead. MACIuses 18.4k to-
kens vs. 6.8k for a single model (2.7 ×overhead). For
high–stakes settings where accuracy reduces costly er-
rors, this overhead is justified; for routine tasks, sin-
gle–model inference may suffice.
Evidence quality stratification.Accuracy corre-
lates with initialQ: cases withQ >0.7reach 78.3% vs.
69.8% forQ <0.3(Table 20, Appx. K). This validatesQ
as a useful signal and suggests retrieval improvements
may benefitMACImore than parameter tuning.
Controller guarantee (theory–lite).With
bounded noise and gated averaging, dispersion D(t)is
non–increasing, and the plateau controller terminates
inO(1/ε)expected rounds; under a contraction, D(t)
decays geometrically and termination is O(log(1/ε))
(Lem. 1, Prop. 2, Cor. 1, Appx. M).
4.1.2 Two Case Studies: Active Control and
Information Acquisition
We illustrate MACI’s scheduling and gating, and show
termination aligns with plateaued IG and low dis-
persion. Each case ends with a targeted informa-
tion–acquisition plan specifyingwhat to fetch nextfor
retrieval–augmented generation. Full transcripts ap-
pear in Appx. G and H.
Case 1: Dengue vs. Chikungunya (3 rounds).
Symptoms:skin rash, joint pain, vomiting, fatigue,
high fever, headache, retro–orbital pain, myalgia, red
spots.Ground truth:Dengue Fever.
Table 5 shows scheduling CL∈ { 0.9,0.7,0.5}.R1
(CL=0.9): GPT–4o proposes specific viral diagnoses
(Dengue 60%, Chikungunya 25%, Zika 15%); Geministays categorical (Viral Infection 60%, Autoimmune
20%, Bacterial 15%). High disagreement ( DJS=0.56).
R2( CL=0.7): targeted challenges shift Gemini to
Dengue (50%) and Chikungunya (30%); DJS=0.37,
CRIT0.75→0.82.R3( CL=0.5): distributions align
(Dengue 60%, Chikungunya 35%, Zika 5%); DJS=0.02,
MI plateaus (0.49); trigger stop flags.
Figure 1 (internal signals).Fig. 1 shows entropy
declining as CLmoves from explore to consolidate;Q
rising as the gate τtightens; and CRITincreasing as
low–quality arguments are filtered. Dispersion DJS
(not shown) falls in parallel; termination matches IG
and dispersion plateaus.
Evaluator robustness.Reversing debater order
leaves the CRITwinner unchanged; cross–family judge
swaps preserve item–level rankings (Appx. E.7).
Informative RAG recommendations.Beyond
“Dengue Fever (60%),” MACIoutputs a plan:Ques-
tions(endemic travel; bleeding/bruising),Labs(CBC
for thrombocytopenia; IgM/IgG; PCR; NS1 antigen),
plus retrieval cues for RAG.
Case 2: Hepatitis differential (4 rounds, adap-
tive stopping). Symptoms:itching, vomiting, fa-
tigue, weight loss, high fever, jaundice, dark urine,
abdominal pain.Ground truth:Hepatitis C.
Table 6 shows CL∈ { 0.9,0.7,0.5,0.3}.R1( CL=0.9):
GPT–4o emphasizes HCV (40%); Claude emphasizes
HBV (35%) and alcoholic hepatitis (15%); DJS=0.22.
R2(CL=0.7): reasoning integration; DJS=0.12,CRIT
rises.R3( CL=0.5): near convergence (HCV 35%,
HBV 25–30%, cirrhosis 20%);D JS=0.04, MI 0.49.R4
(CL=0.3): distributions nearly identical; termination.
Table 6:Signal evolution: Hepatitis case (4 rounds).
RoundCL fW1MICRITD JS
1 0.9 0.52 0.39 0.76 0.22
2 0.7 0.45 0.41 0.83 0.12
3 0.5 0.33 0.49 0.89 0.04
4 0.3 0.31 0.49 0.92 0.03
Change−40% +26% +21%−86%
Internal signals (case view).As in Case 1, entropy
declines as CLschedules down;Qand CRITrise with
tighter gates; DJSfalls toward zero. Fig. 2 visualizes
monotonic dispersion declines for both cases.
Evaluator robustness.The same checks apply: or-
der swap and cross–family judge swap leave outcomes
stable (Appx. E.7).6

Table 4:Main results (1,500 cases).MACIimproves accuracy and calibration with fewer tokens and rounds.
Method Acc@1 MRR ECE Brier Tokens (103) Rounds
Single best 72.1 0.802 0.126 0.187 6.8 1.0
Majority vote (20) 73.9 0.817 0.139 0.196 92.3 1.0
Self-consistency (20) 74.7 0.827 0.147 0.201 128.6 1.0
Liang et al. [Liang et al., 2024] 74.1 0.834 0.103 0.173 22.6 3.3
MACI(ours) 77.8 0.856 0.081 0.158 18.4 2.7
(a) Entropy convergence.
 (b) Evidence quality (Q) improves.
 (c) Argument quality (CRIT) improves.
Figure 1:Internal signals over debate rounds (clinical).Left:entropy declines under scheduled CL.Middle:evidence
qualityQrises as the evidence gate τQtightens.Right:argument quality CRITrises as low-quality arguments are filtered.
Termination coincides with plateaued IG and low dispersion (not shown).
Table 5:Signal evolution on a Dengue case. fW1:
Wasserstein-1 distance; MI: mutual information; DJS:
Jensen–Shannon divergence. Lower is better for fW1and
DJS; higher is better forMI. (Appx D)
RoundCL fW1 MID JS
1 0.9 0.68 0.43 0.56
2 0.7 0.44 0.46 0.37
3 0.5 0.36 0.49 0.02
Change (R1→R3)−47% +14%−96%
Informative RAG recommendations. MACIout-
puts:Labs(HAV/HBV/HCV serology; LFTs AL-
T/AST/bilirubin/alkaline phosphatase; CBC; INR),
Imaging(abdominal ultrasound),History(alco-
hol/medications, injection drug use, transfusions),
Monitoring(encephalopathy, coagulopathy).
4.2G2: News Bias Detection (Cross–Domain)
Why this tests portability.News bias differs from
G1 onevidence(full articles vs. symptom lists),labels
(5–point continuous vs. discrete), andground truth
(partisan–anchored vs. objective). A self–supervised
study maps CLto linguistic patterns across topics
(Table15, Appx.I):high CL=0.9isconfrontational, low
CL=0.3is cooperative. A shallow classifier predicts CL
bins with AUC0 .83(Table 17), indicating measurable
behavior modulation beyond domain content.Dataset and setup.We use 619 scandal articles
from [Budak et al., 2016] with Democrat/Republican
five–point ratings ( −2to+2). Hyperparameters match
G1:CL∈ { 0.9,0.7,0.5}, gates ( τQ,τO), and the same
stopping rule. Two agents (GPT–4o, Claude 3 Opus)
start with asymmetric partisan framings.
Table 7:News bias results (619 articles).Tokens are re-
ported in thousands( ×103). MAD: mean absolute deviation
from centrist (0).
Method MAD ECE Brier Tokens Rounds
D–annotator 0.84 0.168 0.221 – –
R–annotator 0.76 0.173 0.218 – –
Average (D+R) 0.51 0.142 0.189 – –
Unscheduled 0.38 0.127 0.178 11.7 3.8
MACI0.12 0.093 0.167 8.2 2.4
Results #1: Narrowing partisan gaps.Partisan
annotators diverge (MAD 0.84 vs. 0.76). Averaging
yields 0.51; unscheduled debate 0.38. MACIachieves
0.12MAD (76%vs. averaging,68%vs. unscheduled),
with better calibration (ECE 0.093 vs. 0.127–0.173)
and30%fewer tokens (8.2 vs. 11.7). Fig. 3a shows
centering between partisan peaks.
Results #2: Which controls matter (ablations).
Removing scheduling raises MAD by +0.14–0.19; re-
moving theQgate by +0.17; uniform (no–reliability)
weights by +0.06 and ECE by +0.014 (Table 8).7

Figure 2: Convergence in two cases:D JSand allied distances decrease monotonically under scheduling.
Table 8:Ablations. Positive∆MAD indicates worse bias.
Variant MAD∆ECE∆
MACI(full) 0.12 – 0.093 –
No schedule (high) 0.31 0.19 0.134 0.041
No schedule (low) 0.26 0.14 0.118 0.025
NoQgate 0.29 0.17 0.141 0.048
Uniform weights 0.18 0.06 0.107 0.014
Figure 3:Convergence signals during bias mitigation (news
bias). Wasserstein distance falls, normalized MI rises and
then plateaus, and cross-entropy declines. Debates stop
when dispersion and information gains plateau, mirroring
the clinical setting (Appx. J.3).
Results #3: Signal convergence and behavioral
transfer.Debates average2 .4rounds and stop when
dispersion and information gains plateau. Convergence
traces mirror G1; see Appx. J.3 for signal plots.
Example (one case).On “Christopher Columbus,”
one agent begins Euro–centric (expansion and trade)
and the other foregrounds indigenous harms. Over four
rounds the former acknowledges severe consequences
within broader colonial policy and recommends bal-
anced presentation with citations (Table 20, Appx. J.3).
Portability.Same code and hyperparameters, no do-
main tuning. Controls ( CRITandQgates, CLschedul-ing,DJSandOstopping) transfer across evidence types
and label structures. Per–article distributions appear
in Appx. J.3.
Budget-aware learned scheduler.We also pro-
vide a drop–in scheduler that replaces the fixed( α, γ)
schedulewith asmall banditover(∆ α,∆γ)moves, opti-
mizing per–round IG gains and dispersion drops under
a token budget. The policy enjoys ˜O(√
KT)no–regret
against the best fixed schedule with zero expected bud-
get violation (Prop. 1, Appx. L). We keep the fixed
schedule in G1/G2 main tables for comparability.
5 Conclusion
Advancing multi-agent orchestration.We pre-
sented MACI, a control framework that extends con-
tentiousness modulation Chang [2023a,b] with (1)dual-
dialcontrol separating information admission from be-
havioral stance, (2)information-theoreticstopping via
relative progress ratios, and (3)adaptiveinitialization
to reduce wasted exploration. Unlike passive aggrega-
tion or fixed debate, MACIsteers deliberation using
measurable signals—disagreement, overlap, evidence
quality, and argument quality—and halts when gains
plateau.
On clinical diagnosis (1,500 cases), MACIimproves
accuracy (+3.9pp over majority vote; +3.7pp over
fixed-contentiousness) and calibration (ECE0 .081
vs.0.103) while using19%fewer generation tokens;
case studies show convergence (disagreement ↓86–96%,
CRIT↑21–26%). CRITis order-invariant and stable
under cross-family judge swaps (2–3% winner flips;
Appx. E.7). Theory-lite guarantees bound dispersion
and ensure termination in O(1/ε)rounds (improving
toO(log(1/ε))), and a budget-feasible UCB scheduler
attains ˜O(√
KT)no-regret with zero expected budget
violation (Appx. M, L). News-bias validation (619 arti-
cles) shows portability: the same controls reduce the
partisan gap by68%without domain tuning.8

Limitations and directions.
•Scope.We evaluate two domains with subjective
judgment; broader testing on tasks with objective
ground truth (math, code, factual QA) and longer
chains remains future work. Retrieval quality medi-
ates performance (Sec. 4.1), suggesting gains from
stronger evidence acquisition.
•LLM-judge dependence. CRITuses cross-family
evaluators and may inherit bias. We mitigate via
identity masking, span-grounded justifications, and
judge-swap checks (2–3% flips; Appx. E.7). Lower-
capability regimes may require larger K, tighter stop
tolerances, and periodic human audits.
•Cost trade-offs. MACIuses2 .7×tokens vs. single-
model inference; while accuracy gains justify over-
head for complex cases, domain-specific cost–benefit
studies are needed.
•Hyperparameters. CLschedules, gates, and
plateau tolerances are tuned on dev data; generaliza-
tion beyond diagnosis/news is open. Scaling to 3–5
agents and meta-learned initialization are natural
next steps.
Closed-loop RAG.A promising direction is coupling
MACI’s acquisition plans to budgeted RAG with end-
to-end evaluation; transcripts (Appx. G, H) enable
auditability.
Overall, MACIreframes multi-agent debate as a con-
trollable, auditable, andprovably terminatingcon-
troller, turning orchestration from heuristics into a
budget-aware, measurable process.
References
R.Bommasani, D.A.Hudson, E.Adeli, R.Altman, and
more. On the opportunities and risks of foundation
models, 2022. URL https://arxiv.org/abs/2108.
07258.
T. Brown, B. Mann, N. Ryder, et al. Language models
arefew-shotlearners.Advances in neural information
processing systems, 33:1877–1901, 2020.
C. Budak, S. Goel, and J. M. Rao. Fair and Balanced?
Quantifying Media Bias through Crowdsourced Con-
tent Analysis.Public Opinion Quarterly, 80(S1):
250–271, 04 2016. doi: 10.1093/poq/nfw007. URL
https://doi.org/10.1093/poq/nfw007.
M. Cemri, M. Z. Pan, A. Parameswaran, D. Klein,
K. Ramchandran, M. Zaharia, J. E. Gonzalez, and
I. Stoica. Why Do Multi-Agent LLM Systems Fail?,
2025. URL https://arxiv.org/abs/2503.13657 .
arXiv:2503.13657.
E. Y. Chang. Crit: Prompting large language models
with the socratic method. InIEEE Computing andCommunication Workshop and Conference (CCWC),
2023a. URL https://arxiv.org/abs/2303.08769 .
arXiv:2303.08769.
E. Y. Chang. Examining GPT-4’s Capabilities and
Enhancement with SocraSynth. InThe10thInter-
national Conf. on Comp. Science and Comp. Intelli-
gence, December 2023b.
E. Y. Chang.Multi-LLM Agent Collaborative Intelli-
gence: The Path to Artificial General Intelligence.
SocraSynth, March 2024; ACM Books (accepted),
2025.
H. K. Choi, X. Zhu, and Y. Li. Debate or vote:
Which yields better decisions in multi-agent large lan-
guage models?, 2025a. URL https://arxiv.org/
abs/2508.17536.
S. Choi, C. Y. Park, K. Halmekyt, H. Chung, I. Kim,
J. Seo, J. Song, D. Gwak, and Y. Kang. Debate or
vote: Which yields better decisions in multi-agent de-
liberation?arXiv preprint arXiv:2508.17536, 2025b.
URLhttps://arxiv.org/abs/2508.17536.
M. J. A. N. d. C. Condorcet.Essai sur l’application
de l’analyse à la probabilité des décisions rendues à
la pluralité des voix. De l’Imprimerie Royale, Paris,
1785.
T. M. Cover and J. A. Thomas.Elements of Informa-
tion Theory. Wiley-Interscience, 2nd edition, 2006.
Z. Dai et al. Gpts are gpts: An early look at the labor
market impact of large language models. InarXiv
preprint arXiv:2304.09797, 2023.
A. Estornell and Y. Liu. Multi-llm debate: Framework,
principles, and interventions. InAdvances in Neural
Information Processing Systems 37 (NeurIPS), 2024.
K. Gödel. On formally undecidable propositions of
Principia Mathematicaand related systems i. In
J. van Heijenoort, editor,From Frege to Gödel:
A Source Book in Mathematical Logic, 1879–1931,
pages 596–616. Harvard University Press, 1967.
Translated by Jean van Heijenoort.
G. Hong et al. Mixtures of in-context learners. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (ACL), Bangkok,
Thailand, 2025. Association for Computational Lin-
guistics. URL https://aclanthology.org/2025.
acl-long.1277/.
R. Hong, H. Zhang, X. Pang, D. Yu, and C. Zhang.
A Closer Look at the Self-Verification Abilities
of Large Language Models in Logical Reasoning.
arXiv:2311.07954, 2024. URL https://arxiv.org/
abs/2311.07954.
G. Irving, P. Christiano, and D. Amodei. AI safety
via debate.arXiv:1805.00899, 2018. URL https:
//arxiv.org/abs/1805.00899.9

L. V. Kantorovich. On the translocation of masses.
Doklady Akademii Nauk, 37(7-8):199–201, 1942.
J. Kaplan, S. McCandlish, T. Henighan, et al. Scaling
laws for neural language models.arXiv preprint
arXiv:2001.08361, 2020.
Z. Kenton, N. Y. Siegel, J. Kramár, J. Brown-Cohen,
S. Albanie, J. Bulian, R. Agarwal, D. Lindner,
Y. Tang, N. D. Goodman, and R. Shah. On
scalable oversight with weak llms judging strong
llms.arXiv preprint arXiv:2407.04622, 2024. URL
https://arxiv.org/abs/2407.04622.
S. Kullback.Information Theory and Statistics. John
Wiley & Sons, 1951.
W. Li, Y. Lin, M. Xia, and C. Jin. Rethinking mixture-
of-agents: Is mixing different large language models
beneficial?, 2025. URL https://arxiv.org/abs/
2502.00674.
T. Liang, Z. He, W. Jiao, X. Wang, and more. Encour-
aging divergent thinking in large language models
through multi-agent debate. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 17889–17904, Nov. 2024.
doi: 10.18653/v1/2024.emnlp-main.992. URL https:
//aclanthology.org/2024.emnlp-main.992/.
J. Lin. Divergence measures based on the shannon
entropy.IEEE Transactions on Information theory,
37(1):145–151, 1991.
Y. Liu, J. Cao, Z. Li, R. He, and T. Tan. Breaking
mental set to improve reasoning through diverse
multi-agent debate. InInternational Conference on
Learning Representations (ICLR), 2025.
A. Panickssery, S. R. Bowman, and S. Feng. LLM
evaluators recognize and favor their own genera-
tions. InAdvances in Neural Information Process-
ing Systems 37 (NeurIPS 2024), 2024. URL https:
//neurips.cc/virtual/2024/poster/96672.
B. Recht. A tour of reinforcement learning: The view
from continuous control.Annual Review of Control,
Robotics, and Autonomous Systems, 2:253–279, 2019.
B. Settles. Active learning literature survey. Techni-
cal Report 1648, University of Wisconsin-Madison
Department of Computer Sciences, 2009.
C. E. Shannon. A mathematical theory of communica-
tion.Bell System Technical Journal, 27(3):379–423,
1948.
J. E. Shore and R. W. Johnson. Axiomatic derivation of
the principle of maximum entropy and the principle
of minimum cross-entropy.IEEE Transactions on
Information Theory, 26(1):26–37, 1980.
A. P. Smit, P. Duckworth, N. Grinsztajn, T. D. Bar-
rett, and A. Pretorius. Should we be going MAD?a look at multi-agent debate strategies for llms. In
Proceedings of the 41st International Conference on
Machine Learning (ICML 2024), volume 235 ofPro-
ceedings of Machine Learning Research, pages 49941–
49956. PMLR, 2024. URL https://proceedings.
mlr.press/v235/smit24a.html.
L. Tharmalingam. Kaggle Disease Symp-
toms and Patient Profile Dataset, 2023.
URL https://www.kaggle.com/code/rospel/
basic-medical-info-a-key-to-predict-disease .
A. Vaswani et al. Attention is all you need.Advances
in neural information processing systems, 2017.
J. von Oswald, E. Niklasson, E. Randazzo, J. Sacra-
mento, A. Mordvintsev, A. Zhmoginov, and M. Vla-
dymyrov. Transformers learn in-context by gradient
descent. InProceedings of the 40th International
Conference on Machine Learning, ICML’23, pages
35151–35174, 2023.
J. Wang, J. Wang, B. Athiwaratkun, C. Zhang, and
J. Zou. Mixture-of-agents enhances large language
model capabilities.arXiv:2406.04692, 2024a. URL
https://arxiv.org/abs/2406.04692.
Q. Wang, Z. Wang, Y. Su, H. Tong, and Y. Song.
Rethinking the bounds of llm reasoning: Are multi-
agent discussions the key? InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (ACL 2024), pages 6106–6131, 2024b.
doi: 10.18653/v1/2024.acl-long.331. URL https:
//aclanthology.org/2024.acl-long.331.
S. Wang, C.-H. H. Yang, J. Wu, and C. Zhang.
Bayesianexampleselectionimprovesin-contextlearn-
ing for speech, text and visual modalities. In
Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 20812–20828, 2024c. doi: 10.18653/v1/2024.
emnlp-main.1158. URL https://aclanthology.
org/2024.emnlp-main.1158.pdf.
X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi,
S. Narang, A. Chowdhery, and D. Zhou. Self-
consistency improves chain of thought reasoning
in language models. InICLR, 2023. URL https:
//arxiv.org/abs/2203.11171.
S.M.Xie, A.Raghunathan, P.Liang, andT.Ma. Anex-
planation of in-context learning as implicit bayesian
inference.International Conference on Learning Rep-
resentations (ICLR), 2022.
10

Appendix Overview
•Appendix A: Notation and symbols
•Appendix B: Configuration, hyperparameters, ex-
perimental setup
•Appendix C:MACIalgorithm specification
•Appendix D: Information-theoretic signals
•Appendix E: CRIT evaluator details
•Appendix F: Adaptive initialization strategy
•Appendix G: Dengue case study (full transcript)
•Appendix H: Hepatitis case study (full transcript)
•Appendix I: Linguistic analysis and behavioral
validation
•Appendix J: News bias detection
•Appendix K: Majority vote failure analysis
•Appendix L: Learned scheduler for( α, γ)(algo-
rithm and no-regret analysis)
•Appendix M: Plateau guarantees (monotone dis-
persion and termination)
A Notation and Symbols
Table 9: Key symbols used throughout.
Symbol Meaning
MACI Multi-Agent Collaborative Intelligence (dual-
dial control framework)
QTarget-aligned evidence quality
DJS Inter-agent disagreement (Jensen-Shannon di-
vergence)
OSupport overlap (Jaccard over cited spans)
ˆINormalized information gain (entropy drop)
CRIT Argument-quality score from independent
LLMs
CLBehavior dial (contentiousness level)
τQ Evidence-quality gate (information dial)
τCRIT Argument-quality gate (evaluator dial)
τO Overlap floor used only in the stopping rule
ω(t)
u Reliability weights (EMA ofCRIT), agentu
p(t)
u Agent u’s probability distribution at round t
p(t)Reliability-weighted mixture distribution
E(t)
u Evidence spans cited by agentuat roundt
r(t)
I, r(t)
DRelative progress ratios (information gain, dis-
agreement)
π(t)
I, π(t)
DPlateau detection flags
fW1Normalized Wasserstein-1 distance
BConfiguration and Hyperparameters
B.1 Default Hyperparameters
Unless specified otherwise, MACIuses these settings
across all experiments:Control parameters.
•Reliability smoothing (EMA): λ= 0.8,ϵ=
10−6.
•Initial gates:τ(1)
Q= 0.50,τ(1)
CRIT = 0.30.
•Gate tightening:( τ(t+1)
Q, τ(t+1)
CRIT) ←
min
(τ(t)
Q, τ(t)
CRIT) +γ π(t)
I,(τmax, τmax)	
.
•Overlap floor for stopping (not an admission
gate):τ O= 0.30.
Initialization bins (development set).
•High quality (Q ≥0.7) and low disagreement
(DJS≤0.3):CL(1)= 0.7.
•Low quality (Q ≤0.5) and high disagreement
(DJS≥0.5):CL(1)= 0.9.
•Otherwise:CL(1)= 0.8.
Contentiousness schedule.Standard progression:
CL∈ { 0.9,0.7,0.5,0.3}with adaptive transitions trig-
gered by plateau flags.
B.2 Clinical Diagnosis (G1) Setup
Dataset.Kaggle Symptom-Disease Corpus, strati-
fied split: 1,500 test cases (50 per disease, 30 condi-
tions). Train/dev/test files and random seeds provided
in supplement.
Retrieval.BM25 with dense reranker (all-MiniLM-
L6-v2),k ret= 32, max context 4k tokens.
Backbones.
•GPT-4o (gpt-4o-2024-08-06)
•Claude 3 Opus (claude-3-opus-20240229)
•Gemini 1.5 Pro (gemini-1.5-pro-001)
•Temperature: 0.7 for all agents
Evaluation.20 bootstrap resamples for confidence
intervals. BCa method. Paired Wilcoxon tests with
Benjamini-Hochberg FDR correction atq= 0.05.
B.3 News Bias (G2) Setup
Dataset.619 scandal-focused articles with partisan
annotations (5-point scale). [Budak et al., 2016] col-
lected 803,146 articles from fifteen U.S. news out-
lets,cidentifying 10,502 political articles. We focus on
scandals as this content exhibits the strongest partisan
disagreement.

Agents.GPT-4o and Claude 3 Opus with asym-
metric initialization (one Democrat-framed, one
Republican-framed).
Identical hyperparameters.Same CLschedule,
gates, stopping criteria as G1. No domain-specific
tuning.
C MACI Algorithm Specification
Algorithm 1MACIModerator with Dual-Dial Control
1:Initialize debate record ψ(1); setCL(1)via dev binning; set
gatesτ(1)
Q= 0.5,τ(1)
CRIT= 0.3.
2:fort= 1,2, . . .do
3:Provide A(t)= 
ψ(t), CL(t), τ(t)
Q, τ(t)
CRIT
to both agents.
4:foru∈ {A, B}do
5:Agent uemits claim–justification pairs, p(t)
u∈∆(Y),
cites ˆE(t)
u.
6:end for
7: Admission (dual gate).For each candidate argument
ifrom either agent, compute a per-argument evidence
score q(t)
i(analogous to Eq. (1), using only spans cited
byi) and itsCRIT(t)
i.
8:Admit iintoψ(t+1)iffq(t)
i≥τ(t)
QandCRIT(t)
i≥τ(t)
CRIT;
otherwise discard.
9:Update reliability:Γ(t)
u←λ Γ(t−1)
u + (1−λ)CRIT u,t;
computeω(t)
uvia Eq. (4).
10:Form mixture: p(t)=P
uω(t)
up(t)
u; compute ˆI(t)via
Eq. (5).
11:Compute signals: D(t)
JS(Eq. 2),O(t)(Eq. 3),Q(t)(Eq. 1).
12:Update progress ratios r(t)
I,r(t)
Dand flags π(t)
I,π(t)
Dvia
Eqs. (6)–(7).
13:Update dials: CL(t+1)via Eq.(8); apply Eq. (9)compo-
nentwiseto obtain{τ(t+1)
Q, τ(t+1)
CRIT}.
14:ifstopping criterion Eq. (10) satisfiedthen
15: returncalibrated p(t)and curated debate record ψ(t).
16:end if
17:end for
Default hyperparameters: w=3,η=10−3,λ=0.8,
ϵ=10−6,τstop=2,αI=αD=0.2,γ=0.1,τ(1)
Q=0.50,
τ(1)
CRIT=0.30.
D Information-Theoretic Signals for
Debate Evaluation
This appendix presents the mathematical foundations
underlying MACI’s measurement and control frame-
work. We detail each metric’s properties, computa-
tional requirements, and specific role in debate moder-
ation.D.1 Target prototype construction(θ)
D.2 Metric Selection and Usage in MACI
We highlight each metric’s properties and how MACI
employs them in practice:
1.Cross-Entropy (CE).Likelihood surrogate that
mixesuncertaintyandmismatch[ShoreandJohnson,
1980, Cover and Thomas, 2006].MACI use:report-
ing and ablation studies, calibration assessment—
not a primary control signal due to asymmetry and
sensitivity to probability shifts.
2.Entropy.Quantifies uncertainty in a single predic-
tion [Shannon, 1948].MACI use:per-roundinfor-
mation gainvia normalized entropy reduction of the
reliability-weighted aggregate.
3.Kullback–Leibler (KL) divergence.Directed
discrepancymeasurethatisasymmetricandsupport-
sensitive [Kullback, 1951, Cover and Thomas, 2006].
MACI use:auxiliary analysis to detect directional
belief shifts; not the primary disagreement metric
due to asymmetry concerns in symmetric debate
settings.
4.Jensen–Shannon ( DJS) divergence.Symmetric,
bounded [0,1], and finite even under disjoint support;√DJSsatisfies triangle inequality, forming a true
metric[Lin,1991].MACI use:primarydisagreement
signal for discrete labels via unified distance measure
D(t).
5.Mutual Information (MI) and Normalized
MI.Measures shared information between random
variables [Cover and Thomas, 2006]; text-based MI
estimators can exhibit high variance.MACI use:
logged for analysis and visualization; excluded from
control loop due to estimation instability.
6.Wasserstein Distance (WD).Geometry-aware
discrepancy that respects label structure; efficiently
computable in 1D via cumulative distribution func-
tions [Kantorovich, 1942].MACI use:disagreement
measurement for ordered or structured outputs via
unified distance measureD(t).
7.Information Gain (IG).Entropy reduction of
the reliability-weighted mixture, normalized by
log|Y|for interpretability [Cover and Thomas, 2006].
MACI use:primary convergence signal with moving
average plateau detection.
D.3 Integration in MACI Framework
MACI employs these signals in three distinct capacities:12

Primary Control Signals.The unified distance
measure D(t)adapts to output type: D(t):=
DJS({B(t)
i})for categorical labels and D(t):=
fW1({B(t)
i})for structured outputs. This provides con-
sistent plateau detection logic while respecting task-
appropriate geometry.
Progress Tracking.Normalized information gain
bI(t)with moving average MAwquantifies uncertainty
resolution. Plateau detection (∆ bI(t)< εIforτstop
rounds) triggers contentiousness reduction: CL(t+1)=
[CL(t)−αI]+.
Analysis and Validation.KL divergence reveals
directional bias in agent updates. Mutual information
tracks shared representational structure. Cross-entropy
supports calibration assessment. These signals inform
post-hoc analysis but don’t participate in real-time
control.
D.4 Mathematical Formulations
Normalized Information Gain
bI(t):=H(p(t−1))−H(p(t))
log|Y|, p(t)=X
uω(t)
up(t)
u.
Moving Average Plateau Detection
∆bI(t):= MA w(bI(t))−MA w(bI(t−1))
where MAw(x(t)) =1
wPw−1
i=0x(t−i)is the length- w
moving average.
Jensen-Shannon Divergence
DJS(P, Q) =1
2DKL(P∥M) +1
2DKL(Q∥M),
M=1
2(P+Q).
Normalized Wasserstein DistanceFor cumula-
tive distribution functionsF PandF Q:
fW1(P, Q) =W1(P, Q)
max P′,Q′W1(P′, Q′).
Unified Distance Measure
D(t)=(
DJS({B(t)
i}M
i=1)categorical labels
fW1({B(t)
i}M
i=1)structured labels
D.5 Control Logic and Termination
Contentiousness Reduction.Define[ x]+:=
max(0, x). Two plateau monitors trigger intensity re-
duction:
IG plateau:∆ bI(t)< εIforτstoprounds⇒CL(t+1)= [CL(t)−αI]+(11)
Dispersion plateau:|D(t)−D(t−1)|< ε D
forτstoprounds⇒CL(t+1)= [CL(t)−αD]+(12)
Termination Criteria.Debate stops when the
Eq. (10) plateau conditions hold (for τstoprounds);
we also stop early on budget exhaustion or sustained
low CRIT (validator stalemate).
•IG plateau:∆ bI(t)< εIforτstoprounds
•Dispersion stabilization: |D(t)−D(t−1)|< ε Dfor
τstoprounds
•Budget exhaustion: token/time limits reached
•Validator stalemate: CRIT quality below threshold
for extended period
D.6 Practical Interpretation Guidelines
Typical Value Ranges and Decision Thresholds.
Based on empirical analysis across clinical reasoning
and synthesis tasks:
•D JSvalues: >0.3indicates substantial disagree-
ment warranting continued debate; <0.1suggests
near-consensus
•Information Gain: Normalized values >0.1in-
dicate meaningful uncertainty reduction; <0.02
suggests convergence plateau
•Wasserstein Distance: Interpretation depends on
label structure; focus on relative changes rather than
absolute values
•CRIT scores: Socratic evaluation yields [0,1] range;
typical thresholdτ= 0.3
Default Hyperparameters.Standard settings:
w= 3(moving average window), τstop =
2(plateau detection) ,εI= 0.02,εD= 0.05,αI=
αD= 0.2(reduction rates).
E CRIT: Cross-Family Argument
Quality Evaluator
E.1 Design Principles
CRITuses a cross-family LLM evaluator (a differ-
ent model family than the debating agents) to score
argument quality in[0 ,1], mitigating self-preference
bias [Panickssery et al., 2024] and following the So-
cratic evaluation protocol Chang [2023a].13

Evaluation criteria.Each argument is rated on
three equally weighted dimensions (w c= 1/3):
1.Evidence support: claims backed by cited spans;
2.Logical coherence: valid, non-contradictory rea-
soning;
3.Query relevance: directly addresses the task.
Scoring process.Given debate snippet dwith claim
Ω, supporting reasons R, and opposing reasons R′: (1)
validate each link r⇒Ωwith quality γr∈[0,1]; (2)
aggregate to a compositeΓ =P
cwc·score c; (3) return
Γwith a brief justification.
E.2 Multi-Evaluator Aggregation
With Kevaluators (default K=3) and Nu,targuments
from agentuin roundt,
CRIT u,t=1
Nu,tKNu,tX
i=1KX
k=1s(k)
u,t,i,
where s(k)
u,t,iis judge k’s composite score for argument
i. The round-level score is
CRIT(t)=1
NtKX
uNu,tX
i=1KX
k=1s(k)
u,t,i, N t=X
uNu,t.
Using K≥ 3reduces single-judge variance; on our
development sets we observe Krippendorff’sα≈0.68.
E.3 Reliability Weighting in MACI
Exponential moving average tracks per-agent quality:
Γ(t)
u=λΓ(t−1)
u+(1−λ) CRIT u,t,Γ(0)
u= 0.5, λ= 0.8.
Normalized mixture weights with stabilizerϵ= 10−6:
ω(t)
u=Γ(t)
u+ϵ
P
v(Γ(t)
v+ϵ),
and the final mixture is p(t)(y) =P
uω(t)
up(t)
u(y). Re-
moving reliability weighting (uniform ωu= 0.5) costs
−2.0pp Acc@1 (Table 3).
E.4 Implementation Details
Evaluator selection.Use a skeptical, diversity-
seeking judge from a different model family: (i) for
GPT-4o + Claude pairs, use Gemini 1.5 Pro as eval-
uator; (ii) for Gemini pairs, use Claude 3 Opus as
evaluator; temperature0.3.Independence.Judges see only the argument text
andminimaltaskcontext. Agentidentityandopponent
text are masked. Items are randomized per round.
Quality gate.Arguments are rejected unless Q≥
τQ(t)and CRIT≥τ CRIT(t); both gates tighten as CL
is reduced (Eq. 9).
E.5 Example Scoring
Clinical diagnosis prompt:
Rate this medical argument (0–1 scale):
Claim: “Bilateral small-joint arthralgia
favors Chikungunya over Dengue.”
Evidence: [Cites CDC arboviral
guidelines 2024]
Reasoning: Joint pain pattern is more
severe and prolonged in Chikungunya.
Score each: Evidence support, Logical
coherence, Query relevance.
Evaluator response:
Evidence: 0.90 (authoritative source
cited)
Logic: 0.85 (valid differential
reasoning)
Relevance: 1.00 (directly addresses
diagnosis)
Composite: 0.92
E.6 Hyperparameters
Defaults: K=3,λ=0.8(EMA), ϵ=10−6, judge temp
= 0.3,w c=1
3,τ(1)
Q=0.5,τ(1)
CRIT=0.3.
E.7 CRIT Reliability and Role inMACI
Protocol.We adopt the CRITevaluator with cross-
family judges, identity masking, and span-grounded
scoring, following the Socratic protocol. We tune on a
disjointdevelopmentsplitandvalidateonaheld-outset
of 150 clinical-debate arguments. Judges see only the
argument text, cited spans, and minimal task context.
Each judge returns three sub-scores (evidence support,
logical coherence, query relevance) and a composite in
[0,1]with equal weights. Unless noted, 95% confidence
intervals (CIs) for means are computed by nonparamet-
ric bootstrap (10,000 resamples). Proportion CIs use
the Wilson method. For rank correlations we report
bootstrap CIs on Fisher-ztransformed coefficients.
Inter-rater agreement.We compute pairwise Pear-
son and Spearman correlations across all judge pairs,
and Krippendorff’s αfor overall reliability. Table 10
shows α=0.68with strong pairwise correlations (Pear-
son0 .71±0.05, Spearman0 .69±0.04). This level is14

acceptable for research use, so we employ CRITcon-
servatively downstream.
Judge-swap stability ( n=100).We swap the pri-
mary judge family (for example, replace Gemini with
Claude) on n=100debates and recompute winner rank-
ings. Depending on tie-handling for near-equal scores,
the number of winner flips is either (i) k=2⇒2.0%
(Wilson 95% CI [0.55%, 7.00%]) or (ii) k=3⇒3.0%
(Wilson 95% CI [1.03%, 8.45%]). In flipped cases,
score margins are small (typical |∆| ≈0.06, range
[0.04,0.08]), consistentwithnear-thresholditemsrather
than systematic bias.
Order invariance ( n=50).We reverse debater pre-
sentation order ( A→Bbecomes B→A) for n=50argu-
ments. The mean absolute score difference is0 .038(SD
0.024), with94%of cases showing |∆|<0.05. Spearman
rank correlation between forward and reverse orders is
0.92(95% CI [0.87, 0.95]).
Table 10: CRIT reliability on 150 validation arguments
with K=3cross-family judges. Means use bootstrap
95% CIs; proportions use Wilson CIs. Judge-swap
reports bothk=2andk=3outcomes forn=100.
Metric Value 95% CI
Inter-rater agreement
Krippendorff’sα0.68 [0.63, 0.73]
Pearson (avg pairwise) 0.71 [0.66, 0.76]
Spearman (avg pairwise) 0.69 [0.64, 0.74]
Judge-swap stability (n=100)
Winner-flip rate (k=2 ofn=100) 2.0% [0.55%, 7.00%]
Winner-flip rate (k=3 ofn=100) 3.0% [1.03%, 8.45%]
Order invariance (n=50)
Mean|score(A→B)0.038 [0.031, 0.045]
−score(B→A)|
Spearman rank correlation 0.92 [0.87, 0.95]
Cases with|∆|<0.0594% [88%, 98%]
Usage in MACI.Given these properties, we use
CRITin two conservative roles: (i)soft aggregation
weightsfor agent synthesis via ω(t)
uin Section E and (ii)
a component of acompound stop rule. Let ¯Γtbe the
mean CRITscore at round tandVartits across-judge
variance. We stop when
Vart< εand| ¯Γt−¯Γt−1|< δ,
for two consecutive rounds, with small thresholds ε, δ
selected on a development split. This couples evaluator
stability with diminishing returns and prevents over-
debating.
Limitations. CRITrelies on LLM evaluators that
may exhibit unknown biases or failure modes. Cross-
family judging, identity masking, and span groundingmitigate these risks but do not eliminate them. System-
atic validation against human experts remains impor-
tant future work, especially for high-stakes deployment.
Finally, our stability estimates likely depend on judge
competence: using high-capability, cross-family LLMs
to assess rubric-based “reasonableness” contributes to
the observed robustness, whereas weaker or poorly
aligned judges may not yield comparable results; in
such settings we recommend increasing K, tightening
εandδin the stop rule, and adding periodic human
audits.
F Adaptive Initialization Strategy
Before debate, we compute two signals from query and
retrieved context:
Initial evidence quality.Cosine similarity between
mean retrieved embedding and target prototype:
Q0= cos( ¯e(0),θ), ¯e(0)=1
|E(0)|X
e∈E(0)ϕ(e)
∥ϕ(e)∥
where E(0)is the initial retrieved evidence set and θis
the task-specific target prototype.
Initial disagreement.Jensen-Shannon divergence
between agents’ initial single-pass predictions:
DJS,0=1
2KL(p(0)
A∥m(0)) +1
2KL(p(0)
B∥m(0)),
m(0)=1
2(p(0)
A+p(0)
B)
F.1 Initialization Policy
We bin( Q0, DJS,0)on development set and map to
starting parameters:
Intuition.Cases with high-quality evidence and low
initial disagreement likely converge quickly, so start
with lower contentiousness. Cases with poor evidence
quality and high disagreement need more exploration,
so start with higher contentiousness.
Evidence gate.Initial gate τ(1)= 0.3for all cases.
Gate rises during debate via Eq. (9) in Section 3.
F.2 Robustness
This simple binning strategy is stable across:
•Encoder layer choice (late layers preferred)
•Pooling method (mean vs. last-token)
•Retrieval depth (k ret∈ {16,32,48})15

ConditionCL(1)Rationale
HighQ 0(> P 70), LowD JS,0(< P 30) 0.7 High-quality, low disagreement
Middle range 0.8 Moderate complexity
LowQ 0(< P 30), HighD JS,0(> P 70) 0.9 Low-quality, high disagreement
Table 11: Development-set percentile bins for initialization.P XdenotesXth percentile.
Percentiles computed once on disjoint development
split. After initialization, all parameter updates use
live signals (Q(t),D(t)
JS,O(t),CRIT(t)) as described in
Section 3.
G Detailed MACI Case Study:
Dengue vs. Chikungunya Diagnosis
G.1 Three-Phase Debate Walkthrough and
Results
This appendix provides a complete case from the medi-
cal validation (Section 4.1), showing how MACI’s con-
tentiousness scheduling ( CL) and convergence control
yield collaborative synthesis. Signals, weighting, and
stop rules followAlgorithm1.
Case overview.Two heterogeneous agents (GPT-4o,
Gemini 1.5 Pro) jointly diagnose a patient with: skin
rash, joint pain, vomiting, fatigue, high fever, headache,
nausea, loss of appetite, retro-orbital pain, back pain,
malaise, myalgia, and red spots. Ground truth:Dengue
Fever. The debate follows CL∈ { 0.9,0.7,0.5}and
terminates adaptively via the relative plateau rule.
The case exhibits MACI’s three-phase trajectory (ex-
ploration →integration →consolidation). Information-
theoretic evolution is summarized below (weighted JSD
across agent beliefs; normalized 1-Wasserstein fW1be-
tween mixturesp(t)).
Table 12: Signal evolution during MACI-orchestrated
Dengue vs. Chikungunya diagnosis.
RoundCL fW1 MI CRIT JSD
1 0.9 0.68 0.43 0.75 0.56
2 0.7 0.44 0.46 0.82 0.37
3 0.5 0.36 0.49 0.87 0.02
Change (%)−47% +14% +16%−96%
Adaptive termination.The relative dispersion ratio
satisfies max 0≤s<τ r(3−s)
D < εrel
D, so the controller stops
after Round 3 (Section 3); CLscheduling thus runs
only as long as capacity-to-go remains.G.2 Complete Debate Transcript (Round by
Round)
Transcript purpose.We provide complete agent di-
alogue for transparency and reproducibility. Linguistic
analysis in Section G.3 demonstrates how CLmodula-
tion produces observable behavioral changes in agent
communications.
Readers primarily interested in results may skip to
Section G.3 for the pattern analysis.
Note on metrics:The transcript below reportsWD
(raw Wasserstein-1) and DJSas originally logged. In
the main text we reportnormalized fW1and weighted
DJS; magnitudes may differ due to normalization.
G.2.1 Round 1.A: Moderator’s Prompt to
GPT-4o (High Contentiousness,
CL=0.9)
“Identify potential diseases linked to the symptoms (skin
rash, joint pain, vomiting, etc.). Provide top-3 predic-
tions with justifications.”
GPT-4o’s Opening Assessment Top-3 Predic-
tions:Dengue Fever (60%), Chikungunya (25%), and
Zika Virus (15%).
Justifications: Most symptoms are classic signs of
Dengue Fever. Red spots over the body could cor-
respond to petechiae, often seen in Dengue due to
thrombocytopenia (a decrease in platelet count).
Chikungunya can cause similar symptoms to Dengue,
such as fever, severe joint pain, muscle pain, headache,
nausea, and rash. Joint pain from Chikungunya is often
very debilitating and can be more prolonged than in
Dengue.
Zika Virus symptoms can include fever, rash, joint pain,
malaise, and conjunctivitis, with a milder presentation
compared to Dengue or Chikungunya.
G.2.2 Round 1.B: Moderator’s Prompt to
Gemini (High Contentiousness,
CL=0.9) — Exploratory Phase
“Analyze the symptoms (skin rash, joint pain, vomiting,
etc.) and propose differential diagnoses. Provide your
top-3 predictions with supporting rationales, including
pathophysiological mechanisms. Critically refute your16

opponent’s diagnostic hypotheses.”
Gemini’s Opening Assessment — Maximum En-
tropy Difference Top-3 Predictions:Viral In-
fection (60%), Autoimmune Disease (20%), Bacterial
Infection (15%).
Justifications: Symptoms common in viral infections.
Autoimmune diseases is possible, but less likely due
to vomiting and nausea. Bacterial infection is less
probable given rash and muscle pain.
Information metrics (transcript-level): WD = 1.7, MI
= 0.43, CRIT score = 0.75,D JS= 1.366
G.2.3 Round 2.A: GPT-4o (Moderate
Contentiousness, CL=0.7) —
Transition Phase
Prompt:All subsequent prompts state to counter-
argue the opponent LLM and strengthen its own argu-
ments at the modulated contention level =CL.
GPT-4o’s Response Top-3 Predictions:Dengue
Fever (60%), Chikungunya (25%), and Zika Virus
(15%).
Justifications: GPT-4o maintains its predictions,
countering Gemini’s assessment with targeted critiques.
While viral infections can cause these symptoms, the
specific combination suggests a more precise diagnosis.
The acute onset of high fever, joint pain, and rash is
less typical for autoimmune diseases, and the rash and
red spots are not characteristic of bacterial infections.
G.2.4 Round 2.B: Gemini (Moderate
Contentiousness, CL=0.7) —
Transition Phase
Gemini’s Reassessment — Reduced Entropy
Gap Top-3 Predictions:Dengue Fever (50%),
Chikungunya (30%), and Viral Infection (20%). Gem-
ini concedes to GPT-4o in the second round!
Justifications: Gemini acknowledges that its initial
broad prediction of viral infection lacks specificity.
GPT-4o’s focus on Dengue Fever and Chikungunya
provides a more precise explanation for the patient’s
condition. The symptom profile, particularly joint pain
and rash pattern, aligns more closely with mosquito-
borne diseases than with general viral or autoimmune
conditions.
Information metrics (transcript-level): WD = 1.1, MI
= 0.46,CRITscore = 0.82,D JS= 0.905G.2.5 Round 3.A: GPT-4o (Low
Contentiousness, CL=0.5) —
Exploitative Phase
GPT-4o’s Refined Analysis Top-3 Predictions:
Dengue Fever (60%), Chikungunya (35%), and Zika
Virus (5%).
Justifications: GPT-4o slightly adjusts probabilities,
increasing confidence in Chikungunya while reducing
Zika. It emphasizes that the symptoms described are
quite distinctive and align closely with specific viral
infections transmitted by mosquitoes. Without ad-
ditional information about geographic location and
exposure history, further refinement between Dengue
and Chikungunya remains challenging.
G.2.6 Round 3.B: Gemini (Low
Contentiousness, CL=0.5) —
Exploitative Phase
Gemini’s Convergent Assessment — Minimal
Entropy Difference Top-3 Predictions:Dengue
Fever (60%), Chikungunya (35%), and Zika Virus
(5%).
Justifications: Gemini fully converges with GPT-4o’s
assessment, acknowledging the superior specificity of
focusing on mosquito-borne viral diseases. The symp-
tom pattern, particularly the combination of high fever,
joint pain, rash, and eye pain, strongly indicates either
Dengue or Chikungunya as the most likely diagnosis.
Information metrics (transcript-level): WD = 0.9, MI
= 0.49, CRIT score = 0.87,D JS= 0.059
G.2.7 Final Joint Recommendations
(Convergence Achieved)
With convergence achieved across information metrics
(Wasserstein stabilized, MI peaked, CRIT improved),
the agents provide a unified diagnosis ofDengue Fever
(60%). Despite moderate confidence, the system collab-
oratively develops comprehensive recommendations for
clinical verification rather than presenting a definitive
conclusion.
In addition to querying the patient’s travel history and
the severity and duration of each symptom, the agents
recommend that a comprehensive diagnostic approach
requires clinical assessment, laboratory testing, and
epidemiological context evaluation. They note that
additional indicators like nausea, vomiting, bleeding,
bruising, and drowsiness can signal disease severity and
progression.
Recommended Laboratory Tests
•Complete Blood Count (CBC): Identify thrombocy-
topenia or leukopenia.17

•Serology Tests: Detect specific IgM and IgG anti-
bodies for Dengue, Chikungunya, and Zika.
•Polymerase Chain Reaction (PCR): Direct detec-
tion of viral RNA.
•NS1 Antigen Test for Dengue: Early and specific
detection of Dengue virus.
•Urine Test for Zika: Viral presence in urine.
DiscussionThis example demonstratesMACI’s
principled transition from exploration to exploitation
through controlled reduction in contentiousness. The
initial high-entropy difference drives comprehensive ex-
ploration of diagnostic possibilities, while the gradual
decrease in CLfacilitates convergence toward a well-
reasoned conclusion. Quantitative improvements in
information metrics (WD decreased by 47%, MI in-
creased by 14%, CRIT improved by 16%, and DJS
reduced by 96%) verify the effectiveness of MACI’s
information-theoretic orchestration.
Key Capabilities
•Transparent Reasoning: Complete audit trails en-
able post-mortem analysis and verification.
•Epistemic Humility: Quantified uncertainty with
actionable next steps rather than overconfident con-
clusions.
•Training Enhancement: Reasoning-rich, diverse
traces that can improve future LLM performance
on similar tasks.
G.3 Linguistic Patterns vs. Contentiousness
Rather than attempting fine-grained linguistic marker
counts on limited transcript text, we analyze three
clearly observable patterns that demonstrate MACI’s
behavioral modulation effects:
G.3.1 Pattern 1: Diagnostic Hypothesis
Specificity Evolution
Round 1 - Divergent Exploration
–GPT-4: Specific viral diagnoses (Dengue, Chikun-
gunya, Zika) with precise probabilities.
–Gemini: Broad categorical thinking (Viral Infection,
Autoimmune Disease, Bacterial Infection).
– Contrast: High vs. Low diagnostic granularity.
Round 2–3 - Progressive Convergence
–GPT-4: Maintains specificity while adding counter-
argumentative language.
–Gemini: Rapid shift from broad categories to specific
viral diseases, explicitly adopting GPT-4’s frame-
work.–Convergence: Both agents reach identical specific
diagnoses by Round 3.
Quantifiable change:Diagnostic entropy decreases from
maximum divergence (broad vs. specific) to complete
alignment on specific viral diseases.
G.3.2 Pattern 2: Collaborative Language
Emergence
Round 1
–Independent assessments with minimal cross-
reference.
–GPT-4: “Most symptoms are classic signs...” (confi-
dent independent judgment).
–Gemini: Brief, categorical statements with little elab-
oration.
Round 2
– Explicit opponent acknowledgment emerges.
–GPT-4: “While viral infections can cause these symp-
toms...” (direct counter-argument).
–Gemini: “Gemini concedes to GPT-4 in the second
round!” (explicit acknowledgment).
Round 3
– Collaborative assessment language.
–Gemini: “acknowledging the superior specificity of
focusing on mosquito-borne viral diseases”.
–Both: Shared reasoning about diagnostic challenges
(“Without additional information about geographic
location...”).
Final Round
–Unified voice: “the LLMs recommend,” joint clinical
suggestions.
–Collaborative uncertainty: “Despite moderate confi-
dence, the system collaboratively develops...”
G.3.3 Pattern 3: Reasoning Justification
Depth
Evolution of explanatory detail:
Round 1:
–GPT-4: 3–4 sentences per diagnosis with pathophysi-
ological details.
–Gemini: 1 sentence per diagnosis, minimal elabora-
tion.
Round 2:
–GPT-4: Maintains detail while adding comparative
analysis.18

Table 13: Observable linguistic and reasoning patterns across MACI rounds
Dimension Round 1 Round 2 Round 3
Diagnostic specificity Divergent Converging Aligned
Cross-agent reference Minimal Explicit Collaborative
Reasoning depth Asymmetric Improving Symmetric
Uncertainty handling Confident/Vague Comparative Humble/Precise
Clinical recommendations None Limited Comprehensive
–Gemini: Substantial increase in explanatory depth,
diagnostic reasoning.
Round 3:
–Both agents: Complex reasoning about differential
diagnosis challenges.
– Shared acknowledgment of diagnostic limitations.
G.3.4 Summary of Observable Changes
Key finding:The transcript demonstrates clear be-
havioral modulation effects without requiring detailed
linguistic marker analysis. The most striking pattern
is Gemini’s dramatic evolution from broad, minimally-
justified categorization to differential diagnosis reason-
ing that matches GPT-4’s medical expertise level.
Table 13 summarizes the systematic evolution across
all three patterns, demonstrating MACI’s behavioral
modulation effects at multiple levels simultaneously.
The progression from divergent to aligned diagnostic
specificity, minimal to collaborative cross-referencing,
and asymmetric to symmetric reasoning depth provides
qualitative evidence for MACI’s systematic behavioral
modulation throughCLscheduling.
H Extended MACI Case Study #2:
Hepatitis Differential Diagnosis
This appendix provides a second detailed case study
from the medical diagnosis validation (Section 4.1),
demonstrating MACI’s effectiveness on a harder dif-
ferential (several overlapping liver-related conditions)
that required extended collaboration and adaptive ter-
mination.
Case overview.Two heterogeneous agents (GPT-4o,
Claude3Opus)collaborateonapatientwith: itching,
vomiting, fatigue, weight loss, high fever, yellowish skin,
dark urine, and abdominal pain. The debate follows
the contentiousness schedule CL∈ { 0.9,0.7,0.5,0.3},
reaching convergence in four rounds.
H.1 Extended MACI Progression
This case exhibits a four-phase trajectory (explo-
ration →integration →consolidation →synthesis).Information-theoretic signals evolve monotonically to-
ward consensus; mutual information (MI) islogged for
analysis only(not used in control), consistent with
Section 3.
Table 14: Signal evolution for a complex hepatitis
differential. fW1is the normalized 1-Wasserstein; DJSis
the weighted Jensen-Shannon divergence across agents’
belief states.
RoundCL fW1 MI CRITD JS
1 0.9 0.52 0.39 0.76 0.22
2 0.7 0.45 0.41 0.83 0.12
3 0.5 0.33 0.49 0.89 0.04
4 0.3 0.31 0.49 0.92 0.03
Change (%)−40% +26% +21%−86%
Adaptive termination.The relative-dispersion ratio
r(t)
D(Section 3) falls below εrel
Dat Round 4, i.e.,
max 0≤s<τ r(4−s)
D < εrel
D, triggering stop precisely when
incremental progress plateaus.
H.2 Condensed Debate Analysis
Round 1 (Exploration; CL= 0.9).GPT-4o: Hep-
atitis C (40%), Hepatitis B (30%), Cirrhosis (15%),
Obstructive Jaundice (10%), Acute Liver Failure (5%).
Claude: Hepatitis B (35%), Hepatitis C (25%), Ob-
structive Jaundice (20%), Alcoholic Hepatitis (15%),
Hepatitis A (5%).Key divergence:HCV vs. HBV
primacy; inclusion of alcoholic etiology.
Round 2 (Selective Integration; CL= 0.7).Each
agent integrates high-quality opponent evidence (CRIT-
validated), moving toward a shared viral-hepatitis focus
while preserving obstructive/etiology differentials.
Round 3 (Consolidation; CL= 0.5).Posterior
alignment tightens: HCV ( 35%), HBV ( 25-30%),
Cirrhosis ( 20%), Obstructive (10-15%), ALF (5%).
DJS≈0.04.
Round 4 (Synthesis; CL= 0.3).Probability vectors
match; unified protocol and pending-info19

I Mapping Between Contentiousness
Intensity and Linguistic
Characteristics
Table 15 reports aself-supervised ablationthat maps
the behavior dial CLto stable linguistic and reason-
ing markers. We generated two-agent debates with
CL∈ { 0.9,0.7,0.5,0.3,0.1}. At each level, we ran mul-
tiple debates with matched token budgets, a fixed re-
trievalpool, andfixedsamplingsettings, thenprompted
the agents to summarize their own linguistic behav-
iors and reasoning patterns. Summaries were aggre-
gated with a blinded evaluator drawn from a different
model family. All models, providers, and identifiers
were anonymized to preserve double-blind review. Full
prompts and aggregation details appear in the sup-
plementary protocol; quantitative diagnostics ( DJS,bI,
lexical markers, and citation patterns) are reported in
the figures that accompany this appendix.
Extremes of CLbehave as expected: values near1 .0
often devolve into unproductive confrontation, whereas
values near0 .0lack critical evaluation. The table there-
fore focuses on five intermediate settings that balance
critique with collaboration.
Quantitative validation.We ran N=5topics ×
5seeds per level of CL∈ { 0.9,0.7,0.5,0.3,0.1}with
matched token budgets and a fixed retrieval pool. Fig-
ure A.1 reports mean DJSandbIper level with 95%
bootstrap CIs. We observe a positive monotonic as-
sociation between CLand predictive disagreement at
round 1 (Spearman ρ=0.82,p<10−3), and a mild in-
verted U-shaped trend for bIacross rounds with a peak
near CL≈0.5. Support overlapOrises as CLdecreases
in late rounds (Spearmanρ=−0.71,p<10−3).
Manipulation check.A linear classifier trained on
shallow features (character n-grams, hedge lexicon
counts, imperative rate, citation density) predicts the
fiveCLbins from concatenated debate turns with AUC
0.83±0.02under 5-fold cross validation. Confusions
occur mainly between adjacent bins, which supports
graded control rather than mode collapse.
Notes on anonymity and measurement.We do
not name models or providers. Behavior prompts were
templated and differ from evaluation prompts. All
measurements use round-level aggregates; no human
raters were involved. Additional figures report how CL
shifts DJSandbIin parallel with the qualitative pat-
terns above, consistent with the behavior–information
coupling in the main text.
Figure 4:Annotator rating distributions. Left: Democrat
scandals. Right: Republican scandals. Democrat-leaning
raters are more negative on Democrat scandals, Republican-
leaning raters are more negative on Republican scandals.
The typical gap is about one class step.
JNews Bias Detection and Mitigation
with MACI
We stress–test MACIin a non-clinical domain where
evidence style and label semantics differ from diagnosis.
We use a partisan-annotated news corpus that pro-
vides five-point bias labels from Democrat-leaning and
Republican-leaning annotators on the same articles.
This setting isolates behavioral control from domain
knowledge and tests whether the MACIschedule, the
CRITgate, and the DJS–O–Qsignals still move a de-
bate from breadth to depth with calibrated consensus.
J.1 Bias evaluation results
Due to space limits, full details and per-article justifi-
cations appear inAppendixJ.3. We summarize the
key pattern here.
Figure 4 shows four scenarios: (1) D-leaning raters on
Democrat scandals, (2) R-leaning raters on Democrat
scandals, (3) D-leaning raters on Republican scandals,
and (4) R-leaning raters on Republican scandals. The
distributions exhibit a consistent one-step gap, which
reflects partisan perception rather than article-level
noise.
Running two heterogeneous agents under MACIwith
an asymmetric start, the moderator admits arguments
byCRIT, tracksD JSandO, and stops on plateaus.
Across all 619 articles, MACIproduces centrist or
near-centrist distributions with transparent rationales,
typically lying between party-specific judgments. We
present detailed case analysis of 31 representative ar-
ticles in Appendix J.3.1. Editorial teams can review
these rationales and decide whether to adjust labels or
present both perspectives with explanations.20

Table 15: Self-supervised mapping from contentiousness intensity CLto linguistic tone, emphasis, and language
patterns. Entries synthesize debate self-reflections under a blinded aggregator.
CLTone Emphasis Language patterns
0.9 Highly confrontational;
frequent direct challenges.Risks, failure modes, unintended
effects, equity concerns.Polarized declaratives; e.g.,
“should not be allowed”, “unac-
ceptable risk”.
0.7Still confrontational,
more open to counter-
evidence.Conditional acceptance under
strict safeguards; residual cau-
tions.Qualified critiques; e.g., “seri-
ous concerns remain”, “requires
stronger evidence”.
0.5Balanced and dialectical. Symmetric weighing of pros and
cons; search for discriminating
tests.Neutral scaffolds; e.g., “on the one
hand / on the other hand”, “evi-
dence suggests”.
0.3 Moderately agreeable, re-
tains skepticism.Support with caveats; emphasis
on verification and scope.Positive but careful; e.g., “promis-
ing if”, “contingent on”.
0.1 Cooperative and consoli-
dating.Decisive evidence, synthesis, and
next steps.Compressive summaries; e.g.,
“converge on”, “final justification”.
Table 16: Monotonic trends across contentiousness
levels. Means±95% CI over topics and seeds.
CLD JSRound 1 bIpeak (R 1–3)OFinal Round
0.90.46±0.04 0.18±0.03 0.34±0.05
0.70.41±0.04 0.21±0.03 0.41±0.05
0.50.35±0.030.23±0.03 0.55±0.04
0.30.28±0.03 0.19±0.03 0.66±0.04
0.10.22±0.03 0.12±0.020.74±0.04
Table 17: Predicting CLbins from debate text (macro
AUC).
Model AUC
Linear (bag ofn-grams + style feats) 0.83±0.02
RoBERTa-small probe 0.88±0.01
Placebo label (ignore-CLprompt) 0.52±0.02
J.2 Bias mitigation results
We also test whetherMACIcanreducepolarized bias
gaps during a short debate. Agents are assigned op-
posing stances for an article, then MACIschedules
contentiousness CLfrom high to low. The moderator
scores arguments with CRIT, monitors DJS, normal-
ized cross-entropy, normalized mutual information, and
stops when disagreement stabilizes, overlap rises, and
evidence quality is sufficient.
As a running example, we use the Christopher Colum-
bus page. Two agents begin with opposing fram-
ings. Table 20 (appendix) shows how Agent A’s stance
evolves after CRIT-gated counter-evidence. Figure 5
traces the round-by-round dynamics. The debate ends
when DJSand cross-entropy are near zero and mu-
tual information has plateaued, which indicates shared
understanding rather than forced agreement.
Figure 5:Convergence during bias debates. Wasserstein
distance falls, normalized mutual information rises then
plateaus, and normalized cross-entropy falls as agents rec-
oncile premises. The same pattern that drives consolidation
in diagnosis appears here.
Takeaway.The same MACIcontrols that improved
clinical diagnosis also narrow partisan bias gaps here,
while leaving an auditable trail. This supports the
claim that behavioral modulation, not only information
content, is a portable lever for reliable reasoning.
J.3 Supplemental: News Bias Detection and
Mitigation with MACI
Dataset and labels.We analyze 619 scandal-
focused articles with five-point bias labels from
Democrat-leaning and Republican-leaning annotators
Budak et al. [2016]. Each article has labels from both
partisan groups, enabling bias-gap analysis. We ran
MACI on all 619 articles to compute aggregate met-
rics (Table 7 in the main text). For transparency and
detailed mechanism analysis, we provide per-article
breakdowns for 31 representative cases (15 Democrat
scandals, 16 Republican scandals in Tables 18 and
19), sampled to cover diverse news sources (New York
Times, Fox News, Breitbart, HuffPost, etc.) and vary-21

ing degrees of initial partisan disagreement.
Protocol.Two instruction-tuned agents receive the
same article. MACIuses an asymmetric start, a short
CLschedule from high to low, and a CRITgate. Per
round, agents output a five-class bias distribution and
cite evidence. The moderator tracks DJS,O,Q, and
mutual information. Stopping uses the plateau rule
from Section 3.3.
J.3.1 Bias evaluation summaries
Detailed results and discussion for Democrat scandals
appear inAppendixJ.3.2, and for Republican scan-
dals inAppendixJ.3.3. The main pattern is a party-
consistent one-step gap between partisan distributions,
with MACI’s joint distribution typically centered be-
tween them. Representative comparisons with GPT-4
and Claude one-shot baselines are included for context.
J.3.2 Results on Democrat scandals
Table 18 samples 15 articles to compare judgments
from Democrats (D), Republicans (R), MACI(S), and
one-shot baselines from Claude (c) and GPT-4 (g).
The average gap between g and S is about 0.69 scale
steps. In several cases the MACIdebate persuades
the one-shot baseline to revise toward neutrality after
CRIT-admitted counter-evidence. Full rationales for
each item appear in the per-case supplement.
J.3.3 Results on Republican scandals
Table 19 uses 16 Republican-scandal subset to show
a mirrored pattern. The DR distance is largest, and
MACIagain lies between partisan judgments.
J.3.4 Bias mitigation, arguments, and
convergence traces
Figure 5 in the main text shows fW1decreasing, normal-
ized mutual information rising and then stabilizing, and
normalized cross-entropy decreasing. This mirrors the
clinical setting and supports the portability of MACI’s
behavior scheduling andCRITgating.
Notes on sufficiency of annotations.Partisan
self-identification creates natural disagreement pairs.
Articles receive labels from both sides, which provides a
strong testbed for evaluating whether MACIcan bridge
polarized readings. Adding more annotators is less
informative than auditing how a controlled debate nar-
rows the gap with cited evidence. SeeAppendixJ.3.2
and J.3.3 for extended examples.K Why Majority Voting Fails on
Open-Ended Diagnosis
K.1 Problem regime formalization
We formalize when MACIis preferable to majority
voting. Let |Y|be the outcome cardinality and define
initial signals(Q , DJS)from the query and retrieved
context. We say a task is in the MACIregime if any
holds:
1.Large output space and ambiguity: |Y| ≥τ Yand
Q≤τlo
QandD JS≥τhi
DJS.
2.Evidence synthesis required: multi-document or
cross-source grounding with low initial overlap
O(1)≤τlo
O.
3.Calibration risk: baseline majority vote has ECE≥
δat development time.
For small multiple-choice ( |Y| ≤ 5) with adequate cal-
ibration ( ECE< δ ), majority vote typically suffices.
Thresholds( τY, τlo
Q, τhi
DJS, τlo
O, δ)are set by percentiles
on the development set (Appendix B).
K.2 Classical Result: Small Answer Spaces
Majority voting excels when the answer space is small.
The Condorcet Jury Theorem [Condorcet, 1785] shows
that for binary choices with nindependent voters hav-
ing accuracyp >0.5:
Pn(correct) =nX
k=⌈n/2⌉n
k
pk(1−p)n−k→1asn→ ∞
This favorable result extends to small multiple-choice
problems ( k≤5options) where weak alternatives are
easily eliminated. In these settings, independent errors
cancel through aggregation and majority voting reliably
identifies correct answers.
K.3 Breakdown with Large Answer Spaces
Agreement probability collapses.When the an-
swer space grows large, the probability that two voters
agree on the same answer decreases dramatically. With
|Y|uniformly distributed choices, the agreement prob-
ability is approximately:
P(agree)≈1
|Y|+δ
where δcaptures weak correlation from shared context
or evidence. For clinical diagnosis with |Y|= 100+
possible diseases:22

Table 18:Democrat scandals (15 sampled). Comparison of bias assessments for D, R, MACI(S), plus Claude (c) and
GPT-4 (g). Distances DR, DS, SR are class-step differences.
News # Category Negative W. Neg. Neutral W. Pos. Distances (DR,DS,SR) Source
D1∗Civil Rights – D,R,S,c g – 0,0,0 HuffPost
D2∗Civil Rights D,S – R,c,g – 2,0,2 HuffPost
D8 Civil Rights D – S,c,g R 3,2,1 BBC
D31 Environment D – R,S,c,g – 2,2,0 CNN
D37 Politics – D,R,S,c,g – 0,0,0 Yahoo
D69 Healthcare D,c g R,S – 2,2,0 Breitbart
D81∗Economy – D,S R,c g 1,0,1 Breitbart
D98 Economy D,S,c,g R – – 1,0,1 Breitbart
D101 Education c D,S R,g – 1,0,1 NYT
D106 Election – g D,R,S,c – 0,0,0 USA Today
D109 Elections – D,S,c,g R – 1,0,1 Reuters
D157 International – D,S,c R,g – 1,0,1 NYT
D174 International – S,c D,R,g – 0,1,1 LA Times
D188 Nat. Security – S,c,g D,R – 0,1,1 WSJ
D278 Civil Rights – D,S,c R,g – 1,0,1 Fox News
D336 Politics – – D,R,S,c,g – 0,0,0 NYT
Total 15,8,11
Table 19:Republican scandals (16 sampled). Comparison of bias assessments. D and S often align to the right of R.
Distances are class-step differences.
News # Category Negative W. Neg. Neutral W. Pos. Distances (DR,DS,SR) Source
R1 International R,S – D – 2,2,0 NYT
R7 Nat. Security – – D,R,S – 0,0,0 NYT
R15 Economy – R D,S – 1,0,1 HuffPost
R69 Elections – D,S,R – – 0,0,0 Reuters
R124 Gay Rights R S D – 2,1,1 Fox
R125 Crime – R,S D – 1,1,1 Fox
R180 Elections – – D,R,S – 0,0,0 AP
R191 Elections – R D,S – 1,0,1 CNN
R214 Gay Rights R,S – D – 2,2,0 DailyKos
R221 Economy – R D,S – 1,0,1 WSJ
R233 Economy – R,S D – 1,1,0 Fox
R235 Civil Rights D,R – S – 0,2,2 Reuters
R269 Healthcare – R D,S – 1,0,1 NYT
R274 Healthcare – R D,S – 1,0,1 USA Today
R280 Politics D,S – R – 2,0,2 Fox
Total 15,9,11
•Base agreement probability:P≈0.01
•With shared context:P≈0.01-0.05
•Requiring majority from 20 voters: P(majority )≈
0.001
The curse of dimensionality prevents passive aggrega-
tion from working: votes scatter across the large answer
space rather than concentrating on the correct answer.
Empirical validation on G1 dataset.We eval-
uated 20-voter ensembles on 1,500 clinical diagnosis
cases with over 100 possible diseases. Results confirm
the theoretical prediction:
Majority voting provides only +1.8pp improvement
over the best single model. When evidence quality
is low, agreement rates collapse and voting offers no
benefit.K.4 Why Deliberation Dominates in
High-Dimensional Spaces
Multiple choice vs. open diagnosis.The funda-
mental difference between these regimes:
Multiple choice(small |Y| ≤ 5): Passive aggregation
works because (1) weak options are obviously wrong,
(2) independent errors cancel, (3) agreement probability
is high (≈0.2-0.4).
Open diagnosis(large |Y| ≥ 100): Agreement requires
shared evidence synthesis. Votes scatter unless agents
reason through the same evidence to the same conclu-
sion. Agreement probability is low (≈0.01-0.05).
What deliberation provides.Active debate ad-
dresses the dimensionality problem through:23

Table 20:Debate arguments that moved an initially Euro-centric framing toward neutrality. Excerpts illustrate how
CRIT-admitted counter-evidence shifts stance across rounds.
Agent Arguments
Agent A,
Round 1Emphasizes expansion of knowledge and trade, economic impact, historical
significance, and cultural exchange.
Agent B,
Round 1Raises indigenous suffering and genocide, economic exploitation, cultural era-
sure, environmental impact, and ethical concerns.
Agent A,
Round 3Acknowledges severe consequences as part of broader colonial policies, argues
for historical-context evaluation while maintaining significance claims.
Agent A,
Round 4Recognizes inclusion of indigenous perspectives and evolving scholarship, moves
toward balanced narrative.
Case Type Agreement % Acc@1
High-quality evidence(Q >0.7) 60-70% 78.3%
Medium-quality evidence 25-40% 74.1%
Low-quality evidence(Q <0.3) 5-15% 69.8%
Overall 31% 73.9%
Best single model – 72.1%
Table 21: Majority voting performance by evidence
quality.
1.Explicit reasoning exchange: Agents share diag-
nostic logic, not just votes
2.Evidence quality filtering: Poor arguments are
challenged and excluded
3.Differential diagnosis refinement: Agents itera-
tively narrow the hypothesis space together
4.Adaptive stopping: Debate continues until con-
vergence, not for fixed rounds
This explains G1 results: MACIimproves +2.9pp over
majority voting by actively steering agents from ex-
ploration (high CL) to consolidation (low CL), while
voting passively aggregates without improving individ-
ual reasoning quality.
K.5 Regime Classification
Problem complexity determines optimal aggregation
strategy:
•Binary/small multiple choice( k≤4): Majority
voting dominates. Debate overhead not justified.
•Moderate complexity(5 ≤k≤ 20): Debate
may outperform if it improves individual reasoning
enough to offset cost.
•Open-ended problems( k≥100or continuous):
Majority voting fails due to dimensionality. Debate
becomes necessary.
The transition point depends on evidence quality, agent
capability, and computational budget. Our resultssuggest the crossover occurs around k≈ 10-20for
current LLMs.
L Learned Scheduler for(α, γ)
This appendix gives the scheduler that replaces a fixed
(α, γ)schedulewithatinylearningpolicyoveradiscrete
action set. The policy optimizes per-round information
gain and dispersion reduction under a fixed token bud-
get and comes with a no-regret guarantee versus the
best fixed action.
L.1 Action space, signals, reward, and cost
LetA={a1, . . . , a K}be a discrete set of control
moves, where each akmaps to a pair(∆ αk,∆γk)∈
{−δ,0,+δ}2with a clip operator
clip(x;a, b) := min{max{x, a}, b}.
At roundtwe observe summary signals
xt= 
D(t),∆D(t−1), Q(t),∆Q(t−1),IG(t),
∆IG(t−1), b t
,
where D(t)is dispersion, Q(t)is evidence quality, IG(t)
is normalized information gain, and bt∈[0,1]is the
remaining budget fraction.
Define the bounded reward
rt(a) :=w 1[IG(t+1)−IG(t)] +
+w 2[D(t)−D(t+1)] +−λct(a)
Cmax,
with weights w1, w2, λ∈ [0,1], token cost ct(a), normal-
izerCmax>0, and[ x]+=max(0, x). The remaining
budget updates asB t+1=B t−ct(at)withB 1=B.
L.2 Budget-feasible UCB scheduler
We maintain an empirical mean ˆµkand count nkfor
each action. At round twe screen out actions that
would breach the budget in expectation and then apply
UCB on the feasible set.24

Algorithm 2Budget-feasible UCB scheduler for( α, γ)
1:Input: A={(∆αk,∆γk)}K
k=1, total budget B, weights
w1, w2, λ, step sizeδ
2:Initialize α0, γ0, estimates ˆµk←0, counts nk←0,
remainingB 1←B
3:fort= 1,2, . . .do
4:Observe xtand estimate expected per-round costs
bct(k)fork= 1..K
5: Feasible setS t← {k:bc t(k)≤B t}
6:k t←arg max k∈Stˆµk+p
2 logt/max{1, n k}
7: (αt, γt)← 
clip(αt−1+ ∆αkt; 0,1),clip (γt−1+
∆γkt; 0,1)
8:Roll one debate step using( αt, γt)to obtain
D(t+1), Q(t+1),IG(t+1)and realized costc t
9:r t←w 1[IG(t+1)−IG(t)]++w2[D(t)−D(t+1)]+−
λ ct/Cmax
10:ˆµ kt←nktˆµkt+rt
nkt+ 1,n kt←n kt+ 1
11:B t+1←B t−ct
12:ifplateau test satisfied orB t+1≤0then
13:returntranscript and schedule
14:end if
15:end for
L.3 Assumptions
•Rewards are conditionally1-sub-Gaussian and lie in
[0,1]after clipping.
•Costs satisfy0 ≤ct(a)≤C maxand the cost estima-
tor is conservative:bc t(k)≥E[c t(k)| F t].
•The feasible filter Stblocks actions whose expected
cost would breach the remaining budgetB t.
L.4 Main guarantee
Proposition 1(No-regret versus best fixed schedule).
LetTbe the number of rounds executed before the
plateau test fires or the budget is exhausted. Under the
assumptions above, the expected regret of Algorithm 2
with respect to the best fixed action a⋆∈ Athat satisfies
the budget is
E[RT] :=E"TX
t=1 
rt(a⋆)−r t(at)#
=O p
KTlogT
,
and the expected budget violation is zero by construction
ofS t.
Proof.Condition on the filtration {Ft}. Rewards are
bounded and sub-Gaussian, hence the standard UCB
analysis for finite arms yields O(√KTlogT )pseudo-
regret. The feasibility filter Stremoves any action
whose estimated expected cost would exceed Bt. Since
bctis conservative, the realized cumulative cost is a
supermartingale dominated by B, which implies no
expected violation. Summing per-round instantaneous
regret and taking expectations gives the stated bound.L.5 Contextual variant (optional)
Embed xtintoRdand use a linear model with LinUCB
on the same feasible set St. Under realizability and
bounded design, the regret becomes ˜O(d√
T)while
preserving zero expected budget violation.
L.6 Practical configuration
•Discretize(∆ α,∆γ)∈ {− 0.2,0,+0.2}2and include
a no-op. This yieldsK= 10actions.
•Setw1=w2= 0.5unless domain knowledge sug-
gests otherwise. Tune λto match the desired tokens
per round.
•Normalize cost with Cmaxas the 95th percentile of
per-round tokens on a small development set.
•Initialize( α0, γ0)from Section 3.4. Use δ= 0.2to
keep schedules stable.
L.7 Ablations and diagnostics to report
1.Bandit scheduler versus fixed schedule under equal
total token budget.
2.Three-agent scaling curve with accuracy versus bud-
get and plateau rounds.
3.Budget sensitivity: vary Band show trade-offs in
IG gains,Ddrops, and tokens.
4.Window wand penalty λsweeps to confirm robust-
ness.
M Plateau Guarantees
This appendix analyzes the controller used in Sec-
tion 3.3. We formalize the dispersion process, state
mild regularity conditions, and prove that dispersion is
nonincreasing under the gate and that the plateau test
terminates in O(1/ε)expected rounds. Under a con-
traction condition the bound improves to O(log(1/ε)).
M.1 Setup and notation
Letp(i)
t∈∆(Y)be agent i’s distribution at round t,
and¯p t:=1
mPm
i=1p(i)
tthe mixture.
Dispersion choice (theory vs. plots).For the
guarantees we set
DKL(t) :=1
mmX
i=1KL
p(i)
t∥¯pt
,
with all logarithms base2. This choice fixes the refer-
ence distribution ¯pt, enabling a convexity argument. In
figures we also report JSD; on the δ-smoothed simplex25

it is locally Lipschitz in p, hence co-monotone with
DKL(t)up to the noise/gate tolerances used below.
The controller applies a gated averaging update with
bounded noise:
p(i)
t+1= (1−β t)p(i)
t+β t¯pt+ξ(i)
t,
βt∈[β min, βmax]⊂(0,1],(13)
withE[ξ(i)
t| Ft] = 0and ∥ξ(i)
t∥1≤σalmostsurely. The
gate rejects updates that would increase dispersion by
more than ν≥0. We use the same plateau test as in
Eq. (10) of the main text.
Smoothing.For stability and to avoid boundary
effects, we assume either that model outputs are δ-
smoothed, p← (1−δ)p+δuwith uuniform and
δ∈(0,1/|Y|), or equivalently that all distributions lie
in the interior of the simplex with componentwise lower
bound δ. This is consistent with standard temperature
smoothing and with the CRIT gating step.
M.2 Monotonicity of dispersion
Lemma1(Monotonicityundergatedaveraging).Con-
sider the noiseless update p(i)
t+1= (1−βt)p(i)
t+βt¯pt
withβ t∈(0,1]. Then¯p t+1= ¯ptand
DKL(t+1)≤(1−β t)DKL(t).
With noise satisfying (13)and a gate that rejects steps
withD KL(t+1)> D KL(t) +ν, we have
E[D KL(t+1)| F t]≤(1−β t)DKL(t) +c σ+ν,
for a constant cthat depends only on the local Lips-
chitz constant of KL(·∥¯pt)on the δ-smoothed simplex.
In particular, if ν= 0and σ= 0, then DKL(t)is non-
increasing and contracts by a factor at most(1 −βt)at
every accepted step.
Proof.Averaging the agent updates gives ¯pt+1=
1
mP
i 
(1−βt)p(i)
t+βt¯pt
= (1−βt)¯pt+βt¯pt=¯pt.Fix
tandi. By convexity of KL(·∥¯pt)in its first argument,
KL
(1−β t)p(i)
t+βt¯pt¯pt
≤(1−β t) KL
p(i)
t∥¯pt
+β tKL(¯p t∥¯pt)
= (1−β t) KL
p(i)
t∥¯pt
.
Averaging over iyields DKL(t+1)≤(1−βt)DKL(t).
With noise, use the local Lipschitz property of KL(·∥¯pt)
on the δ-smoothed simplex to bound the change in-
duced by ξ(i)
tin expectation by c σ. The gate ensures
any residual increase larger than νis rejected. Com-
bining the three pieces gives the stated inequality.Consequence.Whenthegateuses ν= 0andnoiseis
negligible after CRIT filtering, the dispersion sequence
is nonincreasing by construction. This matches the
implementation where updates that increase dispersion
are discarded.
M.3 Signal regularity and drift
Assumption 1(Signal regularity).There exist con-
stantsL Q, LJ>0such that
|Q(t+1)−Q(t)| ≤L Q∥pt+1−pt∥1,
|CRIT(t+1)−CRIT(t)| ≤L J∥pt+1−pt∥1,
and the noise components of both signals are zero-
mean and bounded by σQ, σJalmost surely. These
conditions hold under δ-smoothing and bounded per-
round transcript changes.
M.4 Termination of the plateau controller
Proposition 2(Termination in O(1/ε)expected
rounds).Let ε >0be the plateau tolerance used in
Eq.(10), and let w≥1be the moving-average window.
Suppose the gate uses ν= 0, βt≥βmin>0, and As-
sumption 1 holds. If the controller has not stopped by
round t, then over any window of length wat least one
monitored quantity among {DKL, Q,CRIT} changes by
at leastε. Consequently, the stopping timeτsatisfies
E[τ]≤DKL(0)−D ⋆
κ ε+Q⋆−Q(0)
κ ε
+CRIT⋆−CRIT(0)
κ ε=O1
ε
,
where D⋆is the noise floor induced by σ,Q⋆≤1and
CRIT⋆≤1are plateaus within their noise floors, and
κ > 0depends only on( βmin, LQ, LJ)and the local
geometry ofKLon theδ-smoothed simplex.
Proof.While the controller is active, at least one of the
plateau predicates in Eq. (10)fails on every window.
Therefore there exists a round within each window
where either DKLdecreases by at least ε, orQin-
creases by at least ε, orCRITincreases by at least ε
after accounting for bounded noise. The update (13)
with βt≥β minand Lemma 1 imply a lower bound
on expected ℓ1movement per accepted step, which
combined with Assumption 1 yields a constant κ >0
such that the expected per-window improvement is at
least κε. Since DKL∈[0,∞)and Q,CRIT∈ [0,1],
summing these improvements until each signal reaches
itsε-tube around its floor or ceiling gives the bound
onE[τ].26

M.5 Contraction case
Corollary 1(Geometric contraction yields
O(log(1/ε))).If, in addition, the noiseless part of (13)
is a contraction in the sense that DKL(t+1)≤c D KL(t)
for some constant c <1whenever the gate admits the
step, then
E[D KL(t)]≤ctDKL(0) +O(σ),
E[τ] =O 
log(1/ε)
.
Proof.The first claim follows by iterating DKL(t+1)≤
c DKL(t)and adding the bounded noise floor. The stop-
ping time bound follows because the number of rounds
needed for DKL(t)to enter an ε-tube around D⋆under
geometric decay is O(log(1/ε)). The controller halts
once all monitored signals lie within their respective
tubes, which occurs no later than the time required for
DKLto do so.
M.6 Remarks and implementation notes
•The gate with ν= 0is a one-line change to the
implementation. It enforces nonincreasing DKL(t)
without affecting empirical behavior.
•Smoothing with δ∈[10−6,10−3]is sufficient to guar-
antee the local Lipschitz property used in Lemma 1.
In practice this is already satisfied by temperature
scaling and by the reliability-weighted mixture.
•The constants( κ, c)are empirical in nature and
depend on the local curvature of KL(·∥¯pt)and on
encoder geometry through Q. We do not attempt
to compute them exactly since the guarantees are
asymptotic in εand qualitative for reviewer reassur-
ance.
27