# Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology for Diffusion-as-Inference on Structured Reasoning Tasks

**Authors**: Libo Sun, Po-Wei Harn, Zewei Zhang, Peixiong He, Xiao Qin

**Published**: 2026-06-25 00:28:08

**PDF URL**: [https://arxiv.org/pdf/2606.26476v1](https://arxiv.org/pdf/2606.26476v1)

## Abstract
Warm-started diffusion samplers accelerate iterative inference, but it is rarely clear which part of the pipeline carries the gain. We study \textbf{retrieval-warmed energy-based reasoning (RW-EBR)} -- an IRED energy-based diffusion model \cite{du2024ired} augmented with a Modern Hopfield trajectory memory -- and contribute a \textbf{five-arm ablation methodology} (oracle, best-constant, per-query-random, shuffled, aligned) that separates three confounded effects: class-prior bias shift, stochastic warm-starting, and graph-aligned value reuse. The diagnostic decomposition is adapted from LLM-RAG evaluation \cite{ru2024ragchecker}. On \textbf{connectivity-2} (Erdős--Rényi all-pairs reachability), the aligned-vs-shuffled-oracle swing reaches \textbf{$+35$\,pp} balanced accuracy on a fixed 1{,}000-graph validation-set diagnostic, with value distribution and retrieval mechanics fixed, only per-graph alignment destroyed, while per-query random initialisation falls below cold -- per-graph alignment, not bias shift or stochasticity, dominates. Yet the \emph{deployable} cold-prediction pipeline misses the acceptance gate at stored-value quality. The same diagnostic logic, stopped at the key-quality screen, applied to \textbf{Sudoku} with a task-specific key encoder produces a clean negative at a \emph{different} component -- key quality, under the current setup. The decomposition names the first blocking component on each task. The setting -- graph reachability refined by an iterative diffusion sampler, with explainability of failure modes as the lens -- places the work within structured and spatio-temporal reasoning.

## Full Text


<!-- PDF content starts -->

Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology
for Diffusion-as-Inference on Structured Reasoning Tasks
Libo Sun1,Po-Wei Harn2,Zewei Zhang1,Peixiong He1,Xiao Qin1,†
1Department of Computer Science and Software Engineering, Auburn University, Auburn, AL 36830,
USA
2Department of Information Management, National Central University, Taoyuan 320317, Taiwan
†Corresponding author.
libo@auburn.edu, harnpowei@ncu.edu.tw, zez0001@auburn.edu, pzh0029@auburn.edu,
xqin@auburn.edu
Abstract
Warm-started diffusion samplers accelerate itera-
tive inference, but it is rarely clear which part of
the pipeline carries the gain. We studyretrieval-
warmed energy-based reasoning (RW-EBR)—
an IRED energy-based diffusion model [Duet al.,
2024 ]augmented with a Modern Hopfield trajec-
tory memory — and contribute afive-arm ab-
lation methodology(oracle, best-constant, per-
query-random, shuffled, aligned) that separates
three confounded effects: class-prior bias shift,
stochastic warm-starting, and graph-aligned value
reuse. The diagnostic decomposition is adapted
from LLM-RAG evaluation [Ruet al., 2024 ]. On
connectivity-2(Erd ˝os–R ´enyi all-pairs reachabil-
ity), the aligned-vs-shuffled-oracle swing reaches
+35ppbalanced accuracy on a fixed 1,000-graph
validation-set diagnostic, with value distribution
and retrieval mechanics fixed, only per-graph align-
ment destroyed, while per-query random initialisa-
tion falls below cold — per-graph alignment, not
bias shift or stochasticity, dominates. Yet thede-
ployablecold-prediction pipeline misses the accep-
tance gate at stored-value quality. The same diag-
nostic logic, stopped at the key-quality screen, ap-
plied toSudokuwith a task-specific key encoder
produces a clean negative at adifferentcomponent
— key quality, under the current setup. The de-
composition names the first blocking component on
each task. The setting — graph reachability refined
by an iterative diffusion sampler, with explainabil-
ity of failure modes as the lens — places the work
within structured and spatio-temporal reasoning.
1 Introduction
Iterative inference procedures — diffusion samplers, energy-
based reasoning models — are increasinglywarm-started:
rather than initialising from noise, the sampler is seeded with
a candidate solution, often one retrieved from a memory of
past solutions, to cut the number of refinement steps. When
such a pipeline improves, or fails, it is rarely clearwhichpartis responsible. A warm-start can help because the retrieved
content is genuinely task-relevant, because it shifts the ini-
tialisation toward a better region regardless of content, or
simply because any per-query perturbation breaks a degen-
erate equilibrium. These explanations imply very different
things about when retrieval warm-starting will generalise, yet
a single end-to-end accuracy number cannot tell them apart.
We frame this as explainable failure attribution for structured-
reasoning systems: localising which component of a reason-
ing pipeline drives an outcome is a question of diagnostic
evaluation, not of aggregate benchmark performance. The
setting that grounds the study — relational structure (all-pairs
reachability over an Erd ˝os–R ´enyi graph) refined by an itera-
tive diffusion sampler — is a graph- and iteration-shaped in-
stance of structured and spatio-temporal reasoning.
We study this attribution problem inretrieval-warmed
energy-based reasoning (RW-EBR): an IRED energy-based
diffusion model [Duet al., 2024 ]augmented with a Mod-
ern Hopfield trajectory memory [Ramsaueret al., 2021 ]
that supplies a per-query warm-start. Our contribution is
afive-arm ablation methodology— oracle, best-constant,
per-query-random, shuffled, and aligned — that separates
three confounded effects of a retrieval warm-start: class-prior
bias shift, stochastic warm-starting, and graph-aligned value
reuse. We organise the analysis with a three-component de-
composition — key quality, warm-start mechanism, stored-
value quality — adapting the retriever- and generator-side di-
agnostic logic of LLM-RAG evaluation [Ruet al., 2024 ]. We
claim neither the decomposition nor the partial-noise warm-
start mechanism as novel — SDEdit [Menget al., 2022 ]is
the mechanism’s predecessor, with WSD [Scholz and Turner,
2025 ]the closest learned-warm-start competitor. The con-
tribution is the ablation methodology and its application to
retrieval-warmed iterative inference, plus the two empirical
findings it surfaces.
Thefirst findingis an alignment effect on connectivity-
2 (all-pairs reachability on Erd ˝os–R ´enyi graphs). Under or-
acle memory, the swing between the aligned arm and the
shuffled arm — gold values whose (key, value) pairings are
permuted across queries, holding the value distribution and
retrieval mechanics fixed — is+35ppin balanced accu-
racy. A constant-init sweep bounds the bias-shift contribu-
tion to≤+8pp, and per-query random initialisation lands atarXiv:2606.26476v1  [cs.LG]  25 Jun 2026

−1.5to−3.1pp: per-graph alignment, not bias shift and not
per-query stochasticity, is the dominant lever. These warm-
start arms are run as a fixed validation-set diagnostic (1,000
graphs, seed 20260420); multi-seed warm-start replication
is left for a larger study. The deployable cold-prediction
pipeline nonetheless misses the−2pp acceptance gate (∆bal
=−4.09pp); the same decomposition localises that failure to
stored-value quality.
Thesecond findingis that the failure mode is heteroge-
neous. We apply the same diagnostic logic — contrastive key
training and the quality-ratio gate, stopped at the key-quality
screen — to Sudoku with a task-specific encoder, and ob-
tain a clean negative at adifferentcomponent: under the cur-
rent mask-aware solved-board target and 500-candidate pool,
the key encoder itself cannot clear its quality gate, whereas
on connectivity the encoder passes and stored-value qual-
ity is the bottleneck. The two case studies show the three-
component decomposition surfacing a different bottleneck on
each task, rather than collapsing them into a single end-to-end
number.
We wrap IRED as the base reasoning model and do not
modify it: IRED’s sampler initialises from a Gaussian (its
Algorithm 2 hard-codes˜y∼ N(0, I)). Our principled neg-
atives concern the retrieval addition we study, not the IRED
backbone.
In summary, we contribute (i) a five-arm ablation method-
ology for retrieval-warmed iterative inference, separating
bias shift, stochastic warm-starting, and aligned value
reuse; (ii) an aligned-vs-shuffled-oracle alignment effect on
connectivity-2 that isolates per-graph value alignment as the
dominant lever; and (iii) a heterogeneous-failure case study
— under the same diagnostic workflow, connectivity-2 fails
at stored-value quality and Sudoku at key quality — identify-
ing the first blocking component on each task.
2 Methods
We evaluate onconnectivity-2 [Duet al., 2024 ]: predict
the all-pairs reachability matrix of an undirected Erd ˝os–
R´enyiG(N=12, p=0.2)graph from its adjacency. Ad-
jacency and targets are rescaled to±1; this is cell-
level binary classification with positive (“path exists”)
prior≈0.63. Training and evaluation both use IRED’s
GraphConnectivityDataset. The base reasoning
model (G0) is IRED’s 32-channel GraphEBM, trained for
30k steps under the IRED denoising objective (MSE on
rescaled±1targets) at 10 diffusion timesteps, with energy-
landscape supervision and inner-loop optimisation enabled.
A sample drawn atTinference timesteps from the unaug-
mented model is thecoldT=Tbaseline.
Retrieval keys come from a 3-layer GIN [Xuet al., 2019 ]
with alabel-ordered readout— per-node features concate-
nated in node-label order rather than sum-pooled — and a
learned per-position id embedding on the layer-0 features.
Label-ordered readout is engineering for our setting: sum-
pool readouts are permutation-invariant, which would de-
stroy the label-indexed structure the warm-start consumer
needs, since retrieved reachability matrices must align with
the query’s node labelling; the id embedding distinguishesdegree-symmetric collapse cases. The encoder is trained with
a supervised contrastive loss [Khoslaet al., 2020 ]on the
4 target-nearest neighbours of each anchor at temperature
τ=0.1for 3,000 steps; the pair-similarity target is per-edge
Hamming agreement of reachability. GIN and SupCon are
off-the-shelf and the design choices are engineering, not a
contribution; Section 4.1 reports the resulting key quality.
A capacity-10,000 Modern Hopfield trajectory memory
[Ramsaueret al., 2021 ]stores (key, value) pairs whose val-
ues are base-model trajectories, populatedwrite-onceduring
a warm-up phase by running cold inference atT anchor =10
over a random stream of 10,000 training examples; there are
no eval-time writes. At eval time a query’s key retrieves a
value via aβ-temperature softmax over the top-8 cosine sim-
ilarities, where the inverse temperatureβcontrols retrieval
peakedness, and the retrieved value seeds the IRED sampler
in one of two ways.Option Areplaces thet=0initialisation
with the retrieved value and runsK refine optimisation-step it-
erations.Option B— the primary reported path — forward-
noises the retrieved value to an injection timestept inject via
the standard diffusion forward marginalq(x t|x0), then
runs the reverse IREDp sample loopfromt inject down
to 0. Reported runs uset inject=2— a mild re-noising on the
model’s 10-timestep diffusion schedule — withβ=20; Op-
tion A atK refine=10is reported alongside it for robustness.
Both reduce forward-pass count relative to a full cold sample.
All connectivity G0 and G1 runs share a fixed cached val-
idation set of 1,000 graphs (seed 20260420). The headline
metric isbalanced accuracy=1
2(rec ++ rec −), reported
with raw accuracy and per-class recall; the≈63/37class im-
balance makes raw accuracy a misleading gate, whereas bal-
anced accuracy is immune to prior-collapse on either class.
TheG1 acceptance gaterequires∆bal acc(warm−cold)≥
−2pp at a forward-pass speedup≥2×. We use this gate as
anoperational diagnostic screenfor the present study — a
sanity threshold for when to stop reporting an arm as a candi-
date deployable warm-start, not a claim about external task-
level success. The−2pp tolerance is roughly15×the per-
seed cold noise floor (balanced-accuracy std0.13pp across 5
seeds; Section 4.3): any violation lies well outside per-seed
sampling jitter, while still permitting a small accuracy drop
when offset by the speedup. PASS/FAIL labels below refer to
this internal screen.
We additionally exercise component K on Sudoku using
the SATNet-style [Wanget al., 2019 ]dataset from IRED
[Duet al., 2024 ]. The key encoder is a 3-layer ResNet
(∼593k parameters) trained with the same SupCon objec-
tive for 3,000 steps; the per-anchor similarity target is per-
cell argmax agreement restricted to query unknowns, scored
against a 500-candidate in-batch pool. Pass criteria are
quality ratio≥0.85andret topw(β=20)≥0.30.
The warm-start mechanism and stored-value components
were not exercised on Sudoku; Section 5 reports the result.
Reproducibility.Upon publication we plan to release a
supplementary archive containing training and evaluation
scripts for the connectivity-2 G0 backbone, the contrastive
key encoder, and all five G1 warm-start arms (cold, oracle,
shuffled, best-constant, and per-query random), together with

query
xkey
φ(x)top-K
retrievewarm init
q(x_t | v)reverse
samplerŷ
Hopfield memory
(keys, values)K
key qualityM
warm-start mechanism
V
stored-value qualityA. Retrieval-warmed inference pipeline
Arm KkeyMmechanismVvalueDiagnostic role
aligned cold-pred ✓ ✓ ✓ deployable reference
aligned oracle ✓ ✓ ★ V-quality upper bound
shuffled oracle ✓ ✓ ★× K-V alignment break
best constant ∅ ✓ C bias-only warm start
per-query random ∅ ✓ R stochastic value control
✓ learned/real ★ oracle value ★× mis-keyed oracle value
∅ bypassed retrieval C constant value R random valueB. Five ablation arms × three failure componentsFigure 1:Diagnostic apparatus for retrieval-warmed inference.(A) RW-EBR pipeline annotated with the three testable components: key
quality (K), warm-start mechanism (M), stored-value quality (V). (B) Five-arm suite as a component matrix:✓= learned/real,⋆= oracle
stored values,⋆×= mis-keyed oracle,∅= bypassed task-informative retrieval, C/R = constant or random stored values.
their fixed configurations; the Sudoku key-encoder training
script and SupCon configuration; the validation-set seed and
the 5-seed cold noise-floor script; and the figure scripts pro-
ducing all six figures of the paper. The archive reproduces all
reported tables and figures against the cached validation set,
and includes a unit-test suite covering the evaluation utilities.
3 Decomposing Retrieval-Warmed Inference
We decompose retrieval-warmed inference into three compo-
nents that can fail independently: (K)key quality— does
the encoder map inputs to keys whose retrieved values warm-
start usefully? Measured byquality ratio: the target
similarity between a query and its top-1 retrieved candidate,
divided by the best target similarity available among all can-
didates; (M)warm-start mechanism— given a retrieved
value of fixed quality, does the inference loop refine it toward
the true target? Isolated by anoracle-memory ablationwrit-
ing ground-truth values directly to memory; (V)stored-value
quality— does the cold model produce predictions useful as
future warm-starts? This adapts LLM-RAG diagnostic de-
compositions [Ruet al., 2024; Sivakumaret al., 2026 ]to
retrieval-warmediterativeinference. Figure 1 summarises
the apparatus and the five arms.arm / config balr +r− spd∆bal
Sanity arm — deployable cold-pred memory:
cold (T=10) .755 1.000 .5111.0×—
warm, cold-pred Opt B .715.999.4303.3× −4.09
Oracle arm — ground-truth memory:
cold (T=10) .753 1.000 .5051.0×—
warm, oracle Opt B .977.993.9603.3×+22.39
warm, oracle Opt A.957 .989 .9265.5×+20.45
Table 1: Warm-start arms on connectivity-2 (β=20; Opt B att inj=2,
Opt A atK ref=10). Sanity arm: deployable cold-pred memory
misses the gate at stored-value quality (V FAIL). Oracle arm: both
warm-start variants clear it (M PASS). Within-arm∆bal vs each
arm’s own cold reference. Fixedn=1,000validation set; multi-seed
caveat in §4.5.
4 Connectivity-2: A Stored-Value-Quality
Failure
We walk the three components in turn on connectivity-2: the
key encoder (Section 4.1), the warm-start mechanism (Sec-
tion 4.2), and stored-value quality (Section 4.3). The five-
arm ablation suite then decomposes the oracle result (Sec-
tion 4.4), and two further interventions characterise the fail-
ure (Section 4.5).

4.1 Key Quality
Validationquality ratiosaturates at≈0.95
(gtpred top1≈0.83,gt best≈0.88);
rettopw(β=20)— the peakedness of softmax re-
trieval weights — reaches 0.49 against the1/8 = 0.125
uniform floor (a prior MLP+MSE baseline plateaued at
0.40). The encoder passes the targeted relabeled-isomorph
regression test — two labelings of a 4-node perfect matching
that share all-degree-1 nodes yet whose reachability matrices
agree on only 50% of cells; we do not claim universal
isomorph discrimination.Component K: PASS.
4.2 Warm-Start Mechanism
The oracle ablation replaces cold-prediction memory writes
with ground-truth reachability writes. This is non-deployable
— we do not have ground truth at memory-write time in
practice — but it measures what the inference mechanism
achieves in this finite-memory setting given perfect stored
values. The result is not a global upper bound; finite mem-
ory coverage, retrieval mismatch on the held-out validation
set, top-Kaveraging, and sampler stochasticity all remain
potential limits even in the oracle condition. We report both
warm-start variants from Methods:Option Batt inject=2and
Option AatK refine=10. The two paths are independent —
Krefine does not appear in Option B,t inject does not appear
in Option A.
Two observations follow from Table 1’s oracle arm. First,
theload-bearing lift is inrec −: it moves from 0.505 (chance)
cold to 0.960 under Option-B warm inference with ground-
truth memory;rec +stays at≈0.99. The mechanism is cor-
recting the negative class, not inflating accuracy via prior col-
lapse. Second, both variants beat the2×speedup gate (3.3×
Option B,5.5×Option A) at near-saturated balanced accu-
racy.
Aβsweep (β∈ {20,40,80,160}, both variants — eight
oracle cells total) passes the gate by≥+18.91pp in every
case. Notable: with oracle memory,lowerβslightly out-
performs higherβ(Option B∆bal atβ=20:+22.39pp; at
β=160:+20.75pp), inverting the cold-pred-memory trend
(Section 4.3). Interpretation: high-quality stored values ben-
efit from top-Kaveraging that smooths cell-level disagree-
ment between near-correct neighbours; low-quality stored
values prefer peakyβto concentrate on the least-bad neigh-
bour.Optimalβdepends on memory quality, not an ar-
chitectural constant. Component M: PASS.
4.3 Stored-Value Quality
Replacing oracle memory writes with cold-prediction mem-
ory writes is the only change from Section 4.2 — same key
encoder, warm-start dispatch, retrieval temperature, and val-
idation set. Both warm arms come from the same run so
the comparison is direct; the two cold rows reflect within-run
sampling jitter (≈0.3pp here; the cold noise floor measured
separately across 5 seeds is balanced accuracy std0.13pp).
All∆s reported here and in Section 4.4 dominate the noise
floor by≥11×(smallest, per-query uniform) to≥170×(or-
acle).
The cold-pred warm row (Table 1, sanity arm) misses the
−2pp gate by2.09pp. More informatively,rec −gets worse,
0.4 0.6 0.8 1.0
Negative-class recall (rec−)cold-pred memory V FAIL
cold 0.511
warm 0.430
Δrec− -8.1 pp
oracle memory M PASS
cold 0.505 warm 0.960
Δrec− +45.5 ppFigure 2:Negative-class recall carries the connectivity-2 story
(Option B,t inject=2,β=20). Cold-prediction memorydegrades
rec−under warm-start (0.511→0.430); oracle memory recovers
it sharply (0.505→0.960).rec +stays near1.0throughout.
comparison Option B Option A
oracle aligned vs. cold+22.39 +20.45
best constant vs. cold+7.92 +5.65
per-query uniform[−1,1]vs. cold−1.54−10.25
per-queryN(0,1)vs. cold−3.09−10.46
shuffled oracle vs. cold−12.80−14.30
aligned−best constant+14.47 +14.80
aligned−shuffled+35.19 +34.75
Table 2: Decomposition of the oracle lift,∆bal in pp atβ=20. Best
constant:c=−0.25(Option B),c=−0.75(Option A).
not better:0.511cold→0.430warm in the same arm (Fig-
ure 2). The warm path inherits and amplifies the cold model’s
class bias rather than failing to fix it; retrieval finds high-
similarity neighbours (key quality is fine), but every neigh-
bour’s stored prediction is itself biased toward the positive
class.
The oracle/cold-pred gap (∆bal= +22.39pp vs−4.09pp
within the same run) is the cleanest component-attribution ev-
idence we have:26.5pp of balanced accuracy separatesiden-
ticalmechanism and retrieval, differing only in stored-value
quality. An independent earlier run reproduces the cold-pred
result at∆bal=−3.75pp on a different cold draw, qualita-
tively similar (both miss the−2pp gate by comparable mar-
gins). By the decomposition of Section 3, the mechanism and
retrieval components are not the bottleneck on this task; the
stored-value component is.Component V: FAIL.
4.4 Bias Shift Versus Alignment: The Five-Arm
Suite
Sections 4.2 and 4.3 contrasted the oracle mechanism’s
+22pp gain with the deployable pipeline’s failure. We now
scrutinise the oracle lift itself: is it genuine per-graph value
reuse, or class-prior shift alone? Equivalently: would any
sufficiently-negative warm init achieve a similar lift by break-

aligned
oraclebest
constantuniform
randomnormal
randomshuffled
oracle−100102030Δ balanced accuracy (pp)cold
(zero ref)+22.39pp
+7.92pp
-1.54pp-3.09pp
-12.80ppalignment lift
+14.47pp
alignment swing
+35.19ppConnectivity-2 alignment decompositionFigure 3:Alignment decomposition on connectivity-2(Option B,
tinject=2,β=20). The aligned oracle exceeds the best bias-only
warm-init by+14.47pp and a shuffled oracle (same gold values,
(key, value) pairings permuted across queries) by+35.19pp, iso-
lating per-graph alignment as the dominant lever. Per-arm∆bal in
Table 2.
ing the cold model’s positive-bias equilibrium, with retrieval
contributing nothing? Three ablations decompose it: a con-
stant warm-init sweepc∈ {−1, . . . ,1}; ashuffled-oracle
arm (gold values, but with (key, value) pairings permuted
across queries — preserving the value distribution while de-
stroying per-graph alignment); and aper-query stochastic
initarm (self-retrieve with per-entry random storage atβ=20,
tinject=2; uniform[−1,1]andN(0,1)).
The constant sweep stores a constant tensor of valuecat
every memory slot; retrieval over identical values is a no-op,
so the warm-start receives exactlyc. The per-query stochastic
arms use self-retrieve with per-entry random storage — top-
1 self-similarity dominates the softmax (mean retrieval-top
weight0.887), so each query retrieves predominantly its own
pre-stored random vector. Table 2 and Figure 3 report the
results.
The decomposition tells a consistent story. Bias shift is
bounded: no constant warm init exceeds+8pp∆bal. On top
of that best bias-only baseline, aligned ground-truth retrieval
adds+14.5pp, and the two warm-start variants — which con-
sume the init differently — agree on this gap to within0.4pp.
Misalignment does not merely fail to help but actively poi-
sons: the shuffled oracle falls≈20ppbelowthe best con-
stant, so alignment is a dominant, non-additive lever rather
than a small perturbation. And per-query stochasticity is not
the missing axis — both random-init arms fall at or below
cold under Option B.
We state the contribution precisely. That lift is thealigned-
vs-best-constantgap, not a one-sided “retrieval contributes
+14.5pp” claim; the full alignment effect is the+35pp
swing from misaligned to aligned ground truth. A tuned con-
stant prior delivers up to+8pp; the remaining lift requires
per-graph alignment between key and stored value, which
contributes nearly twice what the best bias shift does. The
objection that the oracle merely supplies a better class prior
or a generic warm-init regulariser does not survive these con-
trols.
1 5 20−8−6−4−20Δbal (pp)
-7.69-6.41-3.75V FAIL
G1 gateCold-pred memory: Δbal improves with β
20 40 80 160
retrieval softmax temperature β (log scale)20.521.021.522.022.5Δbal (pp)
+22.39
+21.86
+21.07
+20.75M PASSOracle memory: Δbal softens with βFigure 4:β-trend inversion:∆bal’s response toβdepends on
stored-value quality(Option B,t inject=2). Cold-prediction mem-
ory:∆bal improves monotonically withβbut never clears the−2pp
gate. Oracle memory: trend inverts — highest at the lowestβtested,
softening asβgrows. Non-overlappingβ-grids; diagnostic trends,
not a head-to-head sweep.
4.5 Why Stored-Value Quality Fails
Two interventions further characterise the failure as represen-
tational rather than procedural.
Threshold tuning.Cold raw outputs lie in[−0.99,+0.99]
and are evaluated at thresholdτ=0. Sweepingτ∈[−1,1]
in steps of0.02over the validation set and selectingτfor
balanced accuracy yields a maximum lift of+0.47pp (at
τ=+0.96: acc0.821, bal. acc0.760,rec +1.000,rec −0.520,
againstτ=0:0.817,0.755,1.000,0.511), small relative to the
−2pp gate and not qualitatively corrective. Crucially,rec +
stays at1.000up toτ= + 0.96— the model does not even
consider assigning negative class until the threshold reaches
96% of its prediction range. Threshold shifting reveals no re-
coverable balanced-accuracy fix under this probe; features are
sharply committed and largely class-biased. The “right fea-
tures, wrong decision boundary” diagnosis is not supported.
Class-weighted retraining.A more invasive intervention:
modify IRED’s denoising MSE to apply a4×per-cell weight
on negative-class cells, by subclassing the IRED denoising
objective with a per-cell loss weight applied inside the diffu-
sion loss, mirroring the existing per-cell weighting precedent
in IRED’s shortest-path task. After 30,000 steps under the
same schedule, balanced accuracydecreasedby3.2pp: the
original G0 reaches bal. acc0.755(rec +1.000,rec −0.511
— strongly biased toward the positive class), while the re-

metric gate best (step) step 3000
quality ratio≥0.850.434 (2000) 0.420
rettopw(β=20)≥0.300.242 (3000) 0.242
recall@1diagnostic 0.010 (1500) 0.006
loss EMA (≈4.16) — 5.290
Table 3: Sudoku key-encoder training (3000 SupCon steps). Both
key-quality kill criteria fail by margin, and validation metrics plateau
despite continued loss descent; we stop before evaluating M/V .
balanced G0 reaches0.724(rec +0.447,rec −1.000— flips
toward the negative class). The retrained model did not find
a balanced equilibrium; it flipped polarity. Two degenerate
equilibria with no balanced middle suggest the learned rep-
resentation does not support balanced classification under the
tested training recipe — class weighting decides only which
side training collapses to.
The two probes converge on the same conclusion: the
stored-value limit is consistent with a learned-representation
limit under this training recipe rather than with a tunable
boundary or loss weight. Two interventions thatshouldhave
helped if the failure were only a decision-threshold or class-
weighting artifact both fail to. Theβ-trend inversion of Fig-
ure 4 is the same story read through retrieval temperature: the
direction of the trend depends on stored-value quality.
Several questions remain open. We have not shown that
a larger or differently-architected G0 — a deeper GNN, an
attention-based variant, an alternative loss — could not solve
connectivity-2, nor that the oracle ceiling generalises to tasks
where the cold model is already strong. By extension we ex-
pect iterative refill of warm predictions into memory to am-
plify cold bias, but we have not tested this formally. Our
reverse path is the stochastic DDPMp sample; whether
a deterministic-reverse DDIM path behaves identically is
untested — Option B covers the forward-init half, not the re-
verse. Finally, the warm-start ablations run on a single fixed
validation set, and multi-seed warm-start replication is left for
a larger study.
5 Sudoku: A Key-Quality Failure
Applying the same diagnostic logic to Sudoku — stopped at
the key-quality screen, before the warm-start mechanism and
stored-value components are exercised — produces a clean
negative at adifferentcomponent than connectivity. The
ResNet key encoder (Methods) trained 3,000 steps reaches
quality ratio0.42(gate≥0.85, missed by half) and
rettopw(β=20) 0.242(gate≥0.30, missed by0.058ab-
solute); both metrics plateau (Table 3). Loss descends but
validation metrics decouple after step 1500; we do not pro-
ceed to warm-start diagnostics under this setup.
The Hamming-similarity distribution on solved boards
(Figure 6B) is unimodal at the iid baseline (mean= 1/9≈
0.111; q95= 0.185, max= 0.395).gt best— the in-batch
maximum target similarity per anchor under the mask-aware
target across 500 candidates — plateaus at≈0.30across
all training checkpoints. Even with a perfect encoder, only
≈30%of the query’s unknown cells would match the true
digit — a weak warm-start target relative to the cold end-point’s0.97cell accuracy. The downstream warm-start effect
is not measured here: Sudoku stops at the K-screen, before
M/V are exercised.
Two distinct limits hold under the current setup. The en-
coder under-retrieves:quality ratio0.42means top-1
retrievals reach less than half the in-batch best available sim-
ilarity, so the encoder retrieves substantially worse than the
candidate pool allows. Separately, the in-batch best itself
plateaus atgt best≈0.30— a candidate-pool / target ceil-
ing, not an encoder verdict, so even a perfect retriever deliv-
ers warm-starts at only≈0.30absolute target similarity to the
query. A better encoder alone could in principle clear the nor-
malizedquality ratiogate; producing a useful warm-
start signal on Sudoku, however, requires moving both limits.
Sudoku fails at component K (key quality, under the current
setup); connectivity fails at component V (stored-value qual-
ity, with the mechanism working) — Figure 5. A symmetry-
aware encoder, a canonicalised or larger candidate pool, or a
different value target could move these limits; we did not run
those configurations.
6 Related Work
RW-EBR wraps IRED [Duet al., 2024 ], whose energy-based
diffusion solves the connectivity-2 and Sudoku tasks stud-
ied here; IRED’s ablations cover gradient-on-energy, multi-
step refinement, and contrastive shaping — not memory or
warm-start initialisation. IREM [Duet al., 2022 ]is the ear-
lier energy-minimisation ancestor, also without memory or
warm-start. The warm-start step itself — forward-noise a re-
trieved value tot inject , then denoise fromt inject down to 0 —
is the standard partial-noise warm-start introduced by SDEdit
[Menget al., 2022 ]for image editing; the forward step is the
standard diffusion marginalq(x t|x0)shared by DDPM [Ho
et al., 2020 ]and DDIM [Songet al., 2021 ], with DDIM dis-
tinguished by its deterministic reverse sampler (we use the
stochastic DDPMp sample). The closest learned-warm-
start competitor is WSD [Scholz and Turner, 2025 ], which
learns a per-query informed Gaussian prior for conditional
generation. Sampler-distillation approaches such as Consis-
tency Models [Songet al., 2023 ]are a complementary accel-
eration axis (few-step approximation rather than warm-start
init), not exercised here.
RDM [Blattmannet al., 2022 ]and kNN-Diffusion
[Sheyninet al., 2022 ]condition image diffusion models on
retrieved nearest neighbours; MEMENTO [Chalumeauet al.,
2025 ]conditions a combinatorial-optimisation solver on a
memory bank. In all of these the retrieved item enters as a
conditioning signal alongside the input; RW-EBR differs in
the role of retrieval, where the retrieved vector is the warm-
start initialisation of the sampler’s dynamical update rather
than a conditioning vector. The trajectory memory itself is
a Modern Hopfield network [Ramsaueret al., 2021 ]; a re-
cent theoretical line [Ambrogioni, 2024; Phamet al., 2025 ]
bridges diffusion models and associative memory, which we
use operationally — retrieve a candidate trajectory and feed
it as a warm-start — rather than developing it theoretically.
RAGChecker [Ruet al., 2024 ]and RAG-X [Sivakumaret
al., 2026 ]decompose LLM retrieval-augmented-generation

K
Key quality
encoder retrievalM
Mechanism
oracle memoryV
Stored values
cold-pred memory
Connectivity-2
G(12, 0.2)
reachability
Sudoku
Sudoku 9×9
current setupquality_ratio ≈ 0.95
GIN + SupCon
label-ordered readout
✓   PASSΔbal = +22.39pp
oracle ablation
(Option B, t_inject=2, β=20)
✓   PASSΔbal = -4.09pp
cold-pred memory; gate ≥ -2pp
×   FAIL · bottleneck
quality_ratio ≈ 0.42
gate ≥ 0.85
gt_best ≈ 0.30 cap (500 cands)
×   FAIL · bottleneck(not run)
STOP at component 1
–   not evaluated(not run)
STOP at component 1
–   not evaluated
✓passes gate ×misses gate –stopped after earlier failureFigure 5:Heterogeneous failure modes across two reasoning tasks(connectivity rows: Option B,t inject=2,β=20; Sudoku: key-quality
screen only). On connectivity-2, K and oracle M pass; the deployable cold-pred pipeline then fails at V (−4.09pp vs the−2pp gate). On
Sudoku, the encoder itself blocks at K (quality ratio≈0.42vs the≥0.85gate,gt best≈0.30candidate-pool ceiling).
500 1000 1500 2000 2500 3000
step0.750.850.95score
gate 0.85quality_ratio 0.95
gt_best 0.88
top1 0.83A. Connectivity encoder validation
0.0 0.1 0.2 0.3 0.4
Hamming similarity0.00.10.20.30.4pair shareiid 1/9 q95 0.185B. Sudoku solved-board similarity
Figure 6: (A) Connectivity-2 encoder validation:quality ratio
clears the≥0.85gate by step 500, saturating near0.95against
the per-batchgt best≈0.88ceiling. (B) Sudoku solved-board
Hamming similarity (999,000 pairs): unimodal at iid1/9, q95=
0.185,gt best≈0.30— the candidate-pool side of Sudoku’s K-
screen failure (see Section 5 for the encoder-side limit).
pipelines into retriever- and generator-side diagnostic met-
rics for failure attribution. Our three-component decomposi-
tion (key quality / warm-start mechanism / stored-value qual-
ity) adapts this logic to retrieval-warmediterative inference;
we do not claim the decomposition, the partial-noise warm-
start mechanism, or the oracle-memory upper bound as novel
primitives. The slot that differs from the LLM-RAG settingis the warm-start mechanism — the retrieved vector enters a
dynamical update, not a context window. What is distinctive
across these prior threads is the application: we are not aware
of any that tests its warm-start against a constant baseline, a
per-query random control, and a shuffled-oracle arm, or that
reports a component-level decomposition across two reason-
ing tasks under iterative inference.
7 Conclusion
We introduced a five-arm ablation methodology — ora-
cle, best-constant, per-query-random, shuffled, and aligned
— for attributing the behaviour of retrieval-warmed itera-
tive inference. The full suite is exercised on connectivity-
2 through an IRED energy-based diffusion backbone aug-
mented with a Modern Hopfield trajectory memory; the
same diagnostic logic is applied to Sudoku as a key-quality
screen. On connectivity-2 the suite isolates per-graph key-
value alignment as the dominant lever: a+35pp aligned-
vs-shuffled-oracle swing that a tuned constant prior and per-
query stochastic initialisation together cannot explain. The
same decomposition identifies the first blocking component
of the deployable pipeline on each task: stored-value quality
on connectivity-2, key quality on Sudoku under the current
setup. The oracle arms are diagnostic upper bounds, not de-
ployable results; their value is in attribution. Coverage be-
yond these two tasks remains future work. Where a single
end-to-end accuracy number reports only that retrieval warm-
starting helped or failed, the methodology names the compo-
nent — key quality, warm-start mechanism, or stored-value
quality — that carried it: a localised verdict no aggregate
score can give.
References
[Ambrogioni, 2024 ]Luca Ambrogioni. In search of dis-
persed memories: Generative diffusion models are asso-
ciative memory networks.Entropy, 26(5):381, 2024.
[Blattmannet al., 2022 ]Andreas Blattmann, Robin Rom-
bach, Kaan Oktay, Jonas M ¨uller, and Bj ¨orn Ommer. Semi-

parametric neural image synthesis. InAdvances in Neural
Information Processing Systems (NeurIPS), 2022.
[Chalumeauet al., 2025 ]Felix Chalumeau, Refiloe Shabe,
Noah De Nicola, Arnu Pretorius, Thomas D. Barrett, and
Nathan Grinsztajn. Memory-enhanced neural solvers for
routing problems. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS), 2025.
[Duet al., 2022 ]Yilun Du, Shuang Li, Joshua B. Tenen-
baum, and Igor Mordatch. Learning iterative reasoning
through energy minimization. InProceedings of the 39th
International Conference on Machine Learning (ICML),
2022.
[Duet al., 2024 ]Yilun Du, Jiayuan Mao, and Joshua B.
Tenenbaum. Learning iterative reasoning through energy
diffusion. InProceedings of the 41st International Con-
ference on Machine Learning (ICML), 2024.
[Hoet al., 2020 ]Jonathan Ho, Ajay Jain, and Pieter Abbeel.
Denoising diffusion probabilistic models. InAdvances in
Neural Information Processing Systems (NeurIPS), 2020.
[Khoslaet al., 2020 ]Prannay Khosla, Piotr Teterwak, Chen
Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron
Maschinot, Ce Liu, and Dilip Krishnan. Supervised con-
trastive learning. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS), 2020.
[Menget al., 2022 ]Chenlin Meng, Yutong He, Yang Song,
Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Er-
mon. SDEdit: Guided image synthesis and editing with
stochastic differential equations. InInternational Confer-
ence on Learning Representations (ICLR), 2022.
[Phamet al., 2025 ]Bao Pham, Gabriel Raya, Matteo Negri,
Mohammed J. Zaki, Luca Ambrogioni, and Dmitry Kro-
tov. Memorization to generalization: Emergence of diffu-
sion models from associative memory, 2025.
[Ramsaueret al., 2021 ]Hubert Ramsauer, Bernhard Sch ¨afl,
Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas
Adler, Lukas Gruber, Markus Holzleitner, Milena
Pavlovi ´c, Geir Kjetil Sandve, Victor Greiff, David Kreil,
Michael Kopp, G ¨unter Klambauer, Johannes Brandstetter,
and Sepp Hochreiter. Hopfield networks is all you need.
InInternational Conference on Learning Representations
(ICLR), 2021.
[Ruet al., 2024 ]Dongyu Ru, Lin Qiu, Xiangkun Hu, Tian-
hang Zhang, Peng Shi, Shuaichen Chang, Cheng Ji-
ayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao
Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo
Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang.
RAGChecker: A fine-grained framework for diagnosing
retrieval-augmented generation. InAdvances in Neural
Information Processing Systems (NeurIPS) Datasets and
Benchmarks Track, 2024.
[Scholz and Turner, 2025 ]Jonas Scholz and Richard E.
Turner. Warm starts accelerate conditional diffusion, 2025.
[Sheyninet al., 2022 ]Shelly Sheynin, Oron Ashual, Adam
Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, and
Yaniv Taigman. KNN-Diffusion: Image generation via
large-scale retrieval, 2022.[Sivakumaret al., 2026 ]Aswini Sivakumar, Vijayan Sugu-
maran, and Yao Qiang. RAG-X: Systematic diagnosis of
retrieval-augmented generation for medical question an-
swering, 2026.
[Songet al., 2021 ]Jiaming Song, Chenlin Meng, and Ste-
fano Ermon. Denoising diffusion implicit models. InInter-
national Conference on Learning Representations (ICLR),
2021.
[Songet al., 2023 ]Yang Song, Prafulla Dhariwal, Mark
Chen, and Ilya Sutskever. Consistency models. InPro-
ceedings of the 40th International Conference on Machine
Learning (ICML), 2023.
[Wanget al., 2019 ]Po-Wei Wang, Priya L. Donti, Bryan
Wilder, and J. Zico Kolter. SATNet: Bridging deep learn-
ing and logical reasoning using a differentiable satisfiabil-
ity solver. InProceedings of the 36th International Con-
ference on Machine Learning (ICML), 2019.
[Xuet al., 2019 ]Keyulu Xu, Weihua Hu, Jure Leskovec, and
Stefanie Jegelka. How powerful are graph neural net-
works? InInternational Conference on Learning Rep-
resentations (ICLR), 2019.