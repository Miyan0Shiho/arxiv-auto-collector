# Relational Synthesis: Structure-Mediated Concatenative Synthesis for Foley and Retrieval-Augmented Audio Generation

**Authors**: Keren Shao, Ayaka Kawano, Shlomo Dubnov

**Published**: 2026-10-05 04:14:08

**PDF URL**: [https://arxiv.org/pdf/2610.05768v1](https://arxiv.org/pdf/2610.05768v1)

## Abstract
We ask: given a retrieved source audio $S$ and a separate reference audio $R$, can we synthesize novel audio $Y$ out of this pair $(S,R)$ such that $Y$ remains acoustically consistent with $S$, while not persistently copying segments of $S$ or $R$? The first clause is a well-known goal in Foley audio production, and the second is a well-known issue in neural RAG when $S$ and $R$ are naively injected into neural generators. We show that both clauses can be addressed simultaneously using a method we coin relational synthesis, a variation of concatenative synthesis where target cost is replaced by a relational Gromov-like structural cost. Rather than imitating the content of $R$, relational synthesis exploits it from the "other side of the hill": it transfers the temporal structure and directed amplitude motion of $R$ to reorganize and concatenate the grains of $S$ in a novel manner that protects $S$'s acoustic information. Our experiments show that relational synthesis integrates naturally with neural RAG and produces Foley audio that performs well on metrics measuring temporal agreement, acoustic fidelity, and leakage persistence, while maintaining distribution-level quality and text alignment.

## Full Text


<!-- PDF content starts -->

RELATIONAL SYNTHESIS: STRUCTURE-MEDIATED CONCATENATIVE SYNTHESIS FOR
FOLEY AND RETRIEV AL-AUGMENTED AUDIO GENERATION
Keren Shao, Ayaka Kawano, Shlomo Dubnov
University of California San Diego, La Jolla, CA, USA
ABSTRACT
We ask: given a retrieved source audioSand a separate reference au-
dioR, can we synthesize novel audioYout of this pair(S, R)such
thatYremains acoustically consistent withS, while not persistently
copying segments ofSorR? The first clause is a well-known goal in
Foley audio production, and the second is a well-known issue in neu-
ral RAG whenSandRare naively injected into neural generators.
We show that both clauses can be addressed simultaneously using
a method we coinrelational synthesis, a variation of concatenative
synthesis where target cost is replaced by a relational Gromov-like
structural cost. Rather than imitating the content ofR, relational
synthesis exploits it from the “other side of the hill”: it transfers the
temporal structure and directed amplitude motion ofRto reorganize
and concatenate the grains ofSin a novel manner that protectsS’s
acoustic information. Our experiments show that relational synthesis
integrates naturally with neural RAG and produces Foley audio that
performs well on metrics measuring temporal agreement, acoustic fi-
delity, and leakage persistence, while maintaining distribution-level
quality and text alignment.
Index Terms—relational synthesis, concatenative synthesis,
Foley sound synthesis, retrieval-augmented generation, temporal
control
1. INTRODUCTION
Foley synthesis must satisfy two coupled demands:what the sound
is, for example event class, material, timbre, acoustic identity, and
when and how it happens, for example onset/offset, rhythm, inten-
sity, temporal structure. Temporal-event and vocal-imitation control
make this split explicit [1, 2]. Retrieval-augmented audio genera-
tion (RAG) supplies complementary acoustic evidence that may be
difficult to specify in text [3, 4, 5], but direct audio conditioning
introduces a different risk: useful retrieved/reference material may
persist as copied audio rather than merely guide generation [6].
LetSbe the retrieved source,Ran independent temporal refer-
ence, andYthe output. We seek
A(Y)≈ A(S),T(Y)≈ T(R),Π(Y, S),Π(Y, R)small.
HereAdenotes event-class acoustic identity,Ttemporal organiza-
tion, andΠpersistent local replication. ThusSshould supply what
the event sounds like,Rshould supply how it unfolds, and neither
waveform should simply survive insideY.
Concatenative synthesis is attractive because it recombines
source material rather than parametrically injectingSorRdirectly
[7], making theΠconstraints comparatively easier to satisfy. The
obvious candidate is pointwise target matching, but this creates a
granularity dilemma. Fine grains retain enough degrees of freedom
to followT(R), yet force local slices from potentially differentacoustic spaces, for example environmental sound and vocal imita-
tion, to match directly; differences in local dynamics and variance
can make onset/offset and joins awkward, while larger reference
structure is ignored. Longer segments preserve more source-native
continuity and therefore favorA(S), but remain local matches, sac-
rifice framewise temporal freedom as segments lengthen, and let
longer contiguous spans ofSsurvive, working againstΠ.
We therefore changehow grains are matched.Relational syn-
thesis1keeps framewise source selection, but replaces reference-
to-source pointwise matchingd(r t, sxt)with constraints on within-
domain relations. Crucially, this makes theA-Ttrade-off less rigid:
recurrence structure fromRspecifies non-local similarity and dis-
similarity patterns, while relational amplitude structure ofRtrans-
fers the relative strength and direction of rises and falls without re-
quiring source and reference envelopes to match in absolute value,
leaving room for the solver to retain acoustic fidelity when select-
ing grains ofS. The same framewise search can therefore pursue
A(Y)≈ A(S)andT(Y)≈ T(R)together while recombination
continues to limit persistent copy-through,Π(Y, S)andΠ(Y, R).
To solidly realize this synthesis principle, we formulate weighted re-
currence structures with three models: contextual recurrence graph
(CRG), factor oracle (FO), and variable-Markov oracle (VMO). We
then specify the relational amplitude structure with directed ampli-
tude motion (DAM), and solve the resulting grain-selection opti-
mization with an annealed discrete solver.
2. METHODS
All variants implement the same coupling principle: a methodM
defines relational pressures ofRand a framewise path through
Srealizes them acoustically. LetS= (s 1, . . . , s M)andR=
(r1, . . . , r N)be 180-ms source/reference grains at 100-ms hop. The
synthesis variablex t∈ {1, . . . , M}selects the source grain real-
izing reference timet, andX M= OLA(s x1, . . . , s xN). For any
graing, the shared heuristic acoustic embedding is
z=ϕ(g) = norm[ MFCC, ∆MFCC,Env 8,logRMS],
with pooled MFCC/delta trajectories, a coarse normalized envelope,
and log RMS, followed by standardization andℓ 2normalization.
This embedding consists of handcrafted acoustic features commonly
used in acoustic modeling and can be readily replaced by, e.g., mod-
ern self-supervised acoustic embeddings. We leave this exploration
to future work. Letκ S(i, j)∈[0,1]denote the cosine similarity
between the embeddingsz(s i)andz(s j). Each reference structure
produces positive and negative edgesE+, E−and weightsw+, w−.
1Audio examples are available athttps://smoothken.github.
io/relational_synth_demo/.
arXiv:2610.05768v1  [cs.SD]  5 Oct 2026

Text Prompt 
Source Reference 
Synthesized 
Neural GeneratorNeural Output 
Relational Pointwise Concatenative (Baseline)Fig. 1. Relational synthesis pipeline. ReferenceRspecifies temporal relations while sourceSsupplies selectable acoustic material. Method
Mreorganizes source grains into carrierX M, which conditions pretrained generatorG θtogether with promptPto produce neural output
Y. In our experiments,Ris a vocal imitation paired with targetTof the same class asS.
All three recurrence structure specifiers, CRG, FO, and VMO, share
Lrel(x) =X
(t,u)∈E+w+
tu(1−κ S(xt, xu))2
+λ−X
(t,u)∈E−w−
tu[κS(xt, xu)−m]2
+.(1)
Hereλ −≥0weights negative relations andm∈[0,1]is their
similarity margin. We useλ −=.35andm=.55. ThusRde-
termineswhich times should relate, while the solver uses source
self-similarity to realize those relations without a reference-source
pointwise acoustic distance. This construction is Gromov-like in that
incomparable observations are coupled through intra-domain rela-
tions rather than a pointwise ground metric [8]. To keep the present
scope focused, we restrict the formulation to reference-side rela-
tional structure rather than jointly modeling and matching relational
structures in bothSandR, as in full-fledged Gromov-Wasserstein.
2.1. Contextual Recurrence Graph
The most direct relational structure is a contextual recurrence graph
(CRG): repeated trajectories appear as off-diagonal paths in audio
self-similarity matrices, motivating recurrence scores based on lo-
cal diagonal context rather than isolated frame pairs [9, 10]. Let
κR(t, u)be cosine similarities among the reference embeddings.
With contextual radiusrand diagonal decay scaleσ, the contextual
recurrence significanceΓ R(t, u)is computed as a weighted average
of similarities along the local diagonal.
ΓR(t, u) =Pr
δ=−re−|δ|/σκR(t+δ, u+δ)Pr
δ=−re−|δ|/σ.(2)
After excluding|t−u| ≤r,E+
t= TopKuΓR(t, u)andE−
t=
BottomK uΓR(t, u), withw+
tu∝Γ R(t, u)pandw−
tu∝(1−
ΓR(t, u))p, wherepcontrols concentration on stronger edges. We
set(r, σ, K, p) = (4,2,8,2)in experiments. Diagonal averaging
asks not merely whethertanduare similar, but whether their sur-
rounding contexts evolve similarly.
2.2. Factor-Oracle Graph
Factor oracles provide a sparse variable-length view of recurrence:
the suffix link of timesteptpoints to a timestep recognizing the
longest repeated suffix (LRS) ending att[11, 12]. Intuitively, the
length of such repeated context provides a natural measure of rela-
tional evidence.
Because FOs operate on symbolic sequences while our audio
embeddings are continuous-valued, we quantize the reference em-
beddings using greedy exemplar clustering,
Fig. 2. Out-of-dataset example with shakuhachi sourceSand
humpback-whale referenceR. Raw source conditioning exhibits
persistent source copying; pointwise concatenation reduces copying
but poorly followsR’s temporal envelope. FO relational synthe-
sis instead transfers the reference pattern while reorganizing source
grains. Red boxes mark corresponding envelope structure.
symbolt=(
arg max jκR(t, ej),max jκR(t, ej)≥θ FO,
new symbol,otherwise.
wheree jdenotes the current exemplar time of clusterj. We use
θFO= 0.8.
Once the clusters are formed, we construct the factor oracle on
the symbolic sequence. Letsfx(t)denote the suffix-linked timestep
andℓ tthe LRS length att. Whenℓ t≥ℓmin= 3, FO specifies the
sparse positive relationE+
t={sfx(t)}with
w+
t=
min
1,ℓt
ℓ0.9qp
,(3)
whereℓ 0.9qis the90th percentile of valid LRS lengths and robustly
saturates very long repeats. Negative relations retain the contextual-
recurrence construction of CRG. FO therefore concentrates positive
relational pressure where the symbolic sequence provides evidence
of non-trivial repeated context.
2.3. Variable-Markov-Oracle Graph
FO separates acoustic quantization from oracle construction. VMO
instead determines acoustic clusters during oracle construction,
jointly coupling symbolization with repeated-context discovery
[13, 14]. Its clustering thresholdθ, rather than being fixed be-
forehand as in FO, is selected via information rate optimization
θ⋆= arg max θIR(Q θ), whereIRis information rate andQ θis
the resulting VMO symbol sequence at thresholdθ. In practice, this
is usually done via grid searchθ∈[0.05,1]with0.05increments
and selecting the run with the bestIR, rather than in a continuously
differentiable manner with respect toθ.

To compare FO and VMO consistently, and in particular to iso-
late the effect of separate greedy clustering versus oracle-integrated
clustering, we use the same definitions ofE+
t,w+
t,E−
t, andw−
t
as in FO. In other words, CRG establishes relations through fixed-
length context; FO introduces variable-length context through the
longest repeated suffix; and VMO further enriches variable-order
context with clusters optimized during oracle construction, at the
cost of greater algorithmic complexity. These are parallel relational
models rather than successive refinements: VMO is not assumed to
dominate FO or CRG.
2.4. Directed Amplitude Motion
A similarity graph can have a hard time distinguishing opposite am-
plitude motions, such as natural onset- and offset-like trajectories,
quiet→loud from loud→quiet.
across time-scales. To complement these recurrence relations, we
therefore exploit another prominent relational structure of the wave-
form: directed amplitude motion. Let the log-RMS coordinates be
aR
t= logRMS(r t), aS
i= logRMS(s i).
In the experiments,H={1,4,16,32}. For each lagh∈H, define
the log-amplitude change
∆h(t) =a t+h−at,
and letbFhdenote the empirical CDF of such changes. Its quan-
tile coordinate is
qh(t) =bFh(∆h(t)).
Intuitively,qR
h(t)≈1indicates an unusually strong rise in the
reference at timet,qR
h(t)≈0an unusually strong fall, andqR
h(t)≈
1
2a typical motion.
Consequently, for the synthesis to inherit the relative amplitude
motion ofR, we want the motion betweenx t+handx t, evaluated in
the source as
qS
h(i, j) = bFS
h(aS
j−aS
i),
to closely follow therelative extremenessqR
h(t)for allh∈H
andt. In other words, we want
Ldam=1
|H|X
h∈HX
thqS
h(xt, xt+h)−qR
h(t)−εi2
+.(4)
Hereε=.1is an insensitivity tolerance in percentile space. No
reference/source amplitude is matched directly.
2.5. Annealed Discrete Solver
Eqs. (1) and (4) induce a nonlocal, loopy interaction graph rather
than a fixed-order Markov chain, so standard Viterbi recursion does
not apply. We optimize
L=L rel+λdamLdam+L tr,(5)
whereL trdenotes the usual concatenative-synthesis transition reg-
ularization, including MFCC/delta-MFCC boundary compatibility
and simple continuation, repetition, and reuse preferences. We use
λdam= 1. TheCRG nodam ,FO nodam , andVMO nodam ablations
in the experiments section setλ dam= 0.We use eight parallel restarts and twelve random-order coordi-
nate sweeps, adding Gumbel perturbations whose temperature de-
creases linearly from0.08to zero, and retain the lowest-cost so-
lution. This nonlocal discrete search does not guarantee a global
optimum.
3. EXPERIMENTS
3.1. Setup
We use ESC-50 [15] and ESC-50-V oice [2]. ESC-50-V oice contains
vocal imitations, which we use as referencesR, for 31 classes of
environmental sounds in ESC-50, which we use as targetsT. The
targets are used only for evaluation; for eachT, we pair it with one
corresponding vocal imitationR.
ESC-50 divides each class into five folds. Two Fold-1 targets
per class are used to tune conditioning strength. All eight Fold-2 ex-
amples per class are then used as evaluation targets, while the source
Sis sampled independently, without replacement, from the same
class in Folds 3-5. Same-class source sampling serves as oracle re-
trieval, isolating synthesis from retrieval quality. The final evalu-
ation therefore contains31×8 = 248distinct(S, R, T)triplets.
All inputs are 5-s mono audio at 22.05 kHz. LetPdenote the
promptA realistic <class> environmental sound.
TrackType: SFX, where<class>is replaced at runtime by
the class label ofS.
We instantiateG θas the pretrained Stable Audio 3 Small-SFX
[16], using eight generation steps and three generation seeds per
triplet. On Fold-1, we sweep init-noise levels{0.25,0.50,0.75}and
select0.50for the final Fold-2 evaluation. No task-specific training
or additional conditioning network is introduced. The RAG condi-
tions are:None, using text prompt only;Ref, directly usingR;Mix,
using equal-RMSS+R;Raw, directly usingS;Shuffle, using block-
shuffledS;Pointwise-SR, using MFCC-nearest-grain-based point-
wise concatenative synthesis; andCRG/FO/VMO, using relational
synthesis. Unless marked with the nodam subscript, DAM is enabled
for all relational-synthesis conditions.
3.2. Metrics
LetYdenote the final neural output of the pipeline. Recall that we
started withT(Y)→ T(R)≈ T(T), thetemporalconstraints;
A(Y)→ A(T), theacousticconstraints; andΠ(Y, S),Π(Y, R),
theleakage/copyingconstraints. For the acoustic constraint, we
evaluate againstTrather thanS: by setup,Thas the same class
asS, while chasingA(S)may reward persistent copying. Since the
final stage is also a text-to-audio generator, we additionally report
standard prompt-alignment and distributional-quality checks.
Metric Computation
T:L T, LR↓Smoothed RMS-envelope L1 [1]
T:O T, OR↑Onset F1 (50 ms)
A:M T↑Log-mel cosine [6]
A:C T↑CLAP audio cosine [17]
A:KL T↓PANN KL [18]
Π:R2s
S, R2s
R Best-match 2-s log-mel cosine
Π:C S, CR CLAP audio cosine
Prompt:C P↑CLAP audio–text cosine
Distribution:IS↑PANN Inception Score [18]
Distribution:
FDV/C/P ↓VGGish/CLAP/PANN Fr ´echet distance [19]
Leakage similarities are diagnostic rather than monotone objec-
tives: high similarity is undesirable, but minimizing similarity itself
is not the goal. Pair-level metrics first average the three neural seeds
for each(S, R, T)triplet before aggregation.

Table 1. Stable Audio 3 results on 248 evaluation triplets. Pair-level metrics first average the three neural seeds of each(S, R, T)triplet;
KLT,IS, andFD V/C/P are set-level.Boldmarks the best non-shortcut source-carrier result. Underlining marks a numerical winner that is
principally a shortcut or control bound.†marks persistence of a directly exposed waveform. Arrows indicate the favorable direction for each
evaluation role; Leakage metrics are diagnostic rather than minimization objectives, so numerical winners are not highlighted.
Temporal Acoustic Leakage Prompt Distribution
ConditionL T↓LR↓OT↑OR↑MT↑CT↑KL T↓CS↓R2s
S↓CR↓R2s
R↓C P↑IS↑FD V/C/P ↓
None .086 .094 .354 .305 .913 .540 1.806 .542 .167 .177 .118 .246 8.83 8.82/.413/52.04
Direct reference-audio controls
RefR.083 .059 .402 .421 .939 .524 2.196 .511 .170 .243 .365†.227 7.9711.24/.405/56.74
MixS+R.080 .084 .395 .347 .927 .562 1.727 .590 .303 .198 .196†.240 9.21 7.49/.354 /47.31
Source-carrier conditions – central comparison
RawS.088 .095 .380 .322 .923 .552 1.865 .596 .353†.190 .137 .234 9.34 7.53/.354/47.34
Shuffle .091 .098 .378 .319 .921 .5471.733.588 .299 .196 .137 .224 8.49 6.70/.371/46.65
Pointwise-SR .095 .101 .389 .335 .921 .523 2.011 .552 .286 .191 .177 .214 8.47 6.79/.372/47.22
CRG (ours) .096 .097.394.343 .921 .515 2.110 .545 .307 .187 .151 .225 8.636.40/.389/47.35
FO (ours).087 .086.379.343.925.5511.737 .592 .299 .195 .144.2378.70 7.29/.365/45.66
VMO (ours) .087 .087 .389 .341.926.543 1.764 .584 .300 .194 .145 .2338.807.10/.368/46.54
Table 2. DAM ablation on the same 248 triplets. Eachnodamcondition removes Eq. (4). All other recurrence structure is unchanged.Bold
marks the favorable value within each matched pair.FD V/C/P denotes VGGish/CLAP/PANN Fr ´echet distance.
Temporal Acoustic Leakage Prompt Distribution
ConditionL R↓OR↑C T↑R2s
S↓R2s
R↓C P↑KL T↓IS↑FD V/C/P ↓
CRG nodam .110 .328 .491 .310 .141 .216 2.197 8.23 6.62/.410/50.14
CRG.097 .343 .515.307 .151.225 2.110 8.63 6.40/.389/47.35
FOnodam .096 .320 .548 .306 .142 .233 1.8218.82 7.18/.362/46.59
FO.086 .343 .551.299 .144.237 1.7378.70 7.29/.365/45.66
VMO nodam .096 .324.546.294 .139 .230 1.7709.08 6.79/.359/46.49
VMO.087 .341.543 .300 .145.233 1.7648.80 7.10/.368/46.54
3.3. Results and Discussion
We will read the results in the same order as the evaluation logic.
1. Temporal transferT(Y)→ T(R)≈ T(T):The central
question is whether use of relational information fromRcan im-
prove temporal organization when the generator still receives acous-
tic material only fromS. Relative to the source-carrier controls,
all three relational constructions increase reference-onset agreement:
OR=.322/.319/.335for Raw/Shuffle/Pointwise-SR, compared
with.343/.343/.341for CRG/FO/VMO. FO gives the clearest com-
plementary Event-L1 result: relative to Raw,L Rdecreases from
.095to.086whileO Rrises from.322to.343. Both changes are
significant (paired 95% CIs, FO−Raw:∆L R∈[−.012,−.007],
∆OR∈[.008, .034]). The direct-reference controls confirm that
still stronger timing can be obtained whenRitself is exposed to the
generator (L R=.059, O R=.421), but those conditions have ac-
cess to different acoustic information and serve as reference-access
controls rather than the central source-carrier comparison.
2. Acoustic-fidelityA(Y)→ A(T):The useful temporal gain
should preserve target-class acoustic fidelity carried bySand re-
quested byP. FO is the most stable relational realization on this
axis: Raw→FO essentially preserves target CLAP (.552→.551),
slightly raises target log-mel similarity (.923→.925), preserves
prompt CLAP (.234→.237), and improves PANN KL (1.865→
1.737). Thus the timing gain does not require moving the generated
event identity toward the vocal reference.
3. Persistent LeakageΠ(Y, S),Π(Y, R):The strongest evidence
for the structural bottleneck appears when timing and persistence are
read together. FO reduces local source persistence relative to Raw,
R2s
S=.353→.299(paired 95% CI[−.068,−.039]), while its ref-erence persistence remains near source-carrier controls:C R=.195
andR2s
R=.144for FO, compared with.190/.137for Raw and
.196/.137for Shuffle. Direct Ref, by contrast, undesirably reaches
R2s
R=.365. Hence improved reference timing does not require
passing the reference waveform through the generator, while the
source is also not simply replayed locally.
4. Prompt and distributional bounds.FO preserves the pretrained
generator’s semantic operating point within the source-carrier com-
parison: prompt CLAP is essentially unchanged from Raw (.234→
.237). Its VGGish FAD improves from7.53to7.29, PANN Fr ´echet
distance from47.34to45.66, and PANN KL from1.865to1.737;
CLAP FAD changes moderately from.354to.365, and IS from9.34
to8.70. These set-level quantities are therefore treated as bounds
rather than the primary optimization target: the desired temporal,
acoustic, and leakage behavior is not purchased by a large collapse
in semantic or distributional behavior.
5. Directed-amplitude-motion mechanism ablation.The DAM
ablation is deliberately internal to each relational structure. En-
abling DAM changesL Rfrom.110→.097,.096→.086, and
.096→.087for CRG, FO, and VMO, whileO Rchanges from
.328→.343,.320→.343, and.324→.341. All six temporal
deltas have paired 95% CIs excluding zero. Acoustic and prompt
alignment are generally preserved or improved across the matched
pairs, whereas leakage changes are comparatively small and the set-
level effects are mixed. This is consistent with the intended reduction
of the acoustic-temporal trade-off: DAM strengthens temporal trans-
fer without forcing cross-domain envelope values to coincide, while
the acoustic fidelity measures improve rather than being traded away.
Its benefit is therefore complementary to recurrence structure.

4. REFERENCES
[1] Y . Chung, J. Lee, and J. Nam, “T-FOLEY: A Con-
trollable Waveform-Domain Diffusion Model for
Temporal-Event-Guided Foley Sound Synthesis,” in
Proc. IEEE ICASSP, pp. 6820–6824, 2024, doi:
10.1109/ICASSP48485.2024.10447380.
[2] Y . Okamoto, K. Imoto, S. Takamichi, R. Nagase, T.
Fukumori, and Y . Yamashita, “Environmental Sound
Synthesis from V ocal Imitations and Sound Event La-
bels,” inProc. IEEE ICASSP, pp. 411–415, 2024, doi:
10.1109/ICASSP48485.2024.10446906.
[3] Y . Yuan, H. Liu, X. Liu, Q. Huang, M. D. Plumbley,
and W. Wang, “Retrieval-Augmented Text-to-Audio Gener-
ation,” inProc. IEEE ICASSP, pp. 581–585, 2024, doi:
10.1109/ICASSP48485.2024.10447898.
[4] M. Yang, B. Shi, M. Le, W.-N. Hsu, and A. Tjan-
dra, “Audiobox TTA-RAG: Improving Zero-Shot and Few-
Shot Text-To-Audio with Retrieval-Augmented Genera-
tion,” inProc. Interspeech, pp. 1243–1247, 2025, doi:
10.21437/Interspeech.2025-1032.
[5] P. Fang, Y . He, Y . Xing, Q. Chen, S.-N. Lim, and H. Yang,
“AC-Foley: Reference-Audio-Guided Video-to-Audio Synthe-
sis with Acoustic Transfer,” inProc. ICLR, 2026.
[6] D. Bralios, G. Wichern, F. G. Germain, Z. Pan, S.
Khurana, C. Hori, and J. Le Roux, “Generation or
Replication: Auscultating Audio Latent Diffusion Mod-
els,” inProc. IEEE ICASSP, pp. 1156–1160, 2024, doi:
10.1109/ICASSP48485.2024.10447705.
[7] D. Schwarz, “Corpus-Based Concatenative Synthesis,”IEEE
Signal Process. Mag., vol. 24, no. 2, pp. 92–104, 2007, doi:
10.1109/MSP.2007.323274.
[8] S. Cohen, G. Luise, A. Terenin, B. Amos, and M. P. Deisen-
roth, “Aligning Time Series on Incomparable Spaces,” inProc.
AISTATS, PMLR 130, pp. 1036–1044, 2021.
[9] M. M ¨uller and F. Kurth, “Enhancing Similarity Matrices for
Music Audio Analysis,” inProc. IEEE ICASSP, pp. 437–440,
2006, doi: 10.1109/ICASSP.2006.1661199.
[10] J. Foote and M. Cooper, “Visualizing Musical Structure and
Rhythm via Self-Similarity,” inProc. ICMC, pp. 419–422,
2001.
[11] S. Dubnov, G. Assayag, and A. Cont, “Audio Oracle: A New
Algorithm for Fast Learning of Audio Structures,” inProc.
ICMC, pp. 224–227, 2007.
[12] A. Cont, S. Dubnov, and G. Assayag, “Guidage: A Fast Audio
Query Guided Assemblage,” inProc. ICMC, 2007.
[13] C.-i. Wang and S. Dubnov, “Guided Music Synthesis with Vari-
able Markov Oracle,” inProc. 3rd Int. Workshop on Musical
Metacreation (MUME), AAAI AIIDE, pp. 55–62, 2014, doi:
10.1609/aiide.v10i5.12767.
[14] C.-i. Wang and S. Dubnov, “Pattern Discovery from Audio
Recordings by Variable Markov Oracle: A Music Information
Dynamics Approach,” inProc. IEEE ICASSP, pp. 683–687,
2015, doi: 10.1109/ICASSP.2015.7178056.
[15] K. J. Piczak, “ESC: Dataset for Environmental Sound Classifi-
cation,” inProc. ACM Multimedia, pp. 1015–1018, 2015, doi:
10.1145/2733373.2806390.[16] Z. Evans, J. D. Parker, M. Rice, C. J. Carr, Z. Zukowski, J. Tay-
lor, and J. Pons, “Stable Audio 3,” arXiv:2605.17991, 2026.
[17] Y . Wu, K. Chen, T. Zhang, Y . Hui, T. Berg-Kirkpatrick, and
S. Dubnov, “Large-Scale Contrastive Language-Audio Pre-
training with Feature Fusion and Keyword-to-Caption Aug-
mentation,” inProc. IEEE ICASSP, pp. 1–5, 2023, doi:
10.1109/ICASSP49357.2023.10095969.
[18] Q. Kong, Y . Cao, T. Iqbal, Y . Wang, W. Wang, and M. D.
Plumbley, “PANNs: Large-Scale Pretrained Audio Neural Net-
works for Audio Pattern Recognition,”IEEE/ACM Trans. Au-
dio, Speech, Lang. Process., vol. 28, pp. 2880–2894, 2020, doi:
10.1109/TASLP.2020.3030497.
[19] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Fr ´echet
Audio Distance: A Reference-Free Metric for Evaluating Mu-
sic Enhancement Algorithms,” inProc. Interspeech, pp. 2350–
2354, 2019, doi: 10.21437/Interspeech.2019-2219.