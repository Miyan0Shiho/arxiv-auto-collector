# LLMs Should Express Uncertainty Explicitly

**Authors**: Junyu Guo, Shangding Gu, Ming Jin, Costas Spanos, Javad Lavaei

**Published**: 2026-04-07 01:20:29

**PDF URL**: [https://arxiv.org/pdf/2604.05306v1](https://arxiv.org/pdf/2604.05306v1)

## Abstract
Large language models are increasingly used in settings where uncertainty must drive decisions such as abstention, retrieval, and verification. Most existing methods treat uncertainty as a latent quantity to estimate after generation rather than a signal the model is trained to express. We instead study uncertainty as an interface for control. We compare two complementary interfaces: a global interface, where the model verbalizes a calibrated confidence score for its final answer, and a local interface, where the model emits an explicit <uncertain> marker during reasoning when it enters a high-risk state. These interfaces provide different but complementary benefits. Verbalized confidence substantially improves calibration, reduces overconfident errors, and yields the strongest overall Adaptive RAG controller while using retrieval more selectively. Reasoning-time uncertainty signaling makes previously silent failures visible during generation, improves wrong-answer coverage, and provides an effective high-recall retrieval trigger. Our findings further show that the two interfaces work differently internally: verbal confidence mainly refines how existing uncertainty is decoded, whereas reasoning-time signaling induces a broader late-layer reorganization. Together, these results suggest that effective uncertainty in LLMs should be trained as task-matched communication: global confidence for deciding whether to trust a final answer, and local signals for deciding when intervention is needed.

## Full Text


<!-- PDF content starts -->

LLMs Should Express Uncertainty Explicitly
Junyu Guo∗
University of California, BerkeleyShangding Gu
University of California, BerkeleyMing Jin
Virginia Tech
Costas Spanos
University of California, BerkeleyJavad Lavaei
University of California, Berkeley
Abstract
Large language models are increasingly deployed in settings where uncertainty
must drive downstream decisions such as abstention, retrieval, and verification,
yet most existing methods treat uncertainty as a latent quantity to estimate after
generation rather than as a signal the model is trained to communicate. We instead
study uncertainty as aninterface: when should it be exposed, and in what form
should downstream systems consume it? We compare two complementary inter-
faces: aglobalinterface, in which the model verbalizes a calibrated confidence
score for its final answer, and alocalinterface, in which the model emits an explicit
<uncertain> marker during reasoning when it enters a high-risk state. The two
interfaces produce different but complementary gains. The verbalized-confidence
interface substantially improves calibration, sharply reduces overconfident errors,
outperforms strong calibration baselines, and yields the strongest overall Adaptive
RAG controller while using retrieval more selectively. The reasoning-time interface,
by contrast, makes previously silent failures visible during generation, improves
wrong-answer coverage, and provides an effective high-recall intervention signal
for retrieval control. Our analyses also explain why these gains arise: epistemic-
versus-aleatoric error analysis shows that calibration training shifts errors away
from confident epistemic failures toward explicitly uncertain behavior; PCA shows
that calibration sharpens an existing confidence manifold; and CKA together with
parameter-drift analysis shows that verbal confidence is implemented largely as a
geometry-preserving confidence-mapping refinement, whereas the reasoning-time
interface induces a stronger late-layer rewrite. Together, these results argue that
effective uncertainty in LLMs should be trained as task-matched communication:
global confidence for deciding whether a final answer should be trusted, and local
reasoning-time signaling for deciding when the model needs intervention.
1 Introduction
Large language models are probabilistic systems and therefore inevitably carry uncertainty throughout
generation He et al. [2025], Vashurin et al. [2025]. In practice, that uncertainty matters because
downstream systems increasingly need toacton it: whether to abstain, retrieve evidence, call a tool,
verify an answer, or trust the model. For such decisions, it is not enough for uncertainty to exist
somewhere inside the model. It must be exposed in a form that is legible to users and usable by
controllers. The central problem is therefore not only how tomeasureuncertainty, but how to make it
an actionable interface for decision-making.
This distinction is important because most current methods still treat uncertainty as a latent quantity
to recover after the fact. Recent research suggest that hesitation-like tokens and other high-entropy
∗Corresponding author: junyuguo24@berkeley.edu
Preprint.arXiv:2604.05306v1  [cs.LG]  7 Apr 2026

Figure 1: Overview of the two uncertainty interfaces.
transitions in reasoning traces correlate with internal uncertainty Wang et al. [2025]. Existing adaptive
retrieval systems similarly infer when to intervene from scalar confidence estimates, entropy statistics,
response features Jeong et al. [2024], Moskvoretskii et al. [2025], Su et al., Yao et al. [2025]. These
approaches are useful, but they leave a key visibility problem unresolved: downstream control often
depends on signals that are only indirectly inferred, so one must still ‘read between the lines’ of the
model’s behavior to decide what went wrong and whether intervention is needed.
Our motivation is to make uncertaintyexplicit. If a model emits a special uncertainty token during
reasoning, then uncertainty becomes visible as an event rather than a hidden pattern that must be
detected post-hoc. If a model verbalizes its final confidence, then uncertainty becomes visible as a
degree of belief rather than a score reconstructed from external probes or calibration heuristics. In
both cases, the goal is the same: to expose uncertainty in a form that lets us directly observe when the
model is unsure, how unsure it is, and how that signal should drive downstream action.
This perspective also changes how calibration should be understood. Calibration is often treated as a
property of the final scalar output, but for decision-making it is also a problem ofexpression: how
accurately does the model communicate its own reliability, and how faithfully does that expression
reflect the internal state that led to the answer? Recent work has shown that uncertainty-aware training
can improve reasoning and calibration even without gold labels Li et al., Wu et al. [2025], Zhao
et al., suggesting that uncertainty is not merely a diagnostic quantity but a trainable aspect of model
behavior. This motivates a stronger objective than post-hoc recalibration alone: we seek to improve
both thequalityof the model’s calibration and thevisibilityof how uncertainty is expressed.
This framing connects naturally to token-level control. Learned tokens can compress or package
complex behavior, as in gist tokens for prompt compression Mu et al. [2023], while recent work
on neologisms argues that new tokens can provide compact handles for controllability and self-
verbalization Hewitt et al. [2025b,a]. Related ideas also appear in reasoning, alignment, and retrieval
settings, where models are trained to mark particular reasoning states or actions explicitly Shao et al.
[2024], Zhang et al. [2024], Asai et al. [2023]. These results suggest that special tokens are not
merely formatting devices; they can serve as structured interfaces to internal computation. The same
idea may apply to uncertainty itself.
A practical caveat is that actionable uncertainty should not come at the expense of inference efficiency.
External intervention is only useful when the expected gain from retrieval or tool use outweighs its
cost Liu et al. [2024]. An effective uncertainty interface should therefore not only reveal when the
model is uncertain, but do so selectively enough to support efficient downstream control. This leads
to our central question:
How should uncertainty be exposed in LLMs so that it becomes useful for control?
2

Most existing work treats uncertainty as a property of thefinal output, for example by estimating a
confidence score after the model has already completed its answer. While useful, this view captures
only one mode in which uncertainty arises. In practice, uncertainty appears at least at two distinct
scales. First, a model may be uncertain about whether itsfinal answeris correct. Second, a model
may encounter uncertaintyduring the reasoning process, when it reaches a missing fact, ambiguous
evidence, or a fragile intermediate step. These two forms of uncertainty support different downstream
decisions and need not be exposed in the same way.
We therefore study uncertainty in LLMs as aninterface design problem: a global interface should
summarize final-answer reliability, while a local interface should expose intervention points before
the model fully commits to a trajectory. We instantiate these two regimes with verbalized confidence
for outcome uncertainty and an explicit <uncertain> marker for process uncertainty. The former is
trained to better align stated confidence with empirical correctness; the latter is trained to surface
fragile reasoning states as visible events that downstream systems can act on.
We implement both interfaces within a unified post-training framework and evaluate them in terms of
calibration quality, behavioral reliability, mechanism, and downstream retrieval control. Verbalized-
confidence training substantially improves calibration and suppresses overconfident errors while
preserving, and sometimes slightly improving, answer accuracy. Reasoning-time uncertainty training
converts many previously silent epistemic failures into explicit uncertainty events, substantially
increasing downstream coverage of wrong answers. Together, these results argue that actionable
uncertainty in LLMs is not a single output format, but a multi-scale design space spanning both
global reliability summaries and local intervention signals.
Contributions.Our main contributions are as follows.
1.We frame uncertainty in LLMs as aninterface design problemand distinguish two comple-
mentary interfaces: a global signal for deciding whether a final answer should be trusted,
and a local signal for deciding when intervention is needed during reasoning.
2.We show that verbalized confidence can serve as a strong global uncertainty interface. It
improves calibration, suppresses overconfident errors, and outperforms simpler calibration
baselines as a signal for selective prediction and retrieval control.
3.We show that reasoning-time <uncertain> signaling can serve as a strong local uncertainty
interface. It turns previously silent failures into visible intervention points and improves
downstream retrieval control relative to adaptive retrieval baselines.
4.We provide mechanistic evidence that the two interfaces operate differently internally. Verbal
calibration primarily sharpens how existing uncertainty is expressed, whereas reasoning-time
signaling induces a broader late-layer reorganization before uncertainty is emitted.
2 Preliminaries
We study uncertainty in LLMs as aninterfaceproblem: uncertainty is useful for downstream control
only when it is exposed in a form that external systems can act on. In this paper, we distinguish two
complementary interfaces: aglobalinterface that summarizes the reliability of the final answer, and a
localinterface that marks high-risk states during reasoning. Given an input question x, the model
generates a reasoning trajectory z1:T= (z 1, . . . , z T), with hidden states ht=fθ(ht−1, x, z <t),
t= 1, . . . , T . The final response induces an answer ˆy, and we write Y∈ {0,1} for its correctness
indicator. We assume that the hidden trajectory h1:Tcontains not only task-relevant semantic
information, but also latent uncertainty information about whether the current reasoning path is
reliable or likely to fail. Our goal is to train the model to expose this uncertainty explicitly.
Theglobal uncertainty interfaceproduces a scalar confidence after the trajectory is complete:
c=R global(h1:T), where c∈[0,1] is intended to summarize final-answer reliability, ideally
approximating P(Y= 1|h 1:T). Thelocal uncertainty interfaceproduces a step-level intervention
signal during reasoning: at=R local(ht)∈ {0,1} , where at= 1indicates that the model has entered
a high-risk reasoning state at step t. In our setting, this local signal is instantiated by emitting the string
<uncertain> . The two interfaces serve different roles. The global interface is a trajectory-level
summary suited to question-level control such as abstention or retrieval triggering. The local interface
is a reasoning-time signal suited to mid-trajectory intervention when the model reaches a fragile state.
3

Accordingly, we treat the two interfaces differently in the remainder of the paper. Section 3 focuses
on the global verbalized-confidence interface and develops its training-dynamics interpretation under
uncertainty-aware reinforcement learning. Section 4 studies the local <uncertain> interface as an
intervention-oriented uncertainty signal.
3 Global Uncertainty Interface via Verbalized Confidence
We first study uncertainty at theoutcome level, where the model produces a scalar estimate of the
correctness of its final answer. Our goal is not merely to improve calibration metrics, but to understand
how such a global uncertainty signal can be learned without degrading the underlying reasoning
process. Our central hypothesis is that outcome-level uncertainty can be implemented as areadout
of the reasoning trajectory, rather than a modification of the reasoning policy itself. Concretely,
given a reasoning trajectory with hidden states h1:T, the model learns a mapping h1:T7→c , where
capproximates the probability that the final answer is correct. If this mapping can be improved
independently of the token-level generation dynamics, then calibration quality can increase without
disrupting reasoning.
To test this hypothesis, we train the model with a simple confidence-aware reward: r(x, y, p) =p if
the final answer is correct and r(x, y, p) =−p otherwise. This directly rewards justified confidence
and penalizes overconfident errors, while remaining a purely outcome-level signal applied after the
full reasoning trajectory is completed.
GRPO as trajectory reweighting.Let πθ(z|x) denote the model distribution over complete
reasoning trajectories zfor input x, and let each trajectory induce both a final answer g(z) and a
verbalized confidence p(z)∈[0,1] . Under a small policy-improvement step, uncertainty-aware
reinforcement learning can be idealized as reweighting trajectories according to their reward: πθ′(z|
x)∝π θ(z|x) exp(η r(z;x)) , where η >0 is an effective step size. This tilted-distribution view
captures the first-order effect of GRPO: higher-reward trajectories gain probability mass, while
lower-reward ones are suppressed. A generic one-step comparison result is deferred to Appendix A;
the main text focuses only on the calibration-specific consequences.
Proposition 1(Confidence-aware trajectory compression).Under the tilted update, if two wrong
trajectories z1, z2satisfy p(z1)> p(z 2), then the more overconfident one is suppressed more strongly:
πθ′(z1|x)
πθ′(z2|x)<πθ(z1|x)
πθ(z2|x).(1)
Symmetrically, among correct trajectories, higher-confidence ones are relatively amplified. This is
the local path-level mechanism of calibration: GRPO compresses overconfident error trajectories
and shifts mass toward better-supported ones.
Latent-answer extraction under confidence-aware reweighting.The key question is why this
outcome-only reward can sometimes improve final prediction quality even though it neither supervises
intermediate reasoning steps nor introduces new external knowledge. Under the tilted-policy view,
GRPO cannot create new trajectory support; it can only reallocate probability mass over trajectories
that the pretrained model already assigns nonzero probability. The next theorem formalizes that
answer-level consequence.
Theorem 1(Latent-answer extraction under support-preserving reweighting).Fix an input x, let
Mθ(y|x) =P
z:g(z)=yπθ(z|x) be the answer-level probability mass of answer y, lety⋆be the
correct answer, and let ¯y̸=y⋆be any competing wrong answer. If every trajectory producing y⋆has
reward at leastaand every trajectory producing¯yhas reward at mostb, for constantsa > b, then
Mθ′(y⋆|x)
Mθ′(¯y|x)≥exp 
η(a−b)Mθ(y⋆|x)
Mθ(¯y|x).(2)
Moreover, the tilted update is support-preserving, so it cannot create a new correct trajectory that
was absent underπ θ; it can only amplify correct trajectories already latent in the pretrained model.
Interpretation.Proposition 1 and Theorem 1 together give the main takeaway of the theory section:
GRPO improves calibration by suppressing overconfident wrong paths, and it can improve final
answers only by increasing the probability of correct paths that were already latent in the pretrained
4

LOW MID HIGH
Confidence bin16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32Transformer layer0.07 0.35 0.57
0.09 0.34 0.57
0.12 0.32 0.56
0.14 0.32 0.54
0.14 0.37 0.49
0.13 0.41 0.46
0.14 0.35 0.51
0.16 0.25 0.60
0.13 0.19 0.68
0.10 0.16 0.74
0.08 0.14 0.78
0.08 0.12 0.80
0.09 0.11 0.80
0.09 0.11 0.80
0.10 0.09 0.81
0.11 0.08 0.81
0.12 0.09 0.79Correct
(n=191)
LOW MID HIGH
Confidence bin0.12 0.44 0.44
0.14 0.45 0.41
0.18 0.43 0.39
0.22 0.42 0.36
0.23 0.46 0.31
0.23 0.51 0.26
0.28 0.46 0.26
0.33 0.37 0.30
0.32 0.32 0.36
0.30 0.30 0.40
0.30 0.28 0.42
0.32 0.24 0.44
0.34 0.23 0.43
0.35 0.22 0.43
0.37 0.19 0.43
0.40 0.18 0.42
0.42 0.18 0.40Wrong
(n=309)
0.00.10.20.30.40.50.60.7
Probability mass in binBinned Logit Lens  base_llama3_late  (layers 16 32)
(a) Base
LOW MID HIGH
Confidence bin16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32Transformer layer0.06 0.43 0.52
0.05 0.37 0.58
0.06 0.32 0.62
0.07 0.32 0.61
0.08 0.36 0.56
0.09 0.41 0.50
0.12 0.38 0.50
0.15 0.30 0.55
0.13 0.28 0.59
0.10 0.30 0.60
0.10 0.33 0.57
0.11 0.36 0.53
0.12 0.41 0.47
0.13 0.47 0.39
0.15 0.51 0.34
0.17 0.52 0.31
0.18 0.52 0.30Correct
(n=191)
LOW MID HIGH
Confidence bin0.14 0.51 0.35
0.16 0.46 0.38
0.18 0.42 0.40
0.22 0.41 0.37
0.25 0.44 0.31
0.27 0.49 0.24
0.34 0.46 0.20
0.42 0.38 0.20
0.42 0.36 0.22
0.41 0.37 0.22
0.43 0.37 0.20
0.47 0.36 0.17
0.50 0.38 0.12
0.54 0.38 0.07
0.60 0.35 0.05
0.64 0.32 0.04
0.67 0.30 0.03Wrong
(n=309)
0.00.10.20.30.40.50.6
Probability mass in binBinned Logit Lens  calibrated_late  (layers 16 32)
 (b) Calibrated
Figure 2: Mechanistic evidence for verbalized confidence calibration. Calibration sharpens late-layer
confidence routing and yields a cleaner final-layer confidence manifold.
model. The full proof, the generic one-step log-ratio result, the answer-margin corollary, and the
specialization of Theorem 1 to the verbal-confidence reward are deferred to Appendix A. Confidence-
aware updates mainly redistribute mass over existing trajectories rather than forcing the model to
construct entirely new reasoning paths. We also evaluated two alternative reward designs to check
whether the main objective was unusually sensitive: a regression-style confidence target and an
F1-based target. In practice, neither trained reliably. The regression variant collapsed to extreme
confidence values, while the F1-based variant provided a noisy supervisory signal and yielded little
reward improvement. We therefore keep the main text focused on the binary-correctness reward and
defer the ablation details to Appendix C.1.
On the calibration evaluation, training preserves response quality while sharply improving uncertainty
quality: accuracy rises slightly from 0.345 to0.358 , while ECE drops from 0.383 to0.049 , Brier
score from 0.504 to0.166 , NLL from 4.987 to0.498 , and the overconfidence gap from +0.523 to
+0.045 . The full metric table is deferred to Appendix C.1 (Table 5). More importantly, calibration
fundamentally changes thefailure modeof the model. The baseline is dominated by confidently wrong
predictions, whereas the calibrated model assigns substantially lower confidence to incorrect answers.
This indicates that calibration does not simply rescale confidence, but suppresses overconfident error
without degrading reasoning accuracy. Training dynamics and the ablations over alternative reward
formulations are reported in Appendix C.1, including the reward curve in Figure 15a.
Logit-Lens Analysis of Confidence Decoding.We analyze the hidden state at the confidence token
position with a logit lens and aggregate the predicted digits into LOW, MID, and HIGHconfidence
bins. Figure 2 shows the main pattern: in the base model, correct answers and many wrong answers
both end with dominant mass in the HIGHbin, whereas the calibrated model redirects low-confidence
errors away from HIGHand toward LOW. Correct answers also become more conservative rather
than saturating the maximum confidence digit by default. This is consistent with calibration acting
primarily on a late-stage confidence mapping: the underlying uncertainty signal already exists, but its
translation into verbalized confidence becomes more selective after training. Detailed digit-level and
bin-level routing analysis is deferred to Appendix C.1 (Figure 13).
PCA Visualization of Confidence Representations.We also project the final- layer hidden states
at the confidence token position with PCA. The base model already contains a weak confidence-
related manifold, but it is broad and diffuse; after calibration, the geometry becomes smoother and
more ordered along a low-to-high confidence axis. Together with the routing analysis, this suggests
that calibration does not create uncertainty from scratch, but sharpens an existing latent structure and
exposes it more faithfully at the output level.
Error Analysis of the Verbalized Confidence Model.We next analyze how calibration changes
thetypeof errors the model makes. We defineepistemicerrors as wrong answers with confidence
above 0.5, andaleatoricerrors as wrong answers with confidence at most 0.5. We also report a stricter
epistemic category with confidence above0.7, which isolates strongly overconfident hallucinations.
We classify incorrect responses as epistemic or aleatoric using an LLM judge that reads the reasoning
text and explicitly ignores the final confidence value; the full prompt is shown in Appendix D.1.
5

−100 −50 0 50
PC1 (44.7%)−100−50050100PC2 (22.4%)
Base
zoom
0.00.20.40.60.81.0
Verbalized confidence(a) Base model
−100 −50 0 50
PC1 (44.7%)−100−50050100PC2 (22.4%)
Calibrated
0.00.20.40.60.81.0
Verbalized confidence (b) Calibrated model
Figure 3: Final-layer PCA of the confidence-token hidden state, colored by verbalized confidence.
The base model already exhibits a weak confidence gradient, but the geometry is diffuse and partially
entangled. After calibration, the representation is organized into a much cleaner low-to-high confi-
dence manifold, consistent with a sharper and more monotonic confidence readout.
Table 1 shows the sharpest qualitative shift in the section. In the baseline, almost all errors are
epistemic and most are strongly overconfident. After calibration, the majority of errors become low-
confidence errors, and the strict epistemic rate drops by more than an order of magnitude. This is the
main behavioral conclusion of the verbal interface: the model changes from being confidently wrong
to being uncertain when wrong. Detailed confidence-band, per-dataset conversion, and separation
analyses are reported in Appendix C.1 (Table 7).
The same conversion holds across datasets, though its strength varies. The largest reductions
occur on MuSiQue and HotpotQA, while Natural Questions remains the hardest case: strongly
overconfident errors nearly disappear, but some mistakes remain in the moderate-confidence range.
Error type Base Cal.
Epistemic 92.4% 34.9%
Aleatoric 7.6% 65.1%
Strict epi. 88.6% 3.9%
Table 1.Aggregate error decomposition.Appendix C.1 also shows that calibration increases confi-
dence separation between correct and incorrect answers,
rather than simply shifting all scores downward. Taken
together, these results show that verbalized confidence
becomes a calibrated global reliability signal by sup-
pressing overconfident errors without materially rewrit-
ing the underlying reasoning process.
4 Local Uncertainty Interface via Reasoning-Time Signaling
The previous section studied uncertainty exposedaftergeneration through verbalized confidence.
We now consider the complementary case in which uncertainty is exposedduringreasoning. The
goal here is not to estimate the probability that the final answer is correct, but to mark specific points
along the trajectory where the current reasoning state appears unreliable — points where retrieval or
correction can still change the outcome.
Concretely, we train the model to emit the literal string <uncertain> whenever it encounters such a
high-risk state during generation. This signal is alocal uncertainty interface: it does not summarize
final correctness after the fact, but exposes candidate intervention points before the model has fully
committed to an answer. In Adaptive RAG settings, this is exactly the granularity needed — the
signal arrives in the middle of reasoning, when there is still time to act.
4.1<uncertain>-Based Training for Factual Reasoning and Retrieval Control
Setup and objective.We train the model with GRPO to emit the literal string <uncertain>
whenever it enters a high-risk reasoning state, while still ending each response with an explicit final
answer. The signal is learned compositionally from existing tokenizer pieces rather than introduced as
6

a new vocabulary item. At inference time, each emitted <uncertain> becomes a candidate control
point, and a lightweight hidden-state probe decides whether retrieval should actually be triggered.
The training instruction is:
You are a helpful reasoning assistant. Think step by step. If at any point you are uncertain
about a fact, emit the special token <uncertain> to signal that you need more information.
End your response with ‘Answer: <your answer>’ on the last line.
Correctness is determined from the final answer line using normalized exact match, with yes/no
matching, date matching, and token-F1 fallback. The reward is ordered as
r(correct, no emit)> r(correct, emit)> r(wrong, emit)> r(wrong, no emit),(3)
with concrete values 5.0>3.5>0.0>−2.0 in our implementation and an additional repetition
penalty when <uncertain> appears more than twice. The key asymmetry is that silent failure is
penalized more heavily than uncertain failure, so the model is encouraged to expose likely failure
states rather than remain silently overconfident. Unlike the verbalized-confidence objective, which
trains a global post-hoc summary, this objective acts directly on the reasoning trajectory and is
designed to produce intervention-oriented uncertainty.
0.0 0.2 0.4 0.6 0.8 1.0
Normalized first-emission position02468101214DensityBase
0.0 0.2 0.4 0.6 0.8 1.0
Normalized first-emission positionCalibrated
emit + wrong
emit + correctWhere the first <uncertain> event appears in the response
Figure 4: First <uncertain> emission posi-
tion as a fraction of response length.Figure 4 shows where in the response the model
first emits <uncertain> . Emissions are distributed
across the full range of response positions, not clus-
tered near the end. This confirms that the training
objective has successfully instilled mid-reasoning
signaling: the model raises the flag while reason-
ing is still in progress, not after the trajectory has
already been committed to. Across six factual
reasoning datasets, the calibrated model improves
macro-average answer accuracy from 17.67% to
28.53% , raises answer-line completion from 58.90%
to99.93% , and increases the fraction of wrong answers that co-occur with <uncertain> emission
from 37.97% to58.70% . This means the model not only answers more accurately, but also surfaces
a much larger share of failures as explicit intervention candidates. Appendix C.2, Table 8, gives the
full per-dataset reasoning-time signaling breakdown, including accuracy, answer-line completion,
emit rate, and emitted wrong/correct fractions.
0 5 10 15 20 25 30
Transformer layer0.50.60.70.80.91.0Metric valuebest calibrated layer = 13Layer-wise readability of the first <uncertain> event
AUROC (base)
AUROC (calibrated)
AUPRC (base)AUPRC (calibrated)
F1 (base)
F1 (calibrated)
Figure 5: Probe performance peaks in the
middle layers.Hidden-State Probe for Retrieval Triggering.To
convert <uncertain> emission into an actual retrieval
decision, we train a linear probe on emitted examples
to predict whether the final answer is wrong. The probe
uses span-aware hidden-state features around the first
emitted <uncertain> span together with a small set
of scalar response features. Figure 5 shows that the
strongest signal appears in the middle layers rather than
only at the final layer: AUROC and trigger F1 peak
around layer 16, indicating that the uncertainty state
is assembled before the token is emitted, not only at
the moment of emission. This is a high-recall retrieval
proxy: it asks whether a reasoning trace that already surfaced uncertainty should be escalated
for intervention, not whether retrieval is guaranteed to help that example counterfactually. The
detailed feature construction, emitted-subset composition, and full layer-sweep table are deferred to
Appendix C.2 (Table 9b and Table 9a).
From the perspective of Adaptive RAG, the key quantity is global wrong-answer coverage on the
full dev set, not just probe accuracy inside the emitted subset. The left panel of Table 2 shows that
the calibrated-model pipeline at layer 13sends 576of653wrong dev answers to retrieval, covering
88.2% of all failures. The base pipeline covers only 128of848wrong dev answers ( 15.1% ), a5.8×
gap in coverage. The main benefit is therefore not a marginal gain in detector quality, but the fact that
training creates a much broader intervention set worth acting on.
7

A. Retrieval-control usefulness B. Error-type decomposition
Metric Base Calibrated Metric Base Calibrated
Triggered-case precision 53.5% 83.2% Total wrong 847 653
Emitted-case wrong recall 91.5% 79.9% Epistemic 707 (83.5%) 561 (85.9%)
Wrong answers sent to retrieval 128 / 848 576 / 653 Aleatoric-like 140 (16.5%) 92 (14.1%)
Global wrong-answer coverage 15.1% 88.2% Epistemic + emit 296 (35.0%) 523(80.1%)
Epistemic + no emit 411 (48.5%) 28 (4.3%)
Table 2: Local-interface behavioral summary.
4.2 Heuristic Error-Type Analysis
On the matched test set ( 1000 examples), we use a simple heuristic split betweenaleatoric-like
near misses andepistemicfactual misses, and then further divide epistemic errors by whether
<uncertain> was emitted. The right panel of Table 2 shows that the aleatoric-like fraction is
small and nearly unchanged across models. The main shift is instead in how epistemic failures are
surfaced: silent epistemic errors fall from 48.5% to4.3% of all wrong answers, while epistemic
errors with <uncertain> rise from 35.0% to80.1% . The training therefore does not mainly resolve
ambiguity; it converts previously silent failures into explicit intervention signals. Overall, these
results show that <uncertain> functions as a high-recall reasoning-time intervention signal: it
converts previously silent epistemic failures into actionable retrieval candidates, and the full six-task
and layer-sweep evidence is retained in Appendix C.2. More broadly, this section and the previous
one establish that local and global uncertainty interfaces are complementary rather than competing:
verbalized confidence provides a calibrated summary of overall reliability, while <uncertain>
provides mid-reasoning intervention signals before the model fully commits to a trajectory.
5 Mechanistic Analysis of Calibration Interfaces
A central question is why uncertainty quality can improve substantially without degrading reasoning
quality. To investigate this, we focus on two complementary analyses: where calibration-related
changes are concentrated across token positions, and how strongly the model’s internal representations
are altered. Taken together, these analyses suggest that uncertainty is constructed in a distributed
way along the reasoning trajectory and becomes observable only at designated output positions. The
two interfaces, however, expose this latent signal differently. The verbalized-confidence interface
behaves like a geometry-preserving readout, refining how uncertainty is decoded while leaving the
representation of reasoning states largely intact. By contrast, the <uncertain> interface induces a
broader internal uncertainty state that reshapes late-layer representations before producing an explicit
emission.
Localization: at which positions does calibration act?We compute the token-level KL divergence
between base and calibrated model distributions at every position in the assistant turn, and group
positions by their semantic type (confidence digit, structural label, reasoning token, uncertainty token,
nearby context). This directly reveals which positions absorb the distributional change.
Figure 6 shows that both training objectives successfully localize their effect at the intended output
position. The verbal interface produces a point-like signature: only the digit token is changed, leaving
the surrounding format and all reasoning tokens largely unaffected. The <uncertain> interface
produces a wider footprint: KL is elevated not just at the emission token but in the tokens immediately
surrounding it, indicating that the explicit signal is preceded by a change in the model’s local
computation state. Localization is therefore a property of both interfaces. Additional hidden-state
patching results, reported in Appendix B.2, provide supporting evidence that the signal position is
better interpreted as an exposure point than as a self-contained causal circuit. We treat those results
as suggestive rather than definitive, since the intervention changes only a single token state.
Representation geometry: how deeply does calibration rewrite the model?We measure this
using Centered Kernel Alignment (CKA), which compares the geometry of hidden representations at
signal-token positions between the base and calibrated model, layer by layer. A CKA value of 1.0
means the representations are geometrically identical; values below 1.0indicate structural divergence.
8

(a) Verbal interface
 (b) Special-token interface
Figure 6: Token-level KL by position type. Both objectives concentrate distributional change at their
signal positions, but the special-token interface has a broader local footprint.
(a) Verbal interface
 (b) Special-token interface
Figure 7: Layer-wise CKA between base and calibrated models. Verbal calibration preserves
representation geometry, whereas the special-token interface induces increasing late-layer divergence.
Figure 7 provides the clearest contrast between the two interfaces. The verbal model achieves a
large improvement in calibration quality while leaving internal representations completely unchanged
— the CKA curve is flat at 1.0from input to output layer. This means the model learned to better
decode existing uncertainty representations rather than creating new ones: a soft confidence-mapping
refinement that operates on top of the pretrained geometry. The <uncertain> model takes a different
path: late-layer representations diverge progressively, indicating that explicit mid-reasoning emission
requires the model to actively build a new internal state, not just refine an existing output mapping.
An important implication is that raw parameter movement is not sufficient to explain behavioral
interference. Appendix B.3 shows that the two calibrated models exhibit broadly similar parameter-
space drift patterns, concentrated in attention v_proj /o_proj and MLP projections, with little
drift in LayerNorm terms (Figure 10). Yet these similarly sized and similarly located updates
have sharply different representation-level consequences: the verbal interface preserves geometry,
whereas the <uncertain> interface rewrites late-layer states. The key distinction between the two
interfaces is therefore not how much they update the model, but whether the objective can be realized
as a lightweight re-decoding of existing latent uncertainty or instead requires constructing a new
uncertainty-expressing computation. Additional patching and per-example linkage analyses are
deferred to Appendix B.2 and Appendix B.4.
6 Evaluation
We evaluate the trained models on five widely used factual and multi-hop QA benchmarks: Hot-
potQA Yang et al. [2018], MuSiQue Trivedi et al. [2022], 2WikiMultihopQA Ho et al. [2020], Natural
Questions Kwiatkowski et al. [2019], and TriviaQA Joshi et al. [2017]. Together, these datasets cover
both multi-step compositional reasoning and open-domain factual recall, providing a diverse testbed
for evaluating whether uncertainty signals improve calibration and retrieval control across different
knowledge and reasoning demands.
9

A. Verbal confidence interface B. Special-token uncertainty interface
Method EM F1 Brier ECE OConf Method Emit Prec. Recall Acc ¬t Wrong/Pos.
Base 24.5 37.3 -0.108 0.357 88.5 Emit heur. 0.3360.9590.444 0.392 0.726
P(True) 24.4 37.2 -0.096 0.340 39.7 Hidden probe 0.699 0.889 0.856 0.653 0.726
Global TS 24.5 37.3 +0.116 0.185 0.0 Output clf. 0.925 0.7540.9610.622 0.726
ATS 24.5 37.3 +0.123 0.166 0.0 Self-RAG 0.444 0.861 0.478 0.250 0.799
SFT-Conf 21.1 33.5 +0.083 0.226 7.3 FLARE 0.586 0.738 0.598 0.300 0.722
SFT-KWDK 22.4 34.5 +0.105 0.204 8.6 ADARAGUE 0.527 0.216 0.687 0.6900.166
Ours 27.4 38.2 +0.210 0.036 3.2 Ours0.592 0.799 0.8830.7190.528
Table 3:Unified baseline comparison across the two calibration interfaces. Panel Areports the
verbal confidence setting on counterfactual evaluation set ( n= 1000 ). EM is strict exact match; F1
denotes Acc. (F1≥0.3). OConf is the percentage of wrong answers with confidence>0.7.Panel B
reports the <uncertain> setting on the counterfactual evaluation set ( n= 1000 ). Prec. = P(wrong |
trigger), Recall = P(trigger |wrong), Acc ¬tis accuracy on examples left untouched by the trigger,
and Wrong/Pos. is the fraction of wrong examples within the triggered set.
6.1 Calibration Evaluation
Sections 3 and 4 established the main empirical effects of the two uncertainty interfaces in their native
settings. Here we ask a narrower question: can those gains be explained by simpler alternatives?
For verbal confidence, we compare against post-hoc recalibration baselines (Global TS,ATS),
a verbal self-check baseline (P(True)), and supervised confidence-tuning baselines (SFT-Conf,
SFT-KWDK) Guo et al. [2017], Xie et al. [2024], Kapoor et al. [2024], Luo et al. [2025]. For the
<uncertain> interface, we compare against prompt-only emission, passive wrongness detectors,
and retrieval-controller baselines evaluated under the same protocol, includingSelf-RAG,FLARE,
andADARAGUEAsai et al. [2023], Jiang et al. [2023], Moskvoretskii et al. [2025]. Concretely,
P(True)replaces the emitted scalar with a yes/no self-assessed correctness probability;Global TS
andATSrescale the base model’s existing confidence either globally or per example;SFT-Confand
SFT-KWDKsupervise oracle-derived continuous or bucketed confidence labels. On the local side,
Emit heur.uses raw base-model <uncertain> emission,Hidden probeandOutput clf.predict
wrongness from internal states or surface cues, andSelf-RAG,FLARE, andADARAGUEprovide
retrieval-oriented trigger analogs. Full implementation details are given in Appendix C.3.
Verbal confidence interface.The verbal comparison is best read as a baseline check rather than
a second proof of calibration. Post-hoc recalibration helps substantially, and ATS is the strongest
simple alternative, but neither post-hoc method matches the calibrated verbal GRPO model. P(True)
is also informative here: asking the model to assess its own answer reduces overconfidence relative
to the base model, but it does not improve answer quality and still leaves ECE far above the GRPO
verbal interface. The same holds for the supervised baselines: both SFT variants improve on the raw
base model, yet both remain clearly behind GRPO on calibration quality and answer quality. The
main conclusion is therefore unchanged from Section 3: the verbal gains are not reducible to simple
rescaling, verbal self-checking, or supervised confidence relabeling on top of base-model trajectories.
Special-token uncertainty interface.The <uncertain> comparison asks a different question:
not whether a scalar is calibrated, but whether wrongness can be surfaced early enough to support
intervention. The base-model heuristics, probes, and retrieval-controller baselines show that failure is
already partially detectable without end-to-end training. Under the updated protocol, however, our
Uncertain-Calibrate model is not merely harder to evaluate because it changes the positive-set
distribution; it is also strong on the intervention metrics that matter. In particular, it attains the
best untouched-set accuracy and remains competitive on precision and recall against much more
aggressive detectors. This sharpens the same point established in Section 4: end-to-end training does
not simply place a detector on top of the base model; it changes the generator so that more failures
are surfaced in a form that is useful for downstream control.
6.2 Downstream Task Performance: Adaptive RAG Triggering
We evaluate downstream retrieval control on a counterfactual reasoning benchmark containing 500
questions from each of five factual QA datasets. Each method first produces a no-retrieval answer,
then optionally triggers one retrieval step according to its uncertainty policy, and finally answers again
10

Table 4: Adaptive RAG evaluation results. EM, F1 (both %), and trigger rate T(%) are shown per
dataset when available. For each EM/F1 column, the top three values are highlighted. For each
trigger-rate column, only the highestselectivetrigger rate is highlighted; Ret-All is excluded since
it always triggers retrieval.
HotpotQA MuSiQue 2WikiMultiHop NQ TriviaQA Overall
Method EM F1 T EM F1 T EM F1 T EM F1 T EM F1 T EM F1 T
No-Ret 23.4 31.9 – 5.6 10.3 – 16.4 20.1 – 29.2 42.0 – 54.8 61.2 – 25.9 33.1 –
SR-7B 4.2 16.6 57.8 0.6 5.2 56.0 4.4 14.8 45.6 17.8 22.0 18.0 5.6 24.9 53.6 6.5 16.7 46.2
SR-13B 2.6 16.0 42.4 0.8 6.1 39.8 3.4 16.6 30.0 30.8 36.9 5.4 6.4 36.5 29.8 8.8 22.4 29.5
Ret-All 34.0 44.9 100 9.8 17.4 100 30.0 34.7 100 25.4 35.8 100 43.0 49.1 100 28.4 36.4 100
ADARAGUE 27.8 37.2 57.0 9.6 15.9 53.8 20.6 25.8 57.4 28.6 40.0 21.8 52.2 58.1 29.0 27.8 35.4 43.8
FLARE 21.8 35.2 99.26.4 13.8 99.815.2 22.6 99.419.8 31.5 99.041.0 49.9 97.420.8 30.6 99.0
DRAGIN 34.6 48.4 87.6 17.0 27.1 87.0 34.4 43.985.2 23.8 35.9 70.4 53.6 60.6 54.0 32.7 43.2 76.8
Base-Verbal 21.2 28.8 15.2 10.8 17.5 13.6 17.0 19.7 9.2 17.8 29.7 22.8 42.4 43.8 4.6 21.8 27.9 13.1
Base-UncTok 22.5 33.1 3.5 8.8 16.2 4.3 17.7 21.8 2.6 18.8 28.1 3.7 34.8 36.8 4.5 20.5 27.2 3.5
Verbal-Calibrate (global) 42.0 52.861.6 21.8 28.876.8 38.4 42.9 48.2 52.4 54.425.0 63.2 72.5 28.8 41.6 50.548.1
Uncertain-Calibrate (local) 42.6 52.7 67.4 17.6 24.1 94.2 36.2 39.6 59.2 41.4 51.0 52.0 66.6 73.234.0 40.9 48.1 61.4
using the retrieved evidence. Table 4 reports the final post-retrieval EM and F1 together with trigger
rateT, which measures how often a method chooses to retrieve. We compare against both no-retrieval
and always-retrieval baselines, as well as Self-RAG Asai et al. [2023], ADARAGUE Moskvoretskii
et al. [2025], FLARE Jiang et al. [2023], and DRAGIN Su et al.. We also include two base-model
interface baselines:Base-Verbal, which asks the base model to verbalize confidence for retrieval
triggering, andBase-UncTok, which asks it to emit<uncertain>instead.
Main result.Both learned uncertainty interfaces remain clearly stronger than the non-adaptive
baselines, including No-Ret , the always-retrieve variants, and fixed classifier-style controllers.
Verbal-Calibrate achieves the best overall adaptive performance at 41.6% EM and 50.5% F1 with
a48.1% trigger rate, while Uncertain-Calibrate reaches 40.9% EM and 48.1% F1 at a higher
61.4% trigger rate. Relative to ADARAGUE ,Verbal-Calibrate improves overall EM/F1 by a large
margin while operating at a comparable retrieval budget, indicating that the learned verbal-confidence
signal is a substantially stronger gating variable than a post-hoc uncertainty classifier. Compared
with retrieval-heavy baselines such as FLARE andDRAGIN , the adaptive methods also deliver better
aggregate performance, showing that the gain is not simply due to retrieving more often, but to
retrieving more selectively.
Comparison across baselines.Among the non-adaptive retrieval methods, DRAGIN is the strongest,
reaching 32.7% EM and 43.2% F1 overall, substantially above FLARE (20.8% /30.6% ) and the
always-retrieve baselines. However, both Verbal-Calibrate andUncertain-Calibrate still
outperform DRAGIN by a wide margin on overall EM and F1. This gap is especially notable because
FLARE retrieves on nearly every example and DRAGIN also uses a very high trigger rate, so the
difference cannot be explained by retrieval frequency alone. Instead, the results suggest that adaptive
retrieval quality depends critically on whether the control signal tracks genuine answer uncertainty,
rather than merely encouraging aggressive retrieval. The weak performance of Base-Verbal and
Base-UncTok supports the same conclusion: exposing the interface alone is not enough; the control
signal itself must be explicitly trained.
Difference between the two uncertainty interfaces.The relative pattern between
Verbal-Calibrate andUncertain-Calibrate is also consistent with their intended roles.
Verbal-Calibrate is stronger overall and more retrieval-efficient, suggesting that a global confi-
dence estimate is better aligned with question-level gating. By contrast, Uncertain-Calibrate
retrieves more aggressively and obtains its clearest advantages on datasets such as HotpotQA
and TriviaQA, where it achieves the best EM. Its behavior on MuSiQue is particularly revealing:
the trigger rate rises to 94.2% , close to always retrieving, yet its accuracy still remains below
Verbal-Calibrate . This pattern suggests that the local <uncertain> signal behaves more like
a high-recall intervention mechanism during reasoning, whereas verbal confidence acts as a more
precise global controller over whether retrieval is needed at all.
11

7 Conclusion
We studied uncertainty in large language models as an interface-design problem rather than only a post-
hoc estimation problem. Within a unified post-training framework, we compared a global interface
that verbalizes confidence for the final answer with a local interface that emits <uncertain> during
reasoning. The two interfaces produce different but complementary benefits: verbalized confidence
is most effective for final-answer trust and retrieval gating, while reasoning-time signaling is most
effective for exposing silent failures early enough for intervention.
Our results also show that these gains are not merely formatting effects. Verbal calibration suppresses
overconfident error paths and sharpens an existing latent confidence structure without substantially
rewriting representation geometry, whereas the <uncertain> interface induces a broader late-layer
uncertainty state that supports explicit mid-reasoning signaling. Together, these findings suggest that
effective uncertainty in LLMs should be trained as task-matched communication: global confidence
when the decision is whether to trust the final answer, and local reasoning-time signals when the
decision is whether the model needs intervention before it fully commits.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.arXiv preprint arXiv:2310.11511, 2023.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. InInternational conference on machine learning, pages 1321–1330. PMLR, 2017.
Jianfeng He, Linlin Yu, Changbin Li, Runing Yang, Fanglan Chen, Kangshuo Li, Min Zhang, Shuo
Lei, Xuchao Zhang, Mohammad Beigi, et al. Survey of uncertainty estimation in llms-sources,
methods, applications, and challenges.Information Fusion, page 104057, 2025.
John Hewitt, Robert Geirhos, and Been Kim. We can’t understand ai using our existing vocabulary.
arXiv preprint arXiv:2502.07586, 2025a.
John Hewitt, Oyvind Tafjord, Robert Geirhos, and Been Kim. Neologism learning for controllability
and self-verbalization.arXiv preprint arXiv:2510.08506, 2025b.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the 28th International
Conference on Computational Linguistics, pages 6609–6625, 2020.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity.arXiv
preprint arXiv:2403.14403, 2024.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation.arXiv, 2023.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. InProceedings of the 55th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–
1611, 2017.
Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian
Weller, Samuel Dooley, Micah Goldblum, and Andrew G Wilson. Large language models must be
taught to know what they don’t know.Advances in Neural Information Processing Systems, 37:
85932–85972, 2024.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a
benchmark for question answering research.Transactions of the Association for Computational
Linguistics, 7:453–466, 2019.
Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, and Ivan Oseledets. Confidence
is all you need: Few-shot rl fine-tuning of language models, 2025a.arXiv, 2.
12

Jingyu Liu, Jiaen Lin, and Yong Liu. How much can rag help the reasoning of llm?arXiv preprint
arXiv:2410.02338, 2024.
Beier Luo, Shuoyuan Wang, Sharon Li, and Hongxin Wei. Your pre-trained llm is secretly an
unsupervised confidence calibrator.arXiv preprint arXiv:2505.16690, 2025.
Viktor Moskvoretskii, Maria Lysyuk, Mikhail Salnikov, Nikolay Ivanov, Sergey Pletenev, Daria
Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina, and Alexander Panchenko.
Adaptive retrieval without self-knowledge? bringing uncertainty back home.arXiv preprint
arXiv:2501.12835, 2025.
Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens.Advances
in Neural Information Processing Systems, 36:19327–19352, 2023.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathemat-
ical reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
W Su, Y Tang, Q Ai, Z Wu, and Y Liu. Dragin: Dynamic retrieval augmented generation
based on the real-time information needs of large language models. arxiv 2024.arXiv preprint
arXiv:2403.10081.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554, 2022.
Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev, Lyudmila Rvanova, Daniil Vasilev, Akim
Tsvigun, Sergey Petrakov, Rui Xing, Abdelrahman Sadallah, Kirill Grishchenkov, et al. Benchmark-
ing uncertainty quantification methods for large language models with lm-polygraph.Transactions
of the Association for Computational Linguistics, 13:220–248, 2025.
Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen,
Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive
effective reinforcement learning for llm reasoning.arXiv preprint arXiv:2506.01939, 2025.
Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Tianle Cai, and Wenhao Huang. Mit-
igating llm hallucination via behaviorally calibrated reinforcement learning.arXiv preprint
arXiv:2512.19920, 2025.
Johnathan Xie, Annie S Chen, Yoonho Lee, Eric Mitchell, and Chelsea Finn. Calibrating language
models with adaptive temperature scaling. InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, pages 18128–18138, 2024.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.arXiv preprint arXiv:1809.09600, 2018.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Linmei Hu, Liu Weichuan, Lei Hou, and Juanzi Li.
Seakr: Self-aware knowledge retrieval for adaptive retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 27022–27043, 2025.
Yiming Zhang, Jianfeng Chi, Hailey Nguyen, Kartikeya Upasani, Daniel M Bikel, Jason Weston, and
Eric Michael Smith. Backtracking improves generation safety.arXiv preprint arXiv:2409.14586,
2024.
Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason
without external rewards, 2025.arXiv, 2.
13

A Proofs for the trajectory-reweighting analysis
In this appendix, we give short proofs for the theoretical claims in Section 3. Throughout, we use the
tilted-distribution idealization
πθ′(z|x)∝π θ(z|x) exp 
η r(z;x)
,(4)
as a first-order analytical model of one-step uncertainty-aware policy improvement under GRPO,
whereη >0is an effective step size.
Normalization form.For fixed inputx, define the partition function
Z(x) =X
zπθ(z|x) exp 
η r(z;x)
.(5)
Then the reweighted policy can be written as
πθ′(z|x) =πθ(z|x) exp 
η r(z;x)
Z(x).(6)
Proposition 2(One-step relative improvement under uncertainty-aware RL).For any two trajectories
z1, z2for the same inputx, the reweighted policy satisfies
logπθ′(z1|x)
πθ′(z2|x)= logπθ(z1|x)
πθ(z2|x)+η 
r(z1;x)−r(z 2;x)
.(7)
Hence a single RL improvement step increases the relative likelihood of higher-reward trajectories
and decreases that of lower-reward ones.
Corollary 1(Selective suppression of overconfident errors).Consider two wrong trajectories z1, z2
with confidences p(z1)> p(z 2). Under the main-text reward, r(z1;x)< r(z 2;x), so after one
improvement step,
πθ′(z1|x)
πθ′(z2|x)<πθ(z1|x)
πθ(z2|x).(8)
That is, among incorrect trajectories, the more overconfident one is suppressed more strongly.
Symmetrically, among correct trajectories, higher-confidence ones are relatively amplified.
Corollary 2(Answer improvement without new knowledge).Let
Sθ(y|x) =X
z:g(z)=yπθ(z|x)p(z),(9)
denote the confidence-weighted score of answery, and define the margin
Γθ(x) =S θ(y⋆|x)−max
y̸=y⋆Sθ(y|x),(10)
where y⋆is the correct answer. If the update increases the relative mass of correct high-confidence
trajectories enough that Γθ′(x)>0 while Γθ(x)≤0 , then the model’s prediction flips from incorrect
to correct without introducing any new reasoning trajectory.
Proof of Proposition 2. For any two trajectories z1, z2for the same input x, using the normalized
form above,
πθ′(z1|x)
πθ′(z2|x)=πθ(z1|x) exp 
η r(z 1;x)
/Z(x)
πθ(z2|x) exp 
η r(z 2;x)
/Z(x).(11)
The normalization constant cancels, giving
πθ′(z1|x)
πθ′(z2|x)=πθ(z1|x)
πθ(z2|x)exp
η 
r(z1;x)−r(z 2;x)
.(12)
Taking logarithms yields
logπθ′(z1|x)
πθ′(z2|x)= logπθ(z1|x)
πθ(z2|x)+η 
r(z1;x)−r(z 2;x)
.(13)
Therefore, whenever r(z1;x)> r(z 2;x), the post-update log-odds of z1against z2increase; when
r(z1;x)< r(z 2;x), they decrease.
14

Proof of Corollary 1. Consider two wrong trajectories z1, z2with confidences p(z1)> p(z 2). Under
the main-text reward, wrong trajectories receive reward
r(z;x) =−p(z).(14)
Hence
r(z1;x) =−p(z 1)<−p(z 2) =r(z 2;x).(15)
Applying Proposition 2,
logπθ′(z1|x)
πθ′(z2|x)= logπθ(z1|x)
πθ(z2|x)+η 
r(z1;x)−r(z 2;x)
,(16)
and the final term is strictly negative. Therefore,
logπθ′(z1|x)
πθ′(z2|x)<logπθ(z1|x)
πθ(z2|x),(17)
which implies
πθ′(z1|x)
πθ′(z2|x)<πθ(z1|x)
πθ(z2|x).(18)
Thus, among incorrect trajectories, the more overconfident one is suppressed more strongly.
The statement for correct trajectories is symmetric. If z1, z2are both correct and p(z1)> p(z 2), then
under the main-text reward,
r(z1;x) =p(z 1)> p(z 2) =r(z 2;x),(19)
so Proposition 2 implies that the relative likelihood ofz 1increases after the update.
Proof of Corollary 2.Recall the confidence-weighted answer score
Sθ(y|x) =X
z:g(z)=yπθ(z|x)p(z),(20)
and the answer margin
Γθ(x) =S θ(y⋆|x)−max
y̸=y⋆Sθ(y|x),(21)
wherey⋆is the correct answer.
Suppose that before the GRPO update,
Γθ(x)≤0,(22)
so the correct answer does not strictly dominate all competing answers under the confidence-weighted
score. Suppose further that after the update,
Γθ′(x)>0.(23)
Then by definition,
Sθ′(y⋆|x)> S θ′(y|x)for ally̸=y⋆. (24)
Therefore the confidence-weighted decision rule
ˆyθ(x) = arg max
ySθ(y|x)(25)
changes from not selectingy⋆before the update to selectingy⋆after the update.
Finally, under the tilted-distribution model, the update only changes the relative weights of existing
trajectories through πθ′(z|x) ; it does not introduce new trajectory support. Hence the prediction
flips from incorrect to correct without requiring any new reasoning trajectory to be created.
Remark.These proofs establish only the consequences of the one-step tilted-distribution model.
They should therefore be interpreted as a local analytical account of how uncertainty-aware RL can
improve calibration and answer selection by redistributing probability mass over existing trajectories,
rather than as an exact global characterization of GRPO training dynamics.
15

A.1 Proof of Theorem 1
We restate the theorem for convenience.
Theorem 2(Latent-answer extraction under support-preserving reweighting).Fix an input x, and let
πθ(z|x) be the model distribution over complete reasoning trajectories z. Suppose the post-update
policy is given by
πθ′(z|x) =πθ(z|x) exp 
ηr(z;x)
Z(x),(26)
where
Z(x) =X
zπθ(z|x) exp 
ηr(z;x)
,(27)
andη >0.
Define the answer-level probability mass
Mθ(y|x) =X
z:g(z)=yπθ(z|x),(28)
where g(z) denotes the final answer induced by trajectory z. Lety⋆be the correct answer and let
¯y̸=y⋆be any competing wrong answer.
Assume that every trajectory producing the correct answer satisfies
r(z;x)≥a,∀zsuch thatg(z) =y⋆,(29)
and every trajectory producing¯ysatisfies
r(z;x)≤b,∀zsuch thatg(z) = ¯y,(30)
for constantsa > b. Then:
Mθ′(y⋆|x)
Mθ′(¯y|x)≥exp 
η(a−b)Mθ(y⋆|x)
Mθ(¯y|x).(31)
Moreover,
suppπ θ′(· |x) = suppπ θ(· |x).(32)
Proof.We first prove the answer-mass ratio bound. By definition,
Mθ′(y⋆|x) =X
z:g(z)=y⋆πθ′(z|x)(33)
=X
z:g(z)=y⋆πθ(z|x) exp 
ηr(z;x)
Z(x).(34)
For every trajectoryzsuch thatg(z) =y⋆, the assumption in Eq. (29) implies
exp 
ηr(z;x)
≥exp(ηa).(35)
Substituting this into Eq. (34) gives
Mθ′(y⋆|x)≥X
z:g(z)=y⋆πθ(z|x) exp(ηa)
Z(x)(36)
=exp(ηa)
Z(x)X
z:g(z)=y⋆πθ(z|x)(37)
=exp(ηa)
Z(x)Mθ(y⋆|x).(38)
Similarly,
Mθ′(¯y|x) =X
z:g(z)=¯yπθ′(z|x)(39)
=X
z:g(z)=¯yπθ(z|x) exp 
ηr(z;x)
Z(x).(40)
16

For every trajectoryzsuch thatg(z) = ¯y, the assumption in Eq. (30) implies
exp 
ηr(z;x)
≤exp(ηb).(41)
Therefore,
Mθ′(¯y|x)≤X
z:g(z)=¯yπθ(z|x) exp(ηb)
Z(x)(42)
=exp(ηb)
Z(x)X
z:g(z)=¯yπθ(z|x)(43)
=exp(ηb)
Z(x)Mθ(¯y|x).(44)
Combining Eq. (38) and Eq. (44), we obtain
Mθ′(y⋆|x)
Mθ′(¯y|x)≥exp(ηa)
Z(x)Mθ(y⋆|x)
exp(ηb)
Z(x)Mθ(¯y|x)(45)
= exp 
η(a−b)Mθ(y⋆|x)
Mθ(¯y|x).(46)
This proves Eq. (31).
We now prove support preservation. Since exp(ηr(z;x))>0 for every trajectory zandZ(x)>0 ,
Eq. (26) implies
πθ′(z|x)>0⇐⇒π θ(z|x)>0.(47)
Therefore,
suppπ θ′(· |x) = suppπ θ(· |x),(48)
which proves Eq. (32).
Finally, Eq. (32) yields the main interpretation used in the paper: the update cannot create a correct
trajectory that was absent from the original support. It can only increase the relative mass of correct
trajectories that were already present but underweighted underπ θ.
Specialization to the verbal-confidence reward.Under the reward
r(z;x) =p(z), g(z) =y⋆,
−p(z), g(z)̸=y⋆,(49)
suppose every correct-answer trajectory satisfies p(z)≥α and every trajectory producing ¯ysatisfies
p(z)≥β. Then one may choose
a=α, b=−β.(50)
Substituting into Eq. (31) yields
Mθ′(y⋆|x)
Mθ′(¯y|x)≥exp 
η(α+β)Mθ(y⋆|x)
Mθ(¯y|x).(51)
Thus, when wrong trajectories are highly confident, they are exponentially downweighted relative to
correct trajectories, which formalizes the intuition that the objective acts as an anti-overconfidence
filter.
B Additional Mechanistic Evidence
This appendix provides additional mechanistic detail supporting the main-text claim that calibration
is integrated into the reasoning process rather than appended as a purely superficial output-formatting
step. The appendix has two goals. First, it expands several analyses that are informative but not
central enough for the main body. Second, it clarifies the limits of what the current experiments do
and do not show. In particular, these analyses support a mechanistic account of uncertainty-aware
reasoning, but they do not by themselves prove that the emitted confidence is the true posterior
probability that the answer is correct.
17

B.1 Expanded Token-Level Divergence Analysis
The main text emphasizes per-token localization, since this is the clearest way to show that calibration-
induced change is concentrated at uncertainty-related positions. A complementary view is to examine
thetotalKL mass allocated to each token type. This diagnostic is useful, but it must be interpreted
carefully because long reasoning spans dominate total mass simply by occupying many more positions
than uncertainty tokens.
(a) Verbal model. Most total KL mass lies in reasoning
tokens, even though confidence digits are strongly
enriched on a per-token basis.
(b) Special model. The <uncertain> token is en-
riched, but total KL mass remains dominated by rea-
soning and nearby context tokens.
Figure 8:KL mass fractions by token type.These plots complement the main-text boxplots by
showing total KL allocation rather than per-token enrichment. Because reasoning spans are much
longer than uncertainty spans, raw mass fractions should be interpreted together with the per-token
statistics in the main text.
Figure 8 shows why the per-token view is the right primary lens. In the verbal model, the confidence
digit is strongly enriched relative to ordinary reasoning tokens, but reasoning still accounts for most
KL mass because it occupies far more positions. The same logic holds for the special model: the
<uncertain> token is a meaningful concentration point, but the surrounding reasoning sequence still
carries most of the aggregate divergence. This is consistent with a mechanism in which uncertainty is
computed across the reasoning trace and only becomes especially visible at a small number of output
positions.
Two additional details matter for interpretation. First, the verbal model’s Confidence: label itself is
essentially inert, reinforcing the conclusion that calibration training altered the emitted scalar value
rather than the output template. Second, the special model shows elevated KL in the nearby pre- and
post-windows around <uncertain> , which is not seen for the verbal model. This broader footprint
suggests that the special model enters a local uncertainty-related computation regime around the
emission event, whereas the verbal model behaves more like a clean endpoint readout.
A caveat is that these token-type summaries are affected by sequence truncation. The current analyses
used max_seq_len=512 , and the main analysis already noted that this likely increases the residual
other category and may undercount some structured output regions. This does not undermine the
core localization result, but it does mean that the exact token-type mass fractions should be treated as
approximate.
B.2 Hidden-State Patching as Supporting Evidence
A natural but incorrect inference from token-level localization is that the uncertainty token itself
contains the full causal mechanism. The activation- patching results provide supporting evidence
against that stronger claim. Localization identifies where the calibration effect becomes most visible
in the output distribution, but not necessarily where that effect is fully computed.
For the verbal model, patching hidden states at confidence-digit positions produces almost no
disruption. In contrast, patching random reasoning positions produces substantially larger changes on
average. This pattern is consistent with the confidence value being read out from information that has
already been assembled across earlier reasoning tokens and stored in the accumulated attention state.
For the special model, patching the <uncertain> position does matter, but it is still not the dominant
causal locus under the current intervention. Random reasoning positions are even more disruptive
on average. This indicates that the explicit uncertainty marker participates in the mechanism, but
18

(a) Verbal interface
 (b) Special-token interface
Figure 9:Hidden-state patching at the signal position versus reasoning positions.In both
interfaces, patching the reasoning trace is at least as disruptive as patching the signal token itself,
which is consistent with uncertainty being assembled across the trajectory and only exposed at the
designated output position.
(a) Verbal model. Drift is concentrated in value/out-
put projections and MLP projections, with minimal
change in normalization layers.
(b) Special model. A similar module-level drift pattern
appears despite stronger late-layer representational
divergence.
Figure 10:Relative Frobenius weight drift across layers and module types.Both calibrated models
show similar update structure in parameter space, with the largest changes in v_proj ,o_proj , and
MLP projection layers. This makes the difference in representation geometry especially noteworthy:
similar magnitudes of weight drift yield very different geometric consequences.
does not by itself define the full uncertainty computation. The local token is part of a broader process
rather than a self-contained switch.
This distinction helps clarify the phraselocalized but distributed. The calibration effect is localized
in the sense that it becomes especially visible at uncertainty-related output positions. However, the
supporting computation is distributed over the reasoning trajectory that precedes those positions.
Because the current intervention changes only a single token state, we view this analysis as suggestive
supporting evidence rather than a complete causal account.
B.3 Parameter-Space Drift and Embedding Repositioning
The weight-drift analysis addresses an important question left open by the representation results:
if the verbal model preserves the base geometry so strongly, did it simply undergo a much smaller
parameter update than the special model? The answer is no. Both models exhibit broadly similar
parameter-space drift patterns, which makes their difference in representation-space behavior more
striking.
Figure 10 shows that both calibrated models place most of their parameter drift in the same broad
module classes, especially the attention value/output projections and MLP projections. LayerNorm
terms change very little. This pattern is similar across verbal and special calibration, and the overall
update magnitudes are also comparable. Thus, the fact that verbal CKA remains essentially unchanged
while special CKA diverges in late layers cannot be explained simply by one model being updated
much more than the other.
19

Figure 11:Mechanism-to-behavior linkage for the verbal model.Localization-related features
predict per-example confidence shifts with cross-validated R2= 0.51 , indicating that the strength of
the learned confidence mechanism varies meaningfully across examples rather than appearing only as
a population-level average.
Figure 12:Mechanism-to-utility linkage for the special model.Under the current proxy utility
target, the special model shows weak and unstable within-subset linkage. This likely reflects a
mismatch between the current utility proxy and the model’s operative mechanism, which is more
naturally framed as a binary emission decision than as graded variation within already-emitting
examples.
This produces an informative contrast. In the verbal model, similarly sized weight-space changes
largely preserve local representation geometry at the uncertainty readout position. In the special
model, similarly sized changes accumulate into more visible late-layer geometric divergence. One
interpretation is that the inductive bias of the calibration objective matters as much as raw update size:
a trajectory-level scalar-confidence objective can be realized through a relatively geometry-preserving
readout adjustment, whereas an explicit mid-reasoning uncertainty marker encourages a deeper
reorganization of the computation that produces that marker.
The embedding-drift analysis reinforces this distinction. In the special model, the token embeddings
corresponding to the components of <uncertain> drift more than a random-token baseline, con-
sistent with targeted repositioning of the explicit uncertainty marker. In the verbal model, those
same component tokens drift less than baseline on average, and common bracket tokens are un-
changed. This further supports the interpretation that verbalized calibration does not rely on explicit
uncertainty-token specialization, whereas special calibration does.
B.4 Mechanism-to-Utility Linkage and Its Limits
The mechanistic analyses in the main text establish where uncertainty-related changes occur and how
deeply they alter the model’s internal states. A remaining question is whether those measurements
are also predictive at the level of individual examples, rather than only in aggregate.
For the verbal model, Figure 11 shows that the localization structure captured in the distributional
analysis is not an artifact of averaging: it varies meaningfully across examples and predicts how
strongly the confidence output differs from the base model on each individual instance. Examples
in the top localization quartile show confidence shifts 86% larger than those in the bottom quartile
(0.207 vs.0.111 ). This supports the interpretation that verbal calibration is not merely learning a
surface format; the model is genuinely learning when to engage a stronger confidence adjustment
based on the information accumulated in the reasoning trace.
However, the current utility target is still a proxy: it is the magnitude of change at the uncertainty
token, not the true probability that the answer is correct. Therefore, the verbal-model linkage result
should be interpreted as evidence that the model learns a structured scalar uncertainty readout from
the reasoning trajectory, rather than as proof that the emitted scalar is already a perfectly calibrated
posterior probability of correctness.
20

Figure 12 clarifies why the same analysis is weak for the special model. The special interface is
likely governed by a different operative mechanism. The important decision is whether to emit
<uncertain> at all, rather than how much to vary a continuous confidence valuewithinthe subset of
examples that already emitted the marker. Under that interpretation, a within-emission regression
is simply not the best target for the special model. A stronger analysis for that interface would
instead compare emitting and non-emitting examples directly, treating uncertainty emission as a
selective-prediction or abstention-like decision.
B.5 Summary
The additional analyses in this appendix reinforce three points. First, the localization observed in
the main text is real, but should be understood in per-token rather than raw-mass terms. Second,
localization does not imply that the uncertainty token itself is the complete causal mechanism;
the supporting computation remains distributed across the reasoning trajectory. Third, verbal and
special calibration differ not in whether they affect uncertainty, but in how deeply they rewrite the
computation that supports it. Verbalized calibration is consistent with a geometry-preserving readout
of distributed uncertainty, whereas the special interface is consistent with a more explicit uncertainty
mode that is assembled during reasoning and expressed through a dedicated marker.
C Supplementary Quantitative Results
C.1 Detailed Results for Verbalized Confidence
The main text keeps only the quantitative results needed to establish the verbal interface’s central
claim: calibration improves final-answer uncertainty while preserving or slightly improving answer
quality. This subsection retains the more detailed breakdowns that support that conclusion but are not
needed in the main narrative.
0 5 10 15 20 25 30
Transformer layer0.00.20.40.60.81.0Expected confidence  E[0.x]Correct answers
(base n=191, cal n=191)
Base
Calibrated
0 5 10 15 20 25 30
Transformer layerWrong answers
(base n=309, cal n=309)
Base
CalibratedLayer-wise expected confidence: Base vs Calibrated
Figure 13: Detailed binned routing view for verbalized confidence calibration. In the base model,
correct answers and many wrong answers both terminate with dominant mass in the HIGHconfidence
bin. After calibration, low-confidence errors are redirected away from HIGHand into LOW, while
correct answers remain more conservative. This makes the main mechanism visually explicit:
calibration sharpens the late-stage mapping from hidden states to confidence outputs rather than
uniformly lowering confidence everywhere.
C.2 Additional Results for Reasoning-Time Signaling
The local-interface section in the main text emphasizes the end-to-end behavioral story: the model
surfaces more failures early enough for intervention, and a downstream probe can turn those emissions
into useful retrieval triggers. This subsection retains the broader six-task factual evaluation, the
layer-sweep evidence for the probe, and the emitted-subset composition used to interpret those results.
21

−100 −50 0 50
PC1 (44.7%)−100−50050100PC2 (22.4%)
Base
zoom
Wrong, low conf  (n=125)
Correct  (n=70)
Wrong, high conf  (n=5)(a) Base model
−100 −50 0 50
PC1 (44.7%)−100−50050100PC2 (22.4%)
Calibrated
Wrong, low conf  (n=125)
Correct  (n=70)
Wrong, high conf  (n=5) (b) Calibrated model
Figure 14: Final-layer PCA of the confidence-token hidden state, grouped by outcome type. In
the base model, correct and wrong low-confidence examples remain substantially mixed. After
calibration, the representation aligns more cleanly with outcome structure: wrong low-confidence
cases concentrate on the low-confidence side of the manifold, while correct examples occupy the
higher-confidence region more consistently.
Table 5: Calibration metrics for Llama-3-8B before and after calibration training.
Metric Llama-3-8B (base) Llama-3-8B (calibrated)
Accuracy (↑) 0.345 0.358
Avg. verbalized confidence 0.869 0.403
Overconfidence gap (conf−acc) (↓) +0.523 +0.045
ECE (↓) 0.383 0.049
Brier score (↓) 0.504 0.166
NLL (confidence) (↓) 4.987 0.498
Parse rate (↑) 0.996 1.000
0 20 40 60 80 100 120 140 160 180
Training step0.2
0.00.20.40.6RewardTrain reward
train
0 20 40 60 80 100 120 140 160 180
Training step0.1
0.00.10.20.30.4RewardValidation reward
val
(a) Training and validation reward curves over training
steps.
0.0 0.2 0.4 0.6 0.8 1.0
Verbalized confidence0.00.20.40.60.81.0Empirical accuracy
Perfect calibration
Llama-3-8B (base)
Llama-3-8B (calibrated)(b) Reliability diagrams for the base and calibrated
Llama-3-8B models. Perfect calibration corresponds
to the diagonal dashed line.
Figure 15: Training dynamics and calibration quality of the calibrated Llama-3-8B model.
22

Accuracy ECE AUSC Overconf. Conf. on wrong
DatasetnBase Cal. Base Cal. Base Cal. Base Cal. Base Cal.
2WikiMultihopQA 500 22.428.00.4070.1640.2670.45884.813.30.8230.343
HotpotQA 50035.735.6 0.4590.0850.4770.53692.24.00.8540.293
MuSiQue 500 12.114.40.5430.0890.1260.22484.60.00.8090.207
NQ 500 46.548.20.3670.0330.5600.62596.30.40.8880.426
TriviaQA 500 63.968.40.2420.2110.7490.83894.40.60.8660.329
Aggregate 2500 35.4 38.0 0.408 0.119 0.430 0.526 89.9 3.5 0.869 0.360
Table 6: Dataset-level summary for verbalized-confidence calibration. “Overconf.” denotes the
fraction of wrong answers with confidence>0.7.
Band Base Cal.
c >0.71482 (88.6%) 63 (3.9%)
0.5< c≤0.763 (3.8%) 500 (31.0%)
0.3< c≤0.53 (0.2%) 1 (0.1%)
0.1< c≤0.345 (2.7%) 512 (31.8%)
c≤0.179 (4.7%) 535 (33.2%)
Mean conf on err. 0.837 0.306
Median conf on err. 0.900 0.200
(a) Error bands of wrong answers.Dataset Base Epist. Cal. Epist. Conv. Basec >0.7Cal.c >0.7Primary wrong mode
2WikiMultihopQA 88.4 38.6 -49.8 84.8 13.30.1–0.3+≤0.1
HotpotQA 95.3 29.2 -66.1 92.2 4.00.1–0.3
MuSiQue 90.1 19.2 -70.9 84.6 0.0≤0.1
NQ 98.1 60.2 -38.0 96.3 0.40.5–0.7
TriviaQA 96.7 37.3 -59.4 94.4 0.60.1–0.3
(b) Per-dataset error conversion.
Model / Dataset Mean Conf (Correct) Mean Conf (Wrong) Sep.
Baseline (overall)∼0.9280.837+0.091
2WikiMultihopQA 0.704 0.343+0.361
HotpotQA 0.534 0.293+0.241
MuSiQue 0.389 0.207+0.182
NQ 0.581 0.426+0.155
TriviaQA 0.545 0.329+0.216
(c) Confidence separation.Metric Baseline Calibrated∆
Corr.(greedy conf,pass rate)0.1800.524+0.344
Corr.(mean sampled conf,pass rate)0.3110.561+0.250
Mean within-question conf std 0.1210.062-0.059
Pass rate at conf≥0.70.2360.781+0.545
Pass rate at conf<0.30.071 0.100 +0.029
High–low pass-rate gap 0.1650.681+0.516
(d) Question-level consistency.
Confidence bin N Pass rate Cal. gap
[0.0,0.1)22 0.000 0.000
[0.1,0.2)97 0.066 -0.034
[0.2,0.3)137 0.140 -0.060
[0.5,0.7)182 0.376 -0.224
[0.7,0.9)5 0.800 0.000
[0.9,1.0)57 0.779 -0.121
(e) Residual calibration by confidence bin.
Table 7: Supplementary diagnostics for verbalized-confidence calibration.
C.3 Baseline Implementation Details
This subsection records the concrete implementations behind Table 3. The two panels are evaluated on
separate held-out sets. Panel A (verbal confidence) uses the 2WikiMultihopQA verbalized confidence
evaluation set ( n= 500 ); the model always emits an answer and a decimal confidence p∈[0,1] , and
we report EM, relaxed accuracy (token-F1 ≥0.3 ), Brier reward, ECE, and the rate of overconfident
wrong answers. Panel B (special-token uncertainty) uses the counterfactual <uncertain> evaluation
set (n= 1100 ); each method produces a binary trigger analogous to uncertainty emission, and we
report trigger rate, trigger precision, trigger recall, untouched-set accuracy, and the wrong rate within
triggered examples.
Panel A: Verbal confidence interface. Base.The uncalibrated Llama-3.1-8B-Instruct model is
prompted with the shared verbal-confidence template and its native emitted decimal confidence is
used directly.P(True).We first generate a standard answer, then re-query the same model with a
binary correctness prompt asking whether its own proposed answer is correct. The confidence score
is computed from the normalized probability mass assigned to affirmative versus negative tokens, and
23

Accuracy Answer Line Emit Rate Wrong / Correct + Emit
Dataset Base Calibrated∆Base Calibrated Base Calibrated W+E C+E
2WikiMultihopQA 12.4 25.6 +13.2 51.4 100.0 43.2 59.8 43.2 / 55.4 0.0 / 4.4
HotpotQA 22.0 27.8 +5.8 62.8 99.6 36.6 70.8 35.8 / 59.0 0.8 / 11.8
MuSiQue 4.2 6.6 +2.4 50.8 100.0 49.8 94.6 49.8 / 89.0 0.0 / 5.6
NQ 21.6 40.8 +19.2 77.0 100.0 23.6 56.6 23.0 / 39.6 0.6 / 17.0
TriviaQA 36.2 56.0 +19.8 72.0 100.0 31.8 44.2 30.2 / 31.6 1.6 / 12.6
Macro Avg. 17.67 28.53 +10.86 58.90 99.93 38.53 68.87 37.97 / 58.70 0.57 / 10.17
Table 8: Base model vs. calibrated model on six factual reasoning datasets. “Answer Line” is the
fraction of responses containing an explicit final answer line. “Emit Rate” is the overall <uncertain>
emission rate. Each entry in the last two columns isBase / Calibrated: the fraction of wrong answers
co-occurring with emission (W+E) and the fraction of correct answers co-occurring with emission
(C+E). All values are percentages.
Layer Dev AUROC Dev AUPRC Trigger Precision Trigger Recall Trigger F1
0 0.6136 0.8365 0.8008 0.8216 0.8111
8 0.72910.89510.80400.87980.8402
160.73820.89400.83240.86570.8487
24 0.6915 0.8662 0.8171 0.8597 0.8379
Final 0.7371 0.8926 0.81900.87980.8483
(a) Layer sweep on emitted dev examples.Statistic Base Cal.
Train emit cases 1233 6334
Dev emit cases 133 649
Wrong@emit (dev) 0.9774 0.7689
Correct@emit (dev) 0.0226 0.2311
Total wrong (dev) 848 653
(b) Emitted subset.
Table 9: Probe diagnostics for the local uncertainty interface.
replaces the original Confidence: value.Global TS.A single scalar temperature is fit on the base
model’s training predictions by minimizing Bernoulli negative log-likelihood in logit space, then
applied post hoc to the base model’s test confidences.ATS.Adaptive temperature scaling predicts
an example-specific temperature from lightweight response features, including the raw confidence
logit, response length, answer length, and reasoning depth; the feature weights are fit on base-model
outputs with L2-regularized Bernoulli NLL.SFT-Conf.This supervised baseline fine-tunes the
model to reproduce the base model’s reasoning and answer text while replacing the final confidence
with a clipped token-F1-derived target in [0.05,0.95] . Training uses full fine-tuning on roughly 9.5K
base-model generations collected from five QA datasets.SFT-KWDK.This variant uses the same
data and training pipeline as SFT-Conf, but replaces the continuous F1 target with a four-bucket
confidence mapping. It tests whether coarse uncertainty supervision is sufficient, as opposed to
continuous confidence regression.
Panel B: Special-token uncertainty interface. Emit heuristic.The base Llama-3.1-8B-Instruct
model is prompted with the same <uncertain> instruction used for GRPO training, and a trigger is
fired whenever the literal token string appears anywhere in the greedy response.Hidden probe.We
extract a hidden-state representation from the base model at a designated pre-answer readout position,
fit a logistic regression probe to predict wrongness, choose the best layer by development AUPRC,
and tune the decision threshold on held-out development data.Output classifier.This baseline
fits logistic regression on surface response features only, including response length, reasoning-line
count, hedging cues, and whether <uncertain> already appears. It tests whether special-token
calibration can be matched by shallow textual signals without access to model internals.SELF-RAG.
We use the public Self-RAG checkpoint and interpret the model’s internal [Retrieval] control
token as the binary uncertainty trigger. Retrieval is not actually executed in this baseline; only the
signal quality of the trigger is evaluated.FLARE.FLARE inspects first-pass token probabilities and
triggers if any token within the look-ahead window falls below a fixed probability threshold. In our
implementation the threshold is 0.4, and the baseline is evaluated as a pure trigger policy without
downstream retrieval.ADARAGUE.ADARAGUE is an adaptive retrieval pipeline that uses a verbal
confidence controller to decide whether to retrieve external evidence. In Panel B we map its retrieval
decision to the same binary trigger protocol, so it serves as a retrieval-oriented uncertainty baseline
rather than a pure detector.
Shared implementation choices.All baseline generations use greedy decoding with vLLM. All
methods except SELF-RAG share the same Llama-3.1-8B-Instruct base model. Post-hoc verbal
methods are fit and evaluated only on the base model’s emitted confidences, whereas the SFT baselines
24

retrain the generator. For Panel B, all baselines are evaluated under the same binary-trigger protocol
and the same relaxed correctness criterion, so the comparison isolates the quality of the control signal
rather than differences in answer extraction or evaluation code.
25

D Epistemic Error and Aleatoric Error Examples
D.1 Judge Prompt for Epistemic vs. Aleatoric Error Classification
LLM Judge Prompt for Error-Type Classification
You are evaluating a factual QA model response to classify the nature of its error.
The model answered the questionINCORRECTLY. Your task is to read the response and determine
whether the error isEPISTEMICorALEATORIC, based on the content and tone of the reasoning —
NOTthe confidence number at the end.
DEFINITIONS:
EPISTEMIC— The model’s reasoning is confident and assertive. It presents a definitive chain of reasoning
with no hedging, no expressed doubt, and no acknowledgement that it might be wrong. The model “thinks
it knows” even though it is wrong. Look for: confident assertions (“The answer is. . . ”, “According to X, it
is clearly. . . ”, “I know that. . . ”), absence of uncertainty markers, and committed factual claims stated as
certain.
ALEATORIC— The model’s reasoning expresses genuine uncertainty. It acknowledges it may not
know, hedges its answer, or explicitly signals it is guessing. The model “knows it doesn’t know.” Look
for: hedging (“I’m not entirely sure. . . ”, “I think it might be. . . ”, “I’m not certain”), explicit admissions
of ignorance (“I couldn’t find information on this”, “I don’t have enough information”), or speculative
language (“probably”, “perhaps”, “I believe but am not sure”).
IMPORTANT:Ignore the final Confidence number. Judge solely on the reasoning text.
Question:{question}
Model response:
{response}
Gold answer:{gold_answer}
Respond in this exact format:
Classification: EPISTEMIC | ALEATORIC
Reasoning: <one sentence explaining the key signal in the response text>
D.2 Qualitative Examples of Epistemic and Aleatoric Errors
26

Example 1: Baseline epistemic error — confident hallucination (HotpotQA)
Question:“Glad to Be Unhappy” is a song composed by what American?
Gold answer:Richard Charles RodgersPrediction:Randy NewmanConfidence:0.9
Model response:
To answer this question, I’ll do some research. . .
According to my sources, “Glad to Be Unhappy” is a song composed by Randy Newman.
Answer: Randy Newman Confidence: 0.9
Error type: Epistemic— the model asserts a specific, factually incorrect composer with no hedging or
uncertainty in the reasoning text. The phrase“According to my sources”signals false confidence: the model
presents a retrieved-sounding fact it does not actually possess, a hallucination that calibration training must
suppress.
Example 2: Residual epistemic error in the calibrated model — wrong intermediate fact (2WikiMulti-
HopQA)
Question:Are the filmsJe Suis Né D’Une CigogneandLa Chair De L’Orchidéefrom the same country?
Gold answer:YesPrediction:NoConfidence:0.9
Model response:
Step 1: Identify the countries of origin for each film.
•Je Suis Né D’Une Cigogneis from France.
•La Chair De L’Orchidéeis from Belgium.
Step 2: Since France and Belgium are not the same country, the answer is No.
Answer: No Confidence: 0.9
Error type: Residual epistemic— calibration substantially reduces but cannot fully eliminate confident
errors. Here, the second film is misattributed to Belgium (both are French productions), yet the reasoning
chain is internally coherent and therefore sustains high confidence. This is the characteristic failure mode for
in-distribution multi-hop questions: a wrong intermediate fact propagates to a wrong but confidently stated
conclusion.
Example 3: Aleatoric error in the calibrated model — explicit uncertainty signal (Natural Questions)
Question:When did Spanish Town become Jamaica’s capital?
Gold answer:1534Prediction:1704Confidence:0.05
Model response:
Step 1: Identify the location “Spanish Town. ”
Step 2: According to my knowledge, Kingston is the capital of Jamaica. Spanish Town is a town in Jamaica,
but I’m not sure if it has ever been the capital.
Answer: 1704 Confidence: 0.05
Error type: Aleatoric— despite producing an incorrect answer, the model’s reasoning text explicitly signals
ignorance (underlined), independently of the final confidence number. This is the target behaviour of calibration
training: the model hedges rather than hallucinating, making its uncertainty available to downstream systems for
abstention or retrieval augmentation.
Figure 16: Qualitative examples of epistemic and aleatoric errors before and after GRPO calibration
training.Epistemic errors(Examples 1–2) are wrong answers delivered with high stated confidence
and no uncertainty signal in the response text; they represent knowledge gaps the model fails to
recognise.Aleatoric errors(Example 3) are wrong answers accompanied by explicit hedging in
the reasoning, indicating the model correctly identifies the limits of its knowledge. Error type is
determined by the response content independently of the confidence number, motivating the use of an
LLM judge (see §D.1) in addition to the verbalized confidence threshold.
27

Example 1: Baseline epistemic error — absent uncertainty signal (HotpotQA)
Question:Who released the 2012 record “Red”?
Gold answer:Big Machine RecordsPrediction:Taylor SwiftEmitted<uncertain>:No
Model response:
Let’s break this down step by step!
Red is an album by Taylor Swift, an American singer-songwriter.
Answer: Taylor Swift
Error type: Epistemic— the model confuses the recording artist with the releasing label and delivers the wrong
answer without any<uncertain>signal in the reasoning chain. Even though the system prompt instructs the
model to emit <uncertain> when uncertain, the baseline has not learned to use it: its output is indistinguishable
from a confident correct answer.
Example 2: Residual epistemic error in the calibrated model — token absent despite wrong answer
(TriviaQA)
Question:In which sport would you find positions called driver, bucket, hole, and point?
Gold answer:Water poloPrediction:GolfEmitted<uncertain>:No
Model response:
Let’s break this down step by step!
The sport is golf.
Answer: golf
Error type: Residual epistemic— the model commits to a confidently wrong answer in a single step, skipping
any deliberation that could trigger <uncertain> . The extreme brevity of the response (no intermediate
reasoning) is itself diagnostic: there is no reasoning chain in which uncertainty could surface. This reflects a
residual failure of the <uncertain> training signal: the model’s prior confidence is so high for simple-seeming
questions that no token is inserted.
Example 3: Aleatoric error in the calibrated model — <uncertain> emitted at knowledge boundary
(MuSiQue)
Question:Where was the founder of the National Women’s Party born?
Gold answer:Mount Laurel TownshipPrediction:<uncertain>(abstain)Emitted<uncertain>:Yes
Model response:
Let’s break this down step by step!
The National Women’s Party was founded in 1916. I found that the party was founded by Alice Paul, an
American suffragist and women’s rights activist.
However, I couldn’t find the birthplace of Alice Paul.<uncertain>
According to some sources, Alice Paul was born in 1885, but I don’t have the exact birthplace.
Answer:<uncertain>
Error type: Aleatoric— the model correctly resolves the first reasoning hop (National Women’s Party →Alice
Paul) but reaches a genuine knowledge boundary on the second hop (Alice Paul’s birthplace). The <uncertain>
token is emitted precisely at the step where knowledge runs out, and the model propagates this uncertainty to
the final answer rather than hallucinating. This is the intended behaviour: a structured abstention that exposes
the exact reasoning step at which information is missing.
Figure 17: Qualitative examples for the <uncertain> token checkpoint, which signals uncertainty
via a discrete trainable token rather than a verbalized confidence score.Epistemic errors(Examples
1–2) occur when the model produces a wrong answer without emitting <uncertain> in the reasoning
chain; the baseline (Example 1) never uses the token, while the trained model (Example 2) still
fails to trigger it for short, over-confident responses.Aleatoric errors(Example 3) occur when
<uncertain> is correctly inserted at the knowledge boundary, signalling that the error is not a
hallucination but an honest absence of information. Unlike the verbalized confidence setting (Fig. 16),
the uncertainty signal here isbinary and localised: the token either appears or it does not, and its
position in the chain-of-thought identifies the specific reasoning step that fails.
28