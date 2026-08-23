# GRIP: Grounded Reasoning via Information-Restricted Premises

**Authors**: Lirui Teng

**Published**: 2026-08-17 16:23:49

**PDF URL**: [https://arxiv.org/pdf/2608.16776v2](https://arxiv.org/pdf/2608.16776v2)

## Abstract
High-capacity encoders in retrieval-augmented generation (RAG) can let the query dominate the latent state, leaving retrieved evidence functionally irrelevant. We call this failure mode query dominance. To address it, we introduce \textbf{GRIP} (Grounded Reasoning via Information-Restricted Premises), which imposes capacity asymmetry: the decoder keeps full-dimensional access to the query, while retrieved evidence passes through a severe stochastic bottleneck. This forces the evidence channel to encode only the residual information unavailable from the query. Across five reasoning benchmarks, GRIP outperforms strong iterative baselines, cuts a query--latent mutual-information diagnostic by roughly 30$\times$ (14.8 $\to$ 0.47 bits), and reduces hallucination by 73\%. Residual-alignment analysis further shows that the bottleneck output occupies subspaces less aligned with the query than baseline representations.

## Full Text


<!-- PDF content starts -->

GRIP: Grounded Reasoning via
Information-Restricted Premises
Lirui Teng
University of Waterloo, Waterloo ON, Canadalteng@uwaterloo.ca
Abstract.High-capacity encoders in retrieval-augmented generation
(RAG) can let the query dominate the latent state, leaving retrieved
evidence functionally irrelevant. We call this failure mode query dom-
inance. To address it, we introduceGRIP(Grounded Reasoning via
Information-Restricted Premises), which imposes capacity asymmetry:
the decoder keeps full-dimensional access to the query, while retrieved
evidence passes through a severe stochastic bottleneck. This forces the
evidence channel to encode only the residual information unavailable
from the query. Across five reasoning benchmarks, GRIP outperforms
strong iterative baselines, cuts a query–latent mutual-information diag-
nostic by roughly 30×(14.8→0.47 bits), and reduces hallucination
by 73%. Residual-alignment analysis further shows that the bottleneck
output occupies subspaces less aligned with the query than baseline rep-
resentations.
Keywords:Retrieval-augmented generation·Information bottleneck·
Grounded reasoning·Query dominance·Mutual information
1 Introduction
Retrieval-augmented generation (RAG) is intended to condition language mod-
els on external evidence, approximatingP(Y|Q, E). In practice, LLMs often
under-use retrieved text and fall back on parametric knowledge, even when it
conflicts with the evidence [14]. The failure is representational:Qenters the de-
coder through a high-capacity path while evidence shares the same latent space,
so optimization—already finding a low-loss solution underP(Y|Q)—treats ev-
idence as a marginal correction. We call the resulting regimequery dominance.
Existing methods intervene at decoding or supervision time but leave the
latent geometry of query–evidence fusion largely unchanged. Self-RAG adds re-
flection tokens for retrieval critique [2]; context-aware decoding reweights token
probabilities toward retrieved content [17]; RAFT-style training teaches models
to ignore distractors [23]. High-dimensional representations can therefore still
allocate most capacity to query-aligned features and parametric shortcuts [6,7],
leaving retrieval under-utilisation as a capacity-allocation problem rather than
only a content-selection one.
We address this gap through deliberatecapacity asymmetryrather than
richer fusion.GRIP(Grounded Reasoning via Information-Restricted Premises;
arXiv:2608.16776v2  [cs.AI]  19 Aug 2026

2 L. Teng
(A) Standard RAG I(Q;z)≈14.8 bits
QueryQ
“Where was Carol Reed born?”
EvidenceE
“Reed was born in Putney.”EabsorbedDecoder“Putney”
×parametric prior
(B) GRIP(capacity asymmetry)
I(Q;z k) = 0.47 bits
QueryQ
EvidenceEfull-dimensional bypass (d model )
frozen:retriever·extractor·NLI gate
verified spanp k→C k+1 (text, next step)Decoder
(Q, C k, zk)“Putney”
✓grounded
Fig.1.Information flow in (A) standard RAG versus (B) GRIP. Ribbon width is
proportional to channel capacity. (A) Evidence enters the same full-dimensional latent
as the query and is absorbed by the query-aligned flow, leaving answers driven by
the parametric prior. (B) The query retains a full-dimensional bypass while evidence—
produced by a frozen retrieve–extract–verify pipeline—is squeezed through a stochastic
dz=4bottleneck (≈2–4 bits/step); the entailment-verified span additionally persists as
text inC k+1(Sect. 4.5). Boxed values give measured query–latent mutual information
(Table 3).
Fig. 1) routesQthrough a full-dimensional bypass while forcing retrieved evi-
dence through an aggressively low-dimensional, stochastic bottleneck (d z≈4).
Becausethedecoderretainshigh-bandwidthaccesstoQ,query-correlatedbitsin
the bottleneck are redundant under a tight capacity budget; gradients pressure
the channel to transmit only theinformation residual—the signal the query does
not already provide. Per query, GRIP operates on a cumulative total of roughly
25 entailment-verified evidence tokens across its two reasoning steps (averaging
∼11tokens per step), compared to the baseline’s∼4,000-token raw retrieved
context.
Contributions.First, we formalise query dominance as a failure of conditional
independence and introduceQuery–Latent (QL) Dependence, the mutual infor-
mationI(Q;z k)between the query and the evidence representation, as a model-
agnostic diagnostic: elevated QL dependence indicates thatz khas collapsed into
a compressed copy of the query and coincides with elevated hallucination. Sec-
ond, we introduce GRIP, which enforces low-capacity, noise-regularised evidence
representations that we argue are consistent with an information-residual mech-
anism. Third, GRIP outperforms strong baselines on HotpotQA, StrategyQA,
2Wiki, ProofWriter, and SQuAD 2.0, reducing QL dependence by roughly20–
37×,suppressinghallucinationby73%,andproducingresidualsmoreorthogonal
to the query than unconstrained baselines.

GRIP: Grounded Reasoning via Information-Restricted Premises 3
2 Query Dominance in Latent States
Although RAG is intended to approximateP(Y|Q, E), language models in
practice often behave closer toP(Y|Q): they generate from parametric knowl-
edge while showing limited sensitivity to the retrieved context [3,10,14,17,23],
a phenomenon that recent analyses show worsens with model capacity. We re-
fer to the representational form of this failure asquery dominance: the latent
state used by the decoder remains highly predictable from the query and only
weakly responsive to evidence variation: retrieval is present in the pipeline but
functionally marginal in generation.
2.1 The Parametric Prior Pathology
This failure is encouraged by the geometry of pretrained transformer representa-
tions: contextual embeddings are anisotropic, with a few high-variance directions
carrying broad semantic and frequency information [6]. Query features tend to
occupy these dominant directions, so retrieved evidence—even when relevant—is
treated as a perturbation to an already strong query-conditioned trajectory, a
form of shortcut learning in which a dominant signal suppresses genuinely joint
representations [7].
2.2 Diagnosing Query Dominance
We model an evidence-conditioned system as
ˆY=G(z, Q), z=ϕ(Q, E),(1)
wherezis the internal evidence representation passed to the decoder. Let
F(Q, E) =G(ϕ(Q, E), Q)denote the end-to-end output map. We assume that,
for fixedQ, the effect of evidence on the output is mediated throughz; that
G(·, Q)is locally Lipschitz under the chosen output discrepancy metricd; and
that representations are bounded. Under these assumptions, when contrastive
evidence produces little separation in latent space, the decoder cannot produce
large output separation either: query dominance can arise from collapse or re-
dundancy inϕ(Q, E)itself.
For a fixed queryq, letE+
qandE−
qdenote contrastive evidence distribu-
tions that support different task-level answers. We definecontrastive evidence
sensitivityas
SE(q) =Ee+∼E+
q, e−∼E−
q
d 
F(q, e+), F(q, e−)
.(2)
A model is behaviourally query-dominant atqwhenS E(q)is small while
the model remains sensitive to changes in the query. To rule out triv-
ial constant-output collapse, we define query-swap sensitivityS Q(q) =
Eq′∼Q, e∼E+
q[d(F(q, e), F(q′, e)) ]and say that query dominance occurs when
SE(q)≤τ EandS Q(q)≥τ Qfor thresholdsτ E≪τQ.

4 L. Teng
At the representation level, we measureQuery–Latent (QL) Dependence:
DQL=I(Q;z k).(3)
HighD QLindicates that the evidence-channel statez kis strongly predictable
from the query, suggesting that it carries query-aligned information rather than
evidence-specific content. In a well-conditioned retrieval system,z kshould in-
stead encode conditional innovation: information supplied by evidence that is
not already available fromQ. We estimateI(Q;z k)using the Contrastive Log-
ratio Upper Bound (CLUB) estimator [4]. Because neural MI estimators ex-
hibit dimension-dependent bias [5] and can violate basic self-consistency proper-
ties [18], we rely on relative comparisons across conditions rather than absolute
values. A complementary shuffle control—breaking query–evidence correspon-
dence and verifying that the estimate collapses toward zero—can further vali-
date the estimator. QL dependence is a necessary but not sufficient diagnostic—
randomization, collapse, or decoder null-space effects can also reduceI(Q;z k)—
so we pair it with behavioural randomization tests and residual-alignment mea-
surements in Section 5.
3 Design Principle: Capacity-Asymmetric Evidence
Given the query-dominant regime described above, the central design question
is not only which evidence to retrieve, but how much representational capac-
ity the evidence pathway should receive. GRIP adopts a capacity-asymmetric
principle: the query and reasoning context retain full-dimensional access to the
decoder, while evidence is routed through a deliberately restricted channel. The
goal is not to remove the query signal, but to prevent the evidence representa-
tion from cheaply duplicating information already available through the query
path. GRIP thus differs from evidence-compression methods such as xRAG, CO-
COM, PISCO, and gist tokens, which compress context to maximise retention
and efficiency, whereas GRIP restricts evidence capacity to counteract query
dominance.
Information-bottleneck approaches to RAG.Zhu et al. [24] filter retrieval noise
by maximising mutual information between a compressed representation and the
output while minimising it with the passage—a largely deterministic noise filter.
Swin-VIB [21] integrates variational IB models that adaptively regulate evidence
compression to guide an LLM under knowledge conflicts. GRIP differs in three
respects: (i) deliberate capacityasymmetry(full-dimensional query access versus
a severely restricted evidence channel) rather than an adaptive conflict adapter;
(ii) a fixed, severe stochastic bottleneck (d z=4, additive Gaussian noise,≈2–4
bits per step) as a first-class design principle rather than a learned compres-
sion ratio; and (iii) a motivation of counteracting query dominance rather than
arbitrating knowledge conflicts.

GRIP: Grounded Reasoning via Information-Restricted Premises 5
3.1 Capacity-Limited Evidence Representations
Abstractly, GRIP maps an extracted, verified premisep kto a low-dimensional
noisy state
zk=B θ(pk) +ε k, ε k∼ N(0, σ2Idz), d z≪d model,(4)
whereB θdenotes the evidence compressor. Under the standard Gaussian chan-
nel approximation,z kcannot transmit unrestricted premise information; Sec-
tion 4.3 gives the explicit capacity bound at the implementation level. Low
dimensionality limits transmittable features, premise-level compression reduces
passage verbosity, and additive noise discourages deterministic copying of brit-
tle correlations—together making it inefficient forz kto serve as a second query
representation.
3.2 Residualization Pressure
Under a tight evidence capacity budget, query-correlated information inz khas
an opportunity cost: capacity spent on query-predictable features is inefficient
unless those features also help predictYgivenQ—exactly the trade-off captured
by the Conditional Information Bottleneck objective:
LCIB=−I(Z k;Y|Q) +βI(Z k;Q).(5)
GRIPdoesnotoptimizeEq.(5)explicitly;thecapacityasymmetrycreatesasim-
ilar pressure—preserve evidence information predictive ofYwhile discouraging
redundant query information—which we treat as a mechanism-level interpreta-
tion, not a formal equivalence.
3.3 Expected Diagnostic Effects
The capacity-asymmetry hypothesis yields three empirical predictions. First,
query–latent dependence should decrease:I(Q;z k)GRIP≪I(Q;z k)RAG. Second,
if the decoder genuinely relies on the restricted evidence channel, randomizing
zkshould cause a larger performance drop than the analogous intervention on
baselines:∆GRIP
rand > ∆baseline
rand, where∆ rand = Acc(z k)−Acc(˜z k)and˜z kis a
randomized bottleneck state. Third, residual-alignment measurements should
show thatz koccupies subspaces less aligned with query-dominant directions
than baseline representations.
Section 5 tests all three predictions: QL dependence and∆ randacross the
five benchmarks, with the architecture-matched Llama-3 Iterative control on
HotpotQA, andρin Fig. 3 and Table 3. Convergence of the three diagnostics, to-
getherwiththeablationpatternofSection5.3,supportsthecapacity-asymmetry
account.

6 L. Teng
QueryQ ContextC k dashed = frozen solid = trainable
corpus
MextretrieverDPRentropy
re-rankextractor
RoBERTaNLI gateP>0.75Decoder
(Q, C k, zk)pk zk
inferencei k iteration:C k+1←C k⊕(p k, ik) (span persists as text)
Fig.2.GRIP implementation pipeline: at each step, candidate passages are retrieved
and entropy-ranked, reduced to a predictive span, filtered by an NLI verifier, com-
pressed toz k, and passed to the decoder alongside the full-dimensional query and
context.
4 Architecture
Thecapacity-asymmetricprincipleofSection3isrealisedasafour-stagepipeline
(Fig. 2): iterative retrieval, predictive span extraction, stochastic compression,
and asymmetric decoding. The retriever, extractor, and verifier are frozen; gra-
dients flow only through the bottleneck and decoder.
4.1 Retrieval via Entropy-Guided Re-ranking
A dense passage retriever [11] returns top-mcandidates, which are re-ranked by
the conditional entropy of the next-step prediction:
s(r|C k, Q) =−H Θ 
ik|r, C k, Q
.(6)
Passages making the next step more predictable score higher; entropy is com-
puted under teacher-forced decoding through the candidate. A curriculum defers
the entropy criterion until the decoder is calibrated (Section 4.6).
4.2 Predictive Span Extraction
The selected passager∗(k)is reduced to a minimal predictive span before
it reaches the bottleneck. A frozen RoBERTa-based extractorΘ extproduces
pk⊂r∗(k)by KL-matching the next-step distribution conditioned on the span
versus the full passage, with a length-sparsity penalty (full objective in the sup-
plement). A frozen DeBERTa-v3-large NLI verifier admits only spans satisfying
PNLI(entailment|r∗(k), pk)>0.75. To prevent premise representations from
becoming query-dependent, query tokens are masked in the extractor’s cross-
attention, isolatingΘ ext’s gradients from query embeddings.

GRIP: Grounded Reasoning via Information-Restricted Premises 7
4.3 Stochastic Bottleneck
The verified span is then mapped to a low-dimensional latent through a noisy
projection:
zk=W 2σ 
W1·pool(p k)
+εk, ε k∼ N(0, σ2Idz),(7)
withd z= 4,σ2= 1.0, mean pooling, and ReLU activation. For ad z-dimensional
channel with additive Gaussian noise, the Gaussian channel capacity gives
I(pool(p k);zk)≤dz
2log 
1 +P/σ2
,(8)
wherePis the average power of the projected pre-noise vector. WithP≈1
after normalisation, the per-step budget is approximately 2–4 bits—the explicit
form of the capacity asymmetry argued for in Section 3.
4.4 Asymmetric Decoding
The decoder conditions on(Q, C k, zk): the query enters as a high-bandwidth
prefix, the contextC kcarries prior reasoning steps, andz kis injected as a single
special token with a learned positional embedding. BecauseQandC kenter at
fulldimensionalitywhilez kisconstrainedtofourdimensionswithadditivenoise,
the query/context and bottleneck pathways differ in capacity by roughly three
orders of magnitude.
4.5 Forward-Pass Procedure
Algorithm 1 composes the four stages into a step-wise inference loop; entropy-
guided selection (Eq. (6)) is a non-differentiable inference-time decision based
on the decoder’s state.
The context updateC k+1←C k⊕(p k, ik)retains the verified span as text,
so the decoder keeps a full-dimensional semantic pathway alongside the bot-
tleneck. This is deliberate rather than a leak: ablating the raw-span pathway
costs8.2accuracy points (Section 5.3), while at stepkthe decoder commits to
inferencei kbeforep kbecomes contextually available, and randomisingz kstill
costs35.3points with a71.6:1ratio of correct-to-wrong versus wrong-to-correct
transitions (Section 5.4). The bottleneck thus regulates the evidence signal gov-
erning each inference step; the persistent span text carries sentence-level seman-
tics but cannot substitute for the bottleneck state. What persists is the minimal
entailment-verified span, not the retrieved passage.
4.6 Training Objective and Schedule
The trainable parametersΘ gen={W 1, W2, Θdec}maximise the likelihood of the
target reasoning steps,
Ltask=−KX
k=1logP Θgen(ik|zk, Ck, Q),(9)

8 L. Teng
Algorithm 1GRIP Step-Wise Inference
Require:QueryQ; corpusM ext; frozen modules (retriever,Θ ext,NLI); trained mod-
ules (Bottleneck,Decode); max stepsK; entailment thresholdτ
Ensure:Final answerˆa
1:C 1← ∅▷empty reasoning context
2:fork= 1, . . . , Kdo
3:R(k)
m←DenseRetrieve(Q, C k)
4:r∗(k)←arg minr∈R(k)
mHΘ(ik|r, C k, Q)
5:p k←Θ ext(r∗(k), Ck)
6:ifNLI(p k, r∗(k))< τthen
7:continue▷discard stepk; advance with context unchanged
8:end if
9:z k←Bottleneck(p k) +ε k, ε k∼ N(0, σ2Idz)
10:i k←Decode(Q, C k, zk)
11:C k+1←C k⊕(p k, ik)
12:ifi kemits[answer]then
13:returni k ▷early termination
14:end if
15:end for
16:returni K ▷fallthrough: no[answer]withinKsteps
under a two-phase curriculum. In Phase 1 (epochs 1–5) the entropy re-ranker is
bypassedandtoppassagesareselectedbydenseretrievalalone,allowingthebot-
tleneck and decoder to stabilise before the entropy signal becomes load-bearing.
In Phase 2 (epochs 6–20), entropy-guided selection (Eq. (6)) is enabled while
the retriever, extractor, and NLI verifier remain frozen throughout. BecauseQ
reaches the decoder at full dimensionality through the bypass, query-redundant
featuresinz kdonotreducetheconditionallikelihoodandareprunedbygradient
descent [1].
5 Experiments
We evaluate GRIP on five reasoning benchmarks to test whether capacity-
asymmetric evidence processing improves task performance and evidence use.
5.1 Experimental Configuration
Baselines.We compare GRIP against three baselines spanning the matched-
control and prior-art axes.Standard RAGuses DPR retrieval [11] with pas-
sage concatenation and Llama-3-8B decoding [12].Self-Ask[15] uses iterative
sub-question prompting without modifying the evidence pathway.Llama-3-8B
Iterativeis the architecture-matched control: it follows GRIP’s two-step rea-
soning schedule (K=2, same retriever, entropy re-ranking, span extraction, and
NLI gate) but injects the verified premisep kinto the decoder as ordinary full-
dimensional text rather than through the stochastic bottleneck, isolating the

GRIP: Grounded Reasoning via Information-Restricted Premises 9
contribution of capacity asymmetry from that of iterative reasoning. Decoding
hyperparameters match GRIP.
All systems share the same frozen DPR-Wiki index, tokenization, and com-
pute budget. GRIP performsK= 2reasoning steps withm= 10retrieved
passages per step. The shared component is the retrieval substrate (retriever
and index); the evidence reaching each decoder still differs after re-ranking, ex-
traction,andNLIfiltering,sothecomparisonisolateshoweachmethodprocesses
evidence. Descriptive statistics of the retrieval–verification pipeline are reported
in the supplement.
Datasets.We evaluate on five benchmarks covering different reasoning regimes:
HotpotQA[22] for distractor multi-hop QA,StrategyQA[8] for implicit
multi-step reasoning,2WikiMultihopQA[9] for explicit two-hop reasoning,
ProofWriter[19] for symbolic Horn-clause deduction, andSQuAD 2.0[16]
for single-hop extractive QA. Primary metrics are exact match (EM) or task
accuracy, with F1 where applicable (Table 1). Hallucination is the percentage of
generated claims not entailed by retrieved evidence. The in-pipeline DeBERTa-
v3 verifier that scores entailment is also the training-time selection signal; an in-
dependent verifier provides an evaluation check on this circularity (Section 5.2).
Atomicity of extracted premises is reported in the supplement.
Optimization.Trainable components use AdamW [13] (lr10−4, weight decay
0.01, 1,000-step warmup, cosine decay, gradient clipping 1.0, global batch 128),
with nucleus sampling (p=0.9,T=0.7), trained for 20 epochs on4×A100 80GB
GPUs. Reported results are averaged over three random seeds; on HotpotQA
the standard deviation over seeds is±0.45EM,±0.38F1,±0.72hallucination,
and±0.04bits QL dependence.
5.2 Main Results
Table 1 reports task performance across all five datasets. GRIP improves
over the strongest non-GRIP baseline on every dataset, and outperforms the
architecture-matched Llama-3 Iterative control on all five benchmarks—by+7.2
EM on HotpotQA and+4.1accuracy points on StrategyQA—indicating that
capacity asymmetry contributes beyond the iterative reasoning schedule. Paired
bootstrap comparisons are significant atp <0.01on HotpotQA (+7.2) and
SQuAD 2.0 (+3.7). Self-Ask is competitive on the multi-hop settings but trails
Standard RAG on single-hop SQuAD 2.0 (76.5 vs. 78.4 EM), consistent with
decomposition overhead when explicit multi-hop decomposition is unnecessary.
Hallucination drops substantially in every condition—from 31.7% to 8.6% on
HotpotQA, from 31.2% to 9.8% on 2Wiki, and from 28.7% (Llama-3 Iterative)
to 8.6% (GRIP) under the matched control. This grounding result is robust to
the choice of verifier: rescoring with MiniCheck [20] yields 89.0% agreement with
the in-pipeline verifier (Cohen’sκ= 0.77), and the HotpotQA hallucination rate
rises only from 8.6% to 10.1%, remaining well below every baseline—the gains
are thus not verifier-specific, though this does not establish model-agnosticism.

10 L. Teng
Table 1.Task performance across five reasoning benchmarks. EM/F1 follow standard
conventions for HotpotQA, 2Wiki, and SQuAD 2.0; StrategyQA reports accuracy only
(yes/no); ProofWriter reports proof accuracy. Em dashes denote metrics not applicable
(accuracy-only tasks).
Dataset Model EM/Acc F1 Hall. (%)↓
HotpotQAStandard RAG 68.2 72.5 31.7
Self-Ask 71.3 75.8 19.8
Llama-3 Iterative 69.3 73.6 28.7
GRIP 76.5 80.3 8.6
StrategyQAStandard RAG 65.2 – 33.4
Self-Ask 67.5 – 32.1
Llama-3 Iterative 69.3 – 28.7
GRIP 73.4–10.1
2WikiStandard RAG 62.8 68.4 31.2
Self-Ask 66.3 71.5 19.7
Llama-3 Iterative 64.8 69.8 28.4
GRIP 71.2 76.1 9.8
ProofWriterStandard RAG 74.3 – 26.1
Self-Ask 78.5 – 15.9
Llama-3 Iterative 77.0 – 23.7
GRIP 85.6–6.8
SQuAD 2.0Standard RAG 78.4 82.1 18.2
Self-Ask 76.5 80.3 19.4
Llama-3 Iterative 78.0 82.3 17.7
GRIP 82.1 85.7 6.4
5.3 Ablation Studies
Table 2 reports component and capacity ablations on HotpotQA. The bottleneck
is the most load-bearing component: removing it raises QL dependence from
0.47 to 14.20 bits and reduces accuracy by 5.3 points. Removing extraction or
NLI verification degrades performance through different failure modes: without
extraction, verbose passage content enters the bottleneck and hallucination rises;
without NLI verification, unsupported spans are admitted, raising hallucination
while leaving QL dependence low.
The capacity sweep is non-monotonic. A very narrow bottleneck (d z= 2)
suppresses QL dependence most strongly but loses task-relevant evidence; a
wider one (d z= 16) restores capacity but allows query-redundant information
to re-enter. The best configuration is therefore not the smallest channel, but the
channel that balances residual evidence transmission against query redundancy.
The mechanism controls in Table 2 follow the deterministic-versus-stochastic
pairing of information-bottleneck designs in prior RAG work [21,24]. Neither di-
mensionalrestrictionnorstochasticcorruptionaloneissufficient:atfixedd z= 4,
removing stochasticity increases QL dependence from 0.47 to 2.38 bits and hallu-
cination by 4.6 points, while retaining stochasticity at full dimension (d= 4096)
raises QL dependence to 10.85 bits and hallucination to 24.7%. The combina-

GRIP: Grounded Reasoning via Information-Restricted Premises 11
Table 2.Component and capacity ablations on HotpotQA.∆reports accuracy change
relative to full GRIP. Component ablations isolate individual modules; capacity abla-
tions vary the bottleneck widthd zat fixed noiseσ2= 1.0. For “No bottleneck”,I(Q;z k)
is measured on the pooled premise embedding that replacesz k(no low-rank projection
or noise).
ConfigurationI(Q;z k)(bits)↓Hall. (%)↓Acc.∆
Full GRIP(d z= 4)0.47 8.6 76.5–
Component ablations
No bottleneck 14.20 24.3 71.2−5.3
No extraction 3.10 18.7 73.4−3.1
No NLI verification 0.52 12.1 75.1−1.4
Capacity ablations
dz= 20.31 9.2 74.2−2.3
dz= 81.82 8.9 75.1−1.4
dz= 164.73 11.5 72.8−3.7
Mechanism controls
Deterministic (d z= 4,σ2=0) 2.38 13.2 74.0−2.5
Noise-only (d=4096, stochastic) 10.85 24.7 72.1−4.4
tion of restricted capacity and stochastic encoding is therefore critical to GRIP’s
information-control behaviour.
Two bypass ablations complete the picture. Removing the query bypass (the
decoder receivesz kandC kbut no full-dimensional access toQ) causes severe
generation degradation of 23–33 accuracy points across the five datasets even
though QL dependence stays at 0.45 bits: the 4D stochastic channel alone is
too narrow to carry the semantic burden of generation. Removing the raw-text
bypass instead (the decoder receivesQ, the inference history, andz k, but no
persisted span text) costs 8.2 points on HotpotQA/2Wiki and raises hallucina-
tion to 18.4%. The architecture thus requires both restricted evidence flow and
high-capacity semantic access.
5.4 Mechanism Diagnostics
Because aggregate accuracy does not reveal whether the bottleneck changes evi-
dence use, we report three diagnostics: QL dependenceI(Q;z k), estimated with
the CLUB upper bound [4]; the randomization drop∆ rand= Acc(z k)−Acc(˜z k),
where˜z kis sampled from the empirical marginal of bottleneck states; and resid-
ual alignmentρ(z k,Q) =∥projQ(zk)∥2
2/∥zk∥2
2, whereQis the top principal-
component subspace of query embeddings (90% of variance, 10K-sample valida-
tion pool). Table 3 consolidates all three diagnostics across the five datasets for
GRIP, the matched control, and Standard RAG.
GRIP reduces QL dependence by20×–37×across all five benchmarks, track-
ing the capacity constraint rather than dataset-specific properties. Low QL de-
pendence alone, however, does not prove evidence use—a collapsed state would

12 L. Teng
Fig.3. Empirical CDFs of evidence–query alignment on HotpotQA(lowerρ
⇒weaker query alignment). The GRIP distribution (solid, deep blue) dominates the
Llama-3 Iterative baseline (dashed, gray) at every threshold: atρ= 0.22, 77% of GRIP
samples fall below this threshold versus∼8% of baseline samples, so the reduction is
systematic rather than outlier-driven. Inset: meanρper method (ρcolumn of Table 3).
Table 3.Mechanism diagnostics across datasets, as defined in Section 5.4: lower
I(Q;z k)⇒less query-redundant representation; higher∆ rand⇒stronger decoder de-
pendence on evidence; lowerρ⇒weaker geometric query alignment. Em dashes denote
diagnostics not run (∆ randfor Standard RAG).
Dataset ModelI(Q;z k)(bits)↓∆ rand↑ρ↓
HotpotQAStandard RAG 14.8 – 0.72
Llama-3 Iterative 11.2 7.5 0.61
GRIP 0.47 35.3 0.18
StrategyQAStandard RAG 13.2 – 0.68
Llama-3 Iterative 10.1 4.2 0.58
GRIP 0.52 30.5 0.19
2WikiStandard RAG 15.1 – 0.74
Llama-3 Iterative 11.5 6.8 0.63
GRIP 0.41 35.0 0.17
ProofWriterStandard RAG 11.8 – 0.69
Llama-3 Iterative 9.2 8.1 0.56
GRIP 0.38 42.5 0.13
SQuAD 2.0Standard RAG 12.4 – 0.78
Llama-3 Iterative 10.4 5.5 0.65
GRIP 0.61 22.5 0.24
also have low mutual information with the query. The randomization test re-
solves this: replacingz kwith samples from its empirical marginal drops GRIP
accuracy by 35.3 points on HotpotQA against only 7.5 for the matched-control
Llama-3 Iterative baseline, ruling out the iterative schedule as the source of
evidence dependence. Both diagnostics generalise across all five datasets:∆ rand
rangesfrom42.5pointsonProofWriter,wheretheeffectisstrongest,toaweakest
but still substantial 22.5 points on SQuAD 2.0—in every case several times the
matched control’s 4.2–8.1-point drop—withρcorrespondingly low (0.13–0.24).

GRIP: Grounded Reasoning via Information-Restricted Premises 13
Decomposing the HotpotQA randomization drop at the sample level, 35.8% of
predictions flip from correct to wrong against only 0.5% from wrong to correct
(40.7% remain correct, 23.0% remain wrong)—a71.6:1destructive-to-corrective
ratioconsistentwith∆ rand= 35.3.Randomizationthusoverwhelminglydestroys
correct predictions rather than causing symmetric churn, though it does not by
itself separate marginal failure from collapse. Geometrically, GRIP retains 18%
of its bottleneck energy in the query subspace versus 61% for Llama-3 Iterative
and 72% for Standard RAG. The three diagnostics converge and, with Table 2’s
ablation and control pattern, are consistent with the capacity-asymmetry ac-
count; counterfactual, unanswerable, and inconsistent-context evaluations are
reported in the supplement.
6 Limitations
Mechanism ambiguity.We do not prove that the bottleneck enforces con-
ditional residualisation. The deterministic and noise-only controls (Section 5.3)
establish that neither dimensional restriction nor stochasticity alone reproduces
GRIP’s information-control behaviour, and the bottleneck output occupies sub-
spacesweaklyalignedwiththequery.Whatremainsopenisaformalisedcounter-
factual evaluation protocol, generalisation beyond the single Llama-3-8B back-
bone, and reliable CLUB estimation below roughlyN= 500samples.
MI estimator.CLUB is a loose upper bound with dimension-dependent
bias [5], and variational MI estimators can violate basic self-consistency proper-
ties [18]; we assume the bias is approximately consistent across models so that
relative comparisons remain meaningful.
Dataset dependence.SQuAD 2.0 and StrategyQA may overlap with
Llama-3-8B’s parametric knowledge, so hallucination reductions there cannot
be cleanly attributed to evidence use versus elicitation of stored knowledge;
HotpotQA and 2Wiki more cleanly probe evidence dependence.
Scope and failure modes.GRIP assumes explicit separation between
queryandevidencepathways;architecturesthatfusethemearliermaynotadmit
thesamemechanism.Severecompressionatd z= 4alsotradesrare-entityfidelity
for redundancy suppression: rare entities can be lost when their distinguishing
features fall outside the retained subspace, with a frequent near-neighbour sub-
stituted. Implicit reasoning that requires high-bandwidth intermediate represen-
tations may be similarly limited.
7 Conclusion
GRIP routes evidence through a low-dimensional, noisy bottleneck while retain-
ing a high-capacity query bypass; across five benchmarks it reduces estimated
query–bottleneck mutual information by roughly 30×, decreases hallucination
by 73%, and improves accuracy by 8.0 points on average over Standard RAG.
Residual-alignment analysis shows the bottleneck output is weakly aligned with
the query, and mechanism controls show neither dimensional restriction nor

14 L. Teng
stochasticity alone reproduces this behaviour: their combination is the opera-
tive design principle.
Full appendices and additional diagnostics are provided in an online supple-
ment.
References
1. Achille, A., Soatto, S.: Emergence of invariance and disentanglement in deep rep-
resentations. Journal of Machine Learning Research19(50), 1–34 (2018)
2. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-RAG: Learning to retrieve,
generate, and critique through self-reflection. In: The Twelfth International Con-
ference on Learning Representations (ICLR) (2024)
3. Bi, B., Huang, S., Wang, Y., Yang, T., Zhang, Z., Huang, H., Mei, L., Fang, J., Li,
Z.,Wei,F.,Deng,W.,Sun,F.,Zhang,Q.,Liu,S.:Context-DPO:Aligninglanguage
models for context-faithfulness. In: Findings of the Association for Computational
Linguistics:ACL2025.pp.10280–10300.AssociationforComputationalLinguistics
(2025)
4. Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., Carin, L.: CLUB: A contrastive
log-ratio upper bound of mutual information. In: Proceedings of the 37th Interna-
tional Conference on Machine Learning (ICML). Proceedings of Machine Learning
Research, vol. 119, pp. 1779–1788. PMLR (2020)
5. Czyż, P., Grabowski, F., Vogt, J.E., Beerenwinkel, N., Marx, A.: Beyond normal:
On the evaluation of mutual information estimators. In: Advances in Neural Infor-
mation Processing Systems (NeurIPS). vol. 36 (2023)
6. Ethayarajh, K.: How contextual are contextualized word representations? compar-
ing the geometry of BERT, ELMo, and GPT-2 embeddings. In: Proceedings of the
2019ConferenceonEmpiricalMethodsinNaturalLanguageProcessing(EMNLP).
pp. 55–65. Association for Computational Linguistics (2019)
7. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M.,
Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine In-
telligence2, 665–673 (2020)
8. Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., Berant, J.: Did Aristotle
use a laptop? a question answering benchmark with implicit reasoning strategies.
Transactions of the Association for Computational Linguistics9, 346–361 (2021)
9. Ho, X., Nguyen, A.K.D., Sugawara, S., Aizawa, A.: Constructing a multi-hop QA
datasetforcomprehensiveevaluationofreasoningsteps.In:Proceedingsofthe28th
InternationalConferenceonComputationalLinguistics(COLING).pp.6609–6625.
International Committee on Computational Linguistics, Barcelona, Spain (Online)
(2020)
10. Joren, H., Zhang, J., Ferng, C.S., Juan, D.C., Taly, A., Rashtchian, C.: Sufficient
context: A new lens on retrieval augmented generation systems. In: The Thirteenth
International Conference on Learning Representations (ICLR) (2025)
11. Karpukhin,V.,Oğuz,B.,Min,S.,Lewis,P.,Wu,L.,Edunov,S.,Chen,D.,tauYih,
W.: Dense passage retrieval for open-domain question answering. In: Proceedings
of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP). pp. 6769–6781. Association for Computational Linguistics (2020)
12. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., tau Yih, W., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Advances in Neural Information
Processing Systems (NeurIPS). vol. 33, pp. 9459–9474 (2020)

GRIP: Grounded Reasoning via Information-Restricted Premises 15
13. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International
Conference on Learning Representations (ICLR) (2019)
14. Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., Hajishirzi, H.: When
not to trust language models: Investigating effectiveness of parametric and non-
parametric memories. In: Proceedings of the 61st Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long Papers). pp. 9802–9822.
Association for Computational Linguistics, Toronto, Canada (2023)
15. Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N.A., Lewis, M.: Measuring and
narrowing the compositionality gap in language models. In: Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2023. pp. 5687–5711. Association
for Computational Linguistics, Singapore (2023)
16. Rajpurkar, P., Jia, R., Liang, P.: Know what you don’t know: Unanswerable ques-
tions for SQuAD. In: Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers). pp. 784–789. Association
for Computational Linguistics, Melbourne, Australia (2018)
17. Shi, W., Han, X., Lewis, M., Tsvetkov, Y., Zettlemoyer, L., tau Yih, W.: Trusting
your evidence: Hallucinate less with context-aware decoding. In: Proceedings of the
2024 Conference of the North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies (Volume 2: Short Papers). pp.
783–791. Association for Computational Linguistics, Mexico City, Mexico (2024)
18. Song, J., Ermon, S.: Understanding the limitations of variational mutual informa-
tion estimators. In: International Conference on Learning Representations (ICLR)
(2020)
19. Tafjord, O., Mishra, B.D., Clark, P.: ProofWriter: Generating implications, proofs,
and abductive statements over natural language. In: Findings of the Association
for Computational Linguistics: ACL-IJCNLP 2021. pp. 3621–3634. Association for
Computational Linguistics (2021)
20. Tang, L., Laban, P., Durrett, G.: MiniCheck: Efficient fact-checking of LLMs on
grounding documents. In: Proceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing (EMNLP). pp. 8818–8847. Association for
Computational Linguistics (2024)
21. Wang, J., Xu, Z., Jin, D., Yang, X., Li, T.: Accommodate knowledge conflicts in
retrieval-augmented LLMs: Towards robust response generation in the wild. arXiv
preprint arXiv:2504.12982 (2025)
22. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning,
C.D.: HotpotQA: A dataset for diverse, explainable multi-hop question answering.
In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing (EMNLP). pp. 2369–2380. Association for Computational Linguistics
(2018)
23. Yoran, O., Wolfson, T., Ram, O., Berant, J.: Making retrieval-augmented language
models robust to irrelevant context. In: The Twelfth International Conference on
Learning Representations (ICLR) (2024)
24. Zhu, K., Feng, X., Du, X., Gu, Y., Yu, W., Wang, H., Chen, Q., Chu, Z., Chen,
J., Qin, B.: An information bottleneck perspective for effective noise filtering on
retrieval-augmented generation. In: Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1044–
1069. Association for Computational Linguistics (2024)