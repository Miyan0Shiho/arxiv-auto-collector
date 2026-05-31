# What Makes a Medical Checker Trainable? Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wan

**Published**: 2026-05-25 16:06:06

**PDF URL**: [https://arxiv.org/pdf/2605.25988v1](https://arxiv.org/pdf/2605.25988v1)

## Abstract
Medical RAG needs evidence-grounded claims, so plugging a claim-level NLI checker into retrieval-augmented RL is intuitive. \textbf{We find that the checker's \emph{output distribution} during training, not its held-out accuracy, decides whether it provides trainable gradient.} We compare four NLI checker back-ends as process rewards inside a GRPO-trained medical RAG agent (Qwen2.5-7B, replicated on Qwen3-4B and Llama-3.1-8B) across four held-out medical QA benchmarks. Three diagnostic findings emerge. \textbf{(i)} Signal collapse is log-prob-specific: LLM log-probability scoring labels over 97\% of claims neutral -- collapsing the RL gradient to zero -- while a calibrated MedNLI classifier scores the same pairs non-degenerately. \textbf{(ii)} Moderate signal beats strong signal on answer quality: a strong proprietary checker triggers a three-step reward-hacking cascade -- ultra-short answers, search avoidance, language collapse -- so a moderate-signal local classifier trains a higher-quality model (\textbf{+12\% BERTScore over zero-shot, no GPT dependency}). \textbf{(iii)} Signal strength is policy-dependent: the same checker registers as moderate on one policy but strong on another without triggering the cascade end-state. We frame these as boundary conditions for verifier-as-reward systems.

## Full Text


<!-- PDF content starts -->

What Makes a Medical Checker Trainable? Diagnosing Signal Collapse
and Reward Hacking in Checker-Guided RAG for Biomedical QA
Yuelyu Ji Min Gu Kwak Hang Zhang Xizhi Wu Chenyu Li Yanshan Wang
University of Pittsburgh, Pittsburgh, PA, USA
yueluji@gmail.com
Abstract
Medical RAG needs evidence-grounded claims,
so plugging a claim-level NLI checker into
retrieval-augmented RL is intuitive.We find
that the checker’soutput distributiondur-
ing training, not its held-out accuracy, de-
cides whether it provides trainable gradient.
We compare four NLI checker back-ends as
process rewards inside a GRPO-trained med-
ical RAG agent (Qwen2.5-7B, replicated on
Qwen3-4B and Llama-3.1-8B) across four held-
out medical QA benchmarks. Three diagnos-
tic findings emerge.(i)Signal collapse is log-
prob-specific: LLM log-probability scoring la-
bels over 97% of claims neutral—collapsing
the RL gradient to zero—while a calibrated
MedNLI classifier scores the same pairs non-
degenerately.(ii)Moderate signal beats strong
signal on answer quality: a strong proprietary
checker triggers a three-step reward-hacking
cascade—ultra-short answers, search avoid-
ance, language collapse—so a moderate-signal
local classifier trains a higher-quality model
(+12% BERTScore over zero-shot, no GPT
dependency).(iii)Signal strength is policy-
dependent: the same checker registers as mod-
erate on one policy but strong on another with-
out triggering the cascade end-state. We frame
these as boundary conditions for verifier-as-
reward systems.
1 Introduction
Consider a patient asking “Could eating psoriasis
flakes cause prion disease?” A medical RAG sys-
tem should retrieve evidence, decompose its draft
answer into atomic claims, and verify each claim
Code, prompts, and diagnostics available at: https://
anonymous.4open.science/r/medchecker-verl-7FC1/Question (shared across all three regimes)
“Could eating psoriasis flakes cause prion disease?”
✗Collapsed checkerLikelihood-NLI (Meditron-3-8B)
A(394 chars): “Your concerns are valid, but it is highly
unlikely that you have prion disease . . . ”
Checker: 97%NEUTRALSupport0%
→zero RL gradient; reasonable but unverified.
✗Strong checker, hackedGPT-NLI, unguarded
A(85 chars): “It may not lead to prion disease, but avoid
eating psoriasis flakes.”
Checker:ENTAIL(1/1) Support86.1%(higher is not better
here)
→reward hacking, stage 1 of 3: ultra-short answers (§6);
stages 2–3 follow as search avoidance and language col-
lapse.
✓Calibrated checker (ours)MedNLI-Cls
A(full length): “Based on available evidence, no strong
support for prion risk from psoriasis flakes . . . ”
Checker:ENTAIL/NEUTRALmix Support54.0%
→moderate signal, no GPT dependency,highest answer
quality.
Figure 1: The same medical question under three
checker regimes.Top: log-prob scoring collapses to
NEUTRAL, providing no training signal (§5).Middle:
a strong GPT checker provides high support but trig-
gers a three-step reward-hacking cascade (§6)—note
that higher support here isnotbetter.Bottom: a local
MedNLI classifier produces moderate, non-degenerate
verdicts and yields the highest answer quality without
GPT dependency.
against the evidence before answering. The intu-
itive design is to plug a claim-level NLI checker
into the RL reward path, so that the policy learns
to generate evidence-grounded claims through
retrieval-augmented RL (Jin et al., 2025; Xiong
et al., 2024; Zhu et al., 2025). This design is be-
ing explored in concurrent medical RAG-RL sys-
tems (Tamo et al., 2026; Xu et al., 2025; Song
et al., 2026; Chen et al., 2026; Wang et al., 2026,
2025; Liu et al., 2026, 2025b), and builds on aarXiv:2605.25988v1  [cs.CL]  25 May 2026

broader literature where claim-level verification is
used for evaluation and post-hoc correction (Min
et al., 2023; Vladika et al., 2025a,b), with the
headline metric being NLIaccuracyon held-out
pairs. However, no prior work treats the checker’s
output distribution under the policy’s own roll-
outs—as opposed to its accuracy on a held-out
NLI benchmark—as a measurable training-time
variable, and we will show that this distinction is
where the design fails.
In medical RAG, plugging an off-the-shelf NLI
checker into the reward path fails in two ways that
accuracy does not predict. A biomedical LLM scor-
ing entailment by log-probability (Sallinen et al.,
2025) can label >97% of claims NEUTRALre-
gardless of evidence (collapsing the per-claim sup-
port rate, the fraction labelled ENTAIL, to zero
gradient), even though the same model is accurate
on balanced NLI tests. A strong proprietary ver-
ifier (Achiam et al., 2023) that doesnotcollapse
instead opens a reward-hacking pathway (Skalse
et al., 2022): ultra-short answers, then search avoid-
ance, then generating non-English answers to evade
the English-only NLI prompt; each shortcut closes
under a countermeasure but exposes the next.
Both failure modes are invisible to held-out NLI
accuracy.Our central thesis is that the checker’s
output distribution under the policy’s rollouts,
not its held-out accuracy, decides whether it
provides trainable gradient.We therefore fo-
cus on the per-claim support rate—the fraction of
generated claims the checker labels ENTAIL—as
the primary diagnostic variable throughout the pa-
per.This paper is a diagnosis of the checker-as-
reward design space, not a competing system;
we make no SOTA claim.We build a multi-turn
medical RAG agent on Qwen2.5-7B-Instruct (Hui
et al., 2024) trained with GRPO (Shao et al., 2024),
extending Search-R1 (Jin et al., 2025) with a dense
retriever over the MedRAG corpus (Xiong et al.,
2024) and a claim-level NLI verification server as
process reward. To isolate how the checker’s output
distribution shapes training, we instantiate four NLI
back-ends as drop-in replacements inside the same
reward pathway:Likelihood-NLI(Meditron-3-8B
log-prob),MedNLI-Cls(PubMedBERT classifier),
Hybrid(GPT atomic extraction +local log-prob
scoring), andGPT-NLI(GPT-4o-mini end-to-end).
The four span the (extractor,scorer) design space
along a GPT-dependency gradient (Figure 1, Ta-
ble 2); the per-claim mechanics are deferred to §3.
We describe two operating regimes of per-claimsupport rate during training:moderate( ≲60%, e.g.
MedNLI-Cls at 54%) andstrong( ≳75%, e.g. Hy-
brid at 76%, GPT-NLI at 86%).1We report three
diagnostic findings.First, signal collapse is spe-
cific to log-probability scoring: MedNLI-Cls pro-
duces non-degenerate verdicts on thesamepairs
Likelihood-NLI labels almost entirely NEUTRAL
(§5).Second, the relation between signal strength
and answer quality is non-monotonic: a moderate-
signal classifier outperforms a strong-signal pro-
prietary checker (BERTScore 0.600 vs. 0.591) be-
cause the latter triggers a three-step cascade of
shortcuts (ultra-short answers, search avoidance,
language collapse) that each undermine clinical
groundedness (§6).Third, signal strength is a prop-
erty of the policy–checker pair, not of the checker
alone; we support this with one cross-model swap
(Qwen2.5-7B →Qwen3-4B) and treat it as a work-
ing hypothesis to be tested at larger scale (§ ??,
App. 25). We frame these findings as boundary
conditions for concurrent verifier-as-reward sys-
tems.
2 Related Work
We organise prior and concurrent work along four
axes: process rewards in retrieval-augmented RL,
claim-level verification for biomedical RAG, NLI
as a reward model, and failure modes in LLM
RL. Table 1 summarises five concurrent verifier-as-
reward systems on seven design dimensions and
locates our contribution in that design space.
Process rewards in retrieval-augmented RL.
Search-R1 (Jin et al., 2025) trains LLMs to in-
terleave reasoning and search through RL with
outcome-based rewards on general-domain QA,
using F1 alone as the reward signal. Concurrent
verifier-as-reward systems extend this paradigm to
medicine: EvidenceRL (Tamo et al., 2026) pairs
PubMedBERT-MNLI / MedNLI with GRPO and
reports a grounding-style reward; VERITAS (Xu
et al., 2025) integrates faithfulness rewards into
Search-R1 training and notes that NLI judgements
in RAG settings can be unreliable; Med-R3 (Lu
et al., 2025) uses progressive training with retrieval-
effectiveness rewards; Fleming-R1 (Liu et al.,
2025a) combines curriculum learning with RL for
medical reasoning; and MedResearcher (Yu et al.,
2025) trains agents for multi-hop medical evidence
gathering. The empty cells underCollapseand
1These are descriptive labels for observed regimes, not
strict cutoffs; what matters for the cascade (§6) is behaviour,
not a fixed threshold.

System Med.curriculum.Verifier-as-reward Claim Collapse Cascade Triage
Search-R1 (Jin et al., 2025) –✓F1 only – – partial –
EvidenceRL (Tamo et al., 2026)✓ ✓PubMedBERT-MNLI✓– – –
VERITAS (Xu et al., 2025) –†✓faithfulness✓partial‡– –
MedRAGChecker (Ji et al., 2026)✓–∗eval-time✓– – –
Med-R3 (Lu et al., 2025)✓ ✓retrieval-eff. – – – curr.
Ours✓ ✓4 back-ends✓ ✓ ✓ ✓
Table 1: Concurrent verifier-as-reward systems.Med.: medical domain.M-turn: multi-turn agent.Verifier-as-
reward: RL reward source.Claim: scores atomic claims.Collapse/Cascade: isolates as primary phenomena.Triage:
question-level budget control.†RAG-broad with medical sub-domains;‡reports unreliable NLI without localising to
log-prob;∗evaluation framework, not RL.
Cascadein Table 1 are the gap our work fills: none
of these systems treats checker output distribution
as a measurable training-time variable, so the fail-
ure modes that arise specifically from reward-side
bias have so far gone undocumented.
Claim-level verification for biomedical RAG.
FActScore (Min et al., 2023) introduces atomic
factuality evaluation by decomposing generations
into verifiable units; RAGChecker (Ru et al., 2024)
provides fine-grained diagnostic metrics including
faithfulness, hallucination rate, and noise sensi-
tivity; Self-RAG (Asai et al., 2023) trains reflec-
tion tokens for adaptive retrieval and self-critique
at generation time. Medical extensions include
MedRAGChecker (Ji et al., 2026), which extends
atomic factuality to biomedical RAG with claim-
level verification; the atomic-fact decomposition
for medical answers of Vladika et al. (2025a);
and the iterative step-by-step verification system
of Vladika et al. (2025b). The MIRAGE bench-
mark and MedRAG toolkit (Xiong et al., 2024)
and Med-PaLM 2 (Singhal et al., 2025) estab-
lish the medical-RAG evaluation backdrop. These
works use verification for evaluation or generation-
time guidance; we study a substantially different
setting—using verification as an RL reward signal
during training—where systematic checker biases
directly corrupt the training objective rather than
merely producing noisy evaluation.
NLI as a reward model.NLI is the backbone of
most claim-verification systems. In the medical do-
main, MedNLI (Romanov and Shivade, 2018) pro-
vides a clinical NLI benchmark, and biomedical en-
coders such as PubMedBERT (Gu et al., 2021) and
SciFive (Phan et al., 2021) achieve strong perfor-
mance on biomedical NLI; FActScore (Min et al.,
2023) popularised NLI-style atomic verification at
evaluation time. We provide, to our knowledge, the
first systematic comparison of how different NLIarchitectures behave as process rewards inside an
RL loop.
Failure modes in LLM RL.Reward hacking
is well-documented in general RL (Skalse et al.,
2022). In LLM RL, concurrent work on Search-
R1 variants reports that F1-based rewards cause
training collapse through answer avoidance, and
VERITAS (Xu et al., 2025) reports unreliable NLI
judgements in RAG settings. We extend this litera-
ture in two ways. We localise NLI failure to LLM
log-probability scoring specifically and show that
calibrated MedNLI-Cls classifiers do not collapse
on the same claim–evidence pairs (§5); and we
document a three-step reward-hacking cascade—
ultra-short answers, search avoidance, language
collapse—that emerges only understrongverifi-
cation signal (§6). Together with Table 1, this
positions our contribution as a diagnosis of when
verifier-as-reward designs train, not a competing
system.
3 Method
3.1 System Overview
Figure 2 gives an overview of our framework. We
build a multi-turn medical RAG agent on Qwen2.5-
7B-Instruct (Hui et al., 2024) trained with Group
Relative Policy Optimization (GRPO; Shao et al.,
2024), extending the Search-R1 framework (Jin
et al., 2025).
Terminology.Throughout,checkerrefers to the
NLI-based verification component, whether dis-
cussed conceptually (the reward source) or as the
concrete <check> tool the agent invokes. The four
checker back-endsof §3.2 are alternative imple-
mentations of this same component. The agent
has two tools:Search, a dense retriever over the
MedRAG corpus (Xiong et al., 2024) (PubMed,
StatPearls, textbooks); andCheck, an NLI-based

1Input & Triage2Multi-turn Agent3NLI Verification Back-Ends (Four Checkers)4Reward & Policy Update
MedicalQuestion  Triage Controller1) Difficulty AssignmentTraining data: ensemble agreement labelsEasyMediumHardHeld-out data: heuristic scoreQuestion lengthClinical keywordsBullet structureDifficulty tier(Easy / Medium / Hard)2) Budget Allocation (per tier)TierSearchCheckTurnsEasy113Medium225Hard4373) Online Escalation⚡
Trigger if contradiction > 0.30, support < 0.40, or failed search → escalate to next higher tier Checker-GuidedMedical RAG AgentAuto-check after eachsearch–answer cycle
Draft answer /atomic claims fromretrieved evidence  MedRAG corpusPubMed abstractsStatPearlsmedical textbooks  NLI Verification Back-Ends(Four Checkers)Draft answer + retrieved evidence→ atomic claims / claim-evidence pairsA. Likelihood-NLIMeditron-3-8B | sentence split + log-probcollapsesB. MedNLI-ClsPubMedBERT fine-tuned on MedNLI | sentence split + classifiercalibratedC. HybridGPT atomic extraction + local log-prob scoringpartial GPTD. GPT-NLIGPT atomic extraction + GPT verificationfull GPT Training RewardAnswer quality (F1)+NLI verification (r_check)+Format control (r_format)  GRPOpolicy update✅
  Grounded final answer
    Search
   Check
Support / Neutral / Contradict verdicts → r_check
Figure 2: Overview of our framework.(1) Input & Triage: a triage controller assigns each question to
easy/medium/hard with per-tier search, check, and turn budgets (held-out data scored on five surface features—
question length, multi-hop indicators, clinical keywords, multi-question, bullet structure; full formula in App. 21),
and escalates online when the checker signals high contradiction (§ ??).(2) Multi-turn Agent: the policy interleaves
<search> and<check> over the MedRAG corpus, with auto-check after each search–answer cycle.(3) Checker
Back-Ends: four NLI checkers compared as process rewards (§3.2).(4) Reward & Policy Update: rbase,ϕcheck,
andP fmtcombine into a GRPO update toward grounded answers (§3.3).
verification server that decomposes a draft into
atomic claims and verifies each against retrieved
evidence. At each turn, the agent either emits a tool
call ( <search> or<check> ) or outputs a final an-
swer in <answer> tags. Anauto-check loopinjects
a verification call after every search-and-answer
cycle when the check budget is not exhausted, en-
suring checker signal availability during early train-
ing before the model learns to invoke check au-
tonomously.
Retriever.A frozen MedCPT (Jin et al., 2023) bi-
encoder serves top-5 passages from the MedRAG
corpus (Xiong et al., 2024) (truncated to 768 to-
kens; see §5.2); held constant across runs. Full
details in Appendix 13.
3.2 Checker Back-Ends
The four back-ends share the same NLI task
(labelling each (claim, evidence) pair as EN-
TAIL/NEUTRAL/CONTRADICT) but differ in mech-
anism (Table 2).
Likelihood-NLIuses Meditron-3-8B (Sallinen
et al., 2025) with raw log-probability scoring and
sentence-split extraction; we call this the “stu-
dent checker” when contrasted with GPT-NLI.
Meditron-3-8B is the strongest open-weight medi-
cal LLM at this scale (81.2% NLI accuracy when
distilled, Ji et al., 2026), so collapse under this con-Back-end Extractor Scorer Collapse?
Likelihood-NLI sent. split log-probYes
MedNLI-Cls sent. split classifier No
Hybrid GPT atomic log-prob No⋄
GPT-4o-mini GPT atomic prompted No
Table 2: Back-ends decomposed into claim extractor
and NLI scorer.⋄Hybrid avoids full collapse (75.8%)
because GPT atomic extraction mitigates the structural
log-prob bias, but a residual gap to MedNLI-Cls re-
mains (§5.2); log-prob is a necessary but not sufficient
condition for collapse.
figuration (§5) cannot be attributed to weak checker
capacity, isolating the failure to the scoring mecha-
nism.
MedNLI-Clsfine-tunes PubMedBERT-base
(Gu et al., 2021) on MedNLI (Romanov and Shiv-
ade, 2018) with a 3-way classification head (84.1%
test accuracy; training, serving, and reproducibility
details in Appendix 15).
Hybridpairs GPT-4o-mini atomic-claim extrac-
tion with the Likelihood-NLI scorer, confining GPT
dependency to extraction.
GPT-NLIuses GPT-4o-mini for both extraction
and verification in a single call, with a prompt that
penalises neutral verdicts.

3.3 Reward Function
We use aprocess reward: a per-step signal com-
puted on intermediate generations (claims and their
evidence), in contrast to outcome-only rewards
that score only the final answer. For a rollout
τ= (q,D,ˆa) with retrieved evidence Dand fi-
nal answer ˆa, the reward couples a correctness
base rbase(τ)∈[0,1] with a faithfulness multi-
plier ϕcheck(ˆa,D)∈[−1,+1] (Eq. ??) and a for-
mat penalty Pfmt(ˆa)∈{−0.5,−0.3,0} asR(τ) =
rbase(τ)· 
1+α ϕ check(ˆa,D)
+P fmt(ˆa)withα= 1
throughout, giving range R∈[−1.5,+2] : the mul-
tiplier doubles a correct answer’s reward or zeros
it on full contradiction, but cannot turn a wrong
answer positive.
Base reward rbase.A normalised weighted
combination of EM, F1, and FMTSCOREwith
weights (0.35,0.15,0.10) re-normalised to sum to
1.2EM and F1 follow SQuAD-style normalisation;
FMTSCOREis a piecewise length reward saturating
atL tgt=80chars (full formulas in Appendix 10).
Faithfulness multiplier ϕcheck.The checker
decomposes ˆainto atomic claims {c1, . . . , c K},
assigns each a verdict vkwith confidence
pk∈[0,1] , and aggregates to ϕcheck =
1
KP
kskpkwhere sk∈ {+1,0,−1.5} for EN-
TAIL/NEUTRAL/CONTRADICTrespectively (Ap-
pendix 10, Eq. ??). Neutral verdicts contribute
zero, so a checker that labels almost every claim
neutral (§5) collapses the multiplier to 1. The
asymmetric contradiction weight reflects that an
evidence-contradicted claim is strictly worse than
an unverified one in medical QA.
Neutral verdicts contribute zero, so a checker
that labels almost every claim neutral (§5) col-
lapses the multiplier to 1and removes its gradi-
ent contribution. The asymmetric contradiction
weight reflects that, in medical QA, an evidence-
contradicted claim is strictly worse than an unveri-
fied one.
Format penalty Pfmt.A separate penalty term
targeting reward-hacking shortcuts (§6): Pfmt=
−0.5 when the <answer> tag is missing, −0.3
when the answer is shorter than 50characters, and
0otherwise. FMTSCORE(inside rbase) rewards for-
mat quality on the positive side; Pfmtpunishes de-
generate shortcuts on the negative side. The two are
kept distinct so that the negative-side penalty can-
not be diluted by the multiplier.GRPO advantage
2Multiplicative coupling prevents high faithfulness from
substituting for correctness; see Appendix 17.and format penalty.We sample Nrollouts per
question and compute standard group-normalised
advantages (Shao et al., 2024) (Appendix 14). A
separate format penalty Pfmt∈ {−0.5,−0.3,0}
targets missing tags and ultra-short answers (§6),
kept distinct from FMTSCOREso the negative-side
penalty cannot be diluted by the multiplier.
4 Experiments
4.1 Setup
Model and training.All experiments fine-tune
Qwen2.5-7B-Instruct with GRPO on 2 ×H100
GPUs ( ∼2 h per run); per-run training-step counts
and full hyperparameters (batch size, learning rate,
KL and entropy coefficients) are in Appendix 23.
Datasets.Training combines CSIRO
MedRedQA (Nguyen et al., 2023), LiveQA-
Med (Liu et al., 2020), and PubMedQA (Jin
et al., 2019) into 1,513 train / 87 in-domain
test (“Medical”). Held-out evaluation adds four
out-of-distribution sets: MedicationQA (674, full
standard test set, drug safety) (Liu et al., 2020),
BioASQ-Y/N (618, MIRAGE 2019–2023 split,
biomedical literature) (Tsatsaronis et al., 2015),
MedQuAD (100, NIH consumer health subsam-
ple) (Ben Abacha and Demner-Fushman, 2019),
and MEDIQA (85, clinical summarisation) (Xun
et al., 2025); total evaluation coverage is 1,564
samples. The closest non-RL comparison points
on these datasets are MedBioLM (Kim, 2025a)
and MedBioRAG (Kim, 2025b), both based on
fine-tuned GPT-4o; we compare against them on
MedicationQA (Table 4).
Metrics.Foranswer qualitywe report
BERTScore (Zhang et al., 2019) (primary, seman-
tic similarity to physician-authored references),
token F1 (Rajpurkar et al., 2016), and ROUGE-
L (Lin, 2004). Model outputs run 200–400
characters and references exceed 500, so F1
is systematically low across all configurations
(zero-shot baseline F1 = 0.191), which differs
from accuracy on closed-form benchmarks such
as MedQA or MedBrowseComp (Lu et al., 2025;
Chen et al., 2025). Forevidence grounding,
we report two claim-level metrics following
FActScore (Min et al., 2023) and RAGChecker (Ru
et al., 2024). We defineSupport Rateas the
fraction of extracted claims labelled ENTAIL
by the verifier, andFaithfulnessas the fraction
of samples with no CONTRADICTverdict. For

behavioural signalswe trackTag%(fraction
of outputs with a valid <answer>...</answer>
wrapper) andSearch(mean <search> calls per
question, surfacing the search-avoidance shortcut
in §6).
Cross-model generalisation to Llama-3.1-8B and
cross-domain transfer to HotpotQA are reported in
Appendix 12.
4.2 Main Results
The central claim of §1 is that the verifier’soutput
distribution, not its accuracy, determines whether it
provides trainable gradient. We test this by holding
the policy, retriever, and reward weights fixed and
varying only the NLI back-end. Evaluation runs
on Qwen2.5-7B across four held-out medical QA
benchmarks (1,479 samples; MEDIQA reported
separately in Appendix 9). Table 3 summarises
performance averaged across the four benchmarks;
Figure 3 shows per-dataset detail.
Table 4 positions our setting against closed-
frontier baselines on the one dataset where direct
comparison is possible. Qwen3-4B + MedNLI-Cls
reaches F1 = 0.191, within 0.029 of MedBioLM
(0.220, GPT-4o +SFT +RAG) andabovethe
GPT-4o fine-tuning baseline (0.158) by 0.033, de-
spite using a ∼10×smaller backbone with no su-
pervised data. The 0.029 gap to MedBioLM does
not close the open-vs-closed divide, but is unusu-
ally small given ∼10×less backbone capacity and
no supervised fine-tuning.
MedNLI-Cls leads BERTScore (0.600) without
GPT, while GPT-NLI leads support rate (86.1% vs.
54.0%) at the cost of quality. Likelihood-NLI is
competitive on quality (BERT 0.599) but yields
zero RL gradient under collapse (§5): its 80.1%
support rate is an external GPT-evaluator score
on trained outputs, not the training gradient. Per-
dataset (Figure 3), MedNLI-Cls leads BERTScore
on 3 of 4 sets, losing only on the in-domain Medical
split (.575 vs. Likelihood-NLI’s .597); this is con-
sistent with the collapse pathway providing format
regularisation without GPT co-adaptation (§5.1),
which helps most on the training-distribution split.
GPT-NLI leads support rate on 3 of 4, losing only
on MedicationQA, where Hybrid’s GPT atomic-
claim extraction recovers slightly finer-grained ver-
dicts (85.2% vs. 84.4%). The trade-off direction is
consistent across datasets; the four-dataset average
is not driven by any single benchmark.
The fully-engineered GPT-NLI configuration
(triage + format penalty + English constraint) recov-
Medical MedQA BioASQ MedQuAD0255075100Support Rate (\%)(a) Support Rate ↑
Medical MedQA BioASQ MedQuAD0.500.550.600.650.70BERTScore(b) BERTScore ↑Likelihood-NLI MedNLI-Cls Hybrid GPT-NLIFigure 3: Per-dataset breakdown across four medical
QA benchmarks.(a)Support Rate.(b)BERTScore
(zero-shot baseline as dashed line).
ers F1 = 0.203 with 100% tag stability and 100%
support rate, but only matches MedNLI-Cls on
BERTScore (0.601 vs. 0.600) at the cost of GPT
API access and 4 cumulative reward-engineering
interventions. The cascade analysis (§6) explains
why each countermeasure resolves one shortcut
but exposes the next. The Qwen3-4B rows in Ta-
ble 3 preview an unexpected finding: the MedNLI-
Cls checker that registers as amoderatesignal on
Qwen2.5-7B (54.0% support) becomes astrong
signal on Qwen3-4B (85.3%) on the same claim–
evidence pairs, yet does not trigger the cascade
end-state. We unpack this in §4.2 and Appendix 12.
Cross-Model ValidationTo assess generaliza-
tion beyond Qwen2.5-7B, we replicate selected
configurations on Llama-3.1-8B and Qwen3-
4B (Appendix 12). Signal collapse with the
Likelihood-NLI checker reproduces on both:
Llama achieves 0% support and Qwen3-4B (with
the Likelihood-NLI checker) shows the same
characteristic gap between policy quality (BERT
∼0.59) and verifier-recognized grounding ( <60%
support). Format stability under the GPT-4o-mini
checker depends on base model: Qwen2.5-7B
sustains the multi-turn format (Tag = 98%) while
Llama-3.1-8B drops to 50% on average (and 14.9%
on the medical split). We discuss base-model de-
pendence further in §6 and Appendix 12. These

Base Checker BERT Supp% Faith Tag% GPT?
Baseline
Qwen2.5-7B Zero-shot .538 n/a n/a 100 n/a
Qwen2.5-7B (primary)
Qwen2.5-7B Likelihood-NLI .599 0 (80.1†) .964 95.7 No
Qwen2.5-7B MedNLI-Cls.60054.0 .972100No
Qwen2.5-7B Hybrid .565 75.8.98795.9 Partial
Qwen2.5-7B GPT-NLI .59186.1.972 98.3 Full
Qwen3-4B (cross-model)
Qwen3-4B Likelihood-NLI .594 51.0†.952100No
Qwen3-4B MedNLI-Cls .591 85.3 .980100No
Qwen3-4B Hybrid .542 53.2 .833 92.1 Partial
Qwen3-4B GPT-NLI .582 87.2 .962 98.4 Full
Table 3: Main results on four held-out medical QA benchmarks (1,479 samples; MEDIQA in Appendix 9).BERT:
BERTScore.Supp%: % of claims labelled ENTAIL.Faith: % of samples with no CONTRADICT.Tag%: % of
outputs with valid <answer> tags.GPT?: GPT API dependency. Token F1 and ROUGE-L in Appendix 11; Llama-
3.1-8B in Appendix 12.†Likelihood-NLI self-evaluation gives 0% under collapse; values shown are GPT-NLI
evaluation (§5).
System Setup F1 R-L
Ours: open-weight, RL-only, no SFT
Qwen2.5-7B Likelihood-NLI 7B + RL .150 .137
Qwen2.5-7B MedNLI-Cls 7B + RL .170 .144
Qwen2.5-7B Hybrid 7B + RL .140 .120
Qwen2.5-7B GPT-NLI 7B + RL .149 .127
Qwen3-4B Likelihood-NLI 4B + RL .185 .141
Qwen3-4B MedNLI-Cls 4B + RL.191 .153
External: closed frontier+SFT
GPT-4o (FT) (Kim, 2025a) FT .158 .134
MedBioLM (Kim, 2025a) FT+RAG .220 .187
Table 4: MedicationQA head-to-head on the full n=674
test set. External systems use closed GPT-4o with su-
pervised fine-tuning; our open-weight RL-only systems
reach competitive F1 with a ∼10×-smaller backbone
and no SFT.
two patterns directly support findings (i) and (iii)
from §1: signal collapse is mechanism-specific
(Llama reproduces it on the log-prob pathway),
and signal strength is policy-dependent (the same
MedNLI-Cls checker registers as moderate on
Qwen2.5-7B but strong on Qwen3-4B).
5 Checker Signal Collapse
We definechecker signal collapseas the condition
where an NLI verifier, used as a process reward,
assigns a single label to the overwhelming majority
of claims regardless of evidence—rendering the
reward effectively constant. With Meditron-3-8B
log-prob scoring, the checker labels 97–99% of
claims neutral, mapping ϕcheck to 0 across rollouts.5.1 Calibrated Classifiers Do Not Collapse
A natural objection is that collapse reflects a poor
checker choice. We test this with PubMedBERT
fine-tuned on MedNLI (a calibrated classification
head3) on thesameclaim–evidence pairs. MedNLI-
Cls produces non-degenerate verdicts (54.0% sup-
port rate averaged across four held-out datasets, vs.
0% for Likelihood-NLI; see also Table 3), local-
ising the failure to thescoring mechanism, not to
local NLI in general.
Verification signal vs. format regularisation.
Collapsed-checker policies still outperform search-
only baselines (BERTScore 0.595 vs. 0.537). The
mechanism isformat regularisation: the auto-
check loop forces parseable output (Tag rate 98.9%
vs. 58.6% search-only), independent of verdict sig-
nal. External GPT-NLI evaluation of these outputs
gives 80.1% support rate, confirming the policy
producesgrounded answers; collapse is in the veri-
fier’s ability torecognisegrounding. This decom-
position rules out reward–evaluator co-adaptation
(the policy learning to please the GPT evaluator
rather than to ground its claims) as the mechanism
behind any reported BERTScore gain, including
the moderate-signal advantage in §??.
5.2 Three Root Causes of Log-Prob Collapse
A controlled diagnostic chain on the log-prob path-
way isolates three independent causes (Table 5).
Evidence truncationat the default 256-token tool-
3Throughout,calibratedmeans the checker produces a
non-degenerate softmax distribution over {ENTAIL, NEU-
TRAL, CONTRADICT}, in contrast to the log-prob pathway
which collapses to NEUTRAL.

Diagnostic step Support Rate
Likelihood-NLI (baseline) 0%
+ Evidence fix (256→768 tok) 35.7%
+ GPT claim extraction 37.5%
+ GPT verification 65.6–77.5%
MedNLI-Cls (no fixes needed) 54.0%
Table 5: Diagnostic chain: each step isolates one cause
of log-prob collapse. A calibrated classifier (MedNLI-
Cls) avoids collapse without any of these fixes.
Checker Triage BERT Search Tag% Supp%
MedNLI-Cls
MedNLI-Cls off .599 1.02 95.7 80.1
MedNLI-Cls on .6050.3098.2 79.4
GPT-NLI
GPT-NLI off .591 0.94 98.3 86.1
GPT-NLI on .6010.31100 88.2
Likelihood-NLI (collapse)
Likelihood-NLI off .599 0.81 95.7 0.0‡
Likelihood-NLI on .6050.3098.4 0.0‡
Table 6: Triage controller ablation on Qwen2.5-7B.
Triage cuts search-call rate ∼3×without quality loss;
Likelihood-NLI rows show triage cannot rescue col-
lapse. F1 and matched-seed details in Appendix 21.
‡Self-evaluation; GPT-NLI evaluation gives 80.1% (§5).
response limit strips retrieved passages to 1–2
sentences; extending to 768 tokens raises sup-
port rate from 0% to 35.7%.Non-atomic claim
extractionfrom sentence splitting yields multi-
proposition claims; replacing with GPT-4o-mini
extraction raises support rate to 37.5%.Resid-
ual log-prob biaspersists even with full evidence
and atomic claims, but is largely closed by swap-
ping the scorer (Hybrid: 75.8%; fully GPT-verified:
65.6–77.5%). MedNLI-Cls does not exhibit any
of these—collapse is specific to the log-prob path-
way. Per-cause analysis is in Appendix 19. The
model trained with MedNLI-Cls achieves thehigh-
estBERTScore (0.600) and F1 (0.215, averaged
across four held-out datasets) of any configuration
despite its lower support rate (Table 3); we return
to this in §??.
6 Reward Hacking Cascade
Replacing Likelihood-NLI with GPT-NLI resolves
signal collapse but does not improve answers; in-
stead, the policy discovers a sequence of shortcuts,
each countermeasure exposing the next.Stage 1:
Ultra-short answers.Average answer length col-
lapses from 394 to 130 characters—fewer claimsmeans fewer contradictions. A graded format
penalty ( Pfmt=−0.5/−0.3/0 for missing tag /
<50 chars / otherwise) restores length to 216 char-
acters.Stage 2: Search avoidance.98% of sam-
ples emit zero search calls, the policy answering
from the base model’s internal weights instead of
retrieved evidence. A +0.1 search bonus raises
search rate to 96.6% but partially reverts length to
141 characters.Stage 3: Language collapse.Under
triage budgets, the policy switches to Chinese to ex-
ploit Qwen2.5’s multilingual pretraining; “Always
respond in English” in the prompt removes it (Ap-
pendix 18). The full sequence demonstrates that
strong verification signal under single-pass train-
ing is fundamentally unstable; only the moderate
MedNLI-Cls signal (§5.1) avoids this pattern.
RL regularisation cannot rescue this.The α
sweep (App. 26) shows weakening the checker
reward to α= 0.5 still leaves the policy in the
strong-signal regime (Support 74.3%) and below
MedNLI-Cls quality. Clip and KL penalties con-
strain step magnitude, not gradient direction; the
cascade isstructural, not a tuning artifact.
7 Discussion
Our results answer two reviewer-facing concerns.
First, the main finding is not circular GPT super-
vision: the best configuration, MedNLI-Cls, uses
no GPT in the reward path, and answer quality is
measured against physician-authored references.
Second, reference metrics alone are insufficient for
medical RAG: collapsed Likelihood-NLI reaches
high BERTScore but 0% training support, showing
that answer similarity and evidence-grounded train-
ability diverge. The practical implication is to treat
checker output distribution as a first-class training
diagnostic.
8 Conclusion
We compared four NLI checker back-ends as RL
process rewards in medical RAG. Three findings
refine prior intuitions about verified RL: signal col-
lapse is a property of LLM log-probability scor-
ing rather than of NLI in general; moderate ver-
ification signal outperforms strong signal on an-
swer quality due to a reward-hacking cascade; and
signal strength is policy-dependent. Practitioners
should prefer calibrated classifiers over LLM log-
prob scoring for the reward path, and design reward
components holistically rather than incrementally.

Limitations
Evaluation independence.Support Rate is com-
puted by the same GPT-4o-mini checker used as a
training reward component. This parallels standard
practice in RL for text generation, where F1 serves
as both reward and evaluation metric (Jin et al.,
2025). The checker evaluates claim–evidence en-
tailment, an objective NLI judgement independent
of training dynamics. The wide variation across
configurations (0% to 100%) confirms that the met-
ric is not trivially optimised. Collapsed-checker
configurations receive the same F1 reward but pro-
duce zero support tate, showing that the two signals
measure distinct properties.
Single-seed reporting.All reported numbers are
from a single training seed per configuration. The
sub-percent BERTScore gap between MedNLI-Cls
(0.600) and GPT-NLI (0.591) is within plausible
evaluation noise on n=1,479 samples; we treat
the qualitative ordering (MedNLI-Cls ≥GPT-NLI
on quality, GPT-NLI ≥MedNLI-Cls on support
tate) as the robust finding rather than the exact
magnitude. The support tate gap (54.0% vs. 86.1%)
and the collapse contrast (97% Neutral vs. 54%
Entail on the same pairs) are large enough that seed
variance is unlikely to flip their sign.
Test set size.Our in-domain Medical test set
contains 87 samples, limiting statistical power on
that split alone; held-out coverage (1,479 samples
across four out-of-distribution benchmarks) miti-
gates this for generalisation claims.
Limited cross-model coverage.Our primary re-
sults use Qwen2.5-7B-Instruct with all four checker
back-ends. We replicate selected configurations on
Qwen3-4B and Llama-3.1-8B (Appendix 12); full
back-end replication on these models is left for
future work due to compute constraints. The identi-
fied failure modes (evidence truncation, non-atomic
claim extraction, log-prob bias, reward hacking) are
likely model-independent, but language collapse is
specific to Qwen’s multilingual pretraining.
Trade-off between quality and grounding.The
MedNLI-Cls checker provides the highest answer
quality without GPT dependency, but trails GPT-
4o-mini on support tate (54.0% vs. 86.1%). Prac-
titioners requiring strict evidence grounding may
need GPT API access; those prioritising answer
quality or reproducibility can use the fully local
configuration.Medical domain scope.We conduct all RL train-
ing and primary evaluation on medical QA datasets.
While the checker itself is domain-agnostic, the
RL training dynamics (shortcuts, collapse patterns)
may differ in other domains.
Metric limitations.Token F1 penalises length
mismatches between model outputs (200–400
chars) and reference answers (500+ chars).
BERTScore is more robust to length variation but
introduces its own biases. Human evaluation would
strengthen the findings.
Incomplete cascade resolution.No single con-
figuration simultaneously maximises all desirable
properties (high F1, high support tate, high search
utilisation, stable format). The cascade may con-
tain additional undiscovered shortcuts.
Ethics Statement
This work uses publicly available medical QA
datasets (CSIRO MedRedQA, LiveQA-Med, Pub-
MedQA for training; MedicationQA, BioASQ,
MedQuAD, MEDIQA for held-out evaluation) that
do not contain identifiable patient information. Our
system is a research prototype not intended for
clinical deployment. Generated medical answers
have not been validated by clinical experts and
should not be used for health decisions. API calls
to GPT-4o-mini are subject to OpenAI’s data usage
policies.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08032.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Asma Ben Abacha and Dina Demner-Fushman. 2019. A
question-entailment approach to question answering.
BMC Bioinform., 20(1):511:1–511:23.
Shan Chen, Pedro Moreira, Yuxin Xiao, Sam
Schmidgall, Jeremy Warner, Hugo Aerts, Thomas
Hartvigsen, Jack Gallifant, and Danielle S Bitter-
man. 2025. Medbrowsecomp: Benchmarking medi-
cal deep research and computer use.arXiv preprint
arXiv:2505.14963.

Yisong Chen, Chuqing Zhao, Yishan Zhong, Ziyu Wang,
Jiazhao Shi, and Wenjia Zheng. 2026. Applying the
deep learning–sector–governance (dlsg) framework
to the u.s. healthcare system: Opportunities, deploy-
ment pathways, and policy -aligned evaluation.Jour-
nal of Technology Innovation and Society, 4:1–18.
Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto
Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng
Gao, and Hoifung Poon. 2021. Domain-specific lan-
guage model pretraining for biomedical natural lan-
guage processing.ACM Transactions on Computing
for Healthcare, 3(1):1–23.
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang,
Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun
Zhang, Bowen Yu, Keming Lu, and 1 others. 2024.
Qwen2. 5-coder technical report.arXiv preprint
arXiv:2409.12186.
Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi
Wu, Chenyu Li, and Yanshan Wang. 2026.
Medragchecker: Claim-level verification for biomed-
ical retrieval-augmented generation.arXiv preprint
arXiv:2601.06519.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval.Bioinformatics,
39(11):btad651.
Seonok Kim. 2025a. Medbiolm: Optimizing medi-
cal and biological qa with fine-tuned large language
models and retrieval-augmented generation.arXiv
preprint arXiv:2502.03004. Reports ROUGE-1/2/L,
BLEU, BERTScore, BLEURT on LiveQA-Med and
MedicationQA for fine-tuned GPT-4o; verify final
venue before camera-ready.
Seonok Kim. 2025b. Medbiorag: Semantic
search and retrieval-augmented generation with
large language models for medical and biological
qa.arXiv preprint arXiv:2512.10996. Reports
ROUGE/BLEU/BERTScore/BLEURT on LiveQA-
Med, MedicationQA, PubMedQA, BioASQ; verify
final venue before camera-ready.Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81.
Chi Liu, Derek Li, Yan Shu, Robin Chen, Derek Duan,
Teng Fang, and Bryan Dai. 2025a. Fleming-r1: To-
ward expert-level medical reasoning via reinforce-
ment learning.arXiv preprint arXiv:2509.15279.
Qianying Liu, Sicong Jiang, Yizhong Wang, and Su-
jian Li. 2020. Liveqa: A question answering dataset
over sports live. InProceedings of the 19th Chinese
National Conference on Computational Linguistics,
pages 1057–1067.
Weisi Liu, Guangzeng Han, and Xiaolei Huang. 2025b.
Examining and adapting time for multilingual clas-
sification via mixture of temporal experts. InPro-
ceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers), pages 6151–6166,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Weisi Liu, Guangzeng Han, and Xiaolei Huang.
2026. Knowledge-driven augmentation and re-
trieval for integrative temporal adaptation.Preprint,
arXiv:2604.22098.
Keer Lu, Zheng Liang, Youquan Li, Jiejun Tan, Xili
Wang, Da Pan, Shusen Zhang, Guosheng Dong, Bin
Cui, Yunhuai Liu, and 1 others. 2025. Med-r 3: En-
hancing medical retrieval-augmented reasoning of
llms via progressive reinforcement learning.arXiv
preprint arXiv:2507.23541.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. Factscore:
Fine-grained atomic evaluation of factual precision
in long form text generation. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100.
Vincent Nguyen, Sarvnaz Karimi, Maciej Rybinski, and
Zhenchang Xing. 2023. Medredqa for medical con-
sumer question answering: Dataset, tasks, and neural
baselines. InProceedings of the 13th International
Joint Conference on Natural Language Processing
and the 3rd Conference of the Asia-Pacific Chap-
ter of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 629–648.
Long N. Phan, James T. Anibal, Hieu Tran, Shau-
rya Chanber, Lars Peltason, Loren M. Tesche, and
Joyce C. Ho. 2021. Scifive: a text-to-text transformer
model for biomedical literature.arXiv preprint
arXiv:2106.03598.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 conference on empirical methods in natural
language processing, pages 2383–2392.

Alexey Romanov and Chaitanya Shivade. 2018.
Lessons from natural language inference in the clini-
cal domain.arXiv preprint arXiv:1808.06752.
Dongyu Ru, Lin Qiu, Xiaoyu Georgiev, and 1 others.
2024. RAGChecker: A fine-grained framework for
diagnosing RAG. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing.
Alexandre Sallinen, Antoni-Joan Solergibert, Michael
Zhang, Guillaume Boyé, Maud Dupont-Roc, Xavier
Theimer-Lienhard, Etienne Boisson, Bastien Bernath,
Hichem Hadhri, Antoine Tran, and 1 others. 2025.
Llama-3-meditron: An open-weight suite of medical
llms based on llama-3.1. InWorkshop on Large Lan-
guage Models and Generative AI for Health at AAAI
2025.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Yang Wu, and 1 others. 2024.
Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres,
Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin
Clark, Stephen R Pfohl, Heather Cole-Lewis, and
1 others. 2025. Toward expert-level medical ques-
tion answering with large language models.Nature
medicine, 31(3):943–950.
Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov,
and David Krueger. 2022. Defining and characteriz-
ing reward gaming.Advances in Neural Information
Processing Systems, 35:9460–9471.
Yun Song, Wenjia Zheng, Tiedan Chen, Ziyu Wang,
Jiazhao Shi, and Yisong Chen. 2026. Deep neural
network architectures for electrocardiogram classifi-
cation: A comprehensive evaluation.arXiv preprint
arXiv:2602.17701.
J Ben Tamo, Yuxing Lu, Benoit L Marteau, Micky C
Nnamdi, and May D Wang. 2026. Evidencerl: Re-
inforcing evidence consistency for trustworthy lan-
guage models.arXiv preprint arXiv:2603.19532.
George Tsatsaronis, Georgios Balikas, Prodromos
Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R Alvers, Dirk Weissenborn, Anastasia
Krithara, Sergios Petridis, Dimitris Polychronopou-
los, and 1 others. 2015. An overview of the bioasq
large-scale biomedical semantic indexing and ques-
tion answering competition.BMC bioinformatics,
16(1):138.
Juraj Vladika, Annika Domres, Mai Nguyen, Rebecca
Moser, Jana Nano, Felix Busch, Lisa C. Adams,
Keno K. Bressem, Denise Bernhardt, Stephanie E.
Combs, Kai J. Borm, Florian Matthes, and Jan C.
Peeken. 2025a. Improving reliability and explain-
ability of medical question answering through atomic
fact checking in retrieval-augmented LLMs.arXiv
preprint arXiv:2505.24830.Juraj Vladika, Ivana Hacajova, and Florian Matthes.
2025b. Step-by-step fact verification system for med-
ical claims with explainable reasoning. InProceed-
ings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies
(Volume 2: Short Papers), pages 805–816. Associa-
tion for Computational Linguistics.
Yuhui Wang, Tanqiu Jiang, Jiacheng Liang, Charles
Fleming, and Ting Wang. 2026. Mage: Safeguarding
llm agents against long-horizon threats via shadow
memory.Preprint, arXiv:2605.03228.
Yuhui Wang, Rongyi Zhu, and Ting Wang. 2025.
Self-destructive language model.Preprint,
arXiv:2505.12186.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics: ACL 2024, pages
6233–6251.
Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng,
Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun
Tian, Xuan Qi, Weikang Qiu, and 1 others. 2025.
Beyond correctness: Rewarding faithful reasoning
in retrieval-augmented generation.arXiv preprint
arXiv:2510.13272.
Siyi Xun, Yue Sun, Jingkun Chen, Zitong Yu, Tong
Tong, Xiaohong Liu, Mingxiang Wu, and Tao Tan.
2025. Mediqa: A scalable foundation model for
prompt-driven medical image quality assessment. In
International Conference on Medical Image Com-
puting and Computer-Assisted Intervention, pages
339–349. Springer.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Ailing Yu, Lan Yao, Jingnan Liu, Zhe Chen, Jiajun Yin,
Yuan Wang, Xinhao Liao, Zhiling Ye, Ji Li, Yun Yue,
and 1 others. 2025. Medresearcher-r1: Expert-level
medical deep researcher via a knowledge-informed
trajectory synthesis framework.arXiv preprint
arXiv:2508.14880.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
uating text generation with bert.arXiv preprint
arXiv:1904.09675.
Rongyi Zhu, Yuhui Wang, Tanqiu Jiang, Jiacheng Liang,
and Ting Wang. 2025. Self-improving model steer-
ing.Preprint, arXiv:2507.08967.

9 MEDIQA Breakdown
We exclude MEDIQA from the main cross-dataset
table (Table 8) because the MedNLI-Cls configu-
ration performs zero search calls on this subset at
inference, so no retrieved evidence is available for
faithfulness or support calculation; the resulting 0%
is a measurement artefact rather than a verification
failure. For completeness, Table 7 reports the full
MEDIQA column for all four back-ends.
Back-end Supp% BERT Faith
Likelihood-NLI 75.0 .6671.0
MedNLI-Cls 0.0‡.6790.0‡
Hybrid100.6731.0
GPT-4o-mini (full) 87.5 .6571.0
Table 7: MEDIQA (n = 85) breakdown.‡MedNLI-
Cls performs zero searches on MEDIQA, so Sup-
port and Faithfulness default to 0 (no evidence re-
trieved). BERTScore is computed on the generated
answer against the reference and is unaffected. We omit
F1 from this table because logs for MEDIQA F1 are not
retained at per-back-end granularity.
10 Base Reward and FmtScore
The base reward (§3.3) is:
rbase= ˜w emEM+ ˜w f1F1+ ˜w fmtFMTSCORE,(1)
with ˜wk=w k/P
k′wk′and(wem, wf1, wfmt) =
(0.35,0.15,0.10) . EM and F1 follow SQuAD-
style normalisation (lower-case, articles and punc-
tuation removed, max over reference answers).
Letℓ=|ˆa| denote the answer length in charac-
ters, with ℓ= 0 when the <answer> tag is missing.
Then
FMTSCORE(ˆa) =

0ℓ= 0
0.5 0< ℓ <20
0.5 +ℓ−20
12020≤ℓ <80
1ℓ≥80
(2)
where Ltgt= 80 characters is the target length
above which format reward saturates.
11 Cross-Dataset Numerical Breakdown
For completeness, Table 8 provides the numerical
values underlying Figure 3.
12 Cross-Model Validation
To assess whether our findings generalise beyond
Qwen2.5-7B, we replicate selected configurationson two additional base models: Llama-3.1-8B-
Instruct (larger, different family) and Qwen3-4B-
Instruct (smaller, same family). Table 9 reports the
per-dataset breakdown averaged over four held-out
benchmarks (MEDIQA excluded as in the main
results).
Signal collapse is not Qwen-specific.Llama-
3.1-8B with the Likelihood-NLI checker achieves
BERTScore 0.597 and F1 0.210—comparable to
Qwen2.5-7B on the same checker (0.599, 0.197)—
but produces 0% support under self-evaluation.
The collapse pattern reproduces across base mod-
els, confirming that the failure is in the verifier’s
scoring mechanism rather than the policy.
Format stability under GPT-4o-mini is base-
model-dependent.Llama-3.1-8B with the GPT-
4o-mini checker degrades to 50% answer-tag rate
(averaged over four datasets; 14.9% on the med-
ical split alone) and BERTScore 0.437: the pol-
icy partially fails to sustain the multi-turn search-
check-answer format under the strong reward sig-
nal, with the worst collapse on the medical split.
Qwen2.5-7B and Qwen3-4B both maintain ≥97%
Tag rate across all four datasets. This suggests that
the reward hacking cascade (§6) is shaped by the
base model’s instruction-following capacity: sig-
nal collapse is universal but the specific shortcuts
the policy discovers depend on what shortcuts are
reachable within its capability.
Smaller base model: Qwen3-4B.Qwen3-4B
with the Likelihood-NLI configuration reaches
BERTScore 0.594 and 51.0% support rate, be-
tween the collapsed Meditron configuration on
Qwen2.5-7B (0.599 BERT, 0% self-support) and
the strongest checkers. With MedNLI-Cls, Qwen3-
4B reaches BERTScore 0.591, F1 0.217, and sup-
port tat 85.3% (Table 11). Format stability is 100%
across all datasets, suggesting that smaller models
in the same family can sustain the multi-turn format
under RL pressure.
checker signal strength is policy-dependent.
The MedNLI-Cls checker that registers as amoder-
atesignal on Qwen2.5-7B (54.0% support) reg-
isters as astrongsignal on Qwen3-4B (85.3%
support) on the same claim–evidence pairs and
the same checker checkpoint. This is direct ev-
idence for the boundary-condition prediction in
Appendix 25: signal strength is a property of the
policy–checker pair, not of the checker alone. The

Checker Back-End Medical MedicationQA BioASQ MedQuAD Avg
(n=87) (n=674) (n=618) (n=100) (1,479)
support tate (%)↑
Likelihood-NLI (no GPT) 72.5 78.8 84.1 85.1 80.1†
MedNLI-Cls (no GPT) 29.5 49.6 63.7 73.2 54.0
Hybrid (GPT extract + NLI) 60.3 85.2 75.9 81.7 75.8
GPT-4o-mini (full) 77.1 84.4 91.5 91.3 86.1
BERTScore↑
Likelihood-NLI (no GPT) .597 .567 .665 .566 .599
MedNLI-Cls (no GPT) .575 .578 .671 .576 .600
Hybrid (GPT extract + NLI) .554 .518 .623 .565 .565
GPT-4o-mini (full) .569 .571 .656 .568 .591
Faithfulness (no contradiction)↑
Likelihood-NLI (no GPT) .944 .985 .942 .987 .964
MedNLI-Cls (no GPT) .984 .919 1.0 .986 .972
Hybrid (GPT extract + NLI) .966 1.0 .982 1.0 .987
GPT-4o-mini (full) .957 .978 .963 .990 .972
Token F1↑
Likelihood-NLI (no GPT) .193 .150 .261 .182 .197
MedNLI-Cls (no GPT) .215 .170 .278 .197 .215
Hybrid (GPT extract + NLI) .173 .140 .240 .191 .186
GPT-4o-mini (full) .173 .149 .243 .172 .184
Table 8: Cross-dataset breakdown (Qwen2.5-7B).Cell color encodes per-column rank(darker green = better);
boldmarks per-column best. Avg averages over 4 datasets (1,479 samples); MEDIQA in Appendix 9.†Likelihood-
NLI self-evaluation gives 0% (§5); shown values are GPT-4o-mini evaluation.
Base Checker BERT F1 Sup% Faith Tag%
Reference: Qwen2.5-7B
Qwen2.5-7B Likelihood-NLI .599 .197 80.1†.964 95.7
Qwen2.5-7B MedNLI-Cls.600 .21554.0 .972100
Qwen2.5-7B Hybrid .565 .186 75.8.98795.9
Qwen2.5-7B GPT-NLI .591 .18486.1.972 98.3
Llama-3.1-8B
Llama-3.1-8B Likelihood-NLI .597 .210 0‡.814 98.8
Llama-3.1-8B MedNLI-Cls .521 .163 49.4 .823 93.2
Llama-3.1-8B Hybrid .563 .195 72.0 .934 96.4
Llama-3.1-8B GPT-NLI .437 .173 0‡.726 50.0♭
Qwen3-4B
Qwen3-4B Likelihood-NLI .594 .209 51.0 .952100
Qwen3-4B MedNLI-Cls .591.21785.3 .980100
Qwen3-4B Hybrid .555 .170 93.0 .980 98.0
Qwen3-4B GPT-NLI .565 .205 82.0 .983 97.4
Table 9: Cross-model validation. Averages over
four held-out datasets (1,479 samples); MEDIQA ex-
cluded.†Likelihood-NLI self-evaluation gives 0%;
shown is GPT-4o-mini evaluation.‡Self-evaluation.
♭Format degradation: only 50% of outputs contain valid
<answer> tags on average (14.9% on medical), see text.
Qwen3-4B + MedNLI-Cls pair sits in the strong-
signal regime our cascade analysis predicts is risky,
yet doesnotexhibit the cascade end-state (Tag rate
stays at 100%, F1 matches Qwen2.5-7B MedNLI-
Cls at 0.217). The most parsimonious reading is
that smaller policies have fewer reachable shortcuts
under the same reward; testing this directly is left
for the planned cascade training-dynamics replayDataset BERT F1 Supp% Faith Tag%
Medical (n=87) .573 .194 37.3 .941 100
MedicationQA (n=674) .570 .185 38.4 .911 100
BioASQ (n=618) .642 .232 59.1 .968 100
MedQuAD (n=100) .593 .224 69.4 .988 100
Avg (4 datasets) .594 .209 51.0 .952 100
Table 10: Qwen3-4B + Likelihood-NLI checker per-
dataset breakdown. MEDIQA excluded (search=0
anomaly).
Dataset BERT F1 Supp% Faith Tag%
Medical (n=87) .570 .207 86.2 .979 100
MedicationQA (n=674) .575 .191 80.6 .982 100
BioASQ (n=618) .629 .241 84.4 .984 99
MedQuAD (n=100) .591 .230 89.8 .977 100
Avg (4 datasets) .591 .217 85.3 .980 99.75
Table 11: Qwen3-4B + MedNLI-Cls per-dataset break-
down. MEDIQA excluded (search=0 anomaly).
(Appendix 20).
Cross-model coverage.We replicate the
Likelihood-NLI and MedNLI-Cls back-ends
on Qwen3-4B and the Likelihood-NLI and
GPT-4o-mini back-ends on Llama-3.1-8B; the
remaining (base model, checker) cells are deferred
for compute reasons and discussed in Limitations.

Cross-domain transfer to HotpotQA.We ad-
ditionally evaluate Qwen2.5-7B medical-trained
models on HotpotQA (Yang et al., 2018). Mod-
els achieve F1 = 0.071 with 98% zero-search rate,
compared to Search-R1’s reported 0.403 average
EM when trained in-domain. RL-trained tool-use
behaviour is thus domain-specific. The GPT-4o-
mini checker itself, however, correctly identifies
entailment and contradiction on general-domain
claim–evidence pairs—domain specificity arises
from the RL training, not the verification compo-
nent.
13 Retriever Details
This appendix gives the full specification of the
<search> tool referenced in §3. The retriever is
held constant across every training run and every
evaluation result reported in this paper.
Encoder.We use MedCPT (Jin et al., 2023),
a biomedical bi-encoder pretrained on ∼255M
PubMed query–article click pairs, with separate
query and article towers ( ∼110M parameters each)
producing 768-dimensional embeddings. The en-
coder is frozen for all RL training; only the policy
is updated.
Corpus.We index the MedRAG corpus (Xiong
et al., 2024), which covers four medical sources:
•PubMed abstracts: 23.9M entries, title +ab-
stract.
•StatPearls: 0.3M passages chunked from clin-
ical reference articles.
•Medical textbooks (the 18 textbooks shipped
with the MedRAG release): 0.1M chunks.
•Wikipedia medical pages (optional source in
MedRAG): enabled.
Each source contributes pre-chunked passages of
256 tokens with 32-token overlap; the resulting
index contains 26M passage embeddings.
Index and serving.Embeddings are stored in
a FAISS flat inner-product index ( IndexFlatIP ;
no quantisation) and served from a single-node
HTTP API. Each <search> call sends a UTF-8
query string; the server encodes it with the Med-
CPT query tower, runs an exact top- kinner-product
search, and returns the top- k= 5 passages with
their source identifiers. End-to-end latency is
<80 ms per call on a single A100.Query construction.The agent emits free-form
queries between <search> . . .</search> tags.
Queries are not rewritten or expanded by the re-
triever side; query rewriting, if any, is learned im-
plicitly by the policy through GRPO updates.
Evidence formatting.Returned passages are
concatenated in rank order with a short source-tag
prefix (e.g. [StatPearls] ) and inserted into the
rollout context inside a <information> block. The
total injected evidence is truncated to 768 tokens
(this truncation is the source of the evidence-fix
ablation in §5.2).
Why hold the retriever fixed.The four checker
back-ends compared in §3 can in principle interact
with retrieval (e.g., a stronger checker may train
the policy to issue more precise queries). To at-
tribute quality–grounding differences to checker be-
haviour rather than retrieval variation, we share the
same encoder, index, top- k, and truncation across
all training and evaluation runs. Joint optimisation
of the retriever with the policy is left to future work.
14 GRPO Advantage Computation
For each question qwe sample Nrollouts {τi}N
i=1
with rewards {Ri}and compute group-normalised
advantages
ˆAi=Ri−µg
σg+ϵ,(3)
µg=1
NX
jRj,(4)
σg=s
1
NX
j(Rj−µg)2,(5)
withϵ= 10−6.ˆAiis broadcast to every response
token of τiand consumed by the standard PPO-
clipped surrogate.
15 MedNLI-Cls Training Details
Base model.We start from PubMedBERT-
base (microsoft/BiomedNLP-PubMedBERT-base-
uncased-abstract-fulltext) (Gu et al., 2021), a
110M-parameter encoder pretrained from scratch
on PubMed abstracts and PubMedCentral full-text
articles.
Dataset.We use MedNLI (Romanov and Shiv-
ade, 2018), a clinical natural language inference
benchmark with premise sentences drawn from

MIMIC-III discharge summaries and physician-
written hypotheses labelledENTAIL,NEUTRAL, or
CONTRADICT. The official split contains 11,232
training, 1,395 development, and 1,422 test pairs.
Fine-tuning.We attach a 3-way linear classi-
fication head over the [CLS] representation and
fine-tune end-to-end with AdamW at learning rate
2×10−5, batch size 32, max_length = 256, weight
decay 0.01, and 3 epochs. Inputs are formatted as
[CLS] premise [SEP] hypothesis [SEP] . We
select the checkpoint with the highest development-
set accuracy.
Test-set performance.The selected checkpoint
reaches 84.1% accuracy on the 1,422-pair MedNLI
test set, comparable to published MedNLI-Cls base-
lines. Per-label F1 isENTAIL0.85,NEUTRAL0.81,
CONTRADICT0.86.
Inference pipeline.At checker time, the agent’s
draft answer is split into sentences (claim units);
each claim is paired with the concatenated re-
trieved evidence and scored by the classifier. The
argmax label feeds into Eq. ??to produce ϕcheck∈
[−1,+1].
Serving.The checker is served as an HTTP end-
point with batched inference; a typical rollout re-
ceives a verdict in <50 ms on a single A100, com-
parable to the latency of the Likelihood-NLI path
and substantially faster than the GPT-4o-mini API.
Reproducibility.All hyperparameters above are
deterministic given a fixed seed. We will release the
fine-tuning script and the serving wrapper along-
side the policy checkpoints.
16 Rollout AlgorithmAlgorithm 1Checker-Guided Rollout with Auto-
Check
Require: question q; reference y∗; policy πθ; retriever R;
checkerV; budgets(B s, Bc, Bt)
1:H←[q];n s, nc, nt←0;ϕ check←0
2:whilen t< B tdo
3:a∼π θ(· |H);n t←n t+ 1
4:ifa=⟨search⟩q′andn s< B sthen
5:e← R(q′);H←H⊕e;n s←n s+ 1
6:else ifa=⟨check⟩yandn c< B cthen
7:v← V(y, e);H←H⊕v;n c←n c+ 1
8:ϕ check←SCORE(v)▷Eq.??
9:else ifa=⟨answer⟩ˆythen
10:ifn c< B cthen▷auto-check
11:v← V(ˆy, e);ϕ check←SCORE(v)
12:end if
13:break
14:end if
15:end while
16:r base←Eq. 1;P fmt←format penalty
17:returnR=r base·(1 +α ϕ check) +P fmt
17 Multiplicative vs. Additive Reward
Coupling
The reward function in §3.3 couples the checker re-
ward multiplicatively rather than additively. When
the answer is incorrect ( rbase→0),∂R/∂ϕ check→
0, so a high faithfulness score on a wrong answer
cannot substitute for correctness. Conversely, when
the answer is correct, the multiplier sharpens the
gradient toward grounded answers. Additive cou-
pling ( R=r base+α ϕ check) admits a failure mode
in which the policy optimises ϕcheck at the expense
ofrbase—one face of the cascade we diagnose in
§6.
18 Language Collapse and English-Only
Fix
In the triage-guarded GPT-NLI configuration,
64.4%of generated answers are in Chinese de-
spite English-only training data. The policy ex-
ploits Qwen2.5’s multilingual pretraining distribu-
tion when triage restricts the turn budget, since
Chinese tokens carry comparable F1 reward but
avoid checker contradiction risk in the English-
conditioned NLI pipeline. When the mixed-
language outputs are post-hoc filtered to English-
only samples, F1 on that subset reaches0.214, the
highest single-answer F1 we observed across all
configurations—confirming that the model is not
failing on the underlying QA task but is exploiting
a language-conditional reward gap.
Fix.Adding “Always respond in English” to
the system prompt and retraining the policy from

scratch eliminates Chinese outputs and yields
F1 = 0.203 (across all samples) with full format sta-
bility and checker support. The retrained number
is slightly lower than the post-hoc filtered subset
because it is averaged over all questions rather than
the easier English-by-default subset.
Stage Shortcut Countermeasure
GPT-NLI checker Ultra-short (130
chars)Format penalty
+ Format penalty No search
(98%)Search bonus
(+0.1)
+ Search bonus Short again
(141 chars)Triage budget
+ Triage Chinese
(64.4%)English-only
prompt
Table 12: The reward hacking cascade end-to-end.
Cascade summary table.
Domain note.Language collapse is specific to
Qwen’s multilingual pretraining; we did not ob-
serve it on Llama-3.1-8B (which has weaker non-
English coverage) or on Qwen3-4B (which inherits
Qwen multilingualism but shows the cascade atten-
uation discussed in §4.2).
19 Three Root Causes of Log-Prob
Collapse
The body’s collapse summary (§5.2) is supported
by a controlled three-step diagnostic chain on the
log-prob pathway.Evidence truncationat the
default 256-token tool-response limit strips re-
trieved passages to 1–2 sentences; extending to
768 tokens raises support rate from 0% to 35.7%.
Non-atomic claim extractionfrom sentence split-
ting produces multi-proposition claims; replacing
with GPT-4o-mini extraction raises support rate
to 37.5%.Residual log-prob biaspersists even
with full evidence and atomic claims, but is largely
closed by swapping the scorer (Hybrid back-end:
75.8%; fully GPT-verified: 65.6–77.5%). A small
gap to MedNLI-Cls remains in borderline cases
where log-prob scoring favours neutral and the clas-
sifier commits to entail or contradict.
20 Reward Hacking Cascade: Training
Dynamics
The cascade analysis in §6 reports theend-state
behaviour of each shortcut; Figure 4 plots per-step
training dynamics for the unguarded GPT-4o-mini
and the MedNLI-Cls configurations. We track four
0 25 50 75 100 125 150 175 200
Training step0100200300400Mean length (chars)length
collapse(a) Answer length
0 25 50 75 100 125 150 175 200
Training step0.00.20.40.60.81.0Mean searches / rolloutsearch
avoidance(b) Search-call rate
0 25 50 75 100 125 150 175 200
Training step0.000.050.100.150.20Fraction of rolloutslanguage
drift(c) Non-English token fraction
0 25 50 75 100 125 150 175 200
Training step0.00.20.40.60.81.0rcheck
reward
saturation(d) Mean checker rewardGPT-4o-mini (strong) PubMedBERT-MedNLI (moderate)Figure 4:Cascade dynamics under strong vs. moder-
ate verification.Training trajectories for the unguarded
GPT-4o-mini configuration (green, strong signal) and
the MedNLI-Cls configuration (purple, moderate signal).
(a)Mean answer length per rollout.(b)Mean number
ofsearch calls per rollout.(c)Fraction of rollouts con-
taining non-English tokens.(d)Mean checker reward
¯rcheck. Under strong verification, the four panels exhibit
asequential phase transition: reward saturates first (d,
∼step 50), followed by length collapse (a, ∼step 60),
search avoidance (b, ∼step 110), and language drift (c,
∼step 140). The moderate-signal configuration remains
stable across all four signals throughout training, sup-
porting the hypothesis that strong verification signals
trigger the cascade (§6) while moderate signals do not.
series: (a) mean answer length; (b) per-rollout
search-call rate; (c) Fraction of rollouts containing
non-English tokens (sampling-time; eval greedy-
decode reaches 64.4%, App. 18); (d) mean checker
reward ¯rcheck. The strong-signal (GPT-4o-mini)
configuration shows a sequential phase transition:
reward saturates first ( ∼step 50), answer length col-
lapses next ( ∼step 60), search rate drops ( ∼step
110), and language drift surfaces last ( ∼step 140).
The moderate-signal (MedNLI-Cls) configuration
shows none of these transitions and maintains a sta-
ble but lower checker reward throughout, support-
ing the hypothesis that strong verification signals
trigger the cascade (§6) while moderate signals do
not.
21 Triage Ablation: Per-Checkpoint
Details
Budget configuration.The reported experiments
use per-tier budgets (search, check, turn) of (1, 1,
3) for easy, (2, 2, 5) for medium, and (4, 3, 7) for
hard, overriding the code defaults of (1, 0, 2), (2, 1,
4), and (4, 2, 6) respectively—the override ensures
the checker receives at least one budget unit even
for easy questions.

Classification priority.The difficulty tier is as-
signed from three sources in order of precedence:
(1) an explicit field in extra_info (manual
override, used in cross-domain evaluation); (2) a
faithfulness_score field from a previous roll-
out (warm-start from prior run statistics); (3) the
rule-based heuristic (Eq. 6 below), used in all train-
ing configurations reported in this paper.
Heuristic scoring formula.For held-out evalu-
ation datasets without ensemble-agreement labels
(MedicationQA, BioASQ, MedQuAD, MEDIQA),
the heuristic scores each question on five surface
features:
score(q) = 0.30x long+ 0.20x multihop
+ 0.20x clinical + 0.15x multiq
+ 0.15x bullets,(6)
where xlong=⊮[|q| w≥120or|q| c≥700] ,
xmultihop =⊮[why/how/vs/differential∈q] ,
xclinical =⊮[≥3 clinical keywords ],xmultiq =⊮[≥
2?], and xbullets =⊮[bullet structure] . A score be-
low0.35 iseasy; [0.35,0.65) ismedium; ≥0.65 is
hard.
Escalation triggers.Four conditions trigger an
upgrade: search failure, empty search result,
checker contradiction rate above 0.30, or checker
support rate below 0.40. Escalation is enabled by
default ( online_escalation=True ) and resets the
per-call counters upon triggering.
Triage cannot rescue checker collapse.Across
four Likelihood-NLI checkpoints (no-triage step
93,+triage step 93, explicit step 188, +evidence-
fix step 188), self-evaluated Support remains 0%
and Neutral ≥97% regardless of triage state. Re-
covery requires the targeted log-prob fixes in Ta-
ble 5.
22 Infrastructure vs. Signal Ablation
A natural follow-up question is whether the
moderate-signal benefit (§ ??) comes from Pub-
MedBERT’s biomedical pretraining or fromany
calibrated classification head. Our main results
hold the architecture fixed (MedNLI-Cls, i.e.,
PubMedBERT-MedNLI) and vary only the scor-
ing mechanism (log-probability vs. classification
head), so the isolated claim is about the scoring
mechanism. Swapping in DeBERTa-MNLI closes
most of the moderate-signal gap (rows 3–4 in Ta-
ble 13).Table 13 reports a second-order ablation that
holds the RL infrastructure constant (Qwen2.5-
7B base, α= 1 , triage controller on) and
varies the checker identity at fixed scoring mech-
anism (classification head with softmax over EN-
TAIL/NEUTRAL/CONTRADICT). The “no-checker”
row removes the auto-check loop entirely (F1 + for-
mat reward only). The “checker-loop-no-reward”
row keeps the auto-check loop active but masks
the checker reward ( α= 0 ); this isolates whether
benefits come fromseeingchecker feedback during
rollouts or fromreward-shapingon it.
Verifier BERT F1 Supp%
No checker 0.558 0.182 32.3
Loop only (α=0) 0.576 0.191 34.5
DeBERTa-MNLI head 0.591 0.202 45.6
MedNLI-Cls (ours) 0.600 0.215 54.0
Table 13: Infrastructure-vs-signal ablation. “No
checker” = F1 +format only; “Loop only” keeps the
auto-check loop with α=0 . Rows 3–4 contrast a generic
NLI head with a biomedical one (§??).
23 Training Hyperparameters
All runs use GRPO with batch size 4, learning rate
10−6, entropy coefficient 0.005 , and KL penalty
0.001 . Training-step counts vary across configu-
rations (93–377 steps); the GPT-NLI unguarded
run is trained for 377 steps as the entry point to
the cascade analysis (§6). All other configurations
are trained until checker reward and answer-length
metrics stabilise (typically 150–200 steps).
24 Concurrent Verifier-as-Reward
Systems
The most directly overlapping concurrent system is
EvidenceRL (Tamo et al., 2026), which also pairs
PubMedBERT-MNLI / MedNLI with GRPO inside
a medical RAGagent and reports a grounding-style
reward (Table 1). Two differences shape how our
results should be read alongside theirs. First, our
contribution is not a competing system; it is a di-
agnosis of the verifier-as-reward design space. We
treat checker output distribution, not checker ac-
curacy, as the primary training-time variable, and
show that LLM log-probability scoring collapses
on thesameclaim–evidence pairs that a calibrated
MedNLI-Cls head scores non-degenerately (§5).
Second, we report a three-step reward-hacking
cascade that emerges only understrongverifica-

tion signal, where an 86.1%-support GPT-4o-mini
checker trains a model withloweranswer qual-
ity than a 54.0%-support MedNLI-Cls checker
(§6). Read in this frame, EvidenceRL, VERITAS,
and MedRAGChecker occupy specific operating
points in the design space: EvidenceRL lands at
the moderate-signal regime our cascade analysis
predicts is safer; VERITAS reports checker unre-
liability without isolating log-probability scoring
as the failure mechanism; MedRAGChecker uses
claim-level verification at evaluation rather than
training time, so the reward hacking cascade does
not apply.
Table 1 summarises five concurrent verifier-as-
reward systems on seven design dimensions.
25 Predicted Boundary Conditions
Three conditions are likely to break the findings
in §??; we name them so reviewers and follow-up
work can test them directly.
(i) checker strong but uncollapsed.Our cas-
cade emerges from thecombinationof high support
rate and on-policy GRPO training without an ex-
plicit shortcut-detection step. A checker with com-
parable accuracy but more calibration noise (e.g.,
a sharpened-softmax PubMedBERT fine-tuned to
push entailment mass toward 80%) might trigger
the cascade even without GPT-class semantics.
(ii) Different base-policy capacity.Reachabil-
ity of cascade shortcuts depends on the policy’s
instruction-following capacity. Our Qwen3-4B
replication is a partial test: MedNLI-Cls registers
as astrongsignal on Qwen3-4B (85.3% support, vs.
54.0% on Qwen2.5-7B) yet the cascade end-state
does not appear (Tag rate 100%, F1 0.217 matches
the Qwen2.5-7B MedNLI-Cls configuration; see
Appendix 12). This is consistent with smaller poli-
cies having fewer reachable shortcuts under the
same reward, and shows that signal strength is a
property of the policy–checker pair rather than of
the checker alone. A stronger base policy in the
30–70B range may inhabit the opposite regime.
(iii) Open-domain or non-medical RAG.The
benefits we attribute to MedNLI-Cls are domain-
specific: in open-domain settings, a generic NLI
classifier may collapse where the medical-domain
head does not, inverting the moderate-signal advan-
tage.
We treat these as falsifiable predictions; failing
any of (i)–(iii) would localise our findings furtherrather than refute them.
26 Reward Weight Sensitivity
The reward function R=r base·(1+α ϕ check)+P fmt
(§3.3) uses α= 1 in all main-text results. To
probe robustness to this choice, Table ??reports
a sensitivity sweep across α∈ {0.5,1.0,2.0}
on Qwen2.5-7B-Instruct with both MedNLI-Cls
and GPT-4o-mini back-ends. The cascade hy-
pothesis predicts a monotone trend for GPT-4o-
mini: higher αamplifies the cascade and worsens
BERTScore. For MedNLI-Cls, the moderate-signal
regime should be robust acrossα.
VerifierαBERTScore F1 Supp%
MedNLI-Cls 0.5 0.531 0.220 46.4
MedNLI-Cls 1.0 0.600 0.215 54.0
MedNLI-Cls 2.0 0.545 0.142 32.1
GPT-4o-mini 0.5 0.547 0.187 74.3
GPT-4o-mini 1.0 0.591 0.184 86.1
GPT-4o-mini 2.0 0.584 0.173 88.6
27 Zero-Shot Evaluation on
MedBrowseComp and MIRAGE
To test out-of-distribution generalisation, we evalu-
ate the two checker back-ends zero-shot (no further
fine-tuning) on MedBrowseComp (Xiong et al.,
2024; Chen et al., 2025) and the broader MIRAGE
benchmark suite. Table 14 reports zero-shot accu-
racy under both verifiers; results follow the same re-
porting format as Table 8. The boundary-condition
prediction in Appendix 25 states that MedNLI-Cls
should retain its moderate-signal advantage in the
medical sub-domain of MIRAGE but may close to
GPT-4o-mini on retrieval-heavy MedBrowseComp
queries; we report whether this prediction holds.
Table 14 confirms it: MedNLI-Cls leads on MI-
RAGE (58.3% vs. 52.1%) but the gap narrows on
retrieval-heavy MedBrowseComp (8.4% vs. 6.2%).

Benchmark Verifier Accuracy Search
MIRAGE MedNLI-Cls 58.3% 1.2
MIRAGE GPT-NLI 52.1% 0.2
MedBrowseComp MedNLI-Cls 8.4% 0.6
MedBrowseComp GPT-NLI 6.2% 0.17
Table 14: Zero-shot evaluation on MedBrowseComp
and MIRAGE. Note: Our agent is trained on free-
text generation; accuracy on multi-choice (MIRAGE)
and short-fact (MedBrowseComp) tasks is reported
only to test format-transfer, not to compete with task-
specialised systems.