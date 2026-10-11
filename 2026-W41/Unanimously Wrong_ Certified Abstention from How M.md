# Unanimously Wrong: Certified Abstention from How Medical LLM Consensus Forms

**Authors**: Xiaoyang Wang, Tianrui Wang, Christopher C. Yang

**Published**: 2026-10-06 00:59:31

**PDF URL**: [https://arxiv.org/pdf/2610.07570v1](https://arxiv.org/pdf/2610.07570v1)

## Abstract
In clinical practice, agreement among independent experts is treated as evidence of reliability, and multi-round consensus has become a core mechanism of agentic medical question-answering systems. When such a system must decide whether to trust its own answer, the prevailing signal is again agreement, now among the sampled answers. But agreement is a fragile proxy for correctness. A system can be unanimously wrong, returning the same incorrect answer on every sample, and on these questions agreement-based signals carry no information. The cause is that these signals read only the final state of the consensus and discard how it was reached. Agreement that was reached by resolving disagreement with evidence looks identical, at the end, to agreement that was present from the first sample because every sample shares one misconception. ProbeGuard is a certified abstention framework that bases the abstention decision on how the consensus formed. Process features trace agreement trajectories, minority persistence, and retrieval saturation. For unanimous votes, rationale semantic entropy checks whether the reasons behind the vote cohere, and an active probe retrieves counter-evidence and measures whether the consensus survives. A stratified Learn-then-Test calibration then converts these scores into a distribution-free bound on selective risk. We evaluate ProbeGuard on three medical QA benchmarks and a hard-frontier reference, with a published multi-round agentic RAG substrate, against six abstention baselines. On MedQA, 13.4% of unanimous votes are wrong, and no agreement-based signal can flag them. Process signals raise the discrimination of correct from incorrect consensus from chance to 0.696 AUROC. The certified rule answers six in ten unanimous-layer questions at an observed selective risk of 9.0%, and nine in ten once in-domain calibration data accumulate.

## Full Text


<!-- PDF content starts -->

Unanimously Wrong: Certified Abstention from How
Medical LLM Consensus Forms
Xiaoyang Wang
Drexel University
xw388@drexel.eduTianrui Wang
University of Michigan, Ann Arbor
tianruiw@umich.eduChristopher C. Yang
Drexel University
chris.yang@drexel.edu
Abstract
In clinical practice, agreement among independent experts is treated as evidence of
reliability, and multi-round consensus has become a core mechanism of agentic
medical question-answering systems. When such a system must decide whether
to trust its own answer, the prevailing signal is again agreement, now among the
sampled answers. But agreement is a fragile proxy for correctness. A system
can beunanimously wrong, returning the same incorrect answer on every sample,
and on these questions agreement-based signals carry no information. The cause
is that these signals read only the final state of the consensus and discard how
it was reached. Agreement that was reached by resolving disagreement with
evidence looks identical, at the end, to agreement that was present from the first
sample because every sample shares one misconception.ProbeGuardis a certified
abstention framework that bases the abstention decision on how the consensus
formed. All of its components are computed from the system’s execution log.
Process features trace agreement trajectories, minority persistence, and retrieval
saturation. For unanimous votes,rationale semantic entropychecks whether the
reasons behind the vote cohere, and an active probe retrieves counter-evidence and
measures whether the consensus survives. A stratified Learn-then-Test calibration
then converts these scores into a distribution-free bound on selective risk. This lets
the system separate unanimous votes that are backed by knowledge from those that
are not, and answer only where its error rate is provably controlled. We evaluate
ProbeGuard on three medical QA benchmarks and a hard-frontier reference, with
a published multi-round agentic RAG substrate, against six abstention baselines.
On MedQA, 13.4% of unanimous votes are wrong, and no agreement-based signal
can flag them. Process signals raise the discrimination of correct from incorrect
consensus from chance to 0.696 AUROC. The certified rule answers six in ten
unanimous-layer questions at an observed selective risk of 9.0%, and nine in
ten once in-domain calibration data accumulate. Code: https://github.com/
wangxiaoyang0412/probeguard.
1 Introduction
Language models now encode substantial clinical knowledge [Singhal et al., 2023] and are moving
into clinical question answering, where their answers increasingly inform triage, guideline lookup,
and decision support. In these settings the failure that matters most is aconfidentlywrong answer. In
a randomized trial, physicians exposed to erroneous LLM suggestions lost 14 percentage points of
diagnostic accuracy, despite prior AI-literacy training and voluntary consultation [Qazi et al., 2026];
an answer delivered without any signal of doubt propagates directly into downstream decisions. The
model itself cannot be trusted to raise that signal: frontier models almost never abstain when they
should, and reasoning-oriented training makes abstentionworse[Kirichenko et al., 2025]. A deployed
system therefore needs reliable abstention, so that the questions it cannot answer are routed to a
clinician [Machcha et al., 2026, Li et al., 2024b].
Preprint.
arXiv:2610.07570v1  [cs.AI]  6 Oct 2026

Q2: A patient hospitalized for one week after a stroke develops fever, purulent cough, and a new lung consolidation. What is the most likely causal microorganism? A) Streptococcus pneumoniaeB) Mycobacterium tuberculosisC) Haemophilus influenzaeD) Staphylococcus aureusQ1: The only immunoglobulin found as a dimer has what primary function? A) Protect against invasive helminth infectionB) Protect against viral infectionsC) Inhibiting bacterial adherence and colonization of mucous membranesD) Provides the most specific recognition to circulating antigens in the bloodstream
Agent Votes (n = 8)    8/8 UnanimousAnswerC
Agent Votes (n = 8)    8/8 UnanimousHigh ConfidenceAnswerA
Unanimously wrong  no alarm raised
Unanimously correct
(a) Snapshot-based abstentionQ1
Agent Votes (n = 8)    8/8 UnanimousRound1–InstantUnanimity
RationaleBubbleInterlinking
Tightlyinterlinkedrationales——knowledge-backedQ2
Agent Votes (n = 8)    8/8 UnanimousRound1–InstantUnanimity
RationaleBubbleInterlinking
Counter-evidenceHolds
Answer+Certificate:Risk≦α
Counter-evidence
flips
Abstain → refer to Clinician 
Disconnected rationales——inherited error(b) ProbeGuard (Ours)Figure 1: Two unanimous consensuses are indistinguishable to snapshot-based abstention (a), yet
one is knowledge-backed and one is inherited from a shared misconception. ProbeGuard (b) checks
whether the rationales behind a unanimous vote support one another and stress-tests the consensus
with retrieved counter-evidence, answering only where selective risk is certifiably bounded.
The dominant abstention signals estimate confidence from a single round of samples: the model’s
verbalized certainty [Kadavath et al., 2022, Tian et al., 2023, Xiong et al., 2024b], or the agreement
among independently sampled answers [Wang et al., 2023, Manakul et al., 2023, Kuhn et al., 2023].
Agreement scores have even been calibrated into conformal abstention rules directly [Yadkori et al.,
2024]. Moving these signals to a modern multi-round consensus system [Wu et al., 2026] exposes
their limits. Agreement can beinheritedfrom a shared training-data error, in which case a model
gives the same wrong answer eight times. Multi-round systems stop deliberating as soon as agreement
is reached, so the terminal vote is 1 for every converged question. And on the questions where the
system is unanimously wrong, no vote-based signal can raise an alarm. On our substrate, 59% of
MedQA questions reach first-round unanimity, and 13.4% of those unanimous votes are wrong; on
expert-level questions the rate rises to 79.0% (Figure 1).
These signals discard theformation process. Consider a panel of physicians who end up agreeing.
All of them may have known the answer from the start, or all of them may share one misconception.
A forced vote can also produce agreement, and so can a discussion in which they disagreed until one
of them found the evidence that settled the question. The final tally is identical in each case, and the
differences lie only in the record of the discussion. Multi-round consensus systems write this record to
their execution logs, including which candidates disagreed, for how many rounds, which documents
were retrieved, and whether the vote changed afterward. Yet prior work discards the trajectory once
the answer is produced [Wu et al., 2026, Manczak et al., 2025]. When the vote is unanimous from
the start, the trajectory is trivial and the check moves one level down, to the rationales. Thereasons
behind the vote can be checked for coherence, which extends semantic entropy [Farquhar et al., 2024]
from answers to rationales, and the consensus can be tested directly by supplying opposing evidence.
Present work.We introduce ProbeGuard, which turns consensus formation into a certified abstention
decision. Process features computed from the execution log trace how the vote evolved; for unanimous
votes, rationale semantic entropy and an active counter-evidence probe read the evidence behind the
vote; and Learn-then-Test calibration [Angelopoulos et al., 2022, 2024] within process strata yields a
selective-risk bound that holds without distributional assumptions, where existing medical abstention
methods provide heuristic scores [Dang et al., 2025, Shankar et al., 2025]. Every multi-round system
already writes this log, and because the scores are training-free, the entire calibration set can be spent
on threshold selection, and the thresholds it returns carry a guarantee a deployment can act on.
We make four contributions. (1) To our knowledge, we give the first abstention decision derived
fromhowmulti-round consensus forms, with a distribution-free selective-risk guarantee, in place of
snapshot-based confidence assessment. (2) We instrument the execution log with trajectory features
at zero additional inference cost, extend semantic entropy from answers to the concluding claims
of rationales, and introduce an active counter-evidence probe whosekeep ratemeasures whether
a unanimous vote survives opposing evidence. (3) On questions where all N=8 candidates agree
2

Table 1: Comparison of abstention approaches for medical QA. External resources, scoring signals,
and risk guarantees are reported for each method.
Method External resource Abstention score Risk guarantee
KnowGuard medical knowledge graph evidence sufficiency✗
Energy-based scoring separately trained scorer energy✗
Conformal abstention none answer agreement global
ProbeGuard (ours) none consensus formationper-stratum
immediately, every vote-based signal is constant, yet 100 of 746 such votes on MedQA are wrong.
We measure this failure mode across four benchmarks and three backbones. (4) The process score
separates correct from incorrect consensus at 0.696 AUROC on MedQA, where terminal agreement
scores 0.512, and at 0.704 on MMLU-Pro medicine. Wrong consensus flips under counter-evidence at
14.0–26.5%, against 3.3–5.5% for correct consensus. Stratified Learn-then-Test calibration answers
0.553 of the unanimous layer at 9.0% observed risk (α=0.15, 50 splits).
2 Related Work
Abstention in medical question answering.Abstention is increasingly treated as a required
capability of clinical language systems. KnowGuard grounds abstention in evidence-sufficiency
scores over an external medical knowledge graph [Dang et al., 2025]; MediQ converts insufficient
information into follow-up questions in interactive settings [Li et al., 2024b]; MedAbstain contributes
a unified evaluation protocol with explicit abstention options and adversarial variants [Machcha et al.,
2026]; and, closest to our setting, an energy-based model scores out-of-scope queries for a medical
RAG assistant [Shankar et al., 2025]. These systems either require resources beyond the deployed
pipeline or produce heuristic scores whose threshold comes with no error guarantee. ProbeGuard
needs only the system’s own execution log and a probe built from its own retrieval stack, and returns
a threshold with a certified selective-risk bound (Table 1).
Consistency- and confidence-based uncertainty.For black-box LLMs, the dominant signal fam-
ilies elicit confidence verbally [Kadavath et al., 2022, Tian et al., 2023, Xiong et al., 2024b] or
measure agreement across independently sampled responses: self-consistency [Wang et al., 2023],
sample-divergence checks [Manakul et al., 2023], and semantic entropy, which clusters samples by
bidirectional entailment [Kuhn et al., 2023, Farquhar et al., 2024]; recent surveys place consistency-
based estimators among the most widely used black-box approaches [Shorinwa et al., 2025, Liu et al.,
2025]. All of these compute their score from a single round of samples. As discussed in §1, agreement
can be a shared error, and answer-level agreement can hide disagreement at the reasoning level even
in debate systems [Wang and Yang, 2026]. Our features use the same samples and additionally the
rounds between the first sample and the final vote.
Multi-round deliberation and its evaluation.Agentic RAG pipelines improve question answering
by coordinating specialized agents [Nguyen et al., 2025], and in medical QA by iterating conflict-
guided retrieval until consensus [Wu et al., 2026], while recent evaluations show that reliability
degrades under multi-turn interaction [Manczak et al., 2025] and that reasoning-oriented training
can make abstentionworse[Kirichenko et al., 2025]. This literature uses deliberation to improve
accuracy and leaves the intermediate states unused once the answer is produced. To our knowledge,
no prior work uses the deliberation trace as an abstention signal.
Conformal guarantees for language models.Conformal risk control and Learn-then-Test provide
distribution-free selective guarantees [Angelopoulos et al., 2024, 2022], instantiated for LLMs in
sampling-based generation [Quach et al., 2024], factuality filtering [Mohri and Hashimoto, 2024],
retrieval-augmented QA [Li et al., 2024a], answer-set prediction under outliers [Wang et al., 2025],
selective trust of generated reports [Gui et al., 2024], and consistency-scored abstention [Yadkori
et al., 2024]; in clinical QA, behavioural features of single reasoning traces have been calibrated
with conformal sets [Testoni and Calixto, 2025]. In every case the calibrated score is a single-round
quantity and the calibration is global over the population. We apply the same procedure to scores
computed from the consensus process and calibrate within each process stratum, including the
unanimous stratum on which the scores used in prior work are constant.
3

3 Method
Problem Formulation.Let xbe a question posed to a multi-round consensus system. We seek an
abstention decision
A:x7→ {ANSWER,ABSTAIN},(1)
such that, among the questions the system chooses to answer, the error rate is certifiably bounded
by a user-specified level α. We build on a published multi-round agentic RAG system for medical
QA [Wu et al., 2026], used unmodified at a pinned commit. At each round tthe system samples
N=8 candidate answers together with free-text rationales. If the candidates disagree, it generates
conflict-guided queries, retrieves evidence from a medical corpus, injects the evidence into the
next round’s context, and re-answers, stopping at unanimity or after T=8 rounds. We add only
a logging layer and modify no decision logic. For every round it records the candidate answers
with their full rationales, the answer distribution, and all retrieval events (queries and document
identifiers). We write st(a)∈ {0, 1/N, . . . ,1} for the fraction of candidates voting option aat round
t,mt= arg max ast(a)for the modal answer, Tstop≤T for the stopping round, and Dtfor the
set of documents retrieved after round t. ProbeGuard derives its abstention decision from this log
through trajectory features of the consensus process (§3.1), evidence signals that remain informative
when the vote is unanimous (§3.2), and a stratified calibration that certifies a decision rule from these
scores (§3.3).
3.1 Consensus-Process Features
The trajectory recorded in the log carries information that is absent from the terminal vote. Two
single-round statistics bracket the trajectory. Terminal agreement sTstop(mTstop)equals 1for every
converged question because of the early-stopping rule.First-round agreement
a1=s 1(m1)(2)
is the strongest statistic available to single-round sampling methods. The trajectory between them
is summarized by thetrajectory-averaged agreement, which rewards early, stable consensus over
late or forced convergence, and by themajority-flip count, the number of rounds at which the
leading answer is overturned. Two further families capture who disagrees and what the retriever
is doing.Minority persistenceis the share of rounds in which the alternative answer that persists
longest retains a supporter; a single alternative that survives every round signals genuine ambiguity,
whereas rotating, short-lived minorities are sampling noise. Thedocument-repeat rateis the fraction
of newly retrieved documents already seen in earlier rounds; when it rises, the evidence space is
saturating and further rounds are unlikely to resolve the disagreement. The full registry comprises 34
features across five families (first-round and trajectory agreement, convergence statistics, minority
dynamics, rationale-level signals, and retrieval-side behavior), each with a definition fixed in advance
of evaluation; formal definitions are in Appendix B. A combined process score is fit by ℓ2-regularized
logistic regression over these features with question-level cross-validation, median imputation and
missingness indicators computed within training folds (Appendix L).
3.2 Evidence Signals for Unanimous Votes
When the first-round vote is unanimous ( a1= 1), every agreement-based feature in §3.1 is constant:
the trajectory has length one and no retrieval is triggered. These are also the questions on which a
shared training-data error produces a confident mistake. If the answer rests on knowledge, the eight
rationales cite the same mechanism. If it rests on a shared error, no common mechanism exists to
cite, and each sample constructs its own justification, so the rationales diverge.
Rationale semantic entropy (passive).We formalize the question of whether eight identical answers
share one justification asmutual entailment among the concluding claimsof the rationales, treating the
consensus as knowledge-backed when its rationales entail one another. This reduces our problem to
one with a well-studied estimator, semantic entropy [Kuhn et al., 2023, Farquhar et al., 2024], which
clusters sampledanswersby bidirectional NLI entailment and computes the entropy of the cluster
distribution. On a unanimous vote its answer-level value is identically zero, so we apply the estimator
to the rationales instead, conditioned on answer agreement. From each of the Nrationales we extract
the concluding claim by a deterministic rule frozen before any rationale data were inspected (rule and
unit tests in Appendix C). We score concluding claims instead of full rationales because full rationales
4

exceed the NLI context length and share enough surface vocabulary that topical similarity masks
logical disagreement. Each ordered claim pair is then scored by an NLI model, yielding entailment
probabilities eij. Hard entailment clustering saturates at lnN on the most suspicious questions, so
we define the score on the continuous probabilities directly,
RSE = mean i<jmin(e ij, eji),(3)
where the minimum requires each claim pair to support each other in both directions, and the mean
runs over all N
2
unordered pairs, so that the score rises with the mutual support among the rationales.
Counter-evidence probe (active).The probe reuses the substrate’s retrieval loop with the direction
reversed. On contested questions the substrate retrieves evidence to resolve a disagreement; the
probe retrieves evidence against an existing agreement and checks whether the agreement survives.
If the vote rests on knowledge, the model can state why the challenge does not apply and the vote
should survive the strongest available counter-evidence; if it rests on a shared misconception, no
mechanism is available to rebut the challenge and we expect the vote to be revised. Concretely, for
each unanimous question we (i) prompt the model to generate retrieval queries that would support
alternativeanswers or contradict the consensus, (ii) retrieve challenge documents with the system’s
own retrieval stack under its standard parameters, and (iii) let all Ncandidates re-answer with the
challenge documents in context, under an explicit devil’s-advocate instruction that licenses revision
without demanding it (prompts in Appendix A; Algorithm 1 in Appendix D). The probe statistic is
thekeep rate
κ=1
N{i:candidateiretains the consensus answer},(4)
where the count runs over the Nre-answers. The unanimous-layer score combines the passive and
active signals as z(κ) +z(RSE) , where z(·)denotes per-dataset standardization. The probe fires only
on the unanimous subset and costs one query-generation call, one retrieval round, and Nre-answer
calls per probed question; per-question latency and token accounting are reported in Appendix J.
3.3 Stratified Certification
Deployment needs a threshold whose selective risk is bounded with stated confidence, which a point
estimate on held-out data does not provide. The unanimous and contested subpopulations differ
in base error rate, in which signals are defined, and in score scale, so a single global calibration is
dominated by this heterogeneity: thresholds that are valid on average can be badly miscalibrated
within each subpopulation. Selective risk is also not monotone in the threshold, since admitting
more questions canlowerthe error rate among answered questions, so naive quantile calibration
over-rejects or over-admits depending on where the errors sit. We therefore certify per stratum,
selecting each threshold with a multiple-testing procedure over a grid of candidates.
The target guarantee is
Pr
risk(ˆλ)≤α
≥1−δ,(5)
where risk(λ) is the error rate among the questions a stratum answers at threshold λ, those scoring
strictly above λ, and the probability is over the calibration draw. The stratification gwas fixed
before any calibration data were seen: S1(unanimous first round), scored by z(κ) +z(RSE) ;S2
(contested but converged), scored by the trajectory-averaged agreement; and S3(never converged),
which abstains as a whole. Within each certified stratum we select the threshold by Learn-then-Test
[Angelopoulos et al., 2022]: each candidate coverage level con a grid places a threshold λcat the top-
cquantile of the calibration scores, and its selective risk is tested against the null H0: risk(λ c)> α
with the exact binomial tail
pc= Pr
Bin(n c, α)≤E c
,(6)
where ncis the number of calibration questions scoring strictly above λcandEcthe number of errors
among them; the grid is tested at Bonferroni-corrected level δ/|C|(and across certified strata), and ˆλis
the threshold of the largest coverage whose null is rejected (Algorithm 2, Appendix D).
Assumption 1.Within each stratum, calibration and deployment questions are drawn i.i.d. from the
same distribution.
Theorem 1.Under Assumption 1, the threshold ˆλreturned by Algorithm 2 satisfies the guarantee of
Eq.(5)for every certified stratum. (Proof in Appendix E.)
5

1 2 3 4
Deliberation round0.800.850.900.95Mean agreement rate
Correct consensus
Wrong consensusAgreement over rounds
1 2 3 4 5
Stopping round0200400600Questions concludedOutcome by stopping round
0.5 0.6 0.7
AUROCVote-basedRationale entropyKeep rateCombined
constant0.5870.6420.685Unanimous layer
0204060
13264558
52
overallError rate (%)Figure 2: (Left) Correct consensus starts near-unanimous, while wrong consensus starts from
disagreement and converges to it slowly (MedQA, mean agreement per round, shaded standard
errors). (Middle) Error rate rises with the round at which consensus concludes (bars: questions
concluding per round; line: error rate at that round; dashed: overall). (Right) On the unanimous layer
every vote-based signal is constant, and only the formation signals discriminate.
The guarantee does not depend on the quality of the score; a weak score lowers certified coverage but
not the validity of the bound. Because the scores entering certification ( RSE ,κ,¯a) are training-free,
the full calibration split is available for threshold selection, and we report certified coverage and
realized risk as means over repeated calibration/test splits rather than a single draw.
4 Experiments and Results
4.1 Experimental Settings
Datasets.We evaluate on MedQA [Jin et al., 2020], the medicine subset of MMLU-Pro [Wang et al.,
2024], and MedMCQA [Pal et al., 2022], in the form distributed with the substrate. MedXpertQA
[Zuo et al., 2025] is a hard-frontier reference on which substrate accuracy approaches the guessing
floor. Specialty labels on MedQA and MedMCQA support Appendix I.
Substrate and backbones.All experiments instrument the entropy-guided variant of the substrate at
a pinned commit, with its retrieval stack unchanged (BM25 with MedCPT cross-encoder reranking
[Jin et al., 2023] over MedCorp [Xiong et al., 2024a]). Final answers are scored by the substrate’s own
rule (strict majority, ties as errors), and a logging layer records every round verbatim. We evaluate
Qwen3-8B [Yang et al., 2025] as the primary backbone, Llama-3.1-8B-Instruct [Grattafiori et al.,
2024], and the medical reasoning model HuatuoGPT-o1-8B [Chen et al., 2024], all under identical
decoding configuration and seed (Appendix L).
Baselines.We compare against the abstention signals available to a deployed consensus system:
terminal agreement, first-round agreement [Wang et al., 2023], mean sequence log-probability,
verbalized confidence [Tian et al., 2023, Xiong et al., 2024b], and P(True) [Kadavath et al., 2022].
On multiple choice, semantic entropy [Kuhn et al., 2023, Farquhar et al., 2024] reduces to answer-
distribution entropy, which we report as its representative. As the certification control, first-round
agreement is calibrated with the same stratified machinery, following the single-round consistency-
plus-conformal recipe [Yadkori et al., 2024].
Metrics and certification.Discrimination is reported as AUROC, selective prediction as risk–
coverage curves, and probe behavior as flip and keep rates (Eq. (4)); flip rates carry Wilson intervals
and certification numbers carry standard errors over repeated splits. The combined process score is
fit by logistic regression under question-level cross-validation with fold-internal preprocessing, and
rationale claim pairs are scored by DeBERTa-large fine-tuned on MNLI [He et al., 2021] (Appendix L).
Certification follows the pre-registered stratified protocol of §3.3, calibrated independently per domain
on an even calibration/test split, sweepingα∈ {0.05,0.10,0.15,0.20}.
4.2 Discrimination on the Full Question Stream
Figure 2 shows that wrong and correct consensus form differently. Correct consensus starts near-
unanimous; wrong consensus starts from disagreement, and the error rate rises from 13.4% at instant
consensus to 58.2% by round four. Table 2 reports how well each signal separates correct from
incorrect final answers. Terminal agreement, the deployed default, is at chance on every benchmark.
This follows from the early-stopping rule (§3.1). The process score, computed from the same runs,
6

Table 2: Discrimination (AUROC, higher is better) and selective risk (AURC, lower is better) of
abstention signals on the Qwen3-8B substrate. Best value per column in bold. Verbalized confidence
and P(True) were elicited on the two primary benchmarks.
AUROC↑AURC↓
Signal MedQA MMLU-Pro MedMCQA MedQA MMLU-Pro MedMCQA
Terminal agreement 0.512 0.504 0.507 0.221 0.300 0.330
First-round agreement [Wang et al., 2023] 0.677 0.698 0.640 0.155 0.200 0.250
Sequence log-probability 0.400 0.474 0.534 0.320 0.376 0.347
Answer entropy [Kuhn et al., 2023] 0.681 0.703 0.641 0.1540.1990.250
Verbalized confidence [Tian et al., 2023] 0.585 0.651 – 0.179 0.207 –
P(True) [Kadavath et al., 2022] 0.602 0.644 – 0.160 0.205 –
ProbeGuard (process score)0.696 0.704 0.659 0.1450.2050.234
Table 3: Process score versus terminal agreement across backbones. ProbeGuard separates correct
from incorrect consensus at 0.696–0.825 AUROC where terminal agreement is chance, and has the
lower selective risk in every cell.
MedQA MMLU-Pro med
AUROC↑AURC↓AUROC↑AURC↓
Backbone ProbeGuard Term. ProbeGuard Term. ProbeGuard Term. ProbeGuard Term.
Qwen3-8B0.6960.5120.1450.2210.7040.5040.2050.300
HuatuoGPT-o1-8B0.7750.5190.0960.1930.7010.5260.2250.353
Llama-3.1-8B0.8250.5480.1420.2890.7820.5510.2460.406
raises discrimination to 0.696 on MedQA, 0.704 on MMLU-Pro medicine, and 0.659 on MedMCQA,
and yields the lowest selective risk on MedQA and MedMCQA (risk-coverage curves in Appendix F).
The remaining single-round signals fall in a narrow band, which the process score exceeds on every
benchmark. This aggregate comparison averages over the unanimous layer, which §4.3 examines
separately.
The other backbones behave alike (Table 3): the largest gap is on Llama-3.1-8B, where the process
score reaches 0.825 AUROC against 0.548 for terminal agreement. HuatuoGPT-o1-8B preserves
the ordering, and the process score has the lower AURC in all six cells, less than half of terminal
agreement’s on Llama/MedQA.
4.3 The Unanimous Layer
Questions on which all eight candidates agree in the first round are the ones where a wrong answer
is hardest to catch. This layer covers 59%, 59%, and 55% of MedQA, MMLU-Pro medicine, and
MedMCQA, where 13.4%, 17.7%, and 22.3% of unanimous answers are wrong. The error rate
grows with difficulty; on expert-frontier MedXpertQA, where 29% of questions still reach first-round
unanimity, 79.0% of those unanimous votes are wrong.
Rationale semantic entropy, computed from the existing rationales, separates correct from incorrect
unanimity at 0.587 AUROC on MedQA and 0.613 on MMLU-Pro medicine, on a subpopulation
where the answers themselves are identical.
The counter-evidence probe separates the two more sharply. Figure 3 shows each unanimous
consensus re-asked with retrieved counter-evidence in context: wrong consensus flips at 14.0–26.5%
across the three benchmarks, while correct consensus stays at 3.3–5.5%. The per-question keep rate
scores this asymmetry (0.642, 0.638, and 0.729 AUROC on the three benchmarks), and combining it
with rationale entropy lifts the unanimous layer to 0.685 and 0.688 on the two primary benchmarks.
Sequence log-probability, at chance or inverted on the full stream (Table 2), recovers to 0.680 inside
the layer, while the self-report baselines stay behind (P(True) 0.602, verbalized confidence 0.599).
Adding it to the keep-and-entropy pair lifts the layer to 0.730 AUROC, so the likelihood and process
signals are complementary.
Control arms on MedQA attribute the flips to the retrieved content (Appendix G). The devil’s-advocate
instruction alone flips none of the 746 consensus votes, irrelevant documents flip 1.0% of wrong
7

MedQA
Qwen3-8BMMLU-Pro
Qwen3-8BMedMCQA
Qwen3-8BMedQA
Huatuo-o1MMLU-Pro
Huatuo-o1MedQA
Llama-3.1MMLU-Pro
Llama-3.1020406080Flip rate (%)
 14.016.326.5 15.415.354.2
45.5
4.2 3.3 5.5 3.74.628.323.8Wrong consensus
Correct consensusFigure 3: Flip rates of wrong and correct consensus under counter-evidence. A consensus counts as
flipped when the majority vote of the N=8 re-answers no longer lands on the original answer. Flip
rates per backbone and benchmark with Wilson 95% confidence intervals. The asymmetry holds in
every cell, and the per-question keep rate discriminates on all seven (AUROC 0.638–0.729).
consensus, and the counter-evidence itself under a neutral instruction retains nearly the full separation
(12.0% versus 3.6%).
The asymmetry recurs on the other backbones. On HuatuoGPT-o1-8B, whose MedQA unanimous
layer errs at only 6.2%, wrong consensus still flips at 15.4% against 3.7% for correct. Llama-3.1-8B
revises far more readily and flips even correct consensus at up to 28.3%, but wrong consensus flips at
54.2%. Across all seven cells of Figure 3 the keep-rate AUROC stays within 0.638–0.729.
4.4 Certification Results
We now calibrate these scores to the guarantee of Eq. (5). Without stratification, naive quantile
calibration on the full question stream misses its target (realized test risk 15.7% at α=0.10 on
MedQA), and Learn-then-Test applied globally is valid but returns no certified threshold at any αwe
sweep. Both fail for the heterogeneity described in §3.3: no single threshold on the pooled stream is
both valid and non-trivial.
With per-stratum certification, on the MedQA unanimous layer the combined keep-and-entropy
score certifies 0.553 ±0.038 of questions at α=0.15 over 50 splits, with a realized risk on answered
questions of 9.0%, well below the bound. Every vote-based signal of Table 2 is constant on this
stratum, so none of them can certify non-zero coverage here. Multiplied by the layer’s share of
the benchmark, this certified region amounts to about 32% of all MedQA questions. Enlarging the
calibration side with in-domain unanimous questions ( n=875 ) lifts certified coverage at α=0.15 to
0.892±0.004 and certifies α=0.10 for the first time (0.219 ±0.023, realized risk 6.5%), with the test
half untouched. On the other backbones, without augmentation, HuatuoGPT-o1-8B certifies 0.410 of
its unanimous layer at α=0.10 and 0.997 at α=0.15 , and Llama-3.1-8B 0.838 at α=0.15 (Figure 4,
Appendix E).
Calibration is per domain: pooling the calibration sets of MedQA and MMLU-Pro medicine yields an
empty certified region at any level, because the two unanimous layers differ in base error rate (§4.3).
Per-domain calibration is also what Assumption 1 requires.
5 Analysis and Discussion
5.1 Ablation Studies
Only the rationale family is non-redundant: dropping it lowers MedQA AUROC from 0.696 to 0.682.
The trajectory families are redundant with one another; each alone supports an AUROC close to
the full score, and each tracks difficulty proxies (retrieval volume, stopping round) at |ρ| ≥0.93 .
Rationale entropy is the exception on both counts ( |ρ| ≤0.39 ): it is the only family whose removal
lowers AUROC and the only one that does not track difficulty (full drop-one matrices in Appendix H).
Inside the unanimous layer, the two evidence signals are complementary: their combination outper-
forms either alone (§4.3). Replacing hard entailment clustering with the continuous score of Eq. (3)
raises the unanimous-layer AUROC from 0.578 to 0.587 (MedQA) and 0.593 to 0.613 (MMLU-Pro
medicine).
8

First-round agreement isolates the effect of aggregating votes without any process information; inside
the unanimous layer, which carries 100 wrong answers on MedQA alone, it is constant, and the
separation reported in §4.3 comes entirely from the process signals.
5.2 Cost and Sensitivity
The probe fires only on unanimous first-round votes, at a median of 114 s and 4381 completion tokens
on MedQA, against 72 s for generation itself (full accounting in Appendix J); all passive features
come from logs the system already writes. Excluding the probe-pilot questions from evaluation shifts
the combined AUROC by 0.007, and gradient-boosted trees in place of logistic regression change no
conclusion (Appendix H).
5.3 Discussion and Limitations
A hospital deploying a consensus QA system would calibrate ProbeGuard on its own case mix and
refer every question outside the certified region to a clinician. Each referred question carries its
challenge documents and the record of dissent, so the clinician sees the evidence that led the system
to withhold an answer. The signals are defined for any system that deliberates in rounds and require
nothing beyond its execution log. Tighter certificates require more calibration data, and in-domain
unanimous questions accumulate as the system runs (§4.4). The guarantee applies in any stratum
whose base error rate is at or below the targetα; a stratum above it receives no certified region.
The evaluation is limited to multiple-choice benchmarks, so whether the flip asymmetry and the
rationale signals carry over to free-text clinical questions is untested. The three backbones are
8B-scale models on a single published substrate; the signal definitions transfer to other deliberating
systems, but we have not measured them elsewhere. The certificates hold under Assumption 1, so a
deployment must calibrate on its own case mix and recalibrate when that mix drifts. And we have not
evaluated whether clinicians find the returned dissent records useful; a prospective evaluation with
clinicians is the next step.
6 Conclusion
We presented ProbeGuard, a certified abstention framework for multi-round consensus medical QA.
Its abstention decision is computed from the record of how the agreement was reached: process
features from the execution log, rationale entropy and a counter-evidence probe for unanimous votes,
and a stratified Learn-then-Test calibration of the resulting scores at a user-specified selective-risk
level. Across three backbones and three benchmarks, terminal agreement is at chance, the process
signals separate correct from incorrect consensus on the full stream and inside the unanimous layer,
and stratified calibration certifies answering on part of that layer at the target risk.
References
Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua
Lei. Learn then test: Calibrating predictive algorithms to achieve risk control, 2022. URL
https://arxiv.org/abs/2110.01052.
Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal
risk control, 2024. URLhttps://arxiv.org/abs/2208.02814.
Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou,
and Benyou Wang. Huatuogpt-o1, towards medical complex reasoning with llms, 2024. URL
https://arxiv.org/abs/2412.18925.
Xilin Dang, Kexin Chen, Xiaorui Su, Ayush Noori, Iñaki Arango, Lucas Vittor, Xinyi Long, Yuyang
Du, Marinka Zitnik, and Pheng Ann Heng. Knowguard: Knowledge-driven abstention for multi-
round clinical reasoning, 2025. URLhttps://arxiv.org/abs/2509.24816.
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large
language models using semantic entropy.Nature, 630(8017):625–630, 2024. ISSN 1476-4687. doi:
10.1038/s41586-024-07421-0. URLhttp://dx.doi.org/10.1038/s41586-024-07421-0.
9

Aaron Grattafiori et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.
21783.
Yu Gui, Ying Jin, and Zhimei Ren. Conformal alignment: Knowing when to trust foundation models
with guarantees, 2024. URLhttps://arxiv.org/abs/2405.10301.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert
with disentangled attention, 2021. URLhttps://arxiv.org/abs/2006.03654.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What
disease does this patient have? a large-scale open domain question answering dataset from medical
exams, 2020. URLhttps://arxiv.org/abs/2009.13081.
Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, W. John Wilbur, and Zhiyong
Lu. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot
biomedical information retrieval, 2023. URLhttps://arxiv.org/abs/2307.00589.
Saurav Kadavath et al. Language models (mostly) know what they know, 2022. URL https:
//arxiv.org/abs/2207.05221.
Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. Abstentionbench:
Reasoning llms fail on unanswerable questions, 2025. URL https://arxiv.org/abs/2506.
09038.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language generation, 2023. URL https://arxiv.org/abs/
2302.09664.
Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. Traq: Trustworthy retrieval augmented
question answering via conformal prediction, 2024a. URL https://arxiv.org/abs/2307.
04642.
Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S. Ilgen, Emma Pierson, Pang Wei
Koh, and Yulia Tsvetkov. Mediq: Question-asking llms and a benchmark for reliable interactive
clinical reasoning, 2024b. URLhttps://arxiv.org/abs/2406.00922.
Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, and Hua Wei. Uncertainty
quantification and confidence calibration in large language models: A survey. InProceedings of
the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V .2, KDD ’25,
page 6107–6117. ACM, 2025. doi: 10.1145/3711896.3736569. URL http://dx.doi.org/10.
1145/3711896.3736569.
Sravanthi Machcha, Sushrita Yerra, Sahil Gupta, Aishwarya Sahoo, Sharmin Sultana, Hong Yu, and
Zonghai Yao. Knowing when to abstain: Medical llms under clinical uncertainty, 2026. URL
https://arxiv.org/abs/2601.12471.
Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box
hallucination detection for generative large language models, 2023. URL https://arxiv.org/
abs/2303.08896.
Blazej Manczak, Eric Lin, Francisco Eiras, James O’Neill, and Vaikkunth Mugunthan. Shallow
robustness, deep vulnerabilities: Multi-turn evaluation of medical llms, 2025. URL https:
//arxiv.org/abs/2510.12255.
Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees,
2024. URLhttps://arxiv.org/abs/2402.10978.
Thang Nguyen, Peter Chin, and Yu-Wing Tai. Ma-rag: Multi-agent retrieval-augmented generation via
collaborative chain-of-thought reasoning, 2025. URLhttps://arxiv.org/abs/2505.20096.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa : A large-scale
multi-subject multi-choice dataset for medical domain question answering, 2022. URL https:
//arxiv.org/abs/2203.14371.
10

Ihsan Ayyub Qazi, Ayesha Ali, Asad Ullah Khawaja, Muhammad Junaid Akhtar, Ali Zafar Sheikh, and
Muhammad Hamad Alizai. Automation bias in large language model–assisted diagnostic reasoning
among physicians trained in ai literacy — a randomized clinical trial.NEJM AI, 3(5), 2026. ISSN
2836-9386. doi: 10.1056/aioa2501001. URLhttp://dx.doi.org/10.1056/AIoa2501001.
Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, and Regina
Barzilay. Conformal language modeling, 2024. URLhttps://arxiv.org/abs/2306.10193.
Ravi Shankar, Sheng Wong, Lin Li, Magdalena Bachmann, Alex Silverthorne, Beth Albert, and
Gabriel Davis Jones. Energy landscapes enable reliable abstention in retrieval-augmented large
language models for healthcare, 2025. URLhttps://arxiv.org/abs/2509.04482.
Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar. A survey
on uncertainty quantification of large language models: Taxonomy, open research challenges,
and future directions.ACM Computing Surveys, 58(3):1–38, 2025. ISSN 1557-7341. doi:
10.1145/3744238. URLhttp://dx.doi.org/10.1145/3744238.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne,
Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip
Mansfield, Dina Demner-Fushman, Blaise Agüera y Arcas, Dale Webster, Greg S. Corrado,
Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle
Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language
models encode clinical knowledge.Nature, 620(7972):172–180, 2023. ISSN 1476-4687. doi:
10.1038/s41586-023-06291-2. URLhttp://dx.doi.org/10.1038/s41586-023-06291-2.
Alberto Testoni and Iacer Calixto. Mind the gap: Benchmarking llm uncertainty and calibration
with specialty-aware clinical qa and reasoning-based behavioural features, 2025. URL https:
//arxiv.org/abs/2506.10769.
Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea
Finn, and Christopher D. Manning. Just ask for calibration: Strategies for eliciting calibrated
confidence scores from language models fine-tuned with human feedback, 2023. URL https:
//arxiv.org/abs/2305.14975.
Xiaoyang Wang and Christopher C. Yang. The consistency illusion: How multi-agent debate hides
reasoning misalignment, 2026. URLhttps://arxiv.org/abs/2606.08457.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models,
2023. URLhttps://arxiv.org/abs/2203.11171.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi
Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language
understanding benchmark, 2024. URLhttps://arxiv.org/abs/2406.01574.
Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, and
Kaidi Xu. Sconu: Selective conformal uncertainty in large language models, 2025. URL https:
//arxiv.org/abs/2504.14154.
Wenhao Wu, Zhentao Tang, Yafu Li, Shixiong Kai, Mingxuan Yuan, Zhenhong Sun, Chunlin Chen,
and Zhi Wang. From conflict to consensus: Boosting medical reasoning via multi-round agentic
rag, 2026. URLhttps://arxiv.org/abs/2603.03292.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented
generation for medicine, 2024a. URLhttps://arxiv.org/abs/2402.13178.
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms
express their uncertainty? an empirical evaluation of confidence elicitation in llms, 2024b. URL
https://arxiv.org/abs/2306.13063.
11

Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, András György, Adam Fisch, Arnaud Doucet,
Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvári, Ali Taylan Cemgil, and
Nenad Tomasev. Mitigating llm hallucinations via conformal abstention, 2024. URL https:
//arxiv.org/abs/2405.01563.
An Yang et al. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding,
and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding,
2025. URLhttps://arxiv.org/abs/2501.18362.
12

A Prompts
This appendix reproduces every prompt in the pipeline verbatim. The first group belongs to the
substrate and is unchanged from the pinned commit; we list it for self-containment, and because two
of its design choices matter for interpreting our results. The second group is ours: the two probe
templates and the two self-report elicitation prompts. Template variables appear in double braces.
Substrate prompts.Candidate answering uses a plain system instruction with a tagged-answer
format; the same system prompt is reused by later rounds and by our probe re-answering, so no probe
effect can be attributed to a change of role or format.
Candidate answering (system, all rounds)
You are a medical assistant, please answer my medical questions. Give your final
choice (capital option) closed in tag ‘<answer>X</answer>‘ after your analysis.
Your response should be as detailed as possible, but please do not use any
subheadings.
Candidate answering (user, round 1)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
Please analyze the question and give your answer. Give your analysis and final
choice.
Two features of the substrate’s later-round prompt bear directly on §4.2. First, candidates are shown
the previous answers sorted by token-entropy confidence, together with an explicit caveat that the
ranking may be wrong. Second, the prompt closes by instructing the model to answertowards lower
entropy: convergence is not only an emergent behavior of the panel but an instructed objective. A
system whose members are told to converge will converge on wrong answers as readily as on right
ones, which is one more reason the terminal agreement it produces carries no information about
correctness.
Evidence-conditioned re-answering (user, rounds≥2)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
### Documents
{{documents}}
–-
I will provide several assistant’s previous answers, which may be incorrect.
The previous answers are sorted by their confidence (low entropy means high
confidence), but the entropy metric cannot accurately reflect whether the answer
is correct or not.
Please analyze the previous answers and then re-answer.
{{answers}}
–-
The above are the question along with the assistant’s previous answers, and
some relevant documents. Please analyze and then re-answer. Please give your
response towards **lower entropy**. Give your analysis and final answer.
When the panel disagrees, the substrate generates retrieval queries from the disagreement itself. The
query prompt asks for thedispute pointsamong the answers, which is precisely the structure our
counter-query prompt inverts.
13

Conflict-guided query generation (system)
### Role & Goal:
You are a medical expert. I will provide multiple different answers to the
same question, which may be incorrect. Your task is to identify contradictions,
ambiguities, and core dispute points among the answers, and extract key concept
queries for further retrieval and verification.
### Processing Steps:
1. Analyze different answers and summarize the differences between them.
2. Extract keywords from these differences for retrieval.
3. Generate 1-4 precise queries for further search using **BM25** retriever.
### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries)
Probe prompts.The counter-query prompt keeps the substrate’s query-generation structure (role
header, numbered steps implied by the format, BM25-targeted output) and changes only the direction
of search: instead of resolving a disagreement, it retrieves evidence against an agreement. Keeping
the format aligned means the probe exercises the same retrieval machinery under the same interface,
so flip behavior reflects the consensus, not a novel pipeline.
Counter-evidence query generation (system; thinking mode, temperature 0)
### Role & Goal:
You are a skeptical medical expert. All assistants unanimously chose the
same answer to a question. Your task is to actively search for evidence that
could CHALLENGE or CONTRADICT that consensus: evidence supporting plausible
alternative options, known exceptions, contraindications, or updated guidelines.
### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries, 2-4 total, for a **BM25** retriever)
Counter-evidence query generation (user)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
–-
All assistants unanimously answered: {{consensus}}.
Generate 2-4 precise BM25 queries to retrieve evidence that could challenge this
consensus (e.g., evidence favoring other options or against {{consensus}}). Give
your queries in the given format.
The re-answering prompt needs the most care. A naive challenge instruction produces sycophantic
flipping: models abandon any answer once an authority questions it, which would erase the very
asymmetry the probe measures. The wording below therefore allows but does not require a change of
answer; the key sentence isdo not change your answer unless the evidence genuinely warrants it.
Under this instruction correct consensus holds at 3.3–5.5% while wrong consensus flips at 14.0–26.5%
(Figure 3), and the control arms of Appendix G confirm the division of labor directly: the instruction
alone flips nothing, and the counter-evidence carries the asymmetry.
Devil’s-advocate re-answering (user; system prompt identical to candidate answering)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
14

### Challenge Documents
A skeptical reviewer retrieved the following documents that may challenge the
previous consensus:
{{documents}}
–-
Previously, all assistants unanimously answered {{consensus}}. A devil’s
advocate now asks: in light of the challenge documents above, critically
re-examine whether {{consensus}} is really correct, or whether another option
is better supported. Weigh the challenge evidence seriously, but do not change
your answer unless the evidence genuinely warrants it. Give your analysis and
final choice.
Self-report elicitation prompts.Both baselines are elicited post hoc on the system’s final answer,
one call per question at temperature 0. Verbalized confidence follows the 0–100 elicitation of Tian
et al. [2023], Xiong et al. [2024b]; P(True) follows Kadavath et al. [2022], with the score read from
the first generated token’s top-20 log-probability mass overTrue/Falsevariants rather than from the
sampled text.
Verbalized confidence (system + user)
You are a careful medical assistant. Rate how confident you are that a
given answer to a multiple-choice question is correct. Reply with a single
integer from 0 to 100 inside <conf></conf> tags, then one short sentence of
justification.
### Question
{{question}}
### Options
{{options}}
### Proposed answer: {{answer}}
How confident are you (0-100) that this answer is correct? Reply with
<conf>N</conf> first.
P(True) (system + user)
You are a careful medical assistant. Judge whether a given answer to a
multiple-choice question is correct. Reply with exactly one word: True or
False.
### Question
{{question}}
### Options
{{options}}
### Proposed answer: {{answer}}
Is the proposed answer correct? Reply with exactly one word: True or False.
B Feature Definitions
The process score is fit over five feature families, each built from a small set of base quantities and
their standard temporal aggregations (value at the first and last round, mean, slope, and maximum
over rounds). Table 4 summarizes what each family measures; the representative statistics introduced
in §3.1 are thetrajectory-averaged agreement
¯a=1
TstopPTstop
t=1st(mt),(7)
themajority-flip count
nflip={t < T stop:mt+1̸=m t},(8)
minority persistence
π= max
a̸=m∗1
Tstop{t:s t(a)≥ 1/N},(9)
where m∗is the final answer and the count is the number of rounds in which option aretains at least
one supporter, and thedocument-repeat rate
ρt=Dt∩S
t′<tDt′
|Dt|,(10)
15

with first-round agreement a1as in Eq. (2)and the rationale representative as in Eq. (3). The
answer-distribution family is built on the per-roundanswer entropy
Ht=−P
ast(a) lns t(a),(11)
aggregated as its first- and last-round values, its slope, and the fraction of rounds in which it does not
increase, together with the Jensen–Shannon divergence between the answer distributions of adjacent
rounds.
Table 4: Feature families of the process score.
Family Measures Representative
Agreement dynamics how fast and how stably the vote converges Eqs. (2), (7), (8)
Answer-distribution dynamics entropy and drift of the candidate answer distribution Eq. (11)
Minority dynamics whether dissent is persistent or transient Eq. (9)
Rationale signals mutual entailment among the reasons behind the votes Eq. (3)
Retrieval dynamics whether deliberation still finds new evidence Eq. (10)
Every definition, the probe protocol, and the stratification were frozen in a versioned registry before
the corresponding data were collected; the machine-readable registry with exact formulas for all
features ships with the code.
C Claim Extraction Rule
Rationale semantic entropy scores mutual entailment among theconcluding claimsof the panel’s
rationales, not among the rationales themselves. This appendix documents why that restriction is
necessary and gives the exact extraction rule.
Scoring full rationales fails in two ways. Rationales run to hundreds of tokens, well past the effective
input of the entailment model, so truncation decides what gets compared. More fundamentally,
two rationales about the same clinical question share most of their surface vocabulary whether
or not they agree: both walk the same options, cite the same anatomy, and eliminate the same
distractors. Entailment scores over full rationales therefore saturate on topical similarity, and the
logical disagreement, which lives almost entirely in the final verdict sentences, is drowned out.
Restricting the comparison to concluding claims is what makes the NLI calls well-posed.
Prior work that decomposes generations into checkable units typically prompts a strong LLM to do
the splitting [Mohri and Hashimoto, 2024]. We use a deterministic rule instead, for three reasons.
A fixed rule can be frozen before any rationale data are inspected, which an LLM extractor cannot
guarantee; it adds zero inference cost to a feature set whose selling point is being free; and it avoids a
circularity, since asking a model to summarize its own reasoning gives it a second chance to smooth
over exactly the incoherence the feature is trying to measure.
The rule itself has four steps and was fixed before any rationale data were inspected. The answer tag
is stripped and the remaining text is split into sentences. Within the last five sentences, every sentence
that contains a conclusion cue is kept, up to three of them and in their original order; the cue list has
nine entries:answer,option,choice,correct,therefore,thus,hence,most likely, andbest. When no
sentence carries a cue, the final sentence serves as the claim. The implementation and its unit tests
ship with the code.
Worked example (MedQA, dimeric-immunoglobulin question, candidate 2)
Rationale (final portion of 1,367 characters):“. . . Option C, ‘Inhibiting bacterial adherance and colo-
nization of mucous membranes,’ is the most accurate description of the primary function of dimeric IgA.
<answer>C</answer>”
Extracted claim:“Option C, ‘Inhibiting bacterial adherance and colonization of mucous membranes,’ is the
most accurate description of the primary function of dimeric IgA.”
The rule is deliberately permissive: when several tail sentences carry cues, all of them survive,
including option-elimination lines. This costs precision on individual extractions but keeps the rule
simple, auditable, and free of tuned thresholds, and the resulting score separates knowledge-backed
from inherited unanimity at 0.587 and 0.613 AUROC on the two primary benchmarks (§4.3).
16

D Algorithms
Algorithm 1 gives the probe of §3.2; each line is one call into machinery the substrate already runs,
with the prompts of Appendix A. Algorithm 2 gives the calibration of §3.3; following Quach et al.
[2024], the statistical content stays in Eqs. (5)and(6), and the box only fixes the order of operations.
Algorithm 1Counter-evidence probe
Require: question xwith unanimous first-round consensus m∗(a1= 1); panel size N; retriever R
of the substrate; query budgetk≤4
Ensure:keep rateκ∈ {0, 1/N, . . . ,1}
1:q1, . . . , q j←COUNTERQUERIES(x, m∗),j≤k ▷temperature 0, thinking mode
2:D ch← R(q 1, . . . , q j)▷same BM25 + rerank stack,≤8documents
3:fori= 1toNdo
4:˜a i←REANSWER(x, D ch, m∗)▷devil’s-advocate instruction; revision licensed, not
demanded
5:end for
6:κ←{i: ˜a i=m∗}/ N
7:returnκ ▷lowκ: the consensus collapses under challenge
Algorithm 2Stratified Learn-then-Test calibration
Require: calibration set Dcal; stratification gwith certified strata {S1, S2}; per-stratum scores sS;
risk levelα; error budgetδ; coverage gridC
Ensure:thresholds ˆλS, orABSTAINfor strata certifying nothing
1:foreach certified stratumSdo
2:D S← {(x, y)∈ D cal:g(x) =S}
3:forc∈ Cin descending orderdo
4:λ c←score at the top-cquantile ofD S
5:n c←{x∈ D S:sS(x)> λ c};E c←errors among them
6:p c←Pr
Bin(n c, α)≤E c
▷Eq. (6)
7:ifp c≤δ /(|C| · |{S 1, S2}|)then
8: ˆλS←λ c;break▷largest certified coverage forS
9:end if
10:end for
11:ifno rejectionthen ˆλS←ABSTAIN▷stratum uncertified at levelα
12:end for
13:return{ ˆλS}▷answerx∈Siffs S(x)> ˆλS;S3abstains as a whole
E Certification Protocol and Proof
Protocol.The coverage grid runs from 0.05 to1.00 in steps of 0.05, giving twenty candidate
thresholds per stratum. The family-wise error budget is δ= 0.05 , Bonferroni-corrected across the
twenty grid points and the two certified strata, and we sweep α∈ {0.05,0.10,0.15,0.20} , each level
constituting its own guarantee. Calibration and test halves are split evenly at random, and every
certification number in the paper is the mean over 50 such splits with fixed seeds: a single split
yields a valid certificate, but its reported coverage inherits the variance of one draw, and averaging
over repeated splits reports the procedure rather than one particular partition. The per-split violation
fraction is the quantity the 1−δ guarantee bounds, and it is 0 of 50 for every Learn-then-Test arm
reported: no split’s realized test risk exceeds its target α. Calibration is per domain throughout; §4.4
reports what happens when domains are pooled.
Why a grid rather than fixed-sequence testing.Selective risk is not monotone in the threshold,
so the fixed-sequence variant of Learn-then-Test, which walks thresholds from the most selective
downward and stops at the first failure, is brittle here: one wrong answer among the few highest-
scoring calibration questions terminates the walk at zero certified coverage even when broader
coverage levels are certifiable. Testing the whole grid under a Bonferroni correction is robust to the
location of individual errors and costs only a constant factor in the effective test level.
17

Proof of Theorem 1.Fix a certified stratum with calibration pairs (si, yi)i≤nof score and correct-
ness, and fix a grid point c. The threshold λcis an order statistic of the calibration scores, so the
argument conditions on it. Let A={i:s i> λc}be the calibration questions above the threshold,
so that nc=|A| . The event that λcequals a value tand that the set above it equals Aconstrains
the pairs in Aonly through si> t, and places all its remaining conditions on the pairs outside A.
Under Assumption 1 the pairs are independent, so conditional on (λc, A)the pairs in Aare i.i.d.
draws from the distribution of a question given s > λ c. This is the population that the rule answers at
threshold λc, and its error rate is risk(λ c). Conditional on (λc, A), the error count therefore satisfies
Ec∼Bin(n c,risk(λ c)). Whenever risk(λ c)> α , this count stochastically dominates Bin(n c, α),
and the tail probability of Eq. (6)obeys Pr[p c≤u|λ c, A]≤u for every u∈[0,1] . Taking the
expectation over (λc, A)gives Pr[ risk(λ c)> α, p c≤u]≤u , which makes pca valid p-value for
H(c)
0: risk(λ c)> α at the data-dependent threshold. Algorithm 2 rejects only when pc≤δ/(|C| ·2) ,
so by the union bound over all grid points and both certified strata, the probability that any true null
is rejected is at most δ. On the complementary event, of probability at least 1−δ , every rejected
null is false; in particular the selected ˆλ, which corresponds to a rejected null, satisfies risk(ˆλ)≤α .
The union-bound step is the Learn-then-Test argument of Angelopoulos et al. [2022] with the exact
binomial tail. The conditioning step extends it to thresholds placed at calibration quantiles, and it is
why the test counts the questions strictly above the threshold: the question that defines the threshold
is not a draw from the answered population.□
Scope of the guarantee.The guarantee is marginal over the calibration draw and holds per stratum
and per domain; it is distribution-free in the sense that no assumption is placed on the score. Training-
free scores are what make the full calibration split available for threshold selection: no data are spent
fitting a scorer.
0.050 0.075 0.100 0.125 0.150 0.175 0.200Target risk level α0.040.060.080.100.120.14Realized selective riskrisk =α Qwen3-8B
Qwen3-8B + in-domain cal.
HuatuoGPT-o1-8B
Llama-3.1-8B
Naive quantile
0.050 0.075 0.100 0.125 0.150 0.175 0.200Target risk level α0.00.20.40.60.81.0Certified coverage
Qwen3-8B
Qwen3-8B + in-domain cal.
HuatuoGPT-o1-8B
Llama-3.1-8B
Figure 4: Realized risk and certified coverage of the stratified rule. (Left) Realized selective risk
against the target level: every Learn-then-Test arm stays below the risk =α diagonal at every level it
certifies, while naive quantile calibration crosses it. (Right) Certified coverage per backbone on the
MedQA unanimous layer; printed values are the α=0.15 coverages, and stars mark the two operating
points of §4.4. Lines are means over 50 calibration/test splits; bands are±1 standard deviation.
F Full Per-Dataset Results
Table 5 extends the main results to every signal, benchmark, and backbone, reporting both discrimi-
nation and selective risk. Unanimous-layer metrics, computed on a different subpopulation, are in
§4.3 and Figure 3 rather than here.
The full grid confirms the structure of §4.2 without exception. Terminal agreement is at chance in
all seven cells, on the strongest backbone as on the weakest. Sequence log-probability is the only
signal that getsworsethan chance, and its selective risk degrades sharply on Llama-3.1-8B, where
long free-text rationales make raw likelihood an actively misleading confidence proxy. Answer-
distribution entropy tracks first-round agreement closely everywhere, as both are functionals of the
same round-one vote distribution. The process score has the best discrimination in six of the seven
18

cells and the best selective risk in five, and its margin over the strongest snapshot signal is widest
where the backbone is weakest.
Figure 5 plots the risk-coverage curves behind the AURC column on the Qwen3-8B substrate. The
process score traces the lowest risk over nearly the whole coverage range on all three benchmarks,
and the ordering of the other signals is stable: terminal agreement is flat, sequence log-probability is
inverted at low coverage, and the two round-one functionals move together.
Table 5: Discrimination (AUROC) and selective risk (AURC) of all signals across benchmarks and
backbones. Dashes mark cells outside the evaluation design: the cross-backbone replication covers
the two primary benchmarks.
AUROC↑AURC↓
Signal MedQA MMLU-Pro MedMCQA MedQA MMLU-Pro MedMCQA
Qwen3-8B
Terminal agreement 0.512 0.504 0.507 0.221 0.300 0.330
First-round agreement [Wang et al., 2023] 0.677 0.698 0.640 0.155 0.200 0.250
Sequence log-probability 0.400 0.474 0.534 0.320 0.376 0.347
Answer entropy [Kuhn et al., 2023] 0.681 0.703 0.641 0.154 0.199 0.250
Verbalized confidence [Tian et al., 2023] 0.585 0.651 – 0.179 0.207 –
P(True) [Kadavath et al., 2022] 0.602 0.644 – 0.160 0.205 –
ProbeGuard (process score) 0.696 0.704 0.659 0.145 0.205 0.234
HuatuoGPT-o1-8B
Terminal agreement 0.519 0.526 – 0.193 0.353 –
First-round agreement [Wang et al., 2023] 0.766 0.701 – 0.093 0.232 –
Sequence log-probability 0.350 0.422 – 0.265 0.399 –
Answer entropy [Kuhn et al., 2023] 0.768 0.707 – 0.093 0.230 –
ProbeGuard (process score) 0.775 0.701 – 0.096 0.225 –
Llama-3.1-8B
Terminal agreement 0.548 0.551 – 0.289 0.406 –
First-round agreement [Wang et al., 2023] 0.781 0.736 – 0.159 0.254 –
Sequence log-probability 0.205 0.217 – 0.547 0.640 –
Answer entropy [Kuhn et al., 2023] 0.790 0.744 – 0.156 0.252 –
ProbeGuard (process score) 0.825 0.782 – 0.142 0.246 –
0.25 0.50 0.75 1.00
Coverage020406080100Selective risk (%)MedQA
0.25 0.50 0.75 1.00
Coverage020406080100MMLU-Pro medicine
0.25 0.50 0.75 1.00
Coverage020406080100MedMCQATerminal agreement First-round agreement Sequence log-probability Answer entropy Process score
Figure 5: Risk-coverage curves of the always-available signals on the Qwen3-8B substrate; lower is
better, and the AURC values of Table 2 are the areas under these curves. The process score uses the
same five-fold out-of-fold protocol as the tables; coverage below 5% is omitted.
19

G Probe Control Arms
Table 6 isolates what drives the flip asymmetry of §4.3. Each control arm re-asks the same 746
unanimous MedQA questions with the same N=8 re-answers, varying one factor at a time: the
instruction-only arm keeps the devil’s-advocate framing and removes all documents; the irrelevant-
documents arm keeps the framing and substitutes documents retrieved for other questions’ counter-
queries; the neutral-instruction arm keeps this question’s counter-evidence documents and removes
the adversarial framing.
The instruction alone flips no majority at all, and documents without evidential content flip 1.0% of
wrong consensus. The counter-evidence under a neutral instruction retains nearly the full asymmetry
(12.0% versus 3.6%), and the paired difference in asymmetry between the probe and each content-free
arm excludes zero (bootstrap 95% intervals [3.0,16.8] and[1.7,16.8] percentage points). The flip
responds to the content of the counter-evidence, and the probe inherits its discrimination from that
content.
Table 6: Flip rates of the counter-evidence probe and its control arms on the MedQA unanimous layer
(Qwen3-8B,N=8re-answers per question).
Arm Instruction Documents flip(wrong) flip(right)
Counter-evidence probe devil’s advocate counter-evidence 14.0% 4.2%
Instruction only devil’s advocate none 0.0% 0.0%
Irrelevant documents devil’s advocate unrelated 1.0% 0.2%
Neutral instruction neutral counter-evidence 12.0% 3.6%
H Ablation Matrices
Table 7 extends the drop-one ablation of §5.1 to all three benchmarks, removing one feature family at
a time from the full process score, and adds the classifier swap.
Table 7: Drop-one family ablation and classifier swap (AUROC, Qwen3-8B substrate). Each row
removes the feature family marked✗from the full process score (top row).
Feature family AUROC
Agreement Distribution Minority Rationale Retrieval MedQA MMLU-Pro MedMCQA
✓ ✓ ✓ ✓ ✓0.696 0.704 0.659
✗ ✓ ✓ ✓ ✓0.703 0.705 0.660
✓ ✗ ✓ ✓ ✓0.703 0.714 0.657
✓ ✓ ✗ ✓ ✓0.697 0.705 0.658
✓ ✓ ✓ ✗ ✓0.682 0.674 0.645
✓ ✓ ✓ ✓ ✗0.700 0.697 0.658
GBT in place of logistic regression 0.644 0.643 0.619
The pattern of §5.1 holds on every benchmark, not only MedQA: removing the rationale family costs
discrimination in all three columns, while removing any other family leaves the score unchanged
or marginally improved. The trajectory families are informative individually but interchangeable
collectively, each a different projection of the same consensus dynamics; the rationales are the one
signal source the rest cannot substitute for. The classifier swap points the same direction on all
benchmarks as well, so the choice of a default logistic regression is not doing hidden work. The
hard-versus-soft aggregation comparison for the rationale score is in §5.1, and the pilot-question
sensitivity check in §5.2.
I Specialty-Level Certification
Specialty labels are available for 899 MedQA test questions, of which 529 fall in the unanimous layer.
Table 8 breaks the layer-level certificate of §4.4 down by specialty, using the same score, level, and
repeated-split protocol; specialties with fewer than 25 unanimous questions are omitted.
20

Table 8: Per-specialty breakdown of the certified rule on the MedQA unanimous layer ( α=0.15 , 50
splits). The certificate itself is calibrated at the layer level; this table reports how its coverage and
realized risk distribute across specialties.
SpecialtynUnanimous-layer error (%) Certified coverage Realized risk (%)
Cardiology 42 14.3 0.70 9.0
Hematology 29 17.2 0.69 13.3
Obstetrics and gynecology 51 11.8 0.66 14.5
Gastroenterology 57 15.8 0.60 11.1
Pediatrics 35 11.4 0.60 8.9
Psychiatry 33 12.1 0.59 7.3
Nephrology 30 13.3 0.59 8.4
Emergency medicine 42 26.2 0.57 19.1
Infectious diseases 36 13.9 0.57 10.0
Endocrinology 51 11.8 0.52 4.0
Neurology 37 10.8 0.49 7.7
Emergency medicine is the outlier on both axes: its unanimous consensus is wrong about twice as
often as the layer average, and it is the one specialty whose realized risk under the layer-level threshold
exceeds the target. This is consistent with clinical experience, as acute multi-system presentations are
where a shared quick prior is most likely to mislead all eight candidates at once. Coverage does not
simply track the error rate: endocrinology and neurology have among the lowest unanimous-layer
error rates yet receive the most conservative coverage, while obstetrics and cardiology combine similar
error rates with far wider certified regions, meaning the process signals are differently informative
across specialties, not merely calibrated to difficulty. The spread matters for deployment: a layer-level
certificate is honest in aggregate, and a site that needs per-specialty control, routing emergency-
medicine questions more conservatively than cardiology ones, can re-run the same calibration within
specialty strata once enough per-specialty calibration data accumulate.
J Cost Accounting
The passive layers are computed from logs the system already writes, at no inference cost. The probe
fires only on unanimous first-round votes (59% of MedQA, 59% of MMLU-Pro medicine, 55% of
MedMCQA) and costs one counter-query call, one retrieval round, and Nconcurrent re-answers:
a median of 114 s and 4381 completion tokens per probed MedQA question, against the 72 s the
substrate already spends answering, so probing roughly doubles the cost of a unanimous question and
adds nothing to a contested one. Rationale scoring is a single batched NLI pass, minutes per dataset
on one GPU. All experiments ran on one node with six A40 GPUs; no API models were used.
K Qualitative Cases
The two cases of Figure 1, with their probe trajectories as logged. Both end in a unanimous panel;
what differs is which answer survives.
Case 1: unanimously wrong, repaired ( κ= 0 ).A patient hospitalized for a week after a stroke
develops fever, purulent cough, and a new consolidation; all eight candidates answerStreptococcus
pneumoniae, each rationale invoking the same community-acquired prior and none engaging the
hospitalization timeline. The reasoning is not careless: one rationale methodically rules out tuber-
culosis andHaemophilus influenzaebefore concluding“Given the clinical presentation, the most
likely causal microorganism is Streptococcus pneumoniae”; each step is sound, and the error is in
what the chain never considers. The probe’s counter-query generator re-reads the stem and targets the
discarded cue, generating“Hospital-acquired pneumonia in stroke patients: common pathogens”and
“Role of Staphylococcus aureus in hospital-acquired pneumonia with consolidation”among its four
queries. With the eight retrieved challenge documents in context, all eight re-answers abandon the
consensus and converge, again unanimously, on the gold answerStaphylococcus aureus. The probe
did not surface unusual evidence; it directed attention back to context the candidates had ignored.
Case 2: unanimously correct, held ( κ= 1 ).On the dimeric-immunoglobulin question, all eight
candidates answer that secretory IgA inhibits bacterial adherence to mucous membranes, and the
eight rationales cite the same mechanism. The probe generates genuine counter-queries, one asking
21

directly for“Evidence against IgA’s role in inhibiting bacterial adhesion”, and returns eight challenge
documents. All eight re-answers weigh the challenge and retain the consensus.
These two trajectories are the endpoints of the keep rate; intermediate values, a consensus that
partially fragments under challenge, populate the separation of Figure 3. The complete probe records,
including all generated queries, retrieved document identifiers, and per-candidate re-answer votes,
are in the released logs.
L Implementation Details
The substrate runs unmodified at a pinned commit, and every component downstream of it is
deterministic given the seed. Entailment is scored by DeBERTa-large fine-tuned on MNLI [He
et al., 2021]; reranking uses the MedCPT cross-encoder [Jin et al., 2023]; both are used off the shelf
without fine-tuning. Table 9 lists the hyperparameters of each stage. None of them were tuned on
evaluation data: the generation and retrieval settings are the substrate’s own defaults, the classifier is
a scikit-learn default configuration, and the certification constants of Appendix E were fixed before
calibration data were collected. A single seed governs generation, data splits, cross-validation, and
bootstrap resampling; exact software versions and the per-dataset settings manifest ship with the
code.
Table 9: Hyperparameters by pipeline stage.
Parameter Value
Consensus generation
Panel sizeN8
Maximum roundsT8
Candidate temperature0.7
Candidate max new tokens2048
Query temperature (thinking mode)0
Retrieval
BM25(k 1, b) (0.9,0.4)
Candidates per query32
Kept after reranking2
Documents per round≤8
Rationale scoring and process score
NLI input limit (tokens)256
Classifierℓ 2logistic regression
Cross-validation folds5
Random seed (all stages)42
22