# When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1

**Authors**: Weimeng Luo

**Published**: 2026-08-13 13:42:48

**PDF URL**: [https://arxiv.org/pdf/2608.13237v1](https://arxiv.org/pdf/2608.13237v1)

## Abstract
Multi-round retrieval-augmented generation (RAG) must decide when to stop searching as evidence accumulates. Because the deployed policy is determined by the first STOP on each trajectory, this is a sequential selection problem rather than an independent state-classification task. We adapt S2G-RAG's structured sufficiency-and-gap judgment to a frozen Search-R1 pipeline and train a Qwen3.5-2B judge on 3,009 states from 900 disjoint HotpotQA questions. Search-R1's reasoner, retriever, corpus, prompt, and search budget remain unchanged, while the judge checkpoint and stopping threshold are selected on grouped validation and frozen before confirmatory evaluation. On the confirmatory test set, the resulting policy reduces retrieval calls by 77 (3.70\%) relative to Native Search-R1, while Official Exact Match decreases by 0.625 percentage points. Thus, the trained S2G-style structured judge reduces retrieval while broadly preserving answer accuracy. The result does not imply unchanged or improved accuracy, safe stopping, or lower total inference cost.

## Full Text


<!-- PDF content starts -->

When Should Multi-Round RAG Stop?
Structured Stopping Judgments and Retrieval
Reduction in Search-R1
W eimeng Luo
Unaﬀiliated
weimengluo@gmail.com
Abstract
Multi-round retrieval-augmented generation (RAG) must decide
when to stop searching as evidence accumulates. Because the de-
ployed policy is determined by the first STOP on each trajectory , this
is a sequential selection problem rather than an independent state-
classification task. W e adapt S2G-RAG’s structured suﬀiciency-and-
gap judgment to a frozen Search-R1 pipeline and train a Qwen3.5-2B
judge on 3,009 states from 900 disjoint HotpotQA questions. Search-
R1’s reasoner, retriever, corpus, prompt, and search budget remain
unchanged, while the judge checkpoint and stopping threshold are se-
lected on grouped validation and frozen before confirmatory evaluation.
On the confirmatory test set, the resulting policy reduces retrieval calls
by 77 (3.70%) relative to Native Search-R1, while Oﬀicial Exact Match
decreases by 0.625 percentage points. Thus, the trained S2G-style
structured judge reduces retrieval while broadly preserving answer ac-
curacy . The result does not imply unchanged or improved accuracy ,
safe stopping, or lower total inference cost.
Keywords: multi-round retrieval-augmented generation; adaptive retrieval;
stopping policy; risk–coverage; structured judge; Search-R1
1 Introduction
Iterative retrieval lets a language model decide which evidence to request as
reasoning unfolds. IRCoT interleaves retrieval and reasoning [ 1]. Self-RAG
learns retrieval and reflection decisions [ 2], Adaptive-RAG routes questions
among strategies of different complexity [ 3], and Search-R1 learns to reason
and call a search engine through reinforcement learning [ 4]. These methods
improve flexibility , but they also create a stopping problem. Retrieving too
little can leave the answer unsupported. Retrieving after a suﬀicient state
consumes latency and compute, and the additional context can change a
correct candidate into an incorrect one.
1
arXiv:2608.13237v1  [cs.IR]  13 Aug 2026

Recent work has made the stopping decision explicit. SIM-RAG learns
whether to continue search from generated answers and rationales [ 5]. S2G-
RAG asks a judge to predict both suﬀiciency and the missing information
needed for the next retrieval [ 6]. T ASR studies training-free stopping for iter-
ative retrieval [ 7], while RAG-Critic uses critic feedback to guide an agentic
workflow [ 8]. These systems motivate learned stopping, but their reported
state-level scores or end-to-end averages do not by themselves identify why
a deployed stopping policy succeeds or fails.
The missing link is sequential selection. A state classifier is evaluated
over many intermediate states, whereas an online policy is determined by
the first threshold crossing on each trajectory . One early false positive can
suppress every later state. Conversely , a perfectly conservative classifier can
leave the system unchanged. The feasibility of improvement is also bounded
by candidate reachability: if no correct answer can be generated from any
frozen reachable state, changing the stopping time cannot make the question
correct. This paper therefore asks four questions:
1. How often does an audited strong search policy continue after a correct
answer is already reachable?
2. How much quality and retrieval-cost headroom exists when only the stop-
ping time may change?
3. Does structured suﬀiciency-and-gap supervision improve state ranking
and the deployed earliest-stop policy?
4. What answer risk accompanies the retrieval reduction that survives con-
firmatory evaluation?
W e answer these questions in two stages. First, an exploratory reachabil-
ity audit runs a frozen, non-intervening answer-only probe at native Search-
R1 states. It estimates empirical conditional oracles without changing the
search trajectory . Second, a scale-up experiment trains an S2G judge on
disjoint training questions, freezes model and threshold choices before final
evaluation, and reserves indices 200–999 of a 1,000-question HotpotQA set
for confirmatory analysis.
The paper makes three contributions. First, it adapts an S2G-style
structured judging mechanism to Search-R1 and trains a Qwen3.5-2B judge
specifically on Search-R1 states. Second, it provides a trajectory-aware for-
mulation that separates candidate reachability , state ranking, first-crossing
policy behavior, answer quality , and cost, including a controlled mechanism
ladder in which binary supervision improves average precision but harms
the system. Third, it shows on the confirmatory test set that the com-
plete policy reduces retrieval calls while keeping the loss in answer accuracy
within the prespecified two-percentage-point tolerance. Safe stopping is not
established.
2

2 Related W ork and Positioning
2.1 Adaptive and Iterative Retrieval
Adaptive retrieval encompasses several decision interfaces. Query-level
routers decide among no retrieval, one retrieval, and multi-step retrieval
before the trajectory starts [ 3]. Interleaved systems decide what to retrieve
during reasoning [ 1,4]. Generation-time systems emit control or reflection
tokens that can trigger retrieval or critique [ 2]. Answer-level controllers
instead decide whether an existing candidate should be accepted or
whether search should continue [ 5]. These interfaces are related, but their
oracles and error semantics differ. An answer-level STOP label cannot be
transferred to a pre-query router without changing the decision problem.
Search-R1 is the upstream policy studied here [ 4]. Our objective is not
to improve its reasoner or retriever. W e freeze the 7B reasoner, E5 top-3
retrieval, Wiki-2018 corpus, prompting, and four-search budget. The only
intervention is whether a reachable state is accepted early . This restriction
makes the resulting oracle conditional on the frozen interface rather than a
global upper bound over possible queries, documents, or models.
2.2 Learned Judges and Structured Gaps
SIM-RAG directly models whether a search process should continue [ 5]. Our
released-checkpoint pilot treats it as an external answer-level controller and
finds a cross-system transfer failure: on 200 questions, it performs eight
harmful changes and no rescues relative to native Search-R1. W e retain
that result only as exploratory evidence because its trajectories, reasoner,
and interface differ from the training environment of the released critic.
S2G-RAG argues that a binary suﬀicient/insuﬀicient target discards in-
formation about what remains missing [ 6]. W e transfer this structured judg-
ing mechanism to the Search-R1 state interface and train a judge specifi-
cally on Search-R1 states. Our judge outputs a frozen JSON schema with a
Boolean sufficient field and a list of gap_items . The online policy uses
the log-probability margin for the Boolean field at a frozen prefix, while the
structured target regularizes training toward explicit missing-information
judgments. W e do not claim to reproduce the complete S2G-RAG system,
nor do we train or modify Search-R1 itself; the evaluated system is the com-
plete policy formed by frozen Search-R1 and the trained S2G-style judge.
2.3 Risk, Coverage, and System Utility
Selective classification separates the conditional error of accepted predictions
from their coverage [ 9]. Certified risk work extends this concern to RAG
generation [ 10]. W e use the same conceptual separation, but do not claim a
formal risk guarantee. The confirmatory gate controls an EM non-inferiority
3

1  Reachability
Can the current state
answer correctly now?2  State ranking
Can the judge rank
safe stop states?3  First crossing
Which state crosses
the threshold first?4  System effects
Answer quality, stop risk,
and retrieval costTrajectory-aware evaluation of multi-round RAG stopping
Illustrative trajectory: first crossing at s2
s0 s1 s2 s3 s4
SEARCH SEARCH STOP not reached not reached
a false first crossing truncates every later stateFigure 1: T rajectory-aware evaluation of multi-round RAG stopping. A cor-
rect reachable candidate is necessary but not suﬀicient; the judge must rank
it, the earliest threshold crossing must select it, and the resulting answer
quality , selected-stop risk, and retrieval cost must be evaluated separately .
margin and tests whether paired search calls decrease. STOP precision,
early-stop safety , average precision, calibration, and end-to-end EM remain
distinct outcomes.
This positioning matters because a retrieval reduction is not equivalent
to total eﬀiciency . The judge itself consumes inference compute, and our
experiment does not normalize wall-clock time, energy , or token-equivalent
judge cost. The paper therefore reports retrieval calls as the cost endpoint
and reserves “total eﬀiciency” for future matched-compute evaluation.
3 Problem F ormulation
Figure 1summarizes the evaluation chain. Each layer answers a differ-
ent question: reachable candidates bound what stopping can achieve, state
scores diagnose representation quality , the first threshold crossing deter-
mines the deployed decision, and system outcomes establish the quality–
risk–cost trade-off.
3.1 Reachable States and Stop-Now Answers
F or question qi, native Search-R1 produces a trajectory of reachable states
zi,t= (qi, Ci,t, hi,t, anative
i,t ), (1)
4

where Ci,t is the accumulated retrieved context, hi,t is the native reasoning
history , and anative
i,t is the next native action. W e consider the initial state and
each state after a successful retrieval. A frozen answer-only probe gmaps
the question and current context to a candidate answer ˜ai,t=g(qi, Ci,t).
The probe output never re-enters the native trajectory .
Letyi,t= 1 when ˜ai,t matches the oﬀicial answer under benchmark nor-
malization. A safe early-stop opportunity exists when native Search-R1
proposes another SEARCH while yi,t= 1 . This target does not assert that
the retrieved evidence is complete in a supporting-fact sense. It states only
that the frozen answerer can already recover the oﬀicial answer from the
current interface.
3.2 Conditional Oracles
The quality oracle selects the earliest reachable correct probe answer, if one
exists. Its EM is an empirical upper bound for answer selection over the
frozen states. The native-preserving cost oracle applies an earlier correct
stop only to questions that native Search-R1 already answers correctly . It
therefore holds native EM fixed and estimates how many search calls can be
removed without changing those final labels.
Neither oracle is deployable. Both use labels after trajectory generation.
Their role is diagnostic: when quality and cost headroom are negligible, a
learned stopping model has little room to improve the frozen interface.
3.3 Scores, Earliest Stopping, and Metrics
The judge produces a scalar score si,t. F or threshold θ, the online policy
stops at
τi(θ) = min{t:si,t≥θ}. (2)
If no state crosses the threshold, execution follows the frozen native fall-
back. State-level average precision (AP) evaluates ranking over eligible
states. STOP precision and recall evaluate the thresholded state decision.
System-level early-stop safety counts whether the selected first crossing is a
safe opportunity . End-to-end quality is oﬀicial EM over all questions, and
cost is the mean number of search calls.
The main paired effects for model m relative to native execution are
∆ EM(m) =1
N∑
i[EM i,m−EM i,native ], (3)
∆ search (m) =1
N∑
i[Si,m−Si,native ]. (4)
Confidence intervals use 10,000 question-level paired bootstrap resamples
with seed 20260728.
5

4 Experimental Design
4.1 Audited Upstream System
The upstream agent is the oﬀicial Search-R1 Qwen2.5-7B PPO v0.2 check-
point with E5 top-3 retrieval over Wiki-2018 and at most four searches
[4]. A separate 1,200-question NQ/HotpotQA audit obtained oﬀicial EM of
45.67% and 46.00%, respectively . The reported reference values fell inside
source-specific Wilson 95% intervals and differed by less than five percent-
age points. W e call this status configuration-verified and compatible , not a
strict reproduction.
The main scale-up data use the first 1,000 questions of the HotpotQA
distractor development set [ 11], materialized through the S2G-code-aligned
preprocessing path. Question identity and normalized text were checked
against every training, grouped-validation, and reserve question.
4.2 Exploratory Reachability Audit
Gate H0 reuses 200 frozen native trajectories. It evaluates 723 native reach-
able states: 200 initial states and 523 states preceding native SEARCH ac-
tions. Seventy-five states created only by a forced-continue intervention are
retained as a counterfactual supplement and excluded from native-behavior
attribution. Probe prompting, deterministic decoding, parsing, state dedu-
plication, and cost accounting were frozen before labels were made available.
All 798 probes parsed successfully .
H0 is exploratory because its 200 labels had already been used for earlier
pilot analyses. It determines whether further stopping research is warranted
and supplies failure decompositions, but it does not support the final system
claim.
4.3 T raining the Expanded Structured Judge
The training source is the HotpotQA training split, separated by question
into 700 fresh training questions, 100 grouped-validation questions, and 200
reserve questions. The reserve was activated according to a predeclared
data-suﬀiciency gate, yielding 900 training questions and 3,009 clean states.
T eacher outputs with generation or schema failures were removed by frozen
rules. Labels were not manually repaired. Supporting-fact labels were used
only for one-way conflict filtering after teacher generation and were never
provided to the judge.
The judge is Qwen3.5-2B. It is trained for three epochs (12,654 steps) to
emit the top-level key order sufficient followed by gap_items . All losses
are finite and no training OOM occurs. Epoch 1 is selected using grouped
teacher-target token negative log-likelihood, without using answer labels or
6

final-set metrics. The frozen expanded adapter tree has SHA-256 cfed584b
95025cc4f1fd555cd2236b2773f1289c67dab03d26b2829659578767 .
W e compare three judges: the unadapted Structured Base model, an
older S2G LoRA retained only as a historical comparator, and the expanded
S2G LoRA. Each receives the same state representation and score extraction.
No final labels are used to alter prompts, checkpoints, thresholds, budgets,
input order, or statistics.
4.4 Threshold F reezing and Confirmatory Analysis
F or each judge, grouped validation chooses the threshold that maximizes
recall subject to empirical STOP precision of at least 0.90 and at least 10
predicted STOP states. If no threshold is feasible, the frozen implementa-
tion encodes fallback=never_stop as a finite sentinel equal to the maxi-
mum validation score plus one. This guarantees zero STOP predictions on
grouped validation, but not literal never-stopping under distribution shift.
The expanded judge freezes at a margin of approximately 7.875. Structured
Base and the old LoRA use sentinels of 2.5 and approximately 8.0, respec-
tively .
The first 200 development questions remain exploratory . Indices 200–
999 form the confirmatory test set. Before evaluation, we specify that the
expanded policy may lose at most two EM percentage points relative to
Native. F ormally , the lower endpoint of the paired 95% interval for ∆ EM
must exceed −0.02. The upper endpoint of the paired 95% interval for
∆ search must also be below zero. The first condition is the statistical non-
inferiority criterion; in plain language, the loss in answer accuracy must
remain within the predefined acceptable range. It does not require accuracy
to be exactly unchanged. F ull 1,000-question results are descriptive only .
Final inference follows a blind sequence. Native trajectories, answer-only
probes, and all judge scores undergo structure and order audits before final
labels are isolated for evaluation. A preblind script computed a SHA-256
hash of the final-label file despite the no-read rule. The hash was not used
for selection and the file was not mounted into inference containers, but the
access is a procedural nonconformance and is disclosed rather than silently
waived.
5 Results
5.1 A Strong Search Policy Still Has Stopping Headroom
On the exploratory H0 set, 143 of 523 native SEARCH actions (27.3%) occur
at states where the frozen probe already produces the correct oﬀicial answer.
These opportunities occur in 70 of 200 questions. Sixty questions show cost-
only over-search: an early probe and the native final answer are both correct.
7

0.0 0.5 1.0 1.5 2.0 2.5 3.0
Mean search calls per question152025303540455055Official EM (%)
k=0k=1k=2k=3 k=4+5.0 pp
same EM, fewer searchesa  Fixed budgets versus conditional oracles
Fixed budget
Native
Quality oracle
Cost oracle
0 10 20 30
Estimate with question-level 95% CI (%)Quality
headroom
Avoidable
searches5.00%
25.05%b  Exploratory headroomFigure 2: Exploratory H0 stopping headroom on 200 questions. Panel
(a) compares uniform search budgets with native Search-R1 and two label-
dependent conditional oracles. Panel (b) reports question-level bootstrap
95% intervals. The quality oracle is retrospective and the cost oracle pre-
serves Native EM; neither is a deployable policy or a confirmatory result.
T en questions show harmful over-search: an earlier probe is correct but the
native final answer is wrong.
Native EM is 46.5%. The quality oracle reaches 51.5%, an exact gain
of 10/200 or 5.0 percentage points. The native-preserving cost oracle re-
moves 131 of 523 searches (25.05%) while holding native EM fixed. Fixed
budgets do not realize the same question-specific trade-off: answer-only EM
at budgets k= 0,1,2,3,4is 18.5%, 34.0%, 43.0%, 46.0%, and 47.0%. This
gap between a per-question oracle and uniform budgets motivates a learned
state-dependent controller.
5.2 State Ranking Does Not Determine the Policy
The exploratory mechanism ladder exposes a ranking–policy mismatch. A
binary LoRA raises STOP AP from 0.732 to 0.828 on the Core200 diagnostic,
yet the zero-threshold policy reduces oﬀicial EM from 36.5% to 31.5% and
mean searches from 2.635 to 1.430. The model has learned a stronger state
ranking signal but stops too aggressively under the deployed threshold.
Replacing the binary target with a minimal S2G target reduces wrong
STOP decisions from 99 to 24 and premature STOP decisions from 18 to 4.
Oﬀicial EM recovers from 31.5% to 35.5%, while mean searches increase from
1.430 to 2.390. Relative to Structured Base, however, the EM difference is
only +0.015 with a paired 95% interval of [−0.010,0.040] . This diagnostic
8

supports structured gaps as a mechanism for reducing over-stopping, not a
system-level superiority claim.
5.3 Grouped V alidation F reezes a Conservative Policy
On 100 grouped-validation questions, the expanded judge obtains STOP
precision 0.9091, recall 0.1205, and AP 0.5285 at its frozen threshold. Its
end-to-end EM equals Native Search-R1 at 0.53, and mean search calls fall
from 2.45 to 2.33. Structured Base and the old LoRA cannot satisfy the
validation precision-and-support rule and therefore receive the finite conser-
vative sentinels described above. These measurements select the final policy .
They are not part of the confirmatory claim.
5.4 Confirmatory T est Set: Limited Accuracy Loss and
F ewer Searches
Both confirmatory criteria pass. On the confirmatory test set comprising
indices 200–999, Native Search-R1 reaches EM 0.44875 with 2.60125
mean search calls, whereas the expanded S2G policy reaches EM 0.44250
with 2.50500 searches. The paired EM difference is −0.00625 (95% CI
[−0.01250 ,0]): the point estimate loses 0.625 percentage points, and
the interval remains within the predeclared maximum acceptable loss of
two percentage points. The search-call difference is −0.09625 (95% CI
[−0.12000 ,−0.07375] ); its upper endpoint is below zero.
In absolute terms, Native makes 2,081 retrieval calls and the expanded
policy makes 2,004, a reduction of 77 calls or 3.70%. Six questions change
from Native-correct to policy-wrong, while one changes from Native-wrong
to policy-correct, yielding the net −5/800 = −0.625 percentage-point EM
difference. T able 1places each effect, interval, and frozen criterion together
so that non-inferiority is not read as equivalence or superiority .
T able 1: Primary results on the confirmatory test set. Differences are Ex-
panded S2G minus Native; both frozen criteria pass.
Metric Native Expanded Difference [95% CI] F rozen criterion V erdict
Oﬀicial EM 0.44875 0.44250 −0.00625 [ −0.01250 ,0] CI lower >−0.02 P ASS
Mean searches 2.60125 2.50500 −0.09625 [ −0.12000 ,−0.07375] CI upper <0 P ASS
Expanded S2G improves STOP AP over Structured Base by 0.03983
(95% CI [0.00571 ,0.07228] ), providing confirmatory evidence of better state
ranking. Ranking improvement does not establish safe early stopping. T a-
ble 2separates state-level diagnostics from the system-level first selected
stop. One test-set state exceeds each finite fallback sentinel for Structured
Base and the old S2G model, so these baselines implement validation-relative
conservative fallbacks rather than mathematically infinite thresholds.
9

T able 2: Judge diagnostics on the confirmatory test set. STOP P/R/AP
are state-level metrics over all eligible states; early-stop counts evaluate the
first threshold crossing per question.
Judge STOP P STOP R STOP AP Early stops (safe/unsafe)
Structured Base 0.0000 0.0000 0.36375 1 (0/1)
Old S2G LoRA 1.0000 0.0019 0.36596 1 (1/0)
Expanded S2G LoRA 0.6216 0.0880 0.40358 69 (42/27)
5.5 The Retrieval Reduction Carries Material Stop Risk
The expanded policy makes 69 system-level early stops, or 8.63% of the 800
questions. F orty-two are safe under the frozen target and 27 are unsafe,
so the selected early-stop unsafe fraction is 39.13%. The 42 safe selections
cover only 14.29% of the 294 questions that contain a safe opportunity . State-
level STOP precision is 0.6216 and recall is 0.0880. The distinction between
42/69 system outcomes and state-level precision is intentional: the former
evaluates the first selected stop per question, whereas the latter evaluates
all eligible states under the threshold.
The grouped-validation STOP precision of 0.9091 is based on only 11
predicted STOP states and falls to 0.6216 at the frozen threshold on the
confirmatory test set. The validation precision constraint therefore does not
transfer to the final distribution. Calibration is also weak. Expanded S2G
has Brier score 0.2952 and ten-bin expected calibration error (ECE) 0.2911,
compared with 0.1992 and 0.1179 for Structured Base. The expanded judge
ranks safe opportunities better, but its raw confidence scale is less calibrated.
At a post-hoc state-level operating point that reaches at least 0.90 precision,
recall falls to 0.0057. These facts rule out describing the policy as risk-
certified or broadly safe.
The 27 unsafe early stops do not map one-to-one to newly wrong final
questions. Relative to Native, six questions become wrong and one becomes
correct. The remaining unsafe stops occur where the Native outcome is
already wrong or where an early state fails the frozen safe-opportunity defi-
nition even though the final answer remains correct. The EM non-inferiority
test measures average quality over all questions, whereas selected-stop risk
measures conditional risk among the policy’s active acceptances. They an-
swer different questions and must both be reported.
Figure 3places these three facts together: the EM interval remains above
the frozen non-inferiority margin, the retrieval interval excludes zero in the
favorable direction, and the selected early stops retain a material unsafe
fraction. This is the paper’s positive result and its principal boundary in
one view.
10

−2 0
EM difference (pp)non-inferiority
margin
-0.625 pp
[-1.25, 0.00]a  Quality bound
−0.1 0.0
Search-call difference-0.09625
[-0.12000, -0.07375]b  Retrieval reduction
0 20 40 60
System-level early stops (n)42 27
unsafe fraction = 39.13%c  Selected-stop risk
Safe: 42
Unsafe: 27Figure 3: Confirmatory test-set quality–cost–risk result for Expanded S2G
relative to Native Search-R1. Panels (a) and (b) show paired bootstrap
95% intervals; the dashed lines mark the frozen −2percentage-point EM
non-inferiority margin and zero search difference. Panel (c) decomposes
the 69 system-level first selected early stops. The result supports retrieval
reduction within the quality bound, not answer-quality superiority or safe
stopping.
5.6 F ull 1,000-Question Results Are Descriptive
Across indices 0–999, native EM/searches are 0.4490/2.6100 and expanded
S2G obtains 0.4430/2.5210. Expanded STOP precision/recall/AP are
0.6163/0.0805/0.3987. The system makes 80 early stops, 49 safe and 31
unsafe. These values are useful for inventory and reproducibility , but the
first 200 questions had exploratory status and cannot be pooled into the
confirmatory claim.
6 Discussion
6.1 What the Results Establish
The experiments support a clear central claim. W e adapt an S2G-style
structured judging mechanism to frozen Search-R1 and train a Qwen3.5-
2B judge on Search-R1 states. On the confirmatory test set, the resulting
complete policy reduces retrieval calls while keeping the loss in answer accu-
racy within the two-percentage-point range specified before evaluation. The
95% interval for the retrieval difference excludes zero, and the EM interval
remains above the allowed-loss boundary . The reachability audit explains
why Search-R1 has removable retrieval calls, while the ranking–policy diag-
11

nostics show why the judge must be evaluated through question-level first
crossings. The system result belongs to the complete combination of train-
ing data, structured supervision, checkpoint, and threshold and cannot be
attributed to any one component in isolation.
6.2 What the Results Do Not Establish
The expanded judge does not improve answer quality . Its point estimate
is 0.625 percentage points below native EM. Keeping the loss within an
acceptable range does not mean that accuracy is exactly unchanged, and it
does not imply superiority .
The policy is also not a safe stopping rule. T wenty-seven of 69 early
stops are unsafe under the frozen definition, and the 0.90 validation precision
constraint does not transfer to final evaluation. A deployment that requires a
low conditional error rate among accepted early stops would need a separate
risk-control procedure, more calibration data, a more conservative policy ,
or abstention. The present quality gate can tolerate some unsafe STOP
decisions because it averages over all questions.
Finally , fewer retrieval calls do not prove lower total cost. The Qwen3.5-
2B judge runs on eligible states. Its compute, memory , and latency are not
converted into retrieval-equivalent units. The result is best interpreted for
environments where retrieval dominates marginal cost or where judge in-
ference can be amortized. Matched-latency and matched-energy evaluation
remain open.
6.3 Implications for Stopping Research
F uture stopping work should report a four-part evaluation. First, establish
candidate reachability and oracle headroom on the exact decision interface.
Second, report state ranking to diagnose representation learning. Third,
replay or execute the earliest-stop policy and report question-level safe and
unsafe selections. F ourth, pair quality with retrieval, token, latency , and
judge-compute costs. Any one of these views alone is incomplete.
Structured gap prediction remains promising because it reduces aggres-
sive over-stopping in the mechanism ladder and improves AP in the scale-up
study . Its next test should target the large gap between AP and calibrated
decision utility . Candidate directions include calibration-aware training,
trajectory-level losses that penalize the first unsafe crossing, and explicit
cost-sensitive objectives. These are hypotheses for new experiments, not
claims supported by the present study .
12

7 Conclusion
W e adapt an S2G-style structured judging mechanism to frozen Search-R1
and train a Qwen3.5-2B judge specifically on Search-R1 states. On the
confirmatory test set, the complete policy makes 77 fewer retrieval calls than
Native Search-R1: mean searches fall from 2.60125 to 2.50500, a difference
of−0.09625 or 3.70% (95% CI [−0.12000 ,−0.07375] ). Oﬀicial EM falls from
0.44875 to 0.44250, a loss of 0.625 percentage points (95% CI [−1.25,0]
percentage points), which remains within the maximum acceptable loss of
two percentage points specified before evaluation. The experiment therefore
shows that the trained S2G-style judge reduces retrieval calls while broadly
preserving answer accuracy . Here, “broadly preserving” allows a small loss
within the prespecified range; it does not mean that accuracy is unchanged
or improved.
The policy is not yet a safe stopping rule: 27 of its 69 early stops are
unsafe, for selected-stop risk of 39.13%. Keeping overall answer accuracy
within an acceptable range and making every early stop safe are distinct
objectives.
Ethics and Responsible Reporting
The study uses public question-answering benchmarks and does not involve
human participants or personal data. The main reporting risk is scientific
overstatement. W e therefore distinguish exploratory , validation, confirma-
tory , and descriptive populations, disclose infrastructure and blinding inci-
dents, and state negative boundaries alongside positive results.
AI-Assistance Disclosure
Generative AI tools were used for code assistance, experiment-monitoring
support, language editing, and manuscript restructuring. The author re-
tained responsibility for experimental design, execution authorization, evi-
dence verification, statistical interpretation, and every scientific claim. AI-
generated text was checked against the frozen result artifacts and cited pri-
mary sources.
Data, Code, F unding, and Competing Interests
The public reproduction repository contains the S2G Judge/Search-R1 im-
plementation, frozen protocols, aggregate results, and paper source: https:
//github.com/luobostorm/search-r1-s2g-stopping . Per-question tra-
jectories, benchmark labels, and model weights are not redistributed. Model
13

checkpoints and benchmark data remain subject to their original licenses.
No external funding is declared. No competing interests are declared.
References
[1] Harsh T rivedi, Niranjan Balasubramanian, T ushar Khot, and Ashish
Sabharwal. Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. In Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (V ol-
ume 1: Long Papers) , 2023. doi: 10.18653/v1/2023.acl-long.557. URL
https://aclanthology.org/2023.acl-long.557/ .
[2] Akari Asai, Zeqiu W u, Yizhong W ang, A virup Sil, and Hannaneh Ha-
jishirzi. Self-RAG: Learning to retrieve, generate, and critique through
self-reflection. arXiv preprint arXiv:2310.11511 , 2023. doi: 10.48550/a
rXiv.2310.11511. URL https://arxiv.org/abs/2310.11511 .
[3] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong
Park. Adaptive-RAG: Learning to adapt retrieval-augmented large lan-
guage models through question complexity . In Proceedings of the 2024
Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language T echnologies (V olume 1: Long
Papers) , pages 7036–7050, 2024. doi: 10.18653/v1/2024.naacl-long.389.
URL https://aclanthology.org/2024.naacl-long.389/ .
[4] Bowen Jin, Hansi Zeng, Zhenrui Y ue, Dong W ang, Hamed Zamani, and
Jiawei Han. Search-R1: T raining LLMs to reason and leverage search
engines with reinforcement learning. arXiv preprint arXiv:2503.09516 ,
2025. URL https://arxiv.org/abs/2503.09516 .
[5] Diji Y ang, Linda Zeng, Jinmeng Rao, and Yi Zhang. Knowing you
don’t know: Learning when to continue search in multi-round RAG
through self-practicing. arXiv preprint arXiv:2505.02811 , 2025. URL
https://arxiv.org/abs/2505.02811 .
[6] Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang, and Guodong Zhou.
S2G-RAG: Structured suﬀiciency and gap judging for iterative retrieval-
augmented QA. In Proceedings of the 64th Annual Meeting of the
Association for Computational Linguistics (V olume 1: Long Papers) ,
pages 25846–25862, 2026. doi: 10.18653/v1/2026.acl-long.1185. URL
https://aclanthology.org/2026.acl-long.1185/ .
[7] Adrian Kieback, Uyiosa Philip Amadasun, Aman Chadha, and Aaron
Elkins. T ASR: T raining-free adaptive stopping for iterative retrieval.
arXiv preprint arXiv:2606.13814 , 2026. doi: 10.48550/arXiv.2606.1381
4. URL https://arxiv.org/abs/2606.13814 .
14

[8] Guanting Dong, Jiajie Jin, Xiaoxi Li, Y utao Zhu, Zhicheng Dou, and
Ji-Rong W en. RAG-Critic: Leveraging automated critic-guided agentic
workflow for retrieval augmented generation. In Proceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (V ol-
ume 1: Long Papers) , pages 3551–3578, 2025. doi: 10.18653/v1/2025.a
cl-long.179. URL https://aclanthology.org/2025.acl-long.179/ .
[9] Y onatan Geifman and Ran El-Y aniv. Selective classification for deep
neural networks. In Advances in Neural Information Processing Systems ,
2017. URL https://arxiv.org/abs/1705.08500 .
[10] Mintong Kang, Nezihe Merve Gürel, Ning Y u, Dawn Song, and Bo Li. C-
RAG: Certified generation risks for retrieval-augmented language mod-
els. In Proceedings of the 41st International Conference on Machine
Learning , volume 235 of Proceedings of Machine Learning Research ,
2024. URL https://proceedings.mlr.press/v235/kang24a.html .
[11] Zhilin Y ang, Peng Qi, Saizheng Zhang, Y oshua Bengio, William Cohen,
Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A
dataset for diverse, explainable multi-hop question answering. In Pro-
ceedings of the 2018 Conference on Empirical Methods in Natural Lan-
guage Processing , pages 2369–2380, 2018. doi: 10.18653/v1/D18-1259.
URL https://aclanthology.org/D18-1259/ .
A Descriptive F ull-Set T able
T able 3: Descriptive results on indices 0–999. These values are not the
confirmatory analysis.
System EM Searches STOP P STOP R STOP AP
Native Search-R1 0.4490 2.6100 – – –
Structured Base 0.4490 2.6090 0.0000 0.0000 0.3661
Old S2G LoRA 0.4490 2.6090 1.0000 0.0015 0.3694
Expanded S2G LoRA 0.4430 2.5210 0.6163 0.0805 0.3987
B F rozen Claim Boundaries
• Supported: the expanded S2G judge trained for Search-R1 reduces re-
trieval calls on the confirmatory test set while keeping the loss in answer
accuracy within the two-percentage-point range specified before evalua-
tion.
15

• Unsupported: the policy improves answer quality , certifies low STOP risk,
dominates all fixed or matched-cost baselines, or lowers total inference
cost.
• Scope: S2G-code-aligned HotpotQA distractor development indices 200–
999, the frozen Search-R1 7B/E5/Wiki-2018/top-3/four-search configu-
ration, and the frozen expanded adapter and threshold.
16