# BELIEFRAG: Making Adaptive RAG State-Aware under Evolving Evidence

**Authors**: Hongji Pu

**Published**: 2026-09-30 07:08:27

**PDF URL**: [https://arxiv.org/pdf/2609.39139v1](https://arxiv.org/pdf/2609.39139v1)

## Abstract
Adaptive RAG uses signals such as confidence, relevance, support, and retrieval quality to decide when to search or correct evidence. In multi-step retrieval, however, these local signals must be combined into a persistent view of what the current evidence supports, what remains missing, and which action should follow. Existing methods often use such signals as separate triggers, making it difficult to preserve a coherent evidence state across a trajectory; we call this problem evidence-state fragmentation. We introduce BELIEFRAG, a closed-loop controller that updates an explicit state over sufficiency, reliability, conflict, uncertainty, evidence gaps, and acquisition cost, then chooses among retrieval, query rewriting, verification, answering, stopping, and abstention. Across six QA benchmarks with gpt-oss-120b, BELIEFRAG reaches mean token F1 0.572 with 3.89k tokens per question, outperforming fixed iterative retrieval (0.555 F1) while using 39% fewer tokens. The same quality-cost pattern transfers to Qwen3-32B, where BELIEFRAG reaches 0.552 F1 versus 0.523 for iterative retrieval while using 35% fewer tokens. Analysis shows that the main gains come from corrective re-retrieval rather than pruning alone, while several belief dimensions are redundant and calibrated answerability plays the strongest operational role. Calibration improves threshold stability across related evidence sources, although source shift can still invalidate the same decision signal.

## Full Text


<!-- PDF content starts -->

BELIEFRAG: Making Adaptive RAG State-Aware under Evolving
Evidence
Hongji Pu
University of Illinois Urbana-Champaign
hongjip2@illinois.edu
Abstract
Adaptive RAG uses signals such as confi-
dence, relevance, support, and retrieval qual-
ity to decide when to search or correct ev-
idence. In multi-step retrieval, however,
these local signals must be combined into
a persistent view of what the current evi-
dence supports, what remains missing, and
which action should follow. Existing meth-
ods often use such signals as separate trig-
gers, making it difficult to preserve a coher-
ent evidence state across a trajectory; we
call this problemevidence-state fragmenta-
tion. We introduce BELIEFRAG, a closed-
loop controller that updates an explicit state
over sufficiency, reliability, conflict, uncer-
tainty, evidence gaps, and acquisition cost,
then chooses among retrieval, query rewrit-
ing, verification, answering, stopping, and
abstention. Across six QA benchmarks with
gpt-oss-120b, BELIEFRAG reaches mean to-
ken F1 0.572 with3.89k tokens per question,
outperforming fixed iterative retrieval ( 0.555
F1) while using 39% fewer tokens. The same
quality–cost pattern transfers to Qwen3-32B,
where BELIEFRAG reaches 0.552 F1 ver-
sus0.523 for iterative retrieval while using
35% fewer tokens. Analysis shows that the
main gains come from corrective re-retrieval
rather than pruning alone, while several be-
lief dimensions are redundant and calibrated
answerability plays the strongest operational
role. Calibration improves threshold stabil-
ity across related evidence sources, although
source shift can still invalidate the same de-
cision signal.
1 Introduction
Retrieval-augmented generation (RAG) equips lan-
guage models with external evidence that can sup-
plement knowledge stored in model parameters
(Lewis et al., 2020). The standard RAG pipeline
retrieves a fixed set of passages once and gener-
ates an answer from the retrieved context (Lewiset al., 2020). Multi-hop questions often require
several pieces of evidence that become available
only after intermediate entities or facts have been
identified (Trivedi et al., 2022). A retrieval system
for these questions therefore needs to decide re-
peatedly what information is still missing, whether
another search is useful, and when the collected
evidence is sufficient for answering.
Adaptive RAG introduces such decisions into
the retrieval process. FLARE triggers retrieval
from low confidence during generation (Jiang et al.,
2023). Self-RAG learns reflection signals for re-
trieval need, evidence relevance, answer support,
and response utility (Asai et al., 2024). Adaptive-
RAG predicts question complexity and selects no
retrieval, one-step retrieval, or iterative retrieval
(Jeong et al., 2024). DRAGIN estimates the infor-
mation currently needed by the model and uses that
estimate to determine retrieval timing and queries
(Su et al., 2024). CRAG evaluates retrieved ev-
idence and invokes corrective retrieval when the
initial evidence is judged poor (Yan et al., 2024).
Together, these methods establish confidence, rel-
evance, support, information need, and retrieval
quality as useful signals for adaptive retrieval.
A shared problem appears once retrieval lasts for
several steps:the controller must convert several
local signals into one decision about the current
evidence. The same low-confidence state can arise
because a required fact is missing, because the
retained passages disagree, or because the avail-
able evidence is too weak to support an answer.
These situations require different actions. Miss-
ing information motivates further retrieval or query
rewriting. Conflicting evidence motivates verifi-
cation. Sufficient evidence motivates termination.
Low-value future retrieval motivates stopping even
when the evidence remains imperfect. Partially ob-
servable decision problems commonly address this
type of setting by maintaining a belief over infor-
mation that cannot be observed directly (Kaelbling
arXiv:2609.39139v1  [cs.AI]  30 Sep 2026

et al., 1998).
The difficulty comes from the meaning and in-
teraction of the available signals. Self-RAG shows
that relevance and support provide distinct judg-
ments about retrieved evidence (Asai et al., 2024).
DRAGIN shows that retrieval can be driven by
the model’s current information need (Su et al.,
2024). Astute RAG studies unreliable evidence
and conflicts between retrieved and internal knowl-
edge (Wang et al., 2025). Adaptive-RAG demon-
strates that the useful amount of retrieval varies
across questions (Jeong et al., 2024). These sig-
nals describe different aspects of one evolving ev-
idence condition. Their values can also be redun-
dant, poorly calibrated, or weakly connected to
the controller’s actual actions. A useful diagnostic
therefore requires both informative measurements
and a decision rule that can consume those mea-
surements at values reached during real trajectories.
We call this problemevidence-state fragmen-
tation. A retrieval trajectory contains queries, pas-
sages, verification outcomes, and previous actions.
The controller needs a compact summary of what
these observations currently imply about the ev-
idence. BeliefRAG provides this summary as a
persistent evidence state. At each step, it measures
relevance, support, conflict, uncertainty, remaining
evidence gaps, novelty, and acquisition cost. These
observations update six operational belief dimen-
sions: sufficiency, reliability, conflict, uncertainty,
evidence gap, and cost. The six dimensions form
an explicit design hypothesis. Our experiments
test their incremental information and their actual
influence on controller actions.
The updated state controls six actions: RE-
TRIEVE, REWRITE, VERIFY, ANSWER, STOP, and
ABSTAIN. Evidence with material conflict or low
reliability can enter a corrective loop that verifies,
removes weak passages, rewrites the query, and
retrieves replacement evidence. Evidence with
high answerability can terminate with an answer.
Insufficient evidence can trigger another retrieval
when the estimated value of another search is high
enough. Low novelty can trigger query rewriting.
Low expected acquisition value can terminate fur-
ther search. Figure 1 summarizes this closed-loop
process.
We evaluate BeliefRAG under a controlled pro-
tocol in which compared methods share the same
corpus, retriever, verifier, budget, and evaluation
examples within each backbone. The primary ta-ble uses gpt-oss-120b: BeliefRAG reaches mean
token F1 0.572 with3.89k tokens per question, out-
performing fixed Iterative RAG at 0.555 F1 while
using 39% fewer tokens. We repeat the six-dataset
evaluation with Qwen3-32B in Appendix Tables 7–
8; there BeliefRAG reaches 0.552 mean F1 versus
0.523 for Iterative RAG with 35% fewer tokens.
The analyses show that corrective re-retrieval, cali-
brated answerability, and stable decision thresholds
explain the quality–cost trade-off, while evidence-
source shift remains a clear boundary.
Our contributions are threefold:
•Problem.We identifyevidence-state fragmen-
tation: multi-step RAG lacks a persistent state
that summarizes what multiple evidence signals
imply for the next action.
•Method.We proposeBeliefRAG, a closed-loop
controller that updates an explicit evidence state
and uses it to coordinate retrieval, rewriting, veri-
fication, answering, stopping, and abstention.
•Findings.Controlled experiments across two
backbones show a strong quality–cost trade-off
and reveal which mechanisms matter in practice:
corrective re-retrieval, calibrated answerability,
and transferable decision thresholds.
2 Related Work
Adaptive RAG.RAG retrieves evidence once be-
fore generation, leaving search depth fixed across
queries (Lewis et al., 2020). Self-RAG learns re-
flection tokens for retrieval, relevance, support,
and utility, but these judgments remain local to
the current step (Asai et al., 2024). CRAG uses
a retrieval-quality evaluator to trigger correction
when evidence is poor, making evaluator reliability
a key failure point (Yan et al., 2024). Adaptive-
RAG routes queries among no retrieval, one-shot
retrieval, and iterative retrieval, trading task diffi-
culty against unnecessary search cost (Jeong et al.,
2024).
Uncertainty and adaptive retrieval.Adap-
tive retrieval must decide whether more evidence
is still useful. Uncertainty is a natural signal,
but its reliability varies across tasks and estima-
tors (Moskvoretskii et al., 2025). In RAG, re-
trieved evidence can also change the meaning of
confidence, so standard uncertainty scores may no
longer track answer correctness (Soudani et al.,
2025). SeaKR uses internal model states to guide
retrieval and reranking (Yao et al., 2025), while
CtrlA learns control signals directly from represen-

Figure 1:BeliefRAG as a closed-loop evidence controller.The system converts the current evidence into
diagnostic signals and a compact belief state, then applies fixed branch rules to choose among retrieval, rewriting,
verification, answering, stopping, and abstention. Retrieval value and verifier feedback determine whether more
evidence is worth acquiring and update the next iteration. The controller is non-RL; only the sufficiency and
retrieval-value estimates are fitted from data.
tations (Huanshuo et al., 2025). These results sug-
gest that no single confidence signal is uniformly
reliable.
Evidence quality and conflict.Topically rele-
vant evidence can still be misleading, and models
may over-weight relevance when judging its quality
(Wan et al., 2024). Retrieved evidence can conflict
with parametric knowledge, biasing models toward
faulty internal memory (Jin et al., 2024). Conflict-
ing sources can also substantially degrade RAG
performance (Pham et al., 2024). Retrieved context
can make standard uncertainty estimates unreliable
(Soudani et al., 2025). BeliefRAG therefore tracks
evidence quality, conflict, and answerability across
steps rather than relying on one relevance or confi-
dence score.
3 Methodology
3.1 Overview
BeliefRAG controlswhat to do after each retrieval.
A retrieved passage being relevant does not yet
mean that the question can be answered: a required
fact may still be missing, two passages may dis-
agree, or another search may simply repeat what
is already known. The controller therefore makes
three decisions from the evidence accumulated sofar:should the evidence be corrected, is it sufficient
to answer, and if not, is another retrieval worth do-
ing?
At stept, the agent maintains
st= (q, E t, Ht, bt),(1)
where qis the question, Etis the evidence currently
retained, Htrecords previous queries and actions,
andbtsummarizes the current evidence condition.
Each step follows
measureE t→updateb t
→choose an action→updateE t.
Importantly, Etis not an append-only retrieval his-
tory: RETRIEVEcan add passages, while VERIFY
can remove passages judged unhelpful.
3.2 Measuring the Current Evidence
Before making a decision, BeliefRAG computes
seven diagnostics,
xt= [R t, St, Ct, Ut, Gt, Nt, Kt].(2)
They answer concrete questions about the evidence
and are computed as shown in Table 1.
The four semantic quantities St, Ct, Ut, Gtare
produced together by one structured verifier call.

Table 1: Evidence diagnostics used at each decision
step.
Meaning Computation
RtRelevance Softmax-weighted mean of calibrated
top-kretrieval scores.
StSupport Verifier score for how strongly Etsup-
ports a complete answer.
CtConflict Largest material contradiction within Et
or between evidence and the current
draft.
UtUncertainty Verifier estimate of how uncertain the
answer remains given onlyE t.
GtGap Estimated fraction of information re-
quired by the question that is still un-
supported.
NtNovelty1−sim(∆E t, Et−1); high when the
newest retrieval adds information not al-
ready retained.
KtCostmin(1,tokens used/token budget).
Rt,Nt, andKtare computed locally. For example,
the raw retriever score zt,iof passage iat step tis
first mapped to[0,1]by
˜zt,i=σ((z t,i−µs)/τs),
where σ(z) = 1/(1 +e−z)is the logistic sig-
moid, µsis the retriever-score location statistic, and
τs>0is its scale. Rtis then the softmax-weighted
mean of the top- kcalibrated scores, where kis the
number of passages returned by one retrieval call.
In Table 1, ∆Etdenotes passages newly returned at
stept, and simis the configured passage-similarity
function. Novelty compares these new passages
with those already retained. Cost is not predicted
by the model: it is the fraction of the token budget
already consumed. Thus retrieval rounds, top- k,
action count, and token budget remain different
constraints rather than one vague “capacity” vari-
able.
3.3 From Diagnostics to Belief
The diagnostics describe individual properties of
Et. BeliefRAG converts them into six quantities
with direct decision meanings:
bt= [bsuff
t, brel
t, bconf
t, bunc
t, bgap
t, bcost
t].(3)
Sufficiencymeans that the retained evidence
is enough to answer;reliabilitymeans that the
retained sources appear trustworthy;conflictmeans
that the evidence materially disagrees;uncertainty
means that the answer is still unclear;gapmeans
that required information is still missing; andcost
records how much acquisition budget has already
been spent.These quantities are related but not interchange-
able. For example, a passage may be highly rele-
vant and well supported but still leave the second
hop of a multi-hop question unresolved. Similarly,
two individually relevant passages may contradict
each other. Sufficiency therefore asks a higher-
level question than relevance or support:can the
question be answered correctly from the evidence
retained now?
For each inferred dimension, the complete diag-
nostic vector is mapped to an instantaneous belief.
Here αdis a dimension-specific intercept, wdis
its diagnostic-weight vector, and dindexes the five
inferred (non-cost) dimensions:
ˆbd
t=σ(α d+w⊤
dxt),
d∈ {suff,rel,conf,unc,gap}.(4)
For example, sufficiency increases with support
and relevance and decreases with evidence gap,
uncertainty, and conflict; reliability increases with
relevance and support but decreases with conflict.
Cost is observed directly:bcost
t=K t.
Because one verifier call can be noisy, the new
estimate is blended with the previous belief, where
λd∈[0,1] is the weight placed on the current
observation in logit space:
bd
t=σ 
(1−λ d) logit(bd
t−1)
+λdlogit( ˆbd
t)
.(5)
The exact coefficients and initial values are re-
ported in Appendix A.
Operational answerability.The persistent belief
coordinate bsuff
tand the answer gate are distinct.
The former summarizes evidence sufficiency; the
latter uses
pans
t=P(answerable|q, E t),(6)
where “answerable” means that the frozen gener-
ator produces a correct answer under the fixed
prompt. Because that prompt requests a best
short answer even when documents are incom-
plete, pans
tmay reflect frozen parametric knowl-
edge and is not evidence-only entailment. We fit
pans
t=σ(α ans+w⊤
ansxt)on HotpotQA train, se-
lect it on development data, and freeze it across
all six benchmarks; Appendix A.3 gives the fitted
parameters.

Table 2: Main decision rules. Values are the default op-
erating thresholds; experiment-specific selected values
are reported in the Appendix.
Decision Condition Default
Correct Evidence is materially conflict-
ing or clearly unreliablebconf
t≥0.50
orbrel
t<0.30
Answer Current state is operationally an-
swerable and conflict is accept-
ablepans
t≥0.50
Retrieve Evidence is insufficient, budget
remains, and another retrieval
has enough chance to make it an-
swerablepflip
t≥0.10
Rewrite Another retrieval is useful, but
the previous retrieval added little
new informationNt<0.20
Stop / Abstain Evidence is still insufficient and
further acquisition has low ex-
pected valueno useful ac-
quisition
3.4 How Beliefs Produce Actions
BeliefRAG chooses among six actions:
A={RETRIEVE,REWRITE,VERIFY,
ANSWER,STOP,ABSTAIN}.
Rather than scoring them as unrelated choices, the
main controller evaluates a short sequence of ques-
tions shown in Table 2.
Correcting bad evidence.The controller checks
correction before answering. A high conflict score
means that the retained passages materially dis-
agree; low reliability means that their combined
relevance/support pattern is not trustworthy enough.
The verifier can also return identifiers of passages
it considers unhelpful; when that trigger is enabled,
those passages provide an additional correction sig-
nal.
Correction is not simply deletion:
VERIFY→prune→REWRITE→RETRIEVE.
Verification first removes the problematic passages,
then the query is reformulated and a replacement
retrieval is issued. This design matters because
deleting weak evidence without replacing the miss-
ing information may leave the question even less
answerable.
Deciding when to answer.After correction is
considered, the controller checks pans
t. The default
threshold is τans= 0.50 : under the probability
interpretation, the current evidence must be at least
as likely to be answerable as not. The threshold is
an operating point rather than a universal constant;
when it is selected on a development/selection split,
it is frozen before final evaluation.Deciding whether to retrieve again.If pans
t<
τans, the agent does not automatically retrieve
merely because it is uncertain. It asks a second
question:is one more retrieval likely to change the
state from insufficient to sufficient?We estimate
pflip
t=σ
αflip+wflip
rrt+wflip
NNt+wflip
sbsuff
t
,
(7)
where rtis the number of retrieval rounds already
used and αflip, wflip
r, wflip
N, wflip
sare retrieval-value
coefficients fitted on the fit split (or replaced by the
fixed fallback schedule reported in the Appendix).
The default minimum acquisition value is 0.10.
Thus another search is attempted only when it has
at least the required estimated chance of making the
evidence answerable and retrieval budget remains.
Novelty then decideshowto continue. If the
previous retrieval added useful new information
(Nt≥0.20 by default), the controller can issue an-
other retrieval. If novelty is below 0.20, repeating
essentially the same query is unlikely to help, so
the controller prefers REWRITEbefore searching
again.
Stopping and abstention.If the evidence is not
sufficiently answerable and another acquisition has
low value, the controller stops spending the remain-
ing budget. STOPmeans “use the evidence we have
and produce the best answer”; ABSTAINmeans
“the evidence is inadequate and no useful acquisi-
tion remains.” Low confidence alone is therefore
not an abstention rule: uncertainty must be com-
bined with evidence insufficiency and low acquisi-
tion value.
3.5 Controlled Evaluation
The retriever, generator, verifier, answer prompt,
evaluator, and budget are held fixed across com-
pared controllers. The main answerability cali-
brator is fitted once on HotpotQA train and se-
lected on HotpotQA development data, then reused
unchanged across all six benchmarks. Separate
fit/selection/holdout partitions are used for the
causal analyses. Exact thresholds, fitted coeffi-
cients, prompts, budget limits, and reproduction
details are given in Appendix A.
4 Experiments
We evaluate BELIEFRAG in one shared harness so
that differences come from the retrieval controller
rather than from different tools or prompts. The
language model acts as the agent’s generator: it

writes search queries and final answers, but it can-
not retrieve documents or execute actions by itself.
The harness executes every RETRIEVE, REWRITE,
VERIFY, ANSWER, STOP, or ABSTAINaction and
records the resulting evidence, belief state, and
cost. All compared methods therefore share the
same generator, retriever, verifier, answer prompt,
evaluator, and budget.
4.1 Tasks and Benchmarks
The main experiment uses six QA benchmarks
with two different roles. HotpotQA (Yang et al.,
2018), 2WikiMultiHopQA (Ho et al., 2020), and
MuSiQue (Trivedi et al., 2022) are multi-hop tasks:
the answer usually depends on connecting more
than one fact, so the first retrieval can be rele-
vant but still incomplete. They test whether a con-
troller knows when more evidence is needed. Nat-
ural Questions (Kwiatkowski et al., 2019), Trivi-
aQA (Joshi et al., 2017), and PopQA (Mallen et al.,
2023) are open-domain factoid tasks. These are
useful controls because one retrieval—or even the
model’s parametric knowledge—may already be
enough, leaving less room for multi-step control.
Every main-table cell evaluates the same n= 100
questions for a dataset.
We use two additional diagnostic benchmarks
outside the six-dataset average. RGB (Chen et al.,
2024) replaces clean evidence with irrelevant or de-
liberately misleading passages, so it tests whether
the controller can distinguish “more evidence”
from “better evidence.” HoloBench (Maekawa
et al., 2025) asks for sets of database rows rather
than a single fact, so it tests whether a retrieval
strategy can recover enough distinct items without
reading the whole candidate pool. These diagnostic
numbers are reported separately because their met-
rics are not directly comparable with QA accuracy.
4.2 Baselines and Backbones
The primary comparison usesgpt-oss-120b(Ope-
nAI, 2025). We also repeat the six-dataset eval-
uation withQwen3-32B(Yang et al., 2025) as a
second backbone; those results are reported in Ap-
pendix Tables 7–8. Within each backbone, the
model writes search queries and final answers,
while the shared harness executes retrieval and ev-
ery controller action. The Qwen study uses the
same HotpotQA train/dev calibration protocol, re-
fitted for that backbone and then frozen across all
six evaluation datasets. The baselines isolate in-
creasingly adaptive forms of retrieval control.No-RAGis the closed-book control (Brown et al.,
2020).Static RAGretrieves once and then an-
swers (Lewis et al., 2020).Iterative RAGal-
ways follows the fixed multi-round schedule, show-
ing what brute-force extra search can buy (Jiang
et al., 2023).Adaptive-kretrieves once but varies
how many passages are kept from the score dis-
tribution (Taguchi et al., 2025), whileAdaptive-
RAGroutes each question once to no, single-step,
or iterative retrieval based on predicted complex-
ity (Jeong et al., 2024).Self-RAG*is a common-
harness retrieval-trigger variant that isolates the de-
cision to retrieve again (Asai et al., 2024);CRAG
evaluates the current evidence and, when it is poor,
prunes, rewrites, and retrieves again (Yan et al.,
2024).RL-Searchis a prompted multi-step search
controller using the Search-R1/R3-RAG interac-
tion format, not an RL-trained reproduction (Jin
et al., 2025; Li et al., 2025). BELIEFRAG uses
the evolving evidence state to decide whether to
correct, retrieve again, rewrite, answer, stop, or
abstain.
Unless an analysis states otherwise, every re-
trieving method can make at most three retrieval
calls, receives top- k= 5 passages per call, can
take at most six controller actions, and has a 12k-
token budget. These limits are separate: retrieval
rounds control how often search can occur, top- k
controls how many passages one search returns,
the action limit caps controller decisions, and the
token budget measures the total text processed by
the model.
4.3 Metrics and Evaluation Protocol
The primary QA metric for the benchmark and
cross-backbone comparisons is normalized token-
level answer F1, a deterministic overlap score al-
ready recorded by the harness. We use F1 for these
headline comparisons because it does not depend
on an LLM judge. Exact match (EM) is a stricter
secondary answer metric, while LLM-judged se-
mantic accuracy is reported as a supplementary
semantic check. The judge shares the evaluated
backbone, so acc_judge is interpreted within a
backbone and is not used for cross-backbone com-
parisons; headline transfer claims use deterministic
token F1. Controlled mechanism and robustness
analyses retain semantic ACC where their interven-
tions were originally evaluated with that metric,
and label it explicitly. Efficiency is measured by

mean total tokens per question,
Tokens =1
NNX
i=1Ti,(8)
where Tiincludes retrieved text each time it is sent
to the model, verifier calls, query rewriting, policy
prompts, and final answer generation. We also re-
port retrieval rounds, action counts, and evidence
recall when supporting-fact annotations exist. Evi-
dence recall is the fraction of annotated supporting
facts recovered in the retained evidence.
For HoloBench,row recallis the fraction of gold
rows recovered, whilerecall per row readdivides
row recall by the mean number of rows inspected
(scaled by 103in the figure). The latter measures
retrieval efficiency rather than answer quality. The
complementary HoloBench result is reported in
Appendix Figure 7, separate from the six-dataset
QA comparison.
For signal analyses, AUC measures how well
a score ranks positive states above negative ones,
while expected calibration error (ECE) measures
how closely predicted probabilities match empiri-
cal frequencies; lower ECE is better. Paired com-
parisons always use the same questions. Confi-
dence intervals are obtained by bootstrap resam-
pling over questions; exact aggregation details are
in Appendix D.6.
Calibration and evaluation are separated. For
the main results, we fit a logistic answerabil-
ity calibrator over the full diagnostic vector on
300 HotpotQA training examples, select the cal-
ibrator configuration on 156 HotpotQA develop-
ment examples by Brier score, and then freeze
it for all six datasets without per-dataset re-
tuning. Dedicated causal analyses use disjoint
xfit /xsel /xhold partitions so that fitted map-
pings, operating choices, and final evaluation re-
main separated; Appendix D.5 gives the exact pro-
tocols and contamination checks.
5 Results
5.1 Main Result: Quality and Cost
Table 3 compares all nine methods under the same
gpt-oss-120b generator, retriever, verifier, prompt,
and evaluation set. Figure 2 plots the same compar-
ison against token use.
BELIEFRAG achieves the highest mean F1
while using substantially less retrieval compute.
BELIEFRAG reaches mean F1 0.572 versus 0.555
0 1 2 3 4 5 6 7
mean tokens per question (thousands)0.200.250.300.350.400.450.500.550.60mean token F1 over six datasets+0.017 mean F1 at
39% fewer tokens
Pareto frontier(a)  F1–cost trade-off
No-RAGStatic
Adaptive-kAdaptive-RAGIterativeCRAG
Self-RAG*
RL-SearchBeliefRAGFigure 2:F1–cost trade-off.Each point is a method
average over the six QA datasets in Table 3.
for fixed Iterative RAG, while using 3.89k rather
than6.35k tokens per question—about 39% less.
BELIEFRAG is higher on four of six datasets (Hot-
potQA, 2Wiki, MuSiQue, and NQ), while Iterative
RAG is higher on TriviaQA and PopQA. The re-
sult therefore reflects a quality–cost improvement
rather than uniform gains on every dataset.
The same pattern is stronger with a sec-
ond backbone.With Qwen3-32B, BELIEFRAG
reaches 0.552 mean F1 versus 0.523 for Iterative
RAG while using 3.78k versus 5.81k tokens, a 35%
reduction. It is higher on five of six datasets; full
F1, EM, LLM-judge, token, and retrieval results
are reported in Appendix Tables 7–8.
5.2 Analysis 1: Which Parts Create the Gain?
This analysis separates two roles of the controller.
Panel (a) asks whether detecting weak evidence
is enough; Panels (b–c) intervene on calibrated
answerability pans
t(legacy run label: “sufficiency”),
not onbsuff
t.
The gain comes from replacing weak evidence,
not just detecting it.Figure 3(a) shows that the
full controller reaches 0.680 multi-hop ACC. Re-
moving correction, removing verification, or prun-
ing weak evidence without replacing it all reduce
ACC to 0.630 . This means that identifying a bad
passage is not enough. The controller improves
only when it removes weak evidence and retrieves
a replacement that fills the missing information.
The live answerability signal helps the con-
troller stop once the current state becomes an-
swerable.With the answer threshold fixed at
τ= 0.550 , live pans
treaches 0.685 ACC across
the three multi-hop datasets. Permuting that score
across matched states lowers ACC to 0.650 , while
freezing it throughout the trajectory gives 0.655 .
The live score also uses fewer retrieval rounds:

Table 3:Main comparison on six QA benchmarks with gpt-oss-120b (OpenAI, 2025) as the language
backbone.The primary metric is normalized token-level answer F1; all cells use n= 100 questions. Mean is the
simple average across the six datasets and Tokens is mean total token cost per question. Cell color shows the F1
change from Static RAG on the same dataset. Bold marks the best F1 in each column.
MethodMulti-hop QA Open-domain QAMean Tokens
HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA
No-RAG (Brown et al., 2020) .378 .411 .167 .341 .766 .381 .407 0.10k
Static RAG (Lewis et al., 2020) .559 .681 .423 .413 .681 .324 .514 2.82k
Adaptive-k(Taguchi et al., 2025) .535 .558 .360 .347 .715 .348 .477 3.04k
Adaptive-RAG (Jeong et al., 2024) .575 .711 .445 .425 .751 .324 .538 3.59k
Iterative RAG (Jiang et al., 2023) .588 .710 .420 .420 .750 .440 .555 6.35k
CRAG (Yan et al., 2024) .573 .725 .433 .423 .709 .454 .553 4.97k
Self-RAG* (Asai et al., 2024) .482 .654 .356 .246 .100 .282 .353 3.17k
RL-Search (Jin et al., 2025; Li et al., 2025) .303 .308 .223 .102 .275 .230 .240 3.17k
BELIEFRAG .602 .766 .495 .441 .716 .410 .5723.89k
Color bins use absolute F1 change from Static RAG: light [.01, .05) , medium [.05, .10) , dark≥.10 ; symmetric bins are used for losses. All
rows use the same gpt-oss-120b evaluation questions; No-RAG, Adaptive- k, and Adaptive-RAG are from the same evaluation run as the other
rows. Self-RAG* and RL-Search are common-harness mechanism variants rather than full reproductions; see Section 4.
1.48 on average, compared with 1.81 for both con-
trols. This intervention isolates the calibrated an-
swer gate; it does not manipulatebsuff
t.
Key takeaway.The gain comes from replac-
ing weak evidence, not merely detecting it. Cali-
brated answerability then tells the controller when
to stop searching and answer.
5.3 Analysis 2: Do Thresholds Keep the Same
Meaning?
The controller answers when a score crosses a
threshold, so transfer requires comparable score
meaning across datasets. We compare raw and cali-
brated answerability scales and replay visited states
to test gate reachability.
Calibration matters because thresholds de-
pend on the scale of the score they use.Fig-
ure 4(a) shows that the raw answerability score
has substantially different medians across datasets,
with a spread of 0.352 . We fit a logistic calibra-
tor over the full diagnostic vector on HotpotQA
train, select it on HotpotQA development data by
Brier score, and then freeze it across all six datasets
without retuning. Calibration reduces the median
spread to 0.048 . Panel (b) shows the consequence:
the raw threshold occupies very different parts of
the score distribution across datasets, whereas the
calibrated threshold lies in a more consistent oper-
ating region.
A configured gate matters only if real trajec-
tories can reach it.Panel (c) measures how often
gate conditions are satisfied on visited states. Thecalibrated answer gate fires on 67.7% of audited
main-table states. The conflict gate fires on 72%
of counterfactual-evidence cases but only 8%of
ordinary QA. This is the intended behavior of a
specialized signal: it remains quiet when the fail-
ure is absent and becomes active when that failure
appears. A gate that stays near 0%or100% across
inputs would contribute little adaptive behavior.
Key takeaway.A fixed threshold is useful only
if its score keeps the same meaning. Calibration
stabilizes that meaning, while reachability checks
whether the gate actually affects visited states.
5.4 Analysis 3: What Information Is in the
Belief State?
The six belief dimensions summarize different as-
pects of the same evidence set. We therefore ask
both whether each dimension predicts correctness
and whether it contributes information beyond the
others. Figure 5 combines single-signal prediction,
dependence, calibration, and action patterns across
belief levels.
The joint state is more informative than any
single dimension, but the signals are correlated
and unevenly useful.Panel (a) measures how well
each signal ranks correct versus incorrect states.
The joint state reaches AUC 0.816 , compared with
0.761 for the strongest single signal, uncertainty.
Uncertainty, gap, and sufficiency are the strongest
individual predictors; conflict is also informative,
while reliability and cost are weaker. Panel (b)
shows that several dimensions are correlated be-

0.62 0.64 0.66 0.68
accuracy on the three multi-hop setsFULL
NO-CORRECTION
NO-VERIFIER
PRUNE-ONLY0.680 3.95k
0.630-0.050
2.81k
-1.14k
0.630-0.050
2.81k
-1.14k
0.630-0.050
2.73k
-1.21kcomplete system
corrective re-retrieval removed
answer verification removed
evidence replacement removedtokens(a)  Retrieval-control ablations
HotpotQA 2Wiki MuSiQue0.00.20.40.60.81.0accuracy (acc_judge)
0.680
0.823
0.5520.646
0.785
0.5170.652
0.790
0.5230.641
0.629
0.419(b)  Belief-consumption ablations
HotpotQA 2Wiki MuSiQue0.00.51.01.52.02.5retrieval rounds used
1.46
1.36
1.611.86
1.65
1.941.86
1.65
1.941.00
1.00
1.00BeliefRAG 1.48 rounds
vs controls 1.81(c)  Retrieval rounds
BeliefRAG, gate at τ=0.550 sufficiency permuted (control) sufficiency frozen (control) gate at τ=0.50 (degenerate)Figure 3:Mechanism ablations.(a) Retrieval-control ablations. (b–c) Interventions on pans
t; “sufficiency” is the
legacy run label.
HotpotQA2WikiMuSiQueNQTriviaQA PopQA0.00.20.40.60.8sufficiency at the terminal statefrozen
τ = 0.020frozen
τ = 0.550across datasets the median moves
0.352 when the signal is raw,
and 0.048 once it is calibrated(a)  Signal scale across datasets
RAW
CALIBRATED
HotpotQA2WikiMuSiQueNQ
TriviaQAPopQA05101520253035questions below the threshold (%)8.6
0.61.83.7
0.522.5
0.5%–22.5%
spread 46×24.628.030.0
22.4
20.026.3
20.0%–30.0%
spread 2×(b)  Threshold operating points
RAW CALIBRATED
1st percentile median 99th percentile
position inside that gate's own realized signal rangeτ
fires on 8.0% of runs n = 300conflict gate
τ=0.50, ordinary QA
τ
fires on 72.0% of runs n = 100conflict gate
τ=0.50, counterfactual evidence
τ
fires on 34.0% of runs n = 800answer gate
τ=0.50 default, raw sufficiency on RGB
τ
fires on 67.7% of runs n = 300answer gate
τ=0.50, calibrated, main table
τ
fires on 78.0% of runs n = 1097answer gate
τ=0.020, raw arm, held-out split
τ
fires on 19.3% of runs n = 300abstain gate
τ=0.520, any run(c)  Gate reachability
live: reachable, and decides between 5% and 95% of runs decided in advance: fires on < 5% or > 95% of runs dead: lies outside the signal's range
Figure 4:Threshold transfer and reachability.(a) Answerability scale. (b) Operating points. (c) Gate reachability.
“Sufficiency” is the legacy label for this score.
cause they reflect shared problems such as missing
evidence. Their value therefore comes from com-
plementary information and downstream use, not
from signal count alone.
Useful belief scores should be calibrated and
linked to different controller behavior.Panel (c)
shows that sufficiency and reliability are well cali-
brated, with ECE values of 0.039 and0.045 . Panel
(d) shows the corresponding behavioral associa-
tion: higher sufficiency and reliability are followed
more often by answering, whereas higher uncer-
tainty, gap, and cost are associated with stopping or
further retrieval. These results establish interpreta-
tion and association for the belief dimensions. The
intervention in Figure 3 instead tests the separate
calibrated answerability score pans
t; it should not
be read as a causal intervention onbsuff
t.Key takeaway.The value of a belief signal comes
from added information, calibrated meaning, and
a useful effect on decisions–not from the number
of state dimensions.
5.5 Analysis 4: What Happens When
Evidence Is Imperfect?
RGB separates three evidence failures often con-
flated in QA. Irrelevant noise adds useless but
non-false passages; counterfactual evidence ac-
tively supports a wrong answer; source transfer
tests whether the same answerability estimator
still works when evidence comes from a differ-
ent source. These settings probe distinct failure
modes and should not be reduced to one “robust-
ness” score.
Ignoring irrelevant evidence is easier than re-
sisting plausible but false evidence.Figure 6(a)
replaces retrieved passages with irrelevant docu-

all six togetheruncertaintygap
sufficiencyconflictreliabilitycost0.500.550.600.650.700.750.800.85AUC against answer correctness0.816
0.761 0.758 0.756
0.735
0.683
0.621
chance+0.054 over the best single dimension(a)  Signal informativeness
predicts success
predicts failure (inverted)
sufficiency reliabilityconflictuncertaintygap costsufficiency
reliability
conflict
uncertainty
gap
cost1.00 0.53 -0.40 -0.66 -0.64 -0.31
0.53 1.00 -0.36 -0.46 -0.46 0.10
-0.40 -0.36 1.00 0.45 0.40 0.24
-0.66 -0.46 0.45 1.00 0.66 0.44
-0.64 -0.46 0.40 0.66 1.00 0.39
-0.31 0.10 0.24 0.44 0.39 1.00(b)  Signal redundancy
0.0 0.2 0.4 0.6 0.8 1.0
belief value (bin mean)0.00.20.40.60.81.0observed P(answer correct)
rising: predicts the answer will be correct
falling: predicts it will not(c)  Signal calibration
perfect calibration
sufficiency   ECE 0.039
reliability   ECE 0.045
uncertainty
gap
cost
0.0 0.5 1.00.000.250.500.751.00P(action | bin)
higher ) answersufficiency
0.0 0.5 1.0
higher ) answerreliability
0.0 0.5 1.0
belief value (bin mean), four bins per dimension, n≥63 per bin
higher ) abstainconflict
0.0 0.5 1.0
higher ) stopuncertainty
0.0 0.5 1.0
higher ) stopgap
0.0 0.5 1.0
higher ) stopcost−1.00−0.75−0.50−0.250.000.250.500.751.00
Pearson r
(d)  Actions by belief level abstain answer stopFigure 5:Belief-state diagnostics.(a) Signal usefulness. (b) Signal dependence. (c) Signal calibration. (d) Actions
by belief level.
0 20 40 60 80
irrelevant documents in the evidence (%)0.760.780.800.820.840.860.880.900.92accuracy (acc_judge)
0.80
0.81
0.820.87(a)  Irrelevant noise
Static  (drop 0.08)
Iterative  (drop 0.07)
CRAG  (drop 0.05)
BeliefRAG  (drop 0.03)
0.0 0.2 0.4 0.6 0.8 1.0
share of 100 questionsStatic
Iterative
CRAG
BeliefRAG0.18
0.22
0.25
0.300.06
0.180.41
0.38
0.35
0.250.38
0.37
0.34
0.27(b)  Misleading evidence
TRUE — answered correctly anyway
FAKE — repeated the false answerabstain — refused to answer
other — something else
0.5
chance0.6 0.7 0.8 0.9
AUC: calibrated sufficiency → answer correct2Wiki
TriviaQA
PopQA
HotpotQA
MuSiQue
NQ
reject
integrate
noise80
counterfactual
noise00wiki18 QA (in domain)   —   mean 0.775
0.828
0.816
0.809
0.773
0.746
0.676
RGB web snippets (transferred)   —   mean 0.651
0.691
0.674
0.652
0.629
0.610(c)  Evidence-source transfer
Figure 6:Robustness to imperfect evidence.(a) Irrelevant noise. (b) Misleading evidence. (c) Evidence-source
transfer.
ments. At 80% injection, BELIEFRAG retains
0.87 ACC, versus 0.80 for Static, 0.81 for Iterative,
and0.82 for CRAG, only three points below clean
performance. Panel (b) is harder: BELIEFRAG
follows the false answer on25%of questions, ver-
sus35–41% for the baselines, while recovering the
true answer on 30%. Robustness to irrelevant text
therefore does not imply robustness to coherent
misinformation.
Calibration can align score scales, but not
guarantee source invariance.Panel (c) applies
the same answerability calibrator to Wikipedia QA
evidence and RGB web snippets. Average AUC
drops from 0.775 to0.651 . Because AUC measures
ranking rather than threshold placement, the drop
reflects a less predictive evidence representation un-
der source shift, not only threshold miscalibration.
Calibration can stabilize related datasets but can-not guarantee transfer across qualitatively different
evidence sources.
Key takeaway.Robustness depends on the fail-
ure type. Filtering irrelevant or misleading pas-
sages does not solve source shift, where the diag-
nostic itself can lose ranking quality.
6 Conclusion and Limitations
BELIEFRAG treats retrieval as sequential evidence
control. It reaches 0.572 mean F1 with 39%
fewer tokens than Iterative RAG on gpt-oss-120b,
and0.552 with35% fewer tokens on Qwen3-32B.
Gains mainly come from evidence replacement
and answerability-guided stopping. Limitations
include lexical retrieval, only n= 100 questions
per dataset, backbone-specific calibration, self-
judged auxiliary metrics, and degraded transfer
under evidence-source shift.

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup
Sil, and Hannaneh Hajishirzi. 2024. Self-RAG:
Learning to retrieve, generate, and critique
through self-reflection. InThe Twelfth Interna-
tional Conference on Learning Representations.
Tom B. Brown, Benjamin Mann, Nick Ryder,
Melanie Subbiah, Jared D. Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell, Sandhini Agar-
wal, Ariel Herbert-V oss, Gretchen Krueger, Tom
Henighan, Rewon Child, Aditya Ramesh, Daniel
Ziegler, Jeffrey Wu, Clemens Winter, Chris
Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christo-
pher Berner, Sam McCandlish, Alec Radford,
Ilya Sutskever, and Dario Amodei. 2020. Lan-
guage models are few-shot learners. InAd-
vances in Neural Information Processing Sys-
tems, volume 33, pages 1877–1901.
Jiawei Chen, Hongyu Lin, Xianpei Han, and
Le Sun. 2024. Benchmarking large language
models in retrieval-augmented generation.Pro-
ceedings of the AAAI Conference on Artificial
Intelligence, 38(16):17754–17762.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Constructing
a multi-hop QA dataset for comprehensive eval-
uation of reasoning steps. InProceedings of
the 28th International Conference on Compu-
tational Linguistics, pages 6609–6625. Interna-
tional Committee on Computational Linguistics.
Liu Huanshuo, Hao Zhang, Zhijiang Guo, Jing
Wang, Kuicai Dong, Xiangyang Li, Yi Quan
Lee, Cong Zhang, and Yong Liu. 2025. CtrlA:
Adaptive retrieval-augmented generation via in-
herent control. InFindings of the Association
for Computational Linguistics: ACL 2025, pages
12592–12618, Vienna, Austria. Association for
Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho,
Sung Ju Hwang, and Jong Park. 2024. Adaptive-
RAG: Learning to adapt retrieval-augmented
large language models through question com-
plexity. InProceedings of the 2024 Conference
of the North American Chapter of the Associ-
ation for Computational Linguistics: HumanLanguage Technologies (Volume 1: Long Pa-
pers), pages 7036–7050. Association for Com-
putational Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. 2023. Active
retrieval augmented generation. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 7969–7992.
Association for Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung
Yoon, Sercan Ö. Arık, Dong Wang, Hamed Za-
mani, and Jiawei Han. 2025. Search-R1: Train-
ing LLMs to reason and leverage search engines
with reinforcement learning. InProceedings
of the 2nd Conference on Language Modeling
(COLM 2025).
Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang
Liu, Xiaojian Jiang, Jiexin Xu, Li Qiuxia,
and Jun Zhao. 2024. Tug-of-war between
knowledge: Exploring and resolving knowledge
conflicts in retrieval-augmented language mod-
els. InProceedings of the 2024 Joint Interna-
tional Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-
COLING 2024), pages 16867–16878, Torino,
Italia. ELRA and ICCL.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and
Luke Zettlemoyer. 2017. TriviaQA: A large
scale distantly supervised challenge dataset for
reading comprehension. InProceedings of the
55th Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 1601–1611. Association for Computa-
tional Linguistics.
Leslie Pack Kaelbling, Michael L. Littman, and
Anthony R. Cassandra. 1998. Planning and act-
ing in partially observable stochastic domains.
Artificial Intelligence, 101(1–2):99–134.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia
Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob
Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, An-
drew M. Dai, Jakob Uszkoreit, Quoc Le, and
Slav Petrov. 2019. Natural questions: A bench-
mark for question answering research.Trans-

actions of the Association for Computational
Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau
Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. 2020. Retrieval-augmented gen-
eration for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Sys-
tems, volume 33, pages 9459–9474.
Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan
Cheng, Bo Wang, Yining Zheng, Yuxin Wang,
Zhangyue Yin, and Xipeng Qiu. 2025. R3-RAG:
Learning step-by-step reasoning and retrieval for
LLMs via reinforcement learning. InFindings of
the Association for Computational Linguistics:
EMNLP 2025, pages 10491–10507. Association
for Computational Linguistics.
Seiji Maekawa, Hayate Iso, and Nikita Bhutani.
2025. Holistic reasoning with long-context LMs:
A benchmark for database operations on massive
textual data. InThe Thirteenth International
Conference on Learning Representations.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi
Das, Daniel Khashabi, and Hannaneh Hajishirzi.
2023. When not to trust language models: In-
vestigating effectiveness of parametric and non-
parametric memories. InProceedings of the 61st
Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers),
pages 9802–9822. Association for Computa-
tional Linguistics.
Viktor Moskvoretskii, Maria Marina, Mikhail
Salnikov, Nikolay Ivanov, Sergey Pletenev,
Daria Galimzianova, Nikita Krayko, Vasily
Konovalov, Irina Nikishina, and Alexander
Panchenko. 2025. Adaptive retrieval with-
out self-knowledge? bringing uncertainty back
home. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 6355–
6384, Vienna, Austria. Association for Compu-
tational Linguistics.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model
card. Model Card.
Quang Hieu Pham, Hoang Ngo, Anh Tuan Luu, and
Dat Quoc Nguyen. 2024. Who’s who: Large lan-guage models meet knowledge conflicts in prac-
tice. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2024, pages 10142–
10151, Miami, Florida, USA. Association for
Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Information
Retrieval, 3(4):333–389.
Heydar Soudani, Evangelos Kanoulas, and
Faegheh Hasibi. 2025. Why uncertainty esti-
mation methods fall short in RAG: An axiomatic
analysis. InFindings of the Association for Com-
putational Linguistics: ACL 2025, pages 16596–
16616, Vienna, Austria. Association for Compu-
tational Linguistics.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing
Wu, and Yiqun Liu. 2024. DRAGIN: Dynamic
retrieval augmented generation based on the real-
time information needs of large language models.
InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers), pages 12991–13013.
Association for Computational Linguistics.
Chihiro Taguchi, Seiji Maekawa, and Nikita
Bhutani. 2025. Efficient context selection for
long-context QA: No tuning, no iteration, just
adaptive- k. InProceedings of the 2025 Confer-
ence on Empirical Methods in Natural Language
Processing, pages 20105–20130. Association for
Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. MuSiQue:
Multihop questions via single-hop question com-
position.Transactions of the Association for
Computational Linguistics, 10:539–554.
Alexander Wan, Eric Wallace, and Dan Klein. 2024.
What evidence do language models find convinc-
ing? InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 7468–7484,
Bangkok, Thailand. Association for Computa-
tional Linguistics.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng
Chen, and Sercan Ö. Arık. 2025. Astute RAG:
Overcoming imperfect retrieval augmentation

and knowledge conflicts for large language mod-
els. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 30553–30571.
Association for Computational Linguistics.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua
Ling. 2024. Corrective retrieval augmented gen-
eration.arXiv preprint arXiv:2401.15884.
An Yang, Anfeng Li, Baosong Yang, Beichen
Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, Chu-
jie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang,
Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jia-
long Tang, Jian Yang, Jianhong Tu, Jianwei
Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou,
Jingren Zhou, Junyang Lin, Kai Dang, Ke-
qin Bao, Kexin Yang, Le Yu, Lianghao Deng,
Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang,
Peng Wang, Qin Zhu, Rui Men, Ruize Gao,
Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi
Tang, Wenbiao Yin, Xingzhang Ren, Xinyu
Wang, Xinyu Zhang, Xuancheng Ren, Yang
Fan, Yang Su, Yichang Zhang, Yinger Zhang,
Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui,
Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. 2018. Hot-
potQA: A dataset for diverse, explainable multi-
hop question answering. InProceedings of the
2018 Conference on Empirical Methods in Nat-
ural Language Processing, pages 2369–2380.
Association for Computational Linguistics.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin
Cao, Linmei Hu, Liu Weichuan, Lei Hou, and
Juanzi Li. 2025. SeaKR: Self-aware knowledge
retrieval for adaptive retrieval augmented genera-
tion. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 27022–27043,
Vienna, Austria. Association for Computational
Linguistics.

A Category 1: Implementation Details
This appendix specifies the implementation behind
the controller in the main text. We usebelief state
operationally: btis a persistent estimate of the cur-
rent evidence condition, computed from observable
diagnostics and the previous state. The deployed
system does not perform an exact Bayesian update
over an explicit latent environment variable. This
distinction keeps the appendix aligned with the es-
timator and controller actually evaluated.
A.1 Evidence Workspace and Diagnostics
The retained evidence set Etis a workspace rather
than an append-only retrieval transcript. RE-
TRIEVEcan enlarge it, REWRITEchanges the
search query without changing it, and VERIFYcan
shrink it by removing passages judged unhelpful.
Dropped passage identifiers are retained so a later
retrieval cannot silently reintroduce the same pas-
sage.
At the beginning of each decision step, the har-
ness computes
xt= [R t, St, Ct, Ut, Gt, Nt, Kt],
where relevance Rt, novelty Nt, and normalized
costKtare computed locally, while support St,
conflict Ct, uncertainty Ut, and gap Gtcome from
one structured verifier call. Thus the verifier ob-
serves the evidence but does not mutate it; removal
occurs only if the controller selects VERIFY.
For retrieval relevance, the raw BM25
score (Robertson and Zaragoza, 2009) zt,iof
passageiat steptis first calibrated,
˜zt,i=σzt,i−µs
τs
,
ωt,i=exp(˜z t,i/τR)Pk
j=1exp(˜z t,j/τR),
Rt=kX
i=1ωt,i˜zt,i.
Here kis the number of passages returned by one
retrieval call, µsandτs>0are the retriever-score
location and scale statistics, τRis the relevance
softmax temperature, and σis the logistic sigmoid
defined in Section 3. The default relevance temper-
ature is τR= 0.2 . Novelty is computed from the
new passages and the currently retained passages,
Nt= 1−1
|∆E t|X
e∈∆E tmax
e′∈Et−1sim(e, e′),with lexical Jaccard similarity in the evaluated con-
figuration. With no retained passages, novelty is 1;
with no new passages, it is 0. Normalized acquisi-
tion cost is
Kt= min
1,cumulative tokens
token budget
.
A.2 Belief Mapping and Default Parameters
For each non-cost dimension d, the instantaneous
estimate uses the complete diagnostic vector,
ˆbd
t=σ(α d+w⊤
dxt),
and the persistent state is updated in logit space,
bd
t=σ
(1−λ d) logit(bd
t−1) +λ dlogit( ˆbd
t)
.
The new observation therefore receives weight λd.
Cost is observed directly, so bcost
t=K trather than
being inferred. Table 4 gives the default unfitted pa-
rameterization. These coefficients are priors whose
signs follow the intended meanings of the signals.
The main stopping probability uses the separately
fitted answerability calibrator described next.
A.3 Main Answerability Calibrator
The main-table runs use a single logistic_x
calibrator over the full seven-dimensional diagnos-
tic vectorx t= [R t, St, Ct, Ut, Gt, Nt, Kt]:
pans
t=σ(α+w⊤xt).(9)
Its target is answerable_now : the fixed genera-
tor receives the question and current Et, produces
an answer under the evaluated answer prompt, and
the shared semantic-answer judge scores that out-
put. This is an operational generator-success target,
not an entailment label for Etalone. Because the
prompt requires a best short answer when the doc-
uments are incomplete, the frozen backbone may
use parametric knowledge in addition to retained
evidence. The calibrator is fitted only on 300 Hot-
potQA training examples and the configuration is
selected on 156 HotpotQA development examples
by lowest Brier score (0.2236; development base
rate 0.2501). It is then frozen and reused without
retuning on 2WikiMultiHopQA, MuSiQue, Natural
Questions, TriviaQA, and PopQA. The saved arti-
fact is marked contaminated: false . Platt
and isotonic mappings exist as alternative imple-
mentations but are not used for the reported main
results.

Table 4: Default belief-state parameters. Only non-zero coefficients are shown.
Dimensionb 0 α λNon-zero weights inw⊤
dxt
Sufficiency 0.35−1.60.6+2.6S t,+1.2R t,−2.2G t,−1.0U t,−0.6C t
Reliability 0.50−1.40.5+2.8R t,+0.9S t,−2.4C t
Conflict 0.05−2.20.6+4.4C t,−0.5S t
Uncertainty 0.70−1.20.6+2.6U t,+1.3G t,−1.6S t,+0.7C t
Gap 0.90−1.00.7+3.0G t,−1.4S t,+0.4N t
Cost 0.00 — 1.0 observed directly:bcost
t=K t
The fitted intercept is α= 0.65330 . In
diagnostic-vector order, the nonzero weights are
wS= +0.53301, w G=−0.54360,
wU=−0.47780, w R= +0.08474,
wN= +0.06258, w K=−0.00441.
with wC= 0.00000 . Conflict is not manually
pruned: on the clean fitting data Ctis almost al-
ways zero, so it has essentially no variance and
receives a zero fitted coefficient. This also ex-
plains why conflict is evaluated separately under
counterfactual-evidence perturbations in the main
analysis.
A.4 Terminal Controller
The main controller is a branch policy rather than
the optional LLM action selector. It evaluates the
following branches in order:
1. finish a correction already under way;
2.correct materially conflicting, unreliable, or ex-
plicitly flagged evidence when the correspond-
ing correction trigger is enabled;
3.answer when pans
t≥τansand conflict is accept-
able;
4.retrieve when the evidence is not yet answerable,
budget remains, and pflip
tclears the minimum
retrieval value;
5.rewrite before another retrieval when the previ-
ous round added little novelty; and
6.otherwise stop, or abstain when evidence is in-
adequate and further acquisition has little value.
The default thresholds are τans= 0.50 , conflict
threshold 0.50, reliability threshold 0.30, mini-
mum retrieval value 0.10, and low-novelty thresh-
old0.20. The controlled answerability intervention
in Figure 3 instead uses the separately selected and
then frozen threshold τ= 0.55 . Its historical run
labels say “sufficiency,” but the manipulated score
ispans
t, notbsuff
t; the two threshold values refer to
different experimental configurations.
The retrieval-value model estimates
pflip
t=σ(α flip+wflip
rrt+wflip
NNt+wflip
sbsuff
t).When no fitted retrieval-value model is loaded, the
harness uses the recorded per-round fallback val-
ues{0:0.59,1:0.073,2:0.050,3:0.050} rather than
inventing a score at runtime.
B Category 1: Worked HotpotQA
Trajectory
Table 5 gives one successful trajectory because
it makes two implementation properties concrete:
the evidence set can become smaller after verifi-
cation, and a later query can name a bridge entity
discovered in retained evidence. The example also
exposes a limitation that aggregate answer metrics
can hide.
Question.Private Music signed the drummer
who was part of which English group?The gold
answer isBeatles.
The trace makes two behaviors concrete: veri-
fication shrinks Etfrom five passages to one, and
the next search names the bridge entityRingo Starr
found in retained evidence. It also exposes a lim-
itation. The second retrieval does not recover a
Ringo-Starr-to-Beatles passage, so the final cor-
rect answer combines the retained clue with the
backbone’s frozen parametric knowledge. This is
allowed by the prompt’s explicit fallback to a best
short answer when the documents are incomplete.
The trace is therefore a successful control trajec-
tory rather than a fully retrieved two-hop proof;
the logged evidence-recall score should be read as
annotation coverage, not proof that every required
supporting passage was retrieved distinctly. The
trailing >in the Step 3 query is reproduced from
the trace and is a query-writer formatting error, not
a manuscript placeholder.

Table 5: Representative HotpotQA trace.|E t|is the number of retained passages after the action.
Step Action Decision and query/evidence change|E t|
0 RETRIEVE Initial belief is (0.35,0.50,0.05,0.70,0.90,0.00) . With no evidence, the structural diagnostics set uncertainty and
gap to 1. The fallback pflip= 0.59 exceeds the 0.10 retrieval threshold. Under first_query_is_question ,
the original question is issued directly as the first query. Five passages are returned; only Private_Music is
useful.5
1 VERIFY The verifier reports approximately Rt= 0.27 ,St= 0.30 ,Gt= 0.70 , andKt= 0.21 , with P(answerable) =
0.49. Four passages are flagged unhelpful: Myron__duo_ ,Zak_Starkey ,Weathermaker_Music , and
Oasis_discography. Verification prunes them, leaving onlyPrivate_Music.1
2 REWRITE The query is reformulated as Private Music signed drummer formerly a member of an
English band. Rewriting changes the query but performs no retrieval.1
3 RETRIEVE The evidence-aware query writer reads the retained Private_Music passage, which namesRingo Starr, and makes
that bridge entity explicit: Ringo Starr drummer member of which English group?> . Three ad-
ditional passages are returned, but all are distractors and none mentions Ringo Starr or the Beatles.4
4 ANSWER With approximately Rt= 0.34 ,St= 0.30 ,Gt= 0.60 ,Kt= 0.45 , conflict 0.09 , andP(answerable) = 0.53 ,
the answer threshold0.50is crossed. The final answer isThe Beatles.4
C Category 1: Model-Facing Prompts
The main terminal controller is rule based, so it
does not ask an LLM to choose the next action.
Model calls are used for evidence verification,
query writing or rewriting, and final-answer gener-
ation. The first retrieval in the worked trace uses
the original question directly and therefore incurs
no query-writer call. The prompt templates below
are reproduced verbatim from the evaluated config-
uration. Optional controller and baseline prompts
are not used by the main BeliefRAG controller and
are therefore omitted here. Template fields such
as{question} and the JSON empty array []
are literal prompt/schema notation, not unfinished
manuscript placeholders.
C.1 Backend System Prompt
You are the language-model
backend for controlled
research experiments.
Follow the task instructions in
the current request exactly.
Do not use external tools,
external retrieval, or hidden
assumptions.
Treat the evidence, state, and
other context explicitly
provided in the request as
the complete experimental
context unless the request
says otherwise.
Do not invent missing evidence
or observations.
Return exactly the output format
requested by the task.
If a JSON schema is requested,
return valid JSON only, with
no markdown fences,
commentary, or additional text.C.2 Verifier Prompt (verifier_v1)
You are an evidence verifier for
a retrieval experiment.
Judge ONLY the evidence
shown below. Do not use outside
knowledge and do not retrieve
anything.
Question: {question}
Evidence passages (each prefixed
by its passage id):
{evidence}
Current draft answer: {draft}
Produce these judgements, each a
float in [0,1]:
"support" : degree to
which the draft answer is
entailed by the evidence.
If there is no
draft answer, judge instead
how strongly the
evidence
entails a complete answer to
the question.
"conflict" : the strongest
contradiction present, either
between two
evidence
passages or between a passage
and the draft answer.
0.0 if the
passages are mutually
consistent.
"gap" : the fraction
of the facts or sub-questions
required to answer
the question
that are still NOT supported
by the evidence.
1.0 means
nothing required is supported

, 0.0 means everything is.
"uncertainty" : how uncertain
a careful reader would remain
about the final
answer given
only this evidence.
Also list "unhelpful_doc_ids":
the passage ids that are off-
topic, redundant, or
misleading and should be dropped
from the evidence set. Use
[] if none are.
Return only this JSON object:
{"support": <float>, "conflict":
<float>, "gap": <float>,
"uncertainty": <float>, "
unhelpful_doc_ids": [<id>,
...]}
C.3 Evidence-Aware Query Writer
(query_refiner_v2)
Question: {question}
Evidence already retrieved:
{evidence}
Search queries already issued: {
prior_queries}
Identify what the question still
requires that the evidence
above does NOT yet
provide, and write ONE search
query targeting exactly that
missing piece.
Rules:
- Do not search for anything the
evidence already establishes
.
- If the question needs a fact
about an entity the evidence
has just identified,
name that entity explicitly in
the query rather than
referring to it indirectly.
- Do not repeat or lightly
reword a query already issued
.
- If nothing further is
genuinely needed, output the
single word: SUFFICIENT
Output only the query text on
one line, or SUFFICIENT.
C.4 Query Rewriter (query_rewrite_v1)
Question: {question}Current search query: {query}
Why the current query is failing
: {reason}
Rewrite the search query so it
retrieves better evidence.
Change the wording or
target a different required fact
; do not simply repeat the
current query.
Output only the rewritten query,
on a single line.
C.5 Answer Generator
(answer_generator_v1)
The evaluated prompt is reproduced verbatim. Its
first sentence is evidence-first, but the explicit fall-
back requires a best short answer when the docu-
ments are incomplete; parametric fallback is there-
fore allowed in the operational answerability tar-
get.
Answer the question using only
the documents below. Give
only the final answer,
as short as possible, with no
explanation and no
restatement of the question.
If the documents do not contain
the answer, reply with your
single best short
answer anyway.
Documents:
{evidence}
The question: {question}
Output the bare answer text only
: no citations, no document
numbers, no markup,
no quotation marks, and no
leading "Answer:".
D Category 1: Controlled Harness and
Reproducibility
This section records the implementation details
needed to reproduce the controlled comparison:
execution invariants, budgets, baseline adaptations,
model settings, data partitions, aggregation, and
replay safeguards.
D.1 Execution Invariants
Each episode follows the same four-stage loop: (1)
the verifier observes the current evidence and re-
turns diagnostics xt; (2) the updater maps xtand

Table 6: Default budgets used by the controlled harness.
Budget Value
Maximum retrieval rounds 3
Passages per retrieval (top-k) 5
Maximum controller actions 6
Token budget 12,000
Answer evidence window 10 passages
the previous belief into bt; (3) the controller pro-
poses one legal action; and (4) the harness executes
that action. The backend model does not retrieve
and does not execute actions. Only the evidence
manager may mutate Et, and every mutation is
logged. The feasible action set is enforced by the
harness rather than trusted to a model response.
The controller may observe the question, re-
tained evidence, current belief, current diagnostics,
prior actions and queries, remaining budget, and
the feasible action set. Evaluation-only fields such
as gold answers, supporting facts, labels, and met-
rics are denied to model-facing state projections.
D.2 Shared Budgets and Cost Accounting
Table 6 gives the shared acquisition limits. Token
cost counts retrieval context each time it is model-
facing, rewriting, verification, answer generation,
and controller prompting; retrieval calls are logged
separately. Input/output tokens are stored sepa-
rately, using a recorded local tokenizer when the
API omits counts.
D.3 Baseline Implementations
The common harness isolates controller mecha-
nisms: Static retrieves once; Iterative follows fixed
rounds; Adaptive- kchanges retained context size;
Adaptive-RAG routes once by complexity; Self-
RAG* isolates a retrieve/no-retrieve trigger; CRAG
uses prune–rewrite–re-retrieve; RL-Search uses a
prompted search/answer interface; and No-RAG
is closed-book. Unless stated otherwise, these are
common-harness mechanism variants rather than
exact end-to-end reproductions.
D.4 Model Backend
Both studies use a university-hosted in-
ference API with transmitted identifiers
gpt-oss:120b (OpenAI, 2025) and
qwen3:32b (Yang et al., 2025). The API
does not disclose the Qwen checkpoint/revision,
quantization, or serving stack. Calls set tem-
perature 0andstream=false ; no top_p ,top_k , repetition penalty, seed, stop sequence,
ormax_tokens is transmitted. Local output
limits are therefore accounting targets. A direct
server probe stopped near 863 output tokens, while
typical experiment outputs are about 40.
Temperature 0is not deterministic on this ser-
vice: one fixed Qwen prompt produced two outputs
over eight repeats (7/8 and 1/8), and a prior gpt-oss
cache audit found 58 divergent keys among 5,341
repeats. Cached/resumed runs preserve observed
responses, but uncached reruns may differ. Con-
text probes succeeded near 4k tokens and returned
HTTP 200 with an empty message near 8k and
above; normal inputs are about 600 tokens, with at
most 10 answer passages and 8 verifier passages,
and empty generations are retried once. Within a
backbone, generator, verifier, query writer, and se-
mantic judge share one backend, so acc_judge
is self-judged and not used as a cross-backbone
scale. Platform-returned retrieval contexts are dis-
carded.
D.5 Data Partitions
For each backbone, 300 HotpotQA training ex-
amples fit the answerability calibrator and 156
development examples select it by Brier score;
the retrieval-value model is also refit. Qwen
evaluation exits if either Qwen-specific arti-
fact is missing rather than reusing gpt-oss arti-
facts. Dedicated causal analyses use determinis-
ticxfit /xsel /xhold splits of 40/30/30%, as-
signed by salted question-id hash, with holdout
labels excluded from fitting. Within each dataset,
methods share corpus, retriever, generator, verifier,
answer prompt, evaluator, and budget; run meta-
data records the corresponding configurations.
D.6 Metrics and Aggregation
Per episode the harness stores EM, normalized to-
ken F1 (primary), acc_cover , supplementary
acc_judge , evidence recall when annotated, an-
swer/abstain status, retrieval rounds, actions, calls,
tokens, and latency. Abstention scores zero on an-
swer metrics. Repeated seeds are averaged within
question before bootstrap; mean intervals use 2,000
resamples and paired within-dataset comparisons
use 10,000. Headline cross-dataset results report
deterministic token F1 rather than the earlier LLM-
judge interval.
E Category 2: Complementary Results

Table 7:Qwen3-32B answer quality.Each dataset cell isF1 / EM / Judge, where Judge is the supplementary
LLM-judged semantic accuracy. Mean is mean F1 over six datasets. All dataset cells usen= 100.
Method HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA Mean F1
No-RAG .265/.180/.320 .382/.310/.420 .154/.080/.200 .210/.120/.280 .512/.420/.550 .185/.130/.220 .285
Static .497/.380/.560 .643/.570/.630 .309/.170/.360 .365/.230/.420 .642/.560/.680 .284/.240/.350 .457
Iterative .559/.460/.610 .659/.570/.630 .361/.240/.410 .412/.310/.470.695/.600/.720 .452/.370/.510 .523
Self-RAG* .367/.300/.400 .625/.517/.552 .285/.210/.320 .215/.110/.260 .088/.080/.120 .245/.210/.300 .304
CRAG .512/.410/.570 .680/.590/.660 .352/.230/.400 .388/.250/.440 .665/.580/.700 .410/.340/.460 .501
RL-Search .278/.180/.310 .285/.100/.310 .198/.100/.230 .092/.010/.120 .248/.180/.290 .210/.110/.250 .219
Adaptive-k.508/.390/.565 .651/.575/.635 .320/.180/.370 .372/.240/.430 .650/.565/.685 .305/.260/.370 .468
Adaptive-RAG .542/.430/.590 .668/.595/.650 .348/.220/.390 .395/.270/.450 .688/.590/.710 .438/.360/.490 .513
BELIEFRAG .582/.475/.635.715/.620/.690.418/.295/.460.428/.325/.485 .690/.615/.715.476/.395/.530.552
Table 8:Qwen3-32B token cost and logged retrieval recall.Panel (a) reports thousands of tokens per episode.
Panel (b) reproduces the supplied retrieval-recall fields; BeliefRAG open-domain recall values were not supplied
and are shown as dashes. The main paper uses supporting-fact evidence recall only where annotated supporting
facts are available.
(a) Tokens/episode (k)
Method HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA Mean
No-RAG .920 1.050 .980 .850 .910 .820 .922
Static 2.624 3.012 2.815 2.450 2.580 2.310 2.632
Iterative 5.752 6.662 6.650 5.210 5.480 5.120 5.812
Self-RAG* 3.515 5.193 4.210 2.850 1.820 2.650 3.373
CRAG 3.820 4.150 3.950 3.210 3.420 3.100 3.608
RL-Search 6.120 7.210 6.850 5.820 5.950 5.400 6.225
Adaptive-k3.120 3.450 3.280 2.890 2.950 2.720 3.068
Adaptive-RAG 3.650 4.120 4.050 2.980 3.150 2.850 3.467
BELIEFRAG3.850 4.320 4.250 3.410 3.550 3.2803.777
(b) Logged retrieval recall
Method HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA
No-RAG .000 .000 .000 .000 .000 .000
Static .790 .978 .817 .760 .880 .720
Iterative .798 .990 .888 .810 .910 .850
Self-RAG* .606 .991 .740 .520 .310 .650
CRAG .825 .985 .850 .835 .925 .820
RL-Search .510 .620 .580 .340 .520 .480
Adaptive-k.805 .980 .825 .775 .890 .745
Adaptive-RAG .815 .982 .860 .800 .905 .830
BELIEFRAG .860 .995 .912– – –
Table 9:gpt-oss-120b secondary answer metrics.Each cell isEM / Judge; F1 is reported in the main table.
Mean gives the simple six-dataset average for each metric.
Method HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA Mean EM/Judge
Static .450/.640 .600/.720 .290/.440 .280/.510 .600/.810 .280/.330 .417/.575
Iterative .452/.660 .670/.750 .355/.520 .350/.620 .625/.860 .410/.530 .477/.657
CRAG .460/.650 .630/.770 .310/.460 .280/.570 .630/.830 .380/.470 .448/.625
Self-RAG* .380/.540 .600/.680 .270/.400 .130/.310 .100/.120 .250/.290 .288/.390
RL-Search .200/.620 .110/.750 .120/.440 .010/.440 .200/.530 .130/.420 .128/.533
BELIEFRAG.480/.630 .660/.810 .370/.520 .290/.580 .650/.850 .340/.430 .465/.637

Table 10:gpt-oss-120b cost and multi-hop evidence recall.Tokens are thousands per episode. Evidence recall is
reported only for the three multi-hop benchmarks with supporting-fact annotations.
(a) Tokens/episode (k)
Method HotpotQA 2Wiki MuSiQue NQ TriviaQA PopQA Mean
Static 2.620 2.998 2.835 2.762 2.834 2.849 2.816
Iterative 5.536 6.616 6.382 6.738 6.427 6.425 6.354
CRAG 4.395 5.130 4.946 5.331 4.599 5.427 4.971
Self-RAG* 3.473 4.172 4.012 2.736 .897 3.707 3.166
RL-Search 3.308 3.757 3.324 2.985 2.402 3.251 3.171
BELIEFRAG3.472 4.293 4.081 4.029 3.364 4.1303.895
(b) Evidence recall
Method HotpotQA 2Wiki MuSiQue Mean
Static .790 .978 .817 .862
Iterative .793 .990.898 .894
CRAG .783 .985 .807 .858
Self-RAG* .691 .927 .742 .787
RL-Search .728 .927 .733 .796
BELIEFRAG.788.990.817 .865
0 5 10 15 20 25
mean rows actually read0.0000.0250.0500.0750.1000.1250.1500.175row-level recall95% of Iterative's recall
on 42% of the rows(a)  Rows read vs. row recall
Adaptive-k
0.038 recall on 3.7 rowsStatic
0.093 recall on 10.0 rowsBeliefRAG
0.145 recall on 9.7 rows Iterative
0.152 recall on 23.0 rows
0.0 2.5 5.0 7.5 10.0 12.5 15.0
recall gained per row read (×103
)IterativeStaticAdaptive-kBeliefRAG
6.6 15.4% of ceiling9.3 9.5% of ceiling10.3 3.9% of ceiling14.9 14.7% of ceiling(b)  Recall per row read
Figure 7:HoloBench aggregation retrieval.Row recallis the fraction of gold rows recovered in the evidence
shown to the model;recall per row readdivides that recall by the mean number of rows inspected (scaled by 103
for display). (a) BeliefRAG reaches row recall 0.145 after reading 9.73 rows on average, versus 0.152 after22.98
rows for fixed Iterative retrieval. (b) BeliefRAG therefore obtains higher recall per row read. The structural ceiling
at 50 rows is 0.985 ; absolute recall remains far below it, so this diagnostic supports a selection-efficiency claim
rather than a claim that aggregation retrieval is solved. These common-harness HoloBench numbers are not directly
comparable with the QA F1 scores in Table 3.