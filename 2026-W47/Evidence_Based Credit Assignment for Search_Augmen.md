# Evidence-Based Credit Assignment for Search-Augmented Language Models

**Authors**: Wen Wang, Jindong Wang, Jiahua Bao, Tu Yongsiqi, Yihao Liu, Junwei He, Wenkui Fan

**Published**: 2026-11-18 04:24:51

**PDF URL**: [https://arxiv.org/pdf/2611.23824v1](https://arxiv.org/pdf/2611.23824v1)

## Abstract
Reinforcement learning from terminal answer rewards enables search-augmented language models to interleave reasoning and retrieval without step-level annotations, yet outcome supervision assigns identical credit to every search in a trajectory regardless of whether it retrieved useful information. This limitation is especially severe when all sampled rollouts for a question fail, collapsing group-centered outcome advantages to zero even when trajectories gather markedly different evidence. We present Evidence-based Credit Assignment (ECA), which augments terminal outcome rewards with a training-only auxiliary signal measuring how retrieved documents improve the likelihood of a reference answer. Rather than scoring the full dialogue prefix, ECA queries a frozen snapshot of the current policy under an evidence-only prompt containing solely the question and the deduplicated documents observed so far, isolating the utility of retrieved passages from surface variations in generated thoughts or search queries. Within this framework, we compare two return formulations, ECA-Step (which accumulates step-clipped potential increments) and ECA-Horizon (which preserves the telescoping horizon gain in potential), paired with cross-fold group normalization, novelty gating, and search-local token assignment. Across seven Search QA benchmarks, ECA-Horizon improves strict macro exact match over an outcome-only GRPO baseline by 1.03 and 4.80 percentage points on Qwen2.5-3B- and 7B-Instruct, respectively, outperforming both RLOO and GiGPO. Zero-shot evaluation on MultiHop-RAG further demonstrates strong transfer to multi-hop inference queries.

## Full Text


<!-- PDF content starts -->

Evidence-Based Credit Assignment
for Search-Augmented Language Models
Wen Wang1,2*Jindong Wang2†Jiahua Bao3Tu Yongsiqi4Yihao Liu5Junwei He1Wenkui Fan6
1University of the Chinese Academy of Sciences2Tencent3Harbin Institute of Technology
4Renmin University of China5Peking University6Beijing Institute of Technology
Abstract
Reinforcement learning from terminal answer rewards en-
ables search-augmented language models to interleave rea-
soning and retrieval without step-level annotations, yet out-
come supervision assigns identical credit to every search in
a trajectory regardless of whether it retrieved useful infor-
mation. This limitation is especially severe when all sam-
pled rollouts for a question fail, collapsing group-centered
outcome advantages to zero even when trajectories gather
markedly different evidence. We presentEvidence-based
Credit Assignment (ECA), which augments terminal out-
come rewards with a training-only auxiliary signal measur-
ing how retrieved documents improve the likelihood of a ref-
erence answer. Rather than scoring the full dialogue prefix,
ECA queries a frozen snapshot of the current policy under
an evidence-only prompt containing solely the question and
the deduplicated documents observed so far, isolating the
utility of retrieved passages from surface variations in gener-
ated thoughts or search queries. Within this framework, we
compare two return formulations,ECA-Step(which accumu-
lates step-clipped potential increments) andECA-Horizon
(which preserves the telescoping horizon gain in potential),
paired with cross-fold group normalization, novelty gating,
and search-local token assignment. Across seven Search
QA benchmarks, ECA-Horizon improves strict macro exact
match over an outcome-only GRPO baseline by 1.03 and
4.80 percentage points on Qwen2.5-3B- and 7B-Instruct, re-
spectively, outperforming both RLOO and GiGPO. Zero-shot
evaluation on MultiHop-RAG further demonstrates strong
transfer to multi-hop inference queries.
1. Introduction
Search-augmented language models tackle knowledge-
intensive questions by alternating between chain-of-thought
reasoning, issuing search queries, and reading retrieved pas-
*Work done during an internship at Tencent.
†Corresponding author.
Figure 1. ECA can credit useful searches in all-failure groups while
gating repeated evidence (illustrative example).
sages [ 31]. Recent work has shown that reinforcement learn-
ing (RL) with terminal answer rewards can train these multi-
turn interactions directly in retrieval environments [ 8,20,33]
using critic-free group-relative estimators such as GRPO and
RLOO [ 1,18]. Despite its simplicity, terminal outcome su-
pervision provides a coarse signal for multi-step search: a
single trajectory-level advantage is broadcast uniformly to
every generated token. Consequently, a well-targeted search
that uncovers a bridging document is penalized if the model
later misstates the final answer, whereas a redundant query is
reinforced whenever the model answers correctly from para-
metric memory or an earlier turn. This credit-assignment
bottleneck becomes especially acute on hard questions where
every sampled rollout in a group fails: centered outcome ad-
vantages collapse to zero across the group, discarding the
gradient even when rollouts retrieved substantially different
evidence.
Recent methods seek finer-grained supervision by group-
ing rollouts at repeated environment states [ 4], scoring re-
trieved passages against supporting-document annotations
1
arXiv:2611.23824v1  [cs.AI]  18 Nov 2026

[6,32], or shaping turn-level rewards via changes in the like-
lihood of the ground-truth answer [ 23,28]. When reference-
answer likelihood is evaluated on the full dialogue history,
however, the resulting potential reflects not only retrieved
documents, but also the agent’s intermediate reasoning and
query phrasing. A policy can therefore alter its own potential
merely by rewording a thought or echoing candidate answers,
even when the underlying evidence set has not changed.
This motivates a central question:can we assign effective
step-level credit to search actions using an evidence-only
potential that depends strictly on the question and retrieved
documents?
To address this question, we proposeEvidence-based
Credit Assignment (ECA). During training, a frozen snap-
shot of the current policy teacher-forces a canonical refer-
ence answer under a dedicated scoring prompt containing
only the question and the deduplicated documents consumed
up to each search slot. Stripping generated thoughts and
search queries from the scoring context makes the potential
Φ(D) invariant to surface reasoning variations: a search ac-
tion changes the potential if and only if it introduces new doc-
uments that shift the likelihood of the reference answer. This
readout requires no gold passage labels, separate process
reward models, or rollout continuations, and is discarded
entirely at test time.
Within the ECA framework, we study how step-wise
potential changes should be aggregated and normalized
into token-level advantages. We compare two return con-
structions:ECA-Step, which clips single-step potential in-
crements before accumulating future progress, andECA-
Horizon, which preserves the unclipped telescoping differ-
ence between the terminal horizon potential and the potential
prior to a search. Both variants standardize returns viacross-
fold normalization(so a trajectory’s own evidence gain does
not inflate its normalization baseline) and apply anovelty
gatethat withholds positive suffix credit when a query re-
turns no new document IDs. Crucially, the resulting auxiliary
advantage is added to the full search action (both the search
call and its preceding reasoning tokens), leaving final-answer
tokens under pure outcome supervision.
We evaluate ECA across seven Search QA benchmarks
totaling 51,713 test questions with Qwen2.5-3B- and 7B-
Instruct [ 29], and test zero-shot transfer to MultiHop-RAG
[21]. Our main contributions are:
•We formulate an evidence-only answer potential and a
bounded search-local advantage for outcome-supervised
RL, restoring informative credit in over 97% of all-failure
rollout groups without passage annotations or external
reward models.
•We analyze how step-wise increment clipping interacts
with multi-step evidence accumulation, comparing step-
clipped returns (ECA-Step) and horizon-level telescop-
ing returns (ECA-Horizon) alongside ablations on tokenplacement and cross-fold normalization.
•Across seven Search QA benchmarks, ECA-Horizon im-
proves strict macro exact match over our matched GRPO
baseline by +1.03 percentage points (pp) at 3B and +4.80
pp at 7B, outperforming RLOO and GiGPO at both scales
while achieving strong zero-shot transfer on MultiHop-
RAG inference questions.
2. Related Work
Credit assignment for multi-turn agents.Assigning step-
level credit from sparse outcomes is a central challenge
in RL for reasoning and tool-use agents. Process reward
models supervise intermediate steps using human or auto-
mated annotations [ 14,26], whereas VinePPO estimates
step values via Monte Carlo rollout continuations [ 10], and
recent multi-turn formulations separate terminal, delayed,
and per-turn rewards to reduce variance across long hori-
zons [ 12,27]. For retrieval agents, several works derive
intermediate signals from corpus annotations: StepSearch
rewards gold-document coverage while penalizing redun-
dant queries [ 32], PiCA trains a reward model from pivot
annotations [ 15], and related methods ground credit in an-
notated retrieval dependencies [ 6] or counterfactual query
verification [ 7]. Without passage annotations, IGPO [ 23]
and TIPS [ 28] shape turn-level rewards using changes in
reference-answer likelihood across the dialogue prefix, with
IGPO normalizing immediate length-scaled log-likelihood
gains before discounted accumulation. ECA builds on this
likelihood-based paradigm, but restricts the scoring prompt
strictly to the question and deduplicated retrieved documents
(excluding generated thoughts and queries) and studies step-
clipped versus horizon-level returns under search-local credit
assignment.
Group-relative policy optimization.Critic-free estima-
tors such as GRPO and RLOO compute relative advantages
across multiple rollouts per prompt [ 1,18], a paradigm
widely adopted in outcome-supervised search agents includ-
ing Search-R1, ReSearch, and R1-Searcher [ 2,8,19]. Be-
cause trajectory-level normalization assigns one scalar to all
turns and collapses when all rollouts fail, recent extensions
refine the comparison group or reweight round contribu-
tions: GiGPO groups actions at identical or similar interme-
diate states [ 4], KnowGPO clusters rollouts by LLM-verified
atomic knowledge conditions [ 3], and CW-GRPO uses an
LLM judge to reallocate outcome advantages across rounds
[24]. Rather than requiring state collisions, external LLM
verifiers, or successful donor trajectories, ECA retains the
group-normalized terminal advantage and adds a bounded
search-local residual standardized via cross-fold normaliza-
tion within each question’s rollout group (see Appendix D
for a detailed comparison).
2

3. Evidence-Based Credit Assignment
3.1. Search Trajectories and Outcome Supervision
Given a question x, we sample a group of Gtrajectories
{τi}G
i=1from the current behavior policy πθ0. Each trajec-
tory interleaves generated reasoning ( <think> ), search calls
(<search> ), retrieved document observations, and a termi-
nal answer ( <answer> ). Let Ri∈ {0,1} denote the binary
exact-match reward of the final answer. Following GRPO
[18], the trajectory-level outcome advantage is standardized
across the group:
Aout
i=Ri−¯R
sR+ϵ,(1)
where ¯RandsRdenote the group mean and sample standard
deviation of {Rj}G
j=1, andϵ= 10−6. Retrieved observation
tokens are masked out of the policy gradient loss. Whenever
allGtrajectories answer incorrectly ( R1=···=R G= 0),
Aout
ivanishes for every token in the group.
To align intermediate retrieval steps across trajectories
of varying lengths, we index search slots over a fixed hori-
zonk∈ {1, . . . , H} , where H= 3 is the maximum search
budget. Let Di,kdenote the cumulative set of unique docu-
ments consumed by the policy after its k-th search, starting
from Di,0=∅. If a trajectory terminates in fewer than H
searches, its remaining slots are padded by carrying forward
its final document set ( Di,k=D i,k−1 ) and receive no token-
level credit. Defining returns over a fixed horizon Havoids
dividing cumulative gains by a variable, policy-chosen num-
ber of searches, which would otherwise reward premature
termination after a single lucky retrieval.
3.2. Evidence-Only Answer Potential
A core principle of ECA is to measure the utility of re-
trieved documents without confounding it with how the agent
phrased its intermediate thoughts or search queries. If a po-
tential function conditions on the full dialogue prefix, the
policy can inflate answer likelihood simply by hedging or
hinting at candidate answers in its chain-of-thought, even
when retrieval fails.
To isolate evidence progress, we evaluate reference-
answer likelihood under anevidence-onlycontext (Fig-
ure 2b). From the ground-truth answer aliases for question
x, we select a canonical target y∗with the fewest tokenizer
tokens (breaking ties lexicographically) and denote its token
length by L. Using a frozen snapshot of the current behavior
policy πθ0, we define the evidence potential of document
setDas the length-normalized log-likelihood of y∗under
teacher forcing:
Φi(D) =1
LLX
ℓ=1logπ θ0(y∗
ℓ|C(x, D), y∗
<ℓ),(2)where C(x, D) is a structured prompt containing only the
question xand the deduplicated documents in Dsorted
canonically by document ID. All generated thoughts, search
queries, and sibling rollouts are excluded from C(x, D) , and
gradients are stopped throughΦ i(D).
In practice, each document in Dis represented by its
observed 1,024-character prefix within a 6,144-token scor-
ing context (aborting scoring on context overflow so that
evidence states are never silently altered by truncation). Be-
cause many rollout prefixes within a question group share
identical document sets (including Di,0=∅ and repeated
retrievals), we cache Φi(D) by document-ID set within each
rollout batch. The scorer πθ0is held fixed within each rollout
batch and synchronized with the updated actor before the
next batch, requiring only batched forward passes during
training and zero overhead at test time.
3.3. Step-Wise and Horizon Potential Returns
Let∆i,kdenote the single-step change in evidence potential
produced by thek-th search:
∆i,k= Φi(Di,k)−Φ i(Di,k−1).(3)
Because a search at slot kcan unlock follow-up queries
in slots j > k , we evaluate slot kby the cumulative evi-
dence progress from kthrough H. We compare two return
constructions prior to normalization and gating (Figure 3).
ECA-Step (step-clipped return).Log-likelihood incre-
ments ∆i,jcan exhibit sharp spikes when a passage directly
names the target entity or introduces strong distractors. To
cap the influence of any single step, ECA-Step clips each
increment to [−c, c] (with c= 2 ) before summing over the
remaining steps:
US
i,k=1
cHHX
j=kclip(∆ i,j,−c, c).(4)
Scaling by cHbounds |US
i,k| ≤(H−k+1)/H , though clip-
ping before summation breaks telescoping whenever positive
and negative steps saturate asymmetrically.
ECA-Horizon (horizon-gain return).To preserve the ex-
act cumulative change in evidence potential across the re-
maining horizon, ECA-Horizon sums the unclipped incre-
ments from slot ktoH, telescoping directly to the difference
between the terminal horizon potential and the potential
immediately before searchk:
UH
i,k=1
cHHX
j=k∆i,j
=Φi(Di,H)−Φ i(Di,k−1)
cH.(5)
3

Training only
(a) Search trajectory
(b) Evidence-only readout
(c) Turn-level credit
?
Question
Search policy
Retriever
Search 1
Search 2
Search 3
Answer
?
repeat
Outcome credit: all generated tokens
?
Unique docs
Frozen
current policy
think ...
query ...
Excluded
Evidence potentials
Answer log-likelihood
No thought / query input
ECA-Step
step-wise sum
ECA-Horizon
horizon gain
Cross-fold
normalization
A
B
Clip + novelty gate
Token advantages
Search 1
Search 2
Search 3
Answer
Generated reasoning + query
Consumed document evidence
Auxiliary credit is assigned only to search actions.
thinkquery q1thinkquery q2thinkquery q3
{d1} {d1, d2}{d1, d2}Ri
Aout
i Aout
i Aout
i Aout
iQuestion x
score y∗
Training answer
y∗
Φ0Φ1Φ2 Φ3 Aout
i
λ/tildewideZi,1Aout
i
λ/tildewideZi,2Aout
i
≤0Aout
i
0
Ai,t=Aout
i+λ/tildewideZi,k, λ = 0.1Figure 2. Overview of Evidence-based Credit Assignment (ECA).(a)During rollout, the policy interleaves reasoning and search queries to
accumulate documents, while terminal outcome credit Aoutapplies uniformly to all generated tokens.(b)During training, a frozen snapshot
of the current policy scores the canonical reference answer y∗given only the question xand deduplicated documents D, excluding generated
thoughts and queries.(c)Step-wise potentials are aggregated via ECA-Step or ECA-Horizon, standardized via cross-fold normalization,
clipped, and passed through a novelty gate before being added to search-action tokens.
While UH
i,kpreserves exact horizon progress regardless of
intermediate spikes, stability is maintained by the shared
post-normalization clipping stage below.
When all increments in a group satisfy |∆i,j| ≤c , ECA-
Step and ECA-Horizon produce identical returns. When
increments exceed c, however, step-wise clipping can alter
both the magnitude and ranking of search actions. As shown
in Figure 3, if three searches yield potential increments
(+4,−3,0) withc= 2 andH= 3 , the trajectory finishes
with a horizon gain of +1, giving UH
i,1= 1/6 under ECA-
Horizon. Under ECA-Step, the +4gain and −3drop saturate
to+2and−2, cancelling out to US
i,1= (2−2 + 0)/6 = 0
and erasing credit for the productive first search.
3.4. Cross-Fold Normalization and Novelty Gating
To convert raw returns Um
i,k(m∈ {S,H} ) into well-scaled
advantages across questions of varying difficulty, we stan-
dardize returns within each question group. Unlike binary
terminal rewards Ri∈ {0,1} , intermediate evidence returns
are continuous and heavy-tailed: on a hard question where
only one of G= 8 rollouts discovers a high-value passage
(u >0 ) while the other seven yield 0, normalizing across
all eight rollouts includes the spike uin its own reference
mean ( u/G) and standard deviation ( s=u/√
G), causing
the magnitude uto cancel out of (u−u/G)/s onces≥σ min
(Appendix C).
To keep baseline statistics independent of the evaluated
trajectory while training on all Grollouts, we randomly
partition the G= 8 trajectories of each question into two
disjoint folds of four (independently of rewards) and stan-dardize each trajectory against the mean µm
−f(i),kand sample
standard deviationsm
−f(i),kof theoppositefold:
Zm
i,k= clip 
Um
i,k−µm
−f(i),k
max(sm
−f(i),k, σmin),−2,2!
,(6)
where σmin= 0.25 prevents division by near-zero dispersion
when reference rollouts have similar potentials, and clipping
to[−2,2] bounds outliers. Because fold statistics do not
filter by terminal answer correctness, informative searches
in failed trajectories can still receive positive credit. We omit
mbelow as the remaining steps are shared.
Finally, because Ui,kaccumulates suffix progress from
slotkthrough H, a search at slot kthat retrieves only already-
seen documents ( Di,k=D i,k−1 , so∆i,k= 0) could oth-
erwise inherit a positive return from a productive search at
slotk+ 1 . To prevent redundant queries from free-riding on
subsequent discoveries, we apply anovelty gate:
eZi,k=(
min(0, Z i,k), D i,k=D i,k−1,
Zi,k,otherwise.(7)
Equation 7 zeroes out positive auxiliary credit whenever a
search introduces no novel document IDs, while preserving
negative penalties (when a wasted search leaves the rollout
behind its peers) and the trajectory-level outcome advantage
Aout
i.
3.5. Search-Local Policy Update
An effective search requires not only emitting the query
string inside <search> , but also formulating the information
4

(a) Same evidence potentials
(b) Returns at search 1
Evidence
ECA-Step: step-wise sum
ECA-Horizon: horizon gain
Shared: cross-fold normalize  →  clip advantage  →  novelty gate
Potential Φ
−8−6−4
−8
D0−4
D1−7
D2−7
D3+4−3
0
∅ d1 d1, d2 d1, d2+2
0
−2+2
−20 +2−2 + 0
6= 0
Horizon gain: +1
−8 −7
D0 D3−7−(−8)
6=1
6
c= 2, H = 3; before normalizationFigure 3. Comparison of ECA-Step and ECA-Horizon return constructions on an illustrative trajectory with potentials (−8,−4,−7,−7)
and step-wise increments (+4,−3,0) (c= 2, H= 3 ).(a)The first search raises the potential from −8to−4, followed by a noisy
second search ( −3) and a redundant third search ( 0).(b)Clipping each step to [−2,2] under ECA-Step saturates the +4gain and −3
drop to +2and−2, yielding zero return at search 1 ( US
i,1= 0). ECA-Horizon preserves the horizon gain from D0toD3, yielding
UH
i,1= (−7−(−8))/6 = 1/6before the shared normalization and gating stages.
need in the preceding <think> block. Let Mi,k,t∈ {0,1}
indicate whether generated policy token tbelongs to the
k-th executed and consumed search action, spanning both
its reasoning tokens and its search-call tokens while exclud-
ing retrieved observations and the final-answer action. We
combine the global outcome advantage and local evidence
residual as:
Ai,t=Aout
i+λHX
k=1Mi,k,teZi,k, λ= 0.1,(8)
so that the auxiliary term adjusts search-action advantages
within [−0.2,0.2] while final-answer tokens are updated
solely by Aout
i. With probability ratio ρi,t(θ) =π θ(ai,t|
hi,t)/πθ0(ai,t|hi,t)over generated policy tokens Pi, we
perform a single fresh-batch policy gradient update per roll-
out batch ofBtrajectories:
L(θ) =−1
BBX
i=1X
t∈P iρi,t(θ) sg(A i,t),(9)
without ratio clipping, KL penalties, or entropy regulariza-
tion. Because our GRPO baseline optimizes Eq. 9 with
λ= 0 , comparing ECA against GRPO directly isolates the
effect of the search-local evidence residual λeZi,k. In all-
failure rollout groups where Aout
i= 0, variation in eZi,k
continues to provide a structured learning signal for search
actions (formal properties in Appendix C).
4. Experimental Setup
Models and training protocol.We conduct experiments
on Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. All RL
methods are trained on an 8,192-question pool (4,096 from
Natural Questions and 4,096 from HotpotQA), with an inde-
pendent 1,024-question in-domain development set. Train-
ing runs for 256 optimizer updates with learning rate 10−6,sampling 32 questions with G= 8 rollouts per question at
each update (256 trajectories per step; 65,536 total). Forward
passes use BF16 mixed precision with FP32 master weights
and AdamW states (with CPU optimizer offloading at 7B).
Search environment.Within each episode, the agent can
take up to four actions (at most H= 3 search calls followed
by a final answer), generating up to 512 tokens per action
within a 4,096-token context window. Rollouts are sampled
at temperature 1.0 during training and decoded greedily at
evaluation. Retrieval is served by a frozen E5-base-v2 re-
triever [ 25] over Wikipedia18, returning the top-3 passages
per query via exact FP32 inner-product search. Malformed
or unfinished episodes are retained in evaluation and scored
as incorrect if they fail to produce a matching answer.
Baselines and checkpoint selection.We compare ECA-
Step and ECA-Horizon against three RL baselines repro-
duced under the same 8,192-question training pool and
65,536-trajectory budget: (i)GRPO( λ= 0 in Eq. 8), (ii)
RLOO[ 1], and (iii)GiGPO (w/std)[ 4], alongside the un-
trainedBase agent. We also report published results for
R1-Instruct, Search-R1, ZeroSearch, and StepSearch from
Feng et al. [4], and TIPS from Xie et al. [28]. Checkpoints
are evaluated every 32 updates on the 1,024-question de-
velopment set (breaking ties toward earlier steps), and the
single best development checkpoint per run is used across
all test suites and transfer evaluations. Both ECA variants
useλ= 0.1 , selected on the development set for ECA-Step
and transferred directly to ECA-Horizon without retuning
(Appendix B).
Evaluation benchmarks and metrics.Following Search-
R1 and GiGPO [ 4,8], we evaluate on seven QA bench-
5

marks totaling 51,713 test questions: two in-domain datasets
(NQ, HotpotQA) and five out-of-domain datasets (TriviaQA,
PopQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle).
Our primary metric is strict question-level Exact Match
(EM). For question q, leteq= 1 if the normalized non-
empty prediction matches any normalized ground-truth alias
(after lowercasing and stripping punctuation, articles, and
extra whitespace), and eq= 0 otherwise. We report the
unweighted macro-average across the seven datasets:
EM =100
77X
s=11
|Qs|X
q∈Q seq,(10)
whereQ sis the test set of benchmarks(Appendix B.3).
Zero-shot transfer to MultiHop-RAG.To test whether
learned search behaviors generalize to a new corpus and
longer documents without further tuning, we evaluate the
selected Search QA checkpoints zero-shot on MultiHop-
RAG [ 21]. The agent retrieves top-3 chunks (using E5-base-
v2 and up to three searches within a 16,384-token context)
over 6,532 chunks segmented from 609 news articles. We
report performance on the 816 multi-hop inference questions
following the TreeHop protocol [13].
5. Results and Analysis
5.1. Main Results on Search QA
Table 1 presents strict question-level EM across the seven
Search QA benchmarks. Starting from the untrained Base
agents ( 21.75 macro EM at 3B and 29.93 at 7B), outcome-
supervised RL substantially improves multi-turn retrieval
and answering, with GRPO reaching 37.19 (3B) and 37.82
(7B). Augmenting terminal outcome supervision with our
evidence-only credit signal yields consistent further gains
across both model scales. At 3B,ECA-StepandECA-
Horizonreach 38.05 and38.22 macro EM, outperforming
GRPO by +0.86 and+1.03 percentage points (pp) and im-
proving over the Base agent by +16.29 and+16.46 pp,
respectively. At 7B, the advantage of evidence-based credit
is even more pronounced:ECA-Stepreaches 41.65 EM
(+3.83 pp over GRPO) andECA-Horizonachieves 42.62
EM, surpassing GRPO by +4.80 pp, RLOO ( 40.56 ) by
+2.06pp, and the Base agent by+12.69pp.
ECA-Horizon also outperforms the state-grouped GiGPO
(w/std) baseline at both scales ( 38.22 vs.36.96 at 3B, +1.26
pp;42.62 vs.41.09 at 7B, +1.53 pp). Compared against
published likelihood-shaping and search RL baselines, ECA-
Horizon exceeds TIPS ( 33.60 at 3B and 41.71 at 7B [ 28])
by+4.62 and+0.91 pp and Search-R1 ( 32.5 and38.5) by
+5.72and+4.12pp.
Examining individual benchmarks reveals two notable
trends. First, the benefit of ECA scales strongly with modelcapacity: moving from 3B to 7B widens ECA-Horizon’s
margin over GRPO from +1.03 pp to +4.80 pp, with large
gains on HotpotQA ( +4.09 pp), Bamboogle ( +8.00 pp),
and NQ ( +5.57 pp). Because ECA uses the policy itself as
the frozen scorer Φi(D), a stronger 7B backbone provides
sharper answer-likelihood estimates while avoiding GRPO’s
early plateau on multi-hop retrieval. Second, preserving
horizon-level gains (ECA-Horizon) outperforms step-wise
clipping (ECA-Step) on aggregate EM at both scales, es-
pecially on 7B 2WikiMultiHopQA ( 39.35 vs.35.95 ) and
Bamboogle ( 40.00 vs.37.60 ), while ECA-Step remains com-
petitive on splits such as 7B MuSiQue (17.75vs.16.88).
5.2. Ablations on Credit Assignment Design
Table 2 isolates the core design choices of ECA at 3B
(λ= 0.1 ). First,whereauxiliary credit is applied along
the trajectory is critical. Restricting ECA-Step’s credit mask
to the <search> query tokens alone (excluding the preced-
ing<think> tokens) causes macro EM to drop from 38.05
to35.27 , falling below outcome-only GRPO ( 37.19 ). Con-
versely, broadcasting the first-slot evidence advantage across
all trajectory tokens (without local novelty gating) reaches
only36.42 EM. Decidingwhatto query happens during the
reasoning tokens preceding the search call: supervising only
the query tag decouples search planning from query execu-
tion, whereas trajectory-wide broadcasting blurs turn-level
attribution and leaks search credit into the final answer step.
Second, comparingreturn constructionsat 3B, both
ECA-Step ( 38.05 EM,46.76 F1) and ECA-Horizon ( 38.22
EM, 45.80 F1) substantially outperform outcome-only
GRPO ( 37.19 EM) as well as the query-only ( 35.27 EM)
and broadcast ( 36.42 EM) ablations (see Appendix C for the
analysis of cross-fold vs. full-group normalization). Between
the two variants, ECA-Step attains slightly higher token F1
at 3B, whereas ECA-Horizon achieves higher strict EM at
both 3B ( +0.17 pp) and 7B ( +0.97 pp), confirming that
preserving unclipped horizon gains becomes increasingly
beneficial as model capacity scales.
5.3. Signal Recovery in All-Failure Groups and
Agent Behavior
A primary motivation for ECA is recovering informative gra-
dients when every rollout for a question fails terminal EM
(Aout
i= 0). Across the 256 training updates at 3B, ECA-
Horizon produces nonzero auxiliary advantages in 3,213 of
3,285 all-failure question groups ( 97.8% ). Crucially, how-
ever, simply injecting nonzero variance into failed groups is
insufficient on its own: the query-only ablation in Table 2
also generates nonzero signals in 3,371 of3,441 all-failure
groups ( 98.0% ), yet degrades test accuracy below GRPO.
Dense evidence signals translate into stronger policies only
when paired with search-local credit placement that rein-
forces the reasoning behind productive queries.
6

Table 1. Question-level exact match (EM, %) across the seven-dataset Search QA benchmark, with macro average (Avg., Eq. 10). †denotes
published reference results from Feng et al. [4](Table 2) and Xie et al. [28] (Table 1). Best and second-best scores per column within each
model scale are shown inboldand underlined .
Method NQ TriviaQA PopQA Hotpot 2Wiki MuSiQue Bamboogle Avg.
Qwen2.5-3B-Instruct
R1-Instruct†27.0 53.7 19.9 23.7 29.2 7.2 29.3 27.1
Search-R1†34.1 54.5 37.8 32.4 31.9 10.3 26.4 32.5
ZeroSearch†41.4 57.4 44.8 27.4 30.0 9.8 11.1 31.7
StepSearch†– – – 34.5 32.017.4 34.4–
TIPS†43.46 58.80 42.76 31.40 29.25 8.73 20.80 33.60
Base agent 26.20 40.06 23.53 19.64 14.33 6.12 22.40 21.75
GRPO 41.16 58.38 41.24 38.53 37.10 14.32 29.60 37.19
RLOO 42.55 59.68 41.43 35.60 33.41 11.34 24.00 35.43
GiGPO (w/std) 44.63 61.19 45.4933.98 34.54 11.67 27.20 36.96
ECA-Step 42.24 59.74 42.01 38.3838.2613.69 32.00 38.05
ECA-Horizon44.7159.49 43.1339.1237.25 13.41 30.4038.22
Qwen2.5-7B-Instruct
R1-Instruct†21.0 44.9 17.1 20.8 27.5 6.0 19.2 22.4
Search-R1†39.3 61.0 39.7 37.0 40.1 14.6 36.8 38.5
ZeroSearch†43.6 61.851.534.6 35.2 18.4 27.8 39.1
StepSearch†– – – 38.6 36.622.6 40.0–
TIPS†43.38 64.31 44.52 42.95 42.9617.05 36.80 41.71
Base agent 33.55 54.84 36.36 27.09 21.45 10.63 25.60 29.93
GRPO 42.05 59.92 41.58 38.96 36.74 13.49 32.00 37.82
RLOO 45.93 63.38 46.87 40.23 37.88 15.27 34.40 40.56
GiGPO (w/std) 44.46 62.58 45.15 38.54 41.21 16.51 39.20 41.09
ECA-Step 47.06 63.90 46.80 42.48 35.95 17.75 37.60 41.65
ECA-Horizon47.62 64.7246.7043.0539.35 16.8840.00 42.62
(a) 3B macro query repetition (%)
(b) Multi-hop query repetition (%)
RLOO
GRPO
ECA-Step
ECA-Horizon
Base agent
RLOO
GRPO
ECA-Step
ECA-Horizon
HotpotQA
2Wiki
MuSiQue
Bamboogle
RL methods execute 3.00searches/episode (Base: 2.28)
0% 20% 40% 60%54.1%
42.4%
30.3%
24.8%
23.9%−17.6 pp
vs. GRPO
Novelty gate suppresses repeated queries ( −29.2 pp vs. RLOO)0%20%40%60%
35
2632
1931
1842
20
ECA-Horizon cuts multi-hop query repetition to 18%–26%
Figure 4. Episode-level duplicate search query rate (%) on Qwen2.5-3B-Instruct across (a) the seven-dataset macro average and (b) individual
multi-hop QA benchmarks. All RL-trained policies execute the full budget of 3.00 searches per episode (vs. 2.28 for the Base agent);
pairing search-local evidence credit with novelty gating suppresses the redundant repeat queries induced by outcome-only RL.
Examining agent interaction patterns across all 51,713
test questions (Figure 4 and Appendix A, Tables S1–S2)
revealshowECA improves retrieval quality. While the un-
trained Base models average 3.36 (3B) and 2.66 (7B) actionsper episode, all RL-trained methods converge to the full bud-
get of 4.00 actions and 3.00 searches while virtually elimi-
nating malformed actions. At 3B, however, outcome-only
RL frequently pads this three-search budget by repeating
7

Table 2. Credit-assignment ablations on Qwen2.5-3B-Instruct ( λ=
0.1). EM and token F1 are macro averages (%) across the seven
Search QA datasets. Best and second-best scores in each column
are shown inboldand underlined .
3B variant EM F1
GRPO 37.19 45.87
ECA-Step 38.05 46.76
ECA-Horizon38.2245.80
ECA-Step (query only) 35.27 43.56
ECA-Step (broadcast) 36.42 44.96
Table 3. Zero-shot transfer on MultiHop-RAG Inference-816 (EM,
%), evaluated under the shared search agent and TreeHop scoring
protocol. Best and second-best scores in each column are shown in
boldand underlined .
Method 3B 7B
Base 59.93 76.84
GRPO 80.02 82.84
RLOO 83.46 88.11
GiGPO (w/std) 65.93 87.25
ECA-Step84.6888.85
ECA-Horizon 82.4889.22
earlier queries (Figure 4): the fraction of episodes contain-
ing duplicate search queries rises from 23.86% in the Base
agent to 42.40% under GRPO and 54.07% under RLOO. Be-
cause ECA pairs step-local evidence credit with novelty gat-
ing that withholds positive credit from redundant retrievals
(Di,k=D i,k−1 ), it cuts 3B query repetition down to 30.26%
(ECA-Step) and 24.84% (ECA-Horizon), converting wasted
repeat searches into diverse follow-up queries across multi-
hop benchmarks while generating fewer policy tokens ( 200.9
for ECA-Horizon vs. 257.1 for ECA-Step). On a paired
per-question basis against the Base agent, ECA-Horizon con-
verts 12,340 previously wrong answers to correct at 3B (vs.
2,166 regressions) and 8,861 to correct at 7B (vs. 2,158
regressions).
5.4. Zero-Shot Transfer to MultiHop-RAG
Table 3 reports zero-shot transfer to the 816 multi-hop infer-
ence questions of MultiHop-RAG [ 13,21]. On these compo-
sitional inference queries, where the agent must synthesize
facts across multiple news articles, both ECA variants out-
perform all baselines: at 3B, ECA-Step reaches 84.68 EM
(+4.66 pp over GRPO’s 80.02 and+1.22 pp over RLOO’s
83.46 ), while at 7B, ECA-Horizon reaches 89.22 EM (+6.38
pp over GRPO’s 82.84 ,+1.97 pp over GiGPO’s 87.25 , and
+1.11pp over RLOO’s88.11).6. Conclusion
We presented Evidence-based Credit Assignment (ECA),
a training-only credit assignment framework for search-
augmented language models. By evaluating the log-
likelihood of a canonical reference answer under an evidence-
only prompt containing solely the question and deduplicated
retrieved documents, ECA isolates the informational con-
tribution of each search action from surface variations in
intermediate reasoning and query phrasing. Comparing step-
clipped increments (ECA-Step) against telescoping horizon
potential differences (ECA-Horizon) within a shared train-
ing pipeline, we showed that ECA-Horizon consistently im-
proves strict macro exact match over outcome-only GRPO,
RLOO, and state-grouped GiGPO across seven Search QA
benchmarks at both 3B and 7B scales, and transfers ef-
fectively zero-shot to MultiHop-RAG multi-hop inference
queries.
Limitations
Several limitations of our study point to directions for future
work. First, our current empirical evaluation focuses on the
search-augmented question answering domain and has not
yet been extended to broader multi-turn agent settings such
as code generation, interactive debugging, or general tool-
use environments. Adapting evidence-only credit assignment
to these broader agent domains (where intermediate observa-
tions take the form of execution traces, compiler diagnostics,
or structured API responses) is an important direction for fu-
ture work. In addition, computing Φi(D) introduces batched
teacher-forced forward passes during training, although test-
time inference cost remains unchanged.
Second, reference-answer log-likelihood under a sin-
gle canonical alias y∗is an imperfect proxy for doc-
ument utility: a passage may provide an intermediate
bridge entity without immediately raising the probability
of final answer tokens, and document-ID deduplication
does not detect near-duplicate text across different articles.
References
[1]Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh
Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and
Sara Hooker. Back to basics: Revisiting REINFORCE style
optimization for learning from human feedback in LLMs.
arXiv preprint arXiv:2402.14740, 2024. 1, 2, 5
[2]Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun,
Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen
Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng
Chen. ReSearch: Learning to Reason with Search for LLMs
via Reinforcement Learning. InAdvances in Neural Informa-
tion Processing Systems, 2025. 2
[3]Siqi Chen, Zhengxi Lu, Yuchen Yan, Xingyu Wu, Qipeng
Chen, Xin Zhang, Aiting Liu, Chao Deng, Jie Liu, Jin Ma,
8

Weiming Lu, Jun Xiao, and Yongliang Shen. KnowGPO:
Knowledge-grouped policy optimization for search agents. In
Proceedings of the 2026 Conference on Empirical Methods in
Natural Language Processing, pages 53526–53546, Budapest,
Hungary, 2026. Association for Computational Linguistics. 2,
13
[4]Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-
in-group policy optimization for LLM agent training. In
Advances in Neural Information Processing Systems, 2025. 1,
2, 5, 7, 11, 13
[5]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and
Akiko Aizawa. Constructing a multi-hop QA dataset for com-
prehensive evaluation of reasoning steps. InProceedings
of the 28th International Conference on Computational Lin-
guistics, pages 6609–6625, Barcelona, Spain (Online), 2020.
International Committee on Computational Linguistics. 12
[6]Wenyu Huang, Xinyu Hou, Pavlos V ougiouklis, Ruofei Lai,
and Jeff Z. Pan. Beyond Outcome Rewards: Constructing and
Assigning Retrieval Credit for Search Agents.arXiv preprint
arXiv:2610.10179, 2026. 2, 13
[7]Angqing Jiang, Gaoming Zhang, Chaoqun Zhang, Jianchun
Song, Liyuan Kong, Kena Qi, Wei Lin, and Defu Lian. IGSD:
Environment-Verified Hindsight Self-Distillation for Search
Agents.arXiv preprint arXiv:2609.32694, 2026. 2, 13
[8]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan
Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1:
Training LLMs to reason and leverage search engines with
reinforcement learning.arXiv preprint arXiv:2503.09516,
2025. 1, 2, 5
[9]Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettle-
moyer. TriviaQA: A large scale distantly supervised challenge
dataset for reading comprehension. InProceedings of the 55th
Annual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 1601–1611, Vancouver,
Canada, 2017. Association for Computational Linguistics. 12
[10] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance,
Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nico-
las Le Roux. VinePPO: Refining credit assignment in RL
training of LLMs. InProceedings of the 42nd Interna-
tional Conference on Machine Learning, pages 29557–29590.
PMLR, 2025. 2
[11] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield,
Michael Collins, Ankur Parikh, Chris Alberti, Danielle Ep-
stein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina
Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang,
Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.
Natural questions: A benchmark for question answering re-
search.Transactions of the Association for Computational
Linguistics, 7:452–466, 2019. 12
[12] Junbo Li, Peng Zhou, Rui Meng, Meet P. Vadera, Lihong Li,
and Yang Li. Turn-PPO: Turn-level advantage estimation with
PPO for improved multi-turn RL in agentic LLMs. InFind-
ings of the Association for Computational Linguistics: EACL
2026, pages 6227–6243, Rabat, Morocco, 2026. Association
for Computational Linguistics. 2
[13] Zhonghao Li, Kunpeng Zhang, Jinghuai Ou, Shuliang Liu,
and Xuming Hu. TreeHop: Efficient embedding-level queryrewriter.arXiv preprint arXiv:2504.20114, version 3, 2026.
6, 8
[14] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Ed-
wards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman,
Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step. In
International Conference on Learning Representations, 2024.
2
[15] Dongyi Liu, Yifan Niu, Qinwen Wang, Han Xiao, and Jia
Li. PiCA: Pivot-Based Credit Assignment for Search Agentic
Reinforcement Learning.arXiv preprint arXiv:2605.09287,
2026. 2, 13
[16] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel
Khashabi, and Hannaneh Hajishirzi. When not to trust lan-
guage models: Investigating effectiveness of parametric and
non-parametric memories. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto, Canada,
2023. Association for Computational Linguistics. 12
[17] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. Measuring and narrow-
ing the compositionality gap in language models. InFindings
of the Association for Computational Linguistics: EMNLP
2023, pages 5687–5711, Singapore, 2023. Association for
Computational Linguistics. 12
[18] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y . K. Li,
Y . Wu, and Daya Guo. DeepSeekMath: Pushing the limits
of mathematical reasoning in open language models.arXiv
preprint arXiv:2402.03300, 2024. 1, 2, 3
[19] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen.
R1-Searcher: Incentivizing the Search Capability in LLMs via
Reinforcement Learning.arXiv preprint arXiv:2503.05592,
2025. 2
[20] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou,
Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren
Zhou. ZeroSearch: Incentivize the search capability of LLMs
without searching.arXiv preprint arXiv:2505.04588, 2025. 1
[21] Yixuan Tang and Yi Yang. MultiHop-RAG: Benchmarking
retrieval-augmented generation for multi-hop queries.arXiv
preprint arXiv:2401.15391, 2024. 2, 6, 8
[22] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal. MuSiQue: Multihop questions via single-
hop question composition.Transactions of the Association
for Computational Linguistics, 10:539–554, 2022. 12
[23] Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei
Yao, Yong Deng, Xiaofeng Wu, and Zhenzhe Ying. Informa-
tion Gain-based Policy Optimization: A Simple and Effective
Approach for Multi-Turn Search Agents. InInternational
Conference on Learning Representations, 2026. 2, 13
[24] Junzhe Wang, Zhiheng Xi, Yajie Yang, Hao Luo, Shihan
Dou, Tao Gui, and Qi Zhang. Enhancing LLM-based Search
Agents via Contribution Weighted Group Relative Policy Op-
timization.arXiv preprint arXiv:2604.14267, 2026. 2, 13
[25] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Lin-
jun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text
embeddings by weakly-supervised contrastive pre-training.
arXiv preprint arXiv:2212.03533, 2022. 5
9

[26] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai,
Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd:
Verify and reinforce LLMs step-by-step without human an-
notations. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long
Papers), pages 9426–9439, Bangkok, Thailand, 2024. Associ-
ation for Computational Linguistics. 2
[27] Quan Wei, Siliang Zeng, Chenliang Li, Zhongruo Wang,
William Brown, Oana Frunza, Wei Deng, Anderson Schnei-
der, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, and
Mingyi Hong. Reinforcing Multi-Turn Reasoning in LLM
Agents via Fine-Grained Reward Structure and Credit As-
signment.arXiv preprint arXiv:2505.11821, version 3, 2026.
2
[28] Yutao Xie, Nathaniel Thomas, Nicklas Hansen, Yang Fu,
Li Erran Li, and Xiaolong Wang. TIPS: Turn-level
information-potential reward shaping for search-augmented
LLMs. InInternational Conference on Learning Representa-
tions, 2026. 2, 5, 6, 7, 13
[29] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo
Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang,
Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei
Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin,
Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei
Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin,
Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren,
Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical
report.arXiv preprint arXiv:2412.15115, 2024. 2
[30] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. HotpotQA: A dataset for diverse, explainable multi-
hop question answering. InProceedings of the 2018 Confer-
ence on Empirical Methods in Natural Language Processing,
pages 2369–2380, Brussels, Belgium, 2018. Association for
Computational Linguistics. 12
[31] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing
Reasoning and Acting in Language Models. InInternational
Conference on Learning Representations, 2023. 1
[32] Xuhui Zheng, Kang An, Ziliang Wang, Yuhang Wang, and
Yichao Wu. StepSearch: Igniting LLMs search ability via
step-wise proximal policy optimization. InProceedings of
the 2025 Conference on Empirical Methods in Natural Lan-
guage Processing, pages 21805–21830, Suzhou, China, 2025.
Association for Computational Linguistics. 2, 13
[33] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyu-
manshan Ye, Pengrui Lu, and Pengfei Liu. DeepResearcher:
Scaling deep research via reinforcement learning in real-world
environments. InProceedings of the 2025 Conference on Em-
pirical Methods in Natural Language Processing, pages 414–
431, Suzhou, China, 2025. Association for Computational
Linguistics. 1
10

Evidence-Based Credit Assignment
for Search-Augmented Language Models
Supplementary Material
A. Behavior Before and After Training
To examine how reinforcement learning changes agent inter-
action patterns beyond exact-match accuracy, we compare
each selected checkpoint against the untrained Qwen2.5-
Instruct Base model of the same size using the identical
evaluation harness across all 51,713 questions. Table S1
reports dataset-macro averages of the number of model ac-
tions (including search calls, invalid actions, and the terminal
answer), executed retrieval calls, generated policy tokens (ex-
cluding prompts and retrieved passages), and the fraction of
episodes containing at least one malformed action.
Interaction budget, query repetition, and output length.
Before RL training, the Base agents average 3.36 actions
(2.28 searches) at 3B and 2.66 actions ( 1.64 searches) at 7B,
with malformed action rates of 10.62% and1.44% , respec-
tively. After RL training, GRPO, RLOO, ECA-Step, and
ECA-Horizon all converge to the full interaction budget of
4.00 actions and 3.00 searches per question while driving
invalid action rates close to zero ( ≤0.44% ). At 3B, however,
expanding from 2.28 to3.00 searches under outcome-only
supervision causes the policy to frequently re-issue iden-
tical queries: the episode-level duplicate-query rate rises
from 23.86% in the Base agent to 42.40% under GRPO and
54.07% under RLOO. By pairing step-local evidence credit
with novelty gating ( Di,k=D i,k−1 =⇒eZi,k≤0), ECA
suppresses 3B query repetition to 30.26% (ECA-Step) and
24.84% (ECA-Horizon), matching the Base model’s repeti-
tion rate despite executing 3.00 searches per episode (at 7B,
the stronger backbone already maintains low query repeti-
tion below 8.5% across all methods). Output token counts
also show that verbosity alone does not dictate accuracy: at
3B, ECA-Horizon achieves higher macro EM than ECA-
Step ( 38.22 vs.38.05 ) while generating 22% fewer policy
tokens per episode ( 200.9 vs.257.1 ), whereas at 7B ECA-
Horizon generates slightly more tokens than ECA-Step ( 92.0
vs.83.6) and fewer than RLOO (158.6).
Paired question-level transitions.Table S2 tracks
question-by-question correctness transitions from the un-
trained Base model to ECA across the pooled 51,713 test
examples. At 3B, ECA-Horizon corrects 12,340 questions
(23.86% ) that the Base agent answered incorrectly, while
regressing on 2,166 previously correct questions ( 4.19% ),
yielding a net pooled gain of +19.67 pp. At 7B, ECA-
Horizon converts 8,861 wrong answers to correct ( 17.13% )against 2,158 regressions ( 4.17% ), for a net pooled gain of
+12.96pp.
B. Implementation and Reproducibility Details
B.1. Training and Checkpoint Configuration
Table S3 lists the shared hyperparameters for ECA-Step and
ECA-Horizon. Every run trains for 256 updates and is evalu-
ated every 32 updates on the 1,024-question in-domain de-
velopment set, selecting the single best development check-
point per run for all test evaluations. For memory efficiency,
the shared trainer processes microbatches of 8 trajectories
(16,384-token padded budget) at 3B and 4 trajectories (8,192-
token budget with CPU AdamW state offloading) at 7B, ac-
cumulating gradients over all 256 trajectories per update
before taking a single optimizer step.
For our reproduced GiGPO baselines, we follow Feng
et al. [4]with discount factor γ= 0.95 , step-advantage
weight 1.0, and state-similarity threshold 0.9under the same
8,192-question training pool and 65,536-trajectory budget.
B.2. Step-by-Step ECA Training Loop
Each training update proceeds as follows:
1.Rollout generation: Sample G= 8 trajectories for each
of the 32 training questions using πθ0at temperature 1.0,
and compute terminal exact-match rewards Ri∈ {0,1}
and outcome advantagesAout
i(Eq. 1).
2.Evidence state reconstruction: For each trajectory i
and search slot k∈ {1,2,3} , collect the cumulative
set of consumed retrieved passages Di,k, deduplicated
and sorted canonically by document ID, and select the
shortest-token reference answer aliasy∗.
3.Cached potential evaluation: Teacher-force y∗under
the frozen current policy πθ0conditioned on C(x, D i,k)
(Eq. 2), reusing cached log-likelihoods whenever two
prefixes share the same document-ID set within the batch.
4.Return construction: Compute slot-level returns over
theH= 3 horizon using either step-clipped increments
US
i,k(Eq. 4, ECA-Step) or horizon potential differences
UH
i,k(Eq. 5, ECA-Horizon), padding unused trailing slots
with unchanged evidence states (∆ i,j= 0).
5.Cross-fold normalization and novelty gating: Partition
the 8 rollouts per question into two random folds of 4,
normalize each trajectory’s return against the opposite
fold’s mean and floored standard deviation (Eq. 6), and
clamp positive auxiliary credit to zero if Di,k=D i,k−1
(Eq. 7).
11

Table S1. Agent interaction statistics before and after RL across the seven Search QA benchmarks (unweighted macro averages over 51,713
questions). ∆EM is measured relative to the same-size Base agent. Actions count all model turns (including invalid actions and the final
answer); Searches count executed retrieval calls; Dup. Query is the percentage of episodes containing at least one repeated search query;
Tokens count generated policy tokens (excluding prompts and retrieved text); and Invalid is the percentage of episodes with at least one
malformed action.
Method EM (%)∆EM (pp) Actions Searches Dup. Query (%) Tokens Invalid (%)
Qwen2.5-3B-Instruct
Base agent 21.75 +0.00 3.36 2.28 23.86 104.2 10.62
GRPO 37.19 +15.44 4.00 3.00 42.40 178.9 0.00
RLOO 35.43 +13.68 4.00 3.00 54.07 120.8 0.02
ECA-Step 38.05 +16.29 4.00 3.00 30.26 257.1 0.07
ECA-Horizon38.22+16.46 4.00 3.00 24.84 200.9 0.14
Qwen2.5-7B-Instruct
Base agent 29.93 +0.00 2.66 1.64 1.69 72.1 1.44
GRPO 37.82 +7.89 4.00 3.00 0.71 76.7 0.06
RLOO 40.56 +10.63 4.00 3.00 7.07 158.6 0.02
ECA-Step 41.65 +11.72 4.00 3.00 8.37 83.6 0.44
ECA-Horizon42.62+12.69 4.00 3.00 8.42 92.0 0.01
Table S2. Paired question-level correctness transitions from the untrained Base agent to ECA across all 51,713 pooled test questions (count
and percentage of total questions). ∆acc. denotes the net change in pooled accuracy, which weights all questions equally rather than
weighting the seven datasets equally.
Model Variant Wrong→correct Correct→wrong Both correct Both wrong∆acc. (pp)
3B ECA-Step 12,115 (23.43) 2,081 (4.02) 10,186 (19.70) 27,331 (52.85) +19.40
3B ECA-Horizon 12,340 (23.86) 2,166 (4.19) 10,101 (19.53) 27,106 (52.42) +19.67
7B ECA-Step 8,473 (16.38) 2,320 (4.49) 15,275 (29.54) 25,645 (49.59) +11.90
7B ECA-Horizon 8,861 (17.13) 2,158 (4.17) 15,437 (29.85) 25,257 (48.84) +12.96
Table S3. Shared training and evaluation configuration for ECA.
Zero-shot transfer on MultiHop-RAG uses a 16,384-token policy
context.
Hyperparameter Value
Training pool / development set 8,192 / 1,024
Questions per update / rollout groupG32 / 8
Total optimizer updates 256
Optimizer / learning rate AdamW /10−6
Max actions / search budgetH4 / 3
Max generated tokens per action 512
Policy context limit (Search QA) 4,096
Evidence scorer context limit 6,144
Increment clipc/ std floorσ min 2 / 0.25
Normalized advantage clip[−2,2]
Auxiliary coefficientλ0.1
Checkpoint evaluation interval 32 updates
6.Policy gradient step: Add the scaled auxiliary credit
λeZi,k(λ= 0.1 ) to all policy tokens in the k-th consumed
search action (both <think> and<search> tokens), ac-
cumulate the surrogate loss (Eq. 9) across microbatches,
updateθ, and refreshπ θ0←π θfor the next update.B.3. Benchmark Sizes
The seven Search QA test sets contain 51,713 questions in
total: NQ [11] (3,610), TriviaQA [9] (11,313), PopQA [16]
(14,267), HotpotQA [ 30] (7,405), 2WikiMultiHopQA [ 5]
(12,576), MuSiQue [ 22] (2,417), and Bamboogle [ 17] (125).
B.4. MultiHop-RAG Corpus and Transfer Protocol
For MultiHop-RAG transfer, we segment all 609 source news
articles into 256-token spans (under the E5 tokenizer) with a
32-token overlap while preserving article titles and metadata
headers, producing 6,532 index chunks. All models are
evaluated zero-shot with top-3 E5-base-v2 retrieval, greedy
decoding, a maximum of 3 searches (4 actions), and a 16,384-
token context limit.
C. Formal Properties of the Auxiliary Signal
Invariance to reasoning and query rephrasing.For any
fixed question x, canonical target answer y∗, frozen scorer
πθ0, and canonicalized document set D, the potential Φi(D)
in Eq. 2 depends exclusively on (x, D, y∗). Consequently, al-
tering the generated chain-of-thought or search query syntax
without changing the retrieved document IDs leaves Φi(D)
12

Table S4. Mechanism-level comparison of intermediate credit assignment and group optimization methods for search agents.
Method Intermediate signal Additional source or
computationDistinction from ECA
GiGPO [4] Relative returns at repeated/similar
statesState grouping across sampled
trajectoriesGroups outcomes, rather than
scoring answer likelihood from
document sets.
IGPO [23] Mean answer log-likelihood change
on rollout prefixesCurrent-policy teacher forcing;
separate process/outcome
normalizationNormalizes immediate rewards
before discounted accumulation;
ECA constructs returns before
cross-fold normalization.
TIPS [28] Answer-likelihood potential
differencesFrozen/refreshed teacher on
dialogue contextUses dialogue-context
potentials; ECA restricts its
readout to questions and
documents and applies a
search-local mask.
StepSearch [32] Gold-document coverage and
repetition penaltiesSub-question/search supervision
and evidence referencesECA uses answer labels without
gold supporting-passage
annotations.
PiCA [15] History-dependent success-potential
progressPivot-annotated trajectories;
trained reward modelECA reuses the current actor for
a document-only readout.
KnowGPO [3] Gains in satisfied atomic knowledge
conditionsConstructed conditions and
LLM verificationECA requires no external LLM
verifier or constructed atomic
conditions, relying only on
document-ID deduplication and
current-policy answer scoring.
Retrieval credit [6] Coverage, dependency-aware
coverage, answer matchingCorpus-grounded evidence
annotationsCompares scalar versus
query-aligned incorporation;
ECA’s residual covers the full
search action.
CW-GRPO [24] LLM-judged round contributions Retrieval/reasoning judge Reweights outcome advantages;
ECA adds a bounded residual.
IGSD [7] Verified utility difference between
paired queriesSuccessful-sibling hindsight
and counterfactual query
executionWe score consumed evidence
without donor-conditioned
distillation or new query
continuations.
and∆ i,kstrictly unchanged.
Equivalence of ECA-Step and ECA-Horizon without sat-
uration.Whenever |∆i,j| ≤c for all j≥k , the step-
wise clip in Eq. 4 acts as the identity, reducing US
i,kto
(cH)−1PH
j=k∆i,j=UH
i,k(Eq. 5). If |∆i,j| ≤c holds
across all trajectories and slots in a question group, ECA-
Step and ECA-Horizon produce identical normalized ad-
vantages eZi,k. Notably, even if a target trajectory’s own
increments lie within [−c, c] , saturation in an opposite-fold
trajectory can still change the reference mean µ−f(i),k or
standard deviation s−f(i),k , causing the normalized advan-
tages of the two variants to diverge.
Boundedness and signal recovery in all-failure groups.
By construction (Eqs. 6–8), the auxiliary token correction sat-
isfies|λeZi,k| ≤2λ= 0.2 for every search token, and equals0whenever all rollouts in a group collect identical evidence
returns ( Ui,k=µ−f(i),k ). When all Grollouts fail to match
the terminal answer ( R1=···=R G= 0),Aout
i= 0, so
the policy gradient is driven purely by differences in eZi,k
across search actions. Because eZi,kis nonlinearly normal-
ized, clipped, and masked to search actions over trajectory-
dependent states, it serves as a bounded process-shaping
residual rather than an action-independent potential-based
shaping term.
Why full-group normalization attenuates sparse evi-
dence spikes.In terminal outcome GRPO (Eq. 1), rewards
Ri∈ {0,1} are binary, so all successful rollouts share the
same unit scale. Intermediate evidence returns Ui,k, by con-
trast, are continuous and often sparse on difficult multi-hop
questions: suppose a single rollout discovers a high-value
passage with return u >0 while the remaining G−1 rollouts
13

yield 0, giving the group vector (u,0, . . . ,0)∈RG. Under
full-group normalization across all Gtrajectories (including
the evaluated trajectory itself), the sample mean is ¯U=u/G
and the unbiased sample standard deviation is sU=u/√
G.
Whenever u/√
G≥σ min, the self-included z-score of the
successful search becomes
u−¯U
sU=u−u/G
u/√
G=
1−1
G√
G,(S1)
in which the return magnitude ucancels out completely be-
tween the numerator and denominator, assigning the exact
same pre-clip advantage whether the retrieved passage yields
a modest likelihood gain or a decisive breakthrough. Nor-
malizing each trajectory against the disjoint opposite fold
(µ−f(i),k , s−f(i),k in Eq. 6) decouples the baseline statistics
from Ui,kitself. Empirically, replacing 4 + 4 cross-fold
normalization in 3B ECA-Step with full-group normaliza-
tion across all G= 8 trajectories (keeping zero-padding and
novelty gating identical) reduces test macro EM by2.04pp
from38.05to36.01.
D. Comparison with Related Credit Mecha-
nisms
Table S4 contrasts ECA with closely related multi-turn credit
assignment and group optimization methods along three
axes: the source of intermediate supervision, its computa-
tional or annotation requirements relative to outcome-only
rollouts, and how step-level signals enter the policy update.
Whereas corpus-supervised methods rely on gold supporting-
document labels or learned reward models, and dialogue-
prefix likelihood methods (IGPO, TIPS) evaluate answer
probability on the full interaction history, ECA combines
an evidence-only likelihood readout with cross-fold return
normalization, novelty gating, and search-action-local as-
signment.
E. Qualitative Comparisons with GRPO
Tables S5–S7 present three representative test cases in which
ECA-Horizon answers correctly while outcome-only GRPO
fails, illustrating multi-hop bridge-entity retrieval, temporal
disambiguation, and decomposed entity searches.
Each comparison uses the same model size, retrieval cor-
pus, and evaluation harness with the development-selected
checkpoints in Appendix B. All six trajectories execute
three searches with no invalid actions. We report the is-
sued queries, relevant excerpts from retrieved passages, final
answers, and normalized EM, whereNew passage IDs per
searchcounts retrieved document IDs not seen in earlier
turns of the trajectory.
14

Table S5.Following a bridge entity (3B, MuSiQue).Both models retrieve the country name. GRPO repeats the initial query and returns
this intermediate entity; ECA-Horizon follows it to a passage stating the country’s location.
Question:Where is the country with ISO code ISO 3166-2:CV located?
Gold:central Atlantic Ocean.ID:dev_88.
GRPO ECA-Horizon
1.Where is the country with ISO code ISO 3166-2:CV
located
2.Where is the country with ISO code ISO 3166-2:CV
located
3.Where is the country with ISO code ISO 3166-2:CV
located1.Where is the country with ISO code ISO 3166-2 CV
located
2.Cabo Verde location
3.Cabo Verde location map
Search 1:ISO 3166-2:CV(passage 18933183).
“ISO 3166-2:CV is the entry for Cabo Verde in ISO 3166-2”Search 1:ISO 3166-2:CV(passage 18933183).
“ISO 3166-2:CV is the entry for Cabo Verde in ISO 3166-2”
Search 2:Cape Verde(passage 5648183).
“an island country spanning an archipelago of 10 volcanic islands in
the central Atlantic Ocean.”
New passage IDs per search:3 / 0 / 0New passage IDs per search:3 / 3 / 1
Final answer:Cabo Verde (EM = 0)Final answer:central Atlantic Ocean (EM = 1)
Table S6.Resolving the award year before the club year (3B, Bamboogle).GRPO follows a retrieved passage about the 2012 award to
Messi. ECA-Horizon retrieves the 2007 winner, Kaká, and a passage linking him to Real Madrid in 2011–12.
Question:For which club did the winner of the 2007 Ballon d’Or play for in 2012?
Gold:Real Madrid.ID:test_104.
GRPO ECA-Horizon
1.2007 Ballon d’Or winner club 2012
2.Lionel Messi club 2012
3.Lionel Messi club 20121.For which club did the winner of the 2007 Ballon
d’Or play for in 2012
2.2007 Ballon d’Or Kaká club 2012
3.Kaká 2012 club
Search 1:2012 FIFA Ballon d’Or(passage 17186262).
“The odds-on favourite Lionel Messi won the award.”Search 1:2007 Ballon d’Or(passage 20994564).
“was awarded to Kaká.”
Search 3:Kaká(passage 13277486).
“In 2011–12, Real Madrid won La Liga with a record 100 points
that year, with Kaká providing nine assists”
New passage IDs per search:3 / 3 / 0New passage IDs per search:3 / 3 / 2
Final answer:Barcelona (EM = 0)Final answer:Real Madrid (EM = 1)
15

Table S7.Separating two entity searches (7B, 2WikiMultiHopQA).GRPO’s repeated joint query retrieves an unrelated longevity record.
ECA-Horizon retrieves both biographies; their returned dates support the correct lifespan comparison.
Question:Who lived longer, Horace Lawson Hunley or Ernst Ziehm?
Gold:Ernst Ziehm.ID:dev_12551.
GRPO ECA-Horizon
1.Horace Lawson Hunley lifespan Ernst Ziehm lifespan
2.Horace Lawson Hunley lifespan Ernst Ziehm lifespan
3.Horace Lawson Hunley death Ernst Ziehm death1.Horace Lawson Hunley
2.Ernst Ziehm
3.Horace Lawson Hunley died
Search 1:Waldo McBurney(passage 13943908).
“Ralph Waldo McBurney (October 3, 1902 – July 8, 2009)”Search 1:Horace Lawson Hunley(passage 14541340).
“June 20, 1823 in Sumner County, Tennessee – October 15, 1863
off Charleston, South Carolina”
Search 2:Ernst Ziehm(passage 7050528).
“1 May 1867 – 7 July 1962”
New passage IDs per search:3 / 0 / 3New passage IDs per search:3 / 3 / 1
Final answer:Horace Lawson Hunley (EM = 0)Final answer:Ernst Ziehm (EM = 1)
16