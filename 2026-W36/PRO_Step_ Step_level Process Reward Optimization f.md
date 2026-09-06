# PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation

**Authors**: MinKeon Kim, Namjun Lee, Jaekwang Kim

**Published**: 2026-08-31 14:34:10

**PDF URL**: [https://arxiv.org/pdf/2609.01658v1](https://arxiv.org/pdf/2609.01658v1)

## Abstract
Retrieval-Augmented Generation enhances Large Language Models by grounding responses in external knowledge, but multi-hop reasoning remains vulnerable to error propagation, where early retrieval failures confound subsequent steps. Standard outcome-based optimization only rewards the final answer, leaving intermediate retrieval and reasoning errors undetected. While existing process-based methods introduce step-level signals, they still score each step against the final answer, rewarding spurious successes where flawed retrieval coincidentally produces the correct answer. Step-level supervision in RAG requires evaluating both logical validity and evidential grounding at each step. We introduce PRO-STEP: we train a generative PRM that evaluates both dimensions, employ PRM-guided value tree search to construct preference pairs contrasting valid steps against flawed ones, and optimize the policy via step-level Direct Preference Optimization. Experiments on single and multi-hop QA datasets demonstrate that PRO-STEP achieves the best average EM and F1 across five benchmarks. Code, models, and training data are publicly available at https://github.com/keemminnke/PRO-Step.

## Full Text


<!-- PDF content starts -->

PRO-Step: Step-level Process Reward Optimization for
Retrieval-Augmented Generation
MinKeon Kim • Namjun Lee • Jaekwang Kim*
Department of Applied Artificial Intelligence,
Convergence Program for Social Innovation,
Sungkyunkwan University, Seoul, South Korea
alsrjs0418@skku.edu,namejun12000@g.skku.edu,linux@skku.edu
Abstract
Retrieval-Augmented Generation enhances
Large Language Models by grounding re-
sponses in external knowledge, but multi-hop
reasoning remains vulnerable to error propa-
gation, where early retrieval failures confound
subsequent steps. Standard outcome-based op-
timization only rewards the final answer, leav-
ing intermediate retrieval and reasoning errors
undetected. While existing process-based meth-
ods introduce step-level signals, they still score
each step against the final answer, rewarding
spurious successes where flawed retrieval co-
incidentally produces the correct answer. Step-
level supervision in RAG requires evaluating
both logical validity and evidential grounding
at each step. We introduce PRO-STEP: we
train a generative PRM that evaluates both
dimensions, employ PRM-guided value tree
search to construct preference pairs contrast-
ing valid steps against flawed ones, and op-
timize the policy via step-level Direct Pref-
erence Optimization. Experiments on single-
and multi-hop QA datasets demonstrate that
PRO-STEPachieves the best average EM and
F1 across five benchmarks. Code, models,
and training data are publicly available at
https://github.com/keemminnke/PRO-Step.
1 Introduction
Large Language Models (LLMs) (Singh et al.,
2026; DeepSeek-AI, 2025) have demonstrated ca-
pabilities across a wide range of reasoning tasks,
including mathematics and code generation. How-
ever, they still suffer from hallucinations, often gen-
erating plausible but unsupported claims when they
lack relevant or up-to-date knowledge. Retrieval-
Augmented Generation (RAG) (Lewis et al., 2020)
addresses these issues by equipping LLMs with ex-
ternal knowledge sources, improving factual accu-
racy and timeliness. However, traditional RAG fol-
lows a linear retrieve-then-generate workflow. This
*Corresponding author.design struggles with multi-step reasoning tasks
that require iterative retrieval and dynamic query
reformulation. To overcome this, recent work has
proposed agentic RAG systems (Li et al., 2025a;
Singh et al., 2025), enabling models to dynamically
decide when and what to retrieve through iterative
retrieval, query rewriting, and self-reflection mech-
anisms (Jiang et al., 2023; Su et al., 2024; Asai
et al., 2024).
Early approaches (Singh et al., 2025) relied on
prompt engineering, while Supervised Fine-Tuning
(SFT) methods directly optimized model parame-
ters for retrieval-augmented workflows. More re-
cently, outcome-supervised Reinforcement Learn-
ing (RL) (Ouyang et al., 2022) has achieved im-
provements: Search-R1 (Jin et al., 2025a) and R1-
Searcher (Song et al., 2025) incorporate search
engines into the LLM’s environment and use final-
answer correctness as the reward signal, demon-
strating that end-to-end RL can enhance agentic
RAG capabilities.
However, outcome-based optimization presents
limitations in multi-hop reasoning (Zhang et al.,
2025a). First, a minor retrieval error in the early
steps, such as an imprecise query or an irrelevant
document, propagates through subsequent reason-
ing steps, but outcome-based optimization lacks the
means to detect or penalize these intermediate fail-
ures. Second, because the reward signal is derived
solely from final-answer correctness, it is sparse.
The model receives feedback only after completing
the entire reasoning chain, requiring more training
data to converge.
Process Reward Model (PRM) approaches
(Lightman et al., 2024; Luo et al., 2024; Wang et al.,
2024; Zheng et al., 2025a) address this limitation by
providing step-level feedback that guides models
along correct reasoning trajectories. However, ap-
plying existing PRM approaches to RAG presents
two challenges. First, they focus on logical con-
sistency in domains like mathematics and cannot
1
arXiv:2609.01658v1  [cs.CL]  31 Aug 2026

verify whether a claim is factually grounded in re-
trieved documents. Second, most PRM approaches
produce only scalar scores without explaining why
a step is flawed, limiting the richness of the super-
vision signal.
To address these challenges, we propose PRO-
STEP, a framework that integrates process-level
supervision into RAG optimization. PRO-STEP
consists of two stages: (1)Generative PRM train-
ing.We generate diverse Chain-of-Thought trajec-
tories where reasoning and retrieval are interleaved,
and train a Generative PRM that evaluates each step
with a correctness label and a rationale. (2)Policy
optimization.We construct step-level preference
pairs via PRM-guided value tree search (VTS), and
apply Direct Preference Optimization (DPO) to
train the policy model to reason correctly at each
step. Our contributions are as follows:
•We identify that existing process-level RAG
methods still optimize against final-answer
outcomes, failing to penalize spurious reason-
ing because they lack mechanisms to evaluate
both logical validity and evidential grounding
at each step.
•We propose PRO-STEP, a framework that
combines Generative PRM training, PRM-
guided VTS for preference pair construction,
and process-supervised DPO. By providing
dense step-level feedback, this approach mit-
igates the reward sparsity issue inherent in
outcome-based methods, enabling efficient op-
timization for retrieval-augmented reasoning.
•Through experiments on single-hop and multi-
hop QA datasets, we demonstrate that PRO-
STEPachieves the best average EM and F1
among the evaluated methods across five QA
benchmarks.
2 Related Work
2.1 Retrieval-Augmented Generation
Early RAG systems followed a retrieve-then-
generate paradigm, appending fixed passages to the
input prompt (Asai et al., 2024). While effective
for single-hop queries, this approach struggles with
multi-hop scenarios necessitating iterative retrieval
and reasoning. Prompt-based methods introduced
uncertainty-triggered retrieval (FLARE; Jiang et al.
2023), dynamic query reformulation (DRAGIN;
Su et al. 2024), and interleaved reasoning-retrievalchains (Trivedi et al. 2023). RL-based frameworks
subsequently enabled autonomous retrieval deci-
sions: Search-R1, R1-Searcher, and related sys-
tems incorporate search environments using final-
answer correctness as the reward, showing that
end-to-end outcome-based RL improves agentic
retrieval performance (Singh et al., 2025). Despite
this success, outcome-only rewards remain sparse
and provide no signal on intermediate retrieval or
reasoning quality, motivating recent shifts toward
process-level supervision.
Process-level supervision has been applied to
RAG through step-level rewards (Li et al., 2025b;
Wang et al., 2026; Zheng et al., 2025b; Wu et al.,
2026) and tree search (Peng and Wei, 2025; Zhang
et al., 2025a). However, these methods still score
intermediate steps against final-answer correctness,
failing to filter out spurious successes where flawed
retrieval coincidentally produces the correct an-
swer.
2.2 Process Reward Model Approaches
PRM approaches provide step-level verification
for multi-step reasoning. Lightman et al. (2024)
first demonstrated that PRM-based verification im-
proves mathematical reasoning by independently
verifying steps. Subsequent works scaled PRM
training using automated Monte Carlo rollouts
(OmegaPRM; Luo et al. 2024) and step-level out-
come estimation (Math-Shepherd; Wang et al.
2024).
Moving beyond scalar evaluation, GenPRM
(Zhao et al., 2025) introduced a generative
paradigm where the model articulates rationales
before assigning a correctness label. By generating
these rationales, the PRM not only scores steps but
explicitly identifies errors, providing richer feed-
back for policy optimization.
However, existing PRM research remains con-
centrated on formal reasoning, where step correct-
ness is logically verifiable. RAG environments
present a different challenge: each step interleaves
internal reasoning with external retrieval, requiring
verification of both logical validity and evidential
grounding in retrieved passages. While VersaPRM
(Zeng et al., 2025) extends PRM approaches to mul-
tiple domains, it does not address the dependency
between retrieval and reasoning. PRO-STEPad-
dresses this gap by training a Generative PRM that
evaluates both dimensions, providing step-level sig-
nals that guide value tree search for preference pair
construction and policy optimization.
2

BAD
GOOD
Rationale 
GOODRelevant
query
GOODIrrelevant
search
Error
recoveryLogical
stepStage 1: Process Reward Modeling
Candidate 
TrajectoriesJudge LLMc<Think>
Reasoning
<Sear ch>
Query
<Document>
Retrieved Doc
<Answer>
Final Answer
Policy
ModelPRMDPO Loss
Reward: 1
Reward: 0
PRMPRM-guided value tr ee sear ch
1 0 1Stage 2: Policy Optimization
Chosen  
...Rejected
...
Process-Supervised
Policy ModelFigure 1:PRO-STEPOverview:Detailed architecture of the process-supervised optimization framework.
3 PRO-STEP
We introduce PRO-STEP, a two-stage framework.
Section 3.1 describes the construction of process-
supervised data, including trajectory generation,
step annotation, and generative PRM training. Sec-
tion 3.2 presents policy optimization using a PRM-
guided value tree search and step-level DPO.
3.1 Process Reward Modeling
Although existing PRM approaches evaluate log-
ical steps in mathematical reasoning, they lack
mechanisms for factual grounding in external ev-
idence, failing to outperform majority voting in
RAG contexts (Figure 3). To address this limita-
tion, we construct a process-supervised dataset by
generating reasoning trajectories, annotating step-
level labels, and training a generative PRM specific
to RAG.
Trajectory generation.We formulate trajec-
tory generation as a multi-turn interaction be-
tween the policy model πθ(using the prompt
in Section A) and an external search engine
E.πθgenerates its internal reasoning within
<think>...</think> . When external factual
grounding is necessary, the model triggers retrieval
by generating <search>...</search> . Upon de-
tecting a search request, the system halts genera-
tion, queries Ewith the generated search string, and
appends the retrieved documents back into the se-
quence as <documents>...</documents> . This
interleaved generation-and-retrieval cycle contin-
ues until the model outputs a final prediction for-
matted as <answer>...</answer> or exhausts the
maximum interaction steps. Using this procedure,
we sample 16 candidate reasoning trajectories per
question from 2,000 instances drawn from Hot-
potQA (Yang et al., 2018) and MuSiQue (Trivediet al., 2022). After discarding incomplete paths or
those lacking formatted answers, we retain 31,728
trajectories.
Step annotation.We use QwQ-32B (Qwen
Team, 2025b) to annotate each model-generated
stepstbased on the evaluation prompt provided
in Section A. Given qand the preceding context
s≤t, QwQ-32B generates a rationale vtand a bi-
nary correctness label rt∈ {0,1} . Unlike prior
PRM methods that discard all subsequent steps
once an error occurs, we retain the entire trajectory.
In RAG contexts, because πθcan recover from an
incorrect step by issuing a revised search query,
a local failure ( rt= 0) does not invalidate subse-
quent steps. All 31,728 trajectories are annotated to
formDPRM(dataset statistics in Section B, labeling
rubric in Section A, and annotation quality audit in
Section E).
Generative PRM training.We train a gener-
ative process reward model πψ(initialized from
DeepSeek-R1-Distill-8B (DeepSeek-AI, 2025;
Qwen Team, 2025a)) on DPRM. The model is op-
timized to generate both the rationale vtand the
binary correctness label rtby minimizing the fol-
lowing objective:
LPRM=−E (q,τ,v t,rt)∼D PRMh
logπ ψ(vt|q, s≤t)
+ logπ ψ(rt|q, s≤t, vt)i
(1)
where the expectation is computed over the model-
generated steps. At inference, πψfirst generates the
rationale, and then outputs the binary prediction:
ˆr(st) = arg max
r∈{0,1}πψ(r|q, s ≤t, vt)(2)
3

Question:  Which American actor , screenwriter , poet and singer died in 1999 and acted in Amok Time?
Gold Answer:  DeForest Kelley
Step 1 1
✔ Grounded, specific
<documents> Doc 1: "...DeForest Kelley died on June 1 1, 1999..." </documents>
Step 2 1
✔ Verifies  
     before answering
<documents> No mention of Richard Mulligan in Amok Time cast </documents>
Step 3a 1✔ New search strategy
Pivots from failed entityStep 3b 0✗ Redundant search
Repeats same failed query
→ Answer: DeFor est Kelley ✔ → Stuck: "Richard Mulligan" ✗
→ DPO Training Pair: Chosen  (path through Step 3a) vs Rejected  (path through Step 3b)User  Query
search: "Amok Time cast who died 1999"1
depth 1
❌ "Mulligan "search: "verify 
Mulligan in cast"1
❌ "Mulligan"❌ "Mulligan"depth 2
search: "Amok
Time cast list"
→ finds Kelley ✔1
search: "Mulligan
Amok Time"
✗ redundant0
search: "Mulligan
Amok Time cast"
✗ same query❌ "Mulligan"depth 3
select best / worst
Chosen
Path: 1→3→6Rejected
Path: 1→3→7
PRM=1 PRM=0 terminal (wrong) question<think> Goal: US actor in "Amok Time" who died in 1999. </think> 
<search> Amok Time cast died 1999 </search>
<think> Mulligan died in 1999. Verify his appearance in "Amok Time". </think> 
<search> Richard Mulligan Amok Time cast </search>
<think> Mulligan not in cast. Pivot to general
search. </think> 
<search> Amok Time cast list </search> 
→ Retrieved: DeFor est Kelley  ✔<think> Mulligan died in 1999. Double-check his
role. </think> 
<search> Richard Mulligan Amok Time </search>
→ Irrelevant results again ✗
Process Reward Model Evaluation
PRM = 1  : Specific search query , new strategy after failure, entity verification
PRM = 0  : Redundant query , repeated failed search, no new information0Figure 2: Example of tree-search data generation with a PRM. Left: reasoning trace showing PRM evaluation
at each step. Right: search tree where green nodes (PRM=1) indicate sound reasoning and red nodes (PRM=0)
indicate flawed steps. The best-scoring path becomes chosen and the worst becomes rejected for DPO training.
3.2 Policy Optimization
Standard process-level methods evaluate interme-
diate steps by simulating trajectories to comple-
tion and scoring the final outcome. However, this
approach lacks mechanisms to penalize flawed
reasoning that spuriously reaches the correct an-
swer. Unlike prior process-based methods that
still rely heavily on final outcomes, PRO-STEP
actively filters out these spurious successes by us-
ing a PRM-guided value tree search to explicitly
penalize flawed intermediate steps.
PRM-guided Value Tree Search.Following
modern tree search paradigms such as AlphaGo
Zero (Silver et al., 2017), we adopt a value tree
search framework that explicitly replaces stan-
dard random rollouts with direct value estimation.
Within this framework, each node represents a state
st= (q, s ≤t), and an action corresponds to sam-
pling the next model-generated step from πθ. If
the sampled step is a <search> action, Eappends
the corresponding <documents> block before the
next action is taken. The tree is built via iterating
selection, expansion, and evaluation using upper
confidence bound (UCB)-based selection (Kocsis
and Szepesvári, 2006) and back-propagation (de-
tailed in Appendix C).
For a terminal node, the reward is the depth-
discounted F1 score Q(sT). For a non-terminal
nodest, we directly use the binary prediction from
πψ:
Q(st) = ˆr(s t)·γd(st)(3)
where γ∈(0,1) penalizes long trajectories, and
d(·)denotes the depth of the node. The prediction
ˆr(st)is computed after node expansion and propa-gated to all ancestor nodes, updating the expected
value ¯Q(s) without inflating visit counts. Detailed
VTS hyperparameters and a branching-factor abla-
tion are provided in Section C.
Preference data construction.For each parent
node, we rank its child nodes based on a value
function that balances the outcome quality with the
assessment fromπ ψ:
V(s) = ¯Q(s) +α·ˆr(s)(4)
Here, ¯Q(s) aggregates the quality of descendant tra-
jectories through back-propagation, while ˆr(s)∈
{0,1} assesses the correctness of step s. For any
sibling pair siandsjwhere V(si)−V(s j)> δ ,
we assign the higher-valued node as the chosen
step ( s+=si) and the other as the rejected step
(s−=sj).
DPO training.We optimize πθusing step-level
DPO (Rafailov et al., 2023). First, we define the
implicit utilityu θ(x, y t):
uθ(x, y t) =βlogπθ(yt|x, y <t)
πref(yt|x, y <t)(5)
Using this utility, we minimize the DPO loss, mask-
ing<documents>blocks from the computation:
LDPO(θ) =−E Ddpoh
logσ(u θ(x, yw
t)
−uθ(x, yl
t)i (6)
where Ddpois the step-level preference dataset of
tuples (x, y <t, yw
t, yl
t). Here, xdenotes the ques-
tion with system prompt, y<tis the shared reason-
ing prefix, yw
tandyl
tare the chosen and rejected
steps at depth t, and the hyperparameter βcontrols
the KL constraint.
4

Table 1: Comparison of different RAG methods across 5 benchmarks using EM and F1 metrics.Bolddenotes
the best performance, and underlining denotes the second-best.†indicates statistically significant improvements
(p <0.05 ) over representative RL baselines (Search-R1, ReasonRAG) using a two-sided paired t-test. Full
significance tests, including comparisons with StepSearch, are detailed in Appendix F.
Type MethodPopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Zero-shotDirect 14.8 19.0 18.7 26.6 24.6 29.3 12.0 19.2 4.0 11.9 14.8 21.2
Standard RAG 37.4 44.1 31.6 41.9 28.3 34.6 16.8 25.1 4.8 12.3 23.8 31.6
AdvancedFLARE 15.5 19.6 18.8 26.6 24.6 29.0 12.0 19.2 4.0 11.6 15.0 21.2
Self-RAG 30.0 38.6 18.2 29.6 13.2 25.0 5.6 18.0 4.7 11.8 14.3 24.6
IRCoT 32.7 40.7 31.2 42.8 26.2 35.1 23.2 33.6 6.2 13.0 23.9 33.0
Search-o1 10.7 15.0 18.5 26.5 19.6 23.4 25.6 38.5 4.5 11.8 15.8 23.0
RL-basedAutoRAG 24.1 31.2 14.8 24.1 11.0 20.4 23.2 34.3 6.5 13.1 15.9 24.6
Search-R140.746.8 37.9 49.6 34.9 42.5 33.6 43.6 13.0 21.2 32.0 40.7
ReasonRAG 37.8 44.9 36.4 47.5 39.8 46.338.446.9 10.6 19.2 32.6 41.0
StepSearch 39.2 45.038.750.7 40.4 47.1 33.6 44.213.8 23.1 33.2 42.0
Ours PRO-STEP40.5 47.4†38.7 51.6†44.1†51.4†36.8 47.612.5 22.4 34.5†44.1†
4 Experiments
In this section, we present the experimental settings
and results.
4.1 Experimental Setup
Datasets and Evaluation Metrics.We evaluate
PRO-STEPon five benchmark datasets spanning
single-hop and multi-hop question answering (QA).
For single-hop QA, we use PopQA (Mallen et al.,
2023). For multi-hop QA, we use HotpotQA (Yang
et al., 2018), 2WikiMultiHopQA (Ho et al., 2020),
Bamboogle (Press et al., 2023), and MuSiQue
(Trivedi et al., 2022). We report Exact Match (EM)
and F1 score using the FlashRAG evaluation frame-
work (Jin et al., 2025b) for reproducibility.
Implementation Details.We employ Qwen2.5-
7B-Instruct (Qwen Team, 2024) as the primary
backbone for PRO-STEPand most baselines. For
specific baselines, we utilize their official model
checkpoints to ensure a fair comparison: Self-RAG
employs its specialized Llama-2-13B variant (Asai
et al., 2024), and AutoRAG utilizes Llama-3-8B-
Instruct (Yu et al., 2024). For retrieval, we use the
2018 Wikipedia dump consisting of 5.9M docu-
ments, indexed with the BGE-base-en-v1.5 (Xiao
et al., 2024) retriever, and set the number of re-
trieved passages to k= 3 across all methods to
ensure consistency. We train PRO-STEPon a cu-
rated mixture of 5,000 multi-hop QA questions,
comprising 2,000 instances each from thetrainingsplitsof HotpotQA and MuSiQue, and 1,000 from
2WikiMultiHopQA. For PRM training, we sample
16 trajectories for each of the 2,000 questions from
Section 3.1 above. For policy optimization, we
then construct step-level preference pairs via PRM-
guided VTS. Detailed data statistics and filtering
procedures are deferred to Section B.
Baselines.We evaluate PRO-STEPagainst 10
baselines categorized into three groups:Zero-
shot Baselines(Naive Generation, Standard RAG
(Lewis et al., 2020)),Advanced RAGfeaturing
adaptive retrieval mechanisms and iterative reason-
ing capabilities (FLARE (Jiang et al., 2023), Self-
RAG, IRCoT (Trivedi et al., 2023), Search-o1 (Li
et al., 2025a)), andRL-based RAGparadigms
that leverage autonomous exploration and test-
time scaling, includingoutcome-rewardmeth-
ods (AutoRAG, Search-R1 (Jin et al., 2025a))
andprocess-rewardmethods (ReasonRAG (Zhang
et al., 2025a), StepSearch (Zheng et al., 2025b)).
See Section D for training hyperparameters.
4.2 Main Results
The main results comparing PRO-STEPwith base-
line methods across five datasets are presented in
Table 1. From the results, we make the following
key observations:
Overall results.PRO-STEPachieves the highest
average performance across all evaluated methods,
recording 34.5 EM and 44.1 F1 (significance tests
5

Table 2: Component ablation study evaluating the impact of the Process Reward Model (PRM). The table reports
EM and F1 scores across 5 benchmarks.Bolddenotes the better performance.
VariantPopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
PRO-STEP40.5 47.4 38.7 51.6 44.1 51.4 36.8 47.6 12.5 22.4 34.5 44.1
Without PRM 38.5 45.0 36.8 49.3 41.4 48.4 34.4 44.2 12.322.432.7 41.9
Table 3: Impact of different optimization strategies on PRO-STEP’s effectiveness across five benchmarks.Bold
denotes the best performance, and underlining denotes the second-best.
MethodPopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
PRO-STEP(SFT) 38.8 45.7 36.7 48.3 32.9 41.3 31.2 40.4 11.3 20.0 30.2 39.1
PRO-STEP(KTO)41.846.3 32.3 43.5 36.1 42.0 20.8 32.7 8.9 18.5 28.0 36.6
PRO-STEP(DPO)40.5 47.4 38.7 51.6 44.1 51.4 36.8 47.6 12.5 22.4 34.5 44.1
in Section F).
Limitations of heuristic adaptive retrieval.Ad-
vanced RAG methods relying on heuristic triggers,
such as FLARE and Self-RAG, underperform the
much simpler Standard RAG on average (approxi-
mately 15.0 vs. 23.8 EM). Even IRCoT falls behind
Standard RAG on complex multi-hop tasks such
as HotpotQA and 2WikiMultiHopQA. In contrast,
PRO-STEPlearns retrieval timing directly via pref-
erence optimization, bypassing the inefficiencies
of heuristics.
Process vs. outcome in multi-hop reasoning.
Process-level optimization improves performance
on datasets requiring extended logical chains. On
average, PRO-STEPoutperforms the outcome-
based Search-R1 by 2.5 EM and 3.4 F1 points.
This gap widens substantially on complex multi-
hop tasks such as 2WikiMultiHopQA, where our
method achieves a 9.2 EM and 8.9 F1 improve-
ment. The dense step-level feedback alleviates the
reward sparsity problem and improves sample effi-
ciency: PRO-STEPsurpasses Search-R1 using 5k
seed questions, whereas Search-R1 reports roughly
170k training instances.
Comparison with process-reward baselines.
PRO-STEPachieves higher average EM and F1
than ReasonRAG and StepSearch. While these
baselines operate at the process level, they opti-
mize each step against the final F1 score, which
fails to penalize spurious reasoning where flawedretrieval accidentally produces the correct answer.
In contrast, our generative PRM evaluates the logi-
cal and evidential validity of each step, filtering out
such false positives. Notably, prior methods distill
reasoning trajectories from frontier models (e.g.,
GPT-4o (Hurst et al., 2024)), and some addition-
ally invoke multiple LLM calls during inference
for document refinement. PRO-STEPavoids both,
internalizing step-level guidance into the policy
through PRM-guided VTS, yielding higher aver-
age results with a single model at inference.
4.3 Effectiveness of Process Reward Model
To isolate the contribution of our generative PRM,
we compare PRO-STEPagainst a variant without
PRM scoring that constructs preference pairs using
only the depth-discounted F1 reward. As shown
in Table 2, removing the PRM degrades average
performance from 34.5 to 32.7 EM and from 44.1
to 41.9 F1, with the largest drops on multi-hop
datasets such as 2WikiMultiHopQA (-2.7 EM) and
Bamboogle (-2.4 EM). This larger degradation sug-
gests that PRM supervision is particularly benefi-
cial for multi-hop reasoning. Without PRM evalu-
ation, preference pairs rely solely on final-answer
correctness and cannot penalize spurious reasoning
that coincidentally leads to correct answers, which
the PRM filters out by scoring step-level validity.
This validates that considering both logical valid-
ity and evidential grounding is essential for RAG.
Detailed analysis is provided in Section H.
6

1 2 4 8 16 32 64 12845464748495051F1 (%)
PopQA (WMV)
1 2 4 8 16 32 64 12848505254565860F1 (%)
HotpotQA (WMV)
1 2 4 8 16 32 64 1283234363840424446F1 (%)
2WikiMultiHopQA (WMV)
1 2 4 8 16 32 64 128
Number of generated trajectories414243444546474849F1 (%)
PopQA (BoN)
1 2 4 8 16 32 64 128
Number of generated trajectories48505254565860F1 (%)
HotpotQA (BoN)
1 2 4 8 16 32 64 128
Number of generated trajectories283032343638404244F1 (%)
2WikiMultiHopQA (BoN)
Majority Voting WMV BoN PRO-STEP PRM VersaPRM Qwen-Math-PRMFigure 3: Comparison of WMV (top) and BoN (bottom) using a RAG-specific PRM against open-source PRM
approaches across three QA datasets.
4.4 Impact of Optimization Strategies
We compare three optimization strategies on the
same process-supervised data to validate that DPO
is best suited for process-level learning. PRO-
STEP(SFT) trains only on the chosen steps yw
t
via next-token prediction. PRO-STEP(KTO) uses
unpaired binary correctness labels rt∈ {0,1} with
Kahneman-Tversky Optimization (KTO) (Etha-
yarajh et al., 2024). PRO-STEP(DPO) is our de-
fault approach using contrastive pairsyw
t, yl
t.
As shown in Table 3, DPO achieves the highest
average performance with 34.5 EM and 44.1 F1
by providing explicit step-wise contrast between
valid and flawed reasoning. SFT lacks contrastive
signals and cannot learn to recover from intermedi-
ate errors, leading to weaker performance on com-
plex multi-hop tasks such as 11.3 vs. 12.5 EM
on MuSiQue. KTO underperforms even SFT on
average with 28.0 EM and 36.6 F1, as absolute
binary rewards without relative comparison cause
instability across datasets such as 20.8 vs. 31.2
EM on Bamboogle. These results confirm that pair-
wise contrast, rather than positive-only or unpaired
signals, is essential for internalizing the PRM’s
step-level guidance.
4.5 Comparison of PRM Approaches on
Trajectory Reranking
To validate the necessity of a RAG-specific PRM,
we compare our model against existing process re-ward models. We sample 128 reasoning trajectories
per question for 500 instances from PopQA, Hot-
potQA, and 2WikiMultiHopQA using Qwen2.5-
7B-Instruct (Qwen Team, 2024). We evaluate
two models, VersaPRM (Zeng et al., 2025) and
Qwen2.5-Math-PRM-7B (Zhang et al., 2025b),
alongside two trajectory-selection baselines: major-
ity voting (MV), which ignores PRM scores, and
weighted majority voting (WMV), which weights
each candidate by the minimum step score assigned
by a PRM. We also report best-of- N(BoN), which
directly selects the highest-scoring trajectory, while
scaling the candidate pool size Kfrom 1 to 128. To
evaluate the quality of the final selected trajectories,
we report token-level F1.
The results in Figure 3 demonstrate that general-
purpose and math-specific PRM approaches strug-
gle in RAG environments. Under the BoN strategy,
both VersaPRM and MathPRM degrade as Kin-
creases, frequently falling below the MV baseline
across all three datasets. Because they evaluate log-
ical coherence without verifying factual grounding,
these PRM approaches often assign high scores
to hallucinated but plausible trajectories. Conse-
quently, as the candidate pool expands, BoN in-
creasingly selects these confidently flawed trajecto-
ries.
In contrast, our PRM consistently improves F1
as the candidate pool size increases under WMV .
By evaluating both logical validity and the factual
7

PopQA HotpotQA 2WikiMQA0102030405060F1 Score (%)40.445.745.147.451.6 51.4
48.052.0 51.9Retrieval Docs
k=1 k=3 k=5Figure 4: Impact of the number of retrieved documents
(k) on performance.
1 2 All
Retrieval Steps010203040506070F1 Score38.544.647.4
31.447.351.6
21.243.651.4PopQA
HotpotQA
2WikiMQA
020406080100
Data Coverage %
Figure 5: Cumulative average F1 score (bars) and data
coverage (dashed lines) across different retrieval steps
(1, 2, All) on PopQA, HotpotQA, and 2WikiMulti-
HopQA.
grounding of the retrieved documents, our model
filters out spurious trajectories. At K= 128 with
WMV , our PRM achieves46.5 F1on 2WikiMul-
tiHopQA,59.5 F1on HotpotQA, and50.8 F1on
PopQA, outperforming both MV (43.9 / 54.6 / 49.2)
and VersaPRM (43.4 / 53.9 / 49.5) across all three
datasets. Under the BoN strategy, our PRM re-
mains stable as Kgrows, performing comparably
to MV on PopQA and 2WikiMultiHopQA (within
0.2 F1) and substantially outperforming it on Hot-
potQA (+4.9 F1), while VersaPRM and MathPRM
degrade sharply at larger K. These results validate
our motivation for building a RAG-specific PRM
that jointly evaluates logical validity and evidential
grounding, rather than relying on general-purpose
reasoning critics.
4.6 Impact of Retrieval
In this section, we investigate the performance of
PRO-STEPunder different numbers of retrieved
documents and retrieval steps to evaluate its context
utilization and planning adaptivity.Impact of the number of retrieved documents.
Figure 4 compares PRO-STEPunder different top-
kretrieval settings. Increasing kfrom 1 to 3 consis-
tently improves performance across all datasets (av-
erage +6.4 F1), and expanding to k= 5 yields fur-
ther gains. These findings highlight PRO-STEP’s
capacity to effectively leverage additional retrieved
documents, particularly in complex multi-hop sce-
narios.
Impact of the number of retrieval steps.Fig-
ure 5 illustrates the cumulative F1 score and data
coverage at retrieval-step cap K. PRO-STEPau-
tonomously adjusts search depth to task difficulty:
on single-hop PopQA, 65.6% of instances are re-
solved within the first step (cumulative F1 = 38.5),
while on multi-hop HotpotQA and 2WikiMulti-
HopQA only 52.0% and 36.6% are resolved at step
one, with reasoning naturally extending to addi-
tional steps. On 2WikiMultiHopQA, cumulative
F1 climbs from 21.2 ( K= 1 ) to 43.6 ( K= 2 ) as
coverage grows from 36.6% to 77.5%, indicating
that the model successfully acquires critical miss-
ing evidence during its second retrieval. Cumula-
tive F1 increases monotonically across all datasets,
demonstrating that PRO-STEPmitigates the error-
propagation degradation typical of iterative RAG
frameworks (qualitative examples in Section J).
4.7 Generalization Across Backbones
Our main experiments use Qwen2.5-7B-Instruct as
the policy backbone.
We further examine the generalization of PRO-
STEPbeyond this configuration. Specifically, we
evaluate PRO-STEPwith a different model family
(Llama-3.1-8B-Instruct), without instruction tun-
ing (Qwen2.5-7B Base), and at a smaller model
scale (Qwen2.5-3B-Instruct). Table 4 reports re-
sults across all five benchmarks.
As shown in Table 4, PRO-STEPimproves
Llama-3.1-8B-Instruct across all five datasets, in-
creasing the average performance from 5.08 / 13.08
to 16.16 / 22.57 EM / F1 (+11.1 EM and +9.5 F1).
This shows that the gains are not restricted to the
Qwen model family.
Without instruction tuning, Qwen2.5-7B Base
with PRO-STEPachieves 32.85 / 42.01 average
EM / F1, compared with 34.51 / 44.09 for the main
instruction-tuned backbone. The relatively small
average difference (1.66 EM and 2.08 F1) shows
that the framework retains most of its performance
without an instruction-tuned initialization.
8

Table 4: Generalization of PRO-STEPacross different policy backbones. Results are reported using EM and F1.
Bolddenotes the better performance within each comparison.
MethodPopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Llama-3.1-8B-Instruct
Base 5.49 16.14 5.71 14.63 5.50 12.97 7.20 15.60 1.49 6.08 5.08 13.08
PRO-STEP22.94 28.12 18.88 27.12 25.40 31.74 8.80 16.99 4.76 8.86 16.16 22.57
Qwen2.5-7B Base/Instruct
PRO-STEP-Base 40.42 46.04 34.69 47.47 41.09 48.57 36.00 45.38 12.0822.61 32.85 42.01
PRO-STEP-Instruct40.47 47.37 38.73 51.63 44.07 51.43 36.80 47.63 12.4922.41 34.51 44.09
Qwen2.5-3B-Instruct
Search-R1 36.87 42.14 31.82 41.30 35.96 41.86 18.40 27.97 8.07 14.27 26.22 33.51
PRO-STEP39.88 46.10 32.82 44.33 44.10 50.77 28.00 38.96 10.38 19.85 31.04 40.00
Table 5: Final EM on recovered trajectories. nis given
as Search-R1 / PRO-STEP.
DatasetnSearch-R1 PRO-STEP
HotpotQA 1,726 / 1,081 51.354.4
PopQA 915 / 623 44.649.6
2WikiMulti 4,002 / 2,945 47.452.8
Bamboogle 34 / 25 82.484.0
MuSiQue 550 / 418 38.438.5
At the 3B scale, PRO-STEPoutperforms Search-
R1 across all five datasets, improving the average
from 26.22 / 33.51 to 31.04 / 40.00 EM / F1 (+4.82
EM and +6.49 F1). Together, these results show
that PRO-STEPextends across different model
families, alignment settings, and model scales. The
3B configuration additionally uses an SFT warmup
on trajectories from the 7B policy. Additional 3B
results and training analysis are provided in Sec-
tion G.
4.8 Recovery from Retrieval Failure
Outcome-based optimization lacks a direct signal
to detect or penalize intermediate retrieval failures,
allowing early errors to propagate through sub-
sequent reasoning steps. To examine this failure
mode, we evaluate PRO-STEPand the outcome-
based Search-R1 onrecovered trajectories, where
the initial retrieval fails to surface the gold passage
but a later retrieval succeeds. Since this subset is
determined by each model’s own search behavior,
the two systems are evaluated on overlapping but
non-identical question sets.
As shown in Table 5, PRO-STEPachieves higher
final EM than Search-R1 on all five datasets, and itssubsets are consistently smaller because it fails less
often at the initial retrieval. On 2WikiMulti, the
gains hold at every recovery depth kwith sufficient
support (+4.1, +8.3, and +6.8 EM at k= 1,2,3 ),
and PRO-STEPconcentrates its recoveries earlier,
resolving 87% of its subset at k= 1 versus 80%
for Search-R1. The advantage is not driven by
subset selection, as PRO-STEPalso outperforms
Search-R1 when the initial retrieval succeeds, by
12.1 EM on 2WikiMulti and 29.2 EM on Bam-
boogle. Detailed results are provided in Section I.
These results show that PRO-STEPbetter recovers
from retrieval failures, mitigating error propagation
through subsequent reasoning steps.
5 Conclusion
We introduced PRO-STEP, a framework that inte-
grates process-level supervision into RAG to miti-
gate error propagation in multi-hop reasoning. Our
approach trains a generative PRM to evaluate both
the logical validity and the evidential grounding
of intermediate steps. We employ this PRM to
guide VTS, proactively constructing step-level pref-
erence pairs by explicitly filtering out spurious
trajectories that coincidentally reach correct final
outcomes. Using these targeted preference pairs,
we optimize the policy model using DPO to en-
hance its ability to navigate interleaved generation
and retrieval. Experiments demonstrate that PRO-
STEPaddresses intermediate reasoning flaws and
achieves the best average EM and F1 across the
five evaluated QA benchmarks. By mitigating the
limitations of sparse outcome rewards, PRO-STEP
offers an effective approach for fine-grained policy
optimization in multi-hop QA tasks.
9

Limitations
Limitations of QwQ-32B as a PRM labeler.
While frontier closed-source models (e.g., GPT-
4o) might marginally reduce potential label noise,
we utilize QwQ-32B to ensure our pipeline is fully
reproducible without restrictive API licenses. To
validate this choice and quantify reliability, we con-
ducted a 500-sample audit against Claude Opus 4.7
(detailed in Section E). The results demonstrated
substantial agreement ( κ= 0.6104 ) without sys-
tematic permissive bias, confirming that our open-
source approach democratizes process-supervised
training without sacrificing critical step-level accu-
racy.
Trade-offs of open-source, small-data supervi-
sion.While PRO-STEPachieves the highest av-
erage EM and F1, it does not uniformly domi-
nate every dataset. The largest gap appears on
MuSiQue, where PRO-STEPtrails StepSearch by
1.3 EM. We attribute this to our deliberate choice to
avoid closed-source teachers and train on a smaller
pool (5,000 seed questions, 15,877 pairs) than
StepSearch’s GPT-4o-distilled 19k corpus. This is
a transparent trade-off: PRO-STEPis reproducible
without API access, at the cost of some headroom
on the hardest tasks.
Bamboogle’s small test size limits statistical re-
liability.Bamboogle contains only 125 test in-
stances, yielding wide confidence intervals. Our
paired t-test against ReasonRAG returns p= 0.66
(Section F), indicating the apparent gap is statisti-
cally indistinguishable from noise.
PRM-feedback-guided regeneration disrupts
policy alignment.We explored augmenting pref-
erence pairs via PRM-feedback-guided regenera-
tion (Zhao et al., 2025), but the augmented variant
degrades evaluation by 10.7 EM and 9.8 F1 on aver-
age (Section K). Regenerated trajectories drift from
the natural search-and-reason pattern of inference-
time generation, and the outcome-margin filter can-
not detect this stylistic drift. This motivates future
work on action-format-consistent augmentation.
Ethical Considerations
This work presents PRO-STEP, a framework
for optimizing retrieval-augmented reasoning in
large language models. All datasets used in this
study (HotpotQA, MuSiQue, 2WikiMultiHopQA,
PopQA, Bamboogle) are publicly available bench-marks intended for research purposes. Our training
data is derived solely from these existing datasets,
and no personally identifiable information is col-
lected or processed. The proposed framework does
not introduce new risks beyond those inherent to
LLM-based systems, such as residual hallucination
or retrieval of outdated information. Practitioners
deploying this framework in domains should apply
appropriate human oversight.
AI writing assistance was used solely for gram-
mar and language checking of the manuscript.
Acknowledgments
This research was supported by the Basic Science
Research Program through the National Research
Foundation of Korea (NRF) funded by the Korean
government (MSIT) (No. IITP-2026-RS-2024-
00346737), and by grants from the Institute for
Information & Communications Technology Plan-
ning & Evaluation (IITP) funded by the Ministry
of Science and ICT (MSIT), Korea, through the
Global Scholars Invitation Program (No. IITP-
2026-RS-2024-00459638), the Graduate School
of Metaverse Convergence at Sungkyunkwan Uni-
versity (No. IITP-2026-RS-2023-00254129), and
the ICT Challenge and Advanced Network of HRD
(ICAN) support program (No. IITP-2026-RS-2023-
00259497).
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. SELF-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations (ICLR).
DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing rea-
soning capability in LLMs via reinforcement learning.
arXiv preprint arXiv:2501.12948.
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff,
Dan Jurafsky, and Douwe Kiela. 2024. Model align-
ment as prospect theoretic optimization. InProceed-
ings of the 41st International Conference on Machine
Learning, ICML’24.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics
(COLING).
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Os-
trow, Akila Welihinda, Alan Hayes, Alec Radford,
10

et al. 2024. GPT-4o system card.arXiv preprint
arXiv:2410.21276.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP).
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Ji-
awei Han. 2025a. Search-R1: Training LLMs to
reason and leverage search engines with reinforce-
ment learning. InProceedings of the 2nd Conference
on Language Modeling (COLM 2025).
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025b. FlashRAG: A mod-
ular toolkit for efficient retrieval-augmented gener-
ation research. InCompanion Proceedings of the
ACM on Web Conference 2025, WWW 2025, Sydney,
NSW, Australia, 28 April 2025 - 2 May 2025. ACM.
Levente Kocsis and Csaba Szepesvári. 2006. Bandit
based Monte-Carlo planning. InEuropean Confer-
ence on Machine Learning. Springer.
J Richard Landis and Gary G Koch. 1977. The mea-
surement of observer agreement for categorical data.
Biometrics, pages 159–174.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yu-
jia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025a. Search-o1: Agentic search-enhanced
large reasoning models. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP), pages 5420–5438.
Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng,
Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue
Yin, and Xipeng Qiu. 2025b. R3-RAG: Learning
step-by-step reasoning and retrieval for LLMs via
reinforcement learning. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025.
Association for Computational Linguistics.
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2024. Let’s verify step by step. InThe Twelfth In-
ternational Conference on Learning Representations
(ICLR).
Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat
Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei
Shu, Yun Zhu, Lei Meng, Jiao Sun, and AbhinavRastogi. 2024. Improve mathematical reasoning in
language models by automated process supervision.
arXiv preprint arXiv:2406.06592.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness and limitations of parametric and non-
parametric memories. InProceedings of the 61st
Annual Meeting of the Association for Computational
Linguistics (ACL).
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Car-
roll L. Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback. InAdvances in neural information
processing systems 35. Neurips.
Xianshu Peng and Wei Wei. 2025. GRAT: Guiding
retrieval-augmented reasoning through process re-
wards tree search. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics, pages 27861–27875, Vienna, Austria. Asso-
ciation for Computational Linguistics.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Qwen Team. 2024. Qwen2.5 technical report.arXiv
preprint arXiv:2412.15115.
Qwen Team. 2025a. Qwen3 technical report.arXiv
preprint arXiv:2505.09388.
Qwen Team. 2025b. QwQ-32B: Embracing the
power of reinforcement learning. https://qwen.
ai/blog?id=qwq-32b. Blog post. Accessed: 2025.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D. Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model. InAdvances in
Neural Information Processing Systems (NeurIPS).
David Silver, Julian Schrittwieser, Karen Simonyan,
Ioannis Antonoglou, Aja Huang, Arthur Guez,
Thomas Hubert, Lucas Baker, Matthew Lai, Adrian
Bolton, et al. 2017. Mastering the game of Go with-
out human knowledge.nature, 550(7676):354–359.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart,
Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin,
Aiden Low, AJ Ostrow, Akhila Ananthram, et al.
2026. OpenAI GPT-5 system card.arXiv preprint
arXiv:2601.03267.
Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei
Khoei, and Athanasios V Vasilakos. 2025. Agentic
retrieval-augmented generation: A survey on agentic
rag.arXiv preprint arXiv:2501.09136.
11

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-Searcher: Incentivizing the
search capability in LLMs via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. DRAGIN: Dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (ACL).
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics (TACL).
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (ACL).
Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai
Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui.
2024. Math-shepherd: Verify and reinforce LLMs
step-by-step without human annotations. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers).
Zhao Wang, Ziliang Zhao, and Zhicheng Dou. 2026.
ProRAG: Process-supervised reinforcement learning
for retrieval-augmented generation.arXiv preprint
arXiv:2601.21912.
Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu
He, Xinya Du, and Zhiyu Chen. 2026. HiPRAG: hier-
archical process rewards for efficient agentic retrieval
augmented generation. InInternational Conference
on Learning Representations.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack:
Packed resources for general Chinese embeddings.
InProceedings of the 47th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing
(EMNLP).
Tian Yu, Shaolei Zhang, and Yang Feng. 2024.
Auto-RAG: Autonomous retrieval-augmented gen-
eration for large language models.arXiv preprint
arXiv:2411.19443.Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian
Classen, Daewon Chae, Ethan Ewer, Minjae Lee,
Heeju Kim, Wonjun Kang, Jackson Kunde, Ying
Fan, Jungtaek Kim, Hyung Il Koo, Kannan Ram-
chandran, Dimitris Papailiopoulos, and Kangwook
Lee. 2025. VersaPRM: Multi-domain process reward
model via synthetic reasoning data. InProceedings
of the 42nd International Conference on Machine
Learning (ICML).
Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao
Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang,
Derong Xu, Zhaocheng Du, Huifeng Guo, Ruim-
ing Tang, and Xiangyu Zhao. 2025a. Process vs.
outcome reward: Which is better for agentic RAG re-
inforcement learning. In39th Conference on Neural
Information Processing Systems (NeurIPS 2025).
Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen
Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jin-
gren Zhou, and Junyang Lin. 2025b. The lessons of
developing process reward models in mathematical
reasoning. InFindings of the Association for Compu-
tational Linguistics: ACL 2025, pages 10495–10516.
Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou,
Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian,
Biqing Qi, Xiu Li, and Bowen Zhou. 2025. Gen-
PRM: Scaling test-time compute of process reward
models via generative reasoning.arXiv preprint
arXiv:2504.00891.
Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji
Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren
Zhou, and Junyang Lin. 2025a. ProcessBench: Iden-
tifying process errors in mathematical reasoning. In
Proceedings of the 63rd Annual Meeting of the Asso-
ciation for Computational Linguistics. Association
for Computational Linguistics.
Xuhui Zheng, Kang An, Ziliang Wang, Yuhang Wang,
and Yichao Wu. 2025b. StepSearch: Igniting LLMs
search ability via step-wise proximal policy optimiza-
tion. InProceedings of the 2025 Conference on Em-
pirical Methods in Natural Language Processing,
pages 21816–21841.
12

Appendix Overview
The appendix includes the following sections:
•Section A: Prompts and Models Refer-
ence.Details the prompts used for the pol-
icy model’s trajectory generation, the PRM’s
training-data annotation, and the PRM’s VTS
inline scoring. This serves as a supplement to
Section 3.
•Section B: Data Statistics.Provides detailed
statistics of the PRM training corpus and the
DPO preference pairs, along with an ablation
on the outcome-margin filter. This serves as a
supplement to Sections 3 and 4.
•Section C: PRM-guided Value Tree Search.
Presents the mathematical formulations and
hyperparameters for the PRM-guided value
tree search, together with a branching-factor
ablation. This serves as an extended explana-
tion of Section 3.2.
•Section D: Implementation Details and Hy-
perparameters.Reports the full PRM weight
(α) sweep and training hyperparameters for
SFT, DPO, and KTO. This serves as a supple-
ment to Sections 4 and 4.4.
•Section E: Direct Validation of the PRM.
We complement the indirect PRM evaluation
in Section 4.5 with direct validation of label
quality using both human and frontier-model
audits.
•Section F: Statistical Significance of Main
Results.Provides bootstrap confidence in-
tervals and per-dataset paired t-tests against
prior baselines. This serves as a supplement
to Table 1.
•Section G: Generalization Across Model
Families, Scales, and Alignments.Provides
additional 3B analysis, including base-model
results, statistical significance, and training
details, supplementing Section 4.7.
•Section H: Intermediate-Flawed-But-
Correct Analysis.Quantifies trajectories
that contain flawed intermediate steps despite
reaching the correct final answer, directly
evaluating the failure mode targeted by
process-level supervision.•Section I: Detailed Recovery Analysis.Pro-
vides the full breakdowns of the recovery ex-
periment, including token-level F1, per-depth
results, and the complementary case where the
initial retrieval already succeeds. This serves
as a supplement to Section 4.8.
•Section J: Case Study.Presents two qualita-
tive examples illustrating how PRM step-level
supervision prevents entity confusion and er-
ror propagation. This serves as an extended
example of Section 4.
•Section K: PRM-Feedback-Guided Regen-
eration.Documents a failed augmentation
strategy in which PRM rationales were used
to regenerate training trajectories, along with
a diagnosis of the resulting distributional mis-
match. This serves as an extended discussion
of the limitations.
•Section L: Released Resources.Provides
links to the released implementation, models,
and training datasets.
A Prompts and Models Reference
We provide the prompts used for policy trajec-
tory generation, PRM training-data annotation, and
PRM-guided VTS evaluation.
13

Policy Model: Trajectory Generation
System Prompt.You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To
answer questions, you must first reason through the available information using <think> and</think> . If you identify
missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system
will provide you with relevant documents enclosed in <documents> and</documents> . You can search as many times
as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide
your final answer. Ensure your answer is concise, using nouns or short phrases whenever possible. Conclude with: “So
the answer is<answer>answer</answer>”.
User Prompt.
Question: {question}
PRM Training-Data Generation: QwQ-32B Annotation Prompt
This prompt is used by QwQ-32B for full-trajectory evaluation and generation of PRM training labels. The gold answer is
provided as a reference, while intermediate steps are evaluated according to the criteria below.
System Prompt.
You are a strict process supervisor for a multi-hop question answering agent that uses retrieval-augmented generation
(RAG). Your task is to evaluate each step of the agent’s trajectory and assign a binary label: GOOD or BAD.
The agent operates using four XML-tagged actions: <think> (Internal reasoning), <search> (Retrieval query),
<documents>(Retrieved passages), and<answer>(Final answer).
Labeling Principles
1. Default label is BAD. Assign GOOD only when a step makes a verifiable contribution toward answering.
2. No information gain means BAD (e.g., restating the question, repeating prior reasoning).
3. Do not use the model’s own world knowledge. Unverified claims not in<documents>are hallucination.
Evaluation Criteria
R1. Entity/Relation. BAD if <think> misidentifies an entity, reverses a relation, or drifts. GOOD if it maintains correct
grounding.
R2. Search. GOOD if the query is specific, returns relevant <documents> , or tries a new strategy after a failure. BAD if
it repeats a failed attempt, is vague, or returns irrelevant data.
R3. Reasoning. GOOD if it extracts useful information from <documents> or identifies a concrete knowledge gap. BAD
if it merely restates, repeats, or hallucinates.
R4. Answer. BAD if empty, uncertain, requires unsupported logical leaps, or contradicts <documents> . GOOD if
specific and logically derivable.
R5. Recovery. GOOD if the agent issues a new, meaningfully different <search> after a failure. BAD if it proceeds to
<answer>while ignoring the failure.
R6. Overconfidence.BAD if the agent produces<answer>without prior successful search.
User Prompt.
Question: {question} Ground Truth: {gold_answer}
## Trajectory
{full trajectory with all steps}
Expected Output.
Return a JSON array:
[{"step": 1, "label": "GOOD", "reasoning": "..."}, ...]
14

PRM-Guided VTS: Inline Step Evaluation Prompt
This prompt is used by the deployed 8B PRM for step-level evaluation during VTS expansion.
System Prompt.
You are a step-level PRM for evaluating reasoning quality in multi-hop question answering. The trajectory uses XML
tags: <think> ,<search> ,<answer> , and <documents> . Analyze each step’s logical soundness and evidence grounding.
First explain your reasoning inside[REASONING]tags, then output a label (1=good, 0=bad).
User Prompt.
Question: {question}
Previous Steps
## Step 1
<think>...</think> <search>...</search> <documents>...</documents>
···
Current Step to Evaluate
## Step N
<think>...</think> <search>...</search>
Expected Output Format
[REASONING]{PRM reasoning}[/REASONING]
Label: {1 or 0}
B Data Statistics
We present detailed statistics for the datasets used
in the two training stages. To avoid ambiguity,
we distinguish between the original seed-question
pool, the subset retained after filtering, and the final
number of step-level supervision instances.
PRM Training Data.The PRM is trained on
109,664 step-level decisions extracted from 31,728
reasoning trajectories sampled across 2,000 source
multi-hop QA questions. Table 6 summarizes
the action-type distribution and the QwQ-32B
GOOD/BAD verdicts. The majority of the steps
are search actions (71.2%), reflecting the highly
interactive nature of our multi-hop RAG environ-
ment. The Reason action (internal thought) ac-
counts for only 0.5% of the steps and has a high
rejection rate (73.7% BAD), indicating that the
PRM penalizes unsupported internal hallucinations
and forces the model to rely on external search.
We validate the reliability of these QwQ-generated
annotations through an independent frontier-LLM
audit reported in Section E.
DPO Preference Pairs.For policy optimization,
we began from the full pool of 5,000 seed multi-hop
QA questions described in Section 4. After running
VTS, we retained only preference pairs satisfyingStep Type Count (%) GOOD BAD
Search 78,055 (71.2%) 55,710 (71.4%) 22,345 (28.6%)
Answer 31,062 (28.3%) 14,436 (46.5%) 16,626 (53.5%)
Reason 547 (0.5%) 144 (26.3%) 403 (73.7%)
Total 109,664 (100%) 70,290 (64.1%) 39,374 (35.9%)
Table 6: Statistics of the PRM training corpus: 109,664
steps generated from 31,728 trajectories over 2,000
source questions (averaging 3.5 steps per trajectory).
Labels assigned by QwQ-32B under the R1–R6 rubric
(Section A).
an outcome-margin filter (chosen F1≥0.2 and
∆F 1≥0.2 ), yielding 2,866 unique questions and
15,877 contrastive pairs, or approximately 5.5pairs
per retained question.
We adopt this filter for training and data effi-
ciency rather than for performance. As shown in
Table 7, removing the filter yields 39,578 pairs
(2.5× more data) with comparable average perfor-
mance (34.65 vs. 34.51 EM), indicating that the
discarded low-margin pairs contribute little addi-
tional signal relative to their training cost.
As detailed in Table 8, the DPO pairs exhibit
rich diversity in action types, tree depths, and re-
ward margins. About three-quarters ( 75.0% ) of the
comparisons pit actions of the same type against
each other (Search vs. Search: 57.5% , Answer vs.
Answer: 17.5% ), allowing the policy to learn fine-
15

grained strategic differences rather than mere for-
mat adherence. The step-level distribution peaks
at depth 2 ( 45.7% ) and depth 3 ( 29.5% ), directly
aligning with the multi-hop reasoning requirements.
The reward difference between chosen and rejected
steps ensures a robust gradient: 98.4% of pairs
have a substantial margin ( ≥0.20 ), with over half
(52.0%) exhibiting a high margin (≥0.50).
C VTS Implementation Details
This section provides the detailed mathematical for-
mulations and algorithmic mechanics for the VTS
procedure introduced in Section 3.2. Specifically,
we detail how the step-level feedback from our
generative PRM is integrated into the tree search
to actively balance exploration and exploitation,
ultimately guiding the generation of high-quality,
process-supervised trajectories for PRO-STEP.
Selection.
UCB(s t) =¯Q(st) +cs
lnN(parent(s t))
N(s t)
where N(s t)is the visit count of st,¯Q(st) =
W(s t)/N(s t)is its mean backed-up value, W(s t)
is its cumulative backed-up value, and cis the ex-
ploration constant. Unvisited nodes ( N= 0 ) are
selected immediately.
Terminal reward.
Q(sT) = F1(s T, y∗)·γd(sT)
where γ∈(0,1) discourages unnecessarily long
trajectories andd(·)denotes node depth.
Back-propagation.After evaluating a terminal
node, we propagateQ(s T)to all ancestors:
N(s)←N(s) + 1, W(s)←W(s) +Q(s T)
When πψevaluates a non-terminal node st, we
apply a retroactive update to its ancestors without
changing their visit counts:
W(s)←W(s) + ˆr(s t)·γd(st)
VTS Parameters.During the process-supervised
data generation phase, we utilize VTS to explore
reasoning trajectories. Table 9 details the specific
hyperparameter values used for the tree search. To
balance the reward scale, the terminal F1 score
is discounted by 0.9per depth level, and the in-
termediate step score is computed as a weightedsum of the backed-up F1 reward and the PRM’s
feedback ( α= 0.3 ). Because the PRM provides
binary labels {0,1} whereas the F1 score is contin-
uous in [0,1] , this weight allows the process reward
to break ties among steps with similar outcomes
without overriding substantial F1 differences.
Branching-factor ablation.We additionally
evaluate the effect of the branching configuration
used during VTS trajectory generation. As shown
in Table 10, the main configuration improves av-
erage performance by 1.9 EM and 2.0 F1 over the
reduced-branching setting. The reduced setting
nevertheless remains competitive, indicating that
the effectiveness of PRO-STEPis not tied to a sin-
gle branching configuration.
D Implementation Details and
Hyperparameters
PRM Weight Sweep.Table 11 reports the full
sweep over αfor the combined value V(s) =
¯Q(s)+α·ˆr(s) . The results justify our use of PRM-
based step evaluation: every configuration that in-
corporates PRM signals ( α >0 , except α= 0.5 )
outperforms the no-PRM baseline ( α= 0 , 32.68
EM), with α= 0.3 achieving the best average
(34.51 EM, 44.09 F1). This confirms that PRM
signals provide supervision beyond what outcome
rewards alone can offer, and that process and out-
come rewards function as complementary signals.
Training Hyperparameters.Table 12 summa-
rizes the hyperparameter configurations for train-
ing the policy model across different optimization
strategies (SFT, DPO, KTO). All policy models
share the same base model (Qwen2.5-7B-Instruct)
and apply document masking during training to
prevent the model from memorizing the retrieved
context. We utilize LoRA for parameter-efficient
fine-tuning across all models, targeting all linear
layers.
E Direct Validation of the PRM
We complement the indirect PRM evaluation in
Section 4.5 with direct validation of label quality
using both human and frontier-model audits.
Human audit.Our research team manually eval-
uated a sample of 50 trajectories using the same
annotation criteria as the QwQ-32B labeler. The
resulting step-level annotations showed 84% agree-
ment with QwQ-32B, providing an independent
manual validation of the generated supervision.
16

Table 7: Effect of the outcome-margin filter on data scale and downstream performance. Removing the filter
increases the pair count by2.5×but yields comparable average performance.
Variant Pairs HotpotQA PopQA 2WikiMulti Bamboogle MuSiQue Average
PRO-STEP(with margin filter) 15,877 38.73 / 51.63 40.47 / 47.37 44.07 / 51.43 36.80 / 47.63 12.49 / 22.4134.51 / 44.09
w/o margin filter 39,578 37.62 / 50.58 39.20 / 46.40 44.47 / 52.62 39.20 / 47.70 12.78 / 23.16 34.65 / 44.09
Pair Type % Reward Diff Margin %
Search vs. Search 57.5<0.05(Low) 0.3
Answer vs. Answer 17.50.05–0.100.6
Answer vs. Search 16.50.10–0.200.6
Search vs. Answer 4.10.20–0.5046.4
Other 4.4≥0.50(High) 52.0
Step Level (Depth) % Action (Ch / Rej) %
Step 1 24.0 Search 61.9 / 74.1
Step 2 45.7 Answer 37.7 / 21.7
Step 3 29.5 Reason 0.4 / 4.2
Step 4 0.8 – –
Table 8: Detailed statistics of the 15,877 DPO prefer-
ence pairs under the outcome filter (chosen F1≥0.2
AND ∆F1≥0.2 ). Reward margin is computed on
V(s) = ¯Q(s) +α·ˆr(s) withα= 0.3 . Action per-
centages represent the proportion of Chosen / Rejected
actions respectively.
Parameter Value
Branching factor (K) 3
Max depth 7
Iterations per question 64
F1 discount (γ) 0.9 (Reward= F1·0.9d)
PRM weight (α) 0.3 (Combined= Reward + 0.3·PRM)
PRM scoring interval Every 4 iterations
Retrieved docs per search 3
Preference margin (δ) 0.01
Table 9: Hyperparameters for the VTS rollout phase.
Frontier-model audit.We further audit 500 strat-
ified samples drawn from QwQ-32B-labeled trajec-
tories (held-out from PRM training) by having an
independent frontier-class LLM (Claude Opus 4.7)
re-evaluate them under theidenticallabeling rubric
(R1–R6, Section A).
Findings. κ= 0.61 falls in thesubstantial
agreement range under the standard Landis–Koch
(Landis and Koch, 1977) interpretation ( κ∈
[0.61,0.80] = substantial). The disagreement direc-
tion is mild (frontier slightly stricter, ∼7% fewer
GOOD calls), with no systematic permissive bias
from QwQ. Per step type, agreement is almost per-
fect on answer steps ( κ= 0.82 ) and moderate on
search steps ( κ= 0.52 ); the lower agreement on in-
termediate reasoning steps ( κ= 0.12 ) reflects the
inherent subjectivity of judging whether a ⟨think⟩
block provides “new useful information” — butreasoning steps account for only 10% of the audit,
and the labels that drive PRM signal most directly
(answer correctness and search quality) are pre-
cisely where frontier agreement is strongest. This
confirms QwQ-32B as a trustworthy open-source
labeler.
F Statistical Significance of Main Results
We supplement Table 1 with bootstrap confi-
dence intervals and per-dataset paired t-tests, com-
puted on per-question results against PRO-STEP.
All evaluations are conducted with temperature
0 (greedy decoding), which removes sampling
stochasticity and yields deterministic per-question
outputs. We therefore report significance over the
per-question score distribution rather than across
multiple seeds.
Findings.PRO-STEPsignificantly outperforms
Search-R1 and ReasonRAG in macro-A VG EM
and F1 (95% CI). Against StepSearch, F1 im-
provements are significant; while the EM CI
marginally includes 0, PRO-STEPleads signif-
icantly on PopQA and 2WikiMultiHopQA. The
decisive 2WikiMultiHopQA lead ( p <10−75) in-
dicates the gap is not an averaging artifact.
G Generalization Across Model Families,
Scales, and Alignments
The main generalization results are reported in Sec-
tion 4.7. Here, we provide additional analysis of
the 3B setting that is not included in the main text,
including the underlying base-model performance,
statistical significance, and training details.
Additional analysis at the 3B scale.Table 17
reports the complete 3B results, including the un-
derlying Qwen2.5-3B Base model. PRO-STEP
achieves 31.04 / 40.00 average EM / F1 compared
with 26.22 / 33.51 for Search-R1 3B, while using
5k seed questions compared with Search-R1’s 170k
training questions.
Training at the 3B scale.Direct DPO on the 3B
base model achieves only 12.40 EM. We therefore
17

Table 10: Ablation of the VTS branching configuration. The main configuration improves average performance by
1.9 EM and 2.0 F1 over the reduced-branching setting.
Configuration PopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
Reduced branching (K= 2) 39.5 / 46.5 38.0 / 50.3 39.6 / 48.4 32.8 / 42.513.2/23.032.6 / 42.1
Main configuration (K= 3)40.5/47.4 38.7/51.6 44.1/51.4 36.8/47.612.5 / 22.434.5/44.1
Table 11: Performance comparison across different Process Reward Model (PRM) weight ( α) values. The table
reports EM / F1 scores across 5 benchmarks.⋆denotes the selected configuration.
αHotpotQA PopQA 2WikiMultiHopQA Bamboogle MuSiQue Average
0.0 (no PRM) 36.75 / 49.31 38.48 / 45.01 41.44 / 48.40 34.40 / 44.20 12.33 / 22.35 32.68 / 41.85
0.3⋆38.73/51.63 40.47/47.37 44.07/51.43 36.80/47.6312.49 / 22.41 34.51/44.09
0.5 34.95 / 47.53 36.62 / 44.33 41.28 / 48.21 33.60 / 41.91 10.47 / 20.41 31.38 / 40.48
1.0 37.97 / 50.60 39.94 / 46.10 42.68 / 50.38 32.80 / 41.3913.49/23.4633.38 / 42.39
Parameter SFT DPO KTO
Base ModelQwen2.5-7B-Instruct
Data Size15,877 (Chosen) 15,877 pairs 31,754 (15,877 +/15,877 -)
ObjectiveNext-token Sigmoid (β= 0.1) KTO (β= 0.1)
Learning Rate2e-5 2e-5 5e-6
Effective Batch Size16 16 16
Epochs1 1 1
Max Length8,192 8,192 8,192
Document MaskingYes Yes Yes
LoRA Configuration
Rank (r)64 64 64
Alpha (α)128 128 128
Dropout0.05 0.05 0.05
Target Modulesall-linear
Table 12: Training hyperparameters for the policy mod-
els. The effective batch size of 16 is achieved via 1×8
gradient accumulation ×2 GPUs across all three ob-
jectives. KTO uses both chosen and rejected from the
outcome filter pair set as desirable / undesirable samples
(hence 31,754 total).
Table 13: Agreement between QwQ-32B (our
annotator) and Claude Opus 4.7 on the 500-sample
audit.
Metric Value
Audited samples 500
Agreement 80.56%(402 / 499 with
valid labels)
Cohen’sκ0.6104(substantial)
QwQ-labeled GOOD 274
Frontier-labeled GOOD 255
Table 14: Per-step-type agreement on the 500-sample
audit.
Step type N Agreement Cohen’sκ
answer 200 91.00%0.82
search 249 77.11% 0.52
reason 50 56.00% 0.12
use SFT warmup to learn the interaction format
before preference optimization. The successful 3BTable 15: Macro-A VG bootstrap 95% confidence inter-
vals ( B= 104) for∆vs. each prior baseline. Signifi-
cance:⋆ ⋆ ⋆if the 95% CI excludes 0.
vs. Baseline∆EM (95% CI)∆F1 (95% CI)
Search-R1 +2.51 [+1.01, +4.06]⋆ ⋆ ⋆+3.37 [+1.94, +4.82]⋆ ⋆ ⋆
ReasonRAG +1.93 [+0.46, +3.36]⋆ ⋆ ⋆+3.14 [+1.56, +4.65]⋆ ⋆ ⋆
StepSearch +1.36 [-0.29, +3.00]NS+2.10 [+0.38, +3.77]⋆ ⋆ ⋆
Table 16: Per-dataset paired t-test on EM. n= test split
size.
Datasetnvs. Search-R1 vs. ReasonRAG vs. StepSearch
HotpotQA 7,405 +0.85,p=0.10NS+2.36,p <10−6⋆ ⋆ ⋆+0.01,p=0.98NS
PopQA 14,267 -0.18,p=0.56NS+2.69,p <10−15⋆ ⋆ ⋆+1.23,p <10−3⋆ ⋆ ⋆
2WikiMultiHopQA12,576+9.20,p <10−75⋆ ⋆ ⋆+4.27,p <10−18⋆ ⋆ ⋆+3.69,p <10−13⋆ ⋆ ⋆
Bamboogle 125 +3.20,p=0.40NS-1.60,p=0.66NS+3.20,p=0.43NS
MuSiQue 2,417 -0.50,p=0.46NS+1.90,p=0.004⋆⋆-1.32,p=0.054NS
configuration additionally uses trajectories gener-
ated by the larger policy model.
H Intermediate-Flawed-But-Correct
Analysis
We provide a direct quantitative analysis of the fail-
ure mode targeted by PRO-STEP: trajectories that
contain flawed intermediate steps while neverthe-
less producing the correct final answer.
Definition.We define anintermediate-flawed-
but-correct(IFBC) trajectory as a trajectory con-
taining at least one flawed intermediate step (all
steps except the final-answer step) while still pro-
ducing a correct final answer. We report
IFBC =P(flaw≥1|correct).(7)
Here, flaw denotes at least one flawed interme-
diate step, and correct denotes a correct final an-
swer. Lower IFBC therefore indicates that correct
18

Table 17: Results at the 3B scale (EM / F1, %).Bold
denotes the best result; “*” denotes p <0.05 against
Search-R1.
Dataset Base 3B Search-R1 PRO-Step
HotpotQA 5.31 / 17.36 31.82 / 41.3032.82 / 44.33*
PopQA 7.24 / 22.23 36.87 / 42.1439.88* / 46.10*
2WikiMulti 2.27 / 15.28 35.96 / 41.8644.10* / 50.77*
Bamboogle 4.00 / 10.64 18.40 / 27.9728.00* / 38.96*
MuSiQue 1.24 / 6.27 8.07 / 14.2710.38* / 19.85*
Average4.01 / 14.35 26.22 / 33.5131.04 / 40.00
answers are less frequently supported by flawed
intermediate trajectories. Steps are annotated using
QwQ-32B under the same R1–R6 rubric used for
PRM supervision, with the annotator blind to sys-
tem identity. The analysis uses the trajectories un-
derlying the main experiments for 500 paired ques-
tions each from HotpotQA, 2WikiMultiHopQA,
and MuSiQue.
Table 18: Intermediate-Flawed-But-Correct (IFBC) rate
among correct trajectories. Statistical tests compare
each method with PRO-STEP. Lower is better.
Method IFBC↓pvs. Ours # Correct
PRO-STEP26.8% –474
Outcome-only ablation 40.3%<10−4447
Search-R1 33.3% 0.033 439
StepSearch 33.0% 0.037 457
ReasonRAG21.6%0.071 (NS) 430
Results.The controlled comparison with the
outcome-only ablation isolates the effect of process-
level preference construction: both variants use
the same training trees and DPO hyperparameters,
while differing in how preference pairs are selected.
PRM-guided selection reduces IFBC from 40.3%
to 26.8%, a reduction of 13.5 percentage points
(p <10−4), while increasing the number of correct
trajectories from 447 to 474.
The same pattern appears in the training signal.
Outcome-only selection promotes 19.8% flawed-
but-successful trajectories to the chosen side of the
DPO pairs, whereas PRM-guided selection reduces
this rate to 10.7%.
PRO-STEPalso exhibits significantly lower
IFBC rates than Search-R1 and StepSearch. Rea-
sonRAG has a numerically lower IFBC rate, but
the difference is not statistically significant ( p=
0.071 ) and it produces fewer correct trajectories
(430 vs. 474). Thus, the IFBC analysis is inter-
preted jointly with task performance rather than asTable 19: Token-level F1 on recovered trajectories.
Dataset Search-R1 PRO-STEP∆
HotpotQA 66.669.0+2.5
PopQA 54.257.6+3.4
2WikiMulti 58.266.1+7.8
Bamboogle87.684.0−3.6
MuSiQue 48.951.1+2.1
Table 20: Final EM on 2WikiMulti by recovery depth
k.
Methodk= 1k= 2k= 3k≥4
n3,193 / 2,565 578 / 332 150 / 33 81 / 15
Search-R1 49.0 42.6 38.7 34.6
PRO-STEP53.1 50.9 45.5 60.0
∆+4.1 +8.3 +6.8 +25.4
a standalone measure of system quality.
I Detailed Recovery Analysis
We provide the full breakdowns underlying Sec-
tion 4.8. A retrieval succeeds when the gold pas-
sage appears among the top-3 retrieved documents,
and recovery depth kdenotes the first successful re-
trieval step, with the initial retrieval indexed as step
0. Throughout, nis reported as Search-R1 / PRO-
STEP, since the recovered subset is defined by each
model’s own search behavior.
Table 19 reports token-level F1 on the same re-
covered subsets as Table 5. PRO-STEPleads on
four of five datasets, with the largest margin on
2WikiMulti (+7.8). The exception is Bamboogle,
whose recovered subset contains only 25 questions.
Table 20 breaks 2WikiMulti down by recovery
depth. Gains are consistent at k= 1 through k= 3 .
Thek≥4 bucket favors PRO-STEPby a wide mar-
gin, but both subsets are small and the comparison
is not reliable.
Table 21 reports the complementary case in
which the initial retrieval already surfaces the gold
passage, so no recovery is required. PRO-STEP
performs competitively or better on four of five
datasets, indicating that its advantage on recovered
trajectories does not stem from an easier subset.
J Case Study
We present qualitative examples to illustrate the
limitation of outcome-based learning: even when
generating reasoning step-by-step, models trained
solely on final answers fail to penalize intermediate
19

Table 21: Final EM when the initial retrieval already
succeeds. Subset sizes ( n, ours / Search-R1) are 3,136
/ 3,434, 8,597 / 8,776, 4,302 / 4,732, 23 / 25, and
348 / 405, in row order.
Dataset Search-R1 PRO-STEP∆
HotpotQA 53.154.1+1.0
PopQA62.361.0−1.3
2WikiMulti 42.554.6+12.1
Bamboogle 34.864.0+29.2
MuSiQue 25.327.2+1.9
logical gaps. In contrast, PRO-STEP’s PRM eval-
uates the validity of every single step, effectively
preventing early mistakes from ruining the entire
trajectory.
Case Study 1: Resolving Entity Confusion.As
shown in Figure 6, the baseline confuses the tar-
get “Vernon Sewell” with “John Sewell.” Blind to
this semantic drift, it prematurely outputs an incor-
rect answer at Step 2. Conversely, PRO-STEP’s
PRM immediately detects this entity mismatch and
assigns a BAD signal. Forced to reject the spuri-
ous path, the model continues exploring alternative
queries until it retrieves the correct evidence at Step
8 and answers correctly at Step 9.
Case Study 2: Preventing Error Propagation.
Figure 7 demonstrates the danger of error propaga-
tion in multi-hop queries. The baseline (Reason-
RAG) incorrectly decomposes an intermediate hop
(e.g., targeting “NBC” instead of “CBS”). With-
out step-level correction, it stubbornly repeats the
same futile search four times before hallucinating.
PRO-STEP, however, receives step-level PRM ver-
ification. By recognizing the invalid intermediate
conclusion early, it pivots its strategy and success-
fully constructs the correct logical chain (NBC →
CBS→Date).
K PRM-Feedback-Guided Regeneration:
A Failed Augmentation Strategy
To test whether augmenting our preference set
with critic-feedback-guided regeneration would im-
prove downstream performance, we ran the fol-
lowing pipeline. For each of the 1,456 ques-
tions where the policy failed to produce a correct
answer (final F1= 0), we injected the PRM’s
critic_feedback from the failure point into the
user prompt and had the base Qwen2.5-7B-Instruct
regenerate from the last sound prefix. The regener-
ated trajectories were re-scored by the PRM, andwe applied the same outcome-margin filter (chosen
F1≥0.2 and∆F 1≥0.2 ) as in our main pipeline.
This yielded 20,567 preference pairs over 3,128
questions (compared to 15,877 pairs over 2,866
questions in the original outcome filter set). We
then trained DPO under identical hyperparameters
(β= 0.1 , LoRA r= 64/α= 128 ) on the aug-
mented set.
Training metrics improve while evaluation col-
lapses.Table 22 shows that the augmented train-
ing run yields substantially stronger DPO statis-
tics: the loss roughly halves (0.42 →0.22), the
preference margin grows by 60% (2.17 →3.48),
and training accuracy climbs from 81% to 91%.
However, downstream evaluation collapses on ev-
ery dataset, with average EM dropping from 34.51
to 23.83 and average F1 from 44.09 to 34.27 (Ta-
ble 23).
Table 22: Training metrics for the outcome filter main
set vs. the regen-augmented set under identical DPO
hyperparameters (β= 0.1).
Variant Loss Margin Acc
outcome filter main 0.42 2.17 81%
+ regen 0.22 3.48 91%
Diagnosis: action-format mismatch between
training and inference.The regenerated chosen
trajectories were produced by the base policy un-
der explicit critic-feedback prompting, which alters
the model’s output distribution: the regenerated
⟨think⟩ blocks become verbose meta-commentary
referencing the feedback (e.g., reasoning about why
the previous step was flawed), and the action selec-
tion becomes biased toward what the feedback sug-
gests rather than what the trajectory state requires.
At inference time, the deployed policy operates
without any such feedback channel, so the distribu-
tion it must produce diverges from the regenerated
distribution it was trained to match.
The outcome-margin filter, which enforces F1≥
0.2and∆F 1≥0.2 , can verify answer-level qual-
ity but is blind to this stylistic drift in interme-
diate tokens. As a result, the augmented train-
ing set rewards the policy for matching feedback-
conditioned outputs that never appear at test time,
and the strong DPO metrics reflect overfitting to
this artificial distribution rather than genuine im-
provement in multi-hop reasoning. We view this
as a concrete failure mode of naive PRM-rationale-
20

Table 23: Downstream evaluation: the regen-augmented variant degrades on every benchmark despite stronger
training metrics.
Variant PopQA HotpotQA 2WikiMulti Bamboogle MuSiQue Average
outcome filter main 40.47 / 47.37 38.73 / 51.63 44.07 / 51.43 36.80 / 47.63 12.49 / 22.4134.51 / 44.09
outcome filter + regen 28.39 / 37.55 26.02 / 39.08 33.25 / 42.63 24.80 / 35.30 6.70 / 16.80 23.83 / 34.27
∆−12.08/−9.82−12.71/−12.55−10.82/−8.80−12.00/−12.33−5.79/−5.61−10.68/−9.82
guided augmentation, and a meaningful constraint
that future work in this direction must address.
L Released Resources
Our implementation and complete resource index
are available in our GitHub repository.
Models.
•Policy Model:PRO-STEP-Policy-7B.
•Process Reward Model:PRO-STEP-PRM-
8B.
Datasets.
•DPO Preference Pairs:PRO-STEP-
Preference-Data.
•PRM Training Annotations:PRO-STEP-
PRM-Data.
21

Case 1: Entity Confusion (2W ikiMultiHopQA)
Q: Where was the place of death of the director of film Curse Of The Crimson Altar?
Gold Answer:  Durban, South Africa
Search-R1 (Outcome Reward) Ours (Process Reward)
Step 1  — Query: Vernon Sewell place of death
Retrieved: George Sewell, Cecil Sewell, John Sewell (cricketer)
✗ Vernon Sewell document NOT found
Step 2  — Reasoning:
"It is reasonable to assume that the place of death mentioned is the same person"
→ Confuses John Sewell  (cricketer , d.1897) with Vernon Sewell  (director , d.2001)
Answer: Pietermaritzburg ✗
⚠ Entity Confusion: uses wrong person's death place as answerStep 1  — Query: director of film Curse Of The Crimson Altar
Retrieved: Vernon Sewell  identified as director ✓
Step 2  — Query: Vernon Sewell place of death
"John Sewell (cricketer) mentions a place of death, which is not the director we are looking for"
→ Correctly distinguishes entities ✓
Steps 3–7:  Progressive query refinement (5 attempts)
→ "Vernon Sewell biography ...", "Vernon Sewell date of death..."
Step 8  — Query: Vernon Sewell ... film Curse of the Crimson Altar
Retrieved: "Vernon Sewell died on 21 June 2001 in Durban, South Africa "
Answer: Durban, South Africa ✓
✓ PRM rewards correct entity distinction and persistent multi-step search
Key Difference:  Both methods receive the same retrieval results for "V ernon Sewell place of death".
Outcome Reward model confuses entities and stops early (2 steps). Process Reward model correctly identifies the mismatch and persists (9 steps).Figure 6: Case Study 1: Entity confusion. The baseline prematurely answers based on a mismatched entity (John
Sewell), whereas PRO-STEPrejects the flawed step via PRM feedback and persistently searches for the correct
target.
Case Study 2: Error Propagation from W rong First Hop  (MuSiQue)
Q: When is Celebrity Big Brother coming to the network which, along with ABC and the original broadcaster of Here and Now , is the other major broadcaster based in NY?
Gold:  February 7, 2018   Hops:  Here and Now → NBC → {ABC, NBC, CBS } → Celebrity Big Brother on CBS → Feb 7, 2018
ReasonRAG (Outcome Reward, 4 Prompts) ProStep (Process Reward)
Decomposition:
Sub-Q1: Here and Now  → NBC (correct)
Sub-Q2: Other major broadcaster → NBC  ✗ (should be CBS )
Search 1:  "When is Celebrity Big Brother coming to NBC?"
Retrieved: "premiered on CBS  on January 21, 2019"
→ Evidence shows CBS, but searching for NBC → evidence ignored
Searches 2–4:  Repeats identical query "Celebrity Big Brother coming to NBC"
→ Same CBS results returned 3 more times → still ignored
Cannot find Celebrity Big Brother on NBC → Hallucinates answer
Answer: 2023 ✗
Error Propagation: wrong first hop (NBC) contaminates entire chainStep 1   Search: When did Here and Now start and which network originally broadcast it?
Here and Now (1992 TV series) on NBC  ✓ — Distinguishes radio (NPR) vs TV version
Step 2   Search: Which network co-broadcasts with ABC and NBC, based in NY?
ABC, NBC are two major NY  broadcasters → the other is CBS  ✓
Step 3   Search: When is Celebrity Big Brother coming to CBS?
Retrieved: "Celebrity Big Brother premiered on CBS on February 7, 2018 " ✓
Steps 4–5:  Verifies CBS is based in NY , confirms premiere date
Answer: February 7, 2018 ✓
PRM ensures each reasoning hop is completed correctly before proceeding
Insight:  ReasonRAG's wrong first-hop decomposition (NBC instead of CBS) propagates through the entire chain, leading to repeated failed searches and eventual hallucination. ProStep solves each hop sequentially — identifying NBC first, then deducing CBS, then searching for
the date — with process-level feedback ensuring no step proceeds on incorrect reasoning.
Figure 7: Case Study 2: Error propagation. The baseline repeats useless searches after an incorrect intermediate
hop. PRO-Step evaluates each hop, avoids the error made by the baseline, and completes the logical chain.
22