# Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training

**Authors**: Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang

**Published**: 2026-02-26 03:31:00

**PDF URL**: [https://arxiv.org/pdf/2602.22576v1](https://arxiv.org/pdf/2602.22576v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, yet traditional single-round retrieval struggles with complex multi-step reasoning. Agentic RAG addresses this by enabling LLMs to dynamically decide when and what to retrieve, but current RL-based training methods suffer from sparse outcome rewards that discard intermediate signals and low sample efficiency where failed samples contribute nothing. We propose Search-P1, a framework that introduces path-centric reward shaping for agentic RAG training, comprising two key components: (1) Path-Centric Reward, which evaluates the structural quality of reasoning trajectories through order-agnostic step coverage and soft scoring that extracts learning signals even from failed samples, and (2) Dual-Track Path Scoring with offline-generated reference planners that assesses paths from both self-consistency and reference-alignment perspectives. Experiments on multiple QA benchmarks demonstrate that Search-P1 achieves significant improvements over Search-R1 and other strong baselines, with an average accuracy gain of 7.7 points.

## Full Text


<!-- PDF content starts -->

Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic
RAG Training
Tianle Xia∗, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang
Liqun Liu†,Peng Shu,Huan Yu,Jie Jiang
Tencent
{tianlexia,flemingxu,lingxianghu,emanuelsun,wenweiwwli,faelynshang}@tencent.com
{liqunliu,archershu,huanyu,zeus}@tencent.com
∗Equal contribution.†Corresponding author.
Abstract
Retrieval-Augmented Generation (RAG) en-
hances large language models (LLMs) by in-
corporating external knowledge, yet traditional
single-round retrieval struggles with complex
multi-step reasoning. Agentic RAG addresses
this by enabling LLMs to dynamically decide
when and what to retrieve, but current RL-
based training methods suffer from sparse out-
come rewards that discard intermediate signals
and low sample efficiency where failed sam-
ples contribute nothing. We propose SEARCH-
P1, a framework that introducespath-centric
reward shapingfor agentic RAG training,
comprising two key components: (1)Path-
Centric Reward, which evaluates the struc-
tural quality of reasoning trajectories through
order-agnostic step coverage and soft scor-
ing that extracts learning signals even from
failed samples, and (2)Dual-Track Path Scor-
ingwith offline-generated reference planners
that assesses paths from both self-consistency
and reference-alignment perspectives. Exper-
iments on multiple QA benchmarks demon-
strate that SEARCH-P1 achieves significant im-
provements over Search-R1 and other strong
baselines, with an average accuracy gain of 7.7
points.
1 Introduction
Large Language Models (LLMs) have demon-
strated strong reasoning capabilities (Zhong et al.,
2023; Xia et al., 2025; Hu et al., 2025), but
their static knowledge often leads to hallucina-
tions on knowledge-intensive queries. Retrieval-
Augmented Generation (RAG) (Lewis et al., 2020)
addresses this by incorporating external knowledge,
yet single-round retrieval is insufficient for complex
multi-step reasoning—a common need in industrial
applications such as advertising guidance, where
answering a question often requires synthesizing
information across multiple knowledge domains.
NQTriviaQAPopQA
HotpotQA
2Wiki
Musique
BamboogleAD-QA608050
50
40
30
5090(a) Qwen2.5-7B
NQTriviaQAPopQA
HotpotQA
2Wiki
Musique
BamboogleAD-QA608050
40
40
20
4080(b) Qwen2.5-3BPerformance Comparison on QA Benchmarks
Search-P1 (Ours) Search-R1 RAG Search-o1Figure 1: Performance comparison of SEARCH-P1
against baselines on QA benchmarks. Our method
achieves the highest average accuracy across all datasets
on both (a) Qwen2.5-7B and (b) Qwen2.5-3B models.
Agentic RAG extends traditional RAG by en-
abling LLMs to dynamically invoke search and
iteratively refine answers. Recent methods like
Search-R1 apply RL with outcome-based rewards,
but this approach has three limitations: (1)sparse
rewardsthat ignore intermediate reasoning quality,
(2)low sample efficiencywhere partially correct
trajectories receive zero reward, and (3)slow con-
vergencedue to weak training signals when most
samples share similar binary rewards.
We propose SEARCH-P1, a framework introduc-
ingpath-centric reward shapingfor agentic RAG
training that addresses all three limitations. Instead
of evaluating only final answers, our reward design
comprises: (1)dual-track path scoringthat pro-
vides dense intermediate signals by evaluating rea-
soning trajectories from both self-consistency and
reference-alignment perspectives, directly alleviat-
ing reward sparsity; and (2)soft outcome scoring
that assigns partial credit to incorrect trajectories,
converting zero-reward samples into useful training
signals to improve sample efficiency. Together, the
denser reward landscape accelerates convergence
by providing more informative gradients through-
out training. Experiments on public QA bench-
1arXiv:2602.22576v1  [cs.CL]  26 Feb 2026

marks and an internal advertising dataset (AD-QA)
show SEARCH-P1 outperforms existing methods
with an average accuracy gain of 7.7 points, while
also transferring effectively to enterprise knowl-
edge base systems. Our contributions:
•We propose dual-track path scoring that eval-
uates trajectories from self-consistency and
reference-alignment perspectives with order-
agnostic matching.
•We design a path-centric reward shaping
framework that extracts learning signals even
from failed trajectories via path-level reward.
•Extensive experiments on public benchmarks
and an industrial dataset demonstrate consis-
tent improvements across models and settings.
2 Related Work
Prompt-Based Agentic RAG.Initial efforts
leverage prompts to guide LLMs through multi-
step retrieval (Singh et al., 2025; Li et al., 2025a).
These approaches interleave reasoning with re-
trieval actions (Yao et al., 2023; Trivedi et al., 2023)
or enhance reasoning through sophisticated re-
trieval strategies (Li et al., 2025b; Wang et al., 2025;
Guan et al., 2025). However, prompt-based meth-
ods depend heavily on the base model’s instruction-
following ability.
RL-Based Agentic RAG.Recent work applies
reinforcement learning to train adaptive search
agents (Zhang et al., 2025a; Jin et al., 2025).
Follow-up methods incorporate auxiliary signals to
stabilize training (Song et al., 2025a; Chen et al.,
2025; Huang et al., 2025) or improve search effi-
ciency (Sha et al., 2025; Song et al., 2025b; Wu
et al., 2025b). Some work explores process rewards
for RAG (Sun et al., 2025; Wu et al., 2025a; Zhang
et al., 2025b), but still relies primarily on binary
outcome feedback. Our work proposes path-centric
reward shaping offering denser training signals.
3 Methodology
We first formalize the problem setting (§3.1), then
describe the path-centric reward framework includ-
ing dual-track scoring and soft outcome scoring
(§3.2). Figure 2 provides an overview.
3.1 Problem Formulation
We consider an agentic RAG system where a lan-
guage model πθgenerates a reasoning trajectoryTin response to a question q. In standard agentic
RAG frameworks, the trajectory consists of inter-
leaved reasoning and action steps:
T= (r 1, a1, o1, . . . , r n, an, on, rfinal,ˆa)(1)
where ridenotes reasoning, aidenotes a search
action, oiis the observation (search results), and ˆa
is the final answer.
We make the implicit planning in r1explicit by
restructuring the trajectory as:
T= (p, r 1, a1, o1, . . . , r n, an, on, rfinal,ˆa)(2)
where pis an explicit planner that outlines the rea-
soning strategy. This serves two purposes: (1) pro-
viding a self-declared plan against which execution
can be evaluated, and (2) making the intended rea-
soning structure observable for path-centric evalua-
tion.
Standard GRPO assigns binary rewards based
on answer correctness:
Routcome =⊮[match(ˆa, a∗)](3)
where a∗is the ground-truth answer. This formula-
tion ignores the quality of the reasoning path and
suffers from the limitations discussed in §1.
3.2 Path-Centric Reward
We propose a path-centric reward that evaluates
trajectory quality rather than solely relying on final
answer correctness, addressing the three limitations
of outcome-based methods. The complete reward
function is:
Rtotal=λp·Rpath+λa·Routcome +λf·Rformat (4)
where Rpathis the path-centric reward computed
via dual-track evaluation, Routcome is the soft out-
come score that extracts signals even from incor-
rect answers, Rformat encourages well-structured
outputs, andλ p,λa,λfare balancing coefficients.
3.2.1 Reference Planner Generation
We generate reference planners offline through re-
jection sampling and LLM voting. For each train-
ing sample (q, a∗), we generate Kcandidate tra-
jectories using a high-capability LLM, filter for
correct answers, and apply LLM voting to distill
an optimized reference plannerP ref:
Pref=V ote({T i}K
i=1|correct(T i))(5)
The voting identifies the minimal set of essential
steps across successful trajectories, yielding a ref-
erence reasoning pathR ref={s 1, s2, . . . , s m}.
2

Input
Question :Who directed the 1997 
film starring the Academy Award 
winner for The Revenant?
1
Ref_Plan Generation
Question  + Answer
Rejection  Sampling
LLM V oting
Ref_Plan
Generate Ktrajectories Filter correct ones
Extract  consensus  steps32
Planner
 Search
 Think
 Search
 Think
 Answer
𝑹𝒑𝒍𝒂𝒏𝒏𝒆𝒓 𝑹𝒔𝒆𝒂𝒓𝒄𝒉𝒆𝒓 𝑹𝒐𝒖𝒕𝒄𝒐𝒎𝒆
Planner
 Search
 Think
 Search
 Think
 Answer
𝑹𝒑𝒍𝒂𝒏𝒏𝒆𝒓 𝑹𝒔𝒆𝒂𝒓𝒄𝒉𝒆𝒓 𝑹𝒐𝒖𝒕𝒄𝒐𝒎𝒆
Planner
 Search
 Think
 Search
 Think
 Answer
𝑹𝒑𝒍𝒂𝒏𝒏𝒆𝒓 𝑹𝒔𝒆𝒂𝒓𝒄𝒉𝒆𝒓 𝑹𝒐𝒖𝒕𝒄𝒐𝒎𝒆
Trajectory  Generation
Track A: Self -Consistency
Self-Declared 
PlanPlan Execution
Check
𝑺𝒔𝒆𝒍𝒇=𝒓𝒑𝒍𝒂𝒏𝒏𝒆𝒓×𝒏𝒆𝒙𝒆𝒄
𝒏𝒑𝒍𝒂𝒏×𝒏𝒆𝒙𝒆𝒄
𝒏𝒂𝒄𝒕𝒊𝒐𝒏𝒔Track B : Ref -Alignment
Ref-defined 
PlanPlan Execution
Check
𝑺𝒓𝒆𝒇=𝒏𝒆𝒙𝒆𝒄
𝒏𝒑𝒍𝒂𝒏×𝒏𝒆𝒙𝒆𝒄
𝒏𝒂𝒄𝒕𝒊𝒐𝒏𝒔
Format Reward𝑹𝒇𝒐𝒓𝒎𝒂𝒕
Path -Centric Process Reward𝑹𝒑𝒓𝒐𝒄𝒆𝒔𝒔
Outcome Score𝑹𝒐𝒖𝒕𝒄𝒐𝒎𝒆
𝑹𝒕𝒐𝒕𝒂𝒍=𝝀𝒑⋅𝑹𝒑𝒓𝒐𝒄𝒆𝒔𝒔+𝝀𝒐⋅𝑹𝒐𝒖𝒕𝒄𝒐𝒎𝒆+𝑹𝒇𝒐𝒓𝒎𝒂𝒕Dual -Track Path Scoring 4
Policy Model
 𝜋𝜃
maxGRPO Policy Update
Figure 2: Overview of SEARCH-P1 framework. Our approach introduces path-centric reward shaping for agentic
RAG training, comprising: (1) Dual-Track Path Scoring that evaluates trajectories from both self-consistency and
reference-alignment perspectives, and (2) Soft Outcome Scoring that extracts training signals even from incorrect
answers.
3.2.2 Dual-Track Path Scoring
We evaluate trajectory quality from two comple-
mentary perspectives. Track A (Self-Consistency)
assesses whether the model effectively executes its
own stated plan:
Sself=r planner×nself
exec
nplan×nself
exec
nactions(6)
where rplanner rates the plan quality, nself
execcounts
executed steps, nplanis the total planned steps, and
nactions is the total actions in the trajectory. Track
B (Reference-Alignment) measures coverage of
essential steps from the reference planner using
order-agnostic matching:
Sref=ncovered
|Rref|×ncovered
nactions(7)
where ncovered counts accomplished reference steps
regardless of execution order. Both tracks incor-
porate an efficiency rationeffective
nactionsto prevent re-
ward hacking through excessive redundant steps
and encourage concise reasoning trajectories. Theconcrete criteria for determining effective steps
and covered steps—including the LLM-based se-
mantic matching procedure—are detailed in Ap-
pendix D.3. The final path-centric reward Rpath=
max(S self, Sref)takes the maximum rather than a
weighted combination, so that when the reference
plan is suboptimal or the model discovers a better
strategy, the self-consistency track can dominate
without being diluted by a low reference score (and
vice versa).
3.2.3 Soft Outcome Scoring
To improve sample efficiency, we extract learning
signals from trajectories with incorrect final an-
swers through soft scoring:
Routcome =(
1.0if correct
α·r acc+ (1−α)·r reason otherwise
(8)
where α= 0.8 ,raccindicates partial answer cor-
rectness and rreason evaluates reasoning quality in-
dependent of the final answer. This converts previ-
ously zero-reward failed samples into useful train-
3

ing signals based on their path quality.
4 Experiments
4.1 Experimental Setup
Datasets.Following prior work, we evaluate
on seven public QA benchmarks spanning two
categories: (1)General QA: NQ (Kwiatkowski
et al., 2019), TriviaQA (Joshi et al., 2017), and
PopQA (Mallen et al., 2023); (2)Multi-Hop
QA: HotpotQA (Yang et al., 2018), 2WikiMul-
tiHopQA (Ho et al., 2020), Musique (Trivedi
et al., 2022), and Bamboogle (Press et al., 2023).
Additionally, we evaluate onAD-QA, a fully
anonymized proprietary advertising QA dataset
containing 1,000 multi-hop test instances from an
internal business to assess real-world applicability
(details in Appendix A). Following Search-R1, we
merge the training sets of NQ and HotpotQA to
form a unified training dataset. Evaluation is con-
ducted on all datasets to assess both in-domain (NQ,
HotpotQA) and out-of-domain (TriviaQA, PopQA,
2WikiMultiHopQA, Musique, Bamboogle, AD-
QA) generalization.
Models.We conduct experiments with Qwen2.5-
7B-Instruct and Qwen2.5-3B-Instruct (Qwen et al.,
2025), denoted as 7B and 3B for brevity. For re-
trieval, we use the 2018 Wikipedia dump as the
knowledge source and E5 as the retriever, with top-
3 passages returned per search step.
Evaluation Metric.We use Accuracy (ACC)
as the primary evaluation metric, which checks
whether the ground-truth answer is contained in
the model’s generated response.
Baselines.We compare against the following
methods: (1)Direct Inference: Generation with-
out retrieval, including direct prompting and Chain-
of-Thought (CoT); (2)Standard RAG: Single-
round retrieval before generation; (3)Prompt-
Based Agentic RAG: IRCoT and Search-o1 that
use prompting for multi-step retrieval; (4)RL-
Based Agentic RAG: Search-R1 and HiPRAG that
use reinforcement learning for training. All RL-
based methods share identical training and retrieval
configurations (detailed in Appendix B); the only
difference is the reward function.
4.2 Main Results
As shown in Table 1, SEARCH-P1 achieves the
highest average accuracy across both model sizes,
outperforming all baselines by a clear margin (+7.7
0 20 40 60 80 100
Training Steps01020304050Accuracy (%)(a) Accuracy During Training
w/o Format
Strict Format
Soft Format
0 20 40 60 80 100
Training Steps0.00.20.40.60.8Reward(b) Reward During Training
w/o Format
Strict Format
Soft FormatEffect of Format Reward Design on Training DynamicsFigure 3: Training dynamics comparison of different
format reward strategies. Soft Format (our buffered
design) achieves faster ACC improvement and higher
stable rewards compared to Strict Format (zero reward
for invalid format) and Without Format baseline.
NQ TriviaQA PopQA HotpotQA 2Wiki Musique Bamboogle AD-QA
Dataset0255075100Accuracy (%)Effect of Soft Outcome Scoring Across Datasets
w/o Soft Outcome
w/ Soft Outcome
Figure 4: Effect of soft outcome scoring across datasets.
Gray bars show accuracy without soft scoring (binary
outcome), blue bars show accuracy with soft scoring.
Per-dataset results are in Appendix F.6.
Avg. ACC over Search-R1 on 7B). The gains are es-
pecially pronounced on the internal AD-QA bench-
mark (+20.6 over Search-R1 on 7B), a real-world
advertising QA dataset with complex multi-hop
queries, confirming the practical value of path-
centric rewards in industrial settings. Notably, the
improvements are consistent across model scales,
with the 3B model achieving +7.9 Avg. ACC over
Search-R1, demonstrating that path-centric rewards
are effective even for smaller models.
4.3 Ablation Study
We conduct ablation studies to validate the contri-
bution of each reward component in SEARCH-P1:
format reward, path-centric reward, and outcome
reward.
4.3.1 Format Reward
As shown in Figure 3, we compare three strate-
gies: (1) Soft Format (our buffered design), (2)
Strict Format (zero reward for violations), and (3)
Without Format. Our soft format achieves signifi-
cantly faster convergence by providing continuous
gradient feedback, while the strict approach yields
near-zero rewards in early training steps due to
frequent formatting errors.
4

MethodGeneral QA Multi-Hop QAAvg.Internal
NQ†TriviaQA PopQA HotpotQA†2Wiki Musique Bamboogle AD-QA
Qwen2.5-7B
Direct 13.4 40.8 14.0 18.3 25.0 3.1 12.0 18.1 10.3
CoT 4.8 18.5 5.4 9.2 11.1 2.2 23.2 10.6 8.7
RAG 34.9 58.5 39.2 29.9 23.5 5.8 20.8 30.4 60.4
IRCoT 22.4 47.8 30.1 13.3 14.9 7.2 22.4 23.9 52.3
Search-o1 15.1 44.3 13.1 18.7 17.6 5.8 29.6 20.6 48.5
Search-R1 42.9 62.3 42.7 38.6 34.6 16.2 40.0 39.6 65.6
HiPRAG 46.5 65.8 45.8 42.0 46.1 14.0 40.0 42.9 75.6
SEARCH-P1 56.6 78.6 47.5 42.939.821.8 44.0 47.3 86.2
Qwen2.5-3B
Direct 10.6 28.8 10.8 14.9 24.4 2.0 2.4 13.4 7.8
CoT 2.3 3.2 0.5 2.1 2.1 0.2 0.0 1.5 5.2
RAG 34.8 54.4 38.7 25.5 22.6 4.7 8.0 27.0 54.7
IRCoT 11.1 31.2 20.0 16.4 17.1 6.7 24.0 18.1 45.8
Search-o1 23.8 47.2 26.2 22.1 21.8 5.4 32.0 25.5 42.1
Search-R1 39.7 56.5 39.1 33.1 31.0 12.4 23.2 33.6 58.3
HiPRAG 43.0 59.8 42.0 36.0 40.5 10.8 24.0 36.6 70.2
SEARCH-P1 53.0 74.5 47.9 36.236.613.3 28.8 41.5 79.5
Table 1: Main results (ACC %) on seven public QA benchmarks and one internal dataset. Best results are inbold,
second best are underlined .†denotes in-domain datasets used for training; others are out-of-domain. AD-QA is a
proprietary advertising QA dataset. HiPRAG results are from our reproduction using the same retrieval setup.
Method Avg. ACC
SEARCH-P1 (Full) 47.3
w/o Reference-Alignment 42.0
w/o Self-Consistency 44.2
Search-R1 (Baseline) 39.6
Table 2: Ablation study on path-centric reward com-
ponents (Qwen2.5-7B). Per-dataset results are in Ap-
pendix F.4.
4.3.2 Path-Centric Reward
As shown in Table 2, removing reference-
alignment causes a 5.3% accuracy drop, confirm-
ing that reference planners provide valuable path-
centric guidance. Removing self-consistency re-
sults in a 3.1% decrease. The full dual-track model
achieves the best performance, validating that both
external guidance and internal consistency are com-
plementary signals.
4.3.3 Outcome Reward
As shown in Figure 4, soft outcome scoring pro-
vides modest gains for single-hop tasks (+1.2%),
larger improvements for multi-hop QA (+3.5%),
and the highest gain for AD-QA (+8.8%), con-
firming that complex scenarios benefit most from
partial credit signals.
0.2 0.3 0.4
Process Reward Weight (p)
424446485052Acc (%)(a) Process Reward Weight
Acc (%)
Process R
Outcome R
0.6 0.8 1.0
Accuracy Weight (a)
424446485052Acc (%)(b) Outcome Accuracy Weight
Acc (%)
Outcome R
0.40.60.81.0
Reward
0.300.350.400.450.500.550.60
RewardFigure 5: Hyperparameter sensitivity analysis. All re-
wards are averaged over steps 195–205. (a) Effect of
path reward weight λp. (b) Effect of accuracy weight
λa. Per-dataset results are in Appendix F.7.
5 Analysis
5.1 Hyperparameter Sensitivity
We investigate the impact of two critical hyper-
parameters in our reward formulation: the path
reward weightλ pand the accuracy weightλ a.
As shown in Figure 5, both λpandλaexhibit
clear sweet spots. Too little path weight provides
insufficient supervision, while too much induces
reward overfitting where path metrics improve but
accuracy drops. Similarly, over-weighting accuracy
neglects reasoning quality and leads to reward hack-
ing. The optimal configuration ( λp=0.3 ,λa=0.6 )
balances accuracy as the primary objective with
reasoning quality as a regularizer.
5

0 50 100 150 200
Training Step1020304050Accuracy (%)
(a) Training Efficiency
Ours Acc.
Base Acc.Ours Turns
Base Turns
Single-hop Multi-hop Adversarial
Dataset Type0.00.51.01.52.02.53.03.54.0Avg. Turns(b) Inference Efficiency
Ours (Succ.)
Ours (Fail)Base (Succ.)
Base (Fail)
1.52.02.53.03.54.0
Avg. Turns
Figure 6: Efficiency analysis. (a) Training efficiency:
accuracy and interaction turns comparison between
SEARCH-P1 and Search-R1 during training. (b) Infer-
ence efficiency: turns by outcome across dataset types.
Model RL Single Multi AD
Qwen2.5-3B GRPO 58.5 28.7 79.5
Qwen2.5-3B PPO 57.2 27.5 77.8
Llama-3.2-3B GRPO 56.8 27.1 76.3
Llama-3.2-3B PPO 55.6 26.2 74.6
Table 3: ACC (%) across base models and RL algo-
rithms. All models use Instruct versions. Per-dataset
results are in Appendix F.5.
5.2 Efficiency Analysis
Training EfficiencyFigure 6(a) compares train-
ing dynamics. SEARCH-P1 converges signifi-
cantly faster, reaching Search-R1’s final accuracy
(∼40%) within 60 steps versus over 150. Mean-
while, SEARCH-P1’s interaction turns steadily de-
crease, indicating path-centric rewards guide to-
ward higher accuracy and more concise reasoning,
while Search-R1’s turns remain flat or increase.
Inference EfficiencyFigure 6(b) compares turn
distributions across dataset types. Two key find-
ings emerge: (1) Both methods require more turns
for complex adversarial queries. (2) SEARCH-P1
maintains consistent turn counts between success-
ful and unsuccessful cases, while Search-R1 ex-
hibits larger gaps for multi-hop (+60%) and adver-
sarial (+47%) tasks.
5.3 Model and RL Algorithm Analysis
Table 3 examines the impact of base models
and RL algorithms. Qwen2.5-3B-Instruct (Qwen
et al., 2025) slightly outperforms Llama-3.2-3B-
Instruct (Grattafiori et al., 2024) across all task
types, likely due to stronger instruction-following
and reasoning capabilities in the base model.
GRPO (Shao et al., 2024) achieves marginally
higher accuracy than PPO (Schulman et al., 2017);
however, PPO exhibits more stable training dynam-
ics with lower variance across runs. Importantly,
path-centric rewards yield consistent gains acrossEvaluator ACCHuman Agree. (%)
Plan Step Outc.
HY 2.0-Inst. 47.3 91.2 94.5 88.7
Qwen3-32B 46.5 89.0 92.5 85.0
Qwen3-8B 44.1 83.5 88.0 78.5
Table 4: Effect of LLM evaluator choice on SEARCH-P1
Avg. ACC and human agreement. Per-dataset results
are in Appendix F.8.
all model–algorithm combinations, suggesting that
our approach is orthogonal to the choice of base
model and RL algorithm.
5.4 LLM Evaluator Analysis
Our dual-track scoring and soft outcome scor-
ing rely on an external LLM evaluator during
training (at inference time, no evaluator calls are
needed). To examine sensitivity, we replaced the
default evaluator (HY 2.0-Instruct) with Qwen3-
32B and Qwen3-8B, and sampled 200 trajectories
to measure human agreement. As shown in Ta-
ble 4, Qwen3-32B achieves comparable accuracy
(−0.8) and human agreement, while Qwen3-8B
degrades by 3.2 points with lower outcome scoring
agreement (78.5%). Nevertheless, step coverage—
the core component of our path-centric reward—
remains robust even with the 8B evaluator (88.0%
agreement), confirming that SEARCH-P1 is not
tightly coupled to a specific evaluator.
5.5 Case Study
To qualitatively illustrate SEARCH-P1’s advan-
tages, we present case studies comparing reasoning
trajectories with baseline methods. Appendix E
provides a representative example from multi-hop
QA, demonstrating how path-centric rewards lead
to more structured decomposition, precise query
formulation, and effective information synthesis.
6 Conclusion
We presented SEARCH-P1, a framework that in-
troduces path-centric reward shaping for agentic
RAG training. By evaluating the structural qual-
ity of entire reasoning paths rather than isolated
elements, our approach provides fine-grained su-
pervision while respecting the inherent diversity
of multi-step reasoning. Extensive experiments on
public QA benchmarks and an internal advertsing
dataset demonstrate significant improvements in
accuracy and efficiency, validating path-centric re-
wards in both academic and industrial settings.
6

Ethics Statement
Our work focuses on improving the training of
AI systems for information retrieval and reason-
ing. We use publicly available datasets for training
and evaluation. The internal AD-QA dataset is
fully anonymized with all personally identifiable
information removed prior to use. The improved
efficiency of agentic RAG systems could reduce
computational resources required for deployment,
contributing to more sustainable AI.
References
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou,
Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen
Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and
Weipeng Chen. 2025. Research: Learning to rea-
son with search for llms via reinforcement learning.
Preprint, arXiv:2503.19470.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.Preprint, arXiv:2407.21783.
Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2025. Deeprag: Thinking to retrieval
step by step for large language models.Preprint,
arXiv:2502.01142.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Lingxiang Hu, Chenyang Hei, Fuliang Li, Chengxi Gao,
Jiaxing Shen, and Xingwei Wang. 2025. Smarttc: A
real-time ml-based traffic classification with smartnic.
In2025 IEEE/ACM 33rd International Symposium
on Quality of Service (IWQoS), pages 1–10. IEEE.
Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng
Niu, Hao Peng, Julia Hockenmaier, and Tong Zhang.
2025. Rag-rl: Advancing retrieval-augmented gen-
eration via rl and curriculum learning.Preprint,
arXiv:2503.12759.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
Preprint, arXiv:2503.09516.Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Jian Li, Xiaoxi Li, Yan Zheng, Yizhang Jin, Shuo Wang,
Jiafu Wu, Yabiao Wang, Chengjie Wang, and Xiao-
tong Yuan. 2025a. A survey on ai search with large
language models.Preprints.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025b. Search-o1: Agentic search-enhanced
large reasoning models.Preprint, arXiv:2501.05366.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Prox-
imal policy optimization algorithms.Preprint,
arXiv:1707.06347.
7

Zeyang Sha, Shiwen Cui, and Weiqiang Wang. 2025.
Sem: Reinforcement learning for search-efficient
large language models.Preprint, arXiv:2505.07903.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag.Preprint,
arXiv:2501.09136.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025a. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Preprint, arXiv:2503.05592.
Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng
Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min,
Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025b.
R1-searcher++: Incentivizing the dynamic knowl-
edge acquisition of llms via reinforcement learning.
Preprint, arXiv:2505.17005.
Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue
Zang, Kai Zheng, Jun Xu, Xiao Zhang, Yang Song,
and Han Li. 2025. Rearter: Retrieval-augmented
reasoning with trustworthy process rewarding. In
Proceedings of the 48th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’25, page 1251–1261, New
York, NY , USA. Association for Computing Machin-
ery.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Liang Wang, Haonan Chen, Nan Yang, Xiaolong
Huang, Zhicheng Dou, and Furu Wei. 2025.
Chain-of-retrieval augmented generation.Preprint,
arXiv:2501.14342.
Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu
He, Xinya Du, and Zhiyu Chen. 2025a. Hiprag:
hierarchical process rewards for efficient agentic
retrieval augmented generation.arXiv preprint
arXiv:2510.07794.Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, and
Zhiyu Zoey Chen. 2025b. Search wisely: Mitigating
sub-optimal agentic searches by reducing uncertainty.
Preprint, arXiv:2505.17281.
Tianle Xia, Liang Ding, Guojia Wan, Yibing Zhan,
Bo Du, and Dacheng Tao. 2025. Improving complex
reasoning over knowledge graph with logic-aware
curriculum tuning. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 39, pages
12881–12889.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin,
Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi
Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang
Chen, Chen Zhang, Yutao Fan, Zihu Wang, Song-
tao Huang, Yue Liao, Hongru Wang, Mengyue Yang,
and 6 others. 2025a. The landscape of agentic re-
inforcement learning for llms: A survey.Preprint,
arXiv:2509.02547.
Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao
Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang,
Derong Xu, Zhaocheng Du, Huifeng Guo, and 1 oth-
ers. 2025b. Process vs. outcome reward: Which is
better for agentic rag reinforcement learning.arXiv
preprint arXiv:2505.14069.
Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and
Dacheng Tao. 2023. Can chatgpt understand too?
a comparative study on chatgpt and fine-tuned bert.
arXiv preprint arXiv:2302.10198.
8

A AD-QA Dataset
AD-QA is a fully anonymized multi-hop QA bench-
mark from a real-world advertising domain, con-
taining 1,000 test instances requiring multi-step
reasoning across domains such as campaign config-
uration, bidding strategies, audience targeting, and
conversion tracking. All instances are derived from
authentic user queries with all personally identifi-
able information removed.
Each question requires synthesizing information
from at least two distinct knowledge domains, mak-
ing it a challenging benchmark for multi-hop rea-
soning in enterprise settings. Ground-truth answers
are curated by domain experts and verified through
cross-validation.
B Implementation Details
B.1 Training Configuration
For GRPO training, we set the policy learning rate
to1×10−6with a warm-up ratio of 0.1. Training
is conducted on 8 ×H20 GPUs using a total batch
size of 512, with a mini-batch size of 256. The
micro-batch size per GPU is set to 8 for 7B models
and 16 for 3B models.
The maximum prompt length and response
length are both set to 4,096 tokens, with a maxi-
mum model context length of 8,192 tokens. We en-
able gradient checkpointing for memory efficiency
and use Fully Sharded Data Parallel (FSDP) with
reference model parameter offloading.
For efficient rollout generation, we use SGLang
with tensor parallel size of 1 and GPU memory
utilization of 0.8 (7B) or 0.75 (3B). Rollout sam-
pling uses temperature τ= 0.6 , top-k= 20 , and
top-p= 0.95 . We sample 16 candidate responses
per prompt for 7B models and 32 for 3B models
with an over-sample rate of 0.1. The KL divergence
coefficient βis set to 0.001 with low-variance KL
loss, and the clip ratio ranges from 0.2 to 0.28.
B.2 Reward Computation
The path-centric reward combines three compo-
nents with the following default weights: format
reward weight λf= 0.1 , path reward weight λp=
0.3, and outcome accuracy weight λa= 0.6 . The
reference planner uses a proprietary instruction-
tuned model (anonymized as HY 2.0-Instruct) to
generate guidance trajectories, which are cached
offline before training to avoid runtime overhead.
For self-consistency scoring, we sample 3 inde-
pendent reasoning paths per query and computepairwise agreement using Jaccard similarity on ex-
tracted evidence spans. The soft outcome scoring
applies a decay factor of 0.5 for partial matches
when the final answer is incorrect but the reasoning
path demonstrates high path quality.
B.3 Computational Cost
Reference planners are generated offline for all 90K
training samples using HY 2.0-Instruct, with each
sample requiring on average 1.91 LLM calls. This
is a one-time cost cached before RL training and
amortized over all subsequent runs.
B.4 Inference Settings
During inference, we set the maximum action bud-
getB= 4 , allowing up to 4 search-reason itera-
tions per query. The retriever returns top-3 pas-
sages per search step. We use sampling with tem-
perature 0.6 and top- p0.95 for validation. Model
checkpoints are saved every 10 steps, and we select
the checkpoint with the highest validation accuracy
for final evaluation.
C Algorithms
This section provides algorithmic descriptions of
the key components in SEARCH-P1: (1) offline
reference planner generation, (2) agentic RAG in-
ference, and (3) path-centric reward computation.
Algorithm 1 describes reference planner genera-
tion using a high-capability LLM (HY 2.0-Instruct)
to produce structured plans and reference reasoning
paths, cached offline for training.
Algorithm 2 illustrates agentic RAG inference:
the model iteratively generates reasoning, issues
search queries via <tool_call> , and receives re-
trieved passages as <tool_response> until the ac-
tion budget is exhausted or an answer is produced.
Algorithm 3 details the reward computation com-
bining format, dual-track path-centric, and soft out-
come signals.
D Prompt Templates
This section presents the prompt templates used
in SEARCH-P1 for inference, reference planner
generation, and reward evaluation.
D.1 Agentic RAG Inference Prompt
Figure 7 shows the prompt template used dur-
ing both training rollouts and inference. The
prompt instructs the model to decompose ques-
tions into sub-tasks, execute searches iteratively,
9

Algorithm 1Reference Planner Generation
Require:Training datasetD={(q i, ai)}N
i=1, reference LLMM ref
Ensure:Reference trajectoriesT ref={(p i, ri)}N
i=1
1:foreach(q, a)∈ Ddo
2:promptp←PLANNERPROMPT(q){Generate planning prompt}
3:p← M ref(promptp){Generate reference plan}
4:promptr←REASONINGPROMPT(q, p){Generate reasoning prompt}
5:r← M ref(promptr){Generate reference reasoning path}
6:T ref← T ref∪ {(p, r)}
7:end for
8:returnT ref
Algorithm 2Agentic RAG Inference
Require:Questionq, policy modelπ, retrieverR, action budgetB, top-K
Ensure:Generated trajectoryywith final answer
1:y←<reasoning>;t←1
2:whilet≤Bdo
3:∆←GENERATE(π, y)until</tool_call>or</answer>
4:y←y∥∆
5:ifCONTAINS(y,</answer>)then
6:break{Final answer generated}
7:end if
8:ifCONTAINS(∆,<tool_call>)then
9:query←EXTRACT(∆,<tool_call>)
10:docs← R(query, K){Retrieve top-Kpassages}
11:y←y∥<tool_response>∥docs∥</tool_response>
12:t←t+ 1
13:end if
14:end while
15:if notCONTAINS(y,</answer>)then
16:y←y∥<answer>∥GENERATE(π, y)until</answer>
17:end if
18:returny
and produce structured outputs with <reasoning> ,
<tool_call>, and<answer>tags.
D.2 Reference Planner Generation Prompt
Figure 8 shows the prompt used to generate ref-
erence plans and reasoning paths from HY 2.0-
Instruct. Given a question and its correct answer,
the reference LLM produces an optimized search
strategy that serves as guidance during path reward
computation.
D.3 Dual-Track Evaluation Prompt
Figure 9 presents the prompt used for dual-track
path evaluation. An evaluator LLM assesses
the model’s trajectory along two dimensions:
self-consistency (execution of its own plan) andreference-alignment (coverage of expert reference
steps), along with outcome quality scoring.
E Case Study
To qualitatively illustrate SEARCH-P1’s advan-
tages, we present a representative case from
MuSiQue demonstrating how path-centric reward
shaping leads to more accurate multi-hop reason-
ing.
E.1 Multi-Hop Reasoning Comparison
Figure 10 compares Search-R1 and SEARCH-P1
on a multi-hop question. Without explicit plan-
ning, Search-R1 misinterprets “rock & roll” as a
genre descriptor, retrieving information about the
wrong entity. In contrast, SEARCH-P1’s planning
10

Algorithm 3SEARCH-P1 Reward Computation
Require:Trajectoryy, ground trutha∗, reference planp ref, reference pathr ref
Ensure:Total rewardR(y)
1:// Format Reward
2:ifVALIDFORMAT(y)andHASANSWER(y)andHASTOOLCALL(y)then
3:r f←0.1
4:else ifHASANSWER(y)andHASTOOLRESPONSE(y)then
5:r f←0.05
6:else
7:return0{Invalid trajectory}
8:end if
9:
10:// Path-Centric Reward via Dual-Track Evaluation
11:eval←LLMEVALUATE(y, p ref, rref){Call evaluator LLM}
12:r planner←eval.planner_score {Plan quality: 0.2/0.6/1.0/1.2}
13:
14:// Track A: Self-Consistency
15:s self←r planner×eval.eff_steps_self
eval.model_plan_steps
16:
17:// Track B: Reference-Alignment
18:s ref←eval.eff_steps_ref
|steps(r ref)|
19:
20:r p←max(s self, sref){Best of dual tracks}
21:
22:// Outcome Reward with Soft Scoring
23:ifEXACTMATCH(GETANSWER(y), a∗)then
24:r o←1.0
25:else
26:r o←0.8×eval.acc_score+ 0.2×eval.reason_score
27:end if
28:
29:R(y)←λ f·rf+λp·rp+λo·ro
30:returnR(y)
correctly identifies “Bang Bang Rock & Roll” as a
complete album title, leading to the correct answer.
F Additional Results
F.1 Impact of Retrieved Documents per
Search
Table 5 shows how the number of retrieved doc-
uments per search iteration affects model perfor-
mance. Retrieving too few documents may miss
relevant information, while retrieving too many can
introduce noise and increase context length.
F.2 Effect of Format Reward on Output
Compliance
Table 6 analyzes the relationship between the for-
mat reward component and the model’s ability toproduce properly formatted outputs.
F.3 Search Iterations Analysis
Table 7 presents the distribution of search itera-
tions for successful and failed cases across different
datasets.
Key Observations.(1) General QA datasets
achieve most successes with single-iteration
searches. (2) Multi-hop datasets show successful
cases concentrated at 2 iterations. (3) Failed cases
consistently show higher 3+ iteration rates, suggest-
ing excessive searching indicates difficulty. (4) The
3B model requires slightly more iterations than 7B.
11

Agentic RAG Inference Prompt
You are a meticulousDeep Research Agent. Your goal is to provide a comprehensive and accurate
answer by conducting multiple rounds of search.
## CRITICAL INSTRUCTIONS
1. Detailed Planning (<reasoning>):
• In the first turn, you MUST break the question down intomultiple dependent sub-questions.
• Focus on one sub-question at a time.
2. Step-by-Step Execution (<tool_call>):
• Execute only ONE search query per turn.
• After receiving results, verify: “Is this sufficient? Do I need more details?”
3. No Guessing:
• If results are incomplete, issue another search. Do NOT hallucinate.
4. Final Answer (<answer>):
• Only output <answer> when ALL necessary information is gathered.
## CURRENT TASK
Question:{question}
Figure 7: Prompt template for agentic RAG inference. The model is instructed to plan, search iteratively, and
provide structured outputs.
Model # DocsGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
7B1 45.8 67.2 38.5 35.2 32.0 15.5 36.0 38.6 72.8
2 52.0 74.2 44.0 39.5 36.8 18.8 41.0 43.8 80.5
3 56.6 78.6 47.5 42.9 39.821.8 44.0 47.3 86.2
5 55.8 77.8 46.8 42.5 39.222.4 45.2 47.1 85.5
10 53.5 74.5 44.5 40.5 37.2 20.1 42.8 44.7 82.5
3B1 42.8 63.5 38.2 28.8 28.5 9.2 21.2 33.2 64.5
2 48.5 69.8 43.6 32.8 33.2 11.5 25.2 37.8 73.2
3 53.0 74.5 47.9 36.236.6 13.328.8 41.5 79.5
5 52.2 73.8 47.2 36.536.0 12.830.4 41.3 78.8
10 49.8 71.2 45.0 33.8 34.2 11.8 27.6 39.1 75.5
Table 5: Performance (ACC %) with different numbers of retrieved documents per search. Retrieving 3 documents
achieves the best average performance. While 5 documents shows advantages on specific datasets (MuSiQue,
Bamboogle for 7B; HotpotQA, Bamboogle for 3B), the overall best configuration is 3 documents.
F.4 Detailed Ablation on Path-Centric
Reward Components
Table 8 provides the complete per-dataset break-
down for the path-centric reward component abla-
tion study (extending Table 2 in the main paper).
F.5 Detailed Model and RL Algorithm
Analysis
Table 9 extends Table 3 with per-dataset accuracy
for different base models and RL algorithms.
F.6 Detailed Soft Outcome Scoring Analysis
Table 10 provides the per-dataset breakdown of the
soft outcome scoring ablation (corresponding to
Figure 4 in the main paper).F.7 Detailed Hyperparameter Sensitivity
Analysis
Tables 11 and 12 provide the per-dataset break-
down for hyperparameter sensitivity analysis (cor-
responding to Figure 5 in the main paper).
F.8 Detailed LLM Evaluator Analysis
Table 13 provides the per-dataset breakdown for
the LLM evaluator analysis. The Qwen3-8B evalu-
ator shows larger drops on multi-hop datasets (e.g.,
−4.0 on MuSiQue, −4.0 on 2Wiki) where accu-
rately counting covered reasoning steps is more
challenging.
12

Reference Planner Generation Prompt
You are an expert planner and reasoning optimizer.
Current Question:{question}
Correct Answer:{golden_answers}
Your task is to generate:
1. Optimized Reasoning Path:A sequence of search queries that would lead directly to the correct
answer in the most efficient way. Format as a numbered list.
2. Optimized Planner:A concise, step-by-step instruction on how a reasoning agent should solve
this question correctly and efficiently.
Important:
• Focus on the minimal set of queries needed.
• Avoid redundant or inefficient steps.
Output format:
<correct_reasoning_path>
1. query 1
2. query 2
</correct_reasoning_path>
<optimized_planner>
To solve this, first search for... then...
</optimized_planner>
Figure 8: Prompt template for reference planner generation. HY 2.0-Instruct generates optimal search strategies for
each training sample.
Model MethodGeneral QA Multi-Hop QA Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
7Bw/o Format Reward 82.8 84.2 81.5 79.1 76.8 75.2 83.6 88.2
w/ Format Reward 95.1 95.6 94.2 91.5 88.8 87.6 96.4 97.5
3Bw/o Format Reward 72.8 75.5 72.2 67.1 61.8 62.4 65.2 78.8
w/ Format Reward 87.2 88.1 85.4 80.2 74.8 75.1 78.0 91.5
Table 6: Format compliance rate (%) with and without format reward. Adding format reward significantly improves
the model’s ability to produce properly structured responses with parseable answers.
13

Dual-Track Evaluation Prompt
You are an expert RL researcher evaluating an AI agent’s trajectory.
Your task is to conduct aDual-Track Evaluation:
1.Self-Consistency Track: How well did the agent execute its OWN plan?
2.Reference-Alignment Track: How well did the agent follow the Expert plan?
3.Outcome Evaluation: Assess accuracy and reasoning quality.
Evaluation Inputs:
•Question:{question}
•Correct Answer:{golden_answers}
•Reference Planner:{ref_planner}
•Reference Path:{ref_reasoning_path}
•Model Trajectory:{trajectory}
Scoring Criteria:
•Planner Score: 0.2 (Bad) / 0.6 (Average) / 1.0 (Good) / 1.2 (Excellent)
•Outcome Accuracy: 0.0 (Wrong) / 0.5 (Partial) / 1.0 (Correct)
•Reasoning Quality: 0.0 / 0.5 / 0.8 / 1.0
Output:JSON with planner_score ,model_plan_steps ,effective_steps_self ,
effective_steps_ref,outcome_accuracy_score,outcome_reasoning_score.
Figure 9: Prompt template for dual-track evaluation. The evaluator LLM assesses both self-consistency and
reference-alignment of model trajectories.
Query: Who is the lead singer of the band performing Bang Bang Rock & Roll? Answer:
Eddie Argos
Search-R1 (Incorrect)
<think> I need to find who performs “Bang
Bang Rock & Roll.”</think>
<search> song Bang Bang rock and roll
genre </search>
<result> In 2017, Dan Auerbach collabo-
rated with Robert Finley on “Bang Bang,”
featuring a classic rock and roll sound...
</result>
<search> lead singer of The Black Keys
</search>
<result> Dan Auerbach serves as the lead
vocalist...</result>
Final:✗Dan AuerbachSEARCH-P1 (Correct)
<reasoning> I need to: 1. Find the band
performing “Bang Bang Rock & Roll”. 2.
Find the lead singer.</reasoning>
<tool_call> “Bang Bang Rock & Roll”
album </tool_call>
<tool_response> “Bang Bang Rock &
Roll” is the debut album by British rock band
Art Brut...</tool_response>
<tool_call> lead singer of Art Brut
</tool_call>
<tool_response> ...frontmanEddie
Argos’ enthusiastic vocal delivery...
</tool_response>
Final:✓Eddie Argos
Figure 10: Comparison of reasoning trajectories. Search-R1’s imprecise query retrieves valid but irrelevant results;
SEARCH-P1’s planning-driven query retrieves the correct information. Highlighted text shows search queries.
14

Model DatasetSuccessful Cases Failed Cases
1 iter 2 iter 3+ iter 1 iter 2 iter 3+ iter
7BNQ 68.5% 22.3% 9.2% 45.2% 28.6% 26.2%
TriviaQA 72.1% 19.8% 8.1% 48.3% 26.4% 25.3%
PopQA 65.8% 24.5% 9.7% 42.1% 29.8% 28.1%
HotpotQA 28.5% 48.2% 23.3% 35.4% 30.1% 34.5%
2Wiki 25.1% 50.6% 24.3% 33.8% 29.5% 36.7%
MuSiQue 18.5% 52.3% 29.2% 31.2% 28.6% 40.2%
Bamboogle 22.4% 45.6% 32.0% 28.3% 25.4% 46.3%
AD-QA 35.2% 42.5% 22.3% 22.8% 28.5% 48.7%
Average 42.0% 38.2% 19.8% 35.9% 28.4% 35.7%
3BNQ 62.3% 26.1% 11.6% 40.5% 30.2% 29.3%
TriviaQA 66.8% 23.4% 9.8% 44.1% 28.5% 27.4%
PopQA 60.2% 28.3% 11.5% 38.6% 31.2% 30.2%
HotpotQA 22.4% 45.8% 31.8% 30.2% 28.5% 41.3%
2Wiki 20.3% 47.2% 32.5% 28.5% 27.8% 43.7%
MuSiQue 14.2% 48.6% 37.2% 26.4% 26.2% 47.4%
Bamboogle 16.8% 42.1% 41.1% 24.5% 23.8% 51.7%
AD-QA 28.5% 40.2% 31.3% 18.2% 25.6% 56.2%
Average 36.4% 37.7% 25.9% 31.4% 27.7% 40.9%
Table 7: Distribution of search iterations for successful and failed cases. General QA datasets (NQ, TriviaQA,
PopQA) show high success rates with single-iteration searches, while Multi-Hop QA datasets require more iterations.
Failed cases consistently show higher proportions of 3+ iterations, suggesting that excessive searching indicates
difficulty in finding relevant information.
MethodGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
Qwen2.5-7B
SEARCH-P1 (Full) 56.6 78.6 47.5 42.9 39.8 21.8 44.0 47.3 86.2
w/o Reference-Alignment 50.8 73.0 43.2 37.5 34.0 16.8 39.5 42.1 78.8
w/o Self-Consistency 53.0 75.5 44.8 39.8 37.2 19.2 40.2 44.2 82.5
Search-R1 (Baseline) 42.9 62.3 42.7 38.6 34.6 16.2 40.0 39.6 65.6
Qwen2.5-3B
SEARCH-P1 (Full) 53.0 74.5 47.9 36.2 36.6 13.3 28.8 41.5 79.5
w/o Reference-Alignment 47.8 68.5 43.4 31.2 30.5 10.1 24.0 36.5 70.5
w/o Self-Consistency 49.8 71.8 45.2 33.8 34.0 11.8 26.0 38.9 75.2
Search-R1 (Baseline) 39.7 56.5 39.1 33.1 31.0 12.4 23.2 33.6 58.3
Table 8: Detailed ablation study on path reward components (ACC %). Removing reference-alignment causes
larger drops on multi-hop datasets where external guidance is more critical, while removing self-consistency affects
general QA more where the model’s own planning suffices.
Model RLGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
Qwen2.5-3B-Inst. GRPO 53.0 74.5 47.9 36.2 36.6 13.3 28.8 41.5 79.5
Qwen2.5-3B-Inst. PPO 51.6 73.1 46.8 34.8 35.4 12.6 27.6 40.3 78.1
Llama-3.2-3B-Inst. GRPO 50.2 71.8 45.4 33.9 34.1 11.8 26.0 39.0 75.8
Llama-3.2-3B-Inst. PPO 48.9 70.2 44.2 32.8 33.0 10.9 24.8 37.8 74.2
Table 9: Detailed accuracy (%) across different base models and RL algorithms on all datasets. Qwen2.5 consistently
outperforms Llama-3.2, and GRPO achieves slightly higher accuracy than PPO across all datasets.
15

Model MethodGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
7Bw/o Soft Scoring 55.4 77.5 46.0 39.2 36.5 18.5 40.8 44.8 77.4
w/ Soft Scoring 56.6 78.6 47.5 42.9 39.8 21.8 44.0 47.3 86.2
∆ +1.2 +1.1 +1.5 +3.7 +3.3 +3.3 +3.2 +2.5 +8.8
3Bw/o Soft Scoring 51.8 73.2 46.5 32.6 33.1 10.4 24.5 38.9 68.5
w/ Soft Scoring 53.0 74.5 47.9 36.2 36.6 13.3 28.8 41.5 79.5
∆ +1.2 +1.3 +1.4 +3.6 +3.5 +2.9 +4.3 +2.6 +11.0
Table 10: Effect of soft outcome scoring (ACC %). Multi-hop QA datasets benefit more from soft scoring
(+3.0–3.7%) compared to general QA datasets (+1.1–1.5%), while the internal AD-QA dataset shows the largest
improvement (+8.8–11.0%), confirming that complex enterprise queries benefit most from partial credit signals.
λpGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
0.2 55.5 77.8 46.8 41.8 38.5 20.844.5 46.5 83.5
0.3 56.6 78.6 47.5 42.9 39.8 21.844.0 47.3 86.2
0.4 55.8 78.0 47.2 42.2 39.2 21.5 43.8 46.8 85.0
Table 11: Effect of path reward weight λpon performance (ACC %, Qwen2.5-7B). The optimal value is λp= 0.3 ,
which achieves the best average performance. While λp= 0.2 shows slight advantage on Bamboogle, λp= 0.3
provides the best overall balance.
λaGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
0.6 56.6 78.6 47.5 42.9 39.821.8 44.0 47.3 86.2
0.8 56.0 78.2 47.0 42.5 39.222.2 44.8 47.1 85.5
1.0 54.8 76.8 45.8 41.2 38.0 20.8 43.2 45.8 83.2
Table 12: Effect of outcome accuracy weight λaon performance (ACC %, Qwen2.5-7B). The optimal value is
λa= 0.6 , which achieves the best average performance. While λa= 0.8 shows slight advantages on MuSiQue and
Bamboogle,λ a= 0.6provides better overall results.
LLM EvaluatorGeneral QA Multi-Hop QAAvg.Internal
NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle AD-QA
HY 2.0-Instruct 56.6 78.6 47.5 42.9 39.8 21.8 44.0 47.3 86.2
Qwen3-32B 55.5 77.5 46.2 41.8 38.8 20.8 42.5 46.2 84.2
Qwen3-8B 52.8 74.2 43.5 39.2 35.8 17.8 39.8 43.3 79.5
Table 13: Detailed accuracy (%) across LLM evaluators (Qwen2.5-7B + GRPO). Qwen3-32B shows modest
degradation ( −1.1 Avg.), while Qwen3-8B exhibits larger drops on multi-hop tasks where step coverage evaluation
is more challenging.
16