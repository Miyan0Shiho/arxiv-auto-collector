# CRITIC-R1: Learning Structured Critics for Retrieval-Augmented Generation

**Authors**: Wenhan Xiao, Ziwei Zhang, Chuanyue Yu, Xingcheng Fu, Qingyun Sun, Runhua Xu, Jianxin Li

**Published**: 2026-05-28 13:11:36

**PDF URL**: [https://arxiv.org/pdf/2605.29886v1](https://arxiv.org/pdf/2605.29886v1)

## Abstract
Retrieval-augmented generation (RAG) improves knowledge-intensive question answering by incorporating external evidence. However, existing RAG methods still suffer from hallucinations and subtle reasoning errors. Recent studies introduce external critics to refine RAG outputs, yet they often provide coarse-grained and weakly structured feedback, exhibit over-aggressive intervention, and lead to noisy and unreliable refinement, limiting their effectiveness for correction. To tackle these issues, we propose CRITIC-R1, a structured critic framework that formulates and learns RAG critique as an explicit error diagnosis problem using reinforcement learning (RL). Our framework categorizes common RAG errors into multiple diagnostic dimensions, including verdict, error location, reasoning analysis, and fix generation. To learn these capabilities, we design two reward functions: Conservative Judgement Alignment (CJA) first encourages calibrated high-level judgements while mitigating the over-aggressive phenomenon, whereas Diagnostic Quality Alignment (DQA) further improves fine-grained diagnostic feedback through gated rewards. We train the critic model using GRPO-based RL with process-level supervision collected from external LLM teacher models. Experiments across five QA benchmarks show that CRITIC-R1 consistently improves answer quality over strong RAG baselines.
  Our source code is available at https://anonymous.4open.science/r/critic-r1-FCB0

## Full Text


<!-- PDF content starts -->

CRITIC-R1: Learning Structured Critics for Retrieval-Augmented
Generation
Wenhan Xiao1, Ziwei Zhang2*, Chuanyue Yu1, Xingcheng Fu3, Qingyun Sun2, Runhua Xu2, Jianxin Li2
1Nankai University2Beihang University3Guangxi Normal University
2213121@mail.nankai.edu.cn,zwzhang@buaa.edu.cn,yuchuanyue@mail.nankai.edu.cn,fuxc@gxnu.edu.cn
sunqy@buaa.edu.cn,runhua@buaa.edu.cn,lijx@buaa.edu.cn
Abstract
Retrieval-augmented generation (RAG) im-
proves knowledge-intensive question answer-
ing by incorporating external evidence. How-
ever, existing RAG methods still suffer from
hallucinations and subtle reasoning errors. Re-
cent studies introduce external critics to refine
RAG outputs, yet they often provide coarse-
grained and weakly structured feedback, ex-
hibit over-aggressive intervention, and lead to
noisy and unreliable refinement, limiting their
effectiveness for correction. To tackle these is-
sues, we propose CRITIC-R1, a structured
critic framework that formulates and learns
RAG critique as an explicit error diagnosis
problem using reinforcement learning (RL).
Our framework categorizes common RAG er-
rors into multiple diagnostic dimensions, in-
cluding verdict, error location, reasoning analy-
sis, and fix generation. To learn these capabili-
ties, we design two reward functions: Conserva-
tive Judgement Alignment (CJA) first encour-
ages calibrated high-level judgements while
mitigating the over-aggressive phenomenon,
whereas Diagnostic Quality Alignment (DQA)
further improves fine-grained diagnostic feed-
back through gated rewards. We train the
critic model using GRPO-based RL with
process-level supervision collected from ex-
ternal LLM teacher models. Experiments
across five QA benchmarks show that CRITIC-
R1 consistently improves answer quality over
strong RAG baselines. Our source code
is available at https://anonymous.4open.
science/r/critic-r1-FCB0
1 Introduction
With the rapid development of Retrieval-
Augmented Generation (RAG) (Lewis et al.,
2020; Lin et al., 2023; Izacard et al., 2023),
Large Language Models (LLMs) have achieved
strong performance on knowledge-intensive
question answering tasks by leveraging external
*Corresponding author.evidence. However, existing RAG methods
still produce answers that are not faithfully
grounded in the retrieved evidence, exhibiting
hallucinations (Huang et al., 2025) or subtle
reasoning errors. To address these issues, recent
works have explored incorporating critique (Wei
et al., 2025; Jiang et al., 2025) and refinement
mechanisms (Asai et al., 2023; Madaan et al.,
2023; Yan et al., 2024) into the RAG pipeline. In
particular, external critics, where another model
evaluates generated outputs and provides feedback,
have emerged as a promising direction, as they
offer independent assessment signals beyond the
generator itself (Xu et al., 2024), and enable more
reliable error detection and correction.
Despite their potential, existing external RAG
critics suffer from several key limitations: (1) Most
critics provide coarse or weakly diagnostic feed-
back, which does not clearly separate different
critic capabilities. As a result, their critiques may
be difficult to parse, evaluate, and directly used for
targeted correction. (2) Existing critics often ex-
hibit over-aggressive intervention, i.e., raise false
alarms and flag correct responses as erroneous, or
even trigger harmful edits. (3) The supervision sig-
nals used for training critic models often rely on
heuristic and noisy strict string matching, which
is not fully reliable and therefore hinders the critic
model from learning error diagnosis.
To address these challenges, we propose
CRITIC-R1, a structured critic framework for er-
ror diagnosis and correction in RAG using rein-
forcement learning. Our framework is built upon
three key components. First, we present a system-
atic categorization of error types in RAG, including
retrieval failures, reasoning errors, and answer gen-
eration issues. This taxonomy provides a structured
view of where and how RAG systems fail, form-
ing the foundation for fine-grained error diagnosis
and supervision. Based on this taxonomy, we fur-
ther develop a structured critic framework, which
1arXiv:2605.29886v1  [cs.CL]  28 May 2026

LLM
Question: What nationality was James Henry Miller's wife?
(C) External -Critic
(D) Critic -R1(A) LLM -Only (B) RAG
Answer Question
Question
Incorrect
Coarse
Feedback
Fine-Grained Critique
Mismatch/ 
IrrelevantFocus/
Make sure
Inference:3External
EvidenceStructured 
DiagnosisReliable 
Refinement
CorrectContinue
Reasoning 
Steps
Answer
Back to
TrajectoryRefinementScore:7/10
Possibly
incorrect
His wife was an 
American
Question
 LLM RetrieveDoc1 Jim is an American
Doc2 Jane  teaches German
LLMDoc3 James  stays in Edinburg h
AnswerShe was a 
German
Question
 LLM TrajectoryCritic
LLM
Question
 LLM TrajectoryTrajectory LLMOver -aggressive
Correct
Answer
Wrong
AnswerHarmful 
After verification, 
James Henry Miller's 
wife was English.She was a 
German
She was a 
GermanHis wife was 
English
EditFalse
Alarm
Retrieved 
ChunksFigure 1: An illustrative comparison of different RAG paradigms. (A) LLM-only method directly generates an
incorrect answer without external evidence. (B) RAG retrieves documents but is misled by ambiguous or irrelevant
information, leading to incorrect reasoning. (C) External Critic gives coarse feedback and can be over-aggressive,
making refinement unreliable. (D) Our method produces structured critiques that explicitly identify error locations
and reasons, and provide actionable suggestions, enabling informed and reliable refinement decisions.
trains a critic model using reinforcement learning
through tailored reward designs. Specifically, the
Conservative Judgement Alignment (CJA) reward
first encourages the critic model to make calibrated
high-level judgements. Then, the Diagnostic Qual-
ity Alignment (DQA) reward further incentivizes
the critic model to produce fine-grained diagnostic
feedback, including error type, location, reason-
ing, and suggested fixes, which can be selectively
used to guide the generation refinement. To col-
lect supervision signals for calculating rewards,
we propose process-level supervision collection
using an external LLM teacher model for annota-
tion. Lastly, we introduce a two-stage RL training
framework based on Group Relative Policy Opti-
mization (GRPO) (Shao et al., 2024) to train the
critic model, which separates high-level judgement
from fine-grained diagnosis.
Our contributions are summarized as follows:
•We formulate RAG critique as a structured error
diagnosis problem and provide a systematic cat-
egorization of common RAG error types. This
schema organizes critique into multiple diag-
nostic fields, enabling parseable feedback and
explicit modeling of critic capabilities.
•We propose two tailored reward functions for
RAG critic: Conservative Judgement Alignment
(CJA) to encourage the critic model to makehigh-level judgement while avoiding being over-
aggressive, and Diagnostic Quality Alignment
(DQA) to produce fine-grained diagnosis.
•We also propose process-level supervision col-
lection using LLM-based teacher models and a
two-stage RL framework to train the critic. Ex-
perimental results verify the effectiveness of the
proposed method.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has been
widely adopted for knowledge-intensive tasks by
combining parametric models with external re-
trieval systems. Early works such as RAG (Lewis
et al., 2020), FiD (Izacard and Grave, 2021), and
Atlas (Izacard et al., 2023) demonstrate the effec-
tiveness of integrating retrieved evidence into gen-
eration. Recent methods such as ReAct (Yao et al.,
2022), Search-R1 (Jin et al., 2025), and Search-
o1 (Li et al., 2025) optimize search and reasoning
trajectories to enable better performance.
Despite these advances, RAG-based methods
still suffer from unreliable grounding and reason-
ing failures (Tonmoy et al., 2024). In particular,
retrieved evidence can be incomplete or misleading,
and errors in intermediate steps may propagate to
the final answer. These limitations remain central
2

challenges for building reliable RAG systems.
2.2 Critique and Iterative Refinement
A growing line of work improves RAG outputs
through critique, verification, and iterative refine-
ment. Self-RAG (Asai et al., 2023) introduces
special tokens to monitor generation quality and
guide refinement, while Self-Refine (Madaan et al.,
2023) iteratively generates feedback to improve
model outputs. Meta-RAG (Zhou et al., 2024) ex-
plores multi-view reasoning to critique and revise
answers, and Self-Contrast (Zhang et al., 2024)
identifies inconsistencies across multiple reasoning
paths. CRAG (Yan et al., 2024) further enhances
retrieval by introducing a retrieval evaluator.
Beyond self-reflection, external critique and
verification mechanisms have also been explored.
CRITIC (Gou et al., 2023) incorporates external
tools to verify intermediate reasoning steps and pro-
vide corrective feedback. Align-RAG (Wei et al.,
2025) introduces a critique-and-optimize frame-
work, where an external model evaluates generated
reasoning and guides iterative refinement. RAG-
STAR (Jiang et al., 2025) scores candidate answers
and refines reasoning trajectories based on feed-
back. RAG-Critic (Dong et al., 2025) further de-
velops a critic-guided agentic workflow that uses
error feedback to select correction flows.
Despite their potential, existing critique methods
provide limited support for fine-grained error lo-
calization and actionable correction. A concurrent
work Doctor-RAG (Jiao et al., 2026) has also stud-
ied error categorization in RAG. However, its er-
ror categories are mainly used for analyzing rather
than for learning a critic model with explicitly op-
timized diagnostic capabilities. In contrast, our
work treats RAG error categorization as the foun-
dation for structured critic training, where the critic
learns to produce calibrated judgements, precise
error locations, and actionable fixes.
3 Method
In this section, we present CRITIC-R1, a struc-
tured critic learning framework for RAG. First, we
present analyses of RAG error types to motivate
our method. Next, we introduce our reward designs
for training the critic model, followed by how to
collect supervision signals. Lastly, we introduce
our optimization method using RL. Figure 2 shows
the overall framework.3.1 Structured Critique Formulation
To make critique more explicit and actionable, we
formulate critique in RAG as a structured error
diagnosis problem. We first conduct an error analy-
sis on RAG outputs over HotpotQA examples, as
shown in Figure 3. The results show that errors
are distributed across multiple stages, including
retrieval failures, reasoning inconsistencies, and
answer generation issues. These error distributions
indicate that a single correct/incorrect judgment is
insufficient to guide effective correction. There-
fore, a useful critic should not only judge whether
an answer is correct, but also identify where the
error occurs, explain why it occurs, and suggest
how to fix it.
Motivated by this analysis, we propose a struc-
tured critique with four components: <verdict> ,
<location> ,<reason> , and <fix> . The ver-
dict indicates whether the reasoning trajectory
isCORRECT ,INCORRECT , orUNSURE . The location
identifies the error stage or evidence index. The rea-
son explains the failure in detail. The fix provides
actionable guidance for refinement. This structured
output makes the critique interpretable and directly
usable for downstream correction.
3.2 Reward Design for Critique
To train the critic, we design two tailored reward
designs with two complementary objectives: CJA
for conservative high-level judgement and DQA
for fine-grained diagnostic quality.
3.2.1 Conservative Judgement Alignment
The critic must first make reliable intervention deci-
sions. If the critic is overconfident, it will inevitably
lead to an over-aggressive intervention, i.e., set
off false alarms and trigger unnecessary or even
harmful edits. To address this issue, we introduce
Conservative Judgement Alignment (CJA), which
encourages the critic to make calibrated high-level
judgements before providing detailed diagnostic
feedback. CJA has the following components.
Verdict reliability.The critic predicts a verdict
from CORRECT ,INCORRECT , and UNSURE . To dis-
courage over-aggressive intervention, we adopt a
conservative verdict reward that penalizes false
alarms more strongly and allows abstention under
uncertainty. The detailed rewards are provided in
Table 7 in Appendix A.1.
Structured format.The critic is also required to
follow the predefined output schema. This encour-
3

(a) Structured Critique Formulation
Process -Level Supervision
LLM
Verdict Location
Reason
VerdictFix
Fix Reason
Judge1:{INCORRECT,
Information:Doc3,
Irrelevant,Re -Search}
Judge2:{CORRECT}
Judge3:{INCORRECT,
Answer
Entity Mismatch,
Verify the entity}Fine-Grained
Process -Level
Signal①Conservative Judgement Alignment (CJA)
②Diagnostic Quality Alignment (DQA)Coarse -Grained Fine-Grained Teacher
Parse Output
QuestionInfo Answer Infer
 Search
Verdict
INCORRECT
UNSURE
Info:docK
Think:StepK
Mismatch
Irrelevant
Off-Target Search:StepK
Verify Entity
Query ControlAlign Answer
Location Reason
CORRECTFix
Auxiliary
 Penalt y
Auxiliary
 Penalt y𝑶i:Output 𝒓𝒊 :Reward 𝑨𝒊:Advantage𝑶𝟏
𝑶𝟐
𝑶𝒏…
Reward Function𝒓𝟏
𝒓𝟐
𝒓𝒏… …𝑨𝟏
𝑨𝟐
𝑨𝒏
Supervision 𝑻𝒌 𝑻𝒌′
Verdict Reliability
Format & Verdict Alignment
Predicted ෝ𝒗Reference 𝒗∗
I[ොv==𝒗∗]Yes
No
DQA Inactive(b) Critique Learning FrameworkTrajectory Set
SUCCESSFUL
FAILED
AMBIGUOUS
VerdictFine-Grained Refinement
Trajectory
Critic
Search Error
 Retrieval Error Reasoning Error Answer Error
Search Question Think Info
 Answer
FeedbackTrainingRefining
Over -Aggressive
Gated Diagnostic Reward
Supervision
{Question}
{Information}
{Trajectory}
Multi -Sample 
SuperviseFigure 2: An overview of CRITIC-R1: (a) We formulate RAG critic as a structured critique framework, including
verdict, error location, reasoning analysis and fix generation, to enable fine-grained refinement. (b) Critique
learning framework, containing process-level supervision and two tailored reward functions, where CJA encourages
calibrated high-level judgement and DQA improves fine-grained diagnoses.
Answer 
Error
Retrieval
Error
Reasoning
ErrorSearch
ErrorMissing Key 
Evidence
Irrelevant
Retrieval
Multi -Hop ErrorEvidence 
MisuseWrong Entity
Missing 
Constraint
12.2%Entity 
MismatchWrong Type
Off Target14.5%
38.2%
35.1%9.7%37.4%
0.8%
34.4%11.5%3.1%
1.0%
1.5%
0.8%Failure
Cases
Figure 3: Error distribution on HotpotQA using Search-
R1 (Jin et al., 2025). Among the analysed samples,
errors contain diverse failure modes across retrieval,
reasoning, and answer generation.
ages critiques to be well-formed, interpretable, and
suitable for downstream refinement. Formally, the
format reward is:
rformat =(
α,valid format,
−γ,otherwise.(1)Combining these components, we define
rCJA=r format +r verdict +r aux,(2)
where rauxdenotes auxiliary penalties for degener-
ated critiques, including trivial responses or overly
generic feedback that fails to provide meaningful
signals, as detailed in Appendix A.1. Overall, the
CJA reward teaches the critic model when not to
edit, mitigating the over-aggressive intervention be-
fore optimizing fine-grained diagnostic feedback.
3.2.2 Diagnostic Quality Alignment
To enable sufficiently informative correction, the
critic should not only decide whether intervention
is needed, but also diagnose where the error occurs,
why it occurs, and how it can be fixed. To this
end, we introduce Diagnostic Quality Alignment
(DQA), which optimizes the fine-grained useful-
ness of structured critiques.
Diagnostic dimensions.DQA focuses on three
complementary capabilities. First, the critic should
localize the error in the RAG trajectory, such as
retrieved evidence, reasoning steps, or final answer
generation. Second, it should provide a diagnostic
4

reason explaining why the trajectory fails. Third,
it should propose a concrete fix that can guide the
generator during refinement. We aggregate these
components into a diagnostic reward:
rDQA=r loc+r reason +r fix,(3)
where rloc,rreason , and rfixevaluate localization,
explanation, and correction quality, respectively.
Detailed formulations of these reward components
are provided in Appendix A.1.
3.2.3 Process-Level Supervision
Trajectory-Level AnnotationAnswer-level cor-
rectness labels are insufficient for training a diag-
nostic critic model, as they do not specify where
a trajectory fails or how it should be repaired. We
therefore use a strong external LLM as a teacher
to provide multi-sample supervision over complete
RAG trajectories.
Consensus-based ConstructionWe parse the
teacher outputs into structured fields and aggregate
them through consensus. Specifically, we first de-
termine the reference verdict by agreement across
sampled critiques. The resulting annotations serve
as fine-grained process-level signals for construct-
ing the CJA and DQA rewards. Details are pro-
vided in Appendix A.2.
3.3 Optimization
3.3.1 GRPO
We train the critic model using reinforcement learn-
ing based on GRPO (Shao et al., 2024). Given
an input question q, the critic generates a struc-
tured output oas the critique according to a policy
πθ(o|q) . A reward function r(q, o) is used to
evaluate the quality of the generated critique. The
objective is to maximize the expected reward while
maintaining proximity to a reference policy:
max
πθEq∼D, o∼π θ(·|q)[r(q, o)−β· D KL(πθ∥πref)],
(4)
where qdenotes the input question sampled from
the dataset D,ois the generated critique, r(q, o) is
the reward function evaluating critique quality, πref
is a reference policy used for regularization, DKL
is the KL divergence andβcontrols the strength.
Following GRPO, for each input q, we sample a
group of Goutputs {oi}G
i=1using the current policy.
Instead of estimating a value function, GRPO com-
putes advantages based on the relative performanceof samples within the group. Let ri=r(q, o i)de-
note the reward of each sample. The advantage is
computed as: Ai=ri−µr
σr, where µrandσrdenote
the mean and standard deviation of rewards within
the group. This formulation encourages outputs
that perform better than their peers and stabilizes
training without an explicit value function.
3.3.2 Training Pipeline
We adopt a two-stage RL training pipeline. In the
first stage, we mainly optimize the critic model
with CJA, encouraging valid structured outputs and
conservative verdict prediction while penalizing
trivial critiques.
In the second stage, we further introduce DQA
with a gating mechanism:
r(2)=r format+rverdict +I[ˆv=v∗]·(r DQA+r aux),
(5)
where ˆvandv∗denote the predicted and reference
verdicts. Here, rDQA aggregates the localization,
reason, and fix rewards, while rauxincludes penal-
ties for trivial or generic feedback. The gating
term ensures that diagnostic rewards and auxiliary
feedback-quality penalties are activated only when
the high-level verdict is correct; otherwise, DQA
remains inactive. This prevents the model from
being rewarded for plausible but miscalibrated ex-
planations when its judgement is wrong. As a re-
sult, the critic model learns to produce more useful
fine-grained diagnostic feedback.
4 Experiment
In this section, we conduct experiments, aiming to
answer four research questions:
•RQ1: Does CRITIC-R1 improve downstream
performance?
•RQ2: What components of CRITIC-R1 con-
tribute to its effectiveness?
•RQ3: Is training a separate critic model more
effective than directly improving the generator?
•RQ4: Does CRITIC-R1 regulate the critic’s
intervention behavior during inference?
4.1 Experiment Setup
DatasetWe follow prior work (Wei et al., 2025;
Gao et al., 2023) and adopt widely-used ques-
tion answering benchmarks, including Natural
Questions (NQ) (Kwiatkowski et al., 2019), Hot-
potQA (Yang et al., 2018), TriviaQA (Joshi
et al., 2017), PopQA (Mallen et al., 2023) and
ASQA (Stelmakh et al., 2022). We train our model
5

MethodNQ HotpotQA TriviaQA ASQA PopQA
F1 SBERT Acc F1 SBERT Acc F1 SBERT Acc F1 SBERT Acc F1 SBERT Acc
Base LLM Generation
Vanilla 9.9 33.5 37.6 7.9 29.4 33.9 9.6 35.4 45.2 7.6 29.0 28.7 11.7 32.3 27.9
Reasoning and Retrieval Baselines
Naive-RAG 9.6 35.4 45.1 8.3 32.1 32.5 12.9 44.7 74.1 9.5 32.2 42.8 11.5 33.2 44.6
CoT 11.8 36.7 42.1 13.6 30.6 33.6 16.0 46.1 70.9 10.8 33.3 48.5 16.8 47.9 33.8
∆Search-R1 51.4 70.4 51.6 44.5 64.4 48.6 74.9 84.8 75.0 46.2 64.3 48.7 50.1 66.8 66.1
Reflective / Critique Methods
Self-RAG 50.9 69.2 54.7 19.6 42.1 38.4 10.1 41.2 68.0 25.667.549.2 21.0 41.9 67.3
Self-Refine 47.0 32.0 25.6 35.6 45.7 36.6 74.3 44.1 54.2 30.2 34.9 21.2 46.3 34.5 48.3
Align-RAG 49.7 33.3 35.4 47.131.3 43.6 71.1 42.7 41.2 31.8 36.1 34.2 41.9 36.0 47.3
CRITIC-R1 (ours) 52.2 71.0 57.5 46.3 65.4 50.6 75.3 85.0 76.3 52.066.4 51.6 50.8 67.3 69.2
Improvement over∆ +0.8 +0.6 +5.9 +1.8 +1.0 +2.0 +0.4 +0.2 +1.3 +5.8 +2.1 +2.9 +0.7 +0.5 +3.1
Table 1: Performance comparison across five benchmarks. Best results are shown inboldand second-best results
are underlined .∆denotes the criticized base RAG model.
solely on HotpotQA and evaluate it on both in-
domain (HotpotQA) and out-of-domain datasets
(other datasets). Other details are in Appendix B.
MetricsFollowing (Gao et al., 2023), we eval-
uate our method using three metrics: F1 (Wang
et al., 2024) measures the token-level overlap be-
tween the predicted answer and the ground-truth
references. SBERT (Thakur et al., 2021) computes
the cosine similarity between sentence embeddings
of the predicted and reference answers. Accuracy
(Acc) (Song et al., 2025) measures answer correct-
ness using an LLM-based judge.
BaselinesWe compare our method with a broad
set of representative baselines, including base
generation, retrieval-augmented reasoning, search-
based reasoning, and reflection- or critique-based
methods. Specifically, we include Vanilla, Naive-
RAG (Lewis et al., 2020), CoT (Wei et al., 2022),
Search-R1 (Jin et al., 2025), Self-RAG (Asai et al.,
2023), Self-Refine (Madaan et al., 2023), and
Align-RAG (Wei et al., 2025). More details about
baselines are provided in Appendix B.5.
4.2 Main Results
To answer RQ1, Table 1 reports the overall per-
formance across five QA benchmarks. CRITIC-
R1 shows consistent and substantial improvements
over the criticized base RAG model Search-R1 (de-
noted as ∆). Across the five datasets, CRITIC-
R1 improves F1, SBERT, and Acc in nearly all
settings, indicating that our critic process brings
broad gains in lexical overlap, semantic similarity,
and answer-level correctness.ModelHotpotQA NQ
Imp Harm(↓) Prec Corr Imp Harm(↓) Prec Corr
Base 2.6 23.1 48.0 9.2 2.9 4.3 61.2 11.3
CJA only 0.6 0.9 72.2 30.8 0.3 0.5 55.6 20.0
Full (CJA&DQA) 28.2 1.8 84.7 48.0 16.0 4.0 65.2 53.3
Table 2: Ablation results of CJA and DQA rewards.
Imp: corrected answers; Harm: harmful edits; Precision
(Prec): edit precision; Correct (Corr): successful edits.
The largest gains appear in answer-level accu-
racy, with an average improvement of 3.0% over
Search-R1. Together with the consistent gains in
F1 and SBERT, this shows that structured critique
improves both final correctness and answer quality,
rather than merely changing surface wording.
Overall, these results demonstrate that explicit
judgement and diagnostic feedback provide reliable
signals for RAG refinement.
4.3 Analyses
4.3.1 Rewards and Training Stages Ablations
To answer RQ2, we analyze how different reward
functions affect the refinement behavior.
Table 2 shows that the base critic frequently
causes harmful edits, indicating over-aggressive
intervention. CJA substantially reduces harm by
teaching the critic model to abstain from uncertain
edits, but it provides limited gains on actually in-
correct answers. Adding DQA leads to a higher im-
provement rate and correctness while maintaining
a low harm rate, showing that DQA complements
CJA by improving the usefulness of corrections.
To analyze how different training stages affect
6

MethodHotpotQA NQ
F1 SBERT Acc F1 SBERT Acc
w/o Critic 44.5 64.4 48.6 51.4 70.4 51.6
w/o Training 41.4 61.3 46.8 49.6 68.4 48.4
w/o DQA 47.2 66.448.8 52.570.2 53.3
Full 46.3 65.4 50.6 52.2 71.0 57.5
Table 3: Downstream QA performance with ablations
on different critic training stages.
our method, we compare the following variants:
w/o Criticdenotes the generator without critic;w/o
Traininguses the untrained base critic for refine-
ment;w/o DQAuses the CJA-trained critic;Full
denotes the complete CRITIC-R1 model with both
CJA and DQA.
Table 3 shows the results. These results verify
the necessity of both stages: CJA controls over-
aggressive refinement, while DQA improves cor-
rection quality.
4.3.2 Conservative Refinement
From Table 2, we find that avoiding harmful edits
is more crucial than aggressively correcting an-
swers. To further verify the ability of our model in
reducing harmful edits and improving successful
corrections, we show the confusion matrix for er-
ror verdict and location in Figure 4 and Figure 5,
respectively.
For the verdict, the critic exhibits a very low
false positive rate, indicating fewer unnecessary
refinements. Although this conservative policy may
leave some erroneous trajectories uncorrected, such
false negatives are generally less harmful than false
positives in refinement. The improved successful
corrections suggest that the critic learns selective
rather than passive intervention.
For the location, Figure 5 provides a fine-grained
view of where the critic assigns detected errors
within the trajectory. The matrix shows that
CRITIC-R1 can localize errors to the correspond-
ing trajectory components rather than only making
a coarse correctness judgement. Although some
errors remain among fine-grained locations, the
overall pattern suggests that the critic learns mean-
ingful localization behavior, which helps make its
feedback more actionable for refinement.
4.3.3 Comparing with Improving Generator
To answer RQ3, we compare two ways of using
the same annotated trajectories: continuously opti-
mizing the generator and training a separate critic
Incorrect Correct
Our MethodIncorrect
CorrectGround Truth0.18 0.82
0.05 0.95
0.00.20.40.60.81.0Figure 4: The confusion matrix of CRITIC-R1 for error
verdict.
search information think answer
Our Methodsearch
information
think
answerGround Truth0.84 0.00 0.11 0.05
0.00 0.95 0.02 0.03
0.04 0.04 0.80 0.12
0.01 0.01 0.08 0.89
0.00.20.40.60.81.0
Figure 5: The confusion matrix of CRITIC-R1 for error
location.
model for refinement. Since these samples have
already been seen by the generator, this setting
tests whether additional supervision can still be
extracted from previously seen data.
Table 4 summarizes the results. We find that
directly continuing generator training on already
seen data yields little to no improvement. In con-
trast, training a separate critic model on the same
data still brings measurable gains when used for
refinement. The key difference lies in the learn-
ing target. Continued generator training mainly
reinforces answer generation on familiar questions,
whereas critic training learns from the structure
of failed and successful trajectories. This makes
reused data more informative for the critic, because
the same question-answer instance can still reveal
7

MethodNQ HotpotQA
F1 SBERT Acc F1 SBERT Acc
Base Generator 51.4 70.4 51.6 44.5 64.4 48.6
Continued Generator Training 49.6 64.9 51.7 45.1 63.6 50.5
Generator + Trained Critic 52.2 71.0 57.5 46.3 65.4 50.6
Table 4: Comparison of continued generator training
and critic-based refinement with the same amount of
annotated data.
NQ
HotpotQATriviaQAASQA PopQA020406080100Percentage (%)Correct Incorrect Unsure
(a) Before training
NQ
HotpotQATriviaQAASQA PopQA020406080100Percentage (%)Correct Incorrect Unsure
(b) After training
Figure 6: Critique distributions before and after training.
Training reduces over-aggressiveINCORRECTandUN-
SUREjudgements.
useful patterns about when to intervene and how to
guide refinement.
4.3.4 Analysis of Correction Behavior
To answer RQ4, we further analyze in detail how
our training process changes the critic’s interven-
tion behavior. The results are provided in Figure 6
and Tables 5 and 6, respectively. We have the fol-
lowing observations.
Figure 6 shows that training reducesINCOR-
RECTandUNSUREpredictions while increasing
CORRECTdecisions across datasets, indicating a
more conservative policy with fewer unnecessary
interventions.
Table 5 further confirms this trend: the trained
critic substantially reduces false alarms while im-
proving the precision of error detection.
We next examine whether this selective trigger-DatasetBase Ours (∆)
Precision Recall False Alarm Precision Recall False Alarm
NQ 73.3 31.1 10.9 71.9(-1.4)17.6(-13.5)5.9(-5.0)
HotpotQA 68.8 30.1 17.0 79.1(+10.3)16.7(-13.4)5.0(-12.0)
TriviaQA 49.6 31.8 9.2 60.9(+11.3)16.2(-15.6)2.7(-6.5)
ASQA 78.1 36.6 11.9 81.9(+3.8)20.3(-16.3)4.5(-7.4)
PopQA 67.7 35.8 15.5 76.2(+8.5)17.8(-18.0)4.7(-10.8)
Avg 67.5 33.1 12.9 74.0(+6.5)17.7(-15.4)4.6(-8.3)
Table 5: Comparison of error detection behavior be-
tween the base model and the trained critic across
datasets. Values in parentheses indicate the absolute
change from the base model.
DatasetBase Critic Ours
Corr./Trig. Corr./Wrong Corr./Trig. Corr./Wrong
NQ 35.8 21.2 39.4(+3.6)22.3(+1.1)
HotpotQA 44.9 33.9 51.7(+6.8)38.9(+5.0)
TriviaQA 47.7 21.7 52.1(+4.4)26.9(+5.2)
ASQA 30.7 20.0 31.5(+0.8)20.9(+0.9)
PopQA 31.9 19.3 37.7(+5.8)18.4(-0.9)
Avg. 38.2 23.2 42.5(+4.3)25.5(+2.3)
Table 6: Comparison of correction behavior between
the base critic and the trained critic across datasets.
Corr./Trig. denotes the fraction of triggered refine-
ments that successfully correct an initially wrong an-
swer. Corr./Wrong denotes the fraction of initially
wrong predictions that are successfully corrected af-
ter critic-guided refinement.
ing leads to effective corrections when the critic is
activated. Table 6 shows that CRITIC-R1 adopts
a more effective intervention strategy: rather than
correcting aggressively, it intervenes selectively
and delivers reliable improvements when activated.
Overall, these results suggest that the gains of
CRITIC-R1 are not only reflected in final answer
accuracy, but also in the critique process: the critic
becomes more conservative in deciding when to
intervene and more effective when it does intervene.
Illustrative cases are provided in Appendix C.2.
5 Conclusion
In this paper, we propose CRITIC-R1, a structured
critic framework for error diagnosis in RAG. Our
approach formulates critique as a structured predic-
tion problem and introduces two tailored reward
functions with a two-stage RL training framework.
Experiments show consistent improvements over
baselines across multiple QA benchmarks.
Future work includes tighter critic–generator in-
tegration and extending structured critique learning
to more complex reasoning tasks.
8

Limitations
Following previous RAG settings, our experiments
are mainly conducted on QA tasks, and the ef-
fectiveness of the proposed framework on more
open-ended generation scenarios, remains to be
further explored. Besides, our method assumes a
structured trajectory format that exposes the inter-
mediate steps of RAGs. Therefore, applying it to
black-box systems requires additional adaptation.
Ethical Consideration
CRITIC-R1 relies on retrieved evidence and
language-model-based feedback, and may inherit
biases, outdated information, or factual errors from
the retrieval corpus and base models. Structured cri-
tique may also make refined answers appear more
reliable, which could increase over-trust in sensi-
tive applications. We therefore encourage transpar-
ent reporting of retrieval sources, critique proce-
dures, and evaluation settings, together with human
verification in high-stakes scenarios.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Guanting Dong, Jiajie Jin, Xiaoxi Li, Yutao Zhu,
Zhicheng Dou, and Ji-Rong Wen. 2025. Rag-critic:
Leveraging automated critic-guided agentic workflow
for retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 3551–3578.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488.
Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong
Shen, Yujiu Yang, Nan Duan, and Weizhu Chen.
2023. Critic: Large language models can self-correct
with tool-interactive critiquing.arXiv preprint
arXiv:2305.11738.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics, pages 874–880.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie
Wang, Wayne Xin Zhao, Yang Song, and Tao Zhang.
2025. Rag-star: Enhancing deliberative reasoning
with retrieval augmented verification and refinement.
InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association for
Computational Linguistics, pages 7064–7074.
Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan
Wang, Yifan Li, and Lina Yao. 2026. Doctor-rag:
Failure-aware repair for agentic retrieval-augmented
generation.arXiv preprint arXiv:2604.00865.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics, pages
1601–1611.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yu-
jia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5420–5438.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Rich James, Pedro Rodriguez, Jacob
Kahn, Gergely Szilvasy, Mike Lewis, and 1 others.
2023. Ra-dit: Retrieval-augmented dual instruction
tuning.arXiv preprint arXiv:2310.01352.
9

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
and 1 others. 2023. Self-refine: Iterative refinement
with self-feedback.Advances in neural information
processing systems, 36:46534–46594.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st annual meeting of
the association for computational linguistics, pages
9802–9822.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Yang Wu, and 1 others. 2024.
Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. Asqa: Factoid questions meet
long-form answers. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 8273–8288.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogenous benchmark for zero-shot evalua-
tion of information retrieval models.arXiv preprint
arXiv:2104.08663.
SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vip-
ula Rawte, Aman Chadha, and Amitava Das. 2024.
A comprehensive survey of hallucination mitigation
techniques in large language models.arXiv preprint
arXiv:2401.01313.
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi
Zhang, and Tyler Derr. 2024. Knowledge graph
prompting for multi-document question answering.
InProceedings of the AAAI conference on artificial
intelligence, volume 38, pages 19206–19214.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie
Qiu, Noah Wei, Jinzhe Li, Wanli Ouyang, and Siqi
Sun. 2025. Retrieval is not enough: Enhancing rag
through test-time critique and optimization. InThe
Thirty-ninth Annual Conference on Neural Informa-
tion Processing Systems.Wenda Xu, Guanglei Zhu, Xuandong Zhao, Liangming
Pan, Lei Li, and William Wang. 2024. Pride and
prejudice: Llm amplifies self-bias in self-refinement.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 15474–15492.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models.arXiv preprint arXiv:2210.03629.
Wenqi Zhang, Yongliang Shen, Linjuan Wu, Qiuying
Peng, Jun Wang, Yueting Zhuang, and Weiming Lu.
2024. Self-contrast: Better reflection through incon-
sistent solving perspectives. InProceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
3602–3622. Association for Computational Linguis-
tics.
Yujia Zhou, Zheng Liu, Jiajie Jin, Jian-Yun Nie,
and Zhicheng Dou. 2024. Metacognitive retrieval-
augmented large language models. InProceedings of
the ACM Web Conference 2024, pages 1453–1463.
10

A Method Details
A.1 Detailed Reward Design
Before the two-stage RL training, we first perform
a lightweight cold-start SFT step using the struc-
tured critique annotations. This step initializes the
critic to follow the required output schema and
produce valid fields for <verdict> ,<location> ,
<reason> , and <fix> . The cold-start stage is used
only to stabilize subsequent RL optimization, while
the critic capabilities are further optimized through
the two-stage reward design described below.
To summarize the reward computation during
RL training, we provide the two-stage gated reward
procedure in Algorithm 1.
Algorithm 1Two-Stage Gated Critique Reward
Computation
Require: Generated critique y, reference critique
y∗, training stages∈ {1,2}
Ensure:RewardR
1:Validate the structured format ofy
2:ifyviolates the required tag structurethen
3:return−γ
4:end if
5:Extract(ˆv, ˆℓ,ˆr,ˆf)fromy
6:Extract(v∗, ℓ∗, r∗, f∗)fromy∗
7:Compute format rewardr format
8:Compute verdict rewardr verdict
9:ifs= 1then▷Stage 1: CJA stage
10:Compute auxiliary penaltyr aux
11:R←r format +r verdict +r aux
12:returnR
13:end if
14:ifs= 2then▷Stage 2: DQA stage with
gated diagnostic rewards
15:R DQA←0
16:r aux←0
17:ifˆv=v∗then▷Activate diagnostic
rewards only when verdict is correct
18:Compute location rewardr loc(ˆℓ, ℓ∗)
19:Compute reason rewardr reason(ˆr, r∗)
20:Compute fix rewardr fix(ˆf, f∗)
21:R DQA←r loc+r reason +r fix
22:Compute auxiliary penaltyr aux
23:end if
24:R←r format +r verdict +R DQA+r aux
25:end if
26:returnR
Here, rauxdenotes auxiliary penalties that dis-
courage degenerated critiques, including trivialresponses, overly generic feedback, and non-
actionable fix suggestions.
A.1.1 Format Reward
To ensure that the generated critiques strictly fol-
low a predefined structure, we enforce format con-
straints on the model outputs. Each critique is re-
quired to be fully contained within a structured tem-
plate consisting of four components: <verdict> ,
<location> ,<reason> , and <fix> , each enclosed
by corresponding tags.
An example valid output is:
<verdict> INCORRECT </verdict>
<location> information:Doc3 </location>
<reason> The information does not ...
</reason> <fix> search for additional
evidence ... </fix>.
We require that all components appear exactly
once, follow a fixed order, and that no additional
content is present outside the structured tags. These
constraints ensure that the output is well-formed,
fully interpretable, and can be reliably parsed for
downstream evaluation.
The format reward is defined as:
rformat =(
α,valid format,
−γ,otherwise.(6)
where α >0 is a slight positive reward for valid
structured output, and γ >0 is a heavy penalty for
invalid format.
A.1.2 Verdict Reward
The verdict reward measures whether the critic
makes a correct judgement about the generator’s
answer, namely whether the answer is CORRECT ,
INCORRECT , orUNSURE . We use the reward matrix
in Table 7, where false alarms are penalized more
strongly than cautious uncertainty.
Ground Truth\Prediction CORRECT INCORRECT UNSURE
CORRECT 0.7 -1.0 -0.1
INCORRECT -0.3 0.5 -0.1
UNSURE 0.1 -0.2 0.0
Table 7: Verdict reward matrix in CJA reward.
This matrix is used to compute rverdict in both
stages. In Stage 2, the same verdict reward is re-
tained, while the gated DQA reward further opti-
mizes fine-grained diagnostic feedback.
11

A.1.3 Location Reward
The location reward encourages the critic to iden-
tify where an error occurs in the RAG trajec-
tory, such as reasoning ( think ), retrieved evidence
(information:Docn ), or final answer generation
(answer).
We parse each location into a tuple:
ℓ= (ℓ type, ℓidx),(7)
where ℓtypedenotes the error stage and ℓidxdenotes
the corresponding step or document index when
applicable.
Given the predicted location ˆℓand the ground-
truth locationℓ∗, we define:
rloc(ˆℓ, ℓ∗) =r type+r idx,(8)
where rtype=λ tif the location type is correctly
predicted, and 0otherwise. Similarly, ridx=λiif
the location type is correct and the corresponding
index is also matched, and0otherwise.
This reward is used only in Stage 2 and is acti-
vated only when the predicted verdict matches the
ground-truth verdict.
A.1.4 Reason Reward
The reason reward measures the lexical alignment
between the predicted reason and the reference
reason using token-level F1.
Letsreason∈[0,1] denote the token-level F1
similarity between the generated explanation and
the standard explanation. Instead of using this score
directly, we map it to a bounded reward through a
normalized exponential transformation:
rreason =Rmax
reason·exp(β rsreason)−1
exp(β r)−1,(9)
where Rmax
reason is the maximum reason reward and
βrcontrols the sharpness of the reward curve.
A.1.5 Fix Reward
The fix reward evaluates whether the predicted fix
aligns with the reference correction signal. We
combine token-level F1 and keyword coverage:
sfix=α sF1
fix+ (1−α)skw
fix,(10)
where α∈[0,1] controls the trade-off between
lexical similarity and keyword coverage.
The final fix reward is obtained through the same
normalized exponential transformation:
rfix=Rmax
fix·exp(β fsfix)−1
exp(β f)−1,(11)where Rmax
fixis the maximum fix reward and βf
controls reward sharpness.
Generic or non-actionable fix suggestions are
treated as degenerated critiques and penalized
throughr aux.
A.1.6 Auxiliary Penalty
We introduce an auxiliary penalty term rauxto dis-
courage degenerated critiques. It is applied when
the expected diagnostic fields are missing, overly
short, or filled with generic non-actionable feed-
back, such as simply suggesting to “search again”
without specifying what evidence or constraint
should be checked.
This penalty encourages the critic to provide
meaningful diagnostic information rather than vac-
uous outputs.
A.2 Structured Critique Supervision
Construction
To improve the stability and quality of process-
level supervision, we construct structured critique
annotations through a consensus-based LLM-as-
judge framework.
Rather than relying on a single teacher response,
we sample multiple candidate critiques for each
trajectory and aggregate them through consensus
voting and quality-aware selection. This design re-
duces annotation noise, improves the reliability of
judgement signals, and stabilizes fine-grained diag-
nostic supervision for error localization, reasoning,
and correction.
The overall supervision construction procedure
is summarized in Algorithm 2.
A.3 Prompt Templates for CRITIC-R1
To improve reproducibility, we provide the prompt
templates used in our LLM-as-judge framework.
The following Figure 7 summarizes the two com-
ponents of the prompting setup: (a) the system
prompt, which specifies the critic role, grounding
rules, and evaluation policy; and (b) the instruc-
tion prompt, which defines the structured critique
output format, label space, and output constraints.
After generating the critic outputs, we further
use a refinement prompt to guide the generator in
selectively incorporating external critiques during
answer regeneration. Unlike standard feedback-
following prompts, the critique is treated as a verifi-
able hint rather than oracle feedback: the generator
is instructed to independently re-solve the ques-
tion, verify critique suggestions through reasoning
12

Algorithm 2Consensus-Based Structured Critique
Supervision
Require: Question q, trajectory τ, reference an-
swer a∗, retrieved context c, number of sam-
plesK
Ensure:Consensus supervision critiquey∗
1:fori= 1toKdo
2:Sample critique
yi∼p judge(y|q, τ, a∗, c)
3: ifyiviolates structured format constraints
then
4:Attempt format recovery
5:end if
6:ify iremains invalidthen
7: Replace with fallback parse-failure cri-
tique
8:end if
9:end for
10:Extract verdicts
{vi}K
i=1
11:Compute consensus verdict
v∗= arg max
vX
i1(vi=v)
12:Form consensus candidate set
C={y i|vi=v∗}
13:foreachy i∈ Cdo
14: Compute structural-diagnostic quality
score
S(yi) =s keyword +s reason +s fix+s location
15:end for
16:Select highest-quality consensus supervision
y∗= arg max
yi∈CS(yi)
17:ifkeywords missingthen
18: Recover keywords from supporting context
19:end if
20:returny∗
System Prompt in CRITIC -R1
You are an external critic for a retrieval -augmented QA system.
You audit a trajectory produced by another QA model.
The trajectory may contain reasoning, search steps, retrieved information, and a 
final answer
IMPORTANT RULES**
You MUST use <information> as the ONLY source of knowledge
DO NOT use your own parametric knowledge or world knowledge
If the <information> does not contradict the answer, you should 
assume the answer is acceptable
Do NOT invent missing facts from your own knowledge
Evaluation policy**
If the reasoning and answer are consistent with <information>, judge CORRECT
Only judge INCORRECT if there is a clear contradiction or a clear entity mismatch
If the evidence is insufficient but not contradictory, judge CORRECT or UNSURE(a) System Prompt for Structured Critique Supervision
CRITIC_INSTRUCTION
Input: - QUESTION - TRAJECTORY with tags: <think> <search> <information> <answer>
Output (STRICT, one line, in this exact 
order):<verdict>...</verdict><location>...</location><reason>...</reason><fix>...</fix>
Allowed values: - verdict: CORRECT | INCORRECT | UNSURE - location: none | answer | 
information:DocK  | search:stepK  | think:stepK
Constraints:
Output MUST start with "<verdict>" (no leading text). - Keep <reason> <= 25 words
Keep <fix> <= 20 words. - If verdict is CORRECT: location must be "none" and fix must 
be "keep" . - If verdict is INCORRECT or UNSURE: location must NOT be "none"
If there is no clear contradiction with <information>, prefer CORRECT.
Example (format only):
<verdict>INCORRECT</verdict><location>information:Doc1</location><reason>Doc1 
contradicts the final answer.</reason><fix>search: retrieve correct entity</fix>
(b) Instruction Prompt for Structured Critique Output
Figure 7: Prompt templates used in the structured cri-
tique supervision framework. The system prompt de-
fines critic behavior and evidence-grounded judgement
principles, while the instruction prompt specifies struc-
tured critique outputs and formatting constraints.
and retrieval, and ignore unsupported or mistaken
critiques.
The refinement prompt is governed by the fol-
lowing principles:
You are also given:
1. A previous trajectory from an earlier
attempt.
2. An external critique of that previous
trajectory.
Important rules:
- The previous trajectory may contain
mistakes.
- The previous final answer may be wrong.
- The external critique may also be
wrong.
- Do NOT blindly trust the previous
trajectory.
- Do NOT blindly trust the critique.
- Use the critique only as a hint about
possible problems to check.
- Re-solve the question with fresh
reasoning instead of simply copying the
previous answer.
- If the critique points out a possible
issue, verify it by your own reasoning
and search.
- If the critique is unsupported or
mistaken, ignore it.
13

- Do not change your answer just because
the critique suggests a change.
- Base your final answer on your own
reasoning process and the retrieved
information.
- You MUST end with exactly one final
answer inside <answer> and </answer>.
This refinement design is important for robust
critic–generator interaction, as it encourages selec-
tive use of critiques rather than blind correction.
B Experimental Setup
This section provides implementation details for
reproducing our experiments. Unless otherwise
specified, all methods are evaluated under the same
model scale, retrieval setting, decoding configura-
tion, answer extraction procedure, and evaluation
protocol.
B.1 Unified Model and Data Configuration
We use a unified model and dataset config-
uration for fair comparison. The generator
is a Search-R1-trained checkpoint based on
Qwen2.5-7B-Instruct , and the critic is initialized
from Qwen2.5-3B-Instruct . For semantic simi-
larity evaluation, we use all-MiniLM-L6-v2 as the
SBERT model. The critic is trained only on Hot-
potQA and evaluated on HotpotQA, NQ, TriviaQA,
ASQA, and PopQA. For baselines involving cri-
tique or feedback, we use the same critic backbone.
B.2 Retrieval Configuration
Item Configuration
Retrieval corpusHotpotWiki18
RetrieverE5-base-v2
Index typeFAISS Flat
Retrieval depth Top-5
Retrieval calls Single retrieval step
Query construction Original question field
Table 8: Retrieval configuration used in all retrieval-
based methods.
For most datasets, we use the original question as
the retrieval query. For datasets with special input
fields, such as ASQA, we use [sub_question] as
the retrieval query. No dataset-specific retrieval
depth or retriever tuning is applied.
B.3 Training Setup
The critic is trained only on HotpotQA. We use
5,000 training examples with structured critique
annotations. Training follows the two-stage proce-
dure described in Section 3.Item Configuration
Training data HotpotQA training subset
Training size 5,000
Base checkpointQwen2.5-3B-Instruct
Stage-2 initialization Stage-1 LoRA adapter
Fine-tuning method LoRA
LoRA rank 16
LoRA alpha 16
Target modules All linear layers
Learning rate1×10−6
Train batch size 4
mini-batch size 4
Micro-batch size per GPU 1
Number of epochs 5
KL loss coefficient 0.003
Max prompt length 3192
Max response length 256
Number of GPUs 2
Table 9: Training configuration for the critic model.
Before the two-stage RL training, we perform
a lightweight cold-start SFT step on these annota-
tions to initialize the critic with the required struc-
tured output format.
In Stage 1, the model is trained to produce well-
formed structured critiques and reliable high-level
verdicts. In Stage 2, diagnostic rewards are intro-
duced to improve error localization, reasoning anal-
ysis, and fix generation. Diagnostic rewards are
activated only when the predicted verdict matches
the reference verdict.
The total computational budget is roughly es-
timated to be within 200–250 GPU hours, with
variations depending on the GPU type, runtime
environment, and hardware utilization.
B.4 Inference Procedure
During inference, we use a single-search tra-
jectory generation procedure. The generator
produces reasoning within <think> tags, issues
at most one retrieval query through <search>
tags, receives the top- 5retrieved passages within
<information> tags, and generates the final an-
swer within <answer> tags. The final prediction is
extracted from the last<answer>span.
Item Value
Max search calls 1
Retrieved documents 5
Temperature 0.7
Top-p0.9
Repetition penalty 1.1
Max generation length 1024
Table 10: Inference hyperparameters.
14

B.5 Baseline Implementation
We implement or adapt all baselines under the same
model scale, retrieval setup, and evaluation proto-
col whenever applicable. For retrieval-based base-
lines, the same retrieved documents are used. For
critique- or refinement-based baselines, the same
critic backbone is used unless otherwise specified.
•Vanilla: The base language model without re-
trieval, directly generating answers from the
question.
•Naive RAG (Lewis et al., 2020): Retrieved
documents are directly concatenated with the
question as input, without additional prompt
engineering.
•CoT (Wei et al., 2022): Augments the genera-
tion process with intermediate reasoning steps,
enabling the model to perform multi-step rea-
soning before producing the final answer.
•Self-RAG (Asai et al., 2023): Introduces self-
reflection and iterative retrieval to refine inter-
mediate reasoning steps during generation.
•Self-Refine (Madaan et al., 2023): Performs
iterative answer refinement by generating feed-
back and improving responses over multiple
rounds.
•Align-RAG (Wei et al., 2025): Enhances RAG
through alignment-based optimization to im-
prove consistency between retrieved evidence
and generated answers.
•Search-R1 (Jin et al., 2025): A strong baseline
that integrates structured search and reasoning
trajectories for improved retrieval and genera-
tion.
•CRITIC-R1: Our proposed method, which in-
troduces an external critic to provide structured
feedback for improving answer correctness and
reasoning quality.
Implementation Details.To ensure a fair and
reproducible comparison, we use a unified exper-
imental setup across all methods. All generation-
based methods use Qwen2.5-7B-Instruct as the
backbone model, and all critique- or feedback-
based methods use Qwen2.5-3B-Instruct as the
critic model. For retrieval-based methods, we use
the same Wikipedia-based corpus, retriever, and in-
dex, with a single retrieval call and top- 5retrieveddocuments for each question. We do not tune re-
trieval depth or retrieval configuration separately
for different datasets. For baselines that require
critique or refinement, such as Align-RAG, we im-
plement them under the same model scale, retrieval
setting, prompt format, and inference pipeline as
our method whenever applicable. Finally, all meth-
ods are evaluated using the same decoding config-
uration, answer extraction procedure, evaluation
metrics, and LLM-based judging protocol.
B.6 Hardware and Environment
Critic training is conducted on NVIDIA A100
40GB GPUs, while inference and evaluation are
conducted on NVIDIA V100 32GB GPUs.
Item Version / Configuration
Python 3.12.0
PyTorch 2.8.0+cu128
CUDA 12.8
Transformers 4.56.1
vLLM 0.15.1
FAISS 1.8.0
Table 11: Main software dependencies.
B.7 Dataset Statistics
Dataset Split Usage #Examples
HotpotQA Train Critic training 5,000/90447
HotpotQA Dev In-domain 7,405/7,405
NQ Dev Out-of-domain 3,610/3,610
TriviaQA Validation Out-of-domain 17,944/17,944
ASQA Dev Out-of-domain 948/948
PopQA Test Out-of-domain 14,267/14,267
Table 12: Dataset statistics used in our experiments.
C Additional Experimental Results
C.1 Detailed Per-Run Results for Efficiency
Comparison
To complement the averaged results reported in
Table 4, we provide detailed per-run results for the
efficiency comparison in this appendix.
For Continued Generator Training, results are
reported from two independent inference runs on
both HotpotQA and NQ, and the main-text results
are obtained by averaging these runs. For Genera-
tor + Trained Critic, we report the corresponding
inference results used in the main table.
These detailed results help illustrate the stability
of continued generator training and provide addi-
15

tional transparency for the efficiency comparison
under equal supervision budget.
Method F1 SBERT Acc
Continued Generator Training (Run 1) 52.8 67.7 51.9
Continued Generator Training (Run 2) 46.4 62.1 51.4
Average 49.6 64.9 51.7
Generator + Trained Critic 52.2 71.0 57.5
Table 13: Detailed per-run results for efficiency compar-
ison on NQ.
Method F1 SBERT Acc
Continued Generator Training (Run 1) 41.4 62.4 49.6
Continued Generator Training (Run 2) 48.8 64.8 51.4
Average 45.1 63.6 50.5
Generator + Trained Critic 46.3 65.4 50.6
Table 14: Detailed per-run results for efficiency compar-
ison on HotpotQA.
C.2 Case Studies for CRITIC-R1
We provide representative examples to illustrate
how CRITIC-R1 supports refinement. Rather than
presenting full trajectories, we summarize the key
evidence, critique signal, and refinement outcome
for readability.
C.3 Case A: Critic-Guided Error Correction
The following example illustrates how CRITIC-
R1 identifies an error in the initial trajectory and
provides critique that leads to a corrected version.
In which year was the King who made the 1925 Birthday Honours  
born?
<think>I need to find out in which year the King who made the 1925 
Birthday Honours  was born. I'll search for it.</think>
<search>In which year was the King who made the 1925 Birthday 
Honours  born</search>
<information>Doc 1(Title: ''1925 Birthday Honours '') The 1925 Birthday 
Honours  were appointments by King George V ... published in ''The 
London Gazette'' on 3 June 1925...</information>
<answer>1964</answer>Question
Initial Trajectory
<verdict>INCORRECT</verdict><location>answer</location>
<reason>The trajectory identifies King George V as the relevant 
king but does not verify his birth year. The final answer is 
unsupported by the retrieved evidence.</reason>
<fix>Search for King George V's birth year and revise the 
answer based on direct evidence.</fix>Structured Critique
The critique accurately marks the trajectory as incorrect, localizes the error 
to the answer stage, and specifies the missing evidence needed for 
refinement.The initial trajectory retrieves the key evidence that the 1925 Birthday 
Honours  were made by King George V, but it fails to verify George V's birth 
year and outputs an unsupported answer.
Refined Trajectory
refinement.
<think>The critique indicates that the relevant king is King George V, but 
his birth year was not verified. I need to search for direct evidence about 
King George V's birth year.</think>
<search>King George V birth year</search>
<information>Doc 1(Title: ``George V'') George V ... 3 June 1865 -- 20 
January 1936 ... \\Doc 2(Title: ``George V'') George was born on 3 June 
1865, in Marlborough House, London. \\...</information>
<answer>1865</answer>Refined Trajectory
Guided by the structured critique, the model performs a targeted verification 
step and retrieves direct evidence that King George V was born in 
1865.The refinement corrects the unsupported initial answer and produces 
the final answer “1865”.C.4 Case B: Selective Non-Intervention
The following example illustrates how CRITIC-R1
recognizes that the initial trajectory has no obvi-
ous error and therefore chooses not to intervene,
thereby avoiding over-aggressive intervention on
an already correct answer.
Scott Parkin has been a vocal critic of ExxonMobil and another 
corporation that has operations in how many countries?
<think>I need to find out the other corporation criticized by Scott Parkin 
and how many countries it operates in. I'll search for it.</think>
<search>Scott Parkin critic of ExxonMobil another corporation 
operations in how many countries</search>
<information>Doc 1(Title: ``Scott Parkin'') Scott Parkin ... has been a 
vocal critic of ... corporations such as ExxonMobil and Halliburton. \\Doc 
5(Title: ``Halliburton'') The Halliburton Company ... has operations in 
more than 70 countries. \\...</information>
<answer>70</answer>Question
Initial Trajectory
<verdict>UNSURE</verdict><location>none</location>
<reason>The retrieved evidence supports the identified 
corporation and indicates that Halliburton operates in more than 
70 countries. There is no strong contradictory evidence requiring 
correction.</reason>
<fix>Preserve the answer unless additional evidence clearly 
contradicts it.</fix>Structured Critique
The critique avoids over -aggressive intervention by not forcing a revision 
when the retrieved evidence already supports the answer.This  reflects a 
conservative decision policy: uncertain cases are not automatically treated as 
errors.The initial trajectory correctly identifies Halliburton as the other corporation 
and retrieves evidence that it has operations in more than 70 
countries.Although  the answer is slightly abbreviated, it is supported by the 
retrieved evidence.
<think>The critique does not identify a clear error. The retrieved evidence 
already links Scott Parkin to Halliburton and states that Halliburton 
operates in more than 70 countries, so I should preserve the supported 
answer.</think>
<answer>70</answer>Refined Trajectory
Guided by the conservative critique, the model keeps the evidence -
supported answer rather than making an unnecessary change.
C.5 Case C: Critique-Guided Evidence
Selection Repair
The following example illustrates how CRITIC-
R1 identifies an evidence-selection error in the ini-
tial trajectory and provides critique that guides the
model to select the evidence matching the question
constraint.
16

The author of  Sexual Politics attended which British University?
<think>I need to find out which British University the author of Sexual 
Politics attended. I'll search for it.</think>
<search>The author of Sexual Politics attended which British 
University</search>
<information>Doc 1(Title: ''Sexual Politics'') Sexual Politics is a 1970 
book by Kate Millett, based on her PhD dissertation. \\Doc 2(Title: ''Kate 
Millett'') Kate Millett ... attended Oxford University ... and is best known 
for her book Sexual Politics, which was based on her doctoral 
dissertation at Columbia University...</information>
<answer>Columbia University</answer>Question
Initial Trajectory
<verdict>INCORRECT</verdict><location>answer</location>
<reason>The answer is not directly supported by the retrieved 
evidence; the trajectory skips the evidential step needed to 
justify this entity.</reason>
<fix>Revisit the retrieved documents and revise the answer only 
according to explicitly supported evidence.</fix>Structured Critique
The critique localizes the error to the answer stage and identifies the 
specific evidence -selection mismatch.It  guides the model to distinguish the 
dissertation institution from the university required by the question.The initial trajectory retrieves the correct author, Kate Millett, and the 
relevant evidence.However , it selects Columbia University from the 
dissertation context instead of the British university she attended.
<think>The author is Kate Millett. The question asks for the British 
university she attended, not the university where her dissertation was 
completed.</think >
<search>The author of Sexual Politics attended which British 
University</search >
<information>Doc 2(Title: ''Kate Millett'') Kate Millett attended Oxford 
University ... Sexual Politics was based on her doctoral dissertation at 
Columbia University.</information >
<answer>Oxford University</answer>Refined Trajectory
Guided by the structured critique, the model performs a targeted verification 
step and selects the evidence matching the question constraint.The  
refinement repairs the evidence -selection error and changes the answer 
from ''Columbia University'' to ''Oxford University''.D Additional Discussions
D.1 Artifacts and Licenses
We will release our code, prompts, and evaluation
scripts for research purposes. All datasets used in
this work are publicly available, and their use fol-
lows the corresponding dataset licenses and terms.
Our use of these datasets and pretrained models is
limited to research-oriented training, evaluation,
and analysis, which is consistent with their in-
tended research use. Any derived supervision data
or model artifacts will be released only when per-
mitted by the licenses and terms of the original
datasets and base models.
D.2 Potential Risks
Our critic may still produce incorrect diagnoses, es-
pecially when retrieved evidence is incomplete or
misleading. Since the supervision relies on LLM-
as-judge annotations, it may also inherit biases or
errors from the judge model. The method should
therefore not be treated as a guarantee of factual
correctness, and human verification remains neces-
sary in high-stakes applications.D.3 Data Privacy and Content
We use publicly available QA benchmarks and do
not collect new data from human participants or
private user interactions. We do not intentionally
include personally identifying information, and any
released artifacts will exclude private metadata, lo-
cal paths, API keys, or other identifiers.
D.4 Use of AI Assistants
We used AI assistants to assist with language pol-
ishing, grammar checking, code debugging, and
improving clarity of presentation.
17