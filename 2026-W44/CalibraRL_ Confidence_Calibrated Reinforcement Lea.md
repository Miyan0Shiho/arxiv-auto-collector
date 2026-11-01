# CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG

**Authors**: Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, Zhiming Zheng

**Published**: 2026-10-26 10:11:24

**PDF URL**: [https://arxiv.org/pdf/2610.35142v1](https://arxiv.org/pdf/2610.35142v1)

## Abstract
Large language models (LLMs) often produce plausible but incorrect answers, motivating reinforcement learning methods that reward correct answers, penalize incorrect answers, and allow abstention. However, fixed outcome-based rewards do not distinguish incorrect responses with different confidence levels or account for differences in answering reliability across queries. Consequently, such rewards may inadequately penalize overconfident errors and fail to discourage unnecessary abstention. We propose CalibraRL, a reinforcement learning framework that incorporates response confidence and query-level answering evidence into reward design. For non-abstaining responses, CalibraRL combines correctness-based utility with a Brier-based penalty that encourages agreement between confidence and correctness. For abstentions, it estimates conditional answering reliability using a Beta posterior derived from the correctness of non-abstaining responses within each sampled group. The abstention reward increases with the posterior probability that the model's accuracy on non-abstaining responses is below a predefined threshold. CalibraRL reuses training rollouts without requiring additional annotations indicating whether the model knows the answer to each question. Experiments on three QA datasets demonstrate that CalibraRL improves answer precision and reduces hallucinations compared with strong baselines. Further analysis shows that CalibraRL reduces mean expected calibration error (ECE) by 30.8\% across two backbone models and three datasets, highlighting its potential to improve the reliability of retrieval-augmented generation (RAG) in web search.

## Full Text


<!-- PDF content starts -->

CalibraRL: Confidence-Calibrated Reinforcement Learning for
Mitigating Hallucinations in RAG
Qianchi Zhang∗
School of Artificial Intelligence,
Beihang University
Beijing, China
zhangqianchi@buaa.edu.cnHainan Zhang†
School of Artificial Intelligence,
Beihang University
Beijing, China
zhanghainan@buaa.edu.cnLiang Pang
Institute of Computing Technology,
Chinese Academy of Sciences
Beijing, China
pangliang@ict.ac.cn
Hongwei Zheng
Beijing Academy of Blockchain and
Edge Computing
Beijing, ChinaZhiming Zheng
School of Artificial Intelligence,
Beihang University
Beijing, China
Abstract
Large language models (LLMs) often produce plausible but incorrect
answers, motivating reinforcement learning methods that reward
correct answers, penalize incorrect answers, and allow abstention.
However, fixed outcome-based rewards do not distinguish incorrect
responses with different confidence levels or account for differ-
ences in answering reliability across queries. Consequently, such
rewards may inadequately penalize overconfident errors and fail
to discourage unnecessary abstention. We proposeCalibraRL, a
reinforcement learning framework that incorporates response con-
fidence and query-level answering evidence into reward design. For
non-abstaining responses, CalibraRL combines correctness-based
utility with a Brier-based penalty that encourages agreement be-
tween confidence and correctness. For abstentions, it estimates
conditional answering reliability using a Beta posterior derived
from the correctness of non-abstaining responses within each sam-
pled group. The abstention reward increases with the posterior
probability that the model’s accuracy on non-abstaining responses
is below a predefined threshold. CalibraRL reuses training rollouts
without requiring additional annotations indicating whether the
model knows the answer to each question. Experiments on three
QA datasets demonstrate that CalibraRL improves answer preci-
sion and reduces hallucinations compared with strong baselines.
Further analysis shows that CalibraRL reduces mean expected cal-
ibration error (ECE) by 30.8% across two backbone models and
three datasets, highlighting its potential to improve the reliability
of retrieval-augmented generation (RAG) in web search.
1 Introduction
Large language models (LLMs) have become increasingly capable of
answering knowledge-intensive questions, yet they remain prone
to producing plausible but factually incorrect responses [ 14,16].
Providing external documents can improve access to relevant infor-
mation, but does not guarantee that a model will use the evidence
correctly or recognize when it cannot answer reliably [ 18,24]. Re-
liable question answering therefore requires balancing accurate
answering with appropriate abstention, as insufficient abstention
∗This work was supported by Beijing Advanced Innovation Center for Future
Blockchain and Privacy Computing.
†Corresponding author.
Same Outcome, Different Reliability
(A) Same Error, Different Confidence
Two incorrect answers
Low confidenceHigh confidence
“Maybe...” “Definitely!”Fixed: equal error penalty
Low-confidence error → -1
High-confidence error → -1
Ours: confidence penalty
Low-confidence error smaller−
High-confidence errorlarger− − −
(B) Same IDK, Different Sampling Outcomes
Two queries, both answered “IDK”
Other attempts: mostly wrong
Other attempts: mostly correctFixed: same reward for every IDK
Other attempts mostly wrong → 0
Other attempts mostly correct→ 0
Ours: posterior IDK reward
Mostly wrong reward IDK (+)
Mostly correct penalize IDK (-)Figure 1: Reward design motivation. IDK denotes absten-
tion. (A) Unlike fixed penalties, our method penalizes higher-
confidence errors more strongly. (B) Unlike fixed abstention
rewards, our posterior-adaptive reward reflects the correct-
ness of non-abstaining rollouts for the same query.
exposes users to incorrect information, whereas excessive absten-
tion withholds answers that the model could otherwise provide
correctly with the available evidence.
Recent studies have explored incorporating abstention into model
training [ 38,48]. R-Tuning constructs refusal-aware instruction
data to teach models to answer questions within their estimated
capabilities and acknowledge uncertainty otherwise [ 48]. TruthRL
adopts a ternary reward that rewards correct answers, penalizes
incorrect answers, and assigns a neutral reward to abstentions [ 38].
This design provides a learning signal for the trade-off between
answering and abstaining. However, fixed ternary rewards assign
raw reward values solely according to the final outcome category,
without explicitly accounting for reliability differences within that
category. Our central observation is that the same outcome category
can correspond to different reliability characteristics. As shown in
Figure 1, two incorrect responses may have substantially different
1
arXiv:2610.35142v1  [cs.CL]  26 Oct 2026

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
generation confidence. Although both require correction, a high-
confidence error exhibits a larger discrepancy between confidence
and correctness. Likewise, abstentions on different queries may
be supported by different degrees of answering evidence. For one
query, most of the other sampled non-abstaining responses may be
incorrect, whereas for another, most may be correct. The former
provides greater empirical support for abstention, while the lat-
ter suggests a potentially missed opportunity to answer correctly.
These distinctions motivate training feedback that accounts not
only for the observed outcome, but also for response confidence
and answering evidence from the same query.
Our preliminary experiments support this motivation. First, em-
pirical answer accuracy generally increases with generation confi-
dence, indicating that confidence contains information about an-
swer reliability. Nevertheless, incorrect responses span a broad con-
fidence range, including a substantial proportion of high-confidence
errors. These observations motivate examining confidence along-
side correctness when designing training rewards. Second, among
queries that elicit abstention, independent reference sampling re-
veals variation in correctness rates conditional on producing a
non-abstaining response. The same abstention label therefore en-
compasses different empirical answering opportunities. Together,
these observations motivate investigating whether confidence and
answering evidence can provide useful training signals beyond
outcome labels.
Motivated by these observations, we proposeCalibraRL, a re-
inforcement learning framework combining confidence-aware an-
swer rewards with posterior-adaptive abstention rewards. For non-
abstaining responses, CalibraRL augments a signed correctness
reward with a Brier-based penalty [ 5,10]. It uses generation con-
fidence derived from token probabilities to capture disagreement
between confidence and correctness, imposing a larger additional
penalty on high-confidence errors. For abstentions, CalibraRL con-
structs a Bayesian posterior over conditional answer accuracy from
the correct and incorrect non-abstaining responses in the same roll-
out group. The posterior probability that this accuracy falls below
a specified threshold determines the abstention reward, accounting
for both observed outcomes and estimation uncertainty under finite
sampling. Both components are incorporated into the raw rewards
used to optimize the policy through Group Relative Policy Opti-
mization (GRPO) [ 34]. The framework reuses training rollouts and
reference answers for correctness verification, without requiring
additional, precomputed labels of whether each query is known or
unknown to the model.
Experiments on three QA datasets demonstrate that CalibraRL
improves answer precision and reduces hallucinations compared
with strong baselines. Further analysis shows that CalibraRL re-
duces mean expected calibration error (ECE) by 30.8% across two
backbone models and three datasets, highlighting its potential to
improve the reliability of retrieval-augmented generation (RAG) in
web search.
Our main contributions are as follows:
•We identify reliability differences within outcome categories
through empirical analyses of confidence among incorrect
responses and the accuracy of non-abstaining responses ob-
tained through independent reference sampling for queries
eliciting abstention.•We propose CalibraRL, which combines a confidence-aware
answer reward with a Bayesian posterior-adaptive absten-
tion reward derived from same-query rollout evidence.
•Across three QA datasets and two backbones, we demon-
strate improvements in answer precision and overall truth-
fulness over strong baselines, and analyze the contributions
of both reward components through ablation studies.
2 Related Work
2.1 LLM Hallucinations
LLMs may generate fluent but factually incorrect or contextually un-
supported responses [ 32]. Hallucinations can involve violations of
factuality or faithfulness [ 14], while evaluations on TruthfulQA
show that models may reproduce common human misconcep-
tions [ 25]. SelfCheckGPT [ 26] detects inconsistencies across sam-
pled responses, whereas semantic entropy [ 7] measures uncertainty
over semantically distinct answers. However, consistency across
responses does not guarantee factual correctness.
Existing mitigation methods improve factual reliability through
retrieval augmentation [ 2,17,22,37,50,54], response verifica-
tion [ 6], contrastive decoding [ 3], and factuality-oriented train-
ing. Self-Alignment for Factuality [ 52] derives factuality-based
preferences for Direct Preference Optimization (DPO) [ 31], while
UAlign [ 46] incorporates confidence estimates and semantic en-
tropy into reinforcement learning. Another line of research mit-
igates hallucinations by enabling models to abstain when they
cannot answer reliably. R-Tuning [ 48] constructs refusal-aware
training data based on knowledge gaps, while RLKF [ 23] derives
reinforcement learning feedback through knowledge-state probing
and consistency checking. TruthRL [ 38] employs Group Relative
Policy Optimization (GRPO) [ 34], assigning positive rewards to cor-
rect answers, negative rewards to hallucinations, and zero rewards
to abstentions. Together, these studies highlight the importance of
improving factual reliability while enabling appropriate abstention.
2.2 Knowledge Boundary of LLMs
Knowledge-boundary research examines what models know [ 29]
and whether they recognize their knowledge limits [ 20]. Prior
work [ 20] distinguishes response correctness probability, 𝑃(True) ,
from the probability of knowing an answer, 𝑃(IK) , providing ev-
idence of models’ self-assessment capabilities. However, models
often express overconfidence, and reliable confidence estimation
remains challenging on difficult tasks [ 43]. Models also learn previ-
ously unknown facts more slowly than knowledge-consistent ex-
amples during fine-tuning, potentially increasing hallucinations [ 9].
Self-evaluation research further suggests that models may fail to
reliably express knowledge they possess [ 52], while knowledge-
state probing examines their ability to recognize and use internal
knowledge states [23, 27].
These observations distinguish knowledge availability from re-
liable expression: a single incorrect response does not necessarily
indicate missing knowledge. A central challenge for abstention
is therefore to translate answerability estimates into reliable an-
swering or abstention decisions without unnecessarily withholding
correct answers.
2

CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG Conference’17, July 2017, Washington, DC, USA
0.0 0.2 0.4 0.6 0.8 1.0
Confidence0.00.20.40.60.81.0Accuracy
ECE = 0.318Perfect calibration
Empirical accuracy
0.0 0.2 0.4 0.6 0.8 1.0
Confidence015003000450060007500Incorrect Responseγ=0.8Lower confidence
High confidence
0.0 0.2 0.4 0.6 0.8 1.0
Reference Answer Accuracy025050075010001250Question Countτ=0.5Below threshold
Potential over-IDK
(A) Confidence Calibration (B) Confidence of Errors (C) Answerability of IDK Questions
Figure 2: Empirical motivation using LLaMA3-8B-Instruct on NQ. (A) Empirical accuracy generally increases with confidence. (B)
Incorrect responses exhibit heterogeneous confidence. (C) Questions eliciting abstention differ in the accuracy of non-abstaining
responses obtained through independent reference sampling. IDK denotes“I don’t know”, indicating abstention. Reference
groups containing only abstentions are excluded from (C).
3 Preliminaries
3.1 Problem Formulation
We consider question answering with abstention, where a language
model answers using provided documents or abstains. Let D=
{(𝑥𝑗,A𝑗)}𝑁
𝑗=1, where𝑥𝑗=(𝑞𝑗,𝑑𝑗)contains a question and its asso-
ciated documents, and A𝑗is the set of acceptable reference answers.
Given𝑥, an autoregressive policy 𝜋𝜃generates𝑜=(𝑜 1,...,𝑜𝑇)and
is prompted to outputI don’t know(IDK) when uncertain.
Let𝑎(𝑜)∈{ 0,1}indicate abstention and 𝑦(𝑜,A)∈{ 0,1}denote
the correctness of a non-abstaining response, assessed against A.
Responses are categorized as correct answers, incorrect answers,
or abstentions. Our objective is to encourage correct answers and
appropriate abstention while discouraging incorrect answers and
unnecessary abstention.
Following prior work [ 28,36], we use the mean probability of
generated tokens as a proxy for response confidence. For a non-
abstaining response:
𝑐𝜃(𝑥,𝑜)=1
𝑇𝑇∑︁
𝑡=1𝜋𝜃(𝑜𝑡|𝑥,𝑜 <𝑡),(1)
where𝑇is the response length and 𝑜<𝑡denotes preceding tokens.
This measure requires no additional prompting or multiple response
samples.
3.2 Motivation
We motivate our reward design through two questions: ➊Does
model confidence align with answer correctness? ➋Does the model
abstain on questions it can answer correctly?
Figure 2(A) shows that accuracy tends to increases with confi-
dence but generally lies below the ideal calibration line, with an ECE
of0.318. Thus, confidence is informative about correctness but ex-
hibits substantial overconfidence. Figure 2(B) shows that incorrect
responses span different confidence levels, including many above
0.8. A correctness-only reward treats these errors equally, failing
to distinguish low-confidence errors from confidently incorrectresponses. These observations motivate incorporating confidence–
correctness alignment into the answer reward.
We next examine reference answering performance for questions
that elicit IDK responses. For each question, reference accuracy
is the proportion of correct responses among its non-IDK refer-
ence responses. All-IDK reference groups are excluded because this
conditional accuracy is undefined. Figure 2(C) shows substantial
variation: many questions have low reference accuracy, but some
exceed0.5and others approach1. Thus, an IDK response does not
necessarily indicate consistently unsuccessful reference answering.
High reference accuracy suggests potentially unnecessary absten-
tion, motivating an abstention reward informed by query-level
answering evidence.
These findings motivate two reward-design principles: ➊pe-
nalizing discrepancies between response confidence and correctness;
➋adapting abstention rewards to query-level answering performance.
4 Methodology
As shown in Figure 3, we sample a group of responses per query,
identify abstentions, and evaluate the correctness of answered re-
sponses. For answered responses, we combine correctness utility
with a confidence-based Brier penalty. For abstentions, we derive
rewards from a Bayesian reliability estimate based on the group’s
correct and incorrect answers. Finally, the resulting rewards are
normalized within each group and used for GRPO optimization.
4.1 Confidence-Aware Answer Reward
For each input 𝑥, we sample a response group G𝑥={𝑜𝑖}𝐺
𝑖=1. For
a non-abstaining response 𝑜𝑖, let𝑦𝑖=𝑦(𝑜𝑖,A) denote its correct-
ness and𝑐𝑖denote its confidence computed according to (1). A
correctness-only reward does not distinguish between responses
with the same correctness label but different confidence levels. We
therefore define the answer reward as:
𝑟ans(𝑜𝑖,𝑥)=2𝑦𝑖−1−𝜆𝑐(𝑐𝑖−𝑦𝑖)2,(2)
where𝜆𝑐≥0controls the strength of the confidence penalty.
3

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
A. Sample and VerifyB. Confidence-aware Answer Branch
C. Posterior-adaptive Abstention BranchD. Policy Optimization
Query
Documents
⋯
Policy
G responses
IDK
Verify answers
Correct: C
Incorrect: W
Abstain: IDKAnswered response
Token confidence
 Answer reward
correct
incorrectBeta posterior
 IDK reward
Shared by all IDK responses
in the groupAssign rewards
Group normalization
GRPO update
Inference
Query
Policy
Answer / IDK
Figure 3: Overview of our proposed framework. IDK denotes“I don’t know”, indicating abstention.
The correctness utility2 𝑦𝑖−1assigns+1to correct answers
and−1to incorrect answers. The squared-error term is based on
the Brier score [ 10] and penalizes discrepancies between response
confidence and correctness. Equivalently,
𝑟ans(𝑜𝑖,𝑥)=(
1−𝜆𝑐(1−𝑐𝑖)2, 𝑦𝑖=1
−1−𝜆𝑐𝑐2
𝑖, 𝑦𝑖=0.(3)
When𝜆𝑐>0, higher confidence increases rewards for correct
answers and strengthens penalties for incorrect answers. Rewards
for incorrect answers never exceed −1, so lowering confidence
cannot make these rewards positive. The penalty thus provides
finer reward distinctions among responses that would otherwise
receive the same correctness-based reward.
We use raw confidence scores without post-hoc calibration and
treat them as fixed when computing policy gradients. These scores
influence policy updates through the rewards and the resulting
group-relative advantages.
4.2 Posterior-Adaptive Abstention Reward
To adapt abstention rewards to query-level answering performance,
we estimate answering reliability from the non-abstaining responses
in each sampled group. Let I𝑥={𝑖 :𝑎(𝑜𝑖)=0}denote the index set
of non-abstaining responses. The numbers of correct and incorrect
answers are:
𝐶𝑥=∑︁
𝑖∈I𝑥𝑦𝑖, 𝑊𝑥=∑︁
𝑖∈I𝑥(1−𝑦𝑖).(4)
Abstentions are excluded because they provide no answer whose
correctness can be evaluated.
Let𝑝𝑥denote the probability of a correct answer conditional on
answering under the policy and sampling configuration used to
generate the group:
𝑝𝑥=Pr(𝑦=1|𝑥,𝑎(𝑜)=0).(5)
We model the correctness labels of non-abstaining responses as con-
ditionally independent Bernoulli observations with success prob-
ability𝑝𝑥. With a prior 𝑝𝑥∼Beta(𝛼 0,𝛽0), where𝛼0,𝛽0>0, theposterior is:
𝑝𝑥|G𝑥∼Beta(𝛼 0+𝐶𝑥,𝛽0+𝑊𝑥).(6)
For a reliability threshold 𝜏∈( 0,1), we compute the posterior
probability that conditional answer accuracy falls below the thresh-
old:
𝑞𝑥=Pr(𝑝𝑥<𝜏|G𝑥)=𝐼𝜏(𝛼0+𝐶𝑥,𝛽0+𝑊𝑥),(7)
where𝐼𝜏is the regularized incomplete beta function. The poste-
rior incorporates both the observed correctness outcomes and the
number of non-abstaining responses available in the group.
We map this probability to an abstention reward:
𝑟abs(𝑥,G𝑥)=𝜆𝑎(2𝑞𝑥−1),(8)
where𝜆𝑎∈[0,1]controls the reward magnitude. The reward lies
in[−𝜆𝑎,𝜆𝑎]. For𝜆𝑎>0, it is positive when 𝑞𝑥>1/2and increases
with the posterior probability that answering reliability falls below
𝜏. Thus, stronger evidence of unreliable answering leads to a higher
abstention reward, whereas stronger evidence of reliable answering
leads to a lower reward. All abstentions within the same group share
this reward.
For each sampled group, we compute the posterior from the
same fixed prior and the correctness labels obtained by comparing
non-abstaining responses with the ground-truth answers. If a group
contains only abstentions, 𝐶𝑥=𝑊𝑥=0, and the posterior remains
equal to the prior. Under the default uniform prior 𝛼0=𝛽 0=1and
threshold𝜏=0.5, such a group receives zero abstention reward. Its
identical rewards yield zero group-relative advantages and hence
no reward-driven learning signal.
4.3 Confidence-Calibrated Policy Optimization
We adopt GRPO [ 34] as our optimization backbone, which computes
advantages from relative rewards within a response group without
requiring a separate value model. For each input 𝑥, a behavior policy
𝜋𝜃oldsamples a groupG𝑥={𝑜𝑖}𝐺
𝑖=1. We combine the confidence-
aware answer reward and the posterior-adaptive abstention reward
4

CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG Conference’17, July 2017, Washington, DC, USA
to assign a scalar reward to each response:
𝑟𝑖=(
𝑟abs(𝑥,G𝑥), 𝑎(𝑜𝑖)=1,
𝑟ans(𝑜𝑖,𝑥), 𝑎(𝑜 𝑖)=0.(9)
We set𝜆𝑐=𝜆𝑎=0.5by default. With these weights, correct answers
receive rewards no lower than abstentions, while abstentions re-
ceive higher rewards than incorrect answers. The reward gaps vary
with response confidence and group-level answering reliability.
Given the rewards{𝑟 𝑖}𝐺
𝑖=1, the group-relative advantage is:
A𝑖=𝑟𝑖−¯𝑟
𝑠𝑟+𝜀std, ¯𝑟=1
𝐺𝐺∑︁
𝑖=1𝑟𝑖,(10)
where𝑠𝑟is the within-group reward standard deviation and 𝜀std>0
ensures numerical stability. Each response-level advantage is shared
across all tokens of that response.
For the𝑡-th token of response 𝑜𝑖, the importance ratio between
the current and behavior policies is:
𝜌𝑖,𝑡(𝜃)=𝜋𝜃(𝑜𝑖,𝑡|𝑥,𝑜𝑖,<𝑡)
𝜋𝜃old(𝑜𝑖,𝑡|𝑥,𝑜𝑖,<𝑡).(11)
The clipped surrogate contribution is:
ℓ𝑖,𝑡(𝜃)=minn
𝜌𝑖,𝑡(𝜃)A𝑖,
clip 𝜌𝑖,𝑡(𝜃),1−𝜖,1+𝜖A𝑖o
,(12)
where𝜖>0controls the clipping range. We compute the policy
loss by averaging over tokens within each response and then over
responses within each group:
Lpg(𝜃)=−E𝑥,G𝑥"
1
𝐺𝐺∑︁
𝑖=11
𝑇𝑖𝑇𝑖∑︁
𝑡=1ℓ𝑖,𝑡(𝜃)#
,(13)
where𝑇𝑖is the response length, and the expectation is over training
inputs and response groups sampled from the behavior policy.
We regularize deviations from a fixed reference policy 𝜋refusing:
LKL(𝜃)=D KL(𝜋𝜃∥𝜋ref),(14)
where DKLdenotes the KL regularization term between the current
and reference policies. The total training loss is:
Ltrain(𝜃)=L pg(𝜃)+𝛽 KLLKL(𝜃),(15)
where𝛽KL≥0controls the regularization strength. The confidence
penalty and the adaptive abstention reward influence policy op-
timization through the group-relative advantages. Rewards and
advantages are treated as fixed during policy updates, with no
gradients propagated through the confidence scores or posterior
estimates used to compute the rewards.
4.4 Inference
At inference time, the trained policy takes a question and its asso-
ciated documents as input and directly generates a concise answer
or an abstention. The prompt in Appendix B.4 instructs the model
to answer using the provided documents only when it is confident
that its answer is correct. Otherwise, it should output“I don’t know”.
The model makes this decision during generation, without apply-
ing an explicit confidence threshold. Inference does not require
ground-truth answers, group-level reliability estimation, or reward
computation.5 Experiments
5.1 Experimental Setup
5.1.1 Datasets.We evaluate our method on three widely used QA
benchmarks covering different reasoning settings. For open-domain
QA, we adopt Natural Questions (NQ) [ 21] and TriviaQA [ 19], two
representative benchmarks for evaluating factual question answer-
ing. To further assess the model’s ability to integrate evidence across
multiple documents, we conduct experiments on HotpotQA [ 47], a
representative multi-hop QA benchmark. Together, these datasets
enable us to evaluate our method across factual question answer-
ing and more challenging multi-hop reasoning scenarios. Detailed
dataset statistics are reported in Appendix B.1.
5.1.2 Evaluation Metrics.We evaluate reliability and confidence
calibration using four metrics:Precision(Prec.) [ 35,46], which
measures accuracy among non-abstained responses;Truthfulness
(Truth.) [ 45,52], which measures the proportion of correct an-
swers or abstentions;Effective Reliability(ER) [ 8,39], which bal-
ances correct answering against incorrect answering; andExpected
Calibration Error(ECE) [ 12], which measures confidence calibra-
tion on non-abstained responses. Higher values indicate better
performance for Prec., Truth., and ER, whereas lower ECE is better.
Detailed definitions and computation procedures are provided in
Appendix B.2.
5.1.3 Baselines.We compare our method with the following base-
lines on the same test set. Training-free methods include In-Context
Learning (ICL), which uses answer-and-abstention demonstrations,
andPrompting, which instructs the model to answer when confi-
dent and abstain otherwise. SFT-based methods include standard
Supervised Fine-tuning (SFT), which trains the model to answer
every question, andR-Tuning[ 48], which teaches the model to
abstain beyond its knowledge. RL-based methods includeDTA[ 35],
which models knowledge boundaries using a four-quadrant parti-
tion and employs DPO [ 31];Binary, which uses GRPO [ 34] with
rewards of 1 for correct answers and 0 otherwise; andTruthRL[ 38],
which uses GRPO with rewards of 1, 0, and -1 for correct answers,
abstentions, and incorrect answers, respectively.
5.1.4 Implementation.We use LLaMA3-8B-Instruct [ 11] and Qwen2.5-
7B-Instruct [ 30] as the backbone models following prior work [ 1,40,
41,53]. To ensure fair and consistent comparisons [ 4,33], we follow
prior work [ 44,49,51] and provide all baselines and our method
with the same Top-5 Wikipedia passages retrieved by Contriever-
MS MARCO [ 15]. Specifically, the retrieved passages are kept fixed
across different methods for each query, ensuring that all methods
have access to identical external evidence. All training experiments
are conducted on four NVIDIA RTX PRO 6000 GPUs. Furthermore,
we use the same data splits, retrieved contexts, generation settings,
and evaluation protocols across all experiments to ensure consistent
and reproducible evaluations. Additional implementation details
are provided in Appendix B.3.
5.2 Main Results
We compare our method with all baselines on three QA benchmarks
using two backbone models. Table 1 summarizes the main results.
5

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
Table 1: Main results (%) on NQ, TriviaQA, and HotpotQA using LLaMA3-8B-Instruct and Qwen2.5-7B-Instruct as base models.
Results are reported in terms ofPrec.,Truth., andER, with higher values indicating better performance for all metrics. Avg.
denotes the unweighted mean across the three benchmarks. The best results are in bold and the second are underlined .
MethodNQ TriviaQA HotpotQA Avg.
Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑
Base Model: LLaMA3-8B-Instruct
Prompting 42.88 63.43 -9.11 68.26 76.22 27.36 38.76 68.38 -11.61 49.97 69.34 2.21
ICL 13.17 36.45 -53.91 34.10 58.61 -19.97 6.11 33.61 -62.07 17.79 42.89 -45.32
SFT 43.71 43.88 -12.55 63.98 63.98 27.96 34.23 34.43 -31.45 47.31 47.43 -5.35
R-Tuning 61.36 85.96 8.25 75.87 84.12 34.05 51.45 81.77 1.09 62.89 83.95 14.46
DTA 49.88 66.59 -0.17 74.97 80.80 38.30 35.57 58.07 -18.79 53.47 68.49 6.45
Binary 49.50 49.53 -1.00 73.54 73.54 47.06 38.46 38.57 -23.04 53.83 53.88 7.67
TruthRL 63.48 80.61 14.32 78.35 81.1249.4451.95 77.30 1.84 64.59 79.6821.87
CalibraRL (Ours) 68.65 87.89 14.40 80.27 85.0345.9457.10 84.46 5.14 68.67 85.7921.83
Base Model: Qwen2.5-7B-Instruct
Prompting 42.96 54.24 -11.30 69.36 74.47 32.27 42.81 65.48 -8.68 51.71 64.73 4.10
ICL 41.32 51.66 -14.29 67.52 71.88 30.35 37.02 53.95 -18.98 48.62 59.16 -0.97
SFT 40.18 40.94 -19.39 63.36 63.39 26.69 33.65 34.27 -32.39 45.73 46.20 -8.36
R-Tuning 54.72 77.154.76 74.23 80.54 36.60 48.81 74.16 -1.20 59.25 77.28 13.39
DTA 45.12 54.82 -8.03 71.32 75.67 36.17 41.89 62.59 -10.45 52.78 64.36 5.90
Binary 48.06 48.09 -3.88 74.79 74.7949.5842.96 42.96 -14.07 55.27 55.28 10.54
TruthRL 54.69 74.04 5.37 75.60 80.75 40.40 50.51 76.61 0.48 60.27 77.13 15.42
CalibraRL (Ours) 56.4976.43 7.04 80.41 84.6147.77 58.67 81.48 7.77 65.19 80.84 20.86
Table 2: Ablation results (%) on three QA benchmarks.w/o Answer Confidenceremoves the confidence reward term;w/o Adaptive
Abstentionreplaces the adaptive abstention reward with a fixed reward of zero. Avg. denotes the unweighted mean across the
three benchmarks. Best results are shown in bold.
VariantNQ TriviaQA HotpotQA Avg.
Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑
Base Model: LLaMA3-8B-Instruct
CalibraRL (Ours) 68.65 87.89 14.40 80.27 85.0345.9457.10 84.46 5.14 68.67 85.79 21.83
w/oAnswer Confidence 66.97 86.50 13.87 78.97 83.3046.0156.05 83.48 4.55 67.33 84.43 21.48
w/oAdaptive Abstention 65.81 84.65 14.20 79.17 83.58 45.99 56.26 83.52 4.71 67.08 83.92 21.63
Base Model: Qwen2.5-7B-Instruct
CalibraRL (Ours) 56.49 76.43 7.04 80.41 84.61 47.77 58.67 81.487.7765.19 80.84 20.86
w/oAnswer Confidence 56.19 75.32 6.98 79.17 83.74 45.54 58.21 79.767.9564.52 79.61 20.16
w/oAdaptive Abstention 55.48 74.97 6.16 79.86 84.12 47.09 58.18 80.46 7.64 64.51 79.85 20.30
(i) Overall performance.Our method achieves the highest preci-
sion in all six model–dataset settings and the highest truthfulness
in five of them. These results indicate higher accuracy among an-
swered questions and a lower proportion of incorrect outputs.
(ii) Effectiveness on multi-hop QA.Our method outperforms
all baselines on all three metrics on HotpotQA with both backbone
models. The improvements in precision and ER indicate that our
method achieves higher accuracy on answered questions while
also improving the balance between correct and incorrect answers
across the full evaluation set.
(iii) Consistency across backbone models.Our method achieves
higher precision and truthfulness than TruthRL on all three bench-
marks with both backbone models, demonstrating consistent gains
across the two evaluated backbones.5.3 Further Analysis
5.3.1 Ablation Study.As shown in Table 2, we evaluate the con-
tributions of the two reward components on NQ, TriviaQA, and
HotpotQA using both backbone models, keeping training data, ini-
tialization, sampling settings, and optimization hyperparameters
unchanged. Thew/o Answer Confidencevariant sets 𝜆𝑐=0, remov-
ing the Brier-shaped penalty and reducing the answer reward to
2𝑦𝑖−1while retaining adaptive abstention. Thew/o Adaptive Ab-
stentionvariant sets 𝜆𝑎=0, replacing the posterior-adaptive absten-
tion reward with zero while retaining answer-confidence shaping.
Abstaining responses remain in the sampled groups and GRPO
advantage computation. Removing either component reduces av-
erage precision, truthfulness, and ER on both backbones. The full
method also achieves the highest precision and truthfulness in all
6

CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG Conference’17, July 2017, Washington, DC, USA
Table 3: Effect of SFT initialization. CalibraRL applies reinforcement learning directly to the Instruct backbone, whereas SFT +
CalibraRL first performs SFT following R-Tuning and then applies CalibraRL. All scores are reported as percentages, with
higher values indicating better performance. Avg. denotes the unweighted mean across the three benchmarks. Best results are
shown in bold.
StrategyNQ TriviaQA HotpotQA Avg.
Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑Prec.↑Truth.↑ER↑
Base Model: LLaMA3-8B-Instruct
SFT + CalibraRL 58.92 74.68 11.00 74.15 77.20 42.59 47.22 72.41−2.9160.10 74.76 16.89
CalibraRL68.65 87.89 14.40 80.27 85.03 45.94 57.10 84.46 5.14 68.67 85.79 21.83
Base Model: Qwen2.5-7B-Instruct
SFT + CalibraRL 50.41 67.87 0.53 77.74 81.43 46.27 56.39 77.45 6.61 61.51 75.58 17.80
CalibraRL56.49 76.43 7.04 80.41 84.61 47.77 58.67 81.48 7.77 65.19 80.84 20.86
NQ TriviaQA HotpotQA
TargetNQ
TriviaQA
HotpotQASource68.65 83.13 64.23
58.22 80.27 48.54
60.67 81.67 57.10
50556065707580Prec. (%)
(a) Precision
NQ TriviaQA HotpotQA
TargetNQ
TriviaQA
HotpotQASource87.89 90.90 91.34
75.37 85.03 75.82
81.52 88.88 84.46
7678808284868890Truth. (%) (b) Truthfulness
NQ TriviaQA HotpotQA
TargetNQ
TriviaQA
HotpotQASource14.40 35.74 6.89
9.70 45.94 -1.38
10.03 38.44 5.14
010203040ER (%) (c) Effective Reliability
Figure 4: Cross-dataset generalization results of CalibraRL using LLaMA3-8B-Instruct. Rows indicate source datasets used for
training, and columns indicate target datasets used for evaluation. All scores are reported as percentages, with higher values
indicating better performance.
six model–dataset settings, although some ablated variants yield
slightly higher ER on individual datasets. These results indicate
that both reward components contribute to the overall performance
of CalibraRL.
5.3.2 Effect of SFT Initialization.We investigate whether SFT ini-
tialization using the R-Tuning procedure benefits CalibraRL. We
compare applying CalibraRL directly to the Instruct backbone with
first performing SFT following R-Tuning and then applying Cali-
braRL, denoted as SFT + CalibraRL. As shown in Table 3, directly
applying CalibraRL improves average precision, truthfulness, and
ER across the three benchmarks by 8.58, 11.03, and 4.93 percentage
points, respectively, on LLaMA3-8B-Instruct. The corresponding
gains on Qwen2.5-7B-Instruct are 3.68, 5.26, and 3.06 percentage
points. These results favor applying CalibraRL directly to the In-
struct backbones over using R-Tuning-based SFT initialization un-
der our experimental setup.
5.3.3 Confidence Calibration Analysis.Table 4 reports ECE before
and after CalibraRL training on three QA benchmarks. CalibraRL
reduces ECE on all three benchmarks with both backbones. The
average ECE decreases by 24.9% for LLaMA3-8B-Instruct and 35.1%
for Qwen2.5-7B-Instruct, with an overall reduction of 30.8% across
both backbones and all three benchmarks. These results indicate
improved confidence calibration on answered examples across the
evaluated backbones and QA benchmarks.Table 4: ECE performance before and after CalibraRL train-
ing on three datasets. Lower values indicate better calibra-
tion.
Stage NQ TriviaQA HotpotQA
Base Model: LLaMA3-8B-Instruct
Original Model 0.318 0.147 0.433
Ours 0.193 0.133 0.348
Base Model: Qwen2.5-7B-Instruct
Original Model 0.444 0.239 0.513
Ours 0.266 0.172 0.338
5.3.4 Cross-Dataset Generalization.We evaluate the generaliza-
tion of CalibraRL across QA datasets using LLaMA3-8B-Instruct.
Specifically, we train a separate model on the training split of each
source dataset and evaluate it on the held-out evaluation splits of all
three datasets without further training or parameter updates. This
setting examines how the learned answering and abstention behav-
ior transfers across data distributions. As shown in Figure 4, diag-
onal entries report in-domain performance, whereas off-diagonal
entries report cross-dataset performance. Models trained on NQ
and HotpotQA achieve positive ER on both out-of-domain datasets.
However, transfer performance varies across source–target pairs:
the model trained on TriviaQA achieves an ER of 9.70 on NQ
7

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
0.25 0.5 0.75
Answerability Threshold τ55606570758085Prec. (%)
0.25 0.5 0.75
Answerability Threshold τ8284868890Truth. (%)
0.25 0.5 0.75
Answerability Threshold τ01020304050ER (%)
NQ TriviaQA HotpotQA
Figure 5: Sensitivity of CalibraRL to the reliability threshold
𝜏using LLaMA3-8B-Instruct. The bold tick label indicates
the threshold used in the main experiments. All other hy-
perparameters are held fixed.
00.25 0.50.751
Confidence Penalty Weight λc55606570758085Prec. (%)
00.25 0.50.751
Confidence Penalty Weight λc8284868890Truth. (%)
00.25 0.50.751
Confidence Penalty Weight λc01020304050ER (%)
NQ TriviaQA HotpotQA
Figure 6: Effect of the confidence penalty weight 𝜆𝑐using
LLaMA3-8B-Instruct. Setting 𝜆𝑐=0removes the confidence
penalty while retaining the posterior-adaptive abstention
reward. The bold tick label indicates the weight used in the
main experiments. All other hyperparameters are held fixed.
and−1.38on HotpotQA. These results suggest that CalibraRL re-
tains useful answering behavior in several cross-dataset settings,
although its transfer performance depends on the source and target
distributions.
5.3.5 Effect of the Reliability Threshold.The reliability threshold 𝜏
in Eq. (7)defines the cutoff used to compute the posterior probability
that conditional answer accuracy falls below𝜏. This probability is
then used to compute the abstention reward in Eq. (8). We vary
𝜏over{0.25,0.5,0.75}and evaluate its effect on NQ, TriviaQA,
and HotpotQA using LLaMA3-8B-Instruct. As shown in Figure 5,
precision and truthfulness on NQ increase as 𝜏increases, while
performance on TriviaQA changes only slightly. On HotpotQA,
truthfulness decreases at 𝜏=0.5and then increases at 𝜏=0.75,
while precision varies more modestly. ER remains relatively stable
across the tested thresholds on all three benchmarks. These results
indicate that the effect of 𝜏varies across datasets and metrics, with
no single tested threshold achieving the best performance in every
setting.
5.3.6 Effect of the Confidence Penalty Weight.The confidence
penalty weight 𝜆𝑐controls the contribution of the squared-error
penalty in the answer reward, 𝑟ans(𝑜𝑖,𝑥)= 2𝑦𝑖−1−𝜆𝑐(𝑐𝑖−𝑦𝑖)2,
where𝑦𝑖∈{0,1}indicates answer correctness and 𝑐𝑖denotes re-
sponse confidence. We vary 𝜆𝑐over{0,0.25,0.5,0.75,1}on NQ,
TriviaQA, and HotpotQA using LLaMA3-8B-Instruct, with all other
hyperparameters held fixed. Setting 𝜆𝑐=0recovers thew/o An-
swer Confidenceablation by removing the confidence penalty while
0.5k 1k 2k 5k 8k 10k 12k 15k
Training Data Size0153045607590Performance (%)Precision Truthfulness ERFigure 7: Effect of training data size using LLaMA3-8B-
Instruct on NQ. The bold tick label indicates the training
size of 10k examples used in the main experiments.
retaining the posterior-adaptive abstention reward. As shown in
Figure 6,𝜆𝑐=0.5yields the highest precision and truthfulness on
NQ, although some other tested values yield higher ER. On Trivi-
aQA, precision and truthfulness peak at 𝜆𝑐=1, while ER changes
only slightly across the tested weights. Performance on HotpotQA
also varies modestly. Overall, the effect of the confidence penalty
depends on the dataset and metric, with no single tested weight
performing best across all settings.
5.3.7 Effect of Training Data Size.We investigate the effect of
training data size using LLaMA3-8B-Instruct on NQ by varying
the number of training examples from 0.5k to 15k. As shown in
Figure 7, Precision, Truthfulness, and ER generally improve as the
training set grows to 10k examples. At this size, CalibraRL achieves
a precision of 68.65%, a truthfulness of 87.89%, and an ER of 14.40%.
Increasing the training set beyond 10,000 examples further improves
ER but reduces precision and truthfulness. These results indicate
that larger training sets do not consistently improve all metrics. In
our main experiments, we use 10k training examples, which provide
strong overall performance without requiring a larger training set.
6 Conclusion
We identify a limitation of using fixed rewards for incorrect an-
swers and abstentions: incorrect responses vary in confidence, while
queries on which the model abstains differ in estimated answer-
ability. We propose CalibraRL, which combines a Brier-based con-
fidence penalty with a posterior-adaptive abstention reward de-
rived from group-level correctness evidence. By reusing training
rollouts, CalibraRL requires no additional annotations indicating
whether the model knows the answer to each question. Experiments
on three QA benchmarks with two instruction-tuned backbones
demonstrate consistent improvements in precision and truthfulness
over TruthRL. Further analysis shows improved confidence calibra-
tion on answered examples across both backbone models. These
findings highlight the value of incorporating confidence and esti-
mated answering reliability into reinforcement learning rewards for
more reliable answering and abstention. Future work will explore
robustness to noisy correctness feedback and extend CalibraRL to
long-form generation.
8

CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG Conference’17, July 2017, Washington, DC, USA
Acknowledgements
This work was funded by the National Natural Science Foundation
of China (NSFC) under Grants No. 62406013 and No. U25B2070, the
Beijing Advanced Innovation Center Funds for Future Blockchain
and Privacy Computing, and the Fundamental Research Funds for
the Central Universities.
References
[1]Zouying Cao, Yifei Yang, Xiaojing Li, and Hai Zhao. 2025. AutoHall: Automated
factuality hallucination dataset generation for large language models.IEEE
Transactions on Audio, Speech and Language Processing34 (2025), 184–195.
[2]Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang,
Dawei Yin, Yiming Yang, and Jiaxin Mao. 2026. Improving retrieval-augmented
generation through multi-agent reinforcement learning.Advances in Neural
Information Processing Systems38 (2026), 121336–121367.
[3]Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and
Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality
in large language models. InInternational Conference on Learning Representations,
Vol. 2024. 54158–54183.
[4]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare
Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The
power of noise: Redefining retrieval for rag systems. InProceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 719–729.
[5]Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon
Kim, and Jacob Andreas. 2026. Beyond binary rewards: Training lms to reason
about their uncertainty. InInternational conference on learning representations,
Vol. 2026. 59456–59493.
[6]Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli
Celikyilmaz, and Jason Weston. 2024. Chain-of-verification reduces hallucina-
tion in large language models. InFindings of the association for computational
linguistics: ACL 2024. 3563–3578.
[7]Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting
hallucinations in large language models using semantic entropy.Nature630,
8017 (2024), 625–630.
[8] Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran,
and Yulia Tsvetkov. 2024. Don’t hallucinate, abstain: Identifying llm knowledge
gaps via multi-llm collaboration. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 14664–14690.
[9]Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart,
and Jonathan Herzig. 2024. Does fine-tuning LLMs on new knowledge encourage
hallucinations?. InProceedings of the 2024 conference on empirical methods in
natural language processing. 7765–7784.
[10] W Brier Glenn. 1950. Verification of forecasts expressed in terms of probability.
Monthly weather review78, 1 (1950), 1–3.
[11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783
(2024).
[12] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of
modern neural networks. InInternational conference on machine learning. PMLR,
1321–1330.
[13] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large
Language Models. InInternational Conference on Learning Representations. https:
//openreview.net/forum?id=nZeVKeeFYf9
[14] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A
survey on hallucination in large language models: Principles, taxonomy, chal-
lenges, and open questions.ACM transactions on information systems43, 2 (2025),
1–55.
[15] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv preprint arXiv:2112.09118
(2021).
[16] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation.ACM computing surveys55, 12 (2023), 1–38.
[17] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. InSecond Conference on
Language Modeling. https://openreview.net/forum?id=Rwhi91ideu
[18] Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and
Cyrus Rashtchian. 2025. Sufficient context: A new lens on retrieval augmentedgeneration systems. InInternational Conference on Learning Representations,
Vol. 2025. 20310–20334.
[19] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A
Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.
InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 1601–1611.
[20] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain,
Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-
Johnson, et al .2022. Language models (mostly) know what they know.arXiv
preprint arXiv:2207.05221(2022).
[21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[23] Yuxin Liang, Zhuoyang Song, Hao Wang, and Jiaxing Zhang. 2024. Learning to
trust your feelings: Leveraging self-awareness in llms for hallucination mitigation.
InProceedings of the 3rd Workshop on Knowledge Augmented Methods for NLP.
44–58.
[24] Hsin-Yi Lin, Yu-Che Chang, and Jen-Tzung Chien. 2025. Variational dehalluci-
nation via retrieval-augmented prompt learning.IEEE Transactions on Audio,
Speech and Language Processing34 (2025), 259–271.
[25] Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring How
Models Mimic Human Falsehoods. InProceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers). 3214–3252.
[26] Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-
Resource Black-Box Hallucination Detection for Generative Large Language
Models. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. 9004–9017.
[27] Shiyu Ni, Keping Bi, Jiafeng Guo, Lulu Yu, Baolong Bi, and Xueqi Cheng. 2025.
Towards Fully Exploiting LLM Internal States to Enhance Knowledge Bound-
ary Perception. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 24315–24329.
[28] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas
Kotek, and Yonatan Belinkov. 2025. Llms know more than they show: On the
intrinsic representation of llm hallucinations. InInternational Conference on
Learning Representations, Vol. 2025. 66880–66913.
[29] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin,
Yuxiang Wu, and Alexander Miller. 2019. Language Models as Knowledge Bases?.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Pro-
cessing (EMNLP-IJCNLP). 2463–2473.
[30] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[31] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural information processing
systems36 (2023), 53728–53741.
[32] Abhilasha Ravichander, Shrusti Ghela, David Wadden, and Yejin Choi. 2025.
HALoGEN: Fantastic LLM Hallucinations and Where to Find Them. InProceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). 1402–1425.
[33] Alireza Salemi and Hamed Zamani. 2024. Evaluating retrieval quality in retrieval-
augmented generation. InProceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval. 2395–2400.
[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al .2024. Deepseekmath: Pushing
the limits of mathematical reasoning in open language models.arXiv preprint
arXiv:2402.03300(2024).
[35] Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Yuehe Chen, Bowen
Song, Zilei Wang, Weiqiang Wang, and Liang Wang. 2025. Divide-then-align:
Honest alignment based on the knowledge boundary of rag. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 11461–11480.
[36] Hexiang Tan, Wanli Yang, Junwei Zhang, Xin Chen, Rui Tang, Du Su, Jingang
Wang, Yuanzhuo Wang, Fei Sun, and Xueqi Cheng. 2026. BaseCal: Unsupervised
Confidence Calibration via Base Model Signals. InProceedings of the 64th Annual
9

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
5172–5187.
[37] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. 2025. Robust fine-
tuning for retrieval augmented generation against retrieval defects. InProceedings
of the 48th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 1272–1282.
[38] Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Jingxiang Chen, Moham-
mad Kachuee, Teja Gollapudi, Yiwei Liao, Nicolas SCHEFFER, Rakesh Wanga,
Anuj Kumar, Yu Meng, Wen tau Yih, and Xin Luna Dong. 2026. TruthRL: Incen-
tivizing Truthful LLMs via Reinforcement Learning. InForty-third International
Conference on Machine Learning. https://openreview.net/forum?id=TnJMJm3Fbi
[39] Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe,
and Lucy Lu Wang. 2025. Know your limits: A survey of abstention in large
language models.Transactions of the Association for Computational Linguistics13
(2025), 529–556.
[40] Fei Wu, Zhenrong Zhang, Qikai Chang, Jianshu Zhang, Quan Liu, and Jun Du.
2026. Step potential advantage estimation: Harnessing intermediate confidence
and correctness for efficient mathematical reasoning. InFindings of the Association
for Computational Linguistics: ACL 2026. 6361–6379.
[41] Tengxiao Xi, Chen Wang, and Jiajun Zhang. 2026. Confidence introspection:
A self-reflection method for reliable and helpful large language models.IEEE
Transactions on Audio, Speech and Language Processing(2026).
[42] Xi Xiao, Chen Liu, Chih-Ting Liao, Yunbei Zhang, Qizhen Lan, Yuxiang Wei,
Lin Zhao, Janet Wang, Jianyang Gu, Muchao Ye, et al .2026. Staying VIGIL ant:
Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs. In
European Conference on Computer Vision. Springer, 287–305.
[43] Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan
Hooi. 2024. Can llms express their uncertainty? an empirical evaluation of confi-
dence elicitation in llms. InInternational Conference on Learning Representations,
Vol. 2024. 23650–23678.
[44] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving retrieval-
augmented LMs with context compression and selective augmentation. InInter-
national Conference on Learning Representations, Vol. 2024. 43478–43502.
[45] Hongshen Xu, Zichen Zhu, Situo Zhang, Da Ma, Shuai Fan, Lu Chen, and Kai
Yu. 2024. Rejection Improves Reliability: Training LLMs to Refuse Unknown
Questions Using RL from Knowledge Feedback. InFirst Conference on Language
Modeling. https://openreview.net/forum?id=lJMioZBoR8
[46] Boyang Xue, Fei Mi, Qi Zhu, Hongru Wang, Rui Wang, Sheng Wang, Erxin Yu,
Xuming Hu, and Kam-Fai Wong. 2025. Ualign: Leveraging uncertainty estimations
for factuality alignment on large language models. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). 6002–6024.
[47] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[48] Hanning Zhang, Shizhe Diao, Yong Lin, Yi Fung, Qing Lian, Xingyao Wang,
Yangyi Chen, Heng Ji, and Tong Zhang. 2024. R-Tuning: Instructing Large
Language Models to Say ‘I Don’t Know’. InProceedings of the 2024 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers). 7113–7139.
[49] Qianchi Zhang, Hainan Zhang, Liang Pang, Yongxin Tong, Hongwei Zheng, and
Zhiming Zheng. 2026. Less is more: Compact clue selection for efficient retrieval-
augmented generation reasoning. InProceedings of the ACM Web Conference 2026.
1971–1982.
[50] Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng.
2024. Adacomp: Extractive context compression with adaptive predictor for
retrieval-augmented large language models.arXiv preprint arXiv:2409.01579
(2024).
[51] Qianchi Zhang, Hainan Zhang, Liang Pang, Hong-Wei Zheng, and Zhiming
Zheng. 2026. Stable-rag: Mitigating retrieval-permutation-induced hallucinations
in retrieval-augmented generation. InProceedings of the 64th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers). 25907–
25926.
[52] Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Lifeng Jin, Linfeng Song,
Haitao Mi, and Helen Meng. 2024. Self-alignment for factuality: Mitigating
hallucinations in llms via self-evaluation. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
1946–1965.
[53] Zhenrong Zhang, Fei Wu, Jun Du, Jianshu Zhang, and Si Wei. 2026. ReDiPPO:
Reference-Guided Value Calibration and Discrepancy-Aware Token Reweighting
for Mathematical Reasoning.arXiv preprint arXiv:2607.27631(2026).
[54] Yujia Zhou, Wenbo Zhang, Jingying Shao, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin
Qian, Zheng Liu, Chaozhuo Li, Jason Chen Zhang, et al .2024. Trustworthi-
ness in retrieval-augmented generation systems: A survey.arXiv preprint
arXiv:2409.10102(2024).A Algorithm Details
Algorithm 1 summarizes CalibraRL training. Responses receive
confidence-aware answer rewards or posterior-adaptive abstention
rewards, with the posterior derived from a fixed prior and the
correctness outcomes of the group’s non-abstaining responses.
Algorithm 1:CalibraRL
Input: Training dataD, initial policy 𝜋𝜃, fixed reference policy 𝜋ref, group
size𝐺, prior parameters 𝛼0,𝛽0, reliability threshold 𝜏, reward weights
𝜆𝑐,𝜆𝑎, clipping parameter𝜖, KL weight𝛽 KL, stability constant𝜀 std
Output:Trained policy𝜋 𝜃
1:foreach training iterationdo
2:𝜃 old←𝜃
3:Sample a training batchBfromD
4:foreach(𝑥,A)∈Bdo
5:SampleG 𝑥={𝑜𝑖}𝐺
𝑖=1from𝜋𝜃old(·|𝑥)
6:Identify abstentions𝑎 𝑖←𝑎(𝑜𝑖)for all𝑖
7:I 𝑥←{𝑖:𝑎𝑖=0}
8:foreach𝑖∈I 𝑥do
9:𝑦 𝑖←𝑦(𝑜𝑖,A)
10:𝑐 𝑖←𝑐𝜃old(𝑥,𝑜𝑖)using Eq. (1)
11:𝑟 𝑖←2𝑦𝑖−1−𝜆𝑐(𝑐𝑖−𝑦𝑖)2
12:end for
13:𝐶 𝑥←Í
𝑖∈I𝑥𝑦𝑖
14:𝑊 𝑥←Í
𝑖∈I𝑥(1−𝑦𝑖)
15:𝑞 𝑥←𝐼𝜏(𝛼0+𝐶𝑥,𝛽0+𝑊𝑥)
16:foreach𝑖such that𝑎 𝑖=1do
17:𝑟 𝑖←𝜆𝑎(2𝑞𝑥−1)
18:end for
19: ¯𝑟←𝐺−1Í𝐺
𝑖=1𝑟𝑖
20:𝑠 𝑟←Std(𝑟 1,...,𝑟𝐺)
21:for𝑖=1,...,𝐺do
22:A 𝑖←(𝑟𝑖−¯𝑟)/(𝑠𝑟+𝜀std)
23:end for
24:StoreG 𝑥and its fixed advantages{A 𝑖}𝐺
𝑖=1
25:end for
26:foreach policy update on the sampled batchdo
27:Compute token-level ratios𝜌 𝑖,𝑡(𝜃)using Eq. (11)
28:Compute clipped termsℓ 𝑖,𝑡(𝜃)using Eq. (12)
29:ComputeL pg(𝜃)over the batch using Eq. (13)
30:ComputeL KL(𝜃)against𝜋 refusing Eq. (14)
31:L train←L pg+𝛽 KLLKL
32: Update𝜃by minimizingLtrain, keeping rewards and advantages
fixed
33:end for
34:end for
35:return𝜋 𝜃
B More Experimental Settings
B.1 Datasets
We conduct experiments on three widely used QA datasets that
cover both single-hop and multi-hop question-answering scenarios.
Table 5 summarizes the key statistics of these datasets. Specifically,
NQ[ 21] andTriviaQA[ 19] are representative single-hop datasets,
where each question can typically be answered using information
from a single passage retrieved from the corpus. These datasets
primarily evaluate a model’s ability to locate and extract factual
evidence efficiently. In contrast,HotpotQA[ 47] is a challenging
10

CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG Conference’17, July 2017, Washington, DC, USA
Table 5: Dataset statistics, including reasoning types and the
number of examples in each split.
Dataset Type # Train # Dev # Test
NQ single-hop 79.1k 8.7k 3.6k
TriviaQA single-hop 78.7k 11.3k 8.8k
HotpotQA multi-hop 88.9k 5.6k 5.6k
multi-hop dataset that requires integrating and reasoning over mul-
tiple pieces of evidence distributed across different documents to
derive the final answer. This dataset is particularly useful for testing
a model’s reasoning and compositional understanding capabilities.
Together, these datasets enable us to evaluate the answering re-
liability and abstention behavior of CalibraRL across factual and
multi-hop QA settings.
B.2 Evaluation Metrics
Let𝑁denote the total number of evaluation examples, and let 𝑁cor,
𝑁inc, and𝑁absdenote the numbers of correct answers, incorrect
answers, and abstentions, respectively. Thus, 𝑁=𝑁 cor+𝑁 inc+𝑁 abs,
and the number of answered examples is𝑁 ans=𝑁 cor+𝑁 inc.
Precision.Precision measures the proportion of correct answers
among answered examples. It evaluates answer accuracy condi-
tional on the model choosing to answer.
Truthfulness.Truthfulness measures the proportion of eval-
uation examples on which the model either answers correctly or
abstains: Truthfulness=𝑁cor+𝑁abs
𝑁=1−𝑁inc
𝑁.Because abstentions
contribute positively to this metric, we interpret it jointly with
precision and ER.
ER.Effective reliability assigns a utility of +1to correct answers,
−1to incorrect answers, and0to abstentions: ER=𝑁cor−𝑁inc
𝑁. It
measures the net utility of answering across the full evaluation set.
ECE.We compute ECE [ 12] only on answered examples. For
each answered example𝑖, let𝑐 𝑖∈[0,1]denote the response confi-
dence and𝑦𝑖∈{0,1}indicate whether the answer is correct. We
divide the confidence interval [0,1]into𝑀equal-width bins and
assign each answered example to the corresponding bin. Let 𝐵𝑚
denote the set of examples in bin 𝑚. For each nonempty bin, empiri-
cal accuracy and average confidence are Acc(𝐵𝑚)=1
|𝐵𝑚|Í
𝑖∈𝐵𝑚𝑦𝑖
andConf(𝐵𝑚)=1
|𝐵𝑚|Í
𝑖∈𝐵𝑚𝑐𝑖.ECE is the weighted average of
the absolute differences between bin accuracy and bin confidence:
ECE=∑︁
1≤𝑚≤𝑀
|𝐵𝑚|>0|𝐵𝑚|
𝑁ans|Acc(𝐵𝑚)−Conf(𝐵 𝑚)|.(16)
We use𝑀= 10in all experiments. Lower ECE indicates closer
agreement between confidence and empirical accuracy on answered
examples.Table 6: Prompt for the base models.
Prompt for the Base Models
<system>
You are a helpful, respectful, and honest assistant.
Answer the question with a couple of words
using the provided documents only when you are
confident the answer is correct. Otherwise, respond
exactly with ’I don’t know’.
</system>
<user>
Question:{query}
Documents:
Doc1:{Document 1}
Doc2:{Document 2}
......
</user>
B.3 Training Details
We use LLaMA3-8B-Instruct1[11] and Qwen2.5-7B-Instruct2[30]
as the backbone models following prior work [ 1,40,41,53]. These
models and their variants have been widely adopted across various
tasks and applications [ 42]. We train CalibraRL using GRPO with
LoRA [ 13] for parameter-efficient fine-tuning. For both backbones,
we set the dataset size parameter to 10k and the validation ratio
to 0.1. LoRA is configured with a rank of 64, a scaling parameter
of 128, and a dropout rate of 0.. We train both models for one
epoch with a learning rate of1 ×10−6, a batch size of 4, and 8
gradient accumulation steps. The evaluation batch size is 32, and the
warmup ratio is 0.03. The maximum prompt and response lengths
are 2,048 and 16 tokens, respectively. During training, we sample
32 responses per query for LLaMA3-8B-Instruct and Qwen2.5-7B-
Instruct, with sampling temperatures of 1.0 and 1.2, respectively.
Both models use nucleus sampling with 𝑝=0.95. We set the GRPO
clipping parameter to 0.2 and the KL regularization coefficient
to 0.01. The confidence penalty weight is 0.5. For the posterior-
adaptive abstention reward, we use a uniform Beta prior with 𝛼0=
𝛽0=1, a reliability threshold of 𝜏= 0.5, and a reward scale of
𝜆𝑎=0.5. No additional confidence calibration transformation is
fitted during training. We fix the random seed to 42 for all runs.
B.4 Prompt
Table 6 presents the prompt template used for the base models.
The system instruction encourages concise answers grounded in
the supplied documents and requires the model to respond exactly
with“I don’t know”when uncertain. The user message contains
the question and the retrieved documents, which provide external
evidence for answering.
1https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
11

Conference’17, July 2017, Washington, DC, USA Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng
0.0 0.2 0.4 0.6 0.8 1.0
Confidence0.00.20.40.60.81.0Accuracy
ECE = 0.147Perfect calibration
Empirical accuracy
0.0 0.2 0.4 0.6 0.8 1.0
Confidence03000600090001200015000Incorrect Responseγ=0.8Lower confidence
High confidence
0.0 0.2 0.4 0.6 0.8 1.0
Reference Answer Accuracy0350700105014001750Question Countτ=0.5Below threshold
Potential over-IDK
(A) Confidence Calibration (B) Confidence of Errors (C) Answerability of IDK Questions
Figure 8: Empirical motivation using LLaMA3-8B-Instruct on TriviaQA. (A) Empirical accuracy generally increases with
confidence. (B) Incorrect responses exhibit heterogeneous confidence. (C) Questions eliciting abstention differ in the accuracy
of non-abstaining responses obtained through independent reference sampling.
0.0 0.2 0.4 0.6 0.8 1.0
Confidence0.00.20.40.60.81.0Accuracy
ECE = 0.433Perfect calibration
Empirical accuracy
0.0 0.2 0.4 0.6 0.8 1.0
Confidence03000600090001200015000Incorrect Responseγ=0.8Lower confidence
High confidence
0.0 0.2 0.4 0.6 0.8 1.0
Reference Answer Accuracy0450900135018002250Question Countτ=0.5Below threshold
Potential over-IDK
(A) Confidence Calibration (B) Confidence of Errors (C) Answerability of IDK Questions
Figure 9: Empirical motivation using LLaMA3-8B-Instruct on HotpotQA. (A) Empirical accuracy generally increases with
confidence. (B) Incorrect responses exhibit heterogeneous confidence. (C) Questions eliciting abstention differ in the accuracy
of non-abstaining responses obtained through independent reference sampling.
C More Experimental Results
C.1 Additional Empirical Motivation
We extend the empirical analyses in the preliminaries to TriviaQA
and HotpotQA. Figure 8 and Figure 9 show the confidence–accuracy
relationship, the confidence distribution of incorrect responses,
and reference answer accuracy for questions eliciting abstention,
following the same protocol used for NQ.
C.2 Effect of Rollout Group Size
We vary the rollout group size 𝐺over{4,8,16,32,64}. Group size
affects abstention reward estimation and group-relative advantage
computation. As shown in Figure 10, increasing 𝐺from4to32gen-
erally improves performance across datasets. Precision increases
by10.63,5.10, and4.72percentage points on NQ, TriviaQA, and
HotpotQA, respectively, with improvements in truthfulness and
ER. These results suggest that larger groups provide more informa-
tive samples for abstention reward estimation and group-relative
advantage computation. Increasing 𝐺to64yields limited and in-
consistent gains: NQ shows marginal improvements in precision
4816 3264
Rollout Group Size G5055606570758085Prec. (%)
4816 3264
Rollout Group Size G808284868890Truth. (%)
4816 3264
Rollout Group Size G01020304050ER (%)
NQ TriviaQA HotpotQAFigure 10: Effect of rollout group size 𝐺on (a) precision, (b)
truthfulness, and (c) effective reliability (ER) across NQ, Triv-
iaQA, and HotpotQA. The vertical dashed lines indicate the
default setting𝐺=32.
and truthfulness but a slight decline in ER, TriviaQA exhibits a
small increase in ER despite decreases in the other two metrics,
and HotpotQA shows declines in all three metrics. Overall, 𝐺= 32
achieves the best or near-best performance, supporting its use as
the default group size.
12