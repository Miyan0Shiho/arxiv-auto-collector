# Lessons from Training Grounded LLMs with Verifiable Rewards

**Authors**: Shang Hong Sim, Tej Deep Pala, Vernon Toh, Hai Leong Chieu, Amir Zadeh, Chuan Li, Navonil Majumder, Soujanya Poria

**Published**: 2025-06-18 14:58:13

**PDF URL**: [http://arxiv.org/pdf/2506.15522v1](http://arxiv.org/pdf/2506.15522v1)

## Abstract
Generating grounded and trustworthy responses remains a key challenge for
large language models (LLMs). While retrieval-augmented generation (RAG) with
citation-based grounding holds promise, instruction-tuned models frequently
fail even in straightforward scenarios: missing explicitly stated answers,
citing incorrectly, or refusing when evidence is available. In this work, we
explore how reinforcement learning (RL) and internal reasoning can enhance
grounding in LLMs. We use the GRPO (Group Relative Policy Optimization) method
to train models using verifiable outcome-based rewards targeting answer
correctness, citation sufficiency, and refusal quality, without requiring gold
reasoning traces or expensive annotations. Through comprehensive experiments
across ASQA, QAMPARI, ELI5, and ExpertQA we show that reasoning-augmented
models significantly outperform instruction-only variants, especially in
handling unanswerable queries and generating well-cited responses. A two-stage
training setup, first optimizing answer and citation behavior and then refusal,
further improves grounding by stabilizing the learning signal. Additionally, we
revisit instruction tuning via GPT-4 distillation and find that combining it
with GRPO enhances performance on long-form, generative QA tasks. Overall, our
findings highlight the value of reasoning, stage-wise optimization, and
outcome-driven RL for building more verifiable and reliable LLMs.

## Full Text


<!-- PDF content starts -->

arXiv:2506.15522v1  [cs.CL]  18 Jun 2025Lessons from Training Grounded LLMs with Verifiable Rewards
Shang Hong Sim1*, Tej Deep Pala1*, Vernon Toh1*, Hai Leong Chieu2, Amir Zadeh3, Chuan Li3,
Navonil Majumder1, Soujanya Poria1
1Singapore University of Technology and Design,2DSO National Laboratories,3Lambda Labs
*Equal contribution
Abstract
Generating grounded and trustworthy responses remains a key
challenge for large language models (LLMs). While retrieval-
augmented generation (RAG) with citation-based grounding
holds promise, instruction-tuned models frequently fail even
in straightforward scenarios: missing explicitly stated answers,
citing incorrectly, or refusing when evidence is available. In
this work, we explore how reinforcement learning (RL) and
internal reasoning can enhance grounding in LLMs. We use
the GRPO (Group Relative Policy Optimization) method to
train models using verifiable outcome-based rewards targeting
answer correctness, citation sufficiency, and refusal quality,
without requiring gold reasoning traces or expensive anno-
tations. Through comprehensive experiments across ASQA,
QAMPARI, ELI5, and ExpertQA we show that reasoning-
augmented models significantly outperform instruction-only
variants, especially in handling unanswerable queries and gen-
erating well-cited responses. A two-stage training setup, first
optimizing answer and citation behavior and then refusal, fur-
ther improves grounding by stabilizing the learning signal.
Additionally, we revisit instruction tuning via GPT-4 distil-
lation and find that combining it with GRPO enhances per-
formance on long-form, generative QA tasks. Overall, our
findings highlight the value of reasoning, stage-wise optimiza-
tion, and outcome-driven RL for building more verifiable and
reliable LLMs.
1 Introduction
The deployment of large language models (LLMs) in
information-seeking applications has grown rapidly, yet en-
suring that these models generate factually reliable and
grounded outputs remains an open challenge. A core con-
cern is the problem of hallucination , where models produce
fluent but fabricated or unfaithful content (Ji et al. 2023).
Retrieval-Augmented Generation (RAG) offers a promising
solution by conditioning responses on external documents,
which improves factual accuracy and enables verification
(Asai et al. 2023). Despite this, even advanced models like
GPT-4 often mishandle retrieved information—leading to
RAG-specific hallucinations when evidence is misused, omit-
ted, or over-cited (Gao et al. 2023b; Song et al. 2025b).
Recently, several works have begun integrating retrieval
more tightly into internal reasoning processes. For exam-
ple, Search-o1 and RAG-Star embed retrieval into multi-step
reasoning pipelines to dynamically acquire and filter sup-porting evidence (Li et al. 2025b; Jiang et al. 2024). Other
approaches like ReSearch and R1-Searcher train LLMs to
autonomously issue retrieval queries using reinforcement
learning (RL) (Chen et al. 2025; Song et al. 2025a), while
CoT-RAG enhances chain-of-thought prompting with sub-
query decomposition and knowledge graph guidance (Li et al.
2025a). While these methods improve retrieval-driven reason-
ing, their primary focus is on complex QA or planning tasks
and not directly on grounding alignment, such as accurate
citation or proper refusal when no answer is found.
In this paper, we focus specifically on improving grounded
response quality in LLMs—ensuring that answers are correct,
verifiably cited, and appropriately refused when evidence is
lacking. We propose Ground-GRPO , a two-stage reinforce-
ment learning framework based on Group Relative Policy
Optimization (GRPO). We propose a novel hierarchial re-
ward system that uses verifiable outcome-based rewards to
enhance grounding in 3 directions: Answer Correctness, Ci-
tation Quality and Refusal.
Ground-GRPO proceeds in two stages:
1.Answer and Citation Optimization: The model is
trained on answerable examples using rewards that en-
courage correct answers and minimal yet sufficient cita-
tions.
2.Refusal Learning: The model is then trained on a mix-
ture of answerable and unanswerable examples, using
a refusal-specific reward to discourage unsupported an-
swers.
Our results show that large reasoning models out-
perform instruction-tuned variants across multiple QA
benchmarks—ASQA, QAMPARI, and ELI5—under the
TrustScore metric (Song et al. 2025b), with notable gains
in identifying unanswerable questions and providing precise,
grounded citations. We find that reasoning-enhanced models
benefit significantly more from Ground-GRPO than their
instruction-only counterparts, highlighting the synergy be-
tween intermediate reasoning and reinforcement learning.
Additionally, prior work has shown that GPT-4-based dis-
tillation improves model groundedness and citation quality.
We find that distillation performs well in structured, list-
style tasks like QAMPARI, while Ground-GRPO achieves
stronger results on open-ended, long-form QA tasks such as
ELI5, where generating coherent, well-cited explanations is

more challenging.
Key Takeaways
•Reasoning boosts grounding: Large Reasoning
Models generate more grounded answers com-
pared to instruct models.
•RL is more effective on reasoning models:
Ground-GRPO yields greater improvements on
reasoning models than on instruction-tuned mod-
els.
•Staged training improves alignment: Decompos-
ing rewards across stages—first for answer quality,
then for refusal—stabilizes training and leads to
better overall grounding.
•Distillation and RL are complementary on long-
form QA datasets: GPT-4 distillation and fine-
tuning small models through SFT or DPO im-
proves both the shortform and longform QA per-
formance; GRPO further enhances grounding, es-
pecially in open-ended, long-form tasks.
2 Problem Definition and Method
Given a question Qand a set of relevant documents D,
the LLM is instructed to generate a response Sconsisting
of a set of citation-grounded statements {s1, s2, ...}. Each
statement siis accompanied by a set of inline citations
Ci={ci,1, ci,2, ..}that refers to the documents in D, e.g.,
statement1 [1][2] statement2 [3] . If the rele-
vant documents Dis insufficient to answer the question, the
question is deemed unanswerable , and the gold response S
would be a refusal statement, such as, “I apologize, but I
couldn’t find an answer to your question in the search re-
sults” . Otherwise, the question is answerable .
2.1 Two-stage Reinforcement Learning
To facilitate robust learning of training signals by the LLM,
we propose a two-stage RL approach that incorporates verifi-
able rewards. In Stage 1, the model is trained to accurately
answer answerable questions with appropriate citations while
adhering to specified output format constraints. In Stage 2, the
model is further trained on a mixture of answerable and unan-
swerable questions, enabling it to provide cited responses
when possible and appropriately refuse to answer when ques-
tions are unanswerable.
Stage 1. The model is trained only on answerable questions
in this stage. The reward function consists of three compo-
nents: exact match (EM), citation, and format rewards. The
primary objective is for the model to learn how to effectively
utilize the provided documents to generate accurate responses
to answerable questions.
The EM (Exact Match) reward is computed at the level of
individual statements si. It is defined as follows:
REM=0.5 if statement sicontains EM ,
0 otherwise(1)This formulation assigns a reward of 0.5 for each statement
that is deemed correct, and no reward for incorrect statements.
The citation reward is computed at the level of each correct
EM statement si. It it defined as follows:
Rcitation =

0.5 if statement sicontaining EM has correct citation ,
−0.5 if statement sicontaining EM has incorrect citation ,
0 otherwise
(2)
This encourages the model not only to produce accurate
statements but also to support them with appropriate citations,
while penalizing unsupported or incorrectly cited statements.
We further enforce the output format of the model’s re-
sponse where the thinking process should be enclosed within
<think>...</think> tags, while the final answer only must
appear within <answer>...</answer> tags. The <answer> sec-
tion should not include any reasoning or justification. This
constraint is enforced through a hard formatting reward,
Rformat , and a soft tag count reward, Rtag_count , defined as
follows:
LetT={"<think>" ,"</think>" ,"<answer>" ,"</answer>" }.
Rtag_count =1
|T |X
t∈T1{count (t)=1} (3)
Rformat=1.0 if output format is correct,
0 otherwise(4)
Importantly, if output format is incorrect, the sample will
not receive rewards for either REMorRcitation . In such cases,
the final reward defaults to Rtag_count alone. The overall re-
ward of Stage-1 is computed as the sum of the EM, citation,
and formatting rewards:
RStage-1 =REM+Rcitation +Rtag_count +Rformat (5)
Stage-2. In Stage-2, the model is trained on a mix of an-
swerable and unanswerable questions. This stage introduces
a refusal reward, Rrefusal , to teach the model to identify when
provided documents lack sufficient information to answer a
question.
Rrefusal =

0 ifanswerable andrscore>0.85
0.5 ifanswerable andrscore<0.85
rscore ifunanswerable andrscore>0.85
0 ifunanswerable andrscore<0.85(6)
where rscore refers to the refusal score that is calculated
using a fuzzy matching algorithm, producing a value between
0 and 1, with values closer to 1 indicating a higher similarity
to the gold standard refusal response. The final reward for
Stage-2 is the sum of Stage-1 Rewards with the addition of
Rrefusal .
The hierarchical rewards can be visualized in Algorithm 1.

Figure 1: Overview of the Ground-GRPO framework for reward-guided reinforcement learning in retrieval-augmented genera-
tion (RAG) settings. Given a user query, the Policy Model generates a set of candidate outputs O1, . . . , O N, each of which is
evaluated by a Reference Model to compute three distinct reward signals: (1) EM Reward , which checks for exact matches
against ground-truth answers and provides a score based on the number of correct hits; (2) Citation Reward , which uses natural
language inference (NLI) to verify whether the cited documents support the predicted answer; and (3) Refusal Reward , which
incentivizes the model to appropriately abstain from answering unanswerable questions based on the evidence provided. The
computed rewards R1, . . . , R Nare aggregated using Group Computation to update the policy via reinforcement learning. The
lower section of the figure illustrates examples for answerable and unanswerable queries, showing the model’s internal reasoning,
selected answer, and corresponding reward computations. For answerable queries, both EM and citation rewards are used, while
for unanswerable queries, a correct refusal is rewarded. Ground-GRPO thus encourages grounded, precise, and abstaining
behavior in RAG systems.
Algorithm 1: Hierarchical Rewards
Require: Policy Model M
1:Input: Questions Q, Task instruction I, Documents D,
Generated response S
2:Initialize reward R= 0
3:R+=Rtag_count
4:ifRformat= 1andRtag_count = 1then
5: R+=Rformat
6: ifis_answerable( Q, D )then
7: ifrscore(S)<0.85then
8: R+=Rrefusal
9: R+=Rcorrectness
10: else if not is_answerable( Q, D )then
11: R+=Rrefusal
12:Return R
3 Experimental Setup
Training Data We train our models using samples from
the Trust-Align SFT dataset, which is constructed using ques-
tions from the ASQA, QAMPARI, and ELI5 datasets, paired
with documents retrieved from web sources for each question.For Stage 1 training, we select 100 answerable questions
from each of the three datasets, focusing solely on samples
where the answer can be found within the provided docu-
ments. For Stage 2 training, we use a larger dataset of 1,000
samples per subset (3,000 total), composed of an equal mix
of answerable and unanswerable questions.
3.1 Training Algorithm
We train our models using the GRPO algorithm, as imple-
mented in the TRL library. Each training sample is associated
with 8 generated responses, and training is conducted for 4
epochs. We use a global batch size of 384, a learning rate of
1.0×10−5, a cosine learning rate scheduler, and a warm-up
ratio of 0.1.
3.2 Models and Baselines
Our main experiments use three backbone models:
LLaMA3.1-8B, Qwen3-4B, and Qwen3-8B (Grattafiori and
et al. 2024; Team 2025). These models were selected because
both reasoning and instruction-tuned variants are publicly
available. Notably, the Qwen3 series supports hybrid genera-
tion modes—with and without explicit reasoning—making it
suitable for studying the effects of reasoning supervision. For

AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score98.7
62.5
42.569.5
58.299.6
51.2
40.474.9
55.589.6
66.5
58.2
49.758.183.0
55.464.584.4
68.1Ideal AR: 64.3%Qwen-3-4b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score97.1
31.9
26.8
4.321.099.8
0.523.121.1
14.978.6
10.649.8
43.4
34.676.0
42.152.8
39.244.7
Ideal AR: 29.5%Qwen-3-4b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.0
25.0
18.536.2
26.698.2
25.5
19.631.5
25.679.2
30.638.9
29.833.160.9
31.051.4
43.341.9
Ideal AR: 20.7%Qwen-3-4b - ELI5
Prompt (Instruct)
Ground-GRPO (Instruct)
Prompt (Reasoning)
Ground-GRPO (Reasoning)
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.0
63.2
42.074.7
60.070.8
51.766.788.0
68.887.4
67.4
62.8 62.864.372.9
61.966.184.5
70.8
Ideal AR: 64.3%Qwen-3-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score95.0
38.7
30.1
3.023.952.5
13.968.0
50.0
44.076.5
8.952.5
43.4
34.952.3
36.268.6
37.347.4
Ideal AR: 29.5%Qwen-3-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.3
24.1
18.134.8
25.734.2
26.364.273.7
54.780.2
30.038.7
29.032.552.5
32.055.9
41.443.1
Ideal AR: 20.7%Qwen-3-8b - ELI5
AR (%) F1_AC F1_GR F1_GC TRUST020406080100120Score
1.53.028.686.5
39.4100.0
47.5
39.177.0
54.597.4
56.2
46.356.2
52.963.2
54.467.582.5
68.1Ideal AR: 64.3%LLaMA-3.1-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100120Score
3.95.948.6
20.224.9100.0
0.922.820.2
14.681.8
17.346.3
41.2
34.953.1
37.368.4
43.249.6
Ideal AR: 29.5%LLaMA-3.1-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100120Score
0.0 0.044.2
0.014.7100.0
15.617.140.5
24.495.0
23.0 23.6 23.2 23.247.4
32.358.7
24.338.4
Ideal AR: 20.7%LLaMA-3.1-8b - ELI5Reasoning vs Instruct Models Performance Comparison
(Red line shows ideal AR rate)Figure 2: Performance comparison between reasoning and instruct models across three model families (Qwen-3-4b, Qwen-3-8b,
LLaMA-3.1-8b) and three datasets (ASQA, QAMPARI, ELI5). Four configurations are compared: Prompt (Instruct) in dark blue,
Ground-GRPO (Instruct) in light blue, Prompt (Reasoning) in dark orange, and Ground-GRPO (Reasoning) in light orange.
Reasoning models consistently outperform their instruct counterparts on grounding metrics ( F1GR) and trust scores ( TRUST ),
demonstrating superior reliability in handling answerable and unanswerable questions. The Ground-GRPO training objective
further enhances performance for both instruct and reasoning variants, with Ground-GRPO (Reasoning) achieving the best
overall results across most metrics. Notable improvements are observed in grounded refusals and trust scores, validating the
effectiveness of reasoning capabilities for responsible question answering. F1AC:= Answer Correctness F1; F1GR:= Grounded
Refusals F1; F1GC:= Grounded Citations F1; TRUST :=Trust-Score .
LLaMA3.1-8B, we use the DeepSeek R1 distilled version as
the reasoning-tuned variant (DeepSeek-AI 2025). To further
strengthen the generality of our findings, we include addi-
tional results using the Gemma2-9B model for instruction-
tuned experiments (Team 2024).
As baselines, we adopt the Trust-Align SFT and DPO
datasets, which consist of approximately 10K and 17K sam-
ples, respectively. These externally sourced, distilled pref-
erence datasets train instruction-tuned models and serve
as benchmarks against which we compare our proposed
Ground-GRPO . In contrast to previous work that hasn’t
leveraged reasoning-enhanced models for groundedness,
our baseline comparisons focus exclusively on instruction-
tuned variants. The Trust-Align dataset offers a distilled,
high-quality alignment resource comprising about 19K care-
fully curated examples. Each sample pairs a “preferred” re-
sponse—grounded in retrieved documents with proper cita-
tions—with an “unpreferred” counterpart, which may exhibithallucination, mis-citation, over-responsiveness, or unwar-
ranted refusal. Data was constructed by sampling questions
from benchmarks like ASQA, QAMPARI, and ELI5; retriev-
ing supporting documents; and generating responses (both
positive and negative) using strong models such as GPT -4.
The resulting Trust-Align dataset is used by Song et al.
(2025b) to align LLMs for grounded response generation
using SFT and DPO methods. We consider Trust-Align
baselines as distillation approaches ( Figure 7).
3.3 Evaluation Benchmark
For evaluation, we use the Trust Score test set. The bench-
mark was constructed using the test splits of the ASQA, ELI5,
and QAMPARI datasets, following the same method as the
Trust-Align dataset. To assess the models’ robustness, the
benchmark also includes samples from ExpertQA, enabling
evaluation on out-of-domain (OOD) questions

3.4 Evaluation Metrics
Following Song et al., we use the Trust-Score metric
to evaluate the overall groundedness of model responses.
Trust-Score is computed as the average of three key
components:
1.Answer Correctness (F1 AC): Assesses the factual cor-
rectness of the model’s answer by comparing it against a
list of ground-truth answers (can have more than one per
question).
2.Grounded Refusals (F1 GR): Measures the model’s ability
to detect when a question is unanswerable based on the
provided documents and to appropriately refuse to answer.
3.Grounded Citations (F1 GC): Evaluates the quality of
citations by measuring the precision and recall of evi-
dence linked to statements in the response. Responses are
penalized for including unsupported claims, redundant
citations, or failing to cite necessary evidence.
Additionally, we report the Answer Ratio (AR), which mea-
sures the proportion of questions answered by the model.
Although we do not directly compare models based on this
metric, we expect models to align with the ideal AR for each
dataset. The AR is computed as:
AR=#questions answered
#total questions in the dataset
Large deviations from the ideal AR may indicate that a model
is either overly responsive or overly cautious in answering
questions. Note that this behavior of the model is captured in
F1GR.
The detailed formulas for metrics can be found in the Ap-
pendix A.
4 Results
4.1 Reasoning Promotes Grounded Responses
Figure 2 demonstrates that reasoning models1consistently
outperform their instruct counterparts across the ASQA,
QAMPARI, and ELI5 datasets. Specifically, when prompted
in a zero-shot setting (i.e., without any additional fine-tuning),
reasoning models achieve higher Trust-Score scores in 8
out of 9 configurations. This performance gap is still present
even after training with Ground-GRPO .
A deeper examination of the individual components of
Trust-Score reveals that the most substantial gains come
from improvements in F1GR. This suggests that reasoning en-
hances the model’s ability to identify unanswerable questions
based on provided documents and to refuse appropriately.
Notably, this pattern holds across both prompting ( ↑13.1%)
and GRPO ( ↑20.4%) across all configurations, emphasizing
the general utility of explicit reasoning in promoting more
grounded responses.
1Models with internal thought processes trained to produce high-
quality reasoning traces.TL;DR of Reasoning Promotes Grounded Responses
•Reasoning improves the Groundedness of Models
across different datasets
•Reasoning is especially helpful in enhancing the
model’s ability to identify and refuse unanswerable
questions
4.2 Training with RL Improves Groundedness
Figure 2 further indicates that training with GRPO leads to
substantial gains in groundedness across all model types
and datasets. For each of the three models evaluated on
ASQA, QAMPARI, and ELI5, Trust-Score improves
consistently following GRPO training. The gains are espe-
cially pronounced in reasoning models, which improve by
an average of 11.5 points, compared to 7.0 points for their
instruct counterparts.
A breakdown of the Trust-Score subcomponents re-
veals differing patterns of improvement. For instruct models,
the most notable gains are in F1GC(↑16.4%), suggesting
that GRPO helps these models better justify their responses
with minimally sufficient citations. In contrast, reasoning
models exhibit broad improvements across all three submet-
rics: F1AC(↑8.0%),F1GR(↑15.2%), and F1GC(↑11.3%).
These results suggest that reasoning-capable models are bet-
ter positioned to benefit from GRPO, learning not only to cite
sources but also to assess answerability and provide more
accurate and grounded responses overall.
Across all datasets, excessively high AR% often corre-
lates with lower refusal-grounding (F1 GR) and citation qual-
ity (F1 GC), suggesting that blind answering harms grounded
performance, while overly low AR% (e.g. LLaMA -3.1 In-
struct) forfeits recall entirely. Qwen -3 models under the In-
struct setting exhibit near-ceiling AR% on ASQA, QAM-
PARI, and ELI5, but this comes at the expense of low F1 GR
and F1 GC, especially on QAMPARI (4.31%/26.82%) and
ELI5 (36.22%/18.54%). In contrast, toggling to Reason-
ing prompts reduces AR% by 10–20 points while boosting
F1GRby 10–20 points and F1 GCby 5–25 points across all
datasets, yielding net TRUST gains of 5–10 points. Applying
Ground-GRPO further amplifies these effects across nearly
all models and settings (e.g., for Qwen -3-8b, F1 GRon ASQA
jumps from 41.97 to 66.67 and F1 GCfrom 74.74 to 87.97), al-
beit sometimes at the cost of AR%. One notable exception is
Qwen -3-4b under the Instruct setting, where Ground -GRPO
actually lowers F1 AC(62.50->51.25), F1 GR(42.53->40.42)
and overall TRUST (58.19 →55.53), indicating that ground-
ing in this case corrodes correctness rather than bolstering
it on ASQA. LLaMA -3.1 Instruct, which initially answers
almost nothing (AR% in the range of 1–4), flips to full cov-
erage (100% AR) under Ground-GRPO but sees only mod-
est F1 GR/F1 GCof 22–39%, indicating that forced answer-
ing without reasoning degrades grounded fidelity. Finally, in
the Reasoning setting, Ground-GRPO yields the most bal-
anced trade-off: AR% sits in the 50–80 range, F1 ACremains
within 5 points of prompt-only baselines, and F1 GR/F1 GC
improvements exceed 15 points in several cases, driving
10–15 point TRUST increases. These patterns demonstrate

AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score96.6
58.8
49.179.9
62.692.1
62.7
57.079.0
66.283.0
55.464.584.4
68.1Ideal AR: 64.3%Qwen-3-4b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score91.5
41.4
37.435.838.276.3
43.153.0
40.145.476.0
42.152.8
39.244.7
Ideal AR: 29.5%Qwen-3-4b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score94.0
26.5 25.539.3
30.483.8
27.235.639.4
34.160.9
31.051.4
43.341.9
Ideal AR: 20.7%Qwen-3-4b - ELI5
Stage 1 only
Stage 2 only
Stage 1 + 2
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score98.8
67.3
44.978.9
63.787.0
67.8
63.378.6
69.972.9
61.966.184.5
70.8
Ideal AR: 64.3%Qwen-3-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.8
31.0
27.532.530.371.2
43.656.4
43.047.752.3
36.268.6
37.347.4
Ideal AR: 29.5%Qwen-3-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.5
27.1
18.436.4
27.378.2
30.440.8
34.1 35.152.5
32.055.9
41.443.1
Ideal AR: 20.7%Qwen-3-8b - ELI5
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score97.9
55.4
44.664.1
54.797.0
54.7
46.868.6
56.763.2
54.467.582.5
68.1Ideal AR: 64.3%LLaMA-3.1-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score91.7
29.434.932.6 32.388.0
26.239.0
35.033.453.1
37.368.4
43.249.6
Ideal AR: 29.5%LLaMA-3.1-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score98.7
22.418.9 18.519.997.7
22.620.123.722.147.4
32.358.7
24.338.4
Ideal AR: 20.7%LLaMA-3.1-8b - ELI5Stage-wise Training Ablation Study Across Models and DatasetsFigure 3: Stage-wise training ablation study across three models (Qwen-3-4b, Qwen-3-8b, LLaMA-3.1-8b) and three datasets
(ASQA, QAMPARI, ELI5). Each subplot shows performance comparison between three training strategies: Stage 1 only (blue),
Stage 2 only (gray), and Stage 1 + 2 combined (green). Stage 1 + 2 training consistently improves grounding metrics ( F1GR)
across all model-dataset combinations. The combined training approach achieves the highest trust scores ( TRUST ) in most cases,
indicating better overall reliability.
that combining reasoning with Ground-GRPO consistently
optimizes grounded correctness and citation quality across
diverse datasets (see Figure 2).
TL;DR of RL Improves Groundedness
•Ground-GRPO consistently improves overall
groundedness ( Trust-Score ) across all model
types and QA datasets.
•Reasoning models benefit more broadly, with gains
across answer correctness, grounded refusal, and
citation quality.
•Improvements in grounding ( F1GC,F1GR) some-
times come at the cost of answer correctness
(F1AC), especially for short-form QA or weaker
base models.
•Applying Ground-GRPO on reasoning models
yields more balanced trade-offs and stronger over-
all alignment.4.3 Decomposing Training Into Stages Improves
Learning of Grounded Objectives
Generating grounded responses involves three distinct sub-
tasks: (1) determining whether a question is answerable given
the evidence, (2) extracting the correct answer when it is, and
(3) providing minimally sufficient citations to support the an-
swer. When these objectives are trained jointly using GRPO,
the resulting reward signal can be noisy, making it difficult
for the model to optimize all three behaviors simultaneously.
To address this, we investigate a two-stage training strat-
egy. In Stage 1, we train the model on a smaller dataset (300
samples) of answerable questions to focus on learning answer
extraction and citation generation. In Stage 2, we introduce
a larger, mixed dataset (3000 samples) containing both an-
swerable and unanswerable questions, training the model to
additionally learn to judge answerability.
Figure 3 shows that this two-stage curriculum (Stage 1
+ Stage 2) consistently improves groundedness compared
to either stage alone. Specifically, it outperforms Stage 2-
only training in 7 out of 9 model-dataset configurations.

Table 1: Results of applying Ground-GRPO to distilled versions of LLaMA-3.1, Qwen-3, and Gemma-2 models on the ASQA,
QAMPARI, and ELI5 datasets. The distillation is performed using SFT or DPO on the Trust-Align dataset, itself a distilled
dataset from GPT-4. Ground-GRPO is applied to these distilled variants, and the results show that it is beneficial only for
longform QA datasets such as ELI5.
Model TypeASQA (610 answerable, 338 unanswerable) QAMPARI (295 answerable, 705 unanswerable) ELI5 (207 answerable, 793 unanswerable)
AR (%) F1 AC F1GR F1GC TRUST AR (%) F1 AC F1GR F1GC TRUST AR (%) F1 AC F1GR F1GC TRUST
Qwen-3
-4bGround-GRPO (Instruct) 99.58 51.25 40.42 74.91 55.53 99.80 0.46 23.10 21.15 14.90 98.20 25.51 19.63 31.52 25.55
SFT 72.78 46.88 61.34 75.76 61.33 32.73 45.41 71.61 40.78 52.60 34.90 16.31 58.65 34.66 36.54
SFT→Ground-GRPO 68.04 47.22 62.88 81.94 64.01 57.00 29.83 61.60 36.93 42.79 29.00 20.32 61.85 51.71 44.62
Ground-GRPO →SFT 75.11 44.68 55.28 71.69 57.22 12.30 22.97 57.66 38.00 39.54 41.70 11.65 50.85 22.73 28.41
SFT→DPO 65.61 50.25 64.77 79.60 64.88 29.20 37.82 71.67 42.99 50.83 26.80 20.49 61.76 38.01 40.09
SFT→DPO→Ground-GRPO 48.31 44.00 61.84 86.79 64.21 46.80 39.32 68.32 44.00 50.55 20.70 20.29 64.36 59.18 47.94
Ground-GRPO →SFT→DPO 67.62 45.55 54.06 75.74 58.45 15.20 18.79 57.50 39.02 38.44 37.40 12.97 51.85 24.68 29.83
Qwen-3
-8bGround-GRPO (Instruct) 70.78 51.73 66.67 87.97 68.79 52.50 13.90 67.96 49.98 43.95 34.20 26.29 64.22 73.71 54.74
SFT 67.72 63.86 66.37 80.71 70.31 29.50 56.61 75.00 50.85 60.82 15.80 25.84 64.31 38.68 42.95
SFT→Ground-GRPO 66.14 52.37 66.86 89.32 69.52 50.50 44.00 67.50 40.78 50.76 38.80 25.21 63.99 53.38 47.53
Ground-GRPO →SFT 84.81 49.70 57.16 72.92 59.93 22.60 33.40 66.91 39.27 46.52 37.80 14.25 53.01 25.36 30.87
SFT→DPO 52.43 63.22 64.94 76.97 68.38 31.90 51.79 75.09 53.20 60.03 7.90 21.68 58.79 41.97 40.81
SFT→DPO→Ground-GRPO 44.09 47.60 63.03 94.92 68.52 47.20 42.76 70.29 47.05 53.37 29.40 25.35 67.64 62.86 51.95
Ground-GRPO →SFT→DPO 77.32 52.11 58.00 80.49 63.54 19.70 20.33 65.77 43.49 43.19 32.80 14.02 55.22 27.64 32.29
LLaMA-3.1
-8bGround-GRPO (Instruct) 100.00 47.46 39.15 77.02 54.54 100.00 0.93 22.78 20.20 14.63 100.00 15.63 17.15 40.47 24.42
SFT 72.15 53.52 64.71 82.80 67.01 33.70 54.75 72.24 47.26 58.08 21.10 21.77 65.82 44.81 44.13
SFT→Ground-GRPO 56.65 44.96 65.24 82.80 64.33 40.90 34.38 67.56 40.93 47.62 34.00 22.36 63.89 46.31 44.19
Ground-GRPO →SFT 68.35 36.74 55.12 54.97 48.94 3.68 6.15 47.06 32.61 28.61 26.00 6.64 53.21 17.94 25.93
SFT→DPO 33.65 45.48 58.95 91.94 65.46 28.10 49.31 73.42 55.04 59.26 4.60 17.26 56.79 69.73 47.93
SFT→DPO→Ground-GRPO 31.22 36.13 56.45 87.50 60.03 44.60 35.63 66.02 40.39 47.35 17.70 18.14 62.29 54.52 44.98
Ground-GRPO →SFT→DPO 56.33 36.56 51.51 60.96 49.68 20.30 20.08 58.82 27.74 35.55 15.40 7.20 52.85 19.53 26.53
Gemma-2
-9bGround-GRPO (Instruct) 100.00 50.01 39.15 80.19 56.45 100.00 35.98 22.78 22.21 26.99 100.00 14.83 17.15 48.35 26.78
SFT 75.32 56.12 66.95 85.12 69.40 36.90 52.71 76.10 50.58 59.80 25.20 23.09 65.08 48.87 45.68
SFT→Ground-GRPO 47.15 49.20 63.55 89.48 67.41 49.30 42.13 66.91 44.57 51.20 26.10 21.58 65.69 64.37 50.55
Ground-GRPO →SFT 68.35 37.85 51.34 51.86 47.01 7.50 12.97 50.26 24.78 29.34 32.40 7.34 50.13 12.38 23.29
SFT→DPO 62.45 57.04 69.54 88.96 71.85 33.30 47.45 75.86 55.13 59.48 14.80 24.32 63.53 67.11 51.65
SFT→DPO→Ground-GRPO 46.73 48.60 65.51 89.69 67.93 49.70 41.92 67.18 39.29 49.46 25.50 21.50 65.38 69.90 52.26
Ground-GRPO →SFT→DPO 58.33 45.90 55.63 55.01 52.18 6.40 6.69 48.91 27.97 27.85 29.60 13.85 53.12 14.09 27.02
On average, two-stage training yields a 6.8% gain over
Stage 2-only and a 12.5% gain over Stage 1-only training in
Trust-Score . The most significant improvement comes
inF1GR(Grounded Refusal), which increases by 15.8% on
average, suggesting that early training in grounded answer
generation, followed by a broader training phase, helps mod-
els better integrate all aspects of groundedness.
TL;DR of Staged Training
•Decomposing training into two stages yields more
stable and targeted learning, helping models better
optimize for grounded objectives.
•Stage 1 focuses on grounded answer generation,
while Stage 2 introduces answerability, resulting
in stronger integration of all three behaviors.
•This strategy especially improves Grounded Re-
fusal , with a 15.8% average gain, indicating better
identification of unanswerable questions.
4.4 Distillation Helps
Previous work in Trust-Align demonstrated that dis-
tilling high-quality responses from GPT-4 significantly im-
proves the groundedness and citation quality of instruction-
following models. They constructed a supervised fine-tuning
(SFT) dataset with 10K samples and a Direct Preference
Optimization (DPO) dataset with approximately 17K sam-
ples, both based on GPT-4 outputs. Models trained on this
distilled data using SFT and DPO showed substantial gainsin response trustworthiness. Building on this, we investigate
how reinforcement learning via on-policy GRPO compares to
distillation-based approaches, and whether distillation before
or after GRPO further improves model performance. Given
that the Trust-Align Dataset was built for instruct models, we
only investigate the impact of distillation on instruct variants
of models.
RL vs. Distillation Table 1 shows that instruct models
trained with the Trust-Align SFT and DPO pipeline outper-
form those trained solely with Ground-GRPO . This perfor-
mance gap may stem from several factors. First, instruct mod-
els appear less capable of effectively leveraging on-policy
learning compared to reasoning models. Second, the Trust-
Align approach benefits from a substantially larger training
corpus—over 27K samples—while our GRPO experiments
use only 3K examples. Finally, the high-quality supervision
signal provided by GPT-4 through distillation may be espe-
cially valuable for instruction-following models, particularly
when their initial performance is insufficient to reliably obtain
reward signals during RL training. These observations high-
light the complementary strengths of distillation and GRPO
for instruction-tuned models, motivating us to investigate
whether applying GRPO on top of distillation can further
enhance groundedness and citation quality.
Applying Ground-GRPO to Distilled Models As shown
in Table 1, combining distillation with Ground-GRPO leads
to notable performance improvements across all three QA
benchmarks. When comparing distilled models trained with

300 1000 3000020406080100Performance Score
Ideal AR: 64.3%ASQA
300 1000 3000
Training Data Size020406080100
Ideal AR: 29.5%QAMPARI
300 1000 3000020406080100
Ideal AR: 20.7%ELI5AR (%) F1_AC F1_GR F1_GC TRUSTFigure 4: Performance trends across different training data sizes for three question-answering datasets. The analysis reveals
contrasting patterns: while Answer Ratio (AR) consistently decreases with larger training sets across all datasets, most other
metrics show improvement. F1 GR(F1 score for grounded responses) demonstrates the most consistent positive correlation with
data size, particularly pronounced in ASQA and QAMPARI datasets. F1 GC(F1 score for grounded citations) shows steady
improvement in ASQA and QAMPARI, but remains relatively stable in ELI5. F1 AC(Answer Correctness) exhibits distinct
behaviors: ASQA maintains stable performance ( 52%) regardless of training size, QAMPARI shows substantial improvement
from 300 to 1,000 samples (25.96 →37.84) before plateauing, and ELI5 demonstrates gradual but consistent gains (22.40 →28.38).
The TRUST metric exhibits moderate improvement with increased data size across all datasets. Notably, ELI5 displays the
most volatile behavior, with AR showing the steepest decline and F1 GCremaining nearly constant, suggesting dataset-specific
characteristics may influence the effectiveness of scaling training data.
Ground-GRPO (SFT→Ground-GRPO ) against models
trained with Ground-GRPO alone, we observe significant
gains in Trust-Score : a 7.5% improvement on ASQA,
23.0% on QAMPARI, and 14.8% on ELI5. These results
suggest that distillation serves as a strong initialization for
reinforcement learning, improving the model’s ability to
earn rewards and enabling more effective alignment toward
grounded response generation.
However, applying Ground-GRPO to already distilled
models (SFT and SFT →DPO) shows mixed results when
compared to the distilled models. On long-form QA tasks
such as ELI5, which demand detailed, explanatory answers
grounded in retrieved documents, Ground-GRPO improves
performance by 4.2% on average. Specifically, SFT mod-
els improve by 4.4%, and SFT →DPO models by 4.2%.
In contrast, on short-form QA tasks like QAMPARI, which
require concise, list-style answers with exact formatting, per-
formance declines significantly: SFT models drop by 9.7%,
and SFT →DPO models by 7.2%.
A closer look reveals that the most significant degra-
dation occurs in the Answer Correctness component of
Trust-Score , with an average drop of 6.2%, and up to
10% on QAMPARI. On ASQA, We also observe an aver-
age of 6.6% from the SFT models and a 9.9% drop from
the SFT →DPO models when they are further trained with
Ground-GRPO . This decline appears to stem from two main
factors. First, both ASQA and QAMPARI benchmarks rely
on exact string matching to assess answer correctness, mak-ing them particularly sensitive to minor formatting differ-
ences such as extra punctuation or slight variations in phras-
ing. While the model’s responses may be factually correct,
such surface-level mismatches can lead to lower scores. Sec-
ond, the reward function used in Ground-GRPO is relatively
strict, which may cause the model to adopt a more conser-
vative answering style returning only one or two confident
answers even when the documents contain more, thereby
reducing the Answer Correctness score.
Importantly, since ASQA and QAMPARI feature relatively
short, factoid-style answers, the benefits of Ground-GRPO
appear limited when applied on top of already well-
performing distilled models. In these settings, the task of
extracting concise factual answers is relatively straightfor-
ward, and distillation alone often suffices. As a result, the
additional optimization from Ground-GRPO can introduce
unnecessary cautiousness or formatting inconsistencies, ul-
timately hurting performance. In contrast, long-form tasks
like ELI5 demand more complex reasoning, nuanced answer
synthesis, and careful grounding, which are areas where the
alignment signal from Ground-GRPO is more impactful.
These findings suggest that while GRPO complements dis-
tillation in open-ended, generative tasks, it may offer dimin-
ishing returns, or even degrade performance, on structured,
short-form QA tasks where distillation is already highly ef-
fective.
Nevertheless, we observe an average gain of 2.4% in F1GC
across models and datasets, indicating that Ground-GRPO

AR (%) F1_AC F1_GR F1_GC TRUST %Align020406080100ScoreIdeal AR: 64.3%88.2
57.061.384.1
67.566.283.0
55.464.584.4
68.1
65.3Qwen-3-4b - ASQA
w/ process reward
w/o process reward
AR (%) F1_AC F1_GR F1_GC TRUST %Align01020304050607080Score
Ideal AR: 29.5%74.9
44.654.3
42.247.057.476.0
42.152.8
39.244.755.0Qwen-3-4b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST %Align01020304050607080Score
Ideal AR: 20.7%73.7
30.144.3
41.0
38.567.5
60.9
31.051.4
43.341.955.9Qwen-3-4b - ELI5
AR (%) F1_AC F1_GR F1_GC TRUST %Align020406080ScoreIdeal AR: 64.3%78.4
53.966.380.8
67.072.5
63.2
54.467.582.5
68.1
57.0LLaMA-3.1-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST %Align010203040506070Score
Ideal AR: 29.5%63.9
39.661.6
44.548.653.0 53.1
37.368.4
43.249.6
39.6LLaMA-3.1-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST %Align0102030405060Score
Ideal AR: 20.7%51.0
28.155.8
20.834.956.6
47.4
32.358.7
24.338.443.7LLaMA-3.1-8b - ELI5Process Supervision Ablation Study Across Models and DatasetsFigure 5: Process supervision ablation study across two language models and three question-answering datasets. The comparison
between models with and without process reward supervision reveals dataset-dependent effects. Notably, F1 GR(grounded
responses) demonstrates contrasting behaviors: while Qwen-3-4b shows slight decreases with process rewards in ASQA and
QAMPARI, LLaMA-3.1-8b exhibits significant improvements in QAMPARI (61.63 vs 68.44) but decreases in other datasets.
The %Align metric, indicating alignment between reasoning steps and final answers, generally favors process reward supervision,
particularly for Qwen-3-4b across all datasets and LLaMA-3.1-8b in ASQA and ELI5. These results suggest that process
supervision benefits vary significantly by model architecture and dataset complexity, with reasoning alignment showing the
most consistent improvements .
improves the model’s ability to provide accurate, non-
redundant citations. This suggests that even in cases
where answer correctness plateaus or slightly declines,
Ground-GRPO can still enhance the faithfulness and clarity
of citation attribution.
Although applying Ground-GRPO to distilled models
generally leads to improved performance on ELI5, LLaMA-
3.1-8b is a notable exception. The reason for this remains un-
clear. One possible explanation is that LLaMA-3.1-8b has un-
dergone a different post-training process compared to Qwen
models, which were fine-tuned using reasoning-focused, RL-
based methods. A similar trend is observed with Gemma2-9b,
where the performance gain on ELI5 is minimal. We also see
this pattern in another longform QA dataset, ExpertQA, as
discussed in Section 4.5.
Finally, we also compare the ordering of distillation and
Ground-GRPO training. Across all models and datasets, we
find that applying distillation before Ground-GRPO ( SFT
→Ground-GRPO and SFT →DPO→Ground-GRPO )
consistently outperforms the reverse order ( Ground-GRPO
→SFT and Ground-GRPO →SFT→DPO) by a mar-
gin of 15%. This result aligns with our earlier observations:distillation provides a strong, high-quality initialization that
enables the model to better leverage the reward signal during
reinforcement learning. In contrast, when Ground-GRPO
is applied first, the model may not yet possess the necessary
grounding or fluency to benefit fully from subsequent super-
vised fine-tuning, leading to suboptimal alignment. These
findings reinforce the importance of sequencing in training
pipelines and highlights that distillation, when used as a pre-
cursor to reinforcement learning, provides a more effective
foundation for improving trustworthiness in instruct models.

TL;DR of Ground-GRPO with Distilled Models
•Ground-GRPO complements distillation on long-
form QA datasets such as ELI5 and ExpertQA.
•On shortform QA datasets like ASQA and QAM-
PARI, Ground-GRPO offers no noticeable bene-
fits when applied to distilled models.
•The effectiveness of Ground-GRPO with dis-
tilled models depends on the underlying backbone
LLM. It performs better with Qwen-3 models com-
pared to LLaMA-3.1 and Gemma-2. This may be
attributed to Qwen-3 models receiving reasoning-
focused RL-based post-training, unlike the latter
two.
•Ordering of distillation and Ground-GRPO mat-
ters. Applying Distillation Before Ground-GRPO
shows a significant performance increase com-
pared to applying Ground-GRPO first.
4.5 Out-of-Distribution (OOD) Performance
Table 2 presents results on the ExpertQA benchmark, evaluat-
ing the generalization ability of various model configurations.
Across all three models’ reasoning variants, we observe that
Ground-GRPO with two-stage training (Stage 1 + Stage
2) consistently achieves higher TRUST scores compared to
models trained using only one stage, highlighting the benefits
of progressive alignment.
Moreover, reinforcement learning via Ground-GRPO
consistently enhances ExpertQA performance in 5 out
of 6 instruct model configurations. Notably, applying
Ground-GRPO on top of SFT or SFT →DPO leads to
substantial gains in F1 GC, up to 17% improvement, show-
ing that Ground-GRPO helps models better support their
claims with grounded evidence even in unfamiliar domains.
These improvements also translate to overall TRUST score
gains, such as a 10-point increase in the Qwen-3-4B instruct
variant when Ground-GRPO is applied on top of SFT →
DPO. Given that ExpertQA is a longform dataset, this find-
ing further supports our point that applying Ground-GRPO
over distilled models would greatly improve performance in
complex open-ended tasks.
In contrast, standard instruct and reasoning models with-
out alignment perform poorly on ExpertQA, despite high
answer rates (AR), suggesting overconfidence and a lack
of reliable refusal behavior. In contrast, the improved per-
formance of Ground-GRPO -trained and two-stage models
highlights their ability to more accurately recognize when
not to answer and to cite appropriately, a key requirement for
trustworthy generalization in OOD settings.
4.6 Data Size Ablation
We investigated the impact of Stage 2 dataset size in
Ground-GRPO on model performance with 3 different
dataset sizes: 300, 1000 and 3000 samples. As shown in
Figure 4, increasing the number of training samples in
Stage 2 resulted in a consistent improvement in overall
Trust-Score and across each of its individual compo-
nents. These results indicate that larger datasets provideTable 2: Results on ExpertQA. Ground-GRPO consistency
improves the performance when applied to base or distilled
model variants.
Model Type AR (%) F1 AC F1GR F1GC TRUST
Qwen-3
-4bInstruct Variant
Prompt (Instruct) 98.39 35.31 26.32 52.80 38.14
Ground-GRPO (Instruct) 98.34 32.94 26.29 46.27 35.17
SFT 35.04 20.48 63.95 42.84 42.43
SFT→Ground-GRPO 27.39 20.56 67.14 59.37 49.02
SFT→DPO 28.08 22.33 63.55 50.27 45.39
SFT→DPO→Ground-GRPO 16.78 24.67 67.62 75.62 55.97
Reasoning Variant
Prompt (Reasoning) 82.47 39.23 42.23 40.77 40.74
Ground-GRPO w/ Stage 1 only 94.26 35.41 31.46 49.98 38.95
Ground-GRPO w/ Stage 2 only 85.07 35.69 41.68 52.37 43.24
Ground-GRPO w/ Stage 1 + 2 68.51 33.90 52.86 52.29 46.35
Qwen-3
-8bInstruct Variant
Prompt (Instruct) 98.62 37.01 25.95 54.19 39.05
Ground-GRPO (Instruct) 29.14 29.58 70.63 82.54 60.92
SFT 19.59 33.03 68.41 58.85 53.43
SFT→Ground-GRPO 35.27 29.05 71.43 66.95 55.81
SFT→DPO 34.30 29.50 70.12 60.24 53.29
SFT→DPO→Ground-GRPO 20.01 26.08 68.87 77.26 57.40
Reasoning Variant
Prompt (Reasoning) 82.63 40.29 43.67 42.72 42.23
Ground-GRPO w/ Stage 1 only 99.42 36.74 25.94 50.83 37.83
Ground-GRPO w/ Stage 2 only 80.82 38.88 45.56 48.95 44.46
Ground-GRPO w/ Stage 1 + 2 52.70 39.64 60.92 57.09 52.55
LLaMA-3.1
-8bInstruct Variant
Prompt (Instruct) 90.36 36.24 34.26 48.16 39.55
Ground-GRPO (Instruct) 100.00 19.26 23.92 47.51 30.23
SFT 23.93 26.84 69.89 62.00 52.91
SFT→Ground-GRPO 23.84 24.10 70.32 58.76 51.06
SFT→DPO 7.01 17.27 57.25 75.05 49.86
SFT→DPO→Ground-GRPO 11.85 18.85 63.91 71.08 51.28
Reasoning Variant
Prompt (Reasoning) 97.97 33.02 26.59 23.48 27.70
Ground-GRPO w/ Stage 1 only 98.48 32.07 25.98 28.50 28.85
Ground-GRPO w/ Stage 2 only 97.69 31.80 27.01 32.96 30.59
Ground-GRPO w/ Stage 1 + 2 45.46 37.99 63.77 39.44 47.06
a stronger supervision signal for aligning models with
grounded behavior and that Ground-GRPO can scale ef-
fectively with additional training data.
4.7 Process Supervision Analysis
In some experiments, we observed that the model’s internal
reasoning was occasionally misaligned with its final answer.
In particular, there were instances where the reasoning chain
indicated that the model had found sufficient evidence to
answer the question, but the final response incorrectly re-
fused to answer, claiming that no relevant information was
found (see Appendix B for an illustrative example). This
mismatch suggests that the model’s decision-making during
generation is not always consistent across the reasoning and
final answer, highlighting the potential need for more explicit
process supervision.
To this end, we introduce a process reward using an NLI
model that evaluates whether the model’s reasoning supports
its final decision to answer or refuse:
Rprocess =nli_score (7)
Here, nli_score reflects the degree to which the rea-
soning trace entails the final decision (either answering or
refusing).
To measure the alignment between reasoning and final
output, we define a simple consistency metric:

AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.6
51.2
40.474.9
55.5100.0
57.1
39.178.6
58.3Ideal AR: 64.3%Qwen-3-4b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score99.8
0.523.121.1
14.9100.0
34.9
22.824.527.4Ideal AR: 29.5%Qwen-3-4b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score98.2
25.5
19.631.5
25.6100.0
20.3
17.140.3
25.9
Ideal AR: 20.7%Qwen-3-4b - ELI5
Ground-GRPO (with penalty)
w/o bad citation penalty
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score70.8
51.766.788.0
68.8100.0
52.2
39.179.0
56.8Ideal AR: 64.3%Qwen-3-8b - ASQA
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score52.5
13.968.0
50.0
44.0100.0
38.6
22.827.929.8 Ideal AR: 29.5%Qwen-3-8b - QAMPARI
AR (%) F1_AC F1_GR F1_GC TRUST020406080100Score
34.2
26.364.273.7
54.7100.0
21.8
17.138.8
25.9
Ideal AR: 20.7%Qwen-3-8b - ELI5Reward Design Ablation Study: With vs Without Bad Citation Penalty
(Red line shows ideal AR rate)Figure 6: Reward Design Ablation Study: Impact of Bad Citation Penalty on Model Performance. We compare Ground-
GRPO training with and without the bad citation penalty component across two model variants (Qwen-3-4b and Qwen-3-8b) on
three datasets. The red dashed lines indicate the ideal Answer Rate (AR) for each dataset, computed as the ratio of answerable to
total questions: ASQA (64.4%), QAMPARI (29.5%), and ELI5 (20.7%). Key findings: (1) Removing the bad citation penalty
consistently drives AR to near 100%, significantly overshooting the ideal rates and indicating over-answering behavior across
both models. (2) The impact on other metrics shows contrasting patterns between models : For Qwen-3-4b, removing the penalty
generally improves F1 ACand TRUST scores with minimal impact on F1 GR, while for Qwen-3-8b, it dramatically degrades F1 GR
performance (e.g., from 67.96 to 22.78 on QAMPARI) despite improving F1 AC. (3) These model-dependent effects suggest that
optimal reward configuration varies by model architecture and capacity, with larger models (8b) being more sensitive to citation
penalty removal. (4) The penalty serves different roles across models: acting as a mild regularizer for smaller models but as a
crucial constraint preventing performance collapse in larger models. This highlights the importance of model-specific reward
tuning in retrieval-augmented generation training.
%Align =1
NNX
i=11[NLI(ri, ai)→entailment ](8)
Figure 5 shows that incorporating this process reward re-
sults in a modest decrease in Trust-Score in 5 out of 6
configurations, with an average drop of 1.2%. However, we
observe a notable average improvement of 8% in %Align,
indicating that the NLI-based process reward improves the
consistency between reasoning and final answers. These find-
ings suggest that while process supervision introduces trade-
offs in overall groundedness, it holds promise for reducing
reasoning-answer misalignment. Future work could focus on
refining this approach to maintain or enhance groundedness
while promoting internal consistency.TL;DR of Process Supervision
•We discovered that RL-trained models sometimes
generate reasoning that contradicts the final an-
swer, indicating internal misalignment.
•We introduce a process reward based on NLI to
encourage consistency between reasoning and final
output.
•This improves reasoning-answer alignment by 8%
on average, but leads to a small (1.2%) drop in
overall Trust-Score .
4.8 Reward Design Ablation
We conducted an ablation study to examine the reward design
used in Ground-GRPO training, focusing on the impact of
the bad citation penalty. As shown in Figure 6, removing

this penalty has a substantial influence on model behavior.
First, we observe that eliminating the bad citation penalty
consistently drives the AR close to 100%, indicating a strong
tendency toward over-answering, even when the input docu-
ments do not support a grounded response. Second, the effect
on other evaluation metrics varies significantly by model size.
For the smaller Qwen-3-4B model, removing the penalty
slightly improves F1ACand overall Trust-Score scores,
with minimal effect on F1GC. In contrast, for the larger Qwen-
3-8B model, removing the citation penalty leads to a drastic
drop in F1GC(e.g., from 67.96 to 22.78 on QAMPARI), de-
spite modest gains in F1GC. These findings suggest that the
role of the bad citation penalty is model-dependent: it acts
as a light regularizer in smaller models, but as a critical con-
straint in larger models, where its absence can lead to citation
failure and collapse in groundedness. Overall, this highlights
the importance of model-specific reward tuning for grounded
response generation.
5 Related works
5.1 Attributable Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has become a foun-
dational approach for enhancing factual accuracy by ground-
ing LLM outputs in retrieved evidence (Karpukhin et al. 2020;
Lewis et al. 2021; Gao et al. 2023d). However, models can
still be misled by irrelevant passages, leading to hallucina-
tions and ungrounded outputs (Shi et al. 2023; Yoran et al.
2024; Xu, Shi, and Choi 2023).
5.2 Training-Based Grounded Generation
Approaches to enhancing grounding through training can be
categorized into training-free and training-based paradigms.
Training-free methods focus on citation reflexivity, such as
in-context learning with citation examples (Gao et al. 2023c),
post-hoc retrieval for attribution (Gao et al. 2023a; Li et al.
2024b), or even chain-of-thought prompting to improve ci-
tation alignment (Ji et al. 2024). Training-based methods
include supervised fine-tuning (SFT) on citation-rich data
(Asai et al. 2024; Slobodkin et al. 2024; Xia et al. 2024; Ye
et al. 2024) and preference-based learning via RLHF and
Direct Preference Optimization (DPO) (Ouyang et al. 2022;
Rafailov et al. 2024). Huang et al. use fine-grained rewards
and PPO to refine attribution, and Li et al. develop a modified
DPO framework to enhance citation control. Unlike these,
our method Trust-Align introduces a staged GRPO train-
ing that decomposes grounding behaviors—first optimizing
answer and citation quality, then refusal, without requiring
gold reasoning traces.
6 Recommendations
Based on our preliminary testing with a few LLMs, we pro-
vide the following recommendations to enhance the accu-
racy and grounding of RAG systems. We strongly encourage
readers to rigorously validate these recommendations across
different LLM families and model sizes.Recommendations
•Use reasoning models for RAG, as they improve
both answer generation accuracy and grounding.
•Design relevant reward functions and apply GRPO
for on-policy enhancement of LLM accuracy and
grounding in RAG. Since not all reward functions
benefit every model, perform model-specific eval-
uations.
•Knowledge distillation from larger models such as
GPT-4 can boost the accuracy and grounding of
RAG systems.
•Apply GRPO to distilled models for long-form
question answering. For short-form or simpler
questions, we do not recommend applying GRPO
to distilled models, as our experiments show no
improvement and, in some cases, a decline in per-
formance.
References
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. arXiv:2310.11511.
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. In The Twelfth International Con-
ference on Learning Representations .
Chen, M.; Li, T.; Sun, H.; Zhou, Y .; Zhu, C.; Wang, H.; Pan,
J. Z.; Zhang, W.; Chen, H.; Yang, F.; Zhou, Z.; and Chen, W.
2025. ReSearch: Learning to Reason with Search for LLMs
via Reinforcement Learning. arXiv:2503.19470.
DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Rea-
soning Capability in LLMs via Reinforcement Learning.
arXiv:2501.12948.
Gao, L.; Dai, Z.; Pasupat, P.; Chen, A.; Chaganty, A. T.; Fan,
Y .; Zhao, V .; Lao, N.; Lee, H.; Juan, D.-C.; and Guu, K. 2023a.
RARR: Researching and Revising What Language Models
Say, Using Language Models. In Rogers, A.; Boyd-Graber,
J.; and Okazaki, N., eds., Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , 16477–16508. Toronto, Canada:
Association for Computational Linguistics.
Gao, T.; Yen, H.; Yu, J.; and Chen, D. 2023b. Enabling
Large Language Models to Generate Text with Citations.
arXiv:2305.14627.
Gao, T.; Yen, H.; Yu, J.; and Chen, D. 2023c. Enabling
Large Language Models to Generate Text with Citations.
arXiv:2305.14627.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; and Wang, H. 2023d. Retrieval-augmented gener-
ation for large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Grattafiori, A.; and et al. 2024. The Llama 3 Herd of Models.
arXiv preprint arXiv:2407.21783. Includes Meta LLaMA -3.1
models (8B, 70B, 405B). Available via Hugging Face: https:
//huggingface.co/meta-llama/Llama-3.1-8B-Instruct.

Huang, C.; Wu, Z.; Hu, Y .; and Wang, W. 2024. Training
Language Models to Generate Text with Citations via Fine-
grained Rewards. arXiv:2402.04315.
Ji, B.; Liu, H.; Du, M.; and Ng, S.-K. 2024. Chain-of-Thought
Improves Text Generation with Citations in Large Language
Models. Proceedings of the AAAI Conference on Artificial
Intelligence , 38(16): 18345–18353.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y .; Ishii,
E.; Bang, Y . J.; Madotto, A.; and Fung, P. 2023. Survey
of Hallucination in Natural Language Generation. ACM
Computing Surveys , 55(12): 1–38.
Jiang, J.; Chen, J.; Li, J.; Ren, R.; Wang, S.; Zhao, W. X.;
Song, Y .; and Zhang, T. 2024. RAG-Star: Enhancing Deliber-
ative Reasoning with Retrieval Augmented Verification and
Refinement. arXiv:2412.12881.
Karpukhin, V .; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov,
S.; Chen, D.; and Yih, W.-t. 2020. Dense Passage Retrieval
for Open-Domain Question Answering. In Webber, B.; Cohn,
T.; He, Y .; and Liu, Y ., eds., Proceedings of the 2020 Confer-
ence on Empirical Methods in Natural Language Processing
(EMNLP) , 6769–6781. Online: Association for Computa-
tional Linguistics.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin,
V .; Goyal, N.; Küttler, H.; Lewis, M.; tau Yih, W.; Rock-
täschel, T.; Riedel, S.; and Kiela, D. 2021. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks.
arXiv:2005.11401.
Li, D.; Sun, Z.; Hu, B.; Liu, Z.; Hu, X.; Liu, X.; and Zhang,
M. 2024a. Improving Attributed Text Generation of Large
Language Models via Preference Learning. In Ku, L.-W.;
Martins, A.; and Srikumar, V ., eds., Findings of the Associ-
ation for Computational Linguistics ACL 2024 , 5079–5101.
Bangkok, Thailand and virtual meeting: Association for Com-
putational Linguistics.
Li, F.; Fang, P.; Shi, Z.; Khan, A.; Wang, F.; Feng, D.;
Wang, W.; Zhang, X.; and Cui, Y . 2025a. CoT-RAG: In-
tegrating Chain of Thought and Retrieval-Augmented Gen-
eration to Enhance Reasoning in Large Language Models.
arXiv:2504.13534.
Li, W.; Li, J.; Ma, W.; and Liu, Y . 2024b. Citation-Enhanced
Generation for LLM-based Chatbots. arXiv:2402.16063.
Li, X.; Dong, G.; Jin, J.; Zhang, Y .; Zhou, Y .; Zhu, Y .; Zhang,
P.; and Dou, Z. 2025b. Search-o1: Agentic Search-Enhanced
Large Reasoning Models. arXiv:2501.05366.
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright,
C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray,
A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens, M.;
Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and Lowe,
R. 2022. Training language models to follow instructions
with human feedback. arXiv:2203.02155.
Rafailov, R.; Sharma, A.; Mitchell, E.; Ermon, S.; Man-
ning, C. D.; and Finn, C. 2024. Direct Preference Opti-
mization: Your Language Model is Secretly a Reward Model.
arXiv:2305.18290.
Shi, F.; Chen, X.; Misra, K.; Scales, N.; Dohan, D.; Chi,
E. H.; Schärli, N.; and Zhou, D. 2023. Large Language Mod-
els Can Be Easily Distracted by Irrelevant Context. In Krause,A.; Brunskill, E.; Cho, K.; Engelhardt, B.; Sabato, S.; and
Scarlett, J., eds., Proceedings of the 40th International Con-
ference on Machine Learning , volume 202 of Proceedings of
Machine Learning Research , 31210–31227. PMLR.
Slobodkin, A.; Hirsch, E.; Cattan, A.; Schuster, T.; and Dagan,
I. 2024. Attribute First, then Generate: Locally-attributable
Grounded Text Generation. arXiv:2403.17104.
Song, H.; Jiang, J.; Min, Y .; Chen, J.; Chen, Z.; Zhao, W. X.;
Fang, L.; and Wen, J.-R. 2025a. R1-Searcher: Incentivizing
the Search Capability in LLMs via Reinforcement Learning.
arXiv:2503.05592.
Song, M.; Sim, S. H.; Bhardwaj, R.; Chieu, H. L.; Majumder,
N.; and Poria, S. 2025b. Measuring and Enhancing Trustwor-
thiness of LLMs in RAG through Grounded Attributions and
Learning to Refuse. arXiv:2409.11242.
Team, G. 2024. Gemma.
Team, Q. 2025. Qwen3 Technical Report. arXiv:2505.09388.
Xia, S.; Wang, X.; Liang, J.; Zhang, Y .; Zhou, W.; Deng, J.;
Yu, F.; and Xiao, Y . 2024. Ground Every Sentence: Improv-
ing Retrieval-Augmented LLMs with Interleaved Reference-
Claim Generation. arXiv:2407.01796.
Xu, F.; Shi, W.; and Choi, E. 2023. RECOMP: Improving
Retrieval-Augmented LMs with Compression and Selective
Augmentation. arXiv:2310.04408.
Ye, X.; Sun, R.; Arik, S. Ö.; and Pfister, T. 2024. Effective
Large Language Model Adaptation for Improved Grounding
and Citation Generation. arXiv:2311.09533.
Yoran, O.; Wolfson, T.; Ram, O.; and Berant, J. 2024. Making
Retrieval-Augmented Language Models Robust to Irrelevant
Context. arXiv:2310.01558.

A Metric Details
We report performance using Trust-Score , a composite metric that evaluates the trustworthiness of model responses across
three dimensions: response truthfulness, factual accuracy, and attribution groundedness.
A.1 Response Truthfulness
This component assesses whether the model correctly answers questions when evidence is available and appropriately refuses
when it is not. It is computed as a macro-average of two F1 scores:
Grounded Refusal [F1 GR]:We define two subsets: Agand¬Agas the ground truth sets of answerable and unanswerable
questions, and Arand¬Aras the sets of questions the model answers and refuses, respectively.
•F1ref: Measures the model’s ability to correctly refuse unanswerable questions.
•F1ans: Measures the model’s ability to correctly answer answerable questions.
These are computed via precision and recall and combined into:
F1GR=1
2(F1ref+F1_ans ) (9)
This score captures the balance between under-refusal and over-refusal behavior.
A.2 Factual Accuracy
This component evaluates whether the model’s answers are factually accurate and grounded in the provided evidence.
Answer Correctness [F1 AC]:For a question q, letAGbe the set of gold claims, ADthe claims supported by retrieved
documents, and ARthe claims generated in the model’s response. The calibrated answer correctness for qis:
ACq=|AG∩AD∩AR|
|AG∩AD|(10)
This restricts evaluation to claims verifiable from retrieved evidence. To aggregate across the dataset, we compute precision-
oriented ( PAC) and recall-oriented ( RAC) variants, depending on whether the denominator is the number of answered or
answerable questions. The combined F1 is:
F1AC=2PAC·RAC
PAC+RAC(11)
This metric discourages hallucination by rewarding answers that are both accurate and present in the evidence.
A.3 Attribution Groundedness
This component measures whether cited documents genuinely support the statements they are associated with.
Grounded Citation F1 [F1 GC]:Each response consists of statements Sand citations C. We assess:
•Recall : Whether the full set of cited documents supports each statement.
•Precision : Whether each individual citation contributes uniquely to the support of a statement.
We compute:
F1GC=2·Pcite·Rcite
Pcite+Rcite(12)
This score penalizes both spurious citations and incomplete attribution.
A.4 Overall Metric
We combine the three components into a single aggregate trustworthiness score:
Trust-Score =1
3(F1GR+F1AC+F1GC) (13)
This composite score enables consistent model ranking across correctness, calibration, and citation quality.

B Need for Process Supervision
In our analysis, we observed cases where the model’s final answer was a refusal, despite the reasoning trace indicating a clear
and confident answer. This mismatch between the model’s internal reasoning and final output suggests that the model may not be
using its reasoning trace to inform its final decision, highlighting a gap in alignment.
For instance, consider the following example:
Q:Who has the highest goals in world football?
Reasoning Trace: ... Since Bican’s is the official count, he’s the highest. Chitalu’s goals might not be official, so he’s not the
answer. So, the answer is Josef Bican with 805 goals, cited by RSSF and IFFHS.
Final Answer: I’m sorry, but I couldn’t find an answer to your question in the search results.
Despite clearly identifying the correct answer in the reasoning trace, the model ultimately refused to answer. This inconsistency
may stem from a lack of explicit supervision over how the reasoning trace should support the final decision. In such cases, the
model might learn to produce refusals in the final output without developing the proper reasoning chain to justify that refusal.
To address this issue, we introduced a process-level reward based on Natural Language Inference (NLI) to better align the
model’s reasoning trace with its final decision (see Section 4.7). The example above illustrates the motivation for this process
reward: encouraging the model to maintain consistency between its intermediate reasoning and its final answer.
CTrust-Align
Top-100
documents
Filter top ~3k
knowledge
intensive
questionsSeed Prompt Curation
Augmented Prompt Curation
Question
Top-100
DocsCombination
of 5 docsQuestion
Combination
of 5 docsQuestion3
70k
Questions
DocumentsAug. Set
Find different
sets of 5 docs
covering
distinct subset
of claims...ASQA
ELI5
QAMPARIQuestions
Gold claim 1
Gold claim 2
Gold claim 3
Answerability Labelling
Doc 1
[1,1,0]Doc 2
[1,1,0]
Doc 3
[1,0,0]
Doc 4
[1,0,0]
Doc 5
[0,0,0][1,1,0]Union
Answerable2Matt Prater
(Gold Claim 2)
Positive Answer Generation
LLaMA-2-7b
70k
Responses
Alignment
DPO-
LLaMANegative Answer Generation
Trainable
Parameters
Frozen
ParametersSeed SetAugmented
SetDocument
Ove Johansson
(Gold Claim 1)1 4
5
6Questions
DocumentsSeed SetRetrieve top 100
docs using
Wikipedia,
Sphere GPT-4
synthesizer... Matt Prater at 64 yards
[Gold Claim 2], ... Ove
Johansson in a 1976
... [Gold Claim 1].... Matt Prater at 64
yards [1][3], ... Ove
Johansson in a 1976
...[2][4].Answerable Questions
Unanswerable QuestionsPositive Answer: “I apologize, but I couldn't find an answer to your question in the search results.”
Supervised Finetuning 
SFTCalculate
Hallucination
ScoreFilter
top-50%
Direct Preference
OptimizationInference10k
Positive Answer (r  )+
Questions
Oracle Docs
Positive AnsSeed Set
Aug. Set
Questions
Set of 5 DocsNegative
Answer (r  )-19kQuestions
Set of  5 DocumentsPositive Answer
 Negative AnswerAug. SetOnly 40k have ~19k
Use TRUE NLI to check if doc
supports gold claimRepeat for k=5 docs
Doc 1
x7Greedily find
the smallest
subset of
documents
that support
claim using
TRUE NLI
Figure 7: The pipeline of Trust-Align .