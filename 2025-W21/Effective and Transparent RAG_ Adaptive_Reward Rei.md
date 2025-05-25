# Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability

**Authors**: Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li, Weizhi Ma, Yang Liu

**Published**: 2025-05-19 15:40:29

**PDF URL**: [http://arxiv.org/pdf/2505.13258v1](http://arxiv.org/pdf/2505.13258v1)

## Abstract
Retrieval-Augmented Generation (RAG) has significantly improved the
performance of large language models (LLMs) on knowledge-intensive domains.
However, although RAG achieved successes across distinct domains, there are
still some unsolved challenges: 1) Effectiveness. Existing research mainly
focuses on developing more powerful RAG retrievers, but how to enhance the
generator's (LLM's) ability to utilize the retrieved information for reasoning
and generation? 2) Transparency. Most RAG methods ignore which retrieved
content actually contributes to the reasoning process, resulting in a lack of
interpretability and visibility. To address this, we propose ARENA
(Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator
framework trained via reinforcement learning (RL) with our proposed rewards.
Based on the structured generation and adaptive reward calculation, our
RL-based training enables the model to identify key evidence, perform
structured reasoning, and generate answers with interpretable decision traces.
Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments
with various RAG baselines demonstrate that our model achieves 10-30%
improvements on all multi-hop QA datasets, which is comparable with the SOTA
Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses
show that ARENA has strong flexibility to be adopted on new datasets without
extra training. Our models and codes are publicly released.

## Full Text


<!-- PDF content starts -->

arXiv:2505.13258v1  [cs.CL]  19 May 2025Effective and Transparent RAG: Adaptive-Reward
Reinforcement Learning for Decision Traceability
Jingyi Ren1,2, Yekun Xu1, Xiaolong Wang1,2, Weitao Li1,2, Weizhi Ma2∗, Yang Liu1,2∗
1Department of Computer Science and Technology, Tsinghua University, Beijing, China
2Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China
{mawz,liuyang2011}@tsinghua.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has significantly improved the perfor-
mance of large language models (LLMs) on knowledge-intensive domains. How-
ever, although RAG achieved successes across distinct domains, there are still
some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on
developing more powerful RAG retrievers, but how to enhance the generator’s
(LLM’s) ability to utilize the retrieved information for reasoning and generation?
2) Transparency. Most RAG methods ignore which retrieved content actually con-
tributes to the reasoning process, resulting in a lack of interpretability and visibility.
To address this, we propose ARENA (Adaptive- Rewarded Evidence Navigation
Agent), a transparent RAG generator framework trained via reinforcement learning
(RL) with our proposed rewards. Based on the structured generation andadap-
tive reward calculation , our RL-based training enables the model to identify key
evidence, perform structured reasoning, and generate answers with interpretable
decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abun-
dant experiments with various RAG baselines demonstrate that our model achieves
10–30% improvements on all multi-hop QA datasets, which is comparable with the
SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further
analyses show that ARENA has strong flexibility to be adopted on new datasets
without extra training. Our models and codes are publicly released2.
1 Introduction
Retrieval-Augmented Generation (RAG) has become a powerful paradigm for enhancing large
language models (LLMs) with non-parametric knowledge [ 8,12,25,55], particularly in knowledge-
intensive domains such as medicine [ 23], law [ 49], and finance [ 37]. While retrieved information
can significantly improve generation performance [ 22,52], existing research has mainly concen-
trated on improving retrievers, with relatively less attention given to the generator component. Our
empirical analysis reveals a notable performance gap: on widely-used and challenging multi-hop
QA benchmarks, 7B-scale models using identical retrieved contexts underperform state-of-the-art
reasoning models like OpenAI-o1 [ 30] and DeepSeek-R1 [ 42] by 15–35% in answer accuracy (see
Appendix A). This indicates that the generator’s reasoning capability, not just the quality of retrieval,
remains a key bottleneck in current RAG pipelines.
Beyond boosting effectiveness , advancing the generator opens up valuable opportunities for improving
transparency — critical qualities for deploying LLMs in real-world, high-stakes applications to
understand which retrieved document is involved in reasoning and generation [ 41,48,24]. Many
∗Corresponding authors.
2https://github.com/ren258/ARENA
Preprint. Under review.

generators produce unstructured answers without exposing the underlying decision process, limiting
their interpretability and trustworthiness. The central challenge lies in enabling the generator to
identify, integrate, and reason over relevant evidence from noisy or redundant contexts [ 13]. Smaller
open-source models face greater difficulty in such scenarios due to limited reasoning capacity [9].
Recently, reinforcement learning (RL) has shown promise in improving the reasoning ability of
LLMs, especially in math [ 53] and code [ 26] generation, which often rely on reward signals derived
from structured task objectives, enabling models to learn reasoning behavior beyond what supervised
fine-tuning can offer. This inspires us to explore RL as a path for improving RAG generation. Existing
RL-based approaches—such as GRPO [ 11]—typically use generic reward designs and output formats,
which fail to capture the structure of multi-hop QA [ 34]. Moreover, unstable KL regularization often
leads to divergence during training, limiting their applicability in retrieval-grounded tasks.
To address these challenges, we propose ARENA (Adaptive- Rewarded Evidence Navigation Agent),
a transparent and effective RAG generator framework trained via reinforcement learning. ARENA
introduces a structured output format that includes selected references, explicit reasoning traces, and
final answers, enabling end-to-end interpretability. We further design a suite of adaptive, task-specific
rewards and a stabilized optimization strategy tailored for multi-hop QA. Applied to open-source
models like Qwen2.5-7B-Instruct [ 50] and Llama3.1-8B-Instruct [ 10], ARENA achieves 10–30%
accuracy improvements across three datasets, rivaling the best closed-source systems while offering
better transparency and generalizability. Our main contributions are listed as follows:
•We identify two key challenges in current RAG systems, namely limited reasoning ability
and lack of interpretability, and call for greater attention to the generator optimization.
•We propose ARENA , a reinforcement learning-based framework for RAG that integrates
structured generation, adaptive reward design, and stabilized training for transparent and
effective RAG reasoning.
•Experiments demonstrate that ARENA outperforms all open-source baselines of comparable
scale, achieving results on par with leading closed-source models across three challenging
multi-hop QA datasets. ARENA also generalizes well across datasets and backbones.
•All the codes and models are released to support future study on optimizing RAG generator.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG has emerged as an effective approach to enhancing model performance across a wide range
of NLP tasks by incorporating externally retrieved documents into the input [ 14,8], particularly
for knowledge-intensive applications such as open-domain question answering (ODQA) [ 22]. By
grounding the model’s generation in retrieved evidence, RAG can substantially mitigate hallucinations
and factual inaccuracies [ 40,2]. Moreover, recent studies have demonstrated that the benefits of
retrieval become increasingly pronounced as model scale grows [ 28,39], further underscoring the
practical value of retrieval-based methods. In the ODQA task, the prevailing paradigm follows
a retriever-and-reader framework [ 1], where significant effort has been devoted to enhancing the
retriever. Approaches include decomposing complex queries into sub-questions using prompts [33],
rewriting queries to enhance retrieval quality [ 27], guiding retrieval through CoT reasoning [ 46],
constructing summaries to select the most plausible answer among candidates [ 20], and compressing
prompts while integrating retrieval information [ 18]. Despite these substantial advances, how to
optimize the RAG generator was ignored by most studies.
The multi-hop task [ 51,15,45,33] poses significant challenges in the field of RAG, as they require
LLMs to integrate information across multiple retrieved documents in order to generate a correct
answer, which requires more powerful reasoning and generation ability. However, the presence of
irrelevant or noisy contexts can adversely affect answer generation, resulting in factual errors [ 32,6].
Thus, strengthening the reader’s capability to reason over and selectively utilize retrieved content is
critical for achieving robust performance.
Beside, some studies have demonstrated that verifying the reliability of LLM-generated content based
on citations is an effective approach to mitigating hallucinations and provides transparent RAG for
2

Policy
ModelReward
FunctionsReference
Model
Group
ComputationStablized KL
Input Question
Query
References with numbers
Reference Context
Reference Context
......
Reference Context......Rollout Reward Advantage
Reward Calculation
Accuracy Relevance Bonus Format
✅ ✅✅
✅ ✅✅ ✅❌
❌❌ ❌
❌
.......
✅ ✅ ❌
...... ...... ......
Rollout Content
Rollout Content
.....
Rollout Content
<relevance>
[1,5]
</relevance >
<analysis>
Reference [1] states that...
Reference [5] indicates that...
</analysis>
<answer> ......</answer>
Figure 1 Overview of the ARENA ( Adaptive- Rewarded Evidence Navigation Agent) framework. The
system is composed of three key components: 1⃝Structured Generation , where the policy model
generates a multi-part response including selected evidence, reasoning analysis, and a final answer;
2⃝KL Stabilization , we modify the KL formulation to improve stability; 3⃝Adaptive Reward
Calculation , where the model outputs are evaluated on four axes—format, accuracy, relevance, and
bonus—to provide interpretable and fine-grained training signals.
users [ 17]. But most of these studies are post-hoc based, which means the citations are not generated
during LLM reasoning. We propose that, this step can also be finished by the RAG generator.
2.2 LLMs with RL
RL serves as a powerful tool for LLMs to improve their alignment to human values and reasoning
abilities. In particular, reinforcement learning with human feedback (RLHF) has demonstrated the
effectiveness of aligning model outputs with human preferences [ 31]. Beyond alignment, models such
as OpenAI-o1 [ 30], DeepSeek-R1 [ 11] and Kimi k1.5 [ 44] showcase significant gains in mathematical
reasoning and code generation following RL-based training. More recently, R1-Searcher [ 42] and
ReSearch [ 4] have leveraged Reinforce++ [ 16] and GRPO [ 38], respectively, to enhance the abilities
of the models to interact with retrieval systems and produce accurate responses. Collectively,
these methods highlight how reinforcement learning reward signals can systematically improve the
performance of LLM in unsupervised settings. However, prior methods have primarily focused on
strengthening general reasoning ability, without explicitly designing reward functions tailored to
specific tasks such as RAG. In this work, we introduce specialized incentives to address this gap.
3 ARENA: Adaptive-Rewarded Evidence Navigation Agent
We introduce ARENA (Adaptive- Rewarded Evidence Navigation Agent), a novel reinforcement
learning framework for the training of RAG generators. ARENA enhances the reasoning capabilities
of language models and improves decision transparency, enabling more interpretable and reliable
deployment in knowledge-intensive domains. As shown in Figure 1, ARENA consists of three key
components: (1) Structured generation , where the model produces evidence-grounded multi-part
responses; (2) KL stabilization , we stabilized KL regularization for policy refinement; and (3)
Adaptive reward calculation , which provides interpretable feedback across multiple dimensions.
We will introduce these modules in detail in the following subsections.
3

3.1 Framework formalization
Given a natural language question qand a set of kretrieved reference passages C={c1, c2, ..., c k},
the ARENA generator produces a structured response with three clearly defined components:
• A set of selected reference indices I ⊆ { 1,2, ..., k}indicating which passages are used.
• A reasoning trace Zthat synthesizes the selected passages to generate the final answer.
• A concise final answer Oderived from the reasoning trace.
This defines the generator as a mapping:
(q,C)7→(I,Z,O)
To support training and evaluation, we construct a structured prompt format as shown in Table 1
(Figure 1 Module 1). Notably, we introduce an explicit <relevance> part to enforce interpretable
evidence selection, which is critical for transparent multi-hop reasoning. This part requires the model
to generate only the reference indices it relies on, making the reasoning process externally verifiable.
Table 1 ARENA prompt template format. The <relevance> part is the key to encourage explicit and
auditable evidence selection. The placeholders {question} and {references} are filled in at runtime.
A conversation between User and Assistant. The user asks a question and
gives some references. The assistant should answer the question based on the
references.
User’s input will always contain:
<question> [ the question to answer ] </question>
<references> [ references starting with numbers ] </references>
Assistant’s response must contain EXACTLY three sections:
<relevance> [list ONLY reference numbers that provide useful information in square brackets, e.g.
[1,5]] </relevance>
<analysis> [ combine information from relevant references to build the answer. Explicitly mention which
references support each claim ] </analysis>
<answer> [ answer with ONLY a short phrase or single word. no explanations ] </answer>
User:
<question> {question} </question>
<references> {references} </references>
Each generation component serves a specific role/function: the <relevance> part ensures explicit
evidence grounding, the <analysis> part compels the LLM to articulate its reasoning based on the
selected references, and the <answer> part provides a concise and definitive result. Together, this
format enforces traceable generation and supports external auditing, aligning with ARENA’s goals of
enhancing reasoning quality and interpretability.
3.2 Reinforcement learning with stability enhancements
In complex reasoning tasks, supervised fine-tuning (SFT) often falls short due to the lack of high-
quality, annotated reasoning traces [ 5]. Recently, reinforcement learning (RL) has emerged as a
powerful alternative to enhanceperformance through self-improvement. Motivated by this, ARENA
employs reinforcement learning to activate and refine the reasoning capabilities of instruction-tuned
models to enhance the RAG generation ablities of them.
Brief review of Group Relative Policy Optimization (GRPO)
GRPO [ 38] is an advanced policy optimization algorithm that improves over Proximal Policy
Optimization (PPO) [ 36] by eliminating the need for a critic model. For each training in-
stance q, the policy πθgenerates Goutputs {oi}G
i=1. Each output is scored using the task-
specific reward function (detailed in Section 3.3), and a group-wise normalized advantage is
4

computed as: Ai= min
πθ(oi|q)
πθold(oi|q),clip
πθ(oi|q)
πθold(oi|q),1−ϵ,1 +ϵ
ri−mean({r1,r2,···,rG})
std({r1,r2,···,rG}), where
ri=PN
j=1wj·Rj(oi|q)is the weighted reward for output oi.
The KL divergence between the current policy and a fixed reference model is incorporated as a
regularization term to prevent policy collapse. The final GRPO objective is:
JGRPO (θ) =Eq,{oi}∼πθold"
1
GGX
i=1(Ai−βDKL(πθ∥πref))#
.
KL divergence stabilization
The KL divergence term plays a key role in constraining the updated policy from diverging excessively
from the reference model. In standard GRPO, it is estimated using the following unbiased formulation:
r=πref(o|q)
πθ(o|q),DKL(πθ∥πref) =r−logr−1,
which is theoretically appealing: it is non-negative, unbiased, and has low variance under common
assumptions. However, this estimator becomes numerically unstable when the policy assigns near-
zero probability to sampled outputs ( πθ(o|q)→0), causing the importance ratio r→ ∞ . This
results in gradient explosions and erratic updates that severely degrade training stability.
To address this, we aim to retain the desirable properties of the original KL estimator—namely, non-
negativity, low variance, and smooth penalization—while ensuring numerical stability in rare-event
sampling. Motivated by this, we use a simple but effective replacement [35]:
KLstable (o) =1
2(logr)2,
which maintains non-negativity and provides a symmetric, smooth penalty for large divergences. It
avoids singularities even when πθ(o|q)is small, thus offering more stable gradients in training.
3.3 Adaptive reward design
While recent math-oriented studies demonstrated that simple format andaccuracy rewards suffice
to activate reasoning capabilities in GRPO, we find that such task-agnostic metrics fall short in
capturing the nuanced objectives of RAG generation. To this end, we propose a set of task-specific,
interpretable reward functions that provide fine-grained supervision aligned with reasoning quality
and evidence traceability, namely:
Format Reward Rformat (oi|q).Outputs that match the expected structural format—consisting of
<relevance> ,<analysis> , and <answer> parts in order—receive a reward of 1; others receive 0.
Accuracy Reward Raccuracy (oi|q).To evaluate the correctness of the final answer, we apply
normalized Exact Match. The output from the <answer> field is lowercased, stripped of punctuation
and articles, and compared to the gold answer. If the normalized strings match exactly, the model
receives a reward of 1; otherwise 0. Exact Match is more effective than F1/LLM judgment here.
Relevance Reward Rrelevance (oi|q).This reward measures whether the model correctly iden-
tifies the supporting evidence. The predicted reference indices from the <relevance> section are
compared with ground truth. A full match yields 1 point, partial overlap yields 0.5, and no overlap
yields 0. This encourages models to explicitly ground their reasoning in relevant sources.
Bonus Reward Rbonus (oi|q).To promote holistic reasoning behavior, we add a high-value bonus
reward when the model simultaneously satisfies all three criteria above. If the format, accuracy, and
relevance rewards are all 1, an additional bonus of 10 is assigned; otherwise, the bonus is 0. This
reinforces complete, well-aligned outputs.
The final reward used to train the policy model is computed as the sum of the four components and
no reweighting strategy/training is adopted:
ri=Rformat (oi|q) +Raccuracy (oi|q) +Rrelevance (oi|q) +Rbonus (oi|q),
where all weights are set to 1 by default. This combined signal balances structural, semantic, and
evidential quality in a unified reward framework.
5

4 Experimental Setups
4.1 Datasets
We use three widely-used and challenging multi-hop QA datasets for evaluation: HotpotQA [51],
2WikiMultiHopQA [15], and MuSiQue [45]. Each QA consists of a natural language question, a
final answer, and 10–20 retrieved paragraphs from Wikipedia (predefined in the public datasets). We
randomly selected a portion from the whole dataset for training and testing as previous studies [ 42,47,
19], statistics are shown in Table 2. Detailed dataset preprocessing steps are described in Appendix B.
Table 2 Detailed dataset statistics. Hop counts are derived from supporting_facts fields.
Dataset Split Data Size # Paragraphs 1-hop 2-hop 3-hop 4-hop Avg. Hops
HotpotQA Train 10,000 10 3 9977 20 0 2.00
HotpotQA Test 500 10 0 500 0 0 2.00
2WikiMultiHopQA Train 10,000 10 7 7839 3 2151 2.39
2WikiMultiHopQA Test 500 10 0 406 0 94 2.19
MuSiQue Train 5,000 20 0 3595 1107 298 2.85
MuSiQue Test 500 20 0 251 153 96 2.73
All Train – 25,000 – 10 21,411 1130 2449 2.30
4.2 Evaluation metrics
Following prior work [ 51,8], we evaluate model performance using three metrics: normalized Exact
Match (EM), F1 score (F1), and LLM-as-a-Judge (LJ) [ 54]. EM and F1 capture surface-level
correctness, while LJ uses GPT-4o [ 29] (version 2024-11-20) to assess semantic correctness. The
implementation details of all three metrics are provided in Appendix C.
4.3 Baselines
Prompt-based RAG models. We select open-source reasoning models of similar scale, in-
cluding DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B , as well as our backbone models
Qwen2.5-7B-Instruct andLlama3.1-8B-Instruct . Those models answer the question with ref-
erences directly3. We also include two prompting-based methods: SuRe [20], which enhances LLM
performance via summarized retrieval, and self-ask [33], which decomposes complex questions into
sub-questions and answers them iteratively. These serve as strong zero-shot and few-shot baselines.
SFT-based RAG models. We include two supervised fine-tuning baselines: SFT-direct , which
directly generates the final answer, and SFT-reasoning , a reasoning-capable model that outputs
structured responses. Training details for both models are provided in Appendix D.
RL-based RAG models. This category includes models trained with reinforcement learning. We
first implement a Naive GRPO baseline using the original GRPO algorithm [ 38], where models
are trained to produce <think> ...</think> and<answer> ...</answer> format responses with
format and accuracy reward. Three SOTA RL-based reasoning frameworks are also involved: R1-
Searcher [42],SimpleDeepSearcher [43], and ReSearch [3]. Each of these methods introduces a
unique approach to reasoning—such as external search invocation, distillation-driven retrieval, or
search-integrated generation. For consistency, we use their released checkpoints and replace their
retrieved contexts with those provided by our datasets.
4.4 Implementation details
We conduct ARENA onQwen2.5-7B-Instruct andLlama3.1-8B-Instruct . Our framework is
implemented based on the Open-R1 [7]. During inference, we concatenate all retrieved paragraphs
provided by the dataset in their original order. All models are trained using 8 NVIDIA A100-80G
GPUs, with 7 GPUs allocated for policy optimization and 1 GPU dedicated to rollout inference via a
vLLM [21] engine. More training details are introduced in Appendix E.
3For reasoning-capable models, the internal thinking process is implicitly conducted during generation in the
direct setting.
6

Table 3 Main experimental results across three datasets and three evaluation metrics. Models are
grouped by training methodology. Our ARENA models are placed alongside their corresponding
base models at the bottom for direct comparison. Bold indicates the best score; underline indicates
the second-best. Metrics: EM = Exact Match (%), F1 = F1 score (%), LJ = LLM-as-a-Judge (%).
ModelHotpotQA 2WikiMultiHopQA MuSiQue
EM F1 LJ EM F1 LJ EM F1 LJ
Prompt-based models
DeepSeek-R1-Distill-Qwen-7B 0.332 0.487 0.712 0.290 0.407 0.658 0.116 0.184 0.278
Qwen3-8B 0.582 0.719 0.768 0.652 0.727 0.784 0.336 0.397 0.394
SuRe-GPT-4o 0.490 0.695 0.742 0.490 0.607 0.632 0.194 0.307 0.324
SuRe-Qwen-7B 0.504 0.648 0.696 0.364 0.448 0.440 0.146 0.233 0.226
self-ask-GPT-4o 0.408 0.577 0.662 0.430 0.565 0.614 0.162 0.305 0.346
self-ask-Qwen-7B 0.016 0.112 0.246 0.004 0.128 0.086 0.006 0.052 0.087
SFT-based models
Qwen-7B-SFT-direct 0.498 0.628 0.662 0.540 0.609 0.620 0.168 0.254 0.214
Qwen-7B-SFT-reasoning 0.400 0.532 0.599 0.536 0.615 0.622 0.112 0.172 0.174
RL-based models
Naive-GRPO 0.534 0.662 0.732 0.620 0.684 0.720 0.332 0.435 0.420
R1-Searcher-7B 0.590 0.731 0.791 0.632 0.704 0.730 0.250 0.370 0.327
SimpleDeepSearcher-7B 0.500 0.633 0.686 0.632 0.707 0.748 0.258 0.347 0.344
ReSearch-7B 0.578 0.729 0.778 0.586 0.658 0.684 0.300 0.398 0.407
Ours (ARENA)
Llama3.1-8B-Instruct (Original) 0.528 0.676 0.738 0.398 0.478 0.484 0.248 0.370 0.324
ARENA-Llama-8B 0.552 0.703 0.780 0.622 0.710 0.735 0.358 0.445 0.466
Qwen2.5-7B-Instruct (Original) 0.484 0.628 0.660 0.334 0.424 0.412 0.252 0.354 0.306
ARENA-Qwen-7B 0.628 0.762 0.812 0.660 0.752 0.786 0.400 0.520 0.508
Table 4 Evaluation of ARENA-Qwen-7B under different training dataset combinations. “+WM”
denotes training on 2 WikiMultiHopQA + MuSiQue, “+HM” on HotpotQA + MuSiQue, and “+HWM”
on all three datasets. Results are reported on all three test sets to assess cross-domain generalization.
ModelHotpotQA 2WikiMultiHopQA MuSiQue
EM F1 LJ EM F1 LJ EM F1 LJ
ARENA-Qwen-7B + WM 0.568 0.711 0.764 0.642 0.732 0.734 0.394 0.504 0.486
ARENA-Qwen-7B + HM 0.598 0.741 0.796 0.578 0.671 0.680 0.404 0.519 0.478
ARENA-Qwen-7B + HWM 0.628 0.762 0.812 0.660 0.752 0.774 0.400 0.520 0.508
5 Experimental Results
5.1 Main results
As shown in Table 3, ARENA-Qwen-7B achieves the best performance across all datasets and metrics,
consistently outperforming all baselines of similar scale. Compared to its backbone models, ARENA
brings 10–30% improvements, with particularly strong gains on MuSiQue, highlighting its ability
to handle longer and more complex reasoning chains. ARENA-Llama-8B also outperforms the
original Llama3.1-8B-Instruct significantly, showing that our method is capable of distinct backbones.
We also observe that RL-based models consistently outperform SFT-based models, confirming the
advantage of learning through reward-guided exploration. In contrast, supervised fine-tuning fails to
deliver meaningful gains, suggesting that it struggles to equip models with structured reasoning.
5.2 Generalization ability
The generalization ability of ARENA-Qwen-7B when trained on different subsets of the available
datasets is shown in Table 4. Specifically, “+WM” and “+HM” denote training on two datasets while
7

HotpotQA 2Wiki MuSiQue0.40.50.60.70.80.9LLM-as-a-Judge score (%)
HotpotQA 2Wiki MuSiQue0.700.750.800.850.900.951.00Format score (%)
HotpotQA 2Wiki MuSiQue0.50.60.70.80.91.0Relevance score (%)Naive-GRPO w/o KL stablization ARENA-7BFigure 2 Ablation results comparing Naive GRPO, ARENA without KL stabilization, and full
ARENA on three evaluation metrics across datasets. ARENA shows consistent improvements in
output quality and interpretability.
holding out one, allowing us to assess zero-shot transfer to an unseen domain. In both cases, the
model performs strongly on the held-out dataset, indicating that the reasoning capabilities learned
by ARENA generalize well across different data distributions. This supports our claim that the
improvements from our method are not tied to a specific dataset, but reflect transferable multi-hop
reasoning skills.
5.3 Ablation study
We evaluate the effect of ARENA’s key components by comparing three variants: Naive GRPO
(basic reward and flat output format), w/o KL stabilization (adaptive reward without stable KL
estimation), and the full ARENA-Qwen-7B . As shown in Figure 2, ARENA achieves the best results
across all datasets in LLM-as-a-Judge score, format score, and relevance score, demonstrating its
more powerful ability in generate accurate and interpretable outputs.
Table 5 Comparison between ARENA-Qwen-7B
and three strong closed-source models on Hot-
potQA. ARENA shows strong competitiveness
despite smaller model size.
Model EM F1 LJ
GPT-4o 0.628 0.788 0.820
OpenAI-o1 0.656 0.816 0.890
DeepSeek-R1 0.618 0.781 0.836
ARENA-Qwen-7B 0.628 0.762 0.812Table 6 Comparison of different accuracy reward
formulations used during RL training. EM-based
reward leads to the best performance on Hot-
potQA across multiple evaluation dimensions.
Judge Method EM F1 LJ
GPT-4o 0.612 0.752 0.804
F1 0.600 0.738 0.794
EM (ours) 0.628 0.762 0.812
5.4 Further analyses
ARENA Vs. SOTA Commercially-developed Models. We compare ARENA-Qwen-7B with three
strong commercially-developed LLMs on HotpotQA in Table 5, including GPT-4o &OpenAI-o1
from OpenAI and DeepSeek-R1. Despite its smaller size, our ARENA shows competitive results
across all metrics, with performance close to GPT-4o in particular. This highlights ARENA’s potential
as a lightweight yet effective alternative for interpretable multi-hop reasoning.
Reward Analysis. We also test different reward formulations for the accuracy component during RL
training, including EM, F1, and LLM-based judgment. As shown in Table 6, using Exact Match as
the reward provides the best overall results across EM, F1, and LJ evaluation. This confirms that a
simple, binary EM-based reward provides a more stable and effective training signal for multi-hop
QA tasks. More experimental results and analyses on 2WikiMultiHopQA and MuSiQue datasets are
provided in Appendix G.
8

Q: What party  was the secretary of commerce and housing  under the 42 Governor of Kansas ？Input:
Refer ence 1~Refer ence 5.  David "Dave" Lopez...... Omit 392 words
Refer ence 6.  Joan Finney  (February 12, 1925 – July 28, 2001), served as the 42nd Governor of
Kansas  from 1991 to 1995.
Refer ence 7.  William Thornton is an American politician from...... Omit 48 words
Refer ence 8.  Robert G. "Bob" Knight  (born July 31, 1941) was the Republican  mayor of Wichita,
Kansas for seven terms. He first ran for of fice, Wichita City Council, in 1979. He also served
under Democratic Governor Joan Finney  as Kansas Secretary of Commerce and Housing . He is
generally known as "Bob Knight."
Refer ence 10.  The Kansas Department of Commerce is a...... Omit 55 wordsRefer ence 9.  Natalie Shirley (born 1957) is an American lawyer ......Omit 88 words
ARENA  (Ours)
<relevance> [6,8] </relevance>
<analysis> Governor Joan Finney , the 42nd Governor of Kansas, served from 1991 to 1995 [6].
Under her administration, Robert G. "Bob" Knight served as the Kansas Secretary of Commerce and
Housing [8]. Therefore, the party af filiation of the secretary of commerce and housing under the
42nd Governor of Kansas can be inferred from Knight's party af filiation. Reference [8]  also
confirms that Knight was appointed by the Democratic Governor Joan Finney , implying that Knight
was a member of the Republican party .</analysis>
<answer> Republican </answer>
Naive GRPO
<think> The question asks about the party of the secretary of commerce and housing under the 42nd
Governor of Kansas\nThe reference information states that Joan Finney was the 42nd Governor of
Kansas from 1991 to 1995\nThe references do not provide specific information about the party of
the secretary of commerce and housing under Joan Finney </think>
<answer> Democratic </answer>
Figure 3 A Case Study of Naive GRPO and ARENA on the HotpotQA. ARENA correctly extracts
relevant paragraphs, provides a structured explanation, and delivers the right answer.
Case Study. Figure 3 illustrates a representative case. While Naive GRPO produces a valid-looking
but unsupported answer, ARENA identifies the correct references and generates a transparent reason-
ing path. The benefit of KL stabilization is further supported by training curves (see Appendix F),
where the stabilized version maintains smooth KL values and avoids extreme divergence spikes.
6 Conclusion
We propose ARENA , anAdaptive- Rewarded Evidence Navigation Agent that enhances the gener-
ator component of RAG systems via reinforcement learning with task-specific rewards. ARENA
enables models to identify relevant evidence, reason in a structured and interpretable way, and
produce accurate, verifiable answers. Extensive experiments across three multi-hop QA benchmarks
demonstrate that ARENA significantly outperforms all open-source models of similar scale, with
10–30% gains in accuracy. It also generalizes well across datasets and backbone models, highlighting
its robustness and transferability. By generating explicitly structured outputs, ARENA facilitates
transparent reasoning, addressing a critical need for interpretability in real-world applications.
Limitation Despite its effectiveness and transparent, ARENA assumes access to high-quality
retrieved content and may underperform when critical evidence is missing or noisy. Its reward and
KL stabilization are designed for RAG, which may unable to be direct adopted to other domains.
Future work may explore retrieval-aware reward shaping and extend the framework to other tasks.
9

References
[1]Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to
answer open-domain questions. In Proceedings ofthe55th Annual Meeting oftheAssociation
forComputational Linguistics (V olume 1:Long Papers) , pages 1870–1879, Vancouver, Canada.
Association for Computational Linguistics.
[2]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large lan-
guage models in retrieval-augmented generation. In Thirty-Eighth AAAI Conference on
Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference onInnovative Applications
ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium onEducational Advances in
Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada , pages 17754–
17762. AAAI Press.
[3]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z
Pan, Wen Zhang, Huajun Chen, Fan Yang, et al. 2025. Research: Learning to reason with search
for llms via reinforcement learning. ArXiv preprint, abs/2503.19470.
[4]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Fan Yang, Zenan Zhou,
Weipeng Chen, Haofen Wang, Jeff Z Pan, et al. 2025. Learning to reason with search for llms
via reinforcement learning. ArXiv preprint, abs/2503.19470.
[5]Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V Le, Sergey Levine, and Yi Ma. 2025. Sft memorizes, rl generalizes: A comparative
study of foundation model post-training. ArXiv preprint, abs/2501.17161.
[6]Antonia Creswell, Murray Shanahan, and Irina Higgins. 2023. Selection-inference: Exploit-
ing large language models for interpretable logical reasoning. In TheEleventh International
Conference onLearning Representations, ICLR 2023, Kigali, Rwanda, May 1-5,2023 . Open-
Review.net.
[7] Hugging Face. 2025. Open r1: A fully open reproduction of deepseek-r1.
[8]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language
models: A survey. ArXiv preprint, abs/2312.10997.
[9]Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, and Haofen Wang. 2025. Synergizing
rag and reasoning: A systematic review. ArXiv preprint, abs/2504.15909.
[10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024.
The llama 3 herd of models. ArXiv preprint, abs/2407.21783.
[11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability
in llms via reinforcement learning. ArXiv preprint, abs/2501.12948.
[12] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. A comprehensive survey of
retrieval-augmented generation (rag): Evolution, current landscape and future directions. ArXiv
preprint, abs/2410.12837.
[13] Devansh Guttikonda, Deepika Indran, Lakshmi Narayanan, Tanishka Pasarad, and Sandesh
BJ. 2025. Explainable ai: A retrieval-augmented generation based framework for model
interpretability.
[14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval
augmented language model pre-training. In Proceedings ofthe37th International Conference
onMachine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings
ofMachine Learning Research, pages 3929–3938. PMLR.
[15] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing
a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of
the28th International Conference onComputational Linguistics , pages 6609–6625, Barcelona,
Spain (Online). International Committee on Computational Linguistics.
10

[16] Jian Hu. 2025. Reinforce++: A simple and efficient approach for aligning large language
models. ArXiv preprint, abs/2501.03262.
[17] Jie Huang and Kevin Chang. 2024. Citation: A key to building responsible and accountable
large language models. In Findings oftheAssociation forComputational Linguistics: NAACL
2024, pages 464–473, Mexico City, Mexico. Association for Computational Linguistics.
[18] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and
Lili Qiu. 2023. Longllmlingua: Accelerating and enhancing llms in long context scenarios via
prompt compression. ArXiv preprint, abs/2310.06839.
[19] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani,
and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning. ArXiv preprint, abs/2503.09516.
[20] Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-
Woo Ha, and Jinwoo Shin. 2024. Sure: Summarizing retrievals using answer candidates for
open-domain qa of llms. ArXiv preprint, abs/2404.13081.
[21] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language
model serving with pagedattention. In Proceedings ofthe29th Symposium onOperating
Systems Principles, pages 611–626.
[22] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks.
InAdvances inNeural Information Processing Systems 33:Annual Conference onNeural
Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
[23] Junkai Li, Yunghwei Lai, Weitao Li, Jingyi Ren, Meng Zhang, Xinhui Kang, Siyu Wang, Peng
Li, Ya-Qin Zhang, Weizhi Ma, et al. 2024. Agent hospital: A simulacrum of hospital with
evolvable medical agents. ArXiv preprint, abs/2405.02957.
[24] Weitao Li, Junkai Li, Weizhi Ma, and Yang Liu. 2024. Citation-enhanced generation for
llm-based chatbots. ArXiv preprint, abs/2402.16063.
[25] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang,
and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced large reasoning models. ArXiv
preprint, abs/2501.05366.
[26] Shaoteng Liu, Haoqi Yuan, Minda Hu, Yanwei Li, Yukang Chen, Shu Liu, Zongqing Lu, and
Jiaya Jia. 2024. Rl-gpt: Integrating reinforcement learning and code-as-policy. ArXiv preprint ,
abs/2402.19299.
[27] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query rewriting
in retrieval-augmented large language models. In Proceedings ofthe2023 Conference on
Empirical Methods inNatural Language Processing , pages 5303–5315, Singapore. Association
for Computational Linguistics.
[28] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric
and non-parametric memories. In Proceedings ofthe61st Annual Meeting oftheAssociation
forComputational Linguistics (V olume 1:Long Papers) , pages 9802–9822, Toronto, Canada.
Association for Computational Linguistics.
[29] OpenAI. 2024. Hello GPT-4o. Accessed on June 16, 2024.
[30] OpenAI. 2024. Learning to reason with llms. Accessed on September 12, 2024.
[31] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton,
Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Chris-
tiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions
11

with human feedback. In Advances inNeural Information Processing Systems 35:Annual
Conference onNeural Information Processing Systems 2022, NeurIPS 2022, New Orleans,
LA,USA, November 28-December 9,2022.
[32] Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H
Miller, and Sebastian Riedel. 2020. How context affects language models’ factual predictions.
ArXiv preprint, abs/2005.04611.
[33] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. 2023.
Measuring and narrowing the compositionality gap in language models. In Findings ofthe
Association forComputational Linguistics: EMNLP 2023 , pages 5687–5711, Singapore. As-
sociation for Computational Linguistics.
[34] Tejaskumar Pujari, Anil Kumar Pakina, and Anshul Goel. 2023. Explainable ai and governance:
Enhancing transparency and policy frameworks through retrieval-augmented generation (rag).
[35] John Schulman. 2020. Approximating kl divergence.
[36] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms. ArXiv preprint, abs/1707.06347.
[37] Spurthi Setty, Harsh Thakkar, Alyssa Lee, Eden Chung, and Natan Vidra. 2024. Improving
retrieval for rag based question answering models on financial documents. ArXiv preprint ,
abs/2404.07221.
[38] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. ArXiv preprint, abs/2402.03300.
[39] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis,
Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-augmented black-box lan-
guage models. In Proceedings ofthe2024 Conference oftheNorth American Chapter ofthe
Association forComputational Linguistics: Human Language Technologies (V olume 1:Long
Papers), pages 8371–8384, Mexico City, Mexico. Association for Computational Linguistics.
[40] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval
augmentation reduces hallucination in conversation. In Findings oftheAssociation for
Computational Linguistics: EMNLP 2021 , pages 3784–3803, Punta Cana, Dominican Re-
public. Association for Computational Linguistics.
[41] Chandan Singh, Jeevana Priya Inala, Michel Galley, Rich Caruana, and Jianfeng Gao. 2024.
Rethinking interpretability in the era of large language models. ArXiv preprint , abs/2402.01761.
[42] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei
Fang, and Ji-Rong Wen. 2025. R1-searcher: Incentivizing the search capability in llms via
reinforcement learning. ArXiv preprint, abs/2503.05592.
[43] Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Lei Fang,
Zhongyuan Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2025. Simpledeepsearcher: Deep
information seeking via web-powered reasoning trajectory synthesis.
[44] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li,
Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement
learning with llms. ArXiv preprint, abs/2501.12599.
[45] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue:
Multihop questions via single-hop question composition. Transactions oftheAssociation for
Computational Linguistics, 10:539–554.
[46] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Inter-
leaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.
InProceedings ofthe61st Annual Meeting oftheAssociation forComputational Linguistics
(V olume 1:Long Papers) , pages 10014–10037, Toronto, Canada. Association for Computa-
tional Linguistics.
12

[47] Jinyu Wang, Jingjing Fu, Rui Wang, Lei Song, and Jiang Bian. 2025. Pike-rag: specialized
knowledge and rationale augmented generation. ArXiv preprint, abs/2501.11551.
[48] Yeo Wei Jie, Ranjan Satapathy, Rick Goh, and Erik Cambria. 2024. How interpretable are
reasoning explanations from prompting large language models? In Findings oftheAssociation
forComputational Linguistics: NAACL 2024 , pages 2148–2164, Mexico City, Mexico. Asso-
ciation for Computational Linguistics.
[49] Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie,
Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, and Bruno Fleisch. 2024. Cbr-rag:
case-based reasoning for retrieval augmented generation in llms for legal question answering.
InInternational Conference onCase-Based Reasoning, pages 445–460. Springer.
[50] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 technical report. ArXiv preprint ,
abs/2412.15115.
[51] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop
question answering. In Proceedings ofthe2018 Conference onEmpirical Methods inNatural
Language Processing , pages 2369–2380, Brussels, Belgium. Association for Computational
Linguistics.
[52] Wenhao Yu. 2022. Retrieval-augmented generation across heterogeneous knowledge. In
Proceedings ofthe2022 Conference oftheNorth American Chapter oftheAssociation for
Computational Linguistics: Human Language Technologies: Student Research Workshop ,
pages 52–58, Hybrid: Seattle, Washington + Online. Association for Computational Linguistics.
[53] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. 2025.
Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the
wild. ArXiv preprint, abs/2503.18892.
[54] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion
Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances inNeural
Information Processing Systems 36:Annual Conference onNeural Information Processing
Systems 2023, NeurIPS 2023, New Orleans, LA,USA, December 10-16,2023.
[55] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang,
Sicong Liang, Xilin Liu, Yuchi Ma, et al. 2025. In-depth analysis of graph-based rag in a unified
framework. ArXiv preprint, abs/2503.04338.
13

A Settings of the Pilot Experiments
We present a comprehensive comparison of multiple vanilla models under two inference settings:
direct , where the model generates the answer based only on the question, and direct RAG , where
the model receives both the question and all retrieved contexts as input. As shown in Table 7,
RAG inputs consistently improve the performance of all models, highlighting the effectiveness of
retrieval-augmented generation.
Moreover, even under the same RAG context, different LLMs exhibit large performance discrepancies.
Notably, models like OpenAI-o1 and DeepSeek-R1 outperform open-source 7B models by 15–35%
across multiple datasets and metrics. These results underscore the variability of RAG generator
capabilities and motivate our work to improve them through reinforcement learning.
B Dataset Preprocessing
We follow the preprocessing pipeline from PIKE-RAG [ 47] to format the train and dev splits for all
datasets. Each example includes:
• A natural language question.
• A list of retrieved paragraphs from Wikipedia.
• A gold-standard answer (often as a list of acceptable synonyms).
•A set of supporting_facts , which mark the sentences essential for answering the ques-
tion.
To construct the <references> field, we concatenate all retrieved paragraphs in their original order
with paragraph-level IDs (1 to n). To supervise relevance prediction and reasoning, we use the
supporting_facts annotations to locate which paragraphs contain the necessary sentences. These
are mapped to their paragraph indices to produce a ground-truth list of relevant reference IDs.
For training, we randomly sample 10,000 instances each from HotpotQA and 2WikiMultiHopQA,
and 5,000 from MuSiQue. For testing, we sample 500 development examples from each dataset,
following the setup in [42, 47, 19] to ensure comparability.
Table 7 Performance of vanilla models under two inference strategies: direct anddirect RAG . All
models are evaluated using identical retrieved contexts. Bold indicates the highest value per column.
Metrics: EM = Exact Match (%), F1 = F1 score (%), LJ = LLM-as-a-Judge (%).
ModelHotpotQA 2WikiMultiHopQA MuSiQue
EM F1 LJ EM F1 LJ EM F1 LJ
direct
Qwen2.5-7B-Instruct 0.188 0.247 0.262 0.222 0.253 0.234 0.040 0.090 0.061
Llama3.1-8B-Instruct 0.194 0.283 0.284 0.210 0.246 0.232 0.038 0.080 0.054
DeepSeek-R1-Distill-Qwen-7B 0.086 0.150 0.162 0.144 0.206 0.234 0.012 0.043 0.026
Qwen3-8B 0.248 0.344 0.367 0.220 0.263 0.264 0.030 0.092 0.060
GPT-4o 0.378 0.500 0.500 0.342 0.386 0.366 0.100 0.192 0.124
OpenAI-o1 0.510 0.667 0.726 0.530 0.634 0.684 0.292 0.404 0.392
DeepSeek-R1 0.422 0.564 0.590 0.454 0.536 0.546 0.202 0.330 0.294
direct RAG
Qwen2.5-7B-Instruct 0.484 0.628 0.660 0.334 0.424 0.412 0.252 0.354 0.306
Llama3.1-8B-Instruct 0.528 0.676 0.738 0.398 0.478 0.484 0.248 0.370 0.324
DeepSeek-R1-Distill-Qwen-7B 0.332 0.487 0.712 0.290 0.407 0.658 0.116 0.184 0.278
Qwen3-8B 0.582 0.719 0.768 0.652 0.727 0.784 0.336 0.397 0.394
GPT-4o 0.628 0.788 0.820 0.606 0.687 0.717 0.505 0.617 0.593
OpenAI-o1 0.656 0.816 0.890 0.702 0.805 0.837 0.622 0.744 0.762
DeepSeek-R1 0.618 0.781 0.836 0.721 0.801 0.838 0.559 0.702 0.703
14

Answer Format. The answer labels are often represented as a list of acceptable values (e.g.,
synonyms, numbers, yes/no), and models are considered correct if their normalized output matches
any item in the list. This setting is used consistently across all automatic and LLM-based evaluations.
C Details of Evaluation Metrics
We use three evaluation metrics in our experiments: Exact Match (EM), F1 Score (F1), and LLM-as-
a-Judge (LJ). Below we describe the implementation details of each.
C.1 Exact match and normalization
The Exact Match (EM) metric determines whether the model’s answer exactly matches any of the
reference answers after normalization. The normalization process includes:
• Lowercasing the text.
• Removing punctuation and articles ( a,an,the).
• Replacing underscores with spaces.
• Collapsing multiple spaces into a single space.
This ensures robustness to minor formatting or surface-level differences. A score of 1.0 is returned
for an exact match; otherwise, 0.0.
C.2 F1 score calculation
The F1 score is computed at the token level based on normalized answers. For each reference answer,
we calculate the precision and recall between the predicted and ground truth tokens, and derive the F1
score as their harmonic mean. The maximum F1 score across all ground truths is taken as the final
score for a given prediction.
C.3 LLM-as-a-Judge evaluation
To evaluate semantic correctness that may not be captured by surface-level metrics, we adopt a
GPT-based LLM-as-a-Judge (LJ) method. For each question, the model’s generated response and
the reference answers are embedded into a prompt and passed to GPT-4o (version 2024-11-20). The
model is instructed to answer "yes" if the response is semantically consistent with any of the correct
answers (including paraphrases or synonyms), and "no" otherwise.
The prompt template is shown in Table 8:
Table 8 Prompt template used in LLM-as-a-Judge evaluation. The placeholders {correct_answer}
and{model_response} are replaced at runtime.
*************Consider a knowledge Q&A RAG task to test the capability of a
testing model, the correct answer list is:*************
{correct_answer}
*************Here is the model’s response:*************
{model_response}
*************Please check if the model’s answer is correct. As long as the
model’s answer hits any item (or synonym) in the correct answer list, it can be
considered correct. You only need to answer "yes" or "no".*************
When calling the GPT-4o API, we set the temperature to 0.0 to ensure deterministic judgment
behavior.
D Details of Supervised Fine-Tuning
To highlight the benefits of reinforcement learning, we design two supervised fine-tuning (SFT)
baselines:
15

SFT-direct. This variant trains the model to generate the final answer directly. During training, we
concatenate the input question with all retrieved paragraphs (in their original order) and instruct the
model to output a short answer without any intermediate reasoning or evidence selection. This setup
resembles conventional instruction-tuning on flat context.
SFT-reasoning. This variant follows a two-stage pipeline to teach the model structured reasoning.
Specifically, for each training instance, we collect:
• Question q
• Retrieved reference set C
• Ground-truth relevant reference IDs I
• Ground-truth final answer O
These components are fed into the backbone model using a reasoning prompt shown in Table 9,
prompting it to generate a coherent reasoning trace. The output reasoning trace is then integrated into
the structured format described in Section 3.1:
(q,C)→(I,Z,O)
This structured format is then used for supervised fine-tuning with full output supervision.
Table 9 Prompt format for collecting structured reasoning traces in SFT-reasoning. The model is
instructed to derive the answer using only specified relevant references.
Generate a thinking process showing how to derive the given answer using ONLY
the specified relevance IDs from provided references.
Question: {question}
References: {references}
Relevance IDs: {relevance_ids}
Answer: {answer}
E Details of Training Configuration
We train all models using the Open-R1 reinforcement learning framework with DeepSpeed ZeRO
Stage 2 optimization and bfloat16 mixed-precision. The total batch size is 256. The learning rate
is set to 3e-6. We generate 7 rollout samples per input for reward estimation. We set temperature
= 0.9 during rollout, KL coefficient β= 0.04, number of iterations per batch µ= 1, and clipping
parameter ϵ= 0.2.
F KL Divergence Stabilization
To examine the effect of our stabilized KL divergence formulation (see Section 3.2), we compare
its training dynamics with the standard KL estimator. As shown in Figure 4, the stabilized variant
exhibits smooth and bounded updates throughout training, while the original estimator suffers from
catastrophic spikes. This confirms that our modified KL term not only improves theoretical stability
but also provides better empirical convergence in practice.
G Additional Experiemntal Results on Reward Variants
To further evaluate the impact of different accuracy reward designs during training, we conduct the
same experiments on 2WikiMultiHopQA and MuSiQue. Results are shown in Table 10 and Table 11.
On 2WikiMultiHopQA, EM-based reward continues to yield the best performance across all three
metrics, confirming its generalizability on structured reasoning tasks.
On MuSiQue, however, we observe that GPT- and F1-based judgment outperform EM-based reward.
This may be attributed to the greater difficulty of the MuSiQue dataset—each question is paired
with 20 retrieved paragraphs instead of 10—making exact matching harder and rendering fuzzy
16

0 20 40 60 80
Training Steps0.00.10.20.30.40.5KL Value
0 20 40 60 80
Training Steps01,000,0002,000,0003,000,0004,000,0005,000,000KL Value
Figure 4 Training KL divergence curves for stabilized vs. unstabilized variants. ARENA’s stabilized
KL (left) leads to smoother convergence, while the standard estimator (right) suffers from extreme
spikes.
evaluation criteria more helpful for guiding learning. These findings suggest that while EM-based
reward is generally more stable and interpretable, looser metrics might be preferable in high-noise or
high-ambiguity retrieval settings.
Table 10 Reward method comparison on 2Wiki-
MultiHopQA. EM-based reward remains the
most effective.
Judge Method EM F1 LJ
GPT-4o 0.644 0.730 0.740
F1 0.646 0.736 0.732
EM (ours) 0.660 0.752 0.774Table 11 Reward method comparison on
MuSiQue. F1- and GPT-based rewards show
advantage in this more challenging setting.
Judge Method EM F1 LJ
GPT-4o 0.430 0.533 0.516
F1 0.438 0.548 0.502
EM (ours) 0.400 0.520 0.508
17