# Mitigating Hallucination in Financial Retrieval-Augmented Generation via Fine-Grained Knowledge Verification

**Authors**: Taoye Yin, Haoyuan Hu, Yaxin Fan, Xinhao Chen, Xinya Wu, Kai Deng, Kezun Zhang, Feng Wang

**Published**: 2026-02-05 14:49:05

**PDF URL**: [https://arxiv.org/pdf/2602.05723v1](https://arxiv.org/pdf/2602.05723v1)

## Abstract
In financial Retrieval-Augmented Generation (RAG) systems, models frequently rely on retrieved documents to generate accurate responses due to the time-sensitive nature of the financial domain. While retrieved documents help address knowledge gaps, model-generated responses still suffer from hallucinations that contradict the retrieved information. To mitigate this inconsistency, we propose a Reinforcement Learning framework enhanced with Fine-grained Knowledge Verification (RLFKV). Our method decomposes financial responses into atomic knowledge units and assesses the correctness of each unit to compute the fine-grained faithful reward. This reward offers more precise optimization signals, thereby improving alignment with the retrieved documents. Additionally, to prevent reward hacking (e.g., overly concise replies), we incorporate an informativeness reward that encourages the policy model to retain at least as many knowledge units as the base model. Experiments conducted on the public Financial Data Description (FDD) task and our newly proposed FDD-ANT dataset demonstrate consistent improvements, confirming the effectiveness of our approach.

## Full Text


<!-- PDF content starts -->

MITIGATING HALLUCINATION IN FINANCIAL RETRIEV AL-AUGMENTED
GENERATION VIA FINE-GRAINED KNOWLEDGE VERIFICATION
Taoye Yin†Haoyuan Hu†Yaxin Fan Xinhao Chen
Xinya Wu Kai Deng Kezun Zhang‡Feng Wang
Ant Group, Hangzhou, China
ABSTRACT
In financial Retrieval-Augmented Generation (RAG) sys-
tems, models frequently rely on retrieved documents to gen-
erate accurate responses due to the time-sensitive nature of
the financial domain. While retrieved documents help ad-
dress knowledge gaps, model-generated responses still suffer
from hallucinations that contradict the retrieved information.
To mitigate this inconsistency, we propose aReinforcement
Learning framework enhanced withFine-grainedKnowledge
Verification (RLFKV). Our method decomposes financial
responses into atomic knowledge units and assesses the cor-
rectness of each unit to compute the fine-grained faithful
reward. This reward offers more precise optimization signals,
thereby improving alignment with the retrieved documents.
Additionally, to prevent reward hacking (e.g., overly con-
cise replies), we incorporate an informativeness reward that
encourages the policy model to retain at least as many knowl-
edge units as the base model. Experiments conducted on
the public Financial Data Description (FDD) task and our
newly proposed FDD-ANT dataset demonstrate consistent
improvements, confirming the effectiveness of our approach.
Index Terms—Retrieval-augmented generation, Finan-
cial, Hallucination, Atomic knowledge
1. INTRODUCTION
With the increasing prevalence of large language models
(LLMs), RAG has emerged as a critical technology for en-
hancing the accuracy and timeliness of LLM knowledge[1,
2, 3]. By integrating LLMs with external knowledge bases,
RAG effectively addresses key challenges such as knowledge
obsolescence in LLMs[4].
In the financial domain, RAG systems always rely more
heavily on retrieved documents due to the strong timeliness
requirements of financial queries. As illustrated in Fig. 1,
for queries such as “Did the net profit per share of Kwei-
chow Moutai meet expectations?”, the LLM must analyze the
most recently retrieved documents to generate an accurate
response. However, the generated answers may still contain
†Co-first authors.‡Corresponding author: kezun.zkz@antgroup.com
Fig. 1. An example of hallucination by the base model
Qwen3-8B in financial data description.
hallucinations that conflict with the retrieved content. The
orange highlighted section in Fig. 1 reveals a critical error:
although the retrieved document explicitly states that as of
March 31, 2025, the company’s earnings per share (EPS)
was 70.86 yuan, the model’s output incorrectly associates
this value with May 15, 2025, demonstrating a clear temporal
inconsistency.
Recent studies [5, 6, 7, 8] have attempted to mitigate
hallucinations in RAG systems using reinforcement learning,
which typically relies on binary reward signals derived from
human-annotated reference answers. However, these meth-
ods suffer from two major limitations: first, their reliance on
manual answer annotation incurs high labeling costs; second,
the discrete signals from coarse binary rewards fail to provide
a stable optimization direction for model training.
To address these challenges, we propose the Reinforce-
ment Learning with Fine-grained Knowledge Verification
(RLFKV) framework. It enhances the factual consistency of
generated responses with retrieved documents by providing
fine-grained optimization signals, eliminating the need for
annotated reference answers. Specifically, each response is
decomposed into atomic knowledge units, which are mini-
mal self-contained expressions of financial facts. These units
are rigorously evaluated for their support in the retrievedarXiv:2602.05723v1  [cs.AI]  5 Feb 2026

Fig. 2. The framework of RLFKV .
documents, and the evaluation results provide fine-grained
rewards to guide the model optimization. To prevent reward
hacking whereby the model might generate overly brief re-
sponses, we introduce a binary pairwise constraint to ensure
the policy model retains at least the same number of knowl-
edge units as the base model. The policy model is optimized
by maximizing the faithfulness and informativeness rewards
jointly. Experiments on the public Financial Data Description
(FDD) task from BizFinBench and our proposed FDD-ANT
demonstrate the effectiveness of our method.
2. RELATED WORK
Previous research on RAG has primarily focused on improv-
ing LLM performance through document denoising [9, 10],
query rewriting [11, 12], and iterative retrieval [13, 14]. The
advancement of LLMs [15, 16] has further advanced RAG
systems, especially through chain-of-thought reasoning for
complex queries [1]. Recent studies [5, 6, 7, 8] have incor-
porated reinforcement learning to enhance LLMs, but these
methods depend on reference answers and coarse-grained re-
wards. This approach not only incurs high costs for human
annotation but also provides unstable guidance for optimiz-
ing response quality. In contrast, our RLFKV framework
eliminates the need for annotated answers and provides fine-
grained rewards for stable optimization, leading to higher-
quality generation.
3. METHOD
Given the training query setQ={q 1, . . . , q n}, retrieved doc-
umentsD={d 1, . . . , d m}, and the base modelπ 0to be opti-
mized, our goal is to learn a policy modelπthat generates re-
sponses faithful toD. As illustrated in Fig. 2, our framework
consists of two steps: 1)Atomic Knowledge Unit Decom-
position and Verification(§ 3.1): The responsey iis decom-
posed into atomic units using an evaluation model (Qwen3-
32B), and each unit is then verified for consistency againstD.
2)Optimization with Faithful and Informative Rewards
(§ 3.2). The evaluation results are converted into two reward
signals: a faithful reward and an informative reward. Both
these rewards collectively guide the optimization fromπ 0toπ, ensuring that the generated responses exhibit high faithful-
ness to the retrieved documentDwhile retaining informative-
ness.
3.1. Atomic Knowledge Unit Decomposition and Verifica-
tion
For a queryq iand its corresponding responsey i, the evalua-
tion model first decomposesy iinto a set of atomic knowledge
unitsU i={u 1, u2, . . . , u k}, where each unitu jrepresents
a minimal knowledge unit in the financial response. Then,
the evaluation model verifies the factual consistency between
eachu jand the retrieved documentsD, enabling the deriva-
tion of granular reward signals that guide model optimization.
3.1.1. Atomic Knowledge Unit
Building upon the atomicity of knowledge graph triplets,
we introduce a financial quadruple structure (entity, metric,
value, timestamp) to precisely capture the minimal knowl-
edge units in financial-related descriptions. This design
specifically addresses two core characteristics of financial
texts: strict temporal sensitivity and quantitative-oriented
representation. For example, the expressionAs of March
31, 2025, the company’s earnings per share stood at 70.86
yuanin Fig. 1 is represented as “(Kweichow Moutai, basic
earnings per share, 70.86 yuan, As of March 31, 2025)”. This
quadruple structure enforces a completeness constraint where
the absence of any element invalidates the factual assertion.
3.1.2. Atomic Knowledge Unit Decomposition
To decompose the responsey iinto atomic knowledge units,
we designed a specialized prompt to guide the evaluation
model. The decomposition process is formally defined as
{ui}k
i=1=f(y i)(1)
whereu irepresents an atomic knowledge unit,kdenotes the
number of atomic units iny i, andfis the evaluation model.
Our prompt engineering explicitly specifies four critical di-
mensions: entities, metrics, values, and timestamps. Specifi-
cally, we further provide a financial metric dictionary for the

metric elements for reference. Please refer to the code repos-
itory1for prompt details.
3.1.3. Atomic Knowledge Unit Verification
Given the extracted atomic knowledge units{u i}k
i=1, we em-
ploy the evaluation model to assess their factual consistency
with the retrieved documentD. This verification process is
formally defined as:
{si}k
i=1=f({u i}k
i=1,D)(2)
wheres i∈ {0,1}is a binary verification score, and1indi-
catesu iis factually consistent withD.
3.2. Optimization with Faithful and Informative Reward
After obtaining the fine-grained evaluation results{s i}k
i=1,
we transform them into two distinct reward signals to guide
model optimization: (1) a faithful rewardr fevaluating fac-
tual consistency, and (2) an informative rewardr iassessing
informational depth.r fis computed as:
score=kX
i=1I(si= 0),
rf=1
eη·min(score,γ)(3)
where score represents the count of incorrect atomic knowl-
edge units,ηdenotes the decay rate, min is the minimum
function, andγserves as an upper limit for error counts.
This formulation ensures the smooth reward by capping the
penalty for excessive errors while maintaining sensitivity to
factual inaccuracies.r iis computed as:
ri=(
1ifk≥k 0
0otherwise(4)
wherek 0represents the number of atomic knowledge units
generated by the base modelπ 0. This binary reward scheme
ensures that the policy modelπmaintains at least the same
level of informational depth as the base model, thereby effec-
tively preventing reward hacking behavior.
The overall rewardris the average of the two rewards,
and the GRPO[17] is employed to optimize the model using
the following objective function:
L=E"NX
i=1πθ(ai|s)
πθold(ai|s)·ri−¯r
σr+ϵ−β·KL(π θ∥πθold)#
(5)
whereNis the number of samples in the group,π θis the
optimized policy,π θoldis the old policy,¯randσ rare the mean
and standard deviation of the rewards within the group,ϵis a
small constant for numerical stability, and the KL divergence
term constrains the magnitude of the policy updates.
1https://github.com/antgroup/ANT-Fin-RAG4. EXPERIMENTS
4.1. Experiment Settings
Dataset.We evaluated the effectiveness of the RLFKV
framework on two datasets. The first is the Financial Data
Description (FDD) dataset [18], which is designed to assess a
model’s ability to analyze financial data using retrieved doc-
uments. It consists of 1,461 samples but is limited to stock-
related descriptions. To further validate the generalizability
of our approach in real-world scenarios, we constructed a
more diverse dataset, FDD-ANT, covering stocks, funds, and
macroeconomic indicators. This dataset was collected from
real business scenarios, with sensitive information manually
redacted, and contains 2,000 samples. As neither dataset
includes a training set, we followed the same protocol and
randomly collected 4,000 samples for training purposes.
Evaluation Metrics.We evaluate response quality along two
dimensions: faithfulness and informativeness. For faithful-
ness (faith), we follow prior work [18] and employ a point-
wise assessment using GPT-4o as the evaluator. Specifically,
each response is classified into one of three categories: Fully
Correct (100 points), Partially Correct (60 points), or Con-
taining Significant Errors (0 points)2. For informativeness
(info), we use GPT-4o to count the number of atomic knowl-
edge units present in each response and report the average
across all samples.
Baseline.We compare our model against the following base-
line methods: general-purpose models including DeepSeek-
V2-Lite-Chat-16B[19], Qwen3-8B[20] and LLaMA3.1-8B-
Instruct[21], as well as financial domain-specific models
Xuanyuan-13B[22] and Dianjin-R1-7B[23].
Implementations.Our training process is implemented us-
ing the ms-swift framework. We employ Qwen3-8B and
LLaMA3.1-8B-Instruct as base models and directly perform
reinforcement learning. We train for 1 epoch with a learn-
ing rate of 1e-6. The batch size is set to 1 with 2 gradient
accumulation steps. The maximum response length is 2048
tokens. The rollout numberNis set to 8. All experiments
were conducted on 8 NVIDIA H20 GPUs.
4.2. Main Result
Table 1 presents the performance comparison of various
models on the FDD and FDD-ANT datasets. On the FDD
dataset, our RLFKV LLaMA3 improves the faithfulness score
by 3.6 points over LLaMA3, while RLFKV Qwen3 achieves a
3.0-point gain over Qwen3. A similar trend is observed on
FDD-ANT, where RLFKV LLaMA3 outperforms LLaMA3 by
1.6 points in faithfulness, and RLFKV Qwen3 surpasses Qwen3
by 3.1 points. These results indicate that our method effec-
2For detailed criteria, refer tohttps://github.com/
HiThink-Research/BizFinBench/blob/main/benchmark_
code/BizFinBench/eval_financial_description.py

ModelFDD FDD-ANT
faith info faith info
open-source model
DeepSeek V2 Lite 69.1 8.8 76.1 5.4
LlaMA3 80.0 11.4 80.5 6.5
Qwen3 86.5 13.4 90.2 10.8
finance model
Xuanyuan3 57.8 7.9 64.6 5.8
Dianjin-R1 78.3 10.8 84.7 6.8
ours
RLFKV LLaMA3 83.6 11.7 82.1 8.1
w/o info reward 83.2 10.3 81.4 7.0
RLFKV Qwen3 89.5 13.5 93.3 12.3
w/o info reward 89.0 12.0 91.9 11.2
Table 1. Performance comparison of various models on the
FDD and FDD-ANT datasets.
tively reduces hallucination by generating responses that are
more consistent with the retrieved documents.
When the informativeness reward is ablated (w/o info re-
ward), the faithfulness score remains comparable, but the in-
formativeness metric drops by 1.4 and 1.5 points for LLaMA3
and Qwen3, respectively, on FDD, and by 1.1 points for both
on FDD-ANT. This suggests that optimizing solely for faith-
fulness may lead to shorter and less informative responses. In
contrast, by integrating both rewards, our RLFKV framework
enhances faithfulness while maintaining informativeness.
4.3. Effectiveness of Fine-Grained Reward
To further demonstrate the effectiveness of our fine-grained
reward, we compared it with the coarse-grained one that as-
signs binary rewards (1 if the response contains no factual er-
rors, otherwise 0). As shown in Fig. 3(a), models trained with
the fine-grained reward achieved consistently higher faithful
scores. Furthermore, Fig. 3(b) illustrate the evolution of re-
ward values during training. The fine-grained reward demon-
strates more stable optimization and converges more rapidly
(within 2k steps), indicating that it offers smoother learning
signals and ultimately leads to superior performance.
4.4. Error Analysis
We conducted an error analysis to further investigate poten-
tial improvements for our RLFKV framework. While our ap-
proach effectively reduces factual hallucinations, three persis-
tent error types remain, as illustrated in Fig. 4. The predom-
inant issue is time omissions (55%), where critical temporal
references are absent despite being present in the retrieved
documents. Other significant errors include time inaccura-
cies (28%), particularly involving relative time expressions
and fiscal-to-calendar year conversions, and numerical errors
(17%), which primarily occur in imprecise rounding. These
(a) Performance comparison on FDD-ANT datasets.
(b) Reward trends during the training process.
Fig. 3. Performance comparison and reward analysis of fine-
vs. coarse-grained rewards based on Qwen3-8B model.
Fig. 4. Error analysis on FDD-ANT dataset.
findings illuminate promising pathways for future research
and development.
5. CONCLUSION
In this paper, we propose a reinforcement learning frame-
work with fine-grained knowledge verification (RLFKV) to
improve the consistency between generated responses and
retrieved documents. RLFKV decomposes responses into
atomic knowledge units and evaluates the factuality of each
unit, providing stable learning signals for model optimiza-
tion. Additionally, we introduce FDD-ANT, a financial data
description dataset spanning multiple data types, to validate
the general applicability of our method in real-world scenar-
ios. Experimental results on both the public FDD dataset
and our released FDD-ANT dataset confirm the effective-
ness of RLFKV . Future work will focus on refining reward
mechanisms to enhance temporal and numerical accuracy.

6. REFERENCES
[1] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yu-
jia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou,
“Search-o1: Agentic search-enhanced large reasoning
models,” 2025.
[2] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu, “Deep-
Researcher: Scaling deep research via reinforcement
learning in real-world environments,” 2025.
[3] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin
Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong
Yu, “Chain-of-Note: Enhancing robustness in retrieval-
augmented language models,” inProceedings of the
EMNLP, 2024, pp. 14672–14685.
[4] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel,
Sebastian Riedel, and Douwe Kiela, “Retrieval-
augmented generation for knowledge-intensive nlp
tasks,” inProceedings of the NeuIPS, 2020, pp. 9459–
9474.
[5] Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and Jie
Zhou, “DeepRAG: Thinking to retrieve step by step for
large language models,” 2025.
[6] Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou,
Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang,
Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng
Chen, “ReSearch: Learning to reason with search for
llms via reinforcement learning,” 2025.
[7] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong
Wen, “R1-Searcher: Incentivizing the search capability
in llms via reinforcement learning,” 2025.
[8] Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng,
Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin,
and Xipeng Qiu, “R3-RAG: Learning step-by-step rea-
soning and retrieval for llms via reinforcement learn-
ing,” 2025.
[9] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiao-
jun Chen, and Ruifeng Xu, “Enhancing noise robustness
of retrieval-augmented language models with adaptive
adversarial training,” inProceedings of the ACL, 2024,
pp. 10028–10039.
[10] Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Whang, “Why so gullible?
enhancing the robustness of retrieval-augmented modelsagainst counterfactual noise,” inFindings of the NAACL
2024, 2024, pp. 2474–2495.
[11] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and
Nan Duan, “Query rewriting in retrieval-augmented
large language models,” inProceedings of the EMNLP,
2023, pp. 5303–5315.
[12] Tianhua Zhang, Kun Li, Hongyin Luo, Xixin Wu,
James R. Glass, and Helen M. Meng, “Adaptive query
rewriting: Aligning rewriters through marginal proba-
bility of conversational answers,” inProceedings of the
EMNLP, 2024, pp. 13444–13461.
[13] Tian Yu, Shaolei Zhang, and Yang Feng, “Auto-RAG:
Autonomous retrieval-augmented generation for large
language models,” 2024.
[14] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig, “Active retrieval aug-
mented generation,” inProceedings of the EMNLP,
2023, pp. 7969–7992.
[15] Aaron Jaech, Adam Kalai, Adam Lerer, and et al., “Ope-
nAI o1 system card,” 2024.
[16] Daya Guo, Dejian Yang, Haowei Zhang, and et al.,
“DeepSeek-R1: Incentivizing reasoning capability in
llms via reinforcement learning,” 2025.
[17] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo, “DeepSeek-
Math: Pushing the limits of mathematical reasoning in
open language models,” 2024.
[18] Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wen-
qiao Zhu, and Ji Liu, “BizFinBench: A business-
driven real-world financial benchmark for evaluating
llms,” 2025.
[19] Aixin Liu, Bei Feng, Bin Wang, and et al., “DeepSeek-
V2: A strong, economical, and efficient mixture-of-
experts language model,” 2024.
[20] An Yang, Anfeng Li, Baosong Yang, and et al., “Qwen3
technical report,” 2025.
[21] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
and et al., “The LLaMA 3 herd of models,” 2024.
[22] Xuanyu Zhang, Qing Yang, and Dongliang Xu, “Xu-
anYuan 2.0: A large Chinese financial chat model with
hundreds of billions parameters,” 2023.
[23] Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo,
Feng Chen, and Chi Zhang, “DianJin-R1: Evaluat-
ing and enhancing financial reasoning in large language
models,” 2025.