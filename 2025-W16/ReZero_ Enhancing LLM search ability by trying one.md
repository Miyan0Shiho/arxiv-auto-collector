# ReZero: Enhancing LLM search ability by trying one-more-time

**Authors**: Alan Dao, Thinh Le

**Published**: 2025-04-15 09:18:21

**PDF URL**: [http://arxiv.org/pdf/2504.11001v1](http://arxiv.org/pdf/2504.11001v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves Large Language Model (LLM)
performance on knowledge-intensive tasks but depends heavily on initial search
query quality. Current methods, often using Reinforcement Learning (RL),
typically focus on query formulation or reasoning over results, without
explicitly encouraging persistence after a failed search. We introduce ReZero
(Retry-Zero), a novel RL framework that directly rewards the act of retrying a
search query following an initial unsuccessful attempt. This incentivizes the
LLM to explore alternative queries rather than prematurely halting. ReZero
demonstrates significant improvement, achieving 46.88% accuracy compared to a
25% baseline. By rewarding persistence, ReZero enhances LLM robustness in
complex information-seeking scenarios where initial queries may prove
insufficient.

## Full Text


<!-- PDF content starts -->

ReZero: Enhancing LLM search ability by trying
one-more-time
Alan Dao (Gia Tuan Dao)1, Thinh Le1
Menlo Research
alan@menlo.ai, thinh@menlo.ai
1Equal contribution.
April 16, 2025
Abstract
Retrieval-Augmented Generation (RAG) improves Large
Language Model (LLM) performance on knowledge-
intensive tasks but depends heavily on initial search
query quality. Current methods, often using Reinforce-
ment Learning (RL), typically focus on query formulation
or reasoning over results, without explicitly encouraging
persistence after a failed search. We introduce ReZero
(Retry-Zero), a novel RL framework that directly rewards
the act of retrying a search query following an initial
unsuccessful attempt. This incentivizes the LLM to ex-
plore alternative queries rather than prematurely halting.
ReZero demonstrates significant improvement, achieving
46.88% accuracy compared to a 25% baseline. By reward-
ing persistence, ReZero enhances LLM robustness in com-
plex information-seeking scenarios where initial queries
may prove insufficient.
1 Introduction
Large Language Models (LLMs) have demonstrated im-
pressive capabilities in understanding and generating hu-
man language, yet they often struggle with tasks requiring
access to up-to-date, specific, or proprietary knowledge
not captured during their pre-training phase Fan et al.
[2024], Jeong et al. [2024]. Retrieval-Augmented Gen-
eration (RAG) has emerged as a dominant paradigm to
address this limitation, equipping LLMs with the ability
to query external knowledge sources, typically search en-
gines or vector databases, to ground their responses in
relevant information Kim et al. [2024].
However, the effectiveness of RAG systems hinges crit-
ically on the quality of the interaction between the LLM
and the retrieval system. Complex questions often neces-
sitate multi-step reasoning, where information gathered
in one step informs subsequent queries or reasoning pro-
cesses Trivedi et al. [2023], Yao et al. [2023]. Even with
advanced techniques, the initial query formulated by an
LLM might be suboptimal, ambiguous, or fail to retrieve
the necessary information on the first attempt. Exist-
ing approaches often focus on refining the reasoning pro-
cess over retrieved documents Madaan et al. [2023]. For
instance, methods like ReARTeR Sun et al. [2025] uti-
Figure 1: Receives a reward signal for retrying after fail-
ure.
lize Process Reward Models (PRMs) and Process Expla-
nation Models (PEMs) to score and refine intermediate
reasoning steps within a RAG pipeline, aiming for trust-
worthy multi-step reasoning. Other approaches, such as
DeepRetrieval Jiang et al. [2025], employ reinforcement
learning (RL) to directly optimize the query generation
process itself, using retrieval metrics like recall or NDCG
as rewards to train the LLM to formulate more effective
queries through trial and error.
While these methods enhance reasoning quality and
query effectiveness, they primarily reward the successful
outcome of a reasoning step or a retrieval action. They
often implicitly assume that a single, well-formed query
or reasoning step is the goal. However, they may not
sufficiently incentivize the model to persist when an ini-
tial search attempt yields unsatisfactory results. The in-
herent difficulty in formulating a perfect query from the
outset, especially for complex or exploratory information
needs, suggests that the ability to recognize inadequacy
and retry the search process could be a valuable, yet un-
derexplored, capability.
1arXiv:2504.11001v1  [cs.CL]  15 Apr 2025

To address this gap, we introduce ReZero (Retry-Zero),
a novel framework designed to improve LLM search abil-
ity by explicitly incentivizing the model to ”try one more
time.” Our core idea is simple: we augment the standard
reinforcement learning process for RAG with a specific
reward component that encourages the model to retry its
search query if the initial attempt is deemed insufficient.
Unlike approaches focusing solely on the correctness of
the final answer or the relevance of retrieved chunks,
ReZero directly rewards the act of retrying the search in-
teraction. This is based on the intuition that persistence-
trying again, potentially with a reformulated query based
on the initial failure or intermediate reasoning-can sig-
nificantly improve the chances of eventually finding the
necessary information, echoing the real-world problem-
solving strategy: ”if at first you don’t succeed, try, try
again.”
ReZero employs a reinforcement learning strategy
where the LLM interacts with a search environment. The
reward function is designed not only to reflect the quality
of the final answer but also to provide a positive signal
when the model executes a ”retry” action following an
initial search. This encourages the model to explore dif-
ferent querying strategies and learn when persistence is
beneficial, rather than giving up or hallucinating after a
single failed attempt.
The main contributions of this work are:
•We propose ReZero, a novel RL-based framework
that explicitly rewards the act of retrying search
queries within a RAG system, fostering persistence
in information seeking.
•We introduce a modified reward function that in-
centivizes the ”retry” mechanism, encouraging explo-
ration and potentially leading to better performance
on complex, knowledge-intensive tasks where initial
searches may fail.
•We position ReZero relative to existing work on RAG
reasoning and query optimization, highlighting its
unique focus on rewarding persistence rather than
solely immediate outcomes.
2 Related Work
Our work builds upon and extends several lines of re-
search in enhancing LLMs, particularly in the context of
Retrieval-Augmented Generation (RAG) and Reinforce-
ment Learning (RL).
2.1 Retrieval-Augmented Generation (RAG)
RAG systems Fan et al. [2024], Jeong et al. [2024], Xu
et al. [2024] have become a standard method for ground-
ing LLM outputs in external knowledge, improving factu-
ality and handling knowledge-intensive tasks. Early RAG
approaches often involved a single retrieval step before
generation. However, complex tasks frequently require
multiple interactions with the knowledge source, lead-
ing to research in multi-step RAG Trivedi et al. [2023],Yao et al. [2023]. Methods like Self-Ask Press et al.
[2023] and IRCoT Trivedi et al. [2023] integrate Chain-
of-Thought (CoT) reasoning with iterative retrieval, al-
lowing the model to decompose questions and gather in-
formation incrementally. While these methods improve
reasoning over retrieved context, they often rely on so-
phisticated prompting or assume the model inherently
knows when and how to retrieve effectively. ReZero differs
by focusing on the robustness of the retrieval interaction
itself, specifically by encouraging retries when initial at-
tempts fail, a dimension less explored in iterative RAG
frameworks which often focus on sequential information
gathering rather than correcting failed searches.
2.2 Learning and Search for Reasoning in RAG
Recent work has explored using learning-based methods,
particularly RL and process supervision, to enhance the
reasoning capabilities of RAG systems, often following
the ”learning and search” principle [32].
Sun et al. [2025] exemplifies this direction by focusing
on trustworthy multi-step reasoning. It employs Process
Reward Models (PRM) to score intermediate reasoning
steps and Process Explanation Models (PEM) to provide
critiques for refinement, using techniques like MCTS and
preference optimization (KTO) Ethayarajh et al. [2024],
Pang et al. [2024]. ReARTeR aims to improve the qual-
ity and reliability of each reasoning step. While it uses
rewards and refinement, its focus is on the correctness
and trustworthiness of the reasoning process given the
retrieved information. ReZero, in contrast, focuses on
the search interaction itself, specifically incentivizing the
model to retry the search if the initial attempt fails or
seems inadequate, adding a layer of persistence before or
during the reasoning-over-documents phase.
Jiang et al. [2025] tackles the problem from the query
generation perspective. It uses RL (PPO) to train LLMs
to generate or rewrite queries that maximize retrieval per-
formance (e.g., Recall, NDCG) on various retrieval sys-
tems, including real-world search engines. DeepRetrieval
learns effective query formulation through trial-and-error,
optimizing for the outcome of the search. ReZero com-
plements this by adding a reward for the process of retry-
ing. While DeepRetrieval optimizes the quality of a sin-
gle (potentially refined) query attempt for maximal suc-
cess, ReZero encourages the model to make multiple at-
tempts if necessary, rewarding the persistence mechanism
directly.
2.3 Reinforcement Learning for LLM Alignment
and Reasoning
RL has become a cornerstone for aligning LLMs with hu-
man preferences (RLHF) Ouyang et al. [2022] and en-
hancing specific capabilities like reasoning DeepSeek-AI
et al. [2025] and tool use Schick et al. [2023]. Meth-
ods like ReFT Wu et al. [2024] use RL (e.g., PPO) to
fine-tune LLMs for reasoning tasks based on outcome or
process rewards. The concept of using rewards to shape
LLM behavior is central to ReZero. However, while prior
2

work has used RL to optimize reasoning steps [21, 41] or
query generation Jiang et al. [2025], ReZero introduces a
novel application of RL by incorporating a reward signal
specifically tied to the retry action in a search context.
This aligns with the broader goal of using RL to encour-
age desirable behaviors, but targets the specific behavior
of persistence in information retrieval.
The idea of iterative improvement is also present in
self-correction or self-refinement methods Huang et al.
[2024], Madaan et al. [2023], where LLMs critique and
revise their own generated outputs. These methods typi-
cally focus on refining the generated text (e.g., reasoning
steps, final answers) based on internal checks or external
feedback (like verifier scores). ReZero differs in its focus:
instead of refining the LLM’s generated output, it encour-
ages retrying the interaction with the external search tool,
addressing potential failures at the information-gathering
stage itself.
In summary, ReZero occupies a unique space by lever-
aging RL to explicitly incentivize persistence in the search
process within RAG. While related to work on improv-
ing RAG reasoning (ReARTeR) and query optimization
(DeepRetrieval), its core novelty lies in rewarding the
”retry” mechanism, aiming to make LLMs more robust
and effective information seekers, especially when faced
with complex or initially ambiguous queries.
3 Methodology
3.1 Overview
ReZero is a reinforcement learning (RL) framework de-
signed to enhance the search capabilities of large language
models (LLMs) in retrieval-augmented generation (RAG)
systems. Inspired by recent advancements in RL for rea-
soning tasks DeepSeek-AI et al. [2025] and motivated by
findings suggesting RL fosters better generalization com-
pared to supervised fine-tuning, ReZero utilizes Group
Relative Policy Optimization (GRPO) Shao et al. [2024]
to explicitly incentivize persistence—rewarding the model
for retrying search queries when initial attempts fail.
3.2 Reinforcement Learning Framework
ReZero operates within a search environment where the
LLM interacts with an external retrieval system. We em-
ploy the Group Relative Policy Optimization (GRPO) al-
gorithm, noted for its effectiveness in training LLMs for
reasoning tasks without requiring a separate critic model.
The RL loop involves standard components:
•State: The current conversation history, including
the user’s prompt, the LLM’s previous responses
(potentially including <search> and<information>
tags), and retrieved information.
•Action: The LLM’s generation, which could be an
internal thought process ( <think> ), a search query
(<search> ), a final answer ( <answer> ), or, critically,
the decision to issue another search query after a
previous one.•Reward: A scalar signal derived from evaluating the
LLM’s outputs against predefined criteria using the
reward functions described below. These functions
collectively act as a self-teacher.
•Policy: The LLM’s strategy for generating actions,
fine-tuned using GRPO to maximize cumulative re-
ward.
3.3 Reward Functions
ReZero employs multiple reward functions to provide the
training signal for GRPO. These functions evaluate dif-
ferent aspects of the LLM’s generation:
1.reward correctness : Evaluates the final answer’s
accuracy against a ground-truth, checks response
structure validity, and outputs a binary reward. The
binary reward is determined by the model itself, act-
ing as a self-judge (LLM-as-a-Judge) Gu et al. [2025]
using the base model’s capabilities.
2.reward format : Ensures adherence to the required
conversational format and tag usage (e.g., tag se-
quence, valid markup), outputting a binary reward.
3.reward retry : This reward function encourages the
model to persist when initial search attempts do not
yield sufficient information. It assigns a positive re-
ward for each subsequent <search> query issued af-
ter the first one within a single generation sequence
(i.e., for retries). The magnitude of the reward could
potentially diminish with each additional retry to en-
courage efficiency. Crucially, this reward is condi-
tional on task completion, it is only awarded if the
model’s final generated output in the sequence in-
cludes the complete <answer>...</answer> tags. If
the sequence concludes without a well-formed answer
enclosed in these tags, the reward retry component
contributes zero to the total reward for that trajec-
tory, regardless of how many retries were performed.
This mechanism prevents the model from learning
to accumulate reward simply by retrying repeatedly
without ever successfully generating a final answer.
4.reward emchunk : Verifies if the correct information
chunk was retrieved by comparing the content in
<information> tags against a ground-truth chunk
using exact matching, outputting a binary reward.
5.reward search strategy : Evaluates the quality of
the search process by checking adherence to a desired
conversational flow: initiating a broad <search> , an-
alyzing retrieved <information> (verified by spe-
cific keywords within <think> tags), executing sub-
sequent refined <search> queries based on this anal-
ysis, and finally synthesizing an <answer> grounded
in the analyzed information. Outputs a graded score
(0.0-1.0) based on the successful execution of these
sequential phases.
3

6.reward search diversity : Assesses the variety
within the sequence of ¡search¿ queries used during
generation. It rewards distinct query concepts and
semantic dissimilarity between queries (measured us-
ing normalized string comparison). Bonus rewards
are allocated for the effective use of diverse search op-
erators (e.g., site:, ””, OR). Penalties are applied to
discourage submitting exact duplicate or highly simi-
lar queries, promoting broader exploration. Outputs
a graded score (0.0-1.0) rewarding unique queries and
operator diversity, while penalizing repetition and
high similarity.
These functions collectively guide the policy towards cor-
rectness, format adherence, effective information gather-
ing, and search persistence.
3.4 Training Process
The LLM is fine-tuned directly from a pre-trained base
model using reinforcement learning, specifically employ-
ing the Group Relative Policy Optimization (GRPO) al-
gorithm. The training operates within an interactive
framework involving a verifier-in this case a search engine
Snell et al. [2025], drawing parallels to setups studied for
improving model generalization through RL Chu et al.
[2025]. The initial reference policy ( πref) for GRPO is the
base pre-trained model itself.
•Iterative Interaction Loop (Rollout): The core
training dynamic involves the LLM interacting with
the search engine (verifier). As illustrated in Figure
1, for a given input prompt, the LLM (starting from
the base pre-trained policy) generates a response se-
quence. This sequence can include emitting a search
query ( <search> ). The verifier processes this query
and returns information chunks ( <information> ).
The LLM receives these chunks and continues gener-
ating. Critically, the model might iteratively repeat
this query-retrieval process within the same genera-
tion sequence if the initial results are deemed insuffi-
cient, before finally producing an answer ( <answer> ).
This iterative generation forms one complete trajec-
tory or rollout per input prompt.
•Reward Calculation: Upon completion of the en-
tire generated sequence, the sequence is evaluated.
The total reward for this trajectory is computed
using the suite of reward functions described in
Section 3.3( reward correctness ,reward format ,
reward retry ,reward emchunk ). This aggregated
reward reflects the overall quality and effectiveness
of the generated sequence, including the persistence
demonstrated through retries.
•Policy Update (GRPO): The calculated rewards
for multiple sampled trajectories (generated using
the current policy πθ) are fed into the GRPO al-
gorithm. GRPO updates the LLM’s parameters ( θ)
by comparing the reward of each trajectory against
the average reward of the group (batch), aiming toincrease the probability of generating higher-reward
sequences. A KL divergence term against the refer-
ence policy ( πref, which is the initial base model) is
typically used to stabilize training and prevent dras-
tic deviations from the initial capabilities.
•Noise Injection for Robustness: To specifically
strengthen the model’s ability to generalize its retry
strategy, we introduce noise during training at the
vector database level. This simulates imperfect re-
trieval by randomly perturbing the relevance or qual-
ity of returned chunks for some queries. This en-
courages the LLM to learn robust retry mechanisms
rather than overfitting to scenarios where the first
search attempt always yields perfect results.
This process directly fine-tunes the base LLM using
RL, teaching it not only to answer correctly but also to
strategically and persistently use the search tool, even
when facing retrieval imperfections, without an interme-
diate supervised fine-tuning stage.
4 Experiments and Results
Our experimental setup implements the ReZero frame-
work detailed in Section 3. We fine-tuned a pre-trained
large language model using Group Relative Policy Opti-
mization (GRPO) Shao et al. [2024], following the re-
inforcement learning approach described in the paper.
The training process operates within an interactive en-
vironment involving a search engine acting as a veri-
fier Snell et al. [2025], without an intermediate super-
vised fine-tuning stage. The implementation utilized the
unsloth training library and borrowed significantly from
the codebase presented in dCaples [2022]. For our exper-
iments, we employed the Apollo 3 mission dataset. This
dataset was divided into 341 distinct data chunks, with
32 chunks specifically reserved for evaluating model per-
formance. The training was executed on a single NVIDIA
H200 GPU and ran for a total of 1000 steps, which corre-
sponds to approximately 3 epochs over the training por-
tion of the dataset (309 chunks).We choose Llama3.2-
3B-Insruct as the base model for training Grattafiori
et al. [2024]. To isolate and assess the impact of the pro-
posed reward retry mechanism, we conducted experi-
ments comparing two model configurations:
•Baseline: This model was trained using
three reward functions: reward correctness ,
reward format , and reward emchunk . It lacked the
explicit incentive to retry search queries provided by
the fourth reward function.
•ReZero (with reward retry ):This model repre-
sents the full implementation of our proposed frame-
work. It was trained using all four reward functions,
crucially including the reward retry component de-
signed to encourage persistence by rewarding sub-
sequent search attempts within a single generation
sequence, conditional on successful final answer gen-
eration.
4

Both models started from the same base pre-trained
weights and underwent the same fundamental RL train-
ing procedure using GRPO, differing only in the inclu-
sion of the reward retry signal. This controlled com-
parison allows for a direct evaluation of the contribu-
tion of rewarding the retry action. Model performance
was evaluated periodically on the held-out evaluation set
(32 chunks) throughout the 1000 training steps. The pri-
mary metric reported is accuracy, defined as the percent-
age of correctly answered queries within the evaluation
set. The results, depicted in Figure 2, clearly indicate
the effectiveness of the reward retry component. The
ReZero model achieved a peak accuracy of 46.88% at
250 training steps. In contrast, the Baseline model, with-
out the retry incentive, reached a maximum accuracy of
only25.00% at 350 steps. The ReZero model not only
achieved a significantly higher peak performance but also
demonstrated a faster initial learning rate compared to
the baseline. Both models exhibited a decline in accuracy
after reaching their peaks, potentially due to overfitting
or instability in the later stages of RL training, with ac-
curacy dropping to 0% by step 700 for the Baseline and
step 450 for the ReZero model (persisting at 0
010020030040050060070080090010000152535455565758595
46.88%
Training StepsAccuracy (%)Accuracy Comparison during Training
ReZero (with reward retry )
Baseline (without reward retry )
Figure 2: Comparison of evaluation accuracy between the
ReZero model (incorporating the reward retry compo-
nent) and the Baseline model (lacking the retry incentive)
over 1000 training steps on the held-out Apollo 3 dataset
chunks. Peak accuracies are highlighted.
The substantial gap in peak accuracy (46.88% vs
25.00%) strongly suggests that explicitly rewarding the
act of retrying search queries via the reward retry func-
tion significantly enhances the model’s ability to effec-
tively utilize the search tool and ultimately arrive at
correct answers, particularly in scenarios where initial
queries might be insufficient.5 Discussion
The experimental results presented in Section 4 pro-
vide compelling evidence for the efficacy of the pro-
posed ReZero framework, particularly the contribution
of the reward retry component. This mechanism mir-
rors aspects of human information-seeking, where ini-
tial attempts often require refinement or alternative ap-
proaches. The most striking finding is the substantial
performance gap observed between the ReZero model and
the Baseline configuration, as depicted in Figure 2. The
ReZero model, equipped with the incentive to retry search
queries, achieved a peak accuracy of 46.88%, nearly dou-
ble the 25.00% peak accuracy attained by the Baseline
model which lacked this specific reward signal. This sig-
nificant difference strongly validates our central hypoth-
esis: explicitly rewarding the act of retrying a search
query enhances the LLM’s ability to navigate informa-
tion retrieval challenges. Crucially, this reward was con-
ditional on the successful generation of a final answer
(as detailed in Section 3), ensuring it incentivized pro-
ductive persistence rather than simply encouraging re-
peated, fruitless queries. The faster initial learning rate
observed in the ReZero model further suggests that en-
couraging persistence accelerates the model’s adaptation
to effective search strategies. The use of noise injection
during training likely amplified the scenarios where initial
searches failed, potentially providing more effective learn-
ing signals for the reward retry mechanism compared to
a cleaner retrieval environment, although the precise in-
teraction warrants further study.
However, the performance trajectory also reveals a crit-
ical challenge and an area for future investigation. Both
models, despite their differing peak performances, exhib-
ited a notable decline in accuracy after reaching their re-
spective zeniths (ReZero after 250 steps, Baseline after
350 steps), eventually collapsing to 0% accuracy on the
evaluation set. This phenomenon, while not uncommon
in reinforcement learning Chu et al. [2025], suggests that
the current limitations may lie less in the core concept
of rewarding retries and more in the specifics of the RL
training process itself. Factors such as potential overfit-
ting to the training data chunks, instability inherent in
the GRPO algorithm over extended training, or subopti-
mal balancing of the different reward components could
contribute to this degradation. The relatively short train-
ing duration (1000 steps, 3 epochs) might not be suffi-
cient to achieve stable convergence, or conversely, it might
be pushing the model into unstable policy regions quickly.
This points towards a need for further research into opti-
mizing the RL training regime for sustained performance.
Furthermore, a significant limitation of this study is
its reliance on a single dataset derived from the Apollo
3 mission. While this dataset provided a controlled envi-
ronment for comparing the two model configurations, it
represents a specific and relatively narrow domain. The
types of questions, the nature of the information within
the chunks, and the overall complexity might not be rep-
resentative of the diverse scenarios LLMs encounter in
5

real-world RAG applications. Therefore, the general-
izability of the observed performance gains to broader
knowledge domains or different types of information-
seeking tasks remains an open question. The pronounced
performance gap seen here might be amplified or dimin-
ished when applied to datasets with different characteris-
tics.
Future research should prioritize addressing these lim-
itations. Evaluating ReZero across a wider range of
datasets spanning multiple domains and varying query
complexities is essential to ascertain the robustness and
general applicability of the approach. Concurrently, fur-
ther investigation into stabilizing the RL training process
is crucial. This could involve exploring alternative RL
algorithms, refining hyperparameters, implementing ad-
vanced regularization, or dynamically adjusting reward
components. Additionally, exploring variations of the
reward retry function itself, such as incorporating di-
minishing returns or conditioning on retrieval quality im-
provement, could yield further benefits. It would also be
valuable to conduct a qualitative analysis of the generated
search queries to understand how the model adapts its
strategy during retry attempts – whether it learns mean-
ingful reformulations or simpler persistence patterns. An-
alyzing the practical trade-off between accuracy gains
and the potential increase in latency or computational
cost associated with additional search queries will also
be important for real-world deployment. Finally, com-
bining ReZero with orthogonal techniques, such as ad-
vanced query rewriting Jiang et al. [2025] or sophisticated
reasoning-over-retrieved-documents methods Sun et al.
[2025], could lead to synergistic benefits.
6 Conclusion
This work addressed the challenge of enhancing the ro-
bustness of Retrieval-Augmented Generation (RAG) sys-
tems, particularly when initial search queries fail to re-
trieve the necessary information. We introduced ReZero
(Retry-Zero), a novel reinforcement learning framework
built upon Group Relative Policy Optimization (GRPO),
designed to explicitly incentivize persistence in the search
process. Unlike existing approaches that primarily fo-
cus on query formulation or reasoning over retrieved re-
sults, ReZero incorporates a specific reward component,
reward retry , which encourages the Large Language
Model (LLM) to attempt subsequent search queries fol-
lowing an unsuccessful initial attempt, conditional on suc-
cessfully generating a final answer.
Our experiments, conducted on the Apollo 3 mis-
sion dataset, demonstrated the significant impact of
this approach. The ReZero model, trained with the
reward retry incentive, achieved a peak accuracy of
46.88%, substantially outperforming the 25.00% peak ac-
curacy of a baseline model trained without this specific
reward signal. This result strongly supports our hypoth-
esis that explicitly rewarding the act of retrying enhances
the LLM’s ability to overcome initial search failures and
improve task success rates.Despite the promising results, we acknowledge limita-
tions. The observed decline in performance after reaching
peak accuracy highlights challenges related to the sta-
bility of the RL training process over extended periods,
suggesting a need for further investigation into optimiza-
tion techniques or regularization methods. Furthermore,
the evaluation was confined to a single, specific domain
(Apollo 3 mission data), limiting the current claims of
generalizability across diverse knowledge areas and query
types.
Future research should prioritize validating ReZero
across a wider range of datasets and task complexities
to ascertain its broader applicability. Stabilizing the RL
training dynamics is crucial for achieving sustained per-
formance. Exploring refinements to the reward retry
function itself (e.g., diminishing returns, conditioning on
retrieval improvement), conducting qualitative analyses
of the learned retry strategies, investigating the latency
and computational cost trade-offs, and exploring the inte-
gration of ReZero with complementary RAG techniques
(like advanced query rewriting or reasoning methods) rep-
resent promising avenues for further enhancement.
In conclusion, ReZero offers a valuable contribution by
demonstrating that directly rewarding persistence—the
willingness to ”try one more time”—can significantly im-
prove the effectiveness of LLMs in complex information-
seeking scenarios. This work highlights the potential of
incorporating mechanisms that mirror human problem-
solving strategies into the training of capable and robust
AI systems operating within RAG frameworks.
References
Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang
Tong, Saining Xie, Dale Schuurmans, Quoc V. Le,
Sergey Levine, and Yi Ma. Sft memorizes, rl gener-
alizes: A comparative study of foundation model post-
training, 2025. URL https://arxiv.org/abs/2501.
17161 .
dCaples. Autodidact. https://github.com/dCaples/
AutoDidact , 2022. Accessed: 2025-04-11.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing
Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda
Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Er-
hang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo
Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao,
Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo,
Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang
Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian
Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang
Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong
Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu,
6

Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui
Tang, Meng Li, Miaojun Wang, Mingming Li, Ning
Tian, Panpan Huang, Peng Zhang, Qiancheng Wang,
Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang,
Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi
Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen,
Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng
Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shao-
qing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu
Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen
Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao
Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin
Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li,
Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin,
Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxi-
ang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang,
Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang
Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun,
Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yil-
iang Xiong, Ying He, Yishi Piao, Yisong Wang, Yix-
uan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yu-
jia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You,
Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu,
Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu,
Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan,
Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu,
Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng
Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu,
Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng
Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and
Zhen Zhang. Deepseek-r1: Incentivizing reasoning ca-
pability in llms via reinforcement learning, 2025. URL
https://arxiv.org/abs/2501.12948 .
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan
Jurafsky, and Douwe Kiela. Model alignment as
prospect theoretic optimization. In Proceedings of the
41st International Conference on Machine Learning ,
ICML’24. JMLR.org, 2024.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. A survey on rag meeting llms: Towards retrieval-
augmented large language models. In Proceedings of the
30th ACM SIGKDD Conference on Knowledge Discov-
ery and Data Mining , KDD ’24, page 6491–6501, New
York, NY, USA, 2024. Association for Computing Ma-
chinery. ISBN 9798400704901. doi: 10.1145/3637528.
3671470. URL https://doi.org/10.1145/3637528.
3671470 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, Arun Rao, Aston Zhang, Aurelien Ro-
driguez, Austen Gregerson, Ava Spataru, BaptisteRoziere, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris
Marra, Chris McConnell, Christian Keller, Christophe
Touret, Chunyang Wu, Corinne Wong, Cristian Can-
ton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel
Song, Danielle Pintz, Danny Livshits, Danny Wy-
att, David Esiobu, Dhruv Choudhary, Dhruv Maha-
jan, Diego Garcia-Olano, Diego Perino, Dieuwke Hup-
kes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova,
Emily Dinan, Eric Michael Smith, Filip Radenovic,
Francisco Guzm´ an, Frank Zhang, Gabriel Synnaeve,
Gabrielle Lee, Georgia Lewis Anderson, Govind That-
tai, Graeme Nail, Gregoire Mialon, Guan Pang,
Guillem Cucurell, Hailey Nguyen, Hannah Korevaar,
Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arri-
eta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Ev-
timov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Gef-
fert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet
Shah, Jelmer van der Linde, Jennifer Billock, Jenny
Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu
Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bit-
ton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua
Johnstun, Joshua Saxe, Junteng Jia, Kalyan Va-
suden Alwala, Karthik Prasad, Kartikeya Upasani,
Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone,
Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuen-
ley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren
Rantala-Yeary, Laurens van der Maaten, Lawrence
Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish
Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat,
Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti,
Mannat Singh, Manohar Paluri, Marcin Kardas, Maria
Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya
Pavlova, Melanie Kambadur, Mike Lewis, Min Si,
Mitesh Kumar Singh, Mona Hassan, Naman Goyal,
Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoy-
chev, Niladri Chatterji, Ning Zhang, Olivier Duchenne,
Onur C ¸elebi, Patrick Alrassy, Pengchuan Zhang, Peng-
wei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava,
Pratik Dubal, Praveen Krishnan, Punit Singh Koura,
Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srini-
vasan, Raj Ganapathy, Ramon Calderer, Ricardo Sil-
veira Cabral, Robert Stojnic, Roberta Raileanu, Ro-
han Maheswari, Rohit Girdhar, Rohit Patel, Romain
Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross
Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hos-
seini, Sahana Chennabasappa, Sanjay Singh, Sean
Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang
Nie, Sharan Narang, Sharath Raparthy, Sheng Shen,
Shengye Wan, Shruti Bhosale, Shun Zhang, Simon
Vandenhende, Soumya Batra, Spencer Whitman, Sten
Sootla, Stephane Collot, Suchin Gururangan, Syd-
ney Borodinsky, Tamar Herman, Tara Fowler, Tarek
Sheasha, Thomas Georgiou, Thomas Scialom, To-
bias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ra-
manathan, Viktor Kerkez, Vincent Gonguet, Virginie
Do, Vish Vogeti, V´ ıtor Albiero, Vladan Petrovic, Wei-
wei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers,
7

Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xi-
aoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia,
Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yas-
mine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue
Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng
Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh,
Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam
Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva
Goldstand, Ajay Menon, Ajay Sharma, Alex Boesen-
berg, Alexei Baevski, Allie Feinstein, Amanda Kallet,
Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu,
Andres Alvarado, Andrew Caples, Andrew Gu, Andrew
Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchan-
dani, Annie Dong, Annie Franco, Anuj Goyal, Apara-
jita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan,
Beau James, Ben Maurer, Benjamin Leonhardi, Bernie
Huang, Beth Loyd, Beto De Paola, Bhargavi Paran-
jape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock,
Bram Wasti, Brandon Spence, Brani Stojkovic, Brian
Gamido, Britt Montalvo, Carl Parker, Carly Burton,
Catalina Mejia, Ce Liu, Changhan Wang, Changkyu
Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu,
Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cyn-
thia Gao, Damon Civin, Dana Beaty, Daniel Kreymer,
Daniel Li, David Adkins, David Xu, Davide Testug-
gine, Delia David, Devi Parikh, Diana Liskovich, Di-
dem Foss, Dingkang Wang, Duc Le, Dustin Holland,
Edward Dowling, Eissa Jamil, Elaine Montgomery,
Eleonora Presani, Emily Hahn, Emily Wood, Eric-
Tuan Le, Erik Brinkman, Esteban Arcaute, Evan
Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng
Tian, Filippos Kokkinos, Firat Ozgenel, Francesco
Caggioni, Frank Kanayet, Frank Seide, Gabriela Med-
ina Florez, Gabriella Schwarz, Gada Badeer, Geor-
gia Swee, Gil Halpern, Grant Herman, Grigory Sizov,
Guangyi, Zhang, Guna Lakshminarayanan, Hakan
Inan, Hamid Shojanazeri, Han Zou, Hannah Wang,
Hanwen Zha, Haroun Habeeb, Harrison Rudolph, He-
len Suk, Henry Aspegren, Hunter Goldman, Hongyuan
Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov,
Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake
Weissman, James Geboski, James Kohli, Janice Lam,
Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff
Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein,
Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang,
Joe Cummings, Jon Carvill, Jon Shepard, Jonathan
McPhie, Jonathan Torres, Josh Ginsburg, Junjie
Wang, Kai Wu, Kam Hou U, Karan Saxena, Kar-
tikay Khandelwal, Katayoun Zand, Kathy Matosich,
Kaushik Veeraraghavan, Kelly Michelena, Keqian Li,
Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle
Huang, Lailin Chen, Lakshya Garg, Lavender A, Le-
andro Silva, Lee Bell, Lei Zhang, Liangpeng Guo,
Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Ma-
dian Khabsa, Manav Avalani, Manish Bhatt, Martynas
Mankus, Matan Hasson, Matthew Lennie, Matthias
Reso, Maxim Groshev, Maxim Naumov, Maya Lathi,
Meghan Keneally, Miao Liu, Michael L. Seltzer, MichalValko, Michelle Restrepo, Mihir Patel, Mik Vyatskov,
Mikayel Samvelyan, Mike Clark, Mike Macey, Mike
Wang, Miquel Jubert Hermoso, Mo Metanat, Moham-
mad Rastegari, Munish Bansal, Nandhini Santhanam,
Natascha Parks, Natasha White, Navyata Bawa, Nayan
Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta,
Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng,
Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem
Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan
Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux,
Piotr Dollar, Polina Zvyagina, Prashant Ratanchan-
dani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel
Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu
Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky
Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin
Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh,
Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru
Pan, Saurabh Mahajan, Saurabh Verma, Seiji Ya-
mamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun
Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy
Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang,
Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji
Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen
Chen, Steve Kehoe, Steve Satterfield, Sudarshan
Govindaprasad, Sumit Gupta, Summer Deng, Sungmin
Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury,
Sydney Goldman, Tal Remez, Tamar Glaser, Tamara
Best, Thilo Koehler, Thomas Robinson, Tianhe Li,
Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook
Shaked, Varun Vontimitta, Victoria Ajayi, Victoria
Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal
Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu
Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang,
Wenwen Jiang, Wes Bouaziz, Will Constable, Xi-
aocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu,
Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu,
Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang,
Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen
Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait,
Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu
Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd
of models, 2024. URL https://arxiv.org/abs/2407.
21783 .
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang,
Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A
survey on llm-as-a-judge, 2025. URL https://arxiv.
org/abs/2411.15594 .
Jie Huang, Xinyun Chen, Swaroop Mishra,
Huaixiu Steven Zheng, Adams Wei Yu, Xinying
Song, and Denny Zhou. Large language models cannot
self-correct reasoning yet. In The Twelfth International
Conference on Learning Representations , 2024. URL
https://openreview.net/forum?id=IkmD3fKBPQ .
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. Adaptive-RAG: Learning
8

to adapt retrieval-augmented large language models
through question complexity. In Kevin Duh, He-
lena Gomez, and Steven Bethard, editors, Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers) , pages 7036–7050, Mexico City, Mexico,
June 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.naacl-long.389. URL https:
//aclanthology.org/2024.naacl-long.389/ .
Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian,
SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei
Han. Deepretrieval: Hacking real search engines and
retrievers with large language models via reinforcement
learning, 2025. URL https://arxiv.org/abs/2503.
00223 .
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, and
Jinwoo Shin. Sure: Summarizing retrievals using an-
swer candidates for open-domain qa of llms, 2024. URL
https://arxiv.org/abs/2404.13081 .
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha
Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank
Gupta, Bodhisattwa Prasad Majumder, Katherine
Hermann, Sean Welleck, Amir Yazdanbakhsh, and Pe-
ter Clark. Self-refine: Iterative refinement with self-
feedback. In Thirty-seventh Conference on Neural In-
formation Processing Systems , 2023. URL https:
//openreview.net/forum?id=S37hOerQLB .
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul F Christiano, Jan Leike, and Ryan Lowe.
Training language models to follow instructions
with human feedback. In S. Koyejo, S. Mohamed,
A. Agarwal, D. Belgrave, K. Cho, and A. Oh, ed-
itors, Advances in Neural Information Processing
Systems , volume 35, pages 27730–27744. Curran Asso-
ciates, Inc., 2022. URL https://proceedings.
neurips.cc/paper_files/paper/2022/file/
b1efde53be364a73914f58805a001731-Paper-Conference.
pdf.
Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun
Cho, Sainbayar Sukhbaatar, and Jason E Weston. Iter-
ative reasoning preference optimization. In The Thirty-
eighth Annual Conference on Neural Information Pro-
cessing Systems , 2024. URL https://openreview.
net/forum?id=4XIKfvNYvx .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. Measuring and nar-
rowing the compositionality gap in language mod-
els. In Houda Bouamor, Juan Pino, and KalikaBali, editors, Findings of the Association for Com-
putational Linguistics: EMNLP 2023 , pages 5687–
5711, Singapore, December 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.
findings-emnlp.378. URL https://aclanthology.
org/2023.findings-emnlp.378/ .
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. Tool-
former: Language models can teach themselves to use
tools. In Thirty-seventh Conference on Neural In-
formation Processing Systems , 2023. URL https:
//openreview.net/forum?id=Yacmpz84TH .
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junx-
iao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Push-
ing the limits of mathematical reasoning in open lan-
guage models, 2024. URL https://arxiv.org/abs/
2402.03300 .
Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Avi-
ral Kumar. Scaling LLM test-time compute opti-
mally can be more effective than scaling parameters
for reasoning. In The Thirteenth International Confer-
ence on Learning Representations , 2025. URL https:
//openreview.net/forum?id=4FWAwZtd2n .
Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang,
Kai Zheng, Jun Xu, Xiao Zhang, Song Yang, and
Han Li. Rearter: Retrieval-augmented reasoning with
trustworthy process rewarding, 2025. URL https:
//arxiv.org/abs/2501.07861 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. Interleaving retrieval with
chain-of-thought reasoning for knowledge-intensive
multi-step questions. In Anna Rogers, Jordan Boyd-
Graber, and Naoaki Okazaki, editors, Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers) , pages
10014–10037, Toronto, Canada, July 2023. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/
2023.acl-long.557. URL https://aclanthology.org/
2023.acl-long.557/ .
Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atti-
cus Geiger, Dan Jurafsky, Christopher D Manning,
and Christopher Potts. ReFT: Representation fine-
tuning for language models. In The Thirty-eighth An-
nual Conference on Neural Information Processing Sys-
tems, 2024. URL https://openreview.net/forum?
id=fykjplMc0V .
Fangyuan Xu, Weijia Shi, and Eunsol Choi. RE-
COMP: Improving retrieval-augmented LMs with con-
text compression and selective augmentation. In The
Twelfth International Conference on Learning Rep-
resentations , 2024. URL https://openreview.net/
forum?id=mlJLVigNHp .
9

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. React:
Synergizing reasoning and acting in language models.
InThe Eleventh International Conference on Learning
Representations , 2023. URL https://openreview.
net/forum?id=WE_vluYUL-X .
10