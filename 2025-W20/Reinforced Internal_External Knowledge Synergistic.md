# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu

**Published**: 2025-05-12 14:21:57

**PDF URL**: [http://arxiv.org/pdf/2505.07596v1](http://arxiv.org/pdf/2505.07596v1)

## Abstract
Retrieval-augmented generation (RAG) is a common strategy to reduce
hallucinations in Large Language Models (LLMs). While reinforcement learning
(RL) can enable LLMs to act as search agents by activating retrieval
capabilities, existing ones often underutilize their internal knowledge. This
can lead to redundant retrievals, potential harmful knowledge conflicts, and
increased inference latency. To address these limitations, an efficient and
adaptive search agent capable of discerning optimal retrieval timing and
synergistically integrating parametric (internal) and retrieved (external)
knowledge is in urgent need. This paper introduces the Reinforced
Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could
indentify its own knowledge boundary and prioritize the utilization of internal
knowledge, resorting to external search only when internal knowledge is deemed
insufficient. This is achieved using a novel knowledge-boundary aware reward
function and a knowledge-boundary aware training dataset. These are designed
for internal-external knowledge synergy oriented RL, incentivizing the model to
deliver accurate answers, minimize unnecessary retrievals, and encourage
appropriate external searches when its own knowledge is lacking. Evaluations
across multiple knowledge reasoning tasks demonstrate that IKEA significantly
outperforms baseline methods, reduces retrieval frequency significantly, and
exhibits robust generalization capabilities.

## Full Text


<!-- PDF content starts -->

Reinforced Internal-External Knowledge Synergistic
Reasoning for Efficient Adaptive Search Agent
Ziyang Huangαβ, Xiaowei Yuanαβγ, Yiming Juγ, Jun Zhaoαβ, Kang Liuαβ
αInstitute of Automation, Chinese Academy of Sciences
βUniversity of Chinese Academy of Sciences
γBeijing Academy of Artificial Intelligence
huangziyang2023@ia.ac.cn
Code :https://github.com/hzy312/knowledge-r1
Abstract
Retrieval-augmented generation (RAG) is a common strategy to reduce hallucina-
tions in Large Language Models (LLMs). While reinforcement learning (RL) can
enable LLMs to act as search agents by activating retrieval capabilities, existing
ones often underutilize their internal knowledge. This might lead to redundant re-
trievals, potential harmful knowledge conflicts, and increased inference latency. To
address these limitations, an efficient adaptive search agent capable of discerning
optimal retrieval timing and synergistically integrating parametric (internal) and
retrieved (external) knowledge is in urgent need. This paper introduces the Rein-
forced Internal-External Knowledge Synergistic R Easoning Agent ( IKEA ), which
could indentify its own knowledge boundary and prioritize the utilization of internal
knowledge, resorting to external search only when internal knowledge is deemed
insufficient. This is achieved using a novel knowledge-boundary aware reward
function and a knowledge-boundary aware training dataset. These are designed
for internal-external knowledge synergy oriented RL, incentivizing the model to
deliver accurate answers, minimize unnecessary retrievals, and encourage appro-
priate retrievals when its own knowledge is lacking. Evaluations across multiple
knowledge-intensive reasoning tasks demonstrate that IKEA significantly outper-
forms baseline methods, reduces retrieval frequency significantly, and exhibits
robust generalization capabilities.
1 Introduction
The advancement of large-scale reinforcement learning (RL) with verifiable reward systems [ 31,34]
has significantly enhanced the capabilities of reasoning models like Deepseek R1 [ 5]. For knowledge-
intensive tasks [ 8], R1-like models could activate their internal pre-trained knowledge through
reasoning. However, constrained by the finite nature of pre-training corpora and the dynamic essence
of world knowledge, they remain susceptible to hallucinations [ 13]. To address the knowledge
deficiencies, current research typically empowers models to invoke search engines, essentially
training them as search agents [ 16,33,3]. During reinforcement learning, these models progressively
learn to decompose tasks and retrieve relevant knowledge for each subtask to aid reasoning. Despite
this, the approach remains suboptimal for several reasons:
Firstly, it primarily leverages the tool-calling and information extraction capabilities of LLMs, largely
underutilizing its potential as an intrinsic knowledge base (i.e., LLM-as-KB) [ 11,46]. This leads
to substantial retrieval redundancy, as external searches are performed even when the necessary
information might already be implicitly encoded within the parameters of the model. Secondly,
the limited capabilities of the retriever can introduce noise into the retrieval results [ 6], potentially
Preprint. Under review.arXiv:2505.07596v1  [cs.CL]  12 May 2025

generating unnecessary knowledge conflicts [ 7]. A common issue is erroneous retrieved knowledge
overriding the accurate parametric knowledge [ 41]. Furthermore, the iterative nature of search engine
calls frequently interrupts the generation process of LLM, resulting in significant inference delays
[43]. Thus, a critical research question emerges: How can we train an efficient adaptive search agent
that comprehensively integrates both parametric (internal) and retrieved (external) knowledge?
This paper argues that such an agent needs to posses the following three key knowledge behaviors:
(1) Self-knowledge Boundary Division (determine know/unknown): the ability to decompose a query
into atomic queries and determine whether each sub-query falls within the knowledge boundary of
the agent [ 20,27,39]; (2) Internal Knowledge Recall (search in parameter): the ability to generate
relevant background knowledge to assist in answering questions that fall within its knowledge
boundary [ 4,23]; (3) External Knowledge Recall (search in corpus): the ability to generate effective
search queries for questions outside its knowledge boundary and utilize search engines to acquire
the desired knowledge [ 45]. In all, an efficient adaptive search agent needs to accurately determine
whether to search in parameter or corpus, and should minimize the use of external knowledge by
leveraging its internal knowledge as much as possible. Therefore, the retrieval timing becomes the
core. Existing research determines retrieval timing either via external indicators/classifiers, which
often generalize poorly and require external tools [ 15,14], or through complex data engineering for
imitation/preference learning to enable autonomous decision-making [ 43,10,36]. However, how to
imbue a model with the capacity to determine the optimal retrieval timing for adaptive retrieval via
RL has not been fully investigated.
To address these issues and enable the model to exhibit the aforementioned knowledge behaviors,
we propose the Reinforced Internal-External Knowledge Synergistic R Easoning Agent ( IKEA ), an
efficient adaptive search agent powered by RL [ 29,31]. First, we design the IKEA agent framework,
which explicitly prompts the model to determine its internal knowledge boundary and prioritize the
utilization of knowledge within its parameters. The search engine is invoked to retrieve external
knowledge only when internal knowledge is deemed uncertain or insufficient. Next, we introduce
two key components: a knowledge-boundary aware reward function and a corresponding knowledge-
boundary aware training dataset for internal-external knowledge synergy oriented RL. The reward
function incentivizes correct answers while minimizing unnecessary external knowledge retrieval
for questions where the LLM possesses sufficient internal knowledge, and conversely, encouraging
retrieval for questions beyond its internal knowledge boundary. This approach aims to improve
the perception of its self-knowledge. The training dataset, meticulously constructed, comprises an
equal mix of questions that the model is likely to answer using its internal knowledge and those
requiring external knowledge. This balanced dataset is crucial for training the model to adaptively
and synergistically leverage both internal and external knowledge.
We conducted evaluations on multiple datasets involving both single-hop [ 19,22] and multi-hop
[42,12] knowledge reasoning tasks. IKEA outperforms baseline methods across various datasets,
achieving exceptional performance, and demonstrates strong generalization capabilities on out-
of-distribution datasets. Compared to naive reinforcement learning approaches (i.e. Search-R1)
[16,33,3], it can significantly reduce the number of retrievals while improving performance. This
fully showcases the effectiveness and efficiency of our proposed method. The contribution of this
paper are as follows:
•This paper addresses the limitations of current search agents, which often over-rely on external
searches and underutilize their intrinsic knowledge, leading to retrieval redundancy.
•This paper proposes Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA),
an efficient adaptive search agent via reinforcement learning, which could delineate the self-
knowledge boundary and prioritize parametric knowledge before resorting to external retrieval.
•This paper provides detailed analysis to explain that knowledge-boundary aware reward design and
data construction are both key to training efficient adaptive search agents.
2 Preliminary
2.1 Multi-turn Reinforcement Learning with Verifiable Reward for Large Language Model
This work considers an LLM agent πoperating within an environment Eto complete a task t. The
interaction proceeds in Nrounds: in each round k, the agent takes action ak, receives observa-
2

tionok+1. Both actions and observations are token sequences: ak= (ak,1, . . . , a k,ℓ), ok+1=
(ok+1,1, . . . , o k+1,ℓ). The state skat round kis the concatenation of all preceding tokens
(t, a0, o1, . . . , a k−1, ok). Upon task completion after Nrounds, a final reward ris provided by
the reward model. A trajectory is like: τ= (t, a0, o1, . . . , a N−1, oN, r). Reinforcement Learning
trains the policy π(a|s)on collected trajectories to maximize the expected cumulative reward, aiming
for an optimal policy that maximizes the total reward.
Proximal Policy Optimization (PPO) [ 29] is a common RL baseline, optimizing the policy via the
clipped surrogate loss:
LPPO(θ) =−ˆEt∼T1PN−1
k=0|ak|N−1X
k=0|ak|X
ℓ=1h
min
rθˆAτ,clip(rθ,1−ϵ,1 +ϵ)ˆAτi
, rθ=πθ(ak,l|sk)
πθold(ak,l|sk)(1)
ˆAτis the advantage, and ϵis the clipping ratio. It is noteworthy that we only compute the loss on
the action tokens. We mask the loss from the observation tokens because they are from the external
environment and are not generated by the LLM. However, PPO requires training a separate value
model (typically similar in size to the policy model) to estimate the value function Vand compute
the advantage ˆA(often using GAE [28]), resulting in significant additional memory overhead.
Therefore, this paper adopts Group Relative Policy Optimization (GRPO) [ 31] as the default RL
algorithm. As shown in the top of the Figure 1, GRPO performs multiple rollouts per task and
calculate the relative reward within the group as the advantage. This method avoids the need for a
separate value model and has shown performance comparable to or exceeding PPO. Its loss function
is:
LGRPO(θ) =−ˆEt∼T,τi∼πold(τ|t)1
GGX
i=11PN−1
k=0|ai,k|
N−1X
k=0|ai,k|X
ℓ=1h
min
rθˆAτi,clip(rθ,1−ϵ,1 +ϵ)ˆAτi
−βDKL[πθ||πold]i
, rθ=πθ(ai,k,l|si,k)
πθold(ai,k,l|si,k)(2)
where ˆAτi=ri−µr
σris the estimated advantage of trajectory τibased on group-relative rewards. µris
the mean and the σris the standard deviation of the rewards within the group.
2.2 Knowledge Boundary of Large Language Model
We use the term Knowledge Boundary to distinguish between the internal and external knowledge of
a specific LLM. Internal knowledge refers to the knowledge that can be extracted from the model
through some knowledge probing method, while external knowledge refers to the relative complement
of the internal knowledge in the whole world knowledge; that is, the knowledge that does not exist
within the parameters of the model. We also use knowledge boundary to differentiate between various
questions. Whether a question falls inside or outside the knowledge boundary refers to whether the
knowledge required to answer that question is internal knowledge or external knowledge.
3 Method
3.1 Basic Setting
In the context of knowledge-intensive reasoning, each task is framed as a query related to world
knowledge. The environment (search engine) comprises a text corpus and a retriever. The agent
interacts with the environment by generating a sequence of action tokens and receiving a sequence of
observation tokens. As illustrated in the middle of the Figure 1, a typical LLM-based search agent
[16,33,3] will generate reasoning thought, search query, and final answer in its action tokens. We
detail the agentic workflow and training method of this line of work as follows:
To facilitate the parsing of the executable actions for interaction with the environment, we de-
fine three sets of special tags to structure the action token sequence: <THINK >[REASONING
CONTENT ]</THINK >,<SEARCH >[SEARCH QUERY ]</SEARCH >, and <ANSWER >[FINAL AN -
SWER ]</ANSWER >. It is noteworthy that although the content within the <THINK >tags does
3

LLM AgentTask…𝜏!…Obs1Act	1Act	n𝜏"#!…Obs1Act	1Act	n𝜏"…Obs1Act	1Act	nRolloutRef ModelReward Model𝑟!𝑟"𝑟"#!…𝐴!𝐴"𝐴"#!…Group Normalization𝐴$=𝑟$−𝜇%𝜎%EnvInteractSearch EngineSearch R1TaskRolloutReward ModelQuery
🔍
📄Doc…𝜏!…Doc	1Think	+	SC	1Answer𝜏"#!…Doc	1Think	+	SC	1Answer𝜏"…Doc	1Think	+	SC	1Answer……𝜏!Think	+SP	+	Answer𝜏"#!Doc	Think	+	SC	Answer𝜏"DocThink	+	SP	+	SCThink	+	SP	+	Answer…Search EngineIKEATaskRolloutQuery
🔍
📄DocKL
𝑟=𝑟!"#=𝑓$%(𝐴𝑛𝑠,𝐺𝑜𝑙𝑑𝐴𝑛𝑠)Reward Model𝑟=𝑟!"#+𝑟&'𝑟&'=𝑓()𝑅𝑇𝑟!"#=0,SCSP
👍
👎𝑟!"#=1,SCSP
👍
👎knowledge-boundaryawarerewardLLMKnowledgeProbeeasyhardkb-awaredatasetknowledge-boundaryawaretrainingdatasetactiontokens, calculatelossobservationtokens, masklossSC (Search in Corpus): generate a search query to retrieve docs from external corpusSP (Search in Parameter): recall knowledge from internal parameterenvironmentfrozen modeltrainable modelFigure 1: The top of the figure illustrates the training process for Multi-turn Reinforcement Learning
with Verifiable Reward for LLM-Agent. In the middle is Search-R1, and at the very bottom is IKEA.
Search-R1 and IKEA are special types of LLM-agents. We highlight the differences from the training
of general LLM-agents, and to save space, we have omitted the common parts, such as the calculation
of KL and Advantage.
not directly interact with the environment, it is generated by the model and is considered part of
the action token sequence. In each turn, the agent must first generate reasoning content within
the<THINK >tag to analyze the current state and subsequently generate either a <SEARCH >or an
<ANSWER >tag to interact. When a <SEARCH >tag is generated, the model produces a search query
within it, which is then used by the retriever to get relevant knowledge from the corpus. We define a
special set of tags: <CONTEXT >[RETRIEVED CONTEXT ]</CONTEXT >, and the retrieved relevant
documents are placed between these tags and inserted as an observation after the generated token
sequence, allowing the interaction to continue to the next turn. The content within the <CONTEXT >
tags is not generated by the model and is therefore masked out during loss calculation. When an
<ANSWER >tag is generated, the model outputs the answer of the task query, at which point the entire
task execution process completes. We refer to such a complete process as a rollout. Upon obtaining
the final answer, we evaluate the reward for the current trajectory using an Exact Match Reward
Function. Then we can repeat the rollout processing many times to get a group of trajectories and
leverage the loss function detailed in Section 2 to optimize the agent.
3.2 IEKA: Reinforced Internal-External Knowledge Synergistic Reasoning Agent
Existing search agents often primarily leverage the planning capability of the LLM, decomposing
a task query into multiple subqueries and iteratively retrieving relevant evidence documents for
each subquery to aid reasoning. Such search agents fail to fully utilize the capacity of the LLM
as a parametric knowledge base, resulting in much redundant retrieval. This not only introduces
significant inference latency but also potentially leads to harmful knowledge conflicts [ 41,18] (wrong
external knowledge overrides the correct internal knowledge). Building upon this, this paper argues
that an efficient adaptive search agent is needed to address these problems. It should possess the
ability to delineate its own knowledge boundary and leverage internal parametric knowledge as much
as possible within this boundary, while employing retrieval for knowledge outside this boundary.
To this end, we propose the Reinforced Internal-External Knowledge Synergistic R Easoning Agent
(IKEA ). As shown in the bottom of Figure 1, this paper first designs the agent prompt template to
enable the model to autonomously perform synergistic reasoning using both internal and external
knowledge. Subsequently, we design a knowledge-boundary aware reward function and construct
4

a knowledge-boundary aware training dataset, which will encourage the model to clarify its own
knowledge boundary while utilizing different knowledge behaviors both inside and outside this
boundary. Finally, we apply reinforcement learning to finetune the agent towards internal-external
knowledge synergy.
IKEA Agent Prompt Template To begin, we incorporate output format constraints into the
prompt to ensure the agent interacts with the environment in the format described in the Section 3.1.
Furthermore, we prompt the model to evaluate all subqueries, encouraging it to utilize its internal
parametric knowledge whenever it is confident. When the model is uncertain about the knowledge
regarding specific information, it is encouraged to retrieve relevant knowledge from the external
knowledge base. The detailed prompt template is provided in the Appendix A.
Knowledge-boundary Aware Reward Design Due to the probabilistic nature of LLMs, existing
LLMs have a blurred perception of their self-knowledge boundaries. They cannot definitively
distinguish which questions pertain to internal knowledge and which require external knowledge.
As shown in the bottom of the Figure 1, for the same task, τ1only uses internal knowledge, τG−1
only use external knowledge, and τGuses both internal and external knowledge. Consequently,
prompt-based agents may exhibit knowledge misidentification behaviors, leading to the generation of
hallucinated answers for questions outside their knowledge boundaries, while utilizing redundant
retrieval to for questions within their knowledge boundaries.
To address this, we design a knowledge-boundary aware reward composed of several components.
First, the answer reward ( rans) is 1 if the final answer matches the gold answer, and 0 otherwise.
Second, the knowledge boundary reward ( rkb) is determined as follows: if rans= 1,rkbis a linear
function increasing as the retrieval times ( RT) decrease, ranging from 0 to rkb+. Ifrans= 0, then
rkb= 0 when the number of retrievals is 0, and rkb=rkb−(a small value) when the number of
retrievals is greater than 0. Finally, for the format reward, if the generated trajectory violates format
constraints of IKEA, the total reward is -1; otherwise, it is rans+rkb.
The expression for the reward function is as follows:
R=−1 if trajectory format is incorrect
rans+rkbif trajectory format is correct(3)
rans=1ans == gold ans
0ans != gold ans, rkb=

rkb+×
1−RT
RTmax
ifrans= 1
0 ifrans= 0andRT= 0
rkb− ifrans= 0andRT > 0(4)
Here, RTmax denotes the maximum number of retrievals, rkb−is a small value, rkb+is the maximum
possible knowledge boundary reward. During exploration, when the agent obtains the correct answer
(rans= 1), it may utilize internal or external knowledge. The reward rkb+is designed to incentivize
the agent to minimize retrieval attempts, thereby favoring the use of internal knowledge. Conversely,
when the agent fails to obtain the correct answer ( rans= 0), indicating high uncertainty regarding
relevant knowledge, the reward rkb−encourages reliance on external knowledge. To prevent the
development of excessive retrieval behavior, we establish rkb−≪rkb+.
Knowledge-boundary Aware Dataset Construction We use In-context Learning with three Chain-
of-Thought exemplars to probe the internal knowledge of the model. For each question, we sample
the answer Ntimes. A question is labeled Qeasy if the correct answer is obtained at least once,
indicating the model likely possesses the relevant knowledge. Otherwise, it’s labeled Qhard.
If the training dataset exclusively contained data from Qeasy, the model would be more likely to
utilize internal knowledge during rollout, and relying solely on internal knowledge would yield higher
rewards than using retrieval. Consequently, after full training, the model would tend to avoid retrieval
for any question. Conversely, if the training dataset only comprised Qhard questions, the model
would be more inclined to use external retrieved knowledge during the rollout, and using the retriever
would result in higher rewards than not using it. Thus, after full training, the model would tend to use
retrieval exclusively for all questions.
5

Table 1: Performance of Qwen2.5-3B and Qwen2.5-7B (Base & Instruct). "-Zero" are trained from
Base. EM = exact match, RT = number of valid searches. The original checkpoint of Search-R1-
Zero-3B might be over-optimized (hard to count RT).†DeepRAG EM/RT results are from its paper.
MethodNQ PopQA HotpotQA 2WikiAvg
Easy Hard Easy Hard Easy Hard Easy Hard
EM RT EM RT EM RT EM RT EM RT EM RT EM RT EM RT EM RT
Qwen2.5-3B
w/o parameter update (re-implementation)
Direct 36.33 0 3.91 0 56.05 0 2.54 0 50.39 0 1.56 0 50.98 0 11.72 0 26.69 0.00
RAG 59.77 1 30.47 1 68.16 1 31.64 1 54.30 1 13.87 1 40.04 1 12.70 1 38.87 1.00
Iter-Retgen 59.57 4 30.27 4 68.55 4 32.81 4 55.86 4 16.02 4 41.60 4 15.23 4 39.99 4.00
IR-COT 35.74 3.26 15.04 3.34 48.05 3.15 24.80 3.17 43.36 3.64 9.77 3.60 25.39 3.82 9.18 3.70 26.42 3.46
FLARE 34.57 0.21 4.49 0.41 52.15 0.18 4.49 0.52 48.44 0.16 1.76 0.61 50.00 0.03 11.13 0.32 25.88 0.31
Reinforcement learning (re-implementation for search-r1)
R1-Zero 62.34 0 10.55 0 72.66 0 3.71 0 59.57 0 4.49 0 57.22 0 13.28 0 35.48 0.00
R1 59.77 0 7.23 0 70.11 0 3.13 0 58.01 0 3.71 0 57.81 0 13.67 0 34.18 0.00
Search-R1-Zero***66.60 - 28.51 - 77.73 - 27.93 - 64.45 - 13.67 - 52.54 - 13.48 - 43.11 -
Search-R1 66.41 1.17 32.61 1.30 73.43 1.22 29.49 1.53 65.23 1.86 22.27 1.88 51.17 2.16 26.56 2.00 45.90 1.64
IKEA-Zero 71.29 1.00 34.18 1.00 78.90 1.00 35.94 1.02 68.94 1.05 21.09 1.14 54.69 1.19 23.63 1.39 48.58 (+5.47) 1.10
IKEA 72.46 1.00 31.44 1.02 79.69 1.00 33.59 1.02 69.92 1.04 20.11 1.13 59.37 1.15 20.70 1.21 48.41 (+2.51) 1.07 (-34.76%)
Qwen2.5-7B
w/o parameter update (re-implementation)
Direct 41.41 0 4.30 0 61.13 0 2.34 0 54.69 0 4.10 0 51.95 0 10.94 0 28.86 0.00
RAG 57.23 1 26.37 1 69.73 1 31.64 1 58.98 1 17.77 1 40.82 1 11.33 1 39.23 1.00
Iter-Retgen 58.79 4 26.95 4 70.90 4 31.25 4 61.52 4 19.73 4 43.36 4 14.45 4 40.87 4.00
IR-COT 40.04 2.59 14.26 2.68 58.98 2.48 25.78 2.56 46.09 3.09 14.26 2.94 17.58 3.18 12.50 3.07 28.69 2.82
FLARE 39.65 0.16 5.27 0.28 59.96 0.11 3.13 0.67 52.93 0.08 4.10 0.375 51.17 0.03 11.52 0.35 28.47 0.26
SFT/DPO (results from the original paper, shown as EM/RT)
DeepRAG†- 40.60/ 32.10/ 40.40/ - -
Reinforcement learning
R1-Zero 66.80 0 15.23 0 72.65 0 6.25 0 64.65 0 5.66 0 53.32 0 18.16 0 37.84 0.00
R1 62.50 0 14.06 0 73.04 0 5.27 0 64.06 0 5.47 0 57.23 0 14.45 0 37.01 0.00
Search-R1-Zero 68.55 1.19 35.55 1.34 76.37 1.16 33.59 1.30 69.73 1.78 25.78 1.77 46.68 2.38 26.56 2.13 47.85 1.63
Search-R1 65.63 1.34 33.40 1.51 78.13 1.24 32.62 1.51 68.17 2.00 24.02 2.07 35.35 2.67 22.66 2.47 45.00 1.85
IKEA-Zero 74.80 1.00 37.89 1.00 80.47 1.00 33.20 1.00 74.22 1.01 23.43 1.08 57.42 1.03 27.34 1.23 51.10 (+3.25) 1.04 (-36.20%)
IKEA 74.61 0.59 32.23 0.89 80.08 0.56 31.84 1.09 71.88 0.60 26.56 1.20 54.49 0.93 28.71 1.38 50.05 (+5.05) 0.91 (-50.81%)
204060801001200.80.911.11.21.3Model Configurationikea-zero-qwen-3bikea-qwen-3bikea-zero-qwen-7bikea-qwen-7b
StepNumber of Valid Searches
20406080100120500600700800900100011001200Model Configurationikea-zero-qwen-3bikea-qwen-3bikea-zero-qwen-7bikea-qwen-7b
StepResponse Length
20406080100120−0.8−0.6−0.4−0.200.20.40.60.8Model Configurationikea-zero-qwen-3bikea-qwen-3bikea-zero-qwen-7bikea-qwen-7b
StepTraining Rewards(a) Num of Valid Search(b) Response Length(c) Training Rewards
Figure 2: The training log of IKEA-3B-Zero, IKEA-3B, IKEA-7B-Zero and IKEA-7B. We show the
curve of number of valid searches, response length and trainign rewards.
To achieve a balanced use of internal and external knowledge, we construct the training dataset with
a 1:1 ratio of Qeasy andQhard questions. This promotes adaptive retrieval and synergy between
internal and external knowledge.
4 Experiment
4.1 Setting
Test sets (easy and hard subsets) were constructed like the training set (Section 3.2), including
two in-distribution and two out-of-distribution sets (details in Appendix B). We benchmarked our
method against baselines (Appendix C) using various model sizes and types, with training specifics
in Appendix D. Performance was evaluated by exact match (EM) and efficiency by the number of
valid searches (RT).
4.2 Overall Results
Experimental results are presented in Table 1, with corresponding training logs illustrated in Figure
2. A detailed analysis was conducted to demonstrate the advantages of the proposed method and
provide insights for future research. It is posited that questions in the Easy subset primarily require
knowledge within the knowledge boundary, whereas those in the Hard subset likely necessitate
knowledge beyond it.
6

Baselines without parameter updates struggle to effectively synergize internal and external
knowledge. "Direct" relies on internal knowledge, while "RAG" and "Iter-Retgen" (which performs
iterative retrieval) use external knowledge. External knowledge significantly improves LLM perfor-
mance on knowledge-intensive tasks, especially Hard subsets, indicating LLMs’ internal knowledge
deficiencies. However, constant retrieval causes conflicts and latency. Adaptive RAG methods,
like IR-COT (LLM autonomously decides retrieval timing) and FLARE (retrieves based on low-
probability tokens), aim to mitigate these issues. IR-COT improves performance on "Hard" tasks but
degrades "Easy" ones due to knowledge conflicts. FLARE performs few retrievals, yielding perfor-
mance similar to "Direct," suggesting token probability is not an effective retrieval trigger. The core
finding is that internal and external knowledge must be synergistically used: internal when sufficient,
external when insufficient. However, un-finetuned models struggle to autonomously determine when
to leverage external knowledge.
Reinforcement learning baselines can effectively activate the model’s capacity to solely utilize
its internal knowledge or solely utilize external knowledge accessed via retrieval. R1, which
relies on internal knowledge only based on reasoning, significantly improves performance on Easy
subsets by reinforcing knowledge expressions through RL. However, its gains on Hard subsets are
limited, highlighting the necessity of external knowledge retrieval. Search-R1, by generating search
queries for external knowledge retrieval, addresses the internal knowledge deficit. It outperforms
other methods (e.g., Iter-Retgen) with fewer retrievals, demonstrating that RL improves its planning
and tool-using abilities for external knowledge access. While both R1 and Search-R1 show RL
can boost the utilization of internal and external knowledge separately, neither method effectively
integrates these two knowledge sources synergistically.
IKEA can adaptively combine internal and external knowledge for synergistic knowledge
reasoning. During multiple rollouts, the model can choose to utilize only internal knowledge,
only external knowledge, or a combination of both. Through a knowledge-boundary aware reward,
RL encourages the model to leverage internal knowledge as much as possible when both internal
and external knowledge are effective to reduce calls to retrieval tools, and to utilize retrieval to
acquire external knowledge when internal knowledge is insufficient. As shown in the table, compared
to R1, IKEA improves performance by over 10%, with the improvement primarily coming from
difficult subsets. This indicates that it can fully utilize external knowledge based on its internal
knowledge. Compared to Search-R1, IKEA significantly reduces the number of retrievals while
improving performance. This suggests that in the process of self-exploration, it learns to delineate
its own knowledge boundaries, leveraging parametric knowledge as much as possible within these
boundaries and retrieval knowledge outside of them. This not only effectively overcomes potential
knowledge conflicts but also improves the overall process efficiency. It is worth noting that it also
performs well on two out-of-distribution datasets, indicating that the knowledge-seeking behavior
acquired through self-exploration can generalize effectively.
The IKEA training method is effective across models of different sizes and types. Figure 2
illustrates the IKEA training process based on different initial models. IKEA models, initialized
from instruction-tuned models, start with higher rewards due to better instruction-following. IKEA-
Zero models, starting from base models, begin with lower rewards but gradually learn the desired
format. Both converge to similar reward levels by the end of training process, demonstrating that
reinforcement learning can teach collaborative reasoning without cold start. Larger models (e.g.,
7B vs. 3B) achieve higher initial and final rewards and converge faster. Retrieval counts initially
increase before decreasing, indicating early benefit from more retrieval, followed by refinement to
eliminate retrieval redundancy. Response length trends similarly for IKEA models (initial increase
then decrease), while IKEA-Zero models show a consistent decrease, signifying the reduction of
meaningless redundancy in the inital stage as they learn a fixed format.
5 Ablation Study
We conducted ablation studies based on Qwen2.5-3B-Instruct, which fully validated the effectiveness
of the proposed method.
7

20 40 60 80 100 120500600700800900Reward Mechanism
w/o r_kb
w/o r_kb-
original reward
StepResponse Length
20 40 60 80 100 1200.20.30.40.50.60.70.8Reward Mechanism
w/o r_kb
w/o r_kb-
original reward
StepTraining Rewards
20 40 60 80 100 1200.811.21.41.6Reward Mechanism
w/o r_kb
w/o r_kb-
original reward
StepNumber of Valid Searches
(a) Num of Valid Search (b) Response Length (c) Training RewardsFigure 3: The training logs of different reward design. We show the curve of number of valid searches,
response length and trainign rewards.
Table 2: The ablation results of reward design.
MethodNQ PopQA HotpotQA 2WikiAvg
Easy Hard Easy Hard Easy Hard Easy Hard
IKEA (EM) 72.46 31.44 79.69 33.59 69.92 20.11 59.37 20.70 48.41
RT 1.00 1.02 1.00 1.02 1.04 1.13 1.15 1.21 1.07
IKEA w/o rkb−(EM) 66.01 28.91 74.61 32.42 66.99 20.90 55.27 0.21 43.17
RT 0.48 0.68 0.53 1.00 0.58 1.08 0.64 1.11 0.89
IKEA w/o rkb(EM) 71.09 34.57 76.37 32.23 70.12 25.59 53.32 25.20 48.56
RT 1.40 1.54 1.35 1.63 1.94 2.12 2.40 2.48 1.86
5.1 The effects of reward design
We present the training process using different rewards in Figure 3 and the final test results in Table 2.
Without the knowledge boundary aware reward (" w/o r kb"), both effective retrievals and response
length show a consistent upward trend, significantly surpassing models with the original reward.
This is because early in training, retrieval is more frequently rewarded than relying on parametric
knowledge, leading to gradient updates that suppress the latter. Consequently, the model develops a
bias for "retrieval > no retrieval", eventually maximizing reliance on retrieved knowledge, akin to the
Search-R1 strategy. For the " w/o r kb-" case (excluding the negative component of the knowledge
boundary aware reward), retrieval count and response length are significantly less than the original
reward. Because the positive reward component ( rkb+) encourages greater reliance on internal
knowledge. This leads to incorrect generalization, where the model increasingly defaults to the R1
strategy even for questions requiring external knowledge. Final results show that IKEA " w/o r kb"
achieves a similar EM score but with significantly more retrievals. Conversely, IKEA " w/o r kb−"
exhibits considerably degraded performance alongside a substantial decrease in retrievals. Therefore,
we conclude that an effective knowledge boundary aware reward function must appropriately balance
internal and external knowledge utilization to achieve their synergistic application.
5.2 The effects of dataset difficulty
204060801001200.40.60.811.21.4Difficulty Settinghardeasymixed
StepNumber of Valid Searches
20406080100120300400500600700800900Difficulty of Datasethardeasymixed
StepResponse Length(a) Num of Valid Search(b) Response Length(c) Training Rewards
204060801001200.20.40.60.81Difficulty of Datasethardeasymixed
StepTraining Rewards
Figure 4: The training logs of different the difficulty of training datasets. We show the curve of
number of valid searches, response length and trainign rewards.
We illustrate the training processes using datasets of varying difficulty in Figure 4 and present the
final test results in Table 3. Training on datasets of varying difficulty (easy, mixed, hard) revealed
a consistent trend during training: Hard > Mixed > Easy for both effective number of searches
8

Table 3: The ablation results of the difficulty of the training datasets.
MethodNQ PopQA HotpotQA 2WikiAvg
Easy Hard Easy Hard Easy Hard Easy Hard
IKEA (EM) 72.46 31.44 79.69 33.59 69.92 20.11 59.37 20.70 48.41
RT 1.00 1.02 1.00 1.02 1.04 1.13 1.15 1.21 1.07
IKEA w/ easy (EM) 66.99 21.88 76.17 25.59 66.70 15.43 56.45 16.80 43.25
RT 0.28 0.54 0.29 0.80 0.34 0.84 0.16 0.70 0.49
IKEA w/ hard (EM) 66.02 33.98 75.39 0.35 64.65 25.00 46.09 23.63 41.89
RT 1.03 1.07 1.05 1.11 1.46 1.59 2.08 2.10 1.44
and response length. This is because the model uses parametric knowledge for problems within
its knowledge boundary and retrieval knowledge for those beyond it. Training on the Easy dataset
showed a continuous decrease in retrieval attempts and response length, indicating that models
converge to behaviors characteristic of the training data’s difficulty. On the test set, both easy and
hard variants of the IKEA model showed substantially lower Exact Match (EM) scores compared to
the original. Retrieval attempts dropped significantly for the easy variant and increased substantially
for the hard variant. This highlights that disproportionately favoring one type of knowledge hinders
full performance, underscoring the importance of synergistically using both internal (parametric) and
external (retrieval-based) knowledge for effective reasoning.
6 Related Work
RL for LLM-based Agent Reinforcement Learning (RL) [ 38] is a crucial technique for post-
training Large Language Models (LLMs), enabling the alignment of pre-trained models’ values [ 24]
and enhancing their capabilities in specific downstream tasks [ 9]. The community has developed
various distinctive RL algorithms, such as PPO [ 29], DPO [ 26], RLOO [ 1], ReMax [ 21], and GRPO
[31]. Building upon this, by constructing different environments and reward functions, LLMs can
be trained into intelligent agents capable of autonomous decision-making and interaction with the
environment. A typical application in this area is the Search Agent [ 16,3,33], which interacts with
search engines to continuously acquire knowledge from the environment and perform reasoning,
ultimately completing knowledge-intensive tasks.
The Knowledge Boundary of LLM Large Language Models (LLMs) possess parametric (internal)
knowledge [ 46] and can access external knowledge. The concept of Knowledge Boundary [20,40]
distinguishes between these. This boundary is probed using template-based methods (evaluating
responses to specific prompts [ 25]) or internal state-based methods (classifying based on model
features like hidden states [ 2] or SAEs [ 44]). Understanding this boundary is crucial for Retrieval
Augmented Generation (RAG) models [ 27] to adapt their behavior to different questions and avoid
hallucinations.
7 Conclusion and Limitations
This paper introduced the Reinforced Internal-External Knowledge Synergistic Reasoning Agent
(IKEA), an innovative approach to developing efficient and adaptive search agents. IKEA addresses
critical limitations in existing RL-based search agents, namely the underutilization of internal knowl-
edge, which can lead to redundant retrievals, potential knowledge conflicts, and increased inference
latency. The core of IKEA lies in its ability to discern its own knowledge boundary, prioritizing
the use of its internal parametric knowledge and resorting to external search only when the internal
knowledge is deemed insufficient or uncertain. This is achieved through a novel knowledge-boundary
aware reward function and a meticulously constructed knowledge-boundary aware training dataset.
This approach significantly enhances reasoning efficiency and accuracy on knowledge-intensive tasks.
Despite these achievements, IKEA’s reliance on specific dataset construction and model probing for
knowledge boundary awareness may limit its universal applicability, the reward function parameters
might require grid searching, and the RL training process is computationally expensive. Future work
9

could explore more dynamic knowledge boundary learning methods, investigate applicability across
a broader range of tasks, and aim to reduce training resource requirements.
References
[1]Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier
Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization
for learning from human feedback in llms, 2024.
[2]Lida Chen, Zujie Liang, Xintao Wang, Jiaqing Liang, Yanghua Xiao, Feng Wei, Jinglei Chen,
Zhenghong Hao, Bing Han, and Wei Wang. Teaching large language models to express
knowledge boundary from their own signals, 2024.
[3]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z.
Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. Research: Learning
to reason with search for llms via reinforcement learning, 2025.
[4]Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang, and William Yang Wang. Understand-
ing the interplay between parametric and contextual knowledge for large language models,
2024.
[5]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli
Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian
Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan
Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian,
Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong
Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan
Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting
Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun,
T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu,
Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao
Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su,
Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang
Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y . K. Li, Y . Q. Wang, Y . X.
Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao
Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang
Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He,
Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y . X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan
Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu,
Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang,
and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning, 2025.
[6]Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and Zhicheng Dou.
Understand what llm needs: Dual preference alignment for retrieval-augmented generation. In
Proceedings of the ACM on Web Conference 2025 , WWW ’25, page 4206–4225, New York,
NY , USA, 2025. Association for Computing Machinery.
[7]Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. Enhancing
noise robustness of retrieval-augmented language models with adaptive adversarial training.
In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages
10028–10039, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
10

[8]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A
survey, 2024.
[9]Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, and Christopher D. Manning. Synthetic
data generation & multi-step rl for reasoning & tool use, 2025.
[10] Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han,
Le Sun, and Jie Zhou. Deeprag: Thinking to retrieval step by step for large language models,
2025.
[11] Benjamin Heinzerling and Kentaro Inui. Language models as knowledge bases: On entity
representations, storage capacity, and paraphrased queries. In Paola Merlo, Jorg Tiedemann,
and Reut Tsarfaty, editors, Proceedings of the 16th Conference of the European Chapter of
the Association for Computational Linguistics: Main Volume , pages 1772–1791, Online, April
2021. Association for Computational Linguistics.
[12] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing
a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott,
Nuria Bel, and Chengqing Zong, editors, Proceedings of the 28th International Conference
on Computational Linguistics , pages 6609–6625, Barcelona, Spain (Online), December 2020.
International Committee on Computational Linguistics.
[13] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination
in large language models: Principles, taxonomy, challenges, and open questions. ACM Transac-
tions on Information Systems , 43(2):1–55, January 2025.
[14] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity,
2024.
[15] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation, 2023.
[16] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Za-
mani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning, 2025.
[17] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. Flashrag: A modular
toolkit for efficient retrieval-augmented generation research. CoRR , abs/2405.13576, 2024.
[18] Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang,
Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting
and mitigating knowledge conflicts in language models. In Lun-Wei Ku, Andre Martins, and
Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024 ,
pages 1193–1215, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research. Transactions of the
Association for Computational Linguistics , 7:452–466, 2019.
[20] Moxin Li, Yong Zhao, Yang Deng, Wenxuan Zhang, Shuaiyi Li, Wenya Xie, See-Kiong Ng,
and Tat-Seng Chua. Knowledge boundary of large language models: A survey, 2024.
[21] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo.
Remax: A simple, effective, and efficient reinforcement learning method for aligning large
language models, 2024.
[22] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. When not to trust language models: Investigating effectiveness of parametric and
non-parametric memories. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors,
Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 9802–9822, Toronto, Canada, July 2023. Association for
Computational Linguistics.
11

[23] Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu
Chen. Generation-augmented retrieval for open-domain question answering. In Chengqing
Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting
of the Association for Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers) , pages 4089–4100, Online, August
2021. Association for Computational Linguistics.
[24] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton,
Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano,
Jan Leike, and Ryan Lowe. Training language models to follow instructions with human
feedback, 2022.
[25] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang
Wu, and Alexander Miller. Language models as knowledge bases? In Kentaro Inui, Jing
Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP) , pages 2463–2473, Hong Kong, China, November
2019. Association for Computational Linguistics.
[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model.
InThirty-seventh Conference on Neural Information Processing Systems , 2023.
[27] Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong
Wen, and Haifeng Wang. Investigating the factual knowledge boundary of large language
models with retrieval augmentation, 2024.
[28] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
dimensional continuous control using generalized advantage estimation, 2018.
[29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms, 2017.
[30] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen.
Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy.
In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for
Computational Linguistics: EMNLP 2023 , pages 9248–9274, Singapore, December 2023.
Association for Computational Linguistics.
[31] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y . K. Li, Y . Wu, and Daya Guo. Deepseekmath: Pushing the limits of
mathematical reasoning in open language models, 2024.
[32] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua
Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In
Proceedings of the Twentieth European Conference on Computer Systems , EuroSys ’25, page
1279–1297. ACM, March 2025.
[33] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning, 2025.
[34] Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu.
Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains, 2025.
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Anna
Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages
10014–10037, Toronto, Canada, July 2023. Association for Computational Linguistics.
[36] Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, and Furu Wei. Chain-
of-retrieval augmented generation, 2025.
[37] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan
Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.
arXiv preprint arXiv:2212.03533 , 2022.
12

[38] Shuhe Wang, Shengyu Zhang, Jie Zhang, Runyi Hu, Xiaoya Li, Tianwei Zhang, Jiwei Li, Fei
Wu, Guoyin Wang, and Eduard Hovy. Reinforcement learning enhanced llms: A survey, 2025.
[39] Zhihua Wen, Zhiliang Tian, Zexin Jian, Zhen Huang, Pei Ke, Yifu Gao, Minlie Huang, and
Dongsheng Li. Perception of knowledge boundary for large language models through semi-
open-ended question answering. In The Thirty-eighth Annual Conference on Neural Information
Processing Systems , 2024.
[40] Hongshen Xu, Zichen Zhu, Situo Zhang, Da Ma, Shuai Fan, Lu Chen, and Kai Yu. Rejection
improves reliability: Training LLMs to refuse unknown questions using RL from knowledge
feedback. In First Conference on Language Modeling , 2024.
[41] Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei
Xu. Knowledge conflicts for LLMs: A survey. In Yaser Al-Onaizan, Mohit Bansal, and
Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing , pages 8541–8565, Miami, Florida, USA, November 2024. Association
for Computational Linguistics.
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors,
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing ,
pages 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational
Linguistics.
[43] Tian Yu, Shaolei Zhang, and Yang Feng. Auto-rag: Autonomous retrieval-augmented generation
for large language models, 2024.
[44] Yu Zhao, Alessio Devoto, Giwon Hong, Xiaotang Du, Aryo Pradipta Gema, Hongru Wang,
Xuanli He, Kam-Fai Wong, and Pasquale Minervini. Steering knowledge selection behaviours
in LLMs via SAE-based representation engineering. In Luis Chiruzzo, Alan Ritter, and
Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter
of the Association for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers) , pages 5117–5136, Albuquerque, New Mexico, April 2025. Association for
Computational Linguistics.
[45] Yukun Zhao, Lingyong Yan, Weiwei Sun, Guoliang Xing, Chong Meng, Shuaiqiang Wang,
Zhicong Cheng, Zhaochun Ren, and Dawei Yin. Knowing what llms do not know: A simple
yet effective self-detection method, 2024.
[46] Danna Zheng, Mirella Lapata, and Jeff Z. Pan. How reliable are llms as knowledge bases?
re-thinking facutality and consistency, 2024.
13

A IKEA agent template
We use the system template in Table 4 to prompt the agent to interact with the environment:
B Dataset Construction
We use NQ [ 19] and HotpotQA [ 12] as the in-distribution datasets. We use the PopQA [ 22] and
2Wikimultihopqa [ 12] as the out-of-distribution datasets. Following the knowledge-boundary training
dataset construction method, we construct easy and hard subset for each dataset. We use the Qwen-
2.5-3B-Instruct as the sampling model. There are 512 examples in each subset of each dataset.
C Baselines
We compared methods that do not require training (e.g., zero-shot or few-shot prompting), those that
utilize Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), and reinforcement
learning-based approaches. The corresponding baselines are shown as follows:
•Direct We directly prompt the model to answer the relevant question using only its internal
knowledge.
•RAG We retrieve documents using the question and prompt the model to answer the relevant
question relying solely on the retrieved knowledge.
•Iter-Retgen [30] It is an iterative retrieval-generation method that achieves strong perfor-
mance by synergizing parametric and non-parametric knowledge. We set the default ret-gen
turns as 4.
•IR-COT [35] It is method for multi-step question answering, which interleaves retrieval
with steps in the chain-of-thought, using CoT to guide retrieval and retrieval results to
improve CoT. It will adatpively determine the turns of retrieval according to the knowledge
needs. And we set the max turns as 4.
•FLARE [15] This method introduces a forward-looking active retrieval-augmented genera-
tion (FLARE) approach that iteratively uses predictions of upcoming sentences to anticipate
future content and retrieves relevant documents when a sentence contains low-confidence
tokens, in order to regenerate that sentence. It uses a specific criteria to determine the
retrieval timing. We set the max number of search as 4.
•DeepRAG [10] This method introduces a framework that models retrieval-augmented
generation as a Markov Decision Process (MDP), enabling strategic and adaptive retrieval to
improve retrieval efficiency and answer accuracy. It collects offline trajectories to finetune
the base model with SFT and DPO.
•R1[5] It uses reinforcement learning to encourage the model to reason in order to activate
its internal knowledge. This method only uses the internal knowledge.
•Search-R1 [16,33,3] The model’s capacity to employ external retrieval tools is activated
via multi-turn reinforcement learning. This technique exclusively relies on the model’s
external knowledge. We set the max number of search as 4.
D Implementation Details
We use e5-base [ 37] as the retriever model and wikipedia2018 as the corpus for retrieval. We employ
Qwen2.5-3B(-Instruct) and Qwen2.5-7B(-Instruct) as the initial models. Models with the "-Zero"
suffix are trained from the Base model, while those without it are trained from the Instruct model.
we use FlashRAG [ 17] to reproduce the baseline results. We utilize the verl [ 32] framework for
training. GRPO [ 31] is used as the reinforcement learning algorithm. We use the NQ and HotpotQA
to construct training datasets. For each one, we sample 4000 easy samples and 4000 hard samples.
We set the number of rollouts as 16 for one task. We set the learning rate as 5e-7, warmup ratio as
75%, batch size as 256, training steps as 120. We set rkb+as 0.6 and rkb−as 0.05, RTmax as 3. We
use 8 A100 GPUs for all the experiments.
14

You are an expert assistant capable of solving knowledge-intensive tasks efficiently. You will be
given a question to answer as accurately as possible.
You can use your own knowledge or call external search engines to gather additional information,
but searching should only occur when necessary. Specifically, you should search only when
encountering a clear knowledge gap or uncertainty that prevents you from confidently answering
the question.
To arrive at the answer, you will proceed step-by-step in a structured cycle of ’<think>thinking
content</think>’, ’<search>search query</search>’ (optional), and ’<context>returned external
information</context>’ (optional) sequences. You can only generate content within these special
tags.
Remember that <search>xxx</search> and <context>xxx</context> are optional. You can skip
them if you have enough knowledge to answer the question. And skip is them is encouraged and
preferable.
Thinking Phase (<think>): Recall your own knowledge, analyze current information, and decide
whether further search is needed. If enough knowledge is available, skip searching. For question,
it may be decomposed into sub-questions for you to think about. Some sub-questions may be
answered by searching, while others may not. You can also use the <think> tag to express your
uncertainty about the sub-question.
Searching Phase (<search>): Formulate a search query only if required to fill a knowledge gap
or verify uncertainty. Skip if unnecessary. Information Phase (<context>): Use search results as
context for further steps. If no search was performed, proceed without this phase.
Answering Phase (<answer>): Provide a concise and accurate answer within <answer> tags once
you have enough knowledge. The answer should be short and precise, such as <answer> Beijing
</answer>.
Here are a few examples:
—
Example 1: search is needed, search more than once
Question: xxx
<think> xxx </think>
search> xxx </search>
<context> xxx </context>
<think> xxx </think>
(search more than once)
<think> xxx </think>
<answer> xx </answer>
Example 2: search is needed, only search once
Question: xxx?
<think> xxx </think>
<search> xxx </search>
<context> xxx </context>
<think> xxx </think>
<answer> xxx </answer>
—
Example 3: search is not needed
Question: xxx?
<think> xxx </think>
<answer> xxx </answer>
—
You can search 0 - N times. 0 is preferable. Each search should be focused on one sub-question.
The answer within <answer> tags should be short and precise, such as <answer> yes </answer>.
Now it is your turn to answer the question.
Question: {question}
Table 4: System prompt of IKEA.
15

NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and follow the (optional) supplemental material. The checklist does NOT count
towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
•[NA] means either that the question is Not Applicable for that particular paper or the
relevant information is Not Available.
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
•Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist" ,
•Keep the checklist subsection headings, questions/answers and guidelines below.
•Do not modify the questions and only use the provided macros for your answers .
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We clarify the contributions and the scope in abstract and the introduction.
Guidelines:
•The answer NA means that the abstract and introduction do not include the claims
made in the paper.
•The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
•The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
•It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We discuss the limitations of the work in Section 7.
16

Guidelines:
•The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
•The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
•The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
•The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
•The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
•If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
•While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [NA]
Justification: The paper does not include theoretical results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
•All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
•All assumptions should be clearly stated or referenced in the statement of any theorems.
•The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
•Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We provide the details in Section A, B and D.
Guidelines:
• The answer NA means that the paper does not include experiments.
17

•If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
•If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
•Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
•While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a)If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b)If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c)If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d)We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5.Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: We provide the data and code in supplementary materials.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
•Please see the NeurIPS code and data submission guidelines ( https://nips.cc/
public/guides/CodeSubmissionPolicy ) for more details.
•While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
•The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( https:
//nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
•The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
•The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
•At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
18

•Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6.Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes]
Justification: We provide the details in Section A, B and D.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
•The full details can be provided either with the code, in appendix, or as supplemental
material.
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: The computation resources are too expensive for our lab to repeat much more
times.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
•The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
•The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
•It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
•It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
•For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
•If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: We show it in Section D.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
19

•The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
•The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9.Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?
Answer: [Yes]
Justification: This paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
•The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
•If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
•The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: We think our work will not have a significant social impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
•If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
•Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
•The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
•The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
•If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
20

•Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
•Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
•We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: All assets used in this paper are properly credited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
•The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
•For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
•If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
•For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
•If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: We provide the documentation in the code repo.
Guidelines:
• The answer NA means that the paper does not release new assets.
•Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
•The paper should discuss whether and how consent was obtained from people whose
asset is used.
•At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14.Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
21

Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
•Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
•According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15.Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
•Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
•We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
•For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16.Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA]
Justification: The core method development in this research does not involve LLMs as any
important, original, or non-standard components.
Guidelines:
•The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
•Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM )
for what should or should not be described.
22