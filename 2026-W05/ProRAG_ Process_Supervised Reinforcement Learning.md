# ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation

**Authors**: Zhao Wang, Ziliang Zhao, Zhicheng Dou

**Published**: 2026-01-29 16:04:59

**PDF URL**: [https://arxiv.org/pdf/2601.21912v1](https://arxiv.org/pdf/2601.21912v1)

## Abstract
Reinforcement learning (RL) has become a promising paradigm for optimizing Retrieval-Augmented Generation (RAG) in complex reasoning tasks. However, traditional outcome-based RL approaches often suffer from reward sparsity and inefficient credit assignment, as coarse-grained scalar rewards fail to identify specific erroneous steps within long-horizon trajectories. This ambiguity frequently leads to "process hallucinations", where models reach correct answers through flawed logic or redundant retrieval steps. Although recent process-aware approaches attempt to mitigate this via static preference learning or heuristic reward shaping, they often lack the on-policy exploration capabilities required to decouple step-level credit from global outcomes. To address these challenges, we propose ProRAG, a process-supervised reinforcement learning framework designed to integrate learned step-level supervision into the online optimization loop. Our framework consists of four stages: (1) Supervised Policy Warmup to initialize the model with a structured reasoning format; (2) construction of an MCTS-based Process Reward Model (PRM) to quantify intermediate reasoning quality; (3) PRM-Guided Reasoning Refinement to align the policy with fine-grained process preferences; and (4) Process-Supervised Reinforcement Learning with a dual-granularity advantage mechanism. By aggregating step-level process rewards with global outcome signals, ProRAG provides precise feedback for every action. Extensive experiments on five multi-hop reasoning benchmarks demonstrate that ProRAG achieves superior overall performance compared to strong outcome-based and process-aware RL baselines, particularly on complex long-horizon tasks, validating the effectiveness of fine-grained process supervision. The code and model are available at https://github.com/lilinwz/ProRAG.

## Full Text


<!-- PDF content starts -->

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
Zhao Wang
Gaoling School of Artificial
Intelligence, Renmin University of
China, Beijing, China
lilin22wz@gmail.comZiliang Zhao
Gaoling School of Artificial
Intelligence, Renmin University of
China, Beijing, China
zhaoziliang@ruc.edu.cnZhicheng Dou∗
Gaoling School of Artificial
Intelligence, Renmin University of
China, Beijing, China
dou@ruc.edu.cn
Abstract
Reinforcement learning (RL) has become a promising paradigm for
optimizing Retrieval-Augmented Generation (RAG) in complex rea-
soning tasks. However, traditional outcome-based RL approaches
often suffer from reward sparsity and inefficient credit assignment,
as coarse-grained scalar rewards fail to identify specific erroneous
steps within long-horizon trajectories. This ambiguity frequently
leads to “process hallucinations”, where models reach correct an-
swers through flawed logic or redundant retrieval steps. Although
recent process-aware approaches attempt to mitigate this via static
preference learning or heuristic reward shaping, they often lack
the on-policy exploration capabilities required to decouple step-
level credit from global outcomes. To address these challenges, we
proposeProRAG, a process-supervised reinforcement learning
framework designed to integrate learned step-level supervision
into the online optimization loop. Our framework consists of four
stages: (1)Supervised Policy Warmupto initialize the model with
a structured reasoning format; (2) construction of anMCTS-based
Process Reward Model (PRM)to quantify intermediate reason-
ing quality; (3)PRM-Guided Reasoning Refinementto align
the policy with fine-grained process preferences; and (4)Process-
Supervised Reinforcement Learningwith a dual-granularity
advantage mechanism. By aggregating step-level process rewards
with global outcome signals, ProRAG provides precise feedback for
every action. Extensive experiments on five multi-hop reasoning
benchmarks demonstrate that ProRAG achieves superior overall
performance compared to strong outcome-based and process-aware
RL baselines, particularly on complex long-horizon tasks, validating
the effectiveness of fine-grained process supervision. The code and
model are available at https://github.com/lilinwz/ProRAG.
CCS Concepts
•Computing methodologies →Natural language generation;
Reinforcement learning;•Information systems →Question
answering.
Keywords
Retrieval-Augmented Generation, Reinforcement Learning, Ques-
tion Answering, Large Language Models
1 Introduction
Recent Large Language Models (LLMs) have demonstrated remark-
able reasoning capabilities [ 4,16], significantly improving their per-
formance on complex, knowledge-intensive tasks. These advances
∗Zhicheng Dou is the corresponding author.
Step 1 
(reliable)Step 2 
(unreliable)Answer
(correct)
√ × √
+ Positive score -Negative score + Positive score√√
?Offline Data
Distribution Shift
PolicyCurrent RolloutStep 1 
(reliable)Step 2 
(unreliable)Answer
(correct)
√
Spurious correlation!√ + Global Positive score
Offline DPOOutcome -based RL
ProRAG -OursFigure 1: Comparison of optimization paradigms. (a)
Outcome-based RL suffers from spurious correlations due
to sparse global signals. (b) Offline DPO is limited by dis-
tribution shifts caused by static datasets. (c) ProRAG (Ours)
utilizes dense step-level supervision for precise credit assign-
ment and dynamic error correction.
have propelled Retrieval-Augmented Generation (RAG) from static
pipelines into agentic systems, where models act as autonomous
decision-makers that actively plan retrieval strategies, interact with
external tools, and iteratively refine their reasoning based on feed-
back [ 3,14]. However, optimizing such dynamic workflows with
standard supervised learning remains challenging. Effective reason-
ing and retrieval trajectories are often open-ended, which makes
fine-grained ground-truth supervision for intermediate decisions
unavailable or prohibitively expensive. To bridge this gap, Rein-
forcement Learning (RL) has been increasingly adopted to optimize
the model policy from the final outcome signals [9, 21].
However, traditional outcome-based reinforcement learning of-
ten suffers from reward sparsity and the credit assignment problem,
making it ineffective for complex multi-hop RAG tasks. In such
long-horizon trajectories, a coarse-grained scalar reward based
on final answer correctness often fails to distinguish the specific
contributions of intermediate actions. This ambiguity frequently
results in “process hallucinations”, where the model is rewardedarXiv:2601.21912v1  [cs.AI]  29 Jan 2026

Zhao Wang, Ziliang Zhao, and Zhicheng Dou
for reaching the correct answer while relying on flawed reasoning,
redundant retrieval, or unverified assumptions. Consequently, such
outcome-oriented feedback can inadvertently reinforce spurious
correlations rather than authentic reasoning abilities. Furthermore,
without dense supervision to pinpoint specific errors, the model
may struggle to correct early deviations during training, resulting
in inefficient optimization.
To address these challenges, recent studies have explored opti-
mization strategies for incorporating process-level signals, broadly
categorized into offline and online approaches. Offline methods [ 12,
22,37] leverage search algorithms like MCTS to synthesize prefer-
ence trajectories for Direct Preference Optimization (DPO), yet this
reliance on static data constrains the model’s ability to adapt to on-
policy distribution shifts. Conversely, online methods often resort
to heuristic reward shaping [ 6,31], step-wise normalization [ 28,32],
or multi-agent interaction [ 26] to stabilize training. However, these
approaches primarily focus on regulating reward magnitudes or
relying on expensive LLM judges, rather than verifying the logi-
cal validity of intermediate steps. Furthermore, while tree-based
exploration [ 36] and inference-time refinement modules [ 29] can
help mitigate errors, they introduce considerable latency and fail
to internalize reasoning capabilities into the policy. Therefore, it
remains challenging to provide dense, exploration-derived process
supervision directly during online training.
In this paper, we proposeProRAG, a process-supervised re-
inforcement learning framework designed to resolve this credit
assignment problem in the multi-hop RAG task. Unlike previous
approaches that rely on sparse outcome signals, static offline pref-
erences, or heuristic rules, ProRAG integrates learned step-level
supervision directly into the online optimization loop. Specifically,
we start with a Supervised Policy Warmup to initialize the model
with a structured reasoning format. Leveraging this policy as a
prior, we establish a fine-grained evaluator by training a Process
Reward Model (PRM) on diverse reasoning paths constructed via
Monte Carlo Tree Search (MCTS). To bridge the alignment gap
between the initial policy and this evaluator, we introduce a Rea-
soning Refinement stage, which effectively warms up the model
using high-quality trajectories filtered by the PRM. Finally, we im-
plement a Process-Supervised Reinforcement Learning stage with
a dual-granularity mechanism. By aggregating step-level process
advantages from the PRM with global outcome rewards, ProRAG
provides immediate and precise feedback for each action. This
mechanism allows the model to dynamically identify and correct
intermediate errors, learning how to reason correctly rather than
overfitting to the final answer.
We conduct extensive experiments on five challenging multi-
hop reasoning benchmarks, including PopQA [ 15], HotpotQA [ 34],
2WikiMultihopQA [ 7], MuSiQue [ 23], and Bamboogle [ 17]. Experi-
mental results demonstrate that ProRAG achieves superior perfor-
mance compared to strong outcome-based and process-aware base-
lines, particularly on complex long-horizon tasks, validating that
dense process supervision provides a more effective optimization
signal than sparse outcome rewards or offline preference learning.
Furthermore, comprehensive ablation studies confirm the essential
contributions of each component in our framework, showing that
the PRM-guided refinement and the dual-granularity advantagemechanism are critical for the observed performance gains. In ad-
dition, our analysis shows that ProRAG effectively regulates the
retrieval process, remaining robust to irrelevant documents while
adaptively planning reasoning steps according to task complexity.
Efficiency evaluations further demonstrate that by internalizing
reasoning capabilities, ProRAG achieves high data efficiency and
low inference latency, facilitating real-world deployment.
The main contributions of this paper include:
(1) We propose ProRAG, a process-supervised reinforcement
learning framework that resolves the credit assignment problem in
multi-hop RAG tasks via a dual-granularity advantage mechanism
combining learned step-level process rewards with outcome signals.
(2) We introduce an MCTS-based Process Reward Model that
leverages tree-search exploration to quantify intermediate reason-
ing quality and effectively identify process hallucinations.
(3) We design a PRM-Guided Reasoning Refinement strategy to
align the policy with fine-grained process preferences, ensuring
stable convergence and mitigating the cold-start problem in RL.
2 Related Works
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has emerged as an impor-
tant paradigm for mitigating hallucination and knowledge obsoles-
cence in LLMs by dynamically incorporating relevant information
from external knowledge sources [ 13,38]. Early research on RAG
primarily focused on a standard retrieve-then-generate framework,
where a retriever identifies relevant documents based on the in-
put query, and a generator synthesizes the final answer [ 5,10]. To
handle more complex multi-hop queries that require multiple re-
trieval steps, subsequent studies have introduced iterative retrieval
mechanisms. Methods such as Iter-RetGen [ 19] and IRCoT [ 24]
alternate between retrieval and generation steps to progressively
refine the context. Furthermore, active strategies like FLARE [ 8]
and Self-RAG [ 1] improve efficiency via dynamic retrieval triggered
by low-confidence generation or token-level signals.
However, these pipelines typically follow predefined workflows,
limiting their adaptability to handle open-ended complex reasoning
tasks. To address this, recent research has explored Agentic RAG,
where LLMs serve as autonomous agents that orchestrate the re-
trieval and reasoning process [ 2]. In this paradigm, models explicitly
plan search queries, interact with search engines, and evaluate re-
trieved evidence. Foundational works like ReAct [ 35] demonstrated
the potential of interleaving reasoning traces with external tool
actions. Building on this, recent advanced systems such as Search-
o1 [14] and DeepRAG [ 3] model the RAG process as a multi-step
decision-making task to handle long-horizon queries. Despite these
architectural advances, most existing agentic RAG systems rely
primarily on supervised fine-tuning or complex prompting engi-
neering, fundamentally limiting their ability to generalize to novel
scenarios where optimal reasoning trajectories are unavailable.
2.2 Reinforcement Learning for RAG
Reinforcement learning (RL) has become a promising paradigm for
optimizing RAG systems by maximizing task-specific rewards to
overcome the limitations of static supervision. Recent advances,
such as Search-R1 [ 9] and R1-Searcher [ 21], have demonstrated

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
User Query
...…
correct unreliable wrongAction 1
Action 2
Action 3User Query :Has the current Premier League 
leader ever won the Champions League?
Model Output:
<step> I need to first identify which team is 
currently top of the Premier League table.</step>
<subquery> current Premier League table 1st 
place</subquery>
<retrieval> As of today, Arsenal sits at the top of 
the Premier League table.</retrieval>
<step> I first check the retrieved article to identify 
the current leader of the Premier League, which 
shows that Arsenal is in first place.</step >
<subanswer >The current Premier League leader is 
Arsenal.</ subanswer >
……
<step>Since the current leader (Arsenal ) has won 
the title, the answer is affirmative.</step>
<answer>Yes</answer>Rejected Chosen
Figure 2: The left panel presents the reasoning format of our model, while the right illustrates the Monte Carlo Tree Search
(MCTS) mechanism used to train the Process Reward Model.
that outcome-supervised RL can significantly enhance the model’s
ability to invoke search tools and reason over retrieved content.
However, as noted in recent studies [ 12,37], outcome-based ap-
proaches can suffer from reward sparsity and gradient conflict in
long-horizon tasks. The correct final answer does not guarantee a
correct reasoning process, often leading to “process hallucinations”
where models are reinforced for reaching the right conclusion via
flawed logic or redundant retrieval [31].
To address this credit assignment problem, recent works have
explored integrating process-level supervision, broadly categorized
into offline and online approaches. Offline methods [ 12,22,37]
utilize search algorithms like MCTS or decoupled execution to syn-
thesize high-quality preference datasets for DPO. However, these
methods rely on static datasets, limiting the model’s ability to
adapt to on-policy distribution shifts. Conversely, online meth-
ods attempt to provide denser feedback during training. Some ap-
proaches [ 28,31,32] introduce step-wise heuristic reward shaping
or normalization strategies. More recent frameworks [ 6,26] lever-
age multi-agent interactions or multi-dimensional reward functions
to guide reasoning. While effective, these methods typically rely on
expensive external LLM judges or manual rule design, lacking the
efficiency of a learned verifier. Furthermore, while tree-based explo-
ration [ 36] and inference-time refinement modules [ 29] can help
mitigate errors, they introduce considerable computational latency
during training and inference, often failing to efficiently internal-
ize reasoning capabilities into the policy model. In contrast, our
ProRAG integrates a learned MCTS-based Process Reward Model
directly into the online RL loop. By employing a dual-granularity
advantage mechanism, ProRAG provides more precise, dense, and
efficient supervision.3 ProRAG
Existing reinforcement learning approaches for RAG often suffer
from reward sparsity and inefficient credit assignment in com-
plex multi-step reasoning. To address these challenges, we propose
ProRAG, which introduces process rewards for step-level fine-
grained optimization, thereby enhancing the reasoning capabilities
of the RAG model. Our framework consists of four stages: (1)Su-
pervised Policy Warmupto align the model with a multi-step
reasoning format; (2)MCTS-based Process Reward Modeling
to construct a step-level process reward model; (3)PRM-Guided
Reasoning Refinementto rapidly enhance the model policy; and
(4)Process-Supervised Reinforcement Learningto perform
the final optimization via fine-grained rewards.
3.1 Supervised Policy Warmup
To equip the LLM with the capability for complex multi-hop reason-
ing and autonomous retrieval, we begin with a Supervised Policy
Warmup. In this stage, we fine-tune the model using a constructed
dataset that follows a structured reasoning–action format, estab-
lishing a stable reference policy𝜋 sftfor subsequent training.
Since standard QA benchmarks typically provide sub-query
chains and corresponding retrieval documents without any explicit
reasoning process, we construct a high-quality dataset to teach the
model this capability. To achieve this, we leverage a superior LLM
(e.g., GPT-4o) to synthesize detailed chain-of-thought trajectories.
By prompting the teacher model to generate the reasoning process
for each step based on the provided sub-query and document, we
obtain a diverse set of reasoning paths that bridge the gap between
the initial query and the final answer.
To enable fine-grained process supervision, we organize the
synthesized trajectories into a structured schema, as illustrated in

Zhao Wang, Ziliang Zhao, and Zhicheng Dou
the left panel of Figure 2. Each reasoning trajectory consists of a
sequence of steps enclosed by special control tokens: <step> marks
the internal thought process for planning; <subquery> denotes the
generated keywords for the retrieval system; <retrieval> wraps
the external evidence returned by the environment; <subanswer>
indicates intermediate conclusions; and <answer> represents final
answer. This structured format transforms unstructured reasoning
into a sequence of discrete and evaluable actions.
Leveraging these structured trajectories, we perform supervised
fine-tuning to establish the reference policy 𝜋SFT. Rather than train-
ing on continuous sequences where the learning signal might be
obscured by long reasoning process and retrieval documents, we
formulate the training data as specific input-target pairs. The input
comprises the historical context, including frozen retrieval content,
while the target output consists of the current reasoning and ac-
tion block (i.e., <step> and<subquery> ). This approach directs the
model’s attention entirely towards the immediate generation task.
To further enforce structural format, we adopt a format-aware
training objective:
LSFT=−∑︁
𝑡∉Tctrllog𝑃(𝑦𝑡|𝒄,𝒚 <𝑡)−𝜆∑︁
𝑡∈T ctrllog𝑃(𝑦𝑡|𝒄,𝒚 <𝑡),
(1)
whereTctrlrepresents the set of special control tokens (e.g., <step> ,
<subquery> ). By setting 𝜆> 1, we ensure our model robustly
follows the format schema in this stage.
3.2 MCTS-based Process Reward Modeling
Although the SFT phase equips the model with the fundamen-
tal reasoning format, only relying on final answer correctness for
supervision remains insufficient. This coarse-grained feedback of-
ten leads to “process hallucinations”, where the model arrives at
a correct answer via flawed logic or redundant retrieval actions.
To mitigate this, we introduce a Process Reward Model (PRM) to
evaluate the validity of intermediate reasoning steps.
To achieve this, we employ Monte Carlo Tree Search (MCTS) to
construct a dataset of diverse reasoning paths with step-level labels.
Specifically, we formulate the reasoning process as a tree search
problem, utilizing the SFT policy 𝜋sftas a prior. Here, the state 𝑠𝑡
represents the accumulated reasoning history (including retrieval
contexts), and the action 𝑎𝑡corresponds to a generated reasoning
step.
In the selection phase, we traverse the tree andselectchild nodes
maximizing the Predictor Upper Confidence Bound (PUCT) to bal-
ance exploitation of high-value paths and the exploration of uncer-
tain ones:
𝑎𝑡=arg max
𝑎 
𝑄(𝑠𝑡,𝑎)+𝑐 puct·𝜋SFT(𝑎|𝑠𝑡)√︁Í
𝑎′𝑁(𝑠𝑡,𝑎′)
1+𝑁(𝑠𝑡,𝑎)!
.(2)
Upon reaching a leaf node, weexpandnew candidate steps by
sampling candidate steps from 𝜋𝑠𝑓𝑡at a high temperature. We then
performsimulationto generate complete trajectories and evaluate
the terminal state against the ground truth to obtain a binary cor-
rectness score 𝑣∈{ 0,1}, which isbackpropagatedalong the search
path to update visit counts and mean action values. For a nodecorresponding to step𝑡≤𝑇, we update its action value as:
𝑄(𝑠𝑡,𝑎𝑡)←𝑄(𝑠𝑡,𝑎𝑡)·𝑁(𝑠𝑡,𝑎𝑡)+𝛾𝑇−𝑡·𝑣
𝑁(𝑠𝑡,𝑎𝑡)+1,(3)
where𝛾is a decay factor that discounts rewards for redundant
trajectories and encourages more direct solutions.
Through iterative simulations, MCTS effectively explores the
solution space and accumulates a rich set of trajectories that distin-
guish between valid reasoning paths and those leading to errors.
While MCTS enables efficient exploration and assigns values
to reasoning trajectories, the resulting Q-values remain primarily
outcome-oriented. A reasoning step may receive a high value simply
because it leads to a correct final answer, even if the step itself
contains logical errors or unsupported assumptions. Therefore,
such outcome-based signals are insufficient for reliably supervising
fine-grained reasoning quality.
To obtain fine-grained process-level supervision, we further con-
struct contrastive step-level preference pairs from the MCTS search
tree. Specifically, we select sibling nodes that share the same par-
ent context but differ at the current reasoning step, and employ
GPT-4o as a logical judge to compare their reasoning quality. The
model labels each pair as “Chosen” ( 𝑦+) or “Rejected” ( 𝑦−) based
on strict logical validity. To validate the reliability of these labels,
we manually evaluated 50 randomly sampled pairs and observed a
96% agreement rate between GPT-4o and human annotators, indi-
cating high label reliability. These contrastive pairs filter out noise
from outcome-based signals and provide direct supervision for the
PRM, enabling it to distinguish logically valid steps from flawed
alternatives under identical contexts.
Finally, we train the Process Reward Model R𝜙by these con-
structed contrastive pairs, to quantify the quality of intermediate
reasoning. The model takes the query context and a specific reason-
ing step as input and outputs a scalar score. We optimize R𝜙using
a pairwise ranking loss to maximize the score margin between the
chosen and rejected steps:
LPRM=−E(𝑥,𝑦+,𝑦−)∼D
log𝜎 R𝜙(𝑥,𝑦+)−R𝜙(𝑥,𝑦−)
,(4)
where𝜎denotes the sigmoid function and 𝐷represents the collected
preference dataset. By minimizing this loss, R𝜙serves as a dense
verifier to provide step-level feedback for the subsequent stages.
3.3 PRM-Guided Reasoning Refinement
Before the reinforcement learning phase, it is crucial to bridge the
alignment gap between the initial SFT policy and the fine-grained
preferences of the PRM. While 𝜋sftlearns the desired structural
format via SFT, its output distribution remains broad in quality,
often deviating from the logical standards encoded in the PRM.
Directly optimizing such an unaligned policy via RL typically suffers
from severe cold-start issues due to the sparsity of high-reward
trajectories in the initial search space. To address this, we propose
a Reasoning Refinement stage as an intermediate step between
SFT and reinforcement learning. By fine-tuning the policy model
on the high-quality reasoning trajectories filtered by the PRM, we
improve step-level process correctness and reduce the distribution
mismatch between the SFT policy and the PRM. As a result, the
refined policy provides a well-aligned and stable initialization for
subsequent reinforcement learning.

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
Process 
Reward Model
Outcome 
Reward FunctionQueryPolicy Model
EnvironmentRollout𝑜1,1,…,𝑜1,𝑙1Step 1
𝑜2,1,…,𝑜2,𝑙1Step 2
𝑜𝑡,1,…,𝑜𝑡,𝑙𝑡Step t……𝑟1𝑝𝑟𝑜𝑐
𝑟2𝑝𝑟𝑜𝑐
𝑟𝑡𝑝𝑟𝑜𝑐
𝑟𝑡𝑜𝑢𝑡
Group norm
broadcast
Group norm⊕
broadcast𝐴1
𝐴2
…
𝐴𝑡…
Figure 3: The framework of Process-Supervised Reinforcement Learning.
To achieve this, we employ a Step-level Rejection Sampling Fine-
Tuning (RFT) strategy to construct a refinement dataset. Specifically,
we utilize the SFT policy 𝜋sftto generate 𝑁candidate trajectories
for each query. To ensure that the policy model learns only from
optimal reasoning paths, we apply a dual-criterion filtering mecha-
nism. First, we strictly require outcome correctness, retaining only
those trajectories that reach the correct final answer. Second, we
impose process validity by evaluating every intermediate step using
our trained PRMR𝜙. A specific (context, action) pair is preserved
for training if and only if its reward score exceeds a predefined
threshold (e.g.,R𝜙(𝑠𝑡>0)). This selective mechanism allows us to
collect high-quality reasoning segments, thereby selective aligning
the policy with the PRM’s high-value regions.
Finally, we fine-tune the initial policy 𝜋SFTon these curated
trajectories. By treating the (context, action) pairs as high-quality
demonstrations, we optimize the model using the standard next-
token prediction objective. This process effectively distills the sparse
and high-value signals from the PRM into the parameters of the
refined policy model 𝜋RFT. As a result, 𝜋rftexhibits substantially
stronger reasoning capability than the initial policy 𝜋sft, effectively
mitigating the cold-start problem in the subsequent reinforcement
learning stage.
3.4 Process-Supervised Reinforcement Learning
Following the Reasoning Refinement stage, we employ Reinforce-
ment Learning (RL) to further enhance the model’s capabilities.
However, traditional RL algorithms such as PPO [ 18] and GRPO [ 20]
often rely on sparse outcome rewards and suffer from the credit
assignment problem in multi-hop RAG tasks, preventing the pol-
icy model from identifying specific erroneous steps or verifying
intermediate reasoning. To address this, we propose ProRAG, a
reinforcement learning framework with a dual-granularity advan-
tage mechanism. As illustrated in Figure 3, it aggregates group-
normalized advantages from both step-level process signals and
final outcome rewards, providing fine-grained and dense supervi-
sion throughout the entire reasoning process.
3.4.1 Group Trajectory Sampling.To estimate policy variance, we
sample groups of trajectories from the current policy 𝜋𝜃, which
is initialized from the refinement model 𝜋rft. Specifically, for a
given query 𝑞, we sample a group of 𝐺independent trajectories
Y={𝑦 1,𝑦2,...,𝑦𝐺}from𝜋𝜃. Each trajectory 𝑦𝑖is formulated as a
sequence of discrete reasoning steps 𝑦𝑖=(𝑠𝑖,1,𝑠𝑖,2,...,𝑠𝑖,𝑇), whereeach step𝑠𝑖,𝑡=(𝑜𝑖,𝑡,1,...,𝑜𝑖,𝑡,𝑙𝑖,𝑡)corresponds to a sub-sequence of
tokens generated by the policy.
3.4.2 Reward Formulation.To evaluate these generations, we de-
rive two distinct reward signals from dense step-level supervision
and sparse trajectory-level verification.
First, we compute astep-level process reward 𝑟step
𝑖,𝑡for each
intermediate reasoning step 𝑠𝑖,𝑡within a trajectory 𝑦𝑖. This reward
assesses the quality of the action based on the local format con-
straint and the probability score from our frozen PRM R𝜙. Formally,
given the historical context𝒄 𝑖,𝑡, the process reward is defined as:
𝑟step
𝑖,𝑡=R𝜙(𝒄𝑖,𝑡,𝑠𝑖,𝑡)+𝜈 1·Istep(𝑠𝑖,𝑡),(5)
where Istep(·)is an indicator function that returns1if the current
step𝑠𝑖,𝑡strictly adheres to the specified tag schema (e.g., correct
usage of <step> or<subquery> ) and0otherwise. 𝜈1is a coefficient
for this format bonus.
Second, we assign a terminaloutcome reward 𝑟out
𝑖after the
completion of the trajectory to evaluate the global utility. This
reward is determined by the correctness of the final answer and
the format integrity of the complete reasoning chain:
𝑟out
𝑖=F1( ˆ𝑎𝑖,𝑎∗)+𝜈 2·Itraj(𝑦𝑖).(6)
Here, ˆ𝑎𝑖denotes the predicted answer extracted from 𝑦𝑖, and𝑎∗is
the ground truth. The function F1(·) computes the token-level ac-
curacy between the prediction and the ground truth. The indicator
function Itraj(𝑦𝑖)evaluates whether the entire trajectory 𝑦𝑖follows
the complete reasoning-action workflow, discouraging the policy
from bypassing required retrieval or intermediate reasoning steps.
𝜈2is a coefficient for this format bonus.
3.4.3 Dual-Granularity Advantage Estimation.To address the credit
assignment problem in multi-turn trajectories, we propose a dual-
granularity advantage estimation strategy, which aggregates dense
step-level supervision and sparse outcome reward.
For a generated trajectory 𝑦𝑖, we broadcast the rewards to the
token level and define theprocess advantage 𝐴proc
𝑖,𝑡,𝑘andoutcome
advantage𝐴out
𝑖,𝑡,𝑘for the𝑘-th token within step𝑠 𝑖,𝑡as follows:
𝐴proc
𝑖,𝑡,𝑘=𝑟step
𝑖,𝑡−𝜇step
𝜎step, 𝐴out
𝑖,𝑡,𝑘=𝑟out
𝑖−𝜇out
𝜎out,(7)

Zhao Wang, Ziliang Zhao, and Zhicheng Dou
where𝜇and𝜎denote the group-wise mean and standard deviation
of the corresponding rewards. By this mechanism, the reward sig-
nals are broadcast to all tokens 𝑘within each step 𝑠𝑖,𝑡, providing
dense supervision for optimization.
By weighted aggregation of the process advantage and the out-
come advantage, we obtain thetotal advantage𝐴 𝑖,𝑡,𝑘as:
𝐴𝑖,𝑡,𝑘=𝐴out
𝑖,𝑡,𝑘+𝛽·𝐴proc
𝑖,𝑡,𝑘.(8)
This dual-granularity advantage estimation mitigates the lim-
itations of single-source supervision. Using only sparse outcome
rewards provides insufficient credit assignment for intermediate
reasoning steps, while relying solely on step-level process rewards
may encourage optimization of local signals at the expense of global
correctness. By combining outcome advantages with weighted pro-
cess advantages, ProRAG delivers dense token-level supervision
while preserving alignment with the final reasoning objective.
3.4.4 Policy Optimization.Finally, we optimize the policy 𝜋𝜃by
maximizing the total advantages over the trajectory groups. Consis-
tent with GRPO [ 20], we compute baselines via the group statistics
in Eq. 8 instead of a value network, ensuring memory efficiency.
The loss function is formulated as:
L(𝜃)=−E 𝑞∼D,Y∼𝜋 old1
𝐺𝐺∑︁
𝑖=1𝑇𝑖∑︁
𝑡=1𝑙𝑖,𝑡∑︁
𝑘=1min
𝜌𝑖,𝑡,𝑘𝐴𝑖,𝑡,𝑘,
clip(𝜌𝑖,𝑡,𝑘,1−𝜀,1+𝜀)𝐴 𝑖,𝑡,𝑘
,(9)
where𝜌𝑖,𝑡,𝑘=𝜋𝜃(𝑜𝑖,𝑡,𝑘|𝒉𝑖,𝑡,𝑘)
𝜋old(𝑜𝑖,𝑡,𝑘|𝒉𝑖,𝑡,𝑘)represents the probability ratio be-
tween the current and reference policies, where 𝒉𝑖,𝑡,𝑘includes all
context preceding token𝑜 𝑖,𝑡,𝑘and𝜀is the clipping coefficient.
4 Experiment
4.1 Dataset and Metrics
We evaluate ProRAG and all baseline models on five diverse bench-
marks, including one general QA dataset PopQA [ 15], and four
multi-hop QA datasets: HotpotQA [ 34], 2WikiMultiHopQA [ 7],
MuSiQue [ 23], and Bamboogle [ 17]. These datasets encompass a
wide spectrum of challenges, from long-tail knowledge retrieval to
complex multi-step deductive reasoning, providing a comprehen-
sive evaluation of the models’ ability to plan, retrieve, and reason
across varying levels of difficulty and distribution shifts. For these
benchmarks, we report Exact Match (EM) and F1 Score as metrics
to evaluate the answer accuracy. Furthermore, to verify the statis-
tical significance of our performance gains, we conduct a paired
t-test comparing ProRAG against the strongest baseline with a
significance level of𝑝<0.05.
4.2 Baselines
We compare ProRAG against a comprehensive set of baselines,
which can be categorized into three types as follows:
(1)Standard Baselines.These include Naïve Generation and
Standard RAG, which represent the lower bound of performance.
Naïve Generation relies solely on the parametric knowledge of
the pre-trained LLM without external retrieval. Standard RAG [ 13]
represents the traditional retrieve-then-generate framework, whichperforms a single-step retrieval based on the initial query before
generating the final answer.
(2)Advanced Baselines.This category encompasses inference-
time methods that dynamically adjust the search process or em-
ploy agentic planning. We evaluate Iter-RetGen[ 19] and IRCoT[ 24],
which interleave retrieval and reasoning steps in an iterative loop to
progressively gather evidence. Additionally, we include FLARE[ 8],
an active retrieval strategy that triggers search actions based on
generation confidence. Furthermore, we benchmark against Search-
o1[14], a strong agentic baseline that treats LLMs as autonomous
agents capable of multi-step planning, task decomposition, and
reflection to explicitly control the retrieval workflow.
(3)Reinforcement Learning Baselines.Finally, we compare
against state-of-the-art methods that optimize the policy through
reinforcement learning. For outcome-based RL, we selectSearch-
R1[9] as the primary baseline, which employs PPO to optimize the
policy based on the final answer. For offline process supervision, we
includeReasonRAG[ 37], which uses MCTS to synthesize static
preference data for DPO, providing step-level signals for optimiza-
tion. For online process-aware RL, we compare with HiPRAG [ 31],
a recent method that shapes rewards via sub-goal heuristics.
4.3 Implementation Details
We implement our framework based on the HuggingFace Trans-
formers [ 30] and TRL [ 25] Libraries, with Qwen3-8B [ 33] serving
as the backbone for both our method and all baselines. For the
retrieval environment, we use the 2018 Wikipedia dump [ 11] as the
knowledge source and E5-base [ 27] as the retriever, retrieving the
top 3 documents for each query.
We conduct all experiments on 4 NVIDIA A100 (80GB) GPUs. For
the Supervised Policy Warmup stage, we perform full-parameter
fine-tuning on 109k MuSiQue context–action pairs for one epoch
to initialize the model’s reasoning capability. In the Process Re-
ward Modeling stage, we sample 728 queries from HotpotQA and
MuSiQue, and then perform MCTS with 200 simulations, an ex-
pansion width of 𝐾= 5, a maximum depth of 10, an exploration
constant𝑐puct=2.5, and a discount factor 𝛾=0.99. For these trees,
we collect 8,255 contrastive pairs labeled by GPT-4o, and train the
scalar regression head from the SFT checkpoint for three epochs.
Next, in the Reasoning Refinement Stage we select 105k high-quality
trajectories from the two datasets filtered by the trained PRM, and
fine-tune the policy model for one additional epoch to enhance
reasoning ability. Finally, in the Process-Supervised RL stage, we
apply LoRA fine-tuning to optimize the policy on 10k held-out
queries drawn from both datasets for 1 epoch, with a generation
group size of𝐺=8and a dual-granularity weight𝛽=0.3.
4.4 Main Results
The overall performance of ProRAG and baseline models across
five benchmarks is presented in Table 1.
Comparison with Standard and Advanced Baselines.As
shown in the table, standard and advanced baselines generally
struggle to achieve high accuracy compared to RL-based meth-
ods. Specifically, Naïve Generation performs worst due to internal
knowledge limitations, while Standard RAG provides only modest
improvements, suggesting that single-step retrieval is insufficient

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
Table 1: Overall results of ProRAG and baselines on five benchmarks. The best and second-best scores are highlighted in bold
and underlined .†indicates statistically significant improvements over the best baseline (Search-R1) with𝑝<0.05.
MethodPopQA HotpotQA 2Wiki MuSiQue Bamboogle Avg.
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Standard Methods
Naïve Generation 11.7 16.6 9.8 19.0 15.8 21.0 2.1 9.5 4.8 11.0 8.8 15.4
Standard RAG 27.0 35.9 20.0 31.5 16.1 23.0 3.3 10.3 12.0 20.1 15.7 23.3
Advanced Methods
Iter-RetGen 32.9 37.8 15.9 21.2 9.3 11.5 4.8 8.5 17.6 23.7 16.1 20.5
IRCoT 33.6 42.2 27.2 37.9 25.1 30.4 5.1 13.1 16.8 28.6 21.6 30.4
FLARE 39.3 46.3 28.9 40.1 24.1 29.5 5.8 13.5 27.2 37.8 25.1 33.4
Search-o1 38.4 43.9 28.6 38.1 22.6 27.9 13.0 19.9 38.4 50.1 28.2 36.0
Reinforcement Learning-based Methods
Search-R1 45.9 50.1 42.7 54.639.9 45.2 20.6 29.5 43.2 54.1 38.5 46.7
ReasonRAG 41.5 46.2 38.4 48.9 43.6 50.4 12.8 20.6 36.0 45.5 34.5 42.3
HiPRAG 34.3 42.7 36.8 49.1 37.0 43.2 15.6 25.0 39.2 51.6 32.6 42.3
ProRAG (Ours) 47.2†51.6†41.4 52.8 46.0†51.1†23.5†34.1†45.6 56.4 40.7†49.2†
for complex multi-hop dependencies. Although iterative strategies
like Iter-RetGen and IRCoT improve upon standard methods by
interleaving retrieval and reasoning, their performance gains are
limited by the accumulation of noise from retrieved documents. Sim-
ilarly, while dynamic methods such as FLARE and agentic baselines
like Search-o1 achieve competitive results on specific tasks (e.g.,
Bamboogle), they still fall short of the optimal performance. Search-
o1, for instance, is constrained by general-purpose prompting and
lacks task-specific policy optimization. In contrast, reinforcement
learning baselines consistently outperform these approaches across
all metrics, highlighting the necessity of explicitly aligning the
retrieval and reasoning policy.
Comparison with Reinforcement Learning Baselines.For
the RL methods, ProRAG demonstrates superior overall perfor-
mance, surpassing the strongest baseline by 2.5% in average F1
score. We observe that outcome-based approaches like Search-R1 re-
main competitive on HotpotQA, primarily because this benchmark
serves as a standard in-domain task where the reasoning chains are
relatively shallow and easier to fit using sparse rewards. However,
this advantage disappears on more complex tasks. As shown in
Table 1, ProRAG achieves statistically significant improvements
(𝑝<0.05) on challenging datasets like MuSiQue and 2WikiMulti-
hopQA, confirming that sparse outcome signals are insufficient for
learning long-horizon dependencies. Similarly, offline optimization
methods like ReasonRAG lag behind, as its optimization on static
data lacks the on-policy exploration required to discover novel so-
lution paths. In contrast, by integrating a learned Process Reward
Model with online optimization, ProRAG provides dense feedback
that effectively navigates the complex search space, leading to the
highest average accuracy.
Generalization and Robustness on Complex Tasks.Beyond
overall averages, ProRAG exhibits superior generalization and ro-
bustness across different benchmarks. On out-of-domain datasetssuch as PopQA and 2WikiMultihopQA, ProRAG consistently out-
performs all baseline methods, indicating that our process-level
supervision fosters transferable reasoning skills rather than merely
overfitting to the training distribution. Furthermore, we observe
distinct performance trends on tasks requiring deeper reasoning.
While baselines tend to plateau on the complex MuSiQue dataset,
ProRAG maintains a clear lead (+4.6% F1 over Search-R1), validating
the necessity of step-level verification for long reasoning chains.
We also note that on Bamboogle, ProRAG achieves a substantial
absolute improvement of 2.3% F1; however, due to the extremely
small test set size (125 examples), this dataset provides insufficient
statistical power to establish significance, despite the evident per-
formance margin. Overall, these results demonstrate that ProRAG
is the most robust framework, ensuring both in-domain stability
and reliable out-of-domain generalization.
4.5 Ablation Studies
To validate the effectiveness of each component in ProRAG, we
conduct ablation studies by comparing our full model against four
variants. We define these variants as follows: (1)SFT Policy: The
base model fine-tuned only on the supervised warmup data; (2)
GRPO Baseline: A strong outcome-based RL baseline that opti-
mizes the SFT policy using Online GRPO with only sparse final
rewards; (3)w/o RL: The model derived from the Reasoning Re-
finement stage, which is fine-tuned on high-quality PRM-filtered
trajectories but lacks subsequent reinforcement learning; (4)w/o
Refinement: A variant that skips the Reasoning Refinement stage
and directly applies Process-Supervised RL to the SFT policy.
The experimental results are summarized in Table 2. We analyze
the impact of each module as follows.
Effectiveness of Process-Supervised RL.Comparing ProRAG
with theGRPO Baselinein Table 2, we observe a significant and
consistent performance improvement across all datasets. This ad-
vantage is further corroborated by the learning curves presented

Zhao Wang, Ziliang Zhao, and Zhicheng Dou
Table 2: Ablation analysis of different framework components. w/o Refinement removes the PRM-guided warmup stage; w/o
RL relies solely on supervised fine-tuning on high-quality filtered data; GRPO Baseline utilizes only outcome rewards for
optimization. ProRAG (Full) integrates all components to achieve optimal performance.
MethodPopQA HotpotQA 2Wiki MuSiQue Bamboogle Avg.
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
ProRAG (Ours)47.2 51.641.4 52.8 46.0 51.1 23.5 34.1 45.6 56.4 40.7 49.2
w/o Refinement47.3 51.738.0 49.7 43.2 48.6 23.2 32.8 39.2 52.2 38.2 47.0
w/o RL 32.1 36.9 36.8 48.3 38.9 44.4 23.6 33.0 39.2 48.7 34.1 42.3
SFT Policy 27.9 32.7 28.8 38.8 29.0 33.3 23.3 32.5 36.0 46.8 29.0 36.8
GRPO Baseline 30.2 34.8 33.9 44.8 38.6 43.1 23.5 32.4 38.4 49.9 33.6 42.6
in Figure 4, where ProRAG not only exhibits a markedly faster con-
vergence rate but also achieves a substantially higher asymptotic
reward compared to the outcome-based baseline. These results
strongly validate that our proposed dual-granularity advantage
mechanism effectively addresses the credit assignment problem
in complex multi-step reasoning. By introducing dense, step-level
supervision, ProRAG enables the model to identify and reinforce
correct intermediate reasoning steps, thereby leading to more effi-
cient and stable optimization than approaches that rely solely on
sparse outcome signals.
Importance of Reasoning Refinement.The comparison be-
tween ProRAG and the varintw/o Refinementdemonstrates the
importance of the reasoning refinement stage. Although directly
applying RL to the SFT policy can lead to competitive results, the full
model achieves higher overall accuracy and stability. As shown in
the learning curves in Figure 4, thew/o Refinementvariant initiates
with a significantly lower reward and consistently trails behind the
full model throughout training, ultimately converging to a subopti-
mal solution. This indicates that directly exposing the SFT policy to
RL leads to a misalignment between the policy’s initial distribution
and the PRM’s preference landscape. The Refinement stage effec-
tively bridges this gap by warming up the policy on high-quality,
PRM-filtered trajectories, thereby mitigating the cold-start problem
and ensuring that the subsequent RL stage begins from a region of
higher reasoning validity.
Necessity of Online Exploration.Finally, when comparing
ProRAG with thew/o RLvariant, we observe that removing the
online RL stage leads to a sharp and consistent performance drop
across multiple benchmarks. While the PRM-guided Refinement ef-
fectively distills high-quality reasoning patterns from static data, it
remains a supervised process constrained by the distribution of pre-
collected samples. Consequently, the model lacks the capability to
generalize to novel scenarios. The subsequent Process-Supervised
RL stage addresses this limitation by enabling on-policy exploration,
allowing the model to actively interact with the environment and
discover higher-reward reasoning trajectories that extend beyond
the boundaries of the offline dataset.
/uni00000018/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000014/uni00000018/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013 /uni00000016/uni00000018/uni00000013 /uni00000017/uni00000013/uni00000013 /uni00000017/uni00000018/uni00000013 /uni00000018/uni00000013/uni00000013 /uni00000018/uni00000018/uni00000013 /uni00000019/uni00000013/uni00000013 /uni00000019/uni00000018/uni00000013 /uni0000001a/uni00000013/uni00000013 /uni0000001a/uni00000018/uni00000013 /uni0000001b/uni00000013/uni00000013 /uni0000001b/uni00000018/uni00000013
/uni00000037/uni00000055/uni00000044/uni0000004c/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000036/uni00000057/uni00000048/uni00000053/uni00000056/uni00000014/uni00000011/uni00000016/uni00000014/uni00000011/uni00000017/uni00000014/uni00000011/uni00000018/uni00000014/uni00000011/uni00000019/uni00000014/uni00000011/uni0000001a/uni00000032/uni00000058/uni00000057/uni00000046/uni00000052/uni00000050/uni00000048/uni00000003/uni00000035/uni00000048/uni0000005a/uni00000044/uni00000055/uni00000047
/uni00000033/uni00000055/uni00000052/uni00000035/uni00000024/uni0000002a/uni00000003/uni0000000b/uni00000029/uni00000058/uni0000004f/uni0000004f/uni0000000c
/uni0000005a/uni00000012/uni00000052/uni00000003/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048/uni00000050/uni00000048/uni00000051/uni00000057
/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni00000045/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048Figure 4: Training curves of ProRAG compared to the GRPO
baseline and the unrefined variant (w/o Refinement).
4.6 Impact of Retrieval
In this subsection, we investigate the performance of ProRAG under
different numbers of retrieved documents and retrieval steps to
evaluate its context utilization and planning adaptivity.
4.6.1 Impact of the Number of Retrieved Documents.As shown
in Figure 5, the performance gain from increasing the retrieval
context (𝑘) correlates strongly with task complexity. On the single-
hop PopQA dataset, the performance curve saturates early as 𝑘
increases, indicating that a single relevant document is typically
sufficient for factual recall. Notably, the model’s performance does
not decrease at 𝑘=5, suggesting strong robustness against noise
from irrelevant documents. In contrast, the complex multi-hop
benchmark MuSiQue demonstrates significant sensitivity to con-
text size, with the F1 score jumping by 6.2 points when 𝑘increases
to 3. This finding suggests that a restricted context window ( 𝑘=1)
creates an information bottleneck for reasoning. By providing mul-
tiple documents, ProRAG can effectively identify the key evidence,
which is crucial for resolving long-horizon dependencies.
4.6.2 Impact of the Number of Retrieval Steps.Figure 6 illustrates
the cumulative average F1 score and data coverage across retrieval
steps, highlighting the dynamic planning capability of ProRAG.
The data coverage curves reveal distinct adaptivity patterns, where
nearly 100% of PopQA queries are efficiently resolved within the

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
PopQA HotpotQA Mulsique010203040506070F1 Score47.5 47.1
27.951.652.8
34.152.955.1
35.7F1 Score Comparison across Datasets (k=1, 3, 5)
Retrieval Docs
k=1
k=3
k=5
Figure 5: Performance comparison (F1 Score) on PopQA, Hot-
potQA, and Musique datasets with different numbers of re-
trieved documents (𝑘∈{1,3,5}).
1 2 All
Retrieval Steps01020304050607080F1 Score51.5 51.6 51.649.152.7 52.8
29.936.734.1PopQA HotpotQA Mulsique
020406080100Data Coverage %
Performance vs. Sample Distribution
Figure 6: Cumulative Average F1 Score (bars) and Data Cov-
erage (dashed lines) across different retrieval steps ( ≤1,≤2,
All) on PopQA, HotpotQA, and Musique datasets.
first step, thereby minimizing unnecessary computational for sim-
ple tasks. Conversely, the model autonomously extends the reason-
ing chain for complex benchmarks such as HotpotQA and MuSiQue,
which often requires more than single step. Furthermore, while
longer reasoning chains in typical iterative frameworks often suffer
from performance degradation due to error propagation, ProRAG
maintains robust accuracy with cumulative F1 scores increasing
as retrieval steps grow. For instance, the F1 score on MuSiQue
increases from 29.9 to 34.1, demonstrating that process-level super-
vision helps guide the model search in complex spaces and stabilize
long-horizon reasoning.
4.7 Sensitivity Analysis
We investigate the sensitivity of model performance to the dual-
granularity weight 𝛽. As shown in Table 3, performance exhibits an
inverted-U trend, peaking at 𝛽=0.3. At𝛽=0.0, the model relies
solely on sparse outcome signals, struggling to identify the specific
reasoning steps that lead to the correct answer, which results in the
lowest performance. In contrast, introducing process supervision
(𝛽>0) significantly boosts performance, with 𝛽=0.3achieving the
optimal results (49.2% F1). This indicates that the dense feedback
from the PRM serves as a critical guide to stabilize the reasoningTable 3: Sensitivity analysis of 𝛽on the average performance
across all benchmarks. A balanced weight of 𝛽=0.3yields
the best results for both EM and F1.
MetricDual-Granularity Weight𝛽
0.0 0.10.30.5 0.7 0.9
EM 38.3 40.340.739.2 38.5 38.4
F1 score 46.7 48.349.247.7 47.1 46.8
trajectory and resolve the credit assignment problem. However, as
𝛽increases further ( 𝛽≥0.3), we observe a consistent decline. This
suggests that although the PRM provides dense signals, it inevitably
contains estimation errors. Consequently, an overweighted PRM
causes the model to overfit to noisy step-level signals rather than
the ground-truth objective, degrading the final reasoning accuracy.
4.8 Efficiency Analysis
In this subsection, we analyze ProRAG from two perspectives: data
efficiency in training and computational cost in inference.
4.8.1 Data Efficiency in Training.Standard outcome-based RL meth-
ods typically suffer from reward sparsity, requiring extensive train-
ing (e.g.,90k queriesfor Search-R1) to learn reasoning patterns.
In contrast, ProRAG achieves superior performance using only10k
queriesfor the RL stage, significantly reducing the required train-
ing data. Furthermore, compared to offline process-optimization
methods that rely on computationally expensive MCTS over the
entire training set to construct preference pairs (e.g.,5k queries
for ReasonRAG), ProRAG exhibits substantially higher annotation
and optimization efficiency. By performing MCTS on only728 seed
queries, we train a generalized PRM that serves as a dense and
reliable evaluator for online policy optimization, thereby avoiding
expensive tree-search construction.
4.8.2 Computational Cost in Inference.Recent approaches often
trade inference latency for performance improvement by employing
intensive test-time compute strategies, such as tree-based explo-
ration or iterative generation-evaluation-refinement loops. While
effective, these methods significantly increase the inference latency
and system complexity due to multiple model invocations and tra-
jectory regeneration. In contrast, ProRAG follows the principle of
policy internalization. By incorporating process-level feedback in
training, the model learns to reason and plan effectively. During
inference, ProRAG generates high-quality reasoning trajectories
in a single pass without external judges or iterative regeneration.
This design ensures that ProRAG preserves the inference efficiency
of standard LLMs, making it a more practical solution for latency-
sensitive deployment.
5 Conclusion
In this paper, we introduce ProRAG, a novel process-supervised
reinforcement learning framework designed to resolve the credit
assignment problem in multi-hop RAG tasks. Unlike traditional
approaches that rely on sparse outcome signals, static offline pref-
erences, or heuristic rules, ProRAG integrates learned, step-level

Zhao Wang, Ziliang Zhao, and Zhicheng Dou
supervision directly into the online optimization loop. This mecha-
nism effectively mitigates “process hallucinations,” where models ar-
rive at correct answers via flawed reasoning or redundant retrieval.
By leveraging Monte Carlo Tree Search to construct a high-quality
Process Reward Model and employing a dual-granularity advantage
mechanism, our framework provides precise feedback for every in-
termediate action, enabling the model to distinguish between valid
logical deductions and spurious correlations. Comprehensive exper-
iments across five diverse benchmarks demonstrate that ProRAG
achieves superior overall performance, particularly on complex
long-horizon reasoning tasks. Furthermore, our analysis highlights
that ProRAG achieves these gains with high data efficiency and pre-
serves low inference latency by internalizing reasoning capabilities
into the policy, making it efficient for real-world deployment. Fu-
ture work will explore extending this process-supervision paradigm
to more dynamic, open-ended environments to further advance the
reliability of autonomous systems.
References
[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511 [cs.CL] https://arxiv.org/abs/2310.11511
[2]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[3] Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin,
Xianpei Han, Le Sun, and Jie Zhou. 2025. DeepRAG: Thinking to Retrieve Step
by Step for Large Language Models. arXiv:2502.01142 [cs.AI] https://arxiv.org/
abs/2502.01142
[4] Daya Guo, Dejian Yang, Haowei Zhang, et al .2025. DeepSeek-R1 incentivizes
reasoning in LLMs through reinforcement learning.Nature645, 8081 (Sept. 2025),
633–638. doi:10.1038/s41586-025-09422-z
[5]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei
Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training.
arXiv:2002.08909 [cs.CL] https://arxiv.org/abs/2002.08909
[6]Jie He, Victor Gutiérrez-Basulto, and Jeff Z. Pan. 2025. From Sufficiency to
Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented
Reasoning for LLMs. arXiv:2507.22716 [cs.CL] https://arxiv.org/abs/2507.22716
[7] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. arXiv:2011.01060 [cs.CL] https://arxiv.org/abs/2011.01060
[8] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. arXiv:2305.06983 [cs.CL] https://arxiv.org/abs/2305.
06983
[9]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL]
https://arxiv.org/abs/2503.09516
[10] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. arXiv:2004.04906 [cs.CL] https://arxiv.org/abs/
2004.04906
[11] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. arXiv:2004.04906 [cs.CL] https://arxiv.org/abs/
2004.04906
[12] Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong, Bojian Xiong, Yurong Zhang,
Yan Gao, Yi Wu, Yao Hu, and Deyi Xiong. 2025. DecEx-RAG: Boosting Agentic
Retrieval-Augmented Generation with Decision and Execution Optimization via
Process Supervision. arXiv:2510.05691 [cs.CL] https://arxiv.org/abs/2510.05691
[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[14] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models. arXiv:2501.05366 [cs.AI] https://arxiv.org/abs/2501.05366[15] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Han-
naneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating Effec-
tiveness of Parametric and Non-Parametric Memories. arXiv:2212.10511 [cs.CL]
https://arxiv.org/abs/2212.10511
[16] OpenAI, Josh Achiam, Steven Adler, et al .2024. GPT-4 Technical Report.
arXiv:2303.08774 [cs.CL] https://arxiv.org/abs/2303.08774
[17] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. arXiv:2210.03350 [cs.CL] https://arxiv.org/abs/2210.03350
[18] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs.LG] https:
//arxiv.org/abs/1707.06347
[19] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing Retrieval-Augmented Large Language Models with
Iterative Retrieval-Generation Synergy. arXiv:2305.15294 [cs.CL] https://arxiv.
org/abs/2305.15294
[20] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeek-
Math: Pushing the Limits of Mathematical Reasoning in Open Language Models.
arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300
[21] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin
Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search
Capability in LLMs via Reinforcement Learning. arXiv:2503.05592 [cs.AI] https:
//arxiv.org/abs/2503.05592
[22] Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu,
Xiao Zhang, Song Yang, and Han Li. 2025. ReARTeR: Retrieval-Augmented
Reasoning with Trustworthy Process Rewarding. arXiv:2501.07861 [cs.CL] https:
//arxiv.org/abs/2501.07861
[23] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573
[24] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. arXiv:2212.10509 [cs.CL] https://arxiv.org/abs/
2212.10509
[25] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tris-
tan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gal-
louédec. 2020. TRL: Transformer Reinforcement Learning. https://github.com/
huggingface/trl.
[26] Junlin Wang, Zehao Wu, Shaowei Lu, Yanlan Li, and Xinghao Huang. 2025.
SIRAG: Towards Stable and Interpretable RAG with A Process-Supervised Multi-
Agent Framework. arXiv:2509.18167 [cs.CL] https://arxiv.org/abs/2509.18167
[27] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised
Contrastive Pre-training.arXiv preprint arXiv:2212.03533(2022).
[28] Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang,
and Yichao Wu. 2025. StepSearch: Igniting LLMs Search Ability via Step-Wise
Proximal Policy Optimization. arXiv:2505.15107 [cs.CL] https://arxiv.org/abs/
2505.15107
[29] Tongyu Wen, Guanting Dong, and Zhicheng Dou. 2026. SmartSearch: Process
Reward-Guided Query Refinement for Search Agents. arXiv:2601.04888 [cs.AI]
https://arxiv.org/abs/2601.04888
[30] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement De-
langue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,
Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest,
and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language
Processing. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations. Association for Computational Lin-
guistics, Online, 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6
[31] Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu He, Xinya Du, and
Zhiyu Chen. 2025. HiPRAG: Hierarchical Process Rewards for Efficient Agentic
Retrieval Augmented Generation. arXiv:2510.07794 [cs.CL] https://arxiv.org/
abs/2510.07794
[32] Peiran Xu, Zhuohao Li, Xiaoying Xing, Guannan Zhang, Debiao Li, and Kunyu
Shi. 2025. Hybrid Reward Normalization for Process-supervised Non-verifiable
Agentic Tasks. arXiv:2509.25598 [cs.AI] https://arxiv.org/abs/2509.25598
[33] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou,
Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao
Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su,
Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui,
Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.
arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388

ProRAG: Process-Supervised Reinforcement Learning
for Retrieval-Augmented Generation
[34] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL]
https://arxiv.org/abs/1809.09600
[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] https://arxiv.org/abs/2210.03629
[36] Tianhua Zhang, Kun Li, Junan Li, Yunxiang Li, Hongyin Luo, Xixin Wu, James
Glass, and Helen Meng. 2026. TreePS-RAG: Tree-based Process Supervision
for Reinforcement Learning in Agentic RAG. arXiv:2601.06922 [cs.CL] https://arxiv.org/abs/2601.06922
[37] Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng
Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, and
Xiangyu Zhao. 2025. Process vs. Outcome Reward: Which is Better for Agentic
RAG Reinforcement Learning. arXiv:2505.14069 [cs.IR] https://arxiv.org/abs/
2505.14069
[38] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A survey.arXiv preprint
arXiv:2402.19473(2024).