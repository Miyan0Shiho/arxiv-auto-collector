# InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning

**Authors**: Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi

**Published**: 2025-04-17 15:41:39

**PDF URL**: [http://arxiv.org/pdf/2504.13032v1](http://arxiv.org/pdf/2504.13032v1)

## Abstract
Recent advancements in large language models (LLMs) have enabled their use as
agents for planning complex tasks. Existing methods typically rely on a
thought-action-observation (TAO) process to enhance LLM performance, but these
approaches are often constrained by the LLMs' limited knowledge of complex
tasks. Retrieval-augmented generation (RAG) offers new opportunities by
leveraging external databases to ground generation in retrieved information. In
this paper, we identify two key challenges (enlargability and transferability)
in applying RAG to task planning. We propose InstructRAG, a novel solution
within a multi-agent meta-reinforcement learning framework, to address these
challenges. InstructRAG includes a graph to organize past instruction paths
(sequences of correct actions), an RL-Agent with Reinforcement Learning to
expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to
improve task generalization for transferability. The two agents are trained
end-to-end to optimize overall planning performance. Our experiments on four
widely used task planning datasets demonstrate that InstructRAG significantly
enhances performance and adapts efficiently to new tasks, achieving up to a
19.2% improvement over the best existing approach.

## Full Text


<!-- PDF content starts -->

InstructRAG: Leveraging Retrieval-Augmented Generation on
Instruction Graphs for LLM-Based Task Planning
Zheng Wang∗
Huawei Singapore Research Center
Singapore
wangzheng155@huawei.comShu Xian Teo∗
Huawei Singapore Research Center
Singapore
teo.shu.xian@huawei.com
Jun Jie Chew
Huawei Singapore Research Center
Singapore
chew.jun.jie@huawei.comWei Shi
Huawei Singapore Research Center
Singapore
w.shi@huawei.com
Abstract
Recent advancements in large language models (LLMs) have en-
abled their use as agents for planning complex tasks. Existing
methods typically rely on a thought-action-observation (TAO) pro-
cess to enhance LLM performance, but these approaches are often
constrained by the LLMs’ limited knowledge of complex tasks.
Retrieval-augmented generation (RAG) offers new opportunities
by leveraging external databases to ground generation in retrieved
information. In this paper, we identify two key challenges (enlarga-
bility and transferability) in applying RAG to task planning. We
propose InstructRAG , a novel solution within a multi-agent meta-
reinforcement learning framework, to address these challenges.
InstructRAG includes a graph to organize past instruction paths
(sequences of correct actions), an RL-Agent with Reinforcement
Learning to expand graph coverage for enlargability, and an ML-
Agent with Meta-Learning to improve task generalization for trans-
ferability. The two agents are trained end-to-end to optimize overall
planning performance. Our experiments on four widely used task
planning datasets demonstrate that InstructRAG significantly en-
hances performance and adapts efficiently to new tasks, achieving
up to a 19.2% improvement over the best existing approach.
CCS Concepts
•Information systems →Information retrieval .
Keywords
large language model, retrieval-augmented generation, agent plan-
ning
ACM Reference Format:
Zheng Wang, Shu Xian Teo, Jun Jie Chew, and Wei Shi. 2025. InstructRAG:
Leveraging Retrieval-Augmented Generation on Instruction Graphs for
LLM-Based Task Planning. In Proceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval (SIGIR
∗Equal Contribution.
Please use nonacm option or ACM Engage class to enable CC licenses
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’25, July 13–18, 2025, Padua, Italy
©2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1592-1/2025/07
https://doi.org/10.1145/3726302.3730009’25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, 16 pages.
https://doi.org/10.1145/3726302.3730009
1 INTRODUCTION
With the significant advancement of large language models (LLMs),
a recent trend has emerged in employing LLMs as intelligent agents
to tackle diverse real-world planning tasks. These tasks include
multi-hop reasoning [ 38], embodied tasks [ 24,25,34], web shop-
ping [ 39], and scientific reasoning [ 29], etc. Many recent solutions
to the planning problem, such as ReAct [ 41], KnowAgent [ 42],
WKM [ 19], Reflexion [ 23], FireAct [ 3], NAT [ 30], and ETO [ 27],
follow a thought-action-observation (TAO) process. In the thought
phase, the LLM leverages its reasoning ability to create a plan by
breaking down a task into a series of subtasks. In the action phase,
the LLM determines the specific actions required, such as selecting
which tool to use. In the observation phase, it captures the results
of executing the action and provides feedback from the external
environment to the LLM, facilitating the planning of subsequent
TAO steps. Within this process, the thought and action are gen-
erated by the LLM, while the observation is implemented by the
environment. Existing solutions adopt diverse strategies, including
prompting [ 23,41] or fine-tuning [ 3,19,27,30,42], to improve
LLM-generated thoughts and actions for more effective planning.
In particular, KnowAgent [ 42] integrates pre-defined rules into
prompts to ensure that generated thoughts exhibit logical action
transitions. For example, it prevents looking up an entity without
first performing a search operation on the topic, as seen in Hot-
PotQA [ 38]. Reflexion [ 23] incorporates self-reflection summaries
into the TAO process to guide subsequent trials. WKM [ 19] trains a
world knowledge model to generate thoughts based on knowledge
acquired from human task-solving experiences.
While these existing methods aim to enhance LLM planning,
they are often constrained by the inherent limitations of the LLMs
themselves, such as their limited knowledge on complex tasks. The
rapid development of retrieval-augmented generation (RAG) pro-
vides new opportunities to address these limitations by leveraging
external databases. By anchoring LLM generation in retrieved in-
formation, RAG improves performance through the integration
of relevant data during the planning process. In this context, we
recognize that task-specific nature of information retrieval plays a
crucial role for effective planning generation. For example, consider
the question “Were Scott Derrickson and Ed Wood of the samearXiv:2504.13032v1  [cs.AI]  17 Apr 2025

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
nationality?” from HotPotQA, as shown in Figure 1(a). A poten-
tial retrieved plan for this question involves a sequence of actions
(referred to as instruction paths in this paper): first, use Google
Search to find information about Scott Derrickson (denoted by
Search[Scott Derrickson] ), then look up a sentence containing
the keyword “nationality” ( Lookup[nationality] ), followed by
Search[Ed Wood] andLookup[nationality] . These instructions
are specific to the question at hand and may vary depending on
the topics or entities involved. A similar phenomenon can also be
observed in ALFWorld embodied tasks, where instructions might
include Goto[shelf 6] , then Take[vase 2 from shelf 6] .
In this paper, we discuss the task-specific nature in two aspects,
with the goal of bridging the gap between task-specific questions
and instructions derived from past experiences stored in a database
through RAG. (1) Enlargeability : This refers to a task where the
question falls within the scope of those covered by the external
database. Specifically, we pre-store successful instruction paths in
the database, with each path tailored to a specific task. To address
questions related to these tasks, we explore a paradigm for combin-
ing instructions to expand the database’s coverage. As illustrated
in Figure 1(a), there are two successful instruction paths, 𝑃1
1and𝑃1
2,
for solving questions 𝑄1
1and𝑄1
2, respectively. These paths consist
of five instructions: searching for (a) Scott Derrickson, (b) Ed Wood,
and (c) Christopher Nolan, and looking up entities related to (d)
nationality and (e) birthplace. By combining these instructions in
sequences such as(𝑎)→(𝑒)→(𝑐)→(𝑒), we can generate a new
instruction path for solving a novel question (i.e., ˆ𝑄) that was not
covered by the original paths (but shares the same task of querying
a location). The enlargeability is proposed to enhance the database’s
ability to address a wider range of questions. (2) Transferability :
This refers to a task where the question is outside the scope of tasks
covered by the external database. We note that LLM-based task
planning is provided as a capability to support a wide range of
tasks in practice. Transferability is essential for bridging the gaps
between different tasks (i.e., those in the pre-built database and the
questions at hand) within the RAG system. To achieve this, it is
necessary to expand the database to incorporate new instructions
required for different tasks, such as updating it with instructions
relevant to the new tasks based on a development set. Additionally,
certain trainable modules associated with the RAG can be rapidly
adapted to accommodate the new tasks.
New Solution. Although recent research efforts [ 11,12,22] have
attempted to apply RAG techniques to task planning, these methods
often fall short in several aspects: i) [ 12] is primarily tailored for
specific domains, such as decision-making in video games, making
it challenging to generalize their designs to broader planning tasks
as studied in this paper. ii) [ 22] focuses on multi-hop reasoning
via search engines (e.g., using Google Search to access Wikipedia
knowledge), but their effectiveness in tasks where the search en-
gines are inapplicable (e.g., embodied tasks or web shopping) re-
mains unexplored. iii) [ 11] simply relies on storing past experiences
and retrieving similar ones using AKNN, without identifying key
aspects such as enlargability and transferability. This gap results in
suboptimal performance, as evidenced by our experiments.
To this end, we propose InstructRAG , a new solution based
on a multi-agent meta-reinforcement learning framework. For (1) ,we design an instruction graph to instantiate the database. In this
graph, nodes and edges represent two sets: nodes contain similar
instructions, while edges represent corresponding tasks, all derived
from successful instruction paths in past experiences. The rationale
behind this approach is two-fold: 1) The graph provides a natural
structure for organizing paths and facilitates the integration of new
paths by clustering similar instructions related to various tasks.
2) Each node acts as a junction that enables the creation of new
paths by combining stored instructions within it, and each edge
records the tasks (with associated questions) along the path. This
organization allows us to structure past experiences effectively
within the database. Further, we design an RL-Agent that utilizes
Reinforcement Learning to identify candidate paths on the graph,
with the goal of optimizing the database’s coverage to enhance its
enlargeability. For (2) , we explore a meta-learning approach into
the RAG pipeline. Specifically, we introduce an additional agent, re-
ferred to as the ML-Agent, which Meta-Learns to select a path from
the candidate paths provided by the RL-Agent. This selected path
is then used as an in-context learning exemplar within the prompt,
aiming to enhance the LLM’s generalization to new tasks by updat-
ing it with only a few QA pairs during the meta-update phase. Here,
the two agents collaborate within the TAO process [ 41] to facilitate
task planning via grounding the generation of thoughts and actions
by LLMs. We note that the RL-Agent generates candidate paths
for the ML-Agent to select, and the ML-Agent then assesses the
end-to-end effectiveness of the selected path, to incorporate this
feedback as the reward for the RL-Agent. This interaction creates a
positive loop, leading to improved planning performance.
To summarize, we make the following contributions.
•We conduct a systematic study of leveraging RAG for LLM-based
task planning, and identify two key properties (i.e., enlargability
and transferability) that a potential technique should possess. To
our best knowledge, this is the first attempt of its kind.
•We propose a new solution called InstructRAG , which includes
three key components: an instruction graph, an RL-Agent, and an
ML-Agent. These components are integrated into a multi-agent
meta-reinforcement learning framework that explicitly trains to
optimize end-to-end task planning performance.
•We conduct extensive experiments on four widely used task plan-
ning datasets: HotpotQA [ 38], ALFWorld [ 25], Webshop [ 39], and
ScienceWorld [ 29], across three typical LLMs. Our InstructRAG
can be integrated with both trainable LLMs (e.g., GLM-4 [ 9]) for
fine-tuning and frozen LLMs (e.g., GPT-4o mini [ 1] and DeepSeek-
V2 [7]). The results demonstrate that InstructRAG improves per-
formance by approximately 19.2%, 9.3%, 6.1%, and 10.2% over
the best baseline method on the four datasets, respectively. In
addition, InstructRAG adapts rapidly to new tasks, achieving
effective performance with few-shot learning.
2 RELATED WORK
LLM-based Agent Planning. To solve complex tasks, humans
typically decompose them into smaller sub-tasks and then evaluate
the plan’s effectiveness. Similarly, LLM-based agents follow this
routine, and we categorize existing techniques based on whether
the agent receives feedback during the planning process. A detailed
survey of LLM-based agent planning can be found in [ 28,37].In

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
planning without feedback , agents do not receive feedback that in-
fluences their future actions. The main techniques for this category
include (1) single-path reasoning [ 21,35], (2) multi-path reason-
ing [2,31,40], and (3) using an external planner [ 5,13]. Specifically,
for (1), CoT [ 35] illustrates the reasoning steps for LLMs to tackle
complex tasks using prompts, thereby guiding LLMs to plan and
execute actions step-by-step. For (2), CoT-SC [ 31] explores diverse
reasoning paths to solve complex tasks. Initially, it utilizes CoT to
generate multiple reasoning paths and their respective answers.
Subsequently, it selects the answer with the highest frequency as
the final output. For (3), external planners are designed to generate
plans for specific domains. For example, LLM+P [ 13] focuses on
robot planning tasks by defining them using formal Planning Do-
main Definition Languages (PDDL). It utilizes an external planner,
such as the Fast Downward planner [ 10], which employs heuristic
search to handle PDDL formulations. The results generated by the
planner are then translated back into natural language by LLMs.
In planning with feedback , effectiveness is generally improved
by receiving feedback after actions are taken, which supports long-
horizon planning. This feedback can come from (1) environments [ 3,
41], (2) humans [ 19,42], and (3) models [ 16,23]. For (1), ReAct [ 41]
proposes the TAO process, where a language model generates the
thought for planning, the action involves interacting with the envi-
ronment, and the observation consists of external feedback (such
as search engine results) based on the action. FireAct [ 3] gener-
ates the TAO using various methods, which are then converted
into the ReAct format to fine-tune a small language model. For
(2), KnowAgent [ 42] integrates action knowledge, which includes
rules determining action transitions, into prompts to enhance the
planning capabilities of LLMs. This knowledge is derived from both
human input and GPT-4 [ 1]. Further, WKM [ 19] is introduced to fa-
cilitate agent planning using a world knowledge model. This model
is trained by comparing selected trajectories (annotated by humans)
with rejected trajectories (explored by an experienced agent). For
(3), Reflexion [ 23] employs verbal feedback to enhance the agent’s
planning based on previous failures. It transforms binary or scalar
feedback from self-evaluation into a textual summary, which is then
added as additional context for the agent in subsequent planning. In
this paper, we explore a new RAG-based approach to task planning,
emphasizing two key properties: enlargability and transferability
in technique development.
Retrieval-Augmented Generation. RAG enhances LLM gener-
ation by querying an external database to obtain relevant infor-
mation, which grounds the subsequent text generation. Recent
studies utilize RAG for task planning [ 11,12,22,33]. Specifically,
RAT [ 33] enhances CoT by iteratively revising each thought step
with retrieved information relevant to the task query, thereby im-
proving LLMs’ ability to reason over long-horizon generation tasks.
RAP [ 11] stores past experiences, including context and action-
observation trajectories, and retrieves them based on their sim-
ilarity to the current situation. The goal is to facilitate deriving
appropriate actions by leveraging memory examples from similar
tasks. PlanRAG [ 12] is designed for decision QA tasks, following
a plan-then-retrieval approach. The LLM first generates a plan to
guide the analysis, then retrieves information from an external
database by formulating queries. It also continuously evaluatesthe need for re-planning during the process. GenGround [ 22] ex-
plores a generate-then-ground approach for multi-hop reasoning
tasks. It breaks down a complex question into sub-questions, gen-
erates an immediate answer for each, then revises it with retrieved
information. This revised answer informs the next sub-question,
iterating until the final answer is achieved. In this paper, we propose
InstructRAG within a multi-agent meta-reinforcement learning
framework to systematically address the gap in leveraging RAG for
task-specific questions and stored past experiences.
Meta-learning for Improving LLMs via In-context Learning
(ICL). To enhance the transferability of LLMs to unseen tasks,
meta-learning approaches [ 4,6,18,26] have been developed. These
approaches fine-tune pre-trained LLMs using a diverse set of tasks,
formatted as ICL instances by pre-appending task-specific exem-
plars to the prompts during training. These methods follow Model-
Agnostic Meta-Learning (MAML) principles [ 8]. Specifically, MAML-
en-LLM [ 26] explores a wide parameter space to learn truly gener-
alizable parameters that perform well on disjoint tasks and adapt
effectively to unseen tasks. MTIL [ 6] investigates the application
of meta-learning to multi-task instructional learning [ 32], aiming
to enhance generalization to unseen tasks in a zero-shot setting.
MetaICL [ 18] adapts a LLM to perform in-context learning across a
broad range of training tasks. It aims to improve the model’s ability
to learn new tasks in context during testing, by conditioning on
a few training examples without requiring parameter updates or
task-specific templates. In this paper, we propose a novel meta-
reinforcement learning framework to improve transferability, with
two cooperative agents tailored for planning tasks. This approach
is distinctly different from existing methodologies in the field.
3 PROBLEM STATEMENT
We explore the problem of LLM-based task planning through RAG,
grounded in an external database (i.e., instruction graph). In this
context, we identify two practical properties that should be met:
-Enlargeability: It should expand the scope of the instruction
graph by traversing existing instructions (nodes) on the graph
and combining them into new sequences of instructions (paths).
This will help the LLM in completing tasks that do not have
pre-defined paths during the graph’s construction.
-Transferability: Task planning as a capability in practice involves
developing techniques that achieve rapid adaptation to new tasks.
For example, the trained model should be able to quickly learn a
new task from a small amount of new data.
4 METHODOLOGY
4.1 Overview of InstructRAG
The proposed InstructRAG tackles the challenges of LLM-based
task planning by focusing on two key properties: enlargeability and
transferability. It comprises several components: instruction graph
construction (Section 4.2), RL-Agent (Section 4.3), and ML-Agent
(Section 4.4). These components are integrated into a multi-agent
meta-reinforcement learning framework, which is detailed in terms
of three stages: training, few-shot learning, and testing (Section 4.5).
Training . We illustrate the overall framework in Figure 1. Specif-
ically, the training tasks (seen tasks) are divided into a support

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
Fill Into Prompt      T raversed Paths 
    : Were SD and CN of the
same birthplace?
Top-1 Path
Refer the path to plan:Planning Paths: Instruction Definition:
(1) Search[topic] : Use Google Search…  
(2) Lookup[entity] : Return sentence … Task      (Location) - Support Set
     : Were SD and EW of the
same nationality ?
     : Were EW and CN of the
same birthplace?Task      (Date) - Support Set
     : What year was the country
where SD was born founded?     : What year was the country
where SD was born founded?
Instruction Path𝟛
𝟜 𝟝
Instruction Graph
Graph T raversal RL-AgentPath EncoderQuestion Encoder QPA 
QPM  
LLMPr ompt  
Task Description:
Demonstrations:
Here are some examples:{examples}  Your task is to answer a question:
...
(a) Instruction Graph (b) RL-Agent for Enlargeability (c) ML-Agent for T ransferabilityInstruction PathML-Agent      T raversed Instructions
Task      (Location) - Query Set
Frozen
RewardTAO 
Trainable 𝟚SD=Scott Derrickson  EW=Ed Wood  CN=Christopher Nolan  
Figure 1: Architecture of the proposed InstructRAG with multi-agent meta-reinforcement learning, illustrated on HotpotQA.
set and a query set. For support set , it is used to construct the in-
struction graph by extracting instruction paths from questions
and forming the graph based on these paths. Additionally, the
paths and corresponding questions help warm-start the RL-Agent,
and pre-train the ML-Agent with two pre-training tasks: Ques-
tion Path Alignment (QPA) and Question Path Matching (QPM).
For query set , it is used to query the graph and trains the RL-Agent
and the ML-Agent within a multi-agent framework. The RL-Agent
finds candidate paths through graph traversal, which is modeled as
a Markov Decision Process (MDP) and optimized using reinforce-
ment learning. The RL-Agent is trained to handle questions not
seen during the graph construction, enlarging the capability of the
instruction graph by combining instructions into the paths to ad-
dress these questions. The ML-Agent then selects the most relevant
path among the candidate paths based on their representations,
which is used to form the prompt for a LLM to predict the final
answer, following the TAO process. We note that the transferability
is considered through an in-context learning manner, where the
LLM learns a new task by conditioning on the task-specific path
in the prompt. The ML-Agent optimizes this process, either with a
trainable or frozen LLM, via meta-learning.
Few-shot Learning and Testing . Once the parameters of the
RL-Agent and ML-Agent are meta-trained, we rapidly adapt the
model parameters using few-shot examples from the support set
on testing tasks (unseen tasks). The testing is then conducted based
on the query set of these testing tasks.
4.2 Instruction Graph
Instruction. An instruction 𝐼represents a specific action per-
formed by LLMs, e.g., Search[Scott Derrickson] is an instruc-
tion, meaning to use Google Search to find relevant information
about Scott Derrickson as shown in Figure 1(a).
Instruction Path. An instruction path 𝑃𝑗
𝑖=⟨𝐼1,𝐼2,...,𝐼|𝑃|⟩is rep-
resented as a sequence of instructions that LLMs follow step-by-step
to perform actions and complete the 𝑖-th question of the 𝑗-th task,e.g.,𝑃2
1:Search[Scott Derrickson] →Lookup[nationality]
→Lookup[year of founding] denotes an instruction path to
address the question 𝑄2
1of the task𝑇2as shown in Figure 1(a).
Instruction Graph. An instruction graph 𝐺(V,E)is represented
as a directed graph that organizes instruction paths of questions
belonging to various tasks, where VandErepresent the nodes
and edges of the graph, respectively. Each node I∈Vdenotes an
instruction set, i.e., I={𝐼1,𝐼2,...,𝐼|I|}, clustering similar instruc-
tions. Each edge T∈Edenotes a task set, i.e., T={𝑇1,𝑇2,...,𝑇|T|},
recording the tasks with associated questions involved on the path.
The Graph Construction and Insights. We present the graph
construction in two steps, illustrated with a running example in
Figure 1(a). The detailed process is outlined in Algorithm 1.
Step-1 (Generating Instruction Paths) : We divide the dataset into
two parts: the support set and the query set, following the meta-
learning setup. The support set is used for graph construction, while
the query set is used to query the graph to train enlargeability and
transferability, to be discussed in Section 4.3 and Section 4.4, re-
spectively. For each question in the support set, we generate its
instruction path using existing task planning techniques [ 23,41,42].
We select the path that correctly plans the question for construc-
tion, ensuring the planning is grounded in the prepared database,
aligning with the goal of RAG.
Step-2 (Inserting Instructions with a Threshold 𝛿): Then, we it-
eratively insert each instruction in the generated paths, i.e., 𝑃1
1,𝑃1
2,𝑃2
1,
into the graph 𝐺. The first two instructions in 𝑃1
1are initialized
to create two node sets, i.e., I1←Search[Scott Derrickson]
andI2←Lookup[nationality] , which correspond to an edge
setT1recording the involved task {𝑇1}. Here, we note that ad-
jacent instructions are not inserted into the same node, as this
would break the transition between them. To insert the next in-
struction Search[Ed Wood] , we perform an AKNN search [ 17]
on all instructions excluding the instructions in its adjacent node
(i.e., Lookup[nationality] ∈I2), identifying the most similar in-
struction Search[Scott Derrickson] , which is associated with a

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
similarity value (e.g., cosine similarity) 𝜓in the node set I1. Then,
we define a threshold 𝛿to control the insertion. If 𝜓<𝛿, a new
node set I3is created and the instruction is inserted into this new
node, i.e., I3←Search[Ed Wood] ; otherwise, the instruction is
inserted into the identified node I1. The process continues until
all instructions are inserted. Additionally, we note that when the
instruction Lookup[nationality] of𝑃2
1is inserted into I2(with a
cosine similarity of 1.0), the task 𝑇2is also added to the edge set T1,
resulting in{𝑇1,𝑇2}.
We present two key insights into graph construction: (1) Graphs
naturally organize instruction paths, where nodes and edges are
represented as sets to enable flexible integration of similar instruc-
tions across tasks. (2) The threshold controls instruction similarity,
forming junction nodes that create new paths beyond those in the
original data. For instance, merging Lookup[nationality] and
Lookup[birthplace] intoI2enables novel paths like I1→I2→
I4→I2, thus improving graph expandability to cover more ques-
tions (e.g., ˆ𝑄in Figure 1(a)).
Algorithm 1: The Instruction Graph Construction
Require : a support set S; a threshold 𝛿
1𝐼𝐶←3,𝑇𝐶←2// two counters for node and edge sets
2foreach𝑇𝑗∈S(1≤𝑗≤|S|)do
3 foreach𝑄𝑗
𝑖∈𝑇𝑗(1≤𝑖≤|𝑇𝑗|)do
4 obtain a correct 𝑃𝑗
𝑖=⟨𝐼1,𝐼2,...,𝐼|𝑃𝑗
𝑖|⟩for𝑄𝑗
𝑖
5 I′←∅ // record the last node set
6 for𝑘=1,2,...,|𝑃𝑗
𝑖|do
7 if𝑖=1and𝑗=1and𝑘<3then
8 I1.add(𝐼1),I2.add(𝐼2),T1←Edge(I1,I2)
9 T1.add(𝑇1),𝐺.addEdge(T1),I′←I2
10 continue
11 recall I𝑠and𝜓for𝐼𝑘with AKNN on 𝐺.V−I′
12 if𝜓<𝛿then
13 I𝐼𝐶.add(𝐼𝑘),T𝑇𝐶←Edge(I′,I𝐼𝐶)
14 T𝑇𝐶.add(𝑇𝑗),𝐺.addEdge(T𝑇𝐶)
15 I′←I𝐼𝐶,𝐼𝐶←𝐼𝐶+1,𝑇𝐶←𝑇𝐶+1
16 else
17 I𝑠.add(𝐼𝑘)
18 ifEdge(I′,I𝑠)∈𝐺then
19 obtain the edge of (I′,I𝑠)denoted by T′
20 T′.add(𝑇𝑗)
21 else
22 T𝑇𝐶←Edge(I′,I𝑠),T𝑇𝐶.add(𝑇𝑗)
23 𝐺.addEdge(T𝑇𝐶),𝑇𝐶←𝑇𝐶+1
24 I′←I𝑠
25Return the instruction graph 𝐺
4.3 RL-Agent: Retrieving Instruction Paths on
Instruction Graph
Given an instruction graph 𝐺, we explore its enlargeability through
graph traversal to retrieve various instruction paths that solve ques-
tions denoted by ˆ𝑄not present during construction (i.e., questions
in the query set). To achieve this, we train an agent for traversal,
which examines each path in the graph, e.g., via depth-first search(DFS). For each node, the agent decides whether to include or ex-
clude the node (i.e., actions) in the path based on the instructions
contained in the node and the tasks connected by its edges (i.e.,
states). A high-quality retrieved path benefits subsequent plan-
ning, reflected by an end-to-end metric such as F1 scores on Hot-
PotQA [ 38] (i.e., rewards), which can then inform instruction selec-
tion. This process forms a Markov Decision Process (MDP), and we
employ Reinforcement Learning (RL) to optimize it.
Constructing Decision Environment. The instruction graph 𝐺
typically contains numerous instruction paths, formed by combin-
ing different instructions at each node. To manage this, we limit
the RL-Agent’s retrieval to 𝐾relevant instruction paths, denoted
asˆ𝑃1,ˆ𝑃2,..., ˆ𝑃𝐾, which are then utilized for planning in the next
phase by the ML-Agent (to be introduced in Section 4.4), where the
𝐾is a hyperparameter that can be tuned for optimal performance.
We first perform an AKNN search on all instructions for a query
ˆ𝑄. The agent’s traversal starts from the most similar instructions
(corresponding to the nodes) using DFS. Once the agent excludes
a node and backtracks to another branch, an instruction path is
formed. This process continues until 𝐾paths are retrieved.
States. Suppose we have an input question ˆ𝑄and visit a node I(a
set of instructions), along with its in-degree edge T(a set of tasks).
We define the state susing three cosine similarities 𝐶𝑆(·,·), that is
s={max
𝐼𝑖∈I𝐶𝑆(vˆ𝑄,v𝐼𝑖),max
𝑇𝑗∈T𝐶𝑆(vˆ𝑄,v𝑇𝑗),max
𝑄𝑐
𝑘∈𝑇𝑐𝐶𝑆(vˆ𝑄,v𝑄𝑐
𝑘)},
v𝑇𝑗=1
|𝑇𝑗||𝑇𝑗|∑︁
𝑘=1v𝑄𝑗
𝑘and𝑐=arg max
𝑇𝑐∈T𝐶𝑆(vˆ𝑄,v𝑇𝑐),(1)
where v·denotes an embedding vector. We construct the state by (1)
examining the most similar instruction in the node, (2) identifying
the most similar task, denoted by 𝑇𝑐, in the edge (whose embedding
is calculated as the average of the question embeddings belonging
to the task), and (3) finding the most similar question within 𝑇𝑐.
Actions. Let𝑎denote an action, which has two choices during the
graph traversal: including the visited node by selecting the most
similar instruction into the path ˆ𝑃𝑖(1≤𝑖≤𝐾) and searching
its connected nodes, or excluding the node and backtracking the
search from another branch, then an instruction path is formed.
The action𝑎is formally defined as:
𝑎=1(including) or 0(excluding). (2)
Considering the consequence of performing an action, it transitions
the environment to the next state s′, affecting which node or edge is
used for constructing the state. Notably, some predefined rules may
be further incorporated to constrain the action space (e.g., a rule of
avoiding Lookup information without first performing Search on
HotpotQA [42]), which benefits more accurate path selection.
Rewards. Let𝑟denote a reward, which corresponds to the end-to-
end feedback of an instruction path that contributes to the generated
answer ˆ𝐴by a LLM for ˆ𝑄. Specifically, when an instruction path
from the𝐾paths is selected and written into the prompt by the
ML-Agent, the LLM generates an answer ˆ𝐴. This answer can be
evaluated using a specific metric Δ(·,·)(e.g., F1 score), defined as:

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
𝑟=Δ(ˆ𝐴,𝐴), (3)
where𝐴denotes the ground truth answer. The rationale for design-
ing the reward is to enable joint optimization for the two agents in
a multi-agent setup, where the RL-Agent provides paths for the ML-
Agent to write into the prompt, and the feedback from the prompt
affects the path retrieval by the RL-Agent. Therefore, the two agents
can be jointly optimized to improve the overall performance.
Policy Learning. We involve two phases for training the MDP
policy: warm-start (WS) and policy gradient (PG). In WS , the goal
is to equip the agent with the basic ability to include or exclude
instructions. To achieve this, we randomly sample questions from
the support set. For each question, we randomly sample nodes on
𝐺and construct its state using Equation 1. If the node is on the
instruction path for the question, the state is associated with an
action labeled 1; otherwise, it is labeled 0. We collect these state-
action pairs and train the RL-Agent using binary cross-entropy:
LWS=−𝑦∗log(𝑃)+(𝑦−1)∗log(1−𝑃), (4)
where𝑦denotes the label, and 𝑃is the predicted probability of
the positive class. In PG , the primary goal is to develop a policy
𝜋𝜃(𝑎|s)that guides the agent in performing actions 𝑎based on
the given states sfor questions on the query set, with the aim of
maximizing the cumulative reward 𝑅. We employ the REINFORCE
algorithm [ 36] to learn this policy, where 𝜃represents the parame-
ters of the RL-Agent. The loss function is defined as:
LPG=−𝑅ln𝜋𝜃(𝑎|s). (5)
4.4 ML-Agent: Generating Prompts for Planning
In the ML-Agent, the most relevant path identified by the RL-Agent
is selected and integrated into the prompt for a LLM. We man-
age transferability through the ML-Agent using Meta-Learning
(ML). The rationale is that the agent is trained to structure the
prompt as an in-context learning (ICL) instance by pre-appending
the exemplar planning path, which can potentially improve LLM
generalization to new tasks by updating with only a few examples,
as evidenced in [18, 26]. Below, we discuss the model architecture
and training details for the ML-Agent.
Model Architecture. As shown in Figure 1(c), our ML-Agent uses
the text encoder structure from [ 20] for both the question encoder
and the path encoder. It employs two transformer modules with
shared self-attention layers to capture potential features. We treat
the instruction path and question as two text sequences ending with
[EOS] tokens, and derive their feature representations from the
activations of the highest transformer layer at these [EOS] tokens.
The ML-Agent is trained to align the question and instruction path
representations, and the most relevant path is retrieved based on
the cosine similarities of these representations. Notably, the model
does not use a 𝐾-classifier for path selection, ensuring that the
architecture remains independent of the 𝐾hyperparameter and
does not require retraining when 𝐾is adjusted.
Training ML-Agent. The ML-Agent training involves two phases:
pre-training (PT) and fine-tuning (FT). In PT , we optimize the agent
using two pre-training tasks: Question Path Alignment (QPA) andQuestion Path Matching (QPM). For QPA, the objective is to align
question and path representations by bringing similar pairs closer
together and pushing dissimilar pairs apart through a contrastive
approach. Specifically, we sample a batch of question-path pairs
from the support set (e.g., 𝑄1
1and𝑃1
1as shown in Figure 1). For
each pair, denoted by <𝑄𝑗
𝑖,𝑃𝑗
𝑖>, where𝑄𝑗
𝑖∈Q and𝑃𝑗
𝑖∈P, we
obtain their embedding vectors v𝑄
𝑖,𝑗and v𝑃
𝑖,𝑗via the two encoders.
We treat v𝑃
𝑖,𝑗as the positive example for v𝑄
𝑖,𝑗(the anchor), since 𝑄𝑗
𝑖
and𝑃𝑗
𝑖are paired, while the other paths in the batch are considered
as negatives. The contrastive loss, denoted as L𝑄,𝑃, encourages the
paths to align with the anchor question by comparing their positive
and negative pairs, that is
L𝑄,𝑃=∑︁
𝑄𝑗
𝑖∈Q−logexpv𝑄
𝑖,𝑗·v𝑃
𝑖,𝑗/𝜏
Í
𝑃𝑗′
𝑖′∈P,𝑖′≠𝑖,𝑗′≠𝑗expv𝑄
𝑖,𝑗·v𝑃
𝑖′,𝑗′/𝜏,(6)
where𝜏denotes a temperature parameter. Symmetrically, we can
defineL𝑃,𝑄by anchoring at v𝑃
𝑖,𝑗. The overall lossLQPAis then
defined as:
LQPA=(L𝑄,𝑃+L𝑃,𝑄)/2. (7)
For QPM, we align questions with paths through a binary classi-
fication task. The model predicts whether a question-path pair is a
match (labeled 1) or a mismatch (labeled 0). The training objective
uses binary cross-entropy loss, which is defined as follows:
LQPM=−𝑦∗log(𝑃)+(𝑦−1)∗log(1−𝑃). (8)
where𝑦denotes the label and 𝑃represents the predicted probability
of the positive class. Finally, the ML-Agent is trained using a multi-
task learning approach, with the loss function LPTis defined as:
LPT=LQPA+L QPM. (9)
In FT , we further fine-tune the model using questions from the
query set. Specifically, for each question ˆ𝑄∈ˆQ, we retrieve 𝐾paths
using the RL-Agent. We employ a hard negative mining strategy
where the𝐾retrieved paths are considered as hard negative samples,
since they are relevant to the question ˆ𝑄. Additionally, we sample
paths from other questions and add them to the 𝐾paths, forming
a path pool denoted by ˆP. The performance is then evaluated by
comparing the ground truth answer 𝐴with the generated answer
ˆ𝐴via a LLM for each path in ˆP. Based on a specific metric Δ(ˆ𝐴,𝐴),
the best path denoted by ˆ𝑃is identified as the positive example for
ˆ𝑄, and the other paths in the pool are considered as negatives. The
loss functionLFTfor the fine-tuning phase is defined as:
LFT=(L′
𝑄,𝑃+L′
𝑃,𝑄)/2
L′
𝑄,𝑃=∑︁
ˆ𝑄∈ˆQ−logexp
vˆ𝑄·vˆ𝑃/𝜏
Í
𝑃∈ˆP,𝑃≠ˆ𝑃exp
vˆ𝑄·v𝑃/𝜏,(10)
where vˆ𝑄and vˆ𝑃denote the embedding vectors for ˆ𝑄and ˆ𝑃, re-
spectively.L′
𝑃,𝑄is a symmetric definition based on L′
𝑄,𝑃.
Prompt Structure for LLM Generation. The path ˆ𝑃returned by
ML-Agent is used to construct a prompt that guides the LLM in

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
generating an answer, denoted as ˆ𝐴. Our prompt is composed of
four parts, as illustrated in Figure 1(c). (1) Task Description: This
part introduces the task, detailing the specific question ˆ𝑄to be
solved. (2) Instruction Definitions: This part provides definitions
for each instruction, such as Search[topic] orLookup[entity] .
(3) Planning Path: The path ˆ𝑃is integrated to create a structured
plan, guiding the LLM through step-by-step actions to address ˆ𝑄. (4)
Demonstrations: Examples of planning paths are provided to offer
reference and context for the LLM. Additionally, the InstructRAG
framework supports integration with both trainable LLMs (e.g.,
fine-tuning GLM-4 [ 9] with the ground truth paths following [ 42]),
and frozen LLMs (e.g., GPT-4o mini [ 1] and DeepSeek-V2 [ 7]) to
leverage its inherent capabilities for planning.
Algorithm 2: TheInstructRAG - Training Stage
Require : a training support set S; a training query set Q
1randomly initialize 𝜃for RL-Agent and 𝜂for ML-Agent
2construct the instruction graph 𝐺withSby Algorithm 1
3while not done do
4 sample a batch of tasks T
5 foreach𝑇𝑖∈T( 1≤𝑖≤|T| )do
6 evaluate∇𝜃L𝑇𝑖
WS(RL-Agent𝜃)by Eq 4 wrtBquestions for
𝑇𝑖inS
7 compute adapted 𝜃′
𝑖←𝜃−𝛼∇𝜃L𝑇𝑖
WS(RL-Agent𝜃)
8 evaluate∇𝜃L𝑇𝑖
PT(ML-Agent𝜂)by Eq 9 wrtBquestions for
𝑇𝑖inS
9 compute adapted 𝜂′
𝑖←𝜂−𝛼∇𝜃L𝑇𝑖
PT(ML-Agent𝜂)
10 update𝜃←𝜃−𝛽∇𝜃Í
𝑇𝑖L𝑇𝑖
PG(RL-Agent𝜃′
𝑖)by Eq 5 wrt
questions for all sampled tasks in Q
11 update𝜂←𝜂−𝛽∇𝜂Í
𝑇𝑖L𝑇𝑖
FT(ML-Agent𝜂′
𝑖)by Eq 10 wrt
questions for all sampled tasks in Q
12Return trained RL-Agent𝜃and ML-Agent𝜂
Algorithm 3: TheInstructRAG - Few-Shot Learning Stage
Require : a testing support set S′; RL-Agent𝜃; ML-Agent𝜂
1insert S′into𝐺by Algorithm 1, and obtain 𝐺′
2foreach𝑇𝑖∈S′(1≤𝑖≤|S′|)do
3𝜃′
𝑖←𝜃−𝛼∇𝜃L𝑇𝑖
WS(RL-Agent𝜃)−𝛽∇𝜃L𝑇𝑖
PG(RL-Agent𝜃)by
Eq 4 and Eq 5 wrtBquestions for 𝑇𝑖inS′
4𝜂′
𝑖←𝜂−𝛼∇𝜂L𝑇𝑖
PT(ML-Agent𝜂)−𝛽∇𝜂L𝑇𝑖
FT(ML-Agent𝜂)by
Eq 9 and Eq 10 wrt Bquestions for 𝑇𝑖inS′
5Return adapted RL-Agent𝜃′
𝑖and ML-Agent𝜂′
𝑖for each task
Algorithm 4: TheInstructRAG - Testing Stage
Require : a testing query set Q′; RL-Agent𝜃′
𝑖; ML-Agent𝜂′
𝑖
1foreach𝑇𝑖∈Q′(1≤𝑖≤|Q′|)do
2 run RL-Agent𝜃′
𝑖and ML-Agent𝜂′
𝑖for questions in 𝑇𝑖
3 evaluate the effectiveness with a metric Δ(·,·)
4Return the average effectiveness across all tasks4.5 The InstructRAG Framework
We present the InstructRAG framework in three stages: (1) the
Training Stage, (2) the Few-Shot Learning Stage, and (3) the Testing
Stage. In (1), the framework employs a meta-learning approach [ 8]
to collaboratively train two agents using both support and query
sets from seen tasks. In (2), the agents’ parameters are quickly
adapted to unseen tasks using few-shot examples on the support
set. In (3), the effectiveness of the adaptation is evaluated using the
query set on these unseen tasks.
Training Stage. As shown in Algorithm 2, the process inputs a
support set and a query set from the seen training tasks and outputs
the trained RL-Agent and ML-Agent. The support set is used to
construct the instruction graph 𝐺as detailed in Algorithm 1. The
two agents are then trained iteratively. In each iteration, the RL-
Agent and ML-Agent are represented as RL-Agent𝜃andML-Agent𝜂
with parameters 𝜃and𝜂, respectively. When adapting to a new
task𝑇𝑖, the parameters 𝜃and𝜂are updated to 𝜃′and𝜂′using
Equations 4 and 9 based on the support set ( 𝛼denotes a learning
rate). The updated parameters are quickly computed using one or
more gradient descent updates with Bquestions. Following this,
the model parameters are optimized by improving the performance
ofRL-Agent𝜃′
𝑖using Equation 5 and ML-Agent𝜂′
𝑖using Equation 10,
with respect to 𝜃and𝜂across sampled tasks from the query set ( 𝛽
denotes a learning rate). Our training approach aims to optimize
both agents so that a minimal number of gradient steps on a new
task will produce the most effective behavior for that task.
Few-shot Learning Stage. As shown in Algorithm 3, the process
adapts the trained RL-Agent𝜃andML-Agent𝜂to separate models,
denoted as RL-Agent𝜃′
𝑖andML-Agent𝜂′
𝑖, for each task 𝑇𝑖. This adap-
tation involves extending the graph 𝐺to𝐺′using Algorithm 1 on
the testing support set S′. For each task, gradient descent is per-
formed to adapt RL-Agent𝜃(by Equations 4 and 5) and ML-Agent𝜂
(by Equations 9 and 10) with few-shot questions from S′.
Testing Stage. As shown in Algorithm 4, each task 𝑇𝑖is per-
formed using the corresponding adapted models ( RL-Agent𝜃′
𝑖and
ML-Agent𝜂′
𝑖) on the testing query set Q′. The average effectiveness,
evaluated using a specific metric Δ(·,·), is reported across all tasks.
5 EXPERIMENTS
5.1 Experimental Setup
Datasets. In line with previous research [ 19,42], we conduct exper-
iments on four widely-used task planning datasets: HotpotQA [ 38],
ALFWorld [ 25], Webshop [ 39], and ScienceWorld [ 29]. HotpotQA is
designed for multi-hop reasoning tasks, consists of approximately
113K QA pairs sourced from Wikipedia. ALFWorld enables agents to
complete embodied tasks in a simulated environment (e.g., placing
a washed apple in the kitchen fridge). Webshop is a web application
that simulates an online shopping environment, where an agent
navigates webpages to find, customize, and purchase an item based
on a text instruction specifying the product requirements. Science-
World assesses agents’ scientific reasoning abilities at the level of
an elementary school science curriculum.
To set up the meta-learning, for HotpotQA, we define tasks
using the 12 answer types in the dataset (e.g., Person, Location,

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
Date), where we randomly select 6 types as the seen training tasks
and 6 types as the unseen testing tasks. For ALFWorld, we use
their provided seen tasks and unseen tasks for training and testing,
respectively. For Webshop, we define tasks by product category,
where we randomly sample 60% categories for training and the
remaining for testing. For ScienceWorld, we utilize it to evaluate the
generalization capability of InstructRAG across datasets, focusing
on tasks that are entirely new to a trained InstructRAG model.
Baselines. We carefully review the literature and identify the fol-
lowing baseline methods: ReAct [ 41], WKM [ 19], Reflexion [ 23],
GenGround [22], and RAP [11]. These correspond to recent repre-
sentative techniques discussed in Section 2. For GenGround, we
employ a retriever implemented by LlamaIndex [ 14] to find infor-
mation that grounds LLM-generated answers, where we store TAO
triplets from previous successful experiences across the datasets
and utilize the retrieved similar triplets as the retriever’s output. The
same data (i.e., support sets from both training and testing tasks)
is used to prepare the external database for the retrieval-based
methods (i.e., GenGround and RAP). In addition, we incorporate
theInstructRAG and baselines into three typical LLMs, namely
GLM-4 [ 9], GPT-4o mini [ 1], and DeepSeek-V2 [ 7] for comparison.
To ensure fair performance comparisons , we note that: 1) Both
the baselines and InstructRAG are configured with same setups,
including the same retrievers and backbone LLMs; 2) we follow the
hyperparameter settings specified in their original papers.
Evaluation Metrics. Following [ 15,19,42], we evaluate the effec-
tiveness of InstructRAG on four datasets. For HotPotQA, the F1
score is used, comparing the agent’s answers with the ground truth.
For ALFWorld, the success rate, a binary metric (0 or 1), indicates
whether the agent successfully completed the task. For WebShop
and ScienceWorld, a reward score ranging from 0 to 1 is employed
to measure the level of task completion. Overall, higher values indi-
cate superior results. We note that all reported experimental results
are statistically significant , verified by a t-test with 𝑝<0.05.
Implementation Details. We implement InstructRAG and base-
lines using Python 3.7. The threshold 𝛿for constructing instruction
graphs is set to 0.4. In RL-Agent, we implement a two-layered feed-
forward neural network. The first layer consists of 20 neurons
using the tanh activation function, and the second layer comprises
2 neurons corresponding to the action space to include or exclude a
node. We use the Adam stochastic gradient descent with a learning
rate of 0.001 to optimize the policy, and the reward discount is set
to 0.99. In ML-Agent, the hyperparameter 𝐾for selecting a path
is empirically set to 3. To boost training efficiency, we cache the
inputs and outputs generated by the LLMs during training.
5.2 Experimental Results
(1) Effectiveness Evaluation (comparison with baseline meth-
ods). We evaluate the effectiveness of InstructRAG against base-
lines across three LLMs on unseen tasks in Table 1, InstructRAG
consistently outperforms the baselines, demonstrating superior ef-
fectiveness. Notably, it achieves improvements of 19.2%, 9.3%, and
6.1% over the best baseline (RAP) on HotpotQA, ALFWorld, and
Webshop, respectively. This improvement can be attributed to two
factors: 1) InstructRAG employs a graph-based organization of in-
struction paths, enabling the combination into new paths for moreTable 1: Effectiveness of InstructRAG on unseen tasks.
Backbone Method HotpotQA ALFWorld Webshop
GLM-4ReAct [41] 24.04 47.01 62.13
WKM [19] - 64.18 68.14
Reflexion [23] 26.88 52.99 67.91
RAP [11] 27.86 64.18 69.45
GenGround [22] 26.97 58.96 63.18
InstructRAG 33.61 72.39 71.25
GPT-4o miniReAct [41] 25.45 42.54 49.16
WKM [19] - 55.22 54.56
Reflexion [23] 27.39 50.75 51.31
RAP [11] 27.66 56.71 56.31
GenGround [22] 28.99 44.03 53.73
InstructRAG 31.05 58.21 64.18
DeepSeek-V2ReAct [41] 25.35 52.24 57.58
WKM [19] - 72.39 67.46
Reflexion [23] 28.69 67.16 61.13
RAP [11] 29.82 72.39 72.72
GenGround [22] 33.50 67.16 62.24
InstructRAG 37.17 81.34 74.00
Table 2: Effectiveness of InstructRAG across datasets (Train-
ing: HotpotQA, Testing: ScienceWorld).
HotpotQA→ScienceWorld GLM-4 GPT-4o mini DeepSeek-V2
RAP 24.37 23.49 32.15
InstructRAG 26.85 25.10 33.96
Table 3: Effectiveness of InstructRAG on seen tasks.
Backbone Method HotpotQA ALFWorld Webshop
GLM-4RAP [11] 27.27 63.33 68.35
InstructRAG 34.99 72.50 73.60
GPT-4o miniRAP [11] 27.18 60.00 57.92
InstructRAG 31.09 64.17 64.92
DeepSeek-V2RAP [11] 31.53 75.83 71.01
InstructRAG 38.81 84.17 79.17
Table 4: Robustness to erroneous historical paths.
Noise rate 0% 10% 20% 30% 40% 50%
RAP [11] 29.82 28.74 27.42 26.01 24.10 21.72
InstructRAG 37.17 36.64 36.17 35.45 34.29 33.04
effective planning, rather than independently storing them in an
external database as RAP does, and 2) it utilizes a meta-learning
approach to efficiently adapt the trained model to diverse tasks.
(2) Effectiveness Evaluation (generalization capabilities across
datasets). We further evaluate the generalization capabilities of
InstructRAG, by applying the trained InstructRAG model from Hot-
potQA to entirely new tasks in the ScienceWorld dataset. The results
are reported in Table 2. Consistently, InstructRAG outperforms the
best baseline method, RAP, with 6%-10% improvements.
(3) Effectiveness Evaluation (performance on seen training
tasks). We also report the performance on seen training tasks. Com-
pared to RAP, similar improvements are observed in Table 3, with
average improvements of 21.9%, 10.8 %, and 10.4% on HotpotQA,
ALFWorld, and Webshop, respectively.
(4) Effectiveness Evaluation (robustness to noise). We examine
the impact of erroneous historical paths in the instruction graph on
task performance, by introducing noisy paths (i.e., failed instruction
paths from past experiences) with a controlled noise rate ranging

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
20% 40% 60% 80% 100%
T ask Ratio363636363737373738F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
T ask Ratio100150200250300T otal Learning Time (mins)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio3535363637F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio100150200250300T otal Learning Time (mins)
HotpotQA
(a) F1 Score (#Unseen tasks) (b) Time (#Unseen tasks) (c) F1 Score (#Samples) (d) Time (#Samples)
Figure 2: F1 scores and few-shot learning times wrt the number of unseen tasks or samples with DeepSeek-V2 on HotpotQA.
Table 5: Ablation study for verifying enlargeability (RL-
Agent) and transferability (ML-Agent) on HotpotQA.
Components F1 score
InstructRAG 37.17
w/o Instruction Graph 32.87
w/o RL-Agent 33.45
w/o warm-start in RL-Agent 34.37
w/o policy gradient in RL-Agent 36.18
w/o ML-Agent 34.78
w/o pre-training in ML-Agent 36.19
w/o fine-tuning in ML-Agent 36.24
Table 6: Impacts of threshold 𝛿and runtime efficiency.
𝛿 0.0 0.2 0.4 0.6 0.8 1.0
F1 score 34.02 35.19 37.17 36.61 36.48 35.93
Construction (s) 19.61 21.27 20.87 21.65 23.08 22.07
Training (hours) 23.26 23.64 23.93 24.81 25.13 25.17
Few-shot (mins/task) 26.35 26.89 27.10 28.01 28.47 28.51
Testing (s) 33.87 34.46 34.74 35.85 36.45 36.47
# of nodes 5 29 286 666 720 725
Table 7: Impacts of the number of retrieved 𝐾paths.
𝐾 1 2 3 4 5
F1 score 34.78 36.16 37.17 36.98 36.77
Testing (s) 32.05 33.57 34.74 35.31 42.09
from 0% to 50%. For comparison, we use RAP, the best baseline
method, which also includes noisy paths in its database. The F1
score results, based on DeepSeek-V2 for HotPotQA, are reported in
Table 4. Notably, even with a noise rate of 50%, the performance of
InstructRAG remains relatively stable, with only a 11.1% decrease.
This robustness stems from the diverse instruction combinations
that help select appropriate paths and mitigate noise effectively.
(5) Ablation Study. We perform an ablation study to assess the con-
tributions of different components within InstructRAG in Table 5.
We evaluate the following modifications: (1) omitting the instruc-
tion graph and allowing InstructRAG to retrieve relevant paths
directly from stored individual paths; (2) omitting the RL-Agent
and using a threshold-based method to determine node inclusion
or exclusion, (3) the warm-start stage, (4) the policy gradient stage;
(5) omitting the ML-Agent and relying solely on the path returned
by the RL-Agent for testing on unseen tasks, (6) the pre-training
stage, (7) the fine-tuning stage. We observe that the knowledge
in the instruction graph contribute to a significant improvementof 11.6%, and both the RL-Agent and ML-Agent contribute to the
overall improvements of 11.1% and 6.9%, respectively.
(6) Parameter Study (threshold 𝛿for constructing instruction
graphs and runtime efficiency). As shown in Table 6, we vary
the threshold 𝛿from 0.0 to 1.0 to control the graph construction pro-
cess. As𝛿increases, more nodes are created, but the construction
time remains stable. This is because the total number of indexed
instructions with AKNN is not sensitive to the threshold. We ob-
serve that the F1 score initially increases and then decreases as
𝛿increases. When 𝛿=0.0, there are few instruction node sets to
manage all instructions, making it difficult to accurately identify
instructions for a given question due to the large set size. Con-
versely, when 𝛿=1.0, the graph reduces to individual instruction
paths, losing the flexibility to combine instructions into new paths.
Therefore, a moderate 𝛿leads to the best performance. In addition,
we present the training, few-shot learning, and testing times as 𝛿
increases. Notably, training and few-shot learning require signifi-
cantly more time than the graph construction, primarily due to the
higher computational demands of language generation compared to
the algorithmic construction. Furthermore, the graph construction
is a one-time process conducted during data pre-processing.
(7) Parameter Study (the number of retrieved candidate paths
𝐾).We vary the number of retrieved paths 𝐾from 1 to 5 and report
the F1 scores and testing times in Table 7. As expected, testing
time increases with larger 𝐾due to the consideration of more
candidate paths. We observe that overall performance converges
when𝐾reaches 3, at which point a potentially optimal path can be
retrieved from the instruction graph.
(8) Impact of Few-shot Learning. InstructRAG includes a few-
shot learning stage to quickly adapt to each task. We report its
effectiveness and few-shot learning time based on the number of
unseen tasks or the number of samples per task with DeepSeek-V2.
As shown in Figure 2(a)-(b) , we vary the task ratio from 0.2 to 1.0,
and observe that the effectiveness remains stable as the number of
tasks increases, indicating a strong transferability across different
tasks. The running time increases with more tasks due to the inclu-
sion of additional training data. Additionally, we vary the sample
ratio from 0.2 to 1.0 for each task. As shown in Figure 2(c) and Fig-
ure 2(d), we observe that the effectiveness improves and converges
around 80% of the samples, while the running time increases as
more samples are used for training. We note that, on average, a
task requires 27.1 minutes for adaptation, and different tasks can
be processed in parallel. The results for GLM-4 and GPT-4o mini
show similar trends and are therefore omitted for brevity.

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
6 CONCLUSION
In this paper, we conduct a systematic study on leveraging RAG for
task planning and identify two critical properties: enlargability and
transferability. We introduce InstructRAG , a novel multi-agent
meta-reinforcement learning solution that integrates an instruction
graph, an RL-Agent, and an ML-Agent to optimize end-to-end task
planning performance. Our extensive experiments on four widely
used datasets, across various LLMs demonstrate that InstructRAG
delivers superior performance and exhibits the ability to rapidly
adapt to new tasks using few-shot examples. As a future direction,
we plan to extend InstructRAG to accommodate more tasks.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski,
Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr
Nyczyk, et al .2024. Graph of thoughts: Solving elaborate problems with large
language models. In AAAI , Vol. 38. 17682–17690.
[3]Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and
Shunyu Yao. 2023. Fireact: Toward language agent fine-tuning. arXiv preprint
arXiv:2310.05915 (2023).
[4]Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. 2022. Meta-
learning via Language Model In-context Tuning. In ACL. 719–730.
[5]Gautier Dagan, Frank Keller, and Alex Lascarides. 2023. Dynamic planning with
a llm. arXiv preprint arXiv:2308.06391 (2023).
[6]Budhaditya Deb, Ahmed Hassan, and Guoqing Zheng. 2022. Boosting Natural
Language Generation from Instructions with Meta-Learning. In EMNLP . 6792–
6808.
[7]DeepSeek-AI. 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-
of-Experts Language Model. arXiv preprint arXiv:2405.04434 (2024).
[8]Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-
learning for fast adaptation of deep networks. In ICML . PMLR, 1126–1135.
[9]Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego
Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al .2024. ChatGLM: A Family
of Large Language Models from GLM-130B to GLM-4 All Tools. arXiv preprint
arXiv:2406.12793 (2024).
[10] Malte Helmert. 2006. The fast downward planning system. JAIR 26 (2006),
191–246.
[11] Tomoyuki Kagaya, Thong Jing Yuan, Yuxuan Lou, Jayashree Karlekar, Sugiri
Pranata, Akira Kinose, Koki Oguri, Felix Wick, and Yang You. 2024. Rap: Retrieval-
augmented planning with contextual memory for multimodal llm agents. arXiv
preprint arXiv:2402.03610 (2024).
[12] Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024. PlanRAG: A Plan-then-
Retrieval Augmented Generation for Generative Large Language Models as
Decision Makers. arXiv preprint arXiv:2406.12430 (2024).
[13] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas,
and Peter Stone. 2023. Llm+ p: Empowering large language models with optimal
planning proficiency. arXiv preprint arXiv:2304.11477 (2023).
[14] Jerry Liu. 2022. LlamaIndex . https://doi.org/10.5281/zenodo.1234
[15] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy,
Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al .2023. Bolaa:
Benchmarking and orchestrating llm-augmented autonomous agents. arXiv
preprint arXiv:2308.05960 (2023).
[16] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah
Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al .2024.
Self-refine: Iterative refinement with self-feedback. NeurIPS 36 (2024).
[17] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs. TPAMI
42, 4 (2018), 824–836.
[18] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2021.
Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943 (2021).
[19] Shuofei Qiao, Runnan Fang, Ningyu Zhang, Yuqi Zhu, Xiang Chen, Shumin Deng,
Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2024. Agent Planning
with World Knowledge Model. arXiv preprint arXiv:2405.14205 (2024).
[20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InICML . PMLR, 8748–8763.[21] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius,
and Stefanie Tellex. 2022. Planning with large language models via corrective
re-prompting. In NeurIPS Workshop .
[22] Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao, Pengjie Ren, Zhumin Chen,
and Zhaochun Ren. 2024. Generate-then-Ground in Retrieval-Augmented Gener-
ation for Multi-hop Question Answering. ACL (2024).
[23] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and
Shunyu Yao. 2024. Reflexion: Language agents with verbal reinforcement learning.
NeurIPS 36 (2024).
[24] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han,
Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. Alfred: A benchmark
for interpreting grounded instructions for everyday tasks. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition . 10740–10749.
[25] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam
Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and em-
bodied environments for interactive learning. arXiv preprint arXiv:2010.03768
(2020).
[26] Sanchit Sinha, Yuguang Yue, Victor Soto, Mayank Kulkarni, Jianhua Lu, and
Aidong Zhang. 2024. MAML-en-LLM: Model agnostic meta-training of LLMs for
improved in-context learning. KDD (2024).
[27] Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. 2024.
Trial and error: Exploration-based trajectory optimization for llm agents. arXiv
preprint arXiv:2403.02502 (2024).
[28] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang,
Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on large
language model based autonomous agents. Frontiers of Computer Science 18, 6
(2024), 186345.
[29] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu.
2022. ScienceWorld: Is your Agent Smarter than a 5th Grader?. In EMNLP . 11279–
11298.
[30] Renxi Wang, Haonan Li, Xudong Han, Yixuan Zhang, and Timothy Baldwin.
2024. Learning From Failure: Integrating Negative Examples when Fine-tuning
Large Language Models as Agents. arXiv preprint arXiv:2402.11651 (2024).
[31] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).
[32] Yizhong Wang, Swaroop Mishra, et al .2022. Benchmarking generalization via
in-context instructions on 1,600+ language tasks. arXiv preprint arXiv:2204.07705
2 (2022).
[33] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. 2024.
Rat: Retrieval augmented thoughts elicit context-aware reasoning in long-horizon
generation. arXiv preprint arXiv:2403.05313 (2024).
[34] Zhaowei Wang, Hongming Zhang, Tianqing Fang, Ye Tian, Yue Yang, Kaixin
Ma, Xiaoman Pan, Yangqiu Song, and Dong Yu. 2024. DivScene: Benchmarking
LVLMs for Object Navigation with Diverse Scenes and Objects. arXiv preprint
arXiv:2410.02730 (2024).
[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models. NeurIPS 35 (2022), 24824–24837.
[36] Ronald J Williams. 1992. Simple statistical gradient-following algorithms for
connectionist reinforcement learning. Machine learning 8, 3 (1992), 229–256.
[37] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming
Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al .2023. The rise and potential
of large language model based agents: A survey. arXiv preprint arXiv:2309.07864
(2023).
[38] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. In EMNLP . 2369–2380.
[39] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop:
Towards scalable real-world web interaction with grounded language agents.
NeurIPS 35 (2022), 20744–20757.
[40] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and
Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with
large language models. NeurIPS 36 (2024).
[41] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.
arXiv preprint arXiv:2210.03629 (2022).
[42] Yuqi Zhu, Shuofei Qiao, Yixin Ou, Shumin Deng, Ningyu Zhang, Shiwei Lyu,
Yue Shen, Lei Liang, Jinjie Gu, and Huajun Chen. 2024. Knowagent: Knowledge-
augmented planning for llm-based agents. arXiv preprint arXiv:2403.03101 (2024).

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
A Appendix
A.1 Overview of InstructRAG in Three Stages
The three stages of InstructRAG —training, few-shot learning, and
testing—are summarized in Table 8.
A.2 Prompts
We provide the InstructRAG prompts for HotpotQA, ALFWorld,
and Webshop in Table 9, Table 10, and Table 11, respectively.
A.3 Discussion on the Use of Multi-Agent for
Task Planning
We provide a discussion to explain the rationale behind using both
RL-Agent and ML-Agent for task planning instead of modifying
a single agent (e.g., RL-Agent by setting 𝐾=1) to address both
enlargeability and transferability. The task planning in this study
requires addressing two key properties: enlargeability and transfer-
ability. These properties are somewhat orthogonal: enlargeability
involves combining instructions for questions within the seen tasks,
while transferability focuses on rapid adaptation to the unseen
tasks. It is challenging for a single agent to optimize effectively in
both directions simultaneously. Therefore, we design a multi-agent
framework collaborated with two distinct agents: the RL-Agent
provides candidate paths for the ML-Agent, while the ML-Agent
supplies rewards for the RL-Agent. This strategic division of labor
enables us to explicitly optimize for both enlargeability and trans-
ferability through multi-agent meta-reinforcement learning, and
we validate the solution via an ablation study presented in Table 5.A.4 Few-shot Learning with Other LLMs
We report the F1 scores and few-shot learning times for GLM-4 and
GPT-4o mini in Figure 3. Overall, similar trends can be observed,
consistent with the results from DeepSeek-V2.
A.5 Qualitative Results
Both InstructRAG and RAP leverage past experiences (e.g., instruc-
tion paths) to guide LLM planning. Table 12, Table 13, and Table 14
illustrate the planning trajectories of InstructRAG and RAP for
HotpotQA, ALFWorld, and Webshop, respectively. We note that
InstructRAG combines multiple paths from related tasks into an
instruction path, effectively guiding LLM planning. This is demon-
strated by the overlap of several instructions (highlighted in yellow)
from the instruction path in successful plans. Specifically, we an-
alyze the planning results in Table 12. InstructRAG demonstrates
an advantage by combining two paths based on a common instruc-
tion, search[Piers Haggard]. This approach effectively links two
key items—Anthony Minghella, representing the novel, and Piers
Haggard, representing the film adaptation of “The Talented Mr.
Ripley”. These connections enable the LLM to generate a correct
query formulation, allowing it to retrieve information about the
film director within the generated thoughts (e.g., Thought 5). In
contrast, RAP struggles to produce a correct query based on its re-
trieved experiences. Its generated thoughts fail to support effective
query formulation, often resulting in a planning deadlock.

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
Table 8: Overview of InstructRAG in three stages.
InstructRAG Training Stage Few-Shot Learning Stage Testing Stage
Task Seen training tasks Unseen testing tasks Unseen testing tasks
Data given Support set and query set Support set Query set
Instruction Graph G (construct with the support set) G’ (extend the support set to 𝐺)G’
ObjectiveFor each iteration:
1. Sample batch of tasks
2. Optimize RL-Agent by L𝑊𝑆and ML-Agent
byL𝑃𝑇on the support set for each task
3. Jointly optimize RL-Agent and ML-Agent by L𝑃𝐺
andL𝐹𝑇on the query sets across all sampled tasks1. Update trained RL-Agent by
L𝑊𝑆and ML-Agent byL𝑃𝑇on
the support set for each task
2. Jointly update RL-Agent and
ML-Agent byL𝑃𝐺andL𝐹𝑇on
the support set for each taskReport the average
effectiveness on the
query sets across all tasks
Table 9: Prompt for Overall Plan on HotpotQA.
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and
Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar
entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{Examples}
Here is the provided action sequence:
{Instruction Path}.
Assess the initial understanding of the task and adjust the approach if new insights or requirements arise during the process.
If an action does not yield useful information or leads to a dead end, reconsider the previous steps or switch between “Search” and “Lookup”
to gather more relevant data.
Now you have to complete the following task:
{Question}
Table 10: Prompt for Overall Plan on ALFWorld.
Interact with a household to solve a task. The following are legal actions: go, take, clean, use, examine, look, heat, cool, open, close, toggle,
put, think. When generating an action, the first word of your response must be one of the legal actions listed above.
Here are some examples.
{Examples}
Here is the provided action sequence:
{Instruction Path}.
Assess the initial understanding of the task and adjust the approach if new insights or requirements arise during the process.
Now you have to complete the following task:
{Question}

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
Table 11: Prompt for Overall Plan on Webshop.
You are an advanced reasoning agent tasked with interacting with a shopping website. The following are legal actions:
(1) search[keyword]: You can perform a search using specific keywords (if “has_search_bar” is True). Keep the keyword short and concise.
Avoid overly detailed descriptions. Only include keywords that help identify the product.
(2) click[clickables]: You can click on available clickable items.
Here are some examples.
{Examples}
Here is the provided action sequence:
{Instruction Path}.
Assess the initial understanding of the task and adjust the approach if new insights or requirements arise during the process.
Now you have to complete the following task:
{Question}
20% 40% 60% 80% 100%
T ask Ratio333333333334F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
T ask Ratio75100125150175200225250275T otal Learning Time (mins)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio2628303234F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio100150200250T otal Learning Time (mins)
HotpotQA
(a) F1 Score (#Unseen tasks) (b) Few-shot Time (#Unseen tasks) (c) F1 Score (#Samples) (d) Few-shot Time (#Samples)
20% 40% 60% 80% 100%
T ask Ratio303030303131313132F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
T ask Ratio75100125150175200225250275T otal Learning Time (mins)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio2929303030303031F1 Score (%)
HotpotQA
20% 40% 60% 80% 100%
Sample Ratio50100150200250T otal Learning Time (mins)
HotpotQA
(e) F1 Score (#Unseen tasks) (f) Few-shot Time (#Unseen tasks) (g) F1 Score (#Samples) (h) Few-shot Time (#Samples)
Figure 3: F1 scores and few-shot learning times wrt the number of unseen tasks or samples on HotpotQA, where (a)-(d) are for
GLM-4, and (e)-(h) are for GPT-4o mini.

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
Table 12: Comparison of InstructRAG and RAP with HotpotQA trajectories, where we highlight tasks, successful results, and
failure results in purple, green, and red, respectively. Different colors are used to label the instructions in the instruction path,
which combines Path 1 and Path 2. Junction instructions from the paths are highlighted in gray, and instructions (actions) in
the planning trajectories that overlap with the instruction path are highlighted in yellow.
Task Description : Your task is to answer a question.
Between the directors of “The Decline of Western Civilization” and “The Talented Mr. Ripley”, who is older?
InstructRAG
Instruction Path : Search[Anthony Minghella] → Search[Piers
Haggard]→Search[The Decline of Western Civilization] →
Search[Penelope Spheeris] (Derived from combining tasks: “Are both
Anthony Minghella and Piers Haggard film diectors?” and “Which
director, Piers Haggard or Penelope Spheeris, is best known as a
documentary film director whose works include the trilogy titled ‘The
Decline of Western Civilization’?”)
Combined from:
Path 1 : Search[Anthony Minghella] →Search[Piers Haggard] →Fin-
ish[yes]
Path 2 :Search[Piers Haggard] →Search[The Decline of West-
ern Civilization]→Search[Penelope Spheeris] →Finish[Penelope
Spheeris]
Thought 1: I need to search the directors of ‘The Decline of Western
Civilization’ and ‘The Talented Mr. Ripley’, find their birthdates, and
then compare them to determine who is older.
Act 1: Search[The Decline of Western Civilization]
Obs 1: The Decline of Western Civilization is a 1981 American docu-
mentary filmed through 1979 and 1980...
Thought 2: The director of ‘The Decline of Western Civilization’ is
Penelope Spheeris. I need to search her birthdate.
Act 2: Search[Penelope Spheeris]
Obs 2: Penelope Spheeris (born December 2, 1945)[1] is an American
film director, producer, and screenwriter...
...
Thought 5: I need to search for the director of the 1999 film adaptation
of ‘The Talented Mr. Ripley’. The correct search term should be ‘The
Talented Mr. Ripley (film)’.
Act 5: Search[The Talented Mr. Ripley (film)]
Obs 5: The Talented Mr. Ripley is a 1999 American psychological thriller
film written and directed by Anthony Minghella, based on Patricia
Highsmith’s 1955 novel of the same title...
Thought 6: The director of ‘The Talented Mr. Ripley’ (1999 film) is
Anthony Minghella. I need to search his birthdate.
Act 6: Search[Anthony Minghella]
Obs 6: Anthony Minghella, CBE (6 January 1954 – 18 March 2008) was
a British film director, playwright, and screenwriter...
Thought 7: Anthony Minghella was born on January 6, 1954. Compar-
ing the birthdates, December 2, 1945 (Penelope Spheeris) < January 6,
1954 (Anthony Minghella), so Penelope Spheeris is older.
Act 7: Finish[Penelope Spheeris]
Obs 7: Episode finished, Answer is CORRECT
Success flag = True, Success rate = 1.0RAP
Paths : Separately retrieved from the tasks: “Which director, Piers Hag-
gard or Penelope Spheeris, is best known as a documentary film di-
rector whose works include the trilogy titled ‘The Decline of Western
Civilization’?”, “Are both Anthony Minghella and Piers Haggard film
diectors?”, “Extras, is a British sitcom about extras working in tele-
vision, film, and theatre, the series was co-produced by the BBC and
HBO and was created, written, and directed by Ricky Dene Gervais,
an English comedian, actor, writer, producer, director, singer, and mu-
sician, born on which date?”, and “What teen sex comedy film written
by Adam Herz and directed by brothers Paul and Chris Weitz, starred
an actor best known for her role as Jessica ?”.
Thought 1: I need to identify the directors of ‘The Decline of Western
Civilization’ and ‘The Talented Mr. Ripley’ to determine who is older.
Act 1: Search[The Decline of Western Civilization]
Obs 1: The Decline of Western Civilization is a 1981 American docu-
mentary filmed through 1979 and 1980...
Thought 2: Penelope Spheeris is the director of ‘The Decline of Western
Civilization’. Next, I need to find the director of ‘The Talented Mr.
Ripley’.
Act 2: Search[The Talented Mr. Ripley]
Obs 2: The Talented Mr. Ripley is a 1955 psychological thriller novel
by Patricia Highsmith...
Thought 3: The search for ‘The Talented Mr. Ripley’ provided infor-
mation about the novel and its adaptations but did not mention the
director of the 1999 film adaptation. I need to search for the director
of the 1999 film ‘The Talented Mr. Ripley’.
Act 3: Search[director of The Talented Mr. Ripley 1999 film]
Obs 3: Could not find director of The Talented Mr. Ripley 1999 film...
Thought 4: I need to search for the director of the 1999 film ‘The Tal-
ented Mr. Ripley’ directly by searching for the film title and identifying
the director from the information provided.
Act 4: Search[The Talented Mr. Ripley 1999 film]
Obs 4: Could not find The Talented Mr. Ripley 1999 film...
Thought 5: I need to search for the director of the 1999 film ‘The Tal-
ented Mr. Ripley’ directly by searching for the film title and identifying
the director from the information provided.
Act 5: Search[The Talented Mr. Ripley 1999 film]
Obs 5: Could not find The Talented Mr. Ripley 1999 film...
Thought 6: I need to search for the director of the 1999 film ‘The Tal-
ented Mr. Ripley’ directly by searching for the film title and identifying
the director from the information provided.
Act 6: Search[The Talented Mr. Ripley 1999 film]
Obs 6: Could not find The Talented Mr. Ripley 1999 film...
Thought 7: I need to search for the director of the 1999 film ‘The Tal-
ented Mr. Ripley’ directly by searching for the film title and identifying
the director from the information provided.
Act 7: Search[The Talented Mr. Ripley 1999 film]
Obs 7: Could not find The Talented Mr. Ripley 1999 film...
Success flag = False, Success rate = 0.0

InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning SIGIR ’25, July 13–18, 2025, Padua, Italy
Table 13: Comparison of InstructRAG and RAP with ALFWorld trajectories.
Task Description : You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a
countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a
towelholder 1.
Your task is to: put two soapbar in garbagecan.
InstructRAG
Instruction Path : think: to solve the task, i need to find and take a
soapbar, then clean it with sinkbasin, then put it in cabinet →go to
toilet 1→take soapbar 1 from toilet 1 →go to sinkbasin 1→clean
soapbar 1 with sinkbasin 1 →go to drawer 1→open drawer 1→take
dishsponge 3 from drawer 1 →go to garbagecan 1 →put dishsponge
3 in/on garbagecan 1 (Derived from combining tasks: “put a clean
soapbar in cabinet”, “put a clean soapbar in drawer” and “put two
dishsponge in garbagecan”)
Combined from:
Path 1 : think: to solve the task, i need to find and take a soap-
bar, then clean it with sinkbasin, then put it in cabinet →go to
toilet 1→take soapbar 1 from toilet 1 →go to sinkbasin 1 →
clean soapbar 1 with sinkbasin 1 →go to cabinet 1→open cabi-
net 1→put soapbar 1 in/on cabinet 1
Path 2 : think: to solve the task, i need to find and take a soap-
bar, then clean it with sinkbasin, then put it in drawer. →go to
bathtubbasin 1→take soapbar 1 from bathtubbasin 1 →go to
sinkbasin 1→clean soapbar 1 with sinkbasin 1 →go to drawer 1
→open drawer 1→put soapbar 1 in/on drawer 1
Path 3 : think: to solve the task, i need to find and take the first dish-
sponge, then put it in garbagecan, then find and take the second dish-
sponge, then put it in garbagecan. →go to drawer 1→open drawer 1
→take dishsponge 3 from drawer 1 →go to garbagecan 1 →put dish-
sponge 3 in/on garbagecan 1 →go to countertop 1→take dishsponge
2 from countertop 1 →go to garbagecan 1 →put dishsponge 2 in/on
garbagecan 1
Act 1: think: To solve the task, I need to find and take the first soapbar,
then put it in garbagecan, then find and take the second soapbar, then
put it in garbagecan.
Obs 1: OK.
Act 2: go to toilet 1
Obs 2: On the toilet 1, you see a soapbar 2, and a soapbar 1.
...
Act 4: take soapbar 1 from toilet 1
Obs 4: You pick up the soapbar 1 from the toilet 1.
...
Act 6: go to garbagecan 1
Obs 6: On the garbagecan 1, you see nothing.
Act 7: put soapbar 1 in/on garbagecan 1
Obs 7: You put the soapbar 1 in/on the garbagecan 1.
...
Act 9: go to toilet 1
Obs 9: On the toilet 1, you see a soapbar 2.
...
Act 11: take soapbar 2 from toilet 1
Obs 11: You pick up the soapbar 2 from the toilet 1.
...
Act 13: go to garbagecan 1
Obs 13: On the garbagecan 1, you see a soapbar 1.
Act 14: put soapbar 2 in/on garbagecan 1
Obs 14: You put the soapbar 2 in/on the garbagecan 1.
Success flag = True, Success rate = 1RAP
Path: Separately retrieved from the tasks: “put a soapbottle in garbage-
can”, “put a clean soapbar in drawer”, “put a clean soapbar in toilet”
and “put two dishsponge in garbagecan”
Act 1:think: To solve the task, I need to find and take the first soapbar,
then put it in the garbagecan, then find and take the second soapbar,
then put it in the garbagecan
Obs 1: OK.
Act 2: go to countertop 1
Obs 2: On the countertop 1, you see a candle 3, a candle 1, a soapbottle
1, and a spraybottle 2.
Act 3: take soapbar 1 from countertop 1
Obs 3: Nothing happens.
Act 4: go to sinkbasin 1
Obs 4: On the sinkbasin 1, you see nothing.
Act 5: go to sinkbasin 2
Obs 5: On the sinkbasin 2, you see nothing.
Act 6: go to countertop 1
Obs 6: On the countertop 1, you see a candle 3, a candle 1, a soapbottle
1, and a spraybottle 2.
Act 7: take soapbar 1 from countertop 1
Obs 7: Nothing happens.
Act 8: go to cabinet 1
Obs 8: The cabinet 1 is closed.
Act 9: open cabinet 1
Obs 9: You open the cabinet 1. The cabinet 1 is open. In it, you see a
cloth 2.
Act 10: take soapbar 1 from cabinet 1
Obs 10: Nothing happens.
...
Act 46: go to sinkbasin 1
Obs 46: On the sinkbasin 1, you see nothing.
Act 47: go to sinkbasin 2
Obs 47: On the sinkbasin 2, you see nothing.
Act 48: go to countertop 1
Obs 48: On the countertop 1, you see a candle 3, a candle 1, a soapbottle
1, and a spraybottle 2.
Act 49: take soapbar 1 from countertop 1
Obs 49: Nothing happens.
Success flag = False, Success rate = 0

SIGIR ’25, July 13–18, 2025, Padua, Italy Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi
Table 14: Comparison of InstructRAG and RAP with Webshop trajectories.
Task Description : Find me home office furniture sets for dining room, living room with color: navy | red, and item shape: round, and size:
21 in x 35 in, and price lower than 60.00 dollars.
Your task is to: look for home office furniture sets for dining room, living room with color: navy | red, and item shape: round, and size: 21 in
x 35 in, and price lower than 60.00 dollars.
InstructRAG
Instruction Path : search[home office furniture sets dining room liv-
ing room navy red round 1 ft 9 in x 7 ft 2 in price < 60.00] →
click[b07fkgqkz1] →click[21 in x 35 in] →click[taupe | orange] →
click[round]→click[3 ft 7 in x 5 ft 2 in] →click[buy now] (Derived
from combining tasks: “Find me home office furniture sets for dining
room, living room with color: navy | red, and item shape: round, and
size: 1 ft 9 in x 7 ft 2 in, and price lower than 60.00 dollars”, “Find me
home office furniture sets for dining room, living room with color:
taupe | orange, and item shape: runner, and size: 21 in x 35 in, and price
lower than 60.00 dollars” and “Find me home office furniture sets for
dining room, living room with color: taupe | orange, and item shape:
round, and size: 3 ft 7 in x 5 ft 2 in, and price lower than 60.00 dollars” )
Combined from:
Path 1 : search[home office furniture sets dining room living room navy
red round 1 ft 9 in x 7 ft 2 in price < 60.00] →click[b07fkgqkz1] →
click[1 ft 9 in x 7 ft 2 in] →click[navy | red]→click[round]→click[1
ft 9 in x 7 ft 2 in]→click[buy now]
Path 2 : search[taupe orange runner 21 in x 35 in furniture set] →
click[b07fkgqkz1] →click[21 in x 35 in] →click[taupe | orange]
→click[runner]→click[21 in x 35 in] →click[buy now]
Path 3 : search[home office furniture sets dining room living room color
taupe orange shape round size 3 ft 7 in x 5 ft 2 in price < 60.00] →
click[b07fkgqkz1]→click[3 ft 7 in x 5 ft 2 in] →click[taupe | orange]
→click[round]→click[3 ft 7 in x 5 ft 2 in] →click[buy now]
Act 1: reset
Obs 1: ‘has_search_bar’: True, ‘clickables’: [‘search’]
Act 2: search[home office furniture sets dining room living room navy
red round 21 in x 35 in price < 60.00]
Obs 2: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘next >’,
‘b07fkgqkz1’, ...... , ‘b07fvhpk5x’]
Act 3: click[b07fkgqkz1]
Obs 3: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’, ‘1 ft 9 in x 7 ft 2 in’, ...... ,
‘21 in x 35 in’, ...... , ‘navy-multi’, ‘taupe’]
Act 4: click[21 in x 35 in]
Obs 4: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’, ‘1 ft 9 in x 7 ft 2 in’, ...... ,
‘21 in x 35 in’, ‘runner’, ‘rectangular’, ‘round’, ‘black-gray’, ...... , ‘taupe’]
Act 5: click[round]
Obs 5: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’, ‘1 ft 9 in x 7 ft 2 in’, ...... ,
‘21 in x 35 in’, ‘runner’, ‘rectangular’, ‘round’, ‘black-gray’, ...... , ‘navy |
red’, ...... , ‘taupe’]
Act 6: click[navy | red]
Obs 6: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’, ‘1 ft 9 in x 7 ft 2 in’, ...... ,
‘21 in x 35 in’, ‘runner’, ‘rectangular’, ‘round’, ‘black-gray’, ...... , ‘taupe’]
Act 7: click[buy now]
Success flag = True, Success rate = 1.0RAP
Paths : Separately retrieved from the tasks: “Find me home office fur-
niture sets for dining room, living room with color: blue | gray, and
item shape: runner, and size: 21 in x 35 in, and price lower than 60.00
dollars”, “Find me home office furniture sets for dining room, living
room with color: navy | red, and item shape: round, and size: 1 ft 9 in x
7 ft 2 in, and price lower than 60.00 dollars ” and “Find me home office
furniture sets for dining room, living room with color: blue, and item
shape: round, and size: 1 ft 9 in x 7 ft 2 in, and price lower than 60.00
dollars”.
Act 1: reset
Obs 1: ‘has_search_bar’: True, ‘clickables’: [‘search’]
Act 2: search[home office furniture sets]
Obs 2: ‘has_search_bar’: False, ‘clickables’: [‘back to search’,
‘next >’, ‘b09688c4xm’, ‘b09jc84p1m’, ‘b099yq75wt’, ‘b01n4qb5wp’,
‘b09m6vxd6w’, ‘b0048u51n4’, ‘b07tnktkf4’, ‘b09n8slfrj’, ‘b000gltj3m’,
‘b08lnfnxsd’]
Act 3: click[b09jc84p1m]
Obs 3: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’]
Act 4: click[back to search]
Obs 4: ‘has_search_bar’: True, ‘clickables’: [‘search’]
Act 5: search[home office furniture sets navy red round 21 in x 35 in
price < 60.00]
Obs 5: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘next >’,
‘b07fkgqkz1’, ‘b09jc84p1m’, ‘b07fvhpk5x’, ‘b09688c4xm’, ‘b08p8lrfz4’,
‘b09byx42dw’, ‘b085rbzchh’, ‘b09c1ywg8w’, ‘b09cq45zrb’, ‘b06xfzxxtc’]
Act 6: click[b09cq45zrb]
Obs 6: ‘has_search_bar’: False, ‘clickables’: [‘back to search’, ‘< prev’,
‘description’, ‘features’, ‘reviews’, ‘buy now’, ‘15.7x23.6in+15.7x47.2in’,
‘19.7x31.5in+19.7x47.2in’, ‘19.7x31.5in+19.7x63in’, ‘20x24in+20x48in’,
‘23.6x35.4in+23.6x70.9in’, ‘cargoo5209’, ‘christmas-005goo7317’,
‘christmas-010goo9911’, ‘christmasgoo1729’, ‘christmasgoo3302’,
‘christmasgoo3848’, ‘christmasgoo6658’, ‘easter3540lgoo6518’, ‘east-
ergoo0555’, ‘luckygoo4356’, ‘luckygoo4796’, ‘rabbit3555lgoo1957’,
‘snowmangoo0240’, ‘spt-026goo3844’]
Act 7: click[buy now]
Success flag = False, Success rate = 0.25