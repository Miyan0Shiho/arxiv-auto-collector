# Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration

**Authors**: Sibo Xiao, Zixin Lin, Wenyang Gao, Yue Zhang

**Published**: 2025-05-27 02:05:42

**PDF URL**: [http://arxiv.org/pdf/2505.20625v1](http://arxiv.org/pdf/2505.20625v1)

## Abstract
Processing long contexts has become a critical capability for modern large
language models (LLMs). Existing works leverage agent-based divide-and-conquer
methods for processing long contexts. But these methods face crucial
limitations, including prohibitive accumulated latency and amplified
information loss from excessive agent invocations, and the disruption of
inherent textual dependencies by immoderate partitioning. In this paper, we
propose a novel multi-agent framework XpandA (Expand-Agent) coupled with
question-driven workflow and dynamic partitioning for robust long-context
processing. XpandA overcomes these limitations through: 1) dynamic partitioning
of long texts, which adaptively modulates the filling rate of context windows
for input sequences of vastly varying lengths; 2) question-guided protocol to
update flat information ensembles within centralized shared memory,
constructing consistent inter-agent knowledge across partitions; and 3)
selectively replaying specific partitions based on the state-tracking of
question-information couples to promote the resolution of inverted-order
structures across partitions (e.g., flashbacks). We perform a comprehensive
evaluation of XpandA on multiple long-context benchmarks with length varying
from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long
sequences and its significant effectiveness in enhancing the long-context
capabilities of various LLMs by achieving 20\% improvements and 1.5x inference
speedup over baselines of full-context, RAG and previous agent-based methods.

## Full Text


<!-- PDF content starts -->

Long Context Scaling: Divide and Conquer via
Multi-Agent Question-driven Collaboration
Sibo Xiao♣, Zixin Lin♣, Wenyang Gao♣,♢, Yue Zhang♢
♣Zhejiang University,♢Westlake University
{sibodotxiao}@gmail.com
Abstract
Processing long contexts has become a critical capability for modern large language
models (LLMs). Existing works leverage agent-based divide-and-conquer methods
for processing long contexts. But these methods face crucial limitations, including
prohibitive accumulated latency and amplified information loss from excessive
agent invocations, and the disruption of inherent textual dependencies by immoder-
ate partitioning. In this paper, we propose a novel multi-agent framework XpandA
(Expand-Agent) coupled with question-driven workflow and dynamic partitioning
for robust long-context processing. XpandA overcomes these limitations through:
1) dynamic partitioning of long texts, which adaptively modulates the filling rate of
context windows for input sequences of vastly varying lengths; 2) question-guided
protocol to update flat information ensembles within centralized shared memory,
constructing consistent inter-agent knowledge across partitions; and 3) selectively
replaying specific partitions based on the state-tracking of question-information
couples to promote the resolution of inverted-order structures across partitions (e.g.,
flashbacks). We perform a comprehensive evaluation of XpandA on multiple long-
context benchmarks with length varying from 1k to 1M, demonstrating XpandA’s
feasibility for processing ultra-long sequences and its significant effectiveness
in enhancing the long-context capabilities of various LLMs by achieving 20%
improvements and 1.5x inference speedup over baselines of full-context, RAG and
previous agent-based methods.
1 Introduction
Like human cognition—which depends on long-term memory, accumulated knowledge, and life
experience—Artificial General Intelligence (AGI) development requires the ability to associate
distant information and retain context over extended content [ 38;58]. Long-context tasks, such as
question answering [ 76;92;68], summarization [ 105] and coding [ 7], consistently present significant
challenges due to issues like hallucination and overlooking critical information [50].
Recent efforts to tackle these challenges have focused on three key directions: training data strategy,
architecture and workflow design. The training data strategy conducts long-context data filtering
[86;71], mixture [ 23;33], and synthesis [ 24;27] on pre-training and post-training phase. Architecture
methods employ position embeddings [ 73;64;25;74] and attention mechanisms such as transformer-
based [ 5;88;108;95;22], linear-complexity [ 65] and hybrid architectures [ 59;47;53]. Despite these
efforts, the lack of extensibility in context windows leads to input truncation when processing longer
sequences, while training long-context language models entails prohibitively high computational
costs[ 5]. Workflow design, including external memory modules [ 80;56], retrieval-augmented
generation (RAG) [ 8;34], and agent-based methods (LongAgent [ 101], Chain of Agents [ 100]),
etc., can mitigate the challenges mentioned above but introduce issues such as information loss for
inappropriate extraction [ 100], Inter-Agent Misalignment and Specification Violation inherent in
Preprint. Under review.arXiv:2505.20625v1  [cs.CL]  27 May 2025

Context
Extra
Parts
(>n*M)Dynamic Chunking
Agent:
ExplorerAgent: 
Decider
Extension
ExtensionChunk Rules:
Overlap Rules:
Overlap Size
Context LengthContext LengthChunk Size
Chunk Num
(a) (b)
(c)Figure 1: An Overview of XpandA. A plug-and-play, robust, interpretable, and efficient multi-agent
system framework that introduces a dynamic chunking strategy (a) and selective replay (b) in its
question-driven workflow (c). XpandA accepts long context input and splits the input into chunks with
overlap, and each chunk is fed to corresponding Explorer Agents {E[i,1]|i∈1,2,3}sequentially.
Then Decider D1 decides the action of the next step is to Replay or Conclude. If Replay is proposed,
E[i,2]will be called in a reverse direction with a start point according to the question state tracker.
the multi-agent system [ 9]. Furthermore, the most concerning aspect of agent-based methods is the
frequent model invocation. This not only potentially amplifies intermediate errors but also leads to an
accumulation of model call latency.
Motivated by the aforementioned challenges, we propose a multi-agent framework XpandA (Expand-
Agent) that incorporates fine-grained question-driven communication protocols and, for the first time,
extends the input sequence length of agent-based methods beyond 128k tokens. By introducing a
Saturating Function within its partitioning strategy, XpandA can practically process input sequences
with vastly varying lengths and achieve a trade-off between unit inference time and accumulated
model call latency. Furthermore, XpandA effectively guides the model’s attention transfer across
partitions via its question-driven communication protocol, thereby mitigating information dependency
fragmentation caused by partitioning. Inspired by Activity On Vertex (AOV) networks, we implement
an iterative selective replay mechanism based on collected information-question couples. This
approach effectively resolves potential topological structures of information across different text
partitions (such as narratives with flashbacks or interpolations that cannot be resolved sequentially).
We conduct comprehensive experiments on long-context benchmarks with diverse tasks of vastly
varying lengths (from 1k to 1M). Utilizing open-source LLMs, we compare XpandA with strong
baselines of full-context, Retrieval-Augmented Generation (RAG), and prior Multi-agent methods,
demonstrating its significant improvement over all baselines by up to 20% and 1.5x inference speedup.
In summary, our contributions to the research community of long-context processing include:
1.Proposing a multi-agent framework with fine-grained question-driven communication protocols
that can practically expand the LLM’s context window beyond 128k without training.
2.Developing a dynamic partitioning strategy that achieves asymptotically optimal inference
speedup while enhancing inference performance by trade-off between unit inference time and
accumulated model call latency with input sequences of vastly varying lengths (from 1k to 1M).
3.Establishing a paradigm for multi-agent system in long-context tasks domain to mitigate inter-
agent misalignment and specification violations using principled prompt and structured work-
flow.
2

2 Related Work
2.1 Long Context Language Modeling
Long context processing in LLMs has recently become a central focus [ 49], driven by tasks such as
long-document QA [ 35;15;76;55;21], summarization [ 29;105], retrieval-augmented generation
(RAG) [ 39], in-context learning [ 46;90], and large-codebase understanding [ 32;7]. These require
models to locate, aggregate, and reason over dispersed information, while overcoming "lost in the
middle" [50] issues. Existing approaches fall into three main categories: Training data strategies,
architectural and workflow design [ 48]. Data strategies enhance a model’s capacity to process lengthy
texts through the filter [ 86;71], mixture [ 23;33], and synthesis [ 24;27] of long-context data in
both pre-training and post-training phases. Architecture design aims to develop model structures
capable of effectively handling extended texts by position embeddings [ 73;64;25;74] and attention
mechanisms such as transformer-based [ 5;88;108;95;22], linear-complexity [ 65] and hybrid
architectures [ 59;47;53]. Workflow-based methods minimize model changes, using techniques like
external memory [ 80;56], RAG [ 8;34], or agent-based reasoning [ 11;45;100], to further improve
the long context capability.
2.2 LLM Reasoning with Search and Planning
Human problem-solving relies on decomposing complex tasks into manageable steps [ 19], a paradigm
successfully adapted to LLMs. Early work by [ 85] and [ 36] introduced Chain-of-Thought (CoT)
prompting, using intermediate steps to guide reasoning, while later studies like [ 20] and [ 106]
proposed iterative decomposition into subtasks. To enhance reasoning structures, [ 107] and [ 3]
developed frameworks for self-discovering and dynamically adapting reasoning modules. Further,
methods like [ 17] and [ 99] address error correction in intermediate steps, particularly for smaller
models. Beyond linear reasoning, exploration-based approaches leverage search over multiple paths,
inspired by human problem-solving [ 72]. While early methods like self-consistency [ 81] generated
limited diversity, advanced frameworks such as Tree-of-Thoughts [ 93] and Graph-of-Thoughts [ 6]
enable finer-grained branching. Aggregation strategies further improve robustness, ranging from
ensemble-based voting [77; 44] to verifier-guided selection [82].
2.3 Multi-Agent LLMs
Research efforts in Multi-Agent Systems have encompassed diverse aspects of collaboration, covering
mechanisms like cooperation [ 14;70], competition [ 12;102], and coopetition [ 1;16], as well
as strategies such as rule-based [ 97;109], role-based [ 14;28], and model-based approaches [ 42;
89]. Furthermore, other key considerations include communication structures (centralized [ 31;61],
decentralized [ 14;94], and hierarchical [ 41;52]), which offer scenario-dependent advantages and
disadvantages, and coordination architectures (static or dynamic [ 13;30;83]), providing trade-offs
between stability and flexibility. MASs demonstrate broad potential for applications across numerous
domains, including 5G/6G networks [ 84;104], Natural Language Generation (NLG) and Question
Answering (QA) [ 31;75;87] and socio-cultural simulations [ 40;60], among others. Nevertheless,
the field still confronts numerous challenges [ 10]. These include achieving unified governance,
understanding the inherent limitations of individual agents [ 97;69], scalability, efficient resource
management [ 79], developing robust evaluation benchmarks [ 51;63] and addressing ethical risks and
safety concerns [2; 18; 57].
3 Method
Figure 1 illustrates our XpandA framework, which operates in three stages. In Stage 1, Dynamic
partition strategy is employed to decompose the input sequence of vastly varying length into chunks
of appropriate length. In Stage 2, text chunks are assigned one-to-one to Explorers. Explorers
then process the chunks sequentially by: decomposing problems, searching information and then
generating new problems. Problems and relevant information are added to shared information
memory. In Stage 3, Decider evaluates information completeness: if sufficient, it generates the final
answer; otherwise, it proposes a selective replay trajectory of potentially informative chunks.
3

3.1 Stage 1: Dynamic Partition of Long Input Sequence
Algorithm 1: XpandA Workflow
Input: Context C, Query Q.
Parameters: LLM (θ), instruct prompts (IE, ID),
chunk size K, chunk overlap P,
max replay times MRT
Output: Answer A.
SplitCintoxchunks {c1, c2,···, cx};
Create empty dictionary T,P;
Initialize o←1,rev←1;
while replay times ≤MRT do
foriino, o+rev,···do
Ti, Pi←Explorer (IE, Q, T, P, c i);
T←Merge (T, Ti);
P←Merge (P, Pi);
forevery k∈Pi.keys and k∈T.keys do
Delete (k, T[k])fromT;
end
end
Action ,ˆA←Decider (ID, Q, P );
ifReplay in Action theno←max(min( T.values −1,1) ifrev > 0
min(max( T.values + 1, l)otherwise
rev← −rev;
replay times ←replay times + 1;
end
else Break ;
end
A←Parse (ˆA);
return A;The Dynamic Partition method uses a satura-
tion function: for shorter inputs, it increases unit
chunk length; for longer inputs, the length sat-
urates and chunk count grows. Let wbe the
input token length, nthe target chunk count,
[L, K]the range of overlap, and Mthe maxi-
mum chunk size. The partitioning follows:
Chunk size =w
n
ifw
n≤M
M ifw
n> M(1)
where ⌈·⌉denotes the ceiling function. In the
first case, chunks maintain approximately equal
size with an overlap δ. In the second case, the
number of chunks grows dynamically as:
Number of chunks =w
M−δ
(2)
where δ(with increase rate α, upper limit Kand
lower limit L)1equals to:
δ=max(L, min (α∗w, K )) (3)
3.2 Stage 2: Question-guided Exploration
In Stage 1: XpandA operates on a sequence
oflExplorers. Each Explorer accepts the con-
catenation of a document chunk ci, user query
q, instruction for a specific reasoning task ID
and gathered information, denoted as "(ques-
tion,[answer]) pairs" P. Explorer is instructed to output enlarged Piand generate newly-generated
problems included in "Unsolved Problem Tracer" Tiexpressed as:
Ti, Pi←Explorer (IE, Q, T, P, c i) (4)
T,Pwill be updated with Ti,Piwhen the processed chunk is switched.
T←Merge (T, Ti)
P←Merge (P, Pi)(5)
In the workflow, intricate problems are broken down into sub-problems qy
x(theythproblem from xth
chunk), which are supplemented by new information, and then new problems are proposed based
on the newly-acquired information am
n(themthanswer for the nthproblem). The process can be
specified by:
qn
i←Expand (Pi)
aj
n←Answer (qn
x),∀x <=i(6)
Expand function has two types of behaviors: 1) Breaking down the generalized problem into
sub-problems, and 2) expanding new problem entities based on a problem and the corresponding
information. While the loop iterates, the problem-solving focus of large language model is gradually
expanded across the chunks.
3.3 Stage 3: Iterative Selective Replay
Chunking inherently suffers from information loss: interrelated information across different chunks
may be lost if not added to public memory, meaning the model cannot spontaneously notice dangling
nodes in broken reasoning chains beyond its context window. For instance, if reasoning about
1The chunking strategy in Xpanda sets n= 3, L= 10, K= 2000 , α= 0.1, M= 102400 .
4

information in chunk a depends on reasoning about information in chunk b (where b < a ), the model
cannot initiate reasoning for chunk b. To address this, we propose a heuristic replay strategy. The
Unsolved Problem Tracer Testablished in the previous stage is designed precisely for this strategy.
Upon completion of each unidirectional reasoning phase, the Decider is activated to assess whether
the currently gathered information suffices for reasoning and determines whether to initiate replay.
Action, ˆA←Decider (ID, Q, P ) (7)
The replay performs backward search from the last unsolved chunk, applying two heuristics: 1) the
target chunk cannot answer all pending questions, and 2) all relevant information precedes it:
o←max(min( T.values −1,1) ifrev > 0
min(max( T.values + 1, l)otherwise(8)
This dependency forms an AOV network where chunks are vertices and dependencies are edges.
When MRT (max replay times) > x−1, XpandA guarantees to complete dependency resolution A if
LLM can finish information extraction in a chunk unit.
4 Experiment
4.1 Experiment Setup
Datasets. We conduct comprehensive experiments on long context datasets from LongBench [4],
LV-Eval [96],∞Bench [98] and two benchmarks MRCR2,GraphWalks3released by OpenAI
(Table 1):
•Question Answering. HotpotQA [92] is a Wikipedia-based multihop QA dataset. It requires
reasoning across multiple passages to find the answer. NarrativeQA [68] is a dataset focusing on
answering questions based on stories or scripts, including understanding of important elements
such as characters, plots, themes, etc. loogle-MR-mixup andloogle-CR-mixup originate from
LooGLE [ 43] Long-dependency QA task, specifically the Multiple information Retrieval and
Comprehension and Reasoning subtasks and are mixed up with mix up distracting documents and
supporting documents of stepwise growing length.
•Retrieval. OpenAI-MRCR is a long context dataset for benchmarking an LLM’s ability to
distinguish between multiple needles hidden in context. The evaluation consists of retrieval of the
response corresponding to a specific instance from multi-turn synthetic dialogue.
•Code Tasks. We consider two datasets from ∞Bench. Code.Run is a dataset that simulates
multi-step function executions that involve basic arithmetic operations such as addition, subtraction,
and nested function calls. Code.Debug is a multi-choice dataset to do the bug localization of
recomposed repositories sourced from PyPI4.
•Misc. We pick GraphWalks . Graphwalks is designed to require reasoning across multiple
positions in the context and cannot be solved sequentially. In Graphwalks, the model is given a
graph represented by its edge list and asked to perform an operation to search the parent or the
breadth-first search (BFS) results from certain nodes.
LLM Backbones. The backbone models we deploy through include the well-established and popular
open-source model Llama3.1-8B-Instuct [ 26], Qwen2.5-7B-Instruct [ 66] as well as the latest long-
context language model Qwen2.5-7B-Instruct-1M [ 91] model. We choose efficient engine vllm [ 37]
and 4*A100_80G GPU as infrastructures for the experiments to handle immense GPU memory
required for ultra-long input sequences up to 1M tokens.
Baselines. Full-context directly places the query and the long text into the context window where
truncation may occur. RAG is implemented through RAG-Fusion [ 67] by utilizing reciprocal rank
fusion of relevance and word frequency ranking to reach strong and robust retrieval performance. We
introduce previous Agent-based works CoA [ 100] and LongAgent [ 101] to enhance the comprehen-
siveness of our baseline to highlight the improvement of our works.
2https://huggingface.co/datasets/openai/mrcr
3https://huggingface.co/datasets/openai/graphwalks
4https://pypi.org/
5

Table 1: Dataset Statistics . Benchmark datasets from LongBench, ∞Bench, OpenAI.
Longbench LV-Eval ∞Bench OpenAI
HotpotQA NarrativeQA Loogle-CR-mixup Loogle-MR-mixup Code.Run Code.Debug GraphWalks OpenAI-MRCR
Avg len 9151 18409 99.7k 95.4k 75.2k 114.7k 237.9k 223.5k
Max len 16k 65k 256k 256k 80k 200k 1m 1m
Task Type Multi-doc QA Single-doc QA Multi-hop QA Multi-hop QA Code Code Misc Retrieval
Eval Metric F1 F1 keyword-recall-based F1 keyword-recall-based F1 Accuracy Accuracy F1 Seq Match
Metrics. We evaluate the QA tasks and Misc by F1Score [ 4] and Sequence Match Ratio5for
Retrieval, code tasks by exact match . Besides, we introduce fine-grained Progress Score ffrom
AgentBoard [ 54] for quantification of the multi-turn LLM agents’ performance which is denoted by:
rt= max
i,0≤i≤t|G ∩ P i|
|G|(9)
where an overall goal can be done by conjunction g1∧g2, . . . ,∧gmof atomic subgoal G=
{g1, g2, . . . , g m}. While Pi={p1, p2, . . . , p m}express processed subgoals at ithstep.
4.2 Overall Results of XpandA
Table 2: Overall Performance evaluation of XpandA on benchmarks for Question-Answering,
Code, Miscellaneous, and Retrieval, showing XpandA achieved optimal or leading Performance over
full-context, RAG and previous agent-based works across multiple domains (Wilcoxon Signed-Rank
Test: p<0.05 , details in Appendix ??).
Question Answering Code Tasks Misc. Retrieval
LLMs Method HotpotQA Narr.QA CR MIR Code.Debug Code.Run GraphWalks MRCR
Qwen2.5-7B-InstFull-context 48.7 11.9 5.94 4.01 21.1 9.1 14.9 5.95
RAG 53.8 14.2 7.54 5.29 22.1 3.2 25.6 8.8
XpandA 67.3 20.7 15.1 12.6 24.8 21.2 29.8 16.3
LLama3.1-7B-InstFull-context 53.2 12.3 8.3 7.41 20.2 7.7 14.9 2.09
RAG 55.1 15.7 11.8 10.2 25.4 1.4 15.5 12.9
XpandA 69.4 24.5 10.1 11.9 24.3 20.6 13.3 14.5
Qwen2.5-7B-Inst-1MFull-context 56.5 20.1 8.78 7.87 27.8 21.4 29.3 5.19
RAG 57.1 20.3 9.42 7.49 25.3 2.3 27.6 5.57
CoA 62.1 25.3 13.4 9.7 33.3 18.4 30.3 11.9
LongAgent 57.1 21.9 14.2 8.8 34.2 17.7 31.1 9.85
XpandA 64.5 26.4 17.4 13.9 40.5 25.8 32.5 15.3
Question Answering. Table 2 shows the results of QA tasks on three backbone models. XpandA
outperformed the full-context models on all seven datasets, with improvements of up to 18.6% on
HotpotQA, 12.2% on NarrativeQA, 9.16% on LooGLE-CR-mixup, and 8.39% on LooGLE-MIR-
mixup, respectively. Furthermore, XpandA surpassed RAG systems that utilize the same backbone
model as their generator in performance. Notably, XpandA’s performance on QA tasks also exceeded
that of previous agent-based methods, CoA and LongAgent, with gains of up to 5.1%.
Code Tasks, Retrieval & Misc. Table 2 displays the test results for these three classification tasks.
Similarly, XpandA achieved optimal scores on the evaluation metrics for the majority of these tasks.
However, it is noteworthy that RAG + Llama3.1-8B-Instruct delivered the best performance on the
GraphWalks dataset, with XpandA ranking second. This outcome may be attributed to the fact that
the text in GraphWalks is presented in a structured "node->node" format (where each ’node’ is a
hexadecimal hash string). Such a structure potentially allows RAG to retrieve the intended textual
information with greater precision compared to attention-based language models.
4.3 Comparison with RAG and Long Context LLMs
Table 2 shows that XpandA’s optimal performance in eight datasets is greater than the optimal
performance of RAG on the corresponding datasets, with increases of 3.4% 22 .6%respectively. It is
5https://docs.python.org/3/library/difflib.html
6

Figure 2: F1 Score of Full-context ,RAG andXpandA on all three backbone models in datasets
LooGLE-MIR-mixup andLooGLE-CR-mixup of length varing from 16k to 256k. Xpanda achieves
superiority on 5/6datasets. Besides, Xpanda mitigates the performance decay in the case of input
sequences of different lengths and reduces the gap between different generations of backbone models.
noted in Figure 2 that Llama + RAG performs slightly better than Llama + XpandA, possibly due
to Llama’s weaker ability to follow the multi-agent workflow instructions in XpandA, leading to
a performance decrease. Additionally, RAG performs poorly in multi-hop tasks, likely because it
cannot iteratively retrieve and reason about information step-by-step, a core mechanism enabling
XpandA to handle such problems effectively. For tasks involving 16k to 1M tokens, we evaluate
Qwen2.5-7B-Instruct-1M, a long-context language model (LCLM) supporting up to 1 million tokens,
against models with smaller context windows (SCLMs, typically 128k tokens). Figure 2 shows that
while LCLMs outperform SCLMs on 64k–256k token inputs, both exhibit performance degradation
as sequences lengthen. XpandA mitigates this decay, maintaining stable F1 scores for 16k–256k
inputs and enabling SCLMs to match LCLM performance on 256k-token tasks.
4.4 Comparison with Other Multi-Agent Frameworks
Table 3: Comparison with other agent-based works onGraphWalks andMRCR . XpandA
significantly outperforms LongAgent and CoA in both success and progress rates on GraphWalks
and MRCR, demonstrating superior efficiency in information acquisition and structured reasoning for
multi-hop long-context problems.
LongAgent CoA XpandA
success progress/steps success progress/steps success progress/steps
GraphwalksDepth=2 53.2 63.2/6.4 55.6 71.8/3 62.4 ↑6.8 80.2↑8.4/3.9
Depth=4 21.4 27.7/8.5 6.25 21.4/3 38.2 ↑16.8 53.6↑25.9/6.5
Depth=8 6.7 10.2/8.9 2.3 11.4/3 19.4 ↑12.7 42.9↑31.5/7.7
MRCR2 Needles 61.3 63.7/6.1 66.4 68.6/3 71.8 ↑5.4 77.8↑9.2/4.3
4 Needles 25.7 25.4/8.2 29.4 54.3/3 39.1 ↑9.7 68.3↑14.0/5.1
8 Needles 5.1 12.3/8.8 19.4 40.5/3 20.6 ↑1.2 44.8↑4.3/8.2
As shown in Table 3, XpandA outperforms LongAgent and CoA in success rate by up to 16.8% and
progress rate by up to 32.7% on GraphWalks, and by 13.4% and 42.9% respectively on MRCR. It
is noteworthy that, when compared to CoA, which lacks a replay mechanism, XpandA’s primary
additional computational cost is the necessary overhead in operational steps due to its replay process.
The higher progress rate reflects that XpandA more purposefully acquires information when traversing
text chunks and more efficiently identifies the requisite information for each step in multi-hop
problem-solving. The higher success rate signifies that XpandA develops a structured understanding
7

of the collected information, thereby providing better answers rather than relying on superficial or
decontextualized interpretations.
5 Analysis
5.1 Diverse Chunking Strategies Meet Context of Vastly Varying Length
Figure 3: Impact of different chunking strategies
on Xpanda in dataset MRCR (2 Needle). Optimal
static performance occurs when the unit chunk
size is [1
16,1
4]of the input length, while dynamic
chunking controls the proportion within this range
and achieves 18% average improvement.We analyze the impact of chunking strategies
on XpandA’s performance, shown in Figure 3,
via ablation experiments on the MRCR (2
Needles) benchmark (16k–1M sequences),
evaluating retrieval success rate. Static chunking
(2k–128k) shows that when the chunk size
exceeds the input length, XpandA degenerates to
a single-agent system, suffering full-context-like
performance decay. Conversely, overly small
chunks introduce excessive agents, collecting
irrelevant noise and degrading accuracy. Optimal
static performance occurs when unit chunk size
is[1
16,1
4]of the input sequence length.
XpandA’s dynamic chunking adaptively divides
the input sequence into four to eight equal
parts which fall into the range, thus achieving
integrated peak performance of static strategy.
This improves the average retrieval success rate
by18% over the best static variant.
5.2 LLM Reasoning Scheme Ablation
(a) Reasoning scheme (F1 Score).
 (b) Replay strategy (Time).
 (c) Replay strategy (F1 Score).
Figure 4: Ablation studies on NarrativeQA. (a) Comparison of XpandA with alternative planning
schemes. (b) Impact of different replay strategies on average inference time. (c) Impact of different
replay strategies on F1 score. Selective replay in XpandA achieves a strong balance between
performance and efficiency.
To further explore the effectiveness of the question-driven scheme in xpanda, we systematically
remove and modify the component from XpandA to observe the performance changes. Plan-and-
Solve [78] and Fact-and-Reflection [103] can help improve the LLM reasoning performance through
the management of atomic plans or atomic facts, behaving just like the subquestion state tracking in
XpandA, and thus can replace the planning scheme in XpandA without reconstruction of pipeline.
We implement Directly Answer by setting the question tracker P,Tto null value and gathering all
Explorers’ direct answers to be chosen by Decider. As shown in Figure 4a, XpandA outperforms
other reasoning schemes by 4–12% in F1 score of NarrativeQA, which shows that the inclusion of
question-driven scheme in XpandA efficiently reinforces the multi-agent workflow by providing
instruction that is more explicit to identify and explore the context and mitigates the inter-agent
misalignment.
8

5.3 Selective Replay Matters in Question-driven Workflow
Figure 4b, 4c illustrate the performance of XpandA on the GraphWalks dataset (context length
128k) in terms of runtime and F1 score, comparing its standard selective replay mechanism against
three alternative modes: no-replay, random replay, and brute-force replay (the latter executed either
unidirectionally or by alternating directions with each replay). In the random replay setup, the
number of chunks processed by selective replay was first recorded, and then an equivalent number of
unique chunks was randomly selected and replayed sequentially. The experimental results indicate
that selective replay achieves a nearly identical F1 score to that of the best-performing brute-force
bidirectional replay, while utilizing only 56.5% of the latter’s runtime. The superiority in F1 score
is primarily attributed to selective replay’s fine-grained, atomic management of information states,
which enables it to select chunks for replay that are most likely to uncover beneficial information and
drive problem resolution. Furthermore, its enhanced time efficiency can also be ascribed to selective
replay’s ability to determine if the currently collected information is sufficient to answer the question,
thereby enabling early stopping.
5.4 Empirical Inference Time Analysis
Figure 5: XpandA scheme exhibits 1.5x
speedup in inference time compared to SOTA
long context language model on long texts.We analyze the inference time of XpandA on
4×A100_80G GPUs (Figure 5), using Qwen2.5:7B-
Instruct-1M as the backbone to avoid API-related la-
tency. For comparison, we include Llama3.1-Instruct-
1048k [ 62] in full-context mode. XpandA with dy-
namic chunking outperforms all baselines, achiev-
ing a 1.5 ×speedup over Qwen2.5-1M (with SOTA
sparse attention optimization [ 91]) while maintaining
linear complexity with fixed chunk sizes. In contrast,
traditional full-context models (limited to 256k con-
text on 4 ×A100) are significantly slower. Dynamic
chunking also balances inference time and model-call
latency better than static strategies (2k/8k) used in
CoA [ 100] and LongAgent [ 101], which suffer from
accumulated per-agent latency—especially critical
for commercial API calls.
6 Conclusion
In this paper, we propose XpandA, a novel paradigm for multi-agent LLM workflow in long context
language modeling through continuous questioning-answering. It is a plug-and-play, robust, inter-
pretable, and efficient multi-agent system framework that introduces a dynamic chunking strategy
and selective replay in its question-driven workflow to further enhance its performance. Extensive
experiments show that XpandA achieves superior performance in various tasks with a wide span of
context lengths (10k to 1M) compared to Full-context, RAG, and previous agent-based methods. It
is worth noting that through XpandA expansion, SOTA (state-of-the-art) long context models have
seen further improvements in performance (20% metrics improvement and 1.5x speedup). Analysis
indicates that XpandA, through the multi-module coupling of its question-driven workflow, effec-
tively alleviates the failure problem of multi-agent systems, performing better in long-text processing
scenarios.
7 Limitations
Although XpandA provides an excellent paradigm for addressing the weak constraints and operational
inefficiencies of Long Context Multi-Agent Systems in prior research, future directions can address
the following limitations to further enhance its capabilities and efficiency. First, the scalability of
XpandA on models with weak instruction-following abilities could be improved through in-context
learning, simplifying workflow instructions, or exploring more efficient workflows. Second, the
current experiments limit input sequence lengths to under 1M; future work could further investigate
XpandA’s behavior and performance on longer sequences.
9

References
[1]Sahar Abdelnabi et al. Cooperation, competition, and maliciousness: LLM-stakeholders
interactive negotiation. In The Thirty-eight Conference on Neural Information Processing
Systems Datasets and Benchmarks Track , 2024.
[2]Canfer Akbulut et al. All too human? mapping and mitigating the risk from anthropomorphic
ai.Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society , 7:13–26, October
2024.
[3]Krishna Aswani, Huilin Lu, Pranav Patankar, Priya Dhalwani, Xue Tan, Jayant Ganeshmohan,
and Simon Lacasse. Auto-evolve: Enhancing large language model‘s performance via self-
reasoning framework. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Findings of the Association for Computational Linguistics: EMNLP 2024 , pages 13243–13257,
Miami, Florida, USA, November 2024. Association for Computational Linguistics.
[4]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao
Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench:
A bilingual, multitask benchmark for long context understanding. In Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) , pages 3119–3137, Bangkok, Thailand, August 2024. Association for Computational
Linguistics.
[5]Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Trans-
former, December 2020. arXiv:2004.05150 [cs].
[6]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas
Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al.
Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of
the AAAI Conference on Artificial Intelligence , volume 38, pages 17682–17690, 2024.
[7]Egor Bogomolov, Aleksandra Eliseeva, Timur Galimzyanov, Evgeniy Glukhov, Anton Shapkin,
Maria Tigina, Yaroslav Golubev, Alexander Kovrigin, Arie van Deursen, Maliheh Izadi, and
Timofey Bryksin. Long code arena: a set of benchmarks for long-context code models, 2024.
[8]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang,
Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving,
Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.
Improving language models by retrieving from trillions of tokens, 2022.
[9]Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh
Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia,
Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent llm systems fail?, 2025.
[10] Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh
Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Za-
haria, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent llm systems fail? ArXiv ,
abs/2503.13657, 2025.
[11] Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. Walking down the
memory maze: Beyond context limit through interactive reading, 2023.
[12] Junzhe Chen et al. LLMArena: Assessing capabilities of large language models in dynamic
multi-agent environments. In Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 13055–13077, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
[13] Pei Chen, Shuai Zhang, and Boran Han. CoMM: Collaborative multi-agent, multi-reasoning-
path prompting for complex problem solving. In Kevin Duh, Helena Gomez, and Steven
Bethard, editors, Findings of the Association for Computational Linguistics: NAACL 2024 ,
pages 1720–1738, Mexico City, Mexico, June 2024. ACL.
10

[14] Weize Chen et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent
behaviors. In The Twelfth International Conference on Learning Representations , 2024.
[15] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. A
dataset of information-seeking questions and answers anchored in research papers. ArXiv ,
abs/2105.03011, 2021.
[16] Tim Ruben Davidson et al. Evaluating language model agency through negotiations. In The
Twelfth International Conference on Learning Representations , 2024.
[17] Shumin Deng, Ningyu Zhang, Nay Oo, and Bryan Hooi. Towards a unified view of answer
calibration for multi-step reasoning. In Bhavana Dalvi Mishra, Greg Durrett, Peter Jansen, Ben
Lipkin, Danilo Neves Ribeiro, Lionel Wong, Xi Ye, and Wenting Zhao, editors, Proceedings of
the 2nd Workshop on Natural Language Reasoning and Structured Explanations (@ACL 2024) ,
pages 25–38, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[18] Ameet Deshpande et al. Anthropomorphization of ai: Opportunities and risks, 2023.
[19] Jiˇrí Dostál. Theory of problem solving. Procedia - Social and Behavioral Sciences , 174:2798–
2805, 2015. International Conference on New Horizons in Education, INTE 2014, 25-27 June
2014, Paris, France.
[20] Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. Successive prompting for
decomposing complex questions. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang,
editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing , pages 1251–1265, Abu Dhabi, United Arab Emirates, December 2022. Association
for Computational Linguistics.
[21] Yongqi Fan, Hongli Sun, Kui Xue, Xiaofan Zhang, Shaoting Zhang, and Tong Ruan.
MedOdyssey: A Medical Domain Benchmark for Long Context Evaluation Up to 200K
Tokens, June 2024. arXiv:2406.15019 [cs].
[22] Tianyu Fu, Haofeng Huang, Xuefei Ning, Genghan Zhang, Boju Chen, Tianqi Wu, Hongyi
Wang, Zixiao Huang, Shiyao Li, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang.
Moa: Mixture of sparse attention for automatic large language model compression, 2024.
[23] Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao
Peng. Data engineering for scaling language models to 128k context, 2024.
[24] Chaochen Gao, Xing Wu, Qi Fu, and Songlin Hu. Quest: Query-centric data synthesis
approach for long-context scaling of large language model, 2025.
[25] Olga Golovneva, Tianlu Wang, Jason E Weston, and Sainbayar Sukhbaatar. Contextual position
encoding: Learning to count what’s important. ArXiv , abs/2405.18719, 2024.
[26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang,
and et al. Angela Fan. The llama 3 herd of models, 2024.
[27] Junqing He, Kunhao Pan, Xiaoqun Dong, Zhuoyang Song, LiuYiBo LiuYiBo, Qianguosun
Qianguosun, Yuxin Liang, Hao Wang, Enming Zhang, and Jiaxing Zhang. Never lost in the
middle: Mastering long-context question answering with position-agnostic decompositional
training. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) , pages 13628–13642, Bangkok, Thailand, August 2024. Association for Computational
Linguistics.
[28] Sirui Hong et al. MetaGPT: Meta programming for a multi-agent collaborative framework. In
The Twelfth International Conference on Learning Representations , 2024.
[29] Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions
for long document summarization. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer,
Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and
11

Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies , pages
1419–1436, Online, June 2021. Association for Computational Linguistics.
[30] Shankar Kumar Jeyakumar, Alaa Alameer Ahmad, and Adrian Garret Gabriel. Advancing
agentic systems: Dynamic task decomposition, tool integration and evaluation using novel
metrics and dataset. In NeurIPS 2024 Workshop on Open-World Agents , 2024.
[31] Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. LLM-blender: Ensembling large language
models with pairwise ranking and generative fusion. In Proceedings of the Annual Meeting of
the Association for Computational Linguistics , Jul. 2023.
[32] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and
Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues?
InThe Twelfth International Conference on Learning Representations , 2024.
[33] Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Chia-Yuan Chang, and Xia Hu.
Growlength: Accelerating llms pretraining by progressively growing training length, 2023.
[34] Ashutosh Joshi, Sheikh Muhammad Sarwar, Samarth Varshney, Sreyashi Nag, Shrivats
Agrawal, and Juhi Naik. Reaper: Reasoning based retrieval planning for complex rag systems,
2024.
[35] Tomás Kociský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor
Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transac-
tions of the Association for Computational Linguistics , 6:317–328, 2017.
[36] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa.
Large language models are zero-shot reasoners. Advances in neural information processing
systems , 35:22199–22213, 2022.
[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large
language model serving with pagedattention, 2023.
[38] Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building
machines that learn and think like people, 2016.
[39] Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan, Michael
Boratko, Yi Luan, S’ebastien M. R. Arnold, Vincent Perot, Sid Dalmia, Hexiang Hu, Xudong
Lin, Panupong Pasupat, Aida Amini, Jeremy R. Cole, Sebastian Riedel, Iftekhar Naim, Ming-
Wei Chang, and Kelvin Guu. Can long-context language models subsume retrieval, rag, sql,
and more? ArXiv , abs/2406.13121, 2024.
[40] Cheng Li et al. Culturepark: Boosting cross-cultural understanding in large language models,
2024.
[41] Guohao Li et al. CAMEL: Communicative agents for ”mind” exploration of large language
model society. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
[42] Huao Li et al. Theory of mind for multi-agent collaboration via large language models. In
Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing , pages 180–192, Singapore, December
2023. Association for Computational Linguistics.
[43] Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context
language models understand long contexts?, 2024.
[44] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need.
Transactions on Machine Learning Research , 2024.
[45] Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei
Qu, Yangguang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng. Graphreader: Building graph-
based agent to enhance long-context abilities of large language models, 2024.
12

[46] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle
with long in-context learning. ArXiv , abs/2404.02060, 2024.
[47] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez
Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon,
Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich,
Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid
transformer-mamba language model, 2024.
[48] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun
Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, Yuanxing Zhang, Zhuo Chen, Hangyu Guo,
Shilong Li, Ziqiang Liu, Yong Shan, Yifan Song, Jiayi Tian, Wenhao Wu, Zhejian Zhou, Ruijie
Zhu, Junlan Feng, Yang Gao, Shizhu He, Zhoujun Li, Tianyu Liu, Fanyu Meng, Wenbo Su,
Yingshui Tan, Zili Wang, Jian Yang, Wei Ye, Bo Zheng, Wangchunshu Zhou, Wenhao Huang,
Sujian Li, and Zhaoxiang Zhang. A comprehensive survey on long context language modeling,
2025.
[49] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Moore
Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, Yuanxing Zhang, Zhuo Chen, Hangyu Guo,
Shilong Li, Ziqiang Liu, Yong Shan, Yifan Song, Jiayi Tian, Wenhao Wu, Zhejian Zhou, Ruijie
Zhu, Junlan Feng, Yang Gao, Shizhu He, Zhoujun Li, Tianyu Liu, Fanyu Meng, Wenbo Su,
Ying Tan, Zili Wang, Jian Yang, Wei Ye, Bo Zheng, Wangchunshu Zhou, Wenhao Huang,
Sujian Li, and Zhaoxiang Zhang. A comprehensive survey on long context language modeling.
ArXiv , abs/2503.17407, 2025.
[50] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts, 2023.
[51] Xiao Liu et al. Agentbench: Evaluating LLMs as agents. In The Twelfth International
Conference on Learning Representations , 2024.
[52] Zijun Liu et al. A dynamic LLM-powered agent network for task-oriented agent collaboration.
InFirst Conference on Language Modeling , Oct. 2024.
[53] Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran
He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun
Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Neo Y . Zhang, Zhilin
Yang, Xinyu Zhou, Mingxing Zhang, and Jiezhong Qiu. Moba: Mixture of block attention for
long-context llms, 2025.
[54] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan,
Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn
llm agents, 2024.
[55] Ahmed Masry and Amir Hajian. LongFin: A Multimodal Document Understanding Model for
Long Financial Domain Documents, January 2024. arXiv:2401.15050 [cs].
[56] Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc
Najork, Emma Strubell, and Donald Metzler. Dsi++: Updating transformer memory with new
documents, 2023.
[57] Alexander Meinke et al. Frontier models are capable of in-context scheming, 2024.
[58] Meredith Ringel Morris, Jascha Sohl-dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe,
Aleksandra Faust, Clement Farabet, and Shane Legg. Levels of agi for operationalizing
progress on the path to agi, 2024.
[59] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave No Context Behind:
Efficient Infinite Context Transformers with Infini-attention, August 2024. arXiv:2404.07143
[cs].
13

[60] Tuan-Phong Nguyen, Simon Razniewski, and Gerhard Weikum. Cultural commonsense knowl-
edge for intercultural dialogues. In Proceedings of the 33rd ACM International Conference on
Information and Knowledge Management , CIKM ’24, page 1774–1784, New York, NY , USA,
2024. Association for Computing Machinery.
[61] Xuefei Ning et al. Skeleton-of-thought: Prompting LLMs for efficient parallel generation. In
The Twelfth International Conference on Learning Representations , 2024.
[62] Leonid Pekelis, Michael Feil, Forrest Moret, Mark Huang, and Tiffany Peng. Llama 3 gradient:
A series of long context models, 2024.
[63] Ji-Lun Peng et al. A survey of useful llm evaluation. arXiv preprint arXiv:2406.00936 , 2024.
[64] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases
enables input length extrapolation. ArXiv , abs/2108.12409, 2021.
[65] Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning
Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language
Models, January 2024. arXiv:2401.04658 [cs].
[66] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji
Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang
Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5
technical report, 2025.
[67] Zackary Rackauckas. Rag-fusion: A new take on retrieval augmented generation. International
Journal on Natural Language Computing , 13(1):37–47, February 2024.
[68] Tomáš Koˇ ciský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann,
Gábor Melis, and Edward Grefenstette. The NarrativeQA reading comprehension challenge.
Transactions of the Association for Computational Linguistics , TBD:TBD, 2018.
[69] Erfan Shayegani et al. Survey of vulnerabilities in large language models revealed by adversar-
ial attacks. arXiv preprint arXiv:2310.10844 , 2023.
[70] Noah Shinn et al. Reflexion: language agents with verbal reinforcement learning. In Thirty-
seventh Conference on Neural Information Processing Systems , 2023.
[71] Shuzheng Si, Haozhe Zhao, Gang Chen, Yunshui Li, Kangyang Luo, Chuancheng Lv, Kaikai
An, Fanchao Qi, Baobao Chang, and Maosong Sun. GATEAU: Selecting Influential Samples
for Long Context Alignment, February 2025. arXiv:2410.15633 [cs].
[72] Keith E Stanovich and Richard F West. Individual differences in reasoning: Implications for
the rationality debate? Behavioral and Brain Sciences , 23(5):645–665, 2000.
[73] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer
with rotary position embedding. ArXiv , abs/2104.09864, 2021.
[74] Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav
Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. ArXiv ,
abs/2212.10554, 2022.
[75] Mirac Suzgun and Adam Tauman Kalai. Meta-prompting: Enhancing language models with
task-agnostic scaffolding, 2024.
[76] Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng
Bao, Xiangkun Hu, Zheng Zhang, Qian Wang, and Yue Zhang. Novelqa: Benchmarking
question answering on documents exceeding 200k tokens, 2025.
[77] Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. Soft self-consistency
improves language model agents. arXiv preprint arXiv:2402.13212 , 2024.
14

[78] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng
Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large
language models, 2023.
[79] Qineng Wang et al. Rethinking the bounds of LLM reasoning: Are multi-agent discussions the
key? In Proceedings of the Annual Meeting of the Association for Computational Linguistics ,
Aug. 2024.
[80] Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei.
Augmenting language models with long-term memory, 2023.
[81] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in
language models. In The Eleventh International Conference on Learning Representations ,
2023.
[82] Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. arXiv preprint
arXiv:2402.10200 , 2024.
[83] Zhenhailong Wang et al. Unleashing the emergent cognitive synergy in large language models:
A task-solving agent through multi-persona self-collaboration. In Proceedings of the 2024
Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers) , Jun. 2024.
[84] Zhenyi Wang et al. Large language model enabled semantic communication systems, 2024.
[85] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[86] Longyun Wu, Dawei Zhu, Guangxiang Zhao, Zhuocheng Yu, Junfeng Ran, Xiangyu Wong,
Lin Sun, and Sujian Li. LongAttn: Selecting Long-context Training Data via Token-level
Attention, February 2025. arXiv:2502.16860 [cs].
[87] Qingyun Wu et al. Autogen: Enabling next-gen LLM applications via multi-agent conversation,
2024.
[88] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao
Fu, and Song Han. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and
Streaming Heads, October 2024. arXiv:2410.10819 [cs].
[89] Lin Xu et al. Magic: Investigation of large language model powered multi-agent in cognition,
adaptability, rationality and collaboration. In ICLR 2024 Workshop on Large Language Model
(LLM) Agents , 2023.
[90] Xiaoyue Xu, Qinyuan Ye, and Xiang Ren. Stress-testing long-context language models with
lifelong icl and task haystack. ArXiv , abs/2407.16695, 2024.
[91] An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang,
Jianhong Tu, Jianwei Zhang, Jingren Zhou, Junyang Lin, Kai Dang, Kexin Yang, Le Yu, Mei
Li, Minmin Sun, Qin Zhu, Rui Men, Tao He, Weijia Xu, Wenbiao Yin, Wenyuan Yu, Xiafei
Qiu, Xingzhang Ren, Xinlong Yang, Yong Li, Zhiying Xu, and Zipeng Zhang. Qwen2.5-1m
technical report, 2025.
[92] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors,
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing ,
pages 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational
Linguistics.
[93] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and
Karthik R Narasimhan. Tree of thoughts: Deliberate problem solving with large language
models. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
15

[94] Zhangyue Yin et al. Exchange-of-thought: Enhancing large language model capabilities
through cross-model communication. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
pages 15135–15153, Singapore, December 2023. Association for Computational Linguistics.
[95] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda
Xie, Y . X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wen-
feng Liang, and Wangding Zeng. Native Sparse Attention: Hardware-Aligned and Natively
Trainable Sparse Attention, February 2025. arXiv:2502.11089 [cs].
[96] Tao Yuan, Xuefei Ning, Dong Zhou, Zhijie Yang, Shiyao Li, Minghui Zhuang, Zheyue Tan,
Zhuyu Yao, Dahua Lin, Boxun Li, Guohao Dai, Shengen Yan, and Yu Wang. Lv-eval: A
balanced long-context benchmark with 5 length levels up to 256k, 2024.
[97] Jintian Zhang et al. Exploring collaboration mechanisms for LLM agents: A social psychology
view. In Proceedings of the Annual Meeting of the Association for Computational Linguistics ,
Aug. 2024.
[98] Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han,
Zhen Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. ∞Bench: Extending long context
evaluation beyond 100K tokens. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 15262–15277, Bangkok, Thailand, August 2024. Association
for Computational Linguistics.
[99] Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae
Lee, Honglak Lee, and Lu Wang. Small language models need strong verifiers to self-
correct reasoning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of
the Association for Computational Linguistics: ACL 2024 , pages 15637–15653, Bangkok,
Thailand, August 2024. Association for Computational Linguistics.
[100] Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Ö. Arik. Chain
of agents: Large language models collaborating on long-context tasks, 2024.
[101] Jun Zhao, Can Zu, Hao Xu, Yi Lu, Wei He, Yiwen Ding, Tao Gui, Qi Zhang, and Xu-
anjing Huang. Longagent: Scaling language models to 128k context through multi-agent
collaboration, 2024.
[102] Qinlin Zhao et al. CompeteAI: Understanding the competition dynamics of large language
model-based agents. In Agentic Markets Workshop at ICML 2024 , 2024.
[103] Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Tongshuang Wu, and
Jianshu Chen. Fact-and-reflection (FaR) improves confidence calibration of large language
models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Associa-
tion for Computational Linguistics: ACL 2024 , pages 8702–8718, Bangkok, Thailand, August
2024. Association for Computational Linguistics.
[104] Yaru Zhao et al. Lamosc: Large language model-driven semantic communication system
for visual transmission. IEEE Transactions on Cognitive Communications and Networking ,
10(6):2005–2018, 2024.
[105] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan
Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir Radev. QMSum: A new
benchmark for query-based multi-domain meeting summarization. In Kristina Toutanova,
Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan
Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference
of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies , pages 5905–5921, Online, June 2021. Association for Computational
Linguistics.
[106] Denny Zhou, Nathanael Sch"arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale
Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most
prompting enables complex reasoning in large language models. In The Eleventh International
Conference on Learning Representations , 2023.
16

[107] Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V Le, Ed H. Chi,
Denny Zhou, Swaroop Mishra, and Steven Zheng. SELF-DISCOVER: Large language
models self-compose reasoning structures. In The Thirty-eighth Annual Conference on Neural
Information Processing Systems , 2024.
[108] Qihui Zhou, Peiqi Yin, Pengfei Zuo, and James Cheng. Progressive Sparse Attention:
Algorithm and System Co-design for Efficient Attention in LLM Serving, March 2025.
arXiv:2503.00392 [cs].
[109] Benhui Zhuang, Chunhong Zhang, and Zheng Hu. Pose: Suppressing perceptual noise in
embodied agents for enhanced semantic navigation. IEEE Robotics and Automation Letters ,
9:963–970, 2024.
A Proof of Completeness
Hypothetical Premise : the LLM( I,θ) acting an agent can complete the problem decomposition and
reasoning of a specified context chunk within k tokens context window.
Problem Definition : Reasoning based on information distributed in different chunks has a certain
dependency sequence. If chunks are not processed in this sequence, it may lead to the breakage of
the reasoning process. This sequential dependency between reasoning across different chunks can be
reduced to a topological sorting problem of predecessor-successor relationships in an AOV (Activity
On Vertex) network . In the scenario of long text reasoning, the dependency order of chunks can
change along with the variation of the entities of concern, which can be expressed as:
M=
m1,1m1,2··· m1,y
m2,1m2,2··· m2,y
............
mx,1mx,2···mx,y
(10)
where mx,ydenotes the the ythchunk should be dealt with on the mx,ythone in the topological
sequence. For example, {4,2,1,3}means reasoning should be done one-by-one on chunk 3, 2, 4, 1
successively.
Explorer scans through the chunks in one direction and Decider proposes replay when reaching
the last column. The next starting point is set at the chunk, denoted as othchunk, where the last
unanswered problem uis proposed. The starting points of different rows are the same .
o= max
i∈[1,x]
arg max
jui,j
(11)
The Explorer accepts mx,y=1 and then 2, and so on until all chunks are received. Specifically, in
(y−1) rounds of scan direction reverse, the Explorer would accept all items in M.
For the first starting point of reverse:
∀i∈[1, x],∀j > o, m i,j̸=mi,o+ 1 (12)
∀i∈[1, x],∃j0∈[1, o], mi,j0=mi,o+ 1 (13)
The starting point of the second round of reverse can be determined by:
o= min
i∈[1,x]
arg min
jui,j
(14)
For the second starting point of reverse:
∀i∈[1, x],∀j < o, m i,j̸=mi,o+ 1 (15)
∀i∈[1, x],∃j0∈[o, y], mi,j0=mi,o+ 1 (16)
The two processes are mirrored,
Mmir=
m1,ym1,y−1··· m1,1
m2,ym2,y−1··· m2,1
............
mx,ymx,y−1···mx,1
(17)
17

Augment the matrix by concatenate MandMmir,
fM=h
MMmirMMmir···i
x×y2. (18)
Scan from left to right. For every y chunks, at least one chunk is accepted. So all items in M can be
accepted in yscan, namely y−1replay. It’s not non-trivial, for:
∀x, DrA accept M. (19)
B Supplementary Experiments
(a) Full model
 (b) RAG model
 (c) XpandA
Figure 6: Dual-Needle-in-a-Haystack test. (a) Full model. (b) RAG model. (c) XpandA.
The Dual-Needle-in-a-Haystack test is designed to rigorously evaluate a model’s capability to concur-
rently identify, associate, and utilize two distinct pieces of information ("needles") embedded within
a voluminous and distractor-rich textual corpus ("haystack"). This task extends beyond single-needle
retrieval by imposing a more significant challenge on the model’s long-range dependency understand-
ing, information integration, and sustained attention across intensive contexts. This appendix presents
a detailed performance analysis of the Full-Context, RAG, and XpandA methodologies on this task.
The "Length of Haystack", represented on the X-axis of Figure 6, was varied across discrete token
counts from 1k to 128k. Additionally, two distinct "needles" were synthetically generated for each
trial, whose depth distribution is represented on the Y-axis of the heatmaps. For a given trial, the
two needles were semantically independent but both were required to answer the query. The "Depth
Percent of Needles", shown on the Y-axis of the heatmaps, indicates the predefined contextual zones
where the needles were inserted.
Figure 6a presents the performance heatmap for the Full-Context method. The overall average
accuracy achieved was 52.58%. The general trend of degrading accuracy is observed as the haystack
length increases beyond approximately 64k tokens, indicating the increasing difficulty in processing,
retaining, and associating information from both needles when the context window is saturated.
Successfully retrieving and associating two needles becomes more challenging when one or both are
deeply embedded and when their relative distance is large within an extensive context.
The RAG method in Figure 6b achieved an overall average accuracy of 64.01%. As the heatmap
indicates, performance is sensitive to haystack length. While RAG’s retrieval mechanism improves
overall accuracy compared to the Full-Context method (Figure 6a), it exhibits limitations in effectively
locating and leveraging information in increasingly longer documents. Its effectiveness negatively
correlates with document length, suggesting potential retrieval challenges in complex long-context
scenarios.
Our proposed multi-agent framework, XpandA in Figure 6c, achieved an overall average accuracy of
72.53%, representing a significant improvement over the RAG (64.01%) and Full-Context (52.58%)
methods on this task. The heatmap provides compelling visual evidence of XpandA’s enhanced
robustness, effectively mitigating the substantial performance degradation typically observed in
baseline approaches, particularly under challenging conditions such as extended context lengths, deep
needle locations, and considerable relative distances between needles.
18