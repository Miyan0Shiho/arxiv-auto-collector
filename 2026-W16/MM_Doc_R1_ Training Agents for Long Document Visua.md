# MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning

**Authors**: Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, Xuanjing Huang

**Published**: 2026-04-15 07:39:08

**PDF URL**: [https://arxiv.org/pdf/2604.13579v1](https://arxiv.org/pdf/2604.13579v1)

## Abstract
Conventional Retrieval-Augmented Generation (RAG) systems often struggle with complex multi-hop queries over long documents due to their single-pass retrieval. We introduce MM-Doc-R1, a novel framework that employs an agentic, vision-aware workflow to address long document visual question answering through iterative information discovery and synthesis. To incentivize the information seeking capabilities of our agents, we propose Similarity-based Policy Optimization (SPO), addressing baseline estimation bias in existing multi-turn reinforcement learning (RL) algorithms like GRPO. Our core insight is that in multi-turn RL, the more semantically similar two trajectories are, the more accurate their shared baseline estimation becomes. Leveraging this, SPO calculates a more precise baseline by similarity-weighted averaging of rewards across multiple trajectories, unlike GRPO which inappropriately applies the initial state's baseline to all intermediate states. This provides a more stable and accurate learning signal for our agents, leading to superior training performance that surpasses GRPO. Our experiments on the MMLongbench-Doc benchmark show that MM-Doc-R1 outperforms previous baselines by 10.4%. Furthermore, SPO demonstrates superior performance over GRPO, boosting results by 5.0% with Qwen3-8B and 6.1% with Qwen3-4B. These results highlight the effectiveness of our integrated framework and novel training algorithm in advancing the state-of-the-art for complex, long-document visual question answering.

## Full Text


<!-- PDF content starts -->

MM-Doc-R1: Training Agents for Long Document Visual Question
Answering through Multi-turn Reinforcement Learning
Jiahang Lin1‡*, Kai Hu2∗, Binghai Wang1, Yuhao Zhou1, Zhiheng Xi1,
Honglin Guo1,Shichun Liu1,Junzhe Wang1,Shihan Dou1,Enyu Zhou1,
Hang Yan2,Zhenhua Han2†,Tao Gui1†,Qi Zhang1,Xuanjing Huang1
1Fudan University,2Shanghai Qiji Zhifeng Co., Ltd
Abstract
Conventional Retrieval-Augmented Generation
(RAG) systems often struggle with complex
multi-hop queries over long documents due
to their single-pass retrieval. We introduce
MM-Doc-R1, a novel framework that em-
ploys an agentic, vision-aware workflow to
address long document visual question an-
swering through iterative information discov-
ery and synthesis. To incentivize the infor-
mation seeking capabilities of our agents, we
proposeSimilarity-based Policy Optimiza-
tion (SPO), addressing baseline estimation bias
in existing multi-turn reinforcement learning
(RL) algorithms like GRPO. Our core insight
is that in multi-turn RL, the more semanti-
cally similar two trajectories are, the more
accurate their shared baseline estimation be-
comes. Leveraging this, SPO calculates a more
precise baseline by similarity-weighted aver-
aging of rewards across multiple trajectories,
unlike GRPO which inappropriately applies
the initial state’s baseline to all intermediate
states. This provides a more stable and accurate
learning signal for our agents, leading to supe-
rior training performance that surpasses GRPO.
Our experiments on the MMLongbench-Doc
benchmark show thatMM-Doc-R1outper-
forms previous baselines by10.4%. Further-
more,SPOdemonstrates superior performance
overGRPO, boosting results by5.0%with
Qwen3-8B and6.1%with Qwen3-4B. These
results highlight the effectiveness of our inte-
grated framework and novel training algorithm
in advancing the state-of-the-art for complex,
long-document visual question answering.
1 Introduction
Long document visual question answering presents
a challenging yet highly practical research prob-
lem, primarily due to the difficulty of effectively
*Equal contributions.‡Work done during an internship
at Shanghai Qiji Zhifeng Co., Ltd.†Corresponding authors:
hzhua201@gmail.com, tgui@fudan.edu.cn.
ExtractTOCLOOPTool resultParserMarkdown
Picture
Doc
Read
Search
RetrievalTool result:
… QWEN2.5-VLresult:
Query
Planner
SeekerPage ID:13 Score:4.70 Snippet:content_13 
AnswerCall toolPage ID:72 Score:3.82 Snippet:content_72 
MM-Doc-R1Visual Elements summary: The image contains … Query-Relevant Information: The chart shows … 
Figure 1: Introduction to MM-Doc-R1. MM-Doc-R1
employs a seeker for iterative key information retrieval
within documents, leveraging a VLM (Visual Language
Model) as a reading tool to ensure accurate processing
of visual elements.
identifying and extracting salient information from
lengthy, multi-page documents (Van Landeghem
et al., 2023; Appalaraju et al., 2021; Ma et al.,
2024). Existing work is always based on Retrieval-
Augmented Generation (RAG), where textual or
visual content is encoded into embeddings, and rel-
evance is determined by similarity scores with re-
spect to the original query (Peng et al., 2024; Lewis
et al., 2020; Han et al., 2025). These approaches
typically rely solely on the initial user query for re-
trieval, which limits their effectiveness in handling
multi-hop questions that require iterative informa-
tion gathering across multiple document segments.
To address the limitations of existing models in
multi-turn information retrieval from documents,
we proposeMM-Doc-R1, a novel framework that
integrates a vision-aware long document question
answering workflow with an end-to-end multi-turn
reinforcement learning algorithm. As illustrated
in Figure 1, our workflow comprises three spe-
cialized agents. First, a planner generates an ini-
tial information-seeking plan by parsing the docu-
ment’s table of contents. Following this, a seeker
acts as a tool-driven agent, performing multi-turn
information retrieval through iterative interactionsarXiv:2604.13579v1  [cs.CL]  15 Apr 2026

with the document to gather relevant evidence. The
seeker utilizes a “search” tool for text-based re-
trieval and a “read” tool, powered by a vision-
language model (VLM), to extract visual details
from specific pages. This iterative decomposition
of sub-questions and strategic tool invocation en-
ables the seeker to precisely access relevant pages
and retrieve accurate information. Finally, the col-
lated relevant information is fed into an answer
agent to generate the ultimate response. To en-
hance the agents’ information-seeking capabilities
and refine their decision-making throughout this
iterative workflow, we employMulti-turn Rein-
forcement Learningto train our agents.
In multi-turn reinforcement learning,
GRPO (Song et al., 2025a,b) is commonly
employed. It operates by first generating a com-
plete trajectory through rollout and then computing
the advantage as the difference between the total
accumulated reward and a baseline. However,
GRPO estimates this baseline only from the
initial state’s rollout, which is then inappropriately
applied to intermediate states. This introduces sig-
nificant estimation bias in those intermediate steps.
To tackle this problem, we are introducing SPO, a
new multi-turn RL algorithm built on GRPO. Our
core idea is this: the more semantically similar
two trajectories are, the greater their overlap in
intermediate states. This increased overlap directly
leads to a more accurate baseline estimation.
SPO capitalizes on this by weighting rewards
based on trajectory similarity, yielding a more
accurate and consistent baseline estimation. This
enhanced baseline effectively reduces variance and
guides the learning process toward more precise
convergence, thereby significantly boosting the
agent’s information-seeking ability in multi-turn
RL training.
Extensive experiments on the MMLongbench-
doc benchmark demonstrate that our method,MM-
Doc-R1, outperforms previous RAG baselines by
10.4%. Furthermore, our proposedSPOapproach
delivers substantial improvements overGRPO,
yielding a6.1%performance increase with Qwen3-
4B and a5.0%performance increase with Qwen3-
8B. These results collectively underscore the supe-
rior capability of MM-Doc-R1 in handling complex
long-document and visually-rich question answer-
ing tasks. Our key contributions are summarized
as follows:
•We proposeMM-Doc-R1, a novel end-to-endframework for long document visual ques-
tion answering that integrates a vision-aware
workflow with multi-turn reinforcement learn-
ing. Our framework empowers agents with
iterative information discovery and synthesis,
which significantly boosts retrieval accuracy
and ultimately improves answering precision
in complex document understanding.
•We introduceSimilarity-based Policy Opti-
mization (SPO), a new reinforcement learn-
ing algorithm specifically developed to en-
hance agents’ information-seeking capabili-
ties within our framework. This approach pro-
vides more stable and accurate baseline esti-
mates for agents in multi-turn settings, thereby
enabling faster learning convergence and im-
proved overall performance.
•We validate the effectiveness of MM-Doc-
R1 through extensive experiments. Our ap-
proach consistently improves the performance
of Qwen3 models with 4B and 8B parame-
ters across various subsets of MMLongBench-
Doc, achieving overall superior results com-
pared to existing baselines.
2 Related Work
2.1 DocVQA Task
The Document Visual Question Answering
(DocVQA) task needs models to answer questions
by jointly reasoning over both textual and visual
information present in documents (Mishra et al.,
2019; Suri et al., 2024; Gao et al., 2023). Various
approaches extract visual information from figures
as a means to process different modalities (Memon
et al., 2020; Wu et al., 2024). Early research pri-
marily concentrated on extracting concise answers
from single, short documents (Mathew et al., 2021).
However, with the advent of large language models,
addressing QA tasks involving multiple or lengthy
documents has become a significant challenge (Yu
et al., 2024; Tanaka et al., 2025). Some bench-
marks like MMLongbench-Doc (Ma et al., 2024)
and LongDocURL (Deng et al., 2024) focus on the
long document question answering. One approach
to tackle this involves employing Optical Character
Recognition (OCR) and text-based retrieval to iden-
tify the most relevant document chunks (Khattab
and Zaharia, 2020; Zhang et al., 2024). Another
method utilizes visual encoders to obtain visual em-
beddings, which are then used for retrieval based

Stage1: Planning<think>step-by-step reasoning</think>Here is my proposed initial plan.(1)Use Search Tool with keywords [“disintegration time”, “gastric emptying”] (2)Read Page for the page(s) containing \"Subject 2 Fasted\" and \"Subject 2 Fed\"(3)…
Planner
Query
AnswerAgent<think>reasoning</think>```tool_code<search>[“keyword1”,“keyword2”]</search>```<think>reasoning</think>```tool_code<read>[{“page_ids”: [44],“query”: “mean disintegration time fasted”}, …]</read>``` <think>reasoning</think>```tool_code\n<FINISH>\n```
<search_results>Search Results for sub query: disintegration time valuesPage ID: 35, 44BM 25 Retrieval Score: 1.7Snippet: content<\search_results>
<read_results>Read Query:   Read Sub Query: Read Page IDs: [44] Page Summarization: •Visual Elements Summary: The image contains…•Query-Relevant Information : The chart shows …</read_results>
BM25 text retrievalDoc OCRQWEN2.5-VL Reading
<think>reasoning</think>My answer is …<document_toc>1.Introduction 2.Related Work3.System Design …</document_toc>
Tool results+
TOCStep1Step2
search_resultsStage2: Executing
StepT⋮
read_resultsTool Results
⋮Figure 2: Detailed framework of MM-Doc-R1. The framework operates in three sequential stages. First, the planner
module formulates a reasoning plan based on the preprocessed document TOC and the user query. Subsequently,
the information seeker executes a multi-turn retrieval and reading process, leveraging the “search" and “read" tools
to gather relevant information. Finally, the collected knowledge is integrated by the answer agent , which generates
a coherent and contextually accurate response.
on a query’s embedding; a recent example of this
is ColPALI (Faysse et al., 2024). The embedding
retrieval method proves to be a highly valuable
tool, finding application in multi-agent systems.
Consequently, several agentic approaches, such as
M3docrag (Cho et al., 2024) and MDocagent (Han
et al., 2025), have leveraged multi-agent systems
to solve the DocVQA problem. These systems in-
tegrate text embedding-based RAG (Retrieval Aug-
mented Generation) and image embedding-based
RAG through different agents. By fostering coop-
eration between Vision-Language Models (VLMs)
and Large Language Models (LLMs), these sys-
tems aim to achieve superior results in the DocVQA
task.These methods primarily address single-hop
questions. However, multi-hop questions necessi-
tate a multi-turn approach and the generation of
concise sub-queries. Our proposed method focuses
on resolving this particular challenge.
2.2 Retrieval-Augmented Generation
RAG (Retrieval Augmented Generation) frame-
works significantly enhance Large Language Mod-
els (LLMs) by integrating external knowledge
retrieval (Zhao et al., 2024; Jiang et al., 2023),thereby enabling the generation of more factually
grounded responses (Han et al., 2023). While tradi-
tional retrieval methods, such as BM25 and dense
retrievers like BGE-M3 (Chen et al., 2024), ex-
cel at lexical or semantic matching, they often en-
counter limitations when tackling complex multi-
hop queries (Gao et al., 2023). Recent advance-
ments have led to multi-modal extensions, exem-
plified by models like ColQwen2.5, which builds
upon ColPALI (Faysse et al., 2024), that incor-
porate visual features to enrich the retrieval pro-
cess. However, these models still face challenges
in terms of iterative refinement for complex ques-
tion answering. Furthermore, some web search
retrieval methods, such as Search-R1 (Jin et al.,
2025) and Deepreasearcher (Zheng et al., 2025),
employ multiple retrieval steps to address long-
context QA problems. Yet, these methods primar-
ily focus on the text modality and rely on generic
web search tools. In contrast, our proposed method
distinguishes itself by combining both visual and
text modalities through the integration of a special-
ized “visually-read" tool. This unique multimodal
approach enables our method to address a broader
range of problems within the visually-rich Question

Answering (VQA) domain.
2.3 Reinforcement Learning Algorithm
The earliest and most widely adopted Reinforce-
ment Learning method for training Large Language
Models is Proximal Policy Optimization (Schul-
man et al., 2017). PPO utilizes a critic model
to estimate the baseline, which represents the av-
erage reward of all possible actions in a given
state. Recently, with the development of mod-
els like DeepSeek-R1 (Guo et al., 2025), Group
Relative Policy Optimization (GRPO, Shao et al.
(2024)) is gaining increasing traction for training
LLMs. Compared to PPO, GRPO estimates the
baseline using a group mean reward, thereby elim-
inating the need for a separate critic model. This
can lead to significant savings in computational
resources and memory. Other methods, such as
REINFORCE++ (Hu, 2025), also propose alter-
native baseline estimations, like using the batch
mean reward. More recently, VRPO (Zhu et al.,
2025) revisits value modeling for robust RL train-
ing under noisy supervision. This trend highlights
ongoing research into more efficient and stable RL
algorithms for LLM alignment.
3 Method
In this section, we present the core components of
our proposed MM-Doc-R1 framework. First, we
design an autonomous, structured agentic workflow
to flexibly process multi-page documents. This
workflow consists of three key agents: a planner,
a seeker, and an answer agent. This design al-
lows our agents to iteratively search for crucial
information within documents. Secondly, we in-
troduce an innovative reinforcement learning algo-
rithm calledSimilarity-basedPolicyOptimization
(SPO). We use SPO to train our agents from scratch,
and this training method significantly enhances our
agents’ information-seeking capabilities, empow-
ering them to efficiently locate key information in
documents. Our workflow is illustrated in Figure 2.
3.1 Agentic Workflow
To accurately and comprehensively respond to com-
plex questions that necessitate integrating infor-
mation from diverse sources or performing multi-
step reasoning, our agent adheres to a meticulously
structured workflow. This sequential yet itera-
tive process not only mimics human cognitive ap-
proaches to problem-solving but also endows themodel with dynamic planning and essential self-
correction capabilities. The entire workflow un-
folds across five distinct, interconnected phases.
3.1.1 Document Parsing
When a document is received, we first use the OCR
tool Doc2X1to parse it, extracting tables, figures,
and text into Markdown format. After we get the
OCR output, we create a table of contents (TOC)
by the markdown text, providing a main abstract
of the document. Subsequently, the document is
chunked by pages. This process generates both a
TOC and a list of chunks. Each chunk is derived
from an OCR result and includes a corresponding
image, which is a screenshot of the relevant page.
3.1.2 Initial Planning
Following document parsing, a planning agent is
employed to formulate a global strategy. Its in-
puts include the TOC and detected figures’ caption.
The planner is responsible for breaking down the
initial query into granular sub-queries and orches-
trating the selection of necessary tools. This global
perspective, integrated into the seeker’s history,
critically informs and guides the agent’s decision-
making process.
3.1.3 Toolbox
Our agent is equipped with two specialized tools
to handle multi-modal documents. The “read" tool
takes a page ID and a sub-query, using a Vision-
Language Model (VLM) to extract and interpret
relevant text, charts, and images from the specified
page. This enhances our framework’s ability to
process visual information, enabling more accurate
understanding and reasoning of multimodal content
within documents. The “search" tool uses BM25
with a sub-query to perform fast, keyword-based
retrieval across the document, returning the Top-K
most relevant text snippets. BM25 is chosen for its
efficiency and effectiveness in rapid information
lookup.
3.1.4 Self-Refine & Information Seeking
This phase forms the core iterative loop of MM-
Doc-R1. The agent dynamically decomposes the
complex question into executablesub-queriesand,
in each iteration, selects and invokes appropriate
Tools(read or search) to retrieve needed informa-
tion. It generates a sub-query, chooses a tool, exe-
cutes it with parameters, and analyzes the output. If
1https://github.com/NoEdgeAI/pdfdeal

QTrajectory2Trajectory1
TrajectoryNO1
ONO2
… …Reward 
modelR1
RNR2A1
ANA2
… …Refence 
model
Policy 
modelSPO/GRPO
GRPO  
×R1
RN… =Bi1
𝑁
1
𝑁B1
BNB2 - =
TiT1
T2
TN…S1
S2
SN…W1
W2
WN…Normalize ×R1
R2
RN…=BiSPO BaselineFigure 3: SPO and GRPO’s advantage estimation. The bottom panel shows the baseline computation of SPO and
GRPO.
the sub-query remains unresolved, the agent refines
its plan and iterates. This process enables adaptive,
step-by-step reasoning through complex document
queries.
3.1.5 Answer Generation
After gathering sufficient information, the agent
enters theanswer generationphase. It synthesizes
all retrieved context—including tool outputs and
OCR text—enabling the LLM to reason over the ev-
idence and produce a final, accurate, and coherent
answer that fully addresses the original query.
3.2 Training Method: Similarity-based Policy
Optimization (SPO)
To enable our agent to learn optimal strategies
for tool invocation, sub-query decomposition, and
overall workflow navigation, we propose a novel
reinforcement learning algorithm called Similarity-
based Policy Optimization (SPO). SPO serves as
a significant enhancement to existing policy opti-
mization techniques, particularly improving upon
the GRPO algorithm by providing a more precise
and stable learning signal for agentic decision-
making. Figure 3 shows the difference between
SPO and GRPO.
3.2.1 Group Relative Policy Optimization
GRPO is a popular algorithm proposed by
DeepSeek; it is an improvement over PPO. The
loss function of GRPO isJGRPO(θ) =E q,o1:G1
GGX
i=11
|oi||oi|X
t=1min
rt(θ)ˆAi,t,
clip(r t(θ),1−ε,1 +ε) ˆAi,t
−βD KL[πθ∥πref]
(1)
where rt(θ) =πθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)is the importance
sampling ratio, ˆAi,tis the advantage estimate, ε
is the clipping threshold, and βcontrols the KL
regularization strength.
In multi-turn RL, GRPO calculates the advan-
tage by comparing the reward of the current gen-
erated policy ( Ti) with the average reward of all
policies within the batch. The advantage function
for GRPO is given by:
AGRPO(Ti) =R(T i)−1
NNX
j=1R(T j)(2)
Here, R(T i)represents the reward obtained for
the current trajectory Ti, andNis the total number
of trajectory in the current group. Note that in a
multi-turn reinforcement learning training process,
the entire response in the same trajectory receives
the same reward.
In traditional single-turn or fixed-environment
RL settings, it’s typically assumed that all re-
sponses within a group share the same initial con-

ditions, often referred to as the “prompt". How-
ever, this fundamental assumption becomes prob-
lematic in multi-turn RL training. As training pro-
gresses across multiple interaction rounds, the di-
versity among individual trajectories rapidly in-
creases. This divergence means that even within
the same group, two trajectories can evolve under
significantly different intermediate states or con-
texts. Consequently, one cannot directly assume
that the rewards derived from these disparate trajec-
tories are directly comparable, as their underlying
conditions are no longer uniform.
3.2.2 Similarity-based Policy Optimization
In multi-turn reinforcement learning, a trajectory
is typically modeled as a sequence of states and
actions:
s0→a 0→s 1→a 1→ ··· →s T→a T,(3)
where stdenotes the state (eg. prompt and the en-
vironment feedback) at step t, andatis the agent’s
response. In GRPO, ntrajectories are sampled
in parallel from the same initial state s0, and a
shared baseline V(s 0)is used for advantage esti-
mation. While computationally efficient, this ap-
proach assumes that all trajectories remain seman-
tically aligned throughout the interaction, which
is an assumption that quickly breaks down as re-
sponses diverge over turns.
As dialogue progresses, even trajectories starting
from the same prompt can evolve into significantly
different contexts due to stochastic generation and
feedback dynamics. Consequently, their value es-
timates V(st)become increasingly heterogeneous.
Using a single baseline derived from s0introduces
high bias in advantage estimation, especially for
later turns, leading to unstable policy updates.
To address this, we propose Similarity-based
Policy Optimization (SPO), which replaces the uni-
form baseline with a semantically weighted average
over rewards in the batch. The key insight is that
trajectories with similar semantic content are more
likely to share underlying value structures and thus
should serve as better baselines for one another.
The advantage in SPO is defined as:
ASPO(Ti) =R(T i)−NX
j=1wijR(T j),(4)
where R(T i)represents the total reward of trajec-
toryTi. The weight wijreflects the semantic simi-larity between trajectory TiandTj, and is normal-
ized across the group:
wij=similarity(emb(T i),emb(T j))PN
k=1similarity(emb(T i),emb(T k)),(5)
Here, emb(T) denotes the dense vector representa-
tion of trajectory T, computed using the BGE-M3
model (Chen et al., 2024), which is frozen during
training. The similarity function computes the co-
sine similarity between embeddings.
By constructing a dynamic, content-aware base-
line, SPO reduces estimation variance and miti-
gates bias caused by trajectory divergence. It ef-
fectively prioritizes comparisons within semanti-
cally coherent groups, yielding more accurate ad-
vantages and stabler learning, particularly in long-
horizon, multi-turn settings where traditional base-
lines fail.
3.2.3 Reward Function Design
We employ a comprehensive set of metrics to eval-
uate the performance of all models, reflecting both
answer accuracy and the ability to correctly identify
unanswerable questions. Our primary reward sig-
nal for reinforcement learning is theFinal Reward,
calculated by summing theread page Recalland a
Correctness Scorefor the final answer. The Final
Reward is defined as:
Final Reward=Recall+Correctness Score
The read page Recall measures how effectively the
system directs the agent to relevant pages contain-
ing the answer, defined as:
Recall=|set(read pages)∩set(evidence pages)|
|set(evidence pages)|
The Correctness Score for the final answer is deter-
mined by following the answer judgment method-
ology from MMlongbench-doc (Ma et al., 2024),
specifically, we leverageQwen2.5-72B-Instructto
extract a precise answer, and then perform a match-
ing calculation against the ground-truth answer to
derive this score.
4 Experiments
This section details the experimental setup, evalua-
tion metrics, and comprehensive results demon-
strating the efficacy of our proposedMM-Doc-
R1framework, particularly highlighting the per-
formance gains achieved through our RL algorithm
SPO.

Evidence Modality Evidence Count Overall
MethodText Layout Chart Table Figure Single Multi Unans. ACC F1
Human Performance
Human — — — — — — — — 65.8 66.0
Upper Bounds (Ground-Truth Evidence)
Qwen2.5-VL-7B 33.8 38.7 31.8 32.3 34.1 46.6 20.5 92.8 46.8 41.7
Qwen3-8B 44.3 37.7 25.7 59.3 22.8 42.7 35.7 89.2 49.6 46.9
RAG Baselines
BM25 30.9 23.4 22.3 28.5 9.2 30.7 14.1 88.3 36.4 31.0
BGE-M3 32.0 20.8 21.7 40.3 14.7 35.4 18.5 84.3 39.3 34.8
Colqwen 27.8 25.0 16.5 22.4 23.7 33.9 13.7 82.5 36.5 31.2
Mdoc agent 33.1 29.3 25.8 32.6 30.0 43.7 18.4 43.4 35.0 33.3
M3doc RAG 39.2 26.7 29.8 39.0 32.0 50.3 21.2 40.7 38.4 36.7
Ours: MM-Doc-R1
Qwen3-4B 28.9 23.8 23.1 35.3 22.4 37.1 18.6 72.2 37.7 32.2
+GRPO 36.3 35.2 29.5 40.2 27.1 44.5 22.5 58.7 39.9 36.3
+SPO 41.1 37.2 35.6 47.2 30.5 50.5 27.5 68.0 46.0 41.2
Qwen3-8B 39.6 37.6 37.8 45.6 27.3 47.3 27.5 73.9 45.7 41.5
+GRPO 40.9 36.8 35.7 49.9 28.3 48.5 30.2 60.5 44.7 41.9
+SPO46.2 38.1 40.8 52.8 35.9 56.0 31.268.249.7 46.1
Table 1: Performance comparison on the MMLongBench-Doc dataset (1,082 questions). The evaluation includes
text-based RAG methods (BM25, BGE-M3), multi-modal RAG (ColQwen2.5-7B-multilingual), and agent-based
systems (Mdoc Agent, M3doc RAG). Text-based methods use the top-4 retrieved pages and Qwen3-8B for answer
generation. Multi-modal RAG uses the top-4 retrieved image pages and Qwen2.5-VL-7B. Agent-based methods
operate over the top-5 retrieved pages, we use Qwen2.5-VL-7B as the VLM and Qwen3-8B as the LLM. Metrics
include overall Accuracy (ACC) and F1, as well as performance on sub-categories by evidence modality (Text,
Layout, Chart, Table, Figure), number of required evidences (Single, Multi), and unanswerable questions. Best
scores among baselines and our methods are marked in bold, second-best in underline, considering only “RAG
Baselines", and “Ours MM-Doc-R1" sections.
4.1 Experimental Setup
The MMLongbench-Doc dataset serves as our pri-
mary benchmark for evaluating multi-modal long
document question answering. This dataset fea-
tures complex documents requiring multi-step rea-
soning and spans various content types. For RL
training, we use a subset of 300 samples from Long-
DocURL as the validation set, and the remaining
data as the training set.
4.2 Results and Discussion
Our experimental results, summarized in Table 1,
clearly demonstrate the superior performance of
MM-Doc-R1, particularly when trained with SPO,
across various dimensions of the MMLongBench-
Doc benchmark. As shown in the table, even in
its untrained form, MM-Doc-R1 with Qwen3-8B
achieves an overall Accuracy (ACC) that surpasses
the current state-of-the-art baseline by 10.4%. In
single-evidence questions—where the answer can
be derived from a single retrieved page—our
method outperforms the best existing approach by
5.7%. More notably, on multi-evidence questions
that require information integration across multi-ple pages and often involve multi-hop reasoning,
MM-Doc-R1 achieves a gain of 10.0%, highlight-
ing its strong capability in handling complex, long-
context reasoning tasks. Furthermore, our frame-
work consistently achieves top performance across
all evidence modalities, including Text, Layout,
Chart, Table, and Figure, demonstrating its robust-
ness and effectiveness in processing heterogeneous
multi-modal document content.
In terms of reinforcement learning, SPO exhibits
clear advantages over GRPO. When applied to the
Qwen3-4B model, SPO improves overall accuracy
by 6.1% compared to GRPO; on Qwen3-8B, the im-
provement reaches 5.0%. This consistent gain con-
firms the effectiveness of our semantic similarity-
based advantage estimation in mitigating the bias
introduced by trajectory divergence in multi-turn
dialogue settings. As illustrated in Figure 4, SPO
not only achieves higher final performance but also
demonstrates more stable and faster convergence
during training. These results validate our hypothe-
sis that leveraging semantically aligned trajectories
as dynamic baselines leads to more accurate pol-
icy updates, especially in long-horizon, multi-step

Figure 4: Comparison of SPO and GRPO in Training; the base model in this figure is Qwen3-8B.
Model ACC F1 Unanswerable
MM-Doc-R1 (Qwen3-8B)45.7 41.573.9
w/o TOC 44.3 39.9 73.5
w/o Read pages’ OCR 43.4 38.7 74.7
w/o VLM Read 42.1 37.381.6
Table 2: Ablation study of MM-Doc-R1 components on
MMLongBench-Doc, using Qwen3-8B. All metrics are
in %.
reasoning scenarios.
4.3 Ablation Study
In order to assess the individual contributions of
each component within MM-Doc-R1, we con-
ducted an ablation study, summarized in Table 2.
The removal of any component consistently led to
a degradation in performance, underscoring the vi-
tal role of the table of contents (TOC), page OCR
reading, and VLM reading modules in enhancing
the model’s overall efficacy. These findings collec-
tively emphasize the synergistic effectiveness of all
components within MM-Doc-R1.
4.4 Recall Analysis
Table 3 presents the Recall performance of various
models. Our MM-Doc-R1 framework achieves a
recall of 66.3% when just read 3.35 pages, signifi-
cantly outperforming traditional BM25 (42.0%),
embedding-based BGE-M3 (51.3%), and multi-
modal Colqwen-2.5-VL-7B (65.4%). Notably,
MM-Doc-R1 achieves this highest recall while re-
quiring an average of just 3.35 pages read, under-
scoring the effectiveness of its integrated agentic
framework. This superior recall ensures the agent
accesses more relevant evidence, crucial for accu-
rate question answering in multi-modal contexts.Model average pages Recall
BM25 5 42.0
BGE-M3 5 51.3
Colqwen-2.5-VL-7B 5 65.4
MM-Doc-R1(SPO) 3.35 66.3
Table 3: Recall Performance
5 Conclusion
We presentedMM-Doc-R1, addressing the limita-
tions of single-pass RAG in long-document visual
QA through an iterative, agentic workflow. Cen-
tral to our framework isSimilarity-based Policy
Optimization (SPO), which mitigates baseline es-
timation bias in multi-turn RL by leveraging se-
mantic trajectory similarity for precise reward av-
eraging. Our experiments on MMLongbench-Doc
demonstrate that MM-Doc-R1 outperforms prior
baselines by 10.4%, with SPO significantly surpass-
ing standard GRPO across multiple model scales.
These results validate that integrating vision-aware
reasoning with trajectory-informed RL effectively
advances the state-of-the-art for complex, multi-
modal information discovery.
6 Limitations
While MM-Doc-R1 and the SPO algorithm demon-
strate substantial improvements in long-document
visual reasoning, several inherent limitations
should be noted.
First, the effectiveness of our framework ispar-
tially dependent on the quality of initial docu-
ment parsing. Although MM-Doc-R1 employs a
robust seeker to navigate content, its planning and
retrieval efficiency still rely on the fidelity of the
structural metadata (such as the Table of Contents)
and the OCR accuracy of the ingestion engine. In
scenarios where documents are severely degraded

or lack standard hierarchical formatting, the perfor-
mance may be constrained by the noise introduced
during the pre-processing stage.
Second, our current study primarily focuses on
understanding static documents. The proposed
multi-turn workflow and reinforcement learning
strategy are optimized for fixed document formats
like PDFs and high-resolution images. However,
real-world digital documents can be dynamic or
semi-structured (e.g., interactive reports or web-
based content). The adaptation of the agentic
seeker to environments where document content or
layouts might dynamically evolve during interac-
tion remains an area for future exploration.
Finally, while SPO effectively reduces baseline
estimation bias, its current validation is centered
on English-centric benchmarks. The cross-lingual
robustness and generalizability of the seeker across
diverse linguistic structures have yet to be exten-
sively investigated.
Acknowledgments
The authors wish to thank the anonymous review-
ers for their helpful comments. This work was
partially funded by National Key R&D Program
of China No.2025ZD0215702, National Natural
Science Foundation of China (No. 62576106,
62476061, 62376061).
References
Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota,
Yusheng Xie, and R Manmatha. 2021. Docformer:
End-to-end transformer for document understanding.
InProceedings of the IEEE/CVF international con-
ference on computer vision, pages 993–1003.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-
Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, and 1 others. 2024. Longdocurl: a com-
prehensive multimodal long document benchmark
integrating understanding, reasoning, and locating.
arXiv preprint arXiv:2412.18424.Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2024. Colpali: Efficient document retrieval with
vision language models. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li,
Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A
multi-modal multi-agent framework for document
understanding.arXiv preprint arXiv:2503.13964.
Yikun Han, Chunjiang Liu, and Pengfei Wang. 2023. A
comprehensive survey on vector database: Storage
and retrieval technique, challenge.arXiv preprint
arXiv:2310.11703.
Jian Hu. 2025. Reinforce++: A simple and efficient
approach for aligning large language models.arXiv
preprint arXiv:2501.03262.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024. Mmlongbench-
doc: Benchmarking long-context document un-
derstanding with visualizations.arXiv preprint
arXiv:2407.01523.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawa-
har. 2021. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Jamshed Memon, Maira Sami, Rizwan Ahmed Khan,
and Mueen Uddin. 2020. Handwritten optical charac-
ter recognition (ocr): A comprehensive systematic lit-
erature review (slr).IEEE access, 8:142642–142668.
Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh,
and Anirban Chakraborty. 2019. Ocr-vqa: Visual
question answering by reading text in images. In
2019 international conference on document analysis
and recognition (ICDAR), pages 947–952. IEEE.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.arXiv preprint arXiv:2408.08921.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Y Wu, and 1 others. 2024. Deepseek-
math: Pushing the limits of mathematical reason-
ing in open language models.arXiv preprint
arXiv:2402.03300.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025a. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng
Chen, and Yuhuan Wu. 2025b. Yingqian min, wayne
xin zhao, lei fang, and ji-rong wen. r1-searcher++:
Incentivizing the dynamic knowledge acquisition
of llms via reinforcement learning.arXiv preprint
arXiv:2505.17005.
Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A Rossi, and Dinesh
Manocha. 2024. Visdom: Multi-document qa
with visually rich elements using multimodal
retrieval-augmented generation.arXiv preprint
arXiv:2412.10704.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 24827–24837.
Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski,
Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anck-
aert, Ernest Valveny, and 1 others. 2023. Documentunderstanding dataset and evaluation (dude). InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision, pages 19528–19540.
Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang,
and Qingyun Wu. 2024. Stateflow: Enhancing llm
task-solving through state-driven workflows. InFirst
Conference on Language Modeling.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and 1 others. 2024. Vis-
rag: Vision-based retrieval-augmented generation
on multi-modality documents.arXiv preprint
arXiv:2410.10594.
Junyuan Zhang, Qintong Zhang, Bin Wang, Linke
Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Con-
ghui He, and Wentao Zhang. 2024. Ocr hin-
ders rag: Evaluating the cascading impact of ocr
on retrieval-augmented generation.arXiv preprint
arXiv:2412.02592.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-augmented generation for ai-generated con-
tent: A survey.arXiv preprint arXiv:2402.19473.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.arXiv
preprint arXiv:2504.03160.
Dingwei Zhu, Shihan Dou, Zhiheng Xi, Senjie Jin, Guo-
qiang Zhang, Jiazheng Zhang, Junjie Ye, Mingxu
Chai, Enyu Zhou, Ming Zhang, Caishuang Huang,
Yunke Zhang, Yuran Wang, and Tao Gui. 2025. Vrpo:
Rethinking value modeling for robust rl training un-
der noisy supervision.Preprint, arXiv:2508.03058.

A Appendix
A.1 Evaluation Metrics
We adopt the exact same evaluation protocol as
MMLongBench-Doc (Ma et al., 2024). We re-
port the overall F1 score (F1) and the overall ac-
curacy (ACC). To assess modality-specific perfor-
mance, we break down accuracy by content type —-
Text, Layout, Chart, Table, and Figure —-reflecting
the diverse modalities present in long documents.
Questions are further categorized by the number
of evidences required:Single EvidenceandMulti
Evidence, to evaluate reasoning capabilities in sim-
ple versus complex scenarios. Additionally, we
measureUnanswerable Accuracy, i.e., the per-
centage of questions correctly identified as unan-
swerable, which is crucial for assessing the robust-
ness of QA systems in real-world settings.
A.2 Baselines
To rigorously evaluateMM-Doc-R1, we compare
it against a comprehensive set of state-of-the-art
RAG and agent-based baselines across different
paradigms. All text-based methods use Qwen3-8B
to generate answer, and multi-modal methods use
Qwen2.5-VL-7B.
For text-based RAG methods, we include BM25
and BGE-M3. BM25 is a classical sparse retrieval
approach using keyword matching. BGE-M3 is
a dense retrieval method that leverages semantic
embeddings for document retrieval. Since the orig-
inal PDFs are image-based, we apply Doc2X for
high-fidelity OCR to extract textual content before
indexing. The top-4 retrieved pages are used as
evidence for answer generation.
For multi-modal RAG, we evaluate ColQwen2.5-
7b-multilingual, which retrieves relevant document
pages using vision-language understanding and per-
forms end-to-end answer generation from images.
This method uses the top-4 retrieved pages as in-
put to maximize coverage of visual and structural
content.
We also compare against recent agent-based doc-
ument QA systems: Mdoc Agent (Han et al., 2025)
and M3doc RAG (Cho et al., 2024). Both systems
operate over the top-5 retrieved pages to ensure
consistent input scope.
All methods use the same candidate evidence
pool and are evaluated under consistent metrics to
ensure fair comparison.A.3 Implementation Details
Our framework is implemented using the Qwen3-
8B and Qwen3-4B Large Language Models as the
core agent orchestrator. The read tool incorpo-
rates Qwen2.5-7B-VL, a Vision-Language Model
(VLM) designed for multi-modal content extrac-
tion, which is employed in a zero-shot manner (i.e.,
without additional training). The search tool uti-
lizes the BM25 algorithm for text retrieval, return-
ing the top 10 most relevant pages for each query.
The final input to the answer model consists of a
short snippet from the search result and the detailed
context from the read result, where the search con-
tent occupies only a minimal portion of the overall
context and is not fed in full. For reinforcement
learning training, we specifically evaluate two pol-
icy optimization algorithms: GRPO and our pro-
posed SPO. Both GRPO and SPO are trained under
the same settings, except for the advantage estima-
tion method.
We employ theverlframework as our reinforce-
ment learning (RL) backbone. Within this frame-
work, we apply a custom patch to the advantage
computation function to accommodate our pro-
posed algorithmic enhancements.
For training infrastructure, we utilize a dis-
tributed setup powered by8 NVIDIA H100 GPUs.
The system leverages a hybrid of data and model
parallelism to support the efficient training of large-
scale policy models (e.g., Transformer-based ar-
chitectures). All computations are accelerated via
CUDA, and Automatic Mixed Precision (AMP) is
employed to significantly enhance training through-
put while maintaining numerical stability.
Our implementation is built upon theGRPO
algorithm, with key modifications to the advantage
computation and update dynamics. The primary
hyperparameters are configured as follows:
-Rollout length: Set to 4, meaning 4 steps of
environment interaction are collected per policy
sampling cycle before performing value estimation
and policy update. This balances training stability
with timely feedback.
-Batch size: Set to 4, indicating the number of
complete rollout trajectories used in each policy up-
date. Given that each rollout contains multiple time
steps, the effective number of state-action pairs per
update is4×4 = 16.
-Temperature: Set to 0.8, which controls
the level of exploration in the policy distribu-
tion. A lower temperature encourages more deter-

ministic outputs, promoting convergence on high-
confidence actions while preserving moderate ex-
ploration.
-Learning rate: The policy network is trained
with a learning rate of 5e-7 (5×10−7), using the
Adam optimizer. This conservative learning rate
is chosen to accommodate the high-dimensional
parameter space and high-precision gradient com-
putation enabled by the H100 GPUs, helping to
prevent policy collapse.
-KL coefficient: Set to 0.005 , serving as a reg-
ularization term to constrain the Kullback–Leibler
(KL) divergence between the old and new poli-
cies during updates. This prevents excessive policy
shifts and enhances training robustness.
A.4 Computational Overhead Analysis
Since the iterative agentic workflow naturally in-
curs more LLM/VLM calls than single-pass RAG,
we quantify the actual per-query latency to assess
practical usability. Table 4 reports the wall-clock
time per question on the full MMLongBench-Doc
set (1,082 questions).
Method Avg Min Max
ColQwen2.5 RAG (one-pass)∼40 – –
MM-Doc-R1 (Qwen3-4B, no train) 84.47 10 724
MM-Doc-R1 (Qwen3-8B, no train) 109.32 11 397
MM-Doc-R1 (Qwen3-8B, SPO) 154.44 23 316
Table 4: Per-question latency (seconds) on
MMLongBench-Doc.
Test-time scaling, not free overhead.MM-Doc-
R1 belongs to the family oftest-time scalingap-
proaches (cf. DeepSeek-R1, Gemini Deep Re-
search): additional compute is exchanged for sub-
stantially higher accuracy on multi-hop, multi-
evidence questions. The 109.32 s average for
the 8B variant is well within practical document-
analysis usage, while delivering the +10.4% abso-
lute ACC gain reported in §4.
Lightweight retrieval scales better with docu-
ment length.Embedding-based RAG must vec-
torizeallpages of every new document, so its la-
tency grows roughly linearly with document length
and incurs additional storage cost. MM-Doc-R1
instead uses BM25 for first-stage retrieval, which
requires no precomputed embeddings and supports
real-time DocQA on freshly arrived documents.
Consequently, as documents grow longer, the rel-
ative overhead of MM-Doc-R1 over RAGshrinksrather than grows, making the agentic workflow
particularly suited to long-document scenarios.
Why the SPO-trained model is slower.The
SPO-trained agent (154.44 s) takes longer than
the untrained one (109.32 s) because RL training
encourages more proactive multi-turn exploration,
which directly contributes to the recall and accuracy
gains; the cost–benefit trade-off remains favorable.
A.5 BM25 Topk choose
Figure 5: Performance with BM25 topk
Figure 5 presents the impact of varying the
top-kparameter on the performance of MM-Doc-
R1.Performance across all metrics generally im-
proves with increasing values of parameter k. How-
ever, substantial gains are primarily observed when
k <10 . For k >10 , the rate of improve-
ment slows, and performance may even decline,
attributable to the increased context length poten-
tially diluting relevant information. Consequently,
k= 10was selected as our experimental setting.
A.6 Prompt
Listing 1: Prompt for Planner
You are an expert in **long document analysis**
and **information retrieval planning**,
capable of designing systematic and
efficient exploration strategies using
multiple types of clues.
Your task:
Based on the user's query, the document's table
of contents (TOC) and the document's list of
figures, create an **initial research plan
** that is goal-oriented and progresses step
by step, helping downstream modules
efficiently understand and extract
information from the document.
You may use the following tools in your planning:
(1) Search Tool: Input keywords to search for
relevant content in the document. It returns
page IDs that contain those keywords. You
can search multiple keywords at once.

(2) Read Page: Read up to 5 pages of the
document and prepare the content for
analysis.
Notes:
- Your plan should not exceed 10 steps. Keep the
logic clear and progressive.
Please output the initial plan in the following
format:
<output_format>
Here is my proposed initial plan.
(1) ...
(2) ...
(3) ...
(4) ...
(5) ...
(6) ...
...
</output_format>
{example}
<input>
- Query: {query}
- document'd table of contents:
<document_toc>
{document_toc}
</document_toc>
<list_of_figures>
{list_of_figures}
</list_of_figures>
</input>
Listing 2: Prompt for Seeker
You are performing a step-by-step task of
information extraction and understanding.
Based on the current query goal and the
steps already taken (plan_done), you need to
:
(1) First, explain your reasoning process,
including:
- What information is still missing or unclear?
- What is the next key issue or sub-goal to
address?
- Which tools can help fill in this gap? Do you
need a combination of text and visual
content?
(2) Then, based on the above analysis, decide on
the next tool invocation(s).
You can use the following tools. In each step,
you may choose a reasonable number of
queries:
---
**(1) Search Tool**: Search whether certain
keywords or topics appear in the document.
Returns brief summaries of the pages where
matches are found.
Format:
```tool_code
<search>["keyword1", "keyword2"]</search>
````**(2) Read Page**: Read the content of specified
pages. Returns a summary relevant to your
query.
You may read up to 3 pages at a time. Clearly
state your intent for using this tool.
Format:
```tool_code
<read>
[{{
"page_ids": [4, 5, 8],
"query": "sub_query1"
}}, {{
"page_ids": [13, 19, 20],
"query": "sub_query2"
}}]
</read>
```
**(3) Termination Marker**: When you determine
that enough information has been gathered
and the task can end, return:
```tool_code
<FINISH>
```
Special Notes:
* Each step must be concise, strategic, and
limited to one tool only.
* You are encouraged to demonstrate **structured,
progressive strategic thinking**.
{example}
Query:
{query}
Steps already taken:
{plan_done}
Your reasoning and next-step plan:
Listing 3: Prompt for reader
You are a professional page summary expert. Your
task is to extract key information about
the origin query and sub query from given
pages.
Your input consists of:
1. An origin query
2. A sub query
#### Instructions
1. First, extract all visual elements:
- Tables
- Figures
- Charts
- Images
- Text content
2. Then, identify information relevant to:
- Origin Query
- Sub Query
3. Format your output as:
- Visual Elements Summary

- Query-Relevant Information (text and visual
elements)
- Key Findings
#### Important Notes
- Base your analysis strictly on the provided
images
- Do not make assumptions or add information
beyond what is shown
- If required information is missing, clearly
state: "Cannot answer due to insufficient
data"
- The sub query is the most recent and relevant
query, while the origin query is the earlier
context query
Input:
Origin Query:
{origin_query}
Sub Query:
{sub_query}
Image of Pages:
Listing 4: Prompt for answer
Please answer the question using only the
available information. Do not fabricate or
assume any details beyond what has been
provided.
If the necessary information is not available,
clearly state that you cannot answer the
question due to lack of relevant data.
question:{origin_query}
Related Information:{past_information}
A.7 Use of AI
We utilized generative AI tools to assist in refin-
ing the manuscript’s language and optimizing the
structure of our source code. We used these tools to
improve clarity and coding efficiency; however, we
reviewed and edited all outputs to ensure technical
accuracy. We take full responsibility for the final
content of the manuscript and the integrity of the
implemented code.