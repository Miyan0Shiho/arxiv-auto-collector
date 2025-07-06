# Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search

**Authors**: Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yang Zhao, Hongjin Qian, Zhicheng Dou

**Published**: 2025-07-03 14:18:08

**PDF URL**: [http://arxiv.org/pdf/2507.02652v1](http://arxiv.org/pdf/2507.02652v1)

## Abstract
Complex information needs in real-world search scenarios demand deep
reasoning and knowledge synthesis across diverse sources, which traditional
retrieval-augmented generation (RAG) pipelines struggle to address effectively.
Current reasoning-based approaches suffer from a fundamental limitation: they
use a single model to handle both high-level planning and detailed execution,
leading to inefficient reasoning and limited scalability. In this paper, we
introduce HiRA, a hierarchical framework that separates strategic planning from
specialized execution. Our approach decomposes complex search tasks into
focused subtasks, assigns each subtask to domain-specific agents equipped with
external tools and reasoning capabilities, and coordinates the results through
a structured integration mechanism. This separation prevents execution details
from disrupting high-level reasoning while enabling the system to leverage
specialized expertise for different types of information processing.
Experiments on four complex, cross-modal deep search benchmarks demonstrate
that HiRA significantly outperforms state-of-the-art RAG and agent-based
systems. Our results show improvements in both answer quality and system
efficiency, highlighting the effectiveness of decoupled planning and execution
for multi-step information seeking tasks. Our code is available at
https://github.com/ignorejjj/HiRA.

## Full Text


<!-- PDF content starts -->

arXiv:2507.02652v1  [cs.AI]  3 Jul 2025Decoupled Planning and Execution: A Hierarchical Reasoning Framework for
Deep Search
Jiajie Jin1, Xiaoxi Li1, Guanting Dong1, Yuyao Zhang1, Yutao Zhu1,
Yang Zhao1, Hongjin Qian2, Zhicheng Dou1*
Gaoling School of Artificial Intelligence, Renmin University of China
BAAI
{jinjiajie, dou }@ruc.edu.cn
Abstract
Complex information needs in real-world search scenarios
demand deep reasoning and knowledge synthesis across di-
verse sources, which traditional retrieval-augmented gener-
ation (RAG) pipelines struggle to address effectively. Cur-
rent reasoning-based approaches suffer from a fundamen-
tal limitation: they use a single model to handle both high-
level planning and detailed execution, leading to inefficient
reasoning and limited scalability. In this paper, we intro-
duce HiRA, a hierarchical framework that separates strate-
gic planning from specialized execution. Our approach de-
composes complex search tasks into focused subtasks, as-
signs each subtask to domain-specific agents equipped with
external tools and reasoning capabilities, and coordinates the
results through a structured integration mechanism. This sep-
aration prevents execution details from disrupting high-level
reasoning while enabling the system to leverage specialized
expertise for different types of information processing. Ex-
periments on four complex, cross-modal deep search bench-
marks demonstrate that HiRA significantly outperforms state-
of-the-art RAG and agent-based systems. Our results show
improvements in both answer quality and system efficiency,
highlighting the effectiveness of decoupled planning and ex-
ecution for multi-step information seeking tasks. Our code is
available at https://github.com/ignorejjj/HiRA.
Introduction
The information explosion on the internet has made it in-
creasingly difficult to find comprehensive answers to com-
plex queries, leading to the rapid development of deep
search tasks that require understanding complex informa-
tion needs and synthesizing accurate answers from multiple
sources (Li et al. 2024, 2025b). However, traditional search
engines only return ranked web pages based on keyword
matching, requiring users to manually filter and synthesize
information.
While large language models combined with search en-
gines can provide direct answers, they typically only uti-
lize shallow information from search results, lacking deep
reasoning and comprehensive analysis capabilities (Lewis
et al. 2020; Gao et al. 2024). This has motivated the de-
velopment of specialized AI agents for deep search tasks,
*Corresponding author.
Copyright © 2025, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Reasoning DistillationQ
R
R
R
AQ
R
T
R
T
AQ
R
C
R
TR
C
R
C
R
RR
R
AR
T
RSearch-enhanced
Reasoning MultiModal-enhanced  
Expert ExpertExpertRReasoning Node
CCoordinator
TBasic Tools(a) Direct
Reasoning(b) Search-Augmented
Reasoning(c) Hierarchical Reasoning （ HiRA ）
Meta Reasoning PathFigure 1: Comparison of current approaches for deep
search tasks: (a) Direct Reasoning with LRMs, (b) Search-
Augmented Reasoning that enables LRMs to use search en-
gine during reasoning, and (c) Hierarchical Reasoning that
autonomously interacts with expert agents and tools in con-
tinuous thinking process.
such as OpenAI DeepSearch (OpenAI 2025) and Grok
DeepSearch (Grok 2025), which aim to bridge the gap be-
tween simple search-enhanced models and deep information
seeking systems.
Conventional approaches typically employ Retrieval-
Augmented Generation (RAG) methods with predefined
workflows (Lewis et al. 2020; Trivedi et al. 2023; Shao
et al. 2023), which incorporate components like query de-
composition, document summarization, and self-reflection
to improve effectiveness. Recently, Large Reasoning Models
(LRMs) such as OpenAI-o1 (OpenAI 2024a) and DeepSeek-
R1 (DeepSeek-AI et al. 2025) have introduced new oppor-
tunities by integrating search and web browsing capabilities
within their reasoning processes (Li et al. 2025a; Jin et al.
2025; Song et al. 2025). As shown in Figure 1, these search-
augmented reasoning methods can autonomously plan and
acquire external knowledge for complex information re-
trieval in an end-to-end manner, significantly improving
deep search performance.
However, existing approaches suffer from architectural
limitations due to their reliance on a single reasoning model
to handle all tasks. Current methods typically work by
prompting reasoning models to generate special tokens (Li
et al. 2025a; Jin et al. 2025; Li et al. 2025b), which are

used to trigger corresponding tool activations, during their
thinking process. For example, in WebThinker, the search
action is triggered by <|begin searchquery|> and
<|endsearchquery|> . This monolithic paradigm in-
troduces two critical deficiencies: (1) Limited Capability Ex-
tensibility: Adding new tools or capabilities requires care-
fully redesigning prompts to teach the model new token pat-
terns and their usage contexts. This process is brittle and of-
ten requires re-engineering token systems or extensive fine-
tuning and reinforcement learning to ensure reliable token
generation and tool coordination. (2) Reasoning Disruption:
As shown in Figure 1, external execution results are di-
rectly injected into the reasoning chain, introducing noise
that interferes with core reasoning processes. This disruption
weakens the model’s logical thinking and occupies limited
context windows with irrelevant operational details. These
limitations stem from the assumption that one agent should
handle all aspects of complex reasoning tasks. We argue that
effective agent execution should follow a hierarchical struc-
ture, which is shown in Figure 1(c): a meta-agent for high-
level planning, a coordinator for task reasoning transfer, and
specialized execution agents for specific operations. Each
execution agent focuses on completing a single assigned
subtask through multi-step reasoning and iterative tool us-
age, allowing for deeper analysis without contaminating the
overall planning process.
Based on this insight, we propose Hierarchical
ReAsoning (HiRA), a reasoning-model driven frame-
work designed to enhance deep search effectiveness by
separating planning from execution. This architecture
consists of three components: the Meta Reasoning Planner,
the Adaptive Reasoning Coordinator, and the Domain-
Specialized Executors. The Meta Reasoning Planner breaks
down complex tasks into subtasks through a reasoning
process. These subtasks are then dispensed to specialized
agents by the Adaptive Reasoning Coordinator, which
assigns them based on task complexity and the required
expertise. Each Domain-Specialized Executor leverages
specific reasoning models and external tools to execute the
assigned subtask, with reasoning result distillation into the
planner’s process via the coordinator.
This hierarchical design effectively decouples the strate-
gic planning from the execution details, allowing for scal-
able, coherent reasoning in complex information retrieval
tasks. By integrating specialized expertise at the execution
level and maintaining a coordinated flow across the hi-
erarchy, HiRA ensures both the flexibility and efficiency
necessary for tackling advanced reasoning challenges. We
conduct experiments on four complex, cross-modal, multi-
scenario deep search tasks demonstrate that our framework
significantly surpasses existing methods in all aspects.
Overall, the key contributions of this work are threefold:
•(1) Hierarchical Reasoning Architecture: We propose
a novel hierarchical reasoning framework that integrates
specialized tool-augmented reasoning agents as cognitive
modules, eliminating the need for external tool orches-
tration or rigid predefined pipelines used in existing ap-
proaches.•(2) Enhanced Capability Integration: Our Domain-
Specialized Executors enable plug-and-play integra-
tion of diverse reasoning capabilities and tools. Exist-
ing search agents can be directly incorporated without
prompt engineering or model retraining, preserving es-
tablished workflows while enhancing performance.
•(3) Superior Empirical Performance: Experiments
across four complex cross-modal search tasks demon-
strate significant improvements over traditional RAG and
current agent-based methods, validating the effectiveness
of hierarchical reasoning augmentation for complex in-
formation retrieval.
Related Work
From Retrieval-Augmented Generation to Deep Search.
Retrieval-Augmented Generation (RAG) combines external
knowledge with LLMs’ parametric knowledge (Lewis et al.
2020; Borgeaud et al. 2022). Early approaches used single-
step retrieval mechanisms (Gao et al. 2024; Jin et al. 2024a),
while iterative RAG pipelines later incorporated query de-
composition (Chan et al. 2024; Zhang et al. 2025), document
refinement (Xu, Shi, and Choi 2023; Jin et al. 2024b), and
multi-round search (Shao et al. 2023; Trivedi et al. 2023).
However, RAG methods rely on predefined workflows that
limit adaptive decision-making for complex queries. Recent
work with LRMs integrate retrieval directly into reasoning
processes (Li et al. 2025a; Jin et al. 2025), but still requires
inserting documents into reasoning chains or using auxil-
iary models for summarization (Li et al. 2025b). These lim-
itations motivate our exploration of hierarchical reasoning
augmentation, where specialized agents serve as cognitive
extensions while expanding additional capabilities.
Planning-Execution Separation Approaches. To ad-
dress information overload in model reasoning, recent work
separates planning from execution using dedicated plan-
ners and executors (Bilal et al. 2025). Existing approaches
fall into two categories. Action-level separation assigns ex-
ecutors to single-step tasks, as in Plan-Act (Erdogan et al.
2025) for HTML manipulation and CoAct (Hou et al. 2024)
with local replanning capabilities. Query-level separation
decomposes problems at higher granularity: REMA (Wan
et al. 2025) trains RL-based planners for mathematical rea-
soning, while LLMCompiler (Kim et al. 2023) and Query
Compiler (Zhang et al. 2025) break down QA tasks into par-
allel execution graphs. However, these methods suffer from
rigid task decomposition and limited executor specialization
beyond prompt variations. Our work addresses these limi-
tations through dynamic reasoning delegation and domain-
specialized agents within a hierarchical framework.
Methodology
Problem Formulation
Formally, given a complex information retrieval question q
and a predefined external environment E, our objective is to
design a framework that produces a final solution containing
an answer Aand the corresponding reasoning process R.

TaskMeta Reasoning Planner
Reasoning
Domain-Specialized Executors
Delegation
Sub task
 Reasoning
 Tool call
Result
Final result
Distillation
Reasoning
Executor
Sub task 1
Response
Final Result
CodeImage 
Video AudioSearch
BrowseSpecialized ToolsAdaptive Reasoning Coordinator
Delegation
Distillation
Executor
Sub task 2
Response
Reasoning
Dual Channel
Memory
Retrieve
Extract
Memory
MemoryCoherent Reasoning
Search Expert Code Expert
Executor
Memory Type
Fact
Memory
Resource
MemoryFigure 2: Overview of the HiRA Framework.
The generation process can be represented as:
P(R, a|q,E) =TRY
t=1P(Rt| R<t, q,E<t)·P(a|q,R),
where TRrepresents the token generation steps for the rea-
soning process, Adenotes the final answer, and E<t=
{E(R<s)}s<tdenotes the collection of all environment in-
teraction results prior to timestep t. While existing agentic
reasoning approaches typically use tools directly as the ex-
ternal environment, our work introduces a higher-level ab-
straction where the environment Econsists of a collection of
expert agents, each capable of reasoning and utilizing spe-
cific tools to accomplish specialized tasks.
Overview of the HiRA Framework
The HiRA framework is a hierarchical reasoning system that
enhances deep search effectiveness through the separation
of planning and execution. The architecture consists of three
tiers: (1) Meta Reasoning Planner that decomposes com-
plex tasks into subtasks through step-by-step reasoning, (2)
Adaptive Reasoning Coordinator that assigns subtasks to
appropriate expert agents based on task complexity and ca-
pabilities, and (3) Domain-Specialized Executors that ex-
ecute subtasks using specialized reasoning models and ex-
ternal tools. Results are integrated back into the planner’s
reasoning process through the coordinator. This hierarchical
design decouples strategic planning from execution details,
enabling scalable and coherent reasoning for complex infor-
mation retrieval tasks.Hierarchical Reasoning Paradigm
Meta Reasoning Planner The Meta Reasoning Planner
serves as the framework’s core orchestrator, responsible for
planning, reasoning, and answer generation. Unlike conven-
tional tool-augmented approaches that require models to di-
rectly invoke tools with specific parameters, our Meta Plan-
ner generates high-level subtasks containing strategic in-
structions for expert agents. This design enables natural col-
laboration between meta and expert agents, ensuring smooth
information transfer while eliminating the noise and over-
head associated with direct tool invocation and execution-
level decision making.
To enable dynamic subtask generation during the reason-
ing process, we design a meta planner prompt that instruct
the model to use special tokens for subtask dispatch. The
overview of the process is shown in Figure 2. In reasoning
process, the model will automatically generate special to-
kens and place the content and requirements of subtasks in
the middle, which is similar to the process of humans is-
suing tasks. As shown in Equation (1), the generated sub-
task content skis based on previous task execution results
{E(sj)}j<kand current reasoning progresses O<t, naturally
enabling reflection, correction, supplementation, and conti-
nuity generation of prior tasks. We impose no explicit con-
straints on subtask scope, difficulty, size, or relationships
with previous subtasks to preserve overall reasoning flexi-
bility.
PM(sk) =PM(sk|q,O<t,{E(sj)}j<k). (1)
Then, the coordinator layer processes and assigns skto
corresponding expert agents for execution. Execution results

Ak(sk)are wrapped in special tokens, and then integrated
into the model’s reasoning process for continued genera-
tion. Notably, the subtask execution results incorporated into
the reasoning process contain essential execution procedures
and final conclusions, rather than vanilla tool invocation re-
sults that may contain noise and require subsequent process-
ing.
During the generation process, the model incrementally
conditions on the original query q, the prior decoding history
O<t, and the set of executed subtask results Aj(sj)j≤Kto
derive the final answer a, formalized as:
PM(a) =PM(a|q,O<t,Aj(sj)j≤K), (2)
where Kdenotes the total number of subtasks.
This design enables modular task planning by decoupling
high-level goals from execution, allowing the model to gen-
erate subtasks without knowing specific expert agents.
Adaptive Reasoning Coordinator While separating ex-
ecution from planning provides clear scalability advantages
and reduces computational noise, it introduces the risk of in-
formation loss between components. To mitigate this chal-
lenge, we design an adaptive reasoning coordinator that in-
corporates bidirectional, task-specific reasoning transfer and
a dual-channel memory mechanism. This coordinator facil-
itates seamless reasoning delegation from the meta agent to
expert agents while enabling reasoning distillation in the re-
verse direction, thus preserving efficient inter-agent commu-
nication and maintaining the architectural benefits of sepa-
ration. In the following parts, we detail three core functions
of the coordinator.
(1) Reasoning Transfer Process. The coordinator is de-
signed to interpret subtasks provided by the meta planner
and identify the most suitable expert agent for task execu-
tion. Given the current subtask skand detailed information
about all experts IE={IA}A∈E, the coordinator first ana-
lyzes the subtask requirements, then evaluates agent capabil-
ities across two key dimensions before making the optimal
selection. Our instruction framework Iselect encompasses:
(1) Required capabilities: the domain knowledge and tool
utilization abilities necessary for task completion, and (2)
Task complexity: the computational difficulty of the sub-
task and the required depth of analysis. For tasks within
the same category (e.g., information retrieval), target infor-
mation often resides at varying depths across data sources.
While deploying the most sophisticated agent can ensure
problem resolution, it may introduce unnecessary compu-
tational overhead and analytical redundancy. Therefore, the
coordinator prioritizes selecting the most efficient agent for
each specific task to optimize overall system performance.
Formally, we model this selection process as a classifica-
tion problem:
A∗
k=argmaxA∈EPC(O(k)
dele,A |sk,IE,Iselect),
where the coordinator generates selection reasoning O(k)
dele
and identifies the optimal expert agent A∗
kthrough chain-
of-thought reasoning.
(2) Reasoning Distillation Process. The coordinator is
responsible for understanding and refining the expert agent’sreasoning process before integrating the results into the meta
reasoning planner’s cognitive flow to maintain overall rea-
soning coherence. Unlike traditional tool-augmented rea-
soning approaches that primarily focus on final execution
outputs, our framework considers both the expert agent’s in-
termediate reasoning steps and the final conclusions. This
dual consideration enables the meta planner to comprehend
the underlying reasoning logic without being encumbered by
low-level execution details, thereby facilitating autonomous
reflection and critical evaluation of execution outcomes.
Specifically, given subtask skand the expert agent’s rea-
soning process O(k)
expert, the distilled reasoning process O(k)
dist
and refined conclusion R(k)
distare generated through:
PC
O(k)
dist,R(k)
dist|sk,O(k)
expert
=PC
O(k)
dist| O(k)
expert,·
| {z }
Reasoning Refinement·PC
R(k)
dist| O(k)
dist,O(k)
expert,·
| {z }
Conclusion Extraction.
Subsequently, O(k)
distandR(k)
distare concatenated and seam-
lessly integrated as reasoning distillation feedback into the
meta reasoning planner’s ongoing cognitive process.
(3) Dual-Channel Memory Mechanism. To enable ef-
fective information sharing and knowledge transfer among
expert agents, we design a dual-channel memory mechanism
tailored specifically for deep search task scenarios. The com-
prehensive memory repository Mencompasses two distinct
types of memory: fact memory Mfand resource mem-
oryMr. Fact memory archives factual discoveries and in-
sights extracted from expert agents’ reasoning processes.
Each memory entry comprises a factual assertion paired
with its corresponding source attribution (URL, file name,
or webpage identifier) to ensure traceability and verifica-
tion. To maintain memory efficiency and reduce redundancy,
multiple similar factual statements originating from identi-
cal sources are intelligently aggregated. Resource memory
maintains a repository of informational resources encoun-
tered during expert agent execution processes. Each entry
contains a descriptive summary alongside the corresponding
access path (e.g., webpage URL, file path), designed to pro-
vide subsequent agents with valuable exploration insights
from previous agent interactions, thereby preventing redun-
dant exploration and enhancing overall system efficiency.
Memory construction and utilization operate in conjunc-
tion with the reasoning transfer pipeline. During the reason-
ing distillation phase, relevant memory components are ex-
tracted from the reasoning process Okand systematically
updated within the global memory repository M. During
reasoning delegation, the coordinator intelligently retrieves
pertinent memory entries based on semantic relevance be-
tween memory content and subtask requirements, as well as
memory quality metrics, subsequently providing this con-
textual information as supplementary guidance to expert
agents.
Domain-Specialized Executors To cover the diverse ca-
pabilities required for deep search tasks, we design three or-
thogonal agent capability dimensions, ensuring the system
can handle complex and varied deep search scenarios:

Table 1: Overall performance on various deep search tasks, with accuracy results for each dataset obtained using llm-as-judge.
For 32B models, the best results are indicated in bold , and the second-best results are underlined . Results from larger or
closed-source models are presented in gray for reference. For the GAIA dataset, queries without files are used to ensure a fair
comparison with the baseline.
MethodGeneral AI Assistant WebWalkerQA Humanity’s Last Exam SimpleQA
Level 1 Level 2 Level 3 Avg. Easy Med. Hard Avg. NS CE SF Avg. Acc
Direct Reasoning
Qwen3-32B-no-thinking 14.3 6.1 10.5 9.5 4.4 2.1 0.8 2.8 7.1 6.0 3.1 6.2 6.5
Qwen3-32B-thinking 26.2 12.1 0 14.9 6.9 1.1 2.9 3.1 14.6 9.8 8.4 12.6 10.5
DeepSeek-R1-32B 21.5 13.6 0.0 14.2 7.5 1.4 4.2 3.8 6.6 5.1 6.5 6.4 5.5
QwQ-32B 30.9 6.5 5.2 18.9 7.5 2.1 4.6 4.3 11.5 7.3 5.2 9.6 6.5
GPT-4o 23.1 15.4 8.3 17.5 6.7 6.0 4.2 5.5 2.7 1.2 3.2 2.6 39.0
DeepSeek-R1-671B 40.5 21.2 5.2 25.2 5.0 11.8 11.3 10.0 8.5 8.1 9.3 8.6 42.4
o1-preview†- - - - 11.9 10.4 7.9 9.9 12.9 8.1 6.6 11.1 42.7
Single-Capability Enhanced
Vanilla RAG 40.5 21.2 5.2 25.2 57.4 44.6 40.0 46.0 10.6 3.7 11.6 9.6 72.5
Search-o1 45.3 25.8 5.3 29.1 70.2 44.6 40.0 49.0 13.0 8.5 12.6 12.2 74.0
WebThinker 50.0 34.9 10.5 36.2 55.3 53.0 50.0 52.5 13.9 9.7 12.6 13.0 78.0
CodeAct 26.2 15.1 0.0 16.5 6.4 4.8 4.3 5.0 9.9 6.1 10.5 9.4 7.5
Multimodal Enhanced 23.8 9.1 0.0 12.6 4.3 0 4.3 4.0 9.3 9.8 8.4 9.2 10.5
Multi-Capability Enhanced
Plan-and-Solve 28.6 18.2 0.0 18.9 44.7 33.7 24.3 33.0 10.2 4.9 7.3 8.8 57.5
ReAct 45.3 28.8 5.2 30.7 46.8 31.3 31.4 35.0 12.7 11.0 20.0 13.8 73.5
HiRA (ours) 61.9 37.9 15.8 42.5 59.6 54.2 51.4 54.5 15.2 11.0 13.7 14.2 81.5
•Information Acquisition: This dimension is responsible
for acquiring and integrating information from the web.
•Cross-Modal Understanding: This dimension handles
the understanding and fusion of multimodal information,
capable of processing data from different modalities such
as images, videos, and audio.
•Computational Reasoning: This dimension handles
mathematical computation, file processing, and other
computational reasoning tasks, capable of transforming
abstract problems into executable code solutions.
Based on these three dimensions, we implement four rea-
soning model-driven, specialized agents. For information
acquisition , we design two search agents with different ex-
ploration depths: one based on a simple RAG pipeline that
performs single retrieval for subtasks followed by reasoning,
and another based on the WebThinker implementation that
can perform deep search and information acquisition on the
internet. The combination of these two approaches enables
flexible solutions for both simple and complex tasks. For
Cross-Modal Understanding , we embed multimodal mod-
els as tools within the reasoning model’s inference process
to achieve dynamic understanding of information in mul-
timodal data. For Computational Reasoning , we embed
code interpreters into the reasoning model’s thought process.
For the aforementioned reasoning-driven agents, their
reasoning process follows the tool-augmented reasoning ex-
ecution flow, capable of dynamically outputting special to-
kens during reasoning to trigger corresponding tools. Thereasoning process is:
P(O(k)|sk,T,Mk) =TkX
t=1P(O(k)
t| O(k)
<t,{Tj}<t,·),
whereO(k)represents the reasoning process of expert agent
for subtask sk,Mkrepresents the memory related to sk,
Trepresents the tools used in the process (e.g., code inter-
preter), and {Tj}<trepresents all tool invocation results be-
fore time step t. Based on this process, we can embed ar-
bitrary tools into the reasoning model’s inference process.
More details can be found in the appendix.
Inference Process of HiRA
The inference process of HiRA follows an agentic reason-
ing approach. For a given question, the inference process
begins with reasoning by the meta agent. During reason-
ing, the meta planner decodes special token pairs, wrapping
the subtasks to be executed between them. The coordinator
then processes and distributes the subtasks to expert agents
for execution. After the expert agents perform reasoning and
multiple rounds of tool invocation, subtask execution results
are obtained.
This reasoning process and results are then processed
by the coordinator through the reasoning distillation pro-
cess to obtain refined results, which are integrated into the
meta planner’s reasoning chain to continue generation. Dur-
ing this process, the meta planner dynamically adjusts its
plan and corresponding subtasks to be distributed based on
execution results, until all information has been collected
and the final answer is provided. The detailed algorithmic
flowchart can be found in the appendix.

Experimental Settings
Tasks and Datasets To comprehensively evaluate our
method’s performance on deep search tasks, we follow prior
work and introduce more scenarios (e.g., multimodal and
file input) to thoroughly test our method’s ability to inter-
act with diverse data types and environments. We use the
following datasets for evaluation: (1) GAIA (Mialon et al.
2024), a benchmark for evaluating general-purpose AI assis-
tants with questions requiring multi-step reasoning and ex-
ternal information retrieval, where we use all samples from
its validation set, categorized into text-only, multimodal, and
with-file; (2) WebWalkerQA (Wu et al. 2025), designed to
evaluate models’ ability to navigate web pages and extract
information from single or multiple pages, covering both
English and Chinese queries, from which we sample 200
test questions; (3) SimpleQA (Wei et al. 2024), testing mod-
els’ factual accuracy, knowledge breadth, and factual capa-
bilities, from which we sample 200 test questions; and (4)
Humanity’s Last Exam (Phan et al. 2025), a high-difficulty
dataset requiring complex reasoning, containing academic
problems across mathematics, physics, and computer sci-
ence, where models need to retrieve relevant external infor-
mation, from which we sample 500 validation questions. To
ensure fair and accurate evaluation, we adopt LLM-as-judge
approach for all datasets, using the Qwen2.5-72B-Instruct
model (Yang et al. 2024) to compute accuracy scores. Evalu-
ation prompts and detailed settings are provided in appendix.
Baselines We compare our method against three base-
line approaches: (1) Direct Reasoning : Direct question
answering using models’ inherent reasoning capabilities.
We evaluate both open-source models (Qwen3-32B (Yang
et al. 2025), QwQ-32B (Team 2024), DeepSeek-R1-
32B (DeepSeek-AI et al. 2025)) and commercial mod-
els (GPT-4o (Hurst et al. 2024), DeepSeek-R1 (DeepSeek-
AI et al. 2025), o1-preview (OpenAI 2024b)). (2) Single-
Capability Enhanced : Methods that augment reasoning
with a single specialized capability. This includes search-
enhanced approaches like Search-o1 (Li et al. 2025a) and
WebThinker (Li et al. 2025b), code execution capabilities
like CodeAct (Wang et al. 2024), and multimodal reason-
ing capabilities through our implemented Multi-Enhanced
Reasoning baseline. (3) Multi-Capability Reasoning : Ap-
proaches that integrate multiple capabilities or employ struc-
tured reasoning frameworks. This includes planning-based
methods like Plan-and-Solve (Wang et al. 2023) that sep-
arate planning from execution, and multi-tool frameworks
like ReAct (Yao et al. 2022) that directly orchestrate multi-
ple capabilities within a single reasoning process. These two
methods and our approach use the same set of tools to ensure
a fair comparison.
Implementation Details Following current work, we use
QwQ-32B as the base model for both the meta reason-
ing planner and expert agents. We employ a same-sized
Qwen2.5-Instruct model as the Adaptive Reasoning Coor-
dinator. For all models, we set temperature to 0.7, top p to
0.95, top k to 20, and the model context window to 128k to-
kens. Our search tool uses the Bing Web Search API with
US-EN as the region setting and retrieves the top-10 searchTable 2: Ablation studies of HiRA, showing results for each
layer. GAIA-B refers to GAIA queries without associated
files (as in the main result), while GAIA-F refers to the sub-
set with files.
Method GAIA-B GAIA-F Web. HLE Simp. Avg.
HiRA 42.5 42.1 54.5 14.2 81.5 44.9
Coordinator Layer
w/o Reasoning Transfer 33.9 36.8 44.5 10.4 76.5 40.4
w/o Memory 37.8 31.6 52.0 11.8 79.0 42.4
Executor Layer
w/o Search 15.7 31.6 4.0 12.4 9.5 14.6
w/o Code 33.9 28.9 51.5 12.8 76.5 40.7
w/o Multimodal 36.2 36.8 55.0 13.6 81.0 44.5
results. We employ Qwen2.5-Omni-7B (Xu et al. 2025) as
our multimodal tool, powered by Qwen’s official API. We
construct a sandbox based on a Python interpreter as our
code execution environment, with necessary security restric-
tions to ensure safe operation. In the main experiments, we
set the maximum number of subtasks for the meta reasoning
planner to 10.
Experimental Results
Main Results
As shown in Table 1, we evaluate our method on the deep
search tasks against several baselines that integrate rea-
soning with tool usage. We have the following observa-
tions: (1) Overall Performance Superiority : Our method
consistently outperforms all baseline methods. It signifi-
cantly surpasses direct reasoning models without tool us-
age and achieves notable improvements over existing tool-
augmented approaches. Compared to the strongest search
agent baseline WebThinker, our approach demonstrates sub-
stantial advantages on both complex tasks (GAIA and HLE).
(2) Hierarchical Reasoning Design Advantages : The ex-
perimental results show that our hierarchical design achieves
better performance when using the same set of tools. For
Plan-and-Solve which relies on a fixed plan executed se-
quentially, performs poorly (e.g., 18.9 in GAIA), highlight-
ing the necessity of dynamic planning during reasoning.
While ReAct enables dynamic planning by integrating mul-
tiple tools into the reasoning chain, its performance suffers
in multi-tool scenarios such as GAIA due to the overhead
of tool selection and noisy intermediate tool outputs. In con-
trast, our method outperforms WebThinker on GAIA, which
requires diverse capabilities, while also achieving superior
results on general web search tasks. (3) Robustness Across
Task Complexity : Our framework shows moderate gains on
simpler tasks (e.g., SimpleQA and WebWalkerQA), but ex-
hibits much larger improvements on more complex tasks,
demonstrating its strength in handling complex reasoning
scenarios.
Ablation Study
We conduct ablation studies to investigate the contribution
of each component within the framework by removing mod-

1 2 3 4 5 10 15
Max Subtask Number15202530354045Performance Score
No Agent Description
With Agent DescriptionFigure 3: Performance comparison on whether the expert
agent description is provided to the meta planner and maxi-
mum number of sub-tasks limit.
Reasoning & Web browsing Multi-modality File Understanding
Capability1020304050Score 23.3
0.028.932.0
12.531.240.8
16.736.8 35.9
8.331.644.7
33.342.1Direct RAG WebThinker ReAct Ours
Figure 4: Comparison of our method with the baseline on
three GAIA subsets, evaluating performance across different
dimensions of capability.
ules from both the Coordinator Layer and the Executor
Layer. Results are shown in Table 2.
In the Coordinator Layer, removing the Reasoning Trans-
fer mechanism leads to significant performance drops, espe-
cially on complex reasoning tasks, with nearly 30% decrease
in performance. In comparison, the Memory mechanism has
a smaller impact, except on file-related tasks (approx. 10%
drop), indicating that the resource component in memory ef-
fectively supports information propagation.
In the Executor Layer, removing individual expert agents
leads to substantial performance degradation. Among them,
removing the Search Agent causes the most severe drop
across all datasets, highlighting the essential role of in-
formation acquisition on web. Similarly, removing the
Code Agent significantly impacts performance on multi-
functional datasets such as GAIA, showing its importance
for general-purpose tasks. Removing the Multimodal Agent
results in a slight drop on GAIA, where some cases require
multimodal capabilities, but has little impact on standard
web tasks.
Generalization and Effectiveness of Meta Planner
In our framework, the meta-planner receives expert capa-
bility information for better subtask planning, while also
limiting the number of subtasks to control computational
GAIA WebWalker HLE05000100001500020000Reasoning LengthReasoning Length
WebThinker
ReAct
Ours
GAIA WebWalker HLE01234Interaction TimesInteraction Times
WebThinker
ReAct
OursFigure 5: Comparison of different methods in terms of rea-
soning length (number of output tokens during model infer-
ence) and interaction times (number of interactions with the
environment during inference) in three datasets.
cost. These two factors affect both generalization (support-
ing more agents) and effectiveness (achieve higher perfor-
mance). To evaluate both aspects, we introduce an addi-
tional setting where no expert information is provided to
the meta-planner, and we vary the maximum number of al-
lowed subtasks. Results are shown in Figure 3, leading to
two key observations: (1) Decoupled Meta-Planner De-
sign: Even without access to expert information, the meta-
planner achieves comparable performance. This suggests
that, within our architecture, the meta-planner and expert
agents are relatively decoupled—the generated subtasks pri-
marily depend on the task semantics rather than the spe-
cific agents, enabling better scalability to new agents. (2)
Subtask Scaling Trade-off : As the number of subtasks
increases, performance first improves then degrades. This
resembles an inference-time scaling effect: allowing more
subtasks enables deeper reasoning chains, while overly re-
stricting subtasks may prevent sufficient exploration. How-
ever, excessive subtasks may introduce inefficient plans,
eventually hurting performance.
Capability Analysis
Beyond the main experiments, we further evaluate our sys-
tem on additional subsets of GAIA to assess its capabilities
beyond web search (e.g., multimodal understanding, file un-
derstanding), which are also critical for more comprehensive
information acquisition. As shown in Figure 4, our method
achieves the best performance across all capability dimen-
sions, demonstrating its broad applicability. While baselines
like ReAct perform well in purely textual reasoning and web
browsing, our integration of the DeepSearch agent yields
superior results. Moreover, due to our multimodal architec-
ture, we achieve additional gains in non-textual tasks. It is
worth noting that, although ReAct can invoke multimodal
and code tools, its planning model struggles to coordinate
multiple tools simultaneously, often leading to sub-optimal
performance (e.g., 8.3 vs. 12.5 in multimodal tasks), in some
cases even underperforming pure search-based approaches.
Efficiency Analysis
To assess the overall efficiency of our method, we compared
the number of inference tokens and the number of inter-
actions with the environment between HiRA and baselines

across three datasets of varying difficulty. Based on the re-
sult in Figure 5 and Table 1, we make two observations: (1)
Overall, more difficult datasets tend to result in longer rea-
soning chains, but they do not necessarily lead to more inter-
actions with the environment, which may also be influenced
by the nature of the task itself. (2) Compared to WebThinker,
which directly integrates the search function into the main
reasoning chain, our hierarchical reasoning structure results
in shorter reasoning chains and fewer interaction times, indi-
cating that our approach is more efficient in each subtask call
than directly invoking tools. (3) The ReAct method, which
integrates multiple functionalities into a single model, leads
to fewer tool calls and insufficient reasoning, which may
be due to interference between the descriptions of multiple
tools, resulting in suboptimal performance.
Conclusion
In this paper, we proposed HiRA, a hierarchical reasoning
framework that addresses the limitations of monolithic rea-
soning models in deep search tasks. By decoupling high-
level planning from low-level execution, HiRA introduces
a multi-agent architecture consisting of a Meta Reasoning
Planner, an Adaptive Reasoning Coordinator, and Domain-
Specialized Executors. This design enables scalable and
modular reasoning by embedding specialized agents as cog-
nitive components rather than relying on rigid pipelines or
single-agent paradigms. Through dual-channel memory and
coordinated task routing, our approach supports coherent
knowledge synthesis and dynamic integration of diverse rea-
soning capabilities. Extensive experiments across five com-
plex, multi-modal deep search scenarios demonstrate that
HiRA significantly outperforms conventional RAG systems
and prior agent-based methods, validating the effectiveness
and flexibility of hierarchical reasoning in addressing so-
phisticated information retrieval challenges.
References
Bilal, A.; Mohsin, M. A.; Umer, M.; Bangash, M.
A. K.; and Jamshed, M. A. 2025. Meta-Thinking in
LLMs via Multi-Agent Reinforcement Learning: A Survey.
arXiv:2504.14520.
Borgeaud, S.; Mensch, A.; Hoffmann, J.; Cai, T.; Ruther-
ford, E.; Millican, K.; van den Driessche, G.; Lespiau, J.-B.;
Damoc, B.; Clark, A.; de Las Casas, D.; Guy, A.; Menick, J.;
Ring, R.; Hennigan, T.; Huang, S.; Maggiore, L.; Jones, C.;
Cassirer, A.; Brock, A.; Paganini, M.; Irving, G.; Vinyals,
O.; Osindero, S.; Simonyan, K.; Rae, J. W.; Elsen, E.; and
Sifre, L. 2022. Improving Language Models by Retriev-
ing from Trillions of Tokens. In International Conference
on Machine Learning, ICML 2022, 17-23 July 2022, Balti-
more, Maryland, USA , volume 162 of Proceedings of Ma-
chine Learning Research , 2206–2240. PMLR.
Chan, C.; Xu, C.; Yuan, R.; Luo, H.; Xue, W.; Guo, Y .; and
Fu, J. 2024. RQ-RAG: Learning to Refine Queries for Re-
trieval Augmented Generation. CoRR , abs/2404.00610.
DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.;
Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang,
X.; Yu, X.; Wu, Y .; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao,Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.;
Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.;
Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li,
G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin,
H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen,
J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen,
J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.;
Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu,
L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang,
M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen,
Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen,
R. J.; Jin, R. L.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye,
S.; Wang, S.; Yu, S.; Zhou, S.; Pan, S.; and Li, S. S. 2025.
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. CoRR , abs/2501.12948.
Erdogan, L. E.; Lee, N.; Kim, S.; Moon, S.; Furuta, H.; Anu-
manchipalli, G.; Keutzer, K.; and Gholami, A. 2025. Plan-
and-Act: Improving Planning of Agents for Long-Horizon
Tasks. arXiv:2503.09572.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Guo, Q.; Wang, M.; and Wang, H. 2024. Retrieval-
Augmented Generation for Large Language Models: A Sur-
vey. arXiv:2312.10997.
Grok. 2025. Grok 3 Beta — The Age of Reasoning Agents.
https://x.ai/news/grok-3.
Hou, X.; Yang, M.; Jiao, W.; Wang, X.; Tu, Z.; and Zhao,
W. X. 2024. CoAct: A Global-Local Hierarchy for Au-
tonomous Agent Collaboration. arXiv:2406.13381.
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Za-
mani, H.; and Han, J. 2025. Search-r1: Training llms to rea-
son and leverage search engines with reinforcement learn-
ing. arXiv preprint arXiv:2503.09516 .
Jin, J.; Zhu, Y .; Yang, X.; Zhang, C.; and Dou, Z. 2024a.
FlashRAG: A Modular Toolkit for Efficient Retrieval-
Augmented Generation Research. CoRR , abs/2405.13576.
Jin, J.; Zhu, Y .; Zhou, Y .; and Dou, Z. 2024b. BIDER:
Bridging Knowledge Inconsistency for Efficient Retrieval-
Augmented LLMs via Key Supporting Evidence. In Ku, L.;
Martins, A.; and Srikumar, V ., eds., Findings of the Asso-
ciation for Computational Linguistics, ACL 2024, Bangkok,
Thailand and virtual meeting, August 11-16, 2024 , 750–761.
Association for Computational Linguistics.
Kim, S.; Moon, S.; Tabrizi, R.; Lee, N.; Mahoney, M.;
Keutzer, K.; and Gholami, A. 2023. An LLM Compiler for
Parallel Function Calling. arXiv .
Lewis, P. S. H.; Perez, E.; Piktus, A.; Petroni, F.;
Karpukhin, V .; Goyal, N.; K ¨uttler, H.; Lewis, M.; tau
Yih, W.; Rockt ¨aschel, T.; Riedel, S.; and Kiela, D. 2020.
Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks. In Advances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Pro-
cessing Systems 2020, NeurIPS 2020, December 6-12, 2020,
virtual .

Li, X.; Dong, G.; Jin, J.; Zhang, Y .; Zhou, Y .; Zhu, Y .; Zhang,
P.; and Dou, Z. 2025a. Search-o1: Agentic Search-Enhanced
Large Reasoning Models. CoRR , abs/2501.05366.
Li, X.; Jin, J.; Dong, G.; Qian, H.; Zhu, Y .; Wu, Y .; Wen,
J.; and Dou, Z. 2025b. WebThinker: Empowering Large
Reasoning Models with Deep Research Capability. CoRR ,
abs/2504.21776.
Li, X.; Jin, J.; Zhou, Y .; Zhang, Y .; Zhang, P.; Zhu, Y .; and
Dou, Z. 2024. From Matching to Generation: A Survey on
Generative Information Retrieval. CoRR , abs/2404.14851.
Mialon, G.; Fourrier, C.; Wolf, T.; LeCun, Y .; and Scialom,
T. 2024. GAIA: a benchmark for General AI Assistants.
InThe Twelfth International Conference on Learning Rep-
resentations, ICLR 2024, Vienna, Austria, May 7-11, 2024 .
OpenReview.net.
OpenAI. 2024a. Learning to Reason with LLMs. https://
openai.com/index/learning-to-reason-with-llms.
OpenAI. 2024b. Learning to Reason with LLMs. https:
//openai.com/index/learning-to-reason-with-llms.
OpenAI. 2025. Introducing deep research. https://openai.
com/index/introducing-deep-research.
Phan, L.; Gatti, A.; Han, Z.; Li, N.; Hu, J.; Zhang, H.; Shi,
S.; Choi, M.; Agrawal, A.; Chopra, A.; Khoja, A.; Kim,
R.; Hausenloy, J.; Zhang, O.; Mazeika, M.; Anderson, D.;
Nguyen, T.; Mahmood, M.; Feng, F.; Feng, S. Y .; Zhao, H.;
Yu, M.; Gangal, V .; Zou, C.; Wang, Z.; Wang, J. P.; Kumar,
P.; Pokutnyi, O.; Gerbicz, R.; Popov, S.; Levin, J.; Kazakov,
M.; Schmitt, J.; Galgon, G.; Sanchez, A.; Lee, Y .; Yeadon,
W.; Sauers, S.; Roth, M.; Agu, C.; Riis, S.; Giska, F.; Ut-
pala, S.; Giboney, Z.; Goshu, G. M.; of Arc Xavier, J.; Crow-
son, S.; Naiya, M. M.; Burns, N.; Finke, L.; Cheng, Z.; Park,
H.; Fournier-Facio, F.; Wydallis, J.; Nandor, M.; Singh, A.;
Gehrunger, T.; Cai, J.; McCarty, B.; Duclosel, D.; Nam, J.;
Zampese, J.; Hoerr, R. G.; Bacho, A.; Loume, G. A.; Galal,
A.; Cao, H.; Garretson, A. C.; Sileo, D.; Ren, Q.; Cojoc,
D.; Arkhipov, P.; Qazi, U.; Li, L.; Motwani, S.; de Witt,
C. S.; Taylor, E.; Veith, J.; Singer, E.; Hartman, T. D.; Ris-
sone, P.; Jin, J.; Shi, J. W. L.; Willcocks, C. G.; Robinson,
J.; Mikov, A.; Prabhu, A.; Tang, L.; Alapont, X.; Uro, J. L.;
Zhou, K.; de Oliveira Santos, E.; Maksimov, A. P.; Vendrow,
E.; Zenitani, K.; Guillod, J.; Li, Y .; Vendrow, J.; Kuchkin,
V .; and Ze-An, N. 2025. Humanity’s Last Exam. CoRR ,
abs/2501.14249.
Shao, Z.; Gong, Y .; Shen, Y .; Huang, M.; Duan, N.; and
Chen, W. 2023. Enhancing Retrieval-Augmented Large
Language Models with Iterative Retrieval-Generation Syn-
ergy. arXiv:2305.15294.
Song, H.; Jiang, J.; Min, Y .; Chen, J.; Chen, Z.; Zhao, W. X.;
Fang, L.; and Wen, J.-R. 2025. R1-searcher: Incentiviz-
ing the search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592 .
Team, Q. 2024. Qwq: Reflect deeply on the boundaries of
the unknown. Hugging Face .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2023. Interleaving Retrieval with Chain-of-Thought Rea-
soning for Knowledge-Intensive Multi-Step Questions. InRogers, A.; Boyd-Graber, J. L.; and Okazaki, N., eds., Pro-
ceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), ACL
2023, Toronto, Canada, July 9-14, 2023 , 10014–10037. As-
sociation for Computational Linguistics.
Wan, Z.; Li, Y .; Wen, X.; Song, Y .; Wang, H.; Yang, L.;
Schmidt, M.; Wang, J.; Zhang, W.; Hu, S.; and Wen, Y . 2025.
ReMA: Learning to Meta-think for LLMs with Multi-Agent
Reinforcement Learning. arXiv:2503.09501.
Wang, L.; Xu, W.; Lan, Y .; Hu, Z.; Lan, Y .; Lee, R. K.-W.;
and Lim, E.-P. 2023. Plan-and-Solve Prompting: Improving
Zero-Shot Chain-of-Thought Reasoning by Large Language
Models. arXiv:2305.04091.
Wang, X.; Chen, Y .; Yuan, L.; Zhang, Y .; Li, Y .; Peng, H.;
and Ji, H. 2024. Executable code actions elicit better llm
agents. In Forty-first International Conference on Machine
Learning .
Wei, J.; Karina, N.; Chung, H. W.; Jiao, Y . J.; Papay, S.;
Glaese, A.; Schulman, J.; and Fedus, W. 2024. Mea-
suring short-form factuality in large language models.
arXiv:2411.04368.
Wu, J.; Yin, W.; Jiang, Y .; Wang, Z.; Xi, Z.; Fang, R.; Zhang,
L.; He, Y .; Zhou, D.; Xie, P.; and Huang, F. 2025. Web-
Walker: Benchmarking LLMs in Web Traversal. CoRR ,
abs/2501.07572.
Xu, F.; Shi, W.; and Choi, E. 2023. RECOMP: Improving
Retrieval-Augmented LMs with Compression and Selective
Augmentation. CoRR , abs/2310.04408.
Xu, J.; Guo, Z.; He, J.; Hu, H.; He, T.; Bai, S.; Chen, K.;
Wang, J.; Fan, Y .; Dang, K.; Zhang, B.; Wang, X.; Chu,
Y .; and Lin, J. 2025. Qwen2.5-Omni Technical Report.
arXiv:2503.20215.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou,
F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.;
Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,
J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.;
Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,
R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,
X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y .; Su, Y .; Zhang,
Y .; Zhang, Y .; Wan, Y .; Liu, Y .; Wang, Z.; Cui, Z.; Zhang,
Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report.
arXiv:2505.09388.
Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.;
Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu,
K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.;
Zhu, Q.; Men, R.; Lin, R.; Li, T.; Xia, T.; Ren, X.; Ren, X.;
Fan, Y .; Su, Y .; Zhang, Y .; Wan, Y .; Liu, Y .; Cui, Z.; Zhang,
Z.; and Qiu, Z. 2024. Qwen2.5 Technical Report. CoRR ,
abs/2412.15115.
Yao, S.; Zhao, J.; Yu, D.; Shafran, I.; Narasimhan, K. R.; and
Cao, Y . 2022. ReAct: Synergizing Reasoning and Acting in
Language Models. In NeurIPS 2022 Foundation Models for
Decision Making Workshop .

Zhang, Y .; Dou, Z.; Li, X.; Jin, J.; Wu, Y .; Li, Z.; Ye, Q.;
and Wen, J.-R. 2025. Neuro-Symbolic Query Compiler.
arXiv:2505.11932.

Appendix
Datasets
(1)GAIA : A benchmark for evaluating general-purpose AI assistants, where questions require multi-step reasoning and external
information retrieval. We use all samples from the validation set and categorize them into three types based on input and
required information: text-only, multimodal, and with-file. Specifically, with-file represents queries where the input question
comes with accompanying files, text-only represents queries that only require text capabilities, which are filtered through
annotated metadata, and the remaining queries are categorized into multimodal. (2) WebWalkerQA : A benchmark designed to
evaluate models’ ability to navigate web pages and extract information. Questions require aggregating knowledge from single
or multiple web pages and cover both English and Chinese queries. We sample 200 questions from the test set. (3) SimpleQA :
A benchmark testing models’ factual accuracy, examining knowledge breadth and factual capabilities. Since models inherently
suffer from hallucination issues and may struggle on this dataset, this benchmark effectively tests models’ ability to answer
questions by incorporating external knowledge. We sample 200 questions from the test set. (4) Humanity’s Last Exam : A high-
difficulty dataset requiring complex reasoning, containing academic problems across multiple domains including mathematics,
physics, and computer science. Models need to retrieve relevant external information to support their reasoning. We sample 500
questions from the validation set.
Evaluation Metrics
To fairly evaluate the effectiveness of our methods, we adopted the LLM-as-Judge approach across all datasets, using Qwen2.5-
72B-Instruct to assess the consistency between model-generated answers and golden answers. The instruction used here follows
Webthinker (Li et al. 2025b), as shown in following instruction . It should be noted that in the construction process of the Web-
Walker dataset we used, each question’s answer is derived solely from the corresponding annotated webpage, without collection
and organization from the web, which leads to biased and inconsistent golden answers. This results in models providing more
detailed answers being incorrectly judged as wrong (due to inconsistency with standard answers). Therefore, in evaluation of
WebWalker, we optimized the instruction template, as shown in following instruction .
Evaluation Instruction for Normal QA Tasks
You are an evaluation assistant. Please determine if the model output is equivalent to
the labeled answer.
Question: {question}
Labeled Answer: {labeled_answer}
Model Output: {pred_answer}
Did the model give an answer equivalent to the labeled answer? Please respond with
"Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not
include any other text.
Evaluation Instruction for WebWalker
You are an evaluation assistant. Please determine if the predicted answer is equivalent
to the labeled answer in terms of core content.
Question: {question}
Labeled Answer: {labeled_answer}
Predicted Answer: {pred_answer}
Evaluation Guidelines:
1. Mark as "Correct" if the predicted answer contains all key information from the
labeled answer, even if it provides additional details
2. Mark as "Correct" if the predicted answer matches the core meaning of the labeled
answer, regardless of different phrasing
3. Only mark as "Incorrect" if the predicted answer misses critical information or
contains incorrect information

Please respond with only "Correct" or "Incorrect". Do not include any other text.
Implementation Details
Baseline Details All our baselines are implemented based on QwQ-32B as the base model. The implementation details for
each baseline are described below:
•Vanilla RAG : We directly utilize the original question for search and then provide the retrieved results to the generator for
answer generation.
•Search-o1 : This approach integrates search capabilities during the reasoning process, dynamically generating queries and
performing searches. Our implementation is based on the official Search-o1 repository.
•WebThinker : This method enables dynamic searching, link clicking, and report generation during the reasoning process.
Since our evaluation does not involve report generation, we only utilize the question-answering functionality. Our imple-
mentation is likewise based on the official repository.
•CodeAct : This approach can invoke code tools to assist reasoning during the inference process. We employ a Python Sand-
box as the code tool and design custom prompts for implementation, with the instruction shown in following instruction .
•Multimodal Enhanced : Similar to CodeAct, we replace the Python Sandbox with the multimodal tool used in our method
(i.e., the Qwen2.5-omni-7B model) as a tool that can be dynamically invoked during reasoning, with the instruction shown
in following instruction .
•Plan-and-Solve : This method first comprehends the original question and generates an overall plan, then executes this plan
step-by-step until reaching the final result. The instruction for plan generation is shown in following instruction . During
plan execution, to ensure fair comparison with our method, we automatically determine the most appropriate expert agent
for each step in the plan through the model, using the same set of expert agents as in our approach.
•ReAct : This approach allows the model to directly utilize multiple tools as needed during reasoning to assist inference. Tool
descriptions are provided to the model at the beginning, and tool outputs are directly inserted into the chain of thought.
Instruction Template for Query Planning in Simple Search Agent
You are a reasoning assistant. Your task is to generate a detailed query plan for
answering the user's question by breaking it down into sub-queries. Each query should
be a precise and suitable query for web search.
Please analyze the question and break it down into multiple sub-queries that will help
gather all the necessary information to answer it completely.
Remember:
1. Each sub-query should be a precise and suitable query for web search.
2. Subqueries should cover different aspects to avoid duplicate searches.
3. Each subquery needs to be necessary, do not add irrelevant ones. You need to use as
few subqueries as possible to obtain accurate results.
Output your query plan in JSON format as follows:
```json
{{
"query_plan": [
"sub-query-1",
"sub-query-2",
...
]
}}
```
Task: {question}

Instruction Template for Answer Generation in Simple Search Agent
You are a knowledgeable assistant that uses the provided documents and previous
exploration results as your current memory to answer the user's question step by step.
You need to carefully analyze the given context step by step and then provide a
reliable, detailed answer to the task. If you find that there is no valid information
in the document, please directly state that there is no valid information.
Your answer needs to include specific analysis and relevant citations (detail page url
and title).
Question: {question}
Documents: {documents}
Instruction Template for Answer Generation in Computation Reasoning Agent
You are a reasoning assistant with the ability to execute Python code to help you
answer the user's question accurately.
You have special tools:
- To execute a python code: write <|begin_code_call|>python your python code here
<|end_code_call|>. Then, the Python code interpreter will execute the python code, and
provide you with execution result in the format <|begin_code_call_result|> ...code call
results... <|end_code_call_result|>. If the code encounters errors while running, try
to fix them and give another code call. You can repeat the code execution process
multiple times if necessary. The maximum number of code execution attempts is limited
to {MAX_CODE_CALL_NUM}. Once you have all the information you need, continue your
reasoning. You are given previous exploration results as your current memory, please
first check the current memory for relevant information before making search. But you
need to be aware that the facts involved in memory may not be comprehensive.
Example: {example}
Remember:
- Use <|begin_code_call|> to start Python code and end with <|end_code_call|>.
- Always explain your reasoning before and after executing code.
- When done with code execution, provide a clear final answer or solution. Please
answer the following question step by step. When encountering scenarios that require
computation, try to use code for verification instead of solely using own knowledge.
You should provide your final answer in the format \boxed{{YOUR_ANSWER}}.
Current Memory: {current_memory}
Question: {task_info}
Instruction Template for Answer Generation in Multimodal Reasoning Agent
You are an AI assistant with multimodal understanding capabilities. You can analyze
images, video and audio to answer user questions.
You have access to a special multimodal tool: - To analyze an image/video/audio and
answer questions about it, use the format: <|begin_multimodal_call|> data: [path of
image/video/audio] question: [your specific question] <|end_multimodal_call|> The
system will provide analysis results in the format: <|begin_multimodal_result|>
...analysis results... <|end_multimodal_result|> You can ask multiple questions about
different aspects of the image/video/audio if needed. The maximum number of multimodal
analysis calls is limited to {MAX_MM_CALL_NUM}.
You are given previous analysis results as your current memory. Please check the
current memory for relevant information before making new analysis requests. Example:
{example}
Remember: - Always explain your reasoning before and after multimodal analysis -
Provide your final answer in \boxed{YOUR_ANSWER} format

Current Memory: {current_memory}
Question: {task_info}
Instruction Template for Plan-and-Act
You are a reasoning assistant. For the given task, come up with a simple step by step
plan.
This plan should involve individual tasks, that if executed correctly will yield the
correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has
all the information needed - do not skip steps.
Give your plan in JSON format as follows:
```json
{{
"plan": [
"step1", # each step is a string of the task description
"step2",
...
]
}}
Remember:
1. Each step should be a self-contained task, describe the task in detail.
2. The result of the final step should be the final answer.
3. Output your plan in JSON format.
```
Task:
{question}
Details of Domain-Specific Agents In our experiments, we design four types of expert agents to implement different func-
tionalities and accomplish various types of tasks. The specific design details for all agents are as follows:
•Simple Search Agent : Responsible for searching and integrating information on the internet, suitable for simple and efficient
information gathering and fact verification. Its workflow includes understanding the given subtask and generating a query
plan, then conducting searches and collecting information to provide answers. The instruction templates for query planning
and answer generation are shown in following instruction and following instruction , respectively.
•Deep Search Agent : Responsible for conducting deep exploration and information collection on the internet, requiring capa-
bilities such as utilizing search engines and clicking webpage links. We directly adopt the implementation of WebThinker (Li
et al. 2025b) as our deep search agent.
•Computational Reasoning Agent : This agent can autonomously invoke a Python interpreter during the reasoning process
to assist its thinking and information collection. The covered functionalities include but are not limited to: reading files,
downloading files, and obtaining webpage information through code. We follow the approach of Search-o1 (Li et al. 2025a),
using the QwQ-32B reasoning model to drive a Python tool encapsulated in a sandbox, with the specific instruction shown
in following instruction .
•Multimodal Agent : Possesses the capability to invoke multimodal tools during the reasoning process. To simplify deploy-
ment and reduce usage complexity, we employ Qwen-omni-7B as our multimodal tool, which can understand files of various
modalities. We embed it into the model’s reasoning process, enabling the model to invoke this tool during thinking to handle
multimodal scenarios, with the specific instruction shown in following instruction .
Details of Coordinator We employ Qwen2.5-32B-Instruct as the base model for our coordinator, which primarily incorpo-
rates three functions: (1) Reasoning Transfer Process : The model is assigned a subtask and subsequently identifies the most
suitable expert agent through classification. During this process, comprehensive information about all available expert agents
is provided to the model to facilitate its understanding of each agent’s capabilities. The specific instruction is formulated as
follows:

Instruction for Reasoning Transfer Process
You are an agent selection system. Analyze the given task and select the most suitable
agent based on:
1. Required Capabilities:
- What specific skills/knowledge does this task demand?
- How well does each agent's expertise match these requirements?
2. Task Difficulty:
- Complexity level (simple fact vs multi-step problem-solving). You should consider the
effective time cost of each agent.
- Depth of analysis needed (surface information vs deep exploration)
- You need to choose the model **that can complete the task **with the lowest
cost/complexity as much as possible.
**Only output the JSON format **with the following fields:
- reason: The reason for selecting the agent
- selected_agent_name: The name of the selected agent
Example Output: {example}
Agents Available: {agent_info}
Task: {task}
Analyze the task and respond **ONLY the json format **, without any additional
explanation.
(2)Reasoning Distillation Process : Upon completion of a subtask by an expert agent, the coordinator comprehends and
summarizes the agent’s reasoning trajectory, subsequently refining it into a distilled reasoning process and final conclusion for
reporting to the meta-planner. This mechanism enables the meta-planner to verify the logical validity of the reasoning pathway
and evaluate the derived results.
(3)Memory Construction : Beyond the conclusion, the coordinator extracts and records key findings from the expert agent’s
reasoning process into the global memory. Specifically, we design the coordinator to capture two memory components: facts and
resources. To enhance computational efficiency, this functionality is implemented within the same instruction as the reasoning
distillation process. The detailed instruction is formulated as follows:
Instruction for Reasoning Distillation Process and Memory Construction
You are a specialized Summarization Agent. Your role is to analyze a problem and a
model's thinking process, then produce a structured summary with two key components:
1. CONCLUSION: Create a concise string that captures:
- reasoning_process: A concise string outlining the necessary reasoning steps in
logical order, including key actions, searches, and findings.
- final_conclusion: A concise string that captures the final answer and conclusion
of the given task.
2. MEMORY: **Organize useful information for future tasks **(only include the
information that highly likely to be useful for future tasks, don't include current
memory facts):
- fact_memory: List important facts discovered during reasoning
*Each entry must include both content AND source
*Sources must be specific (exact URLs, specific document names, or "Model
Inference" for reasoned conclusions)
*Consolidate related facts into single entries to avoid fragmentation. Only keep
facts that are relevant to the future tasks.
*If no valid source is found, set it as [Source: Not Specified]
- resource_memory: Map useful resources as "description": "path" pairs
*Paths must be complete, valid URLs or precise variable names
*Descriptions should be clear and specific
*Include only verified, accessible resources

*If no valid resources exist, set resource_memory as an empty dictionary
Produce ONLY a properly formatted JSON object with these components. Include nothing
else.
Example output: {example}
Reasoning Chain:
{reasoning_chain}
Task: {task_description}
(4)Memory Utilization : During subtask delegation, the coordinator retrieves relevant memory segments from the global
memory and provides them to the expert agent alongside the subtask. The instruction is structured as follows:
Instruction for Memory Utilization
You are an assistant specialized in filtering memory based on a specific task. Your
task is to analyze the given memory and select ONLY the most task-relevant memories,
with a strict maximum limit of 5 entries.
Key Requirements:
1. Relevance First:
- Each selected memory MUST have a direct and strong connection to the current task
- Reject memories that are only tangentially or weakly related
- If there are fewer than 5 highly relevant memories, select only those that are
truly relevant
2. Quality Control:
- Filter out any memories with invalid or suspicious URLs
- Remove memories about failed attempts or negative experiences
- Exclude memories that contain speculative or unverified information
3. Output Format:
- Output the filtered memories in the following format:
```
Memory Fact 1: [memory1]
Memory Fact 2: [memory2]
...
```
Remember: It's better to return fewer but highly relevant memories than to include
marginally related ones just to reach 5 entries.
Memory:
{memory}
Task:
{task}
Filtered Memory:
Case Study
The following is a case from the GAIA dataset that demonstrates the capability of our method, shown in Table 3.

Table 3: An example of our method inference process from the GAIA dataset.
Example # from GAIA Dataset
Question:
In terms of geographical distance between capital cities, which 2 countries are the furthest from each other within the ASEAN bloc
according to wikipedia? Answer using a comma separated list, ordering the countries by alphabetical order.
Labeled Answer:
Indonesia, Myanmar
Inference Process
Okay, so the user wants to know which two ASEAN countries have their capital cities the farthest apart geographically. Hmm, first I need
to recall what countries are part of ASEAN. ... The first subtask call would be:
<begin callsubtask >Retrieve the list of ASEAN member countries along with their respective capital cities from Wikipedia.
<end callsubtask >
<begin subtask result>Result from Search Agent:...The list of ASEAN member countries and their respective capital cities is: 1. Brunei
- Bandar Seri Begawan, 2. Cambodia - Phnom Penh, 3. Indonesia - Jakarta, 4. Laos - Vientiane, 5. Malaysia - Kuala Lumpur, 6. Myanmar
- Nay Pyi Taw, 7. Philippines - Manila, 8. Singapore - Singapore, 9. Thailand - Bangkok, 10. Vietnam - Hanoi. <end subtask result>
Okay, got the list. Now, the next step is to get the coordinates for each capital. Since there are 10 capitals, doing a separate search for each
would take 10 subtasks, which is too many. So better to use code to batch process. ...Therefore, the second subtask is:
<begin callsubtask >Write Python code to retrieve the geographic coordinates of each ASEAN capital city from Wikipedia using the
Wikipedia API, calculate the pairwise distances between all capital cities using the Haversine formula, identify the pair with the maximum
distance, and return the two countries in alphabetical order. <end callsubtask >
<begin subtask result>Result from Code-Agent:Identified the list of ASEAN countries and their capitals. Wrote a function to fetch
geographic coordinates for each capital using the Wikipedia API. Implemented the Haversine formula to compute pairwise distances
between all capital cities. Tracked the pair of capitals with the largest distance. Sorted the two countries alphabetically and returned
them.Final Conclusion:The two ASEAN countries with the farthest capital cities, in alphabetical order, are Indonesia and Vietnam.
<end subtask result>
Wait, but is that correct? Let me think. Jakarta (Indonesia) and Hanoi (Vietnam). Are they the farthest apart?Alternatively, maybe Brunei
and Indonesia’s capitals are further apart? Let me think of approximate coordinates:Bandar Seri Begawan: approx 4.9°N, 115°EJakarta:
-6.2°S, 106.8°EDistance between these would be roughly from around 4.9N to -6.2S, so total latitude difference of 11.1 degrees....
Alternatively, since I can only use up to 10, and already used 2, perhaps proceed as follows:- Acknowledge that the first code result was
incorrect due to the capital name, so need to adjust.- Re-run the code with corrected capital names, but to do that, I need to specify the
correction in the code. Thus, the second code subtask would be:
<begin callsubtask >Write Python code to calculate the geographical distance between all pairs of ASEAN capital cities using their
precise coordinates obtained from Wikipedia, correcting the capital of Myanmar to ’Naypyidaw’. The countries and capitals are: Brunei
(Bandar Seri Begawan), Cambodia (Phnom Penh), Indonesia (Jakarta), Laos (Vientiane), Malaysia (Kuala Lumpur), Myanmar (Naypyi-
daw), Philippines (Manila), Singapore (Singapore), Thailand (Bangkok), Vietnam (Hanoi). Use the Haversine formula and return the two
countries with the farthest capitals in alphabetical order. <end callsubtask >
<begin subtask result>Result from Code-Agent:Corrected the capital of Myanmar to Naypyidaw. ...
The two countries with the farthest capitals in alphabetical order are (’Indonesia’, ’Myanmar’). <end subtask result>
Ah, that would be the correct result. Thus, the second code subtask, after correcting the capital name, gives Indonesia and Myanmar.But
since I already used two subtasks (the first search and the first code), and this is the third, that’s okay.Wait, but according to the initial plan,
the first code subtask might have failed because of the capital name, so I need to fix that....
But since the first code result was wrong, perhaps I need to adjust.So following this adjusted plan:First subtask (already done) gives the
initial list, but with a typo in Myanmar’s capital.Second subtask (search for correct capital name of Myanmar):
<begin callsubtask >Confirm the correct spelling of the capital city of Myanmar according to Wikipedia. <end callsubtask >
<begin subtask result>Result from Search Agent:Analyzed multiple documents to determine the correct spelling of Myanmar’s capital
city. Document 1 from Simple English Wikipedia primarily uses ’Naypyidaw’. Document 2 from Britannica uses ’Nay Pyi Taw’. Document
3 from Britannica also uses ’Nay Pyi Taw’ as the primary spelling. Document 4 from Wikiwand uses ’Naypyidaw’ as the primary spelling
and mentions ’Nay Pyi Taw’ as the officially romanized version. Document 5 from Britannica provides context on naming conventions
in Myanmar but does not directly address the capital city’s spelling.Final Conclusion:The correct spelling of the capital city of Myanmar
according to Wikipedia is ’Naypyidaw’. <end subtask result>
Okay, so the correct spelling is Naypyidaw.Now, use the code-agent with the corrected capital names.The code would now have the
correct list:capitals = [(’Brunei’, ’Bandar Seri Begawan’), (’Cambodia’, ’Phnom Penh’), (’Indonesia’, ’Jakarta’), (’Laos’, ’Vientiane’),
(’Malaysia’, ’Kuala Lumpur’), (’Myanmar’, ’Naypyidaw’), (’Philippines’, ’Manila’), (’Singapore’, ’Singapore’), (’Thailand’, ’Bangkok’),
(’Vietnam’, ’Hanoi’)]Then, the code can proceed to get their coordinates via the Wikipedia API with the correct names.Fourth subtask
(code):
<begin callsubtask >Write Python code to retrieve coordinates for each of the ASEAN capitals (using the corrected capital name for
Myanmar as Naypyidaw) from Wikipedia, compute pairwise distances with Haversine formula, identify the maximum distance pair, and
return them in alphabetical order. <end callsubtask >
<begin subtask result>Result from Code-Agent:The code attempts to scrape coordinates for ASEAN capitals from Wikipedia, ensuring
that Myanmar’s capital is correctly listed as Naypyidaw. ...The maximum distance pair between ASEAN capitals is Jakarta and Naypyidaw,
returned in alphabetical order. <end subtask result>
So the user’s answer should be boxed {Indonesia, Myanmar }