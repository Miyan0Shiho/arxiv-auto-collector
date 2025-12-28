# X-GridAgent: An LLM-Powered Agentic AI System for Assisting Power Grid Analysis

**Authors**: Yihan, Wen, Xin Chen

**Published**: 2025-12-23 21:36:20

**PDF URL**: [https://arxiv.org/pdf/2512.20789v1](https://arxiv.org/pdf/2512.20789v1)

## Abstract
The growing complexity of power system operations has created an urgent need for intelligent, automated tools to support reliable and efficient grid management. Conventional analysis tools often require significant domain expertise and manual effort, which limits their accessibility and adaptability. To address these challenges, this paper presents X-GridAgent, a novel large language model (LLM)-powered agentic AI system designed to automate complex power system analysis through natural language queries. The system integrates domain-specific tools and specialized databases under a three-layer hierarchical architecture comprising planning, coordination, and action layers. This architecture offers high flexibility and adaptability to previously unseen tasks, while providing a modular and extensible framework that can be readily expanded to incorporate new tools, data sources, or analytical capabilities. To further enhance performance, we introduce two novel algorithms: (1) LLM-driven prompt refinement with human feedback, and (2) schema-adaptive hybrid retrieval-augmented generation (RAG) for accurate information retrieval from large-scale structured grid datasets. Experimental evaluations across a variety of user queries and power grid cases demonstrate the effectiveness and reliability of X-GridAgent in automating interpretable and rigorous power system analysis.

## Full Text


<!-- PDF content starts -->

1
X-GridAgent: An LLM-Powered Agentic AI
System for Assisting Power Grid Analysis
Yihan (Logon) Wen, Xin Chen
Abstract—The growing complexity of power system operations
has created an urgent need for intelligent, automated tools to
support reliable and efficient grid management. Conventional
analysis tools often require significant domain expertise and
manual effort, which limits their accessibility and adaptability.
To address these challenges, this paper presentsX-GridAgent, a
novel large language model (LLM)-powered agentic AI system
designed to automate complex power system analysis through
natural language queries. The system integrates domain-specific
tools and specialized databases under a three-layer hierarchi-
cal architecture comprising planning, coordination, and action
layers. This architecture offers high flexibility and adaptability
to previously unseen tasks, while providing a modular and
extensible framework that can be readily expanded to incorporate
new tools, data sources, or analytical capabilities. To further
enhance performance, we introduce two novel algorithms: (i)
LLM-driven prompt refinement with human feedback, and (ii)
schema-adaptive hybrid retrieval-augmented generation (RAG)
for accurate information retrieval from large-scale structured
grid datasets. Experimental evaluations across a variety of user
queries and power grid cases demonstrate the effectiveness
and reliability ofX-GridAgentin automating interpretable and
rigorous power system analysis.
Index Terms—Large language models, agentic AI, automated
power grid analysis, hierarchical architecture, grid agent.
I. INTRODUCTION
MODERN power systems are undergoing a profound
transformation driven by factors such as the high
penetration of renewable energy, the widespread deployment
of distributed energy resources [1], the integration of large
data center loads [2], increasing stakeholder involvement,
and ambitious decarbonization goals [3]. These developments
have substantially increased the complexity of power system
planning, operation, and analysis in dynamic and intercon-
nected environments. Moreover, conventional power system
analysis and decision-making require significant manual effort
and deep domain expertise. Engineers and operators need
to navigate multiple specialized software tools and datasets
for computation, simulation, and analysis, which are time-
consuming, costly, and labor-intensive. As system complexity
continues to grow, this manual and fragmented workflow
becomes increasingly unsustainable. It underscores the urgent
need for advanced artificial intelligence (AI)-driven tools [4],
[5] that can automate routine tasks, streamline analysis, and
support reliable and efficient grid operations.
In particular, large language models (LLMs) [6], such as
OpenAI’s GPT [7] and Google’s Gemini [8], have recently
demonstrated remarkable capabilities in reasoning, knowledge
integration, and context comprehension and generation across
Y . Wen and X. Chen are with the Department of Electrical and Com-
puter Engineering, Texas A&M University, USA; correspondence email:
xinchen@tamu.edu.various domains [9], [10]. These advances in LLMs are driving
the rapid evolution of generative AI techniques and present
significant potential for supporting power system applications.
Recent studies have explored the use of LLMs to enhance
power system simulations [11], generate power grid models
[12], and visualize power networks [13], among others. A
comprehensive review of LLM applications in power systems
is provided in [14]. However, when applied to automate
complex domain-specific tasks, the direct use of LLMs reveals
several fundamental limitations. First, most existing LLMs are
trained to solve general-purpose problems and thus lack the
specialized domain knowledge, engineering expertise, and ac-
cess to proprietary or confidential data, which are essential for
real-world power system applications. Second, although LLMs
excel at processing and generating natural language, they
are not inherently designed for performing precise numerical
computations or addressing mathematically rigorous problems
such as power flow analysis, optimization, and control [15].
These limitations pose significant challenges to deploying
LLMs as standalone solutions in critical engineering domains
such as power systems.
To overcome these challenges, recent advances have intro-
ducedagentic AIsystems [16], [17], which are LLM-based
autonomous agents equipped with external tools, domain-
specific databases, and enhanced reasoning capabilities. Unlike
standalone LLMs, agentic AI systems can dynamically interact
with structured data sources, plan and execute multi-step work-
flows, and perform specialized computations using domain-
specific tools. A key technique for these systems is Retrieval-
Augmented Generation (RAG) [18], [19], which allows LLMs
to retrieve and incorporate relevant information from local
specialized knowledge bases, thus grounding responses in
accurate and domain-specific data. In addition, the emerging
open standard Model Context Protocol (MCP) [20], [21]
provides a unified framework for seamless communication
between LLM agents and external tools. MCP defines stan-
dardized interfaces and coordination mechanisms that enable
agents to access, query, and utilize software tools, application
programming interfaces (APIs), and data services in a modular
and extensible manner. The integration of these advanced
techniques transforms LLMs from passive text generators into
autonomous AI agents capable of reasoning, planning, and
acting to solve complex real-world tasks.
In this paper, we present a novel LLM-powered agentic AI
system, calledX-GridAgent, for automating comprehensive
power grid analysis using only natural language. This system
bridges the gap between high-level language interfaces and
complex power grid analysis tasks by integrating RAG, MCP,
domain-specific tools, and specialized databases under a hier-
archical reasoning architecture. X-GridAgent enables users toarXiv:2512.20789v1  [eess.SY]  23 Dec 2025

2
Fig. 1. The four key features of the proposed X-GridAgent system.
perform a wide range of professional power system analyses,
including power flow analysis, contingency analysis, optimal
power flow (OPF), short-circuit calculation, topology search,
and more. All the grid analyses are conducted using domain-
specific tools to ensure trustworthy and interpretable results,
and merely through conversational queries,withoutthe need
for direct interaction with simulation software or programming
environments. To achieve this, we develop a series of MCP-
based tool servers, each tailored to handle a distinct category of
grid analysis tasks (see Section II-B for details). By combining
the advanced reasoning capabilities of LLMs with professional
tools and structured knowledge bases, X-GridAgent aims to
automate complex power grid analysis and make it more ac-
cessible, interpretable, and efficient for engineers, researchers,
system operators, and other stakeholders.
Contributions.The key contributions of this work include:
1) We develop anintegratedagentic AI systemX-GridAgent
to automate complex power grid analysis through natural
language queries. Leveraging the reasoning capabilities of
LLMs, it dynamically generates task-specific workflows
and invokes domain-specific tools for rigorous power
system analysis. It also incorporates both short-term and
long-term memory for coherent multi-turn interactions
and contextual information retrieval from databases. The
key features of X-GridAgent are shown in Figure 1.
2) We design a novelthree-layer hierarchical architecture
for the X-GridAgent system, consisting of theplanning,
coordination, andactionlayers, to support complex,
multi-step power grid analysis tasks. As shown in Figure
2, the planning layer interprets user intents and generates
a plan of sequential tasks, the coordination layer routes
each task to a designated server and manages its execu-
tion, and the action layer interfaces with embedded tools
to perform computation or operation. This architecture of-
fers high flexibility and adaptability to previously unseen
tasks, and provides amodularandextensibleframework
that can be easily expanded to incorporate new tools, data
sources, or other analytical capabilities.
3) We propose two novel algorithms to further enhance
the performance of X-GridAgent: (i)LLM-driven prompt
refinement with human feedbackand (ii)schema-adaptive
hybrid RAG. The former leverages the LLM to automati-
cally construct and iteratively refine system prompts with
minimal input from human experts, which significantly
improves efficiency and consistency while substantiallyreducing the manual effort typically required for prompt
engineering. The latter addresses the limitations of con-
ventional RAG methods in retrieving information from
large-scale structured power system datasets; it dynam-
ically selects and reconstructs the most relevant data
tailored to the user query and employs a hybrid retrieval
strategy to improve accuracy and reliability.
Related Work.The application of LLM-based agentic AI
in the power systems domain has recently garnered growing
attention. Prior works have explored a range of use cases,
such as bidding strategy generation in electricity markets [22],
unit commitment and power dispatch [23]–[26], OPF modeling
and solving [27], [28], power grid control [29], and fault
analysis [30]. These studies primarily focus on a single type
of power system decision-making task, without addressing the
full pipeline required for comprehensive and automated power
grid analysis. Some works generate power system decisions
directly from LLMs, without invoking domain-specific tools
or validated code-based methods. Such approaches often lack
grounding in physical laws and engineering constraints, and
therefore cannot guarantee trustworthy or interpretable results.
In the context of power grid analysis, reference [31] proposes
an LLM-based multi-agent AI system that can perform OPF
and contingency analysis via natural language queries and
function calls. However, this system remains an early-stage
prototype with limited functionality and a simple architecture
that may not support complex multi-step grid analysis tasks.
In [32], an agentic AI system is developed to perform distribu-
tion grid analysis, while annotated expert-built workflows are
incorporated to enhance its generalizability to unseen tasks.
In contrast, X-GridAgent offers several key advantages over
existing agentic AI systems for power grid analysis: (i) it
supports comprehensive grid analysis functionality, together
with effective memory management and information retrieval
capabilities; (ii) its hierarchical architecture and modularized
server design enable flexible handling of complex and previ-
ously unseen tasks, while remaining easily extensible to new
data sources and functionalities; and (iii) it integrates auto-
matic prompt refinement and customized RAG techniques that
significantly enhance its scalability, efficiency, and response
quality. A demonstration video showcasing our X-GridAgent
system in action is available online [33].
The remainder of this paper is organized as follows. Sec-
tion II outlines the hierarchical architecture of X-GridAgent
with the three key layers and specialized servers. Section III
introduces our innovative methods on prompt refinement and
RAG. Section IV presents the system implementation and test
experiments. Conclusions are drawn in Section V.
II. X-GRIDAGENTARCHITECTURE ANDDESIGN
This section presents the hierarchical architecture of X-
GridAgent comprising three fundamental layers:planning,
coordination, andaction. A series of specialized X-GridAgent
servers is designed to perform rigorous power system analysis
by integrating with professional tools and external databases.
A. Overview of X-GridAgent Architecture
The X-GridAgent system features a novel three-layer hierar-
chical architecture, comprising theplanning,coordination, and

3
Fig. 2. Overview of the three-layer hierarchical architecture of X-GridAgent.
actionlayers, which are designed to jointly address complex
power grid analysis tasks. The X-GridAgent architecture is
illustrated in Figure 2. When a natural-language user query
qis submitted, the planning layerPfirst interprets the query
and generates a workflow plan that decomposes the query into
a sequence of tasks to resolve it. Each task is routed to its
corresponding domain-specific server for execution. Then, the
coordination layerCmanages the sequential execution of tasks
and monitors their progress. It also maintains a short-term
memory moduleM, which stores key intermediate results and
relevant contextual information to facilitate effective informa-
tion exchange across tasks and ensure consistency throughout
the workflow. The actual execution of each task is handled
by the action layerA, which selects appropriate tools within
the corresponding server to perform professional computations
and actions. It also incorporates a reflection mechanism to
determine whether the task has been successfully completed.
Note that the X-GridAgent workflow is not fixed but is
dynamically generated based on the specific user queryq.
This three-layer hierarchical architecture enhances the flexi-
bility, reliability, and scalability for automating power system
analysis. Below, we elaborate on the domain-specific servers
in X-GridAgent and the three integrated layers (P,C,A).
B. X-GridAgent Servers
To endow X-GridAgent with rigorous power system analysis
capabilities and domain-specific expertise, we develop a series
of specialized servers that integrate professional computational
tools, structured databases, and technical documentation. This
integration is implemented using MCP [20], [21], an emerging
open standard that provides a unified framework for seamless
communication between LLM agents and external tools or data
sources. Specifically, tools refer to callable external functions
or APIs that AI agents can invoke to retrieve information, per-
form computations, or execute actions. These tool-integrated
servers constitute X-GridAgent’s core capabilities.In particular, built on the open-source power system soft-
warePandapower[34], we have implemented eight modular
MCP-based servers, each tailored to handle a distinct category
of domain-specific tasks and computational problems. These
X-GridAgent servers include (1) theRetrievalserver,
which retrieves relevant information from embedded docu-
mentation and power grid datasets using advanced RAG tech-
niques; (2) thePowerFlowserver, which performs AC and
DC power flow studies; (3) theOPFserver, which solves OPF
problems to determine optimal generation dispatch and costs
under network constraints; (4) theContingencyserver,
which conducts N-1 contingency analysis on power lines or
transformers and identifies violations of operational limits;
(5) theShortCircuitserver, which computes short-circuit
currents in accordance with the IEC 60909 standard [35] under
common fault types; (6) theTopologyserver, which sup-
ports connectivity assessment, island detection, and network
traversal analysis; (7) theEditserver, which modifies grid
parameters and configurations (e.g., load values, voltage limits,
and line addition or removal); and (8) thePlotserver, which
visualizes power networks and plots related datasets.
Each server described above is associated with a set of
related tools (or functions), enabling flexible execution and
extended functionality. These tools can be dynamically in-
voked accroding to the user query. For example, in the
PowerFlowserver, the functionssolve ACpowerflow()and
solve DC powerflow()are available to perform AC and DC
power flow calculations, respectively. Users can also specify
input parameters for these functions directly through natural-
language queries. For instance, a user may request a specific
solution algorithm, such as Newton-Raphson or fast-decoupled
methods, by simply stating this preference in the query; it is
then automatically translated into appropriate input parameters
for tool invocation. In addition, two default functions,NetInit()
andNetSave(), are included in all servers to support the loading
and saving of power network cases for analysis.
By decomposing a user query into a chain of structured tasks

4
and routing each task to a specialized server for execution, X-
GridAgent can flexibly and effectively handle complex power
grid analysis queries. Moreover, ourmodularizedserver frame-
work is inherently extensible, as new capabilities can be in-
tegrated easily by deploying additional servers or augmenting
existing ones with more tools (functions). This design enables
X-GridAgent to continuously evolve in response to emerging
analytical requirements and advancements in domain-specific
tools, allowing it to remain adaptable, up-to-date, and flexible
for supporting a wide range of power system applications.
C. Planning Layer
Given a user queryqas input, the planning layerPis
designed to interpret its intent and generate an executable
step-by-step plan composed of logically structured tasks to
address the query. Specifically, the planning layerPcombines
the user queryqwith a well-calibrated system promptπ∗
Pto
form an augmented input, which is then passed to the LLM for
reasoning and plan generation. See Section III-A for a detailed
introduction to our prompt design approach. Guided by the
system prompt for planningπ∗
Pand leveraging the LLM’s
reasoning capabilities, the planning layerPdecomposes the
queryqinto a chain of tasksΓexecuted in sequence:
Γ:={T 1→ T 2→ ··· → T N}=P(q;π∗
P),(1)
where each taskT iis configured with a specific objective and
is linked to a designated X-GridAgent server for execution.
For example, consider an illustrative multi-step user query:
Query 1.“Can you run AC power flow on the IEEE 118-
bus grid to find the top 3 most heavily loaded lines, and then
run contingency analysis on those lines to see if there are any
operational limit violations?”
In response to Query 1, the planning layerPgenerates the
following sequence of tasks defined by server-objective pairs:
•[Task 1]:T 1={Server:PowerFlow, Objective:Run AC
power flow analysis on the IEEE 118-bus system}.
•[Task 2]:T 2={Server:Retrieval, Objective:Retrieve
the top three power lines with the highest loading ratios
from the power flow results}.
•[Task 3]:T 3={Server:Contingency, Objective:Run
N-1 contingency analysis by individually tripping the three
lines identified in the previous step and report any opera-
tional limit violations}.
In this way, unstructured natural-language user queries are
translated into structured executable workflows represented as
sequences of multi-step tasks by the planning layerP. These
workflows are not fixed but are dynamically generated ac-
cording to specific user queries, enabling flexible, explainable,
and automated power grid analysis. The chain of tasksΓis
then passed to the coordination layerCfor management and
execution, as detailed in the next subsection.
D. Coordination Layer
The coordination layerCorchestrates the execution of the
chain of tasksΓgenerated by the planning layer. It routes each
task to the designated X-GridAgent server, manages the inputsand outputs of each step, and maintains short-term memory to
track intermediate results and contextual information through-
out the execution process. The actual execution of each task
within a X-GridAgent server is carried out by the action layer
A, which is introduced in Section II-E.
Short-Term Memory Management. The tasks in the planned
sequenceΓare logically connected, with the output of each
preceding task providing necessary context and numerical
results for those that follow. To support these inter-task depen-
dencies, the coordination layer maintains a short-term memory
moduleM, which is incrementally updated after each task
and exposes relevant information to downstream tasks. As
illustrated in Figure 2, for each taskT i(i= 1,2,···), the
current memoryM i−1is concatenated with the task to form
an augmented input to the designated server for execution.
The initial memoryM 0captures historical conversations and
prior results. After taskT icompletes, its outputy iis incor-
porated into the short-term memory to produce an updated
Mi, which is then used by the subsequent taskT i+1. This
short-term memory management mechanism enables seamless
information flow across tasks, maintains an up-to-date record
of contextual information, and supports multi-turn interactions
with coherent responses to successive user queries.
E. Action Layer
By employing MCP and leveraging the advanced reasoning
capabilities of the LLM, the action layerAautonomously
selects and invokes the appropriate tools in the designated X-
GridAgent server and manages their input-output data flows
to fulfill a task. As illustrated in Figure 2, the workflow within
the action layerAis iterative and adapts to each taskT i
(i= 1,2,···). Specifically, the action layerAis provided
with a list of available tools and their capabilities from the
designated X-GridAgent server. Based on the specific task
contextT iand the current short-term memoryM i−1, it selects
an appropriate tool for execution and obtains the corresponding
results. Areflectionmechanism [36] is designed to evaluate
whether the task has been successfully solved, leveraging the
LLM’s reasoning to assess alignment with the task objective.
If the task is deemed complete, an LLM-based summarization
module generates a concise and relevant outputy i. If not, the
evaluation outcome, together with prior computational results,
is fed back into the reasoning process to guide the selection of
another tool for continued execution. This process continues
until the cumulative outcomes sufficiently address the current
taskT ior the maximum number of iterations is reached. The
final outputy ifor taskT iis then concatenated withM i−1to
produce an updated short-term memoryM i.
Specifically, in iterationk, the above MCP-based tool se-
lection and invocation process defines a mapping:
pi
k=A(T i,Mi, pi
k−1;π∗
A),(2)
wherepi
krepresents the cumulative outcome at iterationkfor
taskiafter a tool invocation. It records prior results and serves
as the input context for the next tool invocation, ensuring
that reasoning remains stateful and coherent across the tool
sequence.π∗
Adenotes the well-calibrated system prompt for
the action layer. Then, the reflection mechanism compares the

5
cumulative outcomepi
kwith the task contextT i, given by:
rk=Reflect(pi
k,Ti),(3)
wherer krepresents the reflection result, indicating whether
the current outcome sufficiently completes the task and iden-
tifying any missing information required for completion. The
reflection outcomer kis then used to guide the next reasoning
step, influencing subsequent tool selection.
Illustrative Example. To illustrate the detailed process, we
consider the execution of Task 3 (T 3) in theContingency
server to address Query 1 presented in Section II-C. After
executing Tasks 1 and 2, the top three power lines with the
highest loading ratios are identified as line 6, line 7, and line
34. Accordingly, the action layerAautonomously invokes a
sequence of functions to complete Task 3, as listed below:
1.NetInit(“analyzednet”: “ieee118”)
2.run contingency(“nminus1 cases”: “line”: “index”: [6])
3.run contingency(“nminus1 cases”: “line”: “index”: [7])
4.run contingency(“nminus1 cases”: “line”: “index”: [34])
5.NetSave()
It first loads the IEEE 118-bus system from an embedded grid
database using the function “NetInit()” and then iteratively
invokes the function “run contingency()” three times with cor-
rect parameter arguments to perform N-1 contingency analysis
on the three identified lines and obtain the limit violation
results. The reflection module assesses whether the task has
been successfully completed after each step and consistently
reports non-completion as well as the missing results until Step
4 has been executed. Lastly, “NetSave()” is invoked by default
to save the power network data and computational results.
III. KEYTECHNICALINNOVATIONS
This section introduces two novel algorithms for enhancing
X-GridAgent’s performance: (1)LLM-driven prompt refine-
ment with human feedbackand (2)schema-adaptive hybrid
RAG. The former leverages the LLM to automatically con-
struct and iteratively refine system prompts with light input
from human experts, guiding X-GridAgent in context-aware
reasoning and appropriate tool use. The latter addresses the
limitations of conventional RAG methods in retrieving in-
formation from large-scale structured power grid datasets by
dynamically selecting the most relevant data tailored to the
user query and using a hybrid retrieval strategy.
A. LLM-Driven Prompt Refinement with Human Feedback
Prompt design [37] is critical in shaping the reasoning be-
havior of LLM-based AI agents: it offers contextual grounding,
structures the reasoning process, specifies output formats, and
communicates available tools and usage guidelines. However,
manually crafting system prompts for complex domains such
as power grid analysis is labor-intensive and often driven by
subjective intuition, leading to inconsistencies and limited gen-
eralizability across tasks. Moreover, misalignment between hu-
man phrasing and machine interpretation can further degrade
response accuracy and efficiency. To address these challenges,
we propose anLLM-driven prompt refinement with human
feedbackframework for constructing and refining system
prompts. The framework leverages the LLM’s capabilities incomprehension, generation, and editing to calibrate prompts by
aligning X-GridAgent’s outputs with ground-truth references.
Human experts make minor revisions to ensure factual accu-
racy and correct detail-related errors. This LLM-led, human-
in-the-loop prompt refinement process progressively enhances
X-GridAgent’s reasoning stability and accuracy.
Specifically, letπdenote the system prompt. Given a user
queryq, the X-GridAgent system configured with promptπ
produces a final outputyas (4):
y= X-GridAgent(q;π).(4)
To calibrate the system promptπ, we use the LLM to automati-
cally generate a set of power system domain-specific queries in
natural language, supplemented with some manually authored
queries from human experts, to form a training query setQ :=
(qi)N
i=1. This set covers a variety of power system analysis
tasks, including power flow calculation, contingency analysis,
optimal power flow, grid information extraction, and more. For
each queryq i, a corresponding ground-truth reference answer
ˆyiis produced by human experts using professional tools,
which provide verified numerical results and domain-accurate
solutions. The resulting datasetD :={(q i,ˆyi)}N
i=1is then used
to iteratively refine X-GridAgent’s system prompt, starting
from an initial versionπ 0and progressing through successive
updatesπ 1→π 2→ ··· →π k→ ···. To support this
process, we construct two LLM-based agents: a judge agentJ,
which automatically evaluates X-GridAgent’s outputs against
ground-truth answers, and an edit agentE, which proposes
prompt modifications based on the evaluation.
At each iterationk, for a selected queryq k∈ Q, the
judge agentJevaluates X-GridAgent’s generated outputy k=
X-GridAgent(q k;πk)and produces a structured discrepancy
analysisd kby comparing it with the ground-truth reference
answerˆy k:
dk=J(q k, yk,ˆyk).(5)
For example, the discrepancy analysis may include feedback
such as “voltage limits are not checked”, “power flow tools
are not correctly invoked”, and “generation costs are missing”.
The edit agentEthen updates the system prompt based on the
discrepancy analysisd k:
πk+1=E(π k, dk).(6)
This evaluation-edit process can operate autonomously or
in a human-in-the-loop mode, where human experts provide
feedback by manually revising the discrepancy analysisd kor
directly correcting the updated promptπ k+1to make necessary
adjustments and to better align with domain-specific needs.
An illustrative example of prompt refinement for correctly
invoking therun contingency()function is shown in Figure 3.
The initial system promptπ 0is first calibrated by the edit agent
Eto yieldπ 1, and then revised toπ 2with expert feedback. The
iterative prompt refinement process terminates when the judge
agentJand human experts detect no significant discrepancies
betweenyandˆyand marks all queries inDas passed. The
system prompt at this point is taken as the final calibrated
promptπ∗, which is then deployed in X-GridAgent.

6
Fig. 3. Illustration of the iterative process of the LLM-driven prompt refinement with human feedback for correctly invoking the “run contingency()” function.
(In the third iteration, although the outcome is correct, it does not specify which contingency causes the voltage violation. This oversight is not detected by
the judge agent, requiring the human expert to point it out in the feedback.)

7
B. Schema-Adaptive Hybrid RAG
1) Background:RAG [38], [39] is a prominent technique
that couples documentation retrieval with text generation to
ground LLM responses in accurate and domain-specific data
sources. In a standard RAG pipeline, the documentation is pre-
processed into a set of passages or chunks, often with a target
lengthNand optional overlap. Given a user query, a retriever
scores and selects the top-kchunks by relevance. Based on
these selected chunks, the LLM generates an output grounded
on the provided evidence, which mitigates hallucinations and
improves performance on domain-specific tasks.
The retrieval mechanisms for RAG have two main types:
semantic(dense) retrieval [40] andlexical(sparse) retrieval
[41].Semantic retrievalmaps both queries and documents into
a shared high-dimensional dense vector space using a semantic
embedding strategy such as Sentence-BERT [42]. It measures
and indexes thesimilaritybetween a queryqand a text chunk
ci(i= 1,2,···), e.g., by cosine similarity:
sim(q, c i) =vq·vci
∥vq∥∥vci∥,(7)
wherev qandv cidenote the embedding vectors of the query
and the text chunk, respectively. Each text chunkc iis obtained
by splitting the corpus into segments of a fixed (or bounded)
length. The retriever selects the top-kchunks by similarity,
capturing semantic relationships and enabling the system to
identify relevant information even when the wording differs.
In contrast,lexical retrievalmatches documents to a query
primarily through exact (or near-exact) term overlap. It rep-
resents the query and each document as sparse vectors over
the vocabulary and ranks documents using token overlap with
term-weighting schemes. For example, BM25 [41] is a widely
used probabilistic ranking function that estimates a document’s
relevance to a query based on term frequency, inverse docu-
ment frequency, and document-length normalization. Given a
user queryq={t 1, t2, . . . , t m}, wheret jdenotes thej-th term
of the query, and a document chunkc i, the BM25 relevance
score is computed as:
BM25(q, c i) =
mX
j=1IDF(t j)·f(tj, ci)·(k 1+ 1)
f(tj, ci) +k 1·
1−b+b·N
avgdl,(8)
wheref(t j, ci)denotes the frequency of termt jinci,N
represents the chunk length, andavgdlis the average docu-
ment length in the corpus. The parametersk 1andbcontrol
the influence of term frequency and chunk length on the
relevance score, respectively. The inverse document frequency
term,IDF(t j), reflects how raret jis in the corpus, assigning
higher weight to terms that appear in fewer documents. Since
BM25 emphasizes lexical overlap, it is well suited to queries
involving precise terminology and structured identifiers.
2) Motivation:Power system analysis often requires re-
trieving relevant information from large-scale power grid
datasets (e.g., load and generation profiles) and from domain-
specific documents (e.g., planning and operation guides, indus-
try standards, and regulations). Conventional RAG approaches
are effective for document-centric retrieval, but they are less
reliable for retrieving information from large-scale power grid
Fig. 4. Comparison between conventional RAG methods and the proposed
schema-adaptive hybrid RAG algorithm.
datasets. This limitation arises because power system data are
highly structured and organized according to explicit schemas
(e.g., buses, lines, generators, loads, and transformers). The
meaning of a record often depends on its fields and its relation-
ships to other components. Representing such structured data
as unstructured text chunks can obscure these dependencies
and reduce retrieval accuracy, particularly at large scales.
Moreover, the information needed to answer a query is often
scattered across multiple components of the grid rather than
contained within a small set of contiguous passages. As
a result, retrieving only based on semantic similarity may
provide insufficient coverage for tasks that require integrat-
ing numerical information across various components, which
makes accurate and complete responses harder to obtain.
For example, consider the query on a large grid case:
Query 2.“Run AC power flow on the Texas 2k-bus grid1and
list the top 30 power lines with the highest loading ratios. ”
To answer this query, X-GridAgent first calls the function
solve ACpowerflow()in thePowerFlowserver to execute
the power flow calculation. It produces many result tables,
such asresult_load,result_gen,result_line,
result_bus, etc. A conventional RAG method may retrieve
theresult_linetable as it has the most relevant col-
umn [loading percent]. However,result_lineis a large
structured table with 3,992 rows (one per transmission line)
and 14 feature columns, including [ID], [from bus], [to bus],
[pfrom mw], [q from mvar], [loading percent], [pl mw],
and more, as shown in Figure 4. Conventional RAG methods
often flatten this table into unstructured text and thus fail to
retrieve the right lines with the highest loading percentages.
3) Our Algorithm:To address these limitations of conven-
tional RAG methods, we propose a novel algorithm called
schema-adaptive hybrid RAGfor effectively retrieving infor-
mation from large-scale power grid datasets. It aligns retrieval
with both the physical structure and the semantic schema of
1The Texas 2k-bus grid is a synthetic power grid dataset [43] developed by
the Texas A&M team, which is added to X-GridAgent for large-scale testing.

8
power grid data. Specifically, the proposed algorithm consists
of two steps: (1)schema-adaptive selectionand (2)hybrid
retrieval. In the first step, an LLM-based selection agentSis
constructed to generate a structured schemaX qbased on the
original datasetsD oriand tailored to the retrieval queryq:
Xq=S(D ori, q).(9)
Here, the schemaX qspecifies the tables and columns relevant
to fulfilling the retrieval query. For example, for the case
shown in Figure 4, the selection agentSoutputs the schema:
Xq:=n
“keyword”: [loading percent],“Table”:

“res_line”: [“ID”,“loading percent”]	o
,(10)
which selects the keywords and identifies the corresponding
tables and feature columns from the original dataset. Then, a
reduced dataset is constructed based on the selected tables and
columns. Moreover, given that power grid analysis typically
focuses on extreme values (e.g., highest line loading ratios or
lowest bus voltages), a sorting algorithm is applied to rank all
rows within each table by the magnitude of the corresponding
keyword column values. This results in a refined datasetD ref
used for the subsequent information retrieval.
In the second step, we construct a hybrid retriever that
combines semantic retrieval with lexical retrieval (BM25) to
leverage their complementary strengths. The hybrid relevance
score between the queryqand each chunkc iof the refined
datasetD refis computed as (11):
shyb(q, ci) =λ·BM25(q, c i) + (1−λ)·sim(q, c i),(11)
which is a linear combination of (7) and (8), and the weight
λ∈[0,1]controls the trade-off between semantic and lexical
retrieval. The hybrid retriever then selects the top-kchunks
with the highest values ofs hyb(q, ci)for queryq.
In this way, our proposedschema-adaptive hybrid RAGal-
gorithm reduces both the volume and complexity of power grid
data by extracting query-relevant information, and enhances
overall retrieval efficiency, relevance, and accuracy through a
hybrid approach that integrates semantic and lexical retrieval.
As shown in Figure 4, our algorithm successfully retrieves the
correct line information in response to Query 2.
IV. IMPLEMENTATION ANDEXPERIMENTS
In this section, we introduce the implementation of the X-
GridAgent system, outline the test experiments, and present
its performance across various user queries.
A. X-GridAgent Setup and Data Sources
For the X-GridAgent system, we employed the OpenAI
GPT-5 API as the LLM, serving as the core reasoning and
language-understanding engine for interpreting and responding
to user queries, while other LLM APIs can also be used.
To enable professional power system analysis, we integrated
the open-source Python-based software toolboxPandapower
[34], which supports power system modeling, analysis, and
optimization. A number of standard power network cases
(e.g., the IEEE 39-bus and 118-bus systems) [44] and large-
scale synthetic grid datasets (e.g., the Texas 2k grid) [43]
Fig. 5. The user interface of the X-GridAgent system. (A user enters a query
in the chat window and clicks “SEND” to submit it. X-GridAgent performs
reasoning and analysis, then displays the generated plan and execution results.
Clicking “CLEAR” clears the historical memory and starts a new chat. The
figure shows the system’s response to a user query requesting a visualization
of the Texas 2k-bus grid network, and a follow-up query to run a DC optimal
power flow (OPF) analysis is currently being typed.)
have been incorporated into X-GridAgent for testing purposes.
To enhance domain-specific expertise and ensure regulatory
compliance, we embedded many publicly available documents
(e.g., the ERCOT Planning Guide [45] and ERCOT Nodal
Operating Guides [46]) into a documentation database, from
which one can retrieve relevant information when responding
to grid-related queries. Moreover, we developed a user-friendly
interface using PySide6 [47], as shown in Figure 5, enabling
users to interact with the X-GridAgent system solely through
natural-language conversations. We showcase X-GridAgent in
action in a demonstration video available online [33].
B. Testing Experiments
We evaluate the X-GridAgent system using a diverse set of
power system analysis queries. Table I presents 11 represen-
tative test cases along with their outputs. Queries 1-8 are rela-
tively simple and are used to test the basic functionality of X-
GridAgent, while Queries 9-11 are more complex and require
multi-step reasoning and coordination. All outputs generated
by X-GridAgent have been verified to be correct by comparing
them against solutions manually obtained through conven-
tional methods. To assess the reliability of the system, we
executed each query 30 times, and the success rate was 100%,
indicating consistent performance. We set thetemperature
parameter of the LLM to zero to mitigate stochastic variability
in the model’s responses. While the exact wording of the
outputs varies slightly between runs, the substantive answers
remain accurate and consistent, demonstrating the reliability
of the X-GridAgent system. In addition, the scalability of X-
GridAgent is empirically verified and stems from its reliance
on professional power system analysis tools for all domain-

9
TABLE I
EXPERIMENTS AND TEST RESULTS OFX-GRIDAGENT ACROSS VARIOUS USER QUERIES
User Query Server Use X-GridAgent Output(some details are omitted for brevity)
Q1 (Grid dataset retrieval). “Load the IEEE-39 bus system.
What are the connected buses for line 11? What are the
voltage limits of bus 3?”Task 1:
Retrieval- Connected buses for line index 11: from bus=5, to bus=10
- V oltage limits for bus 3: vm min=0.94 pu, vm max=1.06 pu
Q2 (Documentation retrieval). “In ERCOT operations, what
is the maximum delay allowed for under-frequency relays
serving load resources?”Task 1:
RetrievalThe under-frequency relay must have a delay of no more than
20 cycles. *Citation: ERCOT Nodal Operating Guides, Section
2.3.1.2, paragraph (6)(a)···
Q3 (Topology). “For the IEEE 39-bus system, what is the
shortest path between bus 1 and bus 10?”Task 1:Topology- path busindices: [1, 2, 3, 4, 5, 10]
- path edges: [[1, 2], [2, 3], [3, 4], [4, 5], [5, 10]]
Q4 (Power flow analysis). “Run AC power flow on the
IEEE 39-bus system using the fast-decoupled method. Show
me the convergence result, the total active power loss, and
the active power flow value of line 11.”Task 1:
PowerFlow- Converged: ‘True’
- Total active power loss: 43.6423 MW
- Line index 11 active power flow: line 11 p from mw:
-322.6527 MW, line 11 p tomw: 323.3765 MW
Q5 (Optimal Power Flow). “Run DC OPF on the Texas 2k
grid and return the optimal generation profile and total cost.”Task 1:OPF- Total generation cost (OPF objective): 184290.887
- Generation profile (p mw): gen id 260: 1100; gen id 158:
1100; gen id 159: 1100; gen id 412: 600; gen id 428: 525;···
Q6 (Contingency). For the IEEE 118-bus system, perform
an N-1 contingency analysis to evaluate the impact of
disconnecting line 171. Summarize the limits violation.Task 1:
ContingencyPost-contingency limit violations:
- Bus voltage (min Vm) violation: bus 117 has Vm = 0.929 pu,
below Vmin = 0.94 pu (undervoltage of 0.011 pu).
- Thermal overloads: none reported.
Q7 (Visualization). “For the Texas 2k grid, show me the
network topology.”Task 1:Plot(See Figure 5 for the output and visualization of the Texas grid.)
Q8 (Short-circuit analysis). “For the IEEE 300-bus system,
suppose that there is a three-phase fault at bus 35, what are
the maximal short-circuit currents? Summarize the results.”Task 1:
ShortCircuitMaximal short-circuit (SC) results at Bus 35:
|Quantity|Value|Unit|
|Ikss (steady-state SC current)|2.5719|kA|
|Skss (SC apparent power)|1536.84|MV A| ···
Q9 (Complex multi-step query). “Can you run AC power
flow on the IEEE 118-bus grid to find the top 5 most
heavily loaded lines, then run contingency analysis on those
and summarize any operational limit violations?”Task 1:
PowerFlow→
Task 2:
Retrieval→
Task 3:
Contingency- Line 6 outage: 1 voltage violation (bus 37 at 0.939 pu<0.94
pu); no branch overloads.
- Line 7 outage: 1 voltage violation (bus 37 at 0.939 pu<0.94
pu); no branch overloads.
- Line 34 outage: no operational limit violations.
- Line 128 outage: no operational limit violations.
- Line 90 outage: no operational limit violations.
Q10 (Complex multi-step query). “For the IEEE 39-bus
system, consider three scenarios: Scenario 1 is the base
case, Scenario 2 takes line 11 out of service, and Scenario 3
doubles all loads. Run OPF for each scenario, and compare
and summarize the total generation cost and the generation
outputs of generators 1, 2, and 3 across the scenarios.”Task 1:OPF→
Task 2:Edit→
Task 3:OPF→
Task 4:Edit→
Task 5:OPF→
Task 6:
RetrievalConcise comparison:
- Scenario 2 vs Scenario 1: total generation cost increased from
41872.30 to 42115.34 (∆≈+243.04). Generator dispatch
shifted: Gen 1 decreased by∼100.18 MW, Gen 2 decreased by
∼12.55 MW, while Gen 3 stayed unchanged (∼508 MW).
- Scenario 3: the AC OPF did not converge under doubled loads,
so no reliable cost or dispatch values are available.
Q11 (Complex multi-step query). “From the ERCOT
Operating Guide, what are the normal operational voltage
limits for buses with a nominal voltage of 138 kV? Convert
these voltage limits to per-unit values based on the nominal
voltage, set those as Vmin/Vmax for all buses in the IEEE
118-bus system, then run an OPF and report total generation
cost and the resulting generation profile.”Task 1:
Retrieval→
Task 2:Edit→
Task 3:OPF- Min voltage magnitude: 131.1 kV (0.95 pu on 138 kV base)
- Max voltage magnitude: 144.9 kV (1.05 pu on 138 kV base)
- AC OPF (case118, with bus voltage limits set to 0.95-1.05 pu):
*Converged: true. *Total generation cost: 129746.0795961705
*Full generation dispatch profile (p mw, q mvar)
|gen id|p mw|q mvar|
|0|26.92149967760338|14.999962115216187|
|1|1.6543434098907168e-05|70.34323261531028| ···
specific computations, allowing it to scale up to large systems
(e.g., the Texas 2k-bus grid) and real-world power grids.
V. CONCLUSION
In this paper, we introduced X-GridAgent, a novel LLM-
powered agentic AI system designed to automate comprehen-
sive power grid analysis through natural language interaction.
X-GridAgent demonstrates strong capabilities in interpreting
user queries, dynamically generating task-specific workflows,
and interfacing with domain-specific tools and databases to
deliver trustworthy and interpretable results across a broad
range of power system analysis tasks. Its three-layer hierarchi-
cal architecture offers flexibility and extensibility, allowing theseamless integration of new tools, data sources, and analytical
capabilities. The current version, regarded as X-GridAgent 1.0,
provides a foundation for extensive future enhancements. For
example, as the present system focuses on steady-state power
grid analysis, future work will aim to support dynamic and
transient grid studies, as well as advanced decision-making
functionalities. In addition, the system architecture can be
further improved to enhance workflow stability and reliability.
REFERENCES
[1] A. Muhtadi, D. Pandit, N. Nguyen, and J. Mitra, “Distributed energy
resources based microgrid: Review of architecture, control, and relia-
bility,”IEEE Transactions on Industry Applications, vol. 57, no. 3, pp.
2223–2235, 2021.

10
[2] X. Chen, X. Wang, A. Colacelli, M. Lee, and L. Xie, “Electricity demand
and grid impacts of AI data centers: Challenges and prospects,”arXiv
preprint arXiv:2509.07218, 2025.
[3] X. Chen, H. Chao, W. Shi, and N. Li, “Towards carbon-free electricity:
A flow-based framework for power grid carbon accounting and decar-
bonization,”Energy Conversion and Economics, vol. 5, no. 6, pp. 396–
418, 2024.
[4] Q. Zhang and L. Xie, “PowerAgent: A road map toward agentic intel-
ligence in power systems: Foundation model, model context protocol,
and workflow,”IEEE Power and Energy Magazine, vol. 23, no. 5, pp.
93–101, 2025.
[5] X. Chen, G. Qu, Y . Tang, S. Low, and N. Li, “Reinforcement learning
for selective key applications in power systems: Recent advances and
future challenges,”IEEE Transactions on Smart Grid, vol. 13, no. 4, pp.
2935–2958, 2022.
[6] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y . Hou, Y . Min, B. Zhang,
J. Zhang, Z. Donget al., “A survey of large language models,”arXiv
preprint arXiv:2303.18223, vol. 1, no. 2, 2023.
[7] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[8] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut,
J. Schalkwyk, A. M. Dai, A. Hauth, K. Millicanet al., “Gemini: a family
of highly capable multimodal models,”arXiv preprint arXiv:2312.11805,
2023.
[9] F. Zeng, W. Gan, Y . Wang, N. Liu, and P. S. Yu, “Large language models
for robotics: A survey,”arXiv preprint arXiv:2311.07226, 2023.
[10] M. U. Hadi, R. Qureshi, A. Shah, M. Irfan, A. Zafar, M. B. Shaikh,
N. Akhtar, J. Wu, S. Mirjaliliet al., “Large language models: a
comprehensive survey of its applications, challenges, limitations, and
future prospects,”Authorea preprints, vol. 1, no. 3, pp. 1–26, 2023.
[11] M. Jia, Z. Cui, and G. Hug, “Enhancing LLMs for power system simu-
lations: A feedback-driven multi-agent framework,”IEEE Transactions
on Smart Grid, vol. 16, no. 6, pp. 5556–5572, 2025.
[12] K. Deng, Y . Zhou, H. Zeng, Z. Wang, and Q. Guo, “Power grid model
generation based on the tool-augmented large language model,”IEEE
Transactions on Power Systems, 2025.
[13] S. Jin and S. Abhyankar, “ChatGrid: Power grid visualization empow-
ered by a large language model,” in2024 IEEE Workshop on Energy
Data Visualization (EnergyVis), 2024, pp. 12–17.
[14] F. Amjad, T. Kor ˜otko, and A. Rosin, “Review of LLMs applications in
electrical power and energy systems,”IEEE Access, vol. 13, pp. 150 951–
150 969, 2025.
[15] S. Majumder, L. Dong, F. Doudi, Y . Cai, C. Tian, D. Kalathil, K. Ding,
A. A. Thatte, N. Li, and L. Xie, “Exploring the capabilities and
limitations of large language models in the electric energy sector,”Joule,
vol. 8, no. 6, pp. 1544–1549, 2024.
[16] Z. Durante, Q. Huang, N. Wake, R. Gong, J. S. Park, B. Sarkar, R. Taori,
Y . Noda, D. Terzopoulos, Y . Choiet al., “Agent AI: Surveying the
horizons of multimodal interaction,”arXiv preprint arXiv:2401.03568,
2024.
[17] D. B. Acharya, K. Kuppan, and B. Divya, “Agentic AI: Autonomous
intelligence for complex goals—a comprehensive survey,”IEEE Access,
vol. 13, pp. 18 912–18 936, 2025.
[18] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li,
“A survey on RAG meeting LLMs: Towards retrieval-augmented large
language models,” inProceedings of the 30th ACM SIGKDD conference
on knowledge discovery and data mining, 2024, pp. 6491–6501.
[19] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic retrieval-
augmented generation: A survey on agentic RAG,”arXiv preprint
arXiv:2501.09136, 2025.
[20] Anthropic, “Introducing the model context protocol,” [Online]. Avail-
able: https://www.anthropic.com/news/model-context-protocol, 2025,
accessed: 2025-10-20.
[21] X. Hou, Y . Zhao, S. Wang, and H. Wang, “Model context protocol
(MCP): Landscape, security threats, and future research directions,”
arXiv preprint arXiv:2503.23278, 2025.
[22] R. Zou, X. Zhou, Y . Cheng, W. Liu, X. Wang, J. Zhao, and X. Cai, “A
large language model-based agent for automated bidding strategy gen-
eration in electricity markets,” in2025 IEEE International Conference
on Power and Integrated Energy Systems (ICPIES), 2025, pp. 475–480.
[23] X. Ren, C. S. Lai, G. Taylor, and Z. Guo, “Can large language model
agents balance energy systems?”arXiv preprint arXiv:2502.10557,
2025.
[24] X. Yang, C. Lin, Y . Yang, Q. Wang, H. Liu, H. Hua, and W. Wu, “Large
language model powered automated modeling and optimization of active
distribution network dispatch problems,”IEEE Transactions on Smart
Grid, 2025.[25] H. Zhao, Y . Cheng, D. Xiang, X. Zhou, J. Zhao, X. Cai, and Z. Y .
Dong, “Large language model-based power dispatch agent: Framework,
application, and challenges,”Application, and Challenges, 2025.
[26] Y . Zhu, Y . Zhou, and W. Wei, “PowerCon: A collaborative framework
integrating LLM and power AI for interactive and regulation-compliant
day-ahead dispatch,”Authorea Preprints, 2025.
[27] Z. Li, H. Yang, Y . Liu, Y . Xiang, H. Gao, J. Liu, and J. Liu, “OptDis-
Pro: LLM-based multi-agent framework for flexibly adapting heuristic
optimal disflow,”IEEE Transactions on Smart Grid, 2025.
[28] F. Bernier, J. Cao, M. Cordy, and S. Ghamizi, “PowerGraph-LLM:
Novel power grid graph embedding and optimization with large language
models,”IEEE Transactions on Power Systems, 2025.
[29] Y . Zhang, A. M. Saber, A. Youssef, and D. Kundur, “Grid-Agent: An
llm-powered multi-agent system for power grid control,”arXiv preprint
arXiv:2508.05702, 2025.
[30] B. K. Saha, V . Aarthi, and O. Naidu, “DrAgent: An agentic approach
to fault analysis in power grids using large language models,” in2025
International Conference on Artificial Intelligence in Information and
Communication (ICAIIC). IEEE, 2025, pp. 0938–0945.
[31] H. Jin, K. Kim, and J. Kwon, “GridMind: LLMs-powered agents for
power system analysis and operations,” inProceedings of the SC’25
Workshops of the International Conference for High Performance Com-
puting, Networking, Storage and Analysis, 2025, pp. 560–568.
[32] E. O. Badmus, P. Sang, D. Stamoulis, and A. Pandey, “PowerChain: A
verifiable agentic AI system for automating distribution grid analyses,”
arXiv preprint arXiv:2508.17094, 2025.
[33] X. Chen and Y . Wen, “X-GridAgent demonstration video,” [Online].
Available: https://www.youtube.com/watch?v=JUqpwO6NncY, 2025,
accessed: 2025-12-17.
[34] L. Thurner, A. Scheidler, F. Sch ¨afer, J. Menke, J. Dollichon, F. Meier,
S. Meinecke, and M. Braun, “pandapower — an open-source python tool
for convenient modeling, analysis, and optimization of electric power
systems,”IEEE Transactions on Power Systems, vol. 33, no. 6, pp. 6510–
6521, Nov 2018.
[35]Short-circuit currents in three-phase a.c. systems – Part 0: Calculation
of currents, DIN; IEC Std. DIN EN 60 909-0 (IEC 60 909-0), 2016,
german version EN 60909-0:2016.
[36] X. Bo, Z. Zhang, Q. Dai, X. Feng, L. Wang, R. Li, X. Chen, and J.-
R. Wen, “Reflective multi-agent collaboration based on large language
models,”Advances in Neural Information Processing Systems, vol. 37,
pp. 138 595–138 631, 2024.
[37] G. Marvin, N. Hellen, D. Jjingo, and J. Nakatumba-Nabende, “Prompt
engineering in large language models,” inInternational conference on
data intelligence and cognitive informatics. Springer, 2023, pp. 387–
402.
[38] W. X. Zhao, J. Liu, R. Ren, and J.-R. Wen, “Dense text retrieval
based on pretrained language models: A survey,”ACM Transactions
on Information Systems, vol. 42, no. 4, pp. 1–60, 2024.
[39] X. Ma, Y . Gong, P. He, N. Duanet al., “Query rewriting in retrieval-
augmented large language models,” inThe 2023 Conference on Empir-
ical Methods in Natural Language Processing, 2023.
[40] Y . Yang, D. Cer, A. Ahmad, M. Guo, J. Law, N. Constant, G. H.
Abrego, S. Yuan, C. Tar, Y .-H. Sunget al., “Multilingual universal
sentence encoder for semantic retrieval,” inProceedings of the 58th
Annual Meeting of the Association for Computational Linguistics:
System Demonstrations, 2020, pp. 87–94.
[41] S. Robertson, H. Zaragozaet al., “The probabilistic relevance frame-
work: Bm25 and beyond,”Foundations and Trends® in Information
Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
[42] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using siamese BERT-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language Processing (EMNLP-
IJCNLP), 2019, pp. 3982–3992.
[43] “Electric grid test case repository,” https://electricgrids.engr.tamu.edu/,
2025, accessed: 2025-12-13; synthetic electric grid datasets from Texas
A&M University.
[44] P. Developers, “Pandapower network examples,” [Online]. Avail-
able: https://pandapower.readthedocs.io/en/latest/networks.html, 2025,
accessed: 2025-12-17.
[45] Electric Reliability Council of Texas (ERCOT), “Current planning
guides,” [Online]. Available: https://www.ercot.com/mktrules/guides/
planning/current, 2025, accessed: 2025-12-17.
[46] ——, “Current nodal operating guides,” [Online]. Available: https:
//www.ercot.com/mktrules/guides/noperating/current, 2025, accessed:
2025-12-17.
[47] The Qt Company, “Qt for Python (PySide6),” https://doc.qt.io/
qtforpython/, 2024, accessed: 2025-12.