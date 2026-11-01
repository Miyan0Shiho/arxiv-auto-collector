# RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG

**Authors**: Zekai Chen, Hao Zhu, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, Rong-Hua Li

**Published**: 2026-10-27 00:45:49

**PDF URL**: [https://arxiv.org/pdf/2610.36444v1](https://arxiv.org/pdf/2610.36444v1)

## Abstract
Web-based applications increasingly rely on understanding long multimodal documents, where answering questions often requires combining information across pages, tables, and figures. Agentic retrieval-augmented generation addresses these tasks through iterative retrieval and reasoning. However, Agentic execution is not always necessary, introducing additional computation and potentially incorrect answers. Our empirical analysis reveals that Agentic execution provides query-dependent benefits, while additional operations may improve, preserve, or degrade answer accuracy. These findings motivate controlling both Agent invocation and subsequent execution. To address these challenges, we propose RouteDRAG, a learning-based framework comprising four components. Unified Document Index organizes multimodal content into a graph-backed index and derives query representations from initial retrieval. Adaptive Path Routing preserves exact answering and predicts the relative benefit of Agent over shorter execution paths. Dynamic Agent Execution maintains an evolving workspace and refines Planner proposals to control tool use and termination. Sequential Policy Learning first trains operation control from observed outcomes and then trains path routing using the resulting Agent's performance. Across two benchmarks, RouteDRAG achieves an average accuracy improvement of 3.47 percentage points over shorter-path-only execution while assigning only 79.3% of queries to Agent. Notably, it also surpasses uniform Agent execution on both benchmarks while requiring fewer answering calls. Our code is now available at https://github.com/Ereshkigal1/RouteDRAG.

## Full Text


<!-- PDF content starts -->

RouteDRAG: Learning to Route and Control Agentic Execution
for Multimodal Document RAG
Zekai Chen*
Beijing Institute of Technology
Beijing, ChinaHao Zhu*
Beijing Institute of Technology
Beijing, ChinaYu Liu
Chinese Academy of Sciences
China
Xunkai Li
Beijing Institute of Technology
Beijing, ChinaZejian Kang
Zhejiang University
Hangzhou, China
Westlake University
Hangzhou, ChinaRui Qian
Fudan University
Shanghai, China
Quanjiang Li
National University of Defense
Technology
Changsha, ChinaRong-Hua Li
Beijing Institute of Technology
Beijing, China
Abstract
Web-based applications increasingly rely on understanding long
multimodal documents, where answering questions often requires
combining information across pages, tables, and figures. Agen-
tic retrieval-augmented generation addresses these tasks through
iterative retrieval and reasoning. However, Agentic execution is
not always necessary, introducing additional computation and po-
tentially incorrect answers. Our empirical analysis reveals that
Agentic execution provides query-dependent benefits, while addi-
tional operations may improve, preserve, or degrade answer accu-
racy. These findings motivate controlling both Agent invocation
and subsequent execution. To address these challenges, we pro-
poseRouteDRAG, a learning-based framework comprising four
components.Unified Document Indexorganizes multimodal
content into a graph-backed index and derives query represen-
tations from initial retrieval.Adaptive Path Routingpreserves
exact answering and predicts the relative benefit of Agent over
shorter execution paths.Dynamic Agent Executionmaintains
an evolving workspace and refines Planner proposals to control
tool use and termination.Sequential Policy Learningfirst trains
operation control from observed outcomes and then trains path
routing using the resulting Agent’s performance. Across two bench-
marks, RouteDRAG achieves an average accuracy improvement
of 3.47 percentage points over shorter-path-only execution while
assigning only 79.3% of queries to Agent. Notably, it also sur-
passes uniform Agent execution on both benchmarks while re-
quiring fewer answering calls. Our code is now available on https:
//github.com/Ereshkigal1/RouteDRAG.
CCS Concepts
•Information systems→Information retrieval.
WWW ’27, Dublin, Ireland
2027.Keywords
multimodal document question answering, retrieval-augmented
generation, selective execution, Agentic evidence acquisition
ACM Reference Format:
Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang
Li, and Rong-Hua Li. 2027. RouteDRAG: Learning to Route and Control
Agentic Execution for Multimodal Document RAG. InProceedings of The
Web Conference 2027 (WWW ’27).ACM, New York, NY, USA, 13 pages.
1 Introduction
Web-based applications, such as financial analysis platforms [ 5,30],
government information portals, academic search engines [ 7], and
online technical support [ 3,11], increasingly rely on understand-
ing long and visually rich documents. Answering questions often
requires locating and combining information distributed across
pages, tables, and figures. For example, explaining a company’s
financial performance may require examining numerical results in
a table and their explanations on other pages. A single retrieval step
may locate only part of the required information, leaving the ques-
tion unanswered. Agentic retrieval-augmented generation (Agentic
RAG) addresses this challenge through iterative retrieval, visual un-
derstanding, and multi-step reasoning, allowing information from
different document regions to be located and combined.
Recent studies have made Agentic RAG more adaptive through
multimodal retrieval, graph-based exploration and iterative reason-
ing [15,18,22,25,32]. These methods allow Agents to dynamically
select document content and adjust retrieval procedures. However,
adapting how information is retrieved does not fully address how
much Agentic execution is needed. As illustrated in Fig. 1, some
questions require multiple retrieval steps to locate and combine
information across different pages, while others can be answered
directly from the initial retrieval results without invoking an Agent.
Unnecessary Agent invocation increases computational cost and
may introduce irrelevant or conflicting information, leading to in-
correct answers. Yet deciding whether to invoke an Agent is only
the first step. As the Agent gathers more information, it must fur-
ther decide whether to retrieve additional content, inspect visual
arXiv:2610.36444v1  [cs.AI]  27 Oct 2026

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
regions, or terminate execution. This raises two fundamental ques-
tions:when should Agentic execution be invoked, and how should its
subsequent operations be controlled?
Our empirical analysis reveals three observations about Agen-
tic execution.O1: Limited Overall Gains.Agent improves
accuracy by 1.00 and 3.12 percentage points on DocBench and
MMLongBench-Doc, respectively, but uses1 .65×and1.81×as
many answering tokens as Standard.O2: Query-Dependent
Benefits.Agent alone answers 3.09% and 6.69% of questions
correctly, while both Agent and Standard succeed on 69.24%
and 44.18%. Conversely, Standard alone succeeds on 2.09% and
3.57%, indicating that Agent execution is not universally beneficial.
O3: Non-Monotonic Operation Gains.Among 917 matched
operation-level comparisons, 84.51% show unchanged correctness,
while additional operations can improve or degrade answers.
These observations motivate selective Agent invocation based
on its expected benefit and adaptive operation control based on
information collected during execution.
To address these challenges, we proposeRouteDRAG, a
learning-based framework for selective Agentic document RAG.
First,Unified Document Indexorganizes multimodal content
into a graph-backed index and derives query representations from
initial retrieval. Next,Adaptive Path Routingpredicts the benefit
of invoking Agent over a shorter path while preserving exact
answering. When Agent is invoked,Dynamic Agent Execution
maintains an evolving workspace and refines Planner proposals to
control subsequent operations and termination. Finally,Sequential
Policy Learningfirst trains operation control from observed
outcomes and then trains path routing using the resulting Agent’s
performance. Together, these components adapt Agent invocation
and execution to individual queries.
Our Contributions. (1) New Perspective. We identify limited
overall gains, query-dependent benefits, and non-monotonic op-
eration gains in Agentic document RAG, motivating execution
decisions at both query and operation levels. (2) New Framework.
We proposeRouteDRAG, integrating Unified Document Index,
Adaptive Path Routing, Dynamic Agent Execution, and Sequential
Policy Learning to control Agent invocation and subsequent opera-
tions. (3) SOTA Performance. We evaluate RouteDRAG against nine
RAG methods on DocBench and MMLongBench-Doc, achieving
75.86% and 53.80% accuracy, respectively, while reducing answering
calls by 16.4% and 14.9% compared with uniform Agent execution.
2 Related Work
2.1 Multimodal and Agentic Document RAG
Document RAG has evolved from text-based retrieval toward
richer multimodal and structural representations. LightRAG [ 13]
organizes entities and relations into a graph, while RAG-
Anything [ 12] extends graph-based retrieval to multimodal
documents. GraphReader [ 17] further employs graph-guided Agent
exploration for long-text understanding.
G2-Reader [ 8] maintains evolving content and planning graphs
for multimodal document question answering.
VisRAG [ 29] and M3DocRAG [ 6] further explore visual retrieval
for document understanding.AgenticOCR [ 26] enables query-driven parsing of relevant visual
regions to reduce redundant document processing.
ReAct [ 28] established a reasoning-and-acting paradigm that
interleaves language-model reasoning with external actions.
Recent Agentic approaches support more flexible information
acquisition. ViDoRAG [ 25] employs iterative exploration and rea-
soning, LAD-RAG [ 22] combines neural and structural document
retrieval, and MAGE-RAG [ 32] dynamically explores and prunes
graph-based evidence. TAP-RAG [15] further adapts graph expan-
sion and visual inspection to query requirements. These methods
improve the acquisition and organization of multimodal informa-
tion. RouteDRAG instead focuses on whether such Agentic ex-
ecution is necessary for a given query and how its subsequent
operations should be controlled.
2.2 Adaptive Retrieval and Execution Selection
Adaptive RAG methods adjust retrieval and execution according to
query requirements. Self-RAG [ 1] learns when retrieval is needed,
while Adaptive-RAG [ 14] selects among no-retrieval, single-step,
and iterative strategies based on query complexity.
Active Retrieval Augmented Generation [ 16] dynamically re-
trieves additional evidence during generation.
MBA-RAG [ 24] further uses bandit-based selection to balance
retrieval quality and efficiency. More recent methods extend adapta-
tion to multi-step execution. MAO-ARAG [ 4] dynamically organizes
retrieval agents into query-specific workflows, while SPARKLE [ 9]
learns a separate policy for controlling retrieval operations. These
studies demonstrate the value of both query-level adaptation and
step-wise control. RouteDRAG connects the two decisions by es-
timating Agent’s benefit relative to a designated shorter path and
refining subsequent operations based on the evolving workspace.
Its sequential learning process further aligns path routing with the
behavior of the resulting Agent.
3 Problem Definition
Given a multimodal document 𝐷and a query 𝑞, letK𝐷denote the
unified document index. We consider four execution paths:
C={Exact,Standard,Direct VLM,Agent},(1)
where the first three provide shorter answering procedures, while
Agent supports iterative retrieval and reasoning. Unlike execution-
path selection alone, selective Agentic execution involves two re-
lated decisions. At the query level, a routing policy 𝜋(𝑞,K𝐷)deter-
mines whether Agent is necessary or a shorter path is sufficient.
Once Agent is invoked, its execution policy selects an operation
𝑎𝑡from the legal action set Γ𝑡based on the evolving workspace
𝑊𝑡, including whether to continue acquiring information or termi-
nate. Our objective is to maintain answer accuracy while avoiding
unnecessary Agent invocation and subsequent operations.
4 Empirical Study
To understand when Agentic execution is necessary, we investigate
its benefits from three perspectives. We use a standalone reference
Agent for these analyses, distinct from the final RouteDRAG Actor
evaluated in the main experiments. We first examine whether uni-
form Agent execution justifies its additional cost, then investigate

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
(a)Short path is sufficient
 (b)Agent is necessary
 (c)More execution can hurt
YearRevenue
2022
2023120
180Multimodal Document
What is the 
revenue in 2023?
Short path (no agent)
Retrieve
 Answer
Initial results 
are enough
Agent path (unnecessary)
Search
Read &
Reason
AnswerSection 3 Figure 2
Multimodal Document
Which section 
and figure 
together explain 
the decline?
Short path (no agent)
Retrieve
 Missing clue
Agent path (needed)
Search
Read 
Documen
t
Link
CluesAnsw
erMultimodal Document
Which language 
appears in the 
figure?
Figure already 
checked
Current agent stateFinish (stop)
Finish
Correct
More search (continue)
Search
Distracted by 
irrelevant content
Why Selective Agentic Execution?
56.8% Both Correct4.9% Agent-only Correct14/331Search Degrades
Figure 1: Why selective Agentic execution? (a) Initial retrieval can suffice, making Agent execution redundant. (b) Others
require linking information through Agent execution. (c) Further retrieval can distract an Agent with sufficient information.
Figure 2: Empirical analysis of Agentic execution. (a) Accuracy and answering-token usage across document categories for
Standard and Agent. (b) Paired query outcomes, illustrating the query-dependent benefits of Agentic execution. (c) Answer
correctness changes after an additional operation compared with immediate termination.
how its benefits vary across queries, and finally assess whether
additional operations consistently improve answers.
O1: Limited Overall Gains.Does Agentic execution justify its
additional computation? Fig. 2(a) compares Standard and Agent
across five DocBench domains and seven MMLongBench-Doc doc-
ument types. Agent improves accuracy by 1.00 and 3.12 percentage
points on DocBench and MMLongBench-Doc, respectively, while
consuming1 .65×and1.81×the answering tokens. These results re-
veal a mismatch between overall accuracy gains and computational
cost: uniformly invoking the Agent incurs substantial overhead for
modest improvements. However, aggregate results cannot identifywhich queries justify this additional computation, motivating a
closer examination of query-level benefits.
O2: Query-Dependent Benefits.Which queries actually ben-
efit from Agent execution? Fig. 2(b) compares the two paths on
the same questions. Correctness remains unchanged for 94.83%
of DocBench queries and 89.73% of MMLongBench-Doc queries.
Meanwhile, Agent recovers answers missed by Standard on 3.09%
and 6.69% of queries, but also turns correct answers into incorrect
ones on 2.09% and 3.57%, respectively. Thus, Agent execution is
neither universally necessary nor uniformly beneficial. Its value
depends on the query, motivatingAdaptive Path Routingto estimate

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
the expected benefit of Agent execution over the designated short
path.
O3: Non-Monotonic Operation Gains.Even when Agent exe-
cution is warranted, does acquiring more information necessarily
improve the answer? We examine 917 matched comparisons be-
tween immediate FINISH and executing one additional operation
before FINISH. As shown in Fig. 2(c), 84.51% of comparisons leave
correctness unchanged, while the remaining operations can either
improve or degrade answers. In particular, VISUAL operations de-
grade correctness more often than they improve it, suggesting that
additional information may introduce noise rather than useful ev-
idence. This motivatesDynamic Agent Execution, which adapts
operation selection to the evolving workspace instead of assuming
that continued execution is always beneficial.
5 Methodology
5.1 Overview
Fig. 3 illustrates the overall framework. RouteDRAG addresses se-
lective Agentic execution through two related decisions: when to
invoke Agent and how to control its subsequent operations. These
decisions are learned from observed execution outcomes at the
query and operation levels. The framework comprises four compo-
nents: (1) Unified Document Index, (2) Adaptive Path Routing, (3)
Dynamic Agent Execution, and (4) Sequential Policy Learning.
5.2 Unified Document Index
Motivation.Effective path routing requires considering not only
the query but also the document content retrieved before execution.
We therefore construct a shared document index and learn a query-
conditioned representation that can be reused during execution.
Graph-Backed Document Organization.Following RAG-
Anything and LightRAG [ 12,13], we organize each multimodal
document into the shared index K𝐷defined in the problem formu-
lation. The index connects text chunks, entities, relations, visual
elements, and structured metadata with their original document
locations. It supports four execution paths: Exact, Standard, Direct
VLM, and Agent. The document index and pretrained retrieval
encoder remain frozen during training.
Query-Conditioned Representation.Given a query 𝑞, we
retrieve the most relevant text chunks and graph entities, retaining
up to eight chunks and four entities. A frozen semantic encoder
produces their embeddings, denoted bye 𝑞,E𝑐, andE𝑔. A shared
trainable projection 𝑃𝜓maps them into a 64-dimensional space.
We then obtain query-conditioned summaries through masked
attention:
z𝑞=𝑃𝜓(e𝑞),z𝑐=Attn𝜓(z𝑞,𝑃𝜓(E𝑐)),z𝑔=Attn𝜓(z𝑞,𝑃𝜓(E𝑔)).
(2)
Here, the attention operation excludes unavailable entries through
a validity mask. To adapt the contribution of each source to the
current query, we compute
𝜶=softmax 𝐺𝜓([z𝑞;z𝑐;z𝑔]),h0=𝐹𝜓 [𝛼𝑞z𝑞;𝛼𝑐z𝑐;𝛼𝑔z𝑔;𝑈𝜓(u0)],
(3)
whereu 0contains query characteristics, retrieved-content statistics,
and similarity scores. 𝐺𝜓and𝐹𝜓denote lightweight gating andfusion networks. The resulting representationh 0summarizes the
query and initial retrieval results before Agent invocation.
5.3 Adaptive Path Routing
Motivation.A complex question does not necessarily require Agen-
tic execution, while a seemingly simple question may need multiple
retrieval operations. We therefore learn whether Agent improves
upon an available non-Agent path, rather than predicting execution
requirements solely from query types.
Relative Path Prediction.RouteDRAG first applies a verified
Exact gate. Otherwise, a fixed policy selects the available shorter
path𝐶(𝑞)∈{Standard,VLM} . Instead of independently estimating
every path’s correctness, we predict the relative advantage of the
Agent:
bΔ𝜃(𝑞)=w⊤
ℎh0+w⊤
𝑢eu0+𝑏,(4)
whereeu0contains normalized query and retrieval statistics. The
first term models interactions between the query and retrieved
document content, while the second preserves direct access to
lightweight descriptive features.
Execution-Path Selection.The resulting routing policy is
𝜋𝜃(𝑞)= 
Exact, 𝑔 exact(𝑞,K𝐷)=1,
Agent, bΔ𝜃(𝑞)>𝜏,
𝐶(𝑞),otherwise,(5)
where𝑔exactverifies complete structured metadata and 𝜏is deter-
mined on the development set. This formulation turns Agent in-
vocation into a relative execution decision: the Agent is selected
when its predicted improvement over the designated shorter path
exceeds the routing threshold.
5.4 Dynamic Agent Execution
Motivation.Agent operations should adapt to the information
accumulated during execution. We integrate learned action refine-
ment with the LLM Planner, using the evolving execution state to
guide its proposals.
State-Conditioned Operation Modeling.Agent starts with
native hybrid retrieval and maintains a workspace 𝑊𝑡containing
retrieved content, source references, and operation history. The
shared projection network produces its state representationh 𝑡.
For each candidate operation 𝑎with embeddinge 𝑎, we compute
action-conditioned summaries of text chunks and graph entities:
z𝑚
𝑡,𝑎=Attn𝜙 z𝑞+e𝑎,𝑃𝜓(E𝑚
𝑡), 𝑚∈{𝑐,𝑔}.(6)
Here,E𝑐
𝑡andE𝑔
𝑡denote current chunk and entity embeddings. We
combine these summaries with the execution state and operation
history:
𝑧𝜙(𝑊𝑡,𝑎)=𝐻𝜙
[h𝑡;z𝑐
𝑡,𝑎;z𝑔
𝑡,𝑎;e𝑎;ℓ𝑡]
,(7)
whereℓ𝑡encodes previous operations. The relative operation score
is defined against immediate termination:
b𝑑𝑡(𝑎)=𝑧𝜙(𝑊𝑡,𝑎)−𝑧𝜙(𝑊𝑡,FINISH).(8)
This provides a common reference for comparing continued execu-
tion with termination.
Constrained Action Refinement.At each step, native exe-
cution rules determine the legal action set Γ𝑡, including retrieval,
graph expansion, structured lookup, visual inspection, and FINISH

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
RouteDRAG: Overall Framework
1 Unified Document Index (Multimodal Query Retrieval and Fusion)
YearRevenue
2022
2023120
180Multimodal Document
Section 3 Figure 2
𝑫
Query𝒒
Which section and 
figure together 
explain the decline?
Initialization Parsing & Structuring
 Graph-based Index Retrieval & Fusion
Text Chunks 𝑻
Visual Regions 𝑽
Metadata
/ Layout 𝑴Entity Graph
1
3
4
2
2
1
3
 2
1
1
 Entities
(Nodes)𝜺
3
2
1
Relations
(Edges)𝑹
Construct
Encode
...
Text Embeddings...
Query
Embeddings
Link
Generate
Visual
Description
𝑮𝑫
Source
Reference𝝆𝑫
Vector 
Index𝑰𝑫
Query𝒒Encode
Similarity Search
Top-k
Chunks
3
2
1
Entities
...
Entities
Embedding...
Chunks
Embedding
...
Query
Embedding
Fused
Embedding𝒉𝟎2Adaptive Path Routing (Choose Shorter Path or Invoke Agent)
Exact
Checking
𝒈𝒆=𝟏?
Yes
NoShorter Path 𝑪(𝒒)
Exact VLM
 Standard
𝒖𝟎
Fused
Embedding𝒉𝟎
Query 
Statistics𝒖𝟎
Route 
Scorer
𝜟>𝝉?
(q)Δˆ
No
Route Decision for Query 𝒒
Threshold
𝝉YesAgent Path 𝑨𝝓(𝒒)
Search Graph
Finish
Structured Visual
Dynamic Agent Execution (Tool -Augmented Reasoning over Document Index) 3
Sequential Policy Learning (Train Dynamic Agent and Adaptive Path Routing)4Document Index 𝑲𝑫
3
2
1
Text Visual
Entities Relations
Metadata
Workspace 𝑾𝒕
Retrieve
Access
Query
Retrieved 
Content
Source 
Reference
History
Action
Search Graph
Finish
Structured Visual
Candidate Actions 𝜞𝒕
Round T: Planner -Controller -Execution Loop
Planner
(propose)
Proposed
Action𝒂𝒕𝑷
 Controller
(score&
refine)
Action
Advantage
(predicted)
True
Action
Execute(
tool use)
Update
Workspace
𝑾𝒕+𝟏
𝒂𝒕
Answer
𝒕=𝒕+𝟏
 Synthesis
Query
Retrieved 
Content
Source 
Reference
History
ActionCurrent Workspace
𝑾𝒕
Immediate Finish
𝑷𝒙={𝐹𝑖𝑛𝑖𝑠ℎ}
One more action
𝒂𝝐𝜞𝒕\{𝐹𝑖𝑛𝑖𝑠ℎ}Predict
Action Value
𝒅𝒕(𝒂)𝒀𝒔𝒕𝒐𝒑
𝒀𝒄𝒐𝒏𝒕𝒊𝒏𝒖𝒆
Controller Loss
𝑳𝑨
Query𝒒
Operation -level Learning Path-level Learning
Agent Path
Shorter Path
𝑨𝝓(𝒒)
𝑪(𝒒)
𝒓𝑨
𝒓𝑪
Preference
𝒓𝑨−𝒓𝑪
Route Scorer
𝒓𝑨−𝒓𝑪
Router Loss
𝑳𝑹
Train
RouterTrain
Controller
Figure 3: Overview of RouteDRAG. A shared document index supports learned path routing and Agent execution control.
Actor-aligned policy learning uses paired execution outcomes to train routing and action decisions in stages.
when permitted. Given the Planner’s proposal 𝑎𝑃
𝑡, the controller
identifies
𝑎∗
𝑡=arg max
𝑎∈Γ𝑡b𝑑𝑡(𝑎).(9)
The selected action is
𝑎𝑡=(
𝑎∗
𝑡,b𝑑𝑡(𝑎∗
𝑡)−b𝑑𝑡(𝑎𝑃
𝑡)>𝑚,
𝑎𝑃
𝑡,otherwise,(10)
where𝑚is a fixed margin. This refinement applies to valid Planner
proposals, while mandatory actions follow native rules. If the con-
troller changes the operation, the Planner generates its arguments,
preserving the original tool execution procedure.
State Transition.The selected operation returns 𝑜𝑡, updating
the workspace as 𝑊𝑡+1=Commit(𝑊 𝑡,𝑜𝑡). The update incorporates
new content, preserves source references, and removes duplicates.
Subsequent decisions use the updated workspace until FINISH or
the execution limit is reached, followed by answer synthesis and
validation.
5.5 Sequential Policy Learning
Motivation.Agent invocation depends on its actual performance,
while operation utility varies with the execution state. We thereforelearn both decisions from paired outcomes, training the router after
updating the Agent controller.
Operation-Level Learning.We compare immediate termina-
tion with one additional operation at observed Agent states. For
state𝑊𝑖,𝑡and operation𝑎, the target is
𝑑𝑖,𝑡,𝑎=𝑌 Stop(T𝑎(𝑊𝑖,𝑡))−𝑌 Stop(𝑊𝑖,𝑡),(11)
where𝑌(·)∈{ 0,1}denotes correctness and Stop applies a fixed
answer-generation procedure. Only evaluated state–action pairs
are used. The controller combines regression, action ranking, and
Planner imitation:
L𝐴=L reg+𝛼L𝐴
rank+𝛽L BC,(12)
whereLregfitsb𝑑𝑡(𝑎)to the observed gain, and LBCpreserves Plan-
ner preferences. The ranking loss is
L𝐴
rank=1
|P𝐴|∑︁
(𝑡,𝑎,𝑏)∈P𝐴softplus
−h
b𝑑𝑡(𝑎)−b𝑑𝑡(𝑏)i
,(13)
whereP𝐴contains evaluated same-state action pairs with 𝑑𝑡(𝑎)>
𝑑𝑡(𝑏).
Path-Level Learning.We execute the updated Agent 𝐴𝜙and
compare its correctness with the fixed shorter path:
Δ𝑖=𝑌(𝐴𝜙(𝑞𝑖))−𝑌(𝐶(𝑞 𝑖))∈{−1,0,1}.(14)

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
The router predicts this relative gain using regression and cross-
query ranking:
L𝑅=E𝑖(ˆΔ𝑖−Δ𝑖)2+𝜇E(𝑖,𝑗)∈P𝑅ℓ(ˆΔ𝑗−ˆΔ𝑖).(15)
where ˆΔ𝑖=bΔ𝜃(𝑞𝑖),ℓ(𝑥)=softplus(𝑥) , andP𝑅contains query
pairs satisfyingΔ 𝑖>Δ𝑗.
Sequential Training.RouteDRAG first trains the shared repre-
sentation and Agent controller from observed operation outcomes.
It then freezes both and trains the routing head using the updated
Agent’s execution results. This ensures that path routing reflects
the Agent’s actual behavior. All parameters remain fixed during
evaluation.
6 Experiments
We conduct extensive experiments on two multimodal document
benchmarks to answer the following research questions.Q1: Over-
all Performance.How does RouteDRAG compare with existing
document RAG methods in answering accuracy?Q2: Ablation
Study.How do adaptive path routing and dynamic Agent execu-
tion contribute to performance?Q3: Robustness Analysis.How
robust is RouteDRAG to noisy inputs, hyperparameter variations,
and different model backbones?Q4: Efficiency Analysis.Can
RouteDRAG achieve a favorable accuracy–computation trade-off
compared with existing RAG methods?
6.1 Experimental Setup
Datasets, Baselines, and Backbones.We evaluate RouteDRAG
on two multimodal document benchmarks, DocBench [ 31] and
MMLongBench-Doc [ 19]. To assess its effectiveness across differ-
ent retrieval and execution paradigms, we compare against four
categories of baselines: (1) direct inference using a budgeted Qwen3-
VL model [ 2]; (2) text-based RAG, including BM25-based sparse
retrieval [ 21] and BGE-M3-based dense retrieval [ 20]; (3) multi-
modal RAG, including M3DocRAG [ 6], and EVisRAG [ 23]; and
(4) graph-based and Agentic RAG, including RAG-Anything [ 12],
MoLoRAG [ 27], ViDoRAG [ 25], and TAP-RAG [ 15]. For a fair com-
parison, all baselines and RouteDRAG use the same Qwen3-VL-
8B-Instruct answering model, with differences arising from their
retrieval and execution strategies rather than backbone capabili-
ties. We additionally compare Short Path Only, Rule Routing, and
Uniform Agent as controlled execution strategies. RouteDRAG em-
ploys a unified document index, frozen BGE-M3 [ 20] for semantic
representation, and Qwen3-VL-8B-Instruct [ 2] for multimodal in-
ference.
Evaluation Metrics and Protocol.We report answering ac-
curacy (Acc.), Agent usage (Agent%), executed actions per query
(Actions/Q), answering calls and tokens per query, inference la-
tency, and peak GPU memory. Accuracy is assessed using a fixed
Qwen-based judge, with failed responses counted as incorrect. For
the main performance comparison, we report mean accuracy and
standard deviation over five independent runs.
6.2 Overall Performance (Q1)
To answer Q1, we compare RouteDRAG with representative docu-
ment RAG methods across direct inference, text-based RAG, multi-
modal RAG, and graph-based and Agentic RAG. All experiments
−2.5−2.0−1.5−1.0−0.50.00.5
Δ Accuracy (pp)
-0.005 0 0.005
Threshold offset τ10
100
1000Regularization α 76.22 76.04 75.68
75.68 75.86 75.68
76.04 75.59 75.68(a) DocBench
-0.005 0 0.005
Threshold offset τ10
100
100053.43 53.53 53.25
53.25 53.80 52.98
53.53 53.34 51.33(b) MMLongBench-DocFigure 4: Sensitivity to regularization strength and routing
threshold on (a) DocBench and (b) MMLongBench-Doc. Cells
report accuracy (%); colors show percentage-point changes
from𝛼=100and zero threshold offset.
are conducted with five independent runs, and we report mean
accuracy with standard deviations. As shown in Tables 2 and 1,
RouteDRAG achieves 75.86% accuracy on DocBench and 53.80% on
MMLongBench-Doc, outperforming the competing RAG baselines
on both benchmarks. Notably, RouteDRAG outperforms uniform
Agent execution by 1.18 and 1.28 percentage points on DocBench
and MMLongBench-Doc, respectively, while reducing answering
calls by 16.4% and 14.9%. These results suggest that selective Agent
invocation can improve the accuracy–efficiency trade-off under
our evaluation settings.The gains also extend across evaluation
categories: RouteDRAG achieves the best results in two of five
document domains and two of four question types on DocBench,
and leads across all five support-source categories and two of three
support-requirement categories on MMLongBench-Doc. These re-
sults demonstrate the effectiveness of RouteDRAG across diverse
document structures and question requirements, motivating fur-
ther analysis of its component contributions and computational
efficiency.
6.3 Ablation Study (Q2)
To answer Q2, we investigate the contributions of adaptive path
routing and dynamic Agent execution. As shown in Table 3,
removing Agent execution reduces accuracy to 73.41%/49.31%,
while uniform Agent execution achieves 74.68%/52.52%. In contrast,
RouteDRAG achieves 75.86%/53.80% while invoking Agent for only
78.49%/80.02% of queries. RouteDRAG-Efficient offers a lower-cost
operating point, achieving 74.05%/50.87% accuracy with only
11.52%/16.41% Agent use, exceeding rule-based routing on both
datasets. These results show how selective execution balances the
benefits of Agentic reasoning against unnecessary or potentially
harmful operations.
Comparison Settings.w/o Adaptive Path Routingreplaces
learned path selection with rule-based capability assignment.w/o
Dynamic Agent Executionassigns every query to a non-adaptive
Agent execution configuration.w/o Agent Routinganswers through
the short-path portfolio, using Exact for supported structured
requests and Standard or Direct VLM for the remaining queries.All
Agent Routingsends every query to the mature Agent.RouteDRAG
combines selective path allocation with dynamic Agent execution.
RouteDRAG-Efficientshares the same mechanism asRouteDRAG
and yields the efficiency-oriented operating point in Table A1 solely
through a different threshold configuration. The two all-Agent
configurations compare execution policies at full Agent coverage;

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
Table 1: Answer accuracy (%) on MMLongBench-Doc. Best and second-best results are bold and underlined, respectively.
Support source Support requirement
Type Method TXT LAY CHA TAB FIG SIN MUL UNA Overall
Direct Direct MLLM 30.761.88 36.302.09 28.581.72 20.522.16 28.461.94 36.301.66 21.362.31 52.461.83 35.00
Text RAGBM25 34.861.19 19.201.37 17.331.12 31.881.48 14.221.06 34.000.94 16.981.56 76.641.21 37.98
BGE-M3 33.721.32 23.171.18 13.961.09 35.821.41 16.151.02 33.150.96 19.991.53 73.361.24 37.86
MultimodalM3DocRAG [6] 43.471.79 31.811.63 39.231.51 33.451.91 37.331.43 53.081.32 25.381.72 25.001.98 37.58
EVisRAG [23] 38.291.22 33.211.34 38.171.11 39.841.46 38.900.98 51.550.92 25.031.28 58.201.57 44.27
Graph/AgenticRAG-Anything [12] 40.002.08 29.301.91 29.211.76 43.452.22 33.731.59 47.591.53 26.931.82 60.252.31 43.61
MoLoRAG [27] 38.882.39 36.822.17 35.861.94 37.552.48 32.802.08 51.281.85 22.312.23 52.462.61 41.96
ViDoRAG [25] 38.211.93 27.411.78 27.451.64 34.662.02 33.371.48 48.691.55 18.741.86 62.702.19 41.85
TAP-RAG [15] 43.251.69 28.361.55 34.911.34 36.791.82 36.221.46 52.201.27 24.841.61 60.451.98 45.01
RouteDRAG 51.680.97 46.221.08 41.570.82 55.051.19 42.660.72 58.750.61 37.990.91 67.621.31 53.80
Table 2: Answer accuracy (%) on DocBench. Bold and underlined values indicate the best and second-best results, respectively.
Document domain Question type
Type Method Acad. Finance Gov. Law News TXT MM META UNA Overall
Direct Direct MLLM 30.362.14 39.582.38 68.921.92 65.972.26 33.142.47 52.431.85 22.402.55 49.222.08 63.712.31 44.55
Text RAGBM25 61.061.45 57.641.62 74.321.27 73.821.73 87.211.18 89.811.06 58.441.84 39.531.91 80.651.39 68.24
BGE-M3 55.781.58 54.511.42 67.571.31 71.201.64 86.631.22 82.771.15 52.921.78 42.251.96 79.031.37 64.52
MultimodalM3DocRAG [6] 61.062.03 65.281.87 66.891.74 71.202.18 81.401.65 85.681.52 68.511.96 53.102.26 57.902.07 67.88
EVisRAG [23] 67.301.43 65.751.56 70.651.28 72.921.72 82.471.16 82.441.08 72.651.47 45.841.85 78.451.31 70.69
Graph/AgenticRAG-Anything [12] 69.642.32 62.502.14 67.571.97 70.682.48 88.371.81 86.891.73 72.401.92 39.532.57 76.612.21 70.60
MoLoRAG [27] 46.862.66 65.282.43 74.322.19 73.302.75 74.882.34 69.172.08 54.872.49 44.962.89 56.452.61 64.32
ViDoRAG [25] 64.032.17 65.622.02 66.221.88 73.302.28 87.791.69 78.641.76 68.182.09 51.162.46 85.481.61 70.05
TAP-RAG [15] 67.071.92 64.221.78 74.871.5474.662.06 76.051.68 87.441.46 64.881.84 46.392.24 71.751.97 70.09
RouteDRAG 74.261.1273.611.26 73.650.98 73.301.41 87.210.86 91.750.7377.271.05 48.451.53 76.611.18 75.86
Table 3: Component ablation. Accuracy and Agent use are percentages; tokens are in thousands. Dashes mark the no-Agent
setting. Best and second-best measured values are bold and underlined.
DocBench MMLongBench-Doc
Method Acc.↑Agent%↓Act./Q↓Calls/Q↓Tok./Q↓Acc.↑Agent%↓Act./Q↓Calls/Q↓Tok./Q↓
w/o Adaptive Path Routing 73.77 15.88 0.300 2.97 29.14 50.32 16.50 0.379 3.31 33.15
w/o Dynamic Agent Execution 74.19 100.00 1.418 7.11 49.30 50.25 100.00 1.850 8.23 56.53
w/o Agent Routing 73.41 – –2.01 24.0749.31 – –2.23 27.14
All Agent Routing 74.68 100.00 1.685 7.46 52.08 52.52 100.00 2.148 8.25 57.02
RouteDRAG-Efficient 74.0511.52 0.1712.58 27.36 50.8716.41 0.3483.20 32.18
RouteDRAG 75.86 78.49 1.349 6.24 46.04 53.80 80.02 1.769 7.02 50.99
the routed configurations additionally expose the accuracy–cost
trade-off of selective invocation.6.4 Robustness Analysis (Q3)
To answer Q3, we examine the sensitivity of routing decisions
to regularization and threshold choices, together with robustness
across model backbones and noisy inputs.
Hyperparameters.Fig. 4 examines the interaction between
regularization strength 𝛼and the routing threshold. For each 𝛼, the

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
0.0 0.1 0.2 0.3 0.4 0.5
Retrieval Noise Ratio6266707478Accuracy (%)
(a) Retrieval Noise
DocBench
RouteDRAG
TAP-RAG
M3DocRAG
RAG-Anything
TextRAG
0.0 0.1 0.2 0.3 0.4 0.5
Retrieval Noise Ratio32364044485256
(b) Retrieval Noise
MMLongBench-Doc
RouteDRAG
TAP-RAG
M3DocRAG
RAG-Anything
TextRAG
0.0 0.1 0.2 0.3 0.4 0.5
Document Distractor Ratio6266707478
(c) Document Distractors
DocBench
RouteDRAG
TAP-RAG
M3DocRAG
RAG-Anything
TextRAG
0.0 0.1 0.2 0.3 0.4 0.5
Document Distractor Ratio32364044485256
(d) Document Distractors
MMLongBench-Doc
RouteDRAG
TAP-RAG
M3DocRAG
RAG-Anything
TextRAG
Figure 5: Robustness comparison under increasing noise ratios. (a–b) Retrieval noise. (c–d) Document distractors.
5 10 15 20 25 30
Inference Latency (s/query)404550556065707580Accuracy (%)RouteDRAG(a) DocBench
5 10 15 20 25 30
Inference Latency (s/query)3540455055Accuracy (%)RouteDRAG(b) MMLongBench-Doc
BGE-M3M3Doc RAG-AMoLoTAP Ours051015202530Peak GPU Memory (GB)11.416.819.121.025.9
20.8(c) DocBench
BGE-M3M3Doc RAG-AMoLoTAP Ours051015202530Peak GPU Memory (GB)10.215.918.620.826.5
20.6(d) MMLongBench-DocDirect MLLM (budgeted)
BM25 TextRAGDense TextRAG
M3DocRAGRAG-Anything
MoLoRAGViDoRAG
TAP-RAGRouteDRAG
Figure 6: Practical efficiency comparison with representative RAG baselines. (a–b) Answer accuracy versus inference latency on
DocBench and MMLongBench-Doc. (c–d) Peak GPU memory consumption across different methods.
predictor is fitted on the training documents, and a reference thresh-
old is calibrated on development documents. The horizontal axis
varies a fixed offset from this reference; each query is then routed
independently. Accuracy varies modestly across the displayed set-
tings on DocBench. MMLongBench-Doc is more sensitive when
strong regularization is combined with a positive threshold off-
set. Stronger regularization compresses the prediction scale, so the
same offset can exclude more Agent executions. Regularization and
threshold calibration should therefore be considered together.
Table 4: Backbone generalization on DocBench and
MMLongBench-Doc. Accuracy (%) and relative reduction in
answering model calls compared with Uniform Agent (%).
VLM Backbone Agent Random Ours Calls Red. (%)
Qwen3-VL-4B 69.60 /45.92 68.20/43.2170.15/46.6513.8/14.2
Qwen3-VL-8B 74.68 /52.52 73.66/49.8875.86/53.8016.4/14.9
Qwen2.5-VL-7B 72.78 /48.4970.40/45.9273.32/48.30 14.1/14.4
InternVL3-8B 73.50 /50.23 71.10/47.5974.23/51.0515.0/14.3
Gemma-3-12B71.96/47.57 69.54/45.97 71.87 /48.4913.7/15.2
Backbone Generalization.Table 4 evaluates RouteDRAG
across multimodal backbones of different scales and architectures.
With Qwen3-VL-4B, RouteDRAG achieves 70.15%/46.65% accuracy
on DocBench/MMLongBench-Doc, improving over uniform Agent
execution by 0.55/0.73 percentage points while reducing answering
calls by 13.8%/14.2%. With InternVL3-8B, it improves accuracy by
0.73/0.82 percentage points. Gains vary across backbones.Robustness to Noisy Information.We consider two pertur-
bations:Retrieval Noise, which adds irrelevant content to retrieved
results, andDocument Distractors, which adds unrelated pages to
documents. Fig. 5 reports accuracy as the perturbation ratio in-
creases from 0 to 0.5. At the highest ratio, RouteDRAG achieves
73.9%/52.3% accuracy on DocBench/MMLongBench-Doc under Re-
trieval Noise and 74.5%/51.3% under Document Distractors. It out-
performs TAP-RAG by 5.9/13.5 and 8.5/12.4 percentage points under
the two perturbations, respectively.
6.5 Efficiency Analysis (Q4)
To answer Q4, we evaluate answering calls and tokens, inference
latency, and peak GPU memory consumption.
Efficiency of Selective Execution.Table 3 shows that Rout-
eDRAG uses 6.24/7.02 answering calls and 46.04k/50.99k tokens
per query on DocBench/MMLongBench-Doc. Compared withAll
Agent Routing, it improves accuracy by 1.18/1.28 percentage points
while reducing calls by 16.4%/14.9% and tokens by 11.6%/10.6%.
Compared withw/o Dynamic Agent Execution, it improves accuracy
by 1.67/3.55 percentage points while reducing both calls and tokens.
The lower costs ofw/o Adaptive Path Routingandw/o Agent Routing
come with reduced accuracy.
Practical Efficiency Comparison.Fig. 6(a–b) compares
answer accuracy against inference latency on DocBench and
MMLongBench-Doc; panels (c–d) compare peak GPU memory
consumption. These measurements complement the answering-call
and token costs in Table 3.

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
7 Conclusion
In this paper, we investigated a fundamental question in Agentic
document RAG: when is Agentic execution necessary, and how
should its operations be controlled? Our empirical findings re-
veal that Agentic execution provides query-dependent benefits,
while additional operations yield gains, ties, and degradations in
answer correctness. Motivated by these observations, we proposed
RouteDRAG, a framework that combines adaptive path routing
with dynamic Agent execution through sequential policy learn-
ing. By learning from observed execution outcomes, RouteDRAG
selectively invokes Agent and refines its subsequent operations.
Experiments on two multimodal document benchmarks demon-
strate improved accuracy with fewer answering calls compared
with uniform Agent execution. These results highlight the impor-
tance of controlling not only whether an Agent is invoked, but also
how it executes. While this work focuses on single-turn document
question answering, extending selective execution to multi-turn
document interactions presents a promising direction for future
research.

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In
The Twelfth International Conference on Learning Representations. OpenReview.net,
Vienna, Austria, 30 pages. https://openreview.net/forum?id=hSyW5go0v8
[2]Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al .2025. Qwen3-VL Technical Report.
arXiv:2511.21631 https://arxiv.org/abs/2511.21631
[3]Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu
Florian, Martin Franz, Dinesh Garg, Dinesh Khandelwal, Scott McCarley, Michael
McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus, John Pitrelli, Saurabh Pujar,
Salim Roukos, Andrzej Sakrajda, Avi Sil, Rosario Uceda-Sosa, Todd Ward, and
Rong Zhang. 2020. The TechQA Dataset. InProceedings of the 58th Annual Meeting
of the Association for Computational Linguistics. Association for Computational
Linguistics, 1269–1278. doi:10.18653/v1/2020.acl-main.117
[4]Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang,
Dawei Yin, and Jiaxin Mao. 2025. MAO-ARAG: Multi-Agent Orchestration for
Adaptive Retrieval-Augmented Generation. arXiv:2508.01005 https://arxiv.org/
abs/2508.01005
[5]Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over
Financial Data. InProceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing. Association for Computational Linguistics, 3697–
3711. doi:10.18653/v1/2021.emnlp-main.300
[6]Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024.
M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-
document Understanding. arXiv:2411.04952 https://arxiv.org/abs/2411.04952
[7]Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt
Gardner. 2021. A Dataset of Information-Seeking Questions and Answers
Anchored in Research Papers. InProceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies. Association for Computational Linguistics, 4599–4610.
doi:10.18653/v1/2021.naacl-main.365
[8]Yaxin Du, Junru Song, Yifan Zhou, Cheng Wang, Jiahao Gu, Zimeng Chen,
Menglan Chen, Wen Yao, Yang Yang, Ying Wen, et al .2026.𝐺2-Reader: Dual
Evolving Graphs for Multimodal Document QA.arXiv preprint arXiv:2601.22055
(2026).
[9]Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald. 2026. SPARKLE: A Structured
and Plug-and-play Agentic Retrieval Policy for Adaptive RAG Models. InProceed-
ings of the 64th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). Association for Computational Linguistics, San Diego,
California, United States, 38699–38719. doi:10.18653/v1/2026.acl-long.1793
[10] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline
Hudelot, and Pierre Colombo. 2025. Colpali: Efficient document retrieval with
vision language models. InInternational Conference on Learning Representations,
Vol. 2025. 61424–61449.
[11] Song Feng, Hui Wan, Chulaka Gunasekara, Siva Patel, Sachindra Joshi, and Luis
Lastras. 2020. doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, 8118–8128. doi:10.18653/
v1/2020.emnlp-main.652
[12] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. RAG-
Anything: All-in-One RAG Framework. arXiv:2510.12323 https://arxiv.org/abs/
2510.12323
[13] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025. Association for Computational Lin-
guistics, Suzhou, China, 10746–10761. doi:10.18653/v1/2025.findings-emnlp.568
[14] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). Association for Computational
Linguistics, Mexico City, Mexico, 7036–7050. doi:10.18653/v1/2024.naacl-long.389
[15] Zhong Ji, Keqi Jin, Yan Zhang, and Jiasheng Li. 2026. TAP-RAG: Task-Aware Policy
Control for Long-Document Multimodal Question Answering. arXiv:2607.18917
https://arxiv.org/abs/2607.18917
[16] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InProceedings of the 2023 conference on empirical methods in natural
language processing. 7969–7992.
[17] Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu,
Xingwei Qu, Yangguang Li, Wanli Ouyang, et al .2024. Graphreader: Building
graph-based agent to enhance long-context abilities of large language models.
InFindings of the Association for Computational Linguistics: EMNLP 2024. 12758–
12786.[18] Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, Weizhuo
Chen, Jianjun Li, and Zhiyuan Ma. 2026. OPERA: A Reinforcement Learning–
Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented
Multi-Hop Retrieval.Proceedings of the AAAI Conference on Artificial Intelligence
40, 38 (2026), 32258–32266. doi:10.1609/aaai.v40i38.40499
[19] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan
Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al .2024. Mmlongbench-doc: Benchmarking
long-context document understanding with visualizations.Advances in Neural
Information Processing Systems37 (2024), 95963–96010.
[20] Multi-Linguality Multi-Functionality Multi-Granularity. 2024. M3-Embedding:
Multi-linguality, multi-functionality, multi-granularity text embeddings through
self-knowledge distillation.arXiv preprint arXiv:2402.03216(2024).
[21] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[22] Zhivar Sourati, Zheng Wang, Marianne Menglin Liu, Yazhe Hu, Mengqing Guo,
Sujeeth Bharadwaj, Kyu J Han, Tao Sheng, Sujith Ravi, Morteza Dehghani, et al .
2026. Lad-rag: Layout-aware dynamic rag for visually-rich document understand-
ing. InProceedings of the 64th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 15945–15968.
[23] Yubo Sun, Chunyi Peng, Yukun Yan, Shi Yu, Zhenghao Liu, Chi Chen, Zhiyuan
Liu, and Maosong Sun. 2025. Visrag 2.0: Evidence-guided multi-image reasoning
in visual retrieval-augmented generation.arXiv preprint arXiv:2510.09733(2025).
[24] Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and Sihong Xie. 2025. MBA-
RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through
Question Complexity. InProceedings of the 31st International Conference on Com-
putational Linguistics. Association for Computational Linguistics, Abu Dhabi,
UAE, 3248–3254. https://aclanthology.org/2025.coling-main.218/
[25] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun
Xie, and Feng Zhao. 2025. ViDoRAG: Visual Document Retrieval-Augmented
Generation via Dynamic Iterative Reasoning Agents. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing. Association
for Computational Linguistics, Suzhou, China, 9113–9134. doi:10.18653/v1/2025.
emnlp-main.464
[26] Zhengren Wang, Dongsheng Ma, Huaping Zhong, Jiayu Li, Wentao Zhang, Bin
Wang, and Conghui He. 2026. Agenticocr: Parsing only what you need for
efficient retrieval-augmented generation.arXiv preprint arXiv:2602.24134(2026).
[27] Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. 2025.
Molorag: Bootstrapping document understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing. 14035–14056.
[28] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.
arXiv preprint arXiv:2210.03629(2022).
[29] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao
Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al .2025. Visrag: Vision-based retrieval-
augmented generation on multi-modality documents. InInternational Conference
on Learning Representations, Vol. 2025. 21074–21098.
[30] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang,
Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question An-
swering Benchmark on a Hybrid of Tabular and Textual Content in Finance. In
Proceedings of the 59th Annual Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference on Natural Language Processing
(Volume 1: Long Papers). Association for Computational Linguistics, 3277–3287.
doi:10.18653/v1/2021.acl-long.254
[31] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng
Zhang, Hai Zhao, and Dong Yu. 2025. DocBench: A Benchmark for Evaluating
LLM-based Document Reading Systems. InProceedings of the 4th International
Workshop on Knowledge-Augmented Methods for Natural Language Processing.
Association for Computational Linguistics, Albuquerque, New Mexico, USA,
359–373. doi:10.18653/v1/2025.knowledgenlp-1.29
[32] Yilong Zuo, Xunkai Li, Jing Yuan, Qiangqiang Dai, Hongchao Qin, and Ronghua
Li. 2026. MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic
Multimodal RAG in Long-Document QA. arXiv:2606.15906 https://arxiv.org/
abs/2606.15906

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
A Experimental Details
A.1 Datasets, Readers, and Execution Paths
We evaluate RouteDRAG on all 1,102 DocBench questions from
229 documents and 1,091 MMLongBench-Doc questions from 135
documents. All paths share the document index and frozen BGE-M3
representations, using Qwen3-VL-8B-Instruct with an FP8 backend;
final Agent synthesis is capped at 8,192 output tokens. The desig-
nated short path uses Standard or eligible Direct VLM, while the
Actordenotes the Agent with its native Planner, Workspace, legal-
action filtering, and learned intervention. For controlled allocation
comparisons, we reuse the recorded short-path and Actor answers
for each query, keeping Exact Specialist outputs fixed across poli-
cies.
A.2 Baseline Configurations
RAG-Anything uses upstream mixed retrieval with entity top- 𝑘=
50and chunk top- 𝑘=10. It reuses the canonical graph index and
uses a Qwen FP8 service with a one-sentence answer instruction.
MoLoRAG retains native multimodal retrieval and traversal. BM25
and BGE-M3 share native MinerU text chunks, including native
table text, with document-local top-20 retrieval and a one-sentence
Qwen FP8 reader. The BGE-M3 baseline retrieves native text chunks
using frozen embeddings.
M3DocRAG follows the comparison adapter: document-local
ColPali [ 10] retrieval selects three pages, rendered at 144 DPI for
the Qwen FP8 reader.
Budgeted Direct MLLM reads all pages up to 128. For the remain-
ing baselines, we preserve their original retrieval and execution
procedures while using the same Qwen3-VL-8B-Instruct answering
model and evaluation protocol across all methods.
A.3 Robustness Evaluation Protocol
We evaluate RouteDRAG and representative RAG baselines on
DocBench and MMLongBench-Doc under two types of perturba-
tions, with noise ratios𝜌∈{0,0.1,0.2,0.3,0.4,0.5}. A ratio of zero
denotes the original setting. Distractors are sampled from other doc-
uments in the same benchmark, excluding the source document of
the current query. The questions, reference answers, model param-
eters, and routing thresholds remain fixed throughout evaluation.
Retrieval Noise.For each query, we obtain the native retrieval
results of each method and denote the number of retrieved units
by𝐾. We then append ⌊𝜌𝐾⌋ irrelevant units sampled from the
distractor pool, preserving the original results. Thus, 𝜌=0.5adds
distractors equal to half the original number of retrieved units,
while preserving the original retrieved content. The document
index and retrieval model remain unchanged, isolating interference
introduced at retrieval time.
Document Distractors.For a document containing 𝑃pages,
we append⌊𝜌𝑃⌋ unrelated pages sampled from other documents,
inserting them at randomly selected positions while preserving the
relative order of the original pages. At 𝜌=0.5, the number of added
pages is half the original page count. Each method processes the
perturbed documents using its native indexing pipeline, including
index reconstruction where required. This setting evaluates the
effect of irrelevant pages present before retrieval.A.4 Inference Efficiency Measurement
We measure inference latency and peak GPU memory for Route-
DRAG and representative RAG baselines under the same hardware
and runtime conditions. All methods use batch size 1, with five
warm-up queries followed by 100 timed queries drawn from the
same question set. We report the mean end-to-end latency per query
and peak GPU memory usage. Experiments run on a single NVIDIA
A100 GPU, with inference precision and backend following each
method’s configuration.
End-to-end latency includes online retrieval, routing, Agent plan-
ning and tool execution, and final answer generation. Offline docu-
ment parsing and index construction are excluded. Peak GPU mem-
ory includes model weights and runtime allocations. All methods
retain their designated retrieval and answering pipelines; Route-
DRAG additionally includes routing and learned execution-control
overhead.
A.5 Routing Parameters
For each query, a fixed policy first designates Standard or Direct
VLM as the short path 𝐶(𝑞) . The learned router then predicts the
signed benefit of Agent execution relative to 𝐶(𝑞) , using the query
representation and short-path identity. Agent is invoked when the
predicted benefit exceeds the fixed threshold 𝜏=0.02; otherwise,
𝐶(𝑞)is executed.
The router is trained on paired Agent and short-path outcomes.
Its weights are fitted on training data, while model configurations
and routing thresholds are selected using development data. The
resulting routing function is applied independently to each query
with its fixed threshold.
A.6 Document-Disjoint Cross-Validation
We evaluate the learned Controller and router using five-fold cross-
validation at the source-PDF level, stratified by dataset. Questions
from the same PDF are assigned to the same outer fold, with dupli-
cate copies grouped under a common PDF identity. In each round,
one fold serves as the test set. Among the remaining documents,
20% form an inner development set and the rest form the train-
ing set. The resulting training/development/test proportions are
approximately 64%/16%/20% of documents. Question counts vary
with the number of questions associated with each document.
In each round, the shared projection and operation Controller
are trained using operation-level outcomes from the training par-
tition, with checkpoint selection based on the inner development
set. They are then frozen, and a separate router is fitted using
the training questions and their paired correctness targets, Δ𝑖=
𝑌(𝐴𝜙(𝑞𝑖))−𝑌(𝐶(𝑞 𝑖)). The inner development set determines the
router configuration and decision threshold. The trained compo-
nents and selected threshold remain fixed when applied to the outer
test fold.
We pool the predictions from the five outer test folds, so each
query in the routing evaluation contributes one out-of-fold routing
decision. Dataset-level routing results are aggregated from these
held-out predictions.

WWW ’27, May 10–14, 2027, Dublin, Ireland Zekai Chen*, Hao Zhu*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li
B Method Implementation Details
B.1 Shared Index and Source References
Figure 3 distinguishes the vector index 𝐼𝐷, the entity–relation graph
G𝐷, and source references 𝜌𝐷. The vector index supports similar-
ity retrieval of text chunks and entity descriptions, while graph
relations enable connected retrieval. Visual descriptions are in-
dexed with links to their original pages and image assets. Source
references retain document-local identifiers and locations, allowing
retrieved descriptions to be traced to their source tables or figures.
Retrieving a visual description and inspecting its original image
are separate operations.
Frozen semantic vectors have 1,024 dimensions. A shared learned
projection maps query, chunk, and entity representations to 64
dimensions. Each input contains up to eight chunk vectors and four
entity vectors, with unavailable entries padded and masked before
attention. The graph branch pools retrieved entity representations,
while relations remain available for graph retrieval. Lightweight
statistics describe query form, available content, and query–content
similarity; these features are computed from information available
at each decision point.
B.2 Exact and Short-Path Interfaces
Exact is a deterministic metadata gate that returns an answer only
when a supported structured request satisfies the required com-
pleteness checks. Otherwise, the fixed short-path policy selects
Standard by default or Direct VLM when visual inspection is eli-
gible for the query. Standard performs graph-backed retrieval and
answering. Direct VLM first applies a text precheck, invokes visual
reading when needed, and falls back to text when visual reading is
unnecessary.
The designated short path is denoted by 𝐶(𝑞) . Adaptive Path
Routing predicts whether the Agent offers sufficient benefit over
the designated shorter path 𝐶(𝑞) to justify additional execution.
Exact outputs remain fixed across allocation policies, and the Agent,
denoted as theActor, retains its native Planner, Workspace, legal-
action constraints, and tool interfaces.
B.3 Workspace, Legal Actions, and Tool
Arguments
The query-specific Workspace stores retrieved text, entities, rela-
tions, source references, tool outcomes, and action history. Each
tool result is committed with its provenance, and duplicate content
is removed. Initial retrieval creates the starting state; subsequent
operations update the same Workspace while reusing the shared
document index.
At step𝑡, execution rules construct the legal action set Γ𝑡from the
available resources and current Workspace. Candidate operations
include text search, graph expansion, structured lookup, visual
inspection, and FINISH when permitted. The learned Controller
scores only legal actions, while the Planner proposes an action and
its arguments. If the proposed action is retained, its arguments are
reused; otherwise, the Planner generates arguments for the selected
tool.
The Controller receives projected query and content rep-
resentations, state statistics, operation embeddings, and aneight-dimensional history vector comprising the step, history
length, five operation counts, and failure count, transformed
bylog(1+𝑥). Operation embeddings have 64 dimensions. The
action-value head uses a 32-unit hidden layer and a scalar output,
with FINISH as the reference and a fixed refinement margin of
𝑚=0.15. The relative scores estimate operation-level correctness
gains against the FINISH reference. After a permitted stop or
execution limit, the original synthesis procedure generates the
final answer from the accumulated content and source references.
B.4 Learning Signals
The labels𝑌stopand𝑌continue in Figure 3 denote answer correctness
before and after one additional operation, respectively, as defined
in Equation 11. Only evaluated state–action pairs contribute to
the regression and ranking losses. Planner imitation provides an
auxiliary signal that preserves the original Planner’s operation
preferences.
At the path level, 𝑟𝐴=𝑌(𝐴𝜙(𝑞))and𝑟𝐶=𝑌(𝐶(𝑞)) denote the
paired correctness outcomes of the Actor and designated short
path. Their difference, 𝑟𝐴−𝑟𝐶, provides the signed routing target in
Equation 14. The shared projection and Controller are trained first
and then frozen while the routing heads learn from the updated
Actor’s outcomes. Reference answers and correctness labels provide
offline supervision; inference uses only the learned representations,
predicted scores, and legal execution rules.
B.5 Training and Policy Selection
Sequential Training.The document index, semantic encoder,
and language-model backbones remain frozen. In the first stage,
the shared projection and operation Controller are trained using
observed operation-level outcomes through regression, same-state
ranking, and auxiliary Planner imitation. In the second stage, the
shared projection and Controller are frozen, and the routing heads
are trained using the paired correctness outcomes of the Actor and
the designated short path. This two-stage procedure is repeated
independently in each fold, using the training, development, and
test partitions defined by source PDF in Appendix A.6.
Checkpoint and Threshold Selection.Each stage selects its
checkpoint using the corresponding development objective. The
operation-refinement margin is fixed at 𝑚= 0.15. The per-query
routing threshold 𝜏is selected on the development set, and both
model parameters and thresholds remain fixed throughout test
evaluation.
Execution Policies.In per-query inference, the router invokes
the Actor when the predicted relative benefit bΔ𝜃(𝑞)exceeds𝜏; other-
wise, it uses the designated short path. Once invoked, the Controller
refines the Planner’s proposal only when a legal alternative exceeds
the proposed action’s score by the fixed refinement margin.
The high-accuracy configuration uses the shared router and
fixed threshold specified in Appendix A.5. Each query is routed
independently: the Agent is invoked when bΔ(𝑞)>𝜏 ; otherwise,
the designated short path is used.

RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG WWW ’27, May 10–14, 2027, Dublin, Ireland
1–10
n=33911–30
n=31531–100
n=194101–200
n=154201+
n=100
Document pages (questions per bin)0.51.01.5Agent tool actions / query
DocBench
1–10
n=1611–30
n=55631–100
n=404101–200
n=110201+
n=5
Document pages (questions per bin)0.00.51.01.52.02.5Agent tool actions / query
MMLRouteDRAG Rule-Based Routing Uniform Agent
Figure A1: Tool use across document-length groups. Counts show questions per
bin; the MML 201+ bin contains five questions from one document.
0 10 20 30 40 50 60
Training epoch0.300.350.400.450.50DEV objective (MSE + 0.25 rank)DEV objective
Selected epochFigure A2: Benefit-predictor validation loss.
The marker identifies the selected epoch,
47.
C Supplementary Execution Profiles
C.1 Natural Document-Length Profile
Figure A1 groups the evaluated documents by page count and
reports tool actions per query. RouteDRAG uses fewer actions than
uniform execution across the displayed groups. Tool use also varies
within the length range, reflecting query requirements as well as
document size. Question counts accompany each bin; the longest
MML bin contains five questions from one PDF.
C.2 Validation-Loss Trace
Figure A2 plots the benefit predictor’s validation objective: signed-
benefit regression plus a ranking term weighted by 0.25. Train-
ing uses paired execution outcomes, and validation selects epoch
47. The trace describes optimization of the routing objective; the
workload and document-disjoint allocation results are reported
separately.
C.3 Per-Query Routing Operating Points
Table A1 compares four routing configurations with fixed thresh-
olds calibrated on the development set. RouteDRAG-Efficient,
RouteDRAG-Medium, RouteDRAG-High, and RouteDRAG share
the accuracy-oriented router with different thresholds. RouteDRAG-
Efficient achieves accuracies of 74.05% and 50.87% on DocBench
and MMLongBench-Doc, respectively, while limiting Agent usage
to 11.52% and 16.41%, with 2.58 and 3.20 calls per query. Increasing
the routing intensity improves accuracy at the cost of higher
resource usage. RouteDRAG-Medium achieves 74.73%/52.29%
accuracy at 42.65%/57.56% Agent usage, while RouteDRAG-High
reaches 75.18%/52.93% accuracy at 59.53%/71.13% Agent usage on
the two benchmarks, respectively. RouteDRAG achieves the highest
accuracy, at 75.86% on DocBench and 53.80% on MMLongBench-
Doc, but also incurs the highest Agent usage (78.49%/80.02%) and
the most calls per query (6.24/7.02). Overall, these operating points
illustrate the trade-off between accuracy and inference cost enabled
by routing decisions.Table A1: Accuracy–cost trade-offs of per-query routing con-
figurations. Accuracy and actual Agent usage are percentages.
DocBench MMLongBench-Doc
Configuration Acc.↑Agent%↓Calls/Q↓Acc.↑Agent%↓Calls/Q↓
RouteDRAG-Efficient 74.05 11.52 2.58 50.87 16.41 3.20
RouteDRAG-Medium 74.73 42.65 4.22 52.29 57.56 5.72
RouteDRAG-High 75.18 59.53 5.09 52.93 71.13 6.57
RouteDRAG 75.86 78.49 6.24 53.80 80.02 7.02