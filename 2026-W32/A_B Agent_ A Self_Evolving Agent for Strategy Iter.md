# A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing

**Authors**: Zhuohang Jiang, Yuxin Chen, Yongsen Pan, Zheng Hu, Wenqi Fan, Qing Li, Hongyang Wang, Jun Wang, Wenwu Ou

**Published**: 2026-08-05 09:28:43

**PDF URL**: [https://arxiv.org/pdf/2608.04625v1](https://arxiv.org/pdf/2608.04625v1)

## Abstract
Industrial recommendation strategy iteration heavily relies on large-scale A/B experimentation. Traditional tuning requires experts to repeatedly design strategies, configure experiments, analyze results, and adjust parameters, making the process labor-intensive and time-consuming. Meanwhile, valuable knowledge from historical experiments is often fragmented, making systematic reuse difficult through manual expert effort alone. Existing RAG agents partially alleviate this burden by retrieving prior strategies, but typically organize experience in a flat manner, overlooking the hierarchical relationships among business scenarios, recommendation stages, optimization objectives, and experimental contexts. This often results in mismatched retrieval and limited cross-scenario transfer, while preventing agents from continuously refining strategies and parameters through sequential A/B feedback. %
To address these limitations, we propose A/B Agent, a closed-loop A/B agent for industrial recommendation strategy optimization. The framework comprises three tightly coupled core components: Historical Strategy Knowledge Organization, Autonomous Target-Aware Strategy Generation, and Experiment-Guided Strategy Self-Evolution. It organizes historical strategies into a hierarchical experience tree, retrieves transferable evidence through multi-path Tree-RAG to generate executable strategies, and continuously analyzes online A/B feedback to guide autonomous tuning and update the experience tree for self-evolution. Extensive offline and online evaluations demonstrate its effectiveness, including a 4.829% improvement in GMV in a real-world short-video e-commerce recommendation system while maintaining positive gains across all guardrail metrics.

## Full Text


<!-- PDF content starts -->

A/B Agent: A Self-Evolving Agent for Strategy Iteration in
Industrial A/B Testing
Zhuohang Jiang∗
The Hong Kong Polytechnic
University
Hong Kong SAR, China
zhuohang.jiang@connect.polyu.hkYuxin Chen†
Kuaishou Technology
Beijing, China
chenyuxin06@kuaishou.comYongsen Pan∗
University of Electronic Science and
Technology of China
Chengdu, SiChuan, China
panys@std.uestc.edu.cn
Zheng Hu∗
Southwest Jiaotong University
Chengdu, SiChuan, China
huzheng@swjtu.edu.cnWenqi Fan†
The Hong Kong Polytechnic
University
Hong Kong SAR, China
wenqifan03@gmail.comQing Li†
The Hong Kong Polytechnic
University
Hong Kong SAR, China
csqli@comp.polyu.edu.hk
Hongyang Wang
Kuaishou Technology
Beijing, China
liboyu08@kuaishou.comJun Wang
Kuaishou Technology
Beijing, China
wangjun03@kuaishou.comWenwu Ou
Kuaishou Technology
Beijing, China
luocheng10@kuaishou.com
Abstract
Industrial recommendation strategy iteration heavily relies on large-
scale A/B experimentation. Traditional tuning requires experts to
repeatedly design strategies, configure experiments, analyze re-
sults, and adjust parameters, making the process labor-intensive
and time-consuming. Meanwhile, valuable knowledge from his-
torical experiments is often fragmented, making systematic reuse
difficult through manual expert effort alone. Existing RAG agents
partially alleviate this burden by retrieving prior strategies, but
typically organize experience in a flat manner, overlooking the hier-
archical relationships among business scenarios, recommendation
stages, optimization objectives, and experimental contexts. This
often results in mismatched retrieval and limited cross-scenario
transfer, while preventing agents from continuously refining strate-
gies and parameters through sequential A/B feedback. To address
these limitations, we proposeA/B Agent, a closed-loop A/B agent
for industrial recommendation strategy optimization. The frame-
work comprises three tightly coupled core components:Historical
Strategy Knowledge Organization,Autonomous Target-Aware Strat-
egy Generation, andExperiment-Guided Strategy Self-Evolution. It
organizes historical strategies into a hierarchical experience tree,
retrieves transferable evidence through multi-path Tree-RAG to
generate executable strategies, and continuously analyzes online
∗Work done during an internship at Kuaishou Technology.
†Corresponding Author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXA/B feedback to guide autonomous tuning and update the experi-
ence tree for self-evolution. Extensive offline and online evaluations
demonstrate its effectiveness, including a4.829%improvement in
GMV in a real-world short-video e-commerce recommendation sys-
tem while maintaining positive gains across all guardrail metrics.
CCS Concepts
•Information systems→Recommender systems.
Keywords
Transfer Learning, Large Language Models, Agent, A/B Test.
ACM Reference Format:
Zhuohang Jiang, Yuxin Chen, Yongsen Pan, Zheng Hu, Wenqi Fan, Qing
Li, Hongyang Wang, Jun Wang, and Wenwu Ou. 2018. A/B Agent: A Self-
Evolving Agent for Strategy Iteration in Industrial A/B Testing. InPro-
ceedings of Make sure to enter the correct conference title from your rights
confirmation email (Conference acronym ’XX).ACM, New York, NY, USA,
10 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
In modern industrial recommender systems [ 11,16,23], continu-
ous strategy development [ 24], large-scale A/B testing [ 26], and
feedback-driven parameter optimization [ 9] are essential for im-
proving recommendation quality and performance. A strategy iter-
ation [ 3,14] typically spans objectives, ranking pipelines, scoring
functions, parameter configurations, and engineering constraints,
followed by small-scale validation, traffic ramp-up, and full deploy-
ment. Because multiple business domains share limited exposure,
optimizing one objective may suppress others and harm user expe-
rience. Therefore, strategy iteration [ 8] must jointly optimize core
metrics and satisfy multiple guardrail constraints [ 25] to ensure
balanced traffic allocation and long-term system health.
However, as shown in Figure 1 (a), existing industrial strat-
egy iteration still relies heavily on human expertise [ 4]. Faced
arXiv:2608.04625v1  [cs.AI]  5 Aug 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhuohang Jiang et al.
A/B Strategy 
History
How can we 
improve Short -
Video  Cart 
GMV at the 
MixRank  stage?(a)  Traditional A/B Tuning
(b) Other RAG Agent
(c)  Our A/B Agent
Manual 
Experiment
Time
Consuming
Fragmented
Experience
Historical 
Knowledge
Autonomous 
Strategy
Generation
Background
Requirement
Title Cross -Domain Interest Transfer
E-Commerce Ranking 2026.03
Background: 
Sparse purchase signals limit user modeling...
Strategy: 
Transfer short -video interests to product ranking
Parameter Update : 
Ranking Score = P CTR * P CVR * P Interest Score .
Performance : 
Cart GMV +3.25%.
Title Object -Aware Re -ranking
Live MixRank 2026.02…
Code Stage…
Scenario
GMV:
+4.829%How can we improve Short -Video  
Cart GMV at the MixRank  stage?
Limited 
Experiment 
Autonomy
Weak 
Knowledge 
Transfer
Self -
Evolution
GMV
+2.679%
GMV
+0.749%
How can we improve Short -Video  
Cart GMV at the MixRank  stage?
Multi -Path
Retrieve
Without Scenario 
& Stage Info 
Strategy: Boost CTR Score
Parameter: PCTR * P CVR * Strategy: Boost Intent Score
Parameter: PIntent *1.2 
Flat RAG
Tree RAG Safety
 Autonomous
Analysis
ThinkingA/B Test
Tired
Figure 1: Motivation of A/B Agent. Traditional A/B tuning re-
quires repeated manual experimentation, while Other RAG
Agent often misses scenario- and stage-specific context. In
contrast, A/B Agent leverages hierarchical historical knowl-
edge, autonomous strategy generation, and self-evolution to
optimize strategies effectively.
with numerous interdependent—and sometimes conflicting—core
and guardrail metrics [ 15], engineers must identify performance
gains [ 2], assess statistical significance, attribute metric changes [ 36],
and balance potential risks. Analyzing a typical experiment and de-
termining the next parameter configuration often requires approxi-
mately hours of work by an experienced engineer [ 6]. Meanwhile,
historical strategy knowledge [ 21] is fragmented across heteroge-
neous sources [ 29], including documents, source code, experiment
configurations, and metric logs. These sources lack a unified ab-
straction [ 27] of strategy mechanisms, applicability conditions, and
transferability boundaries, making historical knowledge difficult to
systematically retrieve and reuse. Even when a similar historical
strategy is identified, differences in user intent, ranking pipelines,
feature availability, and business constraints [ 30] require its param-
eters, thresholds, and traffic allocation to be adapted to the target
scenario [ 34,42] and continuously refined through multiple rounds
of online experimentation [34].
Recent advances in large language models [ 17,22] and retrieval-
augmented generation [ 18,41] provide new opportunities for under-
standing and reusing industrial strategy knowledge. Nevertheless,
conventional RAG agents remain insufficient for complex strategy
iteration [ 5,28]. Flat RAG systems primarily retrieve independent
text chunks according to semantic similarity and cannot adequately
model the hierarchical relationships among application scenarios,
business pipelines, ranking stages, optimization objectives, param-
eter configurations, and experimental outcomes [ 1,40]. This lim-
itation restricts reliable knowledge transfer across scenarios and
pipelines [ 12,32]. Moreover, existing RAG agents [ 31] typicallyprovide only one-shot strategy recommendations and lack a long-
horizon optimization framework that connects strategy generation,
online experimentation, metric analysis, and parameter updates.
Consequently, as shown in Figure 1 (b), they cannot autonomously
refine strategies from continuous feedback or progressively consol-
idate successful practices and failed experiments into an strategy
knowledge base.
To address the aforementioned challenges,A/B Agentis pro-
posed as an agentic framework for industrial recommendation
strategy knowledge construction, strategy initialization, online op-
timization, and experience self-evolution. The framework trans-
forms fragmented historical experiment records into structured and
transferable strategy experiences. Given a new business request, it
retrieves relevant historical knowledge under the target context and
deployment constraints to initialize an executable strategy. After
deployment, the system continuously analyzes core and guardrail
metrics, compares successive experiment variants, and generates
the strategy and parameter configuration for the next testing round.
Validated outcomes are subsequently written back to the strategy
experience tree, forming a closed loop of experience accumulation,
strategy initialization, online tuning, and knowledge evolution.
The framework consists of three core components,Historical
Strategy Knowledge Organization,Autonomous Target-Aware Strat-
egy Generation, andExperiment-Guided Strategy Self-Evolution. To
overcome fragmented experience, as shown in Figure 1 (c), his-
torical A/B records are decomposed into reusable strategy chunks
containing motivations, mechanisms, parameters, applicable con-
texts, outcomes, and risks, and are organized into a hierarchical
experience tree by domain, scenario, recommendation stage, and
objective. To improve the weak knowledge transfer of flat RAG,
the initialization module performs multi-path sparse and dense
retrieval, hierarchy-aware candidate enhancement, and reranking,
thereby selecting context-matched evidence and generating an ex-
ecutable strategy with feasible parameter settings and controlled
guardrail risks. After deployment, the self-evolution module re-
duces manual and time-consuming tuning by organizing successive
strategies, parameter updates, and online metrics into an experi-
ment tree. Parent–child and sibling comparisons identify effective
modifications and promising alternatives, enabling autonomous
local parameter search or mechanism replacement when perfor-
mance saturates, or guardrails are violated. Validated strategies are
then written back to the experience tree, forming a closed loop of
knowledge organization, target-aware initialization, autonomous
A/B optimization, and experience self-evolution.
Extensive offline and online evaluations demonstrate the effec-
tiveness of A/B Agent. Across three industrial e-commerce sce-
narios, A/B Agent achieves the highest average overall score of
7.244, outperforming Claude-Sonnet-4.6 by1 .3%. Compared with
the strongest RAG baseline in each scenario, it improves the overall
score by25.0%,31.7%, and23.5%, respectively. It also consistently
surpasses GPT-5.5 in correctness and Claude-Sonnet-4.6 in novelty.
In real-world deployment, A/B Agent further achieves a4.829%
improvement in Cart GMV while maintaining positive gains across
all guardrail metrics.
The main contributions of this work are summarized as follows:

A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
•A/B Agent is introduced as a closed-loop agent for historical
strategy mining, cross-scenario transfer, A/B analysis, and
parameter optimization in industrial recommender systems.
•A hierarchical strategy experience tree is constructed to
decompose historical strategies into transferable skills and
support structured autonomous target-aware retrieval and
strategy generation.
•An experiment-tree-guided mechanism is developed for A/B
tuning and experience self-evolution, enabling joint metric
analysis, parameter optimization, and continuous knowledge
updates.
•An industrial A/B dataset is built with strategy configura-
tions, experimental outcomes, and multidimensional metrics
for evaluating strategy generation and tuning.
•Extensive offline and online experiments validate the effec-
tiveness of A/B Agent, including a4.829%improvement in
GMV in real-world deployment.
2 Related Work
LLM Agents for Recommendation.Recent advances in large
language models have promoted the development of agentic rec-
ommender systems. Existing methods [ 13,33] mainly use LLM
agents for preference understanding, conversational recommen-
dation, personalized reasoning, recommendation explanation, and
user simulation. By integrating user profiles, memory, planning, and
tool use, these agents can better capture user intents and generate
personalized recommendations. Hao et al . [10] proposed RecAgent,
an LLM-based generative agent framework that simulates users’
browsing, interaction, and social behaviors in recommendation
environments. However, most existing studies focus on user-item-
level decisions, such as preference prediction, item generation, and
feedback simulation. Industrial recommendation systems addition-
ally require strategy-level support for designing ranking strategies,
transferring historical experience, analyzing A/B results, and tun-
ing online parameters. A/B Agent shifts the focus from direct rec-
ommendation to industrial strategy iteration by mining historical
strategies, retrieving transferable experience, generating strategy
suggestions, and refining them through online A/B feedback.
Agentic Systems for Strategy Optimization.General LLM-
based agentic systems [ 19,35] integrate memory, planning, tool
use, reflection, and environmental feedback to solve complex tasks.
These frameworks [ 39] enable agents to decompose tasks, interact
with external tools, accumulate reusable experience, and iteratively
improve their decisions.Yang et al . [38] proposed SWE-agent, an
LLM-based software engineering agent equipped with a specialized
agent–computer interface for repository inspection, code modifi-
cation, and debugging. Nevertheless, most existing systems target
general tasks such as web navigation, software engineering, embod-
ied control, and interactive problem solving. They rarely consider
recommendation strategy optimization, which requires jointly rea-
soning over domain knowledge, historical experiments, ranking
mechanisms, business objectives, and online metrics. A/B Agent
addresses this gap through a recommendation-specific closed loop
that organizes historical strategies into reusable experiences, re-
trieves relevant knowledge for new requests, and uses A/B evidence
to iteratively refine strategy parameters.3 Methodology
As illustrated in Figure 2, A/B Agent consists of three components:
Historical Strategy Knowledge Organization,Autonomous Target-
Aware Strategy Generation, andExperiment-Guided Strategy Self-
Evolution. Historical experiment records are first transformed into
reusable strategy experiences and organized into a hierarchical
experience tree. Given a new optimization request, Tree-RAG re-
trieves relevant experiences according to the business context and
generates an executable initial strategy. After deployment, succes-
sive A/B experiments are organized into an experiment tree, where
strategy and parameter comparisons guide iterative optimization.
3.1 Historical Strategy Knowledge Organization
Historical Experiment Record Structuring.Historical experiment
reports are typically organized by projects or experiment batches
rather than reusable strategy units. A single report may contain
multiple strategy changes, parameter versions, and intermediate
failures, making direct document-level indexing prone to irrelevant
context and weak alignment between modifications and outcomes.
We therefore convert each record into a unified schema covering
the business background, scenario, recommendation stage, opti-
mization objective, strategy description, parameter update, core
metrics, guardrail metrics, and deployment conditions. This stan-
dardized representation serves as the input for strategy extraction
and experience tree construction.
Agentic Multi-Perspective Strategy Extraction.A strategy agent
decomposes each experiment record into atomic and self-contained
strategy chunks. Each chunk corresponds to an independent modi-
fication and preserves its motivation, mechanism, parameter config-
uration, applicability conditions, observed outcomes, and potential
risks. To support diverse query formulations, complementary views
are generated from problem, scenario, and mechanism perspectives.
Schema validation, metric consistency checking, and atomicity
checking are further applied to prevent unsupported content or
unrelated modifications from being merged.
Hierarchical Experience Tree Construction.The extracted chunks
are organized into a hierarchical strategy experience tree:
T𝐸=(V𝐸,E𝐸),(1)
whereV𝐸contains category and strategy nodes, and E𝐸represents
their hierarchical relations. Each strategy chunk 𝑐is assigned to a
semantic path:
𝜋(𝑐)=[𝑑 𝑐,𝑠𝑐,𝑙𝑐,𝑜𝑐],(2)
where𝑑𝑐,𝑠𝑐,𝑙𝑐, and𝑜𝑐denote the domain, scenario, recommenda-
tion stage, and optimization objective, respectively.
Each leaf node stores the strategy description, parameter configu-
ration, metric changes, deployment conditions, risks, and references
to the original experiment. Transferable strategies may be attached
to multiple compatible paths, enabling both precise retrieval from
matched branches and knowledge transfer across related branches.
3.2 Autonomous Target-Aware Strategy
Generation
Multi-Path Strategy Retrieval.Given an optimization request 𝑞,
A/B Agent identifies its business scenario, recommendation stage,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhuohang Jiang et al.
Historical Experiment Records
Title Cross -Domain Interest Transfer
E-Commerce Ranking 2026.03
Background: 
Sparse purchase signals limit user modeling...
Strategy: 
Transfer short -video interests to product ranking
Parameter Update : 
Ranking Score = P CTR * P CVR * P Interest Score .
Performance : 
Cart GMV +3.25%.
Title Object -Aware Re -ranking
Live MixRank 2026.02
Title Dynamic GMV Weighting
Short Video Retrieval 2026.05
…Strategy Agent Chunking
Title Cross -Domain Interest Transfer
E-Commerce Ranking 2026.03
Background: 
Sparse purchase signals limit user modeling...
Strategy: 
Transfer short -video interests to product ranking.
Parameter Update : 
Ranking Score = P CTR * P CVR * P Interest .
Performance : 
Cart GMV +3.25%.
Analyze a strategy record and 
generate self -contained knowledge 
chunks from multiple perspectives .
Chunk 1 
Chunk 2
Chunk nProblem  Perspective
Scenario  Perspective
Mechanism  PerspectiveExperience Tree Construction 
Experience Tree
E-Commence Short Video Live Stream
Retrieval Ranking MixRank Other
26.02
26.05
26.07⋄
⋄ ⋄Cross -Domain Interest Transfer
Strategy i
Experience Tree 
Q:  How can we improve Short -Video  Cart GMV at the MixRank  
stage without significantly hurting user engagement?Chunks: Target: For GMV 
Strategy j
For Order For User Parameter: PCTR * P CVR * P Interest
User Query
How can we improve … 
user engagement?
Short -Video MixRank GMV
 Eq
ETChunk 1
Live MixRank GMV
Retrieval Ranking Score:
  cos( Eq × ET) × wStage×wTarget
Chunk 2
Short -Video MixRank Order
Retrieval Ranking Score:
  cos( Eq × ET) × wStage×wScen
… Relative Boosting Strategy Generation &Judging
Q: How can we 
improve … user 
engagement?Strategy 
Experience…
<Think> I think …
Strategy: Boost CTR Score
Parameter: PCTR * P CVR * 
Agent Judging
Expert Judging
Expert DiscussionSuggestion
Strategy:
Boost CTR Score…
Parameter: 
PCTR * P CVR * …
Automated 
Experiment Launch07.02
07.05
07.07Strategy Root
Core Metrics 
Guardrail Metrics 
Strategy V1
Core Metrics 
Guardrail Metrics Strategy V2
Core Metrics 
Guardrail Metrics
Strategy V1
Core Metrics 
Guardrail Metrics Strategy V3
Core Metrics
Guardrail MetricsExperiment Understanding
Scenario: Short Video
Target: GMV
Stage: MixRank
Strategy: Boost CTR Score…Metrics Reading
Core Metrics: 
Guardrail Metrics: GMV +3pp
Live GMV 
+3ppLive Click 
-5ppLive Show 
+0.3ppGPM -4pp
Experiment Tree Construction
Tree Retrieval & Experiment  Comparison
Current Exp
Para Weight: 2
PCTR: +10pp
GMV: +3pp
Live GMV: -5pp Relate Exp 1
Para Weight: 0.5
Relate Exp 2
Para Weight: 5.0
Experiment Comparison
Iterative Strategy Tuning
<Think> I think …
Strategy: Boost CTR Score…
Weight: 0.3, 0.5, 0.7
Next Round 
A/B Test
Multi -Path 
Retrieval1. Historical Strategy Knowledge Organization
2. Autonomous Target -Aware Strategy Generation3. Experiment -Guided Strategy 
Self-Evolution
E-CommenceExperience Tree
Short Video Live Stream
Ranking MixRank Othe r Retrieva l
Figure 2: Overview of A/B Agent. The framework consists of three components: Historical Strategy Knowledge Organization,
Autonomous Target-Aware Strategy Generation, and Experiment-Guided Strategy Self-Evolution.
optimization objective, guardrail requirements, and deployment
constraints, and maps them to a semantic path in the strategy
experience tree. Candidate strategy chunks are retrieved through
complementary sparse and dense paths. Let 𝑆𝑚(𝑐,𝑞) denote the
relevance score between candidate chunk 𝑐and request 𝑞from
retrieval path𝑚. The initial retrieval score is
𝑆ret(𝑐,𝑞)=𝑀∑︁
𝑚=1𝛼𝑚Norm(𝑆𝑚(𝑐,𝑞)),(3)
where𝑀is the number of retrieval paths, 𝛼𝑚is the corresponding
weight, and Norm(·) aligns score ranges across methods. Sparse re-
trieval captures exact business terms, metric names, and parameter
identifiers, while dense retrieval identifies strategies with similar
mechanisms but different expressions. Their combination balances
exact matching and semantic recall.
Context-Aware Relevance Boosting and Reranking.Textual simi-
larity alone may retrieve strategies that are inconsistent with the
target business context. A/B Agent therefore exploits the hierarchy
of the experience tree to boost candidates that better match the re-
quest in terms of scenario, recommendation stage, and optimization
objective. Let 𝑣𝑙
𝑐and𝑣𝑙
𝑞denote the nodes associated with candidate
𝑐and request 𝑞at hierarchy level 𝑙. Their tree-path relevance isdefined as
𝐵tree(𝑐,𝑞)=∑︁
𝑙𝛽𝑙exp
−𝛾𝑙𝑑T
𝑣𝑙
𝑐,𝑣𝑙
𝑞
,(4)
where𝑑T(·,·) denotes the shortest-path distance between two
nodes in the experience tree, 𝛽𝑙controls the importance of hierar-
chy level𝑙, and𝛾𝑙determines the distance-decay rate. Structurally
matched candidates receive larger boosts, while candidates from
nearby branches retain smaller weights to support knowledge trans-
fer across related scenarios or objectives. The tree-path relevance
is combined with the initial retrieval score to perform coarse rank-
ing, from which the top- 𝐾candidates are selected. Each candidate
chunk is then paired with the original request and passed to Qwen-
Reranker. The reranker performs fine-grained scoring based on
semantic relevance, strategy mechanism, applicability conditions,
and deployment constraints, and produces a refined ranking. The
highest-ranked chunks are retained as supporting evidence for
strategy generation.
Evidence-Grounded Strategy Generation and Judging.The strat-
egy generator takes the optimization request, structured business
context, and reranked historical experiences as input. It extracts
transferable mechanisms and adapts them to the target task by
considering available model inputs, serving costs, valid parame-
ter ranges, and deployment conditions. The generated candidates

A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
How can we improve Short -Video Cart GMV at the MixRank stage 
without affect the user engagement?
(1) Query Understanding
"scenario": "Short Video"
"stage": "MixRank", 
"target": "Cart GMV"
"guardrails":
[ "User Engagement", "Live GMV"]
(2) Multi -Path Retrieval
Experience 
Tree Database
BM 25
TF-IDF
Embedding
Top-k Chunks(3) Relevant Boost & Reranking
Scenario: Short Video
Stage: MixRank
Target: Cart GMVTree-Based Boost Reranker
Ranking 
ScoreQwen
Reranker
A: GMV depends on click and conversion.
B: Add the score at MixRank.
C: Use a tunable weight for balance.(4) Reasoning  & Answer
Strategy Judger
Candidate B is better . 
Can be deploy edwithin the 
existing pipeline. Safety  Check
Rationality Check
Feasibility Check
Strategy: GMV -A ware CTR Boosting 
Initial Weight: w = 1.0, 2.0 , 5.0
StrategyGMV -A ware CTR Boosting 
Initial Weight: w = 1.0, 2.0 , 5.0Cart GMV , GPM, 
Order CountLive GMV , 
Live GPM 
Core Target
 Guardrail Target
(1) Experiment  Launch (2) Experiment  Reading
(3) Strategy  Tuning(4) Strategy  Verify
A/B  Test Link : 
https://www.abtest.com/xxxx
Open  
BrowserOpen  
WebsiteSelect  A/B 
Group
Verify  
ParameterUpload
Parameter
A/B  Test Storage : 
Link: https://www.abtest.com/xxxx
Experiment Record 
Experiment Tree
Skills
Select  A/B 
Group
Select  A/B 
Time
Read MetricsBuild  Experiment Tree
Strategy  Root
V1 V2
V3 V4
Experiment Tree
Strategy  Root
V1 V2
V3 V4
Select  Related ExperimentsParameter Based Retrieval
Our Weight: 2.0
Retrieved Weight: 0.5, 5.0
New  Weight: 
0.3, 0.5, 0.72.0: Safe, but delivers limited gains.
0.5: Significant gains, s afe guardrail metrics.
5.0: Higher GMV, user e xperience risks.Rule Evaluation
Agent Evaluation
Human Evaluation
Format Compliance
Structural ComplianceType  Compliance
Numerical Validity
The agent confirms that the strategy is logically 
sound, metric -consistent, and compliant with 
safety constraints.
Human experts verify that the strategy is 
reasonable, feasible, and suitable for online 
deployment.Skills
Strategy: GMV -A ware CTR Boosting 
Initial Weight: w = 0.7Final Result 
GMV + 4.829%
GPM  + 4.677%
Save to 
Experience TreeStrategy Decision 
Final Decision
Save to 
Experience Tree
Build new 
Experiment TreeExpert Judger Agent JudgerStrategyThinking
<Thinking…>
MixRank is the best stage 
for lightweight score fusion. 
Scenario Extract
Stage ExtractTarget Extract
Guardrail Analysis
Format Requirement
(1) Autonomous Strategy Generation (2) Strategy Self -Evolution
Figure 3: End-to-end workflow of A/B Agent, including autonomous strategy generation via target-aware multi-path retrieval
and strategy self-evolution through online A/B experimentation, feedback-driven tuning, and continuous experience updates.
are then evaluated by agent-based judges in terms of contextual
relevance, evidence consistency, engineering feasibility, expected
business gain, and guardrail risk. Candidates that lack sufficient
evidence, contradict historical outcomes, or violate deployment
constraints are filtered out. The highest-ranked candidate is con-
verted into an executable initial strategy for A/B testing, including
the strategy modification, parameter configuration, supporting evi-
dence, expected gains, and potential risks.
3.3 Experiment-Guided Strategy Self-Evolution
A/B Experiment Understanding and Outcome Extraction.After
deployment, A/B Agent aligns experiment descriptions, configura-
tion files, and platform metadata into a structured representation
containing the business scenario, recommendation stage, optimiza-
tion objective, strategy configuration, and guardrail requirements.
It further extracts changes in core and guardrail metrics together
with experiment duration, traffic allocation, and statistical confi-
dence. To reduce the influence of noisy or inconclusive results, each
metric change is adjusted as
fΔ𝑦𝑗=𝜌𝑗Δ𝑦𝑗,(5)
where Δ𝑦𝑗is the observed change of metric 𝑗,𝜌𝑗∈ [0,1]is its
confidence weight, and fΔ𝑦𝑗is the confidence-adjusted outcome.
A/B Experiment Tree Construction.Successive experiment ver-
sions are organized into an A/B experiment tree:
T∗𝐴=(V∗𝐴,E∗𝐴),(6)whereV∗𝐴 andE𝐴denote experiment nodes and transition edges.
Each node
𝑣𝑡=(𝑎𝑡,𝜽𝑡,y𝑡,𝑥𝑡)(7)
stores the tested strategy, parameter configuration, observed metric
changes, and experiment context. Each directed edge records the
strategy and parameter modifications between two versions. The
root represents the production baseline or initial strategy, child
nodes represent subsequent modifications, and sibling nodes corre-
spond to alternative configurations from the same parent.
Experiment Comparison and Utility Evaluation.For the current
experiment, A/B Agent retrieves nodes with similar scenarios, ob-
jectives, and strategy mechanisms, and performs both parent–child
and sibling comparisons. Parent–child comparison evaluates the
effect of a specific modification, while sibling comparison identifies
the most promising configuration. To jointly consider core gains
and guardrail risks, the utility of an experiment node is defined as
𝑈(𝑣)=∑︁
𝑗∈M core𝑤𝑗fΔ𝑦𝑗−𝜆∑︁
𝑘∈M guard𝑤𝑘max
0,−fΔ𝑦𝑘
,(8)
whereMcoreandMguard denote the sets of core and guardrail
metrics, respectively; 𝑤𝑗and𝑤𝑘are their importance weights;
and𝜆controls the penalty for guardrail degradation. A higher
𝑈(𝑣) indicates a better balance between target improvement and
guardrail safety.
Iterative Strategy Tuning.Based on the current outcomes, related
experiment nodes, and historical strategy experiences, A/B Agent

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhuohang Jiang et al.
Table 1: Dataset Statistics.
Scenario Retrieval Ranking MixRank Others Total
Scenario 1 27 31 5 29 92
Scenario 2 23 22 0 70 115
Scenario 3 11 23 5 64 103
Total 61 76 10 163 310
Note:Scenarios 1–3 represent three e-commerce scenarios deployed at Kuaishou.
recommends the strategy and parameter configuration for the next
round. When a branch consistently improves the target objective
without violating guardrails, the agent performs local search around
its promising parameter range. When the branch saturates unac-
ceptable degradation, the agent retrieves alternative mechanisms
from the strategy experience tree and creates a new branch. This
process repeats until the gains converge, the experiment budget is
exhausted, or guardrail constraints prevent further exploration.
4 Deployment
Figure 3 illustrates the online workflow of A/B Agent, which con-
sists of two main stages:Autonomous Strategy GenerationandStrat-
egy Self-Evolution. In the current deployment, GLM-5.1 serves as
the foundation model for strategy understanding, generation, and
iterative optimization. Given a strategy request, the system first
identifies the business scenario, recommendation stage, optimiza-
tion objective, guardrail constraints, and deployment requirements,
and converts them into a structured query. It then retrieves rele-
vant knowledge from the pre-organized historical strategy experi-
ence tree through Tree-RAG and generates an executable candidate
strategy, including parameter configurations, supporting evidence,
expected gains, and potential risks. Before deployment, rule-based
validators check formatting, numerical ranges, parameter types,
and configuration consistency, while agent and human reviewers as-
sess safety, rationality, and engineering feasibility. Once approved,
the strategy is launched on the online A/B testing platform. During
experimentation, A/B Agent periodically collects core and guardrail
metrics, updates the experiment tree with new strategy variants and
outcomes, and compares the current node with its parent and sib-
ling nodes to identify effective changes in strategies and parameters.
It then performs local optimization around promising configura-
tions or retrieves alternative mechanisms when the current branch
saturates or violates guardrail constraints. The updated strategy is
redeployed for subsequent A/B rounds, while validated outcomes
are distilled back into the strategy experience tree to support con-
tinuous knowledge accumulation and strategy self-evolution.
5 Experiments
5.1 Experimental Setup
Dataset.An industrial strategy generation benchmark is constructed
from 310 historical recommendation strategies across three rep-
resentative scenarios in Kuaishou E-commerce. As summarized
in Table 1, the benchmark covers diverse deployment stages, in-
cluding retrieval, ranking, blending, and other recommendation
components. Its broad coverage across scenarios and pipeline stagesenables a comprehensive evaluation of strategy understanding and
generation in heterogeneous industrial settings.
Baselines.A/B Agent is compared with two groups of baselines.
The first group includes general-purpose LLMs: GLM-5.1, DeepSeek-
V4-Pro, Kimi-K2.6, GPT-5.5, and Claude-Sonnet-4.6, which directly
generate strategies from the given scenario and task description.
The second group consists of representative retrieval-augmented
generation methods, including Vanilla RAG, GAR [ 20], HyDE [ 7],
CRAG [ 37], and Graph RAG [ 5], which retrieve relevant historical
experience before generating the final strategy. For a controlled
comparison, all RAG baselines and A/B Agent use GLM-5.1 as the
shared foundation model, ensuring that performance differences
primarily stem from their knowledge organization, retrieval, and
strategy-generation mechanisms.
Evaluation Protocol.For each case, a recommendation strategy
is generated according to the target scenario, optimization objective,
and available context. GPT-5.5 serves as the automatic evaluator and
assesses each strategy along two dimensions:Correctness, covering
technical feasibility, scenario relevance, and logical consistency;
andNovelty, measuring the ability to provide meaningful insights
beyond direct reuse of historical experience. AnOverallscore is
also reported to reflect holistic strategy quality.
5.2 Main Results
Table 2 presents the main results across three e-commerce scenarios.
A/B Agent achieves the best performance in all evaluation dimen-
sions, demonstrating consistent effectiveness across heterogeneous
recommendation tasks. Among the standalone LLMs, GPT-5.5 ob-
tains the highest correctness, while Claude-Sonnet-4.6 performs
better in novelty and achieves the strongest average overall score
of 7.151, indicating a trade-off between technical reliability and
exploratory strategy generation. The evaluated RAG agents do
not consistently outperform standalone LLMs. HyDE achieves the
highest average overall score among the RAG baselines at 5.728,
followed by Vanilla RAG at 5.692. Compared with the strongest
RAG method in each scenario, A/B Agent improves the overall
score by 25.0%, 31.7%, and 23.5%, respectively. These improvements
indicate that semantic similarity alone is insufficient, since the ef-
fectiveness of historical strategies depends strongly on the target
scenario, pipeline stage, optimization objective, and deployment
constraints. By explicitly modeling these factors and evaluating
strategy transferability, A/B Agent reduces noisy retrieval and en-
ables more effective reuse of industrial experience. Overall, A/B
Agent achieves the highest average overall score of 7.244, outper-
forming Claude-Sonnet-4.6 by 1.3%. It also surpasses GPT-5.5 in
correctness by 0.4%, 2.3%, and 0.7%, while improving novelty over
Claude-Sonnet-4.6 by 4.3%, 5.9%, and 5.0%, respectively. These re-
sults demonstrate that A/B Agent generates strategies that are both
practically reliable and meaningfully differentiated through struc-
tured industrial knowledge and transferability-aware retrieval.
5.3 Strategy Evolution
Table 3 reports the online A/B testing results throughout the it-
erative strategy-tuning process. The overall trajectory follows an
exploration-and-refinement pattern, in which the agent first ex-
plores the attainable performance frontier and then incorporates

A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2: Main experimental results across three recommendation scenarios in Kuaishou E-commerce. The best result in each
column is highlighted in bold, while the second-best result is underlined .
Method ModelScenario 1 Scenario 2 Scenario 3Overall
Correct. Novelty Overall Correct. Novelty Overall Correct. Novelty Overall
LLMGLM-5.1 6.270 5.130 5.909 6.217 5.130 5.879 6.204 5.107 5.845 5.879
DeepSeek-V4-Pro 6.157 5.626 6.030 6.109 5.663 6.026 5.913 5.757 5.919 5.992
Kimi-K2.6 6.713 5.730 6.477 6.511 5.533 6.276 6.583 5.641 6.377 6.384
GPT-5.5 7.322 5.896 6.880 7.207 5.804 6.789 7.184 5.806 6.769 6.816
Claude-Sonnet-4.6 7.252 6.887 7.200 7.163 6.804 7.147 7.126 6.854 7.101 7.151
RAG AgentVanilla RAG 5.974 5.809 5.766 5.728 5.565 5.533 5.961 5.728 5.751 5.692
GAR 5.922 5.774 5.712 5.630 5.543 5.483 5.990 5.709 5.771 5.664
HyDE 6.035 5.826 5.805 5.696 5.587 5.524 6.039 5.777 5.823 5.728
CRAG 5.887 5.826 5.687 5.598 5.576 5.450 6.019 5.718 5.799 5.654
Graph RAG 5.913 5.643 5.700 5.565 5.565 5.422 5.922 5.573 5.707 5.620
A/B Agent Ours 7.348 7.183 7.255 7.370 7.207 7.288 7.233 7.194 7.193 7.244
Table 3: Online A/B testing results throughout the iterative strategy-tuning process.
StrategyCore Metric Important Metrics Guardrail Metrics
GMV GPM OPM Clicks Orders Watch Time Live GMV Plat. GMV Plat. Orders
Expert +2.679% +1.736% +1.492% +0.872% +2.250% -0.093% -0.636% +0.399% +0.026%
Strategy 1 +1.123% +0.545% +1.001% +0.557% +1.543% -0.067% -1.245% -0.355% +0.048%
Strategy 2 +2.984% +2.260% +0.887% +1.167% +1.727% -0.033% -1.114% -0.066% -0.114%
Strategy 3 +3.299% +3.169% +2.320% +0.451% +2.526% +0.085% -0.067% +0.683% +0.191%
Strategy 4 +3.253% +2.742% +1.633% +0.792% +2.194% +0.022% +0.443% +0.308% +0.036%
Strategy 5 +4.829% +4.677% +0.841% +0.370% +1.053% +0.078% +0.812% +0.188% +0.288%
Δ(Strategy 5−Expert) +2.150% +2.941%-0.651% -0.502% -1.197%+0.171% +1.448%-0.211%+0.262%
Note:Dark and light red backgrounds indicate the best and second-best results for the core and important metrics, respectively. Red and green backgrounds indicate positive and
negative guardrail changes, respectively. The final row reports the absolute difference between Strategy 5 and the expert-designed strategy.
online feedback to improve guardrail safety. Strategy 1 achieves
moderate gains across the core and important metrics, but degrades
Watch Time, Live GMV, and Platform GMV, indicating an unsat-
isfactory balance between local business gains and platform-level
stability. Strategy 2 further increases GMV to +2.984%and achieves
the highest click improvement of +1.167%. However, all guardrail
metrics remain negative, suggesting that aggressive optimization
of short-term engagement and conversion may adversely affect
the broader recommendation ecosystem. Strategy 3 continues the
exploration and obtains strong improvements in GMV, GPM, OPM,
and orders, reaching +3.299%,+3.169%,+2.320%, and+2.526%, re-
spectively. Most guardrail metrics also recover, although Live GMV
remains slightly negative at −0.067%. After incorporating these ob-
servations, Strategy 4 shifts toward constrained refinement, retain-
ing strong business gains while achieving positive changes across
all guardrail metrics. Strategy 5 further strengthens the primary
objectives, producing the largest GMV and GPM gains of +4.829%
and+4.677%, respectively, while maintaining positive guardrail
performance. Compared with the expert-designed strategy, Strat-
egy 5 improves GMV and GPM by2 .150%and2.941%points and
improves Watch Time, Live GMV, and Platform Orders by0 .171%,
1.448%, and0.262%points, respectively. Although its OPM, clicks,orders, and Platform GMV are lower, the final strategy achieves
substantially stronger primary business gains while avoiding the
expert strategy’s negative effects on Watch Time and Live GMV.
5.4 Balancing Core Gains and Guardrail Safety
Figure 4 illustrates how A/B Agent progressively balances core-
metric gains and guardrail safety through iterative online A/B
feedback. Strategy 1 achieves a moderate GMV improvement of
+1.123%, but decreases the four guardrail metrics by0 .405%on
average, indicating that the initial configuration still introduces
considerable ecosystem-level risks. After adjusting the objective
weights and decision thresholds, Strategy 2 increases the GMV gain
to+2.984%and improves the average guardrail change to −0.332%.
Although it already outperforms the expert-designed strategy in
GMV, the guardrail performance remains negative. Strategy 3 fur-
ther strengthens the optimization of the core objectives, increasing
GMV to+3.299%, while the average guardrail change becomes pos-
itive for the first time at +0.223%. Strategy 4 achieves a comparable
GMV gain of+3.253%and maintains a positive guardrail average of
+0.202%. The slight fluctuation between Strategies 3 and 4 reflects
an intermediate stability-validation stage rather than a substantial
performance regression. The final Strategy 5 reaches the strongest

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhuohang Jiang et al.
1
2
3
4
5Core GMV Lift (%)
−1.0−0.50.00.5
Live GMV Guardrail (%)S1S2S3S4S5
Guardrail = 0  (non-negative boundary)Pareto-optimal surface
tuning trajectory
Strategy Evolution / Tuning Step
Figure 4: Pareto-oriented tuning trajectory of A/B Agent
across core gains and guardrail safety.
Table 4: Ablation study on key designs of A/B Agent.
Setting Correctness Novelty Overall Decline
A/B Agent 7.348 7.183 7.255–
A. Knowledge Base
w/o Knowledge Base 7.104 7.087 7.017 0.238↓
B. Retrieval Architecture
w/o Strategy-Tree RAG (Flat RAG) 7.296 7.122 7.240 0.015↓
w/o Multi-Path Retrieval 7.200 7.157 7.144 0.111↓
C. Retrieval Enhancement
w/o Objective-Aware Boosting 7.252 7.113 7.185 0.070↓
w/o Scenario-Aware Boosting 7.217 7.070 7.133 0.122↓
w/o Re-ranking Module 7.174 7.070 7.088 0.167↓
overall operating point, improving GMV and GPM by +4.829%and
+4.677%, respectively. Meanwhile, all four guardrail metrics become
positive, with an average improvement of approximately +0.342%.
Compared with the expert-designed strategy, Strategy 5 further
improves GMV and GPM by2 .150%and2.941%points, respectively,
while also delivering clear improvements in Watch Time, Live GMV,
and Platform Orders. Overall, the trajectory demonstrates that A/B
Agent can continuously translate online feedback into effective
parameter updates, gradually moving from a region with limited
gains and guardrail risks toward a Pareto-optimal solution that
jointly improves business effectiveness and platform safety.
5.5 Ablation Study
Table 4 reports the ablation results in scenario 1. Removing any
component reduces performance, confirming that the proposed
modules provide complementary benefits. The knowledge base is
the most important component: removing it decreases the overall
score by 3.3% and correctness by 3.3%, highlighting the value ofTable 5: Online A/B testing results of A/B Agent.
Method GMV GPM OPM CVR Clicks Orders
A/B Agent +4.829% +4.677% +0.841% +0.578% +0.370% +1.053%
historical industrial experience for generating feasible strategies.
The re-ranking module is also critical, with its removal causing a
2.3% decline in overall performance. The retrieval enhancements
further improve strategy transfer. Removing scenario-aware boost-
ing and multi-path retrieval reduces the overall score by 1.7% and
1.5%, respectively, while removing objective-aware boosting re-
sults in a 1.0% decline. These results indicate that effective strategy
generation requires both broad evidence retrieval and explicit align-
ment with the target scenario and optimization objective. Replacing
Strategy-Tree RAG with Flat RAG produces a smaller but consistent
decrease of 0.2%. Although the margin is modest, it confirms that
hierarchical knowledge organization enables more precise retrieval
than treating historical strategies as an unstructured document
collection. Overall, the complete A/B Agent achieves the best per-
formance across all metrics, validating the combined effectiveness
of its knowledge base, retrieval architecture, boosting mechanisms,
and re-ranking module.
5.6 Online A/B Test
A/B Agent has been deployed in the e-commerce short-video sce-
nario on the Kuaishou platform. We conducted an online A/B test
in a real-world production environment to evaluate its practical
effectiveness. As shown in Table 5, the strategy generated by A/B
Agent improves GMV by 4.829%, together with gains of 4.677% in
GPM, 0.841% in OPM, and 0.578% in CVR. It also increases clicks and
orders by 0.370% and 1.053%, respectively, demonstrating that A/B
Agent can translate historical strategy knowledge into measurable
online business improvements.
6 Conclusion
A/B Agent is presented as a self-evolving agentic framework for
industrial recommendation strategy iteration. It consists of three
tightly coupled components:Historical Strategy Knowledge Organi-
zation,Autonomous Target-Aware Strategy Generation, andExperiment-
Guided Strategy Self-Evolution. A/B Agent organizes fragmented
historical records into a hierarchical experience tree, retrieves trans-
ferable knowledge through multi-path Tree-RAG, and continuously
refines strategies and parameters using online A/B feedback. Exten-
sive offline experiments demonstrate consistent improvements over
general-purpose LLMs and conventional RAG-based agents across
multiple recommendation scenarios and pipeline stages. Real-world
deployment at Kuaishou further yields a 4.829% increase in GMV,
accompanied by consistent gains in GPM, OPM, CVR, clicks, and or-
ders. These results highlight the effectiveness and efficiency of A/B
Agent for industrial strategy optimization. Future work will focus
on stronger requirement alignment, more accurate data and metric
attribution, and automated engineering-feasibility verification to
support safer and more reliable strategy evolution.

A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024.
Self-rag: Learning to retrieve, generate, and critique through self-reflection. In
International conference on learning representations, Vol. 2024. 9112–9141.
[2]Alex Deng and Xiaolin Shi. 2016. Data-driven metric development for online
controlled experiments: Seven lessons learned. InProceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining. 77–
86.
[3]Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang
Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative
recommender and iterative preference alignment.arXiv preprint arXiv:2502.18965
(2025).
[4]Pavel Dmitriev, Somit Gupta, Dong Woo Kim, and Garnet Vaz. 2017. A dirty dozen:
Twelve common metric interpretation pitfalls in online controlled experiments.
InProceedings of the 23rd ACM SIGKDD international conference on knowledge
discovery and data mining. 1427–1436.
[5]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[6]Aleksander Fabijan, Pavel Dmitriev, Helena Holmström Olsson, and Jan Bosch.
2018. Effective online controlled experiment analysis at large scale. In2018 44th
Euromicro Conference on Software Engineering and Advanced Applications (SEAA).
IEEE, 64–67.
[7]Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-
shot dense retrieval without relevance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
1762–1777.
[8]Somit Gupta, Ronny Kohavi, Diane Tang, Ya Xu, Reid Andersen, Eytan Bakshy,
Niall Cardin, Sumita Chandran, Nanyu Chen, Dominic Coey, et al .2019. Top
challenges from the first practical online controlled experiments summit.ACM
SIGKDD Explorations Newsletter21, 1 (2019), 20–35.
[9]Jianhua Han, Yong Yu, Feng Liu, Ruiming Tang, and Yuzhou Zhang. 2019. Op-
timizing ranking algorithm in recommender system via deep reinforcement
learning. In2019 international conference on artificial intelligence and advanced
manufacturing (AIAM). IEEE, 22–26.
[10] Chao Hao, Shuai Wang, and Kaiwen Zhou. 2025. Uncertainty-aware gui agent:
Adaptive perception through component recommendation and human-in-the-
loop refinement.arXiv preprint arXiv:2508.04025(2025).
[11] Zheng Hu, Yuxin Chen, Yongsen Pan, Xu Yuan, Yuting Yin, Daoyuan Wang,
Boyang Xia, Zefei Luo, Hongyang Wang, Songhao Ni, et al .2026. Stop Treating
Collisions Equally: Qualification-Aware Semantic ID Learning for Recommenda-
tion at Industrial Scale.arXiv preprint arXiv:2603.00632(2026).
[12] Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili
Ma, Hongzhi Chen, and James Cheng. 2025. Retrieval-augmented generation
with hierarchical knowledge.arXiv preprint arXiv:2503.10150(2025).
[13] Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. 2025.
Recommender ai agent: Integrating large language models for interactive recom-
mendations.ACM Transactions on Information Systems43, 4 (2025), 1–33.
[14] Aryan Jadon and Avinash Patil. 2024. A comprehensive survey of evaluation tech-
niques for recommendation systems. InInternational Conference on Computation
of Artificial Intelligence & Machine Learning. Springer, 281–304.
[15] Olivier Jeunen, Shubham Baweja, Neeti Pokharna, and Aleksei Ustimenko. 2024.
Powerful A/B-testing metrics and where to find them. InProceedings of the 18th
ACM Conference on Recommender Systems. 816–818.
[16] Zhuohang Jiang, Yuxin Chen, Shijie Wang, Haohao Qu, Zhou Jindong, Wenqi
Fan, Li Qing, Dongxu Liang, and Jun Wang. 2026. Atomic Intent Reasoning:
Bringing LLM Semantics to Industrial Cross-Domain Recommendations.arXiv
preprint arXiv:2606.10357(2026).
[17] Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q Chen, Xu Yuan, Ye Jia,
Jiancheng Tu, Chen Li, Peter HF Ng, and Qing Li. 2025. Hibench: Benchmarking
llms capability on hierarchical structure reasoning. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5505–5515.
[18] Zhuohang Jiang, Pangjing Wu, Xu Yuan, Wenqi Fan, and Qing Li. 2025. QA-
Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual
Question Answering.arXiv preprint arXiv:2508.05197(2025).
[19] Zhuohang Jiang, Xu Yuan, Haohao Qu, Shanru Lin, Kanglong Liu, Wenqi Fan,
and Li Qing. 2026. SUPERGLASSES: Benchmarking Vision Language Models as
Intelligent Agents for AI Smart Glasses. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 2165–2175.
[20] Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei
Han, and Weizhu Chen. 2021. Generation-augmented retrieval for open-domain
question answering. InProceedings of the 59th Annual Meeting of the Associa-
tion for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers). 4089–4100.
[21] Mohammad Hossein Namaki, Avrilia Floratou, Fotis Psallidas, Subru Krishnan,
Ashvin Agrawal, Yinghui Wu, Yiwen Zhu, and Markus Weimer. 2020. Vamsa:
Automated provenance tracking in data science scripts. InProceedings of the 26thACM SIGKDD international conference on knowledge discovery & data mining.
1542–1551.
[22] Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi
Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S Yu, et al .2025. A survey of
webagents: Towards next-generation ai agents for web automation with large
foundation models. InProceedings of the 31st ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining V. 2. 6140–6150.
[23] Yongsen Pan, Yuxin Chen, Zheng Hu, Xu Yuan, Daoyuan Wang, Yuting Yin, Song-
hao Ni, Hongyang Wang, Jun Wang, Fuji Ren, et al .2026. Beyond Static Collision
Handling: Adaptive Semantic ID Learning for Multimodal Recommendation at
Industrial Scale.arXiv preprint arXiv:2604.23522(2026).
[24] Hong Peng, Xiaoliang Jin, Qiao Huang, and Supeng Liu. 2025. E-commerce Intelli-
gent Recommendation Optimization and Personalized Marketing Strategy Based
on Big Model. InProceedings of the 2025 International Conference on Information
Economy, Data Modeling and Cloud Computing. 313–317.
[25] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang
Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong
sequential behavior data for click-through rate prediction. InProceedings of the
29th ACM International Conference on Information & Knowledge Management.
2685–2692.
[26] Federico Quin, Danny Weyns, Matthias Galster, and Camila Costa Silva. 2024.
A/B testing: A systematic literature review.Journal of Systems and Software211
(2024), 112011.
[27] Christopher Ré, Feng Niu, Pallavi Gudipati, and Charles Srisuwananukorn. 2019.
Overton: A data system for monitoring and improving machine-learned products.
arXiv preprint arXiv:1909.05372(2019).
[28] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher Manning. 2024. Raptor: Recursive abstractive processing for tree-
organized retrieval. InInternational Conference on Learning Representations,
Vol. 2024. 32628–32649.
[29] Marius Schlegel and Kai-Uwe Sattler. 2023. Management of machine learning
lifecycle artifacts: A survey.ACM SIGMOD Record51, 4 (2023), 18–35.
[30] Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang
Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, et al .2021. One model to
serve all: Star topology adaptive recommender for multi-domain ctr prediction. In
Proceedings of the 30th ACM International Conference on Information & Knowledge
Management. 4104–4113.
[31] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and
Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning.
Advances in neural information processing systems36 (2023), 8634–8652.
[32] Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang, and Xiangmin Xu. 2025.
Treerag: Unleashing the power of hierarchical storage for enhanced knowledge
retrieval in long documents. InFindings of the Association for Computational
Linguistics: ACL 2025. 356–371.
[33] Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah
Cho, Xing Fan, Yanbin Lu, Xiaojiang Huang, and Yingzhen Yang. 2024. Recmind:
Large language model powered agent for recommendation. InFindings of the
Association for Computational Linguistics: NAACL 2024. 4351–4364.
[34] Yuhao Wang, Xiangyu Zhao, Bo Chen, Qidong Liu, Huifeng Guo, Huanshuo Liu,
Yichao Wang, Rui Zhang, and Ruiming Tang. 2023. PLATE: A prompt-enhanced
paradigm for multi-scenario recommendations. InProceedings of the 46th In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval. 1498–1507.
[35] Pangjing Wu, Peter Q Chen, Xiaodong Li, Wenqi Fan, and Qing Li. 2026. Datamart-
Agent: LLM-Driven Game-Theoretic Agent for Data Marketplace Modeling. In
Findings of the Association for Computational Linguistics: ACL 2026. 32509–32531.
[36] Ya Xu, Weitao Duan, and Shaochen Huang. 2018. SQR: Balancing speed, quality
and risk in online experiments. InProceedings of the 24th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery & Data Mining. 895–904.
[37] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective retrieval
augmented generation. (2024).
[38] John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik
Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable
automated software engineering.Advances in Neural Information Processing
Systems37 (2024), 50528–50652.
[39] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.
arXiv preprint arXiv:2210.03629(2022).
[40] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2024. Making retrieval-
augmented language models robust to irrelevant context. InInternational Confer-
ence on Learning Representations, Vol. 2024. 29862–29883.
[41] Xu Yuan, Liangbo Ning, Wenqi Fan, and Qing Li. 2025. mKG-RAG: Multimodal
Knowledge Graph-Enhanced RAG for Visual Question Answering.arXiv preprint
arXiv:2508.05318(2025).
[42] Yuanliang Zhang, Xiaofeng Wang, Jinxin Hu, Ke Gao, Chenyi Lei, and Fei Fang.
2022. Scenario-adaptive and self-supervised model for multi-scenario personal-
ized recommendation. InProceedings of the 31st ACM International Conference on
Information & Knowledge Management. 3674–3683.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhuohang Jiang et al.
0 10 20 30 40 50
Percentage (%)Solution mismatch
Data / label / attribution
Engineering / serving
Evaluation / specification43.1%
29.2%
19.4%
8.3%Error Distribution of A/B Agent
Figure 5: Distribution of error categories for A/B Agent.
Table 6: Performance comparison across different recommen-
dation stages.
Category Model Retrieval Ranking Blending Others
LLMDeepSeek-V4-Pro 5.979 5.954 5.910 6.020
Kimi-K2.6 6.300 6.272 6.590 6.455
GPT-5.5 6.803 6.789 6.880 6.829
GLM-5.1 5.723 5.914 5.880 5.920
Claude-Sonnet-4.6 7.151 7.053 7.200 7.194
RAG AgentVanilla RAG 5.608 5.612 5.550 5.769
GAR 5.580 5.632 5.800 5.701
HyDE 5.589 5.632 5.690 5.827
CRAG 5.693 5.520 5.440 5.715
GraphRAG 5.597 5.607 5.440 5.645
A/B Agent Ours 7.269 7.096 7.280 7.302
A Further Analysis
A.1 Stage Analysis
Table 6 compares different methods across four recommendation
stages. A/B Agent achieves the best performance in retrieval, rank-
ing, blending, and other stages, with scores of 7.269, 7.096, 7.280, and7.302, respectively. Compared with the strongest baseline, Claude-
Sonnet-4.6, it improves the scores by 0.118, 0.043, 0.080, and 0.108
points, demonstrating robust effectiveness across heterogeneous
components of the recommendation pipeline. Among standalone
LLMs, Claude-Sonnet-4.6 performs best, followed by GPT-5.5, while
the RAG-based methods generally lag behind. This indicates that
directly retrieving historical records is insufficient when the knowl-
edge is not explicitly organized by stage, scenario, and optimization
objective. Although ranking remains the most challenging stage
due to its multi-objective and tightly coupled design, A/B Agent
still outperforms all baselines. Overall, the results validate the effec-
tiveness of its stage-aware knowledge organization and retrieval
mechanism.
A.2 Error Analysis
Figure 5 presents the distribution of failure cases produced by A/B
Agent. The dominant error category issolution mismatch, account-
ing for 43.1% of all failures. These errors arise when a generated
strategy is generally reasonable but does not fully align with the
target scenario, optimization objective, pipeline stage, or deploy-
ment constraints. This result indicates that fine-grained strategy
alignment remains the primary challenge for A/B Agent.
Data-, label-, and attribution-related errors constitute the second-
largest category at 29.2%. Such errors typically involve inaccurate
assumptions about feature definitions, training labels, metric attri-
bution, or the availability of historical data. Engineering and serving
issues account for 19.4%, mainly reflecting incompatibilities with
existing pipelines, unavailable features, or excessive deployment
overhead. Evaluation and specification errors represent only 8.3%
of the failures, suggesting that the task specification and evaluation
process are comparatively reliable.
Overall, the remaining errors are primarily associated with scenario-
specific understanding and practical constraint alignment rather
than general strategy generation. Further improvements should
therefore focus on objective-aware knowledge retrieval, explicit
verification of data dependencies, and constraint-aware evaluation
before online experimentation.