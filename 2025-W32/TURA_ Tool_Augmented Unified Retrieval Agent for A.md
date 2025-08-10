# TURA: Tool-Augmented Unified Retrieval Agent for AI Search

**Authors**: Zhejun Zhao, Yuehu Dong, Alley Liu, Lixue Zheng, Pingsheng Liu, Dongdong Shen, Long Xia, Jiashu Zhao, Dawei Yin

**Published**: 2025-08-06 16:24:17

**PDF URL**: [http://arxiv.org/pdf/2508.04604v1](http://arxiv.org/pdf/2508.04604v1)

## Abstract
The advent of Large Language Models (LLMs) is transforming search engines
into conversational AI search products, primarily using Retrieval-Augmented
Generation (RAG) on web corpora. However, this paradigm has significant
industrial limitations. Traditional RAG approaches struggle with real-time
needs and structured queries that require accessing dynamically generated
content like ticket availability or inventory. Limited to indexing static
pages, search engines cannot perform the interactive queries needed for such
time-sensitive data. Academic research has focused on optimizing RAG for static
content, overlooking complex intents and the need for dynamic sources like
databases and real-time APIs. To bridge this gap, we introduce TURA
(Tool-Augmented Unified Retrieval Agent for AI Search), a novel three-stage
framework that combines RAG with agentic tool-use to access both static content
and dynamic, real-time information. TURA has three key components: an
Intent-Aware Retrieval module to decompose queries and retrieve information
sources encapsulated as Model Context Protocol (MCP) Servers, a DAG-based Task
Planner that models task dependencies as a Directed Acyclic Graph (DAG) for
optimal parallel execution, and a lightweight Distilled Agent Executor for
efficient tool calling. TURA is the first architecture to systematically bridge
the gap between static RAG and dynamic information sources for a world-class AI
search product. Serving tens of millions of users, it leverages an agentic
framework to deliver robust, real-time answers while meeting the low-latency
demands of a large-scale industrial system.

## Full Text


<!-- PDF content starts -->

TURA: Tool-Augmented Unified Retrieval Agent for AI Search
Zhejun Zhao∗
Baidu Inc.
Beijing, China
zhaozhejun@baidu.comYuehu Dong∗
Baidu Inc.
Beijing, China
dongyuehu@baidu.comAlley Liu∗
Baidu Inc.
Beijing, China
liuli44@baidu.com
Lixue Zheng
University of Science and Technology
of China
Hefei, China
zhenglx23@mail.ustc.edu.cnPingsheng Liu
Baidu Inc.
Beijing, China
liupingsheng@baidu.comDongdong Shen
Baidu Inc.
Beijing, China
shendongdong@baidu.com
Long Xia
Baidu Inc.
Beijing, China
long.phil.xia@gmail.comJiashu Zhao
Wilfrid Laurier University
Beijing, China
jzhao@wlu.caDawei Yin†
Baidu Inc.
Beijing, China
yindawei@acm.org
Abstract
The advent of Large Language Models (LLMs) is transforming
search engines into conversational AI search products, primar-
ily using Retrieval-Augmented Generation (RAG) on web corpora.
However, this paradigm has significant industrial limitations. Tradi-
tional RAG approaches struggle with real-time needs and structured
queries that require accessing dynamically generated content like
ticket availability or inventory. Limited to indexing static pages,
search engines cannot perform the interactive queries needed for
such time-sensitive data. Academic research has focused on opti-
mizing RAG for static content, overlooking complex intents and
the need for dynamic sources like databases and real-time APIs.
To bridge this gap, we introduce TURA (Tool-Augmented Unified
Retrieval Agent for AI Search), a novel three-stage framework that
combines RAG with agentic tool-use to access both static content
and dynamic, real-time information. TURA has three key compo-
nents: an Intent-Aware Retrieval module to decompose queries
and retrieve information sources encapsulated as Model Context
Protocol (MCP) Servers, a DAG-based Task Planner that models
task dependencies as a Directed Acyclic Graph (DAG) for optimal
parallel execution, and a lightweight Distilled Agent Executor for
efficient tool calling. TURA is the first architecture to systemati-
cally bridge the gap between static RAG and dynamic information
sources for a world-class AI search product. Serving tens of mil-
lions of users, it leverages an agentic framework to deliver robust,
∗Co-first authors with equal contributions.
†Corresponding author
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXreal-time answers while meeting the low-latency demands of a
large-scale industrial system.
CCS Concepts
•Information systems →Question answering .
Keywords
Conversational Search; Large Language Models; Information Re-
trieval
ACM Reference Format:
Zhejun Zhao, Yuehu Dong, Alley Liu, Lixue Zheng, Pingsheng Liu, Dong-
dong Shen, Long Xia, Jiashu Zhao, and Dawei Yin. 2018. TURA: Tool-
Augmented Unified Retrieval Agent for AI Search. In Proceedings of Make
sure to enter the correct conference title from your rights confirmation email
(Conference acronym ’XX). ACM, New York, NY, USA, 10 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Traditional web search engines, built on foundational algorithms
like PageRank [ 22] and the early Google architecture [ 4], have long
relied on the “ten blue links” paradigm. While these systems excel
at retrieving and ranking information from vast corpora of un-
structured web pages, they have historically struggled with queries
demanding structured, real-time, or transactional information, such
as flight availability or weather forecasts. To address these limi-
tations, search engines introduced manually curated components
like Google’s OneBox [ 8] or Baidu’s Aladdin platform [ 40], which
present information from specific, trusted data sources in dedicated
formats. However, this approach proves fundamentally limited by
its reliance on brittle, hand-crafted integrations that are difficult to
scale.
The advent of Large Language Models (LLMs) has catalyzed
a fundamental paradigm shift in information access [ 6], trans-
forming search from keyword-based retrieval to conversational,
answer-centric systems. The dominant architecture in this new era
is Retrieval-Augmented Generation (RAG) [ 10,18], which groundsarXiv:2508.04604v1  [cs.CL]  6 Aug 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhejun Zhao et al.
Bai 
Please book me a high-speed train ticket from 
Beijing to Shanghai for a business trip on August 
2, 2025.
Thinking Finished
As of July 31, 2025 , high-speed train schedules 
for August 1 from Beijing to Shanghai are 
currently available. Note that August 2 schedule 
may be similar, but please confirm during 
actual booking.
Below are August 1’s reference train options 
and fares. For optimal efficiency, we 
recommend prioritizing trains with shorter 
travel times:
(b) AI Search with web results
Ask me a question or Hold to talk1.Train G1 (Fastest Option)
•Beijing South 07:00 → Shanghai 11:29 
(4h 29m)
•Fares: Second Class: ¥669(from), 
Tickets available, First Class/Business 
Class: Sold out.
2. Train G3
• Beijing South 08:00 → Shanghai 12:32 
(5h 38m)
•Fares: From ¥673, Second Class/First 
Class/Business Class: Sold out.
3.Train G103
•Route: Beijing South 08:00 → 
Shanghai 12:32 (4h 32m)
•Fares: Second Class: ¥576(from), 10 
seats left. First Class: Tickets available.
Please book me a high-speed train ticket from 
Beijing to Shanghai for a business trip on August 
2, 2025.
Thinking Finished
•Call a Tool: Ctrip Train Tickets
The high-speed train schedule from Beijing to 
Shanghai on August 2, 2025, is as follows, 
including major train numbers, duration, ticket 
prices, and stops:
(a) TURA using Ctrip MCP
Ask me a question or Hold to talkKey high-speed train services
1.Train G1
•Departure/Arrival: Beijing South 07:00 
→ Shanghai 11:29
•Duration: 4h 29m
•Fare: Second Class: ¥669 (from), 
Tickets available, Business Class: 9 
seats left.
2.Train G103
•Departure/Arrival: Beijing South 06:20 
→ Shanghai Hongqiao 11:58
•Duration: 5h 38m
•Fare: Second Class: ¥598 (from), 
Tickets available.
3.Trian G105
•Departure/Arrival: Beijing South 07:17 
→ Shanghai Hongqiao 13:03
•Duration: 5h 46m
•Fare: Second Class: ¥576 (from), 
Tickets available
Bai 
Figure 1: Demonstration of TURA’s agentic capabilities.
Given a query on July 31, 2025: (a) TURA autonomously
utilizes a tool by calling Ctrip’s API to successfully look up
ticket information. (b) This contrasts with a traditional RAG-
based AI search, which can only retrieve information from
static webpages and is incapable of performing the required
action.
LLM responses in factual knowledge retrieved from web-scale cor-
pora. Commercial deployments like Perplexity AI, ChatGPT’s web
search integration, and Google’s AI Overviews demonstrate the
practical viability of this approach [34].
However, this RAG-centric paradigm inherits a critical limitation
from its search engine ancestry: it primarily operates on a static
snapshot of the web. Current RAG systems, optimized for retriev-
ing from pre-indexed web documents [ 13,35], are fundamentally
incapable of accessing dynamic, real-time information that is not
present on a static webpage but must be generated via interaction.
For instance, they cannot query a flight booking system for ticket
availability on a specific future date or check real-time inventory
from an e-commerce database, as this information is only accessible
through interactive queries to APIs or databases. As illustrated in
Figure 1, a standard RAG system fails to answer a time-sensitive
query because the necessary information must be generated dy-
namically, a capability it inherently lacks. This inability to interact
with live services renders them ill-equipped for a significant class
of user needs that go beyond static information retrieval.
To bridge this critical gap between retrieving from static corpora
and interacting with dynamic data sources, we propose TURA
(Tool-Augmented Unified Retrieval Agent for AI Search). TURA is
a novel three-stage agentic framework that enhances LLMs with
the ability to use external tools, moving beyond passive document
retrieval to active, real-time data acquisition. The system’s design
follows the ReAct framework [ 37] of interleaving reasoning and
acting, and incorporates advanced planning capabilities inspired
by adaptive decomposition methods [ 23]. For efficient execution,TURA implements DAG-based task decomposition for parallel tool
calls [ 16], addressing latency challenges while maintaining the com-
plex reasoning capabilities demonstrated in large-scale API integra-
tion frameworks [ 25]. TURA’s architecture leverages standardized
tool interfaces following the Model Context Protocol (MCP) specifi-
cation [ 1], enabling seamless integration with diverse information
sources through a unified framework.
TURA’s three-stage architecture comprises the following compo-
nents: (1) Intent-Aware Tool Retrieval: This module decomposes
complex user queries into atomic sub-intents and performs dense
retrieval over a semantically-indexed catalogue of available tools,
encapsulating both static document collections and dynamic APIs,
to identify relevant candidates for each sub-intent. (2) DAG-based
Task Planning: This module constructs optimal and parallelizable
execution plans by modeling sub-tasks and their data dependencies
as a Directed Acyclic Graph (DAG), enabling the orchestration of
complex, multi-hop reasoning chains across multiple tool calls. (3)
Distilled Agent Executor: To address critical inference latency
barriers in production settings, we developed a lightweight yet
highly capable agent fine-tuned on curated expert trajectories using
a novel mixed-rationale distillation technique, achieving compara-
ble fidelity to proprietary LLMs at a fraction of the computational
cost and latency.
Since May 2025, TURA has been fully deployed, successfully
serving tens of millions of users. TURA demonstrably expands the
capabilities of AI search, providing accurate, real-time answers for
a broad spectrum of queries, particularly those involving dynamic,
transactional, or non-web data, that were previously intractable for
conventional RAG-based systems.
Our contributions are threefold:
•We propose TURA, a novel and systematic agentic architec-
ture that effectively integrates diverse, dynamic and non-web
information sources into AI search via tool calling, directly
addressing the static-world limitations of traditional RAG
systems.
•We introduce a cohesive framework of synergistic techniques
including intent-aware tool retrieval, DAG-based planning,
and a latency-optimized distilled agent that collectively solve
key industrial challenges of tool selection, planning, and
efficient execution.
•We present the first large-scale industrial validation of a
tool-augmented agentic search system, demonstrating its
viability and effectiveness in a world-class AI search product.
This provides a production-proven blueprint for the next
generation of AI search.
2 Related Work
2.1 Retrieval-Augmented Generation
To mitigate LLM hallucinations and improve factual accuracy [ 9,10],
RAG systems ground the generation process in external knowledge.
These systems typically employ a "retrieve-then-generate" para-
digm, with a significant body of research focused on optimizing
both retrieval [ 17,21,29] and generation [ 20,30]. Recent advances
in RAG have progressed from static retrieve-then-generate pipelines
to more dynamic and adaptive frameworks [ 9,31]. For example,
Self-RAG introduces "reflection tokens" that enable the model to

TURA: Tool-Augmented Unified Retrieval Agent for AI Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
decide on-demand whether retrieval is necessary, avoiding the
inefficiency of indiscriminate retrieval for every query [ 2]. In a
similar vein, Active Retrieval Augmented Generation proposes an
iterative retrieval process that occurs throughout generation, al-
lowing the model to gather information as needed. Meanwhile,
R2AG focuses on bridging the semantic gap between the retriever
and the generator by using more nuanced retrieval features [ 38].
Despite these innovations, the practical application of RAG in com-
mercial systems like Perplexity AI and Google AI Overviews has
highlighted ongoing challenges in maintaining content quality and
robust contextual understanding [ 32]. A fundamental and perva-
sive limitation of existing systems is their foundational reliance
on rigid, predefined workflows. This inherent inflexibility means
they are ill-equipped to dynamically adapt to the demands of com-
plex, multi-faceted queries, which by their very nature necessitate
a seamless and intelligent synthesis of information drawn from a
wide array of diverse sources and the orchestrated application of
various computational tools.
2.2 Tool-Augmented Agents
While RAG systems augment LLMs with textual documents, tool-
augmented agents expand their capabilities by providing them with
access to a wide array of external resources, such as APIs, web
servers, and other computational tools. Much of the research in this
area centers on a three-stage process of plan, action, and reflection
to guide the agent’s behavior [28, 33].
The foundational ReAct framework established a paradigm of
interleaving reasoning and acting, where the model generates both
thought processes and subsequent actions to interact with its en-
vironment. This synergy was shown to significantly improve task
performance and interpretability. Building on this, Toolformer
demonstrated how models could teach themselves to use tools
in a self-supervised manner [ 26]. The scale of tool integration was
dramatically expanded by ToolLLM, which enabled models to lever-
age thousands of real-world APIs through the use of depth-first
search-based decision trees [25]. Recent work has also focused on
optimizing the efficiency of tool use. LLMCompiler, for instance,
introduced a Directed Acyclic Graph (DAG)-based approach for
parallel tool calling, achieving a 3.6x speedup [ 16]. Agent Q further
improved success rates by combining Monte Carlo Tree Search
with a self-critique mechanism [24].
Despite these advances, current systems face critical limitations:
(1) their workflows are often static and cannot adapt to the complex-
ity of a given query; (2) they struggle with the semantic integration
of heterogeneous tools and information sources; and (3) they suf-
fer from coordination inefficiencies, as RAG and tool-augmented
systems typically operate in isolation. TURA is designed to address
these issues by proposing a unified architecture that dynamically
coordinates retrieval decisions, tool selection, and response genera-
tion based on the specific characteristics and intent of the user’s
query.
3 Problem Definition
We are given a user’s natural language query, 𝑞, and a collection
of𝑁heterogeneous MCP Servers, 𝑀={𝑀1,𝑀2,...,𝑀 𝑁}. These
servers function as external, specialized tools that provide access toinformation and capabilities not inherently present in a language
model.
The core problem is to answer the query 𝑞by dynamically com-
posing the functionalities of these servers. This involves selecting
the right servers and coordinating their execution to produce a final,
synthesized answer, 𝐴. We refer to any such specific composition
of server calls as an execution strategy, denoted by 𝜋.
The effectiveness of a strategy is determined by a fundamental
trade-off. We aim to maximize the quality of the final answer, 𝑄(𝐴),
which depends on the chosen strategy, while simultaneously adher-
ing to a strict latency constraint, 𝐿(𝜋). Our objective is to find the
optimal strategy, 𝜋∗, that yields the best possible answer within a
given time budget. This is formalized as the following constrained
optimization problem:
𝜋∗=arg max
𝜋𝑄(𝐴(𝜋))subject to 𝐿(𝜋)≤𝜏max (1)
where𝜏maxrepresents the maximum permissible latency. This for-
mulation encapsulates the challenge of leveraging external tools
effectively under real-world performance requirements.
4 Method
Fig. 2 illustrates the architecture of the TURA, which consists of
three key modules: an Intent-Aware MCP Server Retrieval Mod-
ule, aDAG-based Task Planner Module , and a latency-optimized
Distilled Agent Executor Module .
4.1 Intent-Aware MCP Server Retrieval
This initial stage acts as a filter, efficiently identifying a small, high-
recall set of MCP Servers from the global pool Mthat are most
likely to contribute to answering the query 𝑞. This prevents the
downstream modules from being overwhelmed with irrelevant
options.
4.1.1 LLM-based Multi-Intent Query Decomposition. User queries
are often underspecified and multi-faceted. A monolithic query may
contain multiple, logically distinct information needs. To handle
this, we employ a powerful LLM, 𝑓LLM-de , configured with a specific
prompt (see Appendix A.2) to function as a query decomposer. The
LLM is instructed to parse the raw query 𝑞and transform it into a
structured set of atomic sub-queries, SQ={𝑠𝑞1,𝑠𝑞2,...,𝑠𝑞 𝑘}.
SQ=𝑓LLM-de(𝑞) (2)
Each sub-query 𝑠𝑞𝑗∈ SQ is designed to be unambiguous and
correspond to a single semantic intent (e.g., "find the city where the
Forbidden City is located.", "get weather forecast for a given city").
This decomposition transforms an ambiguous problem into a set of
well-defined, tractable sub-problems. This strategy is inspired by
recent advances in task decomposition for complex reasoning [ 41].
4.1.2 Server-level Semantic Index Augmentation. A significant chal-
lenge in tool use is the "lexical gap" between user vernacular and for-
mal API or server descriptions. To bridge this, we perform an exten-
sive offline index augmentation process. For each server 𝑀𝑖∈M ,
we first define its holistic description 𝐷𝑖. Then, we utilize a genera-
tive LLM,𝑔LLM-gen with a specific prompt (see Appendix A.1), to
produce a large, diverse set of synthetic queries , Qsyn
𝑖, that a user

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhejun Zhao et al.
Sub-
task2LLM
Planner
Sub-
task3SubQuery and MCP Server
Sub-task4
Weather: Mostly sunny, 
30% rain on June 12.
Hotel: The Peninsula 
Beijing
Attrations: The Great Wall,
The Forbidden City...
Direction: To Hotel...Executor
ExecutorAction1Observation1
ReactAction2 ······
User Query
Sub-TasksReponseDataset
Fine-tuning
DAG-based Task Planner 
 Distilled Agent Executor 
 Router
Query: Beijing trip June 10-
15. Need hotel, 2-3 
attractions and things to do.
Weather Forecast
Attraction 
Recommendation
Retrival MCP Server
Subquery 
Hotel BookingLLM Intent 
Decomposition
MCP Server Store 
ANN Recall
Intent-Aware MCP Server Retrieval
Sub-Task1: Search Beijing 
weather
Sub-Task2: Top Attractions
Sub-Task3: Hotel Booking
Sub-Task4: Path  planner
Qwen-4B
Generate
Answer
Easy Query without planner
MCP Server Store 
Sub-
task1
Figure 2: TURA Framework Overview. The framework consists of three stages: Intent-Aware MCP Server Retrieval, DAG-based
Task Planner, and Distilled Agent Executor. Example shows processing a Beijing travel query.
might plausibly issue to access the functionalities of 𝑀𝑖.
Qsyn
𝑖=𝑔LLM-gen(𝐷𝑖,temperature =𝑇high) (3)
By employing a high sampling temperature ( 𝑇high>1.0), we en-
courage the model to explore the periphery of the server’s semantic
space, generating queries with diverse phrasing and implied intent.
This approach, where a model generates training or index data
for retrieval, has proven effective in bridging the semantic gap [ 5].
The final retrievable unit for each server becomes an augmented
document, which is a set of texts 𝐷′
𝑖={𝐷𝑖}∪Qsyn
𝑖. This enriched
representation provides a dense, multi-faceted semantic footprint
for each server.
4.1.3 Dense Vector Retrieval. We employ a dense retrieval approach
using multi-vector embeddings. During the offline indexing phase,
for each server 𝑀𝑖, we compute embeddings for all text segments
𝑡∈𝐷′
𝑖, resulting in a set of vectors V𝑖={𝐸(𝑡)|𝑡∈𝐷′
𝑖}using a
fine-tuned ERNIE[27] model 𝐸(·).
During online retrieval, for each sub-query 𝑠𝑞𝑗, we embed it
as𝑣𝑠𝑞𝑗=𝐸(𝑠𝑞𝑗)and perform an Approximate Nearest Neighbor
(ANN) search against all server embeddings. The relevance score
between a sub-query and a server is determined by the maximum
similarity over all of the server’s vectors:
sim(𝑠𝑞𝑗,𝑀𝑖)=max
𝑣𝑘∈V 𝑖cos(𝑣𝑠𝑞𝑗,𝑣𝑘) (4)
A MaxSim operation allows for a fine-grained matching between
the query’s intent and specific facets of the server’s functionality.
4.1.4 Multi-Query Score Aggregation. The decomposition yields
multiple sub-queries, so we must aggregate their retrieval results.
For each sub-query 𝑠𝑞𝑗, we retrieve a setP𝑗of top-ranked (server,
score) pairs. We first collect all retrieved pairs from all sub-queriesinto a single candidate pool:
Pcand=𝑘Ø
𝑗=1P𝑗 (5)
This pool,Pcand, contains all unique server-score pairs retrieved
across all sub-queries. We then employ a maximum score aggrega-
tion strategy. For each unique server 𝑚that appears in the candidate
pool, its final aggregated score, score(𝑚), is its highest similarity
score across all sub-queries:
score(𝑚)=max{𝑠|(𝑚,𝑠)∈P cand} (6)
This approach ensures that servers demonstrating strong relevance
to at least one sub-query are prioritized. Finally, we produce the
final retrieved set,Mfinal, by selecting the top- 𝐾servers from the
unique servers inPcandbased on their aggregated score score(𝑚).
This setMfinalserves as the high-recall input for the subsequent
DAG-based Task Planner.
4.2 DAG-based Task Planner
The planner receives the query 𝑞, sub-queriesSQ, and retrieved
serversMfinal. A router model then determines the query’s com-
plexity. For queries classified as simple, a single-task execution
plan is constructed without using the DAG planner. For those
deemed complex, a dedicated DAG planner is invoked to generate
a more sophisticated plan. This acknowledges that for complex
reasoning, linear execution plans are often suboptimal, motivating
the exploration of non-linear structures like graphs or trees [3].
The planner, implemented with a highly capable LLM 𝑝LLM-plan ,
is prompted to act as a solution architect with a specific prompt (see
Appendix A.3). It analyzes the relationships between sub-queries
and the capabilities of the retrieved servers to construct a DAG,
G=(V,E).
VerticesV: Each vertex 𝑣𝑘∈V represents a high-level sub-task
𝑠𝑡𝑘. The planner defines each sub-task as a tuple 𝑠𝑡𝑘=(𝑠𝑞′
𝑘,𝑀𝑘),

TURA: Tool-Augmented Unified Retrieval Agent for AI Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
where𝑀𝑘∈M finalis the optimally chosen MCP Server, and 𝑠𝑞′
𝑘is a refined, context-aware sub-query. This sub-query might be
a direct pass-through of some 𝑠𝑞𝑗∈SQ , or it could be a newly
formulated instruction that incorporates the expected output from
a parent node in the DAG.
EdgesE: A directed edge(𝑣𝑎,𝑣𝑏)∈E indicates a strict data
dependency. The planner establishes this edge if the sub-task 𝑠𝑡𝑏
requires the output of sub-task 𝑠𝑡𝑎as part of its input. For example,
for the query "Beijing trip June 10-15. Need hotel, 2-3 attractions
and things to do.", the planner identifies that Sub-task4 (Path plan-
ner) depends on the outputs of Sub-task2 (Top Attractions) and
Sub-task3 (Hotel Booking). Specifically, the path planner needs the
list of attraction locations and the hotel’s address to generate an op-
timal travel route. Therefore, the planner establishes directed edges
(𝑣2,𝑣4)and(𝑣3,𝑣4)to pass the attraction and hotel information to
the path planner sub-task. Meanwhile, Sub-task1 (Search Beijing
weather) can be executed in parallel as it has no dependencies.
The output is a structured representation of this DAG, which
enables an execution engine to identify and run independent tasks
in parallel, drastically reducing the latency L(𝜋)for complex, multi-
hop queries.
4.3 Distilled Agent Executor
The final stage is the execution of the plan G. An orchestrator
traverses the DAG, dispatching each sub-task 𝑠𝑡𝑘=(𝑠𝑞′
𝑘,𝑀𝑘)to
our lightweight Agent Executor, A𝜃, as it becomes executable. For
single-task plans, this process is simplified to a direct execution
of the task without DAG traversal. The agent’s responsibility is to
achieve the goal defined by 𝑠𝑞′
𝑘by interacting exclusively with the
toolsT𝑘within the assigned server 𝑀𝑘.
Directly using a large-scale LLM like Deepseek-V3[ 19] for this
fine-grained execution is infeasible in a real-time system, as the
context for each decision (conversation history, server description,
dozens of tool APIs) would lead to unacceptable inference latency.
We overcome this via agent distillation[15].
4.3.1 Trajectory Synthesis and Data Curation. We first bootstrap
a dataset of expert demonstrations, 𝐷expert [39]. For a large set
of representative sub-tasks, we use a powerful teacher model like
Deepseek-V3 to generate execution trajectories with specific prompts
(see Appendix A.4). Each trajectory 𝜋𝑖is a sequence of ReAct-style
tuples⟨𝑜𝑡,th𝑡,𝑎𝑡⟩𝑡, where𝑜𝑡is the observation, th𝑡is the chain-of-
thought reasoning, and 𝑎𝑡is the chosen action (a specific tool call
within the server).
This raw data is then subjected to a rigorous, automated curation
pipeline. The first stage, Correctness Filtering, employs a judge
model,𝐽correct [11], to validate each step of a trajectory. This judge
scrutinizes for adherence to API schemas, validity of parameter
values and logical soundness of the thought process leading to the
action. Any trajectory failing these checks is discarded.
Subsequently, the second stage, Efficiency Filtering, uses another
judge,𝐽efficient , to analyze the now-correct trajectories for perfor-
mance. It identifies and flags issues such as action redundancy and
path sub-optimality These inefficient trajectories are then either
pruned or programmatically corrected. This two-stage curationtransforms the noisy expert data into a high-quality, optimal distil-
lation dataset, 𝐷distill :
𝐷distill=𝐽efficient(𝐽correct(𝐷expert)). (7)
4.3.2 Mixed-Rationale Supervised Fine-Tuning (SFT). To achieve
minimal inference latency, we fine-tune Qwen3[ 36] series, a much
smaller model than large-scale LLM like Deepseek-V3 on Ddistill
using a mixed-rationale SFT strategy. The training process explicitly
leverages the chain-of-thought data. The agent A𝜃is trained to
predict the full sequence of tokens, including both the thought and
the action. The loss function is the standard cross-entropy loss over
the target sequence:
LSFT(𝜃)=−∑︁
(𝑠𝑡′,𝜋′)∈D distill|𝜋′|∑︁
𝑡=1log𝑃𝜃(𝑦𝑡|𝑦<𝑡,𝑠𝑡′) (8)
where𝑦𝑡are the tokens of the concatenated thought and action at
each step. By training on rationales, the model learns the underlying
reasoning process that maps observations to optimal actions.
Critically, during online inference, we provide the agent A𝜃with
a specialized prompt that instructs it to directly generate the action,
omitting the thought step. Having implicitly learned the reasoning
patterns, the model can produce the correct action without the
costly auto-regressive generation of the rationale text. This "train-
with-thought, infer-without-thought" paradigm allows the agent to
retain the high-quality decision-making of the teacher model while
operating at a fraction of the computational cost and latency.
5 Experiments
In this section, we conduct a series of comprehensive experiments to
rigorously evaluate the performance and efficiency of our proposed
TURA framework. Our evaluation is structured around four key
research questions (RQs):
RQ1: How does TURA perform in end-to-end scenarios compared
to the baselines, both in offline benchmarks and live online
production environments?
RQ2: What is the contribution of our proposed Intent-Aware MCP
Server Retrieval module? Specifically, how do query decom-
position and index augmentation impact retrieval efficacy?
RQ3: To what extent does the DAG-based Task Planner improve
the system’s performance, particularly for complex queries
requiring multi-tool coordination?
RQ4: How effective is our agent distillation strategy in creating
a lightweight yet highly capable executor? Can a smaller
language model, when properly distilled, match the execu-
tion quality of a much larger teacher model while satisfying
strict latency constraints?
5.1 Experimental Setup
5.1.1 Datasets and Benchmarks. To evaluate real-world perfor-
mance, we built MCP-Bench , a comprehensive benchmark from
anonymized production logs that captures natural query distribu-
tions from simple lookups to complex multi-hop requests. Working
with Baidu’s annotation team, we used a rigorous multi-stage pro-
tocol where experts annotated each query’s ground-truth MCP
Servers, execution trajectories, and ideal answers. Cross-validation

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhejun Zhao et al.
with multiple annotators and consensus resolution achieved a Co-
hen’s kappa of 0.87 for reliability.
5.1.2 Baselines. We compare TURA against a strong baseline and
two ablated versions of our own system:
•LLM + RAG: A powerful LLM (Deepseek-V3) combined with
a standard RAG pipeline. Its retriever is a specialized vari-
ant of the Baidu Search API, which bypasses the reranking
stage to provide raw documents. The LLM synthesizes an
answer based on the retrieved web content without actively
executing tools.
5.1.3 Evaluation Metrics. We employ a multi-faceted evaluation
strategy.
•End-to-End Offline Evaluation: We measure Answer
Accuracy andFaithfulness . Answer Accuracy assesses
whether the final generated answer correctly addresses the
user’s query. Faithfulness evaluates whether the answer is
grounded in and consistent with the information returned by
the invoked tools or web pages. Both metrics are evaluated
using a combination of human annotation and LLM-as-a-
judge on a 3-point scale (Correct/Partially Correct/Incorrect).
•Online A/B Testing: In the live production environment, we
track standard industry metrics: Session Success Rate(SSR) [7],
which measures the fraction of user sessions where a sat-
isfactory answer is provided, and Good vs. Same vs. Bad
(GSB) [42], a human-rated comparison of TURA’s output
against the production baseline.
•Component-wise Evaluation: For detailed ablation stud-
ies, we use targeted metrics. For the retrieval module (RQ2),
we report Recall@5 andPrecision@5 . For the agent ex-
ecutor (RQ4), we measure MCP-Tool Calling Accuracy
andAverage Latency per Step .
5.1.4 Implementation Details. Our TURA implementation utilizes
Qwen3-1.7B for query decomposition and ERNIE as the dense re-
trieval encoder. The DAG planner is implemented using Deepseek-
V3. The agent distillation process employs Deepseek-V3 as the
teacher model. The resulting student agents are fine-tuned from
the Qwen3 series. For latency evaluation, 80th percentile measure-
ments were conducted for tool execution processes. The Qwen3
series models were benchmarked on two NVIDIA L20 GPUs config-
ured identically to the production deployment environment, while
Deepseek-V3 was evaluated using the online service hosted on
Baidu Qianfan platform.
5.2 Overall Performance Evaluation (RQ1)
5.2.1 End-to-End Offline Evaluation. We conduct comprehensive
end-to-end evaluation of TURA against a strong LLM + RAG base-
line on the MCP-Bench dataset. Table 1 shows that TURA achieves
substantial improvements in both answer accuracy and faithfulness
across human and automated evaluations.
TURA demonstrates significant performance gains over the RAG
baseline. In answer accuracy, TURA achieves 87.5% versus RAG’s
65.3% in human evaluation. This substantial gain highlights the
limitations of passive retrieval for complex multi-faceted queries
and validates our hypothesis that active tool planning is essential
for robust performance.Table 1: End-to-end performance comparison on MCP-Bench
Method Accuracy Faithfulness
Human LLM Human LLM
LLM + RAG 65.3% 68.1% 72.4% 75.0%
TURA 87.5% 88.3% 96.2% 97.1%
The improvement in faithfulness is even more pronounced. TURA
achieves 96.2% faithfulness compared to RAG’s 72.4% in human
evaluation. This difference stems from a fundamental architectural
advantage: while RAG relies on synthesis from potentially noisy
text corpora and is prone to hallucination, TURA’s framework en-
ables dynamic invocation of verified tools that provide high-fidelity
information.
The strong correlation between human and LLM evaluations
across both methods validates the reliability of automated evalua-
tion approaches for this task domain.
5.2.2 Online Deployment and A/B Testing Results. Following promis-
ing offline results, TURA was deployed in a live A/B test against the
incumbent LLM + RAG production system. We randomly sampled
103 user queries across multiple domains, with human evaluators
assessing response quality on accuracy, content value, and overall
satisfaction using a comprehensive evaluation framework.
As shown in Table 2, TURA delivered statistically significant
improvements across key business metrics, demonstrating consis-
tent advantages in both session satisfaction and response quality
distribution.
Table 2: Online A/B testing results comparing TURA with
the production baseline. GSB shows TURA’s performance
advantage over LLM + RAG baseline.
System SSR GSB
LLM + RAG (Baseline) 55.1% -
TURA (Ours) 64.0% (+8.9%) 13% / 86% / 4%
Table 3: Detailed performance analysis showing TURA’s 8.7%
performance advantage with significantly reduced error rates
across all issue categories.
System Total
IssuesAdvantage
RateSSR
LLM + RAG (Baseline) 66 - 55.1%
TURA (Ours) 55
(-16.7%)8.7% 64.0%
(+8.9%)
The online results confirm TURA’s superiority. It increased the
Session Success Rate by 8.9% and achieved an 8.7% overall perfor-
mance advantage. In head-to-head comparisons, TURA was rated as
"Good" (strictly better than the baseline) in 13% of cases, while main-
taining "Satisfactory" performance in 86% of cases and reducing
"Bad" ratings to only 4%.

TURA: Tool-Augmented Unified Retrieval Agent for AI Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Our analysis revealed that TURA’s tool-calling capabilities were
a key performance driver, enabling it to excel in scenarios requir-
ing real-time data accuracy where the LLM + RAG baseline failed.
For instance, the baseline showes significant temperature devia-
tions in weather queries and major discrepancies in train schedules,
whereas TURA provided precise, up-to-date information directly
from authoritative sources. This superiority translated to a sharp
reduction in critical failures from 9 in the baseline to just 4 in TURA.
Overall, TURA reduced the total issue count by 16.7% (from 66 to
55), with consistent improvements across all categories: accuracy
(-7.1%), content richness (-28.6%), and content value (-17.6%), indi-
cating a substantial enhancement in response informativeness and
reliability.
Given its robust performance improvements and consistent ad-
vantages across multiple evaluation metrics, TURA has demon-
strated clear superiority over traditional LLM + RAG baselines,
validating the effectiveness of synergizing RAG with agentic tool-
use for accessing both static and dynamic information in industrial
AI search productions.
5.3 Ablation Studies and Component Analysis
5.3.1 Analysis of Intent-Aware MCP Server Retrieval (RQ2). To in-
vestigate the efficacy of our retrieval module, we performed detailed
ablations. As shown in Table 4, both query decomposition and index
augmentation are indispensable. Removing decomposition ( w/o
Decomp. ) severely impairs performance, confirming that a single
vector cannot handle multi-intent queries. Removing augmentation
(w/o Augment. ) also causes a significant drop, demonstrating its
necessity in bridging the semantic gap between user queries and
server documentation. The full TURA model, integrating both,
dramatically outperforms all variants.
Table 4: Ablation study of the retrieval module on MCP-
Bench. Both query decomposition and index augmentation
are critical for performance.
Retrieval Method Recall@5 ↑Precision@5↑
Dense Retrieval (ERNIE) 0.4187 0.5505
TURA (w/o Augment.) 0.7500 0.8555
TURA (w/o Decomp.) 0.4530 0.5631
TURA (Full) 0.8289 0.9190
We then analyzed the configuration of the index augmentation.
First, we determined the optimal number of synthetic queries ( 𝑁𝑄).
As shown in Figure 3, performance peaks at 𝑁𝑄=20and then
plateaus. This suggests 20 queries provide sufficient semantic cov-
erage without adding noise, so we fix 𝑁𝑄=20.
Next, we explored how to structure these queries in the index
(Table 5). A Single-Vector approach, which concatenates all text
into one document for embedding, suffers from semantic dilution
and performs worst. In contrast, Multi-Vector approaches, which
create separate embeddings for different parts of the server’s infor-
mation, achieve superior performance. This is because they offer
higher representational granularity, providing focused semantic
targets. While using only synthetic queries ( Queries Only ) per-
forms marginally best, we chose the Queries + Doc strategy. This
Figure 3: Impact of the number of synthetic queries ( 𝑁𝑄)
per server on retrieval Recall@5. Performance peaks at 20
queries.
method retains the original server document as a "safety net," en-
suring robustness for queries not covered by the synthetic data, a
crucial feature for real-world deployment.
Table 5: Comparison of different index representation strate-
gies. Multi-vector approaches excel due to higher representa-
tional granularity. [ 𝑁𝑄=20]
Index Representation Strategy Recall@5 ↑P@5↑
Single-Vector 0.7721 0.8956
Multi-Vector (Queries + Doc) 0.8289 0.9190
Multi-Vector (Queries Only) 0.8299 0.9208
Table 6: Impact of the DAG-based Task Planner on the com-
plex, multi-hop query subset of MCP-Bench.
Planning Method Success Rate (%) Avg. Latency (ms) ↓
Sequential Plan 88.9% 1,650
DAG Plan 89.1% 920
5.3.2 Importance of the DAG-based Task Planner (RQ3). While
Table 1 showes the latency impact of the DAG planner on the entire
dataset, we conduct a targeted analysis on a challenging subset
of MCP-Bench containing only complex, multi-hop queries where
parallelism is possible. This isolates the planner’s contribution to
efficiency.
As Table 6 illustrates, the DAG-based planner reduces average
latency by 44.2% on these complex queries by identifying and exe-
cuting independent sub-tasks in parallel. This substantial efficiency
gain is achieved with no degradation in the execution success rate,
confirming the effectiveness of our planner in optimizing complex
workflows for online latency.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhejun Zhao et al.
Table 7: Performance of the distilled agent executor. Distillation significantly improves both accuracy and latency over the base
models, achieving near-teacher accuracy at a fraction of the cost. P80 latency is reported.
Agent Model Model Size Tool Calling Acc. (%) ↑Avg. Latency/Step P80 (ms) ↓
GPT-4o N/A 81.7 6,800
Teacher (Deepseek-V3) 671B-A37B 82.4 8,700
Qwen3-1.7B 1.7B 43.5 1,500
Qwen3-4B 4B 70.1 2,200
Qwen3-30B-A3B 30B-A3B 76.3 2,600
Qwen3-1.7B Distilled 1.7B 77.6 620
Qwen3-4B Distilled 4B 88.3 750
Qwen3-30B-A3B Distilled 30B-A3B 88.7 760
5.3.3 Effectiveness of Agent Distillation (RQ4). To address RQ4,
we conduct a comprehensive evaluation of our proposed agent
distillation methodology. The objective is to produce compact, low-
latency student models that not only retain but ideally surpass
the task-solving capabilities of the large teacher model. The per-
formance of the distilled student models is benchmarked against
their base versions, the teacher model (Deepseek-V3), and a strong
proprietary baseline (GPT-4o)[ 12], focusing on two key metrics:
function-calling accuracy and P80 inference latency.
The empirical results, presented in Table 7, unequivocally demon-
strate the efficacy of our approach. We highlight two primary find-
ings. First, our distilled models achieve a remarkable level of perfor-
mance, surpassing even the powerful teacher model. Specifically,
theQwen3-4B Distilled andQwen3-30B-A3B Distilled models
attain accuracies of 88.3% and 88.7% respectively. These results are
substantially higher than both the 671B parameter teacher (82.4%)
and the formidable GPT-4o baseline (81.7%). This phenomenon,
where the student outperforms the teacher, validates the high qual-
ity of the synthetic trajectories generated by our data curation
pipeline, which effectively filters noise and crystallizes optimal
reasoning paths into a targeted training dataset.
Second, the distillation process yields significant improvements
over the base models in both accuracy and efficiency. For instance,
theQwen3-4B Distilled model boosts accuracy by +18.2 absolute
percentage points over its base counterpart (from 70.1% to 88.3%)
while concurrently achieving a 66% reduction in P80 latency (from
2,200ms to 750ms). This dual enhancement is a direct consequence
of our "train-with-thought, infer-without-thought" paradigm. Dur-
ing training, this technique imbues the student model with the
teacher’s complex reasoning patterns. At inference, the student di-
rectly generates the concise, final action, minimizing token output
and thus latency.
In selecting the final model for deployment, we considered the
trade-offs between performance and operational cost. While the
Qwen3-30B-A3B Distilled model, a Mixture-of-Experts (MoE)
architecture[ 14], registered the highest accuracy, we opted for the
Qwen3-4B Distilled model. The rationale is rooted in deployment
feasibility and long-term cost-effectiveness. While the 3B-activated
MoE model achieves similar inference performance to the 4B dense
model, it requires dual-GPU L20 deployment due to its large total
parameter size. The 4B model, however, can be efficiently served ona single GPU. This makes the Qwen3-4B Distilled model the most
pragmatic choice, offering a superior balance between accuracy and
sustainable deployment costs. In summary, our agent distillation
framework successfully forges agents that are smaller, faster, and
more accurate, demonstrating a viable path for deploying powerful
yet efficient agents in production systems.
6 Conclusion
This paper introduced TURA, a novel agentic framework designed
to bridge the gap between traditional static RAG systems and the
growing demand for dynamic, real-time information access in mod-
ern AI search. TURA overcomes passive retrieval’s limitations
through a cohesive three-stage architecture: Intent-Aware Retrieval
for precise tool selection, DAG-based Task Planning for latency-
optimized parallel execution, and an efficient Distilled Agent Ex-
ecutor. This empowers AI search to handle complex, multi-faceted
queries previously intractable for conventional RAG systems. Rigor-
ous empirical evaluation, validated by a large-scale online A/B test
in production environment, confirms TURA’s significant superior-
ity. It markedly outperforms strong baselines, delivering substantial
gains in answer accuracy and faithfulness, and a notable increase in
Session Success Rate. This work presents a production-proven blue-
print for the next generation of conversational AI, demonstrating a
clear paradigm shift from passive information retrieval to active,
tool-augmented systems. By enabling the seamless integration of
heterogeneous, real-time data sources, TURA establishes a new
benchmark for building robust and scalable industrial-grade AI
search products.

TURA: Tool-Augmented Unified Retrieval Agent for AI Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Anthropic. 2024. Introducing the Model Context Protocol. https://www.anthropic.
com/news/model-context-protocol. accessed 2025-08-06.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection.
(2024).
[3]Maciej Besta, Florim Memedi, Zhenyu Zhang, Robert Gerstenberger, Nils Blach,
Piotr Nyczyk, Marcin Copik, Grzegorz Kwasniewski, Jürgen Müller, Lukas Gi-
aninazzi, et al .2024. Topologies of reasoning: Demystifying chains, trees, and
graphs of thoughts. CoRR (2024).
[4]Sergey Brin and Lawrence Page. 1998. The anatomy of a large-scale hypertextual
web search engine. Computer networks and ISDN systems 30, 1-7 (1998), 107–117.
[5]Yanfei Chen, Jinsung Yoon, Devendra Singh Sachan, Qingze Wang, Vincent
Cohen-Addad, Mohammadhossein Bateni, Chen-Yu Lee, and Tomas Pfister. 2024.
Re-invoke: Tool invocation rewriting for zero-shot tool retrieval. arXiv preprint
arXiv:2408.01875 (2024).
[6]Gobinda Chowdhury and Sudatta Chowdhury. 2024. AI-and LLM-driven search
tools: A paradigm shift in information access for education and research. Journal
of Information Science (2024), 01655515241284046.
[7]Alex Deng and Xiaolin Shi. 2016. Data-driven metric development for online
controlled experiments: Seven lessons learned. In Proceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining . 77–
86.
[8]Nicholas Elia. 2017. Innovative Product, Innovative Remedy: Essential Facility as
a Compromise for the Antitrust Charges against Google’s OneBox in the United
States and the European Union. Temp. Int’l & Comp. LJ 31 (2017), 465.
[9]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining . 6491–6501.
[10] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. In International conference on
machine learning . PMLR, 3929–3938.
[11] Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing
Xu, and Tiejun Zhao. 2024. An empirical study of llm-as-a-judge for llm evalua-
tion: Fine-tuned judge model is not a general substitute for gpt-4. arXiv preprint
arXiv:2403.02839 (2024).
[12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
[13] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with
generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020).
[14] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche
Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou
Hanna, Florian Bressand, et al .2024. Mixtral of experts. arXiv preprint
arXiv:2401.04088 (2024).
[15] Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, and Sung Ju Hwang.
2025. Distilling llm agent into small models with retrieval and code tools. arXiv
preprint arXiv:2505.17612 (2025).
[16] Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney,
Kurt Keutzer, and Amir Gholami. 2024. An llm compiler for parallel function
calling. In Forty-first International Conference on Machine Learning .
[17] Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae Lee. 2025. Shifting from
Ranking to Set Selection for Retrieval Augmented Generation. arXiv preprint
arXiv:2507.06838 (2025).
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems 33 (2020), 9459–9474.
[19] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 (2024).
[20] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models
use long contexts. arXiv preprint arXiv:2307.03172 (2023).
[21] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
rewriting in retrieval-augmented large language models. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing . 5303–5315.
[22] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The
PageRank citation ranking: Bringing order to the web. Technical Report. Stanford
infolab.
[23] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sab-
harwal, Mohit Bansal, and Tushar Khot. 2023. Adapt: As-needed decomposition
and planning with language models. arXiv preprint arXiv:2311.05772 (2023).
[24] Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Di-
vyansh Garg, and Rafael Rafailov. 2024. Agent q: Advanced reasoning andlearning for autonomous ai agents. arXiv preprint arXiv:2408.07199 (2024).
[25] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin
Cong, Xiangru Tang, Bill Qian, et al .2023. Toolllm: Facilitating large language
models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789 (2023).
[26] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli,
Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves to use tools. Advances in
Neural Information Processing Systems 36 (2023), 68539–68551.
[27] Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Ji-
axiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al .2021. Ernie 3.0: Large-scale
knowledge enhanced pre-training for language understanding and generation.
arXiv preprint arXiv:2107.02137 (2021).
[28] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao,
and Le Sun. 2023. Toolalpaca: Generalized tool learning for language models
with 3000 simulated cases. arXiv preprint arXiv:2306.05301 (2023).
[29] Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang, Zhen Li, Zhengwei Tao,
and Shuai Ma. 2024. Llms are also effective embedding models: An in-depth
overview. arXiv preprint arXiv:2412.12591 (2024).
[30] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng,
and Heng Ji. 2024. Executable code actions elicit better llm agents. In Forty-first
International Conference on Machine Learning .
[31] Ziting Wang, Haitao Yuan, Wei Dong, Gao Cong, and Feifei Li. 2024. Corag: A cost-
constrained retrieval optimization system for retrieval-augmented generation.
arXiv preprint arXiv:2411.00744 (2024).
[32] Rhiannon Williams. 2024. Why Googleś AI Overviews Gets Things
Wrong. https://www.technologyreview.com/2024/05/31/1093019/why-are-
googles-ai-overviews-results-so-bad/. MIT Technology Review, accessed 2025-
08-06.
[33] Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian,
Yanghua Xiao, and Yu Su. 2024. Travelplanner: A benchmark for real-world
planning with language agents. arXiv preprint arXiv:2402.01622 (2024).
[34] Haoyi Xiong, Jiang Bian, Yuchen Li, Xuhong Li, Mengnan Du, Shuaiqiang Wang,
Dawei Yin, and Sumi Helal. 2024. When search engine services meet large
language models: visions and challenges. IEEE Transactions on Services Computing
(2024).
[35] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor nega-
tive contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808
(2020).
[36] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report. arXiv preprint arXiv:2505.09388 (2025).
[37] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models.
InInternational Conference on Learning Representations (ICLR) .
[38] Fuda Ye, Shuangyin Li, Yongqi Zhang, and Lei Chen. 2024. R2AG: Incorpo-
rating Retrieval Information into Retrieval Augmented Generation. In EMNLP
(Findings) .
[39] Fan Yin, Zifeng Wang, I Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T
Le, Kai-Wei Chang, Chen-Yu Lee, et al .2025. Magnet: Multi-turn tool-use data
synthesis and distillation via graph translation. arXiv preprint arXiv:2503.07826
(2025).
[40] Yinying Zhang and Wenqi Duan. 2012. Envelopment-competition Pattern of
E-Business Platform–Insights from the Competition among Taobao, Baidu and
Tencent. In 2012 Fifth International Conference on Business Intelligence and Finan-
cial Engineering . IEEE, 51–55.
[41] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang,
Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al .2022. Least-to-
most prompting enables complex reasoning in large language models. arXiv
preprint arXiv:2205.10625 (2022).
[42] Lixin Zou, Shengqiang Zhang, Hengyi Cai, Dehong Ma, Suqi Cheng, Shuaiqiang
Wang, Daiting Shi, Zhicong Cheng, and Dawei Yin. 2021. Pre-trained language
model based ranking in Baidu search. In Proceedings of the 27th ACM SIGKDD
Conference on Knowledge Discovery & Data Mining . 4014–4022.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Zhejun Zhao et al.
A Prompt Templates
This section provides illustrative examples of the prompt templates
used in our methodology. For brevity and clarity, we have abstracted
the core instructions and structures. These templates are designed
to guide the LLM in performing specific sub-tasks within our frame-
work.
A.1 Tool Profile Generation
Instruction As a tech expert, analyze a tool’s document to
generate diverse, practical example queries that showcase
its core functions.
Input Document: {doc}
Output The expected output is a JSON array of strings, where
each string is an example query.
[
"Example query demonstrating feature A",
"Another query for a different use case",
"Query with specific parameters mentioned
in the doc"
]
A.2 Query Decomposition
Instruction Deconstruct the user query into independent, atomic
sub-tasks. Ensure full coverage of the original intent while
maintaining independence between tasks.
Input User Query: {query}
Output The expected output is a JSON object containing a list
of atomic sub-tasks. For example, if the input query is I need
to book a flight to Shanghai for next week and find a good
local restaurant there. , the output would be:
{
"tasks": [
"book flight to Shanghai for next week",
"find recommended restaurants in Shanghai"
]
}
A.3 Task Planning with DAG
Instruction Analyze the user query and decompose it into
a Directed Acyclic Graph (DAG) of executable sub-tasks.
Define each task and its dependencies to create an optimal
execution plan.
Input User Query: {query}
Output The expected output is a JSON object defining tasks
and their dependencies in a DAG structure:
{
"tasks": {
"T1": "task description 1",
"T2": "task description 2"
},
"dependency": [
"T1->T2"
]}
A.4 Tool Execution
A.4.1 Optimizing for Correctness.
Instruction Given a query and a set of available tools, generate
a step-by-step reasoning trace. Accurately select the best tool,
extract parameters, and verify the result before proceeding.
Input•User Query: {query}
•Available Tools: {available_tools}
Output The expected output is a JSON object representing the
reasoning trace for a single step of execution:
{
"thought": "Reasoning for tool selection...",
"action": {
"tool": "<tool_name>",
"params": { ... }
},
"observation": "Result from tool execution...",
"next_step": "..."
}
A.4.2 Optimizing for Efficiency.
Instruction Given a query and a set of available tools, generate
the most efficient tool-calling trace. Minimize redundant
steps, prefer single-step resolutions, and terminate as soon
as the answer is found.
Input•User Query: {query}
•Available Tools: {available_tools}
Output The expected output is a JSON object representing one
step in the efficient tool-calling trace:
{
"step": 1,
"tool": "<tool_name>",
"params": { ... },
"result": "<output>",
"terminate": true
}
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009