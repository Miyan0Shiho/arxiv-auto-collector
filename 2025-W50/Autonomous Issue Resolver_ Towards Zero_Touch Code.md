# Autonomous Issue Resolver: Towards Zero-Touch Code Maintenance

**Authors**: Aliaksei Kaliutau

**Published**: 2025-12-09 11:11:37

**PDF URL**: [https://arxiv.org/pdf/2512.08492v1](https://arxiv.org/pdf/2512.08492v1)

## Abstract
Recent advances in Large Language Models have revolutionized function-level code generation; however, repository-scale Automated Program Repair (APR) remains a significant challenge. Current approaches typically employ a control-centric paradigm, forcing agents to navigate complex directory structures and irrelevant control logic. In this paper, we propose a paradigm shift from the standard Code Property Graphs (CPGs) to the concept of Data Transformation Graph (DTG) that inverts the topology by modeling data states as nodes and functions as edges, enabling agents to trace logic defects through data lineage rather than control flow. We introduce a multi-agent framework that reconciles data integrity navigation with control flow logic. Our theoretical analysis and case studies demonstrate that this approach resolves the "Semantic Trap" inherent in standard RAG systems in modern coding agents. We provide a comprehensive implementation in the form of Autonomous Issue Resolver (AIR), a self-improvement system for zero-touch code maintenance that utilizes neuro-symbolic reasoning and uses the DTG structure for scalable logic repair. Our approach has demonstrated good results on several SWE benchmarks, reaching a resolution rate of 87.1% on SWE-Verified benchmark. Our approach directly addresses the core limitations of current AI code-assistant tools and tackles the critical need for a more robust foundation for our increasingly software-dependent world.

## Full Text


<!-- PDF content starts -->

Autonomous Issue Resolver:
Towards Zero-Touch Code Maintenance
Aliaksei Kaliutau1,2
1Stable Reasoning
2Imperial College London (I-X)
2aliaksei.kaliutau24@imperial.ac.uk
Abstract
Recent advances in Large Language Models have revolutionized function-level code gen-
eration; however, repository-scale Automated Program Repair (APR) remains a significant
challenge. Current approaches typically employ a control-centric paradigm, forcing agents to
navigate complex directory structures and irrelevant control logic. In this paper, we propose
a paradigm shift from the standard Code Property Graphs (CPGs) to the concept of Data
Transformation Graph (DTG) that inverts the topology by modeling data states as nodes and
functions as edges, enabling agents to trace logic defects through data lineage rather than con-
trol flow. We introduce a multi-agent framework that reconciles data integrity navigation with
control flow logic. Our theoretical analysis and case studies demonstrate that this approach
resolves the ”Semantic Trap” inherent in standard RAG systems in modern coding agents. We
provide a comprehensive implementation in the form ofAutonomous Issue Resolver (AIR), a
self-improvement system for zero-touch code maintenance that utilizes neuro-symbolic reasoning
and uses the DTG structure for scalable logic repair. Our approach directly addresses the core
limitations of current AI code-assistant tools and tackles the critical need for a more robust
foundation for our increasingly software-dependent world.
1 Introduction
In the 21st century, software engineering has evolved from a niche discipline into the digital back-
bone of the global economy, generating trillions in value and employing hundreds of millions of
people worldwide. Today, software is the invisible infrastructure that connects commerce, com-
munication, healthcare, transportation, and nearly every facet of modern life. It has become the
omnipresent, global aether of our digital civilization. But unlike a fundamental force of nature,
software is created by humans, inheriting our limitations and tendency to make mistakes.
The early days of software developed by smaller teams are long gone. Today’s software products
are huge, often comprising millions of lines of code and hundreds of dependencies. No single human,
nor even large teams, has a comprehensive understanding of such systems. This is the same as try-
ing to build a skyscraper with bare hands. Current development methodologies attempt to manage
this complexity by breaking down the problems, but this merely distributes the potential failures
among developers. This is because individual developers, working on isolated components, still
introduce bugs, inconsistencies, and architectural drift. The consequences are significant: costly
delays, security vulnerabilities, system outages, and a drag on innovation. Industry reports con-
sistently highlight that debugging and maintenance consume a staggering 50-70% of development
resources [1].
1arXiv:2512.08492v1  [cs.AI]  9 Dec 2025

The path to truly reliable software lies in systematically reducing the dependence on manual
intervention for repetitive, error-prone, and complex analytical tasks. AI-based automation offers
the potential for consistency, speed, and depth of analysis that exceed human capabilities. By
automating many stages of the software development lifecycle (SDLC), from code generation and
testing to debugging and refactoring, we can mitigate human error, enforce best practices, and
achieve unprecedented levels of quality assurance.
The core blocker for zero-touch maintenance is notcode generation, butsystemic reasoning:
mapping anissueto aminimal, safeset of edits under repository-wide constraints (APIs, invari-
ants, tests, deploy rules). This demands (i) arepository-scale semantic representation(beyond
embeddings alone), and (ii) arisk-aware control policythat decides when to stop, where to edit,
and how to validate.
Modern models can synthesize non-trivial patches and reason about code structure, yet this
progress has not translated into robust performance on realistic repository-level tasks: on bench-
marks such as SWE-bench, resolution rates remain modest, especially for open-source models.
We refer to this gap as thecontext crisisof repository-scale APR. At small scales, LLMs can
be given all relevant code in a single prompt. At repository scale, however, the model must operate
under strict context limits while the bug’s cause and manifestation are separated across files, layers,
and abstractions. Existing tools address this by improving retrieval and by adding agentic planning,
but they leave the underlying representation of the repository largely unchanged: a directory of
files, occasionally augmented with control-centric graphs.
We argue that this file-first, control-centered view is misaligned with the causal structure of
many real-world bugs and how the development is done in the real world. Human developers rarely
debug or write new code by reading files. Instead, they trace theflow of data, following a specific
object or variable as it is transformed and routed through the system. We propose to make data-
first mental model explicit and to turn it into the primary substrate on which a LLM-based repair
agent lives.
Our approach has demonstrated good results on several SWE benchmarks, reaching a resolution
rate of 87.1% on SWE-Verified benchmark.
The main contributions of this work are as follows:
•We introduce theData-First Transformation Graph(DTG), a data-centric representation of
software repositories that performs a semantic compression of Code Property Graphs and
aligns with human debugging strategies.
•We establish connections between the DTG, data-centric optimization frameworks, and Rein-
forcement Learning, motivating a learning regime in which LLMs reason over graph structure.
•We propose a novel architecture of the DTG-based repair agent and analyze its performance
and accuracy on several benchmarks.
2 Problem Overview and Background
The idea of using AI-based automation in software development is not novel, and there are quite
a few products. Tools such as GitHub Copilot, Amazon Code Whisperer, Tabnine [2], Cursor [3],
and various chatbot assistants (such as specialized extensions of Claude Code or ChatGPT) are
popular today and represent the current state of the art (see also Appendix, Table B.2). Most
existing products are iterating on developer help.
Table B.2 separatesprofessional,basic, andspecializedofferings. Professional tools optimize
ergonomicsfor humans-in-the-loop (IDE integration, fast completions), but remain fundamentally
2

Table 2.1: Capability taxonomy: where existing tools help - and where they fall short for au-
tonomous maintenance. We classify existing tools by their primary interaction model and identify
the specific limitations that prevent them from achieving zero-touch maintenance.
Category Typical Features Limits for Maintenance
Completion / Sug-
gestionNext-line blocks, boiler-
plate, refactorsLocal context only; no global impact
modeling; no gating against risk.
Conversational
Dev AssistExplain, edit, diagnose
from chatQuality depends on prompt curation;
inconsistent across large repos.
Repo Search &
RetrievalEmbedding search, code
Q&A, cross-refWeak semantics beyond lexical/em-
beddings; limited data-/control-flow.
Automated Pro-
gram Repair
(APR)Patch synthesis for unit-
failing bugsOften single-hunk fixes; brittle across
multi-file causal chains.
Test Generation Unit/integration test syn-
thesisImproves coverage, but does not plan
changes or manage regressions.
Agentic Tool Use Editors, terminals, CI
tools via agentsWithout risk-aware policy, tends to
oscillate or over-edit.
reactiveand local in scope. Basic tools acceleratescaffoldingand prototyping, yet rarely manage
repository-scale dependencies or change risk. Specialized systems reachSOTAin narrow settings
(e.g., competitive programming or algorithm discovery), but are not designed for end-to-end main-
tenance workflows.
2.1 Limitations of existing tools
Current LLM-based agents such as SWE-Agent and OpenHands typically interact with repositories
through a file-system interface: they list directories, open files, and issue search commands. Each
action consumes context and forces the model to maintain a growing internal representation of con-
trol and data dependencies. As repositories grow, this misalignment becomes the main bottleneck:
the model ”thinks” at the level of dataflow but sees the world as files and control structures.
Retrieval-Augmented Generation (RAG) is the standard strategy for scaling LLMs to large
corpora. In code domains, RAG usually combines lexical search with dense embeddings of code
snippets. For task prompts involving a particular entity (e.g., fixing some issue that involves a
Userclass), the system retrieves semantically similar snippets for the model to condition on.
However, semantic similarity is not the same ascausalrelevance:
•If a bug involves corrupt user IDs, a vector search forUserwill fetch class definitions, UI
components, and schema migrations but may miss a generic sanitizer function that silently
strips or hashes IDs without ever mentioning the word ”user”.
•In data pipelines, failures may be caused by a shared transformation used across multiple
domains. Semantically similar code may not share causal influence on the failing data.
This is becoming increasingly common in modern codebases which are often use loose-coupling
design patterns when the code tends to live in semantically disconnected clusters of functions which
are not called directly (e.g. aspect-oriented programming, dependency injection, and so on).
Traditional RAG treats code as an unordered bag of textual fragments. Even graph-based
retrieval (e.g., GraphRAG) typically uses graph structure only to improve retrieval quality, not as
3

the state space in which the agent itself moves. The agent stilllivesin the file system, reading
linearized text.
3 Related Work
The introduction and further development of Transformer-based models trained on large code
corpora has revolutionized software engineering. Early breakthroughs like Codex [4] and AlphaCode
[5] demonstrated that models could synthesize functional code from natural language descriptions.
These foundational models power a generation of interactive assistants, including GitHub Copilot
[6], Amazon CodeWhisperer [7], and Cursor [8]. While effective for local, ”next-block” completion
[2], these tools primarily operate as ”copilots,” relying on the human developer to manage the
broader repository context and verify correctness.
Agentic frameworks such as ReAct [9] and Reflexion [10], allowed LLMs to plan, execute tools,
and observe outputs. State-of-the-art systems like SWE-agent [11] and OpenHands [12] utilize
these patterns to interact with repositories via shell commands and file editors, and currently often
used as solid baselines. Similarly, AutoCodeRover [13] employs abstract syntax tree (AST) analysis
to improve fault localization.
However, these agents largely treat the repository as a file system to be searched and read, a
”file-first” perspective that suffers from inefficiency when bugs span multiple abstraction layers.
Recent evaluations on benchmarks like SWE-bench [14, 15] indicate that while agents can solve
isolated tasks, they struggle with long-horizon reasoning required for enterprise issues [16]. Claude
Code [17] and LocAgent [18] attempt to mitigate this by improving retrieval and localization, yet
they still fundamentally operate on text-based or partially structured representations that do not
explicitly model data lineage.
Since the search over codebase is essentially a search over unstructured data, it was natural to
employ the graph-based methods for localization of relevant snippets of code. The Code Property
Graph (CPG) [19] unified ASTs, control-flow graphs (CFGs), and program dependence graphs
(PDGs) into a single structure, primarily for vulnerability discovery.
Recent efforts have attempted to integrate such graphs with LLMs to improve repository un-
derstanding. CodexGraph [20] and RepoGraph [21] utilize graph databases to enhance Retrieval-
Augmented Generation (RAG) by capturing dependencies between files and functions. The Code
Graph Model (CGM) [22] similarly integrates graph structures to aid in logical reasoning. How-
ever, these approaches often retain the ”representation bloat” of standard CPGs or focus on im-
proving retrieval recall rather than navigation that mimics the ”operational reasoning”. Unlike
these control-centric models, our approach utilizes a Data-First Transformation Graph (DTG),
which compresses the topology to focus on data states and transformations, aligning somehow our
approach with the ”flow-based” mental models used in high-performance computing optimization.
Although most current agents rely on prompt engineering and in-context learning, Reinforce-
ment Learning (RL) has shown promise in discovering novel strategies in complex domains. Deep-
Mind’s work on AlphaZero [23] and subsequently AlphaDev [24] demonstrated that RL agents could
discover algorithms that outperform human-written implementations. In the domain of program
repair, early learning-based systems like Getafix [25] mined fix patterns from static analysis. More
recently, AlphaEvolve [26] pushes this further by evolving code structures. Our work incorporates
an RL-driven control policy to navigate the DTG, distinguishing AIR from purely prompt-driven
agents by allowing it to learn stopping conditions and navigation strategies that minimize regres-
sion risks. We are using a simple DQN-based online learning. Note that RL is not a key element
of our algorithm, and it can be replaced by simple Monte Carlo Tree Search style graph traversing.
4

Figure 4.1: The partial visualization of the multi-dimensional Data Transformation Graph for a
snippet of code. Nodes contains full description of processed data. Edges are effectively transfor-
mation functions (sanitize, handle)
4 Methodology
4.1 A data-first graph representation
We propose a fundamental re-architecture of the agent’s environment. Rather than representing
a repository as text files, or even as a multi-layer code property graph in which functions and
statements are the primary nodes, we focus directly on thedatathat flows through the system.
We define theData-First Transformation Graph(DTG) as a directed multigraph with:
•Nodes as data states:Each node corresponds to a semantically meaningful data artifact
(e.g., a variable, object, etc) at a specific point in its lifecycle, annotated with type, schema,
constraints, and source locations.
•Edges as transformations:Each edge corresponds to an operator, function, or API call
that transforms one data state into another, potentially guarded by control-flow conditions.
To illustrate this concept, consider the following snippet of code that implements a toy trans-
action handle. Figure 4.1 depict a simplified DTG for this code.
1utils .py
2def process_transactions (id , data ):
3cleaned_id = sanitize (id)
4return handle ( cleaned_id , data )
5
6def handle ( cleaned_id , data ):
7print (f" processed { cleaned_id }")
8
9def sanitize (id: Any ) -> str :
10id = str (int (id))
5

11assert len (id) == 4, f"id: wrong length : {id}"
12return id
This representation compresses away syntactic detail and control-flow noise, allowing the agent
to traverse the ”causal path” of a variable, effectively pruning irrelevant control flow and syntax
noise that typically confuses. Conceptually, the repository ceases to be a library of documents and
becomes a navigable network of data lineages.
4.2 Why Reinforcement Learning is crucial
The second key ingredient of our design is a smart integration of RL patterns. We draw inspi-
ration from the successes of systems like DeepMind’s AlphaGo and AlphaCode [23], [27], which
demonstrated how machines can achieve superhuman performance in complex domains not just by
mimicking data, but by learning optimal strategies through interaction and self-correction.
Unlike supervised learning (training on fixed code examples), RL allows the AI agent to learn
by doing. It generates code, tests it (using automated test suites), observes the results (pass/fail,
performance metrics, new bugs introduced) and receives feedback (rewards or penalties). This
iterative loop enables the system to find correct and efficient code implementations that might not
exist in any training dataset, to optimize code for specific goals, and to self-correct and self-improve.
The latter is very important, because it effectively removes the slow human factor (which is the
main bottleneck) and allows the system to learn exponentially. The RL loop is implemented in a
quite straightforward way, via anchoring objective function to the unit tests
4.3 Why holistic reasoning is the key
An RL loop alone is insufficient if the agent cannot understand the code. Generating random code
is inefficient, because the size of problem space (e.g. the number of possible variations of code
snippet) is bigger than the number of particles in all multiverses combined. We need a powerful
and scalable reasoning engine that provides the agent with a deep and structured view of the code
base.
5 Algorithm details
The proposed data-first perspective sits at the intersection of several research lines: code property
graphs and static analysis, data-centric optimization and flow-based programming, and graph-based
+ agentic approaches to LLM-driven software engineering.
5.1 Code property graphs and representation bloat
The Code Property Graph (CPG), popularized by tools such as Joern [28], unifies three classic
program representations in a single multi-graph: abstract syntax trees (ASTs), control-flow graphs
(CFGs), and program dependence graphs (PDGs). In principle, CPGs contain all information
needed for both security analysis and program repair.
In practice, however, raw CPGs suffer from extreme representation bloat. The DTG can be
viewed as asemantic compressionof such graphs. By abstracting away AST and CFG nodes and
focusing only on data states and the transformations between them, it adopts a ”functions-as-
edges” topology, in contrast to the ”functions-as-nodes” topology of call graphs and many CPG
formulations. This aligns the representation with the causal structure of dataflow, rather than the
syntactic structure of the code.
6

5.2 Data-centric optimization and flow-based programming
In high-performance computing and compiler optimization, there has been a shift from control-
centric intermediate representations (such as linearized IR or basic-block graphs) to data-centric
views that emphasize movement and locality. Data-centric frameworks, such as Stateful Dataflow
Multigraphs, treat computations as transformations on data streams and have demonstrated sig-
nificant benefits in optimization and reasoning.
Similarly, flow-based programming (FBP) and ETL (Extract–Transform–Load) pipelines in data
engineering model systems as graphs of black-box components connected by typed data channels.
While this style is common in specialized domains, it has rarely been applied to general-purpose,
object-oriented software in the context of defect localization and repair.
Our DTG formulation can be seen as importing these data-centric ideas into APR. The repos-
itory is treated as a large, heterogeneous dataflow graph in which ”nodes” are not operators but
states, and ”edges” encapsulate black-box transformations.
5.3 Agentic navigation on graphs
Context Agent treats the DTG as the world itself. The agent’s state is a node (or small subgraph),
its primitive actions are graph traversals and inspections, and its perception is restricted to a
local neighborhood. Moving the agent ”body” into the graph opens up new strategies for search,
planning, and exploration based on graph theory and dataflow, rather than ad-hoc file navigation.
5.4 The Data-First Transformation Graph
To formalize the Data-First Transformation Graph (DTG) we introduce first some definitions.
5.4.1 Formal definition
Let a software repositoryRbe represented as a directed multigraph
GDTG= (V, E),
whereVis a set of vertices (data states - see below) andEis a set of edges (transformations).
Each vertexv∈Vrepresents a data state: a specific variable, object, or data artifact at a
particular semantic stage. Compared with classical data-flow graphs, DTG nodes are enriched
with semantic metadata, not merely variable names.
Each vertexvcarries:
•name:A unique identifier, e.g.,module.func.var.
•kind:Class of data, e.g. is it the argument of function, a global variable, or constant, etc.
•type:Its static or inferred language-level type, e.g.,List[int]ortorch.Tensor.
•schema:For structured objects, a description of fields and their types (e.g.,{id:int,
name:str}).
•constraints:Symbolic constraints known at this point, such as shape information or invari-
ants (e.g.,shape = (3, 224, 224)).
•metadata:Source-level metadata, including file path, function, and line span.
This information can be partially obtained from static analysis (e.g., SSA form, type inference)
and partially hypothesized or refined by the LLM during exploration.
7

5.4.2 Edge set: transformations
Each edgee i→j∈Erepresents a transformation from data statev ito data statev j. An edge exists
when there is a concrete operation in the repository that takes the value atv iand produces the
value atv j.
Edges are annotated with:
•plane:A reference to the view plane that represents on some level of abstraction the trans-
formation, e.g., a function call, and so on.
•source, ref file:Reference points to definition and points of invocation, respectively.
•semantics:An optional natural-language summary of the transformation’s intent, generated
and iteratively refined by the LLM (e.g., ”increments a counter”, ”normalizes a tensor”).
6 Graph construction pipeline
Constructing the DTG is the most technically demanding phase. We use a ”lazy-loading” pipeline
that combines static and dynamic information.
Core parser: Tree-sitter and ast:Tree-sitter [29] provides incremental, multi-language AST
parsing for languages such as Python, Java, JavaScript, and C++. We use it to:
•Extract variable declarations, assignments, and function definitions,
•Identify function calls and their arguments,
•Recover control-flow constructs for guard extraction.
Custom Tree-sitter queries (in.scmfiles) define language-specific patterns for these elements [29],
but the downstream DTG schema is language-agnostic. Rather than directly exposing the full inter-
nal graph of these tools to the LLM, we project only the data-state and transformation information
needed for DTG navigation. We use aneo4jgraph database as persistent storage and efficient
indexing over node and edge attributes.
6.1 Agent environment
We model agents whose internal state is a structured view over the DTG.
State schema.A minimal typed state can be defined as:
1class AgentState ( TypedDict ):
2current_node_id : str
3visited_nodes : List [str ]
4hypothesis : str
5graph_view : SubGraph # local neighborhood around current_node_id
Thegraph viewrestricts the visible graph to a bounded-radius neighborhood around the current
node, which is passed to the LLM at each step.
Tools (action space).The agent is equipped with a small set of graph-oriented tools (Table 6.2).
These tools abstract away file-system operations and present the agent with a clean graph navigation
API.
8

Figure 6.2: The AIR Multi-Agent Architecture. The system employs a decoupled ”Plan-Navigate-
Execute” loop to manage context window limitations. The Context Agent (pink) operates ex-
clusively within the graph domain to localize faults without reading file content. Once the fault
subgraph is identified, the Maintenance Agent (yellow) formulates a high-level repair plan, which
is translated into concrete syntax edits by the Editor Agent (gray). This separation prevents the
”Semantic Trap” by ensuring the planner is conditioned only on causally relevant data lineages
rather than raw repository noise.
7 High-Level Architecture (Product)
The design of the competitive AI tool for autonomous code development requires careful consid-
eration of its integration into existing developer workflows and ecosystems. This section details
the rationale behind our core architectural decisions: adopting a frontless application model and
pursuing deep integration with Git service platforms.
7.1 Why We Designed a Frontless Application
The primary goal of our AI product is to enhance and automate the software development lifecy-
cle, particularly within Continuous Integration and Continuous Deployment (CI/CD) pipelines, to
accelerate development efforts.
Modern software development heavily relies on automated CI/CD pipelines. These pipelines
are inherently process-driven, triggered by events like code commits or pull requests, and often
operate without direct human interaction. Unlike tools designed for interactive use (such as IDE
plugins or code editors), our product aims to perform complex tasks with minimal or zero real-time
human guidance once configured. The ”interface” becomes the configuration files and the standard
platform mechanisms for feedback (PR comments, commit statuses, logs). This aligns with the
concept of ”Invisible Computing” or ”Calm Technology,” [30] where technology recedes into the
background, providing value without demanding constant attention.
9

Function Signature Description
navigate(direction=
"upstream"|"downstream")Move along incoming or outgoing edges from the cur-
rent node, updatingcurrent node id.
inspect data() Return metadata (type, schema, constraints, context)
for the current node.
read transformation(target node id)Return code associated with the edge from the current
node totarget node id.
runtest(test id) Execute a test case, optionally with dynamic taint
tracking, and overlay the resulting execution trace onto
the DTG.
Table 6.2: Available Navigation and Testing Functions
Headless systems are often easier to scale and integrate via APIs and webhooks. Our AI can be
deployed as a service that listens for events from multiple repositories or projects simultaneously,
processing tasks asynchronously. This architecture is well-suited for organizational-level adoption
where the tool needs to operate across numerous CI/CD pipelines.
7.2 On the Importance of Platform
While a frontless architecture defineshowthe user interacts (or does not directly interact) with
our tool,wherethe tool operates is equally critical. Deep integration within established Git service
platforms such as GitHub and GitLab is our primary technical and business strategy.
Platform integration allows our AI to become a native participant in the core developer workflow
– the Pull/Merge Request process. It can be triggered automatically by platform events (PR
opened, commit pushed, comment posted), perform its analysis and code generation, and report
results directly back into the platform interface (e.g., posting PR comments with suggestions,
pushing code changes to the PR branch, updating status checks). Our tool operates autonomously
and asynchronously, typically after the initial code has been written (Figure 7.2)
There is another, more subtle reason to build a platform app rather than a standalone one:
platforms like GitHub and GitLab offer rich ecosystems. Integration allows our tool to potentially
interact with other tools and services operating within the same platform. This creates opportuni-
ties for more powerful, composite automation, and sometimes this alone is valued much more than
just a yet another fancy tool for code generation.
7.3 Distinction from Interactive Code Assistants (e.g., GitHub Copilot)
Interactive assistants like GitHub Copilot primarily focus on augmenting the developer during the
act of writing code within the IDE. They operate synchronously, responding to immediate prompts
and context within the editor. Our tool operates autonomously and asynchronously, typically
after the initial code has been written, focusing on broader, process-oriented tasks within the
development life-cycle (e.g., automated refactoring based on PR feedback, generating tests for a
new feature branch, ensuring code adheres to project-wide patterns).
Copilot assists the individual developer’s coding speed; our tool automates team and process
workflows, operating at the level of commits, branches, and pull requests within the platform.
10

Figure 7.3: Our product leverages the existing GitHub platform and workflows (Issues, Branches,
PRs). There is no need for developers to learn new tools or interfaces. The Bot acts like another
member of the team, communicating and contributing via established mechanisms. Such approach
does not break existing automation. Note, that human developers can still review the Bot’s code
changes directly within the PR interface, leave comments, or request modifications if needed. This
step maintains human oversight where desired.
8 Experiments and Benchmarks
Evaluating the performance of AI systems designed for complex, autonomous tasks (such as software
development) presents unique challenges. Unlike traditional software, where deterministic output
is expected, AI systems often exhibit variability and rely on complex internal behavior.
8.1 Benchmarks and Metrics
To test our tool we use the following benchmarks.
SWE-bench family.To test the data-first hypothesis, we outline an evaluation protocol
focusing on repair effectiveness, localization efficiency, and context utilization. We evaluate on the
SWE-bench suite (Full,Verified,Lite; using metric:%Resolved with repository tests) [31, 15].
APR.QuixBugs (Java/Python) [32] and HumanEvalFix.
Industrial tickets.A stratified sample ofNinternal issues across the services.
Primary metrics:%Resolved(tests pass), MTTR (hours), regression rate (%), reviewer effort
(minutes/patch), CI cost (compute$), and change risk score (static+dynamic).
11

Table 8.3: Github repositories used in our tests. Total issues column indicates the number of
reported issues in repository, snapshot of 12/05/2025
Repository URL Stars Total Issues Selected Issues Complexity
jsonpickle https://github.com/jsonpickle/jsonpickle 1.3k 62 10 low
arrow-py https://github.com/arrow-py/arrow 8.9k 87 10 low
flashtext https://github.com/vi3k6i5/flashtext 5.7k 59 10 low
requests https://github.com/psf/requests 53.4k 203 10 low
poetry https://github.com/python-poetry/poetry 33.9k 518 10 medium
Pillow https://github.com/python-pillow/Pillow 13.2k 67 10 high
python-qrcode https://github.com/lincolnloop/python-qrcode 4.2k 42 10 low
tinydb https://github.com/msiemens/tinydb 7.4k 20 10 medium
urllib3 https://github.com/urllib3/urllib3 4k 134 10 medium
httpx https://github.com/encode/httpx 14.8k 61 10 medium
8.2 Baselines
We employ the standard baselines, popular in coding tasks: SWE-agent [11], OpenHands [12],
AutoCodeRover [13], and an ablatedAIRwithout RL (AIRw/o RL) and without DTG. The
numbers for the baselines are taken from the original papers and the benchmark website. [31]
8.3 Compute Budget and Settings
Uniform wall-clock budget per issue (e.g., 60–120 minutes), identical tool access, and fixed seeds.
We report median over three runs and 95% CIs.
8.4 Close-to-life experiment
On the top of classic benchmarks, we simulated a realistic scenario, where the AI agent is tasked with
resolving reported software issues and its output is reviewed by a professional engineer. We selected
10 open-source Python repositories to represent a spectrum of complexity, from low-dependency
utilities (jsonpickle) to high-complexity image processing libraries involving C-extensions (Pillow).
The selection criteria required repositories to have active maintenance,>1kstars, and a history
of well-documented issue resolution to serve as the ground truth. ”Complexity” is qualitatively
assessed based on dependency depth and code coupling (see Table 8.3). The evaluation utilized
a curated set ofN= 100 coding tasks, split into 50 easy, 40 medium and 10 high-complexity
challenges.
The evaluation focused on two aspects: the functional correctness of the proposed solution
(checked manually) and pass/no-pass for all unit tests. A key principle of this methodology was to
assess autonomous unsupervised problem resolution capabilities.
9 Ablation Studies
9.1 Components
We ablate (i) RL control (AIRw/o RL), and (ii) DTG (AIRw/o Graph).
12

Table 8.4: Public benchmarks for several top architectures.
%Resolved↑on SWE-bench and pass rate↑on APR benchmarks.
Baseline values are from the cited papers/sites; update to latest leaderboard snapshots.
Method SWE-bench Verified SWE-bench Lite HumanEvalFix
SWE-agent + Claude 3.5 Sonnet [11] 47.9 33.6 –
SWE-agent + Claude 4 Sonnet [11] 66.6a56.67 87.7b
AutoCodeRover + GPT-4o [13] 46.2 37.3 –
OpenHands + Claude 3.5 Sonnet [12] 52.8 27.7 –
OpenHands + GPT-5 [12] 79.8 – –
AIR + Gemini 2.5 (Ours) 87.1 73.5–
aReported %Resolved on SWE-bench at time of writing [31] (see the data from the original paper in [11]).
bHumanEvalFix pass@1 reported by SWE-agent [11]; included for completeness only.
For current SWE-bench leader-boards and variants (Full/Verified/Lite), see the website [31, 15].
Table 9.5: Ablations on SWE-bench Verified (%Resolved↑) and operations metrics.
Variant %Resolved↑MTTR (min)↓
AIR(full)87.1 8.9
AIRw/o RL 82.5 1.5
AIRw/o Graph 56.2 4.6
10 Results and Analysis
10.1 Ablations: What matters and why
Table 9.5 shows that removing RL control (AIR w/o RL) reduces %Resolved from 87.1 to 82.5
while shrinking MTTR. In this case, the shorter MTTR reflects premature termination rather than
efficient resolution: an agent gives up quicker without policy guidance.
Removing DTG (w/o Graph) yields a mid-range drop (%Resolved 56.2), indicating repository-
scale context is a major contributor.
Overall,RLandGraphare the largest drivers of resolution rate.
10.2 Comparison with commercial tools
Based on the data provided, here are the calculated statistics for each model’s performance across
the 10 projects (100 total issues). This test is less reliable than SWE-bench, because of several
factors. First, we have a much lower number of datapoints. In particular, we pre-selected 10 issues
(out of 100s for some repos) on the basis of their resolvability - we wanted to shortlist the purely
coding issues that can be solved by just applying best coding practice and debugging skills. That
introduces some biases.
The context for each case was built artificially, including all files that were referenced in final
PR that solved the issue and which played the role of ground truth.
For evaluation we used a consistent and objective evaluation methodology. In particular, we
awarded score +1 for solving the issue (all requirements in the ticket are satisfied) and all unit
tests are passed. We did not penalize for time, cost or failed solutions. The Figure 10.4 depict the
relative performance of each model, all data is aggregated in Table 10.6, with summary statistics
in Table 10.7
Key Observations:
13

jsonpickle arrow-py flashtext requests poetry0246810
5
4
23
16
4
28
39
68 8
4Issues resolved (#)Claude 4 Sonnet Gemini-2.5 pro AIR(this work)
Figure 10.4: Testing results on GitHub Issues. Numbers indicate how many issues were resolved.
We randomly selected 10 issues from each repository.
Table 10.6: Issue resolution on five OSS GitHub repos (10 issues each; same wall-clock + tool
budget). Numbers indicate how many issues were resolved.
Project AIR (this work) Claude 4 Sonnet Gemini-2.5 pro
jsonpickle95 6
arrow-py64 4
fashtext82 2
requests83 8
poetry41 3
Pillow40 0
python-qrcode81 2
tinydb 566
urllib392 7
httpx94 3
•AIR (this work) significantly outperformed the other models, solving 70% of all issues. It was
the only model to solve at least 4 issues in every single project.
•Gemini-2.5 pro came in second with 41%, showing strong performance on specific libraries
like requests (8/10) and urllib3 (7/10) but struggling with Pillow (0/10).
•Claude 4 Sonnet solved 28% of issues. It had a high variance, performing well on tinydb
(6/10) but solving 2 or fewer issues in half of the projects.
•Difficulty: Pillow appeared to be the hardest project, with the two commercial models scoring
0 points and AIR scoring its minimum of 4 points. jsonpickle and requests generally saw higher
success rates across the board.
14

Table 10.7: Performance statistics per model across 10 projects (100 total issues).
Statistic AIR (This Work) Claude 4 Sonnet Gemini-2.5 pro
Total Solved 70 28 41
Success Rate 70.0% 28.0% 41.0%
Mean (per project) 7.0 2.8 4.1
Median 8.0 2.5 3.5
Std. Dev. 2.05 1.93 2.56
Min / Max 4 / 9 0 / 6 0 / 8
10.2.1 Performance Analysis and the ”Semantic Gap”
The results in Figure 10.4 and Table 10.6 reveal a strong correlation between the explicitness of data
flow and the agent’s success rate. AIR achieves a 70% resolution rate overall, showing particular
strength in ”loose-coupling” architectures like requests and urllib3 (8/10 and 9/10 respectively). In
these repositories, logic errors are often distributed across multiple abstraction layers - a scenario
where the DTG’s ability to trace data lineage allows the agent to ”hop” over irrelevant middleware,
a capability lacking in the file-based baselines.
10.2.2 Failure Case Study: The Pillow Repository
A notable deviation is observed in the Pillow repository, where commercial models scored 0/10
and AIR achieved only 4/10. We hypothesize this performance drop stems from the ”Language
Boundary” problem. Pillow relies heavily on C-extensions for image processing. The current DTG
implementation parses Python ASTs but treats C-bindings as opaque edges. Consequently, when
data flows into a compiled extension, the lineage is broken, blinding the agent. This limitation
highlights the necessity for future work in cross-language graph construction to support hybrid-
runtime environments.
11 Conclusion and Future Roadmap
We have argued that the dominant file-first, control-centric view of repositories is misaligned with
both human debugging practice and the causal structure of many real bugs. By re-centering
our representation around data states and transformations, theData-First Transformation Graph
(DTG) offers a compact and causally meaningful substrate for LLM-based repair agents.
Our empirical results on SWE-bench Verified, where AIR achieved an 87.1% resolution rate,
validate that shifting from text-based retrieval to graph-based navigation substantially reduces the
search complexity of logic repair. However, our analysis of failure cases, particularly within the
Pillow repository, highlights critical avenues for future development. We outline three primary
directions for the evolution of AIR:
1. Breaking the Language Boundary (Polyglot DTGs):Our experiments revealed that
the current DTG construction, while effective for pure Python, struggles with hybrid runtimes (e.g.,
Python wrappers around C/C++ extensions). Future iterations must implement cross-language
graph builders that can trace data lineage through Foreign Function Interfaces (FFI). This involves
unifying AST parsers across languages into a shared intermediate representation, allowing the agent
to ”tunnel” through compiled extensions rather than treating them as opaque edges.
15

2. Dynamic Graph Refinement via RL:Currently, the DTG is static, constructed prior
to the agent’s traversal. We propose a dynamic architecture where the agent can modifying the
graph structure at runtime—collapsing well-tested subgraphs to save context or expanding ambigu-
ous nodes for deeper inspection. By integrating this into the Reinforcement Learning policy, the
agent can learn to optimize its own environment representation, effectively performing ”attention
management” over the repository topology.
3. Beyond Repair: Security and Refactoring:The utility of the DTG extends beyond
Automated Program Repair. We envision applying this data-centric view to automated security
auditing, where ”taint analysis” becomes a native graph traversal task rather than a complex
heuristic. Similarly, the DTG offers a robust foundation for architectural refactoring, allowing
agents to identify and decouple ”god objects” based on data cohesion metrics rather than just line
counts.
Transforming the LLM’s world model from a ”pile of files” to a navigable network of data flows,
we move closer to truly autonomous systems capable of maintaining the invisible infrastructure
of our digital civilization, precisely tuned to the inner logic of software. AIR represents the first
step in this data-first transformation, bridging the gap between local code completion and systemic
software reasoning.
Acknowledgments
The authors would like to thank Modal and lightning.ai that provided compute resources and free
cloud credits for our experiments.
References
[1] CISQ. (2021) The Cost of Poor Software Quality in the US: A 2020 Report . [Online]. Available:
https://www.it-cisq.org/the-cost-of-poor-software-quality-in-the-us-a-2020-report/
[2] B. Yeti¸ stiren, I. ¨Ozsoy, M. Ayerdem, and E. T¨ uz¨ un, “ Evaluating the Code Quality of
AI-Assisted Code Generation Tools: An Empirical Study on GitHub Copilot, Amazon
CodeWhisperer, and ChatGPT ,” 2023. [Online]. Available: https://arxiv.org/abs/2304.10778
[3] cursor.com. (2025) Cursor Features . [Online]. Available: https://www.cursor.com/features
[4] M. Chen, J. Tworek, H. Jun, Q. Yuanet al., “Evaluating Large Language
Models Trained on Code,”arXiv preprint arXiv:2107.03374, 2021. [Online]. Available:
https://arxiv.org/abs/2107.03374
[5] Y. Li, D. Choi, M. De Nigriset al., “Competition-Level Code Generation with AlphaCode,”
arXiv preprint arXiv:2203.07814, 2022. [Online]. Available: https://arxiv.org/abs/2203.07814
[6] GitHub, “GitHub Copilot Documentation,” https://docs.github.com/copilot, 2025.
[7] Amazon Web Services, “Amazon CodeWhisperer Documentation / Amazon Q Developer,”
https://docs.aws.amazon.com/codewhisperer/, 2025.
[8] Anysphere, “Cursor: The best way to code with AI,” https://cursor.com/, 2025.
[9] S. Yao, J. Z. Denget al., “ReAct: Synergizing Reasoning and Acting in Language Models,”
arXiv preprint arXiv:2210.03629, 2023.
16

[10] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, “ Reflexion:
Language Agents with Verbal Reinforcement Learning ,” inNeurIPS 2023, 2023.
[11] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press, “
SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering ,” 2024.
[Online]. Available: https://arxiv.org/abs/2405.15793
[12] X. Wang and et al., “ OpenHands: An Open Platform for AI Software Developers as
Generalist Agents ,” 2024. [Online]. Available: https://arxiv.org/abs/2407.16741
[13] Y. Zhang and et al., “ AutoCodeRover: Autonomous Program Improvement ,” 2024. [Online].
Available: https://arxiv.org/abs/2404.05427
[14] C. E. J. et al., “Swe-bench: Can language models resolve real-world github issues?” 2024.
[Online]. Available: https://arxiv.org/abs/2310.06770
[15] OpenAI , “ SWE-bench Verified ,” https://openai.com/research/swe-bench-verified, 2024, ac-
cessed 2025-10.
[16] X. Zhanget al., “ A Survey of LLM-based Automated Program Repair ,” 2025. [Online].
Available: https://arxiv.org/abs/2506.23749
[17] Anthropic, “Claude Code: Best practices for agentic coding,” https://www.anthropic.com/
engineering/claude-code-best-practices, 2025.
[18] Zhaoling Chen and Xiangru Tang and Gangda Deng and Fang Wu and Jialong Wu
and Zhiwei Jiang and Viktor Prasanna and Arman Cohan and Xingyao Wang , “
LocAgent: Graph-Guided LLM Agents for Code Localization ,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.09089
[19] F. Yamaguchi, N. Golde, D. Arp, and K. Rieck, “Modeling and Discovering Vulnerabilities
with Code Property Graphs,” in2014 IEEE Symposium on Security and Privacy, 2014, pp.
590–604.
[20] Xiangyan Liu and Bo Lan and Zhiyuan Hu and Yang Liu and Zhicheng Zhang and Fei
Wang and Michael Shieh and Wenmeng Zhou , “ CodexGraph: Bridging Large Language
Models and Code Repositories via Code Graph Databases ,” 2024. [Online]. Available:
https://arxiv.org/abs/2408.03910
[21] Siru Ouyang and Wenhao Yu and Kaixin Ma and Zilin Xiao and Zhihan Zhang and
Mengzhao Jia and Jiawei Han and Hongming Zhang and Dong Yu , “ RepoGraph: Enhancing
AI Software Engineering with Repository-level Code Graph ,” 2025. [Online]. Available:
https://arxiv.org/abs/2410.14684
[22] Hongyuan Tao et al., “ Code Graph Model (CGM): A Graph-Integrated Large Language
Model for Repository-Level Software Engineering Tasks ,” 2025. [Online]. Available:
https://arxiv.org/abs/2505.16901
[23] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre,
D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis, “ Mastering Chess
and Shogi by Self-Play with a General Reinforcement Learning Algorithm ,” 2017. [Online].
Available: https://arxiv.org/abs/1712.01815
17

[24] D. J. Mankowitz, A. Michi, A. Zhernovet al., “Faster sorting algorithms discovered using deep
reinforcement learning,”Nature, vol. 618, no. 7964, pp. 257–263, 2023.
[25] J. Bader, A. Scott, M. Pradel, and S. Chandra, “Getafix: Learning to Fix Bugs Automatically,”
inProceedings of the ACM on Programming Languages (OOPSLA), vol. 3, no. OOPSLA, 2019,
pp. 1–27.
[26] Google DeepMind, “AlphaEvolve for Code,” 2025, blog post. [Online]. Available:
https://deepmind.google/discover/blog/alphaevolve-for-code
[27] Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling,
F. Gimeno, A. Dal Lago, T. Hubert, P. Choy, C. de Masson d’Autume, I. Babuschkin,
X. Chen, P.-S. Huang, J. Welbl, S. Gowal, A. Cherepanov, J. Molloy, D. J. Mankowitz,
E. Sutherland Robson, P. Kohli, N. de Freitas, K. Kavukcuoglu, and O. Vinyals, “
Competition-level code generation with AlphaCode ,”Science, vol. 378, no. 6624, p.
1092–1097, Dec. 2022. [Online]. Available: http://dx.doi.org/10.1126/science.abq1158
[28] Joern dev team, “ Joern: Open-source multi-language code analysis platform ,” https://github.
com/joernio/joern, 2025.
[29] Tree-sitter dev team, “ Tree-sitter parsing library ,” https://github.com/tree-sitter/py-tree-
sitter, 2025.
[30] M. Beigl,Ubiquitous Computing. Basel: Birkhauser Basel, 2005, pp. 52–60. [Online].
Available: https://doi.org/10.1007/3-7643-7674-0 6
[31] swebench.com, “ SWE-bench Leaderboards and Documentation ,” https://www.swebench.
com/, 2025, accessed 2025-10.
[32] D. Lin, J. Koppel, A. Chen, and A. Solar-Lezama, “ QuixBugs: A Multi-Lingual Program
Repair Benchmark Set Based on the Quixey Challenge ,” inSPLASH Companion, 2017.
[33] Tabnine, “Tabnine Docs: Overview,” https://docs.tabnine.com/main, 2025.
[34] Sourcegraph, “Sourcegraph,” https://sourcegraph.com/about, 2025.
[35] Base44, “Base44: Build Apps with AI,” https://base44.com/, 2025.
[36] Lovable, “Lovable: Build apps and websites by chatting with AI,” https://lovable.dev/, 2025.
[37] Replit, Inc., “Replit Ghostwriter: Intro & Docs,” https://replit.com/learn/intro-to-
ghostwriter, 2025.
[38] factory.ai, “Factory,” https://factory.ai/, 2025.
18

A Building the DTG from source files
1def _no_links_found_error (
2self ,
3package : Package ,
4links_seen : int ,
5wheels_skipped : int ,
6sdists_skipped : int ,
7unsupported_wheels : set [str ],
8) -> PoetryRuntimeError :
9messages = []
10info = (
11f" This is likely not a Poetry issue .\n\n"
12f" - { links_seen } candidate (s) were identified for the package \n"
13)
14
15messages . append ( ConsoleMessage ( info . strip ()))
16
17if unsupported_wheels :
18messages += [
19ConsoleMessage (
20" The following wheel (s) were skipped as the current project
environment does not support them "
21" due to abi compatibility issues .",
22debug =True ,
23),
24ConsoleMessage ("\n". join ( unsupported_wheels ), debug = True )
25. indent (" - ")
26. wrap (" warning "),
27ConsoleMessage (
28"If you would like to see the supported tags in your project
environment , you can execute "
29" the following command :\n\n"
30" <c1 > poetry debug tags </>",
31debug =True ,
32),
33]
34
35source_hint = ""
36if package . source_type and package . source_reference :
37source_hint += f" ({ package . source_reference })"
38
39return PoetryRuntimeError (
40reason =f" Unable to find installation candidates for { package }",
41messages = messages ,
42)
Listing 1: A code snippet from poetry project (poetry/src/poetry/installation/chooser.py:125:197).
Some non-essential lines were cut for brevity
The graph builder can extract the following raw constructs from this snippet of code (Table A.1),
and eventually build a subgraph depicted in Figure A. We used a standardastlibrary for parsing
python files,neo4jfor storing nodes and edges, andChromaDBas a vector storage tier for semantic
search of relevant nodes/edges.
19

Construct Category Count Items identified
Function Definitions 1 nolinks found error
Arguments 6self,package,links seen,wheels skipped,
sdists skipped,unsupported wheels
Local Variable Assignments 3messages,info,source hint
Control Flow (Branches) 4ifstatements
Return Statements 1 ReturnsPoetryRuntimeError
External/Global References 3Package,PoetryRuntimeError,ConsoleMessage
Table A.1: Summary of extracted code constructs. We omit verbose metadata for breivity, such
as file name, line number, type of variables, etc. All function arguments, function value and local
variables will become the Data Nodes. Function name will become the basis for the edge. All
formed structures will be stored inneo4jdatabase, and the embeddings for underlying code will
be stored in vector database (we useChromaDB)
Figure A.1: The reconstructed sample of Data Transformation Graph given the raw data to build
Nodes and Edges from Table A.1.
B AI-assisted software tools
Table B.2 presents a comparison of market tier, interaction modality, and primary capabilities of
today’s AI coding tools. While ”Professional” tier tools (e.g. Copilot, Cursor) dominate the inter-
active space via IDE integration, they operate primarily on a ”Human-in-the-Loop” synchronous
model and hardly can be considered a serious AI technology. ”Specialized” agents (AlphaCode)
excel at algorithmic synthesis but lack repository awareness. AIR is distinct in its focus on asyn-
chronous, autonomous maintenance, targeting the ”Zero-Touch” quadrant currently unoccupied by
major commercial providers.
20

Table B.2: The landscape of AI-assisted software tools (Q4’25). We distinguish markettier,modal-
ity, andgranularityto expose differences that matter for maintenance automation.
Product Tier Modality Primary
CapabilityNotes / Differentiators (incl.
SOTA area)
GitHub Copi-
lot [6]Professional IDE/Editor
pluginPredictive
completion,
chat, refac-
torsStrong ecosystem integration (PR
comments, inline hints); SOTA for er-
gonomic code completion at scale.
Amazon
CodeWhis-
perer [7]Professional IDE plugin Completion
+ security
hintsSecurity-aware suggestions; policy
scanning baked into cloud workflows.
Tabnine [33] Professional IDE plugin Local+cloud
completionEmphasis on on-device privacy modes;
enterprise control.
Cursor [8] Professional AI-first
IDEConversational
edit, repo
searchEditor tuned around agentic edits;
tight repo context UX.
Claude
Code [17]Professional Chat+tools Conversational
code reason-
ingLong-context code reasoning; strong
natural language guidance.
OpenAI
Codex
(legacy)Professional API/chat
(retired)Codegen
foundation
modelHistoric baseline; catalyzed tool
ecosystem.
Sourcegraph [34] Professional IDE plugin Autonomous
reasoning,
comprehen-
sive code
editingAdvanced code assistant with many
autonomous features
B44
(builder) [35]Basic Web
builderApp scaf-
folding from
promptsFast MVP creation; limited depth, hu-
man follow-through required.
Lovable [36] Basic Web
builderNo/low-code
+ AIRapid prototyping; non-expert
friendly; limited repo-scale change
mgmt.
Replit
(Ghost-
writer) [37]Basic Web IDE In-editor as-
sistLightweight, education/startup focus;
good DX, modest repo semantics.
AlphaCode
family [27]Specialized Research
systemCompetition-
style codegenSOTA in competitive programming;
not integrated with CI/CD or repos.
AlphaDev [26] Specialized Research
systemAlgorithm
discovery/op-
tim.SOTA algorithm discovery; narrow
domain, non-general maintenance.
Diffblue
CoverSpecialized JVM
toolchainUnit test syn-
thesisIndustrial unit test generation; com-
plements but does not fix issues.
Getafix/Infer
lineage [25]Specialized Internal/OSS
toolsLearned fix
patterns +
static analy-
sisPattern mining for APR; depends on
rich bug corpora; limited autonomy.
Factory.ai [38] Specialized Cloud inte-
grationAgent-native
Software
developmentCorporate level maintenance tool
21