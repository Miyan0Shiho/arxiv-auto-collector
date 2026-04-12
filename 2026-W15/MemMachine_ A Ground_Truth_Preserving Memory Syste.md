# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan

**Published**: 2026-04-06 16:57:06

**PDF URL**: [https://arxiv.org/pdf/2604.04853v1](https://arxiv.org/pdf/2604.04853v1)

## Abstract
Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memory system that integrates short-term, long-term episodic, and profile memory within a ground-truth-preserving architecture that stores entire conversational episodes and reduces lossy LLM-based extraction. MemMachine uses contextualized retrieval that expands nucleus matches with surrounding context, improving recall when relevant evidence spans multiple dialogue turns. Across benchmarks, MemMachine achieves strong accuracy-efficiency tradeoffs: on LoCoMo it reaches 0.9169 using gpt4.1-mini; on LongMemEvalS (ICLR 2025), a six-dimension ablation yields 93.0 percent accuracy, with retrieval-stage optimizations -- retrieval depth tuning (+4.2 percent), context formatting (+2.0 percent), search prompt design (+1.8 percent), and query bias correction (+1.4 percent) -- outperforming ingestion-stage gains such as sentence chunking (+0.8 percent). GPT-5-mini exceeds GPT-5 by 2.6 percent when paired with optimized prompts, making it the most cost-efficient setup. Compared to Mem0, MemMachine uses roughly 80 percent fewer input tokens under matched conditions. A companion Retrieval Agent adaptively routes queries among direct retrieval, parallel decomposition, or iterative chain-of-query strategies, achieving 93.2 percent on HotpotQA-hard and 92.6 percent on WikiMultiHop under randomized-noise conditions. These results show that preserving episodic ground truth while layering adaptive retrieval yields robust, efficient long-term memory for personalized LLM agents.

## Full Text


<!-- PDF content starts -->

MemMachine: A Ground-Truth-Preserving Memory System
for Personalized AI Agents
Shu Wang∗1, Edwin Yu†1, Oscar Love‡1, Tom Zhang§1, Tom Wong¶1, Steve Scargall‖1, and
Charles Fan∗∗1
1MemVerge, Inc.
March 2026
Abstract
Large language model (LLM) agents require persis-
tent memory to maintain personalization, factual con-
tinuity, and long-horizon task performance, yet stan-
dard context-window and retrieval-augmented gener-
ation (RAG) workflows remain brittle under multi-
session interactions. We present MemMachine, an open-
source memory system that combines short-term mem-
ory, long-term episodic memory, and profile memory in
a ground-truth-preserving architecture that stores raw
conversational episodes and minimizes routine LLM-
based extraction. MemMachine introduces contextual-
ized retrieval that expands nucleus matches with neigh-
boring episode context, improving recall for conversa-
tional queries where semantically related evidence is dis-
tributed across turns.
Across multiple benchmarks, MemMachine demon-
strates strong accuracy-efficiency tradeoffs. On Lo-
CoMo, it achieves an overall score of 0.9169 with gpt-
4.1-mini. OnLongMemEval S(ICLR2025), asystematic
ablation study across six optimization dimensions yields
93.0%overall accuracy, with retrieval-stage optimiza-
tions contributing more than ingestion-stage changes:
retrieval depth tuning (+4.2%), context formatting
(+2.0%), search prompt design (+1.8%), and query
bias correction (+1.4%) each outweigh sentence chunk-
ing (+0.8%). A surprising finding is that GPT-5-mini
outperforms GPT-5 as the answer LLM (+2.6%) when
paired with optimized prompts, yielding the most cost-
efficient configuration. In matched memory-mode com-
parisons, MemMachine uses approximately 80% fewer
input tokens than Mem0. We further evaluate a Re-
trieval Agent that routes queries to direct retrieval, par-
allel decomposition, or iterative chain-of-query strate-
∗shu.wang@memverge.com
†edwin.yu@memverge.com
‡oscar.love@memverge.com
§tom.zhang@memverge.com
¶tom.wong@memverge.com
‖steve.scargall@memverge.com
∗∗charles.fan@memverge.comTable 1: Benchmark snapshot (quick overview). De-
tailed setup and full breakdowns are provided in later
sections.
Benchmark Metric Result
LoCoMo Overall score (gpt-4.1-mini)91.69%
LongMemEval SAblation best (gpt-5-mini)93.0%
HotpotQA hard Retrieval Agent accuracy93.2%
WikiMultiHop†Retrieval Agent accuracy92.6%
Mem0 comparison Input token reduction~80% less
†Randomized-noise setting.
gies, reaching 93.2% on HotpotQA hard and 92.6% on
WikiMultiHop under randomized-noise conditions.
These results suggest that preserving episodic ground
truth while layering adaptive retrieval strategies yields
robust long-term memory behavior for personalized
agents.
1 Introduction
Transformer-based large language models (LLMs) have
become the computational foundation for a rapidly
growing class of autonomous AI applications, from con-
versational assistants and customer-facing agents to
complex multi-agent workflows [1, 2]. Despite their
broad capabilities, LLMs exhibit two fundamental limi-
tations that constrain their utility in persistent, person-
alized applications:
1.Static Parameters.Once trained, an LLM’s
weights are fixed. The model cannot acquire new
knowledge from interactions without costly fine-
tuning or re-training.
2.Restricted Context Window.LLMs operate
within a finite context window, requiring applications
to carefully curate and compress inference data, often
at the cost of losing relevant historical context.
Retrieval-Augmented Generation (RAG) [3] has
emergedasthedominantparadigmforinjectingexternal
knowledge into LLM workflows. However, conventional
RAG architectures are designed for static document col-
1arXiv:2604.04853v1  [cs.AI]  6 Apr 2026

lections and do not support the dynamic, bidirectional
interactions characteristic of AI agents that must learn
from, and adapt to, evolving user contexts across ses-
sions.
What AI agents require is amemory system—a mech-
anism that goes beyond document retrieval to store, or-
ganize, recall, and reason over past experiences. Draw-
ingonestablishedmodelsfromcognitivescience[10,11],
such a system should provide:
•Short-term memory (STM):An immediate
workspace maintaining current context, with limited
capacity.
•Long-term episodic memory:A store of specific
past experiences, providing ground truth about what
occurred.
•Semantic memory:High-level summaries, facts,
and user profiles distilled from raw experience.
•Procedural memory:Learned rules, strategies,
and action patterns that guide agent behavior.
A growing body of systems have begun to address
this challenge. MemGPT [1] explored virtual memory
management for LLMs. Mem0 [4] and Zep [5] provide
long-term memory layers for AI agents. However, these
systems primarily rely on LLMs for data extraction,
update, aggregation, and deletion—a design that in-
troduces high operational cost, accuracy concerns from
probabilistic extraction, and compounding error over
time.
1.1 Contributions
In this paper, we presentMemMachine, an open-
source memory system that takes a fundamentally dif-
ferent approach. Our key contributions are:
1.Ground-truth-preserving architecture.Mem-
Machine stores raw conversational episodes and in-
dexes them at the sentence level, minimizing LLM
dependence for routine memory operations and pre-
serving factual integrity.
2.Contextualized retrieval.A novel retrieval mech-
anism that expands nucleus episodes with neighbor-
ing context to form episode clusters, addressing the
embedding dissimilarity problem inherent in conver-
sational data.
3.Cost-efficient operation.By reserving LLM calls
for summarization and high-level abstraction rather
than per-message extraction, MemMachine achieves
approximately 80% reduction in token usage com-
pared to competing systems.
4.Personalization support.A profile memory sys-
tem that extracts and maintains user preferences,
facts, and behavioral patterns to enable personalized
agent interactions across sessions.
5.Leading LoCoMo performance (as of March
2026).MemMachine achieves 0.9169 on LoCoMo
with gpt-4.1-mini, among the strongest published re-sults for open memory frameworks and above re-
ported Mem0, Zep, Memobase, LangMem, and Ope-
nAI baseline scores.
6.LongMemEval Sablation study.A system-
atic evaluation on LongMemEval (ICLR 2025)
across six optimization dimensions—sentence chunk-
ing, query bias correction, context formatting, re-
trieval depth, search prompt design, and answer-
model selection—achieving 93.0% overall accuracy
and revealing that retrieval-stage optimizations dom-
inate over ingestion-stage changes.
7.Comprehensive evaluation.We provide repro-
ducible benchmark scripts and analyze the impact of
embedding models, reranking strategies, LLM model
selection, and retrieval parameters on memory per-
formance.
8.Retrieval Agent for multi-hop reasoning.An
LLM-orchestrated retrieval pipeline that classifies
queries into structural types and routes them to
purpose-built strategies (direct search, parallel de-
composition, or iterative chain-of-query), achieving
93.2% accuracy on the HotpotQA hard set while
maintaining bounded cost.
MemMachine is released under the Apache 2.0 license
and is available athttps://github.com/MemMachine/
MemMachine.
2 Related Work
2.1 Memory for AI Agents
The need for persistent memory in LLM-based agents
has been recognized across multiple research threads.
Hu et al. [9] provide a comprehensive survey organiz-
ing agent memory by forms (token-level, parametric, la-
tent), functions (factual, experiential, working), and dy-
namics (formation, evolution, retrieval). Park et al. [2]
demonstrated the power of memory in generative agents
that simulate human behavior, using a memory stream
architecture with retrieval, reflection, and planning.
MemGPT [1] introduced an operating-system-
inspired virtual memory hierarchy for LLMs, managing
context by paging information between a main context
and external storage. While pioneering, MemGPT’s
approach requires complex memory management that
can introduce latency and depends on LLM-driven
decisions for memory operations.
2.2 Existing Memory Systems
Mem0[4] provides a production-oriented memory layer
that extracts facts from conversations using LLM calls,
stores them in hybrid vector and graph databases, and
retrieves them for agent inference. While effective, the
per-message LLM extraction approach incurs significant
cost and can introduce factual drift through accumu-
lated extraction errors.
2

Zep[5] implements a temporal knowledge graph ar-
chitecture that tracks how facts evolve over time, com-
bining graph-based memory with vector search. Zep
excels at relationship modeling and temporal reasoning
but introduces complexity in deployment and configura-
tion.
Memobase1andLangMem2represent additional
approaches, with Memobase providing structured mem-
ory storage and LangMem integrating memory into the
LangChain ecosystem.
More recently,Mastra[15] introducedobservational
memory, which uses two background agents (Observer
and Reflector) to compress conversation history into a
dated observation log that stays in context, eliminating
retrieval entirely. This approach achieves strong Long-
MemEval scores (94.87% with GPT-5-mini) and enables
aggressive prompt caching, but trades away the ability
to search a broader external corpus—making it less suit-
able for open-ended knowledge discovery or compliance-
heavy recall use cases where ground truth access is re-
quired.
MemOS[16] takes the most ambitious architectural
stance, proposing a fullmemory operating systemthat
treats memory as a first-class schedulable resource anal-
ogous to CPU or storage in traditional operating sys-
tems. MemOS unifies three memory types under a sin-
gle abstraction calledMemCube:plaintext memory(ex-
ternally injected text, akin to RAG),activation mem-
ory(KV-cache states from inference), andparametric
memory(knowledge embedded in model weights, e.g.,
LoRA adapters). MemCubes carry rich metadata in-
cluding provenance, versioning, access policies, and life-
cyclestate, enablingcross-typetransformations—forex-
ample,promotingfrequentlyaccessedplaintextintoKV-
cache templates for faster inference, or distilling stable
knowledge into parametric weights. The system imple-
ments a three-layer architecture (interface, operation,
infrastructure) with a memory scheduler that orches-
trates predictive preloading and multi-user session man-
agement. On LoCoMo, MemOS reports 75.80 using
GPT-4o-mini as the processing LLM, with a claimed
159%improvementintemporalreasoningoverOpenAI’s
global memory.
While MemOS represents a compelling long-term vi-
sion—particularly its memory lifecycle governance and
cross-type transformation pathways—its scope is sig-
nificantly broader than the other systems discussed
here. The parametric and activation memory types
require direct access to model internals (weights, KV-
caches), which limits portability across LLM providers
and closed-source APIs. By contrast, MemMachine,
Mem0, Zep, and Mastra operate at theapplication
layer, interfacing with LLMs through standard text-
basedAPIs, whichenablesthemtoworkwithanymodel
1https://github.com/memodb-io/memobase
2https://github.com/langchain-ai/langmemprovider without modification.
These diverse approaches highlight a fundamental de-
sign tension in agent memory:compression vs. preser-
vation. Systems that aggressively compress (Mastra,
Mem0) achieve smaller context windows and lower per-
query cost but risk losing critical detail. Systems that
preserve raw data (MemMachine) maintain factual in-
tegrity at the cost of requiring efficient retrieval mecha-
nisms. A related tension exists betweenretrieval-based
approaches (MemMachine, Mem0, Zep) that search se-
lectively andcontext-basedapproaches (Mastra) that
keep compressed history always in context. MemOS
introduces a third dimension:memory-layer depth,
spanning from application-level text memory through
inference-level activation caching to model-level para-
metric adaptation. Table 16 compares these architec-
tural choices.
2.3 Memory Benchmarks
Theevaluationlandscapeforagentmemoryhasmatured
significantly.LoCoMo[6] evaluates very long-term
conversational memory through multi-session dialogues
with single-hop, multi-hop, temporal, and open-domain
question types.LongMemEval[7] benchmarks
five core long-term memory abilities—information ex-
traction, multi-session reasoning, temporal reasoning,
knowledge updates, and abstention—using scalable chat
histories.EpBench[8] focuses on episodic memory
evaluation through synthetic narrative corpora ranging
from 100K to 1M tokens.
2.4 Memory Types in Cognitive Science
Our design draws on the established taxonomy from
cognitive science. Tulving [10] distinguishedepisodic
memory(memoryofspecificpersonalexperiencesbound
to time and place) fromsemantic memory(general
knowledge abstracted from experience). The Atkinson-
Shiffrin model [11] formalized the multi-store architec-
ture with sensory, short-term, and long-term stores. In
the AI agent context, we adopt these distinctions while
acknowledging that current implementations approxi-
mate rather than replicate human memory processes.
3 Memory Types for AI Agents
AI agent memory systems draw inspiration from cogni-
tive science while adapting to the practical requirements
of LLM-based applications. This section describes the
primary memory types and their roles, with emphasis
on those implemented in MemMachine.
3.1 Episodic Memory
Episodic memory stores specific past experiences—what
happened,when,where, and withwhom. In the agent
3

context, each conversational turn or interaction consti-
tutes anepisode, a discrete unit of experience with asso-
ciated metadata (timestamp, participants, session iden-
tifier).
Episodic memory serves as ground truth. When an
agentneedstorecallwhatausersaid, whatwasdecided,
or what sequence of events occurred, it queries episodic
memory for the raw record. This is essential for factual
accuracy, auditability, and trust.
When to use:Factual recall, reconstructing conver-
sation history, answering questions about specific past
interactions, providing evidence for decisions, and main-
taining conversational continuity across sessions.
3.2 SemanticMemory(ProfileMemory)
Semantic memory stores generalized knowledge ab-
stracted from episodic experience—user preferences,
facts, behavioral patterns, and stable attributes. In
MemMachine, this is implemented asProfile Mem-
ory, which extracts and maintains structured user pro-
files from conversational data.
Unlike episodic memory, which preserves the raw
record, semantic/profile memory distills high-level pat-
terns: “The user prefers vegetarian restaurants,” “The
user works in financial services,” or “The user’s preferred
communication style is concise and technical.”
Whentouse:Personalization, preference-awarerec-
ommendations, adapting tone and content, and provid-
ing context that does not require recalling a specific
episode.
3.3 Procedural Memory
Procedural memory encodes learned skills, strategies,
and behavioral rules—howto do things. In agent sys-
tems, this includes tool-use patterns, workflow steps,
anddecisionheuristics. MemMachinedoesnotcurrently
implement procedural memory, though the architecture
can be extended to support it.
When to use:Multi-step task execution, tool selec-
tion, workflow automation, and strategy reuse.
3.4 Temporal Awareness
While not a separate memory type, temporal aware-
ness is a cross-cutting capability. MemMachine tags
all episodes with timestamps and supports temporal fil-
tering during search, enabling agents to reason about
event ordering, recency, and duration. This provides
limited but valuable temporal reasoning without requir-
ing a dedicated temporal memory module.3.5 Episodic vs. Semantic: When to Use
Each
The choice between episodic and semantic retrieval de-
pends on the query type:
Table 2: Episodic vs. Semantic Memory Usage
Criterion Episodic Semantic
Query type Specific past
eventsGeneral prefer-
ences
Accuracy need Ground truth Approximate
Temporal
scopePoint-in-time Cross-session
Data form Raw conversation Extracted facts
Example “What did I say
about X?”“What foods do I
like?”
In practice, effective agents combine both: episodic
memory for factual grounding and semantic/profile
memory for personalization.
4 MemMachine Architecture
MemMachine implements a client-server architecture
with a two-tier memory system comprising episodic
memory (short-term and long-term) and profile memory
(semantic). This section describes the system’s compo-
nents, data flow, and key design decisions.
4.1 System Overview
Figure 1 illustrates the high-level architecture. Agents
interact with MemMachine through three API inter-
faces: a RESTful API (v2), a Python SDK, and a
ModelContextProtocol(MCP)server. Theserverman-
ages memory processing, while the storage layer per-
sists data across PostgreSQL (with pgvector for vector
search), SQLite, and Neo4j (for graph-structured long-
term memory).
4.2 Data Ingestion
Raw messages are submitted to MemMachine with
metadata. The system organizes each message into an
internaldatastructurecalledanEpisode. Eachepisode
represents one conversational turn and carries required
metadata:
•Producer:The source of the message (user, agent,
system).
•Timestamp:When the message was produced.
•Session ID:Grouping episodes into conversational
sessions.
•Custom metadata:Arbitrary key-value pairs for
domain-specific filtering.
Episodesarestoredinacentraldatabaseasarawdata
repository and simultaneously dispatched to episodic
4

MEMMACHINE
Episodic Memory Semantic MemoryAI Agent Python SDK MCP Server
REST API Python SDK
Working
Memory
(short-term)Persistent
Memory
(long-term)Profile
Memory
SQL Vector GraphDatabases
Figure 1: MemMachine system architecture. Clients
(AI agents, Python SDK, MCP servers) access Mem-
Machine through REST API and SDK interfaces. In-
ternally, episodic memory comprises working memory
(short-term) and persistent memory (long-term), while
semantic memory stores user profiles. All memory types
are backed by SQL, vector, and graph databases.
memory and profile memory for ingestion and indexing.
4.3 Short-Term Memory
Short-term memory (STM) maintains a configurable
context window of the most recent episodes, providing
immediate conversational context. Key behaviors:
•Holds a predefined number of recent episodes.
•Generates compressed summaries of session-level in-
teractions via LLM-based abstraction.
•Whencontentexceedsthewindow, bothepisodesand
summaries are compressed for efficient storage and
eventually transferred to long-term memory.
STM ensures that agents always have access to the
immediate conversational context without requiring a
retrieval step, while the summarization mechanism pre-
serves the gist of older context within the window.
4.4 Long-Term Memory
Long-term memory (LTM) provides persistent, search-
able storage for all episodes that have exited the STM
window. The indexing pipeline comprises four stages:
1.Sentence Extraction:Each episode is segmented
into individual sentences using NLTK’s Punkt tok-
enizer [12]. This fine-grained decomposition enables
precise embedding and retrieval at the sentence level
rather than the episode level.
2.Metadata Augmentation:Each sentence inherits
metadata from its parent episode (timestamp, pro-User Query + Filters
STM Search
LTM Vector Search
Contextualization
De-duplicate Episodes
Rerank Clusters
Sort Chronologically
Return Results
Figure 2: Memory recall workflow. The query passes
through STM, LTM vector search, contextualization,
deduplication, reranking, and chronological sorting be-
fore returning results.
ducer, session) and receives a unique identifier.
3.Relational Mapping:Sentences are linked to their
originating episodes, maintaining provenance.
4.Embedding Generation:Semantic embeddings
are generated for each sentence. MemMachine
supports configurable embedding models, enabling
domain-specific models for improved performance.
The original episodes, augmented sentences, and em-
beddings are persisted in the database. Neo4j provides
graph-based storage that enables relational traversal,
while PostgreSQL with pgvector supports efficient vec-
tor similarity search.
4.5 Memory Search and Recall
Memory search in MemMachine follows a staged re-
call pipeline that balances speed, coverage, and factual
grounding. The system first checks near-term context,
then expands into long-term episodic retrieval, and fi-
nally refines candidates before returning results. Fig-
ure 2 (Figure 2) summarizes this end-to-end workflow.
4.5.1 Long-Term Memory Search
LTM search begins with a vector similarity search using
Approximate Nearest Neighbor (ANN) or exact match-
ingagainstthesentenceembeddings. Matchedsentences
are traced back to their originating episodes, with du-
plicates removed. The system then appliescontextual-
ization(Section 4.6).
4.5.2 Episodic Memory Search
Episodic search coordinates STM and LTM. It invokes
LTMsearchforepisodesoutsidetheSTMwindow,dedu-
5

plicates overlapping episodes between STM and LTM,
sorts all results chronologically to preserve conversa-
tional flow, and returns STM episodes, the STM sum-
mary, and LTM episodes for agent-driven context as-
sembly.
4.6 Contextualization
A key challenge in conversational memory retrieval is
that contextually important episodes may have embed-
dings quite dissimilar from the query. Unlike document
retrieval in traditional RAG, where each chunk is rel-
atively self-contained, conversational turns are highly
interdependent. A question about “the restaurant rec-
ommendation” requires not just the turn containing the
recommendation, but the surrounding turns that estab-
lish what was asked, why, and what constraints were
given.
MemMachine addresses this withcontextualized
retrieval:
1. Thenucleus episodeis located via embedding
search.
2. Immediateneighboringepisodesareretrieved(one
preceding, two following) to form anepisode clus-
ter.
3. Episode clusters arererankedusing a cross-encoder
or other reranking model.
4. The top-kclusters are provided for LLM inference.
This approach ensures that the LLM receives not just
the most semantically similar turns, but the conversa-
tional context necessary for accurate reasoning. Our
experimental results (Section 8) demonstrate the signif-
icant impact of contextualization on benchmark perfor-
mance.
4.7 ProfileMemory(SemanticMemory)
Profile memory extracts and maintains user-level facts
and preferences from conversational data. Unlike
episodic memory, which preserves raw interactions, pro-
file memory synthesizes high-level user attributes:
•Demographic information volunteered by the user.
•Stated preferences and interests.
•Behavioral patterns observed across sessions.
•Professional context and domain expertise.
Profile memory is stored in SQL databases (Post-
greSQL or SQLite) and supports both retrieval and up-
date operations. When new information contradicts ex-
isting profile data, the system can update the profile
to reflect the most recent state—supporting theknowl-
edge updatecapability evaluated in benchmarks such as
LongMemEval [7].
4.8 Multi-Tenancy and Isolation
MemMachine implements project-based names-
pace isolation. Each project is identified by anorg_id/project_idpair and maintains separate mem-
ory instances. Sessions are further isolated byuser_id,
agent_id, andsession_id, enabling multi-tenant de-
ployments where multiple users and agents operate with
fully isolated memory stores.
5 Retrieval Agent
MemMachine’s baseline retrieval mechanism—vector
similaritysearchwithoptionalreranking—performswell
on single-hop queries where the answer resides in a sin-
gle episode cluster. However, production AI agents rou-
tinely encounter queries that requiremulti-hop reason-
ing,multi-entity fan-out, orcross-referential dependency
chains. For such queries, a single embedding cannot
capture all the information needed because intermedi-
ate entities are unknown at query time. We call this the
late binding problem: the correct query for a later
retrieval hop cannot be formulated until an earlier hop
has been resolved.
Toaddressthis,MemMachinev0.3introducestheRe-
trieval Agent—an opt-in, LLM-orchestrated retrieval
pipeline that routes queries to purpose-built strate-
gies while maintaining bounded cost and latency. The
Retrieval Agent augments (not replaces) the baseline
search: callers who do not enable agent mode experi-
ence zero behavioral change.
5.1 The Late Binding Problem
Consider a multi-hop query:“What is the current em-
ployer of the spouse of the CEO of Acme?”Answering
this requires three ordered resolution steps: (1) identify
the CEO of Acme→Person X, (2) identify the spouse of
Person X→Person Y, (3) identify the employer of Per-
son Y→Company Z. At query time, only the original
query string is available. Its embedding clusters around
surface terms (“Acme,” “CEO,” “company”) and has no
path to Company Z because the intermediate entities
are unknown. This is not a limitation of the embedding
model—it is a structural property of information depen-
dencies in multi-hop chains that no single-shot vector
retrieval can resolve.
Existing mitigation strategies—query expansion
(HyDE), BM25 hybrid search, chunk-level rerank-
ing—improve single-hop recall but cannot resolve
dependency chains because they still operate on a
single query formulation. Knowledge graph traversal
solves late binding exactly but requires prior graph
construction, which is expensive and lossy for arbitrary
conversational content.
5.2 Architecture
The Retrieval Agent is implemented as acomposable
tool treeassembled inside MemMachine’s long-term
memory module. Figure 3 shows the architecture: a
6

ToolSelectAgent
(LLM query router)
ChainOfQuery
Agent
(multi-hop)SplitQuery
Agent
(fan-out)MemMachine
Agent
(direct)
DeclarativeMemory
(vector search + reranker)routes to exactly one
Figure 3: Retrieval Agent tool tree. The ToolSelectA-
gent classifies each query and routes it to one of three
strategies. Allstrategiesultimatelydelegatetothesame
declarative memory search, ensuring that index and
reranker improvements propagate automatically.
root router dispatches each query to exactly one of three
strategy nodes, all of which ultimately delegate to the
same declarative memory search.
Design principles.All three strategy nodes dele-
gate to the sameMemMachineleaf, ensuring that im-
provements to the underlying vector search propagate
automatically to all strategies. The tree is constructed
once at startup and cached; routing decisions are made
per-query at inference time. Agent mode is enabled
via a single flag (agent_mode=true), with zero behav-
ioral change to existing callers. Each node exposes ad-
visory cost properties (accuracy_score,token_cost,
time_cost) for future budget-aware routing.
5.3 Query Routing
TheToolSelectAgent(root node) uses a single LLM
call to classify each incoming query into one of three
structural types:
•Multi-hop dependency chain:Two or more se-
quentially dependent retrieval steps where a later
step requires the result of an earlier one. Routed to
ChainOfQuery.
•Single-hop multi-entity:Multiple independent
subjects answerable via parallel lookups with no
inter-dependency. Routed toSplitQuery.
•Single-hop direct:Single subject, single lookup, no
decomposition needed. Routed toMemMachine(base-
line search with only the routing overhead).
The classification prompt uses embedded calibration
examples and a tie-breaker rule: if any explicit depen-
dency chain exists, classify as multi-hop even if multi-
ple entities appear. This conservative bias trades extra
LLM calls for higher recall completeness.
We further tuned Retrieval Agent prompts using
the APO (Auto Prompt Optimization) algorithm from
Agent Lightning [18], using the Microsoft APO imple-
mentation3. Inourinternalablations, tuningonlythefi-
3https://microsoft.github.io/agent-lightning/latest/
algorithm-zoo/apo/nal answer prompt improved accuracy by approximately
4% (with baseline prompt quality also improving), while
jointly tuning all agent prompts improved accuracy by
approximately 6%. We do not perform live tuning at
inference time, so these gains add no runtime token or
latency overhead.
5.4 Strategy Details
ChainOfQueryimplementsiterativeevidenceaccumu-
lation for dependency chains. It executes up to 3 itera-
tions, eachconsistingof: (1)retrievalagainstthecurrent
query, (2) a combined sufficiency judgment and query
rewrite via a single LLM call, and (3) evidence accumu-
lation. The sufficiency prompt enforces evidence-only
judgment (no external knowledge), strict completeness
standards, entity grounding (rewrites use only entities
present in retrieved evidence), and calibrated confidence
scoringwithearlystoppingat≥0.8confidence. Thisde-
sign, grounded in the prompt engineering methodology
of Luo et al. [18], formalizes forward-chaining multi-hop
reasoning with bounded cost.
SplitQueryaddresses fan-out queries by decompos-
ing them into 2–6 independent sub-queries via a sin-
gle LLM call, executing all sub-queries concurrently
viaasyncio.gather(), and pooling the results. The
decomposition prompt enforces structural constraints:
sub-queries must each be answerable by a single fact
lookup, derived operations (compare, rank, difference)
are prohibited, and a conservative tie-breaker defaults
to no-split when ambiguous.
MemMachineis the direct retrieval leaf that
callsDeclarativeMemory.search_scored()without
any query transformation. It serves both as the leaf
primitive for other agents’ child calls and as the strategy
for simple single-hop queries, where agent mode incurs
only one routing LLM call overhead.
5.5 Multi-Query Reranking
A key innovation in agent mode ismulti-query
reranking: the final reranker receives a concatenation
ofallqueries used during retrieval (original query plus
all rewrites or sub-queries), not just the original. This
ensuresthatepisodesrelevanttoanystepintheretrieval
chain—including intermediate facts that are critical for
multi-hop reasoning but not directly referenced in the
original query—score well in the final ranking.
5.6 Benchmark Results
WeevaluatetheRetrievalAgentacrossfivebenchmarks,
comparing three modes:Baseline(LLM with full con-
text, no memory),MemMachine(declarative mem-
ory search), andRetrieval Agent(agent-orchestrated
search). Table 3 summarizes accuracy and recall across
all benchmarks.
7

Table 3: Retrieval Agent benchmark results across five datasets. Accuracy is LLM-judge score; Recall measures gold
supporting fact retrieval. Best result per benchmark inbold. All MemMachine and Agent results use surrounding
episodes enabled unless noted.
Benchmark Questions Answer LLM MemMachine Retrieval Agent Baseline
Acc. Recall Acc. Recall Acc.
LoCoMo 1,540 gpt-5-mini 90.5% 82.2%90.2% 82.5%91.7%
WikiMultiHop 500 gpt-4.1-mini 88.8% 90.8%90.0% 92.5%96.7%
WikiMultiHop†500 gpt-5-mini 87.4% 91.7%92.6% 83.3%96.7%
MRCR 300 gpt-5.2 79.6% 99.1%81.4% 99.4%32.3%
EpBench 546 gpt-5-mini 73.4% 66.1% 71.8% 65.3% 77.5%
EpBench 546 gpt-4o-mini 71.4% 67.9%73.3% 69.7%50.1%
HotpotQA 500 gpt-5-mini 91.2% 91.0%93.2% 95.5%93.0%
†WikiMultiHop fully randomized cross-question noise injection; all others with content in order.
Table 4: HotpotQA hard set: per-tool performance
breakdown for the Retrieval Agent (n=500, gpt-5-mini).
Tool Questions Accuracy Recall
MemMachine 201 93.53% 89.31%
SplitQuery 118 94.07% 92.83%
ChainOfQuery 181 92.27%95.31%
Overall (Agent) 50093.20% 92.31%
HotpotQAprovides the clearest demonstration of
the Retrieval Agent’s value. On the hard validation set
(500 questions), the agent achieves 93.2% overall accu-
racy and 92.31% gold-supporting-fact recall, compared
to 91.2% accuracy and 90.98% recall for baseline Mem-
Machine—a +2.0 and +1.3 percentage point improve-
ment respectively. The per-tool breakdown (Table 4)
shows thatChainOfQueryachieves the highest recall
(95.31%) on multi-hop bridge questions, validating the
iterative evidence accumulation strategy.
WikiMultiHopdemonstrates the benefit of agent
orchestration under realistic noise conditions. When
all question contexts are pooled into a single
shared episodic store with fully randomized order-
ing—simulating a production setting where relevant
and irrelevant memories are interleaved—the Retrieval
Agent with gpt-5-mini achieves 92.6% accuracy vs.
87.4% for baseline MemMachine, a +5.2 point improve-
ment.
MRCR(Multi-Round Co-reference Resolution)
shows consistent agent improvement: 81.4% vs. 79.6%
for MemMachine, with near-perfect recall (99.4%). No-
tably, the LLM baseline without memory scores only
32.3%, demonstrating that co-reference resolution fun-
damentally requires memory retrieval.
LoCoMoshows comparable performance between
MemMachine (90.5%) and the Retrieval Agent (90.2%).
This is expected: LoCoMo is predominantly single-hop
conversational questions where baseline vector search is
already effective, and the agent’s routing overhead pro-Table 5: Average token cost per question by Retrieval
Agent strategy (HotpotQA hard set).
Component Input Tokens Output Tokens
ToolSelect (router) 1,049 195
ChainOfQuery 2,874 1,614
SplitQuery 1,229 435
MemMachine 0 0
vides minimal benefit.
EpBenchresults are mixed and benchmark-
dependent. With gpt-4o-mini as the answer model, the
Retrieval Agent improves over baseline MemMachine
(73.3% vs. 71.4%). With gpt-5-mini, baseline MemMa-
chine slightlyoutperforms (73.4% vs. 71.8%), suggesting
model-prompt sensitivity.
5.7 Token Cost Analysis
The Retrieval Agent’s improved recall comes at the cost
of additional LLM calls for routing and strategy exe-
cution. Table 5 shows the per-strategy token cost on
HotpotQA.
For the 36% of queries routed directly toMemMachine,
the total agent overhead is only the routing call
(∼1,244 tokens). For multi-hop queries routed to
ChainOfQuery, total cost reaches∼5,732 tokens per
question—substantially more, but bounded by the 3-
iteration limit. This bounded cost profile is a deliber-
ate improvement over the predecessor unbounded agent
loop (OpenAI Agents SDK withmax_turns=30), which
was retired in favor of architecturally-defined retrieval
strategies.
5.8 When to Use Agent Mode
Agent mode is not universally beneficial. Table 6 pro-
vides deployment guidance based on query characteris-
tics.
The system is designed for seamless coexistence:
8

Table 6: Decision guidance for enabling agent mode.
Use Agent
ModeWhen...
Best fitQueries involve multi-
hop reasoning, co-
reference resolution, or
multi-entity fan-out.
Accuracy is prioritized
over latency. Use
cases include research
assistants, compliance
review, and complex
QA.
Not neededQueries are predomi-
nantly single-hop fac-
tual lookups. Latency
requirements are strict
(<200ms). Token bud-
get is constrained. Use
cases include simple
chatbot recall, prefer-
ence lookup, and ses-
sion context.
agent mode is enabled per-query via a single flag,
and the router itself filters simple queries to the zero-
overhead direct path. In production, applications can
enable agent mode selectively based on query complex-
ity heuristics or always-on with acceptable overhead for
the majority of single-hop queries that route directly.
5.9 OpenClaw Integration
The Retrieval Agent is also available throughOpen-
Claw4, an open-source AI agent framework. The
MemMachine OpenClaw integration is implemented as
a standard plugin using the OpenClaw Plugin SDK
(openclaw/plugin-sdk), not OpenClaw “agent mode.”
Concretely, the plugin imports SDK types and ex-
poses the canonical plugin object withregister(api:
OpenClawPluginApi), where lifecycle hooks and tool
registrations connect MemMachine capabilities to the
host application. Through this plugin interface, appli-
cations can use both declarative memory search and
Retrieval Agent orchestration via a unified tool sur-
face. Benchmark results in Table 3 include OpenClaw-
based evaluation configurations, demonstrating that
the Retrieval Agent’s benefits transfer across agent
frameworks—not just MemMachine’s native evaluation
harness.
4https://github.com/openclaw6 LLM Integration and Model
Impact
6.1 How LLMs Are Used in MemMa-
chine
AdistinguishingdesignprincipleofMemMachineisthat
LLMs are used sparingly and strategically, rather than
for every memory operation. Specifically, LLMs serve
three functions:
1.STM Summarization:When the short-term mem-
ory window overflows, an LLM generates a com-
pressed summary of the session context.
2.Profile Extraction:An LLM extracts structured
user facts and preferences from conversational data
for profile memory.
3.Agent-Mode Inference:When operating in agent
mode, the eval-LLM can iteratively query MemMa-
chine’s memory as a tool to formulate responses.
Critically, LLMs arenotused for per-message fact
extraction, memory deduplication, or routine memory
management—operations that in competing systems ac-
count for the majority of LLM token consumption.
6.2 Impact of Model Choice on Perfor-
mance
The choice of LLM significantly affects both benchmark
performance and operational cost. Table 7 summarizes
the impact of eval-LLM selection on LoCoMo scores.
Table 7: Impact of eval-LLM on MemMachine v0.2 Lo-
CoMo scores.
Configuration LLM Score Mode
gpt-4.1-mini (agent) 0.9169 Agent
gpt-4.1-mini (memory) 0.9123 Memory
gpt-4o-mini (agent) 0.8812 Agent
gpt-4o-mini (memory) 0.8747 Memory
Thetransitionfromgpt-4o-minitogpt-4.1-miniyields
a 3–4 percentage point improvement across both modes,
demonstrating that newer LLM generations improve
memory-augmented reasoning without any changes to
the memory system itself. This also suggests that Mem-
Machine’s architecture isLLM-agnostic—performance
improvements in the underlying model translate directly
to improved memory-augmented agent behavior.
6.3 Cost Considerations
Token usage is a primary cost driver for LLM-based ap-
plications. MemMachine’s architecture substantially re-
duces token consumption:
9

Table 8: Token usage comparison on LoCoMo (gpt-4.1-
mini).
System Input Tokens Output Tokens
MemMachine
v0.2 (mem-
ory)4.20M 43,169
MemMachine
v0.2 (agent)8.57M 93,210
Mem0
main/HEAD
(memory)19.21M 14,840
MemMachine uses approximately 78% fewer input to-
kens than Mem0 in memory mode, translating directly
to lower inference cost and reduced time-to-first-token
latency.
6.4 Context Window Considerations
A natural question is whether memory systems pro-
vide benefit when conversational content fits entirely
within the LLM’s context window. Evidence from
the LoCoMo benchmark suggests that even for con-
versations within context window limits (16K–26K to-
kens), memory-augmented systems outperform raw full-
context baselines [6]. This is because:
•Full-context approaches suffer from the “lost in the
middle” effect [13], where information in the middle
of long contexts is poorly attended.
•Memory systems provideselective retrieval, surfac-
ing only the most relevant episodes rather than over-
whelming the model with full conversation history.
•As conversations grow beyond context limits, mem-
ory systems become essential rather than merely ben-
eficial.
7 Experimental Setup
7.1 Benchmarks
We evaluate MemMachine on the following benchmarks:
LoCoMo[6]: Evaluates very long-term conversa-
tional memory across four question categories: single-
hop (841 questions), multi-hop (282), temporal reason-
ing (321), and open-domain (96). Total: 1,540 scored
questions (the 446 adversarial questions are excluded
from scoring per standard practice). The evaluation
code is based on Mem0’s published evaluation frame-
work.5
LongMemEval S[7]: Benchmarks five core long-
term memory abilities—single-session information ex-
traction (user-stated facts, assistant-stated facts, and
preference inference), temporal reasoning, knowledge
updates, and multi-session reasoning—using 500 cu-
5https://github.com/mem0ai/mem0/tree/main/evaluationrated questions embedded in chat histories of approx-
imately 115k tokens each. We ingest the benchmark’s
chat histories session-by-session, then answer each ques-
tion using MemMachine’s memory search API. Evalua-
tion uses the question-specific judge prompts provided
by the LongMemEval framework. We conduct a system-
atic ablation study across 12 configurations varying six
optimization dimensions.
7.2 Evaluation Metrics
For LoCoMo, we report three metrics:
•LLM Judge Score (llm_score):A binary score (0
or 1) assigned by a judge-LLM comparing the sys-
tem’s answer to ground truth. The overall score is
the weighted mean across categories.
•BLEU Score:n-gram overlap with reference an-
swers.
•F1 Score:Token-level precision and recall against
reference answers.
The primary metric for comparison is thellm_score,
as it best captures semantic equivalence between gener-
ated and reference answers.
For LongMemEval S, we report thellm_scoreper
question category and overall, using the benchmark’s
standard LLM-judge evaluation with GPT-4o-mini as
judge.
7.3 System Environment
Table 9: Benchmark test environment.
Component Specification
Operating Sys-
temUbuntu 24.04 LTS
CPU 8 vCPUs
RAM 16 GiB
GPU Not required (CPU-
only runs)
Python Version 3.11
MemMachine
Versionv0.3.x
Database PostgreSQL + Neo4j
Embedding
ModelOpenAI text-
embedding-3-small
Reranker AWS Cohere rerank-
v3-5:0
Eval-LLM OpenAI gpt-4o-mini /
gpt-4.1-mini / gpt-5 /
gpt-5-mini
Judge-LLM OpenAI gpt-4o-mini
7.4 Compared Systems
We compare MemMachine against the following systems
using publicly reported results:
•Mem0[4]: Tested with Mem0 main/HEAD and re-
10

run with gpt-4.1-mini for fair comparison.
•Zep[5]: Results sourced from Zep’s published evalu-
ation.
•Memobase: Results sourced from Memobase’s pub-
lished benchmark.
•LangMem: Results from Mem0’s comparative eval-
uation.
•OpenAI baseline: ChatGPT’s native memory.
7.5 Reproducibility
All benchmark scripts, configuration files, and run in-
structions are available in the MemMachine repository
underevaluation/.6For reproducible reporting, we
recommend pinning a repository tag or commit hash,
recording model versions and API provider settings, and
saving raw per-question outputs used for score aggrega-
tion. Researchers can reproduce our results with their
own hardware and API keys.
8 Results and Analysis
8.1 LoCoMo Benchmark Results
Table10presentsthecomprehensiveLoCoMoresultsfor
MemMachine v0.2 across both LLM configurations and
both operating modes.
8.2 Comparative Analysis
Table 11 compares MemMachine against competing sys-
tems on LoCoMo.
In our LoCoMo comparison setting, MemMachine
achieves the highest overall score by +9.7 points over
the next-best system (Memobase). Key observations:
•Single-hop(0.9465):MemMachine’ssentence-level
indexing and ground truth preservation enable excep-
tional factual recall.
•Multi-hop (0.8759):The contextualization mech-
anism allows linking related information across ses-
sions.
•Temporal (0.7352):Competitive but trailing
Memobase (0.8505), suggesting room for improve-
ment in temporal reasoning—likely addressable
through enhanced timestamp-aware retrieval.
•Open-domain (0.7083):Strong performance con-
sidering that episodic memory is optimized for user-
centric rather than world-knowledge queries.
With gpt-4.1-mini, MemMachine’s temporal score im-
proves to 0.9159 in agent mode, suggesting that eval-
model capability is a major factor in temporal reasoning
outcomes.
6https://github.com/MemMachine/MemMachine/tree/main/
evaluation8.3 Efficiency Analysis
Beyond accuracy, MemMachine demonstrates substan-
tial efficiency advantages:
•Token Reduction:∼80% fewer input tokens than
Mem0, directly reducing API costs.
•Memory Add Speed:∼75% faster than previous
versions, enabling real-time ingestion.
•Search Speed:Up to 75% faster search operations,
reducing end-to-end response latency.
8.4 LongMemEval SAblation Study
We evaluate MemMachine on the full LongMemEval S
benchmark (n= 500questions) through a systematic
ablation study across six optimization dimensions: sen-
tence chunking, user-query bias correction, context for-
matting, retrieval depth (k), search prompt design, and
answer-model selection. Each dimension is evaluated by
comparing configuration pairs that differ in exactly one
variable. Table 12 summarizes the key configurations;
Table 13 isolates the contribution of each optimization.
8.4.1 Retrieval Depth (k)
The most impactful single parameter is the number of
retrieved episodes. Increasingkfrom 20 to 30 yields
the largest improvement: +4.2 percentage points (C6:
0.870→C7: 0.912). However, further increases show
diminishing or negative returns:k= 50drops to 0.890,
worsethank= 30.
This non-monotonic behavior reflects a tension be-
tween recall and noise. Atk= 20, the retrieval window
missessomerelevantepisodes, particularlyformulti-hop
questions requiring evidence from multiple sessions. At
k= 30, sufficient evidence is captured without over-
whelming the answer LLM with distractors. Atk= 50,
additional episodes introduce irrelevant context that de-
grades reading comprehension, consistent with the “lost
in the middle” phenomenon [13].
Notably, thisk-sensitivity is model-dependent. With
GPT-5-mini (C12–C15), performance improves mono-
tonically fromk= 20(0.922) throughk= 50(0.928)
tok= 100(0.930), though with diminishing marginal
gains. This suggests GPT-5-mini is more robust to dis-
tractor context than GPT-5, possibly due to differences
in attention mechanisms or instruction-following behav-
ior at high context lengths.
8.4.2 User-Query Bias Correction
We observe that MemMachine’s search results can ex-
hibit a bias toward retrieving assistant messages over
user messages. Assistant messages are typically longer
with more sentences and therefore more embedding
keys, while user messages are shorter but often contain
first-hand factual statements with higher informational
value for recall tasks. Prepending the prefix"user:"to
the search query shifts retrieval toward user messages,
11

Table 10: MemMachine v0.2 LoCoMo results by category (LLM Judge Score).
Eval-LLM Mode Multi-hop Temporal Open-domain Single-hop Overall
gpt-4.1-mini Agent 0.8830 0.9159 0.7188 0.95120.9169
gpt-4.1-mini Memory 0.8972 0.8910 0.7500 0.9441 0.9123
gpt-4o-mini Agent 0.8404 0.8069 0.7396 0.9394 0.8812
gpt-4o-mini Memory 0.8759 0.7352 0.7083 0.9465 0.8747
Table 11: LoCoMo benchmark comparison across AI agent memory systems (LLM Judge Score). MemMachine
results are with gpt-4o-mini for fair comparison with published baselines; gpt-4.1-mini results shown separately in
Table 10.
System Single-hop Temporal Multi-hop Open-domain Overall
MemMachine v0.2 0.94650.73520.87590.70830.8747
Memobase (v0.0.37) 0.70920.85050.46880.77170.7578
Zep 0.7411 0.7979 0.6604 0.6771 0.7514
Mem0 0.6713 0.5551 0.5115 0.7293 0.6688
LangMem 0.6223 0.2343 0.4792 0.7112 0.5810
OpenAI 0.6379 0.2171 0.4292 0.6229 0.5290
yielding +1.4% (C5: 0.855→C6: 0.870).
8.4.3 Context Formatting
The format in which retrieved messages are presented
to the answer LLM significantly affects comprehension.
A naive approach concatenating all messages with car-
riage returns produces a wall of text. Using\nas line
terminatorswithinamessagewhileusingactualcarriage
returns toseparatemessages improves the LLM’s ability
to parse message boundaries, yielding +2.0%.
8.4.4 Sentence Chunking
MemMachine’s sentence-level chunking creates one em-
bedding key per sentence rather than per message, pro-
ducing finer-grained index entries. Enabling chunking
yields +0.8% (C6: 0.870→C9: 0.878), a modest but
consistent gain. The relatively small effect suggests that
sentence-level granularity primarily helps on questions
where the relevant information is a single sentence em-
bedded within a longer message.
8.4.5 Search Prompt Design
We evaluate three search prompt variants (Edwin1, Ed-
win2, Edwin3) with increasing refinement. Edwin3
yields +1.8% over Edwin1 (C9: 0.878→C11: 0.896),
demonstrating that the framing of the search query—
notjusttheretrievalalgorithm—materiallyaffectsrecall
quality.
8.4.6 Answer LLM Selection
A surprising finding is that GPT-5-mini outper-
forms GPT-5 as the answer LLM by +2.6% (C11:
0.896→C12: 0.922). The advantage persists across re-
trieval depths: atk= 30, GPT-5-mini achieves 0.916
vs. GPT-5’s 0.902; atk= 50, 0.928 vs. 0.914. Weattribute this to the interaction between prompt de-
sign and model architecture. The Edwin3 prompt is
a direct, concise instruction without chain-of-thought
scaffolding, which aligns with GPT-5-mini’s stream-
lined instruction-following. Conversely, GPT-5’s built-
in reasoning may interfere when given explicit reason-
ing instructions. Since GPT-5-mini is also substantially
cheaper per token, the best configuration is the most
cost-efficient.
8.4.7 Per-Category Analysis
Table 14 presents per-category scores for selected con-
figurations.
Several patterns emerge.Single-session extraction
(SSU, SSA) is nearly saturated: most configurations
achieve 0.98–1.00, indicating that sentence-level index-
ing is well-suited for recalling specific facts from individ-
ual sessions.Single-session preference(SSP) shows
the most dramatic improvement, rising from 0.700 (C5)
to 0.933 (C12–C15). Preference questions require infer-
ring user preferences from indirect cues, which benefits
from both better retrieval and better answer models.
Temporal reasoning(TR) improves steadily with re-
trieval depth and prompt optimization, from 0.798 (C5)
to 0.932 (C14), as retrieving more context helps estab-
lish temporal relationships.Multi-session reasoning
(MS) remains the most challenging category, peaking
at 0.872 (C15,k=100), since synthesizing information
across sessions requires the broadest retrieval window.
8.4.8 Token Cost–Accuracy Tradeoff
Retrieval depth directly affects token consumption. Ta-
ble 15 shows the cost profile for selected configurations.
The Pareto-optimal configuration is C12 (GPT-5-
mini,k=20): it achieves 0.922 with only 2.58M input
12

Table 12: LongMemEval Sconfiguration matrix (n= 500questions per run). “Chunk” = sentence-level chunking
enabled; “User-Q” = user-role query prefix; “JSON-str” = structured message formatting; “Edwin{1,3}” = search
prompt variants of increasing refinement.
ID Answer LLM Chunk User-Q JSON-str Prompt top_kLLM Score
C5 GPT-5 — —✓Edwin1 20 0.855
C6 GPT-5 —✓ ✓Edwin1 20 0.870
C7 GPT-5 —✓ ✓Edwin1 30 0.912
C8 GPT-5 —✓ ✓Edwin1 50 0.890
C9 GPT-5✓ ✓ ✓Edwin1 20 0.878
C11 GPT-5✓ ✓ ✓Edwin3 20 0.896
C12 GPT-5-mini✓ ✓ ✓Edwin3 20 0.922
C13 GPT-5-mini✓ ✓ ✓Edwin3 30 0.916
C14 GPT-5-mini✓ ✓ ✓Edwin3 50 0.928
C15 GPT-5-mini✓ ✓ ✓Edwin3 1000.930
C16 GPT-5✓ ✓ ✓Edwin3 30 0.902
C17 GPT-5✓ ✓ ✓Edwin3 50 0.914
Table 13: LongMemEval Sablation: contribution of each
optimizationtooverallLLMscore,measuredbycompar-
ing configuration pairs that differ in one variable.
Optimization Comparison∆Score
Retrieval-stage optimizations
Retrieval depth (k: 20→30) C6 vs. C7 +4.2%
Context formatting (JSON-str) C4 vs. C5 +2.0%
Search prompt (Edwin1→3) C9 vs. C11 +1.8%
COT→simple prompt (GPT-5) C1 vs. C4 +1.6%
User-query bias correction C5 vs. C6 +1.4%
Ingestion-stage optimization
Sentence chunking C6 vs. C9 +0.8%
Model selection
GPT-5→GPT-5-mini C11 vs. C12 +2.6%
tokens, outperforming C7 (GPT-5,k=30, 0.912) which
requires 4.03M tokens. Reaching the peak score of 0.930
(C15) requires3.8×the input tokens of C12 for only
+0.8% accuracy.
9 Discussion
9.1 Retrieval Stage Dominates Accu-
racy
Our LongMemEval ablation reveals that retrieval-stage
optimizations contribute substantially more to final ac-
curacy than ingestion-stage changes. The cumulative
effect of retrieval-side improvements (retrieval depth:
+4.2%, formatting: +2.0%, search prompt: +1.8%,
COT removal: +1.6%, user-query bias: +1.4%) far ex-
ceeds the ingestion-side contribution of sentence chunk-
ing (+0.8%). This suggests that for memory systems,
howdata is recalled matters more thanhowit is stored,
provided the storage preserves ground truth.This has architectural implications: systems that in-
vest heavily in LLM-based ingestion (extracting facts,
building knowledge graphs) may be over-optimizing the
wrong stage. MemMachine’s approach of storing raw
data with lightweight indexing and investing in retrieval
quality appears to be more effective per unit of engi-
neering effort.
9.2 Model–Prompt Co-optimization
The GPT-5-mini result on LongMemEval highlights
that model selection and prompt design must be co-
optimized. A chain-of-thought prompt designed for
GPT-4.x is suboptimal for GPT-5, and a simple di-
rect prompt can outperform a complex one when paired
with the right model. The advantage persists across
retrieval depths (GPT-5-mini beats GPT-5 by +1.4%
atk=30 andk=50). This argues against the com-
mon practice of reusing prompts across model upgrades,
and suggests that memory system deployments should
re-evaluate prompts whenever the underlying answer
model changes.
9.3 The Role of Personalization
Personalization is perhaps the most compelling reason
for AI agent memory. Without memory, every interac-
tion starts from zero—the agent has no awareness of the
user’s history, preferences, or context. Memory trans-
forms a generic LLM into a personalized assistant that
adapts to individual users over time.
MemMachine’s dual memory architecture directly
supports personalization: episodic memory provides the
factual grounding for “what happened,” while profile
memory captures the distilled “who the user is.” This
combination enables agents to:
•Maintain continuity across sessions without requiring
users to repeat context.
•Adapt responses to user preferences, communication
13

Table 14: LongMemEval Sper-category LLM scores for selected configurations (n= 500). SSU = single-session-user,
SSP = single-session-preference, SSA = single-session-assistant, TR = temporal reasoning, KU = knowledge update,
MS = multi-session.
Config LLM SSU SSP SSA TR KU MS Total
C5 (baseline) GPT-5 0.986 0.700 1.000 0.798 0.925 0.797 0.855
C6 (user-q) GPT-5 0.957 0.667 0.982 0.850 0.885 0.835 0.870
C7 (k=30) GPT-5 0.986 0.800 1.000 0.917 0.923 0.850 0.912
C12 (mini,k=20) GPT-5-mini 0.986 0.933 1.000 0.902 0.962 0.850 0.922
C14 (mini,k=50) GPT-5-mini 1.000 0.933 1.000 0.932 0.949 0.842 0.928
C15 (mini,k=100) GPT-5-mini 1.000 0.933 0.982 0.917 0.949 0.8720.930
Table 15: LongMemEval Stoken consumption per 500-
question run. Input tokens scale withk; output tokens
remain approximately constant.
Config (LLM)kInput (M) Score
C6 (GPT-5) 20 2.94 0.870
C7 (GPT-5) 30 4.03 0.912
C8 (GPT-5) 50 6.47 0.890
C12 (GPT-5-mini) 20 2.58 0.922
C14 (GPT-5-mini) 50 5.97 0.928
C15 (GPT-5-mini) 100 9.79 0.930
style, and domain expertise.
•Build trust through demonstrated recall of past in-
teractions.
•Provide proactive suggestions based on accumulated
user knowledge.
As AI agents move from novelty to daily utility, per-
sonalization through memory will become a differentiat-
ing capability. Users will expect their agents to “know”
them—and memory systems like MemMachine provide
the infrastructure to deliver this expectation.
9.4 Summary vs. Full Context vs. Com-
pressed Observations
A recurring design question is whether to provide the
LLM with a summary of past interactions, the full con-
versational context, or a compressed intermediate form.
Ourfindings, togetherwithrecentworkonobservational
memory [15], suggest that each approach occupies a dis-
tinct point in the design space:
•Full contextoverwhelms the model, triggers the
“lost in the middle” effect [13], and becomes infea-
sible as history grows beyond context limits.
•Summary-onlyapproaches (compaction) lose crit-
ical detail, particularly for factual and temporal
queries where the exact wording or timing matters.
As Mastra’s research notes, compaction produces
“documentation-style summaries” that “strip out the
specific decisions and tool interactions agents need.”
•Compressed observations(Mastra) offer a mid-
dle ground: event-based logs that preserve structurewhile achieving 3–40×compression. This enables
prompt caching and stable context windows but sac-
rifices the ability to retrieve original episodes on de-
mand.
•MemMachine’s approach—STM summaryplus
selectively retrieved raw episodes—provides a differ-
ent tradeoff: the summary gives high-level context,
while retrieved episodes supplyuncompressedfactual
grounding. This is particularly important for use
cases requiring auditability, compliance, or multi-hop
reasoning over exact conversational records.
The choice between these approaches depends on the
deployment context. For tool-heavy agents generating
largeoutputs(codingagents, SREagents), compression-
first approaches may be optimal. For agents serving do-
mainswherefactualprecisionmatters(healthcare, legal,
financial services), ground-truth-preserving retrieval is
essential. A promising future direction is hybrid archi-
tectures that combine compressed observations for high-
level context with on-demand retrieval of raw episodes
when precision is needed.
9.5 Single-Agent vs. Multi-Agent Mem-
ory
Memory benefits increase in multi-agent environments:
•Shared memoryenables agents to coordinate with-
out redundant information gathering.
•Specialized agentscan deposit domain-specific
knowledge that other agents retrieve, enabling emer-
gent division of labor.
•Session continuityallows agent handoffs without
context loss.
•Reduced token usageacross the system, as agents
share rather than regenerate context.
MemMachine’s multi-tenancy architecture
(project/session isolation) naturally supports multi-
agent deployments where agents share a project-level
memory while maintaining session-level isolation for
individual conversations.
14

9.6 Privacy and Data Sovereignty
Memory systems for AI agents raise important privacy
considerations. When user conversations are stored and
indexed, the data sovereignty question becomes critical:
•Local LLMs:Running embedding models and
LLMs locally (e.g., using Ollama or vLLM) keeps
all data on-premises. MemMachine supports local
providers through its configurable model architec-
ture.
•Hosted APIs:Using OpenAI, Google, or AWS ser-
vices means conversational data traverses third-party
infrastructure, subject to their data processing agree-
ments.
•Hybrid approaches:Memory storage can remain
localwhileonlyanonymizedoraggregatedqueriesare
sent to hosted LLMs for summarization or inference.
MemMachine’s open-source, self-hosted architecture
gives organizations full control over their data pipeline.
The configurable provider system allows swapping be-
tween local and hosted models without code changes,
enabling organizations to match their privacy require-
ments to their deployment model.
9.7 Limitations and Threats to Validity
Our results should be interpreted with several limita-
tions in mind. First, benchmark outcomes are sensitive
to eval-model choice, prompt templates, and provider-
side model updates; scores reported here are tied to
the specific configurations listed in Section 7. Sec-
ond, cross-system comparisons mix re-run results and
published numbers, which may differ in preprocess-
ing, prompt settings, or infrastructure. Third, while
LoCoMo, LongMemEval S, HotpotQA, WikiMultiHop,
and EpBench cover important retrieval behaviors, they
do not fully represent all production workloads (for
example, multilingual, multimodal, or strict real-time
constraints). Fourth, token-efficiency comparisons are
workload-dependent and should be treated as direc-
tional outside the reported setup. Fifth, the Long-
MemEval ablation treats each optimization dimension
independently; interaction effects between dimensions
(e.g., whether chunking benefits change at higherk) re-
main unexplored, and the ablation configurations C1–
C4 used partial question subsets before being extended
to the full 500-question evaluation. We therefore view
these results as strong empirical evidence within the
evaluated settings rather than universal performance
guarantees.
9.8 Architectural Design Tensions
The VentureBeat analysis of emerging memory architec-
tures[15]identifiesseveralkeyquestionsthatenterprises
should consider when selecting a memory approach. We
frame these as design tensions that inform MemMa-
chine’s positioning:Retrieval vs. stable context.Retrieval-based sys-
tems (MemMachine, Mem0, Zep) search for relevant
memories each turn, which enables access to arbitrar-
ily large memory stores but invalidates prompt caches
and adds latency. Stable-context systems (Mastra’s ob-
servational memory) keep a compressed log always in
context, enabling prompt caching but capping the to-
tal memory to what fits in the context window. Mem-
Machine’s approach provides the scalability of retrieval
while its STM component ensures that recent context is
always immediately available without a retrieval step.
Prompt cacheability.Modern LLM providers
(OpenAI, Anthropic) offer significant discounts for
cached prompt prefixes. Systems that maintain a sta-
ble prefix can exploit this for 50–90% cost reduction on
cached tokens. MemMachine’s STM summary provides
a semi-stable prefix, though retrieved episodes vary per
query. This represents an area for future optimiza-
tion—for example, caching frequently retrieved episode
clusters.
Infrastructure complexity.MemMachine requires
database infrastructure (PostgreSQL, Neo4j) but pro-
vides full control over data persistence and query pat-
terns. Text-only approaches (Mastra) eliminate the
need for specialized databases but may become con-
strained as memory volume grows. For enterprises
with existing database infrastructure, MemMachine’s
approach integrates naturally; for teams seeking mini-
mal infrastructure, simpler architectures may be prefer-
able initially.
Memory as a top-level primitive.There is grow-
ing consensus that memory is one of the essential primi-
tives for production AI agents, alongside tool use, work-
flow orchestration, observability, and guardrails. Mem-
Machine’s design as a standalone, framework-agnostic
memory layer—accessible via REST API, Python SDK,
and MCP—reflects this view. Agents should be able
to adopt memory without being locked into a specific
orchestration framework.
Table16summarizesthearchitecturaltradeoffsacross
representative memory systems.
9.9 When Memory Helps (and When It
Doesn’t)
Memory provides clear benefits for:
•Multi-session interactions (customer support, health-
care, education).
•Personalization-dependent applications (content rec-
ommendations, personal assistants).
•Complex workflows requiring state persistence
(project management, CRM).
•Compliance and audit scenarios requiring interaction
history.
Memory may be unnecessary or counterproductive
for:
15

Table 16: Architectural design space comparison across agent memory systems.
Property MemMachine Mem0 Zep Mastra OM MemOS Full Context
Memory approach Retrieval Retrieval Retrieval In-context Hybrid In-context
Ground truth preserved✓Partial Partial×Partial✓
Prompt cacheable Partial× ×✓×✓
Scales beyond context window✓ ✓ ✓×✓×
LLM calls per message Low High Moderate Moderate High None
Specialized DB required Yes Yes Yes No Yes No
Parametric/KV-cache memory× × × ×✓×
Works with closed-source LLMs✓ ✓ ✓ ✓Partial✓
Open source✓Partial Partial✓ ✓N/A
•Single-turn, stateless queries (search, translation,
simple QA).
•High-volume, low-personalization tasks (batch pro-
cessing, data extraction).
•Scenarios where privacy constraints prohibit storing
interaction history.
10 Future Work
Several directions merit investigation:
•Procedural memory:Extending MemMachine to
store and retrieve learned action patterns, tool-use
strategies, and workflow recipes.
•Enhanced temporal reasoning:Developing ded-
icated temporal indexing and query expansion tech-
niques to improve performance on temporal bench-
marks.
•LongMemEval Mevaluation:Extending evalua-
tion to LongMemEval M(500 sessions,∼1.5M tokens
per question), which tests memory at production
scale.
•Adaptive retrieval depth:Implementing query-
complexity-awarekselection, informedbyourfinding
that optimalkdepends on both the query type and
the downstream answer model.
•Memory consolidation and forgetting:Imple-
menting cognitive-inspired mechanisms for prioritiz-
ing frequently accessed memories and gracefully re-
tiring stale information.
•Multi-modal memory:Supporting images, audio,
and structured data alongside conversational text.
•Additional database backends:Expanding sup-
port to ChromaDB, Milvus, and other vector stores.
•Reinforcement learning integration:Using
benchmark feedback to optimize retrieval strategies
through learned policies.
•RetrievalAgentextensions:Expandingtheagent
tool tree with specialized agents for temporal rea-
soning, aggregation queries, and comparative anal-
ysis. Implementing budget enforcement (token cost
ceilings, latency limits) with automatic fallback to
cheaper strategies. Enabling per-agent LLM tier se-
lection for cost optimization.•Adaptive retrieval budgets:Dynamically adjust-
ing per-sub-query retrieval limits based on query
complexity estimates and accumulated evidence, re-
ducing redundant episode retrieval in fan-out and
chain-of-query strategies.
•Function-calling code mode:Investigating
function-calling architectures where agents emit
structured executable code (for example, Python or
TypeScript) rather than invoking large predefined
tool lists. Code executes in a secure interpreter
that handles data processing, dynamic tool discov-
ery, andmulti-stepchainingwithfewerrepeatedLLM
roundtrips. Prior reports indicate large token sav-
ings for massive toolsets (for example, 98.7% in An-
thropic’s MCP code execution workflow and up to
99.9% in Cloudflare’s Code Mode) [19, 20]. We will
evaluate both client-side and server-side variants, in-
cluding dynamic directory-based schema loading and
low-overhead search/execute proxy patterns.
11 Conclusion
We have presented MemMachine, an open-source mem-
ory system for AI agents that prioritizes ground
truth preservation, cost efficiency, and personalization.
Through a two-tier architecture of short-term and long-
term episodic memory augmented by profile memory,
MemMachine provides agents with the ability to store,
recall, andreasonoverpastexperienceswithoutthehigh
cost and error accumulation inherent in LLM-dependent
extraction approaches.
Our evaluation spans multiple benchmarks. On Lo-
CoMo, MemMachine achieves state-of-the-art perfor-
mance (0.9169 with gpt-4.1-mini) with approximately
80% fewer tokens than Mem0. On LongMemEval S,
a systematic ablation across six optimization di-
mensions achieves 93.0% overall accuracy and re-
vealsthatretrieval-stageoptimizations—particularlyre-
trieval depth tuning (+4.2%) and context formatting
(+2.0%)—dominate over ingestion-stage changes, and
that smaller models (GPT-5-mini) outperform larger
models (GPT-5) when co-optimized with appropriate
prompts. The Retrieval Agent extends these capa-
16

bilities to multi-hop queries, achieving 93.2% on Hot-
potQA hard and 92.6% on WikiMultiHop with random-
ized noise—demonstrating that MemMachine’s ground-
truth-preserving architecture is composable: intelligent
retrieval strategies can be layered on top without mod-
ifying the underlying storage model.
As AI agents transition from experimental technol-
ogy to production infrastructure, the quality of their
memory systems will determine the quality of their per-
sonalization, accuracy, and trustworthiness. MemMa-
chine provides a foundation for this next generation of
memory-augmented agents.
References
[1] Charles Packer, Sarah Wooders, Kevin Lin, Vivian
Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. MemGPT: Towards LLMs as Operating
Systems.arXiv preprint arXiv:2310.08560, 2024.
[2] Joon Sung Park, Joseph C. O’Brien, Carrie J.
Cai, Meredith Ringel Morris, Percy Liang, and
Michael S. Bernstein. Generative Agents: Interac-
tive Simulacra of Human Behavior.arXiv preprint
arXiv:2304.03442, 2023.
[3] Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InNeurIPS, 2020.
[4] PrateekChhikara, DevKhant, SaketAryan, Taran-
jeet Singh, and Deshraj Yadav. Mem0: Building
Production-Ready AI Agents with Scalable Long-
Term Memory.arXiv preprint arXiv:2504.19413,
2025.
[5] Preston Rasmussen, Pavlo Paliychuk, Travis Beau-
vais, JackRyan, andDanielChalef. Zep: ATempo-
ral Knowledge Graph Architecture for Agent Mem-
ory.arXiv preprint arXiv:2501.13956, 2025.
[6] Adyasha Maharana, Dong-Ho Lee, Sergey
Tulyakov, Mohit Bansal, Francesco Barbieri,
and Yuwei Fang. Evaluating Very Long-Term
Conversational Memory of LLM Agents.arXiv
preprint arXiv:2402.17753, 2024.
[7] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei
Zhang, Kai-Wei Chang, and Dong Yu. Long-
MemEval: Benchmarking Chat Assistants on
Long-Term Interactive Memory.arXiv preprint
arXiv:2410.10813, 2024. Accepted at ICLR 2025.
[8] Alexis Huet, Zied Ben Houidi, and Dario Rossi.
Episodic Memories Generation and EvaluationBenchmark for Large Language Models. InInter-
national Conference on Learning Representations
(ICLR), 2025.
[9] Yuyang Hu, Shichun Liu, Yue Yue,et al.Memory
in the Age of AI Agents: A Survey.arXiv preprint
arXiv:2512.13564, 2025.
[10] Endel Tulving. Episodic and Semantic Memory. In
E. Tulving and W. Donaldson, editors,Organiza-
tion of Memory, pages 381–403. Academic Press,
1972.
[11] Richard C. Atkinson and Richard M. Shiffrin. Hu-
man Memory: A Proposed System and Its Control
Processes. In K. W. Spence and J. T. Spence, ed-
itors,The Psychology of Learning and Motivation,
volume 2, pages 89–195. Academic Press, 1968.
[12] Tibor Kiss and Jan Strunk. Unsupervised Multilin-
gual Sentence Boundary Detection.Computational
Linguistics, 32(4):485–525, 2006.
[13] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin
Paranjape, Michele Bevilacqua, Fabio Petroni, and
Percy Liang. Lost in the Middle: How Language
Models Use Long Contexts.Transactions of the
Association for Computational Linguistics, 12:157–
173, 2024.
[14] Yu Wang, Chi Han, Tongtong Wu, Xiaoxin He,
Wangchunshu Zhou, Nafis Sadeq, Xiusi Chen,
Zexue He, Wei Wang, Gholamreza Haffari,et al.
Towards Lifespan Cognitive Systems.arXiv
preprint arXiv:2409.13265, 2024.
[15] Tyler Barnes and Sam Bhagwat. Observational
Memory: A Human-Inspired Memory System for
AI Agents. Mastra Technical Report, 2026.https:
//mastra.ai/research.
[16] Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen,
Boyu Chen, Shichao Song, Simin Niu, Hanyu
Wang, etal. MemOS:AMemoryOSforAISystem.
arXiv preprint arXiv:2507.03724, 2025.
[17] Zhiyu Li, Shichao Song, Hanyu Wang, Simin
Niu, Ding Chen, Jiawei Yang, Chenyang Xi,
et al. MemOS: An Operating System for Memory-
Augmented Generation (MAG) in Large Language
Models.arXiv preprint arXiv:2505.22101, 2025.
[18] Xiao Luo, Yuxuan Zhang, Zheng He, Zifeng Wang,
Sixun Zhao, Dongming Li, Long K. Qiu, and Yang
Yang. Agent Lightning: Train ANY AI Agents
with Reinforcement Learning.arXiv preprint
arXiv:2508.03680, 2025.
[19] Anthropic. Code execution with MCP. Engi-
neering blog, 2025.https://www.anthropic.com/
engineering/code-execution-with-mcp.
17

[20] Cloudflare. Introducing Code Mode for MCP
servers. Cloudflare blog, 2025.https://blog.
cloudflare.com/code-mode-mcp/.
[21] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. HotpotQA: A
Dataset for Diverse, Explainable Multi-hop Ques-
tion Answering. InProceedings of EMNLP, 2018.
18