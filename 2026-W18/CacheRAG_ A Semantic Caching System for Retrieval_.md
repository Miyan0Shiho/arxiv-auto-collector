# CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering

**Authors**: Yushi Sun, Lei Chen

**Published**: 2026-04-28 23:46:47

**PDF URL**: [https://arxiv.org/pdf/2604.26176v1](https://arxiv.org/pdf/2604.26176v1)

## Abstract
The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) has significantly advanced Knowledge Graph Question Answering (KGQA). However, existing LLM-driven KGQA systems act as stateless planners, generating retrieval plans in isolation without exploiting historical query patterns: analogous to a database system that optimizes every query from scratch without a plan cache. This fundamental design flaw leads to schema hallucinations and limited retrieval coverage. We propose CacheRAG, a systematic cache-augmented architecture for LLM-based KGQA that transforms stateless planners into continual learners. Unlike traditional database plan caching (which optimizes for frequency), CacheRAG introduces three novel design principles tailored for LLM contexts: (1) Schema-agnostic user interface: A two-stage semantic parsing framework via Intermediate Semantic Representation (ISR) enables non-expert users to interact purely in natural language, while a Backend Adapter grounds the LLM with local schema context to compile executable physical queries safely. (2) Diversity-optimized cache retrieval: A two-layer hierarchical index (Domain $\rightarrow$ Aspect) coupled with Maximal Marginal Relevance (MMR) maximizes structural variety in cached examples, effectively mitigating reasoning homogeneity. (3) Bounded heuristic expansion: Deterministic depth and breadth subgraph operators with strict complexity guarantees significantly enhance retrieval recall without risking unbounded API execution. Extensive experiments on multiple benchmarks demonstrate that CacheRAG significantly outperforms state-of-the-art baselines (e.g., +13.2% accuracy and +17.5% truthfulness on the CRAG dataset).

## Full Text


<!-- PDF content starts -->

CacheRAG: A Semantic Caching System for Retrieval-Augmented
Generation in Knowledge Graph Question Answering
Yushi Sun
HKUST
Hong Kong, China
ysunbp@connect.ust.hkLei Chen
HKUST(GZ)
Guangzhou, China
HKUST
Hong Kong, China
leichen@cse.ust.hk
Abstract
The integration of Large Language Models (LLMs) with Retrieval-
Augmented Generation (RAG) has significantly advanced Knowl-
edge Graph Question Answering (KGQA). However, existing LLM-
driven KGQA systems act as stateless planners, generating retrieval
plans in isolation without exploiting historical query patterns:
analogous to a database system that optimizes every query from
scratch without a plan cache. This fundamental design flaw leads
to schema hallucinations and limited retrieval coverage. We pro-
pose CacheRAG, a systematic cache-augmented architecture for
LLM-based KGQA that transforms stateless planners into continual
learners. Unlike traditional database plan caching (which optimizes
for frequency), CacheRAG introduces three novel design principles
tailored for LLM contexts: (1) Schema-agnostic user interface: A
two-stage semantic parsing framework via Intermediate Semantic
Representation (ISR) enables non-expert users to interact purely in
natural language, while a Backend Adapter grounds the LLM with
local schema context to compile executable physical queries safely.
(2) Diversity-optimized cache retrieval: A two-layer hierarchical in-
dex (Domain→Aspect) coupled with Maximal Marginal Relevance
(MMR) maximizes structural variety in cached examples, effectively
mitigating reasoning homogeneity. (3) Bounded heuristic expan-
sion: Deterministic depth and breadth subgraph operators with
strict complexity guarantees significantly enhance retrieval recall
without risking unbounded API execution. Extensive experiments
on multiple benchmarks demonstrate that CacheRAG significantly
outperforms state-of-the-art baselines (e.g., +13.2% accuracy and
+17.5% truthfulness on the CRAG dataset).
Keywords
Retrieval-Augmented Generation, Continual Learning, KGQA
ACM Reference Format:
Yushi Sun and Lei Chen. 2018. CacheRAG: A Semantic Caching System for
Retrieval-Augmented Generation in Knowledge Graph Question Answering.
InProceedings of Make sure to enter the correct conference title from your
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
https://doi.org/XXXXXXX.XXXXXXXrights confirmation email (Conference acronym ’XX).ACM, New York, NY,
USA, 18 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Knowledge Graph Question Answering (KGQA) is a fundamen-
tal task in data management and artificial intelligence, aiming to
provide precise answers to natural language questions over struc-
tured knowledge bases. Recently, the integration of Large Language
Models (LLMs) via the Retrieval-Augmented Generation (RAG) par-
adigm has emerged as a promising solution. By incorporating se-
mantic parsing and query plan generation into the LLM’s reasoning
process, these RAG-based systems can handle complex queries and
generate context-aware responses.
However, when deployed in real-world scenarios, existing LLM-
driven KGQA pipelines expose critical system-level flaws. Most
prominently, these systems act asstateless planners: they gener-
ate retrieval plans (e.g., SPARQL queries or API calls) based solely
on the current question,failing to leverage historical question-
answering experiences. This design mirrors a hypothetical data-
base system that optimizes every query from scratch without a plan
cache: an inefficiency that modern DBMSs resolved decades ago
through execution plan caching [ 9]. Yet, naively adapting traditional
DB plan caching to LLM-based systems fails due to fundamental dif-
ferences: (1) LLM context windows rewarddiversityover frequency
(unlike LRU/LFU), and (2) retrieval plans must beschema-aligned
to prevent hallucinations.
The stateless KGQA reasoning approach inevitably leads to two
major problems, as illustrated in Figure 1 (b): First,susceptibil-
ity to schema hallucination. Consider the query “Who are the
characters in J. K. Rowling’s latest magic novel?” Existing stateless
LLM planners might blindly retrieve her latest general novel (The
Hallmarked Man, a crime fiction) by missing the “magic” constraint.
When forced to self-correct, they frequently hallucinate schema at-
tributes, attempting to query a non-existent topic: magic pred-
icate, leading to empty results and execution failures.
Second,limited retrieval coverage. Even when the entity is
correctly identified, zero-shot planning often generates shallow,
incomplete query structures: missing multi-hop reasoning paths or
failing to explore alternative schema branches when initial retrieval
returns empty results.
Root Cause and Our Solution.Root Cause 1: Schema Misalign-
ment. Existing systems expect users to phrase queries using exact
KG predicates (e.g., “novels with topic=magic”) instead of natural
language. Our CacheRAG method addresses this through a two-
stage decoupling architecture:arXiv:2604.26176v1  [cs.DB]  28 Apr 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
Q: Who are the characters in J. K. Rowling's latest magic novel?(a) Natural Language Question
(b) Baseline: Stateless LLM Execution
author publishDate
characterJ. K. Rowling
❎ wrong
entity
(c) CacheRAG: Cache-A ware Bounded ExecutionQ: What is J.K. Rowling's latest novel that has a magic genre?
A: The Christmas PigExample Plan
author publishDate
genre
characterJ. K. Rowling
author publishDate
genre
style
characterJ. K. Rowling
magic?magic?
......Bounded Depth Expansion
Bounded Br eadth
Expansion
✅ correctThe Hallmarked Man
topic
genreThe Hallmarked Manauthor publishDate
genreJ. K. Rowling
❎ emptymagic?magic?
magic?schema
hallucination
missing or
incomplete
Plan Retrieval Semantic
CacheQ: Who are the
characters in J. K.
Rowling's latest
magic novel?Parser KG Compiler
Dynamic Subgraph
Adjustment
Output: Extracted
Characters(d) An Execution Example of CacheRAG
Dynamic Adjustment
Dynamic Adjustment
Figure 1: Comparison of stateless LLM execution (baseline) and CacheRAG (our approach) on a KGQA task. (a) Input: natural
language question about J.K. Rowling’s latest magic novel. (b) Baseline suffers from schema hallucination (retrieving wrong
entityThe Hallmarked Man, retrieval of non-existent attributes such as topic ) and incomplete retrieval (missing the genre
attribute). (c) CacheRAG workflow: the parser extracts ISR, semantic cache provides historical plans via plan retrieval, and
bounded subgraph expansion dynamically adjusts the query plan. (d) Execution example: CacheRAG refers to the useful cached
example, applies bounded depth/breadth expansion with dynamic adjustment to explore alternative schema paths ( genre ,
style ,publishDate ,character ), successfully identifying the correct KG subgraph for QA.
(1)User Layer (Logical): Users express intents naturally (“magic
novel”). The LLM acts as a syntactic parser to extract an Inter-
mediate Semantic Representation (ISR): {entity: J.K. Rowling,
type: novel, constraints: [magic]}.
(2)System Layer (Physical): The Backend Adapter grounds the
LLM by fetching the actual local schema (e.g., valid edges around
the retrieved entity) and prompts the LLM to compile the logical
ISR into an executable physical query. For instance, given the
constraint “magic” and a local schema containing genre , the
LLM safely maps the intent to the genre: fantasy predicate.
This eliminates the need for user queries to match exact domain-
specific predicate names while substantially mitigating schema
hallucination.
Root Cause 2: Lack of Historical Query Patterns. Traditional
stateless planners reinvent query logic for every question, wasting
API calls and producing shallow plans. As shown in Figure 1(c),
CacheRAG introduces a Semantic Cache that stores successful his-
torical retrieval plans. During plan generation, the cache retrieval
module provides useful examples (e.g., similar queries demonstrat-
ing “author→novels→genre” navigation patterns), enabling the
LLM to learn complex multi-hop reasoning rather than relying
on zero-shot guessing. When initial retrieval is insufficient, the
heuristic dispatcher (Figure 1(d)) triggers bounded depth/breadthexpansion, exploring alternative schema paths (e.g., checking both
genre and style predicates) until the correct answer is found.
This dual-stage design ensures that non-expert users interact
purely via natural language (addressing Root Cause 1), while the
system leverages historical query patterns to guide plan generation
(addressing Root Cause 2): eliminating the need for users to learn
KG-specific predicates or SPARQL syntax (Section 3.1).
Our Key Insight.Successful KGQA interactions exhibit strong
structural reusability: for example, questions about movie awards
consistently require similar multi-hop patterns (e.g., film →direc-
tor→other films→nominations), regardless of specific entities.
Yet, stateless LLMs must rediscover these patterns for every query.
CacheRAG bridges this gap by introducing a systematic cache-
augmented architecture for LLM-based KGQA, transforming state-
less planners into continual learners that exploit historical query
patterns while maintaining rigorous schema fidelity and execution
bounds. Designing such a robust cache-augmented system requires
systematically addressing four fundamental challenges. For each
challenge, CacheRAG introduces a targeted architectural design
choice with strict justifications:
•Challenge 1: Non-Deterministic Attribute Extraction.LLMs
often hallucinate predicates (e.g., extracting non-existent attribute
“topic”), leading to execution failures and making the system un-
intuitive for non-expert users.

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Solution 1: Schema-Constrained Semantic Parsing.We de-
couple the user-facing prompt from KG-specific jargon. The LLM
first extracts an Intermediate Semantic Representation (ISR). A
Backend Adapter then provides the valid local schema context,
repurposing the LLM to compile the ISR into physical KG queries.
This ensures schema fidelity and enablesnon-expert users to in-
teract purely via natural languagewithout learning SPARQL
or KG schemas.
•Challenge 2: Suboptimal Experience Replay (Cache Sched-
uling).Traditional cache metrics (LRU/LFU) or naive semantic
similarity fall short in LLM contexts because injecting structurally
identical cached plans wastes the LLM’s limited token window
and induces mode collapse, providing minimal information gain.
Solution 2: Diversity-Aware Cache Management.We en-
hance cache retrieval by utilizing a two-layer hierarchical index
(Domain→Aspect) and a Maximal Marginal Relevance (MMR)
scoring function. This guarantees the LLM is provided with a
structurally diverse set of topological query patterns, maximizing
its multi-hop reasoning capabilities.
•Challenge 3: Limited Retrieval Coverage in Complex QA.
LLMs often fail to retrieve sufficient context in a single pass,
yet unbounded retry loops risk exhaustive API calls and system
crashes.
Solution 3: Dynamic Plan Adjustment via Bounded Ex-
pansion.We introduce dynamic plan adjustment governed by
an auto-termination judge. If initial retrieval is insufficient, the
system triggers strictly bounded execution operators: Depth Ex-
pansion (auto-chaining) and Breadth Expansion (star queries).
This significantly enhances retrieval recall while maintaining
strict algorithmic bounds on execution complexity.
•Challenge 4: The Cold-Start Dilemma.At system initializa-
tion, the cache is empty, leaving the LLM with no reference plans
to navigate complex schemas.
Solution 4: Offline Auto-Generation.We introduce an ex-
ploratory view materialization strategy. By sampling the KG
schema offline to synthesize initial ⟨Query,Plan,Answer⟩ tuples,
CacheRAG ensures high-quality continual learning from the very
first user query.
In summary,this paper makes the following novel contri-
butions:
•Novel System Paradigm.We propose CacheRAG, introduc-
ing astateful cache-augmented paradigmfor LLM-based
KGQA. Unlike prior stateless planners that reinvent query logic
for every question, CacheRAG systematically integrates contin-
ual learning into query planning. This transforms the LLM into a
history-aware reasoning engine, successfully adapting the proven
concept of database plan caching to the era of large language
models.
•Schema-Agnostic User Interface.We introduce a two-stage
semantic parsing framework via Intermediate Semantic Repre-
sentation (ISR) and a Backend Adapter, enablingnon-expert
users to interact purely in natural languagewithout learning
KG schemas or SPARQL. To the best of our knowledge, this is
thefirst work to systematically decouple user-facing logi-
cal intents from physical query compilationin LLM-basedKGQA, achieving 98% entity extraction accuracy and a 98.9%
query compilation success rate.
•Diversity-Aware Cache Architecture.We design a two-layer
hierarchical index (Domain →Aspect) coupled with MMR-based
retrieval,specifically optimized for LLM context diversity.
•Provably Bounded Expansion Operators.We formalize depth
and breadth expansions as bounded dynamic subgraph opera-
tions with strict complexity guarantees ( 𝐾𝑑𝑒𝑝𝑡ℎ≤3, space com-
plexityO(𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒)). We providerigorous empirical
evidence via controlled ablationthat cache-guided expansion
achieves significantly higher accuracy than blind expansion (up
to 6% improvement), addressing a critical open question in
LLM-based retrieval systems.
•Comprehensive Experimental Validation.Extensive exper-
iments on multiple large-scale benchmarks demonstrate that
CacheRAG significantly outperforms state-of-the-art KGQA base-
lines in accuracy (+13.2% on CRAG) and truthfulness (+17.5% on
CRAG).
2 Related Work
KGQA approaches can be broadly classified into two categories:
Semantic Parsing KGQA and KG-based RAG. Additionally, we con-
textualize our contributions within the recent advancements in
semantic caching and agentic workflows in database systems.
2.1 Semantic Parsing and Natural Language
Interfaces
One major line of research in KGQA is semantic-parsing-based
(SP-based) KGQA, which transforms natural language queries into
logical forms to execute structured queries. SP-based approaches
can be divided into multi-step and seq2seq methods. Multi-step
approaches formulate this as a multi-step search problem, involving
core entity identification and query graph expansion based on entity
attributes and query predicates [ 5,17,18,21,23,35,36]. Seq2seq
approaches treat semantic parsing as a translation task, leveraging
fine-tuned language models to directly generate complete semantic
expressions and retrieve relevant KG content [ 4,6,10–12,25,30,31,
34,37,39]. For example, [ 34] generates and ranks candidate logical
expressions, [ 4] first creates a high-level sketch and then refines its
arguments, and Decaf [ 37] introduces a retriever-reader-combiner
structure that jointly generates answers and logical forms. While
highly precise, these methods depend on strict schemas and exhibit
low tolerance for parsing errors, struggling to handle the semantic
ambiguities inherently present in complex QA.
2.2 KG-based RAG and LLM-driven Execution
Recently, the database and NLP communities have focused on LLM-
based KGQA approaches that integrate semantic parsing and query
generation into the reasoning process of LLMs [ 14,22,26]. Struct-
GPT [ 14] designs a data retrieval interface, using prompts to enable
LLMs to generate API or SPARQL calls. However, its direct tool-
calling can blindly propagate planning errors. Beyond tool calling,
ToG [ 26] introduces beam search on KGs for iterative path explo-
ration. ToG-2 [ 22] extends modality to joint KG-text reasoning
with external Wikipedia. Concurrent to our work, recent database
research has started treating LLMs as core execution engines or

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
query optimizers. For instance, SERAG[ 20] utilizes a continuously
updated RAG vector database to help learned query optimizers
avoid cold-start problems and learn from historical execution feed-
back. In contrast to these approaches, CacheRAG focuses on en-
hancing the LLM planner’s continual learning for graph traversals
via QA history based on relevance and diversity, while introducing
strictly bounded depth/breadth expansions to significantly boost
recall without sacrificing system stability. Two Meta KDD Cup 2024
solutions also developed LLM-based RAG: db3 [ 29] uses an LLM
planner for routing, while apex [ 24] crafts API extraction rules to
match questions.
2.3 Semantic Caching and Agentic Workflows
Caching mechanisms have become indispensable for reducing la-
tency and costs in modern LLM deployments. Recent data manage-
ment literature, such as Cache-Craft [ 2] and RAGCache [ 15], focuses
on systems that optimize RAG workflows by efficiently caching
precomputed Key-Value (KV) states or intermediate retrieved text
chunks across similar queries. Unlike these infrastructure-level
caches, CacheRAG proposes a higher-level semantic plan cache that
stores logical reasoning paths and historical execution schemas to
directly guide the LLM’s multi-hop planning.
Regarding agentic workflows, systems like Buffer-of-Thoughts
(BoT) [ 32] build a meta-buffer storing universal thought-templates
distilled from problem-solving processes, while AFlow [ 38] uses
Monte Carlo Tree Search to generate general task workflows. These
methods focus on free-form tasks, whereas CacheRAG targets
KGQA governed by strict database schema constraints. Despite
both using experience replay, CacheRAG differs fundamentally:
1) CacheRAG relies on unsupervised template generation (Auto
Generation), offering higher generalizability to new KGs compared
to supervised methods. 2) CacheRAG formulates cache retrieval as
a submodular optimization problem, employing a two-layer index
and MMR to balance relevance and structural diversity, whereas
existing systems rely on naive semantic similarity. 3) CacheRAG’s
cached examples provide both reasoning logic and reusable physical
KG triples, ensuring deterministic retrieval coverage.
In summary, CacheRAG is a highly generalizable cache-augmented
KGQA system that enables continual learning, featuring unsuper-
vised template generation, a submodular retrieval index, experience
replay, and bounded two-dimensional KG path exploration.
Positioning: CacheRAG’s Key Contributions.CacheRAG intro-
duces four novel system-level contributions that, while building on
existing primitives (MMR, dense retrieval, graph traversal), consti-
tutethe first principled integration of caching mechanisms
into LLM-based KGQA:
(1) Cache-Augmented Continual Learning for KGQA.We
are thefirst to systematically apply semantic caching to en-
able continual learning in LLM-based query planning. Unlike
traditional DB plan caches that optimize for query acceleration
(LRU/LFU), our cache is designed toteachthe LLM recurring query
patterns. Our ablation (Section 5.7) presents up to 6% performance
gains, validating thatlearning from history fundamentally outper-
forms stateless planning.
(2) Provably Bounded Expansion Operators.We introduce
thefirst formalization of bounded depth/breadth expansionwith strict complexity guarantees ( 𝐾𝑑𝑒𝑝𝑡ℎ≤3, space complexity
𝑂(𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒)) for LLM-driven graph traversal. Unlike prior
work that relies on unbounded LLM-guided exploration (risking
OOM crashes), our design ensuresproduction-ready deployment
while significantly improving retrieval recall.
(3) Schema-Agnostic User Interface via ISR.We propose
a two-stage semantic parsing framework thatdecouples user-
facing natural language from physical KG schemas, achieving
98.9% attribute alignment precision while eliminating the need for
users to learn SPARQL or predicate names.
(4) Diversity-Aware Cache Retrieval via MMR.We introduce
MMR-based cache retrieval to maximize information gain in the
LLM’s limited context window. By balancing semantic relevance
with structural diversity, our approach ensures each cached ex-
ample contributes unique query pattern knowledge, preventing
redundancy and mode collapse.
3 Problem Formalization
3.1 Core Definitions
Definition 1 (Knowledge Graph).A Knowledge Graph is formally
defined as a tupleK=(𝐸,𝑅,𝑇), where:
•𝐸: Finite set of entities (e.g.,Inception,Christopher Nolan)
•𝑅⊆𝐸×𝑃×𝐸 : Set of triples(𝑡1,𝑝,𝑡 2), where𝑡1,𝑡2∈𝐸, and𝑝∈𝑃
is a predicate from the schemaS
•𝑇: Optional temporal annotations (for time-sensitive KGs like
Wikidata)
Definition 2 (Intermediate Semantic Representation).To de-
couple user inputs from physical schema, we define an ISR as a
tuple:
I=(𝑒 raw,Φraw,𝑑).
Here,𝑒rawis the raw topic entity, Φrawis a set of semantic con-
straints extracted from the natural language question 𝑄NL, and𝑑is
the domain hint.
Definition 3 (Query Plan).A query plan is a sequence of retrieval
operations:
𝜋=⟨𝑜𝑝 1,𝑜𝑝 2,...,𝑜𝑝𝑚⟩, 𝑜𝑝𝑖∈{SPARQL,API call}
Executing𝜋onKyields a subgraph𝐺 𝜋⊆K.
Definition 4 (Semantic Cache).The cache Cstores successful
retrieval histories:
C={(𝑄𝑖,𝜋𝑖,𝐴𝑖)}𝑁
𝑖=1, 𝑄𝑖∈NL,𝜋𝑖: query plan,𝐴 𝑖: verified answer
Each entry is indexed by domain 𝑑and aspect attribute 𝑎:C[𝑑][𝑎] .
3.2 Problem Statement
Input:
•A KGKwith schemaS=(𝑃,O)(predicates + ontology)
•A natural language question𝑄 NL
•A semantic cacheC(initially empty or auto-generated)
Goal:Design a system that:
(1) Extracts a schema-agnostic ISRIfrom𝑄 NL
(2)Retrieves diverse historical plans 𝑆∗⊆C (via MMR) to guide
the LLM planner
(3)Generates an executable query plan 𝜋via in-context learning
with𝑆∗

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
nominatedInceptionWhich other films directed by the director  of Inception  have been nominated  for the Academy Award for Best Picture  in 2018 ?
Academy Award for Best
Picture in 201 1
nominated
directedByInception Academy Award for Best
Picture in 201 1
Christopher Nolan
nominated
directedByInception Academy Award for Best
Picture in 201 1
directedBy directedByChristopher Nolan
...
Dunkirk Interstellarnominated
directedByAcademy Award for Best
Picture in 201 1
directedBy directedByChristopher Nolan
nominated...Dunkirk InterstellarInception
Academy Award for Best
Picture in 2018
... ...✅(a) (b)
(c) (d)
Figure 2: Illustration of retrieval path expansion for the running example. (a) Direct prompting: LLM incorrectly checks
nominated attribute of “Inception”. (b) With cache examples: LLM learns to identify the director via directedBy relation.
(c) Bounded Depth Expansion: The 𝜎𝑑𝑒𝑝𝑡ℎ operator safely chains to Christopher Nolan’s other films. (d) Bounded Breadth
Expansion: A Star-Pattern Scan ( 𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ ) retrieves 1-hop neighbors of “Dunkirk”, successfully locating the 2018 Academy Award
nomination.
(4)Dynamically adjusts 𝜋(depth/breadth expansion) if initial re-
trieval𝐺𝜋is insufficient
(5)Returns a verified answer 𝐴and updatesC←C∪{(𝑄 NL,𝜋,𝐴)}
Optimization Objectives:
•Accuracy:MaximizeP(𝐴=𝐴 gold)(answer correctness)
•Efficiency:Minimize |𝜋|(number of queries) while maintaining
high recall
•Diversity:Maximize information gain H(𝑆∗)in retrieved cache
examples
Key Constraints:
•Schema Fidelity:All predicates in 𝜋must satisfy 𝑝∈S (no
hallucination)
•Bounded Execution:Depth expansion limited to 𝐾depth≤3
hops, breadth expansion to Top-𝐾 degree neighbors
System Design Rationale.To realize the aforementioned opti-
mization objectives, the system must organically resolve the four
inherent execution bottlenecks introduced in Section 1 (i.e., schema
hallucination, cache redundancy, incomplete retrieval, and cold-
start degradation). We achieve this through the CacheRAG pipeline,
an end-to-end architecture detailed in the following section.
4 Methodology
The CacheRAG pipeline systematically transforms LLM-based rea-
soning into a rigorous, bounded database execution workflow. Rather
than treating execution bottlenecks in isolation, our key novelty
Question ISRLogical ParserHierarchicalCacheBackend AdapterPlans Physical Plan INCOMPLETECOMPLETEDispatcherBounded OperatorsSummarizerAnswer Update Routing Key AutoGenerationFigure 3: The overall pipeline of CacheRAG, featuring a
dual-layer architecture: logical parsing/compilation (top) and
heuristic execution with bounded operators (bottom).
lies in the synergistic integration of four components: (1) schema-
agnostic semantic parsing via an adapter pattern (Section 4.1), (2)
diversity-optimized hierarchical caching (Section 4.2), (3) determin-
istically bounded expansion operators (Section 4.3), and (4) an of-
fline auto-generation module for cold-start mitigation (Section 4.4).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
Algorithm 1Overall CacheRAG Architecture Process
Function Name:CacheRAG_Pipeline
Input:User question𝑄 NL, query time𝑡
Output:Natural language answer𝐴
1. Initialize the semantic cacheCvia Auto-Generation (Sec 4.4).
2.I←LogicalParser(𝑄 NL)# Extract domain𝑑, constraintsΦ raw
(Sec 4.1)
3.𝑆∗←CacheRetrieval(𝑄 NL,C,𝜆,𝑘,𝑑,𝑎)# Fetch diverse plans
(Sec 4.2)
4.𝜋 init←BackendAdapter(I,S 𝑙𝑜𝑐𝑎𝑙,𝑆∗)# Compile physical plan
5.𝐺 init←Execute(𝜋 init)# Retrieve initial subgraph
6.ifJ dispatcher(𝑄NL,𝑡,𝐺 init,C)==COMPLETEthen:
6.1𝐴←Summarize(𝑄 NL,𝑡,𝐺 init)
6.2 CacheUpdate(C,𝑄 NL,𝜋init,𝐴)# Trigger eviction if|C|>𝐵
6.3return𝐴
7.else:# Insufficient retrieval, trigger heuristic expansion
7.1𝜋 final,𝐺final←HeuristicTraversal(𝑄 NL,𝑡,𝜋 init,𝐺init,C)#
(Sec 4.3)
7.2ifJ dispatcher(𝑄NL,𝑡,𝐺 final,C)==COMPLETEthen:
7.2.1𝐴←Summarize(𝑄 NL,𝑡,𝐺 final)
7.2.2 CacheUpdate(C,𝑄 NL,𝜋final,𝐴)
7.3else:𝐴←LLM_Fallback(𝑄 NL,𝑡)# Answer via LLM directly
7.4return𝐴
To clarify the design choices and structural logic, we illustrate our
core modules while grounding them in our running example (from
Figure 2): “Which other films directed by the director of Inception
have been nominated for the Academy Award for Best Picture in
2018?”
System Architecture Overview.Figure 3 illustrates the overall ar-
chitecture of CacheRAG, which operates as a continuous, stateful
execution pipeline (formalized in Algorithm 1). As depicted, the
workflow is strictly divided into a logical planning phase and a
physical execution phase. Given a natural language question 𝑄NL,
theLogical Parserfirst extracts an Intermediate Semantic Repre-
sentation (ISR). This ISR serves dual purposes: it acts as a routing
key for theHierarchical Cacheto fetch structurally diverse his-
torical plans, and provides the logical constraints for theBackend
Adapterto compile an initial physical plan 𝜋init. During the exe-
cution phase, theHeuristic Dispatcherevaluates the retrieved
subgraph. If incomplete, it triggers theBounded Subgraph Oper-
ators(𝜎𝑑𝑒𝑝𝑡ℎ and𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ ), iteratively updating the subgraph via a
feedback loop until a termination state is reached, at which point
theSummarizergenerates the final answer 𝐴. We detail these
modules in the following subsections.
4.1 Schema-Constrained Semantic Parsing via
Adapter Pattern
Our Design: Two-Stage Compilation Architecture.To address the
severe bottlenecks of schema hallucination and lack of robustness
inherent in zero-shot LLM prompting (as outlined in Section 1),
CacheRAG conceptualizes semantic parsing as a two-stage process.
As depicted in the top logical layer of Figure 3, this strictly decou-
pled design is analogous to the compilation of a logical query plan
into a physical query plan in traditional DBMS.Stage 1: Schema-Agnostic ISR Extraction (Logical Plan).The
Logical Parserfirst acts purely as a syntactic parser, extracting an
Intermediate Semantic Representation (ISR) from the user’s natural
language query. The ISR consists of:
•Topic entity𝑒 raw(e.g., “Inception”)
•Semantic constraints Φraw={𝜙 1,𝜙2,...}(e.g., “directed by”, “Acad-
emy Award”, “2018”)
•Domain hint 𝑑(e.g., “movies”). To ensure high precision in multi-
domain environments, we augment this extraction with a dy-
namic KG description cache (detailed in Appendix A.7), enabling
the parser to iteratively refine its domain routing accuracy based
on execution feedback.
Critically, the ISR extraction promptexplicitly prohibitsthe LLM
from generating KG-specific predicate names, ensuring the extrac-
tion remains independent of the physical schema:
ISR Extraction Prompt (Schema-Agnostic)
Extract the main entity and key constraints from the question. Do NOT
generate KG predicates or schema-specific terms: use only the natural
language expressions from the user’s question. Question: “Which other
films directed by the director of Inception have been nominated for
the Academy Award for Best Picture in 2018?” Your output format: -
Main entity: [entity name] - Constraints: [list of semantic constraints in
natural language]
Stage 2: Schema-Aware Query Compilation (Physical Plan).
To bridge the gap between the platform-agnostic ISR and the under-
lying heterogeneous data sources, CacheRAG employs a determinis-
ticBackend Adapter. Instead of forcing the LLM to blindly output
SPARQL syntax, the adapter first fetches thelocal physical schema
(i.e., the valid 1-hop attributes and relations S𝑙𝑜𝑐𝑎𝑙 associated with
the retrieved topic entity𝑒 raw).
The LLM is then repurposed as aQuery Compiler. It receives
both the logical constraints Φrawand the valid local schema S𝑙𝑜𝑐𝑎𝑙
as context, and is tasked with grounding the natural language con-
straints to the exact physical predicates to assemble the executable
API call or SPARQL query. By strictly decoupling the natural lan-
guage reasoning phase from the physical graph execution, and
forcing the LLM to select only from provided valid predicates, this
design effectively prevents schema hallucination.
Running Example.
•Logical ISR:Constraint “directed by”
•Adapter Context:System fetches local schema for “Inception”:
S𝑙𝑜𝑐𝑎𝑙={directedBy,starring,genre,releaseDate}
•Physical Compilation:The Backend Adapter maps the logical
“directed by” to the physical directedBy and formats the exact
executable SPARQL queries or valid API calls.
Error Handling and Robustness.On CRAG dataset [ 33], this two-
stage decoupling achieves:
•98% entity extraction accuracyon CRAG (errors stem pri-
marily from pronoun ambiguity, e.g., “his latest work”, without
conversational context).
•98.9% physical compilation executable rate, confirming that
providing local schema context significantly reduces API execu-
tion failures.

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
If the LLM fails to map a constraint to the local schema, the sys-
tem deterministically triggersBreadth Expansion(Section 4.3)
to explore multi-hop schema paths, gracefully handling schema
ambiguity.
4.2 Diversity-Aware Plan Caching and Retrieval
Optimization
To provide the LLM with the ability to perform continual learn-
ing without weight updates, CacheRAG maintains a cache Cof
historical query plans. However, a critical design challenge arises
during retrieval: naive semantic similarity search (e.g., standard
KNN) tends to retrieve highly homogeneous queries. For instance,
if the cache returns five structurally identical historical plans asking
about “movie awards”, the marginal information gain for the LLM
is near zero. This redundancy wastes the limited context window
and induces reasoning homogeneity (mode collapse).
To address this, CacheRAG must enforce structural diversity dur-
ing cache retrieval while maintaining strict online latency bounds.
Evaluating a diverse subset from a global cache pool of size 𝑁
naively requiresO(𝑁2)pairwise comparisons. We bypass this bot-
tleneck through a two-stage retrieval architecture: a hierarchical
index for search space pruning, followed by a Maximal Marginal
Relevance (MMR) mechanism for diversity-aware selection.
Stage 1: Hierarchical Indexing for Hard Pruning.To seamlessly
bridge the logical parsing phase with cache retrieval, CacheRAG
utilizes the extracted ISR Ias the routing key (depicted as (𝑑,𝑎)
in Figure 3). As further detailed in Figure 4, we design a two-layer
routing structure to organize the cache:
•Layer 1 (Domain):Routed by the domain hint 𝑑from the ISR,
this layer separates examples by semantic domains (e.g., movies,
music, sports for the CRAG dataset). This strict boundary pre-
vents cross-domain contamination, ensuring that examples from
unrelated domains do not dilute the in-context learning rele-
vance.
•Layer 2 (Aspect Attribute):Further partitions examples by the
core query intent, which is abstracted directly from the semantic
constraints Φraw. For instance, under the “movie” domain, queries
with constraints regarding nominations (abstracted as the “award”
aspect) are isolated from those inquiring about actor relations
(the “cast” aspect). For single-domain KGs (e.g., Freebase in CWQ),
this degrades gracefully to a one-layer aspect index.
This hierarchical structure acts as a hard search space pruning
step. Guided deterministically by the logical ISR, it restricts the
subsequent diversity evaluation from the global cache to a highly
localized bucket, bounding the search complexity to O(𝑏log𝑏) ,
where𝑏=|C 𝑏𝑢𝑐𝑘𝑒𝑡|≪𝑁denotes the size of the target bucket.
Stage 2: MMR-based Soft Diversity Enforcement.Within the tar-
get bucket, CacheRAG implements the Maximal Marginal Rele-
vance (MMR) strategy (detailed in Algorithm 2) to select candidate
plans. At each step, we iteratively select a candidate cached entry
𝑐𝑗=(𝑄𝑗,𝜋𝑗,𝐴𝑗)that maximizes the marginal gain between its
semantic relevance to the input query 𝑄NLand its diversity against
the already selected set𝑆∗:
arg max
𝑐𝑗∈C𝑏𝑢𝑐𝑘𝑒𝑡\𝑆∗
𝜆·𝑆𝑖𝑚(𝑄 NL,𝑄𝑗)−(1−𝜆)·max
𝑐𝑘∈𝑆∗𝑆𝑖𝑚(𝑄𝑗,𝑄𝑘)
(1)
finance open movie music sportsindex by domain
... award actor ... ... ... ...
bucket bucket bucket bucket bucket bucket bucketindex by aspectFigure 4: The two-layer cache structure for multi-domain
KGs (e.g., CRAG dataset). The first layer indexes by domain
(movies, music, sports), while the second layer partitions by
aspect attributes (award, cast, director). For single-domain
KGs like Freebase (CWQ), we simplify to a one-layer index
based on aspect only.
Algorithm 2Diversity-Aware Cache Retrieval via MMR
Function Name:CacheRetrieval
Input:Input query𝑄 NL, semantic cacheC, penalty parameter𝜆,
sample size𝑘, domain𝑑, aspect𝑎
Output:Selected historical plans𝑆∗for compiler context
1.C𝑏𝑢𝑐𝑘𝑒𝑡←C[𝑑][𝑎]# Hard pruning: Fetch from localized index
2.if|C𝑏𝑢𝑐𝑘𝑒𝑡|<𝑘thenC 𝑏𝑢𝑐𝑘𝑒𝑡←C[𝑑][∗]# Fallback: Relax to
domain level
3.𝑆∗←∅# Initialize selected subset
4.while|𝑆∗|<𝑘andC 𝑏𝑢𝑐𝑘𝑒𝑡 ≠∅do:
4.1
𝑐∗←arg max
𝑐𝑗∈C𝑏𝑢𝑐𝑘𝑒𝑡h
𝜆·Sim(𝑄 NL,𝑄𝑗)−(1−𝜆)·max 𝑐𝑘∈𝑆∗Sim(𝑄𝑗,𝑄𝑘)i
4.2𝑆∗←𝑆∗∪{𝑐∗}
4.3C𝑏𝑢𝑐𝑘𝑒𝑡←C𝑏𝑢𝑐𝑘𝑒𝑡\{𝑐∗}
5.return𝑆∗
By dynamically balancing relevance and diversity via the penalty
parameter𝜆, this step deliberately avoids redundant token consump-
tion and formulates a structurally heterogeneous 𝑘-shot prompt
for the Backend Adapter.
Semantic Cache Storage and Bounded Memory Extension.By de-
fault, CacheRAG maintains an unbounded cache to accumulate all
successful historical plans, maximizing the repository for contin-
ual learning. Because our two-layer hierarchical index proactively
restricts the search space to highly localized buckets, the com-
putational overhead of MMR retrieval remains strictly bounded
(O(𝑏log𝑏)) even as the global cache size𝑁grows indefinitely.
However, for practical deployments with strict hardware con-
straints, CacheRAG can be seamlessly extended with a bounded
capacity limit 𝐵per bucket. In such memory-constrained scenar-
ios, applying a standard Least Recently Used (LRU) eviction pol-
icy is sufficient to maintain a high-quality repository. Traditional
frequency-based eviction does not degrade the structural diversity
here, as the hierarchical index has already rigidly partitioned the
query intents prior to eviction. This LRU-based extension is em-
pirically validated to save 64% of storage space with a negligible

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
(<2%) performance drop (as detailed in Section 5.12), proving the
robustness of our architecture under bounded memory footprints.
Justification & Running Example.Consider processing our run-
ning example (“Which other films directed by the director of Incep-
tion have been nominated for the Academy Award for Best Picture
in 2018?”). The system:
(1)Routes the query to the C[movies][award] bucket via the two-
layer index.
(2)Applies MMR (Algorithm 2) to fetch structurally diverse seman-
tic plans, such as:
•A multi-hop plan finding related entities via a shared person
(director→other movies).
•A plan demonstrating temporal filtering logic (year=2018).
•A complex specific query structure resolving “Academy Award
for Best Picture”.
The MMR penalty systematically restricts the system from fetching
redundant “movie nomination” plans. Instead, it compiles a diverse
set of plans that comprehensively teach the LLM the exact com-
positional logic needed to assemble the final complex multi-hop
execution path.
4.3 Dynamic Plan Adjustment via Bounded
Subgraph Operators
When the initial physical query plan 𝜋initis executed, the retrieved
subgraph might be insufficient to answer complex queries. Rather
than allowing unconstrained LLM reasoning, which frequently
leads to combinatorial explosion, unbounded API calls, or Out-Of-
Memory (OOM) errors during deep graph traversal, CacheRAG
formulates the dynamic plan adjustment as a set ofBounded
Heuristic Graph Operators.
As shown in the lower execution layer of Figure 3, the LLM is
decoupled from direct graph navigation and instead acts strictly
as aLightweight Heuristic Dispatcher Jdispatcher(𝑄NL,𝑡,𝐺𝑡,C).
Given the currently retrieved subgraph 𝐺𝑡, the dispatcher evaluates
the semantic completeness of the current state. It then greedily
routes the execution to one of two deterministic physical expansion
operators or to an early termination state ( J=COMPLETE ). To
guarantee predictable latency and prevent context window over-
flow, system-level safeguards are enforced as hard execution bound-
aries during operator invocation.
Bounded Depth Expansion ( 𝜎𝑑𝑒𝑝𝑡ℎ ).Conceptualized as aGraph-
based Index Nested Loop Join, this operator extends the multi-
hop path by exploring contexts along specific relational edges. The
heuristic dispatcher selects the most promising relational hop based
on the current frontier nodes. To prevent infinite loops and tightly
bound the execution cost, the depth of this nested join is strictly
constrained by a system parameter ( 𝐾𝑑𝑒𝑝𝑡ℎ≤3). As illustrated in
Figure 2(c), depth expansion enables the system to safely chain from
“Inception”directedBy−−−−−−−−→ “Christopher Nolan”directed−−−−−−→ [“Interstellar”,
“Dunkirk”, ...].
Bounded Breadth Expansion ( 𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ ).When targeted reasoning
fails due to implicit schema mismatches, CacheRAG executes a
Star-Pattern Neighborhood Scanto broadly explore the 1-hopAlgorithm 3Heuristic Dispatching for Bounded Subgraph Expan-
sion
Function Name:HeuristicTraversal
Input:User question𝑄 NL, query time𝑡, current physical plan𝜋,
retrieved subgraph𝐺 𝑡, semantic cacheC
Output:Extended physical plan𝜋 final, Expanded subgraph𝐺 final
1. state_status←J dispatcher(𝑄NL,𝑡,𝐺𝑡,C)# Evaluate logical
completeness
2.ifstate_status==INCOMPLETEand|𝜋|<𝐾 𝑑𝑒𝑝𝑡ℎ then:
2.1𝑟𝑛𝑒𝑥𝑡,𝑉𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟←ExtractJoinCondition(J dispatcher)
2.2𝐺𝑑𝑒𝑝𝑡ℎ←𝜎𝑑𝑒𝑝𝑡ℎ(𝑉𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟 ⊲⊳𝑟𝑛𝑒𝑥𝑡)# Execute Index Nested
Loop Join
2.3𝐺𝑏𝑟𝑒𝑎𝑑𝑡ℎ←𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ(N(𝑉𝑓𝑟𝑜𝑛𝑡𝑖𝑒𝑟),𝐾𝑑𝑒𝑔𝑟𝑒𝑒)# Execute
Bounded Star Scan
2.4𝐺𝑡+1←𝐺𝑡∪𝐺𝑑𝑒𝑝𝑡ℎ∪𝐺𝑏𝑟𝑒𝑎𝑑𝑡ℎ # Update sub-graph state
2.5𝜋 next←𝜋∪{𝑟 𝑛𝑒𝑥𝑡}# Append execution path
2.6returnHeuristicTraversal(𝑄 NL,𝑡,𝜋 next,𝐺𝑡+1,C)# Recursive
routing
3.else:
# Reached COMPLETE state or hit hard execution bound𝐾 𝑑𝑒𝑝𝑡ℎ
3.1return𝜋,𝐺 𝑡
neighborhoodN(𝑣) of bottleneck entities. To prevent memory ex-
plosion on super-nodes, we apply a Top-K pruning strategy (e.g., via
dense similarity), ensuring the neighborhood size never exceeds an
administrative bound 𝐾𝑑𝑒𝑔𝑟𝑒𝑒 . Consequently, the space complexity
of the traversal is strictly maintained at O(𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒). Fig-
ure 2(d) shows how breadth expansion retrieves all neighbors of
“Dunkirk”, capturing the 2018 Academy Award nomination that
would be missed by targeted queries alone.
Both operators iteratively feed the expanded results back into the
dispatcher (depicted as the “Update 𝐺𝑡+1” feedback loop in Figure 3)
until the information is semantically complete.
Justification & Running Example.Figure 2 illustrates the com-
plete heuristic expansion process for our running example:
(1)Initial Execution (Figure 2(a)):Without cache guidance, the
system generates a logical plan to query the nominated at-
tribute directly for “Inception” in 2018. Since Inception was
nominated in 2011, this returns an empty subgraph. The dis-
patcher evaluates the state as incomplete.
(2)Cache-Guided Refinement (Figure 2(b)):With cached se-
mantic plans fromC[movies][award] , the system compiles the
correct initial topology: first identify the director via directedBy .
However, this single-hop path still lacks the final answer.
(3)Depth Expansion (Figure 2(c)):Algorithm 3 routes execution
to the𝜎𝑑𝑒𝑝𝑡ℎ operator, chaining from “Christopher Nolan” to
his filmography: [“Interstellar”, “Dunkirk”, “Tenet”, ...].
(4)Breadth Expansion (Figure 2(d)):Simultaneously, the 𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ
operator executes bounded star scans for the retrieved entities.
For “Dunkirk”, this safely captures its local topology, including
the “Academy Award for Best Picture” (2018) node. The dis-
patcher evaluates the subgraph as complete and terminates the
traversal.
This architecture guarantees maximum retrieval coverage for
complex multi-hop QA. The strict deterministic bounds ( 𝐾𝑑𝑒𝑝𝑡ℎ≤3,

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
𝐾𝑑𝑒𝑔𝑟𝑒𝑒 pruning) proactively prevent catastrophic execution sce-
narios (e.g., unbounded API calls or OOM errors) common in un-
constrained LLM agents. Meanwhile, the cache-guided heuristic
dispatching significantly improves reasoning accuracy compared
to static, single-round planning (as demonstrated in Section 5.7).
4.4 Cache Pre-Warming via Auto-Generation
As discussed in Section 3, deploying CacheRAG with an entirely
empty memory degrades the system to a stateless planner during
the initial queries. To alleviate this cold-start bottleneck, we imple-
ment an offline Auto-Generation module to populate the semantic
cacheCwith fundamental operational heuristics prior to deploy-
ment. Rather than relying on human annotations, this process acts
as an automated exploratory view materialization, executed in three
straightforward steps:
1. Star-Schema Sampling and Synthesis.We uniformly sample
entities along with their 1-hop relational neighbors (i.e., local star-
schemas consisting of valid KG triples). Instead of assuming these
shallow structures can only answer simple queries, we leverage an
LLM to synthesize diverse natural language questions 𝑄NLbased
on these subgraphs. Crucially, as shown in our prompt skeleton,
the LLM is explicitly instructed to generate not only single-fact
questions but also compositional queries requiring multi-triple
reasoning (e.g., comparison and summarization).
Auto-Generation Prompt Skeleton
You are an expert in question generation. Your task is to generate rea-
sonable natural language questions based on the provided KG triples. -
Generate up to 5 questions that humans might naturally ask. - Avoid
meaningless ID-based or purely numerical questions (Respond NA if
unavoidable). - **Crucial:** Generate both simple questions (1 triple)
AND complex questions (e.g., comparison and summarization involving
multiple triples). Output format: [QUESTION], [TRIPLES], [ANSWER]
Current triples: [SAMPLED TRIPLES]
2. Automated Filtering and Compilation.To ensure the high qual-
ity of the materialization, the LLM acts as a secondary evaluator to
filter out illogical or unanswerable generated pairs. For the surviv-
ing valid tuples, the system deterministically compiles the selected
‘[TRIPLES]’ into executable API calls or SPARQL queries, forming
the physical plan 𝜋. The resulting⟨𝑄NL,𝜋,𝐴⟩ tuples are ingested
into the hierarchical cache to serve as the initial in-context learning
baseline.
3. Handling Zero-Similarity and Online Growth.A natural limita-
tion of offline random sampling is that it cannot guarantee complete
structural coverage of all possible user intents. If a novel online
query exhibits near-zero similarity with all pre-warmed examples
(i.e., a severe cache miss), the system handles this cold-start grace-
fully. Under the MMR formulation, if the relevance score falls below
a minimum threshold, the Query Compiler receives an empty con-
text𝑆∗=∅and degrades to zero-shot execution relying entirely on
the local schemaS 𝑙𝑜𝑐𝑎𝑙 (Section 4.1).
To prevent execution failure in this state, the system determin-
istically falls back on our Bounded Subgraph Operators ( 𝜎𝑑𝑒𝑝𝑡ℎ
and𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ detailed in Section 4.3) to broadly explore the graph.Once this heuristic traversal successfully locates the answer, the
newly discovered execution path is dynamically added to C. Thus,
the auto-generation module only needs to provide a foundational
“warm start”; the cache subsequently grows and adapts organically
to long-tail queries through regular online interactions.
5 Experiments
We present the experimental datasets (Section 5.1), metrics (Sec-
tion 5.2), baselines (Section 5.3), and configurations (Section 5.4).
Our evaluation includes the following aspects: (1) comparison
against the baselines on the CRAG dataset (Section 5.5), (2) gener-
alizability of CacheRAG on SPARQL-based datasets (Section 5.6),
(3) ablation study (Section 5.7), (4) robustness of LLM backbones
(Section 5.8), (5) robustness of two-layer indexing (Section 5.9), (6)
efficiency (Section 5.10), (7) scalability (Section 5.11), (8) parame-
ter sensitivity (Section A.4), (9) detailed head, torso, tail analysis
(Section A.6), and (10) analysis on KG domain routing (Section A.7).
5.1 Experimental Datasets
We consider the widely used CRAG [ 33], QALD-10-en [ 28], We-
bQSP [ 36], and CWQ [ 27] datasets in our experiments. Details are
presented in Section A.1.
5.2 Metrics and Evaluation Scheme
We adopted the auto-evaluation scheme used in CRAG [ 33] to
evaluate the correctness of the answers on the CRAG dataset. Ac-
cording to [ 33], the detailed criterion for accurate, missing, and
hallucination is defined as follows:
•Correct: The response correctly answers the user’s question and
contains no hallucinated content, or the response provides a
useful answer to the user’s question but may contain minor
errors that do not harm the usefulness of the answer
•Missing: The response is “I don’t know”, “I’m sorry I can’t find
...”, a system error such as an empty response, or a request from
the system to clarify the original question.
•Incorrect: The response provides wrong or irrelevant information
to answer the user’s question.
Following the settings of CRAG [ 33], we adopted the Llama-
3.1-70B-Instruct model as the auto-evaluation critic to classify the
answers into correct, missing, and incorrect based on the question
and the ground truth answer. The auto-evaluation critic achieves
an overall reliability of 98.4%, demonstrating the reliability of the
auto-evaluation scheme (Details in Section A.2). We report the
accuracy (𝐴, correct answer rate), miss rate ( 𝑀, missing answer
rate), hallucination rate ( 𝐻, incorrect answer rate), and additionally
thetruthfulness score ( 𝑇), which is defined as 𝑇= 1∗𝐴+ 0∗𝑀− 1∗𝐻
(the difference between accuracy and hallucination rate), penalizing
the cases where the methods hallucinate.
As for the SPARQL-based datasets QALD-10-en, WebQSP, and
CWQ, we followed their respective evaluation metric: the Hit@1
score. Full details of metrics and evaluation scheme are presented
in Section A.2

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
5.3 Baselines
We compare our approach against the following baselines, which
can be categorized into LLM base models (GPT-4o [ 13],Llama-
3.1-70B-Instruct [ 7], andDeepseek-chat-V3-0324 [ 19]), LLM tool-
calling models (StructGPT [ 14]variants), KDD Cup-winning so-
lutions (db3 [ 29]andapex [ 24]), and SPARQL-based state-of-the-art
solutions (ToG [ 26],ToG-2 [ 22],sparql-qa [ 3],Decaf [ 37], andDe-
caf [37]). Full details are presented in Section A.3.
5.4 Configurations
All experiments were performed on four NVIDIA A100 GPUs (80GB).
We used the Deepseek-chat-V3-0324 model as the base LLM for
CacheRAG across the CRAG, QALD-10-en, WebQSP, and CWQ
datasets. The default MMR penalty parameter 𝜆is set to 0.5. The
sample size𝑘for the structurally diverse plans provided to the LLM
Query Compiler is set to 5. The similarity measure ( Sim(·) ) used
in the MMR formulation is based on BM25. We apply Chain-of-
Thought reasoning during the physical query compilation phase
to facilitate complex logical mapping. For the multi-domain CRAG
dataset, we additionally deploy a multi-KG domain router (based on
the Llama-3.1-8B-Instruct model with temperature set to 0) to assist
the Logical Parser in determining the domain hint 𝑑. For the single-
domain SPARQL-based datasets (QALD-10-en, WebQSP, and CWQ),
we utilize the llama-7b-wikiwebquestions-qald7 model1[31] to
aid the extraction of the raw topic entity 𝑒rawwithin the ISR. Since
these datasets do not distinguish domains, we gracefully degrade
the two-layer hierarchical index into a one-layer aspect index, stor-
ing all cached plans within the same global domain bucket. During
the Bounded Breadth Expansion ( 𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ ), a Dense Passage Re-
trieval (DPR) model [ 16] is employed as the Top- 𝐾pruning strategy
to filter irrelevant schema predicates. Specifically, we compute the
similarity between the natural language question 𝑄NLand the 1-
hop predicates, preserving the top-30 ( 𝐾𝑑𝑒𝑔𝑟𝑒𝑒 =30) predicates to
strictly maintain bounded space complexity. When adapting the
system to a new KG, users only need to map specific information
into the predefined slots of the Backend Adapter templates (e.g.,
injecting the user query into [ 𝑄NL]). This schema-agnostic configu-
ration is straightforward and requires no deep expertise in prompt
engineering. We estimate that configuring these structural adapter
templates typically requires only 10 to 20 minutes of manual effort
per dataset, depending on the physical KG schema’s complexity.
These templates function as generalizable compilation blueprints,
ensuring that CacheRAG’s deployment is highly efficient and re-
producible across heterogeneous data sources.
5.5 Main Results
We present the experimental results on the CRAG dataset in Table 1.
In general, CacheRAG achieves the best performance on three out
of four evaluation metrics compared to the baselines. Specifically,
CacheRAG outperforms the state-of-the-art models by 13.2% and
17.5% in terms of accuracy and truthfulness scores, alongside a
dramatically lower miss rate (over 38% lower than the StructGPT
variants).
Regarding the modest increase in the hallucination metric (be-
tween 3% and 6.5%) compared to some baselines, it is crucial to
1https://huggingface.co/stanford-oval/llama-7b-wikiwebquestions-qald7distinguish between two types of hallucinations. As discussed in
Section 4.1, CacheRAG effectively eliminatesschema hallucination
(i.e., generating non-existent graph predicates that cause execution
failures), evidenced by our 98.9% physical compilation executable
rate. The slight increase observed in the evaluation metric reflects
answer content hallucinationduring the final LLM summarization
phase. Because CacheRAG employs bounded depth and breadth
expansions to retrieve significantly more comprehensive and long-
tail multi-hop contexts, the final Summarizer faces a more complex
synthesis task. Given the substantial 13.2% gain in overall accu-
racy and the dramatic reduction in miss rate, this modest increase
in summarization hallucination is a well-contained trade-off for
resolving the fundamental bottleneck of incomplete retrieval.
We believe the reason why CacheRAG outperforms the state-of-
the-art baselines is two-fold: 1) The implementation of a caching
mechanism provides a strong reference for the LLM planner to
learn from its past interactions. The LLM base models, LLM tool
calling, and KDD Cup-winning solutions all ignore the importance
of valuable QA experience. As a result, the planners in these meth-
ods can only generate retrieval plans based on the current context
instead of continuously learning from relevant QA experiences.
Our CacheRAG approach helps the LLM planner effectively learn
from QA experiences and thus achieve higher accuracy and truth-
fulness scores compared to existing methods. 2) Our KG exploration
method provides better retrieval coverage. Instead of enhancing the
retrieval paths in both breadth and depth dimensions, existing base-
lines either ignore the KG content (LLM Base Models) or neglect
exploration in breadth (LLM Tool Calling Models and KDD Cup
Winning Solutions). Our solution implements additional breadth
expansion in case the KG content retrieved after depth expansion
is still not sufficient for QA, which provides better KG recall (the
retrieval recall is improved from0 .756to0.927). As a result, our
method presents significantly lower miss rates.
Table 1: Main Experimental Results on CRAG dataset.
CacheRAG improves over state-of-the-art by 13.2% and 17.5%
in terms of accuracy and truthfulness score.
Model Accu. Hall. Miss. Truth.
GPT-4o 0.341 0.090 0.569 0.251
Llama 0.306 0.080 0.614 0.227
Deepseek 0.400 0.140 0.460 0.259
StructGPT (GPT-4o) [14] 0.4150.0470.542 0.368
StructGPT (Llama) [14] 0.318 0.057 0.623 0.261
StructGPT (Deepseek) [14] 0.475 0.080 0.445 0.395
apex [24] 0.692 0.156 0.152 0.536
db3 [29] 0.555 0.176 0.259 0.379
CacheRAG0.8240.1120.064 0.711
5.6 Generalizability
To comprehensively evaluate the generalizability of CacheRAG
on other KGQA datasets, we adapted the CacheRAG method for
SPARQL-based datasets: QALD-10-en, WebQSP, and CWQ datasets.
Specifically, instead of prompting the CacheRAG model to generate
API function calls, we prompted it to generate SPARQL queries and
execute those queries to get relevant KG triples for summarization.

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2: Experimental results on the QALD-10-en, WebQSP,
and CWQ datasets. The prior fine-tuning SOTAs include the
best-known fine-tuned methods on each dataset: 𝛼: sparql-
qa [3];𝛽: Decaf [ 37];𝛾: CBR [ 6]. The baseline performance
is directly taken from ToG and ToG-2. Unlike other meth-
ods, ToG-2 additionally links to external Wikipedia pages to
obtain context information. CacheRAG outperforms all the
state-of-the-art baselines on the three datasets.
Model QALD-10 WebQSP CWQ
Prior Finetuning SOTAs 0.454𝛼0.821𝛽0.704𝛾
ToG [26] 0.502 0.762 0.695
ToG-2 [22] 0.541 0.811 -
CacheRAG0.587 0.840 0.736
We compared the CacheRAG method against the respective state-
of-the-art methods on these datasets.
We present the experimental results in Table 2. In general, CacheRAG
outperforms all the state-of-the-art solutions on the three datasets.
Specifically, CacheRAG improves the state-of-the-art performance
by4.6%,1.9%, and3.2%on the QALD-10-en, WebQSP, and CWQ
datasets, respectively. We are surprised to see that our approach
outperforms ToG-2 and prior fine-tuning SOTAs, despite the fact
that ToG-2 additionally relies on external Wikipedia pages and that
our method does not include a fine-tuning step to improve sum-
marization quality, which showcases the effectiveness of continual
learning on QA records and the advantages of KG exploration in
depth and breadth.
To investigate why our method outperforms the baseline, we
further analyze the performance of several variants of CacheRAG
on the SPARQL-based KGs and present the results in Table 3. Specif-
ically, we consider CacheRAG w/o exploration (remove depth and
breadth exploration), CacheRAG w/o caching (remove cache), and
CacheRAG prompt (adopt the same set of prompts used by ToG).
We found that the KG exploration and caching components were
most critical to maintaining CacheRAG’s performance, with perfor-
mance declining by an average of 5.6% and 2.8% respectively after
their removal. In contrast, the impact of changing the prompt on
the performance of CacheRAG is within 1%.
Table 3: Detailed analysis.
Model QALD-10 WebQSP CWQ
CacheRAG w/o exploration 0.552 0.787 0.656
CacheRAG w/o caching 0.576 0.816 0.688
CacheRAG prompt 0.584 0.832 0.728
5.7 Ablation Study
To fully evaluate the effect of each components of CacheRAG, we
conducted an ablation study by comparing the performance of
CacheRAG and the following variants.
•CacheRAG w/o depth:represents the variant that we do not
perform depth expansion, we only perform single round retrieval
based on the initial planning.
•CacheRAG w/o breadth:represents the variant that we do not
perform additional breadth expansion, we only perform depth
expansion for KG retrieval paths.•CacheRAG w/o caching:represents the variant that removes
the dynamic semantic cache, forcing the Query Compiler to rely
on a static set of few-shot examplar plans.
•CacheRAG w/o MMR:represents the variant that we replace
the MMR score as BM25.
•CacheRAG one layer:represents the variant that we replace
the two-layer index as one-layee domain index.
•CacheRAG w/o auto-generation:represents the variant that
we consider empty example cache to start.
We present the experimental results in Table 4. Specifically,
depth and breadth expansions are the most influential modules
in CacheRAG: removing depth and breadth expansion leads to 22%
and 18.1% performance drops in terms of truthfulness, while it re-
sults in a significant increase in the answering miss rate (26.1%
and 19%). We attribute this phenomenon to the fact that removing
depth and breadth expansions decreases the retrieval recall rate;
as a result, the miss rate increases, and the overall accuracy and
truthfulness decrease. We further notice that example caching is
another crucial component that significantly influences the perfor-
mance of CacheRAG: by removing the hierarchical semantic cache,
the accuracy and truthfulness drop by 4.3% and 5.5%, while the
miss rate increases by 3%, which indicates that the quality of gen-
erated retrieval plans declines when we remove example caching
for the LLM planner to learn. The MMR, layered index, and auto-
generation mechanism are also influential factors: removing these
leads to truthfulness drops of 2.9%, 2.8%, and 1.8%.
Table 4: Ablation Study.
Model Accu. Hall. Miss. Truth.
CacheRAG w/o depth 0.5830.0920.325 0.491
CacheRAG w/o breadth 0.638 0.108 0.254 0.530
CacheRAG w/o caching 0.781 0.125 0.094 0.656
CacheRAG w/o MMR 0.804 0.123 0.073 0.682
CacheRAG one layer 0.808 0.125 0.067 0.683
CacheRAG w/o auto-gen 0.811 0.118 0.071 0.693
CacheRAG0.8240.1120.064 0.711
5.8 LLM Model Robustness
We evaluated the robustness of CacheRAG with different LLM
backbones by replacing the LLM base model with three widely used
LLMs: Llama-3.1-70B-Instruct, GPT-4o, and Claude-3.5-Sonnet and
present the experimental results in Table 5. We observe that the
performance of CacheRAG is relatively stable: it changes by up
to 5% when we replace the base model with other LLMs, which
validates the robustness of the CacheRAG pipeline across different
LLM backbones.
Table 5: Robustness of LLM Backbones.
Model Accu. Hall. Miss. Truth.
CacheRAG (Llama) 0.791 0.125 0.084 0.666
CacheRAG (GPT) 0.818 0.114 0.067 0.704
CacheRAG (Claude) 0.830 0.111 0.059 0.720
CacheRAG (DeepSeek) 0.824 0.112 0.064 0.711

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
5.9 Indexing Consistency and Robustness
To validate the stability of our two-layer hierarchical index (Domain
→Aspect) proposed in Section 4.2, we evaluate the consistency of
the LLM-based logical parser. Inconsistencies in extracting the do-
main and aspect attributes could potentially lead to cache retrieval
failures. To quantify this, we repeated the indexing process for each
question in the CRAG dataset five times with the LLM temperature
strictly set to 0. The empirical results show an extremely low in-
consistent indexing rate of only1 .1%, confirming that the logical
parsing phase is highly stable. Furthermore, CacheRAG is designed
to be robust against such rare anomalies: if a user’s question is
assigned to a non-existent or empty aspect bucket, our retrieval al-
gorithm (Algorithm 2) deterministically relaxes the search space to
the broader domain level. This fallback mechanism ensures that rel-
evant historical plans can always be retrieved, effectively mitigating
the impact of any parsing inconsistencies.
5.10 Efficiency
We report the mean inference time of CacheRAG against the base-
lines on CRAG in this section. Specifically, the mean inference time
of CacheRAG is 9.44s (Logical Parsing and KG Routing, Hierarchi-
cal Cache Retrieval, Physical Plan Compilation, Bounded Depth
and Breadth Expansion, Heuristic Dispatching, and Summarization
take 2.75s, 0.02s, 0.49s, 3.93s, 0.47s, and 1.78s), while the state-of-
the-art baseline, Apex, takes 5.96s. Although the inference time
of CacheRAG is higher than that of the state-of-the-art solution
on CRAG, we believe this is reasonable, as our approach aims to
improve the effectiveness of existing methods and additionally re-
quires depth and breadth expansion of KG paths, which needs more
time to complete. In other words, our approach trades efficiency
for effectiveness. However, we would like to point out that the
mean inference times of CacheRAG and the state-of-the-art so-
lution, Apex, are of the same order of magnitude. Therefore, we
believe the inference time of our proposed CacheRAG approach is
generally acceptable.
Figure 5: Scalability Experiments.5.11 Time and Memory Complexity and
Scalability
We analyze the time and memory complexity of CacheRAG us-
ing the formal bounds defined in Section 4. For semantic caching,
the hierarchical index routing takes O(1)dictionary lookup time,
while the MMR scheduling takes O(𝑏log𝑏) time, utilizingO(𝑁)
total space (where 𝑏≪𝑁 is the localized bucket size and 𝑁is
the global cache size). During execution, the Bounded Depth Ex-
pansion (Index Nested Loop Join) takes O(𝐾𝑑𝑒𝑝𝑡ℎ log𝑛) time and
O(𝐾𝑑𝑒𝑝𝑡ℎ)space, where 𝐾𝑑𝑒𝑝𝑡ℎ≤3is the maximum hop limit
and𝑛is the total number of KG triples. The Bounded Breadth Ex-
pansion (Star-Pattern Scan) processes the frontier nodes, taking
O(𝐾𝑑𝑒𝑝𝑡ℎ·(𝐾𝑑𝑒𝑔𝑟𝑒𝑒+log𝑛)) time and strictlyO(𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒)
space, where 𝐾𝑑𝑒𝑔𝑟𝑒𝑒 is the Top-𝐾pruning bound. The total time
and memory induced by LLM interactions (Parser, Compiler, Dis-
patcher, and Summarizer) are O(𝐿·𝐾𝑑𝑒𝑝𝑡ℎ)andO(𝐾𝑑𝑒𝑝𝑡ℎ), where
𝐿is the bounded LLM inference latency. The overall online time
complexity isO(𝑏log𝑏+𝐾 𝑑𝑒𝑝𝑡ℎ log𝑛+𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒+𝐿·𝐾𝑑𝑒𝑝𝑡ℎ).
Since𝑏,𝐾𝑑𝑒𝑝𝑡ℎ ,𝐾𝑑𝑒𝑔𝑟𝑒𝑒 , and𝐿are treated as strict system constants,
CacheRAG is highly scalable. Detailed step-by-step derivations are
presented in Appendix A.5.
Empirical Scalability on Synthetic KGs.To empirically validate
the theoretical complexity, we evaluated the retrieval latency on
synthetic KGs of varying sizes. Following the schema of Wikidata,
we generated synthetic KGs containing from 40,000 up to 1.28
million triples, with entity degrees approximating Wikidata’s actual
distribution. We executed 50 point queries (depth expansion) and
50 star queries (breadth expansion) across these KGs. As shown in
Figure 5, the execution time for both query types exhibits a clear
logarithmic growth O(log𝑛) as the KG size increases. This confirms
that CacheRAG’s bounded graph operators are highly scalable and
the retrieval mechanism will not become a performance bottleneck
on larger-scale graphs.
5.12 LRU Cache Design
To further explore the design of two-layered indexing, we imple-
mented a Least Recently Used (LRU) CacheRAG variant (CacheRAG
(LRU)), where a maximum of 10 examples are kept under each
bucket. This design is suitable for scenarios with limited caching
storage space. Compared to storing all historical data, we saved64%
of space on CRAG dataset. As shown in Table 6, the performance
drops by less than2%after implementing the LRU caching, validat-
ing the robustness of the LRU-based caching variant of CacheRAG.
Table 6: LRU-based variant results on CRAG.
Model Accu. Hall. Miss. Truth.
CacheRAG (LRU) 0.808 0.116 0.076 0.692
CacheRAG (DeepSeek) 0.824 0.112 0.064 0.711
6 Conclusion
This paper presents CacheRAG, a systematic architecture that trans-
forms stateless LLM planners into continual learners for Knowl-
edge Graph Question Answering. By synergistically integrating a
schema-constrained semantic parser, a diversity-aware hierarchical

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
cache, and deterministically bounded graph operators, CacheRAG
resolves critical execution bottlenecks like schema hallucination
and combinatorial explosion. Extensive evaluations validate this
stateful paradigm: CacheRAG significantly outperforms state-of-
the-art baselines, achieving a 13.2% accuracy and 17.5% truthfulness
increase on the complex CRAG benchmark. Furthermore, it gen-
eralizes robustly to SPARQL-based KGs and scales logarithmically.
Ultimately, CacheRAG provides a principled foundation for em-
bedding history-aware, verifiable continual learning into future
LLM-augmented database systems.
References
[1]2022. Wikidata Query Blazegraph. Retrieved Nov 18, 2025 from https://github.
com/wikimedia/wikidata-query-blazegraph?tab=readme-ov-file
[2]Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Mahapatra, Archit
Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu, and Shiv Saini. 2025.
Cache-craft: Managing chunk-caches for efficient retrieval-augmented genera-
tion.Proceedings of the ACM on Management of Data3, 3 (2025), 1–28.
[3]Manuel Borroto, Francesco Ricca, Bernardo Cuteri, and Vito Barbara. 2022.
SPARQL-QA enters the QALD challenge. InProceedings of the 7th Natural Lan-
guage Interfaces for the Web of Data (NLIWoD) co-located with the 19th European
Semantic Web Conference, Hersonissos, Greece, Vol. 3196. 25–31. https://ceur-
ws.org/Vol-3196/paper3.pdf
[4]Shulin Cao, Jiaxin Shi, Zijun Yao, Xin Lv, Jifan Yu, Lei Hou, Juanzi Li, Zhiyuan Liu,
and Jinghui Xiao. 2022. Program Transfer for Answering Complex Questions over
Knowledge Bases. InProceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 8128–8140. doi:10.18653/v1/
2022.acl-long.559
[5]Zi-Yuan Chen, Chih-Hung Chang, Yi-Pei Chen, Jijnasa Nayak, and Lun-Wei
Ku. 2019. UHop: An Unrestricted-Hop Relation Extraction Framework for
Knowledge-Based Question Answering. InProceedings of the 2019 Conference
of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, Volume 1 (Long and Short Papers). 345–356.
doi:10.18653/v1/N19-1031
[6]Rajarshi Das, Manzil Zaheer, Dung Thai, Ameya Godbole, Ethan Perez, Jay-Yoon
Lee, Lizhen Tan, Lazaros Polymenakos, and Andrew Mccallum. 2021. Case-based
Reasoning for Natural Language Queries over Knowledge Bases. InProceedings
of the 2021 Conference on Empirical Methods in Natural Language Processing.
9594–9611. doi:10.18653/v1/2021.emnlp-main.755
[7]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al.2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783(2024).
doi:10.48550/arXiv.2407.21783
[8]Wolfgang Fahl, Tim Holzheim, Andrea Westerinen, Christoph Lange, and Stefan
Decker. 2022. Getting and hosting your own copy of Wikidata.. InWikidata@
ISWC.
[9]Goetz Graefe and William J McKenna. 1993. The volcano optimizer generator: Ex-
tensibility and efficient search. InProceedings of IEEE 9th international conference
on data engineering. IEEE, 209–218.
[10] Yu Gu, Xiang Deng, and Yu Su. 2023. Don’t Generate, Discriminate: A Proposal
for Grounding Language Models to Real-World Environments. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). 4928–4949. doi:10.18653/v1/2023.acl-long.270
[11] Yu Gu and Yu Su. 2022. ArcaneQA: Dynamic Program Induction and Con-
textualized Encoding for Knowledge Base Question Answering. InProceedings
of the 29th International Conference on Computational Linguistics. 1718–1731.
https://aclanthology.org/2022.coling-1.148/
[12] Xixin Hu, Xuan Wu, Yiheng Shu, and Yuzhong Qu. 2022. Logical form generation
via multi-task learning for complex question answering over knowledge bases.
InProceedings of the 29th International Conference on Computational Linguistics.
1687–1696. https://aclanthology.org/2022.coling-1.145/
[13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024). doi:10.48550/arXiv.
2410.21276
[14] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong
Wen. 2023. StructGPT: A General Framework for Large Language Model to Reason
over Structured Data. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing. 9237–9251. doi:10.18653/v1/2023.emnlp-main.574
[15] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe Liu, and
Xin Jin. 2025. Ragcache: Efficient knowledge caching for retrieval-augmented
generation.ACM Transactions on Computer Systems44, 1 (2025), 1–27.[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). 6769–6781. doi:10.18653/v1/
2020.emnlp-main.550
[17] Yunshi Lan and Jing Jiang. 2020. Query Graph Generation for Answering Multi-
hop Complex Questions from Knowledge Bases. InProceedings of the 58th Annual
Meeting of the Association for Computational Linguistics. 969–974. doi:10.18653/
v1/2020.acl-main.91
[18] Yunshi Lan, Shuohang Wang, and Jing Jiang. 2019. Knowledge base question
answering with topic units.(2019). InProceedings of the Twenty-Eighth Interna-
tional Joint Conference on Artificial Intelligence. 5046–5052. https://www.ijcai.
org/proceedings/2019/0701.pdf
[19] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report.arXiv preprint arXiv:2412.19437(2024).
[20] Hanwen Liu, Qihan Zhang, Ryan Marcus, and Ibrahim Sabek. 2025. Serag: Self-
evolving rag system for query optimization. (2025).
[21] Kangqi Luo, Fengli Lin, Xusheng Luo, and Kenny Zhu. 2018. Knowledge base
question answering via encoding of complex query graphs. InProceedings of the
2018 conference on empirical methods in natural language processing. 2185–2194.
doi:10.18653/v1/D18-1242
[22] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
InThe Thirteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=oFBu7qaZpS
[23] Barlas Oguz, Xilun Chen, Vladimir Karpukhin, Stan Peshterliev, Dmytro Okhonko,
Michael Schlichtkrull, Sonal Gupta, Yashar Mehdad, and Scott Yih. 2022. UniK-QA:
Unified Representations of Structured and Unstructured Knowledge for Open-
Domain Question Answering. InFindings of the Association for Computational
Linguistics: NAACL 2022. 1535–1546. doi:10.18653/v1/2022.findings-naacl.115
[24] Jie Ouyang, Yucong Luo, Mingyue Cheng, Daoyu Wang, Shuo Yu, Qi Liu, and
Enhong Chen. 2024. Revisiting the solution of meta kdd cup 2024: Crag.arXiv
preprint arXiv:2409.15337(2024). https://openreview.net/forum?id=PUzLjWIgqC
[25] Yiheng Shu, Zhiwei Yu, Yuhan Li, Börje Karlsson, Tingting Ma, Yuzhong Qu,
and Chin-Yew Lin. 2022. TIARA: Multi-grained Retrieval for Robust Question
Answering over Large Knowledge Base. InProceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing. 8108–8121. doi:10.18653/v1/
2022.emnlp-main.555
[26] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InThe Twelfth International Conference on Learning Representations. https:
//openreview.net/forum?id=nnVO1PvbTv
[27] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and
Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans,
Louisiana, 641–651. doi:10.18653/v1/N18-1059
[28] Ricardo Usbeck, Xi Yan, Aleksandr Perevalov, Longquan Jiang, Julius Schulz,
Angelie Kraft, Cedric Möller, Junbo Huang, Jan Reineke, Axel-Cyrille Ngonga
Ngomo, Muhammad Saleem, and Andreas Both. 2023. QALD-10: The 10th chal-
lenge on question answering over linked data.Semantic Web(2023). https:
//api.semanticscholar.org/CorpusID:265577096
[29] Yikuan Xia, Jiazun Chen, and Jun Gao. 2024. Winning Solution For Meta KDD
Cup’24.arXiv preprint arXiv:2410.00005(2024). https://openreview.net/forum?
id=oWNPeoP1uC
[30] Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro
Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I Wang, et al .2022.
UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with
Text-to-Text Language Models. InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing. 602–631. doi:10.18653/v1/2022.emnlp-
main.39
[31] Silei Xu, Shicheng Liu, Theo Culhane, Elizaveta Pertseva, Meng-Hsi Wu, Sina
Semnani, and Monica Lam. 2023. Fine-tuned LLMs Know More, Hallucinate
Less with Few-Shot Sequence-to-Sequence Semantic Parsing over Wikidata. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing. 5778–5791. doi:10.18653/v1/2023.emnlp-main.353
[32] Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang,
Joseph E Gonzalez, and Bin Cui. 2024. Buffer of thoughts: Thought-augmented
reasoning with large language models.Advances in Neural Information Processing
Systems37 (2024), 113519–113544.
[33] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal
Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian
Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen
Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
Anuj Kumar, Wen tau Yih, and Xin Luna Dong. 2024. CRAG – Comprehensive
RAG Benchmark.arXiv preprint arXiv:2406.04744(2024). https://arxiv.org/abs/
2406.04744
[34] Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou, and Caiming Xiong.
2022. RNG-KBQA: Generation Augmented Iterative Ranking for Knowledge
Base Question Answering. InProceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 6032–6043.
doi:10.18653/v1/2022.acl-long.417
[35] Scott Wen-tau Yih, Ming-Wei Chang, Xiaodong He, and Jianfeng Gao. 2015. Se-
mantic parsing via staged query graph generation: Question answering with
knowledge base. InProceedings of the Joint Conference of the 53rd Annual Meet-
ing of the ACL and the 7th International Joint Conference on Natural Language
Processing of the AFNLP. https://aclanthology.org/P15-1128.pdf
[36] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina
Suh. 2016. The value of semantic parse labeling for knowledge base question an-
swering. InProceedings of the 54th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 2: Short Papers). 201–206. https://aclanthology.org/P16-2033.pdf
[37] Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun
Wang, Yiqun Hu, William Yang Wang, Zhiguo Wang, and Bing Xiang. 2022.
DecAF: Joint Decoding of Answers and Logical Forms for Question Answering
over Knowledge Bases. InThe Eleventh International Conference on Learning
Representations. https://openreview.net/pdf?id=XHc5zRPxqV9
[38] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi
Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al .2025. AFlow:
Automating Agentic Workflow Generation. InThe Thirteenth International Con-
ference on Learning Representations.
[39] Lingxi Zhang, Jing Zhang, Yanling Wang, Shulin Cao, Xinmei Huang, Cuiping
Li, Hong Chen, and Juanzi Li. 2023. FC-KBQA: A Fine-to-Coarse Composition
Framework for Knowledge Base Question Answering. InProceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). 1002–1017. doi:10.18653/v1/2023.acl-long.57

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Appendix
A.1 Dataset Details
To evaluate the performance of our method, we consider the widely
used CRAG [ 33] benchmark, where the KGs cover five distinct
domains and are accessed through API functions. We use the KG
questions in CRAG to conduct experiments.
The statistics of the CRAG KG private testing set are presented
in Table 7. CRAG categorizes the questions into head, torso, and
tail based on the popularity of the topic entity. It also considers the
domain, question types, and timeliness of the questions, categoriz-
ing them into different subsets. In total, we experimented on the
head, torso, and tail question splits, which have 187, 203, and 188
questions, respectively.
Table 7: Statistics of each question type in different question
splits of CRAG.
head torso tail
open 24 24 24
finance 57 59 58
movie 60 58 59
music 7 31 7
sports 39 31 40
simple 98 104 94
simple_w_condition 23 26 28
set 9 6 5
comparison 17 22 21
aggregation 16 18 21
multi-hop 8 9 6
post-processing 3 3 3
false premise 13 15 10
real-time 32 38 36
fast-changing 31 32 39
slow-changing 22 27 19
static 102 106 94
total 187 203 188
We also included SPARQL-based datasets used by state-of-the-
art approaches: QALD-10-en [ 28], WebQSP [ 36], and CWQ [ 27]
datasets to comprehensively evaluate the performance of CacheRAG.
A.2 Metric and Evaluation Scheme Details
We present the details on the reliability of the auto evaluation critic
in Table 8.
Table 8: The Auto-evaluation Accuracy of Llama-3.1 70B In-
struct Model.
Llama-3.1 70B
correct answers98.2%
incorrect answers96.8%
missing100%
overall reliability98.4%A.3 Baseline Details
We compare our approach against the following baselines, which
can be categorized into LLM base models, LLM tool-calling models,
KDD Cup-winning solutions, and SPARQL-based state-of-the-art
solutions.
A.3.1 LLM Base Models.On CRAG dataset, we includeGPT-4o [ 13],
Llama-3.1-70B-Instruct [ 7], andDeepseek-chat-V3-0324 [ 19]base
models as baselines, which covers the state-of-the-art open-sourced
and close-sourced LLMs.
A.3.2 LLM Tool Calling Models.On CRAG, we further include
StructGPT [ 14], which enables GPT-4o, Llama-3.1-70B-Instruct, and
Deepseek-chat-V3-0324 base models to freely perform tool calling.
Specifically, we provide the metadata of the API functions (i.e., the
function name, parameters, descriptions, and sample use cases) so
that the LLMs can freely chain and call the API functions to retrieve
relevant content from the KGs based on their own planning and
reasoning abilities.
A.3.3 KDD Cup Winning Solutions.Since CRAG is used to host
the KDD Cup competition in 2024, we also include the winning
solutions to establish the state-of-the-art baselines on CRAG:
•db3 [ 29]:The design of db3 jointly considers inputs from the KG
and web content. We enter “<EMPTY>” into the web content
module of db3 to adapt it to our KGQA settings. A router-based
adaptive RAG pipeline is introduced in apex to retrieve relevant
KG content for answer generation. We use the Deepseek-chat-
V3-0324 model as the base model.
•apex [ 24]:A router-based adaptive RAG pipeline is introduced
to retrieve relevant KG content for answer generation. Similar
to db3, we input “<EMPTY>” as the web content input to the
model to adapt the apex solution to our KGQA settings. We use
the Deepseek-chat-V3-0324 model as the base model.
A.3.4 SPARQL-based State-of-the-art Solutions.On the SPARQL-
based datasets QALD-10-en, WebQSP, and CWQ, we consider the
following state-of-the-art methods:
•ToG [ 26]:ToG integrates large language models (LLMs) with
knowledge graphs (KGs). Through beam search, LLMs are used
to iteratively explore reasoning paths on KGs to enhance the
deep reasoning ability of LLMs, improve knowledge traceability
and correctness, and achieve state-of-the-art performance on
multiple datasets.
•ToG-2 [ 22]:Following ToG, ToG-2 is a hybrid RAG framework that
tightly couples knowledge graphs and documents for iterative
retrieval, enabling deep and faithful reasoning in LLMs with
state-of-the-art performance on multiple datasets. Apart from
the KG itself, ToG-2 additionally relies on external Wikipedia
pages as an auxiliary knowledge source.
•sparql-qa [ 3]:sparql-qa uses a neural architecture combining
NMT and NER with input processing and QQT format to translate
natural language into SPARQL.
•Decaf [ 37]:Decaf jointly generates answers and logical forms for
knowledge base question answering, combines the advantages
of both forms, and uses text retrieval instead of entity linking to
enhance generality.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
0.0 0.2 0.4 0.6 0.8 1.0
0.7000.7250.7500.7750.8000.8250.8500.8750.900
(a) accuracy
0.0 0.2 0.4 0.6 0.8 1.0
0.0000.0250.0500.0750.1000.1250.1500.1750.200
 (b) hallucination
0.0 0.2 0.4 0.6 0.8 1.0
0.0000.0250.0500.0750.1000.1250.1500.1750.200
 (c) miss
0.0 0.2 0.4 0.6 0.8 1.0
0.6000.6250.6500.6750.7000.7250.7500.7750.800
 (d) truthfulness
Figure 6: The parameter𝜆’s sensitivity of CacheRAG on CRAG dataset.
•CBR [ 6]:CBR is a neuro-symbolic method. It retrieves similar
question cases, reuses their logical form components, and revises
the generated form using KB embeddings to handle complex
KBQA and unseen relations.
Note that since the API-based CRAG dataset does not support
SPARQL querying, so we ran them only on SPARQL-based datasets.
A.4 Parameter Sensitivity
In this section, we analyze the sensitivity of an important hyper-
parameter: the retrieval balancing parameter 𝜆. Specifically, we
experimented with CacheRAG by setting 𝜆to 0, 0.25, 0.5, 0.75,
and 1, and we present the corresponding accuracy, hallucination
rate, miss rate, and truthfulness score in Figure 6. As shown in
Figure 6, CacheRAG achieves relatively stable performance when
𝜆∈[ 0.25,1](up to0.7%change in terms of accuracy and up to1 .5%
change in terms of truthfulness). The model achieves the highest
accuracy when 𝜆=0.5, while it achieves the lowest hallucination
rate when𝜆=0.75. In general, we believe selecting 𝜆=0.5could
be an intuitive and effective choice that balances the diversity and
relevance of learning samples for the LLM planner.
A.5 Scalability Details
We analyze the scalability of CacheRAG step by step, utilizing the
formal architectural bounds established in our methodology:
1)Diversity-Aware Cache: The caching mechanism’s efficiency
is decoupled from the KG scale. Navigating the two-layer hierarchi-
cal index (Domain→Aspect) requiresO(1)dictionary lookup time.
Evaluating structural diversity via MMR within the localized bucket
takesO(𝑏log𝑏) time, where 𝑏is the bounded bucket capacity. The
global space complexity is O(𝑁) , where𝑁is the total number of
cached historical plans.
2)Bounded Depth Expansion ( 𝜎𝑑𝑒𝑝𝑡ℎ): As an Index Nested
Loop Join, the queries generated in this step are point queries:
SELECT ?value WHERE {
:specific_entity :specific_attribute ?value .
}
Based on the SPO B+ Tree indices used by KGs like Wikidata [ 1,8],
a single index lookup induces O(log𝑛) time complexity, where 𝑛
is the total number of KG triples. Since this operator is strictly
bounded to trigger at most 𝐾𝑑𝑒𝑝𝑡ℎ times, the total time complex-
ity isO(𝐾𝑑𝑒𝑝𝑡ℎ log𝑛) . Retaining the intermediate trajectory states
requiresO(𝐾 𝑑𝑒𝑝𝑡ℎ)memory.3)Bounded Breadth Expansion ( 𝜎𝑏𝑟𝑒𝑎𝑑𝑡ℎ ): This operator exe-
cutes bounded Star-Pattern Neighborhood Scans:
SELECT ?property ?value WHERE {
:specific_entity ?property ?value .
}
Fetching the subject via the B+ Tree takes O(log𝑛) time, and se-
quentially scanning to retrieve up to 𝐾𝑑𝑒𝑔𝑟𝑒𝑒 top-ranked neighbors
takesO(𝐾𝑑𝑒𝑔𝑟𝑒𝑒)time, yieldingO(𝐾𝑑𝑒𝑔𝑟𝑒𝑒+log𝑛) per entity. Ap-
plied across the frontier nodes (bounded by 𝐾𝑑𝑒𝑝𝑡ℎ ), the total time
complexity isO(𝐾𝑑𝑒𝑝𝑡ℎ·(𝐾𝑑𝑒𝑔𝑟𝑒𝑒+log𝑛)) . Retaining this local
topology strictly requiresO(𝐾 𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒)memory.
4)LLM Execution Overhead: The number of API invocations
to the LLM is deterministic: 1 call for theLogical Parser, 1 for the
Backend Adapter, up to 𝐾𝑑𝑒𝑝𝑡ℎ loop evaluations for theHeuristic
Dispatcher, and 1 final call for theSummarizer. Therefore, the
total number of LLM inferences is bounded by O(𝐾𝑑𝑒𝑝𝑡ℎ). Assuming
a maximum LLM inference latency 𝐿, the time complexity induced
by the LLM isO(𝐿·𝐾𝑑𝑒𝑝𝑡ℎ). Storing the intermediate logical contexts
takesO(𝐾 𝑑𝑒𝑝𝑡ℎ)memory.
Overall Complexity: 𝐾𝑑𝑒𝑝𝑡ℎ and𝐾𝑑𝑒𝑔𝑟𝑒𝑒 are administrative
system bounds designed to prevent combinatorial explosion (e.g.,
𝐾𝑑𝑒𝑝𝑡ℎ≤3,𝐾𝑑𝑒𝑔𝑟𝑒𝑒≤30). The bucket size 𝑏and LLM latency 𝐿
are similarly treated as system constants. Therefore, the overall on-
line time complexity evaluates to O(𝑏log𝑏+𝐾 𝑑𝑒𝑝𝑡ℎ log𝑛+𝐾𝑑𝑒𝑝𝑡ℎ·
𝐾𝑑𝑒𝑔𝑟𝑒𝑒+𝐿·𝐾𝑑𝑒𝑝𝑡ℎ)=O(log𝑛+𝐿) , and the space complexity eval-
uates toO(𝑁+𝐾𝑑𝑒𝑝𝑡ℎ+𝐾𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒)=O(𝑁+𝐾 𝑑𝑒𝑝𝑡ℎ·𝐾𝑑𝑒𝑔𝑟𝑒𝑒).
The critical finding is that the online execution time grows strictly
logarithmically (O(log𝑛) ) with respect to the underlying knowl-
edge graph size, demonstrating excellent production scalability.
A.6 Head Torso Tail Analysis
To comprehensively study the performance of CacheRAG in head,
torso, and tail data splits, we evaluate the performance of CacheRAG
across the three CRAG data splits and compare it with the base
LLM in Table 9. Specifically, the base LLMs GPT-4o, Llama-3.1-70B-
Instruct, and Deepseek-chat-V3-0324 show a significant decreasing
trend in terms of accuracy and truthfulness scores as we move from
head to tail data splits (up to 14%, 17%, and 17% for the three models,
respectively). Our CacheRAG model demonstrates relatively stable
performance across head, torso, and tail (with a 4.2% decrease in
accuracy and a 3.6% decrease in truthfulness score), which illustrates
the robustness of our method concerning the varying popularity of
the questions.

CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 9: Head/torso/tail experimental results on CRAG.
Model A H M T
head GPT-4o 0.4010.0860.513 0.316
Llama 0.390 0.096 0.513 0.294
Deepseek 0.487 0.134 0.380 0.353
CacheRAG0.8400.1230.037 0.717
torso GPT-4o 0.3550.0890.557 0.266
Llama 0.3050.0890.606 0.217
Deepseek 0.394 0.158 0.448 0.236
CacheRAG0.8330.0980.069 0.735
tail GPT-4o 0.266 0.096 0.638 0.170
Llama 0.2230.0530.723 0.170
Deepseek 0.319 0.128 0.553 0.191
CacheRAG0.7980.1170.085 0.681
A.7 Multi-KG Routing
As mentioned in Section 5.1, the CRAG dataset contains five KGs
from five different domains. If the LLM planner is to correctly apply
the API functions, it needs to be routed to the correct KG. Therefore,
we would like to analyze CacheRAG’s performance on multi-KG
domain routing. Specifically, we compare the performance of the
following domain routing designs:
•a) Naive Llama-3.1-8B-Instruct model:Directly prompt the LLM
with the question and KG names to know which domain’s KG
can solve the user’s question.
•b) Llama-3.1-8B-Instruct model + static KG descriptions:Prompt
the LLM with the question and static brief KG descriptions (The
“Initial KG descriptions” column of Table 10) to route the user
questions.
•c) Llama-3.1-8B-Instruct model + static KG descriptions + Chain-
of-Thoughts reasoning:Prompt the LLM with the question and
static brief KG descriptions in a Chain-of-Thoughts manner to
route the user questions.
•d) Our design (Llama-3.1-8B-Instruct model + dynamic KG de-
scriptions + Chain-of-Thought reasoning):Based on design c), we
prompt the LLM to maintain a dynamic KG description cache.
Specifically, if the domain is incorrectly identified, we prompt
the LLM to update the KG descriptions so that the LLM-based
domain router can better understand the KGs by referencing the
KG descriptions generated by itself.
To further explain how the dynamic KG description works, we
present the Multi-KG Domain Description Update template.
Multi-KG Domain Description Update Skeleton
You will be given set of KGs with descrptions, your task is to refine the
descriptions of the KGs.
These are the current descriptions of the KGs:
[𝐾𝑖]: [𝐾 𝑖’s descriptions]
...
In the previous round, you were asked to determine which external data
source is most suitable for answering this question. [Task descriptions].Based on the KG descriptions, you thought the most appropriate data
source was [Previous Domain Selection], for the question [𝑄 𝑖].
Your reasoning process was [Previous Reasoning Steps].
Your selection was wrong.
Based on this answering history, please update the KG descriptions to
better answer the question in the next round. Please focus on identi-
fying the differences between different KGs, try to avoid overlapping
descriptions for different KGs!
[Further Instructions] (Optional)
We also present the initial KG descriptions and the updated final
KG descriptions in Table 10. As shown in Table 10, the initial KG
descriptions are enriched by the LLM router to be more precise and
accurate in understanding the content, coverage, and functionality
of each respective KG. We present the quantitative experimental
results in Figure 7. In general, our design outperforms other multi-
KG domain routing designs by8%, demonstrating the effectiveness
of CacheRAG’s dynamic KG description cache.
a) b) c) d)
Multi-KG domain routing designs0.00.20.40.60.81.0accuracy0.6450.7380.8530.933
Figure 7: The multi-KG domain routing accuracy of different
designs. Our Llama-3.1-8B-Instruct + dynamic KB descrip-
tions + Chain-of-Thoughts reasoning design outperforms the
rest of the domain routing design by8%.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yushi Sun and Lei Chen
Table 10: Exemplar initial KG descriptions and the updated final KG descriptions.
Domain Initial KG descriptions Final KG descriptions
Open This KG includes content in Open domain. The content is based
on Wikidata, you can use it as a general encyclopedia.This KG includes general knowledge about the world, excluding
specific domains like film, finance, and sports, but including
information about history, science, culture, and general infor-
mation about people, organizations, and events that are not
related to any specific domain.
Movie This KG includes content in Movie domain. The content is based
on IMDB, you can find the detailed information of the actors,
movies, and oscar awards.This KG is focused on the film industry, including information
about movie titles, film scripts, and film-related events, but ex-
cluding information about music releases, financial transactions,
general knowledge about the world, and sports.
Music This KG includes content in Music domain. The content is based
on musicBuzz and Billboard, you can find the detailed informa-
tion of the singers, albums, songs, and billboard results.This KG is focused on the music industry, including informa-
tion about musicians, music genres, album releases, and music-
related events, but excluding information about film releases,
financial transactions, general knowledge about the world, and
sports.
Finance This KG includes content in Finance domain. The content is
based on Yahoo finance, you can find the detailed information
of the stock prices, eps, p/e ratio, etc.This KG is focused on financial transactions and market trends,
including economic indicators, stock market data, and financial
news, with a focus on specific financial data, such as stock
prices, financial reports, and market analysis.
Sports This KG includes content in Sports domain. The content is based
on basketball and soccer, you can find the detailed information
of the NBA and Premier League match results and team leaders.This KG is focused on the competitive aspects of sports, in-
cluding information about teams, players, championships, and
tournaments, as well as the rules, strategies, and techniques of
specific sports.