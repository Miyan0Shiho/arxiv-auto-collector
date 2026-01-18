# Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs

**Authors**: Manideep Reddy Chinthareddy

**Published**: 2026-01-13 18:03:41

**PDF URL**: [https://arxiv.org/pdf/2601.08773v1](https://arxiv.org/pdf/2601.08773v1)

## Abstract
Retrieval-Augmented Generation for software engineering often relies on vector similarity search, which captures topical similarity but can fail on multi-hop architectural reasoning such as controller to service to repository chains, interface-driven wiring, and inheritance. This paper benchmarks three retrieval pipelines on Java codebases (Shopizer, with additional runs on ThingsBoard and OpenMRS Core): (A) vector-only No-Graph RAG, (B) an LLM-generated knowledge graph RAG (LLM-KB), and (C) a deterministic AST-derived knowledge graph RAG (DKB) built with Tree-sitter and bidirectional traversal.
  Using 15 architecture and code-tracing queries per repository, we measure indexing time, query latency, corpus coverage, cost, and answer correctness. DKB builds its graph in seconds, while LLM-KB requires much longer graph generation. LLM-KB also shows indexing incompleteness: on Shopizer, 377 files are skipped or missed, reducing embedded chunk coverage and graph size compared to DKB. End-to-end cost is modest for DKB relative to the vector-only baseline but much higher for LLM-KB, especially as repository scale increases. Query latency is similar for No-Graph and DKB, while LLM-KB is slower and more variable. On the Shopizer question suite, DKB achieves the highest correctness, LLM-KB is close behind, and the vector-only baseline performs worst on upstream architectural queries and has the highest hallucination risk. Overall, deterministic AST-derived graphs provide more reliable coverage and multi-hop grounding than LLM-extracted graphs at substantially lower indexing cost.

## Full Text


<!-- PDF content starts -->

Reliable Graph-RAG for Codebases: AST-Derived
Graphs vs LLM-Extracted Knowledge Graphs
Manideep Reddy Chinthareddy
Software Engineer, Centerville, USA
chmanideepreddy@gmail.com
January 2026
Abstract
Retrieval-Augmented Generation (RAG) for software engineering
often relies on vector similarity search, which captures topical simi-
larity but frequently fails on multi-hop architectural reasoning (e.g.,
controller→service→repository chains, interface-driven wiring, and
inheritance-based behavior) [2, 5]. This paper benchmarks three re-
trieval pipelines on Java codebases (detailed on Shopizer; additional
runs on ThingsBoard and OpenMRS Core): (A) No-Graph Naive RAG
(vector-only), (B) an LLM-Generated Knowledge Graph RAG (LLM-
KB), and (C) a deterministic AST-derived Knowledge Graph RAG
(DKB) built via Tree-sitter parsing with bidirectional traversal [20].
Using a fixed suite of 15 architecture and code-tracing queries per
repository (The questions are same for all the 3 approaches while they
differ per repository to make it relevant to the code bases), we re-
port indexing overhead, query-time latency, corpus coverage signals,
and end-to-end cost. Across repositories, DKB builds its ontology
graph in seconds (2.81s on Shopizer; 13.77s on ThingsBoard; 5.60s on
OpenMRS Core), whereas LLM-KB requires substantially longer LLM-
mediated graph generation (200.14s on Shopizer; 883.74s on Things-
Board; 222.17s on OpenMRS Core). Critically, LLM-KB exhibits
probabilisticindexing incompleteness: on Shopizer, the extraction log
flags 377 files asSKIPPED/MISSED by LLM, yielding a per-file success
rate of 0.688 (833/1210) and shrinking the embedded corpus to 3465
chunks (0.641 coverage vs. the No-Graph baseline of 5403 chunks),
compared to deterministic indexing at 4873 chunks (0.902 coverage).
This incompleteness also reduces the extracted graph footprint: LLM-
KB produces 842 nodes versus 1158 nodes for DKB (0.727 node cov-
erage), even though all approaches scan the same discovered files;
however, LLM-KB fails to produce structured records for a subset,
1arXiv:2601.08773v1  [cs.SE]  13 Jan 2026

and those files consequently do not contribute to embeddings/graph.
While batching strategies, prompt tuning, and retries can reduce omis-
sions, completeness remains mediated by stochastic model behavior
and schema-bound extraction failure modes in tool-mediated indexing
pipelines [21,27].
We also quantify end-to-end execution cost (indexing + answering
the full 15-question suite) and report relative cost normalized to the
No-Graph baseline. On Shopizer, representative costs were$0.04 (No-
Graph),$0.09 (DKB), and$0.79 (LLM-KB), corresponding to∼2.25×
for DKB and∼19.75×for LLM-KB. On a larger combined OpenMRS-
core + ThingsBoard workload, costs increase to$0.149 (No-Graph),
$0.317 (DKB), and$6.80 (LLM-KB), yielding∼2.13×for DKB and
∼45.64×for LLM-KB, indicating that LLM-mediated graph construc-
tion can dominate total cost as repository scale increases.
For query-time latency, the No-Graph baseline remains competitive
but less stable across workloads: on Shopizer it achieves 9.52±2.98s
(mean±std), compared to 10.51±4.17s for DKB and 13.36±7.87s for
LLM-KB. On ThingsBoard, mean latencies increase to 10.92±1.43s
(No-Graph), 11.17±1.97s (DKB), and 15.29±4.94s (LLM-KB), while
on OpenMRS Core we observe 11.82±2.79s (No-Graph), 12.21±6.96s
(DKB), and 11.94±4.88s (LLM-KB), with LLM-KB and DKB showing
higher worst-case outliers in several settings. We additionally validate
answer correctness on the full 15-question Shopizer suite: DKB at-
tains the highest correctness (15/15), LLM-KB follows closely (13/15;
2 partial), while No-Graph degrades on upstream architectural queries
and exhibits the highest hallucination risk (6/15). Across repositories,
correctness gains are largest on suites that emphasize multi-hop archi-
tectural tracing and upstream discovery; in some suites (e.g., Things-
Board) DKB ties the vector-only baseline while maintaining higher
coverage and lower indexing overhead than LLM-mediated graph con-
struction.
Keywords:Retrieval-Augmented Generation (RAG), Static Code Analy-
sis, Abstract Syntax Trees (AST), Knowledge Graphs, Multi-Hop Reason-
ing, Software Maintenance.
2

Nomenclature
G= (V, E) Code knowledge graph with nodesVand edgesE
qNatural language query
R(·) Vector retriever returning top-kitems
Nd(v) Graph neighborhood within depthdaround nodev
C(q) Context assembled for queryq
QNumber of benchmark questions (hereQ= 15)
CNumber of embedded text chunks
Nfiles Number of Java files discovered during scanning
1 Introduction
As Large Language Models (LLMs) transition from generic assistants to
enterprise reasoning engines built on Transformer architectures [1], the con-
text window remains a primary constraint. Retrieval-Augmented Genera-
tion (RAG) mitigates this constraint by retrieving relevant context from a
codebase prior to answer generation [2,5]. In software engineering settings,
however, vector similarity often introducescontext flattening: the retrieved
chunks share lexical or semantic overlap with the query, but do not reliably
preserve structural dependencies such as inheritance, dependency injection,
and call relationships.
In complex systems, many questions are inherently multi-hop. For ex-
ample, answering “Which controllers use the shopping cart logic?“ requires
traversing upstream consumers from services to controllers, and frequently
crossing interface boundaries. Vector-only retrieval may retrieve the shopping-
cart implementation class but omit the controllers that depend on it, pro-
ducing incomplete or ungrounded answers. This failure mode is consistent
with known limitations of dense retrieval pipelines (e.g., DPR-style retriev-
ers) when evidence is distributed across multiple non-local contexts [3,4].
This paper compares three retrieval paradigms for code analysis: (A) No-
Graph Naive RAG (vector-only), (B) LLM-KB Graph RAG (LLM-generated
dependency graph during indexing), and (C) DKB, a deterministic compiler-
inspired approach that parses code via ASTs and performs bidirectional
graph expansion at query time [9,20]. Our motivation aligns with repository-
level code understanding needs highlighted by recent benchmark suites and
surveys [10,11,14,15].
3

1.1 Contributions
This paper makes the following contributions:
1.Benchmarking framework for graph-aware retrieval:We pro-
vide an end-to-end comparison of vector-only, LLM-extracted graph
RAG, and AST-derived graph RAG under shared hyperparameters
and identical question sets [5,19].
2.Measured indexing and query-time costs:We report concrete
build times and latency distributions; notably, LLM-based graph con-
struction introduces large indexing overhead compared to determinis-
tic AST parsing [6,20].
3.Indexing reliability analysis:We instrument and reportcoverage/-
consistencysignals from run logs (files scanned, nodes/edges built,
chunks embedded), and show that LLM-KB can skip files during ex-
traction, shrinking both graph size and embedded corpus; this connects
to broader reliability discussions in agentic RAG and tool-mediated
pipelines [21].
4.Bidirectional + interface-aware expansion:We show how pre-
decessor traversal and interface-consumer expansion improve retrieval
for architecture discovery queries that often fail under successors-only
or text-only retrieval [6,19].
5.Correctness validation:We provide a correctness comparison on
representative questions spanning repository-level reasoning and multi-
hop architectural discovery [14–16].
1.2 Paper Organization
Section II reviews background on RAG and program graphs. Section III
formalizes the problem and research questions. Section IV details the three
pipelines and the graph-aware retrieval algorithm. Section V describes the
experimental setup and metrics. Section VI reports quantitative and qual-
itative results. Sections VII–X discuss implications, threats to validity, re-
producibility, and limitations before concluding.
4

2 Background and Preliminaries
2.1 RAG for Software Engineering
In code intelligence, a common baseline is chunk-based embedding retrieval:
source code is split into overlapping segments, embedded, and retrieved by
similarity to a query [2, 5]. This approach is simple and fast, but can fail
to include structurally necessary context when the required evidence resides
in related files (e.g., an interface definition, a parent class, or a controller).
Recent work on lightweight retrieval stacks emphasizes simplicity and speed,
but does not eliminate the need for structured multi-hop grounding in code-
bases [8].
2.2 Program Structure Graphs
Code structure can be represented as graphs: nodes correspond to entities
such as classes, interfaces, methods, and files; edges represent relations such
asextends,implements,injects/uses, andcalls. Static parsing (e.g., AST
analysis) can deterministically recover many such relations for statically-
typed languages like Java [20]. AST-derived structure has also been used
to improve code chunking for retrieval by aligning chunk boundaries with
syntactic structure rather than raw character windows [9].
2.3 Knowledge Graphs and Ontologies in Code
In this work, “ontology” refers to a typed schema for code entities and
relations (node types + edge labels) suitable for retrieval and traversal.
Graph RAG leverages this structure to expand retrieved context beyond
the initial top-kchunks [6,7,19].
3 Related Work
Retrieval-augmented generation.RAG was formalized as a paradigm
for grounding sequence generation in retrieved evidence, combining paramet-
ric and non-parametric memory to improve factuality and reduce hallucina-
tions [2]. Dense retrievers such as DPR established effective neural retrieval
for open-domain QA, and subsequent retrieval-augmented language models
(e.g., ATLAS) showed strong few-shot and knowledge-intensive performance
by tightly coupling retrieval with generation [3, 4]. Recent surveys consoli-
date best practices in retrieval pipelines, chunking, indexing, and evaluation,
5

while noting that retrieval adequacy remains workload-dependent and can
degrade under multi-hop evidence requirements [5].
GraphRAG and knowledge-graph enhanced RAG.GraphRAG-
style systems introduce explicit graph structure to move beyond purely local
similarity matches. Edge et al. propose a “local-to-global” GraphRAG ap-
proach for query-focused summarization, where initial retrieval is expanded
through graph neighborhoods to collect globally relevant supporting con-
text [6]. Domain-focused variants, such as Document GraphRAG in manu-
facturing QA, demonstrate that knowledge-graph signals can improve doc-
ument QA by connecting entities and relations across otherwise disjoint
textual fragments [7]. A recent survey of graph retrieval-augmented gen-
eration discusses a design space spanning graph construction (manual vs.
extracted), graph granularity (document vs. entity), and graph-guided re-
trieval policies [19]. Our work differs by targetingcodegraphs where struc-
tural relations are not merely semantic co-occurrence but compiler-visible
dependencies.
RAG for code and repository-level evaluation.LLMs trained on
code enabled strong program synthesis and code completion behavior, and
evaluation frameworks for code LLMs highlight the importance of realis-
tic, repository-grounded tasks [12, 13]. Repository-level benchmarks such
as SWE-bench and RepoBench emphasize multi-file reasoning, tool use,
and integration with real code artifacts rather than single-function prob-
lems [14, 15]. RepoEval (introduced in repository-level completion work)
similarly focuses on evaluating completion and generation in the presence of
repository context [16]. Surveys specifically focused on retrieval-augmented
code generation summarize repository-level approaches and emphasize that
retrieval quality, chunking strategy, and grounding across file boundaries are
often the limiting factors [10,11].
Structure-aware chunking and code graphs.Structure-aware chunk-
ing using ASTs (e.g., cAST/CAST) improves retrieval by preventing seman-
tically coupled code from being split across chunks and by avoiding mal-
formed fragments that degrade embedding utility [9]. Graph-based retrieval
methods for repository-level code generation, such as RepoGraph, explicitly
leverage repository structure to retrieve and compose evidence across files,
aligning with the core motivation of this paper [17]. Beyond retrieval, graph-
integrated models (e.g., CGM) represent a complementary line of work that
integrates graph signals into model architectures for repository-level soft-
ware engineering tasks [18].
Determinism, robustness, and indexing reliability.While many
GraphRAG systems assume a stable graph artifact, LLM-mediated graph
6

extraction can introduce stochasticity and schema-compliance failures, and
these effects are amplified in agentic RAG settings where multiple tool calls
and intermediate steps can compound variance [21]. In practice, this may
manifest as missing entities/edges or skipped inputs during indexing, pro-
ducing blind spots at query time; our contribution is to measure this effect
directly via run-log coverage signals in a codebase setting. More broadly,
when the downstream task is correctness-sensitive (e.g., vulnerability detec-
tion), systematic benchmarks have found meaningful gaps between LLM-
driven approaches and traditional static analysis under controlled evalua-
tion, reinforcing the value of deterministic signals where available [22].
4 Problem Formulation and Research Questions
4.1 Task Definition
LetSbe a codebase, and letqbe a natural-language query about code be-
havior or architecture. A RAG system assembles contextC(q) and produces
an answera. We study retrieval strategies that differ in whether and how
they represent code structure [2,5].
4.2 Graph-Based Retrieval Formalization
We model code structure as a directed labeled graphG= (V, E), where
nodes represent code entities (classes, interfaces, etc.) and edges represent
relations (e.g.,extends,implements,injects) [19,20].
A graph-aware retriever expands an initially retrieved node setV 0using
neighborhood expansion:
Vd=V 0∪[
v∈V 0Nd(v),(1)
whereN d(v) denotes the set of nodes within hop depthdaround nodev
under selected edge directions (successors, predecessors, or both) [6].
The final assembled context can be expressed as:
C(q) = Concat ({code(v)|v∈V d}).(2)
4.3 Research Questions
•RQ1 (Efficiency):What are the indexing time and query latency
trade-offs across vector-only, LLM-KB, and DKB? [5]
7

•RQ2 (Retrieval adequacy):Does deterministic bidirectional ex-
pansion retrieve evidence that vector-only retrieval misses on multi-
hop questions? [6,19]
•RQ3 (Stability):How variable is each approach in query-time la-
tency, and does LLM-KB introduce higher variance or worst-case out-
liers due to multi-step/tool-mediated processing? [21]
•RQ4 (Correctness):Which approach produces the most accurate
answers across representative code comprehension and architecture
discovery tasks? [14,15]
•RQ5 (Indexing reliability):How often does each approach preserve
corpus coverage (files, entities, chunks), and does LLM-KB exhibit
extraction-time skipping that reduces downstream retrievability?
5 System Overview
5.1 Compared Pipelines
We compare three pipelines:
1.Naive RAG (No-Graph):top-kvector similarity retrieval over code
chunks [2,3].
2.LLM-KB:LLM-generated dependency graph at indexing time, plus
graph-informed context expansion at query time [6,19].
3.DKB (Ours):deterministic AST-derived graph built with Tree-
sitter, used for bidirectional traversal and interface-aware expansion
at query time [9,20].
5.2 Graph Artifacts
Figure 1 shows the two graph artifacts produced for the same codebase.
6 Methodology
6.1 Strategy A: Naive Vector Retrieval (No-Graph)
We implement a standard RAG baseline that relies exclusively on vector
similarity search [2, 3]. Source files are chunked using a recursive character
8

(a) Deterministic AST-derived ontology
graph (DKB).
(b) LLM-extracted dependency graph
(LLM-KB).
Figure 1: Graph artifacts generated for the same Java codebase (Shopizer).
DKB:nodes are project-local Java types (classes/interfaces/enums/record-
s/annotations) discovered via Tree-sitter; edges are typedinjects,
extends, andimplementsrelations.LLM-KB:nodes are LLM-emitted
class nameentities; edges aredepends onrelations extracted from per-file
structured outputs after filtering out standard-library and framework depen-
dencies (e.g.,java.*, Spring packages). (This caption replaces the earlier
placeholder request to specify node/edge definitions and filtering.)
splitter (chunk size=1000,chunk overlap=100), embedded, and indexed
in a local vector store. At query time, the system retrieves the top-k= 10
chunks by similarity and passes them to an LLM for answer generation.
6.2 Strategy B: LLM-Generated Knowledge Graph (LLM-
KB)
The LLM-KB approach constructs a dependency graph by prompting an
LLM during indexing, aligned with graph-based retrieval-augmentation pat-
terns in the broader GraphRAG literature [6, 19]. For each Java file, the
LLM emits a structured JSON object containing entities and dependencies.
A directed graph is then created and used to expand context at query time
by appending neighboring nodes and their code content.
Practical constraint.To fit batch prompts in context limits, the im-
plementation truncates file content when building per-batch inputs. This
9

can contribute to partial extraction or omission of a file from the struc-
tured output, which the run logs surface as “SKIPPED/MISSED by LLM”.
While stricter prompting, smaller batch sizes, or retries can reduce the skip
rate, complete coverage is not guaranteed in multi-step/agentic retrieval
pipelines where intermediate tool outputs must satisfy schemas and control-
flow checks [21].
6.3 Strategy C: Deterministic AST Graph (DKB - Proposed)
DKB deterministically constructs a code graph using Tree-sitter (tree sitter java) [20].
The pipeline adds labeled edges for inheritance (extends/implements) and
dependency injection patterns (field and constructor parameter types). At
query time, DKB expands context usingbidirectionalgraph traversal (suc-
cessors and predecessors), and performsinterface-consumer expansion
to improve upstream discovery. This design is complementary to structure-
aware chunking approaches (e.g., AST-based chunking) that improve re-
trieval signal quality by aligning chunks with syntactic units [9].
6.4 Graph-Aware Retrieval Algorithm
Algorithm 1Graph-aware context assembly (bidirectional)
Require:queryq, vector retrieverR, graphG, depthd, top-k
1:D←R(q)▷Top-kretrieved chunks
2:V 0←EntitiesFromDocs(D)
3:V←V 0
4:forv∈V 0do
5:V←V∪Succ(G, v, d)∪Pred(G, v, d)
6:V←V∪InterfaceConsumerExpand(G, v)▷Optional
7:end for
8:returnC(q)←AssembleCodeContext(V)
6.5 Implementation Snippets and Instrumentation
This section provides small excerpts from the evaluation scripts to make the
indexing and retrieval behaviors concrete and reproducible.
6.5.1 DKB: Deterministic ontology extraction (Tree-sitter)
Listing 1 shows the Tree-sitter query patterns and the two-pass extraction
strategy used to build an AST-derived graph with typed edges (injects,
10

extends,implements) [20].
1# Tree - sitter queries for type discovery + DI signals
2class_query = Query ( JAVA_LANGUAGE , """
3( class_declaration name : ( identifier ) @class_name )
4( interface_declaration name : ( identifier ) @class_name )
5( enum_declaration name : ( identifier ) @class_name )
6( record_declaration name : ( identifier ) @class_name )
7( annotation_type_declaration name : ( identifier ) @class_name )
8""")
9
10injection_query = Query ( JAVA_LANGUAGE , """
11( field_declaration type : ( type_identifier ) @type_name )
12""")
13
14constructor_query = Query ( JAVA_LANGUAGE , """
15( constructor_declaration parameters : ( formal_parameters
16( formal_parameter type : ( type_identifier ) @type_name )))
17""")
18
19def build_spring_graph_treesitter ( root_path : str):
20G = nx. DiGraph ()
21file_map = {}
22
23# Pass 1: discover project types -> file map
24for root , _, files in os. walk ( root_path ):
25for file in files :
26if not file . endswith (". java "): continue
27full_path = os. path . join (root , file )
28content = open ( full_path , "rb"). read ()
29tree = parser . parse ( content )
30captures = class_cursor . captures ( tree . root_node )
31for capture_name , nodes in captures . items ():
32for node in nodes :
33if capture_name == " class_name ":
34class_name = get_node_text (node ,
content )
35file_map [ class_name ] = full_path
36G. add_node ( class_name , path = full_path )
37
38# Pass 2: add edges : injects / extends / implements
39for class_name , file_path in file_map . items ():
40content = open ( file_path , "rb"). read ()
41tree = parser . parse ( content )
42
43# Field + constructor injections
44for _, nodes in injection_cursor . captures ( tree .
root_node ). items ():
45for node in nodes :
11

46dep = get_node_text (node , content )
47if dep in file_map : G. add_edge ( class_name , dep ,
relation =" injects ")
48
49for _, nodes in constructor_cursor . captures ( tree .
root_node ). items ():
50for node in nodes :
51dep = get_node_text (node , content )
52if dep in file_map : G. add_edge ( class_name , dep ,
relation =" injects ")
53
54# Inheritance edges ( extends / implements ) extracted by
field names
55# ... ( omitted for brevity ; see full script )
56return G, file_map
Listing 1: DKB ontology construction via Tree-sitter queries and two-pass
extraction.
6.5.2 DKB: Bidirectional traversal + interface-consumer expan-
sion
Listing 2 shows thebidirectionalgraph expansion used at query time, includ-
ing the key interface-consumer fix that improves controller discovery across
interface boundaries [19].
1def retrieve_with_graph_context ( query : str) -> str:
2docs = retriever . invoke ( query )
3context_text = ""
4processed_classes = set ()
5
6for doc in docs :
7source_path = doc. metadata .get(" source ", "")
8class_name = os. path . basename ( source_path ). replace (".
java ", "")
9
10if class_name in processed_classes : continue
11processed_classes .add( class_name )
12
13# Bidirectional expansion
14if class_name in graph :
15successors = list ( graph . successors ( class_name ))
# downstream deps
16predecessors = list ( graph . predecessors ( class_name ))
# upstream users
17
18# INTERFACE EXPANSION :
12

19# if class implements an interface , also add
consumers of that interface
20for successor in successors :
21edge_data = graph . get_edge_data ( class_name ,
successor )
22if edge_data and edge_data .get(" relation ") == "
implements ":
23interface_users = list ( graph . predecessors (
successor ))
24predecessors . extend ( interface_users )
25context_text += (
26f"\n[ ONTOLOGY INFO ]: { class_name }
implements { successor }. "
27f" Checking consumers of { successor }...\ n"
28)
29
30# Emit relationship summary and append code of
neighbors ( budgeted )
31context_text += f"\n[ ONTOLOGY INFO ]: Relationships
for { class_name }:\ n"
32for dep in successors :
33rel = graph . get_edge_data ( class_name , dep).get(
" relation ", " uses ")
34context_text += f" - [ INJECTS / USES ] -> {dep}
({ rel })\n"
35
36for consumer in predecessors :
37rel = ( graph . get_edge_data ( consumer , class_name
) or {}).get(
38" relation ", " uses (via interface )"
39)
40context_text += f" - [ USED BY] <- { consumer }
({ rel })\n"
41
42return context_text
Listing 2: DKB query-time context assembly with successors, predecessors,
and interface-consumer expansion.
6.5.3 LLM-KB: Batch extraction, truncation, and explicit skip
detection
Listing 3 highlights the probabilistic indexing behavior: file contents are
truncated for batching, and the pipeline prints explicitSKIPPED/MISSED by
LLMlines when a file in the input batch does not appear in the model’s struc-
tured output; this kind of schema-bound omission is a common reliability
concern in agentic/tool-mediated RAG pipelines, and more broadly reflects
13

known limitations and trade-offs in enforcing strict structured outputs in
practice [21,25,26].
1# Batch preparation : truncate each file to fit prompt budget
2for path in batch_files :
3content = open (path , "r", encoding ="utf -8"). read ()
4snippet = content [:15000] # truncation to save context
5batch_input_str += f"\n--- FILE : { path } ---\n{ snippet }\n"
6
7# Invoke LLM for the whole batch ( structured JSON )
8result = extraction_chain . invoke ({" batch_content ":
batch_input_str })
9analyzed_files = result .get(" results ", [])
10
11# Track which input paths appeared in structured output
12processed_paths_in_batch = set ()
13for res in analyzed_files :
14c_path = (res.get(" file_path ") if isinstance (res , dict )
else res. file_path )
15if c_path :
16processed_paths_in_batch .add(os. path . normpath ( c_path ))
17
18# Emit explicit skip signals if an input file is missing from
output
19for original_path in input_batch :
20norm_path = os. path . normpath ( original_path )
21if norm_path not in processed_paths_in_batch :
22# fallback : basename match (LLM may return relative
paths )
23base_name = os. path . basename ( norm_path )
24found = any(os. path . basename (p) == base_name for p in
processed_paths_in_batch )
25if not found :
26print (f" SKIPPED / MISSED by LLM: {os. path . basename (
norm_path )}")
Listing 3: LLM-KB batch analysis with file truncation and explicit skipped-
file detection.
7 Experimental Setup
7.1 Dataset
We evaluate generalization across three Java codebases with distinct ar-
chitectures and domains: (i)Shopizer(Spring-based e-commerce), (ii)
ThingsBoard(IoT platform with transport + actor/rule-engine subsys-
tems), and (iii)OpenMRS Core(clinical EMR platform with web filters,
14

service layer, and modular runtime). For each repository, we run the same
15-question suite under No-Graph RAG, LLM-KB (LLM-built graph), and
DKB (Tree-sitter graph).
7.2 Task Set
We use a fixed suite ofQ= 15 architecture and code-tracing questions for
each of the 3 approaches. Note that the question suite is different for dif-
ferent repositories. This is done to ensure the questions stay relevant to the
repository. This paper reports detailed correctness for the full 15-question
suite spanning repository semantics, multi-hop service tracing, event-driven
triggers, upstream discovery, and ID generation. The emphasis on multi-file,
repository-level reasoning mirrors recent evaluation directions in software
engineering benchmarks [14–16].
8 Evaluation Metrics
8.1 Latency
We report mean, standard deviation, median, and min–max of per-question
latency:
T=1
QQX
i=1Ti.(3)
8.2 Correctness Labels
We label each answer asCorrect,Partial, orIncorrectbased on whether
it matches the repository’s ground-truth behavior/structure and avoids hal-
lucinated entities. For tabular summary, we collapseCorrect+detail level
intoCorrect. This evaluation framing is consistent with repository-level code
tasks that require grounded behavior and multi-file evidence [10,11,14].
8.3 Indexing Reliability and Coverage
In addition to time, we measureindexing reliabilityusing run-log signals
indicating how much of the corpus was successfully indexed.
Chunk coverage.LetC baseline be the number of embedded chunks
produced by the vector-only pipeline over all discovered Java files, and let
15

Capproach be the number of chunks embedded under a given approach. Then:
ChunkCoverage =Capproach
Cbaseline.(4)
Graph coverage.For graph-based methods, we report graph size (|V|,
|E|) as a proxy for entity coverage. When comparing DKB vs. LLM-KB on
the same codebase, we also report:
NodeCoverage (LLM-KB vs. DKB) =|V|LLM-KB
|V|DKB.(5)
Skip indicators.LLM-KB emits explicit log lines of the form “SKIPPED/MISSED
by LLM:<File.java>” whenever a file from the input batch does not ap-
pear in the model’s structured output. We treat this as direct evidence of
probabilistic extraction incompleteness in a multi-step, tool-mediated index-
ing pipeline; this is consistent with broader findings that strict structured-
output requirements can fail or degrade under realistic schema constraints
and generation setups [21,25,26].
Per-file success rate.To emphasize extraction stochasticity, we also
report the fraction of discovered files that were successfully processed into
structured records (and therefore eligible for embedding/graph insertion).
LetN processed denote the number of files that produced an analyzed record.
We define:
FileSuccessRate =Nprocessed
Nfiles.(6)
For LLM-KB, we computeN processed =N files−N skipped , whereN skipped is
the number ofuniquefilenames flagged by the log indicatorSKIPPED/MISSED
by LLM.
9 Results
9.1 Quantitative Performance Benchmarks
Interpreting DB build time under corpus shrinkage.In Table 1,
LLM-KB shows lower vector DB build time than No-Graph on some reposi-
tories. This should not be interpreted as a more efficient embedding pipeline:
because LLM-KB can omit files during structured extraction, it often em-
beds a smaller corpus (fewer chunks) than the No-Graph baseline (Tables 5,
6, 7), which directly reduces DB construction work. Accordingly, we treat
total indexing time(DB + graph) together with coverage signals as the
relevant efficiency comparison.
16

Table 1: Indexing time breakdown.
Repository Approach DB time (s) Graph time (s) Total (s)
Shopizer No-Graph 18.41 0.00 18.41
Shopizer DKB (Ours) 19.28 2.81 22.09
Shopizer LLM-KB 14.95 200.14 215.09
ThingsBoard No-Graph 123.61 0.00 123.61
ThingsBoard DKB (Ours) 129.78 13.77 143.55
ThingsBoard LLM-KB 95.51 883.74 979.25
OpenMRS Core No-Graph 42.20 0.00 42.20
OpenMRS Core DKB (Ours) 42.82 5.60 48.42
OpenMRS Core LLM-KB 29.12 222.17 251.29
9.2 Cost Analysis (Normalized)
While absolute dollar costs depend on provider pricing, tokenization, model
choice, and repository scale,relativecost ratios provide a more stable com-
parison of cost drivers across retrieval strategies. We therefore reportnor-
malized end-to-end run cost(indexing + answering the full 15-question
suite), with the No-Graph baseline normalized to 1.0. In our run, DKB
incurred a modest cost multiple over No-Graph due to additional graph-
aware context assembly, whereas LLM-KB exhibited a much larger multi-
plier dominated byLLM-mediated graph generationduring indexing (Ta-
ble 3)(Table 4).
To make this more portable beyond the Shopizer-scale repository, Fig-
ure 2 visualizes only the normalized ratios (No-Graph = 1.0), rather than
treating absolute currency values as a generalizable claim. In larger enter-
prise repositories, the same qualitative pattern is expected when LLM-KB
performs extensive per-file or batched LLM extraction at index time: cost
scales primarily with the volume of content processed by the model and any
retries required for schema compliance. This scaling pressure is often am-
plified in agentic RAG settings where retrieval, validation, and refinement
may involve multiple tool calls and decision steps [21].
9.3 Cost Analysis on Multi-Repository Runs (OpenMRS +
ThingsBoard)
In addition to Shopizer, we measured end-to-end execution cost (index-
ing + answering all questions) for a combined run overtworepositories:
17

Table 2: Query-time latency over theQ= 15 question suite (from JSON
run artifacts).
Repository Approach Mean±Std (s) Median (s) Min–Max (s)
Shopizer No-Graph 9.52±2.98 9.68 2.85–14.16
Shopizer DKB (Ours) 10.51±4.17 9.95 3.47–21.16
Shopizer LLM-KB 13.36±7.87 10.96 6.53–38.25
ThingsBoard No-Graph 10.92±1.43 11.39 7.77–13.25
ThingsBoard DKB (Ours) 11.17±1.97 10.95 6.88–14.23
ThingsBoard LLM-KB 15.29±4.94 13.17 9.62–26.46
OpenMRS Core No-Graph 11.82±2.79 11.38 8.99–20.76
OpenMRS Core DKB (Ours) 12.21±6.96 10.38 9.31–37.15
OpenMRS Core LLM-KB 11.94±4.88 11.09 8.12–27.81
OpenMRS-core and ThingsBoard. Because absolute dollar costs depend on
provider pricing and tokenization, we report both absolute totals for this
combined run and normalized ratios with No-Graph set to 1.0.
Table 3: End-to-end run cost (absolute USD) across workloads (indexing +
questions).
Workload No-Graph DKB LLM-KB
Shopizer (single repo, 15 questions)$0.04$0.09$0.79
OpenMRS-core + ThingsBoard (15/repo = 30 total questions)$0.149$0.317$6.80
Table 4: End-to-end run cost normalized to No-Graph (No-Graph = 1.0 per
workload).
Workload No-Graph DKB LLM-KB
Shopizer (single repo) 1.00 2.25 19.75
OpenMRS-core + ThingsBoard 1.00 2.13 45.64
Interpretation.The multi-repo cost pattern amplifies the same qualita-
tive driver observed on Shopizer: LLM-KB’s total cost is dominated by
LLM-mediated graph construction at indexing time, whereas DKB’s incre-
mental cost over No-Graph is modest because its AST-derived graph is built
deterministically and cheaply relative to LLM extraction.
18

No-Graph DKB LLM-KB02040
1 2.2519.75
1 2.1345.64Normalized end-to-end run cost (No-Graph=1.0 per workload)Shopizer OpenMRS-core + ThingsBoard
Figure 2: Normalized end-to-end run cost across workloads. Costs are nor-
malized by the No-Graph baselinewithin each workload. LLM-KB’s cost
multiplier increases substantially on the larger workload, while DKB stays
near∼2×.
9.4 Indexing Reliability and Corpus Coverage
Table 5 summarizes corpus coverage signals taken directly from the three run
logs. All approaches discover the same number of Java files (N files= 1210),
but diverge sharply in (i) how much code is embedded into the vector store
and (ii) how complete the derived graph is.
Visual contrast (graph artifacts).Beyond the numeric coverage
counters, Fig. 1 provides an immediate qualitative signal: the determin-
istic DKB artifact appearsclustered/clean, with coherent component-level
groupings and fewer isolated fragments, whereas the LLM-KB artifact is
comparativelyfuzzyand diffuse, consistent with extraction-time omissions
19

Table 5: Code Coverage and Knowledge Extraction Completeness (Shopizer)
Metric No-Graph LLM-KB DKB
Total Java files discovered 1210 1210 1210
Files skipped/missed by LLM — 377 —
Files successfully analyzed — 833 —
File success rate — 0.688 —
Total code chunks embedded 5403 3465 4873
Chunk coverage (vs No-Graph) 1.000 0.641 0.902
Graph nodes — 842 1158
Graph edges — 2552 1503
Node coverage (vs DKB) — 0.727 1.000
(Table 5) and a less typed, schema-constrained dependency surface.
Edge-count comparability.Raw edge counts are not directly compa-
rable between DKB and LLM-KB because the two graphs encode different
relation semantics and granularity. DKB includes only a small, typed rela-
tion set (injects,extends,implements) derived from deterministic AST
signals, whereas LLM-KB emits a broaderdepends onrelation that can con-
flate usage, import-level references, and other non-inheritance dependencies.
As a result, LLM-KB can produce a denser edge set (more edges per node)
even when its node set is smaller, and edge totals should be interpreted as
schema-dependentrather than as a pure measure of structural completeness.
Observation 1 (Chunk shrinkage under LLM-KB).LLM-KB em-
beds 3465 chunks, substantially fewer than the vector-only baseline (5403)
and fewer than deterministic DKB (4873). This corresponds to0.641
chunk coveragerelative to No-Graph, versus0.902for DKB (Table 5).
Because retrieval is constrained to indexed/embedded content, this shrink-
age directly reduces the probability that evidence-bearing files can be re-
trieved at query time, especially for multi-hop questions whose supporting
evidence is distributed across many files [2,5].
Observation 2 (Probabilistic skipping + per-file success rate).
The LLM-KB extraction log prints explicit “SKIPPED/MISSED by LLM” in-
dicators, showing that the model’s batch-level structured output can omit
some input files during indexing. In the updated Shopizer run, the log flags
377skipped/missed files, leaving833files successfully analyzed out of1210
discovered, for aper-file success rate of 0.688(Table 5). This highlights
that Strategy B is stochastic at thefilegranularity (not only at the token
or edge level): when a file is omitted from the structured output, it is also
excluded from downstream embedding and graph insertion, creating system-
20

atic blind spots at query time. While prompts, batching, and retries can
reduce omissions, completeness remains mediated by stochastic model be-
havior and schema-bound extraction requirements, a known reliability risk
in tool-mediated and agentic RAG pipelines [21].
Observation 3 (Graph size reduction and node coverage).Even
after graph cleaning, LLM-KB produces a smaller graph (842nodes) than
DKB (1158nodes), yielding0.727 node coverage(LLM-KB vs. DKB)
(Table 5). This reduction is consistent with extraction-time omissions and
the resulting corpus shrinkage: because LLM-KB’s node set is derived from
the model’s per-file structured records, skipped files can remove entire project-
local entities from the graph and eliminate their edges, reducing the effec-
tiveness of graph-based neighborhood expansion [19].
Cross-repository reliability summary.Across all three repositories,
LLM-KB exhibits file-level extraction omission (File SR 0.650–0.806) that
correlates with reduced chunk coverage (0.633–0.706), while DKB remains
near baseline chunk coverage (0.902–0.993) without probabilistic file skip-
ping (Tables 5, 6, 7).
Observation 4 (Why DKB chunk coverage is below the No-
Graph baseline on Shopizer).Although DKB’s graph construction is
deterministic, its embedding corpus is coupled to the set of Java files that
successfully yield project-local type declarations and are mapped into the
ontology (file map) during AST extraction. In Shopizer, the No-Graph
baseline embeds chunks from all discovered Java files, whereas DKB’s index-
ing path can exclude or reduce contributions from files that do not produce
a parsable/mapped top-level type under the extractor (e.g., non-type util-
ity files, atypical naming or path canonicalization mismatches, generated
sources, or declarations that are not captured by the current Tree-sitter
query patterns). This explains why DKB’s chunk coverage on Shopizer
(0.902 in Table 5) is slightly below the vector-only baseline even though DKB
does not exhibit probabilistic file skipping. Importantly, this gap reflects
implementation coupling between graph node discovery and the embedding
pipeline, not model stochasticity; decoupling embeddings from graph-node
discovery (embed all discovered files, then attach graph edges where avail-
able) is a straightforward engineering improvement and would move DKB
closer to the 0.99-level coverage observed on ThingsBoard and OpenMRS
(Tables 6, 7). Future work will export a structured per-file embedding au-
dit (file→chunk count) to quantify which Shopizer files contribute to the
9.8% delta and verify that the effect is attributable to mapping/query cov-
21

erage rather than hidden filtering. We treat this as an implementation-level
coupling (and thus a fairness confound in chunk-count comparisons), not a
limitation of AST-derived graph construction itself.
9.5 Answer Correctness on Full 15-Question Suite
Table 10 summarizes correctness judgments for the full 15-question bench-
mark suite spanning repository semantics, multi-hop architectural tracing,
and system-level reasoning.
9.6 Generalization Across Repositories
To test whether the observed trade-offs persist beyond Shopizer, we repeat
the same three RAG variants on ThingsBoard and OpenMRS Core. We re-
port (i) indexing time (vector DB build and, where applicable, graph build),
(ii) query latency over the 15-question suite, and (iii) answer correctness la-
bels.
9.6.1 ThingsBoard
No-Graph indexes 4521 Java files into 32428 chunks [28]. DKB builds a
Tree-sitter ontology with 4691 nodes and 8570 edges and indexes 32098
chunks [29]. LLM-KB builds a graph with 4299 nodes and 20419 edges and
indexes 22899 chunks [30].
Table 6: Code Coverage and Graph Completeness (ThingsBoard)
Method Chunks Chunk Cov. Nodes Edges LLM-skipped File SR
No-Graph 32428 1.000 — — — —
LLM-KB 22899 0.706 4299 20419 877 0.806
DKB 32098 0.990 4691 8570 — —
9.6.2 OpenMRS Core
No-Graph indexes 1258 Java files into 11495 chunks [31]. DKB builds a
Tree-sitter ontology with 1371 nodes and 1517 edges and indexes 11411
chunks [32]. LLM-KB builds a graph with 1052 nodes and 2957 edges and
indexes 7281 chunks [33].
22

Table 7: Code Coverage and Graph Completeness (OpenMRS-core)
Method Chunks Chunk Cov. Nodes Edges LLM-skipped File SR
No-Graph 11495 1.000 — — — —
LLM-KB 7281 0.633 1052 2957 440 0.650
DKB 11411 0.993 1371 1517 — —
9.7 Correctness Summary
To summarize Table 10, we collapse any “Correct+” detail level intoCor-
rect. Over the full 15-question Shopizer suite, DKB achieved the high-
est correctness (15/15), LLM-KB followed closely (13/15; 2 partial), while
No-Graph lagged (6/15) and exhibited the highest hallucination risk on
architecture-discovery queries (e.g., Q11). This aligns with broader evi-
dence that repository-level tasks stress retrieval adequacy and grounding
across many files [10,14,15].
Table 8: Correctness Summary on Full Shopizer Question Suite (from Ta-
ble 10)
Approach Correct Partial Incorrect
No-Graph 6 4 5
LLM-KB 13 2 0
DKB (Ours) 15 0 0
23

Table 9: Correctness Summary Across Repositories (Q= 15 per repository;
totals computed from Tables 10, 11, 12).
Repository Approach Correct Partial Incorrect
Shopizer No-Graph 6 4 5
Shopizer LLM-KB 13 2 0
Shopizer DKB (Ours) 15 0 0
ThingsBoard No-Graph 14 1 0
ThingsBoard LLM-KB 12 2 1
ThingsBoard DKB (Ours) 14 1 0
OpenMRS Core No-Graph 11 4 0
OpenMRS Core LLM-KB 13 1 1
OpenMRS Core DKB (Ours) 14 1 0
All (45 questions)No-Graph 31 9 5
All (45 questions)LLM-KB 38 5 2
All (45 questions)DKB (Ours) 43 2 0
9.8 Comparative Analysis
Deterministic Knowledge Base (DKB).Across the evaluated suites,
DKB provides the most reliable structural grounding when questions require
multi-hop reasoningandupstream discovery(e.g., identifying controller-
s/services that consume a dependency through interfaces). This advantage
comes from deterministic AST-derived relations that are frequently under-
specified by text similarity alone (typedinjects/extends/implementsedges) [9,
20]. In Shopizer, this is most visible on upstream questions such as controller
discovery (Q11), where bidirectional traversal (successors and predecessors)
materially improves evidence collection. In particular, Strategy C’s success
on Q11 is driven by theInterfaceConsumerExpandrule in Alg. 1: when a
retrieved concrete class implements an interface, the retriever additionally
pulls inconsumers of that interface(predecessors of the interface node), al-
lowing the system to cross interface boundaries and recover controllers/ser-
vices that depend on the interface rather than a specific implementation.
We note that correctness gains are workload-dependent: on some suites
(e.g., ThingsBoard) the vector-only baseline is already strong and DKB pri-
marily matches it while preserving deterministic coverage and low indexing
overhead (Table 9).
LLM-Generated Knowledge Base (LLM-KB).LLM-KB can achieve
high correctness and produces useful high-level architectural summaries (e.g.,
identifying event-driven triggers such as Q9). However, its indexing pipeline
is probabilistic and schema-bound: run logs show that extraction can omit
24

files (SKIPPED/MISSED by LLM), which reduces both the embedded corpus
and the resulting graph footprint (Tables 5, 6, 7). This is not only a per-
formance concern—missing files at index time create retrieval blind spots
at query time. LLM-KB also exhibits substantially higher end-to-end cost
on larger workloads due to LLM-mediated graph construction (Tables 3, 4).
Finally, raw edge totals should be interpreted cautiously: LLM-KB emits
a broaderdepends onrelation (often denser), while DKB edges are a nar-
rower, typed subset (injects/extends/implements), so edge counts are
not directly comparable as a completeness measure (Table 13).
Vector-Only RAG (No-Graph).No-Graph performs well on local-
ized questions when the necessary evidence is co-located within the top-k
retrieved chunks (e.g., Shopizer Q1 and Q15), and on some repositories it can
be competitive overall (e.g., ThingsBoard in Table 9). Its main failure mode
appears on questions where relevant evidence is distributed across multiple
files connected by architectural structure (inheritance, dependency injection,
interface-driven wiring). When structural neighbors are not retrieved, the
generator may fall back to framework conventions and produce ungrounded
or partially grounded claims (e.g., Shopizer Q11), reflecting known risks
when generation is weakly supported by retrieval [2, 5]. Overall, No-Graph
offers strong simplicity and often favorable latency, but it is less dependable
for multi-hop architectural tracing unless augmented with structure-aware
expansion.
10 Discussion
The results show a practical trade-off: LLM-generated indexing can be ex-
pensive and latency-unstable, while deterministic AST parsing produces a
usable code graph at low indexing cost [6,20]. Although DKB is not univer-
sally faster at query time than vector-only retrieval, it substantially improves
correctness on multi-hop structural questions by explicitly retrieving graph
neighborhoods instead of relying on inferred relationships from disjoint text
chunks [19].
Why indexing reliability matters.Table 5 shows that LLM-KB em-
beds substantially fewer chunks in comparison to the baseline, and its graph
has fewer nodes than DKB. This is an important distinction for enterprise
settings: even if LLM-KB answers many questions correctly, omission of
code artifacts at indexing time can create unpredictable blind spots. Prompt
improvements and retries may reduce skipping, but because the extraction
is probabilistic and schema-bound, completeness cannot be guaranteed in
25

multi-step/agentic workflows where intermediate structured outputs can fail
validation [21].
Key takeaway:For correctness-sensitive repository questions, deter-
ministic structure provides more reliable grounding than probabilistic ex-
traction, particularly for multi-hop and upstream discovery queries.
For correctness-sensitive tasks (impact analysis, controller discovery, trans-
action boundary inference), deterministic retrieval reduces hallucination by
grounding the model in explicit topology and by preserving stable corpus
coverage .
11 Threats to Validity
11.1 Internal Validity
Correctness labels were assigned using a human judgment protocol based on
the repository’s ground truth. Future work should include multiple annota-
tors, inter-rater agreement, and explicit evidence citations per answer. This
aligns with broader concerns in evaluating repository-level systems where
ground truth may be distributed and tool-mediated [14,15].
11.2 External Validity
We evaluate on three Java repositories (Shopizer, ThingsBoard, and Open-
MRS Core). Results may differ for other architectures or languages. Ex-
tending to multiple repos will strengthen generalization, consistent with
repository-level benchmark findings that performance varies by repo topol-
ogy and dependency structure [10,11].
11.3 Construct Validity
Correctness construct.TheCorrect/Partial/Incorrectlabels are intended
to measure repository-grounded correctness: (i) correct entities (classes/meth-
ods/files), (ii) correct structural relationships (e.g., controller→service→repository
chains, inheritance/DI edges), and (iii) avoidance of hallucinated compo-
nents. These labels are necessarily coarse and may not capture nuance (e.g.,
an answer may be structurally correct but omit a key configuration detail).
Stability across trials.Although Table 10 reports one representative
run, we executed multiple trials for each pipeline under the same settings
and observed similar correctness outcomes (and the same relative ordering
of approaches). We do not treat this as a statistical guarantee; future work
26

should report the exact number of trials, per-question agreement rate across
trials, and confidence intervals.
Coverage construct.Coverage signals derived from run logs (e.g., dis-
covered files, embedded chunk counts, graph node/edge counts, and explicit
SKIPPED/MISSED by LLMindicators) are proxies for indexing completeness,
but can be sensitive to instrumentation placement and path canonicaliza-
tion. Future work should export these counters as structured artifacts
(JSON/CSV), validate them against independent filesystem enumeration
(stable unique file IDs), and add additional completeness checks (e.g., per-
centage of files producing at least one embedding; percentage of graph nodes
mapped to an existing source file).
12 Reproducibility and Artifact Availability
All experimental code and (sanitized) run outputs for this paper will be re-
leased in the companion repository:https://github.com/Manideep-Reddy-Chinthareddy/
graph-based-rag-ast-vs-llm. To ensure bitwise-identical retrieval results
are feasible, we recommend pinning (i) a repository commit hash for both
the evaluation code and the target codebase, and (ii) the exact Python de-
pendency versions used to build embeddings and graphs.
12.1 Artifacts Provided
The repository contains three end-to-end evaluation scripts corresponding to
the compared pipelines: (i)nograph gemini.py(vector-only Naive RAG),
(ii)llmkb gemini.py(LLM-generated knowledge graph RAG), and (iii)
dkbgemini v2.py(deterministic Tree-sitter AST graph RAG). Each script
(a) scans the target repository, (b) builds the index (and graph when appli-
cable), (c) executes the same 15-question benchmark suite, and (d) writes a
structured JSON file with per-question answers and latency measurements.
12.2 Target Codebase and Version Pinning
Our experiments use a pinned fork of the Shopizer Java repository to keep
the target codebase static during analysis:https://github.com/Manideep-Reddy-Chinthareddy/
shopizer-fork. [24] For strict reproducibility, the exact fork commit used in
evaluation should be recorded and reported as:SHOPIZER FORK COMMIT=6a4a0a65a3408ee8f62
597b51d1b3aac24b77dee. Likewise, the artifact repository commit should
be recorded as:ARTIFACT COMMIT=efa7c88ff998877d82
5b722a595b9f0201719da4. All reported counts (e.g., discovered Java files,
27

embedded chunk counts, graph node/edge counts) are a function of these
pinned commits.
12.3 Environment and Dependencies
All pipelines are implemented in Python and rely on: (1) a vector store
for embedding retrieval, (2) an embedding model and LLM provider used
through a single framework abstraction, and (3) (for DKB) an incremen-
tal parsing stack based on Tree-sitter [20]. To reproduce results, create a
clean virtual environment and install the exact pinned dependencies from
requirements.txt(recommended) or an equivalent lockfile. The scripts
expect the LLM provider API key to be available via environment variables
(e.g.,GOOGLE APIKEYwhen using Gemini through LangChain wrappers).
12.4 Running the Benchmarks
All three scripts are designed to be executed as standalone entry points.
The only required local configuration is setting the target repository path
(the Shopizer clone) in the script configuration (e.g.,PROJECT ROOT) or via
an environment variable if the repository exposes one. Each script runs
the same fixed suite ofQ= 15 benchmark questions (embedded directly
in the script) using shared retrieval hyperparameters:chunk size=1000,
chunk overlap=100, and top-k= 10 vector retrieval.
12.5 Outputs, Logs, and Metric Extraction
Each pipeline writes a JSON artifact containing: (i) indexing time (db gentime),
(ii) graph build time when applicable (graph generation time), and (iii) a
list of per-question responses with end-to-end latency (time taken seconds).
Default output filenames are:nograph gemini ragresponse.json,llmkb gemini ragresponse.json,
anddkb gemini v2ragresponse.json. For graph-based methods, the
scripts additionally emit rendered graph visualizations (HTML) and/or in-
termediate logs for coverage auditing (e.g., explicitSKIPPED/MISSED by LLM
indicators for LLM-KB).
12.6 Reproducing Summary Tables
Table??is computed directly from the per-question latency fields in the
JSON artifacts (mean, standard deviation, median, and min–max). Table 5
is computed from the indexing logs and JSON counters emitted by each
pipeline (discovered files, embedded chunks, and graph size). To reproduce
28

the reported numbers, users should: (1) run all three scripts on the same
pinned Shopizer commit, (2) keep retrieval hyperparameters identical across
runs, and (3) derive aggregate statistics from the emitted JSONs without
manual editing.
12.7 Notes on Stochasticity
Although DKB’s graph construction is deterministic (AST-derived) given
a fixed codebase and parser version, LLM-KB’s extraction is inherently
stochastic due to model variability and schema compliance failure modes,
and because non-greedy sampling schemes can produce a distribution of
outputs even for identical prompts. [21, 27]. To quantify variance, we
recommend repeating LLM-KB indexing and evaluation for multiple trials
under identical settings, and reporting the distribution of (i) skipped-file
counts, (ii) embedded chunk counts, and (iii) latency outliers.
13 Ethical Considerations
Graph-RAG for codebases can expose sensitive implementation details if
used on proprietary repositories. Enterprise deployments should enforce ac-
cess control, auditing, and avoid transmitting restricted code to third-party
services without appropriate approval. Agentic and tool-using patterns can
amplify both capability and risk in software engineering deployments [21,23].
14 Limitations and Future Work
Deterministic AST extraction can miss behaviors driven by reflection, runtime-
generated code, or dynamic dispatch that is not explicit in syntax [20].
Future work includes hybrid graphs (deterministic backbone + LLM se-
mantic edges), richer call graph resolution, and evaluation across multiple
languages [10,19].
For LLM-KB, future work should (i) export explicit counters for skipped
files and schema failures, (ii) add retries with smaller batch sizes, (iii) add
static validation passes that ensure every discovered Java file yields a cor-
responding analyzed record, and (iv) decouple document embedding from
LLM-derived class maps to prevent corpus shrinkage when graph extraction
is incomplete [21].
29

15 Conclusion
We benchmarked vector-only RAG, an LLM-generated graph RAG (LLM-
KB), and a deterministic AST-derived graph RAG (DKB) for repository-
level code analysis on Shopizer, with additional runs on ThingsBoard and
OpenMRS Core [2, 6, 20]. Across repositories, DKB consistently delivered
strong correctness with low and predictable indexing overhead: on Shopizer,
DKB achieved 15/15 correct answers while remaining close to the vector
baseline in build time (22.09s vs. 18.41s), whereas LLM-KB incurred large
offline graph-generation overhead (215.09s total) despite competitive query-
time performance.
A key finding is that LLM-KB’s indexing pipeline can beincompleteat
the file level. On Shopizer, the run logs report 377 filesSKIPPED/MISSED
by LLM, yielding a per-file success rate of 0.688 (833/1210). This incom-
pleteness correlates with reduced embedded corpus size: LLM-KB indexed
3465 chunks (0.641 coverage vs. the No-Graph baseline of 5403), while DKB
indexed 4873 chunks (0.902 coverage). The extracted graph footprint is also
smaller under LLM-KB on Shopizer (842 nodes) compared to DKB (1158
nodes), which can reduce the effectiveness of graph neighborhood expansion
for multi-hop queries.
Cost measurements further highlight the trade-off. On Shopizer, end-to-
end cost (indexing + 15 questions) was$0.04 for No-Graph,$0.09 for DKB,
and$0.79 for LLM-KB; on the larger combined OpenMRS-core + Things-
Board workload, costs were$0.149 (No-Graph),$0.317 (DKB), and$6.80
(LLM-KB), widening the normalized multiplier for LLM-KB substantially.
Overall, when correctness and coverage reliability matter, DKB provides
a practical “compiler-in-the-loop” retrieval strategy: it preserves indexing
completeness signals close to the vector baseline while improving multi-hop
architectural grounding without the high and variable cost of LLM-mediated
graph construction.
Overall, the cross-repository pattern is consistent: DKB provides a low-
overhead, deterministic structural signal and near-baseline coverage, while
LLM-KB can achieve strong correctness with lesser code to implement but
introduces indexing-time incompleteness and cost inflation due to proba-
bilistic, schema-bound extraction.
30

References
[1] A. Vaswaniet al., “Attention Is All You Need,” inProc. NeurIPS,
10.48550/arXiv.1706.03762, 2017.
[2] P. Lewiset al., “Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks,”arXiv preprint arXiv:2005.11401, 2020.
[3] V. Karpukhinet al., “Dense Passage Retrieval for Open-Domain
Question Answering,” inProc. EMNLP,10.18653/v1/2020.emnlp-
main.550, 2020.
[4] G. Izacardet al., “Few-shot Learning with Retrieval Augmented Lan-
guage Models,”arXiv preprint arXiv:2208.03299, 2022.
[5] Y. Gaoet al., “Retrieval-Augmented Generation for Large Language
Models: A Survey,”arXiv preprint arXiv:2312.10997, 2023.
[6] D. Edgeet al., “From Local to Global: A Graph RAG Approach
to Query-Focused Summarization,”arXiv preprint arXiv:2404.16130,
2024.
[7] S. Knollmeyer, O. Caymazer, and D. Grossmann, “Document
GraphRAG: Knowledge Graph Enhanced Retrieval Augmented Gener-
ation for Document Question Answering Within the Manufacturing Do-
main,”Electronics,10.3390/electronics14112102vol. 14, no. 11, p. 2102,
2025.
[8] Z. Guoet al., “LightRAG: Simple and Fast Retrieval-Augmented Gen-
eration,”arXiv preprint arXiv:2410.05779, 2024.
[9] Yilin Zhang and Xinran Zhao and Zora Zhiruo Wang and Chenyang
Yang and Jiayi Wei and Tongshuang Wu, “cAST: Enhancing Code
Retrieval-Augmented Generation with Structural Chunking via Ab-
stract Syntax Tree,”arXiv preprint arXiv:2506.15655, 2025.
[10] Y. Taoet al., “Retrieval-Augmented Code Generation: A Sur-
vey with Focus on Repository-Level Approaches,”arXiv preprint
arXiv:2510.04905, 2025.
[11] .Zezhou Yang and Sirong Chen and Cuiyun Gao and Zhenhao Li
and Xing Hu and Kui Liu and Xin Xia, “An Empirical Study of
Retrieval-Augmented Code Generation: Challenges and Opportuni-
ties,”arXiv:2501.13742, 2025.
31

[12] M. Chenet al., “Evaluating Large Language Models Trained on Code,”
arXiv preprint arXiv:2107.03374, 2021.
[13] J. Austinet al., “Program Synthesis with Large Language Models,”
arXiv preprint arXiv:2108.07732, 2021.
[14] C. E. Jimenezet al., “SWE-bench: Can Language Models Resolve Real-
World GitHub Issues?” inProc. ICLR arxiv preprint arXiv:2310.06770,
2024.
[15] T. Liu, C. Xu, and J. McAuley, “RepoBench: Benchmarking
Repository-Level Code Auto-Completion Systems,” inProc. ICLR,
arxiv preprint arXiv:2306.03091, 2024.
[16] F. Zhanget al., “RepoCoder: Repository-Level Code Completion
Through Iterative Retrieval and Generation (introducing RepoEval),”
arXiv preprint arXiv:2303.12570, 2023.
[17] S. Ouyanget al., “RepoGraph: Enhancing Repository-Level
Code Generation with Graph-Based Retrieval,” inProc. ICLR,
arXiv:2410.14684, 2025.
[18] Hongyuan Tao and Ying Zhang and Zhenhao Tang and Hongen
Peng and Xukun Zhu and Bingchang Liu and Yingguang Yang and
Ziyin Zhang and Zhaogui Xu and Haipeng Zhang and Linchao Zhu
and Rui Wang and Hang Yu and Jianguo Li and Peng Di, “Code
Graph Model (CGM): A Graph-Integrated Large Language Model for
Repository-Level Software Engineering Tasks,”OpenReview preprint,
2025arXiv:2505.16901.
[19] Qinggang Zhang and Shengyuan Chen and Yuanchen Bei and Zheng
Yuan and Huachi Zhou and Zijin Hong and Hao Chen and Yilin
Xiao and Chuang Zhou and Junnan Dong and Yi Chang and Xiao
Huang, “A Survey of Graph Retrieval-Augmented Generation for
Customized Large Language Models,”OpenReview/arXiv preprint
arXiv:2501.13958, 2025.
[20] Tree-sitter Contributors, “Tree-sitter: A parser generator tool and
an incremental parsing library,” 2025. [Online]. Available:https:
//tree-sitter.github.io/. Accessed: Dec. 29, 2025.
[21] Aditi Singh and Abul Ehtesham and Saket Kumar and Tala Talaei
Khoei, “Agentic Retrieval-Augmented Generation: A Survey on Agen-
tic RAG,”arXiv preprint arXiv:2501.09136, 2025.
32

[22] Damian Gnieciak and Tomasz Szandala,“Large Language Models Ver-
sus Static Code Analysis Tools: A Systematic Benchmark for Vulnera-
bility Detection,”arXiv preprint arXiv:2508.04448, 2025.
[23] GitHub Resources, “Enhancing software development with
retrieval-augmented generation,” 2025. [Online]. Avail-
able:https://github.com/resources/articles/ai/
software-development-with-retrieval-augmentation-generation-rag.
Accessed: Dec. 29, 2025.
[24] Shopizer Team, “Shopizer: Java E-commerce Software,” 2025.
[Online]. Available: https://github.com/shopizer-ecommerce/shopizer.
[Accessed: Dec. 30, 2025].
[25] S. Genget al., “Generating Structured Outputs from Language Models:
Benchmark and Studies,”arXiv preprint arXiv:2501.10868, 2025. DOI:
10.48550/arXiv.2501.10868.
[26] B. Agarwal, I. Joshi, and V. Rojkova, “Think Inside the JSON: Rein-
forcement Strategy for Strict LLM Schema Adherence,”arXiv preprint
arXiv:2502.14905, 2025. DOI: 10.48550/arXiv.2502.14905.
[27] J. Hayeset al., “Measuring memorization in language models via prob-
abilistic extraction,”arXiv preprint arXiv:2410.19482, 2024.
[28] M. R. Chinthareddy, “ThingsBoard: No-Graph (vector-only) run
logs (15-question suite),” ingraph-based-rag-ast-vs-llm(GitHub
repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm.
Accessed: Dec. 29, 2025.
[29] M. R. Chinthareddy, “ThingsBoard: DKB (Tree-sitter AST
graph) run logs (15-question suite),” ingraph-based-rag-ast-vs-llm
(GitHub repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm. Ac-
cessed: Dec. 29, 2025.
[30] M. R. Chinthareddy, “ThingsBoard: LLM-KB (LLM-extracted
graph) run logs (15-question suite),” ingraph-based-rag-ast-vs-llm
(GitHub repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm. Ac-
cessed: Dec. 29, 2025.
33

[31] M. R. Chinthareddy, “OpenMRS Core: No-Graph (vector-
only) run logs (15-question suite),” ingraph-based-rag-ast-vs-llm
(GitHub repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm. Ac-
cessed: Dec. 29, 2025.
[32] M. R. Chinthareddy, “OpenMRS Core: DKB (Tree-sitter AST
graph) run logs (15-question suite),” ingraph-based-rag-ast-vs-llm
(GitHub repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm. Ac-
cessed: Dec. 29, 2025.
[33] M. R. Chinthareddy, “OpenMRS Core: LLM-KB (LLM-extracted
graph) run logs (15-question suite),” ingraph-based-rag-ast-vs-llm
(GitHub repository), 2025. [Online]. Available:https://github.com/
Manideep-Reddy-Chinthareddy/graph-based-rag-ast-vs-llm. Ac-
cessed: Dec. 29, 2025.
[34] OpenMRS Community, “openmrs-core: OpenMRS API and web appli-
cation code,” GitHub repository. [Online]. Available:https://github.
com/openmrs/openmrs-core. Accessed: Jan. 12, 2026.
[35] ThingsBoard Team, “thingsboard: Open-source IoT platform for data
collection, processing, visualization, and device management,” GitHub
repository. [Online]. Available:https://github.com/thingsboard/
thingsboard. Accessed: Jan. 12, 2026.
A Per-question correctness tables
This section shows the full detailed view of the questions and correctness
for each of the question per repository/RAG model
A.1 Shopizer
34

Table 10: Correctness Comparison of RAG Strategies on the Full Shopizer
Question Suite (Q= 15)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q1 Repository save imple-
mentationCorrect(Iden-
tified SimpleJ-
paRepository,
proxy pattern.)Correct(Iden-
tified proxy
interceptors,
JPA repository
delegation.)Correct(Iden-
tified SimpleJ-
paRepository,
proxy pattern.)
Q2 UserStoreHandler respon-
sibilitiesCorrect(Identi-
fied user store op-
erations and re-
lated handlers.)Correct(Iden-
tified User-
StoreHandler
responsibilities
and operations.)Correct(Identi-
fied handler re-
sponsibilities, au-
thentication inte-
gration.)
Q3 Custom vs generic meth-
odsIncorrect(As-
sumed no custom
methods, missed
overrides.)Partial(Men-
tioned custom
methods but
missed some key
overrides.)Correct(Iden-
tified custom
methods and
overrides accu-
rately.)
Q4 PricingService logicPartial(Gave
general pric-
ing description,
missed Product-
PriceUtils.)Correct(Iden-
tified Product-
PriceUtils and
pricing flow.)Correct(Iden-
tified pricing
calculation
details and Pro-
ductPriceUtils.)
Q5 Inventory updates and
triggersIncorrect(Hal-
lucinated direct
inventory update
flow.)Correct(Identi-
fied inventory up-
dates and rele-
vant services.)Correct(Identi-
fied triggers and
correct inventory
update path.)
Q6 CustomerService.save flowPartial(Gave
generic reposi-
tory save flow,
missed key
chain.)Correct(Traced
correct service
chain and region
resolution.)Correct(Identi-
fied correct inher-
itance chain and
resolution.)
Q7 Checkout flow tracePartial(Iden-
tified entry but
missed down-
stream services.)Correct(Iden-
tified controllers
and key down-
stream services.)Correct(Identi-
fied auth checks,
payment/ship-
ping services.)
Q8 Authentication / autho-
rization handlingCorrect(Identi-
fied auth utilities
and flow.)Correct(Identi-
fied auth flow and
relevant security
components.)Correct(Identi-
fied auth flow and
relevant security
components.)
Q9 SearchService triggerIncorrect(Hal-
lucinated direct
trigger instead of
event-driven.)Correct(Iden-
tified event-
driven trigger for
SearchService.)Correct(Iden-
tified IndexPro-
ductEventLis-
tener / event
flow.)
35

Table 10 (continued)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q10 Impact of deletion (mod-
ule breakage)Partial(Gave
generic cascade
impact.)Correct(Iden-
tified impacted
payment/ship-
ping integra-
tions.)Correct(Identi-
fied Stripe/USP-
S/Facebook
API integration
impact.)
Q11 Controllers using Shop-
ping Cart logicIncorrect(Hal-
lucinated con-
trollers and
missed interface
usage.)Correct(Iden-
tified OrderApi
and related
controllers.)Correct(Iden-
tified interface-
consumer ex-
pansion across
boundary.)
Q12 Product / category rela-
tionship logicCorrect(Identi-
fied relationship
mappings and
relevant entities.)Correct(Iden-
tified mappings
and relationship
logic.)Correct(Identi-
fied relationship
mappings and
correct logic.)
Q13 Transactional proof (cre-
ateOrder multi-write
boundary)Incorrect
(Missed trans-
actional an-
notations and
evidence.)Partial(Identi-
fied unit-of-work
but missed some
boundaries.)Correct(Iden-
tified transac-
tional scope
across multi-repo
writes.)
Q14 Shipping module logicCorrect(Iden-
tified shipping
integrations and
service flow.)Correct(Iden-
tified shipping
module and
service logic.)Correct(Iden-
tified shipping
module and
service logic.)
Q15 Order ID generationCorrect(Iden-
tified @Table-
Generator and
table-based
sequencing.)Correct
(Identified
SMSEQUENCER
and table genera-
tor usage.)Correct(Ex-
plained atomic
increment and
sequencing table
usage.)
A.2 ThingsBoard
Table 11: Correctness Comparison of RAG Strategies on the ThingsBoard
Question Suite (Q= 15)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q1 App entry point & startupCorrect
(Identified
ThingsboardServerApplication
/ Spring Boot
startup.)Incorrect
(Claimed no
single entry
point; missed
ThingsboardServerApplication.)Correct
(Identified
ThingsboardServerApplication
and initial boot-
strap.)
36

Table 11 (continued)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q2 Login/authentication
pathCorrect(Traced
AuthController
→auth man-
ager/provider→
token response.)Correct(Iden-
tified security
filter/provider
chain and JWT/-
token creation.)Correct(Traced
request→au-
thentication→
token issuance
clearly.)
Q3 Upstream consumers of
authCorrect(Named
dependent con-
trollers/filters
using the same
token/auth
components.)Correct(Identi-
fied shared auth
filters/compo-
nents and their
consumers.)Correct(Found
upstream con-
sumers of auth
utilities across
controllers/fil-
ters.)
Q4 Telemetry ingestion
pipelinePartial(High-
level service
chain; missing a
fully grounded
end-to-end
trace.)Partial(De-
scribed generic
TB telemetry
path; incomplete
concrete call
chain.)Partial(Ex-
plained in-
gestion/actor
handoff but
noted miss-
ing controller
evidence.)
Q5 MQTT transport handler
flowCorrect(Traced
MQTT inbound
decode→mes-
sage creation→
next dispatch.)Correct(Identi-
fied message ob-
ject + next-hop
dispatch in trans-
port layer.)Correct(Traced
handler→mes-
sage conversion
→routing cor-
rectly.)
Q6 Netty pipeline initializa-
tionCorrect(Listed
pipeline handlers
and responsibili-
ties in initializer.)Correct(Re-
covered handler
order and roles
consistent with
Netty init.)Correct(Iden-
tified initial-
izer wiring:
codec/SSL/han-
dlers in order.)
Q7 TbContext responsibilitiesCorrect
(Explained
TbContextas
runtime services
context for rule
nodes.)Correct
(Described
TbContextrole +
service access for
rule nodes.)Correct(Iden-
tifiedTbContext
responsibilities
and providers.)
Q8 RuleEngine tellNext mes-
sageCorrect
(Explained
RuleNodeToRuleChainTellNextMsg
next-hop rout-
ing.)Correct(Iden-
tified tellNext
as rule-chain
transition mes-
sage; named
consumer.)Correct(Ex-
plained actor
consumption +
next-hop logic for
tellNext.)
Q9 Alarm API to persistenceCorrect(Traced
AlarmController
→service→
DAO/repository
persistence.)Correct(Iden-
tified con-
troller/service/-
DAO chain down
to persistence.)Correct(Cor-
rect end-to-end
alarm flow to
persistence.)
37

Table 11 (continued)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q10 Dashboard access controlCorrect(Lo-
cated permis-
sion/tenant
checks and en-
forcing classes.)Correct(Iden-
tified access-
control checks at
controller/service
boundaries.)Correct(Pinned
enforcement
points for dash-
board permis-
sions.)
Q11 YAML config load/over-
rideCorrect
(Explained
Spring config-
name/properties
binding; noted
tb-mqtt-transport.yml.)Partial(Focused
on test config
props; did not
pin runtime
binding/override
path.)Correct(Iden-
tified runtime
config bind-
ing/override
mechanism
for transport
YAML.)
Q12 Timeseries storage back-
endCorrect(Iden-
tified timeseries
DAO/repository
layer and back-
end selection
logic.)Correct(Named
primary time-
series persistence
components and
selection.)Correct(Cor-
rect timeseries
storage path
and main DAO
classes.)
Q13 Queue type selectionCorrect
(Pointed to
config-driven
Kafka/Rab-
bit/etc selection
and ownership.)Correct(Iden-
tified where
queue impl is
chosen and pro-
ducer/consumer
modules.)Correct(Recov-
ered config points
selecting queue
implementation.)
Q14 ActorSystemContext +
routingCorrect(Ex-
plained Ac-
torSystemCon-
text construction
and tenant/de-
vice partition
routing.)Correct(Identi-
fied routing/par-
tition providers
for tenant/device
decisions.)Correct(Traced
construction +
routing com-
ponents for
partitioning.)
Q15 Add transport or RuleN-
odeCorrect(Out-
lined required
interfaces +
registration/dis-
covery locations.)Correct(Ex-
plained extension
points and
registration
mechanism.)Correct(Gave
correct steps to
add new trans-
port or rule
node.)
A.3 OpenMRS Core
38

Table 12: Correctness Comparison of RAG Strategies on the OpenMRS
Question Suite (Q= 15)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q1 Web init + Context ses-
sion lifecycleCorrect(Iden-
tifiedListener
+OpenmrsFilter
session open/-
close.)Correct(Traced
Listener/boot-
strap and request
filter manag-
ing Context
sessions.)Correct
(Identified
Listener/OpenmrsFilter
order-of-calls.)
Q2 Service lookup / Service-
Context wiringCorrect
(Explained
Context.getService
via Spring-
backed
ServiceContext.)Correct
(Identified
ServiceContext
as Spring reg-
istry for service
beans.)Correct
(Traced Con-
text facade→
ServiceContext
→service impl.)
Q3 Authentication pathCorrect(Traced
Context.authenticate
→user service
check→User-
Context.)Correct(Iden-
tified credential
validation + stor-
ing authenticated
user in context.)Correct(Cor-
rect end-to-end
auth flow and
where user is
stored.)
Q4 Privilege enforcementCorrect
(Identified
@Authorized/AOP
advice + re-
quirePrivilege
checks.)Correct(Lo-
cated privi-
lege checks in
AOP/authoriza-
tion layer and
call sites.)Correct(Cor-
rectly described
authorization ad-
vice + upstream
callers.)
Q5 saveEncounter chain +
transaction boundaryPartial(Traced
service→
DAO chain
but did not pin
@Transactional
boundary.)Correct
(Located
@Transactional
at service layer
around saveEn-
counter.)Correct(Identi-
fied service-layer
transaction
boundary and
DAO persis-
tence.)
Q6 Upstream consumers of
saveEncounterPartial
(Gave generic
HL7/import con-
sumers without
concrete caller
classes.)Incorrect
(Listed life-
cycle/handler
concepts; missing
concrete up-
stream callers of
saveEncounter.)Correct(Named
concrete up-
stream con-
sumers like
ORUR01Handler
and builders.)
Q7 VisitService.saveVisit
child handlingPartial(Dis-
cussed en-
counter/visit
handlers but not
VisitServiceImpl.saveVisit
specifics.)Partial(High-
level description;
lacks grounded
saveVisit/Obs
persistence be-
havior.)Partial(Ex-
plained related
handler seman-
tics; missing
direct saveVisit
implementation
proof.)
39

Table 12 (continued)
Question Theme / Ground-
Truth SignalNo-Graph
(Vector Only)LLM-KB
(Probabilis-
tic)DKB (Deter-
ministic)
Q8 Obs→Concept + valida-
tionCorrect(Ex-
plained Con-
cept linkage +
datatype/nu-
meric validation
points.)Correct(Iden-
tified Concept
references and
where Obs value
validation oc-
curs.)Correct(Cor-
rectly described
validation +
concept binding.)
Q9 Global properties cache/s-
toreCorrect(Traced
Administra-
tionService
global property
persistence +
caching.)Correct(Iden-
tified storage,
cache, and re-
trieval path for
global proper-
ties.)Correct(Cor-
rect end-to-end
global property
flow.)
Q10 User create/update persis-
tenceCorrect(Traced
UserServiceImpl.saveUser
→DAO→do-
main objects.)Correct(Iden-
tified service-
to-DAO flow,
including cre-
dentials/person
linkage.)Correct(Cor-
rect user save
flow down to
persistence.)
Q11 Module classloadingCorrect
(Explained
OpenmrsClassLoader
vs
ModuleClassLoader
responsibilities.)Correct(Identi-
fied module load-
/start classload-
ing usage sites.)Correct(Cor-
rect mapping of
classloader roles
and usage.)
Q12 ModuleActivator lifecycleCorrect(Lo-
cated activator
invocation path
during module
startup/shut-
down.)Correct(Identi-
fied ModuleFac-
tory invoking ac-
tivators and call
path.)Correct(Cor-
rect lifecycle
call path for
activators.)
Q13 Daemon elevated execu-
tionCorrect(Ex-
plained daemon
thread execu-
tion and privi-
lege/user context
for daemon runs.)Correct(Iden-
tified daemon
token/elevation
pattern and
context setup.)Correct(Cor-
rect daemon
mechanism and
context/privi-
leges.)
Q14 messages.properties + lo-
caleCorrect(Traced
MessageSource
lookup and where
locale is chosen.)Correct(Identi-
fied message res-
olution + locale
selection path.)Correct(Cor-
rect i18n message
resolution flow.)
Q15 Core flow + wiring loca-
tionsPartial(Reason-
able layered flow
but did not name
concrete Spring
wiring artifacts.)Correct(Identi-
fied module/ser-
vice boundaries
and Spring
wiring locations
at a high level.)Correct
(Mapped bound-
aries and where
wiring is de-
fined/loaded.)
40

B Prompts
This appendix lists the exact prompt templates used by each pipeline im-
plementation.
B.1 No-Graph (Vector-only) RAG: Answer Prompt
1You are a Senior Java Engineer . Answer based on the code
provided .
2
3Context :
4{ context }
5
6Question : { question }
Listing 4: Answer prompt used by the No-Graph (vector-only) baseline.
B.2 DKB (Tree-sitter AST Graph RAG): Answer Prompt
1You are a Senior Java Engineer . Answer based on the code
provided .
2Use the [ ONTOLOGY INFO ] to understand the architecture .
3
4Context :
5{ context }
6
7Question : { question }
Listing 5: Answer prompt used by DKB (AST-derived graph) with ontology-
aware context.
B.3 LLM-KB (LLM-Generated Graph): Index-Time Extrac-
tion Prompt
1You are a static code analyzer . Analyze the provided batch of
Java files .
2For EACH file , identify the Class Name and any Custom
Dependencies ( other classes in the project that are used ).
3Ignore standard libraries ( java .*, spring framework , etc .).
4
5Return a JSON object with a single key " results " containing a
list of objects , one for each file .
6
7Schema :
8{
41

9" results ": [
10{
11" file_path ": " path /to/ File . java ",
12" class_name ": " NameOfClass ",
13" dependencies ": [" OtherClass1 ", " OtherClass2 "]
14}
15]
16}
17
18Input Files :
19{ batch_content }
Listing 6: Extraction prompt used to produce per-file class + dependency
records for LLM-KB graph construction.
B.4 LLM-KB (LLM-Generated Graph): Answer Prompt
1You are a Senior Java Architect . Answer the question based on
the provided Code Context .
2The context includes the main classes found via vector search ,
AND their dependencies from the Gemini - generated Knowledge
Graph .
3
4Context :
5{ context }
6
7Question : { question }
Listing 7: Answer prompt used by LLM-KB at query time (vector hits +
graph-expanded neighbors).
C Graph Schema and Edge Types (Implemented)
This appendix specifies the concrete node/edge schema and query-time ex-
pansion rules used by the two graph-based systems in this paper: the deter-
ministic Tree-sitter graph (DKB) and the LLM-extracted graph (LLM-KB).
C.1 DKB (Tree-sitter) Ontology Graph
DKB builds a directed graphG= (V, E) by parsing Java source files using
Tree-sitter queries for type discovery and dependency signals (field types and
constructor parameter types). Each discovered type is mapped to its defin-
ing file path and inserted as a node with atypeattribute (class,interface,
42

enum,record,annotation) and apathattribute. This node typing is de-
rived directly from the Tree-sitter declaration node kind (* declaration)
at indexing time.
C.1.1 Node Types
Nodes represent project-local Java types discovered by parsing declarations:
•class(class declaration)
•interface(interface declaration)
•enum(enum declaration)
•record(record declaration)
•annotation(annotation type declaration)
C.1.2 Edge Types (Relation Labels)
Edges are directed from asource typeto atarget typewith arelationlabel:
•injects: emitted when a class has a field whose declaredtype identifier
matches another project type, or when a constructor formal parame-
ter type matches another project type. This is a lightweight proxy for
dependency injection and composition.
•extends: emitted when the Tree-sitter AST provides asuperclass
(classes) or extendedinterfaces(interfaces).
•implements: emitted when a class/record/enum declaresinterfaces.
C.1.3 Inheritance Extraction Rules
Inheritance relationships are extracted by reading the declaration node fields
(e.g.,superclassandinterfaces) and mapping eachtype identifierin
those fields to a known project type. For interface declarations,interfaces
is treated as anextendsrelation (interfaces extending interfaces).
C.1.4 Query-time Expansion Semantics (Bidirectional + Interface-
Consumer Expansion)
At query time, DKB performs:
1.Initial retrieval:top-kvector retrieval over code chunks.
43

2.Bidirectional neighborhood expansion:for each retrieved class
v, include both successorsSucc(v) (downstream dependencies) and
predecessorsPred(v) (upstream consumers).
3.Interface-consumer expansion (critical fix):if a retrieved con-
crete classchas animplementsedge to an interfaceI, then addition-
ally includepredecessors ofI(classes that inject/use the interface),
improving upstream controller discovery across interface boundaries.
C.1.5 Context Assembly Budgeting
To bound prompt size, DKB emits (i) an excerpt of each initially retrieved
chunk and (ii) bounded excerpts from neighbor files when expanding suc-
cessors and predecessors, while also emitting a lightweight relationship sum-
mary ([ONTOLOGY INFO]) to make the traversal trace visible.
C.2 LLM-KB (LLM-extracted) Dependency Graph
LLM-KB builds a directed graph by batching Java files into prompts and
requiring the model to return a structured list of records: (file path,
class name,dependencies). Nodes are added for each extractedclass name,
and edges are added from the class to each dependency with relation label
depends on. Non-project dependencies are intended to be excluded by in-
struction (e.g.,java.*, Spring framework packages).
C.2.1 Node Types
LLM-KB treats all extracted entities as:
•class: nodes added from LLM-emittedclass name.
C.2.2 Edge Types
•depends on: directed edgec→dfor each dependencydemitted by
the LLM for classc.
C.2.3 Indexing Completeness Signals
Because extraction is schema-bound and performed in batches under context
constraints, the implementation truncates file content per prompt batch and
explicitly detects when an input file does not appear in the structured out-
put, emittingSKIPPED/MISSED by LLMfor missing files. This signal is used
in the paper as direct evidence of probabilistic extraction incompleteness.
44

C.2.4 Query-time Expansion Semantics
At query time, LLM-KB performs top-kvector retrieval and then expands
to graph neighbors using:
•Successors:dependencies of the retrieved class (downstream).
•Predecessors:classes that reference the retrieved class (upstream).
Unlike DKB, LLM-KB does not enforce interface-aware expansion based on
explicitimplements/extendsedges, because the extracted schema does not
distinguish inheritance from usage.
C.3 Summary: Why the Schema Matters
The DKB schema istyped(distinguishingextendsvs.implementsvs.injects)
and supports deterministic upstream discovery via predecessors and interface-
consumer expansion. The LLM-KB schema iscoarser(singledepends on
relation), which can be sufficient for many dependency summaries but makes
higher-precision traversal (e.g., inheritance-only expansion or interface bound-
ary fixes) harder without additional post-processing.
45

Table 13: Graph Schema Summary: Node and Edge Types (DKB vs. LLM-
KB)
System Element Type / Label Direction Extraction / Semantics
DKB Nodeclass– Added when a
class declarationis dis-
covered; node storespathto
defining file.
DKB Nodeinterface– Added when an
interface declaration
is discovered; node stores
path.
DKB Nodeenum– Added when an
enum declarationis dis-
covered; node storespath.
DKB Noderecord– Added when a
record declarationis
discovered; node storespath.
DKB Nodeannotation– Added when an
annotation type declaration
is discovered; node stores
path.
DKB EdgeinjectsA→BEmitted whenAhas a field
type or constructor param-
eter type matching project-
local typeB(lightweight
DI/composition signal).
DKB EdgeextendsA→BEmitted whenAdeclares su-
perclassB(class inheritance)
or when an interface extends
another interface.
DKB EdgeimplementsA→IEmitted when
class/record/enumAde-
clares interfaceIin its
implements list.
DKB Expansion
ruleSucc– Downstream neighborhood:
include successors of re-
trieved seed types up to hop
depthd.
DKB Expansion
rulePred– Upstream neighborhood:
include predecessors (con-
sumers) of retrieved seed
types up to hop depthd.
DKB Expansion
ruleInterfaceConsumerExpand– If seed typeAimplements
interfaceI, add predecessors
ofI(interface consumers)
to recover upstream con-
trollers/services across inter-
face boundaries.
LLM-KB Nodeclass– Added from LLM-emitted
class name(per-file struc-
tured analysis record).
LLM-KB Edgedepends onA→BEmitted for each dependency
string/classBreturned by
the LLM for classA; schema
does not distinguish inheri-
tance vs. usage unless post-
processed.
LLM-KB Completeness
signalSKIPPED/MISSED– Logged when an input file
in a prompt batch does not
appear in the model’s struc-
tured output (probabilistic
extraction incompleteness).46