# AI Agent for Reverse-Engineering Legacy Finite-Difference Code and Translating to Devito

**Authors**: Yinghan Hou, Zongyou Yang

**Published**: 2026-01-26 11:31:00

**PDF URL**: [https://arxiv.org/pdf/2601.18381v1](https://arxiv.org/pdf/2601.18381v1)

## Abstract
To facilitate the transformation of legacy finite difference implementations into the Devito environment, this study develops an integrated AI agent framework. Retrieval-Augmented Generation (RAG) and open-source Large Language Models are combined through multi-stage iterative workflows in the system's hybrid LangGraph architecture. The agent constructs an extensive Devito knowledge graph through document parsing, structure-aware segmentation, extraction of entity relationships, and Leiden-based community detection. GraphRAG optimisation enhances query performance across semantic communities that include seismic wave simulation, computational fluid dynamics, and performance tuning libraries. A reverse engineering component derives three-level query strategies for RAG retrieval through static analysis of Fortran source code. To deliver precise contextual information for language model guidance, the multi-stage retrieval pipeline performs parallel searching, concept expansion, community-scale retrieval, and semantic similarity analysis. Code synthesis is governed by Pydantic-based constraints to guarantee structured outputs and reliability. A comprehensive validation framework integrates conventional static analysis with the G-Eval approach, covering execution correctness, structural soundness, mathematical consistency, and API compliance. The overall agent workflow is implemented on the LangGraph framework and adopts concurrent processing to support quality-based iterative refinement and state-aware dynamic routing. The principal contribution lies in the incorporation of feedback mechanisms motivated by reinforcement learning, enabling a transition from static code translation toward dynamic and adaptive analytical behavior.

## Full Text


<!-- PDF content starts -->

AI Agent for Reverse-Engineering Legacy
Finite-Difference Code and Translating to Devito
Yinghan Hou
Department of Earth Science and Engineering
Imperial College London
London, United Kingdom
houyinghan521@outlook.comZongyou Yang
Department of Computer Science
University College London
London, United Kingdom
yzy0624@outlook.com
Abstract—To facilitate the transformation of legacy finite
difference implementations into the Devito environment, this
study develops an integrated AI agent framework. Retrieval-
Augmented Generation (RAG) and open-source Large Language
Models are combined through multi-stage iterative workflows
in the system’s hybrid LangGraph architecture. The agent con-
structs an extensive Devito knowledge graph through document
parsing, structure-aware segmentation, extraction of entity re-
lationships, and Leiden-based community detection. GraphRAG
optimisation enhances query performance across semantic com-
munities that include seismic wave simulation, computational
fluid dynamics, and performance tuning libraries. A reverse
engineering component derives three-level query strategies for
RAG retrieval through static analysis of Fortran source code.
To deliver precise contextual information for language model
guidance, the multi-stage retrieval pipeline performs parallel
searching, concept expansion, community-scale retrieval, and
semantic similarity analysis. Code synthesis is governed by
Pydantic-based constraints to guarantee structured outputs and
reliability. A comprehensive validation framework integrates
conventional static analysis with the G-Eval approach, cover-
ing execution correctness, structural soundness, mathematical
consistency, and API compliance. The overall agent workflow
is implemented on the LangGraph framework and adopts con-
current processing to support quality-based iterative refinement
and state-aware dynamic routing. The principal contribution
lies in the incorporation of feedback mechanisms motivated by
reinforcement learning, enabling a transition from static code
translation toward dynamic and adaptive analytical behavior.
Index Terms—AI agent, Retrieval-Augmented Generation
(RAG), GraphRAG, knowledge graph, Devito, Fortran, Lang-
Graph
I. INTRODUCTION
A. Background and problem description
High-performance scientific computing depends strongly on
large-scale legacy Fortran codebases that support essential
applications, including weather prediction, climate simulation,
nuclear safety analysis, and computational physics. Such codes
embody many decades of accumulated engineering knowledge
and correspond to huge investments. However, the transition
toward contemporary hardware platforms such as GPUs and
multi-core processors, together with rapidly evolving software
ecosystems, has made the effective reuse and integration
of legacy implementations progressively more challenging.
Meanwhile, the population of experienced Fortran developers
is steadily shrinking, as the language has largely vanishedfrom modern computer science education. Current strategies
for code modernisation exhibit notable shortcomings. Conven-
tional tools such as F2PY are mainly oriented toward interface
wrapping rather than deep restructuring. Fully manual refactor-
ing can preserve accuracy, yet the associated time and financial
costs are often unacceptable. Although large language models
demonstrate potential for automated code translation, current
general-purpose solutions lack a specialized focus on scientific
computing and quality validation. To overcome these issues,
this project presents an AI-based agent system. The proposed
system performs automated reverse engineering of legacy For-
tran finite-difference programs and converts them into Devito,
a contemporary domain-specific language designed for finite
difference computations.
B. GraphRAG
Retrieval augmented generation (RAG) enhances the capa-
bilities of large language models by incorporating external
knowledge drawn from databases or document collections
[16]. GraphRAG, introduced by Microsoft Research, expands
this paradigm through the integration of knowledge graph rep-
resentations [25]. In this approach, graphs are constructed in
which nodes correspond to entities or documents, while edges
encode the relationships that connect them [25]. Information
retrieval is carried out at the level of graph communities,
enabling the combination of localized evidence with an over-
arching structural context. Such a mechanism is especially
valuable when dealing with extensive and technically com-
plex corpora. In the present study, GraphRAG is particularly
appropriate because the Devito repository generates tens of
thousands of structured components along with dense net-
works of entity relationships. The framework accelerates query
execution through the application of community detection and
graph sparsification strategies, both of which are essential
for real-time retrieval scenarios. Moreover, GraphRAG also
keeps code-level dependencies and semantic relations. These
properties are well-matched to the task of reverse engineering
Fortran into Devito, where precision and structural fidelity are
critical [8], [15], [17], [30].arXiv:2601.18381v1  [cs.AI]  26 Jan 2026

Fig. 1. LangChain RAG from scratch pipeline overview.
C. Embeddings
Embedding models transform textual content or source code
into numerical vector representations that capture underlying
semantic information. This transformation enables similarity
computation and forms the foundation for effective retrieval
mechanisms. In this study, the BGE-M3 embedding model
is adopted [6]. This selection is motivated by its strong
results on the MTEB benchmark, which assesses embedding
approaches across a wide range of retrieval and clustering
scenarios [26], [29]. BGE-M3 supports extended input lengths
of up to 8,192 tokens and generates embeddings with 1024-
dimensions. Such features are essential in this context, as
technical documentation and code fragments are frequently
lengthy and structurally complex. The model is also efficient
in terms of size and speed, making it suitable for deployment
at large scales. BGE-M3 also provides multimodal capability,
allowing both natural language text and programming code to
be embedded within a unified vector space. This capability
facilitates the construction of a cohesive knowledge base that
unifies documentation, source code, and structured metadata.
Consequently, BGE-M3 is a robust foundation for knowledge
graph construction and supports dependable retrieval within
the Fortran to Devito translation framework.
D. K3-Trans
K3-Trans is an approach to facilitate the development of
structured knowledge bases for program translation tasks [33].
It focuses on three core components, including a library of tar-
get language examples, explicit mapping rules that link source
and target languages, and a repository containing validated
pairs of translated programs. Within this project, the under-
lying concepts of K3-Trans are employed to bridge legacy
Fortran finite difference codes with the Devito framework. The
example collection supplies Devito implementations covering
a range of numerical schemes and boundary condition settings.
The mapping rules formalize the transformation from Fortran
syntactic constructs to symbolic operators used in Devito.The set of translation pairs provides verified correspondences
between scientific formulations and their Devito implementa-
tions. Through this approach, the translation process attains a
high level of accuracy and reproducibility, which is essential
for scientific computing applications that demand reliable and
stable numerical performance.
II. METHODOLOGY ANDIMPLEMENTATION
A. System architecture overview
The system is organized around a five-layer hierarchical
architectural design. Each layer communicates through stan-
dardized interfaces to support coordinated interaction across
layers. The first layer is the Knowledge Base Construction
Layer, which handles documents in multiple formats. Its func-
tions include document parsing, structure-aware segmentation,
entity relationship extraction, and Leiden-based community
detection. The second layer is the Retrieval Enhancement
Layer, where multimodal parallel retrieval is implemented.
GraphRAG strategies are integrated with static analysis of For-
tran code. Four parallel retrieval mechanisms provide accurate
contextual information: community-level retrieval, full text
search, concept expansion, and semantic similarity matching.
The third layer is the Core Components Layer, which con-
tains the RAG Context Builder, the Fortran-Devito conversion
module, and the Quality Validation component. Structured
output is enforced through Pydantic-based constraints, while
code quality checks are supported through Ruff integration.
The fourth layer is the Decision Layer, which implements
dynamic routing based on the LangGraph framework across
eight workflow nodes. Quality-based adaptive decision-making
guides the conversion process. The fifth layer is the Agent
Coordination Layer, which supports parallel execution through
intelligent task queue management. System robustness is en-
sured through unified error handling strategies, while compu-
tational efficiency is optimized through automatic scaling of
agent resources. The architectural flowchart shown in Figure
2 illustrates the complete processing pipeline from Fortran
source input to Devito code generation, demonstrating data
exchange among the five architectural layers, interactions be-
tween components, and the corresponding decision pathways.
B. Knowledge base construction pipeline
1) Devito code repository analysis and preprocessing:The
Devito repository analysis and preprocessing phase performs
an extensive, structured examination of the original Devito
codebase. This stage establishes a unified data foundation
that supports subsequent document parsing and knowledge
extraction procedures. Based on the K3Trans framework
principles described in Section I-D, the system integrates
a triple knowledge enhancement strategy with the specific
demands of scientific computing. Through this process, a
dedicated knowledge base is developed to support Fortran-
Devito translation tasks. Meanwhile, the project constructs a
comprehensive library of validated translation pairs spanning
multiple levels of complexity. Figure 3 presents the statistical

Layer 5 — Multi-Agent
Layer 4 — Decision
Layer 3 — Core
Layer 2 — Retrieval
Layer 1 — Knowledge Base
Docs: .md / .ipynb / .pdf
Parsing (MinerU)
Structure Segmentation
ER Extraction
Leiden
Neo4j
Static Analysis
3-Layer Query
GraphRAG
Modes: Community / Full-T ext / Concept / 
Similarity
RAG Builder
Conv: Fortran→Devito (Pydantic+Ruf f)
Validator (+API)
LangGraph Router
Steps:
Init→Convert→V erify→Score→Route→Opt→Complete/Fail
Policy: >8.5 V erify; 5.5–8.5 Improve; <5.5
Re-convert
Completion
Task Q
Agent1 (convert)
 Agent2 (convert)
 AgentN (convert)
Pool
Monitor
Fortran Code Input
Devito Code Output
Fortran Code Input
guidesFig. 2. System architecture for Fortran-to-Devito pipeline.
characteristics of the Devito knowledge base derived according
to the K3Trans framework principles.
2) Multimodal document processing and MinerU integra-
tion:The multimodal document processing stage provides
structured parsing capabilities comparable to MinerU. It trans-
forms originally unstructured documents into structured meta-
data streams. This stage supports Markdown files (.md),
Jupyter Notebooks (.ipynb), PDF documents (.pdf), and
Python source code. During parsing, textual content, source
code segments, tables, and images are extracted while associ-ated structural metadata is retained. Distinct document formats
are handled through dedicated processing strategies:
•Python:Abstract syntax tree (AST) parsing is employed
to precisely identify program elements such as function
definitions, class declarations, and documentation strings,
thereby maintaining syntactic correctness.
•Jupyter Notebook:Explicit attention is given to preserv-
ing the semantic relationship between executable code
cells and documentation cells.
•Markdown:Heading hierarchies are automatically de-

Fig. 3. Devito knowledge base
tected, and complex structures including tables, lists, and
embedded code blocks are systematically parsed.
Each resulting knowledge block is organized around a section
heading and aggregates all associated text, tables, images, and
code elements contained within that section. Through this pro-
cess, 18,083 initially unstructured data items are transformed
into structured components. Every component is annotated
with metadata such as file location, content length, word count,
and directory category. The final structured metadata stream
consists of 10,287 function elements, 2,423 code blocks, 2,065
class definitions, 2,339 text paragraphs, 600 headings, 249
lists, and 83 tables.
3) Structure-aware semantic segmentation:The system ag-
gregates parsed document pieces into complete knowledge
chunks by applying different content categories. Documents
are segmented according to headers and sections, whereas
source code is divided based on function and class boundaries.
When individual chunks exceed an appropriate size, additional
subdivision is performed to ensure that each unit remains
within a length of 500 to 8,000 characters. Markdown files
follow hierarchical heading structures, while Python source
files are segmented in accordance with intrinsic code organiza-
tion patterns. Each segmentation strategy preserves contextual
continuity. Oversized content units are automatically divided
at semantically natural boundaries. This study avoids splitting
sentences or code functions and uses parent-child relationships
to link related blocks together. As a result of this procedure,
the system produces 6,295 knowledge blocks for inclusion in
the knowledge base. Each block is associated with metadata
describing the file location, a concise content summary, and
structural attributes required for subsequent processing.
4) Entity relationship extraction and graph construction:
This stage performs entity extraction across 6,295 knowledge
chunks and constructs the corresponding graph relationships.
Three complementary extraction strategies are applied. The
first approach relies on code analysis to detect Python classes,
functions, and variables through regular expression patterns.
The second approach employs domain dictionaries to recog-
nize and match concepts unique to Devito, including elements
such asGrid,TimeFunction, andOperator. The thirdapproach uses SpaCy to carry out general entity recognition
of technical terms and supplements.
The extractor creates five relationship types:
•MENTIONSlinks chunks to contained entities.
•CALLStracks function dependencies.
•INHERITSmaps class hierarchies.
•RELATED TOconnects similar content.
•PART OFshows chunk structure.
The procedure generates a total of 12,793 nodes and 62,362
relationships. Entities derived from code encompass key De-
vito classes and essential computational functions. Conceptual
entities represent topics such as finite difference schemes and
boundary condition types. This structured mapping provides a
foundational framework for subsequent community detection
and database integration by encoding both the structural and
semantic relationships inherent in the codebase.
5) Embedding, community detection and graph optimiza-
tion:The project employs BGE-M3 (BAAI/bge-m3) as
the primary embedding model to perform semantic similarity
analysis. Each node within the graph is represented by an
embedding vector generated by this model. In total, 32,416
nodes were processed, comprising 24,265 knowledge blocks
and 8,151 extracted entities. A similarity graph is constructed
from the embedding vectors and analyzed using the Leiden
algorithm. Multiple resolution settings (0.3, 0.5, 0.8, 1.0, 1.2)
are evaluated to identify the most meaningful community
structures. The algorithm detects 70 distinct semantic commu-
nities encompassing 6,230 nodes. Major communities include
symbolic mathematics (690 nodes), seismic wave modeling
(334 nodes), boundary conditions (223 nodes), and finite
difference methods (175 nodes). This detailed community
organization captures both the technical depth and the modular
design characteristic of the Devito framework.
Fig. 4. Community hierarchy by theme categories.
Figure 4 illustrates the hierarchical organization of semantic
communities identified using the Leiden algorithm. The system
automatically identifies multiple semantic clusters and orga-
nizes them into eight overarching technical domains based on
thematic similarity. For instance, spatial discretization encom-

passes grid and dimension handling, fluctuation modeling cov-
ers seismic and acoustic propagation methods, code generation
addresses operator compilation and optimization strategies,
and data structures include core type definitions. This multi-
level arrangement of knowledge enables efficient retrieval
across varying levels of detail. Graph optimization leverages
a Top-Knearest neighbor approach to reduce computational
overhead. Each node maintains links only to its eight most
similar neighbors, preventing the creation of an overly dense
network. This method condenses hundreds of thousands of
possible semantic relationships into 36,702 high-quality edges.
The resulting sparsification enhances query performance by a
factor of 540 for real-time retrieval tasks, while preserving
essential semantic connections.
6) Neo4j migration and index building:The final stage of
the pipeline involves migration to Neo4j. During this process,
schema constraints and indexes are established. Unique con-
straints ensure the integrity of each node, while performance
indexes are applied to community identifiers, chunk categories,
and thematic classifications E. To enhance import efficiency,
batch processing is utilized. Nodes are imported in groups
of 500 using parameterized queries, reducing both security
risks and memory consumption. Parallel processing further
accelerates the operation by simultaneously handling multiple
data types. Although executed only once, the full import
completes in 17 seconds. The resulting database supports high-
performance GraphRAG functionality, providing optimized
query paths and robust semantic search capabilities.
C. GraphRAG retrieval system design
1) Fortran static analysis and query generation:
When Fortran code is provided as input, the system’s
FortranCodeAnalyzerautomatically detects features
such as finite difference schemes, boundary condition types,
and time-stepping approaches, converting them into structured
query vectors. The analyzer extracts information on spatial
dimensions, PDE classifications, and numerical method sig-
natures, assigning confidence scores and complexity metrics
to each identified code element. The GraphRAG component
employs a three-layer fixed strategy mapping architecture.
Using results from static analysis of the Fortran source, the
system generates three categories of characteristic queries and
assigns them to predefined retrieval strategies:
•Primary queries (Comprehensive strategy):Core tech-
nical elements are mapped to a comprehensive retrieval
strategy, ensuring high-fidelity coverage of essential con-
version dependencies (e.g., “2D heat equation finite dif-
ference Devito implementation”).
•Secondary queries (Fast strategy):Queries targeting
auxiliary implementation details are routed to a fast
retrieval strategy, emphasizing quick response times (e.g.,
“boundary condition implementation” or “grid initializa-
tion patterns”).
•Concept queries (Deep strategy):Queries requiring
deeper conceptual insight are directed to a deep retrieval
strategy, enabling the identification of implicit knowledgerelationships (e.g., “mathematical equivalence verifica-
tion”).
2) Multi-modal parallel retrieval strategy:The process
initiates a multi-stage RAG retrieval pipeline that leverages
the generated query types alongside the knowledge graph built
in the preceding stage. Primary queries, following the com-
prehensive strategy, access information from the Neo4j graph
database using four parallel retrieval mechanisms. Figure 5
provides an overview of the full Retrieval-Augmented Gener-
ation (RAG) pipeline, summarizing its complete workflow.
The process begins with community-level retrieval: by com-
paring query keywords against the themes of 70 knowledge
communities, the system rapidly selects the 3-5 most relevant
communities, reducing the retrieval search space. Next, four
search modes are executed concurrently:
•Full-text search:Leveraging Neo4j’s pre-built full-text
index and automatic TF-IDF scoring, this mode performs
semantic matching within the selected communities.
•Precise community search:Using string containment
operations, this mode performs exact matching on knowl-
edge chunk titles and content, constrained to specific
communities via theBELONGS_TOrelationships.
•Concept expansion search:Starting from identified
concept nodes, related knowledge blocks are discovered
through theMENTIONSrelationship.
•Semantic similarity search:Pre-computed cosine sim-
ilarity scores (threshold 0.6) guide graph traversal, ex-
panding results along theSEMANTIC_SIMILARrela-
tionships.
Each mode returns candidate results along with their original
scores, providing multiple dimensions of evidence to support
subsequent fusion and re-ranking.
3) Result fusion and context construction:Once the paral-
lel retrieval phase completes the coarse ranking, the system
advances to fine-grained ranking through advanced result
merging and context assembly. The fusion process begins with
deduplication usingchunk_id, ensuring that each knowledge
chunk appears only once in the candidate set. Composite
scores are then recalculated using multi-factor weighting.
Result-type weighting prioritizes full-text search (1.0), fol-
lowed by community search (0.9), concept expansion (0.8),
and semantic similarity (0.7). Community relevance weighting
adjusts scores slightly to prevent dominance by large commu-
nities, while content-length weighting favors chunks of suffi-
cient size, assigning full weight to items of 1,000 characters.
At the query-level, hierarchical weighted re-ranking is applied:
primary queries carry a weight of 1.0, secondary queries 0.7,
and concept queries 0.5. An additional “analysis relevance”
factor dynamically adjusts scores between approximately 1.0
and 2.0, based on how well the features extracted from the
Fortran code, such as equation types, numerical patterns,
boundary conditions, and underlying mathematical concepts,
align with the retrieved knowledge chunks.
The final phase generates a structured RAG context using
layered prompt engineering. The prompt frames the LLM

Fortran Code Input
Static Analysis & Q uery 
Gener ation
Community-le vel Retrieval 
- (Pre-scr eening )
Select 3-5 R elevant 
Communit ies fr om 70
Multi-mode P arallel 
Retrieval - ( Coarse Ranking )
Full-te xt Search
Within R elevant 
Communit ies
Community-le vel Ex act 
Match
BELONGS _TO Relation
Concep t Exp ansion
MENT IONS R elation
Seman tic Simil arity
SEMANT IC_SIMILA R Relation
Result Fusion & 
Dedupl ication
Based on chunk _id
Re-ranking ( Fine Ranking )
Query T ype W eight
primary:1.0, secondary:0.7, 
concep t:0.5
Result Type W eight
fulltext:1.0, 
community:0.9, concep t:0.8
Community R elevance 
Weight
(Down-w eight Lar ge 
Communit ies)
Content Length W eight
Analysis R elevance W eight
Based on F ortran Feature 
Matching
Final Ranking R esults
Top-15 Kno wledg e Chunk s
RAG Con text Cons truct ion
Structur ed Pr ompt 
EngineeringFig. 5. Retrieval-Augmented Generation (RAG) pipeline
as a “Fortran→Devito conversion expert,” specifying explicit
objectives and quality standards, while the retrieved knowledge
serves as the contextual knowledge base. Research indicates
that structured prompt design enhances reasoning reliability,
particularly when prompts incorporate role definition, con-
straints, and hierarchical task decomposition, as outlined in
the Prompt Canvas framework [2] D. To further improve
conversion fidelity, the workflow incorporates a self-check and
revision loop inspired by Chain-of-Thought (CoT) prompt-
ing. Studies have shown that CoT-based prompt engineer-
ing increases factual accuracy and reasoning depth in struc-
tured tasks [14]. This mechanism enables the model to vali-
date outputs against strict contracts, including Pydantic/JSONstructures containing Devito code, rationale, mappings, and
confidence scores, and iteratively refine them if requirements
are unmet. Ultimately, the system selects the top 15 highest-
scoring candidates, ensuring that only high-quality outputs
guide the Devito conversion process.
D. Agent workflow design
1) Concurrent workflow processing design:During project
development, a major bottleneck became apparent: sequential
LLM API calls significantly reduced processing efficiency.
Each code modification required 3-5 minutes for validation,
which also impaired the responsiveness of interactive work-
flows. Testing of thechat.eseLLM API revealed full

support for concurrent requests within the defined rate lim-
its. Leveraging this capability, a concurrent task processing
architecture was designed, enabling parallel execution of large-
scale Fortran-to-Devito conversion tasks. The system balances
concurrency control and resource management while remain-
ing compliant with API constraints, dramatically improving
throughput and responsiveness.
2) I/O concurrency implementation:The system employs
Python’s asyncio framework to enable true I/O concurrency
during interactions with the LLM API. Unlike traditional
synchronous execution, where CPU resources remain idle
while awaiting API responses, the asyncio event loop allows
multiple tasks to progress simultaneously. When one workflow
engine is paused waiting for an LLM response, the event
loop immediately switches to other pending tasks, maximizing
resource utilization in I/O-bound operations. Recent studies,
such as [5], have shown that asynchronous querying can
substantially increase throughput when interfacing with LLM
endpoints, and [12] highlights the benefits of non-blocking
execution for complex LLM tasks. These findings confirm that
asyncio-based concurrency is highly effective for large-scale
LLM query processing and asynchronous function manage-
ment.
3) Task-level architecture:The architecture adopts separate
workflow agent instances assigned to individual conversion
tasks. Each processing engine contains a fully self-sufficient
pipeline. This design ensures complete isolation during task
execution and integrates all required functional components,
including source code transformation, output quality valida-
tion, and optimized layout generation.
4) Concurrency control:The system supports scalable
execution ranging from two to eight concurrent large lan-
guage model requests, achieving significant performance gains
through the use of asyncio-based semaphore scheduling. Em-
pirical evaluation using a representative benchmark dataset
demonstrates notable acceleration. Relative to sequential ex-
ecution, the configuration with two agents delivers a 2.15×
throughput improvement, increasing productivity from 22.8
to 49.0 files per hour. Expanding to four agents further
raises throughput to 131.2 files per hour, corresponding to a
5.75×speed increase. The observed 5.75 times improvement
in the four-agent scenario exceeds the nominal theoretical
upper bound of four times. A parallel efficiency of 143.7%
highlights the substantial benefits derived from overlapping
I/O operations during concurrent execution.
E. Quality-driven iterative optimization
1) Iterative optimization method based on LangGraph:
This work develops a quality-based iterative optimization
framework motivated by principles from reinforcement learn-
ing. Continuous enhancement of code conversion performance
is achieved through decision routing in LangGraph combined
with adaptive threshold calibration mechanisms. This section
defines navigation criteria derived from quantitative quality
evaluation scores. Three decision thresholds are specified.An excellent level with a score of 8.5 identifies conver-
sion outputs that satisfy verification requirements and can
be released directly. An acceptable level with a score of
5.5 activates focused refinement procedures. A minimum
level of 3 is established as the lowest permissible quality
boundary. Overall, the framework incorporates a four-path
adaptive decision mechanism. High-quality outputs follow a
direct pass route to maximize execution efficiency. Medium
quality outputs undergo targeted refinement, where customized
optimization guidance is generated according to identified
defect categories. Low-quality outputs trigger a reconversion
process, involving adjustments to conversion logic and pro-
cessing strategies. When the predefined upper limit on large
language model responses is reached, an exit procedure is
applied, and the final handling strategy is selected based on the
current quality assessment. The introductory tutorial provided
by LangChain demonstrates the use of theget_graph()
method for graph based visualization [19]. In this study, the
get_graph().draw_mermaid_png()functionality of
LangGraph is employed to generate workflow diagrams. State
transition relationships are constructed usingStateGraph
[11] and are automatically rendered as intuitive flowchart
representations.
retry
accep t
start
initialize
convert
validate
scor e
route
refine
 fail
 final ize
end
Fig. 6. LangGraph-Based optimization loop.
As illustrated in Figure 6, the project implements an iterative
optimization procedure founded on the LangGraph framework,
which follows a continuous enhancement loop. Each Fortran
source file is processed through multiple optimization cycles.
The workflow begins with an analysis of structural character-
istics within the source code, followed by an initial conversion
stage. The resulting output is then evaluated across multiple
criteria by a dedicated quality verification module. Based on

the resulting quality scores, the routing component determines
the subsequent processing path [1].
2) Pydantic structured output constraints:Through the
strict type enforcement capabilities provided by Pydantic,
large language models are constrained to produce structured
JSON outputs that strictly adhere to a predefined schema.
Each conversion output is required to contain complete Devito
code, detailed explanations of conversion decisions, explicit
component mapping relationships, confidence evaluations, and
related metadata. This approach converts inherently unstruc-
tured model-generated text into a representation characterized
by high structural fidelity and semantic coherence. As a result,
the frequent formatting irregularities observed in conventional
large language model-based code generation are effectively
mitigated [18], [20]. The corresponding schema definition is
presented in C.
3) Ruff code quality optimization:Code formatting func-
tionality provided by Ruff is incorporated into the Agent layer
of the system. During the final stage of the processing work-
flow, Ruff is automatically executed to format the generated
Devito source code. This step ensures compliance with Python
PEP 8 conventions and widely accepted coding practices. The
applied Ruff optimizations primarily focus on standardized
code layout and systematic import arrangement, thereby im-
proving both the readability and structural consistency of the
produced code [31].
III. EVALUATION AND ANALYSIS
A. Retrieval quality assessment
1) Experimental design and methodology:To assess
retrieval effectiveness within a knowledge graph-based
GraphRAG framework applied to Fortran to Devito code trans-
formation tasks, a standardized retrieval quality evaluation pro-
tocol is constructed. The evaluation employs eleven benchmark
queries designed to represent common information retrieval
demands arising during the code conversion process. The
benchmark queries are grouped into three tiers of increasing
complexity.
1) Basic pattern level queries, including concepts such as
“finite difference methods” and “boundary conditions”,
are used to examine retrieval correctness for founda-
tional numerical computing knowledge.
2) Intermediate complexity queries, such as “Devito time
stepping implementation” and “2D heat equation solver
optimization”, aim to capture combined technical re-
quirements involving multiple concepts.
3) Advanced technical queries, including “second-order
spatial discretization stencil optimization” and “Devito
compiler backend performance tuning”, are intended to
evaluate the system’s ability to retrieve highly special-
ized and expert-level information.
These test cases reflect the progressive increase in complex-
ity typically encountered in real-world code modernization
pipelines [9]. Each query is paired with predefined groundtruth references, consisting of expected topic categories, rel-
evant keyword collections, and anticipated output types. Re-
trieval performance is evaluated using three widely adopted
information retrieval metrics.
1)Precision@5quantifies the relevance accuracy of the top
five retrieved items by computing the ratio of relevant
results to the total number of returned entries [24].
2)Recall@5measures the proportion of ground truth rel-
evant items successfully retrieved, reflecting the com-
pleteness of the retrieval process [28].
3)MRR Mean Reciprocal Rankassesses the ranking
position of the first relevant result, emphasizing the
capability to prioritize the most pertinent information
[32].
2) Experimental results and analysis:The GraphRAG sys-
tem demonstrates strong performance across all evaluated
indicators, as reported in Table I. The evaluation yields a
Precision@5 value of 0.964 and a Recall@5 value of 0.930.
These results indicate that the integration of knowledge graph
community discovery, multi-path retrieval mechanisms, and
semantic expansion strategies is effective for improving re-
trieval quality. In parallel, the mean query response time
remains exceptionally low at 0.012 seconds. This efficiency is
attributed to optimized indexing provided by Neo4j, combined
with graph sparsification techniques and concurrent query
execution. Collectively, these findings demonstrate that the
system delivers both high-accuracy information retrieval and
real-time responsiveness.
TABLE I
GRAPHRAGSYSTEM RETRIEVAL QUALITY EVALUATION RESULTS.
Metric Value
Precision@5 0.964
Recall@5 0.930
MRR 1.000
Avg. Response Time 0.012s
The reported MRR values “look too good”. This outcome
arises because each query is associated with asetof valid
reference answers rather than a single unique target. When the
top-ranked retrieval result belongs to the predefined ground
truth set, MRR=1. Although this behavior reflects strong
ranking consistency, it also limits the ability of MRR to
differentiate retrieval quality under the current evaluation
configuration.
3) Multi-modal retrieval strategy analysis:This experiment
examines the three retrieval modes implemented within the
GraphRAG system.
1)Fast Strategy: This mode prioritizes response latency
reduction and relies exclusively on direct matching re-
trieval supported by full-text indexing in Neo4j. The
technical realization includes straightforward full-text
index queries, basic keyword matching operations, and
simplified result ranking procedures.
2)Comprehensive Strategy: This mode deploys a com-
plete four-stage retrieval pipeline. The process begins

with community level search to restrict the candidate
scope. It then performs parallel full-text retrieval and
precise intra-community search. Subsequently, concept
expansion is applied, followed by semantic similarity-
based retrieval to refine the final results.
3)Deep Strategy: This mode extends the Comprehensive
approach by introducing an additional semantic expan-
sion phase. Deeper relational knowledge is uncovered
through two-hop connections within the graph struc-
ture. The implementation leverages path-based queries
supported by Neo4j using the Cypher query language,
enabling exploration of indirect semantic associations.
MATCH(seed:Chunk)−[:SEMANTIC SIMILAR *2]−(
distant:Chunk)
This methodology is enabled by the efficient path-based
query capabilities of the Cypher language for graph
database retrieval tasks [30].
Each retrieval strategy independently processed an identical
set of ten benchmark queries, spanning multiple levels of
complexity and diverse query categories, while employing a
unified set of evaluation criteria to ensure comparability. The
experimental design applies a multidimensional assessment
framework comprising five evaluation dimensions. A detailed
comparative analysis of the performance achieved by the four
retrieval strategies is presented in Table II.
The Fast strategy excelled in response speed, achieving the
shortest average query time of 0.002 seconds, but exhibited a
relatively lower recall of 0.625. In contrast, the Comprehensive
strategy achieved an exceptional precision of 0.980 and a
recall of 0.668, confirming the effectiveness of its multi-stage
retrieval pipeline. Overall, this configuration provided the most
favorable compromise between retrieval efficiency and output
quality.
B. Code translation quality assessment
1) API consistency and error mitigation:During the ini-
tial deployment of the system, large language models oc-
casionally produced outdated or non-existent APIs, such as
u.dx.backwardandu.data.initialize(). This be-
havior arises from the presence of early version examples and
outdated Devito parameters in LLM training datasets. Addi-
tional errors were introduced through analogies with similar
frameworks and flawed reasoning patterns. Such occurrences
align with prior observations that LLMs frequently hallucinate
or misuse APIs [3], [7].
To address this issue, a version locked retrieval augmented
generation framework was implemented, governed by rule-
based controls and informed by studies on LLM code re-
liability [13]. The underlying knowledge base contains only
verified Devito examples [10], ensuring that retrieved outputs
are executable. The system enforces explicit denylist and
allowlist rules, along with equivalent substitution mappings.
Additional safeguards include Preflight structural verification
to ensure assignable left-hand values, consistent equation
dimensions, and valid subdomains and subdimensions; Pythonsyntax validation through abstract syntax tree parsing; and API
compliance linting [22], [23]. To demonstrate the improvement
in code correctness provided by the RAG framework compared
to native LLM outputs, several representative error cases have
been compiled in Appendix A. These include instances of fake
APIs and “pretend” Devito but calculate with NumPy. (see A).
2) Multi-dimensional quality validation:A critical com-
ponent for ensuring successful translation from Fortran to
Devito is the assessment of code conversion quality. This study
establishes a quality verification framework that integrates
the G-Eval evaluation methodology with conventional static
analysis techniques. The GPT OSS 120B model was employed
as the large language model evaluator within the open-source
environment. The static quality assessment model in this
framework evaluates code along five key dimensions:
•Execution success: evaluates whether the converted code
can run without errors, accounting for runtime exception
handling, completeness of imports, and syntactic correct-
ness.
•Structural integrity: assesses the presence and correct-
ness of Devito core components, including Grid defini-
tions, Function instantiations, Equation formulations, and
Operator construction.
•API compliance: measures adherence to Devito API
usage best practices through pattern recognition, param-
eter configuration validation, and verification of API call
structures.
•Parameter consistency: examines the accurate transla-
tion of time, grid, and physical parameters, ensuring
boundary conditions, dimension alignment, and numer-
ical consistency are maintained.
•Conversion fidelity: determines the precision of translat-
ing mathematical models and physical equations, includ-
ing correct equation type mapping, preservation of differ-
ential operators, and maintenance of physical semantics
to guarantee the correctness of scientific computations.
The comprehensive quality score is calculated using
weighted averaging:
Quality Score=X
(w i×s i)(1)
In this expression.w iis the weight of thei-th dimension,
ands irepresents the standardized score of thei-th dimension.
G-Eval is an automated evaluation framework that leverages
large language models to assess generated content through
structured prompt design and sequential reasoning. Following
OpenAI’s official RAG system evaluation guidelines, the GPT
OSS 120B open-source model was employed as the evaluation
engine to complement the traditional five-dimensional static
analysis framework [21], [27]. This approach enables the
handling of complex assessment tasks that are difficult to
quantify using conventional metrics, including code logic,
algorithmic elegance, and adherence of the code to the original
specifications. Carefully designed instructions and illustrative
examples guide the large language model to evaluate outputs
according to predefined criteria. To ensure the objectivity of

TABLE II
MULTI-MODAL RETRIEVAL STRATEGY PERFORMANCE COMPARISON ANALYSIS.
Strategy Name Avg Response Time (s) Avg Precision@5 Avg Recall@5 Avg Diversity Success Rate
Fast 0.002 0.960 0.625 0.700 1.000
Comprehensive 0.017 0.980 0.668 0.595 1.000
Deep 0.018 0.980 0.668 0.595 1.000
Hybrid 0.013 0.980 0.654 0.635 1.000
the evaluation, a model distinct from the one used for code
generation was applied, following established practices in the
LLM-as-a-Judge domain. Prior research indicates that using
the same model for both generation and evaluation introduces
substantial self-enhancement bias, with models tending to
assign inflated scores to their own outputs. To mitigate this
effect, the GPT OSS 120B model functions solely as a separate
evaluation engine, independent of the code conversion model.
The evaluation procedure applies multi-criteria scoring across
four dimensions: execution success (30%, code runs without
errors), code structure (25%, Devito core components com-
pleteness), mathematical logic (25%, solution validity within
the same PDE framework), and API usage (20%, appropri-
ateness of Devito API calls). Both the original Fortran code
and the converted Devito code are provided as joint context
input. To eliminate variability caused by stochastic outputs and
ensure reproducibility, the model’s temperature is set to zero
[4]. Additionally, the model incorporates an implicit chain-of-
thought reasoning process for internal step-by-step evaluation
and weighting. The output consists solely of a final score
accompanied by a brief justification.
The overall score is calculated considering both the LLM
and the traditional static method:
Final Score = Traditional Score×(1−λ)+LLM Score×λ, λ= 0.5
(2)
Table III presents the evaluation results for 13 Devito
finite difference test cases covering varying difficulty levels
and application domains. The assessment indicates strong
system performance, with a Grade-A success rate of 76.9%.
All conversions achieved perfect scores of 1.00 in the three
critical dimensions: execution, structural integrity, and API
compliance. Furthermore, the scores generated by the static
assessment framework show a positive correlation with the G-
Eval model judgments, demonstrating convergent validity and
supporting the reliability of the proposed evaluation method-
ology.
The result is illustrated by the radar chart provided in B.
IV. DISCUSSION
A. Current limitations
Although the system is capable of collecting and
storing comprehensive conversion history data, its
capabilities for advanced data mining and pattern
discovery remain limited. Additionally, the existing
quality evaluation framework relies on fixed, predefinedthresholds, such asexcellent_threshold=0.85and
acceptable_threshold=0.55. These criteria are not
yet capable of automatic adjustment based on code type,
complexity, or historical performance. Consequently, the
system’s adaptability across diverse conversion scenarios is
constrained by this static configuration.
B. Future research directions
Future developments will extend beyond using a single
open-source large language model as the evaluation engine.
Specialized, compact evaluator models will be trained or fine-
tuned on historical conversion traces, while reinforcement
learning techniques will dynamically adjust routing policies
and quality thresholds, enabling a self-improving agent. An
agent-level memory system will also be introduced, storing
episodic logs of previous conversions and semantic summaries
of recurring error patterns. This memory allows the retrieval of
prior fixes and the reuse of successful strategies across tasks.
By integrating adaptive learning, this approach enhances both
the reliability and efficiency of code modernization workflows.
V. CONCLUSION
This study presents an AI agent system designed to translate
legacy Fortran finite difference code into the Devito frame-
work. The system integrates Retrieval-Augmented Generation
with open-source large language models within a multi-stage
iterative workflow structured on the LangGraph architecture.
The agent utilizes a Devito knowledge graph implemented
in Neo4j, incorporates GraphRAG for retrieval, and applies
quality-driven iterative refinement with enforced structured
output constraints. Conversion quality is ensured through a
validation framework that combines static code analysis with
LLM-based evaluation. The resulting framework provides an
agent-driven solution for automated legacy code translation in
scientific computing, achieving a balance between accuracy
and computational efficiency.
APPENDIXA
RAG-GOVERNEDDEVITO VS. VANILLALLM
We have listed 3 categories of high-frequency errors:
•Hallucinated or mis-ported Devito APIs (e.g.,
u.dx.backward).
•Devito integration in name only (imports Devito but
performs updates/loops in NumPy).
•Boundary-condition misuse (e.g., post-hoc patching or
bc=onOperator).

TABLE III
UNIFIED QUALITY VALIDATION RESULTS PER CASE.
Case Final Grade Confidence Duration (s) Execution Structure API Parameters Conv. Fidelity LLM Judge
acoustic_wave_2d0.941 A 0.750 36.090 1.000 1.000 1.000 1.000 0.820 0.900
advection_simple0.892 A 0.900 33.290 1.000 1.000 1.000 1.000 0.840 0.800
advection_upwind0.780 B 0.750 36.410 1.000 1.000 1.000 0.950 0.640 0.600
crank_nicolson_heat0.930 A 0.750 15.220 1.000 1.000 1.000 1.000 0.600 0.900
diffusion_3d0.883 A 0.750 9.160 1.000 1.000 1.000 0.950 0.700 0.800
heat_1d_simple0.842 A 0.900 11.150 1.000 1.000 1.000 1.000 0.840 0.700
heat_equation_2d0.842 A 0.950 15.860 1.000 1.000 1.000 1.000 0.840 0.700
laplace_solver0.877 A 0.750 9.200 1.000 1.000 1.000 0.800 0.740 0.800
legacy_advection0.886 A 0.900 10.410 1.000 1.000 1.000 0.750 0.960 0.800
poisson_jacobi0.827 A 0.750 8.610 1.000 1.000 1.000 0.800 0.740 0.700
poisson_simple0.877 A 0.750 8.570 1.000 1.000 1.000 0.800 0.740 0.800
wave_1d_simple0.729 B 0.750 15.300 1.000 1.000 1.000 0.900 0.680 0.500
wave_equation_1d0.729 B 0.750 15.030 1.000 1.000 1.000 0.900 0.680 0.500
A.1 Fabricated Derivative API (u.dx.backward)
Tag:Treating derivatives as chained attributes; misinterpret-
ing central difference as upwind.
Incorrect (minimal example):
fromdevitoimportGrid, TimeFunction, Eq, solve, Operator
, Constant
nx, Lx = 200, 1.0
grid = Grid(shape=(nx,), extent=(Lx,))
x, = grid.dimensions
u = TimeFunction(name=’u’, grid=grid, time order=1,
space order=1)
c = Constant(name=’c’, value=1.0)
# Fabricated API: u.dx.backward (not supported in Devito)
eq = Eq(u.dt, −c *u.dx.backward)
stencil = solve(eq, u.forward)
op = Operator(Eq(u.forward, stencil))
Correct:
fromdevitoimportGrid, TimeFunction, Eq, solve, Operator
, Constant, first derivative
nx, Lx = 200, 1.0
grid = Grid(shape=(nx,), extent=(Lx,))
x, = grid.dimensions
u = TimeFunction(name=’u’, grid=grid, time order=1,
space order=1)
c = Constant(name=’c’, value=1.0)
# First−order upwind: explicit side (c>0 −>left; c<0 −>
right)
dudx = first derivative(u, dim=x, side=’left’)
eq = Eq(u.dt, −c *dudx)
stencil = solve(eq, u.forward)
op = Operator([Eq(u.forward, stencil)])
A.2 Pseudo Integration: Devito Wrapper, NumPy Core
Tag:Devito package imported but not used; all updates
performed in NumPy loops; boundary conditions patched afterexecution.
Incorrect (pseudo integration, Devito unused in core):
fromdevitoimportGrid, TimeFunction, Eq, Operator#
imported but never used
importnumpyasnp
nx, Lx = 100, 1.0
dx = Lx / nx
dt, c = 0.005, 1.0
# Allocate manual NumPy array (ignores Devito
TimeFunction)
u = np.zeros((501, nx), dtype=np.float32)
xcoords = np.linspace(0.0, Lx − dx, nx, dtype=np.float32)
u[0, (xcoords>0.2) & (xcoords<0.4)] = 1.0
# Manual time stepping loop
fortin range(500):
unext = u[t].copy()
foriin range(1, nx):
dudx = (u[t, i] − u[t, i−1]) / dx
unext[i] = u[t, i] − dt *c*dudx
# Incorrect periodic boundary: patched after update
unext[0] = u next[−1]
u[t+1] = u next
Correct (upwind + periodic BC encoded symbolically in
Devito):
fromdevitoimportGrid, TimeFunction, Eq, Operator,
Constant, first derivative
importnumpyasnp
nx, Lx = 100, 1.0
dx = Lx / nx
dtval, c val = 0.005, 1.0
grid = Grid(shape=(nx,), extent=(Lx,))
x, = grid.dimensions
u = TimeFunction(name=’u’, grid=grid, time order=1,
space order=1)

c = Constant(name=’c’, value=c val)
xcoords = np.linspace(0.0, Lx − dx, nx, dtype=np.float32)
u.data[0, :] = ((xcoords>0.2) & (xcoords<0.4)).astype(np
.float32)
dudx = first derivative(u, dim=x, side=’left’)# c>0 −>
left
core = Eq(u.forward, u − dt val*c*dudx)
# Periodic BC: endpoints equal
leftbc = Eq(u.forward.subs({x: x.symbolic min}), u.forward
.subs({x: x.symbolic max}))
right bc = Eq(u.forward.subs({x: x.symbolic max}), u.
forward.subs({x: x.symbolic min}))
op = Operator([core, left bc, right bc])
op.apply(time M=499)
A.3 Boundary Condition Misuse + Variable/Index Errors
Tag:Using bc= as Operator arg; SubDomain misuse; 1-
based indexing (Fortran habit); variable name typos.
Incorrect (minimal example):
fromdevitoimport *
importnumpyasnp
grid = Grid(shape=(101,), extent=(1.0,))
u = TimeFunction(name=’u’, grid=grid, space order=1)
dt, c, nx, nsteps = 1e−4, 1.0, 100, 500
# 1−based indexing + variable name typo (nu data not
defined)
udata = np.zeros((nx + 1), dtype=np.float64)
foriin range(1, nx + 1):
x = (i − 1) *0.01
if0.2<= x<= 0.4:
nudata[i] = 1.0
u.data[0][1:] = nu data# typo + offset assignment
# Operator(bc=...) not valid API; SubDomain one−liner
incorrect
bcs = [SubDomain(’x==0’,{’u’: 0}), SubDomain(’x==1’,{’
u’: 0})]
eq = Eq(u.dt, −c *u.dx)# also uses central diff instead of
upwind
op = Operator(eq, bc=bcs)# invalid/legacy API usage
op.apply(time M=nsteps−1, dt=dt)
Correct (minimal fix: 0-based indexing; BC as equa-
tions):
fromdevitoimportGrid, TimeFunction, Eq, Operator,
Constant, first derivative
importnumpyasnp
nx, Lx = 100, 1.0grid = Grid(shape=(nx,), extent=(Lx,))
x, = grid.dimensions
u = TimeFunction(name=’u’, grid=grid, time order=1,
space order=1)
initial = np.zeros(nx, dtype=np.float64)
xs = np.arange(nx) *(Lx / nx)
initial[(xs>= 0.2) & (xs<= 0.4)] = 1.0
u.data[0, :] = initial
dtval, c = 1e−4, 1.0
core = Eq(u.forward, u − dt val*c*u.dx)# use
first derivative(...) for upwind
leftbc = Eq(u.forward.subs({x: x.symbolic min}), 0.0)
right bc = Eq(u.forward.subs({x: x.symbolic max}), 0.0)
op = Operator([core, left bc, right bc])
op.apply(time M=499)
APPENDIXB
RADAR CHART
ExecutionStructure API
Parameters
Conv. Fid. LLM0.20.40.60.81.0Cases
AW2D
ADV-S
ADV-U
CN-Heat
DIFF-3D
HEAT1D
HEAT2D
LAPL
LEG-ADV
POIS-J
POIS-S
WAVE1D-S
WAVE1D-E
Fig. 7. Radar chart of evaluation results across 13 test cases
APPENDIXC
PYDANTIC SCHEMA
Pydantic schema for structured Fortran-to-Devito out-
put:
Schema FortranToDevitoConversion:
devito code: string
conversion summary: string
key decisions:listof{decision type, rationale}
devito components:listof{component, purpose}
equation type:{parabolic, hyperbolic, elliptic}
spatial dimensions: integer (1−3)
time dependent: boolean
conversion confidence:float(0.0−1.0)
validation:{execution success, structure, api compliance,
parameters, fidelity}
usage notes:listof strings
optimization hints:listof strings

APPENDIXD
PROMPT PSEUDO-CODE
Prompt construction logic (simplified pseudo-code):
Function BuildConversionPrompt(fortran code, rag results):
Extract analysis info: equation type, dimensions, complexity
Extract reference examplesfromrag results
Initialize prompt parts
Append task definitionandconversion requirements
Append standard Devito workflow pattern
If analysis info exists:
Append problemtype, dimensions, complexity
If reference examples exist:
Append retrieved examples
Append source Fortran code
Append implementation guidelines
Append required JSON outputformat(schema)
Return full structured prompt
APPENDIXE
EXAMPLECYPHERQUERIES
Here is a list of several basic Cypher queries for Neo4j.
MATCH(n)
RETURN labels(n)[0]ASNodeType,count( *)AS Count
ORDER BY Count DESC
This query counts the number of nodes by type, providing
a global overview of the database composition.
MATCH(com:Community)
RETURN com.id, com.theme, com.size
ORDER BYcom.size DESC
LIMIT10
This query lists the ten largest semantic communities.
MATCH(c:Chunk)−[:BELONGS TO]−>(com:Community)
RETURN com.theme,count(c)ASmember count
ORDER BYmember countDESC
LIMIT10
This query verifies the assignment of knowledge chunks
to communities by counting their membership distribution.
Figure 8 shows the top communities and their sizes returned
by this query.
Fig. 8. cypher example
CALL db.index.fulltext.queryNodes(”chunk content index”,
”finite difference”)
YIELD node, score
RETURN node.title, score
LIMIT5
This query performs a semantic search using the full-text
index, retrieving chunks most relevant to the keywordfinite
difference.
REFERENCES
[1] Adarishanmukh. Langraph conditional workflow. Online, 2025.
Medium. Available from: https://medium.com/@adarishanmukh15501/
langraph-c61c8fcaac8f [Accessed 18 August 2025].
[2] Xavier Amatriain. Prompt design and engineering: Introduction and
advanced methods.arXiv preprint arXiv:2401.14423, 2024. Submitted
on 24 January 2024; revised 5 May 2024.
[3] Saleema Amershi, Andrew Begel, Christian Bird, Robert DeLine, Harald
Gall, Ece Kamar, Nachiappan Nagappan, Besmira Nushi, and Thomas
Zimmermann. Software engineering for machine learning: a case study.
InProceedings of the 41st IEEE/ACM International Conference on
Software Engineering – Software Engineering in Practice (ICSE-SEIP
’19), pages 291–300. IEEE/ACM, 2019.
[4] Robert E. Blackwell, Jon Barry, and Anthony G. Cohn. Towards
reproducible llm evaluation: Quantifying uncertainty in llm benchmark
scores.arXiv preprint arXiv:2410.03492, 2024. [Accessed 28 August
2025].
[5] Ryan Sze-Yin Chan, Federico Nanni, Angus R. Williams, Edwin
Brown, Liam Burke-Moore, Ed Chapman, Kate Onslow, Tvesha Sippy,
Jonathan Bright, and Evelina Gabasova. Prompto: An open source
library for asynchronous querying of llm endpoints.arXiv preprint
arXiv:2408.11847, 2024. Submitted on 12 August 2024; revised 16
December 2024.
[6] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and
Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality,
multi-granularity text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216, 2024.
[7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke
Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz
Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Fe-
lipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis,
Elizabeth Barnes, Ariel Herbert-V oss, William Hebgen Guss, Alex
Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir
Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N.
Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec
Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer,
Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya
Sutskever, and Wojciech Zaremba. Evaluating large language models
trained on code.arXiv preprint arXiv:2107.03374, 2021. [Accessed 28
August 2025].

[8] W. Bruce Croft, Donald Metzler, and Trevor Strohman.Search Engines:
Information Retrieval in Practice. Addison-Wesley, 2010.
[9] W. Bruce Croft, Donald Metzler, and Trevor Strohman.Search engines:
information retrieval in practice. Addison-Wesley, Boston, MA, 2010.
[10] Devito Project. Devito tutorials. [Online], 2025. [Accessed: 23 July
2025].
[11] J. Exson. Langgraph visualization with get graph. Online, 2025.
Medium. Available from: https://medium.com/@josephamyexson/
langgraph-visualization-with-get-graph-ffa45366d6cb [Accessed 18
August 2025].
[12] In Gim, Seung seob Lee, and Lin Zhong. Asynchronous llm function
calling.arXiv preprint arXiv:2412.07017, 2024. Published on 9
December 2024.
[13] Jingxuan He and Martin Vechev. Large language models for code:
security hardening and adversarial testing. InProceedings of the 2023
ACM SIGSAC Conference on Computer and Communications Security
(CCS ’23), November 2023, Copenhagen, Denmark, pages 1865–1879,
New York, NY , 2023. ACM.
[14] Michael Hewing and Vincent Leinhos. The prompt canvas: A literature-
based practitioner guide for creating effective prompts in large language
models.arXiv preprint arXiv:2412.05127, 2024. Submitted on 6
December 2024.
[15] HKUDS/RAG-Anything Developers. Rag-anything: All-in-one multi-
modal rag system. Online, 2025. Available from: https://github.com/
HKUDS/RAG-Anything [Accessed 18 August 2025].
[16] Marie-Anne Lachaux, Baptiste Roziere, Erwan Chan, et al. Eval-
uating large language models trained on code.arXiv preprint
arXiv:2301.12507, 2023. [Accessed 28 August 2025].
[17] LangChain. Rag from scratch: Part 1 (overview). YouTube video, 2023.
Accessed: 28 August 2025.
[18] LangChain Documentation. Structured outputs. Online, 2025.
LangChain Documentation. Available from: https://python.langchain.
com/docs/concepts/structured outputs/ [Accessed 18 August 2025].
[19] LangGraph Project. Langgraph graphs reference. Online,
2025. Available from: https://langchain-ai.github.io/langgraph/reference/
graphs/ [Accessed 18 August 2025].
[20] Michael Xieyang Liu, Frederick Liu, Alexander J. Fiannaca, Terry Koo,
Lucas Dixon, Michael Terry, and Carrie J. Cai. “we need structured
output”: Towards user-centered constraints on large language model
output. InExtended Abstracts of the CHI Conference on Human Factors
in Computing Systems (CHI EA ’24), 11–16 May 2024, Honolulu, HI,
USA. ACM, 2024.
[21] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and
Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human
alignment. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, pages 2511–2522, Singapore, 2023.
Association for Computational Linguistics.
[22] Mathias Louboutin, Michael Lange, Fabio Luporini, Navjot Kukreja,
Philipp A. Witte, Felix J. Herrmann, Paulius Velesko, and Gerard J.
Gorman. Devito (v3.1.0): an embedded domain-specific language for
finite differences and geophysical exploration.Geoscientific Model
Development, 12(3):1165–1187, 2019.
[23] Fabio Luporini, Mathias Louboutin, Michael Lange, Navjot Kukreja,
Philipp A. Witte, Jan H ¨uckelheim, Charles R. Yount, Paul H. J. Kelly,
Felix J. Herrmann, and Gerard J. Gorman. Architecture and perfor-
mance of devito, a system for automated stencil computation.ACM
Transactions on Mathematical Software, 46(1), April 2020.
[24] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Sch ¨utze.
Introduction to information retrieval. Cambridge University Press,
Cambridge, UK, 2008.
[25] Microsoft Research. Graphrag: Unlocking llm discovery on narrative
private data. Microsoft Research Blog [Online], 2024. [Accessed: 28
August 2025].
[26] Niklas Muennighoff, Nouamane Tazi, Lo ¨ıc Magne, and Nils Reimers.
Mteb: Massive text embedding benchmark. In Andreas Vlachos and
Isabelle Augenstein, editors,Proceedings of the 17th Conference of the
European Chapter of the Association for Computational Linguistics,
pages 2014–2037, Dubrovnik, Croatia, May 2023. Association for
Computational Linguistics.
[27] OpenAI. Doing rag on pdfs using file search in the responses api.
OpenAI Cookbook, 2025. [Accessed 28 August 2025].
[28] Pinecone. Offline evaluation for ranking models: precision and recall
at k. Online, 2023. Available from: https://www.pinecone.io/learn/
offline-evaluation/ [Accessed 18 August 2025].[29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings
using siamese bert-networks. In Kentaro Inui, Jing Jiang, Vincent Ng,
and Xiaojun Wan, editors,Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th Inter-
national Joint Conference on Natural Language Processing (EMNLP-
IJCNLP), pages 3982–3992, Hong Kong, China, November 2019. As-
sociation for Computational Linguistics.
[30] Ian Robinson, Jim Webber, and Emil Eifrem.Graph databases. O’Reilly
Media, Sebastopol, CA, 2nd edition, 2015.
[31] Ruff Documentation. The ruff formatter. Online, 2025. Astral Documen-
tation. Available from: https://docs.astral.sh/ruff/formatter/ [Accessed 18
August 2025].
[32] Ellen M. V oorhees. The trec-8 question answering track report. In
Proceedings of the Eighth Text REtrieval Conference (TREC-8), Novem-
ber 1999, Gaithersburg, MD, USA, volume 500-246 ofNIST Special
Publication, pages 77–82, Gaithersburg, MD, 1999. National Institute
of Standards and Technology (NIST).
[33] Shitao Wang, Zhicheng Wu, Kai Yang, Chen Qian, Xin Wang, Weiqiang
Zheng, and Junfeng Yang. Enhancing llm-based code translation in
repository context via triple knowledge-augmented.arXiv preprint
arXiv:2503.18305, 2025.