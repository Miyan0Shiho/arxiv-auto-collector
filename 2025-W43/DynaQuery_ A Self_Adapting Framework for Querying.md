# DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data

**Authors**: Aymane Hassini

**Published**: 2025-10-20 19:02:35

**PDF URL**: [http://arxiv.org/pdf/2510.18029v1](http://arxiv.org/pdf/2510.18029v1)

## Abstract
The rise of Large Language Models (LLMs) has accelerated the long-standing
goal of enabling natural language querying over complex, hybrid databases. Yet,
this ambition exposes a dual challenge: reasoning jointly over structured,
multi-relational schemas and the semantic content of linked unstructured
assets. To overcome this, we present DynaQuery - a unified, self-adapting
framework that serves as a practical blueprint for next-generation "Unbound
Databases." At the heart of DynaQuery lies the Schema Introspection and Linking
Engine (SILE), a novel systems primitive that elevates schema linking to a
first-class query planning phase. We conduct a rigorous, multi-benchmark
empirical evaluation of this structure-aware architecture against the prevalent
unstructured Retrieval-Augmented Generation (RAG) paradigm. Our results
demonstrate that the unstructured retrieval paradigm is architecturally
susceptible to catastrophic contextual failures, such as SCHEMA_HALLUCINATION,
leading to unreliable query generation. In contrast, our SILE-based design
establishes a substantially more robust foundation, nearly eliminating this
failure mode. Moreover, end-to-end validation on a complex, newly curated
benchmark uncovers a key generalization principle: the transition from pure
schema-awareness to holistic semantics-awareness. Taken together, our findings
provide a validated architectural basis for developing natural language
database interfaces that are robust, adaptable, and predictably consistent.

## Full Text


<!-- PDF content starts -->

DynaQuery: A Self-Adapting Framework for Querying Structured
and Multimodal Data
Aymane Hassini
Al Akhawayn University
Ifrane, Morocco
A.hassini@aui.ma
ABSTRACT
The rise of Large Language Models (LLMs) has accelerated the
long-standing goal of enabling natural language querying over com-
plex, hybrid databases. Yet, this ambition exposes a dual challenge:
reasoning jointly over structured, multi-relational schemas and
the semantic content of linked unstructured assets. To overcome
this, we presentDynaQuery—a unified, self-adapting framework
that serves as a practical blueprint for next-generation “Unbound
Databases.” At the heart of DynaQuery lies theSchema Intro-
spection and Linking Engine (SILE), a novel systems primi-
tive that elevates schema linking to a first-class query planning
phase. We conduct a rigorous, multi-benchmark empirical eval-
uation of this structure-aware architecture against the prevalent
unstructured Retrieval-Augmented Generation (RAG) paradigm.
Our results demonstrate that the unstructured retrieval paradigm is
architecturally susceptible to catastrophic contextual failures, such
asSCHEMA_HALLUCINATION , leading to unreliable query generation.
In contrast, our SILE-based design establishes a substantially more
robust foundation, nearly eliminating this failure mode. Moreover,
end-to-end validation on a complex, newly curated benchmark
uncovers a key generalization principle: the transition from pure
schema-awarenessto holisticsemantics-awareness. Taken together,
our findings provide a validated architectural basis for developing
natural language database interfaces that are robust, adaptable, and
predictably consistent.
Artifact Availability:
The source code, data, and/or other artifacts for this paper have been made
available at: https://github.com/aymanehassini/DynaQuery.
1 INTRODUCTION
The task of translating natural language questions into executable
SQL queries, known as Text-to-SQL, is a cornerstone of modern data
accessibility [ 13]. The advent of Large Language Models (LLMs)
has significantly advanced the state-of-the-art [ 18], promising to
democratize data access for non-technical users—a long-standing
goal of the field [ 1]. However, this progress has also illuminated
a dual challenge inherent in modern data ecosystems. First, users
need to query complex, multi-table relational schemas with high
fidelity (the structured challenge). Second, with databases increas-
ingly storing pointers to unstructured assets [ 2], users need to
reason over the semantic content of linked images and documents
(the multimodal challenge), a task explored by recent multimodal
query systems [7, 28].While this dual challenge is clear, the benchmarks for measur-
ing progress have evolved at different paces. The structured chal-
lenge has seen a significant evolution. The landscape was histor-
ically driven by benchmarks like Spider [ 41], which established
the difficulty of generating syntactically complex, cross-domain
SQL. Recognizing that even this fell short of real-world scenarios,
the community has developed next-generation paradigms [ 15,18].
BIRD [ 18] introduces data-level complexities—such as noisy values,
external knowledge, and the need for query efficiency—while Spi-
der 2.0 [ 15] introduces workflow-level complexities by evaluating
agentic systems. In parallel, the multimodal challenge remains a
more nascent but equally critical frontier. A truly universal data
interface must bridge this divide between the structured and the
unstructured world.
In response to these challenges, leaders in the data systems
community have called for "Unbound Databases" [ 24]—a new gen-
eration of systems designed from the ground up to query the full
spectrum of the world’s data. While this vision provides a powerful
architectural roadmap, the concrete engineering challenges of build-
ing such a system remain a critical open problem. Foundational
work like the ’Generating Impossible Queries’ (GIQ) system [ 26]
demonstrated a novel method for querying linked multimodal data,
but its implementation was a proof-of-concept, leaving the chal-
lenges of generalization and system-level adaptation unsolved.
To address this gap, we introduceDynaQuery, a unified, self-
adapting framework that provides a practical blueprint for the
Unbound Database. Our work is motivated by a core principle:
before the advanced complexities of modern benchmarks can be
consistently met, a system must first master the foundational task
of robust schema linking. DynaQuery is designed around this prin-
ciple, with a novel query planning engine at its core. This work
makes the following scientific contributions:
•A Rigorous Empirical Analysis of Linking Architec-
tures:We provide a definitive, multi-benchmark compar-
ison of our structure-aware linking engine (SILE) versus
unstructured RAG [ 17]. Our programmatic failure analy-
sis systematically demonstrates that a primary source of
RAG’s architectural brittleness is SCHEMA_HALLUCINATION ,
a catastrophic failure mode that our approach nearly elimi-
nates.
•An Empirical Analysis of Architectural Trade-offs in
Pluggable Decision Modules:We provide a rigorous anal-
ysis of the trade-offs in building a robust decision module,
dissecting the critical systems properties of predictability
and generalization. This culminates in a key insight into
the conflict between benchmark alignment and real-world
robustness.arXiv:2510.18029v1  [cs.DB]  20 Oct 2025

Aymane Hassini
•A Unified, Generalizable Framework and its End-to-
End Validation:We formalize and implement DynaQuery,
a unified framework integrating our robust linker with
structured (SQP) and multimodal (MMP) pipelines. We
then conduct a rigorous end-to-end validation of the frame-
work’s ability to generalize to a complex, "in-the-wild" data-
base, identifying key architectural principles for building
truly adaptable systems.
Through DynaQuery, we provide a validated blueprint for hybrid,
self-adapting database interfaces, bridging the gap between high-
level vision and practical application.
2 BACKGROUND AND CONCEPTUAL
FRAMEWORK
This section formally defines the core tasks addressed by our frame-
work and situates our work within the context of relevant prior re-
search. We first provide a formal problem formulation, then discuss
the foundational systems and concepts that motivate our approach,
and finally, we detail the specific technical challenge of schema
linking.
2.1 Formal Problem Formulation
To establish the scope of our work, we formally define the two
distinct yet related tasks that the DynaQuery framework is designed
to solve.
Database Definition:Let a database 𝐷be a collection of tables
𝑇={𝑡 1,𝑡2,...,𝑡𝑛}. Each table 𝑡𝑖has a schema 𝑆𝑖consisting of a set
of columns𝐶𝑖={𝑐𝑖,1,𝑐𝑖,2,...} , a set of primary keys 𝑃𝐾𝑖, and a set
of foreign keys𝐹𝐾 𝑖. The full database schema is𝑆 𝐷={𝑆 1,𝑆2,...}.
Task 1: The NL-to-SQL Task.Given a natural language query
𝑄𝑠and the database D, the goal is to generate a SQL query S such
that its execution S(D) yields the correct answer set 𝑅𝑠that satisfies
the user’s intent expressed in𝑄 𝑠[41].
Task 2: The Multimodal Querying Task.Given a natural
language query 𝑄𝑚and the database D, where one or more columns
𝑐𝑖,𝑗may contain pointers to multimodal data 𝑀(e.g., image URLs,
document paths), the goal is to produce a filtered subset of records
𝑅𝑚that satisfy semantic criteria expressed in 𝑄𝑚, which require
reasoning over the content of 𝑀and are not answerable by standard
SQL operations on𝑆 𝐷[7, 28].
2.2 Foundational Systems and Concepts
Our work is built upon and motivated by two key pillars of re-
cent research: the grand vision for next-generation data systems
articulated in the "Unbound Database" paper, and the conceptual
framework for querying latent attributes established by the ’Gener-
ating Impossible Queries’ (GIQ) system.
The Unbound Database Vision.Our framework is designed
as a direct and concrete instantiation of the "Unbound Database"
vision proposed by Madden et al. [ 24]. This vision calls for a new
generation of data systems designed from the ground up to query
the full spectrum of the world’s data, from structured tables to
unstructured assets. Table 1 explicitly maps the core architectural
concepts of this vision to their concrete implementations within
DynaQuery, demonstrating how our work serves as a practical
blueprint for this ambitious goal.The ’Generating Impossible Queries’ (GIQ) System.The
technical basis for our multimodal pipeline is the GIQ system [ 26],
which first established the use of multimodal LLMs to query latent,
semantic attributes in databases. The work successfully demon-
strated that an LLM could reason over linked text and image data
to answer queries unsolvable by standard SQL. However, the GIQ
implementation was a proof-of-concept on a controlled, single-
table schema. It left the critical engineering challenges of gen-
eralization—such as schema-dependence, multi-table scope, and
dynamic content discovery—unsolved. A primary contribution of
DynaQuery is to address these challenges, thereby operationalizing
the GIQ concept for complex, real-world databases.
2.3 The Schema Linking Problem in NL-to-SQL
Accurate schema linking—the task of correctly identifying the ta-
bles and columns in a database schema that are referenced in a
natural language query—is a critical prerequisite for any Text-to-
SQL system [ 23]. Foundational work by Lei et al. [ 16] provided
the first systematic, data-driven study of this task, arguing that
schema linking is not a minor component but the very "crux" of
the Text-to-SQL problem.
In the modern context of LLMs, the challenge is no longer prov-
ing the importance of schema linking, but rather developing robust
and scalable mechanisms to perform it. As noted in recent work,
providing the full, unfiltered schema of a large database directly into
a query-generation prompt can overwhelm an LLM’s context win-
dow and degrade performance [ 10]. Therefore, an effective linking
and pruning mechanism is essential. Approaches for this generally
fall into two categories, which we compare in our experiments:
Unstructured Context Retrieval (RAG):This paradigm, intro-
duced by Lewis et al.[ 17], reframes the structured schema linking
problem as an unstructured information retrieval task. While so-
phisticated systems like RASL[ 9] employ intelligent decomposition
to avoid fracturing relational integrity, the retrieval-first paradigm
remains architecturally flawed. The process is a decoupled, two-
phase model: a probabilistic retrieval phase precedes a deterministic
reasoning phase. Much like a query plan that prematurely discards
data, if the initial retrieval—acting as a lossy filter—fails to surface a
necessary table or transitive dependency, that structural context is
irrevocably lost. The downstream generator is thus forced to reason
over an incomplete schema, leading to incorrect join paths [ 6] and
fundamentally subordinating guaranteed structural integrity to the
probabilistic recall of an information retrieval model.
Structure-Aware Holistic Analysis:In contrast to unstruc-
tured retrieval, our work adopts a structure-aware, holistic ap-
proach that aligns with state-of-the-art methods prioritizing re-
lational integrity [ 29]. This two-stage process first leverages an
LLM’s reasoning capabilities over a complete, structured view of
the schema to identify the relevant tables. In a second, program-
matic pruning step, it uses this selection to construct a minimal
schema context. This method is architecturally designed to pre-
serve the global relational context, thereby avoiding the risk of
irrecoverable context pruning introduced by RAG’s probabilistic,
retrieval-first architecture.

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
Table 1: Mapping Unbound Database Concepts [24] to DynaQuery’s Concrete Components.
Unbound Database Concept DynaQuery Implementation
Declarative Interface Natural Language query input, processed by the SILE as a "what, not how" request.
Logical Operators (e.g., Filter) The core semantic reasoning step within the MMP, which acts as a powerfulsemantic Filterover
records based on their linked unstructured content.
Multi-Objective Optimization Our framework’s modular designenablesmulti-objective trade-offs. The pluggable Decision
Module (RQ2) demonstrates this by allowing a choice between a high-cost/high-generalization
LLM and a low-cost/specialist BERT model.
Physical Plan Operators The specific Chain-of-Thought prompts and classification modules used within our pipelines,
which represent concrete implementations of logical reasoning steps.
Code Generation / Optimization The Zero-Shot NL-to-SQL Pipeline (SQP), which generates efficient SQL code to delegate structured
query processing to the database engine.
Extensibility ("Recipe Box") The engineered prompts within SILE and the pipelines serve as reusable "recipes" for consistent
reasoning. The modular architecture also allows for new pipelines to be added.
3 THE DYNAQUERY SELF-ADAPTING
FRAMEWORK
The architecture of DynaQuery is founded on the philosophy of
creating a modular, "plug-and-play" system that decouples query
processing logic from the physical database schema. This modular-
ity is key to our framework’s flexibility and allows it to incorporate
established LLM engineering patterns, such as Chain-of-Thought
(CoT) prompting to elicit sophisticated reasoning [ 37]. Unlike static
systems requiring manual configuration for each new database, Dy-
naQuery isself-adapting. It uses a central schema introspection
and linking engine to programmatically discover the schema and
dynamically configure its query pipelines at runtime. This design
aligns with the modular, lifecycle-based view of modern NL2SQL
systems, as surveyed by Liu et al. [20].
3.1 Architectural Overview
As illustrated in Figure 1, the DynaQuery framework is composed of
a central engine and two specialized pipelines. The process begins
when a natural language query 𝑄, first enters theSchema Intro-
spection and Linking Engine (SILE). The SILE acts as a universal
pre-processor, analyzing the query against the database schema
to produce a high-level queryplan. This plan, which includes the
relevant tables and their relationships, forms the foundation for
the framework’s two specialized pipelines. Our design prioritizes
explicit user control. Rather than attempting automated task classi-
fication, the user directs the query and its generated plan to one of
two specialized pipelines: theGeneralized Multimodal Pipeline
(MMP)for complex, semantic queries over unstructured data, or
theZero-Shot NL-to-SQL Pipeline (SQP)for structured data re-
trieval. This modular, orchestrated architecture is realized using
the LangChain framework [5].
3.2 The Core Enabler: The Schema
Introspection and Linking Engine (SILE)
The SILE is the cornerstone of the framework’s adaptability, acting
as a first-class query planning primitive. Its design combines pro-
grammatic schema discovery with LLM-driven strategic reasoning.Programmatic Introspection:Upon its first connection to a
database, the SILE programmatically inspects the database’s catalog
to construct a detailed, in-memory representation of the full schema,
𝑆𝐷. This representation includes all tables, columns, data types,
and relational constraints (e.g., primary and foreign keys). This
structured metadata is generated once and cached, providing a
comprehensive and efficient foundation for all subsequent query
planning operations.
The SILE elevates schema linking from a simple selection task to
a strategic query planning phase. The full schema representation
is provided as context to an LLM, which is prompted to generate
a structured query plan [ 23]. To enhance the robustness of this
process, we leverage Chain-of-Thought (CoT) prompting [ 37], en-
couraging the model to explicitly reason about the entities in the
user’s query and their mapping to the schema before producing
the final plan. This plan is not merely a list of relevant tables; it
is a conceptual blueprint that distinguishes between thebase ta-
ble(containing the primary entity of the user’s request) and any
necessaryjoin tables. By producing a structured, reasoned plan, the
SILE provides a robust strategic foundation for the downstream
pipelines.
3.3 Pipeline A: The Generalized Multimodal
Pipeline (MMP)
This pipeline is our primary contribution, designed to operationalize
and generalize the conceptual framework of the GIQ system for
complex, multi-table schemas. The architecture is founded on the
principle of delegating structured filtering to the database engine
to ensure scalability, a principle actively explored in modern query
optimization [ 40], which we term the "Ultimate Filtered Join". The
overall workflow is formalized in Algorithm 1, and its stages are
detailed below.
(1)Query Planning and Schema Pruning (Lines 2–3):The
workflow begins by invoking the SILE to generate a high-
level query plan. This plan identifies all tables required to
answer the query, which are then used to create a minimal,
pruned schema context, denoted as𝑆′
𝐷.

Aymane Hassini
User Input 
Natural Language Query (Q)
Selected Mode: (SQP | MMP)
Schema Introspection & Linking
Engine (SILE)
     1. Programmatic Introspection 
     2.  LLM-Driven Query Planning
QueDB Schema 
Query (Q) + Plan
Mode ?
Final Result 
Generalized Multimodal
Pipeline (MMP)
(see Algorithm 1)
Zero-Shot NL-to-SQL
Pipeline (SQP)
(see Algorithm 2)
Figure 1: The high-level architecture of the DynaQuery framework
(2)Optimized Query Assembly (Lines 4–8):To construct
a highly selective SQL query, the MMP orchestrates sev-
eral specialized components. A key optimization is per-
formed first: the system programmatically discovers which
columns in the pruned schema contain multimodal data
(Line 4). This allows the final query to be constructed to
pre-filter any rows that lack the necessary unstructured
content. Concurrently, LLM-powered chains generate the
structured WHERE conditions (Line 5) and the necessary
JOIN logic (Line 6). These components are then assembled
into a single, optimized SQL query (Line 8).
(3)Candidate Set Retrieval (Line 9):The assembled SQL
query is executed, retrieving a small and manageable set of
candidate records, 𝑅cand. This step leverages the database’s
query optimizer to perform the vast majority of the filtering
work efficiently.
(4)Per-Record Multimodal Reasoning (Lines 15–17):Each
record in the pre-filtered candidate set is then processed
in a loop. A powerful multimodal LLM generates a ratio-
nale using Chain-of-Thought prompting [ 37], evaluating
the record’s unstructured content against the user’s seman-
tic criteria. This rationale is then passed to a pluggable
Decision Module to assign a final label.
(5)Final Decision and Answer Synthesis (Lines 18–22, 26–
31):If a record is classified as ’ACCEPT’, its primary key
is collected. After all candidates are processed, these keys
are used to execute a final, simple query to retrieve the fullrecords, producing the answer set 𝑅final. This two-phase pro-
cess ensures that expensive multimodal reasoning is only
performed on a small, highly-relevant, and pre-qualified
subset of the data.
3.4 Pipeline B: The Zero-Shot NL-to-SQL
Pipeline (SQP)
This complementary pipeline is responsible for efficiently and accu-
rately handling purely structured queries in a zero-shot manner [ 10].
Its straightforward workflow, formalized in Algorithm 2, leverages
the SILE for robust context before delegating the core generation
task to a powerful LLM.
The workflow proceeds in three stages:
(1)Contextualization (Lines 1–2):Similar to the MMP, the
SQP’s first step is to invoke the SILE to obtain a query plan
and a pruned schema context, 𝑆′
𝐷. Providing this minimal,
high-fidelity context is crucial for preventing LLM confu-
sion and improving the accuracy of the generated SQL [ 23].
(2)Query Generation (Line 3):The linked schema and the
user’s query are passed to a powerful LLM using a carefully
engineered zero-shot prompt. This prompt instructs the
model to act as a SQL expert, leveraging its extensive pre-
trained knowledge to generate a syntactically correct SQL
query.
(3)Sanitization and Execution (Lines 4–5):The raw SQL
output from the LLM is passed through a rule-based sanitiza-
tion layer. This layer performs two critical functions. First, it

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
Algorithm 1The Generalized Multimodal Pipeline (MMP) Work-
flow.
Require:User query𝑄, full schema𝑆 𝐷, database instance𝐷
Ensure:Final set of accepted records𝑅 final
1:functionGetCandidateSet( 𝑄,𝑆𝐷,𝐷)⊲Phase 1: Optimized Retrieval
2:𝑃𝑙𝑎𝑛←SILE(𝑄,𝑆 𝐷)
3:𝑆′
𝐷←PruneSchema(𝑆 𝐷,𝑃𝑙𝑎𝑛.tables)
4:𝐶 multi←DiscoverMultimodalColumns(𝑆′
𝐷,𝐷)
5:𝐶 where←WhereClauseChain(𝑄,𝑆′
𝐷,𝑃𝑙𝑎𝑛)
6:𝐶 join←JoinClauseChain(𝑆′
𝐷,𝑃𝑙𝑎𝑛)
7:𝐶 wnn←BuildIsNotNullClause(𝐶 multi)
8:𝑆𝑄𝐿 cand←AssembleSQL("SELECT *",𝐶 join,𝐶where,𝐶wnn)
9:𝑅 cand←ExecuteSQL(𝑆𝑄𝐿 cand,𝐷)
10:return(𝑅 cand,𝑃𝑙𝑎𝑛,𝐶 multi)
11:end function
12:functionFilterCandidates(𝑄,𝑅 cand,𝐶multi)⊲Phase 2: Multimodal
Reasoning
13:𝑅 acc_keys←∅
14:𝑀 idx←MapColsToIndices(𝐶 multi,𝑅cand.cols)
15:for allrecord𝑟∈𝑅 canddo
16:𝑍 𝑟←GenerateRationale(𝑄,𝑟,𝑀 idx)
17:𝑙𝑎𝑏𝑒𝑙←DecisionModule(𝑄,𝑍 𝑟)
18:if𝑙𝑎𝑏𝑒𝑙=’ACCEPT’then
19:Add PrimaryKey(𝑟)to𝑅 acc_keys
20:end if
21:end for
22:return𝑅 acc_keys
23:end function
// Main Execution
24:(𝑅 cand,𝑃𝑙𝑎𝑛,𝐶 multi)←GetCandidateSet(𝑄,𝑆 𝐷,𝐷)
25:𝑅 acc_keys←FilterCandidates(𝑄,𝑅 cand,𝐶multi)
26:if𝑅 acc_keys is not emptythen
27:𝑠𝑞𝑙←"SELECT * FROM "+𝑃𝑙𝑎𝑛.base_table
+" WHERE pk IN "+𝑅 acc_keys
28:𝑅 final←ExecuteSQL(𝑠𝑞𝑙,𝐷)
29:else
30:𝑅 final←∅
31:end if
32:return𝑅 final
Algorithm 2The Zero-Shot NL-to-SQL Pipeline (SQP) Workflow.
Require:User query𝑄, full schema𝑆 𝐷, database instance𝐷
Ensure:Final SQL query result set𝑅 final
// Phase 1: Contextualization
1:𝑃𝑙𝑎𝑛←SILE(𝑄,𝑆 𝐷)
2:𝑆′
𝐷←PruneSchema(𝑆 𝐷,𝑃𝑙𝑎𝑛.tables)
// Phase 2: Generation and Execution
3:𝑆𝑄𝐿 raw←GenerateSQLZeroShot(𝑄,𝑆′
𝐷)
4:𝑆𝑄𝐿 final←SanitizeSQL(𝑆𝑄𝐿 raw)
5:𝑅 final←ExecuteSQL(𝑆𝑄𝐿 final,𝐷)
6:return𝑅 final
cleans the LLM output by removing common artifacts (e.g.,
markdown code blocks). Second, and more importantly,
it acts as asafety guardrail. The sanitizer programmati-
cally ensures that only SELECT statements are extracted and
passed to the database for execution, explicitly preventingany data modification (DML) or definition (DDL) opera-
tions [ 4]. The cleaned, validated query is then executed to
produce the final result set.
4 EXPERIMENTAL DESIGN
The primary goal of our experimental evaluation is not merely to
report isolated performance metrics, but to rigorously validate the
central claims of this paper. Our experiments are designed to dis-
sect our framework’s architecture and validate its core components
in a bottom-up fashion: we first validate our foundational linking
primitive, then analyze a critical sub-component of our multimodal
pipeline, and finally, test the end-to-end generalization of the uni-
fied system. This methodology is structured to directly answer the
following research questions:
4.1 Research Questions (RQs)
•RQ1 (Foundational Component Validation):Does the
holistic, structure-aware analysis provided by our Schema
Introspection and Linking Engine (SILE) yield higher accu-
racy on complex NL-to-SQL tasks compared to a standard
unstructured RAG baseline?
•RQ2 (Sub-Component Analysis):In the context of our
multimodal pipeline, how does a zero-shot, LLM-native
classification module compare to a fine-tuned BERT base-
line, and what are the architectural trade-offs regarding
robustness and predictability?
•RQ3 (End-to-End System Generalization):Can our uni-
fied DynaQuery framework successfully generalize beyond
standard benchmarks to operate effectively on a complex,
"in-the-wild" database, handling a diverse spectrum of both
structured and multimodal queries?
4.2 Datasets
To rigorously evaluate each component of our framework, we se-
lected and constructed a diverse set of datasets tailored to each
research question.
For Schema Linking (RQ1):
•Spider [ 41]:Our primary evaluation forRQ1uses the
widely-used Spider development set to test performance in
a syntactically complex, cross-domain environment. For our
direct linker evaluation, we use the accompanying ground-
truth labels from the Spider-Schema-Linking Split [34].
•BIRD [ 18]:We use the BIRD development set to stress-
test our linking architecture in a more complex, realistic
setting that includes noisy data and external knowledge
requirements. The rationale for selecting both Spider and
BIRD is detailed in Section 4.4.
For Classifier Comparison (RQ2):
•Annotated Rationale Dataset:To provide a robust train-
ing and evaluation corpus for the decision module, we
created and manually annotated a new dataset of 5,000
(Question, Rationale, Label) triplets. To ensure diversity,
the rationales were generated using a variety of LLMs and
prompt styles. Each triplet was then manually annotated
by an author with a label from ACCEPT, RECOMMEND,

Aymane Hassini
REJECT. These labels correspond to whether the rationale
provides a fully correct, partially correct, or incorrect justifi-
cation for answering the question, respectively. The dataset
is balanced across the three classes and will be made pub-
licly available. For our experiments, we use a standard 80/20
split for training and in-distribution (IID) testing.
•Out-of-Distribution (OOD) Case Study Set:To evaluate
true model robustness, we created a small, challenging OOD
set from a novel e-commerce domain. This set consists of
45 ‘(Question, Rationale)‘ pairs, manually labeled according
to a strict "hierarchical intent" philosophy where primary
constraints are prioritized over secondary filters. This set
is used for our qualitative case study.
For Framework Generalization (RQ3):
•Olist Multimodal Benchmark:To provide a rigorous
end-to-end test forRQ3, we constructed a new benchmark
based on the public Olist E-commerce dataset [ 27]. We aug-
mented this complex, multi-table relational schema by man-
ually linking 100 products to high-quality images, creating a
testbed that embodies the dual challenge of structured com-
plexity, semantic ambiguity, and multimodal reasoning. For
evaluation, we designed a suite of 40 novel questions. This
suite is divided into two parallel sets of 20 queries each: one
for the MMP (containing multimodal constraints) and one
for the SQP (containing purely structured constraints). To
ensure comprehensive evaluation, both 20-query sets were
manually stratified by difficulty according to the official Spi-
der hardness criteria (Easy, Medium, Hard, Extra Hard) [ 41].
The full benchmark, including our curation methodology
and queries, is provided in our artifact repository.
4.3 Baselines and Architectural Alternatives
Our experiments are designed to compare our architectural choices
against strong, relevant baselines and alternative configurations.
•For RQ1 (Linking Architectures):To test our central
hypothesis on schema linking, we compare our structure-
aware SILE against a strongunstructured RAG baseline.
Our RAG implementation is designed for a rigorous, head-
to-head comparison of the core retrieval strategy. It follows
established best practices for foundational RAG from re-
cent surveys [ 35], utilizing a state-of-the-art embedding
model (BAAI/bge-large-en-v1.5) [ 39]. To isolate the effect
of the initial context retrieval—the core of our hypothe-
sis—we deliberately exclude orthogonal, post-retrieval opti-
mizations like reranking, a common step in advanced RAG
pipelines [35], ensuring a fair and direct comparison.
•For RQ2 (Decision Modules):Our analysis of the deci-
sion module is an architectural trade-off study. We compare
our proposed zero-shot generalist (an LLM-native classifier)
against a strong specialist baseline: aBERT-based classi-
fier fine-tuned[ 8] on a large, in-domain dataset of 4,000
labeled examples.4.4 Evaluation Protocol for RQ1
To rigorously test our central hypothesis for RQ1—that a structure-
aware linker is superior to an unstructured RAG baseline—we de-
signed a multi-faceted evaluation across two key benchmarks that
represent the evolution of the Text-to-SQL field.
4.4.1 Benchmark Selection Rationale.The choice of benchmarks is
critical for a meaningful evaluation. We selected Spider and BIRD
for the following strategic reasons:
•Spider [ 41]:We include Spider as a foundational bench-
mark to validate our system’s baseline proficiency in gener-
atingsyntactically complex SQL. While acknowledging
the well-documented risk that its contents may have con-
taminated the pre-training corpora of modern LLMs [ 31],
it remains an essential standard for comparing SQL genera-
tion logic. Crucially, the availability of theSpider-Schema-
Linking Split [ 34]allows us to perform a complete, two-
part analysis, providing a clear causal link between linking
quality and final performance.
•BIRD [ 18]:Our primary end-to-end evaluation is con-
ducted on BIRD. Sourced from real industry data platforms,
BIRD presents a more robust test of a system’s practical
utility by introducing challenges such as noisy database
values, external knowledge requirements, and efficiency
considerations. As it is a more recent and complex bench-
mark, it is also less susceptible to data contamination. We
chose BIRD over the newer Spider 2.0 as our work focuses
on the foundational challenge of single-shot query genera-
tion over complexdata, which is BIRD’s core focus, rather
than the multi-turn agenticworkflowsintroduced in Spider
2.0 [15].
4.4.2 Spider Evaluation Protocol: Random Sampling with Post-Hoc
Validation.Our evaluation on Spider is shaped by a core character-
istic of the benchmark: its difficulty labels (’easy’, ’medium’, ’hard’,
’extra’) are animplicit property. They are not pre-assigned but
can only be determined by parsing the gold SQL query and counting
its syntactic components [41].
Sampling Strategy.In the absence of pre-existing categories for
stratification, we employed the standard and methodologically ap-
propriate approach: random sampling. Our initial sample consisted
of 500 queries from the annotated Spider development split [ 34]; a
programmatic cross-referencing against the canonical Spider bench-
mark [ 41] data yielded a final, unambiguous set of 475 queries for
all experiments.
Execution Protocol.To measure the semantic correctness of
the generated SQL, we employ a direct execution protocol. As
highlighted in multiple recent studies [ 30,42], the official Spider
evaluation script relies on a brittle internal parser that is sensitive
to non-semantic stylistic variations. To ensure a more robust evalu-
ation, our protocol bypasses this parsing logic entirely. A predicted
query is deemed correct if and only if its execution against the
database yields a result set identical to that of the ground-truth
query, with appropriate handling for both ordered and unordered
results. The specific metrics derived from this protocol are detailed
in Section 4.5.

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
Post-Hoc Analysis for Representativeness.To scientifically
validate our random sample, we performed a crucial post-hoc anal-
ysis. The official tool for classifying query difficulty is dependent
on the aforementioned brittle parser. We therefore developed a
robust, heuristic-based classifier that is directly modeled on the
classification rules from the official evaluation script [ 41]. This ap-
proach provides a reliable and reproducible hardness classification
without relying on the legacy parser. Our analysis using this clas-
sifier confirmed that the difficulty distribution of our 475-query
sample is statistically indistinguishable from the full 1,034-query
development set, validating that our results are representative and
generalizable.
4.4.3 BIRD Evaluation Protocol: Stratified Sampling with Explicit
Labels.Unlike Spider, the BIRD benchmark includes an explicit,
human-annotated difficulty label (’simple’, ’moderate’, ’challeng-
ing’) for every query [ 18]. This key difference dictates a different
sampling strategy and allows for the direct use of stratified random
sampling.
Sampling Strategy.We performed stratified random sampling
to construct a reproducible, 500-entry sample from the development
set. This approach guarantees representativeness by design, deliber-
ately creating a sample whose difficulty distribution is proportional
to that of the full benchmark. This ensures our evaluation on BIRD
is perfectly aligned with the challenges defined by the benchmark’s
creators.
Metrics and Execution.To measure both official metrics from
the BIRD benchmark [ 18]—Execution Accuracy (EA) and Valid Effi-
ciency Score (VES)—while maintaining experimental efficiency, we
adopted a multi-phase protocol. First, we generated prediction files
for the full 1,534-entry development set, populating our 500 sam-
pled queries and using placeholders for the rest to satisfy the official
tooling requirements. Second, we ran minimally modified versions
of the official evaluation tools to capture per-query correctness and
performance. Finally, a post-processing script filtered these detailed
reports to calculate the final EA and VES scores based only on our
500-query stratified sample. This protocol ensures our results are
both reproducible and directly comparable to the broader field.
4.5 Metrics
Two standard metrics are used to evaluate Text-to-SQL systems:
Exact Match (EM) and Execution Accuracy (EA) [ 22,41]. While EM
evaluates syntactic form, EA measures true semantic correctness
by comparing the final result sets. As recent studies have exten-
sively documented the unreliability of EM [ 30], the field has largely
adopted Execution Accuracy as the primary metric for correctness.
For classification sub-tasks, we use the standard metrics of Precision
(𝑃), Recall (𝑅), and F1-Score, defined for a given class𝑐as:
𝑃𝑐=𝑇𝑃𝑐
𝑇𝑃𝑐+𝐹𝑃𝑐
𝑅𝑐=𝑇𝑃𝑐
𝑇𝑃𝑐+𝐹𝑁𝑐
𝐹1𝑐=2·𝑃𝑐·𝑅𝑐
𝑃𝑐+𝑅𝑐
Our evaluation focuses on the following specific metrics for each
research question:•Metrics for RQ1 (Linking Architectures):Our evalua-
tion of linking architectures is two-fold:
–Direct Linking Performance:For the isolated sub-task
of identifying the correct tables, we measure standard
Precision, Recall, and F1-Score.
–End-to-End Correctness:For the full Text-to-SQL task,
our primary metric isExecution Accuracy (EA), which
measures the fraction of generated queries that pro-
duce the correct result set when executed. It is formally
defined as:
EA=1
|𝑄|∑︁
𝑞∈𝑄I(exec(𝑆𝑝𝑟𝑒𝑑)≡exec(𝑆 𝑔𝑜𝑙𝑑))
whereI(·)is the indicator function for correctness.
–End-to-End Efficiency (BIRD only):For our evaluation
on the BIRD benchmark, we also report theValid Ef-
ficiency Score (VES)[ 18]. VES combines correctness
with performance by measuring the runtime of valid
queries relative to the human-written gold standard.
It is formally expressed as:
VES=1
|𝑄|∑︁
𝑞∈𝑄I(EA)·√︄
time(𝑆𝑔𝑜𝑙𝑑)
time(𝑆𝑝𝑟𝑒𝑑)
•Metrics for RQ2 (Classifier Performance):For our three-
class classification task, we report macro-averaged Preci-
sion, Recall, and F1-Score.
4.6 Implementation Details
LLM Configuration.All Large Language Model components in
our framework, including the SILE, the SQL generation chains
(for both SQP and MMP), and the LLM-native Decision Module,
were implemented using the Google Gemini API. We utilized the
gemini-2.5-pro (Stable release, June 17, 2025) model version [ 11].
To ensure deterministic and reproducible outputs for our experi-
ments, all API calls were made with a temperature of 0.0.
Fine-Tuned Classifier.The fine-tuned specialist baseline for
our RQ2 evaluation was based on the bert-base-cased model
from the Hugging Face Transformers library [ 38]. The model was
fine-tuned for 4 epochs on ourAnnotated Rationale Datasetusing
the AdamW optimizer [ 21], with a batch size of 32 and a learning
rate of 2e-5.
Hardware and Frameworks.All experiments were conducted
on a workstation equipped with a single NVIDIA RTX 4090 GPU
(24GB VRAM). The DynaQuery framework is implemented in Python
3.10, and the LLM orchestration is built upon the LangChain li-
brary [ 5]. The database backend used for all experiments was
MySQL version 9.3.0.
5 RESULTS AND ANALYSIS
This section presents the results of our experiments, organized by
our three research questions. We first validate our foundational SILE
component (RQ1), then perform a deep-dive into a key design choice
within the MMP (RQ2), and finally, demonstrate the end-to-end
generalization of the full DynaQuery framework on a challenging,
real-world benchmark (RQ3).

Aymane Hassini
5.1 RQ1: The Primacy of Structure-Aware
Linking
To answer RQ1, we evaluated our core SILE component against a
strong RAG baseline across the Spider and BIRD benchmarks. This
experiment was designed to test our central hypothesis: that a holis-
tic, structure-aware approach to schema linking is architecturally
superior to treating the schema as an unstructured document for
retrieval.
5.1.1 Part 1: Component-Level Linking Performance.We first iso-
lated the linking components and measured their ability to identify
the correct set of tables on our 475-query sample of the Spider de-
velopment set. The results, presented in Table 2, reveal a categorical
difference in performance.
Table 2: Direct Schema Linking Performance on the Spider-
dev Set (475-entry sample). Higher is better.
Method Precision Recall F1-Score
RAG Baseline 23.6% 64.0% 32.9%
DynaQuery (SILE) 73.0% 85.6% 77.0%
The underlying metrics expose the fundamental architectural
weakness of the RAG approach for this task. While RAG achieves a
reasonable Recall (64.0%), its Precision is exceptionally low (23.6%).
This demonstrates that the retriever acts as a ’dragnet,’ retrieving
a noisy and confusing context bloated with irrelevant schema in-
formation. In contrast, our SILE’s holistic analysis is vastly more
precise (73.0%). It acts as a ’scalpel,’ preserving relational integrity
to provide a clean, minimal, and structurally coherent context for
the downstream query generator.
5.1.2 Part 2: End-to-End Performance on Spider.Next, we evaluated
the end-to-end impact of these linkers on the final SQL generation
task on Spider. The results in Table 3 confirm that the superior
context from SILE translates directly into a massive improvement
in Execution Accuracy.
Table 3: End-to-End Execution Accuracy (EA) on Spider by
Query Difficulty. Delta indicates the absolute percentage
point improvement.
Difficulty RAG Baseline DynaQuery (SILE) Delta (Pts.)
Easy 61.9% 91.8%+29.9
Medium 55.9% 77.1%+21.2
Hard 54.6% 75.3%+20.7
Extra 61.8% 85.3%+23.5
Overall 57.1% 80.0% +22.9
The superior linking quality of the SILE translates directly into
a massive performance gain in the end-to-end task. Overall, Dyna-
Query achieves an Execution Accuracy of 80.0%, a+22.9 absolute
point improvementover the RAG baseline. The nearly30-point
delta on ’easy’ queries is particularly revealing: it demonstrates that
the noisy context from the RAG baseline is a major source of erroreven for structurally simple questions. This result confirms that in
a syntactically complex environment like Spider, a structure-aware
linking primitive is a critical prerequisite for high-fidelity query
generation.
5.1.3 Part 3: Stress-Testing on the BIRD Benchmark.To validate the
robustness of our architecture in a more realistic, data-grounded
environment, we conducted our end-to-end evaluation on the BIRD
benchmark. The results, presented in Table 4, not only confirm the
superiority of our structure-aware approach but demonstrate that
its advantage widens significantly when faced with real-world data
complexity.
Table 4: End-to-End Execution Accuracy (EA) on BIRD by
Query Difficulty (500-entry stratified sample).
Difficulty RAG (EA) DynaQuery (EA) Delta (Pts.)
Simple 37.42% 66.23%+28.81
Moderate 20.53% 45.03%+24.50
Challenging 36.17% 53.19%+17.02
Overall 32.20% 58.60% +26.40
This stress-test on BIRD reveals the key finding of our evalu-
ation. While both systems’ performance decreases on this more
challenging benchmark, themannerof their degradation exposes a
fundamental architectural difference. The RAG baseline’s perfor-
mancecollapses, plummeting from 57.1% on Spider to just 32.2%.
In contrast, DynaQuery’s accuracydegrades gracefully, with its
lead over the baseline widening to a massive+26.4 absolute points.
This divergence in performance under pressure provides strong
evidence that unstructured retrieval is an unreliable foundation
for this task. Our structure-aware SILE, however, proves to be a
resilient architectural primitive, essential for building systems that
can withstand the challenges of real-world data.
5.1.4 Part 4: Programmatic Failure Analysis on BIRD.To understand
the root cause of this performance divergence, we developed a
programmatic analysis pipeline using the sqlglot library [ 25] to
parse each failed query into its Abstract Syntax Tree (AST) for
systematic categorization. The results, shown in Table 5, reveal a
pronounced, architectural contrast in the failure modes of the two
systems.
Table 5: Programmatic Failure Analysis on the BIRD Bench-
mark. Percentages are of total failures for each model.
Error Category RAG DynaQuery
Contextual Failures
SCHEMA_HALLUCINATION50.74%6.76%
JOIN_TABLE_MISMATCH23.30% 26.57%
Logical & Syntactic Failures
SELECT_COLUMN_MISMATCH10.62%33.82%
WHERE_OR_LOGIC_ERROR4.13%19.32%
Other Minor Errors 11.21% 13.53%

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
The analysis provides quantitative evidence for the architectural
brittleness of the RAG-based approach. Its failures are predomi-
nantly foundational, with over 50% of its errors falling into the
SCHEMA_HALLUCINATION category: a failure mode where the model
generates queries referencing non-existent tables or columns [ 14].
This demonstrates that the unstructured retrieval mechanism fre-
quently fails to provide a coherent and grounded context, causing
the downstream LLM to detach from the reality of the database.
In stark contrast, DynaQuery almost entirely eliminates this
catastrophic failure mode. For our system, the bottleneck shifts
from contextual grounding to logical precision, with failures now
concentrated in downstream reasoning tasks like identifying the
correct select columns SELECT_COLUMN_MISMATCH . However, this
analysis also reveals a crucial limitation of purelyschema-aware
linking. Our SILE excels at this task, correctly mapping query enti-
ties to schema names. The failures on BIRD, however, highlight a
distinct and more challenging task:data-aware linking[ 18]. This
requires the system to correctly map adata valuementioned in
the query to its corresponding column in the schema, a task that
often requires disambiguating between multiple potential columns
across different tables. Since this value-to-column mapping cannot
be resolved using schema metadata alone, our analysis concludes
that while a structure-aware foundation is critical, the path to solv-
ing the full complexity of real-world queries requires this next level
of data-aware reasoning.
5.1.5 Part 5: Validating Query Quality with Efficiency Score (VES).
To move beyond a binary measure of correctness and assess the
practical utility of our systems, we evaluate the quality of the gener-
ated SQL using the Valid Efficiency Score (VES), a metric introduced
by the BIRD benchmark specifically to address SQL performance
on large-scale databases [ 18]. VES elegantly combines correctness
and performance, awarding a score of 0 to incorrect queries and a
performance-based score to correct ones. Our results, presented in
Table 6, provide compelling evidence for the architectural superior-
ity of our structure-aware approach in generating not just correct,
but also high-quality, efficient SQL.
Table 6: Valid Efficiency Score (VES) on the BIRD Benchmark.
The performance gain of DynaQuery (SILE) over the RAG
baseline is shown in absolute points. Higher is better.
Difficulty RAG (VES) DynaQuery (VES) Gain (pts.)
Simple 38.63 67.81+29.18
Moderate 21.81 50.74+28.93
Challenging 39.14 62.96+23.82
Overall 33.60 62.20 +28.60
Evaluating query efficiency reveals a dramatic performance gap.
Overall, DynaQuery (62.20) achieves a score that is nearly double
that of the RAG baseline (33.60), a massive performance gain of
+28.6 points. This demonstrates that the clean, minimal, and rela-
tionally coherent context provided by SILE enables the downstream
LLM to generate more optimal query plans. The low VES of theRAG baseline is a direct consequence of its low Execution Accu-
racy; since incorrect queries receive a VES of 0, its score is heavily
penalized.
Notably, DynaQuery’s overall VES (62.20) is higher than its Ex-
ecution Accuracy (58.60%). This indicates that for the queries it
answers correctly, it frequently generates SQL that is more perfor-
mant than the human-written gold standard, a phenomenon that
aligns with recent studies questioning the optimality of benchmark
ground-truth queries [ 30]. This phenomenon can be attributed to
the "SILE Effect": by providing a simplified problem space, SILE al-
lows the generator LLM to leverage its vast pre-trained knowledge
of diverse SQL patterns to identify more direct relational paths.
This often results in avoiding the unnecessary JOINs or inefficient
subqueries that can arise from the noisy, bloated context provided
by an unstructured RAG retriever. This finding confirms that a
structure-aware linking primitive like SILE is a critical component
for building Text-to-SQL systems that are not just functionally cor-
rect but also efficient enough for practical, real-world deployment.
5.2 RQ2: Architectural Trade-offs in the
Decision Module
Having established the superiority of our structure-aware linking
architecture in RQ1, we now turn our focus inward to a critical
component of the Generalized Multimodal Pipeline (MMP): the final
decision module. This module is responsible for the crucial last step
of translating an LLM’s unstructured, natural language rationale
into a structured, discrete classification ({ACCEPT, RECOMMEND,
REJECT}). The choice of architecture for this component is non-
trivial and has significant implications for the overall system’s
robustness and predictability.
Our RQ2 evaluation therefore dissects the architectural trade-
offs between two primary approaches: afine-tuned specialist
(BERT) [ 8] and aprompt-guided generalist(LLM). Our analysis
reveals that while the specialist excels on in-distribution data, it
proves to be unpredictable and error-prone when faced with novel
semantic challenges. In contrast, we show that a generalist LLM,
when engineered with a precise, rule-based prompt, can be config-
ured as a robust and predictable logical engine—a critical property
for any reliable database system.
5.2.1 Part 1: In-Distribution Performance and Prompt Alignment.
We first established a baseline by evaluating three classifier architec-
tures on our 1,000-sample in-distribution (IID) test set. This dataset
was annotated by a human expert with a nuanced, context-aware
philosophy. The architectures are:
•Fine-Tuned BERT:A specialist model trained on 4,000
labeled examples [8].
•LLM (Descriptive Prompt):A generalist LLM guided by
a detailed, human-like prompt designed to mirror the an-
notation philosophy.
•LLM (Rule-Based Prompt):A generalist LLM guided by
a strict, logical prompt. This prompt instructs the model
to follow a simple, literal interpretation of the labels by
counting the user’s constraints: ACCEPT if ALL constraints
are met, RECOMMEND if SOME (but not all) are met, and
REJECT if NONE are met.

Aymane Hassini
Table 7: Macro F1-Scores on the In-Distribution (IID) Test Set.
The results highlight the effect of prompt-data alignment.
Classifier Architecture Macro F1-Score
Fine-Tuned BERT (Specialist)99.1%
LLM (Descriptive Prompt) 94.7%
LLM (Rule-Based Prompt) 78.0%
As shown in Table 7, both the BERT model and the Descriptive
Prompt LLM achieved high performance, as they are philosoph-
ically aligned with the test set’s labels. Counter-intuitively, the
logically stricter Rule-Based prompt performed worse. A manual
analysis confirmed this was not due to model error, but a philo-
sophical disagreement: the prompt’s rigid ALL/SOME/NONE logic
conflicted with the more nuanced, hierarchical judgments in the
ground truth. This demonstrates that a classifier’s performance on
a static benchmark is not an absolute measure of its capability, but a
measure of its alignment with the benchmark’s labeling philosophy.
5.2.2 Part 2: Out-of-Distribution (OOD) Qualitative Case Study.To
test for true generalization and robustness, we conducted a quali-
tative case study on a small, out-of-domain dataset. We designed
a suite of three strategically chosen queries to stress-test different
reasoning capabilities: simple visual matching, logical negation,
and complex compound logic.
•Q1 (Simple Visual):“Show me products with green lid.”
•Q2 (Negation & Context):“Show me products that do not
have the word ’baby’ in their packaging image.”
•Q3 (Compound Logic):“Show me products with a dis-
penser pump head, a rating ≥4.5, and number of ratings
>100.”
We evaluated all three classifier architectures against a manually
curated ground truth reflecting a nuanced, “hierarchical intent” phi-
losophy. The results, summarized in Table 8, reveal the specialist
model’s failure to generalize and highlight a fundamental diver-
gence in the behavior of the two generalist LLM configurations.
Table 8: Accuracy of each classifier on the three OOD queries
against our human-centric, hierarchical ground truth.
Query Type BERT LLM (Desc.) LLM (Rule)
Q1: Simple Visual 93.3% 100% 100%
Q2: Negation 73.3% 100% 100%
Q3: Compound Logic 73.3% 93.3%20.0%
Overall Accuracy78.9% 97.8%73.3%
Our qualitative analysis revealed three distinct and systematic
behavioral profiles corresponding to the different classifier archi-
tectures.
The Specialist: The Limits of Specialization.The fine-tuned
BERT model, which demonstrated near-perfect accuracy on in-
distribution data, struggled to generalize when faced with novel
semantic structures. Its primary failure mode was an over-reliance
on statistical patterns learned during fine-tuning. This was mostevident in the logical negation task (Q2), where strong keyword
associations (e.g., with the word "baby") appeared to override the
explicit syntactic cues for negation within the rationale. Its incon-
sistent performance on the compound query (Q3) further suggests
that its specialized knowledge did not equip it to handle complex,
unseen logical compositions. This behavior is characteristic of spe-
cialist models: they achieve high performance by optimizing for
known patterns but can fail unpredictably when encountering out-
of-distribution phenomena for which no pattern has been learned.
The Descriptive Generalist: The Challenge of Implicit Hi-
erarchies.The LLM guided by a descriptive, human-like prompt
successfully handled simple matching and negation. However, when
faced with compound logic (Q3), it exposed a fundamental challenge
in translating nuanced human intent into strict query semantics.
The model struggled to infer the implicit priority of constraints,
often treating them as a soft checklist. For instance, it would clas-
sify a record as RECOMMEND for satisfying secondary numerical
criteria, even when it failed the primary structural constraint (e.g.,
lacking a "dispenser pump head"). While this behavior mimics a
flexible, human-like interpretation, it conflicts with the precise, non-
negotiable semantics required by a reliable data retrieval system.
The Rule-Based Generalist: Predictable and Logically Con-
sistent.Conversely, the Rule-based LLM demonstrated the prop-
erties of a truly robust and predictable system. Its strict, logical
framework enabled it to flawlessly handle simple matching and
negation. Its dramatically low accuracy on the compound query
(Q3) is not a failure, but rather the key finding of this analysis.
The model correctly executes a literal, boolean interpretation of
the query, treating all constraints as equally important. For exam-
ple, it correctly labels a product that fails the critical ’dispenser
pump’ constraint but meets a minor secondary one as RECOM-
MEND (since 1 of 3 conditions were met). While this diverges from
our nuanced, hierarchical ground truth (which would label it RE-
JECT), thislogical consistencyis a crucial and desirable property
for a predictable database system. This fundamental trade-off be-
tween inferred intent and literal execution is analyzed further in
our Discussion.
Figure 2 presents a side-by-side analysis of two "golden exam-
ples" from our case study. Panel (a) shows a catastrophic failure of
the BERT specialist on a logical negation query (Q2). Panel (b) pro-
vides a clear illustration of the "Inferred vs. Literal Intent" conflict,
showing how the Rule-Based LLM’s logically consistent decision
diverges from our human-centric ground truth on a complex com-
pound query (Q3). These examples provide definitive, qualitative
evidence for the superior robustness and predictability of the Rule-
Based LLM architecture.
5.3 RQ3: Validating Generalization of
DynaQuery
Our final and most challenging research question tests the end-
to-end generalization of the full DynaQuery framework. To do
this, we use our novel Olist benchmark, which was constructed
to embody the dual challenges ofstructured complexity and
multimodal reasoningwhile also introducing a high degree of
real-worldsemantic ambiguity. Our evaluation proceeds in two
stages. First, we measure the performance of our baseline, purely

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
Userprompt:Showmeproductsthatdoesnothavetheword“baby”intheirpackagingimage
RationaleConclusion:Thisrecorddoesnotmatchtheuser'srequestbecausetheword“Baby”isclearlyvisibleonthepackagingintheproductimagewithinthe"BabyMoments"logo.
Analysis:BERT'slearnedkeywordassociationsoverrodetheexplicitnegativeconclusionintherationale.Itreactedtothepresenceoftheword'Baby'withoutprocessingthelogicalcontextoftheuser'snegativeconstraint.
Userprompt:Showmeproductswithdispenserpumpheadwitharating>=4.5andnumberofratings>100
RationaleConclusion:Theproducthasaflip-topcap(notadispenserpumphead),butmeetstheothertwocriteria(rating4.7,numberofratings>100).
Analysis:TheRule-BasedLLMisnot"wrong";itisfollowingadifferent,literalphilosophy.Thishighlightsthetrade-offbetweenapredictable,logicalengineandasystemthatattemptstoinferauser'snuanced,hierarchicalintent.
Panel (a): Specialist Failure: Logical NegationPanel (b): Generalist Trade-off: Inferred vs. Literal Intent
Rule-Based LLM (Literal Intent):RECOMMEND(INCORRECT)(Correctly follows its 'SOME' logic as 2/3 conditions were met.)Ground Truth (Inferred Intent):REJECT(CORRECT)
(Correctly prioritizes the critical 'dispenser' feature.)LLM (Rule-Based) DECISION:REJECT(CORRECT)
BERT DECISION:ACCEPT(INCORRECT)
Figure 2: Qualitative analysis of systematic behaviors on Out-of-Distribution (OOD) queries. Panel (a) shows the specialist’s
catastrophic failure on logical negation. Panel (b) illustrates the fundamental conflict between a system that infers hierarchical
intent versus one that executes literal, logical intent.
schema-aware framework to establish a performance baseline on
this new, challenging task. Second, we introduce a minimal archi-
tectural adaptation, Schema Enrichment, and measure its impact
on performance to test our hypothesis about the importance of
semantics-awareness for generalization.
5.3.1 Performance of the Baseline Schema-Aware Framework.We
first deployed the framework using the baseline, schema-aware
SILE that proved effective in RQ1. We evaluated both pipelines on
our 20-query, hardness-stratified Olist benchmarks. As shown in
Table 9, the performance of this baseline configuration was erratic
and clearly limited by the benchmark’s semantic complexity.
The baseline framework’s performance profile was inconsistent.
While the SQP achieved perfect accuracy on simple queries, its
performance was unpredictable on more complex tasks, dropping
to 40.00% on medium-difficulty queries before recovering to 80.00%
on hard queries. This erratic behavior, along with the MMP’s gen-
erally low performance, pointed to a systematic issue. A failure
analysis confirmed the root cause: the schema-aware SILE con-
sistently failed to resolve the high degree of semantic ambiguity
in the Olist schema (e.g., its implicit multilingual relationships).
This result empirically demonstrates that pure schema-awareness,Table 9: Performance of the baseline (schema-aware) Dyna-
Query framework on the Olist benchmark, by hardness.
Hardness MMP (F1-Score) SQP (Exec. Acc.)
Easy 57.77% 100.00%
Medium 60.00% 40.00%
Hard 59.00% 80.00%
Extra Hard 40.00% 40.00%
Overall 54.20% 65.00%
while a strong foundation, is insufficient for robust generalization
to complex, real-world database schemas.
5.3.2 Adaptation to Semantic Awareness and Final Performance.
This finding points towards the need for a more advanced linking
capability. While our analysis in RQ1 suggests that a fullydata-
awarelinker is the ultimate goal, we hypothesize that a practical
and powerful intermediate step is to make the systemsemantics-
aware. To test this, we introduced a minimal architectural adapta-
tion:Schema Enrichment. This technique enhances the SILE by
allowing it to ingest an optional Semantic Schema Description file,

Aymane Hassini
providing human-readable comments for schema elements. This
approach respects our “plug-and-play” philosophy, as providing a
data dictionary is a cornerstone of good database management and
a critical artifact for developers [19].
With this simple enhancement, we re-ran the benchmarks. The
impact was transformative, as shown in Table 10.
Table 10: Final performance of the semantics-aware Dyna-
Query framework on the Olist benchmark by hardness.
Hardness MMP (F1-Score) SQP (Exec. Acc.)
Easy 97.77% 100.00%
Medium 95.00% 100.00%
Hard 80.00% 100.00%
Extra Hard 100.00% 80.00%
Overall 93.20% 95.00%
The impact of this adaptation was transformative. With semantic
context, the SQP’s overall Execution Accuracy surged by 30 per-
centage points to a near-perfect95.00%. The effect on the MMP was
even more pronounced, with its F1-score increasing dramatically
from 54.00% to93.20%. This dramatic improvement validates our
hypothesis that for complex, real-world schemas, bridging the ’se-
mantic gap’ is a critical prerequisite for generalization. Our Schema
Enrichment technique provides a practical and effective mechanism
to achieve this.
5.3.3 Conclusion for RQ3.This experiment validates that our full
DynaQuery framework can successfully generalize to complex, "in-
the-wild" databases, provided it is equipped with the necessary
semantic context. It provides a key finding: for many ambiguous,
real-world schemas, the critical step to unlock robust performance
is to move from schema-awareness tosemantics-awareness. Our
Schema Enrichment technique provides a practical, low-overhead
path to achieving this, representing a significant step towards truly
adaptable natural language interfaces.
6 DISCUSSION
Our experimental results do more than just validate our architec-
tural choices; they provide novel insights into the fundamental
challenges of building robust, generalizable natural language inter-
faces for databases. In this section, we synthesize our findings into
three key discussions. First, we propose a conceptual framework,
the "Hierarchy of Awareness," to organize the capabilities required
for a system to handle diverse databases. Second, we analyze the in-
herent challenge of ensuring reasoning consistency in LLM-driven
planning. Finally, we discuss the critical systems properties of pre-
dictability and controllability that emerged from our analysis of
the decision module.
6.1 The Hierarchy of Awareness: Schema,
Semantics, and Data
Our work empirically demonstrates a clear progression of capabili-
ties necessary for a system to robustly handle diverse databases, a
framework we term the Hierarchy of Awareness.Level 1: Schema-Awareness.Our RQ1 results provide defini-
tive, multi-benchmark evidence for the primacy of structured un-
derstanding. The catastrophic failure rate of the RAG baseline due
toSCHEMA_HALLUCINATION [14] confirms that treating a database
schema as an unstructured collection of text is an unreliable foun-
dation. In contrast, our SILE’s programmatic introspection provides
a grounded, high-fidelity context that nearly eliminates this failure
mode. This establishes that aschema-awarearchitecture is the
critical first step for any dependable Text-to-SQL system.
Level 2: Semantics-Awareness.While effective on well-struc-
tured benchmarks like Spider, the performance of our purely schema-
aware system on the more complex BIRD benchmark hinted at the
limits of this approach. Our RQ3 evaluation on the Olist database
confirmed this hypothesis decisively. The Olist schema’s high de-
gree of semantic ambiguity, particularly its implicit cross-lingual re-
lationships, caused the baseline SILE’s performance to collapse. This
led to our key architectural finding: to generalize to bespoke, real-
world databases, a system must be enhanced to becomesemantics-
aware. Our Schema Enrichment technique provides a practical,
low-overhead mechanism to achieve this, and the dramatic per-
formance recovery validates that bridging this ’semantic gap’ is a
necessary step for building truly adaptable systems.
Level 3: Data-Awareness (The Next Frontier).Finally, our
failure analysis on BIRD points to the ultimate frontier. The SILE’s
occasional struggles with queries requiring value-based linking
show that the final level of mastery is to becomedata-aware. This
requires the system to correctly map a data value mentioned in
the query to its corresponding column in the schema, a task that
often requires disambiguating between multiple potential columns.
Conquering this value-to-column mapping challenge, a core focus
of modern benchmarks like BIRD [ 18], is the next critical step for
the field.
6.2 Reasoning Consistency as a Core Challenge
A key finding from our RQ3 benchmark evaluation is the challenge
of ensuringreasoning consistencyin the schema linking process.
Our framework demonstrated a remarkable capability to handle
high structural complexity, successfully generating complex, multi-
hop joins involving up to six tables for several ’Extra Hard’ queries.
Paradoxically, the system exhibited failures on structurally simpler
queries that required a subset of the same relational paths, fail-
ing to include a necessary intermediate ’bridge’ table in its query
plan. This paradoxical result—succeeding on a harder task while
failing on a similar, easier one—is not a flaw in our system’s ar-
chitecture but rather exposes the inherentunpredictabilityof
the LLM’s zero-shot reasoning. This demonstrates that for Text-
to-SQL systems, raw capability is not the only metric of success;
reliability and consistency are paramount. Achieving deter-
ministic, correct reasoning across all variations of natural language
queries remains a critical open research problem, suggesting that
future work must focus on making reasoning processes more robust
and predictable through techniques like self-correction, a method
with both practical agentic implementations [ 33] and a growing
theoretical foundation [36].

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
6.3 Predictability and Controllability as
First-Class System Properties
Our analysis of the MMP’s Decision Module in RQ2 revealed a
fundamental trade-off in natural language interfaces: the conflict
between a system that tries to infer a user’s nuanced, hierarchi-
cal intent and one that acts as a predictable, literal-intent engine.
For a reliable database system, we argue thatpredictabilityis a
first-class architectural property. The logical consistency of a literal-
intent engine is the superior choice, as a system that does exactly
what it is told is a more trustworthy foundation than one that tries
to guess and frequently guesses wrong. Our work demonstrates
that through deliberate prompt engineering, we can design and
select for this critical property of logical consistency. This is not to
say that a specialist model like BERT [ 8] could never learn to handle
such nuance. Its failure in our OOD tests is likely a symptom of
the data scale on which it was fine-tuned (4,000 samples). A model
trained on a massive, web-scale dataset of queries and hierarchical
intents might eventually learn to generalize. However, this high-
lights a critical practical consideration: achieving robustness in a
specialist requires a significant, ongoing investment in data cura-
tion and re-training [ 32]. In contrast, the generalist LLM offers a
different paradigm: robustness throughcontrollability. Our work
demonstrates that through deliberate prompt engineering, we can
configure the generalist to act as a predictable, literal-intent engine.
This is a powerful feature: the system’s logical behavior is not an
emergent property of its training data, but an explicit, auditable
artifact of its prompt [ 3]. If a different logical behavior is desired
(e.g., prioritizing certain constraints), one can simplytweak the
prompt, rather than curating a new dataset and re-training a model.
For building reliable and maintainable database systems, we argue
that this explicit controllability is a crucial architectural advantage.
6.4 Limitations and Future Work
Our discussion has synthesized our findings into a set of core prin-
ciples and challenges for building robust natural language database
interfaces. These insights, in turn, define the limitations of our
current framework and chart a clear course for future work toward
truly "unbound" systems.
The Frontier of Data-Aware Linking.As our failure analysis
on BIRD [ 18] revealed, the challenge ofdata-awarelinking remains
a critical open problem. While our semantics-aware SILE is a signif-
icant step forward, its inability to link based on data values within
the query represents the next major frontier. This requires the
system to correctly map a data value (e.g., a city name) to its corre-
sponding column in the schema, often by disambiguating between
multiple possibilities. Future work should explore efficient and scal-
able techniques for value-based sampling and schema indexing to
bridge this gap, enabling systems to handle the full spectrum of
real-world query complexities.
Scalability of Multimodal Reasoning.The MMP’s per-record
reasoning loop, while powerful, represents a significant compu-
tational cost that is linear in the cardinality of the candidate set
returned by its initial structured filter. For low-selectivity queries,
this can become a prohibitive performance bottleneck. This limita-
tion highlights the need for a more sophisticated, multi-stage query
execution strategy that integrates principles from approximatequery processing. Our findings in RQ1 caution against using proba-
bilistic retrieval for the structurally-sensitive, zero-tolerance task
of schema linking, where a single omission can cause catastrophic
failure. However, the same technique is perfectly suited for the
recall-oriented task of candidate set pruning. Future work should
explore a cascaded filtering architecture where a "coarse-grained"
semantic filter (e.g., a pre-computed vector index) drastically re-
duces the candidate set before it is passed to the "fine-grained"
MMP for expensive, high-precision reasoning. This approach mir-
rors classic multi-stage query optimization, using cheap filters to
protect expensive operators, thereby enabling scalable semantic
querying over massive datasets.
Towards Intent-Aware Interaction.Finally, the "Literal vs.
Inferred Intent" dilemma highlighted by our RQ2 analysis under-
scores the need for more sophisticated,intent-awaresystems. The
current framework forces a choice between a predictable but rigid
literal-intent engine and a more flexible but unreliable model. The
ideal system should bridge this gap. A critical direction for future
research is the design of interactive architectures that can resolve
ambiguity through clarification dialogues [ 43] (e.g., "Is the 4.5 rating
a strict requirement or a preference?"). Developing the principles
and mechanisms for such interactive, intent-aware query refine-
ment is a crucial open problem for the community.
7 RELATED WORK
Our work is situated at the intersection of several active research
areas: the evolution of Text-to-SQL systems, the specific challenge
of schema linking, the emerging field of multimodal database query-
ing, and the trend towards agentic data analysis.
Text-to-SQL and Schema Linking.The challenge of schema
linking has long been recognized as the "crux" of the Text-to-SQL
problem [ 16]. Historically, this involved a trade-off between provid-
ing enough context and avoiding noise. However, with the advent
of large-context, powerful LLMs, a new paradigm is emerging that
challenges the necessity of traditional, external schema linking
filters. Maamari et al. [ 23] argue that for state-of-the-art models,
it is often safer and more effective to provide the full schema con-
text and trust the model’s internal reasoning to identify relevant
elements, rather than risk an imperfect filter removing essential
information.
Our work aligns with and formalizes this modern paradigm.
We conceptualize this "in-context linking" as a first-classquery
planningprimitive. Our Schema Introspection and Linking Engine
(SILE) is a direct implementation of this principle: it leverages an
LLM’s reasoning over the complete schema to produce a structured
query plan. Our contribution is twofold. First, we provide a con-
crete, reusable systems abstraction (the SILE) for this emerging
approach. Second, through our rigorous comparison against a RAG
baseline [ 17] in RQ1, we provide strong empirical evidence for
this paradigm, systematically demonstrating that older, unstruc-
tured filtering methods are architecturally prone to catastrophic
SCHEMA_HALLUCINATION failures. Thus, we argue not for the "death
of schema linking," but for its evolution from an external filtering
step into an integrated, LLM-driven planning phase.
Multimodal Database Querying.The vision of querying un-
structured data linked from relational tables is a growing frontier.

Aymane Hassini
Systems are emerging that follow two main architectural patterns.
The "embed-first" approach, exemplified by Symphony [ 7], pre-
builds a unified cross-modal vector index over all data assets and
uses an LLM to decompose queries for execution across modalities.
In contrast, the "database-first" approach of LOTUS [ 28] extends the
database engine with novel "semantic operators" that invoke LLMs,
leveraging a query optimizer to manage costs. Our work presents a
third, distinct architecture. Building on the "latent attribute" con-
cept from GIQ [ 26], DynaQuery’s MMP uses the relational database
itself as a powerful, on-the-fly filter via our "Ultimate Filtered Join"
pattern. This design prioritizes adaptability to arbitrary schemas
without reliance on pre-built, specialized indexes.
LLM Agents for Data Analysis.A prominent recent trend is
the development of LLM-powered agents for multi-step data analy-
sis, as evaluated by benchmarks like Spider 2.0 [ 15]. These systems,
such as RAISE [ 12], decompose complex analytical tasks into an
interactive, multi-turn loop of planning, tool use (e.g., SQL execu-
tion), and self-correction. This agentic paradigm is powerful but
orthogonal to our work. DynaQuery focuses on perfecting the foun-
dational, single-shot "NL-to-SQL" tool that these agents must rely
on. Our contributions to the robustness and predictability of this
core primitive—particularly by mitigating contextual failures like
SCHEMA_HALLUCINATION —can be seen as providing a more reliable
foundational tool for these next-generation agentic frameworks.
8 CONCLUSION
This paper establishes a foundational architectural principle for
natural language data interfaces: the treatment of schema linking is
a systems-level decision that dictates robustness and generalization.
We provided definitive, multi-benchmark evidence that the preva-
lent paradigm of unstructured RAG is architecturally brittle, prone
to catastrophic SCHEMA_HALLUCINATION failures. In response, we
introducedDynaQuery, a framework built on a core tenet: schema
linking must be a first-class, structure-aware query planning phase.
OurSILEprimitive embodies this principle, proving its superiority
by nearly eliminating these critical errors and establishing a pre-
dictable, robust foundation. Our work yields a conceptual roadmap
for the field—the“Hierarchy of Awareness”—charting the path
from the schema-awareness we demonstrate, to the semantics-
awareness required for generalization, and ultimately to the data-
awareness for truly “unbound” systems. By providing both a vali-
dated architectural blueprint and these core engineering principles,
we bridge the critical gap between the visionary goal of universal
data querying and the practical systems required to achieve it.
REFERENCES
[1] Ion Androutsopoulos, Graeme D. Ritchie, and Peter Thanisch. 1995. Natural Lan-
guage Interfaces to Databases – An Introduction.Natural Language Engineering
1, 1 (1995), 29–81.
[2]Michael Armbrust, Ali Ghodsi, Reynold Xin, and Matei Zaharia. 2021. Lake-
house: A New Generation of Open Platforms that Unify Data Warehousing and
Advanced Analytics. InProceedings of the 11th Annual Conference on Innovative
Data Systems Research (CIDR)(Online).
[3]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Ben-
jamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford,Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learn-
ers. InAdvances in Neural Information Processing Systems 33 (NeurIPS). Curran
Associates, Inc., 1877–1901.
[4] Salmane Chafik, Saad Ezzini, and Ismail Berrada. 2025. Enhancing Security in
Text-to-SQL Systems: A Novel Dataset and Agent-Based Framework.Natural
Language Engineering31 (2025), 1399–1422. https://doi.org/10.1017/nlp.2025.
10008
[5]Harrison Chase. 2022. LangChain. GitHub Repository. https://github.com/
langchain-ai/langchain
[6]Peter Baile Chen, Yi Zhang, and Dan Roth. 2024. Is Table Retrieval a Solved
Problem? Exploring Join-Aware Multi-Table Retrieval. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (ACL)
(Bangkok, Thailand). Association for Computational Linguistics, 2687–2699.
https://aclanthology.org/2024.acl-long.147
[7]Zui Chen, Zihui Gu, Lei Cao, Ju Fan, Samuel Madden, and Nan Tang. 2023.
Symphony: Towards Natural Language Query Answering over Multi-modal
Data Lakes. InProceedings of the 13th Annual Conference on Innovative Data
Systems Research (CIDR)(Amsterdam, The Netherlands).
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (NAACL-HLT)(Min-
neapolis, Minnesota). Association for Computational Linguistics, 4171–4186.
[9]Jeffrey Eben, Aitzaz Ahmad, and Stephen Lau. 2024. RASL: Retrieval Aug-
mented Schema Linking for Massive Database Text-to-SQL. arXiv preprint
arXiv:2407.23104. https://arxiv.org/abs/2407.23104
[10] Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and
Jingren Zhou. 2024. Text-to-SQL Empowered by Large Language Models: A
Benchmark Evaluation.Proc. VLDB Endow.17, 5 (2024), 1132–1145. https:
//doi.org/10.14778/3641204.3641221
[11] Google. 2025. Gemini 2.5 Pro Model Documentation. Google Cloud Documenta-
tion. https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-
pro Accessed on 2025-10-18.
[12] Fernando Granado, Roberto Lotufo, and Jayr Pereira. 2025. RAISE: Reasoning
Agent for Interactive SQL Exploration. arXiv preprint arXiv:2506.01273. (2025).
arXiv:2506.01273 [cs.AI] https://arxiv.org/abs/2506.01273
[13] George Katsogiannis-Meimarakis and Georgia Koutrika. 2023. A Survey on Deep
Learning Approaches for text-to-SQL.The VLDB Journal32, 4 (2023), 905–936.
https://doi.org/10.1007/s00778-022-00776-8
[14] Mayank Kothyari, Dhruva Dhingra, Sunita Sarawagi, and Soumen Chakrabarti.
2023. CRUSH4SQL: Collective Retrieval Using Schema Hallucination For
Text2SQL. arXiv preprint arXiv:2311.01173. https://arxiv.org/abs/2311.01173
[15] Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin
Su, Zhaoqing Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, Victor Zhong,
Caiming Xiong, Ruoxi Sun, Qian Liu, Sida I. Wang, and Tao Yu. 2025. SPIDER 2.0:
Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows.
InInternational Conference on Learning Representations (ICLR). https://arxiv.org/
abs/2411.07763
[16] Wenqiang Lei, Weixin Wang, Zhixin Ma, Tian Gan, Wei Lu, Min-Yen Kan, and
Tat-Seng Chua. 2020. Re-examining the Role of Schema Linking in Text-to-SQL.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP)(Online). Association for Computational Linguistics, 6943–
6954.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems 33. Curran Associates, Inc., 9459–9474.
[18] Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang,
Bowen Qin, Ruiying Geng, Nan Huo, Wenyu Du, Yexiang Zhai, Chao Yan,
Chenxi Li, Fuxuan Wei, Tianyu Zhao, Yitao Nian, Yuxiang Tan, Zhenting Wang,
Ziyan Zhang, Yixin Ou, Libo Lang, Hongyu Lin, Yao Yu, Jian-Guang Lou, Wei-
Wei Tu, and Rong Xiao. 2023. Can LLM Already Serve as a Database Inter-
face? A Big Bench for Large-Scale Database Grounded Text-to-SQLs. InAd-
vances in Neural Information Processing Systems 36 (NeurIPS). Curran Associates,
Inc., 13112–13141. https://proceedings.neurips.cc/paper_files/paper/2023/file/
83fc8fab1710363050bbd1d4b8cc0021-Paper-Datasets_and_Benchmarks.pdf
[19] Mario Linares-Vásquez, Boyang Li, Christopher Vendome, and Denys Poshy-
vanyk. 2016. Documenting Database Usages and Schema Constraints in Database-
Centric Applications. InProceedings of the 25th International Symposium on
Software Testing and Analysis (ISSTA)(Saarbrücken, Germany). ACM, 270–281.
https://doi.org/10.1145/2931037.2931072
[20] Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju
Fan, Guoliang Li, Nan Tang, and Yuyu Luo. 2024. A Survey of Text-to-SQL
in the Era of LLMs: Where are we, and where are we going? arXiv preprint
arXiv:2408.05109. arXiv:2408.05109 [cs.DB] https://arxiv.org/abs/2408.05109
[21] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization.
InInternational Conference on Learning Representations (ICLR)(New Orleans, LA,

DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data
USA). https://openreview.net/forum?id=Bkg6RiCqY7
[22] Yuyu Luo, Guoliang Li, Ju Fan, Chengliang Chai, and Nan Tang. 2025. Natural
Language to SQL: State of the Art and Open Problems.Proc. VLDB Endow.18, 12
(2025), 5466–5471. https://doi.org/10.14778/3750601.3750696
[23] Karime Maamari, Fadhil Abubaker, Daniel Jaroslawicz, and Amine Mhedhbi.
2024. The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned
Language Models. arXiv preprint arXiv:2408.07702. https://arxiv.org/abs/2408.
07702
[24] Samuel Madden, Michael Cafarella, Michael J. Franklin, and Tim Kraska. 2024.
Databases Unbound: Querying All of the World’s Bytes with AI.Proc. VLDB
Endow.17, 12 (2024), 4546–4554. https://doi.org/10.14778/3685800.3685916
[25] Toby Mao. 2020. sqlglot: A no-dependency SQL parser, transpiler, and optimizer.
GitHub Repository. https://github.com/tobymao/sqlglot Accessed on 2025-10-18.
[26] Mehdi Nejjar, Aymane Hassini, and Yousra Chtouki. 2025. Generating Impossible
Queries.Machine Learning with Applications21 (2025), 100677. https://doi.org/
10.1016/j.mlwa.2025.100677
[27] Olist and André Sionek. 2018. Brazilian E-Commerce Public Dataset by Olist.
Kaggle Dataset. https://doi.org/10.34740/KAGGLE/DSV/195341
[28] Liana Patel, Siddharth Jha, Carlos Guestrin, and Matei Zaharia. 2024. LOTUS: En-
abling Semantic Queries with LLMs Over Tables of Unstructured and Structured
Data. arXiv preprint arXiv:2407.11418. https://arxiv.org/abs/2407.11418
[29] Mohammadreza Pourreza and Davood Rafiei. 2023. DIN-SQL: Decomposed In-
Context Learning of Text-to-SQL with Self-Correction. InAdvances in Neural
Information Processing Systems 36 (NeurIPS). Curran Associates, Inc.
[30] Mohammadreza Pourreza and Davood Rafiei. 2023. Evaluating Cross-Domain
Text-to-SQL Models and Benchmarks. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing (EMNLP)(Singapore). Associ-
ation for Computational Linguistics, 1601–1611.
[31] Federico Ranaldi, Elena Sofia Ruzzetti, Dario Onorati, Leonardo Ranaldi, Cristina
Giannone, Andrea Favalli, Raniero Romagnoli, and Fabio Massimo Zanzotto.
2024. Investigating the Impact of Data Contamination of Large Language Models
in Text-to-SQL Translation. arXiv preprint arXiv:2402.08100. https://arxiv.org/
abs/2402.08100
[32] D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar
Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, and Dan Denni-
son. 2015. Hidden Technical Debt in Machine Learning Systems. InAdvances
in Neural Information Processing Systems 28 (NeurIPS). Curran Associates, Inc.,
2503–2511.
[33] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Re-
inforcement Learning. arXiv preprint arXiv:2303.11366. arXiv:2303.11366 [cs.AI]
https://arxiv.org/abs/2303.11366
[34] Yasufumi Taniguchi, Hiroki Nakayama, Kubo Takahiro, and Jun Suzuki. 2021.
An Investigation Between Schema Linking and Text-to-SQL Performance. arXiv
preprint arXiv:2102.01847. https://arxiv.org/abs/2102.01847
[35] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu,
Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng Yin, Changze
Lv, Xiaoqing Zheng, and Xuanjing Huang. 2024. Searching for Best Practices
in Retrieval-Augmented Generation. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing (EMNLP)(Miami, Florida).
Association for Computational Linguistics, 17716–17736.
[36] Yifei Wang, Yuyang Wu, Zeming Wei, Stefanie Jegelka, and Yisen Wang. 2024. A
Theoretical Understanding of Self-Correction through In-context Alignment. In
Advances in Neural Information Processing Systems 38 (NeurIPS). Curran Asso-
ciates, Inc.
[37] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Rea-
soning in Large Language Models. InAdvances in Neural Information Processing
Systems 35 (NeurIPS). Curran Associates, Inc., 24824–24837.
[38] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement De-
langue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,
Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest,
and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language
Processing. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP): System Demonstrations(Online). Association for
Computational Linguistics, 38–45. https://aclanthology.org/2020.emnlp-demos.6
[39] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and
Jian-Yun Nie. 2024. C-Pack: Packed Resources For General Chinese Embeddings.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’24)(Washington, DC, USA). ACM,
1132–1141. https://doi.org/10.1145/3626772.3657878
[40] Yifei Yang, Hangdong Zhao, Xiangyao Yu, and Paraschos Koutris. 2023. Pred-
icate Transfer: Efficient Pre-Filtering on Multi-Join Queries. arXiv preprint
arXiv:2307.15255. https://arxiv.org/abs/2307.15255
[41] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li,
James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir
Radev. 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex andCross-Domain Semantic Parsing and Text-to-SQL Task. InProceedings of the
2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Association for Computational Linguistics, 3911–3921.
[42] Lu Zeng, Sree Hari Krishnan Parthasarathi, and Dilek Hakkani-Tür. 2023. N-
Best Hypotheses Reranking for Text-to-SQL Systems. InProceedings of the IEEE
International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
(Rhodes Island, Greece). IEEE, 1–5.
[43] Fuheng Zhao, Shaleen Deep, Fotis Psallidas, Avrilia Floratou, Divyakant Agrawal,
and Amr El Abbadi. 2024. Sphinteract: Resolving Ambiguities in NL2SQL
Through User Interaction.Proc. VLDB Endow.18, 4 (2024), 1145–1158. https:
//doi.org/10.14778/3717755.3717772