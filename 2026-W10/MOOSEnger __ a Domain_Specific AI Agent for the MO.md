# MOOSEnger -- a Domain-Specific AI Agent for the MOOSE Ecosystem

**Authors**: Mengnan Li, Jason Miller, Zachary Prince, Alexander Lindsay, Cody Permann

**Published**: 2026-03-05 03:06:06

**PDF URL**: [https://arxiv.org/pdf/2603.04756v1](https://arxiv.org/pdf/2603.04756v1)

## Abstract
MOOSEnger is a tool-enabled AI agent tailored to the Multiphysics Object-Oriented Simulation Environment (MOOSE). MOOSE cases are specified in HIT ".i" input files; the large object catalog and strict syntax make initial setup and debugging slow. MOOSEnger offers a conversational workflow that turns natural-language intent into runnable inputs by combining retrieval-augmented generation over curated docs/examples with deterministic, MOOSE-aware parsing, validation, and execution tools. A core-plus-domain architecture separates reusable agent infrastructure (configuration, registries, tool dispatch, retrieval services, persistence, and evaluation) from a MOOSE plugin that adds HIT-based parsing, syntax-preserving ingestion of input files, and domain-specific utilities for input repair and checking. An input precheck pipeline removes hidden formatting artifacts, fixes malformed HIT structure with a bounded grammar-constrained loop, and resolves invalid object types via similarity search over an application syntax registry. Inputs are then validated and optionally smoke-tested with the MOOSE runtime in the loop via an MCP-backed execution backend (with local fallback), translating solver diagnostics into iterative verify-and-correct updates. Built-in evaluation reports RAG metrics (faithfulness, relevancy, context precision/recall) and end-to-end success by actual execution. On a 125-prompt benchmark spanning diffusion, transient heat conduction, solid mechanics, porous flow, and incompressible Navier--Stokes, MOOSEnger achieves a 0.93 execution pass rate versus 0.08 for an LLM-only baseline.

## Full Text


<!-- PDF content starts -->

Highlights
MOOSEnger — a Domain-Specific AI Agent for MOOSE Ecosys-
tem
Mengnan Li, Jason Miller, Zachary Prince, Alexander Lindsay, Cody Per-
mann
•MOOSEnger turns natural-language simulation requests into runnable
MOOSE input files by combining RAG with MOOSE-aware tools for
parsing, validation, and execution.
•A deterministic input-precheck pipeline sanitizes formatting artifacts,
repairs malformed HIT structure with bounded grammar-constrained
edits, andcorrectsinvalidobject/typenamesusingcontext-conditioned
similarity search against the application syntax registry.
•On a 125-prompt benchmark spanning five MOOSE physics families,
placing the MOOSE runtime “in the loop” for smoke tests and iterative
correction raisestheexecution pass rateto 0.93 versus0.08 for an LLM-
only baseline.arXiv:2603.04756v1  [cs.AI]  5 Mar 2026

MOOSEnger — a Domain-Specific AI Agent for
MOOSE Ecosystem
Mengnan Lia, Jason Millera, Zachary Princea, Alexander Lindsaya, Cody
Permanna
aIdaho National Laboratory, 995 MK Simpson Blvd, Idaho Falls, 83401, ID, USA
Abstract
MOOSEnger is a tool-enabled, domain-specific AI agent for the Multiphysics
Object-Oriented Simulation Environment (MOOSE) ecosystem. MOOSE
simulations are authored as “.i” input files in Hierarchical Input Text (HIT)
syntax; while powerful, the breadth of available objects and strict format-
ting rules make first-time setup and debugging time-consuming. MOOSEn-
ger provides a conversational interface that translates natural language in-
tent into runnable MOOSE configurations by combining Retrieval Augment
Generation(RAG)overcurateddocumentation/exampleswithdeterministic,
MOOSE-aware validation and execution tools. A core-plus-domain architec-
ture separates reusable agent infrastructure (configuration, registries, tool
dispatch, retrieval services, persistence, and evaluation) from a MOOSE plu-
gin that adds HIT-based parsing, syntax-preserving ingestion of input files,
and domain-specific utilities for input repair and checking. To reduce brit-
tle one-shot generation, MOOSEnger introduces an input-precheck pipeline
that sanitizes hidden formatting artifacts, repairs malformed HIT structure
using a bounded, grammar-constrained loop, and corrects invalid object type
names via context-conditioned similarity search against an application syn-
tax registry. The system then validates and optionally smoke-tests input files
by placing the MOOSE runtime “in the loop” through an Model Context
Protocol (MCP)-backed execution backend with a local fallback, converting
solverdiagnosticsintoiterative“verify-and-correct” updates. Wealsoprovide
built-in evaluation that measures both RAG quality (faithfulness, answer rel-
evancy, context precision/recall) and end-to-end agent success via actual ex-
Email address:mengnan.li@inl.gov(Mengnan Li)

ecution. Across a 125-prompt benchmark spanning diffusion, transient heat
conduction, solid mechanics, porous flow, and incompressible Navier–Stokes,
MOOSEnger achieves a 0.93 execution pass rate versus 0.08 for an Large
Language Model (LLM)-only baseline, demonstrating substantial reliability
gains for multiphysics case authoring.
Keywords:
Large language model, MOOSE, Multiphysics, RAG, Agentic workflow
1. Introduction
1.1. MOOSE Ecosystem
Multiphysics Object-Oriented Simulation Environment (MOOSE) is a
high-performance computational framework for multiphysics simulations de-
veloped at Idaho National Laboratory[1]. It provides a plug-in architecture
that lets users specify partial differential equations, boundary conditions,
and material models at a high level, without requiring them to directly man-
age the underlying parallel solvers. By leveraging well-defined interfaces and
modular physics components, MOOSE promotes code reuse and enables cou-
pling across multiple physics in a single simulation workflow. The framework
is designed for demanding scientific and engineering applications, supporting
features such as massive parallelism, adaptive mesh refinement, and multi-
scale coupling through sub-applications, making it well suited to complex
problems in areas such as reactor physics [2], thermal fluids [3], and struc-
tural mechanics [4]. In addition to the core framework, MOOSE has grown
into a broader MOOSE ecosystem: a shared infrastructure of physics mod-
ules [5, 6, 7, 8, 9], documentation/tooling, and a large collection of MOOSE-
based applications used across nuclear engineering and other multiphysics
domains.
A central design choice in MOOSE is the decomposition of weak-form
residual equations into discrete, reusable compute kernels. This kernel-based
structure allows users and developers to extend or modify physics capabili-
ties with minimal intrusion, enabling new physics to be integrated and com-
posed without recompiling the entire code base for each modeling change.
In practice, MOOSE realizes this flexibility through a Domain Specific Lan-
guage (DSL) expressed as human-readable input files (with a “.i” extension).
These input files use a nested, block-structured syntax (Hierarchical Input
Text (HIT) format) to configure simulations. Each input typically contains
2

essential top-level blocks (e.g.,[Mesh],[Variables],[Kernels],[BCs],
[Executioner],[Outputs], etc.) and nested sub-blocks with parameters,
allowinguserstodefinemeshes, governingphysics, materialproperties, solver
settings, and output controls without modifying the underlying C++ source
code. While this input-driven design significantly accelerates model develop-
mentanditeration, thebreadthofoptionsandstrictsyntaxalsocontributeto
a steep learning curve, making correct simulation setup and troubleshooting
non-trivial, especially for new users.
1.2. Challenges in Multiphysics Modeling and Simulation Workflow
Solving multiphysics problems with MOOSE-based applications is inher-
ently iterative: users repeatedly refine model assumptions, discretization
choices, solver settings, and post-processing until the results are both nu-
merically stable and physically credible. In this workflow, small mistakes or
suboptimal choices made early can propagate into later stages, increasing the
cost of debugging and slowing progress toward a usable baseline model.
The first challenge isinput authoring, where users must translate phys-
ical intent into MOOSE DSL that is syntactically valid and semantically
consistent with available objects and solver requirements. Minor formatting
or parameter errors can prevent execution entirely, while more subtle mis-
configurations may run successfully yet yield incorrect physics or misleading
results. Closely tied to authoring isnumerical convergence and stability: se-
lecting appropriate meshes, discretizations, and solver/time-integration pa-
rameters is rarely straightforward. Inappropriate mesh resolution, poorly
scaled formulations, or ill-conditioned discretizations can lead to instability,
non-convergence, or excessive computational cost. When runs fail,execu-
tion and debuggingbecome the next hurdle; users must interpret console
output and logs to identify failure modes (e.g., nonlinear/linear solver diver-
gence, unstable time stepping, poorly scaled residuals). Although MOOSE
provides detailed diagnostics, turning those messages into effective correc-
tive actions often requires specific domain expertise. Finally, even when
simulations complete,post-processing and interpretationremain non-trivial:
extracting actionable insights from large output datasets (e.g., field distri-
butions, derived quantities, and time histories) can be time-consuming and
error-prone without a well-defined analysis workflow.
Taken together, these challenges create friction precisely where many
users need the most support: reaching a first valid run and establishing a
trustworthy baseline model. New users may struggle to identify the right
3

configuration patterns and debugging steps, while experienced users still
spend significant time on repetitive setup, troubleshooting, and documen-
tation lookup. This motivates intelligent assistance that can (i) connect user
intent to correct configuration patterns, (ii) catch and repair common issues
early, (iii) guide convergence- and stability-oriented iteration, and (iv) reduce
the overhead of navigating fragmented documentation, examples, and prior
cases.
1.3. Artificial Intelligence (AI) Agent for Scientific Computing and Motiva-
tion for MOOSEnger
AgrowingbodyofworkexploresAI/MachineLearning(ML)—andinpar-
ticular Large Language Model (LLM)-based agents—as workflow assistants
for scientific computing software. In computational fluid dynamics, systems
such as MetaOpenFOAM[10] and OpenFOAM-GPT[11] translate natural-
language problem descriptions into runnable OpenFOAM cases, automating
parts of case setup and iteration. In radiation transport, AutoFLUKA[12]
applies an agent-based pipeline to streamline Monte Carlo input preparation
and execution. A common technical pattern across these efforts is Retrieval
Augment Generation (RAG): domain documentation, examples, and prior
artifacts are indexed and selectively retrieved to ground responses, improv-
ing factuality and reducing hallucinations.
In parallel, agentic coding assistants such as Anthropic’s Claude Code[13]
and OpenAI’s Codex[14] demonstrate a practical interaction model in which
an assistant translates intent into working artifacts by iterating directly on
project assets (files, tests, and tool calls). This suggests an analogous direc-
tion for multiphysics simulation: enabling users to express goals in natural
language while the assistant drafts, validates, executes, and iteratively im-
proves concrete simulation artifacts.
Despite promising progress, robust AI assistance for the MOOSE ecosys-
tem remains comparatively nascent, particularly when the assistant must
tightly couple language-based reasoning with concrete simulation actions
such as input generation, schema validation, execution, and result inspec-
tion. MooseAgent[15] advances in this direction by introducing a multi-agent
framework for automating MOOSE simulations with knowledge retrieval and
iterative input repair; however, much of its domain grounding is mediated
primarily through RAG and prompting, and practical usability still depends
on how directly the assistant can incorporate MOOSE-specific structure and
tool feedback into the loop.
4

MOOSEnger builds on these ideas with an emphasis on deeper MOOSE-
aware integration and an interactive, tool-enabled workflow geared toward
practical engineering use. Rather than relying on fully autonomous multi-
agent orchestration, MOOSEnger adopts a conversational single-agent design
under user oversight: it retrieves grounded context, generates and refines
configuration artifacts, and invokes simulation tools to validate and iterate.
Concretely, it leverages MOOSE-specific structure throughout the loop: it
uses thehitparser to preserve block-level semantics during input handling,
applies an input precheck stage (including syntax/grammar and object va-
lidity checks), and supports iterative error-correction driven by feedback di-
rectlyfromtheMOOSEexecutable. Thisdesigntargetsrecurringbottlenecks
in real workflows—accelerating the time-to-first valid run, improving trou-
bleshooting efficiency, and reducing the overhead of navigating fragmented
documentation, examples, and prior input files.
Our longer-term vision is “vibe-authoring” for MOOSE modeling: users
describe the physics and objectives, and MOOSEnger helps locate relevant
guidance, drafts a correct configuration, and iterates through validation and
fixes until a first successful run is achieved. This aims to lower the entry bar-
rierfornewcomerswhilealsoacceleratingexpertworkflowsbyreducingrepet-
itive setup and debugging, allowing engineers to focus more on analysis and
exploration than on manual input preparation. To keep this assistance de-
pendable as models, documentation, and workflows evolve, MOOSEnger also
emphasizes quality control and continuous improvement: built-in evaluation
hooksmakeitpossibletotrackretrievalandassistanceperformanceovertime
and detect regressions as the underlying knowledge base changes. Finally,
thesamefoundationsupportsreusable,shareableorganizationalknowledge—
an open framework by default, with an option to distribute curated/hosted
knowledge—so teams can standardize workflows, reduce training overhead,
and accelerate simulation setup and debugging across local workstations,
High Performance Computing (HPC) environments, and cloud deployments.
1.4. Technical Highlights
This paper presents MOOSEnger, an AI agent tightly integrated with the
MOOSE simulation framework. Key contributions include:
•MOOSE-aware assistant that converts high-level requests into valid
MOOSE input files, grounded with RAG over documentation and ex-
amples.
5

•Deterministic toolchain for MOOSE DSL authoring: HIT parsing, san-
itation, grammar-constrained repair, and syntax-registry type correc-
tion.
•Performance evaluation of the agent’s capabilities, including metrics
for response quality and context retrieval (faithfulness, relevancy, pre-
cision/recall), demonstrated on curated simulation queries.
•Extensible core–domain architecture: a reusable core (configuration,
retrieval, tooling, agent runtime) with domain plugins for MOOSE-
specific prompts, parsers, tools, and import pipelines, enabling rapid
development of new MOOSE-based application agents and multi-agent
workflows for complex reactor multiphysics.
2. End-to-End Workflow Overview
Figure 1 summarizes the end-to-end workflow of MOOSEnger. The sys-
tem separatesoffline knowledge ingestion(performed periodically as doc-
umentation and examples evolve) frominteractive inference(performed at
runtime in response to user requests). This separation keeps runtime inter-
action lightweight while allowing the retrieval corpus to be refreshed inde-
pendently.
Offline data ingestion (build/update the retrieval corpus).MOOSEnger in-
gests heterogeneous sources such as MOOSE documentation, community dis-
cussions, and example input files. An import pipeline converts these sources
into retrieval-ready units by (i) performing parent–child chunking for prose
documents, (ii) parsing “.i” inputs with a HIT-aware block parser to preserve
syntax-block boundaries, and (iii) attaching structured tags/metadata (e.g.,
source provenance and syntax-derived attributes). The resulting documents
and chunks are embedded and stored in a vector database that serves as the
MOOSE knowledge base used during retrieval.
Interactive inference (tool-augmented authoring loop).At runtime, the user
issues a natural-language physics request through the Command-Line In-
terface (CLI). The CLI performsruntime assemblyby selecting the active
domain profile, binding the allowed toolset, and constructing the shared con-
text (workspace paths, session state, and any domain resources). The agent
then iterates in a conversational loop: it retrieves relevant guidance from
6

Figure 1: MOOSEnger workflow.Left:offline ingestion imports heterogeneous MOOSE
sources into a vector store.Right:at runtime, the CLI assembles a domain-configured,
tool-enabled agent that retrieves context, drafts/edits a MOOSE input file, prechecks it,
and optionally validates/executes it to provide grounded feedback and runnable artifacts.
the knowledge base, drafts or modifies a MOOSE input file, and runs an
automated precheck stage to mitigate runtime errors. When requested, the
agent validates and executes the checked input via a MOOSE backend (local
execution or an Model Context Protocol (MCP)-exposed executable). Tool
outputs (errors, logs, and result artifacts) are fed back into the agent to drive
targeted revisions, and the final response returns a checked/runnable input
file together with a concise summary of the modeling choices and outcomes.
Connection to the detailed methodology.The remainder of this section un-
packstheworkflowcomponentsindetail: theMOOSE-awareingestionpipeline
that preserves “.i” structure for retrieval, the tool-based input validation/re-
pair loop that turns solver feedback into actionable corrections, and the run-
time infrastructure (workspace, session state, and domain tooling) that keeps
multi-turn authoring reproducible and auditable.
3. Methodology
3.1. Core System Architecture
MOOSEnger follows acore-plus-domainarchitecture that separates sta-
ble agent infrastructure from domain-specific knowledge and capabilities.
7

Thecore layerprovides the reusable runtime substrate—configuration, reg-
istries, tool dispatch, retrieval services, and persistence—whiledomain plu-
ginscontribute prompt packs, agent profiles, tools, skill/playbook bundles,
and import/ingestion pipelines. A single plugin interface is the integration
point: domains register assets through uniform hooks, so new application
domains can be added without modifying core code.
Domain discovery, plugin loading, and registries.On startup, the plugin
loader discovers domain modules (built-in or via entry points) and invokes
their registration hooks. These hooks populate shared registries with both
core and domain assets, including (i) prompt packs, (ii) agent profiles, (iii)
tools, (iv) skills, and (v) import pipelines. This registry-based design decou-
ples discovery from execution: the core runtime remains agnostic to which
domain is active beyond selecting a profile that references registered assets.
Runtime assembly: profiles, tools, and runtime resources.For each run, the
CLI resolves anagent profile, which serves as the configuration contract. The
profile selects a prompt pack (system instructions and optional exemplars),
declares an allow-list of tools, and sets runtime parameters such as iteration
limits and safety policies. The runtime then binds the selected tools from
the registry and constructs a sharedToolContextcontaining configuration
options, workspace paths, session state, and any domain resources required
by tools or retrieval (e.g., indices or executors). In agent mode, the runtime
also prepares the agent-facing resources. It will expose during the loop:
a guarded project-local working filesystem (mounted under paths such as
./workspace), a set of persistent memory files to include in context, and
a skills directory (mounted under./skills) to make reusable procedures
discoverable during execution.
Agent2Agent (A2A) protocol server.In addition to the interactive CLI entry
point, MOOSEnger can be deployed as an optional Agent2Agent protocol
(A2A) JSON-RPC service to support remote clients and multi-agent orches-
tration. The A2A server is implemented as a thin transport layer around the
same runtime assembly path used by the CLI: it loads the selected agent pro-
file, binds the same allow-listed tools, and reuses the same workspace/session
services so that local and remote interactions share consistent behavior and
traceability. Theserverexposesastandard./.well-known/agent-card.json
endpoint for capability discovery and a JSON-RPC endpoint that accepts
message/sendrequests. During execution, the server streams incremental
8

status updates and returns generated simulation artifacts as structured task
artifacts; in our current implementation, the latest drafted MOOSE input
is emitted as a canonicalcurrent.iartifact to simplify downstream persis-
tence, diffs, and hand-off to external execution backends. This optional A2A
interface enables richer front-ends (e.g., IDE/GUI integrations) and coordi-
nation with other specialized agents without requiring changes to the core
agent loop.
Filesystem management.MOOSEnger maintains a project-local filesystem
rooted at.moosenger_fs/with a stable layout for agent-authored artifacts
and iterative authoring. The canonical MOOSE input under construction
is stored atworkspace/inputs/current.i, with run outputs placed under
workspace/runs/. The filesystem interface is policy-enforced to reduce risk:
writes are restricted to the workspace prefix, and reads of common secret
locations (e.g.,.env,.ssh, SSH keys) are denied. Optionally, the repository
root can be mounted read-only (e.g., as./project) so the agent can inspect
code and examples without enabling mutation.
Memory management.Persistent memory is treated as a first-class input
to the agent loop. MOOSEnger always includesAGENTS.mdas a memory
source to capture workflow contracts and project conventions, and it can ad-
ditionally mount user-level preferences (e.g.,memories/user_prefs.mdfrom
~/.moosenger/memories/) when available. WhenAGENTS.mdis missing or
empty, MOOSEnger seeds it with default contracts, any domain-provided
contracts, and the resolved instruction prompt (project override→prompt
pack→domain default→built-in fallback). Interactive resets clear the con-
versationthreadbutpreservefilesystemandmemorystate; memoryisscoped
by a per-thread identifier so multi-turn workflows remain reproducible with-
out conflating independent threads.
Skill library and seeding.Skills are reusable, task-oriented procedures loaded
from/skillsandbackedby.moosenger_fs/skills/. Domainpluginsmay
register additional skill paths and seed initial skill packs into the project
on first run. Seeding uses explicit policies (merge,overwrite, orskip)
to support shared updates while preserving local customization, enabling
domains to ship evolving playbooks without breaking user-edited skills.
Tool binding, execution loop, and artifact capture.Tools are registered as
modular functions described by aToolSpec(name, schema, description, im-
plementation) and instantiated from both core (e.g., retrieval helpers) and
9

the active domain (e.g., simulation actions). In agent mode, MOOSEnger
filters tools that would conflict with runtime-managed filesystem operations,
then exposes the remaining tool set to the agent loop. During interaction,
the agent emits structured tool calls; the runtime executes the correspond-
ing tool with workspace-aware logging and returns results to the LLM. Tool
outputs—includingrepairedinputs, logs, meshes, andrunproducts—areper-
sisted as artifacts and recorded inSessionState(IDs, paths, provenance)
so they can be referenced, reused, and audited across turns.
Retrieval and LLM-assisted tagging.To improve grounding and navigability
of context, the core runtime includes a retrieval stack and an automatic tag-
ginglayer. Thetaggingpre-processor(ContextManager.pre_processor)in-
vokes a tagging prompt (resolved byPromptManagerwith model-specific fall-
backs)toproduceastrictJSONschema(e.g.,document_topics,intent_type,
language_code,content_type). Tagsarestoredalongsideimportedcontent
and interactions to enable filtered, topic-aware retrieval. TheRAGManager
then performs parent/child indexing and hybrid retrieval to supply grounded
context to the agent and tools.
Workspace and session persistence.Forreproducibilityandevaluation,MOOSEn-
germaintainsaper-sessionworkspacedirectory(e.g.,.moosenger_workspace/
<session_id>/) that records inputs, outputs, logs, and tool artifacts. A per-
sistentSessionStatestored alongside the workspace tracks artifact meta-
data and a lightweight tool-call history, enabling resumption and auditability
during debugging and benchmarking. Conceptually, the session workspace
provides run-scoped traceability, while the project-local agent filesystem pro-
vides a stable working set (e.g.,current.i) for iterative authoring across
turns.
Key modules.
•Plugin loader + registries:discover domains and aggregate prompt
packs, profiles, tools, skills, and import pipelines.
•PromptManager:resolvesinstructionandtaggingpromptswithmodel-
specific fallbacks.
•Runtimeresources:guardedworkspacefilesystem, mountedmemory
sources, and the skills library.
10

•Tool runtime:ToolSpec-based registration, backend-specific filter-
ing, execution, and artifact logging.
•Tagging + RAG manager:structured metadata tagging, paren-
t/child indexing, and hybrid retrieval for grounded context.
•Workspace + SessionState:persistent run traces, artifact prove-
nance, and resumable multi-turn workflows.
MOOSE-specific extensions as a domain plugin.The MOOSE domain plugin
demonstrates how domain capabilities compose cleanly on top of the core:
apyhit-based input parser/loader that preserves block boundaries and at-
taches syntax-tree metadata; document ingestion utilities (e.g., HTML-to-
Markdown conversion that retains equations and code); syntax-aware type
similarity search to correct invalid object names; grammar-constrained in-
put repair; and a composite input-precheck workflow that chains sanitation,
repair, validation, and optional execution. Execution is abstracted behind
MCP-backed tools with a local MOOSE fallback, enabling the same agent
loop to validate, mesh, or run inputs without changing the core runtime. The
overview architecture design is shown in Fig. 2
3.1.1. MOOSE-aware Data Ingestion
A central requirement for MOOSEnger is to ingest heterogeneous knowl-
edge sources (documentation pages, discussions, and example inputs) while
preserving the structural semantics of MOOSE input files. Unlike prose doc-
uments, MOOSE “.i” files are hierarchical programs written in HIT syn-
tax; native text splitting can fracture block boundaries, disconnect param-
eters from their owning objects, and degrade retrieval quality. To address
this, MOOSEnger implements a MOOSE-aware ingestion pipeline that (1)
performsparent–childchunking at the document level, (2) usespyhitto
chunk input files by syntax blocks so that each[Block] ...[]unit is pre-
served, and(3)automaticallygeneratesrichmetadata—includingsyntax-tree
annotations—to support downstream retrieval and summarization.
Parent–child chunking across modalities.Forgeneraltextsources(e.g.,Mark-
down/HTML/PDF),theingestionworkflowfollowsastandardrecursivestrat-
egy: a file is ingested as aparentdocument (coarse, file-level context), then
split intochildchunks (fine-grained retrieval units) using configurable split-
ters. This allows retrieval to return precise snippets while retaining a link
11

Figure 2: Core-plus-domain architecture of MOOSEnger. Domain plugins provide prompt
packs, agent profiles, tools, import pipelines, and configuration, which the plugin loader
registers into core registries. Via the CLI or optional A2A server, the agent orchestra-
tor runs an iterative LLM–tools–context loop using the prompt/context managers, RAG
manager (vector store), workspace/session services, and runtime resources, with external
backends for LLM inference and optional MOOSE execution (MCP/local).
back to the full source context. For MOOSE input files, however, we by-
pass natural-language splitters entirely and instead produce parent and child
documents directly from the MOOSE syntax tree, ensuring that structural
boundaries are respected.
Syntax-preserving chunking with pyhit.For “.i” inputs, MOOSEnger inte-
gratesmoose-pyhit(pyhit) into the input ingestion pipeline.pyhitparses
the input into a hierarchical tree of HIT blocks; we then emit: (i) a single
parentdocument representing the full input file, and (ii) onechilddocument
per block node. Each child chunk contains the verbatim text of exactly one
MOOSE syntax block (including its nested content), preventing cross-block
contamination during embedding and retrieval. This yields block-aligned re-
trieval units that correspond to the way MOOSE users reason about models
(e.g.,[Mesh],[Kernels],[BCs]).
Automatic metadata generation from the syntax tree.Every emitted docu-
ment includes machine-readable metadata that captures both provenance
and structure. At minimum, we record file identity and parent–child rela-
tionships (e.g.,document_idandparent_id) so downstream components
12

can reconstruct the hierarchy. For each block-level child, we attach syntax-
derived fields such as the block name/path and discovered parameters; op-
tionally, we include sanitizedpyhitmetadata (e.g., line/column ranges, raw
parameter subtrees, and child paths) to enable richer filtering, debugging,
and provenance tracking during retrieval. In the ingestion CLI, enabling the
syntax-tree option propagates this per-block information under a dedicated
metadata tag, and debug mode records the parsed tree and the resulting
documents for inspection.
MOOSE object awareness and type extraction.Beyond structural metadata,
weextracttypeusagefromtheparsedinputtoidentifytheconcreteMOOSE
objectsinstantiatedbyeachsubsystem(e.g.,specificKernelorMeshGenerator
classes). Theparsertraversesrelevantsystemsandrecordsencounteredtypes
together with their locations, which enables two key ingestion-time enhance-
ments: (1) associating blocks with the MOOSE objects configured (useful
for retrieval constraints and faceted search), and (2) collecting object docu-
mentation snippets when available (via a local markdown lookup) to ground
later LLM generated summaries and improve retrieval context.
LLM-based input summarization with syntax-tree grounding.To make full-
input retrieval useful (especially when a user asks high-level questions such
as “what does this input do?”), MOOSEnger generates an automatic natural-
language summary for each imported input file. The summary is produced
during ingestion by a lightweight LLM and is injected into the parent doc-
ument so it is indexed and retrievable alongside the raw input. The sum-
marization prompt is grounded by: (i) the list of detected systems/blocks,
(ii) the extractedtype-instantiated objects, and (iii) (optionally) short doc-
umentation excerpts corresponding to those objects. This design keeps the
summary faithful to the input’s actual structure while providing a concise
narrative description for rapid triage and retrieval. When debug logging is
enabled, the system records both the summary prompt and the generated
summary for reproducibility and analysis.
By aligning chunk boundaries with HIT block structure and enriching
each chunk with syntax-tree metadata, MOOSEnger improves retrieval pre-
cision (returning the correct block for a query), preserves the context needed
to interpret parameters, and enables grounded, ingestion-time summaries
that remain tightly coupled to the actual MOOSE objects and configuration
expressed in the input files.
13

3.1.2. MOOSE Input Generation Validation
BecauseMOOSEngergeneratesMOOSEinputfilesfromnatural-language
requests, it must handle failure modes of both free-form generation (e.g.,
malformed HIT syntax) and domain hallucinations (e.g., non-existent ob-
jecttypenames). To improve reliability before any expensive simulation is
launched, MOOSEnger adds a validation layer that combines (i) determin-
istic input sanitation, (ii) syntax-aware type similarity search for correcting
object names, (iii) grammar-constrained input repair, and (iv) a compos-
ite precheck workflow that chains these stages with semantic checking and
optional execution.
Deterministic input sanitation.In practice, LLM-generated inputs often in-
clude hidden formatting that is visually innocuous, but can break down-
streamparsingandvalidation(e.g., mixednewlineconventions, smartquotes,
non-breaking spaces, or zero-width/control characters copied through chat
interfaces). MOOSEnger addresses this with themoose.sanitize_input
tool, which normalizes the raw text by normalizing newlines to\n, applying
Unicode Normalization Form Compatibility Composition (NFKC) normal-
ization, replacing smart punctuation and Non-Breaking SPace (NBSP) with
ASCII equivalents, and removing zero-width/control characters (excluding
tabs). The tool returns both the sanitized input and an auditable change
report (line/column, Unicode codepoint/name, and replacement), allowing
the agent to explain and trace any automatic edits before subsequent repair
and validation stages run.
Syntax-aware type similarity search for correcting object names.A common
source of errors in LLM-produced inputs is an invalidtypestring (e.g., a
near-miss of a real MOOSE class name). MOOSEnger mitigates this by
extracting candidate type identifiers from the generated input file (primar-
ilytype=parameters within subsystem blocks) and verifying them against
an application-specific registry of valid objects. When an exact match is
not found, MOOSEnger performs asyntax-awaresimilarity search that con-
ditions the correction on the local block context. Concretely, the search
uses: (1) the block path (e.g.,/Kernels/,/BCs/,/Materials/) to restrict
candidates to the appropriate object family, (2) lexical similarity (e.g., edit
distanceandtokenoverlap)tocapturetyposandnamingvariants, and(3)se-
manticsimilarity(e.g., embedding-basednearestneighborsoverobjectnames
and short descriptions) to resolve confusions between related objects. The
14

top-kcandidates are ranked with a composite score, which helps MOOSEn-
ger either apply an automatic substitution when confidence is high or return
a targeted clarification prompt to the user when ambiguity remains. This
same mechanism can be extended to suggest corrections for parameter keys
when the surrounding object type is known.
Grammar-constrained input repair.Before semantic checks can run, the in-
put must be parsable as HIT. Rather than applying unconstrained text edits,
MOOSEnger uses a grammar-constrained repair loop that enforces MOOSE
input structure:
1. Sanitizetherawtexttoeliminateinvisible/Unicodeformattingartifacts
that can trigger spurious parse failures.
2. Parse the candidate input with a HIT-aware parser to produce a syntax
tree (moose-pyhit); if parsing fails, capture the failure location and
expected token class.
3. Apply a minimal, rule-based repair that preserves block integrity (e.g.,
balancing[ ]delimiters, repairing malformed assignments, normaliz-
ing quotes, and fixing list separators) while avoiding edits that would
change block boundaries.
4. Re-parse and iterate until the input is syntactically valid or a bounded
repair budget is exceeded.
Because repairs are constrained by the grammar and applied locally around
parse failures, the system avoids “over-fixing” and can produce diffs that are
easy for users to audit.
Composite input-precheck workflow.MOOSEnger composes the above capa-
bilities into a deterministic input-precheck pipeline that runs prior to any full
simulation (Figure 3). The tool accepts either inline text or a path and or-
chestrates sanitation, repair, type correction,–check-inputvalidation, and
optional execution in a bounded loop until the input file is valid (or the
iteration budget is exhausted).
1. Sanitation: invokemoose.sanitize_inputto normalize Unicode or
newlines and remove invisible/control characters, producing a normal-
ized candidate plus a structured list of sanitation edits.
15

2. Repair: invoke the grammar-constrained repair loop until a valid HIT
syntax tree is obtained (or emit a structured error with the minimal
failing span).
3. Validation: perform semantic checks, including (i) type-name verifi-
cation and context-conditioned similarity correction, and (ii) parame-
ter/schema validation against known object definitions when available
(e.g., required parameters, allowed fields, and block placement con-
straints).
4. Execution: run a lightweight “smoke test” (e.g., a check-input mode, a
zero- or few-step run) to surface runtime issues such as missing files,
incompatible options, or initialization failures.
Each stage emits structured diagnostics and passes a normalized input file to
the next stage. In addition to the final candidate,moose.input_precheck
reports the number of iterations, any object-type replacements, the sani-
tation change summary, and the downstream check/run reports, enabling
MOOSEnger to provide actionable feedback early and reduce costly trial-
and-error cycles for end users.
3.1.3. MOOSE Execution in the Loop
Staticcheckscanensurethatageneratedinputfileisparsableandschema-
consistent, but they cannot guarantee that it will execute under real solver
conditions. To close this gap, MOOSEnger places the MOOSE runtimein
the loopand uses execution feedback as ground truth for “verify-and-correct”
iterations.
To keep execution portable across environments, MOOSEnger introduces
a small execution abstraction with two interchangeable backends: alocal
backend and an MCP-based backend. The local backend runs a configured
MOOSE executable directly (e.g., on a developer workstation). The MCP
backend delegates execution to an external service through a standardized
MCP protocol [16], allowing the agent to target remote or managed resources
(shared workstations, containers, or HPC/cluster services) without embed-
ding environment-specific logic in the core system. Both backends expose
identical tool semantics (validate, mesh-only, run) while keeping executables
and data within approved environments and enabling centralized operational
control when the agent is deployed at scale.
16

MOOSE MCP tool.Execution capabilities are exposed through a compact
tool interface that maps to low-cost stages of the MOOSE workflow: (i)
schema and setup validationvia–check-input, (ii)mesh-only or initial-
ization runsto isolate geometry, BC, material, etc. issues, and (iii) short
smoke-testsolves to surface solver misconfiguration and early convergence
failures. Each call returns a structured response (exit status, stdout/stderr,
and references to generated artifacts such as logs or output files), enabling
the agent to reliably interpret failures, explain them to the user, and propose
targeted repairs (e.g., correcting file paths, replacing invalid object types,
fixing block placement, or adjusting solver tolerances).
Local execution fallback.For offline use or restricted settings, MOOSEnger
provides a local execution backend that implements the same interface as
the MCP backend. This preserves identical agent logic and prompts across
deployments: switching between local and remote execution changes only the
backend binding, not the workflow.
Chaining execution with precheck and repair.In a typical run, MOOSEn-
ger first applies the composite pre-check pipeline (sanitation→grammar-
constrained repair→semantic validation) to obtain a parsable, schema-
consistent input file. It then selects the lightest-weight execution mode that
matches the user’s intent—–check-inputfor fast configuration verification,
mesh-only runs for geometry/mesh debugging, and full runs when numeri-
cal behavior or output files must be validated. Runtime diagnostics (errors,
warnings, and convergence signals) are captured as structured tool outputs
and fed back into the repair loop, allowing MOOSEnger to iteratively tran-
sition from “generate” to “verify and correct” using ground-truth feedback
from MOOSE itself.
3.2. Built-in Evaluation
To ensure that MOOSEnger improves systematically (and not only anec-
dotally), we provide a built-in evaluation module that can be executed as
part of development and continuous integration. The module supports both
RAG-centricevaluation (measuring the quality of retrieved context and its
use in responses) andagent-centricevaluation (measuring end-to-end task
success and alignment with user intent). Concretely, MOOSEnger integrates
theRagasframework for standardized RAG metrics, and extends it with a
customized agent evaluation pipeline that measures theconsistency of user-
query correctness: whether the produced inputs and explanations satisfy the
17

Figure 3: MOOSEnger input-precheck workflow for LLM-generated MOOSE inputs. The
initial input file is first sanitized to remove formatting artifacts, then repaired/validated
with a grammar-constrained HIT check and a syntax-aware object/type verification step.
The resulting input can be further exercised via a lightweight MOOSE syntax check and
an optional short simulation smoke test. The loop repeats until the input passes all checks
or a maximum iteration budget is reached.
requirements expressed in the user request across runs, prompts, and tool-
feedback iterations.
3.2.1. RAG Evaluation
The RAG evaluation path targets the retrieval and grounding compo-
nents in isolation from downstream execution. For each benchmark query,
the evaluation harness records (i) the query, (ii) retrieved chunks (including
their metadata, ranks, and source provenance), and (iii) the assistant’s fi-
nal response. We then compute a suite of retrieval/grounding metrics using
Ragas, emphasizing signals that are directly relevant to technical authoring
and configuration guidance [17]:
•Faithfulness:whether the response is supported by the retrieved con-
text (mitigating hallucinated MOOSE options or parameters).
•Answer relevancy:whether the response addresses the query intent
rather than generic MOOSE guidance.
•Contextprecision/recall:whetherretrievedchunkscontaintheneeded
information without excessive noise.
In addition to aggregate scores, the module produces per-query diagnostics
(retrieval snapshots, supporting spans, and failure categories such as “miss-
ing context” vs. “unsupported claim”), enabling targeted improvements to
18

Figure 4: Example MOOSEnger evaluation report summarizing RAG performance, in-
cluding aggregate metrics (e.g., faithfulness, answer relevancy, context recall/precision)
and per-query breakdowns.
ingestion, chunking, and ranking. An example of MOOSEnger generated
evaluation report is shown in Figure 4.
3.2.2. Agent Evaluation
We evaluate MOOSEnger as aninteractivesystem that must translate
a natural-language request into an executable MOOSE input file under real
solver conditions. Unlike static checks that only test syntax, our agent eval-
uation is end-to-end: it includes planning, retrieval, tool calls, iterative re-
pair/validation, and execution.
End-to-end success criterion.Each evaluation prompt is executed in an iso-
lated workspace to avoid cross-case contamination and to retain intermediate
artifacts for debugging. After the agent finishes, the evaluation runner ex-
tracts the final candidate input file (preferring the lastmoose-fenced code
block when present, otherwise falling back to the workspacecurrent.i) and
executes it viamcp.run_input. A prompt is counted as apassonly if the
runner reportsexit_code=0. This objective gate ensures the generated in-
19

put is not only parsable but runnable by an actual MOOSE executable (local
or remote runner).
Baselines.We compare against an LLM-only baseline (ChatGPT 5.2 API)
that receives the same prompt but is restricted to emitting a singlemoose
code block with no tool calls during generation. The produced input is then
executed with the samemcp.run_inputgate, isolating the impact of tool-
enabled planning, validation, and repair.
Traceability and regression tracking.For every prompt, the evaluation stores
the final input, execution logs, and (when enabled) tool traces in a per-
prompt workspace. This makes failures diagnosable (e.g., missing/invalid
object types vs. HIT-structure errors vs. runtime incompatibility) and sup-
ports regression analysis across prompt/tool/model updates.
4. Experiments
4.1. Benchmark suite
WeevaluateonfiveMOOSEproblemfamilies,eachcomprising25natural-
language prompts (125 total). The complete prompts used in the evaluation
are listed in Appendix A. Prompts request complete runnable setups and
typically specify key physics, geometry, boundary/initial conditions, and re-
quired outputs:
•Diffusion: minimal steady/transient diffusion cases on 1D/2D do-
mains with standard BCs and lightweight outputs.
•Transient heat conduction: time-dependent heat conduction with
explicit material properties and time-varying boundary conditions, re-
quiring correct transient executioner configuration.
•Solid mechanics: small-strain elasticity in 1D/2D with displacemen-
t/traction constraints and stress/reaction outputs.
•Porousflow: Darcy/pressure-diffusionvariantswithpermeability/vis-
cosity/porosity parameters and mixed BCs, including verification-style
prompts.
•Navier–Stokes: steady incompressible flow setups that require cou-
pled velocity–pressure formulations and appropriate solver settings.
20

Test problem MOOSEnger (Pass)Baseline (Pass)
ChatGPT 5.2 API
Diffusion25/25
(100%)9/25
(36%)
Transient heat conduction23/25
(92%)0/25
(0%)
Solid mechanics24/25
(96%)0/25
(0%)
Porous flow23/25
(92%)1/25
(4%)
Navier–Stokes21/25
(84%)0/25
(0%)
Overall116/125
(93%)10/125
(8%)
Table 1: End-to-end agent evaluation on five MOOSE problem families (25 prompts each).
Pass indicates successful execution viamcp.run_input(exit_code=0).
4.2. Evaluation protocol
For each prompt, we run (i) MOOSEnger in agent mode with its de-
fault tool suite (retrieval plus MOOSE input sanitation/repair/validation
and MCP execution tools), and (ii) thebaselinein LLM-only mode (no tool
calls during generation). In both cases, the final input file is executed with
mcp.run_input; success is defined byexit_code=0.
4.3. Illustrative case study: diffusion
Asanexampleinteraction, considerapromptrequesting2Ddiffusionona
square domain with a centered source. MOOSEnger decomposes the request
intorequiredinputblocks(mesh, variables, kernels, BCs/ICsorsourceterms,
executioner, andoutputs), draftsaninputfile, andusesitsprecheckworkflow
to sanitize formatting, repair HIT structure, correct invalidtype=names,
validate withmcp.check_input, and execute a lightweight run. The final
response returns a runnable file and a brief explanation of the modeling
choices.
21

5. Discussion
5.1. End-to-End Performance and Trade-offs
Table 1 reports end-to-end success rates and total suite runtimes. Across
125 prompts, MOOSEnger produces executable input files for 116/125 cases
(pass rate 0.93). The LLM-only baseline succeeds on 10/125 (pass rate 0.08),
yielding a 9.5×relative improvement in executability. The largest gains
appear in the more structured multiphysics suites: the baseline achieves a 0%
pass rate on transient heat conduction, solid mechanics, and Navier–Stokes,
while MOOSEnger maintains 0.84–0.96 pass rates. On the simplest diffusion
suite, MOOSEnger raises the pass rate from 0.36 to 1.0. All together, these
testsindicatethattheimprovementsarenotlimitedtoasinglephysicsfamily.
This gap is largely explained by closing the loop between generation and
solver feedback. Rather than requiring a perfect one-shot draft, MOOSEnger
iteratively converts “nearly correct” inputs into runnable inputs by (i) sani-
tizing and normalizing the generated text to remove formatting artifacts that
breakparsing, (ii)repairingmalformedHITstructureunderanexplicitgram-
mar constraint, (iii) correcting invalid or hallucinated objecttype=names
via similarity search against the application syntax registry, and (iv) vali-
dating and (when required) executing the input viamcp.check_inputand
mcp.run_input. In effect, tool-backed validation turns executability into an
iterative process with measurable intermediate signals (parse success, type
validity, solver checks), whereas the baseline must succeed in a single gener-
ation step.
These reliability gains come with higher wall-clock time. MOOSEnger
performs additional tool calls and, importantly, successfully advances to ex-
pensive simulation stages more often instead of failing early. This trade-off
is expected for an evaluation criterion that requires execution. We also em-
phasize a key limitation of the current metric: our pass criterion measures
executabilityrather than full scientific correctness. An input file can run
while still deviating from modeling intent (e.g., an oversimplified bound-
ary condition or an unintended material model). Future evaluations should
incorporate intent-aware constraints (derived from the prompt) and richer
post-run checks (e.g., verifying boundary conditions, conserved quantities,
or expected solution trends) to better align “pass” with scientific validity.
22

5.2. Lessons Learned from AI Integration with Scientific Software
IntegratinganAIassistantwithalargesimulationframeworklikeMOOSE
highlighted that domain grounding is not optional. Generic LLMs, even
strong ones, are brittle when asked to produce precise domain-specific DSLs.
Curating a knowledge base of documentation and examples—and making it
retrievable at generation time—substantially improved both correctness and
the assistant’s ability to justify its choices. The effort spent parsing and
indexing MOOSE documentation paid off by turning many questions into
retrieval-and-compose rather than pure synthesis.
We also found that specialized parsing and deterministic tooling are es-
sential complements to language models. Early iterations relied on the LLM
to “reason through” raw input syntax, which led to frequent structural errors.
Incorporating formal HIT parsing and a grammar-constrained repair step re-
duced these failures by enforcing well-formed blocks and parameters. More
broadly, combining symbolic constraints (grammar and type registries) with
LLM reasoning was more effective than either alone: the tools provide hard
guarantees and diagnostics, while the model handles ambiguity and intent
translation.
A third lesson concerned autonomy versus control. Prompting and tool
descriptions are ultimately soft constraints: an agent can skip validation
steps, call tools in an inefficient order, or attempt execution before basic
checks. For well-defined subtasks that have a stable procedure (such as in-
put precheck), an explicit workflow graph proved more reliable than uncon-
strained agent planning. Implementing precheck as a fixed, iterative tool
increased the probability that every necessary step (sanitize, repair, vali-
date, and optionally execute) was applied consistently. At the same time,
we preserved flexibility where it matters: the agent can still decide when to
retrieve documentation, inspect session artifacts, or branch into debugging
based on the user’s follow-up goals. In practice, this hybrid design balanced
guardrails with adaptability and made it easier to incorporate additional ca-
pabilities (e.g., memory and reusable skills) without over-constraining the
overall interaction.
From a development perspective, evaluation and testing required a differ-
entmindsetthanconventionaldeterministicsoftware. BecauseLLMbehavior
can vary across prompts and minor wording changes, we relied on scenario-
based testing and an evaluation harness that continuously accumulates new
cases. We also found it important to fail gracefully: when the system can-
not ground an answer in available documentation or cannot resolve an error,
23

it should surface that limitation and propose a concrete next step (e.g., re-
quest missing details or suggest a diagnostic run) rather than persisting with
low-confidence claims. Aligning prompts toward candid uncertainty reduced
misleading outputs and improved user trust.
Looking ahead, we expect MOOSEnger to shape user workflows in two
complementary ways: as a rapid reference (e.g., “what is the syntax for X?”)
and as a debugging partner (e.g., “why does this input fail?”). Beyond au-
tomation, this points to a broader value proposition in knowledge dissemina-
tion—the assistant can surface relevant details that might otherwise remain
buried in manuals or scattered examples, potentially accelerating onboard-
ing and iteration. At the same time, MOOSEnger is intended to augment
rather than replace domain expertise; effective use will still depend on hu-
man judgment about modeling assumptions, verification, and interpretation
of results.
6. Conclusion and Future Work
6.1. Summary of Impact and Potential
We presented MOOSEnger, a tool-enabled AI assistant integrated with
the MOOSE multiphysics simulation framework. MOOSEnger provides a
conversational interface that translates natural-language requests into exe-
cutable MOOSE workflows by combining retrieval over curated documenta-
tion and examples with domain-aware generation, input validation/repair,
and optional solver execution. Rather than relying on one-shot text gen-
eration, the system closes the loop with deterministic tooling to improve
robustness when producing MOOSE input files.
Our evaluation on a 125-prompt benchmark shows that this design ma-
terially improves end-to-end executability relative to an LLM-only baseline,
especially for structured multiphysics configurations where strict syntax and
validobjecttypesareessential. TheseresultssuggestthatLLMagentscanbe
applied effectively in scientific computing when grounded by domain context
and constrained by programmatic checks.
Looking forward, MOOSEnger points to a practical path for making com-
plex engineering software more accessible: an AI layer can shift user effort
away from remembering DSL details and toward expressing modeling in-
tent and interpreting outcomes. We expect this to benefit both experienced
users—by automating routine setup, surfacing relevant references quickly,
24

and assisting with debugging—and new users, by providing guided, inter-
active scaffolding while they learn MOOSE concepts and conventions. Be-
yond MOOSE itself, the architecture offers a reusable pattern for augmenting
legacy simulation codes with retrieval, validation, and execution tools, en-
abling assistants that are both helpful and accountable.
6.2. Future Work
WhileMOOSEngeralreadydemonstratesrobustsingle-agentperformance,
several directions remain to broaden its impact and improve reliability in
real workflows. We will extend the core–plugin architecture to simplify au-
thoring of application-specific agents (prompt packs, skills, tools, and cu-
rated corpora) and to support coordinated multi-agent workflows for com-
plex multiphysics studies. We plan to mine structured diagnostics from the
precheck/repair loop (recurring syntax repairs, frequent type substitutions,
andcommonexecutionfailures)toidentifyknowledgegapsandautomatically
propose updates to retrieval content and prompt/tool guidance. User feed-
back and successful end-to-end runs could be distilled into reusable skills or
indexed as case examples to reduce repeated failure modes. We will integrate
MOOSEnger into interactive environments (Graphical User Interface (GUI),
Integrated Development Environment (IDE) plugins, and notebooks) that
provide block-level navigation, inline diagnostics, guided editing, and quick
visualization of outputs to accelerate iteration. We will extend execution
backends to job schedulers and cloud environments, including queue-aware
submission, asynchronous status updates, and robust handling of credentials
and resource access. Beyond executability, we aim to add intent-aware checks
and physics-informed post-run validation (e.g., boundary-condition verifica-
tion, invariant checks, and expected-trend tests), potentially complemented
by a verifier agent for high-stakes settings.
Overall, these developments move MOOSEnger toward a more general
and trustworthy workflow layer for simulation studies while preserving trans-
parency through auditable tool traces and reproducible workspaces.
7. Acknowledgment
This research was funded by the Office of Nuclear Energy of the U.S.
Department of Energy NEAMS project No. DE-NE0008983. This research
made use of Idaho National Laboratory’s High Performance Computing sys-
tems located at the Collaborative Computing Center and supported by the
25

Office of Nuclear Energy of the U.S. Department of Energy and the Nuclear
Science User Facilities under Contract No. DE-AC07-05ID14517.
Code availability
The MOOSEnger codebase is hosted on INL GitLab and is currently
undergoing laboratory open-source review. Access can be provided upon
request during the review period.
Declaration of generative AI and AI-assisted technologies in the
manuscript preparation process
During the preparation of this work the author(s) used OpenAI chat-
gpt 5.2 in order to improve the writing and formatting of the manuscript,
using OpenAI codex to assistant coding. After using this tool/service, the
author(s) reviewed and edited the content as needed and take(s) full respon-
sibility for the content of the published article.
References
[1] L. Harbour, G. Giudicelli, A. D. Lindsay, P. German, J. Hansel, C. Icen-
hour, M. Li, J. M. Miller, R. H. Stogner, P. Behne, D. Yankura, Z. M.
Prince, C. DeChant, D. Schwen, B. W. Spencer, M. Tano, N. Choi,
Y. Wang, M. Nezdyur, Y. Miao, T. Hu, S. Kumar, C. Matthews,
B. Langley, N. Nobre, A. Blair, C. MacMackin, H. B. Rocha, E. Palmer,
J. Carter, J. Meier, A. E. Slaughter, D. Andrš, R. W. Carlsen, F. Kong,
D. R. Gaston, C. J. Permann, 4.0 MOOSE: Enabling massively parallel
multiphysics simulation, SoftwareX 31 (2025) 102264. URL:https://
www.sciencedirect.com/science/article/pii/S2352711025002316.
doi:https://doi.org/10.1016/j.softx.2025.102264.
[2] Y. Wang, Z. M. Prince, H. Park, O. W. Calvin, N. Choi, Y. S. Jung,
S. Schunert, S. Kumar, J. T. Hanophy, V. M. Labouré, C. Lee, J. Or-
tensi, L. H. Harbour, J. R. Harter, Griffin: A MOOSE-based reactor
physics application for multiphysics simulation of advanced nuclear
reactors, Annals of Nuclear Energy 211 (2025) 110917. URL:https://
www.sciencedirect.com/science/article/pii/S0306454924005802.
doi:https://doi.org/10.1016/j.anucene.2024.110917.
26

[3] A. Novak, R. Carlsen, S. Schunert, P. Balestra, D. Reger,
R. Slaybaugh, R. Martineau, Pronghorn: A multidimen-
sional coarse-mesh application for advanced reactor thermal hy-
draulics, Nuclear Technology 207 (2021) 1015–1046. URL:https:
//www.tandfonline.com/doi/full/10.1080/00295450.2020.1825307.
doi:https://doi.org/10.1080/00295450.2020.1825307.
[4] R. Williamson, K. Gamble, D. Perez, S. Novascone, G. Pas-
tore, R. Gardner, J. Hales, W. Liu, A. Mai, Validating the
BISON fuel performance code to integral lwr experiments, Nu-
clear Engineering and Design 301 (2016) 232–244. URL:https://
www.sciencedirect.com/science/article/pii/S0029549316000789.
doi:https://doi.org/10.1016/j.nucengdes.2016.02.020.
[5] A. Lindsay, G. Giudicelli, P. German, J. Peterson, Y. Wang, R. Freile,
D. Andrs, P. Balestra, M. Tano, R. Hu, et al., Moose navier–stokes
module, SoftwareX 23 (2023) 101503.
[6] J. Hansel, D. Andrs, L. Charlot, G. Giudicelli, The MOOSE thermal hy-
draulics module, Journal of Open Source Software 9 (2024) 6146. URL:
https://doi.org/10.21105/joss.06146. doi:10.21105/joss.06146.
[7] C. T. Icenhour, A. D. Lindsay, C. J. Permann, R. C. Mar-
tineau, D. L. Green, S. C. Shannon, The MOOSE electromag-
netics module, SoftwareX 25 (2024) 101621. URL:https://
www.sciencedirect.com/science/article/pii/S2352711023003175.
doi:https://doi.org/10.1016/j.softx.2023.101621.
[8] A. E. Slaughter, Z. M. Prince, P. German, I. Halvic, W. Jiang, B. W.
Spencer, S. L. Dhulipala, D. R. Gaston, Moose stochastic tools: A
module for performing parallel, memory-efficient in situ stochastic sim-
ulations, SoftwareX 22 (2023) 101345.
[9] Z. M. Prince, L. Munday, D. Yushu, M. Nezdyur, M. Guddati, Moose
optimization module: Physics-constrained optimization, SoftwareX 26
(2024) 101754. doi:https://doi.org/10.1016/j.softx.2024.101754.
[10] Y. Chen, X. Zhu, H. Zhou, Z. Ren, Metaopenfoam: an llm-based multi-
agent framework for cfd, arXiv preprint arXiv:2407.21320 (2024).
27

[11] L. Yue, N. Somasekharan, Y. Cao, S. Pan, Foam-agent: Towards auto-
matedintelligentcfdworkflows, arXivpreprintarXiv:2505.04997(2025).
[12] Z. N. Ndum, J. Tao, J. Ford, Y. Liu, Autofluka: A large language model
based framework for automating monte carlo simulations in fluka, arXiv
preprint arXiv:2410.15222 (2024).
[13] Anthropic, Claude code, 2026. URL:https://code.claude.com/docs/
en/overview, accessed 2026-02-09.
[14] OpenAI, Codex,https://openai.com/codex/, ???? Accessed: 2026-
01-15.
[15] T. Zhang, Z. Liu, Y. Xin, Y. Jiao, Mooseagent: A llm based multi-
agent framework for automating moose simulation, arXiv preprint
arXiv:2504.08621 (2025).
[16] Model Context Protocol, Specification (version 2025-11-25), Model
Context Protocol, 2025. URL:https://modelcontextprotocol.io/
specification/2025-11-25, accessed: 2026-01-23.
[17] S. Es, J. James, L. E. Anke, S. Schockaert, Ragas: Automated eval-
uation of retrieval augmented generation, in: Proceedings of the 18th
Conference of the European Chapter of the Association for Computa-
tional Linguistics: System Demonstrations, 2024, pp. 150–158.
Appendix A Agent evaluation prompts
This appendix lists the 125 natural-language prompts (25 per problem fam-
ily) used in the agent evaluation.
Appendix A.1 Diffusion
D1. Create a minimal 2D transient diffusion input with constant material proper-
ties. Use Dirichlet BC on the left (u=1) and right (u=0). Then run it.
D2. Write a simple 2D diffusion problem on a unit square with a constant source
term and steady solve. Provide the full input and run it.
D3. Generate a 1D transient diffusion example on [0,1] with initial condition u=0,
left boundary u=1, right boundary u=0. Run it.
28

D4. Makea2Ddiffusionsimulationusingageneratedmesh(e.g., GeneratedMesh).
Use a linear solver and run the case.
D5. Produce a very small diffusion input that runs fast (coarse mesh, short end
time). Include BCs and run it.
D6. Create a 2D diffusion on a rectangular mesh (nx=ny∼10). Use steady state
and Dirichlet BCs on all sides. Run it.
D7. Write a 2D diffusion problem with a Gaussian initial condition and transient
time stepping. Run it.
D8. Generate an input for heat conduction (diffusion) in 2D with constant kappa.
Use implicit Euler and run it.
D9. Make a diffusion example with a Neumann flux on the top boundary and
Dirichlet u=0 elsewhere. Run it.
D10. Create a diffusion input that uses a Material to provide diffusivity. Run it.
D11. Write a 1D steady diffusion problem with u(0)=0 and u(1)=1. Use a minimal
mesh and run.
D12. Create a 2D transient diffusion input with u=0 initial condition and a con-
stant volumetric source. Run it.
D13. Produce a tiny diffusion test case that outputs the solution field to Exodus.
Run it.
D14. Generate a diffusion input that uses linear Lagrange FE, solve steady, and
run.
D15. Make a 2D diffusion example on a 1x1 square using a coarse mesh; solve
steady; run.
D16. Create a diffusion case with a time-dependent Dirichlet BC (e.g., u=sin(t))
on the left boundary; run a short transient.
D17. Write a transient diffusion problem with a small dt and short end time. Keep
it minimal and run it.
D18. Generate a 2D diffusion input with separate blocks for Mesh, Variables, Ker-
nels, BCs, Executioner, Outputs. Run it.
D19. Create a simple diffusion example that uses PETSc linear solver defaults and
runs.
29

D20. Make a diffusion case with adaptive time stepping if available, but keep it
simple and runnable. Run it.
D21. Write a steady diffusion input with a source term and Dirichlet BCs; ensure
it’s runnable and run it.
D22. Create a 2D diffusion input with a small mesh and transient solve; run it.
D23. Generate a 1D diffusion input that runs quickly and writes CSV output; run
it.
D24. Makeadiffusionexamplewithufixedonleft/rightandinsulatedtop/bottom;
run it.
D25. Create a minimal diffusion input that you expect to work with common
MOOSE apps; run it.
Appendix A.2 Transient heat conduction
H1. Create a transient 1D heat conduction MOOSE case for a 0.5 m steel rod
(k=45 W/m·K,ρ=7800 kg/m3, cp=470 J/kg·K). Initial T=293 K. At t=0 set
T=373 K at x=0 and T=293 K at x=L. Simulate 0–200 s and output T(x,t)
plus T at x=0.25 m vs time.
H2. Transient 1D aluminum rod L=0.2 m (k=205,ρ=2700, cp=900). Initial
T=300 K. Apply heat fluxq′′=1e5 W/m2at x=0 for0< t <0.5s then
q′′=0; set x=L insulated. Run 5 s and write temperature at x=0 and x=0.1
m to CSV.
H3. Model a 1D polymer slab L=0.1 m (k=0.2,ρ=1200, cp=1500). Initial T=373
K. Both ends convect to ambient 293 K with h=25 W/m2·K. Run 0–600 s and
output average temperature vs time.
H4. 2D square plate 1×1 m with k=10,ρ=5000, cp=400. Initial 293 K. Left edge
fixed at 400 K; other three edges convection to 293 K with h=15. Run 0–50
s (adaptive dt OK). Output Exodus and center temperature vs time.
H5. 2D rectangle 0.3×0.1 m stainless steel (k=16,ρ=8000, cp=500). Initial 293 K.
Add volumetric heat generation Q=2e6 W/m3only in a centered 0.05×0.05 m
region; outer boundaries insulated. Run 0–10 s and output max temperature
vs time.
H6. 2D 0.2×0.2 m plate split at x=0.1 m: left is copper (k=400,ρ=8960, cp=385),
right is glass (k=1.2,ρ=2500, cp=800). Initial 293 K. Left boundary T=373
K;rightboundaryinsulated; top/bottomconvectionto293Kwithh=10. Run
0–100 s and output temperature along the material interface.
30

H7. 2D 0.1×0.1 m epoxy (k=0.3,ρ=1200, cp=1000) with a copper circular inclu-
sion radius 0.015 m at the center. Initial 293 K. Deposit total power 10 W
uniformly in the inclusion (convert to volumetric). All outer edges convect to
293 K with h=30. Run 0–30 s and output center temperature.
H8. 3D cube side 0.05 m, material k=150,ρ=2330, cp=700. Initial 300 K. Bottom
face held at 350 K; all other faces convection to 300 K with h=100. Run 0–2
s and output average T and bottom-face heat flux vs time.
H9. 3Dbrick0.1×0.05×0.02maluminum(k=205,ρ=2700, cp=900). Initial300K.
Apply a spatial Gaussian volumetric heat source centered at (0.05,0.025,0.01)
with total power 50 W and time decay exp(-t/0.2 s). All boundaries insulated.
Run 0–1 s and output max temperature vs time.
H10. Axisymmetriccylinder: radius0.01m, height0.05m, k=20,ρ=7800, cp=500.
Initial 293 K. Outer radius insulated, inner radius convection to 293 K with
h=5, top face heat fluxq′′=2e4 W/m2for 1 s then off. Run 0–10 s; output T
at r=0, z=0.025.
H11. 2D 1×1 m domain, anisotropic kx=50, ky=5 W/m·K, withρ=3000, cp=600.
Initial 293 K. Left edge 350 K, right edge 293 K, top/bottom insulated. Run
0–20 s and output temperature contours (Exodus).
H12. 1D rod L=0.1 m withρ=5000, cp=500 and k(T)=10*(1+0.01*(T-300)). Ini-
tial 300 K. Left boundary ramps 300→600 K over 2 s; right boundary fixed
300 K. Run 0–5 s and output profile snapshots.
H13. 2D 0.2×0.2 m, k=15,ρ=7800, cp(T)=400+0.5*(T-300). Initial 300 K. Con-
vectiononallsidesto300Kwithh=50. AdduniformvolumetricsourceQ=1e6
W/m3for 0<t<0.5 s then off. Run 0–5 s; output average temperature.
H14. 1D slab L=0.5 m withρ=2000, cp=1000, and k(x)=1+9*(x/L) W/m·K.
Initial 300 K. Set x=0 to 400 K; set x=L insulated. Run 0–100 s; output
T(x,t).
H15. 2D0.1×0.1m,k=30,ρ=7800,cp=500. Initial293K.LeftedgeT=293+50sin(2πt)
K; other edges convection to 293 K with h=20. Run 0–5 s; output center
temperature vs time.
H16. 1D rod L=0.2 m (k=100,ρ=8000, cp=400). Initial 293 K. At x=0 apply
square-wave flux:q′′=5e4 W/m2for 0.1 s on / 0.1 s off repeating. At x=L fix
T=293 K. Run 0–2 s and output T at x=0.02 m.
H17. 2D 0.5×0.5 m, k=5,ρ=2500, cp=800. Initial 350 K. All edges convect with
h=10. Ambient temperature steps from 300 K to 280 K at t=10 s. Run 0–60
31

s and output average temperature and total convective heat loss vs time.
H18. 1D slab L=0.05 m, k=2,ρ=1000, cp=2000. Initial 293 K. Left boundary
held at 500 K for t<1 s then instantly set to 293 K; right boundary insulated.
Run 0–10 s; output mid-plane temperature vs time.
H19. 2D 0.2×0.1 m, k=12,ρ=6000, cp=450. Initial 293 K. Left edge fixed 350 K;
right edge insulated; bottom edge convection to 293 K with h=25; top edge
prescribed heat fluxq′′=1e4 W/m2. Run 0–30 s; output max T and boundary
heat rates.
H20. 2D strip 1×0.1 m, k=20,ρ=7800, cp=500. Initial 293 K. Left boundary 400
K, right boundary 293 K, top/bottom insulated. Use a non-uniform mesh
refined near x=0 to capture steep gradients. Run 0–5 s; output centerline
T(x) at multiple times.
H21. 2D 0.2×0.2 m, k=15,ρ=2700, cp=900. Initial condition: T=293+200*exp(-
((x-0.1)2+(y-0.1)2)/(2*0.012)). All boundaries insulated. Run 0–1 s and out-
put peak temperature decay vs time.
H22. 3D cube 0.1 m, k=40,ρ=7800, cp=500. Initial 293 K. Volumetric heat
generation Q=0 for t<5 s, then Q=5e5 W/m3for t≥5 s. All faces convection
to 293 K with h=15. Run 0–30 s; output average and max T vs time.
H23. 2D 0.3×0.3 m, k=8,ρ=3000, cp=700. Initial 293 K. Left edge 323 K, right
edge 293 K, top/bottom convection to 293 K with h=5. Add probe outputs
for T at (0.15,0.15) and (0.25,0.15) to CSV. Run 0–200 s.
H24. 2D domain 0.2 m in x and 0.05 m in y, k=10,ρ=2500, cp=900. Make y
boundaries periodic. At x=0 set T=350 K; at x=0.2 apply convection to 300
K with h=20. Initial 300 K. Run 0–50 s and output y-averaged temperature
along x (or sample along y=midline).
H25. 2D unit square with exact solution T(x,y,t)=300+10*sin(πx)*sin(πy)*exp(-
t). Use k=1,ρ=1, cp=1 and add the needed volumetric source so this is
an exact transient conduction solution. Apply Dirichlet BCs from the exact
function. Run to t=1 and computeL 2error vs the exact solution over time.
Appendix A.3 Solid mechanics
S1. Build and run a small-strain linear elastic 1D bar (L=1 m) with E=210 GPa,
ν=0.3. Fix x=0, prescribe u=1e-3 m at x=L. Output axial stress and reaction
force vs load step.
32

S2. Create a 2D plane strain linear elasticity problem on a unit square. Material:
E=100 GPa,ν=0.25. Fix bottom edge (u_y=0), fix left edge in x (u_x=0),
apply traction t_y=-1e6 Pa on top. Run and output displacement and von
Mises stress.
S3. Write a 2D plane stress tension test on a rectangle [0,2]x[0,1] with E=70 GPa,
ν=0.33. Clamp x=0, prescribe u_x=1e-3 at x=2. Output averageσ_xx and
reaction force.
S4. Generate a 1D bar with a body force (gravity-like): f=1000 N/m3. Fix x=0,
free at x=L. Solve steady and output displacement and stress distribution.
S5. Make a minimal 2D small-strain elasticity input using GeneratedMesh
(nx=ny=10) on a unit square. Apply u_x=0 on left, u_x=1e-3 on right, and
u_y=0 on bottom. Run and output displacement to Exodus.
S6. Create a 2D cantilever beam (L=1, H=0.2) in plane strain. Fix left edge.
Apply downward traction on right edge. Run and output tip displacement
andσ_yy field.
S7. Write a 3D linear elasticity cube (coarse mesh) with E=200 GPa,ν=0.3. Fix
bottomface, applyuniformpressureontopface. Runandoutputdisplacement
magnitude.
S8. Build a 2D shear test: unit square, plane strain. Fix bottom edge, apply
horizontal displacement u_x=1e-3 on top edge. Run and outputτ_xy field.
S9. Create a 1D bar with two materials (left half E=200 GPa, right half E=100
GPa). Fix x=0, prescribe u at x=L. Run and output stress; verify stress
continuity.
S10. Generate a small-strain elasticity problem that outputs strain components to
CSV at a probe point. Use 2D plane strain, apply simple boundary displace-
ments, and run.
S11. Make a 2D plane strain patch test (constant strain): impose linear displace-
ment field on boundaries and verify constant stress. Output max/min stress.
S12. Create a 2D axisymmetric cylinder under internal pressure (if axisymmetric
solid mechanics is available). Define radius/thickness, apply pressure, run and
output radial displacement.
S13. Write a transient small-strain dynamics case (if available): 1D bar with den-
sity and a step load at one end. Run transient and output displacement vs
time at midspan.
33

S14. Generate a 2D thermal expansion-only case (easy): 1D bar (L=1 m), E=200
GPa,ν=0.3,α=12e-6 1/K. Fix both ends (u=0 at x=0 and x=L). Apply
uniform∆T=100 K via a temperature field. Compute and output stress.
S15. Build a 2D elasticity case with a time-dependent prescribed displacement
boundary (e.g., u_y=1e-3*sin(2πt) on top). Run short transient and output
reaction vs time.
S16. Create a 2D linear elasticity problem that computes total strain energy as a
Postprocessor and writes it to CSV. Apply traction loading and run.
S17. Make a 2D notched plate (simple geometry using a structured mesh approxi-
mation) under tension. Run and output stress concentration near notch (max
von Mises).
S18. Write a 3D cantilever (coarse mesh) with tip force applied as a traction.
Output tip deflection to CSV and displacement field to Exodus.
S19. Generate a 2D elasticity case using PETSc defaults and ensure it runs quickly
on a coarse mesh. Output displacement and stress.
S20. Create a 1D bar and verify reaction force matches analytic EAε/L for pre-
scribed end displacement. Output reaction and compare.
S21. Make a 2D elasticity case with mixed BCs: fix left edge, roller support on
bottom (u_y=0), apply traction on right. Run and output reaction compo-
nents.
S22. Write a transient run + restart test for an elasticity problem (quasi-static
with multiple load steps): run first half, restart, complete. Output a history
file.
S23. Create a 2D plane strain compression test: apply uniform downward displace-
ment on top, fix bottom, prevent rigid motion. Outputσ_yy and reaction.
S24. Generate a 2D elasticity problem that estimates Poisson’s ratio by measuring
averageε_xx andε_yy from displacements (report both).
S25. Create a minimal small-strain linear elasticity input expected to work with
common MOOSE solid mechanics modules; include mesh, displacement vari-
ables, material, BCs, executioner, and outputs. Run it.
Appendix A.4 Porous flow
P1. Create and run a steady 1D Darcy flow in a homogeneous column (L=1 m).
Use constant permeability k=1e-12 m2and viscosity mu=1e-3 Pa·s (ignore
34

gravity). Set p=2e5 Pa at x=0 and p=1e5 Pa at x=1. Output pressure vs x
and the Darcy flux.
P2. Write a transient 1D pressure diffusion problem in a porous medium (single-
phase slightly compressible). Use porosity=0.2, permeability=1e-13 m2,
viscosity=1e-3, compressibility=1e-9 1/Pa. Initial p=1e5 Pa. At x=0 impose
p=2e5, at x=1 impose p=1e5. Run to t=1000 s and output p(x,t).
P3. Generate a 2D steady Darcy flow on a unit square with anisotropic permeabil-
ity (kx=1e-12, ky=1e-13). Set p=1e5 on left, p=0 on right, no-flow top/bot-
tom. Run and output pressure and velocity.
P4. Create a 1D gravity-driven hydrostatic pressure profile in a porous column
(height=10 m). Include gravity, density=1000 kg/m3. Impose p=1e5 at top.
Solve steady and output p(z) to verify dp/dz=ρg.
P5. Write a 2D transient injection problem: initial p=1e5, inject at a small
region on left boundary with fixed p=2e5, outlet at right p=1e5. Use
permeability=1e-12, viscosity=1e-3, porosity=0.25. Run short transient and
output pressure field.
P6. Generate a 1D Darcy flow case with a volumetric source term (well) in the
middle: add a point/region source Q. Use fixed pressures at ends. Run and
output flow rates.
P7. Make a 2D porous flow case that outputs mass balance diagnostics: compute
inlet/outlet flux via Postprocessors and write to CSV. Run it.
P8. Create a steady 1D Darcy case with two layers (different permeability blocks).
Setpressuresatendsandoutputpressureandflux; verifyfluxcontinuityacross
the interface.
P9. Write a transient 2D pressure diffusion case with adaptive time stepping en-
abled (if available). Use slightly compressible fluid and run; output dt history
and nonlinear iterations.
P10. Generate a 3D steady Darcy flow in a small box (coarse mesh) with a pressure
drop from z=0 to z=1. No-flow on other faces. Output velocity magnitude.
P11. Createa1Dverification-styleproblem: comparenumericalDarcyfluxagainst
analytic q=-(k/mu)*dp/dx. Output both computed and analytic flux to CSV.
P12. Write a 2D steady flow case driven by a constant body force (equivalent to
hydraulic gradient) instead of boundary pressures. Use periodic left/right if
possible. Run and output average velocity.
35

P13. Makea2Dporousflowcasewithpressure-dependentpermeabilityk(p)=k0*(1+ap)
via a Material. Run steady and output pressure and velocity.
P14. Create a transient 1D pressure pulse: initial p=1e5 everywhere; apply a brief
pressure increase at x=0 (e.g., p=2e5 for t<10 s, then back to 1e5). Run to
t=200 s and output p(x,t).
P15. Generatea2Dradial(orpseudo-radial)welltestifsupported: injectatcenter
with fixed rate, outer boundary fixed pressure. Run transient and output
pressure vs radius/time.
P16. Write a 2D steady Darcy flow that outputs the L2 norm of the residual (or
convergence metric) as a Postprocessor and writes it to CSV. Run it.
P17. Create a 1D Darcy case that outputs pressure at multiple probe points to
CSV each timestep. Run a short transient.
P18. Make a 2D porous flow case using PETSc solver defaults and ensure it runs
on a coarse mesh. Output pressure field.
P19. Generate a 3D transient porous flow case (very coarse mesh) to test restart:
run to t=0.1, checkpoint, restart to t=0.2. Output pressure at a probe.
P20. Createa2DsteadyDarcycasewithmixedBCs: leftfixedpressure, rightfixed
flux (Neumann), no-flow top/bottom. Run and output resulting pressure field.
P21. Write a 1D diffusion-like porous flow case in terms of hydraulic head (if your
setup uses head). Include gravity and convert to pressure outputs. Run it.
P22. Generate a 2D transient case and compute total stored fluid mass vs time
(porosity*compressibility*pressure integral) as a Postprocessor. Output to
CSV.
P23. Create a 1D steady Darcy case with a time-dependent boundary pressure
(e.g., p=1e5+5e4*sin(t) at x=0) and fixed p at x=1. Run a short transient
and output flux vs time.
P24. Make a minimal porous flow input expected to work with your PorousFlow
module app: mesh, primary variable (pressure), material properties, BCs,
executioner, outputs. Run it.
Appendix A.5 Navier–Stokes
N1. Create a minimal 2D steady incompressible channel flow (Poiseuille) on
[0,2]x[0,1]. Use prescribed inlet velocity profile, no-slip walls, and pressure
outlet. Run and output velocity and pressure.
36

N2. Write a 2D transient start-up channel flow: start from u=v=0, ramp inlet
velocity from 0 to U over t=[0,1]. Run and output centerline velocity vs time.
N3. Generate a 2D lid-driven cavity flow on a unit square: no-slip walls, moving
lid at top with u=1. Run and output velocity magnitude.
N4. Create a 2D Couette flow: top wall moving, bottom stationary, periodic left-
/right, incompressible solve. Run and compare with linear velocity profile.
N5. Build a 2D steady Stokes flow test (very low Re): drive with a constant body
force in x, no-slip top/bottom, periodic in x. Run and output u(y).
N6. Create a 2D Taylor–Green vortex decay in a periodic box using an initial
velocity field. Run transient and output kinetic energy decay vs time.
N7. Make a 2D channel flow case with symmetry: model only half the channel
with a symmetry boundary at the centerline. Run and output profile.
N8. Create a 2D pressure-driven channel flow using pressure BCs (p_in and
p_out) instead of a velocity inlet. Run and output the resulting flow rate.
N9. Generate a 3D steady Poiseuille-like flow in a short 3D box channel (coarse
mesh): no-slip on walls, pressure difference from inlet to outlet. Run it.
N10. Create a 2D open-channel style rectangle with left inlet velocity, right outlet
pressure, and no-slip top/bottom. Run and output pressure field.
N11. Write a 2D incompressible flow input using a finite-volume Navier–Stokes
formulation (if available in your build). Run it on a coarse mesh.
N12. Write a 2D incompressible flow input using a finite-element Navier–Stokes
formulation (if available). Run it on a coarse mesh.
N13. Create a 2D transient incompressible flow case that uses adaptive time step-
ping to maintain stability (e.g., based on nonlinear iterations). Run it.
N14. Build a 2D channel flow with variable viscosity mu(y)=mu0*(1+y) via a
Material. Run and output u(y).
N15. Create a 2D steady flow that outputs volumetric flow rate at the outlet using
a Postprocessor and writes it to CSV. Run it.
N16. Generate a 2D channel case and compute inlet vs outlet flow rate to check
mass conservation (Postprocessors). Run and output both to CSV.
N17. Create a lid-driven cavity case and output the velocity at the cavity center
vs time to CSV. Run it.
37

N18. Make a 2D channel flow case with a time-dependent inlet velocity (e.g.,
u_in=sin(t) clipped to positive). Run a short transient.
N19. Create a 2D channel flow with a very coarse mesh (nx∼12, ny∼6) that runs
fast. Solve steady and run it.
N20. Generate a 3D lid-driven cavity flow (coarse mesh) and run a short transient;
output velocity magnitude.
N21. Create an incompressible flow case that writes Exodus plus a CSV of pressure
drop between two points. Run it.
N22. Create a 2D transient channel flow with a sudden inlet velocity step (u_in
jumps from 0 to 1 at t=0). Run to steady state and output velocity magnitude
vs time.
N23. Create a 2D steady flow case using PETSc defaults (no fancy solver tuning)
and ensure it runs.
N24. Generate a 2D channel flow case that outputs the L2 norm of the divergence
(or continuity residual) as a diagnostic Postprocessor. Run it.
N25. Create a minimal incompressible Navier–Stokes input expected to work with
your NavierStokes module app (mesh, velocity/pressure variables, BCs, Exe-
cutioner). Run it.
38