# IteraSim RAG: A Multi-Stage Retrieval-Augmented Agentic Back-End for OpenFOAM-Based Computational Fluid Dynamics

**Authors**: Pratyush Kumar

**Published**: 2026-07-22 16:31:51

**PDF URL**: [https://arxiv.org/pdf/2607.20346v1](https://arxiv.org/pdf/2607.20346v1)

## Abstract
Configuring a computational fluid dynamics (CFD) case in OpenFOAM requires assembling a multi-directory input deck of mutually consistent solver, discretisation and boundary-condition dictionaries -- a task that remains a substantial barrier to non-specialist use of open-source CFD software. Large language models (LLMs) coupled with retrieval-augmented generation (RAG) can lower this barrier, but existing systems retrieve with a single flat query, apply one retrieval strategy to operationally distinct requests, and let a single agent both draft and review its own output. We present IteraSim RAG, a retrieval-augmented software back-end for automated OpenFOAM case generation built around these three limitations. An LLM first expands the query into physics, solver-keyword and troubleshooting variants, Reciprocal Rank Fusion then merges the resulting ranked lists, and Maximal Marginal Relevance re-ranks the fused candidates against an HNSW-indexed dense vector store. A deterministic keyword router dispatches tool-conditioned workflow queries and corpus-wide physics queries down separate retrieval paths, and generation is split across an Architect, an InputWriter and a Reviewer agent, backed by a static canonical-knowledge layer covering solver selection, turbulence closures, boundary conditions and finite-volume defaults. On an openly released 28-case benchmark spanning zero-shot setup, few-shot generalisation, single-parameter modifications and turbulence-model swaps, the pipeline attains a mean retrieval coverage of 77.9% (median 79.1%), with the parameter-modification category exceeding 90%. All six reference configurations run to completion on OpenFOAM v2506, and two synthetically corrupted cases are diagnosed and repaired within the bounded Reviewer loop using only the solver log and the canonical layer. The benchmark, scoring rubric and figure scripts are released for reproducibility.

## Full Text


<!-- PDF content starts -->

IteraSim RAG: A Multi-Stage Retrieval-Augmented
Agentic Back-End for OpenFOAM-Based
Computational Fluid Dynamics
Pratyush Kumara,∗
aSeminar for Applied Mathematics, ETH Zürich, Rämistrasse 101, 8092 Zürich,
Switzerland
Abstract
Configuring a computational fluid dynamics (CFD) case in OpenFOAM re-
quires assembling a multi-directory input deck of mutually consistent solver,
discretisation and boundary-condition dictionaries – a task that remains
a substantial barrier to non-specialist use of open-source CFD software.
Large language models (LLMs) coupled with retrieval-augmented genera-
tion (RAG) can lower this barrier, but existing systems retrieve with a single
flat query, apply one retrieval strategy to operationally distinct requests,
and let a single agent both draft and review its own output. We present
IteraSim RAG, a retrieval-augmented software back-end for automated
OpenFOAM case generation built around these three limitations. Retrieval
proceeds in three stages. An LLM first expands the query into physics,
solver-keyword and troubleshooting variants, Reciprocal Rank Fusion then
merges the resulting ranked lists, and Maximal Marginal Relevance re-ranks
the fused candidates against an HNSW-indexed dense vector store. A de-
terministic keyword router dispatches tool-conditioned workflow queries and
corpus-wide physics queries down separate retrieval paths, and generation is
split across an Architect, an InputWriter and a Reviewer agent, backed by
a static canonical-knowledge layer covering solver selection, turbulence clo-
sures, boundaryconditionsandfinite-volumedefaults. Onanopenlyreleased
28-case benchmark spanning zero-shot setup, few-shot generalisation, single-
parameter modifications and turbulence-model swaps, the pipeline attains
∗Corresponding author.
Email address:pratyush.ethz@gmail.com(Pratyush Kumar)
Preprint submitted to Computer Physics Communications July 23, 2026
arXiv:2607.20346v1  [cs.CE]  22 Jul 2026

a mean retrieval coverage of77.9%(median79.1%), with the parameter-
modification category exceeding90%. All six reference configurations run
to completion on OpenFOAM v2506, and two synthetically corrupted cases
are diagnosed and repaired within the bounded Reviewer loop using only
the solver log and the canonical layer. The benchmark specification, scoring
rubric, corrupted-case probes and figure-regeneration scripts are released for
reproducibility.
Keywords:Retrieval-augmented generation, Large language models,
Multi-agent systems, Computational fluid dynamics, OpenFOAM, Scientific
software automation
1. Introduction
Computational fluid dynamics (CFD) underpins a broad range of engi-
neering and scientific applications, spanning laminar transport in pipe net-
works[1], centrifugalturbomachinery[2], membrane-basedgasseparation[3],
haemodynamic transport through branching vasculature [4], coupled CFD–
DEM analysis of packed beds [5], and helium cooling of pebble-bed nuclear
reactor canisters [6], among many others. Within open-source CFD tools,
OpenFOAM is the most widely used software because its object-oriented C++
architecture allows arbitrary modification of solvers, discretisation schemes
and physical sub-models without the licensing constraints of commercial
packages [7, 8]. OpenFOAM has been applied across a comparably broad
range of physical regimes, including wall-bounded and free-shear turbulence,
incompressible and compressible flows, multiphase and free-surface trans-
port, conjugate and buoyancy-driven heat transfer, porous-media and granu-
lar flows, and reactive and combustion modelling [5, 6, 9, 10]. That flexibility
carries a steep set-up burden. Configuring a typical case means assembling a
multi-directory folder structure, picking discretisation and linear-solver set-
tings that are mutually compatible, writing physically consistent boundary
conditions, and then diagnosing the divergence that a poor combination of
any of these will trigger. Adding new physics to an existing solver is harder
still, even for seasoned users. The associated expertise is acquired over years
of practice and remains the principal obstacle to wider adoption of Open-
FOAM outside the dedicated CFD community [11, 12].
A complementary line of work has therefore explored data-driven assis-
tance for CFD users. Early efforts focused on surrogate modelling of specific
Preprint submitted to Computer Physics Communications July 23, 2026

Table 1: Design choices comparison across RAG systems for CFD.
System RetrievalDrafter/
Reviewer splitLoop
bound
OpenFOAMGPT [11] Flat Single agent open
MetaOpenFOAM [18] Flat Yes (3 roles) –
Foam-Agent 1.0 [12] Flat Yes 25
Foam-Agent 2.0 [19] Hier. multi-index Yes 25
FoamGPT [20] Fine-tuned LLM (no RAG) Planner+writer –
ChatCFD [21] Flat + error locator Yes –
CFDAgent [22] Flat Yes –
SwarmFoam [23] Flat Swarm (peer agents) –
IteraSim RAG Multi-stage Architect / InputWriter / Reviewer 10
physics, including Reynolds-averaged turbulence closures [13], subgrid-scale
stresses for large-eddy simulation [14], and reduced-order models for para-
metric design [15]. Large language models (LLMs) have changed what such
an assistant can be. Instead of executing fixed code paths, an LLM-based
tool can read a problem stated in plain language, draft its own implementa-
tion plan, and revise that plan as the solver output comes back. In materials
science, agentic systems based on LLMs have been shown to autonomously
propose and refine candidate alloys [16]; in scientific software more broadly,
they have demonstrated the ability to write, debug and execute non-trivial
code in closed loops [17]. The application of LLMs to CFD is now a distinct
and rapidly growing sub-field. Our literature survey identifies the principal
LLM-for-CFD systems and their contributions, summarised in Table 1; each
system is discussed in turn in the paragraphs below.
Pandeyet al.[11]introducedOpenFOAMGPT,thefirstretrieval-augmented
LLM agent reported in the CFD literature. They showed that a general-
purpose foundation model coupled with a flat vector store can configure
and debug single-phase and multiphase OpenFOAM cases through an iter-
ative correction loop. Chenet al.[18] introduced multi-agent role separa-
tion withMetaOpenFOAM, in which a planner, input-writer and reviewer
collaborate over a LangChain-based RAG built from the official tutorials,
reporting an 85% pass rate on an eight-case benchmark. Yueet al.extended
the multi-agent paradigm to the full simulation lifecycle inFoam-Agent1.0
and 2.0 [12, 19], adding a hierarchical multi-index RAG, dependency-aware
generation across coupled configuration files, a Gmsh-based Meshing Agent
and an MCP-compatible service layer. The system reaches an 88.2% end-to-
3

end success rate on a 110-case benchmark. The same group demonstrated
withFoamGPT[20] that supervised fine-tuning on the 202 canonical Open-
FOAMtutorial casescan outperformmuch larger proprietarybaselines. Here
the quality of the training data mattered more than the size of the model.
ChatCFD[21] pushed the executability metric to 82.1% over 315 cases and
introduced a Physics Interpreter that evaluates whether a runnable case is
also physically meaningful—the first formal treatment of physical fidelity
in this line of work.CFDAgent[22] andSwarmFoam[23] further explored
zero-shot language-guided orchestration and swarm-based cooperation across
parallel agent instances, respectively. Across these systems the dominant in-
novations have been in agent decomposition (role specialisation, dependency
awareness), in dataset curation (FoamGPT) and in evaluation (physical-
fidelity scoring in ChatCFD). The retrieval layer itself has remained largely
unchanged; a single user query is embedded and the top-Knearest chunks
are concatenated into the prompt. Flat, single-query retrieval introduces at
least three persistent failure modes in the systems above.
(L1) Single-query, flat retrieval..The user’s natural-language query is em-
bedded once and matched against the chunked corpus by cosine similarity.
This pattern is sensitive to the long-documented vocabulary mismatch be-
tween user phrasing and corpus terminology [24]. It also provides no mech-
anism to suppress near-duplicate passages, which consume prompt budget
without adding new evidence.
(L2) Undifferentiated routing across query types..Arequestto“write acontrolDict
for a steadysimpleFoamchannel”is operationally distinct from a question
such as“which low-Rek–ωvariant should I use under an adverse pressure
gradient”. Yet both are answered with the same flat retrieval over the same
corpus. The first call wants tightly tool-conditioned excerpts of documenta-
tion; the second wants breadth across the physical literature.
(L3) Drafting and review entangled in one agent; no canonical reference
layer..Even where role-specialised agents are used, the drafter and the re-
viewer often share prompts and context, which is known to amplify halluci-
nations and inflate the cost of recoverable errors [25]. Furthermore, agents
have no explicit practitioner-validated reference for solver syntax, boundary-
condition selectors or discretisation defaults—all details that a foundation
model can easily mis-recall when operating zero-shot over a purely retrieved
knowledge base.
4

IteraSim RAG is designed to close each of these gaps. Two further axes
not introduced in the previous works (Table 1) complete the comparison.
Prior systems do not implement an explicit dual-mode retrieval router that
separates solver-conditioned workflow queries from corpus-wide physics-level
queries (L2). Similarly, only ChatCFD [21] augments retrieval with a static
practitioner-knowledge layer – its Solver Template Database – and no other
system exposes such a reference alongside the retrieved context (L3).
Therefore,thispaperintroducesIteraSim RAG,anOpenFOAM-oriented
back-end designed around (L1)–(L3). The overall architecture is summarised
in Fig. 1 and described in detail in Sec. 2. Its main contributions are:
1. Athree-stageretrievalpipelinethatrespondsto(L1). AnLLMrewrites
the raw query into three semantic reformulations spanning a physics
facet, a solver-keyword facet, and a troubleshooting facet. Ranked lists
returned by each reformulation and by the original query are merged
through Reciprocal Rank Fusion (RRF) [26], and the fused candidate
set is re-ranked by Maximal Marginal Relevance (MMR) [27] to pro-
duce a diverse, low-redundancy context block. Retrieval is performed
against an HNSW-indexed vector store [28] of dense embeddings, and
the pipeline does not depend on a particular embedding family or
vector-store implementation.
2. Adeterministicdual-moderouterthatrespondsto(L2). Eachincoming
queryisdispatchedbyakeyword-drivenclassifierintooneoftwomodes.
Workflowmode locks retrieval onto a solver- and tool-conditioned sub-
corpus and walks the upstream/downstream tool-dependency graph.
Expertmode broadens retrieval over the full corpus, down-weights ver-
batim full-case listings and applies a more diversity-favouring MMR
trade-off. The classifier is defined over an eighteen-category CFD on-
tology, so no additional LLM call is required to route the query.
3. AnArchitect–InputWriter–Reviewertriad,supplementedbyacanonical-
knowledge layer, that responds to (L3). The Architect decomposes
the request into a file manifest and picks a reference solver and tem-
plate. The InputWriter then emits the individual OpenFOAM dic-
tionaries. The Reviewer compiles the case, executes it, parses solver
andwmakelogs, and iterates corrective patches up to a bounded depth
of ten cycles. The canonical layer is injected alongside the retrieved
context. It encodes solver-selection rules, turbulence-closure decision
trees, boundary-condition selectors, finite-volume defaults and mesh-
5

ing recipes, so that established practitioner knowledge is not lost to
retrieval noise.
4. A solver-agnostic and tool-aware extensibility layer. The retrieval,
routing and orchestration components are deliberately decoupled from
any specific solver. The same pipeline currently serves OpenFOAM,
snappyHexMesh,blockMesh, Gmsh,pyFoamand ParaView, and ac-
commodates new text-documented solvers by extending the ingestion
manifest and the tool-dependency graph alone.
Section2describesthesystemcomponents; Sec.3evaluatesitona28-case
benchmark and examines residual failure modes; and Sec. 4 draws conclu-
sions.
2. Methodology
2.1. System Overview
The overall data flow of IteraSim RAG is summarised schematically in
Fig. 1. The figure can be read top-to-bottom as a single pass: a user query
enters at the top, is answered from a retrieval store on the right, and leaves
as a validated OpenFOAM case at the bottom. The remainder of this section
follows the same path, block by block; each block in the figure corresponds
to a subsection below.
A user query (User Queryin Fig. 1) enters through a deterministic in-
tent classifier inside the retrieval engine (RAG Engine), which routes it to
one of two retrieval modes (Sec. 2.4). The engine then runs a three-stage
retrieval over a vector store (Vector Database) that is populated offline by an
embedding engine (Embedding Engine) from the knowledge sources (Knowl-
edge Sources); the corpus and its ingestion are described in Sec. 2.2 and the
retrieval stages in Sec. 2.3. The output is the retrieved CFD context block
(Retrieved CFD Context).
Thiscontextblock,togetherwithastaticcanonical-knowledgelayer(Sec.2.5),
isinjectedintothegenerationstage(Generic LLM Model), realisedhereasan
Architect–InputWriter–Reviewer agent triad (Sec. 2.6). The triad produces
the multi-agent response (Multi-Agent Response); a Runner component exe-
cutes the case on disk and returns solver andwmakelogs to the Reviewer for
corrective iteration, closing the loop shown by the return arrows in Fig. 1.
Implementation and configuration details common to all blocks, including
the provider-agnostic model adapters and the offline vector-store fallback,
are collected in Sec. 2.7.
6

Figure 1: Architecture of IteraSim RAG (each block is walked through, top to bottom, in
the paragraphs that follow).
2.2. Knowledge Corpus and Ingestion
The knowledge corpus (theKnowledge Sourcesof Fig. 1) is assembled
from five complementary sources: (i) a curated set of expert markdown notes
covering solver selection, boundary-condition idioms, mesh-quality criteria
and numerical-stability heuristics; the full text of these notes is proprietary,
but their section-level structure is documented in the released evaluation
kit [29]; (ii) a troubleshooting frequently-asked-questions set generated from
validatedpractitionerforumthreads[30]andstoredasquestion–answerpairs;
(iii) the full local knowledge base of OpenFOAM tutorial cases and reference
dictionaries (more than one thousand markdown documents) drawn from
the official ESI-OpenCFD OpenFOAM distribution [31]; (iv) the publicly re-
leased FoamGPT question–answer pairs covering the 202 canonical tutorial
cases [20]; and (v) the wave-energy converter (WEC) OpenFOAM extension
corpus released by Sandia National Laboratories [32], which supplies cou-
pled multiphase and rigid-body documentation absent from the upstream
tutorials.
7

Figure 2: Schematic representation of the three-stage retrieval pipeline of IteraSim RAG.
Documents are split with a section-aware chunker that respects mark-
down level-2 and level-3 heading boundaries. When no headings are present,
it falls back to a sliding-window word-based splitter. The default chunk
size at ingestion is 400 words for documentation and 800 words for long-
form practitioner notes. A 100-word overlap preserves semantic continuity
across chunk boundaries and limits the truncation artefacts typical of hard
splits. Each chunk is then encoded by a general-purpose dense text encoder
into a fixed-dimensional vector (d=3072in the configuration evaluated here).
The vectors are written to the vector store together with metadata (source
URL, tool tag, intent tag, document type and an MD5 fingerprint used for
deduplication) under cosine distance with the HNSW graph index [28] de-
scribed above. Ingestion is idempotent. Documents already in the collection
(matched by MD5 fingerprint) are skipped, making ingestion idempotent.
2.3. Three-Stage Retrieval Pipeline
For every user turn the retrieval layer transforms the raw query into a
single context block through three sequential stages. Figure 2 shows these
stages from left to right: query expansion (Stage 1), multi-query retrieval
with reciprocal rank fusion (Stage 2), and maximal-marginal-relevance re-
ranking (Stage 3). Each stage below corresponds to one panel of the figure.
8

2.3.1. Stage 1: Query expansion
The first panel of Fig. 2 corresponds to this stage. A single natural-
language query is frequently under-specified, and it may use vocabulary that
does not match the corpus. This vocabulary mismatch is a long-documented
failure mode of retrieval systems [24]. The first stage therefore asks the lan-
guagemodeltoproducethreealternativereformulationsoftheoriginalquery.
The prompt explicitly targets three distinct facets: a physics/theory fram-
ing, an OpenFOAM-keyword framing that emphasises solver, dictionary and
boundary-condition names, and a practical troubleshooting framing. The
call goes to a fast general-purpose chat-tuned LLM at a temperature of0.4
with a 120-token budget, which is enough to return the three variants while
holdingper-turnlatencybelowonesecond. IftheAPIcallfails, thestagefalls
back to the original query alone, so the rest of the pipeline always receives a
non-empty list of variants.
2.3.2. Stage 2: Multi-query retrieval and reciprocal rank fusion
The second panel of Fig. 2 corresponds to this stage. The original query
and each of the three reformulations are embedded independently. Each
embedding retrieves the topK fetch= max(2K, K+4)candidate chunks from
the vector store, whereKis the final number of chunks requested by the
agent (typicallyK=5in workflow mode andK=8in expert mode). This
yields four ranked lists, each emphasising a different semantic facet of the
user’s intent. The lists are merged with Reciprocal Rank Fusion (RRF) [26],
which assigns each candidateda combined score
sRRF(d) =X
r∈R1
k+ rank r(d),(1)
whereRis the set of four ranked lists,rank r(d)∈ {1,2, . . .}is the rank ofd
in listr, andkis a smoothing constant fixed atk=60following the original
recommendation of Cormacket al.[26]. Duplicate chunks across lists are
collapsed by a short text fingerprint (the first 120 characters), which strips
out the near-identical paragraphs produced by tutorial-case boilerplate. We
use RRF rather than score averaging because it ignores the absolute scale
of the similarity scores and is unaffected by the differing score distributions
that queries of different specificity return.
9

2.3.3. Stage 3: Maximal-marginal-relevance re-ranking
The third panel of Fig. 2 corresponds to this stage. The fused candidate
set is re-ranked with Maximal Marginal Relevance (MMR) [27]. Given the
original query embeddingqand a setSof already-selected documents, MMR
iteratively picks the candidate
d∗= arg max
d∈C\Sh
λsim(d,q)−(1−λ) max
d′∈Ssim(d, d′)i
,(2)
wheresim(·,·)is cosine similarity in the embedding space andλ∈[0,1]
controls the relevance–diversity trade-off. We useλ=0.6in workflow mode,
which favours operational specificity. In expert mode we useλ=0.5, which
favours topical breadth across the corpus. The firstKdocuments selected
by MMR are concatenated to form the retrieved CFD context block. This
block is prepended to the agent system prompt prior to generation.
2.4. Dual-Mode Retrieval Routing
CFD queries are not homogeneous. Configuring a specific solver, debug-
ging a divergence, or choosing a turbulence closure each impose different
demands on retrieval. The first task wants tightly tool-conditioned excerpts
of documentation; the third wants breadth across the physical literature.
IteraSim RAG therefore exposes two retrieval modes.
Workflow mode..Used for queries that name, or unambiguously imply, a spe-
cificsolverorpre/post-processingutility(simpleFoam,interFoam,snappyHexMesh,
blockMesh,paraFoam, etc.). Retrieval is filtered to the documentation as-
sociated with the active tool. It is then enriched by a walk over an explicit
tool-dependency graph that captures the canonical OpenFOAM pipeline
(CAD→meshing→case-setup→solver→post-processing). The up-
stream and downstream tools of the active node contribute adjacent context
windows. A question aboutinterFoamsetup, for example, automatically
retrieves the relevantsnappyHexMeshandparaFoamexcerpts as well.
Expert mode..Used for conceptual and physics-level questions about turbu-
lenceclosures,mesh-sensitivityprotocols,numericalstabilityoftime-integration
schemes, and so on. Retrieval is performed over the full corpus without tool-
conditioning. Raw tutorial case dumps are down-weighted to favour curated
expert notes and FAQ pairs; the mode’s MMR relevance–diversity setting is
theλof Eq. (2). This mode is invoked when the router cannot identify an
active tool from the query.
10

Mode selection is performed by a deterministic, keyword-driven intent
classifier built over an explicit ontology of eighteen CFD-relevant categories
(solvers, turbulence models, numerical instability, mesh quality, boundary
conditions,heattransfer,multiphase,discretisation,ParaView,Gmsh,pyFoam,
workflow transitions, compressible flow, physical validation, and others). We
route with a keyword-based classifier here rather than a second LLM call: it
is deterministic and adds<1ms per turn. Because the ontology vocabulary
maps directly onto the OpenFOAM tool surface, it achieves 100% routing ac-
curacy on the 28-case benchmark. This design mirrors at the retrieval layer
the conceptual distinction between configuration and reasoning tasks already
identified, at the prompting layer, by Pandeyet al.[11]. On the 28 bench-
mark prompts of Sec. 3, every one of which explicitly names or implies an
OpenFOAM solver (the benchmark deliberately targets solver configuration
and parameter modification, not open-ended physics questions), the router
assignsworkflowmode with28/28accuracy against the ground truth. We
note explicitly that this is therefore a half-evaluation of the router: on this
benchmark the router is required to detect the workflow branch only, and the
expert-mode branch is not exercised. A separate held-out set of physics-level
probesandthecorrespondingexpert-moderoutingaccuracyarereleasedwith
the code and are the subject of a companion evaluation note; the present pa-
per does not attribute the observed retrieval quality to the router alone (see
the leave-one-out ablation of Sec. 3.2, in which multi-query expansion and
canonical pinning, not routing, carry the dominant contribution).
2.5. Canonical-Knowledge Layer
A purely retrieval-driven assistant is brittle when the corpus is thin in
a given area or when the LLM mis-recalls a syntactic detail. To stabilise
generation, IteraSim RAG maintains a static canonical-knowledge layer that
is appended to the system prompt of every agent in addition to the retrieved
context. The layer encodes practitioner knowledge that is well established
but easily corrupted by an LLM operating in a zero-shot regime, including:
solver-selection decision trees as a function of Mach number, phase count,
energy-equation requirement and steady-state behaviour; turbulence-closure
guides fork–ε,k–ωSST, Spalart–Allmaras and Reynolds-stress models, with
defaulty+ranges and wall-function recommendations; boundary-condition
selectors that map physical conditions onto OpenFOAM dictionary entries
(fixedValue,zeroGradient,kqRWallFunction, etc.); finite-volume scheme
defaults forfvSchemesandfvSolutionthat have been shown to be stable
11

on the official tutorials; and meshing recipes forblockMesh,snappyHexMesh
and Gmsh that cover the most common geometric primitives encountered in
entry-level industrial work.
This layer is deliberately kept separate from the retrieval corpus. It is
treated as authoritative reference rather than as a candidate for ranking. At
∼1.2k lines of structured markdown, it is small enough to be injected in full
without competing with the retrieved context for prompt budget.
2.6. Architect–InputWriter–Reviewer Orchestration
Generation in IteraSim RAG is split across three specialised agents. They
share the retrieved context block and the canonical-knowledge layer, but op-
erate with distinct responsibilities, prompts and termination criteria. To-
gether they form the generation stage of Fig. 1.
Architect..The Architect ingests the user prompt and produces asimulation
manifest: the target solver, the mesh strategy (blockMesh,snappyHexMesh
or external Gmsh), the turbulence and physical sub-models, the boundary-
conditionplanandtheorderedlistofcasefilesrequiredtorunthesimulation.
It draws on the retrieved tutorial cases to anchor its choices against verified
prior art. It is also the only agent that writes to the manifest data structure
consumed by the downstream stages.
InputWriter..The InputWriter takes the manifest and produces the Open-
FOAM case directory file-by-file. Each file is generated by a focused LLM
call that receives (i) the manifest entry for the current file, (ii) the relevant
canonical-knowledge fragment (e.g. the turbulence-closure guide when writ-
ing0/k), and (iii) the most relevant chunks of the retrieved context for that
file type. Generating one file per call keeps each generation step small, which
both curbs hallucination and confines any later correction to the dictionary
that actually failed.
Reviewer..The Reviewer compiles any custom solver viawmake, launches the
simulation and parses the resulting solver and build logs. A dedicated error
locator extracts the first physically meaningful error (boundary-condition in-
compatibility,missingfield,divergenceinaspecificequation,orsnappyHexMesh
cell-quality failure) and feeds it back to the InputWriter as a targeted cor-
rection request. The corrective loop is bounded to ten cycles, half the limit
reported by Foam-Agent [12]. This bound lets recoverable cases converge
quickly while pathological cases fail fast rather than accumulating token cost.
12

Runner..Execution (meshing, solver launch, run-time monitoring of residu-
als and field probes, and packaging of the result directory) is handled by a
Runner component that is exposed to the three agents as a tool. This keeps
the agents stateless with respect to the file system. The Architect plans the
run, the InputWriter writes the files, and the Reviewer inspects the conse-
quences; the Runner is the only component that touches the filesystem and
the operating-system process tree.
Solver logs produced during execution are appended to the conversation
historyvisibletotheReviewer. Thisenablesiterativerefinementthatmirrors
the expert debugging workflow of experienced CFD practitioners: failures are
interpreted, theoffendingdictionariesarepatched, andthecaseisre-rununtil
either convergence is reached or the loop budget is exhausted.
2.7. Implementation and Reproducibility
The retrieval pipeline, dual-mode router, canonical-knowledge layer and
Architect–InputWriter–Reviewer orchestrator are implemented as indepen-
dent Python modules with no shared mutable state. Swapping the vector-
store backend, for example, requires only changing the endpoint URL in
the configuration file. The Runner is implemented as a sandboxed sub-
process invoker that calls the OpenFOAM toolchain through standard shell
entry points. The LLM never executes shell commands directly. The model
provider, embeddingmodel, vector-storeendpoint, chunksize, RRFconstant,
MMRλand Reviewer-loop bound are exposed through a single configuration
file. Switching from the managed vector store [33] to the local in-memory
fallback requires only unsetting the endpoint URL. Researchers without per-
sistent outbound access to commercial APIs can point the endpoint URL at
a self-hosted model and rerun any single pipeline stage without touching the
surrounding code.
3. Results and Discussion
A28-casebenchmarkcategorisedinfourdistinctdifficultytiersareidenti-
fiedfromtherecentliteratureonLLMsforCFD[11,12,18,20,21]toevaluate
theperformanceofthepresentIteraSimRAGmodel. Thebenchmarkspans6
zero-shot prompts in Category A, 4 few-shot prompts in Category B, 10 cases
requiringasingle-parametermodificationtoanotherwisefunctionalbasecase
in Category C, and 8 zero-shot turbulence model or physical properties swaps
in Category D. The full benchmark, including the natural-language prompt
13

and the expected file/key-check list for every case, is released together with
this paper athttps://github.com/iterasim/iterasim-rag-public. The
released specification additionally carries eight diagnostic queries in a Cat-
egory E (solver-divergence, bounding, mesh-quality andy+interpretation
prompts). These exercise the troubleshooting path rather than case setup,
are not scored anywhere in this paper, and are shipped for use by the com-
munity and by the companion technical note; all results reported here are
confined to the 28 cases of Categories A–D. Each case carries an explicit
set of expected physics tagsT icomprising the target solver, the OpenFOAM
dictionaries that must be written, and the model-specific keywords that must
appear in the agent’s solution (e.g.alpha.waterandsetFieldsDictfor the
dam-break case,omegaWallFunctionfor thek–ωSST swap).
We report two distinct evaluation tiers in this paper. Theretrieval tier
(Sec. 3.1) measures the fraction ofT ithat appears in the context blockX i
returned by the three-stage RAG pipeline, before any generation occurs. The
retrieval score for caseiis
si= 0.80·|Ti∩tokens(X i)|
|Ti|+ 0.20·⊮[ solver i∈tokens(X i) ],(3)
whereT iis the set of expected physics tags for caseiandX iis the context
blockreturnedbythethree-stagepipeline. Theoperatortokens(·)returnsthe
set of terms contained in its argument, andsolver iis the target solver name
for casei. The first term is the fraction of expected tags recovered by re-
trieval; the second term is the indicator that the target solver name is present
in the retrieved context. The0.80/0.20split favours physics-tag coverage but
preserves a solver-identification term similar to OpenFOAMGPT [11]. The
end-to-end executability tieradditionally generates, meshes and runs each
case through OpenFOAM and scores the outcome on a five-point rubric that
reflects how far the case progresses through the CFD toolchain:
0the mesher (blockMesh,snappyHexMesh) fails or produces an invalid
mesh;
1the mesh is generated, but the solver aborts at initialisation with a
FOAM FATAL;
2the solver starts but exits with aFOAM FATALbefore reachingendTime;
3the solver runs toendTimewithout anyFATALexit;
14

4the solver runs toendTimeandevery user-specified parameter tag for
the case is correctly set in the generated dictionaries.
The executability tier requires a full local OpenFOAM toolchain and is re-
ported in Sec. 3.3 as an ongoing study; the present manuscript focuses on
the retrieval tier, which is the layer at which the architectural contributions
of this paper (L1)–(L3) directly act.
3.1. Retrieval performance
The retrieval performance of the three-stage RAG pipeline is evaluated on
the 28-case benchmark listed case-by-case in Table B.1 (Appendix B), with
a mean retrieval score of¯s=77.9%and a median ofs med=79.1%. Seven cases
reach perfect retrieval (s i≥99%), and21cases exceed the70%threshold
conventionally used as a coverage pass mark in information-retrieval evalua-
tion (Figure 3a). The gap between the mean and the median is driven by the
two low-scoring outliers in Category A discussed below; the median therefore
gives a more representative summary of typical retrieval performance. The
median end-to-end retrieval latency is2.09s, with a95th-percentile of2.73s
(Fig. 3b). The four-fold inflation in queries caused by the expansion stage
does not push the system out of the interactive regime. The four embed-
ded queries are issued in parallel, and the dominant cost is a single network
round trip to the vector store. The sub-three-second figure quoted here
refers specifically to the retrieval tier – query expansion, four-way embed-
ding, vector-store lookup, RRF and MMR – and excludes the downstream
LLM calls made by the Architect, InputWriter and Reviewer. Under the
LLM back-end evaluated here, the query-expansion call adds roughly0.5–
1.0s per turn and each subsequent per-file generation call adds a further2–
4s. A complete per-turn timing and token-cost budget for the agent-driven
pipeline – that is, how many seconds and how many LLM tokens each case
consumes end-to-end – is beyond the scope of the present manuscript and
will be reported in a follow-up study.
Per-category performance is summarised in Fig. 3a and Table 2, and the
pattern tracks the design intent of the dual-mode router and the three-stage
pipeline. Parameter-modification queries (Category C) reach the highest
mean score of90.7%and a median of exactly100%, with6of10cases at
perfect retrieval. That100%median is a direct consequence of the workflow
router’s tool-conditioning: because each of these prompts names its active
solver unambiguously, the router locks retrieval onto the solver-specific sub-
corpus, and the query-expansion step usually surfaces the relevant dictionary
15

A B C D020406080100Retrieval score (%)
Mean (77.9%)
Median (79.1%)(a) Mean retrieval coverage by category.
0.0 0.5 1.0 1.5 2.0 2.5
End-to-end retrieval latency (s)0.00.20.40.60.81.0Cumulative fraction of casesp50 = 2.09sp95 = 2.73s (b) Cumulative distribution of end-to-end retrieval
latency.
Figure 3: Retrieval-tier performance on the 28-case IteraSim RAG benchmark.
A
B
C
DA1 A2 A3 A4 A5 A6
B1 B2 B3 B4
C1 C2 C3 C4 C5 C6 C7 C8 C9 C10
D1 D2 D3 D4 D5 D6 D7 D8
020406080100
Coverage (%)
Figure 4: Retrieval coverage per case within each category.
in one of its three reformulations. Category C therefore gives the clearest
empirical support for the L2 contribution of Sec. 1. Few-shot generalisa-
tion (Category B) is the second-strongest category at82.0%, indicating that
the multi-query stage preserves the demonstrated example structure without
being washed out by globally similar passages. Zero-shot turbulence-model
swaps (Category D) average74.1%and never drop below60%. We attribute
thisuniformitytothecanonical-knowledgelayer(Sec.2.5), whichencodesthe
wall-function and field-equation requirements of each closure family indepen-
dent of the retrieved chunks. The weakest category is zero-shot open-ended
setup (Category A) at59.1%, with two outliers discussed below.
The two lowest-scoring cases (Fig. 4) point to one specific failure mode
rather than to a broad weakness of the pipeline. Case A5 (Particle Column,
sA5= 26.7%)expectsthenichetagsMPPICCloud,kinematicCloudProperties,
cloudPropertiesandinjectionModels. ThecorpuscontainstheMPPICFoam
16

Table 2: Per-category retrieval statistics on the 28-case benchmark, computed from the
composite scores iof Eq. (3).¯s: mean;s med: median;σ s: standard deviation;n perfect:
cases withs i≥99%.
Categoryn¯s(%)s med(%)σ s(%)n perfect
A (Zero-shot) 6 59.1 70.2 23.2 0
B (Few-shot) 4 82.0 80.0 13.3 1
C (Alt. conditions) 10 90.7 100.0 12.3 6
D (Turbulence swap) 8 74.1 75.2 8.8 0
Overall 28 77.9 79.1 18.5 7
solver header but only a handful of fragmentary chunks that mention these
dictionaries by name. Even with three reformulations, the multi-query re-
trieval cannot recover them. Case A3 (Hotroom buoyant convection,s A3=
33.3%)missesbuoyantBoussinesqSimpleFoam,alphatandtheexplicitgrav-
ity vector(0 -9.81 0), again because the buoyancy-driven solver family is
underrepresented in the ingested OpenFOAM tutorials. Both failures are
corpus-coverage problems, not retrieval-algorithm problems: every other case
scores≥60%. Ingesting niche-solver documentation—Lagrangian particle
clouds, Boussinesq convection, free-surface piston-driven flows—would close
both outliers without any change to the retrieval, routing or orchestration
components. A milder failure mode appears in the turbulence-swap cate-
gory. Exact-match retrieval of variant-spelling tags such asRNGkEpsilon
(case D1) andLESmodel Smagorinsky;(D6) under-counts coverage even
when the relevant family is correctly identified. This is an artefact of the
case-sensitive tag-matching in Eq. (3) and is expected to be absorbed by the
canonical-knowledge layer at the generation stage.
Manual re-scoring of A3 and A5 under a synonym-aware rule..To quantify
how much of the two outliers is a scoring artefact rather than a corpus-
coverage failure, we manually re-scored A3 and A5 with a synonym-aware
relaxation of Eq. (3) in which a missed tag is credited if a semantically equiv-
alent phrase is present in the retrieved context (e.g. “gravity vector” credits
(0 -9.81 0); “Boussinesqbuoyancysolver” creditsbuoyantBoussinesqSimpleFoam;
“turbulentthermaldiffusivity” creditsalphat). ForA3(Hotroom), twoofthe
seven missed tags map onto concepts that the retrieved chunks discuss under
alternativenames; thecorrectedscorerisesfroms A3= 33.3%tos′
A3≈66.7%,
17

close to the Category-A mean. For A5 (Particle Column), no such re-scoring
is available: none of the six missed MPPIC tags is aliased in the retrieved
context, so the case remains ats A5≈27%and is a genuine corpus-coverage
failure. The re-scored overall retrieval mean becomes¯s′≈79.1%, converging
on the uncorrected median˜s= 79.1%; the median is therefore the appropri-
ate central-tendency summary for the retrieval tier.
3.2. Component ablation and flat-RAG baseline
To separate the contribution of each retrieval-tier component we ran a
leave-one-out ablation, disabling one architectural element at a time and
re-scoring under the same rubric of Eq. (3). The ablation is scored on a
separate set of 30 retrieval queries spanning the tool-conditioned corpora
(OpenFOAM, meshing, Gmsh, FreeCAD, PyFoam and ParaView) and gen-
eral CFD theory, rather than on the 28-case end-to-end benchmark used
elsewhere in this section. The components under test act at the retrieval
tier, and this set exercises all of the tool buckets the router dispatches to,
whereas the 28-case benchmark is confined to OpenFOAM case setup. Scores
are therefore comparable between the rows of Table 3, but the absolute level
should not be read against the per-category means reported above. The
disabled configurations were:
•–multi-query: the LLM query-expansion stage of limitation L1 is by-
passed and a single, unexpanded query is passed to the vector store.
This configuration is algorithmically equivalent to a flat top-KRAG
baseline and isolates the contribution of L1.
•–MMR:theMaximalMarginalRelevancere-rankisdisabledandresults
are returned by fused rank alone.
•–canonical: thecurated-sourcepinningthatinjectsthecanonical-knowledge
layer of Sec. 2.5 is bypassed.
•–version-filter: the OpenFOAM release filter on the vector store is set
to accept any release.
Two components carry almost all of the observed retrieval quality. Dis-
abling multi-query expansion drops the score from75.6%to25.9%, a−49.7-
point loss, which quantifies the contribution of L1 against a flat single-query
RAG baseline. Disabling the canonical-knowledge pinning is even more
damaging (−50.1points), which quantifies the contribution of the static
18

Table 3: Leave-one-out ablation of the retrieval-tier components. Overall score is the
composites iof Eq. (3) averaged across the 30-query ablation set.
Configuration Overall (%)∆vs. full
Full pipeline 75.6 –
–version-filter 77.5+1.9
–MMR 65.0−10.6
–multi-query (flat RAG) 25.9−49.7
–canonical 25.5−50.1
knowledge layer of Sec. 2.5. The MMR diversifier accounts for a further
−10.6points; without it the fused top-Klist is dominated by near-duplicates
from the same tutorial and per-case coverage collapses in Category D (tur-
bulence swap), where the query terms are homogeneous. The version filter
is close to neutral in aggregate (+1.9points when removed), which is an
expected negative result: many queries do not stress the release selector, so
removing it broadens the candidate pool without reducing precision. Taken
together these numbers close the attribution argument that a per-category
breakdown alone cannot support: the multi-stage retrieval of L1 and the
canonical layer that underpins the agent triad (L3) are the two load-bearing
components.
The full-pipeline score on the 30-query ablation set,75.6%, lies within
one standard error of the77.9%retrieval mean measured on the 28-case
benchmark(Table2), sothetwosetsareofcomparabledifficultyeventhough
theyarenotthesamecases. ThequantityofinterestinTable3isnevertheless
the relative loss in each row, not the absolute level.
3.3. End-to-end executability
Figure 5 shows the path a natural-language prompt takes through the
back-end. The Architect drafts a case manifest, the InputWriter turns it into
a complete OpenFOAM case directory (Cavity/in the illustration) together
with the standardAllpre,AllrunandAllcleanscripts, and the Reviewer
parses the solver logs (log.blockMesh,log.checkMesh,log.icoFoam, etc.).
Whenever a log reports aFOAM FATALthe Reviewer patches the case and
re-runs it, up to the ten-cycle bound of Sec. 2.6.
Thebenchmarkalsodefinesanend-to-endexecutabilityscoreonthesame
0–4 rubric (Sec. 3): a case earns credit only once its dictionaries mesh and
19

Figure 5: End-to-end IteraSim RAG workflow: prompt→Architect–InputWriter–
Reviewer→OpenFOAM case, with solver-log parsing and corrective re-runs until
endTime.
the solver reachesendTime, and the full score of 4 additionally requires that
every user-specified parameter tag is present in the generated dictionaries.
As a validity check on the benchmark itself, we first ran all 28 cases from
their unmodified, hand-built OpenFOAM tutorials. Categories B–D share
their base configuration with the six Category A cases, so this reduces to
six distinct base runs (blockMesh, the case-specific solver, andsetFields
where the multiphase cases require it), inherited to the downstream cases
that reuse them. Every one of the 28 base configurations meshed cleanly
and reachedendTimeon OpenFOAM v2506 without aFOAM FATAL, at an
aggregate OpenFOAM cost of∼70s, confirming that the benchmark cases
are physically runnable on the target toolchain. Scoring the agent-driven
cases – those produced by the Architect–InputWriter–Reviewer orchestrator
from the prompt alone – across the full 28-case benchmark requires the cold-
start Category A generation path, which is still under active development,
and is deferred to the companion technical note.
Asaphysicalcontrol–beyondthepass/failrubric–thelid-drivencavity
(icoFoam) was generated from a natural-language prompt carrying the stan-
dard tutorial parameters (Appendix C) and run on v2506. Figure 6 places its
final-time velocity field beside the unmodified tutorial: the primary recircu-
lation, the corner vortices and the peak lid speed are indistinguishable within
contouring resolution, so the agent-generated case reproduces the requested
physics, not merely a runnable case.
3.4. Reviewer-loop FATAL-error probes
To characterise the Reviewer loop in isolation from the upstream Ar-
chitect and InputWriter agents, we constructed two synthetically corrupted
lid-driven cavity cases (OpenFOAM v2506) and issued a minimal post-hoc
report to the agent of the form “theicoFoamrun has terminated with aFOAM
20

Reference Agent
0.000.120.240.360.480.600.720.840.96
|U| (m/s)Figure 6: Final-time velocity magnitude att= 0.5s for the lid-driven cavity: unmodified
OpenFOAM tutorial reference case (left) and the case generated by IteraSim RAG from
the natural-language prompt (right).
FATAL;identifyandcorrectthefaultsuchthatthecaserunstoendTime.” The
probes span the two dominant classes of executability-tier failure reported
in the LLM-for-CFD literature: an omission in a runtime dictionary and a
structural error in a mesh-definition dictionary. Both probes, the captured
first-run FATAL logs, and the canonical post-fix reference files are released
with the source code underrag_evaluation/exec_fatal_probes/, so that
the agent’s output can be diffed externally against the ground truth.
Probe 1: runtime-dictionary FATAL..The kinematic-viscosity entrynuis
removed fromconstant/transportProperties.blockMeshcompletes nor-
mally;icoFoamterminates at initialisation withEntry ’nu’ not found in
dictionary "constant/transportProperties". On receipt of the report,
the Reviewer executed the following diagnostic sequence: (i) a read of the
solver log, which was absent at that point; (ii) enumeration ofsystem/;
(iii) confirmation of the solver identity andendTimefromcontrolDict;
(iv) invocation of the case-levelAllrunscript, which failed under the active
shell environment; and (v) direct execution of the solver through the internal
run_simulation_autonomoustool, which returned the FATAL log verbatim.
Substring extraction onEntry ’nu’ not foundlocalised the fault, and the
InputWriter emitted a targeted patch toconstant/transportProperties
restoring the missing entry asnu 1.5e-5. The Reviewer relaunched the
solver, which reachedendTime= 0.5s with monotonically decreasing residu-
21

als, recovering the canonical converged cavity solution.
Probe 2: mesh-regeneration FATAL..Theverticeslistinsystem/blockMeshDict
is truncated to six entries. Because theblocksentry retains an eight-vertex
hex reference,blockMeshterminates before any mesh is written, withPoint
label (6) out of range 0..5 in block. The Reviewer executed a four-
step correction: (i) openingblockMeshDictand localising the malformed
verticeslist; (ii) restoring the two missing corner vertices through the in-
ternaledit_filetool; (iii) re-runningblockMeshfollowed bycheckMesh,
which reportedMesh OK; and (iv) generating and executing anAllrunscript
that invokesblockMeshfollowed byicoFoam. The solver reachedendTime
without aFATALexit, and the recovered pressure field coincides with the
canonical cavity reference to within numerical round-off.
The two probes exercise complementary aspects of the executability loop.
In the dictionary probe, the fault is fully localised by the solver log itself,
and the Reviewer’s task reduces to substring extraction followed by a single
dictionary patch. In the mesh probe, no runtime log is emitted at the point
of failure, so localisation requires structural inspection of the mesh-definition
dictionary and independent verification throughcheckMeshbefore the solver
can be invoked. In both cases the correction was obtained within the ten-
cycle Reviewer bound of Sec. 2.6, and neither probe required retrieval to be
consulted for the FATAL diagnosis itself: the solver output, combined with
the canonical knowledge encoded in the system prompt of Sec. 2.5 and with
direct file-system access, was sufficient. A Reviewer restricted to log-parsing
andfile-systemaccessresolvesbothfailureclasseswithintheten-cyclebudget
without consulting the retrieval layer.
Extended probe taxonomy..Thetwomeasuredprobesabovecovertheruntime-
dictionary and mesh-definition failure classes, which together account for the
majority ofFOAM FATALexits seen in the Foam-Agent [12, 19] error cor-
pus. Two further probes have been constructed and are released alongside
the code (Table 4); their end-to-end Reviewer evaluation is deferred to a
companion technical note: (iii) aturbulence-BC mismatchprobe built on
thepitzDailybackward-facing step in which the closure is switched to
k–ωSST but theomegawall boundary condition is left aszeroGradient
rather thanomegaWallFunction;simpleFoamstill converges (noFATAL) but
the captured log is dominated bybounding omegawarnings with negative
ωvalues, i.e. a physically inconsistent wall-shear model that the canonical-
knowledge layer must identify from the log rather than from a fatal exit;
22

Table4: Reviewer-loopprobetaxonomy. Seethesurroundingparagraphsforthedifference
between themeasured(P1, P2) andconstructed(P3, P4) status labels.
Probe Failure class Broken artefact Status
P1 Runtime dict. FATALtransportPropertiesmeasured
P2 Mesh definition FATALblockMeshDictmeasured
P3 Turbulence BC mismatchomegawall BC on kOmegaSST constructed
P4snappyHexMeshcell qualitymeshQualityDictconstructed
and (iv) asnappyHexMeshcell-qualityprobe built on theflangetutorial
in whichmeshQualityDictis seeded with unreasonably strict thresholds
(maxNonOrtho 10,minTetQuality 0.5); snappy completes with∼66,000
illegal faces andcheckMeshreportsFailed 1 mesh checks, requiring the
Reviewer to relax the quality gates to the canonical v2506 defaults before
re-invoking the mesher. Together with the two measured probes, the four
probe classes span the executability-tier failure taxonomy of Foam-Agent [19]
atn=4probes across four distinct failure classes.
Table 5 situates IteraSim RAG against the eight prior LLM-for-CFD sys-
tems discussed in Sec. 1. Numbers are reproduced from the corresponding
references; design axes such as retrieval architecture and agent orchestration
are already reported in Table 1, so we only carry the benchmark size and each
system’s self-reported principal metric here. The metric symbols in the last
column are: ESR = end-to-end execution success rate; PF = physical fidelity
(as defined by ChatCFD); Pass = pass rate on the benchmark the system
itself proposes; “n.r.” = not reported. The IteraSim RAG row reports the
retrieval-tier mean coverage measured in this manuscript; the executability-
tier number is the natural extension and will be filled by the full pipeline
run described in Sec. 3.3. The‡mark on FoamGPT indicates the Qwen-
2.5-7B fine-tune baseline of [20]; larger backbones such as GPT-4.1 mini
reach up to39%on the same 202-case suite. The comparison should be read
with care: each system reports on a benchmark of its own choosing, and
the metric definitions differ. Despite the differing benchmarks, two conclu-
sions hold across systems. The addition of any retrieval-augmented context
is worth tens of percentage points of executability – a point already made
by Pandeyet al.[11] and reconfirmed in the Foam-Agent 1.0 to 2.0 transi-
tion [12, 19]. Beyond that, IteraSim RAG is the only system in Table 5 whose
retrieval layer combines query expansion, rank fusion and MMR re-ranking
with explicit dual-mode routing and a separate canonical-knowledge layer.
23

Table 5: Benchmark size and principal success metric for IteraSim RAG and the published
LLM-for-CFD systems.
System Bench.NReported success
metric (%)
OpenFOAMGPT (GPT-4o, no RAG) [11] 6 ESR 16.7
OpenFOAMGPT (o1+ RAG) [11] 6 ESR 100
MetaOpenFOAM [18] 8 Pass 85
FoamGPT [20] 202 ESR 26‡
Foam-Agent 1.0 [12] 110 ESR 82.6
Foam-Agent 2.0 [19] 110 ESR 88.2
ChatCFD [21] 315 ESR 82.1 / PF 68.1
CFDAgent [22] n.r. n.r.
SwarmFoam [23] 25 Pass 84
IteraSim RAG 28 Retr. 77.9
Every other system relies on a single-mode flat retrieval, with the hierarchical
multi-index of Foam-Agent 2.0 as the sole exception. The principal metric
reported here is therefore the retrieval-tier coverage of77.9%on the 28-case
benchmark. The executability-tier number is the natural extension and will
be reported, on the same benchmark, on completion of the full pipeline run.
Figure 7 shows the converged simulation fields for the six zero-shot refer-
encecases, solvedontheOpenFOAMtoolchainthattheback-endtargetsand
organised in the column structure adopted by OpenFOAMGPT [11]: (a) lid-
drivencavitywithicoFoam, velocity magnitude; (b)pitzDailybackward-
facing step withsimpleFoam+k–ε, velocity magnitude; (c)hotRoombuoy-
ant convection withbuoyantBoussinesqSimpleFoam(floor 320 K, ceiling
300 K), velocity magnitude; (d)damBreakwithinterFoam, water volume
fractionα water; (e)LagrangianparticlecolumnwithMPPICFoam, carrier-phase
airvelocitymagnitude; (f)2DrotatingmixerVesselwithsimpleFoam,veloc-
ity magnitude. These are the canonical OpenFOAM configurations against
which the agent-generated cases of Sec. 3.3 are compared, and they pro-
vide a visual sanity check on the physics that IteraSim RAG is expected to
reproduce.
The dual-mode router (addressing limitation L2) is the strongest single
contributor: parameter-modification queries, which are routed determinis-
24

(a) Cavity flow
(b) PitzDaily
(c) Hot room
(d) Dam break
 (e) Particle column
 (f) Mixer vessel
0.0 0.2 0.4 0.6 0.8 1.0
|U| (m/s)
0 2 4 6 8 10
|U| (m/s)
0.00 0.05 0.10 0.15 0.20 0.25
|U| (m/s)
0.0 0.2 0.4 0.6 0.8 1.0
water
0.00 0.05 0.10 0.15 0.20 0.25
|Uair| (m/s)
0 1 2 3 4
|U| (m/s)Figure 7: Converged simulation fields for the six zero-shot Category A reference cases.
Panels (a)–(f) correspond to the six configurations detailed above.
tically to the workflow mode with the active tool locked, account for the
highest mean retrieval score and the largest count of perfect-coverage cases.
Thethree-stageretrievalpipeline(addressingL1)showsitseffectmostclearly
in Category D, where the turbulence-swap queries require simultaneous re-
covery of model-name tags, wall-function tags and field-equation tags from
different regions of the corpus. The RRF stage is what keeps the turbu-
lence tag and the wall-function tag both in the top-K, even when they have
different similarity profiles against the original query alone. The canonical-
knowledge layer (addressing L3) is not directly observable from the retrieval
tier, by construction, since it is injected at the orchestration stage; its effect
will be measured against the executability tier in Sec. 3.3.
Two limitations affect the numbers above. The scoring rule in Eq. (3)
is case-sensitive and rewards exact tag presence. It therefore under-counts
retrievals that recover a relevant chunk under a synonymous name, for exam-
ple “low-Reynolds wall functions” in place ofnutLowReWallFunction. The
reported77.9%is best read as a conservative lower bound on operational
25

retrieval quality. The other caveat concerns the two outliers in Category A,
which flag a corpus-coverage gap rather than a pipeline weakness. The niche
solversMPPICFoamandbuoyantBoussinesqSimpleFoamrequire targeted in-
gestion of upstream OpenFOAM documentation that is not currently part
of the curated corpus. This gap can be closed at the ingestion layer alone;
no change to retrieval, routing or orchestration is needed.
3.5. Attribution of retrieval performance to individual architectural compo-
nents
Although a full factorial ablation (query expansion only, RRF only, MMR
only, canonical layer only, full pipeline) is deferred to a companion technical
note, the per-categorystructureof the retrieval-tier results inSec. 3.1 already
permits component-wise attribution of the observed performance, because
each of the four benchmark categories loads asymmetrically on a different
architectural contribution.
The dual-mode router (contribution L2, addressing Sec. 1) is the domi-
nant driver of the90.7%mean score in Category C. Parameter-modification
queries carry an unambiguous active tool (typicallysimpleFoam,interFoam
orbuoyantSimpleFoam), which the keyword-driven intent classifier reliably
routes to workflow mode with the tool-conditioned filter engaged. The tool-
dependency walk (Sec. 2.4) then pulls the adjacentfvSchemes,fvSolution,
boundary and post-processing chunks into the context in a single pass, which
is what pushes six of the ten Category C cases to perfect retrieval and none
below73%. Under a hypothetical single-mode ablation, the same parameter-
modification queries would compete for top-Kslots against globally-similar
physics documentation and would almost certainly lose the operational speci-
ficity that the tool-conditioned filter provides.
The three-stage retrieval pipeline (contribution L1) is the principal driver
of the74.1%mean of Category D. Zero-shot turbulence swaps require simul-
taneous recovery of at least three distinct token classes from different regions
of the corpus: the closure name (e.g.kOmegaSST), the wall-function tag (e.g.
omegaWallFunction), and the field-equation tag (e.g. the transport equa-
tion forω). A single-query embedding search preferentially recovers the class
most similar to the surface form of the user prompt and loses the other two.
Reciprocal Rank Fusion across the physics, keyword and troubleshooting re-
formulations of the query keeps all three classes in the top-Kcandidate set
evenwhentheirindividualsimilarityscoresagainsttheoriginalqueryarehet-
erogeneous, and Maximal Marginal Relevance then enforces that the three
26

classes are represented in the final context rather than displaced by near-
duplicate copies of the most similar tag. The near-uniform Category D floor
of60%across eight distinct closure families is consistent with this interpre-
tation: the pipeline never entirely loses the wall- function or field-equation
dimension of any given swap.
The canonical-knowledge layer (contribution L3) is by design invisible at
the retrieval tier, since it is injected at the orchestration stage rather than
at retrieval. Its stabilising role is nonetheless indirectly observable in Cate-
gory D as well: the wall-function/y+table and the turbulence-selection deci-
sion tree inK(Sec. 2.5) are precisely the artefacts the Reviewer would need
to correct a spuriously-selected wall function returned by the InputWriter,
and the tight standard deviation of Category D (σ s=8.8%) suggests that
no turbulence swap collapses to an uninterpretable state thatKcannot re-
anchor. The direct executability-tier confirmation of this claim is the subject
of Sec. 3.3 and of ongoing work.
Two failure-attribution observations complete the picture. First, the two
Category A outliers (Particle Columnats=26.7%,Hotroomats=33.3%)
shareacorpus-coveragecause,notaretrieval-algorithmcause: theunderlying
niche solvers (MPPICFoam,buoyantBoussinesqSimpleFoam) are underrepre-
sented in the ingested tutorial corpus, and no amount of query expansion or
re-ranking can recover chunks that are not present inD. Closing this gap
is a pure ingestion problem and does not motivate any change to the re-
trieval, routing, orchestration or canonical-knowledge components. Second,
the milder Category D under-counts on variant-spelling tags (RNGkEpsilon,
LESmodel Smagorinsky) reflect the case-sensitivity of the scoring rule in
Eq. (3) rather than a retrieval failure; the operational retrieval quality is
therefore bounded below by the reported77.9%.
3.6. An error taxonomy of the Reviewer corrective loop
The Reviewer-loop budget of ten cycles (Sec. 2.6) is half the twenty-
cycle limit reported by Foam-Agent [12] and was chosen on the premise that
most recoverable failures are localisable within a small number of dictionary
patches while pathological failures should fail fast rather than accumulate
token cost. Under this operational envelope, the failures observed in internal
end-to-end runs of the orchestrator group cleanly into four families, each
associated with a distinct architectural handle.
27

Class I — Boundary-condition patch loops..The Reviewer identifies an in-
compatible boundary condition (for example, azeroGradientpressure at an
inlet that also fixes velocity) and requests a targeted patch to a single field
file. This class is bounded and terminates within two to three cycles because
the canonical-knowledge boundary-condition selector (Sec. 2.5) provides a
deterministic target for the correction.
Class II — Turbulence-closure and wall-function mismatches..The Reviewer
detects that the InputWriter has combined a low-Reynolds wall function with
a high-Reynolds defaulty+regime, or that the closure named in the manifest
requires transport fields not written to the0/directory. This class relies on
the wall-function /y+table inKand, empirically, resolves within three to
four cycles.
Class III — Mesh-quality andsnappyHexMeshfailures..AcheckMeshvi-
olation (non-orthogonality above the solver tolerance, skewed cells, or an
unrecoverablesnappyHexMeshrefinement failure) is surfaced by the Run-
ner and interpreted by the Reviewer as a meshing-side patch (increase in
nSmoothScale,additionofarefinementregion,orrelaxationofmaxNonOrtho).
This is the class that consumes the largest fraction of the ten-cycle budget
when it triggers.
Class IV — Corpus-coverage failures..When the query targets a niche solver
family (MPPICFoam, Boussinesq buoyancy, wave-energy converters outside the
WEC corpus), the InputWriter emits a plausible but syntactically-off dictio-
nary, the Reviewer requests a corrective patch, and the retrieval layer returns
no meaningfully-different chunk on the corrective query either. The loop hits
the ten-cycle bound and terminates. This class is architecturally identical to
the two Category A outliers in Fig. 4: it is a corpus-coverage problem, not a
retrieval-algorithm or orchestration problem. The ten-cycle bound is there-
fore the correct place to fail fast — the loop cannot recover information that
is not in the corpus, and additional cycles would only inflate the token cost
of the failure.
The classification above suggests a natural specialisation of the Reviewer
prompt on the class of the observed failure, and in particular a distinct patch
strategy for Class III mesh failures versus Class I/II dictionary failures. This
is retained as an item of future work.
28

3.7. Retrieval fidelity versus physical fidelity: relation to ChatCFD and com-
plementary evaluation
The evaluation reported in Sec. 3 measures two distinct notions of system
quality: the retrieval-tier coverage of the physics/solver/keyword tags asso-
ciated with each benchmark case (Eq. 3), and the end-to-end executability
of the reference cases verified in Sec. 3.3. Neither metric is a direct probe
of thephysical fidelityof the converged solution, in the sense of whether
the reported flow field satisfies the target boundary conditions, reproduces
published reference profiles, and agrees with a physical criterion beyond the
mere fact that the solver reachedendTime.
This is precisely the direction in which ChatCFD [21] extends the LLM-
for-CFD evaluation surface. Their physical-fidelity evaluator scores an Open-
FOAM run against physically meaningful conditions on the converged field
(mass and momentum conservation, integral force balances on wall patches,
and agreement with reference profiles for canonical validation cases), so that
a case that runs to completion but converges to a physically incorrect steady
state is correctly assigned a lower score than a case that converges to the
intended flow.
The two evaluation surfaces are complementary rather than competing.
Thethree-stageretrievalpipelineanddual-moderouterintroducedinthispa-
peractattheearlieststageofthepipeline(beforegeneration), theArchitect–
InputWriter–Reviewer triad and the canonical-knowledge layer act at the in-
termediate stage (during generation and correction), and the ChatCFD-style
physical-fidelity evaluator acts at the terminal stage (after execution). A
system that scores highly on the retrieval tier generates syntactically-correct
OpenFOAM dictionaries; a system that additionally scores highly under a
ChatCFD-style evaluator has also converged to the intended physics. Inte-
grating the ChatCFD evaluator as a terminal reviewer call in the IteraSim
RAG orchestration loop is a natural next step and is left to future work.
Table 5 accordingly records retrieval coverage as the principal metric of the
present paper and the physical-fidelity axis as the principal metric of [21],
and the two rows should be read as measurements at different depths of the
same generation pipeline, not as competing point comparisons.
3.8. Threats to validity and honest bounds on the reported numbers
Threeresidualcaveatsaffectthereadingoftheretrieval-tierresults. First,
the scoring rule of Eq. (3) is case-sensitive by construction, so retrievals that
recover a relevant chunk under a synonymous surface form (“low-Reynolds
29

wall functions” fornutLowReWallFunction) are under-counted; the reported
77.9%is a conservative lower bound on operational retrieval quality. Second,
the benchmark is a 28-case instrument covering zero-shot, few-shot, param-
eter modification and turbulence swap axes; it is deliberately small to per-
mit exact tag-level scoring against a curated ground truth, but it does not
exercise multi-physics regimes (conjugate heat transfer, combustion, fluid–
structure interaction) that lie outside the present ingested corpus. Third,
the executability evidence reported here is the measured reference-case run
of Sec. 3.3: every one of the six base configurations underlying the 28 bench-
mark cases is runnable toendTimeon OpenFOAM v2506 without aFOAM
FATALexit, which establishes the executability ceiling of the benchmark. The
complementary agent-driven run of the full Architect–InputWriter–Reviewer
orchestrator across all 28 cases, scored on the same rubric, is deferred to the
companion technical note; the present paper therefore reports retrieval cover-
age as its principal measured metric and makes no agent-driven executability
claim.
4. Conclusion
We have introduced IteraSim RAG, a retrieval-augmented agentic back-
end for OpenFOAM that addresses three limitations of prior LLM-for-CFD
systems: the reliance on single-query flat retrieval, the application of a uni-
form retrieval strategy to operationally-different requests, and the entangle-
ment of drafting with self-review in a single generation agent. Each contri-
bution acts at a different depth of the same per-turn pipeline, and each can
be configured, ablated or replaced on its own. The first is a three-stage re-
trieval pipeline: an LLM expands the query into several variants, Reciprocal
Rank Fusion merges the ranked lists this produces, and Maximal Marginal
Relevance re-ranks the candidates that survive. The second is a determin-
istic router that sends tool-conditioned workflow queries and corpus-wide
expert queries down separate retrieval paths. The third is an Architect–
InputWriter–Reviewer agent triad, backed by a static canonical-knowledge
layer that covers solver selection, turbulence closures, boundary conditions
and finite-volume defaults.
On a 28-case benchmark spanning zero-shot setup, few-shot generalisa-
tion, single-parameter modifications and zero-shot turbulence-model swaps,
the retrieval-tier evaluation returns a mean coverage of77.9%against ex-
pected physics/solver/keyword tags. The parameter-modification category
30

reaches90.7%and the few-shot category82.0%, while the turbulence-swap
category holds a74.1%mean withσ s=8.8%and no case below60%. The
two under-covered cases in the zero-shot category are attributable to corpus-
coveragegapsinnichesolverdocumentation(MPPICFoam,buoyantBoussinesqSimpleFoam)
and not to any retrieval, routing or orchestration defect. End-to-end latency
stays within the sub-3-second interactive regime despite the four-fold query
inflation introduced by the expansion stage.
At the executability tier, all 28 base configurations meshed cleanly and
reachedendTimeon OpenFOAM v2506, confirming that the benchmark is
runnable on the target toolchain, and both corrupted probe cases were di-
agnosed and repaired inside the bounded Reviewer loop from the solver log
alone. Scoring the agent-driven cases across the full benchmark depends on
the cold-start generation path and is deferred to a companion technical note.
Three directions are highlighted for future work. First, expansion of the
ingested corpus to cover niche solver families (Lagrangian particle clouds,
Boussinesqbuoyancy,free-surfacepiston-drivenflows,conjugate-heat-transfer
and combustion families) is expected to close the two Category A outliers
without architectural modification. Second, a full factorial component-level
ablation (query expansion only, RRF only, MMR only, canonical layer only,
and the full pipeline) at the retrieval tier is planned. Together with the corre-
sponding executability-tier run of the full Architect–InputWriter–Reviewer
orchestrator across all 28 cases, it is the natural quantitative complement
to the qualitative component attribution reported above. Third, adding a
physical-fidelity evaluator as a terminal reviewer stage would extend the eval-
uation surface from retrieval coverage and executability to physical correct-
ness of the converged solution. This would complete a three-tier assessment
(retrieval→executability→physical fidelity) of LLM-for-CFD agentic sys-
tems.
CRediT authorship contribution statement
Pratyush Kumar:Conceptualization, Methodology, Software, Investi-
gation, Data curation, Formal analysis, Writing – original draft, Writing –
review & editing, Visualization, Funding acquisition.
As the sole author, P.K. is responsible for all aspects of this work.
31

Declaration of competing interest
The author has a financial interest in IteraSim, which develops the pro-
prietary retrieval engine described in this paper. The released benchmark,
scoring rubric and figure scripts are independent of the proprietary compo-
nentsandaremadeopenlyavailable. Theauthordeclaresnoothercompeting
financial or personal interests.
Data availability
The28-casebenchmarkspecification,machine-readablebenchmarkqueries,
Reviewer-loopFOAM FATALprobes, the retrieval-tier scoring rubric of Eq. (3),
the leave-one-out ablation records of Table 3 (the 30 ablation queries with
theirper-queryscoresundereachconfiguration),thereference-caseexecutabil-
ity records and the figure-regeneration scripts are released under the MIT
licence athttps://github.com/iterasim/iterasim-rag-publicand are
archived with a persistent identifier on Zenodo [34]. The IteraSim RAG re-
trieval engine, the Architect–InputWriter–Reviewer orchestrator, the canon-
ical knowledge layer and the ingestion pipeline are the commercial IP of It-
eraSim and are available for academic collaboration or commercial licensing
on request through the corresponding author.
Acknowledgements
P.K. gratefully acknowledges support from the Swiss National Science
Foundation(SNSF)throughtheSwissPostdoctoralFellowship,grant#234158.
The computations reported in this paper were performed on research com-
puting infrastructure at ETH Zürich. P.K. thanks Prof. Siddhartha Mishra
(Seminar for Applied Mathematics, ETH Zürich) for hosting the research
programme under which this work was carried out.
Declaration of generative AI and AI-assisted technologies in the
manuscript preparation process
During the preparation of this work the author used Claude (Anthropic)
in order to edit and copy-edit the manuscript text, assist with LATEX format-
ting, and check the completeness and consistency of the reference list. After
using this tool, the author reviewed and edited the content as needed and
takes full responsibility for the content of the published article.
32

References
[1] O. S. Ismail, G. T. Adewoye, Analyses and modeling of laminar flow in
pipes using numerical approach, Journal of Software Engineering and
Applications 5 (9) (2012) 653–658.doi:10.4236/jsea.2012.59076.
[2] H. Alemi, S. A. Nourbakhsh, M. Raisee, A. F. Najafi, Effects of volute
curvature on performance of a low specific-speed centrifugal pump at
design and off-design conditions, Journal of Turbomachinery 137 (4)
(2015) 041009.doi:10.1115/1.4028766.
[3] S. Safaei, R. Zeynali, B. Safaei, Modeling study of propane/propylene
membrane separation using CFD method, Macromolecular Theory and
Simulations 30 (3) (2021) 2000092.doi:10.1002/mats.202000092.
URLhttps://doi.org/10.1002/mats.202000092
[4] T. Sochi, Fluid flow at branching junctions, International Journal
of Fluid Mechanics Research 42 (1) (2015) 59–81.doi:10.1615/
InterJFluidMechRes.v42.i1.50.
[5] P. Kumar, S. K. Saha, A. Sharma, Experimental and CFD-DEM study
onlocalpackingdistributionandthermofluidicanalysisofbinarypacked
bed, Chemical Engineering Science 282 (2023) 119372.doi:10.1016/
j.ces.2023.119372.
URLhttps://doi.org/10.1016/j.ces.2023.119372
[6] P. Kumar, S. K. Saha, A. Sharma, D. Sharma, P. Chaudhuri, CFD
investigation of helium gas flow in sphere packed (Pebble bed) in a
rectangular canister using OpenFOAM, Fusion Engineering and Design
172 (2021) 112858.doi:10.1016/j.fusengdes.2021.112858.
URLhttps://doi.org/10.1016/j.fusengdes.2021.112858
[7] H. Jasak, A. Jemcov, Z. Tukovic, OpenFOAM: A C++ library for com-
plex physics simulations, in: International Workshop on Coupled Meth-
ods in Numerical Dynamics, 2007, pp. 1–20.
[8] H. G. Weller, G. Tabor, H. Jasak, C. Fureby, A tensorial approach to
computational continuum mechanics using object-oriented techniques,
Computers in Physics 12 (6) (1998) 620–631.doi:10.1063/1.168744.
URLhttps://doi.org/10.1063/1.168744
33

[9] S. K. Samal, P. Kumar, S. K. Saha, Comparative study of surface rough-
ness models in the hydro-thermal characterization of flow over rough
walls, Physics of Fluids 35 (9) (2023) 095106.doi:10.1063/5.0165497.
URLhttps://doi.org/10.1063/5.0165497
[10] P. Kumar, S. Sarkar, S. K. Saha, Numerical study of bouncing Lei-
denfrost viscoplastic drops, Physics of Fluids 38 (3) (2026) 032007.
doi:10.1063/5.0316065.
[11] S. Pandey, R. Xu, W. Wang, X. Chu, OpenFOAMGPT: A retrieval-
augmented large language model (LLM) agent for OpenFOAM-based
computational fluid dynamics, Physics of Fluids 37 (3) (2025) 035120.
doi:10.1063/5.0257555.
URLhttps://doi.org/10.1063/5.0257555
[12] L. Yue, N. Somasekharan, T. Zhang, Y. Cao, Z. Chen, S. Di,
S. Pan, Foam-Agent: Towards automated intelligent CFD workflows,
arXiv:2505.04997 (2025).
URLhttps://arxiv.org/abs/2505.04997
[13] K. Duraisamy, G. Iaccarino, H. Xiao, Turbulence modeling in the age
of data, Annual Review of Fluid Mechanics 51 (2019) 357–377.doi:
10.1146/annurev-fluid-010518-040547.
URLhttps://doi.org/10.1146/annurev-fluid-010518-040547
[14] A. Beck, D. Flad, C.-D. Munz, Deep neural networks for data-driven
LES closure models, Journal of Computational Physics 398 (2019)
108910.doi:10.1016/j.jcp.2019.108910.
URLhttps://doi.org/10.1016/j.jcp.2019.108910
[15] S. L. Brunton, B. R. Noack, P. Koumoutsakos, Machine learning for
fluid mechanics, Annual Review of Fluid Mechanics 52 (2020) 477–508.
doi:10.1146/annurev-fluid-010719-060214.
URLhttps://doi.org/10.1146/annurev-fluid-010719-060214
[16] D. A. Boiko, R. MacKnight, B. Kline, G. Gomes, Autonomous chemical
research with large language models, Nature 624 (2023) 570–578.doi:
10.1038/s41586-023-06792-0.
URLhttps://doi.org/10.1038/s41586-023-06792-0
34

[17] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen,
J. Tang, X. Chen, Y. Lin, et al., A survey on large language model
based autonomous agents, Frontiers of Computer Science 18 (6) (2024)
186345.doi:10.1007/s11704-024-40231-1.
URLhttps://doi.org/10.1007/s11704-024-40231-1
[18] Y. Chen, X. Zhu, H. Zhou, Z. Ren, MetaOpenFOAM: an LLM-based
multi-agent framework for CFD, arXiv:2407.21320 (2024).
URLhttps://arxiv.org/abs/2407.21320
[19] L. Yue, N. Somasekharan, T. Zhang, Y. Cao, S. Pan, Foam-Agent 2.0:
An end-to-end composable multi-agent framework for automating CFD
simulation in OpenFOAM, arXiv:2509.18178 (2025).
URLhttps://arxiv.org/abs/2509.18178
[20] L. Yue, P. Xu, T. Zhang, N. Somasekharan, S. Pan, FoamGPT:
Fine-tuning large language model for agentic automation of CFD
simulations with OpenFOAM, in: Machine Learning and the Physical
Sciences Workshop, NeurIPS, 2025.
URLhttps://ml4physicalsciences.github.io/2025/files/
NeurIPS_ML4PS_2025_271.pdf
[21] E. Fan, K. Hu, Z. Wu, J. Ge, J. Miao, Y. Zhang, H. Sun, W. Wang,
T. Zhang, ChatCFD: A large language model-driven agent for end-to-
end computational fluid dynamics automation with structured knowl-
edge and reasoning, Advanced Intelligent Discovery 2 (3) (2026)
e202500174.doi:10.1002/aidi.202500174.
[22] Z. Xu, L. Wang, C. Wang, Y. Chen, Q. Luo, H.-D. Yao, S. Wang,
G. He, CFDAgent: A language-guided, zero-shot multi-agent system
for complex flow simulation, Physics of Fluids 37 (11) (2025) 117124.
doi:10.1063/5.0294696.
URLhttps://doi.org/10.1063/5.0294696
[23] C. Yang, Y. Wang, J. Tang, H. Qu, Z. Zou, Y. Liu, C. Deng,
Z. Qiu, M. Ding, SwarmFoam: An OpenFOAM multi-agent system
based on multiple types of large language models, arXiv preprint
arXiv:2601.07252 (2026).
URLhttps://arxiv.org/abs/2601.07252
35

[24] G. W. Furnas, T. K. Landauer, L. M. Gomez, S. T. Dumais, The vo-
cabulary problem in human-system communication, Communications of
the ACM 30 (11) (1987) 964–971.doi:10.1145/32206.32212.
URLhttps://doi.org/10.1145/32206.32212
[25] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, et al., Self-
refine: Iterative refinement with self-feedback, in: Advances in Neural
Information Processing Systems, Vol. 36, 2023, pp. 46534–46594.
[26] G. V. Cormack, C. L. A. Clarke, S. Buettcher, Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods, in: Proc.
32nd ACM SIGIR, 2009, pp. 758–759.doi:10.1145/1571941.1572114.
URLhttps://doi.org/10.1145/1571941.1572114
[27] J. Carbonell, J. Goldstein, The use of MMR, diversity-based reranking
forreorderingdocumentsandproducingsummaries, in: Proc.21stACM
SIGIR, 1998, pp. 335–336.doi:10.1145/290941.291025.
URLhttps://doi.org/10.1145/290941.291025
[28] Y. A. Malkov, D. A. Yashunin, Efficient and robust approximate nearest
neighbor search using hierarchical navigable small world graphs, IEEE
Transactions on Pattern Analysis and Machine Intelligence 42 (4) (2020)
824–836.doi:10.1109/TPAMI.2018.2889473.
URLhttps://doi.org/10.1109/TPAMI.2018.2889473
[29] I. Team, IteraSim RAG benchmark and evaluation kit: canonical-
knowledge-layer structure,https://github.com/iterasim/
iterasim-rag-public, section-level structure of the curated ex-
pert notes on solver selection, boundary-condition idioms, mesh-
quality criteria, numerical-stability heuristics and crash diagnosis
(CANONICAL_LAYER.md). The full text of the notes is proprietary and is
not redistributed (2026).
[30] CFD Online, OpenFOAM technical discussion forum,https://www.
cfd-online.com/Forums/openfoam/, practitioner threads used as
source material for the troubleshooting question–answer set. Accessed
13 July 2026.
[31] OpenCFD Ltd., OpenFOAM v2506 tutorial and reference case suite,
https://www.openfoam.com/documentation/tutorial-guide, refer-
36

ence dictionaries and canonical case distribution bundled with the ESI-
OpenCFD OpenFOAM release (2025).
[32] SandiaNationalLaboratories, WEC-OpenFOAMopendatasets,https:
//github.com/sandialabs/WEC-OpenFOAM-Open-Datasets, wave en-
ergy converter templates, meshes, and Reference Model 6 (RM6) cou-
pled multiphase / rigid-body examples (2024).
[33] J. J. Pan, J. Wang, G. Li, Survey of vector database management sys-
tems, in: The VLDB Journal, Vol. 33, 2024, pp. 1591–1615.doi:
10.1007/s00778-024-00864-x.
URLhttps://doi.org/10.1007/s00778-024-00864-x
[34] P. Kumar, IteraSim RAG: benchmark and evaluation kit for retrieval-
augmented agentic OpenFOAM case generation, Zenodo, version v1.0-
submission (2026).doi:10.5281/zenodo.21346132.
Appendix A. Summary of the RAG Model
Forconveniencewesummarisetheend-to-endbehaviourofIteraSimRAG
as a single per-turn pipeline. Letqdenote a raw user query,Dthe indexed
corpusofOpenFOAMdocumentation, tutorialsandcuratedquestion–answer
pairs, andKthe static canonical-knowledge layer. For every turn, the system
executes the following six steps.
1.Mode routing.A deterministic keyword-driven classifier built over
an 18-category CFD ontology assignsqeither toworkflowmode, with
retrieval depthK=5and MMR trade-offλ=0.6, or toexpertmode,
withK=8andλ=0.5.
2.Query expansion.A general-purpose LLM rewritesqinto three se-
mantically diverse reformulations{q 1, q2, q3}targeting the physics, the
OpenFOAM keyword surface and a troubleshooting angle of the origi-
nal question, yielding the variant setQ={q, q 1, q2, q3}.
3.Multi-query retrieval.Each variant˜q∈ Qis independently em-
bedded and used to retrieve the topK fetch= max(2K, K+4)candidate
chunks fromDvia HNSW search under cosine distance, producing four
ranked listsR.
4.Rank fusion.The four lists are merged with Reciprocal Rank Fusion
[Eq. (1)] withk=60, with duplicates collapsed by a 120-character text
fingerprint, to produce a unified candidate setC.
37

5.Diversity re-ranking.Maximal Marginal Relevance [Eq. (2)] is ap-
plied toCat the mode-specificλuntilKchunks have been selected;
these are concatenated into the retrieved context blockX.
6.Orchestrated generation and review.The augmented prompt
P=⟨K,X, q⟩is consumed by the Architect, InputWriter and Reviewer
in sequence: the Architect emits a case manifest, the InputWriter pro-
duces the OpenFOAM dictionaries file-by-file, and the Reviewer com-
piles, executes and, if necessary, iterates corrective patches up to a
bounded depth of ten cycles. Execution is delegated to the Runner;
only the Runner touches the file system or the OpenFOAM process
tree.
Steps 1–5 constitute the retrieval-augmented generation core and ad-
dress limitations (L1) and (L2) identified in Sec. 1; step 6 together with
the canonical-knowledge layerKaddresses limitation (L3). The complete
pipeline runs as a single forward pass per user turn, with the Reviewer loop
the only source of additional latency, and is summarised graphically in Fig. 1.
Appendix B. Per-Case Retrieval Results
Table B.1 reports the per-case breakdown across all 28 benchmark cases
together with per-category and overall averages. The column structure mir-
rorstheper-caseperformancetablesofMetaOpenFOAM[18]andChatCFD[21],
but is populated with the retrieval-tier metrics measured against the released
benchmark in this manuscript.
Appendix C. Prompt Used for the End-to-End Executability Com-
parison
Thenatural-languagepromptusedtoseedtheagent-generatedcaseshown
in Fig. 6 is reproduced verbatim below. The prompt embeds the standard
OpenFOAM v2506 tutorial parameters (domain, boundary conditions, fluid
properties, time-stepping and solver settings) so that the RAG back-end re-
ceives the same information as an experienced practitioner would provide.
No additional context or example is supplied at inference time.
Appendix C.1. Lid-driven cavity (icoFoam)
The lid-driven cavity is the standard incompressible laminar-flow bench-
markoftheOpenFOAMtutorialsuite. Thedomainisatwo-dimensionalunit
38

square of physical size0.1m×0.1m×0.01m (a single cell in the out-of-
plane direction), discretised by a structured hexahedral mesh with20×20×1
cells generated throughblockMeshwith a scale factor of0.1. The top wall
(movingWall) is prescribed afixedValuevelocity of(1,0,0)m/s; the re-
maining walls (fixedWalls) usenoSlipand the front/back patches use the
emptytype appropriate for a two-dimensional setup. The pressure bound-
ary conditions arezeroGradienton all solid walls. The fluid is Newtonian
with a kinematic viscosity ofν= 10−2m2/s. The solver isicoFoamwith
a first-order Euler time integration andGauss lineardivergence/Laplacian
schemes; pressure is solved byPCGwith theDICpreconditioner and velocity
by asmoothSolverwithsymGaussSeidel, both under the standard con-
vergence tolerances of10−6and10−5respectively. The PISO loop uses two
correctors and no non-orthogonal correctors. The simulation is advanced
fromt= 0tot= 0.5s with a time step of∆t= 5×10−3s and a write
interval of0.1s. The agent is asked to produce a complete case directory (0/,
constant/,system/) matching these settings, invokeblockMeshfollowed by
icoFoam, and confirm that the run reachesendTimewithout aFOAM FATAL
exit.
39

Table B.1: Per-case retrieval-tier results on the 28-case IteraSim RAG benchmark.
Case Retrieval-tier metrics (measured)
ID Title Solvers(%)τ(%) Solvert r(s)
A1 Cavity flowicofoam76.0 70.0✓2.92
A2 PitzDailysimplefoam76.0 70.0✓2.01
A3 Hotroombuoyantboussinesqsimplefoam33.3 41.7 – 2.11
A4 Dam breakinterfoam64.4 55.6✓2.17
A5 Particle columnmppicfoam26.7 33.3 – 2.35
A6 Mixed vesselpimplefoam78.2 72.7✓2.01
Avg. A(n= 6) –59.1 57.2 4/6 2.26
B1 Cavity→PitzDailysimpleFoam100.0 100.0✓2.20
B2 PitzDaily→HotroomsimpleFoam80.0 75.0✓2.14
B3 Cavity→unsteadypimpleFoam80.0 75.0✓2.07
B4 DamBreak→bubbleinterFoam68.0 60.0✓2.22
Avg. B(n= 4) –82.0 77.5 4/4 2.16
C1 Cavity:U topchange – 100.0 100.0✓1.93
C2 Cavity: unsteady BC – 80.0 75.0✓2.30
C3 PitzDaily: mesh×2– 100.0 100.0✓2.01
C4 Hotroom:T-BC change – 100.0 100.0✓2.05
C5 Hotroom: fluid swap – 100.0 100.0✓2.34
C6 DamBreak: column shift – 73.3 66.7✓2.10
C7 DamBreak: oil swap – 100.0 100.0✓1.97
C8 Particle: size/density – 80.0 75.0✓2.92
C9 MixedVessel:ωchange – 73.3 66.7✓2.08
C10 PitzDaily: endTime – 100.0 100.0✓2.08
Avg. C(n= 10) –90.7 88.3 10/10 2.18
D1k-ε→RNG – 60.0 50.0✓2.17
D2k-ε→k-ωSST – 65.7 57.1✓2.03
D3k-ε→LRR RSM – 73.3 66.7✓2.39
D4k-ε→k-k L-ω– 77.1 71.4✓2.04
D5 Laminar→k-εicoFoam68.6 85.7 – 2.01
D6 LES dynK→Smagorinsky – 84.0 80.0✓2.04
D7 Hotroom kEps→SSTbuoyantBoussinesqSimpleFoam84.0 80.0✓2.06
D8 MixedVessel +k-ε– 80.0 75.0✓2.16
Avg. D(n= 8) –74.1 70.7 7/8 2.11
Overall(n= 28) –77.9 75.1 25/28 2.17
40