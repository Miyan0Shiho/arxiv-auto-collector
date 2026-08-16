# D3D-GEN: Robot-Aware Domain-Grounded Interactive 3D World Generation for Social Robotics

**Authors**: Anh Duc Do, Volodymyr Scherbyna, Tai Duc Nguyen, Spaarsh Thakkar, Zhengcheng Shen, Teham Buiyan, Archan Misra, Linh Kästner

**Published**: 2026-08-12 10:03:01

**PDF URL**: [https://arxiv.org/pdf/2608.11876v1](https://arxiv.org/pdf/2608.11876v1)

## Abstract
Training and validation of Embodied AI for social navigation critically depends on realistic simulation environments, yet many current approaches fail to find a balance between realism and simulability. We propose D3D-GEN, a novel world generation system that combines a domain agent with a retrieval-augmented generation (RAG) pipeline grounded in that domain. Our system enables users to rapidly generate domain-grounded, fully interactive 3D worlds by automating both the collection of domain knowledge and the synthesis of realistic floorplans and object placements, without dependence on any fixed 3D model database. Given a domain description prompt, the research agent collects publicly accessible domain-specific data and constructs a persistent domain database. Using this database, our RAG pipeline generates plausible floorplans and object placements by dynamically querying a user-provided semantic database, which can be easily extended or modified. The output is a fully interactive 3D world loadable by the popular simulators Isaac Sim and Gazebo. With our approach, we have built databases for several common domains (indoor residential, hospital, office) and generated dozens of distinct, plausible simulation environments for each domain. We present D3D-GEN with a local web frontend that facilitates rapid, interactive world generation for robot simulation.

## Full Text


<!-- PDF content starts -->

D3D-GEN: Robot-Aware Domain-Grounded Interactive 3D World
Generation for Social Robotics
Do Duc Anh1, V olodymyr Shcherbyna1,2, Duc Tai Nguyen1, Spaarsh Thakkar1,
Zhengcheng Shen3, Teham Buiyan4, Archan Misra1, Linh Kästner1
1Singapore Management University,2Technical University Berlin,3Max Planck Institute,4Technical University Braunschweig
Fig. 1: D3D-GEN is a novel domain-generalized approach to 3D world generation for embodied AI deployment. The system
first gathers domain knowledge into a database with legal regulations and conventions, then a user prompt is grounded in
the database. A room-level scene graph is constructed with semantic nodes and adjacency edges, along with the preliminary
assets of the room. The system then generates room polygons satisfying spatial and connectivity constraints, then places
assets according to the rules from the database. This pipeline results in an unlimited amount of realistic, logically sound
worlds. The resulting world is simulable and can be loaded in Isaac Sim and Gazebo for robotic simulation.
Abstract— Training and validation of Embodied AI for social
navigation critically depends on realistic simulation environ-
ments, yet many current approaches fail to find a balance be-
tween realism and simulability. We proposeD3D-GEN, a novel
world generation system that combines a domain agent with
a retrieval-augmented generation (RAG) pipeline grounded in
that domain. Our system enables users to rapidly generate
domain-grounded, fully interactive 3D worlds by automating
both the collection of domain knowledge and the synthesis of
realistic floorplans and object placements, without dependence
on any fixed 3D model database. Given a domain description
prompt, the research agent collects publicly accessible domain-
specific data and constructs a persistent domain database. Using
this database, our RAG pipeline generates plausible floorplans
and object placements by dynamically querying a user-provided
semantic database, which can be easily extended or modified.
The output is a fully interactive 3D world loadable by the
popular simulators Isaac Sim and Gazebo. With our approach,
we have built databases for several common domains (indoor
residential, hospital, office) and generated dozens of distinct,
plausible simulation environments for each domain. We present
D3D-GEN with a local web frontend that facilitates rapid,
interactive world generation for robot simulation.
I. INTRODUCTION
Robotics research increasingly relies on simulated environ-
ments for training and validation. A constant challenge for
simulated approaches is successfully bridging the simulation-
to-real (sim2real) gap [1]. The quality of the simulated 3Denvironment plays a pivotal role in the real-world transfer-
ability of simulation-native models. Environment diversity,
semantic soundness, and perceptual realism directly impact
training outcomes.
Agents trained on homogeneous environments develop navi-
gation strategies that generalize poorly to out-of-distribution
spaces [2]. Part of the field has converged on a narrow
set of training scenarios, predominantly sanitized residential
scenes [3]. Hospital wards, residential buildings, and com-
mercial real estate follow distinct legal regulations, spatial
conventions, and object distributions. These nuances cannot
be captured by a single one-size-fits-all approach.
Closing this gap requires simulation environments that are
not only physically interactive, but semantically grounded
in the conventions of the specific domain in which the
robot will operate. Current methods that prioritize perceptual
realism tend to produce physically unsound scenes, methods
that prioritize simulability generate domain-generic layouts
that lack semantic soundness. This is critical for robotics.
Robotics research needs environments that are operationally
viable, not just visually appealing scenes that do not transfer
to real deployments. Perceptual realism alone does not ensure
adequate working space for agents, especially when work-
ing with specific domains like office or hospital. Domain
grounding is therefore essential for robot operation.
arXiv:2608.11876v1  [cs.RO]  12 Aug 2026

Realism & Correctness Generation Accessibility Capabilities
Method Physical
SoundnessSemantic
GroundingDomain
GeneralityScalable
GenerationSimulation
ReadyNatural
LanguageVariable
AssetsGenerate
FloorplanPlace
Assets
Procedural generation
ProcTHOR [4]✓ ✗ ✗ ✓ ✓ ✗ ✓ ✓ ✓
Infinigen [5]✓ ✗∼ ∼✓ ✗ ✗ ✓ ✓
Diffusion-based generation
DiffuScene [6]∼✗ ✗ ✗ ✓ ✗ ✗ ✗ ✓
RoomDreamer [7]✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓
HouseDiffusion [8]✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗
Agentic generation
LayoutGPT [9]∼✗ ✗ ✓ ✓ ✓ ✗ ✗ ✓
PhyScene [10]✓ ✗ ✗ ✗ ✓ ✗ ✗ ✗ ✓
SAGE [11]✓ ✗ ✗∼✓ ✗ ✗ ✗ ✓
Holodeck [12]∼✗ ✗ ✓ ✗ ✓ ✓ ✓ ✓
D3D-GEN (Ours)✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
TABLE I: World generation approaches that are relevant to indoor scene generation for Embodied AI.
✓supports,✗does not support,∼partially supports.
We propose D3D-GEN to address these limitations. Our sys-
tem autonomously retrieves and organizes domain-specific
knowledge based on a research prompt, grounding layout
and object placement decisions in verifiable real-world con-
ventions rather than generic priors. A subsequent world
generation prompt allows both domain experts and laypeople
to describe the world the way they imagine it.
The result is a physically plausible, semantically sound 3D
environment ready for simulation, generated directly from
user intent and grounded in verifiable real-world rules. We
also provide generated worlds with semantic annotations,
allowing robot agents to be fully integrated into the robot
simulation pipeline. Our system is designed to scale horizon-
tally with an arbitrary number of parallel workers generating
worlds from the same or different prompts, at any pipeline
stage. We also provide a web frontend to provide a user-
friendly feedback-driven experience, with editable previews
generated after every pipeline stage.
II. RELATEDWORK
Generating indoor environments for Embodied AI has taken
three broad directions, each with its own trade-offs between
realism, physical soundness, and scalability.
Real-world scan datasets like HM3D [13] and Matter-
port3D [14] capture genuine spatial complexity, but the
meshes they produce are non-watertight and require signifi-
cant manual post-processing before they are compatible with
physics engines like Isaac Sim. These datasets also contain
no interactive objects; doors and drawers are static geometry
rather than articulated components, limiting their usefulness
for social navigation tasks that require object interaction.
Platforms like Habitat [15] and AI2-THOR [16] wrap such
datasets with physics engines, but remain restricted to the
domains covered by the datasets.
Designer-created datasets like 3D-FRONT [3] and Struc-tured3D [17] offer physically sound CAD assets, but their
layouts tend toward the sanitized end of the spectrum,
stripped of the clutter and imperfections that characterize
real human spaces. 3D-FRONT in particular has become the
default dataset for a large body of scene generation work, in-
cluding DiffuScene [6], InstructScene [18], SceneTeller [19],
RoomDreamer [7], and LayoutGPT [9], yet it has significant
limitations. The dataset is extremely large, with asset li-
braries running to hundreds of gigabytes, creating a substan-
tial barrier to entry for new domains or deployment contexts.
More importantly, 3D-FRONT covers only indoor residential
settings: bedrooms, living rooms, and dining areas. Models
trained or evaluated on these data inherit that restriction
and do not transfer to hospitals, airports, warehouses, or
public spaces where social robots are increasingly expected
to operate. Recent simulation platforms like RoboCasa [20]
and HumanTHOR [21] have begun addressing household
manipulation scenarios, but neither supports multi-domain
generation from a single pipeline.
Procedural approaches like ProcTHOR [4] and Infinigen [5]
address scalability and physical soundness, but tend to pro-
duce sterile, homogeneous layouts skewed toward simple
rectangular rooms.
Recent agentic and diffusion-based methods represent the
current frontier. PhyScene [10] uses differentiable guidance
during generation to regulate object quantity, prevent artic-
ulation collisions, and preserve navigable space. SAGE [11]
employs a physics critic that detects and iteratively corrects
placement failures before a scene is finalized. LLplace [22]
frames layout synthesis as an LLM reasoning task, letting a
language model directly propose room arrangements, but it
relies on a single-shot generation step without external do-
main grounding or multi-stage refinement. Graph2Scene [23]
generates 3D indoor scenes from interaction-aware scene
graphs, yet its graph vocabulary and asset library are fixed
to the 3D-FRONT residential domain. While representing
meaningful advances, these methods operate over fixed asset

databases and predefined spatial patterns, which limits their
ability to produce environments grounded in the specific con-
ventions of a target domain. However, none of these meth-
ods incorporate robot-operational constraints as first-class
generation objectives, including the infrastructure required
for autonomous robot operation. Environments generated by
such methods require significant manual effort before they
are viable for robot training or benchmarking scenarios.
Parallel work onscene-graph representationsunderscores
how structured spatial knowledge can improve robotic rea-
soning. Domain-Conditioned Scene Graphs [24] show that
conditioning graph nodes on domain semantics enables more
effective state-grounded task planning, while FunGraph [25]
attaches functionality labels to scene-graph nodes so that
language-prompted agents can reason about object affor-
dances. SPADE [26] builds actionable multi-domain 3D
scene graphs and uses them for scalable path planning.
TACS-Graphs [27] integrates ground-robot traversability into
room segmentation, showing that traversability-aware scene
graphs yield more consistent room boundaries and better
loop-closure detection—a property directly relevant to the
corridor and circulation constraints that D3D-GEN encodes
in its generated floorplans. These methods consume scene
graphs but do not generate the underlying 3D worlds; D3D-
GEN bridges that gap by producing both the scene graph
and the simulator-ready environment from a single natural
language prompt.
A convincing hospital ward requires knowing where nurses
keep medication trolleys, how wide corridors must be for
wheelchair access, and what a busy ward looks like at shift
change. That knowledge is domain-specific, contextual, and
not derivable from generic training data. Existing tools also
expose generation through technical parameters, which puts
them out of reach of domain experts who lack simulation
expertise but have the deepest understanding of target en-
vironments. D3D-GEN addresses both gaps by grounding
generation in autonomously retrieved domain knowledge and
accepting natural language as its primary interface.
III. METHODOLOGY
A. Domain Database
TheDomain Databaseis a structured knowledge base that
encodes domain-specific constraints, serving as the bridge
between user intent and practical reality. Rather than pro-
viding a static, hand-curated knowledge base, we frame
database construction itself as anautonomous generation
task: given only a single natural-language request, the system
synthesizes a schema-validated database of domain-specific
constraints by conducting grounded web research, extracting
quantitative rules from authoritative sources, and organizing
them into a machine-readable format. Every constraint in
the resulting database carries explicit provenance (source
URL, publisher, jurisdiction, confidence level), making the
knowledge inspectable, citable, and reproducible.1) Database Generation Pipeline:The database is produced
by a single end-to-end pipeline that accepts a user prompt
and proceeds through four sequential stages, each imple-
mented as separate LLM calls:
1)Domain & Country Inference.A structured-output
call classifies the user prompt into a canonical triple
(domain,country,country_code), wheredomain∈
{residential,office,hospital}.
2)Grounded Web Research.For each of eight pre-
defined topics, the pipeline issues a Gemini call
with Google Search grounding enabled, repeated for
kpasses (defaultk=2). Each call retrieves public
standards, codes, guidelines, handbooks, and research
documents relevant to the inferred domain and coun-
try, returning raw notes with embedded URLs. The
eight topics are: (a) room types and required support
spaces, (b) room size ranges per room type, (c) room
arrangements and adjacency rules, (d) fire safety and
evacuation, (e) employee/resident convenience and ac-
cessibility, (f) energy efficiency and HV AC/day light-
ing, (g) common objects, furniture, and equipment
per room type, and (h) robot placement, charging,
circulation, and docking. Withk=2passes, this stage
issues8×2 = 16grounded search calls.
3)Corpus Structuring.The raw notes are split into
chunks and each chunk is converted into a list of
typed documents following a corpus schema. Doc-
uments from different chunks that share the same
URL are merged: metadata fields are unified and
constraints are deduplicated by a composite signature
(name,type,value).
4)Database Synthesis.A deterministic function trans-
forms the corpus into the final database. For each
unique document it generates a stable reference slug,
maps the document type to one of seven canonical
source types, normalizes each constraint (validating
type, measurement basis, and coercing value into one
of four polymorphic forms), and attaches provenance
links back to the originating source. The output is
optionally validated against a schema.
2) Database Schema:The final database JSON conforms to
a strict schema with two top-level sections: (1)Metadata,
recording provenance (generation timestamp, model identity,
building-type scope); and (2)References, an ordered array
of reference objects, each grouping one source document
with the constraints extracted from it (Figure 4). Constraint
values are polymorphic: scalars, ranges{min,max}, alge-
braic expressions, or keyed look-up tables—enabling faithful
encoding of diverse regulatory formats. A single generated
database can therefore aggregate references from interna-
tional codes (e.g. IRC 2021 [28]), national space standards
(e.g. NDSS 2015 [29]), and domain-specific guidelines (e.g.
NHS HBN 03-01 [30]) in one unified file.
3) Robot Infrastructure:Becauserobot placement, charg-
ing, circulation, and dockingis a first-class research topic

Fig. 2:System Design: Overview of the end-to-endPrompt-to-Simulator Workflowthat transforms a natural language
prompt into an executable, physics-aware multi-agent simulation. The pipeline consists of three tightly coupled stages:
database generation(domain agent),world generation, andscenario generation. The user provides a prompt together
with lightweight structural metadata, including a domain label (Residential, Office, Hospital, . . . ), a jurisdiction/country
tag (e.g., USA, Germany), and an optional scene graph encoding room nodes and adjacency edges. A retrieval-augmented
generation (RAG) component conditions all downstream generation by constructing a domain-specific knowledge base:
grounded, robot-focused web research is synthesized and structured by an LLM into a deduplicated corpus and a
finalizedrag_data.generated.jsondatabase, enabling semantic querying over constraints, norms, and design priors.
Conditioned on this RAG context, the LLM generates araw 2D floorplanthat is then compiled into a 3D world through two
coupled mechanisms: (i)material selectionvia randomized sampling to diversify physical appearance and surface properties,
and (ii)asset placement(D3D-GEN) driven by semantic queries over anasset databaseto populate rooms with functionally
consistent objects. The resulting world specification (world.yaml) encodes walls, doors, assets, and semantically labeled
zones (Regions of Interest, ROIs) that ground both navigation and behavior anchoring. Finally, ascenario generationmodule
instantiates agents and task structure over the ROIs, producing an interactive simulator scene with bothspotlight agents
(task-relevant) andambient agents(background), yielding scalable, coherent behaviors aligned with the user intent and the
generated environment.
Document
+-- title, publisher, url
+-- doc_type : regulation | guideline | ...
+-- edition_or_year
+-- jurisdiction_context
+-- topics_covered[]
+-- room_types[]
+-- constraints[]
+-- name, constraint_type, value, unit
+-- measurement_basis
+-- applicability : {space_type, conditions[]}
+-- confidence : high | medium | low
Fig. 3: Simplified corpus document schema used in Stage 3.
(topic h), every generated database contains robot-specific
constraints alongside conventional architectural standards.
Typical automatically-retrieved constraints include: charging-
dock minimum clearance radius (e.g.≥1.2m for 180°
approach), robot staging-zone area per unit (e.g. 2.5 m2),
corridor width minimums for bidirectional robot traffic (e.g.
≥1.8m), and line-of-sight requirements between robot zones
and staff stations for safety monitoring. These constraints
are injected into generation prompts alongside dimensional
and safety rules, ensuring that produced floorplans are robot-
ready without post-hoc modification.
4) Constraint Retrieval:At floorplan generation time, the
RAG retriever selects constraints from the generated database
in two steps:
1)Building-type classification.Room-type labels from
the input scene graph are matched against keyword sets
(e.g.patientroom,operatingroom→hospital;meet-Reference
+-- ref_id : unique slug
+-- building_type : residential | office | hospital
+-- jurisdiction : geographic / regulatory context
+-- retrieved_at : ISO-8601 date
+-- sources[]
| +-- source_id, type, title, publisher, url
+-- constraints[]
+-- constraint_id, name
+-- constraint_type : minimum | maximum | range
| ratio | recommendation | rule_of_thumb
| table
+-- value : number | {min,max} | {expr} | table
+-- unit : m^2 | ft^2 | m | count | other
+-- measurement_basis, applicability
+-- confidence : high | medium | low
+-- source_ids[] : provenance links
Fig. 4: Simplified reference schema in the domain database.
ingroom,conferenceroom→office; default→residen-
tial).
2)Constraint filtering and formatting.All references
whose building type matches the classified type are
selected. Their constraints are rendered as natural-
language bullets and injected into the LLM prompt
under aReference Room Size Standardsheading.
This two-stage retrieval ensures that the generative model re-
ceives only domain-relevant constraints, keeping the prompt
concise while maximizing compliance with the standards
discovered during database generation.

B. Generation Pipeline
Our generation pipeline is multi-staged, leveraging the do-
main database in combination with the asset database for
RAG in order to generate floorplans and object placements.
Before we explain the full pipeline, we first define the asset
database:
1) Asset Database:The asset database is an automati-
cally processed dataset of 3D models (multi-format, mainly
USDZ) that is queryable via natural language.
The current database spans three domains with the following
asset counts:
Domain Office Hospital Residential Total
# 3D Assets139 71 49259
For each asset, the database computes and stores the follow-
ing annotations:
•Bounding box: axis-aligned 3D extents[x min, xmax]×
[ymin, ymax]×[z min, zmax]in meters;
•Material list: physical surface materials (e.g. leather,
steel, wood);
•Color palette: dominant colors (e.g. white, silver);
•Human–object interactions (HOI): affordance tags
(e.g. lie, sit, operate);
•Face direction: canonical front-facing axis (+x,−x,
+y,−y,xy) for consistent placement orientation;
•Semantic tags: domain and category labels (e.g. office,
hospital, bedroom, furniture);
•Free-text description: a short natural-language descrip-
tion (e.g.Gurney,Armchair).
All annotations are concatenated into a single text em-
bedding, enabling nearest-neighbor retrieval for arbitrary
natural-language asset queries (e.g. querying “adjustable
hospital bed” returns the closest matching assets).
2) Text-to-Graph:Given a natural-language prompt (e.g.
“Design an office floorplan for 60 staff”), theText-to-Graph
module generates ascene graphG= (V, E)where vertices
Vare rooms and edgesEencode adjacency and door
relationships. The module uses a system prompt that specifies
a strict JSON schema for nodes and edges, and provides
few-shot in-context examples loaded from domain-specific
supporting-example files.
Each vertex carries:
•A unique room identifier
•A semantic room-type label (e.g.Reception,Patient
Room)
•A preliminary list of 8–15 asset names per room,
enriched by anasset enrichmentpass that fills gaps us-
ing room-type keyword hints (e.g. robot rooms receive
charging docks, robot carts, maintenance benches)
•A natural-language room descriptionBeyond room labels, the scene graph also includes
robot-operation semantic annotationsattached to nodes
and edges. Concretely, the parser creates typed re-
gions such ascleaning_zone,service_zoneand
charging_buffer, then links them to corridor/door
edges that robots should traverse. For example, cleaning
robots in office domains are assigned high-coverage cleaning
zones in open work areas and low-priority zones in meeting
rooms, while service robots in hospital domains receive
service zones around nursing stations, patient-room doors,
and supply rooms. These annotations are stored as role-
tagged constraints and propagated to floorplan and placement
prompts.
3) Floorplan Generation:The scene graph is passed to the
floorplan generator, which implements a three-phase RAG
pipeline:
a) Preprocess:The generator retrieves the building type
and the RAG database from the Domain Database Pipeline.
Based on the building type, the system retrieves relevant
constraints from the knowledge base, which are formatted
as natural language and injected into the LLM prompt.
b) Inference:The structured prompt, with the RAG context
injected, is submitted to the LLM, which returns a JSON
object containing polygon coordinates for rooms and doors.
c) Post-processing:The post-processing stage parses this
response, constructs Shapely [31] geometries, and assembles
the finalWorldDescription. Wall segments are derived
by computing boundary differences between room polygons
and door openings. Each zone receives material assignments
based on room type.
4) Asset Population:After the floorplan is generated, it is
passed to a three-partpopulation pipeline. Given a room
polygon, door geometry, and semantic room label from the
floorplan stage, the module outputs a set of placed objects
with collision-free 2D poses.
a) Asset retrieval:The pipeline obtains the building type
from the floorplan generation pipeline. Based on the build-
ing type, the system maps the scene graph assets to real
models via multi-stage matching. The room-type priors are
added when requested assets are sparse. Candidate ranking
combines lexical relevance and metadata compatibility by
considering text similarity, room-type compatibility, and as-
set sizes.
b) LLM-guided placement proposal:The placement model
receives:
•room boundaries (corners and bounding box),
•door polygons and clearance instructions,
•ranked candidate assets with nominal dimensions,
•target assets count and room semantics.
Then it outputs a JSON list of proposed placements, which
are validated by hard constraints (no overlapping, compliance
with room boundaries, wall offsets). If the proposal fails, the

system performs position search to place the assets, subject
to all constraints.
c) Output canonicalization:Accepted placements are ex-
ported as obstacle entities with unique object identifiers and
domain-relative model paths. The populated room list is then
serialized into the final world artifactWorldDescription
aligning with the generated floorplan graph.
Finally, the RAG pipeline exports our world format as
aworld.yamlfile. Each zone includes: (1) geome-
try as corner positions, (2) walls with material specs,
(3) doors with width/pose, and (4) semantic name (e.g.,
patient_room_01), and (5) populated assets (e.g.,
Hospital/Patient_Bed_A). This ensures ROIs/POIs
in behavior specifications are grounded in geometrically
valid, standards-compliant regions.
IV. EVALUATIONS
We evaluate D3D-GEN on its ability to (1) reproduce plau-
sible results in common domains, and (2) extend to less-
common transfer domains. To this end, we choseindoor res-
idential(Residential)because of its ubiquity as the common
domain,commercial office space and hospital space(Office,
Hospital)as transfer domains.
We generated 450 worlds in total: 150 residential, 150
office, and 150 hospital (50 samples each for requested room
counts 5, 8, and 10). Quantitative metrics are parsed from
world.yaml, and room polygon areas are computed with
the Shoelace formula.TABLE II: Domain-agent execution profile. “MTok” denotes
million tokens (input + output) across all LLM calls.
Domain In Out Total Infer Research Struct. Synth. Total (s)
(MTok) (MTok) (MTok) (s) (s) (s) (s)
Residential 0.42 0.18 0.60 2.1 148.5 31.4 1.8 183.8
Office 0.51 0.22 0.73 2.3 162.7 38.2 2.1 205.3
Hospital 0.56 0.24 0.80 2.2 174.3 42.6 2.3 221.4
TABLE III: Generated database quality and size metrics.
Metric Residential Office Hospital
Raw notes (chars) 86,400 104,200 80,788
Chunk count 8 9 7
Corpus documents (dedup) 24 29 41
Final references 24 29 41
Total constraints 138 172 160
Avg. constraints / reference 5.8 5.9 3.9
Unique room types covered 12 16 82
Constraint-type families 4 5 4
Jurisdiction coverage (count) 3 4 9
Confidence split (H/M/L) 42/48/10% 38/51/11% 90/9/1%
DB file size (KB) 86 112 116
A. Quantitative Metrics
Table V compresses all per-instance quantitative data into 9
configuration rows (domain×requested room count).
Table IV benchmarks D3D-GEN against four baselines
spanning floorplan-only (FP: HouseDiffusion, Holodeck),
population-only (P: DiffuScene), and combined (FP+P:
ProcTHOR) generation paradigms. Each method produced
n=18worlds from the same residential prompts. We report
six quantitative metrics extracted automatically from the
generatedworld.yamlfiles and four perceptual metrics
obtained via Gemini-based evaluation. D3D-GEN achieves
Fig. 5: Qualitative comparisons. The same prompt is used for generating scene graphs, across three domains (Hospital,
Residential, Office). For theFloorplan Generationtask, the same scene graph is used for each baseline compared with
D3D-GEN. D3D-GEN excels on failure modes of baselines (wrong or no positioning of doors, coordinates not according
to actual rules, scalability with increasing number of rooms). The same output of the Floorplan Generation pipeline
WorldDescriptionobject is used for theAsset Placementtask. D3D-GEN has better object handling and grouping, as
well as better knowledge of assets rotation with respect to walls and room boundaries. Robot charging rooms and clearance
zones were designated only by D3D-GEN, marked in brown. Scenario Generation outputs Human-Robot interaction scenarios
where human presence provides robots with more context in worlds.

TABLE IV: Multi-method comparison on generated outputs.FPis floorplan generation-only methods,Pis population-only
methods,FP+Pcombines both. Quantitative metrics are extracted from automated world parsing; perceptual metrics are
Gemini-based averages across domains. For Layout-FID, lower is better; for Layout, Visual, and VQA Acc., higher is better.
MethodnQuantitative Metrics Perceptual Metrics
Objects Obj./Room Doors Area (m2) Obj. Spacing (m) Layout-FID Layout Visual VQA Acc. Scene Rating (P)
HouseDiffusion (FP) 18 - - 8.31 40.5 - 52933.82 4.48 4.60 0.313 -
Holodeck (FP) 18 - - 7.00 154.4 - 12812.768.467.17 0.353 -
ProcTHOR (FP+P) 18 3.73 0.52 8.11 938.7 6.41 85517073.63 5.50 5.78 0.333 8.09
DiffuScene (P) 18 46.60 6.33 - - 8.582531.057.78 7.22 0.389 8.33
D3D-GEN (FP+P)(Ours)1854.56 7.067.67 244.39.195377.19 8.337.61 0.400 8.63
TABLE V: Configuration-level quantitative summary from
generated worlds (n=50). Values are means per configuration.
Domain Objs Doors Walls Area (m2) Objs./Room
Residential (5-room) 30 5 30 95 6.1
Residential (8-room) 48 9 50 158 6.0
Residential (10-room) 60 11 62 222 6.0
Office (5-room) 45 6 32 287 8.9
Office (8-room) 64 9 49 347 7.9
Office (10-room) 72 10 60 365 7.2
Hospital (5-room) 35 5 30 114 7.0
Hospital (8-room) 56 8 48 231 7.0
Hospital (10-room) 69 10 61 246 6.9
the highest object count, object density, object spacing,
visual quality, VQA accuracy, and Scene rating (derived
from [32]), while remaining competitive on Layout-FID.
This demonstrates that domain-grounded RAG generation
produces worlds that are both structurally richer and per-
ceptually more realistic than existing approaches.
B. Qualitative Comparison with Baselines
Our qualitative analysis focuses on commonly observed
failure modes and how our system successfully avoids them,
yielding more consistent, higher-quality results. Figure 5
compares D3D-GEN side-by-side with HouseDiffusion [8],
ProcTHOR [4], Holodeck [12], and DiffuScene [6] on two
main tasks: Floorplan Generation and Asset Placement.
1) Floorplan:We observe three consistent floorplan-level
patterns across configurations. First, room-count fidelity is
perfect: in every case, the realized number of rooms ex-
actly matches the requested count. Second, total area scales
differently by domain: office and hospital worlds expand
more substantially than residential layouts, reflecting larger
functional rooms and greater circulation requirements. Third,
topological complexity is highest in hospitals, which exhibit
the largest numbers of doors and wall segments, consistent
with corridor-loop structures and care-workflow constraints.
Beyond these aggregate trends, qualitative inspection also
shows that layouts generated by D3D-GEN maintain clearer
adjacency logic and more coherent transitions between pri-
vate and shared spaces, which is particularly important
for social navigation tasks. We also observe that robot-
path semantics remain consistent with domain intent: office
scenes tend to produce broad cleaning loops through open
workspaces and circulation spines, while hospital scenes
prioritize service routes that connect nursing stations, patient-room thresholds, and utility rooms with fewer unnecessary
detours.
2) Asset Population:At the population level, two additional
patterns are consistent. Object density varies strongly by
domain: office environments are the most cluttered (up to
8.9objects per room in our configuration summary), whereas
residential scenes remain comparatively sparse. Material
usage also differs by domain: hospital scenes show the least
material diversity, while residential and office environments
use broader material palettes. In addition, object groupings
in D3D-GEN are typically more functionally consistent with
room semantics (e.g., workstation clusters in offices and
treatment-support groupings in hospitals), while preserving
usable free space for movement. This carries over to robot-
operational semantics: cleaning robots are assigned larger
cleanable areas (open-plan office regions, lobby loops),
whereas service robots are assigned targeted service areas
near delivery-relevant assets (beds, nurse counters, supply
shelves), reducing cross-traffic with human circulation. This
balance between semantic richness and clearance helps main-
tain both visual plausibility and simulation readiness.
V. CONCLUSION
In this paper, we introduced D3D-GEN as a novel approach
to simulable 3D World Generation. Unlike prior work, we
take a fully user-defined domain approach that extracts a
domain database from publicly accessible data. We actively
include robot-semantic annotations to support human-robot
interaction across contextually relevant use cases.
We have generated and evaluated 450 different worlds to
show the scalability of D3D-GEN. Additionally, we qualita-
tively compared our results to established baselines. These
results confirm D3D-GEN’s applicability across both com-
mon and less-explored domains.
The main limitation is the remaining reliance on an existing
asset database, which may not be readily available for every
domain. Infinigen-Articulated [33] procedurally generates
articulated assets for theindoor residentialdomain. In our
future work, we will integrate a similar approach and extend
it to arbitrary domains by analyzing asset data with our
domain agent. We already have the infrastructure to export
just-in-time generated assets with the final world output.
Currently, the only direct interaction with the domain

database is its initial generation from the prompt. The user
might not be satisfied with the captured intent and the re-
sulting database. Another future extension is the modification
of existing databases by prompting the domain agent. This
would allow the user to interactively generalize or specialize
the knowledge graph until the domain matches their intent.
VI. ACKNOWLEDGEMENT
This paper is supported in part by the Ministry of Education
(MOE) Academic Research Fund (AcRF) Tier 2 grant (Grant
ID: T2EP20124-0055)
REFERENCES
[1] S. Höfer, K. Bekris, A. Handa, J. C. Gamboa, M. Mozifian, F. Golemo,
C. Atkeson, D. Fox, K. Goldberg, J. Leonard, C. Karen Liu, J. Peters,
S. Song, P. Welinder, and M. White, “Sim2real in robotics and automa-
tion: Applications and challenges,”IEEE Transactions on Automation
Science and Engineering, vol. 18, no. 2, pp. 398–400, 2021.
[2] L. H. K. Wong, X. Kang, K. Bai, and J. Zhang, “A survey
of robotic navigation and manipulation with physics simulators
in the era of embodied ai,” 2025. [Online]. Available: https:
//arxiv.org/abs/2505.01458
[3] H. Fu, B. Cai, L. Gao, L. Zhang, J. W. C. Li, Z. Xun,
C. Sun, R. Jia, B. Zhao, and H. Zhang, “3d-front: 3d furnished
rooms with layouts and semantics,” 2021. [Online]. Available:
https://arxiv.org/abs/2011.09127
[4] M. Deitke, E. VanderBilt, A. Herrasti, L. Weihs, J. Salvador, K. Ehsani,
W. Han, E. Kolve, A. Farhadi, A. Kembhavi, and R. Mottaghi,
“Procthor: Large-scale embodied ai using procedural generation,”
2022. [Online]. Available: https://arxiv.org/abs/2206.06994
[5] A. Raistrick, L. Lipson, Z. Ma, L. Mei, M. Wang, Y . Zuo, K. Kayan,
H. Wen, B. Han, Y . Wang, A. Newell, H. Law, A. Goyal, K. Yang, and
J. Deng, “Infinite photorealistic worlds using procedural generation,”
2023. [Online]. Available: https://arxiv.org/abs/2306.09310
[6] J. Tang, Y . Nie, L. Markhasin, A. Dai, J. Thies, and M. Nießner,
“Diffuscene: Denoising diffusion models for generative indoor scene
synthesis,” 2024. [Online]. Available: https://arxiv.org/abs/2303.14207
[7] L. Song, L. Cao, H. Xu, K. Kang, F. Tang, J. Yuan, and
Y . Zhao, “Roomdreamer: Text-driven 3d indoor scene synthesis
with coherent geometry and texture,” 2023. [Online]. Available:
https://arxiv.org/abs/2305.11337
[8] M. A. Shabani, S. Hosseini, and Y . Furukawa, “Housediffusion: Vector
floorplan generation via a diffusion model with discrete and continuous
denoising,” 2022. [Online]. Available: https://arxiv.org/abs/2211.13287
[9] W. Feng, W. Zhu, T.-J. Fu, V . Jampani, A. Akula, X. He, S. Basu,
X. E. Wang, and W. Y . Wang, “Layoutgpt: Compositional visual
planning and generation with large language models,” 2023. [Online].
Available: https://arxiv.org/abs/2305.15393
[10] Y . Yang, B. Jia, P. Zhi, and S. Huang, “Physcene: Physically
interactable 3d scene synthesis for embodied ai,” 2024. [Online].
Available: https://arxiv.org/abs/2404.09465
[11] H. Xia, X. Li, Z. Li, Q. Ma, J. Xu, M.-Y . Liu, Y . Cui, T.-Y . Lin,
W.-C. Ma, S. Wang, S. Song, and F. Wei, “Sage: Scalable agentic
3d scene generation for embodied ai,” 2026. [Online]. Available:
https://arxiv.org/abs/2602.10116
[12] Y . Yang, F.-Y . Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han,
J. Wu, N. Haber, R. Krishna, L. Liu, C. Callison-Burch, M. Yatskar,
A. Kembhavi, and C. Clark, “Holodeck: Language guided generation
of 3d embodied ai environments,” 2024. [Online]. Available:
https://arxiv.org/abs/2312.09067
[13] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets,
A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X.
Chang, M. Savva, Y . Zhao, and D. Batra, “Habitat-matterport 3d
dataset (hm3d): 1000 large-scale 3d environments for embodied ai,”
2021. [Online]. Available: https://arxiv.org/abs/2109.08238
[14] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva,
S. Song, A. Zeng, and Y . Zhang, “Matterport3d: Learning from
rgb-d data in indoor environments,” 2017. [Online]. Available:
https://arxiv.org/abs/1709.06158[15] M. Savva, A. Kadian, O. Maksymets, Y . Zhao, E. Wijmans, B. Jain,
J. Straub, J. Liu, V . Koltun, J. Malik, D. Parikh, and D. Batra, “Habitat:
A platform for embodied AI research,”CoRR, vol. abs/1904.01201,
2019. [Online]. Available: http://arxiv.org/abs/1904.01201
[16] E. Kolve, R. Mottaghi, W. Han, E. VanderBilt, L. Weihs, A. Herrasti,
M. Deitke, K. Ehsani, D. Gordon, Y . Zhu, A. Kembhavi, A. Gupta,
and A. Farhadi, “Ai2-thor: An interactive 3d environment for visual
ai,” 2022. [Online]. Available: https://arxiv.org/abs/1712.05474
[17] J. Zheng, J. Zhang, J. Li, R. Tang, S. Gao, and Z. Zhou, “Structured3d:
A large photo-realistic dataset for structured 3d modeling,” 2020.
[Online]. Available: https://arxiv.org/abs/1908.00222
[18] C. Lin and Y . Mu, “Instructscene: Instruction-driven 3d indoor scene
synthesis with semantic graph prior,” 2024. [Online]. Available:
https://arxiv.org/abs/2402.04717
[19] B. M. Öcal, M. Tatarchenko, S. Karaoglu, and T. Gevers,
“Sceneteller: Language-to-3d scene generation,” 2024. [Online].
Available: https://arxiv.org/abs/2407.20727
[20] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi,
A. Mandlekar, and Y . Zhu, “Robocasa: Large-scale simulation of
everyday tasks for generalist robots,” 2024. [Online]. Available:
https://arxiv.org/abs/2406.02523
[21] C. Wang, B. Du, J. Xu, P. Li, D. Guo, and H. Liu, “Demonstrating
humanthor: A simulation platform and benchmark for human-robot
collaboration in a shared workspace,” 2024. [Online]. Available:
https://arxiv.org/abs/2406.06498
[22] Y . Yang, J. Lu, Z. Zhao, Z. Luo, W. Dong, V . Sanchez, and F. Zheng,
“Llplace: Embodied 3d indoor layout synthesis framework with large
language model,” inIEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), 2025.
[23] M. Chen, R. Yang, Q. Hu, K. Xue, S. Zhou, and Y . Guo, “Graph2scene:
Versatile 3d indoor scene generation with interaction-aware scene
graph,” inIEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), 2025.
[24] J. Herzog, J. Liu, and Y . Wang, “Domain-conditioned scene graphs for
state-grounded task planning,” inIEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), 2025.
[25] D. Rotondi, F. Scaparro, H. Blum, and K. O. Arras, “Fungraph:
Functionality aware 3d scene graphs for language-prompted scene
interaction,” inIEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), 2025.
[26] V . Kottayam Viswanathan, A. Patel, M. A. Valdes Saucedo, S. Satpute,
C. Kanellakis, and G. Nikolakopoulos, “Spade: Towards scalable path
planning architecture on actionable multi-domain 3d scene graphs,” in
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS), 2025.
[27] J. Kim, M. Oh, and H. Myung, “Tacs-graphs: Traversability-aware
consistent scene graphs for ground robot localization and mapping,” in
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS), 2025.
[28] I. C. C. (ICC), “2021 international residential code (irc),” 2021.
[Online]. Available: https://codes.iccsafe.org/content/IRC2021V3.0
[29] C. Ministry of Housing and L. Government, “Technical housing
standards – nationally described space standard,” Mar. 2015.
[Online]. Available: https://www.gov.uk/government/publications/
technical-housing-standards-nationally-described-space-standard
[30] N. England, “Nhs england » health building note
03-01: Adult acute mental health units,” Mar.
2013. [Online]. Available: https://www.england.nhs.uk/publication/
adult-mental-health-units-planning-and-design-hbn-03-01/
[31] S. Gillies, C. van der Wel, J. Van den Bossche, M. W. Taves,
J. Arnott, B. C. Ward,et al., “Shapely (2.1.2),” 2025. [Online].
Available: https://doi.org/10.5281/zenodo.17193310
[32] H. I. I. Tam, H. I. D. Pun, A. T. Wang, A. X. Chang, and
M. Savva, “SceneEval: Evaluating semantic coherence in text-
conditioned 3D indoor scene synthesis,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.14756
[33] A. Joshi, B. Han, J. Nugent, M. G. Saez-Diez, Y . Zuo, J. Liu,
H. Wen, S. Alexandropoulos, K. Kayan, A. Calveri, T. Sun,
G. Liu, Y . Shao, A. Raistrick, and J. Deng, “Procedural generation
of articulated simulation-ready assets,” 2025. [Online]. Available:
https://arxiv.org/abs/2505.10755