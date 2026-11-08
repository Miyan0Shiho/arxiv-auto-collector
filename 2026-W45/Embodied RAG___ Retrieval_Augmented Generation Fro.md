# Embodied RAG++: Retrieval-Augmented Generation From What Exists to What Works in Embodied Scenes

**Authors**: Fan Yang, Tatsuya Kikuzuki, Jun Liang, Jun Liu, Wenkai Li, Ryosuke Kawamura, Ananya Hazarika, Prajit KrisshnaKumar, Shaunak Mehta, Yuushi Toyoda

**Published**: 2026-11-05 06:26:12

**PDF URL**: [https://arxiv.org/pdf/2611.07517v1](https://arxiv.org/pdf/2611.07517v1)

## Abstract
To serve a request in a human environment, a robot system must find an object that satisfies a functional requirement and a robot that can feasibly reach it. Scene graphs record which objects exist and where, but they often omit capabilities documented only in product manuals. Retrieval-augmented generation methods can read such manuals, yet they rarely reason jointly over object capabilities, physical instances, robot states, and dynamic connectivity. We introduce E2W-Bench, the Exists-to-Works Benchmark, which links dynamic hybrid scene graphs with real product manuals to evaluate capability-grounded robot--object assignment under availability, mobility, vertical-reach, and connector constraints. We also present Embodied RAG++, a spatially grounded retrieval framework that scopes retrieval to the manuals linked to candidate objects, caches extracted capabilities per product model, and resolves robot--object feasibility through deterministic spatial reasoning. On E2W-Bench, Embodied RAG++ reaches a pair accuracy of 0.90 with Qwen2.5-32B and 1.00 with Llama3.3-70B, compared with at most 0.30 and 0.55 for the baselines, while requiring the shortest preprocessing time among the retrieval-based methods and tying for the lowest total token count.

## Full Text


<!-- PDF content starts -->

Embodied RAG++: Retrieval-Augmented Generation From What Exists to
What Works in Embodied Scenes
Fan Yang1, Tatsuya Kikuzuki1, Jun Liang1, Jun Liu2, Wenkai Li2,
Ryosuke Kawamura1, Ananya Hazarika1, Prajit KrisshnaKumar1, Shaunak Mehta1,
and Yuushi Toyoda1
Abstract— To serve a request in a human environment, a
robot system must find an object that satisfies a functional
requirement and a robot that can feasibly reach it. Scene graphs
record which objects exist and where, but they often omit
capabilities documented only in product manuals. Retrieval-
augmented generation methods can read such manuals, yet they
rarely reason jointly over object capabilities, physical instances,
robot states, and dynamic connectivity. We introduce E2W-
Bench, the Exists-to-Works Benchmark, which links dynamic
hybrid scene graphs with real product manuals to evaluate
capability-grounded robot–object assignment under availabil-
ity, mobility, vertical-reach, and connector constraints. We
also present Embodied RAG++, a spatially grounded retrieval
framework that scopes retrieval to the manuals linked to
candidate objects, caches extracted capabilities per product
model, and resolves robot–object feasibility through deter-
ministic spatial reasoning. On E2W-Bench, Embodied RAG++
reaches a pair accuracy of 0.90 with Qwen2.5-32B and 1.00
with Llama3.3-70B, compared with at most 0.30 and 0.55 for
the baselines, while requiring the shortest preprocessing time
among the retrieval-based methods and tying for the lowest
total token count.
I. INTRODUCTION
Robots operating in human environments need more than
knowledge of which objects exist and where they are located.
To serve a request, a system has to find which physical object
can satisfy the user’s functional requirement and which avail-
able robot can feasibly reach it. Consider a request to heat
a 27 L food container to 240°C. A scene graph may contain
several microwave ovens, but the required cavity capacity and
convection temperature are specified only in their product
manuals. Even after identifying a suitable appliance, the
system has to check whether an available robot can reach it
under the current mobility and connectivity conditions. If an
elevator becomes unavailable, for example, a wheeled robot
may be unable to access another floor, whereas a legged
robot may use the stairs. Resolving such a request therefore
requires combining capabilities documented in manuals with
the current state of the objects, robots, and connectors in the
scene.
Existing work addresses parts of this problem but not
their combination. 3D scene graphs [1], [2], open-vocabulary
maps [3], and hierarchical planners [4] model spatial layouts
and navigation paths, while benchmarks such as ScanRe-
1Fan Yang, Tatsuya Kikuzuki, Jun Liang, Ryosuke Kawamura, Ananya
Hazarika, Prajit KrisshnaKumar, Shaunak Mehta, and Yuushi Toyoda are
with Fujitsu.fan.yang@fujitsu.com
2Jun Liu and Wenkai Li are with Carnegie Mellon University.
Fig. 1: Conventional scene graph versus Embodied RAG++. Left: a
scene graph, represented by 3DSceneGraph [1], records which ap-
pliances exist and their models, but cannot tell whether a microwave
can heat a 27 L container to 240°C, because that fact is stated
only in its product manual. Right: Embodied RAG++ retrieves the
capability from the linked manual, checks the current connector
states and robot mobility, and selects the object–robot pair that
satisfies every constraint.
fer [5] and SQA3D [6] evaluate visual language ground-
ing. These representations capture what exists and where,
but not the non-visual functional attributes stored in ex-
ternal documentation. Graph-structured retrieval-augmented
generation (RAG) [7] frameworks such as GraphRAG [8],
RAPTOR [9], and HippoRAG [10], for their part, retrieve
document knowledge but return textual answers that are
not tied to physical instances and ignore reachability, robot
availability, and changing scene state. EmbodiedRAG [11]
introduces spatial retrieval over observational memory, but
its knowledge is limited to what the agent has perceived,
not to external manuals. In addition, processing multi-page
manuals anew for every query and every physical instance
repeats the same work whenever several instances share a
product model.
We introduceE2W-Bench(Exists-to-Works Benchmark),
a benchmark for capability-grounded robot–object assign-
ment over dynamic hybrid scene graphs. E2W-Bench con-
nects physical object instances with product manuals and
unifies robots, spatial connectors, and dynamic environmen-
tal states within a single graph. Each query has a machine-
verified unique ground-truth pair, so a single exact-match
check evaluates document reasoning and spatial allocation
together.1
arXiv:2611.07517v1  [cs.RO]  5 Nov 2026

We also proposeEmbodied RAG++, a spatially grounded
retrieval framework that couples document-grounded ca-
pability inference with deterministic feasibility verification
(Fig. 1). Embodied RAG++ uses the scene graph’s object-to-
manual links to restrict retrieval to the manuals of candidate
objects. It resolves reachability, mobility, and connectiv-
ity constraints with deterministic spatial reasoning, which
requires no large language model (LLM) calls. To avoid
reprocessing the same manuals, Embodied RAG++ stores
extracted facts inCapability Cards, memory units keyed by
product model that grow incrementally as queries request
new fields. Every physical instance of a model and every
later query that needs the same field reuses the card.
Our contributions are as follows:
•E2W-Bench, which links dynamic hybrid scene graphs to
real product manuals and gives every query a machine-
verified unique ground-truth pair (§III).
•Embodied RAG++, which combines manual-scoped,
aspect-split retrieval, model-level Capability Cards, avail-
ability filtering, and deterministic spatial reasoning (§IV).
•An evaluation with three LLM backbones and four base-
lines, in which Embodied RAG++ reaches a pair accuracy
of 0.90 with Qwen2.5-32B and 1.00 with Llama3.3-70B,
compared with at most 0.30 and 0.55 for the baselines,
with the shortest preprocessing time among the retrieval-
based methods and the lowest total token count, tied with
EmbodiedRAG (§V).
II. RELATEDWORK
A. Scene Graphs and Embodied Scene Representations
Scene graphs represent environments as objects, rooms,
floors, and their spatial and semantic relations [1]. Recent
systems construct these representations online through hi-
erarchical metric-semantic mapping [2], incorporate open-
vocabulary concepts from vision-language models [3], and
support language-conditioned planning over large environ-
ments [4]. Related benchmarks evaluate grounding free-
form language to objects in 3D scenes [5] and answering
questions from an agent’s situated viewpoint [6]. These
methods capture the spatial and semantic structure of a scene,
but they do not recover functional capabilities stated only in
external documents such as product manuals, nor do they
determine whether an available robot can reach and use a
qualifying object as the scene changes.
B. Retrieval-Augmented Generation over Structured Cor-
pora
Retrieval-augmented generation (RAG) grounds language-
model outputs in external documents [7]. Structured variants
organize document collections to improve retrieval and multi-
hop reasoning. GraphRAG constructs an entity-relation graph
and summarizes its communities [8]; RAPTOR recursively
clusters and summarizes passages into a retrieval tree [9]; and
HippoRAG applies personalized PageRank over a knowledge
graph to integrate information across passages [10]. These
methods improve document-grounded question answering,
but their preprocessing generally spans the corpus and theiroutputs remain passages or textual responses. In our setting,
retrieved evidence must instead be grounded to a physical
object instance and checked against the current state of
robots and connectors to produce an executable robot–object
assignment.
C. Embodied Retrieval-Augmented Generation
EmbodiedRAG constructs a hierarchical, non-parametric
memory from an agent’s observations and retrieves multi-
resolution semantic and spatial information for navigation
and language generation [11]. OpenSG-RAG [12] encodes
open-world 3D scene graphs in a vector database to sup-
port retrieval-augmented scene question answering, visual
grounding, instance retrieval, and task planning. RAG-
3DSG [13] instead applies object-level retrieval during scene-
graph construction, using semantically reliable objects as
contextual anchors to refine uncertain object predictions.
These methods retrieve from observed scene information
or use retrieval to improve the scene representation itself.
Embodied RAG++ instead treats the scene graph as an index
into external knowledge: each physical object instance links
to a product manual that states capabilities a camera often
cannot observe. It reuses capabilities extracted per product
model and checks the selected object against the current
robot and connector states to produce an executable robot–
object assignment.
III. E2W-BENCH
A. Task Formulation
We formulate E2W-Bench as a robot–object pair selection
problem over adynamic hybrid scene graphwith four
node/edge types:appliance nodes(category, model, a link
to that model’s real product manual, availability, position);
robot nodes(mobility profile, availability, position); aspatial
hierarchyof scene→floor→room→object/robot; and
connector edges(stairs, elevators, doors) carrying a traversal
cost, an accessibility list of robot types, and a status that can
change between queries. Fig. 2 shows one such graph in full.
Its node and edge states are not fixed at construction time;
they track the physical scene and change as a robot moves
or is dispatched, an appliance is switched on or taken out of
service, or a connector’s status changes. These state updates
make the graphdynamicrather than a static record of object
positions and labels.
Given the graph’s current state and a natural-language
query, a system returns a robot identifier, a target-object
identifier, and both of their positions. A returned pair is cor-
rect only if four independent conditions hold: (i)capability:
the target’s real manual establishes that it meets the query’s
functional requirement; (ii)availability: that instance is not
in_useorout_of_order; (iii)mobility: a candidate
robot can reach the target’s floor given the current connector
states and its own accessibility profile; and (iv)vertical
reach: that robot’s body or arm can physically reach the
object’s mounted height.2

Fig. 2: Overview of one validated E2W-Bench instance and the Embodied RAG++ pipeline, including the dynamic hybrid scene graph,
manual-linked objects, Capability Cards, scoped retrieval, and spatial reasoning.
B. Archetype-Based Generation
We generate benchmark scenes in the MuJoCo simulator
using three robot platforms: a quadruped, a humanoid, and a
wheeled mobile manipulator equipped with an AgileX Piper
arm. The scenes include appliance 3D models reconstructed
from real objects and their corresponding product manuals.
Fig. 3 summarizes the 20 released scenes and the underlying
manual corpus. Each scene is a verified snapshot of the
dynamic graph defined in §III-A: it fixes robot positions,
appliance availability, and connector states, the same fields
a deployed system updates online, rather than replaying
continuous physical dynamics.
Each scene instantiates one of 10 hand-designedquery
archetypes, with two archetypes for each of five appliance
categories: microwave, rice cooker, washing machine, water
dispenser, and bread maker. Each archetype has two scene
variations, yielding 20 scenes in total. Athresholdarchetype
requires the target to satisfy a compound predicate, such as
cavity capacity≥27L and bake temperature≥240°C,
while every listed distractor fails at least one condition.
Anextremumarchetype requires all candidates to satisfy a
baseline predicate, with the target minimizing or maximizing
a specified field. Before scene generation, we manually
extract the relevant capability values from the 13 product
manuals covering these five categories (Fig. 3b) and verify
each predicate against the resulting reference table, so every
scene agrees with its source manuals. The reference table
is used only to construct and validate the ground truth;
at inference time, Embodied RAG++ derives its Capability
Cards directly from the manuals (§IV-B).
Each scene contains the qualifying appliance, all distrac-tors specified by the archetype, and several unrelated manual-
linked appliances and furniture items. These objects are dis-
tributed across one to four floors connected by parallel eleva-
tors and stairs. Each scene also includes two to seven robots
of three types, using their published mobility and vertical-
reach envelopes. Two independentreachability trapsmake
incorrect answers appear plausible when either constraint is
ignored. Themobility trapplaces an otherwise closer, stairs-
incapable wheeled robot behind a connector whose elevator
is out of service. Theheight trapplaces roughly half of the
normally counter-height targets, including microwaves, rice
cookers, and bread makers, on shelves reachable only within
a humanoid’s extended vertical envelope, while an otherwise
suitable wheeled or quadruped robot is positioned nearby.
Because the two traps are independent, a scene may contain
a mobility trap, a height trap, both, or neither.
C. Evaluation Protocol
E2W-Bench evaluates a system along five axes that dis-
tinguishgetting the right answer,getting it for the right
reasons, andgetting it efficiently.Functional correctness
is exact-match accuracy on the returned(robot_id,
target_object_id)pair.Robot–object feasibilityre-
quires the selected pair to be jointly reachable under the
current connector and mobility states, andposition-level
groundingrequires the returned coordinates to match the
recorded positions of the pair.Evidence groundingchecks
whether the qualifying capability is supported by the relevant
manual span or Capability Card field.Retrieval efficiency
measures how much manual text is read and how often
lookups reuse Capability Cards instead of triggering new3

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20051015
Scene indexInstance countManual-attached electronics Robots Other objects (distractors)(a) Electronics, robot, and other-object instances per scene (E2W-0001 – E2W-0020)
(b) Unique Asset library category
Microwave (2)
Rice Cooker (2)
Washing Machine (3)
Water Disp. (2)
Bread Maker (2)
Coffee Machine (2)Sofa
Cleaning Cart Wall MirrorWardrobePrinterExercise BikeTrash BinReception DeskFloor LampFire ExtinguisherTreadmillWhiteboardVending MachineDining TableFiling CabinetOffice ChairServer RackBulletin BoardBookshelfCoffee TablePotted Plant
MicrowaveRice Cooker
Washing MachineWater Disp.Bread MakerCoffee Machine0100200300
169
50118
2996308Pages(c) Manual pages by category, full corpus
Fig. 3: E2W-Bench dataset composition, computed directly from the 20 dynamic scenes, the appliance instance library, and the manual
corpus. (a) Manual-attached electronics, robot, and other-object (distractor) instance counts per scene. (b) The underlying instance category.
(c) Number of pages in the manual corpus PDF files, totaling 770 pages.
extraction. Because Capability Cards cache facts rather than
final decisions (§IV-B), retrieval efficiency is independent
of correctness and is evaluated separately through cache
behavior (§V-C).
Functional correctness, feasibility, and position-level
grounding can all be evaluated from the returned pair. The
ground truth of each sample is a unique pair verified by
brute-force search, and every distractor and reachability trap
described in §III-B, including unmet capability predicates,
unavailable instances, mobility traps, and height traps, leads
to a specific incorrect pair. A system can return the correct
pair only if all four conditions in §III-A hold, and the
positions of a correct pair are fixed by the graph. Exact-
match pair accuracy in §V is thus a joint measure of these
three axes.
Pair correctness does not certify evidence grounding. A
system may return the correct pair while citing an incorrect
manual passage, relying on an unsupported field, or providing
no evidence: the traps detect reasoning failures that change
the final answer, not unsupported reasoning that happens to
reach it. Capability Cards retain the source manual span
for every stored field (§IV-B), so evidence grounding can
be audited, but this paper does not report a quantitative
grounding score.
Each E2W-Bench scene is far denser than the scenes of
existing benchmarks (Table I). Its serialized spatial context
reaches approximately 500K tokens, compared with fewer
than 10K for 3DSceneGraph, ScanQA, and SQA3D, and it
is paired with 770 pages of real product manuals. Each scene
also contains two to seven robots and dynamic connector
and appliance-availability states, so every query must be
answered against a large, interdependent context.TABLE I: Comparison with representative scene-based benchmarks.
Benchmark # Scenes Scene State Robots per Scene Scene Tokens
3DSceneGraph [1] 572 Static 1<10K
ScanQA [14]800Static 1<10K
SQA3D [6] 650 Static 1<10K
E2W-Bench (ours)20 Dynamic2–7≈500K
IV. EMBODIEDRAG++ METHOD
Embodied RAG++ answers a query in two coupled stages
instead of running retrieval and spatial reasoning as indepen-
dent subsystems. The first stage determines which candidate
objects satisfy the functional requirement: it retrieves only
from the manuals linked to the candidates, reuses capability
fields already extracted for the same product model, and
judges each model against the requirement. The second stage
passes the qualifying object to a deterministic spatial layer
that selects a feasible robot. We describe each stage in turn;
Algorithm 1 gives the full procedure.
A. Problem Setup
Let the scene graph provide, for every appliance instance
o, a categorycat(o), a model identifiermodel(o), a link
to a manual documentmanual(o), an availability state, and
a position; and for every robotr, an availability state, a
mobility profile, and a position. Given a queryq, Embodied
RAG++ must return a pair(r⋆, o⋆)such thato⋆satisfiesq’s
functional requirement on manual evidence,o⋆is available,
andr⋆is available and feasible foro⋆under the current
connector graph.
B. Capability Cards
Product manuals are long, multi-topic documents, and re-
reading them for every query does not scale with the number4

of instances or queries. Embodied RAG++ therefore treats
each manual as the authoritative source and maintains a
Capability Card: a per-model cache containing only previ-
ously requested capability fields and their supporting manual
spans. Cards are keyed bymodel identifier, allowing all
instances of the same model to share evidence. Nothing is
extracted in advance. A card is created when a model is first
queried and stores only the fields required by that query,
avoiding an offline corpus pass that would need to anticipate
all capabilities future queries might request. If a required
field is cached, it is reused without retrieval or extraction.
Otherwise, arefinestep retrieves the missing aspect from the
linked manual (§IV-C), extracts it, and adds it to the existing
card. Card coverage grows incrementally, and a card is
never rebuilt. Because cards store capability fields rather than
query-specific verdicts, cached and newly extracted fields
are judged identically against the current query, so caching
reduces retrieval and generation cost without changing the
judgment. When extraction is required, each manual excerpt
is processed separately to reduce the risk of mixing values
from different operating modes.
C. Scoped, Aspect-Split Retrieval
A global top-ksearch over the whole manual corpus
ranks passages only by their similarity to the query wording.
The highest-ranked passage may therefore come from a
short manual that happens to use similar phrasing rather
than from the long, differently worded manual that actually
specifies the candidate. Embodied RAG++ avoids this by
using information the scene graph already provides: the
manualmanual(o)linked to each candidate objecto. The
query is embedded once, a wide candidate pool is retrieved
from a single index over the entire corpus, and the pool is
filtered to passages whose source document ismanual(o).
One corpus-wide index therefore suffices, and every retained
passage comes from the candidate’s own manual. Compound
requirements are further split into per-aspect queries. For
example, “holds at least 10 kg” and “supports a 24-hour delay
start” are retrieved separately, because in a single combined
query the wording of one aspect can dominate the embedding
and push the other aspect’s supporting passage out of the
retrieved pool. Together, scoping and aspect splitting ensure
that every generation call in Embodied RAG++ operates on
a few short passages or on a small dictionary of previously
extracted fields, never on a full manual, which keeps each
call short and traceable to its source.
This design also distinguishes Embodied RAG++ from
document-first graph construction such as GraphRAG [8].
GraphRAG calls an LLM to extract entities and relations
from every chunk of every document before it can answer
any query, so its one-time cost grows with the size of the
corpus. Embodied RAG++ builds no corpus-wide index of
extracted facts; it extracts a field only when a query first
requests it, so its generation cost grows with the number of
distinct capability fields queried rather than with the number
of manuals or chunks in the corpus.D. Graph-Based Selective-Recomputation Retrieval Back-
bone
Every scoped search in §IV-C runs on a pruned proximity
graph over the corpus rather than on a stored embedding
table. Each chunk is a node of a hierarchical navigable small-
world graph [15] built with a graph-index backend [16], and
high-degree-preserving pruning keeps only the edges needed
for long-range navigation. Only the graph topology is stored;
the chunk embeddings used to build it are discarded [17].
At query time, best-first beam search with beam widthef
traverses the graph, and only the embeddings of the visited
chunks are recomputed with a local dense encoder [18]. The
storage cost of the corpus-wide index is therefore that of a
graph rather than an embedding matrix.
Embodied RAG++ keeps one such graph for the entire
manual corpus rather than one per model or per manual,
so adding a new appliance model to a deployment does not
require building a new index. Scoping is applied after the
graph search rather than by restricting it: each query uses
a wider beam and a larger candidate count than plain top-k
search, and the retrieved pool is filtered to passages from
the candidate’smanual(o)before the top-kare kept. Each
aspect-split query runs its own search-and-filter pass over the
same graph.
The backbone is invoked only on a Capability Card miss.
When the required field is already on the model’s card,
JUDGE(Algorithm 1) reads it directly, with no query embed-
ding, graph search, or embedding recomputation. Retrieval
for a given model and capability field therefore runs at most
once across a deployment’s query history, regardless of how
many instances of that model exist or how many later queries
request the same field.
E. Spatial Reasoning
Once a qualifying objecto⋆has been identified from Capa-
bility Card evidence, feasibility is resolved by deterministic
operators over fields the scene graph already stores, without
any LLM call. FEASIBLEROBOTSapplies two checks to ev-
ery candidate robot. REACHABLEruns a breadth-first search
over the floor–connector graph and traverses an edge only if
the connector is operational or clear and the robot’s type ap-
pears in the connector’s accessibility list; an elevator outage
or a robot without stair-climbing capability therefore removes
the corresponding path. HEIGHTREACHABLEcompares the
target’s recorded mounting height with a fixed vertical-reach
envelope for each robot type, derived from the platform’s
published arm and body dimensions, and excludes robots
that can navigate to the target but cannot physically reach
it. FEASIBLEROBOTSkeeps the available robots that pass
both checks, and PICKNEARESTbreaks any remaining tie
by shortest-path navigation distance. Because this layer only
reads the graph, it cannot return a robot identifier absent from
the graph or ignore a connector’s current status. REACH-
ABLE, HEIGHTREACHABLE, and PICKNEARESTapply the
same reachability, height, availability, and tie-breaking rules
used to construct and validate the benchmark ground truth
(§III-B).5

Algorithm 1:Embodied RAG++ Pipeline
Input:Dynamic hybrid scene graphG; queryq; Capability Card
storeCards
Output:Feasible pair(r⋆, o⋆)with positions
1(cat,mode, ϕ)←CLASSIFY(q);
2O ← {o∈G: cat(o) = cat};
3M ← {model(o) :o∈ O };
4foreachm∈ Mdo
5o m←anyo∈ Owithmodel(o) =m;
6c m←Cards.get(m,∅);
7F miss←FIELDS(ϕ)\COVERAGE(c m);
8ifF miss̸=∅then
9P←SCOPEDRETRIEVE(F miss,manual(o m));
10∆c m←EXTRACT(P, F miss);
11c m←MERGE(c m,∆cm);Cards[m]←c m;
12end
13v m←JUDGE(c m, ϕ);
14end
15O⋆← {o∈ O:v model(o) =PASSand available(o)};
16ifmode =EXTREMUMand|O⋆|>1then
17O⋆← {RESOLVEEXTREMUM(O⋆, ϕ)};
18end
19o⋆←the unique element ofO⋆;
20R⋆←FEASIBLEROBOTS(G, o⋆);
21r⋆←PICKNEAREST(R⋆, o⋆);
22return(r⋆, o⋆)with their recorded positions
V. EXPERIMENTS
We evaluate Embodied RAG++ against four baselines on
all 20 released E2W-Bench scenes, one query per scene, with
three LLM backbones. Table II reports exact-match accuracy
on the returned(robot_id, target_object_id)
pair, which jointly measures functional correctness, robot–
object feasibility, and position-level grounding, but not evi-
dence grounding or retrieval efficiency (§III-C). We further
analyze the hit-rate behavior of the Capability Card cache
(§V-C) and ablate each component of Embodied RAG++
(§V-D).
A. Baselines
Every baseline except Naive LLM shares the determinis-
tic Spatial Reasoning (SR) module (§IV-E): each baseline
identifies only the qualifying objecto⋆, and the module
selects the robot for it, exactly as in Embodied RAG++.
Spatial reasoning, which applies the same rules that gen-
erate the benchmark ground truth, is therefore held constant
across Table II, and the comparison isolates the capability-
retrieval architecture. Every method runs on the same LLM
backbones:qwen2.5:32b[19] andllama3.3:70b[20]
served locally via Ollama, andgemini-3.5-flash[21].
Naive LLM.One LLM call per query over the complete scene
graph and every candidate manual’s full text, with no retrieval,
chunking, or ranking; the same call also proposes the robot, so
feasibility is resolved by the LLM itself. This baseline tests whether
full-context prompting solves the task unaided, and at what token
cost.
Naive LLM + SR.The same call, restricted to proposing onlyo⋆;
the SR module selects the robot. Comparing it with Naive LLM
isolates the contribution of the SR module.
GraphRAG + SR.[8] Uses GraphRAG’s default entity/relationship
extraction pipeline over the same inputs (sanitized scene graph,
height constraints, full manual text) to identifyo⋆; one-time con-
struction cost is reported separately from per-query cost.
LightRAG + SR.[22] Builds its own entity–relation graph from
the manual corpus via LLM extraction to identifyo⋆; each sample’sscene graph is inserted as a transient document before querying and
removed after, so scene state never leaks across scenes.
EmbodiedRAG + SR.[11] EmbodiedRAG has no native document
support, so we adapt only its graph-retrieval component: the scene
graph becomes its hierarchical semantic-forest form for scene-node
retrieval, paired with a separate dense document retriever for manual
passages, to identifyo⋆. This baseline tests whether independent
spatial and document retrieval suffices for joint reasoning.
B. Results
The open-weight backbones separate the methods most clearly
(Table II). With Qwen2.5-32B, Embodied RAG++ reaches 0.90
pair accuracy, 60 percentage points (pp) above the strongest base-
lines, GraphRAG and EmbodiedRAG (0.30). With Llama3.3-70B,
it reaches 1.00, 45 pp above GraphRAG (0.55), and it is the
only evaluated system with perfect pair accuracy on an open-
weight backbone. With Gemini-3.5-Flash, most systems approach
the ceiling: Embodied RAG++ reaches 1.00, three + SR baselines
reach 0.95, LightRAG + SR reaches 0.85, and Naive LLM without
SR reaches 0.50.
Fig. 4 compares efficiency. GraphRAG and LightRAG require
3–12 h of index construction; GraphRAG then answers each query
in 6–12 s, whereas LightRAG is the slowest at 46–70 s. Embod-
iedRAG preprocesses in 43–46 s and answers fastest, in 7–10 s.
Embodied RAG++ preprocesses in 14–25 s, the shortest among
the retrieval-based methods, answers in 15–20 s, faster than Naive
LLM and LightRAG, and ties with EmbodiedRAG for the lowest
total token count (0.2M tokens over the evaluation). After the first
queries, most of its capability lookups are served from Capability
Cards without new retrieval or extraction (§V-C).
Because all baselines use the same SR module, the accuracy
differences between them and Embodied RAG++ do not come from
feasibility resolution. Adding the module to Naive LLM raises
robot and pair accuracy on every backbone while leaving object
accuracy unchanged, but Naive LLM + SR remains far below
Embodied RAG++ on the open-weight backbones. The ablations
in Table III locate the remaining gap in retrieval design: removing
aspect splitting or model-level sharing reduces accuracy on both
backbones, whereas replacing Capability Cards with raw retrieved
excerpts does not.
C. Capability Card Cache Behavior
To characterize the Capability Card cache, we replay the 20-
query evaluation log in order from a cold cache and record, for
every judged candidate, whether its model’s card already covered
the required fields (a hit) or required new extraction (a miss).
Fig. 5(a) shows that misses concentrate in the first queries, while
models are still unseen; after a model’s first query, every later query
that involves that model reads the required capability from its card.
In Fig. 5(b), the cumulative hit rate rises throughout the run, and
the steady-state hit rate measured over a full deployment history
exceeds 85%: most capability lookups in deployment require no
new retrieval or extraction call. The baselines have no counterpart
to this curve. GraphRAG and LightRAG pay their extraction cost
upfront over the whole corpus, regardless of which facts later
queries need, while Naive LLM and EmbodiedRAG keep no model-
keyed memory, so their per-query cost does not decrease as queries
about the same appliance model accumulate.
D. Ablation Study
Table III isolates the contribution of each component by remov-
ing one at a time on the same 20 E2W-Bench scenes with the two
open-weight backbones. Gemini-3.5-Flash is excluded because the
full method already reaches near-ceiling accuracy with it (Table II),
which leaves little room to measure degradation.
Capability Cardsdo not affect accuracy. Replacing them with
raw retrieved excerpts, so that JUDGEreasons directly over the
retrieved passages instead of cached fields, yields identical accuracy6

TABLE II: Exact-match accuracy of robot selection (Robot), object selection (Object), and the joint robot–object pair (Pair) on the 20
E2W-Bench queries, one per scene, across systems and LLM backbones. “+ SR” denotes use of the shared deterministic Spatial Reasoning
module (§IV-E). Naive LLM is reported both without and with this module to isolate its contribution; all other baselines use the same
module as Embodied RAG++.
SystemQwen2.5-32B Llama3.3-70B Gemini-3.5-Flash
Robot Object Pair Robot Object Pair Robot Object Pair
Naive LLM 0.20 0.45 0.10 0.45 0.65 0.30 0.50 0.95 0.50
Naive LLM + SR 0.40 0.45 0.20 0.70 0.65 0.35 1.00 0.95 0.95
GraphRAG + SR 0.55 0.45 0.30 0.75 0.70 0.55 1.00 0.95 0.95
LightRAG + SR 0.55 0.45 0.25 0.35 0.45 0.35 1.00 0.85 0.85
EmbodiedRAG + SR 0.60 0.45 0.30 0.60 0.75 0.50 1.00 0.95 0.95
Embodied RAG++ (ours) 0.90 0.90 0.90 1.00 1.00 1.00 1.00 1.00 1.00
Naive LLM GraphRAG LightRAG
EmbodiedRAGOurs101103105
7–10s4–12h
3–5h
43–46s
14–25sseconds (log scale)(a) Preprocessing time
Naive LLM GraphRAG LightRAG
EmbodiedRAGOurs020406080
29–57s
6–12s46–70s
7–10s15–20sseconds(b) Query + generation time
Naive LLM GraphRAG LightRAG
EmbodiedRAGOurs0246
2M6.1M5.8M
0.2M 0.2Mtokens (M)(c) Total tokens
Fig. 4: Efficiency comparison across systems on the same 20 E2W-Bench scenes as Table II. (a) Average one-time preprocessing time,
including graph/index construction where applicable, shown on a log scale. (b) Average per-query retrieval-and-generation time. (c) Total
tokens consumed across the full evaluation sweep. Bars mark the midpoint of each system’s reported range; error bars span the full
reported minimum–maximum.
1234567891011121314151617181920024(a) Per-query cache outcomes (cold start)
Query index (E2W-0001 – E2W-0020, evaluation order)Judged candidate modelscache hit cache miss (extract/refine)
1 5 10 15 20020406080100(b) Cumulative hit rate over the run
Query indexCapability Card hit rate (%)this run (cold start)
steady-state (deployment)
Fig. 5: Capability Card cache analysis: (a) For each query, the number of judged candidate models whose Capability Card was already
covered (hit, green) versus needed fresh extraction or refinement from the manual (miss, red). (b) Cumulative hit rate over the run (solid)
against the steady-state hit rate measured over a full deployment history (dashed).
TABLE III: Ablation study of Embodied RAG++ on the 20 E2W-
Bench queries. Each row removes exactly one component; we report
exact-match accuracy of robot selection (Robot), object selection
(Object), and the joint pair (Pair).
ConfigurationQwen2.5-32B Llama3.3-70B
Robot Object Pair Robot Object Pair
Embodied RAG++ (full) 0.90 0.90 0.90 1.00 1.00 1.00
w/o Capability Cards 0.90 0.90 0.90 1.00 1.00 1.00
w/o Model-level Sharing 0.85 0.85 0.85 0.75 0.75 0.75
w/o Aspect Splitting 0.80 0.75 0.75 0.65 0.65 0.65
w/o Spatial Reasoning 0.65 0.85 0.65 0.45 0.75 0.45
on both backbones. Accuracy therefore depends on the extracted
content, not on whether it is cached. The role of the cards is to avoid
re-retrieving and re-extracting the same fields for later instances andqueries (§V-C).
Removingmodel-level sharing, that is, keeping a separate card
for each physical instance instead of one card per model, lowers
pair accuracy by 5 pp on Qwen2.5-32B and by 25 pp on Llama3.3-
70B, and requires roughly six to seven times more LLM calls.
This variant still extracts and caches fields, but once per instance
rather than once per model. Each instance therefore requires an
independent extraction from the manual, and each extraction is
another opportunity for the LLM to misread it, which affects
Llama3.3-70B more. Sharing cards across the instances of a model
thus reduces both extraction errors and the number of LLM calls.
Removingaspect splittinglowers object accuracy from 0.90
to 0.75 on Qwen2.5-32B and from 1.00 to 0.65 on Llama3.3-
70B. With a single embedding for a compound requirement, one
constraint can dominate the query and push the evidence for the
other out of the retrieved pool (§IV-C).
Removingspatial reasoningcauses the largest drop. Object7

accuracy decreases moderately, to 0.85 and 0.75, while robot
accuracy falls to 0.65 on Qwen2.5-32B and 0.45 on Llama3.3-70B.
No baseline corresponds to this row, because all baselines keep
the SR module (§V-A). The drop shows that the LLM backbones
do not reliably reproduce reachability and height reasoning from
a serialized scene graph, which the deterministic layer performs
exactly.
Both backbones show the same ordering: spatial reasoning has
the largest effect on accuracy, followed by aspect splitting and
model-level sharing, while Capability Cards have no measurable
effect. Llama3.3-70B degrades more than Qwen2.5-32B under most
ablations, so the larger backbone does not remove the dependence
on these components.
VI. CONCLUSION
This paper addressed capability-grounded robot–object assign-
ment: identifying which physical object satisfies a functional re-
quirement documented only in an external manual, and which avail-
able robot can feasibly reach it under dynamic scene conditions.
We introduced E2W-Bench, which links dynamic hybrid scene
graphs to real product manuals with machine-verified unique ground
truth, and Embodied RAG++, which scopes retrieval to the linked
manual, caches extracted capabilities per product model in reusable
Capability Cards, and delegates feasibility to deterministic spatial
reasoning. Across three LLM backbones and four baselines, all but
one of which share the same spatial reasoning module, Embodied
RAG++ achieves the highest pair accuracy and ties EmbodiedRAG
for the lowest total token count. The ablations attribute its advantage
over the baselines to aspect-split retrieval and model-level sharing
of Capability Cards, and show that removing deterministic spatial
reasoning causes the largest accuracy drop. Our evaluation assumes
that each object’s link to its manual is known and is conducted in
simulation. Future work will broaden E2W-Bench’s category and
archetype diversity, infer the object-to-manual link from visual or
textual evidence, and validate the spatial-reasoning layer on physical
robots under real perception and localization noise.
REFERENCES
[1] I. Armeniet al., “3D scene graph: A structure for unified semantics,
3D space, and camera,” inIEEE/CVF International Conference on
Computer Vision (ICCV), 2019, pp. 5664–5673, arXiv:1910.02527.
[2] N. Hugheset al., “Hydra: A real-time spatial perception system for
3D scene graph construction and optimization,” inRobotics: Science
and Systems (RSS), 2022, arXiv:2201.13360.
[3] Q. Guet al., “ConceptGraphs: Open-vocabulary 3D scene graphs
for perception and planning,” inIEEE International Confer-
ence on Robotics and Automation (ICRA), 2024, pp. 5021–5028,
arXiv:2309.16650.
[4] K. Ranaet al., “SayPlan: Grounding large language models using 3D
scene graphs for scalable robot task planning,” inConference on Robot
Learning (CoRL), 2023, arXiv:2307.06135.
[5] D. Z. Chen, A. X. Chang, and M. Nießner, “ScanRefer: 3D object
localization in RGB-D scans using natural language,” inEuropean
Conference on Computer Vision (ECCV), 2020, pp. 202–221.
[6] X. Maet al., “SQA3D: Situated question answering in 3D scenes,” in
International Conference on Learning Representations (ICLR), 2023,
arXiv:2210.07474.
[7] P. Lewiset al., “Retrieval-augmented generation for knowledge-
intensive NLP tasks,” inAdvances in Neural Information Processing
Systems (NeurIPS), vol. 33, 2020, pp. 9459–9474, arXiv:2005.11401.
[8] D. Edgeet al., “From local to global: A graph RAG approach to query-
focused summarization,”arXiv preprint arXiv:2404.16130, 2024.
[9] P. Sarthiet al., “RAPTOR: Recursive abstractive processing for tree-
organized retrieval,” inInternational Conference on Learning Repre-
sentations (ICLR), 2024, arXiv:2401.18059.
[10] B. J. Gutiérrezet al., “HippoRAG: Neurobiologically inspired long-
term memory for large language models,” inAdvances in Neural
Information Processing Systems (NeurIPS), 2024, arXiv:2405.14831.[11] Q. Xieet al., “Embodied-RAG: General non-parametric em-
bodied memory for retrieval and generation,”arXiv preprint
arXiv:2409.18313, 2024.
[12] Y . Feiet al., “Open-world 3d scene graph generation for retrieval-
augmented reasoning,” inProceedings of the AAAI Conference on
Artificial Intelligence, vol. 40, no. 5, 2026, pp. 3894–3902.
[13] Y . Changet al., “RAG-3DSG: Enhancing 3d scene graphs
with re-shot guided retrieval-augmented generation,”arXiv preprint
arXiv:2601.10168, 2026.
[14] D. Azumaet al., “ScanQA: 3d question answering for spatial scene
understanding,” in2022 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR). IEEE, 2022, pp. 19 107–19 117.
[15] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world
graphs,”IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, vol. 42, no. 4, pp. 824–836, 2020.
[16] J. Johnsonet al., “Billion-scale similarity search with GPUs,”IEEE
Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2019.
[17] Y . Wanget al., “LEANN: A low-storage vector index,”arXiv preprint
arXiv:2506.08276, 2025.
[18] G. Izacardet al., “Unsupervised dense information retrieval with
contrastive learning,”Transactions on Machine Learning Research,
2022, arXiv:2112.09118.
[19] Qwen Team, “Qwen2.5 technical report,”arXiv preprint
arXiv:2412.15115, 2024.
[20] Llama Team, AI @ Meta, “The Llama 3 herd of models,”arXiv
preprint arXiv:2407.21783, 2024.
[21] Google DeepMind, “Gemini 3.5 Flash model card,” https://deepmind.
google/models/model-cards/gemini-3-5-flash/, May 2026, published
19 May 2026; accessed 14 August 2026.
[22] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “LightRAG: Simple and
fast retrieval-augmented generation,”arXiv preprint arXiv:2410.05779,
2024.
8