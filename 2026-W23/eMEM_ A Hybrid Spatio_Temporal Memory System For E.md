# eMEM: A Hybrid Spatio-Temporal Memory System For Embodied Agents

**Authors**: A. Haroon Rasheed, Maria Kabtoul

**Published**: 2026-06-02 09:22:12

**PDF URL**: [https://arxiv.org/pdf/2606.03374v1](https://arxiv.org/pdf/2606.03374v1)

## Abstract
We present eMEM (Embodied Memory), a hybrid graph-based memory system for embodied agents operating in physical environments. Current agent memory architectures, such as Generative Agents, MemGPT, and A-MEM, treat memory as text streams or knowledge graphs, but embodied agents require memory that is simultaneously searchable by meaning, space, and time. eMEM fills this gap with a multi-index architecture (SQL ITE for structured storage, hnswlib for approximate nearest neighbour semantic search, and an R-tree for spatial queries) unified behind a single graph model. A tiered consolidation pipeline transforms raw perceptual observations into compressed summaries, mirroring hippocampal-neocortical consolidation in biological systems. Ten agent-facing recall tools expose memory retrieval primitives, including concept-to-location resolution and cross layer recall, as first-class operations for LLM tool calling. The system is fully embedded and runs in-process alongside the agent. In addition we introduce eMEM-Bench v1, a benchmark we construct over ProcTHOR-10K scenes for embodied memory evaluation. The benchmark is organised explicitly around eight cognitive-psychology paradigms (DRM lures, pattern separation, pattern completion, source monitoring, context-dependent retrieval, long-horizon interference, serial position, and a foil augmented retention curve), each chosen so that the result is interpretable against the broader memory-systems literature in humans and prior agent-memory systems; a level of diagnostic that surface-task benchmarks like LoCoMo or OpenEQA cannot provide. eMEM scores 80.8 weighted mean over 988 probes, with a flat retention curve at ceiling from 1 h to 1 yr of simulated delay on room-unique items. We show that a pure RAG baseline (the flat_rag ablation) loses 30 pt on context dependent retrieval and 29 pt on DRM lure rejection, isolating the contribution of multi-layer storage and consolidation respectively. We release both the system and the benchmark code.

## Full Text


<!-- PDF content starts -->

EMEM: A H YBRID SPATIO -TEMPORAL MEMORY SYSTEM FOR
EMBODIED AGENTS
A. Haroon Rasheed Maria Kabtoul
Automatika Robotics
Inria, France *
ABSTRACT
We present eMEM (Embodied Memory), a hybrid graph-based memory system for embodied
agents operating in physical environments. Current agent memory architectures, such as Gen-
erative Agents [Park et al., 2023], MemGPT [Packer et al., 2023], and A-MEM [Xu et al., 2025],
treat memory as text streams or knowledge graphs, but embodied agents require memory that is
simultaneously searchable by meaning ,space , and time. eMEM fills this gap with a multi-index
architecture ( SQL ITEfor structured storage, hnswlib for approximate nearest-neighbour semantic
search, and an R-tree for spatial queries) unified behind a single graph model. A tiered consolidation
pipeline transforms raw perceptual observations into compressed summaries, mirroring hippocampal-
neocortical consolidation in biological systems [McClelland et al., 1995, Kumaran et al., 2016]. Ten
agent-facing recall tools expose memory retrieval primitives, including concept-to-location resolution
and cross-layer recall, as first-class operations for LLM tool calling. The system is fully embedded
and runs in-process alongside the agent. In addition we introduce eMEM-Bench v1 , a benchmark
we construct over ProcTHOR-10K [Deitke et al., 2022] scenes for embodied memeory evaluation.
The benchmark is organised explicitly around eight cognitive-psychology paradigms (DRM lures,
pattern separation, pattern completion, source monitoring, context-dependent retrieval, long-horizon
interference, serial position, and a foil-augmented retention curve), each chosen so that the result
is interpretable against the broader memory-systems literature in humans and prior agent-memory
systems; a level of diagnostic that surface-task benchmarks like LoCoMo or OpenEQA cannot
provide. eMEM scores 80.8 weighted-mean over 988 probes, with a flat retention curve at ceiling
from 1 h to 1 yr of simulated delay on room-unique items. We show that a pure RAG baseline (the
flat_rag ablation) loses 30 pt on context-dependent retrieval and 29 pt on DRM lure rejection,
isolating the contribution of multi-layer storage and consolidation respectively. We release both the
system and the benchmark code.2
Keywords embodied memory ·spatio-temporal memory ·retrieval-augmented generation ·robotics ·consolidation ·
LLM tool use
1 Introduction
Embodied agents (i.e. robots, drones, autonomous vehicles) accumulate experience that is inherently spatial and
temporal. A robot patrolling a building does not just need to remember what it saw; it needs to remember where and
when it saw it, relate observations across perceptual modalities, and compress weeks of patrol data into retrievable
knowledge.
Current LLM-agent memory systems, while increasingly sophisticated, are designed around text-based conversa-
tional agents. Generative Agents [Park et al., 2023] maintains a timestamped memory stream with reflections,
MemGPT [Packer et al., 2023] pages information between context and archival storage, and A-MEM [Xu et al.,
2025] organises memories as interconnected Zettelkasten notes; Mem0 [Chhikara et al., 2025] and Zep [Rasmussen
∗Correspondence: {haroon, maria}@automatikarobotics.com
2Code: https://github.com/automatikarobotics/ememarXiv:2606.03374v1  [cs.RO]  2 Jun 2026

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
et al., 2025] extend this line with production-oriented long-term stores and temporal knowledge graphs. None of these
systems natively handles spatial coordinates, multi-modal perception layers, or the kind of spatial reasoning required by
embodied agents (“where is the kitchen?”, “what objects are near the charging station?”). Robotics research has in
parallel produced sophisticated spatial representations, e.g. 3D scene graphs such as Hydra [Hughes et al., 2022] and
ConceptGraphs [Gu et al., 2024], visual-language maps [Huang et al., 2023], and semantic mapping [Raychaudhuri and
Chang, 2025]; these are, however, typically perceptual systems focused on the current state of the environment, not
memory systems that track how the environment changes over time, consolidate experience into knowledge, or support
episodic recall. eMEM bridges this gap by combining the temporal and consolidation mechanisms of LLM-agent
memory with the spatial grounding of robotic scene understanding.
Each of eMEM’s components is, individually, established art: SQLite, HNSW [Malkov and Yashunin, 2020], and R-tree
indices are standard; DBSCAN [Ester et al., 1996] is an algorithm from 1996; LLM-summarisation-as-consolidation is
implemented by Generative Agents / MemGPT / A-MEM; and ReAct-over-tools is Yao et al. [2023]. The novelty lies
in the combination. Spatial grounding, multi-layer perception, and tiered consolidation in a single embedded package —
and in the locate /recall composition pattern that lets higher-order recall emerge from a small set of primitives. We
state this explicitly so readers can calibrate the contribution against prior work.
Four design commitments shape the system. First, spatial grounding is not optional: every observation, entity,
and summary carries a position, and spatial context is a first-class retrieval axis alongside semantic similarity and
temporal order. Second, memories are organized in layers. An embodied agent may run multiple perception models
simultaneously (e.g. a VLM describing scenes, an object detector listing entities, a place classifier labelling regions),
and eMEM stores these as independent, co-located observations on different layers rather than forcing them into a
hierarchy, such that the structure between the layers is emergant and not enforced. Third, places emerge from data;
there is no dedicated “region” or “place” primitive. If a VLM consistently describes coordinates around (10,10)as
“kitchen”, the kitchen exists as a cluster of observations and gists, and queries like “where is the kitchen?” are resolved
by semantic search and spatial aggregation over the existing primitives. Fourth, we think in recall patterns: when a
person remembers “the kitchen”, they recall where it is (spatial), what it is like (cross-modal synthesis), and what
happened there recently (temporal); new capabilities should therefore emerge from new query shapes over existing
primitives, not from new primitives.
Our contributions are as follows.
•We introduce a hybrid graph-based memory architecture with four node types and six edge types, unified over
three indices (SQLite, HNSW, R-tree) so that semantic, spatial, and temporal queries resolve against the same
underlying store (§2).
•We develop a tiered consolidation pipeline (working →short-term →long-term →archived) with two-
phase archival and DBSCAN-clustered time-window summarisation, giving each observation a biologically
motivated lifecycle (§3).
•We expose a 10-tool retrieval surface covering concept-to-location resolution, spatial-semantic fusion, and
cross-layer recall as first-class LLM tool calls (§5).
•We introduce eMEM-Bench v1 , an embodied-memory benchmark built over ProcTHOR-10K scenes [Deitke
et al., 2022] and organised explicitly around eight cognitive-psychology paradigms drawn from human-
memory research — DRM lures [Roediger and McDermott, 1995], pattern separation [Yassa and Stark,
2011] and completion, source monitoring [Johnson et al., 1993], context-dependent retrieval [Godden and
Baddeley, 1975], long-horizon interference, serial position [Murdock, 1962], and a foil-augmented retention
curve [Ebbinghaus, 1885]. Each paradigm is chosen so that it isolates a distinct memory function with an
established empirical signature, has a natural embodied-agent analogue, and is programmatically generable at
scale. Existing agent-memory benchmarks are either text-only (LoCoMo, conversational dialogue) or category-
defined (OpenEQA, perception-grounded QA); neither exercises the embodied axes (spatial, multi-modal,
interoceptive) while being interpretable against the broader memory-systems literature (§8).
•As a methodological contribution we introduce a foil-augmented probe construction for embodied-agent
benchmarks: yes/no probes paired with cross-scene absent foils, which expose otherwise-hidden bias and
instruction-parroting failure modes (§8.5, §9).
The rest of the paper is organised as follows. §2 describes the architecture, data model, and indices. §3 details the
consolidation pipeline. §5 presents the LLM tool surface. §6 relates eMEM’s design to neuroscience evidence on
memory and spatial representation. §7 positions eMEM against prior work. §8 introduces eMEM-Bench v1, justifies the
paradigm panel, and reports experimental results including the cross-system ablation against pure RAG. §9 discusses
limitations and future directions.
2

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
SPATIO TEMPORAL MEMORY facade
WorkingMemory
in-process bufferTools
10 retrieval primitivesConsolidation
gist + entity
MEMORY STORE (hybrid persistence)
SQLite
structuredHNSW
semanticR-tree
spatial
Figure 1: eMEM system overview. The SPATIO TEMPORAL MEMORY facade orchestrates a working-memory buffer, a
retrieval tool surface, and a consolidation engine, all backed by a hybrid MEMORY STORE combining SQLite, HNSW,
and R-tree indices.
Episode a Episode b
Obs 1 Obs 2 Entity 1 Entity 2
GistBELONGS_TOFOLLOWS
SUBTASK_OF
SUMMARIZESOBSERVED_IN COOCCURS_WITH
Figure 2: Data model. Four node types (observation, episode, gist, entity) connected by six edge types. Gists summarise
groups of observations; entities are tracked across observations and linked by co-occurrence.
2 Architecture
Figure 1 shows the overall system layout. A SPATIO TEMPORAL MEMORY system orchestrates three cooperating
subsystems (a working-memory buffer, a consolidation engine, and a 10-tool retrieval surface) on top of a hybrid
MEMORY STORE that combines SQLite, HNSW, and R-tree indices. All subsystems run in-process, and the only
persistence state is a SQLite file together with a binary HNSW index on disk.
2.1 Data Model
eMEM represents memory as a graph with four node types and six edge types (Figure 2). Observation nodes carry
a single perceptual event recorded with text, coordinates, timestamp, and a layer tag; they are the analogue of a
hippocampal episodic trace. Episode nodes are task containers that group observations and support hierarchical nesting,
playing the role of episodic context. Gist nodes are consolidated summaries with spatial extent (a centre and a radius)
produced from a cluster of observations, and play the role of a neocortical schema. Entity nodes are persistent object
concepts tracked across observations and merged under a learned similarity threshold, giving the system a semantic-
memory store of objects. These nodes are related by six edge types: BELONGS_TO links an observation to its episode;
FOLLOWS andSUBTASK_OF relate episodes to one another; SUMMARIZES links a gist to the observations it subsumes;
OBSERVED_IN records the observation in which an entity was seen; and COOCCURS_WITH records co-presence between
entities.
3

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Working
in-process
bufferShort-term
SQLite + HNSW
+ R-treeLong-term
text & embeddings
preservedArchived
text dropped,
gist preservedflush
(batch/timer)episode end
(consolidate)maintenance
(age-out)
Figure 3: Memory tiers. Observations are encoded into a working buffer, flushed to short-term storage where they
are searchable, consolidated into long-term memory at episode boundaries (with text and embeddings preserved), and
eventually archived (i.e. leaving only a gist searchable) once they age out.
2.2 Memory Tiers
Memory in eMEM progresses through four tiers, paralleling the biological consolidation pathway from hippocampus to
neocortex [McClelland et al., 1995, Diekelmann and Born, 2010]; see Figure 3. At the working tier, observations are
held in an in-process deque buffer that auto-flushes on batch size (default 5) or time interval (default 2 s), with entity
extraction triggered periodically from the flush callback. On flush, observations enter the short-term tier, where they are
stored across SQLite, HNSW, and the R-tree and become fully searchable while awaiting consolidation. At episode
end they are promoted to the long-term tier with text and embeddings preserved; raw observations therefore remain
searchable alongside their gists, which ensures that no detail is lost during active operation. After a configurable age a
maintenance() sweep moves them to the archived tier, dropping raw text and embeddings and leaving only the gist
searchable, which bounds long-term storage growth; a necessity for physical agents expected to operate continuously
over an arbitrarily long period.
2.3 Three-Index Storage
eMEM engages three complementary indices, each tuned to a different retrieval axis, unified under a single store
interface. SQLite holds structured data: observation text, timestamps, layer names, tier status, episode membership,
entity attributes, and graph edges. It is indexed on timestamp, layer, episode, tier, and entity name, and provides the
relational backbone for filtering and joins. HNSW [Malkov and Yashunin, 2020] (via hnswlib ) provides approximate
nearest-neighbour semantic search over embedding vectors; a single index stores embeddings for observations, gists,
and entities, with an ID mapping in SQLite tracking node type. We use cosine distance, build parameters M=16 and
efconstruction =200 , and search ef=50 by default, and the binary format persists across process restarts. The R-tree (via
rtree ) supports 3D spatial range and nearest-neighbour queries; it is rebuilt on startup from SQLite coordinates, and
string UUIDs are mapped to integer IDs internally. The rebuild cost is O(NlogN)in the number of stored observations.
A typical query such as “what did the robot see near the kitchen recently?” engages all three indices: HNSW finds
semantically relevant memories, the R-tree filters by spatial proximity, and SQLite applies layer and time constraints.
Semantic search can optionally be biased towards recent observations. When the recency weight α > 0, the score
combines HNSW distance with temporal recency as score = dHNSW +α·tage/thalflife , which is useful for queries
such as “where is the <name_of_an_object> ?” that should prefer the latest sighting. The weight is zero by default so
that the standard ranking is purely semantic.
2.4 Multi-Layer Perception
Observations carry a layer_name tag that identifies their perceptual source. Layers are user-defined and represent
a semantic context i.e. different questions a perception stack asks about the environment; the same coordinates may
therefore carry observations on several layers simultaneously. A VLM layer might record “a kitchen with white cabinets
and a wooden table”, a detections layer “chair, table, refrigerator, microwave”, a place identification layer “kitchen”,
and a human-presence layer “empty room”, all at the same position. These are four independent observations rather
than attributes of a single entry: all query tools accept an optional layer filter, the get_current_context tool groups
nearby observations by layer for structured presentation, and the consolidation engine uses layer-aware synthesis for
multi-layer clusters.
3 Consolidation
eMEM’s consolidation engine transforms raw observations into compressed long-term knowledge, implementing
the core insight of Complementary Learning Systems theory [McClelland et al., 1995, Kumaran et al., 2016]: rapid
encoding of specific episodes (observations) followed by gradual extraction of regularities (gists).
4

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
(a)
episode
session 1 session 2 session 3
gist1 gist2 gist3temporal
chunking
(b)
time-
windowcluster A cluster Bnoise
gistA gistBDBSCAN
spatial
Figure 4: Consolidation. (a) At episode end, observations are temporally chunked (default gap: 30 min) into sessions;
each chunk produces a focused gist. (b) For non-episodic observations that age past the consolidation window, DBSCAN
spatial clustering produces one gist per cluster; ungrouped observations remain as-is until archived.
3.1 Episode Consolidation
When an episode ends, its observations pass through four stages, the first three illustrated in Figure 4a. Observations are
first sorted by timestamp and split into temporal chunks wherever the gap between consecutive observations exceeds
a configurable window (default 30 min); this ensures that a long episode, e.g. a long running task spanning hours,
produces several focused gists rather than one monolithic summary. Each chunk is then summarised into a separate
gist via an LLM, with a concatenation fallback when no LLM is configured, and multi-layer chunks are presented to
the summariser grouped by layer so that it can produce a structured cross-layer synthesis. Alongside summarisation,
entities are extracted from every episode observation and either merged with existing entities (by semantic similarity
and spatial proximity) or created afresh. Finally, the observations themselves are promoted to the long-term tier with
text and embeddings preserved, so that they remain searchable alongside their gists.
3.2 Two-Phase Archival
Consolidation follows a two-phase gradient inspired by biological consolidation during sleep [Diekelmann and Born,
2010, Born and Wilhelm, 2012]. At episode end the first phase promotes observations to the long-term tier with
text, embeddings, and spatial entries all preserved, so that both the raw observation and the gist remain searchable;
the raw provides detail while the gist provides context. The second phase runs later as a maintenance sweep over
observations older than archive_after_seconds (default one hour, though a deployment might set this to days): text
and embeddings are dropped and only the gist remains searchable. An internal maintenance() method triggers this
phase, which in a deployed system would run during idle periods, loosely analogous to sleep-dependent consolidation
in biological memory.
3.3 Time-Window Consolidation
Not all observations arrive inside an episode. For those that age past the consolidation window outside any episode,
a DBSCAN clustering [Ester et al., 1996] groups observations by spatial proximity and each cluster is consolidated
independently (Figure 4b). This avoids merging observations from distinct locations into incoherent summaries, and it
naturally produces one gist per place visited during a free-running interval.
3.4 Entity Tracking
Entities are persistent object representations that survive consolidation. Extraction happens at two points. At flush time,
observations accumulate in an entity buffer; after a configurable number of flushes (default 10) or a configurable time
interval (default 60 s), whichever comes first, the buffer is drained and entities are extracted. Entities are also extracted
during episode consolidation itself, guarded by the deduplication flag. When the LLM identifies an object, the engine
checks for existing entities with similar names (cosine similarity above 0.85 by default) at nearby positions (within 5 m
5

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Ingest HNSW recall@10 R-tree Size
N(obs/s) ef=16 ef=50 ef=200 rebuild (s) SQLite (MiB) HNSW (MiB)
1032 558 0.59 0.89 1.00 0.02 0.72 1.61
104947 0.15 0.34 0.75 0.42 6.27 16.07
105561 0.02 0.06 0.21 5.25 62.34 160.64
Table 1: Systems benchmarks. Ingest throughput excludes LLM-inference cost (entity extraction disabled; synthetic
random-unit-vector embeddings). HNSW recall@10 measured against a brute-force cosine top-10 over 500 random
queries. R-tree rebuild measured by reopening a closed database. SQLite and HNSW file sizes are post-ingestion
on-disk footprints.
Tool Axis Role
semantic_search meaning Unified ANN search across observations, gists, and entities.
spatial_query location Observations within a radius of a 3D point.
temporal_query time Observations in a time range, chronologically ordered.
episode_summary episodes Gist summaries of a completed episode.
search_gists long-term Semantic search restricted to consolidated gists.
entity_query objects Query persistent entity records (by name or area).
get_current_context situation Nearby observations grouped by layer, plus gists, entities, and
recent activity.
body_status body Latest interoception readings (battery, temperature, joints).
locate concept →place Resolve a semantic concept to a centroid and spread.
recall cross-modal Chain locate +spatial +gists+entities for a full recall.
Table 2: eMEM’s 10-tool retrieval surface. The top block lists the eight core primitives covering meaning, location,
time, and body state. The bottom block lists two higher-order tools that compose the primitives.
by default); matches are merged, with coordinates averaged, observation counts incremented, and temporal bounds
extended, and new entities are created otherwise. Co-occurrence is tracked via COOCCURS_WITH edges, enabling queries
such as “what objects appear together with the red chair?” .
4 Systems Characterisation
Running memory in-process and for arbitrarily long periods of time motivates measuring concrete numbers on ingest
throughput, storage footprint, retrieval fidelity, and startup cost. We measure on an NVIDIA RTX A5000 workstation
using a synthetic random-unit-vector embedder (no embedding-model cost) at three observation counts: N= 103,104,
105. The vector dimension is fixed at 384, matching the default embedding_dim . All measurements use the default
configuration (Table 6) except where noted.
Three observations deserve emphasis. First, storage grows predictably: the SQLite file and HNSW index together
consume ≈23MiB at N= 104and≈223MiB at N= 105on a 384-dimensional store. Linear extrapolation gives
a rough upper bound of ≈2.2GiB at 106— manageable on a robot’s onboard storage, but enough to motivate the
archival phase of the consolidation pipeline (§3.2). Second, R-tree rebuild at startup is sub-second at 104and≈5s at
105, consistent with the O(NlogN)growth (§2.3); an incremental persistence path is a straightforward extension for
much longer-lived deployments. Third, ingest throughput declines with N(from∼2.6k obs/s at 103to∼0.6k obs/s
at105), which tracks the HNSW insertion cost rather than any SQLite or R-tree ceiling.
These numbers exclude entity extraction (which would add an LLM-inference cost per flush interval) and other
LLM-side consolidation cost. The separable LLM-inference cost is reported alongside the benchmark results in §8.
5 LLM Tool Interface
eMEM exposes memory retrieval to an agent as a set of tools; each is a callable with a JSON-schema description
suitable for LLM tool use [Yao et al., 2023]. Tools return formatted strings optimised for token efficiency.
6

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
The core tools (top block of Table 2) cover the three primary retrieval axes. semantic_search ,search_gists ,
andentity_query address the meaning axis; spatial_query addresses location; and temporal_query and
episode_summary address time. Two further primitives address situational awareness ( get_current_context ,
which returns nearby observations grouped by layer together with gists, entities, and recent activity) and body state
(body_status , which returns the latest interoception readings).
Two higher-order tools sit on top of these primitives and implement recall patterns that would otherwise require
multi-step reasoning from the agent. locate(concept) resolves a semantic concept to a spatial position by running
semantic_search , collecting coordinates from matching observations, gists, and entities, and returning a centroid,
a spread radius, and a match count; it therefore bridges “kitchen” (a concept) and (10.3,10.1,0.0)(a position).
recall(query) implements “tell me about X” by chaining locate with spatial aggregation: it locates the concept,
runs spatial_query at that location across every layer, gathers consolidated gists by area, and pulls tracked entities,
formatting the result grouped by layer to produce a cross-modal summary grounded in spatial reality. This composition
pattern, treating retrieval primitives as building blocks from which higher-order recall emerges, is how we apply the
design commitment to think in recall patterns: new capabilities come from new query shapes, not new primitives.
6 Neuroscience Foundations
eMEM is not a model of the brain, but many of its design choices are informed by computational counterparts of
well-established findings in the neuroscience of memory; we sketch that mapping here.
The Complementary Learning Systems theory [McClelland et al., 1995, Kumaran et al., 2016] posits two cooperating
systems: the hippocampus rapidly encodes specific episodes while the neocortex gradually extracts statistical regularities
through consolidation. eMEM’s observation nodes play the role of the hippocampal traces (specific, timestamped,
spatially grounded) and gist nodes play the role of the neocortical abstractions. The four-tier pipeline (working →
short-term →long-term →archived) implements the CLS consolidation gradient more granularly than the two-system
biological model, but preserves its essential dynamic: rich episodic traces are progressively compressed into general
knowledge, with a two-phase archival that matches the observation that recently consolidated memories remain
accessible for a period before full compression [Diekelmann and Born, 2010, Born and Wilhelm, 2012]. A recent
RAG-oriented system, HippoRAG [Jiménez Gutiérrez et al., 2024], also draws on the CLS framing but applies it to
text-document retrieval via personalised PageRank rather than to embodied spatio-temporal memory.
A related line of work concerns the hippocampal spatial coordinate system for memory, from O’Keefe’s place
cells [O’Keefe and Dostrovsky, 1971] and the grid cells of the entorhinal cortex [Hafting et al., 2005] to a recent
perspective [Benna and Fusi, 2021] that reframes place cells as general-purpose memory cells whose spatial tuning
emerges from compressing sequential experience. Our commitment that places emerge from data follows precisely
this view: the kitchen is not a dedicated data structure but a cluster of observations that share spatial proximity and
semantic content, surfaced at query time by locate . Tulving’s distinction between episodic memory (specific events in
spatio-temporal context) and semantic memory (general knowledge abstracted from episodes) [Tulving, 1972, 1983]
maps directly onto eMEM’s node taxonomy: observations and episodes carry the when andwhere of particular events,
while gists and entities hold the knowledge abstracted from them, and the consolidation pipeline is the computational
route from one to the other.
The space-from-sequence intuition has a recent algorithmic instantiation in clone-structured cognitive graphs [George
et al., 2021], generalised to a theory of the hippocampus as a latent-sequence learner [Raju et al., 2024]. The same
family of place-cell and splitter-cell behaviour is recovered from a single mechanism: a cloned hidden Markov model
trained on egocentric, action-conditioned observations. But the position is further sharpened: Euclidean coordinates are
not merely emergent but, on this view, an artefact, with the latent graph alone supporting recall, planning, and transfer.
eMEM uses coordinates because a robot already has them, an engineering choice rather than a philosophical one; both
perspectives agree on graph-shaped memory and on consolidation as the bridge from episodic to semantic, and the
latent-graph machinery composes naturally with our explicit spatial axis. We return to this convergence in §9.
Two further findings motivate the way retrieval and update work in eMEM. Retrieved memories become labile and must
be reconsolidated, allowing new information to integrate into existing traces [Nader et al., 2000]; the entity auto-merge
step implements exactly this, so a re-observed entity updates its record (coordinates, observation count, temporal
bounds) rather than producing a duplicate. Likewise, retrieval is enhanced when context matches encoding [Godden and
Baddeley, 1975, Tulving and Thomson, 1973], and get_current_context exploits this by using the agent’s current
position as an automatic retrieval cue, mirroring how returning to a place tends to trigger recall of experiences from
there.
7

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
System Spatial Temporal Consol. Multi-layer Embedded
LLM-agent memory
Generative Agents [Park et al., 2023] ✗ ✓ ✓ ✗ ✗
MemGPT [Packer et al., 2023] ✗ ✓ ✗ ✗ ✓
A-MEM [Xu et al., 2025] ✗ ✓ ✓ ✗ ✗
Mem0 [Chhikara et al., 2025] ✗ ✓ ✓ ✗ ✗
Zep [Rasmussen et al., 2025] ✗ ✓ ✓ ✗ ✗
Embodied spatial representations
Hydra [Hughes et al., 2022] ✓ ∗ ✗ ✓ ✓
ConceptGraphs [Gu et al., 2024] ✓ ∗ ✗ ✓ ✓
VLMaps [Huang et al., 2023] ✓ ✗ ✗ ✗ ✓
SayPlan [Rana et al., 2023] ✓ ✗ ✗ ✗ ✗
eMEM (ours) ✓ ✓ ✓ ✓ ✓
Table 3: Capability comparison. Spatial : supports 3D coordinate queries. Temporal : observation timestamps as a
first-class query axis. Consol. : background extraction of higher-level summaries. Multi-layer : multiple perception
modalities co-located. Embedded : runs in-process, no external server. ( ∗) Hydra and ConceptGraphs maintain persistent
object instances with update timestamps and therefore capture some temporal dynamics at the scene-graph level; they
do not, however, expose temporal query primitives over an experience stream, which is the contrast we draw here.
Finally, interoception is increasingly recognised as integral to memory rather than peripheral to it. Interoceptive
predictive coding [Seth, 2013, Seth and Critchley, 2013] argues that the brain maintains predictive models of internal
body states and that these states are not metadata but integral to encoding, and the hippocampus itself integrates
interoceptive signals alongside spatial and temporal information [Quigley et al., 2021]. Following this view, and
building on the INTERO framework [Maimon et al., 2025] and the ArmarX memory system [Peller-Konrad et al.,
2023], eMEM treats body state (battery, CPU temperature, joint health) as peer observation layers that flow through the
same ObservationNode →GistNode pipeline as world observations. Co-location with the agent’s pose means that
spatial_query near a steep ramp naturally surfaces both “steep ramp” (VLM) and “rapid battery drain” (interoception),
so that spatial-interoceptive associations emerge from the data without explicit edges.
We stress that these correspondences are motivational rather than literal. eMEM is an engineering system, and the
neuroscience motivation serves to explain why its interface shape is a reasonable one for an embodied agent, not to
make claims about the brain.
7 Related Work
eMEM sits at the intersection of three research threads: LLM-agent memory, embodied spatial representations, and
biological memory models. Table 3 summarises the capability axes on which eMEM differs from prominent prior work.
7.1 LLM-agent memory
Generative Agents [Park et al., 2023] introduced the memory stream with reflections, and eMEM adopts the same core
shape (observations plus consolidation into higher-level summaries); it adds spatial coordinates, multi-layer perception,
and entity tracking, and it provides recency, importance, and relevance as explicit query axes rather than a single
weighted score. MemGPT [Packer et al., 2023] introduced tiered memory with self-directed paging, and eMEM’s
four tiers extend this with automatic consolidation and spatial awareness. MemoryBank [Zhong et al., 2024] applies
the Ebbinghaus forgetting curve, while eMEM implements an analogous decay through tier promotion and archival.
A-MEM [Xu et al., 2025] organises memories as interconnected Zettelkasten notes, and eMEM’s graph achieves similar
connectivity but with spatial and temporal grounding. The production-focused systems Mem0 [Chhikara et al., 2025]
and Zep [Rasmussen et al., 2025] share the graph-based philosophy; eMEM is a production grade framework for
robotics and differentiates from these systems with its spatial index, multi-layer memory model, and embodied focus.
HippoRAG [Jiménez Gutiérrez et al., 2024] is an especially relevant recent addition that makes the hippocampal-CLS
framing explicit in an LLM-RAG setting via personalised PageRank over an open knowledge graph; eMEM shares
the CLS-inspired motivation and the graph-based retrieval surface but targets embodied memory with spatial indexing
and multi-layer perception rather than text-document retrieval. The memory survey of Zhang et al. [2024] provides a
taxonomy onto which eMEM’s architecture maps cleanly.
8

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
7.2 Embodied spatial representations
Hydra [Hughes et al., 2022] builds hierarchical 3D scene graphs in real time from sensor data. eMEM does not replicate
this geometric scene graph; it instead operates at a higher level, storing what the agent remembers about places rather
than the current geometric state. The two systems are therefore complementary, with Hydra providing perception
and eMEM providing memory. ConceptGraphs [Gu et al., 2024] constructs open-vocabulary 3D scene graphs using
foundation models; eMEM’s entity nodes with VLM-generated descriptions are analogous but add temporal tracking
(first_seen ,last_seen , observation counts) and consolidation. SayPlan [Rana et al., 2023] showed that LLMs can
plan over graph-structured spatial representations, and eMEM’s tool interface provides exactly that query substrate.
EmbodiedRAG [Booker et al., 2024] applies retrieval-augmented generation to 3D scene graphs by surfacing task-
relevant subgraphs, and eMEM’s get_current_context andrecall tools implement a similar selective-retrieval
pattern. VLMaps [Huang et al., 2023] fuses CLIP embeddings with spatial reconstructions at the voxel level; eMEM
uses the same idea at the episodic level, enabling temporal reasoning that voxel-grid representations do not support.
7.3 Embodied QA benchmarks
Several benchmarks evaluate aspects of embodied question answering without isolating the memory system.
OpenEQA [Majumdar et al., 2024] presents over 1,600 human-written questions on real-world scenes, but it sends video
frames directly to a VLM and therefore bypasses any structured memory. SQA3D [Ma et al., 2023] tests situated spatial
reasoning from a fixed pose, not memory recall over time. FindingDory [Yadav et al., 2025] does isolate memory from
exploration, but it uses task-oriented navigation scoring on VLMs directly with image sequences. LMEE-Bench [Wang
et al., 2026] couples exploration efficiency with memory recall accuracy. eMEM-Bench complements these bench-
marks by replaying fixed trajectories into a structured memory system and testing six distinct dimensions, including
interoception and entity tracking, that none of the above cover.
Most LLM-agent memory systems lack spatial grounding, and most robotic spatial representations lack temporal
memory, interoception and consolidation. eMEM combines both, and we ground this combination in the neuroscience
framing discussed in §6.
8 Evaluation: eMEM-Bench v1
In order to evaluate eMEM effectively, we created eMEM-Bench v1 , a benchmark constructed over ProcTHOR-
10K [Deitke et al., 2022] scenes that is organised explicitly around eight cognitive-psychology paradigms drawn from
the human-memory literature.
8.1 Why a paradigm based embodied benchmark
Existing agent-memory benchmarks fall into two groups, each with a mismatch to the embodied-agent setting. Lo-
CoMo [Maharana et al., 2024] and similar long-horizon dialogue benchmarks evaluate text-only conversational memory
and do not exercise spatial grounding, multi-modal perception, or interoception; the axes that motivate embodied mem-
ory in the first place. Embodied-QA benchmarks such as OpenEQA [Majumdar et al., 2024] test perception-grounded
question answering in scenes, but the questions are surface-task defined (“what colour is the chair?”) and do not isolate
which memory function produces a correct or incorrect answer.
A category-defined benchmark (“temporal”, “spatial”, “cross-layer”) tells us where a system fails. A paradigm -defined
benchmark tells us which cognitive function is implicated, and makes the result interpretable against a century of
published findings in human memory. eMEM-Bench v1 operationalises eight paradigms, each tied to a specific
empirical phenomenon and a specific class of failure mode that matters for embodied agents over long deployments.
The paradigms were chosen on three criteria: (i)each isolates a distinct memory function with a well-known empirical
signature; (ii)each has a natural embodied-agent analogue, i.e. it is not a contrived translation of a human task; and (iii)
each can be operationalised programmatically over ProcTHOR scenes without per-question human authoring, so that
the benchmark scales.
8.2 Harness
eMEM-Bench v1 uses a replay-based harness. It ingests pre-recorded trajectories into a fresh eMEM instance, drives
a ReAct [Yao et al., 2023] agent against eMEM’s tool surface to answer paradigm probes, and scores predictions
against ground truth with an LLM judge. The ReAct loop is capped at five steps, and a post-processor strips leaked
meta-commentary and maps unanswerable responses to empty strings. Experiments use Qwen 3.6 (27B) [Qwen Team,
9

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
2025] as both agent and entity-extractor, gemma3:27b as the judge (the judge is held constant across all configurations),
andnomic-embed-text-v2-moe [Nussbaum and Duderstadt, 2025] embeddings. No proprietary APIs are in the loop.
8.3 Scenes and trajectories
v1 uses 20 multi-room ProcTHOR-10K houses [Deitke et al., 2022] captured in AI2-THOR [Kolve et al., 2017]. Each
scene is toured along a fixed teleport-and-rotate path that visits every reachable position and rotates 360◦at each
waypoint. Three perception layers are recorded at each view. VLM scene description, object detections, and place
classification, alongside synthetic interoception (battery, CPU temperature). Trajectories are recorded once and replayed
deterministically; different memory configurations differ only in what their memory system surfaces from the same
observation stream. This isolates memory effects from perception variance.
Each paradigm has a fixed schedule : a sequence of IngestPhase /AdvanceClockPhase /ProbePhase operations
that control what the agent ingests, what virtual delay elapses, and what is asked. AdvanceClockPhase can fire
consolidation and archival passes; this lets a paradigm test recall after the memory has been compressed by hours, days,
weeks, or years of simulated time without paying real wall-clock for it.
8.4 The eight paradigms
The paradigms group into three families. We describe each below with its source in human memory research, the
empirical signature it tests, and the embodied-agent failure mode it surfaces.
Every probe in the v1 release was reviewed manually by the authors after programmatic generation; candidates with
ambiguous ground truth, perception-unfriendly items, or inconsistent scene metadata were discarded before the harness
was run.
Encoding and similarity. DRM (Deese-Roediger-McDermott; Roediger and McDermott 1995). Subjects study a list
of associated words ( bed, rest, awake, tired, dream, . . . ) and reliably “remember” a never-studied lure ( sleep ). The
signature is high false-recognition of semantically related lures. Embodied analogue: when a VLM describes a kitchen
with counter, sink, faucet, fridge, kettle , an agent without lure-rejection machinery is liable to report a microwave in
that kitchen. We instantiate the paradigm by selecting four objects unique to one room of a scene, mixing them with a
semantically similar lure absent from the room, and asking the agent to verify each.
Pattern separation (Yassa and Stark 2011; hippocampal CA3/DG). The signature is the ability to encode similar
episodes as distinct traces. Embodied analogue: two different houses visited consecutively have overlapping object
lists; the agent must say “the wine bottle was in House B, not House A” rather than collapsing them. We pair similar
ProcTHOR houses and probe attribution of objects to one house or the other.
Pattern completion (the hippocampal complement to pattern separation). The signature is recovering a complete memory
from a partial cue. Embodied analogue: an operator says “the mug” and the agent must recover “the blue mug on the
kitchen counter at 14:32”. Almost every real-world retrieval uses partial cues; this paradigm tests whether the agent can
resolve them to specific encoded items.
Source and context. Source monitoring (Johnson et al. 1993). The signature is the ability to attribute a memory to its
source — externally perceived vs. internally generated, modality A vs. B, which speaker said it. Embodied analogue:
“Did you learn about the bread from the VLM scene description or from the object detector?”. This is the embodied
analogue of reality monitoring; failure means an agent cannot tell its detector hallucinations from its perceptions, with
safety consequences.
Context-dependent retrieval (Godden and Baddeley 1975). The Godden-Baddeley signature is that recall is best in the
context of encoding (divers recall items learned underwater better when tested underwater). Embodied analogue: “Is
there a chair here?” must condition on the agent’s current position, not return any chair from anywhere in the scene.
This is the most common retrieval pattern for an operating embodied agent, and an explicit test that the system surfaces
spatial co-location as a first-class retrieval axis.
Temporal and interference. Long-horizon interference. Classical memory research distinguishes proactive interfer-
ence (earlier material hurts later) from retroactive (later material hurts earlier). Embodied analogue: two house tours
separated by a simulated day-long gap; the agent must answer questions about House Alpha after ingesting House Beta.
This is the canonical multi-day-deployment test: does yesterday’s patrol survive today’s overlay?
Serial position (Murdock 1962). The signature is the U-shaped recall function over a sequence: primacy (early items
recalled well) and recency (late items recalled well), with a depressed middle. Embodied analogue: a chain of multiple
10

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Paradigm Family N Score Perfect % Tool Acc
DRM [Roediger and McDermott, 1995] Encoding / similarity 60 86.7 86.7 0.67
Pattern separation [Yassa and Stark, 2011] Encoding / similarity 53 71.7 71.7 0.52
Pattern completion Encoding / similarity 73 79.5 75.3 0.43
Source monitoring [Johnson et al., 1993] Source / context 70 78.6 71.4 0.53
Context-dep. retrieval [Godden and Baddeley, 1975] Source / context 136 63.8 61.8 0.30
Long-horizon interference Temporal / interference 62 69.4 69.4 0.48
Serial position [Murdock, 1962] Temporal / interference 54 69.4 66.7 0.46
Retention curve (foils) [Ebbinghaus, 1885] Temporal / interference 480 89.1 87.1 0.68
Overall (weighted) 988 80.8 78.5 0.56
Table 4: eMEM-Bench v1 results across the eight cognitive paradigms retained for evaluation, over 20 multi-room
ProcTHOR-10K scenes [Deitke et al., 2022]. Scored 1–5 by an LLM judge (gemma3:27b), normalised to 0–100.
“Perfect %” is the share of probes scored 5. “Tool Acc” is the fraction of probes for which the agent invoked at least one
tool the paradigm marks as expected.
house tours; the agent is asked when in the chain a given object first appeared (early/middle/late). This tests whether the
memory preserves temporal ordering across many ingested episodes.
Retention curve (Ebbinghaus 1885). The classical forgetting curve: recall as a function of delay since encoding.
We probe at six delays {1h,1d,1wk,1mo,6mo,1yr}of simulated time, firing consolidation/archival at each step so
that the probe at +1yrsees memory state after a year of the pipeline’s pressure, not after raw ingestion. This is the
characterisation for a long-running deployment.
8.5 Foil methodology
The retention curve uses 50/50 yes/no probes; foils are drawn from a cross-scene absent pool — objects that appear (and
were confirmed unique to the room) in some other v1 scene, but are absent from this scene’s ProcTHOR ground-truth
inventory. The pool guarantees foil items are concrete, perception-friendly, and unambiguously absent. Without foils, a
yes/no recall paradigm collapses into a test of yes-bias. Balancing the polarity restores signal-detection discriminability
(hit rate vs. false-alarm rate per delay bucket; Table 5) and converts the paradigm into a test of recall.
We propose foil-augmentation as a general methodological discipline for embodied-agent benchmarks: any yes/no
probe should be balanced against an absent-foil pool, and any free-text probe whose ground truth is fully recoverable
from a single stored observation should be re-engineered so the agent must verify perception, not just retrieve text.
8.6 Results
Table 4 reports the eight paradigms over 988 probes, with an overall weighted mean of 80.8. The pattern across families
is informative. Encoding and similarity paradigms cluster between 71 and 87, with DRM at the top: the consolidation
pipeline produces gists that suppress associated-lure activation, the cognitive signature the paradigm was designed to
detect. Source monitoring scores 78.6, with eMEM’s per-layer tagging making the candidate layers directly addressable
at retrieval time — the embodied analogue of source attribution succeeding for the same reason it does in humans,
namely that the relevant distinction is encoded at storage rather than reconstructed at query. Pattern separation and
pattern completion sit at 72 and 79, indicating that the explicit observation/episode/gist taxonomy gives the agent the
right material to discriminate similar scenes and to complete partial cues. Context-dependent retrieval scores 63.8; here
the limiting factor is the agent’s tool selection under a spatial constraint rather than the memory representation, which
surfaces a tractable agent-side next step. Temporal and interference paradigms range from 69 to 89, with the retention
curve at the top.
Table 5 resolves the retention-curve aggregate into per-delay hit and false-alarm rates. The result is a flat curve at ceiling
on seen items from 1 h to 1 yr of simulated delay: eMEM exhibits no measurable forgetting on room-unique items
across a year of virtual time, consistent with the design claim that semantic clustering combined with spatial co-location
stabilises salient items against consolidation and archival pressure. False alarms hold steady at ∼15.5%across delays
and are item-level — traceable to a small set of near-synonym confusions in the entity-extraction pipeline (“key chain”
confused with “keys”) rather than to time-dependent forgetting — placing the system at the high-hit, low-miss corner of
the standard signal-detection picture for a memory under long delays.
11

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Delay Hit rate Correct-reject False-alarm N
1 h 100.0% 76.3% 15.8% 77
1 d 100.0% 79.5% 15.4% 78
1 wk 100.0% 79.5% 15.4% 78
1 mo 100.0% 79.5% 15.4% 78
6 mo 100.0% 79.5% 15.4% 78
1 yr 100.0% 79.5% 15.4% 78
Table 5: eMEM retention over six delay buckets. Hit rate is P(yes|seen)on probes asking “Did you see the X?”
where Xwas perceived during the trajectory. Correct-reject is the analogue for foil probes where Xwas absent from the
scene. Verdicts are derived from log parsing; per-delay totals differ slightly because ambiguous predictions are reported
separately. The curve is flat at ceiling on seen items from 1 h to 1 yr of simulated delay: no measurable forgetting on
room-unique items across the full range. The ∼15.5% false-alarm rate is constant across delays and is item-level (six
foils are consistently confusable), not time-dependent.
8.7 Ablations
Will be added soon.
Reading the numbers. The paradigm framing makes the 80.8 weighted mean directly interpretable against published
findings in human memory and in prior agent-memory systems. On each paradigm eMEM lands on the productive side
of the diagnostic the paradigm was designed to expose: lure rejection on DRM, source attribution on source monitoring,
scene-level discrimination on pattern separation, time-invariant recall on the retention curve, and instruction retention
plus correct ordering on long-horizon interference and serial position. Where the score is lower, the paradigm itself
points to a concrete next step — context-dependent retrieval, for instance, implicates the agent’s spatial-tool selection
more than the memory representation — rather than to a structural gap. This level of diagnostic resolution is what
surface-task benchmarks cannot deliver, and is the reason for organising the evaluation around paradigms in the first
place. It is pertinent to remember that this evaluation has been performed with open source small models that can run
on a consumer GPU. Both query formulation and tool selection are significantly impacted by model quality, making it
non-trivial to measure performance of the memory system in islation.
9 Limitations and Future Work
Agent-side improvements. A meaningful fraction of the headroom on eMEM-Bench v1 lives in the agent rather than
in the memory primitive. As noted earlier in the §8 section, the evaluation uses open-source models small enough to fit
on a consumer GPU; query formulation, tool selection, and multi-step composition all scale with model quality, so
improvements there should lift scores across the board without changing the memory architecture. Context-dependent
retrieval (63.8) is the clearest case: the limit is the agent’s ability to select spatial tools under a spatial constraint, not
what the memory exposes. The most common low-score pattern more generally is semantic-search phrasing failing to
match an observation’s embedding ( “when did I last see the stool?” fails when the VLM described the object as “bar
seating”), with the symmetric near-synonym confusions (“key chain” vs. “keys”) visible in the retention-curve false
alarms. Scaling the agent, fine-tuning a small model specifically for memory tool use, embeddings tuned for embodied
language, query expansion, and hybrid dense-plus-lexical retrieval (BM25 combined with HNSW) are all orthogonal to
the eMEM architecture and would compose with it.
Richer consolidation. The current pipeline weighs observations uniformly within a time window, so an agent that
spent 20 minutes in a kitchen and 30 seconds glimpsing a bathroom produces a gist that mentions only the kitchen.
Biologically, consolidation is modulated by novelty and reward [Lee and Jung, 2024]; adaptive weighting that favours
transitions and surprise would help briefly visited areas. Relatedly, the recall tool aggregates observations by
proximity but does not explicitly cross-reference perception layers; a dedicated cross-layer synthesis step at each
location would lift descriptive and episodic queries.
Object-frame coordinates. A separable improvement is to bring object-frame coordinates into the spatial axis.
eMEM records observation positions; entity tracking provides centroid positions over multiple sightings, but single-
observation queries still report the robot’s pose. Depth estimation or object-relative positioning, integrated through the
perception stack rather than the memory, would close this gap and directly raise context-dependent retrieval. This sits at
the boundary of memory and perception; the natural composition is for the perception stack to publish object-frame
coordinates as another layer that eMEM ingests.
12

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Latent representations and schema transfer. eMEM is symbolic: it stores what happened and what is known,
as post-perception linguistic descriptions, not raw sensory data, latent representations, or procedural knowledge. A
direction worth pursuing is the latent-graph tradition in computational neuroscience [George et al., 2021, Raju et al.,
2024], in which a cloned hidden Markov model trained on action-conditioned sequences recovers phenomena similar to
place and splitter-cells, from latent inference alone, and supports schema transfer by freezing the latent transition graph
as a content-free topology and rebinding only the observation slots in a new environment.
The most promising direction for future work will be to stack the two layers such that the observable spatio-temporal
graph continues to serve as the language-facing surface that the LLM queries; a latent-state layer underneath assigns
each observation to a clone. This should allow automatic disambiguation of perceptually similar contexts — two
visually similar kitchens in different houses becoming distinct latent states without manual scene tagging — and
schema-based generalisation across environments. Concretely, a robot deployed to a new house could carry over what it
has learned about kitchen topology and rebind only the visual identities, rather than starting from scratch each time.
In this way a long-running embodied agent should accumulate transferable structure across the environments it visits.
From the current results, this extension should improve performance on pattern separation, context-dependent retrieval
and the retention-curve false alarms.
References
Marcus K. Benna and Stefano Fusi. Place cells may simply be memory cells: Memory compression leads to spatial
tuning and history dependence. Proceedings of the National Academy of Sciences (PNAS) , 118(51), 2021.
Matthew Booker, Gregory Byrd, Brendan Kemp, Adam Schmidt, and Christopher Rivera. EmbodiedRAG: Dynamic
3D scene graph retrieval for efficient and scalable robot task planning, 2024.
Jan Born and Ines Wilhelm. System consolidation of memory during sleep. Psychological Research , 76:192–203, 2012.
Prateek Chhikara, Dev Khant, et al. Mem0: Building production-ready AI agents with scalable long-term memory,
2025.
Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric Kolve, Ali
Farhadi, Aniruddha Kembhavi, and Roozbeh Mottaghi. ProcTHOR: Large-Scale Embodied AI Using Procedural
Generation. In Advances in Neural Information Processing Systems (NeurIPS) , 2022. URL https://procthor.
allenai.org/ .
Susanne Diekelmann and Jan Born. The memory function of sleep. Nature Reviews Neuroscience , 11:114–126, 2010.
Hermann Ebbinghaus. Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie . Duncker & Humblot,
Leipzig, 1885. English translation: Memory: A Contribution to Experimental Psychology , 1913.
Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. A density-based algorithm for discovering clusters in
large spatial databases with noise. In Proceedings of the Second International Conference on Knowledge Discovery
and Data Mining (KDD) , pages 226–231. AAAI Press, 1996.
Dileep George, Rajeev V . Rikhye, Nishad Gothoskar, J. Swaroop Guntupalli, Antoine Dedieu, and Miguel Lázaro-
Gredilla. Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps.
Nature Communications , 12:2392, 2021. doi: 10.1038/s41467-021-22559-5.
Duncan R. Godden and Alan D. Baddeley. Context-dependent memory in two natural environments: On land and
underwater. British Journal of Psychology , 66(3):325–331, 1975.
Qiao Gu, Ali Kuwajerwala, et al. ConceptGraphs: Open-vocabulary 3D scene graphs for perception and planning. In
IEEE International Conference on Robotics and Automation (ICRA) , 2024.
Torkel Hafting, Marianne Fyhn, Sturla Molden, May-Britt Moser, and Edvard I. Moser. Microstructure of a spatial map
in the entorhinal cortex. Nature , 436:801–806, 2005.
Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In IEEE
International Conference on Robotics and Automation (ICRA) , 2023.
Nathan Hughes, Yun Chang, and Luca Carlone. Hydra: A real-time spatial perception system for 3D scene graph
construction and optimization. In Robotics: Science and Systems (RSS) , 2022.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG: Neurobiologically inspired
long-term memory for large language models. In Advances in Neural Information Processing Systems (NeurIPS) ,
2024.
Marcia K. Johnson, Shahin Hashtroudi, and D. Stephen Lindsay. Source monitoring. Psychological Bulletin , 114(1):
3–28, 1993.
13

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu,
Abhinav Gupta, and Ali Farhadi. AI2-THOR: An interactive 3D environment for visual AI, 2017.
Dharshan Kumaran, Demis Hassabis, and James L. McClelland. What learning systems do intelligent agents need?
complementary learning systems theory updated. Trends in Cognitive Sciences , 20(7):512–534, 2016.
Jee Hang Lee and Min Whan Jung. Memory consolidation from a reinforcement learning perspective. Frontiers in
Computational Neuroscience , 18:1538741, 2024.
Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. SQA3D: Situated
question answering in 3D scenes. In International Conference on Learning Representations (ICLR) , 2023.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating
very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (ACL) , 2024.
Asaf Maimon, Ido Wald, Mihai Pomarlan, Sen Zhang, Daniel Beßler, Robert Nolte, Dennis Küster, Robert Porzel,
Tanja Schultz, and Rainer Malaka. INTERO: A model of robotic interoceptive sensing. In RobOntics Workshop,
CEUR-WS Vol-4169 , 2025.
Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul
Mcvay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent
Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris
Paxton, Alexander Sax, and Aravind Rajeswaran. OpenEQA: Embodied question answering in the era of foundation
models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 2024.
Yu. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence , 42(4):824–836,
2020.
James L. McClelland, Bruce L. McNaughton, and Randall C. O’Reilly. Why there are complementary learning systems
in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and
memory. Psychological Review , 102(3):419–457, 1995.
Bennet B. Murdock. The serial position effect of free recall. Journal of Experimental Psychology , 64(5):482–488, 1962.
Karim Nader, Glenn E. Schafe, and Joseph E. LeDoux. Fear memories require protein synthesis in the amygdala for
reconsolidation after retrieval. Nature , 406:722–726, 2000.
Zach Nussbaum and Brandon Duderstadt. Training sparse mixture of experts text embedding models, 2025.
John O’Keefe and Jonathan Dostrovsky. The hippocampus as a spatial map: Preliminary evidence from unit activity in
the freely-moving rat. Brain Research , 34(1):171–175, 1971.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT:
Towards LLMs as operating systems, 2023.
Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein.
Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on
User Interface Software and Technology (UIST) , 2023.
Fabian Peller-Konrad, Rainer Kartmann, Christian R. G. Dreher, Andre Meixner, Fabian Reister, Markus Grotz, and
Tamim Asfour. A memory system of a robot cognitive architecture and its implementation in ArmarX. Robotics and
Autonomous Systems , 164:104415, 2023.
Karen S. Quigley, Scott Kanoski, Warren M. Grill, Lisa Feldman Barrett, and Manos Tsakiris. Functions of interoception:
From energy regulation to experience of the self. Trends in Neurosciences , 44(1):29–38, 2021.
Qwen Team. Qwen3 technical report, 2025.
Rajkumar Vasudeva Raju, J. Swaroop Guntupalli, Guangyao Zhou, Carter Wendelken, Miguel Lázaro-Gredilla, and
Dileep George. Space is a latent sequence: A theory of the hippocampus. Science Advances , 10(31):eadm8470, 2024.
doi: 10.1126/sciadv.adm8470.
Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. SayPlan: Grounding
large language models using 3D scene graphs for scalable robot task planning. In Conference on Robot Learning
(CoRL) , volume 229 of Proceedings of Machine Learning Research , 2023.
Preston Rasmussen et al. Zep: A temporal knowledge graph architecture for agent memory, 2025.
Sonia Raychaudhuri and Angel X. Chang. Semantic mapping in indoor embodied AI: A survey on advances, challenges,
and future directions. Transactions on Machine Learning Research (TMLR) , 2025.
14

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International
Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 3982–3992, 2019.
Henry L. Roediger and Kathleen B. McDermott. Creating false memories: Remembering words not presented in lists.
Journal of Experimental Psychology: Learning, Memory, and Cognition , 21(4):803–814, 1995.
Anil K. Seth. Interoceptive inference, emotion, and the embodied self. Trends in Cognitive Sciences , 17(11):565–573,
2013.
Anil K. Seth and Hugo D. Critchley. Extending predictive processing to the body: Emotion as interoceptive inference.
Behavioral and Brain Sciences , 36(3):227–228, 2013.
Endel Tulving. Episodic and semantic memory. In Endel Tulving and Wayne Donaldson, editors, Organization of
Memory , pages 381–403. Academic Press, 1972.
Endel Tulving. Elements of Episodic Memory . Oxford University Press, 1983.
Endel Tulving and Donald M. Thomson. Encoding specificity and retrieval processes in episodic memory. Psychological
Review , 80(5):352–373, 1973.
Shuo Wang et al. Explore with long-term memory: A benchmark and multimodal LLM-based reinforcement learning
framework for embodied exploration, 2026.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM
agents. In Advances in Neural Information Processing Systems (NeurIPS) , 2025.
Karmesh Yadav, Yusuf Ali, Gunshi Gupta, Yarin Gal, and Zsolt Kira. FindingDory: A benchmark to evaluate memory
in embodied agents, 2025.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing
reasoning and acting in language models. In International Conference on Learning Representations (ICLR) , 2023.
Michael A. Yassa and Craig E. L. Stark. Pattern separation in the hippocampus. Trends in Neurosciences , 34(10):
515–525, 2011.
Zeyu Zhang et al. A survey on the memory mechanism of large language model-based agents. ACM Transactions on
Information Systems , 2024.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. MemoryBank: Enhancing large language models
with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence , 2024.
15

eMEM: A Hybrid Spatio-Temporal Memory System for Embodied Agents
A Additional Limitations and Future Directions
Temporal reasoning is the weakest category on both benchmarks. The agent sees timestamps in tool output but struggles
to use them for ordering, duration estimation, or first/last classification. Dedicated temporal tools (e.g. a timeline view
that renders the sequence of visits to a location, or explicit first_seen andlast_seen primitives) could remove the
need for the agent to perform temporal reasoning itself.
eMEM does not support shared memory across multiple agents. Multi-agent spatial memory with conflict resolution and
perspective-taking is an open challenge for collaborative robotics and is a natural direction for future work. Closer to
perception, eMEM stores point coordinates rather than meshes, voxels, or occupancy grids; integration with geometric
systems such as Hydra [Hughes et al., 2022] or ConceptGraphs [Gu et al., 2024] would provide richer spatial reasoning
without changing the memory architecture, since the geometric representations would simply become another perception
layer.
A timing consideration for entity extraction: the two entity-extraction triggers ( entity_extract_flush_interval ,
default 10, and entity_extract_time_interval , default 60 s) interact to bound the latency between an observation
being added and the corresponding entity becoming available via entity_query . With default flush_batch_size = 5
and a steady observation rate r, the flush-interval trigger dominates at high r(entities are extracted every 5×10 = 50
observations) and the time trigger dominates at low r(every 60 s regardless of buffer fill). Worst-case availability
latency for a newly observed entity is therefore bounded below by min(50 /r,60 s) . Deployments needing lower-latency
entity availability can reduce both intervals at the cost of more frequent LLM calls.
Finally, eMEM-Bench v1 evaluates on 20 multi-room ProcTHOR-10K houses with programmatically generated,
manually reviewed questions. Extending to real-world robot trajectories, more diverse environments, and naturalistic
human-authored questions would improve ecological validity.
B Implementation Details
Storage in eMEM uses only embedded libraries: SQLite (stdlib), hnswlib for approximate nearest-neighbour search,
rtree for 3D spatial queries, scikit-learn DBSCAN for time-window consolidation, and NumPy for vector opera-
tions. Embeddings are produced through a pluggable EmbeddingProvider protocol rather than a fixed model: any back-
end that maps text to vectors can be supplied. The experiments in this paper use nomic-embed-text-v2-moe [Nuss-
baum and Duderstadt, 2025] (768-dimensional) served locally over Ollama; an in-process sentence-transformers
backend (SBERT [Reimers and Gurevych, 2019]) is also provided for deployments that prefer to avoid a separate
embedding service. A complete eMEM instance persists as two files: a SQLite database and an HNSW binary index.
The code is organised into a small number of modules: types.py (node and edge types, tiers), config.py (tunable
parameters), embeddings.py (theEmbeddingProvider protocol and implementations), spatial.py (the R-tree
wrapper), working_memory.py (the in-process buffer with auto-flush and entity callbacks), store.py (hybrid persis-
tence over SQLite, HNSW, and the R-tree), consolidation.py (temporal chunking, gist generation, entity extraction,
and archival), tools.py (the 10 LLM-facing tools), and memory.py (theSpatioTemporalMemory facade). The
default configuration parameters used throughout the paper are listed in Table 6.
Parameter Default Meaning
embedding_dim 384 Embedding vector dimension (auto-set by the provider)
working_memory_size 50 Max buffered observations
flush_batch_size 5 Batch-size auto-flush threshold
flush_interval 2.0 s Time-based auto-flush threshold
consolidation_window 1800 s Temporal gap threshold (episode chunking & time-window)
consolidation_spatial_eps 3.0 m DBSCAN clustering radius
consolidation_min_samples 3 Min. cluster size for spatial consolidation
archive_after_seconds 3600 s Time in long-term before archival
entity_extract_flush_interval 10 Extract entities every Nflushes
entity_extract_time_interval 60.0 s Extract entities every Nseconds
entity_similarity_threshold 0.85 Cosine similarity for entity merge
entity_spatial_radius 5.0 m Spatial proximity for entity merge
recency_weight 0.0 Recency-weighting α(0 disables)
recency_halflife 3600 s Time constant for recency decay
Table 6: eMEM default configuration parameters.
16