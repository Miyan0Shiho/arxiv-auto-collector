# UrbanClipAtlas: A Visual Analytics Framework for Event and Scene Retrieval in Urban Videos

**Authors**: Joel Perca, Luis Sante, Juanpablo Heredia, Joao Rulff, Claudio Silva, Jorge Poco

**Published**: 2026-04-16 17:03:11

**PDF URL**: [https://arxiv.org/pdf/2604.15225v1](https://arxiv.org/pdf/2604.15225v1)

## Abstract
Extracting actionable insights from long-duration urban videos is often labor-intensive: analysts must manually sift through raw footage to pinpoint target events or uncover broader behavioral trends. In this work, we present URBANCLIPATLAS, a visual analytics system for exploring long urban videos recorded at street intersections. URBANCLIPATLAS combines retrieval-augmented generation (RAG), taxonomy-aware entity extraction, and video grounding to support event retrieval and interpretation. The system segments extended recordings into short clips, generates textual descriptions with a vision-language model, and indexes them for semantic retrieval. A knowledge graph maps entities and relations from LLM answers onto a domain-specific taxonomy and aligns them with detected objects and trajectories to support visual grounding and verification. URBANCLIPATLAS supports scene retrieval through an augmented chat-based interface and improves scene interpretation by tightly aligning textual outputs with video evidence. This design strengthens the connection between textual reasoning and visual evidence, reducing the effort required to validate model outputs and refine hypotheses. We demonstrate the usefulness of URBANCLIPATLAS on the StreetAware dataset through two case studies involving hazardous scenarios and crossing dynamics at street intersections. URBANCLIPATLAS helps analysts reason about safety- and mobility-related patterns across large urban video collections.

## Full Text


<!-- PDF content starts -->

Preprint
(17 April 2026)
URBANCLIPATLAS: A Visual Analytics Framework for Event and
Scene Retrieval in Urban Videos
Joel Perca1
, Luis Sante1
, Juanpablo Heredia1
, Joao Rulff1,2
, Claudio Silva2
and Jorge Poco1
1Fundação Getulio Vargas, Brazil
2New York University, USA
Figure 1: URBANCLIPATLASinterface.(A) TheChat Paneldisplays the user’s query, the RAG-generated narrative answer, and entity-
level tooltips linked to the knowledge graph. (B) TheVideo Player with tracking overlaysshows the current frame with dynamic entities
and highlighted static layout elements. (C) TheRelated Clips Timelinesummarizes retrieved clips across videos, with cells encoded by their
semantic relevance to the query.
Abstract
Extracting actionable insights from long-duration urban videos is often labor-intensive: analysts must manually sift through
raw footage to pinpoint target events or uncover broader behavioral trends. In this work, we presentURBANCLIPATLAS, a vi-
sual analytics system for exploring long urban videos recorded at street intersections.URBANCLIPATLAScombines retrieval-
augmented generation (RAG), taxonomy-aware entity extraction, and video grounding to support event retrieval and inter-
pretation. The system segments extended recordings into short clips, generates textual descriptions with a vision–language
model, and indexes them for semantic retrieval. A knowledge graph maps entities and relations from LLM answers onto a
domain-specific taxonomy and aligns them with detected objects and trajectories to support visual grounding and verification.
URBANCLIPATLASsupports scene retrieval through an augmented chat-based interface and improves scene interpretation
by tightly aligning textual outputs with video evidence. This design strengthens the connection between textual reasoning and
visual evidence, reducing the effort required to validate model outputs and refine hypotheses. We demonstrate the usefulness of
URBANCLIPATLASon the StreetAware dataset through two case studies involving hazardous scenarios and crossing dynamics
at street intersections.URBANCLIPATLAShelps analysts reason about safety- and mobility-related patterns across large urban
video collections.
CCS Concepts
•Human-centered computing→Visualization; Visualization systems and tools;
© 17 April 2026 The Author(s).arXiv:2604.15225v1  [cs.HC]  16 Apr 2026

2 of 12J. Perca et al. /URBANCLIPATLAS
1. Introduction
Understanding the multifaceted dynamics of urban environments is
fundamental to fostering resilient, safe, and equitable cities. Street
intersections are key nodes where diverse traffic flows converge,
occupying only a small fraction of the street network yet play-
ing a disproportionate role in safety and efficiency. The National
Association of City Transportation Officials (NACTO) calls them
“one of the most critical and most complicated elements in road-
way design,” and the US Department of Transportation (USDOT)
reports that intersections account for a substantial share of traffic
fatalities, with economic and societal costs reaching billions of dol-
lars [Fed24, BMZ∗15].
To optimize these locations, urban planners and transporta-
tion engineers rely on geometric and operational interven-
tions—such as curb extensions, raised medians, and signal timing
changes [SMMSL20]. Yet the intended effect of a design often dif-
fers from its real-world outcome. Intersections are not just static in-
frastructure but dynamic environments shaped by human behavior,
where pedestrians, cyclists, and vehicles negotiate shared space.
Assessing safety and efficiency thus requires granular observations
of how these agents actually move and interact.
New sensing infrastructures now enable extended recording of
intersection activity. Unlike earlier approaches based on aerial im-
agery [CZYW17,BHM∗25] or GPS traces [DHZ∗18,AH20], mod-
ern systems capture rich, high-definition video directly from street
level [PRB∗23]. This street-view perspective provides nuanced
information about vulnerable road users, fine-grained behavioral
cues, and subtle interaction dynamics [RPH∗24].
Despite these advances, domain experts still lack scalable tools
to retrieve and analyze specific events without laborious man-
ual inspection. Existing automated approaches typically rely on
heavily supervised models and a limited vocabulary of predefined
events [ASM∗21, AE24], which restricts exploratory analysis and
rapid adaptation to new policy questions. Still, new graph-LLM-
based solutions are promising [HERMCP25]. Vision–language
models (VLMs) offer a promising alternative: their zero-shot capa-
bilities and natural-language interfaces allow analysts to describe
scenarios in plain text and retrieve matching video segments. How-
ever, interpreting VLM outputs in complex urban scenes remains
challenging. Analysts must relate long textual descriptions to vi-
sual counterparts involving many entities and intricate spatial rela-
tionships, a cognitively demanding process that is amplified when
working with long-duration recordings.
There is thus a need for visual analytics techniques that struc-
ture and enrich VLM outputs, connect them to higher-level seman-
tic models of intersection events, and ground these semantics in
video to support sensemaking. To address this need, we introduce
URBANCLIPATLAS, a visual analytics framework for event and
scene retrieval in long-duration urban intersection videos. URBAN-
CLIPATLAScombines a retrieval-augmented generation (RAG) ar-
chitecture with taxonomy-aware entity extraction and smart video
analysis to support both event retrieval and interpretation. The sys-
tem segments long recordings into clips, generates textual descrip-
tions with a VLM, and indexes them for semantic retrieval. At
query time, natural-language questions are enriched to better match
the indexed descriptions; relevant clips are retrieved; and entitiesmentioned in language-model responses are extracted and orga-
nized into a knowledge graph guided by a domain-specific tax-
onomy of intersection events. Object detection and multi-object
tracking then align these entities with visual instances, provid-
ing grounding that links semantic descriptions to video evidence.
In contrast to existing video RAG systems that operate mainly at
the text-index level, our approach integrates retrieval, taxonomy-
guided entity alignment, and video grounding within an interactive
visual analytics framework. This framework supports output-level
auditing of black-box models through an augmented narrative view
that presents model-generated answers alongside supporting video
snippets, highlighted entities, and trajectories, reducing the effort
required to interpret model outputs. In summary, this work makes
the following contributions:
•RAG-based video processing pipeline.We propose a video
RAG pipeline that segments long-duration recordings into clips,
generates textual descriptions with a vision–language model, in-
dexes them for semantic search, and performs prompt enrich-
ment at query time to support interactive identification of salient
video segments.
•Taxonomy-aware entity alignment and visual grounding.We
introduce a knowledge-graph augmentation strategy that extracts
entities from VLM responses, aligns them with detected visual
objects, and organizes them according to a domain-specific tax-
onomy of urban intersection events, enabling consistent text–
video grounding and higher-level reasoning.
•Visual analytics system and evaluation.We implement UR-
BANCLIPATLAS, an open-source visual analytics system that in-
tegrates these components into an augmented-narrative interface,
bridging textual descriptions with visual evidence and support-
ing human-in-the-loop verification of model outputs. We demon-
strate its usefulness through two case studies on the StreetAware
dataset and report insights from structured interviews with urban
specialists on usability and deployment potential.
2. Related Work
Our work lies at the intersection of RAG, video understanding, and
visual analytics for urban monitoring.
Retrieval-Augmented Generation for Video Understanding.
RAG enhances the factual accuracy and knowledge scope of large
language models (LLMs) by incorporating external knowledge
sources [JKBH25]. Extending RAG to video introduces additional
challenges, including the cost of converting long videos into text
and the loss of multimodal cues such as motion and spatial con-
text when relying solely on textual representations [RXX∗25].
These issues become particularly pronounced for long-duration
corpora, where purely textual indexing often fails to preserve
event-level structure [ADUC24,RXX∗25]. To address these limita-
tions, several systems adopt incremental pipelines that delay expen-
sive processing until necessary. Some enrich representations only
for retrieved segments [ADUC24], while others maintain evolv-
ing knowledge structures to support low-latency retrieval as new
content arrives [SRC25]. Graph-based indices and hierarchical re-
trieval strategies [GXY∗25], as well as cascaded pipelines combin-
ing lightweight models with VLM refinement [ADSUC24], further
reduce computational costs.
© 17 April 2026 The Author(s).

J. Perca et al. /URBANCLIPATLAS3 of 12
Another line of work strengthens retrieval by preserving mul-
timodal signals rather than relying solely on text. Approaches
augment VLM inputs with auxiliary channels such as OCR,
ASR, or object detections [LZY∗24], apply question-aware frame
sampling [TYL∗25], or jointly reason across video, audio, and
text [MPCK∗25]. While these techniques improve retrieval rele-
vance, they still struggle to capture long-range temporal depen-
dencies. More recent approaches construct structured representa-
tions across multiple videos, treating collections as interconnected
knowledge spaces. Graph-based grounding and multimodal en-
coders enable reasoning across entities and events at the corpus
level [RXX∗25, JKBH25, ADC24]. However, these strategies in-
troduce challenges related to entity canonicalization, graph main-
tenance, and multimodal alignment at scale.
Retrieval-oriented approaches are particularly promising for ur-
ban analytics, where scenes evolve rapidly and contextual ground-
ing is essential [ADC24, MPCK∗25]. In contrast to these general-
purpose retrieval methods, URBANCLIPATLASincorporates a
domain-specific taxonomy and an explicit grounding loop to en-
sure that retrieved narratives remain aligned with the semantic re-
quirements of urban safety analysis and can be audited against the
underlying video evidence.
Object Detection and Tracking for Grounding.Grounding lan-
guage in video requires linking textual descriptions to entities
in space and time through detection, tracking, and spatial rea-
soning. Conventional closed-vocabulary detectors often struggle
in open environments and fail to maintain consistent identities
when objects are occluded or change appearance [JCL∗24]. These
issues are amplified in dense urban scenes involving heteroge-
neous agents. Open-vocabulary detection addresses this limita-
tion by aligning visual features with textual embeddings. YOLO-
World [CSG∗24] enables real-time detection of unseen categories,
while OV-STAD [WGQ∗24] combines open-vocabulary detection
with action understanding. Despite these advances, maintaining
temporally stable identities across frames remains challenging.
Recent work also emphasizes spatial reasoning. Jiang et
al. [JCL∗24] propose a pipeline that links entities, attributes, and
relations across frames, supporting more temporally consistent
alignments between visual content and generated text. Spatial-
Bot [CPY∗25] leverages a VLM to reason about distances, di-
rections, and geometric relations, enabling explicit spatial under-
standing. Together, these systems point toward grounding models
that can both identify objects and situate them in their environ-
ment. These capabilities are vital for interpretable, context-aware
video analysis, but they do not by themselves constrain how VLM
or LLM outputs are used. In URBANCLIPATLAS, we close this
loop by combining detection, multi-object tracking, and a special-
ized taxonomy into a grounding layer that explicitly checks model-
generated claims against concrete visual evidence in urban inter-
section videos, exposing mismatches to the analyst.
Visual Analytics for Video Data.Visual video analytics integrates
computer vision with interactive visualization to support human
reasoning over complex video data [SH19]. Prior systems demon-
strate the value of linking automated analysis to visual interfaces
that enable analysts to inspect and guide algorithmic results. Sev-
eral systems combine visual features with textual information. Wuand Qu [WQ20] explore storytelling patterns in TED talks using
transcripts and video features, while Chen et al. [CBH∗20] inte-
grate trajectory analysis with spatial views to study traffic behavior.
Motion Browser [CNC∗20] similarly employs coordinated views
to analyze motion patterns in clinical data. Recent visual analytics
systems have also explored interactive analysis of spatiotemporal
phenomena through coordinated structural and temporal views, in-
cluding association-rule-based exploration of categorical data and
visual analysis of evolving geographic regions [DSP∗25, NDP25].
Although these systems effectively expose spatiotemporal pat-
terns, they typically rely on predefined features and offer limited
support for natural-language queries and generative explanations.
Moreover, high-level descriptions are rarely grounded to concrete
entities in the video. To address these gaps, URBANCLIPATLASin-
troduces a visual analytics framework that couples a RAG-based
video pipeline with taxonomy-aware entity alignment and explicit
grounding. By anchoring model-generated explanations directly to
retrieved clips and verifiable entities in the scene, the augmented
narrative interface turns otherwise opaque VLM/LLM outputs into
auditable, evidence-linked insights for urban safety analysis.
3. System Overview
Prior work on RAG-based video retrieval and visual video analytics
has highlighted persistent challenges in (1) semantically searching
long-duration videos, (2) exposing entity-level spatiotemporal con-
text, and (3) supporting explanation and validation of algorithmic
outputs directly in video frames (e.g., [RXX∗25, ADC24, SH19,
WQ20, CBH∗20]). Guided by these observations and formative
discussions with urban traffic analysts, we derive a set of design
goals and corresponding technical tasks for URBANCLIPATLAS.
We then describe how these goals are operationalized through the
three-stage workflow illustrated in Fig. 2.
3.1. Design Goals
We conceptualize URBANCLIPATLASas a visual analytics system
that sits between raw video streams and domain experts’ questions
about safety and behavior at intersections. The following design
goals capture the capabilities needed to analyze large collections of
urban videos.
G1 Text-based semantic access to long-duration videos.Enable
analysts to formulate information needs in natural language and
retrieve semantically relevant video segments, avoiding manual
inspection of hours of footage.
G2 Entity-centric spatiotemporal context.Provide representa-
tions that expose the locations of entities (e.g., vehicles and
pedestrians) in the scene and how their states and interactions
evolve over time, supporting reasoning about trajectories, con-
flicts, and usage patterns.
G3 Transparent, visually grounded explanations.Couple
model-generated narratives with explicit visual evidence so that
analysts can quickly understand, verify, and critique system out-
puts rather than treating them as black-box predictions, thereby
supporting output-level auditing of VLM/LLM behavior in real
scenes.
© 17 April 2026 The Author(s).

4 of 12J. Perca et al. /URBANCLIPATLAS
Figure 2: URBANCLIPATLASmain workflow.(A)Preprocessing: long videos are segmented into clips, described by a VLM, indexed in
a vector store, and paired with static layout masks, yielding semantic and spatial representations for each video. (B)Augmented Narrative
Generation: at query time, the system enriches the user’s question, retrieves relevant clips, and composes narrative answers using the
precomputed embeddings, knowledge graph, and masks. (C)Visualization: the interface receives user queries, integrates outputs from all
components, and supports interactive exploration and validation of retrieved content.
3.2. Design Tasks
To operationalize these goals in a visual analytics setting, we spec-
ify the following technical design tasks for URBANCLIPATLAS:
T1 Natural-language query parsing and semantic retrieval
(G1).Implement a text interface that accepts free-form ques-
tions, performs query enrichment with domain terminology, en-
codes text into an embedding space, and retrieves candidate clips
from a vector index of video descriptions.
T2 Entity extraction, taxonomy mapping, and knowledge-
graph construction (G2, G3).From RAG-generated answers,
automatically detect entity mentions such as a car turning
right or a pedestrian crossing late, map these mentions to a
domain-specific taxonomy of intersection events, and instantiate
a knowledge graph that captures entity types, roles, and relation-
ships.
T3 Spatiotemporal grounding via detection and tracking (G2,
G3).Run object detection and multi-object tracking over video
clips, associate tracked objects with textual entities from the
knowledge graph, and maintain persistent identifiers and trajec-
tories across frames and clips.
T4 Linked visual encodings for narrative, entities, and video
(G2, G3).Design coordinated views so that the augmented nar-
rative, entity list/graph, and video player share common refer-
ences. Selecting an entity in the text highlights the corresponding
track in the video and vice versa.
T5 Interaction techniques for clip navigation and comparison
(G1–G3).Provide mechanisms to navigate between retrieved
clips, refine queries, and compare alternative narratives (e.g.,
multiple queries or parameter settings), supporting iterative ex-
ploration of complex behaviors across the video corpus and
human-in-the-loop refinement of the system’s interpretations.
3.3. System Workflow
To realize these goals, URBANCLIPATLAStransforms raw urban
videos into an interactive analysis environment through three inte-
grated stages (Fig. 2). In thePreprocessingstage (A), the system
segments long-duration street-level recordings, such as those from
the StreetAware dataset [PRB∗23], into overlapping clips. It then
processes these clips offline to obtain two complementary represen-
tations: textual descriptions generated by a vision-language modeland stored in a vector index for semantic retrieval, and static layout
masks that describe the intersection infrastructure, such as cross-
walks, sidewalks, and lanes. DuringAugmented Narrative Genera-
tion(B), analysts submit natural-language questions. URBANCLI-
PATLASenriches these queries with domain terminology, retrieves
relevant clips from the semantic index, and uses a RAG pipeline
to compose narrative answers. Entities are extracted and mapped
to a taxonomy of intersection events, and then instantiated in a
knowledge graph that can be aligned with downstream object de-
tection and tracking results. Finally, theVisualization and Interac-
tionstage (C) presents these results through an augmented narrative
interface that coordinates a video player, a narrative view, and an
entity/graph view. Analysts can move smoothly between queries,
explanations, and visual evidence to explore and validate safety-
and mobility-related patterns across the video corpus.
4. From Raw Videos to Semantic Representations
To support semantic search and retrieval over long urban
videos, URBANCLIPATLASperforms an offline preprocessing
stage (Fig. 2A) that converts raw intersection footage into struc-
tured, queryable representations. This stage comprises two com-
plementary procedures: (1) clip description and indexing for text-
based retrieval and (2) static layout mask extraction to contextual-
ize events within the physical infrastructure.
4.1. Clip Description and Indexing
Each input videovhas durationT vseconds (not necessarily the
same across videos). To preserve continuity and reduce the risk of
missing important interactions at segment boundaries, we divide
each video into fixed-length clips (τ=30 seconds) with a 5-second
overlap (ω=5). For a given videov, the number of clips is
mv=Tv−τ
τ−ω
+1,
and thei-th clipC v,istarts at timestampt v,i= (i−1)(τ−ω).
Each segmented clipC v,iis then processed by a VLM to gen-
erate a textual description summarizing its visual content and ac-
tivities. Given a prompt templateP(Supplementary Material C.1),
designed to emphasize traffic entities, behaviors, and safety-critical
© 17 April 2026 The Author(s).

J. Perca et al. /URBANCLIPATLAS5 of 12
Figure 3:Hierarchical taxonomy for knowledge-graph construction.The taxonomy organizes urban-traffic concepts into five top-level
categories—Agents,Motion Descriptors,Individual Behaviors,Safety Situations, andEnvironment Entities—providing a consistent semantic
basis for entity extraction, event classification, and grounding across all components ofURBANCLIPATLAS.
situations, we obtainD v,i=VLM(C v,i,P).During this stage, we
also adjust timestamps when needed to ensure temporal consis-
tency across descriptions by correcting offsets introduced by seg-
mentation or buffering.
To enable semantic similarity search, the textual descriptions are
embedded in a continuous vector space. An embedding model E
encodes each descriptionD v,ias
ev,i=E(D v,i),e v,i∈Rd,
whereddenotes the dimensionality of the embedding space. We
store these embeddings, together with their metadata (timestamps,
clip identifiers, and original descriptions), in a vector database,
yielding an indexed collection of clip descriptors that supports effi-
cient semantic retrieval during the query phase.
4.2. Static Layout Mask Extraction
To provide spatial context for retrieved events and to distinguish
moving agents from static infrastructure, URBANCLIPATLASex-
tracts semantic masks of each intersection’s layout. The goal is to
obtain a clean representation of the environment (e.g., crosswalks,
sidewalks, and lanes).
For each videov, we first samplenframes uniformly across
its duration, denotedF v,1, . . . ,F v,n. Each sampled frame is pro-
cessed with the open-vocabulary detector YOLO-World [CSG∗24]
to identify dynamic entities such as vehicles, pedestrians, and
cyclists. We selected YOLO-World for its state-of-the-art open-
vocabulary detection, which allows us to specify the traffic-related
categories of interest. To reduce false positives and enhance preci-
sion, we apply a detection confidence threshold of 0.65 across all
YOLO-World outputs, based on empirical tuning and used consis-
tently in all our experiments.
LetN(F v,j)denote the number of detected objects in frameFv,j. We choose as a reference the frame with the lowest detected
object count,F∗v=argminj∈{1,...,n} N(F v,j),which in practice
yields a view of the intersection where static infrastructure is least
occluded by traffic. We then apply the semantic segmentation
model Mask2Former [CMS∗22] toF∗vto obtain a set of masks
Mv={M v,1, . . . ,M v,p},where each mask corresponds to a spe-
cific static component of the environment (e.g., crosswalks, traffic
signals, road surfaces, sidewalks, and lane markings). These layout
masks serve as spatial reference layers throughout the pipeline,
enabling subsequent modules to contextualize detected events
relative to the intersection’s physical geometry and supporting
later tasks such as entity-level grounding and spatial reasoning.
5. Augmented Narrative Generation and Entity Alignment
The answer generation stage is the core of URBANCLIPATLAS
(Fig. 2B). It operates on the preprocessed clip index and lay-
out masks and (i) retrieves relevant clips through a video RAG
pipeline and (ii) structures and grounds the resulting narrative using
a taxonomy-guided knowledge graph.
5.1. Video RAG
The video RAG component processes a user’s natural-language
query through three steps: query enrichment, semantic retrieval,
and narrative generation. It produces (1) a query-aligned narrative
answer and (2) metadata about the selected segments as evidence.
Query Enrichment.Given a user queryQ, we first expand it into
an enriched queryQ′to better match the indexed clip descriptions.
A language model augmentsQwith traffic-specific terminology,
paraphrases, and contextual hints drawn from a domain contextK
(e.g., intersection events and entities) asQ′=LLM(Q,K).This
improves recall by reducing vocabulary mismatch and also helps
filter invalid or out-of-domain questions.
© 17 April 2026 The Author(s).

6 of 12J. Perca et al. /URBANCLIPATLAS
Semantic Retrieval.We embed the enriched queryQ′in the same
semantic space as the clip descriptions defined in the preprocessing
stage using the embedding model E:
q=E(Q′),q∈Rd.
The vector database then computes cosine similarity betweenqand
all clip embeddingse v,i, ranks clips by similarity, and returns the
top-kcandidatesR k={(v,i)}k
j=1 ,where each pair(v,i)indexes a
retrievedC v,iand its associatedD v,i.
Narrative Generation.The narrative answer is generated by con-
ditioning an LLM on the original query and the retrieved ev-
idence. In our implementation, we use the highest-ranked clip
Cv,itogether with its description and temporal metadataA=
GenerateNarrative(Q,C v,i,Dv,i,tv,i). WhereAfollows domain ter-
minology and explicitly references intersection entities, behaviors,
and safety situations. In the next stage, we convertAinto a struc-
tured representation and align its entities with the video content.
5.2. Taxonomy-Guided Entity Alignment
Raw answers can be long and difficult to interpret. To help ana-
lysts understand and validate the generated narratives, URBANCLI-
PATLASaugments them with a structured, grounded representation.
This process has two parts—knowledge graph construction and en-
tity grounding—whose main components are summarized in Fig. 4.
The top panel depicts the Knowledge Graph Construction module,
and the bottom panel depicts the Entity Grounding Engine.
Knowledge Graph Construction.We represent the answer as a
knowledge graphG= (V,E), where nodesVdenote entities and
edgesEdenote relationships. TheKnowledge Graph Construction
module (top part of Fig. 4) performs this transformation. Entity ex-
traction is guided by a fixed crossroad-interaction taxonomyTin-
spired by Shirazi and Morris [SM17] and adapted to our setting.
Fig. 3 illustrates this taxonomy, which organizes concepts into five
top-level categories: agents, motion descriptors, individual behav-
iors, safety situations, and environment entities, together with ex-
amples of each entity type considered.
An LLM-based extractor takes the answerAand outputs entity
mentions. Each mention has a class inT, a text span, optional at-
tributes (e.g.,red car,turning right), and a position in the answer.
We then perform a light canonicalization step to merge mentions of
the same real-world entity (e.g., “a red car” and later “the vehicle”
in the same scene) using lexical overlap and LLM-based corefer-
ence scoring. The result is a deduplicated set of nodesV∗with
enriched attributes.
Relations are extracted as directed triples(u s,r,u o)where
us,uo∈ V∗andris a relation type such asapproaches,yields-to, or
conflicts-with. The resulting graphG∗= (V∗,E)provides a com-
pact semantic summary of the narrative that can be visualized and
used for grounding.
Entity Grounding Engine.WhileG∗captures the semantics of the
answer, analysts also need to see where entities appear in the video.
The Entity Grounding Engine (bottom part of Fig. 4) attaches vi-
sual footprints (bounding boxes or masks) to the entities inV∗by
Figure 4:Taxonomy-guided entity alignment.TheKnowledge
Graph Constructionmodule uses the generated answer and the
fixed taxonomy to build a structured graphG∗of entities and re-
lations. TheEntity Grounding Enginethen combinesG∗, the re-
trieved clip, and the precomputed masks to localize dynamic en-
tities through detections and tracks, and to select relevant static
environment masks for visualization.
combining YOLO-World detections with the static layout masks
from Sec. 4.2.
Fordynamic entities(e.g., vehicles and pedestrians), we use
YOLO-World with the textual description of each relevant entity
u∈ V∗as prompt. Given a frameF tat timet, the detector returns a
set of bounding boxesB u=YOLOWorld(F t,span(u)),where each
b∈ B uencodes position, size, and confidence. Across frames, these
detections can be linked into short tracks.
Forstatic environment entities(e.g., crosswalks, signals, and
lane markings), grounding uses the precomputed layout masksM v
for the corresponding video. We select only those masks whose
class appears in the entity set:M active ={M∈ M v|class(M)∈
classes(V∗)}.Restricting toM active avoids visual clutter and fo-
cuses attention on environmental elements relevant to the current
answer.
Together, the knowledge graph and grounding engine connect
textual explanations to concrete visual evidence, enabling the inter-
face to highlight entities in both the narrative and the video and to
support entity-centric exploration of complex intersection scenes.
6. URBANCLIPATLAS
The URBANCLIPATLASinterface (Fig. 1) is designed as an interac-
tive visual analytics environment centered on an augmented conver-
sational experience. It supports the exploration, retrieval, and val-
idation of specific events in long-duration urban videos by tightly
coupling narrative answers, entity-level structures, and video ev-
idence. The interface is organized into three coordinated compo-
© 17 April 2026 The Author(s).

J. Perca et al. /URBANCLIPATLAS7 of 12
nents: the Chat Panel, the Video Player with tracking overlays, and
the Related Clips Timeline.
6.1. Chat Panel
TheChat Panel(Fig. 1A) is the main entry point for analysts. It
receives the user’s natural-language query and displays the aug-
mented narrative answerAproduced by the video RAG pipeline.
The answer describes what happens in the retrieved clip and iden-
tifies key actors (e.g., pedestrians, vehicles, and cyclists) as well as
environmental elements (e.g., crosswalks, traffic lights, and side-
walks). Using the entity metadata from the knowledge graph, the
panel highlights mentions of entities directly in the text. Entities
are color-coded according to the taxonomy categories introduced in
Fig. 3, with a legend shown at the bottom of the panel (Fig. 1A.4).
Clicking or hovering over a highlighted entity links its mention in
the text to the corresponding visualization in the video, allowing
analysts to directly identify and confirm the entity’s role and ac-
tions within the event.
When hovering over an entity, an interactive tooltip appears
(Fig. 1A.1–A.3). This tooltip summarizes the local neighborhood
of that node in the knowledge graph, showing related entities and
relationships (e.g.,involvesandcauses). This design follows vali-
dation strategies in prior visual analytics systems that couple LLM
reasoning with graph structures [CE25], helping users assess how
the model connects actors, behaviors, and safety situations.
TheChat Panelalso surfaces links to additional clips that exhibit
similar behaviors or interactions (Fig. 1A.4), together with a confi-
dence indicator reflecting the relevance of the retrieved clip to the
user’s query. Selecting one of these related clips updates theVideo
Playerand theRelated Clips Timeline. This supports comparison
of how a given pattern manifests at different times or in different
locations.
6.2.Video Playerwith Tracking Overlays
TheVideo Player(Fig. 1B) provides the primary visual evidence
supporting the narrative answer. Its role is to allow analysts to visu-
ally confirm and contextualize the retrieved information. The tex-
tual explanation, the knowledge graph, and the video frames remain
tightly aligned. To support this, the player renders multiple visual
layers on top of each frame in response to interactions originating
in the Chat Panel or the Timeline.
Dynamic entities.Dynamic actors, such as vehicles and pedestri-
ans, are localized using YOLO-World. This model is prompted with
textual descriptions of entities extracted from the answer. They are
rendered as colored bounding boxes whose hues follow the taxon-
omy. Examples of these overlays are shown in Fig. 1B.1, B.2, and
B.4. Clicking a dynamic entity mention in theChat Paneljumps the
video to the first frame in which that entity is detected and displays
its subsequent trajectory.
Static environment entities.Static infrastructure elements (e.g.,
crosswalks, sidewalks, lanes, and poles) are rendered using the
precomputed layout masks from Sec. 4.2. These masks are drawn
more subtly than dynamic entities to provide spatial context with-
out overwhelming the scene (Fig. 1B.3 and B.4). Hovering over anenvironmental entity in the text highlights the corresponding region
in the video. This interaction makes explicit how narrative elements
relate to physical infrastructure, revealing where and how entities
interact with the environment.
All overlays remain synchronized with temporal navigation.
Scrubbing, playing back, or jumping via the Timeline keeps the
highlighted entities and trajectories aligned with the narrative. This
enables analysts to move seamlessly between detailed frame-level
inspection and higher-level reasoning about behaviors and safety
situations.
6.3. Related Clips Timeline
TheRelated Clips Timeline(Fig. 1C) acts as a temporal overview
and navigation aid for the retrieved results. It visualizes clips
ranked by semantic relevance to the current query, as computed
by the video RAG component (Sec. 5.1).
Each row in the timeline corresponds to a video, and each cell
corresponds to a retrieved clip within that video. The horizontal
position encodes the clip’s temporal location in the source video,
while cell appearance encodes its semantic similarity score (clips
more relevant to the query are shown more prominently). When the
user restricts the search to a single video, the timeline displays a
single row. When the search spans the corpus, multiple rows appear,
enabling analysts to compare how the queried event or behavior
manifests across different viewpoints or intersections.
Interaction with the timeline is tightly coupled to the other views.
Clicking a cell in the active video row causes theVideo Playerto
seek to the start of that clip. Selecting a cell from another video
switches the player to that source, updates theChat Panel, and up-
dates the overlays accordingly. This coordination allows analysts
to move smoothly between corpus-level exploration (identifying
where relevant events occur) and detailed inspection of individual
scenarios.
6.4. Implementation Details
URBANCLIPATLASis implemented as a Python backend us-
ingFastAPIfor HTTP and WebSocket communication and a
Svelte + TypeScriptfrontend withD3.jsfor interactive
visualizations. The system integrates multiple foundation models,
includinggemini-2.5-profor video captioning,gemini-
embedding-001for semantic indexing, andgpt-4ofor query
enrichment and narrative generation. Object grounding is per-
formed with YOLO-World, while static layout masks are extracted
using Mask2Former.
At query time, the end-to-end response latency is typically
around 6–7 seconds on a consumer workstation. Additional im-
plementation details, system configuration, and prompts used in
the LLM pipeline are provided in Supplementary Material A
and C. To support reproducibility, the source code, configura-
tion files, and documentation for URBANCLIPATLASare pub-
licly available athttps://visualdslab.com/papers/
UrbanClipAtlas/.
© 17 April 2026 The Author(s).

8 of 12J. Perca et al. /URBANCLIPATLAS
Figure 5:Trajectories in Case Study 1.The supporting 3D trajec-
tory chart shows the motion of theman on a bike(green) and the
dark SUV(purple) over time. Insets (a)–(c) show key frames: (a)
the onset of the conflict as both approach the crosswalk; (b) the cy-
clist leaves the crosswalk while the SUV passes; and (c) the cyclist
continues along the trajectory after the near miss.
7. Evaluation
We qualitatively evaluate URBANCLIPATLASusing the
StreetAwaredataset and two case studies. The goal is to il-
lustrate how the system supports event retrieval, multi-entity
reasoning, and visual validation of safety-related situations
through concrete examples based on real urban videos.
7.1. Dataset: StreetAware
We use theStreetAwaredataset [PRB∗23], which contains high-
resolution, synchronized, multimodal street-level videos. These
videos were captured at busy urban intersections, with privacy pre-
served through the removal of audio and the blurring of vehicle li-
cense plates and human faces. StreetAware focuses on high-activity
crossings in Brooklyn, New York, and includes approximately 8
hours of recordings from three intersections. Unlike traditional au-
tomotive datasets captured from a moving ego-vehicle (e.g., KITTI
and Cityscapes), StreetAware employs static, multi-angle cameras
mounted at intersection corners. This setup provides a stable, wide
view of pedestrian–vehicle interactions and the surrounding built
environment. It is therefore well suited for analyzing behaviors
such as yielding, jaywalking, and near-miss incidents.
7.2. Case Studies
We present two case studies to demonstrate URBANCLIPATLAS’s
capabilities. Each case study starts from a natural-language query
and shows how theChat Panel,Video Player, andRelated Clips
Timeline(Fig. 1) work together to surface relevant events and sup-
port detailed inspection.
Conflicts between cyclists and cars.In our taxonomy (Fig. 3), a
Conflictis aSafety Situationinvolving two or more dynamic enti-
ties whose trajectories bring them into spatial and temporal proxim-
ity, creating a risk of collision without an actual crash [SM17]. Thiscase study examines how URBANCLIPATLAShelps identify and
analyze such conflicts between cyclists and vehicles. The analyst
issues the query:“Are there any significant conflict events involv-
ing cyclists and cars (SUV)?”The Video RAG component retrieves
a set of relevant clips and generates a narrative answer, which ap-
pears in theChat Panel(Fig. 1A). The answer highlights three key
entities:dark SUV,man on a bike, andcrosswalk, and describes
their interactions. The knowledge-graph tooltips (Fig. 1A.1–A.3)
expose the main relations:
•dark SUVinvolves− − − − →man on a bike;
•dark SUVcauses− − − − →forces the man on the bike to swerve out of the
crosswalk;
•forces the man on the bike to swerve out of the crosswalkcauses− − − − →
creates a near-miss scenario.
These relations collectively characterize the situation as a conflict
event. TheVideo Playerwith tracking overlays (Fig. 1B) provides
the corresponding visual evidence. The dark SUV and the cyclist
are rendered as colored tracks, allowing the analyst to observe how
the cyclist changes course to avoid the vehicle. Clicking the rele-
vant entities in theChat Panelcenters the playback on the moment
of maximum risk and reveals the full trajectories of both actors.
To further support reasoning about motion, we use the 3D tra-
jectory view (an external visualization not implemented directly
within URBANCLIPATLAS) shown in Fig. 5. The ground plane
shows the reference frame of the intersection, while the vertical
axis encodes time. The green trajectory corresponds to theman on
a bike, and the purple one to thedark SUV. Panels (a)–(c) on the
right show key frames: (a) when the conflict begins as both ap-
proach the crosswalk; (b) the cyclist leaves the crosswalk while the
SUV passes; and (c) the cyclist resumes their trajectory. Although
the projected paths intersect on the ground plane, the temporal axis
reveals that the two entities do not occupy the same space at the
same time. This confirms the event as a near miss rather than a
collision.
TheRelated Clips Timeline(Fig. 1C) lists additional clips ranked
by semantic similarity to the query. Most retrieved clips depict in-
teractions between cyclists and vehicles at varying levels of risk.
Take-away.URBANCLIPATLASsuccessfully retrieves conflict
events that match the analyst’s query and explains them through
coordinated narrative, structure, and video. TheChat Paneland
knowledge graph clarify which entities and relations define the con-
flict, while theVideo Player, the trajectory visualization, and the
Related Clips Timelinesupport inspection of how the event unfolds
and how it relates to similar scenarios. Together, these components
support both the discovery of near-miss events and the nuanced in-
terpretation of their dynamics.
Illegal crossing triggered by bus occlusion.A key advantage of a
chat-based interface is that analysts can iteratively refine their ques-
tions, moving from broad descriptions to targeted risk assessments.
In this case study, we start with a general request:“Identify any
large vehicle blocking the intersection. ”(Fig. 6A.1). We chose this
example because multiple large vehicles traverse the intersection
throughout the day. In response, URBANCLIPATLASretrieves a
clip in which a coach bus initially occupies part of the intersection.
TheVideo Player(Fig. 6A.1) highlights the bus with a bounding
© 17 April 2026 The Author(s).

J. Perca et al. /URBANCLIPATLAS9 of 12
Figure 6:Follow-up querying for risk-oriented analysis.(A) Initial prompt asking for large vehicles blocking the intersection: the Chat
Panel (A.2–A.3) and Video Player (A.1) highlight a coach bus executing a slow but compliant turning maneuver. (B) Follow-up prompt (B.1)
explicitly focuses on safety risks associated with that scene: the updated narrative (B.2) and entity tooltips (B.3–B.4) reveal that a different
bus enables an illegal crossing by pedestrians on bicycles.
box, and the narrative in theChat Panelexplains its behavior:“The
coach bus is initially occupying the intersection. It moves along
its trajectory from camera-left to camera-right at a low, constant
speed, completing a turning movement without blocking the cross-
walk. ”This description, together with the entity-level breakdown in
the tooltip (Fig. 6A.2–A.3), indicates that the bus clears the cross-
walk and the intersection without creating an obstruction. At this
stage, the system characterizes the vehicle’s behavior as compliant.
However, slow turning movements by large vehicles can still cre-
ate risky situations, particularly when other road users attempt to
exploit temporary gaps. This concern motivates a follow-up ques-
tion that focuses explicitly on risk:“Is the bus blocking the in-
tersection being exploited by any pedestrian to act imprudently?”
(Fig. 6B.1). Since the query builds on the previous clip, the system
maintains context while refocusing the narrative toward safety im-
plications. The new answer (Fig. 6B.2) reveals an emergent pattern:
“Yes, two pedestrians on bicycles exploit the stationary state of the
motorized vehicles to perform an illegal crossing. They accept the
gap created by the stopped buses and proceed to cross the intersec-
tion via the crosswalk despite the traffic light being red for them. ”
Entity extraction and the knowledge-graph tooltips (Fig. 6B.3–
B.4) make this structure explicit by connectingtwo pedestrians
on bicyclesto the actionperform an illegal crossingand linking
stopped busesto the relationcauses: accept the gap created by
the stopped buses. In other words, the buses form a visual shield
that encourages risky behavior at the crosswalk.
Importantly, the bus implicated in this second narrative is not the
original coach bus from the initial query (Fig. 6A.1), but a different
bus that arrives immediately afterward (Fig. 6B.4). This follow-up
reframes the scene: the initial query shows compliant bus behav-
ior, but the refined question reveals how a subsequent bus creates
conditions for an illegal red-light crossing.Take-away.This case study illustrates how URBANCLIPATLAS
supports semantic retrieval and iterative, risk-focused exploration
of long-duration urban videos. By allowing follow-up questions an-
chored to specific clips, the system helps analysts move from veri-
fying compliant vehicle behavior to uncovering subtle safety risks.
Here, an illegal crossing emerges due to bus occlusion, and analysts
can identify it without manually re-scrubbing the footage.
8. Discussion and Limitations
Role of the taxonomy and knowledge graph.A curated taxon-
omy of intersection concepts and its associated knowledge graph
are central to URBANCLIPATLAS. By constraining entity classes
and relations, the taxonomy focuses the LLM on traffic-relevant
semantics (e.g.,yielding,illegal crossing, andnear miss) and pro-
vides a stable scaffold for visualization. Deduplication and canon-
icalization further reduce noise by merging repeated mentions of
the same actor or event. Together, these mechanisms support rea-
soning that is closer to how traffic engineers and planners interpret
interactions than approaches based solely on raw detections.
Integration challenges and semantic drift.Relying on multi-
ple heterogeneous—and largely black-box—foundation models in-
troduces cascading sources of error. Miscaptioned clips may de-
grade retrieval quality, noisy extractions can propagate into the
knowledge graph, and imperfect detections can weaken links be-
tween textual entities and visual evidence. We therefore treat VLM-
generated descriptions as hypotheses rather than authoritative in-
terpretations. Analysts validate them through visual grounding by
comparing textual claims with detected objects and trajectories and
by synchronizing video playback. However, entity extraction re-
mains vulnerable: if the LLM diverges from the taxonomy or coref-
erence resolution fails, the graph may drift from the scene (e.g., by
© 17 April 2026 The Author(s).

10 of 12J. Perca et al. /URBANCLIPATLAS
splitting a single vehicle into multiple nodes or conflating distinct
actors). The interface mitigates these effects by co-presenting text,
structure, and video, allowing analysts to interrogate inconsisten-
cies and verify outputs against the visual evidence.
Bridging narratives and grounded evidence.Although VLMs
and LLMs can generate fine-grained descriptions (e.g.,“a pedes-
trian pushing a stroller”), open-vocabulary detectors such as
YOLO-World typically operate at broader categorical levels and
may fail to reliably ground these attributes. Across our experi-
ments, we observed three common failure modes: (1)grounding
misses, where described entities are not localized or tracked; (2)en-
tity misalignment, where behaviors are attributed to incorrect visual
instances; and (3)semantic over-interpretation, where the LLM la-
bels ambiguous interactions as conflicts or safety threats. Examples
are provided in Supplementary Material B. Currently, entities that
cannot be localized remain as ungrounded nodes in the knowledge
graph, which may bias interpretation. Exposing such uncertainties
and low-confidence entities is therefore important for maintaining
transparency and supporting analyst validation.
Expert feedback.As part of our evaluation, we conducted semi-
structured interviews with two domain experts (an architect and an
urban-design scholar), each with more than 10 years of experience
and active use of tools for urban data exploration. The interviews
focused on (i) perceived usefulness relative to current practice and
(ii) opportunities for community engagement toward a larger-scale
user study. Both experts highlighted the flexibility of chat-based
retrieval as a key advantage over interfaces constrained to fixed UI
filters (e.g.,“now you can exactly look for something and type it,
instead of being limited by the filters that you have. ”). They also
valued grounding narratives in video to focus attention, while not-
ing that recognizing actions in crowded scenes remains challeng-
ing and could become a practical bottleneck. Suggested extensions
included an anomaly gallery and curated query starters to guide
exploration. Finally, both identified transportation engineers and
practitioners in safety and microbehavioral studies as primary ben-
eficiaries, and pointed to partners in academia and public agencies
who could support broader adoption and evaluation.
Generality, scalability, and evaluation scope.The current proto-
type is tuned to the StreetAware dataset and to intersection-focused
analysis. Deploying URBANCLIPATLASin other cities or camera
setups would require adapting the taxonomy, revalidating detector
performance, and recalibrating prompts. While our preprocessing
pipeline handles several hours of video, scaling to city-wide de-
ployments with hundreds of cameras would require more efficient
indexing and incremental processing. Our evaluation relies on in-
depth case studies rather than a controlled user study, illustrating
how experts might use the system but not yet quantifying gains in
task performance, trust, or error detection.
Ethical considerations.Acknowledging the fine line between
surveillance-driven technology and systems designed to promote
public safety and resilient urban infrastructure, we incorporate safe-
guards to reduce discrimination and misuse. At the data level, the
StreetAware videos are anonymized, with faces and license plates
blurred and audio excluded. At the system level, the architecture
supports query safeguards that reject profiling requests or othermisuse (e.g., gender- or race-related queries), and the taxonomy
is intentionally restricted to traffic-safety concepts, excluding per-
sonal attributes.
Limitations.A key limitation of URBANCLIPATLASis its re-
liance on proprietary, cloud-hosted LLMs and VLMs, whose inter-
nal reasoning is not fully interpretable. Our design therefore links
each generated claim to its originating clips, detected objects, and
taxonomy-constrained entities, enabling analysts to audit outputs
within a human-in-the-loop workflow. This reliance limits repro-
ducibility, raises costs and privacy concerns, and ties latency to
network and API conditions. On the visual analytics side, we cur-
rently treat model outputs as point estimates, which can obscure
uncertainty in captioning, retrieval, and grounding.
9. Conclusions and Future Work
URBANCLIPATLASintroduces a visual analytics workflow
that combines semantic retrieval, entity-centric reasoning, and
grounded visual exploration for long-duration urban videos. By in-
tegrating lightweight video RAG with taxonomy-aware augmenta-
tion and synchronized visual overlays, the system supports traffic
analysis tasks in which analysts must locate, explain, and validate
complex intersection events. Our case studies show that this inte-
grated approach surfaces relevant interactions more efficiently and
strengthens the connection between narrative explanations and ob-
servable evidence.
Building on these results, we plan to extend URBANCLIPATLAS
in three directions:Grounding and uncertaintyby propagating
uncertainty from captioning, retrieval, and detection into the inter-
face and flagging ungrounded or low-confidence entities;Scala-
bility and model independenceby scaling indexing and prepro-
cessing to larger camera networks while reducing dependence on
proprietary cloud models through open or locally deployable al-
ternatives; andInteraction and multimodalityby incorporating
modalities such as audio or sensor signals and enabling analysts to
correct entity labels, merge or split nodes, and flag false positives
to improve robustness over time. We also plan to strengthen pri-
vacy protection and misuse prevention through more robust query
filtering, anonymization pipelines, and audit mechanisms.
Acknowledgments
The authors acknowledge the use of LLM-based tools (Chat-
GPT and Grammarly) as writing assistants for text polishing.
This work was supported by the National Council for Scientific
and Technological Development (CNPq, grants #311144/2022-5,
#132348/2025-0, and #132349/2025-6), the Carlos Chagas Filho
Foundation for Research Support of the State of Rio de Janeiro
(FAPERJ, grant #E-26/210.585/2025), the São Paulo Research
Foundation (FAPESP, grants #2021/07012-0 and #2023/04868-7),
and the School of Applied Mathematics at Fundação Getulio Var-
gas. This research was also partially supported by the National Sci-
ence Foundation (NSF, Award #OAC-2411221) and by Connected
Cities for Smart Mobility toward Accessible and Resilient Trans-
portation (C2SMART), a Tier 1 University Transportation Center
funded by the U.S. Department of Transportation (USDOT, con-
tract #69A3551747124).
© 17 April 2026 The Author(s).

J. Perca et al. /URBANCLIPATLAS11 of 12
References
[ADC24] AREFEENM. A., DEBNATHB., CHAKRADHARS.: Traffi-
cLens: Multi-camera traffic video analysis using LLMs. InIEEE 27th
International Conference on Intelligent Transportation Systems (ITSC)
(Sept. 2024), pp. 3974–3981.doi:10.1109/ITSC58415.2024.
10920144. 3
[ADSUC24] AREFEENM. A., DEBNATHB., SARWARUDDINM. Y.,
CHAKRADHARS.: ViTA: An efficient video-to-text algorithm using
VLM for RAG-based video analysis system. InIEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition Workshops (CVPRW)
(June 2024), pp. 2266–2274.doi:10.1109/CVPRW63382.2024.
00232. 2
[ADUC24] AREFEENM. A., DEBNATHB., UDDINM. Y. S.,
CHAKRADHARS.: iRAG: Advancing RAG for videos with an incre-
mental approach. 4341–4348. arXiv:2404.12309.doi:10.1145/
3627673.3680088. 2
[AE24] ADEWOPOV. A., ELSAYEDN.: Smart city transportation:
Deep learning ensemble approach for traffic accident detection.IEEE
Access 12(2024), 59134–59147.doi:10.1109/ACCESS.2024.
3387972. 2
[AH20] ALKAISSIZ. A., HUSSAINR. Y.: Delay time analysis and mod-
elling of signalised intersections using global positioning system (GPS)
receivers.IOP Conference Series: Materials Science and Engineering
671, 1 (Jan. 2020), 012110.doi:10.1088/1757-899X/671/1/
012110. 2
[ASM∗21] ABOAHA., SHOMANM., MANDALV., DAVAMIS., ADU-
GYAMFIY., SHARMAA.: A vision-based system for traffic anomaly
detection using deep learning and decision trees. In2021 IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops
(CVPRW)(Los Alamitos, CA, USA, June 2021), IEEE Computer Soci-
ety, pp. 4202–4207.doi:10.1109/CVPRW53098.2021.00475.
2
[BHM∗25] BAHMANYARR., HELLEKESJ., MÜHLHAUSM.,
GSTAIGERV., KURZF.: Traffic pattern analysis at urban in-
tersections through vehicle detection in aerial imagery.IS-
PRS Annals of the Photogrammetry, Remote Sensing and
Spatial Information Sciences X-G-2025(2025), 151–158.
doi:10.5194/isprs-annals-X-G-2025-151-2025. 2
[BMZ∗15] BLINCOEL. J., MILLERT. R., ZALOSHNJAE., LAWRENCE
B.,ET AL.:The Economic and Societal Impact of Motor Vehicle
Crashes, 2010 (Revised). Tech. rep., U.S. Department of Transporta-
tion, National Highway Traffic Safety Administration, 2015. URL:
https://rosap.ntl.bts.gov/view/dot/78697. 2
[CBH∗20] CHENK., BANERJEET., HUANGX., RANGARAJANA.,
RANKAS.: A visual analytics system for processed videos from traf-
fic intersections. InProceedings of the 6th International Conference on
Vehicle Technology and Intelligent Transport Systems (VEHITS 2020)
(2020), Berns K., Helfert M., Gusikhin O., (Eds.), SCITEPRESS, pp. 68–
77.doi:10.5220/0009422300680077. 3
[CE25] COSCIAA., ENDERTA.: VisPile: A visual analytics system
for analyzing multiple text documents with large language models and
knowledge graphs, 2025. URL:https://arxiv.org/abs/2510.
09605,arXiv:2510.09605. 7
[CMS∗22] CHENGB., MISRAI., SCHWINGA. G., KIRILLOVA.,
GIRDHARR.: Masked-attention mask transformer for universal im-
age segmentation. In2022 IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR)(2022), pp. 1280–1289.doi:
10.1109/CVPR52688.2022.00135. 5
[CNC∗20] CHANG. Y.-Y., NONATOL. G., CHUA., RAGHAVANP.,
ALURUV., SILVAC. T.: Motion browser: Visualizing and understand-
ing complex upper limb movement under obstetrical brachial plexus in-
juries.IEEE Transactions on Visualization and Computer Graphics 26,
1 (Jan. 2020), 981–990.doi:10.1109/TVCG.2019.2934280. 3
[CPY∗25] CAIW., PONOMARENKOI., YUANJ., LIX., YANGW.,
DONGH., ZHAOB.: SpatialBot: Precise spatial understanding withvision language models. In2025 IEEE International Conference on
Robotics and Automation (ICRA)(2025), pp. 9490–9498.doi:10.
1109/ICRA55743.2025.11128671. 3
[CSG∗24] CHENGT., SONGL., GEY., LIUW., WANGX., SHANY.:
YOLO-World: Real-time open-vocabulary object detection, Feb. 2024.
arXiv:2401.17270.doi:10.48550/arXiv.2401.17270. 3, 5
[CZYW17] CHENP., ZENGW., YUG., WANGY.: Surrogate safety
analysis of pedestrian-vehicle conflict at intersections using unmanned
aerial vehicle videos.Journal of Advanced Transportation 2017, 1
(2017), 5202150.doi:10.1155/2017/5202150. 2
[DHZ∗18] DENGM., HUANGJ., ZHANGY., LIUH., TANGL., TANG
J., YANGX.: Generating urban road intersection models from low-
frequency gps trajectory data.International Journal of Geographi-
cal Information Science 32, 12 (2018), 2337–2361.doi:10.1080/
13658816.2018.1510124. 2
[DSP∗25] DIAZM., SANTEL., PERCAJ.,DASILVAJ. V., FERREIRA
N., POCOJ.: STRive: An association rule-based system for the explo-
ration of spatiotemporal categorical data.Computers & Graphics 132
(2025), 104410.doi:10.1016/j.cag.2025.104410. 3
[Fed24] FEDERALHIGHWAYADMINISTRATION: About inter-
section safety.https://highways.dot.gov/safety/
intersection-safety/about, 2024. Last updated July 26,
2024; accessed November 28, 2025. 2
[GXY∗25] GUOZ., XIAL., YUY., AOT., HUANGC.: Ligh-
tRAG: Simple and fast retrieval-augmented generation, Apr. 2025.
arXiv:2410.05779.doi:10.48550/arXiv.2410.05779. 2
[HERMCP25] HEREDIAJ., ESTRADA-RAYMEL., MATOS-
CANGALAYAJ., POCOJ.: Interactive exploration and ex-
planation of spatio-temporal anomalies with graph-llm inte-
gration. In2025 38th SIBGRAPI Conference on Graphics,
Patterns and Images (SIBGRAPI)(2025), pp. 1–6.doi:
10.1109/SIBGRAPI67909.2025.11223398. 2
[JCL∗24] JIANGW., CHENGY., LIUL., FANGY., PENGY., LIUY.:
Comprehensive visual grounding for video description.Proceedings of
the AAAI Conference on Artificial Intelligence 38, 3 (Mar. 2024), 2552–
2560.doi:10.1609/aaai.v38i3.28032. 3
[JKBH25] JEONGS., KIMK., BAEKJ., HWANGS. J.: Vide-
oRAG: Retrieval-augmented generation over video corpus, May 2025.
arXiv:2501.05874.doi:10.48550/arXiv.2501.05874. 2, 3
[LZY∗24] LUOY., ZHENGX., YANGX., LIG., LINH., HUANGJ.,
JIJ., CHAOF., LUOJ., JIR.: Video-RAG: Visually-aligned retrieval-
augmented long video comprehension, Dec. 2024. arXiv:2411.13093.
doi:10.48550/arXiv.2411.13093. 3
[MPCK∗25] MAOM., PEREZ-CABARCASM. M., KALLAKURIU.,
WAYTOWICHN. R., LINX., MOHSENINT.: Multi-RAG: A multi-
modal retrieval-augmented generation system for adaptive video under-
standing, June 2025. arXiv:2505.23990.doi:10.48550/arXiv.
2505.23990. 3
[NDP25] NUNESA. L., DIAZM., POCOJ.: MineTracker: Visual analyt-
ics for spatiotemporal analysis of mining areas in the brazilian amazon.
In2025 38th SIBGRAPI Conference on Graphics, Patterns and Images
(SIBGRAPI)(2025), pp. 1–6.doi:10.1109/SIBGRAPI67909.
2025.11223361. 3
[PRB∗23] PIADYKY., RULFFJ., BREWERE., HOSSEINIM., OZBAY
K., SANKARADASM., CHAKRADHARS., SILVAC.: StreetAware: A
high-resolution synchronized multimodal urban scene dataset.Sensors
23, 7 (Jan. 2023), 3710.doi:10.3390/s23073710. 2, 4, 8
[RPH∗24] RULFFJ., PEREIRAG., HOSSEINIM., LAGEM., SILVAC.:
Towards data-informed interventions: Opportunities and challenges of
street-level multimodal sensing, 2024. URL:https://arxiv.org/
abs/2410.22092,arXiv:2410.22092. 2
[RXX∗25] RENX., XUL., XIAL., WANGS., YIND., HUANGC.:
VideoRAG: Retrieval-augmented generation with extreme long-context
videos, 2025. URL:https://arxiv.org/abs/2502.01549,
arXiv:2502.01549. 2, 3
© 17 April 2026 The Author(s).

12 of 12J. Perca et al. /URBANCLIPATLAS
[SH19] SCHÖNINGJ., HEIDEMANNG.: Visual video analytics for
interactive video content analysis. InAdvances in Information and
Communication Networks(Cham, 2019), Arai K., Kapoor S., Bhatia
R., (Eds.), Springer International Publishing, pp. 346–360.doi:10.
1007/978-3-030-03402-3_23. 3
[SM17] SHIRAZIM. S., MORRISB. T.: Looking at intersections: A
survey of intersection monitoring, behavior and safety analysis of recent
studies.IEEE Transactions on Intelligent Transportation Systems 18, 1
(Jan. 2017), 4–24.doi:10.1109/TITS.2016.2568920. 6, 8
[SMMSL20] STIPANCICJ., MIRANDA-MORENOL., STRAUSSJ.,
LABBEA.: Pedestrian safety at signalized intersections: Modelling
spatial effects of exposure, geometry and signalization on a large ur-
ban network.Accident Analysis & Prevention 134(2020), 105265.
doi:10.1016/j.aap.2019.105265. 2
[SRC25] SANKARADASM., RAJENDRANR. K., CHAKRADHARS. T.:
StreamingRAG: Real-time contextual retrieval and generation frame-
work, Jan. 2025. arXiv:2501.14101.doi:10.48550/arXiv.
2501.14101. 2
[TYL∗25] TANX., YEY., LUOY., WANQ., LIUF., CAIZ.: RAG-
Adapter: A plug-and-play RAG-enhanced framework for long video un-
derstanding, Mar. 2025. arXiv:2503.08576.doi:10.48550/arXiv.
2503.08576. 3
[WGQ∗24] WUT., GES., QINJ., WUG., WANGL.: Open-vocabulary
spatio-temporal action detection, 2024. URL:https://arxiv.
org/abs/2405.10832,arXiv:2405.10832. 3
[WQ20] WUA., QUH.: Multimodal analysis of video collections: Vi-
sual exploration of presentation techniques in TED talks.IEEE Transac-
tions on Visualization and Computer Graphics 26, 7 (2020), 2429–2442.
doi:10.1109/TVCG.2018.2889081. 3
© 17 April 2026 The Author(s).