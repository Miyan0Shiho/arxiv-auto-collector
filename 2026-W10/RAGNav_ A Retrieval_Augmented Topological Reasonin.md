# RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal Visual-Language Navigation

**Authors**: Ling Luo, Qiangian Bai

**Published**: 2026-03-04 05:31:33

**PDF URL**: [https://arxiv.org/pdf/2603.03745v1](https://arxiv.org/pdf/2603.03745v1)

## Abstract
Vision-Language Navigation (VLN) is evolving from single-point pathfinding toward the more challenging Multi-Goal VLN. This task requires agents to accurately identify multiple entities while collaboratively reasoning over their spatial-physical constraints and sequential execution order. However, generic Retrieval-Augmented Generation (RAG) paradigms often suffer from spatial hallucinations and planning drift when handling multi-object associations due to the lack of explicit spatial modeling.To address these challenges, we propose RAGNav, a framework that bridges the gap between semantic reasoning and physical structure. The core of RAGNav is a Dual-Basis Memory system, which integrates a low-level topological map for maintaining physical connectivity with a high-level semantic forest for hierarchical environment abstraction. Building on this representation, the framework introduces an anchor-guided conditional retrieval and a topological neighbor score propagation mechanism. This approach facilitates the rapid screening of candidate targets and the elimination of semantic noise, while performing semantic calibration by leveraging the physical associations inherent in the topological neighborhood.This mechanism significantly enhances the capability of inter-target reachability reasoning and the efficiency of sequential planning. Experimental results demonstrate that RAGNav achieves state-of-the-art (SOTA) performance in complex multi-goal navigation tasks.

## Full Text


<!-- PDF content starts -->

RAGNav: A Retrieval-Augmented Topological
Reasoning Framework for Multi-Goal Visual-Language
Navigation
Ling Luoa, Qianqian Baia
aSouthwestern University of Finance and Economics, Chengdu, 611130, China
Abstract
Vision-Language Navigation (VLN) is evolving from single-point pathfinding
toward the more challenging Multi-Goal VLN. This task requires agents to accu-
rately identify multiple entities while collaboratively reasoning over their spatial-
physical constraints and sequential execution order. However, generic Retrieval-
Augmented Generation (RAG) paradigms often suffer from spatial hallucinations
and planning drift when handling multi-object associations due to the lack of
explicit spatial modeling.To address these challenges, we propose RAGNav, a
framework that bridges the gap between semantic reasoning and physical struc-
ture. The core of RAGNav is a Dual-Basis Memory system, which integrates a
low-level topological map for maintaining physical connectivity with a high-level
semantic forest for hierarchical environment abstraction. Building on this repre-
sentation, the framework introduces an anchor-guided conditional retrieval and a
topological neighbor score propagation mechanism. This approach facilitates the
rapid screening of candidate targets and the elimination of semantic noise, while
performing semantic calibration by leveraging the physical associations inherent
in the topological neighborhood.This mechanism significantly enhances the capa-
bility of inter-target reachability reasoning and the efficiency of sequential plan-arXiv:2603.03745v1  [cs.AI]  4 Mar 2026

ning. Experimental results demonstrate that RAGNav achieves state-of-the-art
(SOTA) performance in complex multi-goal navigation tasks.
Keywords:Visual-Language Navigation (VLN), Retrieval-Augmented
Generation (RAG), Topological Map, Hierarchical Retrieval, Retrieval
Augmentation
1. Introduction
Vision-and-Language Navigation (VLN) aims to enable an agent to move in
complex environments by following natural language instructions, serving as a
key task connecting perception, language, and action [1]. In recent years, the task
paradigm has expanded from single-goal navigation to Multi-goal VLN, which is
more practically valuable and challenging [2, 3, 4]. Such tasks (e.g., “First go to
the bedside in the bedroom, then to the desk in the study") require the agent not
only to identify individual goals but also to perform joint reasoning and sequential
planning on the spatial relationships and semantic relationships among multiple
goals.
To conduct exploratory planning in complex environments, constructing a
lightweight internal representation of the environment is crucial. Topological
maps, which abstract continuous environments into graphs composed of key loca-
tions (nodes) and their connectivity (edges), have become a widely adopted form
of spatial memory in VLN [5, 6, 7]. However, traditional topological maps mainly
encode geometric or low-level visual features, exposing inherent limitations when
dealing with multi-goal instructions. First, they are poor in semantic representa-
tion. Topological nodes are usually difficult to associate with high-level seman-
tic concepts mentioned in instructions (e.g., “living room”, “desk”) because their
2

construction process often ignores semantic category information critical for task
reasoning [8, 9].Second, they are weak in relational reasoning. Although spatial
accessible paths can be found through graph search, existing methods struggle to
understand the semantic context between goals, leading to semantically unreason-
able paths or incorrect goal orders in planning. Essentially, traditional topological
maps lack a queryable, multi-modal semantic memory system, making it impos-
sible to effectively store and utilize rich visual, spatial, and semantic information
from embodied experience.
Retrieval-Augmented Generation (RAG) technology significantly improves
the factual accuracy and reasoning traceability of Large Language Models (LLMs)
in knowledge-intensive tasks by equipping them with a retrievable external knowl-
edge base [10, 11]. This provides a new idea to bridge the semantic gap of topo-
logical maps: can we construct a semantic memory based on RAG to enhance or
even partially replace the functions of traditional topological maps? Recent works
such as EmbodiedRAG [12] propose to enhance LLM-based planners by prepro-
cessing task-related entities and attributes and retrieving associated information to
execute natural language robot tasks. Such methods systematically introduce the
“retrieval-generation" mechanism of RAG into embodied scenarios for the first
time, providing a feasible path for building queryable scene semantic memory.
However, directly applying RAG to multi-goal VLN still faces severe chal-
lenges. First, multimodal data querying poses a significant challenge. It still re-
quires specialized design to efficiently organize heterogeneous data such as visual
and spatial pose to enable accurate responses to complex navigation instruction
queries [13, 14]. Second, the challenge of spatial-semantic coupled retrieval. The
vector similarity-based retrieval mechanism of NaiveRAG lacks explicit model-
3

ing of physical spatial layout and connectivity, making it unable to directly an-
swer spatial relationship queries such as “Is it accessible from A to B" [10]; while
GraphRAG methods that can construct complex associations often fail to meet the
real-time requirements of navigation due to their retrieval efficiency [15, 16].
To address the above challenges, this paper proposesRAGNav, a retrieval-
augmented topological semantic reasoning framework for multi-goal Visual-Language
Navigation. The core innovation of this framework lies in the construction of a
dual-basis environmental memory system jointly supported by a low-level topo-
logical map and a high-level semantic forest. In this system, the topological
map serves as the physical skeleton of the environment, maintaining connectivity
among key pose nodes and enforcing spatial topological constraints. Meanwhile,
the semantic forest hierarchically indexes multimodal embodied experiences, en-
abling efficient organization and representation of environmental information at
different granularities. Based on this dual-basis representation, we further propose
a spatial–neighbor dual-dimensional retrieval-augmented strategy. In the spatial
dimension, the system leverages the hierarchical structure of the semantic for-
est to perform rapid pruning, thereby achieving scale alignment and preliminary
localization of target candidate regions within massive environmental data. In
the neighbor dimension, the system deeply exploits physical correlations within
topological neighborhoods, filters out isolated noise through a semantic finger-
print verification mechanism, and performs in-depth reasoning over the sequen-
tial reachability among multiple goals using graph propagation mechanisms. This
strategy not only enables the RAG system to identify specific target attributes,
but also allows it to deeply understand the complex spatial coupling logic among
multiple goals. Ultimately, this paradigm drives a complete task loop composed of
4

perception, planning, execution, and reflection, significantly enhancing the agent’s
robustness and decision-making efficiency when facing uncertain environments.
The main academic contributions of this paper can be summarized as follows
• To address the mismatch between semantic logic and spatial topology in
multi-goal VLN tasks, we propose a new model that leverages non-parametric
memory to achieve hierarchical accumulation of environmental knowledge
and logical reconstruction of long instructions.
• We propose the RAGNav framework, which effectively addresses the spatial-
semantic gap by implementing hierarchical pruning in the semantic forest
and reasoning across topological neighborhoods. This enables deep seman-
tic alignment from high-level task parsing down to low-level physical veri-
fication.
• Experimental results demonstrate that the RAGNav framework achieves
state-of-the-art performance on multi-goal navigation tasks. Its retrieval
module significantly outperforms baseline methods such as NaiveRAG [10],
GraphRAG [15], and LightRAG [17] in both efficiency and accuracy.
2. Related Work
2.1. Multi-Object Navigation
Object goal navigation can be divided into single-object navigation [1, 18, 19]
and Multi-Object Navigation [2, 3, 4] according to the number of task goals. As
a generalization of the former, the latter requires the agent to explore and lo-
cate multiple targets in sequence in an unknown environment based on a given
5

sequence of semantic labels. Its core challenge shifts from a single "localization-
arrival" to the understanding of the spatial and semantic relationship network im-
plied in the instructions, as well as the planning of the global action sequence.
To address this challenge, existing studies have made certain breakthroughs. For
example, Chen et al. [20] first proposed an active camera strategy to solve the
multi-object navigation problem, which significantly improved the success rate;
Sadek et al. [4] proposed a modular hybrid navigation method that decouples high-
level semantic planning from low-level motion control, focusing on the transfer
of strategies from simulation to reality; Marza et al. [21] recently proposed using
a dual neural network to learn the neural implicit representation of scenes, aiming
to dynamically construct and utilize the global contextual information of scenes,
which provides a new paradigm for the multi-object goal navigation task.
However, the reasoning of these methods on the relationships among multiple
targets is often offline or fragmented, failing to form a closed loop that can con-
tinuously and actively "retrieve" the next most relevant target or key path point
from the environmental memory during the planning process. The lack of such a
dynamic retrieval order limits the agent’s reasoning ability in complex scenes and
under lengthy instructions, and the planning is prone to fall into local optimality
or semantic sequence errors. Therefore, there is an urgent need for a new frame-
work that deeply couples spatial topological structure with semantic relationship
retrieval to realize dynamic and interpretable reasoning on the access order of
multiple targets. This work is precisely carried out to address this key issue.
2.2. Retrieval Augmented Generation
Retrieval-Augmented Generation (RAG) [10] is a core method integrating
information retrieval and generation models. By introducing external knowl-
6

edge sources and retrieval mechanisms, it effectively makes up for the limita-
tions of traditional sequence-to-sequence frameworks and significantly improves
the performance of natural language processing tasks. Since then, a variety of
technologies and adaptive methods have been proposed, such as: MuRAG [22]
which enhances question-answering capabilities by retrieving images, Self-RAG
[23] which introduces a self-reflection mechanism to optimize outputs, Instruc-
tRAG [24] which adopts explicit denoising technology to filter noisy information,
GRAFT-Net [25] which fuses corpus texts and knowledge graphs to construct
heterogeneous graphs, and ToG [26] which takes knowledge graphs as the index
structure. However, the above studies mostly focus on pure text or general sce-
narios, while this paper focuses on the field of embodied navigation – the robot
can actively change the retrieval information source, and the information related
to the original query will dynamically change as the robot executes the plan. In
addition, the embodied navigation experience itself has the characteristics of re-
dundancy, hierarchical correlation and spatial anchoring, which make the above
pure text graph construction methods difficult to adapt and perform poorly.
2.3. Navigation Environment Representation
To achieve efficient planning in unknown environments, constructing a lightweight
and semantically rich internal representation is the key. Traditional methods usu-
ally rely on static observations [27] and predefined semantic systems to build topo-
logical semantic maps [5, 6, 28], which makes it difficult for them to adapt to the
fine-grained semantic requirements of open scenarios and lack the ability of on-
line update [29]. Recent relevant studies have made remarkable progress. First,
semantic injection based on Vision-Language Models (VLM) enables fast seman-
tic annotation by attaching descriptive texts generated by VLM to map nodes [30],
7

but this method is only a loosely coupled semantic superposition, lacking explicit
modeling of semantic relationships between objects, and thus hard to support tasks
that require deep spatial-semantic joint reasoning. Second, the unified represen-
tation based on 3D scene graphs [31] encodes objects, semantics and their rela-
tionships in a dense graph structure. Although it has strong reasoning potential,
its construction relies on expensive 3D reconstruction technology, leading to low
efficiency in online construction. In addition, the flat structure lacks hierarchi-
cal abstraction, which results in low retrieval efficiency when facing high-level
semantic queries. Third, the exploration of hierarchical semantic maps [32] at-
tempts to approach human cognition by constructing multi-layer structures such
as "object-region" to improve planning efficiency. However, the high-level ab-
straction of existing methods mostly relies on fixed spatial segmentation or sim-
ple clustering rules, and their semantic division is static and rigid, which cannot
evolve dynamically according to task contexts (e.g., aggregating scattered "coffee
machines" and "sinks" into a "breakfast nook"). Meanwhile, they fail to solve
the core problem of how to efficiently support dynamic retrieval and sequential
reasoning of multi-target and multi-relation instructions based on this structure.
3. Methodology
The RAGNav framework proposed in this paper aims to address the chal-
lenges of long instruction decomposition and spatial relationship reasoning in
multi-objective navigation. As illustrated in Figure??, the framework consists
of two phases: offline environmental memory construction and an online task ex-
ecution loop. In the offline phase, the framework utilizes pose data to construct
environmental representations, forming a dual-baseline environmental memory
8

composed of topological maps and semantic forests. During the online phase, the
system employs a Large Language Model (LLM) to parse long instructions, de-
composing them into subtask chains with logical dependencies while extracting
the spatial and temporal dependencies involved. Leveraging a two-stage retrieval
and topology enhancement mechanism, the system performs spatial consistency
verification and precise associative localization for multiple target entities. By
integrating feedback on topological connectivity, the system conducts sequen-
tial planning for multiple objectives, forming a closed-loop "perception-planning-
execution-reflection" task cycle. This significantly enhances the execution effi-
ciency of multi-objective tasks under complex instructions.
Bottom-Up
ConstructionTop-Down
RetrievalDual-Baseline Memory
Construction
Semantic
Forest
Topological
Mape.g.Root Node: Of fice Zone
e.g.Child Node: Water
Area
e.g.Child Node: Water
Dispenser Section
e.g.Leaf: Paper CuptopK
Re-rankingEnvironmental Data
Instruction Decomposition
Query LLMSpatial Dependency （A near B）
Temporal Dependency （ First A, then B ）
P = {P1, P2, . . . , Pn} LLMRetrieval and
Reasoning
Figure 1: Architecture of the RAGNav Framework
3.1. Embodied Data Acquisition and Autonomous Exploration
Prior to constructing the dual-basis environment memory, the framework es-
tablishes a comprehensive multimodal perception and autonomous mapping sys-
9

tem. During exploration, the agent continuously collects synchronized data in-
cluding timestampst, RGB image observationsI t, LiDAR point cloudsP t, and
six-degree-of-freedom (6-DoF) posesξ t(comprising position(x, y, z)and orien-
tation) generated via odometry fusion. The precise geometric information pro-
vided by LiDAR point clouds is crucial for building the spatial skeleton of the
environment and understanding free space. All raw data are transformed into a
unified global world coordinate system, forming a spatiotemporally aligned raw
data stream:
D={(t, I t, Pt, ξt)}(1)
To achieve efficient environment coverage, the system deploys an active map-
ping algorithm based on Frontier Exploration. The algorithm discretizes the con-
tinuous environment into grids with a resolution of 1.0 m and dynamically tracks
exploration states. Based on the coverage of the current LiDAR frameP t, grids
are marked as “explored” or “unknown.” By scanning the four-connected neigh-
borhood of explored grids, frontier candidate points located at the boundaries are
identified. Breadth-first search (BFS) is then applied for spatial clustering to sup-
press noise, yielding stable frontier regionsC i. Each valid frontier region is rep-
resented by its geometric center as a frontier pointf i. The agent adopts a greedy
strategy, selecting the frontier point closest to its current positionp current as the
next navigation target:
fnext= arg min
f∈F∥pcurrent−f∥ 2 (2)
3.2. Intelligent Task Decomposition and Execution Planning
To address the challenges of lengthy and logically coupled instructions in
multi-goal navigation, the framework introduces an intelligent task decomposition
10

and sequential execution mechanism. The system first utilizes a large language
model (LLM) as the core reasoning module to parse the original natural language
instructionQinto a structured task sequenceT={t 1, t2, . . . , t n}. During this
process, dependency identification is performed, modeling inter-task constraints
into two fundamental logical types that guide subsequent retrieval.
For spatial dependency, when instructions contain explicit spatial constraints
(e.g., “A near B”), the system designates targetBas the anchor and targetAas
the condition-dependent entity. Retrieval then shifts from global blind search to
a conditional probability maximization problem within the topological neighbor-
hoodN(v B)of the anchor nodev B. The spatial scoreS spatial of a candidate node
viis defined as the product of semantic similarity and physical distance constraint:
Sspatial (vi|vB) =sim(ϕ(Q A), ψ(v i))·exp
−∥pi−pB∥2
2σ2
(3)
whereϕ(·)andψ(·)denote text and visual embedding functions, respectively,
and sim(·)represents cosine similarity. The exponential term applies a Gaussian
kernel to softly constrain physical distance, ensuring that candidate targetAlies
spatially close to the anchorBat positionp B, thereby filtering semantic noise.
For temporal dependency, when instructions specify an explicit visiting order
(e.g., “First A, then B”), the system formulates the problem as a cost-sensitive se-
quential planning task. Given the set of retrieved target locationsP={p 1, p2, . . . , p n},
the objective is to find the globally optimal execution pathΠ∗:
Π∗= arg min
π∈Perm(P)n−1X
j=1D(p π(j), pπ(j+1) ) +λ· L(π,S)(4)
Here, Perm(P)denotes all permutations of candidate paths,D(·)is the shortest-
path cost computed from topological connectivity, andL(·)is a semantic deviation
11

penalty measuring inconsistency between execution orderπand the instruction-
defined semantic sequenceS. The weightλbalances physical efficiency and se-
mantic alignment.
During execution, the system applies differentiated scheduling strategies based
on identified dependencies. Independent tasks trigger parallel retrieval to min-
imize latency, while spatially dependent tasks strictly follow an “anchor-first,
local-expansion” strategy. This hierarchical decomposition and sequential plan-
ning mechanism allows RAGNav to transform complex long-horizon tasks into
a stream of spatiotemporally consistent subgoals, effectively addressing semantic
drift and localization failure in multi-goal navigation.
3.3. Task-Driven Dual-Basis Memory Construction
To achieve structured understanding and efficient querying of complex physi-
cal environments, this framework proposes a task-driven dual-base memory model
that transforms dynamic, multimodal embodied observation data into a unified en-
vironmental representation with both spatial connectivity and semantic hierarchy.
This model consists of a bottom-layer topological map and a top-layer semantic
forest, aiming to provide knowledge representation that supports spatiotemporal
continuous reasoning for multi-target vision-and-language navigation tasks.
Unlike traditional methods that only store isolated semantic labels, the topo-
logical mapG t= (V t, Et)constructed in this framework provides a fundamental
spatial connectivity representation for agent navigation. We extract key positions
pt= (x, y, z)from the pose sequence{ξ t}collected during embodied exploration
as topological nodesv i∈V t. Each node is not only associated with the corre-
sponding visual observationI tbut also generates a refined textual descriptiond t
through a vision-language model (e.g., GPT-4o), forming a "spatial fingerprint"
12

containing semantic information. This fingerprint integrates multi-dimensional
attributes such as pose and visual description, enabling nodes to have their own
semantic labels while implicitly encoding their topological order in the global en-
vironment.
To establish spatial adjacency relationships between nodes, we calculate the
Euclidean distance matrixDfor all node pairs. For a nodev i, if there exists
another nodev jsatisfyingD ij< δ spatial (spatial distance threshold, e.g., 2.0 me-
ters), an undirected edgee ij∈E tis established, whose weightw ij=D ijre-
flects the degree of spatial proximity. This multi-dimensional attribute definition
allows each topological node to not only record its own semantic attributes but
also implicitly encode its topological order in the global environmental network,
providing a foundation for path planning and reachability queries.
On the basis of the topological map, we construct a semantic forestT sto real-
ize hierarchical organization and abstraction of environmental semantic informa-
tion. The construction of the semantic forest is based on a hybrid metric criterion
that fuses spatial proximity and semantic consistency. For any two nodesv iand
vj, their comprehensive similarity is defined as:
Sij=ω·Φ spatial(i, j) + (1−ω)·Ψ semantic (i, j)(5)
whereΦ spatial is the spatial similarity function (e.g., negative exponential form
based on Euclidean distance), andΨ semantic is the semantic similarity function (e.g.,
cosine similarity based on textual description embeddings). Meanwhile, the fused
featureF iof nodev iis obtained by adaptively weighting its spatial featurefspa
i
and semantic featurefsem
i:
13

Fi=α·Norm(fspa
i) + (1−α)·Norm(fsem
i)(6)
The weightαcan be dynamically adjusted according to scene characteristics:
reduceαin semantically dense scenes to enhance semantic discriminability, and
increaseαin structurally complex scenes to highlight spatial layout.
Based on the above similarity or features, an agglomerative hierarchical clus-
tering algorithm is adopted to iteratively merge the most similar nodes or clusters
from bottom to top, forming a multi-level abstract structure of "leaf–subtree–forest".
Each merge generates a higher-level parent node (e.g., merging "chair" and "table"
nodes into a "dining room" node), and a large language model is used to summa-
rize the descriptions of subordinate nodes to automatically generate semantic la-
bels and summaries for the parent node, achieving abstraction and generalization
of semantic levels.
The topological mapG tand semantic forestT sare deeply coupled through
topological connections and semantic indexing, together forming a unified dual-
base environmental memoryM={G t, Ts}. The core advantages of this model
lie in its natural support for multi-target logical dependencies and dynamic query
capabilities:
• Multi-level resolution retrieval: The semantic forest supports fast retrieval
of semantic relevance across granularities. For example, when processing
complex instructions such as "Find a chair near the sofa in the office area",
the system can first lock the target functional area at the macro resolution,
and then conduct local refined screening using the spatial fingerprints of
topological nodes at the micro resolution.
14

• Spatiotemporal collaborative verification: During multi-target retrieval, the
semantic forest is responsible for cross-resolution candidate set pruning to
maintain computational efficiency; the topological map uses physical neigh-
bor relationships between nodes to provide spatial co-occurrence verifica-
tion, effectively alleviating semantic drift and localization failure problems
common in long-range planning.
• Task-adaptive representation: The dual-base memory is no longer a static
stack of features but a dynamically queryable associative network. The
agent can dynamically adjust the search strategy according to the granularity
of instructions, achieving seamless switching from coarse-grained regional
localization to fine-grained object recognition.
3.4. Retrieval Augmentation for Multi-Goal
To address complex spatial relationships among entities in multi-goal vision–language
navigation, we propose a two-stage retrieval and topology-enhanced mechanism
that enables precise mapping from high-dimensional semantic space to low-dimensional
physical space.
3.4.1. Two-Stage Anchor-Guided Retrieval
For spatial composite queries such as “Best sofa near a chair,” the system
executes an anchor-guided conditional retrieval process. In the candidate recall
stage, global semantic similarity is computed for the primary targetA, yielding
the top-KcandidatesC A={v 1, v2, . . . , v K}.
In the neighborhood validation stage, each candidate node is examined to de-
termine whether its one-hop or multi-hop neighborhoodN(v i)contains a node
15

matching auxiliary targetB. A distance-weighted combination score is defined
as:
Scombo(vA, vB) =1
1 +d(v A, vB)(7)
whered(v A, vB)denotes Euclidean distance. Candidates lacking valid auxil-
iary targets are pruned early, eliminating semantic noise that violates spatial con-
straints.
3.4.2. Topological Neighbor Boosting
To further improve localization accuracy, we introduce a co-occurrence-based
enhancement mechanism. When retrieving targetA, if its topological neighbors
include contextually relevant targetsB(e.g., “TV” and “remote”), confidence
propagation is applied. The boosted score is:
Sboost=S sem·(1 +η· ¯Sneighbor )(8)
whereS semis the original semantic score, ¯Sneighbor is the average semantic score
of relevant neighbors, andηis a boosting coefficient. This strategy reduces ambi-
guity in dense or semantically overlapping environments.
3.5. Route Planning and Sequential Navigation
After obtaining the optimal coordinates of all target entitiesP={P 1, P2, . . . , P n},
the system enters the global planning stage to generate a spatiotemporally consis-
tent execution trajectory.
During temporal optimization, the system computes the minimum travel cost
matrix between target nodes using Dijkstra’s algorithm over the topological con-
nectivity graph. Solving this constrained planning problem yields the globally
optimal visiting order. The system then performs instruction-aligned navigation,
16

generating a structured guide containing target markers, relevance explanations,
and navigation cues. This guide drives the local planner to visit targets sequen-
tially, ensuring efficient and stable execution under complex multi-goal instruc-
tions.
4. Experiment
4.1. Experimental Setting
4.1.1. Dataset
Experiments are conducted in the high-fidelity AirSim simulation environ-
ment [33], which supports controlled simulation of various complex scenarios.
Based on this environment, a navigation dataset is constructed, comprising 14
object-centric topological graphs. These topological graphs vary in scale, con-
taining an average of approximately 80 nodes, and simulate the semantic and spa-
tial complexity of typical indoor environments. To validate the capability of the
RAGNav framework in processing multimodal information, the agent’s observa-
tions tis designed to include not only RGB images but also additional sensor data.
This setup aims to simulate real-world scenarios where agents can leverage mul-
tiple specialized sensors (e.g., thermal cameras, depth cameras, spectral sensors)
to enhance their environmental understanding and task execution capabilities.
4.1.2. Baselines
To comprehensively evaluate RAGNav, systematic comparisons are conducted
from two perspectives: retrieval efficiency and navigation performance. In terms
of retrieval efficiency, RAGNav is compared with Naive RAG (flat retrieval) [10],
GraphRAG (graph-structured reasoning) [15], and LightRAG (dual-level key in-
17

dexing) [17]. Regarding navigation performance, comparisons are made with
ReMEmbR [34] and ETPNav [29].
4.1.3. Evaluation Metrics
To comprehensively evaluate the system performance, five metrics are de-
signed for retrieval and navigation tasks. (1) Total Task Time: The total time
elapsed from the moment the system receives a complete natural language in-
struction to the generation of the final navigation path and the end of planning.
(2) Retrieval Time: The time spent by the system to locate the target nodes se-
mantically related to the instruction from the dual-base environmental memory
(topological mapG tand semantic forestT s) when executing a query. This metric
directly reflects the efficiency of the retrieval module. (3) Retrieval Accuracy: It
is used to directly evaluate the precision of the retrieval module, defined as the
proportion of queries where the Top-1 retrieval result returned by the system is
consistent with the ground truthAgiven the ground truthA, i.e.,P(Q|A). This
metric reflects the system’s ability to locate targets from environmental memory.
(4) Task Success Rate: The proportion of complete trajectories where the agent
successfully reaches all specified target points in the instruction within the speci-
fied number of exploration steps. (5) Travel Distance: The total length (in meters)
of the actual motion trajectory taken by the agent to complete the task. For calcu-
lation consistency, the following rules are followed:
Rule 1 (Priority): If the edges of the environmental topological graph have an
accurate distance attribute, the total distanceD totalis the sum of all edge attributes
in the path sequence:
18

Dtotal=n−1X
i=1distance(v i, vi+1)(9)
wherev iandv i+1are two adjacent nodes on the path.
Rule 2 (Fallback): If the distance attribute of the edges in the graph is miss-
ing, the Euclidean distance calculated from the node coordinates is used as an
approximation:
Dtotal=n−1X
i=1p
(xi+1−xi)2+ (y i+1−yi)2 (10)
4.2. Experimental Results
Table 1 presents a performance comparison of different retrieval methods un-
der various input modalities. Overall, RAGNav demonstrates a significant advan-
tage in balancing retrieval accuracy and time efficiency.
Under any data query type, the retrieval accuracy of RAGNav is significantly
better than all baseline methods. The baseline models perform significantly worse
across query types, primarily because segmenting multimodal embodied data into
text often fails to retrieve the correct images, leading to retrieval failure in most
cases. A phenomenon worth exploring in depth is that when location informa-
tion is introduced, the accuracy of all methods decreases, with RAGNav dropping
from 0.46 to 0.21. This initially indicates that raw spatial information may intro-
duce noise or have unoptimized conflicts with semantic features. When sensor
data (e.g., NDVI) is further added, the accuracy of RAGNav recovers to 0.34,
while baseline methods show no improvement. In terms of time cost, RAGNav
(185-195 ms) maintains a level similar to the efficiently designed LightRAG and
is significantly better than GraphRAG (420-430 ms), which suffers from complex
19

graph reasoning overhead. This shows that the retrieval enhancement strategy,
while achieving a substantial improvement in accuracy, avoids an explosion in
computational cost through structured pruning, resulting in excellent comprehen-
sive performance.
Table 1: Comparison of Retrieval Efficiency and Accuracy for Different Methods
Method Input Type Retrieval Time(ms)↓Retrieval Accuracy(%)↑
Naive RAG [10] Text 152 0.08
Text+Location 155 0.03
Text+Location+Sensor 160 0.04
GraphRAG [15] Text 420 0.09
Text+Location 425 0.04
Text+Location+Sensor 430 0.04
LightRAG [17] Text 205 0.17
Text+Location 210 0.09
Text+Location+Sensor 215 0.10
RAGNav (Ours) Text 185 0.46
Text+Location 190 0.21
Text+Location+Sensor 195 0.34
Furthermore, to evaluate the system-level performance of RAGNav as a holis-
tic navigation framework, we benchmark it against two representative baselines
with distinct technical underpinnings: ReMEmbR [34], a memory-augmented ap-
proach based on episodic retrieval, and ETPNav [29], a state-of-the-art paradigm
centered on topological planning. As shown in Table 2, RAGNav achieves the op-
timal values across all three metrics. Its task success rate reaches 65%, surpassing
ReMEmbR and ETPNav by 13 and 23 percentage points, respectively. This sig-
20

nificant improvement indicates that RAGNav provides a substantial contribution
to navigation decision-making. Meanwhile, RAGNav achieves the lowest total
time consumption (30.02 s) and travel distance (16.13 m), reducing these metrics
by approximately 21.9% and 20.5%, respectively, compared to the suboptimal
ETPNav. This demonstrates that RAGNav reduces blind exploration throughout
the entire space and avoids unnecessary detours and backtracking.
Table 2: Performance Comparison of Complete Navigation Systems
Method Total Time(s)↓Travel Distance(m)↓Success Rate SR(%)↑
ReMEmbR [34] 45.67 24.85 0.52
ETPNav [29] 38.41 20.30 0.42
RAGNav (Ours) 30.02 16.13 0.65
4.3. Ablation Study
To clarify the source of performance improvement in the RAGNav framework,
we conducted ablation experiments. Traditional Retrieval-Augmented Generation
(RAG) methods are not designed for multimodal embodied navigation tasks, as
they typically simplify complex spatiotemporal observations into flat text, making
it difficult to perform accurate spatial-semantic joint reasoning in multi-goal VLN
tasks. Therefore, by constructing different model variants, sequentially remov-
ing key modules, and testing their performance on the same multi-goal navigation
tasks, we aim to answer: is the excellent performance of the framework driven
by the dual-basis memory (topological map, semantic forest) or the retrieval en-
hancement strategy (spatial, neighbor dimensions)? After removing a module in
each variant, the simplest direct logic is used to replace its function, ensuring that
21

performance changes can be clearly attributed to the absence of the target module.
The experimental results are shown in Table 3.
The experiments show that removing any memory basis leads to a catastrophic
performance decline, but with different patterns. After removing the semantic
forest, the retrieval accuracy plummets to 15.0%, becoming the main bottleneck.
Table 3: Ablation Study of the RAGNav Framework
Model Variant Retrieval Accuracy(%)↑Travel Distance(m)↓Success Rate(%)↑
w/o Semantic Forest 0.15 21.72 0.28
w/o Topological Map 0.24 24.37 0.21
w/o Spatial Enhancement 0.35 17.91 0.28
w/o Neighbor Enhancement 0.39 17.25 0.31
RAGNav (Ours) 0.46 16.13 0.65
5. Conclusion
We propose RAGNav, a retrieval-augmented topological semantic reasoning
framework that can capture embodied memory at any spatial and semantic resolu-
tion in navigation environments, retrieve and generate responses for navigation re-
quests, and realize the "perception-planning-execution-reflection" closed loop for
navigation. Experimental results show that RAGNav outperforms state-of-the-
art baselines in both retrieval accuracy and navigation performance. This study
improves the multi-objective navigation reasoning method and expands the appli-
cation scenarios of RAG in embodied intelligence. However, the current frame-
work is mainly verified in simulation environments and assumes the existence of a
perfect local planner, and its robustness in complex scenarios such as dynamic ob-
stacle avoidance has not been fully verified. In the future, we will promote the mi-
22

gration to real-world scenarios, remove the dependency on perfect local planning,
and study how to combine the high-level semantic reasoning of this framework
with a more robust and real-time low-level obstacle avoidance controller, so as
to comprehensively improve the practicality and safety in dynamic and uncertain
environments.
References
[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid,
S. Gould, A. Van Den Hengel, Vision-and-language navigation: Interpreting
visually-grounded navigation instructions in real environments, in: Proceed-
ings of the IEEE conference on computer vision and pattern recognition,
2018, pp. 3674–3683.
[2] S. Wani, S. Patel, U. Jain, A. Chang, M. Savva, Multion: Benchmarking
semantic map memory using multi-object navigation, Advances in Neural
Information Processing Systems 33 (2020) 9700–9712.
[3] S. Raychaudhuri, T. Campari, U. Jain, M. Savva, A. X. Chang, Mopa:
Modular object navigation with pointgoal agents, in: Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision, 2024,
pp. 5763–5773.
[4] A. Sadek, G. Bono, B. Chidlovskii, A. Baskurt, C. Wolf, Multi-object navi-
gation in real environments using hybrid policies, in: International Confer-
ence on Robotics and Automation (ICRA), 2023.
[5] C. Gao, J. Chen, S. Liu, L. Wang, Q. Zhang, Q. Wu, Room-and-object
aware knowledge reasoning for remote embodied referring expression, in:
23

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021, pp. 3064–3073.
[6] F. Giuliari, G. Skenderi, M. Cristani, Y . Wang, A. Del Bue, Spatial com-
monsense graph for object localisation in partial scenes, in: Proceedings of
the ieee/cvf conference on computer vision and pattern recognition, 2022,
pp. 19518–19527.
[7] H. Zeng, X. Song, S. Jiang, Multi-object navigation using potential target
position policy function, IEEE Transactions on Image Processing 32 (2023)
2608–2619, IEEE.
[8] R. Dang, Z. Shi, L. Wang, Z. He, C. Liu, Q. Chen, Unbiased directed object
attention graph for object navigation, in: Proceedings of the 30th ACM
International Conference on Multimedia, 2022, pp. 3617–3627.
[9] J. Liu, J. Guo, Z. Meng, J. Xue, Revolt: Relational reasoning and
voronoi local graph planning for target-driven navigation, arXiv preprint
arXiv:2301.02382 (2023).
[10] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W.-t. Yih, T. Rocktäschel, et al., Retrieval-augmented generation
for knowledge-intensive nlp tasks, Advances in neural information process-
ing systems 33 (2020) 9459–9474.
[11] H. Li, Y . Su, D. Cai, Y . Wang, L. Liu, A survey on retrieval-augmented text
generation, arXiv preprint arXiv:2202.01110 (2022).
24

[12] M. Booker, G. Byrd, B. Kemp, A. Schmidt, C. Rivera, Embodiedrag: Dy-
namic 3d scene graph retrieval for efficient and scalable robot task planning,
arXiv preprint arXiv:2410.23968 (2024).
[13] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y . Geng, F. Fu, L. Yang, W. Zhang,
J. Jiang, B. Cui, Retrieval-augmented generation for ai-generated content:
A survey, Data Science and Engineering (2026) 1–29, Springer.
[14] R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding, X. Guo,
M. Li, X. Li, et al., Retrieving multimodal information for augmented gen-
eration: A survey, in: Findings of the Association for Computational Lin-
guistics: EMNLP 2023, 2023, pp. 4736–4756.
[15] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, J. Larson, From local to global: A graph rag
approach to query-focused summarization, arXiv preprint arXiv:2404.16130
(2024).
[16] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, S. Tang, Graph
retrieval-augmented generation: A survey, ACM Transactions on Informa-
tion Systems 44 (2025) 1–52, ACM New York, NY .
[17] Z. Guo, L. Xia, Y . Yu, T. Ao, C. Huang, Lightrag: Simple and fast retrieval-
augmented generation, arXiv preprint arXiv:2410.05779 (2024).
[18] Y . Qi, Q. Wu, P. Anderson, X. Wang, W. Y . Wang, C. Shen, A. v. d. Hen-
gel, Reverie: Remote embodied visual referring expression in real indoor
environments, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2020, pp. 9982–9991.
25

[19] A. Ku, P. Anderson, R. Patel, E. Ie, J. Baldridge, Room-across-room: Mul-
tilingual vision-and-language navigation with dense spatiotemporal ground-
ing, arXiv preprint arXiv:2010.07954 (2020).
[20] P. Chen, D. Ji, K. Lin, W. Hu, W. Huang, T. Li, M. Tan, C. Gan, Learning
active camera for multi-object navigation, Advances in Neural Information
Processing Systems 35 (2022) 28670–28682.
[21] P. Marza, L. Matignon, O. Simonin, C. Wolf, Multi-object navigation with
dynamically learned neural implicit representations, in: Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2023, pp. 11004–
11015.
[22] W. Chen, H. Hu, X. Chen, P. Verga, W. Cohen, Murag: Multimodal retrieval-
augmented generator for open question answering over images and text, in:
Proceedings of the 2022 Conference on Empirical Methods in Natural Lan-
guage Processing, 2022, pp. 5558–5570.
[23] A. Asai, Z. Wu, Y . Wang, A. Sil, H. Hajishirzi, Self-rag: Learning to retrieve,
generate, and critique through self-reflection (2024), ICLR.
[24] Z. Wei, W.-L. Chen, Y . Meng, Instructrag: Instructing retrieval-augmented
generation via self-synthesized rationales, in: The Thirteenth International
Conference on Learning Representations, ????
[25] H. Sun, B. Dhingra, M. Zaheer, K. Mazaitis, R. Salakhutdinov, W. Cohen,
Open domain question answering using early fusion of knowledge bases and
text, in: Proceedings of the 2018 conference on empirical methods in natural
language processing, 2018, pp. 4231–4242.
26

[26] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y . Gong, L. Ni, H.-Y . Shum, J. Guo,
Think-on-graph: Deep and responsible reasoning of large language model
on knowledge graph, in: The Twelfth International Conference on Learning
Representations, ????
[27] R. Zellers, M. Yatskar, S. Thomson, Y . Choi, Neural motifs: Scene graph
parsing with global context, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2018, pp. 5831–5840.
[28] J. Ost, F. Mannan, N. Thuerey, J. Knodt, F. Heide, Neural scene graphs for
dynamic scenes, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2021, pp. 2856–2865.
[29] D. An, H. Wang, W. Wang, Z. Wang, Y . Huang, K. He, L. Wang, Etpnav:
Evolving topological planning for vision-language navigation in continuous
environments, IEEE Transactions on Pattern Analysis and Machine Intelli-
gence (2024), IEEE.
[30] X. Guo, Z. Wu, K. Xiong, Z. Xu, L. Zhou, G. Xu, S. Xu, H. Sun, B. Wang,
G. Chen, et al., Genesis: Multimodal driving scene generation with spatio-
temporal and cross-modal consistency, arXiv preprint arXiv:2506.07497
(2025).
[31] T. Zemskova, D. Yudin, 3dgraphllm: Combining semantic graphs and
large language models for 3d scene understanding, in: Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2025, pp. 8885–
8895.
27

[32] G. Zhao, Y . Li, W. Qi, K. Zhang, B. Liu, K. Chen, H. Li, J. Ma, Slam-free vi-
sual navigation with hierarchical vision-language perception and coarse-to-
fine semantic topological planning, arXiv preprint arXiv:2509.20739 (2025).
[33] S. Shah, D. Dey, C. Lovett, A. Kapoor, Airsim: High-fidelity visual and
physical simulation for autonomous vehicles, in: Field and service robotics:
Results of the 11th international conference, Springer, 2017, pp. 621–635.
[34] A. Anwar, J. Welsh, J. Biswas, S. Pouya, Y . Chang, Remembr: Building
and reasoning over long-horizon spatio-temporal memory for robot naviga-
tion, in: 2025 IEEE International Conference on Robotics and Automation
(ICRA), IEEE, 2025, pp. 2838–2845.
28