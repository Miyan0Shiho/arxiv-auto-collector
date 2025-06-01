# DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation

**Authors**: Tianjun Gu, Linfeng Li, Xuhong Wang, Chenghua Gong, Jingyu Gong, Zhizhong Zhang, Yuan Xie, Lizhuang Ma, Xin Tan

**Published**: 2025-05-28 04:46:13

**PDF URL**: [http://arxiv.org/pdf/2505.21969v2](http://arxiv.org/pdf/2505.21969v2)

## Abstract
Adaptive navigation in unfamiliar environments is crucial for household
service robots but remains challenging due to the need for both low-level path
planning and high-level scene understanding. While recent vision-language model
(VLM) based zero-shot approaches reduce dependence on prior maps and
scene-specific training data, they face significant limitations: spatiotemporal
discontinuity from discrete observations, unstructured memory representations,
and insufficient task understanding leading to navigation failures. We propose
DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory
Oriented Navigation), a novel cognitive-inspired framework consisting of
Ventral and Dorsal Streams that mimics human navigation capabilities. The
Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology
Map to handle spatiotemporal discontinuities, while the Ventral Stream combines
RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops
Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON
on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art
performance on both success rate (SR) and success weighted by path length (SPL)
metrics, significantly outperforming existing methods. We also introduce a new
evaluation metric (AORI) to assess navigation intelligence better.
Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot
autonomous navigation without requiring prior map building or pre-training.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21969v2  [cs.RO]  29 May 2025
DORAEMON: Decentralized Ontology-aware
Reliable Agent with Enhanced Memory Oriented
Navigation
Tianjun Gu1Linfeng Li1Xuhong Wang2Chenghua Gong1Jingyu Gong1
Zhizhong Zhang1Yuan Xie1,3Lizhuang Ma1Xin Tan1,2
1East China Normal University,2Shanghai AI Lab,3Shanghai Innovation Institute
Abstract
Adaptive navigation in unfamiliar environments is crucial for household service
robots but remains challenging due to the need for both low-level path planning and
high-level scene understanding. While recent vision-language model (VLM) based
zero-shot approaches reduce dependence on prior maps and scene-specific training
data, they face significant limitations: spatiotemporal discontinuity from discrete
observations, unstructured memory representations, and insufficient task under-
standing leading to navigation failures. We propose DORAEMON (Decentralized
Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation),
a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams
that mimics human navigation capabilities. The Dorsal Stream implements the
Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal
discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM
to improve decision-making. Our approach also develops Nav-Ensurance to ensure
navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D,
and GOAT datasets, where it achieves state-of-the-art performance on both success
rate (SR) and success weighted by path length (SPL) metrics, significantly outper-
forming existing methods. We also introduce a new evaluation metric (AORI) to
assess navigation intelligence better. Comprehensive experiments demonstrate DO-
RAEMON’s effectiveness in zero-shot autonomous navigation without requiring
prior map building or pre-training.
1 Introduction
Adaptive navigation in complex and unseen environments [ 2] is a key capability for household service
robots. This task requires robots to move from a random starting point to the location of a target
object without any prior knowledge of the environment. For humans, navigation appears almost
trivial thanks to spatial cognitive abilities honed through evolution. For robots, however, navigation
remains a highly challenging problem: it demands not only low-level path planning to avoid obstacles
and reach the destination, but also high-level scene understanding to interpret and make sense of the
surrounding environment.
Most existing navigation methods rely on the construction of prior maps[ 4] or require extensive
scene-specific data for task-oriented pre-training[ 36]. However, in entirely unfamiliar environments,
building maps in advance can be time-consuming and labor-intensive, and the scarcity of scene data
further limits the practicality of these approaches. Recently, some works [ 14,34] have begun to
Homepage: https://grady10086.github.io/DORAEMON/

Figure 1: (a) Illustrates limitation of typical VLM navigation (red arrow) in an unfamiliar environment.
(b) DORAEMON’s cognitive inspiration: a Decentralized Ontology-aware approach with a Dorsal
Stream for “What” and a Ventral Stream for “Where”. (c) DORAEMON constructs the Topology
Map and uses Hierarchical Semantic-Spatial Fusion(green arrow).
explore zero-training and zero-shot navigation strategies based on vision-language models (VLMs).
Relying on textual descriptions of the current task, image inputs, and previously observed historical
information, these approaches achieve navigation without dependence on environment- or task-
specific data, gradually shedding the reliance on scene priors.
Although VLM-based zero-shot navigation methods offer a novel perspective for adapting to unfa-
miliar environments, they still face numerous challenges in practical applications. On the one hand,
VLMs typically take task descriptions and observation histories as input [ 34][14]. However, due to
the discrete nature of input image descriptions at each time step, this spatiotemporal discontinuity
often makes it difficult for VLMs to understand the relationships between targets and obstacles in
complex environments. On the other hand, while many existing navigation systems incorporate some
form of memory functionality, most VLM methods [ 25,27,30] adopt a single-step decision paradigm,
treating historical information merely as a reference log. Even though End-to-End methods like
VLMnav[ 14] utilize historical information, they typically store this information in a flat, unstructured
manner, which fundamentally limits their ability to perform long-range navigation. Additionally,
VLMs sometimes insufficient understanding of task semantics often leads to poor decision-making
(e.g., going to look for a TV but finding a computer instead), and the lack of reliable check mecha-
nisms for navigation states frequently results in unreliable behaviors such as spinning in place during
navigation tasks. Fig 1 conceptually illustrates limitations of traditional VLN methods and contrasts
them with the cognitive-inspired approach of our DORAEMON.
Inspired by cognitive science ‘Decentralized Ontology’ principles [ 3] suggesting human knowledge is
often represented and accessed in a distributed and context-dependent manner rather than via a single
monolithic structure, we propose the Decentralized Ontology-aware Reliable Agent with Enhanced
Memory Oriented Navigation (DORAEMON), which consists mainly of Ventral Stream and Dorsal
Stream. The Ventral Stream processes object identity (“what”) information, while the Dorsal Stream
handles spatial (“where”) processing in the human brain.
Our decentralized architecture distributes cognitive functions across complementary streams: The
Dorsal Stream addresses spatio-temporal discontinuities through a Topology Map and a Hierar-
chical Semantic-Spatial Fusion, allowing our agent to reason accurately about target-environment
relationships. Additionally, the Ventral Stream improves task understanding by utilizing a Retrieval-
augmented Generation model (RAG-VLM) and Policy-VLM for navigation. Additionally, DORAE-
MON features a Nav-Ensurance system that enables agents to autonomously detect and respond to
2

abnormal conditions, such as becoming stuck or blocked during navigation. To evaluate navigation
performance more comprehensively, we propose a new metric called the Adaptive Online Route
Index (AORI).
In summary, the main contributions of this work are:
•We propose DORAEMON, a novel adaptive navigation framework inspired by cognitive
principles of decentralized knowledge, consisting of ventral and Dorsal Streams, enabling
End-to-End and zero-shot navigation in completely unfamiliar environments without pre-
training.
•We propose the Dorsal Stream, which involves designing a Topology Map and a Hierarchical
Semantic-Spatial Fusion Network to effectively manage spatio-temporal discontinuities.
Additionally, we introduce the Ventral Stream, incorporating a synergistic reasoning com-
ponent that combines RAG-VLM for understanding ontological tasks and Policy-VLM for
enhanced task comprehension and policy planning.
•We develop Nav-Ensurance, which includes multi-dimensional stuck detection and context-
aware escape mechanisms. We propose a new evaluation metric called AORI to quantify the
efficiency of the agent’s exploration. Our method demonstrates state-of-the-art performance
across various navigation tasks.
2 Related Work
2.1 Zero-shot Object Goal Navigation
Object navigation methods are broadly supervised or zero-shot. Supervised approaches train visual
encoders with reinforcement/imitation learning [ 11,17,23,31] or build semantic maps from training
data [ 24,45,49], struggling with novel scenarios due to data dependency. Zero-shot methods address
this using open-vocabulary understanding, increasingly leveraging foundation models like Vision-
Language Models (VLMs) and Large Language Models (LLMs). LLMs provide commonsense
reasoning via object-room correlation [ 39,41,51], semantic mapping [ 43], and chain-of-thought
planning [ 5,6,33,41], while VLMs align visual observations with textual goals. These foundation
model-guided techniques include image-based methods mapping targets to visual embeddings [ 1,
13,22,38] and map-based approaches using frontier [ 10,19,33,43,45,50,51] or waypoint-based
maps [ 39] with LLM/VLM reasoning. VLM-based strategies either use VLMs for recognition
with traditional planning and extra perception models [ 21,27,37,44,46], or, like PIVOT [ 25]
and VLMnav [ 14], directly produce actions End-to-End via visual prompting. Despite progress,
many zero-shot methods, especially those processing observations independently, face challenges
integrating temporal information and handling complex spatial reasoning in unfamiliar environments.
2.2 Memory Mechanisms in Navigation
Memory representations in navigation systems have evolved through various architectures, including
episodic buffers that maintain observation sequences [ 14,16,34], spatial representations prioritizing
geometric information [ 46,50], graph-based semantic structures capturing object relationships [ 41],
and predictive world models attempting to forecast environmental states [ 7,26]. These systems
typically process semantic and spatial information separately, with limited integration between
perception and reasoning modules. Most approaches focus on either building representations or
enhancing reasoning mechanisms independently. Differently, DORAEMON integrates these aspects
through a hierarchical semantic-spatial fusion network with bidirectional information flow between
ventral and dorsal processing streams.
2.3 Cognitive Neuroscience Inspiration in Navigation
Object navigation systems are influenced by cognitive neuroscience, especially Decentralized
Ontology[ 3], which suggests that human knowledge is organized through interconnected cognitive
systems that enable context-dependent reasoning. Recent models like CogNav[ 7] and BrainNav[ 20]
incorporate cognitive elements, but they do not fully embody Decentralized Ontology. CogNav
utilizes a finite state machine for cognitive states, but may have limitations in knowledge integra-
3

Figure 2: Architecture of the DORAEMON Navigation Framework.
tion. BrainNav mimics biological functions but doesn’t deeply engage in decentralized information
processing.
In contrast, DORAEMON aims for a more comprehensive Decentralized Ontology-aware processing.
It emphasizes the integration and bidirectional exchange of information between Dorsal Stream
and Ventral Stream, allowing for the construction of semantic relationships that enhance spatial
understanding and support flexible, context-aware navigation.
3 Methods
Task Formulation We address the ObjectNav task [ 2], where an agent, starting from an initial pose,
must locate and navigate to a target object within a previously unseen indoor environment. At step t,
the agent receives observation It, current pose Ptand a task specification T, which can be either a
simple object category (e.g., “sofa”) or an instruction (e.g., “find the red chair” or“the plant on the
desk”) for tasks like GOAT [ 18]. Based on these inputs, the agent must decide on an action at. While
many prior works utilize a discrete action space, our End-to-End framework employs a continuous
action representation in polar coordinates (rt, θt), where rtspecifies the forward distance to move,
andθtdenotes the change in orientation. Crucially, the action space also includes a stop action.
The task is considered successful if the agent executes the stop action after meeting successive stop
triggers in steps tandt+ 1. The trigger occurs when 1) the agent is within a predefined distance
threshold dsuccess of the target object; 2) the target object is visually confirmed within the agent’s
current observation It.
Methods Overview Our DORAEMON framework achieves End-to-End and zero-shot navigation
through the ontology of two decentralized cognitive-inspired streams, as depicted in Figure 2. Given
an input with a panoramic image Itand a pose Ptat step t, they are processed by the Action Proposer
Module (Appendix A) and the Dorsal Stream Module (Section 3.1), respectively. In the Action
Proposer Module, a candidate image It
annois generated with a set of action candidates At
finalthrough
parameterized geometric reasoning. Concurrently, the Dorsal Stream extracts semantic and spatial
information from Itusing Hierarchical Semantic-Spatial Fusion and stores it within the Topology Map
as node vt. The relevant node vrelecan be accessed by op-down retrieval. After that, vreleandIt
annoare
input to the Policy-VLM to select the best action based on the given information(Section 3.2.2). At
the same time, the Policy-VLM receives a database containing information on several key dimensions
relevant to the navigation task T, which is generated by the RAG-VLM (Section3.2.1) in the Ventral
Stream Module (Section 3.2). The Policy-VLM integrates the information through a chain of thought
(Appendix H), identifies abnormal conditions (Section 3.3), and outputs the final choice action at.
The agent performs this action atin the environment, navigates, and makes the next decision at step
t+ 1.
4

Figure 3: Architecture of Topological Map and Hierarchical Construction built in Dorsal Stream for
spatio-temporal memory. The top view in the middle shows the content of different nodes during
navigation, and the upper right part represents the Hierarchical Construction of a node.
3.1 Dorsal Stream
The Dorsal Stream, similar to the “where/how” pathway in cognition, is responsible for processing
the spatial information to effectively navigate. As illustrated in Figure 3, at each step t, the agent
constructs vkon the Topology Map (Section 3.1.1). Subsequently, the Hierarchical Semantic-Spatial
Fusion (Section 3.1.2) organizes the information into a hierarchical structure from the bottom up.
3.1.1 Topological Map
The topological map, defined as G= (V,E), represents the environment and incorporates historical
observations. This map is built incrementally as the agent explores. Each node vt∈ V in the
topological map formally integrates multimodal observations as:
vt= (pt, qt, It, Lt, ot,st), (1)
where pt,qt,It,Lt,ot,stcorrespond to the agent’s position, orientation represented as unit quaternion
from agent pose Pt, visual observation, language description of It, target likelihood estimation, and
optional semantic embedding (e.g., CLIP features).The topological map Gis built progressively
through spatio-temporal criteria. A new node vnewis added to the node set Vif either of the
following conditions is met: 1) Temporal Criterion: A fixed number of exploration steps Supdate
have elapsed since the last node addition: tcurr−tprev≥Supdate , ensuring periodic state updates; 2)
Spatial Criterion: The agent’s displacement from the last node exceeds a spatial sampling threshold:
∥pcurr−pprev∥2> δ sample,. where δsample controls exploration granularity. When vnewis added to V, it
is immediately connected to the nearest node vprev.
3.1.2 Hierarchical Semantic-Spatial Fusion
Hierarchical Construction. Building upon the information associated with the Topological Map
nodes vt∈ V, our module organizes information of vtinto a hierarchical structure. The nodes hjon
the hierarchical structure are defined as:
hj=
idj, lj,Pj,Cj
, (2)
where idj,lj∈ {L0, L1, L2, L3},Pj,Cjcorrespond to unique string identifier, hierarchy level tag,
parent node references, and child node references.
5

Figure 4: The structure RAG-VLM in Ventral Stream, handling the task “NA VIGATE TO THE
NEAREST SOFA”.
The memory hierarchy organizes nodes hjinto four semantic levels through structural and functional
relationships (Appendix F): L3(Observation, directly linked to topological map nodes vt),L2(Area),
L1(Room), L0(Environment). The memory hierarchy is constructed bottom-up ( L3→L2→
L1→L0) after an initial exploration phase or periodically. While the overall process involves
sequential clustering or integration steps for each level transition, the specific logic and parameters
differ between levels.
Hierarchical Memory Retrieval. To efficiently find relevant information within the constructed
hierarchy (e.g., observations related to sofa), the system employs a top-down search, conceptually
outlined in the AppendixG. This search is guided by a scoring function S(ni)evaluated at nodes hi
during traversal the constructed hierarchy:
S(hi) =αsemSsemantic (hi, T) +αspaSspatial(hi) +αkeySkeyword (hi, T) +αtimeStime(hi),(3)
where Ssemantic computes embedding similarity between node niand task T,Sspatial measures proxim-
ity to current position, Skeyword evaluates keyword overlap, and Stimeprioritizes recent observations.
The weights αbalance these components based on their relative importance. To manage computa-
tional cost, the retrieval process incorporates beam search, expanding only the top-scoring nodes at
each level.
3.2 Ventral Stream
The Ventral Stream, analogous to the “what” pathway in human cognition, integrates two key
components: RAG-VLM (Section 3.2.1) for semantic knowledge processing and Policy-VLM
(Section 3.2.2) for decision-making.
3.2.1 RAG-VLM
To build a comprehensive understanding of the task, RAG-VLM leverages the extensive world
knowledge embedded within a vision-language model. Upon receiving the task T, the system extracts
key semantic attributes (general description, appearance features, structure/shape, and common
location) to efficiently build the task database.
Figure 4 illustrates this knowledge extraction process. The structured information gathered across
these dimensions forms a database of the task T. This representation enables the agent not only
6

to verify whether an object encountered during navigation matches the task description but also to
potentially inform planning by suggesting likely areas to explore first, thereby interfacing with the
spatial reasoning components of the Dorsal Stream.
3.2.2 Policy-VLM
The Policy-VLM combines visual observations, spatial awareness, and task semantics to determine
optimal actions. It utilizes the reasoning capabilities of large vision-language models through Chain-
of-Thought (CoT). The CoT breaks down the complex navigation task into interpretable sub-steps:
current state analysis, memory integration, goal analysis, scene assessment, path planning, and action
decision.
3.3 Nav-Ensurance
To enhance the evaluation of safety and efficiency in navigation, we present a new metric, the
Area Overlap Redundancy Index (AORI) (Section 3.3.1). Additionally, we develop Nav-Ensurance,
including Multimodal Stuck Detection (Section 3.3.2), context-aware escape strategies (Section 3.3.3),
and adaptive precision navigation (Section 3.3.4), to ensure that navigation systems operate reliably
and effectively.
3.3.1 Area Overlap Redundancy Index (AORI)
We introduce the Area Overlap Redundancy Index (AORI) to quantify the efficiency of the agent’s
navigation strategy by measuring overlap in area coverage. A high AORI indicates excessive path
overlap and inefficient exploration, specifically addressing the limitations of conventional coverage
metrics that neglect temporal-spatial redundancy. AORI is formally defined as:
AORI = 1.0−(wc·(1.0−roverlap)2+wd·(1.0−dnorm)), (4)
where roverlap represents the ratio of revisited areas to total observed areas, dnormis the normalized
density, and wc= 0.8, wd= 0.2are weighting coefficients. For further details, refer to the
Appendix D.
3.3.2 Multimodal Stuck Detection
Agent detects navigation stuck by analyzing metrics calculated over a sliding window of step T:
η=∥pT−p0∥2PT
t=1∥pt−pt−1∥2, ρ =PT
t=1|θt−θt−1|
PT
t=1∥pt−pt−1∥2. (5)
A weighted scoring function combines these metrics:
S=wη·I[η < τ η] +wρ·I[ρ > τ ρ]. (6)
Agent confirms stuck when S≥Sthpersists for kconsecutive windows. This formula detects whether
the agent is stuck or spinning during the navigation process through ηandρ.
3.3.3 Context-aware Escape Strategies
When a stuck state is detected, the system selects an appropriate escape strategy based on the
perceived information from Dorsal Stream(Section 3.1). For instance, in corner traps (perceived dead
ends), a large turn (near 180◦) is executed. In narrow passages, a small backward step followed by a
randomized direction change is employed. If the environmental context is ambiguous, the agent will
analyze recent successful movement directions and attempt to move perpendicularly, significantly
improving escape capabilities from complex trap situations.
7

3.3.4 Adaptive Precision Navigation
As the agent nears the target object, it will activate a precision navigation mode. In this mode,
the distance component dof all proposed actions (d, θ)is scaled down by a factor γstepto enable
fine-grained positioning adjustments:
aprecise = (d·γstep, θ)for action (d, θ)∈Aactions. (7)
Additionally, when activating the precision navigation mode, the system can utilize visual analysis
(using VLM) to create more detailed action options, thereby maximizing final positioning accuracy
relative to the task.
4 Experiments
Datasets We evaluate our proposed DORAEMON within the Habitat simulator [ 32] on three
large-scale datasets: HM3Dv1 (using HM3D-Semantics-v0.1 [ 29] from the 2022 Habitat Challenge,
featuring 2000 episodes across 20 scenes with 6 goal categories), HM3Dv2 (using HM3D-Semantics-
v0.2 [ 40] from the 2023 Habitat Challenge, with 1000 episodes across 36 scenes and 6 goal categories),
and MP3D [ 8] from the 2021 Habitat Challenge, comprising 2195 episodes across 11 scenes with
21 goal categories. We also include evaluations on GOAT [ 18] (using HM3D-Semantics-v0.2), a
benchmark focusing on generalized object semantics with 1000 validation episodes across 100 scenes
and 25 object categories.
Implement Details and Evaluation Metrics The action space includes a stop action, a
move_forward action where the distance parameter is sampled from the continuous range
[0.5m,1.7m], and a rotate action. We adopt standard metrics to evaluate navigation performance:
Success Rate (SR), the percentage of episodes where the agent successfully stops near a target
object; Success weighted by Path Length (SPL), defined as1
NPN
i=1Sili
max( pi,li), rewarding both
success and efficiency; and our proposed Area Overlap Redundancy Index (AORI) (Equation (4)),
which quantifies navigation intelligence by penalizing redundant exploration (lower is better). More
information is set in the Appendix E.
Baselines We compare DORAEMON against several state-of-the-art object navigation methods
on the HM3Dv2[ 40], HM3Dv1[ 29], and MP3D[ 8]. Our main comparison focuses on End-to-End
Vision-Language Model (VLM) approaches [14, 25]. Beyond these direct End-to-End counterparts,
we also consider a broader set of recent methods for non-End-to-End object navigation methods.
More baseline details are set in the Appendix I.
4.1 Methods Comparision
End-to-End Methods: We evaluate our approach on the HM3Dv2 (ObjectNav,val, Table 1 (a))
and HM3Dv1(GOAT, val, Table 1 (b)) with other end-to-end baselines. DORAEMON achieves
state-of-the-art performance on both datasets, outperforming other methods by a significant margin.
Table 1: Comparison of End-to-End navigation methods on different benchmarks.
(a) HM3Dv2 ObjectNav benchmark
Method SR (%) ↑ SPL (%) ↑ AORI (%) ↓
Prompt-only 29.8 0.107 -
PIVOT[25] 24.6 10.6 63.3
VLMNav[14] 51.6 18.3 61.5
DORAEMON (Ours) 62.0 23.0 50.1
Improvement 20.2 10.0 18.5(b) GOAT benchmark
Method SR (%) ↑ SPL (%) ↑ AORI (%) ↓
Prompt-only 11.3 3.7 -
PIVOT[25] 8.3 3.8 64.9
VLMNav[14] 22.1 9.3 63.6
DORAEMON (Ours) 24.3 10.3 56.9
Improvement 10.0 10.8 10.5
Comprehensive Methods Analysis: To ensure a fair comparison with the above non-End-to-End
methods that utilize a discrete action set A:move forward 0.25m, turn left/turn right 30◦,
look up/lookdown 30◦,stop , and a common 500 steps episode limit, we conduct an additional
set of experiments. In these, we normalize our agent’s interactions by approximating an equivalent
number of standard discrete steps for each of DORAEMON’s actions.
8

Compared to the non-End-to-End approach in the Table 2, DORAEMON achieves state-of-the-
art performance on SR, despite normalizing our action to set A. Each action performed by ours
corresponds to several actions in this set (details are provided in the Appendix B).
Table 2: Comprehensive comparison with state-of-the-art methods on ObjectNav benchmarks. TF refers to
training-free, ZS refers to zero-shot, and E2E refers to End-to-End.
Method ZS TF E2E HM3Dv1 HM3Dv2 MP3D
SR(%)↑SPL(%) ↑SR(%)↑SPL(%) ↑SR(%)↑SPL(%) ↑
ProcTHOR [12] × × × 54.4 31.8 - - - -
SemEXP [9] ✓× × - - - - 36.0 14.4
Habitat-Web[31] ✓× × 41.5 16.0 - - 31.6 8.5
PONI [28] ✓× × - - - - 31.8 12.1
ProcTHOR-ZS [12] ✓× × 13.2 7.7 - - - -
ZSON [22] ✓× × 25.5 12.6 - - 15.3 4.8
PSL [35] ✓× × 42.4 19.2 - - - -
Pixel-Nav [5] ✓× × 37.9 20.5 - - - -
SGM [47] ✓× × 60.2 30.8 - - 37.7 14.7
ImagineNav [48] ✓× × 53.0 23.8 - - - -
CoW [13] ✓ ✓ × - - - - 7.4 3.7
ESC [51] ✓ ✓ × 39.2 22.3 - - 28.7 14.2
L3MVN [43] ✓ ✓ × 50.4 23.1 36.3 15.7 34.9 14.5
VLFM [42] ✓ ✓ × 52.5 30.4 63.6 32.5 36.4 17.5
V oroNav [39] ✓ ✓ × 42.0 26.0 - - - -
TopV-Nav [50] ✓ ✓ × 52.0 28.6 - - 35.2 16.4
SG-Nav [41] ✓ ✓ × 54.0 24.9 49.6 25.5 40.2 16.0
DORAEMON (Ours) ✓ ✓ ✓ 55.6 21.4 66.5 20.6 41.1 15.8
Ablation Studies: 1) The effect of different modules: To represent the contribution of each module,
we compared three variants (Dorsal Stream, RAG-VLM of Ventral Stream, and Policy-VLM of
Ventral Stream) on HM3D v2. Removing the Dorsal Stream and RAG-VLM implies that the model
relies solely on the Policy-VLM of the Dorsal Stream in decision-making. The results reported for
SR, SPL, and AORI, as presented in Table 3(a), respectively, show the effectiveness of Dorsal Stream
and Ventral Stream. 2) We further evaluated the performance of different Visual Language Models
(VLMs), as shown in Table 3(b). The Gemini-1.5-Pro demonstrated outstanding capabilities in this
task. Even when using a smaller model, our approach yielded excellent results, indicating that our
framework is effective on its own rather than solely relying on the reasoning capabilities of the VLMs.
More importantly, as VLMs continue to evolve, the effectiveness of our plug-and-play approach will
also improve.
Table 3: Ablation of HM3Dv2 (100 episodes)
(a) Ablation of different modules
Method SR (%) ↑ SPL (%) ↑ AORI (%) ↓
w/o Dorsal & Ventral Stream 51.6 18.3 61.5
w/o Dorsal & RAG-VLM 54.0 19.8 59.1
w/o Dorsal Stream 59.0 22.7 56.3
w/o Nav-Ensurance 60.0 22.5 54.9
DORAEMON 61.0 23.7 48.8(b) Ablation of different VLMs
VLM SR (%) ↑ SPL (%) ↑ AORI (%) ↓
Qwen-7B 49.5 20.6 68.7
Gemini-1.5-Flash 58.0 20.1 54.8
Gemini-2-Flash 59.0 21.5 57.9
Gemini-1.5-Pro 61.0 23.7 48.8
5 Conclusion
In this paper, we present DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced
Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal
Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical
Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the
Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach
also develops Nav-Ensurance to ensure navigation safety and efficiency. Extensive experimental
results demonstrate the superior performance of DORAEMON.
9

References
[1]Ziad Al-Halah, Santhosh K. Ramakrishnan, and Kristen Grauman. Zero experience required: Plug & play
modular transfer learning for semantic visual navigation, 2022.
[2]Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi, Oleksandr Maksymets, Roozbeh Mottaghi, Manolis
Savva, Alexander Toshev, and Erik Wijmans. Objectnav revisited: On evaluation of embodied agents
navigating to objects. CoRR , abs/2006.13171, 2020.
[3]Paolo Bouquet, Fausto Giunchiglia, Frank Van Harmelen, Luciano Serafini, and Heiner Stuckenschmidt.
Contextualizing ontologies. Journal of Web Semantics , 1(4):325–343, 2004.
[4]Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Chai, Davide Scaramuzza, John Leonard, Ian Reid,
and Simon Henein. Past, present, and future of simultaneous localization and mapping: Toward the
robust-perception age. IEEE Transactions on Robotics , 32(6):1309–1332, 2016.
[5]Wenzhe Cai, Siyuan Huang, Guangran Cheng, Yuxing Long, Peng Gao, Changyin Sun, and Hao Dong.
Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill, 2023.
[6]Yuxin Cai, Xiangkun He, Maonan Wang, Hongliang Guo, Wei-Yun Yau, and Chen Lv. Cl-cotnav: Closed-
loop hierarchical chain-of-thought for zero-shot object-goal navigation with vision-language models,
2025.
[7]Yihan Cao, Jiazhao Zhang, Zhinan Yu, Shuzhen Liu, Zheng Qin, Qin Zou, Bo Du, and Kai Xu. Cognav:
Cognitive process modeling for object goal navigation with llms, 2025.
[8]Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran
Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments,
2017.
[9]Devendra Singh Chaplot, Dhiraj Gandhi, Abhinav Gupta, and Ruslan Salakhutdinov. Object goal navigation
using goal-oriented semantic exploration, 2020.
[10] Junting Chen, Guohao Li, Suryansh Kumar, Bernard Ghanem, and Fisher Yu. How to not train your dragon:
Training-free embodied object goal navigation with semantic frontiers, 2023.
[11] Peihao Chen, Dongyu Ji, Kunyang Lin, Weiwen Hu, Wenbing Huang, Thomas H. Li, Mingkui Tan, and
Chuang Gan. Learning active camera for multi-object navigation, 2022.
[12] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric
Kolve, Ali Farhadi, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai
using procedural generation, 2022.
[13] Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Ilharco, Ludwig Schmidt, and Shuran Song. Cows on
pasture: Baselines and benchmarks for language-driven zero-shot object navigation, 2022.
[14] Dylan Goetting, Himanshu Gaurav Singh, and Antonio Loquercio. End-to-end navigation with vision lan-
guage models: Transforming spatial reasoning into question-answering. arXiv preprint arXiv:2411.05755 ,
2024.
[15] Dylan Goetting, Himanshu Gaurav Singh, and Antonio Loquercio. End-to-end navigation with vision
language models: Transforming spatial reasoning into question-answering, 2024.
[16] Hao-Lun Hsu, Qiuhua Huang, and Sehoon Ha. Improving safety in deep reinforcement learning using
unsupervised action planning, 2021.
[17] Apoorv Khandelwal, Luca Weihs, Roozbeh Mottaghi, and Aniruddha Kembhavi. Simple but effective:
Clip embeddings for embodied ai, 2022.
[18] Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew
Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. Goat-bench: A
benchmark for multi-modal lifelong navigation, 2024.
[19] Yuxuan Kuang, Hai Lin, and Meng Jiang. Openfmnav: Towards open-set zero-shot object navigation via
vision-language foundation models. 2024.
[20] Luo Ling and Bai Qianqian. Endowing embodied agents with spatial reasoning capabilities for vision-and-
language navigation, 2025.
10

[21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei
Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for
open-set object detection, 2024.
[22] Arjun Majumdar, Gunjan Aggarwal, Bhavika Devnani, Judy Hoffman, and Dhruv Batra. Zson: Zero-shot
object-goal navigation using multimodal goal embeddings, 2023.
[23] Oleksandr Maksymets, Vincent Cartillier, Aaron Gokaslan, Erik Wijmans, Wojciech Galuba, Stefan Lee,
and Dhruv Batra. Thda: Treasure hunt data augmentation for semantic navigation. 2021.
[24] So Yeon Min, Devendra Singh Chaplot, Pradeep Ravikumar, Yonatan Bisk, and Ruslan Salakhutdinov.
Film: Following instructions in language with modular methods, 2022.
[25] Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess,
Ayzaan Wahid, Zhuo Xu, Quan Vuong, Tingnan Zhang, Tsang-Wei Edward Lee, Kuang-Huei Lee, Peng
Xu, Sean Kirmani, Yuke Zhu, Andy Zeng, Karol Hausman, Nicolas Heess, Chelsea Finn, Sergey Levine,
and Brian Ichter. Pivot: Iterative visual prompting elicits actionable knowledge for vlms, 2024.
[26] Dujun Nie, Xianda Guo, Yiqun Duan, Ruijun Zhang, and Long Chen. Wmnav: Integrating vision-language
models into world models for object goal navigation, 2025.
[27] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision
language models are blind: Failing to translate detailed visual features into words, 2025.
[28] Santhosh Kumar Ramakrishnan, Devendra Singh Chaplot, Ziad Al-Halah, Jitendra Malik, and Kristen
Grauman. Poni: Potential functions for objectgoal navigation with interaction-free learning, 2022.
[29] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Austin Clegg, John M Turner, Manolis
Savva, Angel X Chang, and Dhruv Batra. Habitat-Matterport 3D Dataset (HM3D): 1000 large-scale 3D
environments for embodied AI. In Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV) , pages 16203–16213, 2021.
[30] Santhosh Kumar Ramakrishnan, Erik Wijmans, Philipp Kraehenbuehl, and Vladlen Koltun. Does spatial
cognition emerge in frontier models?, 2025.
[31] Ram Ramrakhya, Eric Undersander, Dhruv Batra, and Abhishek Das. Habitat-web: Learning embodied
object-search strategies from human demonstrations at scale, 2022.
[32] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian
Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A platform for
embodied AI research. In Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV) , pages 9339–9347, 2019.
[33] Dhruv Shah, Michael Equi, Blazej Osinski, Fei Xia, Brian Ichter, and Sergey Levine. Navigation with
large language models: Semantic guesswork as a heuristic for planning, 2023.
[34] Dhruv Shah, Michael Yang, Michael Laskin, Pieter Abbeel, and Sergey Levine. LM-Nav: Robotic
navigation with large pre-trained models of language, vision, and action. In Conference on Robot Learning
(CoRL) , pages 1083–1093. PMLR, 2023.
[35] Xinyu Sun, Lizhao Liu, Hongyan Zhi, Ronghe Qiu, and Junwei Liang. Prioritized semantic learning for
zero-shot instance navigation, 2024.
[36] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John M Turner, Manolis Savva,
Angel X Chang, and Dhruv Batra. Habitat 2.0: Training home assistants to rearrange their habitat. In
Advances in Neural Information Processing Systems (NeurIPS) , volume 34, pages 30153–30168, 2021.
[37] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. Yolov7: Trainable bag-of-freebies
sets new state-of-the-art for real-time object detectors, 2022.
[38] Congcong Wen, Yisiyuan Huang, Hao Huang, Yanjia Huang, Shuaihang Yuan, Yu Hao, Hui Lin, Yu-Shen
Liu, and Yi Fang. Zero-shot object navigation with vision-language models reasoning, 2024.
[39] Pengying Wu, Yao Mu, Bingxian Wu, Yi Hou, Ji Ma, Shanghang Zhang, and Chang Liu. V oronav:
V oronoi-based zero-shot object navigation with large language model, 2024.
[40] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron
Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, Alexander William Clegg, and
Devendra Singh Chaplot. Habitat-matterport 3d semantics dataset, 2023.
11

[41] Hang Yin, Xiuwei Xu, Zhenyu Wu, Jie Zhou, and Jiwen Lu. Sg-nav: Online 3d scene graph prompting for
llm-based zero-shot object navigation, 2024.
[42] Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, and Bernadette Bucher. Vlfm: Vision-
language frontier maps for zero-shot semantic navigation, 2023.
[43] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. L3mvn: Leveraging large language models for visual
target navigation. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) ,
page 3554–3560. IEEE, October 2023.
[44] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon
Hong. Faster segment anything: Towards lightweight sam for mobile applications, 2023.
[45] Lingfeng Zhang, Xiaoshuai Hao, Qinwen Xu, Qiang Zhang, Xinyao Zhang, Pengwei Wang, Jing Zhang,
Zhongyuan Wang, Shanghang Zhang, and Renjing Xu. Mapnav: A novel memory representation via
annotated semantic maps for vlm-based vision-and-language navigation, 2025.
[46] Mingjie Zhang, Yuheng Du, Chengkai Wu, Jinni Zhou, Zhenchao Qi, Jun Ma, and Boyu Zhou. Apexnav:
An adaptive exploration strategy for zero-shot object navigation with target-centric semantic fusion, 2025.
[47] Sixian Zhang, Xinyao Yu, Xinhang Song, Xiaohan Wang, and Shuqiang Jiang. Imagine before go: Self-
supervised generative map for object goal navigation. In 2024 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) .
[48] Xinxin Zhao, Wenzhe Cai, Likun Tang, and Teng Wang. Imaginenav: Prompting vision-language models
as embodied navigator through scene imagination, 2024.
[49] Kaizhi Zheng, Kaiwen Zhou, Jing Gu, Yue Fan, Jialu Wang, Zonglin Di, Xuehai He, and Xin Eric Wang.
Jarvis: A neuro-symbolic commonsense reasoning framework for conversational embodied agents, 2022.
[50] Linqing Zhong, Chen Gao, Zihan Ding, Yue Liao, Huimin Ma, Shifeng Zhang, Xu Zhou, and Si Liu.
Topv-nav: Unlocking the top-view spatial reasoning potential of mllm for zero-shot object navigation,
2025.
[51] Kaiwen Zhou, Kaizhi Zheng, Connor Pryor, Yilin Shen, Hongxia Jin, Lise Getoor, and Xin Eric Wang.
Esc: Exploration with soft commonsense constraints for zero-shot object navigation, 2023.
12

A Action Proposer
DORAEMON employs an Action Proposer[ 15] to generate a refined set of candidate actions, which the Policy-
VLM then evaluates for the final action decision. As shown in Figure 5, first parameterized action candidates At
init
are generated by the parameterized action space (Equation (8)). Second, adaptive filtering (Equation (9)) refines
At
candusing exploration state Vtand historical patterns Ht. Safety-critical recovery (Equation (10)) enforces a
rotation cooldown γthrough viability evaluation F(·). Finally, the projection module visually encodes At
final
intoIt
annowith numeric tagging (0 for rotation) to interface with VLM’s semantic space.
Figure 5: Action proposal: (a) Collision-free action generation within ±θmaxFOV , (b) Exploration-
aware filtering with ∆θangular resolution, (c) Safety-constrained and action projection.
Parameterized Action Space Define the action space through symbolic parameters:
At
init=
(θi,min ( ηri, rmax))θi=k∆θ, k∈ K
. (8)
whereK= [−⌊θmax/∆θ⌋,⌊θmax/∆θ⌋]ensures full FOV coverage. The safety margin ηand collision check are
derived from depth-based navigability analysis.
Adaptive Action Filtering Refinement combines exploration state Vtand historical search patterns Ht:
At
cand=
(θi, ri)∈At
initα(Ht)·s(Vt)> τ, min
θj∈Acand|θi−θj| ≥θδ.
(9)
where α(·)models temporal search impact and s(·)quantifies spatial exploration potential.
Safety-Critical Recovery The next action set enforces, where F(·)evaluates action viability and γcontrols
rotation cool down:
At
final=(
{(π,0)}if,F(At
cand)∧(t−trot> γ)
At
cand otherwise .(10)
Action Projection The following phase focuses on visually anchoring these operational elements within the
comprehensible semantic realm of the VLM. The projection component annotated visual depiction It
annofrom
At
finalandIt. We use numeric encoding, assigning a distinct code to each actionable option that is displayed on
the visual interface. It is worth noting that rotation is assigned the code 0.
B Steps Conversion
To establish temporal equivalence between DORAEMON’s continuous actions and Habitat’s discrete steps, we
implement the conversion protocol formalized in Algorithm 1. Given a polar action a= (r, θ)∈R+×(−π, π]
with radial displacement rmeters and angular rotation θradians:
This formulation enables direct comparison with baseline methods by normalizing both:
Tepisode =500X
t=1Nt≤500 (11)
where Ntdenotes converted steps for action at time step t. During our experiments, one DORAEMON step t
was equivalent to about 7-8 Nt
13

Algorithm 1 Discrete Step Conversion
Require: Polar action (r, θ), displacement unit ∆r= 0.25m, angular unit ∆θ= 30◦
1:ifaction is stop then
2: return 1 ▷Explicit stop handling
3:else
4: sr← ⌈r/∆r⌉ ▷Radial step calculation
5: θdeg←180|θ|/π ▷ Radian-degree conversion
6: sθ← ⌈θdeg/∆θ⌉ ▷Angular step calculation
7: N←max( sr+sθ,1) ▷Step composition
8: return N
9:end if
C Navigation Case
Figure 6 depicts a full navigation episode in which our memory -augmented agent searches for a chair in an
unfamiliar indoor scene. The seven consecutive frames show the agent’s visual observations and planned motions
from entry to target acquisition. Throughout the sequence, the agent (i) reasons about semantic priors—chairs
are more likely near tables or in living/dining areas; (ii) fuses transient visual evidence with its episodic memory
to avoid revisiting explored regions; and (iii) selects actions that maximise expected information gain while
respecting safety constraints. The case, therefore, illustrates how the proposed memory module complements
on-the-fly perception to yield efficient, goal-directed exploration in cluttered, real-world layouts.
The agent is currently in a foyer where no chair is visible. Chairs are more likely deeper inside the house (e.g., living or dining rooms). Among the options, moving forward-right takes the agent directly into the open interior while avoiding an obstructing wall, whereas moving right is less direct and moving backward heads toward a closed door. Hence, Action1 (forward-right) is chosen.
The agent is in a narrow hallway with no chair in sight. Chairs are more likely in a living space glimpsed through the partially open door to the right. Other directions head under stairs, down a narrow dead-end hall, or backward—all unlikely to reveal a chair. Therefore, Action5 (enter through the open door on the right) is selected.
The agent stands in a hallway beside a damaged door. Through the opening it sees chairs in what looks like a dining room. To reach a chair quickly and avoid retracing explored paths, it chooses Action1 (forward-left) to pass through the doorway into that room.
The recent memory of the hallway to my left suggests that’s an already explored area, and going backward would be backtracking.[left direction, 1.7m, 1 steps ago], […], […] Visisted Memory
14

Visisted Memory
memory suggests the chair might be in this room. and the area in front of me seems to be an unexplored area.[backward direction, 1.7m, 1 steps ago], […]The agent is already in a combined living/dining room where chairs are visible, but to cover the still-unexplored central area it chooses Action3, which moves deepest into the room.
Visisted Memory
"Memory suggests the chair might be nearby, potentially in the area I just came from, but I should prioritize exploring new spaces."[backward direction, 1.7m, 1 steps ago], […]The agent is in a living/dining room and spots chairs around a small round table on the right. Since those visible chairs are a more reliable cue than vague earlier memories, it chooses Action3 to move toward that table.
The agent is in a dining-style room with a central table and clearly visible chairs. To reach them most directly it selects Action5, which moves straight toward the table and chairs.
Visisted Memory
[left direction, 1.3m, 1 steps ago], […], […]
“Memory suggests the chair might be in the environment. The closest memory of a possible chair location was 2 steps ago in a different room. However, I’ve already found chairs in this room.”Inside a dining room the agent already sees two chairs near a table and aims to approach the closest one. Among the available moves, Action3 advances slightly nearer to that chair while keeping within unexplored space, so it is selected.Figure 6: Navigation case Each row shows one decision step. Left: the green circle highlights
the action selected for this step. Upper-right dashed green box displays the most relevant episodic
memory retrieved at this step. Lower-right speech bubble is the agent’s natural -language rationale
that fuses (i) semantic priors, (ii) current visual evidence, and (iii) memory cues.
15

D Detailed Description of AORI
D.0.1 Area Overlap Redundancy Index (AORI)
The Area Overlap Redundancy Index (AORI) quantifies exploration efficiency through spatial overlap analysis.
We formalize the computation with parameters from our implementation:
Parameter Basis:
• Map resolution: 5,000×5,000grid (map_size=5000)
• V oxel ray casting resolution: 60×60(voxel_ray_size=60)
• Exploration threshold: 3 observations per voxel (explore_threshold=3)
• Density scaling factor: η= 0.8(e_i_scaling=0.8)
Step-wise Calculation: For each step t∈[1, T]:
1. Compute observed area At=St
i=1V(xi, yi)whereV(x, y)is the visible region defined by:
∥V(x, y)∥=map_size2
voxel_ray_size2·π (12)
2. Calculate overlap ratio roverlap :
roverlap =Pt−1
i=1I[V(xt, yt)∩ V(xi, yi)≥explore_threshold ]
t−1(13)
3. Compute normalized density dnormalized using Poisson expectation:
dnormalized = min
1,Nobs
λ
, λ =η·∥At∥
map_size2·t (14)
where Nobscounts voxels with ≥3 visits, λis expected active voxels
Boundary Cases:
•Optimal Case (AORI=0): When roverlap = 0 & dnormalized = 0⇒1−(0.8·12+ 0.2·1) = 0
•Worst Case (AORI=1):When roverlap = 1 & dnormalized = 1⇒1−(0.8·0 + 0 .2·0) = 1
Calculation Examples:
•Case1: stay still (t=100 steps):
roverlap =99
99= 1.0,
λ= 0.8·π(60/5000)2
1·100≈0.014,
dnorm= min
1,100
0.014
= 1.0,
AORI = 1−[0.8(1−1)2+ 0.2(1−1)] = 1 .0(15)
•Case2: go around (t=500 steps):
roverlap≈38
499≈0.076,
λ= 0.8·π(60/5000)2
1·500≈0.069,
dnorm= min
1,62
0.069
= 1.0,
AORI = 1−[0.8∗(1−0.076)2+ 0.2∗(1−1)]≈0.285(16)
16

E Experimental Setup Details
Implementation Details. The maximal navigation steps per episode are set to 40. The agent’s body has
a radius of 0.17mand a height of 1.5m. Its RGB-D sensors are positioned at 1.5mheight with a −0.45
radian downward tilt and provide a 131◦Field of View (FoV). For rotation, the agent selects an angular
displacement corresponding to one of 60 discrete bins that uniformly discretize the 360◦range. Success requires
stopping within dsuccess = 0.3m of the target object and visually confirming it. Success requires stopping within
dsuccess = 0.3mof the target object and visually confirming it. Our DORAEMON framework primarily utilizes
Gemini-1.5-pro as the VLM and CLIP ViT-B/32 for semantic embeddings, with caching implemented for
efficiency. Key hyperparameters include: topological map connection distance δconnect = 1.0m, node update
interval Supdate = 3 steps, L1hierarchical clustering weight w= 0.4, AORI grid resolution δgrid= 0.1m,
minimum obstacle clearance dmin_obs = 0.5m, and various stuck detection thresholds (e.g., path inefficiency
ηpath<0.25, small area coverage δarea_gain <0.35m2, high rotation/translation ratio ρrot/trans >2.0for short
paths when ∥path∥<0.5m) and a precision movement factor γstep= 0.1.
F Hierarchical Construction
F.1 Level L3: Observation Anchoring
•Input : Raw topological nodes vt∈ V from Eq 1
•Process : Directly mapping to memory nodes
h(3)
j=
id(3)
j, L3,∅,{vt}
. (17)
•Output :h(3)
jnodes storing original pt,stfromvt
F.2 Level L2: Area Formation ( L3→L2)
•Input :h(3)
jnodes with spatial coordinates pt
•Clustering :
1. Compute combined distance:
dcomb= 0.4∥pi−pj∥2+ 0.6
1−si·sj
∥si∥∥sj∥
. (18)
2. Apply adaptive threshold:
θ′
1=

1.5θ1(|O|>20)
0.8θ1(|O|<10)
θ1 otherwise .(19)
3. Generate clusters using scipy.linkage + fcluster
•Functional Labeling :
area_type = arg max
tX
v∈C(2)
jX
k∈KtI[k∈v.Lt]. (20)
•Output :h(2)
mnodes with:
–Parent: h(1)
n(L1room node).
–Children: {h(3)
j}( observations).
–Spatial boundary: Convex hull of ptpositions.
F.3 Level L1: Room Formation ( L2→L1)
•Input :h(2)
mareas with spatial centroids PA
•Two-stage Clustering :
1.Spatial Pre-clustering :
Cspatial=fcluster (linkage (dspatial), θ2= 3.0m). (21)
2.Functional Refinement :
Fs={As,f|f=MapToRoomFunction (area_type )}. (22)
•Output :h(1)
nnodes containing:
–Parent: h(0)
0(L0root)
–Children: {h(2)
m}(L2areas)
17

F.4 Level L0: Environment Root
•Input : Allh(1)
nroom nodes
•Consolidation :
h(0)
0=
GLOBAL_ROOT , L0,∅,{h(1)
n}
. (23)
•Function : Global access point for memory queries
G Memory Retrieval Scoring Details
G.1 Scoring Function Decomposition
The retrieval score combines four evidence components through weighted summation:
S(hi) = 0 .45Ssem+ 0.30Sspa+ 0.20Skey+ 0.05Stime. (24)
G.2 Component Specifications
G.2.1 Semantic Similarity
•Input : CLIP embeddings sq(query) and si(node)
•Calculation :
Ssem=1
2 
1 +s⊤
qsi
∥sq∥∥si∥!
∈[0,1]. (25)
G.2.2 Spatial Proximity
•Input : Agent position pa, node position pi
•Decay function :
Sspa= exp
−∥pa−pi∥2
5.0
. (26)
G.2.3 Keyword Relevance
•Input : Query terms T, node keywords Ki(from Lt)
•Matching score :
Skey=|T∩Ki|
max(|T|,1). (27)
G.2.4 Temporal Recency
•Input : Current time tc, observation time ti
•Decay model :
Stime= exp
−|tc−ti|
600
. (28)
G.3 Parameter Configuration
Table 4: Scoring Component Weights
Component Symbol Value
Semantic Similarity αsem 0.45
Spatial Proximity αspa 0.30
Keyword Relevance αkey 0.20
Temporal Recency αtime 0.05
G.4 Search Process
The beam search executes through these discrete phases:
18

Initialization Phase
• Start from root node(s): F0={hroot}
• Set beam width: B= 5
Iterative Expansion For each hierarchy level l∈ {L3, L2, L1, L0}:
• Score all children: S(hchild)∀hchild∈ C(hj), hj∈ Fl
• Select top- Bnodes
Termination Conditions
•Success : Reached L0nodes and selected top- Kresults
•Failure : No nodes satisfy S(hi)>0.4threshold
G.5 Computational Properties
•Time Complexity :O(B·D)for depth D= 4
•Memory Complexity :O(B)nodes per level
•Score Normalization : X
k∈{sem,spa,key,time }αk= 1.0. (29)
H Chain-of-Thought Prompt
Our Policy-VLM leverages a structured Chain-of-Thought (CoT) prompt to guide the decision-making process.
The complete prompt is provided below:
TASK : NAVIGATE TO THE NEAREST [ TARGET_OBJECT ], and get as close to it
,→as possible .
Use your prior knowledge about where items are typically located
,→within a home .
There are [N] red arrows superimposed onto your observation , which
,→represent potential actions .
These are labeled with a number in a white circle , which represent
,→the location you would move to if you took that action .
[ TURN_INSTRUCTION ]
Let ’s solve this navigation task step by step :
1. Current State Analysis : What do you observe in the environment ?
,→What objects and pathways are visible ?
Look carefully for the target object , even if it ’s partially
,→visible or at a distance .
2. Memory Integration : Review the memory context below for clues
,→about target location .
- Pay special attention to memories containing or near the target
,→object
- Use recent memories ( fewer steps ago) over older ones
- Consider action recommendations based on memory
3. Goal Analysis : Based on the target and home layout knowledge ,
,→where is the [ TARGET_OBJECT ] likely to be?
4. Scene Assessment : Quickly evaluate if [ TARGET_OBJECT ] could
,→reasonably exist in this type of space :
- If you ’re in an obviously incompatible room (e.g., looking for a
,→[ TARGET_OBJECT ] but in a clearly different room type ),
,→choose action 0 to TURN AROUND immediately
5. Path Planning : What ’s the most promising direction to reach the
,→target ? Avoid revisiting
19

previously explored areas unless necessary . Consider :
- Available paths and typical room layouts
- Areas you haven ’t explored yet
6. Action Decision : Which numbered arrow best serves your plan ?
,→Return your choice as {" action ": <action_key >}. Note :
- You CANNOT GO THROUGH CLOSED DOORS , It doesn ’t make any sense to
,→go near a closed door .
- You CANNOT GO THROUGH WINDOWS AND MIRRORS
- You DO NOT NEED TO GO UP OR DOWN STAIRS
- Please try to avoid actions that will lead you to a dead end to
,→avoid affecting subsequent actions , unless the dead end is
,→very close to the [ TARGET_OBJECT ]
- If you see the target object , even partially , choose the action
,→that gets you closest to it
I Detailed Description of Baseline
To assess the performance of DORAEMON , we compare it with 16recent baselines for (zero-shot) object-goal
navigation. Summaries are given below.
ProcTHOR [12]: A procedurally–generated 10K-scene suite for large-scale Embodied AI.
ProcTHOR_ZS [12]: ProcTHOR_ZS trains in ProcTHOR and evaluates zero -shot on unseen
iTHOR/RoboTHOR scenes to test cross-domain generalisation.
SemEXP [9]: Builds an online semantic map and uses goal -oriented exploration to locate the target object
efficiently, achieving state-of-the-art results in Habitat ObjectNav 2020.
Habitat -Web [31]: Collects large -scale human demonstrations via a browser interface and leverages behaviour
cloning to learn object-search strategies.
PONI [28]: Learns a potential -field predictor from static supervision, enabling interaction -free training while
preserving high navigation success.
ZSON [22]: Encodes multimodal goal embeddings (text + images) to achieve zero -shot navigation towards
previously unseen object categories.
PSL [35]: Prioritised Semantic Learning selects informative targets during training and uses semantic expansion
at inference for zero-shot instance navigation.
Pixel -Nav [5]: Introduces pixel -guided navigation skills that bridge foundation models and ObjectNav, relying
solely on RGB inputs.
SGM [47]: “Imagine Before Go” constructs a self -supervised generative map to predict unseen areas and
improve exploration efficiency.
ImagineNav [48]: Prompts vision–language models to imagine future observations, guiding the agent toward
information-rich viewpoints.
CoW [13]: Establishes the “Cows on Pasture” benchmark for language -driven zero -shot ObjectNav and releases
baseline policies without in-domain training.
ESC [51]: Employs soft commonsense constraints derived from language models to bias exploration, markedly
improving zero-shot success over CoW.
L3MVN [43]: Utilises large language models to reason about likely room sequences, while a visual policy
executes the suggested path.
VLFM [42]: Combines VLM goal -localisation with frontier -based exploration, removing the need for reinforce-
ment learning or task-specific fine-tuning.
VoroNav [39]: Simplifies the search space via V oronoi partitions and pairs this with LLM -driven semantic
planning for improved zero-shot performance.
TopV -Nav [50]: Lets a multimodal LLM perform spatial reasoning directly on top -view maps, with adaptive
visual prompts for global–local coordination.
SG-Nav [41]: Online builds a 3D scene graph and uses hierarchical Chain --of--Thought prompting so an LLM
can infer probable target locations.
20