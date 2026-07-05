# Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping

**Authors**: Tao Chen, Lizheng Liu, Jiaxu Wang, Ziyue Jiang, Ruiqi Tian, JiGuang Huo, Zhongxue Gan

**Published**: 2026-06-30 06:30:22

**PDF URL**: [https://arxiv.org/pdf/2606.31200v1](https://arxiv.org/pdf/2606.31200v1)

## Abstract
Generalizable robotic grasping in cluttered environments is essential for deploying manipulators in unstructured human spaces, yet existing VLM-based methods rely on visual similarity for object matching, neglecting physical affordances such as handle graspability and material fragility, and operate open-loop without spatial reasoning or failure recovery, limiting their effectiveness when objects are densely packed or physically diverse. We present Agentic RAG-VLM, a unified framework that bridges VLM-based semantic understanding and physically grounded grasp execution by integrating retrieval-augmented generation (RAG) with vision-language models (VLMs) and agentic self-reflective planning. Agentic RAG-VLM introduces three tightly coupled components: (1) a Hierarchical Affordance-Aware RAG (HAA-RAG) that encodes four-dimensional affordance descriptors, including type, material, fragility, and graspable region, and retrieves strategies by functional affordance compatibility rather than visual appearance; (2) a Scene Graph Constraint Reasoner that constructs spatial relationship graphs from VLM perception and translates proximity, occlusion, and support constraints into concrete grasp parameter adjustments; and (3) an Agentic Self-Reflective Pipeline with a 14-type failure taxonomy and three-level adaptive retry for closed-loop grasp refinement. Evaluated on a 12-task benchmark spanning single-grasp, interactive, and long-horizon scenarios with 360 trials per configuration, Agentic RAG-VLM achieves 78.3 percent overall success, a 53.3 percentage-point absolute gain over VLM-only baselines, demonstrating that affordance-aware retrieval, scene graph reasoning, and agentic recovery are jointly essential for robust manipulation.

## Full Text


<!-- PDF content starts -->

Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented
Generation with Self-Reflective Planning for Robotic Grasping
Tao Chen1, Lizheng Liu1†, Jiaxu Wang1, Ziyue Jiang1, Ruiqi Tian2, JiGuang Huo1, Zhongxue Gan1
Abstract— Generalizable robotic grasping in cluttered envi-
ronments is essential for deploying manipulators in unstruc-
tured human spaces, yet existing VLM-based methods rely
on visual similarity for object matching—neglecting physical
affordances such as handle graspability and material fragility—
and operate open-loop without spatial reasoning or failure
recovery, limiting their effectiveness when objects are densely
packed or physically diverse. We present Agentic RAG-VLM,
a unified framework that bridges VLM-based semantic un-
derstanding and physically grounded grasp execution by in-
tegrating retrieval-augmented generation (RAG) with vision-
language models (VLMs) and agentic self-reflective planning.
Agentic RAG-VLM introduces three tightly coupled compo-
nents: (1) a Hierarchical Affordance-Aware RAG (HAA-RAG)
that encodes four-dimensional affordance descriptors—type,
material, fragility, and graspable region—and retrieves strate-
gies by functional affordance compatibility rather than visual
appearance; (2) a Scene Graph Constraint Reasoner that con-
structs spatial relationship graphs from VLM perception and
translates proximity, occlusion, and support constraints into
concrete grasp parameter adjustments; and (3) an Agentic Self-
Reflective Pipeline with a 14-type failure taxonomy and three-
level adaptive retry for closed-loop grasp refinement. Evaluated
on a 12-task benchmark spanning single-grasp, interactive,
and long-horizon scenarios with 360 trials per configuration,
Agentic RAG-VLM achieves 78.3% overall success—a 53.3 pp
absolute gain over VLM-only baselines—demonstrating that
affordance-aware retrieval, scene graph reasoning, and agentic
recovery are jointly essential for robust manipulation.
I. INTRODUCTION
Robotic grasping in unstructured environments requires
integrating visual perception, semantic understanding, and
physical execution. A general-purpose grasping pipeline typ-
ically involves perceiving the scene from sensor inputs,
planning a grasp strategy, and executing it through a ma-
nipulator. While each stage has seen significant progress
independently—from learning-based grasp detection [1], [2]
to language-conditioned imitation [3]—integrating them into
a coherent system that handles diverse objects in cluttered,
multi-object environments remains largely unsolved. This is
especially hard for long-horizon tasks like table clearing,
where sequential dependencies and accumulated errors re-
quire both accurate planning and adaptive failure recovery.
Recent vision-language models (VLMs) have opened new
directions for bridging perception and planning. RT-2 [4]
pioneers end-to-end VLM-to-action by tokenizing robotic
actions alongside language; SayCan [5] decouples high-level
1Tao Chen, Lizheng Liu, Jiaxu Wang, Ziyue Jiang, JiGuang Huo,
and Zhongxue Gan are with Fudan University, Shanghai, China (e-mail:
lzliu@fudan.edu.cn).
2Ruiqi Tian is with Kean University.
†Corresponding author.planning from low-level execution through learned affor-
dance functions; V oxPoser [6] and Code as Policies [7] gen-
erate 3D value maps and executable programs, respectively;
and ManipLLM [8] fine-tunes multimodal LLMs for 6-
DoF grasp prediction. These advances point toward general-
purpose language-conditioned manipulation. Yet three gaps
remain:
Semantic-Manipulation Gap.VLM-based systems rely
on visual similarity for object matching, but visual resem-
blance does not imply manipulation compatibility. A ceramic
mug and a glass vase may occupy nearby regions in CLIP [9]
embedding space—both are cylindrical, hollow, and similarly
sized—yet the mug should be grasped by its handle with a
firm grip, while the vase requires a cautious side approach
with reduced force. This mismatch between visual similarity
and manipulation compatibility [10] is largely unaddressed.
Scene-Unaware Planning.Most systems reason about
the target object in isolation. In cluttered environments, a
cup placed 5 cm from a fragile wine glass necessitates a
cautious lateral approach with reduced force, whereas the
same cup on an empty table can be grasped directly from
above. Scene graphs have been adopted for task and motion
planning [11], but typically demand ground-truth annotations
and do not translate spatial reasoning into concrete grasp
parameter adjustments. SpatialVLM [12] endows VLMs with
spatial understanding but does not connect it to manipulation
constraints.
Failure Recovery.Classical grasp planners [1], [2], [13]
operate open-loop, generating grasp poses without learning
from failures. Inner Monologue [14] introduces VLM-based
feedback but provides only unstructured observations (“the
object slipped”) without systematic classification. Reflex-
ion [15] demonstrates that verbal self-reflection improves
sequential decision-making, yet applying reflection to physi-
cal manipulation requires translating linguistic feedback into
concrete, physics-grounded corrections—“the object slipped”
must map to “increase grip force by 20%,” not merely a
rephrased retry.
To address these gaps, we propose Agentic RAG-VLM
(Fig. 1), a unified framework integrating affordance-aware
experience retrieval, scene graph constraint reasoning, and
agentic self-reflective planning for robust robotic grasping.
The contributions are summarized as follows:
•We propose Agentic RAG-VLM, a unified framework
that integrates affordance-aware retrieval-augmented
generation with scene graph constraint reasoning and
agentic self-reflective planning, bridging the gap be-
tween VLM-based semantic understanding and physi-arXiv:2606.31200v1  [cs.AI]  30 Jun 2026

cally grounded grasp execution.
•The framework introduces Hierarchical Affordance-
Aware RAG (HAA-RAG), a three-level retrieval
pipeline that matches grasp experiences by functional
affordance rather than visual similarity, and a Scene
Graph Constraint Reasoner that translates spatial rela-
tionships into concrete grasp parameter adjustments for
safe manipulation in cluttered environments.
•An Agentic Self-Reflective Pipeline that employs struc-
tured failure diagnosis with a 14-type taxonomy and
three-level adaptive retry, enabling the system to learn
from failed attempts and progressively refine grasp
execution through physics-grounded corrections.
II. RELATEDWORK
VLMs for Manipulation.Recent VLM-based approaches
range from end-to-end action tokenization (RT-2 [4]) to mod-
ular planning with affordance grounding (SayCan [5]), 3D
value maps (V oxPoser [6]), code generation (Code as Poli-
cies [7]), and fine-tuned grasp prediction (ManipLLM [8],
RoboDexVLM [16], GPT-4V [17]). However, these rely on
parametric knowledge alone without retrieved experience or
structured failure recovery.
RAG and Affordance in Robotics.RAG [18] reduces
hallucination by grounding generation in retrieved docu-
ments, but its robotic application remains nascent. RE-
FLECT [19] summarizes failure experiences without inte-
grating retrieval into planning. Meanwhile, affordance rea-
soning [10] has been realized through pixel-level maps [20]
and part-based analysis [21], but these predictwhereto
grasp, nothow—they identify graspable regions on an ob-
ject surface without specifying the complete grasp strategy
(force, width, approach direction, grasp type) appropriate for
that object’s physical affordance. Our HAA-RAG bridges
this gap by explicitly encoding four-dimensional affordance
descriptors (type, material, fragility, graspable region) and
retrieving complete, physically grounded grasp strategies
matched by functional affordance compatibility rather than
visual similarity.
Failure Recovery and Scene Reasoning.Classical plan-
ners [1], [2], [13] operate open-loop. Inner Monologue [14]
and Reflexion [15] introduce verbal feedback but lack
physics-grounded corrections for manipulation. Scene graphs
have been used for task planning [11] but typically require
ground-truth annotations. SpatialVLM [12] adds spatial rea-
soning to VLMs without connecting it to grasp constraints.
Our pipeline addresses both gaps with a 14-type failure
taxonomy mapping to quantitative corrections and VLM-
constructed scene graphs that infer manipulation constraints
without annotations. In contrast to these approaches, Agen-
tic RAG-VLM uniquely integrates all five capabilities—
affordance-aware retrieval, RAG-based experience ground-
ing, scene graph constraint reasoning, structured failure
recovery, and multi-step agentic planning—within a single
unified framework.
III. AFFORDANCE-AWARERETRIEVAL ANDCONSTRAINTREASONING
This section details how the system transforms a scene
observation into a physically grounded grasp plan through
two complementary stages: affordance-aware experience re-
trieval (Sec. III-A) and spatial constraint reasoning (Sec. III-
B). Given an RGB-D observationIand a natural lan-
guage instructionq, the system produces a grasp action
g= (p,d, w, f, τ)—position, approach direction, gripper
aperture, grip force, and grasp type (τ∈ {power, pinch,
side})—refined through three modular stages: (1) affordance-
aware retrieval via HAA-RAG, (2) scene graph constraint
reasoning, and (3) agentic planning with self-reflective retry
(Sec. IV).
A. Hierarchical Affordance-Aware RAG
Standard RAG retrieves by visual similarity (e.g., CLIP [9]
embeddings), but a bowl must be grasped along its rim
while a cup uses its handle—visual similarity yields wrong
strategies. This is precisely the “neglecting object-specific
physical affordances” problem identified in Sec. I: visually
similar objects (e.g., a ceramic mug and a glass vase) may
require fundamentally different manipulation strategies due
to their distinct affordance properties. HAA-RAG directly
addresses this by matching experiences throughaffordance
compatibilityinstead of visual resemblance (Fig. 2).
The foundation of this approach is a curated knowledge
baseK={e 1, . . . , e N}withN= 116manipulation
experiences, where each entrye i= (c i,ai,gi, yi,vi)stores:
•ci: Object category (e.g., “mug”, “bowl”, “screwdriver”)
•ai= (atype
i, amat
i, afrag
i, areg
i): Affordance descriptor
with primary type (8 categories: GRASPABLE BODY/
HANDLE/EDGE, PINCHABLE, WRAPPABLE, FRAG-
ILE, TOOL HANDLE, CLAMPABLE), material, fragility
afrag∈[0,1], and graspable region
•gi: Grasp parameters that led to the outcome
•yi∈ {0,1}: Binary success label
•vi: Visual feature embedding (CLIP ViT-L/14)
Failed experiences (y i= 0) are also retained with a
contrastive penalty (scores reduced by 30%) to discourage
repeating failed strategies while remaining available as neg-
ative examples for the reflection module (Sec. IV-B).
Leveraging this knowledge base, HAA-RAG implements
a three-level coarse-to-fine retrieval pipeline. In the first
level, the VLM identifies the target object categoryc qfrom
instructionqand the RGB image, retrieving the top-k 1= 30
candidates fromKby category match: exact matches score
scat= 1.0, same-superclass fuzzy matches (e.g., “mug”→
“cup”) score0.5, and unrelated categories are discarded.
In the second level, surviving candidates are re-scored
by affordance similarity, capturing functional compatibility
between the query object and stored experiences:
saff(aq,ai) = 0.5·⊮[atype
q=atype
i]
| {z }
affordance type match+ 0.2·s mat|{z}
material
+ 0.15·(1− |∆f|)|{z }
fragility+ 0.15·s reg|{z}
region(1)

Fig. 1: Overview of Agentic RAG-VLM. The VLM-based Task Planner processes multimodal input and orchestrates a seven-
stage execution pipeline: scene perception, affordance-aware retrieval via HAA-RAG, scene graph constraint reasoning (∆g),
and grasp execution. Upon failure (×), the recovery loop diagnoses failures via a 14-type taxonomy, escalates corrections
through three levels (L1: parameter tuning→L2: method switch→L3: full replan), and stores successful grasps in episodic
memoryMfor cross-task transfer.
wheres mat=⊮[amat
q=amat
i]is material compatibility
(e.g., both ceramic),∆f=afrag
q−afrag
icaptures fragility
similarity, ands reg=⊮[areg
q=areg
i]is graspable region
overlap. The dominant weight on affordance type (0.5)
ensures that functional compatibility is the primary selection
criterion. Candidates scoring below thresholdτ aff= 0.3are
discarded.
Finally, the remaining candidates are re-ranked by CLIP
visual similaritys vis= cos(v q,vi)using the current RGB
observation’s embedding, ensuring that among affordance-
compatible experiences, the most visually similar one is
selected for fine-grained parameter matching. The final re-
trieval score fuses all three levels (Eq. 1) ass final= 0.2s cat+
0.4s aff+0.4s vis, and the top-k= 3experiences are returned
to the planner as candidate strategies.
B. Scene Graph Constraint Reasoning
While HAA-RAG determineshowto grasp the target
object based on its intrinsic affordance properties, real-world
manipulation must also account for theextrinsic spatial
contextimposed by neighboring objects. The Scene Graph
Constraint Reasoner addresses this by constructing a struc-
tured spatial representation and translating it into concrete
grasp parameter adjustments.
From the VLM’s scene analysis, we construct a directed
graphG= (V,E)where each nodev j∈ Vencodes object
attributes(p j, cj,state j, afrag
j)—position, category, physical
state (empty, filled, or stacked), and fragility score—and each
directed edgee jk= (v j, vk, rjk)∈ Ecaptures a spatial
relationr jkfrom 16 predefined types including ON TOP OF,
CONTAINS, OCCLUDES, SUPPORTS, ADJACENT TO, and
Fig. 2: Hierarchical Affordance-Aware RAG (HAA-RAG)
pipeline. Experiences are progressively filtered through three
levels: category matching, affordance scoring, and visual re-
ranking, reducing 116 candidates to 3 physically appropriate
strategies. Right annotations show candidate count at each
stage.
BEHIND. A rule-based Constraint Analyzer then traverses
Gstarting from the target nodev t, examiningv t’s attributes
and neighbors within a radius ofδ check = 0.15m to infer
active constraintsC vt. Four constraint types are defined, each
mapping to specific parameter adjustments:
1)Content Preservation(C content ): For filled containers—

constrains approach to vertical (d= [0,0,−1]) with
slow velocity to prevent spillage.
2)Collision Avoidance(C collision ): When fragile neighbor
vfwithinδ safe= 0.10m—applies force reductionf′=
0.8f, increases approach height by 30 mm, and biases
approach direction away fromv f.
3)Support Dependency(C support ): When a SUPPORTS
edge(v t, vs)exists, the system inserts a prerequisite ac-
tion to remove the supported objectv sbefore attempting
to graspv t.
4)Occlusion Handling(C occlusion ): When occluder blocks
direct line-of-sight to target—modifies approach to lat-
eral path that avoids the occluding obstacle.
The inferred constraints are compiled into aconstraint
adjustment vector∆g= (∆d,∆f,∆h)that additively
modifies the planned grasp action:g′=g⊕∆g. Multiple
active constraints are composed sequentially, with collision
avoidance taking priority (Fig. 3).
IV. AGENTICSELF-REFLECTIVEGRASPPLANNING
Real-world execution is noisy, and initial failures are
common. Agentic RAG-VLM employs an Agentic Self-
Reflective Pipeline providingclosed-loopexecution with
structured failure diagnosis and adaptive recovery.
A. ReAct-Style Planning Loop
Inspired by the ReAct framework [22], the VLM iterates
through structured THOUGHT→ACTION→OBSERVATION
cycles until either a successful grasp is achieved or the
maximum retry budgetR max= 3is exhausted.
Algorithm 1 formalizes the loop. The initial graspg 0is
generated from the top-ranked retrieved experience, adjusted
by scene graph constraints and episodic memoryM—a
session-level cache of successful grasps indexed by object
category. The retry budgetR max= 3balances recovery from
failure modes against preventing indefinite loops.
B. Failure Recovery with Structured Reflection
The Reflection Module generatesquantitativecorrec-
tions grounded in physics. The seven quality factorsϕ=
Fig. 3: Scene graph constraint reasoning. Left: scene graphG
with object nodes and spatial relations. Right: four constraint
types inferred from the graph, each mapped to concrete
parameter adjustments∆gthat modify the planned grasp.Algorithm 1Agentic Self-Reflective Grasp Planning
Require:RGB-D imageI, instructionq, knowledge baseK,
budgetR max
Ensure:Grasp actiong∗or failure report
1:{e 1, . . . , e k} ←HAA-RAG(I, q,K){Retrieve experi-
ences}
2:G ←BUILDSCENEGRAPH(I){Construct scene graph}
3:C ←INFERCONSTRAINTS(G, v t){Infer constraints}
4:g 0←INITGRASP(e 1,C,M){Initial plan from mem-
ory/RAG}
5:forr= 0toR max do
6:Thought:Reason aboutg r,C, and historyH
7:Action:Execute graspg r
8:Observation:(y r, Qr,ϕr)←EVALUATE(g r)
9:ify r=SUCCESSthen
10:M ← M∪{(c q,gr)} {Store in episodic memory}
11:returng r
12:end if
13:F r←CLASSIFYFAILURE(ϕr){14-type taxonomy}
14:∆g r←GETCORRECTION(F r, r){Level-dependent}
15:g r+1←APPLYCORRECTION(g r,∆g r, r)
16:H ← H ∪ {(g r,Fr, Qr)} {Update history}
17:end for
18:returnFAILURE
(ϕ1, . . . , ϕ 7)from the grasp evaluator (Sec. IV-D) are ana-
lyzed to classify failures into 14 types in four groups:Pre-
contact(3: position error, unreachable pose, approach angle);
Contact(3: width mismatch, collision, orientation);Post-
grasp(4: slip, drop, force damage, deformation);Task-level
(4: wrong object, constraint violation, timeout, unknown).
Each type maps to a correction rule (e.g., SLIP→f′=1.2f;
WIDTH MISMATCH→w′=w+0.01m).
Corrections escalate through three levels (Fig. 4): Level 1
(r≤1): targeted parameter tuning (force, width, height);
Level 2 (r=2): grasp type rotation (power→pinch→side)
with 10% force reduction; Level 3 (r=3): complete re-
planning with reset parameters and random perturbations
(ϵ∼ U(−0.01,0.01)3).
C. Episodic Memory and Cross-Task Transfer
Successful grasps are stored in session-level episodic
memoryM={(c j,g∗
j)}indexed by object category, en-
abling within-session transfer: proven parameters for one cup
transfer directly to subsequent cups in table-clearing tasks,
eliminating exploration overhead.
D. Grasp Quality Model
Central to the failure diagnosis and recovery described
above, the analytical grasp quality model evaluates each
grasp attempt using a seven-factor scoring function that
captures complementary aspects of grasp feasibility and

Fig. 4: Agentic self-reflective planning loop. The ReAct cycle
generates Thought→Action→Observation. Failures trigger
reflection using a 14-type taxonomy, escalating through three
retry levels. Successes are stored in episodic memoryMfor
cross-task transfer.
robustness:
Q(g, o) =7X
k=1ωk·ϕk(g, o)(2)
where the seven factors cover position accuracy (ω 1=0.20),
width compatibility (ω 2=0.15), force appropriateness
(ω3=0.10), grasp type matching (ω 4=0.15), approach
clearance (ω 5=0.15), object difficulty (ω 6=0.10), and grip
security (ω 7=0.15). Failure modes are triggered when
anyϕ kfalls below its thresholdθ k(e.g.,ϕ 2<0.4⇒
WIDTH MISMATCH). This ensuresphysically groundedand
reproducibleoutcomes.
V. EXPERIMENTALANALYSIS
A. Experimental Setup
Simulation Environment.We conduct experiments in an
analytical simulation modeling a Franka Emika Panda 7-DoF
arm with a parallel-jaw gripper (0–80 mm aperture) on a
0.6×0.8m tabletop with household objects of diverse shapes,
materials, and fragility (Table I). The simulator implements
forward/inverse kinematics with joint limits, geometric colli-
sion detection using a gripper envelope model, and a seven-
factor grasp quality model (Eq. 2) for reproducible outcome
determination.
Hardware and Model Configuration.We use Qwen3-
VL-8B [23] as the foundation VLM on a single NVIDIA
RTX 5090 GPU (32 GB). INT4 quantization via BitsAnd-
Bytes [24] with FP8 KV cache achieves 155.5 tokens/s
(1.78×over FP16). A complete trial takes∼21 s (SG), 25 s
(IT), and 37 s (LH).
Object Dataset.We curate 12 household objects spanning
10 categories with diverse physical properties (Table I),
covering 7 of the 8 affordance types with fragility ranging
from 0.0 (wood cube) to 0.9 (wine glass).
Task Suite.Following the multi-category evaluation pro-
tocol of [16], we design 12 benchmark tasks organized in
three categories of increasing complexity:
•Single-Grasp (SG, 6 tasks):Pick up an individual
object from an uncluttered tabletop: cup, apple, bottle,TABLE I: OBJECTDATASET: 12 objects spanning 10 cat-
egories and 8 affordance types. Fragility ranges from 0.0
(durable) to 0.9 (very fragile).
Object Affordance Material Dim. (cm) Frag.
Ceramic cup Handle Ceramic 8×8×9 0.3
Apple Wrappable Organic 8 (dia.) 0.1
Plastic bottle Body Plastic 7×7×22 0.1
Banana Wrappable Organic 4×4×18 0.2
Ceramic bowl Edge Ceramic 12 (dia.)×6 0.3
Wood cube Body Wood 5×5×5 0.0
Glass vase Fragile Glass 8×8×15 0.8
Wine glass Fragile Glass 7×7×20 0.9
Screwdriver Tool handle Metal 3×3×20 0.0
Smartphone Clampable Mixed 7×1×15 0.5
Tennis ball Wrappable Fabric 6.5 (dia.) 0.0
Toy block Body Plastic 4×4×4 0.0
banana, bowl, cube. Difficulty ranges from easy (ba-
nana: elongated, wrappable) to hard (bowl: edge grasp,
narrow rim exceeding gripper width).
•Interactive (IT, 4 tasks):Grasp with spatial con-
straints: (a) cup near a fragile glass vase (requires
cautious approach), (b) ball behind a box (occlusion-
aware planning), (c) phone near a wine glass (fragility
with flat object geometry), (d) screwdriver requiring
tool-appropriate side grasp. These tasks evaluate scene
graph reasoning and constraint satisfaction.
•Long-Horizon (LH, 2 tasks):Sequentially clear 3 ob-
jects, requiring episodic memory to transfer successful
parameters across sequential attempts: (a) clear a table
with cup, apple, and cube, (b) sort 3 objects by fragility
level.
Each task receives 30 independent trials (360 total per
configuration across all 12 tasks).
Evaluation Protocol.We evaluate through comprehen-
sive ablation under identical conditions using the analytical
quality model (Eq. 2). The primary baseline isVLM-Only
(Qwen3-VL-8B without RAG, scene graph, recovery, or
memory). For context, published methods achieve 88–97%
on standard single-object benchmarks (GR-ConvNet [1]:
97.7%, AnyGrasp [2]: 88%), but these do not include inter-
active or long-horizon tasks, precluding direct comparison.
Statistical testing uses 5 complete repetitions with different
random seeds.
B. Main Results: Component Analysis
Table II presents the comprehensive component analysis.
Agentic RAG-VLM achieves78.3% overall success rate
(78.7±1.8%over 5 runs, 95% CI: [76.4, 81.0]).Single-
grasp tasksachieve 91.7%, with simple objects (banana,
cube) reaching 100% and more challenging objects (apple:
60.0%, cup: 90.0%) requiring affordance-specific grasping.
Interactive tasks(64.2%) demonstrate the critical benefit
of scene graph reasoning for constraint-aware manipulation
near fragile objects.Long-horizon tasks(66.7%) remain
challenging, as sequential dependencies amplify individual
failure probabilities across multi-step executions. Fig. 5
illustrates representative execution traces for each category.

TABLE II: COMPONENTANALYSISon 12-task benchmark.
Success rate (%) per task category and overall (30 trials
each).∆: change vs. Full System. All evaluated identically.
Best inbold.
Configuration SG IT LH Overall∆
Ours (Full) 91.7 64.2 66.7 78.3–
w/o Recovery 63.3 33.3 23.3 46.7−31.6
w/o Episodic Memory 91.7 64.2 55.0 76.4−1.9
w/o Scene Graph 91.7 25.0 66.7 65.3−13.0
Heuristic Baseline 100.0 25.0 98.3 74.7−3.6
w/o HAA-RAG 50.0 0.0 0.0 25.0−53.3
VLM-Only 50.0 0.0 0.0 25.0−53.3
A clear hierarchy of component importance emerges:
HAA-RAG is indispensable(∆ =−53.3%)—removing
retrieval produces the largest performance collapse, with IT
and LH dropping to 0%. This confirms that affordance-
matched retrieval forms the foundation of effective grasping:
without it, default parameters fail every interactive and long-
horizon task, as these require object-specific grasp config-
urations.Adaptive recovery(∆ =−31.6%) is the second
most critical component, with IT task success dropping from
64.2% to 33.3% and LH from 66.7% to 23.3%—both task
categories require iterative correction.Scene graph reason-
ing(∆ =−13.0%) has major impact on interactive tasks
(IT: 64.2%→25.0%,∆ IT=−39.2%), providing collision-
aware spatial planning that enables safe manipulation near
fragile objects—without it, 3 of 4 interactive tasks fail en-
tirely due to gripper–neighbor collisions.Episodic memory
primarily impacts long-horizon tasks (LH: 66.7%→55.0%,
∆LH=−11.7%), confirming that within-trial parameter
transfer benefits sequential multi-object tasks.
TheHeuristic baseline(size/fragility-based parameter
selection with recovery, no VLM or RAG) achieves 74.7%
overall, excelling on single-grasp (100%) and long-horizon
(98.3%) tasks where object-adapted parameters suffice. How-
ever, it drops to 25.0% on interactive tasks—matching
w/o Scene Graph—because it lacks spatial reasoning to
avoid collisions near fragile neighbors. This confirms that
the full system’s interactive advantage derives specifically
from scene graph constraint inference, not from general
parameter optimization. Notably, w/o HAA-RAG converges
to identical performance as VLM-Only (25.0%), demonstrat-
ing that downstream modules areineffectivewithout a re-
trieval knowledge foundation. The most frequent failures are
WIDTH MISMATCH(gripper-object incompatibility), SLIP
(insufficient grip force), and FORCE DAMAGE(excessive
force on fragile objects), reflecting fundamental parallel-jaw
gripper limitations—the bowl diameter (12 cm) exceeding
gripper aperture (8 cm) accounts for the majority of single-
grasp failures.
C. Recovery Mechanism Effectiveness
To further evaluate the recovery mechanism in isolation,
we compare the full Agentic RAG-VLM against a single-
attempt variant (Table III). For single-grasp tasks, recovery
provides+28.4%improvement, primarily correcting minorTABLE III: RECOVERYMECHANISMEFFECTIVENESS
across task categories. 30 trials per task, average over cate-
gory.
Method Category Succ. (%) Avg. Time (s)
w/o RecoverySG (6 tasks) 63.3 1.7
IT (4 tasks) 33.3 2.5
LH (2 tasks) 23.3 5.8
w/ RecoverySG (6 tasks)91.72.3
IT (4 tasks)64.23.5
LH (2 tasks)66.79.1
TABLE IV: RETRIEVALQUALITYEVALUATIONon 12
object queries against a 116-entry knowledge base.
Method P@1 P@3 MRR Aff. Match
HAA-RAG (ours) 91.7% 91.7% 0.917 91.7%
CLIP-Only 66.7% 66.7% 0.729 63.3%
force and width mismatches through Level 1 parameter
tuning at a modest time overhead (+0.6s average). Inter-
active tasks show a+30.9%gain: first-attempt failures (e.g.,
collision near a fragile neighbor) are systematically diag-
nosed and corrected with constraint-aware adjustments across
multiple retry levels. Long-horizon tasks improve by+43.4%
because the recovery mechanism corrects individual sub-goal
failures within multi-step sequences, preventing early failure
cascading. The time overhead scales with task complexity:
+0.6s for SG,+1.0s for IT,+3.3s for LH—reflecting
that harder tasks require more retry iterations (average 2.48
attempts for the full system, with 53.3% recovery rate).
D. Retrieval Quality Analysis
Table IV compares retrieval quality against aCLIP-Only
baseline (category filtering + visual re-ranking, skipping
affordance matching). HAA-RAG achieves 91.7% P@1 and
0.917 MRR vs. 66.7% and 0.729 for CLIP-Only. The+25.0
pp P@1 improvement confirms that affordance-aware filter-
ing is critical: visual similarity alone frequently selects geo-
metrically similar objects requiring different grasp strategies.
The affordance match rate gap (91.7% vs. 63.3%) shows
CLIP-Only fails to select physically appropriate strategies
for over one-third of queries.
E. Scene Graph Constraint Analysis
Table V isolates the scene graph’s impact on interactive
tasks. Without it, success drops to 25.0% due to gripper–
neighbor collisions. Scene graph construction yields+33.3
pp improvement (25.0%→58.3%), and adding constraint in-
ference enables 75% constraint satisfaction, covering content
preservation, collision avoidance, and force reduction.
F . Out-of-Distribution Generalization
To evaluate generalization, we test on In-Distribution (KB
objects), Near-OOD (same categories, different instances),
and Far-OOD (unseen categories). The system degrades from

Fig. 5: Qualitative execution traces across the three task categories.SG: direct four-step pipeline succeeding on the first
attempt.IT: constraint-aware manipulation with slip failure→L1 retry→recovery.LH: three-object clearing task with
episodic memory transfer accelerating the second same-category grasp.
TABLE V: SCENEGRAPHREASONINGIMPACTon interac-
tive tasks (120 trials total).
Configuration Success (%) Const. Sat. (%)
No Scene Graph 25.0 0.0
+ Scene Graph 58.3 0.0
+ Constraints 58.3 75.0
84.4% (In-Dist) to 54.4% (Far-OOD), expected since Far-
OOD objects lack directly matching affordance entries. Nev-
ertheless, HAA-RAG’s affordance-based retrieval enables
meaningful cross-category transfer (e.g., bottle-body-grasp
for an unseen thermos), and the retry mechanism maintains
above 54% success on entirely unseen categories.
VI. CONCLUSION
This paper presents Agentic RAG-VLM, a unified frame-
work for robotic grasping that integrates affordance-aware
retrieval-augmented generation with scene graph constraint
reasoning and agentic self-reflective planning to bridge the
gap between VLM-based semantic understanding and phys-
ically grounded grasp execution. By unifying hierarchical
affordance retrieval with spatial reasoning and closed-loop
failure recovery, the system demonstrates robust adaptability
across diverse scenarios, from single-object manipulation to
complex multi-stage table-clearing operations.
Key innovations include HAA-RAG, a three-level re-
trieval pipeline that matches grasp strategies by functional
affordance rather than visual similarity, forming the indis-
pensable foundation of effective grasping; a Scene Graph
Constraint Reasoner that translates spatial relationships into
concrete grasp parameter adjustments for safe manipulation
near fragile neighbors; and a structured failure diagnosismechanism with 14-type taxonomy and three-level adaptive
retry that provides physics-grounded closed-loop refinement,
particularly critical for long-horizon tasks. By decoupling
affordance-level reasoning from low-level grasp execution
through a modular knowledge-driven architecture, Agen-
tic RAG-VLM enables generalizable manipulation without
object-specific training.
Future work will focus on scaling the knowledge base
through autonomous experience acquisition during deploy-
ment, extending affordance reasoning to dexterous multi-
finger hands [16] with richer contact geometries, and en-
hancing cross-category generalization using hierarchical af-
fordance ontologies with causal reasoning. This research
represents a step toward general-purpose manipulation sys-
tems capable of operating reliably across diverse objects and
environments with minimal reconfiguration.
REFERENCES
[1] S. Kumra, S. Joshi, and F. Sahin, “Antipodal robotic grasping using
generative residual convolutional neural network,” inIEEE/RSJ Inter-
national Conference on Intelligent Robots and Systems (IROS), 2020.
[2] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y . Xie,
and C. Lu, “AnyGrasp: Robust and efficient grasp perception in spatial
and temporal domains,”IEEE Transactions on Robotics (T-RO), 2023.
[3] M. Shridhar, L. Manuelli, and D. Fox, “CLIPort: What and where
pathways for robotic manipulation,” inConference on Robot Learning
(CoRL), 2022.
[4] A. Brohan, N. Brown, J. Carbajal, Y . Chebotar, X. Chen, K. Choro-
manskiet al., “RT-2: Vision-language-action models transfer web
knowledge to robotic control,”arXiv preprint arXiv:2307.15818, 2023.
[5] M. Ahn, A. Brohan, N. Brown, Y . Chebotar, O. Corteset al., “Do
as i can, not as i say: Grounding language in robotic affordances,” in
Conference on Robot Learning (CoRL), 2022.
[6] W. Huang, C. Wang, R. Zhang, Y . Li, J. Wu, and L. Fei-Fei,
“V oxPoser: Composable 3d value maps for robotic manipulation with
language models,” inConference on Robot Learning (CoRL), 2023.

[7] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence,
and A. Zeng, “Code as policies: Language model programs for
embodied control,” inIEEE International Conference on Robotics and
Automation (ICRA), 2023.
[8] X. Li, M. Zhang, Y . Geng, H. Geng, Y . Long, Y . Shen, H. Wanget al.,
“ManipLLM: Embodied multimodal large language model for object-
centric robotic manipulation,” inIEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024.
[9] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Gohet al., “Learn-
ing transferable visual models from natural language supervision,” in
International Conference on Machine Learning (ICML), 2021.
[10] J. J. Gibson, “The theory of affordances,”Hilldale, USA, vol. 1, no. 2,
pp. 67–82, 1977.
[11] A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian
et al., “Transporter networks: Rearranging the visual world for robotic
manipulation,” inConference on Robot Learning (CoRL), 2020.
[12] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Driess, P. Florence,
D. Sadigh, L. Guibas, and F. Xia, “SpatialVLM: Endowing vision-
language models with spatial reasoning capabilities,”arXiv preprint
arXiv:2401.12168, 2024.
[13] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, “Contact-
GraspNet: Efficient 6-dof grasp generation in cluttered scenes,” in
IEEE International Conference on Robotics and Automation (ICRA),
2021.
[14] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng
et al., “Inner monologue: Embodied reasoning through planning with
language models,” inConference on Robot Learning (CoRL), 2022.
[15] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao,
“Reflexion: Language agents with verbal reinforcement learning,” in
Advances in Neural Information Processing Systems (NeurIPS), 2023.
[16] H. Liuet al., “RoboDexVLM: Visual language model-enabled task
planning and motion control for dexterous robot manipulation,”arXiv
preprint arXiv:2503.01616, 2025.
[17] N. Wake, A. Kanehira, K. Sasabuchi, J. Takamatsu, and K. Ikeuchi,
“GPT-4V(ision) for robotics: Multimodal task planning from human
demonstration,”IEEE Robotics and Automation Letters (RA-L), 2024.
[18] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyalet al.,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,”
inAdvances in Neural Information Processing Systems (NeurIPS),
2020.
[19] Z. Liu, A. Bahety, and S. Song, “REFLECT: Summarizing robot
experiences for failure explanation and correction,” inConference on
Robot Learning (CoRL), 2023.
[20] P. Mandikal and K. Grauman, “DexVIP: Learning dexterous grasp-
ing with human hand pose priors from video,”arXiv preprint
arXiv:2202.00164, 2022.
[21] K. Mo, L. J. Guibas, M. Mukadam, A. Gupta, and S. Mittal,
“Where2Act: From pixels to actions for articulated 3d objects,” in
IEEE/CVF International Conference on Computer Vision (ICCV),
2021.
[22] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“ReAct: Synergizing reasoning and acting in language models,” in
International Conference on Learning Representations (ICLR), 2023.
[23] S. Bai, J. Bai, A. Yang, P. Wang, J. Lin, and C. Zhou, “Qwen3-VL
technical report,”arXiv preprint arXiv:2511.21631, 2025.
[24] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA:
Efficient finetuning of quantized language models,” inAdvances in
Neural Information Processing Systems (NeurIPS), 2023.