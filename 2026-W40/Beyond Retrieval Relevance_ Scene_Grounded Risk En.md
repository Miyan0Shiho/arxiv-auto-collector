# Beyond Retrieval Relevance: Scene-Grounded Risk Entailment for Vision-Language Driving

**Authors**: Jiaxin Liu, Ruilin Yu, Liang Peng, Jingkai Wang, Chengxiang Zhao, Zhenxin Zhu, Bing Wang, Guang Chen, Hangjun Ye, Hong Wang, Jun Li

**Published**: 2026-09-28 02:26:15

**PDF URL**: [https://arxiv.org/pdf/2609.34145v1](https://arxiv.org/pdf/2609.34145v1)

## Abstract
Retrieval-augmented generation (RAG) gives vision--language driving systems access to external safety knowledge, yet a retrieved risk rule may be relevant without applying to the current scene. A vision--language model (VLM) receiving such knowledge must ground objects, bind entities across time, and verify relations before deciding how to act, leaving the support for risk conclusions implicit. We address this relevance--applicability gap with a Driving-Risk Knowledge Graph (DRKG) and Semantic Web Rule Language (SWRL) reasoning stage before VLM decision-making. Structured perception instantiates scene facts, from which SWRL rules derive events and directed risk relations when their antecedents are jointly satisfied. Recognized events, bound risk relations, and semantic descriptions of activated rules form compact evidence that conditions the VLM and diffusion planner. In matched comparisons on nuReasoning, our method improved the nuReasoning planning score (NPS) by 1.30 points and the non-at-fault collision score (NC) by 2.76 points over the relevance retrieval-based baseline. These gains indicate that scene-applicable risk evidence improves safety-weighted planning relative to semantically retrieved risk knowledge.

## Full Text


<!-- PDF content starts -->

Beyond Retrieval Relevance: Scene-Grounded Risk Entailment for
Vision-Language Driving
Jiaxin Liu1,2, Ruilin Yu1,3, Liang Peng1, Jingkai Wang1, Chengxiang Zhao1,2, Zhenxin Zhu2, Bing Wang2,
Guang Chen2, Hangjun Ye2, Hong Wang1∗and Jun Li1
Abstract— Retrieval-augmented generation (RAG) gives
vision–language driving systems access to external safety knowl-
edge, yet a retrieved risk rule may be relevant without applying
to the current scene. A vision–language model (VLM) receiving
such knowledge must ground objects, bind entities across
time, and verify relations before deciding how to act, leaving
the support for risk conclusions implicit. We address this
relevance–applicability gap with a Driving-Risk Knowledge
Graph (DRKG) and Semantic Web Rule Language (SWRL)
reasoning stage before VLM decision-making. Structured per-
ception instantiates scene facts, from which SWRL rules derive
events and directed risk relations when their antecedents
are jointly satisfied. Recognized events, bound risk relations,
and semantic descriptions of activated rules form compact
evidence that conditions the VLM and diffusion planner. In
matched comparisons on nuReasoning, our method improved
the nuReasoning planning score (NPS) by 1.30 points and
the non-at-fault collision score (NC) by 2.76 points over the
relevance retrieval-based baseline. These gains indicate that
scene-applicable risk evidence improves safety-weighted plan-
ning relative to semantically retrieved risk knowledge.
I. INTRODUCTION
Safe driving requires an autonomous system to interpret
current visual observations in light of traffic rules, common-
sense risk knowledge, and prior experience [1]. Retrieval-
augmented generation (RAG) supplements a model’s para-
metric knowledge and immediate visual evidence with ex-
ternal information accessed at inference time [2]. Driving
applications have consequently used RAG for action explana-
tion, decision prediction, and safety-oriented visual question
answering [3], [4], [5], establishing semantic retrieval as
an important route towards knowledge-augmented vision–
language driving.
Semantic retrieval determines which knowledge is relevant
to a scene, but relevance alone does not establish a scene-
specific risk. A retrieved risk rule may be generally valid
and match observed objects or behaviours, yet its conclusion
requires the corresponding objects to be grounded, its entities
and times to be bound, and its spatial, motion, and inter-
action relations to hold jointly. We call this distinction the
relevance–applicability gap: relevance identifies knowledge
worth considering, whereas applicability requires the rule’s
encoded antecedents to be positively supported by the current
scene.
In current RAG-based driving pipelines, retrieved knowl-
edge is typically passed to a vision–language model
1School of Vehicle and Mobility, Tsinghua University
2Xiaomi EV
3Jilin University(VLM), leaving this relevance-to-applicability pathway im-
plicit within generative reasoning [3], [4], [5]. The VLM
must associate visual objects with rule entities, bind temporal
states, test multiple relations, and combine their antecedents.
It must then determine which entity poses risk to which
other entity and decide how to act. This burden grows
with compositional risk complexity, while the support for
an asserted risk is not exposed as an independently check-
able result. Graph-structured question answering, knowledge-
graph-based behaviour prediction, and rule-filtered driving
frameworks show that parts of this reasoning can be struc-
tured [6], [7], [8], but do not separate scene-supported risk
entailment from downstream VLM decision-making.
We replace this retrieval-conditioned relevance-to-
applicability pathway with scene-fact instantiation and
symbolic risk reasoning over a Driving-Risk Knowledge
Graph (DRKG), as shown in Fig. 1. Existing perception
outputs initialize scene entities, measured attributes, and
simple spatial and kinematic relations. Semantic Web Rule
Language (SWRL) reasoning then considers all encoded
rules and composes these direct facts into scene events and
more complex relations. We collect the recognized events
together with each inferred risk relation and the risk rule
that identifies its type asscene-grounded risk evidence.
The VLM integrates a compact semantic serialization of
this evidence with camera observations and ego context for
decision-making and diffusion-based trajectory planning.
Image & Ego 
& Perception
Implicit VLM PipelineRelevant
Knowledge 
RetrieveInstantiation 
& GroundingApplicability 
VerificationReason
& Decision
DRKG-Supported Pipeline
Driving Risk Knowledge Graph (DRKG)
Scene -Fact Instantiation
Risk EntailmentRule Matching / ApplicabilityVLM
Reasoning
DecisionHigh
LowVLM Burden & 
Error LikelihoodDriving
Decision
Driving
Decision
Risk Evidence
Fig. 1. Typical RAG-based driving methods provide semantically relevant
knowledge while leaving scene grounding and applicability assessment
to the VLM, whereas our DRKG-supported framework explicitly derives
scene-grounded risk evidence before VLM reasoning and decision making.
We evaluate this distinction through controlled compar-
isons on nuReasoning [9], varying only the supplied risk
information. Across node- and text-based retrieval strategies,
expanding the candidate set recovers more activated risk
rules but also introduces more scene-inapplicable rules. The
added matches do not reliably improve planning, whereas
arXiv:2609.34145v1  [cs.RO]  28 Sep 2026

scene-grounded risk evidence yields better safety-weighted
planning, particularly in non-at-fault collision avoidance.
Comparisons of evidence representations further show that
scene-bound results and activated rule descriptions together
yield a higher planning score than either alone.
Our contributions are:
•We formulate the relevance-applicability gap in
knowledge-augmented vision-language driving, show-
ing that semantically relevant risk knowledge cannot
be treated as scene-supported evidence without ex-
plicit grounding, binding, relational verification, and
antecedent composition.
•We construct scene-grounded risk evidence by in-
stantiating structured perception as ontology-grounded
DRKG facts and applying closure-based SWRL reason-
ing. The evidence combines recognized events, directed
risk relations between bound risk sources and targets,
and semantic descriptions of activated rules for down-
stream VLM reasoning and planning.
•We demonstrate the downstream value of scene appli-
cability over semantic relevance. Controlled planning
comparisons show that scene-grounded risk evidence
improves the safety-weighted planning over relevance-
retrieved risk knowledge.
II. RELATED WORK
A. Knowledge Access for Vision–Language Driving
Retrieval-augmented driving has shown that external
knowledge can support explanation, decision-making and
planning. RAG-Driver retrieves expert demonstrations for
action explanation and prediction, RAD retrieves driving
experience for meta-action selection, and SafeDriveRAG
retrieves safety-knowledge subgraphs for visual question an-
swering [3], [4], [5]. KnowVal uses scene-derived keywords
to retrieve knowledge-graph entities, expands their neigh-
bouring nodes and filters the resulting driving clauses for
value-guided trajectory assessment [10]. DriveReg retrieves
traffic-regulation paragraphs by scene-conditioned text sim-
ilarity, refines the match at the sentence level and uses an
LLM reasoning agent to assess rule applicability and action
compliance [11]. These pipelines access related knowledge
through distinct mechanisms, but none exposes a risk relation
between bound scene entities that has been formally entailed
from scene facts.
B. Structured Knowledge and Formal Risk Reasoning
Research on structured and symbolic driving has shown
that scene entities, temporal states, relations and formal
conditions can be represented and evaluated outside a VLM.
DriveLM represents dependencies among perception, predic-
tion and planning through graph-structured question answer-
ing, while Hybrid-Driving combines a scenario-evolution
knowledge graph with rule-filtered action selection [6],
[8]. Formal systems also make condition checking explicit.
SGSM++ synthesizes runtime monitors from temporal safety
properties over scene graphs, while Toledoet al.monitor
such properties over spatial relations extracted by a VLM[12], [13]. Their resources and interfaces nevertheless re-
main task-specific. OD-RASE focuses on accident-causing
road structures, whereas formal monitors evaluate predefined
safety properties [14]. Together, these studies provide the
representations and reasoning mechanisms required for exter-
nal formal risk reasoning. However, an executable interface is
still needed to determine applicability across heterogeneous
risks and return recognized events together with object-
bound, type-identified risk relations.
III. METHOD
A. Problem Formulation
For a driving scenes, letO sdenote camera observations,
Psstructured perception results, andx sthe ego state. LetG
denote the fixed Driving-Risk Knowledge Graph (DRKG).
The task is to generate a planned ego trajectoryτ s.
To address the relevance–applicability gap, we derive
scene-grounded risk evidenceZ sfrom structured perception
and ego state usingG. The resulting evidence conditions
VLM decision-making and diffusion planning ofτ salong-
side camera observations and structured scene inputs. We
express these dependencies as
Zs=H(P s,xs;G),τ s= Π(O s,xs,Ps,Zs),(1)
Here,Hdenotesscene-applicability reasoning, which
instantiates scene facts, applies ontology and SWRL en-
tailment, and extracts recognized events and inferred risk
relations paired with their deriving rules. The relations bind
risk sources and targets in the scene, while the paired rules
identify the risk types.Πdenotesrisk-evidence-conditioned
decision and planning, integrating VLM decisions with
diffusion-based trajectory generation. WithinΠ, the VLM in-
tegratesZ swith visual and ego context, and its decision rep-
resentation guides the diffusion planner. The whole sequence,
shown in Fig. 2, places scene-applicability verification inH
before VLM decision-making, avoiding the misleading of
relevant yet inapplicable rules.
B. Scene-applicability Reasoning
1) Driving-Risk Knowledge Graph:We represent the
driving-risk knowledge base as
G= (T,R G),(2)
whereTis an OWL 2 ontology andR Gis a set of
positive SWRL rules defined overT[15], [16]. The ontology
fixes what can be represented, while the rules specify how
additional assertions can be derived from scene facts.
Building on ontology-based traffic-scene modelling [17],
[18],Tis organized into five connected modules. These
modules represent (i) static scene entities and road topology,
including the ego vehicle, traffic participants and scene
regions, (ii) temporal identity and frame order, (iii) temporal
and motion events and activities, (iv) unexpected events and
their causes, and (v) risk phenomena. Spatial, kinematic,
occlusion, topological, behavioural and interaction relations
connect entities within and across the modules. Data proper-
ties record measurable states such as position and velocity.

Front Image
Drive Command
Ego Status
Other StatusForward /Left / …
Vel. & Acc. & Hist. 
Pos. & Vel. & Hist. & 
Category. Driving -Risk Knowledge Graph
Ontology SWRL rules
Layer1: Participants
•Vehicle
•PedestrianEgo/Car/Truck/…
Scene-Fact 
InstantiationAdult/Child/…
Scene Risk 
Entailment
Ego
Car1OnRightFrontNoReasonFind (Car: brake)
(Obj, OnRightFront , Ego)
(Obj, CauseHarmTo , Ego)
Obj2: Risky
Prompt
Ego Status
Speed: 8.0m/s, …
Other Status
(Instantiated)
Car 1: x = 15.0m, …
Risk Evidence
[Risk Results] (2 pairs)
Unknown Obj 2 may harm 
ego
Unknown Obj 2 may harm 
Car 1
[Triggered Rules] (3)
1. If a vehicle brakes … no 
visible reason … unseen 
object …
2. …
[Event Detection] (1)
Car 1: brakeVision 
Language 
Models 
with Risk 
ConditionImage
Diffusion 
Planner
[x, y, heading]Fig. 2. Overview of the scene-grounded risk-evidence vision language-driving pipeline. Structured perception results and ego state instantiate scene facts
using the ontology of the driving-risk knowledge graph (DRKG). Its Semantic Web Rule Language (SWRL) rules derive scene events and risk relations
between scene entities. Recognized events, risk relations and associated rule descriptions are combined with ego and scene context in a prompt, while
camera observations provide the visual input to the vision–language model (VLM). The VLM decision representation conditions a diffusion planner to
generate the ego trajectory.
The risk vocabulary defines one binary predicate,λ risk. For
scene entitiese iande j,λrisk(ei, ej)states that the head
entitye iposes a risk to the tail entitye j.
The DRKG covers simple kinematic risks, cut-in and
crossing interactions, road-structure and right-of-way risks,
occlusion-related risks, and complex compositional risks.
Simple risks depend on a small number of motion condi-
tions, whereas compositional risks combine multiple entities,
temporal states and relations across ontology modules. We
denote byR risk⊆ RGthe risk rules whose conclusions
instantiateλ risk. These rules express different risk forms
while sharing the same conclusion predicate.
Each SWRL ruler∈ R Gcomprises an antecedent and a
conclusion, each written as a conjunction of positive atomic
propositions. The antecedent specifies the entity, temporal,
attribute and relational conditions that must be satisfied
before the conclusion can be derived. Intermediate rules may
conclude a scene event or a complex relation. A risk rule
r∈ R riskconcludesλ risk(ei, ej)for two traffic participant
entities. Deriving this conclusion establishes that every an-
tecedent ofris satisfied in the current scene. The head and
tail ofλ risk(ei, ej)identify the risk source and target, while
the identity ofrspecifies the risk type. The construction
procedure and category-level coverage are detailed in the
Supplementary Material.
2) Scene-Fact Instantiation:Scene-fact instantiation con-
verts structured perception into scene facts expressed with
the ontology vocabulary. Given perception resultsP sand ego
statex s, a scene adapterginitializes the directly obtainable
fact set
Fs=g(P s,xs;T)(3)The setF scontains entity and class assertions, measured
attributes, and simple relations supported directly by the
perception output. The rule setR Gis not applied at this
stage.
The adapter initializes the ego vehicle and each detected
or tracked traffic participant as a scene-local individual. It
also initializes the road and lane entities represented in the
structured inputs. Stable numeric identifiers preserve partic-
ipant identity across frames, while time-indexed attributes
record quantities such as position, velocity, acceleration and
heading.
Some simple spatial and kinematic relations are asserted
inF s. They are read from structured fields or computed
deterministically from measured states and their temporal
changes. These relations include relative positions, basic
motion states and manuevers without assigning a hazard
interpretation. Instances ofλ risk(ei, ej)and other complex
relations are derived in the subsequent SWRL stage.
3) SWRL-Based Scene Risk Entailment:Scene risk entail-
ment uses the SWRL rules inR Gto derive scene events and
complex relations fromF s, ultimately producing instances
ofλ risk(ei, ej)[16]. Mathematically,Cl T,RGdenotes the
closure operator induced by the ontologyTand rule set
RG. Starting fromF s, it repeatedly applies the ontology
and SWRL rules and adds each newly derived assertion to
the current fact set until no further assertion is produced.
Fig. 3 illustrates this progression from initial facts through
intermediate assertions to a scene-specific risk relation and
fianl SWRL closure fact set, and the process can be expressed
as
Fcl
s= ClT,RG(Fs)(4)

Scene FactSet 𝓕𝒔
Car(Ego)
Car(Car1)
InFrontOf (Car1, Car2)
Braking(Car1)
OnLane (Ego, Lane0)
IsRightLane (Lane0, Lane1)
Closure -Based SWRL Risk Entailment
Ontology 𝓣 SWRL Rule Set 𝓡𝑮
(concepts, relations, data properties) (risk patterns and domain knowledge)
Iterative Reasoning
(Pellet Reasoner)
𝓕(𝒕+𝟏)=𝓕(𝒕)∪𝚫(𝒕)
𝓕(𝟎)=𝓕𝒔
Init Facts𝓕(𝟏)=𝓕(𝟎)∪𝚫(𝟏)
+ derived events, 
event reasons
InNearFrontOf (Car1, 
Car2)
Reason(Evt1, Evt2)𝓕(𝟐)=𝓕(𝟏)∪𝚫(𝟐)
+ future events, 
unexpected events
MayBrake (Car2)
NoReasonFind (Evt2)
……𝓕(𝒕+𝟏)=𝓕(𝒕)∪𝚫(𝒕)
+ derived risk 
assertions
CauseHarmTo (Car
2, Ego)…
No new assertion 𝚫
Closure Fact Set 𝓕𝒔𝐜𝐥=𝑪𝒍𝓣,𝓡𝑮(𝓕𝒔)SWRL Rules
General Form
𝒇𝟏𝓕∧𝒇𝟐𝓕∧⋯∧𝒇𝑵𝓕→
𝜹𝟏𝓕 Antecedent
Consequent
Example Rule
𝐶𝑎𝑟?𝑓𝑟𝑜𝑛𝑡∧𝐶𝑎𝑟?𝑟𝑒𝑎𝑟
∧𝐼𝑛𝑁𝑒𝑎𝑟𝐹𝑟𝑜𝑛𝑡𝑂𝑓 ?𝑓𝑟𝑜𝑛𝑡,?𝑟𝑒𝑎𝑟
∧𝐵𝑟𝑎𝑘𝑖𝑛𝑔 ?𝑓𝑟𝑜𝑛𝑡
∧𝐶𝑟𝑢𝑖𝑠𝑖𝑛𝑔 ?𝑟𝑒𝑎𝑟
→𝐶𝑎𝑢𝑠𝑒𝐻𝑎𝑟𝑚𝑇𝑜 ?𝑓𝑟𝑜𝑛𝑡,?𝑟𝑒𝑎𝑟
If a front car is braking while a rear 
car is cruising nearby behind it, the 
front car is inferred to pose a risk to 
the rear carFig. 3. Closure-based SWRL risk entailment from scene facts. The initial fact setF sis expanded by applying the ontologyTand SWRL rulesR Guntil
no new assertion is derived. Intermediate events and relations lead to risk assertions in the closureFcl
s. The right panel shows the general rule form and
an illustrative vehicle-interaction rule.
Fcl
stherefore contains both the initial facts inF sand all
facts derived from them. The vocabulary remains fixed by
T. We implement this entailment process using the Pellet
reasoner [19].
4) Scene-Grounded Risk Evidence:After SWRL entail-
ment, we extract scene-grounded risk evidenceZ sfor down-
stream decision-making. It contains recognized scene events
and inferred risk relations paired with the rules that derive
them. The event setE scomprises temporal, motion and un-
expected events recognized by the third and fourth ontology
modules. We write
Zs=⟨E s,{(λ risk(ei, ej), r)}⟩(5)
The second component includes a pair for each inferred
relation and every rule that derives it in scenes. The relation
identifies the risk source and target, while the rule identifies
the risk type.
For VLM input, we serialize the recognized events and
inferred risk relations, with predefined semantic descriptions
mapping to replace the formal SWRL syntax of the associ-
ated risk rules with human-readable text. This representation
avoids requiring the VLM to parse complex conjunctions of
ontology atoms or repeat low-level kinematic facts already
available from perception.
C. Risk-Evidence-Conditioned Decision and Planning
1) Risk-Evidence-Conditioned VLM Reasoning:The vi-
sual inputO scomprises the current camera frame and four
historical frames sampled at 2 Hz. The VLM also receives
ego and scene contextc sand scene-grounded risk evidence
Zs. The two textual sources are concatenated using a fixed
prompt templateT. The resulting textual prompt is
ps=T(c s,Zs)(6)Conditioned onO sandp s, the VLM produces a four-level
output covering the scene, events, risks and driving decision.
We express this hierarchical output as
ys=Vθ(Os,ps) = 
yscene
s,yevent
s,yrisk
s,ydecision
s
(7)
whereV θdenotes the VLM. The decision level terminates
with a longitudinal meta-action token, a lateral meta-action
token and a plan token. Lethlon
s,hlat
sandhplan
s denote
their final-layer hidden states. Their ordered tuple forms the
decision representation
ds= 
hlon
s,hlat
s,hplan
s
(8)
which is passed to the diffusion planner as its VLM-derived
conditioning signal.
2) Conditioned Diffusion Planning:Because our VLM
primarily reasons about risk and high-level driving intent, we
provide the diffusion planner with complementary geometric
and dynamic context. We adapt the ReCogDrive diffusion
planner [20] to condition trajectory generation on this context
and ond s, which carries the VLM’s longitudinal, lateral
and planning signals. We then extract bird’s-eye-view (BEV)
scene features using the WorldEngine encoder [21], and
encode surrounding agents and ego motion separately. The
resulting conditions are
bs=W BEV(Os),
as=E agent(Ps),
us=E ego(xs,xhist
s).(9)
wherexhist
sdenotes the historical ego states,b sthe BEV
features,a sthe surrounding-agent features andu sthe ego-
motion features.

Starting from a noisy trajectoryτ(K)
s, the planner performs
conditional denoising at each stepk,
τ(k−1)
s =D ϕ
τ(k)
s, k|d s,bs,as,us
,τ s=τ(0)
s.
(10)
HereD ϕdenotes a denoising update, andτ sis the planned
ego trajectory. The VLM hidden states guide trajectory
refinement, BEV features enter through trajectory cross-
attention, and the agent and ego encodings provide additional
conditions.
IV. EXPERIMENTS
To examine the relevance–applicability gap in driving,
we ask whether retrieved risk rules support scene-specific
conclusions and whether this distinction affects planning.
Within the DRKG, SWRL activations provide the formal
reference for measuring recall and applicability of risk-
concluding rules and the rules along their derivation chains.
On nuReasoning, controlled planning comparisons vary the
supplied risk information, and an evidence ablation sep-
arates the contributions of inferred results and activated
rule descriptions. A qualitative occlusion case illustrates the
resulting risk reasoning and driving response.
A. Dataset and Evaluation Metrics
1) Evaluation Dataset:We use nuReasoning dataset be-
cause its long-tail driving scenes involve spatial relations and
agent interactions, and its planning benchmark allows us to
evaluate the downstream effect of risk evidence [9]. We split
the currently available training portion into 1,943 training,
259 validation and 387 test scenes, approximating a 75:10:15
split. Our local test split differs from the official benchmark
test set (which is unavailable now), so published scores on
that set provide context rather than a direct comparison.
2) Metrics:Within the DRKG, we use the SWRL rules
that contribute to each inferred scene risk as the retrieval
reference. We evaluate retrieval of risk rules that conclude
λriskand of all rules activated along the inference chains
leading to those conclusions. Specifically, we assess retrieval
quality using recall and applicability.
•Recall: Are risk-supporting rules retrieved?We aver-
age the fraction of reference rules retrieved over scenes
with an inferred risk.
•Applicability: Do retrieved rules support an inferred
risk?We average the fraction of retrieved rules in the
reference over scenes with nonempty retrieval, including
those without an inferred risk.
For planning, we adopt the nuReasoning Planning Score
(NPS), which combines five normalized component scores
as follows [9].
NPS =s NCsDA(0.3s EP+ 0.2s CF+ 0.5s HL),(11)
where NC and DA denote non-at-fault collision and
driveable-area compliance, respectively, and act as multi-
plicative safety gates. EP, CF, and HL denote ego progress,
comfort, and human-likeness, respectively. Following nuRea-
soning, we also report average displacement error (ADE)
over a five-second planning horizon.B. Implementation Details
Vision-Language Model.We initialize the VLM from
Qwen3-VL-8B [22]. We first conduct driving-domain pre-
training on question-answer pairs from LingoQA and CODA-
LM [23], [24]. This stage provides supervision for inter-
preting consecutive driving frames and acquiring driving-risk
knowledge. We then use GPT-5.4 to align training samples
from the nuReasoning training split and PotentialRiskQA
with the four-level output format defined above [9], [25]. Su-
pervised fine-tuning on the aligned samples trains the VLM
to generate risk-focused reasoning and the longitudinal meta-
action, lateral meta-action and plan tokens whose hidden
states condition the diffusion planner.
Diffusion Planner.We train the planner with a proximity-
weighted objective that combines trajectory accuracy, com-
mand consistency, comfort and collision avoidance. ForN
training clips, the objective is
LDP=1
NNX
s=1ws 
ωtrajL(s)
traj+ωcmdL(s)
cmd
+ωcomfL(s)
comf+ωcollL(s)
coll
,(12)
wherew sis the agent-proximity weight for scenesand the
weightsωterms balance the four losses.
The trajectory loss supervises predicted future waypoints
and endpoint states against the reference trajectory. The
command-consistency loss constrains terminal lateral dis-
placement and heading to agree with the VLM’s high-level
driving command. The comfort loss penalizes dynamically
undesirable trajectories, while the collision loss penalizes
collision between the ego vehicle and surrounding agents.
We assign largerw sto scenes with nearby traffic participants
to emphasize safety-critical interactions during training.
Planner training proceeds in two stages. We first train the
VLM-conditioned diffusion backbone to map the decision
representationd sto future ego trajectories. Then, we add
the BEV , surrounding-agent and ego-history conditions in
Eq. (9) and further fine-tune the planner. The second stage
thus refines an established VLM-to-trajectory mapping with
geometric and dynamic scene information.
TABLE I
RULE RECALL AND APPLICABILITY(APPL.)AGAINSTSWRL-DERIVED
SCENE REFERENCES(%).
Method Risk rules Risk chains
Recall↑Appl.↑Recall↑Appl.↑
KnowVal top-5 18.32 11.11 6.52 1.09
DriveReg top-5 29.30 12.58 27.05 1.55
KnowVal top-16 65.57 8.05 50.73 1.11
DriveReg top-16 33.88 4.04 29.51 0.58
SWRL-activated100.00 100.00 100.00 100.00
Baselines.The KnowVal-style baseline adapts keyword-
based node matching and graph expansion [10]. We index
each DRKG rule as a node using keywords and link nodes

TABLE II
PLANNING RESULTS WITH DIFFERENT RISK-INFORMATION SOURCES.
Risk information NC↑DA↑EP↑CF↑HL↑NPS↑ADE↓
KnowVal top-5 86.88 92.91 89.94 89.76 61.00 62.01 1.675
DriveReg top-5 87.2793.7089.91 89.76 60.76 62.53 1.676
KnowVal top-16 87.53 93.1889.95 90.29 61.5962.881.644
DriveReg top-16 87.14 93.18 89.59 89.24 61.11 62.49 1.681
Scene-grounded risk evidence (ours)90.2992.91 89.7990.2960.1764.181.728
TABLE III
PLANNING ABLATION OF SCENE-GROUNDED RISK-EVIDENCE REPRESENTATION.
Risk-evidence input NC↑DA↑EP↑CF↑HL↑NPS↑ADE↓
None 86.2292.91 90.1989.24 60.78 61.551.700
Inferred results 89.24 92.65 89.6290.55 61.1163.22 1.707
Activated rule descriptions 89.11 92.39 89.63 90.29 60.65 62.68 1.718
Inferred results and rule descriptions (ours)90.29 92.9189.79 90.29 60.1764.181.728
with similar keywords. VLM-extracted scene keywords re-
trieve seed nodes and their neighbours. The DriveReg-
style baseline adapts paragraph-then-sentence text retrieval
[11]. We group semantically similar rules into paragraphs,
match them against a VLM-generated scene summary, and
then retrieve individual rules within the selected paragraphs.
Both baselines search the same DRKG rule base used for
entailment at top-5 and top-16. Planning comparisons hold
camera observations, ego and scene context, the VLM, and
the diffusion planner fixed while varying only the supplied
risk information.
C. Rule Retrieval and Applicability
Broader semantic retrieval improved recall of rules con-
cludingλ riskbut reduced their scene applicability (Table I).
KnowVal recall rose from 18.32% at top-5 to 65.57% at top-
16, while applicability fell from 11.11% to 8.05%. DriveReg
showed the same pattern, with recall rising from 29.30%
to 33.88% and applicability falling from 12.58% to 4.04%.
Wider retrieval therefore covered more activated risk rules
while returning a larger share of rules unsupported by the
current scene.
Chain recall remained below risk-rule recall in every
baseline setting, including 50.73% versus 65.57% for Know-
Val top-16. Chain applicability was only 0.58%–1.55%,
compared with 4.04%–12.58% for rules concluding the risk
relation. Driving scenes repeatedly combine a limited set
of participant, behaviour and road elements, but risk rules
require distinct entity, temporal and relational bindings. Se-
mantic retrieval can therefore match a risk rule while missing
the event and relation rules that establish its scene-specific
conclusion.
Our method applies the full DRKG rule set to scene
facts and returns rules supporting entailed risks, rather than
selecting rules by similarity. Its 100% recall and applicabil-
ity follow by construction against the same SWRL-derived
reference, not from independent validation of perception or
rule correctness.D. Planning with Scene-Grounded Risk Evidence
The low applicability in Table I motivates testing scene-
grounded risk evidence against semantic retrieval in plan-
ning. Table II reports this comparison with camera observa-
tions, ego context, VLM and diffusion planner fixed.
Scene-grounded risk evidence achieved the highest NPS
(64.18) and NC (90.29) in Table II. KnowVal top-16, the
strongest semantic RAG setting by NPS, reached 62.88 and
87.53, respectively. NC was the only component mean to
rise relative to this setting. Because NC and DA gate NPS
multiplicatively in Eq. (11), the higher NC is consistent with
the safety-weighted score improvement.
The NPS gain was not accompanied by closer imitation
of the reference trajectory. Scene-grounded risk evidence
primarily changes safety-critical behaviour rather than tra-
jectory imitation fidelity.
Larger retrieval sets did not reliably improve planning
despite higher risk-rule recall. KnowVal NPS rose from 62.01
to 62.88, whereas DriveReg changed little, from 62.53 to
62.49. Both expanded sets lowered risk-rule applicability
(Table I), so broader coverage did not consistently translate
into planning gains, because the additional rules are often
scene-inapplicable.
E. Ablation of Scene-Grounded Risk Evidence Representa-
tion
To separate scene-bound conclusions from rule semantics,
we varied the content ofZ swhile holding activated rules
and both models fixed. The VLM received no risk evidence,
inferred events and bound risk relations, activated rule de-
scriptions, or the full risk evidence. Table III compares these
four representations without changing the activated rules.
Inferred results alone reached 63.22 NPS, slightly above
62.68 with activated rule descriptions alone (Table III). With
the activated rules fixed, inferred results supplied recognized
events and risk-source and risk-target bindings, whereas
descriptions supplied general rule semantics. This modest
difference suggests that explicit scene results add value
beyond rule text.

TABLE IV
SELECTED PLANNING BASELINES REPORTED ON NUREASONING[9]AND OUR METHOD.
Method NC↑DA↑EP↑CF↑HL↑NPS↑ADE↓
UniAD [26] 88.87 87.62 89.62 92.62 48.80 55.65 2.054
DiffusionDrive [27] 90.22 88.25 90.46 94.96 51.96 57.86 1.930
AutoVLA [28] 90.92 86.48 89.33 99.90 49.89 59.05 2.063
SpanVLA [29] 93.78 88.35 85.72 99.80 49.13 60.59 1.890
Alpamayo-1.5 (zero-shot) [30] 90.26 86.13 86.51 97.93 33.79 50.45 2.925
nuVLA (planning only) [9] 94.87 92.10 87.38 99.70 55.22 64.98 1.937
Ours (single-view VLM) 90.29 92.91 89.79 90.29 60.17 64.18 1.728
Ours:Rule R5-2-2-2-9: Ifamotor vehicle brakes
unexpectedly, andnosupported reason found, itmay
beanunseen object crossing infront ofit.
KnowVal :R4-0-1-13:Ifapedestrian islying
down and positioned directly infront ofavehicle,
then that pedestrian may cause harm tothevehicle .
R5-5-3-1-1-1:Ifpedestrian directly infront ofthe
egovehicle performs aroad-crossing event, then that
event isclassified asavulnerable -pedestrian road-
crossing riskevent .
Front Image (VLM Input)
Top-viewSudden 
Brake
Unseen 
Crossing 
PedestrianRetrieved Rules VLM Reasoning & Decision
Ours:…anunexpected braking vehicle tothefront -
right may beanunseen object crossing infront ofit.\n
[Decision] …tobrake hard while maintaining thecurrent
lateral path.Therefore <LONG_STRONG_DECEL> +
<LAT_STRAIGHT> .".
KnowVal :…apedestrian lying directly infront ofa
vehicle…apedestrian directly infront oftheegovehicle
crossing theroad…\n[Decision] Apply gentle deceleration
toincrease following distance and prepare forapossible
lateral intrusion while maintaining thecurrent lane path.
Therefore <LONG_GENTLE_DECEL> +<LAT_STRAIGHT> .".
Fig. 4. Qualitative comparison in an occluded pedestrian-crossing scene. The front camera shows a vehicle braking suddenly, whereas the top view reveals
a pedestrian hidden from the ego view. Our method identifies the applicable risk rule and selects strong deceleration. The KnowVal-style baseline retrieves
semantically related pedestrian rules (2 of top-5 are shown) and selects gentle deceleration.
Combining inferred results with rule descriptions yielded
the highest NPS (64.18) across the four ablations. Inferred
events and relations supplied scene-specific bindings, while
activated rule descriptions explained the risk type and ra-
tionale. Their complementary roles support the evidence
representation.
Table IV provides context for planner performance rather
than a direct leaderboard comparison. Published rows use the
official test set, while ours uses the evaluation split defined
in the Dataset subsection. Our single-view VLM planner
reached 64.18 NPS, 92.91 DA and 1.728 ADE, with CF at
90.29, establishing that the downstream planner is competi-
tive enough for controlled risk-information experiments.
F . Qualitative Results
Fig. 4 illustrates an occluded pedestrian-crossing risk in
a complex driving scene. The front camera shows another
vehicle braking suddenly, while the top view reveals a
pedestrian hidden from the ego view. Risk inference proceeds
from braking recognition through visible-cause checking,
unexpected-event classification and hidden-cause inference
to an ego-risk conclusion. Our method completed this chain
and activated the applicable rule, leading the VLM to select
strong longitudinal deceleration while maintaining its lane.
KnowVal instead detected the crosswalk in the image and
used it as a keyword to retrieve multiple pedestrian-related
rules. It did not check their scene-specific antecedents, and
the resulting gentle deceleration left the immediate occluded-
crossing risk insufficiently addressed.V. CONCLUSIONS
Semantic relevance alone cannot establish a scene-specific
driving risk because the rule’s antecedents must be supported
by current scene facts. Our framework makes this applica-
bility check explicit through DRKG fact instantiation and
SWRL entailment, supplying recognized events, directed risk
relations and activated rule descriptions to the VLM and dif-
fusion planner. On nuReasoning, broader semantic retrieval
increased risk-rule recall while reducing scene applicability.
Scene-grounded evidence improved NC from 87.53 to 90.29
and NPS from 62.88 to 64.18 relative to the strongest
semantic-retrieval condition. The evidence ablation further
showed that scene-bound conclusions and rule semantics
contribute complementary information to planning.
The framework relies on structured perception for scene-
fact instantiation, and noisy or missed observations can
change which SWRL antecedents are satisfied. It can entail
only risk forms covered by the DRKG rules, so incomplete
rule coverage limits the hazards represented in its evidence.
Future work will improve robustness to noisy perception,
extend the DRKG with broader, automatically acquired risk
knowledge, and enhance the generalization by VLMs.
REFERENCES
[1] J. Liu, L. Peng, X. Yan, L. Zhang, C. Yang, Y . Tao, A. Y . X. Tan, T. Xu,
S. Guo, H. Wang, and J. Li, “Enhancing autonomous vehicle safety
with knowledge graphs and large language models: Comprehensive
review,”Communications in Transportation Research, vol. 6, no. 2, p.
9640023, 2026.

[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdvances in Neural Information Processing Systems,
vol. 33, 2020, pp. 9459–9474.
[3] J. Yuan, S. Sun, D. Omeiza, B. Zhao, P. Newman, L. Kunze,
and M. Gadd, “RAG-Driver: Generalisable driving explanations with
retrieval-augmented in-context multi-modal large language model
learning,” inProceedings of Robotics: Science and Systems, Delft,
Netherlands, July 2024.
[4] Y . Wang, Q. Liu, Z. Jiang, T. Wang, J. Jiao, H. Chu, B. Gao,
and H. Chen, “RAD: Retrieval-augmented decision-making of meta-
actions with vision-language models in autonomous driving,” inPro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops, June 2025, pp. 3877–3887.
[5] H. Ye, M. Qi, Z. Liu, L. Liu, and H. Ma, “SafeDriveRAG: To-
wards safe autonomous driving with knowledge graph-based retrieval-
augmented generation,” inProceedings of the 33rd ACM International
Conference on Multimedia, 2025, pp. 11 170–11 178.
[6] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie,
J. Beißwenger, P. Luo, A. Geiger, and H. Li, “DriveLM: Driving with
graph visual question answering,” inComputer Vision – ECCV 2024,
2024, pp. 256–274.
[7] M. M. Hussien, A. N. Melo, A. L. Ballardini, C. S. Maldonado,
R. Izquierdo, and M. ´A. Sotelo, “RAG-based explainable prediction of
road users behaviors for automated driving using knowledge graphs
and large language models,”Expert Systems with Applications, vol.
265, p. 125914, 2025.
[8] J. Wang, Z. Wu, Q. Dong, L. Meng, Y . Xue, and Y . Yang, “Hybrid-
driving: An autonomous driving decision framework integrating large
language models, knowledge graphs and driving rules,”Proceedings
of the AAAI Conference on Artificial Intelligence, vol. 39, no. 1, pp.
826–833, 2025.
[9] Z. Huang, J. Liu, R. Song, Z. Zhou, R. Yang, Y . Zhang, T. Cai,
H. Zhang, M. Gao, V . Xu, J. Chen, Y . Shen, Y . Guo, T. X. Qi, and
J. Ma, “nureasoning: A reasoning-centric dataset and benchmark for
long-tail autonomous driving,”arXiv preprint arXiv:2605.31572, 2026.
[10] Z. Xia, W. Chen, Y . Wang, and M.-H. Yang, “Knowval: A knowledge-
augmented and value-guided autonomous driving system,” inProceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2026, pp. 3740–3749.
[11] T. Cai, Y . Liu, Z. Zhou, H. Ma, S. Z. Zhao, Z. Wu, X. Han, Z. Huang,
and J. Ma, “Driving with regulation: Trustworthy and interpretable
decision-making for autonomous driving with retrieval-augmented rea-
soning,”Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 40, no. 45, pp. 38 287–38 295, 2026.
[12] T. Woodlief, F. Toledo, S. Elbaum, and M. B. Dwyer, “The SGSM
framework: Enabling the specification and monitor synthesis of safe
driving properties through scene graphs,”Science of Computer Pro-
gramming, vol. 242, p. 103252, 2025.
[13] F. Toledo, S. Elbaum, D. Gopinath, R. Kaur, R. Mangal, C. S.
P˘as˘areanu, A. Roy, and S. Jha, “Monitoring safety properties for
autonomous driving systems with vision-language models,” in2025
IEEE Engineering Reliable Autonomous Systems (ERAS), 2025, pp.
1–8.
[14] K. Shimomura, M. Nambata, A. Ishikawa, R. Mimura, K. Inoue,
T. Yamashita, and T. Kawabuchi, “OD-RASE: Ontology-driven risk
assessment and safety enhancement for autonomous driving,” inPro-
ceedings of the IEEE/CVF International Conference on Computer
Vision, 2025, pp. 26 167–26 177.
[15] P. Hitzler, M. Kr ¨otzsch, B. Parsia, P. F. Patel-Schneider,
and S. Rudolph, “OWL 2 Web Ontology Language
Primer (Second Edition),” World Wide Web Consortium,”
W3C Recommendation, December 2012. [Online]. Available:
https://www.w3.org/TR/owl2-primer/
[16] I. Horrocks, P. F. Patel-Schneider, H. Boley, S. Tabet, B. Grosof,
and M. Dean, “SWRL: A semantic web rule language combining
OWL and RuleML,” World Wide Web Consortium,” W3C Member
Submission, May 2004. [Online]. Available: https://www.w3.org/
Submission/SWRL/
[17] M. Buechel, G. Hinz, F. Ruehl, H. Schroth, C. Gyoeri, and A. Knoll,
“Ontology-based traffic scene modeling, traffic regulations dependent
situational awareness and decision-making for automated vehicles,” in
2017 IEEE Intelligent Vehicles Symposium (IV), 2017, pp. 1471–1476.
[18] L. Huang, H. Liang, B. Yu, B. Li, and H. Zhu, “Ontology-baseddriving scene modeling, situation assessment and decision making
for autonomous vehicles,” in2019 4th Asia-Pacific Conference on
Intelligent Robot Systems (ACIRS), 2019, pp. 57–62.
[19] E. Sirin, B. Parsia, B. Cuenca Grau, A. Kalyanpur, and Y . Katz, “Pellet:
A practical OWL-DL reasoner,”Web Semantics, vol. 5, no. 2, pp. 51–
53, 2007.
[20] Y . Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen,
H. Sun, B. Wang, K. Ma, G. Chen, H. Ye, W. Liu, and X. Wang,
“ReCogDrive: A reinforced cognitive framework for end-to-end au-
tonomous driving,” inProceedings of the International Conference on
Learning Representations, 2026.
[21] T. Li, L. Chen, C. Wang, H. Liu, K. Chitta, Z. Yang, Y . Lu, N. Ye,
Y . Qiu, Y . Wang, L. Zou, J. Peng, J. Pan, Z. Su, A. Bursuc, S. E. Li,
A. Geiger, P. Su, and H. Li, “World engine: Towards the era of post-
training for autonomous driving,”arXiv preprint arXiv:2606.19836,
2026.
[22] S. Bai, Y . Cai, R. Chen,et al., “Qwen3-VL technical report,”arXiv
preprint arXiv:2511.21631, 2025.
[23] A.-M. Marcu, L. Chen, J. H ¨unermann, A. Karnsund, B. Hanotte,
P. Chidananda, S. Nair, V . Badrinarayanan, A. Kendall, J. Shotton,
E. Arani, and O. Sinavski, “LingoQA: Visual question answering for
autonomous driving,” inComputer Vision – ECCV 2024, 2024, pp.
252–269.
[24] K. Chen, Y . Li, W. Zhang, Y . Liu, P. Li, R. Gao, L. Hong, M. Tian,
X. Zhao, Z. Li, D.-Y . Yeung, H. Lu, and X. Jia, “Automated evaluation
of large vision-language models on self-driving corner cases,” in
Proceedings of the Winter Conference on Applications of Computer
Vision, 2025, pp. 7806–7815.
[25] J. Liu, X. Yan, L. Peng, L. Yang, L. Zhang, Y . Luo, Y . Tao, A. Y . X.
Tan, M. Li, L. Zhang, Z. Zhan, S. Guo, H. Wang, and J. Li, “Seeing
before observable: Potential risk reasoning in autonomous driving via
vision language models,”arXiv preprint arXiv:2511.22928, 2025.
[26] Y . Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,
T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y . Qiao, and
H. Li, “Planning-oriented autonomous driving,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 17 853–17 862.
[27] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li,
Y . Zhang, Q. Zhang, and X. Wang, “Diffusiondrive: Truncated diffu-
sion model for end-to-end autonomous driving,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2025, pp. 12 037–12 047.
[28] Z. Zhou, T. Cai, S. Z. Zhao, Y . Zhang, Z. Huang, B. Zhou, and J. Ma,
“Autovla: A vision-language-action model for end-to-end autonomous
driving with adaptive reasoning and reinforcement fine-tuning,”arXiv
preprint arXiv:2506.13757, 2025.
[29] Z. Zhou, R. Yang, X. Qi, Y . Guo, S. X. Chen, T. Feng, K. Pistunova,
Y . Shen, L. Su, and J. Ma, “Spanvla: Efficient action bridging and
learning from negative-recovery samples for vision-language-action
model,”arXiv preprint arXiv:2604.19710, 2026.
[30] Y . Wang, W. Luo, J. Bai, Y . Cao, T. Che, and et al., “Alpamayo-r1:
Bridging reasoning and action prediction for generalizable autonomous
driving in the long tail,”arXiv preprint arXiv:2511.00088, 2025.
SUPPLEMENTARYMATERIAL
DRKG Construction and Coverage
We constructed the DRKG from a hierarchical catalogue
of driving-risk scenarios. Scenarios with similar triggering
conditions, interacting entities and event sequences were
grouped into risk types. We mapped the required scene
entities and relations to the five ontology modules, then
encoded conditional event and risk derivations as SWRL
rules. Six experts reviewed the scenario grouping, ontology
and rules for semantic consistency.
Of the 177 SWRL rules in the DRKG, 39 are risk-specific
and cover 27 risk types. Table V reports the number of
risk types and rules in each category. Vars. and Atoms
give the mean numbers of unique variables and semantic
antecedent atoms per rule, with ranges in parentheses. Atom

counts exclude assertions used only for scene or temporal
membership.
TABLE V
COVERAGE AND COMPLEXITY OF RISK-SPECIFICSWRLRULES.
Risk category Types Rules Vars. Atoms
Vehicle interaction 8 13 6.38 (4–11) 11.08 (6–18)
Vulnerable road
users4 6 4.33 (4–6) 7.67 (7–10)
Road geometry and
traffic conditions11 14 6.00 (5–9) 10.64 (7–20)
Occlusion and
blind-spot risk4 6 4.67 (4–6) 9.00 (7–11)
Total 27 39 5.67 (4–11) 10.08 (6–20)