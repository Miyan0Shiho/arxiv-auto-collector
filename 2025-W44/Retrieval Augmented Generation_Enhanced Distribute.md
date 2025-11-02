# Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles

**Authors**: Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang

**Published**: 2025-10-30 08:23:08

**PDF URL**: [http://arxiv.org/pdf/2510.26242v1](http://arxiv.org/pdf/2510.26242v1)

## Abstract
With increasing urban traffic complexity, Traffic Signal Control (TSC) is
essential for optimizing traffic flow and improving road safety. Large Language
Models (LLMs) emerge as promising approaches for TSC. However, they are prone
to hallucinations in emergencies, leading to unreliable decisions that may
cause substantial delays for emergency vehicles. Moreover, diverse intersection
types present substantial challenges for traffic state encoding and
cross-intersection training, limiting generalization across heterogeneous
intersections. Therefore, this paper proposes Retrieval Augmented Generation
(RAG)-enhanced distributed LLM agents with Emergency response for Generalizable
TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning
framework, which dynamically adjusts reasoning depth based on the emergency
scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to
distill specific knowledge and guidance from historical cases, enhancing the
reliability and rationality of agents' emergency decisions. Secondly, this
paper designs a type-agnostic traffic representation and proposes a
Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3
adaptively samples training experience from diverse intersections with
environment feedback-based priority and fine-tunes LLM agents with a designed
reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies
across heterogeneous intersections. On three real-world road networks with 17
to 177 heterogeneous intersections, extensive experiments show that REG-TSC
reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle
waiting time by 83.16%, outperforming other state-of-the-art methods.

## Full Text


<!-- PDF content starts -->

Retrieval Augmented Generation-Enhanced
Distributed LLM Agents for Generalizable Traffic
Signal Control with Emergency Vehicles
Xinhang Li∗, Qing Guo∗, Junyu Chen∗, Zheng Guo∗, Shengzhe Xu∗, Lei Li∗, Lin Zhang∗†
∗School of Artificial Intelligence, Beijing University of Posts and Telecommunications, Beijing, China
†Beijing Big Data Center, Beijing, China
Abstract—With increasing urban traffic complexity, Traffic
Signal Control (TSC) is essential for optimizing traffic flow and
improving road safety. Large Language Models (LLMs) emerge
as promising approaches for TSC. However, they are prone to
hallucinations in emergencies, leading to unreliable decisions that
may cause substantial delays for emergency vehicles. Moreover,
diverse intersection types present substantial challenges for traffic
state encoding and cross-intersection training, limiting general-
ization across heterogeneous intersections. Therefore, this paper
proposes Retrieval Augmented Generation (RAG)-enhanced dis-
tributed LLM agents with Emergency response for Generalizable
TSC (REG-TSC). Firstly, this paper presents an emergency-
aware reasoning framework, which dynamically adjusts reason-
ing depth based on the emergency scenario and is equipped
with a novel Reviewer-based Emergency RAG (RERAG) to distill
specific knowledge and guidance from historical cases, enhancing
the reliability and rationality of agents’ emergency decisions.
Secondly, this paper designs a type-agnostic traffic representation
and proposes a Reward-guided Reinforced Refinement (R3)
for heterogeneous intersections. R3adaptively samples training
experience from diverse intersections with environment feedback-
based priority and fine-tunes LLM agents with a designed
reward-weighted likelihood loss, guiding REG-TSC toward high-
reward policies across heterogeneous intersections. On three real-
world road networks with 17 to 177 heterogeneous intersections,
extensive experiments show that REG-TSC reduces travel time by
42.00%, queue length by 62.31%, and emergency vehicle waiting
time by 83.16%, outperforming other state-of-the-art methods.
Index Terms—generalizable traffic signal control, LLMs, re-
trieval augmented generation, emergency-aware reasoning
I. INTRODUCTION
Traffic Signal Control (TSC) plays a critical role in ur-
ban traffic management by improving the traffic efficiency
and relieving congestion [1]. Traditional methods, such as
transportation-based and Reinforcement Learning (RL)-based
methods [2] have shown great potential in TSC for structured
intersections with simple traffic patterns. The emergence of
Large Language Models (LLMs) has partially mitigated the
generalization limitations of traditional methods. However,
such generalization is mostly limited to variable traffic flow
and diverse road network structures with simplified same inter-
sections [3], [4]. LLMs still struggle to manage heterogeneous
intersections and to reason reliably in emergency scenarios.
These limitations remain key barriers to the research of LLM-
based traffic signal optimization agents.
Corresponding author: Lei Li (leili@bupt.edu.cn).Emergency scenarios, due to suddenness and urgency, intro-
duce great complexities to TSC and have become a research
focus. [5] employed Deep Q-Network (DQN) to balance the
impact of emergency vehicles on regular traffic flow. [6] and
[7] both proposed novel vehicle-road coordination schemes
to simultaneously manage signal phases and the routing of
emergency vehicles. Wang et al. adopted multiple LLM agents
to assess traffic phases, prioritize emergency vehicles, and
verify rule compliance [8]. However, the above methods are
tailored specifically for emergency vehicles and are not appli-
cable to generalizable traffic signal optimization. In contrast,
several studies consider the occurrence of emergency vehicles
as stochastic events in the traffic [9]–[11], aligning with real-
world conditions. In particular, [10] and [11] leveraged LLMs
to assist or refine RL decision-making in emergency scenarios.
However, the LLM agents in the above methods are prone
to hallucinations and follow fixed reasoning modes across all
scenarios, limiting policy reliability and efficiency.
Retrieval Augmented Generation (RAG), as an advanced
technique in LLMs, enables the incorporation of domain-
specific knowledge to enhance LLMs’ performance on special-
ized tasks [12]. Additionally, with the support of RAG, the hal-
lucination problem of LLMs is mitigated through referenced
knowledge [13]–[15]. For TSC, hallucinations may result
in unsafe or inefficient decisions, posing considerable risks
[16]. Particularly in safety-critical emergency scenarios, the
rationality and reliability of policies are paramount. However,
existing RAG merely retrieve raw information without effec-
tive verification or distillation, unsuitable for highly dynamic
urban traffic with emergency vehicles.
In addition, RAG supports the extension of Chain of
Thought (CoT), allowing LLMs to perform deep and fine-
grained reasoning by extracting additional valuable informa-
tion from prompts and retrieved knowledge [17]. However, the
reasoning with fixed reasoning depth across all intersections
is inefficient. Simple scenarios do not require intensive com-
putation, while emergency scenarios demand more nuanced
analysis. This one-size-fits-all mode not only wastes compu-
tational resources but also hinders real-time decision-making.
Therefore, a key challenge lies in balancing computational
efficiency with decision-making effectiveness.
Recent RL-based TSC methods have increasingly explored
generalization across heterogeneous intersections. These stud-arXiv:2510.26242v1  [cs.AI]  30 Oct 2025

Fig. 1. Architecture of REG-TSC. (a) Observation Collection obtains traffic states and converts them into natural language representations. In (b) Emergency-
Aware Reasoning Framework, (c) Reviewer-Based RAG retrieves critical guidance based on the current traffic and emergency vehicle states. (d) LLM-Based
Signal Optimization Agent performs deep reasoning by integrating the guidance with traffic representations. (e) Simulation-Driven Fine-Tuning is conducted
in two stages: imitation fine-tuning and reward-guided reinforced refinement.
ies mainly improve uniform state and action representations
[18], [19] and optimize transfer training [20], [21], aiming
to adapt RL to varying intersection layouts. Specifically,
[18] employed attention mechanisms to flexibly extract traffic
dynamics at heterogeneous intersections. [19] designed an
encoder-decoder structure to project intersection states into
a unified space. [20] proposed a general scenario-agnostic
RL framework to conduct a large-scale co-training with mul-
tiple scenarios. [21] allowed heterogeneous intersections to
share Control Knowledge. However, as an emerging paradigm,
LLMs still face considerable challenges in unified representa-
tion and transfer learning across heterogeneous intersections.
In summary, emergency scenarios and heterogeneous in-
tersections remain a major obstacle to complex TSC for
LLM-based methods. Therefore, this paper proposes RAG-
enhanced distributed LLM agents with Emergency response
for Generalizable TSC (REG-TSC), as shown in Fig. 1. The
proposed emergency-aware reasoning framework employs a
novel Reviewer-based Emergency RAG (RERAG) to effi-
ciently extract knowledge from historical scenarios to form
a guidance repository. And the appropriate guidance retrieved
from the repository is embedded into the prompts, enabling
the LLMs to perform deep reasoning for effective emergency
response. Besides, during the training process, we design a
Reward-guided Reinforced Refinement (R3) that dynamically
rebalances the training data according to the simulation re-
sults and fine-tunes REG-TSC with reward-weighted loss,
improving generalization across heterogeneous intersections.
The contributions of this paper are summarized as follows.
•This paper proposes REG-TSC for generalizable signal
optimization in urban dynamic traffic. Empowered by
the emergency-aware reasoning framework, REG-TSC
enables LLM agents to perform deep reasoning with
RERAG, generating rational and reliable emergency re-
sponses with critical knowledge and guidance.•This paper designs a type-agnostic traffic representation
to standardize static and dynamic features across het-
erogeneous intersections. We further propose R3, which
adaptively samples the training dataset with priority based
on environmental feedback and refines REG-TSC with a
designed reward-weighted likelihood loss, thereby boost-
ing performance across diverse intersection types.
•This paper constructs three real-world road networks
with the number of intersections varying from 17 to
177. Extensive experiments show that REG-TSC reduces
travel time by 42.00%, queue length by 62.31%, and
emergency vehicle waiting time by 83.16% on average
compared with other state-of-the-art (SOTA) methods.
II. GENERALIZABLETSCWITHEMERGENCYVEHICLES
A. Road Networks with Heterogeneous Intersections
Real-world urban road networks consist of various hetero-
geneous intersections differing in shapes and lane layouts.
This paper constructs high-authenticity urban road networks,
covering four common intersection shapes with different lane
layouts: cross intersections, T-intersections, Y-intersections,
and roundabouts. There areNtypes of heterogeneous inter-
sections. The following are key definitions for representing
traffic networks with heterogeneous intersections.
Definition 1 (Road Network):The road network is modeled
as a directed graph, where nodes correspond toIintersections
and edges represent roads. Each road contains a varying
number of lanes, and each lane has a fixed direction of travel.
Definition 2 (Heterogeneous Intersections):At intersection
i, there areUL iupstream lanes andDL idownstream lanes.
Specifically, the upstream lanes through which vehicles enter
intersectioniare denoted asUL i={ulj
i|j= 1,2, ..., UL i},
and the downstream lanes through which vehicles exit inter-
sectioniare denoted asDL i={dlk
i|k= 1,2, ..., DL i}.

Fig. 2. An illustration of heterogeneous intersections and signal phases.
Definition 3 (Traffic Movement):A traffic movement is
defined as a vehicle crossing an intersection from one upstream
lane to one downstream lane. A movement from laneulj
ito
lanedlk
iis denoted as(ulj
i, dlk
i). At intersectioni, there are
TMitraffic movementsT M i⊆ UL i× DL i.
Definition 4 (Traffic Signal Phase):A traffic signal phase
is defined as a set of non-conflicting traffic movements that
are allowed to travel simultaneously. The set of all phases
at intersectioniis denoted asPH i={PHm
i|m=
1,2, . . . , J i}, whereJ iis the total number of signal phases
of intersectioni. Fig. 2 illustrates signal phases for different
types of heterogeneous intersections.
B. TSC with Emergency Scenarios
In TSC with sudden emergency scenarios, we consider an
urban traffic network consisting ofIintersections, where reg-
ular traffic flows coexist with emergency vehicles that appear
unpredictably. The simulation runs forTcontinuous time
steps, during whichMemergency vehicles are generated at
random time steps, each following a random predefined route
through the road network. Each intersection is controlled by a
distributed LLM agent, which dynamically selects appropriate
signal phases based on traffic conditions. The objective is
to optimize signal phase decisions across all intersections
to allow emergency vehicles to pass through as quickly as
possible and improve the overall traffic condition.
III. EMERGENCY-AWAREREASONINGFRAMEWORK
A. Reviewer-Based Emergency RAG
To enhance adaptive emergency-response decision-making,
we design RERAG, an advanced module that integrates an
LLM-based reviewer, as shown in Fig. 1 (c). Equipped with the
reviewer, RERAG distils critical knowledge and guidance from
historical emergency scenarios and expert-driven emergency
response, enabling TSC agents to generate rational and reli-
able strategies under current traffic conditions. The proposed
RERAG is composed of the following four components.1) LLM-Based Reviewer:The reviewerRevis responsible
for extracting knowledge from a raw case baseBof historical
emergency scenarios to form a structured and generalizable
guidance repository. This paper employs a GPT-4o Mini as
the reviewer to summarize expert-driven responses. Specif-
ically, the input toReis a historical case base including
intersectionitraffic representation of current and next time
steps, emergency vehicle states, expert-driven signal phase
selection,b l= [obsi
t, Evt, ai
t, obsi
t+1, Evt+1]. Guided by well-
designed prompts, the reviewerRevidentifies the conditions,
recommended actions, and intended effects of the guidance
by analyzing control strategies and causal relationships in the
case base. The reviewer outputs a set of guidance itemsg d.
The process is formally expressed asG=Re({b 1, ..., b L}),
whereLandDare the lengths of the historical emergency
case base and the guidance repository, respectively.
2) Embedding Module:The embedding module bridges
the guidance repository and retrieval by converting reviewer-
generated guidanceg dinto a vector,v gd=Em(g d). This
paper adopts a sparse mixture-of-experts-based complex text
encoder with dynamic position encoding as the embedding
module. It is able to understand the relative positioning of
information in extended sequences.
3) LLM-Based Query Generator:The query generator
QGen, built upon GPT-4o Mini, interprets the current traffic
context and generates semantically rich queries for knowledge
retrieval. Given the intersection stateobsi
tand emergency
vehicle stateEv t,QGenformulates a concise queryq tthat
highlights key features of the emergency situation. The gen-
eration process is defined asq t=QGen(obsi
t, Evt).
4) Retrieval Module:The retrieval module identifies the
top-Kmost relevant guidance items from the vectorized
guidance repositoryv G={v g1, ..., v gD}based on the current
queryq t. The query is embedded asv qt=Em(q t)using the
same encoder as for guidance items. The similarity between
vqandv gdis computed via cosine similarity,
sim(v qt, vgd) =vq·vgd
|vq||vgd|.(1)
Based on similarity scores, the relevant guidance itemsRGi
t=
{gd1, ..., g dK}are retrieved to assist the signal phase deci-
sion. The retrieval process can be formalized asRGi
t=
Retrieve(v q, vG). This ensures that the decision is informed
by emergency guidance aligned with the current scenario.
B. RERAG-Based Emergency Response with Deep Reasoning
To handle complex and time-critical emergency scenarios,
the emergency-aware reasoning framework enables REG-TSC
to perform deep reasoning with the guidance retrieved by
RERAG. The deep reasoning enhances the interpretability
and reliability of emergency response, via comprehensively
analyzing the current intersection state and retrieved guidance.
The deep reasoning process is selectively triggered under
emergency-aware conditions. Specifically, it is enabled only
when an emergency vehicle is either currently at intersection
i, or when intersectioniappears in the planned path of the

emergency vehicle. This selective activation ensures that deep
reasoning is used where it is required most, while reducing
unnecessary computational overhead during regular scenarios.
For emergency scenarios, the emergency-aware reasoning
framework firstly leverages RERAG to retrieve relevant guid-
anceRGi
t. The retrieved guidance, together with the current
traffic representationobsi
tand emergency vehicle stateEv t,
is then fed into the LLM-based signal optimization agent to
conduct deep reasoning. The emergency vehicle stateEv t
includes its planned route, its current position, and its current
speed.
Within the LLM-based signal optimization agent, a carefully
constructed prompt guides the deep reasoning, as shown in
Fig. 1 (d). The deep reasoning follows a three-step CoT,Step
1:analyzing the current intersection state and the information
of the emergency vehicle with guidance,Step 2:evaluating
each traffic signal phase by predicting emergency vehicle
arrival time and future queue lengths,Step 3:determining
the appropriate phase selection and providing explanation. The
deep reasoning process is represented as
Anai
t, Prei
t, Ai
t=πEmer
θ (obsi
t, Evt,RGi
t),(2)
whereAnai
t,Prei
tandAi
tdenote analysis, prediction and
phase decision with explanation. Deep CoT focuses REG-TSC
on the emergency vehicle, enabling fine-grained reasoning via
analysis and prediction. To reduce overhead, REG-TSC per-
forms lightweight reasoning in regular scenarios, generating a
decision with explanation,Ai
t=πRegu
θ(obsi
t).
IV. DISTRIBUTEDLLM AGENTS INGENERALIZABLETSC
A. Generalizable TSC for Heterogeneous Intersections
To enable generalization across diverse intersections, we
design a type-agnostic traffic representation encoding both
static and dynamic features. The representationobsi
tis con-
structed from a lane-centric perspective, where each intersec-
tion is decomposed into lanes, allowing every agent to reason
consistently across structural and action-space heterogeneity.
Specifically,obsi
tis composed of the following four parts.
•Intersection topologydescribes the shape of intersection
i, and the traffic movements defined among lanes.
•Action spaceis specified as a set of signal phases, each
mapped to its controlled traffic movements.
•Queuing vehiclesrefer to the number of vehicles stopped
or slowly moving on lanes controlled by each phase.
•Approaching vehiclesdenote vehicles moving faster than
a thresholdv stopwithin a certain segmentsof each lane.
In addition toobsi
t, the prompt fed to the LLMs includes
a concise task description and commonsense knowledge to
guide the generation of optimal control strategies. A detailed
example of the prompt is provided in Appendix A-A.
The LLM-based signal optimization agent outputs the se-
lected signal phase along with a reasoning trajectory. The
phase decision is explicitly marked in the output, allowing
REG-TSC to easily extract the actionai
tand forward it to
the traffic simulator. The simulator then returns a rewardri
tand transitions to the next stateobsi
t+1, enabling ongoing
interaction and learning. The rewardri
tis defined as,
ri
t=λ1QLi
t−QLi
t+1
QLi
t+1+λ2τ−WTEi
t
WTEi
t+γ,(3)
whereQLi
tandWTEi
tdenote the queue length at intersection
iand the waiting time of emergency vehicle during phaseai
t.
B. Simulation-Driven Fine-Tuning
1) Imitation Fine-Tuning:This paper performs imitation
fine-tuning on Llama-3.1-8B to specialize a general LLM into
a TSC agent by imitating high-quality decisions.
Firstly, GPT-4o Mini interacts with a traffic simulator to
collect reasoning trajectories. For intersectioni, GPT-4o Mini
generates a reasoning trajectorycYi
twith a CoT and selected
action according to promptXi
t. To ensure data quality, we ap-
ply areward-based filterthat retains only effective trajectories
aligned with long-term traffic optimization. Specifically, a tra-
jectory is preserved if the cumulative reward over the nextt re
steps exceeds a predefined thresholdη, i.e.Ptre−1
k=0ri
t+k≥η.
These high-quality trajectories are stored into a fine-tuning
dataset. Subsequently, we conduct off-line supervised training
by minimizing the negative log-likelihood (NLL) loss,
LNLL=−|cYi
t|X
ω=1logP πθ(dYi
t,ω|Xi
t,[Yi
t,<ω),(4)
wheredYi
t,ωis theω-th token in the response. Low-rank adapta-
tion is adopted for parameter-efficient fine-tuning. The process
transfers GPT’s reasoning logic to REG-TSC, providing a
TSC-specialized initialization for subsequent training.
2) Reward-Guided Reinforced Refinement:To boost adapt-
ability to diverse intersections, we propose R3, which adap-
tively focuses training on challenging intersection types.
In R3, REG-TSC interacts with the simulator acrossN
heterogeneous intersection types. At time stept, REG-TSC
generates a reasoning trajectoryYi
taccording to a prompt
Xi
tand receives a rewardri
tfrom the simulator. The tuple
(Xi
t, Yi
t, ri
t)is then stored into the experience bufferB ncorre-
sponding to intersection typen. To emphasize low-performing
types, we define the sampling probabilitySPr nas
SPr n=1/(¯r n+|min(¯r)|+ε)
NP
k=1(1/(¯r k+|min(¯r)|+ε)).(5)
Prioritized samples drawn from all experience buffers are
combined into a refinement dataset. A reward-weighted NLL
loss is employed to guide the refinement,
LRNLL =−X
(Xi
t,Yi
t,ri
t)ri
t·|Yi
t|X
ω=1logP πθ(Yi
t,ω|X, Yi
t,<ω).(6)

C. REG-TSC Learning and Reasoning Process
The learning and reasoning process of REG-TSC is shown
in Algorithm 1. Firstly, REG-TSC is initialized and trained
with trajectories generated by GPT interacting with a traffic
simulator. These trajectories are filtered viareward-based filter
to build a high-quality imitation dataset. Secondly, during rein-
forced refinement, emergency intersections retrieve guidance
and perform deep reasoning accordingly, while intersections
in regular scenarios follow the lightweight reasoning. After
executing actions and observing rewards, the transitions are
stored in corresponding experience buffers. REG-TSC is pro-
gressively refined with prioritized samples.
Algorithm 1:REG-TSC Learning and Reasoning
1Initialize REG-TSC and imitation datasetD;
2Interact GPT with a traffic simulation environment ;
3Collect reasoning trajectory(Xi
t,cYi
t)and store toD;
4FilterDwith reward-based filter;
5Fine-tune REG-TSC withDand Eq. (4);
6Initialize experience buffers{B n}and vectorizeG;
7fore=1:max epochdo
8Initialize a traffic simulation environment;
9fort=1:Tdo
10fori=1:Ido
11ifiin emergency scenariothen
12Getobsi
t, Evtfrom the environment;
13Get query and retrieveRGi
tvia Eq. (1);
14ReasonAnai
t, Prei
t, Ai
tvia Eq. (2);
15end
16else
17Getobsi
tfrom the environment;
18ReasonAi
tviaπRegu
θ(obsi
t);
19end
20end
21Extract{ai
t}to perform and observe{ri
t};
22Append(Xi
t, Yi
t, ri
t)to correspondingB n;
23end
24Sample training data from{B n}via Eq. (5);
25Perform reinforced refinement via Eq. (6);
26end
V. EXPERIMENTS ANDRESULTS
A. The Simulator and Settings
To thoroughly validate the effectiveness and robustness
of REG-TSC, we construct three road networks based on
SUMO [22], corresponding to parts of the roads in Jinan,
Hangzhou, and Yizhuang (Beijing), as shown in Fig. 3. The
parameters of traffic flow datasets are presented in TABLE I.
The settings of REG-TSC and simulator are shown in TABLE
II. The evaluation of methods is based on the following
metrics: the average travel time (ATT) and average waiting
time (AWT) of all vehicles, the average queue length (AQL)at all intersections, and the average travel time (ATTE) and
average waiting time (AWTE) of emergency vehicles.
Fig. 3. Simulated Urban Road Networks.
TABLE I
TRAFFICFLOWDATASETS
Traffic Flow DatasetIntersection
NumberIVehicle
NumberArrival Rate
(vehicle/min)
Jinan1171710 57.14
Jinan2 2400 80.00
Hangzhou1191740 58.03
Hangzhou2 2440 81.30
Yizhuang1
1773600 120.00
Yizhuang2 8000 266.67
Yizhuang Extreme 10500 350.88
TABLE II
ALGORITHM ANDSIMULATIONPARAMETERS
Parameters Value Parameters Values
λ1 5 λ2 1
τ5 γ1
η0.5 ε0.1
K1 Learning Rate3×10−4
T1800 M6
B. Ablation Experiments
Ablation experiments are conducted to explore the effects of
emergency-aware reasoning framework and R3, shown in the
last three columns of TABLE III. Method A represents REG-
TSC only trained via imitation fine-tuning without R3. Method
B represents REG-TSC without emergency-aware reasoning.
REG-TSC outperforms Method A under varying traffic
conditions. In Jinan2, REG-TSC reduces ATT by 17.92%
and AQL by 14.91% compared with Method A. In Yizhuang
network, the AWT and AWTE of Method A are 21.52% and
38.96% more than those of REG-TSC on average. Therefore,
R3serves an essential role in optimizing policies across
heterogeneous intersections.
Method B shows a limited effectiveness in handling emer-
gency scenarios. Across the tested six scenes, AWTE of
Method B exceeds that of REG-TSC by 86.13% on average.
Specifically, in Hangzhou1, Method B records an AWTE
82.00 seconds longer than REG-TSC. It indicates that the
emergency-aware reasoning framework enables REG-TSC to
deliver reliable decisions in emergency scenarios.

TABLE III
EVALUATIONRESULTS
Methods MPLight AttendLight PressLight CoLightEfficient-
ColightAdvanced-
ColightDeepSeek-
R1-8BGPT-
4o MiniLLMLightLlama 3.1
-8BMethod
AMethod
BREG-TSC
Jinan1ATT 272.63 811.42 275.19 795.17 709.13 677.13 309.63 245.17 222.10 274.64 266.38 231.74214.04
AWT 165.03 773.99 168.79 746.43 590.99 611.58 202.82 132.24106.23167.48 158.22 120.13 110.10
AQL 20.72 72.40 21.10 54.18 19.37 52.30 19.76 12.84 11.71 16.84 15.23 12.3011.56
AWTE 320.50 1081.67 132.33 1039.17 534.17 1121.83 128.17 103.33 111.50 369.67 62.50 29.5011.83
Jinan2ATT 301.82 743.30 289.66 797.97 768.17 716.03 378.27 296.97 248.35 360.00 290.70 254.46238.62
AWT 198.13 689.88 186.79 740.38 699.52 659.28 256.60 176.13 130.94 244.67 173.46 142.00127.90
AQL 24.72 73.66 23.60 57.35 56.85 59.99 34.57 22.0418.3429.31 21.66 20.46 18.43
AWTE 300.83 609.83 154.67 755.17 1054.83 905.17 175.50 172.50 81.33 442.00 133.33 63.1754.50
Hangzhou1ATT 262.76 800.02 278.23 762.18 526.24 611.79 382.95 276.75 258.98 315.97 270.04 254.29246.72
AWT 136.53 740.82 152.14 701.25 422.16 523.49 243.86 142.01 124.17 189.07 135.55 121.48120.28
AQL 16.17 57.57 18.66 57.52 45.65 46.58 22.33 12.94 12.38 17.39 12.31 11.6911.13
AWTE 112.00 1008.33 145.50 1245.50 269.17 593.33 154.83 81.33 184.33 287.33 93.00 110.8328.83
Hangzhou2ATT 312.55 794.47 297.85 839.87 522.19 655.17 387.85 292.91 290.29 349.89 339.19 303.50284.25
AWT 176.27 740.62 165.14 792.94 411.64 560.66 252.80 161.20148.74215.69 205.32 175.82 152.57
AQL 25.81 93.01 32.12 81.72 51.63 67.93 28.56 20.9818.6726.44 24.56 20.77 20.30
AWTE 129.17 1338.33 240.50 1269.67 534.17 651.00 128.50111.17246.67 168.00 225.83 149.83 129.67
Yizhuang1ATT 462.94 859.63 497.51 868.73 436.01 830.12 597.92 470.16 452.59 485.87 454.42 427.45401.59
AWT 231.88 796.51 274.59 792.75 343.45 721.02 378.94 212.82 193.43 259.40 196.62 177.11169.80
AQL 13.29 26.16 14.27 22.62 40.30 24.44 6.17 3.82 3.57 4.76 3.55 3.463.43
AWTE 186.67 804.17 118.00 305.67 373.17 644.17 351.50 218.33 168.67 452.83 252.33 146.24127.45
Yizhuang2ATT 545.12 882.65 662.68 872.58 754.29 863.98 561.82 561.75 521.01 580.25 537.46 517.71478.30
AWT 327.17 830.13 469.63 803.09 578.01 790.60 341.41 292.53 256.42 337.13 318.73 291.86225.00
AQL 27.61 43.77 31.11 41.74 33.36 42.14 14.17 11.1110.2412.39 10.56 10.47 10.32
AWTE 224.17 1310.00 665.00 1269.67 732.83 1260.67 710.83 264.67 373.83 615.17 184.67 180.83132.17
Fig. 4. ATTE on Jinan and Hangzhou Road Networks.
Fig. 4 provides a clear comparison of ATTE. In Jinan1,
REG-TSC shortens ATTE by 54.17 seconds compared with
Method A. Equipped with emergency-aware reasoning, REG-
TSC achieves 12.28% shorter ATTE than Method B in
Hangzhou2. Across all scenes, REG-TSC maintains the short-
est ATTE, demonstrating strong emergency response ability.
In addition, in Jinan1, the average inference time of REG-
TSC is 4.07 seconds per time step, similar to that of GPT-4o
Mini measured at 3.85 seconds. This shows that REG-TSC
has excellent deployability and low computational overhead.
C. Comparison with Other TSC Methods
We compare REG-TSC with six RL-based approaches:
MPLight, AttendLight [18], PressLight, CoLight, Efficient-
CoLight, and Advanced-CoLight; and a SOTA LLM-based
method: LLMLight [17]. Furthermore, we evaluate the per-
formance of general LLMs integrated within our emergency-
aware reasoning framework, including Deepseek-R1-8B, GPT-
4o Mini, Llama 3.1-8B. The results are shown in TABLE III.
REG-TSC consistently achieves SOTA or comparable per-
formance against all baselines, highlighting outstanding emer-
gency vehicle handling and strong generalization capabili-
ties. Specifically, in Hangzhou1, REG-TSC achieves 3.13%and 10.10% lower AWT and AQL, respectively, compared
with LLMLight, the second-best method. For the complex
Yizhuang2, REG-TSC reduces AWTE by 69.99% on aver-
age, surpassing the other three general LLMs. Moreover,
LLM-based methods basically outperform RL-based meth-
ods, indicating LLM’s superior generalization in optimizing
signal phases across heterogeneous intersections. Notably,
by learning from specific scenes and traffic data, MPLight
and PressLight can effectively develop pressure-based signal
phase policies, achieving reasonable decision-making. How-
ever, their performance still lags behind REG-TSC. For ex-
ample, across six scenes, MPLight’s ATT and AWTE are on
average 13.85% and 54.14% higher than those of REG-TSC.
The results in Fig. 4 highlight REG-TSC’s superior ability
to facilitate rapid emergency vehicle passage compared with
both LLM-based and traditional baselines. For example,in
Hangzhou1, REG-TSC reaches an ATTE of 174.67 seconds,
outperforming GPT-4o Mini and MPLight by margins of 71.66
seconds and 63.66 seconds, respectively.
Fig. 5. Travel Time of Emergency Vehicles (M= 20).

To further evaluate the ability of different methods to handle
emergency scenarios, we set the number of emergency vehicles
to 20 and measure their arrival time on the Jinan and Hangzhou
road networks, as shown in Fig. 5. In Jinan1, REG-TSC
achieves a median travel time of around 180 seconds, which is
substantially shorter than Llama 3.1-8B at about 450 seconds
and GPT-4o Mini at about 260 seconds. Moreover, REG-TSC
consistently exhibits small variances in travel time distribu-
tions across all scenes. These results show that REG-TSC
outperforms other methods in the rationality and reliability
of decision-making when handling emergency vehicles.
D. Generalization Comparison
To evaluate the generalization capability of different meth-
ods, we train all models on the Hangzhou2 dataset except
general LLMs and test them on the Yizhuang Extreme, as
shown in Fig. 6. Compared with RL-based methods, the
LLM-based methods show stronger generalization, making
reasonable decisions even in unseen environments. REG-TSC
achieves the lowest AWT of 341.45 seconds and the shortest
AQL of 12.86, significantly outperforming all baselines. In
contrast, RL-based methods exhibit much higher AWT above
775 seconds and longer AQL exceeding 50. The results
confirm that REG-TSC is able to generalize well to unfamiliar
and extreme traffic conditions with outstanding performance.
Fig. 6. Performance Under Extreme Traffic Conditions (M= 0).
VI. CONCLUSION
TSC plays a pivotal role in optimizing urban traffic. This pa-
per proposes REG-TSC for generalizable signal optimization
with emergency response. The emergency-aware reasoning
framework enables REG-TSC to make reasonable and reliable
decisions in emergency scenarios. R3adaptively trains REG-
TSC across heterogeneous intersections, enhancing general-
ization capability. Extensive experiments in 6 traffic scenes
show that REG-TSC reduces ATT by 42.00% and AWTE by
83.16% compared with other SOTA methods. In the future, we
will consider mixed autonomous driving scenarios and study
vehicle–infrastructure cooperation for LLM-based TSC.
REFERENCES
[1] Y . Jiang, S. Guo, H. Chen, X. Mao, Y . Lin, and H. Wan, “Proactive-
xlight: Proactive traffic signal control with pluggable and reliable traffic
prediction,”IEEE Transactions on Mobile Computing, pp. 1–16, 2025.
[2] M. Movahedi and J. Choi, “The crossroads of LLM and traffic control:
A study on large language models in adaptive traffic signal control,”
IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 2,
pp. 1701–1716, 2025.[3] Q. Ji, X. Wen, J. Jin, Y . Zhu, and Y . Lv, “Urban traffic control meets
decision recommendation system: A survey and perspective,”IEEE/CAA
Journal of Automatica Sinica, vol. 11, no. 10, pp. 2043–2058, 2024.
[4] Z. Han, M. Xiao, H. Tan, and G. Gao, “Two-layer traffic signal optimiza-
tion: A edge-assisted pressure balance approach based on cooperative
game,” in2021 IEEE 27th International Conference on Parallel and
Distributed Systems (ICPADS), 2021, pp. 82–89.
[5] M. Cao, V . O. K. Li, and Q. Shuai, “A gain with no pain: Exploring
intelligent traffic signal control for emergency vehicles,”IEEE Trans.
Intell. Transp. Syst., vol. 23, no. 10, pp. 17 899–17 909, 2022.
[6] H. Su, Y . D. Zhong, B. Dey, and A. Chakraborty, “Emvlight: A
decentralized reinforcement learning framework for efficient passage
of emergency vehicles,” inProceedings of the AAAI Conference on
Artificial Intelligence, vol. 36, no. 4, 2022, pp. 4593–4601.
[7] L. Ding, D. Zhao, Z. Wang, G. Wang, C. Tan, L. Fan, and H. Ma, “Learn-
ing to help emergency vehicles arrive faster: A cooperative vehicle-road
scheduling approach,”IEEE Transactions on Mobile Computing, vol. 22,
no. 10, pp. 5949–5962, 2023.
[8] M. Wang, Y . Chen, A. Pang, Y . Cai, C. S. Chen, Y . Kan, and M.-O.
Pun, “Vlmlight: Traffic signal control via vision-language meta-control
and dual-branch reasoning,”arXiv preprint arXiv:2505.19486, 2025.
[9] Z. Wang, K. Yang, L. Li, Y . Lu, and Y . Tao, “Traffic signal priority
control based on shared experience multi-agent deep reinforcement
learning,”IET Intelligent Transport Systems, vol. 17, no. 7, 2023.
[10] A. Pang, M. Wang, M. Pun, and et al., “iLLM-TSC: Integration rein-
forcement learning and large language model for traffic signal control
policy improvement,”arXiv preprint arXiv:2407.06025, 2024.
[11] M. Wang, A. Pang, Y . Kan, M.-O. Pun, C. S. Chen, and B. Huang,
“LLM-assisted light: Leveraging large language model capabilities for
human-mimetic traffic signal control in complex urban environments,”
arXiv preprint arXiv:2403.08337, 2024.
[12] H. Joren, J. Zhang, C.-S. Ferng, D.-C. Juan, A. Taly, and C. Rashtchian,
“Sufficient context: A new lens on retrieval augmented generation
systems,” inThe Thirteenth International Conference on Learning
Representations (ICLR), 2025.
[13] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu, “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
ACM Trans. Inf. Syst., vol. 43, no. 2, Jan. 2025.
[14] M. L. Bernardi, M. Cimitile, and R. Pecori, “Automatic job safety
report generation using RAG-based LLMs,” in2024 International Joint
Conference on Neural Networks (IJCNN), 2024, pp. 1–8.
[15] Z. Zhu, G. Qi, G. Shang, Q. He, W. Zhang, N. Li, Y . Chen, L. Hu,
W. Zhang, and F. Dang, “Enhancing large language models with
knowledge graphs for robust question answering,” in2024 IEEE 30th
International Conference on Parallel and Distributed Systems (ICPADS),
2024, pp. 262–269.
[16] S. Wandelt, C. Zheng, S. Wang, Y . Liu, and X. Sun, “Large language
models for intelligent transportation: A review of the state of the art and
challenges,”Applied Sciences, vol. 14, no. 17, p. 7455, 2024.
[17] S. Lai, Z. Xu, W. Zhang, H. Liu, and H. Xiong, “LLMLight: Large
language models as traffic signal control agents,” inProceedings of
the 31st ACM SIGKDD Conference on Knowledge Discovery and Data
Mining, 2025, pp. 2335–2346.
[18] A. Oroojlooy, M. Nazari, D. Hajinezhad, and J. Silva, “Attendlight:
Universal attention-based reinforcement learning model for traffic signal
control,”Advances in Neural Information Processing Systems, vol. 33,
pp. 4079–4090, 2020.
[19] J. Zeng, C. Yu, X. Yang, W. Ao, Q. Hao, J. Yuan, Y . Li, Y . Wang,
and H. Yang, “Citylight: A universal model for coordinated traffic
signal control in city-scale heterogeneous intersections,”arXiv preprint
arXiv:2406.02126, 2024.
[20] H. Jiang, Z. Li, Z. Li, and et al., “A general scenario-agnostic reinforce-
ment learning for traffic signal control,”IEEE Transactions on Intelligent
Transportation Systems, vol. 25, no. 9, pp. 11 330–11 344, 2024.
[21] H. Zhu, J. Feng, F. Sun, K. Tang, D. Zang, and Q. Kang, “Sharing control
knowledge among heterogeneous intersections: A distributed arterial
traffic signal coordination method using multi-agent reinforcement learn-
ing,”IEEE Transactions on Intelligent Transportation Systems, vol. 26,
no. 2, pp. 2760–2776, 2025.
[22] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y .-P. Fl ¨otter ¨od,
R. Hilbrich, L. L ¨ucken, J. Rummel, P. Wagner, and E. Wiessner, “Micro-
scopic traffic simulation using sumo,” in21st International Conference
on Intelligent Transportation Systems, 2018, pp. 2575–2582.

APPENDIXA
PROMPTTEMPLATE ANDRESPONSESAMPLE OFREG-TSC
A. Prompt Template for REG-TSC
TABLE IV
PROMPTTEMPLATE FORREG-TSC
[Role]
You are a traffic signal control agent with emergency response.
[Objective]
Based on the real-time traffic representation, emergency vehicle state, critical guidance for emergency scenarios and commonsense knowledge, determine the
most effective next traffic signal phase that allows emergency vehicles to pass through as quickly as possible and improve the overall traffic condition.
[Real-Time Traffic Representation and Emergency Vehicle State]
Intersection Topology: There are 4 bidirectional roads connected to this intersection (ID: road#1, road#2, road#3, road#4), with 2, 2, 2, 2 incoming lanes
respectively. A total of 16 traffic movements are managed by 4 signal phases in this intersection.
Action Space:(Traffic movements allowed by each phase, upstream lane→downstream lane):
Phase 1: road#1 2→-road#3 2; road#1 2→-road#4 2; road#3 2→-road#1 2; road#3 2→-road#2 2
Phase 2: road#1 1→-road#2 1; road#1 1→-road#1 1; road#3 1→-road#4 1; road#3 1→-road#3 1
Phase 3: road#2 2→-road#4 2; road#2 2→-road#1 2; road#4 2→-road#2 2; road#4 2→-road#3 2
Phase 4: road#2 1→-road#3 1; road#2 1→-road#2 1; road#4 1→-road#1 1; road#4 1→-road#4 1
Queuing and Approaching Vehicles: The number of queuing vehicles (QV) is counted on upstream lanes controlled by each signal phase. Each lane is
divided into three equal-length segments from the lane start to the stop line, representing the far, middle, and near sections to the stop line. The count of
approaching vehicles (A V) in each segment is recorded accordingly as far/mid/near.
Phase 1 Phase 2 Phase 3 Phase 4
road#1 2: QV=4; A V=0/3/0 road#1 1: QV=2; A V=1/1/0 road#2 2: QV=3; A V=1/0/2 road#2 1: QV=5; A V=2/1/1
road#3 2: QV=1; A V=3/0/2 road#3 1: QV=1; A V=2/0/1 road#4 2: QV=2; A V=1/1/3 road#4 1: QV=2; A V=1/2/0
Total: QV=5; A V=3/3/2 Total: QV=3; A V=3/1/1 Total: QV=5; A V=2/1/5 Total: QV=7; A V=3/3/1
Emergency Vehicle State:
Emergency Vehicle ID: Ambulance 1
Planned Route: road#72→road#64→road#2→road#4→road#9→road#32
Current Position: road#2 1,276.8mto stop line
Speed:17.4m/s
[Critical Guidance for Emergency Scenarios]
Current Possible Situation: An emergency vehicle is approaching the intersection, but its lane is still occupied by queuing vehicles.
Recommended Action: Promptly select the signal phase for the lane with the emergency vehicle.
Intended Effect: Clear the queuing vehicles in the lane with the emergency vehicle for it rapid passage.
[Commonsense Knowledge]
1. EMERGENCY VEHICLE PRIORITY: Emergency vehicles have the highest priority. Phase selection should PRIMARILY aim to minimize their waiting
time and allow them to pass through the intersection as quickly as possible.
2. MAXIMIZE THROUGHPUT: Choose phases that reduce overall traffic delay and congestion across all lanes.
3. EARLY QUEUE URGENCY: Traffic congestion at intersections is mostly caused by vehicles queued NEAR the stop line. PRIORITIZE lanes with long
queues there, while vehicles in distant segments can wait.
4. DOWNSTREAM BLOCKAGE CAUTION: Avoid activating any lane that would push the downstream link near capacity, as this risks spill back and
network-wide delays.
5. WAITING TIME FAIRNESS: Lanes that have waited excessively must be served once downstream allows—they cannot be skipped indefinitely.
6. LANE RULES: Vehicles are permitted to pass one at a time per lane. All vehicles, INCLUDING emergency vehicles, must follow the queuing order,
meaning a vehicle cannot move until those ahead of it have departed.
[Task Description]Carefully analyze the given information, think step by step, and make well-reasoned decisions.
1.Analysis: Interpret and analyze the current traffic conditions and the emergency vehicle state.
2.Prediction and Evaluation: Evaluate and compare traffic signal phases by predicting emergency vehicle arrival time at the intersection and future queue
lengths. Integrate critical guidance and commonsense knowledge to ensure the emergency vehicle can pass through intersections without delay.
3.Decision-Making: Select an appropriate traffic signal phase that enables emergency vehicles to pass through the intersection as quickly as possible and
reduces overall traffic congestion. Provide the appropriate phase selection and an explanation.
[Output Format]
Your response must strictly follow the format below.
<response>
<traffic analysis>INSERT ANALYSIS HERE</traffic analysis>
<evaluation and explanation>INSERT EV ALUATION AND EXPLANATION HERE</evaluation and explanation>
<signal>INSERT PHASE NUMBER HERE</signal>
</response>
B. Response Sample of REG-TSC
<response> <traffic analysis>The emergency vehicle Ambulance 1 is on road#2 1, 276.8 m from the stop line, moving at 17.4 m/s. This lane has 5
queuing vehicles and approaching vehicles distributed as 2/1/1 (far/mid/near). Phase 4 controls this lane, but the lane is currently blocked by queued vehicles.
Other phases either do not serve the emergency vehicle’s lane or have lower total queue urgency.</traffic analysis> <evaluation and explanation>At
current speed, Ambulance 1 will reach the stop line in 15.9 s. Only Phase 4 will allow clearing the queue ahead. Prioritizing this phase minimizes waiting
time for the ambulance. Other phases would delay the emergency vehicle. Phase 4 also has the highest total QV=7. This aligns with emergency vehicle
priority, early queue urgency, and overall throughput considerations.</evaluation and explanation> <signal>4</signal> </response>