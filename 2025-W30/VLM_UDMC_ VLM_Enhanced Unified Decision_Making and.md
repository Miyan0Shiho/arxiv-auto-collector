# VLM-UDMC: VLM-Enhanced Unified Decision-Making and Motion Control for Urban Autonomous Driving

**Authors**: Haichao Liu, Haoren Guo, Pei Liu, Benshan Ma, Yuxiang Zhang, Jun Ma, Tong Heng Lee

**Published**: 2025-07-21 06:06:27

**PDF URL**: [http://arxiv.org/pdf/2507.15266v1](http://arxiv.org/pdf/2507.15266v1)

## Abstract
Scene understanding and risk-aware attentions are crucial for human drivers
to make safe and effective driving decisions. To imitate this cognitive ability
in urban autonomous driving while ensuring the transparency and
interpretability, we propose a vision-language model (VLM)-enhanced unified
decision-making and motion control framework, named VLM-UDMC. This framework
incorporates scene reasoning and risk-aware insights into an upper-level slow
system, which dynamically reconfigures the optimal motion planning for the
downstream fast system. The reconfiguration is based on real-time environmental
changes, which are encoded through context-aware potential functions. More
specifically, the upper-level slow system employs a two-step reasoning policy
with Retrieval-Augmented Generation (RAG), leveraging foundation models to
process multimodal inputs and retrieve contextual knowledge, thereby generating
risk-aware insights. Meanwhile, a lightweight multi-kernel decomposed LSTM
provides real-time trajectory predictions for heterogeneous traffic
participants by extracting smoother trend representations for short-horizon
trajectory prediction. The effectiveness of the proposed VLM-UDMC framework is
verified via both simulations and real-world experiments with a full-size
autonomous vehicle. It is demonstrated that the presented VLM-UDMC effectively
leverages scene understanding and attention decomposition for rational driving
decisions, thus improving the overall urban driving performance. Our
open-source project is available at https://github.com/henryhcliu/vlmudmc.git.

## Full Text


<!-- PDF content starts -->

1
VLM-UDMC: VLM-Enhanced Unified
Decision-Making and Motion Control
for Urban Autonomous Driving
Haichao Liu, Haoren Guo, Pei Liu, Benshan Ma, Yuxiang Zhang, Jun Ma, and Tong Heng Lee
Abstract —Scene understanding and risk-aware attentions are
crucial for human drivers to make safe and effective driving
decisions. To imitate this cognitive ability in urban autonomous
driving while ensuring the transparency and interpretability,
we propose a vision-language model (VLM)-enhanced unified
decision-making and motion control framework, named VLM-
UDMC. This framework incorporates scene reasoning and risk-
aware insights into an upper-level slow system, which dynam-
ically reconfigures the optimal motion planning for the down-
stream fast system. The reconfiguration is based on real-time en-
vironmental changes, which are encoded through context-aware
potential functions. More specifically, the upper-level slow system
employs a two-step reasoning policy with Retrieval-Augmented
Generation (RAG), leveraging foundation models to process mul-
timodal inputs and retrieve contextual knowledge, thereby gener-
ating risk-aware insights. Meanwhile, a lightweight multi-kernel
decomposed LSTM provides real-time trajectory predictions
for heterogeneous traffic participants by extracting smoother
trend representations for short-horizon trajectory prediction. The
effectiveness of the proposed VLM-UDMC framework is verified
via both simulations and real-world experiments with a full-size
autonomous vehicle. It is demonstrated that the presented VLM-
UDMC effectively leverages scene understanding and attention
decomposition for rational driving decisions, thus improving the
overall urban driving performance. Our open-source project is
available at https://github.com/henryhcliu/vlmudmc.git.
Index Terms —Autonomous driving, vision-language model,
large language model, motion control, multi-vehicle interactions.
I. I NTRODUCTION
Urban autonomous driving has emerged as a critical tech-
nology to address the escalating challenges of traffic conges-
tion, safety risks, and operational inefficiency in populated
cities [1]. To tackle dynamic traffic scenarios involving diverse
Haichao Liu is with the Robotics and Autonomous Systems Thrust, The
Hong Kong University of Science and Technology (Guangzhou), Guangzhou
511453, China, and also with the Department of Electrical and Com-
puter Engineering, National University of Singapore, Singapore (e-mail:
haichao.liu@u.nus.edu).
Haoren Guo, Yuxiang Zhang, and Tong Heng Lee are with the Depart-
ment of Electrical and Computer Engineering, National University of Sin-
gapore, Singapore (e-mail: haorenguo 06@u.nus.edu; zhangyx@nus.edu.sg;
eleleeth@nus.edu.sg).
Pei Liu and Benshan Ma are with the Robotics and Autonomous
Systems Thrust, The Hong Kong University of Science and Technology
(Guangzhou), Guangzhou 511453, China (e-mail: pliu061@connect.hkust-
gz.edu.cn, bma224@connect.hkust-gz.edu.cn)
Jun Ma is with the Robotics and Autonomous Systems Thrust, The
Hong Kong University of Science and Technology (Guangzhou), Guangzhou
511453, China, and also with the Division of Emerging Interdisciplinary
Areas, The Hong Kong University of Science and Technology, Hong Kong
SAR, China (e-mail: jun.ma@ust.hk).
Fig. 1. The ego vehicle is turning left at an unsignalized T-intersection,
while a little girl is crossing the road. The VLM deployed on the ego vehicle
understands and inferences from the environment based on the real-time
images captured by the onboard cameras, as shown from the four perspectives.
participants and long-tail events like the situation shown in
Fig. 1, future autonomous systems necessitate going beyond
reactive control [2], [3], which requires advanced predictive
and reasoning capabilities grounded in contextual understand-
ing and explainable decision-making [4], [5]. Essentially, safe
and efficient urban navigation requires machines to replicate
human drivers’ ability to integrate scene understanding, risk
assessment, and adaptive decision-making, while ensuring
transparency and interpretability for real-world safety-critical
deployment [6]. Recent progress in large language models
(LLMs) and vision-language models (VLMs) has opened new
avenues for designing such cognitive-driven driving systems.
Unlike traditional modular or end-to-end frameworks that
often lack interpretability and generalization, our approach
leverages the representational power of the foundation models
to enable complex scene understanding, adaptive planning,
and transparent reasoning, key attributes for robust urban
autonomy.
Existing approaches to autonomous driving can be broadly
categorized into hierarchical frameworks [7] and end-to-end
architectures [8]. Hierarchical methods decompose the driving
task into modular layers, such as perception, prediction, plan-
ning, and control [9]. Early hierarchical systems rely heavily
on rule-based logic, encoding explicit traffic regulations and
heuristic strategies into predefined decision trees [10], [11]
or finite state machines (FSM) [12]. Generally, lateral and
longitudinal motion commands are generated by iLQR and
PID, respectively [13]. While interpretable, these systems
struggle with the unbounded variability of urban scenes, wherearXiv:2507.15266v1  [cs.RO]  21 Jul 2025

2
PerceptionDifferent Utilities of VLMs/VLAs for AD
Perception
Perception
PerceptionVLA
VLM
VLM
VLMAction of the ego vehicle (acc, steer...)
Waypoints
Parameters
mat(Q), mat(R)...
AttentionsController acc, steer...
MPC acc, steer...
MPC acc, steer...
Risk Field1
2
3
4
Cameras, LiDAR...
Fig. 2. The mainstream VLM/VLA utilities for autonomous driving. From top
to bottom, the models are leveraged to generate the instant control commands,
the predicted future waypoints to be followed by the low-level controller, the
parameters to be integrated into MPC, and the risk attentions to be integrated
into MPC, respectively.
unstructured inputs (e.g., unexpected pedestrian movements,
ambiguous road signs) exceed preprogrammed rules. For more
flexible and optimal driving, optimization-based methods,
such as Model Predictive Control (MPC) [14], [15], advance
this paradigm by formulating driving as an optimal control
problem (OCP) over a receding horizon, considering vehicle
dynamics, safety constraints, and so on. However, traditional
MPC-based frameworks employ static weight parameters for
constant objective functions, such as predefined-trajectory
tracking, safety, comfort, and efficiency [16], which hinders
adaptability to changing conditions, like traffic jams in strange
driving scenarios.
On the other hand, end-to-end frameworks, such as
UniAD [17], leverage Transformer architectures to model
holistic driving policies, encoding sensor inputs directly into
control commands or waypoints via self-attention and cross-
attention mechanisms that capture implicit long-range interac-
tions between agents [8], [18], [19]. In addition, Diffusion-
based methods, such as DiffusionDrive [20], enhance gener-
ative capabilities by offering multiple trajectory candidates
for various future conditions involving surrounding traffic
participants. With the emergence and development of LLMs, to
synchronize driver interactions with the driving system, vision-
language-action (VLA) approaches, like OpenDriveVLA [21],
are introduced to generate specific actions for the ego vehicle
based on language input and real-time perception data, as
illustrated in the first row of Fig. 2. While these models
exhibit strong performance in common scenarios during open-
loop testing, they frequently experience diminished reliabil-
ity in long-tail cases. Moreover, safety-critical decisions and
actions require transparent reasoning rather than black-box
inference [22], which is compromised by the direct action
generation paradigm. Consequently, as described in the last
three rows of Fig. 2, hybrid approaches that combine the
semantic reasoning ability of VLMs with the structured op-
timization of model-based control are gaining more traction,
aiming to balance flexibility with accountability [23]–[26].
However, existing hybrid approaches limit the potential of
VLMs in semantic reasoning by only adaptively adjusting
a few parameters within low-level driving modules, suchas the weight matrices in the cost functions. Such minimal
modifications are unlikely to bring about fundamental changes
to the system’s behavior, as the underlying optimization struc-
ture remains unchanged. Therefore, developing methods that
enable adaptive reformulation of the optimization problem
represents a promising research direction.
Another persistent challenge in existing hybrid methods
is the limited integration of real-time motion forecasting
and adaptive risk modeling [27], [28]. Many VLM-enhanced
controllers, as shown in the third row of Fig. 2, lack dynamic
awareness of surrounding agents’ future trajectories [29],
leading to suboptimal and conservative decisions for extreme
driving safety in interactive scenarios, such as merging with
high-speed mixed traffic or collaborative vehicles [30]–[32].
Additionally, model-based components, such as controllers
with receding horizon manner, often assume that obstacles
are static during the time window, failing to update with
evolving scene contexts [33], [34]. To fill the research gap,
in the time-series forecasting studies, both Long Short-Term
Memory (LSTM) and Transformer-based models [27], [28]
have demonstrated strong capabilities in capturing temporal
patterns and enhancing prediction accuracy. However, LSTM
struggles to capture non-local or cross-period patterns such
as periodicity and abrupt changes, while Transformer-based
models often suffer from under-representation when trained
on small datasets. DLinear [35] has been proposed as a
lightweight and efficient alternative, particularly suitable for
small datasets. By explicitly decomposing time series into
trend and seasonal components, DLinear achieves improved
interpretability and forecasting performance. Nevertheless,
DLinear exhibits excessive sensitivity in trend extraction when
applied to relatively short sequences, often resulting in noisy
or unsmooth trend estimations. This limitation hampers its
performance in short-horizon prediction tasks in autonomous
driving, where the underlying trend may be less pronounced.
Therefore, incorporating a multi-kernel moving average mech-
anism to capture smoother and more robust trajectory trend
representations is a worthwhile endeavor.
Building upon our previous research [16], we propose a
VLM-enhanced unified decision-making and motion control
framework (VLM-UDMC) for urban autonomous driving,
which is designed to effectively emulate the scene understand-
ing and risk-aware attention mechanisms of human drivers.
It consists of an upper slow decision-making system that
incorporates scene reasoning and risk-aware insights, and a
downstream fast motion control system that utilizes the high-
level decisions to adaptively reconfigure the OCP. To sup-
port this framework, we incorporate the Retrieval-Augmented
Generation (RAG) technique alongside a two-step reasoning
strategy. This integration enables foundation models to process
multimodal inputs and retrieve relevant contextual knowl-
edge, leading to the generation of interpretable, risk-aware
insights. These insights are subsequently utilized to construct
the potential functions, which characterize traffic factors, and
are embedded into the objective function of the OCP to
guide motion planning dynamically. Moreover, a compact
yet powerful multi-scale decomposed LSTM enables efficient
and accurate trajectory predictions for heterogeneous traffic

3
participants, supporting proactive anticipation of interactive
behaviors and behavior refinement over future time steps.
Importantly, an in-context learning mechanism continuously
updates the VLM using newly encountered real-world sce-
narios, and this improves the robustness to long-tail cases
while enhancing explainability by grounding decisions in
verifiable perceptual and linguistic cues. Consequently, the key
contributions of this work are summarized as follows:
•We propose VLM-UDMC for urban autonomous driving
that integrates scene reasoning for risk-aware decision-
making in the upper slow system, which is subsequently
used to adaptively reconfigure the optimal motion plan-
ning in the downstream fast system.
•For enhanced scene understanding with multimodal
prompts from VLMs, we develop a two-step reasoning
mechanism combined with RAG-based in-context learn-
ing for attention-aware driving, while also addressing the
long-tail problem through adaptation to newly encoun-
tered scenarios.
•By integrating a lightweight multi-kernel LSTM for real-
time trajectory forecasting with attention-aware objective
updating, the OCP is adaptively informed by heteroge-
neous time-scale inputs, enabling seamless and real-time
decision-making and motion control under the semantic
guidance of VLMs.
•The simulation and real-world experiments demonstrate
the effectiveness of the proposed VLM-UDMC. A com-
prehensive comparison with baseline models highlights
the superiority of our scheme, while an ablation study
assesses the necessity of each functionality.
The rest of the paper is organized as follows. Section
II presents the preliminaries of the presented urban driving
algorithm. Section III illustrates the construction of the OCP,
especially the potential functions for invoking by VLMs. In
Section IV , the organization of the multimodal prompt with
in-context learning for VLMs is elaborated. Next, Section
V details the comparisons and ablations in both simulation
and real-world experiments. Finally, Section VI concludes this
work with a discussion.
II. P RELIMINARIES
A. Notation
In this paper, vectors and matrices are represented by bold-
face letters, with lowercase letters for vectors and uppercase
letters for matrices. Specifically, x∈Rndenotes a vector and
X∈Rn×mdenotes a matrix. For a vector x, the segment
from its i-th to j-th element is denoted as x[i:j]. In the
case of matrices, the i-th row is represented as [X]i,·and the
j-th column as [X]·,j. The symbols N,B, andR+refer to
the sets of non-negative integers, boolean values, and positive
real numbers, respectively. For ease of representation, a matrix
composed of vectors [s1,s2, . . . ,sn]⊤is simplified as [si],
where i∈1,2, . . . , n . Finally, the notation [x]+is used to
represent the function max{x,0}.
For scenario-specific notations, [·]i
τis employed to specify
the state vector of the i-th entity at time step τ. It is crucial
to recognize that each entity has its own distinct state vector.For example, a vehicle’s state vector includes attributes such as
position, velocity, and heading angle. Similarly, the state vector
for a road lane center consists of position and orientation,
whereas a traffic light object’s state vector is defined by the
stop line’s position. When the state vector is expressed in the
autonomous vehicle frame, we append a superscript [·]evto
the notation.
B. Optimal Control Problem for Urban Driving
Optimization-based methods are widely used in autonomous
driving and robot navigation due to their explainability and
consistent performance across diverse scenarios. Within these
methods, the MPC balances computational tractability with
adaptability to dynamic environments, making it ideal for
urban scenarios where interactions with heterogeneous traffic
participants and changing road conditions demand proactive
and forward-looking planning. The OCP within this frame-
work is formulated to minimize a cost function that encap-
sulates reference trajectory tracking accuracy, control effort,
passenger comfort, and safety regulations. Therefore, the cost
function Jis structured to prioritize multiple competing ob-
jectives over a prediction horizon of length N, defined as:
J(X,U,O) =NX
τ=1∥xref,τ−xτ∥2
Q+NX
τ=1∥uτ∥2
R
+NX
τ=2∥uτ−uτ−1∥2
Rd+F(O,X).(1)
Here, X={x1,x2, . . . ,xN} ∈Rm×Nrepresents the se-
quence of vehicle states over the horizon, with each xτ∈Rm
encoding the state of the autonomous vehicle at time step τ.
The control input sequence U={u1,u2, . . . ,uN} ∈Rq×N
consists of the control input ui∈Rqat each time step τ. The
reference trajectory xref,τis dynamically derived according
to the global path and the current vehicle state, ensuring
alignment with high-level route plans.
Subsequently, the first term in Jpenalizes deviations from
the reference trajectory using a positive semi-definite diagonal
weighting matrix Q∈Rm×m, emphasizing accurate path fol-
lowing. The second term discourages excessive control effort
through R∈Rq×q, promoting energy-efficient maneuvers.
The third term, weighted by Rd∈Rq×q, minimizes abrupt
changes in control inputs to enhance passenger comfort and
reduce wear on vehicle actuators. The final term F(O,X)
integrates a risk field that encodes safety-critical interactions
with surrounding objects (e.g., other vehicles, pedestrians,
curbs) in O={o1, o2,···, ok}by converting their states into
potential forces, dynamically adjusting the cost landscape to
avoid collisions.
In addition, respecting the physical dynamics and opera-
tional limits of the autonomous vehicle, the complete OCP is
posed as a constrained optimization problem:
min
{xτ,uτ}J(X,U,O)
subject to xτ+1=f(xτ,uτ),∀τ∈ {1,2, . . . , N }
−umin⪯uτ⪯umax,∀τ∈ {1,2, . . . , N }
−xmin⪯xτ⪯xmax,∀τ∈ {1,2, . . . , N },
(2)

4
Supplementary
DescriptionTask PromptVLM
RAG
Memory......Potential    
 Functions    Vehicles
Pedestrians
Lane StateTraffic Light
Input
Patch GenerationMulti-Kernel
Decomposed
LSTM for
Surrounding
VehiclesSurrounding
Vehicles
Non-Crossable
Lane MarkingCrossable
Lane Marking
Vulnerable Road
UsersPotential Function Items
Trajectory
Tracking
Riding ComfortEnergy Saving
Safety and
RulesSlow System
Fast System
Task Assignment
Spline Trajectory GenerationModel Predictive Control
Multiple ShootingOCP
IPOPT
Traffic
ControlTraffic
ParticipantsPlatformTransportation
Infrustructure
Ego Vehicle
Perception
Objectives
Fig. 3. Framework of the proposed VLM-enhanced risk-aware urban autonomous driving system is designed to address the complexities of urban driving. This
framework comprises four sequential steps and incorporates an asynchronous mechanism to reconcile the latency inherent in scene understanding inference
with the stringent demands of real-time control.
where the first constraint enforces the nonlinear vehicle dy-
namics model, which maps the current state xτand control
input uτto the next state xτ+1(detailed in Section II-C).
The subsequent constraints bound the control inputs and
states within feasible ranges dictated by the vehicle’s phys-
ical limitations (e.g., maximum steering angle, acceleration
capacity), ensuring safe and realizable maneuvers. By solving
this problem iteratively at each time step with the latest sensor
observations and reasoning decisions of the VLM (elaborated
in Section IV-B), the framework adapts to evolving scenes
while maintaining real-time performance, forming the basis
for risk-aware control in urban environments.
C. Dynamics Model for Autonomous Vehicle
In this work, we adopt a nonlinear vehicle dynamics model
from [36], as follows:
xτ+1=
px,τ+Ts(vx,τcosφτ−vy,τsinφτ)
py,τ+Ts(vy,τcosφτ+vx,τsinφτ)
φτ+Tsωτ
vx,τ+Tsaτ
mavx,τvy,τ+TsLkωτ−Tskfδτvx,τ−Tsmav2
x,τωτ
mavx,τ−Ts(kf+kr)
Izvx,τωτ+TsLkvy,τ−Tslfkfδτvx,τ
Izvx,τ−Ts(l2
fkf+l2rkr)
,
(3)
which is chosen for its robust numerical stability and high-
fidelity representation of complex motions. In (3), the state
vector x= [px, py, φ, v x, vy, ω]⊤∈R6captures six degrees
of freedom: longitudinal and lateral positions, heading angle,
longitudinal and lateral velocities, and yaw rate, respectively.The control input vector u= [a, δ]⊤∈R2consists of longi-
tudinal acceleration aand front-wheel steering angle δ, which
act as the primary actuation commands. Besides, the discrete-
time dynamics (3), also expressed by xτ+1=f(xτ,uτ), is
derived using the backward Euler method with a fixed time
stepTs= 0.05s. Here, madenotes the vehicle mass, lfand
lrare the distances from the center of mass to the front and
rear axles, respectively, while kfandkrrepresent the front
and rear tire cornering stiffness, a key parameter for modeling
lateral forces during steering. Izis the polar moment of inertia
about the vertical axis, describing the vehicle’s resistance to
yaw acceleration. The term Lk=lfkf−lrkrsimplifies the
formulation by capturing the differential effect of front and
rear tire forces on yaw dynamics.
III. S PATIAL -TEMPORAL MOTION PLANNING AND
CONTROL FOR URBAN DRIVING
This section presents and analyzes the mathematical mod-
eling of traffic elements, including surrounding vehicles, vul-
nerable road users (VRUs), and components related to traf-
fic rules, for invocation by VLMs, as illustrated in Fig. 3.
Additionally, to enable spatial-temporal motion planning, we
introduce a compact yet powerful trajectory prediction model
based on LSTM networks and analyze the mechanism for
capturing detailed historical data.
A. Traffic Items Description and Modeling
In urban autonomous driving, effective modeling of dy-
namic and static traffic elements is crucial for safe yet efficient

5
trajectory planning. We employ potential field methods to
represent four key categories of traffic items: nearby vehicles
with collision risks, pedestrians as the primary VRUs, adjacent
road lane markings for behavioral guidance, and recognized
red traffic lights. These potential functions are dynamically
activated based on scene understanding and inference from
VLMs, ensuring that only relevant environmental factors with
their specific ID i, j, k, q ∈ S τinfluence the OCP at each
time step. For example, when the VLM identifies that certain
surrounding vehicles pose no collision risk (e.g., traveling
in opposite directions on separate lanes), their corresponding
potential functions are excluded from the OCP formulation,
enhancing computational efficiency and decision relevance.
The potential field F(penv
τ,xτ)at time step τintegrated
into the OCP is a composite of multiple specialized potential
functions, each designed to model specific traffic interactions:
F(penv
τ,xτ) =X
i∈SτFi
NR,τ+X
j∈SτFj
CR,τ+X
k∈SτFk
V,τ
+X
q∈SτFq
VRU,τ+FTL,τ,(4)
where i,j,k, and qindex non-crossable lane markings,
crossable lane markings, surrounding risk vehicles, and VRUs,
respectively. The vector penv
τencapsulates the state informa-
tion of all relevant traffic entities, including their positions,
velocities, and orientations in the global Cartesian coordinate
frame.
1) Lane Marking Potential Functions: The lateral distance
between the autonomous vehicle and lane markings is a critical
factor in defining repulsive potential forces. We represent the
road centerline state vector as pi= [pi
x, pi
y, φi]⊤, where pi
x
andpi
ydenote the global coordinates of the centerline, and
φiis the tangential angle relative to the global x-axis. Con-
sequently, using the plane geometric relationship, the lateral
distance from the autonomous vehicle to a lane marking is
calculated as follows:
sR(x,pi) =σsin(φi) cos( φi)px−pi
x
py−pi
y
+wR
2,(5)
where σis the indicator function denoting the direction of a
specific lane marking: σ= 1 if the lane marking is on the
left side, and σ=−1if it is on the right. Additionally, wR
is a constant representing the lane width in the current traffic
system.
Crossable Lane Marking Items: For permissible lane
changes, such as overtaking, the crossable lane marking po-
tential FCRguides the autonomous vehicle to perform lane-
changing maneuvers and return to the lane center both before
and after the maneuver. This potential function also assists the
vehicle in maintaining its position in the middle of the road.
It is defined as follows:
Fj
CR=(
aCR(sR(x,pj)−bCR)2sRj< b CR
0 sRj≥bCR(6)
where aCRdetermines the force intensity, and bCRdefines the
effective range. The conditions in (6) imply that if the au-
tonomous vehicle remains within the lane without significantdeviation, no force is exerted by the lane markings, ensuring
smooth and stable driving control.
Non-Crossable Lane Marking Items: To enforce traffic
rules and prevent unsafe lane crossings (e.g., near curbs or
sidewalks), we design a repulsive potential function FNRthat
increases orderly as the autonomous vehicle approaches the
marking:
Fi
NR=

ms sRi≤0.1
aNR
sR(x,pi)bNR−es0.1< s Ri<1.5
0 sRi≥1.5(7)
where aNRandbNRcontrol the force intensity and effective
range, and smooth transition parameters esandmsensure
continuity by setting as
es=aNR
1.5bNR, m s=aNR
0.1bNR−es.
2) Surrounding Vehicle Potential Functions: We consider
the geometric approximation for vehicle modeling as ellipsoid
wrappings, designed to generate repulsive forces for collision
avoidance. For computational efficiency, we use an ellipsoid
approximation to represent the surrounding vehicles with
attention as follows:
Fk
V=2X
m=1aV(rarb)2
(r2
b(pmx−pkx)2+r2a(pmy−pky)2)bV, (8)
where raandrbare the ellipsoid major and minor semi-axes,
respectively.
3) Vulnerable Road User Potential Functions: Pedestrians
are modeled as a single circular potential centered at their head
position in bird’s-eye view:
Fq
VRU=aVRU
((px−pq
x)2+ (py−pq
y)2)bVRU, (9)
where aVRUandbVRUcontrol the repulsive force intensity for
theq-th pedestrian.
4) Traffic Light Potential Function: For red traffic lights,
we design a potential that guides the autonomous vehicle to
stop smoothly at the stop line while maintaining lateral control:
FTL=β 
aTL1
devx+aTL2
dev
y,l+aTL2
devy,r!
, (10)
where βis a binary indicator: β= 1 if the traffic light state
is red, and β= 0 vice versa. Besides, aTL1andaTL2are
longitudinal and lateral intensity parameters, and dev
x,dev
y,l,dev
y,r
are the autonomous vehicle’s distances to the stop line and lane
markings, respectively.
The above potential field-based modeling structure, dynam-
ically informed by VLM scene understanding, enables the
VLM-UDMC framework to prioritize safety-critical interac-
tions while maintaining computational efficiency in complex
urban environments. Once the potential functions are inte-
grated into the OCP, the framework also requires the predicted
states of selected traffic elements to facilitate a receding
horizon approach for urban driving.

6
AvgPoolld
AvgPoolld
AvgPoolldMulti-kernels Moving AverageOriginal
Trend
Remainder
Average
Summation
ForecastingLSTM
LSTMProjection Projection
Fig. 4. The architecture of the proposed multi-kernel prediction approach for
real-time motion prediction of the traffic participants, including surrounding
vehicles and VRUs. It leverages the DLinear mechanism to smooth sparse
trajectory data, improving the model’s sensitivity to trend changes.
B. Real-Time Motion Prediction of Traffic Participants
To accurately forecast the motion of heterogeneous traffic
participants in urban scenarios, we propose an efficient pre-
diction approach, as illustrated in Fig. 4. It is a lightweight yet
powerful model that integrates a multi-kernel moving average
mechanism to extract smoother trend representations for short-
horizon trajectory prediction. The subsequent LSTM module
further captures both global and local temporal patterns, en-
abling more efficient, accurate, and robust forecasting of real-
time motion prediction.
1) Multiscale Decomposition for Trajectory Data: To gen-
erate time-series trajectory predictions from input data with a
sparser time point distribution, we introduce a multi-kernels
moving average mechanism based on 1D average pooling
(AvgPool1d ) for multiscale analysis. As shown in Fig. 4,
multiple AvgPool1d operations with different kernel sizes
are applied to the original trajectory time series. This mul-
tiscale processing generates several smoothed representations
of the input.
Mathematically, let the original trajectory time series be
denoted as X= [x1, x2, . . . , x T]⊤, where Tis the number of
time points. For an AvgPool1d operation with kernel size k,
the output Xk
poolat position iis calculated as:
Xk
pool[i] =1
ki+k−1X
j=ixj. (11)
By using multiple kernel sizes (different kvalues), we obtain
multiple multiscale smoothed sequences. These sequences are
then combined (via the average operation as indicated in the
structure) to extract a more robust trend component.
The key advantage of this multiscale analysis is producing
a smoother trend from the relatively sparse trajectory data.
This smoother trend enhances the model’s sensitivity to trend
changes, resulting in the general movement patterns of traffic
participants being more accurately.
2) Trend and Remainder Component Separation: After
multiscale processing, we decompose the trajectory data into
two components: trend ( T) and remainder ( R). The trend
can be regarded as intra information, which is more general
and global, representing the overall direction and pattern of
the traffic participants’ movement. In contrast, the remaindercontains inter information, including more detailed and local
features such as sudden speed changes or small-scale maneu-
vers. The separation process can be described as:
X=T+R, (12)
where Tis obtained from the multiscale-smoothed and com-
bined sequence, and Ris the difference between the original
sequence and the trend sequence.
3) LSTM-Based Prediction for Components: Each of these
components (trend and remainder) is then fed into separate
LSTM networks. The LSTM is well-suited for processing
sequential data and can capture the temporal dependencies
within the trend and remainder components. After passing
through the LSTM layers, projection layers are used to trans-
form the LSTM outputs into the appropriate prediction space.
For the trend component, let the LSTM for trend be LSTM T
and the projection layer be ProjT. The prediction for the trend
component ˆTis:
ˆT=ProjT(LSTM T(T)). (13)
Similarly, for the remainder component, with LSTM Rand
projection layer ProjR, the prediction ˆRis:
ˆR=ProjR(LSTM R(R)). (14)
Finally, the overall motion prediction ˆXis obtained by sum-
ming the predictions of the two components:
ˆX=ˆT+ˆR. (15)
This two-component prediction approach, enabled by multi-
scale decomposition, not only improves the model’s ability to
capture the overall movement trend of traffic participants but
also preserves key local details. As a result, it significantly en-
hances the accuracy of real-time motion prediction in complex
urban traffic environments, providing reliable information for
the subsequent planning and control of autonomous vehicles.
IV. M ULTIMODAL SCENE UNDERSTANDING AND DRIVING
ATTENTION ADAPTATION
This section addresses the fast-slow mechanism designed
to reconcile the slow inference time of VLMs with the rapid
control demands of the OCP. It also explores multimodal
prompt context generation and RAG-based in-context learning
for the urban driving system.
A. Time-Scale Architecture of the Urban Driving System
In urban autonomous driving, the need to balance real-time
responsiveness and semantic reasoning efficiency gives rise to
a fast-slow architecture. Our specific design addresses two key
challenges: the disparity in computational demands between
low-level control and high-level scene understanding, and
the varying relevance of semantic updates based on evolving
driving context.

7
1) Motivation and Necessity: Semantic attention and MPC-
based motion planning are updated on different time scales
to ensure both improved system performance and efficiency.
The low-level MPC, as defined by (2), requires high-frequency
actuation (20 Hz in our system) to react to dynamic urban
environments, such as sudden pedestrian movements or vehicle
cut-ins. In contrast, VLM inference for scene understanding
incurs higher latency, with inference times ranging from 0.3
to 1.5 s for the foundation models like SmolVLM [37] and
Qwen3-4B [38].
The diversity of driving scenarios also motivates the de-
sign of a hierarchical framework with updates operating at
different time scales. For instance, on a highway with a stable
lane-keeping task, attention to road boundaries and forward
traffic suffices, and frequent VLM re-inference probably adds
no value. Conversely, considering dynamic scenarios (e.g.,
navigating a complex intersection), the objects for attention
must adapt quickly (e.g., crossing pedestrians, changing traffic
lights). Thus, the risk zone mechanism is proposed to identify
critical scenarios.
The proposed fast-slow architecture decouples as follows:
Fast sub-system with high-frequency MPC with 20 Hz for
control, using a static OCP formulation between semantic
updates. Slow sub-system with low-frequency foundation
model-driven attention adaptation with 0.3−1.5Hz, refining
the OCP based on scene semantics.
2) Mathematical Formulation of the Fast-Slow Mechanism:
Lettkdenote discrete time steps, with k∈N. The fast sub-
system, MPC, operates at a high frequency, solving the OCP
at each tk:
min
{xτ,uτ}k+N−1
τ=kJ(X,U,P(tm))
subject to xτ+1=f(xτ,uτ),
umin⪯uτ⪯umax,
xmin⪯xτ⪯xmax,(16)
where tm≤tkis the most recent time step when the
slow sub-system with a VLM updated the OCP items Pto
formulate F(penv
tk)in (4). In addition, to enhance the infer-
ence stability and regulate the final decision format, before
updating P, the slow sub-system retrieves relevant memory
itemsM={m1, m2, . . . , m M}, which contains similar scene
embeddings and the relevant responses, and generating a new
OCP formulation via foundation model inference:
P(tm) =FM(C(tm),Retrieve (M,C(tm))), (17)
where C(tm)is the multimodal context including the RGB
images from the cameras mounted on the autonomous vehicle
and the supplementary text describing the task and the re-
sponse format of the VLM. More detailed descriptions about
the prompt can be found in Section IV-B. Between updates
at a time step tk, the fast sub-system reuses the insights from
P(tm)to solve the OCP, ensuring continuity:
P(tk) =P(tm), tk∈ {t|tm< t < t m+1, t∈R+}.(18)
3) Adaptive OCP Evolution: The key insight is that the
OCP’s cost function and constraints evolve only when the slow
Multimodal Info.
RGB Image Ⅰ
Textual Clues T1, T2
AI Response A1, A2
Image Embedding ev
Texts Embedding etExpert Memory
P
Alive Memory
QContext Construction
C1(t) = (S1, H1, D1)
VLM Analysis as  A1
Dialogue (H1, A1)Memory Storage
M
🤗SmolVLM
Message Construction
H2 = R(H1, A1) ∈ C2(t)
LLM Analysis as  A2Dialogue (H2, A2)Fig. 5. Pipeline of the prompt generation and RAG process for the proposed
scene understanding and risk assessment mechanism.
sub-system provides a meaningful change in scene semantics.
For instance, if the VLM infers a new pedestrian entering the
autonomous vehicle’s vicinity and the right road lane changes
from crossable to non-crossable, it updates the potential field
termF(P,X)in the cost function by
F(P(tm+1),X) =F(P(tm),X)+F1
VRU−F2
CR+F2
NR,(19)
where F1
VRU is a new repulsive potential for the pedestrian
with ID = 1,F2
NRis the potential force from the non-crossable
right road lane, while its adversary F2
CRis removed from the
potential field. Nevertheless, if no new objects or semantic
changes are detected, P(tm+1) =P(tm), and the fast sub-
system continues with the existing OCP.
This architecture ensures that the autonomous vehicle re-
acts to immediate threats via high-frequency MPC while
adapting to long-term scene changes through low-frequency
VLM inference. By decoupling temporal scales, it balances
computational efficiency and semantic richness, enabling safe
and responsive urban driving.
B. Multimodal Context Generation for Scene Understanding
To enable foundation models to reason about complex urban
driving scenarios, as illustrated in Fig. 5, we design a two-step
structured multimodal context that integrates visual inputs,
textual information, and historical knowledge with a VLM and
an LLM. The corresponding prompts for the first and second
steps are denoted by C1andC2, respectively. The elements of
the prompts also have their superscripts to indicate the step
order. Inspired by [39], this context serves as the foundation
to take advantage of the VLM’s image understanding ability
and LLM’s specific reasoning ability for driving attention
adaptation, guiding the dynamic reformulation of the OCP.
Concretely, the visual information is analyzed by a VLM to
generate a structured textual description of the images. Then,
the second step is to get the final decision of attention from
an LLM. Note that the prompt to the LLM is generated by
reformulating the response of VLM and adding necessary
supplementary descriptions.
1) Context Composition Framework: The prompt Cis con-
structed as a tuple of three primary components:
C= (S,H,D), (20)

8
       RGB Perception        Risk-Aware Attention
q1
q4q2 q3
Fig. 6. Spatial characterization of risk zones enhances the inference of VLM
by enabling context-specific risk representations tailored to dynamic driving
conditions.
Prompt Structure of Stage 1
System Message
You are a driving assistant tasked with 
analyzing images from the ego -vehicle's 
cameras. Provide concise answers to 
questions based on the images, focusing on 
driving conditions, vehicle attention, VRU 
presence, and traffic rules.
Human  Message
Please analyze the following 
str(len(image_prompts )) images from the ego -
vehicle's str(len(images))  cameras and answer 
the questions:
1. Front Camera :
What is the driving condition (intersection, 
roundabout, straight urban road, highway, or 
other)? Is any vehicle present in front of the 
ego -vehicle? Is a vulnerable road user (VRU) 
present? Is a blocking to wait condition, such 
as red traffic light, visible?
2. Left Camera :
Does any vehicle need attention on the left? 
Can the ego -vehicle cross the left lane 
according to traffic rules?
3. Right Camera :
Does any vehicle need attention? Can the ego -
vehicle cross the right lane according to traffic 
rules?
4. Rear Camera :
Is there a collision risk with any rear vehicle?Prompt Structure of Stage 2
System Message
Your task is to provide urban driving 
suggestions to the ego -vehicle, analyzing the 
driving situation from multiple perspectives. 
After a concise chain of thought, generate 
your final response in the specified format :
```
{
'scene': <choose from intersection, 
roundabout, straight urban road, highway, or 
other>,
'risk zones': {'front': 0 or 1, 'left': 0 or 1, 'right': 
0 or 1, 'rear': 0 or 1},
'candidate lanes': {'left': 0 or 1, 'right': 0 or 1},
'block to wait': 0 or 1,
}
```
Human Message
Your task is to provide urban driving 
suggestions to the ego -vehicle. Your 
responses should strictly adhere to the 
format outlined in the System Message. 
Analyze the Q&A pairs from the front, left, 
right, rear camera perspectives to 
understand the scene : '' .join(answers_step1)
Ensure your final response is formatted 
correctly and considers all perspectives.
Fig. 7. Structure of the prompts to the foundation models. The second stage
prompt relies on the response of the first stage by retrieving and reformulating
processes.
where Srepresents the system message, defining the foun-
dation model’s task and response format, Hdenotes the
human message, containing real-time visual observations (for
VLM) and supplementary spatial information, Dencapsulates
analogical dialogue examples retrieved from memory storage
for in-context learning.
2) System Message Structure: The system message Sfor-
malizes the foundation model’s role and expected output
through a structured template: S= (T,F),where Tis
the task description, instructing the VLM to analyze camera
images and the LLM to identify risk zones and critical objects
(e.g., vehicles, VRUs, lane markings, traffic lights), Fdefines
the response format as a dictionary Aof attention objects:
A=
scene :{s1, , s2, . . . , s p},
zones : [q1, q2, . . . , q 4],
mks: [mleft, m right],
btw:{0,1}	
,(21)TABLE I
PARAMETERS OF THE VEHICLE DYNAMICS AND POTENTIAL FUNCTIONS
OFVLM-UDMC
Notation Value Notation Value Notation Value
kf(N/rad) -102129.83 kr(N/rad) -89999.98 lf(m) 1.287
lr(m) 1.603 m(kg) 1699.98 Iz(kg·m2) 2699.98
aNR 100.0 bNR 2.0 aCR 10.0
bCR 0.5 aV 500.0 bV 1.0
wR 3.5 aTL1200.0 aTL21000.0
ra 2.4 rb 1.0 roffset 0.25
Φfront(deg) (0,0,0) Φleft/right (deg) (25,0,90) Φrear(deg) (0,0,180)
where siis selected from intersection, roundabout, straight
urban road, highway, and others for the scenario-specific
reasoning of the second stage response. qi∈Bis a boolean
indicator indicating whether a zone is at risk. The spatial
delineation of risk zones is illustrated in Fig. 6. mleft/right ∈B
indicates non-crossable and crossable lane markings of the left
and right side as 0 and 1, respectively. In addition, btw ∈B
is a binary flag for blocking to wait, such as in the condition
of encountering red traffic lights.
3) Human Message Representation: The human message
H1for the VLM aggregates instant sensory data into a
structured format:
H1= (I,T1), (22)
where Idenotes the RGB images captured by onboard cam-
eras from front, left, right, and rear directions, represented
as tensors I ∈Rnc×H×W×3, in which ncis the number
of the onboard cameras. As revealed in Fig. 7, T1is the
textual message providing specific commands and questions to
the VLM regarding different perspectives of the autonomous
vehicle. Moreover, the second step of the foundation model
inference requires the response of the VLM A1to generate
the human message H2=Rec(H1,A1), where Rec (·)is the
reconstruction procedure. Along with the task description and
formal response guidance in S2, the basic prompt for the VLM
is ready to be invoked.
4) Dialogue Pipeline with In-Context Learning: To en-
hance contextual reasoning, VLM-UDMC retrieves dialogue
examples D={(Hi,Ai)}Nd
i=1from memory storage, where
Hi= [H1
i,H2
i]⊤is a historical human message and Ai=
[A1
i,A2
i]⊤is the corresponding response from the foundation
models. These examples are selected via similarity matching
to the current scenario with their specific embeddings evand
etgenerated by CLIP, a model trained on a vast amount of
image-text pairs from the internet [40].
Therefore, incorporating the two steps of foundation mod-
els, the complete context generation process can be formalized
as:
C(tm) = (S,H(tm),Retrieve (M,H(tm))), (23)
where Retrieve (·)is a function that selects relevant historical
examples from memory Mbased on similarity evaluation
to the current human message H(t). The VLM processes
this context to generate an updated attention dictionary A(t),

9
TABLE II
COMPARISON OF DRIVING PERFORMANCE IN URBAN DRIVING SCENARIOS
Method Scenario Comp. Time (ms) Col. TRV IB TTC Alarm Duration (s) Travel Time (s)
Autopilot [12] (rule-based)ML-ACC 10.61±3.57 0 0 0 0.10 (0.4%) 26.6
Roundabout 8.93±2.85 2 3 2 0.55 (1.5%) 35.6
Intersection 12.89±3.76 1 0 1 0.65 (3.6%) 18
Mix-T-Junction 5.14±0.64 1 0 0 \ \
InterFuser [41] (learning-based)ML-ACC 178.20±23.78 0 0 0 0 (0.0%) 52.9
Roundabout 179.98±22.62 0 0 0 0 (0.0%) 52.85
Intersection 180.33±25.28 0 0 0 0 (0.0%) 28.95
Mix-T-Junction 162.74±30.41 0 0 0 0 (0%) 45.75
RSS [42] (reactive-based)ML-ACC 11.62±3.14 0 0 0 0.15 (0.6%) 26.7
Roundabout 9.67±3.22 1 0 0 0.85 (2.3%) 37.75
Intersection 13.29±3.48 0 0 0 0.20 (1.0%) 20.5
Mix-T-Junction 6.49±9.81 1 2 0 0 (0.0%) 31.25
UDMC [16] (optimization-based)ML-ACC 35.07±13.00 0 0 0 0 (0.0%) 16.65
Roundabout 30.35±12.68 0 0 0 0 (0.0%) 17.8
Intersection 25.44±14.32 0 0 0 0 (0.0%) 16.5
Mix-T-Junction 27.32±11.34 0 0 0 0 (0.0%) 24.75
VLM-UDMC (Ours)ML-ACC 9.10±4.28 0 0 0 0 (0.0%) 15.5
Roundabout 12.74±5.79 0 0 0 0 (0.0%) 17.75
Intersection 9.81±6.67 0 0 0 0.05 (0.3%) 15.1
Mix-T-Junction 12.11±6.16 0 0 0 0 (0.0%) 22.60
Note: ML-ACC denotes the multilane adaptive cruise control scenario, while Mix-T-Junction represents the T-junction driving scenario
with mixed traffic conditions. Besides, Col, TRV , and IB represent the counts of collisions, traffic rule violations, and impolite behaviors,
respectively. TTC Alarm Duration (s) indicates the duration that the Time-To-Collision (TTC) value remains below 1.5 s.
which dynamically modifies the OCP’s potential functions:
F(P(t),X) =X
v∈V(t)Fv
V+X
p∈P(t)Fp
VRU
+X
m∈mksFm
M+btw·FTL,(24)
where V(t)andP(t)are the filtered vehicles and VRUs within
the identified risk zones defined in (21). Note that FMis
equivalent to FCR, as defined in (6), when mks = 1, and
corresponds to FNR, described in (7) otherwise. This structured
approach ensures that the VLM’s scene understanding is
systematically translated into actionable control commands,
enabling the autonomous vehicle to adapt to dynamic urban
environments while maintaining computational efficiency.
V. E XPERIMENTAL RESULTS AND ANALYSIS
In this section, we evaluate VLM-UDMC in both simulated
and real-world driving scenarios to assess its scene under-
standing and risk mitigation capabilities. We compare our
framework with representative baselines across diverse driving
scenarios, evaluating driving safety and efficiency through both
quantitative and qualitative measures. We also perform an ab-
lation study on key features of VLM-UDMC before assessing
its reasoning performance in real-world driving scenarios.
A. Environmental Settings
All experiments are conducted on a high-performance
workstation equipped with an AMD Ryzen Threadripper Pro
7975WX processor, featuring 32 cores and 64 threads at
3.5 G Hz. The system is configured with 512 GB of DDR5
RAM to support large-scale data processing and multitasking.For accelerated computation, the workstation includes two
NVIDIA RTX 6000 Ada Generation GPUs, each providing
48 GB of GDDR6 memory. The software environment was
based on Ubuntu 22.04 LTS operating system, with CUDA
12.1 and cuDNN 8.9 for GPU acceleration. All models were
implemented using Python 3.9, with deep learning frameworks
including PyTorch 2.4 and training scripts optimized for multi-
GPU execution. The dynamically formulated OCP is solved
using the Interior Point Optimizer (IPOPT) with CasADi [43].
In the CARLA 0.9.15 simulator [12], the autonomous vehi-
cle is controlled by throttle, brake, and steering, which are
transformed by a PID controller with a low-pass filter for
a smoother control input from the acceleration and steering
values generated by MPC. The surrounding vehicles are driven
by the embedded autopilot for a closed-loop evaluation.
The relevant parameters of the VLM-UDMC utilized in the
experiments are exhibited in Table I, which is of general utility
in urban driving scenarios. Φjdenotes the spatial configuration
with the order of (pitch, roll, yaw) of the j-th onboard camera.
B. Comparison and Effective Analysis
To evaluate the rationality of scene understanding and
risk attention generation for available potential functions, we
compare our proposed urban driving algorithm with a variety
of baselines, including rule-based, optimization-based, and
learning-based methods, including Autopilot [12], RSS [42],
InterFuser [41], and UDMC [16]. All methods are deployed
in identical driving scenarios, including multilane adaptive
cruise control, roundabout navigation, intersection crossing,
and T-junction driving with mixed traffic conditions. During
evaluation, surrounding traffic participants are consistently
spawned at the same points across different trials, and they

10
(a) Multiple lane adaptive cruise control with a stop sign
 (b) Roundabout with a front vehicle changing lane
(c) Crossroad with the ego vehicle is at the center of the intersection
 (d) T-junctions involving VRUs
Fig. 8. Demonstration of the multimodal dialogues during urban driving. The vision data contains the images from different perspectives, and the key insights
of the textual dialogue are underlined for clear illustration.
TABLE III
COMPARISON OF MODEL PERFORMANCE AND TOTAL INFERENCE TIME
Model MSE RMSE MAE Infer. Time (ms)
SGPR [34] 9.82 3.13 0.39 6.29
LSTM 1.25 1.12 0.43 0.18
Dlinear 1.41 1.19 0.34 0.17
Transformer 1.10 1.05 0.46 0.43
Ours 0.80 0.90 0.34 0.27
Note: Best results are in bold , second best ones are with underline .
react to the behavior of the ego vehicle to simulate realistic
interactions. The simulation results in Table II demonstrate
that our proposed VLM-UDMC achieves the highest traffic
efficiency across all four representative driving scenarios,
while maintaining robust safety in interactions with other
traffic participants. The fast-slow structure of VLM-UDMC
allows for the neglect of unnecessary traffic participants in
the adaptive OCP, thereby enhancing computational efficiency.
However, it is important to note that a time-to-collision alarm
briefly occurred during the intersection scenario due to the
oversight of a vehicle crossing at the same time.
In terms of the performance of the proposed traffic partic-
ipant prediction module, as shown in Table III, the proposedmulti-kernel prediction approach consistently achieves supe-
rior performance across all evaluation metrics compared to
existing baselines. Specifically, our method attains the lowest
MSE, RMSE, and MAE, representing improvements of 27.3%
in MSE and 14.3% in RMSE over the second-best Transformer
model. For MAE, our method reduces the error by 25.6%
compared to the Transformer. In addition to its state-of-the-art
accuracy, our method maintains a highly competitive inference
speed of 0.27 ms per batch, which is 37.2% faster than the
Transformer (0.43 ms). Reaching inference speeds on the
millisecond scale (under 0.3 ms) guarantees that the model
will not hinder downstream planning and control processes.
Consequently, the inference efficiency of our model is on
par with that of DLinear and LSTM. Moreover, compared
to the traditional SGPR method, our method is not only
substantially more accurate but also over 23 times faster in
inference. The results indicate that our proposed prediction
approach provides an optimal balance between accuracy and
computational efficiency, making it particularly suitable for
real-time or resource-constrained trajectory prediction tasks.
This capability is crucial for enabling the low-level OCP to
deliver vehicle control commands in real-time.
Qualitative analysis of the decisions made by VLM-
UDMC’s slow system is conducted to assess its effectiveness

11
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000014/uni00000011/uni00000015/uni00000018/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003/uni00000033/uni00000029/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048FV/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FV/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTTC/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFTTC/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FPD/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FPD/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000cFNR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FNR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FCR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFCR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTL/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FTL/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(a) Multiple lane adaptive cruise control scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000014/uni00000011/uni00000015/uni00000018/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003/uni00000033/uni00000029/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048FV/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FV/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTTC/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFTTC/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FPD/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FPD/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000cFNR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FNR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FCR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFCR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTL/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FTL/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(b) Roundabout scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000014/uni00000011/uni00000015/uni00000018/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003/uni00000033/uni00000029/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048FV/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FV/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTTC/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFTTC/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FPD/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FPD/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000cFNR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FNR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FCR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFCR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTL/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FTL/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(c) Crossroad scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000014/uni00000011/uni00000015/uni00000018/uni00000031/uni00000052/uni00000055/uni00000050/uni00000044/uni0000004f/uni0000004c/uni0000005d/uni00000048/uni00000047/uni00000003/uni00000033/uni00000029/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048FV/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FV/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTTC/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFTTC/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FPD/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FPD/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000cFNR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FNR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FCR/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFCR/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
FTL/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
FTL/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(d) Mixed traffic scenario
Fig. 9. The value of the assigned potential functions during different driving
scenarios compared with the baseline values.
in scene understanding and rational decision-making. Fig. 8
presents representative dialogue segments generated during
adaptive cruise control, roundabout navigation, intersection
crossing, and T-junction driving scenarios. In Fig. 8(a), as
the ego vehicle approaches the end of a straight urban road,
VLM-UDMC detects a blocking vehicle and a stop sign ahead.
After conducting relevant analysis, the slow system commands
the absorption of traffic participants from the front zone,
suggesting a lane change to the left. Although a vehicle is
present in the left lane, VLM-UDMC disregards it, as there
are no obstacles ahead and the current lane is deemed safe
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000017
/uni00000015
/uni00000013/uni00000015/uni00000017/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
/uni00000014/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c(a) Multiple lane adaptive cruise control scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000017
/uni00000015
/uni00000013/uni00000015/uni00000017/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
/uni00000014/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(b) Roundabout scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000017
/uni00000015
/uni00000013/uni00000015/uni00000017/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
/uni00000014/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(c) Crossroad scenario
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018
/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000017
/uni00000015
/uni00000013/uni00000015/uni00000017/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000037/uni0000004b/uni00000055/uni00000052/uni00000057/uni00000057/uni0000004f/uni00000048/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
/uni00000014/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000002c/uni00000051/uni00000053/uni00000058/uni00000057/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000c
/uni00000036/uni00000057/uni00000048/uni00000048/uni00000055/uni00000003/uni0000000b/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni0000000c
(d) Mixed traffic scenario
Fig. 10. Comparison of control inputs across different driving scenarios
between the proposed method and the baseline.
for the ego vehicle. Conversely, in the roundabout scenario
depicted in Fig. 8(b), VLM-UDMC’s fast system responds by
incorporating both front and left risk zones into the OCP for
attention. Additionally, when the ego vehicle turns right at
an intersection, as shown in Fig. 8(c), VLM-UDMC detects
no nearby traffic participants moving toward the ego vehicle,
identifying no risk zones and permitting a left-turn maneuver.
Finally, in Fig. 8(d), several VRUs are present on pedestrian
crossings, and VLM-UDMC focuses attention on the front
zone within the OCP, as other pedestrian routes do not intersect
with the ego vehicle’s path. Regarding the effectiveness of

12
TABLE IV
ABLATION RESULTS OF KEYFEATURES OF VLM-UDMC
Ablation ItemsCol. TRV IB TTC Alarm Travel Time
TS RZ MEM
✗ ✗ ✗ 2 4 3 0.1 s 43.15 s
✓ ✗ ✗ 1 2 2 0.8 s 16.6 s
✓ ✓ ✗ 0 1 0 0.05 s 15.3 s
✓ ✓ ✓ 0 0 0 0.05 s 15.1 s
Note: TS, RZ, and MEM denote Two-Step Reasoning, Risk Zone,
and Memory-based In-Context Learning, respectively.
the fast system of VLM-UDMC, the rational selection of risk
zones for attention, as depicted in Fig. 9, reduces the sum of
potential functions, thereby decreasing the complexity of the
OCP. This feature facilitates smoother and more consistent
spatial-temporal motion of the ego vehicle. As illustrated in
Fig. 10, across four representative urban scenarios, the control
inputs, including acceleration and steering angles, of our pro-
posed VLM-UDMC are notably smoother, with fewer action
adjustments compared to the baseline. UDMC incorporates all
surrounding traffic participants into the OCP, leading to a more
complex optimization problem with more local optima.
C. Ablation Study
In this part, we conduct an ablation study to evaluate the key
features of the proposed VLM-UDMC framework, focusing
on its application in complex urban scenarios, specifically
intersection crossings. The features under examination include
the two-step reasoning mechanism, risk zone representation,
and memory-based in-context learning.
As illustrated in Table IV, the VLM-UDMC variant lacking
all three features exhibits twice the number of collisions
with surrounding vehicles. This is attributed to the inadequate
scenario understanding provided by the one-step reasoning
process, which fails to accurately interpret the directions of
relevant traffic objects. Additionally, this version records high
counts of traffic rule violations and impolite behaviors, culmi-
nating in unsafe driving performance. Introducing the two-step
reasoning mechanism enhances scene comprehension, thereby
improving driving safety and efficiency. However, the inherent
latency in response from the foundational models results in
delayed updates of vehicle IDs, which are crucial for the
adaptive OCP. The integration of risk zone representation
into the ablated version significantly enhances urban driving
performance, eliminating collisions and impolite behaviors
entirely. Finally, the complete version of VLM-UDMC, which
incorporates all features, demonstrates no traffic rule violation
and achieves the shortest travel time during intersection cross-
ings. This underscores the effectiveness of the memory-based
in-context learning strategy in optimizing driving performance.
D. Field Experiment with a Physical Autonomous Vehicle
The scene understanding capabilities of our proposed urban
driving framework are evaluated using real-world driving
Fig. 11. The perception system of the intelligent vehicle. It contains six RGB
cameras mounted on the top of the vehicle to capture the complete surround
view for scene understanding and risk attention generation.
visual data. For this assessment, we conduct the autonomous
driving process on a university campus. Instantaneous RGB
images are downsampled and processed by locally deployed
SmolVLM and QWen3-4B foundation models. These models
use the two-step reasoning strategy to provide responses A
regarding risk zones and other potential functions related to
traffic rules from the slow system of VLM-UDMC.
As illustrated in Fig. 11, the perception system of the
intelligent vehicle is engineered through the integration of
six RGB cameras, which are strategically positioned on the
vehicle’s roof. This arrangement facilitates the acquisition of
a comprehensive surround view. The RGB cameras capture
images at a frequency of 10 Hz, which allows for near real-
time perception. It is important to note that the camera
configuration of the real vehicle differs slightly from our
proposed simulated camera setup. Consequently, we made
minor modifications to the prompt in the first stage of VLM
reasoning to accommodate the new camera configuration.
We present several response instances from VLM-UDMC to
demonstrate the effectiveness of our proposed two-stage scene
understanding mechanism and risk zone description strategy.
In Fig. 12(a), a silver sedan exits a parking lot and approaches
the ego vehicle. VLM-UDMC accurately identifies this situ-
ation, marking the front zone as at risk and incorporating all
detected obstacles in front of the ego vehicle into the fast
system. Additionally, VLM-UDMC recognizes the road curb
on the right and advises against crossing into the right lane.
Conversely, in another driving condition depicted in Fig. 12(b),
the front vehicle is not deemed an object of attention. Instead,
a person working near a car trunk is identified as a VRU,
activating the corresponding risk zone in the final response A.
The above analysis indicates that our proposed VLM-UDMC
can adapt to different sensor configurations from simulation
to real-world situations, while maintaining effective scene

13
(a) Straight urban road driving with an oncoming vehicle
(b) Straight urban road driving with a VRU on the side
Fig. 12. Multimodal dialogue demonstration during campus driving, with
vision data from six cameras. Key textual insights are underlined for clarity.
understanding and risk zone reasoning capabilities.
VI. D ISCUSSION AND CONCLUSION
A. Discussion
The proposed urban driving framework leverages foundation
models with an adaptive OCP to achieve human-like scene
understanding and the subsequent decision-making and mo-
tion control. The foundation models, enhanced by two-step
reasoning and RAG for in-context learning, leverage relevant
driving dialogues to generate risk zone-based insights that
directly inform the low-level MPC. Unlike previous parameter-
based adjustments by VLMs, this semantic guidance enables
dynamic adaptation of control objectives, ensuring safe and
compliant behaviors in complex scenarios such as left-turning
while yielding to oncoming traffic. The presented multi-kernel
prediction mechanism addresses the need for real-time motion
prediction by decomposing trajectories into trend and remain-
der components with outstanding predictive efficiency. Thisintegration with the fast system of VLM-UDMC improves the
rationality of decision-making and motion control, enabling
the autonomous vehicle to better anticipate and respond to the
movements of surrounding agents.
However, we observe that the performance of VLM-UDMC
is highly influenced by prompt engineering. For instance,
sending images in their original size to the VLM results in
significantly longer inference times and may lead to inac-
curate responses. Additionally, the system message must be
carefully designed to ensure stable and pertinent responses.
Furthermore, due to the inherent limitations of foundation
models, we are restricted to using only two modalities, vision
and language, for scene understanding. This constraint limits
the VLM’s ability to gain accurate spatial insights. Lastly, the
training-free and rapid retrieval capabilities of RAG present
an exciting opportunity for lifelong learning in autonomous
driving, making it a promising area for future research.
B. Conclusion
In this work, we propose a VLM-enhanced UDMC frame-
work with multiple time-scales for urban autonomous driving
that replicates human-like attention capabilities while main-
taining transparency and interpretability. Within this unified
framework, the upper decision-making system seamlessly
reconfigures the OCP by adaptively invoking the potential
functions representing the traffic factors. To enhance scene
understanding, the proposed two-step reasoning policy with
RAG utilizes foundation models to process multimodal inputs
and retrieve contextual knowledge, enabling the generation
of grounded risk-aware insights. This mechanism also pro-
motes greater inference consistency, particularly in long-tail
situations. Meanwhile, the motion control system employs
a lightweight multi-kernel decomposed LSTM to proactively
anticipate future trajectories of surrounding agents by extract-
ing smooth trend representations within the planning horizon.
Simulation and real-world experiments demonstrate the ef-
fectiveness and practicality of the framework. VLM-UDMC
represents a significant step forward in VLM-augmented au-
tonomous driving technology, offering a robust and adaptable
solution for modern urban mobility. Future research will focus
on addressing remaining challenges, including accurate spatial
understanding, multi-sized camera image compatibility, and
lifelong learning, to enhance performance and expand the
framework’s applicability, bringing fully autonomous urban
driving closer to realization.
REFERENCES
[1] L. Chen, Y . Li, C. Huang, B. Li, Y . Xing, D. Tian, L. Li, Z. Hu, X. Na,
Z. Li, et al. , “Milestones in autonomous driving and intelligent vehicles:
Survey of surveys,” IEEE Transactions on Intelligent Vehicles , vol. 8,
no. 2, pp. 1046–1056, 2022.
[2] Z. Gao, Z. Wu, W. Hao, K. Long, Y .-J. Byon, and K. Long, “Optimal
trajectory planning of connected and automated vehicles at on-ramp
merging area,” IEEE Transactions on Intelligent Transportation Systems ,
vol. 23, no. 8, pp. 12675–12687, 2022.
[3] Y . Zhang, X. Liang, D. Li, S. S. Ge, B. Gao, H. Chen, and T. H.
Lee, “Adaptive safe reinforcement learning with full-state constraints
and constrained adaptation for autonomous vehicles,” IEEE Transactions
on Cybernetics , vol. 54, no. 3, pp. 1907–1920, 2023.

14
[4] Z. Gao, Y . Mu, C. Chen, J. Duan, P. Luo, Y . Lu, and S. Eben Li,
“Enhance sample efficiency and robustness of End-to-End urban au-
tonomous driving via semantic masked world model,” IEEE Transac-
tions on Intelligent Transportation Systems , vol. 25, no. 10, pp. 13067–
13079, 2024.
[5] Y . Zhao, L. Wang, X. Yun, C. Chai, Z. Liu, W. Fan, X. Luo, Y . Liu, and
X. Qu, “Enhanced scene understanding and situation awareness for au-
tonomous vehicles based on semantic segmentation,” IEEE Transactions
on Systems, Man, and Cybernetics: Systems , vol. 54, no. 11, pp. 6537–
6549, 2024.
[6] S. Jia, Y . Zhang, X. Li, X. Na, Y . Wang, B. Gao, B. Zhu, and
R. Yu, “Interactive decision-making with switchable game modes for
automated vehicles at intersections,” IEEE Transactions on Intelligent
Transportation Systems , vol. 24, no. 11, pp. 11785–11799, 2023.
[7] H. Fan, F. Zhu, C. Liu, L. Zhang, L. Zhuang, D. Li, W. Zhu, J. Hu,
H. Li, and Q. Kong, “Baidu apollo EM motion planner,” arXiv preprint
arXiv:1807.08048 , 2018.
[8] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang,
W. Liu, C. Huang, and X. Wang, “V AD: Vectorized scene representation
for efficient autonomous driving,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision , pp. 8340–8350, 2023.
[9] L. Chen, Y . Li, C. Huang, Y . Xing, D. Tian, L. Li, Z. Hu, S. Teng,
C. Lv, J. Wang, D. Cao, N. Zheng, and F.-Y . Wang, “Milestones in
autonomous driving and intelligent vehicles—Part I: Control, computing
system design, communication, HD map, testing, and human behaviors,”
IEEE Transactions on Systems, Man, and Cybernetics: Systems , vol. 53,
no. 9, pp. 5831–5847, 2023.
[10] M. Zhang, N. Li, A. Girard, and I. Kolmanovsky, “A finite state
machine based automated driving controller and its stochastic opti-
mization,” in Dynamic Systems and Control Conference , vol. 58288,
p. V002T07A002, American Society of Mechanical Engineers, 2017.
[11] Y . Qi, B. He, R. Wang, L. Wang, and Y . Xu, “Hierarchical motion plan-
ning for autonomous vehicles in unstructured dynamic environments,”
IEEE Robotics and Automation Letters , vol. 8, no. 2, pp. 496–503, 2022.
[12] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V . Koltun,
“CARLA: An open urban driving simulator,” in Conference on Robot
Learning , pp. 1–16, PMLR, 2017.
[13] H. Pan, M. Luo, J. Wang, T. Huang, and W. Sun, “A safe motion
planning and reliable control framework for autonomous vehicles,” IEEE
Transactions on Intelligent Vehicles , vol. 9, no. 4, pp. 4780–4793, 2024.
[14] S. Xu and H. Peng, “Design, analysis, and experiments of preview
path tracking control for autonomous vehicles,” IEEE Transactions on
Intelligent Transportation Systems , vol. 21, no. 1, pp. 48–58, 2019.
[15] T. Br ¨udigam, M. Olbrich, D. Wollherr, and M. Leibold, “Stochastic
model predictive control with a safety guarantee for automated driving,”
IEEE Transactions on Intelligent Vehicles , vol. 8, no. 1, pp. 22–36, 2021.
[16] H. Liu, K. Chen, Y . Li, Z. Huang, M. Liu, and J. Ma, “UDMC: Unified
decision-making and control framework for urban autonomous driving
with motion prediction of traffic participants,” IEEE Transactions on
Intelligent Transportation Systems , vol. 26, no. 5, pp. 5856–5871, 2025.
[17] Y . Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du,
T. Lin, W. Wang, et al. , “Planning-oriented autonomous driving,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pp. 17853–17862, 2023.
[18] P. Liu, H. Liu, H. Liu, X. Liu, J. Ni, and J. Ma, “VLM-E2E: Enhancing
end-to-end autonomous driving with multimodal driver attention fusion,”
arXiv preprint arXiv:2502.18042 , 2025.
[19] W. Liu, P. Liu, and J. Ma, “DSDrive: Distilling large language model
for lightweight end-to-end autonomous driving with unified reasoning
and planning,” arXiv preprint arXiv:2505.05360 , 2025.
[20] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li,
Y . Zhang, Q. Zhang, et al. , “DiffusionDrive: Truncated diffusion model
for end-to-end autonomous driving,” in Proceedings of the Computer
Vision and Pattern Recognition Conference , pp. 12037–12047, 2025.
[21] X. Zhou, X. Han, F. Yang, Y . Ma, and A. C. Knoll, “OpenDriveVLA:
Towards end-to-end autonomous driving with large vision language
action model,” arXiv preprint arXiv:2503.23463 , 2025.
[22] Q. Li, X. Jia, S. Wang, and J. Yan, “Think2Drive: Efficient reinforcement
learning by thinking in latent world model for quasi-realistic autonomous
driving (in CARLA-v2),” in European Conference on Computer Vision ,
2024.
[23] A. Kuznietsov, B. Gyevnar, C. Wang, S. Peters, and S. V . Albrecht,
“Explainable ai for safe and trustworthy autonomous driving: A system-
atic review,” IEEE Transactions on Intelligent Transportation Systems ,
vol. 25, no. 12, pp. 19342–19364, 2024.[24] K. Long, H. Shi, J. Liu, and X. Li, “VLM-MPC: Vision Language
Foundation Model-guided model predictive controller for autonomous
driving,” arXiv preprint arXiv:2408.04821 , 2024.
[25] K. Atsuta, K. Honda, H. Okuda, and T. Suzuki, “LVLM-MPC: Collabo-
ration for autonomous driving: A safety-aware and task-scalable control
architecture,” arXiv preprint arXiv:2505.04980 , 2025.
[26] R. Yao, Y . Wang, H. Liu, R. Yang, Z. Peng, L. Zhu, and J. Ma,
“CALMM-Drive: confidence-aware autonomous driving with large mul-
timodal model,” arXiv preprint arXiv:2412.04209 , 2024.
[27] H. Guo, H. Zhu, J. Wang, V . Prahlad, W. K. Ho, C. W. de Silva, and
T. H. Lee, “Lightweight compressed temporal and compressed spatial
attention with augmentation fusion in remaining useful life prediction,”
in49th Annual Conference of the IEEE Industrial Electronics Society ,
pp. 1–6, 2023.
[28] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun,
“Transformers in time series: a survey,” in Proceedings of the Interna-
tional Joint Conference on Artificial Intelligence , pp. 6778–6786, 2023.
[29] X. Tian, J. Gu, B. Li, Y . Liu, Y . Wang, Z. Zhao, K. Zhan, P. Jia,
X. Lang, and H. Zhao, “DriveVLM: The convergence of autonomous
driving and large vision-language models,” in 8th Annual Conference
on Robot Learning , 2024.
[30] L. Li, C. Qian, J. Gan, D. Zhang, X. Qu, F. Xiao, and B. Ran,
“DCoMA: A dynamic coordinative merging assistant strategy for on-
ramp vehicles with mixed traffic conditions,” Transportation Research
Part C: Emerging Technologies , vol. 165, p. 104700, 2024.
[31] H. Liu, Z. Huang, Z. Zhu, Y . Li, S. Shen, and J. Ma, “Improved con-
sensus admm for cooperative motion planning of large-scale connected
autonomous vehicles with limited communication,” IEEE Transactions
on Intelligent Vehicles , 2024.
[32] H. Liu, R. Yao, W. Liu, Z. Huang, S. Shen, and J. Ma, “CoDriveVLM:
VLM-enhanced urban cooperative dispatching and motion planning
for future autonomous mobility on demand systems,” arXiv preprint
arXiv:2501.06132 , 2025.
[33] H. Lu, M. Zhu, C. Lu, S. Feng, X. Wang, Y . Wang, and H. Yang,
“Empowering safer socially sensitive autonomous vehicles using human-
plausible cognitive encoding,” Proceedings of the National Academy of
Sciences , vol. 122, no. 21, p. e2401626122, 2025.
[34] H. Liu, K. Chen, and J. Ma, “Incremental learning-based real-time
trajectory prediction for autonomous driving via sparse Gaussian process
regression,” in 2024 IEEE Intelligent Vehicles Symposium , pp. 1–7, 2024.
[35] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are Transformers effective
for time series forecasting?,” in Proceedings of the AAAI conference on
artificial intelligence , pp. 11121–11128, 2023.
[36] Q. Ge, Q. Sun, S. E. Li, S. Zheng, W. Wu, and X. Chen, “Numeri-
cally stable dynamic bicycle model for discrete-time control,” in IEEE
Intelligent Vehicles Symposium , pp. 128–134, 2021.
[37] A. Marafioti, O. Zohar, M. Farr ´e, M. Noyan, E. Bakouch, P. Cuenca,
C. Zakka, L. B. Allal, A. Lozhkov, N. Tazi, V . Srivastav, J. Lochner,
H. Larcher, M. Morlon, L. Tunstall, L. von Werra, and T. Wolf,
“SmolVLM: Redefining small and efficient multimodal models,” arXiv
preprint arXiv:2504.05299 , 2025.
[38] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lv, et al. , “Qwen3 technical report,” arXiv preprint
arXiv:2505.09388 , 2025.
[39] Y . Li, M. Tian, D. Zhu, J. Zhu, Z. Lin, Z. Xiong, and X. Zhao, “Drive-
R1: Bridging reasoning and planning in VLMs for autonomous driving
with reinforcement learning,” arXiv preprint arXiv:2506.18234 , 2025.
[40] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. , “Learning transferable
visual models from natural language supervision,” in International
Conference on Machine Learning , pp. 8748–8763, 2021.
[41] H. Shao, L. Wang, R. Chen, H. Li, and Y . Liu, “Safety-enhanced
autonomous driving using interpretable sensor fusion transformer,” in
Conference on Robot Learning , pp. 726–737, 2023.
[42] S. Shalev-Shwartz, S. Shammah, and A. Shashua, “On a formal model
of safe and scalable self-driving cars,” arXiv preprint arXiv:1708.06374 ,
2017.
[43] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl,
“CasADi – A software framework for nonlinear optimization and opti-
mal control,” Mathematical Programming Computation , vol. 11, no. 1,
pp. 1–36, 2019.