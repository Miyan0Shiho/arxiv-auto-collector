# SwarmVLM: VLM-Guided Impedance Control for Autonomous Navigation of Heterogeneous Robots in Dynamic Warehousing

**Authors**: Malaika Zafar, Roohan Ahmed Khan, Faryal Batool, Yasheerah Yaqoot, Ziang Guo, Mikhail Litvinov, Aleksey Fedoseev, Dzmitry Tsetserukou

**Published**: 2025-08-11 09:56:33

**PDF URL**: [http://arxiv.org/pdf/2508.07814v1](http://arxiv.org/pdf/2508.07814v1)

## Abstract
With the growing demand for efficient logistics, unmanned aerial vehicles
(UAVs) are increasingly being paired with automated guided vehicles (AGVs).
While UAVs offer the ability to navigate through dense environments and varying
altitudes, they are limited by battery life, payload capacity, and flight
duration, necessitating coordinated ground support.
  Focusing on heterogeneous navigation, SwarmVLM addresses these limitations by
enabling semantic collaboration between UAVs and ground robots through
impedance control. The system leverages the Vision Language Model (VLM) and the
Retrieval-Augmented Generation (RAG) to adjust impedance control parameters in
response to environmental changes. In this framework, the UAV acts as a leader
using Artificial Potential Field (APF) planning for real-time navigation, while
the ground robot follows via virtual impedance links with adaptive link
topology to avoid collisions with short obstacles.
  The system demonstrated a 92% success rate across 12 real-world trials. Under
optimal lighting conditions, the VLM-RAG framework achieved 8% accuracy in
object detection and selection of impedance parameters. The mobile robot
prioritized short obstacle avoidance, occasionally resulting in a lateral
deviation of up to 50 cm from the UAV path, which showcases safe navigation in
a cluttered setting.

## Full Text


<!-- PDF content starts -->

SwarmVLM: VLM-Guided Impedance Control for Autonomous
Navigation of Heterogeneous Robots in Dynamic Warehousing
Malaika Zafar, Roohan Ahmed Khan*, Faryal Batool*, Yasheerah Yaqoot*, Ziang Guo,
Mikhail Litvinov, Aleksey Fedoseev, and Dzmitry Tsetserukou
Abstract — With the growing demand for efficient logistics,
unmanned aerial vehicles (UA Vs) are increasingly being paired
with automated guided vehicles (AGVs). While UA Vs offer the
ability to navigate through dense environments and varying
altitudes, they are limited by battery life, payload capacity,
and flight duration, necessitating coordinated ground support.
Focusing on heterogeneous navigation, SwarmVLM ad-
dresses these limitations by enabling semantic collaboration be-
tween UA Vs and ground robots through impedance control. The
system leverages the Vision Language Model (VLM) and the
Retrieval-Augmented Generation (RAG) to adjust impedance
control parameters in response to environmental changes. In
this framework, the UA V acts as a leader using Artificial
Potential Field (APF) planning for real-time navigation, while
the ground robot follows via virtual impedance links with
adaptive link topology to avoid collisions with short obstacles.
The system demonstrated a 92% success rate across 12 real-
world trials. Under optimal lighting conditions, the VLM-RAG
framework achieved 80% accuracy in object detection and
selection of impedance parameters. The mobile robot prioritized
short obstacle avoidance, occasionally resulting in a lateral
deviation of up to 50 cm from the UA V path, which showcases
safe navigation in a cluttered setting.
Video of SwarmVLM: https://youtu.be/IdlUhQfz8w0
Keywords: Vision Language Model (VLM), Retrieval
Augmented Generation (RAG), Heterogeneous Robots,
Path Planning, Artificial Potential Field, Impedance Con-
trol
I. I NTRODUCTION
Advancements in automation and artificial intelligence
have significantly improved logistics and warehouse manage-
ment by enabling autonomous systems that reduce human
error and improve operational efficiency [1]. Automated
guided vehicles (AGVs) and mobile robots have traditionally
dominated indoor and last-mile delivery due to their reliable
localization and high payload capacity [2]. Unmanned aerial
vehicles (UA Vs), in contrast, offer fast and flexible movement
in three-dimensional space, making them ideal for navigating
over obstacles and through constrained environments. For
instance, Cristiani et al. [3] proposed a mini-drone swarm for
inventory tracking. However, UA Vs are limited by payload,
flight time, and sensing range. Integrating UA Vs with ground
robots enables a hybrid system that combines aerial agility
with ground-level stability and endurance.
The authors are with the Intelligent Space Robotics Laboratory, Skolkovo
Institute of Science and Technology, Moscow, Russia. {malaika.zafar,
roohan.khan, faryal.batool, yasheerah.yaqoot,
ziang.guo, mikhail.litvinov, aleksey.fedoseev,
d.tsetserukou }@skoltech.ru
*These authors contributed equally to this work.
Fig. 1. Framework for adaptive swarm navigation. The system processes
a top-down view of the environment through the VLM-RAG system to
determine impedance parameters based on the arrangements and number of
obstacles in the environment. Experiments were also performed in the real-
world environment to evaluate the system’s robustness and generalization
capabilities.
To enable such heterogeneous cooperation, we propose
a leader-follower framework called SwarmVLM , where
the drone leads using an Artificial Potential Field (APF)-
based planner, and the mobile robot follows via virtual
impedance linkages [4]. The mobile robot also forms tem-
porary impedance links with short obstacles undetectable
by the drone, while a custom PID controller minimizes its
path deviation. This architecture ensures reliable operation
in dense and dynamic environments. To further enhance
adaptability, we integrate a Vision-Language Model (VLM)
with a Retrieval-Augmented Generation (RAG) framework.
This VLM-RAG module interprets top-down environmental
views and retrieves context-specific impedance parameters
from a scenario database. It enables both agents to adjust
their impedance settings based on obstacle arrangement and
proximity, ensuring safe and intelligent navigation.
SwarmVLM introduces several key contributions:
•An impedance-based coordination mechanism for
drone–robot cooperation in cluttered environments.
•Integration of the VLM-RAG framework for un-
derstanding environmental configuration and adaptivearXiv:2508.07814v1  [cs.RO]  11 Aug 2025

impedance tuning.
•Real-world validation of the system in dynamic indoor
environments.
II. R ELATED WORK
Logistics and warehouse management have become criti-
cal components of modern supply chains, driving the demand
for more efficient and autonomous delivery solutions. In
response, both homogeneous and heterogeneous swarms of
robotic agents have been explored to meet these challenges.
Managing large UA V swarms, however, presents signif-
icant difficulties in coordination, stability, and real-time
decision-making. To address this, several studies have intro-
duced human-in-the-loop control approaches. For instance,
Abdi et al. [5] leveraged EMG signals to enable gesture-
based swarm control via muscle activity, while Khen et al. [6]
combined gesture recognition with machine learning for intu-
itive drone manipulation through natural human movements.
Despite these advances, achieving fully autonomous and
reliable swarm navigation in complex and dynamic envi-
ronments remains a major challenge. Many existing UA V
systems still struggle with robust perception, real-time adap-
tation, and decentralized coordination without human over-
sight. Among the most widely used methods for autonomous
navigation is the Artificial Potential Field (APF) technique,
appreciated for its simplicity and reactive obstacle avoid-
ance capabilities. Batinovic et al. [7] integrated APF with
LiDAR for real-time path planning in unstructured envi-
ronments, demonstrating its suitability for navigating dense
and dynamic spaces. Yu et al. [8] proposed a distributed
algorithm that couples APF with virtual leader formation and
a switching communication topology to ensure robust swarm
behavior.
To improve swarm cohesion and control, impedance-
based strategies have been introduced. Tsykunov et al. [9]
developed SwarmTouch, which utilized virtual impedance
links [4] for drone swarm coordination. Building on this,
Fedoseev et al. [10] analyzed the influence of impedance
topologies on swarm stability. Khan et al. [11] later combined
APF-based leader planning with impedance-based formation
control, but their approach remained limited to static environ-
ments and did not address challenges like energy efficiency
or long-term operation.
To overcome the limitations of aerial-only or ground-only
systems, research has increasingly focused on heterogeneous
swarm architectures. Darush et al. [12] employed virtual
impedance links between a leader octocopter and micro-
drones, while [13] used reinforcement learning to facilitate
docking, transport, and in-air recharging. Chen et al. [14]
proposed a UA V-AGV system for collaborative exploration
in hazardous areas, highlighting the advantages of combining
agents with complementary capabilities to enhance flexibility
and robustness.
Ground robots also play a crucial role in warehouse logis-
tics. Malopolski et al. [15] introduced an autonomous mobile
robot equipped with a hybrid drive system for navigating
flat and rail surfaces, and an integrated elevator for verticalmobility. While effective in structured spaces, such systems
lack aerial reach and adaptability in cluttered or hard-to-
access environments.
Further demonstrating the benefits of aerial-ground col-
laboration, Salas et al. [16] presented a UA V-AGV system
for search and rescue. Their UA V provided aerial pathfinding
using a monocular camera, while the ground robot performed
close-range inspection and mapping, showing the value of
layered autonomy in unstructured terrains.
Parallel to these developments, vision-based learning mod-
els, particularly Vision Transformers (ViTs), have shown
promise in robotic perception and decision-making. Dosovit-
skiy et al. [17] demonstrated the effectiveness of transformers
for image recognition, paving the way for their use in aerial
robotics for tasks like object detection and spatial under-
standing. More recently, Brohan et al. [18] introduced RT-1,
a task-conditioned transformer capable of generalizing across
diverse robotic control tasks in real-world environments.
Multimodal transformers such as Molmo-7B-D [19] in-
tegrate visual, spatial, and linguistic inputs, enabling rich
contextual awareness for autonomous systems. These archi-
tectures are especially useful for coordination and decision-
making in cluttered or uncertain environments. For instance,
FlockGPT [20] introduced a generative AI interface that
allows users to control drone flocks using natural language,
achieving high accuracy and strong usability for dynamic
shape formation.
Building on these foundations, our research proposes a
novel heterogeneous swarm system that combines APF-based
aerial navigation, impedance-guided ground mobility, and a
VLM-RAG-powered framework for adaptive, context-aware
behavior. Inspired by prior work on HetSwarm [21] and
ImpedanceGPT [22], this study introduces a new agile and
safe path planning solution for drone–robot teams operating
in dynamic and cluttered environments.
III. S WARM VLM T ECHNOLOGY
The proposed methodology, illustrated in Fig. 2, consists
of two primary components: a VLM-RAG system for esti-
mating impedance parameters and a high-level control frame-
work that integrates APF planner with impedance-based
coordination. The system enables collaboration between a
drone and a mobile robot in dynamic environments. The
drone uses the APF planner to compute and update its trajec-
tory in real-time, focusing on global navigation and obstacle
avoidance. The mobile robot follows this trajectory through
virtual impedance links, which help maintain formation and
allow local navigation around small obstacles not perceived
by the drone. Additionally, the robot can act as a mobile
landing or recharging platform for the drone.
The methodology was first validated in the Gym PyBullet
[23] simulation environment using custom PID controllers
for both agents to ensure accurate and coordinated path
following. Real-world experiments were then conducted,
demonstrating effective drone–robot collaboration in clut-
tered environments. Communication between the drone and

Fig. 2. System architecture of the proposed SwarmVLM , integrating a VLM-RAG module for impedance estimation with a high-level control system that
combines APF-based path planning and impedance control for heterogeneous navigation.
the mobile robot is handled via ROS, enabling synchronized
behavior through a shared information framework.
A. Artificial Potential Fields for Global Path Generation
In order for the leader drone to navigate efficiently around
the obstacles while setting the path toward the goal, we
applied the APF planning algorithm [24]. This algorithm
allows the UA V to continuously update its trajectory in
response to shifting obstacles, enabling robust navigation in
cluttered and dynamic environments. The equations for the
APF planner are as follows [11]:
Ftotal=Fattraction +Frepulsion , (1)
where
Fattraction (dg) =katt·dg,
Frepulsion (do) =(
0 ifdo> d safe
krep·
1
do−1
dsafe
ifdo≤dsafe,
where dganddoare the distances from the drone to the
goal and to the obstacle, respectively, kattandkrepare the
attraction and repulsion coefficients, respectively.
B. Impedance Controller
To enable smooth coordination, a virtual impedance con-
troller couples the drone and the mobile robot. The mobile
robot acts as a follower, linked to the drone’s APF-based
trajectory through a mass spring damper system that ensures
stable formation tracking. This dynamic coupling is governed
by:
m∆¨x+d∆ ˙x+k∆x=Fext(t), (2)
where ∆x,∆ ˙x, and ∆¨xrepresent deviations in position,
velocity, and acceleration from the desired state; m,d, andkare virtual mass, damping, and stiffness; and Fext(t)is the
virtual force from the drone.
To handle short obstacles undetected by the drone, the
mobile robot temporarily disengages from the drone and
forms local impedance links with the obstacles. The resulting
repulsive displacement is given by:
∆xrobot=kimpF·rimp, (3)
where kimpF is the velocity-dependent force coefficient and
rimpdefines the influence radius of the obstacle. This mech-
anism enables timely, collision-free navigation in cluttered
environments. This equation ensures a collision-free trajec-
tory by applying a repulsive displacement proportional to
the obstacle’s influence radius rimp, redirecting the robot
away from potential collisions. The coefficient kimpFadjusts
the strength of this deflection based on the robot’s velocity,
ensuring timely and stable avoidance.
C. VLM-RAG System
The VLM-RAG system processes a top-down visual view
of the environment using the Molmo-7B-D BnB 4-bit model
to extract key features such as the number and spatial distri-
bution of obstacles. These features are converted into vector
representations and passed to the Retrieval-Augmented Gen-
eration (RAG) system, which retrieves optimal impedance
control parameters — namely, virtual mass ( m), virtual
stiffness ( k), virtual damping ( d), and the impedance force
coefficient ( F). Communication between the VLM-RAG
module, running on the server, and the heterogeneous swarm
system is handled through the ROS framework.
1) Integrating VLM for Obstacle Identification and Spatial
Analysis
The Vision-Language Model (VLM) uses visual input
from a ceiling-mounted camera to semantically analyze the

environment by detecting, and localizing obstacles. For ef-
ficient real-time performance, the lightweight Molmo-7B-D
BnB 4-bit model [19] is employed. The extracted obstacle
information is then forwarded to the RAG system, which
uses the spatial configuration to retrieve suitable impedance
control parameters, thereby enhancing swarm navigation in
both static and dynamic environments.
2) Custom Database for multiple environmental cases
A custom database of six environmental scenarios was
created to support the RAG system. Each scenario repre-
sents an indoor space with different obstacle quantities and
spatial arrangements. The database stores empirically derived
optimal impedance parameters obtained through simulation
experiments using the APF planner for drone navigation.
Table I summarizes the parameters for all six cases.
TABLE I
DATABASE CONTAINS THE OPTIMAL IMPEDANCE PARAMETERS ,
INCLUDING VIRTUAL MASS (M),VIRTUAL STIFFNESS (K),VIRTUAL
DAMPING (D),AND IMPEDANCE OR DEFLECTION FORCE COEFFICIENT
(F), FOR SIX DIFFERENT CASES
Parameter m(kg) k(N/m) d(N·s/m) Fcoeff
Case I 1.0 5.0 2.5 0.68
Case II 1.5 3.0 3.5 0.35
Case III 1.2 3.5 3.0 0.45
Case IV 1.3 3.4 3.6 0.45
Case V 1.4 3.3 3.7 0.15
Case VI 1.2 3.8 4.0 0.65
3) Retrieval-Augmented Generation (RAG) for Impedance
Parameter Generation
Retrieval-Augmented Generation (RAG) systems typically
enhance Large Language Models (LLMs) by integrating
external knowledge retrieval into the generation process. In
this work, a na ¨ıve RAG implementation is used. It employs
a sentence transformer to generate vector embeddings from
textual queries and utilizes Facebook AI Similarity Search
(FAISS) for fast and efficient nearest-neighbor retrieval. An
exact nearest-neighbor search algorithm is adopted to ensure
precise matching. The RAG pipeline processes the textual
output of the VLM, embeds it as a query vector, and performs
a similarity search in the database of impedance parameters
using Euclidean distance given by:
d(X, Y) =vuutnX
i=1(Xi−Yi)2, (4)
where X= (X1, X2, ..., X n)is the query embedding, Y=
(Y1, Y2, ..., Y n)is the stored embedding, n= 384 is the
embedding dimension.
The closest matching case is returned, and its associated
impedance parameters are used to adjust the behavior of the
swarm according to the current environmental context.
IV. E XPERIMENTAL SETUP IN SIMULATION
ENVIRONMENT
A custom simulation environment was developed using the
Gym-PyBullet framework (Fig. 3) to model the dynamics ofthe drone and mobile robot. The setup supports real-time
visualization and allows dynamic repositioning of obstacles
for flexible testing. The drone navigates using an Artificial
Potential Field (APF) planner, while the mobile robot follows
the drone’s trajectory through impedance-based coordination.
Fig. 3. Experimental setup in the Gym Pybullet environment showing the
trajectory of the drone and mobile robot under a dense environment. Where
the red line shows the impedance linkages and the black and blue lines
show the drone and mobile robot paths, respectively.
V. R EAL-WORLD EXPERIMENTAL SETUP
The real-world setup consists of a drone and a mobile
robot communicating via ROS over a shared Wi-Fi network.
The drone operates as the ROS master using an onboard
Orange Pi, while the mobile robot functions as a ROS slave
running on an Intel NUC. Initially operating independently,
both agents are synchronized through ROS for collaborative
operation.
The drone continuously publishes its planned target posi-
tion and the mobile robot’s current position (as tracked by
a VICON motion capture system). Upon synchronization,
the mobile robot subscribes to this data and uses a PID
controller to compute velocity commands, enabling it to
follow the drone’s trajectory while maintaining formation
and avoiding local obstacles. A total of 12 experiments
were conducted with varying environmental configurations
by altering obstacle positions.
A. Experimental Result
Trajectories from three of the experiments are shown: two
in static environments, and Case III evaluated under both
static and dynamic conditions.
1) Results in static environment
Fig. 4 demonstrates how the drone leads the mobile robot
through virtual impedance-based coupling while successfully
avoiding tall obstacles. The mobile robot actively avoids
short ground obstacles, which are not perceived or avoided
by the drone since they do not interfere with its flight path.
2) Results in dynamic environment
In Case III (Fig. 5), one of the short obstacles was
made dynamic to simulate a real-world scenario. As the
responsibility for avoiding short obstacles lies solely with the
mobile robot, the figure shows that while the drone follows
the same planned trajectory in both static and dynamic
scenarios, the mobile robot adjusts its path in real time to
avoid the approaching obstacle.

Fig. 4. Results in static environment. CASE I: One tall obstacle, one short
obstacle, CASE II: two tall obstacles, one short obstacle.
Fig. 5. Results in a dynamic environment. CASE III: One Tall Obstacle
– Two Short Obstacles.
3) Deviation of mobile robot path from drone path
While the drone demonstrates minimal deviation from
its planned path due to real-time APF-based planning, the
mobile robot deviates to avoid short obstacles that are not in
the path of the drone, especially in cluttered environments.
These deviations highlight the mobile robot’s role in ensuring
collision avoidance while maintaining formation. The quan-
titative deviations are summarized in Table II.
TABLE II
MOBILE ROBOT PATH OFFSET DUE TO SHORT OBSTACLE AVOIDANCE
Case I IIIII
(Static)IV
(Dynamic)
Mobile Robot Deviation (m) 0.45 0.45 0.45 0.43
4) Velocity Distribution along the trajectory
The velocity profiles in Fig. 6 capture the real-world
motion behavior of both agents. The drone maintains a rela-
tively high and consistent velocity, with slight accelerations
near tall obstacles followed by gradual deceleration. The
mobile robot exhibits a more variable profile, accelerating
when simultaneously following the drone and avoiding short
obstacles, and decelerating as it reaches the final target.
B. Trajectory Analysis Across Experimental Cases
Table III shows the trajectory lengths of the drone and
mobile robot. The robot consistently takes a longer path due
to short obstacle avoidance, while the drone flies directly
from them. Case III highlights performance in both static
and dynamic settings, reflecting system adaptability.
Fig. 6. Velocity profile along the trajectory in real world scenarios.
TABLE III
TRAJECTORY LENGTH COMPARISON BETWEEN DRONE AND MOBILE
ROBOT
CaseDrone
trajectory (m)Mobile Robot
trajectory (m)
Case I 4.272 5.593
Case II 4.492 5.822
Case III (Static) 4.684 5.634
CASE III (Dynamic) 4.672 5.710
Experiments were conducted under varying conditions.
Out of 12 trials, 11 of them succeeded. One failure resulted
from synchronization issues between both the agents and
limited flight area due to the large mobile robot’s size. The
success rate was computed to be 92%. This high success rate
demonstrates the robustness of the system and its capability
to adapt to dynamic and diverse real-world conditions.
C. Evaluation of VLM-RAG Framework
Fig. 7 shows that the VLM-RAG system achieves 80%
success in detecting and retrieving short and tall objects
under good lighting. In poor lighting, the success rate drops
to 60%, mainly due to difficulty identifying tall obstacles.
Despite lighting variations, the system consistently detects
obstacles regardless of color or placement.
VI. C ONCLUSION AND FUTURE WORK
SwarmVLM introduces a heterogeneous robotic system
that integrates the aerial agility of a drone with the ground-
level adaptability of a mobile robot, enabling resilient naviga-
tion in complex and dynamic environments. Through the use
of impedance-based coordination, APF-driven path planning,

Fig. 7. Performance of the VLM-RAG system under different lighting
conditions.
and vision-language perception via VLM-RAG, the system
achieved a navigation success rate of approximately 92% in
real-world scenarios. The drone serves as the path leader,
while the mobile robot ensures continuous tracking and
obstacle avoidance. Moreover, due to its focus on bypassing
short obstacles, the mobile robot exhibited a deviation of up
to50 cm from the drone’s trajectory. Furthermore, real-time
impedance parameter tuning enhances the system’s adapt-
ability to environmental changes. The VLM-RAG module
also achieved 80% accuracy in detecting relevant objects
and retrieving appropriate control parameters under optimal
lighting conditions. Therefore, these combined capabilities
enable both agents to adjust their behavior dynamically,
ensuring intelligent and dependable operation across varied
environments.
In future work, this research can be extended to incor-
porate multiple drones and mobile robots operating collabo-
ratively. Additionally, integrating advanced computer vision
techniques may further enhance real-time impedance param-
eter tuning by enabling more accurate scene interpretation
and contextual awareness.
REFERENCES
[1] S. L ´opez-Soriano and R. Pous, “Inventory robots: Performance eval-
uation of an rfid-based navigation strategy,” IEEE Sensors Journal ,
vol. 23, no. 14, pp. 16 210–16 218, July 2023.
[2] A. Motroni, S. D’Avella, A. Buffi, P. Tripicchio, M. Unetti, G. Cecchi,
and P. Nepa, “Advanced rfid-robot with rotating antennas for smart
inventory in high-density shelving systems,” IEEE Journal of Radio
Frequency Identification , vol. 8, pp. 559–570, Feb. 2024.
[3] D. Cristiani, F. Bottonelli, A. Trotta, and M. Di Felice, “Inventory
management through mini-drones: Architecture and proof-of-concept
implementation,” in Proc. IEEE Int. Symposium on ”A World of
Wireless, Mobile and Multimedia Networks” (WoWMoM) , Aug. 31-
Sept. 3, 2020, pp. 317–322.
[4] N. Hogan, “Impedance control: An approach to manipulation,” in Proc.
American Control Conference , June 6-8, 1984, pp. 304–313.
[5] S. S. Abdi and D. A. Paley, “Safe operations of an aerial swarm via a
cobot human swarm interface,” in Proc. IEEE Int. Conf. on Robotics
and Automation (ICRA) , May 29-June 2, 2023, pp. 1701–1707.
[6] G. Khen, D. Zhao, and J. Baca, “Intuitive human-swarm interaction
with gesture recognition and machine learning,” in Association for
Computing Machinery , New York, NY , USA, Oct. 23-26, 2023, p.
453–456.[7] A. Batinovic, J. Goricanec, L. Markovic, and S. Bogdan, “Path plan-
ning with potential field-based obstacle avoidance in a 3d environment
by an unmanned aerial vehicle,” in Proc. IEEE Int. Conf. on Unmanned
Aircraft Systems (ICUAS) , June 21-24, 2022, pp. 394–401.
[8] Y . Yu, C. Chen, J. Guo, M. Chadli, and Z. Xiang, “Adaptive formation
control for unmanned aerial vehicles with collision avoidance and
switching communication network,” IEEE Transactions on Fuzzy
Systems , vol. 32, no. 3, pp. 1435–1445, March 2024.
[9] E. Tsykunov, R. Agishev, R. Ibrahimov, L. Labazanova, A. Tleugazy,
and D. Tsetserukou, “SwarmTouch: Guiding a swarm of micro-
quadrotors with impedance control using a wearable tactile interface,”
IEEE Transactions on Haptics , vol. 12, no. 3, pp. 363–374, July 2019.
[10] A. Fedoseev, A. Baza, A. Gupta, E. Dorzhieva, R. N. Gujarathi, and
D. Tsetserukou, “DandelionTouch: High fidelity haptic rendering of
soft objects in vr by a swarm of drones,” in Proc. IEEE Int. Conf. on
Systems, Man, and Cybernetics (SMC) , Oct. 9-12, 2022, pp. 1078–
1083.
[11] R. A. Khan, M. Zafar, A. Batool, A. Fedoseev, and D. Tsetserukou,
“SwarmPath: Drone swarm navigation through cluttered environments
leveraging artificial potential field and impedance control,” in Proc.
IEEE Int. Conf. on Robotics and Biomimetics (ROBIO) , Dec. 10-14,
2024, pp. 402–407.
[12] Z. Darush, M. Martynov, A. Fedoseev, A. Shcherbak, and D. Tset-
serukou, “SwarmGear: Heterogeneous swarm of drones with mor-
phogenetic leader drone and virtual impedance links for multi-agent
inspection,” in Proc. IEEE Int. Conf. on Unmanned Aircraft Systems
(ICUAS) , June 6-9, 2023, pp. 557–563.
[13] S. Karaf, A. Fedoseev, M. Martynov, Z. Darush, A. Shcherbak,
and D. Tsetserukou, “MorphoLander: Reinforcement learning based
landing of a group of drones on the adaptive morphogenetic uav,” in
Proc. IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC) , Oct.
1-4, 2023, pp. 2507–2512.
[14] Y . Chen and J. Xiao, “Target search and navigation in heterogeneous
robot systems with deep reinforcement learning,” Machine Intelligence
Research , vol. 22, no. 1, p. 79–90, Jan 2025.
[15] W. Małopolski and S. Skoczypiec, “The concept of an autonomous
mobile robot for automating transport tasks in high-bay warehouses,”
Advances in Science and Technology Research Journal , vol. 18, pp.
1–10, April 2024.
[16] W. L. Salas, L. M. Valent ´ın-Coronado, I. Becerra, and A. Ram ´ırez-
Pedraza, “Collaborative object search using heterogeneous mobile
robots,” in Proc. IEEE Int. Autumn Meeting on Power, Electronics
and Computing (ROPEC) , vol. 5, Nov. 10-12, 2021, pp. 1–6.
[17] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner et al. , “An image is worth 16x16 words: Transformers
for image recognition at scale,” 2021, arXiv:2010.11929.
[18] A. Brohan, N. Brown, J. Carbajal, Y . Chebotar, J. Dabis, C. Finn
et al. , “RT-1: Robotics transformer for real-world control at scale,”
2023, arXiv:2212.06817.
[19] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y . Yang, J. S. Park et al. ,
“Molmo and PixMo: Open weights and open data for state-of-the-art
multimodal models,” 2024, arXiv:2409.17146.
[20] A. Lykov, S. Karaf, M. Martynov, V . Serpiva, A. Fedoseev, M. Ko-
nenkov, and D. Tsetserukou, “Flockgpt: Guiding uav flocking with
linguistic orchestration,” in 2024 IEEE International Symposium on
Mixed and Augmented Reality Adjunct (ISMAR-Adjunct) , Oct. 21-15,
2024, pp. 485–488.
[21] M. Zafar, R. A. Khan, A. Fedoseev, K. K. Jaiswal, P. B. Sujit, and
D. Tsetserukou, “HetSwarm: Cooperative navigation of heterogeneous
swarm in dynamic and dense environments through impedance-based
guidance,” in Proc. IEEE Int. Conf. on Unmanned Aircraft Systems
(ICUAS) , May 14-17, 2025, pp. 309–315.
[22] F. Batool, M. Zafar, Y . Yaqoot, R. A. Khan, M. H. Khan, A. Fe-
doseev, and D. Tsetserukou, “ImpedanceGPT: Vlm-driven impedance
control of swarm of mini-drones for intelligent navigation in dynamic
environment,” 2025, arxiv:2503.02723.
[23] J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, and A. P. Schoellig,
“Learning to fly—a gym environment with pybullet physics for
reinforcement learning of multi-agent quadcopter control,” in 2021
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS) , 2021, pp. 7512–7519.
[24] H. Li, “Robotic path planning strategy based on improved artificial
potential field,” in Proc. Int. Conf. on Artificial Intelligence and
Computer Engineering (ICAICE) , Oct. 23-25, 2020, pp. 67–71.