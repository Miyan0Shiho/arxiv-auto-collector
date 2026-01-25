# HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation

**Authors**: Yara Mahmoud, Yasheerah Yaqoot, Miguel Altamirano Cabrera, Dzmitry Tsetserukou

**Published**: 2026-01-21 11:04:19

**PDF URL**: [https://arxiv.org/pdf/2601.14874v1](https://arxiv.org/pdf/2601.14874v1)

## Abstract
Humanoid robots must adapt their contact behavior to diverse objects and tasks, yet most controllers rely on fixed, hand-tuned impedance gains and gripper settings. This paper introduces HumanoidVLM, a vision-language driven retrieval framework that enables the Unitree G1 humanoid to select task-appropriate Cartesian impedance parameters and gripper configurations directly from an egocentric RGB image. The system couples a vision-language model for semantic task inference with a FAISS-based Retrieval-Augmented Generation (RAG) module that retrieves experimentally validated stiffness-damping pairs and object-specific grasp angles from two custom databases, and executes them through a task-space impedance controller for compliant manipulation. We evaluate HumanoidVLM on 14 visual scenarios and achieve a retrieval accuracy of 93%. Real-world experiments show stable interaction dynamics, with z-axis tracking errors typically within 1-3.5 cm and virtual forces consistent with task-dependent impedance settings. These results demonstrate the feasibility of linking semantic perception with retrieval-based control as an interpretable path toward adaptive humanoid manipulation.

## Full Text


<!-- PDF content starts -->

HumanoidVLM: Vision–Language–Guided Impedance Control for
Contact-Rich Humanoid Manipulation
Yara Mahmoud
Skolkovo Institute of Science and Technology
Moscow, Russia
yara.mahmoud@skoltech.ruYasheerah Yaqoot
Skolkovo Institute of Science and Technology
Moscow, Russia
yasheerah.yaqoot@skoltech.ru
Miguel Altamirano Cabrera
Skolkovo Institute of Science and Technology
Moscow, Russia
m.altamirano@skoltech.ruDzmitry Tsetserukou
Skolkovo Institute of Science and Technology
Moscow, Russia
d.tsetserukou@skoltech.ru
Figure 1: Overall architecture of the proposed HumanoidVLM framework.
Abstract
Humanoid robots must adapt their contact behavior to diverse
objects and tasks, yet most controllers rely on fixed, hand-tuned
impedance gains and gripper settings. This paper introduces Hu-
manoidVLM, a vision–language driven retrieval framework that en-
ables the Unitree G1 humanoid to select task-appropriate cartesian
impedance parameters and gripper configurations directly from
an ego-centric RGB image. The system couples a vision–language
model for semantic task inference with a FAISS-based Retrieval-
Augmented Generation (RAG) module that retrieves experimentally
validated stiffness–damping pairs and object-specific grasp angles
from two custom databases, and executes them through a task-space
impedance controller for compliant manipulation. We evaluate Hu-
manoidVLM on 14 visual scenarios and achieve a retrieval accuracy
of 93%. Real-world experiments show stable interaction dynamics,
with𝑧-axis tracking errors typically within1 –3.5cm and virtual
forces consistent with task-dependent impedance settings. These
results demonstrate the feasibility of linking semantic perception
with retrieval-based control as an interpretable path toward adap-
tive humanoid manipulation.CCS Concepts
•Human-centered computing →Collaborative interaction;•
Computing methodologies →Vision for robotics;•Computer
systems organization→Robotic control.
Keywords
Human-Robot Interaction, Vision Language Model, Retrieval-Aug-
mented System, Task-Space compliance, impedance control
1 Introduction
Humanoid robots are increasingly expected to operate in unstruc-
tured human environments, where tasks such as placing objects,
applying force on surfaces, grasping diverse items, or manipulating
tools require a combination of semantic understanding and com-
pliant physical interaction [ 8]. Conventional control pipelines for
such systems typically rely on fixed, hand-tuned impedance gains
[12] and manually specified gripper configurations, which restrict
the robot’s ability to adapt to changes in scene geometry, object
properties, or task intent.arXiv:2601.14874v1  [cs.RO]  21 Jan 2026

Yara Mahmoud, Yasheerah Yaqoot, Miguel Altamirano Cabrera, and Dzmitry Tsetserukou
In parallel, recent advances in Vision–Language Models (VLMs)
have demonstrated impressive capabilities in open-world percep-
tion, contextual reasoning, and grounded language understanding
[6], making them promising candidates for high-level task inference
in robotics [ 9]. However, despite their strong semantic understand-
ing, these models remain largely disconnected from the low-level
control parameters that determine how a robot physically interacts
with its environment. As a result, current humanoid systems lack
the capacity to autonomously translate visual semantics into ac-
tionable, task-appropriate interaction behaviors—a limitation that
is especially pronounced in contact-rich manipulation tasks, where
selecting correct stiffness, damping, and grasping configurations is
essential for safe and stable execution.
To bridge this gap, we introduce HumanoidVLM, a retrieval-
augmented vision–language framework that connects high-level
visual reasoning with low-level manipulation parameters for the
Unitree G1 humanoid robot. Given a single egocentric image, Hu-
manoidVLM infers the ongoing manipulation task through struc-
tured visual queries and retrieves the corresponding cartesian impe-
dance parameters and object-specific gripper angle from custom,
experimentally validated databases. This approach enables the robot
to autonomously determine task-appropriate compliance settings
for a given scene.
2 Related Works
Foundation models and semantic manipulation.Recent ad-
vances in foundation models have accelerated progress in seman-
tic manipulation and scene-aware task execution. OmniVIC [ 18]
and ImpedanceGPT [ 7] are most closely related to our approach,
combining a vision–language model with a variable impedance
controller through retrieval. However, OmniVIC focuses on indus-
trial arms and does not address humanoid embodiment, shared
workspaces, or HRI-oriented safety. Vision–language–action sys-
tems such as PG-VLM [ 10], SayCan [ 4], RT-2 [ 19],Bi-VLA [ 11], and
SwarmVLM [ 17] show that multimodal models can generalize ma-
nipulation behavior, but these frameworks operate at the policy
or pose level and rely on fixed impedance gains. Broader surveys
on foundation models for manipulation and robot intelligence [ 15],
[13] highlight the role of semantic reasoning but do not consider
continuous impedance modulation in humanoid settings. Work
on foundation models for collaborative assembly [ 14] emphasizes
contextual reasoning in shared autonomy, yet compliant control
remains largely unaddressed.
Impedance and variable impedance control.Classical impe-
dance control, introduced by Hogan [ 12], provides a structured
formulation for regulating robot–environment interaction through
stiffness and damping shaping. Surveys by Abu-Dakka and Save-
riano [ 3] document advances in variable impedance strategies, in-
cluding learning-based adaptation and physical interaction applica-
tions. In human–robot collaboration, Ajoudaniet al.[ 5] stress the
importance of compliance modulation for safety and ergonomics,
following earlier studies on tactile and impedance-based interaction
in humanoid systems [ 16]. Building on this foundation, our work
links semantic scene understanding to impedance and gripper selec-
tion through retrieval-augmented reasoning, enabling task-aware
compliant control on a humanoid.3 System Architecture
The proposed system enables the Unitree G1 humanoid to adapt
its end-effector impedance and grasping behavior using visual
scene understanding and retrieval-based reasoning. An ego-centric
RGB image from the robot’s head camera is processed by a vision–
language model (VLM), which infers the high-level manipulation
task through structured visual queries. The resulting semantic task
label is embedded and passed to a Retrieval-Augmented Generation
(RAG) module that performs FAISS-based similarity search over
two custom databases: (i) a cartesian impedance database storing
task-specific stiffness and damping coefficients for the end-effector,
and (ii) a gripper-angle database specifying the optimal grasp con-
figuration for the object category.
The retrieved control parameters ( 𝐾,𝐷, and gripper angle 𝛾)
are transmitted to the onboard G1 computer, where a task-space
cartesian impedance controller generates compliant end-effector
trajectories. These desired virtual poses are then converted into
joint targets through inverse kinematics and executed using the
robot’s built-in position controllers. Fig. 1 provides an overview of
the perception, reasoning, and control pipeline.
3.1 VLM–RAG System
The VLM–RAG system processes the ego-centric view using the
Molmo-7B-O BnB 4-bit model [ 1] to identify the task-relevant ob-
jects and scene context. The visual output is converted into a multi-
modal representation and embedded using the all-MiniLM-L6-v2
sentence-transformer [ 2], enabling semantically meaningful re-
trieval within a shared vector space. Given an input image, the
VLM determines the task through sequential yes/no queries. The
RAG system then retrieves the corresponding impedance and grip-
per parameters, providing the controller with context-dependent
gains for safe and adaptive manipulation.
3.1.1 Custom Database for Environmental Scenarios.The database
contains nine manipulation tasks, each associated with experimen-
tally validated cartesian impedance parameters and an optimal grip-
per configuration. Impedance parameters include stiffness coeffi-
cients𝐾=[𝐾𝑥,𝐾𝑦,𝐾𝑧]and damping coefficients 𝐷=[𝐷𝑥,𝐷𝑦,𝐷𝑧],
which determine how the end-effector regulates contact forces and
compliance along the cartesian axes. Each entry also specifies a
preferred gripper angle 𝛾𝑎for secure manipulation of the corre-
sponding object type.
The retrieval system relies on two JSON databases. The impedance
database stores task-specific cartesian impedance parameters col-
lected from real-world experiments on the G1 humanoid. Multiple
settings were tested per task, and the most stable and compliant
configuration was recorded as the canonical entry. The gripper data-
base contains experimentally determined optimal closing angles
for manipulating rigid, soft, deformable, or fragile objects. These
angles were obtained through repeated trials and represent secure
yet compliant grasp strategies.
Together, these two databases map a semantic task label to its
corresponding control variables, namely the cartesian impedance
parameters and the gripper configuration, forming the knowledge
base of the RAG module.

HumanoidVLM: Vision–Language–Guided Impedance Control for Contact-Rich Humanoid Manipulation
3.2 Robot Control
3.2.1 End-effector Impedance Control.The Unitree G1 does not
provide wrist force–torque sensing, and its two-finger grippers lack
reaction wrench feedback. To enable compliant interaction without
force sensors, we adopt a homogeneous cartesian mass–spring–
damper model in task space. Each end-effector is regulated as a
6-DoF point following a desired pose trajectory, which acts as a
virtual reference for the impedance dynamics.
The translational impedance parametersKandDare selected
by the VLM–RAG system, while the virtual massMand rotational
impedance remain fixed for stability. For each arm 𝑎∈{𝐿,𝑅} , let
x𝑎∈R3andR𝑎∈𝑆𝑂( 3)denote the current end-effector position
and orientation, and letx 𝑎,ref,R𝑎,refdenote the desired reference
pose. We define diagonal virtual mass, damping, and stiffness ma-
trices:
M𝑎=diag(𝑀 𝑎,𝑥,𝑀𝑎,𝑦,𝑀𝑎,𝑧),
D𝑎=diag(𝐷𝑎,𝑥,𝐷𝑎,𝑦,𝐷𝑎,𝑧),
K𝑎=diag(𝐾𝑎,𝑥,𝐾𝑎,𝑦,𝐾𝑎,𝑧).(1)
For each arm 𝑎, the translational error is defined as in (2), the
impedance dynamics follow the second order ODE (3), and the
resulting virtual force is given by (4).
e𝑎=x𝑎,ref−x𝑎,¤e𝑎=¤x𝑎,ref−¤x𝑎 (2)
M𝑎¥e𝑎+D𝑎¤e𝑎+K𝑎e𝑎=0(3)
Fvirt
𝑎=K𝑎e𝑎+D𝑎¤e𝑎 (4)
captures the interaction response, acting as a quantitative proxy
for physical contact forces. If no disturbance exists, both pose error
and the corresponding virtual forces converge to zero.
3.2.2 Gripper Configuration.Each hand uses a single-DoF gripper.
The controller receives a discrete action: 𝛾𝑎∈{open,close} , where
each action corresponds to a predefined joint angle target retrieved
from the database. Gripper angles are retrieved from the VLM–RAG
database based on the inferred task and object type, complementing
the impedance parameters selected for the scenario.
4 System Evaluation
Experiments were conducted on the Unitree G1 humanoid robot
equipped with two 1-DoF grippers and an Intel RealSense RGB-D
camera mounted in the head for ego-centric perception. The task-
space impedance controller ran on the on-board PC at 50 Hz, while
the VLM–RAG pipeline was executed on an external workstation
with an RTX 4090 GPU and Intel i9-13900K CPU to ensure real-time
inference.
4.1 Experimental Setup
Evaluation focused on tabletop tasks dominated by normal-direction
interaction along the 𝑧-axis. Five representative task groups were
considered: (1) Following an irregular surface by maintaining smooth
right-hand contact, (2) Applying controlled 𝑧-direction forces while
holding a massage ball, (3) Bimanual placement of two objects: a
sauce bottle (left) and an egg (right), (4) Tool interaction using a
fork to poke a fruit or vegetable, and (5) Grasping and lifting various
tabletop objects.
Each scenario was executed with multiple impedance and gripper
settings to empirically determine the optimal parameters stored in
the database. Fig. 2 shows examples from successful trials.
Figure 2: Images from successful trials of the representative
manipulation scenarios.
4.1.1 VLM–RAG Retrieval Evaluation.The evaluation tests the dis-
crete retrieval correctness of impedance and gripper parameters
from a single ego-centric image. A retrieval is considered correct
for evaluation purposes when the system: (1) classifies the task cor-
rectly via sequential visual yes/no queries, (2) retrieves the correct
impedance entry based on the VLM output, and (3) retrieves the
correct gripper configuration conditioned on both the VLM task
label and the impedance scenario.
4.1.2 Retrieval Process.To assess semantic generalization across
tasks, 14 ego-centric test images were collected. These depict the
nine task types but vary in camera viewpoint, object placement, and
arm pose, ensuring robustness evaluation rather than memoriza-
tion. For each image, the VLM infers the task through hierarchical
visual queries. The resulting task label is used as a FAISS query
to retrieve the impedance scenario. The scenario description is
then concatenated with the VLM label to form a second query for
retrieving the appropriate gripper configuration. This two-stage
retrieval disambiguates visually similar tasks differing in object
type or interaction mode.
4.1.3 Impedance Control Evaluation.To confirm that retrieved pa-
rameters yield stable and bounded interaction behavior, we analyze
𝑧-axis tracking errors and virtual force magnitudes during execu-
tion. Single-object tasks use the right arm; dual-object placement
uses both arms. The evaluation checks whether the selected 𝐾𝑧and
𝐷𝑧values achieve the expected compliance and whether position
errors and virtual forces remain bounded and task-appropriate.
4.2 Results
4.2.1 VLM–RAG System.The VLM–RAG pipeline was evaluated
on 14 ego-centric test images representing variations of the nine
tasks stored in the database (Sec. 4.1.1). The system retrieved the
correct impedance and gripper parameters in 13 out of 14 cases,
resulting in an accuracy of 93% as can be seen in Fig. 3. The single
failure occurred when the primary object was partially occluded,
highlighting a limitation of the current vision-based task inference.
4.2.2 Control Evaluation Results.Table 1 summarizes the retrieved
parameters and corresponding performance metrics. For each sce-
nario, we report the selected normal-direction gains 𝐾𝑧,𝐷𝑧, the

Yara Mahmoud, Yasheerah Yaqoot, Miguel Altamirano Cabrera, and Dzmitry Tsetserukou
Figure 3: Retrieval accuracy of the VLM–RAG system across
14 scenarios.
mean and maximum absolute position error in 𝑧, and the maxi-
mum virtual normal force magnitude. Across all scenarios, 𝑧-axis
tracking errors remained small, and the computed virtual force
magnitudes scaled consistently with the selected impedance gains.
This confirms that the retrieved parameters produce stable and
task-appropriate interaction behavior.
Table 1: Task-space impedance evaluation for representative
tabletop tasks.
Task𝐾 𝑧𝐷𝑧|𝑒𝑧|max|𝑒 𝑧|max|𝐹virt
𝑧|
[N/m] [Ns/m] [m] [m] [arb.]
Follow surface (R) 3.0 2.0 0.016 0.024 0.103
apply pressure (R) 5.0 3.0 0.017 0.035 0.334
Dual placement (R) 2.0 1.0 0.013 0.034 0.089
Dual placement (L) 6.0 1.5 0.009 0.013 0.176
Tool interaction (R) 2.0 1.5 0.006 0.016 0.183
Grasp from table (R) 4.0 1.5 0.013 0.022 0.153
Surface followingSoft stiffness enabled compliant tracking
of curved geometry, with low virtual forces and small errors. As
seen in Fig. 4, the robot tracks the curved object with compliant
contact. Fig. 4 shows that 𝑧error follow the surface profile, while
𝑥deviations stem from orientation adjustments of the arm.Mas-
sage and pressure application:Higher stiffness and damping
resulted in larger virtual forces while maintaining similar tracking
accuracy, demonstrating effective modulation of contact intensity.
Dual-object placement:Asymmetric stiffness values reflected
object fragility. The egg-handling arm used soft settings, while
the bottle-handling arm used stiffer gains, producing correspond-
ingly higher force responses.Tool interaction:Moderate stiffness
and damping supported controlled poking motions with low track-
ing error and transient force peaks appropriate for brief contact
events.Grasp-and-lift:Intermediate stiffness ensured stable lifting
without excessive pressure, with tracking errors remaining below
2.5 cm.
The controller behaved consistently, with low stiffness enabling
compliant contact, high stiffness supporting forceful interaction,
and asymmetric bimanual settings matching object-specific needs.
The clear link between impedance parameters and virtual force
responses indicates that these virtual forces may serve as effective
proxies for real contact forces.
Figure 4: Translational errors in the surface-following task,
showing desired and measured end-effector positions using
forward kinematics.
5 Conclusion and Future Work
This paper presented HumanoidVLM, a vision–language–driven
retrieval framework that links semantic scene understanding with
task-specific cartesian impedance and gripper control for the Uni-
tree G1 humanoid. By combining VLM-based task inference with a
FAISS-based RAG module, the system autonomously selects stiff-
ness, damping, and grasp configurations directly from an ego-
centric image using two small human-validated databases contain-
ing nine manipulation tasks and nine object-specific grasp entries.
Experiments across manipulation tasks showed that the framework
retrieves correct parameters in 93% of the evaluated scenarios and
enables stable, compliant execution in real-world trials.
Given the limited evaluation scale and task diversity, the cur-
rent results should be interpreted as a proof of feasibility rather
than a comprehensive robustness assessment. In future work, we
aim to incorporate task-dependentrotationalimpedance to enable
orientation-sensitive behaviors. Second, substituting the discrete
database with a learned continuous mapping could allow interpo-
lation across unseen tasks and object types. Finally, integrating
force or visuotactile feedback would allow closed-loop impedance
adaptation, improving safety and robustness.
Acknowledgements
Research reported in this publication was financially supported by
the RSF grant No. 24-41-02039.
References
[1]2025.Molmo-7B-O BnB 4bit quantized 7GB. https://huggingface.co/cyan2k/
molmo-7B-O-bnb-4bit
[2] 2025.Sentence Transformer: all-MiniLM-L6-v2. https://huggingface.co/sentence-
transformers/all-MiniLM-L6-v2
[3]Fares J. Abu-Dakka and Matteo Saveriano. 2020. Variable Impedance Control
and Learning—A Review.Frontiers in Robotics and AI7 (2020), 590681. doi:10.
3389/frobt.2020.590681

HumanoidVLM: Vision–Language–Guided Impedance Control for Contact-Rich Humanoid Manipulation
[4]Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes,
Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol
Hausman, et al .2023. Do As I Can, Not As I Say: Grounding Language in Robotic
Affordances.Proc. of the Conf. on Robot Learning205 (2023).
[5]Arash Ajoudani, Andrea Maria Zanchettin, Serena Ivaldi, Alin Albu-Schäffer,
Kazuhiro Kosuge, and Oussama Khatib. 2018. Progress and Prospects of the
Human–Robot Collaboration.Autonomous Robots42, 5 (2018), 957–975. doi:10.
1007/s10514-017-9677-2
[6] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana
Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al .
2022. Flamingo: a Visual Language Model for Few-Shot Learning. InAdvances
in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal,
D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 23716–
23736.
[7]Fatima Batool, Muhammad Zafar, Yasheerah Yaqoot, R. A. Khan, M. H. Khan,
Alexey Fedoseev, and Dzmitry Tsetserukou. 2025. ImpedanceGPT: VLM-driven
Impedance Control of Swarm of Mini-drones for Intelligent Navigation in Dy-
namic Environment. InProc. of the IEEE/RSJ Int. Conf. on Intelligent Robots and
Systems (IROS). Hangzhou, China, 2592–2597.
[8] A. Bicchi and V. Kumar. 2000. Robotic grasping and contact: a review. InProc. IEEE
Int. Conf. on Robotics and Automation. Symposia Proceedings (Cat. No.00CH37065),
Vol. 1. 348–353 vol.1. doi:10.1109/ROBOT.2000.844081
[9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis,
Chelsea Finn, et al .2023.RT-1: Robotics Transformer for Real-World Control at
Scale. arXiv:2212.06817 Retrieved from https://arxiv.org/abs/2212.06817.
[10] Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha
Majumdar, and Dorsa Sadigh. 2024. Physically Grounded Vision–Language
Models for Robotic Manipulation.Proc. of the IEEE Int. Conf. on Robotics and Au-
tomation (ICRA)2024 (2024), 12462–12469. doi:10.1109/ICRA57147.2024.10610090
[11] K. Fidele Gbagbe, Miguel Altamirano Cabrera, Ahmed Alabbas, Omar Alyounes,
Alexey Lykov, and Dzmitry Tsetserukou. 2024. Bi-VLA: Vision-Language-Action
Model-Based System for Bimanual Robotic Dexterous Manipulations. InProc. ofthe IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC). Sarawak, Malaysia,
2864–2869.
[12] Neville Hogan. 1985. Impedance Control: An Approach to Manipulation: Part
I—Theory.Journal of Dynamic Systems, Measurement, and Control107, 1 (1985),
1–7. doi:10.1115/1.3140702
[13] Hyeongyo Jeong, Haechan Lee, Changwon Kim, and Sungtae Shin. 2024. A
Survey of Robot Intelligence with Large Language Models.Applied Sciences14,
19 (2024), 8868. doi:10.3390/app14198868
[14] Yuchen Ji, Zequn Zhang, Dunbing Tang, Yi Zheng, Changchun Liu, Zhen Zhao,
and Xinghui Li. 2024. Foundation Models Assist in Human–Robot Collaboration
Assembly.Scientific Reports14 (2024), 24828. doi:10.1038/s41598-024-75715-4
[15] Dingzhe Li, Yixiang Jin, Yuhao Sun, Yong A, Hongze Yu, Jun Shi, Xiaoshuai Hao,
Peng Hao, Huaping Liu, Fuchun Sun, Jianwei Zhang, and Bin Fang. 2025. What
Foundation Models Can Bring for Robot Learning in Manipulation: A Survey.
The International Journal of Robotics Research36 (2025), 261–268. doi:10.1177/
02783649251390579
[16] Dzmitry Tsetserukou, Naoki Kawakami, and Susumu Tachi. 2008. Obstacle
avoidance control of humanoid robot arm through tactile interaction. InProc. of
the IEEE-RAS Int. Conf. on Humanoid Robots. Daejeon, Korea, 379–384.
[17] Muhammad Zafar, R. A. Khan, Fatima Batool, Yasheerah Yaqoot, Zhi Guo, Mikhail
Litvinov, Alexey Fedoseev, and Dzmitry Tsetserukou. 2025. SwarmVLM: VLM-
Guided Impedance Control for Autonomous Navigation of Heterogeneous Robots
in Dynamic Warehousing. InProc. of the IEEE Int. Conf. on Robotics and Biomimet-
ics (ROBIO). Chengdu, China, 770–775.
[18] Heng Zhang, Wei-Hsing Huang, Gokhan Solak, and Arash Ajoudani. 2025.Om-
niVIC: A Self-Improving Variable Impedance Controller with Vision–Language
In-Context Learning for Safe Robotic Manipulation. doi:10.48550/arXiv.2510.17150
Retrieved from https://arxiv.org/abs/2510.17150.
[19] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin
Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al .2023. RT-2: Vision–
Language–Action Models Transfer Web Knowledge to Robotic Control.Proc. of
the 7th Conf.on Robot Learning229 (2023), 2165–2183.