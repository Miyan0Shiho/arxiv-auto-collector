# OmniVIC: A Self-Improving Variable Impedance Controller with Vision-Language In-Context Learning for Safe Robotic Manipulation

**Authors**: Heng Zhang, Wei-Hsing Huang, Gokhan Solak, Arash Ajoudani

**Published**: 2025-10-20 04:54:22

**PDF URL**: [http://arxiv.org/pdf/2510.17150v2](http://arxiv.org/pdf/2510.17150v2)

## Abstract
We present OmniVIC, a universal variable impedance controller (VIC) enhanced
by a vision language model (VLM), which improves safety and adaptation in any
contact-rich robotic manipulation task to enhance safe physical interaction.
Traditional VIC have shown advantages when the robot physically interacts with
the environment, but lack generalization in unseen, complex, and unstructured
safe interactions in universal task scenarios involving contact or uncertainty.
To this end, the proposed OmniVIC interprets task context derived reasoning
from images and natural language and generates adaptive impedance parameters
for a VIC controller. Specifically, the core of OmniVIC is a self-improving
Retrieval-Augmented Generation(RAG) and in-context learning (ICL), where RAG
retrieves relevant prior experiences from a structured memory bank to inform
the controller about similar past tasks, and ICL leverages these retrieved
examples and the prompt of current task to query the VLM for generating
context-aware and adaptive impedance parameters for the current manipulation
scenario. Therefore, a self-improved RAG and ICL guarantee OmniVIC works in
universal task scenarios. The impedance parameter regulation is further
informed by real-time force/torque feedback to ensure interaction forces remain
within safe thresholds. We demonstrate that our method outperforms baselines on
a suite of complex contact-rich tasks, both in simulation and on real-world
robotic tasks, with improved success rates and reduced force violations.
OmniVIC takes a step towards bridging high-level semantic reasoning and
low-level compliant control, enabling safer and more generalizable
manipulation. Overall, the average success rate increases from 27% (baseline)
to 61.4% (OmniVIC).

## Full Text


<!-- PDF content starts -->

OmniVIC: A Self-Improving Variable Impedance Controller with
Vision-Language In-Context Learning for Safe Robotic Manipulation
Heng Zhang*1, Wei-Hsing Huang*2, Gokhan Solak1, Arash Ajoudani1
1Human-Robot Interfaces and Interaction Lab, Istituto Italiano di Tecnologia, Genoa, Italy.
2Georgia Institute of Technology, Atlanta, USA.
*These two authors contributed equally to this work
Abstract—We present OmniVIC, a universal variable
impedance controller (VIC) enhanced by a vision language
model (VLM), which improves safety and adaptation in any
contact-rich robotic manipulation task to enhance safe physical
interaction. Traditional VIC have shown advantages when the
robot physically interacts with the environment, but lack gener-
alization in unseen, complex, and unstructured safe interactions
in universal task scenarios involving contact or uncertainty. To
this end, the proposed OmniVIC interprets task context—derived
reasoning from images and natural language and generates
adaptive impedance parameters for a VIC controller. Specifically,
the core of OmniVIC is a self-improving Retrieval-Augmented
Generation(RAG) and in-context learning (ICL), where RAG
retrieves relevant prior experiences from a structured memory
bank to inform the controller about similar past tasks, and ICL
leverages these retrieved examples and the prompt of current task
to query the VLM for generating context-aware and adaptive
impedance parameters for the current manipulation scenario.
Therefore, a self-improved RAG and ICL guarantee OmniVIC
works in universal task scenarios. OmniVIC facilitates zero-shot
adaptation to a broad spectrum of previously unseen scenarios,
even when the inputs of VLM are not well-represented. The
impedance parameter regulation is further informed by real-time
force/torque feedback to ensure interaction forces remain within
safe thresholds. We demonstrate that our method outperforms
baselines on a suite of complex contact-rich tasks, both in
simulation and on real-world robotic tasks, with improved success
rates and reduced force violations. OmniVIC takes a step towards
bridging high-level semantic reasoning and low-level compliant
control, enabling safer and more generalizable manipulation.
Overall, the average success rate increases from 27% (baseline)
to 61.4% (OmniVIC). Code, video and RAG dataset are available
at https://sites.google.com/view/omni-vic
I. INTRODUCTION
Robotic manipulation in contact-rich environments presents
significant challenges, particularly in ensuring safe and ef-
fective interactions with objects and the environment [1],
[2]. Tasks such as wiping a surface, inserting a plug into a
socket, or pushing a drawer require not only precise motion
planning but also adaptive compliance to handle uncertainties
and variations in contact dynamics. While the traditional
variable impedance control (VIC) can improve safety and
adaptability in specific scenarios by modulating the robot’s
stiffness and damping to achieve compliance, they typically
require manual tuning of parameters and do not generalize
well across different tasks or environments [3]. Even learning-
based VIC approaches [4] often struggle with generalization
This work was supported by the Horizon Europe Project TORNADO (GA
101189557).due to the limited scope of training data and the complexity
of contact dynamics.
The emergence of VLMs has opened new avenues for
enhancing robotic manipulation by providing rich semantic un-
derstanding and reasoning capabilities [5], [6]. By leveraging
the contextual information provided by VLMs, robots can bet-
ter understand the nuances of a task, such as the required level
of force or compliance, and adjust their behavior accordingly.
However, most existing VLM-based approaches focus on high-
level task planning or trajectory generation, often neglecting
the critical aspect of force-aware compliant control in contact-
rich scenarios. This gap limits the effectiveness in ensuring
safe and adaptive physical interactions, as they do not directly
inform the low-level control strategies needed for such tasks.
While VLMs alone struggle to inform compliant control,
retrieval-augmented generation (RAG) can fill this gap by
recalling relevant past experiences, which has shown promise
in enhancing model capabilities by allowing access to external
knowledge bases at inference time [7]. In robotics, RAG
can enable robots to retrieve relevant prior experiences or
demonstrations that inform their current actions, improving
adaptability and generalization [8]. By integrating RAG with
VLMs, robots can leverage both semantic understanding and
experiential knowledge to better handle the complexities of
contact-rich manipulation tasks.
Moreover, ICL enables robots to adapt behaviors from
contextual examples without retraining [9]. Originally devel-
oped for language models, ICL allows models to perform
new tasks by conditioning on demonstrations provided in the
input prompt. In robotics, this concept has been extended
to enable rapid task adaptation through visual prompts [10],
[11], language instructions [12], or recent interaction experi-
ences [13]. However, existing ICL approaches in robotics often
focus on visual or trajectory demonstrations alone, without
considering the physical interaction forces that are crucial for
safe manipulation.
To address these challenges, we propose OmniVIC, a uni-
versal variable impedance controller enhanced by a VLM and
empowered by RAG and ICL. OmniVIC integrates high-level
semantic reasoning from multimodal inputs with low-level
compliant control, enabling robots to adapt their impedance
parameters for safe and effective manipulation in contact-rich
scenarios. The key contributions of this work are:
•We introduce a new paradigm of VIC that enables
context-aware impedance adaptation by context-awarearXiv:2510.17150v2  [cs.RO]  22 Oct 2025

VLM for safe physical interaction. OmniVIC allows
robots to leverage semantic understanding and common-
sense reasoning capabilities to inform low-level control
strategies.
•We develop a self-improving RAG mechanism that re-
trieves relevant prior experiences from a structured mem-
ory bank, facilitating the impedance adaptation based on
similar past tasks. This approach can locally enhance gen-
eralization and adaptability across diverse manipulation
scenarios.
•We implement an ICL strategy that conditions the VLM
on retrieved examples and current task context, allowing
for dynamic generation of impedance parameters without
retraining. This enables rapid adaptation to new tasks and
environments.
•A suite of contact-rich manipulation tasks was conducted,
demonstrating the effectiveness of OmniVIC in both
simulation and real-world experiments. Our results show
that the proposed method not only enhances task success
rates but also reduces force violations, enabling safer
manipulation.
II. RELATEDWORK
A. Variable impedance control for physical tasks
VIC has been widely recognized as a fundamental approach
for enabling safe and adaptive robot behavior in contact-rich
tasks [2]. Early works demonstrated that learning variable
stiffness improves robustness and safety compared to fixed-
gain control [14] and that humans naturally adapt impedance
to maintain stability in both stable and unstable interactions
[15], providing strong biological motivation. More recently,
reinforcement learning has leveraged VIC as an action space,
improving performance in complex manipulation tasks [16].
Furthermore, it is shown to improve the safety in training and
deployment in learning-based approaches [4].
B. Retrieval-Augmented generation for robotics
RAG augments parametric models with non-parametric
memory, enabling inference-time access to external knowl-
edge [7]. In robotics, this allows agents to query prior em-
bodied experience at run time rather than relying solely on
retraining [8], [17].
Robotic RAG differs from text-only settings: retrieval must
account for evolving spatio-temporal context [18] and often
span multiple embodied modalities (e.g., vision and robot
state signals) [8]. Behavior Retrieval, in turn, shows that
querying datasets for task-relevant behaviors improves perfor-
mance [19] [20].
To the best of our knowledge, no previous studies have
applied RAG to VIC in the context of contact-rich manip-
ulation. To address this research gap, we propose a novel
RAG-based approach for variable impedance control that
uses prior experience without retraining while maintaining
experience-informed compliance; as the database grows, the
robot improves its regulation of impedance parameters for
novel contact-rich scenarios.C. In-context learning
Recent works have explored various approaches to im-
plement ICL in robotic systems. Zhang et al. [9] leverage
ICL for dynamics adaptation, while InCoro [10] uses in-
context learning for robot control with visual prompts. Other
notable works include manipulate-anything [21] which con-
ditions policies on demonstration videos, and MOKA [11]
which combines ICL with meta-learning for rapid adapta-
tion. VLM-based approaches [12], [22] have shown promise
in grounding language instructions through visual context,
while [13] explores physically-grounded ICL for manipulation
tasks. Existing methods largely focus on high-level planning
or trajectory generation, overlooking force-aware compliant
control in contact-rich tasks.
Our method uses force-aware ICL with retrieval-augmented
examples for impedance control. Then VLM infers task-
relevant stiffness and damping, grounding semantic under-
standing in physical interaction for safer, more adaptive ma-
nipulation.
D. Leveraging VLMs for VIC controller
Recent work has explored interpreting task context and
generating adaptive impedance parameters by VLM, but often
focuses on specific task domains or relies on simplified models
that do not fully leverage the capabilities of modern VLMs [5],
[6], [23], [24].
Existing works can be categorized into two main ap-
proaches. First, some studies tackle contact-rich tasks using
impedance controllers [5], [6] but without VLM enhancement,
where the vision-language component only assists in high-
level task planning rather than real-time impedance adaptation.
Second, works such as [23] have integrated VLMs to im-
prove VIC parameter selection, but their evaluation is limited
to simple scenarios but non-contact-rich manipulation tasks.
Additionally, some recent efforts introduce extra modalities to
enhance physical interaction performance, for instance, [24]
proposes a VLM-driven teleimpedance control framework that
utilizes eye tracking, while [25] incorporates tactile informa-
tion for improved contact sensing.
In contrast, our work uses in-context learning to adapt
parameters based on task-specific context, integrates real-time
force/torque feedback to ensure safe physical interactions.
III. METHODOLOGY
In this section, we describe the overall architecture and key
components of OmniVIC (Fig. 1), a universal VIC empowered
by RAG+ICL and assisted by VLM. This integrates high-level
semantic reasoning from multimodal inputs with low-level
compliant control, enabling robots to adapt their impedance
parameters for safe and effective manipulation in contact-rich
scenarios.
We detail the mechanism of VLM-informed parameter
generation for the classical variable impedance control, the
RAG process for leveraging prior experiences, the use of ICL
for adaptive gain synthesis, and the real-time force/torque
feedback regulation that ensures safe physical interaction.

Fig. 1. Overview of the proposed OmniVIC. The system integrates a
VLM with a VIC to enable safe and adaptive manipulation in contact-rich
tasks. The VLM processes multimodal inputs, including visual observations,
natural language instructions, and real-time force/torque feedback, to generate
context-aware impedance parameters (stiffness and damping) for the VIC.
Implementation details are provided to facilitate reproducibil-
ity and highlight the practical considerations of deploying
OmniVIC in both simulated and real-world environments.
A. VLM-informed VIC
A classical Cartesian VIC modulates the robot’s stiffness
Kand dampingDto achieve compliant interactions with the
environment. The control law is typically defined in task space
as:
Fext=D ˙˜x+K ˜x,(1)
where ˜xis the Cartesian pose error between the desired
and actual posex d,x∈R6. Accordingly, ˙˜xis the ve-
locity error between the desired and actual end-effector
velocity, ˙xd,˙x∈R6, respectively. The desired stiffness
matrixK∈R6×6is a diagonal matrix with variable
terms asK=diag{K x, Ky, Kz, ϵK x, ϵK y, ϵK z}, where the
ϵis coefficient for rotation elements to reduce and sim-
plify the parameters tuning. The desired damping matrix
D∈R6×6is also diagonal matrix with variable terms as
D=diag{D x, Dy, Dz, ζD x, ζD y, ζD z}, whereζis the coef-
ficient for rotation elements.
The key to effective VIC lies in the appropriate selection of
the impedance parametersKandD. Traditional approaches
often rely on manual tuning or predefined schedules, which
may not generalize well across different tasks or environments.
In our method, the VLM processes visual and language
inputs to derive the task context, which is then used to generate
adaptive impedance parameters for the VIC controller. To
simplify the problem, the VLM outputs diagonal translational
stiffness and translational damping elements, reducing the
parameter space to three stiffness values(K x, Ky, Kz)and
three damping values(D x, Dy, Dz).To enhance context-aware VLM parameter generation, we
employ a RAG mechanism to fetch relevant prior experiences
from a structured memory bank, see Sec. III-B and III-C.
These experiences inform the VLM about similar past tasks,
improving its ability to generate appropriate impedance pa-
rameters for the current scenario. Then, we use ICL (Sec.
III-D) to condition the VLM on these retrieved examples and
the current task context, allowing for dynamic generation of
impedance parameters without retraining to alleviate heavy
computation. Finally, we incorporate real-time force/torque
feedback into the VIC controller to ensure that interaction
forces remain within safe thresholds, further enhancing safety
during manipulation.
B. RAG data acquisition and storage
RAG data are acquired by deploying any robot policy and
applying two success criteria: a maximum allowable contact
forceF maxand a time budgetT max. A trial is labeledfailure
if the measured force exceedsF max at any time or if the
task time exceedsT max; otherwise, it is labeledsuccess. To
fully automate logging, we employ a lightweight VLM that
processes the instructionTtogether with the current twist and
wrench and returns the controller gains(K, D).
For eachsuccessfulexecution, we append the following to
the RAG database: (i) the instructionT, (iii) the end–effector
twistv t∈R6expressed in the world frame, (iv) the grav-
ity–compensated wrenchw t= [F t, τt]∈R6expressed in the
world frame, (ii) the phase labelp t, and (v) the VLM-predicted
controller impedance parameters(K, D). Items (i)–(iv) are
detailed below.
InstructionWe store the instruction in two complemen-
tary forms: natural-language instruction text, denotedT text
(used later for in-context learning; see Section III-D), and
a precomputed sentence embedding, denotedT emb(used for
retrieval; see Section III-C). The embedding is produced once
at collection time by a text encoder and persisted alongside
the raw text so that retrieval does not require re-encoding.
Throughout the paper,T textrefers to the raw instruction string
andT embrefers to its embedding.
Twist & wrenchUnless otherwise stated, the 6D twistv t
and the 6D wrenchw tare expressed in the world frame via the
standardSE(3)adjoint frame transformation; the wrench is
gravity-compensated prior to storage to isolate external/contact
forces.
Phase labelBy querying a VLM with the instruction
T, the current imageI tcaptured by an overview
camera, and the world-frame twistv tand the
world-frame wrenchw t. the model outputs one of
{Free motion,Approaching,Contact,Retreat}as the phase
label at each step.
When a new successful record arrives: (i) if the bank is not
full, we add it; (ii) if the bank is full, we temporarily pool
the new record with the records belonging to this instruction.
compute pairwise similarities using the method defined in
Sec. III-C, identify the closest pair (i.e., the two most similar
records), and randomly discard one of the two. The remaining

records are kept. This closest-pair replacement prevents near-
duplicate motions from accumulating and preserves diversity
under a fixed memory budget. The bank’s update mechanism
enables the RAG to improve its performance with continuous
use.
C. RAG retrieval
RAG retrieval process that identifies the most relevant prior
experiences from the database to inform impedance parameter
generation for the current request. Given a new request,
pre-processing is in a manner consistent with the storage
pipeline: (i) the instruction raw stringT text(for in-context
learning Sec. III-D) and its embeddingx embcomputed with
the same text encoder used in Sec. III-B; (ii) the phase label
predicted by the VLM usingx text, the current imageI t, and
the proprioceptive signals (twistv tand wrenchw t); No re-
encoding of database instructions is required because their
embeddings were precomputed during collection.
Aiming to reduce the search space and improve retrieval
efficiency, the pre-processed items are then used in a four-step
progressive retrieval process:
1) Step 1: instruction filtering:We compare the query’s in-
struction embeddingT embto all stored instruction embeddings
using cosine similarity, and retrieve the top-M%most similar
instructions by this score.
2) Step 2: phase filtering:As each instruction in the RAG
store is partitioned into four phase-specific records, we refine
the results of Step 1 (Sec. III-C1) by keeping only those
records whose phase matches the query’s phase for each of
the top-M%instructions. The phase-consistent candidate pool
is then used in Sec. III-C3 for similarity computation.
3) Step 3: similarity computation:We compute four
modality-specific similarity scores between the query and each
candidate record in the pool from Step 2 (Sec. III-C2): (1)
force similarity, (2) torque similarity, (3) linear velocity simi-
larity, and (4) angular velocity similarity. All four similarities
use the cosine similarity, which returns a score in[−1,1], as
defined below:
cossim(a, b) =a·b
∥a∥∥b∥(2)
whereaandbare the two vectors being compared, andnis
their dimensionality.
4) Step 4: final score:Although wrenches and twists rep-
resent different physical quantities, in Sec. III-C3 we compute
four modality-specificsimilarityscores: force, torque, linear
velocity, and angular velocity, each using the same well-
known cosine similarity. Because these are similarity scores,
they naturally lie on the common symmetric range[−1,1],
making them directly comparable across modalities. We then
sum the four similarity scores to obtain a single aggregate
measure for each candidate, and keep the top five highest-
scoring candidates as exemplars for in-context learning.
After completing Steps 1–4, all remaining examples are
used as in-context exemplars. For each retained examples, we
extract the instruction text (T text), the phase label, the controllergains(K, D), and the associated motion signals (world-frame
twist and gravity-compensated world-frame wrench), and the
overall similarity score from Sec. III-C4. These items are then
formatted as in-context examples and concatenated with the
current request to form the prompt for the VLM. By leveraging
RAG, we ensure that the VLM is informed by relevant prior
experiences, enhancing its ability to generate context-aware
impedance parameters for the current task.
D. ICL-enhanced impedance parameters prediction
Given a new request, we run RAG retrieval to obtain the
top–Nmost similar records. We then provide the top five sim-
ilar records to the VLM as examples. Each example includes
the instruction text (T text), phase label, controller gains(K, D),
world-frame twist and wrench, and the similarity score. The
VLM outputs(K, D)for the current step using the prompt
(simplified, see detailed in our code) below.
Simplified Prompt for OmniVIC impedance parameter
dynamic generation:
You are an expert robotic impedance
controller capable of analyzing visual
scenes and physical interaction states.
Given the{instruction},{current phase}
,{twist}, and{wrench},{impedance
range}, determine optimal impedance
parameters.
Apply phase-based impedance principles:
•Highest for free motion (precise
control)
•Lowest for contact (maximum
compliance)
Consider motion direction adaptation:
•Increase stiffness in the primary
motion direction when overcoming
resistance.
•Decrease stiffness in the primary
motion direction when maintaining
accuracy.
Reference similar successful example
with{similarity score}:
•Task:{example instruction}
•Phase:{example phase}
•Twist:{example twist}
•Wrench:{example wrench}
•Parameters:
K={example K},D={example D}
Output:K= [K x, Ky, Kz],D= [D x, Dy, Dz]
IV. EXPERIMENTS
We evaluate OmniVIC on a suite of contact-rich manipula-
tion tasks in both simulation and real-robot experiments. The
evaluation metrics include task success rates, force violations,
and overall interaction safety. Through these experiments, we
aim to answer the following research questions:
•Does OmniVIC improve task success rates in physical
manipulation tasks compared to baselines?

Task 1 Task 2 Task 3 Task 4 Task 5
Task 6 Task 7 Task 8 Task 9Task 10
Fig. 2. Tasks used in simulation experiments.
•How effectively does OmniVIC reduce force violations
and enhance safety during manipulation?
•What is the impact of RAG and ICL on the adaptabil-
ity and generalization of impedance parameters across
diverse tasks?
A. Simulation experiments
1) Baselines and Evaluation Tasks:Baselines: As Om-
niVIC is a general-purpose variable impedance controller
rather than a specific policy, we adapt robot policies from
a pretrained VLA model Pi0 [26] to execute tasks. The
default controller used in Pi0 is our baseline, which is a
low-level position controller. Where the default parameters
are fixed transitional stiffnessk p= 150N/m, and damping
Kd= 2p
kp= 24.494Ns/m. We compare the baseline with
OmniVIC as our method.
Evaluation task and protocol: We select 20 related ma-
nipulation tasks from LIBERO benchmark, 10 fromLIBERO-
Objectand 10 fromLIBERO-Goal. By running the original
benchmark but under force constraint, we design the split by
stratified selection based on its outcome. The split is composed
ofQuery SetandKnowledge-Base Setto ensure evaluation
tasks inQuery Setare genuinely unseen in the retrieval
componentKnowledge-Base Set, avoiding train–test leakage
and matching zero-shot evaluation practice in information
retrieval. To do so, we stratify the original outcome into bands
so that tasks are represented evenly. From these bands, we
form two disjoint sets of equal size: aKnowledge-Base Setof
10 tasks (retained solely for building the RAG store and never
used as the later query) and aQuery Setof 10 tasks (held
out solely for issuing queries). The tasks for ICL evaluation
inQuery Setwith specific details illustrated in Fig. 2 and
Table I, providing diverse manipulation scenarios with rich
interactive behaviors.
Safety/termination rule: As we care for safe contact, the
contact force is monitored during the execution. In addition to
the task-specific success criteria defined by the policy, beyond
the policy’s native success criteria, an episode is marked as
failureif the measured interaction force exceeds30 Non three
consecutive steps to avoid sensor noise spikes. We keep this
safety rule active for all methods to ensure a fair comparison.
2) RAG data construction:After stratified selection, we
collect RAG data exclusively from theKnowledge-Base Set.TABLE I
TASKS IN SIMULATION EXPERIMENTS
Task No. Descriptions
Task 1 put the bowl on top of the cabinet
Task 2 pick up the salad dressing and place it in the basket
Task 3 pick up the tomato sauce and place it in the basket
Task 4 pick up the milk and place it in the basket
Task 5 pick up the orange juice and place it in the basket
Task 6 put the wine bottle on top of the cabinet
Task 7 push the plate to the front of the stove
Task 8 put the cream cheese in the bowl
Task 9 open the middle drawer of the cabinet
Task 10 pick up the butter and place it in the basket
To fully automate the pipeline, we pair the robot policy with
a VLM-based assistance module that proposes task-dependent
control parameters(K, D)online. Duringπ 0rollouts on tasks
in the Knowledge-Base Set, at each step the VLM receives
the instruction text, the current phase label, and the previous-
step world-frame twist and gravity-compensated world-frame
wrench, and returns(K, D). If the step completes without
a force violation, we write a RAG record containing: the
instruction as raw textT text, its precomputed text-encoder
embeddingT emb(for retrieval), the phase label, the world-
frame twist, the gravity-compensated world-frame wrench, and
the controller gains(K, D). Precomputing and storing both
TtextandT embavoids re-encoding at query time and keeps
indexing consistent with querying.
The instruction embeddings are computed with the BGE-
M3 text encoder [27]. The VLM used throughout simulation
is GPT-4o-mini. The maximum capacity limit of our RAG
database is designed asB= 20K. In our experiments, we
built the RAG dataset by running 10 tasks, where 20 records
per task instruction–phase pair, resulting in a total capacity of
200 records for the evaluationn.
3) Experiments evaluation for ICL:We conduct simulation
experiments to validate the effectiveness of OmniVIC com-
pared with the baseline inQuery Set. All experiments were
performed using 4 NVIDIA RTX A6000 GPUs.
We evaluate both the baseline and our OmniVIC-enhanced
system on each of the 10 tasks in theQuery Set, running 50
episodes per task per method. In all experiments, we set M
in Step 1 (Sec. III-C1) asM= 20%; consequently, Step 1
retains the top-20% of instructions and Step 4 (Sec. III-C4)
retains the top five of the rest of the candidates for ICL.
In the ICL evaluation, the original controller in Pi0 is
the baseline, while our OmniVIC takes the top five similar
examples retrieved from RAG bank with the current task
context, and prompts together to query VLM to generate the
impedance parameters.
B. Real-world experiments
We validate our approach on a physical robot platform
to demonstrate its effectiveness in real-world contact-rich
manipulation tasks.
1) Experiment setup:We implement OmniVIC on a Franka
Emika Panda robot arm (at 1000 Hz), and set up a Logitech

020406080100120140160180
0 20 40 60 80 100 120
K_x K_y K_z05101520253035
0 20 40 60 80 100 120
D_x D_y D_z-20020406080100120
0 20 40 60 80 100 120
Baseline Ours30NK D Fz
Fig. 3. Deep dive into force and impedance profiles during task execution (Task 7 in Table I) in low-level controller. Left and middle: stiffness and damping
profiles of OmniVIC. Right: force (onlyF z, the main contact force) profiles. The impedance profiles show that OmniVIC dynamically adjusts stiffness and
damping based on task phases, reducing them during contact (middle area) to enhance compliance. The blue dots in force profiles demonstrate that OmniVIC
maintains interaction forces within safe limits, avoiding spikes that violate constrain, while it happens (yellow dots in the red box) in the baseline controller
and terminates the episode as a failure.
Baseline Ours
0%10%20%30%40%50%60%70%80%90%100%
Task-1 Task-2 Task-3 Task-4 Task-536%80%
58%92%
32%76%
50%90%
52%94%
0%10%20%30%40%50%60%70%80%
Task-6 Task-7 Task-8 Task-9 Task-100%38%
24%50%
18%68%
0%10%
0%16%
Fig. 4. Simulation results: Task success rates of baseline (π 0with position
control) vs. OmniVIC (with VLM-informed variable impedance control).
USB camera offering a global view of the robot arm and its
workspace, and a 6-axis ATI Mini45 force/torque sensor for
real-time interaction feedback. The VLM processes visual (∼
1 Hz) from the camera and natural language instructions to
generate adaptive impedance parameters (we set the stiffness
range in[300,1000]). The robot executes a series of contact-
rich manipulation tasks, including drawer closing and object
pushing, to evaluate the performance of our approach in real-
world scenarios.
2) Tasks descriptions:We set up a series of contact-rich
manipulation tasks on a physical robot. Each task is designed
to test the robot’s ability to adapt its impedance parameters
based on visual and language inputs while maintaining safe
interaction forces. 1) Close the top drawer gently; 2) Move
along the negative Y-axis while keeping the same height in the
Z-axis (one ramp); 3) The same task setup as task-2, but with
two ramps along the way. Moreover, inTask-1, we design two
comparison experiments: constant highK&Dand constant
lowK&Dto further validate the performance of OmniVIC.
In these experiments, high and low stiffness values are set to
Fig. 5. Real-world experiments: Top: Task 1: The robot gently closes the
top drawer. Middle: Task 2: The robot moves along the negative Y-axis while
maintaining the same height in Z-axis, where there is a ramp in the middle of
the path to introduce contact. Bottom: Task 3: the same as Task 2, but with
two different ramp shapes to test multiple phases of contact.
1000 and 300 N/m, respectively, while the damping is defined
as2×0.707×√
kaccordingly.
V. EXPERIMENTRESULTS
A. Simulation results and analysis
We evaluate the task success rates of OmniVIC compared
to the baseline. The results show that our approach achieves
significantly higher success rates across all contact-rich ma-
nipulation tasks, demonstrating the effectiveness of integrating
RAG-ICL enhanced VLMs with VIC for safe and compliant
interactions.
1) Force safety analysis:Fig. 3 illustrates the critical
advantage of our approach in maintaining safe interaction
forces from one episode of the taskpush the plate to
the front of the stove. Our OmniVIC dynamically
maintain contact forces within a safe threshold of 30N. In
contrast, the baseline without OmniVIC repeatedly violates
force constraints, demonstrating unsafe interaction behaviors
that could lead to task failure or equipment damage.
As detailed in Fig. 3 (right), during the initial execution
phase, both baseline and OmniVIC maintain low interaction
forces as the robot approaches the target plate to push. Then,
OmniVIC reduces stiffness to allow gentle engagement with
the target, minimizing impact forces. As the task progresses, it
increases stiffness to ensure precise control while still adhering

to force limits. This dynamic adjustment is crucial for tasks
requiring delicate handling, such as placing objects in confined
spaces or interacting with fragile items. The baseline approach,
lacking this adaptability, often applies excessive force during
these phases, leading to violations of safety constraints. It
failed at step 60 in this example.
2) Task performance evaluation:We evaluate task perfor-
mance across all eight simulation scenarios, measuring both
task success rates and force constraint violations. As shown
in Figure 4, the integration of OmniVIC yields substantial
improvements across all evaluated tasks. The baseline achieves
an average success rate of 27.0% across tasks, with frequent
force violations compromising both safety and task comple-
tion. In contrast, the OmniVIC-enhanced system achieves an
average success rate of 61.4%, a 2.27-fold improvement over
the baseline. The success of OmniVIC stems from its ability
to intelligently adapt impedance parameters based on task
context. Through the RAG component, the system retrieves
relevant prior experiences from similar manipulation scenarios,
while the ICL mechanism enables real-time parameter gener-
ation tailored to the current task state. This dual mechanism
allows for nuanced control strategies, applying higher stiffness
during precise positioning phases while reducing stiffness
during contact establishment to prevent excessive forces.
Moreover, the adaptive impedance control allows for
smoother transitions between different motion phases (Fig. 3),
minimizing abrupt force changes that could lead to task
failures or damage to objects. Consecutively, OmniVIC still
maintains success rates even when the baseline fails due to
force violations (Task7 - Task10 in Fig.4), highlighting its
robustness in handling complex contact-rich tasks.
3) Ablation study: RAG vs. RAG+ICL (OmniVIC):To
quantify the contribution of ICL on top of RAG, we compare
two configurations under identical settings:RAG-only, which
applies the retrieved parameters without ICL, andRAG+ICL
(OmniVIC), which feeds the retrieved exemplars to the VLM
for in-context learning to predict the impedance parameters.
We evaluate both cases onTasks 1–10in simulation. Across
the ten tasks,RAG+ICLachieves a1.59-foldhigher task suc-
cess rate thanRAG-only, indicating a substantial benefit from
letting the VLM adapt the retrieved priors to the instantaneous
scene and interaction state. Qualitatively, RAG-only often fails
in contact-rich phases where the retrieved examples are similar
but not fully representative of the current contact geometry or
dynamics; the VLM with ICL compensates bycontextualizing
those priors, yielding better outputs. This ablation supports our
design choice to pair RAG with ICL for robust, fine-grained
modulation of(K, D)in unseen and complex scenarios.
B. Real-world results and analysis
We validate the effectiveness of OmniVIC in three physical
interaction tasks as shown in Fig. 5. Each experiment demon-
strates the robot’s ability to adapt its impedance parameters
based on visual and language context while maintaining safe
interaction forces. We measure task success rates, force vio-
lations, and overall interaction safety during task execution.
5 10 15 20 25 30
steps1020Force MagnitudeOmniVIC (Ours)
Constant High K & D
Constant Low K & D
5 10 15 20 25 30
steps400600800k valueskx
ky
kz
5 10 15 20 25 30
steps20406080d valuesdx
dy
dzFig. 6. Task-1: The robot gently closes the top drawer using OmniVIC vari-
able impedance control. The system adapts stiffness and damping parameters
based on visual and language context to ensure safe interaction and avoid
excessive force.
Starting with Task-1 (Fig. 6), we illustrate how OmniVIC ef-
fectively regulates interaction forces while closing the drawer
gently without safety constraint violation. The force stays
within safe limits throughout task execution, demonstrating the
approach’s real-world suitability. In contrast, the comparison
experiments in Fig. 6 (top) show that constant high impedance
causes excessive force, while constant low impedance fails due
to insufficient force to overcome friction.
Furthermore, we elaborate how the stiffness has been reg-
ulated according to the task executionphasein Task-2 and
Task-3 As shown in Fig. 7, we further validate the adaptability
of OmniVIC in more complex scenarios involving continuous
contact and multiple phases of interaction. The robot success-
fully navigates along a path with a ramp, maintaining safe
interaction forces while adapting its impedance parameters
to the varying contact conditions. The stiffness and damping
parameters are adjusted appropriately in real-time based on
the visual and language context, ensuring smooth transitions
between different phases of contact. The force remains within
safe limits throughout the task execution, demonstrating the
robustness and effectiveness of our approach in handling
complex real-world manipulation tasks.
VI. CONCLUSION
In this work, we present OmniVIC, a systematic vari-
able impedance controller that integrates RAG and ICL with
VLMs to enhance VIC for contact-rich robotic manipula-
tion tasks. OmniVIC leverages the strengths of VLMs in
understanding visual and language context to dynamically
generate impedance parameters, enabling robots to adapt their
interactions based on task requirements and environmental
conditions. Through comprehensive simulation and real-world
experiments, we demonstrate that OmniVIC significantly im-
proves task success rates and reduces force violations com-
pared to baselines with the average success rate increasing
from 27% (baseline) to 61.4% (OmniVIC). The integration

2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
steps5001000k values
5 10 15 20 25 30 35
steps50010001500k values
kx ky kz Force Magnitude*100
Free_motion Approaching Contact RetreatFig. 7. Task-2 (top): The robot moves along the negative Y-axis while
maintaining the same height in the Z-axis using OmniVIC variable impedance
control. The system dynamically adjusts stiffness and damping parameters
based on visual and language context to ensure safe interaction and avoid
excessive force. Task-3 (bottom): The robot performs multiple tasks in
sequence using OmniVIC variable impedance control. The system adapts
stiffness and damping parameters based on visual and language context to
ensure safe interaction and avoid excessive force. The shaded regions indicate
the contact phase as identified by the OmniVIC.
of RAG allows the system to retrieve relevant prior experi-
ences, while ICL enables the VLM to generate context-aware
impedance parameters in real-time. This dual mechanism
enhances the robot’s ability to perform safe and compliant in-
teractions in diverse manipulation scenarios. Future work will
explore further integration of VLMs with VIC and investigate
the generalization of our approach to a wider range of physical
tasks extension to mobile robots and humanoids.
VII. ACKNOWLEDGMENTS
Portions of the literature search and preliminary text editing
for this manuscript were assisted by the AI system (ChatGPT-
5, 4o). The authors reviewed, revised, and edited all AI-
generated material to ensure accuracy and completeness.
REFERENCES
[1] B. Tang, M. A. Lin, I. Akinola, A. Handa, G. S. Sukhatme, F. Ramos,
D. Fox, and Y . Narang, “Industreal: Transferring contact-rich assembly
tasks from simulation to reality,”arXiv preprint arXiv:2305.17110, 2023.
[2] T. Tsuji, Y . Kato, G. Solak, H. Zhang, T. Petri ˇc, F. Nori, and A. Ajoudani,
“A survey on imitation learning for contact-rich tasks in robotics,”arXiv
preprint arXiv:2506.13498, 2025.
[3] Y . Narang, K. Storey, I. Akinola, M. Macklin, P. Reist, L. Wawrzyniak,
Y . Guo, A. Moravanszky, G. State, M. Lu,et al., “Factory: Fast contact
for robotic assembly,”arXiv preprint arXiv:2205.03532, 2022.
[4] H. Zhang, G. Solak, G. J. G. Lahr, and A. Ajoudani, “Srl-vic: A
variable stiffness-based safe reinforcement learning for contact-rich
robotic tasks,”IEEE Robotics and Automation Letters, vol. 9, no. 6,
pp. 5631–5638, 2024.
[5] Z. Zhou, S. Veeramani, H. Fakhruldeen, S. Uyanik, and A. I. Cooper,
“Genco: A dual vlm generate-correct framework for adaptive peg-in-
hole robotics,” in2025 IEEE International Conference on Robotics and
Automation (ICRA), 2025, pp. 16 744–16 751.
[6] H. Li, S. Zhang, and D. Guo, “Robocleaner: Robotic tabletop clean-
ing via vlm-powered multi-agent collaboration,”IEEE Transactions on
Automation Science and Engineering, pp. 1–1, 2025.
[7] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel,et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.[8] Q. Xie, S. Y . Min, P. Ji, Y . Yang, T. Zhang, K. Xu, A. Bajaj,
R. Salakhutdinov, M. Johnson-Roberson, and Y . Bisk, “Embodied-rag:
General non-parametric embodied memory for retrieval and generation,”
arXiv preprint arXiv:2409.18313, 2024.
[9] X. Zhang, S. Liu, P. Huang, W. J. Han, Y . Lyu, M. Xu, and D. Zhao,
“Dynamics as prompts: In-context learning for sim-to-real system iden-
tifications,”IEEE Robotics and Automation Letters, 2025.
[10] J. Y . Zhu, C. G. Cano, D. V . Bermudez, and M. Drozdzal, “Incoro: In-
context learning for robotics control with feedback loops,”arXiv preprint
arXiv:2402.05188, 2024.
[11] F. Liu, K. Fang, P. Abbeel, and S. Levine, “Moka: Open-world robotic
manipulation through mark-based visual prompting,”arXiv preprint
arXiv:2403.03174, 2024.
[12] S. Liu, J. Zhang, R. X. Gao, X. V . Wang, and L. Wang, “Vision-language
model-driven scene understanding and robotic object manipulation,” in
2024 IEEE 20th International Conference on Automation Science and
Engineering (CASE). IEEE, 2024, pp. 21–26.
[13] J. Gao, B. Sarkar, F. Xia, T. Xiao, J. Wu, B. Ichter, A. Majumdar,
and D. Sadigh, “Physically grounded vision-language models for robotic
manipulation,” in2024 IEEE International Conference on Robotics and
Automation (ICRA). IEEE, 2024, pp. 12 462–12 469.
[14] J. Buchli, F. Stulp, E. Theodorou, and S. Schaal, “Learning variable
impedance control,”The International Journal of Robotics Research,
vol. 30, no. 7, pp. 820–833, 2011.
[15] C. Yang, G. Ganesh, S. Haddadin, S. Parusel, A. Albu-Schaeffer, and
E. Burdet, “Human-like adaptation of force and impedance in stable and
unstable interactions,”IEEE transactions on robotics, vol. 27, no. 5, pp.
918–930, 2011.
[16] M. Bogdanovic, M. Khadiv, and L. Righetti, “Learning variable
impedance control for contact sensitive tasks,”IEEE Robotics and
Automation Letters, vol. 5, no. 4, pp. 6129–6136, 2020.
[17] Y . Zhu, Z. Ou, X. Mou, and J. Tang, “Retrieval-augmented embodied
agents,” inProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024.
[18] M. Booker, G. Byrd, B. Kemp, A. Schmidt, and C. Rivera, “Embod-
iedrag: Dynamic 3d scene graph retrieval for efficient and scalable robot
task planning,”arXiv preprint arXiv:2410.23968, 2024.
[19] M. Du, S. Nair, D. Sadigh, and C. Finn, “Behavior retrieval: Few-shot
imitation learning by querying unlabeled datasets,” inRobotics: Science
and Systems, 2023.
[20] L.-H. Lin, Y . Cui, A. Xie, T. Hua, and D. Sadigh, “Flowretrieval: Flow-
guided data retrieval for few-shot imitation learning,” inProceedings
of The 8th Conference on Robot Learning (CoRL), ser. Proceedings of
Machine Learning Research, vol. 270. PMLR, 2025, pp. 4084–4099.
[21] J. Duan, W. Yuan, W. Pumacay, Y . R. Wang, K. Ehsani, D. Fox, and
R. Krishna, “Manipulate-anything: Automating real-world robots using
vision-language models,”arXiv preprint arXiv:2406.18915, 2024.
[22] G. Sarch, L. Jang, M. Tarr, W. W. Cohen, K. Marino, and K. Fragki-
adaki, “Vlm agents generate their own memories: Distilling experience
into embodied programs of thought,”Advances in Neural Information
Processing Systems, vol. 37, pp. 75 942–75 985, 2024.
[23] F. Batool, M. Zafar, Y . Yaqoot, R. A. Khan, M. H. Khan, A. Fedoseev,
and D. Tsetserukou, “Impedancegpt: Vlm-driven impedance control of
swarm of mini-drones for intelligent navigation in dynamic environ-
ment,”arXiv preprint arXiv:2503.02723, 2025.
[24] H. H. Jekel, A. D. Rosales, and L. Peternel, “Visio-verbal teleimpedance
interface: Enabling semi-autonomous control of physical interaction via
eye tracking and speech,”arXiv preprint arXiv:2508.20037, 2025.
[25] J. Bi, K. Y . Ma, C. Hao, M. Z. Shou, and H. Soh, “Vla-touch: Enhancing
vision-language-action models with dual-level tactile feedback,”arXiv
preprint arXiv:2507.17294, 2025.
[26] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fu-
sai, L. Groom, K. Hausman, B. Ichter,et al., “\pi 0 : A vision-
language-action flow model for general robot control,”arXiv preprint
arXiv:2410.24164, 2024.
[27] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “Bge
m3-embedding: Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation,”arXiv preprint
arXiv:2402.03216, 2024.