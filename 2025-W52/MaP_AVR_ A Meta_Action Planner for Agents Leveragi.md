# MaP-AVR: A Meta-Action Planner for Agents Leveraging Vision Language Models and Retrieval-Augmented Generation

**Authors**: Zhenglong Guo, Yiming Zhao, Feng Jiang, Heng Jin, Zongbao Feng, Jianbin Zhou, Siyuan Xu

**Published**: 2025-12-22 14:58:52

**PDF URL**: [https://arxiv.org/pdf/2512.19453v1](https://arxiv.org/pdf/2512.19453v1)

## Abstract
Embodied robotic AI systems designed to manage complex daily tasks rely on a task planner to understand and decompose high-level tasks. While most research focuses on enhancing the task-understanding abilities of LLMs/VLMs through fine-tuning or chain-of-thought prompting, this paper argues that defining the planned skill set is equally crucial. To handle the complexity of daily environments, the skill set should possess a high degree of generalization ability. Empirically, more abstract expressions tend to be more generalizable. Therefore, we propose to abstract the planned result as a set of meta-actions. Each meta-action comprises three components: {move/rotate, end-effector status change, relationship with the environment}. This abstraction replaces human-centric concepts, such as grasping or pushing, with the robot's intrinsic functionalities. As a result, the planned outcomes align seamlessly with the complete range of actions that the robot is capable of performing. Furthermore, to ensure that the LLM/VLM accurately produces the desired meta-action format, we employ the Retrieval-Augmented Generation (RAG) technique, which leverages a database of human-annotated planning demonstrations to facilitate in-context learning. As the system successfully completes more tasks, the database will self-augment to continue supporting diversity. The meta-action set and its integration with RAG are two novel contributions of our planner, denoted as MaP-AVR, the meta-action planner for agents composed of VLM and RAG. To validate its efficacy, we design experiments using GPT-4o as the pre-trained LLM/VLM model and OmniGibson as our robotic platform. Our approach demonstrates promising performance compared to the current state-of-the-art method. Project page: https://map-avr.github.io/.

## Full Text


<!-- PDF content starts -->

MaP-A VR: A Meta-Action Planner for Agents Leveraging Vision
Language Models and Retrieval-Augmented Generation
Zhenglong Guo*, Yiming Zhao*, Feng Jiang, Heng Jin,
Zongbao Feng, Jianbin Zhou, Siyuan Xu
Abstract— Embodied robotic AI systems designed to manage
complex daily tasks rely on a task planner to understand
and decompose high-level tasks. While most research focuses
on enhancing the task-understanding abilities of LLMs/VLMs
through fine-tuning or chain-of-thought prompting, this paper
argues that defining the planned skill set is equally crucial.
To handle the complexity of daily environments, the skill
set should possess a high degree of generalization ability.
Empirically, more abstract expressions tend to be more gener-
alizable. Therefore, we propose to abstract the planned result
as a set of meta-actions. Each meta-action comprises three
components: move/rotate, end-effector status change, relation-
ship with the environment. This abstraction replaces human-
centric concepts, such as grasping or pushing, with the robot’s
intrinsic functionalities. As a result, the planned outcomes
align seamlessly with the complete range of actions that the
robot is capable of performing. Furthermore, to ensure that
the LLM/VLM accurately produces the desired meta-action
format, we employ the Retrieval-Augmented Generation (RAG)
technique, which leverages a database of human-annotated
planning demonstrations to facilitate in-context learning. As
the system successfully completes more tasks, the database will
self-augment to continue supporting diversity. The meta-action
set and its integration with RAG are two novel contributions of
our planner, denoted as MaP-A VR—the meta-action planner for
agents composed of VLM and RAG. To validate its efficacy, we
design experiments using GPT-4o as the pre-trained LLM/VLM
model and OmniGibson as our robotic platform. Our approach
demonstrates promising performance compared to the current
state-of-the-art method. Project page:https://map-avr.github.io/
I. INTRODUCTION
A core capability of intelligent robots designed to as-
sist humans in daily life is the ability to simultaneously
understand both human needs and 3D environments. The
emergence of large language models, particularly multimodal
models such as GPT-4V and GPT-4o, showcases remarkable
intelligence and has ignited hope for developing such ad-
vanced robots [1], [2], [3].
The optimal design of such intelligent robotic systems
continues to be an open and active area of research. Incorpo-
rating language signals into models, which enable networks
originally trained via reinforcement learning or imitation
learning for single-task settings to handle multiple skills
based on varying language inputs, has garnered significant
research interest [4], [5], [6], [7], [8]. These employed lan-
guage signals are typically short action commands rather than
natural language task descriptions. To address this limitation,
—————————————————————————————–
* Equal contribution.
All authors are affiliated with Li Auto.
Email: zhaoyiming3@lixiang.com
Opened,   above, snack package,         
Opened,                   downward, ,                          
Opened,  on, snack package,              
Closed,     upward,                               
Closed,     front, refrigerator handle,    
Closed,                    on, refrigerator handle,
Opened,   inside, refrigerator,
Opened,                    on, shelf,                             
Closed,     backward, ,                         
Closed,                    front, refrigerator handle,    
Closed,     on, refrigerator handle,       task instruction : Grab the snack and put it inside the refrigerator
ee statusmove/rotate location descriptionOpened
Opened
Closed
Closed
Closed
Opened
Opened
Closed
Closed
Closed
Openedmove to,
move to,
move to,
move to,
move to,
move to,
move to,
move to,
move to,
move to,
move to,Fig. 1: An example of a set of meta-actions as the planned
outcome. Each meta-action comprises three essential compo-
nents, which are utilized in subsequent execution functions.
instead of relying on interactive approaches with humans [9],
recent research has shifted towards utilizing multimodal large
language models as task planners responsible for decompos-
ing high-level daily task descriptions into sequences of short
action instructions [10], [11]. Typically, these short action
instructions are skills, such as grasping, placing, pushing,
pulling, moving, and releasing, defined by human experience.
However, whether these predefined human-centric skill sets
can fully capture the vast diversity of daily tasks that the
robot is able to complete remains an open question, raising
concerns about the generalization ability of this type of skill
set definition.
MotivationTo address the aforementioned concerns, we
propose that the planned results be structured as a set of
meta-actions. Our idea is inspired by Richard S. Sutton’s
blog titled “The Bitter Lesson” [12], where he argues that AI
agents should be constructed using only meta-methods capa-
ble of discovering and capturing arbitrary complexity, rather
than relying on built-in human-centric knowledge. These
high-capacity meta-methods are general-purpose and have
the potential to approximate arbitrary complexity, thereby
working towards achieving his vision: “We want AI agents
that can discover like we can, not contain what we have
discovered.” In light of this, we designed a meta-action plan-
ner, where the planned meta-actions represent the operations
that the robot itself can inherently perform, allowing for
a versatile representation of daily tasks through arbitrary
combinations.
In 3D space, almost all robots possess the fundamental
capabilities of translation and rotation. A robot’s end-effectorarXiv:2512.19453v1  [cs.RO]  22 Dec 2025

might be equipped with various tools or attachments, which
Fig. 2: The figure il-
lustrates that meta-actions
represent the foundational
abstraction for many skills
aligned with human expe-
rience. The flexible combi-
nations of meta-actions can
be used to compose a wide
range of daily tasks.we collectively refer to as its
end-effector status. Thus, a
robot’s inherent actions can
be abstracted as move/ro-
tate, ee status change. How-
ever, to address diverse tasks,
these inherent actions need
to be contextualized within
the environment. Therefore,
we propose the following ab-
stract meta-action to struc-
ture the planner’s output, as
shown in Eq. 1. An exam-
ple of how these meta-actions
function as the planned result
is shown in Fig. 1. Moreover,
consider a single-arm robot
with a gripper as the end-
effector; in Fig. 2, we demon-
strate that meta-actions can
be combined in various ways
to achieve higher-level sub-
tasks that correspond to hu-
man concepts, ultimately of-
fering greater flexibility and
generality for solving daily
activity tasks. Please see V-
A for further discussion about
the design intentions of the
meta-action. The execution of
these meta-actions requires
only a few execution func-
tions, as illustrated in III-C.
To ensure that the planned
outcomes can be successfully
executed using the corresponding execution functions, the
outputs of the task planner must adhere to the struc-
tured decomposition outlined by the meta-action frame-
work. Therefore, we further enhance our prompt engineering
using Retrieval-Augmented Generation (RAG) techniques.
Specifically, we construct a database of well-planned tasks
conforming to our desired format. By retrieving the most
similar task, we facilitate in-context learning for the plan-
ner. Crucially, this task database features self-augmentation
capabilities, allowing newly completed and verified tasks to
be automatically incorporated into the database. This process
progressively enhances the system’s robustness. How the task
database interacts with the meta-action planner is shown in
Fig. 3.
ContributionsThis paper presents a novel task planner,
along with its corresponding execution actions. We summa-
rize our contributions as follows:
•We propose MaP-A VR, a meta-action planner for agents
that integrate Vision-Language Models (VLMs) and
Retrieval-Augmented Generation (RAG). By abstracting
skills into meta-actions and utilizing a self-augmented
Fig. 3: This figure illustrates the interaction between the task
database and the planner. The planner retrieves the most
similar example for in-context learning, and successfully
verified tasks are subsequently added to the database.
database with RAG, we establish a novel paradigm
for task decomposition, ensuring the scalability and
generalization ability of the proposed task planner.
•By utilizing the proposed task planner in conjunction
with our defined successive execution functions, the
proposed solution, as a type of agent-style method,
surpasses the performance of recently published state-
of-the-art agent methods. This demonstrates its potential
for application across various domains, which are fur-
ther discussed in V-B.
II. RELATED WORK
Single Task MethodsEarly reinforcement learning and
imitation learning methods typically focused on learning
within a single environment or under the guidance of a single
type of demonstration [4], [5], [13], [14], [15]. These meth-
ods have shown notable success in non-human interactive
areas, such as motion control [16].
Multi-action ModelsTo handle complex, everyday tasks,
robots require the capacity to perform diverse actions. Recent
advancements in multimodal language models have facili-
tated the incorporation of language embeddings as a common
method to integrate interactive information [17]. Conse-
quently, some research has explored introducing language
signals into imitation learning frameworks, enabling neural
networks to execute different actions based on language
instructions [8], [18], [19]. Some studies have gone further by
using token replacement techniques to transform multimodal
vision-language models (VLMs) into multimodal vision-
language-action models [6], [20], [21]. This transformation
imbues action models with the inherent world understanding
acquired by VLMs during training on massive datasets.
Task Planner in Two-Step ApproachesDespite incor-
porating language information into action networks, some
recent researchers favor a two-step approach [2], [10], [11],
[18], [22], [23]: first, employing LLMs/VLMs to comprehend
task descriptions and decompose them into multi-step action
plans; and second, executing these plans using appropriate
action models. However, a critical gap exists in the literature
on task planning: a lack of thorough discussion regarding
the optimal construction of action or skill sets. Most existing
work relies on heuristic approaches, defining actions or skills

Fig. 4: The overview of our proposed MaP-A VR. Compared to previous methods, our pipeline differs in each of its major
components.In understanding the task, we incorporate the Retrieval-Augmented Generation (RAG) technique to search
the database for the closest successful task to facilitate in-context learning.In planning the task, we designed a series of
Chain-of-Thought (CoT) prompts to guide the model generating meta-actions that align with expectations.In executing the
task, we leverage the spatial understanding ability of VLMs, in conjunction with foundation models(such as Sam, DinoV2)
and classical obstacle avoidance algorithms, to ensure that the planned meta-actions are executed effectively and successfully.
based on human intuition and understanding, such as pick,
move, place, open, close, etc.
VLM Powered Robotic AgentBeyond reinforcement
learning, imitation learning, and vision-language-action mod-
els discussed above, AI agent-based approaches have
emerged as a compelling avenue for developing robotic
solutions, attracting considerable attention from researchers.
With a VLM as its core, the system orchestrates a suite of
foundation models (e.g., Grounding-Dino [24] and GraspNet
[25]) and traditional methods (e.g., obstacle avoidance),
forming a training-free AI agent system capable of complet-
ing diverse everyday tasks [26], [27], [28], [29]. In this paper,
our definition of action execution functions draws inspiration
from the implementation strategies employed in these agent-
based approaches.
CoT, ICL and RAG in RoboticRecent advancements
in large language models have introduced numerous pow-
erful techniques that facilitate the emergence of advanced
capabilities in these models, such as Chain-of-Thought
(CoT) prompting [30], in-context learning (ICL) [31], and
Retrieval-Augmented Generation (RAG) [32]. Studies con-
sistently demonstrate that these methods unlock a remarkably
high expressive capacity in large language models [33].
Consequently, research exploring the application of these
techniques to enhance robotic problem-solving is gaining
significant traction. The use of Chain-of-Thought (CoT)
prompting enables LLMs to perform direct path planning
[34] and significantly enhances the capabilities of vision-
language-action (VLA) models [35]. Several studies have
demonstrated the potential of in-context learning (ICL) tech-niques, even for high-precision tasks like grasping [36], [37].
In contrast to CoT and ICL, the application of Retrieval-
Augmented Generation (RAG) in robotics remains relatively
under-explored. While recent work has demonstrated the use
of RAG for robot navigation [38], this paper represents an
early exploration of its potential in enabling robots to handle
everyday tasks.
III. METHODS
A. Preliminary
Note that throughout the remainder of the paper, we use
the following abbreviations: VLMs refer to vision-language
models, RAG refers to Retrieval-Augmented Generation,
ICL refers to in-context learning, and CoT refers to Chain-
of-Thought. The VLM used in this paper is GPT-4o. In
recent papers, VLM, RAG, ICL, CoT, and various foundation
models have been combined in different forms as one or
more components of robotic systems. However, the use of
these techniques in this paper differs from previous works in
the following ways:
In contrast to using RAG for robot navigation [38], we
build the RAG database to assist the planner in producing
the desired outcomes. ICL is used as part of RAG, which
contrasts with some studies where ICL is employed to guide
VLM as a pattern generator that directly determines the
position of the robotic gripper [36], [37]. We design the CoT
prompt from scratch to facilitate the alignment of the VLM’s
outputs with the meta-action format. As for foundation mod-
els, we do not extensively utilize models such as GraspNet
[25]; instead, our method primarily focuses on unlocking

Fig. 5: This figure shows the linguistic structure of the meta-
action we finally guide the VLM to generate.
the spatial understanding capabilities demonstrated in PIVOT
[39].
B. Method Pipeline
We illustrate the entire pipeline of the proposed MaP-A VR
in Fig. 4. When the robot receives an instruction for a daily
task, it first searches its existing experience database for the
most similar task and scenario. The located completed task,
along with its prompting cache, is used as a demonstration
and is sent to the VLM together with the task instruction. We
design a series of prompts, such as having the VLM describe
the scenario and identify task-related objects, among others.
This process results in a collection of step-by-step actions
that are described in natural language and can be verified by
the VLM. After a reasonableness check, we generate a series
of meta-actions as the final planning result. These meta-
actions are sent to the corresponding execution functions
sequentially.
C. Meta-Action Set
1) The definition of the meta-action set:The meta-action
is proposed to achieve two objectives: first, to maintain the
generality of the robotic system, and second, to ensure ex-
ecutability by originating actions from the robot’s actuators.
Based on the design principle in V-A, we define the format
of each instruction set as follows: gripper status before action
(Open or Close), move or rotate, location description such
as the direction (e.g., left/right/up/down/forward/backward)
of the target object, and gripper status after reaching the
target pose. For example, the meta-action could be “opened,
move to, front on, burger, closed.” In Fig. 5, we illustrate
this transition process that guides the VLM to output meta-
actions in the desired fixed linguistic structure. The first
and last elements describe the status change of the end-
effector. The second element involves the basic action of the
end-effector, such as moving or rotating. The middle part
depicts the final relationship between the end-effector and
the environment after the action has been executed, thereby
providing a textual goal for the current action. These three
parts are abstracted in Eq. 1.
{move/rotate, location description, ee status}.(1)
2) The CoT prompt for the meta-action set:As shown in
Figure 6, we designed a multi-turn conversation that includes
a prompt system based on the CoT principle to help the
VLM generate meta-action results. This conversation reflects
an understanding of the scenario and task. It leverages the
Fig. 6: This figure illustrates the prompts used for meta-
action generation.
VLM’s general knowledge capabilities to produce a coherent
task planning sequence. The prompts explicitly guide the
VLM to optimize the task planning sequence for rationality
and correctness. Finally, the definition of the meta-action is
provided, and the VLM is instructed to output the final task
planning sequence in the meta-action format based on its
understanding.
3) Move/Rotate and Gripper executor implementation:As
illustrated in Figures 4 and 5, the meta-actions are carried
out through corresponding functions. The robot utilized in
this study is a single-arm Fetch Robot, therefore, we refer to
the end effector simply as the gripper. The gripper execution
function follows a straightforward logic: meta-actions such
as{Open,..., Close}and{Close,..., Open}will operate the
gripper by closing and opening it, respectively. In contrast,
sequences like{Open,..., Open}and{Close,..., Close}will
maintain the current status of the gripper. The key factor
in the meta-action is thelocation descriptionterm, which
serves as the textual goal for each meta-action. While ex-
ecuting each meta-action, thelocation descriptionwill be
finally transformed to one specific executable target 6-Dof
pose denoted asP target . The transformation process roughly
follows the process below:
•Step 1, utilize VLM to find an initial 3D point along
with a default orientation as theP initvia visual prompt-
ing, shown in Eq. 2. If thelocation descriptioncon-
tains only a preposition, use the last position as the
initial 3D point.
V LM(location description)−→P init (2)
•Step 2, uniformly samplenoffset 6-Dof poses to
preparencandidates, shown in Eq. 3. When the move
action is executed with a single preposition as the
location description, the offset can only sample trans-
lation. For the rotate action, the offset can only sample
rotation.

Fig. 7: This figure illustrates how we uniformly sample
multiple candidate poses around the initial pose and rely on
the VLM to determine the final target pose.
Pi
candidate =P init+Pi
offset (3)
•Step 3, the VLM will be used as a selector to choose the
best candidate that aligns most closely with the overall
goal, shown in Eq. 4.
Ptarget =Select({Pi
candidate , i∈[1, ..., n]})(4)
Instead of directly using foundation grasping models like
GraspNet, the action execution functions make the greatest
possible use of the general knowledge embedded within the
VLM, as highlighted by PIVOT [39]. This specific candidate
selection process is visually shown in Fig. 7. After the arm
reaches the target pose, the gripper action is executed.
D. RAG in Robotic Task Planning
1) Why the RAG is needed in our task planning:
The definition of meta-action requires the output of the
VLM to maintain a specific format. As shown in Fig.
1, the first element of theee statusshould be the same
as the last element in the previous meta-action. For the
location description, the use of prepositions is crucial. For
example, “above” signifies moving to a position over the
object while maintaining a certain distance, whereas “on”
denotes directly interacting with the object. Utilizing RAG
technology can help eliminate the potential confusion that
may arise from pre-trained VLMs. Moreover, we empirically
found that replacing the most similar examples with less
similar ones indeed caused some previously successful plans
to fail. This aligns with the anticipated benefits of utilizing
RAG and ICL.
2) Database preparation and planner performance evalu-
ation via human annotating:To utilize RAG, we need to pre-
pare a database of successful planning demonstrations. Those
initial demonstrations in this database relying on human
involvement. Additionally, since our planner and execution
functions are decoupled, the planner can be used to generate
plans for any scenario independently. However, evaluatingthe planner’s performance on arbitrary data without rely-
ing on the executor also necessitates human involvement.
Inspired by [2], we have developed multiple web interfaces
connected to the database to support these human-involved
functions. These interfaces facilitate the invocation of the
planner, modifications to the entire intermediate planning
process, saving of correct planning results, comparisons of
planning outcomes for arbitrary images with and without
ICL, and manual voting on the correctness of the planning
results, among other functionalities.
3) The design of the RAG integration:With the initial
database in place, we can begin querying and augmenting the
planner using RAG. For the retrieval component, we utilize
VLMs to extract the object scene graph from the image,
serving as a scene description. This scene graph, along with
the task instruction, is used to embed the task into a suitable
representation. When a new task arises, we query the top-
k samples and utilize VLMs to select the most similar one.
For the augmentation component, we retrieve the saved GPT-
cache of the selected demonstration, which contains all the
prompts and replies used in the demonstration’s planning
process, and add the GPT-cache as an additional dialogue
round.
4) The extended of the RAG database:For the RAG
database, In addition to human involvement, it can also grow
autonomously as more tasks are completed. We evaluated
two metrics for every task in the database with method
based VLM (GPT-4o). That are the similarity of relevant
objects and the similarity of sequences from the task planner
normalized to[0,1]. If a new task is executed correctly and
these two metrics calculated across all tasks in the database
are lower than preset, suggesting that the new task has low
cross-correlation with the all existing task in the database,
the new task will be added to the task database. The growth
of the RAG database will enhance the meta-action planner
to predict correctly result for other tasks that has low cross-
correlation with the RAG database.
IV. EXPERIMENTS
A. Datasets
1) OmniGibson:Identifying suitable datasets for evalu-
ating embodied intelligence has long been a challenging
issue in the field. While robotic data gathered from real-
world operations tend to be more realistic and credible, such
data is often closely tied to the specific robot from which
it was collected, making the methods developed on these
datasets difficult to reproduce widely. Testing methods in
simulators are reproducible, but the tasks offered by most
simulators are relatively simple compared to the complex
tasks encountered in daily life. Therefore, to address this
issue, we choose to use OmniGibson as our simulator. Built
upon the powerful NVIDIA Omniverse platform, it allows
users to develop and define their own scenes and tasks. We
develop several scenarios and tasks according to our own
needs, as shown in Fig. 8.

Open the drawer,
then put the 
apple inside of it.
Could you help 
me clean up the 
floor?
Reorient the white 
pen and drop it 
upright into the 
black pen holder.
Make me a cup of 
espresso, please. 
The red button is 
the switch.
Fig. 8: This figure shows some daily tasks that we developed
in OmniGibson to test our method.
2) The mixed image collection:As mentioned in III-
D.2, the planner and subsequent execution functions are
decoupled. For evaluating the task planner, we can perform
planning for any task using any image and subsequently
assess the results through manual voting by humans with-
out involving the execution component. We prepared a
mixed data collection by sampling images from RT-1 [6],
RoboVQA [40], and Droid-100 [41] sourced from the Open
X-Embodiment data pool [42]. Additionally, we captured
several office scenes, which, together with the previously
sampled images, formed the mixed collection used in this
study. Some samples are shown in Fig. 9. Note that all task
instructions were randomly provided by human annotators
based on the images, ensuring that each image has a unique
task.
B. Performance Comparison with the State-of-the-art
TABLE I: Performance Comparison with Rekep on Omni-
Gibson
Task Type n-trials Rekep Ours w/o ICL Ours w ICL
Insert the pen 40 14/40 4/40 18/40
Clean up the floor 40 8/40 3/40 33/40
Open drawer 40 0/40 8/40 10/40
Make coffee 40 0/40 3/40 8/40
success rate 160 13.75% 11.25% 43.13%
We set the benchmark by selecting Rekep [27] as the
baseline state-of-the-art comparison method. Rekep is a
recently published work that has achieved significant results
compared to previous methods. The authors have open-
sourced their code, utilizing OmniGibson as the demonstra-
tion platform. This allows us to directly use their open-source
code, thereby avoiding the uncertainty and risks associated
with re-implementing the algorithm.
The comparison results are shown in Table I. The method
proposed in this study significantly outperforms Rekep. Al-
though Rekep surpasses previous methods such as V oxposer
[43], its success rate is relatively limited and only increases
significantly when the required feature points are manually
Grab me a coke please.Scoop a spoonful of soup, 
then pour it into the cup 
on the right.Could you help me move 
pens on the table back 
into the pen holder?
Fig. 9: This figure displays the images from our mixed
data collection, which are used to evaluate the meta-action
planner.
provided, as reported in their paper. In contrast, our method
achieves higher success rates without the need for human in-
tervention, demonstrating a stronger ability to solve complex
tasks.
C. Ablation Study
1) Failure case analysis:Here, we further analyze which
steps contribute to the final failure in Fig. 10. It can
be seen from the figure that the failure to locate the
target object and the target grasping point is the main
reason (about 26%), which is usually the basic input of
the subsequent VLM. Therefore, the accuracy and ra-
tionality of the target point positioning directly affects
the judgment of the subsequent VLM. Especially for
the ‘open drawer’ and ‘make coffee’ tasks, it is diffi-
cult to find the target points of the red button of the
Fig. 10: Statistical analysis of the
primary failure reason for each trial.coffee machine and
the handle of the
drawer. The second
is the parsing ac-
tion, accounting for
about 25%, which
is mainly manifested
in the error in judg-
ing the rotation axis
when parsing the ro-
tation action through
VLM. In addition,
the proportion of cases caused by unreasonable task planning
and candidate pose generation is relatively small, about 13%
and 10% respectively.
2) How much the RAG can help:Using Retrieval-
Augmented Generation (RAG) enables us to perform in-
context learning. Although the reasons for task failure can
be diverse, a better planner should theoretically be able to
statistically improve the success rate of task completion. In
Table I, we compare the success rates on OmniGibson with
and without in-context learning (ICL). We observe that the
planner equipped with ICL dramatically improves the overall
task success rate.
In addition to indirectly evaluating task success rates, we
prepared a diverse mixed image dataset specifically designed
to test the task planner (see Fig. IV-A.2). Using the web

interface we developed, human annotators can assign arbi-
trary task instructions based on the images they select. These
tasks can vary in complexity, being either simple or complex,
depending on the annotators’ discretion. The back-end calls
our planner and returns the planned results, both with and
without ICL. Human annotators then assess whether, if they
were the robot, they could complete the task by following
the planned result. They click the corresponding buttons to
indicate whether the planned results are correct or not. The
planning success rate is shown in Table II, demonstrating
that ICL significantly improves planning. We also discovered
that in the absence of ICL, the meta-action planner exhibits
certain consistent error patterns, such as providing redundant
actions or confusing directions during push and pull tasks.
TABLE II: Performance Comparison without and with ICL
on the Mixed Data Collection.
Data Source n-tasks w/o ICL with ICL
RT-1 20 6/20 13/20
RoboVQA 20 4/20 13/20
Droid-100 20 8/20 14/20
Office 50 17/50 39/50
success rate 110 31.8% 71.8%
V. DISCUSSION
A. The Design Principle of the Meta-Action
We believe an effective task planner should exhibit both
generalization ability and comprehensiveness. To enhance
generalization ability, we abstract skills into meta-actions
encompassing only “move”, “rotate”, and end-effector ac-
tions. These are the fundamental commands that the robot
is inherently capable of executing. However, this abstrac-
tion introduces challenges in maintaining comprehensiveness
across all scenarios. To address the challenge of maintaining
comprehensiveness while abstracting skills into meta-actions,
we draw inspiration from John Maynard Keynes’s famous
adage: “It is better to be roughly right than precisely wrong”.
Specifically, we introduce alocation description, which
provides a more flexible and approximate representation
of the action’s end status. Thus, our meta-action does not
represent a rigid action command but rather a description of
the intended action and its desired outcome. This approach
ensures that our planner is “roughly right” at the task
planning level. This is crucial because an incorrect plan
at this stage would render the entire system incapable of
completing the task. The process of translating this “roughly
right” plan into a “precisely right” action command is then
deferred to the action execution functions.
B. Integrate the Proposed Planner with Other Methods
The proposed task planner, in conjunction with our defined
execution functions, constitutes an AI agent-based solution
for robotic daily task completion. This solution can inde-
pendently solve problems like similar agent-style methods
V oxPoser [43], VILA [22] + CoPa [23], MOKA [26], and
Rekep [27].Generating Data for Imitation LearningA recent study
introduced an agent-style method and demonstrated that
using it to collect demonstrations for imitation learning
network training could produce results comparable to those
obtained from human-provided demonstrations [29]. The
method presented in this paper exhibits the capability to
achieve a similar outcome.
Working with the Policy NetworkIn many research
papers, the action executor takes the form of a policy net-
work [18], [19]. Unlike most other agent-style methods that
understand tasks as visual prompts coupled with successive
execution [26], [27], the task planner proposed in this paper
can function independently, utilizing a policy network as its
action executor.
VI. CONCLUSIONS
Decomposing human instructions into executable signals
through a task planner is a crucial step toward advanc-
ing robotic intelligence. In this letter, we propose MaP-
A VR, a meta-action planner for agents composed of Vision-
Language Models (VLMs) and Retrieval-Augmented Gener-
ation (RAG), to explore this direction. We define a general-
purpose meta-action to guide the multimodal large language
model in formatting its outputs into directly executable
instructions. Retrieval-Augmented Generation (RAG) is uti-
lized to consolidate the planning process, ensuring better
alignment with our desired format. The experiments were
conducted in robotic simulation environments developed
with OmniGibson, ensuring that all results are reproducible
and extendable. We believe our research benefits the commu-
nity by enhancing how embodied AI can better understand
and perform human daily activities.
REFERENCES
[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna,
S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi,et al., “Open-
vla: An open-source vision-language-action model,”arXiv preprint
arXiv:2406.09246, 2024.
[2] N. Wake, A. Kanehira, K. Sasabuchi, J. Takamatsu, and K. Ikeuchi,
“Gpt-4v (ision) for robotics: Multimodal task planning from human
demonstration,”IEEE Robotics and Automation Letters, 2024.
[3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat,et al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[4] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y . Du, B. Burchfiel, R. Tedrake,
and S. Song, “Diffusion policy: Visuomotor policy learning via ac-
tion diffusion,”The International Journal of Robotics Research, p.
02783649241273668, 2023.
[5] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, “3d diffuser ac-
tor: Policy diffusion with 3d scene representations,”arXiv preprint
arXiv:2402.10885, 2024.
[6] A. Brohan, N. Brown, J. Carbajal, Y . Chebotar, J. Dabis, C. Finn,
K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu,et al., “Rt-1:
Robotics transformer for real-world control at scale,”arXiv preprint
arXiv:2212.06817, 2022.
[7] A. Goyal, J. Xu, Y . Guo, V . Blukis, Y .-W. Chao, and D. Fox, “Rvt:
Robotic view transformer for 3d object manipulation,” inConference
on Robot Learning. PMLR, 2023, pp. 694–710.
[8] A. Goyal, V . Blukis, J. Xu, Y . Guo, Y .-W. Chao, and D. Fox, “Rvt-
2: Learning precise manipulation from few demonstrations,”arXiv
preprint arXiv:2406.08545, 2024.
[9] A. Brohan, Y . Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho,
J. Ibarz, A. Irpan, E. Jang, R. Julian,et al., “Do as i can, not as i say:
Grounding language in robotic affordances,” inConference on robot
learning. PMLR, 2023, pp. 287–318.

[10] Y . Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai,
Y . Qiao, and P. Luo, “Embodiedgpt: Vision-language pre-training
via embodied chain of thought,”Advances in Neural Information
Processing Systems, vol. 36, 2024.
[11] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter,
A. Wahid, J. Tompson, Q. Vuong, T. Yu,et al., “Palm-e: An embodied
multimodal language model,” inInternational Conference on Machine
Learning. PMLR, 2023, pp. 8469–8488.
[12] R. S. Sutton. (2019) The bitter lesson. 2024-11-11. [Online].
Available: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
[13] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch,
S. Levine, and C. Finn, “Bc-z: Zero-shot task generalization with
robotic imitation learning,” inConference on Robot Learning. PMLR,
2022, pp. 991–1002.
[14] D. Kalashnikov, J. Varley, Y . Chebotar, B. Swanson, R. Jon-
schkowski, C. Finn, S. Levine, and K. Hausman, “Mt-opt: Continuous
multi-task robotic reinforcement learning at scale,”arXiv preprint
arXiv:2104.08212, 2021.
[15] T. Z. Zhao, V . Kumar, S. Levine, and C. Finn, “Learning fine-grained
bimanual manipulation with low-cost hardware,”arXiv preprint
arXiv:2304.13705, 2023.
[16] Y . J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman,
Y . Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward
design via coding large language models,” inThe Twelfth International
Conference on Learning Representations, 2024.
[17] G. Kim, T. Kwon, and J. C. Ye, “Diffusionclip: Text-guided diffu-
sion models for robust image manipulation,” inProceedings of the
IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 2426–2435.
[18] C. Lynch, A. Wahid, J. Tompson, T. Ding, J. Betker, R. Baruch,
T. Armstrong, and P. Florence, “Interactive language: Talking to robots
in real time,”IEEE Robotics and Automation Letters, 2023.
[19] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, “3d diffuser actor:
Policy diffusion with 3d scene representations,” inICRA 2024 Work-
shop—Back to the Future: Robot Learning Going Probabilistic, 2024.
[20] A. Brohan, N. Brown, J. Carbajal, Y . Chebotar, X. Chen, K. Choro-
manski, T. Ding, D. Driess, A. Dubey, C. Finn,et al., “Rt-2: Vision-
language-action models transfer web knowledge to robotic control,”
arXiv preprint arXiv:2307.15818, 2023.
[21] S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson,
Y . Chebotar, D. Dwibedi, and D. Sadigh, “Rt-h: Action hierarchies
using language,”arXiv preprint arXiv:2403.01823, 2024.
[22] Y . Hu, F. Lin, T. Zhang, L. Yi, and Y . Gao, “Look before you leap:
Unveiling the power of GPT-4v in robotic vision-language planning,”
inFirst Workshop on Vision-Language Models for Navigation and
Manipulation at ICRA 2024, 2024.
[23] H. Huang, F. Lin, Y . Hu, S. Wang, and Y . Gao, “Copa: General robotic
manipulation through spatial constraints of parts with foundation
models,” inFirst Workshop on Vision-Language Models for Navigation
and Manipulation at ICRA 2024, 2024.
[24] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li,
J. Yang, H. Su, J. Zhu,et al., “Grounding dino: Marrying dino with
grounded pre-training for open-set object detection,”arXiv preprint
arXiv:2303.05499, 2023.
[25] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “Graspnet-1billion: A large-
scale benchmark for general object grasping,” inProceedings of the
IEEE/CVF conference on computer vision and pattern recognition,
2020, pp. 11 444–11 453.
[26] F. Liu, K. Fang, P. Abbeel, and S. Levine, “MOKA: Open-vocabulary
robotic manipulation through mark-based visual prompting,” inFirst
Workshop on Vision-Language Models for Navigation and Manipula-
tion at ICRA 2024, 2024.
[27] W. Huang, C. Wang, Y . Li, R. Zhang, and L. Fei-Fei, “Rekep:
Spatio-temporal reasoning of relational keypoint constraints for robotic
manipulation,” in8th Annual Conference on Robot Learning, 2024.
[28] G. Tziafas and H. Kasaei, “Towards open-world grasping with large
vision-language models,” in8th Annual Conference on Robot Learn-
ing, 2024.
[29] J. Duan, W. Yuan, W. Pumacay, Y . R. Wang, K. Ehsani, D. Fox,
and R. Krishna, “Manipulate-anything: Automating real-world robots
using vision-language models,” in8th Annual Conference on Robot
Learning, 2024.
[30] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V .
Le, D. Zhou,et al., “Chain-of-thought prompting elicits reasoning inlarge language models,”Advances in neural information processing
systems, vol. 35, pp. 24 824–24 837, 2022.
[31] T. B. Brown, “Language models are few-shot learners,”arXiv preprint
arXiv:2005.14165, 2020.
[32] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel,et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances
in Neural Information Processing Systems, vol. 33, pp. 9459–9474,
2020.
[33] Z. Li, H. Liu, D. Zhou, and T. Ma, “Chain of thought empowers
transformers to solve inherently serial problems,” inThe Twelfth
International Conference on Learning Representations, 2024.
[34] T. Kwon, N. Di Palo, and E. Johns, “Language models as zero-shot
trajectory generators,”IEEE Robotics and Automation Letters, 2024.
[35] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine,
“Robotic control via embodied chain-of-thought reasoning,” in8th
Annual Conference on Robot Learning, 2024.
[36] N. D. Palo and E. Johns, “Keypoint action tokens enable in-context
imitation learning in robotics,” inFirst Workshop on Vision-Language
Models for Navigation and Manipulation at ICRA 2024, 2024.
[37] Y . Yin, Z. Wang, Y . Sharma, D. Niu, T. Darrell, and R. Herzig,
“In-context learning enables robot action prediction in llms,”arXiv
preprint arXiv:2410.12782, 2024.
[38] A. Anwar, J. Welsh, J. Biswas, S. Pouya, and Y . Chang, “Remembr:
Building and reasoning over long-horizon spatio-temporal memory for
robot navigation,”arXiv preprint arXiv:2409.13682, 2024.
[39] S. Nasiriany, F. Xia, W. Yu, T. Xiao, J. Liang, I. Dasgupta, A. Xie,
D. Driess, A. Wahid, Z. Xu, Q. Vuong, T. Zhang, T.-W. E. Lee,
K.-H. Lee, P. Xu, S. Kirmani, Y . Zhu, A. Zeng, K. Hausman,
N. Heess, C. Finn, S. Levine, and brian ichter, “PIVOT: Iterative
visual prompting elicits actionable knowledge for VLMs,” inForty-
first International Conference on Machine Learning, 2024.
[40] P. Sermanet, T. Ding, J. Zhao, F. Xia, D. Dwibedi, K. Gopalakrish-
nan, C. Chan, G. Dulac-Arnold, S. Maddineni, N. J. Joshi,et al.,
“Robovqa: Multimodal long-horizon reasoning for robotics,” in2024
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2024, pp. 645–652.
[41] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karam-
cheti, S. Nasiriany, M. K. Srirama, L. Y . Chen, K. Ellis,et al., “Droid:
A large-scale in-the-wild robot manipulation dataset,” inRobotics:
Science and Systems, 2024.
[42] A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta,
A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar,et al.,
“Open x-embodiment: Robotic learning datasets and rt-x models,”
arXiv preprint arXiv:2310.08864, 2023.
[43] W. Huang, C. Wang, R. Zhang, Y . Li, J. Wu, and L. Fei-Fei, “V oxposer:
Composable 3d value maps for robotic manipulation with language
models,” inConference on Robot Learning. PMLR, 2023, pp. 540–
562.