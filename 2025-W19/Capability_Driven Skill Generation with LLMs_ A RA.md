# Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay

**Published**: 2025-05-06 08:27:04

**PDF URL**: [http://arxiv.org/pdf/2505.03295v1](http://arxiv.org/pdf/2505.03295v1)

## Abstract
Modern automation systems increasingly rely on modular architectures, with
capabilities and skills as one solution approach. Capabilities define the
functions of resources in a machine-readable form and skills provide the
concrete implementations that realize those capabilities. However, the
development of a skill implementation conforming to a corresponding capability
remains a time-consuming and challenging task. In this paper, we present a
method that treats capabilities as contracts for skill implementations and
leverages large language models to generate executable code based on natural
language user input. A key feature of our approach is the integration of
existing software libraries and interface technologies, enabling the generation
of skill implementations across different target languages. We introduce a
framework that allows users to incorporate their own libraries and resource
interfaces into the code generation process through a retrieval-augmented
generation architecture. The proposed method is evaluated using an autonomous
mobile robot controlled via Python and ROS 2, demonstrating the feasibility and
flexibility of the approach.

## Full Text


<!-- PDF content starts -->

Capability-Driven Skill Generation with LLMs:
A RAG-Based Approach for Reusing Existing
Libraries and Interfaces
Luis Miguel Vieira da Silva∗, Aljosha K ¨ocher∗, Nicolas K ¨onig∗, Felix Gehlhoff∗, Alexander Fay†
∗Institute of Automation Technology
Helmut Schmidt University, Hamburg, Germany
Email: {miguel.vieira, aljosha.koecher, felix.gehlhoff }@hsu-hh.de
†Chair of Automation
Ruhr University, Bochum, Germany
Email: alexander.fay@rub.de
Abstract —Modern automation systems increasingly rely on
modular architectures, with capabilities and skills as one solution
approach. Capabilities define the functions of resources in a
machine-readable form and skills provide the concrete implemen-
tations that realize those capabilities. However, the development
of a skill implementation conforming to a corresponding capa-
bility remains a time-consuming and challenging task. In this
paper, we present a method that treats capabilities as contracts
for skill implementations and leverages large language models to
generate executable code based on natural language user input. A
key feature of our approach is the integration of existing software
libraries and interface technologies, enabling the generation
of skill implementations across different target languages. We
introduce a framework that allows users to incorporate their
own libraries and resource interfaces into the code generation
process through a retrieval-augmented generation architecture.
The proposed method is evaluated using an autonomous mobile
robot controlled via Python and ROS 2, demonstrating the
feasibility and flexibility of the approach.
Index Terms —Capabilities, Skills, Large Language Models,
LLMs, Ontologies, Semantic Web, Code-Generation, Retrieval-
Augmented Generation, RAG
I. I NTRODUCTION
Modern industrial automation systems increasingly require
modularity and reconfigurability to flexibly adapt to changing
tasks, environments and system configurations [1]. Whether in
manufacturing or multi-robot scenarios, such systems typically
consist of heterogeneous resources and components from var-
ious vendors, leading to considerable integration complexity.
Each resource often exposes proprietary interfaces, protocols,
and data formats, which must be harmonized to ensure reliable
coordination and control. A key requirement for seamless inte-
gration is the availability of machine-interpretable descriptions
of each resource’s functionality [2].
A common concept to realize this functionality description
and implementation is the use of capabilities and skills . A
capability provides an abstract, technology-independent de-
scription of a function a resource can offer (e.g., grasping
an object), while a skill represents its concrete, executable
implementation, exposed via a skill interface [3]. This dis-tinction enables automated planning and orchestration based
on abstract capabilities, while the execution is delegated to
specific skills provided by the resources [4].
Despite the potential of capabilities and skills, two ma-
jor challenges remain for practical adoption. First, creating
machine-interpretable capability models is complex and error-
prone. Prior work has already demonstrated that structured
engineering methods support the modeling of capabilities and
mitigate these challenges (e.g., [5]). The second open chal-
lenge is that manually developing skills is time-consuming,
error-prone, and requires deep technical knowledge about
the target resource, its control interfaces and libraries. Even
when a capability model is available, identifying appropriate
control interfaces and implementing robust behaviors involves
significant engineering effort. This lack of automation slows
down deployment and limits scalability in dynamic or modular
environments where new functions must frequently be added
or reconfigured.
Addressing this challenge requires methods for automati-
cally generating skill implementations from formal capabil-
ity models. Recently, Large Language Models (LLMs) have
shown strong performance in generating code from natural
language instructions, both in general application program-
ming and for more specialized control languages [6], [7].
Based on these findings, LLMs offer a promising solution to
bridge the gap between capability specification and executable
skills. This paper presents an engineering approach that uses
capability models and user-defined intents as input to auto-
matically generate skill implementations. The generated skills
integrate existing libraries and resource-specific interfaces. To
break down this objective, the following research questions
are addressed:
1) How can capability models be used to generate skill
skeletons that reflect the capability constraints?
2) How can executable skill behavior be synthesized from
user intent and capability context using LLMs?
3) How can existing system interfaces and libraries be
leveraged during code generation?arXiv:2505.03295v1  [cs.AI]  6 May 2025

The remainder of this paper begins with an overview of
related work in Section II. Section III introduces the LLM-
Cap2Skill method in detail. Section IV presents the evaluation
setup and results. The paper concludes with a discussion and
outlook in Section V.
II. R ELATED WORK
While research on capabilities and skills has a history of
almost three decades, initial efforts to standardize the relevant
terms and create a unified model were only presented in [3].
In this work, a reference model defined by a working group
is proposed, which defines the core elements of capabilities
and skills and represents them using an abstract UML model.
This foundational model addresses the previously missing
lack of consistency and provides a conceptual framework
for further developments. In [8], we initially introduced an
OWL ontology to model capabilities and skills. This ontology
has been continuously extended and conforms with the meta
model introduced in [3]. The current version of the ontology is
structured into three layers: At the most abstract layer, the CSS
ontology offers a one-to-one implementation of the reference
model in [3]. The CaSk ontology is an extension of CSS, which
extends abstract concepts (e.g., CapabilityConstraint
andStateMachine ) by using separate standard-backed on-
tologies. For this paper, the CaSk ontology acts as the semantic
foundation to represent capabilities.
Even with a reference model, one issue with capabilities and
skills still is the additional effort to develop resource functions
as skills and create the machine-interpretable models [4]. In
[4], abstract methods to create capabilities are described, but
no concrete solutions to generate capability models or skill
implementations are given. For such model or code generation
tasks, LLMs have proven highly promising [6].
We presented an LLM-based engineering method to gener-
ate capabilities from natural-language task descriptions in [5].
However, this method does not generate skill implementations.
The present paper pursues a different goal: generating skill
implementations based on given capability descriptions.
In recent years, a variety of works has emerged that utilize
LLMs for code generation, and which can be extended for
skills. In [9], Chauhan et al. use an OpenAPI specification as
a blueprint to derive endpoints, schemas, routes, and validation
directly from it. The approach in [9] is designed for general IT
requirements and doesn’t cover industrial-automation needs,
e.g., directly interfacing with hardware. Also, it cannot make
use of existing libraries unknown to the LLM.
The study in [7] was one of the first to prove that LLMs can
also handle the specialized domain of automated control [7]. In
this study, ChatGPT was used to generate control code for 100
tasks. The results indicate that LLMs can produce correct and
executable code, sometimes even adding relevant contextual
knowledge. However, the generated solutions require careful
validation to ensure correctness [7].
While the approach in [7] relies on rather simple prompting
to generate control code with ChatGPT, LLM4PLC is a more
advanced pipeline that combines prompt engineering, modelfine-tuning, and automated verification. This results in a higher
code quality and ensures compliance with the strict correctness
and safety requirements of industrial applications [10].
More recently, Agents4PLC [11] introduces a multi-agent
framework that fully automates control code generation and
verification. Unlike LLM4PLC, which focuses on design-level
checks, Agents4PLC performs formal code-level verification
in a closed-loop workflow, improving autonomy and reliability.
In [12], Koziolek et al. investigate how control code can be
generated using LLMs with consideration of existing libraries.
Since these libraries are not inherently known to the LLM,
Koziolek et al. employ a Retrieval-Augmented Generation
(RAG) approach. RAG combines LLMs with a retrieval com-
ponent that searches a vector database of embedded informa-
tion. At inference time, the user query is embedded, relevant
documents are retrieved based on similarity, and the language
model generates its output conditioned on both the query
and the retrieved context. RAG enhances factual accuracy,
allows access to up-to-date or domain-specific knowledge, and
reduces hallucinations in the generated output. The approach
in [12] focuses solely on generating control code targeting
the IEC 61131 standard and its only input is a user task,
i.e., it does not make use of an existing specification such
as in [9]. In this paper, we present a similar approach to [12]
that conforms to a capability description as a contract and
enables code generation across multiple target languages and
supporting different libraries and existing interfaces. This is
achieved by explicitly incorporating a step to embed library
or interface information into our method.
While there are promising LLM-based code-generation ap-
proaches, many fall short for industrial automation: most
ignore existing model-based specifications; some lack support
for hardware interfaces or libraries; and others are hard-coded
to a single language or framework, lacking extensibility.
III. M ETHOD – LLMC AP2SKILL
The manual development of skill implementations requires
in-depth knowledge of the hardware in use and its correspond-
ing control interfaces. In addition, it demands a detailed under-
standing of the structural and behavioral aspects of a skill itself
– such as its state machine logic, skill interface for interaction,
and ontological representation. To automate this challenging
and time-consuming task, we present LLMCap2Skill , our
method for generating skills. It consists of three main steps:
(1) user input, (2) generation of an API documentation, and
(3) automated RAG-based skill generation using an LLM.
The first essential input is the capability ontology, which
provides an abstract description of a function by specifying
its inputs and outputs as products or information, along
with their properties. This capability description serves as
the structural basis for the implementation of the skill – the
executable counterpart of the abstract capability. However, the
capability alone is not sufficient for code generation, as it
does not include any information about the concrete behav-
ior required for execution. Therefore, an additional, simple
natural language skill specification is required as user input in

step 1, detailing the intended behavior of the skill to enable
meaningful implementation.
While the capability description defines the structural frame
and the skill specification provides the intended behavior, it
still remains unclear how to interact with the resource that
provides both the capability and the final skill. To enable
actual control, additional information about the resource and
its interfaces is required. Therefore, in step 2, the resource
interfaces are automatically identified and structured into an
API document to be used in the subsequent code generation.
In step 3, the actual skill implementation is generated
using RAG. To do this, the method automatically searches
the previously generated API document for resource interfaces
that best match the given capability. These relevant resource
interface descriptions, together with the capability ontology
and the skill specification, are passed to the LLM. In addi-
tion, we leverage the pySkillUp framework developed in our
previous work [13]. This framework introduces annotations
for specifying skill metadata, parameters, and states, enabling
the automatic generation of the ontology, skill interface, and
state machine. Together, these prompt inputs allow the LLM
to generate a complete and valid skill implementation that
conforms to the defined behavior and technical context. The
algorithms that implement the method and results of the
evaluation are available online1.
To demonstrate the method, we focus on a representative
implementation using the Robot Operating System 2 (ROS 2)
framework and the Python programming language. This com-
bination is very common in autonomous mobile robots and
provides a practical foundation for demonstrating the method
in a realistic system context. The following subsections explain
each step of the method in detail, concluding with an outline of
its extension to other frameworks and programming languages.
A. User Input
To generate an executable skill from an abstract capability
description, additional user input is required in step 1. The
required user inputs include:
•the capability ontology itself,
•a skill specification describing the intended behavior,
•the programming language: Python, and
•the name of the target implementation framework: ROS 2.
The skill specification is essential, as the capability does not
contain any information about the expected behavior. Only in
very simple cases – such as getposition – can the behavior
be directly inferred from the capability name and structure by
the LLM. For most abstract capabilities, however, an explicit
skill specification is necessary, as a skill may require multiple
control actions that cannot be derived from the capability
description alone. The skill specification consists of two parts,
which are entered through a structured input form designed to
simplify the process: First, metadata such as the skill interface
type through which the skill should be invoked (e.g., REST),
which can be selected from a drop-down menu listing all
1https://github.com/CaSkade-Automation/Cap2Skillsupported types. Additionally, users may provide an optional
natural language description of the skill to improve clarity and
documentation, though it is not required for skill generation.
Second, the form allows users to define a behavioral definition
of the skill logic within each state of the skill’s state machine,
which follows the standardized PackML state machine as
defined in the CaSk ontology. For each state, a text input can
be used to describe the expected behavior in natural language.
While behavior can be defined for all states, specifying the
behavior for the execute state is mandatory, as it represents
the core functionality.
Furthermore, the programming language Python and target
framework ROS 2 determine the automated discovery of avail-
able ROS 2 interfaces in step 2 and subsequent implementation
of executable skills with ROS 2 in Python in step 3. All
user inputs are used in the final prompt to guide the LLM
in generating an appropriate skill implementation.
B. API Document Generation and Preparation
To generate a skill implementation for a given capability,
knowledge of how to control the underlying resource is
essential. Therefore, the resource interfaces – in this case, the
available ROS 2 interfaces – are automatically queried. The
resulting resource interfaces are then automatically structured
into a consistent API documentation used for the next step.
To retrieve all available ROS 2 interfaces, we developed
a script that launches a dedicated ROS 2 node. This node
automatically queries the resource for all active topics, ser-
vices, and actions, and compiles the results into a structured
JSON document. For each interface, the script records its
name and message type, along with detailed information about
the corresponding message definition – including all relevant
parameters required to interact with the interface. In order to
use this script, the node must be executed within the same
ROS 2 network as the resource to be analyzed, as it relies
on runtime introspection to access the resource’s available
interfaces. The extracted information is automatically stored in
objects and organized into a dictionary of resource interfaces.
This step is performed once during system integration and
is not required at runtime. The resource interface report is
persistent and can be reused across multiple skill generations
for the same resource.
As a preparation for the subsequent RAG step, we use
an LLM to generate uniformly structured descriptions of all
collected resource interfaces. The original resource interface
representations vary significantly in terms of length, level of
detail, and formatting – ranging from brief type definitions
to complex multi-level message structures. This heterogeneity
makes direct similarity comparison with capability descrip-
tions unreliable for identifying relevant interfaces required to
implement the skill. To address this, each resource interface
is enriched with a compact, consistently structured natural
language description generated by an LLM. The prompt used
for generating resource interface descriptions instructs the
LLM to include the following elements: (1) the resource
module to which the interface relates, (2) type of tasks for

which the interface is relevant, and (3) typical entities that use
or interact with the interface. These descriptions are later used
as input for the retrieval process in step 3.
Additionally, we introduce an optional automated resource
interface relevance check to determine whether a given in-
terface is actually involved in controlling the resource. The
relevance check is also performed by an LLM. Resource
interfaces are considered irrelevant if they are used exclusively
for debugging, logging, introspection or metadata exchange.
In contrast, resource interfaces are deemed relevant if they
actively participate in control processes, receive control-related
messages, or are part of known control mechanisms. If a
resource interface is classified as irrelevant, it is excluded
from the RAG step, reducing the size of the search space
and improving efficiency. In the following step, both the full
interface specifications and their natural language descriptions
are used for skill generation.
C. Skill Generation with Retrieval-Augmented Generation
In the final step of the proposed method, the actual skill
implementation is generated as shown in Figure 1. To iden-
tify the most relevant resource interfaces for the capability
in question, the previously generated API documentation is
processed using a RAG approach. The capability description,
skill specification, and the retrieved resource interfaces are
then combined into a structured prompt, which is submitted
to an LLM to implement the final skill.
From the generated API documentation, only the uniform
textual descriptions of resource interfaces – previously gener-
ated by the LLM – are extracted and processed individually
as input chunks for embedding. The reduction to the use
of descriptions instead of the entire resource interfaces for
RAG addresses two issues: (1) overlaps in parameter usage
across interfaces, and (2) significant differences in interface
lengths. Both aspects can negatively affect the accuracy and
reliability of the similarity search. Similarly, from the capabil-
ity ontology, only rdfs:comment , i.e., a natural-language
description, is extracted as input for the embedding model. As
a result, both resource interface descriptions and capability
definitions are reduced to natural language format before
embedding.
The embedding model transforms each input text into a
compact vector that captures its semantic meaning. These
vectors are stored in a vector database, which enables fast
similarity searches in a high-dimensional vector space. In
the next step, similarity search is performed to identify the
resource interfaces most relevant to the given capability. Re-
source interface and capability descriptions are compared in
the embedding space, where semantic closeness is measured
using distance metrics such as cosine similarity. This process
narrows the complete set of resource interfaces down to a few
candidates that are most likely relevant for implementing the
given capability. The full resource interface documentations
are selected based on the retrieved descriptions to provide the
LLM with the necessary information to invoke and interact
with the interfaces in the subsequent prompt used for skillgeneration. The embedding and retrieval functionality was
implemented using the open-source library LangChain2.
The resulting set of relevant resource interfaces is then com-
bined with the complete user input – including the capability,
programming language, framework, and skill specification –
and used to construct a prompt for the LLM. The prompt
also includes default instructions on how to structure the skill
implementation. To further support skill automation regarding
ontology, skill interface, and state machine generation, the
prompt is extended with additional framework-specific guid-
ance by including an explanation of the pySkillUp framework.
As a result, the LLM can generate a complete and valid skill
that conforms to the framework specification. Additionally, the
prompt follows a few-shot prompting technique by including
three examples. Few-shot prompting refers to the strategy
of providing the LLM with a small number of input–output
examples to guide the generation process. Each example
contains a capability, a skill specification, and the resulting
implementation. The complete prompt is finally submitted to
the LLM, and the generated skill code is returned to the user.
D. Method Generalizability
While the method has been exemplified for Python and
ROS 2, it is designed to be extensible and adaptable to other
programming languages and frameworks. The user input in
step 1 includes the specification of the target programming
language and framework, which determines how the skill will
be implemented and how resource interfaces are discovered
and documented. Resource interfaces can either be provided
directly in step 1 – for example, as a structured document
or a PDF – or derived in step 2 using custom extraction
routines. If the resource exposes its interfaces at runtime –
for example via introspection mechanisms or standardized
service registries – this information can be automatically
queried within the resource’s network environment and gen-
erate a structured API documentation. Therefore, we provide
a modular and extensible architecture to support alternative
frameworks. Specifically, a set of abstract base classes as
templates to support the documentation process, as illustrated
in Figure 2. The ControlEntity class defines the structure
of an individual control interface and can be extended for
framework-specific cases (e.g., ROS2ControlEntity for
ROS 2). Given the complexity of interface structures in ROS 2
– where communication with an entity is implemented via spe-
cific message types – we further define the ROS2Interface
class, which stores details such as message parameter types.
The second main class, APIDocumentHandling , is re-
sponsible for managing all resource interfaces and interact-
ing with the LLM. When using these templates, developers
must implement the specific querying and storage logic with
generate_api_doc andparse_api_doc methods, as
demonstrated in the ROS2APIHandling implementation.
Our ROS 2 implementation extracts all available topics, ser-
vices, and actions from the running ROS 2 resource in method
2https://python.langchain.com/docs/introduction/

Fig. 1. Skill generation using RAG by retrieving relevant resource interfaces and combining them with user input to construct a prompt for the LLM.
generate_api_doc , parses them into structured interface
objects and stores them in the dictionary-based report in
parse_api_doc .
Fig. 2. Class structure for resource interface documentation templates and
ROS 2 specific implementations.
In the final step of the method, skill generation is adapted
based on the selected programming language. Where available,
an appropriate skill framework is included in the prompt to
support automatic generation of skill interface, ontology, and
state machine. For example, pySkillUp is used for Python,
andskillUp is available for Java. In these cases, the prompt
includes a short explanation of the respective framework, as
described in Subsection III-C. For other languages without apredefined skill framework, the user may provide their own
skill framework structure. If no skill framework is used, only
the core behavior of the skill is generated, which can then be
manually integrated into the desired execution environment.
IV. E VALUATION
In this section, we evaluate the proposed method LLM-
Cap2Skill with regard to its ability to generate executable
and functionally correct skills for a mobile robot programmed
using ROS 2.
A. Setup and Procedure
To systematically assess the effectiveness of LLM-
Cap2Skill , an evaluation framework was designed that covers
both technical correctness and functional performance of the
generated skills. The evaluation setup includes the following
core components, which are described in more detail in the
subsequent paragraphs:
•Platform: Mobile robot (Neobotix MMO-700), simulated
in Gazebo with ROS 2.
•Capabilities: Nine capabilities modeled using the CaSk
ontology in Turtle syntax, which are listed in Table I.
•Target language and framework: Python and ROS 2.
•LLMs: gpt-4o ando3-mini .
•Prompting method: few-shot prompt (three examples).
•Embeddings: text-embedding-3-large stored in a Chroma
vector store.
•Similarity search: Cosine similarity (top four results).

In the first step of the proposed method, each of the nine
capabilities of varying complexity was modeled, providing the
expected inputs, outputs and constraints for the corresponding
skill. Three of nine capabilities – get-position ,set-velocity ,
andnavigate-to-point – serve as few-shot examples within the
prompt for skill generation. They represent different ROS 2
interaction patterns: (1) get-position requires subscription to a
topic, (2) set-velocity requires both publishing and subscribing,
as velocity commands are published to a control topic, while
feedback is received via subscription to verify the applied
velocity, (3) navigate-to-point is the most complex, requiring
the use of an action client and handling various feedback and
result messages. The remaining six capabilities target various
aspects of robot operation, such as navigation, mapping, and
manipulation, and were used as input for skill generation.
TABLE I
OVERVIEW OF THE CAPABILITIES USED
Name Description Inputs Outputs
E1: Set Velocity Set robot’s velocity in for-
ward directionvelin velout
E2: Get Position Retrieve robot’s position - posout
E3: Navigate to
PointNavigate robot to a de-
sired goal pointposin posout
C1: Move
ForwardSet robot’s velocity based
on desired distance and
travel timedist in
time indist out
time out
velout
C2: Get Object Retrieve distance and ori-
entation to robot’s nearest
object- obsdist
obsdegree
C3: Rotate Set robot’s angular veloc-
ity until desired orienta-
tion is reacheddegree in
velindegree out
C4: Collision
AvoidanceMove robot with specific
motion pattern for a set
time and stop if obstacle
distance falls below minvelin
min dist
time invelout
obsdist
obsdegree
time out
C5: Mapping Map desired area by mov-
ing the robot in that areaarea in map out
C6: Move to
PointMove manipulator to a de-
sired goal pointposin posout
For each capability, a corresponding skill specification was
defined, including a description of the intended behavior
within the states of the state machine. For instance, in the
case of the move-forward skill, the behavior of the execute
state was specified as follows:
Set the velocity of the robot to a calculated value
for the desired time only and then reset it. If the
calculated velocity exceeds the maximum velocity,
set the maximum velocity and calculate the new time
necessary to travel the desired distance.
As part of the second step in our method to provide technical
control information, the API documentation for the robot was
generated once with ROS 2. During the documentation phase,
approximately 60 resource interfaces were excluded due to
being irrelevant for control-related tasks. For the remaining∼130resource interfaces (e.g., cmd_vel orodom ), natural
language descriptions were generated using gpt-4o .
Based on this documentation, the capabilities and the skill
specifications, as well as the user-defined target programming
language Python and framework ROS 2, six skill implementa-
tions were generated in the last step of our method using two
different LLMs: gpt-4o ando3-mini , both leveraging the RAG
mechanism and few-shot prompting. To ensure deterministic
behavior, gpt-4o was configured with a temperature of 0 and
toppset to 1. In the future, the underlying LLM can be
flexibly exchanged by the user (e.g., replaced by Claude).
For embedding and retrieval within the RAG process,
OpenAI’s text-embedding-3-large model and cosine similarity
search were applied, with embeddings stored in a Chroma
vector store. The retrieval returns the top four most relevant
resource interfaces based on the embedding of the natural
language resource interface descriptions (chunks) and the
natural language capability description as query. For the
move-forward capability, the following resource interfaces
were returned: cmd_vel ,odom ,joint_trajectory , and
follow_joint_trajectory . Among these, the two re-
quired interfaces ( cmd_vel andodom ) were correctly iden-
tified.
These selected resource interface descriptions, along with
all other user inputs and the few-shot examples (capabil-
ity, skill specification, and skill implementation), were then
passed into the prompt for code generation. Listing 1 shows
Listing 1. Excerpt of move forward skill generated with gpt-4o
1@skill(skill_interface=SkillInterface.REST,
skill_iri="...mmo700/skills/moveForward"...)
2class MoveForwardSkill(ROS2Skill):
3
4 def __init__(self):
5 super ().__init__(’move_forward’)
6 self.velocity_publisher = None
7 ...
8
9 @skill_parameter(is_required=True, ...)
10 def get_desired_distance(self) -> float :
11 return self.desired_distance
12
13 @execute
14 def execute(self) -> None:
15 desired_velocity = self.desired_distance /
self.desired_time
16 ifdesired_velocity > self.MAX_VELOCITY:
17 ...
18
19 self.velocity_cmd.linear.x =
desired_velocity
20 end_time = self.start_time + self.
desired_time
21
22 while time.time() < end_time:
23 self.velocity_publisher.publish(self.
velocity_cmd)
24 time.sleep(0.1)
25
26 self.velocity_cmd.linear.x = 0.0
27 self.velocity_publisher.publish(self.
velocity_cmd)

a selected excerpt from the resulting skill implementation
move forward generated with gpt-4o . The excerpt illustrates
the implementation of a class that inherits from the base class
ROS2Skill provided by the pySkillUp framework. This base
class includes ROS 2-specific functionalities such as managing
node startup and shutdown. The @skill annotation is used
to mark the class as a skill and includes the skill interface
used for interacting with the skill as well as a generated
skill_iri . The excerpt also shows an example parameter
desired_distance representing the target distance that
the robot should travel. Finally, the execute method is
included, in which the required velocity is calculated based
on the desired distance and duration. If the computed velocity
exceeds the maximum value, the method handles this case ac-
cordingly. Otherwise, the velocity is published for the specified
time and then reset to zero.
B. Results and Discussion
Each generated skill was evaluated in terms of structural
correctness and functional behavior in a Gazebo simulation of
the target mobile robot based on the following criteria:
•Syntax correctness of the generated code.
•Annotation completeness: Coverage and correctness of
all required annotations, including skill metadata, param-
eters, outputs, and state machine states.
•Executability: Successful execution of the generated
code without runtime errors.
•Interface usage: Correct identification of relevant re-
source interfaces during retrieval and their appropriate
integration into the generated skill implementation.
•Behavioral accuracy: Correct implementation of the
intended capability behavior, in particular the behavior
specified by the user within the state machine states.
•Lines of code: Code length as a rough indicator of
implementation size and complexity.
•Manual effort: Number of modified lines required to
achieve the correct behavior.
The overall results were promising, with consistently well-
structured skill skeletons and most implementations exhibiting
a high degree of behavioral correctness. A summary of the
evaluation results is provided in Table II.
Although a direct comparison between o3-mini and gpt-
4owas not the focus of this evaluation, o3-mini consis-
tently demonstrated superior performance across most tasks. It
tended to produce more comprehensive code, characterized by
more robust handling of edge cases and enriched with explana-
tory comments that improved readability and transparency.
The structural code skeleton including syntax and annota-
tion completeness was correctly generated for all skills except
collision-avoidance generated with gpt-4o , where one of four
required parameter annotations was missing. Instead, the LLM
inserted a hard-coded default value of 0.5for the minimum
obstacle distance. This value was not mentioned in the original
capability and appears to have been hallucinated.
All generated skills were executable in ROS 2, with one
exception: the skill move-to-point generated with gpt-4o failedat runtime. The LLM inserted placeholders instead of critical
code segments, likely due to insufficient knowledge about the
required logic, leading to execution failure. This particular skill
is a complex task, and it is noteworthy that the LLM omitted
rather than incorrectly implemented the unknown logic.
Regarding resource interface usage, four of six skills were
implemented with the correct resource interfaces, both in
retrieval and in code. For the collision-avoidance skill, one
of the three necessary resource interfaces for reading sensor
data was not retrieved via RAG, but was nonetheless used
correctly by both LLMs. This suggests that the LLM relied
on its pretrained internal knowledge in addition to explicit
retrieval in this case. Similarly, in the mapping skill a motion-
related resource interface was used without being retrieved,
likely due to patterns observed in the few-shot examples.
Furthermore, within the same mapping skill generation, one
additional resource interface relevant for obstacle detection
was neither retrieved nor used in the implementation, resulting
in missing sensor feedback handling. A notable difference
was observed between the two LLMs: while o3-mini used the
correct resource interface for accessing the map data, gpt-4o
used an unrelated resource interface, leading to an incorrect
implementation of this part of the functionality.
Behavioral correctness was fully achieved in four out of six
skills when considering both LLMs together. In the case of
therotate skill, o3-mini produced a correct implementation,
whereas gpt-4o generated code that was functionally close
but contained a critical flaw. Specifically, the calculation of
the target rotation angle was inaccurate leading to incorrect
rotation behavior. Fixing this issue required implementing a
normalization step for the angular difference, resulting in 25
lines of modified code.
Interestingly, both models generated largely correct imple-
mentations of the complex collision-avoidance skill. The robot
executed the desired motion pattern, and stop logic for obstacle
avoidance was mostly implemented: o3-mini missed only a
single line that explicitly triggered the stop transition, although
the obstacle detection condition was correctly implemented.
In contrast, the two most complex skills – mapping and
move-to-point – did not behave as intended. For mapping ,
both LLMs generated basic exploration behaviors: o3-mini
produced a snake-like movement pattern, while gpt-4o fol-
lowed a simpler strategy of moving forward for the desired
distance and then rotating 90 degrees. However, in both cases,
the robot failed to avoid obstacles and collided with them.
The corresponding feedback resource interface was neither
retrieved nor implemented, likely due to an underspecified
behavior description in the user input. In the case of o3-mini ,
the missing obstacle avoidance was corrected by adding logic
to adjust the robot’s heading based on sensor feedback. For
gpt-4o , more changes were necessary, including replacing in-
correct map access and implementing basic avoidance strategy.
While both implementations required noticeable changes, the
total number of modified lines remained manageable.
The move-to-point skill, as previously noted, was not exe-
cutable in the version generated by gpt-4o due to some missing

TABLE II
EVALUATION RESULTS FOR SIX GENERATED SKILLS USING TWO LLM S:gpt-4o AND o3-mini . EACH CELL SHOWS THE RESULT AS gpt-4o /o3-mini .
Syntax Annotation completeness Executability Interfaces Behavior Code lines Manual effort
Skill Parameters Outputs States (code lines)
S1: Get Object  /  / - /  /  /  /  / 59 / 82 0 / 0
S2: Move Forward  /  /  /  /  /  /  /  / 99 / 129 0 / 0
S3: Rotate  /  /  /  /  /  /  / G # / 95 / 183 25 / 0
S4: Collision Avoidance  /  / G # /  /  /  / G # /G #  /G # 115 / 199 0 / 1
S5: Mapping  /  /  /  /  /  / # /G # # /G # 98 / 149 45 / 35
S6: Move to Point  /  /  /  /  / # /  / # /G # 154 / 224 25 / 3
implementation segments and robot-specific identifiers. As a
result, the logic for interacting with the action server remained
incomplete. In contrast, the implementation generated by o3-
mini was functionally correct, but a few robot-specific iden-
tifiers had to be manually adjusted due to the lack of such
specific information in the prompt. After replacing these names
with the correct ones, the skill executed as intended.
Overall, the evaluation shows that LLMCap2Skill can
reliably generate executable and structurally correct skill im-
plementations for complex tasks from capability ontologies.
While minor manual corrections were still necessary – particu-
larly in the implementation of behavioral logic and integration
of robot-specific elements – the core structure was consis-
tently well-formed, and the intended functionality could in
most cases be achieved with manageable effort. Crucially, the
method eliminates the need for manual skill implementation
from scratch, allowing developers to shift their focus from
boilerplate coding to targeted refinements. In combination
with skill frameworks such as pySkillUp , this also includes
the automatic generation of the ontology, the underlying state
machine, and the skill interface required for interacting with
the skill. The combination of capabilities, RAG-based resource
interface selection and code generation proved effective for
translating abstract capabilities into working skills.
V. C ONCLUSIONS AND FUTURE WORK
This paper introduced a novel method for generating ex-
ecutable skills from ontological capability descriptions using
LLMs and a RAG approach. By combining semantic capa-
bility modeling with contextual retrieval of resource interface
documentation, our method enables the automated generation
of skill implementations that integrate existing libraries and
frameworks. The evaluation with a mobile robot platform
demonstrated that the generated skills are largely correct,
executable, and significantly reduce manual coding effort.
In the future, we plan to evaluate the method for additional
frameworks beyond ROS 2, including the use of SkillUp
with Java as well as PLC code, targeting alternative system
interfaces and libraries. We also plan to enhance the retrieval
step of the RAG process by incorporating richer metadata and
exploring more advanced retrieval strategies. Another promis-
ing direction is the automated refinement of skill behavior
through simulation feedback or user interaction. Furthermore,
integrating formal verification tools could enhance trust in the
generated code and ensure safety in critical applications.ACKNOWLEDGMENT
This research in the RIV A project is funded by dtec.bw
– Digitalization and Technology Research Center of the
Bundeswehr. dtec.bw is funded by the European Union –
NextGenerationEU
REFERENCES
[1] Y . Koren, U. Heisel et al. , “Reconfigurable Manufacturing Systems,”
CIRP Annals , vol. 48, no. 2, pp. 527–540, 1999.
[2] A. Perzylo, J. A. Grothoff et al. , “Capability-based semantic interop-
erability of manufacturing resources: A BaSys 4.0 perspective,” IFAC-
PapersOnLine , vol. 52, no. 13, pp. 1590–1596, 2019.
[3] A. K ¨ocher, A. Belyaev et al. , “A reference model for common un-
derstanding of capabilities and skills in manufacturing,” at - Automa-
tisierungstechnik , vol. 71, no. 2, pp. 94–104, 2023.
[4] J. Bock, C. Diedrich et al. (2025) Capabilities,
Skills and Services. Plattform Industrie 4.0. [Online].
Available: https://www.plattform-i40.de/IP/Redaktion/DE/Downloads/
Publikation/2025-i40-capabilities.pdf? blob=publicationFile&v=5
[5] L. M. Vieira da Silva, A. K ¨ocher et al. , “Toward a Method to Generate
Capability Ontologies from Natural Language Descriptions,” in 2024
IEEE 29th International Conference on Emerging Technologies and
Factory Automation (ETFA) , 2024, pp. 1–4.
[6] S. Fakhoury, A. Naik et al. , “LLM-Based Test-Driven Interactive Code
Generation: User Study and Empirical Evaluation,” IEEE Transactions
on Software Engineering , vol. 50, no. 9, pp. 2254–2268, 2024.
[7] H. Koziolek, S. Gruener, and V . Ashiwal, “ChatGPT for PLC/DCS
Control Logic Generation,” in 2023 IEEE 28th International Conference
on Emerging Technologies and Factory Automation (ETFA) , 2023, pp.
1–8.
[8] A. K ¨ocher, C. Hildebrandt et al. , “A Formal Capability and Skill Model
for Use in Plug and Produce Scenarios,” in 2020 25th IEEE International
Conference on Emerging Technologies and Factory Automation (ETFA) ,
2020, pp. 1663–1670.
[9] S. Chauhan, Z. Rasheed et al. , “LLM-Generated Microservice
Implementations from RESTful API Definitions,” 13.02.2025. [Online].
Available: http://arxiv.org/pdf/2502.09766
[10] M. Fakih, R. Dharmaji et al. , “LLM4PLC: Harnessing Large Language
Models for Verifiable Programming of PLCs in Industrial Control Sys-
tems,” in Proceedings of the 46th International Conference on Software
Engineering: Software Engineering in Practice , 2024, pp. 192–203.
[11] Z. Liu, R. Zeng et al. , “Agents4PLC: Automating Closed-loop
PLC Code Generation and Verification in Industrial Control Systems
using LLM-based Agents,” 18.10.2024. [Online]. Available: http:
//arxiv.org/pdf/2410.14209
[12] H. Koziolek, S. Gr ¨uner et al. , “LLM-based and Retrieval-Augmented
Control Code Generation,” in Proceedings of the 1st International
Workshop on Large Language Models for Code , 2024, pp. 22–29.
[13] L. M. Vieira da Silva, A. K ¨ocher et al. , “A Python Framework
for Robot Skill Development and Automated Generation of Semantic
Descriptions,” in 2023 IEEE 28th International Conference on Emerging
Technologies and Factory Automation (ETFA) , 2023, pp. 1–8.