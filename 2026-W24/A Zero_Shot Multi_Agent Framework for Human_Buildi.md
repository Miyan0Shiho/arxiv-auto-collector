# A Zero-Shot Multi-Agent Framework for Human-Building Interaction via Programmatic Reasoning

**Authors**: Yuqi Wang, Gulai Shen, Ali Mehmani

**Published**: 2026-06-09 18:30:34

**PDF URL**: [https://arxiv.org/pdf/2606.11354v1](https://arxiv.org/pdf/2606.11354v1)

## Abstract
Large Language Model (LLM) offers opportunities to enhance Human-Building Interaction (HBI) by enabling more direct interactions through intuitive interfaces to complex building systems. These systems can be characterized by the vast amounts of data across multiple formats, the lack of nonconfidential and generalizable information, and the requirement of domain expertise for interpretation. Applying LLMs to domain-specific tasks like HBI presents additional challenges. Limited training data makes traditional fine-tuning approaches impractical. Meanwhile, the opacity of LLM training data requires careful integration of domain knowledge to ensure reliability. Additionally, different LLMs exhibit varying alignment characteristics, suggesting that achieving both natural interaction and technical accuracy requires a multi-agent approach. These challenges highlight the need for innovative approaches to adapt LLMs for specialized domains while maintaining accuracy and user engagement. In this paper, we develop a hierarchical multi-agent framework that utilizes semantic routing and programmatic reasoning to decouple natural language understanding from building analytics. Instead of standard RAG approaches, our system employs a "Doorman" mechanism for task decomposition and specialized coding agents that generate executable Python scripts for precise arithmetic. We validate this framework on a dataset from more than 200 commercial buildings. Results demonstrate the effectiveness in providing accurate and contextual responses for diverse users, including stakeholders, from tenants to building managers, across various building system applications.

## Full Text


<!-- PDF content starts -->

Proceedings of the ASME 2026 International Design Engineering Technical Conferences &
Computers and Information in Engineering Conference
IDETC/CIE 2026
August 23-August 26, 2026, Huston, TX
DETC2026-194316
A ZERO-SHOT MULTI-AGENT FRAMEWORK FOR HUMAN-BUILDING
INTERACTION VIA PROGRAMMATIC REASONING
Yuqi Wang1, Gulai Shen2, Ali Mehmani1∗
1Nantum AI, New Y ork, NY 10021
2Harvard University, Cambridge, MA 02138
ABSTRACT
Large Language Model (LLM) offers opportunities to en-
hance Human-Building Interaction (HBI) by enabling more di-
rect interactions through intuitive interfaces to complex build-
ing systems. These systems can be characterized by the vast
amounts of data across multiple formats, the lack of noncon-
fidential and generalizable information, and the requirement of
domain expertise for interpretation. Applying LLMs to domain-
specific tasks like HBI presents additional challenges. Limited
training data makes traditional fine-tuning approaches impracti-
cal. Meanwhile, the opacity of LLM training data requires care-
ful integration of domain knowledge to ensure reliability. Addi-
tionally, different LLMs exhibit varying alignment characteris-
tics, suggesting that achieving both natural interaction and tech-
nical accuracy requires a multi-agent approach. These challenges
highlight the need for innovative approaches to adapt LLMs for
specialized domains while maintaining accuracy and user en-
gagement. In this paper, we develop a hierarchical multi-agent
framework that utilizes semantic routing and programmatic rea-
soning to decouple natural language understanding from building
analytics. Instead of standard RAG approaches, our system em-
ploys a “Doorman” mechanism for task decomposition and spe-
cialized coding agents that generate executable Python scripts for
precise arithmetic. We validate this framework on a dataset from
more than 200 commercial buildings. Results demonstrate the
effectiveness in providing accurate and contextual responses for
diverse users, including stakeholders, from tenants to building
managers, across various building system applications.
∗Corresponding Author: amehmani@prescriptivedata.io1 INTRODUCTION
The complex interplay between humans and their con-
structed surroundings profoundly influences daily life, health,
and overall well-being. As urban populations grow, individuals
increasingly inhabit and operate within built structures, spending
over 90% of their time indoors [1]. The most basic interactions
can occur through operable windows, shades, thermostats, and
lighting switches. The result of interaction can be observed from
the building’s energy use, interior environmental conditions, oc-
cupants’ satisfaction, and more [2].
Furthermore, buildings are becoming more complex, tran-
scending from mere protective shelters to embody intricate de-
signs that modulate behavior, comfort levels, productivity, and
social dynamics. This also leads to a higher barrier to under-
standing and altering building behaviors for occupants, espe-
cially those in large buildings centrally managed with Build-
ing Management and Automation Systems (BMS and BAS)
[2]. With the recent advancement in Artificial Intelligence (AI),
specifically Large Language Models (LLM), new interaction
methods can be formed to enhance the relationship between hu-
mans and the built environment [3].
In the remainder of this section, we introduce the concepts of
Human-Building Interaction (HBI) and Large Language Models
(LLMs) before we combine them to propose the enabled next-
level HBI through LLM.
1.1 Human-building Interaction
HBI encompasses a wide range of interdisciplinary re-
searches aim to understand how humans interact, adapt, and af-
1 Copyright © 2026 by ASMEarXiv:2606.11354v1  [cs.ET]  9 Jun 2026

fect the built environment as well as how the built environment
impacts human outcomes and experiences [4]. Viewing HBI
from the perspective of intersecting Human Computer Interac-
tion (HCI), HBI holds its core principle to enable the two-way
interaction between humans and the built environment, in addi-
tion to the technology involved [3].
While building consists of a full lifecycle of the design, con-
struction, operation, maintenance, and eventual demolition or
reuse [5], the main focus of HBI has been on the operational/use
phase during most of its life, when they are most extensively
interacting with humans, including the management teams, engi-
neers, and occupants. The objectives, methods, and subsystems
that different people interact with can vary significantly, leading
to a wide range of research topics. These include concerns about
human impact on building energy consumption and indoor envi-
ronmental conditions [6], as well as building’s impact on human
health, comfort [7], and overall quality of life [8].
Further, the interaction can happen both passively and ac-
tively through direct or indirect interfaces. Traditionally, when
buildings are composed of mechanical and electrical systems,
humans interact and control buildings directly through manu-
ally opening and closing windows and water valves; turning on
and off lights and fans; and adjusting air conditioning systems
through pneumatic controls. Later, direct digital controls (DDC)
emerged in the 1980s, enabling building engineers and occu-
pants to centrally monitor and control buildings through BMS
and BAS [9]. More recently, with the emergence of cloud com-
puting, IoT sensors, smart devices, people are able to interact
with buildings through smart home applications and building op-
erating systems [10, 11, 12]. With the fast development of AI
technology, automated building management now continuously
monitors and optimizes buildings for energy efficiency and in-
door environmental wellness [13]. However, with more tech-
nology, it also becomes less and less intuitive, requiring more
and more training and expertise to understand what is happening
in buildings, interact with buildings, and influence buildings to
make more personalized changes [2].
1.2 Generative AI to Agentic AI
LLM has rapidly evolved over the past few years, with sig-
nificant milestones marking its development. The journey be-
gan with early models like Word2Vec [14] and GloVe [15] in
the early 2010s, which focused on word embeddings. This was
followed by the introduction of transformer-based architectures,
particularly with the release of Google’s BERT in 2018 [16],
which set the stage for context-aware language processing. Ope-
nAI’s GPT series, starting in 2018, marked another leap forward,
culminating in models like GPT-3 and GPT-4, which demon-
strated unprecedented capabilities in generating coherent and
contextually relevant text. Meanwhile, various companies re-
leased their own large language models, including Anthropic’sClaude, Google’s Gemini, and Meta’s LLaMA, available either
for token purchase or as open source.
LLMs have been applied across industries such as health-
care, finance, education, and entertainment. Their ability to un-
derstand and generate text enables advancements in automated
content creation, virtual assistants, decision support systems, etc
[17]. Beyond basic text generation, pre-trained LLMs have been
explored to be applied across specific downstream tasks, includ-
ing programming code fill-in-the-middle tasks [18] and specific
API calling [19], synthesizing 3D scenes [20], arithmetic calcu-
lation [21], etc., while offering improved efficiency, multilingual
support, and enhanced human-computer interaction [17].
Despite all the progress, applying LLMs to specific domains
remains challenging as LLMs are typically trained on large-scale,
general-purpose open datasets, often lacking the depth and speci-
ficity required for specialized domains [22]. These problems can
be characterized by three main challenges.
• There is limited domain-specific training data and a high
cost for fine-tuning. In many cases, information specific to
the problem is unseen by the model during training or dif-
fers from the specific use case in the actual system. While
fine-tuning has been traditionally applied to address this
gap by retraining several layers or the whole model on
smaller, domain-specific datasets [23], it can be computa-
tionally expensive and may introduce biases when the avail-
able datasets are limited. Moreover, each target system, such
as a building, often differs significantly from others, making
it impractical to rely on a single fine-tuning process to meet
the diverse requirements of all potential users.
• Domain-specific tasks demand a deep understanding of spe-
cialized terms, concepts, and tools. Without such knowl-
edge embedded into or fed to the model, it is likely to get
plausible but incorrect outputs from the general models. For
example, in coding generation tasks, while code interpreters
can enable calculated answers to questions requiring mathe-
matical computation [24], and Retrieval-Augmented Gener-
ation (RAG) can reference authoritative knowledge outside
of original training data [25], challenges remain in designing
and integrating these components efficiently and robustly to
ensure accurate domain-specific responses.
• Pre-trained LLMs’ performance varies based on training
data, model complexity, and prompt design, creating the
alignment challenge of matching the right model to the right
application. While multi-agent systems [26], where mul-
tiple specialized agents work collaboratively, have shown
promise, coordinating multiple agents introduces complex-
ity and potential failures, requiring robust design to optimize
interactions and ensure seamless communication.
Prior research has explored several strategies to address
these challenges in domain-specific LLM applications. Fine-
2 Copyright © 2026 by ASME

tuning approaches, such as the adaptive pruning and tuning strat-
egy proposed by Zhao et al. [27], have shown promise in re-
ducing computational costs, but the fundamental limitation of
scarce domain-specific data is still a challenge in fine-tuning. In
terms of task decomposition, Li et al. [28] developed methods for
breaking down complex tasks into semantic sub-tasks, though
their approach was constrained by relying on a single LLM
rather than leveraging the diverse capabilities of different mod-
els. While Du et al. [29] and Crispino et al. [30] demonstrated
the potential of zero-shot strategies in multi-agent systems for
improved reasoning and logical responses, their work remained
untested in specialized domain applications. These limitations in
existing approaches highlight the need for a comprehensive solu-
tion that combines the strengths of Retrieval-Augmented Gener-
ation (RAG), multi-agent systems, and zero-shot learning while
specifically addressing domain-specific challenges.
1.3 AI Agents and Agentic AI Systems for Buildings
With LLMs, everyone now has the potential to interact more
directly and proactively than ever with the ever-smarter and more
complicated buildings. Many of the existing works have be-
gun by exploring different ways to implement LLM to provide
more user-friendly interfaces for complicated problems related
to buildings. The applications range from architectural design
and conceptualization, building modeling, building construction,
to building code compliance, shown in detail in Table 1.
By targeting different areas, these works that integrated
LLM enabled humans with different roles to interact with build-
ings more efficiently, proactively, and directly. The required level
of expertise is also lowered, while the workload for profession-
als can be significantly reduced, especially on repetitive and non-
creative tasks.
These works, emerging in a very short period of time since
models like GPT and Claude became available, further high-
light the promising potential for HBI to be elevated to more
closely connect humans, buildings, and the technologies around
them. However, a structured system to organize and teach LLMs
about different aspects of buildings is lacking, as building in-
formation is scattered across public or private databases, codes,
applications, documents, and the Internet. Further, a consoli-
dated interface is needed to capture human instructions and sur-
face processes and results. Lastly, a significant amount of needs
from building managers and occupants remained unsolved by the
above-mentioned works as they often require access to a wide
range of information in addition to the expertise knowledge, as
well as tools like code interpreters and a user-friendly interface.
2 METHOD AND DESIGN
To address these challenges in the context of Human-
Building Interaction, we propose a zero-shot LLM-based multi-agent system framework that integrates real-time databases, code
repositories, and technical documents. Our framework specif-
ically tackles: (1) the limited training data challenge through
a zero-shot mechanism that eliminates the need for domain-
specific training, (2) the domain expertise requirement via RAG
integration with building-specific documentation and real-time
data, and (3) the alignment challenge through a carefully de-
signed multi-agent collaboration system that matches different
types of queries to specialized agents. In the following section,
we examine the domain of HBI and its specific requirements for
LLM integration, which is the key motivation for the develop-
ment of our framework. The developed framework, shown in
Figure 1, for the LLM-based multi-agent system comprises three
key components: agents, tools, and databases. Agents are cate-
gorized as a Doorman agent, a world knowledge agent, coding
agents, and specialist agents. The Doorman agent has two pri-
mary tasks: (i) splitting the user query into sub-tasks and assign-
ing each to the appropriate agent, and (ii) collecting responses
from the agents and providing the final answer. Each agent is
equipped with tools such as code interpreters, RAG modules,
and LLMs for language-generative tasks. They also access rel-
evant databases, including coding databases, world knowledge
databases, and technical documentation, to obtain necessary do-
main knowledge.
2.1 Agents
Doorman agentThe Doorman agent acts as the central coor-
dinator within the multi-agent system. The agent does not have
the ability to directly solve any question, while it holds knowl-
edge about the capabilities and limitations of all other agents.
Its primary responsibilities include parsing the user’s query and
breaking it down into manageable subtasks, and assigning these
subtasks to the most suitable agents based on their specialized
capabilities. Once the agents complete their tasks, the Door-
man agent collects the responses, synthesizes the information,
and formulates the final answer to the user’s query.
Raw data agentRaw data agents handle the data retrieval
based on the questions and subtasks from the Doorman Agent.
It has a thorough understanding of the database structure, types
of data stored, metadata information, and data relations. It uses
tools like an interpreter and RAG to securely access the database
and retrieve required data.
Coding agentCoding agents specialize in handling
programming-related queries. It utilizes tools like code inter-
preters and domain-specific application programming Interfaces
(API) to generate code for answering logical questions.
Specialist agentsSpecialist agents are domain experts de-
signed to handle queries that require deep technical knowledge in
3 Copyright © 2026 by ASME

TABLE 1: LLM Implementations in Buildings
Area Summary of works
Building construction Scheduling, planning, resource allocation [31, 32, 33].
Hazard recognition and safety training [34].
Regulatory compliance and documentation [33].
Building Information Modeling (BIM) BIM model generation [35].
Building code compliance check and reports [36, 37].
Retrieving building information [38].
Building Energy Modeling (BEM) BEM software tutorial and training [39].
Single object generation and model file modification [40].
Running simulation with EnergyPlus models and engine [41].
BEM result visualization and knowledge extraction [40].
Co-simulation BIM to BEM process automation [42].
Building operation and management Building regulation interpretation [43].
Building energy performance consulting and optimization [44].
Building system human-in-the-loop control [45].
specific areas. Whether it’s engineering design, product, or any
specialized field, these agents leverage their access to specialized
databases and documentation to provide accurate and detailed re-
sponses.
World knowledge agentThe world knowledge agent provides
context and general information that draws from a wide-ranging
knowledge base. This agent is particularly useful for answering
questions related to facts, historical data, and other information
that requires a broad understanding or contextual background.
2.2 Tools
The system is equipped with a variety of tools to enhance
the capabilities of the agents.
InterpreterCode interpreter allows coding agents to execute
and test code snippets in real-time and in the local environment.
In our developed system, the code interpreter is a crucial tool
to run code that retrieves data, interacts with various APIs, and
generates results and plots. It ensures calculations are performed
in a consistent and accurate way while allowing flexibility for
presenting the results depending on the needs.Retrieval-Augmented Generation (RAG)RAG modules en-
able agents to retrieve information from external databases or
documents to generate more accurate and contextually relevant
responses. In the developed system, agents prioritize the avail-
able databases to search for relevant data, documents, and codes
that can significantly improve answers to be more grounded and
accurate.
Language Model Models (LLMs)LLMs are utilized by the
agents for natural language processing tasks, such as understand-
ing queries, formulating responses in a human-like manner, etc.
User Interface (UI)UI enables a chat-based interface for the
users to put in questions and get system responses in the form of
text, images, downloadable files, and potentially more.
2.3 Databases
The agents access several key databases to fulfill their tasks:
Building DataThis includes data related to the buildings and
their surrounding environments. The data includes the numeri-
cal values related to building operations and equipment, building
occupancy, weather conditions, and more. Metadata information
4 Copyright © 2026 by ASME

FIGURE 1: System architecture design for the LLM-based multi-agent system that enables LLM to perform domain-specific tasks using
a wide range of data and human building interaction through natural language.
is stored to allow agents to understand the type of data, the rela-
tionships to each other, and other additional information.
CodebaseThese databases contain a wealth of programming
resources, such as libraries, APIs, and coding examples, which
coding agents use to develop and optimize solutions.
DocbaseSpecialist agents rely on domain documentation
databases that contain in-depth information, such as scientific pa-
pers, engineering manuals, and industry standards.
World KnowledgeThese databases include encyclopedic
knowledge and other information that support the world knowl-
edge agent in providing accurate and contextually appropriate
responses.
3 APPLICATIONS AND RESULTS
3.1 Setup and database
We tested our method given access to Nantum AI [10],
where building data, application documentations, and code
repositories are hosted. It is assumed that people ask questions
with the right level of access to confidential building informa-
tion. A chat-based interface developed as a web application us-ing Streamlit is used as the human building interface for testing.
The conversation management is done using Chainlit. Answers
to the questions are provided in the form of appropriate text, fig-
ures, and data files if necessary. For the experimental setup, the
LLM models used include GPT-4o, GPT-3.5 turbo, Claude 3.5
haiku, and Claude 3.5 sonnet. The models are chosen based on
the agents and their tasks.
3.2 Question design
A set of questions is designed for testing, ranging from
general knowledge about buildings, real-time conditions of the
buildings, to the specific impact of each Energy Conservation
Measure (ECM) running in the building. The full list of ques-
tions is listed in Table 2. We considered three types of occupant
roles, namely tenant, building engineer, and management. For
each of the roles, they would have different levels of knowledge
and varying interests in various aspects of the buildings. For ex-
ample, tenants might be more interested in the comfort, while the
management team might consider the overall energy consump-
tion and sustainability targets on carbon emissions, and so on.
5 Copyright © 2026 by ASME

Human Roles Question Types Questions
Tenant / Occupant Data Retrieval What is the current temperature in the building?
How many people are in the building now?
General What measures are in place to reduce energy consumption?
What is the significance of reducing carbon emissions from buildings?
Building Engineer Energy What was the total electric consumption for last month?
What was the steam total consumption for the last 7 days?
Peak Demand What was the peak demand for the last billing period?
What is today’s peak demand so far?
Occupancy What time do people start to arrive at the building?
What is the maximum occupancy on average for my building?
ECM What are the ECMs that can help me reduce energy consumption?
What is Nantum automated demand management?
Management Sustainability What’s the total GHG emission yesterday?
What is LL97?
How can I lower my GHG emissions?
Engineering What’s the compliance rate of Nantum lunch ramp-up commands last week?
How much energy did the ECMs save for my building?
How can you achieve zero carbon emission?
Property What’s the energy saving in the last billing period?
How does the current GHG compare to last year?
What’s the average daily water consumption in 2019?
How is building A compared to building B’s average daily electricity consumption?
General How can Nantum help reduce the GHG emissions?
What Nantum applications or ECMs help managing and reducing peak demand?
TABLE 2: Designed questions
3.3 Results
In order to evaluate the results, we first quantify the response
time, accuracy in numerical results, data retrieved, and documen-
tation retrieved based on the question shown in Table 3. Next,
human responses are gathered from 63.7% of researchers, 9.1%
of product managers, 9.1% of building operators/managers, and
18.2% of occupants. 36.4% of people have more than 5 years of
experience working with buildings. The full questions and an-swers generated, as well as the user evaluation results, can be
found in Appendix A.
As an example shown in Figure 2, when asked the question
”What time do people start to arrive at the building?”, the multi-
agent system went through the database to find the appropriate
occupancy data and invoked the interpreter for code, figure, and
CSV file generation. The resulting response contains the figure
and data file that are both downloadable. In addition, the text
6 Copyright © 2026 by ASME

answer is provided at the bottom. The answer shows that the
system clearly understood the question and accessed the right
database to provide meaningful and accurate answers with illus-
trative graphs.
FIGURE 2: Example question and answer provided by the de-
signed system.
4 DISCUSSION
The developed system bridges the gap between humans,
building, and computer/AI systems by enabling intuitive,
language-based communication through the chat-based human
building interface. Unlike traditional interaction mechanisms
that require specialized knowledge and training, the developed
system delivers personalized, actionable answers and insights
based on user queries, accessing concrete building information,
including data, documentation, and code. The architecture de-
signed is modular and scalable, with flexibility to incorporate
new data and switch to different LLM models with ease. At
the same time, the multi-agent approach proposed for facilitat-
ing human and buildings can also fit other applications. Such an
LLM-based system can be a great interpreter to translate human
needs to the right ways to retrieve data, write codes, and searchdocuments; on the other hand, it can put data, codes, and tech-
nical documents into texts, figures, and more that can be easily
understood by different types of users.
It is important to note that the choice of not performing fine-
tuning or pretraining comes down to the lack of high-quality la-
beled data. Instead, we treat the pretrained model as a general-
purpose reasoning engine, applying value shaping post hoc rather
than altering its internal knowledge. This approach allows effi-
cient use of data and guides the system to specific tasks and capa-
bilities of different agents. The approach also requires thorough
documentation for code use when accessing different application
APIs and understanding their purposes and limitations. Criti-
cally, it also enables the system to adapt to system changes, as
we do not need to train the solver each time we move to a new
building environment or add/remove new modules.
Furthermore, it is essential to distinguish this architecture
from simpler alternatives like a vanilla single LLM or a single-
agent ReAct loop. A vanilla LLM baseline would inherently fail
on the majority of our tested queries, as it lacks access to the pri-
vate building telemetry and metadata. Even a single-agent ReAct
approach, while capable of tool invocation, often suffers from
reasoning drift or alignment failures when navigating the diverse
data formats and high technical precision required for HBI. By
decoupling high-level task decomposition through the Doorman
agent from specialized execution, our framework ensures techni-
cal accuracy and prevents the hallucinations that typically plague
general-purpose models in specialized engineering domains.
Testing on a range of carefully designed questions, our ex-
periments demonstrate that the system can effectively interpret
diverse user queries, access relevant data from structured and un-
structured sources, and generate accurate, meaningful responses.
For example, queries about real-time building conditions or en-
ergy consumption were met with detailed outputs, including text
explanations, graphical representations, and downloadable data
files. These results validate the system’s potential to make build-
ing interactions more accessible, reduce operational complexity,
and support decision-making processes across different stake-
holder groups. Users generally appreciate the HBI AI chatbot
for its responsiveness, accuracy, and ability to provide relevant
information efficiently. Many users find it helpful for answer-
ing queries quickly and easily navigating complex topics. How-
ever, some users express concerns about occasional redundancies
or generic responses that lack depth. Others feel that the chat-
bot sometimes struggles with nuanced or context-dependent in-
quiries, leading to frustration when detailed or highly specific an-
swers are needed. Additionally, there may be concerns regarding
the chatbot’s ability to handle more complex multi-turn conver-
sations effectively. Overall, while the HBI AI chatbot is valued
for its responsiveness and accuracy, users suggest improvements
in contextual understanding, conciseness, and depth of responses
to enhance its utility.
Limitations for the current solution exist. First, multi-step
7 Copyright © 2026 by ASME

Question Response Time Numerical Accuracy Data Access Accuracy Doc Retrieval Accuracy
Q1 29s 100% 100% 100%
Q2 26s 100% 100% 100%
Q3 15s N/A 100% 100%
Q4 13s N/A 100% 100%
Q5 22s 100% 100% 100%
Q6 25s 100% 100% 100%
Q7 24s 100% 100% 100%
Q8 23s 100% 100% 100%
Q9 20s 100% 100% 100%
Q10 33s 100% 100% 100%
Q11 11s N/A 100% 100%
Q12 15s N/A 100% 100%
Q13 19s 100% 100% 100%
Q14 30s 100% 50% 100%
Q15 27s N/A 100% 100%
Q16 53s 100% 100% 100%
Q17 53s 100% 100% 100%
Q18 12s N/A 100% N/A
Q19 54s 100% 100% 100%
Q20 26s 100% 100% 100%
Q21 22s 100% 100% 100%
Q22 19s 100% 100% 100%
Q23 15s N/A 100% 100%
Q24 19s N/A 100% 100%
TABLE 3: Quantitative evaluation results for the designed questions
sequential and recursive tasks requiring sequential and/or recur-
sive use of different agents cannot be achieved. The doorman
agent is currently able to call the appropriate agent for the task,
but it cannot arrange orders or assign a specific agent to run
multiple times at different stages. Second, the system relies on
cleaned and structured datasets. Uncleaned or unstructured data
may be called and used, but problems with improper units, miss-
ing data, or other issues could cause data loading to fail. Third,
a silent error can still cause confident but wrong answers. One
missing data point could cause the calculation to potentially be
wrong or fail. Furthermore, we have not connected the systemto enable actual control of buildings through BMS/BAS. It is a
natural next step to enable such a system in combination with in-
teractive reinforcement learning and human-in-the-loop control
for building and its subsystems.
5 CONCLUSION
In this study, we developed a novel multi-agent system pow-
ered by LLMs to enhance HBI. The system developed enables
LLM to perform domain-specific tasks that require expert knowl-
edge, interacting with different user types, and utilizing tools.
8 Copyright © 2026 by ASME

The system enables a more intuitive form of human building in-
teraction and bridges the information gap between occupants, en-
gineers, managers, and the ever-smarter building automation and
control systems through direct interactions and an intuitive inter-
face. With a set of carefully designed questions covering differ-
ent stakeholders and topics, we show that the system is capable
of accurately understanding the questions, accessing the correct
information, and constructing meaningful and informative out-
puts. Human building interaction powered by LLM can have a
significant impact on the future of smart buildings to be more
sustainable, that is, efficient, healthy, and human-centric.
ACKNOWLEDGMENT
The authors would like to thank Gurpreet Singh, Amir Beh-
jat, Gerald Toge, Gene Boniberger, John Gilbert, John J. Gilbert
IV , Evan Torkos, Amit Paul, Lauren Long, Bryce Nielsen, and
Gary Chance from Nantum AI, along with Rudin Management
Company, for their support and contributions to this project.
REFERENCES
[1] LEECH, J. A., NELSON, W. C., BURNETT, R. T.,
AARON, S., and RAIZENNE, M. E., 2002. “It’s
about time: A comparison of canadian and american
time–activity patterns”.Journal of Exposure Science
&amp; Environmental Epidemiology,12(6), p. 427–432.
[2] Day, J. K., McIlvennie, C., Brackley, C., Tarantini, M.,
Piselli, C., Hahn, J., O’Brien, W., Rajus, V . S., De Simone,
M., Kjærgaard, M. B., et al., 2020. “A review of select
human-building interfaces and their relationship to human
behavior, energy use and occupant comfort”.Building and
environment,178, p. 106920.
[3] Alavi, H. S., Churchill, E. F., Wiberg, M., Lalanne, D.,
Dalsgaard, P., Fatah gen Schieck, A., and Rogers, Y ., 2019.
Introduction to human-building interaction (hbi) interfac-
ing hci with architecture and urban design.
[4] Becerik-Gerber, B., Lucas, G., Aryal, A., Awada, M.,
Berg ´es, M., Billington, S., Boric-Lubecke, O., Ghahra-
mani, A., Heydarian, A., H ¨oelscher, C., et al., 2022. “The
field of human building interaction for convergent research
and innovation for intelligent built environments”.Scien-
tific Reports,12(1), p. 22092.
[5] Messner, J. I., 2022. The lifecycle of a building project.
Accessed: 2024-09-02.
[6] Norouziasl, S., Jafari, A., and Zhu, Y ., 2021. “Modeling
and simulation of energy-related human-building interac-
tion: A systematic review”.Journal of Building Engineer-
ing,44, p. 102928.
[7] Kim, H., Kang, H., Choi, H., Jung, D., and Hong, T., 2023.
“Human-building interaction for indoor environmental con-trol: Evolution of technology and future prospects”.Au-
tomation in Construction,152, p. 104938.
[8] Becerik-Gerber, B., Lucas, G., Aryal, A., Awada, M.,
Berg ´es, M., Billington, S. L., Boric-Lubecke, O., Ghahra-
mani, A., Heydarian, A., Jazizadeh, F., et al., 2022. “Ten
questions concerning human-building interaction research
for improving the quality of life”.Building and Environ-
ment,226, p. 109681.
[9] Bosch, 2023. Bosch building solutions -
history of building automation.https:
//www.boschbuildingsolutions.
com/xc/en/news-and-stories/
history-of-building-automation/. Accessed:
2023-05-23.
[10] Nantum AI, 2024. Nantum ai. Accessed: 2024-06-22.
[11] Malkawi, A., Ervin, S., Han, X., Chen, E. X., Lim, S., Am-
panavos, S., and Howard, P., 2023. “Design and applica-
tions of an iot architecture for data-driven smart building
operations and experimentation”.Energy and Buildings,
295, p. 113291.
[12] Apple, 2024. The foundation for a smarter home. Accessed:
2024-09-02.
[13] Wang, Y ., Shen, G., and Mehmani, A., 2024. “Indoor envi-
ronmental wellness index (iew-index): Towards intelligent
building systems automation and optimization”.Building
and Environment,247, p. 111039.
[14] Church, K. W., 2017. “Word2Vec”.Natural Language En-
gineering,23(1), Jan., pp. 155–162.
[15] Pennington, J., Socher, R., and Manning, C., 2014. “Glove:
Global Vectors for Word Representation”. In Proceedings
of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), Association for Computa-
tional Linguistics, pp. 1532–1543.
[16] Devlin, J., Chang, M.-W., Lee, K., and Toutanova,
K., 2019. BERT: Pre-training of Deep Bidirec-
tional Transformers for Language Understanding, May.
arXiv:1810.04805.
[17] Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou,
Y ., Min, Y ., Zhang, B., Zhang, J., Dong, Z., et al., 2023.
“A survey of large language models”.arXiv preprint
arXiv:2303.18223.
[18] Gong, L., Wang, S., Elhoushi, M., and Cheung, A.,
2024. “Evaluation of LLMs on syntax-aware code fill-
in-the-middle tasks”. In Proceedings of the 41st Interna-
tional Conference on Machine Learning, R. Salakhutdinov,
Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and
F. Berkenkamp, eds., V ol. 235 ofProceedings of Machine
Learning Research, PMLR, pp. 15907–15928.
[19] Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E., 2023.
“Gorilla: Large language model connected with massive
apis”.arXiv preprint arXiv:2305.15334.
[20] Hu, Z., Iscen, A., Jain, A., Kipf, T., Yue, Y ., Ross, D. A.,
9 Copyright © 2026 by ASME

Schmid, C., and Fathi, A., 2024. “SceneCraft: An LLM
agent for synthesizing 3D scenes as blender code”. In Pro-
ceedings of the 41st International Conference on Machine
Learning, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller,
N. Oliver, J. Scarlett, and F. Berkenkamp, eds., V ol. 235
ofProceedings of Machine Learning Research, PMLR,
pp. 19252–19282.
[21] Zhang, W., Wan, C., Zhang, Y ., Cheung, Y .-M., Tian, X.,
Shen, X., and Ye, J., 2024. “Interpreting and improving
large language models in arithmetic calculation”. In Pro-
ceedings of the 41st International Conference on Machine
Learning, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller,
N. Oliver, J. Scarlett, and F. Berkenkamp, eds., V ol. 235
ofProceedings of Machine Learning Research, PMLR,
pp. 59932–59950.
[22] Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y .,
Wang, Z., Ebrahimi, S., and Wang, H., 2024. Continual
learning of large language models: A comprehensive sur-
vey.
[23] Gururangan, S., Marasovi ´c, A., Swayamdipta, S., Lo, K.,
Beltagy, I., Downey, D., and Smith, N. A., 2020. Don’t
stop pretraining: Adapt language models to domains and
tasks.
[24] Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Ben-
gio, S., and Farajtabar, M., 2024. Gsm-symbolic: Under-
standing the limitations of mathematical reasoning in large
language models.
[25] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al., 2020. “Retrieval-augmented generation for
knowledge-intensive nlp tasks”.Advances in Neural In-
formation Processing Systems,33, pp. 9459–9474.
[26] Li, X., Wang, S., Zeng, S., Wu, Y ., and Yang, Y ., 2024. “A
survey on llm-based multi-agent systems: workflow, infras-
tructure, and challenges”.Vicinagearth,1(1), p. 9.
[27] Zhao, B., Hajishirzi, H., and Cao, Q., 2024. “APT: Adap-
tive pruning and tuning pretrained language models for effi-
cient training and inference”. In Proceedings of the 41st In-
ternational Conference on Machine Learning, R. Salakhut-
dinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, eds., V ol. 235 ofProceedings of Ma-
chine Learning Research, PMLR, pp. 60812–60831.
[28] Li, C., Liang, J., Zeng, A., Chen, X., Hausman, K.,
Sadigh, D., Levine, S., Fei-Fei, L., Xia, F., and Ichter, B.,
2024. “Chain of code: Reasoning with a language model-
augmented code emulator”. In Proceedings of the 41st In-
ternational Conference on Machine Learning, R. Salakhut-
dinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, eds., V ol. 235 ofProceedings of Ma-
chine Learning Research, PMLR, pp. 28259–28277.
[29] Du, Y ., Li, S., Torralba, A., Tenenbaum, J. B., and Mor-
datch, I., 2024. “Improving factuality and reasoning in lan-guage models through multiagent debate”. In Proceedings
of the 41st International Conference on Machine Learning,
ICML’24, JMLR.org.
[30] Crispino, N., Montgomery, K., Zeng, F., Song, D., and
Wang, C., 2024. “Agent instructs large language models to
be general zero-shot reasoners”. In Proceedings of the 41st
International Conference on Machine Learning, ICML’24,
JMLR.org.
[31] Amer, F., Jung, Y ., and Golparvar-Fard, M., 2021.
“Transformer machine learning language model for auto-
alignment of long-term and short-term plans in construc-
tion”.Automation in Construction,132, p. 103929.
[32] Prieto, S. A., Mengiste, E. T., and Garc ´ıa de Soto, B., 2023.
“Investigating the use of chatgpt for the scheduling of con-
struction projects”.Buildings,13(4), p. 857.
[33] Saka, A., Taiwo, R., Saka, N., Salami, B. A., Ajayi, S.,
Akande, K., and Kazemi, H., 2023. “Gpt models in con-
struction industry: Opportunities, limitations, and a use
case validation”.Developments in the Built Environment,
p. 100300.
[34] Uddin, S. J., Albert, A., Ovid, A., and Alsharef, A., 2023.
“Leveraging chatgpt to aid construction hazard recognition
and support safety education and training”.Sustainability,
15(9), p. 7121.
[35] Du, C., Esser, S., Nousias, S., and Borrmann, A., 2024.
“Text2bim: Generating building models using a large
language model-based multi-agent framework”.arXiv
preprint arXiv:2408.08054.
[36] Zheng, Z., Chen, K.-Y ., Cao, X.-Y ., Lu, X.-Z., and Lin, J.-
R., 2023. “Llm-funcmapper: Function identification for
interpreting complex clauses in building codes via llm”.
arXiv preprint arXiv:2308.08728.
[37] Chen, N., Lin, X., Jiang, H., and An, Y ., 2024. “Automated
building information modeling compliance check through
a large language model combined with deep learning and
ontology”.Buildings,14(7), p. 1983.
[38] Zheng, J., and Fischer, M., 2023. “Bim-gpt: a prompt-
based virtual assistant framework for bim information re-
trieval”.arXiv preprint arXiv:2304.09333.
[39] Su, Y ., Wan, C., Sethi, U., Lu, S., Musuvathi, M., and Nath,
S., 2023. “Hotgpt: How to make software documentation
more useful with a large language model?”. In Proceedings
of the 19th Workshop on Hot Topics in Operating Systems,
pp. 87–93.
[40] Zhang, L., Chen, Z., and Ford, V ., 2024. “Advancing build-
ing energy modeling with large language models: Explo-
ration and case studies”.arXiv preprint arXiv:2402.09579.
[41] Jiang, G., Ma, Z., Zhang, L., and Chen, J., 2024. “Eplus-
llm: A large language model-based computing platform
for automated building energy modeling”.Applied Energy,
367, p. 123431.
[42] Forth, K., and Borrmann, A., 2024. “Semantic enrichment
10 Copyright © 2026 by ASME

for bim-based building energy performance simulations us-
ing semantic textual similarity and fine-tuning multilingual
llm”.Journal of Building Engineering,95, p. 110312.
[43] Fuchs, S., Witbrock, M., Dimyadi, J., and Amor, R., 2024.
“Using large language models for the interpretation of
building regulations”.arXiv preprint arXiv:2407.21060.
[44] Xiao, T., and Xu, P., 2024. “Exploring automated en-
ergy optimization with unstructured building data: A multi-
agent based framework leveraging large language models”.
Energy and Buildings, p. 114691.
[45] Yang, H., Siew, M., and Joe-Wong, C., 2024. “An llm-
based digital twin for optimizing human-in-the loop sys-
tems”.arXiv preprint arXiv:2403.16809.
A Survey Instrument and Response Data
The complete survey instrument (including all questions and
answer options) and the anonymized user response dataset are
available at the following links:
Survey instrument: Google Form
Response dataset: Google Sheet
11 Copyright © 2026 by ASME