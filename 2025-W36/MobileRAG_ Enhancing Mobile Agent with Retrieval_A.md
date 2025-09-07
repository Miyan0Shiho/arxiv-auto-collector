# MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation

**Authors**: Gowen Loo, Chang Liu, Qinghong Yin, Xiang Chen, Jiawei Chen, Jingyuan Zhang, Yu Tian

**Published**: 2025-09-04 05:22:42

**PDF URL**: [http://arxiv.org/pdf/2509.03891v1](http://arxiv.org/pdf/2509.03891v1)

## Abstract
Smartphones have become indispensable in people's daily lives, permeating
nearly every aspect of modern society. With the continuous advancement of large
language models (LLMs), numerous LLM-based mobile agents have emerged. These
agents are capable of accurately parsing diverse user queries and automatically
assisting users in completing complex or repetitive operations. However,
current agents 1) heavily rely on the comprehension ability of LLMs, which can
lead to errors caused by misoperations or omitted steps during tasks, 2) lack
interaction with the external environment, often terminating tasks when an app
cannot fulfill user queries, and 3) lack memory capabilities, requiring each
instruction to reconstruct the interface and being unable to learn from and
correct previous mistakes. To alleviate the above issues, we propose MobileRAG,
a mobile agents framework enhanced by Retrieval-Augmented Generation (RAG),
which includes InterRAG, LocalRAG, and MemRAG. It leverages RAG to more quickly
and accurately identify user queries and accomplish complex and long-sequence
mobile tasks. Additionally, to more comprehensively assess the performance of
MobileRAG, we introduce MobileRAG-Eval, a more challenging benchmark
characterized by numerous complex, real-world mobile tasks that require
external knowledge assistance. Extensive experimental results on MobileRAG-Eval
demonstrate that MobileRAG can easily handle real-world mobile tasks, achieving
10.3\% improvement over state-of-the-art methods with fewer operational steps.
Our code is publicly available at:
https://github.com/liuxiaojieOutOfWorld/MobileRAG_arxiv

## Full Text


<!-- PDF content starts -->

MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation
Gowen Loo1*, Chang Liu2*, Qinghong Yin6, Xiang Chen4, Jiawei Chen5, Jingyuan Zhang6, Yu
Tian7†
1Network and Data Security Key Laboratory of Sichuan Province, University of Electronic Science and Technology of China
2DEPARTMENT OF COMPUTING, Hong Kong Polytechnic University
4College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics
5East China Normal University6Kuaishou-Inc
7Dept. of Comp. Sci. and Tech., Institute for AI, Tsinghua University, Tsinghua University
202314090108@std.uestc.edu.cn, chang181.liu@connect.polyu.hk, tianyu181@mails.ucas.ac.cn.
Abstract
Smartphones have become indispensable in people’s daily
lives, permeating nearly every aspect of modern society.
With the continuous advancement of large language models
(LLMs), numerous LLM-based mobile agents have emerged.
These agents are capable of accurately parsing diverse user
queries and automatically assisting users in completing com-
plex or repetitive operations. However, current agents 1)
heavily rely on the comprehension ability of LLMs, which
can lead to errors caused by misoperations or omitted steps
during tasks, 2) lack interaction with the external environ-
ment, often terminating tasks when an app cannot fulfill
user queries, and 3) lack memory capabilities, requiring
each instruction to reconstruct the interface and being un-
able to learn from and correct previous mistakes. To alle-
viate the above issues, we propose MobileRAG, a mobile
agents framework enhanced by Retrieval-Augmented Gen-
eration (RAG), which includes InterRAG, LocalRAG, and
MemRAG. It leverages RAG to more quickly and accu-
rately identify user queries and accomplish complex and
long-sequence mobile tasks. Additionally, to more compre-
hensively assess the performance of MobileRAG, we intro-
duce MobileRAG-Eval, a more challenging benchmark char-
acterized by numerous complex, real-world mobile tasks that
require external knowledge assistance. Extensive experimen-
tal results on MobileRAG-Eval demonstrate that MobileRAG
can easily handle real-world mobile tasks, achieving 10.3%
improvement over state-of-the-art methods with fewer opera-
tional steps. Our code is publicly available at: https://github.
com/liuxiaojieOutOfWorld/MobileRAG arxiv
Introduction
Today, approximately 4.69 billion individuals own smart-
phones, with average daily online activity on smartphone
exceeds 3 hours and 30 minutes1. Users increasingly rely
on smartphones to handle various tedious and complex
tasks. With the rapid development of large language mod-
els (LLMs) (Hurst et al. 2024; Gemini et al. 2024; Guo et al.
2025) , numerous LLM-based mobile agents (Wang et al.
2024b,a, 2025; Wu et al. 2024) have emerged. These agents
can accurately parse users’ diverse instructions and assist
*These authors contributed equally.
†Corresponding author
1https://backlinko.com/smartphone-usage-statistics
Figure 1: Three key challenges for mobile agents in practical
applications.
users automatically complete complex or repetitive opera-
tions. As shown in Figure 1, despite these advancements,
three key challenges remain in practical applications:
1) Previous works commonly rely excessively on LLMs’
comprehension capabilities (Wen et al. 2023; Wang et al.
2024b; Wen et al. 2024), necessitating multiple sequential
steps for even basic mobile device operations. The absence
of robust reasoning mechanisms leads to significant perfor-
mance deficiencies, including operational errors and step
omissions when handling complex tasks. Even seemingly
simple application interactions require multiple cognitive
processes (such as interface comprehension, apps localiza-
tion, and execution initiation). This fundamental limitation
undermines the agent’s autonomous task completion abili-
ties and substantially constrains its practical utility in real-
world deployment scenarios.
2) Current agents (Lee et al. 2024; Zhang et al. 2025;arXiv:2509.03891v1  [cs.CL]  4 Sep 2025

Wang et al. 2025) generally exhibit limited capability to
interact effectively with external environments. confronted
with application functionality constraints or unforeseen cir-
cumstances during task execution, these agents frequently
opt to terminate the task rather than explore alternative so-
lution pathways. For instance, when a user inputs ”I want to
watch Squid Game,” existing mobile agents often abandon
the task due to insufficient knowledge of available stream-
ing platforms. This inflexibility hampers the agents’ adapt-
ability and makes them poorly equipped to handle complex,
dynamic real-world environments.
3) Mobile agents demonstrate pronounced deficiencies in
memory and learning capabilities. Most existing works are
incapable of accumulating experience from prior tasks, ne-
cessitating the reconstruction of operational interfaces with
each new instruction. These approaches not only result in in-
efficient use of computational resources, but also prevent the
system from learning from historical successful cases and
engaging in self-correction. Consequently, these limitations
undermine both the operational efficiency of agents and their
capacity for autonomous, long-term evolution.
Retrieval-Augmented Generation (RAG) introduces a
novel research direction for enhancing LLMs’ comprehen-
sion (Lewis et al. 2020; Gao et al. 2023; Fan et al. 2024), ex-
ternal interaction, and memory learning capabilities. How-
ever, the direct application of RAG to mobile agents presents
several challenges: 1) agents may struggle to discern when
to invoke internal versus external applications; 2) limitations
imposed by the need for model lightweighting; and 3) inher-
ent trade-offs in memory retention mechanisms. To address
these issues, we propose MobileRAG, a RAG-based frame-
work enhanced for mobile agents. This framework com-
prises three core components—InterRAG, LocalRAG, and
MemRAG—that collectively aim to leverage RAG technolo-
gies for more rapid and accurate identification of user intents
and for efficient execution of complex, long-sequence de-
vice operations. Specifically, InterRAG facilitates the invo-
cation of external applications and supports robust informa-
tion retrieval, thereby enhancing the agent’s capacity for ex-
ternal interaction; LocalRAG streamlines on-device deploy-
ment and inference, constructing a lightweight RAG model
optimized for mobile environments; and MemRAG tackles
the challenge of memory retention by preserving representa-
tive and successful user actions, thus improving the real-time
responsiveness and overall efficiency of MobileRAG.
To verify the effectiveness of MobileRAG and its ca-
pability to interact with external knowledge, we construct
MobileRAG-Eval, a more challenging benchmark designed
to require deeper semantic understanding and integration
of external knowledge. Extensive experimental results on
MobileRAG-Eval demonstrate that MobileRAG achieves
superior performance in leveraging external knowledge and
managing complex, long-sequence tasks, with a 10.3% im-
provement over state-of-the-art (SOTA) (Wang et al. 2025)
methods. Notably, our framework significantly streamlines
the process and reduces the likelihood of erroneous opera-
tions by directly retrieving apps via RAG. In summary, our
main contributions are as follows:
• We design a RAG-based framework enhanced for mobileagents, comprising InterRAG, LocalRAG, and Mem-
RAG. It leverages RAG technology to more quickly and
accurately identify customer needs and accomplish more
complex and long-sequence device operation tasks.
• Through the effective interaction of InterRAG and Lo-
calRAG, MobileRAG enables agents to incorporate ex-
ternal knowledge, facilitating a more comprehensive un-
derstanding of user queries and supporting accurate, effi-
cient operation of mobile devices.
• With the memory capabilities of MobileRAG, successful
cases from previous operations can be retained and user
instructions can be matched to historical instances using
MemRAG. It further reduces unnecessary analysis and
operational steps, improving both speed and accuracy.
Related Work
Mobile Agents
In recent years, the rapid development of LLMs has driven
significant advances in mobile agents. Broadly, existing mo-
bile agents can be divided into two main categories. The
first category consists of XML-based mobile agents, which
utilize XML (Cabri, Leonardi, and Zambonelli 2000; Cian-
carini, Tolksdorf, and Zambonelli 2002; Tamayo, Granell,
and Huerta 2011) as a medium for structured data exchange.
Owing to the universality and high readability of XML,
these agents (Chen, Linz, and Cheng 2008; Chou, Ko, and
Cheng 2010) facilitate information integration and collab-
oration across heterogeneous systems. However, limited by
the data representation capabilities and processing mecha-
nisms of XML, such agents often lack the flexibility needed
to adapt to complex and dynamic task requirements. The
second category comprises GUI-based agents (Sun et al.
2022; Li and Li 2022; Gou et al. 2024), which emphasize
interaction through visual interfaces (Hong et al. 2024) to
enable real-time decision-making and task execution. Nev-
ertheless, current GUI-based agents are entirely reliant on
LLMs for interface understanding (Wen et al. 2023; Rawles
et al. 2023; Gur et al. 2023). This dependency can lead to
issues with model recognition and decision accuracy in real-
world applications, potentially impacting the reliable com-
pletion of tasks. To address these issues, we propose Mobil-
eRAG, a framework designed to accurately understand user
queries, effectively perform information retrieval, and en-
hance memory and learning capabilities.
Retrieval-Augmented Generation
RAG is an emerging hybrid architecture that addresses the
limitations of purely generative models (Zhao et al. 2024;
Chen et al. 2024; Es et al. 2024). It improves answer ac-
curacy by retrieving relevant information and generating re-
sponses based on that data. While RAG has been widely ap-
plied in fields such as LLMs (Gao et al. 2023; Fan et al.
2024), LVLMs (Yu et al. 2024), and agent-based systems
(L´ala et al. 2023), its adopt in mobile agents remains un-
derexplored (Gupta, Ranjan, and Singh 2024). The key
challenges here include difficulties in accurately determin-
ing when to invoke internal or external applications, con-
straints imposed by the limited computational power and

Figure 2: Overall framework of MobileRAG. It comprises three key RAG modules: 1) LocalRAG, responsible for managing
and retrieving local applications; 2) InterRAG, dedicated to retrieving external knowledge; and 3) MemRAG, which manages
and retrieves similar successful historical steps. The red paths are exclusively available in the retrieved historical steps.
storage capacity of mobile devices, and trade-offs inherent in
memory retention mechanisms between maintaining infor-
mation integrity and ensuring processing efficiency. Collec-
tively, these issues hinder task execution efficiency and ulti-
mately degrade the overall performance of mobile agents. To
address these challenges, we propose MobileRAG, a RAG-
based framework designed for mobile agents. It includes
three key components: InterRAG, which enables interac-
tion with external applications for up-to-date information;
LocalRAG, which handles on-device knowledge retrieval
for faster access; and MemRAG, which uses memory-based
learning to enhance accuracy and task completion by re-
membering successful operations. These components col-
lectively address existing limitations, improve execution ef-
ficiency, and boost mobile agent performance.
Methods
Overview of MobileRAG
To address inadequate reasoning, limited adaptability to ex-
ternal interactions, and poor memory capabilities of exist-
ing mobile agents, we propose MobileRAG, a RAG-based
framework specifically designed for mobile environments.
While RAG has shown promise in various domains, di-
rectly applying it to mobile scenarios introduces several
challenges, including distinguishing between internal and
external resources, managing lightweight model constraints,
and balancing memory retention trade-offs. MobileRAG re-
solves these issues by integrating tailored components to en-
hance mobile agent performance and efficiency.
MobileRAG consists of three components: MemRAG,
LocalRAG, InterRAG. As shown in Figure 2, upon receiv-
ing a user query, it first utilizes MemRAG to determine
whether identical or similar tasks have previously been ex-
ecuted successfully. For exact matches, the system directly
reuses the recorded operation steps; for similar queries, these
prior cases are provided to the agent as guidance. During
task execution, when the agent encounters unfamiliar knowl-edge or requires additional context, it dynamically retrieves
external information via InterRAG. Subsequently, Local-
RAG is responsible for verifying whether the necessary apps
are already installed on the device, thereby supporting effi-
cient and accurate app management. If an app is not avail-
able locally, LocalRAG initiates its download from Play
Store and updates the local application database accordingly.
After confirming the availability of all required resources
and apps, the agent integrates these components, execut-
ing precise steps including app launching, navigation, con-
tent search, and task completion. Through the coordinated
collaboration of MemRAG, InterRAG, LocalRAG, Mobil-
eRAG seamlessly transforms complex user queries into ac-
tionable and automated workflows on mobile devices.
LocalRAG
To mitigate operational errors or omitted steps caused by
mobile agents’ complete reliance on agent’s comprehension
capabilities, we propose LocalRAG, specifically designed
to directly perceive and manage locally installed app. We
select the BGE-small series as our foundational retrieval
model to address the limitations associated with lightweight
retrieval architectures. To achieve robust and efficient se-
mantic matching, We design an optimized retrieval process
tailored to local apps, enabling effective matching between
user prompts and apps while promptly identifying cases
where no suitable match exists. Specifically, we employ a
LLM to generate user query, for interacting with the local
app based on its description2. To enable LocalRAG to iden-
tify cases where no suitable app exists, we generate a set of
user query and the “None” indicator bit that are not sup-
ported by any local apps. By performing supervised con-
trastive learning on the retriever between the user query and
the app descriptions, we ensure accurate alignment between
user task expressions and app functionalities in mobile sce-
2we collected the “About this app” section from Google Play as
the app description.

narios, while also equipping the model with the capability to
effectively reject unsupported commands.
Upon receiving a user query, LocalRAG calculates the co-
sine similarity between the query and the app descriptions
within the local app library. It subsequently returns the top-3
relevant apps informations to the agent, providing compre-
hensive metadata including app names, package identifiers,
descriptions, and similarity scores. The agent determines the
appropriate app by analyzing the user’s query, and then trig-
gers the selected app through its package identifier. When
LocalRAG yields no matches or the highest similarity score
falls below the established threshold, it notifies the agent that
no appropriate local app is available, prompting LocalRAG
to initiate the downloading process from Play Store. Subse-
quently updates its local app database accordingly.
InterRAG
Previous work lacks adaptive mechanisms to confront with
unknown contexts or app constraints, thus tend to terminate
tasks rather than explore possible alternatives. This restricts
their operation to the boundaries of predefined knowledge
and capabilities. To address these limitations and empower
mobile agent to dynamically extend its knowledge bound-
aries, we introduce InterRAG, a module designed to enhance
agent capabilities through real-time access to external infor-
mation sources. It systematically augments agent in compre-
hending and processing user queries that involve previously
unknown contexts, thereby facilitating continuous knowl-
edge acquisition and adaptive response generation.
As a component for interacting with external knowl-
edge, InterRAG significantly enhances the agent’s adapt-
ability to open-domain tasks. When the agent encounters
entities, terms, or concepts beyond its training corpus or
local app base during user query processing, it delegates
dynamic retrieval to InterRAG. Specifically, InterRAG for-
mulates queries for unfamiliar keywords or entities and ac-
quires real-time information from the internet via the Google
Search API. It parses and filters the search results, extracting
the top-10 relevant results and returning them to the LLM
in a structured format that includes titles and summaries.
Leveraging this up-to-date external information, the agent
can more accurately interpret the semantic intent of user
queries and recommend suitable apps or actions accordingly.
Notably, when apps within the local app base prove insuf-
ficient for user requirements, InterRAG enables the agent
to download relevant apps from Google Play and incorpo-
rate their information into the local app base, thus optimiz-
ing subsequent retrieval efficiency. This workflow allows the
mobile agent to flexibly respond to queries involving newly
emerging or rapidly evolving entities, substantially broad-
ening its operational scope and adaptability in open-domain
tasks.
MemRAG
The existing mobile agent frameworks lack the ability
to leverage successful experiences from previous tasks to
enhance its operational capabilities. Upon receiving new
queries, the agent must reconstruct the operational interfaceBenchmark TasksMulti-App
TasksNo-App
TasksAppsAvg
OpsTotal
Ops
Mobile-Eval 33 3 9 10 5.55 183
Mobile-Eval-v2 44 4 8 10 5.57 245
AppAgent 45 0 0 9 6.31 284
Our Dataset 80 20 30 20 8.01 641
Table 1: Comparison with existing dynamic evaluation
benchmarks on emulator devices.
from scratch, leading to inefficient utilization of computa-
tional resources and an inability to self-correct based on
prior successes. To address these limitations and strengthen
the agent’s memory and learning capabilities, we propose
MemRAG, a module designed to efficiently store and man-
age records of successful operations. During the processing
of new tasks, MemRAG enables the agent to automatically
retrieve and reference relevant historical information, facil-
itating operation reusability and promoting continuous self-
improvement of system capabilities.
Upon successful execution of a user’s query, MobileRAG
stores the user’s query alongside the corresponding opera-
tion steps in the memory database. When a new query is re-
ceived, It employs MemRAG to retrieve relevant historical
queries. If the similarity between the most relevant histori-
cal query and the current query exceeds a predefined thresh-
old, the corresponding historical operation step is provided
to the agent alongside the new query to inform the genera-
tion of an updated execution path. In cases where the queries
are determined to be identical, the system bypasses step re-
generation and executes the task directly using the estab-
lished operational sequence. MemRAG fundamentally en-
hances the agent with sophisticated memory and learning
capabilities. As user queries with the device through the
agent accumulate, the mobile agent undergoes continuous
evolution, progressively strengthening its autonomous capa-
bilities. Additionally, MemRAG improves operational effi-
ciency through two key advantages: 1) fewer operation steps
accelerate the agent’s response speed, 2) a reduced set of
steps minimizes the influence of erroneous cases, thereby
improving the overall accuracy of the agent’s outputs.
Experiments
MobileRAG-Eval
As shown in Table 1, existing benchmarks predominantly
focus straightforward tasks with predefined app specifi-
cations, where performance has largely plateaued. How-
ever, they remain constrained by static local apps, fail-
ing to accommodate dynamic app updates, real-time ex-
ternal information integration, and capabilities for longitu-
dinal task reasoning and operational learning. To address
these limitations, we introduce MobileRAG-Eval, a chal-
lenging benchmark designed for dynamically updating on-
device apps that mandates external knowledge augmentation
in complex real-world scenarios, emphasizing real-time con-
straints, reasoning-intensive planning, and cross-app coordi-

Figure 3: Leveraging LocalRAG and InterRAG for complex user query workflows. InterRAG effectively utilizes external knowl-
edge to interpret user queries, while LocalRAG reduces operational errors or omitted steps that may arise from the LLMs’
insufficient understanding of interfaces, thereby enabling the modeling of long sequential tasks.
nation. It evaluates agents to autonomously select optimal
tools, intelligently interface with external services to over-
come task execution barriers, and self-correct based on his-
torical demand patterns and operational feedback. We detail
the benchmark specifics in Appendix A.
MobileRAG-Eval integrates the MobileAgent-v2 bench-
mark (Wang et al. 2024a) and incorporates 36 novel ad-
vanced instructions, significantly extending both app cov-
erage and task complexity. Its core innovation lies in the es-
tablishment of a progressive three-tier evaluation framework
that spans atomic operations, composite tasks, and open-
environment instructions. Specifically, the framework in-
cludes: (1) a basic atomic operation library (core interaction
primitives within standalone apps); (2) multi-app coordina-
tion directives (chained cross-app procedures); and (3) open-
scenario instructions (application-agnostic tasks requiring
external knowledge retrieval and autonomous tool selec-
tion). The expanded dataset encompasses a dynamically
evolving app ecosystem, covering three critical domains:
core device functionalities (system configuration, app life-
cycle management), high-frequency lifestyle services (mul-
timedia resource access, geolocation navigation, audio con-
trol), and cross-app workflows. This comprehensive struc-
ture enables a systematic assessment of agents’ capabilities
in adaptive tool orchestration, real-time decision-making ro-
bustness, and disturbance resistance within dynamic, uncer-
tain environments.
Experimental Setting
Metrics . We propose a comprehensive five-dimensional
evaluation suite to quantify agent capabilities in dynamic
mobile environments. The metrics are detailed as follow:
•App Selection (AS) is defined as the percentage of tasksin which the optimal app is selected, calculated as: AS =
Correct app selections / Total app selections.
•Action Fidelity (AF) measures the proportion of correctly
executed atomic actions (Tap, Type, Swipe, Back, Stop)
relative to ground truth sequences, calculated as AF = Cor-
rect actions / Total actions.
•Reflection Precision (RP) is the correct diagnosis rate dur-
ing error recovery, calculated as AF = Correct reflections /
Total reflections.
•Task Completion Ratio (TCR) measures the proportion
of achieved sub-goals with atomic verification, calculated
as TCR = Completed sub-goals / Total sub-goals.
•Task Success Rate (TSR) is defined as Binary success
measurement where success requires all user query re-
quirements fulfilled. calculated as TSR = Successful tasks
/ Total tasks.
Baseline . We comprehensively evaluate our method against
three state-of-the-art (SOTA) open-source mobile agent
frameworks, including Mobile-Agent (Wang et al. 2024b),
Mobile-Agent-v2 (Wang et al. 2024a) , and Mobile-Agent-
E (Wang et al. 2025). For fair comparisons, We compared
MobileRAG with previous work on Pixel 9 in Android Stu-
dio, all baselines are implemented under identical experi-
mental conditions, sharing the same atomic action space, vi-
sual perception models, and initial prompting strategies as
MobileRAG. To further validate the framework’s effective-
ness in real-world scenarios, we performed extensive exper-
iments on the Xiaomi 15 Pro.
Implementation Details . Our agent consists of two pri-
mary stages: the App Selection stage and the Action
stage, each leveraging advanced LLMs as backbone, in-

Model TypeApp Selection
(%)↑Action Fidelity
(%)↑Reflection Precision
(%)↑Task Completion Ratio
(%)↑Task Success Rate
(%)↑
Mobile-Agent-v1 Single-Agent 76.3 65.8 - 63.1 47.5
Mobile-Agent-v2 Multi-Agent 77.5 71.2 95.8 66.3 52.5
Mobile-Agent-E Multi-Agent 92.9 87.6 98.8 88.6 72.5
MobileRAG Multi-Agent 100.0 86.4 98.5 91.2 80.0
MobileRAG∗Multi-Agent 100.0 87.8 98.8 92.9 82.5
Table 2: Comparison with SOTA on the MobileRAG-Eval.∗indicates results obtained on the Xiaomi 15 Pro, all other results
are collected on the Pixel 9 within the Android Studio environment.
MobileRAG AS AF RP TCR TSR
Gemini-1.5-pro 96.3 74.3 97.2 90.2 71.3
Claude-3.5-Sonnet 100.0 87.1 98.7 92.7 83.8
GPT-4o 100.0 86.4 98.5 91.2 80.0
Table 3: Results on various LLMs backbones
cluding Claude-3.5-Sonnet (Anthropic 2024)3, Gemini-1.5-
pro (Gemini et al. 2024)4, and GPT-4o (Hurst et al. 2024)5.
Unless specified otherwise, GPT-4o serves as the default
backbone for all models. In the App Selection stage, we
adopt BGE-small as retrieval model to enhanced context re-
trieval and accurate app selection. In the Action stage, the
agent integrates a comprehensive Visual Perception Module
composed of the ConvNextViT-document OCR model (Liao
et al. 2020) from ModelScope for text recognition, Ground-
ingDINO (Liu et al. 2024) for natural language prompt-
based object detection, and Qwen-VL-Int4 (Bai et al. 2023)
for generating detailed icon descriptions, collectively en-
abling precise visual interaction within mobile environ-
ments. A similarity threshold of 0.8 is adopted in MemRAG,
ensuring that only closely related historical queries inform
the response to new inputs.
Experiment Results
Comparison with SOTA frameworks. Table 2 presents the
result of MobileRAG and all baselines on MobileRAG-Eval.
The results demonstrate that our method achieves superior
perfor mance on various metrics, especially on AS, Mo-
bileRAG can achieve a 100% success rate, confirming the
effectiveness of LocalRAG. By constructing app informa-
tion into a local knowledge base, it can effectively miti-
gate operational errors or omitted steps caused by mobile
agents’ complete reliance on LLM’s comprehension capa-
bilities. Meanwhile, we achieve SOTA performance on TSR,
achieveing 10.3% improvement over other baseline. This re-
sult demonstrates that MobileRAG can effectively interpret
users’ ambiguous intentions via InterRAG, making it partic-
ularly well-suited for real-world application scenarios.
3Claude-3.5 version: Claude-3.5-Sonnet-2024-10-22
4Gemini-1.5 version: Gemini-1.5-Pro-2024-04-09
5GPT-4o version: GPT-4o-2024-11-20ModelLLM Action
Single Dual Triple Single Dual Triple
Mobile-Agent-v2 3.0 7.0 11.3 3.0 7.0 11.3
Mobile-Agent-E 1.9 4.8 8.0 1.9 4.8 8.0
MobileRAG 1.9 3.7 4.3 1.0 1.0 1.0
Table 4: Efficiency of MobileRAG on Single-, Dual-, and
Triple-App Tasks. “LLM” indicates the number of LLM
calls, and “Action” indicates the number of mobile steps.
We further validated the effectiveness of MobileAgent in
real-world environments (Xiaomi 15 Pro). As shown in Ta-
ble 2, MobileAgent performs exceptionally well across sev-
eral key indicators and can reliably handle complex and dy-
namic practical app scenarios. Notably, compared to XML-
based and GUI-based mobile agents, MobileRAG can di-
rectly retrieve relevant context via LocalRAG, efficiently ac-
cess and launch target apps, and flexibly leverage system-
level functionalities. Owing to its lightweight architecture
and high compatibility, it is well-suited for widespread de-
ployment on real-world consumer mobile devices.
Table 3 presents a comparison of MobileRAG across dif-
ferent backbone LLMs. MobileRAG consistently outper-
forms across all recent LLMs, owing to its superior frame-
work design. It effectively leverages retrieval of historical in-
formation, local apps, and web data, which substantially mit-
igates comprehension or operational errors caused by large
model hallucinations, and greatly enhances the reliability
and practical value of the system.
Efficiency Analysis. Table 4 reports the efficiency of Mo-
bileRAG by quantifying the number of steps required for
LLM-based agents to invoke and open apps, an essential fac-
tor in evaluating the model’s practical applicability. Com-
pared to other methods, MobileRAG demonstrates a sub-
stantial efficiency gain. It can be attributed to the efficient
retrieval architecture in MobileRAG, where LocalRAG re-
quires only 0.005 seconds per retrieval and interRAG only
0.4seconds, thereby significantly decreasing the number of
LLM-based agent calls and accelerating the overall runtime.
With the support of LocalRAG, MobileRAG significantly
reduces the number of steps to open apps, especially for
multi-app tasks. By enabling direct retrieval and execution
of local apps, LocalRAG greatly enhances the efficiency of
MobileRAG. Additionally, fewer operational steps reduce

Figure 4: Leveraging MemRAG to retrieve previously successful steps enables the mobile agent to perform rapid operations.
With its advanced memory and learning mechanisms, MemRAG empowers the agent to adapt based on past successes. As users
interact with the agent and more queries accumulate, the mobile agent evolves, gradually enhancing its autonomous capabilities.
the likelihood of operational errors or omissions, thereby en-
hancing the overall architectural performance.
Model AS AF RP TCR TSR avg open app
w/o Local 93.8 75.4 96.8 82.1 67.5 11.42 3.12
w/o Inter 88.1 85.3 97.1 79.7 65.0 9.94 1.81
MobileRAG 100 86.4 98.5 91.2 80.0 9.15 1.86
Table 5: Ablation studies of LocalRAG and InterRAG. “w/o
Local” and “w/o Inter” indicate the removal of LocalRAG
and InterRAG, respectively.
Ablation Studies. Table 5 shows the results of Ablation
studies, we find each component of MobileRAG plays a key
role in improving performance. First, we use LLM-based
visual perception instead of LocalRAG. The results show
varying degrees of decline across different metrics, demon-
strating that LocalRAG achieves significant performance
improvements compared to GUI-based mobile agent, partic-
ularly in terms of interaction steps. LocalRAG can directly
validate and operate locally installed apps, thereby eliminat-
ing unnecessary steps such as manual searching and inter-
face understanding. Second, we remove InterRAG in our
framework, we observed a decrease of 11.9% and 18.75%
in AS and TSR, respectively. This finding confirms that In-
terRAG is capable of accessing the updated and comprehen-
sive app information from external sources, thereby enabling
more informed and accurate app selection.
We further analyzed the effectiveness of MemRAG by
testing 15 groups of data, encompassing cases with com-
plete consistency, similarity above 80%, and similarity be-
low 80% (More detail are provided in Appendix B.). As
shown in Table 6, MemRAG effectively enhances model
performance, particularly by reducing the number of oper-
ational steps, with an average reduction of 2.4 steps. These
results clearly demonstrate that MobileRAG is capable oflearning relevant operational procedures from historical suc-
cessful steps via MemRAG, thereby improving both effi-
ciency and overall task accuracy.
Model AS AF RP TCR TSR Step
w/o Mem 100 88.7 98.4 93.8 86.7 10.6
with Mem 100 91.1 98.7 97.6 93.3 8.2
Table 6: Ablation studies of MemRAG. “w/o Mem” indi-
cates the removal of MemRAG.
Case Study. Figure 3 demonstrates the reasoning process
of MobileRAG when no suitable match is found by Mem-
RAG. We find that MobileRAG has the following advan-
tages: 1) utilizing InterRAG enables comprehensive under-
standing of user queries; 2) leveraging LocalRAG allows the
system to directly launch relevant applications from the lo-
cal app library, minimizing accidental touches or erroneous
actions; 3) MobileRAG’s robust retrieval capabilities help
reduce the number of operational steps, particularly benefit-
ing long sequential tasks. Figure 4 illustrates the inference
process when MemRAG successfully identifies a matching
instance. Through MemRAG, It can effectively extract and
leverage information from historically correct operation se-
quences, thereby enhancing agent performance and further
decreasing the number of operational steps.
Conclusion
In this paper, we introduce MobileRAG, a novel framework
that enhances mobile agents with Retrieval-Augmented
Generation (RAG). MobileRAG is designed to address sev-
eral limitations of current LLM-based mobile agents, such
as over-reliance on language model comprehension, lack of
interaction with external environments, and absence of ef-
fective memory. It integrates three components—InterRAG,
LocalRAG, and MemRAG—to enable more accurate un-

derstanding of user queries and efficient execution of com-
plex mobile tasks. Additionally, we propose MobileRAG-
Eval, a challenging benchmark featuring real-world mobile
scenarios that require external knowledge support. Exten-
sive experiments on MobileRAG-Eval demonstrate that Mo-
bileRAG outperforms state-of-the-art methods by 10.3%,
achieving better results with fewer operational steps.
References
Anthropic, S. 2024. Model card addendum: Claude 3.5
haiku and upgraded claude 3.5 sonnet. URL https://api. se-
manticscholar. org/CorpusID , 273639283.
Bai, J.; Bai, S.; Chu, Y .; Cui, Z.; Dang, K.; Deng, X.; Fan,
Y .; Ge, W.; Han, Y .; Huang, F.; et al. 2023. Qwen technical
report. arXiv preprint arXiv:2309.16609 .
Cabri, G.; Leonardi, L.; and Zambonelli, F. 2000. XML
dataspaces for mobile agent coordination. In Proceedings
of the 2000 ACM symposium on Applied computing-Volume
1, 181–188.
Chen, B.; Linz, D. D.; and Cheng, H. H. 2008. XML-based
agent communication, migration and computation in mo-
bile agent systems. Journal of Systems and Software , 81(8):
1364–1376.
Chen, J.; Lin, H.; Han, X.; and Sun, L. 2024. Benchmark-
ing large language models in retrieval-augmented genera-
tion. In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, 17754–17762.
Chou, Y .-C.; Ko, D.; and Cheng, H. H. 2010. An embed-
dable mobile agent platform supporting runtime code mo-
bility, interaction and coordination of mobile agents and host
systems. Information and software technology , 52(2): 185–
196.
Ciancarini, P.; Tolksdorf, R.; and Zambonelli, F. 2002. Coor-
dination middleware for XML-centric applications. In Pro-
ceedings of the 2002 ACM symposium on Applied comput-
ing, 336–343.
Es, S.; James, J.; Anke, L. E.; and Schockaert, S. 2024. Ra-
gas: Automated evaluation of retrieval augmented genera-
tion. In Proceedings of the 18th Conference of the European
Chapter of the Association for Computational Linguistics:
System Demonstrations , 150–158.
Fan, W.; Ding, Y .; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua,
T.-S.; and Li, Q. 2024. A survey on rag meeting llms: To-
wards retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining , 6491–6501.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2(1).
Gemini; Georgiev, P.; Lei, V . I.; Burnell, R.; Bai, L.; Gu-
lati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.; et al.
2024. Gemini 1.5: Unlocking multimodal understand-
ing across millions of tokens of context. arXiv preprint
arXiv:2403.05530 .Gou, B.; Wang, R.; Zheng, B.; Xie, Y .; Chang, C.; Shu, Y .;
Sun, H.; and Su, Y . 2024. Navigating the digital world as
humans do: Universal visual grounding for gui agents. arXiv
preprint arXiv:2410.05243 .
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning. arXiv preprint arXiv:2501.12948 .
Gupta, S.; Ranjan, R.; and Singh, S. N. 2024. A comprehen-
sive survey of retrieval-augmented generation (rag): Evolu-
tion, current landscape and future directions. arXiv preprint
arXiv:2410.12837 .
Gur, I.; Furuta, H.; Huang, A.; Safdari, M.; Matsuo, Y .; Eck,
D.; and Faust, A. 2023. A real-world webagent with plan-
ning, long context understanding, and program synthesis.
arXiv preprint arXiv:2307.12856 .
Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang,
Y .; Wang, Z.; Dong, Y .; Ding, M.; et al. 2024. Cogagent:
A visual language model for gui agents. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 14281–14290.
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
L´ala, J.; O’Donoghue, O.; Shtedritski, A.; Cox, S.; Ro-
driques, S. G.; and White, A. D. 2023. Paperqa: Retrieval-
augmented generative agent for scientific research. arXiv
preprint arXiv:2312.07559 .
Lee, S.; Choi, J.; Lee, J.; Wasi, M. H.; Choi, H.; Ko, S.;
Oh, S.; and Shin, I. 2024. Mobilegpt: Augmenting llm with
human-like app memory for mobile task automation. In Pro-
ceedings of the 30th Annual International Conference on
Mobile Computing and Networking , 1119–1133.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, G.; and Li, Y . 2022. Spotlight: Mobile ui understanding
using vision-language models with a focus. arXiv preprint
arXiv:2209.14927 .
Liao, M.; Wan, Z.; Yao, C.; Chen, K.; and Bai, X. 2020.
Real-time scene text detection with differentiable binariza-
tion. In Proceedings of the AAAI conference on artificial
intelligence , volume 34, 11474–11481.
Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang,
Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino:
Marrying dino with grounded pre-training for open-set ob-
ject detection. In European conference on computer vision ,
38–55. Springer.
Rawles, C.; Li, A.; Rodriguez, D.; Riva, O.; and Lillicrap, T.
2023. Androidinthewild: A large-scale dataset for android
device control. Advances in Neural Information Processing
Systems , 36: 59708–59728.

Sun, L.; Chen, X.; Chen, L.; Dai, T.; Zhu, Z.; and Yu, K.
2022. Meta-gui: Towards multi-modal conversational agents
on mobile gui. arXiv preprint arXiv:2205.11029 .
Tamayo, A.; Granell, C.; and Huerta, J. 2011. Instance-
based XML data binding for mobile devices. In Proceed-
ings of the Third International Workshop on Middleware for
Pervasive Mobile and Embedded Computing , 1–8.
Wang, J.; Xu, H.; Jia, H.; Zhang, X.; Yan, M.; Shen,
W.; Zhang, J.; Huang, F.; and Sang, J. 2024a. Mobile-
agent-v2: Mobile device operation assistant with effective
navigation via multi-agent collaboration. arXiv preprint
arXiv:2406.01014 .
Wang, J.; Xu, H.; Ye, J.; Yan, M.; Shen, W.; Zhang, J.;
Huang, F.; and Sang, J. 2024b. Mobile-agent: Autonomous
multi-modal mobile device agent with visual perception.
arXiv preprint arXiv:2401.16158 .
Wang, Z.; Xu, H.; Wang, J.; Zhang, X.; Yan, M.; Zhang,
J.; Huang, F.; and Ji, H. 2025. Mobile-Agent-E: Self-
Evolving Mobile Assistant for Complex Tasks. arXiv
preprint arXiv:2501.11733 .
Wen, H.; Li, Y .; Liu, G.; Zhao, S.; Yu, T.; Li, T. J.-J.; Jiang,
S.; Liu, Y .; Zhang, Y .; and Liu, Y . 2023. Empowering llm to
use smartphone for intelligent task automation. CoRR .
Wen, H.; Li, Y .; Liu, G.; Zhao, S.; Yu, T.; Li, T. J.-J.; Jiang,
S.; Liu, Y .; Zhang, Y .; and Liu, Y . 2024. Autodroid: Llm-
powered task automation in android. In Proceedings of the
30th Annual International Conference on Mobile Computing
and Networking , 543–557.
Wu, B.; Li, Y .; Fang, M.; Song, Z.; Zhang, Z.; Wei, Y .; and
Chen, L. 2024. Foundations and recent trends in multimodal
mobile agents: A survey. arXiv preprint arXiv:2411.02006 .
Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y .; Liu, Z.;
Wang, S.; Han, X.; Liu, Z.; et al. 2024. Visrag: Vision-based
retrieval-augmented generation on multi-modality docu-
ments. arXiv preprint arXiv:2410.10594 .
Zhang, C.; Yang, Z.; Liu, J.; Li, Y .; Han, Y .; Chen, X.;
Huang, Z.; Fu, B.; and Yu, G. 2025. Appagent: Multimodal
agents as smartphone users. In Proceedings of the 2025 CHI
Conference on Human Factors in Computing Systems , 1–20.
Zhao, P.; Zhang, H.; Yu, Q.; Wang, Z.; Geng, Y .; Fu, F.;
Yang, L.; Zhang, W.; Jiang, J.; and Cui, B. 2024. Retrieval-
augmented generation for ai-generated content: A survey.
arXiv preprint arXiv:2402.19473 .Appendix A: Benchmark Details
Our comprehensive benchmark evaluates mobile agents
across 80 distinct task instructions. We designed and in-
troduced 36 novel task instructions (outlined in Table 7)
specifically crafted for this benchmark, complemented by
44 supplementary instructions sourced from Mobile-Agent-
v2 that address standard operations. Together, these instruc-
tions form a robust evaluation framework that comprehen-
sively assesses mobile device operations across a wide range
of scenarios.
Appendix B: MemRAG Effectiveness Analysis
As shown in Table 8, we conducted a detailed evaluation
of MemRAG’s effectiveness using 15 distinct task groups,
systematically categorized into three similarity levels:
•Original Instructions : Tasks executed in historical op-
erations
•High Similarity (exceeding 80%) : Tasks sharing core
operational patterns with historical cases
•Low Similarity (below 80%) : Tasks with varying de-
grees of novelty
The task design systematically varies complexity while
maintaining real-world applicability, allowing for a detailed
evaluation of MemRAG’s retrieval-augmented operational
capabilities across the spectrum of task similarity.
Appendix C: Case Study and Test Results
Appendix C presents a detailed case study showcasing sev-
eral test results to demonstrate the performance of Mobil-
eRAG.
Figure 5 shows the overall framework of MobileRAG is
tested on a real Xiaomi 15 Pro device. This diagram provides
an overview of how MobileRAG operates, emphasizing its
architecture and functionality in real-time testing.
Figure 6 showcases the testing process with three differ-
ent apps, highlighting the superiority of LocalRAG in app
retrieval. The performance improvements and efficiency of
LocalRAG are clearly depicted, focusing on its ability to re-
trieve app data effectively.
Figure 7 emphasizes how MemRAG enhances the re-
trieval of past successful experiences building upon the pre-
vious figure’s instructions, significantly reducing the steps
and complexity involved in operations. The graph illustrates
how MemRAG streamlines the process by reusing previ-
ously successful tasks, improving the user experience by
minimizing effort and time.
The integrated RAG architecture offers synergistic advan-
tages: InterRAG addresses ambiguous instructions by incor-
porating external context to ensure precise app selection. Lo-
calRAG facilitates accurate app targeting, while MemRAG
leverages historical successes to retrieve optimal operational
sequences, thereby reducing both cognitive load and execu-
tion complexity.

Category Basic Instructions Advanced Instructions
Common1. Set an alarm for 8 am.
2. Write a reminder: I have dinner tonight at 6:30.
3. Find a short, trendy video tutorial on cooking steak and
like it.1. Find a coffee shop near me on Google maps, and then
open Google message to tell Mike that I will be at the place
waiting for him (give him the specific coffee shop name).
2. Summarize the prices of three nearby gas stations in
‘Google Maps’ app and record these information into the
‘Notepad’ app.
3. Find me a coffee shop near me on ’Google Maps’ app that
sells birthday cakes and is within a 10-minute drive. Find the
phone number and create a new note for it in the ‘Notes’ app.
Music single1. Listen to a piano music in the‘YouTube Music’ APP.
2. Play exclusive album ‘Ear’.
3. Play ‘Shape of You’ by Ed Sheeran on Spotify.1. Find the album ‘Happier Than Ever’, and add it to my
library.
2. Play Taylor Swift’s ‘Love Story’, and add this song to a
new playlist named ‘Agent Playlist’.
3. Add the Taylor Swift’s song “1989 (Taylor’s Version)” to
playlist
Movie single1.Like the TV series ‘Good Boy’ season 1 in the ‘Prime
Video’ app.
2. Add the TV series ‘The Boys’ to the viewing list.
3. Add a comedy movie to the Watchlist in the ‘Prime Video’
app.1. Add the movie ‘Heads of state.’ to the Watchlist and like
it
2. Watch the TV series ‘A Dream Within A Dream’ and com-
ments it.
3. Find the TV series ‘The Boys’, select season 3 and add it
to the Watchlist.
Exteral information1. What is the NBA Score today? Send the results to Jelly.
2. Send a message to Jelly to tell her the app that can watch
Squid Game.
3. What is the USD to CNY exchange rate? Find the specific
price rather than link, send a message to Jelly to tell her the
results.1. Download the app to watch Squid Game.
2. Download the most popular music app and open it.
3. Search for the date of the next Winter Olympics opening
ceremony and then set a reminder for that date in the ’Cal-
endar’ app.
Movie multi1. Open the ‘Prime Video’ app to find the movie in my
Watchlist, and then search for this movie in X, choose the
first posts to read and comment it, then send a message to
invite Jelly to watch this movie in ’Google Message’ app.
2. Find the TV series ‘The Boys’, select season 3 and add it
to the Watchlist, then summarize the introduction, and send
a message to Mike inviting him to watch ‘The Boys’ season
3, tell him the introduction.
3. Please open YouTube to search for one short video about
‘Heads of State’ and like it, then find an app that can watch
the movie ‘Heads of State’, and add it to the Watchilist.1. Open the app that can watch ‘A Dream Within A Dream’,
then find the top 3 comments, write these comments into the
notepad.
2. I need to find an app to watch the TV series ‘The Boys’,
then find its introduction, and finally create a note to sum-
marize the introduction.
3. Open the app that can watch ‘A Dream Within A Dream’
in the ‘IQIYI’ app, then find details like director and rating
score, then message to Jelly the info through ’Google Mes-
sage’ app.
Music multi1. Play a piano music in the ‘YouTube Music’ app and then
open the ‘X’ app to search for the song name.
2. Open the ‘Spotify’ app to find the song ’Shape of You’
by Ed Sheeran and add it to liked song, then share this song
to Mike and invite him to join on Spotify through ’Google
Message’ app.
3. Add Taylor Swift’s song “1989 (Taylor’s Version)” to
playlist, and then search for this song in the ‘X’ app, fol-
low one related account and enter his/her post.1. Open two local music apps that can play songs. Check if
the song “Happier Than Ever” is available in each of these
apps. Then, record the availability status of the song in a
notepad app.
2. Open two local music apps that can play songs. Check if
Taylor Swift’s song “1989 (Taylor’s Version)” is available in
each of these apps, if it is add this song into playlist. Then,
send a message to Mike to tell him the availability status of
the song on Google Message.
3. Open three local music apps that can play songs. Check if
the song “Happier Than Ever” is available in each of these
apps. Then, record the availability status of the song in a
notepad app.
Table 7: The 36 Original Task Instructions Specifically Designed for Our Mobile Agent Benchmark

Original Instructions 80-100% Similarity Instructions 0-80% Similarity Instructions
Send a greeting message to Jelly. Send a greeting message to Jelly.Play ‘Shape of You’ by Ed Sheeran on
Spotify.
Like the TV series ‘Good Boy’ season 1
in the ‘prime video’ app.Send a message to Jelly to let her know
which app can be used to watch Squid
Game.Search for the date of the next Winter
Olympics opening ceremony and then
set a reminder for that date in the ’Cal-
endar’ app.
Download the app to watch Squid Game.Add a comedy movie to the Watchlist in
the ‘prime video’ app.Summarize the prices of three nearby
gas stations in ‘Google maps’ app and
record these information into the in the
’Notepad’ app .
Find the TV series ‘The Boys’, select
season 3 and add it to the Watchlist, then
summarize the introduction, and send a
message to Mike inviting him to watch
‘The Boys’ season 3, tell him the intro-
duction.Send a message to Jelly to invite him to
watch ‘The Boys’ season 3, tell him the
introduction.Please open YouTube to search for one
short video about ‘Heads of State’ and
like it, and then find an app that can
watch movie ‘Heads of State’, and add
it to the Watchilist.
Open two local music apps that can play
songs. Check if the song ‘Happier Than
Ever’ is available in each of these apps.
Then, record the availability status of the
song in the ‘Notepad’ app.Open three local music apps that can
play songs. Check if the song ’Happier
Than Ever’ is available in each of these
apps. Then, record the availability status
of the song in in the ‘Notepad’ app.Play a piano music in the ‘YouTube Mu-
sic’ app and then open the ‘X’ app to
search for the song name.
Table 8: MemRAG Task Instruction
Figure 5: Framework Diagram on the Real Machine - Xiaomi 15 Pro.

Figure 6: This case study explores the workflow integration across three distinct mobile applications.
Figure 7: The diagram represents the workflow of multiple apps operating simultaneously with MemRAG support.