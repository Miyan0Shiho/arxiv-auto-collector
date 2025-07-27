# IM-Chat: A Multi-agent LLM-based Framework for Knowledge Transfer in Injection Molding Industry

**Authors**: Junhyeong Lee, Joon-Young Kim, Heekyu Kim, Inhyo Lee, Seunghwa Ryu

**Published**: 2025-07-21 06:13:53

**PDF URL**: [http://arxiv.org/pdf/2507.15268v1](http://arxiv.org/pdf/2507.15268v1)

## Abstract
The injection molding industry faces critical challenges in preserving and
transferring field knowledge, particularly as experienced workers retire and
multilingual barriers hinder effective communication. This study introduces
IM-Chat, a multi-agent framework based on large language models (LLMs),
designed to facilitate knowledge transfer in injection molding. IM-Chat
integrates both limited documented knowledge (e.g., troubleshooting tables,
manuals) and extensive field data modeled through a data-driven process
condition generator that infers optimal manufacturing settings from
environmental inputs such as temperature and humidity, enabling robust and
context-aware task resolution. By adopting a retrieval-augmented generation
(RAG) strategy and tool-calling agents within a modular architecture, IM-Chat
ensures adaptability without the need for fine-tuning. Performance was assessed
across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and
GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance
and correctness, and was further supplemented by automated evaluation using
GPT-4o guided by a domain-adapted instruction prompt. The evaluation results
indicate that more capable models tend to achieve higher accuracy, particularly
in complex, tool-integrated scenarios. Overall, these findings demonstrate the
viability of multi-agent LLM systems for industrial knowledge workflows and
establish IM-Chat as a scalable and generalizable approach to AI-assisted
decision support in manufacturing.

## Full Text


<!-- PDF content starts -->

1  
IM-Chat: A Multi -agent LLM -based Framework for 
Knowledge Transfer in Injection Molding Industry  
  
Junhyeong Lee1,†, Joon -Young Kim1, 2,†, Heekyu Kim1,†, Inhyo Lee1 and Seunghwa Ryu1,* 
 
 
Affiliations  
1Department of Mechanical Engineering, Korea Advanced Institute of Science and 
Technology (KAIST), 291 Daehak -ro, Yuseong -gu, Daejeon 34141, Republic of Korea  
2Industrial Intelligence Research Group, AI/DX Center, Institute for Advanced Engineering 
(IAE), Yongin, Republic of Korea  
 
*Corresponding authors' e -mail: ryush@kaist.ac.kr   
 
Keywords  
Injection Molding, Knowledge  Transfer, Large Language Model, Artificial Intelligence  
 
  

2 Abstract  
The injection molding industry faces critical challenges in preserving and transferring field 
knowledge, particularly as experienced workers retire and multilingual barriers hinder effective 
communication. This study introduces IM -Chat, a multi -agent frame work based on large 
language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM -
Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and 
extensive field data modeled through a data -driven process condition generator that infers 
optimal manufacturing settings from environmental inputs such as temperature and humidity,  
enabling robust and context -aware task resolution. By adopting a retrieval -augmented 
generation (RAG) strategy and tool -calling agents within a modular architecture, IM -Chat 
ensures adaptability without the need for fine -tuning. Performance was assessed across 100 
single -tool and 60 hybrid tasks for GPT -4o, GPT -4o-mini, and GPT -3.5-turbo by domain 
experts using a 10 -point rubric  focused on relevance and correctness, and was further 
supplemented by automated evaluation using GPT -4o guided by a domain -adapted instruction 
prompt.  The evaluation results indicate that more capable models tend to achieve higher 
accuracy, particularly i n complex, tool -integrated scenarios. Overall, these findings demonstrate 
the viability of multi -agent LLM systems for industrial knowledge workflows and establish IM -
Chat as a scalable and generalizable approach to AI -assisted decision support in manufact uring.   

3 1. Introduction  
1.1. Research background  
Injection molding, a core method for mass production of plastic components, 
underpins a significant portion of the global manufacturing industry  [1,2] . Characterized by 
high-mix, low -volume production, injection molding demands advanced on -site expertise to 
manage intricate process settings, ensure production efficiency, and maintain rigorous quality 
control across a diverse product spectrum  [3]. 
In current practice, on-site operators manually inspect products and iteratively adjust 
process parameters to maintain quality  [4]. This process heavily relies on individual experience 
and expertise , as external factors such as  changes in the external environment, characteri stics 
of injection molding machines, and variations in the physical properties of resins directly 
impact the final product qualit y [5]. Consequently, operators must acquire extensive domain  
knowledge to rapidly and accurately respond to such multifaceted  influences .  
Traditionally, knowledge transfer in the injection molding industry has relied on 
structured rule sets for addressing specific defects, equipment troubleshooting manuals, and 
apprenticeship models in which experienced operators directly impar t their expertise to less 
experienced or newly hired workers  [6–8]. However, these legacy approaches face growing 
limitations.  
A shrinking and aging workforce increasingly challenges the industry . As skilled  
veteran operators retire and fewer newcomers enter the field, the implicit knowledge 
accumulated over decades is at risk of  being lost  [9]. Furthermore, as a low -margin industry, 
many injection molding companies have reduced staffing or replaced expert operators with less 
experienced personnel to lower labor costs  [10]. These changes highlight the limitations of 

4 traditional apprenticeship -based knowledge transfer and underscore the urgent need for new 
approaches that can systematize, preserve, and diss eminate expert -level process knowledge  
[11]. Additionally, globalization has further exacerbated the challenge, as the increase in a 
multilingual workforce creates communication barriers in knowledge sharing  [12]. In the 
absence of mechanisms to translate and deliver operational insights across languages and skill 
levels, production efficiency is compromised. As such, there is a critical need for knowledge 
transfer systems that not only preserve expert knowledge but also ensure seamless, real -time 
communication among diverse operators.  To achieve this, the system needs to go beyond 
simple documentation and adopt innovative approaches that can capture, reason with, and 
deliver operational expertise in dynamic and interpretable ways. Such advancement s are 
essential for sustaining operational continuity and enhancing competitiveness in an 
increasingly complex global market.  
Recent advancements in large language models (LLMs) present a promising solution 
to this challenge. LLMs  have significantly improv ed the accuracy of natural language 
reasoning tasks, enabling them to excel in general applications such as inference, mathematical 
problem -solving, logical reasoning, and everyday conversations  [13–17]. Techniques like 
chain of thought prompting and in-context learning have further enhanced their reasoning 
capabilities by guiding models to systematically decompose complex problems and recognize 
patterns from contextual examples without modifying model parameters  [18,19] . These 
advancements indicate that LLMs have the potential to function as systems that seamlessly 
communicate with workers and facilitate knowledge transfer through reasoning.  
Nevertheless, while LLMs have shown strong general performance, domain -specific 
applications still face limita tions , particularly the problem of hallucination, where models 
generate plausible but inaccurate content.  To mitigate these issues, researchers have developed 

5 techniques like fine -tuning  [20–22], which allows users to adapt pre -trained models to specific  
tasks with a labeled dataset, and retrieval -augmented generation (RAG), which integrates 
external knowledge sources into model prompts to improve factual accuracy  [23–26]. 
Additionally, agent -based approaches have emerged, incorporating APIs and algorit hms that 
enable models to dynamically leverage external tools for more precise task execution  [27–29]. 
Expanding on agent -based approaches, multi -agent systems have gained traction as a 
promising framework, where multiple specialized agents collaborate within structured 
workflows to tackle complex problems. These developments reflect a broader shift towards 
creating more flexible, context -aware AI systems that can reliably handle both general and 
specialized tasks, paving the way for their application as  effective knowledge transfer systems  
[30,31] . 
This study introduces IM-Chat  (Injection Molding Chatt ing interface), a multi -agent 
framework that leverages knowledge - and tool -augmented LLMs to address the growing need 
for effective knowledge transfer in  the injection molding industry.  To the best of our knowledge, 
this study is the first  work  to propose an LLM agent -based approach tailored specifically for 
this domain . The benefits of integrating LLM agents into knowledge transfer systems can be 
summarized as follows:  
(i) Resilient knowledge preservation : IM-Chat overcomes the limitations of time -
consuming and fragile apprenticeship systems, offering a scalable solution in the face 
of workforce aging and language diversity.  
(ii) Contextual adaptability via RAG : By integrating RAG, IM -Chat synthesizes 
insights from legacy documentation, expert knowledge, and online resources, enabling 
context -aware and dynamically adaptive knowledge transfer.  

6 (iii) Data -driven precision : IM -Chat incorporates additional AI modules , such as 
surrogate and generative models , via tool -calling, allowing it to provide accurate, 
quantitative guidance for production decision -making under varying environmental 
conditions.  
The overall schematic representation of this fra mework is shown in Figure 1 . 
 The subsequent sections of this paper outline the design and validation of IM -Chat, 
focusing on two key knowledge transfer types : (i) domains with limited field knowledge, where 
experiential or textual information serves as the primary source, and (ii) domains with 
extensive field knowledge, where machine learning and generative models facilitate data -
driven decision -making. The Method s section details the curation of knowledge sources and 
datasets, the development of LLM workflows, and the integration of generative and surrogate 
models to enhance adaptability. The Results  section evaluates IM -Chat’s performance, 
assessing its effective ness across different knowledge transfer scenarios and its compatibility 
with various LLMs. Finally, the Discussion  and Conclusion  sections summarize the study’s 
key contributions, address existing challenges, and explore future directions for leveraging 
LLM-based workflows as advanced knowledge transfer systems in the injection molding 
industry.   

7  
Figure 1 . Schematic overview of the IM -Chat framework. Tasks are categorized as either 
general or injection molding –specific and are processed using ReAct or pl an-and-execute 
workflows. The system integrates limited documented knowledge (e.g., manuals, 
troubleshooting tables) with extensive domain knowledge modeled by diffusion -based 
generative models trained on manufacturing field data.   


8 1.2. Related work  
1.2.1.  Knowledge tra nsfer in manufacturing  
 Knowledge transfer approaches and expert systems have significantly contributed to 
advancements in production efficiency, knowledge management, and process optimization in 
the manufacturing and injection molding indust ries. Numerous studies have explored 
integrating knowledge -based methodologies with advanced computational tools to capture the 
expertise of skilled operators and effectively transfer it for broader application.  
 Early research established foundational fr ameworks by integrating group technology 
and schedule management to optimize production processes  [32]. Subsequent studies further 
expanded these approaches by incorporating domain expertise into automated systems, 
reducing reliance on extensive datasets , streamlining parameter optimization, and minimizing 
operational costs [33]. In manufacturing operations, rule -based and expert systems have been 
widely applied to enhance process efficiency across various domains  [34,35] . 
 To improve knowledge transfer, frameworks have been developed to transform tacit 
knowledge into explicit knowledge, addressing uncertainties in production evaluation [36]. 
This is exemplified in systems that integrate expert knowledge into machine learni ng models 
for accurate quality predictions, even with limited data  availability  [37]. Additionally, case -
based reasoning (CBR) has been applied extensively in injection molding manufacturing 
processes to assist in problem -solving and defect analysis [38]. 
 Advancements in knowledge representation have brought significant improvements to 
the injection molding industry. The use of knowledge graphs has improved the structuring and 
accessibility of domain -specific information  [39], while retrieval systems  based on transformer 
architectures (e.g., BERT) have enhanced the efficiency and precision of knowledge queries  

9 [40]. Similarly, frameworks such as blackboard -based expert systems and interactive 
knowledge -based CAD systems have demonstrated effectivene ss in areas such as mold design, 
defect analysis, and process planning by integrating both static rules and real -time operational 
data [41,42] . 
 Other research has targeted  manufacturability assessment and early -stage design 
optimization. Hierarchical knowledge frameworks and feature representations have improved 
manufacturability evaluations, while structured knowledge systems have helped mitigate risks 
and enhance design c onsistency in mold manufacturing  [43,44] . For instance, interactive CAD 
systems and integrated knowledge management platforms have been instrumental in navigating 
mold design processes and managing historical design knowledge  [45,46] . These systems are  
particularly valuable for small enterprises, enabling scalable and efficient knowledge transfer  
[47]. 
 In parallel, expert systems have been applied to production planning and scheduling 
tasks . Approaches such as fuzzy decision models and multi -objective optimization frameworks 
have enabled improved resource allocation and operational flexibility in complex 
manufacturing scenario s [48,49] . More recent studies have introduced knowledge -based 
synthesis systems and preference -guided optimiza tion strategies to enhance the adaptability of 
process modeling and configuration  [50,51] . 
 Despite these advancements, challenges persist in leveraging tacit knowledge and 
adapting to dynamic industrial conditions. Current systems largely focus on expli cit knowledge, 
often neglecting the intuitive insights that arise from the experience of skilled workers. 
Additionally, many frameworks lack the flexibility to adapt to real -time changes in operational 
environments, limiting their scalability and practical  application. Furthermore, significant 
technical expertise is often required to deploy and use these systems, creating barriers for less -

10 skilled workers and restricting accessibility  [52,53] . 
 
1.2.2. LLM application in manufacturing  
 The integration of LLMs into manufacturing is accelerating the shift from Industry 4.0 
to Industry 5.0, emphasizing human -centric innovation, enhanced automation, and adaptive 
intelligence  [54,55] . Unlike traditional AI systems that primarily rely on structured datasets 
and predefined rules, LLMs possess reasoning capabilities, allowing them to process more 
flexible inputs and enhancing adaptive decision -making. Furthermore, techniques such as 
strate gic prompting, RAG, tool -calling agents, and multi -agent systems further refine their 
accuracy and responsiveness.  
Leveraging these advantages, recent research has increasingly explored their 
application in predictive maintenance, human -machine interactio n, smart automation, process 
planning, and additive manufacturing, driving the development of more adaptive and efficient 
manufacturing environments  [56]. The most straightforward way to integrate LLMs into 
manufacturing is through direct prompting, wher e users interact with a pre -trained model using 
carefully designed queries  [57–59]. For instance, pre -trained LLMs alone have been 
successfully applied to automate the extraction and classification of knowledge from 
unstructured text, reducing the need f or manual preprocessing in data analytics and machine 
learning workflows  [57]. Furthermore, they have demonstrated effectiveness in generating 
context -aware text, which has been leveraged for various tasks such as creating, modifying, 
and refining code, enhancing automation in manufacturing processes  [58]. 
 While prompting offers utility in natural language tasks, its reliance on the model’s pre -
trained knowledge limits its ability to incorporate real -time data or specialized domain 

11 information , often l eading to hallucinated or inaccurate outputs . To address this, several studies 
have fine -tuned LLMs using supplementary domain data such as machine documentation, fault 
logs, and taxonomies to improve their industrial relevance and task alignment  [60–64]. In one 
example, fine -tuning enabled the model to infer the relationship between process parameters 
and product quality outcomes  [65]. Alternatively, RAG provides a scalable means of enhancing 
factual grounding without the computational burden of fine -tuning. By retrieving relevant 
external knowledge and embedding it into the prompt, RAG enhances context alignment and 
output accuracy  [60,63,66,67] .  
Beyond RAG, which solely integrates external information for answering queries, LLM 
agents can also interface with various tools, including sensors, control programs, robotic 
systems, and machine operators, to execute appropriate actions. Their ability to reason and take 
actions has played a crucial role in advancing the automation of manufacturing syste ms, 
spanning design, control, and production. For instance, an LLM agent integrated with 
specialized tools and domain -specific knowledge can enhance process planning for composite 
structures by optimizing decision -making and workflow efficiency  [68]. As manufacturing 
applications advance toward higher levels of autonomy, multiple LLM -based agents 
collaborate to manage complex tasks. Several studies have explored such multi -agent 
frameworks, where distinct roles are assigned to each LLM agent, facilitating  seamless 
communication, information exchange, and coordinated task execution  [69–75]. 
  

12 2. Method   
This section presents the architecture and implementation details of the proposed 
knowledge transfer system, IM -Chat, developed for the injection molding domain. We begin 
with an overview of the framework and its operational flow, describing how various 
components , including agents, tools, and knowledge sources , collaborate to process user 
queries  (Method 2.1 ). We then outline two representative knowledge transfer scenarios:  (i) 
tasks solvable through limited domain knowledge extracted from manuals and troubleshooting 
tables, supported via RAG  (Method 2.2 ); and (ii) tasks requiring extensive field knowledge 
modeled from real -world production data, handled u sing data -driven generative models 
integrated into the agent workflow  (Method 2.3 ). 
 
2.1. IM-Chat workflow  
IM-Chat processes worker input tasks through three sequential stages: (i) input 
formatting, (ii) task solving, and (iii) output formatting ( Figure 2 ). Upon receiving a task, the 
system first reformulates the input using context from recent conversation history to ensure 
clarity and minimize redundant LLM inference. This is handled by the Task Formatter , an 
LLM module that summarizes and restructures t he concatenated user input and prior dialogue 
into a predefined query format. To support multilingual users, the formatted input is translated 
into English, and both the translated version and the original language metadata are retained 
for use during the final output stage.  
  

13  
Figure 2 . Overview of the IM -Chat framework for knowledge transfer in injection molding. 
The system consists of input formatting, task solving (via ReAct or plan -and-execute), and 
output formatting. Four tools (search, troubleshooti ng table, manual retriever, and diffusion 
model) are used during execution to generate responses.  
 
  


14 Following input formatting, a Classifier module  determines whether the query pertains 
to injection molding or not.  This classification is necessary to preserve the generality of the 
framework, as using tools and prompts tailored for injection molding could otherwise constrain 
its broader applicability. It also helps minimize operational costs by preventing unnecessary  
invocation of specialized modules for simple, non -injection -related queries. In addition, it 
enables more specialized and carefully designed workflows for handling injection molding -
specific tasks.  The Classifier module , guided by a system prompt, is desi gned to output a parsed 
string , either “ injection ” or “ no_injection ”, based on an overall assessment of the formatted 
input, which includes both the user’s task and the preceding chat history, in conjunction with 
the capabilities of the available tools wit hin the framework.  
For general queries unrelated to injection molding, a ReAct Agent is employed  [27], 
which operates solely with an internet search tool to retrieve recent information necessary for 
addressing the query. The use of a ReAct  Agent allows the system to combine information 
retrieval and reasoning, enabling flexible  handling of a wide range of general queries. By 
limiting the tools to internet search for non -injection -related tasks, the framework avoids 
unnecessary invocation of tools  specialized for injection -related tasks, thereby reducing 
operational costs and maintaining system robustness. This design choice ensures that IM -Chat 
remains adaptable and efficient even when addressing queries outside of its core manufacturing 
focus. 
For injection molding -related queries, the task is handled through a carefully designed 
planning and execution workflow. Upon receiving the formatted task, a Planner module  
decomposes the task into multiple subtasks. Each subtask is sequentially addressed, s tarting 
with the first subtask at the top of the plan. Specifically, the Executor module  executes only 
the foremost subtask provided by the Planner, ensuring that the system maintains tight control 

15 over the process of solving the task. After execution, a Supervisor module  evaluates whether 
the combined results sufficiently answer the original query. If the results are deemed 
satisfactory, the system proceeds to the output formatting phase. Conversely, if the results are 
insufficient, the Replanner  redefines and adjusts the plan, triggering another cycle of execution.  
During the execution phase, IM-Chat utilizes four external tools:  
(i) an internet searcher,  
(ii) a troubleshooting table retriever,  
(iii) a manufacturing manual retriever, and  
(iv) a diffusion model trained on real -world injection molding machine data  
The first three tools provide limited but interpretable domain knowledge via a RAG 
strategy ( Method 2.2). The diffusion model, by contrast, offers quantitative recommendations 
for process parameters under given environmental conditions ( Method 2.3). 
After the task -solving phase is completed, the system proceeds to the response 
formatting stage, managed by a Reporter module . For non -injection molding -related tasks, the 
Reporter  summarizes the response based on the original input and the generated answer. In the 
case of injection molding related tasks, the Reporter  reviews both the original input and the 
temporary agent history accumulated during the processing of a single query,  including the 
planning steps, tool calls, and execution results, to generate a concise and contextually accurate 
summary.  The finalized response is then translated into the language originally used by the 
worker, ensuring seamless communication. Finally, the pair of input and response is appended 
to the global chat history, enabling the system to maintain continuity across interactions and 
support future reasoning processes . 

16 The IM -Chat framework was implemented using the LangChain  and LangGraph  
libraries,  which provided the necessary infrastructure for orchestrating workflows involving 
multiple agents and reasoning processes using external tools. A user -friendly chat interface was 
developed using Streamlit  to facilitate live interaction with field operator s. Detailed 
descriptions of each LLM module’s prompt design  is provided in the Supplementary Note 1 . 
 
2.2. LLM -based knowledge transfer system for limited domain knowledge  
2.2.1. Knowledge extraction  
 To develop an effective knowledge transfer system powered by the LLM, it is essential 
to collect and structure the necessary production knowledge systematically. In this study, we 
gathered and extracted knowledge from various sources, including interviews with injection 
molding experts, troubleshooting ch arts standardized by the industry, and technical resources. 
Through expert interviews, we collected practical knowledge on how to adjust process 
parameters for specific defect types and prioritized these adjustments based on their 
effectiveness. Additional ly, structured troubleshooting guidelines were referenced to address 
common defects such as short shots, weld lines, and warping, thereby elucidating the 
correlations between those defects and the associated process parameters  [76]. Furthermore, 
technical books and reference materials on plastic injection molding enriched the knowledge 
base by validating expert opinions and troubleshooting data, offering a broader and more 
comprehensive context  [77–80]. By integrating these insights, we identifi ed the relationships 
between specific defect types such as burrs, short shots, and warping and process parameters 
like injection speed, injection pressure, and hold time. This allowed us to systematically 
prioritize corrective actions to produce products w ithout defects. The compiled knowledge is 

17 summarized in a process adjustment table ( Figure 3a ), where each parameter is marked with a 
‘+’ or ‘ –’ sign to indicate whether it should be increased or decreased in response to a specific 
defect. In addition, the  relative importance of each parameter is indicated by a priority value, 
with values closer to 1 denoting higher importance.  
The operation manual for the injection molding machine was also considered a critical 
source of knowledge. For this study, we utili zed the operation manual of a specific injection 
molding machine, available in PDF format, several pages of which are shown in Figure 3b . 
The manual spans a total of 227 pages and covers a wide range of operational and technical 
topics . It provided essential guidance on material -specific process parameter ranges, pre -
operation checkpoints, and regular inspection schedules, as well as further operational details 
that support comprehensive field implementation.  For example, the manual outlined the 
appropriate ranges for injection pressure, barrel temperature, and hold time for materials like 
ABS and PP, enabling operators to adjust process parameters effectively based on resin 
characteristics. Additionally, the manual included detailed preoperative chec kpoints, such as 
inspecting lubrication systems, hydraulic components, and mold alignment, to prevent machine 
malfunctions and ensure stable operation. The regular inspection routines, categorized by 
weekly, monthly, quarterly, semi -annual, and annual sche dules, specified tasks such as motor 
temperature checks, heat band cleaning, and hydraulic oil replacement to maintain optimal 
machine performance.  
 By combining the insights from expert interviews, troubleshooting charts, and machine 
manuals, this study d eveloped a robust and comprehensive knowledge base. This knowledge 
base serves as the foundational component of the proposed LLM -based knowledge transfer 
system, enabling adaptive and efficient decision -making across diverse production scenarios.  

18  
Figure 3. The documented knowledge base  for IM-Chat. (a) The process adjustment table 
based on defect types. (b) The example pages from a 227 -page machine operation manual.   
 
  


19 2.2.2. Retrieval augmented generation (RAG) for IM-Chat 
As described earlier, when an injection molding -related task is identified, IM -Chat does 
not rely solely on LLM reasoning. Instead, it aims to generate factually grounded response by 
retrieving relevant. For the documented  knowledge type, IM-Chat employs a RAG  approach 
to address input subtasks . Specifically, upon receiving the first subtask in the plan generated 
by the Planner , the Executor module  dynamically selects the appropriate knowledge source, 
including a table retriever, manual retriever, or interne t searcher, depending on the subtask 
requirements.  After knowledge retrieval, the output is passed to the Supervisor module , which 
assesses whether the retrieved content sufficiently addresses the subtask requirements. Based 
on this assessment, the system either reports the response as final or returns to the Replanner  
to revise the plan and re -execute the subtask  (Figure 4a ). To ensure efficient retrieval and 
accurate grounding, the system applies distinct RAG configurations tailored to each type of 
knowledge source.  
To begin with, the knowledge retrieval process for the troubleshooting table proceeds 
as follows.  The table, stored in CSV format, is initially loaded and partitioned into chunks, 
where each chunk corresponds to a specific defect type or priority label. Each chunk contains 
detailed descriptions of recommended process parameter adjustments to address the 
corresponding defect, based on senior operators’ experiential knowledge. Each chunk is then 
embedded into a vector space using OpenAI's em bedding models, with one embedding vector 
representing one chunk. When a subtask query is issued, a similarity search is performed 
against the vector database using cosine similarity to retrieve the top two most relevant chunks. 
The retrieved information i s subsequently summarized and reformulated, and the resulting 
augmented output, which consists of the input subtask query paired with the summarized 

20 retrieved information, is passed to the Supervisor  for evaluation in the context of solving the 
original in put task.  This retrieval and augmentation procedure is illustrated in Figure 4b . 
For the manufacturing manual, a structured retrieval pipeline was constructed to handle 
the large volume of information contained within the document. The manual, consisting o f 227 
pages in PDF format, was first processed using a vision -language model (GPT -4o) from the 
LlamaParse  library. Each page was parsed into a structured JSON object, with the extracted 
content stored in Markdown format within each entry . This process enabled simple textual 
descriptions to be generated for embedded images, while tables were cleanly extracted and 
represented in a plain -text format with consistent alignment.  Following this, each page -level 
chunk was embedded into a vector space using Open AI's embedding, with one embedding 
vector representing one page. When a subtask query is issued, an initial similarity search is 
conducted using cosine similarity to retrieve the top 20 candidate chunks. From these 
candidates, 7 chunks are selected using t he Maximal Marginal Relevance (MMR) method, 
which balances relevance to the query and diversity among the selected chunks.  The MMR 
objective can be mathematically defined as:  
MMR (𝐷𝑖)=argmax
𝐷𝑖∈𝑅\𝑆[𝜆∙𝑠𝑖𝑚(𝐷𝑖,𝑄)−(1−𝜆)∙max
𝐷𝑗∈𝑆𝑠𝑖𝑚(𝐷𝑖,𝐷𝑗)] (1) 
where 𝐷𝑖 is the embedding of a candidate chunk under consideration, R is the set of retrieved 
embeddings , S is the set of already selected embedding vectors , Q is the query embedding, and 
𝜆 controls the trade -off between relevance and diversity . Here, 𝑠𝑖𝑚(∙,∙) denotes the cosine 
similarity between two embedding vectors  [81]. The selected chunks are then summarized and 
reformulated in combination with the subtask query, and the resulting augmented output is 
passed to the Supervisor  for evaluation. This retrieval procedure is visually summarized in 
Figure 4b . 

21 Finally, for cases where the required information is not available from the internal 
troubleshooting table or the manufacturing manuals, an external internet search is performed. 
The Tavily API  is adopted to retrieve external information relevant to the subtask query. The 
retrieved knowledge is processed similarly through summarization and augmentation steps 
before being passed to the Supervisor for evaluation.  In situations where external retrie val is 
unnecessary or impractical, the Executor  may opt to solve the subtask purely through internal 
LLM inference without invoking any tools. The specific prompts used for each retrieval tool 
are provided in Supplementary Note 2 . 
 
2.3. LLM -based knowledge  transfer system for extensive domain knowledge  
2.3.1. Diffusion model  
Knowledge in the injection molding process extends beyond general insights obtained 
from expert interviews or technical resources on adjusting process parameters for specific 
defect typ es. While these sources offer a strong foundation, incorporating extensive production 
data significantly broadens the scope of field knowledge. With sufficient production data, 
deeper knowledge extraction and application become possible, transforming raw d ata into a 
valuable resource for improving manufacturing processes. For example, real -world production 
data enables more accurate factory simulations, allowing manufacturers to replicate actual 
operating conditions and dynamically optimize process paramete rs. Moreover, such data 
supports the development of the knowledge transfer system capable of providing precise 
solutions for determining optimal processing conditions, surpassing traditional heuristic 
approaches. By integrating knowledge derived from both experts and data, a more adaptive and 
precise system can be established, ultimately enhancing production efficiency and quality 
control in complex manufacturing environments.  

22 Among various forms of domain knowledge, production data plays a critical role in  
advancing knowledge transfer systems that integrate expert insight with data -driven 
approaches. To explore this potential, we developed an external recommendation tool based on 
diffusion model to suggest optimal process parameters for defect -free producti on. The model 
was trained on structured production data comprising environmental conditions (factory and 
machine temperature and humidity), process parameters, including injection speed and 
pressure, and associated product quality outcomes.  
To implement th is system, we constructed an external tool based on the Classifier -Free 
Guidance Diffusion Model (CFGDM ) [82], a type of score -based generative model known for 
its training stability and sample quality. The model takes five input features: product class 
(good or defective), factory temperature, factory humidity, machine temperature, and machine 
humidity. Based on these inputs, it generates ten output parameters that guide defect -free 
production, including  three -stage injection speeds, three -stage injection pressures, three -stage 
injection positions, and hold time. Validation experiments assessed whether the generated 
process parameters consistently led to good -quality outcomes under various experimental 
conditions, demonstrating the model’s practical applicability to real -world manufacturing 
context. Details of the diffusion model architecture, training procedure, and evaluation results 
are provided in Supplementary Note  3, with further technical explanatio ns available in Kim 
et al [4]. 
 
2.3.2. Diffusion model tool -calling for recommending process parameters  
To leverage the trained diffusion model for recommending process parameters based 
on the worker’s input query, it is essential to provide structured and well -defined input 
conditions, as the model requires inputs in a specified vector format. To ensure thi s, an 

23 Diffusion I nput Formatter  module is placed before the diffusion model to standardize and 
organize incoming queries. If the initial input from the worker lacks sufficient detail, the system 
prompts the worker to supplement or clarify the information t o meet the required specifications. 
Once properly formatted, the input is passed to the diffusion model, which generates 64 
candidate outputs representing potential process parameter sets tailored to the specified 
conditions.  
Following candidate generation , a CatBoost -based surrogate model is employed to 
filter and rank the outputs by predicting the likelihood of defect -free production based on the 
generated process parameters. Each candidate is scored according to its predicted probability 
of yielding a go od (non -defective) outcome. [83]. The candidate with the highest predicted 
success probability is selected, and its corresponding process parameters are reformulated into 
a structured textual output. This refined result is then integrated into the overal l workflow and 
passed to the Supervisor for evaluation alongside other task -solving outputs. This pipeline is 
illustrated in Figure 4c . Representative system prompts for both the input formatter and the 
output parsing format  are provided in the Supplementary Note 4 . 

24  
Figure 4 . Execution framework of IM -Chat for injection molding task resolution.  (a) Plan –
execute –replan loop overview.  (b) Knowledge retrieval using the troubleshooting table and 
manufacturing manual.  (c) Process parameter generation from textual environmental inputs 
using a diffusion model.  
  


25 3. Results  
3.1. Validation of IM-Chat with single -tool task  
Since IM -Chat incorporates a translator module that standardizes all user inputs into 
English before task execution, it inhe rently supports multilingual interactions. As an initial 
check, we validated the system using a small set of representative queries in four languages 
(Korean, English, Vietnamese, and Thai), considering the multilingual environment commonly 
observed in the  injection molding industry in South Korea. These results, summarized in 
Supplementary Note  5, indicate that IM -Chat accurately processed the translated queries and 
returned contextually appropriate responses . Therefore, in the subsequent evaluations, we 
focused on English -language queries, as all user inputs are translated into English by default 
before processing.  
Building on this, we conducted a comprehensive evaluation to benchmark the system’s 
reasoning and tool selection capabilities using a curated set of 100 English -language queries.  
Specifically, in this section, we tested IM -Chat’s performance on 100 single tool tasks, where 
each query was designed to invoke one of four specific tools: troubleshooting table retrieval, 
manufacturing manual retrieva l, diffusion model generation, or ordinary reasoning tasks that 
may or may not involve internet search. For each category, 25 representative queries were 
prepared, resulting in a dataset of 100 question –answer pairs (i.e., chatting logs).   
To assess the qu ality of the previously generated question and answer pairs, we 
developed a customized evaluation metric tailored to the specific needs of the injection molding 
domain. Conventional automatic metrics widely used in natural language processing, such as 
BLEU , ROUGE, perplexity, and semantic similarity  [84,85] , mainly focus on linguistic fluency 
or surface -level similarity. While these metrics are effective for general language tasks, they 

26 are not suitable for evaluating technical accuracy and domain relevance, which are essential in 
manufacturing tasks that require expert -level understanding. In addition, most queries in 
injection molding have clearly defined and objective answers, making the factual correctness 
and appropriateness of the retrieved in formation especially important. These considerations led 
us to design a domain -specific evaluation framework that better reflects the practical 
performance of IM -Chat.  
Based on the customized rubric, human experts were asked to evaluate each response 
on a 10-point scale in terms of relevance and correctness. In addition to expert evaluation, an 
automated LLM -based assessment was also conducted, inspired by the recently proposed 
“LLM -as-a-judge” approach [86]. Specifically, GPT -4o was prompted with an eval uation 
instruction adapted to the domain ( Supplementary Note 6) that guided the model to assess the 
responses with a focus on both correctness and relevance. This LLM -based assessment was 
conducted on the same 10 -point scale to explore whether a language m odel could reliably 
approximate or potentially substitute for expert evaluation, by providing scalable 
benchmarking and independent judgments. The overall evaluation procedure is illustrated in 
Figure 5a . 
The left panel of Figure 5b  shows the evaluation scores assigned by human experts 
across four task categories and three IM -Chat variants (GPT -4o, GPT -4o-mini, and GPT -3.5-
turbo). Overall, GPT -4o exhibited a consistent trend of achieving higher scores across most 
task types, with par ticularly strong performance on troubleshooting table and diffusion model 
tasks, indicating superior technical grounding and response integration capabilities.  In contrast, 
GPT -3.5-turbo recorded the lowest scores in most categories, particularly strugglin g with 
retrieving information from manufacturing manuals and invoking the diffusion model, 
highlighting its limitations in handling domain -specific generation tasks. More specifically, 

27 GPT -3.5-turbo often failed to comprehend the context of lengthy documen ts or to extract 
correct values from tables embedded within manuals. It also exhibited a notably higher 
frequency of tool misselection during both the planning and execution stages . 
The right panel of Figure 5b  presents evaluation scores assigned by GPT -4o acting as 
an automated evaluator. Broadly, a similar pattern of performance degradation across task 
categories was observed. However, the LLM evaluator consistently assigned low scores to 
diffusion model tasks, which often involve interpreting numerical o utputs typically 
encountered in real manufacturing environments. This suggests that when it comes to 
evaluating responses that reflect numerical reasoning encountered in the field, especially those 
produced by deep learning models trained on knowledge spec ific to the domain, evaluation by 
LLMs may fall short of accurate judgment. The LLM evaluator may be more sensitive to 
fluency or coherence at the surface level than to the domain correctness of the generated content.  
To further assess the agreement betwee n human and LLM evaluations, we computed 
Pearson correlation coefficients between the two scoring approaches across the three IM -Chat 
variants. The resulting correlations were −0.14 (GPT -4o), 0.25 (GPT -4o-mini), and 0.35 (GPT -
3.5-turbo). These relatively l ow values indicate weak alignment between the LLM evaluations 
and expert human evaluations, particularly in the case of models that demonstrated superior 
performance. This misalignment highlights that while LLM -based assessment can provide 
scalable benchma rking, it cannot yet be relied upon as a full substitute for expert judgment, 
especially in industrial domains requiring precise technical evaluation, such as injection 
molding. These findings underscore the importance of human -in-the-loop validation when 
evaluating domain -specific chatbot systems.  
Figure 5c  illustrates the average response latency per query for each IM -Chat variant. 
GPT -4o demonstrated a favorable trade -off between latency (24.4 seconds) and accuracy, 

28 consistently outperforming the other m odels in correctness. In contrast, GPT -3.5-turbo 
achieved the fastest response time (17.2 seconds), but this speed often came at the expense of 
performance quality, as it tended to produce shallow reasoning and premature answers. Notably, 
GPT -4o-mini exhib ited the highest latency (32.4 seconds), despite being a lighter model. This 
elevated latency was primarily caused by inefficiencies in its plan -and-execute loop, where 
suboptimal planning and unnecessary re -planning led to excessive iterations and indecis ive 
behavior. These findings underscore that latency alone is insufficient to assess overall system 
effectiveness. Effective deployment requires not only fast inference, but also coherent 
reasoning strategies and well -coordinated agentic control.  
Figure 5d  presents the average cost per query for each IM -Chat variant. GPT -4o 
incurred the highest cost at $0.0190 per query – approximately 10 times more expensive than 
GPT -4o-mini ($0.0017) and about 7 times more than GPT -3.5-turbo ($0.0027). This reflects 
the p remium pricing associated with GPT -4 class modes, which offer state -of-the-art 
performance but at significantly higher inference cost. These results emphasize that pricing 
considerations should be carefully weighed against performance and latency metrics w hen 
selecting LLM variants for real -world deployment.  

29  
Figure 5. Evaluation of IM -Chat performance for single -tool tasks.  (a) Overview of the 
evaluation framework including task types, evaluation setup, and scoring criteria. (b) 
Evaluation scores from human experts (left) and an LLM evaluator (right). GPT -4o achieved 
the highest performance, while GPT -3.5-turbo underperformed  due to frequent tool 
misselection. (c) Average response latency per query, with GPT -3.5-turbo being fastest but 
least reliable. (d) Average query cost, with GPT -4o being most expensive due to deeper 
reasoning and higher token usage.  


30 3.2. Validation of IM -Chat for multi -tool tasks  
To further assess the capabilities of IM -Chat in more complex and realistic industrial 
scenarios, we designed a set of hybrid tasks that combine limited knowledge retrievable from 
documentation or the internet with extensive domai n knowledge embedded within a diffusion 
model trained on field data. Figure 6a  illustrates the evaluation framework applied to these 
tasks. The task pool consists of three hybrid categories, each comprising 20 tasks: (1) Diffusion 
model + Troubleshooting t able, (2) Diffusion model + Manual retrieval, and (3) Diffusion 
model + Internet search. Each task is specifically designed to require the combined use of both 
tools listed in its category, ensuring that IM -Chat must effectively coordinate multiple 
knowled ge sources to generate a valid and complete response.  Each pair of input query and 
generated response  was evaluated using the same protocol described in the evaluation 
involving a single tool: human experts assessed the relevance and accuracy of the chatbot’s 
response using a 10 -point rubric, while GPT 4o also served as an automated evaluator using a 
prom pt adapted to the domain.   
Figure 6b  compares evaluation scores for hybrid tasks involving both retrieval and 
reasoning based on diffusion models across three variants of IM -Chat. According to the human 
expert evaluation, GPT -4o significantly outperforms t he other two models across all hybrid 
task categories. In particular, it exhibits outstanding performance in tasks combining diffusion 
models and troubleshooting (score = 9.2), while maintaining relevantly strong accuracy in 
hybrid tasks involving manuals and internet -based sources (scores = 7 and 5.4, 
respectively).   This superior performance reflects not only the model’s intrinsic reasoning 
capability but also the effective coordination among its internal components: the Planner, the 
Executor, and the Supervisor. These components work in concert to decompose complex 
queries that span multiple sources, select the appropriate tools (e.g., diffusion model or 
document retriever s), and ensure the consistency of the generated responses. As a result, GPT -

31 4o demon strates a robust ability to integrate numeric inference with structured retrieval or 
contextual understanding from static sources. In contrast, GPT -3.5-turbo and GPT -4o-mini 
show substantial performance degradation, especially in tasks requiring deeper coo rdination 
between retrieved knowledge and generated inferences.   
In contrast to human evaluations, the assessments conducted by the LLM show 
compressed score distributions and weaker model differentiation. GPT -4o, which was clearly 
superior in human evalua tions, received moderate but not outstanding scores from the LLM 
evaluator (all ~ 3–4). Meanwhile, GPT -3.5-turbo scores the highest among the three models, 
which contradicts the human assessments. This discrepancy suggests that the LLM evaluator 
may overest imate fluency or plausibility, particularly when responses appear reasonable at the 
surface level, even if they lack the necessary technical rigor. The evaluation prompt may still 
face challenges in accurately assessing reasoning that is both numerically g rounded and derived 
from multiple information sources. In particular, integration of diffusion model outputs may 
not be easily validated by an external language model prompt without access to intermediate 
reasoning steps.  
Figure 6c  presents the average res ponse latency per query for hybrid tasks involving 
both diffusion model inference and retrieval. GPT -4o recorded the highest latency (66.0 
seconds), followed closely by GPT -4o-mini (63.2 seconds), while GPT -3.5-turbo responded 
substantially faster (33.3 se conds). This latency pattern primarily reflects the overhead of 
diffusion model execution, which takes approximately 37 seconds per inference and dominates 
the overall response time. The notably lower latency of GPT -3.5-turbo is partially due to its 
freque nt failure to invoke the diffusion model when required, resulting in shorter but 
incomplete reasoning paths. This discrepancy explains its paradoxically fast response times 
despite lower task performance.  

32 Figure 6d  shows the average query cost for multi -tool tasks involving both retrieval 
and diffusion model inference. GPT -4o incurred a significantly higher cost ($0.0427 per query), 
more than twice the cost of its single -tool usage. This increase stems from longer r easoning 
traces, higher token usage, and multiple agent interactions required to coordinate complex tasks. 
In contrast, GPT -4o-mini and GPT -3.5-turbo maintained low costs (~$0.0027 –0.0028), 
although the latter's cost remained artificially low due to freque nt omission of diffusion model 
invocation, as observed in earlier latency and accuracy analyses.  
  

33  
Figure 6 . Evaluation of IM -Chat performance for hybrid (multi -tool) tasks. (a) Overview of 
the evaluation framework including task types, evaluation setup,  and scoring criteria. (b) 
Evaluation scores from human experts (left) and an LLM evaluator (right). GPT -4o showed 
the best overall performance, while GPT -3.5-turbo failed to invoke required tools, resulting in 
lower human scores. (c) Average response late ncy per query, with GPT -3.5-turbo being fastest 
but often incomplete in execution. (d) Average query cost, with GPT -4o being most expensive 
due to complex tool coordination and higher token usage.  


34 4. Discussion  
This study presents a multi -agent LLM -based framework for knowledge transfer in the 
injection molding industry. The proposed system integrates both documented knowledge (as 
found in troubleshooting tables and manufacturing manuals) and extensive field data  modeled 
using a diffusion generative model. This dual approach goes beyond previously trained general 
knowledge, enabling the delivery of precise, quantitative, and field -relevant expertise to 
support decision making in real industrial settings. Comparati ve validation using three 
representative OpenAI models demonstrated that more capable, recently developed models, 
particularly GPT -4o, produce more accurate and cost -effective responses in complex task 
scenarios. These results highlight the practical value  of leveraging high -performance LLMs for 
intelligent industrial assistance.  
Beyond its immediate application, the IM -Chat framework offers strong potential for 
extensibility. By adopting a RAG approach instead of fine -tuning, the system remains modular 
and adaptable. New tools and pre -trained AI models can be easily integrated into the multi -
agent architecture without modifying the core LLMs. While this work focused on the use of a 
diffusion model for quantitative inference, other generative or predictive m odels can be 
similarly incorporated, depending on the target domain. The flexible model integration 
capability of the framework enables its generalization beyond injection molding, and this study 
represents the first demonstration of such a system specific ally designed for knowledge transfer 
in this domain.  
However, several limitations remain. First, the current framework primarily operates 
on textual and structured data and has limited capabilities for extracting information from 
visual content such as ima ges, tables, diagrams, and engineering drawings. These elements are 
frequently encountered in manufacturing documentation. While initial efforts were made to 

35 incorporate vision language models (VLMs) for parsing such content, the results revealed 
limitatio ns in accuracy and completeness. In addition, even purely text -based documents often 
contain nonstandard formats, special symbols, or embedded figures that further hinder reliable 
parsing. These observations underscore the need for more robust multimodal c apabilities that 
can jointly process both textual and visual information with higher precision.  
Second, the system currently relies on high -performance commercial LLMs (e.g., GPT -
4o). This reliance raises concerns related to data privacy, vendor lock -in, and operational cost, 
particularly for continuous or real -time deployment in industrial settings. To address these 
constraints, future implementations may benefit from incorporating locally hosted small 
language models (SLMs) or fine -tuned lightweight agent s that are optimized for specific tasks. 
This approach would enable cost -effective and secure operation without sacrificing accuracy.  
Third, the current system supports only a limited set of external tools. These primarily 
include troubleshooting tables an d diffusion -based generative models. In real -world 
manufacturing environments, however, a much broader ecosystem of tools is used. These 
include empirical equation solvers, process simulators, optimization modules, and CAD -
integrated analysis software. Suc h tools span the full lifecycle of industrial operations, from 
product design to failure analysis. To fully support these workflows, the IM -Chat framework 
must evolve into a generalized tool -calling infrastructure that is capable of interfacing with 
hetero geneous tools through modular adapters, dynamic invocation protocols, and error -
tolerant input -output handling. This would allow intelligent agents to autonomously leverage 
diverse computational resources, thereby enabling deeper task specialization and mo re precise 
decision support across domains.  
Finally, the quality of the system’s responses was found to be significantly influenced 
by the specificity of user input. In cases where user queries are vague or under -defined, the 

36 system tends to generate lower -quality outputs or, in some cases, no answer at all. This reflects 
the classical principle known as “Garbage In, Garbage Out” (GIGO). To mitigate this issue, 
future versions of the framework should incorporate input refinement mechanisms, such as 
guided question templates, interactive clarification agents, or input -to-query formatting 
modules. These components would help users articulate their intent more clearly and would 
improve the quality, relevance, and interpretability of downstream responses.  
Addressing current challenges such as limited multimodal understanding, concerns 
associated with the use of commercial LLMs, including data privacy, operational cost, and 
long-term sustainability, restricted tool interoperability, and sensitivity to vague u ser inputs 
will enhance the robustness and practical value of the system. As these aspects are gradually 
improved, the framework is expected to evolve into a scalable and general purpose solution for 
knowledge transfer in injection molding and related doma ins. 
  

37 5. Conclusion  
This study presents IM -Chat, a multi -agent LLM framework that bridges documented 
knowledge sources and data -driven insights to support industrial knowledge transfer in the 
injection molding domain. The framework combines RAG with exter nal tool integration, 
enabling it to process structured documents, utilize generative models, and deliver context -
aware, task -specific responses.  
 Through extensive benchmarking, we demonstrated that high -performance models 
such as GPT -4o consistently outp erform lighter alternatives in complex scenarios involving 
tool selection and numerical reasoning. While GPT -4o-mini and GPT -3.5-turbo offer 
advantages in terms of speed and cost, they often struggle with domain -sensitive subtasks that 
require deep reasoni ng or multi -step planning.  
A key strength of IM -Chat lies in its modular and extensible architecture. This 
architecture enables seamless extension to additional models and straightforward adaptation to 
different industrial domains with minimal adjustments.  As a result, IM -Chat serves not only as 
a solution tailored to injection molding but also as a prototype for domain -specific AI systems 
applicable across the broader manufacturing sector.  
Nonetheless, several limitations remain. While the system incorpora tes vision -language 
models to parse visual content such as diagrams, tables, and engineering drawings, its ability 
to extract and utilize such information remains limited and requires further improvement. In 
addition, the framework currently depends on com mercial LLM APIs, raising concerns 
regarding cost, data privacy, and long -term sustainability. Finally, tool interoperability is still 
constrained, and the system's performance remains sensitive to the specificity and clarity of 
user queries.  

38 To address th ese challenges, future work will focus on integrating multimodal 
capabilities, expanding tool interfaces to include simulators and engineering design software, 
and incorporating user input refinement strategies to improve the relevance and clarity of 
syste m responses. Moreover, lightweight, task -specialized agents with selective fine -tuning 
will be explored to reduce deployment overhead while preserving domain adaptability.  
Collectively, this work demonstrates not only the technical feasibility of deploying  
multi -agent LLM systems in real industrial contexts but also offers a roadmap for future 
intelligent systems capable of adaptive reasoning, cross -domain generalization, and human -
aligned interaction in manufacturing environments.  
 
  

39 References  
[1] Agrawal AR, Pandelidis IO, Pecht M. Injection‐molding process control —A review. Polym Eng 
Sci 1987;27:1345 –57. 
[2] Bharti PK, Khan MI. Recent methods for optimization of plastic injection molding process -A 
retrospective and literature review 2010.  
[3] Fernandes C, Pontes AJ, Viana JC, Gaspar‐Cunha A. Modeling and Optimization of the 
Injection‐Molding Process: A Review. Advances in Polymer Technology 2018;37:429 –49. 
[4] Kim J -Y, Kim H, Nam K, Kang D, Ryu S. Development of an injection molding production 
cond ition inference system based on diffusion model. J Manuf Syst 2025;79:162 –78. 
[5] Czepiel M, Bańkosz M, Sobczak -Kupiec A. Advanced Injection Molding Methods. Materials 
2023;16:5802.  
[6] Khosravani MR, Nasiri S, Weinberg K. Application of case -based reasoni ng in a fault detection 
system on production of drippers. Appl Soft Comput 2019;75:227 –32. 
[7] Khosravani MR, Nasiri S, Reinicke T. Intelligent knowledge -based system to improve injection 
molding process. J Ind Inf Integr 2022;25:100275.  
[8] Zhou Z -W, Ting  Y-H, Jong W -R, Chiu M -C. Knowledge management for injection molding 
defects by a knowledge graph. Applied Sciences 2022;12:11888.  
[9] The Manufacturing Institute. The Aging of the Manufacturing Workforce: Challenges and Best 
Practices. 2019.  
[10] Masato D , Kim SK. Global workforce challenges for the mold making and engineering industry. 
Sustainability 2024;16:346.  
[11] Lin C -Y, Gim J, Shotwell D, Lin M -T, Liu J -H, Turng L -S. Explainable artificial intelligence 
and multi -stage transfer learning for injectio n molding quality prediction. J Intell Manuf 2024:1 –
20. 
[12] Escobedo E, Jett F, Lao T. Communication issues in a multilingual business environment: 
Insights from managers. Global Journal of Business Research 2012;6:69 –75. 
[13] Brown T, Mann B, Ryder N, Su bbiah M, Kaplan JD, Dhariwal P, et al. Language models are 
few-shot learners. Adv Neural Inf Process Syst 2020;33:1877 –901. 
[14] Kojima T, Gu SS, Reid M, Matsuo Y, Iwasawa Y. Large language models are zero -shot 
reasoners. Adv Neural Inf Process Syst 2022;35:22199 –213. 
[15] Achiam J, Adler S, Agarwal S, Ahmad L, Akkaya I, Aleman FL, et al. Gpt -4 technical report. 
ArXiv Preprint ArXiv:230308774 2023.  
[16] Dubey A, Jauhri A, Pandey A, Kadian A, Al -Dahle A, Letman A, et al. The llama 3 herd of 
models. ArX iv Preprint ArXiv:240721783 2024.  
[17] Team G, Anil R, Borgeaud S, Alayrac J -B, Yu J, Soricut R, et al. Gemini: a family of highly 
capable multimodal models. ArXiv Preprint ArXiv:231211805 2023.  
[18] Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, et a l. Self -consistency improves chain 
of thought reasoning in language models. ArXiv Preprint ArXiv:220311171 2022.  

40 [19] Wei J, Wang X, Schuurmans D, Bosma M, Xia F, Chi E, et al. Chain -of-thought prompting 
elicits reasoning in large language models. Adv Neur al Inf Process Syst 2022;35:24824 –37. 
[20] Hu EJ, Shen Y, Wallis P, Allen -Zhu Z, Li Y, Wang S, et al. Lora: Low -rank adaptation of large 
language models. ArXiv Preprint ArXiv:210609685 2021.  
[21] Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. Qlora: Eff icient finetuning of quantized 
llms. Adv Neural Inf Process Syst 2024;36.  
[22] Touvron H, Lavril T, Izacard G, Martinet X, Lachaux M -A, Lacroix T, et al. Llama: Open and 
efficient foundation language models. ArXiv Preprint ArXiv:230213971 2023.  
[23] Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, et al. Survey of hallucination in natural language 
generation. ACM Comput Surv 2023;55:1 –38. 
[24] Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, et al. Retrieval -augmented 
generation for knowledge -intensive  nlp tasks. Adv Neural Inf Process Syst 2020;33:9459 –74. 
[25] Tonmoy SM, Zaman SM, Jain V, Rani A, Rawte V, Chadha A, et al. A comprehensive survey 
of hallucination mitigation techniques in large language models. ArXiv Preprint 
ArXiv:240101313 2024.  
[26] Huang L, Yu W, Ma W, Zhong W, Feng Z, Wang H, et al. A survey on hallucination in large 
language models: Principles, taxonomy, challenges, and open questions. ArXiv Preprint 
ArXiv:231105232 2023.  
[27] Yao S, Zhao J, Yu D, Du N, Shafran I, Narasimhan K, et a l. React: Synergizing reasoning and 
acting in language models. ArXiv Preprint ArXiv:221003629 2022.  
[28] Wang L, Ma C, Feng X, Zhang Z, Yang H, Zhang J, et al. A survey on large language model 
based autonomous agents. Front Comput Sci 2024;18:186345.  
[29] Shen Y, Song K, Tan X, Li D, Lu W, Zhuang Y. Hugginggpt: Solving ai tasks with chatgpt and 
its friends in hugging face. Adv Neural Inf Process Syst 2024;36.  
[30] Wu Q, Bansal G, Zhang J, Wu Y, Zhang S, Zhu E, et al. Autogen: Enabling next -gen llm 
applicati ons via multi -agent conversation framework. ArXiv Preprint ArXiv:230808155 2023.  
[31] Guo T, Chen X, Wang Y, Chang R, Pei S, Chawla N V, et al. Large language model based multi -
agents: A survey of progress and challenges. ArXiv Preprint ArXiv:240201680 202 4. 
[32] Kusiak A. Manufacturing systems: A knowledge -and optimization -based approach. J Intell 
Robot Syst 1990;3:27 –50. 
[33] Maier M, Kunstmann H, Zwicker R, Rupenyan A, Wegener K. Autonomous and data -efficient 
optimization of turning processes using exper t knowledge and transfer learning. J Mater Process 
Technol 2022;303:117540.  
[34] Bhatt MR, Buch S. Application of rule based and expert systems in various manufacturing 
processes —A review. Smart Innovation, Systems and Technologies 2016;50:459 –65. 
https:// doi.org/10.1007/978 -3-319-30933 -0_46.  
[35] Jayaraman V, Srivastava R. Expert systems in production and operations management: current 
applications and future prospects. International Journal of Operations & Production 
Management 1996;16:27 –44. 

41 [36] Arévalo  F, MP CA, Ibrahim MT, Schwung A. Production assessment using a knowledge transfer 
framework and evidence theory. IEEE Access 2022;10:89134 –52. 
[37] Link P, Poursanidis M, Schmid J, Zache R, von Kurnatowski M, Teicher U, et al. Capturing and 
incorporating expert knowledge into machine learning models for quality prediction in 
manufacturing. J Intell Manuf 2022;33:2129 –42. https://doi.org/10.1007/S10845 -022-01975 -
4/FIGURES/10.  
[38] Khosravani MR, Nasiri S. Injection molding manufacturing process: Review of c ase-based 
reasoning applications. J Intell Manuf 2020;31:847 –64. 
[39] Zhou Z -W, Ting Y -H, Jong W -R, Chen S -C, Chiu M -C. Development and Application of 
Knowledge Graphs for the Injection Molding Process. Machines 2023;11:271.  
[40] Zhou Z -W, Jong W -R, Ting Y -H, Chen S -C, Chiu M -C. Retrieval of Injection Molding 
Industrial Knowledge Graph Based on Transformer and BERT. Applied Sciences 2023;13:6687.  
[41] Kwong CK, Smith GF. A computational system for process design of injection moulding: 
Combining a blackboard -based expert system and a case -based reasoning approach. The 
International Journal of Advanced Manufacturing Technology 1998;14:350 –7. 
[42] Y. J. Huh SGK. A knowledge based CAD system for concurrent product design in injection 
moulding. Int J Comput Integr Manuf 1991;4:209 –18. 
[43] Pratt SD, Sivakumar M, Manoochehri S. A Knowledge -Based Engineering System for the 
Design of Injection Molded Plastic Parts. Proceedings of the ASME Design Engineering 
Technical Conference 2021;Part F167972 -13:287 –95. https ://doi.org/10.1115/DETC1993 -0316.  
[44] Kim SG, Suh NP. Knowledge -based synthesis system for injection molding. Robot Comput 
Integr Manuf 1987;3:181 –6. https://doi.org/10.1016/0736 -5845(87)90100 -1. 
[45] Jong W -R, Ting Y -H, Li T -C, Chen K -Y. An integrated ap plication for historical knowledge 
management with mould design navigating process. Int J Prod Res 2013;51:3191 –205. 
[46] Mok CK, Chin KS, Ho JKL. An interactive knowledge -based CAD system for mould design in 
injection moulding processes. The International  Journal of Advanced Manufacturing 
Technology 2001;17:27 –38. 
[47] Vukašinović N, Vasić D, Tavčar J. Application of knowledge management system to injection 
mold design and manufacturing in small enterprises. DS 92: Proceedings of the DESIGN 2018 
15th Inter national Design Conference, 2018, p. 1733 –44. 
[48] Custodio LMM, Sentieiro JJS, Bispo CFG. Production planning and scheduling using a fuzzy 
decision system. IEEE Transactions on Robotics and Automation 1994;10:160 –8. 
[49] Metaxiotis KS, Askounis D, Psarras  J. Expert systems in production planning and scheduling: 
A state -of-the-art survey. J Intell Manuf 2002;13:253 –60. 
https://doi.org/10.1023/A:1016064126976/METRICS.  
[50] Nagarajan HPN, Mokhtarian H, Jafarian H, Dimassi S, Bakrani -Balani S, Hamedi A, et al.  
Knowledge -based design of artificial neural network topology for additive manufacturing 
process modeling: A new approach and case study for fused deposition modeling. Journal of 
Mechanical Design 2019;141:442. https://doi.org/10.1115/1.4042084.  
[51] Karls son I, Bandaru S, Ng AHC. Online knowledge extraction and preference guided multi -
objective optimization in manufacturing. IEEE Access 2021;9:145382 –96. 

42 [52] Kumara SRT, Joshi S, Kashyap RL, Moodie CL, Chang TC. Expert systems in industrial 
engineering. In t J Prod Res 1986;24:1107 –25. 
[53] Byrd TA. Expert systems implementation: interviews with knowledgeengineers. Industrial 
Management & Data Systems 1995;95:3 –7. 
[54] Maddikunta PKR, Pham Q -V, Prabadevi B, Deepa N, Dev K, Gadekallu TR, et al. Industry 5.0: 
A survey on enabling technologies and potential applications. J Ind Inf Integr 2022;26:100257.  
[55] Kohl L, Eschenbacher S, Besinger P, Ansari F. Large language model -based chatbot for 
improving human -centricity in maintenance planning and operations. PHM Society European 
Conference, vol. 8, 2024, p. 12.  
[56] Mustapha KB. A survey of emerging applications of large language models for problems in 
mechanics, product design, and manufacturing. Advanced Engineering Informatics 
2025;64:103066.  
[57] Matthes M, Guhr O, Krockert M, Munkelt T. Leveraging LLMs for  Information Extraction 
in Manufacturing, 2024, p. 355 –66. https://doi.org/10.1007/978 -3-031-71637 -9_24.  
[58] Jignasu A, Marshall K, Ganapathysubramanian B, Balu A, Hegde C, Krishnamurthy A. Towards 
Foundat ional AI Models for Additive Manufacturing: Language Models for G -Code Debugging, 
Manipulation, and Comprehension 2023.  
[59] Kernan Freire S, Foosherian M, Wang C, Niforatos E. Harnessing Large Language Models for 
Cognitive Assistants in Factories. Proceed ings of the 5th International Conference on 
Conversational User Interfaces, CUI 2023, Association for Computing Machinery, Inc; 2023. 
https://doi.org/10.1145/3571884.3604313.  
[60] Liu X, Erkoyuncu JA, Fuh JYH, Lu WF, Li B. Knowledge extraction for additive  manufacturing 
process via named entity recognition with LLMs. Robot Comput Integr Manuf 2025;93. 
https://doi.org/10.1016/j.rcim.2024.102900.  
[61] Naqvi SMR, Ghufran M, Varnier C, Nicod JM, Javed K, Zerhouni N. Unlocking maintenance 
insights in industrial text through semantic search. Comput Ind 2024;157 –158. 
https://doi.org/10.1016/j.compind.2024.104083.  
[62] Wang P, Karigiannis J, Gao RX. Ontology -integrated tuning of large language model for 
intelligent maintenance. CIRP Annals 2024;73:361 –4. 
https://doi .org/https://doi.org/10.1016/j.cirp.2024.04.012.  
[63] Xu Q, Zhou G, Zhang C, Chang F, Cao Y, Zhao D. Generative AI and DT integrated intelligent 
process planning: a conceptual framework. The International Journal of Advanced 
Manufacturing Technology 2024;1 33:2461 –85. https://doi.org/10.1007/s00170 -024-13861 -9. 
[64] Zhou B, Li X, Liu T, Xu K, Liu W, Bao J. CausalKGPT: Industrial structure causal knowledge -
enhanced large language model for cause analysis of quality problems in aerospace product 
manufacturing.  Advanced Engineering Informatics 2024;59:102333. 
https://doi.org/https://doi.org/10.1016/j.aei.2023.102333.  
[65] Pak P, Farimani AB. AdditiveLLM: Large Language Models Predict Defects in Additive 
Manufacturing 2025.  
[66] Freire SK, Wang C, Foosherian M, W ellsandt S, Ruiz -Arenas S, Niforatos E. Knowledge 
Sharing in Manufacturing using Large Language Models: User Evaluation and Model 

43 Benchmarking 2024.  
[67] Chandrasekhar A, Chan J, Ogoke F, Ajenifujah O, Barati Farimani A. AMGPT: A large 
language model for c ontextual querying in additive manufacturing. Additive Manufacturing 
Letters 2024;11:100232. https://doi.org/https://doi.org/10.1016/j.addlet.2024.100232.  
[68] Holland M, Chaudhari K. Large language model based agent for process planning of fiber 
composite  structures. Manuf Lett 2024;40:100 –3. 
https://doi.org/https://doi.org/10.1016/j.mfglet.2024.03.010.  
[69] Fan H, Liu X, Fuh JYH, Lu WF, Li B. Embodied intelligence in manufacturing: leveraging large 
language models for autonomous industrial robotics. J Int ell Manuf 2025;36:1141 –57. 
https://doi.org/10.1007/s10845 -023-02294 -y. 
[70] Fan H, Huang J, Xu J, Zhou Y, Fuh JYH, Lu WF, et al. AutoMEX: Streamlining material 
extrusion with AI agents powered by large language models and knowledge graphs. Mater Des 
2025;251:113644. https://doi.org/https://doi.org/10.1016/j.matdes.2025.113644.  
[71] Xia Y, Shenoy M, Jazdi N, Weyrich M. Towards autonomous system: flexible modular 
production system enhanced with large language model agents 2023. 
https://doi.org/10.1109/E TFA54631.2023.10275362.  
[72] Vyas J, Mercangöz M. Autonomous Industrial Control using an Agentic Framework with Large 
Language Models 2024.  
[73] Lim J, Vogel -Heuser B, Kovalenko I. Large Language Model -Enabled Multi -Agent 
Manufacturing Systems 2024.  
[74] Jadhav Y, Pak P, Farimani AB. LLM -3D Print: Large Language Models To Monitor and Control 
3D Printing 2024.  
[75] Jeon J, Sim Y, Lee H, Han C, Yun D, Kim E, et al. ChatCNC: Conversational machine 
monitoring via large language model and real -time data retrieva l augmented generation. J Manuf 
Syst 2025;79:504 –14. https://doi.org/https://doi.org/10.1016/j.jmsy.2025.01.018.  
[76] UL Prospector. Troubleshooting FlipChart for Plastic Injection Molding - Prospector 
Knowledge Center 2016. https://www.ulprospector.com/kn owledge/3635/pe -troubleshooting -
flipchart -for-plastic -injection -molding/ (accessed February 7, 2025).  
[77] Rosato D V, Rosato MG. Injection molding handbook. Springer Science & Business Media; 
2012.  
[78] Dym JB. Injection molds and molding: a practical man ual. Springer Science & Business Media; 
1987.  
[79] Singh D. A guide to injection moulding technique. BookRix; 2018.  
[80] Goodship V. Troubleshooting injection moulding. vol. 15. iSmithers Rapra Publishing; 2004.  
[81] Carbonell J, Goldstein J. The use of MM R, diversity -based reranking for reordering documents 
and producing summaries. Proceedings of the 21st annual international ACM SIGIR conference 
on Research and development in information retrieval, 1998, p. 335 –6. 
[82] Ho J, Salimans T. Classifier -free di ffusion guidance. ArXiv Preprint ArXiv:220712598 2022.  
[83] Dorogush AV, Ershov V, Gulin A. CatBoost: gradient boosting with categorical features support. 

44 ArXiv Preprint ArXiv:181011363 2018.  
[84] Papineni K, Roukos  S, Ward T, Zhu W -J. Bleu: a method for automatic evaluation of machine 
translation. Proceedings of the 40th annual meeting of the Association for Computational 
Linguistics, 2002, p. 311 –8. 
[85] Lin C -Y. Rouge: A package for automatic evaluation of summari es. Text summarization 
branches out, 2004, p. 74 –81. 
[86] Zheng L, Chiang W -L, Sheng Y, Zhuang S, Wu Z, Zhuang Y, et al. Judging llm -as-a-judge with 
mt-bench and chatbot arena. Adv Neural Inf Process Syst 2023;36:46595 –623. 
  
  

45 Supplementary Material  
 
IM-Chat: A Multi -agent LLM -based Framework for 
Knowledge Transfer in Injection Molding Industry  
 
Junhyeong Lee1,†, Joon -Young Kim1, 2,†, Heekyu Kim1,†, Inhyo Lee1 and Seunghwa Ryu1,* 
 
Affiliations  
1Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology 
(KAIST), 291 Daehak -ro, Yuseong -gu, Daejeon 34141, Republic of Korea  
2Industrial Intelligence Research Group, AI/DX Center, Institute for Advanced Engineering (IAE), 
Yongin, Republic of Korea  
 
*Corresponding authors' e -mail: ryush@kaist.ac.kr   
  

46 Supplementary Note 1. IM -Chat Implementation and Prompt Engineering Details  
 This section provides the complete prompt templates used to configure the core modules 
of the IM -Chat framework. To ensure transparency and reproducibility, each prompt is presented 
in its entirety, with variables dynamically inserted at runtime denoted by  curly brackets (e.g., 
{input}). Accompanying explanations of these variables are included to enhance reader 
comprehension and enable accurate interpretation of the prompt structure.  
 In addition, for each module, we explicitly specify its role, task obje ctive, operational 
constraints, and expected output format. This structured specification enables precise control over 
modules behavior and ensures consistent performance across varying task contexts. Notably, the 
design of each module is prompt -driven, al lowing for flexible modification and rapid extension of 
system functionalities. This modularity underscores the adaptability and scalability of the IM -Chat 
framework in diverse manufacturing applications.  
 
  

47 1.1. Prompt for Task formatter  
 The Task formatt er module  is responsible for transforming the user’s current input query 
into a structured format that clearly reflects the user’s intent. To do so, it retrieves and incorporates 
relevant context from the accumulated conversation_history , enabling the syst em to disambiguate 
user instructions and infer implicit information. The output is a structured query that synthesizes 
both the new input and prior conversational knowledge, providing a precise and context -aware 
representation of the user’s task request. T his structured query serves as the foundation for 
subsequent tool selection and execution. The prompt used for the Task formatter  is shown below.  
 
## ROLE ##  
You are a context -aware AI assistant specializing in extracting user intent and relevant 
informati on from conversation history.  
--- 
## TASK ##  
1. **Analyze** the conversation history and determine the user’s current request.   
2. **Retrieve** all necessary details from the history that are relevant to answering the request.   
3. **Format** the extracte d information in a structured way to be used by a subsequent AI 
agent.   
--- 
## ENSURE THAT ##  
- The user’s intent is captured **clearly and concisely**.   
- No important details from the conversation history are **missed**.   
- The output is structured fo r easy parsing by another AI agent.  
--- 

48 ## OUTPUT FORMAT ##  
- User's Current Request:   
(Summarized user request)   
- Relevant Information from Conversation History:   
- (Relevant detail 1)   
- (Relevant detail 2)   
- (Relevant detail 3)  
... 
--- 
Then, Here  is the conversation history (Please use the language the user is speaking):  
{conversation_history } 
  

49 1.2. Prompt for Translator  
 To accommodate the multilingual environments commonly encountered in the injection 
molding industry, where workers may use different native languages, a Translator module was 
introduced as part of the input formatting stage. This module automatically trans lates all user inputs 
into English, ensuring compatibility with downstream reasoning and tool modules that operates in 
a single language. In parallel, it identifies and records the user’s original language, allowing the 
system to deliver responses back in the appropriate language when needed. This design enables 
seamless multilingual interaction while maintaining a consistent internal representation of user 
queries. The following prompt is used to confiture the Translator  module.  
 
You are a translation assistant.   
Your task is to:  
1. Detect the language of the input text.   
2. If the input is already in English, do not translate it. Just return it as is.  
3. If the input is not in English, translate it into English.  
4. Return the re sult in the following format:  
{format_instructions}  
--- 
Input text:   
{input}  
 
The format_instructions  schema employed in the Translator module  is defined by the following 
class. It ensures that the translated query is returned in English, accompanied by metadata 
indicating the original input language.  
 

50 class Translated(BaseModel):  
    translated_query: str = Field(description="The translated input query in English.")  
    language: str = Field(description="The original language of the input query.")  
 
The input  to the Translator module  consists of a structured query, generated by the Task 
Formatter , in the language originally used by the user.  
 
  

51 1.3. Prompt for Classifier  
The Classifier  module is designed to perform conditional routing based on the semantic 
category of the incoming user query. Specifically, it determines whether the query is injection 
molding -related or not. Queries classified as injection -related are processed through a  structured 
plan-and-execute pipeline, which carefully invokes appropriate tools to generate accurate 
responses. In contrast, queries deemed unrelated to injection molding are delegated to a lightweight 
ReAct -style agent that leverages web search without i nvoking specialized tools. Additionally, 
queries that can be resolved solely through retrieval from the existing conversation history are also 
categorized as non -injection -related. The output of the classifier is a binary label: either “injection” 
or “no_i njection”. The corresponding system prompt used to implement this classification logic is 
provided below.  
 
## ROLE ##   
You classify **injection molding queries** based on whether they require tool -based 
processing.  
---   
## CLASSIFICATION RULES ##   
- **`injection`** → If the query is:   
  1. **Directly related to injection molding** (e.g., defects, machine maintenance, material 
settings).   
  2. **Requires at least one available tool** to generate a solution.   
 
- **`no_injection`** → If the query:   
  - Is unrelated to injection molding.   

52   - Covers general knowledge, theory, or industry trends.   
  - Can be answered **fully using conversation history** (without tools).   
---   
## TOOLS ##   
If classified as **`injection`**, process using:   
- **table_retriever** → Retrieves **senior worker -documented process parameter 
adjustments** for each defect type from a predefined table.  
- **manual_retriever** → Retrieves  information about inejction molding machine: Machine 
Maintenance, Machine Inspection , Machine Operation, and Material settings etc.  
- **internet_search** → Finds **external information** only if relevant information is not 
available in the manual or table.  
- **llm_infer** → Relies solely on internal reasoning without using any external to ols..    
- **diffusion_model** → Exclusively used to generate injection molding process parameter 
sets using a diffusion model, based on environmental variables provided by the user. Diffusion 
model needs 4 parameters: Machine temperature, Machine humidity , Factory  temperature, and 
Factory humidity.  
**Do not** provide explanations. If uncertain, default to `no_injection`.  
 
In the Classifier module , the parsing schema is implemented using the BaseModel class and 
incldes the use of Literal types to constrain  specific field values, as detailed below.  
 
class Category(BaseModel):  
    """Classify if it is related to injection molding"""  

53     category: Literal["injection", "no_injection"]  
 
  

54 1.4. Prompt for ReAct Agent  
The ReAct  agent represents a representative class of reasoning agents that iteratively 
alternate between thought generation and tool invocation to solve complex tasks. In the IM -Chat 
framework, it is employed to efficiently handle queries unrelated to injection mol ding, allowing for 
lightweight and rapid response generation without engaging the full plan -and-execute pipeline. The 
system prompt used to configure the ReAct agent is provided below.  
 
## ROLE ##   
You provide **clear, accurate, and user -friendly answers* * on general topics.   
Adapt your tone to match the user's style when appropriate.   
 
## TOOL USAGE ##   
Use **"internet_search"** **ONLY IF**:   
- The information is **time -sensitive** (e.g., news, weather, recent tech updates).   
- The answer is **outsid e your knowledge scope**.   
- External confirmation is **necessary for accuracy**.   
If using **"internet_search"**, summarize findings and cite key sources.   
 
## OUTPUT EXPECTATIONS ##   
- Provide **complete** answers to minimize follow -up questions.   
- Use **step -by-step explanations** when helpful.   
- Break complex topics into sections for clarity.   
- Ensure **accuracy, clarity, and helpfulness** in all responses.   
---   

55 Answer the question as thoroughly as possible, using tools only when necessary.  
 
  

56 1.5. Prompt for Planner  
 The Planner  module is responsible for decomposing a given query related to injection 
molding into a sequence of actionable subtasks to facilitate solution generation. Upon receiving 
the input query ({input}), the Planner  consi ders the scope of executable actions available to the 
Executor  and outputs a structured plan following the schema specified in 
{format_instructions}.The full system prompt used for the Planner  is provided below.  
 
## ROLE ##   
You are the **Planner, respons ible for creating **minimal, step -by-step advisory plans** for 
**injection molding queries**.   
Each step must be **essential, self -contained, and lead efficiently to the final conclusion**.   
---   
## GUIDELINES ##   
- **Use the fewest steps possible to r each a conclusion.**   
- **Each step must provide direct, actionable guidance.**   
- **Avoid unnecessary details, extra interpretation, or redundant steps.**   
- **Ensure the final step gives a clear answer.**   
---   
## AVAILABLE TOOLS ##   
- **table_retriever** → Retrieves **senior worker -documented process parameter 
adjustments** for each defect type from a predefined table.  
- **manual_retriever** → Retrieves  information about inejction molding machine: Machine 
Maintenance, Machine Inspection , Machine Operation, and Material settings etc.  

57 - **internet_search** → Finds **external information** only if relevant information is not 
available in the manual or table.  
- **llm_infer** → Relies solely on internal reasoning without using any external to ols..    
- **diffusion_model** → Exclusively used to generate injection molding process parameter 
sets using a diffusion model, based on environmental variables provided by the user. Diffusion 
model needs 4 parameters: Machine temperature, Machine humidity , Factory  temperature, and 
Factory humidity.  
**Note:**  
- The information obtained from tools is **accurate and does not require review or further 
analysis steps**. Simply use the retrieved data as is.  
**Focus on the fastest path to a useful advice —no unn ecessary steps.**  
--- 
## INPUTS ##  
- Input query: {input}  
--- 
## OUTPUT FORMAT ##  
{format_instructions}  
 
The planning schema is defined by the Plan class, implemented using BaseModel. It includes a 
single field, steps , which is a list of string representing the ordered sequence of tool -invocation 
steps. Each element in the list corresponds to a tuple specifying the tool to be used and a brief 
description of the associated subtasks.  
 
class Plan(BaseModel):  

58     steps: List[str] = Field(  
        description="""  
        A list of ordered steps to follow. Each step is a tuple in the format:  
        [(Tool to be used, Task description),  
         (Tool to be used, Task description),  
         ...] 
          
        Steps must be ordered sequentially.  
        """ 
    ) 
 
 
  

59 1.6. Prompt for Executor  
The Executor  module is designed to operate on the first subtask provided by the Plan. Only 
this initial subtask is passed as input, and the Executor dynamically selects the appropriate tool to 
accomplish the task. To handle execution, we employed a ReAct -style agent capable of invoking 
up to five tools, including LLM -based inference. The corresponding prompt used to configure this 
module is presented below.  
 
## ROLE ##   
You are the **Executor Agent**, responsible for executing **injection molding tasks** as 
instructed by the Planner.   
Execute the task given to you with using the proper t ool (mentioned on the task).  
---   
## AVAILABLE TOOLS ##   
- **table_retriever** → Retrieves **senior worker -documented process parameter 
adjustments** for each defect type from a predefined table.  
- **manual_retriever** → Retrieves  information about ine jction molding machine: Machine 
Maintenance, Machine Inspection, Machine Operation, and Materials Settings etc.  
- **internet_search** → Finds **external information** only if relevant information is not 
available in the manual or table.  
- **llm_infer** → Relies solely on internal reasoning without using any external tools..    
- **diffusion_model** → Exclusively used to generate injection molding process parameter 
sets using a diffusion model, based on environmental variables provided by the user. Diffusion  

60 model needs 4 parameters: Machine temperature, Machine humidity, Factory  temperature, and 
Factory humidity.  
--- 
## IMPORTANT ##   
- **Do not anticipate future steps.**   
- **Do not skip or combine steps.**   
- **Use the tool ONLY ONCE and give the resopns e.** 
- **If the task is not clearly executable or falls outside your capabilities, respond exactly 
with:**   
  `"This is out of my scope. Please specify the task."`  
 
  

61 1.7. Prompt for Supervisor  
The Supervisor  module is responsible for determining whether the current solution 
provided by the Executor , along with the accumulated solution history, is sufficient to fully resolve 
the original input query. If the available information is deemed adequate, the Supervi sor returns the 
signal "respond" to indicate that a final answer can be generated. Conversely, if the solution is 
incomplete or insufficient, the Supervisor outputs "replan" to trigger another round of planning and 
execution. To support this decision -makin g process, the Supervisor prompt is supplied with the 
original query ({input}), the current plan ({plan}), and the execution history ({past_steps}), which 
includes all prior subtasks and their corresponding results. The complete prompt is shown below.  
 
You are a decision -making agent.  
Your task is to read the following input, and based on the conditions, retrun either "replan" or 
"respond" as your decision.  
--- 
## INPUTS ##  
- Input query: {input}  
- Original plan: {plan}  
- Completed Steps: {past_steps}  
--- 
## OPERATION MODE ##   
Determine your role based on the following rules:  
1. If the **Original Plan** is incomplete or unclear, or if **Completed Steps** do not yet 
fulfill the Objective**, then:  
→ Output: "replan"  

62 2. If the **Completed Steps are sufficient  to achieve the Objective**, then:  
→ Output: "respond"  
3. If **necessary information is missing and the Objective cannot be completed without it**, 
then:  
Especially for diffusion model, we strictly need four inputs: Machine temperature, Machine 
humidity, F actory temperature, and Factory humidity.  
→ Output: "respond"  
**Important Notes:**  
- Do NOT try to guess or assume missing information.  
- Base your decision solely on "INPUTS" provided.  
## OUTPUT FORMAT ##  
{format_instructions}  
 
The format_instructions sc hema is implemented using a BaseModel with a Literal type to ensure 
that the output of the Supervisor  is restricted to either "respond" or "replan" as valid decisions, as 
shown below.  
 
class Supervisor_decision(BaseModel):  
    decision: Literal["replan", "respond"]  
 
 
  

63 1.8. Prompt for Replanner  
The Replanner  module functions similarly to the original Planner , generating an actionable 
plan for the Executor  to follow. However, unlike the Planner , it revises the plan by incorporating 
not only the initial input query but also the existing plan and the execution history provided by the 
Executor . To enable this, the prompt includes {input}, {plan}, and {past_steps}, as in the 
Supervisor  module. The o utput is structured using the same Plan class defined in the Planner 
module. The prompt is provided below.  
 
## ROLE ##   
You are the **Replanner Agent**, responsible for updating a step -by-step plan or finalizing 
the planning process and responding to the user.   
You are part of a knowledge transfer system.   
Your role is to deliver accurate, structured knowledge related to the given objective — not to 
modify the objective or take control of the process yourself.  
 
You must decide whether to:  
- Revise and co ntinue the plan, if it is incomplete or suboptimal.  
- Finalize and respond, if all necessary steps have been completed.  
 
## GUIDELINES ##   
- **Use the fewest steps possible to reach a clear and useful conclusion.**   
- **Avoid unnecessary details, redundant logic, or excessive interpretation.**   
- **The final step must clearly address the objective.**  
 

64 ## AVAILABLE TOOLS ##   
- **table_retriever** → Retrieves **senior worker -documented process parameter 
adjustments** for each defect type from a predefined table.  
- **manual_retriever** → Retrieves  information about inejction molding machine: Machine 
Maintenance, Machine Inspection, Machine Operation, and Material settings etc.  
- **internet_search** → Finds **external information** only if relevant information is not 
available in the manual or table.  
- **llm_infer** → Relies solely on internal reasoning without using any external tools..    
- **diffusion_model** → Exclusively used to generate injection molding process p arameter 
sets using a diffusion model, based on environmental variables provided by the user. Diffusion 
model needs 4 parameters: Machine temperature, Machine humidity, Factory  temperature, and 
Factory humidity.  
**Note:**   
- Information retrieved via tool s is **accurate** and should be used **as is** — no need for 
validation or additional analysis.  
**Your goal is to provide the fastest possible path to useful, actionable knowledge.**  
--- 
## INPUTS ##  
- Input query: {input}  
- Original plan: {plan}  
- Complet ed Steps: {past_steps}  
--- 
## OUTPUT FORMAT ##  

65 {format_instructions}  
 
  

66 1.9. Prompt for Reporter (for non -injection tasks)  
 The Reporter  module is designed to reformat and summarize the solution, originally 
generated in English, into the user’s native language for final display. Given the English query 
{input_eng} and its corresponding response {response}, the module utilizes the detected user 
language {language}, as identified by the Translator  module, to generate a localized and user -
friendly summary. This ensures that the final output is both linguistically accessible and 
contextually coherent for multilingual users. The full prompt is p rovided below.  
 
You are the final output formatting agent in a multi -agent system. Your role is to present the 
final response in a clear and natural format suitable for a {language} -speaking user.  
 
The original user query is:  
{input_eng}  
 
The final respons e for this query is:  
{response}  
 
Please follow these guidelines:  
 
1. Regardless of the original input language, make sure the final response is clearly 
understandable in {language}, preserving the original meaning, tone, and nuance.  
2. Do not skip or delete content, even if it seems already understandable. Your job is to make 
sure the user receives a complete, well -formatted, and respectful response in their language.  

67 3. You may retain certain English words if doing so improves clarity or intent, consd iering user 
query and the response.  
4. Do not summarize, omit, or rephrase content unless it's truly necessary for natural readability 
in {language}.  
 
Present only the final, user -facing response, without explanation or markup.  
 
Remember: your job is not j ust translation — it’s to deliver the final output as if a human 
assistant were speaking directly to a {language} -speaking user.  
 
  

68 1.10. Prompt for Reporter (for injection -related tasks)  
For injection -related tasks, the Reporter  module is configured to generate a concise and 
user-friendly summary based not only on the final solution, but also on the full execution history 
captured in {past_steps}. This input includes the sequence of subtasks executed throughout the 
plan-and-execu te loop, along with the associated tool outputs. The prompt is designed to distill this 
information into a coherent response suitable for end -user communication. The complete prompt 
is presented below.  
 
The original user query is:  
{input_eng}  
 
To address this query, we referred to the following reasoning steps:  
{past_steps}  
 
Please do the following:  
 
1. Based on the reasoning steps, extract and summarize only the most relevant information 
necessary to answer the query.  
2. Translate the explanation into {la nguage}, preserving the meaning and tone. Presnet only the 
translated explanation.  
- If the language is English, skip the translation step silently.  
- Use the original English term in parentheses when translating technical words.  
 

69 Keep the final output pol ite, context -aware, and easy to understand. Avoid unnecessary 
elaboration.  
  

70 Supplementary Note 2. Prompt Details for Retrieval Methods  
For tasks related to injection molding, the Executor leverages a set of retriever tools to 
augment responses with factu al information drawn from domain -specific knowledge sources. 
These tools retrieve semantically relevant chunks by performing embedding -based similarity 
search against the input query. Rather than simply returning the matched chunks, the retrieved 
informati on is post -processed —summarized and reformatted —into a structured form that 
facilitates integration into the reasoning workflow. This approach ensures that the Executor 
receives well -organized and contextually relevant knowledge to support informed decisio n-making.  
This section presents the full prompts used for the retrievers responsible for accessing and 
synthesizing information from both the troubleshooting table and the manufacturing manual.  
 
  

71 2.1. Prompt for troubleshooting table retrieval tool  
## ROLE ##  
Given a specific defect type, extract and organize the corresponding process parameters and 
their adjustment directions & orders **strictly based on the information provided in the 
DOCUMENT section**.  
- Use ONLY the content from DOCUMENT.   
- Do NO T make assumptions or add any suggestions.  
--- 
## DOCUMENT ##  
You will receive two documents:  
1. A table indicating whether each process parameter should be increased ("+") or decreased (" -
") to resolve the given defect.  
2. A table specifying the adjustmen t priority (lower numbers indicate higher priority).   
Documents:  
{context}  
--- 
## OUTPUT FORMAT ##  
### **Parameter Adjustments (sorted by adjustment order):**  
**(Priority: 1)** → (Parameter, Increase / Decrease), (Parameter, Increase / Decrease), ...  
**(Priority: 2)** → (Parameter, Increase / Decrease), (Parameter, Increase / Decrease), ...   
... (continue as needed) ...  
--- 
### **If the defect type is ambiguous or unspecified:**   
Respond with:   

72 "I cannot answer that query because the defect type is not specific."  
 
  

73 2.2. Prompt for machine manual retrieval tool  
## ROLE ##   
You are an information extractor about the query about **injection molding machine operation 
& maintenance**  
 
Use ONLY the provided manual. **DO NOT assume or invent information.**   
 
## GUIDELINES ##   
- Answer strictly based on the manual.   
- Indicate whether your answer is from by page number.  
 
## DOCUMENT ##   
{context}   
 
## OUTPUT FORMAT ##   
**Answer:** [Provide response based on the manual]   
**Reference:* * See [page number] for detail.   
 
### **If you cannot find related information from the document:**   
Respond with:   
"The manual does not contain the information you mentioned about."  
 
  

74 Supplementary Note 3. Technical Description of the Classifier -Free Guidance Diffusion 
Model (CFGDM)  
The diffusion model aims to learn a reverse process that reconstructs the original data 
distribution from a noisy prior, enabling new sample generation. Initially, noise is incrementally 
introduced to the original data, co nverting it into a noisy prior (forward process). The model is then 
trained to progressively denoise this prior and recover the original distribution (reverse process). 
Both processes proceed through incremental steps defined by a Markovian chain, where ea ch state 
depends solely on its preceding state. In this scenario, denoting each time step as 𝑡 (where 𝑡=
0,…,𝑇) , both forward and reverse processes, defined by conditional distributions based on a 
Markovian chain, follow Gaussian distributions.   
Formall y, using Bayes' rule, the reverse process can be expressed as:  
𝑞(𝐗𝑡−1|𝐗𝑡)=𝑞(𝐗𝑡|𝐗𝑡−1)𝑞(𝐗𝑡−1)
𝑞(𝐗𝑡)(1) 
Here, 𝑞(𝐗𝑡) indicates the distribution of the sample 𝐗 corresponding  to the specific time step 𝑡. In 
Eq. 1 , direct estimation of 𝑞(𝐗𝑡) and 𝑞(𝐗𝑡−1) is infeasible, preventing computation of 
𝑞(𝐗𝑡−1|𝐗𝑡). To resolve this, an approximation function 𝑝𝜃(𝐗𝑡−1|𝐗𝑡) is introduced and trained via 
machine learning methods. Consequently, the diffusion model is optimized to fulfill the subsequent 
condition.  
𝑝𝜃(𝐗𝑡−1|𝐗𝑡)≈𝑞(𝐗𝑡−1|𝐗𝑡) (2) 
 
3.1. Forward Process  
 In the forward process, Gaussian noise is incrementally added based on a predefined variance 
schedule 𝛽𝑡, typically using linear or cosine schedules. The conditional probability distribution is:  
𝑞(𝐗𝑡|𝐗𝑡−1)≔𝑁(𝐗𝑡 ; 𝛍𝐗𝑡−1,𝚺𝐗𝑡−1)≔𝑁(𝐗𝑡 ;√1−𝛽𝑡𝐗𝑡−1, 𝛽𝑡𝐈) (3) 

75 𝛽𝑡 governs the level of noise added and is commonly adjusted according to the time step 𝑡 using 
various scheduling techniques. For gradient computation, noise addition is reformulated using the 
reparameterization trick, similar to the methods used in variational autoencoder (VAE) [1], 
resulting in a closed -form expression:  
𝑞(𝐗𝑡|𝐗𝑡−1)=√1−𝛽𝑡 𝐗𝑡−1+√βt𝛜𝑡−1,             where  𝛜∼𝑁(𝟎, 𝐈) (4) 
Using the Markov property, the forward process up to any chosen step 𝑇 can be represented as the 
sequential multiplication of conditional probabilities, formulated as:  
𝑞(𝐗1:𝑇|𝐗0)≔∏ 𝑞(𝐗𝑡|𝐗𝑡−1)𝑇
𝑡=1(5) 
Moreover, by assuming that the variables 𝛜𝑡−1, 𝛜𝑡−2,…,𝛜0 follow an independent Gaussian 
distribution 𝑁(𝟎, 𝐈), and utilizing the property that the sum of two Gaussian distributions 
𝑁(𝟎, 𝜎12𝐈) and 𝑁(𝟎, 𝜎22𝐈) yields another Gaussian distribution 𝑁(𝟎, (𝜎12+𝜎22)𝐈), the sampling 
at any given  timestep  𝑡 can be expressed analytically:  
𝐗𝑡=√𝛼𝑡𝐗0+√1−𝛼𝑡𝛜 (6) 
Here, 𝛼𝑡 is defined as 1−𝛽𝑡, and 𝛼𝑡 as ∏ 𝛼𝑖𝑡
𝑖=1, while 𝛜 denotes the standard deviation of the 
merged Gaussian distribution.  
 
3.2. Reverse Process  
 To train the diffusion model, the goal is to maximize the negative log -likelihood of the 
original data 𝐗𝟎. However, since 𝒑𝜽(𝐗𝟎) cannot be directly evaluated, the model instead leverages 
a variational lower bound (VLB), a strategy also used in VAE:  
−log𝑝𝜃(𝐗0)≤−log𝑝𝜃(𝐗0)+𝐷𝐾𝐿(𝑞(𝐗1:𝑇|𝑿0)|𝑝𝜃(𝐗1:𝑇|𝐗0)) (7) 
                                                        =𝔼[log𝑞(𝐗1:𝑇|𝐗0)
𝑝𝜃(𝐗0:𝑇)] 

76                                                         =𝐋VLB 
Sohl-Dickstein et al. [2] made the above objective more analytically tractable by decomposing it 
into three components:  
𝐋VLB =𝔼𝑞[𝐷𝐾𝐿(𝑞(𝐗𝑇|𝐗0)∥𝑝𝜃(𝐗𝑇))+∑𝐷𝐾𝐿(𝑞(𝐗𝑡−1|𝐗𝑡,𝐗0)∥𝑝𝜃(𝐗𝑡−1|𝐗𝑡))−log𝑝𝜃(𝐗0|𝐗1)𝑇
𝑡=2](8) 
Let, 𝐋𝑇≔𝐷𝐾𝐿(𝑞(𝐗𝑇|𝐗0)∥𝑝𝜃(𝐗𝑇)), 𝐋𝑡≔𝐷𝐾𝐿(𝑞(𝐗𝑡−1|𝐗𝑡,𝐗0)∥𝑝𝜃(𝐗𝑡−1|𝐗𝑡)) (1≤𝑡≤
𝑇−1), and 𝐋0≔−log𝑝𝜃(𝐗0|𝐗1). Then, the variational lower bound 𝐋VLB becomes:  
𝐋VLB =𝐋𝑇+𝐋𝑇−1+⋯+𝐋0 (9) 
Among these terms, 𝐋𝑇  is associated with the noise schedule 𝛽𝑡 in the forward process, which is 
predefined —typically via linear or cosine scheduling —and not subject to learning. Additionally, 
since 𝑝𝜃(𝐗𝑇) represents Gaussian noise and contains no learnable parameters, 𝐋𝑇 remains fixed 
during training and can be i gnored. Similarly, 𝐋0 has been shown to have negligible impact on 
model performance and is typically omitted from the training objective [3]. 
  The core objective thus reduces to optimizing 𝐋𝑡 for 1≤𝑡≤𝑇−1. When 𝛽𝑡 is small, the 
forward and reverse processes can be approximated by similar Gaussian conditionals [2], allowing:  
𝑝𝜃(𝐗𝑡−1|𝐗𝑡)=𝑁(𝐗𝑡−1; 𝛍𝜃(𝐗𝑡,𝑡), 𝚺𝜃(𝐗𝑡,𝑡)) for 1<𝑡≤𝑇 (10) 
Here, 𝚺𝜃(𝐗𝑡,𝑡)=𝜎𝑡2𝐈 is set as untrained time dependent constants. Though 𝑞(𝐗𝑡−1|𝐗𝑡) is 
intractable, its conditional form given 𝐗0 is analytically accessible:  
𝑞(𝐗𝑡−1|𝐗𝑡,𝐗0)=𝑁(𝐗𝑡−1;𝛍(𝐗𝑡, 𝐗0),𝛽𝑡𝐈) (11) 
Using Bayes' rule and properties of Gaussian distributions, the mean 𝛍̃𝑡 can be derived as:  
𝛍̃𝑡(𝐗𝑡, 𝐗0)=√𝛼𝑡(1−𝛼𝑡−1)
1−𝛼𝑡𝐗𝑡+√𝛼𝑡−1𝛽𝑡
1−𝛼𝑡𝐗0 (12) 
From Eq. 6, we know:  
𝐗0=1
√𝛼𝑡(𝐗𝑡−√1−𝛼𝑡𝛜𝑡) (13) 

77 Substituting this into the expression for 𝛍̃𝑡, we get: 
𝛍̃𝑡(𝐗𝑡, 𝐗0)= 1
√𝛼𝑡(𝐗𝑡−1−𝛼𝑡
√1−𝛼𝑡𝛜𝑡) (14) 
This leads to the following loss function:  
𝐋𝑡=𝔼𝑞[1
2𝜎𝑡2‖𝛍̃𝑡(𝐗𝑡,𝐗0)−𝛍𝜃(𝐗𝑡,𝑡)‖2]+𝐂 (15) 
Here, C is a constant term that does not depend on 𝜃. The function 𝛍𝜃(𝐗𝑡,𝑡) serves as a natural 
candidate for estimating the target mean 𝛍̃𝑡(𝐗𝑡, 𝐗0). Given that 𝛍𝜃 is trained to approximate 
𝛍̃𝑡(𝐗𝑡, 𝐗0)= 1
√𝛼𝑡(𝐗𝑡−1−𝛼𝑡
√1−𝛼𝑡𝛜𝑡), it can be reparameterized accordingly:  
𝛍𝜃(𝐗𝑡,𝑡)=1
√𝛼𝑡(𝐗𝑡−𝛽𝒕
√1−𝛼̅𝑡𝛜𝜃(𝐗𝑡,𝑡)) (16) 
𝛜𝜃 is a neural network that predicts the noise 𝛜 given 𝐗𝑡. By inserting Eq. 14  and Eq. 16  into Eq. 
15, the loss expression can be reformulated as follows:  
𝐋𝑡=𝔼𝐗0,𝛜[𝛽𝑡2
2𝜎𝑡2𝛼𝑡(1−𝛼̅𝑡)‖𝛜−𝛜𝜃(√𝛼̅𝑡𝐗0+√1−𝛼̅𝑡𝛜,𝑡)‖2] (17) 
Since constant factors do not affect  optimization, the practical loss function simplifies to:  
‖𝛜−𝛜𝜃(√𝛼̅𝑡𝐗0+√1−𝛼̅𝑡𝛜,𝑡)‖ (18) 
Once training is complete, new samples can be generated iteratively using:  
𝐗𝑡−1=1
√𝛼𝑡(𝐗𝑡−𝛽𝒕
√1−𝛼̅𝑡𝛜𝜃(𝐗𝑡,𝑡))+𝜎𝑡𝐳,      𝐳~𝑁(𝟎,𝐈) (19) 
 
3.3. Classifier Free Guidance Diffusion Model  
 The previously described diffusion model allows for the generation of process parameters 
that resemble those found in the training data. However, in an unconditional setting, the model 

78 simultaneously learns the distributions corresponding to both non defective and defective products, 
which can make it difficult to accurately capture the data distribution. Prior research [4,5] has 
shown that introducing class labels as conditioning info rmation can improve generative model 
training by enabling better separation of different data categories. Moreover, in real -world 
scenarios, external factors such as factory temperature and humidity , as well as the temperature and 
humidity of materials, ar e uncontrollable. Therefore, it is essential to generate parameters for non 
defective products under given environmental conditions.  
 To address this, the Classifier -Free Guidance Diffusion Model (CFGDM) [6] is employed. 
Unlike the classifier guidance ne twork introduced by Dhariwal et al. [7], CFGDM achieves 
conditional generation without requiring an external classifier, allowing for easier integration into 
existing diffusion frameworks. This approach jointly optimizes conditional and unconditional 
objectives by randomly dropping the condition y during training with a certain probability. The 
modified prediction is computed as:  
𝛜̌(𝐳𝑡,𝐲,𝑡)=𝑤𝛜(𝐳𝑡,𝐲,𝑡)+(1−𝑤)𝛜(𝐳𝑡,𝑡) (20) 
Here, 𝑤 controls the influence of the conditional prediction, while 𝛜(𝐳𝑡,𝐲,𝑡) and 𝛜(𝐳𝑡,𝑡) refer to 
the outputs of the conditional and unconditional models, respectively. A study by Ho et al. [6] 
explored optimal values for the condition drop rate and guidance strength, recommending a drop 
probability of 0.1 and setting 𝑤 betwee n 0.1 and 4. Following this guidance, we adopt a drop rate 
of 0.1 and use 𝑤=3 in our implementation.  
 The training procedure can be outlined in the following steps:  
1. Perturbed process parameters at time step 𝑡 are generated by injecting noise as defined in 
Eq. 6.  
2. These perturbed parameters are fed into a U -Net, which outputs both the conditional noise 
prediction 𝛜(𝐳𝑡,𝐲,𝑡) and the unconditional prediction 𝛜(𝐳𝑡,𝑡). These are then combined 
to compute 𝛜̌(𝐳𝑡,𝐲,𝑡) using Eq. 20.  

79 3. The model is then optimized based on the loss function specified in Eq. 18.  
  

80 Supplementary Note 4. Prompts for Diffusion and Input Formatter  
 To interface with the diffusion model, five numerical input features are required: four 
environmental variables, machine temperature, machine humidity, factory temperature, and factory 
humidity, and a product quality class indicator (i.e., good or defective). Since the initial user queries 
are expressed in natural language, an input formattin g module is necessary to extract and structure 
these variables appropriately. To achieve this, an LLM -based formatter is employed, guided by a 
prompt designed to infer the required parameters from the query. The system prompt is as follows:  
 
You are a help ful assistant. For each input query, return a dictionary with exactly 5 keys:  
'machine_temperature (Celcius degree)', 'machine_humidity (%)', 'factory_temperature 
(Celcius degree)', 'factory_humidity (%)', and 'class (0 or 1)'.  
Each key must have either a floating -point number as its value, or 'None' if there is not enough 
information to determine a value.  
You can keep the value as "None" if you do not have a clear and sufficient information to fill it 
in. 
For the class, unless there are explicit incident s, let it be marked as good.  
Please provide the output in this format for the input query.  
 
The output of this module is parsed using a structured JSON schema, which ensures compatibility 
with the downstream diffusion model. The schema is defined as fol lows:  
 
function_input_format = {  
    "name": "formatter",  
    "type": "function",  

81     "description": "Try to infer 5 properties: machine_temperature, machine_humidity, 
factory_temperature, factory_humidity, and class",  
    "parameters": {  
        "type": "object",  
        "properties": {  
            "machine_temperature": {  
                "description": "The temperature of the machine",  
                "type": ["number", "null"]  
            }, 
            "machine_humidity": {  
                "description": "The humidity of the machine",  
                "type": ["number", "null"]  
            }, 
            "factory_temperature": {  
                "description": "The temperature of the factory",  
                "type": ["number", "null"]  
            }, 
            "factory_humidity": {  
                "description": "The humidity of the factory",  
                "type": ["number", "null"]  
            }, 
            "class": {  
                "description": "Must be 0 or 1 or None; Good produc t: 0 / Defective product: 1 / 
Unknown: None",  

82                 "type": ["number", "null"],  
                "enum": [0, 1, null]  
            } 
        }, 
        "required": [  
            "machine_temperature",  
            "machine_humidity",  
            "factory_temperature",  
            "factory_humidity",  
            "class"  
        ] 
    } 
} 
 
  

83 Supplementary Note 5. Multilingual Input Support and Validation  
To evaluate the multilingual capabilities of the IM -Chat framework, we conducted a 
preliminary assessment across four languages commonly encountered in industrial settings, Korean, 
English, Thai, and Vietnamese. These languages were selected in consideration of the linguistic 
diversity observed in injection molding factories in South Korea, where ma ny foreign workers are 
employed. Representative queries corresponding to each of the key tool categories were selected, 
and the system's performance was validated to ensure robust handling of multilingual inputs across 
all supported modules.  
This evaluatio n focused on representative queries for each tool type, as the primary 
objective was to verify the correct operation of the multilingual pipeline rather than to conduct 
exhaustive benchmarking. Given that IM -Chat translates all inputs into English prior to  reasoning 
and tool invocation, the system's behavior for each language is expected to be largely consistent 
once the translation is verified to be accurate. Accordingly, validation was conducted for four core 
functionalities, troubleshooting table retriev al, manufacturing manual retrieval, diffusion model 
invocation, and internet search, across four languages (Korean, English, Thai, and Vietnamese) to 
ensure robust multilingual support. The results, as shown in the tables below, indicate that IM -Chat 
produ ced consistent outcomes across different languages, demonstrating the robustness of its 
multilingual processing pipeline.  
  

84 5.1. Task requiring knowledge from troubleshooting table  
English  How should Injection speed be adjusted to reduce 
burr defects?  Specifically, you should prioritize reducing 
Injection Speed 1, followed by Injection Speed 2, 
and then Injection Speed 3.  
Korean  버 결함을  줄이기  위해  사출속도는  어떻게  
조정해야  하나요 ? 버 결함을  줄이기  위해  주입  속도 (injection 
speed)를 조정하는  방법에  대한  조언은  다음과  
같습니다 : 
 
주입  속도  3: 감소  (우선순위 : 5) 
주입  속도  2: 감소  (우선순위 : 6) 
주입  속도  1: 감소  (우선순위 : 7) 
Thai ควรปรับความเร็วการฉีดอย่างไรเพื่อลดข้อบกพร่องของเ
สี้ยน (burr)?  เพื่อปรับลดข้อบกพร่องของเสี้ยน (burr defects) 
ควรปรับลดความเร็วในการฉีด  (injection speed) 
โดยเริ่มจากการลดความเร็วในการฉีดที่ระดับ 3 
(injection speed  3) ตามด้วยระดับ  2 (injection speed 
2) และระดับ  1 (injection speed 1) ค่ะ 
Vietnamese  Tốc độ ép phun nên đư ợc điều chỉnh như th ế nào 
để giảm lỗi bavia?  Để giảm thi ểu khuy ết tật flash, b ạn nên đi ều chỉnh 
tốc độ phun (injection speed) như sau:  
 
Tốc độ phun  3: Gi ảm (Ưu tiên: 5)  
Tốc độ phun 2: Gi ảm (Ưu tiên: 6)  
Tốc độ phun 1: Gi ảm (Ưu tiên: 7)  
 
  

85 5.2. Task requiring knowledge from manufacturing manual  
English  What is the recommended Mold temperature range 
when the material is Acrylonitrile Butadiene 
Styrene  (ABS)?  The recommended mold temperature range for 
Acrylonitrile Butadiene Styrene (ABS) is 
40~60℃. 
Korean  사출하려는  재료가  Acrylonitrile Butadiene 
Styrene (ABS) 일 때, 권장되는  몰드  
온도범위는  어떻게  되나요 ? ABS 재료의  권장  금형  온도  범위는  
40~60℃입니다 . 
Thai เมื่อใช้วัสดุฉีดขึ้นรูปเป็น Acrylonitrile Butadiene 
Styrene (ABS) 
ช่วงอุณหภูมิของแม่พิมพ์ที่แนะน าคือเท่าใด? ช่วงอุณหภูมิแม่พิมพ์ที่แนะน าส าหรับการฉีดขึ้นรูป 
(injection molding) ของ Acrylonitrile Butadiene 
Styrene (ABS) คือ 40~60℃. 
Vietnamese  Khi s ử dụng vật liệu ép phun là Acrylonit rile 
Butadiene Styrene (ABS), d ải nhiệt độ khuôn 
được khuy ến ngh ị là bao nhiêu?  Nhiệt độ khuôn khuy ến ngh ị cho v ật liệu 
Acrylonitrile Butadiene Styrene (ABS) là 
40~60℃. 
 
  

86 5.3. Task requiring knowledge from diffusion model  
English  When Machine temperature  is 20.5°C, Machine 
humidity is 42.0%, Factory temperature is 24.0°C, 
and Factory humidity is 36.0%, can you calculate 
the acceptable product conditions using the 
Diffusion model?  Based on the reasoning steps, the injection 
molding process parameter sets g enerated using 
the diffusion model with the given environmental 
conditions are as follows:  
 
Injection Speed 1: 20.0  
Injection Speed 2: 20.0  
Injection Speed 3: 30.0  
Injection Pressure 1: 120.0  
Injection Pressure 2: 140.0  
Injection Pressure 3: 130.0  
Injectio n Position 1: 44.0  
Injection Position 2: 32.0  
Injection Position 3: 28.0  
Hold Time: 0.0  
These parameters are calculated considering the 
machine temperature of 20.5°C, machine humidity 
of 42.0%, factory temperature of 24.0°C, and 
factory humidity of 36.0%.  
Korean  머신  온도가  20.5°C, 머신  습도가  42.0%, 공장  
온도가  24.0°C, 공장  습도가  36.0%일 때, 
디퓨전  모델로  양품조건을  계산해줄  수 
있나요 ? 주어진  환경  조건을  사용하여  확산  
모델 (diffusion model) 로 생성된  사출  
성형 (injection molding) 공정  매개변수  세트는  
다음과  같습니다 : 
 
사출  속도  1: 38.00  
사출  속도  2: 20.00  
사출  속도  3: 30.00  

87 사출  압력  1: 120.00  
사출  압력  2: 120.00  
사출  압력  3: 130.00  
사출  위치  1: 44.00  
사출  위치  2: 32.01  
사출  위치  3: 28.00  
보유  시간 : 0.90초 
이 정보가  사용자의  요청에  대한  답변입니다 . 
Thai เมื่ออุณหภูมิเครื่องจักรอยู่ที่ 20.5°C 
ความชื้นของเครื่องจักรอยู่ที่ 42.0% 
อุณหภูมิในโรงงานอยู่ที่ 24.0°C 
และความชื้นในโรงงานอยู่ที่ 36.0% 
โปรดค านวณเงื่อนไขการผลิตที่ยอมรับได้โดยใช้แบบจ า
ลองการแพร่  (Diffusion model)  
 ชุดพารามิเตอร์ส าหรับกระบวนการฉีดขึ้นรูปที่สร้างขึ้นโ
ดยใช้แบบจ าลองการแพร่  (diffusion model) 
กับตัวแปรสิ่งแวดล้อมที่ให้มา มีดังนี้: 
 
ความเร็วการฉีด  (Injection Speed) 1: 20.0  
ความเร็วการฉีด  (Injection Speed) 2: 20.0  
ความเร็วการฉีด  (Injection Speed) 3: 30.0  
แรงดันการฉีด  (Injection Pressure) 1: 120.0  
แรงดันการฉีด  (Injection Pressure) 2: 120.0  
แรงดันการฉีด  (Injection Pressure) 3: 150.0  
ต าแหน่งการฉีด  (Injection Position) 1: 44.0  
ต าแหน่งกา รฉีด (Injection Position) 2: 32.0  
ต าแหน่งการฉีด  (Injection Position) 3: 28.0  
เวลาการถือ  (Hold Time): 0.0  
Vietnamese  Nếu nhi ệt độ máy là 20.5°C, đ ộ ẩm máy là 42.0%, 
nhiệt độ nhà máy là 24.0°C và đ ộ ẩm nhà máy là 
36.0%, thì  mô hì nh khu ếch tán có th ể tí nh toán 
điều kiện sản xuất tối ưu không?  Mô hì nh khu ếch tán (diffusion model) đã đư ợc sử 
dụng để tạo ra các b ộ thông s ố quy trì nh ép phun 
(injection molding process parameter sets) d ựa 
trên các bi ến môi trư ờng đã cung c ấp: Nhi ệt độ 

88 máy 20.5°C, Đ ộ ẩm máy 42.0% , Nhi ệt độ nhà 
máy 24.0°C, và Đ ộ ẩm nh à máy 36.0%. Các b ộ 
thông s ố quy trình ép phun đư ợc tạo ra như sau:  
 
Tốc độ phun 1 (Injection Speed 1): 20.0  
Tốc độ phun 2 (Injection Speed 2): 20.0  
Tốc độ phun 3 (Injection Speed 3): 30.0  
Áp suất phun  1 (Injection Pressure 1): 120.0  
Áp suất phun 2 (Injection Pressure 2): 120.0  
Áp suất phun 3 (Injection Pressure 3): 150.0  
Vị trí  phun 1 (Injection Position 1): 44.0  
Vị trí  phun 2 (Injection Position 2): 32.0  
Vị trí  phun 3 (Injection Position 3): 28.0  
Thời gian gi ữ (Hold Time): 0.0  
 
  

89 5.4. Task requiring knowledge from internet search  
English  Which are the leading injection molding machine 
manufacturers in Japan?  Japan is home to several leading manufacturers of 
injection molding machines, known for their high-
quality engineering and advanced technology. 
Here are some of the top companies in this field:  
 
Sumitomo (SHI) Demag: A prominent player in 
the injection molding industry, Sumitomo offers a 
wide range of machines known for their precision 
and efficien cy. 
 
Nissei Plastic Industrial Co., Ltd.: Nissei is 
renowned for its innovative solutions and a broad 
lineup of injection molding machines that cater to 
various industries.  
 
Toshiba Machine Co., Ltd.: Now known as 
Shibaura Machine, this company provides 
advanced injection molding machines with a focus 
on energy efficiency and high performance.  
 
Fanuc Corporation: Known for its automation and 
robotics, Fanuc also produces high -quality 
injection molding machines that integrate 
advanced control systems.  
 
JSW ( The Japan Steel Works, Ltd.): JSW is a 
major manufacturer offering a range of injection 

90 molding machines known for their durability and 
precision.  
 
Toyo Machinery & Metal Co., Ltd.: Toyo is 
recognized for its all -electric injection molding 
machines, which are designed for energy 
efficiency and high precision.  
 
These companies are leaders in the industry due to 
their commitment to innovation, quality, and 
customer satisfaction.  
Korean  일본의  주요  사출성형기  제조사는  
어디인가요 ? 일본은  첨단  기술과  고품질  생산으로  유명한  
여러  주요  사출  성형기  제조업체의  
본고장입니다 . 다음은  주요  
제조업체들입니다 : 
 
Sumitomo (SHI) Demag: 이 분야의  선두  
기업으로 , Sumitomo 는 다양한  사출  성형기를  
제공하며  혁신적인  기술과  에너지  효율적인  
솔루션으로  잘 알려져  있습니다 . 
 
Nissei Plastic Industrial Co., Ltd.: Nissei 는 
고성능  사출  성형기로  유명하며  국내외  
시장에서  강력한  입지를  가지고  있습니다 . 
 
Toshiba Machine Co., Ltd.: 현재는  Shibaura  
Machine으로  알려진  이 회사는  다양한  사출  

91 성형기를  제공하며  정밀성과  신뢰성으로  
인정받고  있습니다 . 
 
Fanuc Corporation: 자동화  및 로봇  공학으로  
유명한  Fanuc는 고효율적이고  첨단  제어  
시스템을  통합한  사출  성형기도  생산합니다 . 
 
JSW (The Japan Steel Works, Ltd.): JSW 는 
내구성과  첨단  기술로  유명한  사출  성형기를  
제공하는  주요  업체입니다 . 
 
Toyo Machinery & Metal Co., Ltd.:  Toyo는 
정밀성과  에너지  효율성으로  인기가  있는  
전기식  사출  성형기로  잘 알려져  있습니다 . 
 
이들  회사는  혁신과  품질에  중점을  두고  
다양한  산업적  요구를  충족시키는  사출  
성형기를  제공함으로써  사출  성형  산업에  
크게  기여하고  있습니다 . 
Thai ผู้ผลิตเครื่องฉีดพลาสติกหลักของญี่ปุ่นมีใครบ้าง? ญี่ปุ่นเป็นที่ตั้งของผู้ผลิตเครื่องฉีดพลาสติกชั้นน าหลาย
ราย นี่คือรายชื่อบริษัทชั้นน าในสาขานี้: 
 
Sumitomo (SHI) Demag: 
เป็นที่รู้จักในด้านเทคโนโลยีขั้นสูงและเครื่องจักรคุณภา
พสูง Sumitomo มีเครื่องฉีดพลาสติกหลากหลายรุ่น 
 

92 Nissei Plastic Industrial Co., Ltd.: Nissei 
เป็นบริษัทที่มีชื่อเสียงซึ่งให้บริการเครื่องฉีดพลาสติกหล
ากหลายประเภทที่มีความแม่นย าและประสิทธิภาพสูง 
 
Toshiba Machine Co., Ltd.: ปัจจุบันรู้จักกันในชื่อ 
Shibaura Machine 
พวกเขามีเครื่องฉีดพลาสติกหลากหลายรุ่นที่เหมาะส าห
รับการใช้งานต่างๆ  
 
Fanuc Corporation: Fanuc 
มีชื่อเสียงในด้านเทคโนโลยีอัตโนมัติและผลิตเครื่องฉีด
พลาสติกไฟฟ้าที่มีประสิทธิภาพและความน่าเชื่อถือสูง 
 
JSW (The Japan Steel Works, Ltd.): JSW 
เป็นผู้เล่นหลักในอุตสาหกรรมนี้ 
โดยมีเครื่องฉีดพลาสติกที่มีความทนทานและประสิทธิภา
พสูง 
 
Toyo Machinery & Metal Co., Ltd.: Toyo 
เชี่ยวชาญในเครื่องฉีดพลาสติกไฟฟ้าทั้งหมด  
ซึ่งมีความประหยัดพลังงานและความแม่นย าสูง 
 
Ube Machinery Corporation, Ltd.: Ube 
ให้บริการเครื่องฉีดพลาสติกขนาดใหญ่และเป็นที่รู้จักใน
ด้านโซลูชันนวัตกรรมในอุตสาหกรรม  
 

93 บริษัทเหล่านี้ได้รับการยอมรับในด้านความก้าวหน้าทางเ
ทคโนโลยีและการมีส่วนร่วมในอุตสาหกรรมการฉีดพลา
สติกทั่วโลก 
Vietnamese  Những nhà s ản xuất máy ép phun chí nh c ủa Nhật 
Bản là ai?  Nhật Bản là quê hương c ủa nhi ều nhà s ản xuất 
máy ép phun n ổi tiếng, đư ợc biết đến với kỹ thuật 
chất lượng cao và s ự đổi mới. Dư ới đây là m ột số 
nhà s ản xuất chí nh:  
 
Sumitomo (SHI) Demag: Là m ột công ty hàng đ ầu 
trong lĩnh v ực này, Sumitomo cung c ấp một loạt 
các máy ép phun, bao g ồm các m ẫu máy đi ện 
hoàn toàn, hybrid và th ủy lực. Họ nổi tiếng với độ 
chí nh xác và hi ệu quả năng lư ợng. 
 
Nissei Plastic Industrial Co., Ltd.: Nissei n ổi tiếng 
với công ngh ệ tiên ti ến và dòng s ản phẩm đa d ạng, 
bao g ồm các máy ép phun đi ện hoàn toàn, hybrid 
và th ủy lực. Họ tập trung vào công ngh ệ bền vững 
và tiết kiệm năng lư ợng. 
 
Toshiba Machine Co., Ltd.: Hi ện nay đư ợc biết 
đến với tên Shibaura Machine, công ty này cung 
cấp nhi ều loại máy ép phun đư ợc sử dụng trong 
các ngành công nghi ệp khác nhau. H ọ nhấn mạnh 
vào đ ộ chính xác và đ ộ tin cậy của máy móc.  
 
Fanuc  Corporation: Đư ợc biết đến với các gi ải 
pháp robot và t ự động hóa, Fanuc cũng s ản xuất 
các máy ép phun đi ện hoàn toàn. Máy c ủa họ được 

94 công nh ận về tốc độ cao, đ ộ chí nh xác và hi ệu quả 
năng lư ợng. 
 
JSW (The Japan Steel Works, Ltd.): JSW cung 
cấp một loạt các máy ép phun, bao g ồm các m ẫu 
máy quy mô l ớn cho các ứng dụng công nghi ệp. 
Họ nổi tiếng với thiết kế chắc chắn và công ngh ệ 
tiên ti ến. 
 
Ube Machinery Corporation, Ltd.: Ube chuyên v ề 
các máy ép phun quy mô l ớn và n ổi tiếng với các 
giải pháp sáng t ạo trong các ngành công nghi ệp ô 
tô và công nghi ệp. 
 
Những công ty này là nh ững ngư ời dẫn đầu trong 
ngành công nghi ệp ép phun, cung c ấp các máy 
móc đư ợc sử dụng trên toàn c ầu trong nhi ều lĩnh 
vực khác nhau, bao g ồm ô tô, đi ện tử và hàng tiêu 
dùng.  
 
  

95 Supplementary Note 6. Prompts for LLM -based Evaluation  
6.1. Prompt for LLM evaluation  
This prompt was used to configure the language model for conducting automated 
evaluations of IM -Chat's responses. Specifically, it enables self -assessment of the generated 
question –answer (Q&A) pairs by prompting the model to act as an expert evaluator. Th e model is 
instructed to assess each response along two key dimensions: (1) Relevance , which evaluates 
whether the answer addresses the user's intent, and (2) Accuracy , which checks the factual 
correctness of the information provided. For each criterion, t he model is required to generate a brief 
textual justification, followed by a numerical score (0 –10), where 10 denotes a highly relevant and 
accurate answer, and 0 indicates a completely inappropriate or incorrect response.  
 
You are an expert evaluator re viewing the quality of an AI -generated answer.  
 
Here is the user's original question:  
{question}  
 
Here is the answer provided by the AI:  
{answer}  
 
Evaluate the answer based on the following two criteria:  
 
1. Relevance: Does the answer appropriately address  the intent of the question?  
2. Accuracy: Is the information factually correct based on general knowledge?  
 

96 Provide a short evaluation for each criterion (1 -2 sentences), followed by an overall score (0 to 
10) where 10 means excellent and 0 means completel y irrelevant or incorrect.  
 
Format:  
Relevance: ...  
Accuracy: ...  
Rating:0~10  
  

97 References  
[1] Kingma DP. Auto -encoding variational bayes. ArXiv Preprint ArXiv:13126114 2013.  
[2] Sohl-Dickstein J, Weiss E, Maheswaranathan  N, Ganguli S. Deep unsupervised learning using 
nonequilibrium thermodynamics. International conference on machine learning, PMLR; 2015, p. 2256 –
65. 
[3] Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models. Adv Neural Inf Process Syst 
2020;33:6 840–51. 
[4] Casanova A, Careil M, Verbeek J, Drozdzal M, Romero Soriano A. Instance -conditioned gan. Adv Neural 
Inf Process Syst 2021;34:27517 –29. 
[5] Bao F, Li C, Sun J, Zhu J. Why Are Conditional Generative Models Better Than Unconditional Ones? 
NeurIPS 2022 Workshop on Score -Based Methods, n.d.  
[6] Ho J, Salimans T. Classifier -Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative 
Models and Downstream Applications, 2021.  
[7] Dhariwal P, Nichol A. Diffusion models beat gans  on image synthesis. Adv Neural Inf Process Syst 
2021;34:8780 –94. 
  
 