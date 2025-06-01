# Complex System Diagnostics Using a Knowledge Graph-Informed and Large Language Model-Enhanced Framework

**Authors**: Saman Marandi, Yu-Shu Hu, Mohammad Modarres

**Published**: 2025-05-27 14:54:49

**PDF URL**: [http://arxiv.org/pdf/2505.21291v1](http://arxiv.org/pdf/2505.21291v1)

## Abstract
In this paper, we present a novel diagnostic framework that integrates
Knowledge Graphs (KGs) and Large Language Models (LLMs) to support system
diagnostics in high-reliability systems such as nuclear power plants.
Traditional diagnostic modeling struggles when systems become too complex,
making functional modeling a more attractive approach. Our approach introduces
a diagnostic framework grounded in the functional modeling principles of the
Dynamic Master Logic (DML) model. It incorporates two coordinated LLM
components, including an LLM-based workflow for automated construction of DML
logic from system documentation and an LLM agent that facilitates interactive
diagnostics. The generated logic is encoded into a structured KG, referred to
as KG-DML, which supports hierarchical fault reasoning. Expert knowledge or
operational data can also be incorporated to refine the model's precision and
diagnostic depth. In the interaction phase, users submit natural language
queries, which are interpreted by the LLM agent. The agent selects appropriate
tools for structured reasoning, including upward and downward propagation
across the KG-DML. Rather than embedding KG content into every prompt, the LLM
agent distinguishes between diagnostic and interpretive tasks. For diagnostics,
the agent selects and executes external tools that perform structured KG
reasoning. For general queries, a Graph-based Retrieval-Augmented Generation
(Graph-RAG) approach is used, retrieving relevant KG segments and embedding
them into the prompt to generate natural explanations. A case study on an
auxiliary feedwater system demonstrated the framework's effectiveness, with
over 90% accuracy in key elements and consistent tool and argument extraction,
supporting its use in safety-critical diagnostics.

## Full Text


<!-- PDF content starts -->

1 
 Complex System Diagnostics Using a Knowledge Graph -Informed and Large Language 
Model -Enhanced Framework  
Saman Marandi1, Yu-Shu Hu2, Mohammad Modarres1 
1Center for Risk and Reliability U niversity of Maryla nd, MD,  USA; 2DML  Inc., Hsinchu, Taiwan  
Correspond ing Author : smarandi@umd.edu   
 
Abstract  
In this paper, we present a novel diagnostic framework that integrates Knowledge Graphs (KGs) and 
Large Language Models (LLMs) to support system diagnostics  in high -reliability systems such as 
nuclear power plants. Traditional diagnostic modeling struggles when systems become too complex, 
making functional modeling a more attractive approach . Our approach introduces a diagnostic 
framework grounded in the functional modeling principles of the Dynamic Master Logic (DML) 
model . It incorporates two coordinated LLM components, including an LLM -based workflow for 
automated construction of DML logic from system documentation and an LLM agent that facilitates 
interactive diagnostics.  The generated logic is encoded into a structured KG, referred to as KG -DML, 
which supports hierarchical fault reasoning. Expert knowledge or operational data can also be 
incorporated to refine the model’s precision and diagnostic depth. In the interaction  phase, users 
submit natural language queries, which are interpreted by the LLM agen t. The agent selects 
appropriate tools for structured reasoning, including upward and downward propagation across the 
KG-DML. Rather than embedding KG content into every prompt, the LLM agent  distinguishes 
between diagnostic and interpretive tasks. For diagnostics, the agent  selects and executes external 
tools that perform structured KG reasoning. For general queries, a Graph -based Retrieval -
Augmented Generation (Graph -RAG) approach is used, retrieving relevant KG segments and 
embedding them into the prompt  to generate natural explanations. A case study on an auxiliary 
feedwater system demonstrated the framework’s effectiveness, with over 90% accuracy in key 
elements and consistent tool and argument extraction, supporting its use in safety -critical diagnostics.  
Keywords:  Large Language Models, Knowledge Graphs, Diagnostics, Dynamic Master Logic 
(DML)  
1. Introduction  
The analysis of complex engineered systems, particularly those in high -reliability and safety -critical 
industries such as nuclear power plants, requires systematic approaches to assess system integrity, 
reliability, and performance. Traditional diagnostic tools have predominantly relied on event -based 
modeling, where the outcomes of specific failure pathways , faults  or abnormal initiating events are 
analyzed. Although effective in some domains, event -based approaches become extremely complex, 
incomplete and  inadequate  when applied to complex, interconnected systems with numerous 
interdependencies. As the scale and complexity of systems increase, functional  modeling approaches are 
considered more suitable, where the emphasis is placed on understanding the roles, dependencies, and 
contributions of system components toward achieving primary system objectives.  Functional modeling 
involves the development of system models based on the actions and relationships of their constituent parts  
[1], [2] . These models are inherently hierarchical, enabling  a systematic  decomposition of the system into 
goals, functions, and subfunctions. One important framework within this category is the Dynamic Master 
Logic (DML) model, introduced by Hu and Modarres  [3], which represents system behavior through a 

2 
 structured hierarchy linking functional objectives to underlying structures. By organizing systems into 
functional and structural layers, DML enables critical causal pathways, system interdependencies, and fault 
propagation mechanisms to be systematically analyzed. This hierarchical framework supports the tracing 
of failures from system -level objectives down to elemental components, offering a powerful tool for 
diagnostic analysis.   Although DML models provide a robust framework for system diagnostics, thei r 
construction, maintenance, and interpretation require significant manual effort and extensive domain 
expertise. As system complexity grows, the burden associated with constructing and interacting with large 
DML models increases accordingly. In response to these challenges, powerful Artificial Intelligence (AI)  
tools such as Large Language Models (LLMs)  [4] have been recognized as offering opportunities to 
enhance the development, interaction, and usability of functional models  for diagnosis of faults and failures 
in complex engineering systems . LLMs have demonstrated strong capabilities in natural language 
understanding, summarization, and reasoning across a wide range of domains. However, certain limitations 
such as hallucination and restricted domain -specific reasoning have been observed. To address these 
challenges and improve diagnostic reliability, stru ctured knowledge representations such as Knowledge 
Graphs (KGs)  [5] can be used alongside LLMs to enable more consistent and interpretable reasoning by 
guiding diagnostic logic through external tools rather than unconstrained language generation . 
The integration of LLMs with KGs provides a mechanism to enhance the accessibility, organization, and 
interpretability of complex functional models. By grounding LLM interactions in verified domain 
knowledge, the risks associated with hallucination can be reduced and the transparency of diagnostic 
outputs can be improved. Given the increasing complexity of engineered systems and the need for scalable, 
interpretable diagnostic tools, there is a strong motivation to explore AI -driven frameworks that combine 
domain knowledge with language -based reasoning. In this paper , an approach is proposed that leverages 
LLMs and KGs to support and enhance interaction with DML models, to reduce  manual effort, improve  
transparency, and facilitate more efficient fault analysis in safety -critical applications.   
The remainder of this paper is organized as follows. Section 2 reviews related work on DML modeling, 
large language models, and the integration of KGs in diagnostic systems. Section 3 provides an overview 
of the research approach. Section 4 presents the proposed diagnostic framework, detailing both the model 
construction and interaction phases. Section 5 describes a case study involving an auxiliary fee dwater 
system in a nuclear power plant, illustrating the application of the framework. Section 6 reports t he 
evaluation results from both the model construction and interaction components, based on repeated runs 
using the case study. Finally, Section 7 concludes the paper with key findings and discusses directions for 
future work.  
2. Background  
2.1. From Traditional Diagnostics to Functional Modeling  
Traditional diagnostic approaches often rely on modeling discrete events, failure modes, or symptom 
triggers. These include methods like Fault Tree Analysis (FTA)  [6], [7] , Event Trees  (ET)  [8], [9] , and rule -
based  [10], [11]  expert systems that map observed events to likely faults. Such models  focus  on specific 
failure events and their consequences . While these event -driven models can effectively represent known 
failure sequences, they typically require an exhaustive list of fault -event combinations,  invariably  making 
them incomplete. If a particular sequence is not anticipated during model development, the system may fail 
to diagnose it. Moreover, as systems grow in complexity, tracing failure pathways become  increasingly 
infeasible using event -based logic alone . Functional modeling reflects a design er’s intent by capturing 
system goals and functions rather than enumerating events . These models encode the functional 
expectations and logical dependencies of the system, enabling the detection of faults based on deviations 
from expected behavior. These  models are hierarchical, representing goals, functions, subfunctions, and 

3 
 supporting structures.  They emphasize how components contribute to overall functions, enabling reasoning 
about failures in terms of lost or abnormal functionality, such as a pump failing to provide flow or a sensor 
failing to deliver information, rather than focusing solely on s pecific failure events  [12]. Functional 
approaches address several limitations of event -based models . First, they support completeness by enabling 
a structured analysis of the system’s functional architecture. This allows for the identification of faults that 
may cause the loss of required functions, even during the design or concept stage before any failu re data is 
available . Second, functional models naturally handle complex, multi -fault scenarios . Since  they capture 
interactions through functional dependencies, they can reason abo ut multiple simultaneous failures or 
cascading effects without relying on  every combination explicitly . Third, they promote generality. The same 
functional model can be applied throughout the system’s life cycle and across various analysis tasks. These 
include design -level Failure Mode and Effects Analysis (FMEA), runtime diagnosis, and what -if analysis. 
This flexibility is possible because the model abstracts away from specific event sequences and instead 
focuses on invariant functional relationships  [13].  
2.2. DML Model Applications  
For system -level modeling, DML models are success logics in the form of powerful hierarchies to represent 
system knowledge  [3], [14], [15], [16] . This hierarchical representation becomes particularly valuable in 
safety -critical domains. System safety focuses on applying engineering principles, standards, and 
techniques to minimize risk while ensuring a system remains effective, reliable, and cost -efficient 
throughout its life cycle. Using a DML, the degree of success (or failure), approximate full -scale system 
physical values, and transition effects  amongst components  can be analyzed. DML provides an effective 
model for describing the causal effect s of failures or disturbances in complex systems  [17]. Figure 1 
conceptually illustrates the DML framework, highlighting the hierarchical decomposition from objectives 
to basic elements and the interdependencies between functional and structural. The functional hierarchy 
(top-down) moves from objectives to functions and sub -functions, capturing the system’s purpose and 
behavior. The structural hierarchy decomposes the system into elements and basic components, reflecting 
its physical or logical com position. Arrows indicate causal ("Why –How") and compositional  ("Part -of") 
relationships, highlighting interdependencies. This structure supports systematic analysis of complex 
systems by linking high -level goals to low -level elements.  
 
Figure 1. Conceptual DML Model  [15] 


4 
 Two key causal relationships can be extracted from DML. The first involves determining the ultimate effect 
of a failure, and the second involves identifying the paths through which a function can be achieved or a 
subsystem can successfully operate. It applies reductionist principles, where qualities represent functions 
and goals, while objects and relationships are structured through success trees and  logical modeling 
including Boolean, physical, and fuzzy logic  [14]. This integration enhances DML’s ability to model time -
dependent behaviors and fault propagation within complex systems. This model  has been applied to various 
applications of modeling . In nuclear power plants, DML has been used to model Direct Containment 
Heating in Pressurized Water Reactor [16]. In renewable energy, it has supported reliability analysis of 
geared wind turbines [18]. DML has also proven effective in analyzing interactions between hardware, 
software, and human elements in cyber -physical systems  [19], [20] . In the aerospace sector, it has been used 
to identify critical points in system reliability [21]. Additionally, DML has supported quality assurance 
across the software development life cycle [22].  
It is important to note that several naming conventions have been used to describe what is fundamentally a 
single family of DML models with common underlying principles. The earliest form of this modeling 
approach was the Goal Tree Success Tree (GTST)  [12], which laid the foundation for functional reasoning 
in complex systems. This was followed by the development of the Master Plant Logic Diagram (MPLD)  
[23], [24] , originally  created for use in nuclear power plants to represent plant logic and support reliability 
assessment . Over time, MPLD evolved into several extensions and refinements.  A prominent variant is the 
GTST  with Master Logic Diagram  (MLD) , referred to as GTST -MLD  [17], [22] . This version combines a 
functional hierarchy, represented by the GTST  with a structural model, represented by the MLD  that 
captures component relationships and dependencies, supporting both dynamic and static system 
representations. Another well -established form is the Dynamic Master Logic Diagram  (DMLD)  [14] which 
emphasizes the modeling of uncertain, evolving, and time -dependent behaviors in complex systems. While 
terminology may differ depending on modeling emphasis, these approaches are functionally equivalent and 
share the common goal of representing syst em logic for diagnostic reasoning and reliability analysis.  
Throughout this paper, this family of models is collectively referred to as DML . 
2.3. LLMs and KGs in Fault Diagnostics  
Recent advancements in LLMs have accelerated the use of  fault diagnostics, particularly in industrial and 
safety -critical domains. Traditional fault diagnosis methods rely heavily on rule -based models or historical 
fault logs, which require extensive domain expertise and are often inefficient in handling compl ex, multi -
layered system interactions. Integrating LLMs with diagnostic methods makes timely detection of faults 
more practical  and easier to scale by enabling an understanding of fault causes and providing decision 
support through NLP . As demonstrated in  [25], a diagnostic model was used to detect faults in a nuclear 
power plant, while an LLM acted as an interface to explain fault conditions and their potential causes to 
human operators, enhancing situational awareness and response.  
A study introduced  the method of FD-LLM  [26], fram ed machine fault diagnosis as a text classification 
task by converting vibration signals into textual token sequences or summaries. Fine -tuned LLaMA models 
were employed, achieving superior performance compared to traditional deep learning models and 
demonst rating that LLMs can effectively process non -textual diagnostic information when appropriately 
structured.  Improvements were reported in accurately identifying fault root causes from symptom 
descriptions, highlighting the potential of LLMs for natu ral language -based symptom interpretation and 
fault retrieval. To enhance the reliability and reasoning capabilities of LLM -based diagnostics, several 
studies have proposed integration with KGs. One approach, Root -KGD  [27], combined industrial process 
data with a structured KG representing system topology and causal dependencies, enabling more accurate 
root cause failure  identification by guiding LLM reasoning through formalized domain knowledge . In the 
context of CNC machine fault diagnosis  [28], a KG -embedded LLM architecture was proposed. A 

5 
 machining -process KG was automatically constructed from maintenance records, and its structured 
information was embedded into the LLM to support fault classification and provide explainable fault 
identification through natural language outputs .  
Integration of KGs into fault diagnostics has been explored to address the need for structured causal 
reasoning. A multi -level KG was developed to represent rotating machinery faults  [29], and Bayesian 
inference was applied to trace symptom -cause pathways within the graph, achieving a 91.1% diagnostic 
accuracy under missing data condition s. This demonstrated that the structured representation of symptom -
cause relationships within a KG can enhance diagnostic robustness.  The combination of structured 
knowledge retrieval and LLM reasoning enabled interpretable and accurate fault diagnosis based on free -
form symptom descriptions. The combination of LLMs with KGs has also demonstrated enhanced fault 
reasoning capabilities.  
A hybrid model for aviation assembly diagnostics achieved 98.5% accuracy in fault localization and 
troubleshooting through subgraph -based reasoning  [30]. In vehicle fault diagnostics, a KG -driven analysis 
system was developed that maps error codes and system alerts to potential failure causes [31]. This system 
leveraged LLMs to process unstructured diagnostic data, such as error logs and maintenance reports, 
transforming it into structured representations within a KG. A reasoning framework was introduced to infer 
root causes by linking symptoms to failure mechanisms stored in the KG.  In [32], a KG-based in -context 
learning approach was proposed to enhance fault diagnosis in industrial sensor networks. A domain -specific 
KG was constructed to encode expert knowledge, and a long -length entity similarity retrieval mechanism 
was used to select relevant knowledge, which was then supplied to a large language model for causal 
reasoning over fault symptom text. The method demonstra ted improved fault localization accuracy and 
enhanced the interpretability of diagnostic outputs compared to traditional LL M approaches.  
These studies  demonstrated that incorporating LLM -driven fault reasoning improved diagnostic accuracy 
and reduced troubleshooting time compared to traditional rule -based or statistical models. Structuring fault 
data within a KG provided traceable and explainable reason ing paths, assisting engineers and technicians 
in understanding failure causes. Collectively, these studies illustrate the growing role of LLMs and KGs in 
fault diagnostics. LLMs enable flexible interpretation of symptom descriptions and support natural 
language interaction, while KGs structure domain -specific knowledge to guide reasoning processes. Their 
integration shows promise  for enhancing fault retrieval and root cause analysis across various technical 
domains.  
3. Research Overview  
As discussed in Section 2 , although LLMs  and KGs  show promise for assisting with  system diagnostic s 
from textual documentation, their role in structured functional decomposition is not fully developed. 
Similarly, KGs are primarily used for retrieving fault information rather than representing system logic or 
dependencies, and few studies have explored integrating com putational reasoning layers to enable real -time 
diagnostic analysis.  To address these gaps, this paper introduces an LLM -informed, KG -based dia gnostic 
framework for constructing  and using  the DML model  derived from system design and operational 
documents. The framework leverages LLMs to support diagnostics, reliability assessment, and decision -
making  for a specific engineering system . It has three  main objectives:  
a. Automate the generation of scalable functional models by extracting structured relationships from 
system documentation , including system descriptions and specifications .  
b. Enable interactive, hierarchical fault analysis through natural language -driven upward and 
downward reasoning. Upward reasoning evaluates how individual component failures affect 

6 
 higher -level system functions, while downward reasoning explores which functional paths and 
component conditions must be satisfied to maintain or restore system objectives .  
c. Support interpretive analysis of system goals, functions, and dependencies by leveraging Graph -
based Retrieval -Augmented Generation (Graph -RAG)  to retrieve and contextualize relevant 
segments of the  KG in response to user queries . 
4. Proposed Approach  
Figure 3 presents the LLM -Informed Diagnostic Framework, which consists of two main stages: model 
construction and model interaction. In the model construction stage, system descriptions are processed by 
an LLM -based workflow  to extract DML logic, which organizes the system into hierarchical functions , 
components  and their relationships . This representation is then used to build a KG, which serves as the core 
system model. The KG can be further enhanced by incorporating expert knowledge and real -time 
operat ional data processed throug h Machine Learning (ML) or Deep Lea rning ( DL) models , allowing it to 
infer and reflect system states and conditions dynamically.  In the diagnostic model interaction section, an 
LLM agent would determine the intention  of the user and would invoke one of the tools available to it based 
on the  query  to execute. These tools are used to trace the KG to generate diagnostic insights. The results 
would be communicated to the user by the LLM.  
 
Figure 2. LLM -Informed Diagnostic Framework  


7 
 4.1. Model Construction  
The development of the diagnostic model begins with processing textual system information, including 
manuals, system descriptions, and technical documentation. Preprocessing techniques such as text 
summarization and LLM -based Named Entity Recognition (NER) are applied to extract key details about 
system components, functions, and dependencies. These extracted elements are then used to derive a DML 
hierarchical structure that organizes the system into goals, functions, subfuncti ons, and components. This 
structured logic is subsequently translated into Cypher code, the query language used to construct the KG 
that represents the system’s components and functional relationships. The resulting KG provides a 
structured repository to support querying and reasoning for diagnostics and decision -making.  In this 
research, the KG is deployed  using Neo4j [33], a graph database platform that represents entities and their 
interconnections as property graphs, with attributes stored as node and relationship properties. Cypher 
enables the definition of system dependencies within the KG, linking components to their success 
conditions and subfunctions to their parent functions.  All language -based tasks in this framework  were 
performed using OpenAI's GPT -4 model through the ChatGPT API.  
4.2. Model Interaction  
Once the model is built, it needs to be used to generate diagnostic insights. This framework aims to go 
beyond simple queries by enabling deeper system analysis , addressing the fundamental diagnostic 
questions :  
• What  is happening? The model identifies current system conditions . For example, it can determine 
which components are degrading and which system functions are at risk.  
• Why is it happening? By tracing system dependencies, the model identifies apparent root causes of 
failures and analyzes contributing factors.  
• How  will it impact the system? The model assesses failure propagation and risk severity, predicting 
how component failure will affect system operations.  
To address these questions, the model enables cause -and-effect reasoning, allowing users to explore system 
behavior beyond basic retrieval. Instead of just fetching stored information, it supports queries such as:  
• If certain components fail, how will it affect the overall system?  
• What conditions must be met for a specific function to succeed?  
• Which components are essential to maintaining system functionality?  
To support this process, a set of predefined functions is made available to the LLM agent as tools for 
interfacing with the KG. These tools enable structured upward and downward tracing to analyze system 
dependencies and generate diagnostic insights. When a user submits a query, the LLM interprets the intent 
and selects the appropriate tool to execute. The output is then passed back to the LLM, which produces a 
human -readable explanation. In addition to tool -based diagnostics, the model supports general sys tem 
queries, such as explaining the system's hierarchy or functional structure . For these interpretive queries, the 
framework employs a Graph -RAG approach where relevant graph segments , such as goals or functions , are 
retrieved and embedded into the LLM’s prompt to support natural language generation . Both approaches 
rely on the KG, but in complementary ways. Diagnostic tools perform structured graph traversals to 
compute logic -based results, while Graph -RAG enables the LLM to produce contextual explanation s based 
on retrieved subgraphs. This dual strategy ensures the KG remains central to both reasoning and 
explanation.  

8 
 5. Case Study  
The proposed system  illustrated in the Piping and Instrumentation Diagram (P&ID ) of Figure 3 has been 
used as a case study to  implement the proposed LLM -informed diagnostic framework. This P&ID 
represents a simplified auxiliary feedwater system of a nuclear power plant. The auxiliary feedwater system 
in a pressurized light water nuclear plant ensures the safe and efficient supply of emergency cooling water 
to the steam generators during unexpected plant tra nsients. The system is structured with the main goal 
defined as "Ensure safe and effective operation of the system". This goal is supported by four primary 
functions: "Supply Feedwater", "Control Water Flow", "Manage System Integration and Response", and 
"Provide Emergency and Automated Response".  
 
Figure 3. P&ID of Simplified Auxiliary Feedwater Syste m [34] 
5.1. Model Construction  
To construct a KG representing DML logic from a system description (which included only major 
components and functions), a structured prompt chaining workflow was implemented, with each stage 
handled by a dedicated LLM call. The hierarchy follows the DML s tructure, starting with a high -level goal 
and breaking down into functions, subfunctions, and components, each linked to success conditions. 
Logical relationships are shown using binary AND or OR logic gates, which define how lower -level 
elements contribut e to achieving higher -level objectives.  
The workflow begins with an initial LLM call that summarizes the system description and extracts goals, 
functions, subfunctions, components, and success conditions . The result is passed to a second LLM that 
converts this information into a structured JSON format aligned with the DML hierarchy. A third LLM then 
transforms the JSON into Cypher queries for KG construction.  Each LLM call is followed by a gate, 
implemented as another LLM, that validates the output before the next stage proceeds. If validation fa ils, 


9 
 the workflow routes the input back to the relevant LLM for revision. The first gate checks for missing or 
incomplete information in the summary, including vague goals, incomplete function chains, or missing 
success criteria. The second gate validates the J SON structure by checking key formatting, nesting, and 
logical gate consistency. The third gate examines the generated Cypher queries to ensure they are 
syntactically correct.   
This gated prompt chaining design improves consistency, filters out errors early, and manages variability 
in LLM outputs. It is especially effective when task steps are clearly defined and expected outputs are 
explicitly structured . Figure 4 illustrates the prompt chaining workflow described above, showing the 
sequential LLM tasks and validation gates leading from the system description to the final KG -DML output. 
Each task is followed by an LLM -based gate, which ensures the correctne ss of the output before  advancing 
to the next stage. Feedback loops are included to allow correction and regeneration when validation fails.  
The specific prompts used for each LLM call in this workflow are provided in the Appendix  to this paper.  
 
Figure 4. Model Construction Implementation Through LLM -Based Workflow  
The KG representing the DML logic is structured hierarchically, starting from a high -level goal and 
descending through functions, subfunctions, components, and finally success conditions. Logical gates , 
such as AND or OR, define how each level contributes to achieving the level above. A goal may be achieved 
by multiple functions, each of which depends on one or more subfunctions that require specific components 
to operate successfully.  At the lowest level of the hierarchy, components are connected to succes s conditions 
through additional gates. Success conditions reflect observable or measurable outcomes that confirm 
whether a component is performing as intended. Attributes are stored within the nodes themselves and may 
include expert knowledge or informatio n derived from ML or DL models based on operational data or 
manual inspections.  
For example, a component such as a turbine -driven pump may contain attributes indicating the probability 
of being in various states , such as operational, degraded, or failed.  Attributes may also be present in higher -
level nodes , such as functions or goals. Their role and interpretation will be discussed in the Model 
Interaction (next section ). This hierarchical structure supports reasoning, traceability, and consistency 
throughout the KG -DML representation.  Figure 5 illustrates how the DML model is represen ted within the 
KG. For example, the subfunction "Manage Condensation Tanks" is fulfilled only if all three Condensation 
Storage Tanks (CSTs) operate successfully, as defined by an AND gate. Each tank must meet two success 
conditions : maintaining an appropriate water level and ensuring the absence of excessive sediment. This 
same hierarchical logic applies when tracing the model upward through functions and system -level goals.  


10 
  
Figure 5. KG Reflecting the DML Model  
5.2. Model Interaction  
As shown in Figure 2, model interaction begins when a user submits a natural language query to the system. 
The LLM agent interprets the query and selects from a set of predefined diagnostic functions, available to 
it as tools. These tools perform specialized tasks such as upw ard fault tracing and generation of success 
path sets. Each tool is implemented as an external code module that analyzes the KG based on the logical 
structure derived from the model. The LLM uses these tools to carry out structured reasoning over the graph  
and generate diagnostic insights. To enable accurate selection, the agent was fine -tuned on a dataset 
consisting of diverse user queries paired with their corresponding tool calls. This included both diagnostic 
queries requiring tool invocation and interpretive queries requiring Cypher query generation for KG 
retrieval . This dataset was manually constructed to include multiple phrasings, semantic variations, and 
tones in which users might pose the same diagnostic intent. These examples were then used to gu ide the 
fine-tuning process so that the model learns to map a wide range of natural language inputs to the 
appropriate tool.  
In the implemented tool for upward propagation, the success probability for each success condition 
𝑗 associated with a component is first evaluated. This is done using Equation 1, which computes the 
probability as a weighted sum over the component’s possible operational states. Each term combines the 
likelihood of the component being in state  𝑖 with the probability that it fulfills success condition 𝑗 in that 
state. A state refers to a possible condition of a component, such as operational, degraded, or failed. Each 
state influences the component's ability to fulfill its associated success conditions.  For example, consider a 
CST in the auxiliary feedwater system. One possible state of the CST is "failed ", which may be inferred 
from sensor data. The success condition for the CST could be defined as "maintains sufficient water level 
for feedwater supply." If operational data indicates  a high probability that the CST is  in a failed state, the 
likelihood of satisfying this success condition would be correspondingly low. This affects the overall 
success probability of the subfunction "Manage Condensation Tanks," which depends on all CSTs through 
an AND gate. Thus, the fail ure of even one CST can reduce the success probability of the higher -level 
function and system goal.  


11 
 𝑃(𝑆𝑢𝑐𝑐𝑒𝑠 𝑠𝑗|𝐷𝑎𝑡𝑎 )= ∑𝑃(𝑆𝑢𝑐𝑐𝑒𝑠 𝑠𝑗|𝑆𝑡𝑎𝑡 𝑒𝑖)𝑃(𝑆𝑡𝑎𝑡 𝑒𝑖|𝐷𝑎𝑡𝑎 )𝑁
𝑖=1  
(1) 
 
• 𝑁: Total number of operational states for a component . 
• 𝑃(𝑆𝑢𝑐𝑐𝑒𝑠 𝑠𝑗|𝑆𝑡𝑎𝑡 𝑒𝑖):  The probability of success for the success condition 𝑗. This reflects how 
likely the component will fulfill the success condition 𝑗 under state  𝑖.  
• 𝑃(𝑆𝑡𝑎𝑡 𝑒𝑖|𝐷𝑎𝑡𝑎 ): The probability of the component being in state 𝑖 given the data which can be 
evidence of events or numerical information.  
After evaluating 𝑃(𝑆𝑢𝑐𝑐𝑒𝑠 𝑠𝑗|𝐷𝑎𝑡𝑎 ) for all success conditions associated with each component in the 
system, the results are aggregated using logical gates to compute a single success probability for each 
component . Success probability represents how well the component is fulfilling its intended function. It is 
based on the combined satisfaction of all defined success conditions. Each condition reflects a specific 
performance indicator. The aggregation of these conditi ons provides a quantitative measure of the 
component’s overall operational effectiveness.  The KG stores each element of Equation 1 as attributes 
within component nodes, including both the conditional success probabilities and the state likelihoods 
deriv ed from data.  In the context of engineering diagnostics, this data may include sensor readings (e.g., 
temperature, pressure, vibration), event logs, failure reports, and maintenance histories.  These attributes 
serve as the basis for upward propagation and are retained in the KG to support traceability  and diagnostic 
reporting.  Once a single success probability is determined for each component, these values are propagated 
upward through the DML hierarchy using additional logical gates. The gates define how component -level 
probabilities combine to determine the success of associat ed subfunctions, functions, and ultimately 
system -level goals. If the success probability of an upper -level node falls below a predefined threshold, the 
tool co nsiders that node to be impacted. The corresponding logic  that performs the probabilistic propagation  
is captured in the pseudocode shown in Figure 6.  
To estimate  𝑃(𝑆𝑡𝑎𝑡 𝑒𝑖|𝐷𝑎𝑡𝑎 ), various strategies can be applied depending on data availability and system 
characteristics. In the absence of real -time sensor data, these probabilities can be derived from expert 
judgment, or reliability reports, which provide baseline estimates of fai lure or degradation likelihoods. 
These priors can be refined as new operational data becomes available.  When numerical indicators such as 
temperature, pressure, or vibration readings are accessible, ML or DL models trained on historical labeled 
data can be used to estimate state probabilities more dynamically. In systems requiring continuous 
monitoring and p robabilistic inference under uncertainty, particle filtering techniques may be used. Particle 
filters apply a sequential Monte Carlo approach to approxi mate probability distributions using a set of 
weighted samples, enabling real -time Bayesian inference even in nonlinear or non -Gaussian conditions  
[35], [36] .  
For downward propagation, given an upper -level node, the tool traces the KG downward to determine the 
required paths for achieving that node’s success. Using the defined gates, it identifies the necessary 
dependencies at each level. The path -set generation method determines the minimal components  required 
for system functionality by recursively traversing the KG. Starting from a specified node, the process 
follows dependencies downward until reaching the Component and Success Condition levels. At each step,  
the method evaluates the logical dependencies based on the gate type. If an AND gate is present, all 
dependencies must be met simultaneously, requiring a Cartesian product of the success path -sets from the 
child nodes to generate valid paths. In contrast,  for an OR gate, only one dependency needs to succeed, so 
the success path -sets from the child nodes are aggregated without combination, representing alternative 
paths to success. This structured approach ensures that the generated success path -sets accura tely reflect 

12 
 the minimal elements necessary to maintain system operability.  The approach of downward propagation is 
formalized by the pseudocode in Figure 7. 
 
Figure 6. Upwards Propagation Pseudocode   


13 
  
Figure 7. Downward Propagation Pseudocode  
5.3. Interaction  Interface  
The diagnostic interface enables natural language interaction between the user and the system, allowing 
users to explore system behavior and fault scenarios. As illustrated in Figure 8, users can ask questions such 
as the impact of a specific component fai lure or how a given function can succeed. When queried about the 
impact of failure of a CST, the LLM agent invokes the upward propagation tool to trace the impact across 
the system hierarchy. Because the CSTs are connected through an AND gate, the success of higher -level 
nodes depends on the simultaneous functionality of all CSTs. Therefore, the failure of even a single CST 
significantly reduces the probability of success for the related subfunctions, functions, and overall system 
goals. Conversely, when as ked about the success conditions for a function like "Supply Feedwater ", the 
agent employs downward tracing to identify all minimal success paths. Results are returned in a human  


14 
 readable format, supporting transparent and intuitive diagnostic analysis without requiring technical  
familiarity with the underlying model.  
 
Figure 8. Interaction Interface Example Containing User Sample Questions   
6. Evaluation  
The evaluation of the proposed framework was designed to assess both the structural accuracy of the KG 
generated from system documentation using the DML hierarchy and the effectiveness of the LLM agent to 
correctly interpret and respond to diagnostic queries through tool invocation and knowledge retrieval.  
6.1. KG Validation  
To assess the accuracy of the model construction pipeline, we conducted five independent runs using the 
same system description for the auxiliary feedwater system. In each run, the framework automatically 
extracted elements of the DML model, including goal s, functions, subfunctions, components, success 
conditions, and logical gates. These outputs were then manually validated. The validation involved 
examining the resulting KG -DML and cross -referencing its contents with the original system description . 
An el ement was considered correctly identified if it was both semantically relevant and structurally 
consistent with the source material. Elements were labeled as hallucinated if they introduced information 
that was not present  in the documentation or if they misrepresented relationships.  The average  results across 
the five runs are summarized in Table 1, which reports the ground truth element counts, the number of 
correctly extracted elements, the average number of hallucinated elements, and the corresponding 
extraction accuracy for each model component.  
 


15 
  
KG Element  Ground 
Truth  Avg. 
Correct  Avg. 
Hallucinated  Extraction Accuracy 
(%) 
Goals  1 1 0 100.0 
Functions  4 3.8 0.2 95.0 
Subfunctions  9 8.6 0.4 95.6 
Components  19 18.2 0.8 95.8 
Logical Gates 
(AND/OR)  33 30.8 2.2 93.3 
Success Conditions  39 37.2 1.8 95.4 
Table 1. Average Validation of KG Elements Across 5 Runs  
6.2. LLM Agent Query Evaluation  
We evaluated the performance of the LLM agent in interpreting natural language queries and selecting the 
appropriate diagnostic tools or the knowledge -based retrieval mechanism. A test set comprising 60 queries 
was developed, with queries evenly distribute d across three primary task types: upward reasoning, 
downward reasoning, and explanatory queries. The u pward reasoning task involved diagnosing how faults 
propagate from component -level failures to higher -level system functions. The d ownward reasoning task  
focused on identifying the minimal set of components required to achieve a particular function or system 
goal. Explanatory queries required the agent to retrieve structural or functional information from the KG 
using a Graph -RAG method, in which relevant nodes and attributes  are extracted via automatically 
generated Cypher queries  by the LLM . The evaluation was conducted across five independent runs of the 
same 60 -query dataset to account for variability in LLM outputs. Each query was assessed based on three 
criteria. These included whether the agent correctly classified the task type, whether the tool or retrieval 
method was selected appropriately, and whether the extracted arguments or generated Cypher queries were 
correct. Argument extraction and Cypher generation accuracy were calculated only for correctly classified 
queries to ensure that e xecution quality reflects successful task interpretation.  
Task Type  Query 
Set Size  Avg. Correct Task 
Classificatio n Avg. Valid Tool/Query 
Input  Extraction 
Accuracy (%)  
Upward 
Reasoning  20 19.8 19.2 97.0 
Downward 
Reasoning  20 19.6 19.6 100.0  
Explanatory 
Query  20 20.0 19.2 96.0 
Table 2. LLM Agent Evaluation by Task Type  Across 5 Run s 
The LLM agent demonstrated consistent performance across all query types. Averaged over five 
independent runs of a 60 -query test set, the agent achieved high classification accuracy, correctly identifying 
the intended reasoning or retrieval task in nearly all cases. For reasoning tasks, the agent achieved an 
argument extraction accuracy of 97.0% for upward reasoning, with an average of 19.2 correct extractions 
out of 19.8 correctly classified queries per run. For downward reasoning, both classification and extraction 
accuracy averaged 19.6 per run, corresponding to an extraction accuracy of 100.0 %. For explanatory 

16 
 queries, the agent correctly classified all 20 queries per run on average and successfully generated Cypher 
queries for 19.2 of them, resulting in a 9 6.0% accuracy following correct classification. These results 
demonstrate the agent’s reliability in distinguishing between diagnostic and interpretive tasks and its 
effectiveness in performing structured reasoning and knowledge retrieval based on the syste m model.  
7. Conclusion and Outlook  
7.1. Limitations and Challenges  
Despite the demonstrated effectiveness of the proposed LLM - and KG -based diagnostic framework for 
automating DML model construction and enabling interactive fault analysis, several limitations remain. 
LLM outputs, while powerful, are varied  and may generate hallucinated or incomplete structures even when 
guided by validation gates and the prompt -chaining workflow. This highlights the need for a human -in-the-
loop process, where domain experts review and revise the automatically generated DML models to ensure  
logical consistency and domain accuracy. The framework also assumes access to well -structured 
documentation and reliable operational or historical data, which may not always be available in practice. 
Although the proposed architecture reduces manual effort, expert oversight remains essential. Furthermore, 
while the evaluation reports high element -level extraction accuracy, it does not capture the semantic impact 
of missing critical nodes. In DML models, elements such as gat es, subfunctions, or success cond itions are 
often essential to maintaining the integrity of fault propagation paths. The omission of even a single high -
impact node can break logical chains and lead to incomplete or misleading diagnostics. This limitation 
suggests a need for broader evalua tion strategies that assess whether generated models preserve full 
diagnostic reasoning capabilities. The current validation also relies on a curated query set based on typical 
expert interactions, which, while practical, may not reflect edge cases, ambigu ous phrasing, or linguistic 
variation. More comprehensive testing involving adversarial queries, paraphrased inputs, and real -user 
feedback will be necessary to improve the robustness and generalizability of the system. Lastly, adapting 
the framework to do mains beyond nuclear diagnostics may require customized prompts and tool 
modifications, which could limit its immediate applicability elsewhere.  
Additionally, while the framework has been demonstrated on a moderately scoped system, its performance 
and scalability in large -scale, highly complex systems remain untested. The current case study involves a 
relatively simple subsystem with limited compon ents and interactions. Future work should investigate how 
the framework performs when applied to large, mission -critical systems with deeply nested hierarchies, 
extensive dependencies, and real -time operational data. Understanding the computational require ments, 
performance bottlenecks, and accuracy trade -offs in such settings will be essential for broader adoption.  
7.2. Conclusion  
The integration of LLMs and KGs into diagnostic modeling marks a significant advancement in automating 
complex system analysis. This research introduces a scalable, AI -driven framework that streamlines the 
generation of structured diagnostic models and enh ances predictive accuracy and fault reasoning through 
natural language interaction. By reducing reliance on manual modeling and enabling structured, explainable 
diagnostics, the approach adapts effectively to evolving system configurations.  Depending on th e query 
type, system knowledge from the KG is either processed through diagnostic tools or embedded into the 
LLM prompt, as previously described . This enables the LLM to generate responses that are both context -
aware and grounded in system logic. The framework also facilitates human -AI collaboration by allowing 
users to interact with system behavior through intuitive queries, lowering the technical barrier to advanced 
diagnostics. It extends the utility of functional modeling techniques such as DML, which h ave traditionally 
required extensive domain expertise and manual effort. While demonstrated on a nuclear power application, 

17 
 the framework is broadly applicable to other mission -critical domains, including advanced manufacturing 
systems such as integrated circuit fabrication facilities. Designed for continual refinement using operational 
feedback and real -time data, the framewor k is positioned to support high -reliability industries with 
improved diagnostics, proactive risk management, and system resilience.  The proposed framework was 
validated through comprehensive evaluations of both KG construction and LLM -based query interpret ation. 
Across five independent runs, the system consistently produced accurate DML -based models, achieving 
over 90% extraction accuracy for critical logic structures such as gates and success conditions, and perfect 
identification of high -level goals. The LLM age nt demonstrated strong performance across 60 diagnostic 
and explanatory queries, with high classification accuracy, reliable tool invocation, and consistent argument 
extraction. These results validate the framework’s ability to generate interpretabl e, graph -based diagnostics 
directly from unstructured documentation. Overall, this work establishes a solid foundation for next -
generation diagnostic systems that combine natural language interaction with structured reasoning.  The 
implementation code and supplementary materials for the proposed framework are available at: 
https://github.com/s -marandi/LLM -Based-Complex -System-Diagnostics  
7.3. Future Work  
Future work will focus on advancing the evaluation and validation of automatically generated DML models. 
Building on the current framework, which includes multi -run assessments of extraction accuracy and 
hallucination rates, future evaluations will incorpo rate deeper semantic validation techniques. One direction 
involves benchmarking generated cut -sets and path -sets against expert -engineered baselines to assess both 
logical soundness and coverage. In addition to element -level accuracy, graph -level structura l metrics such 
as connectivity fidelity, dependency correctness, and fault propagation traceability will be introduced. 
Evaluation will also extend to the robustness of LLM behavior under varied prompt conditions and 
alternative phrasings. Autonomous LLM a gents will be further explored for iterative model refinement, 
leveraging feedback loops to enhance precision, reducing  hallucinations, and adaptively correct errors over 
time by moving toward scalable, self -improving diagnostic model generation.  Advanced diagnostic 
capabilities will also be developed, including probabilistic assessments of component criticality in cases 
where partial functionality can be tolerated. This will enable the framework to recommend optimal 
maintenance or mitigation strategies und er uncertainty. As systems grow more complex, enhancements will 
focus on integrating real -time operational data and applying data fusion techniques to combine inputs such 
as sensor readings, maintenance records, and expert observations into the KG. To bett er model dynamic 
behavior, future efforts will introduce more sophisticated gating mechanisms capable of capturing time -
dependent relationships and evolving system states. The model construction process will also be extended 
with advanced NER techniques to  improve the precision and depth of information extracted from technical 
documentation. For systems with lengthy descriptions that exceed a single LLM context window, more 
effective text chunking and summarization strategies will be implemented, along with  models that support 
larger token capacities. Together, these enhancements will further align the framework with the demands 
of real -time diagnostics in safety -critical, complex engineered systems.  
 
  

18 
 References  
[1] U. Yildirim, F. Campean, and H. Williams, “Function modeling using the system state flow 
diagram,” Artif. Intell. Eng. Des. Anal. Manuf. , vol. 31, no. 4, pp. 413 –435, Nov. 2017, doi: 
10.1017/S0890060417000294.  
[2] M. Modarres, R. Irehvije, and M. Lind, “A Comparison of Three Functional Modeling Methods,” 
Jun. 1995.  
[3] Y.-S. Hu and M. Modarres, “Time -dependent system knowledge representation based on dynamic 
master logic diagrams,” Control Eng. Pract. , vol. 4, no. 1, pp. 89 –98, Jan. 1996, doi: 10.1016/0967 -
0661(95)00211 -5. 
[4] A. Vaswani et al. , “Attention Is All You Need,” Aug. 02, 2023, arXiv : arXiv:1706.03762. doi: 
10.48550/arXiv.1706.03762.  
[5] A. Hogan et al. , “Knowledge Graphs,” 2020, doi: 10.48550/ARXIV.2003.02320.  
[6] C. A. Ericson, Hazard analysis techniques for system safety , 2. ed. Hoboken, NJ: Wiley, 2016.  
[7] M. Stamatelatos and W. E. Vesley, “Fault tree handbook with aerospace applications,” 2002. 
[Online]. Available: https://api.semanticscholar.org/CorpusID:61105226  
[8] R. Mareş and M. P. Stelea, “The application of event tree analysis in a work accident at 
maintenance operations,” MATEC Web Conf. , vol. 121, p. 11013, 2017, doi: 
10.1051/matecconf/201712111013.  
[9] J. D. Andrews and S. J. Dunnett, “Event -tree analysis using binary decision diagrams,” IEEE 
Trans. Reliab. , vol. 49, no. 2, pp. 230 –238, Jun. 2000, doi: 10.1109/24.877343.  
[10] H.-L. Zhu, S. -S. Liu, Y. -Y. Qu, X. -X. Han, W. He, and Y. Cao, “A new risk assessment method 
based on belief rule base and fault tree analysis,” Proc. Inst. Mech. Eng. Part O J. Risk Reliab. , vol. 236, 
no. 3, pp. 420 –438, Jun. 2022, doi: 10.1177/1748006X211011457.  
[11] P. Weber and L. Jouffe, “Complex system reliability modelling with Dynamic Object Oriented 
Bayesian Networks (DOOBN),” Reliab. Eng. Syst. Saf. , vol. 91, no. 2, pp. 149 –162, Feb. 2006, doi: 
10.1016/j.ress.2005.03.006.  
[12] I. S. Kim and M. Modarres, “Application of goal tree -success tree model as the knowledge -base 
of operator advisory systems,” Nucl. Eng. Des. , vol. 104, no. 1, pp. 67 –81, Oct. 1987, doi: 10.1016/0029 -
5493(87)90304 -9. 
[13] J. Wu, X. Zhang, M. Song, and M. Lind, “Challenges in Functional Modelling for Safety and 
Risk Analysis,” in Proceeding of the 33rd European Safety and Reliability Conference , Research 
Publishing Services, 2023, pp. 1892 –1899. doi: 10.3850/978 -981-18-8071 -1_P132 -cd. 
[14] Y.-S. Hu and M. Modarres, “Evaluating system behavior through Dynamic Master Logic 
Diagram (DMLD) modeling,” Reliab. Eng. Syst. Saf. , vol. 64, no. 2, pp. 241 –269, May 1999, doi: 
10.1016/S0951 -8320(98)00066 -0. 
[15] M. Modarres and S. W. Cheon, “Function -centered modeling of engineering systems using the 
goal tree –success tree technique and functional primitives,” Reliab. Eng. Syst. Saf. , vol. 64, no. 2, pp. 
181–200, May 1999, doi: 10.1016/S0951 -8320(98)00062 -3. 
[16] Y.-S. Hu and M. Modarres, “Logic -Based Hierarchies for Modeling Behavior of Complex 
Dynamic Systems with Applications,” in Fuzzy Systems and Soft Computing in Nuclear Engineering , vol. 
38, D. Ruan, Ed., in Studies in Fuzziness and Soft Computing, vol. 38. , Heidelberg: Physica -Verlag HD, 
2000, pp. 364 –395. doi: 10.1007/978 -3-7908 -1866 -6_17.  

19 
 [17] M. Modarres, “Functional modeling of complex systems with applications,” in Annual Reliability 
and Maintainability. Symposium. 1999 Proceedings (Cat. No.99CH36283) , Washington, DC, USA: IEEE, 
1999, pp. 418 –425. doi: 10.1109/RAMS.1999.744153.  
[18] Y. F. Li, S. Valla, and E. Zio, “Reliability assessment of generic geared wind turbines by GTST -
MLD model and Monte Carlo simulation,” Renew. Energy , vol. 83, pp. 222 –233, Nov. 2015, doi: 
10.1016/j.renene.2015.04.035.  
[19] Z. Hao, F. Di Maio, and E. Zio, “A sequential decision problem formulation and deep 
reinforcement learning solution of the optimization of O&M of cyber -physical energy systems (CPESs) 
for reliable and safe power production and supply,” Reliab. Eng. Syst. Saf. , vol. 235, p. 109231, Jul. 2023, 
doi: 10.1016/j.ress.2023.109231.  
[20] F. D. Maio, “Simulation -Based Goal Tree Success Tree for the Risk Analysis of Cyber -Physical 
Systems”.  
[21] C. Guo, S. Gong, L. Tan, and B. Guo, “Extended GTST‐MLD for Aerospace System Safety 
Analysis,” Risk Anal. , vol. 32, no. 6, pp. 1060 –1071, Jun. 2012, doi: 10.1111/j.1539 -6924.2011.01718.x.  
[22] M. Modarres and N. Kececi, Software Development Life Cycle Model to Ensure Software Quality . 
1998.  
[23] R. N. M. Hunt and M. Modarres, “Integrated Economic Risk Management in a Nuclear Power 
Plant,” in Uncertainty in Risk Assessment, Risk Management, and Decision Making , V. T. Covello, L. B. 
Lave, A. Moghissi, and V. R. R. Uppuluri, Eds., Boston, MA: Springer US, 1987, pp. 435 –443. doi: 
10.1007/978 -1-4684 -5317 -1_34.  
[24] M. Modarres, J. H. Zamanali, and J. Wang, “Applications of Master Plant Logic Diagram 
(MPLD) PC -Based Program in Probabilistic Risk Assessment,” Feb. 1991.  
[25] A. J. Dave, T. N. Nguyen, and R. B. Vilim, “Integrating LLMs for Explainable Fault Diagnosis in 
Complex Systems,” Feb. 08, 2024, arXiv : arXiv:2402.06695. doi: 10.48550/arXiv.2402.06695.  
[26] H. A. A. M. Qaid, B. Zhang, D. Li, S. -K. Ng, and W. Li, “FD -LLM: Large Language Model for 
Fault Diagnosis of Machines,” Dec. 02, 2024, arXiv : arXiv:2412.01218. doi: 10.48550/arXiv.2412.01218.  
[27] J. Chen, J. Qian, X. Zhang, and Z. Song, “Root -KGD: A Novel Framework for Root Cause 
Diagnosis Based on Knowledge Graph and Industrial Data,” Jun. 19, 2024, arXiv : arXiv:2406.13664. doi: 
10.48550/arXiv.2406.13664.  
[28] P. Wu, X. Mou, L. Gong, H. Tu, L. Qiu, and B. Yang, “An automatic machine fault identification 
method using the knowledge graph –embedded large language model,” Int. J. Adv. Manuf. Technol. , Apr. 
2025, doi: 10.1007/s00170 -025-15555 -2. 
[29] C. Cai, Z. Jiang, H. Wu, J. Wang, J. Liu, and L. Song, “Research on knowledge graph -driven 
equipment fault diagnosis method for intelligent manufacturing,” Int. J. Adv. Manuf. Technol. , vol. 130, 
no. 9 –10, pp. 4649 –4662, Feb. 2024, doi: 10.1007/s00170 -024-12998 -x. 
[30] P. Liu, L. Qian, X. Zhao, and B. Tao, “Joint Knowledge Graph and Large Language Model for 
Fault Diagnosis and Its Application in Aviation Assembly,” IEEE Trans. Ind. Inform. , vol. 20, no. 6, pp. 
8160 –8169, Jun. 2024, doi: 10.1109/TII.2024.3366977.  
[31] T. Sun, F. Zeng, and X. Liu, “A Fault Analysis and Reasoning Method for Vehicle Information 
Systems Based on Knowledge Graphs,” in 2024 IEEE 24th International Conference on Software Quality, 
Reliability, and Security Companion (QRS -C), Cambridge, United Kingdom: IEEE, Jul. 2024, pp. 926 –
933. doi: 10.1109/QRS -C63300.2024.00123.  

20 
 [32] X. Xie, J. Wang, Y. Han, and W. Li, “Knowledge Graph -Based In -Context Learning for Advanced 
Fault Diagnosis in Sensor Networks,” Sensors , vol. 24, no. 24, p. 8086, Dec. 2024, doi: 
10.3390/s24248086.  
[33] Neo4j, Inc., “Neo4j GitHub Repository,” GitHub. [Online]. Available: 
https://github.com/neo4j/neo4j  
[34] M. Modarres, M. Kaminskiy, and V. Krivtsov, Reliability engineering and risk analysis: a 
practical guide , Third edition. Boca Raton: CRC Press,  Taylor & Francis Group, CRC Press is an imprint 
of the Taylor & Francis Group, an informa business, 2017.  
[35] J. Elfring, E. Torta, and R. Van De Molengraft, “Particle Filters: A Hands -On Tutorial,” Sensors , 
vol. 21, no. 2, p. 438, Jan. 2021, doi: 10.3390/s21020438.  
[36] A. Doucet, N. Freitas, and N. Gordon, Eds., Sequential Monte Carlo Methods in Practice . New 
York, NY: Springer New York, 2001. doi: 10.1007/978 -1-4757 -3437 -9. 
 
  

21 
 Appendix  
 
Figure 9, Step 1  and Gate 1  Prompt s  
 
Figure 10, Step 2 and Gate 2 Prompts  


22 
  
Figure 11, Step 3 and Gate 3 Prompts  
