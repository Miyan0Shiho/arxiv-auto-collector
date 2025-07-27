# GenAI for Automotive Software Development: From Requirements to Wheels

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Krzysztof Lebioda, Andre Schamschurko, Alois Knoll

**Published**: 2025-07-24 09:17:13

**PDF URL**: [http://arxiv.org/pdf/2507.18223v1](http://arxiv.org/pdf/2507.18223v1)

## Abstract
This paper introduces a GenAI-empowered approach to automated development of
automotive software, with emphasis on autonomous and Advanced Driver Assistance
Systems (ADAS) capabilities. The process starts with requirements as input,
while the main generated outputs are test scenario code for simulation
environment, together with implementation of desired ADAS capabilities
targeting hardware platform of the vehicle connected to testbench. Moreover, we
introduce additional steps for requirements consistency checking leveraging
Model-Driven Engineering (MDE). In the proposed workflow, Large Language Models
(LLMs) are used for model-based summarization of requirements (Ecore metamodel,
XMI model instance and OCL constraint creation), test scenario generation,
simulation code (Python) and target platform code generation (C++).
Additionally, Retrieval Augmented Generation (RAG) is adopted to enhance test
scenario generation from autonomous driving regulations-related documents. Our
approach aims shorter compliance and re-engineering cycles, as well as reduced
development and testing time when it comes to ADAS-related capabilities.

## Full Text


<!-- PDF content starts -->

GenAI for Automotive Software Development:
From Requirements to Wheels
Nenad Petrovic1[0000−0003−2264−7369], Fengjunjie Pan1[0009−0005−8303−1156],
Vahid Zolfaghari1[0009−0004−0039−6014], Krzysztof Lebioda1[0000−0002−7905−8103],
Andre Schamschurko1[0009−0000−7030−0955], and Alois
Knoll1[0000−0003−4840−076X]
Technical University of Munich, Germany
nenad.petrovic@tum.de,f.pan@tum.de,v.zolfaghari@tum.de,
krzysztof.lebioda@tum.de, andre.schamschurko@tum.de, k@tum.de
Abstract. This paper introduces a GenAI-empowered approach to au-
tomateddevelopmentofautomotivesoftware,withemphasisonautonomous
and Advanced Driver Assistance Systems (ADAS) capabilities. The pro-
cess starts with requirements as input, while the main generated out-
puts are test scenario code for simulation environment, together with
implementation of desired ADAS capabilities targeting hardware plat-
form of the vehicle connected to testbench. Moreover, we introduce ad-
ditional steps for requirements consistency checking leveraging Model-
Driven Engineering (MDE). In the proposed workflow, Large Language
Models (LLMs) are used for model-based summarization of requirements
(Ecore metamodel, XMI model instance and OCL constraint creation),
test scenario generation, simulation code (Python) and target platform
code generation (C++). Additionally, Retrieval Augmented Generation
(RAG) is adopted to enhance test scenario generation from autonomous
driving regulations-related documents. Our approach aims shorter com-
pliance and re-engineering cycles, as well as reduced development and
testing time when it comes to ADAS-related capabilities.
Keywords: GenAI ·Large Language Model (LLM) ·Model-Driven En-
gineering (MDE) ·Retrieval Augmented Generation (RAG).
1 Introduction
Automotive industry is characterized by strict design, development, testing, and
manufacturing processes that must comply with a variety of regulations and
standards. These constraints often result in lengthy timelines from research and
developmenttofull-scaleproduction.Innovationinthisfieldisfurtherchallenged
by time-intensive, costly procedures that depend heavily on specialized domain
expertise and manual effort. Considering the increasing complexity of Software
Defined Vehicle (SDV) products incorporating autonomous and assisted driv-
ing capabilities over years [5], the development of even standard-sized vehicles
involves managing hundreds of thousands of requirements, which introduces ad-
ditional gap between development and production.arXiv:2507.18223v1  [cs.SE]  24 Jul 2025

2 N. Petrovic et al.
ArtificialIntelligence(AI),especiallythenovelGenAI-basedsolutionsexhibit
strong potential when it comes to bridging this gap and automotive software de-
velopment process automation. Current research indicates that the adoption of
GenAI in automotive software development primarily focuses on areas such as
requirementsmanagement,compliance,testscenariogeneration,andcodegener-
ation. Nevertheless, implementing GenAI in sensitive industries like automotive
brings significant challenges [14] [12]. A key concern is that GenAI models are
prone to hallucinations—producing plausible but inaccurate or fabricated infor-
mation. This inherent uncertainty makes the direct use of unverified AI outputs
impractical, so additional validation steps through formally grounded methods
such as Model-Driven Engineering (MDE) are identified as a possible solution
to tackle them. On the other side, there is a growing need for smaller, locally de-
ployable AI models tailored to specific, narrowly defined tasks, due to potential
constrains of exposing automotive software assets (requirements and code).
In this paper, we consider the adoption of state-of-art GenAI models and
techniques - particularly Large Language Models (LLM) and Retrieval Aug-
mentedGeneration(RAG),consideringthecompletesoftwaredevelopmentwork-
flow - from requirements and compliance documents to tests and target platform
code generation. Additionally, we leverage MDE and hallucination tackling tech-
niquestoinordertoincreasetrustworthinessandreproducibilityoftheapproach.
2 Methodology
In this section, we present a workflow for a GenAI-powered approach to au-
tomotive software development from end to end, building upon our previous
works [10]. The proposed workflow is depicted in Fig. 1, where we indicate step
automation by AI, along with auxiliary techniques (such as MDE). The work-
flow begins with the processing of input documents—such as customer-specific
requirements or regulatory standards—using Retrieval-Augmented Generation
(RAG). This step facilitates efficient extraction of relevant information, which
is then used to build datasets for code and test generation. In this context, reg-
ulatory documents like UN152 [15] often serve as the foundation for generating
test scenarios via RAG. Since visual elements (e.g., diagrams, graphs) in auto-
motive documentation can convey crucial information, Vision-Language Models
(VLMs) may also be used to extract such content [11]. The next key step involves
creating a formal representation, which serves as an intermediary between re-
quirements extraction and code generation. Large Language Models (LLMs) are
employed to summarize and structure the extracted requirements into a formal
template or metamodel, enabling early design-time checks such as compliance
verification. The metamodel itself can be manually crafted or constructed au-
tomatically. Once the requirements are checked for completeness, correctness,
and compliance, code generation can proceed using LLMs. Based on extracted
test scenario, the proposed vehicle configuration is first assessed in simulation
environment (Python code in CARLA). Afterwards, code for target platform
(vehicle mounted on the testbench and connected to simulation environment) is

GenAI for Automotive Software Development: From Requirements to Wheels 3
generated, together with complementary simulation code. While it is still rec-
ommended to include human reviewer in a loop, we try to reduce the need for
manual intervention through the use of multi-agent LLM systems for tackling
the possible hallucinations [13].
Fig.1: GenAI-empowered automotive software development workflow.
3 Implementation
In this section, we provide insights into implementation of distinct workflow
steps. Their integration was done relying on n8n workflow automation tool [6].
3.1 Model Checker
ModelCheckercomponentconsistsoftwoLLMagents:1)modelinstancegenera-
tion2)constraintgeneration.Theaimofthissolutionistoenabletheadoptionof
model-driven approach for consistency checking, where set of Object Constraint
Language Rule (OCL) is checked if satisfied within given XMI model instance.,
as depicted in Fig. 2. For the first task, we take HW/SW specifications and
Fig.2: Model instance and OCL rule generation for consistency checking.
system’s meta-model as input and create a XMI instance model representing

4 N. Petrovic et al.
the target system. We propose Llama 3.1-70B based solution that achieves a
comparable semantic score to GPT-4o. Instead of direct XMI generation, our
method makes use of simpler conceptual model notation as intermediate step in
order to improve performance [8]. On the other side, for OCL rule generation,
we adopt custom, fine-tuned model based on custom-tailored Llama3-8B [7] in
synergy with RAG [4]. The input of OCL generation is meta-model and design
constraint, e.g., from the reference architecture. Solution is locally deployable.
While metamodel itself capturing the main aspects of the automotive system
is usually output of manual design efforts, we also introduce LLM-driven ap-
proach aiming to perform this step automatically as well, incorporating implicit
architecture formalization based on requirements. For this purpose we adopt
locally deployable deepseek-ai/deepseek-llm-7b-chat [2]. Simpler PlantUML no-
tation for metamodel representation is used, as well as iterative approach where
smaller subset of requirements is provided as input in single step. Additionally,
human reviewer has insight into visual representation of the metamodel and can
provide feedback/corrections as input. Once human reviewer is satisfied, model-
to-model transformation will be executed in order to construct Ecore metamodel
from PlantUML, as shown in Fig. 6.
Fig.3: LLM-based iterative metamodeling.
3.2 Regulation-Compliant Scenario Generation
In automotive software development, especially for ADAS and Autonomous
Driving Systems (ADS), producing precise, regulation-compliant test scenar-
ios is essential [16]. Conventional manual methods for interpreting and verifying
compliance with standards such as UN Regulation No. 152 are often inefficient,
error-prone, and expensive. While Large Language Models (LLMs) can help
automate this work, they frequently miss critical numerical details, fail to dis-
tinguish between different test conditions (like laden versus unladen vehicles),
and struggle to process lengthy documents that exceed context window. Supply-
ing the entire regulation to an LLM also consumes excessive tokens, driving up
costs. The aim of this step is extracting test scenarios in textual format from
standards like UN Regulation No. 152, focused on automated emergency braking
systems (AEBS). Our RAG system uses a robust two-stage pipeline, as depicted
in Fig. 4. It applies a SmartChunking approach to preprocess PDF-based stan-
dards by mapping hierarchical paragraph structures, resolving nested references,

GenAI for Automotive Software Development: From Requirements to Wheels 5
and expanding chunks through graph-based traversal, ensuring token spending
efficiency at the same time. Smart Retrieve and Rerank module then performs
query-sensitive retrieval across these enriched chunks, leading to verifiable and
more accurate results than conventional chunking or non-RAG setups.
Fig.4: GenAI-empowered automotive software development workflow.
3.3 Simulation Test Scenario Generation
Using regulation-compliant scenarios derived from UN Regulation No. 152 [15]
and similar sources, this component generates configuration code for a CARLA-
based simulation environment. Separate LLM-driven pipelines are employed to
address the following requirement categories: 1) vehicle definition (including sen-
sor specifications), 2) pre-conditions (such as scene setup, agent positioning, and
weathersettings),and3)post-conditions(includingtelemetrydataandexpected
simulation outcomes). The approach uses GPT-4o and builds upon the method-
ology presented in [3]. The complete workflow is presented in Fig. 5.
Vehicle definition pipeline
Pre-conditions pipeline
Post-conditions pipelineCarlaVehicle config
generatorVehicle config
Simulation
configVehicle
definition
requirements
Pre-conditions
requirements
Post-
conditions
requirementsPost-conditions
config
generatorPre-conditions
config
generatorPre-conditions
config
Post-conditions
configLegend
Executable
Configuration
code (JSON)Natural language
Fig.5: Simulation scenario generation.

6 N. Petrovic et al.
3.4 Target Platform Code Generation
In the last step, C++ code for target testbench platform is also generated (re-
lying on GPT-4o), while its role is complementary to Python simulation code
executed in CARLA, as extension of our work from [9]. While sensing part is
performed within simulation on CARLA server, control (steering, braking, ac-
celeration) is executed on vehicle controller relying on comAPI (alternatively
TC4D or CAN FD API). Events from simulation environment are sent from
Python script which acts as ROS2 publisher, while C++ target platform code
receives them as subscriber. Based on the received events, vehicle commands
are executed. The factors taken into account for code generation are: 1) ex-
periment model - contains the information about both the scenario and vehicle
configuration 2) code templates 3) Vehicle Signal Specification (VSS) [1] catalog
- list of available vehicle signals. Before code generation, corresponding vehicle
signals are mapped based on provided experiment description with respect to
signal catalog, so the list of used signals is established. After that, control logic
is generated, leveraging comAPI invocations based on VSS signals which are
translated to CAN messages for zone ECUs thanks to gateway component.
LLM Code 
GeneratorScenario setup
Vehicle configuration
(Python - Carla)
Behavior 
(C++: ROS2 subscriber
And comAPI publisher)Carla serverSimulation
Vehicle 
Controller
HPC (Linux)
Gateway
Zone ECUs
ROS2 obj
VSS obj
Testbench
comAPI/TC4D 
API/CAN FD API 
templatesVSS catalog
Experiment model
Fig.6: Target platform code generation.
4 Conclusion
Based on the initial results for automated emergency braking, the proposed
approach leveraging GenAI for automotive engineering and re-engineering pro-
cesses automation in order to tackle the cognitive load and increasing SDV sys-
tems complexity, exhibits strong potential for reducing the time needed for in-
novation and testing - from days and hours to order of magnitude of a minute.
Acknowledgments. This research was funded by the Federal Ministry of Research,
Technology and Space of Germany as part of the CeCaS project, FKZ: 16ME0800K.

GenAI for Automotive Software Development: From Requirements to Wheels 7
References
1. COVESA: Vehicle signal specification (vss). COVESA website (2025),
https://covesa.global/vehicle-signal-specification/, accessed: 2025-07-10
2. DeepSeek: deepseek-ai/deepseek-llm-7b-chat. Hugging Face model reposi-
tory (2024), https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat, accessed:
2025-07-11
3. Lebioda, K., Petrovic, N., Pan, F., Zolfaghari, V., Schamschurko, A., Knoll, A.:
Are requirements really all you need? a case study of llm-driven configuration
code generation for automotive simulations. arXiv preprint arXiv:2505.13263 (May
2025), submitted May 19, 2025; DOI: 10.48550/arXiv.2505.13263
4. Li, K.C., Zolfaghari, V., Petrovic, N., Pan, F., Knoll, A.: Optimizing re-
trieval augmented generation for object constraint language. arXiv preprint
arXiv:2505.13129 (May 2025). https://doi.org/10.48550/arXiv.2505.13129,
https://arxiv.org/abs/2505.13129, presented at the First Large Language Models
for Software Engineering Workshop (LLM4SE 2025), part of STAF 2025
5. McKinsey & Company: The case for an end-to-end automotive-
software platform. McKinsey & Company Insight (Jan 2020),
https://www.mckinsey.com/industries/automotive-and-assembly/our-
insights/the-case-for-an-end-to-end-automotive-software-platform
6. n8n: Powerful workflow automation software & tools – n8n. n8n.io (2025),
https://n8n.io/, accessed: 2025-07-09
7. Pan, F., Zolfaghari, V., Wen, L., Petrovic, N., Lin, J., Knoll, A.: Generative ai for
ocl constraint generation: Dataset collection and llm fine-tuning. In: 2024 IEEE
International Symposium on Systems Engineering (ISSE). pp. 1–8. Perugia, Italy
(2024). https://doi.org/10.1109/ISSE63315.2024.10741141
8. Pan, F., Petrovic, N., Zolfaghari, V., Wen, L., Knoll, A.: Llm-enabled instance
model generation. arXiv preprint arXiv:2503.22587 (Mar 2025), accessed: 2025-07-
09
9. Pan, F., Song, Y., Wen, L., Petrovic, N., Lebioda, K., Knoll, A.: Automating
automotive software development: A synergy of generative ai and formal methods.
arXiv preprint arXiv:2505.02500 (May 2025), submitted May 2025
10. Petrovic, N., et al.: Synergy of large language model and model
driven engineering for automated development of centralized vehicu-
lar systems. Technical report, Technical University of Munich (2024),
https://mediatum.ub.tum.de/doc/1738462/1738462.pdf
11. Petrovic, N., Zhang, Y., Maaroufi, M., Chao, K.Y., Mazur, L., Pan, F., Zolfaghari,
V.,Knoll,A.:Multi-modalsummarizationinmodel-basedengineering:Automotive
software development case study. arXiv preprint arXiv:2503.04506 (Mar 2025),
https://arxiv.org/abs/2503.04506, accepted for IntelliSys2025
12. Phatale, A., Kaushik, A.: Generative ai adoption in automotive vehicle technology:
Casestudyofcustomgpt.In:JournalofArtificialIntelligence&CloudComputing,
vol. 3, pp. 1–5 (November 2024). https://doi.org/10.47363/JAICC/2024(3)400
13. Schamschurko, A., Petrovic, N., Knoll, A.C.: RECSIP: RE-
peated Clustering of Scores Improving the Precision (2025).
https://doi.org/10.48550/arXiv.2503.12108, https://arxiv.org/abs/2503.12108,
conference paper accepted for IntelliSys2025
14. Staron, M., Abrahão, S.: Exploring generative ai in automated soft-
ware engineering. In: IEEE Software, vol. 42, pp. 142–145 (2025).
https://doi.org/10.1109/MS.2025.3533754

8 N. Petrovic et al.
15. United Nations Economic Commission for Europe: Un regulation no 152 – uniform
provisions concerning the approval of motor vehicles with regard to the advanced
emergency braking system (aebs) for m1 and n1 vehicles. Publications Office of the
European Union (Oct 2020), uN-ECE Reg. 152; accessed via Publications Office,
DOI not available
16. Zolfaghari, V., Petrovic, N., Pan, F., Lebioda, K., Knoll, A.: Adopting rag for llm-
aided future vehicle design. In: 2024 2nd International Conference on Foundation
and Large Language Models (FLLM). pp. 437–442. Dubai, United Arab Emirates
(2024). https://doi.org/10.1109/FLLM63129.2024.10852467