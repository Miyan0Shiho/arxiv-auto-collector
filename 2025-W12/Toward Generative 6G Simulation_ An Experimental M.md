# Toward Generative 6G Simulation: An Experimental Multi-Agent LLM and ns-3 Integration

**Authors**: Farhad Rezazadeh, Amir Ashtari Gargari, Sandra Lagen, Houbing Song, Dusit Niyato, Lingjia Liu

**Published**: 2025-03-17 17:34:04

**PDF URL**: [http://arxiv.org/pdf/2503.13402v1](http://arxiv.org/pdf/2503.13402v1)

## Abstract
The move toward open Sixth-Generation (6G) networks necessitates a novel
approach to full-stack simulation environments for evaluating complex
technology developments before prototyping and real-world implementation. This
paper introduces an innovative approach\footnote{A lightweight, mock version of
the code is available on GitHub at that combines a multi-agent framework with
the Network Simulator 3 (ns-3) to automate and optimize the generation,
debugging, execution, and analysis of complex 5G network scenarios. Our
framework orchestrates a suite of specialized agents -- namely, the Simulation
Generation Agent, Test Designer Agent, Test Executor Agent, and Result
Interpretation Agent -- using advanced LangChain coordination. The Simulation
Generation Agent employs a structured chain-of-thought (CoT) reasoning process,
leveraging LLMs and retrieval-augmented generation (RAG) to translate natural
language simulation specifications into precise ns-3 scripts. Concurrently, the
Test Designer Agent generates comprehensive automated test suites by
integrating knowledge retrieval techniques with dynamic test case synthesis.
The Test Executor Agent dynamically deploys and runs simulations, managing
dependencies and parsing detailed performance metrics. At the same time, the
Result Interpretation Agent utilizes LLM-driven analysis to extract actionable
insights from the simulation outputs. By integrating external resources such as
library documentation and ns-3 testing frameworks, our experimental approach
can enhance simulation accuracy and adaptability, reducing reliance on
extensive programming expertise. A detailed case study using the ns-3 5G-LENA
module validates the effectiveness of the proposed approach. The code
generation process converges in an average of 1.8 iterations, has a syntax
error rate of 17.0%, a mean response time of 7.3 seconds, and receives a human
evaluation score of 7.5.

## Full Text


<!-- PDF content starts -->

Toward Generative 6G Simulation: An Experimental
Multi-Agent LLM and ns-3 Integration
Farhad Rezazadeh∗, Amir Ashtari Gargari∗, Sandra Lagén∗, Houbing Song†, Dusit Niyato‡, and Lingjia Liu§
∗Centre Tecnológic de Telecomunicacions de Catalunya (CTTC), Barcelona, Spain
†University of Maryland, Baltimore County (UMBC), Baltimore, USA
‡Nanyang Technological University, Singapore
§Virginia Tech, Blacksburg, USA
Contact Email: farhad.rh@ieee.org
Abstract —The move toward open Sixth-Generation (6G)
networks necessitates a novel approach to full-stack simulation
environments for evaluating complex technology developments
before prototyping and real-world implementation. This paper
introduces an innovative approach1that combines a multi-
agent framework with the Network Simulator 3 (ns-3) to
automate and optimize the generation, debugging, execution, and
analysis of complex Fifth-Generation (5G) network scenarios.
Our framework orchestrates a suite of specialized agents—
namely, the Simulation Generation Agent, Test Designer Agent,
Test Executor Agent, and Result Interpretation Agent—using
advanced LangChain coordination. The Simulation Generation
Agent employs a structured chain-of-thought (CoT) reasoning
process, leveraging large language models (LLMs) and retrieval-
augmented generation (RAG) to translate natural language
simulation specifications into precise ns-3 scripts. Concurrently,
the Test Designer Agent generates comprehensive automated
test suites by integrating knowledge retrieval techniques with
dynamic test case synthesis. The Test Executor Agent dynamically
deploys and runs simulations, managing dependencies and
parsing detailed performance metrics. At the same time, the
Result Interpretation Agent utilizes LLM-driven analysis to
extract actionable insights from the simulation outputs. By
integrating external resources such as library documentation
and ns-3 testing frameworks, our experimental approach can
enhance simulation accuracy and adaptability, reducing reliance
on extensive programming expertise. A detailed case study using
the ns-3 5G-LENA module validates the effectiveness of the
proposed approach. The code generation process converges in an
average of 1.8 iterations, has a syntax error rate of 17.0%, a mean
response time of 7.3 seconds, and receives a human evaluation
score of 7.5.
Index Terms —5G/6G, generative simulation, multi-agent LLM,
RAG, chain-of-thought, ns-3
I. I NTRODUCTION
The surge in demand for applications such as extended
reality (XR) and holographic services is driving the rapid
evolution toward 6G networks, which are set to integrate
advanced technologies like terahertz (THz) communication,
AI-driven network management, and non-terrestrial networks
(NTNs) [1], [2]. These innovations require open interfaces
and protocols to ensure seamless interoperability, yet
the heterogeneous and complex nature of 6G demands
1A lightweight, mock version of the code is available on GitHub at https:
//github.com/frezazadeh/LangChain-RAG-Technologysophisticated testing methods—particularly given the scarcity
of full-scale 6G testbeds [3], [4].
A comprehensive, full-stack simulator is essential for
validating novel methodologies across all network layers
by using large-scale network deployments, before actual
standardization, prototyping, and implementation. Tools such
as ns-3 have historically enabled the evaluation of complex
network environments [5], [6]. The ns3 is a powerful and
widely utilized network simulator that plays a critical role in
developing and evaluating networks by modeling and assessing
various approaches proposed by scholars [7]. However, the
simulator’s frequent updates and the inherent complexity of its
C++ codebase can impede rapid prototyping and widespread
adoption.
Recent advancements in language models have opened the
door to automating the generation, debugging, and execution
of simulation code from natural language descriptions [8].
However, these models face challenges with complex tasks
due to their limited contextual integration and dependency
on pre-trained data. Hybrid methodologies such as retrieval-
augmented generation (RAG) have emerged to address
these limitations, enhancing generative models with external,
domain-specific knowledge [9]. Using LLMs in the simulation
process can help reduce the need for manual coding. This
change can speed up innovation and make it easier for
researchers working in dynamic 6G network environments.
A. Contributions
This paper presents a novel case study that integrates Multi-
Agent LLMs with the ns-3 simulator to create a generative
simulation framework explicitly tailored for 5G/6G networks.
Our contributions are summarized as follows:
1)Multi-Agent Architecture: We introduce a highly
coordinated multi-agent system where specialized agents
collaboratively manage the simulation lifecycle. The
Simulation Generation Agent (Agent#1) transforms
natural language simulation requirements into executable
code using state-of-the-art LLMs (via the ChatOpenAI
interface) and prompt templates. In parallel, the Test
Designer Agent (Agent#2) constructs targeted test cases
by leveraging retrieval-augmented techniques with aarXiv:2503.13402v1  [cs.NI]  17 Mar 2025

Fig. 1: The graphical user interface of application.
Pinecone vector store and OpenAIEmbeddings to ensure
simulation accuracy.
2)Seamless Integration of LLMs with Simulation Tools:
Our framework bridges advanced LLM capabilities
with domain-specific simulation tools. It dynamically
orchestrates various APIs and tools such as the
CppSubprocessTool for ns-3 C++ execution and
thePythonREPLTool for Python debugging-allowing
for automated code generation, execution, and iterative
refinement.
3)Dynamic API and Model Optimization: We implement
a flexible toggle mechanism within the Streamlit interface
that enables dynamic selection between optimized
model variants (e.g., gpt-4o-mini ) for cost-efficient
operations. This integration ensures that our system
maintains acceptable accuracy.
4)Case Study on 5G Network Simulation: We validate
our approach with a detailed case study focusing on
5G network scenarios. The case study demonstrates
how the interplay between the Simulation Generation,
Test Designer, Test Executor (Agent#3), and Result
Interpretation Agents (Agent#4) leads to faster simulation
development, robust debugging, and refined simulation
outputs through iterative feedback loops.
Overall, our work lays a robust foundation for integrating
LLM-driven multi-agent systems with traditional simulation
platforms, advancing the state-of-the-art automated network
simulation and providing valuable insights for next-generation
6G network design and analysis.
II. M ETHODOLOGY
Figure 1 shows the graphical user interface of the proposed
framework. This application integrates multiple advanced tools
and models within an intuitive and user-friendly environment
built using Streamlit. Furthermore, it adopts a modular design
approach by utilizing LangChain [10]and LangGraph [11].
The proposed framework implements a technically
sophisticated multi-agent system that leverages language
models, retrieval APIs, and custom tool integrations to
automate the entire process of network simulation generation,
debugging, execution, and analysis in ns-3 environments.
Upon receiving simulation requirements in natural language,a central coordinator agent orchestrates the interaction
among several specialized agents using LangChain. The
Simulation Generation Agent (Agent#1) utilizes OpenAI’s
LLMs (accessed via the ChatOpenAI interface) with detailed
prompt templates to generate simulation code. In tandem,
the Test Designer Agent (Agent#2) constructs targeted
test cases by integrating rule-based logic with retrieval-
augmented strategies—utilizing a Pinecone vector store and
OpenAIEmbeddings—to verify the simulation’s accuracy and
reliability. The Test Executor Agent (Agent#3) subsequently
runs the simulation in the ns-3 environment by interfacing with
custom execution tools, such as the CppSubprocessTool
for C++ code and the PythonREPLTool for Python code. A
dynamic toggle in the Streamlit sidebar allows for optimized
model selection (e.g., gpt-4o-mini ), demonstrating
flexible API integration across agents. Finally, the Result
Interpretation Agent (Agent #4) analyzes the simulation
outputs by interpreting performance metrics and error logs
according to detailed prompt instructions. This actionable
feedback is then replied back to the Simulation Generation
Agent, which triggers an iterative refinement process. Through
this robust integration of API key validations, environment
configurations, and multi-tool execution strategies, the
framework ensures that the agents collaboratively produce a
fully validated and reliable simulation.
Figure 2 shows a comprehensive view of the multi-
agent, LLM-driven ns-3 simulation framework. The end user
(represented as a human in the loop) provides simulation
requirements and feedback to the Agent Orchestration
layer, which coordinates specialized agents for simulation
generation, test design, execution, and result interpretation.
These agents leverage an LLM and external tools (such as ns-3
and documentation) in a cyclical feedback loop to iteratively
refine and analyze network simulations.
A. Multi-Agent Workflow for Network Simulation Generation
1) Simulation Generation Agent (Structured Simulation
Script Synthesis through Iterative Reasoning): The role of
the Simulation Generation Agent (Agent#1) is pivotal in
converting high-level simulation specifications into executable
ns-3 scripts. This is accomplished by leveraging a structured
CoT [12] reasoning process empowered by LLMs via the
ChatOpenAI interface in LangChain. Initially, the agent
processes natural language input using advanced Natural
language processing (NLP) techniques to extract critical
simulation parameters, such as network protocols, mobility
models, and bandwidth allocation. Based on these parameters,
the agent selects suitable ns-3 libraries and modules
(e.g., LTE, Wi-Fi, and 5G-LENA2) and identifies optimal
simulation models that consider factors like the number of
user devices, application demands, environmental conditions,
and propagation models (e.g., 3GPP 38.901, COST 231,
or ITU-R). In the subsequent script construction phase,
the agent automatically generates the simulation script by
2https://5g-lena.cttc.es/

Fig. 2: Overview of the multi-agent LLM-based ns-3 simulation framework. The streamlined workflow is shown in Figure 4.
initializing nodes, configuring network devices, setting up
communication channels, and assigning applications. During
this process, it instantiates complex ns-3 classes such as
SpectrumWifiPhy for Wi-Fi frequency modeling and NrHelper
for mmWave-based networks and leverages integrated tool
APIs (e.g., CppSubprocessTool for C++ code execution
andPythonREPLTool for Python code debugging) to
ensure that each component is rigorously validated. After the
initial script has been generated, the Test Designer Agent
(Agent#2) is activated to conduct static analysis and syntax
validation. This process helps identify missing dependencies,
deprecated API usage, and parameter mismatches. This
iterative refinement process ensures that the generated script
adheres to ns-3’s coding standards and simulation paradigms.
2) Test Designer Agent (Rigorous Validation through
Automated Test Suite Generation): The Test Designer Agent
(Agent#2) validates the generated simulation scripts by
automatically constructing a comprehensive suite of test cases.
Leveraging LangChain for knowledge retrieval and LLMs for
test script generation, this agent creates both primary and
edge test cases. Primary test cases verify standard operations
such as the successful attachment of user equipments (UEs)
to base stations (e.g., eNodeB/gNodeB), the correct data flow
between nodes, and adherence to predefined quality of service
(QoS) parameters. Edge test cases, on the other hand, guide
the simulation to stress conditions such as high mobility
speeds, extreme interference, or increased network load to
assess network capacity and reliability. Utilizing retrieval-
augmented strategies through a Pinecone vector store and
OpenAIEmbeddings, the Test Designer Agent dynamically
incorporates relevant ns-3 testing paradigms and interfaces
with ns-3’s testing frameworks (or custom validation code)
to ensure comprehensive coverage.
3) Test Executor Agent (Dynamic Simulation Execution
and Intelligent Feedback Loop): The Test Executor Agent
(Agent#3) acts as the operational backbone by executing
validated simulation scripts within the ns-3 environment.
This agent manages the deployment of simulation scripts
across local, virtualized, or cloud-based environments. It
captures all dependencies and environment variables to
ensure compatibility with the current version of ns-3. During
simulation execution, the agent collects diverse output data—
including trace files, logs, and performance metrics parsing
.pcap files, FlowMonitor outputs, and custom trace sources to
extract key performance indicators (KPIs) such as throughput,latency, packet loss, and jitter. Furthermore, advanced NLP
techniques and LLM-based analysis are employed to interpret
error logs and warnings (e.g., segmentation faults, assertion
failures, or unexpected behaviors), enabling the agent to
generate detailed, actionable feedback. This feedback is
communicated back to the Simulation Generation Agent,
establishing an intelligent, iterative feedback loop that drives
continuous improvement of the simulation scripts.
4) Result Interpretation Agent ( Analyzing Simulation
Outcomes and Offering Insights): The Result Interpretation
Agent (Agent#4) focuses on post-execution analysis by
interpreting the outputs of ns-3 simulations. After a simulation
run, the agent processes performance metrics and log
data to provide a detailed network behavior analysis.
The agent correlates observed performance metrics with
underlying network conditions by utilizing sophisticated
LLMs with tailored prompt instructions. For example, it
may be determined that an increase in end-to-end delay
is attributable to a high packet loss rate at the physical
layer, potentially caused by wireless channel interference
or suboptimal modulation schemes. By delivering clear,
actionable insights, the Result Interpretation Agent guides
further refinements, ensuring that the simulation meets the
desired performance criteria and adheres to best practices in
network simulation.
Fig. 3: The sample ns-3 generated code.
B. Case Study
This section discusses a specific use case that exemplifies
the framework’s capabilities in simulating a 5G new radio
(NR) environment. The primary objective of this study is
to investigate how the proposed framework can facilitate the
execution of complex simulation scenarios.

Fig. 4: Sequence diagram illustrating the streamlined workflow for simulating a 5G NR environment.
1) Simulation Scenario: The scenario under examination
involves simulating a dense urban microcell (UMi)
environment characterized by the presence of 100 UEs
and one gNB operating at 28 GHz with a bandwidth of
200 MHz. The main focus of this investigation is the
implementation of scheduling mechanisms to effectively
balance the network load across multiple UEs, thereby
ensuring an optimal QoS throughout the network. Figure 3
demonstrates the setup and configuration of the simulation
scenario, including node creation, channel model selection,
and Internet stack installation. The workflow for this use case
is outlined in the following steps, which detail the coordinated
actions of each agent within the framework.
2) Simulation Requirements and Input Parsing (Agent#1):
The user initiates the process by providing a natural language
input: "Simulate a 5G New Radio environment with 100
UEs and one gNB at 28 GHz with 200 MHz bandwidth.
Implement TCP communication and enable traffic steering
using beamforming". The Agent#1 receives this input and
begins by parsing it using NLP techniques to extract the
relevant parameters: frequency, bandwidth, number of UEs,
and the desired communication protocol (TCP). By leveraging
LangChain, the agent selects appropriate network models
from ns-3, such as the 5G-LENA module and 3GPP-based
UMi channel models. Using a structured CoT reasoning
approach, the agent then synthesizes the necessary ns-3 script.
It configures network nodes, assigns UEs, defines the TCP
application, and configures the beamforming method. For
example, the agent chooses the NrHelper class from ns-3
for mmWave channel modeling and BulkSendHelper for
setting up TCP traffic across UEs.
3) Test Case Generation and Script Validation (Agent#2):
After the script is generated, Agent#2 ensures the script’s
accuracy by creating automated test cases. These tests evaluate
the core functionalities of the simulation, such as the correct
association of UEs to the gNB, proper functioning of the
TCP protocol, and the efficiency of traffic steering using
beamforming. Edge cases, such as extreme UE mobility or
high interference conditions, are introduced by the agent to
test the robustness of the simulation. Scalability tests are
also performed, incrementally increasing the number of UEsto assess network performance under heavy traffic loads.
The tests automatically interact with ns-3’s built-in testing
framework to validate these conditions.
4) Simulation Execution and Feedback (Agent#3): With
the validation complete, the Test Executor Agent (Agent#3)
executes the simulation script within the ns-3 environment,
either locally or in a cloud-based setup. The agent ensures that
all necessary dependencies and configurations are in place,
such as version compatibility with ns-3 modules and the
required libraries (e.g., 5G-LENA, TCP/IP). The agent collects
performance data during the simulation, such as throughput,
latency, and packet loss, by analyzing trace files and logs.
It uses FlowMonitor and custom trace sources within ns-
3 to gather KPIs. The agent provides detailed feedback if
any simulation errors occur, such as a segmentation fault or
assertion failure. This information is then passed back to the
Simulation Generation Agent for iterative refinement of the
simulation script, ensuring that the final output is bug-free
and meets all performance standards.
5) Result Interpretation and Insights (Agent#4): Once the
simulation is successfully executed, the Result Interpretation
Agent (Agent#4) analyzes the resulting performance metrics.
For this use case, the agent observes that the traffic steering
mechanism balances the load efficiently across the UEs,
resulting in improved throughput and reduced latency for most
UEs. However, the agent also indicates a slight increase in
end-to-end delay during periods of heavy traffic, suggesting
potential congestion at the gNB. The agent interprets the
results and provides actionable feedback, such as: "The
increase in delay correlates with a higher packet loss rate
during high interference periods, indicating that adjusting the
beamforming method or increasing the number of gNBs could
alleviate this issue." Figure 4 demonstrates the diagram of
interactions between the user, input parsing, test validation,
simulation execution, and result interpretation agents, along
with the ns-3 environment, highlighting the process from
natural language input to actionable feedback.
III. E XPERIMENTAL RESULTS
In our experiments, the simulation platform was deployed
using a combination of Streamlit for the interface, Pinecone

for vector storage, and LangChain and LangGraph integrated
with OpenAI models for processing natural language queries.
The system was configured to handle multiple queries,
including direct simulation commands for Python and C++
environments. Furthermore, the successful initialization of
the Pinecone index and the efficient retrieval of relevant
information via the question-answering (QA) chain validate
the design of our simulation framework. Sample output logs
confirmed the correct execution of simulation tasks and
indicated system performance. For instance, the output "Index
’ap-v0’ created" demonstrates that the vector storage setup
was completed without errors. In addition, integrating the ns-3
simulation with custom agents allowed for seamless handling
of code generation and execution, further highlighting the
system’s flexibility.
Table I presents the experimental results for the document
ingestion process, where the ingestion time increases with the
number of documents processed. The data suggests a nearly
linear relationship between the number of documents and
the processing time, indicating predictable scalability for the
ingestion mechanism.
Table II compares the execution times for Python and C++
simulation runs. Although both methods ultimately execute
the ns-3 simulation, the Python method calls ns-3 using a
subprocess (wrapping the ns-3 execution within a Python
script), while the C++ method invokes ns-3 directly. The Base
ns-3 Time (s) represents the inherent execution time of the
ns-3 simulation engine when running the simulation code.
The overhead is defined as the extra time incurred during the
invocation process beyond the base simulation time. For the
Python invocation, this overhead includes the time required for
setting up and managing a subprocess, handling inter-process
communications, and any additional interpretation or bridging
tasks between Python and the native C++ environment. In
contrast, the C++ invocation method directly executes the
simulation code, thus introducing significantly less overhead.
Table II shows that i) The base simulation time provided
by ns-3 remains constant (5.0 seconds) and ii) The C++
invocation incurs a smaller overhead (1 second) compared to
the Python invocation (3 seconds), leading to slightly lower
total execution time. This comparison helps clarify that while
the underlying simulation engine is the same, the additional
overhead introduced by the invocation method can impact the
overall performance.
Table III presents metrics that describe the responsiveness
of the system across different types of queries. The Average
Response Time (s) metric represents the mean time taken
for the system to generate a response for each query
type, averaged over multiple runs. It indicates the system’s
efficiency in processing various requests—from regular
queries to more complex operations such as code generation
and debugging. The Standard Deviation (s) measures the
variability or consistency of the response times across multiple
trials. A lower standard deviation indicates that the response
times are consistently close to the average value, while a
higher standard deviation suggests greater fluctuations in thetime required to process the queries.
TABLE I: Document Ingestion Performance
Number of Documents Ingestion Time (seconds)
10 1.1
20 2.0
30 2.9
40 3.8
50 4.6
60 5.4
70 6.2
80 7.1
90 7.9
100 8.7
The performance metrics summarized in Table IV provide a
comprehensive evaluation of the LLM-driven code generation
process for the simulation scenario. The Average Iterations
metric indicates the mean number of iterative refinement
cycles required by the LLM to generate syntactically correct
and functionally valid ns-3 simulation code. A lower average
iteration count implies that the LLM’s initial output is close
to the desired final form, necessitating minimal corrections.
The Syntax Error Rate (%) is calculated as the percentage
of syntactical errors present in the generated code relative to
the total lines or segments of code produced. A low syntax
error rate implies that the LLM is effective at generating
code that adheres to ns-3’s syntax and coding standards. The
Human Evaluation Score as a qualitative metric, typically
assessed on a scale from 1 to 10, reflects expert judgments
regarding the clarity, correctness, and overall quality of
the generated code [13]. A higher score corresponds to
better code quality and usability, as determined by human
evaluators. Furthermore, the Pass@k [14] metric is used to
quantitatively evaluate code generation models by measuring
the probability that at least one of the top- kgenerated code
samples successfully passes all test cases. This metric provides
an empirical assessment of a model’s ability to produce correct
and functional code within a given number of attempts.
The Average Iterations metric of 1.8 indicates that, on
average, the LLM required fewer than two refinement cycles
to produce syntactically correct and functionally accurate
ns-3 simulation code, demonstrating that the initial outputs
were largely accurate with minimal adjustments needed. The
Syntax Error Rate of 17.0% confirms a high degree of
syntactic correctness and reduces the need for extensive
manual debugging. In addition, an Average Response Time (in
five runs— k= 5) of 7.3 seconds highlights the computational
efficiency of the system, ensuring that iterative refinements are
performed rapidly. Finally, Human Evaluation Score of 7.5 and
Pass Rate of 0.72 confirm that the generated code meets high
standards.
IV. O PEN ISSUES AND FUTURE STUDY
A. Automated Bug-Free ns-3 Simulations Using LLMs
LLMs face challenges when generating ns-3 simulation
code due to their hierarchical, C++-based structure and
the diversity of protocols (transport, routing, MAC, PHY)

TABLE II: Comparison of ns-3 Simulation Execution Performance for Python and C++ Invocation
Invocation Method Base ns-3 Time (s) Overhead (s) Total Time (s)
Python Invocation (via subprocess) 5.0 3.0 8.0
C++ Invocation (direct) 5.0 1.0 6.0
TABLE III: Query Response Time Performance
Query Type Avg. Response Time (s) Std. Dev. (s)
Regular Query 1.2 0.3
ns-3 C++ Generation 2.7 0.4
ns-3 Python Generation 3.2 0.5
ns3 Execution & Debugging & Interpretation 4.5 0.6
TABLE IV: LLM-Driven Code Generation Performance Metrics
Simulation Scenario Avg. Iterations Syntax Error Rate (%) Avg. Response Time (s) Human Eval Score Pass Rate
Defined Scenario 1.8 17.0 7.3 7.5 0.72
maintained by various institutions. Although LLMs may
produce syntactically similar code, they often lack a deep
understanding of the networking principles needed for correct
logic, requiring expert interaction to improve precision.
B. Architectural Enhancements for Protocol-Specific Code
Generation
Current transformer-based LLMs, tailored for natural
language and general-purpose coding, struggle with network
protocols’ highly structured and syntax-driven nature.
Incorporating concepts from compiler theory, such as
intermediate representations and abstract syntax trees, could
better align these models with the requirements of ns-3 code
generation.
C. Enhancing Support for Domain-Specific Languages in ns-3
Simulations
LLMs are typically trained on popular programming
languages, leaving domain-specific languages (DSLs) and
low-level constructs, which are crucial for ns-3 simulations,
less represented. Enhancing DSL support would enable more
precise automation of ns-3 code generation.
D. Continuous Learning for Adapting to Evolving Network
Standards
Static LLMs risk becoming outdated with continuous
evolution of network standards (e.g., 3GPP, IEEE) and
frequent updates to ns-3 modules. Implementing continuous
learning mechanisms ensures that generated code remains
current and applicable to the latest network technologies.
V. C ONCLUSION
The paper presents a multi-agent framework that integrates
LLMs with the ns-3 network simulator to automate the
creation, validation, execution, and analysis of 5G/6G
simulation scenarios. By dividing the process into specialized
agents—for generating simulation scripts, designing tests,
executing simulations, and interpreting results—the framework
transforms natural language inputs into fully validated
simulation scripts. Experimental validation in a 5G NR
scenario demonstrates its potential to reduce complexityand speed up simulation workflows. However, challenges
such as ensuring bug-free code generation, scaling full-
stack simulations, and adapting to evolving network standards
remain, paving the way for future research.
REFERENCES
[1] M. Polese, J. M. Jornet, T. Melodia, and M. Zorzi, “Toward End-to-End,
Full-Stack 6G Terahertz Networks,” IEEE Communications Magazine ,
vol. 58, no. 11, pp. 48–54, 2020.
[2] A. A. Gargari, A. Ortiz, M. Pagin, W. de Sombre, M. Zorzi,
and A. Asadi, “Risk-Averse Learning for Reliable mmWave Self-
Backhauling,” IEEE/ACM Transactions on Networking , vol. 32, no. 6,
pp. 4989–5003, 2024.
[3] H. Poddar, S. Ju, D. Shakya, and T. S. Rappaport, “A Tutorial on
NYUSIM: Sub-Terahertz and Millimeter-Wave Channel Simulator for
5G, 6G, and Beyond,” IEEE Communications Surveys & Tutorials ,
vol. 26, no. 2, pp. 824–857, 2024.
[4] A. A. Gargari, A. Ortiz et al. , “Safehaul: Risk-Averse Learning for
Reliable mmWave Self-Backhauling in 6G Networks,” IEEE INFOCOM
2023 - IEEE Conference on Computer Communications , pp. 1–10, 2023.
[5] F. Wilhelmi, M. Carrascosa, C. Cano, A. Jonsson, V . Ram, and
B. Bellalta, “Usage of Network Simulators in Machine-Learning-
Assisted 5G/6G Networks,” IEEE Wireless Communications , vol. 28,
no. 1, pp. 160–166, 2021.
[6] A. Zubow, C. Laskos, and F. Dressler, “Toward the Simulation of
WiFi Fine Time measurements in NS3 Network Simulator,” Computer
Communications , vol. 210, pp. 35–44, 2023.
[7] S. Szott, K. Kosek-Szott, P. Gawlowicz, J. T. Gomez, B. Bellalta,
A. Zubow, and F. Dressler, “Wi-Fi Meets ML: A Survey on
Improving IEEE 802.11 Performance With Machine Learning,” IEEE
Communications Surveys & Tutorials , vol. 24, no. 3, pp. 1843–1893,
2022.
[8] B. Roziere, J. Gehring et al. , “Code Llama: Open Foundation Models
for Code,” arXiv preprint arXiv:2308.12950 , 2024.
[9] Y . Ding, W. Fan et al. , “Survey on RAG Meeting LLMs:
Towards Retrieval-Augmented Large Language Models,” arXiv preprint
arXiv:2405.06211 , 2024.
[10] H. Koziolek et al. , “LLM-based and Retrieval-Augmented Control
Code Generation,” 2024 IEEE/ACM International Workshop on Large
Language Models for Code (LLM4Code) , 2024.
[11] J. Wang and Z. Duan, “Empirical Research on Utilizing LLM-based
Agents for Automated Bug Fixing via LangGraph,” arXiv preprint
arXiv:2502.18465 , 2025.
[12] J. Wei et al. , “Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models,” In Advances in Neural Information Processing
Systems , 2022.
[13] D. Ghosh Paul, H. Zhu, and I. Bayley, “Benchmarks and Metrics
for Evaluations of Code Generation: A Critical Review,” 2024 IEEE
International Conference on Artificial Intelligence Testing (AITest) ,
2024.
[14] M. Chen et al. , “Evaluating Large Language Models Trained on Code,”
arXiv preprint arXiv:2107.03374 , 2021.