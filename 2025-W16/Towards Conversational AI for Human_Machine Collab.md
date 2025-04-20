# Towards Conversational AI for Human-Machine Collaborative MLOps

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Anargyros Tsadimas, Dimosthenis Kyriazis

**Published**: 2025-04-16 20:28:50

**PDF URL**: [http://arxiv.org/pdf/2504.12477v1](http://arxiv.org/pdf/2504.12477v1)

## Abstract
This paper presents a Large Language Model (LLM) based conversational agent
system designed to enhance human-machine collaboration in Machine Learning
Operations (MLOps). We introduce the Swarm Agent, an extensible architecture
that integrates specialized agents to create and manage ML workflows through
natural language interactions. The system leverages a hierarchical, modular
design incorporating a KubeFlow Pipelines (KFP) Agent for ML pipeline
orchestration, a MinIO Agent for data management, and a Retrieval-Augmented
Generation (RAG) Agent for domain-specific knowledge integration. Through
iterative reasoning loops and context-aware processing, the system enables
users with varying technical backgrounds to discover, execute, and monitor ML
pipelines; manage datasets and artifacts; and access relevant documentation,
all via intuitive conversational interfaces. Our approach addresses the
accessibility gap in complex MLOps platforms like Kubeflow, making advanced ML
tools broadly accessible while maintaining the flexibility to extend to other
platforms. The paper describes the architecture, implementation details, and
demonstrates how this conversational MLOps assistant reduces complexity and
lowers barriers to entry for users across diverse technical skill levels.

## Full Text


<!-- PDF content starts -->

Towards Conversational AI for Human-Machine
Collaborative MLOps
1stGeorge Fatouros
Innov-Acts Ltd
Nicosia, Cyprus
Dept. of Digital Systems
University of Piraeus
Piraeus, Greece2ndGeorgios Makridis
Dept. of Digital Systems
University of Piraeus
Piraeus, Greece3rdGeorge Kousiouris
Dept. of Informatics and Telematics
Harokopio University
Athens, Greece
4thJohn Soldatos
Innov-Acts Ltd
Nicosia, Cyprus5thAnargyros Tsadimas
Dept. of Informatics and Telematics
Harokopio University
Athens, Greece6thDimosthenis Kyriazis
Dept. of Digital Systems
University of Piraeus
Piraeus, Greece
Abstract —This paper presents a Large Language Model
(LLM) based conversational agent system designed to enhance
human-machine collaboration in Machine Learning Operations
(MLOps). We introduce the Swarm Agent, an extensible archi-
tecture that integrates specialized agents to create and manage
ML workflows through natural language interactions. The system
leverages a hierarchical, modular design incorporating a Kube-
Flow Pipelines (KFP) Agent for ML pipeline orchestration, a
MinIO Agent for data management, and a Retrieval-Augmented
Generation (RAG) Agent for domain-specific knowledge inte-
gration. Through iterative reasoning loops and context-aware
processing, the system enables users with varying technical back-
grounds to discover, execute, and monitor ML pipelines; manage
datasets and artifacts; and access relevant documentation, all via
intuitive conversational interfaces. Our approach addresses the
accessibility gap in complex MLOps platforms like Kubeflow,
making advanced ML tools broadly accessible while maintaining
the flexibility to extend to other platforms. The paper describes
the architecture, implementation details, and demonstrates how
this conversational MLOps assistant reduces complexity and
lowers barriers to entry for users across diverse technical skill
levels.
Index Terms —Large Language Models, Agents, Artificial In-
telligence, Conversational AI, KubeFlow, MLOps
I. I NTRODUCTION
In today’s rapidly evolving landscape of artificial intelli-
gence (AI) and machine learning (ML), the synergy between
human expertise and AI capabilities is increasingly crucial
[1]. By combining human understanding and creativity with
AI’s computational power and efficiency, organizations can
drive more effective decision-making and innovative solutions
across a range of domains, including healthcare, finance, and
scientific research [2]–[4].
As ML workflows become more complex, MLOps platforms
play a critical role in streamlining the development, deploy-
ment, and management of these workflows [5]. Kubeflow1is
a leading open-source solution for orchestrating ML pipelines
1https://www.kubeflow.org/on Kubernetes [6]. However, its feature set can introduce
complexities that challenge data scientists and users without
extensive experience on systems and devops processes. This
highlights the broader need to make advanced ML platforms
more accessible, ensuring that diverse user groups can take
advantage of their functionalities.
To address these challenges, we present the Swarm Agent , a
Large Language Model (LLM) based conversational assistant
with access to tools and services to manage ML workflows
and data operations required in the ML life cycle. Our im-
plementation integrates with Kubeflow using tools such as
the Kubeflow Pipelines (KFP) client, MinIO client, and a
vector database, enabling ML pipeline creation and execution,
data operations, and context-aware guidance from natural
language querying. The design of our assistant is extensible
by integrating additional tools or agents and can be adapted
to other MLOps platforms, offering a flexible approach to
simplifying and expanding access to advanced ML operations
beyond a single technology stack.
This paper presents the design and implementation of our
LLM-powered chat assistant, highlighting its role in enhancing
human–AI collaboration in ML pipeline management. We
describe the system’s architecture and illustrate how each
component contributes to a more intuitive user experience. We
also present indicative outcomes of the presented system which
demonstrate how we reduce complexity for diverse technical
skill levels, ultimately lowering barriers to entry for advanced
ML tools and driving broader innovation in the field.
The structure of this paper is organized as follows: Sec-
tion II reviews related work on Conversational Agents (CAs),
MLOps platforms and applications of LLMs agents. Section III
presents the architecture of the proposed agentic system along
followed by Section IV discussing the implementation details
of the system. Section V provides real-world use cases and
indicative responses from the Swarm Agent. Section VI con-
cludes the paper with key insights and our plans for futurearXiv:2504.12477v1  [cs.AI]  16 Apr 2025

work.
II. R ELATED WORKS
A. Conversational Agents and Natural Language Interfaces
The field of CAs has experienced rapid evolution, transi-
tioning from early rule-based systems to contemporary LLM-
driven applications such as ChatGPT and Perplexity. These
advancements facilitate more natural and context-aware in-
teractions, significantly broadening the scope of chatbot ap-
plications to include complex technical systems [7]. Despite
the extensive internal knowledge embedded within LLMs,
advanced agent-based applications leverage external tools, in-
cluding online search engines, code interpreters, and function-
calling mechanisms [8] to enhance their utility. Function
calling, in particular, enables programmatic retrieval of data
from external services via Application Programming Interfaces
(APIs), allowing on-demand access to information relevant to
user queries and triggering corresponding actions [9]. One
prominent example of integrating external knowledge into
LLM-based agents is Retrieval-Augmented Generation (RAG)
which utilizes external knowledge bases to store information
that may not be present in the training data of an LLM [10].
This method grounds LLM responses, mitigates hallucinations,
and allows accurate resolution of user queries using domain-
specific, proprietary, or dynamically updated data [11]. These
developments form the foundational techniques inspiring our
approach to managing ML workflows through natural language
interactions.
B. MLOps Platforms and Pipeline Orchestration
The advent of MLOps has transformed the deployment and
management of ML workflows. Recent surveys and technical
reports detail the evolution of MLOps frameworks, identi-
fying both their transformative potential and the operational
complexities they introduce [12]. Kubeflow has emerged as
the leading platform for orchestrating ML pipelines on Ku-
bernetes, enabling end-to-end machine learning workflows
[13]. However, its extensive feature set can pose significant
challenges for users without a strong technical background,
especially in the area of systems and software management.
For example, industry experts who have valuable business or
domain knowledge often struggle with the technical complex-
ities of MLOps platforms, while data scientists may lack the
DevOps expertise needed to efficiently deploy and monitor
models in production environments [14], [15]. This skill gap
is particularly evident when industrial specialists attempt to
integrate AI solutions into existing workflows but encounter
barriers due to the difficulty of handling such technologies
[16].
Beyond Kubeflow, other MLOps frameworks like Apache
Airflow [17], Dagster, and MLflow [18] are increasingly used
to orchestrate ML workflows. While our approach is initially
integrated with Kubeflow, it is modular and designed to extend
to any platform offering APIs for managing experiments, runs,
and logging. MLflow, for example, offers robust experimenttracking and model management APIs; Dagster supports de-
tailed logging and monitoring; and Airflow provides a REST
API for workflow orchestration. By leveraging these interfaces,
our system can enable conversational management of ML
workflows across diverse MLOps environments.
C. Integration of LLMs in Operational Tools
The integration of agentic AI applications into coding and
DevOps has led to the development of tools that enhance
software development, productivity and operational workflows
[19]. GitHub Copilot [20], Cursor AI [21] and Devin AI
[22], are prominent LLM-powered coding tools that assist
developers by suggesting whole lines or entire functions
within their editor, streamlining the coding process. CodeGPT
offers AI agents tailored for software development teams,
providing features such as AI coding assistance and automated
code reviews, thereby enhancing productivity through deep
codebase understanding [23]. These applications exemplify
the transformative impact of agentic AI in coding and De-
vOps, aligning with our proposed system’s goal of integrating
conversational AI agents into MLOps platforms to facilitate
streamlined workflow management.
To the best of our knowledge, no existing system offers
a conversational, agentic interface that seamlessly integrates
with MLOps platforms like Kubeflow through modular, tool-
specific agents that enable ML workflow management through
interactive, LLM-driven planning and execution. This fills a
critical gap in making complex MLOps environments acces-
sible and intuitive for users of varying technical backgrounds.
III. S YSTEM ARCHITECTURE
The proposed system, Swarm Agent, facilitates agentic
interaction between users and complex MLOps environments,
with a particular focus on Kubeflow. The architecture leverages
a modular, extensible design that empowers users to manage
data, pipelines, and domain-specific knowledge through natu-
ral language leveraging the context understanding of LLMs.
The architecture of the Swarm Agent is depicted in Fig. 1
with the following subsections analyzing the main architectural
components.
A. Chat UI and Session Management
The user interface (Chat UI) serves as the primary human-
computer interaction (HCI) endpoint in the system, leveraging
asynchronous communication to capture user input and render
agent-generated responses. The system implements a session-
based architecture to maintain conversational context and user
state throughout interactions. The Session Manager encapsu-
lates this functionality by maintaining a persistent message
history for each user session, tracking thread identifiers, and
preserving user authentication context across interactions. This
stateful approach enables context-aware reasoning and person-
alized responses tailored to the user’s ongoing conversation
flow, creating a continuous and coherent interaction experi-
ence.

Fig. 1. Architecture of the Swarm Agent system for conversational MLOps.
B. Swarm Agent Core Architecture
The Swarm Agent serves as a central LLM-powered
controller, enabling intelligent coordination across domain-
specific agents through a combination of intent recognition,
contextual memory, and dynamic orchestration. The architec-
ture comprises three main components: an Intent Recognition
System that analyzes user queries to infer the underlying goals
and extract relevant parameters; a Contextual Memory Module
that maintains a structured representation of the conversation
history, enabling coherent multi-turn interactions and refer-
ence resolution via a message history structure; and a Task
Dispatcher that activates and communicates with downstream
agents through a function-calling mechanism.
The system implements hierarchical orchestration by pars-
ing streaming LLM outputs in real-time to detect tool in-
vocation intents. These tool calls are dynamically routed to
specialized agents using a tool mapper that matches descrip-
tions from the LLM to their corresponding implementation
code. Once agents return their outputs, the Swarm Agent
re-incorporates them into the conversation context, enabling
iterative refinement of the response.
This architecture supports iterative reasoning loops, wherethe agent responses are not immediately surfaced to the user
but are first contextualized as function call results within the
conversation history. The enriched context is used to prompt
the LLM again, allowing for recursive reasoning and synthesis
across multiple cycles. This mechanism enhances the system’s
ability to decompose complex queries into subtasks, handle
failures gracefully (e.g., by retrying with adjusted parameters
or switching tools), and integrate intermediate outputs into a
coherent and high-quality final response.
By combining real-time responsiveness with contextual con-
tinuity and recursive processing, the Swarm Agent enables
intelligent behavior that goes beyond the capabilities of any
individual component. This is particularly impactful in MLOps
and technical support contexts, where it must fluidly combine
code, system-level reasoning, and natural language explana-
tion.
C. KFP Agent
The KFP Agent (Fig. 2) serves as the primary interface
between the conversational system and Kubeflow enabling
its operation with natural language. This agent provides a
complete set of functionalities that encapsulate the lifecycle
of ML pipeline management through specialized functions.

Fig. 2. Internal Architecture of the KubeFlow Pipelines (KFP) Agent.
1) Pipeline Discovery and Metadata Management:
The agent implements search capabilities through the
get_pipelines function, allowing users to discover
available pipelines using flexible search terms, names-
pace filtering, and pagination. For detailed inspection, the
agent provides access to pipeline metadata through the
get_pipeline_details function, which retrieves com-
prehensive information including descriptions, creation times-
tamps, versioning information, and technical specifications.
Theget_pipeline_version_details function addi-
tionally extracts pipeline components and parameters from the
pipeline specification, offering deeper insights into workflow
structure. This detailed metadata extraction includes compo-
nent definitions, input parameters with their default values
and types, enabling users to understand the capabilities and
requirements of a pipeline before execution.
2) Execution Management and Monitoring: The KFP
Agent facilitates pipeline execution through dedicated func-
tions including create_experiment for setting up exper-
iment contexts and run_pipeline for initiating workflow
execution. The execution function accepts multiple parame-
ters including experiment IDs, job names, parameter map-
pings, and pipeline identifiers. The system also provides the
get_pipeline_id utility to retrieve pipeline identifiers
from names, simplifying the execution process for users.
For monitoring purposes, the agent provides visibility into
pipeline executions through functions like list_runs and
get_run_details , supporting filtered queries by status,
experiment context, or search terms. The implementation
includes capabilities for real-time status tracking through state
information providing information about run status and asso-
ciated pipeline metadata, providing users with comprehensive
visibility into their ML workflows through the Chat UI.
D. MinIO Agent
The MinIO Agent serves as the data management coun-
terpart to the KFP Agent, facilitating interaction with object
storage services that house datasets, models, and artifacts
throughout the ML lifecycle as depicted in Fig. 3. This agent
enables users to explore, analyze, and leverage stored data
assets including metrics and results from the ML experiments
through text queries.
1) Storage Exploration and Discovery: The MinIO
Agent implements bucket and object discovery through the
get_minio_info andlist_user_buckets functions,
Fig. 3. Internal Architecture of the MinIO Agent.
allowing users to navigate through storage hierarchies using
text queries. These function accept parameters like bucket
name and prefix to enable flexible exploration of storage
contents. Key to this approach is the LLM’s ability to interpret
abstract, intent-based requests and translate them into precise
parameters for the MinIO client. For example, when a user
asks ”Show me recent image datasets from last month’s ex-
periments,” the LLM maps this request to appropriate bucket,
prefix, and filter parameters without requiring the user to
understand storage structures or query syntax.
2) ML Artifacts Interpretation: A key capability
of the MinIO Agent is its ability to retrieve and
interpret artifacts generated during ML workflow
execution through specialized functions like
get_pipeline_artifacts ,get_model_metrics ,
andget_pipeline_visualization . These functions
enable access to various artifact types including:
•Performance metrics stored as JSON files
•Evaluation visualizations such as ROC curves, confusion
matrices, and learning curves
•Model interpretation plots like feature importance charts
and partial dependence plots
•Validation results across different data splits or cross-
validation folds
The agent not only retrieves these artifacts but can also
interprete them within the conversation, translating ML train-
ing result into actionable insights. This helps users understand
model performance patterns, identify potential issues, and
formulate improvement strategies. By contextualizing perfor-
mance metrics within the broader ML workflow, the agent
provides guidance on hyperparameter tuning, feature selection
refinements, or architectural changes that could improve model
effectiveness.
3) Integration with ML Workflows: The MinIO Agent
maintains awareness of workflow context through integration
with the KFP Agent, enabling it to relate stored artifacts to
specific pipeline executions. This contextual awareness allows
the agent to:
•Connect model outputs to their source pipelines and input
datasets
•Track version evolution across multiple pipeline runs
•Compare performance metrics between workflow itera-
tions

•Provide recommendations that consider both execution
history and data characteristics
By maintaining this workflow context, the MinIO Agent
can deliver significantly more valuable insights than would be
possible through isolated storage access, effectively serving as
the memory layer for the conversational MLOps ecosystem.
E. RAG Agent
The RAG Agent implements Retrieval-Augmented Gen-
eration to enhance the system’s responses with domain-
specific knowledge from documentation about reusable Kube-
flow workflow components. As depicted in Fig. 4, this agent
facilitates context-aware responses by integrating a vector
database with the conversational flow.
Fig. 4. Internal Architecture of the RAG Agent.
1) Knowledge Indexing and Embedding: The RAG Agent
employs a data injection pipeline (Fig. 5) that processes
documentation into a searchable knowledge base. This process
involves several stages as implemented in the data injection
module. First, the system loads the input documents, pri-
marily PDF files containing technical documentation. These
documents undergo semantic chunking, which divides texts
into coherent segments based on semantic boundaries rather
than arbitrary character counts [24]. This approach preserves
the contextual integrity of information while creating manage-
able units for retrieval. Each chunk is then embedded using
an embedding model, transforming textual data into high-
dimensional vector representations that capture semantic re-
lationships. These vectors are stored in a vector database. The
implementation includes metadata preservation, ensuring that
each vector maintains associations with its source document
title and content for efficient retrieval context.
Fig. 5. Architecture of the Knowledge Indexing Pipeline.
2) Knowledge Retrieval and Integration: The core retrieval
functionality is implemented through similarity searches
against the vector database. This implementation retrieves the
most semantically similar document chunks for each query.
Retrieved information from the RAG Agent is appended in the
message history and integrated into the reasoning loops of theSwarm Agent. This integration enables information synthesis
where domain knowledge from documentation is combined
with pipeline execution capabilities and data management
functions. The RAG approach is particularly valuable when
dealing with proprietary code components or custom Kubeflow
extensions that wouldn’t be represented in the LLM’s pre-
trained knowledge. For example, when users inquire about
organization-specific pipeline templates, customized workflow
components, or proprietary model implementations, the RAG
Agent can provide accurate documentation that isn’t available
in public domains. Thus, this agent ensures that users receive
contextually appropriate and factual guidance on Kubeflow
components, enhancing the overall collaborative MLOps ex-
perience.
F . Error Handling and Response Formatting
The system incorporates error handling to manage API fail-
ures, invalid input or system problems. Each function imple-
ments try-catch mechanisms that capture and log detailed error
information allowing the LLM to adapt the input parameters to
the Agent’s functions or request additional information from
the user. Responses are structured as serializable dictionaries,
facilitating smooth integration with the parent Swarm Agent
and ensuring readable information presentation to end users.
This consistent response format enables the LLM to reason
effectively about operation results and formulate appropriate
follow-up responses or actions.
G. Extensibility
The architecture is inherently modular, enabling the inte-
gration of additional specialized agents and external services
with minimal changes to the core system. This extensibility
can support the incorporation of AI model benchmarking tools
to evaluate and compare model performance across different
configurations; data quality assessment services that identify
issues in training datasets; and specialized code assistants
that can generate or modify Kubeflow pipeline definitions
using domain-specific languages (DSL) [25], [26]. Each new
capability is exposed through the function-calling mechanism,
maintaining a consistent interaction pattern while expanding
system functionality. For instance, a code assistant agent could
leverage the KFP client’s DSL to generate complete pipeline
specifications from user’s instructions and then the KFP Agent
upload and run it at KubeFlow. This flexibility allows the
system to evolve as new capabilities can be integrated without
disrupting existing functionality.
IV. I MPLEMENTATION DETAILS
The conversational interface is built on Chainlit2, provid-
ing an asynchronous communication framework with step-
based processing for improved traceability and debugging.
For LLM interactions, the system leverages OpenAI’s API
with structured function calling capabilities. However, any
LLM with function calling capabilities can be supported. The
RAG component incorporates LangChain’s SemanticChunker
2https://github.com/Chainlit/chainlit

for context-aware document segmentation [27], OpenAI’s text-
embedding-3-small model for generating vector representa-
tions, and Pinecone as the vector database. The knowledge
retrieval pipeline uses LlamaIndex’s VectorIndexRetriever with
a top-5 similarity threshold [28]. For MLOps integrations,
the system utilizes the official Kubeflow Pipelines SDK3for
workflow management and the MinIO Python client for object
storage interactions. The system implements authentication
using Keycloak4as the Single Sign-On (SSO) solution. During
login to the chat interface, the system retrieves user-specific
session variables including Kubeflow namespace configura-
tions, MinIO bucket permissions, and KFP client credentials.
This approach ensures that users only access resources they
are authorized to use while maintaining credential isolation.
V. U SECASES AND INDICATIVE RESPONSES
To demonstrate the practical utility of the Swarm Agent
system in enabling conversational MLOps, we present several
representative use cases based on the diabetes classification
pipelines implemented in our experimental environment. These
examples illustrate how the system facilitates human-machine
collaboration across different user skill levels and task com-
plexities.
A. Pipeline Discovery and Exploration
User Query
What ML pipelines are available?
When the user inquires about available ML pipelines,
the system first leverages the KFP Agent to query the
Kubeflow pipeline registry (as shown in the tool response
containing structured JSON data). This raw information is
then processed through the Swarm Agent, resulting in a
well-formatted, human-readable response that organizes the
pipeline information hierarchically.
Tool Response
"content":
"total_pipelines": 2,
"total_available": 4,
"namespace_type": "shared",
"namespace": "shared",
"pipelines": [
"id": "d74...47559c",
"name": "diabetes-svm...",
"description": "A pipeline for...",
"created_at": "...",
"id": "297...92a78",
"name": "diabetes-dt...",
"description": "A pipeline for...",
"created_at": "...."]
3https://kubeflow-pipelines.readthedocs.io/
4https://www.keycloak.org/Agent Response
There are currently 2 ML pipelines available in the system:
1) diabetes-svm-classification-pipeline
•Description: A pipeline for diabetes...
•Created at: 2025-04-14
2) diabetes-dt-classification-pipeline
•Description: A pipeline for diabetes...
•Created at: 2025-04-14
If you want more details about any specific pipeline or want
to explore their runs or results, please let me know!
B. Pipeline Details and Parameter Understanding
User Query
Tell me more about the diabetes SVM classification pipeline
and what parameters it accepts.
This example demonstrates the system’s iterative reasoning
capability and how it uses multiple agent calls to formulate
helpful responses. Behind the scenes, the Swarm Agent
first uses the KFP Agent to retrieve basic pipeline
metadata through the get_pipeline_details function,
identifying the pipeline ID and available versions. Upon
recognizing that more information is needed for a complete
response, it makes a second call to retrieve detailed
component specifications and parameter definitions via
get_pipeline_version_details .
Tool Response
"id": "d74d8...559c",
"name": "diabetes-svm-...",
"description": "A pipeline...",
"created_at": "2025-04-14...",
"pipeline_versions": [...]
Tool Response
"pipeline_spec": "...",
"components":
"comp-train-svm":
"inputDefinitions":
"parameters":
"C": "defaultValue": 1,
"parameterType": "NUMBER_DOUBLE",
"kernel": "defaultValue": "rbf",
"parameterType": "STRING",
"random_state": "defaultValue": 42,
"parameterType": "NUMBER_INTEGER",
"comp-split-data":
"inputDefinitions":
"parameters":
"random_state": "defaultValue": 42,
"parameterType": "NUMBER_INTEGER",
"test_size": "defaultValue": 0.3,
"parameterType": "NUMBER_DOUBLE"

Agent Response
The Diabetes SVM Classification Pipeline is a demonstration
pipeline for diabetes classification using a Support Vector
Machine (SVM) model with artifact tracking in MinIO.
Pipeline Structure:
1) Data loading - Loads the diabetes dataset from scikit-
learn and converts it to a binary classification problem
2) Data splitting - Divides data into training and test sets
3) Preprocessing - Normalizes features using Standard-
Scaler
4) Model training - Trains an SVM classifier on the
preprocessed training data
5) Evaluation - Generates performance metrics and visu-
alizations
6) Artifact storage - Saves models, metrics, and visual-
izations to MinIO
The pipeline accepts the following parameters:
•test size: float (default: 0.3) - Proportion of the dataset
used for testing (e.g., 0.3 means 30% for testing, 70%
for training)
•random state: int (default: 42) - Random seed for
reproducibility of data splitting and model training
•svm C: float (default: 1.0) - Regularization parameter;
smaller values enforce stronger regularization
•svm kernel: string (default: ’rbf’) - The kernel type
to be used in the algorithm, options include ’linear’,
’poly’, ’rbf’, ’sigmoid’
•Various MinIO/authentication parameters (typically set
by system administrators)
Would you like to run this pipeline with specific parameters
or compare it with the Decision Tree pipeline?
C. Comparing Model Performance
User Query
Compare the performance of SVM and Decision Tree models
for diabetes classification.
This example demonstrates the system’s ability to organize
parallel tool calls for efficient data retrieval and comparison.
Upon receiving the user’s query, the Swarm Agent first calls
MinIO’s list_user_buckets function, which returns
information about available storage buckets, identifying the
paths in MinIO containing the SVM and Decision Tree
pipeline artifacts. The system recognizes the two relevant
pipeline names: diabetes-svm-classification
and diabetes-dt-classification . With this
context, the Swarm Agent then makes parallel calls to
theget_model_metrics function—one for each model
type—to retrieve their performance metrics. The system then
synthesizes these parallel data streams, aligning the metrics
for direct comparison and presenting them to the user.
D. Additional Supported Query Types
The Swarm Agent supports diverse query types beyond
the demonstrated examples, enabling comprehensive MLOps
management through natural language.Agent Response
I’ve compared the performance metrics of the SVM and
Decision Tree models for diabetes classification based on
the most recent runs:
Metric SVM Decision Tree
Accuracy 0.752 0.706
Precision 0.739 0.699
Recall 0.773 0.718
F1 Score 0.756 0.709
AUC 0.842 0.708
Confusion Matrices:
•SVM: [[49, 18], [15, 51]]
•Decision Tree: [[77, 34], [31, 79]]
Key insights:
1) The SVM model outperforms the Decision Tree model
across all key metrics including accuracy, precision,
recall, F1 score, and AUC.
2) The higher AUC of the SVM (0.842) indicates better
overall discrimination capability between classes.
3) The confusion matrix for SVM shows fewer false
positives and false negatives compared to the Decision
Tree.
Would you like me to provide visualizations or more detailed
analysis of these models’ results?
These include:
1) Experiment Creation and Modification Queries: Users
can request ”Run a new experiment using sigmoid kernel”
leveraging the system’s understanding of pipeline parameters
and chat history.
2) Data Management Queries: The MinIO Agent facilitates
queries like ”List all datasets uploaded in the past month” or
”Compare feature importance across multiple experiments,”
enabling straightforward data exploration.
3) Workflow Monitoring and Debugging: Users can ask
”Why did the latest diabetes pipeline run fail?” or ”Show
me logs from the preprocessing step of run-123,” with the
system aggregating relevant execution details and presenting
actionable insights.
4) Domain-Specific Knowledge Integration: The RAG
Agent handles queries such as ”How does the XYZ classifier
handle categorical features in diabetes prediction?” by retriev-
ing and contextualizing documentation from proprietary model
libraries not covered in the LLM’s pre-trained knowledge.
5) Cross-Pipeline Analysis: Complex analytical queries
like ”Which pipeline version achieved the highest precision
for minority class samples?” or ”Generate a report comparing
all model runs from April” are processed through coordinated
function calls across multiple specialized agents.
These capabilities collectively transform how research teams
interact with MLOps infrastructure, removing technical barri-
ers and enabling domain experts to directly leverage advanced
ML workflows in their research.
VI. C ONCLUSION AND FUTURE WORK
This paper presented the Swarm Agent, a novel LLM-
powered conversational system for the management of MLOps

through chat interactions. By integrating specialized agents for
KubeFlow Pipelines orchestration, MinIO data management,
and domain-specific knowledge retrieval, our approach bridges
the gap between complex ML infrastructure and users with di-
verse technical backgrounds. Through iterative reasoning and
contextual processing, the system enables pipeline discovery,
execution, monitoring, and results analysis without requiring
expertise in underlying technologies. The demonstrated ex-
amples illustrate how this conversational approach reduces
cognitive load, eliminates technical barriers, and enhances
human-machine collaboration in ML workflow management.
For future work, we plan to evaluate the system’s perfor-
mance through user studies across stakeholders with varying
technical backgrounds and domains, collecting metrics on
task completion efficiency and usability. We will also focus
on addressing edge cases identified during these evaluations,
particularly improving the system’s reasoning for complex
multi-step operations and enhancing its ability to recover from
ambiguous user instructions. Additionally, we intend to inte-
grate a Code Agent that will create new pipeline components
and workflows from human instructions, dynamically compile
them, and upload them to Kubeflow. This capability will
transform the system from primarily supporting discovery and
execution of existing pipelines to facilitating the creation of
new workflows without requiring manual coding, further as-
sisting ML development across research teams and disciplines.
ACKNOWLEDGMENT
Part of the research leading to the results presented in this
paper has received funding from the European Union’s funded
Project HUMAINE under grant agreement no 101120218.
REFERENCES
[1] A. Kiourtis, A. Mavrogiorgou, G. Makridis, D. Kyriazis, J. Soldatos,
G. Fatouros, D. Ntalaperas, X. Papageorgiou, B. Almeida, J. Guedes
et al. , “Xr5. 0: Human-centric ai-enabled extended reality applications
for industry 5.0,” in 2024 36th Conference of Open Innovations Associ-
ation (FRUCT) . IEEE, 2024, pp. 314–323.
[2] G. Fatouros, K. Metaxas, J. Soldatos, and D. Kyriazis, “Can large
language models beat wall street? evaluating gpt-4’s impact on financial
decision-making with marketsenseai,” Neural Computing and Applica-
tions , pp. 1–26, 2024.
[3] G. Makridis, D. Kyriazis, and S. Plitsos, “Predictive maintenance
leveraging machine learning for time-series forecasting in the maritime
industry,” in 2020 IEEE 23rd international conference on intelligent
transportation systems (ITSC) . IEEE, 2020, pp. 1–8.
[4] G. Fatouros, G. Makridis, D. Kotios, J. Soldatos, M. Filippakis, and
D. Kyriazis, “DeepVaR: a framework for portfolio risk assessment
leveraging probabilistic deep neural networks,” Digital finance , vol. 5,
no. 1, pp. 29–56, 2023, publisher: Springer.
[5] M. Zarour, H. Alzabut, and K. Alsarayrah, “Mlops best practices, chal-
lenges and maturity models: A systematic literature review,” Information
and Software Technology , p. 107733, 2025.
[6] J. George and A. Saha, “End-to-end machine learning using kubeflow,”
inProceedings of the 5th Joint International Conference on Data Science
& Management of Data (9th ACM IKDD CODS and 27th COMAD) ,
2022, pp. 336–338.
[7] S. Sch ¨obel, A. Schmitt, D. Benner, M. Saqr, A. Janson, and J. M.
Leimeister, “Charting the evolution and future of conversational agents:
A research agenda along five waves and new frontiers,” Information
Systems Frontiers , vol. 26, no. 2, pp. 729–754, 2024.
[8] C. Ouaddi, L. Benaddi, and A. Jakimi, “Architecture, tools, and dsls
for developing conversational agents: An overview,” Procedia Computer
Science , vol. 231, pp. 293–298, 2024.[9] OpenAI, “Function calling,” https://platform.openai.com/docs/guides/
function-calling, accessed: 2025-04-05.
[10] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel et al. , “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
neural information processing systems , vol. 33, pp. 9459–9474, 2020.
[11] G. Fatouros, K. Metaxas, J. Soldatos, and M. Karathanassis, “Market-
senseai 2.0: Enhancing stock analysis through llm agents,” arXiv preprint
arXiv:2502.00415 , 2025.
[12] H. Shin, W. Park, S. Kim, J. Kweon, and C. Moon, “Driver
identification system based on a machine learning operations platform
using controller area network data,” Electronics , 2025. [Online].
Available: https://api.semanticscholar.org/CorpusID:277029352
[13] R. Bergmann, F. Theusch, P. Heisterkamp, and
N. Grigoryan, “Comparative analysis of open-source ml
pipeline orchestration platforms,” 2024. [Online]. Available:
https://www.researchgate.net/publication/382114154 Comparative
Analysis ofOpen-Source ML Pipeline Orchestration Platforms
[14] A. Keusch, T. Blumauer-Hiessl, A. Furutanpey, D. Schall, and
S. Dustdar, “Platform-agnostic mlops on edge, fog and cloud
platforms in industrial iot,” in International Conference on Web
Information Systems and Technologies , 2024. [Online]. Available:
https://api.semanticscholar.org/CorpusID:274203784
[15] V . Katevas, G. Fatouros, D. Kyriazis, and G. Kousiouris, “Embedding
automated function performance benchmarking, profiling and resource
usage categorization in function as a service devops pipelines,” Future
Generation Computer Systems , vol. 160, pp. 223–237, 2024.
[16] A. Grilo, P. Figueiras, B. R ˆega, L. Lourenc ¸o, A. Khodamoradi,
R. Costa, and R. Jardim-Gonc ¸alves, “Data analytics environment:
Combining visual programming and mlops for ai workflow creation,”
2024 IEEE International Conference on Engineering, Technology,
and Innovation (ICE/ITMC) , pp. 1–9, 2024. [Online]. Available:
https://api.semanticscholar.org/CorpusID:274823746
[17] B. P. Harenslak and J. De Ruiter, Data pipelines with apache airflow .
Simon and Schuster, 2021.
[18] M. Zaharia, A. Chen, A. Davidson, A. Ghodsi, S. A. Hong, A. Konwin-
ski, S. Murching, T. Nykodym, P. Ogilvie, M. Parkhe et al. , “Accelerat-
ing the machine learning lifecycle with mlflow.” IEEE Data Eng. Bull. ,
vol. 41, no. 4, pp. 39–45, 2018.
[19] S. Peng, E. Kalliamvakou, P. Cihon, and M. Demirer, “The impact of ai
on developer productivity: Evidence from github copilot,” arXiv preprint
arXiv:2302.06590 , 2023.
[20] GitHub, “Github copilot · your ai pair programmer,” https://github.com/
features/copilot, accessed: 2025-04-05.
[21] Cursor, “Cursor - the ai code editor,” https://www.cursor.com, accessed:
2025-04-05.
[22] C. Labs, “Devin,” https://devin.ai, accessed: 2025-04-05.
[23] CodeGPT, “Ai agents platform for software development teams,” https:
//codegpt.co, accessed: 2025-04-05.
[24] D. Tomkou, G. Fatouros, A. Andreou, G. Makridis, F. Liarokapis,
D. Dardanis, A. Kiourtis, J. Soldatos, and D. Kyriazis, “Bridging
industrial expertise and xr with llm-powered conversational agents,”
arXiv preprint arXiv:2504.05527 , 2025.
[25] G. Fragiadakis, C. Diou, G. Kousiouris, and M. Nikolaidou, “Evaluating
human-ai collaboration: A review and methodological framework,” arXiv
preprint arXiv:2407.19098 , 2024.
[26] G. Fatouros, G. Makridis, A. Mavrogiorgou, J. Soldatos, M. Filip-
pakis, and D. Kyriazis, “Comprehensive architecture for data quality
assessment in industrial iot,” in 2023 19th International Conference
on Distributed Computing in Smart Systems and the Internet of Things
(DCOSS-IoT) . IEEE, 2023, pp. 512–517.
[27] J. Zhao, Z. Ji, P. Qi, S. Niu, B. Tang, F. Xiong, and Z. Li, “Meta-
chunking: Learning efficient text segmentation via logical perception,”
arXiv preprint arXiv:2410.12788 , 2024.
[28] R. K. Malviya, V . Javalkar, and R. Malviya, “Scalability and perfor-
mance benchmarking of langchain, llamaindex, and haystack for enter-
prise ai customer support systems,” in IJGIS Fall of 2024 Conference .
The New World Foundation, 2024.