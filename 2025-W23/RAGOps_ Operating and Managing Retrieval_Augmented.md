# RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines

**Authors**: Xiwei Xu, Hans Weytjens, Dawen Zhang, Qinghua Lu, Ingo Weber, Liming Zhu

**Published**: 2025-06-03 21:22:23

**PDF URL**: [http://arxiv.org/pdf/2506.03401v1](http://arxiv.org/pdf/2506.03401v1)

## Abstract
Recent studies show that 60% of LLM-based compound systems in enterprise
environments leverage some form of retrieval-augmented generation (RAG), which
enhances the relevance and accuracy of LLM (or other genAI) outputs by
retrieving relevant information from external data sources. LLMOps involves the
practices and techniques for managing the lifecycle and operations of LLM
compound systems in production environments. It supports enhancing LLM systems
through continuous operations and feedback evaluation. RAGOps extends LLMOps by
incorporating a strong focus on data management to address the continuous
changes in external data sources. This necessitates automated methods for
evaluating and testing data operations, enhancing retrieval relevance and
generation quality. In this paper, we (1) characterize the generic architecture
of RAG applications based on the 4+1 model view for describing software
architectures, (2) outline the lifecycle of RAG systems, which integrates the
management lifecycles of both the LLM and the data, (3) define the key design
considerations of RAGOps across different stages of the RAG lifecycle and
quality trade-off analyses, (4) highlight the overarching research challenges
around RAGOps, and (5) present two use cases of RAG applications and the
corresponding RAGOps considerations.

## Full Text


<!-- PDF content starts -->

arXiv:2506.03401v1  [cs.SE]  3 Jun 2025RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines
Xiwei Xua,c, Hans Weytjensb, Dawen Zhanga, Qinghua Lua,c, Ingo Weberb,d, Liming Zhua,c
aCSIRO’s Data61, Australia
bTechnical University of Munich, School of CIT, Germany
cUniversity of New South Wales, School of Computer Science and Engineering, Australia
dFraunhofer Society, Munich, Germany
Abstract
Recent studies show that 60% of LLM-based compound systems in enterprise environments leverage some form
of retrieval-augmented generation (RAG), which enhances the relevance and accuracy of LLM (or other genAI) out-
puts by retrieving relevant information from external data sources. LLMOps involves the practices and techniques for
managing the lifecycle and operations of LLM compound systems in production environments. It supports enhancing
LLM systems through continuous operations and feedback evaluation. RAGOps extends LLMOps by incorporating
a strong focus on data management to address the continuous changes in external data sources. This necessitates
automated methods for evaluating and testing data operations, enhancing retrieval relevance and generation quality.
In this paper, we (1) characterize the generic architecture of RAG applications based on the 4 +1 model view for de-
scribing software architectures, (2) outline the lifecycle of RAG systems, which integrates the management lifecycles
of both the LLM and the data, (3) define the key design considerations of RAGOps across di fferent stages of the RAG
lifecycle and quality trade-o ffanalyses, (4) highlight the overarching research challenges around RAGOps, and (5)
present two use cases of RAG applications and the corresponding RAGOps considerations.
Keywords: LLM, LLMOps, RAG, RAGOps
1. Introduction
Large Language Models (LLMs) can be instructed through prompting to perform a wide range of tasks, such as
programming and translation. A notable trend in their application is the integration of LLMs into compound software
systems, which consist of multiple components beyond the core language model [1]. Compound LLM systems can
perform dynamic behaviors, whereas LLMs by themselves are inherently constrained by their reliance on static train-
ing on datasets from some point in time, resulting in fixed parametric knowledge and limited grounding in specific
contexts in which the systems are used, such as a given organization. In enterprise settings, 60% of the LLM com-
pound systems incorporate some form of retrieval-augmented generation [2] (RAG), which improves the relevance,
accuracy, and dynamism of LLM outputs by retrieving information from external data. RAG o ffers a solution to
common challenges faced by LLMs, such as hallucinations, outdated data, and the di fficulty of removing parametric
knowledge, and open up the possibility to access proprietary, internal data from organizations using existing LLMs.
By integrating real-time information retrieval, RAG enables continuous updates, possibly incorporating very recent
information. RAG systems are compound systems that consist of multiple components blending the LLM’s parametric
knowledge with external data retrieval, including, but not limited to retrieval sources ,retriever , and generator [3].
LLMOps refers to the practices and techniques used to manage the lifecycle and operation of LLM and LLM com-
pound systems in production environments. The current state of LLMOps includes a variety of automated tools123
designed to observe, monitor, optimize, and scale LLM applications. The functionality of these tools includes but
is not limited to, model versioning, performance monitoring, continuous retraining, inference optimization, and in-
frastructure scaling to accommodate fluctuating demand. Current LLMOps tools predominantly focus on model and
1“Managed MLflow,” Databricks, accessed 5 April 2025, https://www.databricks.com/product/managed-mlflow
2“Intelligent Observability,” New Relic, accessed 5 April 2025, https://newrelic.com/
3“LangSmith,” LangChain, accessed 5 April 2025, https://www.langchain.com/langsmith

Figure 1: Scoping RAGOps: a RAG system can run either independently or be embedded in an LLM-based compound system (AI App, potentially
an AI agent). It has a query processing component interacting with data sources to retrieve information and a genAI model (usually an LLM) to
formulate an answer. We exclude LLMOps from our RAGOps discussion.
prompting management, o ffering limited support for data-related aspects, particularly the data created and used after
an LLM has been trained and deployed.
This paper introduces RAGOps, a conceptual framework that builds upon and extends LLMOps, with a focus
on managing both the data lifecycle and operational aspects of RAG systems. Our contributions are threefold: (1)
we conceptualize RAG systems and provide a coherent terminology; (2) we o ffer design guidelines for building and
maintaining RAG systems; and (3) we analyze quality trade-o ffs to address challenges arising from the continuously
evolving data retrieved by RAG applications. Grounded in the paradigm of Observability , RAGOps provides features
and functionalities such as managing diverse data structures, formats, and dynamic data updates, while leveraging
an array of tools and mechanisms to monitor performance and enable continuous improvement. This is achieved
through ongoing evaluation, testing, and the integration of operational feedback. Reliable observability mechanisms
are critical, as they directly impact the dependability of LLM applications (including RAG applications). Operational
aspects are integrated into the system as a whole, and only a comprehensive evaluation and testing of the entire
pipeline, including all individual components, can ensure the system’s overall reliability.
Although LLMs4have their own development pipeline—from data identification and collection to parameteriza-
tion in an LLM model—this process is outside the scope of this paper. Our focus is on RAG applications that utilize
LLMs rather than on training LLMs from scratch. In practice, a RAG system is typically embedded within an AI ap-
plication such as a chatbot or a customer support assistant, or, more recently an AI agent. However, in the body of this
paper we focus on the commonalities in RAG applications and not individual applications, as visualized in Figure 1
– in contrast, the use cases consider individual applications. The right-hand side depicts our corresponding definition
of RAGOps and how this paper is positioned within that framework. Since information retrieval from external data
sources is the defining feature of RAG systems, Figure 1 explicitly includes data management.
In the following section (Section 2), we first discuss related work. In Section 3, we characterize the software
architecture of RAG applications that may influence the entailment of RAGOps, based on the 4 +1 model view. Next,
we outline the lifecycle of RAG architectures, which integrates both the query processing pipeline lifecycle and the
data management lifecycle, in Section 4. The data management lifecycle includes stages such as ingest, verify, and
update [4]. In the context of RAG applications, these two lifecycles are closely interrelated, as the LLM and external
data sources form the core components of a RAG application. We outline the key operational considerations for
lifecycle and operational management across the various stages of the RAG lifecycle in Section 5 and address the
research challenges in Section 6, taking into account the distinct characteristics of RAG applications. Additionally,
we discuss two use cases in Section 7, before summarizing the paper and o ffering suggestions for future research in
Section 8.
4For simplification, we will refer to LLMs in the remainder of this text. For multimodal RAGs, other genAI models will be required.
2

2. Related Work
2.1. LLMOps
The use of LLMs in production environments poses new challenges that expose the limitations of existing method-
ologies in the DevOps space, like MLOps. LLMOps has emerged [5] with a range of tools and best practices primarily
for managing the lifecycle of LLM applications. This includes but is not limited to fine-tuning LLMs, prompt engi-
neering, data provenance for in-context learning, and building infrastructure for training, testing and deploying LLMs.
The focus of LLMOps lies in the operational capabilities and infrastructure necessary to evolve LLMS and deploy the
new versions e ffectively [6].
Despite its advancements, most existing LLMOps work [7] primarily targets metrics specifically designed for
LLM management and prompt management. Existing work provides limited support for observability in the retrieval
process of RAG applications. This gap results in insu fficient observability from the perspective of the RAG pipeline.
2.2. Observability and Monitorability
Monitorability is a quality attribute that refers to the ability to track behavior and performance of software systems
using predefined metrics and alerts. It focuses on addressing “known unknowns” through actionable insights and
alerts [8]. While not exclusive to machine learning (ML) and AI systems, monitorability is crucial for their proper
operation and sustainment [9]. Automated and continuous improvement processes, including model retraining, rely
heavily on e ffective monitorability.
Monitorability spans the whole lifecycle of AI /ML models, establishing a well-structured pipeline [10] to enhance
the performance and other qualities of AI /ML components through iterations of refinement, continuing until the model
cannot be further improved. Monitoring changes in external data and understanding their impacts on both individual
ML components and the overall AI /ML system introduces additional complexity, highlighting the need for e ffective
and e fficient monitoring mechanisms.
Observability in AI /ML systems [11, 8] goes beyond monitoring metrics that are predefined to capture system
health. It empowers practitioners to investigate system behaviors by analyzing historical outputs on “unknown un-
knowns” or conducting “needle-in-a-haystack” queries [12]. It is essential for LLMOps tools to support observability
features, enabling stakeholders to monitor the behavior of the applications, trace the evolution of artifacts, log associ-
ated data, detect anomalies, and assign accountability if incidents occur [13].
3. Characterization of RAG Architecture
Retrieval-Augmented Generation (RAG) is an e ffective solution for handling hallucinations in LLM by leveraging
external data that is not covered by the training data of the LLM. This approach enhances the accuracy and credibility
of LLM outputs, especially in knowledge-intensive tasks. Moreover, RAG facilitates continuous data updates and
enables the integration of domain-specific expertise into LLM compound systems [3, 14, 15]. This section applies
the 4+1 View Model of software architecture to conceptualize and characterize the architecture of RAG applications.
These architectural characteristics impact the design and implementation of RAGOps (Section 4).
The4+1 View Model of software architecture is introduced by Philippe Kruchten in 1995 [16]. It is a widely used
framework in the software engineering and architecture communities. The 4 +1 View Model provides a structured
approach to describing software architecture through five distinct views, each of which addresses specific concerns
from the perspectives of various stakeholders.
•Logic view: Focuses on the functionality the system provides to end-users.
•Process view: Addresses the dynamic behavior of the system at runtime, explaining how system processes
interact and communicate.
•Development view: Represents the system from a developer’s perspective, emphasizing software organization
and management. It is also refereed to as the implementation view.
•Physical view: Focuses on the deployment of software components within the physical layer and the physical
connections between them
3

•Scenarios: Uses a set of use cases to describe and illustrate the software architecture.
A conceptual architecture of a typical RAG system with its key components is demonstrated in Figure 2.
Figure 2: RAG System Architecture.
3.1. Logic View
A basic and typical functionality provided by a RAG architecture involves retrieving relevant documents or pieces
of information from an external data source to compose the context and then relying the LLM to generate responses
with consideration of the retrieved data. As shown in Figure 2, the three key components of a typical RAG architec-
ture include retrieval sources ,retriever , and generator . The user inputs a query through the user interface with the
generator potentially optimizing the query. The following retrieval process can be based on embeddings, keyword
matching, other search algorithms, or a hybrid approach [17]. The retrieval resources may include data sources in
different data structures and formats, including but not limited to an embedding vector database and knowledge graph
in the retrieval sources module and API access to several internal and external sources.
Suppose a user (researcher) asks a RAG application about recent developments in quantum computing. The
application could first retrieve the latest research papers or articles from an academic database and then generate a
summary or explanation based on the retrieved content.
Ensuring the safety and reliability of AI-generated content involves implementing various safeguards [18]. Ap-
propriateness checks screen outputs to eliminate toxic, harmful, or biased content before it reaches users. Accuracy
verification confirms that AI-generated information is factually correct and not misleading, combating hallucinations.
Regulatory compliance ensures that outputs adhere to relevant laws and industry-specific regulations. User alignment
maintains consistency with user expectations and the intended purpose, e.g., ensuring brand consistency. Content
validation assesses outputs against predefined criteria.
4

Human-in-the-loop is a widely adopted strategy in LLM compound systems, where the human is directly involved
to intuitively enhance system performance. Human feedback, as a subjective signal, has been used to helping LLM
compound systems align with human values and preferences.
In the context of RAG, particularly in specialized domains such as scientific research or law, a portion of knowl-
edge remains undocumented, residing instead in the tacit knowledge of specialists. As shown in Figure 2, such
specialists or experts are one of the user types in the system. For instance, when working with tabular data, the
schema may o ffer a basic structural outline, but the deeper rationale behind the data organization, such as semantic
relationships between columns, is not often captured by any document. Subtle characteristics, like patterns of unique-
ness, anomalies, or other nuanced data traits, may go unnoticed or unrecorded. Furthermore, contextual insights, such
as exponential relationships within the data, are typically uncovered during analysis rather than explicitly provided.
Additionally, analyzing such data e ffectively often requires specialized expertise. This includes an intuitive grasp
of data patterns, the significance of specific variables in relation to carefully designed experiments, and the interplay
between variables. Without access to this implicit knowledge and contextual insights, accurate data interpretation can
become challenging. In addition to understanding the semantic significance of specific columns in proprietary tabular
datasets, other examples of such knowledge gaps include interpreting legal clauses in complex scenarios or under new
legislation, and identifying missing connections in graph-based data sources.
In RAG, human feedback can be incorporated as a potential retrieval source, in addition to other passive retrieval
sources. The generator can incorporate human feedback into its prompts, providing just-in-time feedback and leading
to more informed planning and reasoning. In such cases, when experts (as a type of users) interact with RAG, they
become active retrieval sources, accessed through an expert input call.
The RAG system can be used either as a standalone system, as described above, or exposed through an API to an
AI agent. Standards to facilitate this new form of communication are currently under development [19].
3.2. Process View
In addition to the overall architecture, Figure 2 also illustrates the process flow of an advanced RAG-system,
outlining the interactions between its key components. It is important to distinguish between two primary processes:
theQuery Processing Pipeline andData Management .
The Query Processing Pipeline refers to the real-time sequence of operations initiated when a user submits a query.
Data Management, on the other hand, encompasses the ongoing procedures dedicated to organizing and updating the
system’s data and retrieval sources.
3.2.1. Query Processing Pipeline
In advanced RAG-systems, the generator may contain a module that enhances the user’s query to optimize re-
trieval. The retriever’s reasoning and planning module analyzes the (enhanced) query to determine the most e ffective
retrieval strategy. This decision-making process includes selecting appropriate retrieval sources (e.g., vector database
within the retrieval sources module, external APIs, direct access to a proprietary database, or consulting the human ex-
pert), picking specific retrieval methods (e.g., similarity search, keyword search), and formulating queries for various
APIs (e.g., general search engines, specialized databases).
After its decision-making process, the retriever performs the actual retrieval. In the regular cases, it consults
the vector database. That requires a preliminary embedding of the (enhanced) user query. The reranking module
prioritizes the retrieved information, removing redundancies and irrelevant data points. In a more agentic approach,
the autonomous retriever could also decide to reiterate and refine the whole process after detecting insu fficient results.
Both reasoning and planning and reranking can involve heuristic rules, ML models or interaction with an LLM (as
indicated by the arrow connecting the retriever and the LLM in Figure 2).
The generator synthesizes a query integrating the original user query with the retrieved information, which is then
sent to a selected LLM. The model’s response is validated to verify accuracy and relevance before sending it back to
the user interface. When the system has agentic capabilities, it can also decide to reiterate and improve the process,
either in its entirety or for a selected step.
The interaction between the retriever and generator can also become more intertwined, a process referred to as
Retrieval Interleaved Generation (RIG) [20]. In RIG, real-time information retrieval is dynamically embedded within
the LLM’s generation process. Unlike the linear approach, where external data is retrieved before the LLM generates a
5

response, RIG allows for seamless integration of retrieval into generation, enabling finer-grained interactions between
the two components.
Early RAG systems employ a static retrieval process, where the retriever and generator are used linearly. Recently,
RAG systems have progressively evolved toward more dynamic workflows, incorporating adaptive, recursive, and
interactive retrieval processes. These advancements are supported by emerging RAG frameworks [21, 3].
3.2.2. Data management
The retrieval sources module processes data di fferently depending on the modality: texts are chunked to split
long documents into smaller, semantically meaningful units before being converted into embeddings, while other
modalities such as images, audio, and video are directly embedded without chunking. Embeddings are stored in a
vector database for similarity-based retrieval, while other storage solutions (e.g., document stores, knowledge graphs)
can be used for alternative retrieval methods such as keyword-based or structured queries. The mechanisms of data
management which keeps the data lake and retrieval sources current will be discussed in detail in Section 4.2.
3.2.3. Guardrails
Within the RAG-context, guardrails can be implemented at di fferent stages of the pipeline [22, 23, 24, 25] as
illustrated by the
 icons in Figure 2. Guardrails serve as intermediaries between two components when they
interact.
Input rails (I) process user inputs, either rejecting them to halt further processing or modifying them, such as
masking sensitive information or rephrasing before handing them over to the generator. Dialog rails (D) shape the
prompting of the LLM and guide the conversation’s progression. Retrieval rails (R) manage the retrieved data seg-
ments. They can exclude retrieved information from being passed to the retriever or from being used by the generator
to prompt the LLM or modify segments, for instance, to conceal sensitive information or poisoned information. Out-
put rails (O) apply to the LLM’s generated outputs, with the capability to reject responses, preventing them from
reaching the user, or to modify them, such as by removing sensitive content. More rail actions and targets can be
found in [22], but it is important to balance the number and placement of these guardrails to avoid redundancy, which
can lead to unnecessary complexity and ine fficiency.
3.3. Development View
Embeddings in vector databases are the most widely used and typical retrieval sources for RAG. Beyond embed-
dings, external databases and other data sources can be structured and accessed in various ways. Humans with tacit
knowledge can also serve as more active retrieval sources. Below is a list of possible retrieval architectures from
a developer’s perspective, which shows the flexibility of LLMs in integrating with external data sources to deliver
accurate, context-aware, and real-time responses. Each architecture can be refined to specific use cases depending on
the characteristics of the required information and the available technical infrastructure.
•RAG-system retrieval sources contain data from the curated data lake that are processed and stored in a RAG-
specific way. Commonly, the information is stored along embeddings in a vector database. Alternatives include
knowledge graphs (text or images may be associated with the nodes of the graph) and document stores (to
retrieve text chunks via complementary retrieval mechanisms such as keyword matching TF /IDF), etc.
•APIs permit direct access to (possibly non-curated) data to both internal sources, such as databases, email
servers, etc., and external sources, such as the web.
•Human-in-the-retrieval RAG applications integrate humans’ knowledge in several ways. One approach is for
the human experts to manually provide additional context through prompting. Alternatively, RAG applications
can connect to external crowdsourcing knowledge platforms via APIs to integrate collective expertise.
3.4. Physical View
There are various deployment and infrastructure options for RAG systems. Each key component (generator,
retriever, and retrieval sources) requires specific deployment and hosting considerations, such as whether to host the
component on the cloud or on-premises.
6

Selecting the appropriate LLM for RAG systems is akin to choosing computational infrastructure, particularly in
scenarios involving multiple LLMs for di fferent purposes. For instance, one LLM may be used for general generation
while another, such as a dedicated “LLM guard,” verifies content at various stages. An example is reviewing the
execution of SQL queries that access external tabular data [26].
Retrieval sources may need to be constructed from raw data in various formats as shown in Figure 2. Alternatively,
existing data sources can be utilized directly through RESTful APIs, such as knowledge graphs like Wikidata5.
3.5. Scenario
The usage scenarios of RAG systems vary widely, falling primarily into two categories or a combination of the
two: 1) Querying specific details, and 2) Gaining a holistic understanding of a source [27]. The source in question
could be in di fferent format, a text-based document, an image, or a video or a combination of them. An example
of querying specific details in the context of a customer assistant might be, “What is the return policy for online
purchases?” Similarly, in the context of a legal assistant, a user might ask, “What’s the capital gains tax on the sale
of a rental property?” Tasks that require a holistic understanding focus on broader comprehension or synthesis of
information from a big document or multiple sources. For example, a legal assistant might be tasked with summarizing
a lengthy legal document to extract key clauses, implications, and risks.
Some usages scenarios require a combination of both querying specific details and gaining a holistic understand-
ing, particularly in reasoning-intensive tasks. For instance, in a scientific research assistant scenario, a researcher
might query specific data points from di fferent studies while simultaneously synthesizing these details to form a
cohesive understanding of a broader research landscape.
In such reasoning tasks, the RAG application must seamlessly integrate detailed queries with broader context com-
prehension, ensuring accuracy and relevance at every stage. Depending on the characteristics of the usage scenarios,
different data sources with various data structures are selected and /or structured.
4. RAGOps
Excluding LLMOps and the wrapping app (see Figure 1), RAGOps can be broken down into two tightly knit,
yet di fferent lifecycles: the query processing pipeline, that follows the traditional DevOps lifecycle and a more data-
centric data management.
4.1. Query Processing Pipeline
When stripped of its surrounding application, ignoring the LLM(s) that drive it, and setting aside the separate
concerns of data management, a RAG system becomes a typical piece of software. As such, it naturally falls within
the domain of traditional DevOps practices.
The DevOps lifecycle is comprised of seven interconnected phases in a continuous loop, as illustrated in the upper
(white) part of Figure 3. It embraces the principles of continuous integration and continuous deployment.
a.Planning and Development : This phase involves defining system requirements, selecting appropriate tools,
and outlining an architectural blueprint. For RAG systems, this means identifying the knowledge domain,
choosing data sources, and designing retrieval and generation strategies. Teams determine the LLMs, embed-
ding models, vector database and other retrieval sources, prompt engineering techniques, etc.
b.Build : The build phase includes setting up necessary components, writing core logic, and ensuring dependen-
cies are properly configured. In a RAG system, this involves preprocessing and chunking documents, generating
embeddings, indexing them in a vector database for e fficient retrieval, and constructing knowledge graph in the
case of using knowledge graph. Additionally, teams configure the retrieval pipeline, integrate it with the LLM,
and package components—often using containerization.
5Wikidata, accessed 5 April 2025, https://www.wikidata.org/
7

Figure 3: RagOps: intertwined query processing pipeline and data management lifecycles.
c.Testing : While standard testing covers functionality, performance, and security, RAG-specific evaluation also
includes assessing retrieval relevance, response accuracy, and overall coherence. This means verifying that
embeddings align with expected semantic meanings, ensuring retrieved documents are the most relevant, and
evaluating the model’s factual consistency. System load testing is particularly crucial, as retrieval latencies and
model response times can impact real-time performance. Benchmark datasets and automated testing frame-
works are commonly used to validate pipeline e ffectiveness before deployment.
d.Release : The release phase finalizes the system for production, ensuring all components are properly integrated
and meet performance, accuracy, and stability requirements. This involves validating the compatibility of the
retriever, generator, and retrieval sources, refining versioning strategies, and preparing documentation. The
system undergoes final approval, often deployed in a staging environment before full deployment, ensuring
updates can be rolled out smoothly.
e.Deployment : Deployment focuses on infrastructure stability, automated scaling, and continuous integration. In
RAG systems, deployment also involves ensuring seamless interaction between the retriever, retrieval sources,
and language model in real-world conditions. The ongoing challenges concerning data updates could be in-
cluded in the deployment phase as well. Given their frequency and importance, we argue for a treatment as a
separate component of the RAGOps lifecycle.
f.Operate and Monitor : Monitoring covers infrastructure health, logging, and alerting. In a RAG system,
additional tracking is necessary for retrieval e fficiency, response fidelity, and data drift detection. Monitoring
tools assess whether retrieved documents remain relevant, whether response quality degrades due to evolving
data sources, and how changes in embeddings impact retrieval precision. The system is continuously adjusted
to maintain e ffectiveness as the underlying data evolves.
g.Analyze and Feedback : Incident response and performance tuning are core DevOps practices, but in RAG,
this phase also involves diagnosing retrieval errors, detecting data drift, and refining query-document matching.
Teams analyze retrieval logs, user interactions, and model responses to identify systemic issues, such as retrieval
failures, hallucinations, or outdated information. Insights from this analysis guide iterative improvements,
ensuring that the system remains accurate and e ffective over time.
Once the query processing pipeline operates in the production environment, analysis based on measurements taken
will lead to occasional updates, e ffectively closing the loop and triggering new developments. We elaborate on the
measurements in our discussion about monitorability and observability in Section 5.1. However, during deployment,
the data will be updated at a much higher frequency, potentially continuously. Data maintenance requires a dedicated
lifecycle.
4.2. Data management
As a highly dynamic core component of the RAG system, its data demands careful management. Data entering it
needs to be checked to upkeep the system’s performance. Shifts in the data distribution and /or system usage patterns
8

might also impact the system’s end-to-end performance, requiring evaluations of the RAG system’s responses. These
evaluations call for a test data set, whose coverage needs checking as well.
4.2.1. Ingestion
Data can be ingested from diverse sources such as file systems, databases, APIs to web services, the web, real-
time feeds, human experts, etc. Data can be structured or unstructured. It exists in text-based formats (e.g., JSON,
Markdown), in document formats (e.g., PDF, .doc), in image formats (e.g., JPEG, GIF), as structured data (e.g.,
relational SQL-databases, knowledge graphs, graph databases), as audio (e.g., MP3) and video (e.g., MOV) files, etc.
Depending on the nature of the data, methods to automatically pull in the data can be chosen from webhook
triggers, change data capture (CDC), polling mechanisms, file system watchers, stream processing, web crawlers, etc.
Manual inputs (e.g., continuous learning) are equally possible. Most of these methods can also be implemented in
push rather than pull mode. The frequency of these data transfers ranges from near-continuous to infrequent. Note
that, in our context, ingestion also includes deletions and updates of data.
Additionally, ingestion includes necessary format conversions, such as extracting text from PDFs, generating
descriptions for images, or transcribing audio, ensuring that data is ready for verification and later processing.
4.2.2. Verification
Once the data is ingested, it must be systematically verified to remain accurate and consistent. This verification
process contains multiple checks:
•Quality The data should be well-structured and not contain corrupted or malformed content. Where relevant,
the correctness of the data can be verified through external validation.
•Completeness All expected fields, including metadata (e.g., timestamps, authorship, source), should be present
and contain values. Texts should not be truncated.
•Recency Older versions should not be retained, but archived and replaced by more recent versions. Data recency
is evaluated using timestamps from the metadata, embedded timestamps, or semantic analysis.
•Consistency Contradictory information should be detected, e.g., by using semantic similarity checks (embed-
dings, cosine similarity). To resolve conflicts, the data sources can be weighted for trustworthiness and /or
human intervention can be solicited.
•Uniqueness Duplicates should be eliminated, for example with hash-based deduplication. To discover semantic
redundancies, the system can calculate embeddings followed by clustering. Term frequency-inverse document
frequency (TF-IDF) and Latent Dirichlet Allocation (LDA) can be used to identify documents discussing similar
topics.
External sources to which the retriever has direct API access pose a special challenge. In most cases, the checks
described above would lead to unacceptable latencies, as these external sources are consulted in real time. Retrieval
guardrails, however, can help to enhance veracity.
4.2.3. Updating data lake
This phase is about storing the previously ingested and verified data e fficiently in a data lake. In organizations
where multiple RAG systems or applications wrapped around RAG systems share overlapping data sources, main-
taining a separate data lake (see Figure 2) before chunking and embedding in the “retrieval sources” component
offers several advantages. By centralizing the ingested and verified data, the data lake ensures consistency, reduces
redundancy, and provides a single source of truth across systems. This architecture enables di fferent RAG systems,
or other downstream applications, to access pre-validated data without duplicating ingestion and verification e fforts.
Additionally, it supports better data governance, facilitates auditing, and allows for easier updates across multiple
systems.
In this stage, the updates and deletions identified in the previous verification phase are now applied to the data
lake. This includes removing duplicates, replacing outdated records, and correcting errors. Instead of outright dele-
tion, archiving old versions allows for rollback if needed and helps maintain a historical record of changes. Each
9

modification must be versioned to maintain a clear history of changes, to allow the system to revert to earlier versions
if necessary. When updates are applied, the data must remain logically sound (e.g., free of orphans). This means
that when new or updated data is added, any related information should align correctly, and previous references to
outdated data should be adjusted accordingly. Additionally, access control metadata should be stored alongside each
data entry, ensuring retrieval queries respect access permissions and enforce role-based restrictions where necessary.
4.2.4. Updating retrieval sources
Next, the data changes are propagated to the retrieval sources block in the pipeline, where incremental chunking
and embedding are performed on newly added or modified data. Indexing is also updated accordingly, ensuring that
deletions and outdated information are properly removed from the retrieval process.
4.2.5. O ffline testing
The o ffline testing of the RAG system can be triggered by RAG-pipeline changes or data distribution shifts. Fur-
ther to the initial setup, RAG pipeline changes concern modifications of the pipeline components, e.g., switching to
another LLM (version), adapting the query generator, adopting another chunking method, or swapping the embed-
ding algorithm. Data distribution shifts can happen when new data sources from other domains are attached to the
system, e.g. an API to a new database, an additional set of URLs to the web crawler, or the inclusion of another
language. Gradual, latent data distribution shift will happen over time as well and should be monitored. The tracing
of these changes relates to and supports the quality attributes laid out in Section 5.1, especially monitorability and
observability.
We propose a multi-layered platform for o ffline testing (as well as live testing, see Section 4.2.6) which happens
at three levels of granularity: the module (e.g., chunking), the component (e.g., retrieval), and the system level (end-
to-end). After unit testing a module, its e ffect on the component, and end-to-end system are tested as well. Each level
of testing granularity requires di fferent test data and metrics. Metric selection should align with end-user priorities.
Table 1 illustrates the testing platform with the example of embedding. It di fferentiates between traditional DevOps
testing and specific RAG-testing. For a detailed discussion on metrics, refer to existing work [28].
Table 1: Example of usage of the three-level RAG testing platform. Module-level embedding a ffects both the “retrieval” and “retrieval sources”
components, so these should be tested together at the second, component level before testing the end-to-end system.
Level What We Test Test Data Traditional DevOps RAG-Specific
Metrics Metrics
Module : Embed-
ding qualitySimilarity, drift, e ffi-
ciencyBenchmark datasets,
synthetic testsEmbedding generation
time (ms), memory us-
age per embeddingCosine similarity,
embedding drift,
t-SNE /UMAP Cluster-
ing
Component : Re-
trieval qualityRank relevance, re-
silience to noisePublic RAG retrieval
evaluation bench-
marks (BEIR, MS
MARCO), produc-
tion queriesretrieval latency (ms),
Index update speedMean reciprocal rank
(MRR), recall@K,
precision@K, nor-
malized discounted
cumulative gain
(nDCG)
End-to-End Re-
sponse qualityFaithfulness, halluci-
nation, correctnessHuman-labeled, pro-
duction logsFull pipeline latency,
response generation
timeFaithfulness score,
hallucination rate,
BLEU, ROUGE,
BERTScore
The RAG-specific, data-oriented testing calls for domain-specific test datasets. The relevance of these test data sets
must be monitored against actual system usage (see 4.2.7. Coverage checking ). When introducing new data domains,
languages, or formats, the test sets should also be updated or expanded beforehand to reflect these changes. Newly
added test sets ensure the RAG system performs e ffectively on the new data, while existing test sets remain crucial
10

to verify that performance on previously covered domains does not degrade. This dual approach helps maintain both
forward adaptability and backward compatibility.
In addition to these domain-specific test sets, standard benchmarks can also play a valuable role. Widely-used
retrieval and generation benchmarks (e.g., BEIR, MS MARCO) provide a consistent baseline for evaluating pipeline
or data changes, enabling comparisons beyond the organization’s domain-specific test sets. Synthetic data can also
complement testing by enabling controlled simulations of edge cases, rare scenarios, or diverse query variations that
may be underrepresented in the test sets or benchmarks.
RAG systems introduce significant stochasticity in both retrieval (e.g., approximate nearest neighbor search or
ranking tie-breaking) and generation (LLM outputs). This variability makes testing inconsistent, as the same query
may yield di fferent retrieved documents or generated responses across runs, complicating reproducibility and metric
comparison. To mitigate this, fixing random seeds will ensure more consistent and comparable test results.
The o ffline testing phase may result in the exposure of the need for modifications to the RAG system, such as
swapping the LLM, updating prompts, embeddings, etc. Such modifications, in turn, will trigger further testing
before being released and deployed.
4.2.6. Live testing
After a positive evaluation in the o ffline testing phase, the RAG system is released and deployed. To minimize
risk, the following deployment strategies can be considered:
•Shadow testing Running a new version of the pipeline in parallel with the live version without a ffecting users,
to compare outcomes before switching; this approach allows testing both individual components (e.g., retriever
or generator) and the complete pipeline.
•A/B testing Deploying the new system to a small, but representative subset of users while keeping the old
pipeline for others.
•Staged roll-out Deploying the new pipeline incrementally to user subsets.
The deployment marks the start of live testing to achieve the quality attributes (see Section 5.1). The objective of
live testing is to validate the system’s functionality, performance, and user experience in a real-world environment,
identifying issues that may not surface during controlled o ffline testing.
From a data-centric perspective, live testing is a continuous exercise, as gradual changes to the RAG data occur
over time, and usage patterns—such as the types of queries—may shift. Continuous monitoring and iterative ad-
justments are therefore essential to maintain system reliability and relevance. Similar to o ffline testing, live testing
can be conducted at three levels: module-level (testing individual modules, such as query generation or reranking,
in isolation), component-level (evaluating integrated components or sub-parts of the pipeline, such as generation or
retrieval), and end-to-end-level (assessing the entire RAG system under real-world conditions).
Metrics play a central role in live testing to monitor system performance, user experience, and data consistency.
Next to the “traditional” metrics such as latency and throughput (to ensure responsiveness under real-world load),
RAG-specific metrics include retrieval quality (e.g., precision, recall, and relevance of retrieved documents), genera-
tion quality (e.g., response accuracy, coherence, and hallucination rate), and usage analytics (e.g., query distribution
shifts and user engagement). Continuous tracking of these metrics enables timely detection of data drift, performance
degradation, and evolving user needs.
To ensure timely detection of performance degradation, predefined thresholds should be established for key metrics
during live testing. When one or more metrics exceed these thresholds, it signals the need for further investigation
and optimization of the RAG system. This triggers a feedback loop, where the RAG system is revised and then sent
back to the o ffline testing phase before proceeding to the life testing phase again.
4.2.7. Coverage checking
Coverage checking ensures that the test data used during o ffline testing remains representative of the live usage
patterns observed during live testing. Over time, several factors can cause the test data to become outdated or mis-
aligned with actual user interactions. Changes in user behavior, shifts in query distributions, the addition of new data
domains, languages, or content sources, and the introduction of new system components (e.g., a di fferent chunking
11

or embedding method) can all lead to discrepancies. Without adequate coverage checking, the RAG system may per-
form well on outdated test sets but fail to meet real-world user needs, risking degraded retrieval accuracy, hallucination
issues, or slower response times.
Coverage checking can be performed at multiple stages of the RAG system to capture di fferent aspects of repre-
sentativeness:
•Query Coverage At the query generation and retrieval stages, comparing live user queries with test set queries is
essential. Techniques such as TF-IDF vector similarity or embedding-based comparisons (e.g., cosine similarity
using query embeddings) can quantify how closely the test set aligns with current usage. Low similarity scores
or newly emerging query clusters (detected via dimensionality reduction methods like t-SNE or UMAP) indicate
gaps in coverage that require test set updates.
•Retrieval Coverage In the retrieval component, coverage checking assesses whether the retrieved documents
for live queries reflect those considered during o ffline testing. If live queries consistently retrieve documents
outside the scope of the test set, this signals that the data has evolved or user needs have shifted. Using retrieval
metrics like recall@K on both test and live queries can help track this drift.
•Generation Coverage At the generation stage, coverage checking examines whether generated responses from
live queries align with the topics, intents, and styles covered in the test set. Variations can be detected through
semantic similarity analysis using embedding models or by applying topic modeling (e.g., LDA) to compare
topic distributions.
•Vocabulary and Domain Coverage Tools like TF-IDF are particularly useful for identifying vocabulary drift.
By comparing term frequencies between test and live data, new or underrepresented terms can be flagged for
inclusion in updated test sets. This approach is beneficial when new data sources or languages are integrated
into the RAG system.
In practice, coverage checking should be continuous and automated where possible, with predefined thresholds
triggering reviews when significant discrepancies arise. For example, if less than 85% of live queries achieve a certain
similarity score to test queries, the system should prompt a test set expansion. By maintaining strong coverage align-
ment, the RAG system ensures both forward adaptability (supporting new user demands) and backward compatibility
(retaining performance on existing use cases).
5. Impact of RAGOps on Quality Attributes
RAGOps encompasses the techniques and mechanisms for managing, monitoring, and optimizing interactions
between LLMs and external data sources and databases, addressing both retrieval and generation processes as well as
the characteristics of the retrieval sources. The design and implementation of RAGOps has impact to the qualities of
RAG systems. This section examines the impact of RAGOps on various qualities, including the factors that influence
these qualities.
5.1. RAG system Qualities
RAGOps aims to establish a CI /CD infrastructure that supports the full lifecycle of RAG, including both the query
processing pipeline and data management as outlined in Section 4. This infrastructure automates the RAG system,
facilitates seamless connections, and implements feedback loops to enhance performance of RAG applications, in-
cluding improved observability and monitorability. Table 2 highlights the quality attributes of a RAG system that must
be considered, along with the corresponding operational design considerations.
Adaptability. Adaptability is a critical quality for managing the rapid technical evolution of the RAG system. As
discussed earlier in Section 4.2.5, changes to the RAG system involve modifications to all its components, such as
switching to a di fferent LLM, adjusting the query generator, adopting alternative chunking methods, or replacing the
embedding algorithm. To ensure adaptability, the RAG pipeline is designed to support automated updates to retrieval
sources whenever raw data changes or performance degradation is detected. Development design choices, such as the
method of knowledge graph construction and the granularity of retrieval units, influence the e fficiency of adaptation.
12

Table 2: RAG system Qualities
Quality Operational Design Consideration
AdaptabilityData structure construction
Granularity configuration
MonitorabilityPerformance degradation detection: Accuracy, Latency, Relevance
Data source change detection
Clear, consistent, and meaningful performance metrics
ObservabilityRetrieval: Injection attack detection, Relevance
Generation: Context adherence, Toxicity,Usefulness
TraceabilityVersioning
Retrieval routing
Grounded source
ReliabilityResource usage
Bandwidth
Monitorability. Monitorability is essential for maintaining the performance of retrieval sources. It requires
components capable of detecting performance degradation and changes in the data sources used by these retrieval
sources. Clear, consistent, and meaningful metrics must be defined to assess performance and utility.
Observability. RAG systems require comprehensive observability, including their operational functionality, to
ensure reliability.An observability infrastructure layer is essential and must span across all components. This layer
logs all queries, responses, and the inputs and outputs of each component, providing comprehensive visibility into the
application’s operations and enhancing monitoring and evaluation capabilities.
The retriever’s functionality primarily relies on a series of query tools designed to retrieve information from ex-
ternal sources in various data structures and formats without modifying their states. Beyond performance metrics
such as latency and relevance, observability must also detect potential injection attacks that could potentially compro-
mise retrieval sources. For instance, when retrieving tabular data, the SQL queries generated by the LLM should be
thoroughly validated to prevent SQL injection.
After retrieval, the generation functionality adjusts the context retrieved from external sources before sending it
to the LLM model to answer users. Similarly to the retriever, the generator needs to monitor metrics in addition to
performance, like toxicity and context adherence.
Traceability. From a data-centric perspective, traceability is paramount in a dynamic environment where external
data evolves continuously. As data sources change, versioning is required to capture di fferences between updates
of retrieval resources. Input and output data of each component, primarily the retriever (e.g., linking user queries
to retrieved documents, embedding IDs, and any filters or reranking applied) and generator (e.g., reconstructing the
prompt for generation, including the user query and retrieved context), along with their structures, should also be
thoroughly documented to maintain transparency and reproducibility across experiments and production workflows.
Such traceability information must be securely recorded [29] to facilitate debugging and to identify issues related
to specific data versions or retrieval source updates. This approach ensures the reliability and accountability of the
RAG system over time.
Reliability. The reliability of RAG systems can be a ffected by unpredictable factors such as bandwidth usage and
operational costs. For instance, when users upload large documents, it can impact system performance. RAGOps is
required to e ffective manage such operational challenges to ensure reliability of RAG system. Operational strategies
need to be implemented, such as load balancing, auto scaling, and prioritization mechanisms to ensure consistent
performance even under varying workloads.
5.2. Human-in-the-Retrieval
Incorporating human-in-the-loop in the RAG process necessitates redesigning and extending the RAG system to
effectively integrate valuable but fragmented human feedback. From an operational perspective, it is also crucial to
13

support the continuous incorporation of accumulated human feedback into passive retrieval sources, which influences
reasoning and decision-making processes. Tracing updates in human feedback and assessing their impact on the RAG
system introduces additional complexity.
Adaptability. There are two key design considerations for storing human feedback as a passive retrieval source
for adaptation at runtime. The first involves deciding whether to integrate fragmented tacit knowledge into existing
domain knowledge or to store it separately. Human-contributed intermediate knowledge can be merged into existing
sources, such as defining the semantic meaning of specific data columns in a tabular structure or adding missing
relationships in a knowledge graph. However, if the feedback contains confidential or sensitive information—such
as unpublished ideas in scientific contexts—it may be more appropriate to store it separately as a distinct retrieval
source. The second consideration concerns the format or data structure used to store the feedback, particularly when
it is kept separate. Depending on the nature and volume of the feedback, it may be embedded in a vector database or
organized within a knowledge graph.
Observability. It is crucial to consistently monitor human feedback and evaluate its impact on the ongoing RAG
process. These assessments are essential for maintaining the overall quality of RAG, particularly when tacit knowl-
edge from human experts has not yet been integrated into existing passive retrieval sources. To enhance real-world
applicability and reliability, the testing and evaluation capabilities of RAGOps must be extended with a comprehensive
suite of test datasets and the incorporation of user feedback into evaluation cycles.
Traceability. Traceability includes tracking human contributions as inputs to each component, as well as the
outputs produced by each component. This includes the user’s input prompt, intermediate outputs generated for
querying retrieval sources, feedback provided by human experts, and the generator’s final output.
6. Overarching RAGOps Challenges
6.1. Responsible and Safe AI
As with any LLM compound system, RAG applications must adhere to responsible AI guidelines and legal frame-
works to ensure responsible AI use [30], and the same applies to RAGOps. This adherence involves several critical
levels:
First is regulatory compliance. RAGOps must align with regulations such as GDPR and the EU AI Act, along
with their codes of practice. These codes articulate commitments, measures, and key performance indicators (KPIs)
for stakeholders across the AI supply chain. These measures are applicable and extendable to RAGOps. For instance,
the transparency information required by regulations can be automated through RAGOps, streamlining compliance
processes.
Second is alignment with established standards. Compliance with AI safety standards6is also essential. Certain
voluntary guardrails directly influence RAGOps design, such as: “Test AI models and systems to evaluate model
performance and monitor the system once deployed. ” , and “Keep and maintain records to allow third parties to
assess compliance with guardrails. ” .
Third is adherence to responsible AI principles. RAG applications should incorporate high-level responsible AI
principles7, including fairness, accountability, transparency, and inclusivity. From the RAGOps perspective, emphasis
shifts toward the automated testing, evaluation, and monitoring of these principles, ensuring they are upheld through-
out the system’s lifecycle.
To implement responsible AI principles and standards, it is necessary to build responsible /safe-AI-by-design [31]
into the RAG applications through design patterns and tactics. For example, a black box recorder can be designed to
collect critical data of the RAG process at runtime for observability.
6“The 10 guardrails,” V oluntary AI Safety Standard, Australian Government Department of Industry, Science and Resources, accessed 5 April
2025, https://www.industry.gov.au/publications/voluntary-ai-safety-standard/10-guardrails
7“Australia’s AI Ethics Principles,” Australian Government Department of Industry, Science and Resources, accessed 5 April 2025,
https://www.industry.gov.au/publications/australias-artificial-intelligence-ethics-principles/australias-ai-
ethics-principles
14

6.2. Lack of Standard Evaluation Metrics
The absence of standardized evaluation metrics across various functionalities, including retrieval resources, re-
trieval processes, and generation, creates significant challenges in accurately assessing each of the components and
the RAG application as a whole [3]. This lack of uniformity hinders the ability to gain a comprehensive understand-
ing of the system’s overall performance. While various prompting techniques have been developed to enhance RAG
performance, identifying the most e ffective approach for specific retrieval sources remains a complex and unresolved
challenge.
The design decisions made during the early stages, such as chunking strategies and the choice of embedding
models, influence subsequent stages of the data lifecycle and RAGOps. Defining metrics that can e ffectively assess
the impact of these design choices presents a considerable challenge.
6.3. Continuous Improvement Through Observability
As RAGOps systems evolve continuously, observability is required to monitor, assess, and enhance their opera-
tions. Observability provides insights into system performance, data quality, and user interactions, which indicates
iterative updates and refinements. Decoupling and articulating various aspects of monitorability, including data qual-
ity, model quality, and software quality, is challenging. It is crucial to determine the most appropriate monitoring
techniques for each aspect to ensure comprehensive oversight of the RAG application. While existing techniques
address monitoring for individual RAG components, ensuring seamless monitoring of the entire RAG pipeline and its
interactions with the RAG system is critical. Monitoring changes in data sources and understanding their impact on
the RAG system add further complexity.
Key challenges in observability include identifying components responsible for detecting performance degrada-
tion and designing their interfaces within the system. An important aspect of RAG system is the need to continuously
integrate user input and feedback into retrieval process and future performance improvement. This feedback loop
can significantly influence observability requirements for individual RAG components, potentially driving new ar-
chitectural decisions in future iterations. Moreover, extending existing monitoring tactics beyond typical concerns
like performance and resource consumption is essential to address the unique characteristics of RAG system, like
relevance, groundedness and factuality.
6.4. Multimodality
In real-world application scenarios, such as healthcare, data are inherently multimodal with various types, scales,
and formats. For instance, medical records often combine structured tabular data, imaging, and unstructured text,
each providing unique insights into patient care. While multimodality is considered a critical element of intelligence
and a cornerstone for achieving comprehensive understanding, RAG applications and frameworks with multimodal
capabilities are still emerging. Most existing solutions operate with fixed source and target modalities, such as text-to-
image, image-to-text, or image-text pair to image retrieval. Some approaches [32, 33, 34] attempt to unify multimodal
sources into text, leveraging language models to generate responses. However, this conversion process may lead to
the loss of critical information.
Multimodal RAG systems can outperform single-modality models by integrating and analyzing information from
various sources. This integration enhances tasks such as anomaly detection, where combining modalities can lead to
richer and more robust insights than relying on a single data type.
6.5. Human-in-the-Retrieval
A major challenge in integrating humans into the retrieval process lies in determining when their input is nec-
essary [35]. The aim is to involve humans selectively and strategically, particularly in cases involving critical infor-
mation, edge cases, closely ranked outputs, or high uncertainty in AI-driven steps. This adds a layer of necessity
assessment to the RAG workflow, e ffectively positioning human expertise as a dynamic and adaptive retrieval re-
source. Additionally, tools for optimizing human inputs may be required to refine their feedback, converting it into
meaningful intermediate contributions that enhance the retrieval process.
Another challenge is extending human-in-the-loop functionality to other components of the RAG application be-
yond retrieval, such as orchestration and generation. This becomes particularly relevant in high-uncertainty scenarios
15

where expert intervention is needed to assess workflow quality, evaluate retrieved contexts, and validate query formu-
lations. Human corrections and adjustments based on these evaluations play a crucial role in improving the overall
effectiveness and accuracy of both the retrieval and generation stages.
7. Use Cases
7.1. Taxation Assistant
We collaborated on a project with a local AI startup, which recently launched an LLM-based taxation assistant
aimed at helping tax professionals analyze complex tax scenarios. The application provides instant answers with
explanations, integrates real-time updates from government service departments, and performs automatic reference
checks. As part of the project, we assessed their system design and enhanced their application by incorporating an
RAGOps infrastructure.
7.1.1. RAG Design and Development
The initial design decisions on retrieval sources mainly involved selecting the approach for incorporating domain-
specific taxation data, specifically the open data provided by the government department, including topics andtypes .
Topics serve as a fixed data source, while types are organized according to the taxation /law hierarchy, which are up-
dated on a weekly basis. A web crawler is used to collect data from the government website, with crawling scheduled
weekly. Leveraging authorized open data from government significantly reduces the likelihood of hallucinations. Us-
ing RAG to dynamically query the data ensures that any link to the government data can easily be removed by the
company if needed.
The company made a decision between using embedding or knowledge graph to capture the authoritative data from
government. To construct the retrieval source, plain text is stored in a vector database, as the primary functionality
is to identify relevant information. In addition, the focus is specifically on taxation regulations without complex
dependencies on other financial legislation, thus, using a knowledge graph was unnecessarily heavy and not cost-
efficient for this case.
When determining granularity and metadata, the company chose to define each question and answer under a private
ruling (found on a single web page) as a data chunk, which is then constructed as an embedding. The size of these
chunks was carefully considered, balancing the trade-o ffbetween cost and e fficiency — for instance, whether a chunk
should represent an entire web page or just a section of it. Metadata selection was primarily driven by the scenarios
in which tax professionals would use the assistant. For specific taxation cases, metadata includes the decision and the
reasoning behind it. For most other documents, the metadata consists of a summary of the document.
For the retriever , The company implemented a classifier to categorize users’ queries into di fferent types of scenar-
ios, and utilizes scenario-specific prompt templates to help the LLM better understand and interpret queries. A query
generator leverages the output of these templates to transform the query into a vector representation, which is then
compared against vectors of stored chunks in the vector database. This process retrieves two types of information:
rulings andlegislation .
As the retriever provides an initial list of ranked results to the generator, the company introduced a reranker to
further assess their relevance. This reranker employs an LLM-based evaluator to refine the rankings. A privacy
guardrail was also built within the retriever to desensitize sensitive user data before processing queries, ensuring
privacy and security throughout the retrieval process.
Thegenerator has a context constructor that utilizes response templates to structure recommendations, enhanc-
ing readability and helping tax professionals understand the reasoning behind the generated responses. Another key
decision was to incorporate user profile, which can be created by having users input their information or by continu-
ously learning and updating the profile based on interactions. The generator uses the information to refine the output,
ensuring personalized and relevant results.
7.1.2. RAGOps
During the design phase of the taxation assistant, three quality attributes were identified that are partially addressed
and considered from operational perspectives:
•Adaptability: Incorporating user feedback into the assistant to improve future recommendation generation.
16

User story : A tax professional or a runtime evaluator identifies and submits feedback pointing out a mistake
in a recommendation on capital gains tax. The tax assistant integrates the feedback into its agent memory and
adapts its reasoning logic for similar future scenarios involving capital gains tax.
•Observability: Enabling stakeholders to track historical queries, monitor user feedback, and receive alerts on
assistant health, performance or users complaints.
User story : An alert is triggered due to an increasing number of low scores provided by the assistant user about
recommendation on small business tax deductions. The tax assistant automatically logs all queries, responses,
and user feedback.
•Contestability: Allowing users to challenge the assistant’s recommendation and submit feedback for review.
User story : A tax professional disagrees with the recommendation provided on superannuation contributions
and submits feedback to challenge the recommendation. The tax assistant logs the feedback and flags the case
for review by tax experts.
To support and fulfill these quality requirements, we introduce an RAGOps infrastructure layer that cross-cuts
all RAG components, including retrieval sources, retriever and generator. This layer logs all queries, responses,
user feedback, and the input and output of each component, providing comprehensive visibility into the assistant’s
operations and improving monitoring and evaluation capabilities.
7.1.3. Data management
For testing and evaluation, we compiled a set of ground truth data from questions and corresponding certified
replies available on the national tax o ffice’s open forum. For complex scenarios, we initially gathered ground truth
data directly from the company’s domain experts. Subsequently, we used the LLM to generate additional ground truth
data for evaluation purposes.
7.2. Magda Copilot
Magda8is a data catalog system that serves as a centralized hub for cataloging an organization’s data. It pro-
vides open-source software to assist organizations or government agencies in managing data tasks such as collection,
authoring, discovery, usage, sharing across organizations, or publishing to open data portals.
In scientific contexts, our internal scientists from various domains utilize the open data available on the platform.
Within a multidisciplinary project, we collaborate closely with scientists in agriculture and biology to explore and
understand their requirements for using LLMs with domain-specific data sources via RAG to address their daily
scientific questions. From these observations and insights, we developed the Magda Copilot, a tool designed to help
scientists explore and discover data on the Magda platform.
7.2.1. RAG Design and Development
The Magda Copilot is designed to autonomously integrate with scientific tools commonly used by researchers.
The first stage incorporates five primary tools:
•Basic info tool: the tool retrieves basic information about proteins from UniProt database9.
•Sequence similarity tool: the tool retrieves proteins with similar sequences using a BLAST (Basic Local Align-
ment Search Tool)10search based on a UniProt accession code.
•Biochemical characteristics tool: the tool can generate SQL queries to retrieve information from an internal
tabular dataset about the biochemical characteristics of proteins.
8“Magda data catalog ,” CSIRO, accessed 5 April 2025, https://www.csiro.au/en/research/technology-space/cyber/Magda-
data-catalog
9UniProt, accessed 5 April 2025, https://www.uniprot.org
10“Basic Local Alignment Search Tool”, National Center for Biotechnology Information, accessed 5 April 2025, https://blast.ncbi.nlm.
nih.gov/Blast.cgi
17

•Structure tool: the tool utilizes AlphaFold Protein Structure Database to retrieve and display structural informa-
tion about proteins based on UniProt accession code.
•Causal graph discovery tool: the tool extracts causal relationships from scientific literature stored as a repository
of pdf files.
A sample and typical scenario involves identifying enzymes that play a role in breaking down polyethylene tereph-
thalate (PET). For example, the process begins with a known enzyme with UniProt code A0A0K8P6T7, which is
already recognized for PET degradation. Its basic information is retrieved using its UniProt code. Next, the sequence
similarity tool is used to find enzymes having similar sequences. The basic information and the biochemical char-
acteristics are then queried to check whether each candidate enzyme is active on PET under certain conditions. As
such information can be incomplete in the data, the structure information of these enzymes are used to further iden-
tify whether active sites exist that are likely to degrade PET. These capabilities are further enhanced by the causal
graph discovery tool which visualize a causal graph of potential factors involved in the PET degradation activities to
complement insights gained from sequence and structural analysis. The Magda Copilot is able to autonomously exe-
cute the workflow, and after identifying several candidates, they are ultimately validated through experimental studies
conducted by researchers.
The retriever in the system routes user queries to corresponding tools by identifying and decomposing user’s
query into atomic queries. It generates prompts tailored to the five tools mentioned. The generator then synthesizes
the final output based on results returned by these tools, providing customization applied to handle domain-specific
terminology and contextual accuracy. Additionally, Magda Copilot and its tools are wrapped as a suite of REST API
services, enabling straightforward integration with other platforms.
7.2.2. RAGOps
Through the multidisciplinary project, we observed that when domain experts use RAG systems, their questions
often lack su fficient context. Furthermore, domain experts frequently possess tacit knowledge — implicit understand-
ing that is challenging to articulate or extract. This tacit knowledge cannot easily be captured by passive retrieval
sources through written or verbal communication, However, it can be integrated as external data to enhance retrieval
quality or guide the workflow executed by Magda Copilot. Consequently, interactivity emerges as a key quality at-
tribute. In addition, we identified three other critical quality attributes that need to be addressed from an operational
perspective:
•Interactivity: The application should facilitate e ffective user interaction, enabling the Magda Copilot to engage
end users for further information or feedback when needed.
•Observability: When user queries are processed through a series of tools, potentially incorporating tacit knowl-
edge, each step impacts the final result and the overall performance of the retrieval process. Monitoring and
observing the performance of each step and the end-to-end process is essential.
•Traceability: In scientific discovery scenarios, the ability to trace the query history is crucial. This includes
logging the tools selected, as well as the input and output of each step. Comprehensive logging of all the
actions ensures transparency and reproducibility.
•Adaptability: Gradually incorporating tacit knowledge collected during interactions into the passive retrieval
sources is important for improving retrieval quality over time. Moreover, more data sources or tools may be
further integrated into the system, requiring the retrieval process to adapt accordingly.
7.2.3. Data management
To meet these quality requirements, the operational design of Magda Copilot incorporates specific functionalities.
The structure tool retrieves the 3D structure information of proteins from AlphaFold protein structure database. When
the proteins of interests are not included in the database, the system will run the structure prediction using AlphaFold
and store the results locally for future use. By leveraging both online and local databases, the structure tool ensures
seamless integration and accessibility of new structure data as it becomes available. This design enables e fficient and
comprehensive capture of results, enhancing the overall utility and adaptability.
18

Other data sources are continuously expanded to enhance the capabilities of Magda Copilot, and the data within
each source is also continuously enriched to incorporate the latest scientific research progress. This may introduce
new forms of data, and the retrieval process must adapt to accommodate these changes e ffectively. The performance
of the Magda Copilot is assessed using a combination of quantitative and qualitative methods. Feedback from domain
scientists play a critical role in validating the relevance and accuracy of retrieved and generated outputs. Additionally,
user satisfaction from the domain scientists capture provide valuable insights into usability and e ffectiveness from the
end user’s perspective.
Tacit Knowledge Management : Domain scientists play a key role in refining retrieval and generation outputs by pro-
viding invaluable feedback and contributing to iterative improvement processes. Their expertise helps uncover tacit
knowledge that is not captured through traditional documentation. Tacit knowledge can be extracted from the inter-
action with domain experts, such as the correct workflow for analyzing specific data, the e ffective way for conducting
data search, and the reasoning behind these approaches. This knowledge gradually evolves into a new data source
that guides Magda Copilot in executing research workflows. Given the varied granularity and diverse perspectives
of this knowledge, iterative distillation is necessary to refine and structure it into a usable form. By leveraging ex-
pert knowledge and automating complex processes automatically, Magda Copilot has assisted the discovery of new
research insights.
8. Summary and Future Work
Retrieval-Augmented Generation (RAG) has emerged as a promising solution to address key challenges faced by
LLMs (Large Language Models), such as hallucination, outdated or non-removable parametric knowledge, and non-
traceable reasoning processes. Current LLMOps tools predominantly focus on model management, o ffering limited
support for data-related aspects, particularly the data retrieved after an LLM is deployed.
This paper conceptualizes RAGOps, which builds upon LLMOps by emphasizing robust data management to
address the dynamic nature of external data sources. This extension necessitates automated evaluation and testing
methods to improve data operations, ensuring enhanced retrieval relevance and generation quality. This paper char-
acterizes the generic architecture of RAG applications using the 4 +1 model view of software architectures, outlines
the integrated lifecycle of query processing pipeline by combining LLM and data management lifecycles, defines the
key design considerations and corresponding quaility tradeo ffs of RAGOps across di fferent stages, identifies research
challenges associated with each stage, and presents two practical use cases of RAG applications, o ffering valuable
insights for advancing the operationalization of RAG applications.
Several directions remain open for future work. The verification phase (Section 4.2.2) of the knowledge base
maintenance lacks a mechanism to detect the malicious insertion of detrimental data or other forms of attacks. Spe-
cific semantic checks complementing classical cybersecurity measures constitute an avenue for future research. The
applicability of recent developments in agentic approaches, which could equip RAG systems with extensive auton-
omy, also merits further investigation. Another direction is the development of tooling support tailored to RAGOps.
Dedicated tools are needed to assist developers in monitoring data retrieval behaviors, tracing generated outputs to
their sources, and evaluating the performance impact of data changes. Such tooling can significantly lower the barrier
to adoption and foster best practices in managing the full lifecycle of RAG systems.
References
[1] M. Zaharia, O. Khattab, L. Chen, J. Q. Davis, H. Miller, C. Potts, J. Zou, M. Carbin, J. Frankle, N. Rao,
A. Ghodsi, The shift from models to compound AI systems, accessed: 5 April 2025 (2024).
URL https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel,
et al., Retrieval-augmented generation for knowledge-intensive NLP tasks, Advances in Neural Information
Processing Systems 33 (2020) 9459–9474.
[3] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang, H. Wang, Retrieval-augmented
generation for large language models: A survey (2024). arXiv:2312.10997.
URL https://arxiv.org/abs/2312.10997
19

[4] D. Zhang, B. Xia, Y . Liu, X. Xu, T. Hoang, Z. Xing, M. Staples, Q. Lu, L. Zhu, Privacy and copyright protection
in generative AI: A lifecycle perspective, in: Proceedings of the IEEE /ACM 3rd International Conference on AI
Engineering - Software Engineering for AI, CAIN ’24, Association for Computing Machinery, New York, NY ,
USA, 2024, p. 92–97. doi:10.1145 /3644815.3644952.
URL https://doi.org/10.1145/3644815.3644952
[5] J. Diaz-De-Arcaya, J. López-De-Armentia, R. Miñón, I. L. Ojanguren, A. I. Torre-Bastida, Large language
model operations (LLMOps): Definition, challenges, and lifecycle management, in: 2024 9th International
Conference on Smart and Sustainable Technologies (SpliTech), IEEE, 2024, pp. 1–4.
[6] E. Laaksonen, LLMOps: MLOps for large language models, accessed: 5 April 2025 (2023).
URL https://valohai.com/blog/llmops/
[7] L. Dong, Q. Lu, L. Zhu, AgentOps: Enabling observability of LLM agents (2024). arXiv:2411.05285.
URL https://arxiv.org/abs/2411.05285
[8] C. Majors, Observability: A manifesto, accessed: 5 April 2025 (2018).
URL https://www.honeycomb.io/blog/observability-a-manifesto
[9] G. A. Lewis, I. Ozkaya, X. Xu, Software architecture challenges for ML systems, in: 2021 IEEE International
Conference on Software Maintenance and Evolution (ICSME), IEEE, 2021, pp. 634–638.
[10] M. Steidl, M. Felderer, R. Ramler, The pipeline for the continuous development of artificial intelligence
models—current state of research and practice, Journal of Systems and Software 199 (2023) 111615.
doi:https: //doi.org /10.1016 /j.jss.2023.111615.
URL https://www.sciencedirect.com/science/article/pii/S0164121223000109
[11] S. Shankar, A. Parameswaran, Towards observability for production machine learning pipelines (2022).
arXiv:2108.13557.
URL https://arxiv.org/abs/2108.13557
[12] I. Gorton, F. Khomh, V . Lenarduzzi, C. Menghi, D. Roman, Software Architectures for AI Systems: State of
Practice and Challenges, Springer Nature Switzerland, Cham, 2023, pp. 25–39.
[13] L. Bass, Q. Lu, I. Weber, L. Zhu, Engineering AI Systems: Architecture and DevOps Essentials, Addison-
Wesley, 2025.
[14] Y . Hu, Y . Lu, RAG and RAU: A survey on retrieval-augmented language model in natural language processing
(2024). arXiv:2404.19543.
URL https://arxiv.org/abs/2404.19543
[15] X. Wang, Z. Wang, X. Gao, F. Zhang, Y . Wu, Z. Xu, T. Shi, Z. Wang, S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng,
X. Huang, Searching for best practices in retrieval-augmented generation (2024). arXiv:2407.01219.
URL https://arxiv.org/abs/2407.01219
[16] P. Kruntchen, Architectural blueprints–the“4 +1” view model of software architecture, IEEE software 12 (6)
(1995) 42–50.
[17] M. Fowler, Patterns for generative AI, accessed: 5 April 2025 (2024).
URL https://martinfowler.com/articles/gen-ai-patterns/#rag
[18] L. Yee, R. Roberts, M. Pometti, S. Xu, What are AI guardrails?, McKinsey & CompanyAccessed: 5 April 2025
(November 2024).
URL https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-are-ai-
guardrails
20

[19] A. team, Introducing the model context protocol, accessed: 27 April 2025 (2025).
URL https://www.anthropic.com/news/model-context-protocol
[20] P. Radhakrishnan, J. Chen, B. Xu, P. Ramaswami, H. Pho, A. Olmos, J. Manyika, R. Guha, Knowing when to
ask–bridging large language models and data, arXiv preprint arXiv:2409.13741 (2024).
[21] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y . Geng, F. Fu, L. Yang, W. Zhang, J. Jiang, B. Cui, Retrieval-augmented
generation for AI-generated content: A survey (2024). arXiv:2402.19473.
URL https://arxiv.org/abs/2402.19473
[22] M. Shamsujjoha, Q. Lu, D. Zhao, L. Zhu, A taxonomy of multi-layered runtime guardrails for designing founda-
tion model-based agents: Swiss cheese model for AI safety by design, arXiv preprint arXiv:2408.02205 (2024).
[23] Q. Lu, D. Zhao, Y . Liu, H. Zhang, L. Zhu, X. Xu, A. Shi, T. Tan, Evaluating the architecture of large language
model-based agents, SSRN (2024).
URL http://dx.doi.org/10.2139/ssrn.5017297
[24] S. Ganju, Develop secure, reliable medical apps with RAG and NVIDIA NeMo guardrails, NVIDIA Technical
BlogAccessed: 5 April 2025 (May 2024).
URL https://developer.nvidia.com/blog/develop-secure-reliable-medical-apps-with-rag-
and-nvidia-nemo-guardrails/
[25] J. Smith, E. Johnson, D. Lee, Securing retrieval-augmented generation pipelines: A comprehensive framework,
Journal of Computer Science and Technology Studies 12 (1) (2025) 45–62.
[26] R. Pedro, D. Castro, P. Carreira, N. Santos, From prompt injections to SQL injection attacks: How protected is
your LLM-integrated web application? (2023). arXiv:2308.01990.
URL https://arxiv.org/abs/2308.01990
[27] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, J. Larson, From local to global: A graph
RAG approach to query-focused summarization (2024). arXiv:2404.16130.
URL https://arxiv.org/abs/2404.16130
[28] P. Y .-C. Chang, B. Pflugfelder, Retrieval-augmented generation realized: Strategic & technical insights for
industrial applications, Whitepaper, appliedAI Initiative (June 2024).
URL https://www.appliedai.de/uploads/files/retrieval-augmented-generation-realized/
AppliedAI_White_Paper_Retrieval-augmented-Generation-Realized_FINAL_20240618.pdf
[29] Y . Liu, D. Zhang, B. Xia, J. Anticev, T. Adebayo, Z. Xing, M. Machao, Blockchain-enabled accountability in
data supply chain: A data bill of materials approach, in: 2024 IEEE International Conference on Blockchain
(Blockchain), IEEE, 2024, pp. 557–562.
[30] Q. Lu, L. Zhu, J. Whittle, X. Xu, Responsible AI: Best Practices for Creating Trustworthy AI Systems, Addison-
Wesley, 2024.
[31] Q. Lu, L. Zhu, X. Xu, J. Whittle, Responsible-AI-by-design: A pattern collection for designing responsible
artificial intelligence systems, IEEE Software 40 (3) (2023) 63–71.
[32] Q. Z. Lim, C. P. Lee, K. M. Lim, A. K. Samingan, UniRaG: Unification, retrieval, and generation for multimodal
question answering with pre-trained language models, IEEE Access 12 (2024) 71505–71519.
[33] B. Yu, C. Fu, H. Yu, F. Huang, Y . Li, Unified language representation for question answering over text, tables,
and images, arXiv preprint arXiv:2306.16762 (2023).
[34] W. Liu, F. Lei, T. Luo, J. Lei, S. He, J. Zhao, K. Liu, MMHQA-ICL: Multimodal in-context learning for hybrid
question answering over text, tables and images, arXiv preprint arXiv:2309.04790 (2023).
[35] X. Xu, D. Zhang, Q. Liu, Q. Lu, L. Zhu, Agentic RAG with human-in-the-retrieval, in: 4th International Work-
shop on Software Architecture and Machine Learning, IEEE, 2025.
21