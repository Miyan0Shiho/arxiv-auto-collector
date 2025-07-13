# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang

**Published**: 2025-07-07 06:57:34

**PDF URL**: [http://arxiv.org/pdf/2507.04706v1](http://arxiv.org/pdf/2507.04706v1)

## Abstract
Urban general intelligence (UGI) refers to the capacity of AI systems to
autonomously perceive, reason, and act within dynamic and complex urban
environments. In this paper, we introduce UrbanMind, a tool-enhanced
retrieval-augmented generation (RAG) framework designed to facilitate UGI.
Central to UrbanMind is a novel architecture based on Continual
Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates
domain-specific knowledge and evolving urban data to support long-term
adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel
optimization framework, where different layers are treated as interdependent
sub-problems. Each layer has distinct objectives and can be optimized either
independently or jointly through a hierarchical learning process. The framework
is highly flexible, supporting both end-to-end training and partial layer-wise
optimization based on resource or deployment constraints. To remain adaptive
under data drift, it is further integrated with an incremental corpus updating
mechanism. Evaluations on real-world urban tasks of a variety of complexity
verify the effectiveness of the proposed framework. This work presents a
promising step toward the realization of general-purpose LLM agents in future
urban environments.

## Full Text


<!-- PDF content starts -->

arXiv:2507.04706v1  [cs.LG]  7 Jul 2025UrbanMind: Towards Urban General Intelligence via
Tool-Enhanced Retrieval-Augmented Generation and
Multilevel Optimization
KAI YANG, Tongji University, China
ZELIN ZHU, Tongji University, China
CHENGTAO JIAN, Tongji University, China
HUI MA, Xinjiang University, China
SHENGJIE ZHAO, Tongji University, China
XIAOZHOU YE, Asiainfo Technologies, China
YE OUYANG, Asiainfo Technologies, China
Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and
act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced
retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel
architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically
incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The
architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers
are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either
independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting
both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To
remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism.
Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed
framework. This work presents a promising step toward the realization of general-purpose LLM agents in
future urban environments.
Additional Key Words and Phrases: Urban Foundation Model, Retrieval-Augmented Generation, Large Lan-
guage Model, Multilevel Optimization, Continual Learning
ACM Reference Format:
Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang. 2018. UrbanMind:
Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization. J. ACM 37, 4, Article 111 (August 2018), 24 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction and Background
1.1 Motivation for Urban General Intelligence
The rapid expansion of urbanization presents not only new opportunities but also significant chal-
lenges for modern cities. Urban environments are inherently dynamic and complex, characterized by
Authors’ Contact Information: Kai Yang, kaiyang@tongji.edu.cn, Tongji University, Shanghai, China; Zelin Zhu, Tongji
University, Shanghai, China; Chengtao Jian, Tongji University, Shanghai, China; Hui Ma, Xinjiang University, Xinjiang,
China; Shengjie Zhao, Tongji University, Shanghai, China; Xiaozhou Ye, Asiainfo Technologies, Beijing, China; Ye Ouyang,
Asiainfo Technologies, Beijing, China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1557-735X/2018/8-ART111
https://doi.org/XXXXXXX.XXXXXXX
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:2 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
the continuous evolution of infrastructures and the frequent occurrence of unpredictable events[ 6].
Traditional AI systems[ 60,69], which are often developed for static tasks, exhibit fundamental
limitations when applied to such non-stationary settings. To achieve Artificial General Intelligence
(AGI), systems must be dynamic. Unlike narrow AI, AGI requires strong generalization and the
ability to adapt across different tasks[ 31]. Recent studies highlight that multimodal understanding
and the continuous adaption are key to building such systems[ 23,49]. In addition, for areas like IoT
AGI must process real-time data and make decisions accordingly[ 16,44,50]. Achieving UGI can
significantly improve the performance of critical urban tasks, including traffic management[ 70],
public safety[57], disaster response[41].
Realizing UGI requires fundamental advancements in urban foundation model[ 91], continual
learning[ 86], dynamic knowledge integration[ 21], and context-aware decision-making[ 63]. Such
capabilities are essential to support the long-term evolution of urban systems toward safer and
smarter environments. Therefore, UGI represents not only a technical advancement but also a
critical step in redefining the role of AI within the fabric of future cities.
Despite its transformative potential, realizing UGI imposes significant challenges. Urban envi-
ronments are characterized by non-stationary data distributions that evolve due to factors such as
seasonal variations[51] and infrastructure developments[1]. Designing AI systems that can adapt
to such changes without catastrophic forgetting remains a major obstacle[ 86]. Additionally, urban
data is inherently heterogeneous and noisy, further complicating reliable knowledge extraction,
reasoning, and decision-making processes[ 83]. Conventional machine learning paradigms, which
assume static training and deployment conditions, are not well suited for the continuous and
adaptive nature of urban environments.
In this paper, we propose a framework called UrbanMind, which leverages a multilevel optimiza-
tion paradigm to jointly address the core requirements essential for realizing UGI.
1.2 RAG and Continual Learning
RAG[ 47] has emerged as an effective meanings for enhancing the reasoning and generation capabil-
ities of LLMs by integrating external knowledge sources. Unlike traditional models that rely solely
on internal parameters to store factual knowledge[ 24,75], RAG dynamically retrieves relevant
information from external corpora to assist in generation. This mechanism allows the model to
remain lightweight while maintaining access to a continually expanding and domain-specific
knowledge base. Recent studies have demonstrated the benefits of RAG in enhancing factual
accuracy and adaptability across a wide range of tasks[ 15,80,82]. However, most existing RAG
systems are designed for static retrieval settings and do not directly address the challenges posed
by non-stationary environments or continual updates to the knowledge source.
RAG also enhances multi-agent collaboration by enabling agents to access external knowledge
in real time. For example, LLMs agents often rely on RAG to retrieve relevant facts from knowledge
graphs or databases. A recent example is CLADD[ 45], RAG enables specialized agents, such as
those analyzing molecular structures or querying knowledge graphs to retrieve and share domain
specific information, allowing the system to generate more accurate and context-aware answers.
Besides, RAG bridges LLMs with tools as retrieved content can guide tool use, and tool outputs can
be reintegrated into the LLM’s context.
Continual learning[ 76] aims to develop models that can incrementally learn from new data
streams while preserving knowledge acquired from previous experiences. It addresses the critical
limitation of traditional machine learning paradigms, where retraining from scratch or fine-tuning
on new data often leads to catastrophic forgetting[ 61]. Various strategies, such as regularization[ 65],
memory replay[ 77], and dynamic architectural expansion[ 29], have been proposed to mitigate
forgetting and support stable learning over time. While continual learning has shown promise
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:3
in areas such as robotics[ 46], natural language processing[ 8] and vision[ 68], most approaches
assume access to well-structured task boundaries and stable data flows. The integration of continual
learning with RAG, particularly under dynamic and evolving data distributions typical of urban
systems, remains an underexplored area.
Updating the knowledge base plays an essential role in maintaining the performance and accu-
racy of RAG systems. It involves two key components: update triggering and the implementation
of update strategies. Update triggering is driven by factors such as timeliness, user feedback, and
system performance metrics. In domains with rapidly evolving information, such as finance and me-
teorology, continuous analysis of user queries and feedback can reveal knowledge gaps—particularly
when there is a high frequency of queries yielding inadequate or irrelevant responses. Additionally,
monitoring system-level indicators[ 71],including retrieval recall, precision, and response satisfac-
tion, can help detect knowledge degradation. A noticeable decline in these metrics often signals
that the underlying knowledge base is outdated or incomplete, necessitating updates. In response,
various updating strategies can be adopted. Automated updates are particularly suitable for struc-
tured data sources with regular update patterns and constrained memory, with mechanisms such
as sliding windows enabling dynamic memory maintenance[ 22]. Manual updates remain crucial
in scenarios requiring domain expertise or professional validation. Moreover, machine learning-
assisted methods offer scalable solutions by analyzing incoming data streams, classifying content,
and identifying novel knowledge elements[ 87]. For example, DR-RAG[ 35] introduces a two-stage
retrieval mechanism that adaptively selects relevant documents based on user queries, providing
an effective approach for maintaining contextual relevance during knowledge base updates.
1.3 Urban Foundational Model
Urban foundational models are large-scale pre-trained models designed to capture the broad distri-
butions and dynamics inherent in urban environments[ 91]. Similar to general-purpose foundation
models in natural language processing and vision, urban foundational models are trained on diverse
multimodal datasets encompassing transportation patterns, public safety reports, environmental
sensor data, land use information, and social behavioral traces[ 4,91]. The objective is to learn gen-
eralizable representations that can support a wide range of downstream urban tasks with minimal
task-specific fine-tuning. By pre-training across multiple domains and modalities, these models
serve as universal foundations for reasoning and decision-making in complex urban systems.
Training urban foundational models presents several significant challenges due to the complexity
and heterogeneity of urban data[ 54]. First, the multimodal nature of urban information ranging
from structured spatial data to unstructured textual reports requires the development of unified
encoding architectures capable of fusing diverse data types effectively. Second, urban datasets often
suffer from noise and missing values which can impair the quality of learned representations and
limit generalization[ 91]. In addition, achieving scalability while maintaining fine-grained temporal
and spatial resolution is computationally intensive, necessitating efficient data management and
training strategies[ 7]. Finally, ensuring that pre-trained models remain adaptable to continual
retrieval and evolving urban contexts introduces additional demands on model regularization
and dynamic fine-tuning capabilities[ 26]. Addressing these challenges plays an essential role in
constructing reliable urban foundational models.
1.4 Tool-Enhanced Retrieval-Augmented Generation
To bridge the gap between foundation models and domain-specific knowledge or real-world actions,
recent research has focused on integrating LLMs with external tools to form LLM-empowered
agents[ 93]. This paradigm endows LLM agents with enhanced capabilities beyond their intrinsic
parameters, offering a variety of benefits.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:4 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
Among various tool calling strategies, Tool-Enhanced Retrieval Augmented Generation rep-
resents a foundational implementation wherein the LLM formulates a query, retrieves relevant
documents, and incorporates the retrieved evidence into its generation process[ 36,48]. Recent
advances further explore integrating RAG with optimizing tool calling[ 64,79],where LLMs call
tools more wisely or efficiently and LLMs generate API-compatible queries for search engines or
domain tools[36], demonstrating superior performance in knowledge-intensive domains.
In urban environments, RAG-based tool calling can provide an effective means for enabling
LLM-agent architectures to address complex and multimodal problems. Built on top of Urban
Foundation Models and Tool-Enhanced Retrieval Augmented Generation, such agents can leverage
a diverse set of external tools, including for example traffic simulators, spatio-temporal databases,
weather forecasting modules, remote sensing APIs, to perceive, reason, and act within the urban
ecosystem.
1.5 Related Work
AGI refers to synthetic intelligence with broad scope and strong generalization capabilities, fun-
damentally different from narrow AI with limited adaptability[ 31]. Recent research has begun
exploring AGI applications across various domains. In education, AGI could enable personalized
and adaptive learning experiences[ 44]; in IoT, AGI is expected to support real-time, context-aware
decision-making beyond current narrow solutions[ 16]. Key capabilities of AGI include multimodal
understanding, interactivity, and personalization are essential for advancing toward more adaptive
and human-aligned AI systems[23, 49, 59].
The application of AGI[ 81] concepts to urban systems is still at an early stage of exploration.
Existing efforts have largely focused on building specialized AI models for distinct urban tasks, such
as traffic management[ 2] and public safety surveillance[ 74]. While these models have demonstrated
strong task-specific performance, they lack the flexibility and cross-domain reasoning capabilities
necessary for true general intelligence. Recent advances in LLMs, reinforcement learning[ 78],
and multi-agent systems[ 34] have opened up new avenues for broader urban decision-making.
However, the majority of current approaches operate under static assumptions and do not address
the challenges imposed by dynamic and evolving urban environments.
Early attempts at integrating continual learning into urban applications have primarily focused
on incremental model retraining without systematic mechanisms for long-term knowledge preserva-
tion or cross-domain reasoning[ 10,72]. Furthermore, the potential of RAG frameworks to enhance
continual learning in urban contexts has not been thoroughly investigated. These limitations
motivate the need for new architectures that combine retrieval-based knowledge integration with
continual adaptation.
Multilevel optimization has recently attracted increasing attention in machine learning due to
its ability to model nested decision processes encountered in applications such as meta-learning,
hyperparameter tuning[ 42] and hierarchical reinforcement learning[ 28]. Classical bilevel optimiza-
tion methods, which optimize an outer objective subject to the solution of an lower-level problem,
have been widely studied and form the basis for many of these developments[ 11,62]. However,
most existing work focuses on settings where task distributions are static and data availability is
assumed to be complete, making the resulting algorithms unsuitable for dynamic environments
like urban systems.
To address evolving data distributions and structural shifts, a few recent studies[ 3,19] have begun
exploring extensions of multilevel optimization to continual and adaptive settings. Nevertheless,
current methods often either assume access to all task information simultaneously or rely on rigid
update schedules that limit their responsiveness to rapid changes. Moreover, the integration of
external knowledge retrieval within multilevel optimization frameworks remains largely unexplored.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:5
These gaps motivate the need for new formulations that can jointly manage retrieval, continual
learning, and model adaptation in a unified multilevel structure.
1.6 Summary of Contributions
This paper proposes a novel framework, UrbanMind, for advancing UGI by integrating retrieval-
based knowledge acquisition with continual learning under a multilevel optimization perspective.
The key contributions are summarized as follows.
•Tool-Enhanced RAG with Continual Learning: We propose UrbanMind, a tool-enhanced
RAG framework tailored for UGI. UrbanMind implements a C-RAG-LLM architecture that
integrates continual learning and tool-augmented reasoning to support dynamic, context-
aware decision-making in complex urban environments. The system can continuously retrieve
domain-specific knowledge and incrementally adapts to evolving data distributions. Moreover,
this framework can be deployed in a cloud-edge distributed manner, supporting efficient
computation, real-time responsiveness, and privacy preservation by processing sensitive data
locally on edge devices.
•Multilevel Optimization with Expert Modularity: We introduce a novel multilevel optimization
formulation for UGI. To the best of our knowledge, such a framework has not been previously
explored in this context. This formulation provides a unified perspective that jointly models
continual retrieval, knowledge integration, and model adaptation, and is closely aligned with
the Mixture-of-Experts (MoE) architecture, where expert modules are selectively optimized at
different levels. The proposed approach enables principled coordination across components,
ensuring stable learning under non-stationary and Out-Of-Distribution (OOD) data distri-
butions in evolving urban environments. Notably, this multilevel optimization framework
is highly flexible, supporting either end-to-end optimization or selective tuning of specific
components based on available computational resources and deployment requirements.
We also implement the proposed UrbanMind and conduct evaluations on real-world urban tasks,
demonstrating that our proposed framework achieves superior performance compared to baseline
approaches.
2 UrbanMind for Urban General Intelligence
2.1 Problem Definition
We consider a dynamic urban environment where the data distribution evolves over time due to
external factors such as infrastructure changes, policy shifts, and societal behaviors. Let X𝑡denote
the input space and Y𝑡the corresponding output space at time step 𝑡, respectively. At each time
step, the AI system receives a data stream D𝑡={(𝑥𝑖
𝑡,𝑦𝑖
𝑡)}𝑛𝑡
𝑖=1, where𝑛𝑡denotes the number of
samples collected during period 𝑡. Unlike classical supervised learning, where the data distribution
is assumed to be stationary, here the distribution P𝑡(𝑥,𝑦)underlyingD𝑡is non-stationary, and
both the input characteristics and the output semantics may change over time. The goal is to learn
a predictive function 𝑓𝜃𝑡parameterized by 𝜃𝑡, which maintains high performance across all past
and present distributions without retraining from scratch.
To enable continual adaptation, we incorporate a retrieval-augmented mechanism into the
learning process. Specifically, at each time step t, given a query 𝑥𝑡, the agent retrieves a set of
external knowledge entries R𝑡(𝑥𝑡)={𝑟𝑗
𝑡}𝑘
𝑗=1from a dynamic knowledge base K𝑡, where k denotes
the number of items retrieved. The information retrieved is used to augment the model input
or intermediate representations, allowing the predictive function to be conditioned not only on
the raw input 𝑥𝑡but also on relevant contextual knowledge. Formally, the predictive function is
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:6 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
Query Retriever
LLMs Answer
Dynamic Urban
Environment Update 
Control
External
Knowledge
Knowledge
Database 
Tools Set
Retrieved 
Knowledge 
Tool Execution
Results 
Domain -Specific
Service
Augmented
Context
Fig. 1. Tool-Enhanced RAG with Continual Learning
expressed as 𝑓𝜃𝑡(𝑥𝑡,R𝑡(𝑥𝑡)), and the learning objective at each time step is to minimize the expected
lossE(𝑥,𝑦)∼P𝑡[ℓ(𝑓𝜃𝑡(𝑥,R𝑡(𝑥)),𝑦)], whereℓ(·)denotes a task-specific loss function.
The continual learning objective requires that the model parameters 𝜃𝑡evolve across time to
accommodate new tasks while preserving performance on previously seen tasks. To this end,
the training process is formulated as a multilevel optimization problem. The first level optimizes
retrieval mechanisms R𝑡(·), the second level optimizes the model adaptation 𝑓𝜃𝑡based on retrieved
knowledge, and the third level coordinates knowledge database updating and forward knowledge
transfer across time steps. The formal problem can be stated as finding a sequence {𝜃𝑡}𝑇
𝑡=1and
retrieval policies{R𝑡}𝑇
𝑡=1that jointly minimize cumulative loss across all time steps, subject to
stability constraints that prevent catastrophic forgetting and ensure continual improvement.
The proposed UrbanMind framework consists of three main components: (i) continual knowledge
integration module, (ii) dynamic retrieval module, and (iii) adaptive model updating module. These
components are designed to jointly optimize the retrieval, integration, and adaptation processes
over evolving data distributions, thereby enabling long-term stability, forward transfer, and robust
decision-making under non-stationary conditions. Each component operates within a multilevel
optimization hierarchy to ensure coordinated and efficient learning.
At each time step 𝑡, the dynamic retrieval module is responsible for identifying and extracting
relevant information R𝑡(𝑥𝑡)from the evolving knowledge base K𝑡, based on the input query 𝑥𝑡.
The retrieval process is adaptive, allowing the system to dynamically incorporate the most relevant
domain-specific knowledge. The continual knowledge integration module then fuses the retrieved
information with the original query, producing an augmented context that serves as the basis for
subsequent prediction or decision-making.
The adaptive model updating module incrementally refines the model parameters 𝜃𝑡at given
time instance 𝑡to incorporate new information while preserving critical capabilities acquired from
previous tasks. This is achieved through a multilevel optimization strategy, where the retrieval
module and model update module are optimized jointly. The overall framework aims to minimize
cumulative predictive loss while enforcing stability constraints that may mitigate catastrophic
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:7
forgetting. By systematically coordinating retrieval, integration, and adaptation, the UrbanMind
framework provides a resilient foundation for achieving UGI.
One fundamental challenge in continual retrieval lies in maintaining retrieval relevance and
consistency over time. As the knowledge base K𝑡evolves over time, retrieval strategies that are static
or trained on historical distributions may rapidly degrade in effectiveness. It becomes necessary to
design retrieval mechanisms that not only adapt to the dynamic structure of K𝑡but also preserve
semantic consistency with previously retrieved knowledge.
Please note that, in the proposed framework, retriever strategy optimization, knowledge database
updating, and model adaptation are decoupled and operated across different time scales to balance
responsiveness, stability, and computational efficiency. Retriever strategy optimization is executed
at a relatively short time scale, frequently adjusting retrieval policies based on immediate task
relevance and feedback from model performance. This enables the system to maintain high retrieval
precision as the query distribution evolves. In contrast, knowledge updating mechanisms may
operate at an intermediate time scale, periodically incorporating new data into the knowledge base
while validating and pruning outdated or low-relevance entries. This ensures that the retrieval
corpus remains current without introducing instability from overly frequent modifications. Model
adaptation usually occurs at the longest time scale, where fine-tuning is applied to avoid overfitting
to transient data shifts and to mitigate catastrophic forgetting. This multi-timescale design allows
the framework to adapt dynamically to new information while preserving long-term learning
stability and computational scalability. However, while the proposed framework generally adheres
to the described multi-timescale paradigm, where retrieval strategy optimization operates most
frequently, followed by knowledge base updating and then model adaptation. Such a hierarchy
may invert in certain application scenarios due to domain-specific requirements. For instance, in
traffic-prediction [ 17,55,56], where traffic conditions vary rapidly, the knowledge base must be
updated almost in real time to incorporate the latest traffic indicators. In such cases, knowledge
updating operates at the shortest time scale to ensure that the retrieval process accesses the most
current information, even more frequently than retrieval strategy optimization. Conversely, in
highly dynamic dialogue systems for personalized education, the user’s interaction patterns and
feedback may rapidly evolve [ 89]. Here, model adaptation may occur on a shorter timescale than
knowledge updates or retrieval adjustments, especially when personalized fine-tuning is necessary
to ensure responsiveness and effectiveness. These examples highlight that, in practice, the temporal
scheduling of updates must be flexibly adapted to the characteristics of specific tasks and domains.
2.2 Background: Naive RAG Pipeline
RAG is a widely adopted framework for enhancing the reasoning capabilities of large language
models (LLMs) by integrating external knowledge sources [ 47]. Before delving into the proposed
UrbanMind framework, we first introduce the Naive RAG pipeline, which serves as a baseline for
understanding the RAG and its limitations in dynamic urban environments.
The Naive RAG pipeline, as depicted in Figure 2, consists of four key stages designed to incor-
porate external knowledge into the generation process. The workflow begins with the chunking
phase, where raw documents, such as textual reports or structured datasets are segmented into
smaller, semantically coherent chunks. This segmentation ensures that the knowledge is broken
down into manageable units suitable for efficient storage and retrieval. Next, these chunks are
indexed and stored in a database, forming a static knowledge repository that can be queried later.
In the retrieval phase, a user query, e.g., a question about urban traffic conditions is encoded
into a vector representation using a pre-trained encoder, such as BERT [ 13]. The encoded query
is then used to search the vector database, retrieving the top-K most relevant chunks based on
similarity metrics, typically cosine similarity between the query and chunk embeddings. These
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:8 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
Chunking
Retrieval
 DB
Generation
Indexing
Query 
Matching
Fig. 2. Naive RAG Pipeline Workflow
retrieved chunks provide external context that is critical for grounding the model’s reasoning in
factual knowledge. Finally, in the generation phase, the retrieved chunks are combined with the
original query and fed into a large language model, which generates a response by leveraging both
the query and the retrieved knowledge.
2.3 Urban Intelligence Tasks
Urban intelligence tasks encompass a broad spectrum of applications that demand AI systems
capable of reasoning and operating effectively in dynamic environments. As discussed in [ 81], the
UGI foundation platform has been applied to various urban domains, including transportation
and urban economy. Representative tasks include conducting travel surveys within transportation
systems [ 20], selecting optimal business sites in business intelligence [ 53], formulating policies
in urban economic systems [ 43], and managing emergencies in urban society [ 32]. These tasks
can broadly categorized into three major domains, i.e., public safety management ,transportation
systems , and urban planning and development . Each domain presents unique data characteristics, op-
erational constraints, and decision-making requirements that influence the design and deployment
of continual learning and retrieval-augmented frameworks.
Public safety management encompasses tasks such as threat detection, emergency response
coordination, and predictive risk assessment for urban populations. Practical examples include
the timely identification of infectious disease outbreaks through hospital reports or social media
analysis, and early flood warnings enabled by monitoring river water levels using hydrological
sensors. Similarly, anomalous patterns in air quality or radiation levels may indicate emerging
environmental hazards. These tasks rely on heterogeneous data sources, including surveillance
feeds, incident reports, social media streams, and environmental sensing infrastructures [25, 66].
Transportation intelligence focus primarily on tasks such as optimizing traffic flow and predicting
congestion patterns. These tasks are often characterized by real-time data streams generated
from heterogeneous sources such as sensors, GPS devices, and traffic cameras [ 60,70]. In broader
applications, transportation intelligence also encompasses low-altitude logistics e.g., drone-based
delivery, railway logistics, and highway freight systems. The underlying data distributions are
subject to rapid fluctuations driven by daily commuting patterns, weather conditions, and special
events, necessitating models that can quickly adapt without losing historical knowledge of baseline
traffic behaviors.
Urban planning and development , in contrast, typically operate on longer time scales and involve
the integration of census data, land use maps, and environmental assessments [ 12]. These tasks
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:9
require AI systems to reason over structured, semi-structured, and unstructured data formats.
Collectively, the diversity across these categories imposes stringent requirements on retrieval
accuracy, continual learning stability, and adaptive reasoning capabilities.
While these domains differ in timescales and data modalities, they collectively highlight a
common challenge: urban data is inherently dynamic and non-stationary. Temporal variations
arise from periodic patterns (e.g., commuting cycles), sudden disruptions (e.g., emergencies), and
gradual changes (e.g., urbanization). Spatial heterogeneity stems from differences in geography,
infrastructure, and localized behavior. Meanwhile, contextual shifts reflect the evolving nature
of societal, environmental, and policy factors. These characteristics result in non-stationary data
streams that challenge static learning paradigms.
To address these challenges, urban intelligence systems must support continual knowledge
integration and adaptive retrieval. Retrieval mechanisms must dynamically adjust to the evolving
knowledge base, ensuring relevance and robustness against semantic drift. Integration pipelines
must handle noisy, incomplete, and potentially conflicting signals while maintaining alignment with
historical knowledge. In addition, these processes must operate under strict latency requirements
and resource constraints to enable real-time urban decision-making. The ability to continually adapt
while preserving accumulated knowledge is thus fundamental to sustaining high-level reasoning
in complex, real-world urban environments.
2.4 Framework Overview
The proposed UrbanMind framework is designed to address the challenges of dynamic knowledge
acquisition, continual adaptation, and robust decision-making in non-stationary urban environ-
ments. The system architecture is organized into four interconnected layers, including the database
layer, the retrieval layer, the integration layer, and the adaptation layer. In the database layer, data
acquired by multimodal sensors and sources from the dynamic urban environment are stored in the
knowledge base. In addition, a tool set includes multi-domain functions which the urban system
provides is available for retriever to get tool execution results. The retrieval layer dynamically
queries a continually evolving knowledge base to extract task-specific information based on incom-
ing urban inputs. The integration layer fuses the retrieved knowledge with model representations,
enabling contextually informed reasoning. The adaptation layer incrementally updates the model
parameters to incorporate new knowledge while preserving previously learned capabilities. Each
layer is optimized with distinct objectives but coordinated under a unified multilevel optimization
framework to maintain system-wide stability and adaptability.
The retrieval layer interfaces with a dynamic, continuously updated knowledge repository, which
may include structured data e.g., urban maps, policy documents as well as unstructured data e.g.,
sensor feeds, social media reports. Retrieved knowledge is filtered and encoded into a format
compatible with the language model’s internal processing pipeline. The integration layer aligns
this external information with internal contextual embeddings, allowing the model to ground its
reasoning on both historical and newly acquired knowledge. The adaptation layer employs continual
learning strategies, such as regularization and memory replay, to update model parameters while
mitigating catastrophic forgetting. Together, these layers form a tightly coupled system capable of
sustaining high-performance urban intelligence operations over long time horizons.
The knowledge retrieval pipeline in the UrbanMind framework is designed to dynamically
extract relevant information from an evolving urban knowledge base. Upon receiving a query 𝑥𝑡
at time step t, the retrieval module first encodes the query into a latent representation using a
lightweight encoder. This representation is then matched against indexed entries in the knowledge
baseK𝑡using similarity search techniques optimized for dynamic environments. To handle the
heterogeneous nature of urban data, the knowledge base is organized into multiple modalities and
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:10 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
Image
Timeseries
Table
Text[cls]
Token 1
Token 2
Token 3
…
…
…
Token nEncoderEncoderEncoderEncodersExternal Knowledge Alignment LLM embedding Space   Query 
Query Embedding
& UnderstandingLightweight Encoder 
Similarity Search
&Tool Callin gKnowledge 
DatabaseKnowledge 
DatabaseMultimodal and 
Multi -domain 
Knowledge base
Retrieved knowledge
&Tool execution
Filter Integration Layer Retrieval Layer Adaptation Layer
Memory 
Management 
ModuleLLM
•Memory Replay
•Regularization  Decisions
Retrieved 
ContentsContinual Learning 
Dynamic Environment
Sensors
Policies Cameras 
Event  
Reports …Temporal Decay
Redundancy detection
Performance validationUpdate 
ControllerData 
StreamDatabase Layer
Map Engine
Simulator
Optimizer…Tool 
Set
Fig. 3. UrbanMind Framework
domains, allowing the retrieval module to perform targeted, context-aware searches. Retrieved
entriesR𝑡(𝑥𝑡)are filtered based on relevance scores and uncertainty estimates before being passed
to the integration module for downstream processing.
Given the non-stationary nature of urban environments, the continual update of the knowledge
baseK𝑡is critical for maintaining retrieval accuracy and contextual relevance. New information
streams, such as updated traffic reports, environmental sensor readings, or policy changes, are
periodically ingest into K𝑡through an incremental indexing mechanism. Older entries are either
updated or pruned based on criteria such as timestamp relevance, redundancy detection, and
semantic consistency. To mitigate the risks of retrieval noise and knowledge drift, a validation layer
monitors newly ingested entries, employing lightweight classifiers or rule-based filters to enforce
minimal quality standards. This continual update mechanism ensures that the retrieval pipeline
remains robust against concept shifts and information obsolescence.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:11
The integration between continual retrieval and model adaptation is coordinated through a
retrieval memory management module. This module maintains metadata regarding the retrieval
history and past integration decisions, enabling the system to balance exploitation of stable historical
knowledge and exploration of newly retrieved information. By dynamically adjusting retrieval
strategies based on performance feedback, the framework ensures that knowledge integration
remains both adaptive and stable. This continual coupling between retrieval updates and model
adaptation forms the core mechanism that allows the UrbanMind system to achieve long-term
resilience and effective decision-making in evolving urban environments.
Notably, the proposed UrbanMind framework can be seamlessly implemented on a urban Cloud-
Edge system [ 14] (Fig 4), wherein the cloud layer is responsible for centralized orchestration and
the management of global knowledge within LLM, while the edge layer focuses on localized data
processing and personalized retrieval adaptation. Under this architecture, each edge node maintains
an independent local knowledge base that captures region-specific and real-time information,
such as traffic patterns or security surveillance data. Each edge database can be connected with a
lightweight fine-tuning model, referred to as an adapter. The adapter is designed to enable efficient
personalization and task adaptation through minimal parameter updates, without modifying the
core parameters of the pre-trained language model. By allowing each edge node to train its adapter
based on localized context and task-specific requirements, the framework supports the deployment
of highly customized, context-aware intelligent services across heterogeneous urban environments.
Cloud
Edge 
Traffic Database Security Database Environmental Database 
Knowledge Base
LLM
Retriever 
Adapter 1 Adapter 1 Adapter 2
Retrieval of Domain -Specific Tasks
Fig. 4. UrbanMind over Cloud-Edge Architecture
3 Multilevel Optimization Strategy for UrbanMind
UrbanMind adopts a multilevel optimization strategy that aligns naturally with its modular design
and integrates seamlessly with MoE architectures. This approach enables coordinated optimization
across different layers of UrbanMind, allowing each component to specialize and adapt indepen-
dently while maintaining overall system coherence. In addition, please note that this strategy is
highly flexible, i.e., it supports both end-to-end training and targeted optimization of selected
modules, making it suitable for a wide range of deployment scenarios and resource budgets. Its
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:12 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
flexibility and generality ensure broad applicability across diverse urban tasks, providing robust
and scalable performance in dynamic and data-driven environments.
3.1 Multilevel Optimization
Multilevel optimization is a hierarchical optimization framework in which the solution to an upper-
level problem depends on the optimal solution of one or more lower-level problems. Multilevel
optimization serves as a unifying framework that includes robust optimization [ 84,85] and bilevel
optimization [40] as special cases.
min
𝒙1∈X1,𝒙2∈X2,...,𝒙𝐾∈X𝐾F1(𝒙1,..., 𝒙𝐾)
s.t. 𝒙2∈arg min
𝒙′
2∈X2(𝒙1)F2(𝒙1,𝒙′
2)
𝒙3∈arg min
𝒙′
3∈X3(𝒙1,𝒙2)F3(𝒙1,𝒙2,𝒙′
3)
...
𝒙𝐾∈arg min
𝒙′
𝐾∈X𝐾(𝒙1,...,𝒙𝐾−1)F𝐾(𝒙1,..., 𝒙𝐾−1,𝒙′
𝐾),(1)
where 𝒙1,𝒙2,..., 𝒙𝐾represent decision variables at different hierarchical levels, each constrained by
feasible setsX𝑘that may depend on higher level variables. The objective functions F𝑘correspond
to each level’s objective and the lower-level problem’s optimal as constraints.
This structure naturally captures nested decision-making hierarchies, making it well suited
for many real-world tasks with nested dependencies. In the field of machine learning, multilevel
optimization has been widely adopted in various applications such as hyperparameter tuning, meta-
learning [ 42], and neural architecture search [ 38], where it effectively captures the interplay between
model training and evaluation [ 37,52]. Moreover, multilevel optimization can be implemented in a
distributed manner [40, 90].
Given its natural ability to model layered decision structures and integrate local and global
objectives under uncertainty, multilevel optimization is particularly well suited for continual
retrieval-augmented generation with large language models, where both the retrieval module and
the generative model require joint, adaptive, and context-sensitive optimization over time. This is
especially relevant in urban intelligence scenarios, where learning systems are inherently distributed
across edge devices, sensors, and cloud infrastructure. In such settings, multilevel optimization
provides a principled framework to coordinate decentralized learning, handle heterogeneous data
sources, and adapt to dynamic environments in a scalable and robust manner.
3.2 Dynamic Knowledge Retrieval Optimization
To maintain retrieval relevance in dynamic urban environments, the UrbanMind framework em-
ploys a task-aware retrieval strategy that adapts retrieval policies based on the current task context.
At each time step 𝑡, given an input 𝑥𝑡and its associated task descriptor 𝜏𝑡, the retrieval module
dynamically selects a retrieval subspace within the knowledge base K𝑡that aligns with the se-
mantic and operational requirements of 𝜏𝑡. Task descriptors are either explicitly provided e.g.,
transportation prediction or safety monitoring. By restricting retrieval to task-relevant domains,
the system improves retrieval efficiency, reduces noise, and enhances the contextual grounding of
the downstream tasks.
The retrieval scoring function is jointly optimized to account for both semantic similarity between
𝑥𝑡and candidate knowledge entries, and task relevance based on 𝜏𝑡. Formally, the retrieval score
𝑠(𝑥𝑡,𝑟𝑗;𝜏𝑡)for a candidate entry 𝑟𝑗is defined as a weighted combination of similarity metrics and
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:13
task-specific relevance estimators. This composite scoring allows the retrieval module to favor
entries that are not only lexically similar but also operationally significant for the target task.
To adapt to evolving task definitions and domain shifts, the retrieval parameters are updated
continually through feedback signals derived from downstream task performance.
Moreover, to ensure robustness under concept drift and multi-task scenarios, the task-aware
retrieval strategy maintains a dynamic task profile memory. This memory captures historical
retrieval patterns and associated task performance metrics, enabling the system to adjust retrieval
subspace selection and scoring mechanisms over time. When encountering new tasks or unseen
conditions, the system can leverage task memory to perform retrieval by analogy, drawing from
similar prior experiences. By integrating task awareness into the retrieval process, the UrbanMind
framework achieves greater flexibility, relevance, and resilience in knowledge acquisition for
continually evolving urban intelligence applications.
In dynamic urban environments, the retrieval corpus K𝑡is subject to continuous evolution
as new information becomes available and outdated information loses relevance. To maintain
retrieval quality under data drift, the UrbanMind framework implements an incremental corpus
update mechanism. New data streams such as sensor readings, event reports, or policy changes
are continuously processed and indexed into K𝑡through a lightweight ingestion pipeline. Each
incoming entry is associated with temporal metadata, task relevance scores, and uncertainty
estimates to facilitate subsequent retrieval and maintenance decisions. To prevent uncontrolled
corpus growth and semantic inconsistency, stale or low-relevance entries are periodically pruned
based on temporal decay functions, redundancy detection, and performance-driven validation
metrics.
Evaluating retrieval effectiveness in the UrbanMind framework requires metrics that capture
both the relevance and robustness of retrieved knowledge under dynamic conditions. Standard
retrieval metrics such as Top-k accuracy, Mean Reciprocal Rank (MRR), and Normalized Discounted
Cumulative Gain (NDCG) are employed to measure how accurately the retrieved entries align with
the ground-truth or task-specific information. In addition to static retrieval performance, continual
settings necessitate tracking temporal retrieval stability, defined as the consistency of retrieval
quality across evolving data distributions. Drift-aware metrics, such as relevance retention rate and
retrieval degradation rate over time, are further utilized to quantify the system’s resilience to concept
drift [ 5]. Together, these metrics provide a comprehensive evaluation of retrieval performance in
both stationary and non-stationary urban environments.
3.3 Model Adaptation
In the UrbanMind framework, the construction of retrieval-conditioned inputs is a critical step for
effectively integrating external knowledge into the model’s reasoning process. Upon retrieval, the
selected entriesR𝑡(𝑥𝑡)are first processed through a knowledge encoding module that transforms
heterogeneous data types, such as textual descriptions, sensor observations, or structured records,
into unified latent representations. These encoded retrievals are then combined with the original
input representation ℎ(𝑥𝑡)through concatenation or fusion mechanisms specifically designed to
preserve task-relevant information while minimizing redundancy. The retrieval-conditioned input,
denoted as ˜ℎ(𝑥𝑡)=F(ℎ(𝑥𝑡),R𝑡(𝑥𝑡)), serves as the new context for downstream prediction or
decision-making tasks.
To ensure that the retrieval-conditioned representations remain robust across dynamic environ-
ments, the fusion mechanism F(·) is trained to selectively emphasize high-confidence, task-relevant
knowledge while attenuating the influence of noisy or irrelevant retrievals. Attention-based weight-
ing schemes, confidence scoring, and domain-specific gating functions are employed to dynamically
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:14 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
modulate the contribution of each retrieved entry during integration. This retrieval-conditioned
construction process enables the model to ground its reasoning not only on the immediate input
but also on continually evolving external knowledge, providing a foundation for stable and adaptive
learning in non-stationary urban environments.
To accommodate evolving urban knowledge and task distributions, the UrbanMind framework
employs continual fine-tuning strategies conditioned on dynamically retrieved information. At
each time step 𝑡, model updates are performed using retrieval-conditioned inputs ˜ℎ(𝑥𝑡)to align
the model’s internal representations with the most recent knowledge context. Fine-tuning objec-
tives incorporate regularization terms to preserve critical parameters associated with previous
tasks, thereby mitigating catastrophic forgetting while allowing sufficient plasticity for adaptation.
Dynamic sample selection mechanisms prioritize fine-tuning on high-confidence retrievals and
task-critical examples, ensuring that model updates are both efficient and stability-preserving.
Through this retrieval-aware continual fine-tuning process, the model incrementally refines its
predictive capabilities in response to both input distribution shifts and knowledge base evolution.
3.4 Multilevel Optimization for Urban LLMs Training
Recent work leverages Mixture-of-Experts (MoE) architectures for LLMs, which activate only
a small subset of expert networks per input, significantly reducing computation costs without
major performance loss. Models like DeepSeek-R1 [ 33] and GLaM [ 18] show MoE’s effectiveness
in complex tasks such as reasoning, code generation, and domain adaptation, while maintaining
high efficiency.
The MoE framework typically consists of a gating network 𝑔(𝑥;𝜃𝑔)and a collection of experts
{𝑒𝑖(𝑥;𝜃𝑒𝑖)}𝑁
𝑖=1. For a given input 𝑥, the gating function produces a sparse distribution over experts,
activating only a few (e.g., top-1 or top-2) to process the input. The training of such a system
involves two main objectives:
•Expert-specific loss minimization. Each expert 𝑒𝑖is trained to minimize its task-specific loss
when it is activated. Let L𝑒𝑖(𝑥;𝜃𝑒𝑖)denote the loss incurred by expert 𝑖, then the total expert
loss for an input 𝑥is weighted by the gating score:
Lexpert(𝑥)=𝑁∑︁
𝑖=1𝑔(𝑥;𝜃𝑔)𝑖·L𝑒𝑖(𝑥;𝜃𝑒𝑖). (2)
•Routing quality regularization. The gating network itself is trained to produce useful, stable,
and balanced routing decisions. This may involve auxiliary losses such as entropy regular-
ization [ 88], load balancing [ 9], and sparsity constraints [ 27]. We denote this combined loss
as:
L𝑟(𝑥;𝜃𝑔). (3)
From the multilevel optimization perspective, the training of MoE LLM can be naturally formu-
lated as a bilevel optimization problem:
min
𝜃𝑔L𝑟(𝜃𝑔)+E𝑥
Lupper(𝜃𝑔,𝜃∗
𝑒(𝜃𝑔))
s.t.𝜃∗
𝑒(𝜃𝑔)=arg min
𝜃𝑒E𝑥"𝑁∑︁
𝑖=1𝑔(𝑥;𝜃𝑔)𝑖·L𝑒𝑖(𝑥;𝜃𝑒𝑖)#
,(4)
where,𝜃𝑔denotes the routing parameters optimized at the upper-level to minimize routing loss
L𝑟and overall task loss Lupper, while the lower-level optimizes expert parameters 𝜃𝑒={𝜃𝑒𝑖}to
minimize their weighted task losses conditioned on the routing decisions.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:15
This bilevel structure explicitly models the hierarchical interaction in MoE LLM training, such
that routing decisions guide expert specialization, enabling efficient and scalable Urban Foundation
Model training.
3.5 Multi-timescale RAG Optimization
In dynamic urban environments, RAG systems face substantial uncertainty due to frequent and
diverse changes. These uncertainties occur across different timescales and arise from the non-
stationarity of data distributions and the prevalence of OOD scenarios. Examples include seasonal
traffic variations, infrastructure modifications, and unexpected events such as accidents or emergen-
cies. To characterize such uncertainties, we introduce an uncertainty set U, representing potential
distribution shifts. Then, we propose a multilevel formulation for multi-timescale end-to-end RAG
optimization, aiming to ensure robust performance under worst-case scenarios.
To characterize the uncertainty set U, we employ divergence-based metrics to measure the shift
between the empirical training distribution Ptrainand possible test-time distributions. Common
choices include Wasserstein distance [ 73], Jensen-Shannon divergence [ 58], and𝐿2distance [ 67],
each offering different trade-offs in robustness and computational complexity. In this work, we
adopt the Kullback-Leibler (KL) divergence as an example. Accordingly, the uncertainty set is
defined asU={P:𝐷KL(P∥P train)≤𝜌}, where𝜌>0controls the allowable shift from the
training distribution.
The end-to-end optimization strategy of RAG in the UrbanMind framework is formulated as a
bilevel optimization problem, as presented in (5).
min
𝜽F(𝜽,𝝓∗(𝜽))=∑︁
(𝑞𝑗,𝑎𝑗)∈D valℓeval(𝑎𝑗,G(𝑞𝑗,R(𝑞𝑗;𝜽);𝝓∗(𝜽)))
s.t. 𝝓∗(𝜽)=arg min
𝝓Lgen(𝝓;𝜽)=∑︁
(𝑞𝑖,𝑎𝑖)∈D trainℓtrain(𝑎𝑖,G(𝑞𝑖,R(𝑞𝑖;𝜽);𝝓)).(5)
This formulation enables the joint optimization of the retriever R, parameterized by 𝜽, and the
generatorG, parameterized by 𝝓. Given a user query 𝑞𝑡at time step 𝑡, the retriever selects a set of 𝐾
documents,D𝑡=R(𝑞𝑡;𝜽)={𝑑1,𝑑2,...,𝑑𝐾}, from the evolving knowledge base K𝑡. The generator
then produces a response ˆ𝑎𝑡=G(𝑞𝑡,D𝑡(𝑞𝑡;𝜽);𝝓), conditioned on the query and the retrieved
documents.
From the perspective of Distributionally Robust Optimization (DRO) [ 39], the end-to-end for-
mulation can be naturally extended to a multilevel optimization framework, where uncertainty in
dynamic urban environments is explicitly modeled through the uncertainty set U. This approach
provides a principled foundation for enhancing robustness to distributional shifts by optimizing
model performance under worst-case scenarios within U. Suppose that the training dataset Dtrain
is composed of 𝑀domains (e.g., traffic data during peak hours, public safety data, urban planning
data), denoted as{D1,D2,...,D𝑀}. A weight vector 𝒘=[𝑤1,𝑤2,...,𝑤𝑀]is introduced , where
𝑤𝑚represents the importance of domain D𝑚, which incorporates DRO to learn 𝒘, such that the
model generalizes better across diverse and evolving distributions. The multilevel optimization
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:16 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
problem is defined as follows:
min
𝜽F(𝜽,𝝓∗(𝜽),𝒘∗(𝜽))=∑︁
(𝑞𝑗,𝑎𝑗)∈D valℓval(𝑎𝑗,G(𝑞𝑗,R(𝑞𝑗;𝜽);𝝓∗(𝜽)))
s.t. 𝝓∗(𝜽)=arg min
𝝓Lgen(𝝓;𝜽,𝒘∗(𝜽))=𝑀∑︁
𝑚=1𝑤∗
𝑚∑︁
(𝑞𝑖,𝑎𝑖)∈D𝑚ℓtrain(𝑎𝑖,G(𝑞𝑖,R(𝑞𝑖;𝜽);𝝓)),
s.t. 𝒘∗(𝜽)=arg min
𝒘Ldro(𝒘;𝜽,𝝓)=𝑀∑︁
𝑚=1𝑤𝑚∑︁
(𝑞𝑖,𝑎𝑖)∈D𝑚ℓtrain(𝑎𝑖,G(𝑞𝑖,R(𝑞𝑖;𝜽);𝝓)),
s.t.𝑀∑︁
𝑚=1𝑤𝑚=1,∀𝑚∈{1,...,𝑀},
KL(𝒘∥𝒑uniform)<𝜖,(6)
whereDtrainandDvaldenote the training and validation datasets, ℓtrainandℓvalare the training and
evaluation losses, respectively, 𝜆is a regularization coefficient, 𝒑uniform is the uniform distribution
over domains, and 𝜖>0is a small constant to prevent over-concentration of weights. The first-
level problem optimizes the retriever 𝜽to improve overall performance on the validation set. The
second-level problem optimizes the generator 𝝓using a weighted training loss, where the weights
𝒘prioritize contributions from different domains. The third-level problem employs DRO to learn
𝒘, balancing domain contributions with a KL divergence regularization to ensure diversity and
robustness against distributional shifts.
Please note that this multilevel formulation can be adjusted to various temporal scales of module
updates within the RAG framework. For instance, if only the retriever R, parameterized by 𝜽, is to
be optimized while the generator Gand domain weights 𝒘are assumed to be fixed (e.g., pre-trained
or updated less frequently), then the overall optimization reduces to a single-level problem with
respect to 𝜽. Similarly, when the generator parameters 𝝓are fixed, the optimization focuses on
learning a robust retriever under distributional shifts through the upper-level objective involving 𝜽
and𝒘, effectively resulting in a bilevel problem.
In addition, under different update schedules or timescales for retriever and generator compo-
nents, the original multilevel problem can be decomposed or relaxed into a sequence of tractable
sub-problems. Each sub-problem can be solved separately, enabling practical and modular training
strategies aligned with the dynamic nature of continual learning in evolving urban environments.
A key challenge in this framework is the computational complexity of multilevel optimization,
compounded by the non-differentiability of the retriever’s output due to the discrete top- 𝐾doc-
ument selection process. Additionally, to adapt to the evolving knowledge base K𝑡, the retriever
incorporates task-aware scoring, as described in Section 3.2, weighting document relevance based
on task descriptors 𝜏𝑡. This ensures contextual alignment with current urban intelligence tasks.
4 Evaluations
In this section, we present a systematic evaluation framework to rigorously assess the performance
of proposed UrbanMind that enable a plethora of urban generative intelligence tasks [92].
Level-1: Urban tasks focus on retrieving explicit factual information directly from available urban
datasets without requiring complex reasoning or inference. These tasks involve identifying and
extracting specific details such as traffic incident reports, public transportation schedules, air quality
indices, or zoning regulations, which are explicitly recorded in structured or semi-structured data
sources. For example, answering queries like "What is the current congestion level on Highway
XXX?" or "Which zones are designated for residential use in the downtown area?" requires the
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:17
system to locate and extract the relevant factual information without synthesizing or extrapolating
beyond the provided data.
Level-2: Urban tasks for implicit facts from available urban data, requiring basic logical reasoning
or simple cross-referencing across multiple information sources. Unlike Level-1 tasks, where infor-
mation is directly accessible, Level-2 tasks demand that the system perform elementary deductions
or combine dispersed data segments to derive the correct answer. For example, answering a query
such as "Which public transportation lines are most affected by the ongoing road construction
near Central Avenue?" necessitates correlating information about construction zones with transit
route maps and service updates. These tasks test the model’s ability to integrate related factual data
points and apply straightforward reasoning, thereby representing a critical step toward enabling
more context-aware and intelligent urban decision-making.
Level 3: Urban tasks require not only the retrieval of factual information but also the com-
prehension and application of domain-specific complex rationales that govern decision-making
within the urban context. For example, evaluating whether a proposed urban development complies
with zoning regulations involves interpreting legal statutes, procedural workflows, and multi-step
approval processes. Likewise, understanding emergency response prioritization across varying
incident types may involve extracting implicit practices from historical dispatch and resolution
logs.
4.1 Experimental Setup
To evaluate the performance of UrbanMind across the defined levels of urban tasks, we conducted
experiments on a high-performance computing environment. The experiments were run on an
Ubuntu 22.04 LTS server equipped with an NVIDIA RTX 4090 GPU (24GB VRAM), 128GB of
DDR5 RAM, and an Intel Core i9-13900K CPU (24 cores, 32 threads). This setup ensured efficient
processing of large-scale urban datasets and the computational demands of continual learning and
retrieval-augmented generation.
We developed an interactive evaluation pipeline using the Streamlit framework (version 1.32.0),
which facilitated real-time user queries and visualization of model responses for urban intelligence
tasks. The UrbanMind framework was implemented using PyTorch (version 2.3.0) with CUDA
12.1 for GPU acceleration, and large model inference was optimized using vLLM (version 0.6.3)
to enhance throughput and reduce latency during evaluation. The retrieval component leveraged
Milvus (version 2.5.4) for fast similarity search over the dynamic knowledge base K𝑡to handle
structured urban data. Real-time data updates were simulated using synthetic streams generated
from publicly available urban datasets [30].
To demonstrate the advantages of tool-enhanced RAG, including improved factual accuracy and
support for multistep reasoning via domain-specific tool invocation, we develop a prototype of
UrbanMind framework(Fig. 8) for urban travel planning within a broader urban intelligence system.
By integrating RAG with a modular tool-calling mechanism, the system dynamically selects and
executes tools such as time and weather as well as traffic evaluators, based on user travel queries.
The contextual information retrieved from the tool executions is incorporated into the LLM’s
reasoning process, enabling adaptive planning of routes and transportation modes that account for
real-time environmental constraints and urban dynamics.
Specifically, in this tool-enhanced UrbanMind system, we constructed a toolset that includes a
weather checking tool, a time tool, and a traffic availability access tool. A knowledge base with
query-answer memory is maintained to store information about how to generate travel plans, the
available tools, and interface documentation to support accurate tool invocation. The system is built
using the LangChain PlanAndExecute framework, leveraging the cloud-based Qwen2.5-32B-Instruct
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:18 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
model as the base LLM for enhanced generation capabilities, and a local PC-based RAG module for
embedding and storing domain-specific knowledge.
(1) LLM
(2) Static  RAG LLM
(3) C ontinual RAG LLM
Fig. 5. LLM,Static RAG LLM and Continual RAG Com-
parison with Level 1 Task
(1) LLM
(2) Static RAG LLM
(3) C ontinual RAG LLMFig. 6. LLM,Static RAG LLM and Continual RAG Com-
parison with Level 2 Task
4.2 Experimental Results
Fig. 5, Fig. 6, and Fig. 7 present the results of LLM-only generation, LLM with static RAG, and LLM
with continual RAG, respectively, across the three task levels. The LLM-only approach lacks access
to real-time information, while the static RAG model retrieves relevant prior experiences from the
knowledge base to support the task. In contrast, the continual RAG model integrates up-to-date
and time-sensitive data to provide the most accurate guidance. Across all task levels, the continual
RAG-enhanced LLM consistently produces the most satisfactory responses, with its superiority in
generation quality especially pronounced in lower-level tasks.
Fig. 9, Fig. 10, and Fig. 11 illustrate the travel planning outcomes generated by the LLM-only
system, the RAG-LLM system, and the Tool-enhanced UrbanMind system, respectively. The LLM-
only system is capable of suggesting general travel routes from location A to location B; however,
it fails to incorporate contextual factors such as time and weather, thus limiting the relevance and
accuracy of its recommendations. The RAG-LLM system, which augments the LLM with retrieved
knowledge including public schedules and historical query-answer pairs, is able to offer more
informative suggestions, for instance, checking weather conditions or public transport availability,
as well as identifying potential traffic congestion. Nevertheless, it still lacks the ability to generate
a coherent and concrete travel plan. In contrast, the Tool-enhanced UrbanMind system not only
retrieves relevant knowledge through RAG but also enables the LLM to autonomously identify
and invoke appropriate tools. Based on the real-time outputs of these tools, the system delivers a
detailed and context-aware travel plan tailored to the user’s query.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:19
(1) LLM
(2) Static  RAG LLM
(3) C ontinual RAG LLM
Fig. 7. LLM,Static RAG LLM and Continual RAG Com-
parison with Level 3 Task
Quer y
RAG -Retriev erKnowledge bases
HistoryDomain Knowledge
Tool Documentation
Query
Retrieval Knowledge 
tool execution 
answer Retrieved 
Knowledge
LLM 
Taske Plan 
Tool SetAugmented 
Query
Time
Weather
Traffic_AvailabilityLLM Planner
Executor
Task execution 
resultsAnswer
History
Continual UpdatingEdge -based
Cloud -basedFig. 8. Tool-Enhanced UrbanMind Experimental
Framework
To travel from Anting Town to Hongqiao Airport, you have several options:
1. **Subway**: Take Line 11 from Anting Ecopark Station to Hongqiao 
Airport T1/T2 Station. This journey takes about 30 -40 minutes and is the 
most convenient option.
2. **Bus**: Shuttle buses are available but schedules can vary, so check 
local information for details.
3. **Taxi/Ride -hailing Services**: Use a taxi or Didi for door -to-door 
convenience, though this is more expensive. The trip typically takes 30 -
45 minutes.
4. **Car Rental**: Rent a car for flexibility, but be prepared to navigate 
traffic.
5. **Shared Bicycles**: Use shared bicycles to reach the nearest subway 
station and then take the subway to the airport.
**General Tips**:
-Check the subway schedule to avoid delays.
-Plan for traffic if using a taxi or driving.
-Allow extra time for unexpected delays.
Choose the option that best fits your needs and preferences.
I want to travel to HongQiao Airport from Anting Town, how to go there?
Fig. 9. Travel Plan with LLM-Only
5 Conclusion
This paper introduces UrbanMind, a tool-enhanced RAG framework designed to advance urban
general intelligence, which aims to enable urban intelligence system to incrementally incorpo-
rate evolving urban data through corpus updating and supports privacy-preserving, low-latency
inference via cloud-edge deployment. We further formulate the learning process of UrbanMind
within a multilevel optimization framework that aligns naturally with the architecture of MoE
LLMs. This formulation treats retrieval, generation, and model adaptation as interdependent sub-
problems, allowing for selective or multi-timescale end-to-end optimization according to resource
constraints. Empirical evaluations on diverse real-world urban tasks demonstrate the effectiveness
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:20 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
I want to travel to HongQiao Airport from Anting Town, how to go there?
To travel to HongQiao Airport from Anting Town, you can use the 
following methods:
1. **Metro:** Take Metro Line 2 from Anting Town to HongQiao
Airport. The journey takes approximately 30 -40 minutes.
2. **Taxi/Uber:** A taxi or ride -sharing service like Uber will take 
around 20 -30 minutes, depending on traffic.
3. **Bus:** There are bus routes that connect Anting Town to 
HongQiao Airport, but they may be less frequent and take longer.
**Summary:**
To travel from Anting Town to HongQiao Airport, consider using the 
metro (Line 2), a taxi, or a bus. Check the time, weather, traffic, and 
distance before you go. At late night, public transport options may 
be limited, so a taxi or ride -sharing service might be more reliable. 
During weekday traffic peek hours ground traffic may be congested 
during weekday peak hours and holiday periods
Note: Based on retrieved knowledge, It gives suggestion about public service 
availability, potential traffic congestions.
Fig. 10. Travel Plan with RAG-LLM
I want to travel to HongQiao Airport from Anting Town, how to go there?
Given your current situation at 23:00, here are your options to travel to 
HongQiao Airport from Anting Town :
1. **Subway:** The last train of Line 11 has already departed at 22:00, 
so this option is not available.
2. **Car:** There is no traffic congestion, making this a viable option. 
However, you should be cautious due to the rainy weather.
3. **Walking:** The distance is over 10 km, making it impractical and 
unsafe, especially in the rain.
4. **Bus:** No buses are available at this time .
**Recommendation:**
Given the late hour and rainy weather, the best option is to use a car 
(either your own or a taxi/ride -sharing service) to safely reach HongQiao
Airport.
Note: By leveraging the results obtained from tool invocations, including time, 
weather conditions, and public transportation availability, the system was able to 
generate more precise responses and recommendations. Fig. 11. Travel Plan with Tool-enhanced UrbanMind
and versatility of the proposed framework. Collectively, UrbanMind marks a step toward realizing
practical and adaptable UGI systems for future cities.
References
[1]Ali Ben Abbes, ImedRiadh Farah, and Vincent Barra. 2016. Urban growth analysis using multi-temporal satellite
images, non-stationary decomposition methods and stochastic modeling. World Acad Sci Eng Technol Int J Comput
Electr Autom Control Inf Eng 10, 10 (2016), 1791–1797.
[2]Asma Ait Ouallane, Assia Bakali, Ayoub Bahnasse, Said Broumi, and Mohamed Talea. 2022. Fusion of engineering
insights and emerging trends: Intelligent urban traffic management system. Information Fusion 88 (2022), 218–248.
[3]Sarwat Ali and M Arif Wani. 2024. Tri-level Optimization for Gradient-based Neural Architecture Search. In 2024
International Conference on Machine Learning and Applications (ICMLA) . IEEE, 1546–1552.
[4]Pasquale Balsebre, Weiming Huang, Gao Cong, and Yi Li. 2023. Cityfm: City foundation models to solve urban
challenges. arXiv preprint arXiv:2310.00583 (2023).
[5]Firas Bayram, Bestoun S Ahmed, and Andreas Kassler. 2022. From concept drift to model degradation: An overview on
performance-aware drift detectors. Knowledge-Based Systems 245 (2022), 108632.
[6]Luís MA Bettencourt. 2021. Introduction to urban science: evidence and theory of cities as complex systems. (2021).
[7]Simon Elias Bibri. 2021. Data-driven smart sustainable cities of the future: Urban computing and intelligence for
strategic, short-term, and joined-up planning. Computational Urban Science 1, 1 (2021), 8.
[8]Magdalena Biesialska, Katarzyna Biesialska, and Marta R Costa-Jussa. 2020. Continual lifelong learning in natural
language processing: A survey. arXiv preprint arXiv:2012.09823 (2020).
[9]Chang Chen, Min Li, Zhihua Wu, Dianhai Yu, and Chao Yang. 2022. Ta-moe: Topology-aware large scale mixture-of-
expert training. Advances in Neural Information Processing Systems 35 (2022), 22173–22186.
[10] Xu Chen, Junshan Wang, and Kunqing Xie. 2021. TrafficStream: A streaming traffic flow forecasting framework based
on graph neural networks and continual learning. arXiv preprint arXiv:2106.06273 (2021).
[11] Xingdi Chen, Yu Xiong, and Kai Yang. 2024. Robust beamforming for downlink multi-cell systems: A bilevel optimization
perspective. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38. 7969–7977.
[12] Zhuo Chen, Ruoxi Chen, and Songtao Chen. 2021. Intelligent management information system of urban planning
based on GIS. Journal of Intelligent & Fuzzy Systems 40, 4 (2021), 6007–6016.
[13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional
transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the
association for computational linguistics: human language technologies, volume 1 (long and short papers) . 4171–4186.
[14] Beniamino di Martino, Domenico Di Sivo, and Alba Amato. 2025. Cloud, Edge, and Mobile Computing: Synergies for
the Future of Smart Cities. In International Conference on Advanced Information Networking and Applications . Springer,
158–166.
[15] Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, and Marco Pavone. 2024. Realgen: Retrieval augmented
generation for controllable traffic scenarios. In European Conference on Computer Vision . Springer, 93–110.
[16] Fei Dou, Jin Ye, Geng Yuan, Qin Lu, Wei Niu, Haijian Sun, Le Guan, Guoyu Lu, Gengchen Mai, Ninghao Liu, et al .2023.
Towards artificial general intelligence (agi) in the internet of things (iot): Opportunities and challenges. arXiv preprint
arXiv:2309.07438 (2023).
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:21
[17] Shaoyu Dou, Kai Yang, Yang Jiao, Chengbo Qiu, and Kui Ren. 2024. Anomaly Detection in Event-triggered Traffic
Time Series via Similarity Learning. IEEE Transactions on Dependable and Secure Computing (2024).
[18] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou,
Adams Wei Yu, Orhan Firat, et al .2022. Glam: Efficient scaling of language models with mixture-of-experts. In
International conference on machine learning . PMLR, 5547–5569.
[19] Cristian Duran-Mateluna et al .2025. Adaptive Robust Optimization Models for DER Planning in Distribution Networks
under Long-and Short-Term Uncertainties. arXiv e-prints (2025), arXiv–2503.
[20] Fernando Elizalde-Ramírez, Romeo Sanchez Nigenda, Iris A Martínez-Salazar, and Yasmín Á Ríos-Solís. 2019. Travel
plans in public transit networks using artificial intelligence planning models. Applied Artificial Intelligence 33, 5 (2019),
440–461.
[21] Mohammed A Fadhel, Ali M Duhaim, Ahmed Saihood, Ahmed Sewify, Mokhaled NA Al-Hamadani, AS Albahri, Laith
Alzubaidi, Ashish Gupta, Sayedali Mirjalili, and Yuantong Gu. 2024. Comprehensive systematic review of information
fusion methods in smart cities and urban environments. Information Fusion (2024), 102317.
[22] Yuxin Fan, Yuxiang Wang, Lipeng Liu, Xirui Tang, Na Sun, and Zidong Yu. 2025. Research on the Online Update
Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning. arXiv preprint arXiv:2501.07063
(2025).
[23] Nanyi Fei, Zhiwu Lu, Yizhao Gao, Guoxing Yang, Yuqi Huo, Jingyuan Wen, Haoyu Lu, Ruihua Song, Xin Gao, Tao
Xiang, et al .2022. Towards artificial general intelligence via a multimodal foundation model. Nature Communications
13, 1 (2022), 3094.
[24] Luciano Floridi and Massimo Chiriatti. 2020. GPT-3: Its nature, scope, limits, and consequences. Minds and Machines
30 (2020), 681–694.
[25] Francis Fortin, Julie Delle Donne, and Justine Knop. 2021. The use of social media in intelligence and its impact on
police work. Policing in an Age of Reform: An Agenda for Research and Practice (2021), 213–231.
[26] Cátia AR Freire, Fernando AF Ferreira, Elias G Carayannis, and João JM Ferreira. 2021. Artificial intelligence and smart
cities: A DEMATEL approach to adaptation challenges and initiatives. IEEE Transactions on Engineering Management
70, 5 (2021), 1881–1899.
[27] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. 2023. Megablocks: Efficient sparse training with
mixture-of-experts. Proceedings of Machine Learning and Systems 5 (2023), 288–304.
[28] Daniele Gammelli, James Harrison, Kaidi Yang, Marco Pavone, Filipe Rodrigues, and Francisco C Pereira. 2023. Graph
reinforcement learning for network control via bi-level optimization. arXiv preprint arXiv:2305.09129 (2023).
[29] Qiang Gao, Zhipeng Luo, Diego Klabjan, and Fengli Zhang. 2022. Efficient architecture search for continual learning.
IEEE Transactions on Neural Networks and Learning Systems 34, 11 (2022), 8555–8565.
[30] GeoFabrik GmbH. 2025. OpenStreetMap Data Extracts for China. https://download.geofabrik.de/asia/china.html.
Accessed: 2025-04-29.
[31] Ben Goertzel. 2014. Artificial general intelligence: concept, state of the art, and future prospects. Journal of Artificial
General Intelligence 5, 1 (2014), 1.
[32] Fatemeh Golpayegani, Saeedeh Ghanadbashi, and Maha Riad. 2021. Urban emergency management using intelligent
traffic systems: challenges and future directions. In 2021 IEEE International Smart Cities Conference (ISC2) . IEEE, 1–4.
[33] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang,
Xiao Bi, et al .2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 (2025).
[34] Shanshan Han, Qifan Zhang, Yuhang Yao, Weizhao Jin, Zhaozhuo Xu, and Chaoyang He. 2024. LLM multi-agent
systems: Challenges and open problems. arXiv preprint arXiv:2402.03578 (2024).
[35] Zijian Hei, Weiling Liu, Wenjie Ou, Juyi Qiao, Junming Jiao, Guowen Song, Ting Tian, and Yi Lin. 2024. Dr-rag:
Applying dynamic document relevance to retrieval-augmented generation for question-answering. arXiv preprint
arXiv:2406.07348 (2024).
[36] Zhongzhen Huang, Kui Xue, Yongqi Fan, Linjie Mu, Ruoyu Liu, Tong Ruan, Shaoting Zhang, and Xiaofan Zhang. 2024.
Tool Calling: Enhancing Medication Consultation via Retrieval-Augmented Large Language Models. arXiv preprint
arXiv:2404.17897 (2024).
[37] Chengtao Jian, Kai Yang, and Yang Jiao. 2024. Tri-Level Navigator: LLM-Empowered Tri-Level Learning for Time
Series OOD Generalization. Advances in Neural Information Processing Systems 37 (2024), 110613–110642.
[38] Yang Jiao, Kai Yang, Dongjing Song, and Dacheng Tao. 2022. Timeautoad: Autonomous anomaly detection with
self-supervised contrastive loss for multivariate time series. IEEE Transactions on Network Science and Engineering 9, 3
(2022), 1604–1619.
[39] Yang Jiao, Kai Yang, Tiancheng Wu, Dongjin Song, and Chengtao Jian. [n. d.]. Asynchronous Distributed Bilevel
Optimization. In The Eleventh International Conference on Learning Representations .
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:22 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
[40] Yang Jiao, Kai Yang, Tiancheng Wu, Dongjin Song, and Chengtao Jian. 2022. Asynchronous Distributed Bilevel
Optimization. In The Eleventh International Conference on Learning Representations .
[41] Shaheen Khatoon, Amna Asif, Md Maruf Hasan, and Majed Alshamari. 2022. Social media-based intelligence for
disaster response and management in smart cities. In Artificial Intelligence, Machine Learning, and Optimization Tools
for Smart Cities: Designing for Sustainability . Springer, 211–235.
[42] Minyoung Kim and Timothy Hospedales. 2025. A Stochastic Approach to Bi-Level Optimization for Hyperparameter
Optimization and Meta Learning. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 39. 17913–17920.
[43] Karima Kourtit. 2021. City intelligence for enhancing urban performance value: a conceptual study on data decompo-
sition in smart cities. Asia-Pacific Journal of Regional Science 5, 1 (2021), 191–222.
[44] Ehsan Latif, Gengchen Mai, Matthew Nyaaba, Xuansheng Wu, Ninghao Liu, Guoyu Lu, Sheng Li, Tianming Liu, and
Xiaoming Zhai. 2023. Artificial general intelligence (AGI) for education. arXiv preprint arXiv:2304.12479 1 (2023).
[45] Namkyeong Lee, Edward De Brouwer, Ehsan Hajiramezanali, Tommaso Biancalani, Chanyoung Park, and Gabriele
Scalia. 2025. RAG-Enhanced Collaborative LLM Agents for Drug Discovery. arXiv preprint arXiv:2502.17506 (2025).
[46] Timothée Lesort, Vincenzo Lomonaco, Andrei Stoian, Davide Maltoni, David Filliat, and Natalia Díaz-Rodríguez. 2020.
Continual learning for robotics: Definition, framework, learning strategies, opportunities and challenges. Information
fusion 58 (2020), 52–68.
[47] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in neural information processing systems 33 (2020), 9459–9474.
[48] Xinzhe Li. 2025. A Review of Prominent Paradigms for LLM-Based Agents: Tool Use, Planning (Including RAG), and
Feedback Learning. In Proceedings of the 31st International Conference on Computational Linguistics . 9760–9779.
[49] Xiaopeng Li, Pengyue Jia, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng
Du, Xiangyang Li, et al .2025. A Survey of Personalization: From RAG to Agent. arXiv preprint arXiv:2504.10147 (2025).
[50] Xiang Li, Lin Zhao, Lu Zhang, Zihao Wu, Zhengliang Liu, Hanqi Jiang, Chao Cao, Shaochen Xu, Yiwei Li, Haixing Dai,
et al.2024. Artificial general intelligence for medical imaging analysis. IEEE Reviews in Biomedical Engineering (2024).
[51] Huimin Liu, Qingming Zhan, Sihang Gao, and Chen Yang. 2019. Seasonal variation of the spatially non-stationary
association between land surface temperature and urban landscape. Remote Sensing 11, 9 (2019), 1016.
[52] Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. 2021. Investigating bi-level optimization for
learning and vision from a unified perspective: A survey and beyond. IEEE Transactions on Pattern Analysis and
Machine Intelligence 44, 12 (2021), 10045–10067.
[53] Yu Liu, Jingtao Ding, and Yong Li. 2023. Knowsite: Leveraging urban knowledge graph for site selection. In Proceedings
of the 31st ACM International Conference on Advances in Geographic Information Systems . 1–12.
[54] Paul A Longley and Carolina Tobón. 2004. Spatial dependence and heterogeneity in patterns of hardship: an intra-urban
analysis. Annals of the Association of American Geographers 94, 3 (2004), 503–519.
[55] Hui Ma and Kai Yang. 2023. Metastnet: Multimodal meta-learning for cellular traffic conformal prediction. IEEE
Transactions on Network Science and Engineering 11, 2 (2023), 1999–2011.
[56] Hui Ma, Kai Yang, and Man-On Pun. 2023. Cellular traffic prediction via deep state space models with attention
mechanism. Computer Communications 197 (2023), 276–283.
[57] Vinod Mahor, Romil Rawat, Anil Kumar, Bhagwati Garg, Kiran Pachlasiya, et al .2023. IoT and artificial intelligence
techniques for public safety and security. In Smart urban computing applications . River Publishers, 111–126.
[58] María Luisa Menéndez, Julio Angel Pardo, Leandro Pardo, and María del C Pardo. 1997. The jensen-shannon divergence.
Journal of the Franklin Institute 334, 2 (1997), 307–318.
[59] Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement
Farabet, and Shane Legg. 2024. Position: Levels of AGI for operationalizing progress on the path to AGI. In Forty-first
International Conference on Machine Learning .
[60] Mahima Nama, Ankita Nath, Nancy Bechra, Jitendra Bhatia, Sudeep Tanwar, Manish Chaturvedi, and Balqies Sadoun.
2021. Machine learning-based traffic scheduling techniques for intelligent transportation system: Opportunities and
challenges. International Journal of Communication Systems 34, 9 (2021), e4814.
[61] Cuong V Nguyen, Alessandro Achille, Michael Lam, Tal Hassner, Vijay Mahadevan, and Stefano Soatto. 2019. Toward
understanding catastrophic forgetting in continual learning. arXiv preprint arXiv:1908.01091 (2019).
[62] Tarannum Nisha, Duong Tung Nguyen, and Vijay K Bhargava. 2022. A bilevel programming framework for joint edge
resource management and pricing. IEEE Internet of Things Journal 9, 18 (2022), 17280–17291.
[63] Nelson Pacheco Rocha, Ana Dias, Gonçalo Santinha, Mário Rodrigues, Carlos Rodrigues, Alexandra Queirós, Rute
Bastardo, and João Pavão. 2022. Systematic literature review of context-awareness applications supported by smart
cities’ infrastructures. SN Applied Sciences 4, 4 (2022), 90.
[64] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2024. Gorilla: Large language model connected
with massive apis. Advances in Neural Information Processing Systems 37 (2024), 126544–126565.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel
Optimization 111:23
[65] Jary Pomponi, Simone Scardapane, Vincenzo Lomonaco, and Aurelio Uncini. 2020. Efficient continual learning in
neural networks with embedding regularization. Neurocomputing 397 (2020), 139–148.
[66] Nihal Poredi, Yu Chen, Xiaohua Li, and Erik Blasch. 2023. Enhance public safety surveillance in smart cities by fusing
optical and thermal cameras. In 2023 26th International Conference on Information Fusion (FUSION) . IEEE, 1–7.
[67] Qi Qian, Shenghuo Zhu, Jiasheng Tang, Rong Jin, Baigui Sun, and Hao Li. 2019. Robust optimization over multiple
domains. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 33. 4739–4746.
[68] Haoxuan Qu, Hossein Rahmani, Li Xu, Bryan Williams, and Jun Liu. 2025. Recent advances of continual learning in
computer vision: An overview. IET Computer Vision 19, 1 (2025), e70013.
[69] Kadiyala Ramana, Gautam Srivastava, Madapuri Rudra Kumar, Thippa Reddy Gadekallu, Jerry Chun-Wei Lin, Mamoun
Alazab, and Celestine Iwendi. 2023. A vision transformer approach for traffic congestion prediction in urban areas.
IEEE Transactions on Intelligent Transportation Systems 24, 4 (2023), 3922–3934.
[70] Roopa Ravish and Shanta Ranga Swamy. 2021. Intelligent traffic management: A review of challenges, solutions, and
future perspectives. Transport and Telecommunication 22, 2 (2021), 163–182.
[71] Sujoy Roychowdhury, Sumit Soman, HG Ranjani, Neeraj Gunda, Vansh Chhabra, and Sai Krishna Bala. 2024. Evaluation
of rag metrics for question answering in the telecom domain. arXiv preprint arXiv:2407.12873 (2024).
[72] Xue Rui, Ziqiang Li, Yang Cao, Ziyang Li, and Weiguo Song. 2023. DILRS: Domain-incremental learning for semantic
segmentation in multi-source remote sensing data. Remote Sensing 15, 10 (2023), 2541.
[73] Ludger Rüschendorf. 1985. The Wasserstein distance and approximation theorems. Probability Theory and Related
Fields 70, 1 (1985), 117–129.
[74] Shweta Srivastava, Aditya Bisht, and Neetu Narayan. 2017. Safety and security in smart cities using artificial intelli-
gence—A review. In 2017 7th international conference on cloud computing, data science & engineering-confluence . IEEE,
130–133.
[75] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
Batra, Prajjwal Bhargava, Shruti Bhosale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[76] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. 2024. A comprehensive survey of continual learning: Theory,
method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
[77] Liyuan Wang, Xingxing Zhang, Kuo Yang, Longhui Yu, Chongxuan Li, Lanqing Hong, Shifeng Zhang, Zhenguo Li, Yi
Zhong, and Jun Zhu. 2022. Memory replay with data compression for continual learning. arXiv preprint arXiv:2202.06592
(2022).
[78] Shuhe Wang, Shengyu Zhang, Jie Zhang, Runyi Hu, Xiaoya Li, Tianwei Zhang, Jiwei Li, Fei Wu, Guoyin Wang, and
Eduard Hovy. 2024. Reinforcement learning enhanced llms: A survey. arXiv preprint arXiv:2412.10400 (2024).
[79] Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis Ioannidis, Karthik Subbian,
Jure Leskovec, and James Y Zou. 2024. Avatar: Optimizing llm agents for tool usage via contrastive reasoning. Advances
in Neural Information Processing Systems 37 (2024), 25981–26010.
[80] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao. 2024. Rule: Reliable
multimodal rag for factuality in medical vision language models. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing . 1081–1093.
[81] Fengli Xu, Jun Zhang, Chen Gao, Jie Feng, and Yong Li. 2023. Urban Generative Intelligence (UGI): A Foundational
Platform for Agents in Embodied City Environment. CoRR (2023).
[82] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng Wang, and Zheng Li. 2024.
Retrieval-augmented generation with knowledge graphs for customer service question answering. In Proceedings of
the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 2905–2909.
[83] Fei Yang, Yixin Hua, Xiang Li, Zhenkai Yang, Xinkai Yu, and Teng Fei. 2022. A survey on multisource heterogeneous
urban sensor access and data management technologies. Measurement: Sensors 19 (2022), 100061.
[84] Kai Yang, Jianwei Huang, Yihong Wu, Xiaodong Wang, and Mung Chiang. 2014. Distributed robust optimization
(DRO), part I: Framework and example. Optimization and Engineering 15 (2014), 35–67.
[85] Kai Yang, Yihong Wu, Jianwei Huang, Xiaodong Wang, and Sergio Verdú. 2008. Distributed robust optimization for
communication networks. In IEEE INFOCOM 2008-The 27th Conference on Computer Communications . IEEE, 1157–1165.
[86] Li Yang, Zhipeng Luo, Shiming Zhang, Fei Teng, and Tianrui Li. 2024. Continual Learning for Smart City: A Survey.
IEEE Transactions on Knowledge and Data Engineering (2024).
[87] Min Yang, Wenting Tu, Qiang Qu, Kai Lei, Xiaojun Chen, Jia Zhu, and Ying Shen. 2019. MARES: multitask learning
algorithm for Web-scale real-time event summarization. World Wide Web 22 (2019), 499–515.
[88] Masoumeh Zareapoor, Pourya Shamsolmoali, and Fateme Vesaghati. 2024. Efficient Routing in Sparse Mixture-of-
Experts. In 2024 International Joint Conference on Neural Networks (IJCNN) . IEEE, 1–8.
[89] Han Zhang, Yu Lei, Lin Gui, Min Yang, Yulan He, Hui Wang, and Ruifeng Xu. 2024. Cppo: Continual learning for
reinforcement learning with human feedback. In The Twelfth International Conference on Learning Representations .
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:24 Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, and Ye Ouyang
[90] Rongyu Zhang, Yun Chen, Chenrui Wu, Fangxin Wang, and Bo Li. 2024. Multi-level personalized federated learning
on heterogeneous and long-tailed data. IEEE Transactions on Mobile Computing 23, 12 (2024), 12396–12409.
[91] Weijia Zhang, Jindong Han, Zhao Xu, Hang Ni, Hao Liu, and Hui Xiong. 2024. Urban foundation models: A survey. In
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 6633–6643.
[92] Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He, Luna K Qiu, and Lili Qiu. 2024. Retrieval augmented generation
(rag) and beyond: A comprehensive survey on how to make your llms use external data more wisely. arXiv preprint
arXiv:2409.14924 (2024).
[93] Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten De Rijke. 2024. Let me do it for
you: Towards llm empowered recommendation via tool learning. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval . 1796–1806.
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.