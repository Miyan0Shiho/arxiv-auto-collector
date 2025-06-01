# MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models

**Authors**: Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Jihao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhiqiang Yin, Qingchen Yu, Bo Tang, Hongkang Yang, Zhi-Qin John Xu, Feiyu Xiong

**Published**: 2025-05-28 08:27:12

**PDF URL**: [http://arxiv.org/pdf/2505.22101v1](http://arxiv.org/pdf/2505.22101v1)

## Abstract
Large Language Models (LLMs) have emerged as foundational infrastructure in
the pursuit of Artificial General Intelligence (AGI). Despite their remarkable
capabilities in language perception and generation, current LLMs fundamentally
lack a unified and structured architecture for handling memory. They primarily
rely on parametric memory (knowledge encoded in model weights) and ephemeral
activation memory (context-limited runtime states). While emerging methods like
Retrieval-Augmented Generation (RAG) incorporate plaintext memory, they lack
lifecycle management and multi-modal integration, limiting their capacity for
long-term knowledge evolution. To address this, we introduce MemOS, a memory
operating system designed for LLMs that, for the first time, elevates memory to
a first-class operational resource. It builds unified mechanisms for
representation, organization, and governance across three core memory types:
parametric, activation, and plaintext. At its core is the MemCube, a
standardized memory abstraction that enables tracking, fusion, and migration of
heterogeneous memory, while offering structured, traceable access across tasks
and contexts. MemOS establishes a memory-centric execution framework with
strong controllability, adaptability, and evolvability. It fills a critical gap
in current LLM infrastructure and lays the groundwork for continual adaptation,
personalized intelligence, and cross-platform coordination in next-generation
intelligent systems.

## Full Text


<!-- PDF content starts -->

MemOS: An Operating System for
Memory-Augmented Generation (MAG) in Large
Language Models (Short Version)
Zhiyu Li∗,Shichao Song1,3,∗,Hanyu Wang1,3,∗,Simin Niu3,∗,Ding Chen4,∗,Jiawei
Yang1,3,Chenyang Xi1,Huayi Lai3,Jihao Zhao3,Yezhaohui Wang1,Junpeng Ren1,
Zehao Lin1,Jiahao Huo1,Tianyi Chen2,Kai Chen1,Kehang Li2,Zhiqiang Yin3,
Qingchen Yu1,Bo Tang1,†,Hongkang Yang1,†,Zhi-Qin John Xu2,†,Feiyu Xiong1,†
1MemTensor (Shanghai) Technology Co., Ltd. ,2Shanghai Jiao Tong University ,3Renmin
University of China ,4Research Institute of China Telecom
Abstract
Large Language Models (LLMs) have emerged as foundational infrastructure in the pursuit of
Artificial General Intelligence (AGI). Despite their remarkable capabilities in language perception
and generation, current LLMs fundamentally lack a unified and structured architecture for
handling memory. They primarily rely on parametric memory (knowledge encoded in model
weights) and ephemeral activation memory (context-limited runtime states). While emerging
methods like Retrieval-Augmented Generation (RAG) incorporate plaintext memory, they lack
lifecycle management and multi-modal integration, limiting their capacity for long-term knowledge
evolution. To address this, we introduce —a memory operating system designed for LLMs that, for
the first time, elevates memory to a first-class operational resource. It builds unified mechanisms
for representation, organization, and governance across three core memory types: parametric,
activation, and plaintext. At its core is the MemCube , a standardized memory abstraction
that enables tracking, fusion, and migration of heterogeneous memory, while offering structured,
traceable access across tasks and contexts. MemOS establishes a memory-centric execution
framework with strong controllability, adaptability, and evolvability. It fills a critical gap in current
LLM infrastructure and lays the groundwork for continual adaptation, personalized intelligence,
and cross-platform coordination in next-generation intelligent systems.
Date:2025.05.27
Correspondence: {tangb,yanghk,xuzq,xiongfy}@memtensor.cn
Author Legend: *Co-equal primary author, †Correspondence
1 Introduction
Large Language Models (LLMs) are emerging as a foundational pathway toward Artificial General Intelligence
(AGI) [39], yet they remain fundamentally limited in supporting robust memory capabilities. Most current
architectures rely on implicit parametric memory—knowledge embedded within massive model weights—which
is difficult to interpret [ 37], update [ 20], or transfer [ 13]. Although Retrieval-Augmented Generation (RAG)
1arXiv:2505.22101v1  [cs.CL]  28 May 2025

incorporates external knowledge sources [ 3,8,10,11,38], it effectively serves as an ad hoc textual patch and
lacks a structured, unified mechanism for memory management. These architectural shortcomings lead to four
critical issues in real-world applications: inability to model long-term and multi-turn conversational states;
poor adaptability to evolving knowledge; lack of persistent modeling for user preferences and multi-agent
workflows; and the emergence of “memory silos” across platforms, hindering the reuse and migration of prior
interactions. At the root of these challenges lies a fundamental oversight: current LLMs do not treat memory
as an explicit, schedulable, and governable resource.
To address this, we propose —a memory operating system designed for large language models. MemOS
centers memory units as operational resources and establishes a full lifecycle encompassing memory generation,
organization, utilization, and evolution. It offers structured representations, unified interfaces, version control,
and access governance to overcome systemic limitations in memory handling. Rather than merely extending
the RAG paradigm, MemOS introduces a controllable, adaptable, and evolvable memory infrastructure that
empowers LLMs to track knowledge updates, internalize user preferences, and maintain behavioral consistency
across platforms. This represents a fundamental shift in language model architecture: from systems that
merely perceive and generate to those that remember, adapt, and grow over time .
2 Memory in Large Language Models
Stage 1Stage 2
Stage 3
MemOS
ImplicitExplicitLong -termShort -term
abcdea
b
c
d
eMem
Hippocampus
Neocortex
Mindmap
API
Add 
Mem Delete 
MemLLM MemHuman -like LLM MemMem Management
Modify  
MemUpdate  Mem
Figure 1 Memory (Mem) in LLMs.
Research into LLM memory has progressed through three major stages (see Figure 1).
The first is the Memory Definition and Exploration stage, in which researchers classify and analyze memory
mechanisms along dimensions such as parametric vs. non-parametric and short-term vs. long-term memory
[7,23,30]. For implicit memory, pre-training and adapter-based methods embed knowledge directly into model
weights, while knowledge editing techniques enable targeted post hoc modifications [ 1,2,5,9,14,19,24,26,32].
KV-cachesandhiddenstatesconstitutethecoreof implicit short-term memory , preservingcontextualcontinuity
and guiding generation behavior during inference [ 6,16,25,27,28]. Explicit short-term memory typically
involves prompt concatenation within the context window, but remains limited by context length constraints
[18,21]. Explicit long-term memory leverages external retrieval mechanisms, increasingly adopting structured
formats—such as graphs and trees—to improve semantic integration and retrieval efficiency [8, 15, 31, 35].
The second stage involves the Emergence of Human-like Memory , where systems optimized for long-term per-
sistence, context awareness, and self-reflection begin to exhibit structural and behavioral patterns reminiscent
of human memory. Examples include brain-inspired architectures such as HippoRAG and Memory3[12,34],
as well as systems like PGRAG and Second-Me [ 17,29], which support behavior continuity and personalized
memory modeling.
The third stage advances toward Systematic Memory Management , integrating tool-based operations with
OS-inspired governance frameworks. This includes toolkits such as EasyEdit and Mem0, which support
2

explicit memory manipulation [ 4,33,36], as well as systems like Letta [ 22], which implement paged context
management and modular invocation. However, these systems still fall short of providing unified scheduling,
lifecycle governance, and memory fusion across roles or agents.
3MemOS Design Philosophy
As AGI continues to evolve into increasingly complex systems characterized by multi-tasking, multi-role collab-
oration, and multi-modality, language models must move beyond merely “understanding the world”—they must
also “accumulate experience,” “retain memory,” and “continuously evolve.” However, prevailing architectures
remain anchored in static parameters and lack structured modeling and unified management of memory,
rendering them inadequate for supporting knowledge updates, state retention, and personalized adaptation.
We propose that treating memory as a first-class resource and building a memory-centric execution paradigm
is key to enabling continual adaptation and long-term reasoning in future LLMs.
As shown in Figure 2, traditional scaling laws are approaching diminishing returns. The research paradigm
is shifting from data- and parameter-centric pretraining to post-training paradigms focused on alignment
and fine-tuning. Yet even this refined approach faces dual challenges: diminishing performance gains and
increasing engineering complexity. We posit that the next fundamental leap will arise from the ability to
continuously model and schedule memory—enabling LLMs to maintain contextual consistency, adapt to
evolving knowledge, and support iterative refinement across tasks.
ModelPerformance
Pre-trainingPost-trainingMem-trainingGPT3.5/GPT 4.0GPT-O1/DeepSeek-R1Model XCurrentPast-20232024-20252026…Post-training ScalingMem-trainingScalingWhat’s the Next Scaling Law？
Pre-training Scaling
Figure 2 The next leap in model capability evolution
hinges on the introduction of memory systems, marking a
paradigm shift toward “memory training”.To this end, we introduce MemOS —a prototype
system designed to support a new memory-centric
training paradigm , where learning and inference are
no longer separate phases but part of a unified,
memory-driven process. MemOS not only enables
structured memory storage, interface-level invoca-
tion, and lifecycle management, but also provides
unified scheduling and version control mechanisms
that constitute the foundational infrastructure for
sustainable intelligence evolution. In our design vi-
sion,MemOS treats memory as a schedulable core
resource, breaking down silos between agents, users,
applications, and sessions. It adopts evolution as
a central management objective—supporting mem-
ory recomposition, migration, and fusion to facil-
itate long-term capability growth. Simultaneously,
governance is a foundational pillar: MemOS inte-
grates access control, traceability, and interpretabil-
ity mechanisms to ensure safe and compliant model
operation in complex environments.
4MemOS
4.1 Types of Memory in MemOS
InMemOS , memory is not merely a container of knowledge, but serves as the continuous substrate for
perception, understanding, and action within the model. To systematically support LLMs in evolving across
diverse tasks and scenarios, MemOS classifies memory into three core types: Parametric Memory ,Activation
Memory , and Plaintext Memory . Each type differs in its representation, lifecycle, and invocation mechanism,
collectively forming the multi-layered structure of an intelligent agent’s cognitive system.
Parametric Memory refers to long-term knowledge encoded directly into model weights through pretraining
or fine-tuning, embedded within feedforward and attention layers. It can participate in inference without
the need for external retrieval. This memory type underpins fundamental language understanding, general
knowledge, and skill modules—serving as the backbone for zero-shot generation and capability-driven agents.
3

InMemOS , parametric memory includes not only foundational language capabilities but also supports
modular, domain-specific injection—such as legal or medical knowledge—via pluggable LoRA-based modules
for efficient composition and reuse.
Activation Memory denotes the transient cognitive state generated during inference, including hidden layer
activations, attention weights, and KV-cache structures. It plays a critical role in context awareness, instruction
alignment, and behavior modulation. MemOS treats activation memory as a “working memory” layer,
enabling dynamic scheduling for tasks such as context persistence, stylistic control, and behavioral supervision.
Frequently accessed activation states—such as KV-caches or attention patterns—can be transformed into
semi-structured fragments or parametric modules, allowing short-term memory to persist and evolve over
time.
Plaintext Memory comprises explicit knowledge retrieved from external sources, characterized by properties
such as editability, shareability, and governance compatibility. Typical formats include documents, knowledge
graphs, and prompt templates. This memory type addresses the limitations of context window size and
fixed parameters, enabling rapid knowledge updates, personalized injection, and multi-agent collaboration.
InMemOS , plaintext memory contributes to inference context generation and supports versioning, access
control, and invocation tracing—serving as the foundation of knowledge governance.
Thesethreetypesofmemoryareunifiedunderastandardoperationalabstractionin MemOS:the Memory Cube
(MemCube) , which supports cross-type scheduling, lifecycle management, and structured fusion. By enabling
transformation pathways between memory types (e.g., Activation →Plaintext, Plaintext →Parametric),
MemOS establishes a scalable memory runtime that elevates LLMs from mere language generators to
memory-enabled, adaptive, and continually evolving agents.
Consolidation Pathways
Activation PathwaysEncoding Caching
Latent 
Activation DecodingFine Tuning
Parametric DecodingPlaintext Memory
•Explicitly Stored
•Structurally Organized
e.g. Clinical Knowledge 
& Drug Information
Activation Memory
•Inference -Coupled
•State -Responsive
e.g. Doctor's Expressive 
Patterns
Plaintext MemoryParameter Memory
•Implicitly Embedded
•Statically Encoded
e.g. Symptom-to -
Mechanism Reasoner
Figure 3 Transformation paths among three types of memory, forming a unified, controllable, and evolvable memory
space.
4.2 Memory Cube (MemCube) as a Core Resource
InMemOS , the key to unifying and evolving heterogeneous memory resources lies in standardizing their
representation and management mechanisms. To this end, we introduce MemCubeas the system’s fundamental
encapsulation unit (see Figure 4). The memory resources of LLMs span parametric knowledge, KV-caches,
and externally injected content—each differing in origin, lifecycle, and invocation semantics. MemCube unifies
these heterogeneous forms through a consistent data structure and interface, encapsulating both a semantic
payload and structured metadata to enable uniform scheduling, access control, and lifecycle governance.
MemCube metadata is organized into three categories to support memory identification, control, and evolution:
Descriptive Metadata Used to identify the memory unit and define its semantic role. This includes
timestamps (for creation or updates), origin signatures (e.g., user input, inference output), and semantic types
(e.g., user preference, task prompt, domain knowledge).
4

Memory Cube (MemCube)
Metadata Header
"meta": {
"created": "2025 -04-10",
"last_used": "2025-05-01",
"source": "session_3894","model": "LLaMA3-8B","usage": 78,"priority": "mid",
"expires": "2025-06-01",
"access": ["user_483", "admin"],"tags": ["non-sensitive"],
"embedding_fp": "[dim=128]",
"storage_mode": "compressed",}# Lifecycle
# Access Control
# Storage ProfileMemory Payload
"payload": {"type": "explicit",
"format": "text","content": "You are a helpful 
assistant in climate policy."}Plaintext Content
"payload": {"type": "activation",
"format": "tensor","injection_layer": 12,"value": "[tensor]"}Activation State
"payload": {"type": "parametric",
"format": "lora_patch","module": "mlp.6.down_proj","value": "[low- rank-delta]"}Parametric PatchMemScheduler
•Context -Aware Matching
•Priority-Based Loading
•Fine-Grained Access
•Memory Lifecycle Control
•Runtime Memory Injection…
Callable Operations
•Attach External Prompt
•Inject Activation Bias
•Edit Param Block
•Merge Memory Versions
•Evict Aged Memory
•Adjust Access Rights…Prompt: Hello,LLM! 
I need your help!
Figure 4 MemCube: a unified abstraction for heterogeneous memory, comprising a metadata header and semantic
payload—serving as the smallest execution unit of memory in MemOS.
Governance Attributes Enable safe and controlled usage in multi-user environments. These include access
permissions, lifespan policies (e.g., time-to-live or frequency-based decay), priority levels, and compliance
mechanisms such as sensitivity tags, watermarking, and access logging.
Behavioral Indicators Capture runtime usage patterns—automatically collected metrics such as access fre-
quency, context relevance, and version lineage—that inform dynamic scheduling and cross-type transformation.
This mechanism supports automatic adaptations, such as:
•Plaintext ⇒Activation : Frequently accessed plaintext memory is converted into activation templates to
reduce re-decoding costs;
•Plaintext/Activation ⇒Parametric : Stable, reusable knowledge is distilled into parametric structures to
boost inference efficiency;
•Parametric ⇒Plaintext : Rarely used or outdated parameters are externalized into editable plaintext for
greater flexibility.
With contextual fingerprinting and policy-aware scheduling, the system enables on-demand activation,
hierarchical caching, and structural evolution—making MemCube a self-aware and continuously adaptive
memory unit.
4.3MemOS Architecture
To support unified and adaptive memory handling in LLMs, provides an execution framework for memory
parsing, scheduling, and governance. As shown in Figure 5, it manages the full memory lifecycle via the
MemoryCube abstraction. MemOS adopts a modular three-layer architecture, forming a closed-loop memory
governance framework across the Interface Layer, Operation Layer, and Infrastructure Layer (see Figure 6).
TheInterface Layer serves as the system entry point, responsible for parsing natural language requests,
identifying memory-related intents, and invoking standardized Memory APIs. The built-in MemReader
component translates user inputs into structured memory operation chains. The Operation Layer functions
as the central controller, orchestrating components such as MemScheduler ,MemLifecycle , and MemOperator
to support task-aware scheduling, lifecycle control, and structural organization across users and workflows.
TheInfrastructure Layer provides the foundational support for reliable execution, offering memory storage,
access control, and cross-platform interoperability through modules such as MemVault ,MemGovernance , and
MemStore .
5

MemVaultActivation MemoryPlaintext MemoryMemoryTracingMemScheduler
……Vector DBGraph DBMemGovernanceExpiry PolicyWatermarking ServicePrivacy ProtectionAccess ControlMemDecoder
MemLoaderInstall Memory: Cloud / LocalMemory InstantiationMemDumperArchive Memory (Cloud/Local)PromptMemory Augmentation（Auto）Inference
MemBlock
MemStorePurchaseMemory API/PipelineMemory Update（API）Memory Provenance （API）Memory Transfer（Pip）Memory Puriﬁcation（Pip）Memory Rollback（Pip）Memory Log Query（API）MemLifecycle Manage.
PublishIndustry MemoryScenario MemoryExpert MemoryUser MemoryMemory Persistence
Pipeline MemoryExternal Parametric MemoryIntrinsic Parameter MemoryMemoryUpdatingMemory EncodingMemory CachingLarge Language ModelsAgent/User/Pipeline …MemOS
Business ApplicationMemOperatorMemory Organization Memory Search Figure 5 Overview of the MemOS architecture: showing the end-to-end memory lifecycle from user input to API
parsing, scheduling, activation, governance, and evolution—unified via MemCube.
Interface Layer: Memory API and Pipeline The Interface Layer is centered around a unified Memory
API, offering key interfaces including Provenance API ,Update API , and LogQuery API —used respectively
for annotating memory sources, updating memory contents, and querying usage traces. All operations are
encapsulated within the MemoryCube structure and governed by access control mechanisms provided through
MemGovernance . To support multi-stage and composable workflows, MemOS introduces a pipeline-style
operation chain mechanism. Each pipeline node transmits context, state, and intermediate outputs via
MemoryCube , enabling transaction control, customizable topologies, and DAG-based scheduling. Developers
can construct common operation patterns (e.g., Query–Update–Archive) to enable reuse across multi-model
collaboration scenarios and ensure consistent memory operations.
Operation Layer: Memory Scheduling and Lifecycle Management The Operation Layer orchestrates
memory scheduling, lifecycle evolution, and organization. MemScheduler dynamically selects parametric,
activation, or plaintext memory based on user-, task-, or organization-level context, supporting pluggable
strategies such as least-recently-used (LRU), semantic similarity, and label-based matching. MemLifecycle
models the memory lifecycle as a state machine and supports version rollback and freezing mechanisms
to ensure auditability and temporal consistency. MemOperator manages memory through tagging systems,
graph-based structures, and multi-layer partitions, enabling hybrid structural and semantic search. Retrieved
results are linked back to MemScheduler to determine activation paths. Frequently accessed memory entries
are cached at an intermediate layer to optimize performance. Collectively, these components enable effective
structuring, precise invocation, and robust reasoning across tasks and agents.
Infrastructure Layer: Governance and Memory Store The Infrastructure Layer governs memory compli-
ance, storage, and circulation, ensuring system trustworthiness and long-term evolvability. MemGovernance
enforces access permissions, lifecycle policies, and audit trails to ensure secure and accountable memory opera-
tions in multi-user environments. MemVault manages diverse memory repositories and provides unified access
across heterogeneous storage backends. MemLoader and MemDumper facilitate structured memory migration
6

across platforms and agents while preserving contextual integrity. MemStore supports the open publishing
and subscription of memory units, enabling multi-model knowledge sharing and collaborative execution.
Overall, the system operates through a closed-loop Memory I/O Path, with all modules interfacing via
theMemoryCube abstraction. It supports view customization, access isolation, and extensibility to future
multi-modal scenarios.
4.4 System Execution Flow
As illustrated in Figure 6, a MemOS execution begins with a user prompt or triggered task, parsed by
MemReader into a structured Memory API call. This call initiates a pipeline, where context and state are passed
viaMemoryCube units. MemScheduler then selects relevant memory (parametric, activation, or plaintext) based
onaccesspatternsandschedulingpolicies. Retrievedunitsareinjectedintothereasoningcontext. MemOperator
organizes memory semantically and structurally, while MemLifecycle governs state transitions. Archived
memory is persisted in MemVault , managed by MemGovernance , and can be uploaded to or downloaded from
MemStore for inter-agent collaboration. Migration between agents is supported by MemLoader /MemDumper .
This process forms a closed-loop memory flow—from input to activation, transformation, storage, and
reuse—driven by declarative policies and executed through the MemoryCube abstraction.
Interface Layer Prompt
My dog needs help!
Operation Layer
Infrastructure LayerMemOS’s Three-Layer Architecture Message Input
Message OutputMemReader
Memory PipelineMemory API
MemScheduler
MemOperator …MemLifecycle
MemVault
Loader/Dumpe
rMemStore …MemGovernance
…
API/Pipeline Call
Cube Operations
MemoryCube
(extracted)
Meta
HeaderMem
PayloadSemantic Parse
The user has a dog.
Scheduler CallMemory-equipped 
conversationOutput with memory
MemoryCube
(retrieved)
Meta
HeaderMem
PayloadTo raise a dog, you 
need to prepare a 
comfortable…
Figure 6 The three-layer architecture and memory I/O path of MemOS. From user input to scheduling and memory
injection to response generation, each phase is executed via standardized MemoryCube structures that enable traceable
and structured memory lifecycle management.
5 Conclusion
In this work, we introduce a memory operating system designed for Large Language Models, aimed at
collaboratively building foundational memory infrastructure for next-generation LLM applications.
MemOS provides a unified abstraction and integrated management framework for heterogeneous memory types,
including parametric memory, activation memory, and explicit plaintext memory. We propose a standardized
memory unit, MemCube, and implement key modules for scheduling, lifecycle management, structured storage,
and transparent augmentation. These components collectively enhance reasoning coherence, adaptability, and
system scalability in LLMs. Building on this foundation, we envision a future intelligent ecosystem centered
on modular memory resources and supported by a decentralized memory marketplace. This paradigm shift
enables the creation of next-generation AI systems capable of continual learning and long-term evolution.
Looking ahead, we plan to explore the following directions:
•Cross-LLM Memory Sharing : Enable interoperability and module reuse across different foundation
models by sharing parametric and activation memories. To support consistent semantics and secure
exchange, we plan to extend the Memory Interchange Protocol (MIP) to define standard formats,
7

compatibility rules, and trust mechanisms for cross-model/app memory transmission—facilitating
collaborative knowledge transfer among agents.
•Self-Evolving MemBlocks : Develop memory units capable of self-optimization, reconstruction, and
evolution based on usage feedback, reducing the need for manual maintenance and supervision.
•Scalable Memory Marketplace : Establish decentralized mechanisms for memory exchange, supporting
asset-level transactions, collaborative updates, and distributed evolution to foster a sustainable AI
ecosystem.
Overall, with the introduction of MemOS, we aim to transition LLMs from closed, static generation systems
to continuously evolving intelligent agents equipped with long-term memory, integrated knowledge, and
behavioral plasticity. MemOS not only addresses critical architectural limitations in current models but also
lays the groundwork for cross-task, cross-platform, and multi-agent collaborative intelligence. We look forward
to advancing the frontiers of MemOS in collaboration with the community, making memory a first-class
computational resource in the age of general-purpose AI.
References
[1]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen
Krueger, TomHenighan, RewonChild, AdityaRamesh, DanielM.Ziegler, JeffreyWu, ClemensWinter, Christopher
Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner,
Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are Few-Shot Learners,
July 2020. URL http://arxiv.org/abs/2005.14165 . arXiv:2005.14165 [cs].
[2]Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing Factual Knowledge in Language Models, September 2021.
URL http://arxiv.org/abs/2104.08164 . arXiv:2104.08164 [cs].
[3]Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. Walking down the memory maze:
Beyond context limit through interactive reading. CoRR, abs/2310.05029, 2023. doi: 10.48550/ARXIV.2310.05029.
URL https://doi.org/10.48550/arXiv.2310.05029 .
[4]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-
ready ai agents with scalable long-term memory, 2025. URL https://arxiv.org/abs/2504.19413 .
[5]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirec-
tional Transformers for Language Understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, edi-
tors,Proceedings ofthe2019Conference oftheNorthAmerican Chapter oftheAssociation forComputational
Linguistics: HumanLanguage Technologies, Volume1(LongandShortPapers), pages 4171–4186, Minneapo-
lis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL
https://aclanthology.org/N19-1423/ .
[6]Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. Get More with
LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference. June 2024. URL
https://openreview.net/forum?id=uhHDhVKFMW .
[7]Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong,
and Jeff Z. Pan. Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions, May 2025.
URL http://arxiv.org/abs/2505.00675 . arXiv:2505.00675 [cs].
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan
Larson. From local to global: A graph RAG approach to query-focused summarization. CoRR, abs/2404.16130,
2024. doi: 10.48550/ARXIV.2404.16130. URL https://doi.org/10.48550/arXiv.2404.16130 .
[9]Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, and Tat-seng
Chua. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models, March 2025. URL http:
//arxiv.org/abs/2410.02355 . arXiv:2410.02355 [cs].
[10]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. CoRR,
abs/2312.10997, 2023. doi: 10.48550/ARXIV.2312.10997. URL https://doi.org/10.48550/arXiv.2312.10997 .
8

[11]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented
generation. CoRR, abs/2410.05779, 2024. doi: 10.48550/ARXIV.2410.05779. URL https://doi.org/10.48550/
arXiv.2410.05779 .
[12]Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically
inspired long-term memory for large language models. In Amir Globersons, Lester Mackey, Danielle Belgrave,
Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances inNeuralInformation
Processing Systems 38:AnnualConference onNeuralInformation Processing Systems 2024,NeurIPS 2024,
Vancouver, BC,Canada, December 10-15,2024, 2024. URL http://papers.nips.cc/paper_files/paper/
2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html .
[13]Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna,
Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less
training data and smaller model sizes. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors,
Findings oftheAssociation forComputational Linguistics: ACL2023, pages 8003–8017, Toronto, Canada,
July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.507. URL https:
//aclanthology.org/2023.findings-acl.507/ .
[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. LoRA: Low-Rank Adaptation of Large Language Models, October 2021. URL http://arxiv.org/abs/
2106.09685 . arXiv:2106.09685 [cs].
[15]Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through
Memorization: Nearest Neighbor Language Models. September 2019. URL https://openreview.net/forum?id=
HklBjCEKvH .
[16]Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao
Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In
Proceedings ofthe29thSymposium onOperating Systems Principles , pages 611–626, Koblenz Germany, October
2023. ACM. ISBN 979-8-4007-0229-7. doi: 10.1145/3600006.3613165. URL https://dl.acm.org/doi/10.1145/
3600006.3613165 .
[17]Xiang Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Shichao Song, Hanyu Wang, Jiawei Yang, Feiyu Xiong, Bo Tang,
and Chenyang Xi. Empowering large language models to set up a knowledge retrieval indexer via self-learning.
CoRR, abs/2405.16933, 2024. doi: 10.48550/ARXIV.2405.16933. URL https://doi.org/10.48550/arXiv.2405.
16933.
[18]Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the Middle: How Language Models Use Long Contexts. Transactions oftheAssociation forComputational
Linguistics , 12:157–173, 2024. doi: 10.1162/tacl_a_00638. URL https://aclanthology.org/2024.tacl-1.9/ .
Place: Cambridge, MA Publisher: MIT Press.
[19]Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and Editing Factual Associations in
GPT, January 2023. URL http://arxiv.org/abs/2202.05262 . arXiv:2202.05262 [cs].
[20]Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-Editing Memory in a
Transformer, August 2023. URL http://arxiv.org/abs/2210.07229 . arXiv:2210.07229 [cs].
[21]Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,
Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow
instructions with human feedback, March 2022. URL http://arxiv.org/abs/2203.02155 . arXiv:2203.02155 [cs].
[22]Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
MemGPT: Towards LLMs as Operating Systems, February 2024. URL http://arxiv.org/abs/2310.08560 .
arXiv:2310.08560 [cs].
[23]Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, and Yong Wu. Cognitive Memory in Large Language Models,
April 2025. URL http://arxiv.org/abs/2504.02441 . arXiv:2504.02441 [cs].
[24] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and
Yiqun Liu. Parametric Retrieval Augmented Generation, January 2025. URL http://arxiv.org/abs/2501.15915 .
arXiv:2501.15915 [cs].
9

[25]Nishant Subramani, Nivedita Suresh, and Matthew Peters. Extracting Latent Steering Vectors from Pretrained
Language Models. In Findings oftheAssociation forComputational Linguistics: ACL2022, pages 566–581,
Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.48. URL
https://aclanthology.org/2022.findings-acl.48 .
[26]Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. Better wit than wealth: Dynamic Parametric
Retrieval Augmented Generation for Test-time Knowledge Enhancement, March 2025. URL http://arxiv.org/
abs/2503.23895 . arXiv:2503.23895 [cs].
[27]Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte
MacDiarmid. Steering Language Models With Activation Engineering, October 2024. URL http://arxiv.org/
abs/2308.10248 . arXiv:2308.10248 [cs].
[28]Tianlong Wang, Xianfeng Jiao, Yinghao Zhu, Zhongzhi Chen, Yifan He, Xu Chu, Junyi Gao, Yasha Wang,
and Liantao Ma. Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for
Diverse Hallucinations Categories. In Proceedings oftheACMonWebConference 2025, WWW ’25, pages
2562–2578, New York, NY, USA, April 2025. Association for Computing Machinery. ISBN 979-8-4007-1274-6. doi:
10.1145/3696410.3714640. URL https://dl.acm.org/doi/10.1145/3696410.3714640 .
[29]Jiale Wei, Xiang Ying, Tao Gao, Fangyi Bao, Felix Tao, and Jingbo Shang. Ai-native memory 2.0: Second me,
2025. URL https://arxiv.org/abs/2503.08102 .
[30]Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong
Liu. From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs, April 2025.
URL http://arxiv.org/abs/2504.15965 . arXiv:2504.15965 [cs].
[31]Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang Chen, Yixin Liu, Ruoxi Chen, and Lichao Sun. Noderag:
Structuring graph-based rag with heterogeneous nodes, 2025. URL https://arxiv.org/abs/2504.11544 .
[32]Xin Xu, Wei Xu, Ningyu Zhang, and Julian McAuley. BiasEdit: Debiasing Stereotyped Language Models via
Model Editing, March 2025. URL http://arxiv.org/abs/2503.08588 . arXiv:2503.08588 [cs].
[33]Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng,
Huajun Chen, and Ningyu Zhang. EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language
Models, April 2025. URL http://arxiv.org/abs/2504.15133 . arXiv:2504.15133 [cs].
[34]Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu, Zhiyu Li, Bo Tang, Wenqiang Wei, Jinbo Wang, Zeyun Tang,
Shichao Song, Chenyang Xi, Yu Yu, Kai Chen, Feiyu Xiong, Linpeng Tang, and Weinan E. $\text{Memory}^3$:
Language Modeling with Explicit Memory. JournalofMachine Learning , 3(3):300–346, January 2024. ISSN
2790-2048, 2790-203X. doi: 10.4208/jml.240708. URL http://arxiv.org/abs/2407.01178 . arXiv:2407.01178
[cs].
[35]Peiru Yang, Xintian Li, Zhiyang Hu, Jiapeng Wang, Jinhua Yin, Huili Wang, Lizhi He, Shuai Yang, Shangguang
Wang, Yongfeng Huang, and Tao Qi. Heterag: A heterogeneous retrieval-augmented generation framework with
decoupled knowledge representations, 2025. URL https://arxiv.org/abs/2504.10529 .
[36]Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu
Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie,
Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, and Huajun Chen. A Comprehensive Study
of Knowledge Editing for Large Language Models, November 2024. URL http://arxiv.org/abs/2401.01286 .
arXiv:2401.01286 [cs].
[37]Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, and
Mengnan Du. Explainability for large language models: A survey, 2023. URL https://arxiv.org/abs/2309.
01029.
[38]Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao
Zhang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A survey. CoRR, abs/2402.19473,
2024. doi: 10.48550/ARXIV.2402.19473. URL https://doi.org/10.48550/arXiv.2402.19473 .
[39]Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang,
Junjie Zhang, Zican Dong, et al. A survey of large language models. arXivpreprint arXiv:2303.18223 , 1(2), 2023.
10