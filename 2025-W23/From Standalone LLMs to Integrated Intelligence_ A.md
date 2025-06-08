# From Standalone LLMs to Integrated Intelligence: A Survey of Compound Al Systems

**Authors**: Jiayi Chen, Junyi Ye, Guiling Wang

**Published**: 2025-06-05 02:34:43

**PDF URL**: [http://arxiv.org/pdf/2506.04565v1](http://arxiv.org/pdf/2506.04565v1)

## Abstract
Compound Al Systems (CAIS) is an emerging paradigm that integrates large
language models (LLMs) with external components, such as retrievers, agents,
tools, and orchestrators, to overcome the limitations of standalone models in
tasks requiring memory, reasoning, real-time grounding, and multimodal
understanding. These systems enable more capable and context-aware behaviors by
composing multiple specialized modules into cohesive workflows. Despite growing
adoption in both academia and industry, the CAIS landscape remains fragmented,
lacking a unified framework for analysis, taxonomy, and evaluation. In this
survey, we define the concept of CAIS, propose a multi-dimensional taxonomy
based on component roles and orchestration strategies, and analyze four
foundational paradigms: Retrieval-Augmented Generation (RAG), LLM Agents,
Multimodal LLMs (MLLMs), and orchestration-centric architectures. We review
representative systems, compare design trade-offs, and summarize evaluation
methodologies across these paradigms. Finally, we identify key
challenges-including scalability, interoperability, benchmarking, and
coordination-and outline promising directions for future research. This survey
aims to provide researchers and practitioners with a comprehensive foundation
for understanding, developing, and advancing the next generation of
system-level artificial intelligence.

## Full Text


<!-- PDF content starts -->

arXiv:2506.04565v1  [cs.MA]  5 Jun 2025From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI
Systems
JIAYI CHEN, New Jersey Institute of Technology, USA
JUNYI YE, New Jersey Institute of Technology, USA
GUILING WANG, New Jersey Institute of Technology, USA
Compound AI Systems (CAIS) is an emerging paradigm that integrates large language models (LLMs) with external components,
such as retrievers, agents, tools, and orchestrators, to overcome the limitations of standalone models in tasks requiring memory,
reasoning, real-time grounding, and multimodal understanding. These systems enable more capable and context-aware behaviors by
composing multiple specialized modules into cohesive workflows. Despite growing adoption in both academia and industry, the CAIS
landscape remains fragmented, lacking a unified framework for analysis, taxonomy, and evaluation. In this survey, we define the
concept of CAIS, propose a multi-dimensional taxonomy based on component roles and orchestration strategies, and analyze four
foundational paradigms: Retrieval-Augmented Generation (RAG), LLM Agents, Multimodal LLMs (MLLMs), and orchestration-centric
architectures. We review representative systems, compare design trade-offs, and summarize evaluation methodologies across these
paradigms. Finally, we identify key challenges—including scalability, interoperability, benchmarking, and coordination—and outline
promising directions for future research. This survey aims to provide researchers and practitioners with a comprehensive foundation
for understanding, developing, and advancing the next generation of system-level artificial intelligence.
CCS Concepts: •Computing methodologies →Artificial intelligence ;•General and reference →Surveys and overviews .
Additional Key Words and Phrases: Compound AI systems, large language models, retrieval-augmented generation, LLM agents,
multimodal large language models, AI orchestration, tool-augmented model
1 Introduction
Large Language Models (LLMs) based on the Transformer architecture [ 178] have rapidly evolved from academic
prototypes to foundational infrastructure for modern artificial intelligence. These models now support billion-user
chat interfaces, enterprise copilots, scientific discovery tools, and automated code generation systems. Flagship LLMs
such as GPT-4 [ 2], Gemini [ 169], and Claude [ 24] routinely surpass human baselines in reasoning and natural language
processing benchmarks, and the global market for generative AI is projected to exceed $1.3 trillion by 2030 [12].
However, the same properties that make LLMs compelling—massive pretraining on static corpora and autoregressive
token prediction—also introduce structural limitations. First, hallucination : LLMs may produce fluent but factually
inaccurate output, undermining trust in high-stakes domains such as healthcare, law, and scientific analysis. Second,
staleness : LLMs cannot access post-training knowledge, limiting their responsiveness to emerging facts. Third,
bounded reasoning : finite context windows and inference budgets constrain multi-hop reasoning and long-horizon
task decomposition. These limitations impede the safe and effective deployment of LLMs in dynamic, real-world
environments that require recency, factual reliability, and compositional reasoning.
Authors’ Contact Information: Jiayi Chen, jc2693@njit.edu, New Jersey Institute of Technology, Newark, New Jersey, USA; Junyi Ye, jy394@njit.edu, New
Jersey Institute of Technology, Newark, New Jersey, USA; Guiling Wang, gwang@njit.edu, New Jersey Institute of Technology, Newark, New Jersey, USA.
2025. Manuscript submitted to ACM
Manuscript submitted to ACM 1

2 Chen et al.
To overcome these barriers, the community is converging on a new systems paradigm: Compound AI Systems
(CAIS) . We define CAIS as modular and extensible architectures that integrate LLMs with specialized external compo-
nents, including high-recall retrievers, tool-using agents, symbolic planners, long-term memory modules, multimodal
encoders, and orchestration frameworks, to perform complex, dynamic, and high-precision tasks. By decoupling sub-task
responsibilities and intelligently routing them to appropriate modules, CAIS extends the capabilities of LLMs far beyond
what any monolithic model can achieve.
Early deployments demonstrate the transformative potential of this paradigm. Retrieval-augmented assistants, such
as Perplexity.ai, provide real-time answers with chain-of-thought citations [ 128]. GitHub Copilot-X orchestrates code
reasoning, repository search, and test generation, increasing developer throughput by over 55% [ 126]. In radiology,
multimodal pipelines coupled with rule-based triage agents have reduced report turnaround by 30% while maintaining
expert-level accuracy [ 135]. These cases mark a shift in design philosophy: from LLMs as autonomous soloists to
conductors orchestrating heterogeneous AI ensembles.
Despite growing interest, a systematic understanding of CAIS remains elusive. Recent literature has addressed isolated
components of this ecosystem, including surveys on retrieval-augmented generation (RAG) [ 43], LLM-based agents [ 84],
multi-agent frameworks [ 53], and LLM-driven system optimization [ 90]—but these efforts remain isolated. Some
concentrate on narrow aspects such as prompt engineering [ 110], benchmark analysis [ 44], or agent communication
protocols [ 197], without addressing the architectural interactions and trade-offs across the entire CAIS stack. These
works contribute valuable insights into their respective domains, yet none provides a holistic, system-level synthesis.
In contrast, our survey offers the first unified taxonomy and architectural analysis of Compound AI Systems.
We integrate four foundational axes—retrieval, agency, multimodal perception, and orchestration—into a cohesive
framework. We identify recurring design patterns, trade-offs, and failure modes, and propose an evaluation paradigm
that addresses factuality, efficiency, safety, and human-centered utility. By synthesizing this fragmented landscape, we
fill a critical gap in the literature and establish a foundation for the next generation of modular, composable AI systems.
Our analysis is structured along four axes that reflect the core dimensions of CAIS. Specifically, we make the following
contributions:
(1)Multi-dimensional taxonomy. We organize the CAIS landscape across four orthogonal axes: Retrieval-
Augmented Generation (RAG), LLM Agents, Multimodal LLMs (MLLMs), and Orchestration Frameworks, estab-
lishing a shared vocabulary for comparative study.
(2)Architectural synthesis. Drawing on more than 120 peer-reviewed and industrial sources, we distill architectural
blueprints, coordination strategies, and component interactions, highlighting recurring patterns and failure
modes across the CAIS design space.
(3)Evaluation framework. We review existing benchmarks and propose metrics that jointly assess factual accuracy,
robustness, efficiency, and alignment with human goals—essential qualities for trustworthy CAIS deployments.
(4)Research agenda. We outline open challenges in component interoperability, scalable orchestration, privacy-
preserving memory, and sustainable compute, charting a roadmap for future research and development.
The remainder of this survey is structured as follows. Section 2 formalizes the notion of Compound AI Systems and
introduces a high-level architectural template. Sections 3–6 examine each axis of our taxonomy in depth, linking design
decisions to system limitations. Section 7 reviews evaluation methodologies and benchmarks. Section 8 synthesizes
open research directions, and Section 9 concludes with reflections and future outlook.
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 3
2 Architecture of Compound AI Systems
2.1 Definition
The term: Compound AI Systems (CAIS), first appeared in a post from Berkeley Artificial Intelligence Research (BAIR)
[212]. In this post, the authors illustrate the trend where better performance of AI is increasingly obtained by compound
systems, instead of standalone LLMs. In this survey, we provide a comprehensive definition: A CAIS is a framework that
integrates LLMs, components, and system designs. Its purpose is to address complex tasks that exceed the capabilities
of standalone LLMs. It can be described using four dimensions: Retrieval-Augmented Generation (RAG), Multimodal
Large Language Model (MLLM), LLM Agent, and Orchestration of CAIS. Pre-training, Continuous pretraining, and
fine-tuning without combining other components are not considered CAIS as they are refining LLM itself.
2.2 General Representation of Compound AI Systems
From a general perspective, CAIS can be understood as a system comprising integrated interacting components and
core LLMs. Components can be external tools, models, agents, multimodal encoders, prompt engineering techniques,
and self-optimization. LLMs can be any kind of LLM, including general-purpose LLMs or fine-tuned LLMs trained for
specific tasks. Components and LLMs are two essential parts of CAIS, without either one of them, the Compound AI
System is not valid. As such, the general formula of the Compound AI System can be described as:
Compound AI System =𝑓(𝐿,𝐶,𝐷) (1)
Where:
•𝐿={LLM 1,LLM 2,..., LLM 𝑀}: The set of all LLMs in the system. Each LLM 𝑗represents an individual LLM, with
𝑗ranging from 1 to 𝑀.
•𝐶={𝐶1,𝐶2,...,𝐶 𝑁}: The set of all components in the system. Each component 𝐶𝑖provides:
–𝑓𝑖: Functionality of the component.
–𝑜𝑖: Output generated by the component.
–𝑝𝑖: Parameters that control its behavior.
•𝐷: The system design that defines the architecture and interactions between 𝐿(LLMs) and𝐶(components). This
includes orchestration, topology, or other high-level design principles.
The function 𝑓(𝐿,𝐶,𝐷)abstracts the interactions and integration of:
(1)LLMs (𝐿): The core language models contributing to the AI system.
(2)Components ( 𝐶): Auxiliary modules or subsystems that enhance, extend, or refine the functionality of the
LLMs.
(3)Design (𝐷): The overarching framework or strategy guiding how LLMs and components are connected, utilized,
and coordinated to achieve the desired outcomes.
LLMs. The LLMs are the essential part of CAIS. Any kind of LLM can be incorporated into CAIS. The operation of
CAIS includes fine-tuning LLM or directly using a general-purpose LLM. However, CAIS generally doesn’t include
Pretraining of LLM or Continuous Pretraining of LLM.
Components. Components of CAIS serve as enhancements to the LLMs. It can take many forms, including external
tools, Machine learning models, and promotional techniques, among others. CAIS leverages the strengths of various
components and orchestrates them to maximize end-to-end performance, accuracy, and efficiency.
Manuscript submitted to ACM

4 Chen et al.
System Design. Also known as Orchestration. In CAIS, it involves allocating limited resources to optimize the
performance and operation of components and LLMs within CAIS. Alongside LLMs and Components, a sound system
design is crucial because it defines how different elements interact and contribute to solving complex tasks that a
standalone AI cannot handle.
2.3 Dimensions of Compound AI Systems
CAIS are described in four main dimensions: RAG, MLLM, LLM Agent, and Orchestration of CAIS.
RAG. This dimension describes RAG, which combines retrieval and generation by using an external knowledge
source to fetch relevant context, which is then used by an LLM to produce more informed and accurate outputs.
MLLM. This dimension covers MLLMs, which process and reason over multiple data types (e.g., text, images, audio)
by integrating modality-specific encoders with an LLM for unified understanding and generation.
LLM Agent. An LLM Agent is a system that uses autonomous LLMs to reason, plan, and take actions by interacting
with tools, memory, or environments in an iterative loop.
Orchestration of Compound AI System. This dimension focuses on the orchestration layers and overall design of CAIS,
including elements such as its structure, mechanism, and objectives.
3 Retrieval-Augmented Generation (RAG)
3.1 Motivation and General Framework
LLMs are highly capable but exhibit several limitations, including hallucinations [ 60], outdated knowledge, unstable
contextual understanding, and high computational costs associated with retraining or fine-tuning. RAG addresses these
limitations by incorporating external documents, datasets, or search engines as non-parametric memory [ 79]. This
approach allows RAG to enhance the performance of LLMs in a cost-efficient manner without requiring extensive
retraining on massive datasets. Additionally, RAG provides mechanisms for controlling the quality of the generated
responses, thereby improving their reliability and relevance. Some popular RAG libraries, such as LangChain [ 171],
Haystack [ 30], and LlamaIndex [ 170], enable the efficient integration of retrieval with LLMs for scalable and reliable
applications.
The RAG framework typically consists of two primary phases: Retrieval and Generation. In the retrieval phase,
knowledge documents are segmented into smaller chunks, indexed, and represented as vectors through embedding
techniques. Queries are similarly embedded, and the most relevant top- kchunks are selected based on cosine similarity.
In the generation phase, these retrieved documents serve as contextual input, which is combined with the original
query and processed by the LLM. The LLM then generates a response that is both accurate and contextually grounded
[188].
This survey categorizes the RAG architecture into three key components: the Retriever , the Generator , and RAG
Design . The detailed architecture is illustrated in Figure 2. As shown, the Retriever identifies and retrieves relevant
documents from external databases or search engines through various methods. The Generator processes the retrieved
content to produce retrieval-augmented answers, utilizing either pre-trained or fine-tuned large language models
(LLMs). The RAG design encompasses the overall orchestration of the system, characterized by distinct patterns or
frameworks.
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 5
Retrieval Augmented Generation
Retriever Generator RAG Design
Pre-
Retrieval
Retrieval 
Post-
RetrievalGeneral
LLM
Fine-tuned
LLM
Iterative
Recursive
AdaptivePhase Approach
Sparse
Retriever 
Dense
Retriever 
Graph
Retriever 
Hybrid
Retriever Pattern Framework
Co-training
Prompt
Construction
Modular
RerankingQuery
Preprocessing
Knowledge
Filtering
LLM as
Retriever Knowledge
ReﬁnementIndexing
Fig. 1. Taxonomy of Retrieval-Augmented Generation (RAG) systems. The diagram categorizes the core components into three
primary modules: Retriever, Generator, and RAG design.
3.2 Retriever
The primary role of the retriever in RAG systems is to retrieve information and external knowledge. It is essential
that the retriever effectively identifies and retrieves relevant context to support accurate and meaningful responses.
The retriever in RAG operates along two dimensions: Phase andApproach . With regard to Phase , the retriever can be
categorized into three phases: Pre-Retrieval ,Retrieval , and Post-Retrieval , based on the sequence of the retrieval process.
Pre-Retrieval refers to the process of converting external knowledge into vector representations. This phase includes
query preprocessing, chunking, embedding, and indexing [ 11,32]. A key area of focus is query preprocessing, which
includes query regeneration and query rewriting. Frameworks like DSI-QG regenerate queries to represent documents
more effectively, enabling the filtering of the most relevant results [ 223]. Query rewriting involves introducing a
trainable rewriter, often a small language model, to modify input queries for improved retrieval [ 111]. Moreover,
recent research has introduced models like EMAT [ 192], which utilize key-value memory structures to encode external
knowledge for efficient retrieval and indexing. Similarly, approaches like DSI map document text to unique identifiers
(docids) and retrieve documents by generating docids based on the query [168].
Retrieval involves locating documents relevant to a query by measuring similarity. This phase has seen advancements
aimed at improving the relevance of retrieved documents. For example, Yu et al. [ 207] introduced an information
retrieval-based assertion retrieval method leveraging Jaccard similarity, while KAPING employs semantic similarity to
retrieve relevant triples from knowledge graphs [ 6]. Many other innovative designs and frameworks have also been
proposed to improve retrieval relevance and precision, which will be discussed in detail in the section: RAG Design.
Manuscript submitted to ACM

6 Chen et al.
Post-Retrieval focuses on refining the initial retrieval results through additional steps. A second round of retrieval
may be conducted based on the documents retrieved in the initial phase to enhance retrieval performance. Techniques
such as knowledge filtering reduce the search space by applying specific constraints. For instance, Promptagator
employs a round-trip consistency check to filter out low-quality knowledge [ 28], while FILCO implements fine-grained
sentence-wise filtering of retrieved passages [ 182]. Knowledge refinement further enhances the quality, coherence, and
usability of retrieved information. Examples include LLM-AMT, which uses a knowledge self-refiner to filter and refine
retrieved information for relevance [ 180], and RECOMP, which compresses retrieved documents into concise textual
summaries before appending them as input [194].
Besides knowledge filtering and knowledge refinement, reranking involves reordering and reassessing retrieved
results to maximize relevance. Several advanced reranking techniques have been proposed. The approach by Lazaridou
et al. [ 78] employs reranking of multiple generated answers based on probabilistic scoring for improved output quality.
Glass et al. [ 50] leveraged a BERT-based reranker for score adjustment in their RE2G framework. Lin et al. [ 89] combines
dense retrieval and pair-wise reranking to select a subset of upstream examples that are fine-tuned with the base model
to improve performance on unseen tasks. Additionally, reranking techniques further enhance retrieval quality and
consistency by refining document selection and ordering [66, 138, 145].
From a functional perspective, retrievers can be classified into five types: Sparse Retriever ,Dense Retriever ,Graph
Retriever ,Hybrid Retriever , and LLM as Retriever .Sparse Retriever uses sparse representations of text (e.g., BM25) based
on explicit term matching between queries and documents [ 62].Dense Retriever leverages dense vector representations
(e.g., BERT) to capture semantic similarity between queries and documents [ 92,100,199].Graph Retriever utilizes graph
structures (e.g., knowledge graphs) to locate relevant information by traversing nodes and edges [ 6,38,46].Hybrid
Retriever combines both sparse and dense retrieval approaches to leverage the strengths of explicit term matching and
semantic similarity [ 6,50,105].LLM as Retriever involves the use of LLMs to directly retrieve relevant knowledge based
on input queries [112].
3.3 Generator
The Generator in RAG systems is essentially an LLM. It can be an original pre-trained language model, such as T5 [ 136],
FLAN [ 185] and LLaMA [ 174], or a black-box pre-trained language model, such as GPT-3 [ 14], GPT-4 [ 2], Gemini [ 169],
Claude [ 24]. Alternatively, the generator can also be a fine-tuned language model specifically tailored for a particular
task. For instance, BART [ 79] and T5 [ 63] are fine-tuned alongside the retriever, a process commonly referred to as
co-training or dual fine-tuning, to enhance the quality and consistency of retrieval [ 93]. In other scenarios, the generator
is fine-tuned to effectively filter retrieved results, retaining only relevant documents and discarding irrelevant ones
[106,206,217]. Furthermore, the Generator can be trained and fine-tuned within a reinforcement learning framework
to optimize performance in specific contexts [46].
3.4 RAG Design
RAG design is characterized by two key dimensions: pattern andframework . The pattern dimension describes how RAG
systems retrieve and generate answers, encompassing iterative, recursive, and adaptive approaches. The framework
dimension refers to the structural framework of RAG design, including co-training, prompt construction, and modularity.
Iterative is a design pattern that improves the RAG system through repeated cycles of retrieval and generation
to refine outputs incrementally. For example, DSP refines its search state by iteratively retrieving relevant passages
through repeated queries [ 68]. Similarly, LLM-Augmenter iteratively optimizes prompts and LLM responses until the
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 7
outputs meet a predefined utility threshold [ 125]. RepoCoder retrieves code snippets from a broader repository context
and iteratively improves them to produce accurate, context-aware completions [ 215]. ITER-RETGEN utilizes outputs
from previous iterations to refine retrieval and generation processes, effectively addressing multi-hop reasoning and
complex queries [ 149]. Selfmen integrates self-memory into RAG, using the model’s outputs as iterative memory for
subsequent rounds of generation [23].
Recursive is a design pattern where the retrieval and generation processes are applied in a nested manner to handle
complex queries; it breaks down complex queries into simple ones. For instance, IRCoT combines retrieval with iterative
chain-of-thought reasoning for knowledge-intensive, multi-step question answering [ 175]. RAFT trains LLMs with a
mixture of relevant and irrelevant documents to encourage a focus on pertinent content and generate chain-of-thought
reasoning [ 217]. Additionally, LLM-AMT processes lengthy and information-dense passages by splitting them into
smaller sections for efficient handling [180].
Adaptive is a design pattern that dynamically adjusts the retrieval or generation strategies based on the context
or feedback to optimize performance. For example, SELF-RAG enables LLMs to decide dynamically when to retrieve
information, generate responses, and critique their outputs via self-reflection [ 5]. FLARE actively determines retrieval
timing and content through anticipatory next-sentence predictions, using low-confidence tokens as retrieval triggers [ 64].
Similarly, CON generates concise, contextually relevant reading notes for each retrieved document while systematically
assessing their relevance and reliability [209].
Co-training is a framework where the retriever and generator are jointly trained to enhance their collaboration.
For example, RAG-end2end updates all components of the RAG model asynchronously during training, including the
retriever, generator, and external knowledge base encodings [ 155]. Alternatively, DSI replaces traditional multi-stage
retrieval pipelines with a single Transformer-based model that indexes and retrieves documents directly through its
parameters [168].
Prompt Construction is a framework that focuses on designing and optimizing prompts to enable the generator to
efficiently utilize retrieved information, thereby improving the relevance and accuracy of the final response. For example,
Ren et al. [ 142] present new prompting strategies—priori judgement, which evaluates a question before answering, and
posteriori judgement, which assesses the correctness of the answer to explore the impact of retrieval augmentation.
Modular is a framework where RAG itself is designed as independent modules or applications, allowing for flexibility,
easy replacement, and integration of different components. For instance, R2A adopts a modular architecture that allows
integration with various reinforcement learning algorithms [ 52]. AAR employs a plug-and-play approach, adapting
trained retrievers to function with unseen, larger target language models [ 211]. RETROPROMPT enhances input
sequences through retrieval mechanisms to guide training and improve predictions [ 22], while REPLUG augments
LLMs by incorporating external retrieval systems without altering the LLM’s internal parameters [152].
3.5 Limitations and Future Trends
Although RAG systems have developed rapidly with numerous innovative designs, modern RAG systems still face
several limitations. For instance, seamlessly integrating retrieved information into the generation process remains a
significant challenge, particularly for longer contexts or multi-modal tasks [ 23]. Additionally, issues such as retrieval
quality [ 26], scalability [ 99], and knowledge conflicts or inconsistencies [ 175] continue to hinder the effectiveness of
modern RAG systems.
Recent findings suggest several promising directions for the future development of RAG systems. One emerging
trend is the creation of ready-to-use application frameworks that facilitate direct deployment for domain-specific
Manuscript submitted to ACM

8 Chen et al.
tasks [ 146,177,180,196,199,207]. Another significant direction involves greater integration of multi-modal data,
combining text, images, videos, and audio to address complex and multi-dimensional use cases [ 20,59,104,139,204].
Furthermore, the combination of RAG systems with other machine learning models, such as reinforcement learning
and classification models, offers opportunities to improve control and accuracy [ 52,100,111,180]. Finally, leveraging
LLMs in end-to-end retrieval processes is anticipated to produce tightly integrated pipelines, offering an advantage
over the traditional retrieval-then-rank approach [112, 168].
4 LLM Agent
LLM agents have emerged as a powerful paradigm for enabling intelligent, autonomous behavior. The ecosystem of
LLM agents can be broadly categorized into three interconnected layers: Application Scenario, Agent Framework, and
Agent Mechanism, as illustrated in Figure 2. Together, these layers define the operational architecture and capabilities
of LLM agents, guiding both their theoretical design and practical implementation.
4.1 Application Scenario
At the highest level, LLM agents are applied across a spectrum of real-world and simulated environments, serving as
intelligent assistants, embodied agents, or even assistants in scientific experiments. These scenarios reflect the practical
embodiments of LLM capabilities, where agents are tasked with executing domain-specific objectives, ranging from
simulating human interactions to developing software systems.
4.1.1 General Purposes. General-purpose LLM agents are designed to perform a wide array of tasks across diverse
domains, offering flexible problem-solving capabilities without requiring domain-specific customization. For example,
Gato [ 141] is a generalist agent trained using a transformer-based model that can handle multiple modalities (text,
images, proprioception), tasks (captioning, dialogue, control), and embodiments (robots, Atari). Gato uses a single set of
weights to perform over 600 tasks, demonstrating the feasibility of a multi-task, multi-embodiment agent. Another
instance is MINEDOJO [ 42], a novel framework built on Minecraft that combines thousands of natural language-based
open-ended tasks and an internet-scale multimodal knowledge base to train generalist embodied agents using large-scale
pretraining techniques, enabling agents to perform diverse and natural language-prompted tasks.
4.1.2 Embodiment. An embodied LLM agent is situated within a physical or virtual environment, enabling it to perceive,
act upon, and adapt to dynamic surroundings through multimodal interactions. For the physical environment, Inner
Monologue [ 61] is a representative example. As a framework, it enables LLMs to reason and plan in embodied robotic
environments by incorporating closed-loop language feedback, such as scene descriptions and success detection, without
requiring additional training. The method enables more robust and interactive robotic behavior across simulated and
real-world tasks.
For a virtual environment, Voyager [ 179] introduces the first LLM-powered embodied lifelong learning agent in
Minecraft that autonomously explores, acquires skills, and discovers new tasks using a black-box GPT-4 without
fine-tuning. It leverages an automatic curriculum, a skill library of code-based actions, and iterative prompting to enable
continual open-ended learning.
4.1.3 Miscellaneous Scenarios. Miscellaneous scenarios encompass diverse or emerging application settings for LLM
agents that do not fall neatly into established categories, often applied to various use cases. For example, an LLM
agent could serve as a reward designer. EUREKA [ 113] is a universal reward design algorithm that uses coding LLMs,
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 9
LLM
AgentApplication Scenario
Agent Framework
Agent Mechanismgeneral purposes embodiment misc scenarios
LLM AgentSingle Agent 
Multi-agentmulti-agent collaborative
framework
multi-agent debate
frameworkcommunication strategy
role playing
shared message poolmulti-agent systems
LLM AgentOrchestrates  System
Interactive reasoning loop
tool useﬁne-tune prompt-based structured memory
plan & reason actionenvironment
feedbackself-reﬂection
MCP API callsinformation
retrievalweb search
Fig. 2. A structured overview of LLM agents across three dimensions: application scenarios (e.g., general-purpose, embodied), agent
frameworks (single-agent and multi-agent architectures), and agent mechanisms (system orchestration, reasoning loops, and tool
use).
particularly GPT-4, to automatically generate reward functions for reinforcement learning (RL) tasks, achieving or
exceeding human-level performance across a wide variety of tasks.
LLM Agent can be applied in an autonomous driving scenario. DiLu [ 186], a framework that instills human-like
knowledge into autonomous driving via LLMs, leveraging reasoning, reflection, and memory to improve decision-
making in complex driving environments. It aims to move beyond traditional data-driven approaches by incorporating
a knowledge-driven paradigm based on common-sense reasoning.
LLM Agent assists in scientific experiments as well. Boiko et al., 2023 [ 13] presents an LLM-based Intelligent Agent
system capable of autonomously designing, planning, and executing scientific experiments by combining internet
search, documentation parsing, and automation tools, where it introduce a modular system composed of a Planner,
Manuscript submitted to ACM

10 Chen et al.
Web Searcher, Docs Searcher, Code Execution, and Automation modules, which coordinate via LLMs (primarily GPT-4)
to autonomously perform scientific tasks.
4.2 Agent Framework
The agent framework defines how LLM agents are organized, instantiated, and deployed. This layer encompasses the
majority of paradigms in multi-agent frameworks, including multi-agent collaborative frameworks, multi-agent debate
frameworks, and the workflow of multi-agent systems.
4.2.1 Multi-Agent Collaborative Framework. A multi-agent collaborative framework enables multiple LLM agents to
work together toward a shared objective, often leveraging role specialization, communication protocols, and coordinated
planning. For example, MetaGPT [ 56] is a multi-agent collaborative framework for software development that integrates
Standardized Operating Procedures (SOPs) into LLM-based agent workflows to improve coherence and accuracy.
Moreover, AgentVerse [ 21] is a dynamic multi-agent collaboration framework inspired by human group problem-
solving, where agents can adjust their roles, communicate, and collaborate across various tasks, including software
development, consulting, and gaming. AgentVerse models the problem-solving process as a loop of four stages: expert
recruitment, collaborative decision-making, action execution, and evaluation, allowing dynamic team adjustment based
on feedback. Another example worth mentioning is Talebirad et al., 2023 [ 162], where the authors propose a graph-based
multi-agent collaboration framework where agents and plugins form nodes with defined communication channels,
allowing role-specific LLM agents to collaborate, dynamically generate new agents, provide feedback, and manage
execution.
4.2.2 Multi-Agent Debate Framework. In a multi-agent debate framework, agents are intentionally assigned to argu-
mentative roles, utilizing structured dialogue to explore divergent viewpoints, validate reasoning, or reach consensus
through deliberation. For instance, Du et al. [ 36] propose a multi-agent debate framework in which multiple instances
of language models collaboratively reason, critique, and revise their answers to enhance factuality and reasoning across
various tasks. The authors design a system where several LLM agents independently generate responses, then iteratively
review and revise their answers based on peer responses through multiple rounds of debate, using only prompting
and without model fine-tuning. Furthermore, the Multi-Agent Debate (MAD) framework [ 86] involves two LLM-based
agents (affirmative and negative) debating a topic iteratively with a judge LLM that decides when to stop and which
response is correct, enabling exploration of diverse reasoning paths.
4.2.3 Multi Agent Systems. Multi-agent systems encompass any architecture involving two or more autonomous
agents—LLMs that interact within a shared environment to achieve individual or collective goals. Communication
strategy defines the protocols and methods through which LLM agents exchange information, coordinate actions,
and negotiate meanings within a multi-agent setting. For example, ChatEval [ 16], a multi-agent debate framework
that uses multiple LLMs with diverse roles to collaboratively evaluate generated text, aiming to simulate the quality
and depth of human evaluation. ChatEval enables multiple LLM agents, each with a distinct role prompt (e.g., critic,
scientist), to engage in structured discussion through designed communication strategies (one-by-one, simultaneous, or
summarization-enhanced) before aggregating a final evaluation via majority vote or averaging.
Role playing assigns specific identities, functions, or perspectives to each agent, enabling structured interactions
that reflect diverse viewpoints or specialized expertise. For example, in a multi-agent collaborative framework, agents
are assigned roles such as instructor-assistant, where the former is responsible for raising questions and asking
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 11
follow-up questions. The latter is responsible for addressing the questions and making any necessary corrections in
the conversation’s dialogue. For example, Autogen [ 189] proposes a general-purpose framework for building next-
generation LLM applications through multi-agent conversations, allowing agents to interact via programmable, flexible
chat patterns to solve complex tasks collaboratively. Furthermore, CAMEL [ 81] is a framework that enables multiple
language agents to autonomously cooperate via role-playing, facilitating the scalable generation of high-quality
conversational datasets and the analysis of multi-agent LLM behaviors.
Ashared message pool serves as a centralized communication hub where agents can post, access, and interpret
messages, facilitating asynchronous and collective reasoning. For example, in MetaGPT [ 56], the shared message pool is
a centralized, structured communication system that enables asynchronous collaboration among multiple LLM agents,
each assigned a human-like role. Instead of unstructured dialogue, agents publish and subscribe to typed messages
(e.g., Requirement, Design, Task, Code) with clearly defined sender, receiver, and content fields. This publish-subscribe
model allows agents to coordinate without direct interaction, trace the full workflow, and maintain modular, scalable
collaboration. The shared message pool ensures clarity, reduces hallucinations, and supports flexible system extension,
forming the backbone of MetaGPT’s multi-agent orchestration.
4.3 Agent Mechanism
This layer covers the mechanism of the LLM-driven agent. How the system is organized, how the LLM agent reasons,
plans, perceives feedback from the environment, takes action, and reflects on failures. Moreover, tool use, one of the
most essential features of the LLM agent, is also discussed in this layer.
4.3.1 Orchestration System. An orchestration system coordinates the components and execution flow of an LLM
agent, managing memory, tools, and reasoning steps to ensure coherent and goal-directed behavior. Most of LLM
agent systems are prompt-based systems, which means the agent system guides the agent’s behavior and reasoning by
structuring task instructions, examples, and context directly within the input prompt to the language model, without
modifying its internal parameters. Inner Monologue [ 61], Voyager [ 179], and ChatEval [ 16] are some representative
examples that leverage the prompt-based method to compose their agent systems. However, there are agent systems
that apply fine-tuning as well. For instance, GPT4Tools [ 200], a method that enables open-source language models to
use multi-modal tools by generating a tool-usage instruction dataset via self-instruction from GPT-3.5 and fine-tuning
with LoRA. The author fine-tunes models, such as Vicuna-13B, using LoRA for efficient parameter adaptation.
Structured memory management is an essential component of the LLM agent system. Based on different use cases,
memory storage typically employs a combination of short-term memory, long-term memory, and, in some instances,
episodic memory. For example, CoALA [ 157] is a conceptual framework inspired by cognitive science to organize and
design modular, memory-augmented LLM-based agents, where it introduces a modular agent architecture with working
and long-term memory (episodic, semantic, procedural). Moreover, ExpeL [219] utilizes dual memory systems, where
the agent autonomously collects successful and failed task trajectories, extracts reusable natural language insights, and
retrieves relevant past examples to augment its decision-making during evaluation, all without modifying LLM weights.
4.3.2 Interactive Reasoning Loop. The interactive reasoning loop refers to the iterative process through which an LLM
agent perceives input, plans actions, executes tasks, and incorporates feedback to refine its decisions over time. Plan
and reason refers to the agent formulating a strategy by analyzing the current context, setting subgoals, and logically
determining the following steps to achieve its objective. ReAct [ 203] sets the paradigm of planning and reasoning
for an LLM agent that takes autonomous actions. It’s a prompt-based method that enables LLMs to interleave verbal
Manuscript submitted to ACM

12 Chen et al.
reasoning with action-taking, allowing them to solve complex tasks by dynamically planning, acting in environments,
and adjusting based on feedback. ReAct augments the LLM’s action space to include both environment-changing actions
and language-based reasoning traces, prompting models to alternate between thinking and acting using a unified policy,
guided by few-shot demonstrations. Another representative of work exploring the reasoning capacity of LLMs is Tree
of Thoughts [ 202], a framework that enables language models to solve complex problems by reasoning over a search
tree of intermediate “thoughts,” allowing for planning, exploration, and backtracking, unlike left-to-right token-level
generation. Tree of Thoughts introduces a tree-based problem-solving structure where each node is a “thought” (a
coherent reasoning step). The model generates, evaluates, and searches over these thoughts using LLMs and strategies
like breadth-first or depth-first search to select promising reasoning paths.
Action refers to an agent executing a task or issuing a command—such as calling a tool, generating output, or
interacting with an environment—based on reasoning. The key difference between a standalone LLM and an LLM agent
is that an LLM agent can execute autonomous actions without needing textual instructions from a human.
Environment feedback refers to the agent receiving signals or data from the external environment, reflecting the
outcome of its previous actions and informing future decisions. For example, Voyager [ 179] employs an iterative
prompting loop that refines actions through feedback, execution errors, and self-verification. Moreover, LLM-Planner
[156], a high-level planner that leverages LLMs like GPT-3 to perform few-shot planning for embodied agents by
generating and dynamically updating task plans grounded in real-world environments.
Self-reflection refers to the agent critically evaluating its reasoning and performance, identifying errors or inefficiencies
to improve subsequent behavior. For example, Reflexion [ 153] is a framework where LLM-based agents learn via self-
generated verbal feedback rather than weight updates, using natural language reflections stored in episodic memory to
improve performance across trials. Reflexion combines an LLM-based Actor, Evaluator, and Self-Reflection model in a
loop where agents iteratively generate actions, evaluate performance, and produce verbal feedback for memory-based
in-context learning. Another example of self-reflection is Recursive Criticism and Improvement (RCI) [ 69], which is
a prompting method where the LLM iteratively generates an output, critiques it, and then improves upon it. This is
applied across three stages of action grounding—task grounding, state grounding, and agent grounding—to ensure
outputs are appropriate, feasible, and executable in computer task environments.
4.3.3 Tool Use. Tool use empowers LLM agents to go beyond language generation by interacting with external
services—such as APIs, computational tools, or information sources—to perform complex tasks like data processing,
real-time querying, and integration with third-party applications.
Model Context Protocol (MCP) [ 4] is a standardized interface proposed by Anthropic that enables language models to
interact with external tools, memory, documents, and user interfaces in a structured and flexible manner. It defines how
models receive context and respond with function calls or natural language, facilitating advanced agent behaviors. For
example, MCP servers such as AWS KB Retrieval facilitate retrieval from the AWS Knowledge Base using Bedrock Agent
Runtime. A server like Brave Search enables web and local search capabilities using Brave’s Search API. Moreover, a
server like Filesystem provides secure file operations with configurable access controls.
LLM agents can invoke external services via API Calls , enabling access to dynamic functionalities such as data
processing, computation, or third-party applications. For example, ToolLLM [ 131] is a framework that empowers LLMs
to effectively use over 16,000 real-world APIs by combining a new dataset (ToolBench), a decision-making algorithm
(DFSDT), and a trained model (ToolLLaMA) to achieve robust tool-use capabilities.
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 13
Information retrieval allows agents to access relevant documents or data from structured or unstructured sources to
support grounded reasoning and informed responses. For instance, Sun et al.,2023 [ 160] explored and proposed whether
LLMs like ChatGPT and GPT-4 can serve as effective passage re-ranking agents in information retrieval and introduced
a novel permutation generation method that allows LLMs to directly output ranked passage lists.
Web search enables agents to query the internet in real-time, expanding their knowledge beyond pre-training and
allowing them to answer up-to-date or domain-specific questions. For example, paper MIND2WEB [ 31] proposes
MINDACT, a two-stage framework combining a small LM for webpage element ranking and a large LM for action
prediction in a multiple-choice QA format, enabling the model to operate efficiently on long and noisy web pages.
5 Multimodal Large Language Models (MLLMs)
MLLMs extend the capabilities of LLMs beyond text-based understanding and generation to multimodal comprehension
and content creation. These models are designed to process and generate information across multiple modalities,
including text, images, audio, and video. By incorporating cross-modal learning, MLLMs enhance traditional LLMs,
enabling them to integrate and reason across different types of data.
This section categorizes MLLMs into four aspects: architecture of MLLMs, fusion strategy inside MLLMs, MLLMs’
modality approaches, and tasks of MLLMs. From the architecture standpoint, MLLM has four major components:
Encoder, Visual Projector, Fusion Module, and Core LLM Model. Fusion strategy is another key aspect of MLLMs;
different MLLMs employ various fusion strategies, broadly classified into early fusion, late fusion, cross-modal fusion,
and hybrid fusion. In terms of modality, MLLMs can be further classified by their focused modality approaches: image,
audio, video, or text-rich image. Moreover, based on different tasks and usages, MLLMs can be defined as multimodal or
integrated with other techniques, such as MLLMs with Chain-of-Thought (CoT) or MLLMs with RAG.
5.1 Architecture
5.1.1 Encoder. The Encoder is responsible for extracting meaningful representations from raw input data. Depending
on the modality, different encoders are used. The extracted features serve as high-dimensional embeddings, which are
further processed for alignment with the language model.
Vision encoder extracts high-level feature representations from raw visual inputs, which then transform pixel data into
compact embeddings suitable for downstream multimodal processing. For instance, LLaVA [ 98] is a multimodal model
combining a CLIP-based vision encoder with the Vicuna LLM, fine-tuned on GPT-4-generated multimodal instruction-
following data. Additionally, MM1 [ 118] is a family of multimodal models trained using a carefully optimized mix
of image-caption, interleaved image-text, and text-only data. MM1’s model architecture includes the image encoder,
vision-language connector, and LLM.
Audio encoder converts raw audio signals into compact feature representations, capturing temporal and spectral
characteristics, then processes waveform or spectrogram inputs for downstream multimodal tasks. For example,
SALMONN [ 164], a MLLM designed to understand and process general auditory inputs, including speech, audio events,
and music. The model integrates a pre-trained text-based LLM with two audio encoders—Whisper for speech and BEATs
for non-speech audio—using a window-level query transformer (Q-Former) for alignment.
Multiple encoder system consists of separate encoders for different modalities (e.g., text, vision, audio) that process
their respective inputs independently before aligning or fusing them. This approach allows specialized feature extraction
per modality, improving performance in multimodal learning. For example, GLaMM [ 140] consists of an end-to-end
multimodal model architecture that includes a Global Image Encoder, Region Encoder, LLM, Grounding Image Encoder,
Manuscript submitted to ACM

14 Chen et al.
ArchitectureEncoder
Visual
Projector
Fusion
Module
Core LLM
Fusion
StrategyEarly Fusion
Late Fusion
Cross- Modal Attention Fusion
Hybrid Fusion
Modality
ApproachesImage
Audio
Video
Text-Rich
Image
TasksMultimodal
only
With other
applications(1) V ision Encoder , (2) Audio Encoder , (3)
Multiple Encoder System
(1) V isual Tokennizer ,  (2) Q-Former
Modality Alignment 
(1) Pre-trained,
(2) Fine-tuned,
(3) Frozen Large
Multimodal
Models
(1) CLIP , (2) V AE (3) V iT, (4) BLIP
(1) Whisper , (2) W avLM
(1) Flamingo (2) TimeSformer 
(1) OCR, (2) OCR Free
(1) Multimodal Reasoning, (2) Cross-Modal
Retrieval and Alignment, (3) Content
Generation, (4) Contextual Understanding
(1) With CoT  Prompting, (2) With RAG, (3)
With RL  Agents (4) Embodied AIDecoder (1) V ision Decoder , (2) Audio Decoder 
Fig. 3. Overview of Multimodal Large Language Models. The diagram categorizes MLLMs by architecture, fusion strategy, modality,
and tasks.
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 15
and Pixel Decoder. The Global Image Encoder extracts holistic image features, while the Region Encoder refines
region-specific details using a hierarchical feature pyramid. The Pixel Decoder is based on SAM (Segment Anything
Model) and generates segmentation masks from textual queries, enabling seamless grounded conversations.
5.1.2 Visual projector. Since different modalities have distinct feature spaces, a visual projector is employed to transform
encoded features into a shared representation space compatible with the language model. Typically implemented as
a linear projection, multi-layer perceptron (MLP), or cross-attention mechanism, the visual projector ensures that
non-textual features can be effectively processed alongside text embeddings. For example, HoneyBee [ 15] includes two
novel locality-enhanced projectors: C-Abstractor (using convolution) and D-Abstractor (using deformable attention),
designed to maintain both locality and flexibility. Another example is SEED [ 47], a discrete image tokenizer designed to
integrate vision into LLMs by enabling both image-to-text and text-to-image tasks. Additionally, BLIP-2 [ 82] introduces
a Querying Transformer (Q-Former) that bridges the modality gap between frozen pre-trained image encoders and
LLMs. Q-Former consists of a set of learnable query vectors that extract the most relevant visual features from the
frozen image encoder through cross-attention layers. These extracted features serve as a bottleneck representation,
ensuring that only the most informative visual information is passed to the LLM.
5.1.3 Fusion module. Fusion module in MLLMs is a mechanism designed to merge and process data from multiple
modalities (e.g., text and images) into a shared representation space. This allows the model to leverage complementary
information from different modalities. The goal of the fusion module is to combine and align features from different
modalities in a way that enables the model to understand and generate coherent outputs across these modalities. For
instance, ShareGPT4V [ 18], a framework designed to improve MLLMs by leveraging high-quality, detailed captions
for better modality alignment. The authors propose that detailed captions, rather than VQA data or brief descriptions,
significantly enhance the performance of LMMs. Furthermore, Fact-RLHF [161] addresses the problem of multimodal
misalignment in MLLMs by introducing a reinforcement learning-based approach that enhances the reward model
using factual information, such as image captions and ground-truth multi-choice options. This method prevents reward
hacking and improves multimodal alignment by providing more reliable training signals.
5.1.4 Core LLM. LLM is the backbone of the MLLMs, responsible for generating outputs based on the fused multimodal
representations. Typically built upon transformer-based architectures, the core LLM processes the integrated features
and produces coherent responses, leveraging its extensive pre-trained knowledge. This component ensures that the
model retains its language capabilities while incorporating multimodal understanding. In the domain of MLLMs, LLMs
can be pre-trained, fine-tuned, or frozen.
Pre-trained LLM of MLLMs is a model trained on a large-scale, diverse dataset using self-supervised or weakly
supervised learning. The goal is to develop general multimodal representations. For example, PaLM-E [ 35] injects
multimodal data—images, states, and textual descriptions—into an LLM using learned encoders that translate continuous
sensor inputs into language tokens. The model, based on the pre-trained PaLM-562B and Vision Transformer (ViT-22B),
is trained end-to-end on diverse datasets. Similarly, NExT-GPT [ 190] is an end-to-end general-purpose MLLM that
can handle any-to-any modality conversions, meaning it can process and generate outputs in text, image, video, and
audio. It integrates a pre-trained LLM (Vicuna-7B) with multimodal adaptors and diffusion decoders, enabling flexible
cross-modal understanding and generation.
Fine-tuned LLM in MLLMs is the pre-trained model which further trained on a specific dataset for a narrower,
task-specific purpose. The objective of fine-tuning is to optimize performance for a specific downstream task, such as
Manuscript submitted to ACM

16 Chen et al.
image captioning and medical report generation. For example, MultiModal-GPT [ 51] is a multimodal dialogue model
fine-tuned from OpenFlamingo and utilizes Low-rank Adapter (LoRA) layers in both the gated-cross-attention and
self-attention modules to enhance image-text interaction. Likewise, TimeChat [ 143], a large vision-language model
designed to process long videos while accurately localizing temporal events, where its model is fine-tuned on TimeIT,
a novel instruction-tuning dataset with 125K timestamped video instances, ensuring precise event localization and
efficient processing of lengthy video content.
In MLLMs, a Frozen LLM refers to an approach where the core language model (LLM) remains fixed (i.e., its weights
are not updated during training or fine-tuning). Instead, only the multimodal components (like the vision encoder,
visual projector, or fusion module) are trained. The primary motivation for leveraging frozen LLMs is computational
efficiency and efficient multimodal adoption. For instance, FROMAGe [ 74] integrates a frozen LLM (OPT-6.7B) and
a frozen visual encoder (CLIP ViT-L/14) by learning linear mappings between text and visual embedding spaces. It
introduces a special [RET] token to enhance image retrieval accuracy through contrastive learning, enabling the model
to generate text that is interleaved with retrieved images while keeping most parameters frozen for efficiency.
5.1.5 Decoder. Some MLLMs involve decoding when they generate images, videos, or multimodal content. These
model often follows an encoder-decoder paradigm, where encoders extract multimodal representations, and decoders
reconstruct images, videos, or speech. For example, Emu2 [ 158] is a 37-billion-parameter generative multimodal model
designed to enhance in-context learning for multimodal tasks; The model consists of a visual encoder, a multimodal
transformer, and a visual decoder. Additionally, ContextDET [ 213] uses a visual decoder to predict bounding boxes for
detected objects, enabling MLLMs to locate, identify, and associate visual objects with language inputs.
5.2 Fusion Strategy
MLLMs process and integrate multiple modalities, such as text, vision, and audio, to enhance understanding and
reasoning across diverse data sources. A fundamental challenge in MLLMs is modality fusion, which determines how
heterogeneous information is combined to form a unified representation. The choice of fusion strategy has a significant
impact on model performance, influencing its ability to capture cross-modal interactions, preserve modality-specific
semantics, and optimize computational efficiency. Broadly, fusion strategies can be categorized into four types: Early
Fusion, Late Fusion, Cross-modal Attention Fusion, and Hybrid Fusion.
5.2.1 Early fusion. Early fusion refers to the integration of multimodal features at the initial stages of the model
pipeline, typically before any independent unimodal processing occurs. In this approach, raw or low-level extracted
features from different modalities are concatenated or transformed into a shared representation space. This enables the
model to learn cross-modal interactions from the outset, resulting in a tightly coupled representation of multimodal
data. For example, the Gemini family of AI models [ 169] is trained on a large-scale multimodal dataset and leverages
post-training techniques, including supervised fine-tuning (SFT) and reinforcement learning from human feedback
(RLHF), to improve performance and alignment, which exhibits its early fusion strategy.
5.2.2 Late fusion. Late fusion preserves modality-specific feature extraction pathways before merging the outputs at
the final prediction or decision-making stage. This approach is advantageous when different modalities contribute com-
plementary information, but their interaction is not essential at an early stage of processing. For instance, Woodpecker
[205] is a training-free framework that consists of five stages: extracting key concepts from generated text, formulating
verification questions, validating visual knowledge using expert models, generating structured claims about the image,
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 17
and modifying hallucinated responses while incorporating bounding box evidence. The use of a late fusion strategy
ensures that the model’s generated text aligns with the image content.
5.2.3 Cross-modal Attention fusion. Cross-modal attention Fusion aims to establish direct interactions between different
modalities throughout the model’s architecture. Unlike early and late fusion, which either merge modalities at the
input or output stages, cross-modal fusion involves multiple interactions between modality-specific representations via
attention mechanisms, gating functions, or co-learning frameworks. This strategy is commonly employed in transformer-
based architectures where modalities attend to each other through cross-attention layers. Cross-modal fusion enables
deeper integration and alignment of multimodal information, improving contextual understanding and generalization in
tasks such as visual question answering and image captioning. For example, Zhang et al.(2024) [ 218] propose a method
called Multimodal-CoT that integrates both text and vision modalities into a two-stage framework to improve reasoning
performance. The model employed a cross-modal gated fusion strategy, in which a gate mechanism determines the
amount of information from vision to be injected into language representations. Another example is BLIVA [ 57],
which employs learned query embeddings extracted from a Q-Former, and the Q-Former uses cross-attention layers to
extract instruction-aware visual features from the frozen image encoder. These query embeddings act as a compressed
representation of the image and are passed as soft prompts to the LLM.
5.2.4 Hybrid fusion. Hybrid Fusion leverages a combination of early, late, and cross-modal attention fusion strategies
to optimize multimodal representation learning. By integrating multiple fusion techniques within the same model,
hybrid fusion seeks to balance the strengths and weaknesses of individual strategies. A model using hybrid fusion may
first employ early fusion to extract joint feature representations, apply cross-modal interactions for deeper contextual
alignment, and finally use late fusion for decision-level integration. This approach offers flexibility and robustness,
making it particularly useful for complex multimodal tasks that require both localized and holistic reasoning. For
example, KOSMOS-2 [ 127] employs hybrid fusion approach that combines early fusion with cross-attention mechanisms,
discretizes bounding box coordinates into location tokens, combine transformer and text and image features directly
into a unified Transformer model, enhance the model’s ability to perform phrase grounding, referring expression
comprehension, and multimodal referring. Another example of a hybrid fusion strategy is GPT4RoI [ 216], a region-
aware multimodal model that extends large vision-language models to understand fine-grained interactions within
region-of-interest (RoI) areas. It employed a hybrid fusion strategy, which involves early fusion with interleaved spatial
instruction encoding, RoIAlign, or deformable attention. This strategy then interleaves these RoI embeddings with text
embeddings, enabling the model to perform detailed region captioning, reasoning, and interaction beyond traditional
image-level vision-language models.
5.3 Modality Approaches
MLLMs are designed to process and generate outputs across multiple modalities, integrating information from images,
audio, video, and text-rich data. The effectiveness of these models depends on the underlying architectures and
representation learning strategies used for each modality. In this section, we explore key approaches for different
modalities, including image, audio, video, and text-rich image processing.
5.3.1 Image. Image-based MLLMs process visual data to extract semantic representations and generate textual descrip-
tions, answer questions, or perform visual reasoning. CLIP (Contrastive Language-Image Pre-training) [ 133] in MLLMs
serves as the visual encoder, transforming images into feature embeddings that align with text representations, which
Manuscript submitted to ACM

18 Chen et al.
are then processed by a core LLM to generate responses, perform reasoning, or enable downstream multimodal tasks.
It utilizes a dual-encoder architecture, where an image encoder (such as a Vision Transformer or ResNet) and a text
encoder (a Transformer) map images and text into a shared multimodal embedding space. The model is trained using
a contrastive learning objective: given a batch of (image, text) pairs, it maximizes the cosine similarity between the
embeddings of the correct pairs while minimizing the similarity for incorrect ones. This enables CLIP to generalize to
unseen tasks without explicit fine-tuning, allowing for zero-shot classification by comparing image embeddings with
text embeddings of class descriptions.
Variational Autoencoders (VAEs) [ 71] are a deep generative framework that learns latent variable representations
by combining probabilistic graphical models with deep learning. VAEs consist of an encoder (inference network) that
approximates the posterior distribution of latent variables given input data and a decoder (generative network) that
reconstructs data from these latent variables. In MLLMs, VAEs are utilized for learning a structured latent space, thereby
enhancing representation learning across modalities (e.g., text and images) by capturing complex dependencies and
generating coherent multimodal outputs.
The Vision Transformer (ViT) [ 34] is often employed as the visual encoder, extracting rich feature representations
from images that can be aligned with textual inputs, thereby enabling cross-modal reasoning and generation tasks. ViT
applies the Transformer architecture to image recognition by treating images as sequences of patches, rather than using
convolutional layers. By applying self-attention mechanisms directly to image patches, ViT can capture long-range
dependencies and enhance interpretability in visual tasks.
BLIP (Bootstrapping Language-Image Pre-training) [ 83] is a vision-language pre-training (VLP) framework designed
for both vision-language understanding and generation tasks. It introduces a multimodal mixture of encoder-decoder
(MED) architecture, which allows it to function flexibly as a unimodal encoder, an image-grounded text encoder, or
an image-grounded text decoder. In MLLMs, BLIP enhances cross-modal reasoning and text generation by effectively
aligning visual and textual representations, thereby facilitating tasks such as image captioning, visual question answering,
and image-text retrieval.
5.3.2 Audio. The integration of audio into MLLMs enables speech recognition, transcription, and even speech-to-text
or text-to-speech generation. WHISPER (Web-scale Supervised Pretraining for Speech Recognition) [ 134] is a large-scale
weakly supervised automatic speech recognition (ASR) model designed to generalize across multiple languages and
domains without fine-tuning. The model uses a sequence-to-sequence approach, where the encoder processes log-mel
spectrogram representations of speech, and the decoder generates transcriptions in a task-conditioned format, allowing
it to handle speech recognition, translation, and voice activity detection. In MLLMs, WHISPER serves as a speech-text
interface, enabling models to transcribe and understand spoken language inputs for downstream applications such as
audio-based retrieval, multimodal dialogue, and voice-assisted generation.
WavLM [ 19] is a self-supervised pre-trained model designed to learn universal speech representations for a wide range
of speech processing tasks, including speech recognition, speaker verification, speech separation, and diarization. It
employs a masked speech denoising and prediction framework that combines masked speech modeling with a denoising
mechanism, allowing the model to capture both content and speaker-related information. The Transformer-based
architecture incorporates gated relative position bias, enhancing the model’s ability to process temporal dependencies.
In MLLMs, WavLM serves as a robust speech encoder, providing high-quality speech representations that improve audio-
text alignment and multimodal reasoning, thereby enhancing the performance of tasks such as automatic transcription,
spoken language understanding, and cross-modal retrieval.
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 19
5.3.3 Video. Video extends multimodal learning by introducing both temporal and spatial dimensions, requiring
models to process sequential frames and synchronize with textual or auditory components. Flamingo [ 3] is a few-shot
visual language model (VLM) designed for open-ended vision-language tasks by integrating pre-trained vision and
language models with novel cross-modal architectural components. It employs a Perceiver Resampler to convert
high-dimensional visual features into a compact set of tokens, which are then processed by a frozen LLM augmented
with trainable Gated Cross-Attention Dense layers that facilitate multimodal reasoning. Within MLLMs, Models like
Flamingo handle video by encoding frames using a frozen vision model, such as NFNet, and aggregating them into a
fixed number of visual tokens via a Perceiver Resampler. These tokens are then fed into a frozen LLM with trainable
Gated Cross-Attention Dense layers, allowing the model to reason about video content in conjunction with textual
information. TimeSformer [ 10] is a convolution-free Transformer-based architecture for video understanding that
leverages self-attention mechanisms to model spatiotemporal dependencies. It extends the Vision Transformer (ViT)
by treating video as a sequence of frame-level patches, encoding these patches into token embeddings, and then
feeding them into a Transformer encoder. In MLLMs, TimeSformer serves as a video encoder, providing high-quality
spatiotemporal representations for downstream tasks such as video-text retrieval, video captioning, and video-based
question answering, thereby improving the model’s ability to reason over sequential visual information.
5.3.4 Text-rich Image. Text-rich images, such as scanned documents, charts, and infographics, necessitate models
that can jointly process visual layouts and embedded text. This modality is particularly useful in tasks like document
understanding, optical character recognition (OCR), and multimodal reasoning over structured data.
Optical Character Recognition (OCR) [ 120] has been applied extensively in MLLMs, OCR extracts text from images,
enabling MLLMs to process structured documents and handwritten text effectively. On the other hand, OCR-free methods
refer to recent advancements that explore direct modeling of text within images without explicit OCR extraction,
leveraging transformer-based architectures to capture both textual and visual features simultaneously. These methods
aim to improve robustness in noisy or low-resource environments. For example, Donut [ 70] is an OCR-free Visual
Document Understanding (VDU) model designed to directly map document images to structured outputs without
relying on Optical Character Recognition (OCR). It employs a Transformer-based architecture with a Swin Transformer
encoder for extracting visual features and a BART-style decoder that generates structured text sequences, such as JSON
representations, conditioned on the document image. In MLLMs, Donut serves as a document processing backbone,
enabling seamless integration of text and layout information from images into multimodal reasoning tasks without the
need for intermediate OCR-based pipelines.
5.4 Tasks
The capabilities of MLLMs extend beyond traditional unimodal processing, enabling them to tackle a diverse range
of tasks that integrate multiple modalities. These tasks can be broadly categorized into those that are intrinsically
multimodal and those that leverage multimodal capabilities alongside other applications.
5.4.1 Multimodal-Only Tasks. MLLMs can process and reason over multiple modalities without additional external
components, making them well-suited for core multimodal tasks.
Multimodal Reasoning refers to the ability to infer relationships between different modalities, such as answering
questions about an image or performing visual commonsense reasoning. For example, Socratic Models (SMs) [ 214],
a modular framework that composes multiple large pretrained models via multimodal-informed prompting, enables
zero-shot multimodal reasoning without requiring finetuning. The framework utilizes multi-step prompting chains,
Manuscript submitted to ACM

20 Chen et al.
where outputs from one model (e.g., CLIP for vision, GPT-3 for language, and Wav2CLIP for audio) are iteratively
processed and refined by other models, enabling joint inference on multimodal tasks without additional training.
Cross-Modal Retrieval and Alignment refers to Mapping information across different modalities, such as retrieving
relevant images based on textual queries or aligning text with corresponding visual elements. For instance, GILL
[73] a model that fuses frozen text-only LLMs with pre-trained image encoders and decoder models, allowing for
text generation, image retrieval, and novel image generation within multimodal dialogue. The model is trained using
image-caption pairs, with a decision module that determines whether to retrieve an image or generate a new one.
Content Generation refers to producing multimodal outputs, such as generating detailed textual descriptions of
images, creating multimodal narratives, or synthesizing visuals based on textual prompts. For example, Emu [ 159]
is a multimodal foundation model designed for seamless text and image generation within a unified autoregressive
framework. It integrates interleaved image, text, and video data, leveraging a causal transformer to transform visual
features into a sequence of latent embeddings, enabling versatile text-to-image generation tasks.
Contextual Understanding refers to comprehending and interpreting multimodal contexts to improve downstream
applications, such as understanding an image’s scene in relation to provided text. For example, ContextDET [ 213] is
a novel framework for contextual object detection that enhances MLLMs by enabling them to locate, identify, and
associate visual objects with language inputs. The model integrates language and vision tokens, using contextual LLM
outputs as prior knowledge to improve detection accuracy and flexibility.
5.4.2 Tasks with Other Applications. Beyond purely multimodal tasks, MLLMs are increasingly being integrated with
additional AI techniques to enhance their reasoning, retrieval, and interaction capabilities.
With Chain-of-Thought (CoT) Prompting enhancing reasoning by combining multimodal understanding with step-
by-step reasoning techniques. For instance, DDCoT [ 220] utilized and enhanced multimodal CoT prompting. DDCoT
decomposes complex questions into sub-questions, marking those requiring visual information as "uncertain" to prevent
hallucinations. A Visual Question Answering (VQA) model then provides relevant visual inputs, which are integrated
with textual reasoning to generate accurate answers. This structured CoT approach significantly enhances zero-shot
and fine-tuning performance.
With Retrieval-Augmented Generation (RAG) leveraging external knowledge retrieval to supplement multimodal
content generation and question answering. For example, RA-CM3 [ 204], a retrieval-augmented multimodal model
that retrieves relevant text and image data from an external memory and integrates it into a CM3 Transformer-based
generator. The retriever is a CLIP-based dense retriever, and the generator is trained with a novel retrieval-augmented
objective to effectively leverage retrieved documents.
With Reinforcement Learning (RL) Agents enabling interactive decision-making by integrating MLLMs with RL-based
agents for autonomous reasoning, planning, and action-taking in complex environments. For example, ESPER [ 210]
uses reinforcement learning to align multimodal inputs with language model generations. It employs CLIP to provide
a cosine similarity-based reward function for optimization and uses proximal policy optimization (PPO) to train a
lightweight vision-to-text transformation while keeping the language model parameters frozen.
Embodied AI embeds MLLMs into physical or virtual agents that perceive and act in the real world, enabling tasks
such as navigation, object manipulation, and human-AI interaction. For example, PaLM-E [ 35] integrates vision and
language with robotic sensor data to enable grounded reasoning and decision-making for embodied tasks. Similarly,
TaPA [ 193], a framework for embodied task planning, aligns LLMs with visual perception models to generate executable
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 21
Structure Mechanism Objective
Request
Central
ManagerComponent 1
Component 3 Component 4Component 2Input 
Handling
Task Planning
Model
Communication
Memory
ManagementTool Use
Output
GenerationFeedback Loop Component
2Component
1
Subtask Subtask Subtask Subtask
Cost EfﬁciencyPrivacy & Security
Bias
Low LatencyHierarchical
Fig. 4. An overview of the Compound AI System orchestration layer, highlighting the relationship between its structure (e.g.,
hierarchical or centralized component organization), mechanism (such as input handling, task planning, and feedback loops), and
objective (including privacy, low latency, and performance).
plans grounded in real-world physical constraints. It constructs a multimodal dataset and employs open-vocabulary
object detection to ensure that generated action sequences correspond to actual objects present in the scene.
6 Orchestration
The orchestration of CAIS is inherently sophisticated, often requiring complex architectures. LLM-serving systems
are designed to exceed the performance of standalone LLMs by integrating multiple layers, diverse models, or LLM-
supported agents that collaborate to deliver optimal results. We categorize LLM-serving systems into three distinct
layers: the structural layer, the mechanism layer, and the objective layer, as illustrated in Figure 4.
6.1 Structure Layer
The structure layer represents the architectural organization of components within a system, focusing on how tasks are
distributed and coordinated. It includes two primary designs: Hierarchical Structure and Central Structure. This layer
provides the foundational framework for the system’s operational and interaction dynamics.
6.1.1 Hierarchical Structure. In a hierarchical structure, tasks are decomposed into subtasks, which are managed by
components organized in a tree-like hierarchy with clear dependencies. Typically, these systems comprise multiple
layers, with each layer breaking down inputs or requests into smaller, more manageable portions. Components in a
hierarchical structure are modular, meaning they are distinct, self-contained, and interact with others in a coordinated
manner. An example of hierarchical structure, MemGPT [ 124], introduces a hierarchical memory system inspired by
operating systems to address the limitations of LLMs’ fixed context windows. The system comprises a main context
Manuscript submitted to ACM

22 Chen et al.
(prompt tokens within the LLM), external memory (for recall and archival storage), and components such as the queue
manager and function executor to handle data flow and function chaining. This design ensures seamless integration of
dynamic memory management with existing LLM capabilities. Furthermore, Acharya et al. [ 1] proposed a LLM-based
recommendation system, which combines an LLM-based description generator with a GRU-based recommendation
model. Descriptions generated by the LLM are embedded using BERT, combined with item ID embeddings, and processed
through a GRU layer to predict recommendations. The hierarchical structure of LLM-serving systems has also been
widely applied across various domains, including data exploration systems [ 109], multi-agent operating systems [ 119],
and recommendation systems [181].
6.1.2 Central Structure. In a central structure, a central manager oversees and coordinates interactions among com-
ponents to ensure efficient collaboration and coordination. The central manager acts as a scheduler or coordinator,
dispatching and allocating resources to the most suitable components or tools based on the nature of the task. For
example, PagedAttention [ 77] utilizes a centralized scheduler and distributed GPU workers. The KV cache manager
dynamically allocates and maps non-contiguous memory blocks, facilitating scalable and efficient LLM serving while
integrating with various decoding strategies. Moreover, BOLAA [ 103] organizes the agents in a modular structure, with
the controller acting as a central node to route tasks among specialized agents and manage their interactions with
the environment. This topology allows scalable and efficient collaboration between agents. Similar to the hierarchical
structure, the central structure has been effectively applied to diverse domains, including the Internet of Things (IoT)
[27] and multimodal systems [130].
6.2 Mechanism Layer
The mechanism layer defines the operational processes that govern how a system handles tasks and produces results.
This layer ensures seamless execution and adaptability within the system. Besides input handling and output generation,
it includes Task Planning , which determines the steps of tasks required to achieve the desired outcome. For example,
LLM-MARS [ 107], a Multi-Agent Robot Systems, supports dynamic task allocation, environmental feedback, and
real-time question-answering using LoRa adapters. Another example is Infinite-LLM [ 88], where its system integrates a
distributed attention mechanism (DistAttention) with a centralized scheduler (gManager) and instance-level resource
managers (rManagers). This design decouples attention computation from the main inference pipeline, enabling flexible
task allocation across GPUs in a cluster.
6.2.1 Model Communication. Model Communication is a mechanism that enables different AI models or multiple agents
or components within a system to communicate and exchange information. For example, de Zarzà et al. [ 29] proposed
a system that integrates a DNN-based adaptive PID controller for optimal gain prediction and a GPT-3.5-turbo LLM
for real-time feedback. Moreover, TransLLaMa [ 75], a simultaneous machine translation (SiMT) system, integrates an
automatic speech recognition (ASR) model with a fine-tuned LLM. Tan et al. [ 163] proposed a system that involves
multiple expert agents for specific tasks, each working with distinct abstraction spaces. Solutions are generated by
combining primitive functions, structured prompting, and iterative feedback loops to refine outputs.
6.2.2 Tool Use. Tool Use . A key mechanism in CAIS is enabling LLMs to leverage external tools (e.g., APIs, search
engines, databases, code executors) [ 132,151,183]. This overcomes limitations such as static knowledge [ 187] and a
lack of computational precision, allowing LLMs to interact with real-time data and perform specialized tasks, essentially
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 23
acting as controllers for broader computational resources. For instance, agents like OSAgent use standardized APIs to
interact with operating systems [195].
Tool use in LLM agents is generally enabled through two main approaches: training-based and prompt-based
methods. Training-based methods involve fine-tuning models to integrate tool use capabilities. Examples include GPT-4’s
structured function calling mechanism [ 2] and Toolformer’s self-supervised approach, which teaches the model to
invoke tools such as search engines or translation services [ 147]. In contrast, prompt-based methods rely on strategic
prompt design to guide LLM behavior. For instance, the ReAct framework interleaves reasoning with actions like tool
invocation or web search via prompting, enabling complex task decomposition without additional fine-tuning [203].
A specific application is enhancing LLMs with Web Search . This addresses knowledge staleness and improves factuality.
Models can be fine-tuned for web interaction, like WebGPT using a text-based browser to search, navigate, and cite
sources [ 123], or GopherCite using search to find supporting evidence for its claims [ 121]. Prompting techniques, such
as ReAct, can also guide LLMs to issue search queries during their reasoning process [ 203]. Orchestration frameworks,
such as LangChain and AutoGPT, further help manage complex tool interactions and chaining [189].
6.2.3 Memory Management. Memory Management is a function or algorithm that is designed to efficiently allocate,
manage, and optimize the use of memory resources during tasks. For example, PagedAttention [ 77] implements
a memory management system inspired by paging in operating systems, reducing memory waste and improving
throughput by dynamically allocating non-contiguous blocks of memory for key-value caches. Similarly, JungleGPT
[144] utilizes caching nodes for low-latency global data access, serving as distributed snapshots to facilitate cost-efficient
and scalable operations.
6.2.4 Feedback Loop. Feedback Loop refers to a system mechanism where outputs or results are cyclically returned as
inputs to refine or optimize the process. This iterative approach enables continuous improvement and adaptation by
addressing errors and inefficiencies found in previous iterations. For example, in Text-to-SQL systems [ 114], feedback
loops refine SQL query generation based on execution results, improving success rates with each iteration.
6.3 Objective Layer
The objective layer outlines the primary objectives that guide the design and operation of Compound AI System
architectures. This layer focuses on essential priorities, including maintaining privacy and security, reducing bias,
ensuring low response times, and improving cost efficiency. These objectives collectively establish a clear framework to
guarantee that the system remains effective, reliable, and aligned with user requirements.
Privacy and Security in CAIS involve mechanisms to mitigate risks associated with sensitive data exposure and
system vulnerabilities. For instance, architectures like SECGPT [ 191] utilize isolated execution environments and
hierarchical hub-and-spoke models to manage interactions securely, protecting against risks from untrustworthy third-
party apps and imprecise natural language interfaces. These designs ensure controlled data exchange, user permission
management, and encapsulated app environments to safeguard privacy and security. Additionally, Evertz et al. [ 40]
investigate confidentiality vulnerabilities in systems that integrate LLMs with external tools. It proposes a framework
that simulates specific attack scenarios to evaluate and measure risks related to sensitive data leakage when LLMs
interact with integrated tools and services.
Bias in CAIS refers to the tendencies of models to produce outputs influenced by underlying patterns or training
data that may reflect stereotypes, inaccuracies, or a lack of diversity. Sharma et al. [ 150] investigate the effects of
LLM-powered conversational search systems on selective exposure and opinion polarization. It employs studies that
Manuscript submitted to ACM

24 Chen et al.
compare LLM-supported systems to traditional web search, analyzing biases and user attitudes influenced by LLM
opinion alignment.
Low Latency measures the ability to process and respond to requests with minimal delay, ensuring efficient and
timely user interactions. For example, LLM-Slice [ 95] reduces average latency by leveraging dedicated wireless network
slices for efficient resource allocation, achieving significant reductions in transmission times for LLM tasks.
Cost-efficiency in LLM serving systems refers to the ability to optimize resource usage to achieve desired outcomes,
such as performance and quality, while minimizing costs. For instance, PALIMPZEST [ 96] is a declarative system that
optimizes AI workloads by enabling engineers to define tasks at a high level of abstraction. The system compiles these
tasks into optimized execution plans, balancing cost, runtime, and quality through logical and physical optimizations.
7 Benchmarks and Evaluation Metrics
The rapid growth and advancement of CAIS has led to the development of numerous evaluation benchmarks and
datasets. Because CAIS can have different objectives and structures, their respective evaluation benchmarks are diverse.
This chapter introduces the key tasks, datasets, and quantitative metrics used for evaluating CAIS across the domains
of RAG, Multimodal LLMs, LLM Agents, and Orchestration, as detailed in Table 1.
7.1 Evaluation of RAG
To systematically evaluate RAG systems, various standardized datasets and benchmarks are utilized, tailored to assess
both retrieval and generation components across diverse tasks, including question answering, retrieval, summarization,
and fact verification. The goal is typically to measure improvements in areas like retrieval quality (using metrics such as
MRR and nDCG) and generation quality/factuality (using metrics like EM, F1, ROUGE, BLEU, and Accuracy) compared
to standalone LLMs. Table 1 summarizes the primary downstream tasks, associated datasets, and quantitative metrics
used for RAG evaluation.
7.2 Evaluation of LLM Agents
Evaluating LLM agents, the decision-making cores of many CAIS, requires assessing their agentic behaviors, such
as role simulation, interactive reasoning, and tool use, in dynamic environments. Specialized benchmarks assess role
fidelity (Role-Consistency Score), reasoning quality (Reasoning Trace Accuracy), and tool-use efficiency (Tool Call
Accuracy, Token Efficiency). Specific benchmarks and metrics for these aspects are listed in Table 1.
7.3 Evaluation of Multimodal LLMs
With MLLMs, evaluation expands to assess the integration and reasoning over multiple input types, such as images and
text. Key evaluation areas include multimodal reasoning (e.g., visual QA), chart and document understanding, safety
and alignment, and general multimodal capabilities. Standard NLP and vision metrics (Accuracy, F1, EM, BLEU, and
ROUGE) are often used alongside custom safety metrics and specialized benchmarks, as detailed in Table 1.
7.4 Evaluation of Orchestration of Compound AI Systems
Evaluating the orchestration of CAIS, which integrates multiple components like retrievers and agents, involves
assessing efficiency, reliability, and fairness beyond task accuracy. Evaluation often considers the Structure Layer
(infrastructure performance: throughput, Latency), the Mechanism Layer (system coordination: Task Completion Time,
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 25
Table 1. Summary of evaluation dimensions, benchmarks, and metrics for Compound AI Systems.
Dimension Task Datasets/Benchmarks Evaluation Metrics
RAGReasoning QA Natural Questions [ 76], TriviaQA [ 65],
HotpotQA [201], WebQuestions [9]Accuracy, F1 Score, EM
Passage RetrievalWikiAsp [54], MS MARCO [7],
SQUAD [137], TruthfulQA [91]MRR, nDCG, Precision, Recall, F1
Score
Multi-Doc Summa-
rizationOpenBookQA [122], PopQA [115] ROUGE, BLEU, F1 Score
Open-domain QAMulti-News [41], NarrativeQA [72],
MuSiQue [176], BEIR [172],
RealTimeQA [67], UniEval [221]EM, MRR, nDCG, F1 Score, Accu-
racy
Extractive QA RAGAs [39], KILT [129] EM, F1 Score, Precision, Recall
Fact Verification FEVER [ 173], DROP [ 37], TREC-DL [ 25]Accuracy, Precision, Recall, F1
Score
Evaluation Metrics 2WikiMultihopQA [ 55], StrategyQA
[48]EM, F1 Score, MRR, nDCG
Multimodal LLMReasoning and
Commonsense QAMM-Vet [ 208], MMC-Benchmark [ 97],
ALLaVA [17], SEED-Bench [80]Accuracy, F1 Score, EM, Perplexity
Chart and Docu-
ment Understand-
ingChartQA [ 116], DocVQA [ 117],
TextVQA [154]Accuracy, F1 Score, BLEU, ROUGE
Safety and Align-
mentMM-SafetyBench [102] Accuracy, F1 Score, Custom Safety
Metrics
Multimodal Evalua-
tionMME [87], MultiBench [85] Accuracy, F1 Score, MRR, Perplex-
ity, BLEU, ROUGE
Orchestration of
Compound AI
SystemsStructure LayerBigDataBench [45],
AIBench Training [165]Throughput, Latency Overhead,
Bandwidth Utilization
Mechanism Layer AISBench [ 33], WebArena [ 222], Long
Range Arena [ 167], ZeroSCROLLS [ 148]Mean Task Completion Time, Aver-
age API Response Time, Cache Hit
Rate
Objective Layer AI Fairness 360 [8] Fairness-specific metrics
LLM AgentRole-Playing RoleLLM [ 184], AgentBench [ 101] ,
AgentBoard [108]Role-Consistency Score, Self-
Correction Rate
Interactive Reason-
ingAgentQuest [ 49], InfiAgent-DABench
[58], CriticBench [94]Reasoning Trace Accuracy, Gener-
alization Score
Tool Use ML-Bench [ 166], Berkeley Function
Calling Leaderboard [198]Tool Call Accuracy, Token Effi-
ciency
Manuscript submitted to ACM

26 Chen et al.
API Response Time, Cache Hit Rate), and the Objective Layer (alignment with goals: fairness metrics). Representative
benchmarks for each layer are provided in Table 1.
These benchmarks and datasets collectively form a framework for evaluating modern AI systems, spanning RAG,
Multimodal LLMs, LLM Agents, and Orchestration, capturing task performance alongside efficiency, fairness, and
orchestration quality, as summarized in Table 1.
8 Challenges, Limitations, and Opportunities
8.1 Challenges and Limitations
Compound AI Systems represent a significant advancement in the capabilities of LLM-based architectures, yet they are
not without challenges and limitations. As this field matures, it is crucial to understand the key obstacles that hinder its
scalability, efficiency, and safety, while also identifying the most promising future directions.
8.1.1 System Complexity and Scalability. Compound AI Systems often involve the integration of multiple compo-
nents—retrievers, agents, multimodal encoders, memory modules, and orchestration mechanisms. This architectural
complexity introduces engineering overhead, increasing the difficulty of deployment, debugging, and system optimiza-
tion. Moreover, the added complexity can lead to performance bottlenecks and increased inference latency, particularly
when coordinating multiple LLMs and tools in real-time.
8.1.2 Evaluation and Benchmarking. Evaluating Compound AI Systems is inherently difficult due to the diversity of
their components and the dynamic nature of their outputs. Traditional NLP benchmarks do not adequately capture the
interactive or multi-modal capabilities of such systems. There is a lack of unified evaluation frameworks that consider
system-level performance metrics, such as latency, robustness, and resource usage, alongside task-specific accuracy.
8.1.3 Tool and Component Integration. Despite advances in tool-use and agent-based planning, seamless integration
between LLMs and external components remains a challenge. Issues such as API misalignment, inconsistent data
formats, limited error handling, and brittle tool chaining cause failures in real-world applications. Moreover, many tools
and retrievers were not originally designed for interaction with LLMs, leading to interoperability and reliability issues.
8.2 Opportunities
Future opportunities for Compound AI Systems lie in developing unified and modular architectures that simplify
integration across components, advancing multimodal alignment through improved fusion strategies and grounded
reasoning, and enabling end-to-end trainable pipelines for tighter coordination among retrievers, agents, and generators.
Emerging directions also include meta-agent systems with self-adaptive orchestration, enhanced human-AI collab-
oration via interpretable interfaces and controllable planning, and sustainable design practices focused on resource
efficiency and deployment scalability. These advances will pave the way for more robust, flexible, and intelligent
Compound AI Systems capable of operating in dynamic, real-world environments.
9 Conclusion
The survey highlights Compound AI Systems as a significant advancement in the expansion of LLM capabilities by
integrating them with external components such as tools, agents, retrievers, and multimodal encoders. This survey
captures the emergence of Compound AI as a paradigm that systematically enhances performance, adaptability, and
context-awareness across a wide range of applications. We identify four core dimensions—RAG, LLM Agents, MLLMs,
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 27
and Orchestration—that structure the landscape of Compound AI Systems, each contributing to a more intelligent
and flexible AI ecosystem. Technical integrations with methodologies like planning, memory management, and self-
reflection have further broadened their capabilities. While Compound AI Systems show promise in handling complex and
multimodal tasks, challenges remain in orchestrating components effectively and ensuring system-level robustness. The
scope of Compound AI continues to expand into real-world, multimodal, and interactive environments, underscoring
its growing impact on both research and industry. This expanding ecosystem requires improved benchmarking and
evaluation strategies to more effectively capture system-level performance. As Compound AI Systems continue to evolve,
refining these methods will be key to fully realizing their potential in the next generation of intelligent applications.
References
[1]Arkadeep Acharya, Brijraj Singh, and Naoyuki Onoe. 2023. LLM Based Generation of Item-Description for Recommendation System. In Proceedings
of the 17th ACM Conference on Recommender Systems (Singapore, Singapore) (RecSys ’23) . Association for Computing Machinery, New York, NY,
USA, 1204–1207. https://doi.org/10.1145/3604915.3610647
[2]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam
Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).
[3]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm
Reynolds, et al .2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35 (2022),
23716–23736.
[4] Anthropic. 2024. Model Context Protocol. https://www.anthropic.com/news/model-context-protocol. Accessed: 2025-03-29.
[5]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through
self-reflection. arXiv preprint arXiv:2310.11511 (2023).
[6]Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-augmented language model prompting for zero-shot knowledge graph question
answering. arXiv preprint arXiv:2306.04136 (2023).
[7]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri
Nguyen, et al. 2016. Ms marco: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268 (2016).
[8]Rachel KE Bellamy, Kuntal Dey, Michael Hind, Samuel C Hoffman, Stephanie Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep
Mehta, Aleksandra Mojsilović, et al .2019. AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. IBM Journal of
Research and Development 63, 4/5 (2019), 4–1.
[9]Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on freebase from question-answer pairs. In Proceedings of the
2013 conference on empirical methods in natural language processing . 1533–1544.
[10] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. 2021. Is space-time attention all you need for video understanding?. In ICML , Vol. 2. 4.
[11] Multimodal Blog. 2024. How to Chunk Documents for Retrieval-Augmented Generation (RAG) . https://www.multimodal.dev/post/how-to-chunk-
documents-for-rag?utm_source=chatgpt.com Accessed: 2024-12-14.
[12] Bloomberg Intelligence. 2023. Generative AI to Become a $1.3 Trillion Market by 2032, Research Finds. https://www.bloomberg.com/company/
press/generative-ai-to-become-a-1-3-trillion-market-by-2032-research-finds/. Accessed: 2025-05-30.
[13] Daniil A Boiko, Robert MacKnight, and Gabe Gomes. 2023. Emergent autonomous scientific research capabilities of large language models. arXiv
preprint arXiv:2304.05332 (2023).
[14] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
[15] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. 2024. Honeybee: Locality-enhanced projector for multimodal llm. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 13817–13827.
[16] Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2023. Chateval: Towards better
llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201 (2023).
[17] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou
Wang. 2024. Allava: Harnessing gpt4v-synthesized data for lite vision-language models. arXiv preprint arXiv:2402.11684 (2024).
[18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2025. Sharegpt4v: Improving large multi-modal
models with better captions. In European Conference on Computer Vision . Springer, 370–387.
[19] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al .
2022. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing 16, 6
(2022), 1505–1518.
[20] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. 2022. Re-imagen: Retrieval-augmented text-to-image generator. arXiv preprint
arXiv:2209.14491 (2022).
Manuscript submitted to ACM

28 Chen et al.
[21] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al .2023.
Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv preprint arXiv:2308.10848 2, 4 (2023), 6.
[22] Xiang Chen, Lei Li, Ningyu Zhang, Xiaozhuan Liang, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. 2022. Decoupling
knowledge from memorization: Retrieval-augmented prompt learning. Advances in Neural Information Processing Systems 35 (2022), 23908–23922.
[23] Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. 2024. Lift yourself up: Retrieval-augmented text generation with
self-memory. Advances in Neural Information Processing Systems 36 (2024).
[24] Wikipedia contributors. 2024. Claude (language model). https://en.wikipedia.org/wiki/Claude_(language_model). Accessed: 2024-12-24.
[25] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Ellen M Voorhees, and Ian Soboroff. 2021. TREC deep learning track: Reusable
test collections in the large data regime. In Proceedings of the 44th international ACM SIGIR conference on research and development in information
retrieval . 2369–2375.
[26] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio
Silvestri. 2024. The power of noise: Redefining retrieval for rag systems. In Proceedings of the 47th International ACM SIGIR Conference on Research
and Development in Information Retrieval . 719–729.
[27] Hongwei Cui, Yuyang Du, Qun Yang, Yulin Shao, and Soung Chang Liew. 2024. Llmind: Orchestrating ai and iot with llm for complex task
execution. IEEE Communications Magazine (2024).
[28] Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B Hall, and Ming-Wei Chang. 2022. Promptagator:
Few-shot dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755 (2022).
[29] Irene de Zarzà, Joachim de Curtò, Gemma Roig, and Carlos T Calafate. 2023. Llm adaptive pid control for b5g truck platooning systems. Sensors 23,
13 (2023), 5899.
[30] deepset.ai Team. 2024. Haystack GitHub Repository. https://github.com/deepset-ai/haystack. Accessed: 2024-12-24.
[31] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for
the web. Advances in Neural Information Processing Systems 36 (2023), 28091–28114.
[32] DataStax Documentation. 2024. Introduction to Indexing in Retrieval-Augmented Generation (RAG) . https://docs.datastax.com/en/ragstack/intro-to-
rag/indexing.html?utm_source=chatgpt.com Accessed: 2024-12-14.
[33] Jian Dong, Wei Bao, Xiaoqi Cao, Yang Xu, Yuze Yang, Binbin Li, Qi Zhang, and Heng Ye. 2025. AISBench: an performance benchmark for AI server
systems. The Journal of Supercomputing 81, 2 (2025), 1–24.
[34] Alexey Dosovitskiy. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).
[35] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong,
Tianhe Yu, et al. 2023. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378 (2023).
[36] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models
through multiagent debate. In Forty-first International Conference on Machine Learning .
[37] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension
benchmark requiring discrete reasoning over paragraphs. arXiv preprint arXiv:1903.00161 (2019).
[38] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global:
A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024).
[39] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In
Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations . 150–158.
[40] Jonathan Evertz, Merlin Chlosta, Lea Schönherr, and Thorsten Eisenhofer. 2024. Whispers in the Machine: Confidentiality in LLM-integrated
Systems. arXiv preprint arXiv:2402.06922 (2024).
[41] Alexander R Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R Radev. 2019. Multi-news: A large-scale multi-document summarization dataset
and abstractive hierarchical model. arXiv preprint arXiv:1906.01749 (2019).
[42] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima
Anandkumar. 2022. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing
Systems 35 (2022), 18343–18362.
[43] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms:
Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining .
6491–6501.
[44] Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. 2025. From LLM Reasoning to Autonomous AI Agents: A Comprehensive
Review. arXiv preprint arXiv:2504.19678 (2025).
[45] Wanling Gao, Jianfeng Zhan, Lei Wang, Chunjie Luo, Daoyi Zheng, Xu Wen, Rui Ren, Chen Zheng, Xiwen He, Hainan Ye, et al .2018. Bigdatabench:
A scalable and unified big data and ai benchmark suite. arXiv preprint arXiv:1802.08254 (2018).
[46] Manas Gaur, Kalpa Gunaratna, Vijay Srinivasan, and Hongxia Jin. 2022. Iseeq: Information seeking question generation using dynamic meta-
information retrieval and knowledge graphs. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 36. 10672–10680.
[47] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. 2023. Planting a seed of vision in large language model. arXiv preprint
arXiv:2307.08041 (2023).
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 29
[48] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering
benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics 9 (2021), 346–361.
[49] Luca Gioacchini, Giuseppe Siracusano, Davide Sanvito, Kiril Gashteovski, David Friede, Roberto Bifulco, and Carolin Lawrence. 2024. Agentquest:
A modular benchmark framework to measure progress and improve llm agents. arXiv preprint arXiv:2404.06411 (2024).
[50] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Rajaram Naik, Pengshan Cai, and Alfio Gliozzo. 2022. Re2G: Retrieve,
rerank, generate. arXiv preprint arXiv:2207.06300 (2022).
[51] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. 2023.
Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790 (2023).
[52] Anirudh Goyal, Abram Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke, Adria Puigdomenech Badia, Arthur Guez, Mehdi Mirza,
Peter C Humphreys, Ksenia Konyushova, et al .2022. Retrieval-augmented reinforcement learning. In International Conference on Machine Learning .
PMLR, 7740–7765.
[53] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language
model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680 (2024).
[54] Hiroaki Hayashi, Prashant Budania, Peng Wang, Chris Ackerson, Raj Neervannan, and Graham Neubig. 2021. Wikiasp: A dataset for multi-domain
aspect-based summarization. Transactions of the Association for Computational Linguistics 9 (2021), 211–225.
[55] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation
of reasoning steps. arXiv preprint arXiv:2011.01060 (2020).
[56] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou,
et al. 2023. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352 3, 4 (2023), 6.
[57] Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. 2024. Bliva: A simple multimodal llm for better handling of text-rich visual
questions. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38. 2256–2264.
[58] Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, et al .2024. Infiagent-dabench:
Evaluating agents on data analysis tasks. arXiv preprint arXiv:2401.05507 (2024).
[59] Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A. Ross, and Alireza Fathi. 2023. Reveal:
Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory. In 2023 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) . 23369–23379. https://doi.org/10.1109/CVPR52729.2023.02238
[60] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin,
et al.2023. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems (2023).
[61] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al .
2022. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608 (2022).
[62] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020).
[63] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research 24, 251 (2023),
1–43.
[64] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active
retrieval augmented generation. arXiv preprint arXiv:2305.06983 (2023).
[65] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading
comprehension. arXiv preprint arXiv:1705.03551 (2017).
[66] Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, and Sung Ju Hwang. 2024. Knowledge-augmented reasoning distillation for small
language models in knowledge-intensive tasks. Advances in Neural Information Processing Systems 36 (2024).
[67] Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A Smith, Yejin Choi, Kentaro Inui, et al .2023.
Realtime qa: What’s the answer right now? Advances in neural information processing systems 36 (2023), 49025–49043.
[68] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-search-predict:
Composing retrieval and language models for knowledge-intensive nlp. arXiv preprint arXiv:2212.14024 (2022).
[69] Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. Advances in Neural Information Processing
Systems 36 (2023), 39648–39677.
[70] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and
Seunghyun Park. 2022. Ocr-free document understanding transformer. In European Conference on Computer Vision . Springer, 498–517.
[71] Diederik P Kingma, Max Welling, et al .2019. An introduction to variational autoencoders. Foundations and Trends ®in Machine Learning 12, 4
(2019), 307–392.
[72] Tomáš Kočisk `y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The narrativeqa
reading comprehension challenge. Transactions of the Association for Computational Linguistics 6 (2018), 317–328.
[73] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. 2024. Generating images with multimodal language models. Advances in Neural Information
Processing Systems 36 (2024).
Manuscript submitted to ACM

30 Chen et al.
[74] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. 2023. Grounding language models to images for multimodal inputs and outputs. In International
Conference on Machine Learning . PMLR, 17283–17300.
[75] Roman Koshkin, Katsuhito Sudoh, and Satoshi Nakamura. 2024. Transllama: Llm-based simultaneous translation system. arXiv preprint
arXiv:2402.04636 (2024).
[76] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob
Devlin, Kenton Lee, et al .2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational
Linguistics 7 (2019), 453–466.
[77] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles .
611–626.
[78] Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022. Internet-augmented language models through few-shot
prompting for open-domain question answering. arXiv preprint arXiv:2203.05115 (2022).
[79] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K ¨"uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨"aschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33
(2020), 9459–9474.
[80] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: Benchmarking multimodal llms with generative
comprehension. arXiv preprint arXiv:2307.16125 (2023).
[81] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for" mind" exploration of
large language model society. Advances in Neural Information Processing Systems 36 (2023), 51991–52008.
[82] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and
large language models. In International conference on machine learning . PMLR, 19730–19742.
[83] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language
understanding and generation. In International conference on machine learning . PMLR, 12888–12900.
[84] Xinzhe Li. 2025. A Review of Prominent Paradigms for LLM-Based Agents: Tool Use, Planning (Including RAG), and Feedback Learning. In
Proceedings of the 31st International Conference on Computational Linguistics . 9760–9779.
[85] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A Lee, Yuke Zhu, et al .2021. Multibench:
Multiscale benchmarks for multimodal representation learning. Advances in neural information processing systems 2021, DB1 (2021), 1.
[86] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2023. Encouraging divergent
thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118 (2023).
[87] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. 2024. A Survey of Multimodel Large Language Models. In
Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering . 405–409.
[88] Bin Lin, Chen Zhang, Tao Peng, Hanyu Zhao, Wencong Xiao, Minmin Sun, Anmin Liu, Zhipeng Zhang, Lanbo Li, Xiafei Qiu, et al .2024. Infinite-llm:
Efficient llm service for long context with distattention and distributed kvcache. arXiv preprint arXiv:2401.02669 (2024).
[89] Bill Yuchen Lin, Kangmin Tan, Chris Miller, Beiwen Tian, and Xiang Ren. 2022. Unsupervised cross-task generalization via retrieval augmentation.
Advances in Neural Information Processing Systems 35 (2022), 22003–22017.
[90] Matthieu Lin, Jenny Sheng, Andrew Zhao, Shenzhi Wang, Yang Yue, Yiran Wu, Huan Liu, Jun Liu, Gao Huang, and Yong-Jin Liu. 2024. LLM-based
Optimization of Compound AI Systems: A Survey. arXiv preprint arXiv:2410.16392 (2024).
[91] Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958
(2021).
[92] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. How to train your dragon:
Diverse augmentation towards generalizable dense retrieval. arXiv preprint arXiv:2302.07452 (2023).
[93] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al .
2023. Ra-dit: Retrieval-augmented dual instruction tuning. arXiv preprint arXiv:2310.01352 (2023).
[94] Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. 2024. Criticbench: Benchmarking llms for critique-correct reasoning.
arXiv preprint arXiv:2402.14809 (2024).
[95] Boyi Liu, Jingwen Tong, and Jun Zhang. 2024. Llm-slice: Dedicated wireless network slicing for large language models. In Proceedings of the 22nd
ACM Conference on Embedded Networked Sensor Systems . 853–854.
[96] Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baille Chen, Zui Chen, Michael Franklin, Tim Kraska, Samuel Madden, and Gerardo
Vitagliano. 2024. A Declarative System for Optimizing AI Workloads. arXiv preprint arXiv:2405.14696 (2024).
[97] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. 2023. Mmc: Advancing
multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774 (2023).
[98] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems 36
(2024).
[99] Junyi Liu, Liangzhi Li, Tong Xiang, Bowen Wang, and Yiming Qian. 2023. Tcra-llm: Token compression retrieval augmented large language model
for inference cost reduction. arXiv preprint arXiv:2310.15556 (2023).
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 31
[100] Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. WebGLM: Towards an efficient
web-enhanced question answering system with human preferences. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining . 4549–4560.
[101] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al .2023. Agentbench:
Evaluating llms as agents. arXiv preprint arXiv:2308.03688 (2023).
[102] Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024. Mm-safetybench: A benchmark for safety evaluation of multimodal
large language models. In European Conference on Computer Vision . Springer, 386–403.
[103] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit,
et al. 2023. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960 (2023).
[104] Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, and Anton van den Hengel.
2022. Retrieval Augmented Classification for Long-Tail Visual Recognition. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) . 6949–6959. https://doi.org/10.1109/CVPR52688.2022.00683
[105] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svyatkovskiy. 2022. Reacc: A retrieval-augmented code completion
framework. arXiv preprint arXiv:2203.07722 (2022).
[106] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. 2023. Sail:
Search-augmented instruction learning. arXiv preprint arXiv:2305.15225 (2023).
[107] Artem Lykov, Maria Dronova, Nikolay Naglov, Mikhail Litvinov, Sergei Satsevich, Artem Bazhenov, Vladimir Berman, Aleksei Shcherbak, and
Dzmitry Tsetserukou. 2023. Llm-mars: Large language model for behavior tree generation and nlp-enhanced dialogue in multi-agent robot systems.
arXiv preprint arXiv:2312.09348 (2023).
[108] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An
analytical evaluation board of multi-turn llm agents. arXiv preprint arXiv:2401.13178 (2024).
[109] Pingchuan Ma, Rui Ding, Shuai Wang, Shi Han, and Dongmei Zhang. 2023. InsightPilot: An LLM-empowered automated data exploration system.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations . 346–352.
[110] Ruotian Ma, Xiaolei Wang, Xin Zhou, Jian Li, Nan Du, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024. Are large language models good prompt
optimizers? arXiv preprint arXiv:2402.02101 (2024).
[111] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query rewriting for retrieval-augmented large language models. arXiv
preprint arXiv:2305.14283 (2023).
[112] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-tuning llama for multi-stage text retrieval. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information Retrieval . 2421–2425.
[113] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar.
2023. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931 (2023).
[114] Karime Maamari and Amine Mhedhbi. 2024. End-to-end Text-to-SQL Generation within an Analytics Insight Engine. arXiv preprint arXiv:2406.12104
(2024).
[115] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. When not to trust language models:
Investigating effectiveness of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511 (2022).
[116] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts
with visual and logical reasoning. arXiv preprint arXiv:2203.10244 (2022).
[117] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF
winter conference on applications of computer vision . 2200–2209.
[118] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Anton
Belyi, et al .2024. MM1: methods, analysis and insights from multimodal LLM pre-training. In European Conference on Computer Vision . Springer,
304–323.
[119] Kai Mei, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. 2024. Llm agent operating system. arXiv preprint arXiv:2403.16971
(2024).
[120] Jamshed Memon, Maira Sami, Rizwan Ahmed Khan, and Mueen Uddin. 2020. Handwritten optical character recognition (OCR): A comprehensive
systematic literature review (SLR). IEEE access 8 (2020), 142642–142668.
[121] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-
Gillingham, Geoffrey Irving, and Nat McAleese. 2022. Teaching language models to support answers with verified quotes. arXiv:2203.11147 [cs.CL]
https://arxiv.org/abs/2203.11147
[122] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book
question answering. arXiv preprint arXiv:1809.02789 (2018).
[123] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William
Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2022.
WebGPT: Browser-assisted question-answering with human feedback. arXiv:2112.09332 [cs.CL] https://arxiv.org/abs/2112.09332
[124] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. 2023. Memgpt: Towards llms as
operating systems. arXiv preprint arXiv:2310.08560 (2023).
Manuscript submitted to ACM

32 Chen et al.
[125] Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al .2023. Check
your facts and try again: Improving large language models with external knowledge and automated feedback. arXiv preprint arXiv:2302.12813
(2023).
[126] Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. 2023. The impact of ai on developer productivity: Evidence from github copilot.
arXiv preprint arXiv:2302.06590 (2023).
[127] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large
language models to the world. arXiv preprint arXiv:2306.14824 (2023).
[128] Perplexity AI. 2025. Perplexity AI. https://www.perplexity.ai/. Accessed: 2025-05-30.
[129] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin,
Jean Maillard, et al. 2020. KILT: a benchmark for knowledge intensive language tasks. arXiv preprint arXiv:2009.02252 (2020).
[130] Jie Qin, Jie Wu, Weifeng Chen, Yuxi Ren, Huixia Li, Hefeng Wu, Xuefeng Xiao, Rui Wang, and Shilei Wen. 2024. Diffusiongpt: LLM-driven
text-to-image generation system. arXiv preprint arXiv:2401.10061 (2024).
[131] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al .2023. Toolllm: Facilitating
large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789 (2023).
[132] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024. Tool Learning with Large
Language Models: A Survey. arXiv preprint arXiv:2405.17935 (2024).
[133] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack
Clark, et al .2021. Learning transferable visual models from natural language supervision. In International conference on machine learning . PMLR,
8748–8763.
[134] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale
weak supervision. In International conference on machine learning . PMLR, 28492–28518.
[135] RADLogics. 2021. Use of AI to Analyze Chest CT Shortens Turnaround Times in Russia. https://www.auntminnieeurope.com/imaging-informatics/
artificial-intelligence/article/15655440/use-of-ai-to-analyze-chest-ct-shortens-turnaround-times-in-russia. Accessed: 2025-05-30.
[136] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring
the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.
[137] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv
preprint arXiv:1606.05250 (2016).
[138] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-
augmented language models. Transactions of the Association for Computational Linguistics 11 (2023), 1316–1331.
[139] Rita Ramos, Bruno Martins, Desmond Elliott, and Yova Kementchedjhieva. 2023. Smallcap: Lightweight Image Captioning Prompted with Retrieval
Augmentation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) . 2840–2849. https://doi.org/10.1109/CVPR52729.
2023.00278
[140] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan
Yang, and Fahad S Khan. 2024. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition . 13009–13018.
[141] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky,
Jackie Kay, Jost Tobias Springenberg, et al. 2022. A generalist agent. arXiv preprint arXiv:2205.06175 (2022).
[142] Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong Wen, and Haifeng Wang. 2023. Investigating the
factual knowledge boundary of large language models with retrieval augmentation. arXiv preprint arXiv:2307.11019 (2023).
[143] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video
understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 14313–14323.
[144] Sherry Ruan and Tian Zhao. 2024. JungleGPT: Designing and Optimizing Compound AI Systems for E-Commerce. arXiv preprint arXiv:2407.00038
(2024).
[145] Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. 2022. Improving
passage retrieval with zero-shot question generation. arXiv preprint arXiv:2204.07496 (2022).
[146] Jaromir Savelka, Kevin D Ashley, Morgan A Gray, Hannes Westermann, and Huihui Xu. 2023. Explaining legal concepts with augmented large
language models (gpt-4). arXiv preprint arXiv:2306.09525 (2023).
[147] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas
Scialom. 2024. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems 36 (2024).
[148] Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. ZeroSCROLLS: A zero-shot benchmark for long text understanding.
arXiv preprint arXiv:2305.14196 (2023).
[149] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language
models with iterative retrieval-generation synergy. arXiv preprint arXiv:2305.15294 (2023).
[150] Nikhil Sharma, Q Vera Liao, and Ziang Xiao. 2024. Generative Echo Chamber? Effect of LLM-Powered Search Systems on Diverse Information
Seeking. In Proceedings of the CHI Conference on Human Factors in Computing Systems . 1–17.
[151] Zhuocheng Shen. 2024. Llm with tools: A survey. arXiv preprint arXiv:2409.18807 (2024).
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 33
[152] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models. arXiv preprint arXiv:2301.12652 (2023).
[153] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement
learning. Advances in Neural Information Processing Systems 36 (2023), 8634–8652.
[154] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models
that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 8317–8326.
[155] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the
domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for
Computational Linguistics 11 (2023), 1–17.
[156] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. 2023. Llm-planner: Few-shot grounded planning for
embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision . 2998–3009.
[157] Theodore Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas Griffiths. 2023. Cognitive architectures for language agents. Transactions on
Machine Learning Research (2023).
[158] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2024.
Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .
14398–14409.
[159] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang.
2023. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222 (2023).
[160] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at
search? investigating large language models as re-ranking agents. arXiv preprint arXiv:2304.09542 (2023).
[161] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al .
2023. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525 (2023).
[162] Yashar Talebirad and Amirhossein Nadiri. 2023. Multi-agent collaboration: Harnessing the power of intelligent llm agents. arXiv preprint
arXiv:2306.03314 (2023).
[163] John Chong Min Tan and Mehul Motani. 2023. Large language model (llm) as a system of multiple expert agents: An approach to solve the
abstraction and reasoning corpus (arc) challenge. arXiv preprint arXiv:2310.05146 (2023).
[164] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2023. Salmonn: Towards generic
hearing abilities for large language models. arXiv preprint arXiv:2310.13289 (2023).
[165] Fei Tang, Wanling Gao, Jianfeng Zhan, Chuanxin Lan, Xu Wen, Lei Wang, Chunjie Luo, Zheng Cao, Xingwang Xiong, Zihan Jiang, et al .2021.
AIBench training: Balanced industry-standard AI training benchmarking. In 2021 IEEE International Symposium on Performance Analysis of Systems
and Software (ISPASS) . IEEE, 24–35.
[166] Xiangru Tang, Yuliang Liu, Zefan Cai, Yanjun Shao, Junjie Lu, Yichi Zhang, Zexuan Deng, Helan Hu, Kaikai An, Ruijun Huang, et al .2023.
ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code. arXiv preprint arXiv:2311.09835
(2023).
[167] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2020.
Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006 (2020).
[168] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al .2022. Transformer
memory as a differentiable search index. Advances in Neural Information Processing Systems 35 (2022), 21831–21843.
[169] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth,
Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
[170] Llama Team. 2024. LlamaIndex GitHub Repository. https://github.com/run-llama/llama_index. Accessed: 2024-12-24.
[171] LangChain AI Team. 2024. LangChain GitHub Repository. https://github.com/langchain-ai/langchain. Accessed: 2024-12-24.
[172] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot
evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
[173] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and
VERification. arXiv preprint arXiv:1803.05355 (2018).
[174] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
[175] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. arXiv preprint arXiv:2212.10509 (2022).
[176] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. ♪MuSiQue: Multihop Questions via Single-hop Question
Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539–554.
[177] YunDa Tsai, Mingjie Liu, and Haoxing Ren. 2024. RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Model. In Proceedings of
the 61st ACM/IEEE Design Automation Conference . 1–6.
[178] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is
all you need. Advances in neural information processing systems 30 (2017).
Manuscript submitted to ACM

34 Chen et al.
[179] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An
open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023).
[180] Yubo Wang, Xueguang Ma, and Wenhu Chen. 2023. Augmenting black-box llms with medical textbooks for clinical question answering. arXiv
preprint arXiv:2309.02233 (2023).
[181] Yuhao Wang, Yichao Wang, Zichuan Fu, Xiangyang Li, Wanyu Wang, Yuyang Ye, Xiangyu Zhao, Huifeng Guo, and Ruiming Tang. 2024. Llm4msr:
An llm-enhanced paradigm for multi-scenario recommendation. In Proceedings of the 33rd ACM International Conference on Information and
Knowledge Management . 2472–2481.
[182] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. 2023. Learning to filter context for retrieval-augmented
generation. arXiv preprint arXiv:2311.08377 (2023).
[183] Zhiruo Wang, Zhoujun Cheng, Hao Zhu, Daniel Fried, and Graham Neubig. 2024. What are tools anyway? a survey from the language model
perspective. arXiv preprint arXiv:2403.15452 (2024).
[184] Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian
Yang, et al .2023. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. arXiv preprint arXiv:2310.00746
(2023).
[185] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned
language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021).
[186] Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi, Liang He, and Yu Qiao. 2023. Dilu: A knowledge-driven
approach to autonomous driving with large language models. arXiv preprint arXiv:2309.16292 (2023).
[187] Zhihua Wen, Zhiliang Tian, Zexin Jian, Zhen Huang, Pei Ke, Yifu Gao, Minlie Huang, and Dongsheng Li. 2024. Perception of Knowledge Boundary
for Large Language Models through Semi-open-ended Question Answering. In The Thirty-eighth Annual Conference on Neural Information Processing
Systems . https://openreview.net/forum?id=Li9YTHoItP
[188] Wikipedia contributors. n.d.. Retrieval-augmented generation. https://en.wikipedia.org/wiki/Retrieval-augmented_generation [Online; accessed
18-December-2024].
[189] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al .2023. Autogen:
Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155 (2023).
[190] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. 2023. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519
(2023).
[191] Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, and Umar Iqbal. 2024. SecGPT: An execution isolation architecture for llm-based
systems. arXiv preprint arXiv:2403.04960 (2024).
[192] Yuxiang Wu, Yu Zhao, Baotian Hu, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2022. An efficient memory-augmented transformer
for knowledge-intensive nlp tasks. arXiv preprint arXiv:2210.16773 (2022).
[193] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. 2023. Embodied task planning with large language models. arXiv preprint
arXiv:2307.01848 (2023).
[194] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Recomp: Improving retrieval-augmented lms with compression and selective augmentation.
arXiv preprint arXiv:2310.04408 (2023).
[195] Jiaming Xu, Kaibin Guo, Wuxuan Gong, and Runyu Shi. 2024. OSAgent: Copiloting Operating System with LLM-based Agent. In 2024 International
Joint Conference on Neural Networks (IJCNN) . IEEE, 1–9.
[196] Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. 2023. On the tool manipulation capability of open-source large
language models. arXiv preprint arXiv:2305.16504 (2023).
[197] Bingyu Yan, Xiaoming Zhang, Litian Zhang, Lian Zhang, Ziyi Zhou, Dezhuang Miao, and Chaozhuo Li. 2025. Beyond self-talk: A communication-
centric survey of llm-based multi-agent systems. arXiv preprint arXiv:2502.14321 (2025).
[198] Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Berkeley Function Calling
Leaderboard. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html.
[199] Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, and Animashree Anandkumar. 2024.
Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems 36 (2024).
[200] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. 2023. Gpt4tools: Teaching large language model to use tools via
self-instruction. Advances in Neural Information Processing Systems 36 (2023), 71995–72007.
[201] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600 (2018).
[202] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem
solving with large language models. Advances in neural information processing systems 36 (2023), 11809–11822.
[203] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in
language models. In International Conference on Learning Representations (ICLR) .
[204] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih.
2022. Retrieval-augmented multimodal language modeling. arXiv preprint arXiv:2211.12561 (2022).
Manuscript submitted to ACM

From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems 35
[205] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. 2024. Woodpecker:
Hallucination correction for multimodal large language models. Science China Information Sciences 67, 12 (2024), 220105.
[206] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023. Making retrieval-augmented language models robust to irrelevant context. arXiv
preprint arXiv:2310.01558 (2023).
[207] Hao Yu, Yiling Lou, Ke Sun, Dezhi Ran, Tao Xie, Dan Hao, Ying Li, Ge Li, and Qianxiang Wang. 2022. Automated assertion generation via
information retrieval and its integration with deep learning. In Proceedings of the 44th International Conference on Software Engineering . 163–174.
[208] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large
multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 (2023).
[209] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin Ma, Hongwei Wang, and Dong Yu. 2023. Chain-of-note: Enhancing robustness in retrieval-
augmented language models. arXiv preprint arXiv:2311.09210 (2023).
[210] Youngjae Yu, Jiwan Chung, Heeseung Yun, Jack Hessel, JaeSung Park, Ximing Lu, Prithviraj Ammanabrolu, Rowan Zellers, Ronan Le Bras, Gunhee
Kim, et al. 2022. Multimodal knowledge alignment with reinforcement learning. arXiv preprint arXiv:2205.12630 (2022).
[211] Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. 2023. Augmentation-adapted retriever improves generalization of language models as generic
plug-in. arXiv preprint arXiv:2305.17331 (2023).
[212] Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared, Heather Miller, Chris Potts, James Zou, Micha, Jonathan Frankle, Naveen Rao, and Ali Ghodsi.
2024. The Shift from Models to Compound AI Systems. https://bair.berkeley.edu/blog/2024/02/18/compo.
[213] Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, and Chen Change Loy. 2024. Contextual object detection with multimodal large language models.
International Journal of Computer Vision (2024), 1–19.
[214] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo,
Vikas Sindhwani, et al .2022. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598 (2022).
[215] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder:
Repository-level code completion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570 (2023).
[216] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. 2023. Gpt4roi: Instruction tuning
large language model on region-of-interest. arXiv preprint arXiv:2307.03601 (2023).
[217] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. 2024. Raft: Adapting language model to
domain specific rag. arXiv preprint arXiv:2403.10131 (2024).
[218] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023. Multimodal chain-of-thought reasoning in language
models. arXiv preprint arXiv:2302.00923 (2023).
[219] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In
Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38. 19632–19642.
[220] Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, and Sibei Yang. 2023. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning
in language models. Advances in Neural Information Processing Systems 36 (2023), 5168–5191.
[221] Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022. Towards a unified
multi-dimensional evaluator for text generation. arXiv preprint arXiv:2210.07197 (2022).
[222] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al .2023.
Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854 (2023).
[223] Shengyao Zhuang, Houxing Ren, Linjun Shou, Jian Pei, Ming Gong, Guido Zuccon, and Daxin Jiang. 2022. Bridging the gap between indexing and
retrieval for differentiable search index with query generation. arXiv preprint arXiv:2206.10128 (2022).
Manuscript submitted to ACM