# Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges

**Authors**: Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui Zhao, Ziyue Li

**Published**: 2025-06-12 07:01:56

**PDF URL**: [http://arxiv.org/pdf/2506.10408v1](http://arxiv.org/pdf/2506.10408v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to
overcome the knowledge limitations of Large Language Models (LLMs) by
integrating external retrieval with language generation. While early RAG
systems based on static pipelines have shown effectiveness in well-structured
tasks, they struggle in real-world scenarios requiring complex reasoning,
dynamic retrieval, and multi-modal integration. To address these challenges,
the field has shifted toward Reasoning Agentic RAG, a paradigm that embeds
decision-making and adaptive tool use directly into the retrieval process. In
this paper, we present a comprehensive review of Reasoning Agentic RAG methods,
categorizing them into two primary systems: predefined reasoning, which follows
fixed modular pipelines to boost reasoning, and agentic reasoning, where the
model autonomously orchestrates tool interaction during inference. We analyze
representative techniques under both paradigms, covering architectural design,
reasoning strategies, and tool coordination. Finally, we discuss key research
challenges and propose future directions to advance the flexibility,
robustness, and applicability of reasoning agentic RAG systems. Our collection
of the relevant research has been organized into a
https://github.com/ByebyeMonica/Reasoning-Agentic-RAG.

## Full Text


<!-- PDF content starts -->

arXiv:2506.10408v1  [cs.AI]  12 Jun 2025Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic
Retrieval-Augmented Generation for Industry Challenges
Jintao Liang1,Gang Su2,Huifeng Lin3,You Wu3,Rui Zhao5,6and Ziyue Li4∗
1Beijing University of Posts and Telecommunications
2University of Georgia
3South China University of Technology
4Technical University of Munich, University of Cologne
5SenseTime Research
6Qingyuan Research Institute, Shanghai Jiaotong University
ljt2021@bupt.edu.cn, {gangsuedu,huifeng.work,wuyouscut }@gmail.com, zhaorui@sensetime.com,
zlibn@wiso.uni-koeln.de
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as a powerful framework to overcome the
knowledge limitations of Large Language Models
(LLMs) by integrating external retrieval with
language generation. While early RAG systems
based on static pipelines have shown effectiveness
in well-structured tasks, they struggle in real-world
scenarios requiring complex reasoning, dynamic
retrieval, and multi-modal integration. To address
these challenges, the field has shifted toward
Reasoning Agentic RAG , a paradigm that embeds
decision-making and adaptive tool use directly
into the retrieval process. In this paper, we present
a comprehensive review of Reasoning Agentic
RAG methods, categorizing them into two primary
systems: predefined reasoning , which follow
fixed modular pipelines to boost reasoning, and
agentic reasoning , where the model autonomously
orchestrates tool interaction during inference.
We analyze representative techniques under both
paradigms, covering architectural design, reasoning
strategies, and tool coordination. Finally, we
discuss key research challenges and propose future
directions to advance the flexibility, robustness, and
applicability of reasoning agentic RAG systems.
Our collection of the relevant researches has been
organized into a Github Repository.
1 Introduction
Large Language Models (LLMs) [Singh, 2023; Zhao et
al., 2023; Zhu et al. , 2024 ]have demonstrated remarkable
capabilities in natural language understanding and gener-
ation, enabling a wide array of applications from open-
domain question-answer (QA) to task-specific dialogue sys-
tems. However, LLMs rely on static training data, mak-
ing them prone to hallucinations and limiting their abil-
∗Corresponding author.ity to provide accurate, up-to-date information in dynamic
or knowledge-intensive tasks [Rawte et al. , 2023; Zhang
et al. , 2023; Huang et al. , 2025 ]. Retrieval-Augmented
Generation (RAG) [Chen et al. , 2024; Lewis et al. , 2020;
Gao et al. , 2023 ]has attracted significant attention as a
promising approach to overcome the knowledge limitations
of LLMs resulting from static pretraining. By integrating rel-
evant information from external knowledge bases or search
engines, RAG enhances factual accuracy and broadens the
model’s temporal and domain coverage [Zhao et al. , 2024;
Liet al. , 2024a ]. Traditional RAG methods have demon-
strated strong performance when queries are well-formed and
the necessary information is readily available in the retrieved
context.
Despite the effectiveness of basic RAG methods, they often
struggle when applied to real-world, industrial-scale applica-
tions involving complex and heterogeneous data. For example,
in multi-document scenarios, relevant information is spread
across sources, requiring not just retrieval but also coherent
synthesis [Wang et al. , 2025; Wang et al. , 2024b ]. Naively
concatenating retrieved passages can lead to fragmented or
contradictory responses, particularly in domains like legal or
biomedical QA where multi-hop reasoning is critical. Addi-
tionally, most RAG systems are limited to text-only processing
and cannot natively handle multi-modal inputs such as tables,
charts, or images [Maet al. , 2024; Yu et al. , 2025 ]. This limits
their ability to operate in data-rich environments like enterprise
intelligence, scientific reporting, or technical support, where
visual and structured data play a central role [Linet al. , 2023a;
Yuet al. , 2024 ].
To address these limitations of basic RAG in handling com-
plex, real-world tasks, recent research has turned to Agentic
RAG [Ravuru et al. , 2024 ], a paradigm that tightly integrates
retrieval with reasoning and decision-making. Unlike static
pipelines, Agentic RAG treats retrieval not as a one-off pre-
processing step, but as a dynamic, context-sensitive opera-
tion guided by the model’s ongoing reasoning process. This
reasoning-centric perspective is crucial for applications that
demand multi-step problem solving, adaptive information ac-
quisition, and tool-assisted synthesis. Within this paradigm, as

DesignFollowLLMHumanPredefined Reasoning
Reasoning
Agentic ReasoningExternal ToolsTool Calling
Retrieved InformationFigure 1: Overview of two major types of reasoning Agentic Systems.
shown in Figure 1, two major types of reasoning agentic sys-
tems have emerged based on how control and decision-making
are handled: predefined reasoning , which follow structured,
rule-based plans with fixed pipelines to boost reasoning for
retrieval and generation; and agentic reasoning , where the
model actively monitors its reasoning process and determines
when and how to retrieve or interact with external tools. These
two workflows form the basis of Reasoning Agentic RAG ,
which unifies structured and autonomous approaches for more
intelligent, context-aware retrieval-augmented reasoning.
Predefined reasoning adopts structured and modular RAG
pipelines where the retrieval and reasoning steps are explicitly
designed, following fixed control pipeline. These workflows
typically decompose tasks into discrete components such as
query reformulation, document retrieval, re-ranking, and an-
swer synthesis, executed in a linear or orchestrated fashion.
In general, predefined reasoning spans several architectural
variants: route-based methods selectively trigger retrieval
based on context or model uncertainty, such as low confi-
dence scores or ambiguous intermediate outputs [Wang et al. ,
2024a ];loop-based methods enable limited iteration through
retrieval-feedback cycles, supporting multiple rounds of re-
finement [Asai et al. , 2023; Yang et al. , 2024b ];tree-based
methods organize information hierarchically to support struc-
tured exploration [Sarthi et al. , 2024; Hu et al. , 2025 ]; and
hybrid-modular frameworks compose specialized modules
into a flexible but still rule-driven workflow [Jeong et al. , 2024;
Gao et al. , 2024 ]. These workflows prioritize control and
modularity, making them suitable for tasks requiring efficient
computation and customization. However, their reasoning
remains constrained by predesigned execution paths, limiting
flexibility in evolving and open-ended tasks.
Agentic reasoning repositions the LLM as an active deci-
sion maker, that autonomously orchestrates retrieval and tooluse throughout the reasoning process. Instead of executing
a fixed plan, the model identifies knowledge gaps, formu-
lates queries, retrieves external information via tools such
as search engines or APIs, and integrates the retrieved con-
tents into an evolving solution. This dynamic interplay of
reasoning and tool use enables the system to tackle complex,
multi-turn tasks that require iterative refinement and adaptive
information synthesis. There are two primary methods for im-
plementing agentic reasoning. The first is prompt-based meth-
ods, which leverages the in-context reasoning and instruction-
following capabilities of pretrained LLMs [Yaoet al. , 2023;
Press et al. , 2023; Li et al. , 2025a ]. In this setting, the model
is guided by carefully crafted prompts or embedded control
tokens that instruct it when to retrieve, what actions to take,
and how to integrate external information. These methods re-
quire no additional training, making them lightweight and
adaptable across tasks. The second paradigm is training-
based methods, where models are explicitly optimized through
reinforcement learning, to determine when and how to in-
voke external tools [Jiang et al. , 2025; Jin et al. , 2025;
Zheng et al. , 2025 ]. This paradigm enables more fine-grained
and strategic tool usage, enabling models to learn long-term
planning and develop retrieval policies tailored to complex
tasks. Owing to its autonomy and adaptability, agentic rea-
soning has shown strong performance in open-domain QA,
scientific reasoning, and multi-stage decision-making scenar-
ios.
Perspective of Cognitive Science - System 1 and System
2:To further contextualize predefined and agentic reason-
ing within the dual-process theory of cognition—commonly
referred to as System 1 and System 2 thinking [Yang et al. ,
2024a; Li et al. , 2025b ]— we can draw an analogy between
these RAG paradigms and human cognitive modes.
•Predefined reasoning resembles System 1 thinking: fast,
structured, and efficient, relying on predefined heuristics
and modular workflows that mirror habitual or rule-based
cognition. While this enables rapid execution and pre-
dictable behavior, it often lacks the flexibility to adapt
beyond its design.
•In contrast, agentic reasoning aligns more closely with
System 2 thinking: slow, deliberative, and adaptive. Here,
the LLM actively engages in reasoning, planning, and
decision-making, dynamically leveraging external tools
and retrieved knowledge to address complex, novel tasks.
This reflective mode allows the model to identify gaps,
reassess strategies, and adjust its behavior—traits charac-
teristic of conscious, analytical human reasoning.
By framing these paradigms through the lens of cognitive
systems, we highlight the trade-off between efficiency and
adaptability, and the growing capacity of agentic RAG to emu-
late more sophisticated, human-like problem solving. Table 1
aligns predefined and agentic reasoning with the dual-system
theory from cognitive science, illustrating their respective con-
trol structures and behavioral characteristics.
The paper systematically reviews and analyzes the current
research approaches and future development paths of Rea-
soning Agentic RAG, summarizing them into two primary
technical paradigms. The remainder of the paper is organized

System Type Reasoning Workflow Description
System 1 Predefined Reasoning Structured, modular, rule-based execution.
System 2 Agentic Reasoning Autonomous, adaptive, model-driven decision-making.
Table 1: Cognitive system alignment of reasoning workflows.
as follows: Section 2 introduces related work; Section 3 and
Section 4 dive into the two types of reasoning workflows
within Agentic RAG, predefined reasoning and agentic reason-
ing, respectively. Section 5 outlines future research directions,
and Section 6 concludes the paper.
Figure 2: Distributed Works of Reasoning Agentic RAG.
2 Related Work
2.1 Basic RAG
Retrieval-Augmented Generation (RAG) was introduced to
overcome the static knowledge limitations of LLMs by inte-
grating external retrieval mechanisms during inference [Chen
et al. , 2024; Gao et al. , 2023 ]. Naive RAG methods represent
the earliest implementations, typically using sparse retrieval
techniques like BM25 [Robertson et al. , 2009 ]to fetch doc-
uments based on keyword overlap [Maet al. , 2023 ]. While
efficient for simple factoid queries, these approaches offered
limited semantic understanding, thus often retrieving noisy
or redundant content and failing to reason across multiple
sources.
The emergence of Advanced RAG and Modular RAG was
aimed at addressing key limitations of the Naive RAG, par-
ticularly in terms of retrieval precision, information integra-
tion, and system flexibility [Gao et al. , 2023 ]. Advanced
RAG improves retrieval quality through techniques such as
dense semantic matching, re-ranking, and multi-hop query-
ing, while also introducing refined indexing strategies like
fine-grained chunking and metadata-aware retrieval. Modular
RAG rethinks the Naive RAG by breaking down the end-to-
end process of indexing, retrieval, and generation into discrete,
configurable modules. This design allows for greater architec-
tural flexibility and enables system developers to incorporate
diverse techniques into specific stages, such as enhancing re-
trieval with fine-tuned search modules [Linet al. , 2023b ]. In
response to specific task demands, various restructured and
iterative module designs have also emerged. As a result, mod-
ular RAG has increasingly become a dominant paradigm in
the field, supporting both serialized pipeline execution and
end-to-end learning across modular components.Despite their effectiveness, basic RAG workflows are lim-
ited by static control logic and lack the ability to reflect, adapt,
or assess the sufficiency of retrieved information. These con-
straints reduce their suitability for tasks requiring iterative
reasoning, tool use, or multi-modal integration. Thus, Agentic
RAG has proposed to embed reasoning and decision-making
into the retrieval process. This work focuses on reasoning
Agentic RAG approaches that enable more autonomous and
context-aware information processing.
2.2 Reasoning Agentic RAG
The year 2025 is marked as the year of agentic AI, with appli-
cations emerging such as agentic LLMs and so on [Ruan et
al., 2023; Kong et al. , 2024; Zhang et al. ,]. Recent advances
in RAG have seen a shift from static, rule-driven retrieval
pipelines toward dynamic, reasoning-driven architectures, col-
lectively referred to as Reasoning Agentic RAG . These systems
embed decision-making into the retrieval process, enabling
models to actively determine when, what, and how to retrieve
based on their internal reasoning trajectory. As shown in Fig-
ure 3, Reasoning Agentic RAG approaches can be broadly
categorized into two paradigms: predefined reasoning and
agentic reasoning .
Predefined reasoning depends on structured, rule-based
pipelines where the retrieval and reasoning stages are modu-
larized and fixed in advance. These workflows often include
components for query reformulation, document retrieval, re-
ranking, and response generation, coordinated by static control
logic. RAGate [Wang et al. , 2024a ]exemplifies route-based
designs, where retrieval is conditionally triggered based on the
context or model confidence, enabling the system to skip un-
necessary operations and focus on knowledge-intensive inputs.
Self-RAG [Asai et al. , 2023 ]introduces loop-based reasoning
by enabling the model to self-reflect and iteratively refine its
responses, while RAPTOR [Sarthi et al. , 2024 ]leverages a
recursive tree structure to hierarchically summarize and orga-
nize retrieved content, supporting multi-hop and abstractive
reasoning. Building on these foundations, more advanced
frameworks like Adaptive-RAG [Jeong et al. , 2024 ]combine
dynamic routing and retrieval adaptation, enabling models to
select optimal reasoning paths. Modular-RAG [Gao et al. ,
2024 ]extends this idea by dividing the RAG pipeline into
interoperable modules like retrievers, rerankers and genera-
tors, which can be flexibly composed into hybrid workflows.
These designs enabling more flexible orchestration while still
operating under predefined execution paths.
Agentic reasoning empowers the LLM to act as an au-
tonomous agent, dynamically deciding how to interact with
external tools based on its current reasoning state. These
workflows tightly couple reasoning with tool use, enabling
the model to issue retrieval queries, assess results, and itera-

Reasoning Agentic RAGPredefined reasoningRoute-based RAGate [Wang et al. , 2024a ], Self-Route [Liet al. , 2024b ]
Loop-based Self-RAG [Asai et al. , 2023 ], CRAG [Yan et al. , 2024 ]
Tree-based RAPTOR [Sarthi et al. , 2024 ], MCTS-RAG [Huet al. , 2025 ]
Hybrid-modular Adaptive-RAG [Jeong et al. , 2024 ], Modular-RAG [Gao et al. , 2024 ]
Agentic reasoningPrompt-basedReAct [Yao et al. , 2023 ], Self-Ask [Press et al. , 2023 ],
Function calling [Eleti et al. , 2023 ], Search-O1 [Liet al. , 2025a ]
Training-basedDeepRetrieval [Jiang et al. , 2025 ], Search-R1 [Jinet al. , 2025 ], R1-Searcher [Song et al. , 2025 ],
ReZero [Dao and Le, 2025 ], DeepResearcher [Zheng et al. , 2025 ]
Figure 3: A taxonomy of Reasoning Agentic RAG.
2022202320252024Agentic ReasoningPredefined ReasoningSelf-RAGRAPTORCRAGRAGateSelf-RouteMCTS-RAGAdaptive-RAGModular-RAGSelf-ASKReActSearch-O1DeepRetrievalReZeroFunction CallingR1-SearcherSearch-R1Loop-basedRouter-basedTree-basedHybrid-modular 
Prompt-basedTraining-based
Figure 4: Illustration of the evolution of Reasoning Agentic RAG.
tively adapt its actions. Two main implementation strategies
have emerged: prompt-based andtraining-based approaches.
Prompt-based methods leverage the instruction-following abil-
ities of pretrained LLMs to drive agentic behavior without
additional training. For example, ReAct [Yao et al. , 2023 ]
interleaves reasoning steps with tool use to guide retrieval
based on emerging knowledge gaps. Other methods like
Self-Ask [Press et al. , 2023 ]and Search-o1 [Liet al. , 2025a ]
support decomposition into sub-questions or trigger retrieval
mid-generation. Additionally, function calling mechanisms
[Eleti et al. , 2023 ]built into commercial LLMs such as GPT
and Gemini offer structured interfaces for tool use, further
enabling prompt-based agentic control. In parallel, training-
based approaches aim to explicitly teach LLMs to reason
and retrieve in a unified, goal-driven manner by leveraging
reinforcement learning (RL) to optimize tool-use behavior.
DeepRetrieval [Jiang et al. , 2025 ]trains models to reformulate
queries by maximizing retrieval metrics. Search-R1 [Jinet
al., 2025 ]and R1-Searcher [Song et al. , 2025 ]both adopt atwo-stage, outcome-driven RL framework that enables LLMs
to learn when and what to search within a reasoning trajectory.
ReZero [Dao and Le, 2025 ]incentivizes persistence, reward-
ing effective retry strategies. DeepResearcher [Zheng et al. ,
2025 ]pushes further by training agents in open web environ-
ments, enabling robust search and synthesis across diverse,
unstructured sources.
3 Predefined Reasoning
Agents and RAG are increasingly integrated in advanced AI
systems. By augmenting LLMs with external knowledge
retrieval, RAG enables agents to ground their reasoning in
relevant information. In turn, agent-based reasoning which
includes planning, tool use and self-reflection, enhances RAG
by guiding the model on what information to retrieve and
how to incorporate it into the reasoning process. This synergy
supports a predefined reasoning, where the agent iteratively
queries external sources (e.g., a local database or web search)
and refines its reasoning based on the retrieved evidence. We

categorize predefined RAG reasoning workflows into four
broad types based on their structural and reasoning character-
istics as follows.
Route-based Approaches: RAG incorporates dynamic
routing mechanisms that direct queries along different retrieval
or reasoning paths based on predefined conditions—such
as query type, model uncertainty, or confidence estima-
tion—while still operating within a fixed architecture. RAGate
[Wang et al. , 2024a ]uses the conversation context and model
confidence to route only those dialogue turns that truly re-
quire external knowledge to a RAG process. This ensures
the system can bypass retrieval for straightforward prompts
while invoking it for knowledge-intensive queries, exemplify-
ing conditional RAG in dialogue. Self-Route [Liet al. , 2024b ]
introduced dynamically routes queries to either RAG or Long-
Context (LC) models based on the model’s confidence-based
routing. This method significantly reduces computation cost
while maintaining performance comparable to LC models.
Loop-based Approaches: RAG operates within a feedback
loop that supports multiple rounds of refinement. The system
can self-reflect, critique intermediate outputs, and iteratively
update retrieval inputs to improve generation quality. Self-
RAG [Asai et al. , 2023 ]is a foundational example of this
controlled reasoning loop. In the Self-RAG workflow, a single
LLM agent engages in self-reflection during generation to
improve its output. Instead of relying on a fixed retrieved con-
text, the model can decide mid-generation to fetch additional
information or to critique its own draft answer. CRAG [Yan
et al. , 2024 ]introduced loop-based corrective feedback mech-
anism into the retrieval process. In the CRAG workflow, a
lightweight retrieval evaluator assigning the confidence scores
about the quality of the retrieved chunks/documents — cate-
gorized as correct, incorrect, or ambiguous. When retrieval
quality is deemed suboptimal, the system activates corrective
strategies such as query rewriting or external web search to
gather better evidence. The system refines the retrieved con-
tent into a focused context and iteratively improves retrieval
until a satisfactory output is generated.
Tree-based Approaches: RAG organizes the retrieval pro-
cess hierarchically, often using recursive structures such as
trees to support multi-hop reasoning or document summariza-
tion. RAPTOR [Sarthi et al. , 2024 ]introduces a recursive
tree structure from documents, allowing for more efficient and
context-aware information retrieval. This approach enhances
RAG by creating a summary tree from text chunks, providing
deeper insights and overcoming limitations of short, contigu-
ous text retrieval. MCTS-RAG [Huet al. , 2025 ]integrates
a Monte Carlo Tree Search loop into the RAG process for
complex reasoning tasks. MCTS-RAG dynamically integrates
retrieval and reasoning through an iterative decision-making
process. Unlike standard RAG methods, which typically re-
trieve information independently from reasoning and thus
integrate knowledge suboptimally, or conventional MCTS rea-
soning, which depends solely on internal model knowledge
without external facts, MCTS-RAG combines structured rea-
soning with adaptive retrieval. Hybrid-modular Approaches:
RAG in its most flexible form combines routing, looping, re-
flection, and modular orchestration. Tasks are divided among
specialized components, coordinated by an agent that can dy-
Question
LLMSub-QuestionsRetrievalReflection. Are there knowledge gaps?Yes, generate new questionsNo, SummarizeAnswerQuestion DecompositionPredefined ReasoningFigure 5: A demonstration of Predefined Reasoning.
namically reconfigure the workflow according to the query or
reasoning context. Adaptive-RAG [Jeong et al. , 2024 ]extends
the Self-RAG framework by introducing routing mechanisms
that enable dynamic path selection. In addition to allowing the
model to interleave retrieval and generation steps, it equips the
agent with a decision-making router that selects appropriate
retrieval strategies or reasoning pathways based on the query
characteristics or the agent’s own uncertainty. Rather than
simply determining whether to retrieve more information, the
agent can choose which retrieval method to apply, what type
of information to prioritize, or which downstream modules to
engage. Modular-RAG [Gaoet al. , 2024 ]is the most advanced
incarnation that transform RAG into a LEGO-like modular
framework, breaking the RAG process into an orchestrated
pipeline of specialized modules. Rather than a single agent
handling everything, a Modular-RAG architecture compart-
mentalizes tasks, e.g., one module for query reformulation,
one for document retrieval, another for ranking or filtering
results, and another for answer synthesis – all chained together
in a composable workflow. The pipeline is composed by an
agent that coordinates modular components, each of which
can be optimized or swapped independently.
This progression of predefine reasoning workflows reflects a
broader shift from static retrieval pipelines to dynamic, agent-
driven reasoning systems. Modern predefined reasoning in-
creasingly integrates planning, tool use, and decision-making
components that allow flexible orchestration of retrieval and
reasoning strategies. Rather than predefining rigid retrieval
steps, these systems empower agents to determine what in-
formation to seek, how to use it, and when to adapt their
approach—marking a move toward more autonomous and

intelligent knowledge integration. A summary of the repre-
sentative research works and open-source industrial/enterprise
implementations across these predefined RAG workflow types
is provided in Table 2.
4 Agentic Reasoning
Beyond the predefined reasoning mentioned above, a more
dynamic paradigm has emerged: the Agentic Reasoning . In
this setting, the LLM serves as an autonomous agent that not
only generates text, but also actively manages retrieval. With
advances in reasoning and instruction-following capabilities,
the model can identify knowledge gaps, determine when and
what to retrieve, and interact with external tools such as search
engines or APIs. This tight integration of reasoning and tool
use enables iterative decision-making, enabling the system to
refine its responses based on newly retrieved information. As
a result, agentic reasoning supports more flexible and adaptive
problem-solving, extending RAG beyond basic QA to com-
plex tasks such as scientific inquiry, multi-step reasoning, and
strategic decision-making. Agentic reasoning approaches can
be broadly categorized by how the LLM learns to use tools:
•Prompt-Based Approaches: These methods leverage
the instruction-following, in-context learning and reason-
ing capabilities of pretrained LLMs, guiding tool use
through carefully crafted prompts or built-in functionali-
ties without additional training.
•Training-Based Approaches: These methods involve
explicitly training LLMs, typically via reinforcement
learning, to learn when and how to interact with external
tools effectively.
A summary of representative agentic reasoning apporaches
and their characteristics is provided in Table 2. The following
sections examine representative frameworks and techniques
within each approach.
4.1 Prompt-Based Approaches
Prompt-based approaches harness the remarkable capabilities
already present in pre-trained LLMs to enable agentic behavior.
Instead of modifying the model’s weights through training,
these methods rely on sophisticated prompting techniques,
few-shot examples or built-in tool interfaces, to guide the
LLM in its interaction with external tools like search engines.
Function-Calling-Based : A foundational prompt-based
method for agentic behavior, and one way to implement func-
tion calling, is ReAct (Reason+Act) [Yaoet al. , 2023 ]. ReAct
aims to create a synergy between the reasoning processes
and action-taking capabilities within an LLM. Its core mecha-
nism involves prompting the LLM to generate outputs in an
interleaved sequence of Thought, Action, and Observation.
ReAct typically employs few-shot prompting, providing the
LLM with examples that demonstrate this Thought-Action-
Observation trajectory for solving similar tasks. These exam-
ples guide the frozen LLM on how to structure its reasoning,
utilize available tools, and progress towards the goal. The
framework demonstrated significant advantages, particularly
in grounding the LLM’s reasoning. By allowing the model to
actively seek and incorporate external information via actions,ReAct can mitigate the hallucination and error propagation is-
sues sometimes observed in purely internal reasoning methods
like Chain-of-Thought (CoT) [Weiet al. , 2023 ]. The explicit
reasoning traces (“Thoughts”) in ReAct enhance the inter-
pretability and transparency of the model’s decision-making.
Within RAG, ReAct offers a natural agentic reasoning pipeline:
the LLM’s ”Thought” process can identify a knowledge gap,
leading to a search ”Action,” with the retrieved results form-
ing the ”Observation” that informs subsequent reasoning. A
related method, Self-Ask [Press et al. , 2023 ], encourages step-
by-step problem decomposition by prompting the LLM to
generate and answer simpler follow-up questions. These inter-
mediate steps often involve search actions, enabling the model
to gather relevant information before attempting to answer the
main question.
Another prominent prompt-based approach involves lever-
aging the function calling or tool use capabilities that have
been explicitly built into or fine-tuned into certain LLMs, such
as versions of GPT [Eleti et al. , 2023 ], Llama, and Gem-
ini. This feature allows the LLM to interact reliably with
predefined external tools or APIs based on natural language
instructions. Function calling significantly expands the ca-
pabilities of LLMs beyond text generation, enabling them to
access real-time, dynamic information, interact with external
systems and databases, automate tasks, and reliably convert
natural language requests into structured API calls or database
queries. In contrast to the more open-ended ”thought-action-
observation” cycle of ReAct, function calling often bypasses
explicit intermediate reasoning steps. The LLM directly iden-
tifies the relevant tool and generates the necessary parameters
based on its training to recognize and format specific function
calls. This more direct approach relies on the model’s pre-
existing knowledge of available tools and their required inputs.
Furthermore, the format and capabilities of the tools acces-
sible via function calling are typically predefined and have
been integrated into the model’s training or prompt design.
For Agentic RAG, function calling provides a straightforward
and structured way for the LLM agent to invoke a search API
when its internal analysis determines that external information
is required to answer a prompt accurately.
Large Reasoning Model-based : A growing trend in Agen-
tic RAG workflow involves directly utilizing LLMs that pos-
sess inherently strong reasoning capabilities, often referred to
as Large Reasoning Models (LRMs). These models, some-
times developed through techniques like large-scale reinforce-
ment learning (e.g., models analogous to OpenAI’s o1 [Ope-
nAIet al. , 2024 ], DeepSeek-R1 [DeepSeek-AI et al. , 2025 ]),
are designed to excel at complex, multi-step reasoning tasks.
The underlying premise is that an LLM with superior intrin-
sic reasoning abilities will be better equipped to manage the
complexities of an Agentic RAG workflow, including decom-
posing challenging queries, planning information-gathering
steps, assessing the relevance and utility of retrieved infor-
mation, and synthesizing knowledge effectively. In essence,
leveraging LRMs within RAG represents a prompt-based agen-
tic strategy where the model’s powerful inherent reasoning
capabilities drive the process, implicitly deciding when and
how to retrieve information to support its complex thought
processes.

Predefined Reasoning
Approach Strategy Control Type Reasoning Complexity Code
RAGate [Wang et al. , 2024a ] Route-based Adaptive Medium Link
self-RAG [Asai et al. , 2023 ] Loop-based Agentic Medium Link
CRAG [Yanet al. , 2024 ] Loop-based Adaptive Medium Link
MCTS-RAG [Huet al. , 2025 ] Tree-based Agentic High Link
RAPTOR [Sarthi et al. , 2024 ] Tree-based Fixed Medium Link
Adaptive-RAG [Jeong et al. , 2024 ] Hybrid-modular Adaptive Medium Link
Modular-RAG [Gao et al. , 2024 ] Hybrid-modular Fixed Low N/A
DeepSearcher Industry Adaptive Medium Link
RAGFlow Industry Adaptive Medium Link
Haystack Industry Adaptive Medium Link
Langchain-Chatchat Industry Adaptive/Agentic Medium Link
LightRAG Industry Adaptive Medium Link
R2R Industry Agentic High Link
FlashRAG Industry Adaptive Medium Link
Agentic Reasoning
Approach Strategy Training environment Reward design Code
ReAct [Yaoet al. , 2023 ] Prompt-based N/A N/A Link
Self-Ask [Press et al. , 2023 ] Prompt-based N/A N/A Link
Funciton calling [Eleti et al. , 2023 ] Prompt-based N/A N/A N/A
Search-O1 [Liet al. , 2025a ] Prompt-based N/A N/A Link
Search-R1 [Jinet al. , 2025 ] Training-based Local retrieval system Answer reward Link
R1-Searcher [Song et al. , 2025 ] Training-based Local retrieval system Retrieval reward, format reward, answer reward Link
ReZero [Dao and Le, 2025 ] Training-based Local retrieval system Retrieval reward, format reward, answer reward, retry reward Link
DeepRetrieval [Jiang et al. , 2025 ] Training-based Restricted real-world search engine Retrieval reward, format reward Link
DeepResearcher [Zheng et al. , 2025 ]Training-based Real-world search engine Format reward, answer reward Link
Table 2: A summary of Reasoning agentic rag.
However, effectively managing the retrieved context is an-
other significant challenge. LLMs with extremely long con-
text windows can suffer from a ”lost-in-the-middle” problem,
where information presented in the middle of a long input
receives less attention. Furthermore, retrieved documents,
whether in long-context models or standard RAG, often con-
tain verbose, noisy or contradictory content that can disrupt
the coherence of the LLM’s reasoning process. Mitigating
this challenge requires more precise retrieval strategies and
adaptive context management mechanisms. The Search-o1
framework [Liet al. , 2025a ]is specifically designed to en-
hance LRMs by tackling knowledge insufficiency during long,
step-by-step reasoning chains. It integrates two core compo-
nents: an Agentic RAG Mechanism where the LRM dynami-
cally triggers search queries based on self-assessed knowledge
gaps, and a Reason-in-Documents Module that processes re-
trieved content to dist ill relevant information into a refined
format, thereby minimizing noise and maintaining the LRM’s
reasoning integrity. Search-o1 exemplifies a sophisticated
prompt-based agentic approach focused on maintaining rea-
soning integrity in the face of external information retrieval.
4.2 Training-Based Approaches
While prompt-based methods leverage the inherent capabili-
ties of LLMs, their performance in complex tool-use scenarios
can be inconsistent. Achieving highly reliable and optimized
behavior, especially in deciding when and how to interact
with tools like search engines, often benefits from explicit
training. Training-based approaches, particularly those uti-
lizing Reinforcement Learning (RL), enable the LLM agent
to learn sophisticated strategies through trial and error, di-
rectly optimizing its actions towards specific goals such as
maximizing retrieval effectiveness or overall task success. RL
enables agents to develop more robust and strategic interactionpatterns than prompting alone.
Interacting with local retrieval systems : Search-R1 [Jinet
al., 2025 ]tackles a different aspect of agentic search: training
the LLM to autonomously decide when to search and what to
search for during a multi-step reasoning process. It extends
RL-based reasoning frameworks (like DeepSeek-R1) by in-
tegrating search engine interaction directly into the learning
loop. In the Search-R1 framework, the search engine is mod-
eled as part of the RL environment. The LLM agent learns a
policy to generate a sequence of tokens that includes both in-
ternal reasoning steps (often enclosed in <think> tags) and
explicit triggers for search actions. These triggers are special
tokens, <search> and</search> , which encapsulate the
generated search query. This design allows for flexible, multi-
turn interactions where the LLM can interleave reasoning,
searching, processing retrieved information (presented within
<information> tags), and further reasoning or searching
as needed. The framework utilizes a simple outcome-based
reward function, typically based on the correctness of the final
answer generated by the LLM (within <answer> tags) com-
pared to a ground truth, avoiding the complexity of designing
intermediate process rewards. A crucial technique employed
is retrieved token masking. During the calculation of the RL
loss (using algorithms like PPO or GRPO [Shao et al. , 2024 ]),
the tokens corresponding to the content retrieved from the
search engine (i.e., within the <information> tags) are
ignored or masked out, which stabilizes the training process.
Search-R1 has shown significant performance improvements
over various RAG baselines on question-answering datasets.
Its core contribution is training the LLM to learn an optimal
policy for interacting with the search engine as an integrated
part of its reasoning flow, enabling dynamic, context-aware
search decisions. The related R1-Searcher [Song et al. , 2025 ]
framework also proposes a similar two-stage, outcome-based

Original Question:Step1: … Step2: … Step3: ...Sub-Question
LRM
Retrieved DocumentsReason-in-Documents
Search for helpful info
Reasoning for next stepDistilled InformationStep nStep n+1Search Query
Step n+2...Final StepFinal Answer
iterableAgentic ReasoningFigure 6: A demonstration of Agentic Reasoning.
RL approach for enhancing search capabilities.
ReZero (Retry-Zero) [Dao and Le, 2025 ]introduces another
dimension to RL-based agentic search by specifically focusing
on incentivizing persistence. It addresses the common sce-
nario where an initial search query might fail to retrieve the
necessary information, potentially causing the LLM agent to
halt prematurely or generate a suboptimal response. ReZero
aims to teach the agent the value of “trying one more time.”
The framework operates within a standard RL setup (using
GRPO is mentioned) where the LLM interacts with a search
environment. The novelty lies in its modified reward func-
tion, which includes a specific component termed reward retry.
This component provides a positive reward signal whenever
the LLM issues a <search> query after the initial search
query within the same reasoning trajectory. Crucially, this
reward for retrying is conditional upon the agent successfully
completing the task, indicated by generating a final answer
enclosed in <answer> tags. This conditionality prevents the
agent from accumulating rewards simply by retrying indef-
initely without making progress. By directly rewarding the
act of persistence (when productive), ReZero encourages the
LLM to explore alternative queries or search strategies if the
first attempt proves insufficient. This contrasts with methods
that might only implicitly reward persistence through eventual
task success. ReZero positions itself as complementary to
frameworks like DeepRetrieval; while DeepRetrieval focuses
on optimizing a single refined query, ReZero emphasizes the
value of making multiple retrieval attempts when needed.
Interacting with real-world search engines : DeepRe-trieval [Jiang et al. , 2025 ]focuses specifically on improv-
ing the quality of the search queries generated by the LLM
agent. It frames the task of query generation or rewriting as
an RL problem, training the LLM to transform an initial user
query into a more effective query for downstream retrieval
systems. The core mechanism involves the LLM generating
an augmented or rewritten query based on the input query.
DeepRetrieval employs RL algorithms like Proximal Policy
Optimization (PPO) [Schulman et al. , 2017 ]to train this query
generation process. A key innovation lies in its reward signal:
instead of relying on supervised data (e.g., pairs of original and
”gold” rewritten queries), DeepRetrieval uses the performance
of the generated query in the actual retrieval system as the
reward. Metrics such as recall@k, Normalized Discounted Cu-
mulative Gain (NDCG), or evidence-seeking retrieval accuracy
(Hits@N) obtained from executing the generated query against
a restricted real search engine (like PubMed) or document col-
lection are used to provide feedback to the LLM. The model
learns, through trial and error, to generate queries that maxi-
mize these retrieval metrics. To structure the generation, the
model often produces reasoning steps within <think> tags
before outputting the final query in an <answer> tag. This
approach offers significant advantages. By directly optimizing
for the end goal (retrieval performance), it bypasses the need
for expensive and potentially suboptimal supervised query
datasets. Compared to other RL methods, DeepRetrieval’s
primary focus is on optimizing the content and formulation of
the search query itself.
DeepResearcher [Zheng et al. , 2025 ]pushes the boundaries
of training-based Agentic RAG by moving beyond controlled
environments or static corpora to perform end-to-end RL train-
ing directly within real-world web environments. It aims
to equip LLM agents with the capabilities needed for com-
plex, deep research tasks that require navigating the noisy,
unstructured, and dynamic nature of the open web. This ad-
dresses a key limitation of many existing agents, whether
prompt-engineered or trained in simulated/static RAG set-
tings, which often struggle with the complexities of real-world
web interaction. The framework employs RL (specifically
GRPO with an F1 score-based reward for answer accuracy
) to train agents that interact with live web search APIs and
browse actual webpages. DeepResearcher utilizes a special-
ized multi-agent architecture to handle the complexities of
web interaction. This includes a reasoning module, a tool
for invoking web search, and dedicated “browsing agents”
responsible for extracting relevant information from the di-
verse structures of webpages encountered. Training in this
realistic setting was found to foster several emergent cognitive
behaviors not typically observed in agents trained under more
constrained conditions. These include the ability to formulate
initial plans and dynamically adjust them during the research
process, cross-validate information retrieved from multiple
web sources, engage in self-reflection when retrieved infor-
mation seems contradictory or insufficient leading to refined
search strategies, and exhibit honesty by declining to provide
an answer when definitive information cannot be found. Deep-
Researcher demonstrated substantial performance improve-
ments over prompt-engineering baselines and RAG-based RL
agents trained on static corpora, particularly on open-domain

research tasks. The results strongly suggest that end-to-end
training in realistic web environments is crucial for develop-
ing robust and capable research agents, moving closer to the
capabilities hinted at by proprietary systems like OpenAI’s
Deep Research [OpenAI, 2025 ]or Grok’s DeeperSearch.
The progression for the training-based methods, from op-
timizing the decision process of when and what to query
(Search-R1), to fostering persistence (ReZero), optimizing
query formulation (DeepRetrieval), and managing real-world
research workflows (DeepResearcher) reflects the growing
sophistication of RL in agentic search. It reflects a growing
appreciation that effective information seeking by an agent
involves a confluence of factors: query quality, strategic tim-
ing, resilience to failure, and adeptness in navigating realistic
information environments and so on. Future advancements
in RL-based Agentic RAG will likely need to integrate these
facets more holistically, perhaps through more complex re-
ward structures, multi-objective optimization, or architectures
that explicitly model these different dimensions of the search
process, to achieve truly human-like research and problem-
solving capabilities.
5 Future Research Directions
Enhancing tool interaction through advanced configura-
tion. Current agentic reasoning often utilizes search tools with
relatively basic interfaces, primarily focused on generating
text queries. Future work should enable agents to exploit more
advanced configurations offered by external APIs and tools.
This could involve training agents to understand and utilize
options like result filtering (e.g., by date, source type), sorting
criteria, specifying search domains, or interacting with struc-
tured databases via complex queries. Granting finer control
would support more targeted, efficient, and strategic retrieval
aligned with task demands.
Developing finer-Grained and process-oriented reward
functions. Simple outcome-based rewards like exact match
may not offer adequate guidance for complex RAG tasks that
require multi-step reasoning or detailed responses. Future
research should develop fine-grained reward functions that
assess both final answer correctness and intermediate steps
such as document relevance, reasoning coherence, information
cross-validation, and effective problem decomposition. These
signals are vital for training agents to handle queries that
demand more than short factual answers.
Improving Efficiency in Retrieval. The approaches men-
tioned above primarily focus on the accuracy of the final an-
swer, but enhancing the efficiency of the retrieval process itself
is also critical. Agents trained to interact with potentially vast
information sources, must learn to perform retrievals strate-
gically. Future research should focus on techniques that help
agents avoid excessive or unnecessary search queries, select
the most promising sources, and know when sufficient infor-
mation has been gathered. Developing strategies to prevent
agents from getting stuck in loops of unproductive searching
or performing redundant retrievals is vital for practical and
scalable Agentic RAG.
Enhancing Generalization and Robustness in Dynamic
Environments. Robust generalization to new queries, un-seen tools (e.g., sparse, dense, or web retrieval), and changing
environments remains a major challenge. While training in re-
alistic conditions (as in DeepResearcher) improves resilience,
agents still struggle with tool failures or shifting knowledge
availability. Future work should explore adaptive training
methodologies and architectures that ensure robust perfor-
mance in unfamiliar or dynamic settings.
By addressing key areas such as improving agent control
over tools, designing more sophisticated reward signals, in-
creasing efficiency, and enhancing generalization, the field
can move toward building more capable, reliable, and widely
applicable Agentic RAG systems. These advancements are
essential for transitioning agentic AI from research prototypes
to practical systems that can effectively support humans in
complex information tasks.
6 Conclusions
As language models are increasingly deployed in complex,
knowledge-intensive applications, the limitations of static
RAG pipelines have become apparent. Reasoning Agentic
RAG offers a promising path forward by integrating retrieval
with model-driven planning, self-reflection, and tool use. This
paper surveyed the landscape of reasoning workflows within
Agentic RAG, distinguishing between predefined reasoning
with fixed orchestration, and agentic reasoning that enables
dynamic, autonomous decision-making. We reviewed key
methods across both paradigms, highlighting their strengths,
limitations, and use-case applicability. To advance the field,
we identify several crucial directions for future research, in-
cluding fine-grained reward design, enhanced tool control,
automated data synthesis, and robust training in dynamic en-
vironments. These innovations will be essential for realizing
intelligent, context-aware RAG systems capable of addressing
real-world challenges with greater adaptability, transparency,
and reliability.
References
[Asai et al. , 2023 ]Akari Asai, Zeqiu Wu, et al. Self-rag:
Learning to retrieve, generate, and critique through self-
reflection, 2023.
[Chen et al. , 2024 ]Jiawei Chen, Hongyu Lin, et al. Bench-
marking large language models in retrieval-augmented gen-
eration. In Proceedings of the AAAI Conference on Artifi-
cial Intelligence , volume 38, pages 17754–17762, 2024.
[Dao and Le, 2025 ]Alan Dao and Thinh Le. Rezero: En-
hancing llm search ability by trying one-more-time, 2025.
[DeepSeek-AI et al. , 2025 ]DeepSeek-AI, Daya Guo, et al.
Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning, 2025.
[Eleti et al. , 2023 ]Atty Eleti, Jeff Harris, et al. Function call-
ing and other api updates, June 2023.
[Gao et al. , 2023 ]Yunfan Gao, Yun Xiong, et al. Retrieval-
augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2:1, 2023.
[Gao et al. , 2024 ]Yunfan Gao, Yun Xiong, et al. Modular
rag: Transforming rag systems into lego-like reconfigurable
frameworks, 2024.

[Huet al. , 2025 ]Yunhai Hu, Yilun Zhao, et al. Mcts-rag: En-
hancing retrieval-augmented generation with monte carlo
tree search, 2025.
[Huang et al. , 2025 ]Lei Huang, Weijiang Yu, et al. A survey
on hallucination in large language models: Principles, tax-
onomy, challenges, and open questions. ACM Transactions
on Information Systems , 43(2):1–55, 2025.
[Jeong et al. , 2024 ]Soyeong Jeong, Jinheon Baek, et al.
Adaptive-rag: Learning to adapt retrieval-augmented large
language models through question complexity, 2024.
[Jiang et al. , 2025 ]Pengcheng Jiang, Jiacheng Lin, et al.
Deepretrieval: Hacking real search engines and retriev-
ers with large language models via reinforcement learning,
2025.
[Jinet al. , 2025 ]Bowen Jin, Hansi Zeng, et al. Search-r1:
Training llms to reason and leverage search engines with
reinforcement learning, 2025.
[Kong et al. , 2024 ]Yilun Kong, Jingqing Ruan, et al. Tptu-
v2: Boosting task planning and tool usage of large language
model-based agents in real-world industry systems. In
Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing: Industry Track , pages
371–385, 2024.
[Lewis et al. , 2020 ]Patrick Lewis, Ethan Perez, et al.
Retrieval-augmented generation for knowledge-intensive
nlp tasks. Advances in neural information processing sys-
tems, 33:9459–9474, 2020.
[Liet al. , 2024a ]Jiarui Li, Ye Yuan, et al. Enhancing llm
factual accuracy with rag to counter hallucinations: A case
study on domain-specific queries in private knowledge-
bases. arXiv preprint arXiv:2403.10446 , 2024.
[Liet al. , 2024b ]Zhuowan Li, Cheng Li, et al. Retrieval aug-
mented generation or long-context llms? a comprehensive
study and hybrid approach, 2024.
[Liet al. , 2025a ]Xiaoxi Li, Guanting Dong, et al. Search-o1:
Agentic search-enhanced large reasoning models, 2025.
[Liet al. , 2025b ]Zhong-Zhi Li, Duzhen Zhang, et al. From
system 1 to system 2: A survey of reasoning large language
models. arXiv preprint arXiv:2502.17419 , 2025.
[Linet al. , 2023a ]Weizhe Lin, Jinghong Chen, et al. Fine-
grained late-interaction multi-modal retrieval for retrieval
augmented visual question answering. Advances in Neural
Information Processing Systems , 36:22820–22840, 2023.
[Linet al. , 2023b ]Xi Victoria Lin, Xilun Chen, et al. Ra-
dit: Retrieval-augmented dual instruction tuning. In The
Twelfth International Conference on Learning Representa-
tions , 2023.
[Maet al. , 2023 ]Xinbei Ma, Yeyun Gong, et al. Query rewrit-
ing in retrieval-augmented large language models. In Pro-
ceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 5303–5315, 2023.
[Maet al. , 2024 ]Zi-Ao Ma, Tian Lan, et al. Multi-modal
retrieval augmented multi-modal generation: A bench-
mark, evaluate metrics and strong baselines. arXiv preprint
arXiv:2411.16365 , 2024.[OpenAI et al. , 2024 ]OpenAI, :, et al. Openai o1 system
card, 2024.
[OpenAI, 2025 ]OpenAI. Deep research system card, Febru-
ary 2025.
[Press et al. , 2023 ]Ofir Press, Muru Zhang, et al. Measuring
and narrowing the compositionality gap in language models,
2023.
[Ravuru et al. , 2024 ]Chidaksh Ravuru, Sagar Srinivas Sakhi-
nana, et al. Agentic retrieval-augmented generation for time
series analysis. arXiv preprint arXiv:2408.14484 , 2024.
[Rawte et al. , 2023 ]Vipula Rawte, Swagata Chakraborty,
et al. The troubling emergence of hallucination in large lan-
guage models-an extensive definition, quantification, and
prescriptive remediations. Association for Computational
Linguistics, 2023.
[Robertson et al. , 2009 ]Stephen Robertson, Hugo Zaragoza,
et al. The probabilistic relevance framework: Bm25 and
beyond. Foundations and Trends ®in Information Retrieval ,
3(4):333–389, 2009.
[Ruan et al. , 2023 ]Jingqing Ruan, Yihong Chen, et al. Tptu:
Task planning and tool usage of large language model-
based ai agents. In NeurIPS 2023 Foundation Models for
Decision Making Workshop , 2023.
[Sarthi et al. , 2024 ]Parth Sarthi, Salman Abdullah, et al. Rap-
tor: Recursive abstractive processing for tree-organized
retrieval, 2024.
[Schulman et al. , 2017 ]John Schulman, Filip Wolski, et al.
Proximal policy optimization algorithms, 2017.
[Shao et al. , 2024 ]Zhihong Shao, Peiyi Wang, et al.
Deepseekmath: Pushing the limits of mathematical rea-
soning in open language models, 2024.
[Singh, 2023 ]Aditi Singh. Exploring language models: A
comprehensive survey and analysis. In 2023 International
Conference on Research Methodologies in Knowledge Man-
agement, Artificial Intelligence and Telecommunication En-
gineering (RMKMATE) , pages 1–4. IEEE, 2023.
[Song et al. , 2025 ]Huatong Song, Jinhao Jiang, et al. R1-
searcher: Incentivizing the search capability in llms via
reinforcement learning, 2025.
[Wang et al. , 2024a ]Xi Wang, Procheta Sen, et al. Adaptive
retrieval-augmented generation for conversational systems,
2024.
[Wang et al. , 2024b ]Yu Wang, Nedim Lipka, et al. Knowl-
edge graph prompting for multi-document question answer-
ing. In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 19206–19214, 2024.
[Wang et al. , 2025 ]Han Wang, Archiki Prasad, et al.
Retrieval-augmented generation with conflicting evidence.
arXiv preprint arXiv:2504.13079 , 2025.
[Weiet al. , 2023 ]Jason Wei, Xuezhi Wang, et al. Chain-
of-thought prompting elicits reasoning in large language
models, 2023.
[Yanet al. , 2024 ]Shi-Qi Yan, Jia-Chen Gu, et al. Corrective
retrieval augmented generation, 2024.

[Yang et al. , 2024a ]Cheng Yang, Chufan Shi, et al. Llm2:
Let large language models harness system 2 reasoning.
arXiv preprint arXiv:2412.20372 , 2024.
[Yang et al. , 2024b ]Xiao Yang, Kai Sun, et al. Crag-
comprehensive rag benchmark. Advances in Neural In-
formation Processing Systems , 37:10470–10490, 2024.
[Yaoet al. , 2023 ]Shunyu Yao, Jeffrey Zhao, et al. React:
Synergizing reasoning and acting in language models,
2023.
[Yuet al. , 2024 ]Shi Yu, Chaoyue Tang, et al. Visrag: Vision-
based retrieval-augmented generation on multi-modality
documents. arXiv preprint arXiv:2410.10594 , 2024.
[Yuet al. , 2025 ]Qinhan Yu, Zhiyou Xiao, et al. Mramg-
bench: A beyondtext benchmark for multimodal retrieval-
augmented multimodal generation. arXiv preprint
arXiv:2502.04176 , 2025.
[Zhang et al. ,]Bin Zhang, Hangyu Mao, et al. Controlling
large language model-based agents for large-scale decision-
making: An actor-critic approach. In ICLR 2024 Workshop
on Large Language Model (LLM) Agents .
[Zhang et al. , 2023 ]Yue Zhang, Yafu Li, et al. Siren’s song
in the ai ocean: a survey on hallucination in large language
models. arXiv preprint arXiv:2309.01219 , 2023.
[Zhao et al. , 2023 ]Wayne Xin Zhao, Kun Zhou, et al.
A survey of large language models. arXiv preprint
arXiv:2303.18223 , 1(2), 2023.
[Zhao et al. , 2024 ]Penghao Zhao, Hailin Zhang, et al.
Retrieval-augmented generation for ai-generated content:
A survey. arXiv preprint arXiv:2402.19473 , 2024.
[Zheng et al. , 2025 ]Yuxiang Zheng, Dayuan Fu, et al. Deep-
researcher: Scaling deep research via reinforcement learn-
ing in real-world environments, 2025.
[Zhuet al. , 2024 ]Yizhang Zhu, Shiyin Du, et al. Are large
language models good statisticians? arXiv preprint
arXiv:2406.07815 , 2024.