# Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs

**Authors**: Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, Essam Mansour

**Published**: 2025-11-26 00:18:55

**PDF URL**: [https://arxiv.org/pdf/2511.20940v1](https://arxiv.org/pdf/2511.20940v1)

## Abstract
Conversational Question Answering over Knowledge Graphs (KGs) combines the factual grounding of KG-based QA with the interactive nature of dialogue systems. KGs are widely used in enterprise and domain applications to provide structured, evolving, and reliable knowledge. Large language models (LLMs) enable natural and context-aware conversations, but lack direct access to private and dynamic KGs. Retrieval-augmented generation (RAG) systems can retrieve graph content but often serialize structure, struggle with multi-turn context, and require heavy indexing. Traditional KGQA systems preserve structure but typically support only single-turn QA, incur high latency, and struggle with coreference and context tracking. To address these limitations, we propose Chatty-KG, a modular multi-agent system for conversational QA over KGs. Chatty-KG combines RAG-style retrieval with structured execution by generating SPARQL queries through task-specialized LLM agents. These agents collaborate for contextual interpretation, dialogue tracking, entity and relation linking, and efficient query planning, enabling accurate and low-latency translation of natural questions into executable queries. Experiments on large and diverse KGs show that Chatty-KG significantly outperforms state-of-the-art baselines in both single-turn and multi-turn settings, achieving higher F1 and P@1 scores. Its modular design preserves dialogue coherence and supports evolving KGs without fine-tuning or pre-processing. Evaluations with commercial (e.g., GPT-4o, Gemini-2.0) and open-weight (e.g., Phi-4, Gemma 3) LLMs confirm broad compatibility and stable performance. Overall, Chatty-KG unifies conversational flexibility with structured KG grounding, offering a scalable and extensible approach for reliable multi-turn KGQA.

## Full Text


<!-- PDF content starts -->

Chatty-KG: A Multi-Agent AI System for On-Demand
Conversational Question Answering over Knowledge Graphs
Reham Omar
Concordia University
CanadaAbdelghny Orogat
Concordia University
CanadaIbrahim Abdelaziz
IBM Research
USA
Omij Mangukiya
Concordia University
CanadaPanos Kalnis
KAUST
Saudi ArabiaEssam Mansour
Concordia University
Canada
Abstract
Conversational Question Answering over Knowledge Graphs (KGs)
combines the factual grounding of KG-based QA with the interac-
tive nature of dialogue systems. KGs are widely used in enterprise
and domain applications to provide structured, evolving, and reli-
able knowledge. Large language models (LLMs) enable natural and
context-aware conversations, but lack direct access to private and
dynamic KGs. Retrieval-augmented generation (RAG) systems can
retrieve graph content but often serialize structure, struggle with
multi-turn context, and require heavy indexing. Traditional KGQA
systems preserve structure but typically support only single-turn
QA, incur high latency, and struggle with coreference and context
tracking. To address these limitations, we proposeChatty-KG,
a modular multi-agent system for conversational QA over KGs.
Chatty-KGcombines RAG-style retrieval with structured execu-
tion by generating SPARQL queries through task-specialized LLM
agents. These agents collaborate for contextual interpretation, di-
alogue tracking, entity and relation linking, and efficient query
planning, enabling accurate and low-latency translation of natural
questions into executable queries.Experiments on large and diverse
KGs show thatChatty-KGsignificantly outperforms state-of-the-
art baselines in both single-turn and multi-turn settings, achieving
higher F1 and P@1 scores. Its modular design preserves dialogue
coherence and supports evolving KGs without fine-tuning or pre-
processing. Evaluations with commercial (e.g., GPT-4o, Gemini-
2.0) and open-weight (e.g., Phi-4, Gemma 3) LLMs confirm broad
compatibility and stable performance. Overall,Chatty-KGunifies
conversational flexibility with structured KG grounding, offering
a scalable and extensible approach for reliable multi-turn KGQA.
ACM Reference Format:
Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya,
Panos Kalnis, and Essam Mansour. 2025.Chatty-KG: A Multi-Agent AI
System for On-Demand Conversational Question Answering over Knowl-
edge Graphs. In.ACM, New York, NY, USA, 15 pages. https://doi.org/10.
1145/nnnnnnn.nnnnnnn
1 Introduction
Conversational Question Answering over Knowledge Graphs (KGs)
aims to combine the factual accuracy and structured reasoning
of KG-based question answering with the natural, user-friendly
Conference’17, July 2017, Washington, DC, USA
2025. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
  
Who 
is 
the 
author 
of 
Harry 
Potter 
?
Conversational
Systems
Dialogue
KGQA 
Systems
Question 
Understanding
Intermediate
Understanding
N 
SPARQL
Queries
Chatbots
Q
Poor 
Contextual 
Understanding
Linking 
& 
Execution
KG 
Engine
A0
+
A1
+
+
AN
...
High 
Latency 
& 
Low 
Precision
A
Entity: 
Harry 
Potter
 
Model 
Training 
Required
Preprocessing 
Overhead
 
J.K. 
Rowling
  
When 
was 
its 
first 
movie 
released?
2001
Who 
played 
the 
main 
character?
Daniel 
Radcliffe
When 
was 
he 
born?
July 
23, 
1989
What 
else 
has 
he 
starred 
in?
Wait 
...Figure 1: Limitations of current KGQA and chatbot-based
systems for real-time conversational access to arbitrary KGs.
KGQA systems face issues with contextual understanding,
query fragmentation, and latency. Chatbots offer better con-
versational QA over KGs but require expensive training and
preprocessing, limiting adaptability and scalability.
experience of dialogue systems. KGs are widely adopted in domain-
specific and enterprise applications to integrate heterogeneous data
sources and provide reliable and up-to-date knowledge. Examples
include general KGs, such as Wikidata1, biomedical graphs like
BioRDF2and UMLS3, academic KGs, such as DBLP4, and legal
or financial graphs like FinDKG5. Despite their strengths, interact-
ing with KGs typically requires formal query languages, such as
SPARQL or Cypher, creating a steep learning curve for non-expert
users. This limits the deployment of conversational agents that can
reliably answer questions over these knowledge sources.
To make KGs more accessible, Knowledge Graph Question An-
swering (KGQA) systems enable users to ask questions in natural
language. However, most systems rely on domain-specific pipelines,
require heavy KG preprocessing (e.g., EDGQA [ 15]), and depend on
finetuned models (e.g., KGQAn [ 30]), which limits their scalability
and adaptability. These systems are typically designed for single-
turn questions and often suffer from high latency. Such limitations
make them unsuitable for multi-turn conversations that require
context tracking and efficient processing. As shown in Figure 1,
they also struggle with contextual understanding and multi-query
1https://www.wikidata.org/
2https://bio2rdf.org/
3https://www.nlm.nih.gov/research/umls
4https://dblp.org/
5https://xiaohui-victor-li.github.io/FinDKG/arXiv:2511.20940v1  [cs.CL]  26 Nov 2025

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
coordination, resulting in low accuracy and slow response times,
as demonstrated by KGQAn [30] and EDGQA [15].
Recent advancements in large language models (LLMs), such
as GPT-4 [ 32] and Gemini [ 42], have greatly improved natural
language understanding and reasoning. One might expect retrieval-
augmented generation (RAG) to remove the need for explicit KG
access by retrieving a subgraph and using it as context for the LLM.
However, there is a gap between the RAGparadigmand its current
implementations. In practice, graph-RAG systems such as Microsoft
GraphRAG [ 13] must first interpret the question and extract entities
before retrieval, but retrieval itself depends on heavy graph index-
ing and embedding stores that consume large memory and do not
scale to large KGs. The retrieved subgraphs are then serialized into
text, which discards structural information and weakens answer pre-
cision, especially for multi-entity or list-style queries. These systems
may lack dialogue support, preventing multi-turn refinement or con-
text reuse. As illustrated in Figure 1, chatbot-based KG access (e.g.,
CONVINSE [ 5], EXPLAIGNN [ 6]) requires retraining or KG-specific
preprocessing to remain accurate, making it costly and inflexible for
evolving graphs. These limitations show that RAG-as-a-paradigm is
promising, but current graph-RAG implementations fail to deliver
accurate, structure-aware, and dialogue-capable KG access.
To address these limitations, we introduceChatty-KG, a mod-
ular multi-agent framework for conversational KGQA. The key
technical novelty lies in combining RAG-as-a-paradigm with struc-
tured query execution through a training-free multi-agent design
that requires no KG-specific preprocessing and generalizes across
arbitrary graphs. Instead of asking an LLM to infer an answer from
serialized triples,Chatty-KGuses LLM agents to generate SPARQL
queries over the KG, preserving graph semantics and reducing hal-
lucinations. The pipeline is decomposed into lightweight agents
for question understanding, dialogue context tracking, entity link-
ing, and query generation, coordinated through a controller for
low-latency execution.Chatty-KGleverages recent LLM advances
together with orchestration frameworks like LangGraph6while
avoiding costly retraining. This design enables real-time, verifi-
able access toevolving KGsand supports both single-turn and
multi-turn dialogues with minimal overhead.
Our extensive evaluation confirms thatChatty-KGachieves
outstanding performance across five real-world and large KGs of di-
verse application domains. It consistently outperforms state-of-the-
art KGQA systems, including KGQAn [ 30] for single-turn questions,
RAG-based systems, such as GraphRAG [ 13] and ColBERT [ 21,37],
and chatbot systems, such as GPT-4o, Gemini-2.0, CONVINSE [ 5]
and EXPLAIGNN [ 6] for multi-turn dialogue.Chatty-KG’s mod-
ular design supports contextual reasoning and coherent dialogue
without domain tuning. Experiments across a wide range of com-
mercial (e.g., GPT-4o, Gemini-2.0) and open-weight LLMs (e.g.,
Phi-4, Gemma-3) demonstrate strong compatibility and stable per-
formance. In general,Chatty-KGoffers a scalable, extensible, and
reliable framework for conversational KGQA in real time with sig-
nificant improvements in accuracy, efficiency, and adaptability.
In summary, our contributions are:
6https://python.langchain.com/docs/langgraph/•A modular multi-agent architecture for conversational KGQA,
where individual agents can be swapped or enhanced inde-
pendently, supporting future extensibility.
•Supports single- and multi-turn conversations with LLM-
powered context tracking and disambiguation, while main-
taining low latency for interactive use.
•Efficient SPARQL-based KG access via a query planning
agent that eliminates the need for offline preprocessing.
•A comprehensive evaluation across five real KGs from di-
verse domains demonstratesChatty-KG’s superior perfor-
mance in answer correctness and response time compared
to state-of-the-art systems, with consistent results across
commercial and open-weight LLMs.
2 Background and Related Work
This section surveys related work on KGQA, conversational models,
and LLM-based agent frameworks. We outline the challenges of
integrating conversational capabilities with structured knowledge
and motivate our training-free and LLM-driven approach.
2.1 Knowledge Graph Question Answering
Conversational question answering over KGs typically requires
three main steps: (i) Dialogue Understanding, which creates an ab-
stract representation from the key entities and relations extracted
from the question and conversation history; (ii) Linking, which
maps the abstract representation to vertices and predicates from
the KG; and (iii) Answer Generation, which uses the abstract rep-
resentation and the mapped KG vertices and predicates to either
extract the answer from the KG or generate it using a trained model.
In this section, we discuss the two categories of systems summa-
rized in Table 1: KGQA; e.g., [ 15,30], and Conversational Systems;
e.g., [5, 6] and how it compares to our approach.
KGQA systemsrely on generating SPARQL queries that rep-
resent an individual question. Examples include gAnswer [ 14,54],
NSQA [ 20], and WDAqua [ 10]. The top two performing systems,
KGQAn [ 30] and EDGQA [ 15], are examined in greater detail in this
section to better understand their design choices, strengths, and
limitations. KGQAn is the state-of-the-art across multiple datasets
[30], including QALD-9 [ 44], while EDGQA achieves strong results
on the LC-QuAD 1.0 dataset [ 15,43]. However, these systems can-
not handle conversational interactions as they are designed for
single-turn and standalone questions. KGQAn’s pretrained models
are limited by their training on single-turn queries and a relatively
small context window length that cannot fit conversation history.
Similarly, EDGQA’s rule-based approach does not scale to multi-
turn conversations. Both KGQAn and EDGQA have high response
times, making them unfit for conversational use. KGQAn’s latency
is due to repeated similarity-server requests, numerous SPARQL
queries execution for high recall, and an answer-type filtering step.
EDGQA also runs multiple SPARQL queries and uses three different
systems for linking, contributing to its delay [30].
For question understanding, KGQAn fine-tunes an LLM to gen-
erate triples from questions, requiring a manually created dataset
that is time-consuming and subject to annotator bias. It also trains a
separate answer type detection model. In contrast, EDGQA uses the
Stanford CoreNLP parser with rules tailored to a specific dataset,

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
Table 1: Comparison of existing KGQA and conversational QA systems across key design aspects. KGQA systems lack multi-
turn dialogue support and suffer from high latency, while conversational systems depend on fine-tuning and KG-specific
preprocessing.Chatty-KGovercomes these limitations with a LLM-powered multi-agent approach for real-time performance.
System Dialogue Response Time Question Understanding Linking Training Pre-processing
KGQAn[30] x high Fine-tuning Semantic similarity based✓x
EDGQA[15] x high Constituency parsing Index-based x✓
CONVINSE[5]✓low Fine-tuning Index-based✓ ✓
EXPLAIGNN[6]✓low Fine-tuning Index-based✓ ✓
Chatty-KG✓low LLM-Powered LLM-Powered x x
limiting cross-domain applicability. For linking, KGQAn makes
hundreds of similarity-server calls, slowing response time, while
EDGQA uses three indexing systems (Falcon [ 36], EARL [ 12], Dex-
ter [3]), requiring heavy pre-processing and reducing adaptability
to new knowledge graphs. Overall, KGQAn demands extensive
model training, while EDGQA relies on significant pre-processing.
Conversational systemsrely on information retrieval tech-
niques to extract or generate answers. CONVINSE [ 5] and EX-
PLAIGNN [ 6] are end-to-end conversational question-answering
systems on heterogeneous data sources, including KGs, text, and
tables. These systems, which only support the Wikidata KG [ 46],
handle conversational interactions with low response times, mak-
ing them suitable for real-time user interaction. For question un-
derstanding, they transform questions into intent-explicit struc-
tured representations (SR) using fine-tuned transformer models
[45] (BART [ 25] and T5 [ 33]). CONVINSE and EXPLAIGNN for-
malize linking as evidence retrieval and scoring. They use Clocq
[4] for named entity disambiguation and KG-item retrieval. Clocq
requires KG pre-processing before KG facts retrieval. For answer
generation, CONVINSE trains a Fusion-in-Decoder (FiD) model [ 17]
to generate answers based on top-ranked evidence. EXPLAIGNN
constructs a heterogeneous answering graph from retrieved entities
and evidence. Both systems require model training for question
understanding and answer generation. They also require KG pre-
processing, which limits their adaptability to new KGs.
In summary, existing systems face key limitations that hinder
effective conversational question answering over diverse and evolv-
ing KGs. Traditional KGQA systems lack support for multi-turn
dialogue and suffer from high latency. Conversely, conversational
systems improve interactivity and speed but depend heavily on fine-
tuning and KG-specific pre-processing, reducing flexibility across
domains. In contrast,Chatty-KGintroduces a modular, low-latency
conversational framework that eliminates the need for training or
pre-processing. It leverages LLMs for dynamic dialogue tracking,
question understanding, and entity linking to enable scalable and
adaptable QA over arbitrary knowledge graphs.
2.2 LLM Prompting Strategies
LLMs such as GPT-4, Gemini, and Qwen [ 51] have shown strong
language understanding and generation skills across diverse tasks.
Their generalization without task-specific training makes them
especially valuable in modular systems likeChatty-KG, where
they serve as interchangeable agents for reasoning, classification,
or generation. Prompt engineering has emerged as a lightweight,
cost-effective alternative to fine-tuning by steering LLMs throughcarefully crafted instructions [ 40]. Common prompting strategies
include: 1) zero-shot prompting with only a task description is pro-
vided to the LLM, 2) few-shot prompting which adds to the LLM
a handful of examples , and 3) chain-of-thought (CoT) prompting,
which guides intermediate reasoning steps before producing an
answer [ 22,49]. The best strategy depends on the task: classifica-
tion or retrieval often works well with zero-shot prompts, while
generation or complex reasoning benefits from few-shot or CoT
prompting. InChatty-KG, simpler agents (e.g., for classification
or routing) use zero-shot prompts for speed, while agents handling
contextual understanding or answer synthesis apply few-shot or
CoT prompting to support multi-step reasoning. This flexibility op-
timizes both efficiency and quality across the multi-agent pipeline.
2.3 LLM Agents
LLM agents are emerging systems that use the reasoning, planning,
and language capabilities of large language models to autonomously
complete tasks. At their core, these agents treat the LLM as a central
decision-maker that interacts with tools, APIs, and environments
in a loop of observation, reasoning, and action. Recent frameworks,
such as ReAct [ 53], AutoGPT [ 39], and BabyAGI [ 29], combine LLMs
with tool use and planning mechanisms. More structured frame-
works like LangChain, LlamaIndex7, and OpenAgents [ 50] provide
abstractions for integrating tools, retrievers, memory, and control
flows. LangChain enables modular pipelines that compose LLMs
with APIs, databases, and reasoning components. LangGraph8, a re-
cent LangChain extension, introduces stateful graph-based control
for defining multi-agent workflows with dynamic routing.
A key enabler of these systems is tool calling (also known as func-
tion calling), where the LLM invokes external tools at inference time
to offload specific tasks, such as computation, API queries, or struc-
tured data access. Tool calling has become central to modern agent
frameworks, allowing models to extend beyond generation and act
as orchestrators. In our system, agents selectively call tools based
on their roles, for example, to resolve entity mentions, fetch KG sub-
graphs, or execute SPARQL queries. This targeted, task-specific in-
tegration enables agents to remain lightweight and model-agnostic,
while grounding their decisions in accurate, up-to-date KG content.
LLM agents are increasingly used in domains, such as web au-
tomation [ 52], code generation [ 48], and data wrangling [ 26]. In
these settings, they break down complex goals, retrieve intermedi-
ate results, and adapt to user feedback [ 47]. However, most systems
are tailored to unstructured or semi-structured data and struggle
7https://www.llamaindex.ai/
8https://python.langchain.com/docs/langgraph/

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
(qs,QIR)
Chat 
Agent
(q)
Contextual 
Understanding
(qs,QIR)
(FA)
(qs)
Query 
Generation 
and 
Answer 
Retrieval
QIR 
Agent
Classifier 
Agent
(q,Cq)
(q)
<e1, 
r, 
e2>
Triples 
Validator
Query
Execution
Query 
Validator
(A)
(EntToVertex
,
RelToPred)
RDF 
Engine
Rephraser 
Agent
Self/
non-Self
?
(QIR)
?
(q,Cq)
Legend
?
LLM
(A)
Vertex 
Validator(EntToVertex)
(E)
(RF)LLM
(RelToPred)Relation 
Linking
Entity 
Retriever
LLM
Dialogue 
D
Question:
 
q
1
Answer:
 
A
1
Question: 
q
2
Answer:
 
A
2
LLM
LLM
LLM
(D)
Agent 
= 
LLM 
+ 
Tools
Supervisor 
= 
Agent 
+ 
Coordination 
of 
Other 
Agents
Task 
Manager 
- 
LLM 
Powered
Tool 
= 
Function 
Call 
?
Repeat 
= 
Loop 
at 
most 
? 
times
(Q')
Query 
Planning 
Agent
Matching 
Agent
Query 
Generator
(Q)
(QIR,
EntToVertex
,
RelToPred)
Figure 2:Chatty-KG’s hierarchical multi-agent architecture. A top-levelChat Agentcoordinates two supervised modules:Con-
textual UnderstandingandQuery Generation & Answer Retrieval, each composed of specialized LLM-powered agents. This design
enables modular, low-latency KGQA without training or pre-processing, and improves adaptability and real-time performance.
with formal sources, such as relational databases or knowledge
graphs (KGs) [ 38]. Challenges in schema alignment, entity linking,
and precise query generation remain underexplored. Therefore,
the use of LLM agents for KG question answering (KGQA) is still
emerging. In this work, we demonstrate how core agent capabilities,
such as contextual understanding, linking, and modular reasoning,
can be applied to KGQA in a lightweight and efficient manner.
Our system avoids complex planning or orchestration. Instead, our
agents use prompt-guided LLM calls to interpret questions, link KG
elements, and generate a minimal set of executable queries.
3Chatty-KGArchitecture
Chatty-KGadopts a hierarchical design that splits the task into two
main subtasks: language-level processing and graph-level reason-
ing. Language-level tasks include question classification, ambiguity
resolution, and intermediate representation generation. Graph-level
tasks cover entity/relation linking and SPARQL query construc-
tion. Each subtask is handled by a cluster of LLM agents. Each
agent wraps atask-managed LLM, a language model guided by role-
specific logic and tools. Some agents act as supervisors, delegating
to subordinate agents. This modular and domain-agnostic architec-
ture supports accurate reasoning, flexible operation across KGs, and
efficient execution without retraining or KG-specific preprocessing.
The matching agent does not assume a fixed schema and instead
dynamically extracts relevant subgraphs using SPARQL queries.
Key stages incorporate validation and retry mechanisms to mitigate
error propagation. We first define the main representations.
Definition 3.1 (Dialogue).The dialogue history is defined as D=
{(𝑞 1,𝐴1),(𝑞 2,𝐴2),...,(𝑞𝑛−1,𝐴𝑛−1)}, where each 𝑞𝑖is a user-issued
question and 𝐴𝑖=[𝑎 1,𝑎2,...,𝑎𝑚]is the corresponding system-
generated answer. Each 𝑎𝑗may be a count, a Boolean value, or an
entity;𝐴𝑖may also be a list of such values.
Definition 3.2 (Question Intermediate Representation (QIR)).Given
a question𝑞that mentions entities, expresses relations, and targets
an unknown variable, potentially involving intermediate variables
for multi-hop reasoning, theQIRis defined as 𝑄𝐼𝑅=(𝐸,𝑈,𝑅,𝑅𝐹) .
Here,𝐸is the set of mentioned entities, 𝑈is the set of variables
(including the target and any intermediates), Ris the set of relation
phrases in𝑞, and𝑅𝐹is a set of relational facts. Each 𝑟𝑓∈𝑅𝐹 is a
triple⟨𝑒 1,𝑟,𝑒 2⟩, where𝑒 1,𝑒2∈(𝐸∪𝑈)and𝑟∈R.Chatty-KGis structured into three core modules:Contextual
Understanding,Chat Management, andQuery Generation and An-
swer Retrieval(Figure 2). Each module contains one or more agents
and tools with well-defined communication interfaces. The system
operates on a dialogue D, using its history to form the context 𝐶𝑞
for interpreting new user questions. Upon receiving a new question
𝑞𝑖, theChat Agentinitiates processing by forwarding 𝑞𝑖and𝐶𝑞
to theQIR Agent, which builds the intermediate representation
to guide query generation. To manage complexity, the QIR Agent
delegates two subtasks to specialized agents. First, theClassifier
Agentdetermines whether 𝑞𝑖isself-contained(interpretable alone)
orcontext-dependent(requiring prior turns). For example, “Who is
the author of Harry Potter?” is self-contained, while “When was
its first movie released?” depends on earlier context.
The classification ensures downstream agents receive clear and
unambiguous input. If the question is classified as context-dependent,
the QIR Agent forwards both 𝑞𝑖and𝐶𝑞to theRephraser Agent.
This agent reformulates the question into a self-contained version
using dialogue context. For example, it transforms “When was its
first movie released?” into “When was the first Harry Potter movie
released?”. The QIR Agent then processes the resulting question,
whether rephrased or original, to generate the Question Intermedi-
ate Representation (QIR). This involves extracting a set of triples
representing the question’s semantics.
Chatty-KGuses an assertion-based validation approach to en-
sure output correctness. Inspired by software assertions, it defines
task-specific structural and semantic checks that are independent of
input content. For example, theTriples Validatorchecks that each
extracted triple has valid components (subject, predicate, object). If
validation fails, the QIR Agent retries generation up to a threshold
𝜃. Once validated, the triples form the QIR. For instance, the QIR
for the rephrased question is {E: {"Harry Potter"}, U: {"?year"}, RF:
{⟨Harry Potter, released, ?year⟩}}.
TheQuery Planning Agentreceives the QIR and identifies a
minimal set of SPARQL queries to retrieve the answer. It begins
by delegating entity and relation linking to theMatching Agent,
which uses theEntity RetrieverandRelation Linkingtools to
map QIR elements to KG URIs. The resulting mappings are passed to
theVertex Validator, which checks that each entity is syntactically
correct and appears in the candidate list. If validation fails, the
system retries up to 𝜃times. After successful validation, the URIs
are returned to the planning agent for query construction.

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
TheQuery Generatorconstructs candidate SPARQL queries.
TheQuery Planning Agentselects a minimal subset for execution,
typically one to three queries, based on its understanding, without
relying on a fixed threshold. The selected queries are passed to the
Query Validator, which uses assertions to ensure syntactic and
semantic correctness, retrying on failure. This avoids executing
irrelevant queries and reduces overhead. The selected queries are
sent to theQuery Executionmodule to retrieve the raw answer
𝐴. The Chat Agent may rephrase it into a user-friendly response
𝐹𝐴, e.g., turning "2001" into “The first Harry Potter movie was
released in 2001.” The final answer is returned to the user, and the
dialogueDis updated to complete the interaction.
4 Chat Agent and Contextual Understanding
This section introduces two core components: theChat Agent,
which manages dialogue state and controls workflow, and theCon-
textual Understandingmodule, which transforms natural language
questions into a structured Question Intermediate Representation.
4.1 Chat Agent
TheChat Agentacts as the top-level controller inChatty-KG’s
multi-agent framework. It is responsible for: (i) orchestrating mod-
ule execution, (ii) maintaining dialogue history, and (iii) reformulat-
ing answers and enabling multilingual queries through LLM-based
translation.When a new user question 𝑞𝑛arrives, the Chat Agent
retrieves the dialogue history Dand constructs the corresponding
context𝐶𝑞needed to interpret the question. Since 𝐶𝑞is passed as in-
put to prompt-based LLM agents (e.g., the Rephraser Agent), it must
remain within the LLM’s token limit. We define 𝐶𝑞={(𝑞𝑖,𝐴𝐿
𝑖)}𝑛−1
𝑖=1,
where each 𝐴𝐿
𝑖is a truncated prefix of 𝐴𝑖, limited to𝐿items when
|𝐴𝑖|>𝐿. This preserves essential context while ensuring compact
prompts. The pair(𝑞𝑛,𝐶𝑞)is passed to the Contextual Understand-
ing module, which generates the structured Question Intermediate
Representation (QIR). The QIR is then forwarded to the Query
Generation and Answer Retrieval module, which constructs and
executes the corresponding SPARQL query and returns the raw
answer𝐴𝑛. Optionally, the Chat Agent reformulates 𝐴𝑛into a more
natural final answer𝐹𝐴 𝑛using a zero-shot prompting strategy:
𝐹𝐴𝑛=𝑓(𝐴𝑛,𝑞𝑛,𝐼𝐹𝐴),(1)
where𝐼𝐹𝐴9is a prompt instructing the LLM to express short lit-
eral answers in fluent natural language. We adopt zero-shot prompt-
ing due to the simplicity and consistency of the task, which typically
involves converting short outputs into complete sentences. Finally,
the Chat Agent appends (𝑞𝑛,𝐴𝑛)to the dialogue history and awaits
the next question. This process enables context-aware, multi-turn
interaction while maintaining low latency and prompt efficiency.
4.2 Contextual Understanding
To interpret a user question 𝑞𝑖in its dialogue context 𝐶𝑞, the Con-
textual Understanding module runs a multi-stage pipeline that
converts natural language into a structured Question Intermediate
Representation (QIR). This process is managed by theQIR Agent,
which coordinates two key subtasks handled by auxiliary agents:
9All prompts are available in the supplementary materials.Algorithm 1Contextual Understanding Pipeline
Input:𝑞𝑖: User question, 𝐶𝑞: Question context, 𝜃: Retry threshold,
𝐼𝐶: Classification instructions, 𝐼𝑅𝐹: Reformulation instructions, 𝐼𝑄𝑈:
Understanding instructions
Output:𝑄𝐼𝑅=(𝐸,𝑈,𝑅,𝑅𝐹): Intermediate Representation
1:𝑞 type←ClassifierAgent(𝑞 𝑖,𝐼𝐶)
2:if𝑞 type=‘Dependent’then
3:𝑞𝑖𝑠←RephraserAgent(𝑞 𝑖,𝐶𝑞,𝐼𝑅𝐹)
4:else
5:𝑞𝑖𝑠←𝑞𝑖
6:end if
7:valid←False
8:whilevalid=False∧𝜃>0do
9:𝑡𝑟𝑖𝑝𝑙𝑒𝑠←QIRAgent.GenerateTriples(𝑞 𝑖𝑠,𝐼𝑄𝑈)
10:valid←QIRAgent.ValidateTriples(𝑡𝑟𝑖𝑝𝑙𝑒𝑠)
11:𝜃←𝜃−1
12:end while
13:𝑄𝐼𝑅←QIRAgent.ConstructQIR(𝑡𝑟𝑖𝑝𝑙𝑒𝑠)
14:return𝑄𝐼𝑅
classification and reformulation. As shown in Algorithm 1, the
Classifier Agentfirst determines whether 𝑞𝑖is self-contained or
context-dependent. If the question depends on prior context, the
Rephraser Agentgenerates a standalone version 𝑞𝑖𝑠using𝐶𝑞to
resolve references and omissions. Otherwise,𝑞 𝑖𝑠←𝑞𝑖.
The unified question 𝑞𝑖𝑠is then processed by the QIR Agent to
extract semantic triples via prompt-guided LLM interaction. These
triples are validated for structure and meaning, and the agent retries
up to𝜃times if validation fails. Once validated, the resulting QIR is
returned to the Chat Agent for downstream processing. Separating
these subtasks improves LLM performance by reducing prompt
complexity and avoiding instruction overload, as each agent uses a
dedicated session and objective-specific prompt.
4.2.1 Classifier Agent.The Classifier Agent identifies whether a
question isself-containedorcontext-dependentby detecting unre-
solved references or pronouns that imply reliance on earlier dia-
logue turns. It uses a zero-shot prompting strategy with a natural-
language instruction 𝐼𝐶that explains the classification criteria and
guides the decision. We avoid using traditional machine learning
classifiers, which require labeled training data that is costly and of-
ten unavailable. Moreover, this task goes beyond feature-level clas-
sification such as pronoun presence. For instance, “When was the
first movie released?” appears complete but is context-dependent if
the subject entity was mentioned earlier. The LLM-based approach
enables semantic reasoning over such implicit dependencies. The
agent returns a label 𝑞type∈{Self-contained,Dependent} , which
determines the subsequent processing path. The classification step
is formalized as:
𝑞type=𝑓(𝑞𝑖,𝐼𝐶)(2)
4.2.2 Rephraser Agent.The Rephraser Agent converts a context-
dependent question 𝑞𝑖into a self-contained version 𝑞𝑖𝑠by resolving
implicit references using the dialogue history𝐶 𝑞. This is achieved
through a prompt-driven LLM function:

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
𝑞𝑖𝑠=𝑓(𝑞𝑖,𝐶𝑞,𝐼𝑅𝐹)(3)
Here, the model receives the question 𝑞𝑖, its context 𝐶𝑞, and
an instruction prompt 𝐼𝑅𝐹that specifies the criteria for semantic
completeness. The prompt directs the model to resolve dependen-
cies—such as pronouns, ellipses, or anaphora—and rewrite the ques-
tion as a standalone utterance that no longer depends on prior turns.
Unlike rule-based systems that rely on handcrafted heuristics, this
approach leverages the LLM’s pretrained reasoning to generalize
across varied linguistic forms. Although the task involves text gen-
eration, it is bounded in scope: the objective is to restructure the
input without introducing new content. This constraint makes it
well-suited for zero-shot prompting. The resulting question 𝑞𝑖𝑠
preserves the original intent of 𝑞𝑖while satisfying structural and
semantic requirements for QIR generation.
Ambiguous or conflicting entity mentions across dialogue turns
are resolved by the rephraser agent using the full dialogue history.
The history stores explicit entity mentions in 𝐶𝑞, allowing the agent
to replace pronouns with the correct referenced entity before refor-
mulating the question. Since the agent relies on LLMs with strong
long-context coreference abilities, it can disambiguate pronouns
and produce a clear, standalone version of the user query.
4.2.3 QIR Agent.The QIR Agent converts the self-contained ques-
tion𝑞𝑖𝑠into a structured semantic form. It generates a set of triples
𝑅𝐹={⟨𝑠𝑖,𝑟𝑖,𝑜𝑖⟩}𝑘
𝑖=1, where𝑠𝑖,𝑜𝑖∈𝐸∪𝑈 and each relation 𝑟𝑖∈R,
the set of relation phrases in 𝑞𝑖𝑠. These triples represent the core
semantic structure of the question and are incorporated into the
final QIR only after passing a validation step. An initial prompting
method guided the LLM to extract triples by clarifying variable-
entity distinctions, highlighting relevant facts, and enforcing output
format. However, this led to frequent issues such as omitted vari-
ables and generic or ambiguous predicates (e.g.,has) that distorted
intent. To address this, the QIR Agent adopts a chain-of-thought
(CoT) prompting strategy combined with few-shot learning. The
CoT prompt decomposes the task into three reasoning steps: (i) iden-
tifying key entities, (ii) detecting unknowns (answers or interme-
diate variables), and (iii) generating verb- or noun-based relations
between them. This structured reasoning improves precision and
mitigates earlier shortcomings. Two examples (one with verb-based,
one with noun-based relations) are included to guide generation.
The agent receives a declarative prompt 𝐼𝑄𝑈and example set 𝑒,
returning a structured output in JSON format for downstream use.
The process is formalized as:
𝑅𝐹=𝑓(𝑞 𝑖𝑠,𝐼𝑄𝑈,𝑒)(4)
Triples Validator.After generating the candidate triples, the Triples
Validator checks the output against structural and semantic con-
straints. Specifically, the result must satisfy: (1) it is a well-formed
JSON object, (2) each triple includes a valid subject, predicate, and
object, (3) at least one triple references a known entity , and (4) for
non-boolean questions, at least one variable must appear. Boolean
questions such as “Is Michelle the wife of Barack Obama?” are
exempt from the fourth condition, as they may consist entirely
of grounded triples like ⟨Michelle,wife of,Barack Obama⟩ . If the
output fails validation, the QIR Agent retries generation up to athreshold𝜃. Once validated, the QIR is constructed from the vali-
dated triples and returned.
4.2.4 Time and Space Complexity.Algorithm 1 executes a bounded
sequence of LLM agent calls, whose cost is dominated by language
model interactions. Let 𝐶LLMdenote the cost of one LLM invocation.
The algorithm performs at most three LLM calls: one to the Classi-
fier Agent (Line 1), one conditional call to the Rephraser (Line 3),
and up to𝜃calls to the QIR Agent for triple generation (Line 9).
Thus, the total time complexity is 𝑂(𝜃·𝐶 LLM), since the number of
LLM calls is constant. The validation step (Line 10) uses lightweight
rule-based checks and runs in constant time. Space complexity is
dominated by the dialogue history D, which adds one QA pair per
turn, resulting in𝑂(𝑛)space for𝑛turns.
5 Query Generation and Answer Retrieval
This section describes theQuery Planning Agent, which converts
the QIR into a small set of SPARQL queries to retrieve the answer 𝐴.
Entity and relation linking are delegated to theMatching Agent. The
agent executes only a high-confidence subset of candidate queries,
requiring no KG-specific preprocessing, remaining compatible with
standard RDF backends, and reducing latency.
5.1 Matching Agent
Given the𝑄𝐼𝑅=(𝐸,𝑈,𝑅,𝑅𝐹) , the Matching Agent aligns QIR
entities and relations to their corresponding URIs in the target KG by
querying its SPARQL endpoint at runtime. As shown in Algorithm 2,
for each entity 𝑒∈𝐸 (Line 2), theEntity Retrieverissues a keyword-
based SPARQL query to retrieve up to 𝑣𝑙𝑖𝑚𝑖𝑡 candidate vertices (Line
3). Vertex selection is then performed via prompt-guided reasoning
using instructions 𝐼𝑉𝐿(Line 6). The result is validated by theVertex
Validator, which checks both membership in the candidate set and
JSON correctness (Line 7). If validation fails, selection is retried up
to𝜃times (Lines 5–9) to mitigate LLM hallucinations or formatting
errors. Validated mappings are stored in 𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥 (Line 10).
Next, for each relation 𝑟∈𝑅 (Line 12), theRelation Linking Tool
retrieves and ranks candidate predicates to maximize recall. The
resulting ranked list is stored in 𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑 (Line 14). Finally, the
agent returns: (i) 𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥 :𝐸→𝑉 , mapping each entity
to a KG vertex, and (ii) 𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑 :𝑅→List(𝑃) , mapping each
relation to a list of semantically relevant predicates10. This runtime-
only, modular design avoids KG-specific preprocessing, schema
alignment, or offline indexing. As a result, it generalizes well across
diverse KGs while maintaining high precision and adaptability.
5.1.1 Entity Retriever.Entity linking aims to match each known
entity𝑒∈𝐸 in the𝑄𝐼𝑅=(𝐸,𝑈,𝑅,𝑅𝐹) to a unique, semantically
appropriate vertex 𝑣∈𝑉 in the Knowledge Graph (KG). Formally,
this defines a mapping 𝑓:𝐸→𝑉 , where each entity is aligned
to one KG vertex. To ensure compatibility with arbitrary SPARQL
endpoints and avoid KG-specific preprocessing or offline indexing,
the Entity Retriever dynamically queries the RDF engine using
keyword-based SPARQL. It extracts candidate vertices whose labels
contain words from the entity name. The number of candidates per
entity is limited by a configurable parameter 𝑣limit, Once the list
10Here, List(𝑃) denotes a list of predicates, allowing each relation to link to multiple
KG predicates rather than a single one.

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
Algorithm 2Vertex and Relation Linking
Input:𝑄𝐼𝑅=(𝐸,𝑈,𝑅,𝑅𝐹): Intermediate representation,𝑣 𝑙𝑖𝑚𝑖𝑡:
Maximum number of retrieved vertices,𝑒𝑛𝑑𝑝𝑜𝑖𝑛𝑡: SPARQL
endpoint,𝐼 𝑉𝐿: Vertex linking prompt,𝜃: Retry threshold
Output:𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥: Entity to linked vertices map,𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑:
Relation to linked predicates map
1:𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥,𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑←{},{}
2:foreach𝑒∈𝑄𝐼𝑅.𝐸do
3:𝑣𝑙𝑖𝑠𝑡←MAgent.entity_retriever(𝑒,𝑣 𝑙𝑖𝑚𝑖𝑡,𝑒𝑛𝑑𝑝𝑜𝑖𝑛𝑡)
4:𝑣𝑎𝑙𝑖𝑑←False
5:while𝑣𝑎𝑙𝑖𝑑=False∧𝜃>0do
6:𝑣𝑐ℎ𝑜𝑠𝑒𝑛←MAgent(𝑒,𝑣 𝑙𝑖𝑠𝑡,𝐼𝑉𝐿)
7:𝑣𝑎𝑙𝑖𝑑←MAgent.Validate(𝑣 𝑐ℎ𝑜𝑠𝑒𝑛,𝑣𝑙𝑖𝑠𝑡)
8:𝜃←𝜃−1
9:end while
10:𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥[𝑒]←𝑣 𝑐ℎ𝑜𝑠𝑒𝑛
11:end for
12:foreach𝑟𝑓∈𝑄𝐼𝑅.𝑅𝐹do
13:𝑝𝑟𝑒𝑑𝑠←MAgent.rel_Linking(𝑟𝑓,𝑒𝑛𝑑𝑝𝑜𝑖𝑛𝑡,𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥)
14:𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑[𝑟𝑓.𝑟]←𝑝𝑟𝑒𝑑𝑠
15:end for
16:Return EntToVertex, RelToPred
𝑣list⊆𝑉 is retrieved for an entity 𝑒, the Matching Agent invokes a
prompt-guided LLM to select the best candidate:
𝑣=𝑓(𝑒,𝑣 list,𝐼𝑉𝐿),(5)
where𝐼𝑉𝐿is a handcrafted prompt instructing the LLM to assess
label similarity, favoring exact matches when available. To support
weaker models, the prompt includes a definition of "exact match."
Only vertex labels (not URIs) are provided to improve interpretabil-
ity, and the output is formatted in JSON for reliable parsing. This
zero-shot classification setup allows the LLM to act as a semantic fil-
ter under in-context learning, enabling generalization across varied
and domain-specific KGs. By decoupling retrieval from selection
and avoiding retraining, the Entity Retriever ensures modularity,
extensibility, and adaptability without manual schema alignment.
5.1.2 Vertex Validator.The Vertex Validator ensures semantic and
syntactic correctness of the selected vertex 𝑣before finalizing en-
tity alignment. It enforces two key constraints. First, it verifies
that the LLM’s output is a syntactically valid and parsable JSON
object, suitable for downstream processing. Second, it checks that
the selected vertex 𝑣belongs to the retrieved candidate list 𝑣list,
guaranteeing that the output corresponds to a real KG entity and
not a hallucinated or out-of-scope result. If either validation step
fails, the Matching Agent re-invokes the vertex selection tool with a
retry budget bounded by the threshold 𝜃. This mechanism ensures
that only grounded, well-formed vertex selections are propagated
to subsequent stages, thereby preserving system robustness and
preventing semantic drift in the query generation process.
5.1.3 Relation Linking.For relation linking, assigning a single pred-
icate to each semantic relation is often insufficient, as entities in
the KG may be connected through multiple semantically related
predicates. For example, in DBpedia, the entityIntelis linked viapredicates such as founders ,founder , and foundedBy . To accu-
rately answer a question like “Who founded Intel?”, the system
must consider all such alternatives to ensure comprehensive cover-
age. The Relation Linking tool addresses this by associating each
relation𝑟∈𝑄𝐼𝑅.𝑅 with a ranked list of candidate predicates from
the KG. For a given relation 𝑟, the tool queries the SPARQL endpoint
to retrieve all predicates that connect the previously linked source
and target entities. Each predicate URI is converted into a human-
readable label by extracting its final segment. Both the relation
label and predicate labels are encoded using BERT [ 9], and their
semantic similarity is measured via cosine similarity. The resulting
predicates are ranked and stored in the 𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑 mapping, allow-
ing the downstream query planner to select the most appropriate
predicates without relying on KG-specific rules or preprocessing.
5.1.4 Time and Space Complexity.The time complexity of the ver-
tex and relation linking phase is 𝑂(𝑞 cost+𝜃·𝐶 LLM), where𝑞costis the
cost of SPARQL queries used to retrieve candidate vertices and pred-
icates. Since each QIR has only a few entities and relations (typically
<10), both SPARQL access and LLM inference operate on bounded
input sizes, and validation runs in constant time. The value of 𝑞cost
is implementation-dependent and varies with the RDF engine. For
space complexity, storing one vertex per entity yields constant over-
head for EntToVertex , while the main cost comes from RelToPred ,
which stores ranked predicates. Let 𝑝canbe the number of retrieved
candidate predicates; total space usage is𝑂(𝑝 can).
5.2 Query Planning Agent
Once the Matching Agent produces the mappings of EntToVertex
and𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑 , the Query Planning Agent uses them along with the
𝑄𝐼𝑅 and the user question 𝑞𝑖. It manages SPARQL query construc-
tion, selection, and execution through three tools:Query Generator,
Query Validator, andQuery Execution. Query selection is performed
directly by the agent. As shown in Lines 1–2 of Algorithm 3, the
Query Generator creates a candidate setQ={𝜓 1,...,𝜓𝑘}by com-
bining entity mappings from EntToVertex with predicate options
from RelToPred . To limit latency, the candidate set is truncated
to a bounded size 𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚 . The agent then filters Qbased on
predicate relevance. Before execution, selected queries are passed
to theQuery Validator, which checks structural correctness and
semantic validity. Finally, the validated queries Q′are executed
via the SPARQL endpoint using the Query Execution tool, and the
retrieved answers are aggregated into the final result 𝐴, which is
returned to the Chat Agent.
5.2.1 Query Generator.The Query Generator constructs a candi-
date set of SPARQL queries Q={𝜓 1,...,𝜓𝑘}using the QIR as the
structural backbone. It populates each triple ⟨𝑠,𝑟,𝑜⟩∈𝑅𝐹 by replac-
ing entities with their corresponding vertices from EntToVertex ,
and substituting relations with predicate options from RelToPred (𝑟).
Unknowns are included directly as variables. Since each relation
may map to multiple predicates, a single semantic triple can yield
multiple SPARQL triple patterns. The final queries are formed by
joining these patterns based on shared variables or entities. This
combinatorial expansion produces multiple candidate queries, each
representing a distinct predicate configuration. The main unknown
representing the target answer is added to theSELECTclause.

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
Algorithm 3Query Selection and Execution
Input:𝑄𝐼𝑅: Intermediate representation,𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥:
Entity-to-vertex map,𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑: Relation-to-predicate map,
𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚: Query limit,𝑞 𝑖: User question,𝑒𝑛𝑑𝑝𝑜𝑖𝑛𝑡: SPARQL
endpoint,𝐼 𝑄𝑆: Predicate selection prompt,𝜃: Retry threshold
Output:𝐴: Final answer
1:Q←QPAgent.Gen(𝑄𝐼𝑅,𝐸𝑛𝑡𝑇𝑜𝑉𝑒𝑟𝑡𝑒𝑥,𝑅𝑒𝑙𝑇𝑜𝑃𝑟𝑒𝑑)
2:Q←Truncate(Q,𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚)
3:P,PredToQuery←[],{}
4:foreach𝑖,𝜓∈Qdo
5:𝑃𝜓←QPAgent.Pred(𝜓)
6:P←P∪𝑃 𝜓
7:foreach𝑝∈𝑃 𝜓do
8:PredToQuery[𝑝]←PredToQuery[𝑝]∪{𝑖}
9:end for
10:end for
11:valid←False,𝜃←𝜃
12:whilenot valid and𝜃>0do
13:P′←QPAgent(𝑞 𝑖,P,𝐼𝑄𝑆)
14:valid←QPAgent.Validate(P′,P)
15:𝜃←𝜃−1
16:end while
17:Q′←{𝜓∈Q|QPAgent.Pred(𝜓)∩P′≠∅}
18:𝐴←QPAgent.Exec(Q′,𝑒𝑛𝑑𝑝𝑜𝑖𝑛𝑡)
19:return𝐴
5.2.2 Query Selection.Lines 3–17 of Algorithm 3 describe the
query selection process performed by the Query Planning Agent.
The agent iterates over the candidate queries 𝜓∈Q , extracts their
predicate sets Pred(𝜓) , and aggregates them into a global list P.
Simultaneously, it builds an inverted index PredToQuery that maps
each predicate to the queries containing it. To reduce execution
cost and filter out irrelevant queries, the agent applies an LLM-
based predicate filtering strategy. It provides the user question 𝑞𝑖,
predicate listP, and a zero-shot chain-of-thought prompt 𝐼𝑄𝑆to
the LLM. The model returns a filtered subset P′⊆P of predicates
considered relevant to the question. This output is validated using
the Query Validator to ensure it is well-formed JSON and contains
only predicates from the original list. The process is retried up
to𝜃times if the output is invalid or malformed. The agent then
filters the candidate query set using the validated predicate subset:
Q′={𝜓∈Q|Pred(𝜓)∩P′≠∅} . This step reduces query volume
while preserving recall, which help improve efficiency without
sacrificing answer accuracy.
5.2.3 Query Validator.The Query Validator tool verifies the se-
lected predicatesP′by ensuring that (i) the LLM output is a well-
formed, parsable JSON object, and (ii) P′contains only predicates
from the candidate set P. The validator filters out any predicate
inP′not found inP. If the resulting list is empty, the output is
deemed invalid, and the Query Planning Agent retries the selection.
Entities are already grounded by the Matching Agent, and SPARQL
queries are constructed deterministically from the selected predi-
cates. Therefore, validating P′effectively guarantees the correct-
ness of the final queries. These checks mitigate LLM hallucinations
with minimal overhead. Since Pis stored locally as a hash-based
structure, validation is efficient.5.2.4 Query Execution.The Query Execution tool dispatches the
filtered SPARQL queries Q′to the knowledge graph and aggregates
the results into a unified answer set 𝐴. It interfaces with a stan-
dard SPARQL endpoint using API calls, executes each query in Q′,
collects result bindings, removes duplicates, and formats the final
output for the Chat Agent. This component is lightweight, stateless,
and compatible with standard RDF engines.
5.2.5 Error Handling and Fallback Strategy.If no answer is re-
turned, this indicates that the KG may not contain the requested
information. The system does not attempt to generate an alterna-
tive answer, as this may lead to hallucination. This design ensures
the system returns either a factually grounded response based on
available data or explicitly indicates that the requested information
is not present in the KG. Inconsistent results are avoided through
assertion-based validations applied throughout the pipeline, espe-
cially during query selection. These validations ensure that inter-
mediate outputs remain relevant to the user’s question.
Time and Space ComplexityAlgorithm 3 has a time complexity
of𝑂(𝑞 cost+𝜃·𝐶 LLM), where𝑞costdenotes the cost of SPARQL
query execution and 𝐶LLMis the cost of a single LLM call. The loop
over candidate queries (Lines 3–10) is bounded by the truncation
parameter𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚 , typically less than 100, and does not affect
asymptotic complexity. LLM output validation (Lines 11–16) incurs
constant time per attempt, with up to 𝜃retries. Space complexity is
dominated by the storage of candidate queries and the predicate-
to-query index. Both scale linearly with 𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚 , resulting in
overall space complexity of 𝑂(𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚) . Since fewer queries are
usually executed than 𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚 , the algorithm keeps overhead
low while preserving accuracy and robustness.
6 Implementation
We implementChatty-KG11using the LangGraph framework12, an
extension of LangChain that introduces graph-based abstractions
for multi-agent workflows. LangGraph enables structured control
over agent composition and inter-agent communication via a shared
state and programmable routing logic. Each agent inChatty-KGis
implemented as an independent function that reads and updates a
shared AgentState object. This object acts as centralized memory,
storing the user question, intermediate outputs, and final results.
These agent functions are composed into a StateGraph , where
conditional transitions depend on the evolving state. As shown in
Listing 1, execution begins at chat_agent (Line 19) and proceeds
through agent nodes (Lines 12–17), including classifier_agent ,
rephraser_agent , and qir_agent . While Figure 2 depicts a fixed
logical view, the implementation is dynamic: each agent sets the
route field to determine the next node. This decouples routing from
static topology, enabling context-sensitive transitions and modular
reconfiguration without altering control flow.
The modular design ofChatty-KGenables adaptability at mul-
tiple levels.LLM substitutionis straightforward: agents can use
different underlying models (e.g., GPT-4, Phi, Mistral) by updating
internal configurations.Agent extensibilityis also supported,
allowing components to be swapped or extended, e.g., replacing a
11https://github.com/CoDS-GCS/Chatty-KG
12https://python.langchain.com/docs/langgraph/

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
Listing 1: LangGraph-based implementation ofChatty-KG.
The shared agent state tracks intermediate outputs, such as
rephrased questions, SPARQL queries, and routing decisions,
exchanged among agents during dialogue processing.
1# Agent function signatures ( each takes and returns AgentState )
2def chat_agent ( state : AgentState ) -> AgentState : ...
3def classifier_agent ( state : AgentState ) -> AgentState : ...
4def rephraser_agent ( state : AgentState ) -> AgentState : ...
5def qir_agent ( state : AgentState ) -> AgentState : ...
6def query_agent ( state : AgentState ) -> AgentState : ...
7def matching_agent ( state : AgentState ) -> AgentState : ...
8
9# Graph construction
10builder = StateGraph ( state_schema = AgentState )
11
12builder . add_node (" chat_agent ", chat_agent )
13builder . add_node (" classifier_agent ", classifier_agent )
14builder . add_node (" rephraser_agent ", rephraser_agent )
15builder . add_node (" qir_agent ", qir_agent )
16builder . add_node (" query_agent ", query_agent )
17builder . add_node (" matching_agent ", matching_agent )
18
19builder . set_entry_point (" chat_agent ")
20
21# Define transitions between agent nodes
22builder . add_edge (" chat_agent ",
23lambda state : END if state . query_done else (" query_agent " if
state . qir_done else " qir_agent "))
24builder . add_conditional_edges (" classifier_agent ",
25lambda state : state . route )
26builder . add_conditional_edges (" rephraser_agent ",
27lambda state : state . route )
28builder . add_conditional_edges (" qir_agent ",
29lambda state : state . route )
30builder . add_conditional_edges (" query_agent ",
31lambda state : state . route )
32builder . add_conditional_edges (" matching_agent ",
33lambda state : state . route )
34
35builder . set_finish_point (" chat_agent ")
36graph = builder . compile ()
rule-based tool with a neural module, without retraining or reengi-
neering the system. Finally,dialogue mode controlis governed
by a system_mode flag within the shared state, allowing the sys-
tem to toggle between multi-turn and single-turn execution. These
features makeChatty-KGa flexible and extensible platform for
research and deployment in diverse KGQA settings.
Chatty-KGuses four main parameters: (1) 𝜃, the maximum
number of retries for LLM calls when validation fails; (2) 𝐿, the
number of past answers included in the context 𝐶𝑞to fit within
the LLM’s input length; (3) 𝑣𝑙𝑖𝑚𝑖𝑡, the number of candidate vertices
retrieved by the entity retriever; and (4) 𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚 , the maximum
number of candidate queries considered during query selection. In
our experiments, we set these to3,100,600, and40, respectively.
The𝜃parameter helps reduce hallucinations by allowing the LLM
multiple attempts to produce valid outputs. While higher values
improve robustness, they also increase cost and latency. Models like
GPT-4o and Gemini benefit from retries by producing varied and
often valid responses, whereas weaker models like Llama show lim-
ited gains. We set 𝐿=100based on empirical testing. Larger values
may exceed the LLM context window for long answers (e.g., lists of
papers), while smaller values risk omitting context needed by the
rephraser. The 𝑣𝑙𝑖𝑚𝑖𝑡 parameter balances recall and prompt complex-
ity: higher values improve recall but add noise, while lower values
reduce noise but risk missing the correct vertex. The 𝑞𝑢𝑒𝑟𝑦_𝑛𝑢𝑚
value must retain valid queries without increasing runtime. For
fairness in evaluation, we disable natural-language reformulation
and compare against structured gold-standard answers.Table 2: KG Statistics: The number of triples and entities in
Millions, and the number of predicates of each KG.
KG # Entities(M) # Triples(M) # Predicates
DBLP18 263 167
YAGO12 207 259
DBPedia14 350 60,736
MAG586 13, 705 178
Wikidata13119 8,541 12,943
7 Evaluation
We evaluateChatty-KGalong five key dimensions: (1) perfor-
mance on single-turn QA compared to state-of-the-art KGQA sys-
tems, (2) accuracy on multi-turn questions over KGs relative to
top-performing chatbots, LLMs and RAG-based systems, (3) the
contribution of our contextual understanding module, (4) the effect
of our LLM-powered question interpretation, and (5) the impact of
our query planning and selection on system efficiency.
7.1 Evaluation Setup
Baselines:We evaluateChatty-KGagainst state-of-the-art sys-
tems in two settings: single-turn and multi-turn question answer-
ing. For single-turn, we compare against KGQAn [ 30], a leading
KGQA system, and EDGQA [ 15]. For multi-turn (conversational)
QA, we compare against CONVINSE [ 5] and EXPLAIGNN [ 6]. Both
are designed for KG-based dialogue and trained on the ConvMix
dataset [ 5], which augments multi-turn QA over Wikidata. Unlike
Chatty-KG, these systems require task-specific training and are
tightly coupled to the KG they were trained on. We evaluated two
widely usedRAG systems, ColBERT [ 21,37] and GraphRAG [ 13],
on DBLP (our smallest KG). Both require substantial compute and
memory. ColBERT performs late-interaction retrieval without build-
ing a graph, while GraphRAG constructs and indexes one for down-
stream QA. Indexing DBLP with GraphRAG required over 80 hours
using a 512 GB, 64-core VM, used solely for the RAG experiment.
Extrapolating from this, larger KGs would require terabytes of
memory, making RAG indexing impractical in our setting.
Underlying LLMs:We evaluateChatty-KGusing: (1) Commercial
LLMs, GPT-4o [ 32], Gemini-2.0-Flash [ 42], and DeepSeek-V3 [ 8],
accessed via API. (2) Open-weight LLMs, such as Qwen-2.5 [ 51], Phi-
4 [1], Gemma-3 [ 19], CodeLlama [ 35], Gemma-2 [ 34], Granite [ 16],
Vicuna [27], Llama-3 [2], and Mistral [18], all hosted locally.
Five Real-World KGs:To assessChatty-KG’s performance across
domains, we evaluate it on five real-world KGs: DBpedia [ 24],
YAGO [ 11,41], DBLP [ 7], Microsoft Academic Graph (MAG) [ 28],
and Wikidata [ 46]. DBpedia, Wikidata and YAGO cover general
knowledge (e.g., people, places), while DBLP and MAG focus on aca-
demic content, including long entity names like paper titles. These
KGs vary in size, enabling evaluation ofChatty-KG’s scalability
and robustness. Table 2 summarizes the number of entities, predi-
cates, and triples. For wikidata SPARQL execution, we use the public
available endpoint. For the other graph’s SPARQL execution, we use
Virtuoso v7.2.5.2, a standard backend for large KGs, with a separate
endpoint per KG. All experiments use the default Virtuoso setup.
Compute Infrastructure:We use two different setups for our
experiments: (i) Linux machine with 16 cores and 180GB RAM for
13https://www.wikidata.org/wiki/Wikidata:Statistics

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
Table 3: Comparison between state-of-the-art KGQA systems andChatty-KGon single-turn question answering across five
benchmarks covering four knowledge graphs.Chatty-KGis evaluated with various LLMs. Open-weight LLMs are ordered
top-down by parameter count. Bold indicates best performance, underlined indicates second-best. Green highlights performance
better than KGQA systems; yellow highlights results within 1 point. RAG systems require huge memory to run.
QALD-9 LC-QuAD 1.0 YAGO DBLP MAG
System(Model) P R F1 P R F1 P R F1 P R F1 P R F1
KGQAn51.13 38.72 44.07 58.71 46.11 51.65 48.48 65.22 55.62 57.87 52.02 54.79 55.43 45.61 50.05
EDGQA31.30 40.30 32.00 50.5056.0053.10 41.90 40.80 41.40 8.00 8.00 8.00 4.00 4.00 4.00
GraphRAG (GPT-4o)- - - - - - - - - 46.00 37.01 38.99 - - -
COLBERT (GPT-4o)- - - - - - - - - 65.38 55.33 55.29 - - -
Chatty-KG(GPT-4o)61.9141.50 49.69 72.7943.60 54.54 86.52 72.27 78.75 85.96 75.50 80.39 75.0569.05 71.92
Chatty-KG(Gemini-2.0-Flash) 57.23 42.14 48.54 68.27 43.35 53.03 90.04 75.47 82.11 74.80 67.50 70.96 59.55 57.56 58.54
Chatty-KG(DeepSeek-V3) 50.76 40.83 45.26 69.66 45.42 54.99 78.77 74.07 76.34 71.8886.50 78.52 64.4372.36 68.16
Chatty-KG(Phi-4) 55.95 37.75 45.08 69.09 42.43 52.57 86.89 74.07 79.97 81.04 75.50 78.17 63.23 62.70 62.96
Chatty-KG(Qwen2.5-14B-Inst) 50.77 42.17 46.07 68.84 42.69 52.70 83.73 67.07 74.48 65.96 73.50 69.53 57.58 61.93 59.68
Chatty-KG(Qwen2.5-14B) 48.99 39.84 43.94 62.49 41.29 49.72 74.77 71.76 73.23 58.61 68.50 63.17 61.73 62.56 62.14
Chatty-KG(Vicuna-13B-v1.5) 62.40 26.17 36.87 73.78 27.92 40.51 58.53 36.00 44.58 64.90 30.50 41.50 75.93 34.51 47.45
Chatty-KG(CodeLlama-13B) 43.99 32.32 37.26 56.31 38.41 45.67 44.48 53.24 48.47 39.08 68.00 49.64 34.86 61.82 44.58
Chatty-KG(Gemma3-Inst) 53.64 36.93 43.75 66.13 39.09 49.13 72.67 65.87 69.10 71.44 76.50 73.88 49.28 43.52 46.22
Chatty-KG(Gemma3) 57.11 34.76 43.22 63.65 39.33 48.62 75.33 72.79 74.04 57.06 63.50 60.11 54.64 51.30 52.92
Chatty-KG(Gemma2-Inst) 51.16 35.47 41.90 65.88 39.85 49.66 81.57 60.47 69.45 79.81 68.00 73.43 65.95 49.22 56.37
Chatty-KG(Gemma2) 43.19 37.59 40.20 53.95 41.56 46.95 62.93 66.47 64.65 48.46 58.00 52.80 48.19 44.23 46.12
Chatty-KG(Ministral-8B-Inst) 56.72 37.47 45.13 65.31 39.98 49.60 78.92 72.19 75.41 87.9469.50 77.64 58.17 58.54 58.35
Chatty-KG(Granite-3.2-8B-Inst) 47.23 36.68 41.29 66.59 34.27 45.25 69.68 72.46 71.04 74.09 63.00 68.10 51.00 56.71 53.70
Chatty-KG(Llama-3.1-8B-Inst) 56.31 32.54 41.25 65.66 37.63 47.84 67.61 70.66 69.10 66.24 60.01 62.98 63.87 64.06 63.97
Chatty-KG(Mistral-7B-v0.3) 48.75 36.02 41.43 61.39 33.41 43.27 50.02 65.99 56.91 47.36 63.00 54.07 35.09 59.28 44.09
running all experiments onChatty-KGandKGQAn. It is also used
to host the four Virtuoso SPARQL endpoints for the KGs. (ii) Linux
machine with Nvidia A100 40GB GPU, 32 cores, and 64GB RAM
to host the open-weight LLMs, creating a server accessible using
HTTP requests via the VLLM library [23].
Benchmarks:We evaluateChatty-KGon a comprehensive set of
KGQA benchmarks spanning both single-turn and multi-turn QA
over multiple KGs. Forsingle-turn QA, we use standard datasets
such as QALD-9 [ 44] and LC-QuAD 1.0 [ 43], which provide SPARQL-
annotated questions and gold answers over DBpedia. QALD-9 in-
cludes 408 training and 150 test questions, while LC-QuAD 1.0
provides 4,000 training and 1,000 test questions generated from
diverse templates. SinceChatty-KGrequires no training, we only
use the test sets. We also include datasets introduced by [ 30], with
100 questions each for YAGO, DBLP, and MAG, covering varied
domains and KG structures. Formulti-turn QA, we generate 20
dialogues per KG (approximately 100 questions) using a state-of-
the-art dialogue benchmark generator [ 31]. Each dialogue consists
of 5 turns, with every question 𝑞paired with its standalone version,
a corresponding SPARQL query, and the correct answer. These gen-
erated benchmarks allow us to evaluateChatty-KG’s contextual
understanding and reasoning capabilities in a reproducible setting.
Metrics:For single-turn QA, we use standard KGQA metrics: Preci-
sion (P), Recall (R), and F1 score. Recall measures the proportion of
correct answers returned by the system, while Precision measures
the proportion of returned answers that are correct. The F1 score is
the harmonic mean of Precision and Recall. For multi-turn QA, we
adopt the metrics used by CONVINSE and EXPLAIGNN: Precision
at 1 (P@1), Mean Reciprocal Rank (MRR), and Hit at 5 (Hit@5).
P@1 checks whether the top-ranked answer is correct and is com-
mon when systems return a single answer [ 5]. MRR evaluates the
position of the correct answer in a ranked list. Hit@5 measures
whether the correct answer appears among the top five results.7.2 Single-Turn KGQA
We evaluateChatty-KGon single-turn QA using five benchmarks
and compare it with two state-of-the-art systems: KGQAn and
EDGQA. Results are reported in Table 3. We also assess howChatty-
KG’s performance varies when paired with different commercial
and open-weight LLMs.Chatty-KGoutperforms both baselines
when using high-performing LLMs like GPT-4o, Gemini-2.0, and
DeepSeek. For example, GPT-4o achieves F1 scores of 80.39 on
DBLP and 71.92 on MAG, compared to KGQAn’s 54.79 and 50.05.
On QALD and LC-QuAD, it leads with F1 scores of 49.69 and 54.54,
improving over KGQAn by +5.62 and +2.89, respectively.
GPT-4o also yields strong precision, such as 86.52 on YAGO,
showing thatChatty-KG’s query planning effectively filters in-
correct queries. DeepSeek-V3 achieves high recall, e.g., 86.50 on
DBLP. Models like Phi-4 and Qwen2.5-14B-Instruct provide a good
trade-off between precision and recall, outperforming KGQAn in
the five benchmarks. Notably, even smaller open-weight models
like Mistral-7B and Llama-3.1-8B match or exceed KGQAn on some
datasets. This shows thatChatty-KGachieves strong performance
without task-specific fine-tuning. Our modular prompting and accu-
rate query selection enable this capability, even with smaller models.
EDGQA performs competitively on LC-QuAD (F1 = 53.10) but
fails on domain-specific KGs, such as DBLP (F1 = 8.0) and MAG (F1
= 4.0), likely due to its rule-based architecture. KGQAn performs
better overall but struggles with precision on general datasets like
QALD-9 (P = 51.13) and YAGO (P = 48.48), often retrieving irrel-
evant answers. In contrast,Chatty-KGmaintains high precision
across both general and special domains. It delivers strong F1 and
precision scores on datasets like DBLP and YAGO. These results
demonstrate the effectiveness of our query planning agent. Given
GPT-4o’s consistent performance across all datasets and metrics, we
use GPT-4o as the default model for the remainder of our evaluation,
unless otherwise specified.

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
Table 4: Performance ofChatty-KGon multi-turn questions, compared to conversational baselines (CONVINSE, EXPLAIGNN)
and general LLMs (GPT-4o, Gemini, DeepSeek, Phi-4, Qwen) across three KGs. Metrics: P@1, MRR, and Hit@5.Chatty-KG
outperforms all baselines, achieving up to +87% higher P@1 on YAGO and nearly 3×higher accuracy on DBLP.
System Wikidata DBpedia YAGO DBLP
P@1 MRR Hit@5 P@1 MRR Hit@5 P@1 MRR Hit@5 P@1 MRR Hit@5
CONVINSE 27.47 27.47 27.47 28.72 28.72 28.72 34.88 34.88 34.88 28.89 28.89 28.89
EXPLAIGNN 26.37 26.37 26.37 29.79 29.79 29.79 31.40 31.40 31.40 28.89 28.89 28.89
GraphRAG (GPT-4o) - - - - - - - - - 11.10 11.10 11.10
ColBERT (GPT-4o) - - - - - - - - - 17.78 17.78 17.78
GPT-4o 49.45 49.45 49.45 30.85 30.85 30.85 27.91 27.91 30.23 24.44 24.44 24.44
Gemini-2.0-Flash 41.76 41.76 41.76 25.53 24.47 24.47 30.23 31.10 32.56 13.33 13.33 13.33
DeepSeek-V3 40.65 40.65 40.65 20.21 20.21 20.21 26.74 26.74 26.74 16.67 16.67 16.67
Phi-4 25.27 25.27 25.27 26.60 25.53 25.53 15.12 15.31 15.12 17.78 17.78 17.78
Qwen2.5-14B-Instruct 24.18 24.18 24.18 18.09 18.09 18.09 16.28 16.86 17.44 6.67 5.56 5.56
Chatty-KG(GPT-4o) 54.95 54.95 54.95 34.04 35.11 36.17 65.12 65.70 66.28 83.33 83.33 83.33
We also evaluated two RAG systems, GraphRAG and ColBERT, to
contextualize our results. On DBLP, GraphRAG achieved moderate
precision (46.00) but low recall (37.01), while ColBERT performed
better (P = 65.38, R = 55.33, F1 = 55.29). Both struggled with list-
style questions, often returning only partial answers, and failed
on multi-hop queries because text chunking breaks graph struc-
ture. GraphRAG also required heavy indexing and still missed rela-
tions, confirming the scalability challenge. These results support
our observation that current RAG implementations cannot reli-
ably preserve structure or guarantee complete answers. In contrast,
Chatty-KGoperates directly over the KG and executes SPARQL,
enabling accurate, complete retrieval and consistent performance.
7.3 Multi-turn Dialogues
We evaluateChatty-KGon multi-turn question answering using
20 dialogues (up to 100 questions) generated by a state-of-the-art
benchmark generator [ 31]. We compare against two conversational
KGQA systems, CONVINSE and EXPLAIGNN, and several general-
purpose LLMs that do not have access to KGs (GPT-4o, Gemini-2.0-
Flash, DeepSeek-V3, Phi-4, and Qwen2.5-14B-Instruct).
Dialogue Datasets.CONVINSE and EXPLAIGNN were trained
on Wikidata, while we evaluate on DBpedia, YAGO, and DBLP. To
ensure fairness, we manually selected 20 entities that exist in both
Wikidata and our target KGs. For DBpedia and YAGO, we veri-
fied alignment using Wikidata links. For DBLP, which lacks clear
mapping, we selected entities relevant to research discussions (e.g.,
“Gerhard Kramer”). CONVINSE and EXPLAIGNN were accessed
through their official demos1415, and their outputs were converted
to match our evaluation format. The performance of CONVINSE
and EXPLAIGNN in our setup matches the results reported in their
original papers, confirming the fairness of our experimental design.
Performance Results.Table 4 presents results using P@1, MRR,
and Hit@5.Chatty-KGoutperforms all baselines and LLMs across
every KG and metric. On Wikidata, it achieves 54.95 P@1, signifi-
cantly outperforming CONVINSE (27.47) and EXPLAIGNN (26.37)
which were trained on Wikidata. On YAGO, it achieves an 87%
14https://convinse.mpi-inf.mpg.de/
15https://explaignn.mpi-inf.mpg.de/Table 5: Performance comparison ofChatty-KGwith and
without the conversational module using GPT-4o. Dialogue
results reflect context-dependent questions handled with the
module, while Standalone results use context-free questions.
Retention (% = F1 Dialogue / F1 Standalone×100) indicates the
proportion of accuracy retained in dialogue settings.
Dataset Method P R F1 Retention (%)
WikidataDialogue 78.65 57.14 66.1995.28Standalone 79.75 61.54 69.47
DBpediaDialogue 68.19 42.55 52.4092.27Standalone 69.79 47.87 56.79
YAGODialogue 74.86 67.44 70.9685.48Standalone 84.69 81.40 83.01
DBLPDialogue 81.85 74.44 77.9787.09Standalone 89.07 90.00 89.53
performance gain in P@1 compared to CONVINSE. On DBLP, a
domain-specific KG, it reaches 83.33 P@1, nearly 3X the score of
the best baseline. On DBpedia,Chatty-KGstill leads with 34.04
P@1, ahead of EXPLAIGNN’s 29.79. The improvement on DBpe-
dia is smaller due to its scale and structure. As shown in Table 2,
DBpedia has 60,736 predicates, about234 ×more than YAGO and
4×more than Wikidata, which introduces high ambiguity in predi-
cate selection. Many predicates are also redundant or synonymous
(e.g.,founders,founder,foundedByforIntel), making query planning
more difficult. Despite this,Chatty-KGremains competitive on
DBpedia and does so without KG-specific training or preprocessing.
Our modular agents ground responses in the KG through accurate
linking, context reasoning, and minimal query execution. This de-
sign adapts to different graph structures and remains robust across
domains without retraining or pipeline changes.
We also evaluated two RAG systems, GraphRAG and ColBERT,
in the same multi-turn setting. GraphRAG performed poorly (e.g.,
P@1 = 11.10 on DBLP), and ColBERT showed similarly low results
(P@1 = 17.78). Both systems handle each query independently and
lack dialogue-state tracking, so they cannot resolve references or

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
K C K C K C K C020406080Number of Failing questions3688
2485
QALD832
123
YAGO1046
323
DBLP447
124
MAGFailed due to other reasons
Failed due to QU
Figure 3: Number of failed questions (i.e., Recall = 0) in each
benchmark. Each bar is divided to show failures caused by
Question Understanding (QU) and other factors. Shading vari-
ations indicate the source of failure.Chatty-KGconsistently
has the fewest failures across all benchmarks.
preserve context across turns. Their retrieval resets at every step,
breaking conversational coherence and leading to low accuracy.
In contrast,Chatty-KGmaintains dialogue context and delivers
consistent multi-turn performance.
General LLM Limitations.While LLMs used without KG access
like GPT-4o and Gemini show moderate performance on DBpedia
and YAGO, they struggle on DBLP. For instance, GPT-4o drops to
24.44 P@1 on DBLP, far behindChatty-KG. This is expected, as
domain-specific KGs like DBLP may be underrepresented in LLM
training data. In contrast,Chatty-KGgrounds its answers directly
in the KG and executes structured queries. This design enables
higher accuracy, better verification, and lower latency. It handles
evolving domains without retraining or KG-specific pipelines. As
LLMs are limited by static training data,Chatty-KGprovides a
more reliable and up-to-date solution for multi-turn KGQA. Its com-
bination of contextual understanding and precise query planning
ensures high accuracy and low latency without requiring retraining.
7.4 AnalyzingChatty-KG
Contextual Understanding:We evaluateChatty-KG’s ability
to handle multi-turn questions. For this, we reuse the same bench-
marks from the multi-turn evaluation with CONVINSE and EX-
PLAIGNN. Each dialogue in these benchmarks includes aligned
question pairs: a context-dependent version and its correspond-
ing standalone form. We runChatty-KGin two modes: (1) with
the Classifier and Rephraser Agents enabled, using the context-
dependent questions; and (2) with these agents disabled, using the
standalone versions. This setup tests the system’s ability to recon-
struct missing context from dialogue history. As shown in Table 5,
Chatty-KGretains over 85% of its standalone F1 score across all
datasets. This demonstrates the effectiveness of the Classifier and
Rephraser Agents in resolving contextual dependencies with min-
imal performance loss. The slight drop reflects the inherent ambi-
guity in multi-turn inputs, whichChatty-KGmitigates effectively.
Question Understanding:We evaluateChatty-KG’s ability to un-
derstand questions by analyzing failure cases, as shown in Figure 3.
The figure compares the number of failing questions (Recall=0) for
Chatty-KG(denoted asC) andKGQAn(denoted asK) across four
benchmarks: QALD, YAGO, DBLP, and MAG. Each bar is split into
K C K C K C K C K C0.02.55.07.510.012.515.017.520.0Average Response Time (s)4.0 3.7
QALD5.0
3.3
LC-QuAD4.13.5
YAGO8.3
3.1
DBLP19.4
3.1
MAGE&F
Linking
QUFigure 4: Average response time per question (in seconds)
forKGQAn(K) andChatty-KG(C). Each bar is segmented
bottom-up into three stages: Question Understanding (QU),
Linking, and Execution & Filtration (E&F). Shading variations
within each bar indicate the contribution of each stage.
QALD LC-QuAD YAGO DBLP MAG0246810121416Average # of queries14.52
12.21
8.5611.2812.58
2.194.62
1.101.35 1.25KGQAn
Chatty-KG
Figure 5: Average number of queries per question forKGQAn
andChatty-KG. Lower values indicate better query plan-
ning, while alsoChatty-KGachieves higher F1, see Table 3.
two segments: failures due toquestion understanding (QU)and
failures due toother reasons. Darker shades indicate QU-related
failures, while lighter shades capture all other causes. Across all
benchmarks,Chatty-KG(C) shows significantly fewer failures
thanKGQAn(K), especially in question understanding. For exam-
ple, in the YAGO and MAG datasets,Chatty-KGfails due to QU
in only one case, compared to eight and four cases, respectively,
forKGQAn. This demonstrates thatChatty-KG’s LLM-based mod-
ular agents are more robust in interpreting diverse and complex
linguistic questions, outperformingKGQAn’s fine-tuned model.
Improved question understanding helps downstream accuracy,
but gains are not always linear because errors can still happen in
later stages. The remaining errors (lighter shades in Figure 3) come
from structural mismatches, entity linking, and query selection.
In QALD-9, many failures stem from mismatches between natu-
ral language and KG structure. For example,“How many emperors
did China have?”should link toEmperor of China, notChina. Like-
wise,“Who killed Caesar?”should retrieveAssassins of Julius Caesar,
not justCaesar. Some questions also rely on vague KG predicates
(e.g.,subject), which do not clearly capture the intended semantics.
In these cases, question parsing is correct, but the KG organizes
knowledge differently than natural language. Entity linking errors
occur when the model chooses the wrong candidate due to am-
biguity or similar entity names. For example, in “Are Taiko some
kind of Japanese musical instrument?”, the system links to http://

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
Table 6: Results of runningChatty-KGwith the translation
extension module on the 7 languages of QALD-9.
Lang en de pt hi fr nl it
P62.29 66.31 63.62 65.20 64.21 65.80 62.97
R40.83 42.16 42.14 39.36 37.72 36.40 37.14
F149.33 50.79 50.70 49.08 47.52 46.87 46.72
Table 7: Human and model evaluation across quality dimen-
sions, including inter-annotator agreement.
Metric ChatGPT Gemini H1 H2 H3 Avg Std
Response Quality4.48 4.64 4.50 4.43 4.22 4.45 0.15
Fluency4.92 4.86 4.88 4.76 4.86 4.86 0.06
Dialogue Coherence4.52 4.52 4.50 4.45 4.57 4.51 0.04
Weighted Cohen’s𝜅(Human–Human): 0.53 (Moderate Agreement)
Weighted Cohen’s𝜅(Human–LLM): 0.56 (Moderate Agreement)
dbpedia.org/resource/Japanese_musical_instrument instead of http:
//dbpedia.org/class/yago/WikicatJapaneseMusicalInstruments.
Query selection errors arise when the agent either rejects all
queries or picks one with the wrong predicate. For instance, in “How
many grand-children did Jacques Cousteau have?”, it selectsrelative
instead ofchild. These errors can cascade: a wrong entity link
weakens query candidates, causing incorrect rejection or ranking.
Chatty-KG retries up to three times to reduce such failures while
controlling latency and cost. Despite these challenges, it still cuts
both overall errors and QU-specific errors by a large margin.
System Efficiency:We evaluateChatty-KG’s runtime efficiency
againstKGQAnbased on average response time and average num-
ber of executed queries. Both are critical for interactive systems
where fast and precise answers are expected. Results are shown
in Figures 4 and 5. Figure 4 breaks down response time into three
stages: Question Understanding (QU), Linking, and Execution & Fil-
tration (E&F).Chatty-KG(C) consistently outperformsKGQAn(K)
in all stages, with the largest gains in the E&F phase. For example,
on MAG,KGQAntakes 19.4 seconds per question, whileChatty-
KGcompletes in just 3.1 seconds. This demonstratesChatty-KG’s
scalability to large graphs. Figure 5 shows thatChatty-KGexecutes
up to 10×fewer queries thanKGQAn. This reduction lowers re-
sponse time and improves precision by avoiding irrelevant queries,
as shown in Figures 4 and 5. Beyond response time, we also mea-
sured cost on QALD-9 (150questions). RunningChatty-KGwith
GPT-4o used507requests,326K input tokens, and7 .73K output to-
kens, for a total cost of$0 .88. This shows thatChatty-KGremains
cost-effective even with multiple LLM calls.
7.5 AnalyzingChatty-KGand Discussion
Multilingual Supportis enabled by adding an LLM translation
module to the Chat Agent, which converts non-English questions
to English before processing. This allows a single pipeline, since
the KG and SPARQL entities are in English. On QALD-9,Chatty-
KGyields consistent scores across 7 languages (Table 6), with F1
ranging from46.72(it) to50.79(de) vs.49.33(en). Some languages
outperform English as translation often clarifies entity mentions,
improving linking, while lower scores reflect translation quality
variance. We analyzed 11 languages in total16.
16Seesupplementary materials for more details and examples.Table 8: Number of questions solved by CHatty-KG across
different question types in KGQA benchmarks. Results are
shown as system solved/total questions format.
Benchmark Factoid Count Boolean Q Len Preds
QALD-9 63/133 2/13 0/4 7.52 1.77
YAGO-B 77/99 - 0/1 6.97 2.00
DBLP-B 70/92 7/8 - 9.73 1.72
MAG-B 58/81 16/17 2/2 8.63 1.87
DBpedia-Dia 34/64 0/11 6/19 7.10 1.00
YAGO-Dia 48/65 0/3 10/18 8.21 1.02
DBLP-Dia 48/59 2/4 17/27 11.62 1.01
Wikidata-Dia 36/61 1/5 15/25 9.05 1.03
Human and LLM Evaluation:To assess conversational quality,
we usedfive evaluators: three human annotators and two LLMs
(ChatGPT,Gemini).17Evaluators rated each response onResponse
QualityandFluency, and scored the full dialogue forDialogue Coher-
ence. All scores used a 1–5 scale. As shown in Table 7,Chatty-KG
achieved strong results:4.45in Response Quality,4.86in Fluency,
and4.51in Dialogue Coherence, with low variance. Human–human
agreement reached 𝜅= 0.53, and human–LLM agreement 𝜅= 0.56,
indicatingmoderate agreement. This reflects consistent judgments
across annotators and reliable quality ratings.
Question Type Analysis:KGQA systems are best for fact-based
queries (who, what, where, when, count) and are less suitable for
open-ended ones that need external reasoning (why, how). We
group questions into three types:Factoid(entity or literal answers),
Count(requires counting), andBoolean(yes/no verification).18
Table 8 reportsChatty-KG’s performance across these types. Fac-
toid questions dominate benchmarks andChatty-KGsolves most
of them across datasets. Count and Boolean accuracy varies by
dataset, as these tasks are more sensitive to query structure and
KG coverage. We report average question length (no. of words,Q
Len) and average predicates per query (Preds). Dialogue datasets
use fewer predicates since context carries across turns, reducing
the need to explicitly specify all relations in each query.
Training-Free Prompting:Chatty-KG uses task-aware prompt-
ing rather than training. Simple agents use zero-shot prompts (e.g.,
Rephraser), while more complex agents use few-shot prompts (e.g.,
QIR) and chain-of-thought when needed. This avoids dataset cre-
ation and fine-tuning, enabling fast deployment. Although light
fine-tuning could help, it requires task- and KG-specific data for
each module, reducing flexibility on new or evolving graphs. Em-
pirically, prompting alone outperforms KGQAn (which fine-tunes
for question understanding) as shown in Figure 3. Prompting of-
fers scalability, cross-KG generalization, and practical deployment
without specialized training data.
8 Conclusion
Conversational QA over KGs enables accurate and timely access
to enterprise and domain knowledge. Existing systems remain lim-
ited, mostly single-turn and dependent on heavy preprocessing or
fine-tuning. General LLMs also lack grounding in structured data,
17Full protocol and examples are in the supplementary materials..
18More details are in the supplementary materials..

Conference’17, July 2017, Washington, DC, USA Reham Omar, Abdelghny Orogat, Ibrahim Abdelaziz, Omij Mangukiya, Panos Kalnis, and Essam Mansour
reducing reliability on private or evolving graphs.Chatty-KGcom-
bines structured reasoning with LLM dialogue through a modular
multi-agent design for multi-turn KGQA. It uses LLM-powered
agents for context tracking, linking, and query planning without
KG-specific training, enabling domain transfer, coherent dialogue,
and efficient SPARQL generation across diverse graphs.Key learn-
ings: (1) Modular agents allow incremental improvements without
system-wide changes. (2) A shared state enables coordinated exe-
cution. (3) Agent-level validation reduces hallucination by catching
errors early. (4) Bounded retries prevent infinite loops when tasks
exceed LLM capability. (5) The multi-agent design adds little latency
and debugging becomes more complex. These choices deliver accu-
racy, adaptability, and practical performance for real-world KGQA.
References
[1]Marah I Abdin, Jyoti Aneja, Harkirat S. Behl, Sébastien Bubeck, Ronen Eldan, and
et al. 2024. Phi-4 Technical Report.CoRRabs/2412.08905 (2024). arXiv:2412.08905
https://doi.org/10.48550/arXiv.2412.08905
[2]AI@Meta. 2024. Llama 3.1 Model Card. https://github.com/meta-llama/llama-
models/blob/main/models/llama3_1/MODEL_CARD.md
[3]Diego Ceccarelli, Claudio Lucchese, Salvatore Orlando, Raffaele Perego, and
Salvatore Trani. 2014. Dexter 2.0 - an Open Source Tool for Semantically Enriching
Data. InProceedings of the International Semantic Web Conference, Posters &
Demonstrations Track (ISWC), Vol. 1272. 417–420. http://ceur-ws.org/Vol-1272/
paper_127.pdf
[4]Philipp Christmann, Rishiraj Saha Roy, and Gerhard Weikum. 2022. Beyond NED:
Fast and Effective Search Space Reduction for Complex Question Answering over
Knowledge Bases. InWSDM: The ACM International Conference on Web Search
and Data Mining. 172–180. https://doi.org/10.1145/3488560.3498488
[5]Philipp Christmann, Rishiraj Saha Roy, and Gerhard Weikum. 2022. Conversa-
tional Question Answering on Heterogeneous Sources. InSIGIR: The International
ACM SIGIR Conference on Research and Development in Information Retrieval. 144–
154. https://doi.org/10.1145/3477495.3531815
[6]Philipp Christmann, Rishiraj Saha Roy, and Gerhard Weikum. 2023. Explainable
Conversational Question Answering over Heterogeneous Sources via Iterative
Graph Neural Networks. InProceedings of the International ACM SIGIR Conference
on Research and Development in Information Retrieval, SIGIR. 643–653. https:
//doi.org/10.1145/3539618.3591682
[7] DBLP release. 2022. https://dblp.org/rdf/release/dblp-2022-06-01.nt.gz.
[8]DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, and et al. 2024.
DeepSeek-V3 Technical Report.CoRRabs/2412.19437 (2024). arXiv:2412.19437
https://arxiv.org/abs/2412.19437
[9]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
Proceedings of the Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, (NAACL-HLT).
4171–4186. https://doi.org/10.18653/v1/n19-1423
[10] Dennis Diefenbach, Kamal Deep Singh, and Pierre Maret. 2018. WDAqua-core1: A
Question Answering service for RDF Knowledge Bases. InCompanion Proceedings
of the The Web Conference, (ACM). 1087–1091. https://doi.org/10.1145/3184558.
3191541
[11] YAGO downloads. 2022. https://yago-knowledge.org/downloads/yago-4.
[12] Mohnish Dubey, Debayan Banerjee, Debanjan Chaudhuri, and Jens Lehmann.
2018. EARL: Joint Entity and Relation Linking for Question Answering over
Knowledge Graphs. InProceedings of the International Semantic Web Conference
(ISWC), Vol. 11136. 108–126. https://doi.org/10.1007/978-3-030-00671-6_7
[13] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization.arXiv preprint arXiv:2404.16130(April 2024). https://doi.org/10.
48550/arXiv.2404.16130 arXiv:2404.16130 [cs.CL] v2 revised 19 Feb 2025.
[14] Sen Hu, Lei Zou, Jeffrey Yu, Haixun Wang, and Dongyan Zhao. 2018. Answering
Natural Language Questions by Subgraph Matching over Knowledge Graphs.
IEEE Transactions on Knowledge and Data Engineering, (TKDE)30 (2018), 824–837.
https://doi.org/10.1109/TKDE.2017.2766634
[15] Xixin Hu, Yiheng Shu, Xiang Huang, and Yuzhong Qu. 2021. EDG-Based Question
Decomposition for Complex Question Answering over Knowledge Bases. In
Proceedings of the International Semantic Web Conference, (ISWC). 128–145. https:
//doi.org/10.1007/978-3-030-88361-4_8
[16] IBM Research. 2024. Granite Foundation Models. https://www.ibm.com/
downloads/documents/us-en/10a99803c92fdb35[17] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InProceedings of
the Conference of the European Chapter of the Association for Computational
Linguistics: EACL. 874–880. https://doi.org/10.18653/V1/2021.EACL-MAIN.74
[18] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Deven-
dra Singh Chaplot, and et al. 2023. Mistral 7B.CoRRabs/2310.06825 (2023).
arXiv:2310.06825 https://doi.org/10.48550/arXiv.2310.06825
[19] Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Mer-
hej, and et al. 2025. Gemma 3 Technical Report.CoRRabs/2503.19786 (2025).
arXiv:2503.19786 https://doi.org/10.48550/arXiv.2503.19786
[20] Pavan Kapanipathi, Ibrahim Abdelaziz, Srinivas Ravishankar, Salim Roukos, and
Alexander G. Gray et al. 2021. Leveraging Abstract Meaning Representation for
Knowledge Base Question Answering. InFindings of the Association for Compu-
tational Linguistics: (ACL/IJCNLP). 3884–3894. https://doi.org/10.18653/v1/2021.
findings-acl.339
[21] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. InProceedings of the
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 39–48.
[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large Language Models are Zero-Shot Reasoners. InAdvances in
Neural Information Processing Systems : Annual Conference on Neural Information
Processing Systems (NeurIPS). http://papers.nips.cc/paper_files/paper/2022/hash/
8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html
[23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, and et
al. 2023. Efficient Memory Management for Large Language Model Serving with
PagedAttention. InProceedings of Symposium on Operating Systems Principles,
SOSP. 611–626. https://doi.org/10.1145/3600006.3613165
[24] Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, and
et al. 2015. DBpedia - A large-scale, multilingual knowledge base extracted from
Wikipedia.Semantic Web6 (2015), 167–195. https://doi.org/10.3233/SW-140134
[25] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman
Mohamed, and et al. 2020. BART: Denoising Sequence-to-Sequence Pre-training
for Natural Language Generation, Translation, and Comprehension. InProceed-
ings of the Annual Meeting of the Association for Computational Linguistics, ACL.
7871–7880. https://doi.org/10.18653/V1/2020.ACL-MAIN.703
[26] Xue Li and Till Döhmen. 2024. Towards efficient data wrangling with llms using
code generation. InProceedings of the Eighth Workshop on Data Management for
End-to-End Machine Learning. 62–66.
[27] LMSYS Org. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%
ChatGPT Quality. https://lmsys.org/blog/2023-03-30-vicuna/
[28] MAG records. 2022. https://zenodo.org/record/4617285#.YrNszNLMJhH.
[29] Yohei Nakajima. 2023. Task-driven autonomous agent. https://github.com/
yoheinakajima/babyagi
[30] Reham Omar, Ishika Dhall, Panos Kalnis, and Essam Mansour. 2023. A Universal
Question-Answering Platform for Knowledge Graphs.Proceedings of the ACM
on Management of Data1, 1 (2023), 57:1–57:25. https://doi.org/10.1145/3588911
[31] Reham Omar, Omij Mangukiya, and Essam Mansour. 2025. Dialogue Benchmark
Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented
LLMs.Proceedings of the ACM on Management of Data3, 1 (2025), 31:1–31:26.
https://doi.org/10.1145/3709681
[32] OpenAI. 2024. GPT-4o. (2024). https://openai.com/index/hello-gpt-4o/
[33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, and
et al. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-
Text Transformer.Journal of machine learning research21 (2020), 140:1–140:67.
https://jmlr.org/papers/v21/20-074.html
[34] Morgane Rivière, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya
Bhupatiraju, and et al. 2024. Gemma 2: Improving Open Language Models at a
Practical Size.CoRRabs/2408.00118 (2024). arXiv:2408.00118 https://doi.org/10.
48550/arXiv.2408.00118
[35] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, and et
al. 2023. Code Llama: Open Foundation Models for Code.CoRRabs/2308.12950
(2023). arXiv:2308.12950 https://doi.org/10.48550/arXiv.2308.12950
[36] Ahmad Sakor, Isaiah Mulang, Kuldeep Singh, Saeedeh Shekarpour, and Maria-
Esther Vidal et al. 2019. Old is Gold: Linguistic Driven Approach for Entity and
Relation Linking of Short Text. InProceedings of the Conference of the North Amer-
ican Chapter of the Association for Computational Linguistics: Human Language
Technologies, (NAACL-HLT). 2336–2346. https://doi.org/10.18653/v1/n19-1243
[37] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies.
3715–3734.
[38] Liang Shi, Zhengju Tang, Nan Zhang, Xiaotong Zhang, and Zhi Yang. 2025. A
Survey on Employing Large Language Models for Text-to-SQL Tasks.ACM
Comput. Surv.(2025). https://doi.org/10.1145/3737873
[39] Significant Gravitas. 2023. Auto-GPT: An Autonomous GPT-4 Experiment. https:
//github.com/Significant-Gravitas/Auto-GPT

Chatty-KG: A Multi-Agent AI System for On-Demand Conversational Question Answering over Knowledge Graphs Conference’17, July 2017, Washington, DC, USA
[40] Ion Stoica, Matei Zaharia, Joseph Gonzalez, Ken Goldberg, Koushik Sen, and et
al. 2024. Specifications: The missing link to making the development of LLM
systems an engineering discipline.CoRRabs/2412.05299 (2024). arXiv:2412.05299
https://doi.org/10.48550/arXiv.2412.05299
[41] Thomas Tanon, Gerhard Weikum, and Fabian Suchanek. 2020. YAGO 4: A Reason-
able Knowledge Base. InProceedings of the European Semantic Web Conference,
(ESWC), Vol. 12123. 583–596. https://doi.org/10.1007/978-3-030-49461-2_34
[42] Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across
millions of tokens of context.CoRRabs/2403.05530 (2024). arXiv:2403.05530
https://doi.org/10.48550/arXiv.2403.05530
[43] Priyansh Trivedi, Gaurav Maheshwari, Mohnish Dubey, and Jens Lehmann. 2017.
LC-QuAD: A Corpus for Complex Question Answering over Knowledge Graphs.
InProceedings of the International Semantic Web Conference (ISWC), Vol. 10588.
210–218. https://doi.org/10.1007/978-3-319-68204-4_22
[44] Ricardo Usbeck, Ria Gusmita, Axel-Cyrille Ngomo, and Muhammad Saleem.
2018. 9th Challenge on Question Answering over Linked Data (QALD-9). In
Joint proceedings of the Workshop on Semantic Deep Learning (SemDeep-4) and
NLIWoD4: Natural Language Interfaces for the Web of Data (NLIWOD-4) and
9th Question Answering over Linked Data challenge (QALD-9) co-located with
International Semantic Web Conference (ISWC), Vol. 2241. 58–64. http://ceur-
ws.org/Vol-2241/paper-06.pdf
[45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, and Llion Jones
et al. 2017. Attention is All you Need.In Proceedings of the Advances in neural
information processing systems (NeurIPS), 5998–6008. https://proceedings.neurips.
cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
[46] Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative
knowledgebase.Commun. ACM57, 10 (2014), 78–85. https://doi.org/10.1145/
2629489
[47] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang,
Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on largelanguage model based autonomous agents.Frontiers of Computer Science18, 6
(2024).
[48] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng,
and Heng Ji. 2024. Executable code actions elicit better llm agents. InForty-first
International Conference on Machine Learning.
[49] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian
Ichter, and et al. 2022. Chain-of-Thought Prompting Elicits Reason-
ing in Large Language Models. InAdvances in Neural Information Pro-
cessing Systems: Annual Conference on Neural Information Processing
Systems (NeurIPS). http://papers.nips.cc/paper_files/paper/2022/hash/
9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html
[50] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, and et al.
2023. OpenAgents: An Open Platform for Language Agents in the Wild.CoRR
abs/2310.10634 (2023). arXiv:2310.10634 https://arxiv.org/abs/2310.10634
[51] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, and et al.
2024. Qwen2.5 Technical Report.CoRRabs/2412.15115 (2024). arXiv:2412.15115
https://doi.org/10.48550/arXiv.2412.15115
[52] Ke Yang, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik Chaudhari, George
Karypis, and Huzefa Rangwala. 2024. Agentoccam: A simple yet strong baseline
for llm-based web agents.arXiv preprint arXiv:2410.13825(2024).
[53] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, and et al. 2023. ReAct:
Synergizing Reasoning and Acting in Language Models. InThe International
Conference on Learning Representations, ICLR. https://openreview.net/forum?id=
WE_vluYUL-X
[54] Lei Zou, Ruizhe Huang, Haixun Wang, Jeffrey Yu, and Wenqiang He et al. 2014.
Natural language question answering over RDF: a graph data driven approach.
InProceedings of the International Conference on Management of Data, (SIGMOD).
313–324. https://doi.org/10.1145/2588555.2610525