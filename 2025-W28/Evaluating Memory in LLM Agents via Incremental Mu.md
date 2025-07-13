# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley

**Published**: 2025-07-07 17:59:54

**PDF URL**: [http://arxiv.org/pdf/2507.05257v1](http://arxiv.org/pdf/2507.05257v1)

## Abstract
Recent benchmarks for Large Language Model (LLM) agents primarily focus on
evaluating reasoning, planning, and execution capabilities, while another
critical component-memory, encompassing how agents memorize, update, and
retrieve long-term information-is under-evaluated due to the lack of
benchmarks. We term agents with memory mechanisms as memory agents. In this
paper, we identify four core competencies essential for memory agents: accurate
retrieval, test-time learning, long-range understanding, and conflict
resolution. Existing datasets either rely on limited context lengths or are
tailored for static, long-context settings like book-based QA, which do not
reflect the interactive, multi-turn nature of memory agents that incrementally
accumulate information. Furthermore, no existing benchmarks cover all four
competencies. Therefore, we introduce MemoryAgentBench, a new benchmark
specifically designed for memory agents. Our benchmark combines reformulated
existing datasets with newly constructed ones, covering the above four memory
competencies, providing a systematic and challenging testbed for assessing
memory quality. We evaluate a diverse set of memory agents, ranging from simple
context-based and retrieval-augmented generation (RAG) systems to advanced
agents with external memory modules and tool integration. Empirical results
reveal that current methods fall short of mastering all four competencies,
underscoring the need for further research into comprehensive memory mechanisms
for LLM agents.

## Full Text


<!-- PDF content starts -->

arXiv:2507.05257v1  [cs.CL]  7 Jul 2025Evaluating Memory in LLM Agents via Incremental
Multi-Turn Interactions
Yuanzhe Hu1∗, Yu Wang1∗, Julian McAuley1
1UC San Diego
1{yuh127,yuw164,jmcauley}@ucsd.edu
Datasets
 Source Code
Abstract
Recent benchmarks for Large Language Model (LLM) agents primarily focus on
evaluating reasoning, planning, and execution capabilities, while another critical
component—memory, encompassing how agents memorize, update, and retrieve
long-term information—is under-evaluated due to the lack of benchmarks. We
term agents with memory mechanisms as memory agents . In this paper, we
identify four core competencies essential for memory agents: accurate retrieval,
test-time learning, long-range understanding, and conflict resolution. Existing
datasets either rely on limited context lengths or are tailored for static, long-
context settings like book-based QA, which do not reflect the interactive, multi-turn
nature of memory agents that incrementally accumulate information. Furthermore,
no existing benchmarks cover all four competencies. Therefore, we introduce
MemoryAgentBench , a new benchmark specifically designed for memory agents.
Our benchmark combines reformulated existing datasets with newly constructed
ones, covering the above four memory competencies, providing a systematic
and challenging testbed for assessing memory quality. We evaluate a diverse set
of memory agents, ranging from simple context-based and retrieval-augmented
generation (RAG) systems to advanced agents with external memory modules
and tool integration. Empirical results reveal that current methods fall short of
mastering all four competencies, underscoring the need for further research into
comprehensive memory mechanisms for LLM agents.
1 Introduction
Large Language Model (LLM) agents have rapidly transitioned from proof-of-concept chatbots to
end-to-end systems that can write software [ 42], control browsers [ 34], and reason over multi-modal
inputs. Frameworks such as MANUS ,OWL [16],OPENHANDS [42], and CODEX routinely solve
complex, tool-rich tasks and achieve state-of-the-art results on agentic benchmarks like GAIA [ 31]
and SWE-Bench [ 18]. Yet these evaluations focus almost exclusively on reasoning (planning, tool
using, code synthesis) and leave the equally important question of memorization (abstraction, storing,
updating, retrieving) largely under-explored.
Recent memory-centric architectures—ranging from parametric memory systems like Memo-
ryLLM [ 44], SELF-PARAM [ 43], and M+[ 46] to commercial token-level memory solutions such as
MEMGPT [36,27],MEM0[7],COGNEE , and ZEP[37]—employ diverse strategies for storing and
retrieving past information. Despite growing interest, their real-world effectiveness remains largely
anecdotal, and there is currently no unified benchmark for systematically evaluating the quality of
memory in agents. In this paper, we refer to agents equipped with memory mechanisms as Memory
Agents , where memory can take various forms, including parameters, vectors, textual histories, or
external databases. In this paper, we primarily focus on memory agents that utilize textual histories
∗Y . Hu and Y . Wang contribute equally.
Preprint. Under review.

Figure 1: Four complementary competencies that memory agents should have.
and external databases, as these approaches are most commonly deployed in real-world applications.
In contrast, memory encoded in model parameters [ 44,46,51] remains largely within academic
research and is typically less capable than proprietary memory systems equipped on closed-sourced
API models.
To evaluate memory agents, we identify four complementary competencies (Examples shown in
Figure 1): (1) Accurate Retrieval : The ability to extract the correct snippet in response to a query.
This can involve one-hop or multi-hop retrieval, as long as the relevant information can be accessed
with a single query. (2) Test-Time Learning : The capacity to incorporate new behaviors or acquire
new skills during deployment, without additional training. (3) Long-Range Understanding : The
ability to integrate information distributed across extended contexts ( ≥100k tokens) and answer
questions requiring a global understanding of the entire sequence. (4) Conflict Resolution : The skill
to revise, overwrite, or remove previously stored information when faced with contradictory evidence,
aligning with goals in model editing and knowledge unlearning tasks [30, 45].
Previous datasets developed to evaluate memory in language models have notable limitations. Early
benchmarks such as LOCOMO [29] (∼9k tokens), LooGLE[ 23] (∼24k tokens), and LongBench[ 3]
(∼20k tokens) feature relatively short contexts that no longer challenge current models. More recent
datasets like NovelQA[ 40] (∼200k tokens), NOCHA[ 19] (∼127k tokens), Loong[ 41] (∼100k tokens),
and∞-Bench[ 53] (∼150k tokens) extend the context length to evaluate global reasoning and retrieval
capabilities. However, these datasets were primarily designed for evaluating long-context language
models rather than memory agents. The reason that long-context benchmarks cannot be directly
used to evaluate memory agents is as follows. There is a fundamental distinction between memory
and long context: memory serves as a compressed and distilled representation of past information.
Rather than storing all historical content verbatim, memory selectively extracts salient details,
removes irrelevant information, and often incorporates new inferences derived from prior experiences.
Consequently, memory agents are designed to process context incrementally —absorbing input
piece by piece, abstracting and consolidating information over time, generating new inferences, and
learning novel rules from accumulated history. For this reason, datasets that provide the entire context
in a single block are not directly applicable to evaluating memory agents. A more recent effort,
LONG MEMEVAL [48], seeks to address this limitation by using synthetic long-form conversations,
which can be injected into memory gradually, session by session. Nonetheless, its evaluation
framework remains constrained by limited topical diversity and less realistic interaction patterns,
reducing its applicability to real-world memory agent scenarios.
To address these limitations, we introduce a unified benchmark framework, MemoryAgentBench ,
specifically designed to evaluate a broad spectrum of memory mechanisms in agent systems. We
also provide a framework for memory agent evaluation. In this framework, agents are presented with
sequences of textual inputs that simulate multi-turn interactions with users. We repurpose existing
2

datasets originally developed for long-context LLM evaluation by segmenting their inputs into
multiple chunks and feeding them incrementally to the agent. However, since these datasets do not
fully capture all four targeted memory competencies, we also introduce two new datasets: EventQA
andFactConsolidation , designed to evaluate accurate retrieval and conflict resolution, respectively.
Our benchmark includes evaluations of state-of-the-art commercial memory agents (such as Mem0
and MemGPT), long-context agents that treat the full input as memory, and RAG agents that extend
their memory through retrieval methods. We examine how techniques developed for long-context
models and RAG transfer to the memory agent setting, and how commercial memory agents perform
under more challenging, competency-specific tests. By providing a consistent evaluation protocol
across diverse agent architectures and datasets, MemoryAgentBench delivers comprehensive insights
into agent performance across the four core memory competencies.
Our contributions are summarized as follows:
•Datasets: We re-structure existing datasets and create two new datasets to construct a comprehen-
sive benchmark, covering four distinct memory competencies.
•Framework: We provide a unified evaluation framework, and open-source the codebase and
datasets to encourage reproducibility and further research.
•Empirical Study: We implement various simple agents with diverse memory mechanisms, adopt
commercial agents, and evaluate these agents on our proposed benchmark. With our results, we
show that existing memory agents, while effective in some tasks, still face significant challenges on
some aspects.
2 Related Work
2.1 Benchmarks with Long Input
In this section, we review prior work on long-context benchmarks. Early benchmarks designed
for long-context evaluation include LongBench[ 3] and LooGLE[ 23], with average input lengths of
approximately 20k and 24k tokens, respectively. More recent benchmarks—such as ∞-Bench [ 53],
HELMET[ 50], RULER[ 15], NOCHA[ 19], NoLiMa [ 33] and LongBench V2[ 4]—extend context
lengths to over 100k tokens and are primarily intended to evaluate the capabilities of long-context
models. However, despite their scale, these benchmarks are not designed to assess memory agents,
and no prior work has repurposed them for that goal. More recently, LOCOMO [ 29] and Long-
MemEval [ 48] have been proposed specifically for evaluating memory agents. While promising,
LOCOMO still features relatively short conversations ( ∼9k), and LongMemEval uses synthetic
conversations with limited topical diversity, making the dialogues less realistic and potentially less
representative of real-world memory use cases.
2.2 Agents with Memory Mechanisms
Memory mechanisms are attracting more and more attention lately [ 47]. Recent advancements in
LLMs have demonstrated the capability to process extended context lengths, ranging from 100K to
over 1 million tokens. For instance, models such as GPT-4o [ 35] and Claude 3.7 [ 1] can handle inputs
of approximately 100K to 200K tokens, while models like Gemini 2.0 Pro [ 8] and the GPT-4.1 series
extend this capacity beyond 1 million tokens. These strong long-context capabilities enable a simple
yet effective form of memory: storing information directly within the context window. However, this
approach is inherently constrained by a hard limit—once the context window is exceeded, earlier
information must be discarded.
In parallel, RAG continues to serve as a dominant paradigm for managing excessive context. By
retrieving relevant information from earlier context and feeding it to the LLM, RAG allows systems to
overcome context length limitations. For example, OpenAI’s recent memory functionality2combines
explicit user preference tracking with retrieval-based methods that reference prior interactions. RAG
methods can be broadly classified into three categories: 1. Simple RAG : These methods rely on string-
matching techniques such as TF-IDF, BM25 [ 38], and BMX [ 25], which are entirely non-neural and
operate on string-level similarity. 2. Embedding-based RAG : This class leverages neural encoders,
2https://openai.com/index/memory-and-new-controls-for-chatgpt/
3

primarily transformers, to map text into dense vector representations [ 49]. Early methods like
DPR [ 20] and Contriever [ 17] are based on BERT [ 9], while more recent models such as NV-Embed-
v2 [22] utilize decoder-only backbones and achieve significantly improved retrieval performance. 3.
Structure-Augmented RAG : These approaches enhance retrieval with structural representations
such as graphs or trees. Representative systems include GraphRAG [ 10], RAPTOR [ 39], HippoRAG-
V2 [ 12], Cognee, Zep [ 37], and Mem0 [ 7], where Mem0 also offers a graph-augmented variant,
Mem0g, built on structured factual knowledge. Despite their effectiveness, RAG-based methods face
challenges with ambiguous queries, multi-hop reasoning, and long-range comprehension. When
questions require integrating knowledge across an entire session or learning from long, skill-encoding
inputs, the retrieval mechanism—limited to the top-k most relevant passages—may fail to surface
the necessary information. To address these limitations, Agentic Memory Agents introduce an
iterative, decision-driven framework. Rather than relying on a single-pass retrieval, these agents
dynamically process the query, retrieve evidence, reflect, and iterate through multiple retrieval and
reasoning cycles. Examples include MemGPT [ 36], Self-RAG [ 2], and Auto-RAG [ 52]. This agentic
design is particularly effective for resolving ambiguous or multi-step queries. Nonetheless, these
methods remain fundamentally constrained by the limitations of RAG—namely, the inability to fully
understand or learn from long-range context that is inaccessible via retrieval alone.
3 MemoryAgentBench
3.1 Aspects of the Evaluation
The evaluation of memory agents encompasses the following key dimensions:
Accurate Retrieval (AR) The task of accurately retrieving information has been extensively
explored in prior work. In the domain of long-context modeling, the Needle-in-a-Haystack (NIAH)
task is widely used to evaluate a model’s ability to locate a specific value based on a given key
within a lengthy input. Extensions such as multi-value NIAH further test the model’s capacity to
retrieve multiple values scattered across the input context. In the RAG setting, this corresponds to
document-based QA, where the model must identify and extract relevant snippets from one or more
documents to answer a query. These snippets might reside in a single location or be distributed across
multiple documents. In this paper, we focus on agentic settings, where the “long context” or “multiple
documents” become long-form conversations. We define Accurate Retrieval (AR) as the ability of an
agent to identify and retrieve important information that may be dispersed throughout a long dialogue
history.
Test-Time Learning (TTL) An essential capability for real-world agents is the ability to acquire
new skills dynamically through interaction with users. This mirrors the concept of In-Context
Learning (ICL) in LLMs, where the model learns from a prompt containing a small number of
examples, often framed as few-shot classification tasks. Ideally, performance improves with additional
examples in the prompt. In the conversational agent setting, prompts are replaced by dialogue histories.
We define Test-Time Learning (TTL) as the agent’s ability to learn to perform new tasks directly from
the conversation. This property is crucial for enabling self-evolving agents that can continuously
adapt and improve in real-world deployments.
Long-Range Understanding (LRU) Long-range understanding refers to the agent’s ability to form
abstract, high-level comprehension over extended conversations. For example, when a user narrates a
long story, the agent should retain the content and derive a holistic understanding rather than just
recall isolated facts. We define Long-Range Understanding (LRU) as the ability to reason about
long-form inputs and answer high-level questions that require an understanding of the overall content,
rather than detailed recall. An example question might be: “Summarize the main experiences of
Harry Potter.”
Conflict Resolution (CR) In long-term interactions, agents often face evolving or conflicting
information—whether about the external world (e.g., changes in political leadership) or user-specific
facts (e.g., a new occupation). This challenge is closely related to model editing [ 30,11] and
knowledge unlearning [ 45], which focus on modifying or removing factual knowledge from language
models. We define Conflict Resolution (CR) as the agent’s ability to detect and resolve contradictions
between existing knowledge and newly acquired information, ensuring the agent remains aligned
with current realities and user states. CR is distinct from Abstractive Retrieval (AR) in two key ways.
(1) Certain questions requiring CR cannot be answered solely through AR. As illustrated in Figure 1,
an agent that retrieves all facts related to pears may fail to identify the updated information in the
4

Table 1: Datasets categorized by the specific aspects of evaluation.
Capability Benchmarks / Tasks # of Sequences # of QAs Avg Len
Accurate
RetrievalRULER-QA [15] 2 200 309K
RULER-NIAH-MQ [15] 1 100 448K
∞Bench-QA [53] 100 100 183K
LongMemEval (S*) [48] 5 300 355K
EventQA ( ours) 5 500 534K
Test-Time
LearningBANKING-77 1 100
CLINC-150 1 100
NLU 1 100 103K
TREC (Coarse) 1 100
TREC (Fine) 1 100
Movie-Rec Redial [14] 1 200 1.44M
Long-Range Understanding ∞Bench-Sum [53] 100 100 172K
Conflict ResolutionFactConsolidation-SH ( ours) 1 100262KFactConsolidation-MH ( ours) 1 100
second message. (2) In AR, earlier messages remain relevant and should be retained, even when
multiple pieces of evidence are required. In contrast, CR involves identifying outdated or incorrect
information and discarding it. That is, AR requires preservation of all related content, whereas CR
requires overwriting prior facts to reflect the most up-to-date truth.
3.2 Dataset Preperation
In this section, we describe how we adopt existing datasets and construct new ones for evaluating
each aspect introduced in Section 3.1. All datasets with their categories are shown in Table 1. We
introduce the details in datasets curation in Appendix A.
Datasets for Accurate Retrieval (AR) We adopt five datasets to evaluate the accurate retrieval ca-
pability of memory agents. Four are adapted from existing benchmarks, and one is newly constructed:
(1)RULER-QA : This is a NIAH-style QA task where a long passage contains single (QA-1) or
multiple (QA-2) snippets answering the input question. The agent must identify and extract relevant
snippets from the extended context. (2) NIAH-MQ : We use the multiple-query (MQ) version of the
NIAH dataset from RULER [ 15], where each query seeks a different numeric value embedded in a
long passage. The agent must accurately retrieve multiple distinct answers. (3) ∞Bench-En.QA :
This task from ∞Bench presents free-form QA questions based on entire books, with all entities
replaced by fictitious names to avoid contamination from model pretraining. Compared to synthetic
datasets like RULER-QA, this benchmark is more realistic and challenging due to the natural narrative
structure of books. (4) LongMemEval : This benchmark evaluates memory agents on long dialogue
histories. Although task types like information extraction (IE) or multi-session reasoning are included,
most tasks can be reformulated as single-retrieval problems requiring agents to retrieve the correct
segments spanning a long multi-turn conversation. Among these, LongMemEval is already formatted
for agent-based evaluation with session separation. We use the original LongMemEval(S) dataset
(∼110K tokens) and reformulated chat history into five long dialogues ( ∼355K tokens) with 300
questions (LongMemEval (S*) in Table 1). We create LongMemEval (S*) specifically for increasing
the number of questions per context, mitigating the exhaustive needs of reconstructing the memory
for each quesiton. (5) EventQA (ours) : We introduce EventQA this reasoning style NIAH task
to evaluate agents’ ability to recall and reason about temporal sequences in long-form narratives.
In this dataset, the agent is required to read a novel and select the correct event from a series of
candidates after receiving up-to five previous events. For these datasets, which are originally designed
for long-context modeling, we split documents into chunks and sequentially inject them into the
agent.
Datasets for Test-Time Learning (TTL) We evaluate TTL via two task categories: (1) Multi-
Class Classification (MCC) : We adopt five classification datasets used in prior TTL work [ 5,50]:
BANKING77 [ 6], CLINC150 [ 21], TREC-Coarse, TREC-Fine [ 26], and NLU [ 28]. Each task
requires the agent to map sentences to class labels, leveraging previously seen labeled examples
in context. (2) Recommendation (Recom) : We use the Redial [ 24] dataset to evaluate movie
5

recommendation via dialogue. Following the setup from He et al. [13], the agent is exposed to
thousands of movie-related dialogue turns and is asked to recommend twenty relevant movies based
on the long interaction history.
Datasets for Long Range Understanding (LRU) We adopt the Summarization task En.Sum from
∞-Bench [ 53]. The agent is required to analyze and organize the plot and characters of the novel,
and then compose a summary of 1000 to 1200 words.
Datasets for Conflict Resolution (CR) To assess whether an agent can consolidate conflicting fac-
tual updates and reason over them, we construct a new dataset called FactConsolidation. Specifically,
We build this benchmark using counterfactual edit pairs from MQUAKE [54]. Each pair contains a
true fact and a rewritten, contradictory version. These are ordered such that the rewritten (new) fact
appears after the original, simulating a realistic update scenario. We concatenate multiple such edit
pairs to create long contexts of length 32K, 64K, 262K. We then adpot MQUAKE’s original questions
and categorize them into: (1) FactConsolidation-SH (Ours) (SH means Single-Hop), requiring
direct factual recall (e.g., “Which country was tool Acreated in?”), and (2) FactConsolidation-MH
(Ours) (MH refers to Multi-Hop), requiring inference over multiple facts (e.g., “What is the location
of death of the spouse of person B?”). Agents are prompted to prioritize later information in case of
conflict and reason based on the final memory state. This setup directly evaluates the strength and
consistency of conflict resolution over long sequences.
3.3 Different Categories of Memory Agents
We evaluate three major types of memory agents that reflect common strategies for handling long-term
information: Long-Context Agents ,RAG Agents , and Agentic Memory Agents . These approaches
differ in how they store, retrieve, and reason over past inputs.
Long Context Agents Modern language models often support extended context windows ranging
from 128K to over 1M tokens. A straightforward strategy for memory is to maintain a context buffer
of the most recent tokens. For example, in a model with a 128K-token limit, the agent concatenates
all incoming chunks until the total exceeds the window size. Once the limit is reached, the earliest
chunks are evicted in a FIFO (first-in, first-out) manner. This agent design relies solely on positional
recency and assumes the model can attend effectively over the current context window.
RAG Agents RAG-based agents address context limitations by storing past information in an
external memory pool and retrieving relevant content as needed. We consider three RAG variants:
(1)Simple RAG Agents : All input chunks are stored as raw text. During inference, a keyword or
rule-based string matching mechanism retrieves relevant passages. (2) Embedding-based RAG Agents :
Each input chunk is embedded and saved. At query time, the agent embeds the query and performs
retrieval using cosine similarity between embeddings. (3) Structure-Augmented RAG Agents : After
ingesting all input chunks, the agent constructs a structured representation (e.g., knowledge graph or
event timeline). Subsequent queries are answered based on this structured memory.
Agentic Memory Agents Agentic memory agents extend beyond static memory stores by employing
agentic loops—iterative reasoning cycles in which the agent may reformulate questions, perform
memory lookups, and update its working memory. These agents are designed to simulate a more
human-like process of recalling, verifying, and integrating knowledge.
3.4 Datasets and Agents Formulation
Datasets Formulation We standardize all datasets into the format: c1, c2,···, cn(chunks),
q1, q2,···, qm(questions), and a1, a2,···, am(answers), where cidenotes the i-th chunk wrapped
to construct a user message with instructions of memorizing the content in a sequential input, and
c1, c2,···, cnrepresents a single conversation. Each chunk is accompanied by instructions prompting
the agent to memorize its contents. Example prompts are provided in Appendix B.1.
When curating datasets like EventQA and FactConsolidation, we deliberately design scenarios where
multiple questions follow a single context. This allows us to probe the model’s memory multiple times
with one sequential injection. For example, in LME (S*), five contexts are paired with 300 questions
(shown in Table 1). This design choice reflects a key trend: as LLMs support increasingly long
context windows and memory agents become more capable of handling extended inputs, evaluation
datasets must also scale accordingly. Injecting 1M tokens for just one question is resource-inefficient,
whereas associating the same input with many questions provides significantly higher utility.
6

Table 2: Overall Performance Comparison. All RAG agents and commercial memory agents use
GPT-4o-mini as the backbone. Thus we highlight the performance of GPT-4o-mini as the reference.
FactCon-SH and FactCon-MH mean FactConsolidation Single Hop and FactConsolidation Multi
Hop, respectively. We use the NV-Embed-v2 as dense retriever based on the open-source code of
HippoRAG-v2.
AR TTL LRU CR
Agent Type RULER-QA NIAH-MQ ∞Bench-QA LME(S*) EventQA MCC Recom ∞Bench-Sum FactCon-SH FactCon-MH
Long-Context Agents
GPT-4o 61.5 25.0 55.4 32.0 77.2 87.6 12.3 32.2 60.0 5.0
GPT-4o-mini 53.5 22.8 44.9 30.7 59.0 82.4 15.1 28.9 45.0 5.0
GPT-4.1-mini 74.5 94.8 45.8 55.7 82.6 75.6 16.7 41.9 36.0 5.0
Gemini-2.0-Flash 73.0 83.8 53.2 47.0 67.2 84.0 8.7 23.9 30.0 3.0
Claude-3.7-Sonnet 65.0 38.0 50.6 34.0 74.6 89.4 18.3 52.5 43.0 2.0
GPT-4o-mini 53.5 22.8 44.9 30.7 59.0 82.0 15.1 28.9 45.0 5.0
Simple RAG Agents
BM25 61.0 100.0 45.6 45.3 74.6 75.4 13.6 20.9 56.0 3.0
Embedding RAG Agents
Contriever 26.5 2.5 38.1 15.7 66.8 70.6 15.2 21.2 18.0 7.0
Text-Embed-3-Small 52.0 7.2 44.4 48.3 63.0 70.0 15.3 25.7 28.0 3.0
Text-Embed-3-Large 49.0 19.5 50.1 52.3 70.0 72.4 16.2 21.6 28.0 4.0
NV-Embed-v2 83.0 73.5 51.4 55.0 72.8 69.4 13.5 20.7 55.0 6.0
Structure-Augmented RAG Agents
RAPTOR 33.5 15.8 31.3 34.3 45.8 59.4 12.3 13.4 14.0 1.0
GraphRAG 47.0 38.3 35.8 35.0 34.4 39.8 9.8 0.4 14.0 2.0
HippoRAG-v2 71.0 67.5 45.7 50.7 67.6 61.4 10.2 14.6 54.0 5.0
Mem0 28.0 4.8 22.4 36.0 37.5 3.4 10.0 0.8 18.0 2.0
Cognee 33.5 4.0 19.7 29.3 26.8 35.4 10.1 2.3 28.0 3.0
Agentic Memory Agents
Self-RAG 38.5 8.0 28.5 25.7 31.8 11.6 12.8 0.9 19.0 3.0
MemGPT 39.5 8.8 20.8 32.0 26.2 67.6 14.0 2.5 28.0 3.0
Agents Formulation In our framework, all agents are required to take the chunks one by one,
absorb them into memory, and incrementally update the memory. After seeing all the chunks, we ask
the agent to answer the related questions.
4 Experiments
4.1 Experimental Setup
The datasets are split into four categories and the statistics of all datasets are also shown in Table 1.
The evaluation metrics for all datasets are shown in Table 5 in Appendix A, along with more dataset
details. Then for the agents, as described in Section 3.3, we consider three categories of agents:
Long-Context Agents ,RAG agents andAgentic Memory Agents , where RAG Agents can be further
split into Simple RAG Agents ,Embedding-based RAG Agents andStructure-Augmented RAG Agents .
For chunk size settings, we choose a chunk size of 512 for the RULER-QA, NIAH-MQ, and LME(S*)
tasks in AR, as well as for all tasks in CR. This is mainly because these tasks are composed of long
texts synthesized from multiple short texts. For other tasks, we use a chunk size of 4096. Considering
computational overhead, we uniformly use a chunk size of 4096 for the Mem0 and Cognee methods.
We report the settings of the chunk size in Table 15 in Appendix C.
4.2 Overall Performance Comparison
Table 2 presents the overall performance across different benchmarks. We summarize the key
findings as follows: (1) Superiority of RAG methods in Accurate Retrieval Tasks. Most RAG
Agents are better than the backbone model “GPT-4o-mini” in the tasks within the Accurate Retrieval
Category. This matches our intuition where RAG agents typically excel at extracting a small snippet
of text that is crucial for answering the question. (2) Superiority of Long-Context Models in
Test-Time Learning and Long-Range Understanding. Long-context models achieve the best
performance on TTL and LRU. This highlights a fundamental limitation of RAG methods and
commercial memory agents, which still follow an agentic RAG paradigm. These systems retrieve
only partial information from the past context, lacking the ability to capture a holistic understanding
of the input—let alone perform learning across it. (3) Limitation of All Existing Methods on
Conflict Resolution. Although being a well-discussed task in model-editing community [ 32,11],
resolving conflict poses a significant challenge on memory agents. We observe that all methods
fail on the multi-hop situation (with achieving at most 6% accuracy). Only long context agents can
7

512 1024 2048 4096
Chunk Size020406080Accuracy (%)BM25
NVEmbed v2
HippoRAG v2
MemGPT(a) RULER-QA performance
512 1024 2048 4096
Chunk Size05101520Model Based F1BM25
NVEmbed v2
HippoRAG v2
MemGPT (b)∞Bench-Sum performance
Figure 2: Performances on RULER-QA with different chunk sizes.
achieve fairly reasonable results on single-hop scenarios. In Section 4.4.2, we show that current
reasoning models can have much better performance, while it does not change the conclusion that
Conflict Resolution still poses a significant challenge on all memory mechanisms. (4) Limited
Performance of Commercial Memory Agents. Commercial memory agents such as MemGPT
and Mem0 exhibit limited performance across a broad range of benchmarks. This shortfall can
be attributed to three primary factors. First, these systems frequently fail to capture and preserve
sufficient information when storing inputs into memory. For example, Mem0 depends on extracting
factual knowledge from inputs, an approach that inherently discards a substantial portion of the
original content. As a result, reconstructing inputs and supporting downstream tasks such as question
answering becomes significantly more challenging. While Mem0 has demonstrated relatively strong
performance on conversational tasks such as LOCOMO—where information density is comparatively
low—it tends to perform poorly on benchmarks containing dense informational content, including
RULER and ∞-Bench. For tasks emphasizing Time-to-Live (TTL) and Least Recently Used (LRU)
retrieval, these limitations are often even more pronounced. Second, both MemGPT and Mem0
rely on retrieval mechanisms that only access a subset of stored information. In the case of Mem0,
retrieval is typically performed a single time, similar to conventional RAG methods, constraining
the breadth of information available for reasoning. MemGPT, although adopting a more agentic
framework that permits multiple retrieval iterations, does not maintain temporal or structural metadata
about the stored content. Consequently, the agent is unable to reconstruct longer documents in their
original form, which adversely affects performance on LRU and other tasks requiring structured
memory retrieval. Finally, methods such as MemGPT depend heavily on embedding-based retrieval
mechanisms, which can be insufficient for fine-grained tasks like NIAH, where locating specific,
precise information (“the needle in the haystack”) is essential. These embedding-based approaches
often struggle to distinguish subtle contextual nuances that are critical for accurate retrieval in such
settings.
4.3 Ablation Study
In this Section, we conduct experiments and result analysis along four dimensions: Input Chunk
Size, Retrieval TopK, Validation of Dataset, and Computational Latency. More detailed experimental
results are provided in Appendix C.
4.4 Ablation Study on Input Chunk Size
To understand how chunk size impacts performance, particularly for RAG methods and agentic
memory agents, we conduct an additional analysis where we vary the chunk size while fixing the
number of retrieved chunks to 10. The results are presented in Figure 2. From the figure, we
observe the following: (1) In the RULER-QA task, reducing chunk size has little effect on BM25
performance. This is expected, as BM25 relies on term-frequency-based scoring and document-
level ranking, and does not inherently benefit from finer-grained segmentation beyond the impact
on term distributions. In contrast, embedding-based methods—including MemGPT, which uses
text-embedding-3-small as its retriever—consistently perform better with smaller chunks. This
suggests that finer segmentation improves the granularity and relevance of retrieved results for models
that rely on dense semantic representations. (2) In ∞Bench-Sum, however, smaller chunk sizes lead
to worse performance. This task requires the agent to summarize an entire conversation, and smaller
8

2 5 10
T op-K0204060Accuracy (%)RULER QA
2 5 10
T op-KMulti-Class Classification
2 5 10
T op-KBench QA
BM25 NV-Embed-v2 HippoRAG-v2Figure 3: The accuracies on different benchmarks when varying the retrieval top-k to be 2, 5 and 10.
chunks correspond to fewer available tokens per retrieval. As a result, the agent has access to less
context, which degrades summarization quality. The results suggest that, when resources permit,
using smaller chunk sizes and increasing the number of retrieval calls during memory construction
can improve performance on Accurate Retrieval (AR) tasks. Finer-grained segmentation enhances
the relevance of retrieved information, particularly for embedding-based methods. However, for
tasks requiring Long-Range Understanding (LRU), varying the chunk size hurts the performance.
This is likely because RAG methods are inherently less suited for tasks that demand integration of
information across a large, coherent context.
4.4.1 Ablation Study on Retrieval TopK
In our experiments, although we report results with the number of retrieved chunks set to 10 in
Table 2, we also conducted ablation studies with varying retrieval sizes. A subset of these results is
visualized in Figure 3, with the full results provided in Table 10. The results indicate that increasing
the number of retrieved chunks generally improves performance across most tasks. It is worth noting
that, with a chunk size of 4096 tokens, retrieving 10 chunks already yields an input of approximately
40k tokens. This places significant demands on model capacity. Due to this high token volume, we
do not evaluate settings with 20 retrieved chunks.
4.4.2 Validation of Dataset FactConsolidationTable 3: Performances of reasoning models
on the dataset FactConsolidation.
FactCon-SH FactCon-MH
6K 32K 6K 32K
GPT-4o 92.0 88.0 28.0 10.0
O4-mini 100.0 61.0 80.0 14.0As the performance of different models on this dataset
remains drastically low, we turn to the stronger reason-
ing model o4-mini and validate our dataset by check-
ing the performance of o4-mini on a smaller version
of this dataset. The results are shown in Table 3.
4.4.3 Analysis of Computational LatencyTable 4: Computational Latency (in seconds).
M.C.: Memory Construction. Q.E.: Query Ex-
ecution.
512 4096
M.C. Q.E. M.C. Q.E.
GPT-4o-mini 0.09 5.2 0.07 5.1
BM25 0.11 0.79 0.10 1.8
Contriever 7.2 0.76 1.7 2.0
Text-Embed-3-Large 6.3 0.54 5.4 1.8
NV-Embed-v2 93.4 0.83 42.9 1.8
RAPTOR 151 0.51 133 0.60
GraphRAG 123 9.9 90.4 9.4
HippoRAG-v2 817 1.1 284 2.6
Mem0 14644 1.2 2140 1.2
Cognee 8309 33.2 962 4.5
Self-RAG 8.4 2.0 6.7 1.7
MemGPT 413 10.6 93.3 11.4To illustrate the latency of various memory
agents in terms of (1) Memory Construction
(M.C.); (2) Query Execution (Q.E.), we ran-
domly choose 5 examples from RULER-QA2
and LME (S*), and report the latency of vari-
ous memory agents. This part of experiments is
done on a server with Four NVDIA L40 GPU
and AMD EPYC 7713 64-Core CPU. We use
the NV-Embed-v2 (7B) as the embedding model
in HippoRAG-v2. We show the summarized re-
sults in Table 4 and put the full results in Table
12 and 13. From the table, we find that using a
smaller chunk size requires significantly more
time for memory construction, especially for
methods such as HippoRAG-v2, Mem0, Cognee,
and MemGPT. Meanwhile, methods such as
Mem0, Cognee need extremely high resources
9

when constructing the memory, which may pose
challenges in real-world applications.
5 Conclusion and Future Work
In this paper, we introduce MemoryAgentBench , a unified benchmark designed to evaluate mem-
ory agents across four essential competencies: accurate retrieval, test-time learning, long-range
understanding, and conflict resolution. While prior benchmarks focus largely on skill execution
or long-context reasoning, MemoryAgentBench fills a critical gap by assessing how agents store,
update, and utilize long-term information across multi-turn interactions. To build this benchmark, we
restructure existing datasets and propose two new ones— EventQA andFactConsolidation —tailored
to stress specific memory behaviors often overlooked in prior work. We evaluate a wide spectrum of
agents, including long-context models, RAG-based systems, and commercial memory agents, under a
consistent evaluation protocol. Our results reveal that, despite recent advances, current memory agents
still exhibit substantial limitations when faced with tasks requiring dynamic memory updates and
long-range consistency. One limitation of our work is that the datasets used in MemoryAgentBench
are primarily synthetic, which may not fully reflect the characteristics of real-world user conversations.
As future work, we aim to collect and curate more realistic, real-world datasets aligned with our
four competencies to further enrich and diversify the benchmark and provide more comprehensive
evaluations for memory agents.
Acknowledgment
We thank Kevin Lin for engaging in thoughtful discussions around the overall idea and latency
evaluation. His input played a role in shaping the evaluation pipeline for the memory agents.
10

References
[1]Anthropic. Claude 3.7 sonnet, 2025. URL https://www.anthropic.com/news/
claude-3-7-sonnet . This announcement introduces Claude 3.7 Sonnet, described as An-
thropic’s most intelligent model to date and the first hybrid reasoning model generally available
on the market.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations , 2023.
[3]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du,
Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long
context understanding. arXiv preprint arXiv:2308.14508 , 2023.
[4]Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng
Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning
on realistic long-context multitasks. arXiv preprint arXiv:2412.15204 , 2024.
[5]Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon, Jonathan Berant, Matthew R Gormley, and
Graham Neubig. In-context learning with long-context models: An in-depth exploration. arXiv
preprint arXiv:2405.00200 , 2024.
[6]Iñigo Casanueva, Tadas Tem ˇcinas, Daniela Gerz, Matthew Henderson, and Ivan Vuli ´c. Efficient
intent detection with dual sentence encoders. In Tsung-Hsien Wen, Asli Celikyilmaz, Zhou Yu,
Alexandros Papangelis, Mihail Eric, Anuj Kumar, Iñigo Casanueva, and Rushin Shah, editors,
Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI , pages
38–45, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.
nlp4convai-1.5. URL https://aclanthology.org/2020.nlp4convai-1.5/ .
[7]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0:
Building production-ready ai agents with scalable long-term memory. arXiv preprint
arXiv:2504.19413 , 2025.
[8]DeepMind. Gemini pro, 2025. URL https://deepmind.google/technologies/gemini/
pro/ . This page provides an overview of Gemini Pro, highlighting its advanced capabilities
and applications in various fields.
[9]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. In Proceedings of the 2019 conference of
the North American chapter of the association for computational linguistics: human language
technologies, volume 1 (long and short papers) , pages 4171–4186, 2019.
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 , 2024.
[11] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He,
and Tat-Seng Chua. Alphaedit: Null-space constrained knowledge editing for language models.
arXiv preprint arXiv:2410.02355 , 2024.
[12] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory:
Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802 ,
2025.
[13] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bod-
hisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. Large language models
as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international
conference on information and knowledge management , pages 720–730, 2023.
[14] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bod-
hisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. Large language models
as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international
conference on information and knowledge management , pages 720–730, 2023.
11

[15] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia,
Yang Zhang, and Boris Ginsburg. RULER: What’s the Real Context Size of Your Long-
Context Language Models?, August 2024. URL http://arxiv.org/abs/2404.06654 .
arXiv:2404.06654 [cs].
[16] Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye,
Zhaoxuan Jin, Yingru Li, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Ping Luo, and Guohao
Li. Owl: Optimized workforce learning for general multi-agent assistance in real-world task
automation, 2025. URL https://github.com/camel-ai/owl .
[17] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning.
arXiv preprint arXiv:2112.09118 , 2021.
[18] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint
arXiv:2310.06770 , 2023.
[19] Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. One thou-
sand and one pairs: A" novel" challenge for long-context language models. arXiv preprint
arXiv:2406.16264 , 2024.
[20] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InEMNLP (1) , pages 6769–6781, 2020.
[21] Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker
Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason
Mars. An evaluation dataset for intent classification and out-of-scope prediction. In Kentaro
Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP) , pages 1311–1316, Hong Kong, China,
November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1131. URL
https://aclanthology.org/D19-1131/ .
[22] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan
Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist
embedding models. arXiv preprint arXiv:2405.17428 , 2024.
[23] Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context
language models understand long contexts? arXiv preprint arXiv:2311.04939 , 2023.
[24] Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and
Chris Pal. Towards deep conversational recommendations. Advances in neural information
processing systems , 31, 2018.
[25] Xianming Li, Julius Lipp, Aamir Shakir, Rui Huang, and Jing Li. Bmx: Entropy-weighted
similarity and semantic-enhanced lexical search. arXiv preprint arXiv:2408.06643 , 2024.
[26] Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th Interna-
tional Conference on Computational Linguistics , 2002. URL https://aclanthology.org/
C02-1150/ .
[27] Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, and Joseph E
Gonzalez. Sleep-time compute: Beyond inference scaling at test-time. arXiv preprint
arXiv:2504.13171 , 2025.
[28] Xingkun Liu, Arash Eshghi, Pawel Swietojanski, and Verena Rieser. Benchmarking natural
language understanding services for building conversational agents, 2019. URL https://
arxiv.org/abs/1903.05566 .
[29] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and
Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint
arXiv:2402.17753 , 2024.
12

[30] Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Mass-
editing memory in a transformer. In ICLR . OpenReview.net, 2023.
[31] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia:
a benchmark for general ai assistants. In The Twelfth International Conference on Learning
Representations , 2023.
[32] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn.
Memory-based model editing at scale. In ICML , volume 162 of Proceedings of Machine
Learning Research , pages 15817–15831. PMLR, 2022.
[33] Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A Rossi, Se-
unghyun Yoon, and Hinrich Schütze. Nolima: Long-context evaluation beyond literal matching.
arXiv preprint arXiv:2502.05167 , 2025.
[34] Magnus Müller and Gregor Žuni ˇc. Browser use: Enable ai to control your browser, 2024. URL
https://github.com/browser-use/browser-use .
[35] OpenAI. Gpt-4o system card, 2025. URL https://openai.com/index/
gpt-4o-system-card/ . This report outlines the safety work carried out prior to re-
leasing GPT-4o including external red teaming, frontier risk evaluations according to our
Preparedness Framework, and an overview of the mitigations we built in to address key risk
areas.
[36] Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonza-
lez. Memgpt: Towards llms as operating systems. 2023.
[37] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956 ,
2025.
[38] Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2-poisson
model for probabilistic weighted retrieval. In SIGIR’94: Proceedings of the Seventeenth Annual
International ACM-SIGIR Conference on Research and Development in Information Retrieval,
organised by Dublin City University , pages 232–241. Springer, 1994.
[39] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth
International Conference on Learning Representations , 2024.
[40] Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng
Bao, Qian Wang, and Yue Zhang. Novelqa: A benchmark for long-range novel question
answering. arXiv preprint arXiv:2403.12766 , 2024.
[41] Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang
Yu, Nan Xu, Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context
llms with extended multi-doc qa. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing , pages 5627–5646, 2024.
[42] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi
Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software
developers as generalist agents. In The Thirteenth International Conference on Learning
Representations , 2024.
[43] Yu Wang, Xinshuang Liu, Xiusi Chen, Sean O’Brien, Junda Wu, and Julian McAuley. Self-
updatable large language models by integrating context into model parameters. In The Thirteenth
International Conference on Learning Representations .
[44] Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin,
Zheng Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models.
arXiv preprint arXiv:2402.04624 , 2024.
[45] Yu Wang, Ruihan Wu, Zexue He, Xiusi Chen, and Julian McAuley. Large scale knowledge
washing. arXiv preprint arXiv:2405.16720 , 2024.
13

[46] Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan
Gutfreund, Rogerio Feris, and Zexue He. M+: Extending memoryLLM with scalable long-
term memory. In Forty-second International Conference on Machine Learning , 2025. URL
https://openreview.net/forum?id=OcqbkROe8J .
[47] Yu Wang, Chi Han, Tongtong Wu, Xiaoxin He, Wangchunshu Zhou, Nafis Sadeq, Xiusi Chen,
Zexue He, Wei Wang, Gholamreza Haffari, Heng Ji, and Julian J. McAuley. Towards lifespan
cognitive systems. TMLR , 2025/02.
[48] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Long-
memeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint
arXiv:2410.10813 , 2024.
[49] Qiyu Wu, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, and Daxin Jiang. Pcl: Peer-
contrastive learning with diverse augmentations for unsupervised sentence embeddings. arXiv
preprint arXiv:2201.12093 , 2022.
[50] Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe
Wasserblat, and Danqi Chen. Helmet: How to evaluate long-context language models effectively
and thoroughly. arXiv preprint arXiv:2410.02694 , 2024.
[51] Zhangyue Yin, Qiushi Sun, Qipeng Guo, Zhiyuan Zeng, Qinyuan Cheng, Xipeng Qiu, and Xuan-
Jing Huang. Explicit memory learning with expectation maximization. In Proceedings of the
2024 Conference on Empirical Methods in Natural Language Processing , pages 16618–16635,
2024.
[52] Tian Yu, Shaolei Zhang, and Yang Feng. Auto-rag: Autonomous retrieval-augmented generation
for large language models. arXiv preprint arXiv:2411.19443 , 2024.
[53] Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han,
Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. ∞bench: Extending long context evaluation beyond
100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 15262–15277, 2024.
[54] Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen.
Mquake: Assessing knowledge editing in language models via multi-hop questions. arXiv
preprint arXiv:2305.14795 , 2023.
14

A Details of Dataset
Here we provide a detailed introduction to the datasets used for evaluating the four core competencies,
including the dataset curation, corresponding metrics, average context length, and a brief description.
Details are shown in Table 5.
A.1 Accurate Retrieval (AR)
We use five datasets to evaluate the accurate retrieval capability of memory agents.
(1) RULER-QA We adopt two QA datasets from [ 15]. These datasets provide multiple synthetic
contexts of varying lengths, ranging from 3K to over 200K tokens. We select 100 questions from the
datasets with shorter contexts. For each of these 100 questions, we collect all the documents of them,
removed duplicate content, and then shuffled and concatenated them to create new long contexts of
197K or 421K tokens, making sure the new contexts containing the gold passage. Since most answers
are short informational entities, such as years, names, or yes/no responses, we use substring exact
match (SubEM) as the evaluation metric. SubEM measures whether the predicted answer exactly
matches the gold answer as a substring, which is a common standard in question answering systems.
(2) NIAH-MQ We choose a context with a length of 448K tokens, containing 100 groups and a
total of 400 queries. We first check whether these queries appeared in the context. Then, we randomly
shuffl the queries and their corresponding numbers evenly to prevent them from clustering together in
the long context. The primary evaluation criterion is whether the agent can successfully retrieve the
correct numbers. Therefore, we use average recall as the evaluation metric.
(3)∞Bench-En.QA We borrow this dataset from [ 53]. It is a QA task using novels in which
character names have been replaced. This makes the content more coherent and closer to real-world
scenarios. We use ROUGE F1 for this dataset since answers are mostly entity names.
(4) LongMemEval This is a dialogue-based QA dataset. For LME(S*), we use multiple historical
conversation data segments, arrange them in chronological order, and finally concatenate them to
create five long conversation histories, each with a length of approximately 355K tokens. Since some
of the questions have open-ended answers, we adopt the approach used in previous work and employ
the GPT-4o model to assess whether the agent’s responses meet the requirements. If a response
is deemed satisfactory, it is marked as True. Finally, we calculate the proportion of satisfactory
responses as the evaluation metric.
(5) EventQA Using five books from ∞Bench (each >390K tokens, counted using the gpt-4o-mini
tokenizer), we identify the ten most frequently mentioned characters via SpaCy NER. We extract
101 events experienced by key characters using gpt-4o . For each event, we construct a 6-way
multiple-choice question by pairing the true event with five distractors generated via gpt-4o . The
agent receives up-to five previous events and must identify the correct continuation. We report the
mean accuracy over 100 such questions per book, and ultimately present the average accuracy across
all five books.
A.2 Test-time Learning (TTL)
We evaluate TTL via two task categories: (1) Multi-Class Classification (MCC) : We adopt five
classification datasets used in prior TTL work. For dataset curation, we use thousands of sentence
samples from different categories, with each type of sample assigned a number as its label. Following
the format "{sentence} \n Label: {label} \n", we concatenate all the sentences into a long context
and shuffle them to prevent samples of the same type from being too concentrated. In this task,
the agent needs to refer to a long context and correctly classify the input content. Therefore, we
use average accuracy as the evaluation metric. (2) Recommendation (Recom) : We concatenate
multiple short dialogues about movie recommendations from the original dataset, remove duplicate
dialogues, and create a long context containing over a thousand recommendation instances. In
this task, the agent is required to recommend 20 movies based on the content of the dialogue. We
evaluate the recommendations by calculating Recall@5, which measures the overlap between the top
5 recommended movies and the ground truth.
15

Table 5: Overview of evaluation datasets. We select datasets that cover various important long-
context capabilities. SubEM: substring exact match. In the table, we underline the datasets we
constructed ourselves. Avg. Length: Average Context Length (measured using the GPT-4o-mini
model’s tokenizer).
Category Dataset Metrics Avg. Length Description
Accurate
RetrievalRULER-QA1SubEM197KGold passage retrieval QA.RULER-QA2 421K
RULER-NIAH-MQ Recall 448K Retrieve multiple “needles” from the “haystack”.
∞Bench-QA ROUGE F1 183K Novel QA with entity replacement.
LongMemEval (S)Model Based Acc.110KDialogues based QA.LongMemEval (S*) 355K
EventQA ( ours) Accuracy 534K Novel multiple-choice QA on characters events.
Test-time
LearningBANKING77
Accuracy 103KBanking intent classification, 77 labels
CLINC150 Intent classification, 151 labels
NLU Task intent classification, 68 labels
TREC Coarse Question type classification, 6 labels
TREC Fine Question type classification, 50 labels
Movie Recommendation Recall@5 1.44M Recommend movies based on provided dialogues examples.
Long Range
Understanding∞Bench-Sum Model Based F1 172K Novel summarization with entity replacement.
Conflict
ResolvingFactConsolidation-SH ( ours)SubEM 262KConflict solving in single hop reasoning.
FactConsolidation-MH ( ours) Conflict solving in multiple hop reasoning.
A.3 Long-Range Understanding (LRU)
We evaluate LRU via the Summarization task En.Sum from∞-Bench [ 53]. We follow the settings
from [ 50] and use the GPT-4o model in evaluating the summarized text. In this process, we assess
the fluency of the input text (scored as 0 or 1) and use the dot product of this score with the F1 score
as the final evaluation metric.
A.4 Conflict Resolution (CR)
We use counterfactual edit pairs from the MQUAKE [ 54] dataset. Each sentence containing informa-
tion was assigned a number. For each edit pair, the sentence representing outdated information (the
distractor) is given a smaller number, while the sentence representing more recent information (the
one containing the answer) is given a larger number. We then concatenate these sentences into a long
context in order according to their assigned numbers. We evaluate the CR via two datasets: Single-
Hop Editing andMulti-Hop Editing . In these tasks, the agent’s responses are mostly informational
entities. Therefore, we also use SubEM (Substring Exact Match) as the evaluation metric.
B Prompts
We introduce some example prompts used in this section.
B.1 Instructions for Memory Construction
When processing long-context inputs, we split the content into chunks of a specified size and feed
these chunks into the agent as memory. The agent can then extract relevant information from
its memory based on the query to assist with query execution. This chunking approach helps
organize and manage large amounts of contextual information, making retrieval and reasoning more
efficient. In Figure 4, we provide several example instructions that require the agent to memorize the
corresponding context.
B.2 Instructions for Long-Context Agents
In Figure 5, we provide the examples of instructions used on different of datasets. For some existing
datasets, we adjust the prompt settings from previous work such as [ 15,48]. For example, for the
dataset ∞Bench-QA and∞Bench-Sum , we also insert two answer examples as <demo> in the
prompt to help the agent better understand the questions and standardize its outputs.
16

Prompts Used for Memory Construction on Various Tasks
IFLongMemEval :
Memorize the following conversation between the user and the assistant: \n <chunk> \n
ELIF Movie Recommendation :
Memorize the following dialogues between a user and recommender system: \n <chunk> \n
ELIF Fact Consolidation :
Memorize the these following facts: \n <chunk> \n
ELSE:
Memorize the following content: \n <chunk> \n
Figure 4: The prompts we use for the agents to create the memory.
B.3 Instructions for RAG Agents
We provide examples of prompts used for the RAG based Agents in Figure 6. For this type of
agent, after storing the input long context in memory, we use <question> as the memory retrieval
query for most tasks. But for task RULER-NIAH-MQ , we use the entire question "What are all
the special magic numbers for <question> mentioned in the memorized text?" as the query. And for
∞Bench-Sum , we use the entire query without the <demo> for the memory retrieval.
C Detailed Experimental Results
In this section, we provide detailed versions of the results presented in the main text.
C.1 Detailed Results on AR
In Table 6, we present the detailed results for each agent on every dataset. For AR tasks, using
Simple RAG Agents equipped with retrievers like BM25 can significantly improve performance
compared to the backbone model. This is because the GPT-4o-mini is limited by its 128K context
length, which restricts the amount of information it can process at once. Meanwhile, the overall
performance of Embedding RAG Agents surpasses that of both Structure-Augmented RAG Agents
and Agentic Memory Agents. This advantage is primarily attributed to the use of dense retrieval in
Embedding RAG Agents. It enables the extraction of longer contextual information from memory.
As a result, Embedding RAG Agents are able to provide richer and more comprehensive context for
tasks.
C.2 Detailed Results on TTL, LRU and CR
We give detailed results on each dataset in Table 7. For all three types of tasks, RAG-based agents
generally underperform compared to their respective GPT-4o-mini backbones. This observation
highlights certain limitations inherent to the RAG approach. For instance, in TTL tasks, RAG-based
methods often struggle to accurately retrieve context from memory that is closely associated with
the input. In LRU tasks, these methods face challenges in achieving a comprehensive understanding
of long contexts. Furthermore, for CR tasks—especially the multi-hop variants—effective handling
requires strong reasoning and information extraction capabilities, which remain beyond the reach of
most current agents.
C.3 Detailed Results on Ablation Study
In this section, we introduce the detailed results on the ablation study on different chunk sizes, retrieve
number, context length and computation latency.
In Table 8 and 9, we compare the RAG-based Agents on different chunk sizes and datasets. We
selected chunk sizes from the two sets {512, 4096} and {512, 1024, 2048, 4096}. For some datasets
composed of synthetic text, such as RULER-QA, using a smaller chunk size generally helps RAG
Agents andAgentic Memory Agents achieve better test performance. However, for datasets composed
of continuous text, such as ∞Bench-QA, since the retrieval number kremains unchanged, reducing
the chunk size does not lead to performance improvement.
17

Prompts Used for Long-Context Agents on Various Tasks
RULER-QA
The context is given as below: <memory>. \n please memorize it.\n Answer the question based on the memorized
documents. Only give me the answer and do not output any other words. \n Now Answer the Question:
<question> \n Answer:
RULER-NIAH-MQ
The context is given as below: <memory>. \n Please memorize it. \n Some special magic numbers are hidden
within the memorized text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n Now
Answer the Question: What are all the special magic numbers for <question> mentioned in the memorized text?
\n The special magic numbers for <question> mentioned in the memorize text are:
∞Bench-QA
The context is given as below: <memory>. \n Please memorize it. \n Based on the context you memorized,
answer the question as concisely as you can, using a single phrase if possible. \n <demo>.\n Now Answer the
Question: <question>.\n Answer:
LongMemEval
Here are several history chats between you and a user : <memory> \n Please memorize them. \n The history chats
are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using
a single phrase if possible.\n Current Date: <question_date>, \n Now Answer the Question: <question> \n Answer:
EventQA
The context is given as below: <memory>. \n Please memorize it. \n Based on the context you memorized,
complete the task below: \n These are the events that have already occurred: \n <previous_events> \n Below is a
list of possible subsequent events:\n <question> \n Your task is to choose from the above events which event
happens next based on the book excerpt. In your response to me, only include the answer without anything else.
\n The event that happens next is:
Label Matching (BANKING77, etc.)
The context is given as below: <memory>. \n Please memorize it. \n Use the provided mapping from the context
to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n
Question: <question> \n label:
Movie Recommendation
Here are dialogues between a user and recommender system: <memory>. \n Please memorize them. \n Pretend
you are a movie recommender system. You need to recommend movies based on the dialogues you have
memorized. Now I will give you a new conversation between a user and you (a recommender system). Based
on the conversation, you reply me with 20 recommendations without extra sentences. \n For Example:\n
[Conversation] \n The recommendations are: \n 1.movie1 \n 2.movie2 \n ...\n Here is the conversation: <question>
\n The recommendations are:
∞Bench-Sum
The book is given as below: <memory> \n Please memorize it. \n You are given a book above and you are tasked
to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the
story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary. \n
<demo> \n Now summarize the book.
Fact Consolidation
Here is a knowledge pool with lots of new facts: <memory>. \n Please memorize it. \n Pretend you are a
knowledge management system. Each fact in the knowledge pool is provided with a serial number at the
beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge
pool by finding the newest fact. You need to answer a question based on this rule. You should give a very concise
answer without saying other words for the question **only** from the knowledge pool you have memorized
rather than the real facts in real world. \n For example: \n [Knowledge Pool] \n Question: Based on the provided
Knowledge Pool, what is the name of the current president of Country R? \n Answer: Person D. \n Now Answer
the Question: Based on the provided Knowledge Pool, <question> \n Answer:
Figure 5: The prompts we use for the Long-Context Agents in Table 2. Here <memory> refers to the
accumulated text from the sequential inputs.
In Table 10, we evaluate the selected RAG-based Agents on five datasets. We choose different TopK
ranging from {2, 5, 10}. We find that for the AR series of tasks, increasing the retrieve number
(TopK) leads to a significant improvement in performance. However, for the TTL series of tasks, the
performance gains from increasing TopK are much less pronounced.
18

Prompts Used for RAG Based Agents on Various Tasks
RULER-QA
Here is the context retrieved from memory: <memory>.\n Answer the question based on the retrieved context.
Only give me the answer and do not output any other words. \n Now Answer the Question: <question> \n Answer:
RULER-NIAH-MQ
Here is the context retrieved from memory: <memory>.\n Some special magic numbers are hidden within the
retrieved text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n Now Answer the
Question: What are all the special magic numbers for <question> mentioned in the memorized text? \n The
special magic numbers for <question> mentioned in the memorize text are:
∞Bench-QA
Here is the context retrieved from memory: <memory>.\n Based on the context you retrieved, answer the question
as concisely as you can, using a single phrase if possible. \n <demo>.\n Now Answer the Question: <question>.\n
Answer:
LongMemEval
Here are retrieved several history chats between you and a user from memory: <memory> \n The retrieved history
chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can,
using a single phrase if possible.\n Current Date: <question_date>, \n Now Answer the Question: <question> \n
Answer:
EventQA
Here is the context retrieved from memory: <memory>.\n Based on the context you retrieved, complete the task
below: \n These are the events that have already occurred: \n <previous_events> \n Below is a list of possible
subsequent events:\n <question> \n Your task is to choose from the above events which event happens next based
on the book excerpt. In your response to me, only include the answer without anything else. \n The event that
happens next is:
Label Matching (BANKING77, etc.)
Here are the examples retrieved from memory: <memory>. \n Use the retrieved mapping from the context to
numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n
Question: <question> \n label:
Movie Recommendation
Here are retrieved dialogues between a user and recommender system from memory: <memory>. \n Pretend you
are a movie recommender system. You need to recommend movies based on the example dialogues you have
retrieved. Now I will give you a new conversation between a user and you (a recommender system). Based on the
conversation, you reply me with 20 recommendations without extra sentences. \n For Example:\n [Conversation]
\n The recommendations are: \n 1.movie1 \n 2.movie2 \n ...\n Here is the conversation: <question> \n The
recommendations are:
∞Bench-Sum
The book context is retrieved from memory and it is given as below: <memory> \n You are given retrieved
context above and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write
about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide
any analysis or commentary. \n <demo> \n Now summarize the book.
Fact Consolidation
Here is a list of knowledge retrieved from memory: <memory>. \n Pretend you are a knowledge management
system. Each fact in the retrieved knowledge pool is provided with a serial number at the beginning, and the
newer fact has larger serial number. \n You need to solve the conflicts of facts in the retrieved knowledge pool by
finding the newest fact. You need to answer a question based on this rule. You should give a very concise answer
without saying other words for the question **only** from the retrieved knowledge pool you have memorized
rather than the real facts in real world. \n For example: \n [Knowledge Pool] \n Question: Based on the provided
Knowledge Pool, what is the name of the current president of Country R? \n Answer: Person D. \n Now Answer
the Question: Based on the provided Knowledge Pool, <question> \n Answer:
Figure 6: The prompts we use for the Simple RAG Agents ,Embedding RAG Agents ,Structure-
Augmented RAG Agents andAgentic Memory RAG Agents in Table 2. Here <memory> refers to the
retrieved text from the sequential inputs. For MemGPT method, we also add the phrase "Search
Archival Memory" in prompt of each task.
In Table 11, we report the performances of different agents when scaling the input length. We
measure the average context length via the tokenizer of GPT-4o-mini and here 1K is 1024. For
Long-Context Agents, tasks in the AR series generally achieve satisfactory performance at relatively
19

Table 6: Overall Performance Comparison on the datasets for AR. All RAG agents and commercial
memory agents use GPT-4o-mini as the backbone. Thus we highlight the performance of GPT-4o-
mini as the reference.
Agent Type RULER-QA1 RULER-QA2 NIAH-MQ ∞Bench-QA LME(S) LME(S*) EventQA
Long-Context Agents
GPT-4o 72.0 51.0 25.0 55.4 61.4 32.0 77.2
GPT-4o-mini 64.0 43.0 22.8 44.9 55.6 30.7 59.0
GPT-4.1-mini 83.0 66.0 94.8 45.8 61.4 55.7 82.6
Gemini-2.0-Flash 87.0 59.0 83.8 53.2 52.6 47.0 67.2
Claude-3.7-Sonnet 77.0 53.0 38.0 50.6 59.0 34.0 74.6
GPT-4o-mini 64.0 43.0 22.8 44.9 55.6 30.7 59.0
Simple RAG Agents
BM25 68.0 54.0 100.0 45.6 55.2 45.3 74.6
Embedding RAG Agents
Contriever 22.0 31.0 2.5 38.1 32.8 15.7 66.8
Text-Embed-3-Small 60.0 44.0 7.2 44.4 49.0 48.3 63.0
Text-Embed-3-Large 54.0 44.0 19.5 50.1 44.6 52.3 70.0
NV-Embed-v2 90.0 67.0 73.5 51.4 45.4 55.0 72.8
Structure-Augmented RAG Agents
RAPTOR 29.0 38.0 15.8 31.3 38.8 34.3 45.8
GraphRAG 47.0 47.0 38.3 35.8 39.2 35.0 34.4
HippoRAG-v2 76.0 66.0 67.5 45.7 44.2 50.7 67.6
Mem0 24.0 32.0 4.8 22.4 45.0 36.0 37.5
Cognee 31.0 26.0 4.0 19.7 31.3 29.3 26.8
Agentic Memory Agents
Self-RAG 35.0 42.0 8.0 28.5 23.4 25.7 31.8
MemGPT 41.0 38.0 8.8 20.8 41.4 32.0 26.2
Table 7: Overall performance comparison on the datasets for TTL, LRU and CR. All RAG agents
and commercial memory agents use GPT-4o-mini as the backbone.
Agent Type BANKING CLINC NLU TREC C TREC F Recom ∞Bench-Summ FactCon-SH FactCon-MH
Long-Context Agents
GPT-4o 96.0 96.0 90.0 87.0 69.0 12.3 32.2 60.0 5.0
GPT-4o-mini 93.0 93.0 87.0 73.0 66.0 15.1 28.9 45.0 5.0
GPT-4.1-mini 93.0 82.0 85.0 68.0 50.0 16.7 41.9 36.0 5.0
Gemini-2.0-Flash 91.0 90.0 84.0 88.0 67.0 8.7 23.9 30.0 3.0
Claude-3.7-Sonnet 97.0 98.0 86.0 87.0 79.0 18.3 52.5 43.0 2.0
GPT-4o-mini 93.0 93.0 87.0 73.0 66.0 15.1 28.9 45.0 5.0
Simple RAG Agents
BM25 89.0 89.0 84.0 62.0 53.0 13.6 20.9 56.0 3.0
Embedding RAG Agents
Contriever 89.0 88.0 80.0 55.0 41.0 15.2 21.2 18.0 7.0
Text-Embed-3-Small 88.0 89.0 83.0 54.0 36.0 15.3 25.7 28.0 3.0
Text-Embed-3-Large 90.0 91.0 80.0 55.0 46.0 16.2 21.6 28.0 4.0
NV-Embed-v2 88.0 89.0 82.0 40.0 48.0 13.5 20.7 55.0 6.0
Structure-Augmented RAG Agents
RAPTOR 78.0 75.0 73.0 48.0 23.0 12.3 13.4 14.0 1.0
GraphRAG 64.0 54.0 49.0 24.0 6.0 9.8 0.4 14.0 2.0
HippoRAG-v2 81.0 86.0 73.0 38.0 29.0 10.2 14.6 54.0 5.0
Mem0 5.0 4.0 1.0 6.0 1.0 10.0 0.8 18.0 2.0
Cognee 34.0 42.0 42.0 41.0 18.0 10.1 2.3 28.0 3.0
Agentic Memory Agents
Self-RAG 19.0 13.0 6.0 15.0 5.0 12.8 0.9 19.0 3.0
MemGPT 89.0 83.0 79.0 56.0 31.0 14.0 2.5 28.0 3.0
small context lengths (e.g., around 50K tokens). However, as the context length increases, the
performance of these agents declines accordingly. In contrast, for the RAG-based agents Mem0 and
Cognee, their performance is significantly lower than that of their backbone, GPT-4o-mini, even
when the context length is relatively small.
In Table 12 and Table 13, we provide the computational latency on different agents. We choose
two chunk sizes {512, 4096} and two datasets {RULER-QA2, LME(S*)}. The experimental
results demonstrate that, for most agents, selecting a smaller chunk size leads to significantly higher
computational latency. For example, in the case of Cognee, the computational latency at a chunk size
of 512 is nearly 8 to 10 times greater than that at a chunk size of 4096.
20

Table 8: Performance comparison on different datasets and chunk sizes. We choose two different
chunk sizes 512, 4096 and we use k=10 for RAG-based methods.
NIAH-MQ ∞Bench-QA LME(S*) Event-QA FactCon-SH FactCon-MH
512 4096 512 4096 512 4096 512 4096 512 4096 512 4096
Simple RAG Agents
BM25 100 95.5 32.9 45.6 45.3 48.3 69.4 74.6 56.0 44.0 3.0 5.0
Embedding RAG Agents
Contriever 2.5 8.8 28.5 38.1 15.7 19.0 62.8 66.8 18.0 25.0 7.0 5.0
Text-Embed-3-Small 7.2 12.3 42.4 44.4 48.3 39.0 60.8 63.0 28.0 21.0 3.0 4.0
Text-Embed-3-Large 19.5 13.5 42.3 50.1 52.3 39.3 62.0 70.0 28.0 22.0 4.0 3.0
NV-Embed-v2 73.5 31.8 40.7 51.4 55.0 43.0 72.7 72.8 55.0 42.0 6.0 6.0
Structure-Augmented RAG Agents
RAPTOR 15.8 4.5 21.3 31.3 34.3 31.7 43.6 45.8 14.0 19.0 1.0 2.0
GraphRAG 38.3 8.0 35.2 35.8 35.0 36.7 33.8 34.4 14.0 10.0 2.0 3.0
HippoRAG-v2 67.5 23.3 34.5 45.7 50.7 37.3 67.5 67.6 54.0 29.0 5.0 3.0
Agentic Memory Agents
Self-RAG 8.0 7.0 27.7 28.5 25.7 23.0 30.2 31.8 19.0 14.0 3.0 2.0
MemGPT 8.8 3.5 23.5 20.8 32.0 28.0 26.2 25.8 28.0 13.0 3.0 3.0
Table 9: Performance comparison on different datasets and chunk sizes. Here we choose chunk sizes
from {512, 1024, 2048, 4096} and we use k=10 for RAG-based methods.
RULER-QA-1 RULER-QA-2 ∞Bench-Sum
512 1024 2048 4096 512 1024 2048 4096 512 1024 2048 4096
BM25 68.0 67.0 68.0 66.0 54.0 51.0 52.0 56.0 11.5 13.2 19.2 20.9
NVEmbed-v2 90.0 80.0 57.0 57.0 67.0 59.0 52.0 39.0 11.6 13.0 16.8 20.7
HippoRAG-v2 76.0 70.0 57.0 49.0 66.0 63.0 51.0 38.0 4.6 6.0 10.5 14.6
MemGPT 41.0 32.0 24.0 27.0 38.0 33.0 37.0 35.0 1.2 1.8 4.2 2.5
Table 10: Performance comparison on different retrieve number.
RULER NIAH ∞Bench-QA EventQA TTL (MCC)
R=2 R=5 R=10 R=2 R=5 R=10 R=2 R=5 R=10 R=2 R=5 R=10 R=2 R=5 R=10
BM25 49.5 59.5 61.0 34.5 57.0 95.5 26.7 38.3 45.6 66.6 71.2 74.6 67.8 75.4 74.6
Contriever 26.0 38.0 41.0 0.5 5.3 8.8 23.8 37.1 38.1 54.4 66.8 56.0 63.0 70.0 70.6
Text-Embed-3-Large 32.5 33.5 36.5 5.0 9.3 13.5 34.8 41.9 50.1 51.8 62.4 70.0 59.4 69.4 72.4
NV-Embed-v2 37.0 43.5 48.0 17.8 26.8 31.8 42.8 48.1 51.4 59.4 68.4 72.8 63.8 69.4 68.8
RAPTOR 21.0 19.5 23.5 4.3 4.5 4.3 30.9 30.4 31.3 45.8 41.8 40.4 56.3 59.4 57.4
HippoRAG-v2 38.0 42.5 43.5 16.5 23.3 23.3 35.9 45.3 45.7 58.8 67.6 67.4 58.8 61.4 61.4
Self-RAG 29.5 32.5 38.5 4.0 6.3 7.0 21.0 23.9 28.5 28.2 30.6 31.8 9.0 11.6 11.6
Table 11: Performance comparison on different context length.
RULER NIAH EventQA FactCon-SH FactCon-MH
51K 104K 304K 55K 117K 448K 51K 108K 534K 32K 64K 262K 32K 64K 262K
GPT-4o 81.5 76.0 61.5 100 100 25.0 96.8 94.0 77.2 88.0 85.0 60.0 10.0 13.0 5.0
GPT-4o-mini 71.0 68.5 53.5 99.5 99.0 22.8 90.2 85.8 59.0 63.0 58.0 45.0 10.0 5.0 5.0
GPT-4.1-mini 82.5 80.5 74.5 99.5 99.0 94.8 97.0 93.8 82.6 82.0 72.0 36.0 7.0 9.0 5.0
Gemini-2.0-Flash 80.5 74.0 73.0 87.8 93.3 83.8 93.4 88.6 67.2 49.0 62.0 30.0 7.0 9.0 3.0
Claude-3.7-Sonnet 78.5 70.5 65.0 99.5 100 38.0 96.6 95.2 74.6 46.0 45.0 43.0 2.0 2.0 0.0
Mem0 28.5 27.0 28.0 5.5 5.0 4.8 60.8 47.0 40.2 22.0 8.0 18.0 3.0 2.0 2.0
Cognee 37.0 40.0 33.5 17.5 14.3 4.0 53.4 39.0 26.8 39.0 31.0 28.0 4.0 5.0 3.0
D Experimental Settings
In this section, we present the experimental settings.
D.1 Max Output Tokens
We provide the token number limitation for each task in Table 14.
21

Table 12: Computational latency (in seconds) comparison on Long-Context Agents.
RULER-QA2 LME (S*)
GPT-4o 17.0 20.1
GPT-4o-mini 4.9 5.4
GPT-4.1-mini 9.0 7.4
Gemini-2.0-Flash 12.4 10.1
Claude-3.7-Sonnet 23.3 22.7
Table 13: Computational latency (in seconds) comparison on RAG based agents. M.C. means
Memory Construction and Q.E. means Query Execution.
RULER-QA2 LME (S*)
512 4096 512 4096
M.C. Q.E. M.C. Q.E. M.C. Q.E. M.C. Q.E.
BM25 0.12 0.47 0.11 1.7 0.09 1.1 0.08 1.9
Contriever 7.4 0.59 1.7 2.0 6.9 0.92 1.6 1.9
Text-Embed-3-Large 6.1 0.46 5.0 1.7 6.5 0.62 5.8 1.8
NV-Embed-v2 102 0.63 47.0 1.8 85.1 1.0 38.8 1.7
RAPTOR 193 0.41 161 0.67 108 0.60 104 0.53
GraphRAG 97.8 12.8 91.9 10.9 149 7.0 88.8 7.8
HippoRAG-v2 1089 0.71 380 1.71 544 1.5 188 3.5
Mem0 10804 0.79 1334 0.65 18483 1.6 2946 1.7
Cognee 11890 58.7 1185 4.8 4728 7.7 738 4.1
Self-RAG 11.4 3.1 8.1 2.4 5.3 0.82 5.2 1.0
MemGPT 433 9.4 101 10.5 392 11.7 85.5 12.3
Table 14: Maximum output token limits for various tasks
Task Max Output Tokens
RULER-QA 50
RULER-NIAH-MQ 100
∞Bench-QA 10
LongMemEval 100
EventQA 40
ICL_Five 20
Movie Recommendation 300
∞Bench-Sum 1,200
FactConsolidation 10
D.2 Settings of the RAG Agents
For the embedding model selection in Structure-Augmented RAG Agents and Agentic Memory
Agents, most approaches utilize OpenAI’s embedding models, such as Text-Embed-3-Small. While
for the HippoRAG-v2 method, we follow the same experimental setting as in Gutiérrez et al. [12],
employing the NV-Embed-v2 model.
We implement three open-sourced memory agents in our main experiments. (1) For Mem0, we use
memory.add() function to add the message with the content from each context chunk into the agent’s
memory repository during memory consolidation. During query execution, the relevant memory
elements are retrieved through memory.search() function. The retrieved memories are then integrated
into the query before being processed by the GPT-4o-mini backbone model to complete the requested
tasks. (2) For MemGPT, we employ the insert_passage() function during the memory consolidation
phase to inject long context chunks into the Archival Memory structure. During query execution, this
agent processes requests via the send_message() function which generates appropriate responses
based on the archived information. (3) For Cognee, we utilize the cognee.add() andcognee.cognify()
functions to construct the memory graph from input chunks wherein the memory consolidation
22

phase. During query execution, the cognee.search() function is used to retrieve contextually relevant
information from the memory graph based on the input query.
D.3 Settings of the Chunk Size
We use smaller chunk size (512) for synthetic context used in AR and CR. For some tasks based on
continuous text, such as ∞Bench and EventQA, we used a larger chunk size (4096). For tasks such as
MCC, Recom and LME(S), considering the characteristics of these tasks and the computational cost,
we also chose a larger chunk size (4096). For the two memory construction methods that are more
time-consuming, Mem0 and Cognee, we uniformly used a chunk size of 4096 across all datasets.
Chunk Size 512 4096
RULER-QA, NIAH-MQ ∞Bench-QA, ∞Bench-Sum
Dataset FactCon-SH, FactCon-MH MCC, Recom
LME(S*) EventQA, LME(S)
Table 15: The choice of chunk size for different datasets.
23