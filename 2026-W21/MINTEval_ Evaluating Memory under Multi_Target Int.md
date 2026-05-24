# MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems

**Authors**: Hyunji Lee, Justin Chih-Yao Chen, Joykirat Singh, Zaid Khan, Elias Stengel-Eskin, Mohit Bansal

**Published**: 2026-05-18 15:43:35

**PDF URL**: [https://arxiv.org/pdf/2605.18565v2](https://arxiv.org/pdf/2605.18565v2)

## Abstract
Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce MINTEval (Long-Horizon Memory under INTerference Evaluation), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, MINTEval has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are revised or interfered with by subsequent context, with accuracy degrading as the number of intervening updates increases.

## Full Text


<!-- PDF content starts -->

MINTEVAL: Evaluating Memory under Multi-Target
Interference in Long-Horizon Agent Systems
Hyunji Lee1∗Justin Chih-Yao Chen1∗Joykirat Singh1Zaid Khan1
Elias Stengel-Eskin2Mohit Bansal1
1UNC Chapel Hill2The University of Texas at Austin
Abstract
Agents in real-world settings operate over long and evolving horizons, where
information is repeatedly updated and may interfere across memories, requiring
accurate recall and aggregated reasoning over multiple pieces of information. How-
ever, existing benchmarks focus on static, independent recall and fail to capture
these dynamic interactions between evolving memories. In this paper, we study
how current memory-augmented agents perform in realistic, interference-heavy,
long-horizon settings across diverse domains and question types. To this end, we in-
troduce MINTEVAL(Long-Horizon Memory under INTerference Evaluation), an
analytical benchmark which features (1) long, highly interconnected contexts with
frequently updated information that induces substantial interference, (2) diverse
domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub com-
mits), enabling evaluation of domain generalization, and (3) diverse question types
that assess robustness to interference, including(i) single-target recalltasks requir-
ing retrieval of a specific target from long contexts, and(ii) multi-target aggregation
tasks requiring reasoning over multiple relevant pieces of information. Overall,
MINTEVALcontains 15.6k question-answering pairs over long-horizon contexts
averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evalu-
ate seven representative systems, including vanilla long-context LLMs, retrieval-
augmented generation methods, and memory-augmented agent frameworks. Across
all systems, we observe consistently low performance (avg. 27.9% accuracy), espe-
cially on questions requiring aggregated reasoning over multiple pieces of evidence.
Fine-grained analysis shows that performance is primarily limited by retrieval and
memory construction capabilities. Furthermore, current memory systems struggle
to recall and reason over earlier facts that are later revised or interfered with by sub-
sequent context, with performance degrading as the number of intervening updates
increases. These findings highlight the need for more robust memory management
systems for dynamic, long-horizon environments across varying domains. Code
and data are available athttps://github.com/amy-hyunji/MINTEval.
1 Introduction
Memory-augmented agents powered by large language models (LLMs) are increasingly being
developed to support a variety of tasks (e.g., long-horizon tasks [Huang et al., 2026, Gutiérrez
et al., 2025, Hu et al., 2025] and lifelong learning [Zheng et al., 2026, 2025, Liu et al., 2025]),
where information continuously accumulates over time [Ong et al., 2025, Kim et al., 2026]. In
many real-world settings, newly acquired information does not fully overwrite prior information,
but instead revises or builds upon existing states. For example, software systems and documents
evolve through successive revisions that introduce new features or modify existing syntax and
∗Equal contribution; order decided by a coin flip. Correspondence to:{hyunjil,cychen}@cs.unc.edu
Preprint.arXiv:2605.18565v2  [cs.CL]  19 May 2026

State TrackingDialogueWiki RevisionDreamlover was from Mariah Carey's 4th album, released in 1994…Title: Dreamlover (song)
The single was co-written by Mariah and Dave Hall, and co-produced …Dreamlover became Mariah's 8th #1 single on The Hot 100 …Dreamlover was Mariah's 7th #1 single on The Billboard Hot 100 …Long-horizon Input w/ Interference
Diverse and Realistic EnvironmentsGithub Commit
Limitations in Current Memory SystemsFull Context: Costly & Might Exceed Window Size
RAG: Optimized for Static Recall, Bad for Conflict
Memory-Augmented Agents: Focus on Override5 Question Categories w/ Interference 
Simple  History     Order  Count  Multi-Hop 
1M TokensExceed Limit
Costly
Doc Pool      Conflicting FactsMemEvolving with Temporal Conflict (facts are updated from 8th to 7th)Fails to Retrieve the Right Source• Skip/Override Prev. Info• Only Focus Last Change• Fail on Lookback QsMINTEval
Single-Target Recall        Multi-Target AggregationFigure 1:Left:MINTEVALspans four realistic domains: state tracking, dialogue, GitHub
commits, and Wikipedia revisions, with five question categories probing different aspects of memory
behavior.Middle:The contexts are inherently dynamic and continuously evolving, naturally creating
frequent destructive interference.Right:Existing memory systems show distinct failure modes: (1)
full-context methods are computationally expensive and exceed context limits, (2) RAG systems
often retrieve incorrect evidence due to conflicting information, and (3) memory-augmented agents
overemphasize recent information and underuse historical context, hurting lookback-style queries.
behaviors. In such settings, users may query specifications from older versions or compare differences
across revisions when migrating to newer releases. Similarly, during long-term interactions with
conversational agents, users continuously provide new information across multiple interactions that
may reinforce, modify, or contradict earlier preferences or personal attributes [Chen et al., 2026,
Mehri et al., 2026]. Users may ask about facts or preferences they no longer recall, or expect agents
to respond consistently with preferences expressed throughout past interactions. These real-world
settings require agents not only to preserve information over time, but also to understand how newly
acquired information relates to prior states, enabling agents to recall and aggregate information across
interactions rather than simply overwrite existing memories. However, as information accumulates
over long horizons,interference2naturally emerges, which is a well-studied phenomenon in human
memory [Underwood, 1957, Anderson and Neely, 1996] (Fig. 1, middle) where previously stored and
newly acquired information interact and conflict with one another, making retrieval and reasoning
over past information challenging.
A simple solution to answering such questions with long horizon context is to include all avail-
able context in the input, especially as model context lengths have grown substantially in recent
years [Team et al., 2024, Yang et al., 2025], but this remains inefficient and often exceeds practical
context length limits [Kim et al., 2026, Wang et al., 2025]. To address this, memory-augmented agents
have been proposed [Xu et al., 2025, Huo et al., 2026, Packer et al., 2024, Zhou et al., 2025], which
store, update, and retrieve information over time while preserving consistency. These approaches
have demonstrated stronger and more robust performance than both naive full-context usage and
standard retrieval-augmented generation (RAG). However, important gaps remain in understanding
how memory-augmented agents perform in real-world settings, as shown in Fig. 1 (right). As
shown in Table 1 (INTERDEP. and INTERFERENCEcolumns), existing memory benchmarks often
focus on long-horizon inputs composed of largely independent events with sparse interactions (e.g.,
concatenating unrelated contexts into a single long sequence [Hu et al., 2026, Wang et al., 2025]),
failing to capture the dense and evolvinginterference-heavycontexts observed in real-world memory.
Also, existing benchmarks [Wang et al., 2025, Wan and Ma, 2025] primarily focus on recall of recent
information, while overlookinglong-range lookback3(LOOKBACK) and reasoning tasks that require
aggregating multiple relevant targets (AGGR.). Moreover, existing benchmarks are often focused on
specific domains, particularly conversational environments [Tavakoli et al., 2026, Wu et al., 2025],
thereby failing to evaluatedomain generalization(M-DOMAIN).
2Here,interferenceencompasses both proactive interference, where old memories affect encoding of new information, and
retroactive interference, where new information overwrites existing ones.
3Bylong-range lookback, we mean queries about information from much earlier in the interaction history rather than the
latest state, e.g., if a person moved ten times, it may ask where they lived after the third move instead of where they live now.
2

Table 1: Comparison of MINTEVALwith prior memory benchmarks. We categorize benchmarks
by (1) input context properties: interdependent inputs (Interdep.), dense interference ( ≥10 depth;
Interference), and multi-domain coverage (M-Domain); and (2) question properties: multi-target
aggregation (Aggr.) and lookback to earlier context (LookBack).
Input Context Question Type
Benchmark Interdep. Interference M-Domain Aggr. LookBack
MemoryAgentBench [Hu et al., 2026]✗ ✗ ✓ ✓ ✗
Mem-α[Wang et al., 2025]✗ ✗ ✓ ✗ ✗
Locomo [Maharana et al., 2024]✓ ✗ ✗ ✓ ✗
LongMemEval [Wu et al., 2025]✓ ✗ ✗ ✓ ✗
BEAM [Tavakoli et al., 2026]✓ ✗ ✗ ✓ ✗
StoryBench [Wan and Ma, 2025]✓ ✗ ✗ ✗ ✗
OAKS [Kim et al., 2026]✓ ✗ ✗ ✓ ✗
MINTEVAL(Ours) ✓ ✓ ✓ ✓ ✓
To evaluate how memory-augmented agents perform under such settings, we introduce an analytical
benchmark, MINTEVAL(Long-Horizon Memory under INTerference Evaluation), which features
interference-heavy input contexts, queries requiringlong-range lookback and aggregated reasoning,
as well asdiverse domain and question types. As shown in Figure 1 left, MINTEVALspansfour
domains (state tracking, multi-turn dialogue, Wiki revisions, and Git commits), each involving
continuously evolving information streams with accumulated context. The evolution covers both
overwrite-style (edit-based) and append-style (accumulative) streams, enabling evaluation across
different memory dynamics under interference-heavy scenarios. The benchmark also includestwo
primary types of tasks4:Single-target recalltasks evaluate whether models can accurately retrieve
specific pieces of information under interference; (e.g.,“According to the previous revision of the
article, how many floors does the building have?”).Multi-target aggregationtasks require models
to identify and performaggregated reasoningover multiple relevant pieces of context, including
operations such as counting entities, ordering events, and combining information across updates.
For example, a multi-target query like“What syntax changes were made between version 1.2.30
and the current package versions?”requires recalling the syntax of both version 1.2.30 and the
current version, and then reasoning over the differences between them. We construct MINTEVAL
using both synthetic examples from existing benchmarks and LLM-generated questions produced by
Gemini-3.1-Pro [Google, 2026b] conditioned on the full interaction history. Overall, MINTEVAL
is a diverse and scalable benchmark containing an average of 3.9k questions per domain and 15.6k
question-answering pairs in total, built over long-horizon contexts averaging 138.8k tokens and
extending up to 1.8M tokens. Each context contains, on average, 86 temporally ordered updates. For
questions that are generated by the frontier model, we further conduct a human verification with six
annotators on 20% instances and find that 95.6% of them are valid.
Using MINTEVAL, we evaluate seven representative systems using Qwen3.6-35B-A3B [Yang et al.,
2025] and Gemini-3.1-Flash-Lite [Google, 2026a]: Full Context, RAG, HippoRAG [Gutiérrez et al.,
2025], MemAgent [Yu et al., 2025], AtomMem [Huo et al., 2026], Mem- α[Wang et al., 2025], and
SimpleMem [Liu et al., 2026]. Across all systems, MINTEVALremains highly challenging, with an
average accuracy of 27.9%; the best-performing system, MemAgent, achieves only 33.4% on average,
with failure modes described in Fig. 1 (right). We observe that performance varies across tasks and
domains. In particular, memory management systems perform strongly on bAbI [Weston et al., 2015],
which contains relatively short contexts and simple facts, achieving an average improvement of +9.9%
over non-memory baselines. However, on other domains with longer contexts and evolving revisions,
these systems often underperform the same baselines, with an average 3.0% drop. Also, performance
differs significantly by question type: simple recall questions have higher average accuracy (47.5%),
whereas systems perform poorly on questions requiring long-range lookback (avg. 21.0%), and
those requiringmulti-target aggregation(avg. 26.5%). To better understand where these failures
occur, we decompose errors into (1) failures in retrieval or memory construction, and (2) failures of
the answering agent to correctly use relevant information even when it is available in the context.
Our analysis shows that most errors stem from memory construction failures, which account for a
41.7% performance drop, while the answering stage contributes an additional 25.2% drop. Further
4More examples for each question type are in Table 2.
3

analysis shows that memory-augmented agents are sensitive to design choices such as the number
of iterative memory process steps, and are strongly biased toward insertion-based operations (avg.
76.8%) instead of deletion or update. Overall, our analysis reveals key strengths and limitations of
existing memory systems, emphasizing the need for approaches that are robust to interference-heavy
contexts, domain generalization, and various queries, including long-range lookback and aggregated
reasoning.
2 MINTEVAL: Long-Horizon M emory under INT erference Eval uation
Interference-heavy Contexts.MINTEVALfocuses on contexts withdensely interacting updates,
where information is repeatedly modified or contradicted over time (Figure 1, middle). Real-world
memory involves continual revisions and conflicting states. These dynamics expose the core chal-
lenges of memory systems: resolving temporal conflicts, preserving historical state, and maintaining
consistency over time. Such setups naturally induceproactive and retroactive interference[Under-
wood, 1957, Anderson and Neely, 1996] where retroactive interference occurs when new information
disrupts recall of older information, while proactive interference occurs when older memories inter-
fere with learning or recalling newer information. By incorporating both, our setup requires agents to
track evolving states, connect historical information, and resolve interference effectively.
Domains.MINTEVALconsists of four representative domains in which memory is frequently
helpful in practice. These domains differ in information structure, update dynamics, and reasoning
requirements, enabling evaluation of both memory behavior under varied interference patterns and
domain generalizationacross tasks (Examples and more details are in Appendix A.1.).
(1) STATETRACKING(bAbI). We use contexts from bAbI [Weston et al., 2015], where information
is represented as simple symbolic facts that are updated through sequential, localized changes, often
overwriting previous states. Questions query the changing states and facts described in the context.
This domain requires systems to integrate sequential updates, track state transitions, and perform
temporal reasoning over current and historical states.
(2) DIALOGUE-BASEDMULTI-TURNINTERACTIONS(HorizonBench). Building on Horizon-
Bench [Li et al., 2026], a long-horizon personalization benchmark with users and conversation
histories, we form long-horizon multi-turn dialogue contexts by concatenating multiple dialogue
sessions. We then generate new questions targeting personal preferences and attributes whose relevant
information is distributed across interactions and often implicitly expressed through natural language
interactions. This domain evaluates whether memory systems can track and updateimplicituser-state
changes, such as evolving preferences, over time.
(3) FACTUALKNOWLEDGEQA (Wiki Revisions). We introduce the Wiki Revisions split, which we
construct from long-horizon Wikipedia revision histories, where each instance consists of chrono-
logically ordered article revisions. We generate questions targeting both factual knowledge in the
articles and how information evolves across revisions. As facts may be added, modified, contradicted,
or removed over revisions, answering these questions requires memory systems to reconstruct prior
states, track provenance, and distinguish outdated from current information.
(4) CODE ANDFILESEVOLUTION(Git Commits). We also introduce the Git Commits splits, which
constructs long-horizon contexts from GitHub commit histories, where each instance contains a
single repository and its chronological commits. We construct questions that target both code details
in the repository and how implementations evolve across commits. Unlike natural-language revision
histories, code evolution often involves tightly coupled cross-file edits and evolving identifiers (e.g.,
function name or API signature), thus requiring a memory system to recover implicit differences
between snapshots and changing program behavior.
Question Types.MINTEVALincludes two primary categories of tasks that target different aspects
of memory behavior under densely interacting updates and interference-heavy contexts:SINGLE-
TARGET RECALLandMULTI-TARGET AGGREGATION(Examples in Table 2).
SINGLE-TARGETRECALL. These tasks evaluate whether a model can correctly identify and
retrieve a single target from long contexts with dense updates. We considertwovariants: Simple
questions, which require retrieving the most recent state after a sequence of updates, and History
(lookback-style) questions, which require recovering an earlier state despite subsequent updates and
potentially conflicting information. Simple questions evaluate robustness toproactiveinterference,
4

Table 2: Example from each question type in MINTEVAL.
Single-Target Recall
Simple How many floors does the article state the building has?
History In the version two edits prior, which team is named 1919 County Champion?
Multi-Target Aggregation
Ordering In which order was the section added to the article?
Counting How many different individuals have been listed as the album’s producer?
Multihop What was episode 5’s title just before episode 4’s third title change?
Table 3: Dataset statistics across four domains.Depthdenotes the number of turns, revisions, or
commits in each example.kindicates values reported in thousands. Further details in Appendix A.5.
bAbI HorizonBench Wiki Revisions Git Commits
# Sessions 99 100 196 200
# Total Questions 5.7k 6.9k 1.5k 1.6k
Avg. Context
StatisticsDepth 42 142 99 61
Tokens 0.3k 274k 195k 86k
Question
DistributionsSingle-Target Recall 2.7k 3.9k 0.8k 0.9k
Multi-Target Agg. 3k 3k 0.6k 0.7k
where previously stored information may interfere with encoding or retrieving newer states. In
contrast, History questions evaluate robustness toretroactiveinterference, where newly introduced
information may overwrite or obscure previously stored states. History questions require agents to
identify the relevant point in the context using cues and respond using the corresponding information.
Together, these tasks evaluate whether models can both maintain up-to-date representation and
preserve access to prior states over long contexts.
MULTI-TARGETAGGREGATION. These tasks require agents to identify multiple targets distributed
across different updates and aggregate them to produce the correct answer. We considerthree
variants based on the type of aggregation required. (1) Ordering questions require recovering the
correct temporal order of events under dense updates. (2) Counting questions require aggregating
occurrences across updates, such as determining how many times an event happened or how long a
particular state persisted. (3) Multihop questions require reasoning over multiple targets, such as
comparing information across updates or performing bridge reasoning over interdependent events.
These three tasks evaluate whether models can identify multiple targets, integrate information across
updates, and reason over their relationships despite interference from intervening updates.
Question Generation Pipeline.Depending on the availability and structure of metadata in each
domain, we adopted different procedures for constructing question-answer pairs. For bAbI, we parsed
each fact into a (subject, object, verb) tuple and generated a question by filling predefined templates
with the extracted information, following a procedure similar to Kim et al. [2026]. For HorizonBench,
we used the metadata provided by Li et al. [2026], which tracks temporal changes such as evolving
user preferences. We constructed question templates and filled them using the metadata, similar
to bAbI. For Wiki Revisions and Git Commits, we generate question-answer pairs by prompting
Gemini-3.1-Pro [Google, 2026b] with revision metadata, including revision_ids ,timestamp ,
editor ,comment . We conduct a human validation process with six annotators, including three
authors and three non-authors, on 20% of the sessions (40 out of 200 sessions for Git Commits and
42 out of 196 sessions for Wiki Revisions). For each session, annotators are asked to evaluate one
question-answer pair from each question type, for question naturalness and answer correctness. The
results show that 95.6% of the generated samples contain natural questions with correct answers.
More details about question generation and human validation are in Appendix A.3.
Dataset Statistics.Table 3 summarizes the scale and composition of MINTEVALacross domains.
On average, each domain contains 149 sessions, with contexts averaging 86 updates in depth and
138.8k tokens in length. Across domains, MINTEVALincludes an average of 2k questions for
single-target recalland 1.8k formulti-target aggregation. More details are in Appendix A.5.
5

3 Experiments
3.1 Setup
Baselines.Our baselines fall into three main categories.(1) Full Context:methods without an ex-
plicit memory module, where the model receives the entire context as input.(2) Retrieval-Augmented
Generation (RAG):RAG denotes the standard retrieval-augmented generation framework, which re-
trieves relevant documents using dense vector similarity [Lewis et al., 2021]. HIPPORAG [Gutiérrez
et al., 2025] extends this framework with a graph-structured retrieval mechanism that captures richer
relationships between documents. Unless otherwise specified, we retrieve the top-5 contexts.5(3)
Memory-Augmented Agents:We evaluate several trained memory systems that explicitly learn how
to store, update, and retrieve information under different training paradigms. For all methods, we use
the officially released checkpoints. For bAbI, every 15 facts are grouped into a single chunk. For
HorizonBench, each dialogue session is treated as a chunk; for Wiki Revisions and Git Commit, each
revision is treated as a chunk.6MEMAGENT[Yu et al., 2025] is built on Qwen2.5-14B-Instruct [Yang
et al., 2024], and it incrementally updates memory using an overwriting strategy, constructing query-
specific memory representations. ATOMMEM[Huo et al., 2026] formulates memory management as a
sequential decision-making problem, decomposing actions into atomic CRUD (Create, Read, Update,
Delete) operations, and is based on Qwen3-8B [Yang et al., 2025]. MEM- α[Wang et al., 2025] trains
Qwen3-4B model to organize memory into three types, i.e., core, semantic, and episodic memory.
SIMPLEMEM[Liu et al., 2026] is a state-of-the-art memory system consisting of a three-stage
pipeline: semantic structured compression, which converts unstructured interactions into compact
multi-view memory units; online semantic synthesis, which incrementally merges related contexts
to reduce redundancy; intent-aware retrieval, which dynamically determines retrieval scope and
constructs targeted retrieval contexts.
Models.Our evaluation pipeline consists of three components. (1)Memory managerconstructs a
compact memory representation of a long-horizon, evolving input context. For SIMPLEMEM, we use
Gemini-3.1-Flash-Lite [Google, 2026a]. For MEM- α, MEMAGENT, and ATOMMEM, we use their
publicly released checkpoints. (2)Answering agenttakes either the full context, retrieved context,
or managed memory as input and generates the final answer. Unless otherwise specified, we use
Qwen3.6-35B-A3B [Yang et al., 2025] as the answering agent and additionally evaluate Gemini-
3.1-Flash-Lite. We set the maximum context length to 65k and 1M tokens for Qwen3.6-35B-A3B
and Gemini-3.1-Flash-Lite, respectively. (3)Embedding modelis used in retrieval-based systems to
retrieve relevant contexts by computing similarity scores. Unless otherwise specified, we use Qwen3-
Embedding-4B [Zhang et al., 2025] and additionally evaluate Gemini-Embedding-001 [Google,
2025]. Further details are provided in Appendix B.
Evaluation Metrics.We evaluate using Exact Match after standard text normalization, following
prior memory benchmarks [Kim et al., 2026, Wang et al., 2025]. For HorizonBench only, we provide
a set of candidate answers for each question, similar to a multiple-choice evaluation setting, since
answers may not appear verbatim in the context and can admit multiple valid surface forms.
3.2 Results
Existing Methods Struggle on MINTEVAL.As shown in Table 4, existing systems struggle
on MINTEVAL, achieving only 27.7% average accuracy across the six evaluated systems. Even
advanced memory systems perform poorly: the best overall result reaches just 33.4% averaged
across all domains, suggesting that the benchmark remains far from saturated. Across question types,
both RAG and memory-based methods perform relatively well on Simple queries (avg. 47.5%),
suggesting that retrieving the most recent value is comparatively easy. However, performance drops
substantially on History questions that require long-range lookback (avg. 21.0%) and onmulti-target
aggregationquestions (avg. 26.5%), which require tracking updates over time, resolving conflicts,
or aggregating information across multiple targets. Among memory-based approaches, MemAgent
achieves the strongest overall performance (avg. 33.4%) and shows relatively robust generalization
across domains. We hypothesize that this gain comes from the construction of query-specific memory
representations, whereas AtomMem and Mem- αbuild a shared memory from the input context
5We provide an analysis of performance under different numbers of retrieved documents in Appendix C.5, where we
observe that retrieving five documents provides a strong overall performance.
6We additionally provide an ablation study on chunk size in Section 4.4.
6

Table 4: Results on Qwen3.6-35B-A3B. We compare three categories of methods: Full Context,
RAG, and Memory-Augmented Agents. Cells are color-coded by score to highlight performance
patterns, transitioning from dark red (lowest) through light red and light green to dark green (highest).
Full RAG HippoRAG AtomMem Mem-αMemAgent
bAbISimple 57.4 66.7 70.0 65.2 82.6 85.7
History 16.1 16.7 33.3 36.3 44.9 36.0
Ordering 22.0 37.5 50.0 58.1 64.7 59.0
Counting 40.5 80.0 80.0 43.8 70.4 24.3
Multihop 30.8 40.0 41.7 30.7 61.0 51.7
Wiki RevisionsSimple 23.3 36.7 37.9 16.9 49.9 54.2
History 14.5 30.2 31.1 15.7 20.6 28.8
Ordering 10.9 6.9 4.1 2.4 13.5 38.3
Counting 22.2 15.9 19.1 14.3 25.0 36.5
Multihop 11.2 26.8 27.1 16.2 17.3 23.7
Git CommitsSimple 82.0 81.5 81.9 40.8 71.7 82.3
History 27.1 30.1 30.6 19.0 4.8 24.0
Ordering 17.7 40.6 44.5 13.8 17.0 55.9
Counting 21.1 38.9 14.1 24.7 18.8 51.6
Multihop 13.1 12.8 39.6 27.3 8.5 34.7
HorizonBenchSimple 11.3 11.6 12.1 4.4 7.5 7.5
History 9.9 10.3 10.9 2.4 5.8 3.8
Ordering 3.5 4.2 3.9 1.0 0.4 6.7
Counting 0.8 2.8 4.2 2.9 1.6 1.8
Multihop 11.9 29.7 33.9 30.0 24.0 28.1
Overall Avg. 21.0 29.5 32.3 22.1 28.0 33.4
and reason over the same question-agnostic memory structure. Nevertheless, MemAgent’s average
performance on MINTEVALremains low, indicating that MINTEVALis challenging even for strong
existing memory systems.
MINTEVALShows Limited Cross-Domain Generalization.The overall results in Table 4 exhibit
substantial variance, with no single method consistently outperforming others across domains and
question types. For example, MemAgent achieves 85.7% on bAbI Simple but drops to 7.5% on
HorizonBench for the same task, while HippoRAG attains 70.0% on bAbI Simple and remains
relatively more robust on HorizonBench Simple with 12.1%. These results suggest limited cross-
domain generalization. In general, single-target recall tasks ( Simple andHistory ) are easier than
multi-target aggregation tasks ( Ordering ,Counting , and Multihop ), with average accuracies of
34.3% and 26.5%, respectively. This gap arises because aggregation tasks require identifying multiple
relevant targets and performing additional reasoning over them. Within single-target recall, History
questions (21.0%) are consistently harder than Simple questions (47.5%) since they require retrieving
past rather than current states, with difficulty increasing for longer lookback distances (Section 4.2).
Among aggregation tasks, Ordering questions are most difficult (24.0%) as they require recovering
the exact event sequence without partial credit. Overall, MINTEVALhighlights persistent challenges
with interference-heavy contexts and long-horizon dependencies, with large performance gaps across
both domains and question types.
Even the State-of-the-art Memory System Struggles.We further evaluate SimpleMem [Liu et al.,
2026], a state-of-the-art memory system using frontier models (Gemini-3.1-Flash-Lite as answering
agent and Gemini-Embedding-001 as embedding model), to investigate the performance of a strong
memory system combined with the frontier models. Despite using a stronger embedding model and
answering agent, SimpleMem achieves only 30.3 EM on average. We find that this degradation
stems from SimpleMem’s aggressive memory compression strategy. Such compression is effective
on conversational memory benchmarks such as LoCoMo [Maharana et al., 2024], where contexts are
relatively short (avg. 109 characters) and less interconnected. In contrast, as MINTEVALcontains
long, evolving revisions (avg. 184k characters) with substantial interdependence and interference,
and thus, aggressive compression and deduplication are prone to discarding important provenance
information and historical details. Consistent with the observation, SimpleMem performs relatively
well on bAbI, which contains shorter and simpler contexts, but degrades substantially on Wiki
7

RAG HippoRAG AtomMem MemAgent020406080100Accuracy (%)Evidence Exists
Memory GapAccuracy
Answer GapFigure 2: Error due to missing evidence in mem-
ory (green) or incorrect answers despite the ev-
idence being present (green–blue gap). Only
58.3% of cases contain the required evidence,
making retrieval/memory construction the main
bottleneck; answering errors add a 25.2% drop. A
perfect system would reach 100%.
  10   20   30   40   50   60   70   80   90  100
Lookback Distance0.00.10.20.30.40.50.60.70.8Accuracy
Method
Full
BaseRAG
HippoRAG
AtomMem
MemAgentMethod
Full
BaseRAG
HippoRAG
AtomMem
MemAgentFigure 3: Performance (y-axis) vs. Lookback
Distance (x-axis). Accuracy drops as lookback
distance increases, with the largest drops in Full
Context and retrieval methods (RAG, HippoRAG).
Memory-based agents degrade less, suggesting
greater robustness from better temporal encoding
and compact historical representations.
Revisions. In particular, revision provenance is often lost during compression, as facts may be
paraphrased or rewritten. Without explicit metadata linking facts to their originating revisions,
retrieval relies primarily on keywords and embeddings, making queries such as retrieving the content
of “Revision 53” especially challenging. Further analyses are provided in Appendix C.6.
4 Analysis
4.1 Retrieval and Memory Construction Remain the Primary Bottleneck
We note that the RAG and memory-based systems we use consist of two stages: (1) retrieving relevant
information or constructing memory, and (2) generating an answer conditioned on the retrieved
context or constructed memory using an answering agent. Failures can therefore arise from two
sources: failures in retrieval or memory construction, or failures in answer generation. To investigate
the source of these failures, we analyze two retrieval systems (RAG and HippoRAG) and two
memory systems (MemAgent and AtomMem) on the Wiki Revisions. Using Gemini-3.1-Flash-Lite,
we determine whether failures arise from retrieval/memory construction or answer generation by
checking whether the retrieved documents or constructed memories contain the supporting evidence
required to answer the question.7We conduct this analysis on Simple ,History , and Multihop
questions, as answers to Ordering andCounting questions are often not explicitly stated in the
retrieved context.
In Figure 2, we view 100% as the upper bound, since all questions are generated directly from
the full history, meaning that the required evidence always exists by design in the retrieval pool.
Relative to this upper bound, the largest performance degradation comes from retrieval and memory
construction failures, resulting in an average drop of 41.7% (only 58.3% of cases contain the
supporting evidence). When the evidence is present, an average of 25.2% drop can be attributed to
failures of the answering agent (blue bars). These findings indicate that current retrieval and memory
construction are the primary bottleneck, while the strength of the answering agent also plays a non-
trivial role in performance. Although all four systems use the same answering agent, differences in
how retrieved information and memories are constructed and presented lead to substantial performance
gaps. For example, AtomMem shows particularly large degradation in answer generation performance,
as we observe that it produces relatively longer memories compared to other methods. We further
analyze the effect of different answering agents on performance in Appendix C.1. Under the Full
Context setting, replacing answering agent from Qwen3.6-35B-A3B to Gemini-3.1-Flash-Lite yields
7We use an LLM-based evaluation for analysis instead of lexical matching because we observe that the same words may
appear multiple times in the context without being relevant to the question, making simple word matching imprecise.
8

0 1 3 5
# Distractors10203040Accuracy (%)
Full (OOD)
Full (ID)BaseRAG (OOD)
BaseRAG (ID)MemAlpha (OOD)
MemAlpha (ID)Figure 4: Performance under varying numbers
of distractors for bothIn-Domain (ID)andOut-
of-Domain (OOD)settings. Overall, the per-
formance drops as the number of distractors in-
creases, while Full Context shows no significant
difference acrossIDandOODdistractors.
Simple History Ordering Counting Multihop020406080Accuracy (%)CS=7 CS=15 CS=30Figure 5: Performance vs. different chunk sizes
when processing memories for the MemAgent
model (CS = Chunk Size). Increasing CS gen-
erally improves performance, and Simple ques-
tions are the least sensitive to CS, since it only
requires recalling recent information.
a substantial performance improvement (55.7%). In contrast, this gap becomes much smaller when
retrieval or memory systems are introduced (avg. 1.7%), indicating that once memory systems are
involved, performance differences are driven less by the capability of the answering agent itself and
more by how effectively the retrieval or memory system constructs context for the agent.
4.2 Longer Lookback Distances Hurt Performance, and Temporal Markers Help
We analyze how performance changes as the required lookback distance increases for History
questions, which ask about information from earlier revisions (e.g., ‘In the versiontwo edits prior,
which team is named 1919 County Champion?’). Here, lookback distance refers to the number
of revisions between the queried information and the current version. We evaluate five settings
(Full, RAG, HippoRAG, AtomMem, and MemAgent) on the Wiki Revisions subset across questions
with varying lookback distances. As shown in Figure 3, performance generally decreases as the
required lookback distance increases, suggesting that retrieving or preserving information from
distant revisions is increasingly difficult. The largest degradation is observed for the Full Context
setup and retrieval-based methods (RAG and HippoRAG), whose accuracy drops substantially as the
number of lookback distance grows. In contrast, although memory-augmented agents also exhibit
some degradation, the decline is noticeably smaller. We hypothesize that this greater robustness
arises because memory-based agents can better encode temporal order and preserve relationships
between events by accumulating historical information into memory. We further investigate how
incorporating explicit temporal cues into the context and questions affects performance. To study this,
we augment facts and questions with temporal cues such as dates or timestamps (e.g., October 2023).
In Appendix C.3, we find that adding these temporal cues substantially reduces the performance
degradation for both Full Context and RAG systems: the performance drop from the first to the last
lookback step decreases from 13.22 without temporal cues to 5.48 with temporal cues for Full Context,
and from 31.43 to 10.45 for RAG. These results suggest that interference can be mitigated through
explicit markers as they allow agents to distinguish similar or conflicting facts across revisions.
4.3 Adding Distractors Further Degrades Performance, Especially for RAG
To evaluate how agents perform in real-world scenarios with noisy distractors, we study how perfor-
mance changes as different types and numbers of distractors are inserted between facts in the bAbI
dataset. We insert two types of sentence-level distractors with varying numbers of inserted sentences
(1, 3, and 5), measuring performance across Full, RAG, and Mem- α.8The distractor types are: (1)
Out-of-Domain (OOD)distractors drawn from novels (similar to BABILong [Kuratov et al., 2024]),
which differ in style and structure from bAbI facts; (2)In-Domain (ID)distractors, which follow the
same simple, compositional structure as bAbI facts. As shown in Figure 4, performance generally
decreases as the number of distractors increases across all agents. However, the degradation is most
8In both cases, we ensure distractors do not alter the answer by removing sentences sharing the same subject and object.
9

pronounced withOODdistractors for RAG, which tends to retrieve distracting sentences more fre-
quently. In contrast, for Mem- αand the Full Context baseline, the difference in performance between
OODandIDdistractors is relatively small. We provide a more fine-grained analysis in Figure 10
in the Appendix, showing thatIDdistractors more strongly affect questions such as Counting and
History compared to simpler queries like Simple , suggesting that tasks requiring aggregation or
tracking over multiple facts are more susceptible to interference.
4.4 Ablation Studies and Analysis of Memory-Augmented Agents
Fewer Memory Update Iterations Improve Performance.In memory systems, long contexts
can be processed using different chunk granularity (e.g., a 1M token input can be divided into 10
100k-sized chunks or 100 10k chunks). We investigate how different chunk sizes, which determine
the number of memory update iterations, affect overall performance on bAbI using MemAgent. As
shown in Figure 5, increasing the chunk size, i.e., reducing the number of memory modifications,
generally improves performance as more frequent modifications may introduce unintended overwrites
or removals of previously stored information, making it difficult to maintain a coherent memory
representation. This is especially apparent for History orCounting questions, which require
integrating information over long horizons. The impact is relatively small on Simple questions,
which mostly rely on recent information.
Existing Memory Systems Strongly Biased Toward Appending Rather than Editing or Deleting.
Both AtomMem and Mem- αmanage memory through function calls corresponding to three opera-
tions: (1) insertion, (2) modification, and (3) deletion. Analyzing the frequency of these operations,
we observe that both systems are heavily biased toward insertion across all datasets, which accounts
for 87.6% of operations in AtomMem and 65.9% in Mem- αon average (Figure 9 in Appendix).
This suggests that, although revisions in MINTEVALare often incremental refinements of earlier
memory entries, agents struggle to recognize these updates as modifications because many changes
are frequently expressed implicitly and relationships between revisions are not properly captured,
leading to redundant memory insertions. This issue is further exacerbated by the coarse granularity
of memory operations. Both systems tend to operate on large chunks rather than in fine-grained units,
making it difficult to detect and update small differences within existing entries. As a result, even
minor changes are often inserted as new information instead of modifying existing memory. Overall,
these findings highlight the need for more balanced memory management, particularly stronger
modification and deletion capabilities, finer-grained updates, and better distinction between new and
updated information. Detailed results are in Appendix C.4.9
5 Related Work
Memory-Augmented Agents.Memory-augmented agents span several paradigms. RAG-based
approaches such as HippoRAG [Gutiérrez et al., 2025] organize extracted knowledge into graphs
for associative multihop retrieval. Among pipeline-based systems, MemGPT [Packer et al., 2024]
manages OS-inspired hierarchical memory tiers via a controller that pages information in and out
of context, while SimpleMem [Liu et al., 2026] maintains selectively-pruned running summaries.
Among training-based approaches, MemAgent [Yu et al., 2025] and Memory-R1 [Yan et al., 2026]
use RL to learn structured write/retrieve/delete policies; MEM1 [Zhou et al., 2025] trains memory
compression jointly with reasoning via RL; and AtomMem [Huo et al., 2026] learns to decompose
memory management into atomic CRUD operations via SFT and GRPO. Drawing on cognitive
science, structured memory systems assign distinct roles to episodic and semantic memory. Mem-
α[Wang et al., 2025] trains an RL agent over a multi-tier hierarchy, while REMem [Shu et al., 2026]
constructs a dynamic memory graph for episodic retrieval, and SYNAPSE [Jiang et al., 2026] unifies
episodic and semantic memory via spreading activation. Across these lines of work, a common
assumption is that the goal of memory is to surface the most current and relevant state in response to
a query. This shapes not only system design but also evaluation: models are typically assessed based
on whether they return the correct answer for the latest state, while largely overlooking their ability
to recall or aggregate information from earlier states. MINTEVALaddresses this gap by evaluating
how well systems can recall and aggregate information in evolving and interference-heavy contexts.
9We further conduct ablation studies and analysis over RAG performance in Appendix C.5.
10

Memory Evaluation in Large Language Models.A variety of benchmarks have been proposed
to evaluate memory systems in large language models. Conversational benchmarks [Maharana
et al., 2024, Wu et al., 2025] and QA-based benchmarks [Hu et al., 2026] evaluate retrieval and
temporal reasoning, but typically involve less interconnected contexts and focus on questions about
the most recent information. Recent benchmarks such as StoryBench [Wan and Ma, 2025] and
RealMem [Bian et al., 2026] introduce more densely interconnected contexts that naturally induce
interference, but the interference events remain sparse and they still focus on the most recent
information. OAKS [Kim et al., 2026] is the closest benchmark to MINTEVAL, as it also features
naturally occurring interference and question answering over long-form contexts. However, as shown
in Table 1, OAKS contains substantially fewer interference events (avg. 4.7) than MINTEVAL
(avg. 86) and does not include long-range lookback questions across multiple domains. Overall,
MINTEVALprovides a broader and more challenging evaluation setting for memory systems,
covering interference-heavy contexts, diverse lookback distances, and aggregation-based reasoning
across multiple domains.
6 Conclusion
To evaluate memory-augmented agents in realistic long-horizon environments, we introduce MINTE-
VAL, an analytical benchmark characterized by interference-heavy contexts, long-range dependencies,
and multi-target aggregation reasoning. It spans four domains (state tracking, multi-turn dialogue,
Wikipedia revisions, and GitHub commits) and five question types covering both single-target re-
call and multi-target aggregation. Together, these provide a unified framework for evaluating the
robustness of memory systems under interference-heavy settings, long-range lookback reasoning,
aggregation across multiple targets, and cross-domain generalization, capabilities that remain largely
underexplored in prior benchmarks. MINTEVALremains far from saturated: the average accuracy
across systems is only 27.9%, and the strongest model achieves just 33.4%. Performance degrades
substantially on questions that require lookback or aggregated reasoning, with retrieval and memory
construction emerging as the dominant bottleneck. These findings suggest that real-world memory
requires solving not only a long-context retrieval problem, but also faithful preservation of evolving
states, fine-grained memory updates, and reasoning over temporally distributed evidence.
Acknowledgments
We would like to thank the annotators: Hanqi Xiao, Vu Hoang Thien An, and Jefrey Bergl. This work
was supported by Microsoft Agentic AI Research and Innovation (AARI) grant program, NDSEG
PhD Fellowship, NSF-AI Engage Institute DRL-2112635, and NSF-CAREER Award 1846185. The
views contained in this article are those of the authors and not of the funding agency.
References
Michael C. Anderson and James H. Neely. Chapter 8 - interference and inhibition in memory
retrieval. In Elizabeth Ligon Bjork and Robert A. Bjork, editors,Memory, pages 237–313.
Academic Press, San Diego, 1996. ISBN 978-0-12-102570-0. doi: https://doi.org/10.1016/
B978-012102570-0/50010-0. URL https://www.sciencedirect.com/science/article/
pii/B9780121025700500100.
Haonan Bian, Zhiyuan Yao, Sen Hu, Zishan Xu, Shaolei Zhang, Yifu Guo, Ziliang Yang, Xueran Han,
Huacan Wang, and Ronghao Chen. Realmem: Benchmarking llms in real-world memory-driven
interaction, 2026. URLhttps://arxiv.org/abs/2601.06966.
Tiantian Chen, Jiaqi Lu, Ying Shen, and Lin Zhang. Es-memeval: Benchmarking conversational
agents on personalized long-term emotional support. InProceedings of the ACM Web Conference
2026, page 5810–5821. ACM, April 2026. doi: 10.1145/3774904.3792143. URL http://dx.
doi.org/10.1145/3774904.3792143.
Google. Gemini-embedding-001. https://ai.google.dev/gemini-api/docs/embeddings ,
2025.
11

Google. Gemini 3.1 flash-lite preview: Model documentation. https://ai.google.dev/
gemini-api/docs/models/gemini-3.1-flash-lite-preview, 2026a.
Google. Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog.google/
innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/, 2026b.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory:
Non-parametric continual learning for large language models, 2025. URL https://arxiv.org/
abs/2502.14802.
Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu, Wenqi Shao, and Ping Luo. HiAgent: Hier-
archical working memory management for solving long-horizon agent tasks with large language
model. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar,
editors,Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 32779–32798, Vienna, Austria, July 2025. Association for Com-
putational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1575. URL
https://aclanthology.org/2025.acl-long.1575/.
Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental
multi-turn interactions, 2026. URLhttps://arxiv.org/abs/2507.05257.
Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan,
Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong
Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao,
Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen
Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong,
Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali
Payani, Yu Wang, Haitong Ma, Wenya Wang, Chenguang Wang, Yu Zhang, Xin Wang, Yongfeng
Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Xue Liu, Yizhou Sun, Wei Wang, Julian McAuley,
James Zou, Jiawei Han, Philip S. Yu, and Kai Shu. Rethinking memory mechanisms of foundation
agents in the second half: A survey, 2026. URLhttps://arxiv.org/abs/2602.06052.
Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, and Yankai Lin. Atommem : Learnable dynamic
agentic memory with atomic memory operation, 2026. URL https://arxiv.org/abs/2601.
08323.
Hanqi Jiang, Junhao Chen, Yi Pan, Ling Chen, Weihang You, Yifan Zhou, Ruidong Zhang, Andrea
Sikora, Lin Zhao, Yohannes Abate, and Tianming Liu. Synapse: Empowering llm agents with
episodic-semantic memory via spreading activation, 2026. URL https://arxiv.org/abs/
2601.02744.
Jiyeon Kim, Hyunji Lee, Dylan Zhou, Sue Hyun Park, Seunghyun Yoon, Trung Bui, Franck Dernon-
court, Sungmin Cha, and Minjoon Seo. Can large language models keep up? benchmarking online
adaptation to continual knowledge streams, 2026. URL https://arxiv.org/abs/2603.07392 .
Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and
Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack,
2024. URLhttps://arxiv.org/abs/2406.10149.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https:
//arxiv.org/abs/2005.11401.
Shuyue Stella Li, Bhargavi Paranjape, Kerem Oktar, Zhongyao Ma, Gelin Zhou, Lin Guan, Na Zhang,
Sem Park, Lin Chen, Diyi Yang, Yulia Tsvetkov, and Asli Celikyilmaz. Horizonbench: Long-
horizon personalization with evolving preferences, 2026. URL https://arxiv.org/abs/2604.
17283.
Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu
Yao. Simplemem: Efficient lifelong memory for llm agents, 2026. URL https://arxiv.org/
abs/2601.02553.
12

Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang,
Daocheng Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, and Ding Wang.
Memverse: Multimodal memory for lifelong learning agents, 2025. URL https://arxiv.org/
abs/2512.03627.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of llm agents. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
13851–13870, 2024.
Shuhaib Mehri, Priyanka Kargupta, Tal August, and Dilek Hakkani-Tür. Multisessioncollab: Learning
user preferences with memory to improve long-term collaboration, 2026. URL https://arxiv.
org/abs/2601.02702.
Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seung-
won Hwang, Dongha Lee, and Jinyoung Yeo. Towards lifelong dialogue agents via timeline-based
memory management. InProceedings of the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), pages 8631–8661, 2025.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E.
Gonzalez. Memgpt: Towards llms as operating systems, 2024. URL https://arxiv.org/abs/
2310.08560.
Yiheng Shu, Saisri Padmaja Jonnalagedda, Xiang Gao, Bernal Jiménez Gutiérrez, Weijian Qi,
Kamalika Das, Huan Sun, and Yu Su. Remem: Reasoning with episodic memory in language
agent, 2026. URLhttps://arxiv.org/abs/2602.13530.
Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, and J Ross
Mitchell. Beyond a million tokens: Benchmarking and enhancing long-term memory in llms, 2026.
URLhttps://arxiv.org/abs/2510.27246.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, Xinyang Geng, Fred
Alcober, Roy Frostig, Mark Omernick, Lexi Walker, Cosmin Paduraru, Christina Sorokin, Andrea
Tacchetti, Colin Gaffney, Samira Daruki, Olcan Sercinoglu, Zach Gleicher, Juliette Love, Paul
V oigtlaender, Rohan Jain, Gabriela Surita, Kareem Mohamed, Rory Blevins, Junwhan Ahn, Tao
Zhu, Kornraphop Kawintiranon, Orhan Firat, Yiming Gu, Yujing Zhang, Matthew Rahtz, Manaal
Faruqui, Natalie Clay, Justin Gilmer, JD Co-Reyes, Ivo Penchev, Rui Zhu, Nobuyuki Morioka,
Kevin Hui, Krishna Haridasan, Victor Campos, Mahdis Mahdieh, Mandy Guo, Samer Hassan,
Kevin Kilgour, Arpi Vezer, Heng-Tze Cheng, Raoul de Liedekerke, Siddharth Goyal, Paul Barham,
DJ Strouse, Seb Noury, Jonas Adler, Mukund Sundararajan, Sharad Vikram, Dmitry Lepikhin,
Michela Paganini, Xavier Garcia, Fan Yang, Dasha Valter, Maja Trebacz, Kiran V odrahalli,
Chulayuth Asawaroengchai, Roman Ring, Norbert Kalb, Livio Baldini Soares, Siddhartha Brahma,
David Steiner, Tianhe Yu, Fabian Mentzer, Antoine He, Lucas Gonzalez, Bibo Xu, Raphael Lopez
Kaufman, Laurent El Shafey, Junhyuk Oh, Tom Hennigan, George van den Driessche, Seth Odoom,
Mario Lucic, Becca Roelofs, Sid Lall, Amit Marathe, Betty Chan, Santiago Ontanon, Luheng He,
Denis Teplyashin, Jonathan Lai, Phil Crone, Bogdan Damoc, Lewis Ho, Sebastian Riedel, Karel
Lenc, Chih-Kuan Yeh, Aakanksha Chowdhery, Yang Xu, Mehran Kazemi, Ehsan Amid, Anastasia
Petrushkina, Kevin Swersky, Ali Khodaei, Gowoon Chen, Chris Larkin, Mario Pinto, Geng Yan,
Adria Puigdomenech Badia, Piyush Patil, Steven Hansen, Dave Orr, Sebastien M. R. Arnold,
Jordan Grimstad, Andrew Dai, Sholto Douglas, Rishika Sinha, Vikas Yadav, Xi Chen, Elena
Gribovskaya, Jacob Austin, Jeffrey Zhao, Kaushal Patel, Paul Komarek, Sophia Austin, Sebastian
Borgeaud, Linda Friso, Abhimanyu Goyal, Ben Caine, Kris Cao, Da-Woon Chung, Matthew
Lamm, Gabe Barth-Maron, Thais Kagohara, Kate Olszewska, Mia Chen, Kaushik Shivakumar,
Rishabh Agarwal, Harshal Godhia, Ravi Rajwar, Javier Snaider, Xerxes Dotiwalla, Yuan Liu,
Aditya Barua, Victor Ungureanu, Yuan Zhang, Bat-Orgil Batsaikhan, Mateo Wirth, James Qin, Ivo
Danihelka, Tulsee Doshi, Martin Chadwick, Jilin Chen, Sanil Jain, Quoc Le, Arjun Kar, Madhu
Gurumurthy, Cheng Li, Ruoxin Sang, Fangyu Liu, Lampros Lamprou, Rich Munoz, Nathan Lintz,
Harsh Mehta, Heidi Howard, Malcolm Reynolds, Lora Aroyo, Quan Wang, Lorenzo Blanco, Albin
Cassirer, Jordan Griffith, Dipanjan Das, Stephan Lee, Jakub Sygnowski, Zach Fisher, James Besley,
13

Richard Powell, Zafarali Ahmed, Dominik Paulus, David Reitter, Zalan Borsos, Rishabh Joshi,
Aedan Pope, Steven Hand, Vittorio Selo, Vihan Jain, Nikhil Sethi, Megha Goel, Takaki Makino,
Rhys May, Zhen Yang, Johan Schalkwyk, Christina Butterfield, Anja Hauth, Alex Goldin, Will
Hawkins, Evan Senter, Sergey Brin, Oliver Woodman, Marvin Ritter, Eric Noland, Minh Giang,
Vijay Bolina, Lisa Lee, Tim Blyth, Ian Mackinnon, Machel Reid, Obaid Sarvana, David Silver,
Alexander Chen, Lily Wang, Loren Maggiore, Oscar Chang, Nithya Attaluri, Gregory Thornton,
Chung-Cheng Chiu, Oskar Bunyan, Nir Levine, Timothy Chung, Evgenii Eltyshev, Xiance Si,
Timothy Lillicrap, Demetra Brady, Vaibhav Aggarwal, Boxi Wu, Yuanzhong Xu, Ross McIlroy,
Kartikeya Badola, Paramjit Sandhu, Erica Moreira, Wojciech Stokowiec, Ross Hemsley, Dong
Li, Alex Tudor, Pranav Shyam, Elahe Rahimtoroghi, Salem Haykal, Pablo Sprechmann, Xiang
Zhou, Diana Mincu, Yujia Li, Ravi Addanki, Kalpesh Krishna, Xiao Wu, Alexandre Frechette,
Matan Eyal, Allan Dafoe, Dave Lacey, Jay Whang, Thi Avrahami, Ye Zhang, Emanuel Taropa,
Hanzhao Lin, Daniel Toyama, Eliza Rutherford, Motoki Sano, HyunJeong Choe, Alex Tomala,
Chalence Safranek-Shrader, Nora Kassner, Mantas Pajarskas, Matt Harvey, Sean Sechrist, Meire
Fortunato, Christina Lyu, Gamaleldin Elsayed, Chenkai Kuang, James Lottes, Eric Chu, Chao Jia,
Chih-Wei Chen, Peter Humphreys, Kate Baumli, Connie Tao, Rajkumar Samuel, Cicero Nogueira
dos Santos, Anders Andreassen, Nemanja Raki ´cevi´c, Dominik Grewe, Aviral Kumar, Stephanie
Winkler, Jonathan Caton, Andrew Brock, Sid Dalmia, Hannah Sheahan, Iain Barr, Yingjie Miao,
Paul Natsev, Jacob Devlin, Feryal Behbahani, Flavien Prost, Yanhua Sun, Artiom Myaskovsky,
Thanumalayan Sankaranarayana Pillai, Dan Hurt, Angeliki Lazaridou, Xi Xiong, Ce Zheng, Fabio
Pardo, Xiaowei Li, Dan Horgan, Joe Stanton, Moran Ambar, Fei Xia, Alejandro Lince, Mingqiu
Wang, Basil Mustafa, Albert Webson, Hyo Lee, Rohan Anil, Martin Wicke, Timothy Dozat,
Abhishek Sinha, Enrique Piqueras, Elahe Dabir, Shyam Upadhyay, Anudhyan Boral, Lisa Anne
Hendricks, Corey Fry, Josip Djolonga, Yi Su, Jake Walker, Jane Labanowski, Ronny Huang, Vedant
Misra, Jeremy Chen, RJ Skerry-Ryan, Avi Singh, Shruti Rijhwani, Dian Yu, Alex Castro-Ros,
Beer Changpinyo, Romina Datta, Sumit Bagri, Arnar Mar Hrafnkelsson, Marcello Maggioni,
Daniel Zheng, Yury Sulsky, Shaobo Hou, Tom Le Paine, Antoine Yang, Jason Riesa, Dominika
Rogozinska, Dror Marcus, Dalia El Badawy, Qiao Zhang, Luyu Wang, Helen Miller, Jeremy
Greer, Lars Lowe Sjos, Azade Nova, Heiga Zen, Rahma Chaabouni, Mihaela Rosca, Jiepu Jiang,
Charlie Chen, Ruibo Liu, Tara Sainath, Maxim Krikun, Alex Polozov, Jean-Baptiste Lespiau,
Josh Newlan, Zeyncep Cankara, Soo Kwak, Yunhan Xu, Phil Chen, Andy Coenen, Clemens
Meyer, Katerina Tsihlas, Ada Ma, Juraj Gottweis, Jinwei Xing, Chenjie Gu, Jin Miao, Christian
Frank, Zeynep Cankara, Sanjay Ganapathy, Ishita Dasgupta, Steph Hughes-Fitt, Heng Chen,
David Reid, Keran Rong, Hongmin Fan, Joost van Amersfoort, Vincent Zhuang, Aaron Cohen,
Shixiang Shane Gu, Anhad Mohananey, Anastasija Ilic, Taylor Tobin, John Wieting, Anna Bortsova,
Phoebe Thacker, Emma Wang, Emily Caveness, Justin Chiu, Eren Sezener, Alex Kaskasoli,
Steven Baker, Katie Millican, Mohamed Elhawaty, Kostas Aisopos, Carl Lebsack, Nathan Byrd,
Hanjun Dai, Wenhao Jia, Matthew Wiethoff, Elnaz Davoodi, Albert Weston, Lakshman Yagati,
Arun Ahuja, Isabel Gao, Golan Pundak, Susan Zhang, Michael Azzam, Khe Chai Sim, Sergi
Caelles, James Keeling, Abhanshu Sharma, Andy Swing, YaGuang Li, Chenxi Liu, Carrie Grimes
Bostock, Yamini Bansal, Zachary Nado, Ankesh Anand, Josh Lipschultz, Abhijit Karmarkar,
Lev Proleev, Abe Ittycheriah, Soheil Hassas Yeganeh, George Polovets, Aleksandra Faust, Jiao
Sun, Alban Rrustemi, Pen Li, Rakesh Shivanna, Jeremiah Liu, Chris Welty, Federico Lebron,
Anirudh Baddepudi, Sebastian Krause, Emilio Parisotto, Radu Soricut, Zheng Xu, Dawn Bloxwich,
Melvin Johnson, Behnam Neyshabur, Justin Mao-Jones, Renshen Wang, Vinay Ramasesh, Zaheer
Abbas, Arthur Guez, Constant Segal, Duc Dung Nguyen, James Svensson, Le Hou, Sarah York,
Kieran Milan, Sophie Bridgers, Wiktor Gworek, Marco Tagliasacchi, James Lee-Thorp, Michael
Chang, Alexey Guseynov, Ale Jakse Hartman, Michael Kwong, Ruizhe Zhao, Sheleem Kashem,
Elizabeth Cole, Antoine Miech, Richard Tanburn, Mary Phuong, Filip Pavetic, Sebastien Cevey,
Ramona Comanescu, Richard Ives, Sherry Yang, Cosmo Du, Bo Li, Zizhao Zhang, Mariko Iinuma,
Clara Huiyi Hu, Aurko Roy, Shaan Bijwadia, Zhenkai Zhu, Danilo Martins, Rachel Saputro, Anita
Gergely, Steven Zheng, Dawei Jia, Ioannis Antonoglou, Adam Sadovsky, Shane Gu, Yingying
Bi, Alek Andreev, Sina Samangooei, Mina Khan, Tomas Kocisky, Angelos Filos, Chintu Kumar,
Colton Bishop, Adams Yu, Sarah Hodkinson, Sid Mittal, Premal Shah, Alexandre Moufarek, Yong
Cheng, Adam Bloniarz, Jaehoon Lee, Pedram Pejman, Paul Michel, Stephen Spencer, Vladimir
Feinberg, Xuehan Xiong, Nikolay Savinov, Charlotte Smith, Siamak Shakeri, Dustin Tran, Mary
Chesus, Bernd Bohnet, George Tucker, Tamara von Glehn, Carrie Muir, Yiran Mao, Hideto Kazawa,
Ambrose Slone, Kedar Soparkar, Disha Shrivastava, James Cobon-Kerr, Michael Sharman, Jay
Pavagadhi, Carlos Araya, Karolis Misiunas, Nimesh Ghelani, Michael Laskin, David Barker,
14

Qiujia Li, Anton Briukhov, Neil Houlsby, Mia Glaese, Balaji Lakshminarayanan, Nathan Schucher,
Yunhao Tang, Eli Collins, Hyeontaek Lim, Fangxiaoyu Feng, Adria Recasens, Guangda Lai,
Alberto Magni, Nicola De Cao, Aditya Siddhant, Zoe Ashwood, Jordi Orbay, Mostafa Dehghani,
Jenny Brennan, Yifan He, Kelvin Xu, Yang Gao, Carl Saroufim, James Molloy, Xinyi Wu, Seb
Arnold, Solomon Chang, Julian Schrittwieser, Elena Buchatskaya, Soroush Radpour, Martin
Polacek, Skye Giordano, Ankur Bapna, Simon Tokumine, Vincent Hellendoorn, Thibault Sottiaux,
Sarah Cogan, Aliaksei Severyn, Mohammad Saleh, Shantanu Thakoor, Laurent Shefey, Siyuan
Qiao, Meenu Gaba, Shuo yiin Chang, Craig Swanson, Biao Zhang, Benjamin Lee, Paul Kishan
Rubenstein, Gan Song, Tom Kwiatkowski, Anna Koop, Ajay Kannan, David Kao, Parker Schuh,
Axel Stjerngren, Golnaz Ghiasi, Gena Gibson, Luke Vilnis, Ye Yuan, Felipe Tiengo Ferreira,
Aishwarya Kamath, Ted Klimenko, Ken Franko, Kefan Xiao, Indro Bhattacharya, Miteyan Patel,
Rui Wang, Alex Morris, Robin Strudel, Vivek Sharma, Peter Choy, Sayed Hadi Hashemi, Jessica
Landon, Mara Finkelstein, Priya Jhakra, Justin Frye, Megan Barnes, Matthew Mauger, Dennis
Daun, Khuslen Baatarsukh, Matthew Tung, Wael Farhan, Henryk Michalewski, Fabio Viola, Felix
de Chaumont Quitry, Charline Le Lan, Tom Hudson, Qingze Wang, Felix Fischer, Ivy Zheng,
Elspeth White, Anca Dragan, Jean baptiste Alayrac, Eric Ni, Alexander Pritzel, Adam Iwanicki,
Michael Isard, Anna Bulanova, Lukas Zilka, Ethan Dyer, Devendra Sachan, Srivatsan Srinivasan,
Hannah Muckenhirn, Honglong Cai, Amol Mandhane, Mukarram Tariq, Jack W. Rae, Gary Wang,
Kareem Ayoub, Nicholas FitzGerald, Yao Zhao, Woohyun Han, Chris Alberti, Dan Garrette,
Kashyap Krishnakumar, Mai Gimenez, Anselm Levskaya, Daniel Sohn, Josip Matak, Inaki Iturrate,
Michael B. Chang, Jackie Xiang, Yuan Cao, Nishant Ranka, Geoff Brown, Adrian Hutter, Vahab
Mirrokni, Nanxin Chen, Kaisheng Yao, Zoltan Egyed, Francois Galilee, Tyler Liechty, Praveen
Kallakuri, Evan Palmer, Sanjay Ghemawat, Jasmine Liu, David Tao, Chloe Thornton, Tim Green,
Mimi Jasarevic, Sharon Lin, Victor Cotruta, Yi-Xuan Tan, Noah Fiedel, Hongkun Yu, Ed Chi,
Alexander Neitz, Jens Heitkaemper, Anu Sinha, Denny Zhou, Yi Sun, Charbel Kaed, Brice Hulse,
Swaroop Mishra, Maria Georgaki, Sneha Kudugunta, Clement Farabet, Izhak Shafran, Daniel
Vlasic, Anton Tsitsulin, Rajagopal Ananthanarayanan, Alen Carin, Guolong Su, Pei Sun, Shashank
V , Gabriel Carvajal, Josef Broder, Iulia Comsa, Alena Repina, William Wong, Warren Weilun Chen,
Peter Hawkins, Egor Filonov, Lucia Loher, Christoph Hirnschall, Weiyi Wang, Jingchen Ye, Andrea
Burns, Hardie Cate, Diana Gage Wright, Federico Piccinini, Lei Zhang, Chu-Cheng Lin, Ionel
Gog, Yana Kulizhskaya, Ashwin Sreevatsa, Shuang Song, Luis C. Cobo, Anand Iyer, Chetan Tekur,
Guillermo Garrido, Zhuyun Xiao, Rupert Kemp, Huaixiu Steven Zheng, Hui Li, Ananth Agarwal,
Christel Ngani, Kati Goshvadi, Rebeca Santamaria-Fernandez, Wojciech Fica, Xinyun Chen,
Chris Gorgolewski, Sean Sun, Roopal Garg, Xinyu Ye, S. M. Ali Eslami, Nan Hua, Jon Simon,
Pratik Joshi, Yelin Kim, Ian Tenney, Sahitya Potluri, Lam Nguyen Thiet, Quan Yuan, Florian
Luisier, Alexandra Chronopoulou, Salvatore Scellato, Praveen Srinivasan, Minmin Chen, Vinod
Koverkathu, Valentin Dalibard, Yaming Xu, Brennan Saeta, Keith Anderson, Thibault Sellam,
Nick Fernando, Fantine Huot, Junehyuk Jung, Mani Varadarajan, Michael Quinn, Amit Raul,
Maigo Le, Ruslan Habalov, Jon Clark, Komal Jalan, Kalesha Bullard, Achintya Singhal, Thang
Luong, Boyu Wang, Sujeevan Rajayogam, Julian Eisenschlos, Johnson Jia, Daniel Finchelstein,
Alex Yakubovich, Daniel Balle, Michael Fink, Sameer Agarwal, Jing Li, Dj Dvijotham, Shalini
Pal, Kai Kang, Jaclyn Konzelmann, Jennifer Beattie, Olivier Dousse, Diane Wu, Remi Crocker,
Chen Elkind, Siddhartha Reddy Jonnalagadda, Jong Lee, Dan Holtmann-Rice, Krystal Kallarackal,
Rosanne Liu, Denis Vnukov, Neera Vats, Luca Invernizzi, Mohsen Jafari, Huanjie Zhou, Lilly
Taylor, Jennifer Prendki, Marcus Wu, Tom Eccles, Tianqi Liu, Kavya Kopparapu, Francoise
Beaufays, Christof Angermueller, Andreea Marzoca, Shourya Sarcar, Hilal Dib, Jeff Stanway,
Frank Perbet, Nejc Trdin, Rachel Sterneck, Andrey Khorlin, Dinghua Li, Xihui Wu, Sonam
Goenka, David Madras, Sasha Goldshtein, Willi Gierke, Tong Zhou, Yaxin Liu, Yannie Liang,
Anais White, Yunjie Li, Shreya Singh, Sanaz Bahargam, Mark Epstein, Sujoy Basu, Li Lao,
Adnan Ozturel, Carl Crous, Alex Zhai, Han Lu, Zora Tung, Neeraj Gaur, Alanna Walton, Lucas
Dixon, Ming Zhang, Amir Globerson, Grant Uy, Andrew Bolt, Olivia Wiles, Milad Nasr, Ilia
Shumailov, Marco Selvi, Francesco Piccinno, Ricardo Aguilar, Sara McCarthy, Misha Khalman,
Mrinal Shukla, Vlado Galic, John Carpenter, Kevin Villela, Haibin Zhang, Harry Richardson,
James Martens, Matko Bosnjak, Shreyas Rammohan Belle, Jeff Seibert, Mahmoud Alnahlawi,
Brian McWilliams, Sankalp Singh, Annie Louis, Wen Ding, Dan Popovici, Lenin Simicich, Laura
Knight, Pulkit Mehta, Nishesh Gupta, Chongyang Shi, Saaber Fatehi, Jovana Mitrovic, Alex Grills,
Joseph Pagadora, Tsendsuren Munkhdalai, Dessie Petrova, Danielle Eisenbud, Zhishuai Zhang,
Damion Yates, Bhavishya Mittal, Nilesh Tripuraneni, Yannis Assael, Thomas Brovelli, Prateek
Jain, Mihajlo Velimirovic, Canfer Akbulut, Jiaqi Mu, Wolfgang Macherey, Ravin Kumar, Jun Xu,
15

Haroon Qureshi, Gheorghe Comanici, Jeremy Wiesner, Zhitao Gong, Anton Ruddock, Matthias
Bauer, Nick Felt, Anirudh GP, Anurag Arnab, Dustin Zelle, Jonas Rothfuss, Bill Rosgen, Ashish
Shenoy, Bryan Seybold, Xinjian Li, Jayaram Mudigonda, Goker Erdogan, Jiawei Xia, Jiri Simsa,
Andrea Michi, Yi Yao, Christopher Yew, Steven Kan, Isaac Caswell, Carey Radebaugh, Andre
Elisseeff, Pedro Valenzuela, Kay McKinney, Kim Paterson, Albert Cui, Eri Latorre-Chimoto,
Solomon Kim, William Zeng, Ken Durden, Priya Ponnapalli, Tiberiu Sosea, Christopher A.
Choquette-Choo, James Manyika, Brona Robenek, Harsha Vashisht, Sebastien Pereira, Hoi Lam,
Marko Velic, Denese Owusu-Afriyie, Katherine Lee, Tolga Bolukbasi, Alicia Parrish, Shawn Lu,
Jane Park, Balaji Venkatraman, Alice Talbert, Lambert Rosique, Yuchung Cheng, Andrei Sozanschi,
Adam Paszke, Praveen Kumar, Jessica Austin, Lu Li, Khalid Salama, Bartek Perz, Wooyeol Kim,
Nandita Dukkipati, Anthony Baryshnikov, Christos Kaplanis, XiangHai Sheng, Yuri Chervonyi,
Caglar Unlu, Diego de Las Casas, Harry Askham, Kathryn Tunyasuvunakool, Felix Gimeno,
Siim Poder, Chester Kwak, Matt Miecnikowski, Vahab Mirrokni, Alek Dimitriev, Aaron Parisi,
Dangyi Liu, Tomy Tsai, Toby Shevlane, Christina Kouridi, Drew Garmon, Adrian Goedeckemeyer,
Adam R. Brown, Anitha Vijayakumar, Ali Elqursh, Sadegh Jazayeri, Jin Huang, Sara Mc Carthy,
Jay Hoover, Lucy Kim, Sandeep Kumar, Wei Chen, Courtney Biles, Garrett Bingham, Evan Rosen,
Lisa Wang, Qijun Tan, David Engel, Francesco Pongetti, Dario de Cesare, Dongseong Hwang, Lily
Yu, Jennifer Pullman, Srini Narayanan, Kyle Levin, Siddharth Gopal, Megan Li, Asaf Aharoni,
Trieu Trinh, Jessica Lo, Norman Casagrande, Roopali Vij, Loic Matthey, Bramandia Ramadhana,
Austin Matthews, CJ Carey, Matthew Johnson, Kremena Goranova, Rohin Shah, Shereen Ashraf,
Kingshuk Dasgupta, Rasmus Larsen, Yicheng Wang, Manish Reddy Vuyyuru, Chong Jiang, Joana
Ijazi, Kazuki Osawa, Celine Smith, Ramya Sree Boppana, Taylan Bilal, Yuma Koizumi, Ying
Xu, Yasemin Altun, Nir Shabat, Ben Bariach, Alex Korchemniy, Kiam Choo, Olaf Ronneberger,
Chimezie Iwuanyanwu, Shubin Zhao, David Soergel, Cho-Jui Hsieh, Irene Cai, Shariq Iqbal,
Martin Sundermeyer, Zhe Chen, Elie Bursztein, Chaitanya Malaviya, Fadi Biadsy, Prakash Shroff,
Inderjit Dhillon, Tejasi Latkar, Chris Dyer, Hannah Forbes, Massimo Nicosia, Vitaly Nikolaev,
Somer Greene, Marin Georgiev, Pidong Wang, Nina Martin, Hanie Sedghi, John Zhang, Praseem
Banzal, Doug Fritz, Vikram Rao, Xuezhi Wang, Jiageng Zhang, Viorica Patraucean, Dayou Du,
Igor Mordatch, Ivan Jurin, Lewis Liu, Ayush Dubey, Abhi Mohan, Janek Nowakowski, Vlad-Doru
Ion, Nan Wei, Reiko Tojo, Maria Abi Raad, Drew A. Hudson, Vaishakh Keshava, Shubham
Agrawal, Kevin Ramirez, Zhichun Wu, Hoang Nguyen, Ji Liu, Madhavi Sewak, Bryce Petrini,
DongHyun Choi, Ivan Philips, Ziyue Wang, Ioana Bica, Ankush Garg, Jarek Wilkiewicz, Priyanka
Agrawal, Xiaowei Li, Danhao Guo, Emily Xue, Naseer Shaik, Andrew Leach, Sadh MNM Khan,
Julia Wiesinger, Sammy Jerome, Abhishek Chakladar, Alek Wenjiao Wang, Tina Ornduff, Folake
Abu, Alireza Ghaffarkhah, Marcus Wainwright, Mario Cortes, Frederick Liu, Joshua Maynez,
Andreas Terzis, Pouya Samangouei, Riham Mansour, Tomasz K˛ epa, François-Xavier Aubet, Anton
Algymr, Dan Banica, Agoston Weisz, Andras Orban, Alexandre Senges, Ewa Andrejczuk, Mark
Geller, Niccolo Dal Santo, Valentin Anklin, Majd Al Merey, Martin Baeuml, Trevor Strohman,
Junwen Bai, Slav Petrov, Yonghui Wu, Demis Hassabis, Koray Kavukcuoglu, Jeff Dean, and Oriol
Vinyals. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,
2024. URLhttps://arxiv.org/abs/2403.05530.
Benton J. Underwood. Interference and forgetting.Psychological Review, 64(1):49–60, 1957. doi:
10.1037/h0044616.
Luanbo Wan and Weizhi Ma. Storybench: A dynamic benchmark for evaluating long-term memory
with multi turns, 2025. URLhttps://arxiv.org/abs/2506.13356.
Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian
Wu. Mem-alpha: Learning memory construction via reinforcement learning, 2025. URL https:
//arxiv.org/abs/2509.25911.
Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart Van Merriënboer, Armand
Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy
tasks.arXiv preprint arXiv:1502.05698, 2015.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval:
Benchmarking chat assistants on long-term interactive memory, 2025. URL https://arxiv.
org/abs/2410.10813.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic
memory for llm agents, 2025. URLhttps://arxiv.org/abs/2502.12110.
16

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma,
Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schütze, V olker Tresp, and Yunpu Ma. Memory-r1:
Enhancing large language model agents to manage and utilize memories via reinforcement learning,
2026. URLhttps://arxiv.org/abs/2508.19828.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang,
Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report.arXiv preprint arXiv:2412.15115,
2024.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu,
Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang,
Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang
Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger
Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan
Qiu. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-
Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context llm
with multi-conv rl-based memory agent, 2025. URLhttps://arxiv.org/abs/2507.02259.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embed-
ding: Advancing text embedding and reranking through foundation models, 2025. URL
https://arxiv.org/abs/2506.05176.
Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, and
Qianli Ma. Lifelongagentbench: Evaluating llm agents as lifelong learners.arXiv preprint
arXiv:2505.11942, 2025.
Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li, Duzhen Zhang, Chenxing Li, Dong Yu, and
Qianli Ma. Lifelong learning of large language model based agents: A roadmap, 2026. URL
https://arxiv.org/abs/2501.07278.
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan
Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for
efficient long-horizon agents, 2025. URLhttps://arxiv.org/abs/2506.15841.
17

A Additional Benchmark Details
A.1 Four Domains in MINTEVAL
(1)BABI (State Tracking). We build on bAbI [Weston et al., 2015], adopting its fact-based, state-
tracking format with simple, compositional sentences, where each input unit corresponds to an
individual fact describing an entity state. The information is structured as discrete, symbolic facts,
and updates occur through sequential, localized modifications that often explicitly overwrite previous
states. This domain, therefore, requires systems to integrate sequential updates, track precise state
transitions, and perform temporal reasoning to accurately recover both current and historical states.
(2) HORIZONBENCH(Dialogue-based Multi-turn Interactions). Based on HorizonBench [Li et al.,
2026], a long-horizon personalization benchmark with simulated users and 6-month conversation
histories, we construct multi-turn dialogue contexts, where each input unit is a dialogue session
composed of multiple conversational turns. Information is distributed across natural language
utterances and is often implicitly expressed through user interactions. Updates are incremental, noisy,
and indirect, requiring models to interpret evolving user intent and preferences over time. This
domain evaluates whether memory systems can maintain and update such implicit changes over time
through the conversation and answer questions about the resulting user state.
(3) WIKIREVISIONS(Factual Knowledge QA). We construct contexts from Wikipedia revision
histories. Each input instance consists of a single article paired with its full chronological sequence
of revisions, where each revision is a complete document snapshot augmented with metadata, e.g.,
timestamp, editor identity, and edit comment. This setting differs from single-snapshot or synthetic
memory benchmarks in that it exhibits substantial temporal heterogeneity. Facts may be added,
refined, contradicted, or removed over time; sections may be reordered; and a given attribute typically
assumes a sequence of values rather than a single fixed value. Consequently, answering a query
requires reconstructing a prior state of the article, identifying which editor introduced a claim,
counting the number of value changes, or distinguishing outdated information from currently valid
content. A memory system, therefore, must preserve revision-level provenance, track the evolution of
attributes across revisions, and differentiate superseded information from information that remains
valid.
(4) GITCOMMITS(Code and Files Evolution). We construct contexts from GitHub commit histories
in an analogous manner. Each input instance consists of a single repository paired with its full
chronological sequence of commits, where each commit is a complete snapshot of the codebase
augmented with metadata, e.g., author, timestamp, commit message, and the set of modified files. The
requirements introduced in the Wikipedia setting, i.e., preserving provenance, tracking the evolution
of attributes, and distinguishing outdated from currently valid information, transfer directly to this
domain. A key distinction is that each snapshot comprises structured, executable code rather than
prose, and therefore specifies not only a textual state but also concrete program behavior. This gives
rise to phenomena that are largely absent in natural-language histories. First, edits are often cross-file
and tightly coupled, e.g., a single commit may rename a function and update all corresponding call
sites. Second, the same identifier, e.g., a function name, API signature, or configuration key, may
assume a sequence of distinct semantics over time. As a result, a memory system operating in this
setting must additionally recover the implicit differences between successive snapshots and reason
about how program behavior evolves across commits.
A.2 Question Examples for Each Domain
In Table 5, we provide the question examples for each domain and question category.
A.3 Question Generation
For both bAbI and HorizonBench, we generate questions using the provided metadata or parsed facts
using predefined question templates. In the bAbI setting, a subset of the Simple ,Counting , and
Multihop questions is adopted from OAKS-BABI [Kim et al., 2026]. We additionally construct new
questions to align with our task definitions. For the remaining questions, similar to OAKS-BABI
construction, we parse each fact into a structured triplet of (subject, object, verb) and instantiate the
18

Table 5: Example questions for each domain and category.
Domain Category Example Question
WikiSimpleIn what year did the article first mention that Toni Basil provided
choreography for the tour?
HistoryHow does the version of the article 42 edits before the latest version
format the team name for the No. 1 car in the race classification table?
CountingHow many distinct numerical peak chart positions has the article ever
listed for the single ’Uh Huh’, up to the current version?
OrderingAmong all the phrases the article has used to describe what Conan’s
decapitation of Thulsa Doom revealed him to be, what was the longest
span (in days) that any single phrase was listed?
MultihopAt the revision immediately before the article updated the tree’s maxi-
mum height from 20 metres to 50 m, what taxonomic division was
the tree classified under?
GitHubSimpleWhich contributor first introduced the
tests/admin/test_api_revoking_admin_role.py file into the
project?
HistoryWhat value does the version of the project 70 commits before the latest
version set forinferencein the[yolo2]section ofconfig.ini?
CountingHow many distinct version strings has the project ever set in
setup.py, up to the current version?
OrderingFor how many days was MAX_DRAFT_SENTENCES set to 5in
manager.pybefore it was increased?
MultihopJust before the default arm_velocity_limit parameter was modi-
fied for the second time in the arm controller, what network interface
was set as the default for the loco client?
HorizonBenchSimpleWhat is the user’s current value for ‘preferred source types’ in their
“Empirical Evidence Integration Style” preference?
HistoryWhat was the user’s value for ‘preferred response format’ in their
“Collaborative Alternate-History Storytelling Interaction Style” prefer-
ence 4 preference-change events ago?
CountingHow many times has the user changed their value for ‘encouragement
tone preference’ in their “Self-Esteem Rebuilding Communication
Style” preference?
OrderingList in chronological order all distinct values the user has held for
‘language register’ in their “Self-Esteem Rebuilding Interaction Style”
preference, from earliest to most recent. Output as a comma-separated
list.
MultihopFor the user’s ‘openness to nonwestern’ in their “Philosophical
Tradition Affinity” preference, which value have they held longer:
very_highorhigh?
bAbISimpleWho last dropped football?
HistoryWhere was Daniel for the sixteenth most recent time?
CountingHow many total times has milk been picked up?
OrderingList in chronological order all distinct people who have dropped the
milk, from earliest to most recent. Output as a comma-separated list.
MultihopWho most recently traveled directly from office to kitchen?
question template using the parsed information. For HorizonBench, we use the provided metadata10,
which contains information such as user preferences. Similar to the bAbI setup, we design a template
for each question type and populate it with the corresponding metadata fields. Since the exact answer
words may not explicitly appear in the context and are only available in the metadata, we provide
candidate options together for those questions.
For Wiki-Revisions and Git Commits, we use the official APIs to collect revision histories of articles
and repositories, respectively. They are obtained from the MediaWiki11and GitHub12APIs. For
10https://huggingface.co/datasets/stellalisy/HorizonBench
11https://en.wikipedia.org/w/api.php
12https://api.github.com
19

Table 6: Dataset statistics across four domains.Depthdenotes the number of turns, revisions, or
commits in each example.kindicates values reported in thousands.
bAbI HorizonBench Wiki Revision Git Commit
Domain State Tracking Dialogue Wikipedia Code
# Sessions 99 100 196 200
Context
StatisticsDepth (avg) 42 142 99 61
Depth (max) 148 183 100 100
Tokens (avg) 0.3k 274k 195k 86k
Tokens (max) 0.9k 496k 1768k 600k
Question
DistributionsSimple 720 998 319 283
History 1936 2909 524 602
Ordering 1000 1000 247 305
Counting 1000 998 63 128
Multihop 1000 1000 339 260
# Total 5656 6905 1492 1578
Wikipedia, we restrict candidate articles to those in the Featured Articles and Good Articles categories,
i.e., Wikipedia’s community-curated and peer-reviewed quality tiers. In addition, we require each
article’s current size and prose density to fall within a predefined range, excluding stubs, list pages,
and pages dominated by templates or infoboxes. For GitHub, we limit our selection to non-forked
and non-archived Python repositories with at least 100 stars to ensure quality. In both domains,
we keep samples that contain a sufficient number of substantive revisions up to 100. This ensures
that each sample provides adequate temporal depth for probing memory evolution. We also remove
non-substantive edits, e.g., bot-generated changes, markup-only updates, or empty revisions, so that
each retained revision reflects a meaningful modification. We then generate questions using Gemini-
3.1-Pro [Google, 2026b] with descriptions and examples of each question type and the complete
revision history under a structured output schema to generate questions. Specifically, for Wiki
Revision, the article’s earliest version, followed by every subsequent revision with revision metadata
(revision_ids ,timestamp ,editor ,edit_comment ) are provided to Gemini-3.1-Pro. Each
generated question is paired with the revision_ids that serve as supporting evidence. Similarly, for
Git Commit, the repository’s oldest captured commit, followed by every subsequent commit as that
commit’s combined multi-file unified diff against its parent, each augmented with commit metadata
(timestamp,username,commit_message) are given to Gemini-3.1-Pro.
The templates and prompts used in this work are included the official GitHub ( https://github.
com/amy-hyunji/MINTEval) due to their length.
A.4 Human Validation on the Generated Data
We further conduct a human validation on 405 stratified samples drawn from both the Wiki Revisions
and Git Commits subsets, covering five question types. We find that 95.6% of the samples are valid,
meaning that both the question and answer are correctly annotated. Only a small proportion of
cases are invalid, including 1.0% when both question and answer are invalid, 1.7% where only the
question is invalid, and 1.7% where the answer is only invalid. Breaking down the results by dataset,
Git Commits exhibits a 98.0% validity rate, whereas Wiki Revisions shows a slightly lower but
still strong validity rate of 93.2%. Across question categories, Simple questions show the highest
validity of 97.6%, History questions show 93.9%, Counting show 93.8%, Ordering show 97.5%,
andMultihop show 95.0%. Counting and ordering tasks are fully valid, while Simple ,History ,
andMultihop questions show moderately lower validity (86.7%, 80.0%, and 81.8%, respectively),
suggesting that more complex queries are more prone to annotation issues. Overall, these results
indicate that the dataset is generally reliable, with errors concentrated in more complex question
types.
20

Simple History Ordering Counting Multihop020406080AccuracyWiki Revision
Simple History Ordering Counting MultihopGit CommitMemAgent-14B Qwen3.6-35B Gemini3.1-LiteFigure 6: MemAgent performance on Wiki Revisions and Git Commits across different answering
agents. Specialized answering agents such as MemAgent-14B remain competitive on single-target
recall, but drop on multi-target aggregation questions (especially Counting ), which require stronger
aggregation and reasoning capabilities.
A.5 Dataset Statistics
We provide more detailed statistics in Table 6. Across domains, the contexts vary substantially in both
depth and total token length, ranging from short synthetic trajectories to highly long-form histories
exceeding one million tokens. The benchmark also contains a balanced distribution of question types,
including simple recall, historical lookup, ordering, counting, and multihop reasoning, enabling
systematic evaluation of memory retrieval, temporal reasoning, and aggregation capabilities under
interference-heavy contexts.
B More Experimental Details
For all experiments, we set the decoding temperature to 0. Models are instructed to present the final
answer wrapped in \boxed{} . We conduct experiments on a server either with 4x 80GB A100 or 4x
48GB A6000.
C Additional Analysis
C.1 Impact of Answering Agent Choice
Figure 6 shows the performance of MemAgent [Yu et al., 2025] paired with different answering
agents, including the originally trained MemAgent-14B, Qwen3.6-35B-A3B, and Gemini-3.1-Flash-
Lite. We observe that when experimenting with MemAgent-14B, a smaller but specialized answering
agent, the overall performance remains competitive onsingle-target recall, but drops onmulti-target
aggregationquestions, especially on Counting questions, which require stronger aggregation and
reasoning capabilities.
C.2 Using Frontier Models with the Full Context Remains Competitive
In Figure 7, we compare the performance of different methods when using Qwen3.6-35B-A3B
and Gemini-3.1-Flash-Lite as answering agent. Using Gemini-3.1-Flash-Lite with the Full Context
shows the highest performance on bothsingle-target recallandmulti-target aggregationtasks. The
improvement is particularly pronounced forsingle-target recall, where Gemini-3.1-Flash-Lite with
Full Context achieves over 80% accuracy, far surpassing other retrieval-based and memory-augmented
systems. These results suggest that frontier models like Gemini-3.1-Flash-Lite not only support longer
context length, but can also effectively reason over long and interference-heavy contexts. However,
once retrieval or memory modules are introduced, the performance gap between Qwen3.6-35B-A3B
and Gemini-3.1-Flash-Lite becomes relatively small. This indicates that, in memory-augmented
settings, the quality of the context, i.e., retrieved content or memory, is important.
21

Full
BaseRAG HippoRAG AtomMem Mem-alpha MemAgent020406080AccuracySingle-T arget Recall
Full
BaseRAG HippoRAG AtomMem Mem-alpha MemAgentMulti-T arget AggregationQwen3.6-35B Gemini3.1-LiteFigure 7: Comparison of performance across different answering agents (Qwen3.6-35B-A3B and
Gemini-3.1-Flash-Lite). The performance gap is largest under the Full Context setting. Overall, the
gap is larger on Multi-Target Aggregation tasks than on Single-Target Recall task.
2.5 5.0 7.5 10.0 12.5 15.0 17.5
n_steps_back010203040Accuracy (%)
RAG — History
RAG — + Date/TimeFull — History
Full — + Date/Time
Figure 8: Performance on History questions in bAbI as a function of lookback distance (x-axis),
comparing RAG and Full Context methods with and without temporal cues (Historyvs.+Date/Time).
Adding timestamps as explicit markers helps recover the gap caused by interference.
C.3 Effect of Adding Temporal Cues toHistoryQuestions
To investigate whether the performance degradation with increasing lookback distance in Figure 3
is caused by interference among similar facts, we conduct an additional experiment in which we
add explicit cues (date and time information) to both the facts and the questions. These cues help
distinguish otherwise similar facts and make them more discrete. We perform this experiment on
bAbI, where such cues can be easily incorporated into the data generation process. Figure 8 compares
performance with and without datetime information under the same inputs and questions. We observe
that adding these cues substantially mitigates the performance degradation as the lookback distance
increases for both Full Context and RAG systems. In contrast, without the cues, performance drops
sharply as the distance increases.
C.4 Biased Toward Insertion in Memory Systems
In Figure 9, we analyze the distribution of three operations: (1) inserting new information, (2)
modifying or updating existing entries, and (3) deleting outdated information. Comparing the two
systems, Mem- αdemonstrates a substantially higher rate of modification operations (34.1%) than
AtomMem (3.7%), indicating a better ability to update existing memory instead of duplicating it,
suggesting why Mem- αshows stronger overall performance. However, Mem- αtends to underutilize
22

Wiki Git Horizon BABI020406080T ool Usage Rate (%)AtomMem
Wiki Git Horizon BABIMemAlphaAdd/Insert Modify/Update DeleteFigure 9: Rate of tool usage for AtomMem and Mem- α. Mem- αconsistently underutilizes the
delete operation across all datasets, which may partially explain why memory systems struggle in
long-horizon settings with heavy interference: outdated or conflicting information accumulates over
time, leading to progressively greater conflict within memory.
0 1 3 5
# Distractors304050607080AccuracySimple (OOD)
Simple (ID)
Counting (OOD)
Counting (ID)
History (OOD)
History (ID)
Figure 10: Performance under varying distrac-
tor types and numbers of distractors.IDdis-
tractors more strongly affect questions such as
Counting andHistory compared to simpler
queries like Simple , suggesting that tasks requir-
ing aggregation or tracking over multiple facts
are more susceptible to interference.
1510 20 50 75
T op-K222426283032Performance
Qwen3-embedding-4B
Gemini-Embedding-001Figure 11: Average performance across all
datasets for different retrieval models (Qwen3-
Embedding-4B and Gemini-Embedding-001) as
the number of retrieved documents varies, using
Qwen3.6-35B-A3B as the answering agent.
the delete operation across all datasets, which could partially explain why memory systems fail under
long-horizon settings with heavy interference, as outdated or conflicting information accumulates
over time and increases conflicting information.
C.5 Effect of Retrieval Choices on RAG Performance
We analyzed how retrieval design choices—specifically the embedding model and the number of
retrieved documents ( K)—affect downstream question-answering performance in a RAG setting.
Experiments are conducted over average of all four datasets using RAG, while keeping the answering
model fixed as Qwen3.6-35B-A3B. We compare two embedding models: Qwen3-Embedding-
4B [Zhang et al., 2025] and Gemini-Embedding-001 [Google, 2025].
As shown in Figure 11, average performance increases sharply from K= 1 andK= 5 , after
which gains largely plateau. Qwen3-Embedding-4B achieves its best performance at k= 5 , while
Gemini-Embedding-001 peaks around K= 50 , though performance remains relatively similar for
larger Kvalues overall. When comparing retrieval models, Gemini-Embedding-001 consistently
outperforms Qwen3-Embedding-4B over all Kvalues, with the performance gap widening slightly
23

151020 50 75
T op_K1020304050Accuracy
simple
151020 50 75
T op_K
history
151020 50 75
T op_K
counting
151020 50 75
T op_K
multi-hop
151020 50 75
T op_K
ordering
Qwen3-Embedding-4B
Gemini-Embedding-001Figure 12: RAG performance across question types with varying numbers of retrieval documents
and embedding models (Qwen3-Embedding-4B and Gemini-Embedding-001).
Table 7: Results on SimpleMem using Gemini-3.1-Flash-Lite and Gemini-Embedding-001 across
datasets and question types, reported in Exact Match (%). Even the SOTA memory system, combined
with frontier models, struggles on MINTEVAL.
Dataset Simple History Counting Ordering Multi-hop Overall
bAbI 93.2 48.9 74.8 73.8 52.3 67.7
HorizonBench 6.3 5.7 10.8 3.1 23.5 8.8
Wiki Revisions 7.2 20.4 31.8 0.0 11.8 12.7
Git Commits 83.0 13.1 25.0 30.5 26.5 32.2
asKincreases. This indicates that stronger embeddings are more effective at ranking relevant
documents higher when the retrieval pool is larger.
A finer-grained analysis by question type (Figure 12) on Wiki Revision dataset reveals that most of
the performance gap between embedding models arises from more complexmulti-target aggregation
questions, especially Counting andOrdering questions. We hypothesize that this is because these
question types typically require aggregating or comparing information across multiple pieces of
evidence. Increasing Kleads to a higher probability that all necessary evidence is retrieved, which
disproportionately benefits these reasoning-heavy categories. In contrast, simplersingle-target recall
type questions (i.e., Simple orHistory ) show smaller sensitivity to both embedding choice and
retrieval depth, as they often depend on retrieving a single highly relevant document.
C.6 Expanded Discussion on the State-of-the-art Memory System Failure
SimpleMem is a state-of-the-art memory architecture built around a three-stage pipeline: (1)Semantic
Structured Compression, which distills unstructured interactions into compact multi-view memory
units; (2)Online Semantic Synthesis, which incrementally merges related contexts into unified
abstractions to reduce redundancy; and (3)Intent-Aware Retrieval Planning, which dynamically
infers retrieval scope and constructs targeted retrieval contexts. Using frontier models, Gemini-
3.1-Flash-Lite and Gemini-Embedding-001, we successfully reproduced the reported results on
LoCoMo [Maharana et al., 2024], achieving a state-of-the-art F1 score of 54.76%. As shown in
Table 7, performance degrades dramatically on MINTEVAL. The failure arises from a fundamental
mismatch between the assumptions underlying conversational memory benchmarks and the charac-
teristics of revision-centric data. In LoCoMo, each turn contains roughly 109 characters, yielding
approximately 4.4k characters in a memory chunk. Compressing this context into 5–10 structured
memory entries is therefore feasible with limited information loss. In contrast, our benchmark
contains revisions with a median length of 4.6k characters. A memory chunk consequently expands
to approximately 184k characters. Compressing such a window into the same 5–10 memory entries
discards the majority of the source content. Moreover, the compression objective itself is actively
harmful in this setting. SimpleMem explicitly encourages the model to avoid duplication during
memory construction. This assumption is appropriate for dialogue, where repeated statements are
often redundant, but it is detrimental for revision histories. In our dataset, consecutive revisions
exhibit substantial lexical overlap, while the critical information often lies in small localized edits. We
observe that the performance drops much more on HorizonBench, Wiki Revisions and Git Commits,
as revision provenance is not retained through the compression pipeline (i.e., has been paraphrased or
rewritten). As no explicit metadata records which revision produced a given fact, retrieval operates
solely over keywords and embeddings, making queries such as retrieving the contents of “Revision 53”
24

more challenging. We also experimented with Qwen3.6-35B-A3B and Qwen3-4B retrieval model,
but observed near-zero performance across all datasets; therefore, we do not report the results.
D Dataset License
The datasets used in this work are released under permissive licenses that support open research
and reproducibility. Specifically, HorizonBench [Li et al., 2026] is distributed under the Apache-2.0
license, which allows both academic and commercial use with minimal restrictions. The bAbI
dataset [Weston et al., 2015] is released under the Creative Commons Attribution 3.0 (CC BY 3.0)
license, which permits reuse and modification provided appropriate credit is given to the original
authors. These licenses ensure that all datasets used in this study are compliant with open-access and
reproducible research standards.
25