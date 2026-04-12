# HyperMem: Hypergraph Memory for Long-Term Conversations

**Authors**: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, Yafeng Deng

**Published**: 2026-04-09 13:43:23

**PDF URL**: [https://arxiv.org/pdf/2604.08256v1](https://arxiv.org/pdf/2604.08256v1)

## Abstract
Long-term memory is essential for conversational agents to maintain coherence, track persistent tasks, and provide personalized interactions across extended dialogues. However, existing approaches as Retrieval-Augmented Generation (RAG) and graph-based memory mostly rely on pairwise relations, which can hardly capture high-order associations, i.e., joint dependencies among multiple elements, causing fragmented retrieval. To this end, we propose HyperMem, a hypergraph-based hierarchical memory architecture that explicitly models such associations using hyperedges. Particularly, HyperMem structures memory into three levels: topics, episodes, and facts, and groups related episodes and their facts via hyperedges, unifying scattered content into coherent units. Leveraging this structure, we design a hybrid lexical-semantic index and a coarse-to-fine retrieval strategy, supporting accurate and efficient retrieval of high-order associations. Experiments on the LoCoMo benchmark show that HyperMem achieves state-of-the-art performance with 92.73% LLM-as-a-judge accuracy, demonstrating the effectiveness of HyperMem for long-term conversations.

## Full Text


<!-- PDF content starts -->

HyperMem: Hypergraph Memory for Long-Term Conversations
Juwei Yue*1,2,3, Chuanrui Hu∗3, Jiawei Sheng1,2, Zuyi Zhou3, Wenyuan Zhang1,2,
Tingwen Liu1,2,Li Guo1,2,Yafeng Deng3
1Institute of Information Engineering, Chinese Academy of Sciences
2School of Cyber Security, University of Chinese Academy of Sciences
3EverMind AI
Correspondence:shengjiawei@iie.ac.cn, dengyafeng@shanda.com
Abstract
Long-term memory is essential for conversa-
tional agents to maintain coherence, track per-
sistent tasks, and provide personalized interac-
tions across extended dialogues. However, ex-
isting approaches as Retrieval-Augmented Gen-
eration (RAG) and graph-based memory mostly
rely on pairwise relations, which can hardly
capture high-order associations, i.e., joint de-
pendencies among multiple elements, causing
fragmented retrieval. To this end, we pro-
poseHyperMem, a hypergraph-based hierar-
chical memory architecture that explicitly mod-
els such associations using hyperedges. Particu-
larly, HyperMem structures memory into three
levels:topics,episodes, andfacts, and groups
related episodes and their facts via hyperedges,
unifying scattered content into coherent units.
Leveraging this structure, we design a hybrid
lexical-semantic index and a coarse-to-fine re-
trieval strategy, supporting accurate and effi-
cient retrieval of high-order associations. Ex-
periments on the LoCoMo benchmark show
that HyperMem achieves state-of-the-art perfor-
mance with 92.73% LLM-as-a-judge accuracy,
demonstrating the effectiveness of HyperMem
for long-term conversations.1
1 Introduction
Conversational agents (Zhang et al., 2025e) increas-
ingly serve as long-term companions, requiring
coherent multi-hop reasoning, persistent task track-
ing, and personalized interactions across extended
dialogues. However, their fixed context windows
render historical experiences inaccessible as con-
versations grow, necessitating effective and effi-
cient long-term memory management (Chhikara
et al., 2025; Li et al., 2025b; Zhang et al., 2026).
Existing approaches such as Retrieval-
Augmented Generation (RAG) (Gao et al., 2023;
Fan et al., 2024) and graph-based memory (Zhang
*Equal Contribution.
1Our source code is about to be released.
I signed up for the marathon next month.Great! Take it slow at first.03/05/2025
How's your project going?Deadline'ssoon. Working overtime every day.03/18/2025
How's your marathon training?Good! I can run 15km now.05/02/2025
Project launched! Let's celebrate with a run.Sure! Running really helps relieve stress.05/15/2025(b) Chunk-based RAG[Chunk 1]Alice: I signed… Bob: Great! Take…[Chunk 2]Alice: How's… Bob: Deadline's…[Chunk 3]Bob: How's… Alice: Good! I…[Chunk 4]Bob: Project… Alice: Sure! Running…(c) Graph-based RAG[Entity 1]Alice; [Entity 2]Bob…[Entity 4]Poject; [Entity 5]Running…[Relation 1]signed_up; [Relation 2]works_on…[Triple 1](Alice, signed_up, Marthon);[Triple 2](Bob, works_on, Project)…Graph:(d) HyperMem(Ours)[Topic Node 1]Alice's marathon training; [Topic Node 2]Bob's work and project[Episode Node 1]Alice signs up for marathon; [Episode Node 2]Bob faces deadline…[Episode Hyperedge 1] {𝑬!:0.5,𝑬":0.8,}…[Fact 1]Alice signed up for marathon;[Fact 2]Bob's deadline is approaching…[Fact Hyperedge 1] {𝑭!:0.6,𝑭":0.7}…Hypergraph:
(a) Conversation
Retrieval: Sim(Query, Topics) → Top-k Topics → Sim(Query, Episodes) → Top-k Episodes→ Sim(Query, Facts) → Top-k Facts Retrieval: Sim(Query Triple, Graph Triple)  → Top-k Triples
Retrieval: Sim(Query, Chunks) → Top-k Chunks
C1
C2
C3
C4XXXX①Topic:sport
③Topic:sport②Topic:work
④Topic:sport,workFigure 1: Memory structure comparison across Chunk-
based RAG, Graph-based RAG, and our HyperMem.
et al., 2025a; Rasmussen et al., 2025) retrieve
external stored related information to enrich the
context in response to user queries. However,
both paradigms fundamentally rely on pairwise
relationships, which inherently fail to capture
high-order associations, i.e., joint dependencies
among three or more related content elements. As
shown in Figure 1(a), a conversation may cover
multiple topics such assportandwork. Episodes
1, 3, and 4 are jointly associated under the
sport topic and involve multiple facts scattered
throughout the dialogue. Conventional methods,
as shown in Figure 1(b) and (c), can hardly model
the holistic coherence among episodes and facts,
leading to fragmented retrieval.
To explicitly capture the above high-order asso-
ciations, we model long-term memory as a hyper-
graph (Figure 1(d)). Unlike conventional graphs
with pairwise edges, hypergraphs supporthyper-
edgesthat connect arbitrary node sets, making them
uniquely capable of modeling joint dependenciesarXiv:2604.08256v1  [cs.CL]  9 Apr 2026

in dialogue. Our architecture, namelyHyperMem,
organizes a three-level memory hierarchy: (i)Topic
nodes, representing key conversation themes; (ii)
Episodenodes, denoting temporally contiguous di-
alogue segments centered on a single topic; and
(iii)Factnodes, encoding fine-grained details ex-
tracted from episodes. Thereafter, we use hyper-
edges to explicitly group all episodes sharing the
same topic, as well as all facts belonging to the
same episode. These hyperedges may naturally
overlap across episodes and facts, reflecting the
multifaceted nature of conversational content while
preserving semantic coherence within each group.
As a result, semantically scattered information is
unified into coherent units, enabling complete and
efficient retrieval of high-order associations.
To construct HyperMem, we first detect episode
boundaries from the dialogue stream, then aggre-
gate topically related episodes into shared topics
using hyperedges, and finally extract fine-grained
facts from each episode content. For indexing, we
leverage lexical cues and exploit dense semantics
with hypergraph embedding propagation. This en-
ables semantically related memories, even if tempo-
rally distant, to derive aligned embeddings, thereby
facilitating the retrieval of high-order associations.
At retrieval time, HyperMem performs a coarse-to-
fine search: it first identifies relevant topics, then
expands to their constituent episodes, and finally se-
lects the most pertinent facts to construct a focused
context for response generation. Our contributions
are summarized as follows:
•We propose HyperMem, a pioneering three-level
hypergraph memory architecture that explicitly
models high-order associations via hyperedges,
overcoming the limitations of pairwise relation
methods to capture holistic coherence.
•We leverage the HyperMem structure to derive
accurate lexical and semantical indexing, and
design a coarse-to-fine retrieval strategy to enable
efficient early pruning of irrelevant context.
•Experiments on the LoCoMo benchmark achieve
state-of-the-art performance with 92.73% LLM-
as-a-judge accuracy, demonstrating the effective-
ness of HyperMem for long-term conversations.
2 Related works
2.1 Retrieval-Augmented Generation
RAG has proven effective in mitigating hallucina-
tions (Ayala and Béchard, 2024) and improvingreliability (Xia et al., 2025; Asai et al., 2024), and
also serve as a foundation for long-term memory
in LLM-powered agents (Gutierrez et al., 2024;
Gutiérrez et al., 2025; Lin et al., 2025).
Vanilla methods retrieve relevant fragments from
external sources and use them as context for more
grounded responses (Lewis et al., 2020; Kulka-
rni et al., 2024). To enrich relational structures,
GraphRAG (Edge et al., 2024) pioneered knowl-
edge graph construction, inspiring works (He et al.,
2024; Hu et al., 2025b; Luo et al., 2024; Dong
et al., 2024; Chen et al., 2025; Guo et al., 2025; Fan
et al., 2025; Li et al., 2025a) that leverage graph
topology for structure-aware reasoning and multi-
hop retrieval. For hierarchical modeling, RAP-
TOR (Sarthi et al., 2024), SiReRAG (Zhang et al.,
2025c), and HiRAG (Huang et al., 2025) build
tree-structured indices for multi-granular evidence
integration. However, these methods rely on pair-
wise edges that cannot explicitly group multiple
scattered yet semantically related memories.
Recent works (Luo et al., 2025; Feng et al., 2025;
Sharma et al., 2024; Hu et al., 2025a) preliminarily
explore hypergraphs to model multi-entity relations
with hyperedges. However, these approaches are
designed for static knowledge bases with determi-
nate corpora, where agentic memory continuously
evolves with ongoing dialogues. Besides, they lack
a hierarchical retrieval mechanism capable of pre-
serving semantic coherence across extended dia-
logues. Our work pioneers the hypergraph in struc-
turing agentic memory, which has quite different
problem settings and technical designs.
2.2 Memory System of Agents
Recent agents have used RAG to model long-term
memory, where MemoryBank (Zhong et al., 2024),
A-Mem (Xu et al., 2025), Mem0 (Chhikara et al.,
2025), and Zep (Rasmussen et al., 2025) build struc-
tured or graph-based representations for persistence
between sessions and tracking of the evolution of
facts. G-Memory (Zhang et al., 2025a) and Light-
Mem (Fang et al., 2025) further explore hierarchi-
cal structures and compression for efficiency.
In parallel, several approaches eschew explicit
retrieval. MemGPT (Packer et al., 2023) and
MemOS (Li et al., 2025b) draw on abstractions
from operating systems with hierarchical mem-
ory and modular scheduling. MIRIX (Wang and
Chen, 2025) coordinates multi-agent states via
shared memory spaces, while Nemori (Nan et al.,
2025) and MemGen (Zhang et al., 2025b) form

Query: What did Bob do to relieve stress after work?
UserQuery: What did Bob do to relieve stress after work? MemoryContext: Relevant Episodes: {Project launches; they run together}. Relevant Facts: {Running helps relieve stress}.①Topic RetrievalSelected Topic 𝑻!:Bob's work and projectCandidate Episodes: {𝑬!,𝑬"}②Episode RetrievalSelected Episode 𝑬":Project launches; they run togetherCandidate Elements: {𝑭#,𝑭"}③Fact RetrievalSelected Fact: 𝑭#:Running helps relieve stressRetrieval
Answer: Running.
Agent𝑭!: Bob's deadline is approaching𝑭": Running helps relieve stress…𝑻#: Bob's work and projectConnected Nodes: {𝑬#,𝑬"}Episode Nodes 𝑬𝑻$: Alice's marathon trainingConnected Nodes: {𝑬$,𝑬!,𝑬"}Topic Nodes 𝑻
Episode Hyperedges
Node
HyperedgeFact Nodes 𝑭
Fact Hyperedges𝑬": Project launches; they run together𝑬!: Alice can now run 15km𝑬#: Bob faces deadline with overtime𝑬$: Alice signs up for marathon
I signed up for the marathon next month.Great! Take it slow at first.03/05/2025
How's your project going?Deadline'ssoon. Working overtime every day.03/18/2025
How's your marathon training?Good! I can run 15km now.05/02/2025
Project launched! Let's celebrate with a run.Sure! Running really helps relieve stress.05/15/2025IndexingConversationHypergraph
𝑭$: Alice signed up for marathon𝑭#: Alice can run 15km…
(d) Hypergraph Memory(b) Topic Aggregation(a) Episode Detection(c) Fact ExtractionFigure 2: Framework of HyperMem. The indexing detects episode boundaries, aggregates topics via hyperedges,
and extracts facts. The retrieval performs coarse-to-fine search from topics to episodes to facts.
compressible or generative latent representations.
MemInsight (Salama et al., 2025), Mem1 (Zhou
et al., 2025), Memory-R1 (Yan et al., 2025), and
Mem- α(Wang et al., 2025) employ reinforcement
learning to autonomously optimize memory stor-
age and retrieval policies. In contrast, HyperMem
explicitly groups topically related memories via
hyperedges and employs topic-guided hierarchical
retrieval to ensure relevance across temporal gaps.
3 Approach
In this section, we present the HyperMem archi-
tecture for long-term conversational agents, includ-
ing hypergraph memory structure, hypergraph con-
struction from dialogue streams, and hypergraph-
guided retrieval for response generation.
3.1 Hypergraph Memory Structure
To capture higher-order associations among related
elements, we model memories with hypergraphs.
Unlike conventional graphs limited to pairwise rela-
tions, hypergraphs connect multiple nodes via a sin-
gle hyperedge. This enables richer relational mod-
eling and naturally reflects the associative nature
of human memory (Anderson and Bower, 2014).
To effectively organize this memory, we design
a three-level hypergraph architecture, where hyper-
edges link nodes within each level:•Topic-level: Captures dialogues sharing a com-
mon theme across long-term interactions, facili-
tating long-range topical associations.
•Episode-level: Represents temporally contigu-
ous dialogue segments that describe a coherent
event or sub-conversation.
•Fact-level: Encodes atomic facts extracted from
episodes, serving as precise retrieval targets for
query-based access.
Formally, given an input dialogue stream X=
{xt}T
t=1, we construct the memory hypergraph as:
H= (VT∪ VE∪ VF,EE∪ EF),(1)
where VT,VE,VFdenote the topic, episode, and
fact nodes, respectively. Here, hyperedges EEcon-
nect all episode nodes within the same topic along
with each node weight wE∈[0,1] , while hyper-
edgesEFconnect all fact nodes belonging to the
same episode with the node weightwF∈[0,1].
3.2 Hypergraph Memory Construction
To construct the hypergraph memory, we employ
a three-stage process. We first detectepisodesby
segmenting the raw dialogue stream into atomic se-
quences, then aggregate topically related episodes
intotopics, and finally extract queryable informa-
tivefactsgrounded in their context.

3.2.1 Episode Detection
A dialogue stream often interweaves multiple
events and shifts topics over time. Storing it as
a monolithic block would obscure event bound-
aries and entangle events of interest with irrelevant
context. To address this, we introduceEpisode
to enable precise event boundary preservation and
isolate irrelevant content from dialogue context.
Method.To derive episodes, we design an LLM-
driven streaming boundary detection mechanism.
Consider an incoming dialogue stream X=
{xt}T
t=1. We employ a buffer Hto pend the history,
and determine if the incoming dialogue completes
a coherent episode. Specifically, for each incom-
ingxt, we add it to H<tand invoke an LLM-based
boundary detector that evaluates: (1)semantic com-
pletenessof current buffer H≤t, (2) thetime gap
between consecutive dialogues, and (3)linguistic
signalsindicating topic transition or completion.
The detector outputs two signals: should_end ,
i.e., the buffer forms a semantically complete event,
andshould_wait , i.e., the event is still unfold-
ing and requires further input. If should_end is
triggered, we create an informativeEpisode node,
i.e.,vE= (vE
dialogue , vE
title, vE
episode ), where vE
dialogue
stores the raw conversation turns, vE
titleabstracts a
concise subject, and vE
episode offers a brief narrative
summary. The buffer is then cleared, and process-
ing continues with subsequent dialogues. For the
algorithm and prompt, see Algo. 1 and Figure 6.
Remark.In this way, we process dialogue
streams incrementally and segment them into se-
mantically coherent memory units. This reduces ir-
relevant context and also improves the convenience
of topic organization and retrieval.
3.2.2 Topic Aggregation
Episodes capture event-level fragments within con-
tiguous temporal windows. However, as shown
in Figure 1, real-world narratives about a specific
topic can also be temporally dispersed. Existing
designs (Chhikara et al., 2025; Luo et al., 2025)
usually isolate such correlated associations, mak-
ing it difficult to retrieve the full narrative. To
address this, we deviseTopicto aggregate scat-
tered episodes, and leverage hyperedges to connect
multiple episodes that belong to the same topic.
Method.Practically, we design an LLM-driven
streaming topic aggregation mechanism. Given the
current target episode vE
cur, we retrieve historicalsimilar episodes CEusing lexical and semantic sim-
ilarity (detailed in § 3.3.1). By comparing vE
curwith
CE, there are three cases to handle:
1.Topic Initialization.If CE=∅, we create a
new topic vT= (vT
title, vT
summary )forvE
cur. Here,
vT
titleandvT
summary are the title and summary ac-
cording tovE
curgenerated by the LLM.
2.Topic Creation.If CE̸=∅ but the potential
topic of vE
curis different from the existing topics
of episodes in CE, we create a new topic vT=
(vT
title, vT
summary )forvE
cur, by comparing vE
curwith
all episodes inCEby the LLM.
3.Topic Update.If CE̸=∅ and the potential
topic of vE
curexisted in CE, we update each
matched topic incorporating vE
curand regener-
ating its metadatavT= (vT
title, vT
summary ).
After this process, we construct a hyperedge eE
t∈
EElinking the topic to all its constituent episodes,
and the LLM assigns an importance weight wE
e,v∈
[0,1] to each episode based on its contribution to
the topic. For the algorithm and prompt, see Algo. 1
and Figure 7.
Remark.In this way, the resulting topic nodes
act as semantic anchors of episodes potentially
spanning weeks or months. This also enables com-
prehensive retrieval of entire narratives by query
matching, regardless of temporal fragmentation.
3.2.3 Fact Extraction
Episodes preserve rich narrative context but often
contain verbose dialogue that is inefficient for di-
rect query answering. To enable query-oriented
retrieval, we extractFactswith language expres-
sions, the compact assertion grounded in episode
context, as fine-grained memory units.
Method.Given a topic tand its associated
episodes VE
t, we use an LLM to identify salient
factual assertions, using the full topical con-
text to avoid redundant or trivial extractions.
Here, each fact node is formed as vF=
(vF
content , vF
potential , vF
keywords ), where vF
content records
the factual assertion, vF
potential lists query patterns
this fact is likely to answer, enabling proac-
tive alignment with user’s potential intents, and
vF
keywords captures representative terms to facilitate
keyword-based retrieval. To maintain provenance,
each fact is explicitly anchored to the original
episode(s). For each episode vE, we construct a

fact hyperedge eF∈ EFthat connects all the facts
involved, with the LLM assigning an importance
weight wF
e,v∈[0,1] to reflect the relative impor-
tance of each fact. For the algorithm and prompt,
see Algo. 1 and Figure 8.
Remark.In this way, the resulting fact nodes
serve as atomic query-targeted units. Unlike raw
dialogue for retrieval, vF
potential anticipates relevant
queries while vF
keywords supports lexical search, al-
lowing retrieval with concise, directly answerable
evidence rather than verbose transcripts.
3.3 Hypergraph Memory Retrieval
To respond to the user’s query, the agent retrieves
relevant memories through a coarse-to-fine process
that traverses fromtopictoepisodetofact. This
combines an offline indexing phase with an online
retrieval strategy for practical usage.
3.3.1 Offline Index Construction
User queries often exhibit both lexical cues and
semantic intent, which are crucial to accurately
retrieve relevant memories. To fully leverage
both signals, we construct dual indices for all
node types, including topic, episode and fact: a
sparse keyword-based index using BM25 (Robert-
son and Zaragoza, 2009), and a dense semantic
index powered by Qwen3-Embedding-4B (Zhang
et al., 2025d). Specifically, each node is first con-
verted into a textual document for BM25 indexing
to support exact keyword matching, and then en-
coded into a dense vector via the embedding model
to capture deeper semantic similarity.
Hypergraph Embedding Propagation.The
nodes linked by the same hyperedge share a com-
mon topical context, and are expected to acquire
similar representations. To this end, we propose
a lightweight embedding propagation process that
enriches node embeddings by aggregating infor-
mation from their incident hyperedges. First, we
compute a hyperedge embedding as a weighted
aggregation of its constituent node embeddings:
he=X
v∈V(e)αe,vhv,
αe,v=exp(w e,v)P
u∈V(e) exp(w e,u),(2)
where hvdenotes the initial (dense) embedding of
node v, and we,v∈[0,1] is the importance weightassigned during topic aggregation, e.g., by an LLM
based on narrative contribution.
Next, we refine the representation of each node
by aggregating the embeddings of all hyperedges
in which it participates:
h′
v=hv+λ·Agge∈N(v) (he),(3)
where N(v) denotes the set of hyperedges incident
tov,λ≥0 is a hyperparameter to control the
strength of propagation, and Agg is an aggregation
function, e.g., summation. See Algo. 2 for the
algorithm.
Remark.This propagation mechanism is in-
spired by hypergraph neural networks (Feng et al.,
2019), yet remains lightweight without large-scale
fine-tuning. Empirical studies demonstrate its effec-
tiveness. Besides, it enables semantically related
memories to acquire aligned embeddings, which
derive more informative embeddings and also facil-
itate high-order associations during retrieval.
3.3.2 Online Retrieval Strategy
Given a user query q, retrieval proceeds as a struc-
tured coarse-to-fine traversal with progressive top-
kselection at each level.
Stage 1: Topic Retrieval.We retrieve from the
topic-level to establish the topical context. All
topic nodes VTare scored using both keyword and
vector indices, with rankings fused via Reciprocal
Rank Fusion (RRF):
RRF(d) =MX
m=11
k+ rank m(d)(4)
where mindexes individual rankers and kis
a smoothing constant. The RRF-ranked candi-
dates are then refined by a reranker model, which
computes fine-grained query-document relevance
scores to improve ranking precision. We select the
top-kTtopic nodes as candidates, which filters out
most irrelevant topical contexts.
Stage 2: Episode Retrieval.For each selected
topic t, we expand to its constituent episodes VE
t
via the episode-hyperedge eE
t. Following Stage
1, the expanded episodes are scored via RRF and
then refined by the reranker. We retain the top- kE
episodes as the results. This stage ensures that only
the query-relevant temporal segments within each
topic are preserved.

Stage 3: Fact Retrieval.Finally, each retained
episode eis expanded to its supporting facts VF
e
through the fact hyperedge eF
e. Following the same
RRF-then-rerank pipeline, we select the top- kF
facts as the final retrieval result.
Final Response Generation.Instead of using
verbose raw dialogue text, we construct there-
sponse contextfrom the content fields of retrieved
facts, optionally augmented with the summary
fields of their sourced upper-levelepisodesfor nar-
rative context. This design significantly reduces
token consumption while preserving answerable in-
formation. The constructed response context is in-
put into the conversational agent, and the response
is returned as the answer to the user query. See
Algo. 3 for the algorithm.
4 Experiments
In this section, we conduct experiments to evaluate
the effectiveness of our HyperMem.
4.1 Experimental Setup
Benchmark.LoCoMo (Maharana et al., 2024)
is a benchmark dataset designed to evaluate long-
term memory capabilities in conversational AI sys-
tems. It contains multi-session dialogues spanning
several months, with four categories of questions:
single-hop (direct fact retrieval), multi-hop (rea-
soning across multiple dialogue turns), temporal
reasoning (time-related queries), and Open Domain
(open-ended questions requiring broader context
understanding).
Baselines.We compare our approach against rep-
resentative methods from RAG and memory sys-
tem. (1)RAGmethods: RAG, GraphRAG (Edge
et al., 2024), LightRAG (Guo et al., 2025),
HippoRAG 2 (Gutiérrez et al., 2025), and Hy-
perGraphRAG (Luo et al., 2025). (2)Mem-
ory systemmethods: OpenAI2, LangMen3,
Zep (Rasmussen et al., 2025), A-Mem (Xu et al.,
2025), Mem0 (Chhikara et al., 2025), Mem-
Graph (Chhikara et al., 2025), MIRIX (Wang and
Chen, 2025), Memobase4, MemU5, MemOS (Li
et al., 2025b).
2https://openai.com/zh-Hans-CN/index/
memory-and-new-controls-for-chatgpt/
3https://langchain-ai.github.io/langmem/
4https://www.memobase.io/blog/
ai-memory-benchmark
5https://memu.pro/Implementation Details.We implement Hyper-
Mem using Qwen3-Embedding-4B for semantic
encoding and Qwen3-Reranker-4B for reranking.
For answer generation, we employ GPT-4.1-mini
with chain-of-thought prompting. In hierarchical
retrieval, we first retrieve 100initial candidates,
then select top- 10Topics, top- 10Episodes, and
top-30Facts as the final context. Node embeddings
are updated with a propagation weight λ= 0.5 to
incorporate hyperedge information. For evaluation,
we use GPT-4o-mini as the LLM judge and report
the average scores across3independent runs.
4.2 Main Results
Table 1 presents the main results. HyperMem
achieves the best overall accuracy of 92.73%,
outperforming the strongest RAG method Hyper-
GraphRAG (86.49%) by 6.24% and the best mem-
ory system MIRIX (85.38%) by 7.35%.
Regarding category-wise performance, Hy-
perMem excels on reasoning-intensive tasks.
On Single-hop questions, HyperMem achieves
96.08%, surpassing HyperGraphRAG by 5.47%,
as the structured fact layer enables precise retrieval
of atomic information. On Multi-hop questions
requiring evidence aggregation across multiple dia-
logue segments, HyperMem reaches 93.62%, out-
performing LightRAG by 9.58%, demonstrating
that hyperedges effectively bind topically related
episodes scattered across time for comprehensive
evidence collection. On Temporal questions re-
quiring cross-session reasoning, HyperMem attains
89.72%, benefiting from the episode layer’s preser-
vation of temporal anchors and the hierarchical
structure’s ability to trace event progression. Open
Domain remains challenging for all methods due to
broader knowledge requirements beyond the con-
versation history.
These improvements stem from two key de-
signs. Hyperedges explicitly group topically re-
lated episodes, ensuring complete evidence re-
trieval for multi-hop reasoning. Meanwhile, topic-
guided hierarchical retrieval progressively narrows
the candidate pool, filtering irrelevant context while
preserving temporal coherence.
4.3 Ablation Study
As shown in Table 2 and Figure 3, we conduct ab-
lation study to evaluate the contribution of each
component in HyperMem. The results reveal that
Episode context is the most critical component,
as removing it (w/o EC) causes the largest perfor-

Methods Single-hop Multi-hop Temporal Open Domain Overall
GraphRAG (Edge et al., 2024) 79.55 54.96 50.16 58.33 67.60
LightRAG (Guo et al., 2025) 86.68 84.04 60.75 71.88 79.87
HippoRAG 2 (Gutiérrez et al., 2025) 86.44 75.89 78.50 66.67 81.62
HyperGraphRAG (Luo et al., 2025) 90.61 80.85 85.36 70.83 86.49
OpenAI263.79 42.92 21.71 62.29 52.90
LangMem362.23 47.92 23.43 71.12 58.10
Zep (Rasmussen et al., 2025) 61.70 41.35 49.3176.6065.99
A-Mem (Xu et al., 2025) 39.79 18.85 49.91 54.05 48.38
Mem0 (Chhikara et al., 2025) 67.13 51.15 55.51 72.93 66.88
Mem0g(Chhikara et al., 2025) 65.71 47.19 58.13 75.71 68.44
MIRIX (Wang and Chen, 2025)†85.11 83.70 88.39 65.62 85.38
Memobase473.12 64.65 81.20 53.12 72.01
MemU566.34 63.12 27.10 50.01 56.55
MemOS (Li et al., 2025b) 81.09 67.49 75.18 55.90 75.80
HyperMem (Ours) 96.08 93.62 89.7270.8392.73
Table 1: Comparison of HyperMem with RAG-based and memory system methods on the LoCoMo benchmark.
Evaluation metric is LLM-as-a-judge accuracy (%) scored by GPT-4o-mini. †indicates results obtained using
GPT-4.1-mini. Results for RAG-based methods are reproduced using their official implementations. Results for
memory systems are primarily sourced from Chhikara et al. (2025); Wang and Chen (2025); Li et al. (2025b).
HyperMemw/o FC w/o EC w/o TR
w/o TR & FC w/o TR & EC w/o TR & ER889092949698
96.195.7
92.894.7 94.7
91.993.8Single-Hop
HyperMemw/o FC w/o EC w/o TR
w/o TR & FC w/o TR & EC w/o TR & ER84868890929496
93.6
90.8
89.090.891.1
90.4
87.9Multi-Hop
HyperMemw/o FC w/o EC w/o TR
w/o TR & FC w/o TR & EC w/o TR & ER80828486889092
89.7
87.5
84.189.190.0
85.486.3Temporal
HyperMemw/o FC w/o EC w/o TR
w/o TR & FC w/o TR & EC w/o TR & ER65.067.570.072.575.077.580.0
69.874.0
70.874.0 74.0
68.878.1Open DomainLLM-as-a-judge (%)HyperMem
w/o FC
w/o EC
w/o TR
w/o TR & FC
w/o TR & EC
w/o TR & ER
Figure 3: Ablation study across four question categories. FC: Fact context. EC: Episode context. TR: Topic-level
retrieval. ER: Episode-level retrieval. The shaded region highlights the full HyperMem configuration.
mance drop (-3.76% overall), particularly affecting
Temporal reasoning (-5.61%). The hierarchical re-
trieval mechanism also proves essential. Bypassing
Topic retrieval (w/o TR) shows moderate impact,
but completely flattening the hierarchy to Fact-only
retrieval (w/o TR & ER) significantly degrades
Multi-Hop performance (-5.68%), demonstrating
that the hierarchical structure effectively maintains
coherent information flow across granularity lev-
els. Fact context primarily benefits Multi-Hop rea-
soning (-2.84% when removed). These findingsvalidate that our three-level memory architecture
and hierarchical retrieval strategy work synergisti-
cally to achieve optimal performance across diverse
question types.
4.4 Hyperparameter Analysis
We investigate the sensitivity of HyperMem to key
hyperparameters across four dimensions. First, the
fusion coefficient α= 0.5 achieves optimal perfor-
mance (92.66%), indicating that balanced integra-
tion of semantic similarity and structural retrieval

0 0.5 1.0
Lambda ( )
84868890929496LLM-as-a-judge (%)
Lambda Comparison
under sum, 10-15-30
91.69
89.0992.66
88.9091.88
89.48
1 5 10
Topic Top-k707580859095
Topic Top-k Comparison
under sum, =0.5, x-15-30
76.88
74.8790.78
87.6092.66
88.90
10 15 20
Episode Top-k84868890929496
Episode Top-k Comparison
under sum, =0.5, 10-x-30
92.73
88.4492.66
88.9092.47
89.09
20 30 40
Fact Top-k84868890929496
Fact Top-k Comparison
under sum, =0.5, 10-15-x
91.49
88.7792.66
88.9091.62
88.90Episode + Fact Fact OnlyFigure 4: Hyperparameter sensitivity analysis on LoCoMo. We evaluate the impact of embedding fusion weight α
and Top-k selection at each hierarchical level (Topic, Episode, Fact) on retrieval performance.
Configuration Overall (%)∆
HyperMem 92.66–
w/o FC 91.75 0.91↓
w/o EC 88.90 3.76↓
w/o TR 91.94 0.72↓
w/o TR & FC 91.75 0.91↓
w/o TR & EC 88.83 3.83↓
w/o TR & ER 90.19 2.47↓
Table 2: Ablation study. FC: Fact Context, EC: Episode
Context, TR: Topic Retrieval, ER: Episode Retrieval.
yields the best results. Second, topic top-k exhibits
the most significant impact: increasing from k=1 to
k=10 improves accuracy from 76.88% to 92.66%
(+15.78%), demonstrating that adequate topical
coverage is crucial for capturing relevant context.
In contrast, episode top-k shows minimal sensitiv-
ity (92.73% at k=10 vs. 92.47% at k=20), suggest-
ing the system is robust to this parameter. Fact
top-k peaks at k=30 (92.66%) with slight degra-
dation at higher values, indicating potential noise
introduction from excessive fact retrieval. Notably,
the “Fact + Episode” configuration consistently out-
performs “Fact Only” by 3-4% across all settings,
further validating the importance of episode-level
context in our framework.
4.5 Efficient Analysis
Figure 5 shows the efficiency-accuracy trade-off.
HyperMem achieves optimal 92.73% accuracy at
7.5x tokens with the “Episode + Fact” configu-
ration, while the “Fact Only” configuration al-
ready reaches 89.48% at merely 2.5x tokens, both
substantially outperforming RAG-based methods
that require 25-35x tokens for lower accuracy
(GraphRAG: 67.60% at 35.3x, HyperGraphRAG:
86.49% at 26.3x). The “Episode + Fact” configura-
1k 2k 5k 10k 20k 40k
Token Usage6065707580859095LLM-as-a-judge (%)Better
GraphRAG (35.3x)LightRAG (25.6x)HippoRAG 2 (2.2x)HyperGraphRAG (26.3x)
Zep (1.4x)
Mem0 (1.0x)MemOS (2.5x)
MemU (3.9x)F10 (1.3x)
F20 (1.9x)F30 (2.5x)E10+F10 (4.7x)
E10+F20 (5.2x)E10+F30 (7.5x)
RAG Methods
Memory SystemsHyperMem (Fact Only)
HyperMem (Episode + Fact)Figure 5: Token usage vs. accuracy comparison. The
x-axis shows relative token usage (Mem0 as 1.0 ×base-
line), and the y-axis shows LLM-as-a-judge accuracy.
tion consistently outperforms “Fact Only” by 3-4%,
demonstrating that episode summaries provide cru-
cial semantic guidance that cannot be compensated
by retrieving more facts.
5 Conclusion
In this paper, we propose a hypergraph-based agen-
tic memory architecture, namely HyperMem. It
explicitly models high-order associations among
topics, episodes, and facts, overcoming the pair-
wise limitations of existing RAG and graph-based
methods. By organizing memory hierarchically
and linking related elements via hyperedges, Hyper-
Mem unifies scattered dialogue content into coher-
ent units. This enables effective lexical-semantic
indexing with hypergraph embedding propagation
and efficient coarse-to-fine retrieval. On the Lo-
CoMo benchmark, HyperMem achieves state-of-
the-art 92.73% LLM-as-a-judge accuracy, demon-
strating its strength in long-term conversations.

Limitations
The current design assumes a single-user scenario,
and extending to multi-user or multi-agent settings
presents challenges in access control and mem-
ory isolation. Additionally, Open Domain ques-
tions remain challenging as they often require ex-
ternal knowledge beyond the conversation history,
suggesting opportunities for integrating external
knowledge bases.
References
John R Anderson and Gordon H Bower. 2014.Human
associative memory. Psychology press.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.
Orlando Ayala and Patrice Béchard. 2024. Reduc-
ing hallucination in structured outputs via retrieval-
augmented generation. InProceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies: Industry Track, NAACL
2024, Mexico City, Mexico, June 16-21, 2024, pages
228–238. Association for Computational Linguistics.
Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze
Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.
2025. Pathrag: Pruning graph-based retrieval aug-
mented generation with relational paths.CoRR,
abs/2502.14902.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
production-ready AI agents with scalable long-term
memory.CoRR, abs/2504.19413.
Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F. Yang,
and Anton Tsitsulin. 2024. Don’t forget to connect!
improving RAG with graph-based reranking.CoRR,
abs/2405.18414.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.CoRR, abs/2404.16130.
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. Minirag: Towards extremely
simple retrieval-augmented generation.CoRR,
abs/2501.06713.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting llms: Towardsretrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, KDD 2024,
Barcelona, Spain, August 25-29, 2024, pages 6491–
6501. ACM.
Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang,
Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao,
Mengru Wang, Shuofei Qiao, Huajun Chen, and
Ningyu Zhang. 2025. Lightmem: Lightweight and
efficient memory-augmented generation.CoRR,
abs/2510.18866.
Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shi-
hui Ying, Shaoyi Du, Han Hu, and Yue Gao. 2025.
Hyper-rag: Combating LLM hallucinations using
hypergraph-driven retrieval-augmented generation.
CoRR, abs/2504.08758.
Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong
Ji, and Yue Gao. 2019. Hypergraph neural networks.
InProceedings of the AAAI conference on artificial
intelligence, volume 33, pages 3558–3565.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey.CoRR, abs/2312.10997.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InAdvances in Neural Information
Processing Systems 38: Annual Conference on Neu-
ral Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15,
2024.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models. InForty-second International Conference
on Machine Learning, ICML 2025, Vancouver, BC,
Canada, July 13-19, 2025. OpenReview.net.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V . Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. InAdvances in Neural Information
Processing Systems 38: Annual Conference on Neu-
ral Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15,
2024.
Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue,
Xingliang Hou, Zhiqiang Tian, Yue Gao, and
Shaoyi Du. 2025a. Cog-rag: Cognitive-inspired

dual-hypergraph with theme alignment retrieval-
augmented generation.CoRR, abs/2511.13201.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen
Ling, and Liang Zhao. 2025b. GRAG: graph
retrieval-augmented generation. InFindings of the
Association for Computational Linguistics: NAACL
2025, Albuquerque, New Mexico, USA, April 29 -
May 4, 2025, pages 4145–4157. Association for Com-
putational Linguistics.
Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen,
and James Cheng. 2025. Retrieval-augmented
generation with hierarchical knowledge.CoRR,
abs/2503.10150.
Mandar Kulkarni, Praveen Tangarajan, Kyung Kim,
and Anusua Trivedi. 2024. Reinforcement learning
for optimizing RAG for domain chatbots.CoRR,
abs/2401.06800.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
Mufei Li, Siqi Miao, and Pan Li. 2025a. Simple is effec-
tive: The roles of graphs and large language models
in knowledge-graph-based retrieval-augmented gen-
eration. InThe Thirteenth International Conference
on Learning Representations, ICLR 2025, Singapore,
April 24-28, 2025. OpenReview.net.
Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang,
Chen Tang, Simin Niu, Ding Chen, Jiawei Yang,
Chunyu Li, Qingchen Yu, Jihao Zhao, Yezhaohui
Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao
Huo, Tianyi Chen, Kai Chen, Kehang Li, and 20
others. 2025b. Memos: A memory OS for AI system.
CoRR, abs/2507.03724.
Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low,
Anshumali Shrivastava, and Vijai Mohan. 2025. RE-
FRAG: rethinking RAG based decoding.CoRR,
abs/2509.01092.
Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng,
Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Ze-min
Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan.
2025. Hypergraphrag: Retrieval-augmented genera-
tion with hypergraph-structured knowledge represen-
tation.CoRR, abs/2503.21322.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2024. Reasoning on graphs: Faithful
and interpretable large language model reasoning. In
The Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of LLM agents. InProceedings of the
62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), ACL
2024, Bangkok, Thailand, August 11-16, 2024, pages
13851–13870. Association for Computational Lin-
guistics.
Jiayan Nan, Wenquan Ma, Wenlong Wu, and Yize Chen.
2025. Nemori: Self-organizing agent memory in-
spired by cognitive science.CoRR, abs/2508.03341.
Charles Packer, Vivian Fang, Shishir G. Patil, Kevin
Lin, Sarah Wooders, and Joseph E. Gonzalez. 2023.
Memgpt: Towards llms as operating systems.CoRR,
abs/2310.08560.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais,
Jack Ryan, and Daniel Chalef. 2025. Zep: A tempo-
ral knowledge graph architecture for agent memory.
CoRR, abs/2501.13956.
Stephen E. Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.
Rana Salama, Jason Cai, Michelle Yuan, Anna Currey,
Monica Sunkara, Yi Zhang, and Yassine Benajiba.
2025. Meminsight: Autonomous memory augmenta-
tion for LLM agents.CoRR, abs/2503.21760.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
Kartik Sharma, Peeyush Kumar, and Yunqing Li. 2024.
OG-RAG: ontology-grounded retrieval-augmented
generation for large language models.CoRR,
abs/2412.15235.
Yu Wang and Xi Chen. 2025. MIRIX: multi-agent
memory system for llm-based agents.CoRR,
abs/2507.07957.
Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao,
Yuanzhe Hu, Julian J. McAuley, and Xiaojian Wu.
2025. Mem- α: Learning memory construction via
reinforcement learning.CoRR, abs/2509.25911.
Yuan Xia, Jingbo Zhou, Zhenhui Shi, Jun Chen, and
Haifeng Huang. 2025. Improving retrieval aug-
mented language model with self-reasoning. In
AAAI-25, Sponsored by the Association for the Ad-
vancement of Artificial Intelligence, February 25 -
March 4, 2025, Philadelphia, PA, USA, pages 25534–
25542. AAAI Press.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao
Tan, and Yongfeng Zhang. 2025. A-MEM: agentic
memory for LLM agents.CoRR, abs/2502.12110.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong
Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Hin-
rich Schütze, V olker Tresp, and Yunpu Ma. 2025.
Memory-r1: Enhancing large language model agents
to manage and utilize memories via reinforcement
learning.CoRR, abs/2508.19828.
Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu,
Kun Wang, and Shuicheng Yan. 2025a. G-memory:
Tracing hierarchical memory for multi-agent systems.
CoRR, abs/2506.07398.
Guibin Zhang, Muxin Fu, and Shuicheng Yan. 2025b.
Memgen: Weaving generative latent memory for self-
evolving agents.CoRR, abs/2509.24704.
Nan Zhang, Prafulla Kumar Choubey, Alexander R. Fab-
bri, Gabriel Bernadett-Shapiro, Rui Zhang, Prasenjit
Mitra, Caiming Xiong, and Chien-Sheng Wu. 2025c.
Sirerag: Indexing similar and related information for
multihop reasoning. InThe Thirteenth International
Conference on Learning Representations, ICLR 2025,
Singapore, April 24-28, 2025. OpenReview.net.
Wenyuan Zhang, Xinghua Zhang, Haiyang Yu, Shuaiyi
Nie, Bingli Wu, Juwei Yue, Tingwen Liu, and Yong-
bin Li. 2026. Expseek: Self-triggered experience
seeking for web agents.Preprint, arXiv:2601.08605.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025d. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
CoRR, abs/2506.05176.
Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li,
Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong
Wen. 2025e. A survey on the memory mechanism of
large language model-based agents.ACM Trans. Inf.
Syst., 43(6):155:1–155:47.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye,
and Yanlin Wang. 2024. Memorybank: Enhancing
large language models with long-term memory. In
Thirty-Eighth AAAI Conference on Artificial Intelli-
gence, AAAI 2024, Thirty-Sixth Conference on Inno-
vative Applications of Artificial Intelligence, IAAI
2024, Fourteenth Symposium on Educational Ad-
vances in Artificial Intelligence, EAAI 2014, Febru-
ary 20-27, 2024, Vancouver, Canada, pages 19724–
19731. AAAI Press.
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim,
Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan
Kian Hsiang Low, and Paul Pu Liang. 2025. MEM1:
learning to synergize memory and reasoning for effi-
cient long-horizon agents.CoRR, abs/2506.15841.
A Algorithm
As shown in Algorithm 1, 2, and 3, we provide
detailed pseudocode for HyperMem’s core proce-
dures.B Prompt Templates
We present the key prompt templates used in Hy-
perMem. Figure 6 shows the episode boundary
detection prompt. Figure 7 describes the topic
aggregation prompt for linking related episodes.
Figure 8 presents the fact extraction prompt for
distilling key information from episodes.
C Case Study
We present four representative cases from the Lo-
CoMo benchmark to illustrate how HyperMem ad-
dresses different query types where baselines fail.
Single-Hop Task (Figure 9).This case asks
“What new activity did Maria start recently, as men-
tioned on 3 June, 2023?” GraphRAG confuses
“dog shelter” with “homeless shelter,” while Hyper-
GraphRAG retrieves “aerial yoga” from a different
time period. HyperMem’s hierarchical retrieval
navigates through Topic and Episode layers to re-
trieve the exact Fact containing “volunteering at
a local dog shelter,” directly matching the golden
answer.
Multi-Hop Task (Figure 10).Answering “How
many tournaments has Nate won?” requires ag-
gregating evidence from 7 sessions spanning 10
months. GraphRAG only identifies “at least two”
due to its pairwise edge structure fragmenting re-
lated memories across time. HyperMem’s Topic hy-
peredge groups all tournament-related Episodes un-
der a unified thematic anchor, correctly answering
“seven tournaments” with precise dates for each.
Temporal Reasoning Task (Figure 11).For the
query “How many pets did Andrew have, as of
September 2023?” GraphRAG claims Andrew
had no pets by confusing him with another per-
son, while HyperGraphRAG overcounts with “four
pets.” HyperMem correctly answers “one pet dog
named Toby” because its Episode layer preserves
temporal anchors and enables accurate state recon-
struction at the queried time point.
Open Domain Task (Figure 12).For “Would
John be open to moving to another country?” Hy-
perGraphRAG incorrectly answers “Yes” based on
superficial travel mentions. HyperMem correctly
infers “No” by synthesizing evidence about John’s
military aspirations and political campaign goals
that anchor him to the U.S. The potential field in
Fact nodes anticipates such inference patterns.

Algorithm 1Hypergraph Memory Construction
1:Input:Dialogue streamX={x t}T
t=1
2:Output:HypergraphH
3:Initialize VT,VE,VF,EE,EF← ∅ , buffer
B ← ∅
▷Stage 1: Episode Detection
4:foreach incoming dialoguex t∈Xdo
5:B ← B ∪ {x t}
6: Boundary detection: (end,wait)←LLM(B)
7:ifend=Truethen
8:vE←CREATEEPISODE(B)
9:VE← VE∪ {vE},B ← ∅
10:end if
11:end for
▷Stage 2: Topic Aggregation
12:foreach new episodevE
cur∈ VEdo
13:Episode Matching:CE←LLM(vE
cur)
14: ifCE=∅(Case 1: Topic Initialization)then
15:vT←CREATETOPIC(vE
cur)
16:VT← VT∪ {vT}
17:else
18: Topic Matching: CT←LLM(CE, vE
cur)
19:ifCT=∅(Case 2: New Topic)then
20:vT←CREATETOPIC(CE, vE
cur)
21:VT← VT∪ {vT}
22:else(Case 3: Topic Update)
23:UPDATETOPICS(CT, vE
cur)
24:end if
25:end if
26:eE
t←(vT,GETEPISODES(vT),wE)
27:EE← EE∪ {eE
t}
28:end for
▷Stage 3: Fact Extraction
29:foreach topicvT∈ VTdo
30:VE
t←GETEPISODES(vT)
31:Fact Extraction:F t←LLM(vT,VE
t)
32:foreach factvF
t∈ Ftdo
33:VF
t← VF
t∪ {vF
t}
34:AnchorvF
tto its source episode(s)
35:end for
36:foreach episodevE
t∈ VE
tdo
37:eF
t←(vE
t,GETFACTS(vE
t),wF)
38:EF
t← EF
t∪ {eF
t}
39:end for
40:end for
41:returnH= (VT∪ VE∪ VF,EE∪ EF)Algorithm 2Offline Index Construction
1:Input:HypergraphH
2:Output:Indexed hypergraph with propagated
embeddings
▷Node Indexing
3:foreach nodev∈ VT∪ VE∪ VFdo
4:Build BM25 and vector index forv
5:h v←ENCODE(v)
6:end for
▷Hyperedge Embedding
7:foreach hyperedgee∈ EE∪ EFdo
8:h e←P
v∈eαe,vhv
9:end for
▷Embedding Propagation
10:foreach nodevdo
11:h′
v←h v+λ·AGG e∈N(v) (he)
12:end for
Algorithm 3Online Retrieval Strategy
1:Input:Query q, Indexed H, Top- k:
(kT, kE, kF)
2:Output:Retrieved contextR
3:q←ENCODE(q)
▷Stage 1: Topic Retrieval
4:foreachvT∈ VTdo
5:sT←RRF(BM25(q, vT),COS(q,h′
vT))
6:end for
7:T top←TOPK(VT, sT, kT)
▷Stage 2: Episode Retrieval
8:VE
t←S
t∈T topGETEPISODES(t)
9:foreachvE
t∈ VE
tdo
10:sE
t←RRF(BM25(q, vE
t),COS(q,h′
vE
t))
11:end for
12:E top←TOPK(VE
t, sE
t, kE)
▷Stage 3: Fact Retrieval
13:VF
t,e←S
e∈E topGETFACTS(e)
14:foreachvF
t,e∈ VF
t,edo
15:sF
t,e←RRF(BM25(q, vF
t,e),COS(q,h′
vF
t,e))
16:end for
17:F top←TOPK(VF
t,e, sF
t,e, kF)
18:returnR ←COMPOSE(E top,Ftop)

Episode Detection
You are an episodic memory boundary detection expert. Determine if the newly added dialogue should
end the current episode and start a new one.
Input:Conversation history: {history} Time gap info: {time_gap} New messages:
{new_messages}
Decision Criteria:
1.Substantive Topic Change(Highest Priority): Do new messages introduce a completely different
substantive topic? Is there a shift from one specific event to another distinct event?
2.Intent and Purpose Transition: Has the fundamental purpose of the conversation changed
significantly? Has the core question been fully resolved and a new substantial topic begun?
3.Temporal Signals: Significant time gap between messages (hours or days)? Long gaps strongly
suggest new episodes.
4.Structural Signals: Clear concluding statements followed by genuinely new topics? Explicit
topic transition phrases?
Special Rules:Greetings + Topic = ONE episode; Ignore social formalities and pleasantries; Closures
(“Thanks!”, “Take care!”) stay with current episode.
Output: {should_end: bool, should_wait: bool, confidence: float, topic_summary:
str}
Figure 6: Prompt template of episode boundary detection.
Topic Aggregation
You are an expert in identifying whether Episodes describe the SAME situation/event/theme. Your task:
identify which historical Episodes describe the SAME situation as the new Episode.
Input:New Episode: {new_episode} Historical Episodes: {history_episodes} Existing
Topics:{existing_topics}
Same Situation Criteria(ALL must be met):
1.Same Specific Event/Theme: E.g., “Jon’s career transition” at different stages. NOT just related
topics—“Jon’s business” and “Gina’s business” are DIFFERENT situations.
2.Narrative Continuity: Later Episode continues/develops the earlier event. E.g., “Started X” →
“X encountered problem”→“X succeeded” = SAME situation.
3.Identity of Core Subject: Same specific person’s journey, same specific project/initiative, same
specific relationship. NOT just same people or same topic category.
4.Temporal Tolerance: Same situation CAN span multiple time points (weeks or months). Look
for recurring discussions or multi-stage developments across time.
Aggregation Cases: CE=∅ ⇒ Create new Topic; CE̸=∅,CT=∅ ⇒ Aggregate into new Topic;
CT̸=∅ ⇒Update existing Topic.
Output:{title: str, summary: str, keywords: list, episode_weights: dict}
Figure 7: Prompt templates of topic aggregation.

Fact Extraction
You are an expert in extracting queryable facts from Episodes within a Topic context. Extract atomic,
structured facts designed to directly surface answerable evidence.
Input:Topic:{topic}Episodes in this Topic:{episodes}
Extraction Guidelines:
1.Answerable Facts: Focus on facts that directly answer queries, not narrative context. Each Fact
should be a standalone, queryable assertion.
2.Provenance: Maintain explicit links to source Episodes for traceability. Every Fact is anchored to
the Episodes from which it originates.
3.Query Anticipation: Predict potential queries this fact can answer. Store query patterns in the
potentialfield for proactive retrieval alignment.
4.Importance Weights: Assign salience scores w∈[0,1] based on relevance to the Topic, reflecting
each Fact’s contribution.
Output:{content: str, potential: str, keywords: list, importance_weight: float}
Figure 8: Prompt templates of fact extraction.
Conversation & Evidences :
[2022 -06-03] Maria : "That's a great lesson to pass on to your kids, John. Both are really important for strong relationships. Any 
plans to give another pet a loving home?"
[2022 -06-03] John : "We're considering adopting a rescue dog -for love and to teach our kids responsibility and compassion."
[2022 -06-03] Maria : "John, that's such a great idea! It gives the pup a loving home and teaches your kids important values. If 
you need any help, let me know! I just started volunteering at a local dog shelter once a month. "
Query:  What new activity did Maria start recently, as mentioned on 3 June, 2023?
Golden Answer:  volunteering at a local dog shelter once a month  | Answer:
[GraphRAG] 
As of 3 June, 2023, Maria has 
recently started volunteering 
at a homeless shelter …[HyperGraphRAG] 
Maria started a new activity called aerial yoga
recently, as she mentioned at 11:01 am on 17 
December, 2022…
[HyperMem] 
As of June 3, 2023, Maria had recently started volunteering once a month at a local dog shelter , which she mentioned to John as a new 
rewarding ongoing activity.[Mem0] 
Maria joined a gym .[MemOS] 
Joined a gym last week.
Figure 9: Single-Hop Task. HyperMem precisely retrieves “dog shelter” while GraphRAG confuses it with
“homeless shelter.”

Conversation & Evidences:
[2022 -01-21] Nate : "… I won my first video game tournament last week -so exciting!"
[2022 -05-02] Nate : "… Last week I won my second tournament !"
[2022 -06-15] Nate : "… I just won another regional video game tournament last week! …"
[2022 -07-10] Nate : "… I won my fourth video game tournament on Friday! …"
[2022 -08-22] Nate : "Woah Joanna, I won an international tournament yesterday! …"
[2022 -10-06] Nate : "… I won a really big video game tournament last week and it was awesome! …"
[2022 -11-07] Nate : "… I was in the final of a big Valorant tournament last Saturday, and I won! …"
Query:  How many tournaments has Nate won? |  Golden Answer:  7  | Answer:
[HyperMem] 
Nate has won a total of seven video game tournaments in 2022: 1. The first tournament during theweek of January 14 -20, 2022 (Counter -
Strike: Global Offensive). 2. The second tournament on April 25, 2022 (local Street Fighter competition). 3. The third tournamen t 
during the week of May 27toJune 2,2022 (regional tournament). 4. The fourth tournament on July 8, 2022 (online tournament).
5. The fifth tournament on August 21, 2022 (international gaming tournament). 6. The sixth tournament on September 29, 
2022 (significant tournament with prize money). 7. The seventh tournament on November 5, 2022 (Valorant tournament final).[GraphRAG] 
Nate has won at least two regional video game 
tournaments as of late May 2022 -his first…[HyperGraphRAG] 
Nate has won at least five video game 
tournaments, including…[Mem0] 
Four tournaments[MemOS] 
Nate has won five
tournaments.Figure 10: Multi-Hop Task. HyperMem aggregates all 7 tournament mentions across 10 months via Topic
hyperedges, while baselines fragment evidence.
Conversation & Evidences:
[2023-07-11]Andrew : "Hey! So much has changed since last time we talked -meet Toby, my puppy …"
[2022 -07-11] Audrey : "OMG! Toby looks so adorable! Congrats on your new addition… "
[2022 -10-19] Andrew : "Hey Andrew, hope you're doing ok. I recently had a good… "
[2022 -10-19]Audrey : "Hi Audrey! … I've got some awesome news --I recently adopted another pup from a shelter… "
Query:  How many pets did Andrew have, as of September 2023 ?|  Golden Answer:  one  | Answer:
[GraphRAG] 
As of September 2023, Andrew did not have any pets. Audrey, his 
acquaintance, had four dogs: Pepper, Precious, Panda, and Pixie.[HyperGraphRAG] 
As of September 2023, Andrew had four pets (fur babies), 
including his dog Toby.
[HyperMem] 
As of September 2023, Andrew had onepet dog named Toby.[Mem0] 
Two dogs.[MemOS] 
Three pets (Toby, Buddy, Scout). Explanation: As of September 6, 2023, Andrew had Toby 
and Buddy and had recently adopted Scout before November 22, 2023...
Figure 11: Temporal Reasoning Task. HyperMem correctly identifies one pet at the queried time point, while
baselines confuse subjects or miscount.

Conversation & Evidences:
[2023-02-25] John : "Hey Maria! Wanted to let you know that I'm running for office again… "
[2023-07-17] John : "Hey Maria… I visited a veteran's hospital and met some amazing people. It made me appreciate what we 
have and the need to give back ."
[2023 -07-17] Maria : "Wow, John… Did you learn anything cool during your visit? "
[2023 -07-17] John : "I heard some cool stories from an elderly veteran named Samuel. It was inspiring and heartbreaking, but 
seeing their resilience really filled me with hope. It reminded me why I wanted to join the military ."
Query:  Would John be open to moving to another country? |  Golden Answer:  No, he has goals specifically in the U.S. like joining 
the military and running for office. |  Answer:
[GraphRAG] 
…He appreciates experiencing new cultures and environments but has not explicitly stated any openness 
to permanently moving to another country …
[HyperMem] 
John is not indicated to be open to moving to another country. His activities, expressed commitments, and travel plans focus on serving his 
country, improving his local community, and maintaining a strong family presence…[Mem0] 
No explicit mention of 
moving abroad.
[MemOS] 
No, John prefers community 
engagement locally.[HyperGraphRAG] 
Yes, John would be open to moving to another country... Although he maintains a commitment to serving 
his current community and country, his excitement about exploring new cultures and planning trips 
suggests he would consider relocating if the opportunity fits his aspirations and values.Figure 12: Open Domain Task. HyperMem infers John’s commitment to U.S.-based goals, while HyperGraphRAG
incorrectly concludes he would relocate.