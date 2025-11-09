# RAGBoost: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse

**Authors**: Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng Deng, Xuan Sun, Luo Mai

**Published**: 2025-11-05 13:59:01

**PDF URL**: [http://arxiv.org/pdf/2511.03475v1](http://arxiv.org/pdf/2511.03475v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs)
with retrieved context but often suffers from downgraded prefill performance as
modern applications demand longer and more complex inputs. Existing caching
techniques either preserve accuracy with low cache reuse or improve reuse at
the cost of degraded reasoning quality. We present RAGBoost, an efficient RAG
system that achieves high cache reuse without sacrificing accuracy through
accuracy-preserving context reuse. RAGBoost detects overlapping retrieved items
across concurrent sessions and multi-turn interactions, using efficient context
indexing, ordering, and de-duplication to maximize reuse, while lightweight
contextual hints maintain reasoning fidelity. It integrates seamlessly with
existing LLM inference engines and improves their prefill performance by 1.5-3X
over state-of-the-art methods, while preserving or even enhancing reasoning
accuracy across diverse RAG and agentic AI workloads. Our code is released at:
https://github.com/Edinburgh-AgenticAI/RAGBoost.

## Full Text


<!-- PDF content starts -->

RAGBOOST: EFFICIENTRETRIEVAL-AUGMENTEDGENERATION
WITHACCURACY-PRESERVINGCONTEXTREUSE
Yinsicheng Jiang* 1Yeqi Huang* 1Liang Cheng1Cheng Deng1Xuan Sun1Luo Mai1
ABSTRACT
Retrieval-augmented generation (RAG) enhances large language models (LLMs) with retrieved context but
often suffers from downgraded prefill performance as modern applications demand longer and more complex
inputs. Existing caching techniques either preserve accuracy with low cache reuse or improve reuse at the cost
of degraded reasoning quality. We present RAGBOOST, an efficient RAG system that achieves high cache
reuse without sacrificing accuracy throughaccuracy-preserving context reuse. RAGBOOSTdetects overlapping
retrieved items across concurrent sessions and multi-turn interactions, using efficient context indexing, ordering,
and de-duplication to maximize reuse, while lightweight contextual hints maintain reasoning fidelity. It integrates
seamlessly with existing LLM inference engines and improves their prefill performance by 1.5–3 ×over state-
of-the-art methods, while preserving or even enhancing reasoning accuracy across diverse RAG and agentic AI
workloads. Our code is released at:https://github.com/Edinburgh-AgenticAI/RAGBoost.
1 INTRODUCTION
Retrieval-augmented generation (RAG) has become a core
technique for enhancing AI applications, both online (e.g.,
AI search and agentic systems) and offline (e.g., data syn-
thesis and deep research). A typical RAG system comprises
a retrieval module (e.g., FAISS, Qdrant, ElasticSearch) that
retrieves the top- Kmost relevant documents (docs) for a
user query, and an LLM inference engine (e.g., SGLang,
vLLM, TensorRT-LLM) that consumes these documents
as context. During theprefillphase, the engine generates
key–value (KV) caches, which are subsequently reused in
thedecodephase to produce output tokens sequentially. The
primary performance objective of prefilling is to minimize
the time-to-first-token (TTFT). To this end, modern infer-
ence engines employ aprefix cachethat stores KV caches
from prior prompts, reducing computation for repeated or
prefix-sharing inputs.
As RAG evolves toward more knowledge-intensive applica-
tions, workloads increasingly demand longer contexts with
larger and more numerous retrieved documents, substan-
tially amplifying prefill latency. This trend is driven by two
factors: (1) studies show that retrieving hundreds of docu-
ments improves reasoning on complex tasks (e.g., lemmas
*Equal contribution1University of Edinburgh, United Kingdom.
Correspondence to: Yinsicheng Jiang <ysc.jiang@ed.ac.uk >,
Yeqi Huang <yeqi.huang@ed.ac.uk >, Liang Cheng
<L.cheng@ed.ac.uk >, Cheng Deng <cdeng@ed.ac.uk >, Xuan
Sun<xuan.sun@ed.ac.uk>, Luo Mai<luo.mai@ed.ac.uk>.
Preliminary workin AI4Math) (Varambally et al., 2025), enhances up-to-date
knowledge access (e.g., AI4Search) (Alzubi et al., 2025;
Zilliz, 2025), and mitigates hallucination (Ayala & Bechard,
2024; Shuster et al., 2021; AboulEla et al., 2025); and (2)
while document chunking reduces document size, recent
findings (Rajasekaran et al., 2025) indicate it can harm ac-
curacy, and that processing entire documents yields better
results, further increasing prefill load.
To accelerate RAG inference, existing systems adopt two
main caching strategies, yet each faces a trade-off between
reuse efficiency and model accuracy. The first,exact prefix
matching, used in systems such as RadixCache (Zheng et al.,
2024), LMCache (Cheng et al., 2025), and RAGCache (Jin
et al., 2024b), reuses cached KV states only when a new
prompt exactly matches a previous prefix. This approach
preserves accuracy but yields low cache-hit ratios in practice,
as RAG workloads often retrieve large sets of documents in
varying orders, leaving most KV caches unused. The second
category,approximate KV-cache matching, exemplified by
CacheBlend (Yao et al., 2025) and PromptCache (Gim et al.,
2024), matches KV caches by floating-point similarity rather
than exact prefixes. While this increases reuse and shortens
TTFT, we observe in production and evaluation that it can
significantly degrade model accuracy.
To reduce TTFT for long-context RAG inputs without sacri-
ficing reasoning accuracy, we propose a new approach based
on the observation that real-world RAG workloads often
exhibit overlapping retrieved documents and such overlaps
commonly occur (i) across multiple turns within the same
conversation and (ii) among parallel sessions (e.g., promptsarXiv:2511.03475v1  [cs.LG]  5 Nov 2025

2
or user queries) in domain-specific applications. Lever-
aging this observation, we identify three opportunities for
accuracy-preserving context reuse: (1)Orderingretrieved
documents to align prefixes with previously cached contexts,
improving cache-hit ratios; (2)De-duplicatingdocuments
to avoid recomputation for already cached contexts; and
(3)Adding contextual hintsto inform the model of orig-
inal retrieval order and deduplicated document locations,
preserving reasoning quality.
In this paper, we present RAGBOOST, an efficient RAG
system that realizesaccuracy-preserving context reuse. Our
key contributions are summarized below.
(1) Context Indexing.We design an indexing mechanism
that efficiently tracks cached contexts across parallel ses-
sions and multi-turn conversations. The index enables fast
lookup of previously stored contexts by aligning prefix over-
laps and traversing contexts involved in multi-turn histories,
while maintaining low construction and storage overhead.
(2) Context Ordering.We propose a context-ordering al-
gorithm that queries the index to arrange documents both
within and across sessions, maximizing cache hit rates. To
prevent reasoning degradation from reordering, we intro-
duce conciseorder hintsthat convey the original retrieval
order to the LLM, preserving answer quality.
(3) Context De-Duplication.We further enhance reuse
efficiency through context de-duplication. By querying the
index, RAGBoost identifies documents overlapping with
cached contexts and replaces duplicates withlocation hints
that direct the LLM to their original occurrences, avoiding
redundant prefill while maintaining reasoning accuracy.
Extensive evaluations demonstrate the strong performance
of RAGBOOSTacross diverse baselines and real-world
datasets. Across a wide range of RAG workloads, including
multi-turn, multi-session, hybrid, and emerging Chain-of-
Agents applications, RAGBOOSTeffectively reuses con-
texts to boost prefill performance, outperforming state-of-
the-art systems such as CacheBlend, LMCache, Radix-
Cache, and RAGCache by 1.5–3× on datasets Multiho-
pRAG, NarrativeQA, QASPER, and MT-RAG, without any
accuracy loss. In challenging multi-hop reasoning tasks,
RAGBOOSTeven improves accuracy, driven by its con-
textual hint design. Encouraged by these results, we plan
broader production deployment and willopen-sourceand
evolve RAGBOOSTas a foundation for advancing context
engineering, management, and optimization.
Multiple sessions
Retrieval ModuleItem KItem1Item KItem 1Item KDoc 1Doc K…PrefixCachePrefillDecodeMulti-turn for each single session
System Prompt[Doc 1][Doc 2][Doc 3][Doc 2][Doc 3][Doc 1]Turn N Answers (1…M);Turn N+1 Prompts (1…M)Inference EngineContexts
KeyValue...Figure 1.Overview of a RAG system.
2 BACKGROUND ANDMOTIVATION
2.1 Retrieval-augmented generation systems
RAG systems are now integral to both online, latency-
sensitive services, such as semantic search, dialogue, and
deep research (Zilliz, 2025; Guo et al., 2024), and offline,
throughput-oriented pipelines for large-scale annotation and
synthetic data generation (Shen et al., 2025; Zhou et al.,
2024; NVIDIA, 2024; Zhang et al., 2025b). Online RAG
supports advanced reasoning tasks like multi-document sum-
marization, literature review, and code generation, while
offline RAG produces pretraining and fine-tuning data. To-
gether, retrievers and generators form a self-reinforcing loop:
better retrieval improves relevance, and stronger generation
enhances factuality and reasoning.
A typical RAG system ( Figure 1) alternates between re-
trieval and generation across sessions and dialogue turns.
At each turn, the retriever processes Mconcurrent prompts
and returns Krelevant documents per session to form the
context. The inference engine performsprefillto encode
the context anddecodeto generate responses, which feed
into the next retrieval step with updated dialogue history,
enabling efficient multi-turn reasoning.
RAG systems often use prefix caching to improve prefill
efficiency. A trie-based implementation (Zheng et al., 2024)
organizes tokens hierarchically, with each node storing a
token sequence and its KV cache, enabling longest-prefix
matching through a single traversal. An alternative hash-
table design (Kwon et al., 2023) directly maps complete
prefixes to KV-block identifiers.
2.2 Emerging challenge: longer context retrieval
RAG faces a critical prefill latency bottleneck. Modern
LLMs with expanding context windows for two reasons:
(i)increasing the number of retrieved documentsto broaden
information coverage (Li et al., 2024; Jin et al., 2024a),
and (2) enriching contextual information byretrieving com-
plete documentsand applying context engineering meth-
ods (Rajasekaran et al., 2025).
Analysis of our workload data reveals that both approaches

3
deliver significant accuracy gains. Scaling the retrieval pa-
rameter ( k) from lower to higher values enhances RAG ac-
curacy by as much as 20%, while retrieving full documents
achieves similar performance improvements, confirmed by
recent context engineering studies (Zhang et al., 2025a).
However, expanded context windows (i.e., longer retrieved
inputs) introduce substantial prefill overhead. Our trace
data shows that LLM inference engines often process 20k–
130k prefill tokens, leading to 3–10 second latency when
executing 32B dense models on a single H100 GPU. For
larger models such as Mixture-of-Experts (MoEs), the prefill
latency can be even higher. As a result, the prefill becomes
the dominant bottleneck, downgrading user experience and
preventing long-context RAG from being widely deployed.
2.3 Issues of existing KV cache reuse methods
To address the growing cost of longer retrieved contexts,
existing KV-cache reuse methods exhibit several issues:
Exact-prefix matching yields low KV-cache reuse.Exist-
ing prefix-caching mechanisms rely heavily on exact token-
level matching, e.g., RadixCache (Zheng et al., 2024), or
document-level matching, e.g., LMCache (Cheng et al.,
2025) and RAGCache (Jin et al., 2024b): even minor
variations, such as whitespace differences or slightly re-
ordered tokens and documents, prevent reuse. Our evalua-
tion shows that despite substantial overlap in retrieved docu-
ments across related queries, cache hit rates remain persis-
tently low. For example, for the dataset multihopRAG with
Qwen3-32B, the KV-cache hit rate is only 4.6%, indicating
low KV cache reuse. For NarrativeQA with Llama3.3-70B,
the hit rate is also only 5.5%, leaving most cache unused.
Approximate KV-cache matching degrades quality.To
improve low cache-hit ratios, recent techniques such as
CacheBlend (Yao et al., 2025) adopt approximate KV-cache
matching. Instead of exact-prefix matching, they measure
similarity in KV values (floating-point vectors) and reuse
cached states when the proximity exceeds an empirically
decided threshold. However, KV-value similarity isnota
reliable indicator of whether cached states can be reused
across different reasoning contexts and requests. Approxi-
mate matching degrades reasoning quality, with errors com-
pounding over multi-turn interaction. Our evaluations show
that across multiple models (e.g., Qwen3-32B, Qwen3-4B,
Llama3.3-70B) and datasets (e.g., MultihopRAG, Narra-
tiveQA, QASPER), approximate matching can degrade rea-
soning quality by 9–11% (dropping from around 60% to
approximately 50%), preventing its deployment in many
services where reasoning fidelity is necessary.
“Where was John Kennedy born?”“When did John Kennedy die?”“Where was John Kennedy assassinated?”[Doc1]Kennedy’s birthplace; [Doc2] Kennedy’s death date; [Doc3] Location of Kennedy’s assassination;[Doc2] Kennedy’s death date; [Doc1]Kennedy’s birthplace; [Doc3] Location of Kennedy’s assassination[Doc3] Location of Kennedy’s assassination;[Doc2] Kennedy’s death date; [Doc1]Kennedy’s birthplaceTop 3Top 3Top 3User 1:User 2:User 3:(a) Multi-session overlap.
“When did John Kennedydie?”“I see…, one more question: Where was John Kennedy assassinated?”[Doc3] Kennedy’s death date; [Doc2] location of Kennedy’s assassination;[Doc1]Kennedy’s birthplace;“November 22, 1963”[Doc2] location of Kennedy’s assassination;[Doc3] Kennedy’s death date; [Doc1]Kennedy’s birthplace;“Dallas, Texas”Top 3User 1:
User 1: (b) Multi-turn overlap.
C= [1, 2, 3]Original Prompt:System Prompt: You are a helpful assistant that answers questions using your own knowledge and relevant external information when available.User: The context information is below[item 1] …[item 2] …[item 3] …Given the context information, answer the question:{Question}C= [2, 3, 1]OrderedOrderedPrompt:System Prompt: You are a helpful assistant that answers questions using your own knowledge and relevant external information when available.User: The context information is below[item 2] …[item 3] …[item 1] …Given the context information, answer the question:{Question}NOTE: Please read the context in the following priority order: [item 1] > [item 2] > [item 3] to answer the question.Assistant: <think>The user requests an answer based on the following priority: [item 1], [item 2], and [item 3]. [item 1] discusses … [item 2] covers … [item 3] addresses … </think>Based on this order, the answer is: {answer}Assistant: <think>The user requests an answer based on given contexts [item 1], [item 2], and [item 3].[item 1] is … [item 2] describes… [item 3] says … </think>Based on this order, the answer is: {answer}
(c) Contextual hint for recovering retrieval order semantics.
Figure 2. Context overlap and context reuse opportunities in RAG.
3 DESIGNOVERVIEW
3.1 Observation: significant overlap in retrieval
Our design is motivated by a key observation: real-world
RAG workloads exhibit substantial overlap in retrieved doc-
uments across both sessions and conversation turns:
(1) Overlap across sessions.Figure 2a illustrates over-
lapping retrievals among multiple users querying different
aspects of the same person. Although the retrieved doc-
uments appear in different orders reflecting per-query rel-
evance, their content largely coincides. Trace studies on
MultihopRAG (Tang & Yang, 2024), NarrativeQA (Ko ˇcisk´y
et al., 2017), and QASPER (Dasigi et al., 2021) confirm this
trend: 79.2%, 57.4%, and 49.6% of questions respectively
draw from the top 20% most frequently accessed documents,
indicating extensive context sharing across sessions.
(2) Overlap within multi-turn conversations.Figure 2b
shows that in multi-turn interactions, users often revisit
related topics, causing the retriever to return the same doc-
uments with slightly different rankings. As previous turns
become part of the input, later retrievals frequently dupli-
cate content already present in the cached history. Our
MT-RAG (Katsis et al., 2025) trace study quantifies this
effect: on average, 40% of retrieved documents in any turn
overlap with earlier ones in the same session.

4
3.2 Design opportunities for context reuse
The significant overlap among retrieved documents reveals
clear opportunities forcontext reuse, boosting KV-cache
hit rates. Specifically, we identify three opportunities that
commonly arise in real-world RAG applications:
(1) Ordering retrieved documents across sessions boosts
KV-cache reuse.As shown in Figure 2a, if the retrieved
documents for the second and third users are reordered to
match the first user’s sequence, all three contexts would
share an identical prefix, achieving 100% KV-cache reuse.
Trace-based reordering experiments on MultihopRAG, Nar-
rativeQA, and QASPER confirm this potential. Aligning
document order with prefix-cache structure raises KV-cache
hit rates to 38.9%, 20.2%, and 16.5%, respectively, repre-
senting 3–8 ×higher utilization than the baseline. Thus,
strategic document reordering can dramatically cut redun-
dant prefill computation across users.
Crucially, such reordering incurs minor accuracy loss: only
0.1–3.3% on the same datasets. The small degradation arises
because prefix-optimized orderings can occasionally move
important documents toward the middle of the list, exposing
them to the lost-in-the-middle effect (Liu et al., 2023). We
later discuss strategies to fully recover this minor loss.
Note that the context ordering poses no additional privacy
or security risks, sharing the same guarantees as prior KV-
cache reuse methods (e.g., RadixCache).
(2) De-duplicating multi-turn overlaps reduces prefill
cost.Figure 2b shows that multi-turn retrievals often re-
turn overlapping documents across conversation turns. By
deduplicating these documents and processing only new
content together with dialogue history, the amount of con-
textual data during prefill can be greatly reduced, lowering
computation cost.
Our MT-RAG trace study quantifies this benefit and shows
that de-duplication causes only 1–3% accuracy degradation,
which can be recovered with techniques discussed later.
This minor loss occurs because the LLM can still access
the deduplicated content through prior conversation history,
preserving quality while avoiding duplicated computation.
(3) Contextual hints preserve accuracy by restoring or-
dering and de-duplication semantics.Carefully designed
contextual hints can recover the minor accuracy loss caused
by document ordering and de-duplication. As shown in Fig-
ure 2c, these hints help the model reconstruct the original
ordering relationships within its internal reasoning tokens,
preserving reasoning fidelity.
Evaluations show that contextual hints fully restore reason-
ing quality—and in some cases, even improve it. On Narra-
tiveQA and MultihopRAG, for instance, accuracy increases
Retrieval SystemUser prompts
Inference EngineOutputUser prompts + Updated contextsContext IndexContext OrderingContext DeduplicationUser prompts + Contexts  Search matched contextTraverse current contextUpdate cache statusRAGBoostFigure 3.System Overview of RAGBoost.
by 0.3–3.9%, confirming the effectiveness of incorporating
contextual hints for enhanced reasoning.
Note that the contextual hint does not affect the model’s
instruction-following ability, as it only restores minimal
retrieval information without altering the user prompt.
3.3 RAGBoost system overview
RAGBOOSTrealizes the three design opportunities above,
achievingaccuracy-preserving context reuse. It features a
clean, minimal interface compatible with common retrieval
modules (e.g., FAISS and ElasticSearch) and inference en-
gines (e.g., SGLang and vLLM), enabling rapid and broad
deployment. Specifically, RAGBOOSTtakes user prompts
and their retrieved results, updates the context to enable
effective reuse, and then passes the updated context to the
inference engine for processing.
Figure 3 illustrates the key components in RAGBOOST.
Thecontext indexcontinuously tracks KV-cache generation
and eviction within the prefix cache, recording which KV
caches are stored and exposing efficient search and traversal
operations to facilitate context reuse. Building on this in-
dex, thecontext ordering mechanismreorders documents to
align prefixes and schedules prefix-sharing contexts consec-
utively to maximize cache hits, whileorder hintspreserve
the original retrieval ranking to maintain accuracy. Finally,
thecontext de-duplication mechanismremoves redundant
documents in multi-turn conversations. Guided by the index,
it detects cross-turn duplicates and injectslocation hintsdi-
recting the LLM to prior content, eliminating redundant
prefilling while preserving reasoning quality. The following
sections describe each component in detail.
4 CONTEXTINDEX
The context index is designed to: (1) efficiently track the in-
ference engine’s prefix-cache status to enable KV-cache
reuse; (2) support fast lookup of previously stored KV
caches via prefix matching, enabling cross-session context

5
4023C3 {4,1,0}C4 {1,2} C5 {1} 𝑑!"𝑑#$𝑑!#C2 {2,6,1} 𝑑"#C1 {2,1,3}16∅
Context Index TreePrefix Cache (Radix Tree)
Figure 4.Context index construction with prefix-cache semantics.
reuse when overlaps exist; and (3) traverse KV caches in
multi-turn conversations to detect duplicated context.
4.1 Key designs for context index
Figure 4 illustrates the structure of the context index with
an example. The left panel shows the index tree, and the
right panel shows the corresponding prefix-cache status.
The index is organized as a tree whose root represents an
empty context. Each node corresponds to a prefix stored in
the prefix cache and contains child nodes that extend this
prefix. Every node maintains four attributes: (1) the context
containing retrieved document IDs, (2) the search path from
the root to this node, (3) the current sequence length, and
(4) a multi-turn status flag.
Novelty of this context index.Unlike existing prefix-
matching systems (e.g., SGLang-LPM) that abort on order
mismatches, our context index enables three capabilities: (1)
discovering overlapping documents across different order-
ings for reuse, (2) maintaining visibility into cached prefix
paths through the tree structure, and (3) determining optimal
reorderings to maximize prefix matches. These capabilities
unlock cache hits that prior work cannot achieve.
Index creation.The index is built via hierarchical cluster-
ing based on prefix matching. First, we compute pairwise
distances between all contexts using their overlap rate. Next,
we iteratively merge the closest pair, creating a virtual node
whose context is the sorted intersection representing their
shared prefix. Finally, each leaf node records its search path
from the root, enabling efficient traversal for both cross-
session prefix matching and multi-turn duplicate detection.
As shown in Figure 4, the process begins with C1 {2,1,3} ,
C2{2,6,1} , and C3 {4,1,0} as leaf nodes. Since C1 and
C2 have the smallest distance (sharing {1,2} ), they merge
first into a virtual node C4 with context {1,2} . C3 then
merges with C4 to form the root C5 with context {1}. The
resulting tree has C1–C3 as leaves storing their search paths
from C5, while C4 and C5 serve as virtual nodes represent-ing shared prefixes for cache reuse.
This construction runs in O(N2)time, where Nis the num-
ber of contexts, and is fully parallelizable on CPUs and
GPUs. In practice, building the index for 2,000 contexts
takes 8 s on CPUs and 0.82 s on GPUs. The space com-
plexity is O(N·K) , where Kis the average number of
retrieved documents per context. Because the index stores
only document IDs and metadata rather than full texts, its
space overhead is minimal.
Quantifying the overlapping between contexts.A key
challenge in index construction is quantifying the overlap
between contexts. We propose acontext distance function
that satisfies two requirements: (1) it captures the number
of shared documents between contexts, and (2) it accounts
for their positional alignment, since retrieval systems rank
documents by query relevance.
To illustrate the need for this design, consider four con-
texts: A {3,5,1,7} , B{2,6,3,5} , C{3,5,8,9} , and D
{2,6,4,0} . A naive overlap-only metric assigns identical
distances (0.5) to pairs A–B, B–C, and B–D because each
shares two documents. However, B and D share {2,6} at
positions 1–2, while A and B share {3,5} at different posi-
tions. Our distance function (Equation 1) assigns a smaller
distance to B–D, as their overlaps occur in similar positions,
reflecting both overlap magnitude and positional alignment.
Such patterns cannot be captured by conventional distance
measures like cosine, L1, or L2 similarity, which ignore
positional structure. More formally, our distance function is
defined as:
dij= 1−|Sij|
max(|C i|,|Cj|)+α·P
k∈S ij|pi(k)−p j(k)|
|Sij|(1)
where Sijdenotes the set of shared docs, pi(k)is the po-
sition of doc kin context i, and α∈[0.001,0.01] ensures
overlap count remains the dominant factor while incorporat-
ing positional alignment.
Index update.Whenever the inference engine updates
its prefix cache, it notifies RAGBoost to update the index
accordingly. The update process first maintains a min-heap
tracking all active nodes by last access time. If an eviction
event occurs, the specified number of tokens are removed
from the least recently used nodes by decrementing their
token counts. Newly generated output tokens are then added
to the corresponding node’s context length. When a node’s
token count reaches zero, a traversal operator (detailed later)
locates the node via its search path and removes it from
its parent’s child list, recursively deleting empty parents.
The overall update cost is O(h) , where his the tree height,
as only a single traversal is required, making the update
extremely fast in practice.

6
C3 {4,1,0}C4 {1,2} C5 {1} C2 {2,6,1} C1{2,1,3}∅Pick parent nodePick parent nodePick parent nodeC1{1,2,3}ReorderC2 {1,2,6}C3 {1,4,0}ReorderReorderContexts used to initialize the treeNew contextsC6 {2,1,4}C7 {5,7,8}C8 {1,9,2}C6C8Best matchC4 {1,2}C5 {1} ∅
C1 {1,2,3}C2 {1,2,6}C3 {1,4,0}C6{2,1,4}C8{1,9,2}C7{5,7,8}No match!ReorderC6{1,2,4}ReorderC8{1,2,9}Append to best-match childrenC6 {1,2,4}C7 {5,7,8}C8 {1,2,9}Ordered contextsSearchOutput(1)(2)(3)
Figure 5.Context ordering: (1) find best-match nodes; (2) reorder by shared prefix and append as child; (3) output the ordered context.
4.2 Key operations with context index
The context index provides two key operations:
Context search.RAGBOOSTfrequently searches for pre-
viously stored contexts based on the current one to enable
reuse. The index search algorithm efficiently locates match-
ing contexts by greedily descending from the root, selecting
at each level the child with the minimum distance while
recording positions to form a search path. The search stops
upon reaching a leaf or when all children are equidistant,
indicating the longest shared prefix. Updates are localized
and efficient: matching an internal node appends the new
context as a child ( O(1) ), while matching a leaf creates a
new internal node with their intersection ( O(|C|) ). Unlike
K-Means re-clustering or HNSW graph rebuilding, these up-
dates require no tree restructuring, enabling dynamic index
maintenance with minimal overhead.
For example, given context C6 {2,1,4} , we search the index
in Figure 4. C6 first compares with the root’s child C5 and
finds a shared prefix {1}, descending to C5 and recording
its position [0]. At C5, C6 shares {1,2} with C4 but only
{1}with C3, so it selects C4 and appends another [0], yield-
ing [0,0]. At C4’s children C1 {1,2,3} and C2 {1,2,6}
(reordered per Section 5), all have equal distance, so the
search stops and identifies C4 as the best match with path
[0,0]. C6 is then inserted into C4’s children list at position
2, forming the final search path [0,0,2].
Search complexity scales with tree height. For contexts
with common prefixes, h=O(logn) yields O(|C| ·logn)
complexity, where ndenotes the number of stored contexts.
Empirically, this takes approximately 18 µs, negligible com-
pared to prefill latency.
Context traversal.In multi-turn conversations, RAG-
BOOSTupdates node context lengths by traversing the index
using the stored search path. Starting from the root, it se-
quentially follows indices along the path until reaching the
target node, then performs the update. Traversal costs O(h)
and completes in around 5µs.5 CONTEXTORDERING
The context ordering mechanism aims to: (1) reorder re-
trieved contexts according to the current prefix-cache sta-
tus to maximize KV-cache reuse; (2) schedule the ordered
contexts to the inference engine with awareness of cache
generation and eviction policies to enhance hit rates; and
(3) insert concise contextual hints that recover pre-ordering
semantics and maintain reasoning quality.
5.1 Context ordering algorithm
Formally, the context ordering algorithm takes a batch of
RAG requests with their retrieved contexts as input, reorders
them based on prefix matches from the context index, and
returns ordered contexts with maximized shared prefixes.
As illustrated in Figure 5, we begin with initialization con-
texts C1 {2,1,3} , C2{2,6,1} , and C3 {4,1,0} , followed
by new contexts C6 {2,1,4} , C7{5,7,8} , and C8 {1,2,9} .
Initialization contexts inherit prefixes from their parent
nodes (C1, C2 from C4 with {1,2} ; C3 from C5 with {1}),
while new contexts search the index (C6 and C8 match
C4 and inherit {1,2} ). Each context then concatenates its
matched prefix with remaining documents in their original
order, producing C1 → {1,2,3} , C2→ {1,2,6} , C6→
{1,2,4} , and C8 → {1,2,9} . Unmatched contexts (e.g.,
C7) remain unchanged and form standalone branches. This
strategy ensures overlapping contexts share common pre-
fixes while preserving the ranking of non-shared documents.
The algorithm is invoked whenever RAGBOOSTprocesses
a new request. It runs in O(|C| ·logn) time, where nis the
number of stored contexts—negligible compared to prefill.
5.2 Scheduling requests with ordered contexts
After ordering contexts, RAGBOOSTmust schedule their
execution to align with the inference engine’s KV-cache
generation and eviction policies; otherwise, cache reuse
becomes ineffective. We therefore design a scheduling al-
gorithm that: (1) reuses the search paths obtained during

7
Final contextsC6 {1,2,4}C7 {5,7,8}C8 {1,2,9}Path [0,0,2]Path [1]Path [0,0,3]Group by first element of the search pathC3 {1,4,0}Path [0,1]C6 {1,2,4}C7 {5,7,8}C8 {1,2,9}Path [0,0,2]Path [1]Path [0,0,3]C3 {1,4,0}Path [0,1]Group 0Group 1Sort each group by path length (longest first).C6 {1,2,4}C7 {5,7,8}C8 {1,2,9}Path [0,0,2]Path [1]Path [0,0,3]C3 {1,4,0}Path [0,1]Ordered contexts
Figure 6.Example of scheduling requests with ordered contexts.
context ordering to avoid redundant tree lookups; (2) groups
contexts by the first element of their search path, naturally
separating cache regions; and (3) sorts contexts within each
group by path length in descending order, ensuring longer
prefix matches execute before shorter ones.
Figure 6 illustrates this process. In the baseline order C6, C3,
C7, C8, limited cache capacity allows only one context: C6
caches {1,2,4} , but C3 reuses only {1}and evicts {2,4} .
C7 causes a full miss, caching {5,7,8} and evicting all
previous entries, which then forces another miss for C8
despite its shared prefix {1,2} with C6. This inefficiency
arises because contexts with shared prefixes are not executed
consecutively.
Our scheduler reorders execution to C6, C8, C3, C7,
grouping prefix-sharing contexts together. C6 first caches
{1,2,4} , then C8 immediately reuses {1,2} before evic-
tion. C3 and C7 run afterward without disrupting this reuse,
maximizing cache hit rates.
Our scheduler performs O(N) grouping by root-prefix
path and O(NlogN) in-group sorting over Ncontexts,
with negligible real-time overhead. In contrast, exist-
ing indexing methods such as RAGCache and SGLang’s
LPM use aglobal prefix selection that rescans a radix
tree with Mnodes at each decision point, yielding
O(NlogM) +O(NlogN) overall as cache state evolves.
By draining groups sequentially, our methodavoids re-
peated tree searches, better preserves reuse under tight KV
budgets, and keeps complexity independent ofM.
5.3 Contextual hints for retrieval order
We provide the LLM with concise hints indicating the origi-
nal retrieval order of documents. Reordering contexts alters
the retrieval system’s original ranking, which encodes docu-
ment relevance critical for answer quality. Consider context
C6, where the retriever returns documents in order {2,1,4} .
The baseline prompt is:
[system prompt] →[Doc 2]→[Doc 1]→[Doc 4]→[question]
After reordering to {1,2,4} for cache efficiency, we append
anorder hintbefore the question:
[system prompt]→[Doc 1]→[Doc 2]→[Doc 4]→
[order hint]→[question]The hint explicitly specifies the original retrieval priority:
“Please read the context in the following priority order:
[Doc 2]>[Doc 1]>[Doc 4] and answer the question. ”
This short instruction adds negligible token overhead during
prefill yet effectively preserves the model’s ability to attend
to documents by their original relevance ranking. As a
result, RAGBOOSTachieves aggressive cache optimization
without compromising reasoning accuracy.
6 CONTEXTDEDUPLICATION
The context de-duplication mechanism has two goals: (1)
eliminate contexts that share an exact prefix with previ-
ously processed ones whose KV caches are already stored,
thereby minimizing redundant prefill computation; and (2)
provide hints informing the LLM which content has been de-
duplicated and where the corresponding information resides
in the earlier context.
Context de-duplication algorithm.We illustrate the con-
text de-duplication algorithm with an example. Consider a
session where user C6 initially retrieves context {1,2,4} in
the first turn, which is stored in the index tree. In the second
turn, after the multi-turn flag is activated, a new retrieval
yields{1,5,2} . To remove duplicates, C6 follows its stored
search path to the first-turn context {1,2,4} and identifies
{1,2} as overlapping documents. These are filtered out,
leaving only the novel document {5}to be processed. The
deduplicated document is appended to a copy of the first-
turn context state for future comparisons, and the cumulative
context length is updated accordingly. The algorithm runs
inO(N)time and incurs negligible overhead.
Contextual hints for de-duplicated contexts.Simply re-
moving duplicates can degrade answer quality. To maintain
quality, we insertlocation hintsthat direct the LLM to cor-
responding documents in the conversation history.
For C6’s second turn, the baseline prompt is:
[first turn context]→[first turn Q&A]→[Doc 1]→
[Doc 5]→[Doc 2]→[second turn question]
With de-duplication and location hints, the prompt becomes:
[first turn context]→[first turn Q&A]→[hint 1]→
[Doc 5]→[hint 2]→[second turn question]
Here,hint 1andhint 2are references such as“Please refer
to [Doc 1] in the previous conversation”and“Please refer
to [Doc 2] in the previous conversation. ”These hints guide
the LLM to prior context without repeating prefill.

8
Table 1.Multi-session RAG results: F1 (%) and prefill throughput for four methods across three models on three datasets.
LMCache CacheBlend Radix CacheRAGBoost (Ours)
Dataset ModelF1 Prefill Throughput F1 Prefill Throughput F1 Prefill Throughput F1 Prefill Throughput
MultihopRAGQwen3-4B-Instruct-2507 46.1 34710.9 39.8 85405.3 46.1 58990.150.2 106799.5
Qwen3-32B 60.4 14708.6 50.1 36128.6 60.4 17682.664.2 36296.1
Llama3.3-70B-Instruct 64.6 11596.4 54.9 14134.2 64.6 14777.166.6 30046.7
NarrativeQAQwen3-4B-Instruct-2507 18.8 39276.4 11.3 42819.5 18.8 39492.422.7 57681.3
Qwen3-32B 31.8 15514.0 19.8 16913.2 31.8 15598.832.1 22780.4
Llama3.3-70B-Instruct 40.3 12575.7 31.3 13710.5 40.3 12644.940.8 18468.8
QASPERQwen3-4B-Instruct-250734.929349.2 25.9 36273.534.933034.6 34.846619.9
Qwen3-32B41.115568.4 29.3 20238.941.117523.0 40.924733.4
Llama3.3-70B-Instruct 44.3 12000.1 35.7 14829.7 44.3 13507.344.6 19061.0
7 EVALUATION
Our evaluation of RAGBOOSTshows: (1) RAGBOOSTim-
proves prefill throughput by 1.2-3 ×and reasoning accuracy
by 0.3-4.1% over numerous state-of-the-art systems and
methods cross multi-turn, multi-session, and hybrid RAG
workloads, (2) It outperforms strong baselines by 1.5-3 ×
in throughput and 1.9-3.7% in accuracy in emerging agen-
tic AI applications, (3) Each component (context ordering,
de-duplication, and hint design) yields clear gains and ro-
bustness with negligible overhead, and (4) Benefits scale
with longer contexts and larger retrieval sizes.
Evaluation setup.Our RAGBOOSTimplementation sup-
ports SGLang 0.4.6 and vLLM 0.10.0 with only about 10
lines of modification each. The changes are confined to
their cache eviction handlers, where we insert context index
updates, making the changes easy to upstream and merge.
We compare RAGBOOSTagainst the following baselines:
(i) LMCACHE(version 0.3.8), representing the state of the
art in prompt caching; (ii) CACHEBLEND, the state of the
art in KV-cache matching, integrated with LMCache; and
(iii) RADIXCACHE, based on SGLang’s implementation
with a Longest-Prefix-Match scheduling policy.
We omit additional baselines that achieve comparable perfor-
mance to those above. (iv) HICACHE(Xie et al., 2025) ex-
tends RadixCache by expanding prefix caches to lower-tier
memory; since it directly builds on RadixCache, we com-
pare against RadixCache instead. (v) RAGCACHEadopts
a similar radix-tree structure at document granularity and
shows comparable performance to RadixCache. It is not
open-sourced and thus excluded from our evaluation.
Our evaluation is conducted on two GPU clusters: (1) a
16× H100 GPU cluster and (2) a 12× A6000 GPU cluster.
For all baselines, we tune system parameters for optimal
performance and accuracy, aligning our configurations with
the best results reported in their respective papers.7.1 Application performance: RAG and Agentic AI
We evaluate on four RAG datasets: QASPER (Dasigi
et al., 2021), MultihopRAG (Tang & Yang, 2024), Nar-
rativeQA (Ko ˇcisk´y et al., 2017), and MT-RAG (Katsis et al.,
2025). For QASPER, MultihopRAG, and NarrativeQA, we
use a chunk size of 1024 following (Bhat et al., 2025), while
MT-RAG performs document-level retrieval without chunk-
ing. We use gte-Qwen2-7B-Instruct as the embed-
ding model, FAISS for similarity search on MultihopRAG
and NarrativeQA, and BM25 for QASPER and MT-RAG.
This setup demonstrates RAGBOOST’s effectiveness across
diverse retrieval paradigms.
We further test RAGBOOSTunder three RAG work-
loads: (1) multi-session, where independent users query
concurrently; (2) multi-turn, for extended single-user
dialogues; and (3) combined multi-session and multi-
turn, representing production-scale conversational sys-
tems.Multi-session RAG.We evaluate multi-session
RAG on QASPER, MultihopRAG, and NarrativeQA using
three models—Qwen3-4B-Instruct-2507, Qwen3-32B, and
Llama3.3-70B-Instruct—on H100 GPUs with top-k=15.
Table 1 reports F1 scores and prefill throughput. RAG-
BOOSTdelivers up to 3.08×, 2.05×, and 2.25× speedups
over LMCache, RadixCache, and CacheBlend on Multiho-
pRAG, and 1.18–1.59× gains on NarrativeQA and QASPER.
These gains arise from context reordering that maximizes
prefix overlap, improving cache hits and reducing redundant
computation. In contrast, LMCache and RadixCache de-
pend on exact prefix matching, causing recomputation even
for overlapping content, while LMCache also incurs high
CPU offloading costs for long contexts.
RAGBOOSTmaintains accuracy via order hints that pre-
serve original retrieval rankings. CacheBlend, however,
degrades sharply—F1 drops to 11.3 on NarrativeQA with
Qwen3-4B versus 22.7 for RAGBOOST—due to approxi-
mate KV matching and selective recomputation disrupting
coherence. Notably, RAGBOOSTcan even improve accu-
racy (e.g., 60.4 →64.2 on MultihopRAG with Qwen3-32B)

9
Table 2. Performance metrics across different tasks: (a) MT-RAG results, (b) Hybrid RAG sessions, (c) Context index construction latency.
Method MetricQwen3-
4BLlama3.1-
8BQwen3-
30B
LMCacheAcc. 62.5668.4675.12
TTFT 0.76 1.04 1.08
CacheBlendAcc. 50.33 56.52 X
TTFT 0.30 0.48 X
RadixCacheAcc. 62.5668.4675.12
TTFT 0.44 0.49 0.61
RAGBoostAcc.64.2768.1275.81
TTFT0.22 0.31 0.35
(a) Accuracy (%) and time-to-first-token
(TTFT)(s) on MT-RAG for four methods
across three models. X = not supported.# Sessions
Method 2 4 8 16 32
LMCache 0.81 0.87 0.94 1.19 1.72
CacheBlend 0.40 0.42 0.45 0.54 0.78
RadixCache 0.46 0.49 0.53 0.67 0.97
RAGBoost 0.24 0.29 0.34 0.41 0.65
(b) Time-to-first-token (seconds) perfor-
mance in seconds for Qwen3-4B-Instruct-
2507 model evaluated under hybrid RAG
workloads with varying concurrent session
counts ranging from 2 to 32 sessions.# Contexts(N ctx)
k128 512 4k 8k 12k 100k
3 0.64 0.65 1.51 3.54 7.48 868.92
5 0.66 0.66 1.55 3.58 7.55 867.83
10 0.67 0.68 1.59 3.63 7.63 869.23
15 0.69 0.69 1.62 3.67 7.69 866.49
20 0.71 0.72 1.66 3.72 7.78 869.98
(c) Context index construction latency (sec-
onds) as a function of the number of contexts
Nctxand top- k. Columns labeled128–100k
denote the total contexts inserted at build
time (N ctx;1k=1,000).
as order hints help models prioritize key documents during
prefill, yielding richer contextual reasoning.
Multi-turn RAG.We evaluate multi-turn RAG on the MT-
RAG dataset using Qwen3-4B-Instruct-2507, Llama3.1-8B-
Instruct, and Qwen3-30B-A3B-Thinking-2507 on a single
H100 GPU. Models with longer context windows are used
to handle the growing conversation history. Answer ac-
curacy is measured via the LLM-as-a-judge method from
RADBench (Kuo et al., 2025) with GPT-5, as recommended
by MT-RAG.
Table 2a reports accuracy and time-to-first-token (TTFT).
RAGBOOSTcuts TTFT by removing redundant document
processing across turns through context de-duplication.
It achieves 3.45×, 3.35×, and 3.09× speedups over LM-
Cache on Qwen3-4B, Llama3.1-8B, and Qwen3-30B, re-
spectively, and up to 2.00× over RadixCache and 1.55× over
CacheBlend. These gains arise from detecting and skipping
duplicated contexts that baselines repeatedly recompute.
RAGBOOSTalso preserves accuracy via location hints that
direct models to previously seen documents. CacheBlend,
by contrast, drops to 50.33% on Qwen3-4B versus 64.27%
for RAGBOOST, as its approximate KV matching and se-
lective recomputation disrupt multi-turn coherence. RAG-
BOOSTeven improves accuracy in some cases (e.g., from
62.56% to 64.27% on Qwen3-4B) by maintaining contex-
tual continuity while avoiding redundant computation.
Multi-session, multi-turn RAG.We evaluate the combined
multi-session and multi-turn scenario under real-world de-
ployment using Qwen3-4B-Instruct-2507 on H100 GPUs,
varying concurrency from 2 to 32 sessions.
Table 2b shows that RAGBOOSTachieves the lowest TTFT
at all concurrency levels by ordering contexts to max-
imize prefix overlap. At 2 sessions, it delivers 3.38×,
1.92×, and 1.67× speedups over LMCache, RadixCache,
and CacheBlend, respectively; at 32 sessions, the gains
remain substantial at 2.65×, 1.49×, and 1.20×. These consis-
tent improvements demonstrate that RAGBOOST’s contextordering effectively maintains prefix overlap as concurrent,
diverse queries scale, while baselines suffer from cache
thrashing and redundant recomputation.
Agentic AI applications with RAG.We evaluate RAG-
BOOSTwithin the agentic AI frameworkChain-of-Agent
(CoA)(Zhang et al., 2024), which mitigates long-context
limitations by coordinating multiple specialized agents
through natural-language communication. Each worker
agent handles a document segment for localized reason-
ing, while a manager agent aggregates their intermediate
results into a final answer.
RAGBOOSTenhances CoA through agent-aware routing.
In multi-session settings, when documents appear in multi-
ple queries, they are routed to the agent that processed them
before, enabling KV-cache reuse. In multi-turn conversa-
tions, repeated documents are deduplicated, and location
hints direct agents to prior content, eliminating redundant
prefilling while preserving CoA’s distributed reasoning.
We deploy three CoA configurations on MultihopRAG, each
with 15 agents using Llama3.1-8B, Llama3.2-3B, or Qwen3-
4B-Instruct, where each agent processes one retrieved doc-
ument ( k=15 ). With Qwen3-4B, accuracy increases from
48.3%to50.2%and throughput by1.8×; with Llama3.1-8B,
accuracy rises from50.7%to54.4%with a2.1×speedup.
These results show that RAGBOOSTreduces redundant
processing and boosts both accuracy and throughput via
effective KV-cache reuse across agents.
7.2 Performance breakdown, overhead and robustness
Performance breakdown.We analyze the impact of
RAGBOOST’s context ordering components—orderingand
scheduling—on multi-session RAG. Experiments are con-
ducted on MultihopRAG ( k=15 ) using two inference en-
gines (SGLang, vLLM) and two models (Qwen3-32B,
Llama3.3-70B-Instruct) on H100 GPUs.
Figure 7 shows incremental gains in cache hit rate. For
SGLang with Qwen3-32B, the baseline hit rate (8.49%) rises

10
Vanilla +Order
-ing+Schedul
-ing0.07.014.021.028.035.0Cache hit rate (%)SGLang
Llama3.3-70B
Qwen3-32B
(a) RAGBOOSTwith SGLang.
Vanilla +Order
-ing+Schedul
-ing0.09.018.027.036.045.0Cache hit rate (%)vLLM
Llama3.3-70B
Qwen3-32B (b) RAGBOOSTwith vLLM.
Figure 7.Performance breakdown of key components.
3 5 10 15
k010000Prefill TP (tokens/s)MultihopRAG
3 5 10 15
kNarrativeQARadix Cache LMCache CacheBlend Ours
Figure 8.Prefill throughput under different top k values.
to 20.56% with ordering and 33.97% with scheduling—a
4× improvement. For vLLM with Llama3.3-70B, results are
similar: 10.7% →30.8%→43.2%. These gains directly
translate to reduced prefill computation and lower TTFT.
System overhead.We previously showed that index search
and update operations complete within 15 µs. Here, we eval-
uate index construction overhead as context count scales
from 128 to 100K under varying retrieval top- kvalues on
NVIDIA A6000 GPUs. Smaller counts (128–512) repre-
sent online deployments, larger ones (up to 12K) offline
workloads, and 100K an extreme stress test.
Table 2c shows that construction scales smoothly with
context count. At 12K contexts, it completes in
7.48 s—negligible relative to total prefill latency. Even at
100K, RAGBOOSTfinishes within 15 minutes on GPU, far
faster than multi-hour offline prefilling, and can be further
accelerated via multi-GPU indexing.
Construction time remains stable across retrieval depths
(k= 5 –20), showing minimal sensitivity to retrieval size.
Overall, RAGBOOST’s index construction is a lightweight,
one-time cost amortized across many queries, practical for
large-scale RAG deployments.
Worst-case analysis.We evaluate a synthetic RAG work-
load with no retrieval overlap. RAGBOOSTadds only 0.72 s
of prefill latency for 1K contexts (one-hour job), showing
negligible overhead.
Impact of context length.We evaluate RAGBOOSTunder
varying retrieval depths ( k=3,5,10,15 ) on NarrativeQA
and MultihopRAG using A6000 GPUs to assess scalabilityas context length grows. Figure 8 shows that RAGBOOST
consistently achieves the highest prefill throughput across all
kvalues. On MultihopRAG, it sustains 1.5–2.0× speedups
over baselines as kincreases from 3 to 15; on NarrativeQA,
it maintains 1.3–1.6× gains. While baseline throughput de-
clines with larger kdue to reduced cache reuse, RAGBOOST
remains stable—its context ordering preserves prefix over-
lap even with longer retrieved contexts.
Hint robustness.We tested multiple positions (before and
after questions) and observed negligible impact on accuracy,
with variations below 0.5%. Attention heatmap analysis
further confirms that the hints effectively guide the model
to focus on relevant documents, improving downstream
accuracy. We include this analysis in the supplementary
materials (Appendix A) due to page limit.
Impact of cache size and cache optimality.We assess
cache size effects using two GPU clusters with different
memory capacities: A6000 (48 GB) and H100 (96 GB).
On MultihopRAG with Qwen3-32B and Llama3.3-70B
across SGLang and vLLM, larger memory yields clear
scaling benefits. From A6000 to H100, vanilla SGLang’s
hit rate rises from 4.62% to 8.49%, and vLLM’s from
4.20% to 11.2%. With RAGBOOST, improvements are
far greater—SGLang from 29.64% to 33.97%, and vLLM
from 35.90% to 43.4%—showing that larger cache capacity
further amplifies ordering and scheduling benefits.
For offline RAG, context ordering achieves near-optimal
cache reuse since all prompts are known in advance. In
online RAG, where only 64–512 prompts are visible per
batch, hit rates remain within 3–5% of optimal and improve
as GPU memory scales.
8 RELATEDWORKS
RAG system optimization.System-level approaches such
as METIS (Ray et al., 2025) and Chameleon (Jiang et al.,
2024) optimize workflow and hardware efficiency, jointly
tuning retrieval settings or using heterogeneous accelera-
tors to reduce latency and boost throughput. RAGBOOST
complements them by improving KV-cache reuse.
Reranking in retrieval systems.Rerankers refine retrieval
results via learned ranking models (Adeyemi et al., 2024;
Li et al., 2023; Zhang et al., 2025c); HyperRAG (An et al.,
2025) further enables KV reuse at the reranking stage. RAG-
BOOSTinstead operates downstream, ordering reranked
document IDs to maximize prefix overlap while preserving
relevance through order hints.
Fine-tuning with positional re-encoding.Methods like
BlockAttention, KVLink, and TurboRAG (Ma et al., 2025;
Yang et al., 2025a; Lu & Tang, 2024) fine-tune models to

11
reuse KV caches via position re-encoding and pre-stored
states. They improve efficiency but require heavy training
and large cache storage. RAGBOOSTis training-free and
shows strong promise in numerous RAG applications.
Faster KV-cache compute.CacheBlend (Yao et al., 2025)
and related works (Liu et al., 2024; Agarwal et al., 2025;
Yang et al., 2025b; Deng et al., 2025; Liu et al., 2025) opti-
mize KV-cache computation through compression, parallel
encoding, or cache sharing. RAGBOOSTcomplements
these by operating at the context level, ordering and de-
duplicating inputs to maximize reuse, and can be combined
with them for further gains.
9 CONCLUSION
This paper presents RAGBOOST, an efficient context reuse
system that achieves faster prefill performance without com-
promising accuracy. Its design opens new directions for
system research, including efficient context indexing, or-
dering and de-duplication, contextual hints that preserve
reasoning fidelity, while ensuring broad compatibility with
existing methods and systems. We view RAGBOOSTas
an important step toward efficient, long-context RAG and
agentic AI. With its open-source, modular design, RAG-
BOOSTprovides a foundation for future advances in context
engineering, management, and optimization.
REFERENCES
AboulEla, S., Zabihitari, P., Ibrahim, N., Afshar, M., and
Kashef, R. Exploring rag solutions to reduce halluci-
nations in llms. In2025 IEEE International systems
Conference (SysCon), pp. 1–8, 2025. doi: 10.1109/
SysCon64521.2025.11014810.
Adeyemi, M., Oladipo, A., Pradeep, R., and Lin, J. Zero-
shot cross-lingual reranking with large language mod-
els for low-resource languages. In Ku, L.-W., Mar-
tins, A., and Srikumar, V . (eds.),Proceedings of the
62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 2: Short Papers), pp. 650–656,
Bangkok, Thailand, August 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.acl-short.
59. URL https://aclanthology.org/2024.
acl-short.59/.
Agarwal, S., Sundaresan, S., Mitra, S., Mahapatra, D.,
Gupta, A., Sharma, R., Kapu, N. J., Yu, T., and Saini,
S. Cache-craft: Managing chunk-caches for efficient
retrieval-augmented generation.Proc. ACM Manag.
Data, 3(3), June 2025. doi: 10.1145/3725273. URL
https://doi.org/10.1145/3725273.
Alzubi, S., Brooks, C., Chiniya, P., Contente, E., von Ger-
lach, C., Irwin, L., Jiang, Y ., Kaz, A., Nguyen, W., Oh, S.,Tyagi, H., and Viswanath, P. Open deep search: Democ-
ratizing search with open-source reasoning agents, 2025.
URLhttps://arxiv.org/abs/2503.20201.
An, Y ., Cheng, Y ., Park, S. J., and Jiang, J. Hyper-
rag: Enhancing quality-efficiency tradeoffs in retrieval-
augmented generation with reranker kv-cache reuse, 2025.
URLhttps://arxiv.org/abs/2504.02921.
Ayala, O. and Bechard, P. Reducing hallucination in struc-
tured outputs via retrieval-augmented generation. InPro-
ceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 6: Industry
Track), pp. 228–238. Association for Computational Lin-
guistics, 2024. doi: 10.18653/v1/2024.naacl-industry.
19. URL http://dx.doi.org/10.18653/v1/
2024.naacl-industry.19.
Bhat, S. R., Rudat, M., Spiekermann, J., and Flores-Herr,
N. Rethinking chunk size for long-document retrieval: A
multi-dataset analysis, 2025. URL https://arxiv.
org/abs/2505.21700.
Cheng, Y ., Liu, Y ., Yao, J., An, Y ., Chen, X., Feng, S.,
Huang, Y ., Shen, S., Du, K., and Jiang, J. Lmcache:
An efficient kv cache layer for enterprise-scale llm in-
ference, 2025. URL https://arxiv.org/abs/
2510.09665.
Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., and
Gardner, M. A dataset of information-seeking questions
and answers anchored in research papers, 2021. URL
https://arxiv.org/abs/2105.03011.
Deng, Y ., You, Z., Xiang, L., Li, Q., Yuan, P., Hong,
Z., Zheng, Y ., Li, W., Li, R., Liu, H., Mouratidis,
K., Yiu, M. L., Li, H., Shen, Q., Mao, R., and
Tang, B. Alayadb: The data foundation for efficient
and effective long-context llm inference. InCom-
panion of the 2025 International Conference on Man-
agement of Data, SIGMOD/PODS ’25, pp. 364–377,
New York, NY , USA, 2025. Association for Comput-
ing Machinery. ISBN 9798400715648. doi: 10.1145/
3722212.3724428. URL https://doi.org/10.
1145/3722212.3724428.
Gim, I., Chen, G., seob Lee, S., Sarda, N., Khandelwal,
A., and Zhong, L. Prompt cache: Modular attention
reuse for low-latency inference, 2024. URL https:
//arxiv.org/abs/2311.04934.
Guo, S., Deng, C., Wen, Y ., Chen, H., Chang, Y ., and Wang,
J. Ds-agent: Automated data science by empowering
large language models with case-based reasoning.arXiv
preprint arXiv:2402.17453, 2024.

12
Jiang, W., Zeller, M., Waleffe, R., Hoefler, T., and Alonso,
G. Chameleon: a heterogeneous and disaggregated accel-
erator system for retrieval-augmented language models.
Proc. VLDB Endow., 18(1):42–52, 2024.
Jin, B., Yoon, J., Han, J., and Arik, S. O. Long-context
llms meet rag: Overcoming challenges for long inputs
in rag, 2024a. URL https://arxiv.org/abs/
2410.05983.
Jin, C., Zhang, Z., Jiang, X., Liu, F., Liu, X., Liu, X., and Jin,
X. Ragcache: Efficient knowledge caching for retrieval-
augmented generation, 2024b. URL https://arxiv.
org/abs/2404.12457.
Katsis, Y ., Rosenthal, S., Fadnis, K., Gunasekara, C., Lee,
Y .-S., Popa, L., Shah, V ., Zhu, H., Contractor, D., and
Danilevsky, M. Mtrag: A multi-turn conversational
benchmark for evaluating retrieval-augmented generation
systems, 2025. URL https://arxiv.org/abs/
2501.03468.
Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L.
General-purpose in-context learning by meta-learning
transformers.arXiv preprint arXiv:2212.04458, 2022.
Koˇcisk´y, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann,
K. M., Melis, G., and Grefenstette, E. The narrativeqa
reading comprehension challenge, 2017. URL https:
//arxiv.org/abs/1712.07040.
Kuo, T.-L., Liao, F.-T., Hsieh, M.-W., Chang, F.-C., Hsu,
P.-C., and Shiu, D.-S. Rad-bench: Evaluating large
language models capabilities in retrieval augmented di-
alogues, 2025. URL https://arxiv.org/abs/
2409.12558.
Kwon, W., Li, Z., Zhuang, S., Sheng, Y ., Zheng, L., Yu,
C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Ef-
ficient memory management for large language model
serving with pagedattention, 2023. URL https://
arxiv.org/abs/2309.06180.
Li, C., Liu, Z., Xiao, S., and Shao, Y . Making large language
models a better foundation for dense retrieval, 2023.
Li, Z., Li, C., Zhang, M., Mei, Q., and Bendersky, M. Re-
trieval augmented generation or long-context llms? a
comprehensive study and hybrid approach.arXiv preprint
arXiv:2407.16833, 2024.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts, 2023. URL https:
//arxiv.org/abs/2307.03172.
Liu, Y ., Li, H., Cheng, Y ., Ray, S., Huang, Y ., Zhang, Q.,
Du, K., Yao, J., Lu, S., Ananthanarayanan, G., Maire, M.,Hoffmann, H., Holtzman, A., and Jiang, J. Cachegen: Kv
cache compression and streaming for fast large language
model serving, 2024. URL https://arxiv.org/
abs/2310.07240.
Liu, Y ., Huang, Y ., Yao, J., Feng, S., Gu, Z., Du, K., Li,
H., Cheng, Y ., Jiang, J., Lu, S., Musuvathi, M., and
Choukse, E. Droidspeak: Kv cache sharing for cross-
llm communication and multi-llm serving, 2025. URL
https://arxiv.org/abs/2411.02820.
Lu, S. and Tang, Y . Turborag: Accelerating retrieval-
augmented generation with precomputed kv caches for
chunked text.arXiv preprint arXiv:2410.07590, 2024.
Ma, D., Wang, Y ., and Tian, L. Block-attention for efficient
prefilling, 2025. URL https://arxiv.org/abs/
2409.15355.
NVIDIA. How nvidia uses rag to generate synthetic data for
llm improvement. https://developer.nvidia.
com/blog, 2024.
Rajasekaran, P., Dixon, E., Ryan, C., and Hadfield, J. Ef-
fective context engineering for AI agents, 2025. URL
https://www.anthropic.com . Anthropic Engi-
neering blog.
Ray, S., Pan, R., Gu, Z., Du, K., Feng, S., Anantha-
narayanan, G., Netravali, R., and Jiang, J. Metis: Fast
quality-aware rag systems with configuration adapta-
tion. InProceedings of the ACM SIGOPS 31st Sym-
posium on Operating Systems Principles, SOSP ’25,
pp. 606–622, New York, NY , USA, 2025. Associa-
tion for Computing Machinery. ISBN 9798400718700.
doi: 10.1145/3731569.3764855. URL https://doi.
org/10.1145/3731569.3764855.
Shen, H., Yan, H., Xing, Z., Liu, M., Li, Y ., Chen, Z., Wang,
Y ., Wang, J., and Ma, Y . Ragsynth: Synthetic data for
robust and faithful rag component optimization, 2025.
URLhttps://arxiv.org/abs/2505.10989.
Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J.
Retrieval augmentation reduces hallucination in conversa-
tion, 2021. URL https://arxiv.org/abs/2104.
07567.
Tang, Y . and Yang, Y . Multihop-rag: Benchmarking
retrieval-augmented generation for multi-hop queries,
2024. URL https://arxiv.org/abs/2401.
15391.
Varambally, S., V oice, T., Sun, Y ., Chen, Z., Yu, R., and Ye,
K. Hilbert: Recursively building formal proofs with in-
formal reasoning, 2025. URL https://arxiv.org/
abs/2509.22819.

13
Xie, Z., Xu, Z., Zhao, M., An, Y ., Mailthody, V . S., Mahlke,
S., Garland, M., and Kozyrakis, C. Strata: Hierarchical
context caching for long context language model serv-
ing, 2025. URL https://arxiv.org/abs/2508.
18572.
Yang, J., Hou, B., Wei, W., Bao, Y ., and Chang, S. Kvlink:
Accelerating large language models via efficient kv cache
reuse, 2025a. URL https://arxiv.org/abs/
2502.16002.
Yang, X., Chen, T., and Chen, B. Ape: Faster and longer
context-augmented generation via adaptive parallel en-
coding, 2025b. URL https://arxiv.org/abs/
2502.05431.
Yao, J., Li, H., Liu, Y ., Ray, S., Cheng, Y ., Zhang, Q.,
Du, K., Lu, S., and Jiang, J. Cacheblend: Fast large
language model serving for rag with cached knowl-
edge fusion. InProceedings of the Twentieth Euro-
pean Conference on Computer Systems, pp. 94–109,
2025. doi: 10.1145/3689031.3696098. URL https:
//doi.org/10.1145/3689031.3696098.
Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru,
V ., Rainton, J., Wu, C., Ji, M., Li, H., Thakker, U., Zou,
J., and Olukotun, K. Agentic context engineering: Evolv-
ing contexts for self-improving language models, 2025a.
URLhttps://arxiv.org/abs/2510.04618.
Zhang, X., Zhang, P., Luo, S., Tang, J., Wan, Y ., Yang, B.,
and Huang, F. Culturesynth: A hierarchical taxonomy-
guided and retrieval-augmented framework for cultural
question-answer synthesis, 2025b. URL https://
arxiv.org/abs/2509.10886.
Zhang, Y ., Sun, R., Chen, Y ., Pfister, T., Zhang, R., and Arik,
S. Chain of agents: Large language models collaborating
on long-context tasks.Advances in Neural Information
Processing Systems, 37:132208–132237, 2024.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang, B.,
Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou,
J. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176, 2025c.
Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H.,
Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Bar-
rett, C., and Sheng, Y . Sglang: Efficient execution
of structured language model programs, 2024. URL
https://arxiv.org/abs/2312.07104.
Zhou, J., Liu, Z., Liu, Z., Xiao, S., Wang, Y ., Zhao, B.,
Zhang, C. J., Lian, D., and Xiong, Y . Megapairs: Massive
data synthesis for universal multimodal retrieval, 2024.
URLhttps://arxiv.org/abs/2412.14475.Zilliz. Deepsearcher: Open-source deep research on pri-
vate data. GitHub repository, 2025. URL https:
//github.com/zilliztech/deep-searcher.

14
A HINT ROBUSTNESS ANALYSIS WITH
ATTENTION MAP
Since context engineering (Rajasekaran et al., 2025) and
in-context learning (Kirsch et al., 2022) strongly influence
model inference, we analyze attention patterns when explicit
hints are introduced to recall the original retrieval order.
Figure 9 and Figure 10 compares the final-layer attention
maps of Qwen and LLaMA under this setup. When given
explicit document-priority cues, both models exhibit con-
sistent attention behaviors despite architectural differences.
They correctly focus on document tokens ([Doc 1], [Doc 2],
[Doc 3]), reflecting awareness of the mismatch between
the reordered and original sequences, as indicated by inter-
sections between queries in the hint region and keys in the
context region. As the context re-aligns with the original
sequence, both models emphasize ([Doc 2]) while parsing
([Doc 1]) and ([Doc 3]), showing that the cue ([Doc 2]>
[Doc 1]>[Doc 3]) effectively directs cross-document at-
tention. Hence, explicit hints reshape internal attention,
aligning it with semantic rather than positional priority.
This finding supports a central hypothesis of RAGBoost:
explicit hints can make a comeback from the accuracy lost
to reordering and filtering by re-establishing alignment with
the original retrieval semantics.
Figure 9. Attention map of the last layer attention of LLaMA (Head
14) for the prompt “The retrieved documents are [Doc 1] ABCD
[Doc 2] EFGH [Doc 3] IJKL. Please read the context in the fol-
lowing priority order: [Doc 2]>[Doc 1]>[Doc 3]. Where is
E?”.
Figure 10. Attention map of the last layer attention of Qwen (Head
9) for the prompt “The retrieved documents are [Doc 1] ABCD
[Doc 2] EFGH [Doc 3] IJKL. Please read the context in the fol-
lowing priority order: [Doc 2]>[Doc 1]>[Doc 3]. Where is
E?”.