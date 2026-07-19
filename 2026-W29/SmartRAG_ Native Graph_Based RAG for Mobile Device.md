# SmartRAG: Native Graph-Based RAG for Mobile Device

**Authors**: Zhihan Jiang, Meng Li, Shenghao Liu, Keran Li, Ruiben Zhou, Xianjun Deng, Shuai Wang, Haipeng Dai

**Published**: 2026-07-16 07:28:31

**PDF URL**: [https://arxiv.org/pdf/2607.14661v1](https://arxiv.org/pdf/2607.14661v1)

## Abstract
Deploying large language models (LLMs) as personal assistants on mobile devices demands privacy, low latency, and offline availability, yet the computational cost of giant models clashes with strict edge-hardware budgets. We argue that this tension cannot be resolved by model compression alone; it requires decomposing on-device intelligence into complementary functional roles. We present SmartRAG, a fully on-device framework that organizes an intelligent assistant around four coordinated modules -- Perception, Memory, Focus, and Thinking. At the core of SmartRAG is EvoNER, a continually learnable named-entity recognizer that incrementally expands its label inventory through teacher-distilled updates, enabling the system to absorb previously unseen entity types without retraining the backbone LLM. Extracted knowledge is stored in MRGraph, a three-layer provenance-preserving knowledge graph, and retrieved at query time through a hybrid pipeline combining graph traversal, lexical matching, and dense semantic search. The on-device LLM is invoked only for high-value semantic operations -- labeling, planning, and answer synthesis -- keeping inference costs bounded. Experiments on four QA benchmarks (TriviaQA, Natural Questions, HotpotQA, MultiHopQA) show that SmartRAG with a quantized 1.7B-parameter backbone achieves multi-hop reasoning performance competitive with models up to 18$\times$ larger, while running entirely on commodity smartphones within practical memory and latency envelopes.

## Full Text


<!-- PDF content starts -->

SmartRAG: Native Graph-Based RAG for Mobile
Device
Zhihan Jiang1, Meng Li1, Shenghao Liu2, Keran Li1, Ruiben Zhou1
Xianjun Deng2, Shuai Wang3, Haipeng Dai1
1Nanjing University2HUST3Southeast University
zhihan_jiang@smail.nju.edu.cn,meng@nju.edu.cn
Abstract
Deploying large language models (LLMs) as personal assistants on mobile devices
demands privacy, low latency, and offline availability, yet the computational cost of
giant models clashes with strict edge-hardware budgets. We argue that this tension
cannot be resolved by model compression alone; it requires decomposing on-device
intelligence into complementary functional roles. We presentSmartRAG, a fully
on-device framework that organizes an intelligent assistant around four coordinated
modules—Perception,Memory,Focus, andThinking. At the core of SmartRAG
isEvoNER, a continually learnable named-entity recognizer that incrementally
expands its label inventory through teacher-distilled updates, enabling the system
to absorb previously unseen entity types without retraining the backbone LLM.
Extracted knowledge is stored inMRGraph, a three-layer provenance-preserving
knowledge graph, and retrieved at query time through a hybrid pipeline combining
graph traversal, lexical matching, and dense semantic search. The on-device LLM
is invoked only for high-value semantic operations—labeling, planning, and answer
synthesis—keeping inference costs bounded. Experiments on four QA benchmarks
(TriviaQA, Natural Questions, HotpotQA, MultiHopQA) show that SmartRAG with
a quantized 1.7B-parameter backbone achieves multi-hop reasoning performance
competitive with models up to 18 ×larger, while running entirely on commodity
smartphones within practical memory and latency envelopes.
1 Introduction
In recent years, as the application of large language models (LLMs) in real-world interactive tasks
continues to deepen, memory is gradually emerging as a critical capability in LLM system determining
whether the system can support long-term personalization, continuous knowledge updates, and
contextually consistent reasoning. This issue is of particular significance for on-device scenarios, as
the core value of such deployments lies precisely in data locality, privacy protection, and personalized
services capabilities that inherently require the system to establish and maintain a local memory
built around the user’s private data, device-local content, and environmental context. Given the strict
constraints on system-memory (i.e., RAM usage and runtime memory footprint on edge devices),
power consumption, and latency budgets inherent to devices, heavyweight LLMs are difficult to
deploy directly; furthermore, relying solely on the limited parametric memory of lightweight models
often proves insufficient to support continuous updates, user customization, and knowledge-intensive
reasoning. Retrieval-Augmented Generation (RAG) addresses this challenge by integrating LLMs
with external, non-parametric memory, thereby providing a memory mechanism that is updatable,
structured, and more easily personalized; consequently, it is widely regarded as a highly promising
solution.
arXiv:2607.14661v1  [cs.AI]  16 Jul 2026

Although on-device RAG has recently achieved preliminary progress, most existing methods remain
centered on naive vector retrieval—that is, relying on semantic similarity between queries and
document snippets to facilitate memory access. While this paradigm is well-suited for handling
problems involving local matching, it struggles to address complex queries that require cross-segment
evidence aggregation, entity linking, and multi-step reasoning. To address this limitation, on-cloud
RAG research has begun to leverage graph structures to enhance the retrieval process, explicitly
modeling the connections between entities, relations, and evidence to improve multi-hop question-
answering capabilities. However, the majority of existing graph-based RAG methods are built upon
powerful models, relying heavily on models with tens of billions of parameters to execute critical
steps such as graph construction, summarization, and relation organization. While this premise may
hold true in cloud-based environments, it clearly fails in on-device scenarios: on-device systems
simply cannot bear the computational, latency, energy, and cost overheads associated with models of
such scale.
Based on this issue, we believe that the key to on-device RAG lies not in directly compressing
cloud-based graph RAGs onto the device, but in rethinking how structured memory should be built
and utilized under resource-constrained conditions. In other words, graph construction capabilities do
not necessarily stem from the generation capabilities of super-powerful models; they can also arise
from more sophisticated system architecture design. By rationally dividing module responsibilities,
controlling the granularity of knowledge organization, and designing incrementally evolving memory
formation mechanisms, lightweight components can also gradually build a structured knowledge
foundation that supports complex reasoning. Unlike traditional graph-based RAG methods, our
approach emphasizes a systematic decoupling of knowledge extraction, memory organization, and
reasoning support, allowing these processes to be collaboratively completed by multiple lightweight
modules under on-device constraints. This provides an implementation path for on-device RAGs that
differs from the cloud paradigm.
However, several key technical challenges remain to be addressed before this approach can be
truly implemented on device. First, due to budget constraints related to system-memory, power
consumption, and latency, only lightweight models are deployable on device. These models are
insufficient in open environments to identify unknown entities, long-tail concepts, and new knowledge,
making it difficult to directly construct high-quality structured knowledge like large cloud-based
models. Therefore, the system must possess the ability to discover, aggregate, and continuously
absorb unseen entities. Furthermore, complex queries often require entity association tracing and
multi-step retrieval across multiple pieces of evidence. However, edge scenarios cannot rely on
powerful models to repeatedly perform query decomposition, path planning, and graph exploration.
This makes achieving structured focus for complex problems within a limited budget another core
challenge. Finally, even after relevant evidence has been retrieved, lightweight generation models
struggle to reliably complete reasoning within lengthy and noisy contexts. Therefore, a controlled
evidence organization method is needed to confine the final generation to a compact and high-value
context.
Based on this understanding, we proposeSmartRAG, a structured RAG framework for on-device
scenarios. Its core objective is to build external memory capable of supporting complex reasoning
under lightweight models and limited resources. Unlike existing methods that rely on powerful models
to directly complete graph construction and graph-based reasoning, SmartRAG decouples knowledge
extraction, memory organization, retrieval focus, and answer generation through modular design,
enabling these processes to operate collaboratively under edge-side constraints. Thus, SmartRAG
transforms the structured knowledge capability from LLMs capability into a system architecture
problem, providing a more practical implementation method for on-device RAGs.
Our contributions are as follows:
•We identify the core feasibility problem of on-device structured RAG: complex reasoning requires
structured and updatable memory, but existing graph-construction pipelines depend too heavily on
powerful LLMs to remain sustainable on edge devices.
•We presentSmartRAG, an on-device structured RAG method that reallocates high-frequency
memory writing to lightweight trainable components while reserving the LLM for sparse, high-
value semantic operations.
2

•We developEvoNER, a continually learnable entity recognizer that absorbs previously unseen
semantic categories through lightweight incremental updates, enabling the system to recognize and
organize emerging concepts without retraining the backbone LLM.
•We integrate provenance-preserving structured memory and hybrid retrieval into a full on-device
pipeline, and demonstrate on real smartphones that this design yields strong end-to-end QA
performance under strict device constraints.
2 Related work
On-device LLMs.Recent advances in quantization [Frantar et al., 2023] and efficient inference en-
gines [Gerganov, 2023] have made it feasible to run billion-parameter models on mobile hardware [Xu
et al., 2024]. Models such as Qwen3 [Qwen Team, 2025] and Llama 3 [Grattafiori et al., 2024] offer
compact variants targeting edge deployment. However, these efforts focus on theinference engine
and do not address thememoryandcontinuous adaptationrequirements of a personal assistant.
Retrieval-augmented generation.RAG [Lewis et al., 2020] augments LLMs with external knowl-
edge at inference time. Dense passage retrieval [Karpukhin et al., 2020] provides the standard
retrieval backbone. Self-RAG [Asai et al., 2024] internalizes retrieval behaviors through specialized
training, and xRAG [Cheng et al., 2024] compresses retrieved context into compact representations.
Graph-based approaches [Edge et al., 2024] organize retrieved knowledge into structured graphs
for improved multi-hop reasoning. Multi-hop QA methods [Trivedi et al., 2023, Press et al., 2023]
interleave retrieval with chain-of-thought reasoning. Most of these systems assume cloud-scale
compute; our work targets their adaptation to strict on-device budgets.
Named entity recognition and continual learning.Span-based NER [Yu et al., 2020, Fu et al.,
2021] formulates entity detection as span classification over enumerated candidates, avoiding the
label dependency of sequential taggers. Continual learning for NER [Monaikul et al., 2021] addresses
the challenge of absorbing new entity types without forgetting old ones, commonly using knowledge
distillation [Hinton et al., 2015] or regularization [Kirkpatrick et al., 2017]. Our EvoNER com-
bines span-based detection with a reserved-label classifier and teacher-distilled incremental updates,
specifically designed for on-device expansion of the entity vocabulary.
Knowledge graphs for LLMs.The integration of knowledge graphs with LLMs [Ji et al., 2022,
Pan et al., 2024] has attracted growing attention. LLM-driven KG construction [Edge et al., 2024]
achieves high-quality extraction but requires large models. Our approach separates the extraction
pipeline into a lightweight, trainable perception layer and a sparse LLM control plane, achieving
predictable ingestion cost on mobile devices.
3 Method
3.1 Architecture overview
The SmartRAG architecture (Figure 1) consists of four interacting modules. ThePerceptionmodule
transforms unstructured text into structured knowledge via EvoNER (an on-device NER component,
Section 3.2) and a lightweight relation classifier, producing entity–relation triplets. TheMemory
module organizes these triplets into MRGraph, a persistent knowledge structure that binds entities and
relations to their source paragraphs for provenance (Section 3.3). At query time, theFocusmodule
retrieves and ranks relevant evidence through a hybrid pipeline combining graph traversal, lexical
matching, and dense semantic search (Section 3.4). TheThinkingmodule invokes the on-device
LLM only for high-value semantic operations: labeling emerging entity types, planning multi-hop
retrieval, and synthesizing final answers (Section 3.5).
3.2 Perception: EvoNER and entity discovery
Span-based NER with character injection.We designEvoNER(EvolutionaryNamedEntity
Recognition) to satisfy four on-device requirements: lightweight co-existence with the LLM, accurate
entity boundary and type prediction, robustness to out-of-vocabulary surface forms, and support for
3

QueryWhat did Freud present regarding the mechanism of the unconscious in "The Interpretation of Dreams"
LLM
HybridRetrieval
AnswerDream thinking shows primary-process logic: irrational, timeless, rational thoughtTriplet Construction
Documents
SLM-NER…Entity Set
SLM-RE
VectorDatabaseMemory Construction
LightweightLLMDecomposer
TripletArticle & Original Graph
Entity-Level Graph=JIN-GraphArticleIndexing
Structured Multi-hop PlanSemantic Triplet
Intermediary information
GraphRetrieval
VectorRetrievalGlobalFallbackRetrievedSubgraphRetrievedTextDocumentsFigure 1: Overview of the SmartRAG architecture. Incoming text flows throughPerception(EvoNER
+ relation extraction) intoMemory(MRGraph), which is queried byFocus(hybrid retrieval) and
consumed byThinking(LLM-driven planning and answer generation). Dashed arrows indicate
asynchronous background processes (clustering, incremental training).
continual label expansion. Given an input sentence, we encode it into sub-word representations and
enumerate candidate spans from word-start positions up to a maximum widthW:
r= (i, j),1≤i≤j≤n,(j−i+ 1)≤W,(1)
where rdenotes a candidate span, iandjare its inclusive word-level start and end indices, nis the
sentence length in words, and Wcaps span width ( W=10 in our implementation), yielding O(nW)
candidates instead of all O(n2)spans. To improve robustness to misspellings and rare surface forms,
we augment the contextual representation with a character-level feature injected at the first subword
of each word:
ht= [e t;mt·c(t)],(2)
where htis the final representation at subword position t,etis the Transformer contextual embedding
att,mt∈ {0,1} is1on the first subword of a word and 0otherwise, c(t) is the character-level
embedding aligned with that word, and [ ; ]denotes concatenation. Each candidate span is represented
using boundary features, element-wise interaction, and a learned width embedding, then passed to a
classifier.
Reserved-label mechanism for continual expansion.A key requirement for continual on-device
learning is adding new entity types without reinitializing the classifier. We pre-allocate a fixed label
capacity and partition the span classifier into abasehead (no-entity label plus deployed entity types)
andreservedrows for future types. For span (i, j) with pooled representation zij∈Rd, logits
4

decompose as
ℓbase
ij
ℓresv
ij
=
Wbase
Wresv
zij+
bbase
bresv
,ℓ ij= [ℓbase
ij;ℓresv
ij],(3)
where Wbase,bbase(resp. Wresv,bresv) are base (resp. reserved) weights and biases. Reserved slots
are initialized inactive ( bresv≪0,Wresv≈0). When new types are discovered, each is mapped to
an unused reserved row without changing output dimensionality.
Incremental learning with teacher distillation.When new entity types emerge, EvoNER must
incorporate the supervision without degrading performance on existing types. We pair the reserved-
label design with teacher-guided distillation on the base-label subspace only. Let Ω(x) be the set of
candidate spans in sentencex. The task loss matches offline training:
Ltask=L span+1
2 
Lstart+L end
,(4)
where Lspan is masked span classification over candidates and Lstart,Lendare auxiliary boundary
losses on mention endpoints. The full incremental objective is
Linc=L task+λKDLKD,(5)
withλKDtrading stability against plasticity. For each span r∈Ω(x) , letℓT,base
r ,ℓS,base
r∈R1+n base
denote teacher and student logitsrestricted to the base subspace, where nbase is the number of
deployed base entity types (the leading dimension accounts for the no-entity logit). With temperature
Tkdand confidence thresholdτ, define the gate
mr=I
max
1≤k≤1+n base
softmax 
ℓT,base
r /Tkd
k≥τ
,(6)
i.e., the gate is on when the teacher’s peak probability on base classes exceeds τ. The distillation loss
is
LKD=T2
kdP
r∈Ω(x)mr+ϵX
r∈Ω(x)mrKL
softmax 
ℓT,base
r /Tkdsoftmax 
ℓS,base
r /Tkd
,(7)
where ϵavoids division by zero. Reserved logits arenotdistilled, so new types remain plastic while
the teacher anchors old boundaries. The teacher is frozen for an entire incremental round, then the
updated student may become the next teacher.
Three-stage parameter-efficient updates.To accommodate the unpredictability of mobile schedul-
ing, we adopt a three-stage optimization schedule with progressively increasing trainable capacity.
LetΘdenote all parameters, Θhead non-encoder parameters (heads and adapters), and ΘtopK the
top-KTransformer blocks (counted from the output side):
Θ1= Θ head,Θ 2= Θ head∪Θ topK,Θ 3= Θ.(8)
Stage 1 adapts only heads; Stage 2 adds partial encoder layers; Stage 3 allows brief full-model
consolidation at a low learning rate. Training may stop after any stage if the device budget is
exhausted.
Triplet discovery.Extracted entities are initially categorized into coarse types. For unseen entities,
we extract stable semantic vectors via a frozen lightweight embedder and store them in a local vector
database. When the count exceeds a threshold, agglomerative clustering with Ward linkage [Ward,
1963] groups semantically similar mentions. For clusters A, B with sizes |A|,|B| and embedding
centroidsµA,µB,
d(A, B) =|A| · |B|
|A|+|B|∥µA−µB∥2
2.(9)
Given threshold ϵ >0 , we iteratively merge clusters: whenever some pair (A, B) satisfies d(A, B)<
ϵ, merge them; stop when no such pair remains (all surviving clusters are pairwise at least ϵ-separated
under d). The Thinking module then assigns category labels to large coherent clusters. For known-
type entities, a lightweight relation classifier identifies relations, producing triplets written into
MRGraph.
5

3.3 Memory: MRGraph
MRGraph is a three-layer on-device knowledge graph designed for continual writes and efficient
retrieval with explicit provenance.
Layer 1: Entity–paragraph graph.The paragraph is the atomic unit for both writing and retrieval.
Each paragraph node stores raw content, an in-document index, a pointer to the source document,
and an embedding vector. Entity nodes are deduplicated via stable name hashing and linked to
paragraphs through association edges. Relations are stored as keyed edges with observation counts to
prevent uncontrolled graph growth. Ingestion follows anevidence-first, structure-secondprinciple:
extracted triplets are first aligned to their best-matching paragraphs, then materialized as relation
edges, ensuring every structured fact remains traceable.
Layer 2: Abstraction layer.Above entities, we maintain semantic clusters whose centroids are
updated with smoothing as new entities arrive. An anchor–centroid drift metric detects and prevents
topic drift. Cluster summaries are maintained incrementally during normal operation, with higher-
quality refinement triggered during idle-and-charging periods.
Layer 3: Runtime raw-text view.At query time, retrieved paragraph evidence and the local graph
neighborhood are assembled into a context bounded by a token budget, providing the Thinking
module with both structured and unstructured evidence.
3.4 Focus: hybrid retrieval
At query time, the Thinking module decomposes the question into a retrieval plan P=
(e0,{rh, th}H
h=1), where e0is the seed entity, His the number of hops, and for each hop
h∈ {1, . . . , H} ,rhis a natural-language relation phrase (e.g., “published in”) and this the desired
target entity type for that hop.
Stage 1: Plan-guided multi-hop text retrieval.For each hop, the system constructs local queries by
combining the current center entity with the relation phrase, retrieves candidate paragraphs using
fused lexical and dense similarity scores, and selects evidence under a token budget via maximal
marginal relevance (MMR). Retrieved paragraphs are used to infer next-hop entity centers.
Stage 2: Graph-structured augmentation.Bridge entities from Stage 1 seed a graph traversal. For
each node, we compare its concatenation with the query against abstraction-layer summaries, select
top-ranked clusters, and perform top-down retrieval to expand the subgraph. Personalized PageR-
ank [Page et al., 1999] estimates node importance, and connected subgraph blocks are assembled
with supporting paragraphs.
Stage 3: Full-document fallback.If the first two stages provide insufficient evidence, global retrieval
over documents provides a last-resort mechanism. Because on-device context length is limited, entities
inevitably split across paragraphs; full-document fallback recovers these cross-boundary references.
Finally, retrieved evidence is post-processed: local context windows around entity mentions are
extracted, sentence-level compression removes redundancy, and hash-based deduplication filters
near-duplicates before packing the final context for the LLM.
3.5 Thinking: LLM-driven semantic operations
The on-device LLM is treated as a reusable semantic engine invoked only for operations requiring
high-level language understanding, through four roles:
Labeling.For entity clusters that cannot be confidently typed, the LLM proposes new type names.
LetMdenote the multiset of mentions awaiting labels; cluster(M)groups them into clusters, and
∆T=LLM 
cluster(M)
(10)
returns a set of new type strings ∆Tfor coherent clusters. Semantically dispersed clusters are rejected
to avoid polluting the type space.
Summarization.The knowledge graph is periodically clustered into communities, and the LLM
produces summaries re-inserted as upper-layer paragraphs, forming a self-compressing memory loop.
Planning.Complex questions are decomposed into multi-hop retrieval plans before evidence
acquisition, reducing search ambiguity under strict on-device budgets.
6

Answering.The LLM generates responses conditioned on retrieved evidence, constrained to struc-
tured schemas that favor extractive grounding to reduce hallucination.
4 Experiments
4.1 Setup
We use two compact LLM families in this work: Qwen3 [Qwen Team, 2025] and Ministral3 [Liu
et al., 2026a]. Both are quantized to Q6_K via llama.cpp [Gerganov, 2023]. All offline one-time
preparation, including quantization, LoRA training, and dataset pre-processing, is performed on
a cloud server with two NVIDIA RTX A6000 GPUs, while the full online SmartRAG pipeline—
memory ingestion, indexing, retrieval, and answer generation—runs entirely on-device. For multi-hop
retrieval planning, we train a task-specific LoRA decomposer [Hu et al., 2022] on a self-constructed
dataset derived from MultiHopQA and HotpotQA, where decomposition targets are generated offline
using Claude Opus 4.6 and then filtered by rules with manual verification. The decomposer is
used only to generate retrieval plans; final answers are always produced by the quantized backbone
conditioned on retrieved evidence. All on-device experiments are conducted on two real smartphones:
OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM) and OnePlus 15 (Snapdragon 8 Elite Gen 5, 16 GB
RAM).
We evaluate on four QA benchmarks: TriviaQA [Joshi et al., 2017], Natural Questions [Kwiatkowski
et al., 2019], HotpotQA [Yang et al., 2018], and MultiHopQA [Ho et al., 2020]. Baselines are
divided into two groups. ForLLM-onlycomparison, we evaluate generation without retrieval
using Qwen3-1.7B, Qwen3-4B, Qwen3-32B, Ministral3-3B, and Ministral3-14B. Among these,
Qwen3-1.7B, Qwen3-4B, and Ministral3-3B represent the compact-model regime that is executable
on-device, with 1.7B/3B corresponding to practically deployable scales and 4B serving as a larger
but still runnable case. By contrast, Qwen3-32B and Ministral3-14B are treated as large-model
capacity references, allowing us to contextualize how far SmartRAG extends the capability of
compact backbones. ForRAG, we evaluate Naïve RAG [Lewis et al., 2020], InstructRAG [Wei et al.,
2024], StableRAG [Zhang et al., 2026], and TruthfulRAG [Liu et al., 2026b], instantiated with two
lightweight backbones, Qwen3-1.7B and Ministral3-3B, to match the compact-model regime targeted
in this work. These baselines are compared under the same inference budget and retrieval context
constraints; additional details on their cloud-side simulation setting are provided in Section C.
We report two complementary metrics.Correctness (Cr)measures whether a prediction is factually
correct with respect to the reference answer, using an LLM-as-judger protocol [Zheng et al., 2024]
under a fixed rubric.Token-F1measures token-level overlap between prediction and reference using
a rule-based normalization and matching procedure. Together, these metrics capture both factual
validity and surface-level answer overlap. Full evaluation details, including the judge prompt and the
Token-F1 computation procedure, are provided in Section A.
4.2 Main results: structured memory improves small-backbone QA under on-device
constraints
Table 1 compares SmartRAG against both LLM-only capacity references and lightweight retrieval
baselines across the Qwen3 and Ministral3 families. Overall, SmartRAG delivers the clearest gains
on NQ, HotpotQA, and MultiHopQA, while remaining competitive on TriviaQA. This pattern is
consistent across both model families, suggesting that the main benefit of SmartRAG lies in structured
evidence acquisition and composition.
SmartRAG is most effective on knowledge-intensive and evidence-composition benchmarks.
Across both backbone families, the clearest gains of SmartRAG appear on NQ, HotpotQA, and
MultiHopQA, where it consistently ranks at or near the top within each block. By contrast, TriviaQA
is more mixed: SmartRAG remains competitive, but its advantage is less uniform than on the other
three benchmarks. This pattern suggests that SmartRAG is especially effective when answering
requires retrieving, organizing, and composing dispersed evidence, whereas direct open-domain
factoid questions can often be handled well by larger parametric models or simpler retrieval strategies.
Comparison with RAG baselines.The baseline pattern also reflects the different objectives of
the compared methods. InstructRAG explicitly learns denoising through self-synthesized rationales,
7

Table 1: Correctness (Cr, %) and F1 (%) on four QA benchmarks. SmartRAG uses a quantized
Qwen3-1.7B backbone running entirely on-device. Models ≥8B are evaluated on A6000 as capacity
references. Within each base-model block, the best result in each column is shown inboldand the
second-best in underlined .
Base Model Method TriviaQA NQ HotpotQA MultiHopQA
Cr F1 Cr F1 Cr F1 Cr F1
Qwen3
LLM-only (no retrieval)
Qwen3-1.7B 22.10 21.52 9.10 10.17 14.15 13.00 22.00 19.18
Qwen3-4B 28.70 37.66 23.15 19.95 24.00 22.29 30.05 29.29
Qwen3-32B51.4542.26 40.15 35.92 33.25 30.56 31.54 32.69
Retrieval-augmented (1.7B)
Naïve RAG 47.00 36.31 62.24 37.23 41.55 37.80 45.66 34.58
InstructRAG 43.15 22.67 41.30 34.99 33.90 26.15 13.90 12.94
StableRAG 45.16 21.89 17.30 11.14 41.15 26.11 5.05 0.75
TruthfulRAG 47.10 22.86 45.10 26.38 40.71 25.51 19.95 12.64
SmartRAG (ours)50.00 48.68 66.68 54.84 63.93 51.27 50.17 41.86
Ministral3
LLM-only (no retrieval)
Ministral3-3B 35.84 28.27 24.43 18.81 16.98 19.19 17.33 26.56
Ministral3-14B60.70 60.8737.80 34.55 22.40 25.34 20.12 31.45
Retrieval-augmented (3B)
Naïve RAG 41.71 33.02 42.99 32.02 36.96 27.08 15.95 14.81
InstructRAG 50.25 28.74 50.60 45.39 42.65 37.41 28.75 26.27
StableRAG 15.95 10.86 27.35 18.11 44.03 32.94 13.71 9.07
TruthfulRAG 47.86 25.52 53.00 31.31 47.86 35.38 43.20 33.67
SmartRAG (ours)59.74 36.76 63.86 54.07 66.74 46.08 57.60 43.37
StableRAG mitigates permutation sensitivity by aggregating reasoning across multiple retrieval
orders, and TruthfulRAG focuses on resolving factual conflicts through triple extraction, KG retrieval,
and conflict-aware filtering. These are all meaningful objectives, but they place more burden on
the generator’s ability to denoise, aggregate, or filter evidence. Under the smaller Qwen3-1.7B
regime, such additional reasoning or filtering stages may not pay off as reliably as direct flat retrieval,
which helps explain why these baselines often underperform Naïve RAG there. Under Ministral3-3B,
however, the stronger compact backbone appears better able to benefit from these extra mechanisms,
so the same baselines become consistently stronger than Naïve RAG. SmartRAG differs in that it
does not rely primarily on one-shot denoising or conflict filtering; instead, it combines persistent
structured memory with hybrid retrieval, which appears to be more robust across both compact
backbone families.
Compact backbones versus larger capacity references.The goal of SmartRAG is to improve
what compact backbones can achieve under strict on-device constraints. Table 1 reflects this trade-off
clearly. Larger standalone models still remain strongest on some direct factoid settings, most notably
Qwen3-32B and Ministral3-14B on TriviaQA. At the same time, SmartRAG repeatedly matches
or exceeds these larger LLM-only references on NQ, HotpotQA, and MultiHopQA. In conclusion,
these results support the central claim of the paper: under device constraints, improving memory
construction and evidence organization can be at least as important as increasing the parameter count
alone.
4.3 On-device feasibility
A central question for SmartRAG is whether theentirepipeline remains usable when memory
construction, retrieval, and answer generation are all executed on-device. To answer this, we profile
two smartphone platforms with two different model sizes, and separately measure the end-to-end
8

Table 2: Device–backbone combinations for on-device profiling. Each pair is denotedS0–S3.
Qwen3-1.7B Qwen3-4B
OnePlus 15 (Snapdragon 8 Elite Gen 5, 16 GB) S0 S1
OnePlus 13 (Snapdragon 8 Elite, 16 GB) S2 S3
Table 3: Average PSS memory and power consumption across device–backbone configurations.
S0–S3 follow the mapping in Table 2.
ConfigTriplet Extraction QA
PSS Memory (MB) Power (W) PSS Memory (MB) Power (W)
S05527.31 4.74 4149.72 2.85
S16717.55 4.77 6803.68 5.36
S25099.07 2.93 4153.38 2.12
S36643.01 2.37 6512.62 2.12
RAG pipeline. Table 2 defines shorthand labels S0–S3 for each hardware–backbone pair, which are
used throughout this section.
In general, the measurements reveal a clear usability boundary. While both backbones can be
executed on-device, the 1.7B model remains in the practically deployable regime, whereas the 4B
model substantially weakens interactivity once the full retrieval-augmented pipeline is taken into
account.
We first examine resource usage during triplet extraction and QA. As summarized in Table 3, the 4B
backbone consistently incurs a substantially heavier memory footprint than 1.7B on both devices.
This gap appears in both triplet extraction and end-to-end QA, where the 1.7B settings remain around
the 4–5.5 GB range while the 4B settings rise to roughly 6.5–6.8 GB. This difference matters because
SmartRAG runs as a full on-device pipeline, with model weights, retrieval-conditioned prefilling, and
runtime memory operations sharing the same limited mobile memory budget.
More broadly, the measured power remains relatively low because we intentionally adopt a conserva-
tive system configuration: we prioritize stable long-term execution so that SmartRAG can remain
deployable as a persistent on-device system. This choice leaves some hardware capacity unused,
which in turn contributes to the non-trivial TTFT observed in the full QA setting.
The more decisive separation appears in latency, shown in Figure 2. Question decomposition remains
comparatively light-weight across all settings. For the S0 configuration (Snapdragon 8 Elite Gen 5
with Qwen3-1.7B), the median decomposition TTFT is 1.32 s , which confirms that this planning stage
remains near the low-second regime. By contrast, full QA is much more expensive because the model
must absorb retrieved evidence before decoding the answer. Under the same S0 setting, the median
QA TTFT rises to 36.9 s , indicating that the dominant latency cost comes from retrieval-conditioned
prefilling rather than from the decomposition step itself. The 4B settings (S1 and S3) are visibly
slower in Figure 2, moving further away from an interactive operating point.
The same conclusion is reinforced by the generation stage in Figure 3. In the S0 runlog, the median
decoding time is 8.11 s for question decomposition and 8.55 s for QA, while the mean QA decoding
time is higher ( 14.81 s ) because answer lengths vary substantially across samples. This pattern again
suggests that the main systems bottleneck is not the decomposition step itself, nor even decoding
alone, but the long prefilling stage required by retrieval-conditioned QA. More broadly, Figure 3
shows the same qualitative ordering across devices and backbone sizes: the 1.7B backbone decodes
faster than 4B, and the smaller configurations remain more favorable for end-to-end deployment.
The memory, TTFT, and decoding measurements explain why we standardize on the 1.7B backbone
in the main accuracy experiments. Although the 4B model is technically executable on smartphones,
its larger memory footprint, much longer retrieval-conditioned prefilling time, and slower generation
place it beyond a practically interactive regime for end-to-end on-device RAG. We also note that
system software still matters: even under closely related Snapdragon-class hardware, OS scheduling
and runtime policies can noticeably affect realized throughput. The reported numbers should therefore
9

S0 S1 S2 S3
Configuration0100200300TTFT (s)QA Time to First Token
S0 S1 S2 S3
Configuration02468TTFT (s)Decomposition Time to First TokenFigure 2: Time to first token (TTFT) under configurations S0–S3. Left: QA TTFT. Right: question
decomposition TTFT. Bars show median values and error bars indicate interquartile range.
S0 S1 S2 S3
Configuration051015Tokens/sQA Decoding Speed
S0 S1 S2 S3
Configuration0246810Tokens/sDecomposition Decoding Speed
Figure 3: Decoding speed under configurations S0–S3. Left: QA decoding speed. Right: question
decomposition decoding speed. Bars show median values and error bars indicate interquartile range.
be interpreted as conservative end-to-end baselines for deployable on-device operation rather than
absolute hardware limits.
4.4 Ablation study
Using MultiHopQA as the test set, we ablate three core components of SmartRAG: MRGraph
(no_MRGraph ), the LLM-based retrieval planner ( no_THINKING_PLAN ), and the hybrid retrieval
design (no_Hybrid_Retrieval, i.e., graph-only retrieval). The results are shown in Figure 4.
Key findings.As shown in Figure 4, the full SmartRAG system achieves the best performance,
with a Token-F1 of 0.5484. Removing MRGraph causes the largest degradation, dropping Token-F1
to 0.4151, which indicates that the structured memory module is the most important component for
downstream multi-hop QA in this setting. Disabling the Thinking planner also reduces performance
to 0.5069, suggesting that explicit multi-hop planning improves evidence acquisition even when the
remaining retrieval and answering pipeline is preserved. Replacing hybrid retrieval with graph-only
retrieval further lowers Token-F1 to 0.4749, showing that graph traversal alone is insufficient and
that SmartRAG benefits from combining graph-based signals with complementary retrieval channels.
Overall, the ablation results support the full SmartRAG design: structured memory, planning, and
hybrid retrieval each make non-trivial contributions, and their combination yields the strongest
end-to-end QA performance.
5 Discussion and conclusion
We have presented SmartRAG, an on-device architecture for adaptive question answering that
decomposes intelligence into four coordinated functional roles. By shifting continuous adaptation to
efficient structured computation and reserving LLM inference for high-value semantic operations,
SmartRAG achieves multi-hop reasoning performance competitive with substantially larger cloud-
based models while running entirely on commodity smartphones.
10

Full
No MRGraphNo Thinking Plan No Hybrid Retrieval(Only Graph)0.400.450.500.55F10.5484
0.41510.5069
0.4749Ablation StudyFigure 4: Ablation results on NQ. We compare the full SmartRAG system with variants removing
MRGraph, the Thinking planner, or the hybrid retrieval design. Token-F1 is reported as the primary
downstream QA metric.
The retrieval–precision trade-off.Our analysis of the HotpotQA gap between SmartRAG and
Naïve RAG reveals a fundamental trade-off in graph-structured retrieval. While graph traversal
introduces relationships between nodes that enhance multi-hop reasoning, it also surfaces structurally
related but contextually peripheral evidence. For open-ended questions where direct vector matching
suffices, this can decrease recall. Intelligent search strategies that dynamically balance graph depth
against retrieval precision represent a promising research direction.
Why not LLM-native extraction?Although a fully LLM-native extraction pipeline could reduce
modular error accumulation and support richer predicate structures, our profiling shows that even
query-time RAG prefilling can push TTFT to the minute scale on commodity smartphones. Making
LLMs the primary writer of a continuously updated knowledge graph would introduce unacceptable
latency and energy costs, and would require on-device training capabilities that remain impractical.
The separation into a lightweight, trainable perception layer and a sparse LLM control plane yields
predictable ingestion costs while retaining LLM expressivity where it matters most.
Limitations.(1) The current planner is heuristic and does not uniformly benefit all question
types; learned planning strategies could address this. (2) On-device QA latency, while acceptable
for the 1.7B backbone, is dominated by retrieval-conditioned prefilling and would benefit from
evidence compression techniques. (3) Typed NER remains low in AI and Politics despite surface
gains; improving type-consistent discovery and clustering is still a major lever, even when Token-F1
and Hit@1 already move in the right direction. (4) Our evaluation is limited to English-language
benchmarks; multilingual adaptation remains future work.
Broader impact.SmartRAG enables privacy-preserving personal assistants that operate entirely
on-device without sending user data to the cloud. While this is broadly positive for user privacy, we
note that any system capable of building detailed personal knowledge graphs raises data governance
concerns if the device is compromised. Appropriate encryption and access control mechanisms
should complement the on-device architecture.
Acknowledgments and Disclosure of Funding
Acknowledgments will be added in the camera-ready version.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve,
generate, and critique through self-reflection. InInternational Conference on Learning Representations, 2024.
Xin Cheng, Linzhe Luo, Hao Huang, Shuai Li, Daliang Wang, Yankai Lin, Zhiyuan Chen, and Maosong Sun.
xRAG: Extreme context compression for retrieval-augmented generation with one token. InarXiv preprint
arXiv:2405.13792, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan
Larson. From local to global: A graph RAG approach to query-focused summarization. InarXiv preprint
arXiv:2404.16130, 2024.
11

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization
for generative pre-trained transformers.International Conference on Learning Representations, 2023.
Jinlan Fu, Xuanjing Huang, and Pengfei Liu. SpanNER: Named entity re-/recognition as span prediction. In
Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pages 7183–7195,
2021.
Georgi Gerganov. llama.cpp.GitHub repository, 2023.https://github.com/ggerganov/llama.cpp.
Aaron Grattafiori et al. The Llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. InNIPS Deep
Learning and Representation Learning Workshop, 2015.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset
for comprehensive evaluation of reasoning steps. InProceedings of the 28th International Conference on
Computational Linguistics, pages 6609–6625, 2020.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. LoRA: Low-rank adaptation of large language models. InInternational Conference on Learning
Representations, 2022.
Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S Yu. A survey on knowledge graphs:
Representation, acquisition, and applications.IEEE Transactions on Neural Networks and Learning Systems,
33(2):494–514, 2022.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting of the Association
for Computational Linguistics, pages 1601–1611, 2017.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing, pages 6769–6781, 2020.
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu,
Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath,
Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. InProceedings
of the National Academy of Sciences, volume 114, pages 3521–3526, 2017.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: A benchmark for question
answering research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive NLP tasks. InAdvances in Neural Information Processing Systems,
volume 33, pages 9459–9474, 2020.
A. H. Liu et al. Ministral 3.arXiv preprint arXiv:2601.08584, 2026a.
Shuyi Liu, Yuming Shang, and Xi Zhang. Truthfulrag: Resolving factual-level conflicts in retrieval-augmented
generation with knowledge graphs. InProceedings of the AAAI Conference on Artificial Intelligence, 2026b.
Natawut Monaikul, Giuseppe Castellucci, Simone Filice, and Oleg Rokhlenko. Continual learning for named
entity recognition.Proceedings of the AAAI Conference on Artificial Intelligence, 35(15):13570–13577, 2021.
Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The PageRank citation ranking: Bringing
order to the web.Stanford InfoLab Technical Report, 1999.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language
models and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and Data Engineering, 2024.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and
narrowing the compositionality gap in language models. InFindings of the Association for Computational
Linguistics: EMNLP 2023, 2023.
Qwen Team. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
12

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with
chain-of-thought reasoning for knowledge-intensive multi-step questions. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics, pages 10014–10037, 2023.
Joe H Ward. Hierarchical grouping to optimize an objective function.Journal of the American Statistical
Association, 58(301):236–244, 1963.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. Instructrag: Instructing retrieval-augmented generation via self-
synthesized rationales.arXiv preprint arXiv:2406.13629, 2024.
Jiajun Xu et al. On-device language model: A comprehensive review. InarXiv preprint arXiv:2409.00088, 2024.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. InProceedings
of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
Juntao Yu, Bernd Bohnet, and Massimo Poesio. Named entity recognition as dependency parsing. InProceedings
of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6470–6476, 2020.
Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng. Stable-rag: Miti-
gating retrieval-permutation-induced hallucinations in retrieval-augmented generation.arXiv preprint
arXiv:2601.02993, 2026.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric P Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena. InAdvances in Neural Information Processing Systems, volume 36, 2024.
A LLM-as-judger protocol
We assess factual correctness using an LLM-as-judger approach [Zheng et al., 2024]. Given a
question, a reference answer (or set of acceptable answers), and the model’s prediction, a judge
LLM (Claude Haiku 4.5) determines whether the prediction is factually correct under a strict rubric.
The judge outputs a binary correctness label per sample; we report the mean across all samples as
Correctness (Cr, %). SQuAD-styleF1is computed as standard token-level precision–recall against
reference answers. All models are evaluated under identical inference budgets: 4096 token context,
top-K=8 retrieval cutoff, and Q6_K quantization.
B Device specifications
Table 4: Evaluation device specifications.
OnePlus 13 OnePlus 15
SoC Snapdragon 8 Elite Snapdragon 8 Elite Gen 5
RAM 16 GB LPDDR5X 16 GB LPDDR5X
Storage 256 GB UFS 4.0 256 GB UFS 4.0
OS ColorOS 16 ColorOS 16
C Baseline implementation details and deployment notes
To make the comparison with prior RAG baselines as fair and transparent as possible, we implemented
Naïve RAG, InstructRAG, StableRAG, and TruthfulRAG under a unified evaluation backend and
quantization setting. In all cases, we used the same llama.cpp inference backend and the same
model quantization scheme as in our main experiments, and instantiated the baselines with the same
compact backbone families reported in Table 1. This appendix clarifies an important distinction:
unlike SmartRAG, these baselines werenotprofiled as end-to-end real-phone deployments. Instead,
they were executed in a cloud-side simulation environment designed to match the same inference
backend, quantized models, and retrieval budget as closely as possible.
13

Why these baselines were not deployed as real-phone end-to-end systems.Our goal in the
baseline comparison is to evaluate answer quality under a matched compact-backbone regime, rather
than to claim that every baseline is equally suitable for direct mobile deployment. The original
formulations of InstructRAG, StableRAG, and TruthfulRAG introduce additional stages that make
them difficult to treat as drop-in phone pipelines. InstructRAG explicitly denoises retrieved contents
through self-synthesized rationales and supports both in-context learning (ICL) and supervised
fine-tuning (SFT). TruthfulRAG introduces additional KG-oriented processing steps, including
knowledge-graph construction, graph retrieval, and entropy-based filtering. StableRAG emphasizes
repeatability, frozen retrieval, and structured output normalization, with a pipeline that depends
on OpenAI-style generation and embedding calls together with a persistent retrieval store. These
original designs are meaningful in their intended settings, but they differ substantially from a single-
pass compact on-device assistant stack. For this reason, we treat them asreference RAG baselines
evaluated under a matched backend and quantization setup, rather than as directly deployable mobile
systems.
InstructRAG.InstructRAG is a framework that explicitly denoises retrieved contents by generating
rationales, and supports both in-context learning (ICL) and supervised fine-tuning (SFT). In our
experiments, we implemented the method as described in the original paper, replacing the original
serving stack with the same quantized compact backbones and llama.cpp backend used in our main
experiments. We did not attempt to reproduce the original large-scale fine-tuning regime; instead, we
adapted the method to the same compact-backbone comparison regime used for the other baselines.
We therefore interpret the resulting numbers as measuring how well theInstructRAG-style denoising
strategytransfers to compact quantized backbones under a unified inference setting, rather than as a
claim about the maximum achievable performance of the original method.
TruthfulRAG.TruthfulRAG is a KG-based framework for resolving factual-level conflicts in RAG,
whose pipeline includes knowledge-graph generation, graph retrieval, and entropy-based filtering
before final response generation. In our evaluation, we preserved this overall conflict-aware graph-
processing logic while replacing the original model-serving setup with the same compact quantized
backbones and llama.cpp backend used elsewhere in the paper. Because TruthfulRAG contains
additional graph-construction and filtering stages beyond a standard lightweight RAG loop, we did
not deploy it as an end-to-end real-phone pipeline. Instead, we evaluate it in the same cloud-side
simulation environment as the other RAG baselines. This makes the comparison more controlled at
the backbone and inference level, while acknowledging that TruthfulRAG was originally designed
for a somewhat different operating point than a strict mobile deployment target.
StableRAG.StableRAG is a method designed to improve repeatability and robustness in RAG
systems, mitigating retrieval-permutation-induced hallucinations through stable retrieval ordering and
structured rendering. In our experiments, we retained the stability-oriented evaluation logic described
in the original paper while replacing the original serving configuration with the same compact
quantized backbones and llama.cpp backend used in the rest of the paper. As with TruthfulRAG and
InstructRAG, we treat the resulting system as asimulated reference implementation under matched
inference conditions, not as a true smartphone deployment. This distinction is important because
StableRAG is primarily designed to study response stability and retrieval repeatability, whereas
SmartRAG is optimized for persistent structured memory and end-to-end on-device execution.
Interpretation of the comparison.We emphasize that these baselines are included as strong and
relevant points of comparison, not as strawmen. InstructRAG targets denoising under noisy retrieval;
TruthfulRAG targets factual conflict resolution with KG support; and StableRAG targets repeatability
and robustness to retrieval variation. These are all reasonable objectives, and each may be better
aligned than SmartRAG with other deployment scenarios. Our comparison should therefore be
read narrowly: under a unified compact-backbone, quantized, llama.cpp -based setting, SmartRAG
is more effective for the particular combination of goals studied in this paper—namely, persistent
structured memory, repeated evidence composition, and end-to-end QA under edge-style resource
constraints. We do not claim that the released original forms of these methods are inferior in general,
nor that they were designed for the same deployment target as SmartRAG.
14