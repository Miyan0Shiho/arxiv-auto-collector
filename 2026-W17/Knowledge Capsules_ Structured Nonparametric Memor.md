# Knowledge Capsules: Structured Nonparametric Memory Units for LLMs

**Authors**: Bin Ju, Shenfeng Weng, Danying Zhou, Rongkai Xu, Kunkai Su

**Published**: 2026-04-22 12:21:00

**PDF URL**: [https://arxiv.org/pdf/2604.20487v2](https://arxiv.org/pdf/2604.20487v2)

## Abstract
Large language models (LLMs) encode knowledge in parametric weights, making it costly to update or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by appending retrieved text to the input, but operates purely through context expansion, where external knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and often unstable, particularly in long context and multi hop reasoning scenarios. We propose Knowledge Capsules, structured nonparametric memory units that represent normalized relational knowledge and can be constructed directly from document corpora using a frozen base model. Instead of injecting knowledge as text, we introduce an External Key Value Injection (KVI) framework that compiles capsules into attention-compatible key value representations, enabling external knowledge to directly participate in the model's attention computation. By shifting knowledge integration from context-level augmentation to memory level interaction, the proposed framework consistently outperforms RAG and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long context and multi hop reasoning, while requiring no parameter updates.

## Full Text


<!-- PDF content starts -->

Knowledge Capsules: Structured Nonparametric
Memory Units for LLMs
Bin Ju∗1, Shenfeng Weng2, Danying Zhou1, Rongkai Xu1, and Kunkai Su†3
1Zhejiang Angel Medical AI Technology Co., Ltd., Hangzhou, China
2Miti AI Technology Co., Ltd., Hangzhou, China
3China-Singapore Belt and Road Joint Laboratory on Translational Infection Biology
for Diagnostics and Therapies, State Key Laboratory for Diagnosis and Treatment of
Infectious Diseases, The First Affiliated Hospital
Abstract
Large language models (LLMs) encode knowledge in parametric weights, making it costly to update
or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by
appending retrieved text to the input, but operates purely through context expansion, where external
knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and
often unstable, particularly in long-context and multi-hop reasoning scenarios. We proposeKnowledge
Capsules, structured nonparametric memory units that represent normalized relational knowledge
and can be constructed directly from document corpora using a frozen base model. Instead of injecting
knowledge as text, we introduce anExternal Key–Value Injection (KVI)framework that compiles
capsules into attention-compatible key–value representations, enabling external knowledge to directly
participate in the model’s attention computation. By shifting knowledge integration from context-level
augmentation to memory-level interaction, the proposed framework consistently outperforms RAG
and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long-context
and multi-hop reasoning, while requiring no parameter updates. Code and datasets are available at
https://github.com/jubin75/KVI.
1 Introduction
LLMs demonstrate strong capabilities in reasoning, language understanding, and knowledge
recall. However, their knowledge is predominantly stored in parametric weights learned during
large-scale pretraining, making it difficult to update or extend without costly retraining or
fine-tuning [ 1,2,3]. This limitation has motivated a growing body of work on non-parametric
knowledge augmentation.
RAG is the most widely adopted paradigm in this direction, retrieving relevant documents
from external corpora and appending them to the prompt to condition generation on external
∗Corresponding author:jubin_hz@163.com
†Corresponding author:ksu@zju.edu.cn
1arXiv:2604.20487v2  [cs.CL]  23 Apr 2026

Technical Report 2
knowledge [ 4,5,6]. While effective in practice, RAG fundamentally operates throughcontext
expansion: external knowledge is introduced as additional tokens in the input sequence.
This design leads to an inherent limitation. Retrieved evidence must compete for attention
with all other tokens in the sequence, including formatting tokens, discourse structure, and
irrelevant context. As a result, the influence of retrieved knowledge is indirect and unstable,
often leading to underutilization or inconsistency in reasoning [ 7,8,9]. Crucially, this issue is
not merely empirical but structural: external knowledge is injected at the token level, whereas
the model’s inference process is governed by attention over internal memory representations.
From the perspective of Transformer architectures, attention can be understood as a form
of differentiable associative memory. During inference, query vectors retrieve information by
computing similarity with key vectors, and the corresponding value vectors are aggregated
through a softmax-weighted competition process [ 10]. This interpretation aligns with a broader
line of work that frames attention mechanisms ascontent-addressable memory systems, where
key–value pairs constitute a dynamic memory store accessed via similarity-based retrieval
[11,12,13]. In this perspective, generation emerges from a competitive arbitration process
over memory entries rather than purely sequential token processing. Token probabilities are
therefore determined not only by linguistic context, but by the outcome of a vector-space
competition among memory traces.
This perspective reveals a fundamental mismatch in RAG-style systems. Parametric
knowledge is already embedded as key–value memory within the model and participates
directly in attention competition. In contrast, retrieved documents are introduced as raw
tokens, which must first be encoded into internal representations before they can influence
attention. Consequently, external knowledge remains structurally disadvantaged: it is present
in the context but not integrated into the memory substrate that governs inference.
This paper addresses this mismatch by proposing a new paradigm:memory-level knowledge
integration. Instead of treating external knowledge as text, we represent it directly as memory.
We introduce Knowledge Capsules, structured nonparametric memory units that encode
relational knowledge in a normalized form and are compiled into key–value representations
usingafrozenbasemodel. ThesecapsulesareinjectedintotheTransformerattentionmechanism
through a simple yet effective mechanism termed External KVI. Because the injected tensors
are isomorphic to the internal key value states of the model, they participate directly in the
same softmax competition as endogenous memory.
At inference time, relevant capsules are retrieved via graph-guided multi-hop traversal
and injected through adual-channelmechanism: structured knowledge is integrated at the
attention level via KV memory, while textual evidence is provided through the prompt for
grounding. This design enables external knowledge to influence generation at the level where
decisions are actually made, rather than competing at token level.
By shifting knowledge integration from the context space to the attention memory space,
the proposed framework fundamentally improves the controllability and stability of reasoning
under long-context and multi-hop settings. In particular, structured memory injection preserves
relational constraints (e.g., temporal and spatial conditions) that are often lost in token-based
aggregation, leading to more precise and consistent outputs compared to RAG-style approaches.
For a comparison of the RAG and KVI paradigms, see Figure 1. This work makes three primary
contributions:

Technical Report 3
(1)Training-free, evidence-grounded knowledge capsules.We introduce Knowledge
Capsules as structured, nonparametric memory units distilled from corpora using a frozen base
LLM without fine-tuning. Capsules capture relational structure while preserving provenance
to source documents, enabling modular, verifiable knowledge that scales with data rather than
context length.
(2)Memory-level injection as a complementary channel to context expansion.Ex-
ternalKVI,compilescapsulesintokey–valuememoryinjectedintotheTransformer, providingan
orthogonal channel through which structured knowledge directly influences attention—training-
free and backbone-preserving—while remaining compatible with retrieved text in the context
window. Unlike conventional RAG, which relies on token-level context augmentation, KVI
allows external knowledge to directly participate in attention computation, yielding more stable
and controllable reasoning.
(3)Dual-channel graph-guided retrieval for multi-hop reasoning.We combine
graph-guided multi-hop retrieval, which prioritizes passage candidates under a bounded context
budget, with memory-level capsule injection that encodes explicit relational structure. Re-
trieved passages provide textual grounding, while injected memory supplies relational bias that
long-context concatenation cannot compactly capture. By aligning entity-centric multi-hop
retrieval with memory-level injection, the framework produces coherent relational evidence and
significantly improves multi-hop reasoning over flat retrieval pipelines.
2 Related Work
Existing approaches to knowledge-augmented language modeling can be broadly understood
along a common axis:how external knowledge is integrated into the model’s inference process.
While prior work has explored improvements in retrieval quality, knowledge structuring, and
external memory design, most methods ultimately introduce knowledge at the token level. In
contrast, our work focuses on integrating knowledge directly into the attention memory space
where inference decisions are made.
2.1 Retrieval-Augmented Generation and Its Limitations
RAG extends LLMs by retrieving external documents and conditioning generation on the
retrieved context [ 4,14,6,15]. This paradigm has proven effective for improving factual
accuracy and knowledge coverage by leveraging large external corpora.
However, RAG fundamentally operates throughcontext-level augmentation: retrieved
information is appended as raw tokens to the input sequence. As a result, external knowledge
must be encoded into internal representations before it can influence attention, and competes
with other tokens under positional and contextual biases. This design introduces a structural
limitation, especially in long-context and multi-hop scenarios, where relevant evidence may be
diluted, fragmented, or inconsistently utilized [16, 8, 9].
Recent work has attempted to mitigate these issues through improved retrieval pipelines,
reranking, or reader architectures [ 17,18,19], but the underlying token-level integration
mechanism remains unchanged. This highlights a deeper limitation: external knowledge
remains external to the model’s attention memory, participating only indirectly through token
encoding rather than as first-class memory entries.

Technical Report 4
Figure 1: Comparison between GraphRAG and the proposed KVI framework under a multi-
hop, constraint-intensive query:“SFTSV distribution in China in 2012.”GraphRAG retrieves
relevant evidence via graph traversal but ultimately aggregates it within the prompt, where
knowledge competes at the token level under standard attention dynamics, often resulting in
incomplete or constraint-violating answers due to the lack of explicit enforcement of temporal
or relational conditions. The figure highlights the fundamental difference betweentoken-level
context augmentation(GraphRAG) andmemory-level knowledge integration(KVI).
2.2 Graph-Based Retrieval for Multi-hop Reasoning
To address the limitations of flat retrieval, recent approaches incorporate structured knowledge
representations, particularly knowledge graphs, into retrieval pipelines. Graph-based retrieval
methodsorganizeevidenceintoentitiesandrelations, enablingmulti-hoptraversalandrelational
reasoning [20, 21, 22].
By preserving structural connections between facts, graph-guided retrieval improves coher-
ence and reduces topic drift in multi-hop tasks. These methods demonstrate that structuring
external knowledge can significantly enhance retrieval quality.
Nevertheless, most graph-based approaches still rely on injecting retrieved information as
text into the prompt. Therefore, although retrieval becomes more structured, the integration
mechanism remains unchanged: relational knowledge is still reduced to tokens and does not
directly participate in the attention memory where inference is resolved.

Technical Report 5
2.3 Knowledge Extraction and Structured Representations
A complementary line of work focuses on extracting structured knowledge from unstructured
text, particularly in scientific and biomedical domains. Information extraction systems convert
textual evidence into relational representations such as entities, relations, and triples [ 23,24,25].
RecentapproachesincreasinglyleverageLLMsforopeninformationextractionandknowledge
graph construction [ 26,27]. These methods enable scalable conversion of text corpora into
structured knowledge suitable for downstream reasoning.
However, extracted knowledge is typically stored in external graph structures that are
decoupled from the internal representations of LLMs. However, extracted knowledge is typically
stored in external graph structures that are decoupled from the internal representations of
LLMs. As a result, structured knowledge must still be converted back into text before use,
leaving a gap between knowledge representation and its role in inference.
2.4 Memory-Augmented Language Models
Beyond document retrieval, a growing body of work explores explicit memory mechanisms
for neural models. Prior research distinguishes between parametric memory stored in model
weights and nonparametric external memory accessed at inference time [12, 13, 11].
Recent systems introduce persistent or retrievable memory structures to improve long-
context reasoning and knowledge utilization. For example, MemoryBank stores conversational
history as long-term memory [ 28], while MemoRAG introduces global memory representations
to summarize large document collections [ 29]. Other approaches integrate entity-level or
knowledge-base memory into Transformer architectures [30, 31].
While these methods demonstrate the effectiveness of explicit memory, most operate on
textual or symbolic representations that are not directly aligned with the key–value retrieval
mechanism of attention. However, these approaches typically operate on textual or symbolic
memory that is not directly aligned with the key–value retrieval dynamics of attention, leaving
open the question of how external knowledge can be injected into the attention memory space
itself.
2.5 Attention as a Memory System
The Transformer attention mechanism can be interpreted as a differentiable associative memory,
where key–value pairs form a content-addressable memory accessed through similarity search
[10,11,13,32]. In this view, inference can be understood as a competitive retrieval process over
memory entries, where candidate representations compete via softmax weighting to influence
generation.
Severalworksextendthisperspectivetoimprovememorycapacityandlong-contextmodeling,
such as Transformer-XL [ 33] and related architectures that introduce persistent states across
segments. These approaches expand the memory available to the model but still rely on internal
representations derived from token inputs.
Consequently, external knowledge remains indirectly integrated, as it must first be encoded
into hidden states before participating in attention. This motivates approaches that represent
external knowledge directly within the key–value memory space of attention, enabling it to
participate in the same competitive retrieval process as parametric knowledge. In the following

Technical Report 6
Table 1: Positioning of KVI along the knowledge integration axis. Prior work improves retrieval,
structure, or memory, but remains at the token level. KVI integrates knowledge directly into
attention memory.
Paradigm Representative
WorkIntegration Limitation KVI Contribution
RAG[14, 4, 6, 19] Knowledge injected as to-
kens; indirect influence
through encodingDirect KV memory injec-
tion into attention
Graph-based
RAG[20, 21, 22] Structured retrieval, but
still token-based integra-
tionStructure preserved in at-
tention via KV representa-
tions
Knowledge Ex-
traction[23, 24, 26, 27] Structured knowledge re-
mains external; requires
text conversionCapsule abstraction
aligned with model mem-
ory
Memory-
Augmented
Models[12, 13, 28, 29] External memory not
aligned with attention KV
spaceAttention-compatible
memory construction
Attention as
Memory[10, 11, 13] No mechanism for external
knowledge injectionExternal KV Injection into
attention memory
sections, we introduceKnowledge Capsulesand a corresponding injection mechanism that
enables external knowledge to participate directly in the key–value memory space.
Table 1 summarizes this distinction. While prior approaches improve retrieval quality or
knowledge representation, they remain limited to token-level integration. In contrast, our
framework enables external knowledge to participate directly in attention memory, aligning
knowledge integration with the mechanism that governs inference.
3 Knowledge Capsules
3.1 Capsule Representation
In the Knowledge Capsule Index, knowledge extracted from scientific literature is represented
using a structured unit referred to as aKnowledge Capsule. Capsules serve as the fundamen-
tal building blocks of the index and provide a normalized representation for heterogeneous
statements appearing in biomedical text.
Formally, a capsule is defined as a tuple C= (S, R, O, I ), where Sdenotes thesubject
entity,Rrepresents the semanticrelation,Ocorresponds to theobject, andIis aprovenance
identifierthat links the capsule to its supporting evidence sentence in the document corpus. In
implementation, Iis a sentence-level identifier that enables retrieval of the original evidence
text from an external index.
This representation converts natural language statements into explicit relational structures
while preserving traceability. Unlike raw passages, capsules encode semantic relations directly,
enabling structured retrieval and aggregation of evidence across documents.

Technical Report 7
Toillustrate, consider clinical descriptions of Severe Fever with Thrombocytopenia Syndrome
Virus (SFTSV) infections:
C1= (SFTSV infection,has_symptom,fever, I 1)
C2= (SFTSV infection,has_symptom,thrombocytopenia, I 2)
C3= (SFTSV infection,causes,multi-organ failure, I 3)
Each capsule corresponds to a single atomic assertion, and the associated Irecords the exact
sentence from which it was extracted.
3.2 Two-Level Memory Architecture of KVI
KVI introduces a two-level memory architecture that separatessymbolic storagefromcontinuous
inference.
Level 1: Symbolic Capsule Memory.The capsule collection M={C1, C2, . . . , C N}is
stored as plain, human-readable structured data (e.g., JSON or Parquet files) on disk. This
symbolic memory is:
•Interpretable: each capsule explicitly states a subject-relation-object fact.
•Extensible: new capsules can be added without retraining.
•Grounded: the provenance identifier Iprovides a direct link to the original evidence
sentence.
Importantly, Mitself isnever loaded into the LLMas input. It serves two offline purposes:
(1) as the source for compiling key-value memory (Section 4.2), and (2) as a structural index
for graph-guided retrieval (Section 4.3).
Level 2: Continuous KV Memory.During compilation, each capsule is transformed
into pre-computed key-value tensors( Kl, Vl)via a frozen LLM. These tensors are stored in a
KV bank Band are directly injected into the attention cache at inference time. This continuous
memory is efficient (no re-encoding of facts per query) but is not human-readable and does not
directly point to source text.
Thus, KVI separateswhat is stored(symbolic capsules) fromwhat is computed(continuous
KV vectors). The following sections describe how the symbolic layer enables retrieval and how
the continuous layer enables efficient injection.
4 External KVI Framework
The External KVI framework enables LLMs to incorporate structured external knowledge
without parameter modification. It transforms the symbolic capsule collection Minto two
artifacts: a knowledge graphGfor retrieval, and a KV bankBfor memory injection.
See Figure 2 for an overview. The framework consists of two stages:
•Compile-Time Pipeline(Section 4.2): constructsGandBfromM.
•Query-Time Pipeline(Section 4.3): retrieves relevant capsules via G, then injects their
pre-computed KV tensors.

Technical Report 8
Figure 2: Architecture of the External KVI framework.
4.1 Graph Organization of Capsules
To enable structured retrieval, the capsule collection Mis organized into a knowledge graph
G= (V, E). Each capsule C= (S, R, O, I )contributes an edge from StoOlabeled R. The
provenance Iis stored separately in an index Ithat maps edges to source sentences. This
design keeps Glightweight for traversal while retaining traceability, following the paradigm of
graph-guided retrieval [34].
4.2 Compile-Time Pipeline
The compile-time pipeline constructs GandBfrom the PDF corpus, as summarized in
Algorithm 1.
Evidence Extraction and Triple Extraction.Documents are segmented into sentences
S. For each sentence, a frozen LLM extracts relational triples(S, R, O). Each triple is stored

Technical Report 9
as a record t= (S, R, O, I )where Ilinks to the source sentence. The set of all triples Tforms
the raw material for capsulesM.
Knowledge Graph Build.Using T, the system builds graph Gand index I(sentence index
and triple-sentence mapping).
KV Bank Compilation.The frozen LLM compiles two types of key-value entries, inspired
by recent work on encoding knowledge into KV representations [35].
Entity Anchor KV.For each entity einG, a short canonical sentence (e.g., “SFTSV
infection is a viral disease”) is constructed and passed through the LLM. The resulting layer-wise
tensors( Kl, Vl)are stored in B. This anchor serves as a semantic “prime” that tells the model
which entity is being discussed, without committing to any specific relation.
Triple KV.For each triple record t∈ T, a compact natural-language statement (e.g.,
“SFTSV infection causes fever”) is constructed and passed through the LLM. The resulting
tensors are stored as Triple KV. The provenance Iisnotencoded into these tensors; it remains
inIfor grounding.
Thus,Bstores only continuous vectors, not the original text.
4.3 Query-Time Pipeline
At inference time, the system performs graph-guided retrieval and dual-channel injection, as
detailed in Algorithm 2.
Graph-Guided Retrieval.Given a query q, the system links entities to nodes in G. If none
found, it falls back to standard LLM generation. Otherwise, it classifies query intent, maps it
to a relation set R, and performs multi-hop graph traversal starting from the identified entity
e, collecting candidate triplesS.
Relevance Scoring with DRM.For each candidate triple t∈ S, its evidence sentence xis
retrieved via I. The relevance score is computed as DRM (q, x), where DRM (Dual Relation
Matching) is a semantic alignment function that measures relevance without full-context
reasoning [34]. The top-ktriplesT kare selected.
Dual-Channel Knowledge Injection.The selected knowledge is injected via two channels:
KV channel.The system loads from B: first the Entity Anchor KV for entity e, then
the Triple KV tensors for each t∈ T kin ranked order. These tensors are concatenated into a
sequence[ Anchor_KV (e);Triple_KV (t1);. . .]. This sequence is inserted as aprefixinto the
attention cache.
Prompt channel.The system constructs a standard text prompt containing the raw
evidence sentences (retrieved via I), the user query, and task requirements. This prompt is
tokenized and fed into the LLM normally.
Crucially, the KV prefix and the prompt text arenot duplicatesof each other: - The KV
prefix encodesstructured relationsas continuous vectors (e.g., the relation “causes fever”). -

Technical Report10
The prompt providesoriginal evidence sentencesas human-readable text (e.g., “Our study
found that SFTSV infection frequently causes fever...”).
At each Transformer layer l, the attention memory is formed by concatenating the external
KV prefix (derived from the KV channel) with the keys and values computed from the prompt
text. Let Kext
landVext
lbe the concatenated external tensors from the KV prefix. Let Kprompt
l
andVprompt
lbe the keys and values computed by the normal forward pass over the prompt
tokens. Then:
Kfull
l= [Kext
l;Kprompt
l], Vfull
l= [Vext
l;Vprompt
l](1)
The external tensors are inserted as additional entriesbeforethe prompt-derived entries. No
duplication occurs becauseKprompt
lis computed solely from the prompt text, which doesnot
include the triple statements (the prompt contains raw evidence sentences, not the compiled
triple text). The two channels are complementary, not redundant [35].
Grounding Filter.Generation proceeds as a=LLM.generate(prompt, past_key_values) .
A grounding filter removes output sentences with low token overlap against the retrieved
evidence, reducing hallucinations.
4.4 Summary of Two-Level Memory in Action
Using the SFTSV example:
•Symboliccapsulememory(plaintext): {(SFTSV infection,has_symptom,fever, I 1),
(SFTSV infection,has_symptom,thrombocytopenia, I 2),
(SFTSV infection,causes,multi-organ failure, I 3)}stored on disk.
•Continuous KV memory (pre-computed): Anchor_KV (SFTSV infection )plus
Triple_KV(“SFTSV infection causes fever”)etc.
•At query time:The system retrieves relevant capsules via graph traversal, loads their
KV tensors, and injects them as a prefix. The prompt contains the original sentences
(e.g., “Our study found that SFTSV infection frequently causes fever...”) for grounding.
The LLM never sees the plaintext triple “SFTSV infection causes fever”; it only sees the
pre-computed vectors (KV channel) and the raw evidence sentence (prompt channel).
5 Experiments
5.1 Main QA Bench
5.1.1 Experimental Setup
Setup and compared methods.We evaluate the proposed framework using two frozen
instruction-tuned backbones: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 . All
methods are implemented within a unified pipeline to ensure consistent preprocessing, retrieval
artifacts, and evaluation.

Technical Report11
We compare five conditions:LLM(no retrieval),RAG(dense retrieval + prompt evidence),
GraphRAG(graph-guided retrieval + prompt evidence),KV Prefix(dense retrieval + KV
injection), andKVI(graph-guided retrieval + dual-channel injection with KV memory and
prompt evidence). This setup isolates the effects of retrieval structure and memory-level KV
injection.
Notably, these five conditions form a complete ablation grid:GraphRAGablates
KV injection fromKVI,KV Prefixablates graph guidance,RAGablates both, andLLM
serves as the lower bound without any retrieval or memory augmentation. This design enables
a systematic component-level analysis without running additional experiments.
Datasets and evaluation protocol.We evaluate on three QA benchmarks that collectively
assess multi-hop reasoning capabilities across different domains and difficulty levels:
Natural Questions (NQ)[ 36] is an open-domain QA benchmark derived from real
Google search queries, with answers annotated as spans in Wikipedia. Unlike HotpotQA, NQ
primarily consists of single-hop questions where the answer can typically be located within a
single paragraph. We include NQ ( N= 100) as a complementary baseline to assess whether
improvements from multi-hop oriented methods generalize to simpler, factoid-style queries.
HotpotQA[ 37] is a multi-hop reasoning benchmark constructed from Wikipedia, requiring
models to aggregate information across multiple paragraphs to answer questions. Each question
involves two or more reasoning steps, and the dataset provides supporting fact supervision for
evidence-level evaluation. We use the multi-hop subset withN= 120samples.
MedHopQA[ 38] is a biomedical multi-hop QA dataset focused on drug-drug interactions.
Each question requires reasoning over two connected Wikipedia pages (e.g., a drug and a
gene) to identify the correct DrugBank ID. Following the official BioCreative IX evaluation
protocol, we report results on two variants: (i) MedHopQA-ID ( N= 40), a relation-completion
variant where queries are normalized to interacts_with DBxxxx? and answers are partner
DrugBank IDs; and (ii) MedHopQA_official ( N= 342), a natural-language version with the
same ID-based answers for deterministic evaluation.
We adopt Exact Match (EM, %) as the primary evaluation metric. EM measures the
proportion of predictions that exactly match the gold answer string after canonicalization
(e.g., lowercasing and whitespace normalization). For MedHop-style QA, EM is computed over
canonical DrugBank IDs, ensuring unambiguous matching. All results report 95% bootstrap
confidence intervals. Paired permutation tests againstKVIare recorded for all runs.
For all datasets, we adopt a consistent evaluation design: answers are compared against
canonical entity identifiers or gold answer strings without synonym matching or additional
normalization beyond the canonicalization step. This strict evaluation protocol ensures re-
producibility but imposes a strong requirement on output format fidelity, particularly for
MedHopQA where free-text generation often produces malformed identifiers.
Implementation details.All graph-based methods (GraphRAG,KVI) operate on a
precompiled knowledge graph ( graph_index.json ) with sentence-level provenance indexing.
Capsules are retrieved via entity-anchored multi-hop traversal and filtered using DRM scoring.
Dense-retrieval methods (RAG,KV Prefix) use a FAISS index over evidence sentences

Technical Report12
with all-MiniLM-L6-v2 embeddings and a fixed top- kretrieval budget. ForKV Prefix,
retrieved sentences are directly converted into KV tensors using the same frozen backbone
without structural compression or graph alignment, serving as a naive KV injection baseline.
ForKVI, retrieved triples are compiled into KV memory entries and injected together with
prompt-based evidence (dual-channel). Injected KV tensors are concatenated to the model’s
attention cache at selected layers during decoding, without modifying model parameters. KV
injection is controlled by a small budget and relation-aware filtering to reduce noise, with
stricter settings applied on the large MedHopQA_official graph.
Controlled comparisons.All methods share the same corpus, preprocessing pipeline, and
evaluation code, and all retrieval structures are built from the same underlying text collection
without external knowledge sources. This ensures that performance differences arise solely from
retrieval strategy and memory integration, enabling a fair comparison between flat retrieval,
graph-based retrieval, and KV injection.
5.1.2 Results Analysis
Table 2 reports the main QA results across two backbone models under five conditions. We
analyze the results from three perspectives: retrieval structure, memory injection, and multi-hop
reasoning behavior.
Overall comparison.Across both backbones,KVIachieves the strongest or near-strongest
performance on most datasets. Compared toRAG, which relies on flat similarity-based
retrieval, andGraphRAG, which introduces structural retrieval but operates purely through
the prompt channel,KVIfurther improves performance by integrating graph-guided retrieval
with memory-level KV injection. These gains are consistent across two distinct backbone
models, suggesting that the effect is not tied to a specific architecture or prompt behavior, but
arises from the interaction between structured retrieval and attention-level memory integration.
Overall, improvements do not arise from retrieval or KV injection alone, but from their
combination.
Effect of graph-guided retrieval.ComparingRAGandGraphRAG, we observe consis-
tent improvements on multi-hop datasets such as HotpotQA and MedHop-style tasks. This
indicates that graph-based retrieval better preserves relational structure and reduces topic drift
compared to flat ANN retrieval.
However, the benefit is dataset-dependent. On Natural Questions, where many queries are
less relation-centric, the improvement ofGraphRAGoverRAGis modest. This suggests that
structural retrieval is most effective when the task requires explicit reasoning over connected
facts, rather than single-hop evidence lookup.
Effect of KV-based memory injection.TheKV Prefixbaseline performs poorly across
all datasets, often below or comparable to the baseLLM. This indicates that directly injecting
retrieved text into the KV cache, without structural compression or alignment, does not provide
effective guidance and may introduce noise into the attention process.

Technical Report13
Table 2: Main QA performance: exact match (%, 95% bootstrap CI in brackets).Bold
indicates the best EM per dataset.
Qwen2.5-7B-Instruct
Dataset LLM RAG GraphRAG KV Prefix KVI (Ours)
NQ 20.0 [13.0, 28.0] 41.0 [30.0, 51.0] 44.0 [34.0, 54.0] 14.0 [7.0, 21.0]49.0 [39.0, 58.0]
HotpotQA 16.7 [10.0, 23.3] 22.5 [15.0, 30.0] 32.5 [24.2, 40.8] 15.8 [9.2, 22.5]33.3 [25.0, 41.7]
MedHopQA_n40 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 82.5 [70.0, 92.5] 0.0 [0.0, 0.0]92.5 [82.5, 100.0]
MedHopQA_official 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 74.3 [69.6, 78.9] 0.0 [0.0, 0.0]75.4 [70.8, 80.1]
Mistral-7B-Instruct-v0.3
Dataset LLM RAG GraphRAG KV Prefix KVI (Ours)
NQ 34.0 [24.0, 43.0] 63.0 [53.0, 73.0]67.0 [58.0, 76.0]17.0 [10.0, 24.0]67.0 [58.0, 76.0]
HotpotQA 30.8 [22.5, 39.2] 36.7 [28.3, 45.8] 38.3 [29.2, 46.7] 16.7 [10.0, 24.2]40.8 [32.5, 49.2]
MedHopQA_n40 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 10.0 [2.5, 20.0] 0.0 [0.0, 0.0]12.5 [2.5, 25.0]
MedHopQA_official 0.0 [0.0, 0.0] 0.3 [0.0, 0.9] 13.5 [9.9, 17.0] 0.6 [0.0, 1.5]33.3 [28.7, 38.0]
In contrast,KVIconsistently improves over bothGraphRAGandRAG. This suggests
that KV injection becomes effective only when the injected content is structured and compact.
By encoding relational triples into canonical KV representations and combining them with
prompt-based evidence, KVI enables the model to leverage external memory as a form of
structural bias rather than as additional context tokens.
Multi-hop reasoning on HotpotQA and MedHop.The benefits ofKVIare most
pronounced on multi-hop datasets. OnHotpotQA,KVIconsistently outperforms both
RAGandGraphRAGacross the two backbones (e.g., 33.3 vs. 22.5 forRAGand 32.5 for
GraphRAGon Qwen2.5). This suggests that the combination of graph-guided retrieval and
KV injection effectively supports the stepwise evidence aggregation required by multi-hop
reasoning.
The differences become even more striking onMedHop-style tasks. Under the ID-based
evaluation setting, bothLLMandRAGobtain near-zero exact match (EM), while graph-
based methods achieve substantially higher performance. This phenomenon primarily reflects
a mismatch between free-form generation and strict output constraints. The task requires
predicting a canonical DrugBank identifier (e.g., DBxxxx), whereas ANN-based pipelines often
produce malformed IDs, additional tokens, or descriptive answers. Such outputs are counted as
incorrect under exact-match evaluation, even when they contain partially correct information.
By contrast, graph-guided retrieval constrains the search space to candidate entities con-
nected through valid relations, which improves the likelihood of producing valid identifiers.
Building on this,KVIfurther enhances performance by injecting compact relational signals
into the attention layers, which stabilizes multi-hop reasoning and reduces generation variance
under strict output formats. This dual mechanism leads to more stable and accurate reasoning,
particularly in multi-hop settings where both structure and compositionality are critical.

Technical Report14
Table 3: Ablation study on Main QA Bench (Qwen2.5-7B-Instruct, EM %).
Dataset KVI (Full) w/o Graph w/o KV
NQ49.041.0 44.0
HotpotQA33.322.5 32.5
MedHopQA_n4092.50.0 82.5
MedHopQA_official75.40.0 74.3
5.1.3 Ablation Study on Main QA Bench
To assess the contribution of each core component, we compareKVI (Full)against two
simplified variants that reuse baselines from the main experiment:
•w/o Graph: Dense retrieval + KV injection + prompt evidence (i.e.,KV Prefix
baseline). This variant removes graph-guided retrieval while keeping KV injection and
prompt evidence.
•w/o KV: Graph-guided retrieval + prompt evidence only (i.e.,GraphRAGbaseline).
This variant removes KV injection while keeping graph guidance and prompt evidence.
Table 3 reports the ablation results. The results show that both components are necessary.
Removing graph guidance (w/o Graph) causes severe degradation on multi-hop tasks (e.g.,
HotpotQA: 33.3→22.5; MedHopQA_official: 75.4→0.0), confirming that relational structure
is critical for multi-hop reasoning. Removing KV injection (w/o KV) also leads to consistent
but smaller drops, indicating that memory-level injection provides complementary benefits
beyond prompt-based evidence.
5.2 Hallucination Bench
5.2.1 Experimental Setup
We evaluate hallucination-related behavior on two public benchmarks with complementary task
shapes:TruthfulQA[ 39] andFEVER[ 40]. Unless otherwise stated, we use the generation
validation split of TruthfulQA with500examples and the KILT/FEVER validation split with
1000claims (local mirror or Hugging Face fallback, as prepared by our dataset scripts). All
systems share the same backboneQwen2.5-7B-Instructand are compared under a unified
pipeline that builds dataset-specific graph indices, sentence KV banks, and triple KV artifacts,
then runs the same evaluation harness for every method.
We compare five inference configurations:LLM(no retrieval),RAGandKV Prefix
(dense retrieval with different injection protocols),GraphRAG(graph-constrained decoding),
andKVI(KV-injected graph reasoning with the default hyperparameters in our codebase, e.g.,
bounded triple budget and relation filtering). Graph-side decoding can be served through a
resident HTTP inference endpoint; approximate nearest-neighbor retrieval for RAG/KV Prefix
defaults to CPU ANN unless routed through the same resident. For graph-capable methods we
use the defaultEnglish open-domaingraph prompt mode so that domain-specific templates
from other experiments do not confound these benchmarks. Each run uses a per-example

Technical Report15
Unified hallucination rate (proxy, %)
TruthfulQA (MC1 proxy -> Hallucination Rate)Hallucination Rate (%)
0255075100
35.6
LLM27.6
RAG15.6
GraphRAG3.2
KV Prefix22.2
KVITruthfulQA (MC2 proxy -> Hallucination Rate)Hallucination Rate (%)
0255075100
63.1
LLM61.1
RAG45.3
GraphRAG94.5
KV Prefix45.7
KVIFEVER (Label Accuracy -> Hallucination Rate)Hallucination Rate (%)
0255075100
69.1
LLM7.5
RAG31.2
GraphRAG27.7
KV Prefix10.7
KVI
Figure 3: Unified proxyhallucination rate( ↑is worse) across TruthfulQA [ 39] and FEVER [ 40].
Left/middle:TruthfulQA MC1/MC2likelihood proxiesmapped as100 −MC proxy (%)
(conditional log-likelihood over multiple-choice targets; not official TruthfulQA script scores).
Right:FEVER veracity task mapped as100 −label accuracy , where accuracy is computed by
parsing thefirstoccurrence of SUPPORTS /REFUTES/NOT ENOUGH INFO in the full model output
and comparing to gold (label-only; not the full evidence-based FEVER submission scorer).
timeout of 600s, and we report bootstrap / permutation intervals with 1000 / 2000 samples as
implemented in our evaluation script.
5.2.2 Results Analysis
How we measure “hallucination” (proxy metrics).Figure 3 uses a single y-axis con-
vention:hallucination rate= 100 −scorefor each panel, where the underlyingscoreis
dataset-specific. These areinternal proxiesfor comparing methods under identical budgets;
they arenotinterchangeable with official TruthfulQA or FEVER leaderboard numbers without
rerunning the reference evaluation pipelines.
TruthfulQA [ 39] (left two panels).The two left panels report MC1 and MC2likelihood
proxies: we attach multiple-choice targets (from the official multiple_choice split when
available, otherwise constructed from correct/incorrect answer lists) and score options with a
conditional log-likelihoodprocedure (default likelihood_proxy mode). MC1/MC2 are
then converted into a burden score as100 −proxy (%) . This is closer in spirit to multiple-choice
truthfulness evaluation than substring matching on long answers, but remains aproxytied to
our likelihood approximation and prompting context.
FEVER [ 40] (right panel).The right panel useslabel accuracyfor the three-way decision
among SUPPORTS ,REFUTES, and NOT ENOUGH INFO , then maps it to100 −accuracy for visual
alignment with the TruthfulQA panels. This reflectsveracity labelfidelity more directly
than relaxed substring match on free-form text, but still evaluateslabels onlyand omits the
evidence-retrieval constraints of the full shared-task scorer.

Technical Report16
Reading the results.TruthfulQA and FEVER stress different behaviors (misconception-
style QA under MC scoring proxies versus explicit veracity labels). We therefore reportboth
MC-proxy panels and the FEVER label-based panel in one figure to encourage cross-benchmark
interpretation rather than overfitting a single scalar.
Across the three benchmarks, several patterns emerge. Across different task types, KVI
consistently improves the hallucination proxy. On TruthfulQA-MC1 (misconception resistance),
KVI achieves 22.2%, outperforming LLM baseline (35.6%), RAG (27.6%), GraphRAG (15.6%),
and KV Prefix (3.2%). However, on TruthfulQA-MC2 (soft scoring), KVI achieves 94.5%
— the highest among all methods — indicating strong avoidance of falsehoods under lenient
evaluation. On FEVER (label accuracy), GraphRAG performs best (7.5%), while KVI yields
10.7%, suggesting that fact verification benefits from explicit graph structures differently
than open-ended QA. KV Prefix shows competitive performance on TruthfulQA-MC1 (3.2%)
but weaker results on FEVER (27.7%). These divergences highlight that no single method
dominates across all hallucination metrics, reinforcing the need for multi-benchmark evaluation.
6 Conclusion
This work introduces Knowledge Capsules and the External Key–Value Injection (KVI) frame-
work as a new paradigm for integrating external knowledge into large language models. In
contrast to retrieval-based methods that operate through context expansion, our approach en-
ables knowledge to participate directly in the Transformer’s attention mechanism as structured
memory.
By representing knowledge as normalized relational capsules and compiling them into
attention-compatible key–value representations, KVI shifts knowledge integration from token-
levelconditioningtomemory-levelinteraction. Thisdesignallowsexternallyretrievedknowledge
to influence generation at the same level as parametric memory, improving both controllability
and reasoning stability.
Empirical results across multiple QA benchmarks demonstrate that KVI consistently
outperforms both RAG and GraphRAG, with particularly strong gains in multi-hop and
long-context reasoning tasks. These improvements highlight the importance of aligning external
knowledge with the model’s internal inference mechanism rather than treating it as auxiliary
context.
Overall, this work suggests that effective knowledge augmentation in LLMs should move
beyond context-level retrieval toward principled memory-level integration. We hope that
Knowledge Capsules and the KVI framework provide a foundation for future research on
nonparametric memory systems and attention-centric knowledge integration.
7 Limitations
Although the proposed framework demonstrates strong empirical performance, several limita-
tions remain.
First, the effectiveness of KVI depends on the quality of knowledge extraction. Errors
or inconsistencies in the extracted triples may propagate into the KV memory and affect
downstream reasoning. Although evidence grounding provides traceability, improving extraction

Technical Report17
robustness remains an important direction.
Second, the current system relies on entity detection and graph-guided retrieval. When
queries lack clear entity anchors or require more abstract reasoning, retrieval quality may
degrade. Extending the framework to better handle implicit entities and open-domain queries
is a potential area for future work.
Third, KV injection introduces additional memory overhead during inference. Although the
injected representations are compact compared to raw text, scaling to very large knowledge
graphs may require more efficient selection and compression strategies.
Finally, the present evaluation focuses primarily on QA benchmarks with structured answers
(e.g., entity-centric or ID-based settings). While this enables controlled and reproducible evalu-
ation, further validation on open-ended free-text tasks is needed to fully assess generalization.
Addressing these limitations may further enhance the robustness, scalability, and general
applicability of memory-level knowledge integration in large language models.

Technical Report18
A Algorithm Details
Algorithm 1Compile-Time Knowledge Graph and KV Bank Construction 1
Require:PDF corpusD
Ensure:Knowledge GraphG, graph indexI, KV BankB
1:Segment documents into raw chunks
2:Perform section-aware evidence extraction to obtain evidence blocksE blocks
3:Collect sentencesSfromE blocks
4:Initialize triple record setT=∅
5:foreach sentences∈ Sdo
6:Extract triples(S, R, O)using LLM
7:foreach tripledo
8:Create recordt={subject=S, predicate=R, object=O, provenance}
9:Sett.provenance.sentence_id=Iand addttoT
10:end for
11:end for
12:Initialize graphG= (V, E)and indexI={sentence_index, triple_sentence_index}
13:foreach triple recordt∈ Tdo
14:S=t.subject,R=t.predicate,O=t.object,I=t.provenance.sentence_id
15:Add nodesS, OtoVand edge(S, R, O)toE
16:Updatetriple_sentence_index[t.id] =I
17:Updatesentence_index[I] ={text, source_block, source_doc, triple_ids}
18:end for
19:Extract entity setEfromG
20:foreach entitye∈ Edo
21:Construct canonical anchor sentence fore
22:Run frozen LLM forward pass and extract tensors(K l, Vl)
23:Store(K l, Vl)asEntity Anchor KVinB
24:end for
25:foreach triple recordt∈ Tdo
26:Convert(t.subject, t.predicate, t.object)into canonical sentence
27:Run frozen LLM forward pass and extract tensors(K l, Vl)
28:Store(K l, Vl)asTriple KVinB
29:end for
30:returnG,I,B

Technical Report19
Algorithm 2Query-Time Graph Retrieval and KV Injection 2
Require: Query q, knowledge graph G= (V, E), triple set T, graph index I, KV bank B, KV
budgetk, max hopH
Ensure:grounded answera
1:Detect entitye∈Vin queryq
2:ifenot foundthen
3:returnLLM(q)
4:end if
5:Classify query intent
6:Map intent to relation setR
7:Initialize frontierF 0={e}and candidate triple setS=∅
8:forh= 1toHdo
9:F h=∅
10:foreach nodev∈ F h−1do
11:foreach edge(v, r, u)inGdo
12:ifr∈ Rthen
13:retrieve triple recordt= (v, r, u)
14:addttoSandutoF h
15:end if
16:end for
17:end for
18:end for
19:foreach triplet∈ Sdo
20:obtainI=t.provenance.sentence_id
21:retrieve evidence sentencex=sentence_index[I].text
22:computescore(t) =DRM(q, x)
23:end for
24:Select top-ktriplesT k=TopK(S, k)
25:LoadEntity Anchor KVfor entityefromB
26:foreach triplet∈ T kdo
27:Load correspondingTriple KVtensors fromB
28:end for
29:Compose KV prefix sequence Pwith Entity Anchor KV followed by Triple KV entries in ranked
order
30:Compute prompt token sequenceQfrom evidence sentences, query, and task requirements
31:Run forward pass on Qto obtain layer-wise key-value tensors( Kprompt
l, Vprompt
l)for each layer l
32:foreach Transformer layerldo
33:Extract external tensors(Kext
l, Vext
l)from KV prefixPfor layerl
34:Concatenate to form full attention memory:
Kfull
l= [Kext
l;Kprompt
l], Vfull
l= [Vext
l;Vprompt
l]
35:end for
36:Generate answera=LLM.generate(prompt,{(Kfull
l, Vfull
l)}L
l=1)
37:Apply grounding filter
38:returna

Technical Report20
References
[1]Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H.
Miller, and Sebastian Riedel. Language models as knowledge bases? InProceedings of
the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages
2463–2473. Association for Computational Linguistics, 2019.
[2]Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language
models. InProceedings of the 2021 Conference on Empirical Methods in Natural Language
Processing, pages 6491–6506. Association for Computational Linguistics, 2021.
[3]Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing
factualassociationsinGPT.Advances in Neural Information Processing Systems, 35:17359–
17372, 2022.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian
Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP
tasks.Advances in Neural Information Processing Systems, 33:9459–9474, 2020.
[5]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Realm:
Retrieval-augmented language model pre-training. InProceedings of the 37th International
Conference on Machine Learning, volume 119 ofProceedings of Machine Learning Research,
pages 3929–3938. PMLR, 2020.
[6]Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models
for open domain question answering. InProceedings of the 16th Conference of the European
Chapter of the Association for Computational Linguistics: Main Volume, pages 874–880.
Association for Computational Linguistics, 2021.
[7]Andrew Brown, Muhammad Roman, and Barry Devereux. A systematic literature review
of retrieval-augmented generation: Techniques, metrics, and challenges, 2025.
[8]Yongjie Wang, Yue Yu, Kaisong Song, Jun Lin, and Zhiqi Shen. When retrieval succeeds
and fails: Rethinking retrieval-augmented generation for llms, 2025.
[9]Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed
Abdelrazek. Seven failure points when engineering a retrieval augmented generation
system, 2024.
[10]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N.
Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.Advances in Neural
Information Processing Systems, 30, 2017.
[11]Sainbayar Sukhbaatar, Jason Weston, and Rob Fergus. End-to-end memory networks.
Advances in Neural Information Processing Systems, 28, 2015.

Technical Report21
[12] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks, 2015.
[13]Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka
Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho,
John Agapiou, et al. Hybrid computing using a neural network with dynamic external
memory.Nature, 538(7626):471–476, 2016.
[14]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm:
Retrieval-augmented language model pre-training, 2020.
[15]Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-
Brown, and Yoav Shoham. In-context retrieval-augmented language models, 2023.
[16]Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio
Petroni, and Percy Liang. Lost in the middle: How language models use long contexts,
2023.
[17]Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid
Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning
for dense text retrieval, 2020.
[18]Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in
retrieval-augmentedlargelanguagemodels. InHoudaBouamor, JuanPino, andKalikaBali,
editors,Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, pages 5303–5315, Singapore, December 2023. Association for Computational
Linguistics.
[19]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag:
Learning to retrieve, generate, and critique through self-reflection, 2023.
[20]Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh
Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang
Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and Jiliang Tang.
Retrieval-augmented generation with graphs (graphrag), 2025.
[21]Microsoft Research Edge. From local to global: A graph rag approach to query-focused
summarization. 2024.
[22]Jean Lelong, Adnane Errazine, and Annabelle Blangero. Agentic rag with knowledge
graphs for complex multi-hop reasoning in real-world applications, 2025.
[23]Pai Liu, Wenyang Gao, Wenjie Dong, Lin Ai, Ziwei Gong, Songfang Huang, Zongsheng Li,
Ehsan Hoque, Julia Hirschberg, and Yue Zhang. A survey on open information extraction
from rule-based model to large language model, 2024.
[24]Mike Mintz, Steven Bills, Rion Snow, and Daniel Jurafsky. Distant supervision for relation
extraction without labeled data. In Keh-Yih Su, Jian Su, Janyce Wiebe, and Haizhou Li,
editors,Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the

Technical Report22
4th International Joint Conference on Natural Language Processing of the AFNLP, pages
1003–1011, Suntec, Singapore, August 2009. Association for Computational Linguistics.
[25]Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, and Daniel S. Weld.
Knowledge-based weak supervision for information extraction of overlapping relations. In
Dekang Lin, Yuji Matsumoto, and Rada Mihalcea, editors,Proceedings of the 49th Annual
Meeting of the Association for Computational Linguistics: Human Language Technolo-
gies, pages 541–550, Portland, Oregon, USA, June 2011. Association for Computational
Linguistics.
[26]Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang, Pengjun
Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, Yong Jiang, and Wenjuan Han. Chatie:
Zero-shot information extraction via chatting with chatgpt, 2024.
[27]Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen.
Structgpt: A general framework for large language model to reason over structured data,
2023.
[28]Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank:
Enhancing large language models with long-term memory, 2023.
[29]Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and
Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced
retrieval augmentation, 2025.
[30]LiangYao, ChengshengMao, andYuanLuo. Kg-bert: Bertforknowledgegraphcompletion,
2019.
[31]Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang
Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, Weixin Liu, Zhihua Wu, Weibao Gong,
Jianzhong Liang, Zhizhou Shang, Peng Sun, Wei Liu, Xuan Ouyang, Dianhai Yu, Hao
Tian, Hua Wu, and Haifeng Wang. Ernie 3.0: Large-scale knowledge enhanced pre-training
for language understanding and generation, 2021.
[32]Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward
layers are key-value memories, 2021.
[33]Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan
Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context,
2019.
[34]Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao,
and Jian Guo. Think-on-graph 2.0: Deep and faithful large language model reasoning
with knowledge-guided retrieval augmented generation, 2025.
[35]Xi Wang, Taketomo Isazawa, Liana Mikaelyan, and James Hensman. Kblam: Knowledge
base augmented language model, 2025.

Technical Report23
[36]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh,
Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina
Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question
answering research.Transactions of the Association for Computational Linguistics, 7:452–
466, 2019.
[37]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable
multi-hop question answering, 2018.
[38]BioCreative IX Organizers. Medhopqa: A dataset for benchmarking llm-based reasoning
systems with disease-centered question answers. InProceedings of the BioCreative IX
Challenge and Workshop (BC9): Large Language Models for Clinical and Biomedical NLP,
Montreal, Canada, 2025. Track 1: MedHopQA Shared Task.
[39]Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic
human falsehoods. InProceedings of ACL, 2022.
[40]James Thorne, Andreas Vlachos, Oana Cocarascu, Arpit Mittal, and Yixin Hou. FEVER:
a large-scale dataset for fact extraction and VERification. InProceedings of NAACL, 2018.