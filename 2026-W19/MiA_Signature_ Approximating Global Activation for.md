# MiA-Signature: Approximating Global Activation for Long-Context Understanding

**Authors**: Yuqing Li, Jiangnan Li, Mo Yu, Zheng Lin, Weiping Wang, Jie Zhou

**Published**: 2026-05-07 15:25:02

**PDF URL**: [https://arxiv.org/pdf/2605.06416v1](https://arxiv.org/pdf/2605.06416v1)

## Abstract
A growing body of work in cognitive science suggests that reportable conscious access is associated with \emph{global ignition} over distributed memory systems, while such activation is only partially accessible as individuals cannot directly access or enumerate all activated contents. This tension suggests a plausible mechanism that cognition may rely on a compact representation that approximates the global influence of activation on downstream processing. Inspired by this idea, we introduce the concept of \textbf{Mindscape Activation Signature (MiA-Signature)}, a compressed representation of the global activation pattern induced by a query. In LLM systems, this is instantiated via submodular-based selection of high-level concepts that cover the activated context space, optionally refined through lightweight iterative updates using working memory. The resulting MiA-Signature serves as a conditioning signal that approximates the effect of the full activation state while remaining computationally tractable. Integrating MiA-Signatures into both RAG and agentic systems yields consistent performance gains across multiple long-context understanding tasks.

## Full Text


<!-- PDF content starts -->

MiA-Signature: Approximating Global Activation for
Long-Context Understanding
Yuqing Li1,2∗Jiangnan Li3∗Mo Yu4∗Zheng Lin1,2†
Weiping Wang1Jie Zhou3
1Institute of Information Engineering, Chinese Academy of Sciences
2School of Cyber Security, University of Chinese Academy of Sciences
3Pattern Recognition Center, WeChat AI, Tencent
4Hunyuan Team, Tencent
liyuqing@iie.ac.cn {jiangnanli,moyumyu}@tencent.com
Abstract
A growing body of work in cognitive science suggests that reportable conscious
access is associated withglobal ignitionover distributed memory systems, while
such activation is only partially accessible as individuals cannot directly access
or enumerate all activated contents. This tension suggests a plausible mechanism
that cognition may rely on a compact representation that approximates the global
influence of activation on downstream processing. Inspired by this idea, we in-
troduce the concept ofMindscape Activation Signature (MiA-Signature), a
compressed representation of the global activation pattern induced by a query. In
LLM systems, this is instantiated via submodular-based selection of high-level con-
cepts that cover the activated context space, optionally refined through lightweight
iterative updates using working memory. The resulting MiA-Signature serves
as a conditioning signal that approximates the effect of the full activation state
while remaining computationally tractable. Integrating MiA-Signatures into both
RAG and agentic systems yields consistent performance gains across multiple
long-context understanding tasks.
1 Introduction
Recent advances in large language models (LLMs) and retrieval-augmented systems have significantly
improved performance on knowledge-intensive tasks by combining parametric knowledge with
external memory. A dominant paradigm has emerged in which a query is processed, relevant
documents are retrieved, and reasoning is performed over the retrieved context. Despite its empirical
success, this paradigm implicitly assumes that reasoning can be grounded in a relatively small set of
locally retrieved evidence.
However, this assumption appears at odds with insights from cognitive science. A growing body
of work suggests that reportable conscious access is associated withglobal ignition—a transient,
large-scale activation over distributed memory systems [ 6,7,30]. At the same time, such activation
is only partially accessible: as human beings, we cannot directly access or enumerate all activated
contents. Instead, cognition appears to rely on a compact internal representation that approximates
the global influence of activation on downstream processing [25, 26, 30].
Motivated by this perspective, we argue that memory access in LLM systems can be more effectively
modeled as a two-stage process: global activation followed by representation. Rather than directly
∗Equal contribution.
†Corresponding author.
Preprint.arXiv:2605.06416v1  [cs.CL]  7 May 2026

mapping queries to a small set of retrieved documents, a query first induces a global activation pattern
over a semantic memory space, which is then approximated by a tractable representation used to
guide downstream computation.
To operationalize this idea, we introduce the notion of amindscape, a global semantic memory
space over which activation can be defined. Building on this, we propose theMindscape Activation
Signature (MiA-Signature), a compressed representation of the activation pattern induced by a query.
In practice, MiA-Signatures are constructed via submodular-based selection of high-level concepts
that cover the activated context space, optionally refined through lightweight iterative updates using
working memory. This representation serves as a conditioning signal that captures a holistic view of
relevance, beyond what is available from local retrieval alone.
This perspective leads to a shift in how memory is integrated into reasoning systems. Instead of
treating retrieval as the primary interface to memory, we treat activation as the underlying process
and signatures as its usable representation. This allows downstream components—such as retrievers,
rerankers, or reasoning modules—to operate under a more globally informed semantic context,
improving coherence and robustness in long-context settings.
Remark: Supporting overcomplete memory.In realistic settings, memory management systems
may produce a large set of memory items, e.g., generated by sleep-time consolidation [ 1], sometimes
even exceeding the number of raw input items, with substantial redundancy and overlap. By selecting
a minimal supporting set that covers the global activation pattern, MiA-Signatures naturally cooperate
with suchovercomplete memory. This allows downstream computation to operate on a holistic
approximation of the activated context without incurring the complexity of excessively long inputs
recalled from memory.
We evaluate this approach by integrating MiA-Signatures into both retrieval-augmented generation
(RAG) pipelines and agentic systems. Empirical results show consistent performance gains across
multiple long-context understanding tasks. These improvements suggest that approximating global
activation provides a more effective interface to memory than relying solely on local retrieval.
In summary, our contributions are as follows:
•We introduce a cognitively inspired perspective that models memory access as global
activation over a mindscape followed by compact representation.
•We propose theMindscape Activation Signature (MiA-Signature)as a practical instantia-
tion of this idea in LLM systems, providing a compact query-conditioned global state for
retrieval, generation, and agentic memory.
•We develop a submodular-based construction method, optionally enhanced with lightweight
iterative refinement, and demonstrate that integrating MiA-Signatures into both RAG and
agentic systems yields consistent improvements on long-context understanding tasks.
We believe this work provides a step toward bridging cognitive insights and practical system design,
highlighting the importance of global activation in memory-driven reasoning.
2 Related Work
2.1 Evidence Supporting Signatures
Global workspace and global ignition.The idea that conscious processing involves a form
of global information sharing originates from the Global Workspace Theory (GWT) [ 3,4], which
proposes that information becomes consciously accessible when it is broadcast to a set of distributed
cognitive modules. This framework was later grounded in neurobiological mechanisms through
the Global Neuronal Workspace (GNW) theory [ 6–8], which associates conscious access with a
nonlinearglobal ignitionprocess—a sudden, large-scale activation sustained by long-range recurrent
connectivity. These works establish the existence of global activation as a key substrate of conscious
processing.
Limits of access and partial awareness.While GNW posits global activation, subsequent work
highlights that such activation is only partially accessible. Recurrent Processing Theory (RPT) [ 19]
distinguishes between local recurrent processing and global broadcasting, suggesting that not all
2

activated representations reach reportable awareness. Empirical studies on partial awareness and
graded consciousness [ 18,25] further support the view that individuals cannot directly access or
enumerate all activated contents, even when global activation occurs. These findings point to a gap
between the existence of global activation and the form in which it is available for cognition.
Integration and compression of global states.Complementary to GNW, Integrated Information
Theory (IIT) [ 30,31] emphasizes that conscious states are highly integrated and structured, rather
than collections of independent elements. From this perspective, global brain states are intrinsically
compressed representations of distributed activity. Although IIT differs from GNW in its theoretical
foundations, both suggest that cognition operates on representations that reflect global structure rather
than raw activation patterns.
From global activation to usable representations.Despite these advances, existing theories do not
explicitly specify how globally distributed activation is transformed into representations that can guide
downstream computation. In parallel, current LLM-based systems, including retrieval-augmented
generation (RAG) pipelines, typically access memory through local retrieval mechanisms, implicitly
assuming that relevant information can be captured by a small set of retrieved documents. This stands
in contrast to the cognitively motivated view that reasoning is shaped by global context.
Our perspective.In this work, we build on these lines of research by proposing that cognition
operates on a compact representation that approximates the influence of global activation. We
introduce theMindscape Activation Signature (MiA-Signature)as a computational instantiation
of this idea: a compressed representation of a global activation pattern over a semantic memory
space. Rather than modeling memory access as direct retrieval, our framework treats it as a two-stage
process—global activation followed by signature-based approximation—providing a bridge between
cognitive theories of global processing and practical LLM system design.
2.2 Related Systems: RAG, Memory, and Long-Context Agents
Retrieval as local evidence access.A dominant line of work improves memory access by making
retrieval more iterative, selective, or reasoning-aware, while still treating retrieval itself as the primary
interface to external memory. IRCoT [ 32] interleaves reasoning with retrieval, and FLARE [ 14]
triggers retrieval when generation becomes uncertain. Self-RAG [ 2], Adaptive-RAG [ 13], and
DeepRAG [ 9] further study when and how retrieval should be invoked. More recent systems such
as Search-o1 [ 21] and Search-R1 [ 15]expose search as an explicit reasoning action, allowing large
reasoning models to interleave thinking with multi-step retrieval and evidence refinement. Despite
these advances, the state propagated across steps remains largely local: the current query, reasoning
trace, or retrieved passages. Memory access is therefore still framed primarily as iterative evidence
lookup rather than as an approximation of a global activated context.
Structured retrieval over long documents.Another line of work improves long-document retrieval
by constructing richer external structures over the source. RAPTOR [ 28] organizes documents into a
hierarchy of recursive summaries, enabling retrieval at multiple levels of abstraction. HippoRAG [ 11]
builds a graph-based memory index inspired by hippocampal retrieval. These methods highlight the
importance of global organization for long-context reasoning, moving beyond retrieval over isolated
flat chunks. Our work is complementary: rather than treating such structures only as static retrieval
substrates, we use them as a mindscape over which a query can induce a compact, query-conditioned
activation signature. This signature can then guide retrieval, condition generation, and evolve during
multi-step reasoning.
Memory-augmented long-context agents.Recent long-context agents go further by equipping
the model with explicit memory states while reading or navigating large inputs. ReadAgent [ 20]
compresses long documents into gist memories, and ComoRAG [33] emphasizes stateful reasoning
through a dynamic memory workspace. Moreover, MemAgent [ 35] and ReMemR1 [ 29]study how
memory can be updated, revisited, or controlled across long reasoning trajectories. These systems are
highly relevant to our setting because they move beyond one-shot retrieval to persistent external state.
However, their focus is mainly on how to store, revisit, or manage memory during reasoning. Our
focus is orthogonal: before local evidence is selected or revisited, we ask how theglobal influenceof
a query over a semantic memory space can be approximated in a tractable representation.
3

…chunk1 chunk2 chunkn
memory items
Embedding
Model 1query
 chunks
initial signature
queryEmbedding 
Model 2
chunks
signature
(rewritten) 
query
answerLoop
evidence memory evidence memorydecision
Signature Initialization
answerRAG
initial signatureAgent Loop
{ chunk1: , chunk2:
 ,
, …, chunkn : 
 } chunk3: 
chunk→memo . map: Memory Construction
Refine
Answer
activated memory
activated memory
activated memory
Figure 1: Overview of MiA-Signature. A query first induces a broad activation pattern over the
mindscape; MiA-Signature compresses this activated region into a compact, query-conditioned global
signal, which then guides retrieval and reasoning in both static RAG and an iterative agent.
3 Method
We first formalize the mindscape, the query-induced activation pattern, and the MiA-Signature as a
compact surrogate of that activation (Sec. 3.1). We then instantiate the same signature interface in
two settings: a static one used once in standard RAG, and a dynamic one maintained as an evolving
memory state in an agent loop (Sec. 3.2).
3.1 Preliminaries: MiA-Signature as an Activation Surrogate
AMiA-Signatureis a compact, query-conditioned global state that approximates the memory
region activated by a query, and that exposes this global signal to downstream retrieval and
reasoning.
3.1.1 Mindscape, Activation, and Signature
Mindscape.Let Ddenote a long source, such as a novel, a dialogue history, or a document
collection. We assumeDis associated with a memory pool:
M(D) ={m 1, . . . , m N},
where each miis grounded in a subset of finer-grained evidence from the source (e.g., passages,
chunks). We refer to this organized memory substrate as themindscape. Memory pools of this kind
often contain redundancy, overlap, and multiple levels of abstraction; summaries, extracted entities,
and offline-consolidated memories [ 1] may coexist. This motivates a compact representation of the
globally relevant region rather than direct reliance on the full pool.
Activation.Given a query q, memory access need not be limited to a few locally matched passages.
The query typically brings into play a broader semantic region of the mindscape. We represent this
query-induced activation as
aq:M(D)→R ≥0,(1)
where aq(m) measures how strongly mbelongs to the activated region. In practice, aqis only
approximately observed through retrieval. This is consistent with the broader view that globally
activated context may be only partially accessible to downstream processing [ 6,25], and it motivates
constructing a compact, usable surrogate of this global signal.
MiA-Signature.To make the activation usable, we operate at a higher level of abstraction within
the mindscape. Let H(D) ={h 1, . . . , h M} ⊆ M(D) denote a set of high-level memory units—e.g.,
session summaries or concept-level abstractions—obtained as a coarser-grained projection of M(D) .
For a query q, letHq⊆ H(D) be the subset supported by the activated region. We define the
MiA-Signature as a compact subset
σ⋆(q) = arg max
σ⊆H q,|σ|≤KF 
σ;q,H q
,(2)
4

whereFscores how well a candidate signature serves as a surrogate of the currently activated context,
favoring signatures that are relevant toq, cover the activated region, and avoid redundancy.
Importantly, σ⋆(q)is not intended as a shortened summary of D. It is a compact global state that
approximates which part of the mindscape has been activated by the query, and it is meant to coexist
with locally retrieved evidence rather than replace it. In the agent setting, the signature is further
refined as new evidence is consolidated, yielding an evolving global state rather than a one-shot
summary [26, 30].
3.1.2 Mindscape-aware Retrieval Interface
We use two retrievers with distinct roles, both taken from MiA-RAG [ 22]. The first, E1, is a query-
only retriever instantiated by SFT-Emb-8B,3used to obtain an initial view of the relevant memory
region before any signature is available. The second, E2, is a mindscape-aware retriever instantiated
by MiA-Emb-8B,4whose query representation is conditioned on both the input query and a global
memory signal. The retriever mechanism is stated in Appendix B.
In our framework, that global signal is instantiated by the current MiA-Signatureσ t, soE 2retrieves
with the pair (qt, σt):qtcarries the immediate search intent, while σtsupplies the current global
memory signal. As σtevolves, the retrieval distribution evolves with it, letting the system track a
changing view of the activated memory region.
3.2 Instantiating MiA-Signatures in RAG and Agentic Systems
MiA-Signature provides a common memory interface for two settings. In RAG, the signature is
constructed once and used as a fixed conditioning signal. In the agent setting, it is maintained as an
evolving global state and updated alongside a local evidence memory as new retrieval steps unfold.
3.2.1 Step-0Initialization: Submodular Selection for Global Coverage
Given a query q, we first perform a broad retrieval over fine-grained evidence units using the query-
only retriever E1. In all experiments, we retrieve the top- K0candidates with K0= 50 . Each candidate
is then mapped to its associated high-level memory unit, yielding a summary pool: H0(q)⊆ H(D).
This pool provides a coarse, memory-level view of the mindscape region activated by the query, but
can be redundant because many retrieved chunks may correspond to overlapping sessions or concepts.
A simple way to construct the initial signature isFirst- Ktruncation: deduplicate the summaries
according to the ranking induced by the step- 0retrieval and keep the first Ksum. This preserves the
local ordering of the initial retriever, but can underrepresent parts of the activated region that appear
later in the ranking. We instead select the initial signature with a coverage-aware objective:
σ0(q) = arg max
σ⊆H 0(q),|σ|≤K sumF 
σ;q,H 0(q)
,(3)
where Fbalances query relevance, coverage of the activated region, and diversity among selected
memory units. We optimize this set-selection objective with a greedy approximation. Thus, the initial
signature is chosen from the same pool as First- K, but by how well the selected summaries jointly
represent the activated region rather than by their inherited chunk order. Appendix A provides the
objective, greedy procedure, and comparison with First- Kinitialization. The resulting σ0serves as
the initial MiA-Signature.
3.2.2 Static Integration: Signature-Augmented RAG
In the RAG setting, the signature is constructed once and used as a fixed global conditioning signal.
Starting from σ0, we perform a second retrieval pass with the mindscape-aware retriever E2. Each
candidate evidence unitcis scored by
s(c|q, σ) = (1−α)s qry(c|q) +α s sig(c|σ),(4)
where sqry(c|q) measures query relevance, ssig(c|σ) measures consistency with the signature, and
α∈[0,1]controls the strength of the global signal (illustrated in Appendix B).
3https://huggingface.co/MindscapeRAG/SFT-Emb-8B
4https://huggingface.co/MindscapeRAG/MiA-Emb-8B
5

Algorithm 1MiA-Signature agent over a long source
Require: Query q; source Dwith memory pool M(D) and high-level memory set H(D) ; stopping budget
Nstop; query-only retrieverE 1; mindscape-aware retrieverE 2; update modelM upd; generatorM gen.
1:σ 0←INITSIGNATURE(q, D;E 1)// Eq. (3)
2:q0←q;E 0←∅
3:fort= 0toN stop−1do
4:P t←RETRIEVE(q t, σt;E2)// Eq. (4)
5:H t←SUMMARIES(P t)
6:(d t, qt+1, σt+1, Et+1)←UPDATE(q t, σt, Pt, Et,Ht;Mupd)
7:ifd t=ANSWERthen
8:returnM gen(q, P t, σt+1, Et+1)
9:end if
10:end for
11:returnM gen(q, P Nstop−1, σNstop, ENstop)
The top- Kevidence units under this score are passed to the generator. The signature does not replace
retrieved evidence; it changes the retrieval interface from query-only matching to query–signature
conditioning. When the generator can use global conditioning,σ 0is also included in the generation
input, either for an LLM with strong context-integration ability or for a smaller mindscape-aware
generator trained for this interface, such as MiA-Gen-14B [ 22]. Thus, static MiA-RAG preserves
the efficiency of a two-stage RAG pipeline while exposing a compact approximation of the activated
memory region to retrieval, and optionally to generation.
3.2.3 Dynamic Evolution: Iterative Signature Refinement
In the agent setting, the same query–signature retrieval interface is reused inside an iterative reasoning
loop. Starting from the initial signature σ0in Eq. (3), we set q0=qandE0=∅. At step t, the agent
retrieves chunks with the mindscape-aware retriever E2conditioned on the current pair (qt, σt), using
the score in Eq. (4). LetPtbe the retrieved chunks and let Ht⊆ H(D) be the associated high-level
memory units.
The state-update model then updates the agent state:
(dt, qt+1, σt+1, Et+1) =M upd(qt, σt, Pt, Et,Ht),(5)
where dtdecides whether to answer or continue retrieval. The rewritten query qt+1captures the next
local information need, the evidence memory Et+1stores grounded facts accumulated so far, and the
refined signature σt+1carries the updated global memory state. The agent therefore does not rely
on query rewriting alone; it navigates long-context memory through the joint evolution of the query,
local evidence memory, and global signature.
3.2.4 Signature-Grounded Answer Generation
When the agent decides to answer at step t, or when the refinement budget is exhausted, the generator
receives the original query, the latest retrieved evidence, and the updated memory state:
ˆy=M gen(q, P t, σt+1, Et+1).(6)
Generation remains grounded in local evidence while using the refined signature as the compact
global state produced by the loop.
4 Experiments
4.1 Experimental Setup
We evaluate MiA-Signatures in two long-context memory-access settings: a static RAG pipeline
and an iterative agent. The static setting tests a one-shot signature as a compact global conditioning
signal, while the agent setting tests whether the same interface remains useful as an evolving memory
state over multiple retrieval steps.
6

4.1.1 Datasets and Metrics
We evaluate on four long-context benchmarks covering multiple-choice QA, open-ended QA, multi-
hop QA, and claim verification. DetectiveQA [ 34] evaluates multiple-choice reasoning over detective
novels in English and Chinese. NarrativeQA [ 17] evaluates open-ended question answering over
narrative texts. NovelHopQA [ 10] evaluates multi-hop reasoning over long novel excerpts, and
NoCha [16] evaluates claim verification over full novels.
For DetectiveQA and NarrativeQA, we adopt aseries-book construction. Instead of treating each
novel as an independent source, we merge books from the same series into a single long document,
e.g., Agatha Christie’sMiss MarpleandHercule Poirotseries for DetectiveQA. The questions remain
tied to episode-specific evidence, but retrieval is performed over a larger memory space containing
related characters, events, and distractors. Appendix C.1 details the aggregation procedure, and
Appendix C.2 provides a single-book vs. series-book comparison showing such retrieval interference.
We use accuracy for multiple-choice QA, F1 score for open-ended QA, and accuracy together with
pair accuracy for NoCha. We also report Recall@10 when gold evidence annotations are available.
4.2 Implementation Details
Unless otherwise specified, the agent uses DeepSeek-V3.2 [ 24] as both the state update model Mupd
and the final answer generator Mgen. The agent runs for at most three refinement steps. At step 0, the
query-only retriever returns 50 candidate chunks; these chunks are mapped to high-level memory
units, from which at most five session summaries are selected to form the initial signature. Each
subsequent retrieval step returns 20 chunks. The dual-signal retrieval score uses α= 0.5 to balance
query relevance and signature consistency.
The high-level memory set H(D) is constructed offline by splitting each document into source-
order windows of W= 20 chunks and summarizing each window once using GPT-4o with a fixed
summary-construction prompt. The resulting session summaries are cached, query-independent,
and reused across all queries over the same document. Appendix A.1 gives the formal chunk-
to-summary mapping. Submodular selection follows Appendix A. In the static RAG setting, the
coverage-aware variant scores all three terms with BGE-M3 [ 5] CLS embeddings, using default
weights (λQ, λC, λD) = (0.3,0.4,0.3) . The First- Kvariant used by the agent reuses step- 0retriever
scores directly and invokes no additional encoder.
4.2.1 Baselines
We compare two families of systems: static RAG pipelines and iterative agents. The RAG experiments
isolate where the MiA-Signature is used in a retriever–generator pipeline, while the agent experiments
test whether the signature remains useful as an evolving memory state.
RAG methods.Each RAG system is reported as a retriever–generator pair.Query-only RAG
retrieves with the input query alone and does not use a MiA-Signature. We evaluate three query-only
variants:Qwen3-Emb / Qwen-14Bas a 14B-scale reference,Qwen3-Emb / DS-V3.2as a stronger
generator baseline, andMiA-Emb / DS-V3.2as a retriever-backbone control without signature
conditioning. In theMiA-Embconfiguration, the MiA-Signature is used to condition retrieval, while
the generator receives only the retrieved chunks.MiA-RAGfurther provides the same signature to
the generator, forming the full signature-aware RAG interface. We evaluate MiA-RAG with both
DS-V3.2 and MiA-Gen-14B [ 22]. All static signature-based methods use the same coverage-aware
submodular step-0initialization (§ 3.2.1).
Agent methods.All agent variants start from the same broad step- 0retrieval and use at most
three refinement steps.Agent w/o Sig.follows the same iterative retrieval process as MiA-Agent
but removes the signature from the agent state. We report two answer-time inputs for this baseline:
retrieved chunks only, and retrieved chunks plus accumulated evidence memory (EVI.).MiA-Agent
maintains an evolving signature σtand retrieves with (qt, σt)at each step. To separate retrieval-time
and generation-time effects, we vary the final generator input. All variants use the final retrieved
chunks; we additionally provide the final signature (SIG.), accumulated evidence memory (EVI.),
or both (SIG.+EVI.). These variants test whether the evolving signature only steers retrieval or also
helps answer generation. MiA-Agent initializes σ0with the lightweight First- Ksubmodular variant,
as it is later refined online.
7

Table 1:RAG results.MiA-Emb uses the MiA-Signature only for retrieval: the retriever is con-
ditioned on both the query and the signature, while the generator receives retrieved chunks only.
MiA-RAG uses the full signature-aware interface, where the same signature is used by both the
retriever and the generator. Avg. Perf. averages the main task metric of each benchmark, using
PairAcc for NoCha. Best final-task results are inbold.
Method Retriever GeneratorDetectiveQA (EN/ZH) NarrativeQA NovelHopQA NoCha Avg.
R@10Acc R@10F1 R@10F1 Acc / PairAcc R@10Perf.
Query-only RAG Qwen3-Emb Qwen-14B 24.7 / 29.3 50.7 / 56.7 48.5 36.6 34.3 35.8 64.3 / 31.8 36.6 39.5
Query-only RAG Qwen3-Emb DS-V3.2 24.7 / 29.3 58.7 / 68.0 48.5 41.8 33.7 37.0 74.6 / 49.2 36.4 47.8
Query-only RAG MiA-Emb DS-V3.2 34.0 / 42.7 59.3 / 76.0 51.3 41.1 36.6 38.0 81.0 / 61.9 42.1 52.2
MiA-Emb MiA-Emb (+sig) DS-V3.2 44.7 / 42.7 70.7 / 78.0 59.5 45.1 36.8 38.5 79.4 / 58.7 46.7 54.2
MiA-RAG MiA-Emb (+sig) MiA-Gen-14B (+sig) 44.7 / 42.7 70.7 / 73.3 59.5 48.0 36.8 27.9 71.4 / 47.6 46.7 48.9
MiA-RAG MiA-Emb (+sig) DS-V3.2 (+sig) 44.7 / 42.7 74.7 / 80.0 59.5 42.8 36.8 38.7 82.5 / 65.1 46.7 56.0
4.3 Main Results
We organize the experiments around three questions.
RQ1: Does conditioning retrieval on a MiA-Signature improve static RAG?Table 1 first
evaluates whether the MiA-Signature helps static RAG at the retrieval stage. The Qwen3-Emb rows
serve as general query-only baselines: they retrieve with the input query alone and do not have a
mechanism for using a global memory state. Previous studies also show that simply appending a
summary to a general embedding model can hurt retrieval, as the added global context may blur the
query focus rather than guide selection [ 22]. This motivates a signature-aware retriever rather than a
query-plus-summary shortcut.
Under the same retriever and generator backbone, conditioning retrieval on the MiA-Signature
improves average R@10 by 10.9% and average task performance by 3.8%. Since the generator input
remains the retrieved chunks only, the gain comes from changing how evidence is selected before
generation, rather than from giving the generator more context. The improvement is most meaningful
on DetectiveQA and NarrativeQA, where the answer often depends on a dispersed region of related
events, entities, or claims. In such cases, query-only retrieval can find locally plausible chunks while
missing the broader semantic region; the signature helps reduce this mismatch. NovelHopQA shows
a smaller gain, marking a boundary of this mechanism: the signature helps locate a relevant semantic
region, but multi-hop questions still require composing specific evidence chains that a compact global
state may not fully specify. These results support MiA-Signature as a retrieval-side memory interface.
It does not replace local evidence; it changes how local evidence is selected.
RQ2: Does the signature remain useful as memory access becomes iterative?Table 2 evaluates
whether the retrieval-side benefit of the MiA-Signature extends from static RAG to an iterative agent.
Compared withAgent w/o Sig.,MiA-Agentimproves retrieval recall on every benchmark with
retrieval annotations, with the clearest gains on DetectiveQA-ZH and NovelHopQA. Compared with
the static MiA-RAG reference, MiA-Agent largely matches or improves retrieval despite starting
from a lightweight First- Ksignature, suggesting that iterative signature updates can compensate for
a simpler initial state. This matters because iterative retrieval can otherwise become overly tied to the
current rewritten query. As the agent accumulates evidence, the local query may narrow or drift, while
the original problem may still require a broader activated memory region. MiA-Agent addresses
this by maintaining an evolving signature σtalongside the query and a working evidence memory.
The signature guides search at the global level, while the evidence memory preserves grounded facts
already retrieved.
These results extend the retrieval-side conclusion from RQ1: the MiA-Signature is not only useful
as a one-shot retrieval-conditioning signal, but also as a stable global state that keeps iterative
search aligned with the query-induced activated region across steps. The two memory states are not
interchangeable; we analyze their answer-time effects in RQ3.
RQ3: Which memory state should be exposed to the final generator?The answer-time ablations
show that retrieval-time and generation-time uses of memory should be separated. In static RAG,
MiA-RAGimproves overMiA-Emb, indicating that the signature can provide useful global context
to the generator in addition to guiding retrieval. However, this benefit is not automatic. The MiA-
8

Table 2:Agent results and answer-time ablation.All iterative agent variants use DeepSeek-V3.2
with a three-step refinement budget. The static MiA-RAG row is included as a non-iterative reference
using the same generator. All answer-time inputs include the final retrieved chunks; SIG. denotes the
final MiA-Signature, and EVI. denotes the accumulated evidence memory.
System Answer-time inputDetectiveQA (EN/ZH) NarrativeQA NovelHopQA NoCha
R@10Acc R@10F1 R@10F1 Acc PairAcc
Agent w/o Sig. Chunks 42.7 / 46.7 68.0 / 82.0 53.7 42.4 33.9 37.4 77.8 57.1
Agent w/o Sig. Chunks + EVI. 42.7 / 46.7 76.0 / 80.0 53.7 43.4 33.9 36.4 84.9 69.8
MiA-RAG (static) Chunks + SIG. 44.7 / 42.7 74.7 / 80.0 59.5 42.8 36.838.7 82.5 65.1
MiA-Agent Chunks 46.7 / 52.7 68.7 / 81.3 59.145.3 39.338.7 80.2 61.9
MiA-Agent Chunks + SIG. 46.7 / 52.776.7/ 82.0 59.1 44.9 39.3 37.1 83.3 68.3
MiA-Agent Chunks + EVI. 46.7 / 52.7 73.3 /86.0 59.1 43.6 39.3 36.2 83.3 66.7
MiA-Agent Chunks + SIG. + EVI. 46.7 / 52.7 73.3 / 80.0 59.1 44.3 39.3 35.6 85.7 71.4
Gen-14B variant achieves the best NarrativeQA F1, but does not dominate across benchmarks. This
suggests that answer-time use of the signature depends on both the task and the generator’s ability
to exploit it. Moreover, the agent ablations make this distinction clearer. The final signature and
the working evidence memory encode different types of information. The signature summarizes the
broader activated memory region, while the evidence memory preserves grounded facts accumulated
during the agent loop. On NoCha, where local factual continuity is important, exposing both states
gives the best result. By contrast, on NarrativeQA and NovelHopQA, the best MiA-Agent variants use
retrieved chunks alone. Once the retrieved chunks already contain a usable answer path, additional
memory state may distract the generator rather than provide useful structure.
Taken together, retrieval benefits from the signature more consistently than generation does. The
signature is a reliable search-guiding state, but its answer-time value is selective. It helps when
global constraints are needed to interpret local evidence, and it can be unnecessary when the retrieved
chunks already provide a direct and composable evidence path.
4.4 Analysis
We include two targeted studies to further examine the mechanism behind the main results. First,
Appendix A.5 compares the two submodular initializers, Coverage-aware and First- K, under the
static RAG pipeline. Both variants use the same step- 0candidate pool, so the comparison isolates
whether coverage-aware selection provides benefit beyond simply taking the ranking prefix. Second,
Appendix D studies query rewriting in the agent loop. We find that rewriting is best treated as a
control knob rather than the core mechanism. It helps when refinement should narrow the search, but
can be harmful when the task requires preserving multiple evidence paths. Accordingly, we keep
the query fixed on NovelHopQA and rewrite it on the other benchmarks. Finally, the case study in
Appendix E illustrates the same division of labor observed in the aggregate results: local chunks
provide grounded evidence, working evidence memory preserves accumulated facts across steps, and
the MiA-Signature maintains a compact global state that keeps retrieval and generation aligned with
the activated memory region.
5 Conclusion
We introduced MiA-Signature, a compact representation of the global activation pattern induced by a
query over a structured memory space. This representation serves as a tractable interface between
broad memory activation and downstream LLM computation. We instantiate this idea in both static
RAG and agentic systems, showing that a compact activation signature can improve how LLMs
access and use external memory across different inference settings. Across long-context benchmarks,
MiA-Signatures consistently improve over query-only counterparts. The results suggest that compact
representations of global activation can provide useful memory context without replacing local
evidence or requiring direct access to the full activated memory state. These findings support a view
of memory access in LLM systems as global activation followed by compact representation. MiA-
Signature offers one practical step toward this interface, connecting distributed memory activation
with local evidence-based reasoning.
9

References
[1]Anthropic. Claude code: Ai-powered coding assistant. https://claude.com/solutions/
coding, 2024. Accessed: 2026-04-13.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. InThe Twelfth International
Conference on Learning Representations, 2023.
[3] Bernard J. Baars.A Cognitive Theory of Consciousness. Cambridge University Press, 1988.
[4]Bernard J. Baars.In the Theater of Consciousness: The Workspace of the Mind. Oxford
University Press, 1997.
[5]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-
embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through
self-knowledge distillation, 2024.
[6]Stanislas Dehaene and Jean-Pierre Changeux. Experimental and theoretical approaches to
conscious processing.Neuron, 70(2):200–227, 2011.
[7]Stanislas Dehaene and Lionel Naccache. Towards a cognitive neuroscience of consciousness:
basic evidence and a workspace framework.Cognition, 79(1-2):1–37, 2001.
[8]Stanislas Dehaene, Laurent Cohen, Mariano Sigman, and Fabien Vinckier. The neural code for
written words: a proposal.Trends in cognitive sciences, 9(7):335–341, 2005.
[9]Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han,
Le Sun, and Jie Zhou. Deeprag: Thinking to retrieve step by step for large language models.
arXiv preprint arXiv:2502.01142, 2025.
[10] Abhay Gupta, Kevin Zhu, Vasu Sharma, Sean O’Brien, and Michael Lu. Novelhopqa: Diag-
nosing multi-hop reasoning failures in long narrative contexts. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, pages 26145–26162, 2025.
[11] Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobio-
logically inspired long-term memory for large language models.Advances in neural information
processing systems, 37:59532–59569, 2024.
[12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv
preprint arXiv:2410.21276, 2024.
[13] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages
7036–7050, 2024.
[14] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings
of the 2023 conference on empirical methods in natural language processing, pages 7969–7992,
2023.
[15] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Za-
mani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with
reinforcement learning.arXiv preprint arXiv:2503.09516, 2025.
[16] Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. One thousand
and one pairs: A "novel" challenge for long-context language models. InProceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024,
Miami, FL, USA, November 12-16, 2024, 2024. URL https://doi.org/10.18653/v1/
2024.emnlp-main.948.
10

[17] Tomáš Ko ˇcisk`y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gá-
bor Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge.
Transactions of the Association for Computational Linguistics, 6:317–328, 2018. URL
https://aclanthology.org/Q18-1023.pdf.
[18] Sid Kouider, Vincent De Gardelle, Jérôme Sackur, and Emmanuel Dupoux. How rich is
consciousness? the partial awareness hypothesis.Trends in cognitive sciences, 14(7):301–307,
2010.
[19] Victor AF Lamme. Towards a true neural stance on consciousness.Trends in cognitive sciences,
10(11):494–501, 2006.
[20] Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. A human-inspired
reading agent with gist memory of very long contexts.arXiv preprint arXiv:2402.09727, 2024.
[21] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. InProceedings of
the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5420–5438,
2025.
[22] Yuqing Li, Jiangnan Li, Zheng Lin, Ziyan Zhou, Junjie Wu, Weiping Wang, Jie Zhou, and Mo Yu.
Mindscape-aware retrieval augmented generation for improved long context understanding,
2025. URLhttps://arxiv.org/abs/2512.17220.
[23] Yuqing Li, Jiangnan Li, Mo Yu, Guoxuan Ding, Zheng Lin, Weiping Wang, and Jie Zhou.
Query-focused and memory-aware reranker for long context processing.arXiv preprint
arXiv:2602.12192, 2026.
[24] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu,
Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open
large language models.arXiv preprint arXiv:2512.02556, 2025.
[25] George A Mashour, Pieter Roelfsema, Jean-Pierre Changeux, and Stanislas Dehaene. Conscious
processing and the global neuronal workspace hypothesis.Neuron, 105(5):776–798, 2020.
[26] Lionel Naccache. Why and how access consciousness can account for phenomenal conscious-
ness.Philosophical Transactions of the Royal Society B: Biological Sciences, 373(1755):
20170357, 2018.
[27] George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approxima-
tions for maximizing submodular set functions—i.Mathematical Programming, 14:265–294,
1978.
[28] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. InThe Twelfth
International Conference on Learning Representations, 2024.
[29] Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, and
An Zhang. Look back to reason forward: Revisitable memory for long-context llm agents.
arXiv preprint arXiv:2509.23040, 2025.
[30] Giulio Tononi. An information integration theory of consciousness.BMC neuroscience, 5(1):
42, 2004.
[31] Giulio Tononi. Consciousness as integrated information: a provisional manifesto.The Biological
Bulletin, 215(3):216–242, 2008.
[32] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In
Proceedings of the 61st annual meeting of the association for computational linguistics (volume
1: long papers), pages 10014–10037, 2023.
11

[33] Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, and Liyan Xu.
Comorag: A cognitive-inspired memory-organized rag for stateful long narrative reasoning. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 33557–33565,
2026.
[34] Zhe Xu, Jiasheng Ye, Xiaoran Liu, Xiangyang Liu, Tianxiang Sun, Zhigeng Liu, Qipeng Guo,
Linlin Li, Qun Liu, Xuanjing Huang, and Xipeng Qiu. DetectiveQA: Evaluating long-context
reasoning on detective novels. InWorkshop on Reasoning and Planning for Large Language
Models, 2025. URLhttps://openreview.net/forum?id=9ExIs5ELlk.
[35] Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin
Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context
llm with multi-conv rl-based memory agent.arXiv preprint arXiv:2507.02259, 2025.
A Submodular Initialization: Coverage-aware vs. First-K
This appendix details the submodular selection framework used to construct the initial MiA-Signature
at step 0. We consider two variants:Coverage-aware submodular, which uses query relevance,
chunk-level coverage, and diversity, andFirst- Ksubmodular, a relevance-only degenerate variant
that is modular and therefore trivially submodular. Static RAG uses the Coverage-aware variant,
while the agent uses the First- Kvariant because the signature is later refined online. We formalize
the objective, discuss its submodularity, describe the algorithm, and compare the two variants under
identical retrieval pipelines.
A.1 Problem Setup and Motivation
Letqdenote the query and let C= (c 1, c2, . . . , c M)be the rank-ordered list of candidate chunks
returned by the step- 0query-only retriever (cf. Sec. 3). Each chunk ciis associated with asession
summary sπ(i)∈ S={s 1, . . . , s N}via a deterministic mapping π:C → S induced by document
sessionization. Distinct chunks may share a session summary, so typically N≪M . Our goal is to
select a subset A ⊆ S with|A| ≤K whose concatenation forms the initial signature σ0fed to the
retriever at step1.
Session summaries.We construct session summaries once for each document Dbefore any query
is issued. Let X(D) = (x 1, . . . , x L)denote the source-order chunk sequence of D. We divide this
sequence into non-overlapping contiguous windows of W= 20 chunks and summarize each window
with a single GPT-4o [ 12] call using a fixed summary-construction prompt [ 23]. The full prompt is
given in Appendix F.
The resulting summaries form
S={s 1, . . . , s J}, J=L
W
.
In our experiments, we instantiate the high-level memory set as H(D)≡ S . The chunk-to-summary
mapping is fixed by the source-order window assignment:
π(xℓ) =s ⌈ℓ/W⌉ .
For a retrieved candidate chunk c,π(c) denotes the cached session summary assigned to the source
chunk from which coriginates. This construction is deterministic, query-independent, and fixed
at indexing time. We use the same window size, GPT-4o summarizer, and summary-construction
prompt across all datasets. Since Sis cached, summary construction adds no query-time LLM calls
and the summaries are reused across all queries overD.
Why not First- K?The simple variant selects A={s π(i)}K
i=1, the session summaries associated
with the top- Kchunks. This is efficient but can be redundant, since multiple top-ranked chunks
may map to the same session. It can also miss useful summaries just below the rank cutoff, and the
chunk-level ranking may not reflect summary-level alignment with the overall information need.
Coverage-aware submodular selection addresses these issues by adding chunk-level coverage and
diversity to the relevance-only objective.
12

A.2 Objective Formulation
Leteq∈Rd,esj∈Rd,eci∈Rdbeℓ2-normalized BGE-M3CLSembeddings of the query, summary
sj, and chunkc i, respectively. Define:
(i) Query relevance.The sum of cosine similarities between the query and the selected summaries:
fQ(A) =X
s∈Ae⊤
qes (7)
(ii) Chunk coverage.For a candidate chunk ciof rank ri(with r1= 1), assign a rank-decaying
weight wi= 1/(r i+ 1) . Define thematch scorebetween summary sand chunk casm(s, c) =
max(0,e⊤
sec). LetCov(s)⊆ C denote the chunks whose session summary equals s(i.e.,Cov(s) =
π−1(s)). The coverage term is:
fC(A) =X
ci∈Cwi·max
s∈A, c i∈Cov(s)m(s, c i),(8)
with the convention that the inner max is zero when no selected summary covers ci. Two compound-
ing weights are at play: wibiases toward chunks highly ranked by the step- 0retriever, while m(s, c i)
is a semantic fidelity check that a chunk is genuinely reflected in its session summary.
(iii) Diversity.Penalize angular similarity to the existing selection:
fD(s| A) =1A=∅
1−max s′∈Ae⊤
ses′otherwise(9)
Combined marginal gain.Letting ˜fQ,˜fCdenote max-normalized variants of fQ, fC(to place the
three terms on a common scale), the marginal gain of adding summary sto the current selection Ais:
∆(s| A) =λ Q∆˜fQ(s| A) +λ C∆˜fC(s| A) +λ DfD(s| A),(10)
with default weightsλ Q= 0.3,λ C= 0.4,λ D= 0.3.
A.3 Submodularity Analysis
Recall that a set function f: 2S→R ismonotoneif f(A)≤f(B) whenever A ⊆ B , and
submodularif for allA ⊆ B ⊆ Sands /∈ B,
f(A ∪ {s})−f(A)≥f(B ∪ {s})−f(B)
(the diminishing-returns property).
Proposition 1 ( fQis modular).The query-relevance term fQis a sum of element-wise quantities
e⊤
qesand is thereforemodular, hence trivially submodular and monotone.
Proposition 2 ( fCis monotone submodular).The coverage term fCin Eq. 8 has the form of a
weightedMAX-COVERAGE:
fC(A) =X
ci∈Cwi·max
s∈A
m(s, c i)·1[c i∈Cov(s)]
.
Each summand is a weighted maximum of non-negative quantities over A; themax operator over
a growing set is monotone and submodular, and non-negative linear combinations preserve both
properties [27].
Status of fD.The diversity term fDisnotmonotone: Adding a summary ssimilar to an existing
selection can decrease its value. Consequently ∆(· | ·) in Eq. 10 is not globally submodular. This is
a deliberate design choice: the diversity term mainly acts as a tie-breaker between candidates with
similar fQ/fCprofiles, and the (1−1/e) approximation guarantee of greedy maximization under
monotonesub-modular objectives [27] still holds for thef Q+fCsub-objective obtained by setting
λD= 0.
13

Approximation guarantee.Ignoring the diversity term, the remaining objective fQC(A) =
λQ˜fQ(A) +λ C˜fC(A)is monotone submodular and non-negative. The classical result of Nemhauser,
Wolsey, and Fisher [27] guarantees that greedy selection returns a solutionA greedy such that
fQC(Agreedy )≥ 
1−1
e
·fQC(A∗),(11)
where A∗is the optimal size- Ksubset. This (1−1/e)≈0.632 bound applies to the dominant
monotone component of our objective.
A.4 Algorithmic Procedure
We implement greedy maximization of Eq. 10 (Algorithm 2), which expands the INITSIGNATURE
call at line 1 of the main-paper agent loop (Alg. 1 in Sec. 3.2.1). Embeddings are computed once per
call in a single batched forward pass through BGE-M3 [ 5], then cached. Complexity is O(K·N·M)
set operations after the embedding pass, which is negligible relative to the retriever call itself.
Algorithm 2Coverage-Aware Sub-modular Summary Selection
Require: Query q; candidate chunks Cwith ranks; summary set S; coverage map π−1; budget K;
weightsλ Q, λC, λD
Ensure:Selected summariesA, size≤K
1:Compute embeddingse q,{es},{e c}(batched BGE-M3, CLS,ℓ 2-norm)
2:w i←1/(r i+ 1)for eachc i∈ C
3:Pre-computeq s←e⊤
qesfor alls∈ S
4:Pre-computecov s←P
c∈π−1(s)wc·m(s, c)
5:Normalize:˜q s←qs/max s′qs′;˜ cov s←cov s/max s′covs′
6:A ←∅;covered(c)←0for allc∈ C
7:fork= 1toKdo
8:∆⋆← −∞,s⋆←None
9:for alls∈ S \ Ado
10:∆ Q←˜qs {modular, Eq. 7}
11:∆ C←1
ZCP
c∈π−1(s)wc·m(s, c)·(1−covered(c)){marginal coverage, Eq. 8}
12:∆ D←1−max s′∈Ae⊤
ses′, or1ifA=∅{Eq. 9}
13:∆←λ Q∆Q+λC∆C+λD∆D
14:if∆>∆⋆then
15:∆⋆←∆,s⋆←s
16:end if
17:end for
18:ifs⋆=Nonethen break
19:A ← A ∪ {s⋆}
20:For eachc∈π−1(s⋆):covered(c)←1
21:end for
22:returnA
Implementation note.The normalization constant ZCin line 11 is max s′covs′(the same denom-
inator used in line 5), which keeps the three ∆-terms on a comparable [0,1] scale. The BGE-M3
model is loaded once per process and cached, so the algorithm adds no overhead when submodular
selection is disabled.
A.5 Coverage vs. First-KInitialization
All main results use submodular initialization to construct the initial MiA-Signature. To isolate
the effect of this design choice, we compare it with a simple First- Kinitializer, which takes the
session summaries associated with the top- Kchunks from the step- 0query-only ranking. All other
components are kept fixed, including the retriever backbone, generator, and refinement budget.
Findings.First- Kand Coverage-aware submodular use the same step- 0query-only candidate pool,
so the initial evidence frontier is unchanged. Their difference lies in which high-level summaries
14

Table 3:Coverage-aware vs. First-Ksubmodular initialization.
Method InitDetectiveQA NarrativeQA NovelHopQA NoCha Avg.
R@10(EN/ZH) Acc-EN Acc-ZH R@10F1 R@10F1 PairAcc R@10Perf.
MiA-EmbFirst-K 44.7 / 42.7 69.3 79.3 57.9 42.6 37.0 37.7 58.7 46.2 53.3
Submodular 44.7 / 42.7 70.7 78.0 59.5 45.1 36.8 38.5 58.7 46.7 54.2
MiA-RAG (+MiA-Gen-14B)First-K 44.7 / 42.7 70.7 70.7 57.9 47.2 37.0 27.7 47.6 46.2 48.3
Submodular 44.7 / 42.7 70.7 73.3 59.5 48.0 36.8 27.9 47.6 46.7 48.9
MiA-RAG (+DS-V3.2)First-K 44.7 / 42.7 74.0 80.7 57.9 42.7 37.0 39.6 63.5 46.2 55.8
Submodular 44.7 / 42.7 74.7 80.0 59.5 42.8 36.8 38.7 65.1 46.7 56.0
Embedding Model
query </s>Mindscape -aware Embedding Model
query </s> signature </s>α 1-α
Figure 2: How the query-only embedding model and the mindscape-aware embedding model work
when encoding the query end.
are selected to form σ0. Since the second retrieval pass is conditioned on both the query and this
signature, different initializations lead to different query–signature retrieval distributions.
Coverage-aware submodular gives a small but consistent improvement in average R@10 across
all three signature-based variants, and also improves average task performance in each case. The
clearest gain appears on NarrativeQA, where the activated context is broad and redundant; selecting
summaries for chunk coverage is therefore more useful than simply taking the first Ksummaries
from the ranking.
The effects are smaller and sometimes mixed on DetectiveQA, NovelHopQA, and NoCha. This
is expected: when the activated region is narrower, or when the answer depends on precise local
distinctions, the First- Ksubmodular variant can already provide a reasonable signature. Overall, the
ablation shows that adding coverage-aware terms to the submodular objective is a modest but reliable
improvement for static RAG, and justifies our default of using First- Ksubmodular initialization in
the agent, where later refinement steps can compensate for a simpler initial objective.
B Retriever Mechanism
We employ two types of retrievers to search chunks (evidences), utilizing either the query alone or
the combination of the query and a signature. The first is referred to as a query-only retriever, which
encodes the query using conventional last-token pooling; SFT-Embedding and Qwen3-Embedding
belong to this category. The second is a mindscape-aware retriever (MiA-Embedding), which takes
both the query and the signature as input following Eq. 4. It captures two sources of information via
interpolation: query-only information and signature-situated query information. This combination
is effective because the causal attention mechanism in the decoder-structured retriever masks the
signature tokens when processing the first </s> token. An illustration of the two retriever types is
provided in Fig. 2. Note that interpolation is applied only when encoding the query–signature pair;
the chunk encoding process remains identical to that of the query-only retrievers.
15

C Dataset Construction and Statistics
C.1 Series Aggregation Details
We aggregate books from original benchmarks to form coherent multi-volume series. ForDetec-
tiveQA, we group 13 novels into Miss Marple and Hercule Poirot series. ForNarrativeQA, we
select 37 books to form 11 series based on sequential arcs or shared protagonists. Table 4 summarizes
the aggregation details.
Table 4: Statistics of aggregated series for DetectiveQA and NarrativeQA.
Benchmark Series / Arc #Books #Questions
DetectiveQA Miss Marple (A. Christie) 8 81
Hercule Poirot (A. Christie) 5 69
NarrativeQA Anne of Green Gables 4 42
Balzac, La Comédie Humaine 6 28
Sherlock Holmes (A. C. Doyle) 3 35
Other (Indiana Jones, Star Wars, etc.) 24 295
Total 13 Series 50 550
C.2 Single-Book vs. Series-Book Control Details
As a sanity check that series-book indexing is strictly harder than single-book indexing, we run a
retrieval-side control with all other pipeline components held fixed. We use the same query-only
retriever (SFT-Emb-8B) and compare two indexing granularities:Single-Book, where only the gold
book is indexed and each question is answered against its own book; andSeries-Book, where all
books of a series are merged into a single document and retrieval is performed over the merged index.
Table 5 reports retrieval recall under the two settings.
C.3 On Full Global-Summary Baselines
Prior MiA-RAG [ 22] conditions retrieval and generation on a document-level global summary. This
differs from theMiA-RAGsystem in this paper, where the conditioning signal is a query-induced
MiA-Signature rather than a pre-existing document summary.
We do not include the prior full global-summary baseline in the series-book setting because the
required summary is not well defined. A summary of the merged series would mix plots, characters,
and events from multiple books, introducing semantic interference. A summary of only the gold book
would require knowing the target book before retrieval, which leaks information. Our setting instead
tests whether the system can identify the query-relevant region from an overcomplete memory space.
MiA-Signature is designed for this regime because it is induced by the query and does not assume
access to a reliable document-level summary in advance.
Table 5: Query-only retrieval recall (%) under single-book vs. series-book indexing, using SFT-
Emb-8B with identical chunking and retrieval budget. Merging books of a series into a single index
consistently lowers R@k , confirming that cross-book context acts as semantic interference rather
than useful side information.
BenchmarkSingle-Book Series-Book
R@5R@10 R@5R@10
DetectiveQA (EN) 30.0 40.0 29.3 38.0
DetectiveQA (ZH) 40.0 56.0 34.0 48.0
NarrativeQA 48.2 61.0 44.0 55.5
The drop from Single-Book to Series-Book indicates that merging related books introduces additional
semantic interference for query-only retrieval. This control motivates our use of the series-book
16

setting in the main experiments: it better tests whether a memory interface can identify the query-
relevant region before chunk-level matching. We therefore report all main-paper results under the
harder series-book setting.
Table 6:Query-rewrite ablation.All variants use an evolving signature and differ only in whether
the query is rewritten at each step; the four blocks vary what the generator receives at answer time.
Variant Query rewriteDetectiveQA Acc NarrativeQA NovelHopQA NoCha Avg.
EN ZH F1 F1 Acc PairAcc Perf.
chunks only✓ 68.7 81.3 45.3 37.7 80.2 61.9 55.0
× 70.0 78.0 44.0 38.7 80.2 60.3 54.3
+sig✓ 76.782.0 44.9 35.7 83.3 68.3 57.1
× 69.3 80.0 43.8 37.1 84.1 68.3 56.0
+evidence✓ 73.3 80.7 43.3 35.9 83.3 66.7 55.7
× 73.386.0 43.6 36.2 84.1 68.3 56.9
+sig+evidence✓ 73.3 80.0 44.3 35.0 85.7 71.4 56.8
× 71.3 82.0 43.0 35.6 84.1 68.3 55.9
Perf.∆for +sig+evidence (✓− ×) +2.0−2.0 +1.3 −0.6 +1.6 +3.1 +0.9
D Query-Rewrite Ablation
While σtalways evolves with newly retrieved evidence, we ablate whether the query qtshould also
be rewritten at each step. Table 6 reports final performance under the four answer-time interfaces,
toggling query rewriting on and off.
Findings.Rewriting helps when refinement should narrow the search—most clearly on NarrativeQA
and NoCha, where partial evidence can be turned into a more specific follow-up query—but is not
universally beneficial. NovelHopQA is the exception: keeping the query fixed yields higher F1,
since multi-hop questions benefit from preserving parallel evidence paths rather than specializing
around the first evidence found. Accordingly, our main experiments keep qtfixed on NovelHopQA
and rewrite it elsewhere. Query rewriting and signature evolution therefore play distinct roles. The
signature carries the evolving global memory state; query rewriting only controls how narrowly the
next retrieval step is posed, and we treat it as a benchmark-dependent control rather than a core
mechanism.
E Case Study
We provide a DetectiveQA trace below. The compared systems retrieve locally plausible evidence
for option B, but fail to maintain the global identity binding needed for the causal answer. MiA-
Emb commits to the local surface reading, MiA-RAG’s signature does not bind the hostess role to
Charlotte-as-Letitia, and the agent without a signature has no state that preserves this binding across
steps. In contrast, MiA-Agent updates the signature once the binding is surfaced, allowing later
retrieval and generation to select the correct answer.
Case study.DetectiveQA–EN, Miss Marple merged series.
Question.The hostess’s close friend, Dora, died on the second day of her birthday party. What was the cause of her
death?
Options.
A. Rudi gives Dora poison to kill her in advance.
B. Dora ingested poison.
C. Someone entered the room and killed Dora while she was sleeping.
D. The mistress substituted the sleeping pills that Dora took with poison.(gold)
Why this case probes the signature.Option B is supported by local chunks: the retriever surfaces that the pill Dora took
was not aspirin. Option D requires a global binding from a different part of the novel: the hostess “Letitia” Blacklock is
17

in fact Charlotte impersonating her dead sister, and Charlotte is the one who swapped the bedside bottle. Since retrieval
runs over the 8-book Marple series, both the poisoning subplot and the identity subplot can be activated. The central
question is whether the system can carry their binding across retrieval and generation.
Compared systems that answer incorrectly.
MIA-EMB. PredictsB. No signature is provided to the generator; the retrieved context contains the party
scene and the non-aspirin pill, so the model commits to the local surface reading.
MIA-RAG. PredictsB. The generator receives a signature, but the signature does not yet bind the hostess
role to Charlotte-as-Letitia.
(AGENT WITHOUT SIGNATURE) . PredictsB. The query is rewritten across steps, but no global signature is
carried to preserve the Charlotte–Letitia binding.
MiA-Signature (ours).Two refinement steps, confidenceMEDIUM→HIGH.
Step 0.Initial signature construction
The query-only retriever E1returns the top- 50chunks from the merged Marple series. Their session summaries
deduplicate to a candidate pool with |H0(q)|= 10 . First- Ksubmodular selection keeps the top K= 5 summaries
by query relevance.
Selected summaries include:
•Miss Marple reveals that Charlotte Blacklock assumed her sister Letitia’s identity. Charlotte invited Dora
Bunner, but Dora’s forgetfulness threatened Charlotte’s secret.(q= 0.89)
•Poisoned aspirins that killed Dora Bunner were placed during the birthday party; the positions of the partici-
pants are reconstructed.(q= 0.84)
•Dora Bunner was found dead after taking pills from Blacklock’s bedside; one remaining pill was not aspirin.
(q= 1.00)
•Two additional top-ranked summaries, including one semantically related Marple scene that acts as distractor
context.
The resulting σ0activates both the identity subplot and the poisoning subplot, but it has not yet committed the
binding between the hostess identity and the pill substitution.
Step 1.REFINE, confidenceMEDIUM
Queryq 0.What was the cause of Dora’s death?
Evidence memoryE 1grounded in retrieval.
• Dora took aspirin from a bottle on MissLetitiaBlacklock’s bedside; the remaining pill was not aspirin.
•The poisoned aspirins were placed during Dora’s birthday party; anyone present could have switched
them.
•Charlotte Blacklock, posing as Letitia, killed Dora because Dora’s forgetfulness threatened Charlotte’s
secret identity.
Reason. E1establishes the mechanism, namely a non-aspirin pill, and places the swap at the birthday party.
However, the agent of the swap is not yet bound to the hostess identity referred to by option D. The agent therefore
refines rather than answers.
Co-evolution at the end of Step 1.The agent updates(q 0, σ0, E0)into(q 1, σ1, E1).
Rewritten query q1.Who substituted the aspirin pills that Dora Bunner took, and what specific poison
caused her death?
Refined signature σ1written by Mupd.Miss Marple reveals the full truth at the parsonage. Charlotte
Blacklock, living under her dead sister Letitia’s identity, saw her old friend Dora Bunner as a threat
due to Dora’s increasing forgetfulness and talkativeness. To protect her secret, Charlotte substituted the
aspirin in her own bedside bottle with poisoned pills, knowing Dora would take them.
The binding “hostess=Charlotte-as-Letitia=agent of the swap” is now encoded insideσ 1.
18

Step 2.ANSWER, confidenceHIGH
Signature-guided retrieval E2now retrieves under (q1, σ1). Miss Marple’s parsonage reveal is promoted because
σ1biases retrieval toward the identity-bound region, not merely because the chunks mention Dora.
Evidence memoryE 2updated this step.
• Charlotte substituted the aspirin in her, that is, “Letitia’s”, bedside bottle with poisoned pills.
• The specific poison was aconite.
• Charlotte framed the killing as an act of mercy for Dora.
Reason.With σ1identifying the hostess as the agent of the swap, “the mistress” in option D becomes bindable.
Option B remains locally true, but it is only the surface mechanism. Option D gives the causal explanation. The
agent answers.
Final answer. D✓
F Prompt Templates
We list the prompts used by MiA-Signature. The SESSION-SUMMARYprompt is called once per
sessionization window during offline preprocessing to construct the high-level memory set H(D) .
The UPDATEprompt is called at each refinement step by Mupd, and the answer prompt is sent to
Mgenonce the agent decides to answer. Placeholders are shown in{braces}.
SESSION-SUMMARYPrompt
You are provided with a raw text segment from a book (Part {idx}/{total}).
This segment consists of approximately 20 consecutive chunks combined.
<Raw_Text>
{raw_text}
</Raw_Text>
Please generate a **Detailed Narrative Summary** following these strict guidelines:
1. **Narrative Reconstruction**: Do not list events. Rewrite the content as a coherent story in the
third person, past tense. It should read like a condensed version of the original text.
2. **Detail Preservation**:
- Preserve specific **Character Names** and their relationships.
- Keep key **Dialogues** that drive the plot.
- Note specific **Locations** or setting changes.
3. **Noise Filtering**:
- IGNORE any copyright notices, Project Gutenberg headers, page numbers, or tables of contents.
- If the text starts or ends in the middle of a sentence, ignore the broken fragments and focus on
the complete thoughts.
4. **Style**:
- NO meta-commentary (e.g., do NOT say "The text describes...", "In this chunk...").
- Directly tell the story.
5. **Length**: 50--100 words.
Output the summary directly.
UPDATEPrompt
[System]
You are a retrieval planning agent. You decide whether retrieved passages contain enough evidence to
answer a question, and if not, co-refine the retrieval signature and search query.
You work with a mindscape-aware retriever guided by TWO evolving signals:
1. Signature: A short narrative summary --- the "compass" steering retrieval toward the right
storyline, characters, and events.
2. Search Query: A rewritten question targeting specific missing evidence. The retriever combines both
signals to find passages.
Information hierarchy:
- Session Summaries: high-level narrative overviews for orientation.
- Retrieved Passages: specific text chunks from the current retrieval step.
19

- Evidence Memory: running notes accumulated across all steps.
Decision process:
- Most evidence found -> ANSWER
- Key evidence missing + steps remaining -> REFINE
- LOW confidence + steps remaining -> REFINE
- Last step -> ANSWER
When action=REFINE:
- Output both a refined signature and a rewritten query.
- The signature targets the missing evidence and relevant storyline.
- The query targets concrete missing evidence.
Output format:
<evidence_memory> bullet list of key findings </evidence_memory>
<confidence>HIGH/MEDIUM/LOW</confidence>
<thought>reasoning: what is found, missing, chosen action</thought>
<action>ANSWER or REFINE</action>
<refined_signature>updated narrative compass, only if REFINE</refined_signature>
<rewritten_query>targeted search query, only if REFINE</rewritten_query>
[User]
Question:
{question}
Options:
{options_str}
Step {step}/{max_steps} | {remaining_steps_hint}
Current signature:
{signature}
Current search query:
{current_query}
Session summaries:
{summaries_text}
Evidence memory:
{evidence_memory}
Retrieved passages:
{chunks_text}
{history_section}
Do the passages above, understood in the context of the session summaries and
your current signature, contain enough evidence to answer the question?
Answer-time input variants.All answer-time variants include the final retrieved chunks. The
following table only shows which additional memory states are prepended to the generator context.
Variant Generator context
Chunks{context}
Chunks + SIG.{signature}+{context}
Chunks + EVI.{evidence_memory}+{context}
Chunks + SIG. + EVI.{signature}+{evidence_memory}+{context}
Answer Prompts and Dataset-Specific Output Formats
--- DetectiveQA ---
[System]
You are a helpful assistant.
[User]
{answer_context}
Please answer the question based on the current novel content:
{question}
{options_str}
20

Please strictly follow the format {"answer":"x","reasoning":"xxx"}. The answer field should only
contain A, B, C, or D.
------------------------------------------------------------
--- Open QA (NarrativeQA, NovelHopQA) ---
[System] You are an expert reading comprehension assistant. Analyze provided passages to answer the
question as concisely as possible, using a single phrase if possible. Do not provide any
explanation.
[User] ... {context} ...
Now, answer the question based on the story as concisely as you can, using a single phrase if possible.
Do not provide any explanation.
Question: {question}
Answer:
------------------------------------------------------------
--- Claim Verification (NoCha) ---
[System] You are an expert reading comprehension assistant. You verify whether statements about novels
are TRUE or FALSE based on provided evidence.
[User] ... {context} ...
<statement>{claim}</statement>
<question>Based on the context provided, is the above statement TRUE or FALSE?</question>
Answer TRUE if the statement is true in its entirety based on the context provided.
Answer FALSE if any part of the statement is false based on the context provided.
First provide an explanation in at most one paragraph, then your final answer:
<explanation>YOUR EXPLANATION</explanation>
<answer>YOUR ANSWER</answer>
G Limitations
Our results show that MiA-Signatures offer an effective memory interface for long-context narrative
understanding, especially when evidence is dispersed across a large source. Still, our experiments are
centered on literary and narrative domains where memory naturally forms chapter- or session-level
units; whether the same activation–signature formulation transfers to code repositories, scientific
literature, or multimodal interaction remains to be tested. The current signature construction is also
training-free and based on submodular selection over precomputed summaries, which keeps the
method modular but does not optimize the signature end-to-end with the retriever, generator, or task
objective. Finally, MiA-Signature should be understood as a global-structure prior rather than a
replacement for local evidence: it helps when answers require synthesis across dispersed context, but
can be unnecessary or distracting when the answer is already locally supported. Adaptive control
over when to expose the signature to the generator remains future work.
21