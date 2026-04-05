# Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference

**Authors**: Zifan He, Rui Ma, Yizhou Sun, Jason Cong

**Published**: 2026-03-30 21:03:39

**PDF URL**: [https://arxiv.org/pdf/2603.29002v1](https://arxiv.org/pdf/2603.29002v1)

## Abstract
Modern large language models (LLMs) increasingly depends on efficient long-context processing and generation mechanisms, including sparse attention, retrieval-augmented generation (RAG), and compressed contextual memory, to support complex reasoning. We show that these optimizations can be unified into a four-step memory processing pipeline: Prepare Memory, Compute Relevancy, Retrieval, and Apply to Inference. Through systematic profiling, we identify a 22%-97% memory processing overhead in LLM inference and strong heterogeneity in its computational characteristics. Motivated by this insight, we argue that \textbf{heterogeneous systems} are well-suited to accelerate memory processing and thus end-to-end inference. We demonstrate this approach on a GPU-FPGA system by offloading sparse, irregular, and memory-bounded operations to FPGAs while retaining compute-intensive operations on GPUs. Evaluated on an AMD MI210 GPU and an Alveo U55C FPGA, our system is $1.04\sim2.2\times$ faster and requires $1.11\sim4.7\times$ less energy across multiple LLM inference optimizations than the GPU baseline (similar results hold on NVIDIA A100). These results establish heterogeneous systems as a practical direction for efficient LLM memory processing and inform future heterogeneous hardware design.

## Full Text


<!-- PDF content starts -->

Understand and Accelerate Memory Processing Pipeline for
Disaggregated LLM Inference
Zifan He1Rui Ma2Yizhou Sun1Jason Cong1
Abstract
Modern large language models (LLMs) increas-
ingly depends on efficient long-context processing
and generation mechanisms, including sparse at-
tention, retrieval-augmented generation (RAG),
and compressed contextual memory, to support
complex reasoning. We show that these opti-
mizations can be unified into a four-step mem-
ory processing pipeline:Prepare Memory,Com-
pute Relevancy,Retrieval, andApply to Inference.
Through systematic profiling, we identify a 22%-
97% memory processing overhead in LLM infer-
ence and strong heterogeneity in its computational
characteristics. Motivated by this insight, we ar-
gue thatheterogeneous systemsare well-suited
to accelerate memory processing and thus end-
to-end inference. We demonstrate this approach
on a GPU-FPGA system by offloading sparse,
irregular, and memory-bounded operations to FP-
GAs while retaining compute-intensive opera-
tions on GPUs. Evaluated on an AMD MI210
GPU and an Alveo U55C FPGA, our system is
1.04∼2.2× faster and requires 1.11∼4.7×
less energy across multiple LLM inference opti-
mizations than the GPU baseline (similar results
hold on NVIDIA A100). These results establish
heterogeneous systems as a practical direction
for efficient LLM memory processing and inform
future heterogeneous hardware design.
1. Introduction
In recent years, large language models (LLMs) have demon-
strated capabilities beyond standard question answering.
LLM-based agents can now solve complex tasks through
multi-step reasoning (Yao et al., 2023; Wang & Zhou, 2024),
tool invocation (Yao et al., 2022), and long-horizon plan-
ning (Wang et al., 2023), which increasingly demands the
1Department of Computer Science, University of California,
Los Angeles2Microsoft Research Asia. Correspondence to: Zifan
He<zifanhe1202@g.ucla.edu>.
Preprint. Under Review.
Figure 1.The GPU-FPGA heterogeneous system (1 MI210 + 1
Alveo U55C) can provide 1.2−1.8× speedup and 1.3−4.7× en-
ergy cost reduction consistently over a wide range of long-context
LLM inference optimizations. “SA-R” stands for SeerAttention-R
and “DSA” stands for DeepSeek Attention.
ability to memorize and process long inputs. State-of-the-art
models (Comanici et al., 2025; Shen et al., 2025; Grattafiori
et al., 2024) can process and generate 128k to 1 million to-
kens per request when users prompt for paper reading, deep
reasoning, and creative writing. However, LLMs typically
maintain all contexts as key-value (KV) caches, incurring
substantial hardware costs and runtime overhead. For exam-
ple, storing KV cache for 1M tokens requires up to 69 GB
of GPU memory for the GPU-OSS-120B model (Agarwal
et al., 2025), and repeatedly accessing the cache further
amplifies memory pressure during auto-regressive decod-
ing. To mitigate this issue, the latest LLMs have developed
several algorithmic optimizations to improve LLM mem-
ory efficiency. Representative approaches include sparse
attention (Beltagy et al., 2020; Liu et al., 2025; Yang et al.,
2025b) that selectively attends to a subset of tokens, and
contextual memory (Behrouz et al., 2025; Sun et al., 2024)
that compresses past tokens into embeddings. Additionally,
retrieval augmented generation (RAG) (Lewis et al., 2020)
that offloads static knowledge to an external database can
also be considered as an approach to expand the context.
While these LLM inference optimizations effectively reduce
the end-to-end latency of long document processing, prior
work largely treats them as isolated techniques and lacks a
systematic understanding of their computational character-
istics and hardware efficiency implications. This deficiency
hinders further acceleration of LLM inference for both ex-
isting and emerging methods. In this work, we make three
insightful claims that will open a new way to understand
and systematically accelerate the LLM inference.
1arXiv:2603.29002v1  [cs.DC]  30 Mar 2026

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Claim 1:Mod ernLLM inference involves amem ory
processingpipeline. (Section 3)Existing LLM inference
optimizations mostly address the challenge of efficient mem-
ory processing. We formally definememory1in LLMs and
unify diverse methods under a common four-step pipeline:
(1)Prepare Memory, which preprocesses raw memory into
a compact or structured format; (2)Compute Relevancy,
which assigns importance scores to each memory entry;
(3)Retrieval, which selects and extracts information based
on these scores using specific heuristics; and (4)Apply to
Inference, which integrates the retrieved content into the de-
coding process. Through systematic profiling, we observe
that memory processing accounts for 22%−97% of the total
latency. This varies based on memory size and methods.
These findings not only reveal substantial opportunities for
accelerating LLM, but also highlight thata single system
solution targeting memory processing can provide benefits
across existing and future LLM inference paradigms.
Claim 2:Com putations inmem oryprocessingarehet-
erogeneous. (Section 4)Quantitatively, each step exhibits
distinct arithmetic intensity, leading to different utilization
of compute and memory resources. Qualitatively, their
memory access patterns and data dependencies also dif-
fer substantially. For example, generating compressed key
embeddings in sparse attention involves regular, consec-
utive accesses and is compute intensive when processing
multiple attention heads, whereas score computation and
retrieval rely on skinny matrix-vector multiply and top- k
search that are memory-bound, irregular in access pattern,
and data-dependent across tokens. We observe that such
computational heterogeneity is pervasive in the memory
processing pipeline of LLM inference.
Claim 3:Heterogeneous systems canacceleratemem -
oryprocessing. (Section 5)Motivated by the heterogeneity
in computational characteristics, we argue that mapping
LLM memory processing onto a heterogeneous system is
preferred to achieve optimal acceleration. In this work, we
present a solution based on off-the-shelf devices for demon-
stration, and the same paradigm can be used to inspire future
heterogeneous hardware designs. In particular, given the
speed and energy advantages of Field Programmable Gate
Array (FPGA) devices for sparse, irregular, and memory-
bound workloads over GPUs and CPUs (Song et al., 2022;
He et al., 2024), we propose executing the memory process-
ing pipeline of LLM inference on a GPU-FPGA heteroge-
neous system, with consideration of both computational het-
erogeneity and data locality. Evaluated on a platform com-
prising an AMD MI210 GPU and an Alveo U55C FPGA
connected via PCIe, we achieve:
•1.5∼5.7× speedup for the sparse attention (e.g.,
1In this work, “memory” refers to the processed data the LLMs
used for generation in Definition 3.1 (not the device memory).DeepSeek Sparse Attention (Liu et al., 2025)), resulting
in up to1.49×end-to-end speedup.
•5.16∼7.65× speedup for RAG (e.g., DRAGIN (Su et al.,
2024)), resulting in up to2.2×end-to-end speedup.
•1.3∼1.6× end-to-end speedup for Memory as Context,
and1.8×for synthesized memory (MemAgent).
•1.11∼4.66× lower geomean energy cost per request,
which can significantly reduce the serving cost of LLMs
equipped with these methods.
2. Background
We review representative long context/document LLM infer-
ence optimizations, focusing on methods that fundamentally
change how memory is accessed and managed.
Sparse Attention.Standard transformers incur quadratic
complexity during prefill (input processing) and linear com-
plexity during decoding (token generation) (Sheng et al.,
2023), making attention increasingly expensive for long con-
texts. Sparse attention mitigates this by selectively attending
to a subset of past tokens. Recent decode-stage methods in-
clude DeepSeek Attention (Liu et al., 2025), which retrieves
top-kimportant tokens with a lightweight indexer, LServe
(Yang et al., 2025b), which introduces a hierarchical paged
KV cache for fast retrieval, and SeerAttention-R (Gao et al.,
2025), which extends an auxiliary attention predictor to
the decoding phase. These approaches significantly reduce
memory access overhead while preserving model quality.
Retrieval-Augmented Generation (RAG).For static
knowledge sources, storing information as KV cache is
inefficient due to per-layer vector storage. RAG (Lewis
et al., 2020) instead gets relevant documents from an ex-
ternal corpus and concatenates them with the query. Rep-
resentative dynamic RAG systems include FLARE (Jiang
et al., 2023), which triggers retrieval when model confidence
drops, and DRAGIN (Su et al., 2024), which leverages at-
tention statistics to detect uncertainty. Fixed-sentence RAG
(Trivedi et al., 2023) performs retrieval at every sentence
boundary, enabling fine-grained context updates. Recently,
more advanced two-stage RAG with reranker (Moreira et al.,
2024) improve the accuracy by sacrificing the retrieval over-
head. These methods improve factuality while reducing
long-context storage overhead.
Compressed Contextual Memory.Compressed memory
stores information in a compact form (embeddings or sum-
marized texts), enabling LLMs to handle extremely long
inputs. Synthesized memory such as MemAgent (Yu et al.,
2025) summarizes long inputs into textual memories condi-
tioned on the query. Recurrent models, including HMT (He
et al., 2025a) and Memory as Context in Titans (Behrouz
et al., 2025), compress input segments into latent embed-
dings and retrieve them based on relevancy. These methods
2

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Table 1.Summary of LLM inference optimizations and the computations in their memory processing pipeline.
LLM Opt. Related Works Prepare Memory Compute Relevancy Retrieval Apply to Inference
Sparse AttentionDeepSeek Attention (Liu et al.,
2025)Linear Projections +
RoPEMulti-headed Inner Prod-
uctTop-k Fine-grain Sparse At-
tention
SeerAttention-R (Gao et al.,
2025)Linear Projections +
PoolingInner Product Top-k/ Threshold Block Sparse Attention
LServe (Yang et al., 2025b) Page-wise Min/Max
PoolingInner Product + Max Re-
ductionTop-k Block Sparse Attention
Retrieval Aug-
mented Generation
(RAG)Two-stage RAG (Moreira et al.,
2024)Embedding Model /
TokenizationInner Product / BM25 +
RerankerTop-kAppend to query
Fixed-Sentence RAG (Trivedi
et al., 2023)Tokenization BM25 Top-kAppend to query
Dynamic RAG (Su et al., 2024;
Jiang et al., 2023)Tokenization BM25 Top-kAppend to query
Synthesized Mem-
oryMemAgent (Yu et al., 2025;
Shen et al., 2025)Model Decoding N/A Nearest Retrieval Model Prefilling
Memory as ContextTitans (Behrouz et al., 2025),
HMT (He et al., 2025a)Forward pass Linear Projection + Inner
ProductTop-k/ Weighted
SumAppend to segment
Test-time TrainingTTT, LaCT (Sun et al., 2024;
Zhang et al., 2025b)Backward pass Compute loss N/A Forward pass
effectively trade computation for runtime data efficiency and
assist far-distance information retrieval in the latent space.
Test-time Training (TTT).TTT treats model parameters
as internal memory and adapts them during inference. The
seminal work (Sun et al., 2024) formulates TTT as a recur-
rent update rule, alternating between backpropagation and
generation. LaCT (Zhang et al., 2025b) extends this idea
with batched updates to improve GPU utilization.
Although these methods differ in their computations, they
all transform a processed input into an output through a com-
mon sequence of steps referred to asmemory processing.
Figure 2.Four-Step Memory Processing Pipeline in LLMs:
Prepare Memorypreprocesses and structures raw memory for
efficient access;Compute Relevancyassigns relevance scores to
memory entries with respect to the input query;Retrievalextracts
the most relevant memory based on these scores; andApply to
Inferenceintegrates retrieved content and input into intermediate
outputs, used in the rest operations in LLMs to produce tokens.
3. Memory Processing in LLM Inference
Definition 3.1.Define the generative language model as
L(g(·), f(·,·),{x i}i<t, xt) =y t, where {xi}i<tis the past
input sequence, xtandytare current input and output, g
is the memory generator, and fis the memory processor.
Lstarts with generated memory M<t=g({x i}i<t). Then
during inference, it initiates f(one-time or repeatedly) to
get intermediate output O<t=f(M <t, xt)and utilize O<t
to generate final outputy t.For example, gcan be the projections to produce KV cache
M<t, and fis the sparse attention mechanism to create
attention score O<t. Under this definition, the methods
mentioned in Section 2 are all improving the memory pro-
cessing ( f) efficiency. This section describes the common
properties of the memory processing.
3.1. Memory Processing Pipeline
To utilize models’ memory, LLM inference employs a four-
step memory processing pipeline (Figure 2):
•Prepare Memory( prep(M <t) =I <t): LLM first converts
the memory into a memory index that facilitates memory re-
trieval and usage. For instance, DeepSeek attention projects
latent KV vectors in MLA (multi-headed latent attention)
(Liu et al., 2024) into lightweight indexing vectors.
•Compute Relevancy( comp(I <t, xt) =S ): In this step,
the model utilizes the processed memory and current input,
or query, to identify which part of the memory is relevant
and should be extracted. The output is the relevancy scores
where higher score indicates higher relevancy. For example,
DeepSeek attention computes the multi-head dot product
scores between the indexing vectors and the query vector.
•Retrieval( ret(M <t, S) =M′
<t): Given the relevancy
score and the original memory, the model selects a subset
of memory entries or constructs refined memory based on
the score and certain heuristics. For example, DeepSeek
attention applies top-kselection on each token.
•Apply to Inference( apply(M′
<t, xt) =O <t): Finally, the
retrieved memory is incorporated into subsequent computa-
tions alongside the target inputs that the model transforms
to generate the output. For instance, the KV latent embed-
dings corresponding to the top- kindex scores are used in
3

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
the MLA computations. For RAG, the selected documents
are concatenated with the query to augment the inference.
Table 1 outlines the LLM inference method types with rep-
resentative works and the detailed computations involved in
each step of the memory processing pipeline. Some meth-
ods skip a few steps. For example, MemAgent (Yu et al.,
2025) skips relevancy computation since it always retrieves
memory from the preceding segment. TTT does not perform
retrieval and incorporates parameterized memory through a
direct forward pass. Moreover, the timing and execution fre-
quency of each step in the end-to-end inference vary across
different memory types. RAG typically prepares memory
once and repeatedly performs retrieval, while sparse atten-
tion processes memory for every token.
3.2. Memory Processing is Time Critical
Figure 3.Percentage of latency spent on memory processing for
sparse attention methods. With 1M tokens, memory processing
can take 22%–81% of the decoding time.
By profiling the latency breakdowns of LLM inference op-
timizations based on the experimental settings illustrated
in Section 6.1, we observe a large proportion of inference
latency is spent on memory processing as the memory size
increases. For sparse attention (Figure 3), the percentage
of decoding latency for memory processing grows from
1−11% for 4K-token sequences to 22−81% for 1M-token
sequences. Given that modern LLMs (Comanici et al., 2025)
can process more than 1M tokens per sample, memory pro-
cessing can become the critical path when sparse attention is
applied. For RAG (Figure 4), our profiling reveals a similar
phenomenon when processing 20M documents ( 40−61% ).
For two-stage RAG, the reranker dominates memory pro-
cessing latency, leading to a high latency percentage with a
slower increase as the document count grows. As depicted
in Figure 5, for parameterized memory and synthesized
memory, memory processing is time-consuming even with
short contexts. The reasons are diverse: MemAgent em-
ploys the model to generate textual memory, and Titans
and HMT involve multiple linear layers to project segment
embeddings and memory embeddings into the same latent
space.
Furthermore, we observe that the latency breakdown in thememory processing pipeline varies among methods. For
example, sparse attention and RAG are dominated by the
Compute Relevancy and Retrieval, whereas MemAgent in-
curs up to 97% of latency in Prepare Memory (details in
Appendix B). Since memory processing constitutes a pri-
mary bottleneck in LLM, accelerating it delivers substantial
gains in both end-to-end latency and energy efficiency.
Figure 4.Percentage of latency on memory processing for RAG
using the Wikipedia dump (Su et al., 2024). For two-stage RAG,
reranking is time consuming, leading to a high percentage at 500K
and slow increment as document count grows.
Figure 5.Left: Percentage of latency on memory processing for
parameterized memory (Titans/HMT, LaCT). Right: Percentage
of latency on memory processing for MemAgent.
4. Computational Heterogeneity
We further analyze the computational properties of the mem-
ory processing pipeline and reveal its heterogeneity. Quanti-
tatively, arithmetic intensity (FLOPs/byte) (Williams et al.,
2009) characterizes whether a step is memory-bound (fre-
quent data access) or compute-bound (more arithmetics),
while qualitatively, we examine data access patterns and
dependencies. Table 2 summarizes these properties for each
LLM inference optimization, with detailed arithmetic inten-
sity and latency breakdown provided in Appendix B.
Sparse Attention & RAG: Sparse attention and RAG exhibit
similar heterogeneity across memory processing steps. Com-
pute Relevancy and Retrieval are memory-bound (skinny
matrix-matrix and matrix-vector multiplication) and in-
volve irregular accesses and data dependencies, such as
BM25 scoring, top- kselection, and max reduction. For
instance, BM25 lookups token frequency histograms in
a non-deterministic order across documents, while top- k
4

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Table 2.Summary of LLM inference optimizations and computational properties of their memory processing pipelines for single batch.
We show arithmetic intensity (FLOPs/byte, higher means more compute-bound) with only orders of magnitude depicted (details in
Appendix B). “Local Memory” denotes independent computation across memory entries, and “Regular” indicates consecutive data access.
LLM Opt. Prepare Memory Compute Relevancy Retrieval Apply to Inference Rest Ops in LLM
Sparse AttentionArith. Intensity10–100 1–10110–100 1–10
Access PatternRegular Regular Irregular Regular Regular
Data RequirementLocal Memory Local Memory Across Memories Local Memory Local Memory
RAGArith. Intensity1–100 1–1010>100
Access PatternIrregular (tokenizer) Irregular Irregular Regular Regular
Regular (embeddings)
Data RequirementLocal Memory Across Memories Across Memories no calculations Local Memory
Synthesized MemoryArith. Intensity1–10-0>100>100
Access PatternRegular - Regular Regular Regular
Data RequirementAcross Memories - No Calculation Local Memory Local Memory
Memory as ContextArith. Intensity>100 1–1010>100
Access PatternRegular Regular Regular Regular Regular
Data RequirementLocal Memory Local Memory Across Memories Local Memory Local Memory
Test-time TrainingArith. Intensity>100 1–10->100>100
Access PatternRegular Regular - Regular Regular
Data RequirementLocal Memory Local Memory - Local Memory Local Memory
maintains running maximum scores with data dependencies
and irregular data eviction. In contrast, Prepare Memory
and Apply to Inference are primarily dense linear algebra
with consecutive and independent accesses, including linear
projections and attention operations.
Synthesized Memory: The memory processing of MemA-
gent is essentially a sequence of LLM inferences. At the
Prepare Memory step, the LLM decodes to generated textual
memory, which is a memory-bound operation. At the Apply
to Inference step, the LLM performs prefilling to consume
the memory and the text from the current segment. This
operation is compute-bound.
Memory as Context: Computations in Memory as Context
are similar to sparse attention and RAG, except that there
are extra calculations to generate the query from sequences
in Compute Relevancy. These calculations are independent
of the model forward pass and can be parallelized and fused
with other kernels.
TTT: The heterogeneity is insufficient. Although computing
the loss function (Compute Relevancy) is more memory-
intensive than the other steps, the latency bottleneck of mem-
ory processing in LaCT is dominated by compute-bound
operations (forward and backward pass). Thus, we do not
deploy it on the heterogeneous system.
This heterogeneity of memory processing pipeline in LLM
inference motivates the mapping of LLM onto a heteroge-
neous system as discussed in the next section.
5. GPU-FPGA Heterogeneous System
5.1. GPU vs. FPGA
GPUs are well-known for their efficiency in accelerating
LLMs with abundant highly parallel compute cores andlarge HBM bandwidth. However, GPUs suffer from under-
utilization of computational resources and off-chip mem-
ory bandwidth when processing irregular data accesses and
memory-bound operations (Boutros et al., 2020; Song et al.,
2022; Rajashekar et al., 2024; He et al., 2024). In contrast,
FPGAs allow users to customize the microarchitecture for
data control. The advantages of FPGAs over GPUs include:
1) larger SRAM capacity with higher bandwidth, 2) flexible
data control with minimized scheduling efforts, and 3) low
power consumption with competitive performance. These
features highlight the potential of FPGAs to accelerate mem-
ory processing of LLM inference, especially operations that
are not hardware-friendly to GPUs.
5.2. Heterogeneous System Overview
For demonstration, we consider a system with an FPGA
and a GPU, both equipped with HBM and connected via
PCIe. Our general mapping criteria for memory processing
steps and data are: 1) deploying steps based on the strengths
of the FPGA and GPU (i.e., compute-bounded and regular
data access on the GPU; irregular, data dependent, and
memory-bound operations on the FPGA), and 2) balancing
the trade-off between cross-device communication overhead
and kernel-level speedup. We prioritize criterion 2 over 1
for minimal end-to-end latency. For example, although
extracting the KV cache for top- ktokens is memory bound,
we do not fuse it with top- kselection on the FPGA because
the PCIe overhead outweighs the fusion benefit. Instead,
we transfer only the top- kindices to minimize PCIe latency
and perform KV cache extraction on the GPU. Figure 6
illustrates an overview of the mapping for different cases.
General Setup: This design applies for both sparse atten-
tion and RAG. We execute the Prepare Memory and Apply
Memory steps on the GPU and deploy a fused Compute Rel-
5

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 6.Kernel mapping and data communication on the GPU-FPGA system. (a) Sparse attention and RAG employ general setup, where
the GPU prepares the memory and apply to tthe inference and the FPGA execute an efficient fused kernel for compute relevancy and
retrieval. (b) For MemAgent, we utilize prefill-decode disaggregation where the FPGA operates LLM decoding and the GPU handles
prefilling. (c) For Memory as Context, we update the data mapping for locality: memory on the FPGA, retrieved memory delivered from
the FPGA to the GPU, and output is directly streamed to Prepare Memory.
evancy and Retrieval kernel on the FPGA (Figure 6(a)). The
GPU HBM stores the memory, target data, and output data,
while the FPGA stores the query and the processed memory
generated by the GPU. In sparse attention, the processed
memory is a compressed KV cache. The GPU transfers
the compressed KV cache for the entire input sequence dur-
ing prefilling and only the next token during decoding. In
RAG, Prepare Memory is a one-time process and amortized
across subsequent steps. After retrieval, the FPGA returns
the retrieved indices to the GPU for memory access.
Synthesized Memory: Similar to other works on prefill-
decode disaggregation between the GPU and the FPGA
(Yang et al., 2024a), we deploy Prepare Memory (LLM de-
coding) to FPGA and Apply Memory (LLM prefilling) to
GPU (Figure 6(b)). For each segment, the GPU delivers the
KV cache to the FPGA and the FPGA returns the token ids
for the synthesized memory to the GPU for concatenation.
Memory as Context: The kernel mapping follows the Gen-
eral Setup, but with a recurrent loop communication for
input segments. We revise data placement as follows: (1)
the retrieved memory embedding is transferred to the GPU,
which only incurs communication overhead comparable to
the retrieved index in the General Setup; (2) memory is
stored only on the FPGA, since GPU-side Prepare Memory
only requires the retrieved memory and the next input seg-
ment to generate new memory. This enables more kernel
fusion on the FPGA for higher efficiency, while sparing
GPU memory for model storage.
5.3. Kernel Design
For GPU kernels, we reuse existing optimized libraries,
while our design effort focuses on FPGA kernels, which re-
quire substantially more development time. For MemAgent
and Memory as Context, we adopt the design paradigm of
prior FPGA-based LLM accelerators (Yang et al., 2024a;
Zeng et al., 2024; He et al., 2025b;c; Zhang et al., 2026).
This section presents the general kernel design. We include
more technical detail in Appendix E.
Figure 7 illustrates the FPGA kernel architecture. Here, we
use DeepSeek Attention as an example for illustrating the
computations. One advantage of FPGAs over GPUs is that
Figure 7.Architecture of the FPGA kernel for General Setup. A
streaming dataflow design connecting two modules: (a) Compute
Relevancy uses fast SRAM (BRAM+URAM) and HBM to store
compressed keys, where an inner-product engine consumes queries
and computes scores; (b) Retrieval performs partial and weighted
sum across query heads and feeds results to a top- kretriever that
continuously maintains a running top-klist.
users can build streaming dataflow designs: executions are
data driven, reducing explicit control overhead and time-
consuming off-chip memory accesses. In General Setup,
modules are connected in a dataflow manner. The Compute
Relevancy module computes inner product scores between
each query head and all key vectors. All past key vectors are
stored in a three-level physical memory hierarchy: BRAM
with 21.8 TB/s, URAM with 10.4 TB/s, and HBM with 460
GB/s. The BRAM and URAM can store 40MB of data in
total for U55C. Each memory tier has a different capacity,
and the key loader writes through the memory hierarchy
with a write arbiter. Key vectors with smaller token IDs
are stored in faster memory to maintain high access speed.
Vectors are streamed to inner product engine to calculate
the score. The resulting scores are streamed to the Retrieval
module, which chains a reduction unit and a top- kretriever.
The reduction unit aggregates partial sums and computes the
final score per key. The top- kretriever maintains a running
top-klist by comparing incoming scores using a parallel
reduction tree. After scanning all keys, it outputs the indices
of the top-kscores.
5.4. Deployment
Traversing the methods in Table 1, we implement each step
(standalone or fused) as reusable kernels to form a library.
Existing methods use provided host binaries, while users can
build new algorithm by recombining kernels and interfacing
6

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
them through our GPU-FPGA communication API (e.g.,
block-based RAG with BM25 and max-reduction kernels).
Arbitrary methods may require custom kernels. We plan to
reduce this effort via design automation in our future work.
6. Evaluation
6.1. Experiment Setup
Hardware.For the heterogeneous system, we run experi-
ments on a node with an AMD Alveo U55C FPGA and an
AMD Instinct MI210 GPU, with AMD EPYC 7v13 as the
host CPU (detail in Appendix D). For the baseline, we use
the same GPU model for fair comparison. We do not have
physical access to a system with both NVIDIA A100 and
U55C on the same node, but we provide additional analysis
and performance estimation in Appendix H to demonstrate
the generalizability of our system. Notably, U55C is fabri-
cated in an older process technology than the GPU (16 nm
vs. 6 nm) and costs half as much. A newer FPGA model
(e.g., AMD Versal V80) can further improve performance.
Baseline Measurement.For each workload in Table 1, we
use the same datasets as in the original work. For sparse
attention, we measure per-token latency under varying past-
token lengths; for RAG, MemAgent, and Memory as Con-
text, we measure total request latency (prefill and decode)
under different input sequence lengths. We report both
end-to-end latency and the fraction attributable to memory
processing for the baselines, enabling quantitative evalua-
tion of the acceleration achieved by heterogeneous system.
Total latency is measured with Python performance coun-
ters, while PyTorch Profiler (Paszke et al., 2019) measures
the time fraction spent on memory processing. This is used
to derive the latency breakdown without interfered by trac-
ing overhead. To ensure fairness, all methods use optimized
implementations and identical experimental settings for la-
tency measurement and profiling. Specifically,
•DeepSeek Attention(Liu et al., 2025): we use vLLM
(Kwon et al., 2023) and modify the codebase to only load
the first layer of DeepSeek V3.2 Exp due to the limited GPU
memory of MI210.
•SeerAttention-R(Gao et al., 2025): we utilize TileLang
(Wang et al., 2025) optimized kernel. The base model is
Qwen 3 8B (Yang et al., 2025a) and block size is 64.
•LServe(Yang et al., 2025b): We use the LServe codebase
and employ HIPIFY (ROCm Organization, 2026) to port
custom CUDA kernels to HIP kernel for the AMD GPU.
The base model is Llama 3.1 8B (Grattafiori et al., 2024).
•RAG: For single-stage RAG (DRAGIN, FLARE, and
Fixed-sentence RAG), we follow the experiment setup in
DRAGIN (Su et al., 2024) and utilize Llama 2 7B (Touvron
et al., 2023) as the generator model. For two-stage RAG, weadhere the setup of RAG-EDA (Pu et al., 2024) with Llama
3.1 8B as the generator model.
•Memory as Context: Since there is no official Titans
(Behrouz et al., 2025) implemenetation, we employ the
open-source implementation of HMT and update the sum-
marization step into a linear projection to replicate Titans.
•MemAgent(Yu et al., 2025): We use Qwen 2.5 7B (Team
et al., 2024) as the base model and the default hyperparame-
ters defined in the codebase.
•TTT/LaCT(Zhang et al., 2025b; Sun et al., 2024): We use
the codebase for LaCT (Zhang et al., 2025b) for profiling
and benchmarking.
Details of Hyperparameters in Appendix D.
Tools.We utilize ROCm 6.3 for GPU kernel development.
FPGA kernels are designed using Xilinx Vitis HLS 2024.2
and implemented using Vivado 2024.2. P2P mode is en-
abled to allow DMA for the FPGA. Details for cross-device
communication is in Appendix C.
6.2. Latency
Figure 8.End-to-end speedup of the GPU-FPGA heterogeneous
system over the baseline for sparse attention mechanisms.
Figure 9.Speedup for the memory processing steps deployed on
the GPU-FPGA heterogeneous system for sparse attentions.
Each method benefits from the GPU-FPGA system in
three aspects: 1) the FPGA’s large, high-bandwidth on-
chip memory permits faster data access than the GPU; 2)
operations within or across memory processing steps are
7

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 10.Left: End-to-end speedup of the GPU-FPGA system
over the baseline for RAG. Right: Speedup of memory processing
for the single stage RAG (DRAGIN/FLARE/FS-RAG) and two
stage RAG. The reranker of the two stage RAG is executed on
GPU.
pipelined through a streaming dataflow, facilitating finer-
grained computation-communication overlap than on GPUs;
3) the flexible memory system design maximizes HBM
bandwidth utilization, achieving higher decoding through-
put even when GPUs offer higher peak bandwidth.
Case 1: Large On-chip Memory.By storing compressed key
vectors in FPGA on-chip memory (URAM and BRAM),
the U55C provides about 5×more effective bandwidth
than GPU SRAM for Compute Relevancy and Retrieval.
This yields 1.8–2.2× kernel speedup for SeerAttention-R
(top-k),2.6–4.9× with threshold, and 1.2–5.6× for LServe
(Figure 9), translating to a 1.04–1.49× end-to-end speedup.
When the sequence length exceeds 1M tokens, LServe and
DeepSeek Attention will experience a drop of speed on the
FPGA due to accessing the HBM. Practically, the system
can dynamically fall back to GPU-only execution to avoid a
performance loss.
Case 2: Pipelined and Flexible Datapath.The FPGA ex-
ploits fine-grained pipelining and optimized random access
to overlap communication and computation in BM25, top-
k, and dependent memory-processing steps. This benefit
generalizes across sparse attention, RAG, and Memory-as-
Context. For DeepSeek Attention, we achieve a 1.3–2.2×
speedup in memory processing and a 1.1–1.2× end-to-end
speedup (Figure 9). For RAG, single-stage methods improve
memory processing speeds by 5.1–6.6× over the BM25S
baseline (Figure 10), while two-stage RAG is limited to 1.1–
2.1× due to reranker dominance, resulting in up to 1.47–
1.84× end-to-end speedup. For Memory-as-Context, fusing
query generation with cross attention yields a 3.1–4.0×
memory-processing speedup and a 1.3–1.6× end-to-end
speedup (Figure 11).
Case 3: Faster Decoding.LLM decoding is fundamen-
tally memory-bound, and FPGA architectures can precisely
control HBM transactions to sustain higher effective band-
width than GPUs, whose peak bandwidth is often underuti-
lized during decoding. Since MemAgent relies on decod-
ing to generate memory, this advantage directly improves
system performance. As shown in Figure 12, under prefill–
decode disaggregation, the GPU–FPGA system consistentlyTable 3.The energy efficiency and the improvement of the GPU-
FPGA system over the baseline. DSA stands for DeepSeek Atten-
tion, and SA-R denotes for SeerAttention-R.
Category Method GPU-FPGA Baseline Max Geomean
(J/req. or J/tok) (J/req. or J/tok) Improve Improve
Sparse Attn.DSA 15.86 25.621.69×1.61×
SA-R (Thres.) 0.30 0.341.26×1.14×
SA-R (Top-k) 0.32 0.361.21×1.11×
LServe 0.29 0.431.62×1.43×
RAGDRAGIN 328.16 362.571.34×1.10×
FLARE 241.99 275.531.45×1.14×
FS-RAG 259.88 315.251.71×1.21×
Two-stage 150.33 160.681.23×1.07×
Synthesized
Memory MemAgent 3202 136624.66×4.66×
Memory-as
-Context HMT / Titans 16.55 27.311.74×1.65×
achieves a1.8×speedup over a GPU-only baseline.
We contain detailed analysis on the source of improvement
from the FPGA in Appendix F.
Figure 11.Left: End-to-end latency of the GPU-FPGA system
vs.the baseline for memory as context method. Right: Latency for
memory processing in memory as context. Similar to Titans, we
use a linear projection on the current segment for query generation.
Figure 12.Left: End-to-end latency of the GPU-FPGA heteroge-
neous system vs. the GPU-centric system for MemAgent. Right:
Latency for memory processing in MemAgent.
6.3. Energy Efficiency
Another important aspect of LLM inference is energy ef-
ficiency, which directly affects serving cost. Table 3 lists
the geomean energy efficiency of the GPU-FPGA system
and the baseline for each method mentioned in Section 6.2.
Overall, the GPU-FPGA system can achieve 1.11−1.61×
lower energy for sparse attention, 1.07−1.21× for RAG,
4.66× for MemAgent, and 1.65× for Memory as Context.
Furthermore, energy efficiency improvements generally in-
crease with memory size, except for DeepSeek Attention
and LServe due to the decreasing performance after 1M
8

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Table 4.Geomean speedup of the GPU-FPGA system over the
GPU-centric baseline across batch sizes for various LLM optimiza-
tion approaches.
Category Method BS=1 BS=2 BS=4 BS=8 BS=32
Sparse Attn.SA-R (Thres.) 1.12×1.21×1.33×1.47×1.60×
DSA 1.02×1.05×1.09×1.14×1.15×
SA-R (Top-k) 1.07×1.15×1.24×1.30×1.32×
LServe 1.19×1.34×1.51×1.67×1.83×
RAGDRAGIN 1.14×1.20×1.29×1.44×1.92×
FLARE 1.19×1.26×1.38×1.55×2.11×
FS-RAG 1.26×1.32×1.42×1.58×2.10×
2-Stage RAG 1.16×1.22×1.28×1.32×1.37×
Synthesized
MemoryMemAgent 1.85×1.65×0.93×0.49×0.13×
Memory-as
-ContextHMT/Titans 1.48×1.47×1.45×1.38×1.15×
tokens for HBM access (stop at 1.43× and1.07× respec-
tively). The energy efficiency improvement does not solely
come from the speedup: the FPGA kernels generally have a
lower operating power than the corresponding GPU kernels
(Appendix G).
6.4. Batch Inference Scaling
Table 4 reports the geometric mean speedup of the GPU-
FPGA system over GPU-centric baselines across different
batch sizes. We observe the speedup increases with batch
size for sparse attention and RAG methods, decreases for
Memory-as-Context, and slowdown for MemAgent. We
explain the reason of the batch inference scaling for these
methods as follows:
•Sparse attention.KV cache and latent indexing embed-
dings are not shared across samples within a batch (Zhao
et al., 2024), so increasing batch size does not improve data
reuse for score computations in sparse attention on GPUs.
The GPU-FPGA system can still exploit the advantages of
high HBM bandwidth utilization. In contrast, dense com-
ponents such as linear projections and feedforward layers
benefit from higher weight reuse. As batch size increases, a
larger fraction of latency is attributed to sparse attention, am-
plifying the benefit of offloading and resulting in increasing
speedup.
•RAG.Methods such as DRAGIN, FLARE, and FS-RAG
rely on lexical retrieval with BM25 scoring, where both data
access and computation are input-dependent and cannot be
shared across batch samples. Similar to sparse attention,
batching primarily improves GPU efficiency for dense com-
ponents but not retrieval. Consequently, the relative cost of
retrieval increases with batch size, leading to larger speedups
with GPU-FPGA systems. Two-stage RAG can obtain ben-
efits by batch inference through improved weight reuse in
the reranker, enhancing GPU utilization and moderating the
speedup gain.•Memory-as-Context.We offloaded the cross attention to
FPGA, which contains linear projections. With larger batch
sizes, these linear projections achieve higher weight reuse
and improved GPU utilization. This reduces the relative ad-
vantage of offloading, resulting in decreasing speedup. How-
ever, since the memory embeddings remain independent
across samples, FPGA acceleration still provides benefits
in long-sequence regimes for computing the cross attention
score and perform selection over the memory embeddings.
•MemAgent.The memory processing pipeline in MemA-
gent is a standard LLM inference. Under batching, the
decode stage significantly benefits from weight reuse on
GPUs. Given the lower compute throughput of FPGAs for
dense operations, this leads to performance degradation as
batch size increases.
For MemAgent, the system can dynamically select the op-
timal configuration. For example, when the batch size is
larger than 2 in MemAgent, we switch to a GPU-centric
deployment to avoid slowdown.
7. Conclusion
In this work, we demonstrate that a GPU-FPGA heteroge-
neous system can accelerate memory processing in LLM
inference. By unifying LLM inference optimizations into a
four-step memory processing pipeline, we expose its nontriv-
ial contribution to end-to-end latency and exploit its compu-
tational heterogeneity with a GPU-FPGA design, achieving
1.04∼2.2× speedup and 1.11∼4.7× energy reduction.
While our prototype uses off-the-shelf devices, a heteroge-
neous ASIC could further improve energy efficiency and
eliminate cross-device communication overhead.
8. Impact Statement
This paper presents work whose goal is to advance the field
of machine learning. There are many potential societal
consequences of our work, none of which we feel must be
specifically highlighted here.
References
Advanced Micro Devices, Inc. rocBLAS: Rocm basic linear
algebra subprograms library. https://rocm.docs.
amd.com/projects/rocBLAS/ , 2025. Version as
of retrieval; HIP-based BLAS optimized for AMD GPUs.
Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A.,
Arbus, E., Arora, R. K., Bai, Y ., Baker, B., Bao, H., et al.
gpt-oss-120b & gpt-oss-20b model card.arXiv preprint
arXiv:2508.10925, 2025.
Behrouz, A., Zhong, P., and Mirrokni, V . Titans: Learning to
9

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
memorize at test time.arXiv preprint arXiv:2501.00663,
2025.
Beltagy, I., Peters, M. E., and Cohan, A. Long-
former: The long-document transformer.arXiv preprint
arXiv:2004.05150, 2020.
Boutros, A., Nurvitadhi, E., Ma, R., Gribok, S., Zhao, Z.,
Hoe, J. C., Betz, V ., and Langhammer, M. Beyond
peak performance: Comparing the real performance of
ai-optimized fpgas and gpus. In2020 International Con-
ference on Field-Programmable Technology (ICFPT), pp.
10–19, 2020. doi: 10.1109/ICFPT51103.2020.00011.
Bulatov, A., Kuratov, Y ., and Burtsev, M. Recurrent memory
transformer.Advances in Neural Information Processing
Systems, 35:11079–11091, 2022.
Chen, H., Zhang, J., Du, Y ., Xiang, S., Yue, Z., Zhang, N.,
Cai, Y ., and Zhang, Z. Understanding the potential of
fpga-based spatial acceleration for large language model
inference.ACM Transactions on Reconfigurable Technol-
ogy and Systems, 18(1):1–29, 2024.
Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I.,
Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang,
D., Rosen, E., et al. Gemini 2.5: Pushing the frontier
with advanced reasoning, multimodality, long context,
and next generation agentic capabilities.arXiv preprint
arXiv:2507.06261, 2025.
Elasticsearch, B. Elasticsearch.software], version, 6(1),
2018.
Gao, Y ., Zeng, Z., Du, D., Cao, S., Zhou, P., Qi, J., Lai,
J., So, H. K.-H., Cao, T., Yang, F., et al. Seerattention:
Learning intrinsic sparse attention in your llms.arXiv
preprint arXiv:2410.13276, 2024.
Gao, Y ., Guo, S., Cao, S., Xia, Y ., Cheng, Y ., Wang, L., Ma,
L., Sun, Y ., Ye, T., Dong, L., et al. Seerattention-r: Sparse
attention adaptation for long reasoning.arXiv preprint
arXiv:2506.08889, 2025.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.
Gu, A. and Dao, T. Mamba: Linear-time sequence mod-
eling with selective state spaces. InFirst conference on
language modeling, 2024.
He, Z., Song, L., Lucas, R. F., and Cong, J. Levelst: Stream-
based accelerator for sparse triangular solver. InProceed-
ings of the 2024 ACM/SIGDA International Symposium
on Field Programmable Gate Arrays, pp. 67–77, 2024.He, Z., Cao, Y ., Qin, Z., Prakriya, N., Sun, Y ., and Cong,
J. Hmt: Hierarchical memory transformer for efficient
long context language processing. InProceedings of the
2025 Conference of the Nations of the Americas Chapter
of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp.
8068–8089, 2025a.
He, Z., Truong, A., Cao, Y ., and Cong, J. Intar: Inter-
task auto-reconfigurable accelerator design for high data
volume variation in dnns. In2025 IEEE 33rd Annual
International Symposium on Field-Programmable Cus-
tom Computing Machines (FCCM), pp. 123–132. IEEE,
2025b.
He, Z., Ye, S., Ma, R., Wang, Y ., and Cong, J. Lut-llm: Effi-
cient large language model inference with memory-based
computations on fpgas.arXiv preprint arXiv:2511.06174,
2025c.
Jiang, H., Li, Y ., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han,
Z., Abdi, A. H., Li, D., Lin, C.-Y ., et al. Minference
1.0: Accelerating pre-filling for long-context llms via dy-
namic sparse attention.Advances in Neural Information
Processing Systems, 37:52481–52515, 2024.
Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-
Yu, J., Yang, Y ., Callan, J., and Neubig, G. Active re-
trieval augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing, pp. 7969–7992, 2023.
Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The
efficient transformer.arXiv preprint arXiv:2001.04451,
2020.
Kwon, W., Li, Z., Zhuang, S., Sheng, Y ., Zheng, L., Yu,
C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient
memory management for large language model serving
with pagedattention. InProceedings of the 29th sym-
posium on operating systems principles, pp. 611–626,
2023.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C.,
Dengr, C., Ruan, C., Dai, D., Guo, D., et al. Deepseek-v2:
A strong, economical, and efficient mixture-of-experts
language model.arXiv preprint arXiv:2405.04434, 2024.
Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu,
B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3.
2: Pushing the frontier of open large language models.
arXiv preprint arXiv:2512.02556, 2025.
10

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
L`u, X. H. Bm25s: Orders of magnitude faster lexi-
cal search via eager sparse scoring.arXiv preprint
arXiv:2407.03618, 2024.
Moreira, G. d. S. P., Ak, R., Schifferer, B., Xu, M.,
Osmulski, R., and Oldridge, E. Enhancing q&a text
retrieval with ranking models: Benchmarking, fine-
tuning and deploying rerankers for rag.arXiv preprint
arXiv:2409.07691, 2024.
Niu, T., Joty, S., Liu, Y ., Xiong, C., Zhou, Y ., and
Yavuz, S. Judgerank: Leveraging large language mod-
els for reasoning-intensive reranking.arXiv preprint
arXiv:2411.00142, 2024.
NVIDIA Corporation. cuBLAS: Cuda basic linear al-
gebra subprograms library. https://developer.
nvidia.com/cublas , 2025. Version as of retrieval;
GPU-accelerated BLAS on NVIDIA GPUs, part of
CUDA Toolkit.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., et al. Pytorch: An imperative style, high-performance
deep learning library.Advances in neural information
processing systems, 32, 2019.
Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho,
S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella,
M., et al. Rwkv: Reinventing rnns for the transformer era.
arXiv preprint arXiv:2305.13048, 2023.
Pu, Y ., He, Z., Qiu, T., Wu, H., and Yu, B. Customized
retrieval augmented generation and benchmarking for
eda tool documentation qa. InProceedings of the 43rd
IEEE/ACM International Conference on Computer-Aided
Design, pp. 1–9, 2024.
Rajashekar, M. B., Tian, X., and Fang, Z. Hispmv: Hy-
brid row distribution and vector buffering for imbal-
anced spmv acceleration on fpgas. InProceedings of
the 2024 ACM/SIGDA International Symposium on Field
Programmable Gate Arrays, pp. 154–164, 2024.
ROCm Organization. Hipify: Convert cuda to portable c++
code, 2026. URL https://github.com/ROCm/
HIPIFY . GitHub repository, last accessed January 2026.
Shen, W., Yang, Z., Li, C., Lu, Z., Peng, M., Sun, H., Shi, Y .,
Liao, S., Lai, S., Zhang, B., et al. Qwenlong-l1. 5: Post-
training recipe for long-context reasoning and memory
management.arXiv preprint arXiv:2512.12967, 2025.
Sheng, Y ., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen,
B., Liang, P., R ´e, C., Stoica, I., and Zhang, C. Flexgen:
High-throughput generative inference of large language
models with a single gpu. InInternational Conference
on Machine Learning, pp. 31094–31116. PMLR, 2023.Song, L., Chi, Y ., Guo, L., and Cong, J. Serpens: A high
bandwidth memory based accelerator for general-purpose
sparse matrix-vector multiplication. InProceedings of
the 59th ACM/IEEE design automation conference, pp.
211–216, 2022.
Su, W., Tang, Y ., Ai, Q., Wu, Z., and Liu, Y . Dragin:
dynamic retrieval augmented generation based on the in-
formation needs of large language models.arXiv preprint
arXiv:2403.10081, 2024.
Sun, Y ., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G.,
Dubois, Y ., Chen, X., Wang, X., Koyejo, S., et al. Learn-
ing to (learn at test time): Rnns with expressive hidden
states.arXiv preprint arXiv:2407.04620, 2024.
Team, Q. et al. Qwen2 technical report.arXiv preprint
arXiv:2407.10671, 2(3), 2024.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y ., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models.arXiv preprint arXiv:2307.09288,
2023.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions. InPro-
ceedings of the 61st annual meeting of the association for
computational linguistics (volume 1: long papers), pp.
10014–10037, 2023.
Wang, G., Xie, Y ., Jiang, Y ., Mandlekar, A., Xiao, C., Zhu,
Y ., Fan, L., and Anandkumar, A. V oyager: An open-
ended embodied agent with large language models.arXiv
preprint arXiv:2305.16291, 2023.
Wang, L., Cheng, Y ., Shi, Y ., Tang, Z., Mo, Z., Xie, W.,
Ma, L., Xia, Y ., Xue, J., Yang, F., et al. Tilelang: A com-
posable tiled programming model for ai systems.arXiv
preprint arXiv:2504.17577, 2025.
Wang, X. and Zhou, D. Chain-of-thought reasoning without
prompting.Advances in Neural Information Processing
Systems, 37:66383–66409, 2024.
Williams, S., Waterman, A., and Patterson, D. Roofline:
an insightful visual performance model for multicore
architectures.Communications of the ACM, 52(4):65–76,
2009.
Wu, Y ., Liang, S., Zhang, C., Wang, Y ., Zhang, Y ., Guo,
H., Tang, R., and Liu, Y . From human memory to ai
memory: A survey on memory mechanisms in the era of
llms.arXiv preprint arXiv:2504.15965, 2025.
Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. C-pack:
Packaged resources to advance general chinese embed-
ding, 2023.
11

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025a.
Yang, F., Yang, X., Wang, H., Wang, Z., Zhu, Z., Zeng, S.,
and Wang, Y . Glitches: Gpu-fpga llm inference through a
collaborative heterogeneous system. In2024 IEEE High
Performance Extreme Computing Conference (HPEC),
pp. 1–7. IEEE, 2024a.
Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta net-
works: Improving mamba2 with delta rule.arXiv preprint
arXiv:2412.06464, 2024b.
Yang, S., Guo, J., Tang, H., Hu, Q., Xiao, G., Tang, J., Lin,
Y ., Liu, Z., Lu, Y ., and Han, S. Lserve: Efficient long-
sequence llm serving with unified sparse attention.arXiv
preprint arXiv:2502.14866, 2025b.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . React: Synergizing reasoning and
acting in language models. InThe eleventh international
conference on learning representations, 2022.
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y .,
and Narasimhan, K. Tree of thoughts: Deliberate problem
solving with large language models.Advances in neural
information processing systems, 36:11809–11822, 2023.
Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang,
Y .-Q., Ma, W.-Y ., Liu, J., Wang, M., et al. Memagent: Re-
shaping long-context llm with multi-conv rl-based mem-
ory agent.arXiv preprint arXiv:2507.02259, 2025.
Zeng, S., Liu, J., Dai, G., Yang, X., Fu, T., Wang, H., Ma,
W., Sun, H., Li, S., Huang, Z., et al. Flightllm: Efficient
large language model inference with a complete mapping
flow on fpgas. InProceedings of the 2024 ACM/SIGDA
International Symposium on Field Programmable Gate
Arrays, pp. 223–234, 2024.
Zhang, D., Li, W., Song, K., Lu, J., Li, G., Yang, L., and Li,
S. Memory in large language models: Mechanisms, eval-
uation and evolution.arXiv preprint arXiv:2509.18868,
2025a.
Zhang, J., He, Z., Fraser, N., Blott, M., Sun, Y ., and Cong,
J. Flexllm: Composable hls library for flexible hybrid
llm accelerator design.arXiv preprint arXiv:2601.15710,
2026.
Zhang, T., Bi, S., Hong, Y ., Zhang, K., Luan, F., Yang, S.,
Sunkavalli, K., Freeman, W. T., and Tan, H. Test-time
training done right.arXiv preprint arXiv:2505.23884,
2025b.
Zhang, Y ., Long, D., Xu, G., and Xie, P. Hlatr: en-
hance multi-stage text retrieval with hybrid list awaretransformer reranking.arXiv preprint arXiv:2205.10569,
2022.
Zhao, Y ., Lin, C.-Y ., Zhu, K., Ye, Z., Chen, L., Zheng, S.,
Ceze, L., Krishnamurthy, A., Chen, T., and Kasikci, B.
Atom: Low-bit quantization for efficient and accurate llm
serving.Proceedings of Machine Learning and Systems,
6:196–209, 2024.
12

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
A. Related Works
Surveys on memory in LLM inference.Recent surveys on LLM memory provide comprehensive characterizations of
memory types and their usage patterns. Wu et al. draw parallels between LLM memory and human cognition, proposing a
three-dimensional, eight-quadrant (3D-8Q) taxonomy based on memory origin, representation form, and retention duration.
Zhang et al. evaluate memory effectiveness across four categories: parametric memory (model weights), contextual memory
(KV caches), external memory (indexed vectors), and procedural memory (event stores). However, these surveys do not
articulate the shared procedural structure by which memory is generated, accessed, and updated. This omission hinders a
systematic understanding of memory processing in LLM inference and limits opportunities for optimizations.
Sparse attention.Longformer (Beltagy et al., 2020) first propose an effective sparse attention that combines the sliding
window attention and a global token attention. Reformer (Kitaev et al., 2020) employs local sensitive hashing to select the
relevant tokens to compute the attention scores. MInference (Jiang et al., 2024) employs mixed static-dynamic sparsity
across heads for high precision sparse attention at prefill stage. SeerAttention (Gao et al., 2024) trains an auxiliary predictor
to identify important blocks. These methods primarily optimize input processing rather than decoding.
Advanced RAG systems.JudgeRank (Niu et al., 2024) leverages prompt engineering to distill key information from each
document, utilizing users’ queries before passing to the reranker. HLATR (Zhang et al., 2022) adds a second reranking pass:
it concatenates all candidate documents selected during the initial retrieval stages and then computes multiple similarity
scores over this combined text using the second reranker model. By scoring the concatenated content all at once, HLATR
provides more holistic relevance judgments.
Context compression and recurrent models.QwenLong-L1.5 (Shen et al., 2025) extends MemAgent by introducing
planning tokens for structured memory generation. RMT (Bulatov et al., 2022) proposes a recurrent memory transformer
that compresses sequences into embeddings but lacks explicit retrieval mechanisms. Mamba (Gu & Dao, 2024) introduces a
selective state space model that enables efficient long-context processing by replacing attention with recurrent state updates.
Gated DeltaNet (Yang et al., 2024b) improves Mamba by controlling state updates through learnable gates. RWKV (Peng
et al., 2023) unifies RNN and Transformer paradigms by using time-mixing and channel-mixing mechanisms, enabling
constant-memory inference.
B. Detail Computation Properties of Memory Processing Pipeline
Figures 13 and 14 illustrates the arithmetic intensity of memory processing in each LLM inference optimization. Higher
arithmetic intensity means the operation is more compute bounded. The arithmetic intensity and computation pattern
analysis are widely discussed in prior works (Chen et al., 2024) related to LLM inference acceleration. For the latency
distribution among each step of memory processing pipeline:
•Sparse Attention & RAG: the bottleneck is compute relevancy and retrieval, and the proportion of latency is increasing as
the memory size grows.
•MemAgent: the bottleneck is prepare memory, which is essentially LLM decoding.
•Memory as Context: similar to sparse attention and RAG, the bottleneck is compute relevancy and retrieval, but the
proportion of latency grows slower than them.
•TTT/LaCT: the bottleneck is prepare memory and apply to inference, which is the LaCT block forward and backward pass.
For sparse attention, RAG, MemAgent, and Memory as Context methods, the dominant latency originates from memory-
centric and irregular operations such as relevance computation and top- kretrieval. By offloading these bottleneck steps
to the FPGA, which enables customized data paths and fine-grained control, the system mitigates GPU inefficiencies on
memory-bounded workloads and shortens the critical path of inference. Consequently, this mapping yields a substantial
improvement in end-to-end inference speed.
C. Cross-device Communication
For generalization across different vendors of FPGAs and GPUs, we consider PCI Express (PCIe) as the interconnect
standard to communicate between devices. The devices are installed in the same node with the same root complex to
13

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 13.Arithmetic intensity (FLOPs/byte) of memory processing pipeline and the rest operations in LLM inference for sparse attention,
single-stage RAG, Memory as Context, and TTT/LaCT. For RAG, prepare memory is a one-time operation and will be amortized with
multiple queries.
minimize PCIe overhead. The standard method of communicating FPGA with GPU is through memory copy runtime APIs
of each device and utilize CPU and system DRAM to handle the control (Yang et al., 2024a) (Figure 15(a)). This method can
only achieve 1/20 of the peak PCIe bandwidth. In this work, we configure the system alternatively using PCIe peer-to-peer
(P2P) data transfer: Both FPGA’s and GPU’s HBM can initiate direct memory access (DMA) to the pinned memory buffer
(non-swappable by the operating system) allocated on CPU (Figure 15(b)). The transaction will involve two hops of DMA
with CPU as the intermediate, bypassing the system DRAM.
A common concern regarding PCIe-based data transfer is its limited bandwidth. Compared to GPU–GPU communication
over NVLink (600 GB/s for NVLink 3.0), PCIe provides substantially lower throughput (32 GB/s for PCIe 3.0). However,
our profiling results indicate that the communication overhead introduced by PCIe in our configuration is sufficiently low
and more than compensated for by the performance gains achieved through our customized FPGA kernels (Section 6.2).
C.1. PCIe Latency
One concern of GPU-FPGA system in accelerating memory processing in LLM inference is the PCIe overhead between
the FPGA and the GPU. Figure 16 depicts the latency profiling of PCIe latency. For sparse attention and RAG, the data
transferred between the FPGA and the GPU are retrieved index and processed memory, which are under 1KB per request.
Consequently, the transaction latency is in the order of microseconds (9us). For MemAgent and Memory as Context, the
transferred embeddings has a size in the order of MBs, which takes microseconds to miliseconds to deliver. PCIe overhead
is not the bottleneck because methods that require larger data movement also exhibit substantially longer end-to-end latency,
making PCIe overhead relatively small in comparison. The following numbers provide an order-of-magnitude comparison
between PCIe latency and corresponding GPU kernel latency:
•Sparse Attention:Transfers include new key indexing vectors and retrieved indices, taking approximately 12 µs. The
corresponding GPU kernels take 128–2450 µs.
•RAG:Transfers only include retrieved indices, taking approximately 7 µs. The corresponding GPU kernels take
23–1596 ms.
•Memory-as-Context:Transfers include memory, query, and retrieved embeddings for each segment, taking approxi-
mately 20–320 µs. The corresponding GPU kernels take 26–498 ms.
14

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 14.Arithmetic intensity of memory rocessing pipeline and the rest operations in LLM inference for two-stage RAG. Each stage
has a compute relevancy and retrieval step.
Figure 15.(a) Standard cross device data transfer using memory copy for heterogeneous system. (b) PCIe P2P data transfer to bypass
system DRAM accesses.
•MemAgent:Transfers include KV cache and token IDs for each segment, taking approximately 14–218 ms. The
corresponding GPU kernels take 17–534 s.
These comparisons show that PCIe communication overhead remains small ( 1000x difference) relative to computation time
across all methods, even when data transfer size increases.
D. Detail Experiment Settings
The system specification is shown in Table 5.
Table 5.Hardware Specification
Device Model Memory
CPU AMD EPYC 7v13 @ 2.5GHz (7 nm) 1 TB (51.2 GB/s)
GPU AMD Instinct MI210 (6 nm) 64 GB (1.6 TB/s)
FPGA AMD Alveo U55C (16 nm) 16 GB (460 GB/s)
DeepSeek AttentionDeepSeek attention (Liu et al., 2025) introduces a lightning indexer to the Multi-headed Latent
Attention (MLA) (Liu et al., 2024) module. For every compressed query embedding and KV latent embedding, it first
generates 64 query heads and a key indexing vector by applying partial RoPE embedding, computes the dot products
15

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 16.Transfer latency on PCIe bus against the transfer data size. For transferring indices (KB) and single-token KV embeddings
(MB) are in the order of us, which is negligible compared to the latency of memory processing.
between key vectors and all query heads, and weight-average them based on the query weights derived from the input token.
The final scores are used for top- kselection for MLA module to attend the token, where kis 2048 in the DeepSeek V3.2
Exp model.
We utilize the vLLM (Kwon et al., 2023) optimized DeepSeek V3.2 Exp inference kernel for the GPU baseline. Since the
original DeepSeek V3.2 Exp is larger than the HBM capacity of MI210 GPU, we modify vLLM to load only the first layer
of the model. Since kof DeepSeek attention is constant across layers, we calculate the end-to-end latency by multiplying
the first-layer latency with the number of layers.
SeerAttention-RDuring inference, SeerAttention-R (Gao et al., 2025) first down project the query vectors and key vectors
with average pooling, then compute the dot products between the the projected queries and keys. Each score is for a block of
tokens. The attention module only attend the current tokens to the selected blocks. During inference, user can choose to
determine the retrieved tokens by either a token budget with top- kselection, or a threshold based selection (select if the
score is above the threshold).
We utilize the TileLang (Wang et al., 2025) optimized kernel for SeerAttention-R. The base model is Qwen 3 8B. We set the
block size to 64, token budget to 4096 for top-kmode, and threshold to 5e-4 for threshold mode.
LServeLServe (Yang et al., 2025b) organize blocks of tokens into logical pages and physical pages, where each physical
page can have multiple logical page. Each logical page is represented by two vectors: one with the minimum and one
with maximum values for each channel. The score is computed by finding the maximum dot products between the query
and these two vectors. Selection is in the granularity of the physical page, where the score for each physical page is the
maximum score of logical pages.
To use the CUDA kernels in the LServe codebase, we use HIPIFY (ROCm Organization, 2026) to port them to HIP
kernel and compile them with hipcc to deploy on the MI210 GPU. We use the default hyperparameters in their original
benchmarking script for profiling.
DRAGIN, FLARE, and Fixed-sentence RAGFollowing the experiment setup in DRAGIN (Su et al., 2024), all three
methods utilize Llama 2 7B (Touvron et al., 2023) as the generator model and BM25 indexing as the retrieval heuristic. The
original retriever backend is ElasticSearch (Elasticsearch, 2018). We replace it with a faster backend specific for BM25
indexing (BM25S (L `u, 2024)). The system will retrieve 64 documents and the maximum token generated is 32.
Two-stage RAGA two-stage RAG first execute an hybrid search (semantic embedding and BM25 lexical search) to retrieve
the top- Nrelevant documents, then filter the documents using a reranker to obtain the top- kdocuments in the selected
Ndocuments. The reranker is usually a transformer model. In the experiment, we follow the setup in RAG-EDA (Pu
et al., 2024), where the first stage comprises an embedding model ( bge-large-en-v1.5 (Xiao et al., 2023)) and a
BM25 indexer, and the second stage is a reranker ( bge-reranker-large (Xiao et al., 2023)). The first stage select 64
documents and the second stage select 10 documents. The maximum token generated is 32.
Memory as contextIn Titans (Behrouz et al., 2025), Memory as Context is a type of recurrent models that chunk sequence
16

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
into segments, utilize soft prompts to generate latent embeddings as memory, and convert each segment into query embedding
to find relevant embeddings stored in the past. In the experiment, we set the segment length is set to 1024 tokens and the
output sequence length is 32.
MemAgentFollowing the experiment setup in Yu et al., we set the segment length to 5000 tokens, the memory size to 1024
tokens, and the output max token length to 32.
E. FPGA Kernel Design
For GPU kernels, we use (cuBLAS/cuSparse (NVIDIA Corporation, 2025) and rocBLAS/rocSparse (Advanced Micro
Devices, Inc., 2025)) for linear operations and custom CUDA/HIP kernels for non-linear operations to ensure that steps
deployed on the GPU achieve state-of-the-art performance.
For Memory as Context, we follow the HMT plugin design in FlexLLM (Zhang et al., 2026). The FPGA loads the segment
embeddings to HBM from CPU when the input is streaming. As show in Figure 17, the segment loader will read the segment
embeddings on chip and stream to the query linear projection module to generate the query vector. At the same time, the
memory loader read past generated memory embeddings and perform cross attention with the query to extract and amplify
the memory that are most relevant to the current segment. The output memory will be write back to the HBM and deliver to
the GPU for processing the proceeding segments. Each modules are connected with FIFO streams.
Figure 17.FPGA kernel architecture for Memory as Context method. The kernel is a dataflow design with each module fully data driven
compute a single operation. Past memory embeddings are cached in the HBM and the segment embeddings are loaded from CPU directly
for each incoming segment.
For MemAgent, the FPGA only perform the LLM decoding. Therefore, we can directly follow the designs in previous works
(Zeng et al., 2024; He et al., 2025c; Yang et al., 2024a) and optimize them to specialize for decoding, e.g., the attention is
a sequence of GEMV operations and we can increase the parallelism in the hidden dimension. Figure 18 illustrates the
overall architecture. We follow the similar design as FlightLLM (Zeng et al., 2024) with separate special function units
(SwiGLU, LayerNorm) and matrix/vector multiplication engines (Linear Projection, Attention). However, we align with the
LUT-LLM (He et al., 2025c) with separate attention and linear projection engines since attention require higher precision
than linear projections to maintain accuracy. A global buffer is used to store partial weight matrices and intermediate data.
Data are streamed in each engine to reduce on-chip data store requirements, and executions are sequentially scheduled
between engines to ensure high computational throughput.
F. FPGA Kernel Improvement Analysis
Case 1: Large On-chip Memory.For block-sparse attention, compressed key vectors are either fully stored (SeerAttention-R)
or partially cached (LServe) in FPGA on-chip memory. The aggregated on-chip memory (URAM + BRAM) bandwidth
on U55C is about 5×higher than the effective SRAM bandwidth of MI210, with each bank managed as a scratchpad
to maximize target data access bandwidth. Since the Compute Relevancy and Retrieval stages are memory-bound, this
yields 1.8–2.2× speedup for SeerAttention-R with top- k,2.6–4.9× with threshold, and 1.2–5.6× for LServe (Figure 9),
translating to 1.04–1.25× and1.15–1.49× end-to-end speedups for SeerAttention-R and LServe. LServe performance
degrades beyond 256K tokens and becomes worse than the GPU at 1M tokens. This is because the FPGA kernel starts to
read from HBM, while the GPU’s HBM utilization scales with sequence length and eventually surpasses that of the FPGA
at 1M tokens. Therefore, the systemdynamically falls backto GPU-only execution for sequences longer than 1M tokens.
Case 2: Pipelined and Flexible Datapath.The memory processing pipeline has strong data dependency between steps.
17

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 18.FPGA kernel architecture for MemAgent. Following the design in prior works (Zeng et al., 2024; He et al., 2025c;b), we
design the kernel specialized to LLM decoding, where the KV cache is delivered from the GPU through PCIe. Linear projection are
executed in INT4 to align with the weight precision, and the rest operations (attention, SwiGLU, LayerNorm) are calculated at FP32 to
maintain accuracy of the model.
Moreover, operations such as BM25 and top- kselection involve irregular data access pattern as illustrated in Section 4.
With custom logic for fine-grained pipelining and optimized random access, the U55C FPGA can better reduce and overlap
communication and computation latency than GPUs. Some methods (e.g., RAG) adopt CPU offloading as a baseline to
accelerate these operations relative to GPU execution. Nevertheless, the U55C still achieves higher performance due to its
3.5×higher peak TOPs and its substantially higher HBM bandwidth compared to system DRAM.
We observe that these benefits generalize across sparse attention, RAG, and Memory as Context methods. For sparse
attention, we pipeline index score computation with top- k/threshold-based selection, achieving a 1.3–2.2× speedup in
memory processing for DeepSeek Attention, resulting in a 1.1–1.2× end-to-end speedup. Note that both LServe and
DeepSeek attention require reading key vectors from HBM for a subset of tokens, incurring lower bandwidth than on-chip
memory. Similar to LServe, the system dynamically falls back to GPU-only execution when the sequence length exceeds
1M. For RAG (Figure 10), we pipeline BM25 score computation with embedding search and top- kselection. For single-
stage RAG methods (DRAGIN, FLARE, and FS-RAG), the GPU-FPGA system achieves a 5.1–6.6× speedup in memory
processing over the baseline with the state-of-the-art BM25S kernel. In contrast, for two-stage RAG, the speedup drops to
1.1–2.1× due to the reranker dominating GPU execution time, leading to an overall end-to-end speedup of up to 1.47–1.84× .
For Memory as Context methods, we fuse query generation, which is a linear projection on the current segment embeddings,
with cross attention that outputs memory embeddings. Figure 11 compares end-to-end and memory-processing latency for
Memory as Context between GPU-FPGA system and the baseline. The GPU-FPGA system achieves a 3.1−4.0× speedup
for memory processing, resulting in a1.3−1.6×end-to-end speedup.
Case 3: Faster Decoding.Prior studies have shown that FPGAs can achieve faster LLM decoding than GPUs despite
having fewer raw computational resources (Zeng et al., 2024; He et al., 2025b;c). This advantage arises because LLM
decoding is fundamentally memory-bound, and FPGAs enable customized architectures that precisely control off-chip
memory transactions to fully exploit HBM bandwidth. In contrast, although GPUs provide higher peak HBM bandwidth,
this bandwidth is often underutilized even with highly optimized kernels (Zeng et al., 2024). Since MemAgent relies on
LLM decoding to generate memory, the decoding efficiency of FPGAs directly translates into substantial system-level
benefits. As shown in Figure 12, under the prefill-decode disaggregation scheme, the GPU-FPGA heterogeneous system
consistently achieves a1.8×speedup over a GPU-only baseline.
G. Expanded Results for Sparse Attention and RAG
Figures 19, 20, 21, and 22 illustrates the absolute value of latency for end-to-end inference and memory processing of sparse
attention and RAG. Figures 23 and 24 show the corresponding energy efficiency comparison in joule per token (for sparse
attention) and joule per request (for RAG).
18

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
For kernel power consumption:
• DeepSeek Attention: U55C: 26.4 W, MI210: 55 W
• SeerAttention-R threshold: U55C: 24.9 W, MI210: 45 W
• SeerAttention-R top-k: U55C: 25.3 W, MI210: 46 W
• LServe: U55C: 26.2 W, MI210: 47 W
• RAG: U55C 29.7 W, MI210: 106 W, EPYC 7v13: 34 W
• MemAgent: U55C: 44.2 W, MI210: 99 W
• HMT/Titans: U55C: 42.6 W, MI210: 94 W
Figure 19.End-to-end latency of each sparse attention mechanism on the baseline and GPU-FPGA system with respect to the sequence
length.
H. Results with NVIDIA A100
Compared with the AMD MI210, the NVIDIA A100 is a more widely adopted GPU for LLM inference. Although we do not
have access to a platform that hosts both the A100 and the U55C within the same node, we estimate the end-to-end latency
by aggregating the measured latency components of the FPGA, GPU, and PCIe communication, while profiling kernel
execution latency separately. Figures 25 and 26 present a representative case study using DeepSeek Attention. The results
show that, even when paired with the MI210, the GPU–FPGA heterogeneous system can outperform the A100 in certain
configurations. For end-to-end latency, since the A100 generally outperforms the MI210 under identical optimizations, the
MI210+U55C configuration can be slower than an A100-only system. However, when the GPU is upgraded to the A100, the
heterogeneous system continues to deliver consistent speedups. This demonstrates that the proposed heterogeneous system
is effective and largely agnostic to the specific GPU model.
19

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 20.Latency of memory processing in each sparse attention mechanism with respect to the sequence length.
Figure 21.End-to-end latency of each RAG system on the baseline system and GPU-FPGA system with respect to the document counts.
20

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 22.Latency of memory processing in single stage RAG using BM25 as the retrieval heuristic (DRAGIN, FLARE, FS-RAG) and
two stage RAG with respect to the document counts.
Figure 23.Energy efficiency of sparse attention mechanisms in Joule per token.
21

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 24.Energy efficiency of RAG systems in Joule per request.
Figure 25.The relative speedup of end-to-end inference of DeepSeek V3.2 Exp with DeepSeek Attention when deployed on MI210,
A100, MI210 + U55C, and A100 + U55C (estimated). A100 is generally faster in LLM inference than MI210. When integrating U55C
with A100, the GPU-FPGA heterogeneous system can still speed up the inference.
22

Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference
Figure 26.The relative speedup of memory processing in DeepSeek Attention when deployed on MI210, A100, MI210 + U55C, and
A100 + U55C (estimated). Even with MI210, the GPU-FPGA heterogeneous system can still outperform A100.
23