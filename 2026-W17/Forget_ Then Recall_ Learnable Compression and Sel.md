# Forget, Then Recall: Learnable Compression and Selective Unfolding via Gist Sparse Attention

**Authors**: Yuzhen Mao, Michael Y. Li, Emily B. Fox

**Published**: 2026-04-22 04:22:32

**PDF URL**: [https://arxiv.org/pdf/2604.20920v1](https://arxiv.org/pdf/2604.20920v1)

## Abstract
Scaling large language models to long contexts is challenging due to the quadratic computational cost of full attention. Mitigation approaches include KV-cache selection or compression techniques. We instead provide an effective and end-to-end learnable bridge between the two without requiring architecture modification. In particular, our key insight is that interleaved gist compression tokens -- which provide a learnable summary of sets of raw tokens -- can serve as routing signals for sparse attention. Building on this, we introduce selective unfolding via GSA, which first compresses the context into gist tokens, then selects the most relevant gists, and subsequently restores the corresponding raw chunks for detailed attention. This yields a simple coarse-to-fine mechanism that combines compact global representations with targeted access to fine-grained evidence. We further incorporate this process directly into training in an end-to-end fashion, avoiding the need for external retrieval modules. In addition, we extend the framework hierarchically via recursive gist-of-gist construction, enabling multi-resolution context access with logarithmic per-step decoding complexity. Empirical results on LongBench and RAG benchmarks demonstrate that our method consistently outperforms other compression baselines as well as inference-time sparse attention methods across compression ratios from $8\times$ to $32\times$. The code is available at: https://github.com/yuzhenmao/gist-sparse-attention/

## Full Text


<!-- PDF content starts -->

Forget, Then Recall: Learnable Compression and
Selective Unfolding via Gist Sparse Attention
Yuzhen Mao, Michael Y. Li, Emily B. Fox
Stanford University
{yuzhenm, michaelyli, ebfox}@stanford.edu
Abstract
Scaling large language models to long contexts is challenging due to the quadratic computational cost of full
attention. Mitigation approaches include KV-cache selection or compression techniques. We instead provide
an effective and end-to-end learnable bridge between the two without requiring architecture modification. In
particular, our key insight is that interleaved gist compression tokens—which provide a learnable summary of
sets of raw tokens—can serve as routing signals for sparse attention. Building on this, we introduce selective
unfolding via GSA, which first compresses the context into gist tokens, then selects the most relevant gists, and
subsequently restores the corresponding raw chunks for detailed attention. This yields a simple coarse-to-fine
mechanism that combines compact global representations with targeted access to fine-grained evidence. We
further incorporate this process directly into training in an end-to-end fashion, avoiding the need for external
retrieval modules. In addition, we extend the framework hierarchically via recursive gist-of-gist construction,
enabling multi-resolution context access with logarithmic per-step decoding complexity. Empirical results on
LongBench and RAG benchmarks demonstrate that our method consistently outperforms other compression
baselines as well as inference-time sparse attention methods across compression ratios from 8×to32×. The
code is available athttps://github.com/yuzhenmao/gist-sparse-attention/.
1 Introduction
Attention can capture rich interdependencies across a context, at the cost of quadratic complexity.
However, in many realistic long-context settings, these dependencies are highly structured rather
than uniformly dense. Treating all tokens as fully interacting with one another is therefore both
computationally inefficient and misaligned with the underlying structure of the data. Thus, we ask,
“Why not turn compression into routing: compress first, then selectively unfold the right details?”
Long-context modeling has emerged as a critical capability for next-generation large language
models (LLMs), driven by diverse real-world applications including in-depth reasoning (Guo et al.,
2025, Zelikman et al., 2022), repository-level code generation (Zhang et al., 2023a), and multi-turn
autonomous agent systems (Park et al., 2023). However, the quadratic computational cost of standard
attention mechanisms poses a fundamental challenge for scaling to long sequences. As context
lengths grow from thousands to millions of tokens, the attention computation becomes increasingly
dominant in the overall cost, presenting significant obstacles for both training and inference.
For example, in retrieval-augmented generation (RAG), the context typically consists of multiple
retrieved documents, most of which are irrelevant to a given query, yet standard attention still
performs full cross-document interactions. This leads not only to unnecessary computation, but also
to a poor inductive bias.
1arXiv:2604.20920v1  [cs.LG]  22 Apr 2026

g gDocument 1 Document 2 Document 1 Document 2Figure 1: Comparison of cross-document attention patterns. Left: Existing approaches perform
token-wise cross-attention between documents, introducing noise and distracting attention. Right:
GSA first compresses each document into high-level gist tokens (g), and restricts cross-document
interaction to them. Raw tokens attend locally within each document, resulting in a cleaner, more
structured, and efficient attention mechanism.
Sparse attention offers a promising direction for mitigating this bottleneck. By restricting each query
to attend to only a subset of keys, sparse methods can substantially reduce the cost of attention while
preserving model capabilities. Existing approaches can be broadly categorized along two axes:when
sparsity is introduced (inference-only versus training-time) andhowrelevant tokens are identified
(fixed patterns versus learned selection).
On the inference side, methods such as H 2O (Zhang et al., 2023b), StreamingLLM (Xiao et al.,
2023), and Quest (Tang et al., 2024) apply sparsity during decoding by evicting or selecting KV-
cache entries. While effective at reducing memory footprint, these approaches operate on models
pretrained with full attention, limiting the model’s ability to learn optimal sparse patterns from
scratch. On the training side, recent methods including Native Sparse Attention (NSA; Yuan
et al., 2025), Deepseek Sparse Attention (DSA; Liu et al., 2025) and MoBA (Lu et al., 2025) have
demonstrated that incorporating sparsity during pretraining can preserve or even improve model
quality while achieving substantial speedups. NSA, for instance, modifies the model architecture
by replacing standard attention with three parallel branches—coarse-grained compressed tokens,
fine-grained selectively retained tokens, and a sliding window for local context. DSA introduces an
external lightweight indexer that operates outside the Transformer to score and select relevant tokens
before sparse attention is performed. MoBA, on the other hand, applies the Mixture-of-Experts
(MoE) principle to attention, using mean-pooled block representations as routing signals to assign
each query to the most relevant KV blocks during the prefill stage. Despite their effectiveness, these
approaches each introduce additional constraints: NSA requires architectural modifications to the
attention mechanism, DSA relies on an external module that is not trained end-to-end with the
main network, and MoBA depends on non-differentiable mean pooling for routing, precluding
gradient-based optimization of the selection process.
A parallel line of work has explored context compression throughgist tokens. In these approaches,
models are trained to encode long contexts into a small set of summary tokens, whose KV-cache
can replace the original raw tokens during subsequent decoding. Two principal strategies have
been studied for positioning gist tokens. The first, adopted by Gist (Mu et al., 2023), AutoCompres-
sors (Chevalier et al., 2023), and KVLink (Yang et al., 2025), append all gist tokens to the end of the
sequence—each gist token can attend to the entire compressed context, providing high flexibility
2

but requiring complex coordination across tokens. The second, used by ActivationBeacon (Zhang
et al., 2024), GistPool (Petrov et al., 2025), and UniGist (Deng et al., 2025), interleaves gist tokens
throughout the sequence, inserting one gist token at regular intervals (e.g., every 16 tokens under a
16× compression ratio). In this case, each gist token is associated with a specific sub-region of the
context, resulting in a more structured and simpler learning problem at the cost of reduced flexibility.
In this paper, we begin with a key observation: interleaved gist tokens, as information-dense
representations of local chunks, can naturally serve as pivots for selecting relevant sub-contexts
under sparse attention. This is conceptually analogous to the compressed tokens in NSA’s coarse-
grained branch or the mean-pooling gates in MoBA, yet gist tokens arelearnedrepresentations rather
than simple statistical summaries. Specifically, after compressing the context into interleaved gist
tokens, we perform top- kselection over these tokens based on their attention scores with respect to
the current query, and then selectivelyunfoldthe corresponding chunks by reintroducing their raw
tokens into the context. Under the same token budget, this selective unfolding strategy significantly
improves accuracy compared to standard compression baselines, including both append-only and
interleaved gist methods, as well as inference-only sparse attention techniques.
Motivated by this observation, we incorporate the selective unfolding mechanism into the training
process, yielding a unified framework that bridges gist-based compression and training-time sparse
attention. We refer to our method as Gist Sparse Attention (GSA). While GSA is related to prior
training-time sparse attention methods, it differs from each in important ways. Compared to NSA,
our method does not require modifications to the model architecture, operating entirely within the
standard Transformer framework. Compared to DSA, it is end-to-end trainable without relying on
an external indexer. Compared to MoBA, it replaces non-differentiable mean-pooling gates with
learnable gist tokens, enabling gradient-based optimization of the selection process.
Furthermore, GSA naturally extends to achievelog-linearcomplexity through recursive construction
of gist-of-gist representations. By hierarchically compressing gist tokens into higher-level summaries
(meta-gist), we obtain a multi-resolution representation of the context. The selective unfolding
mechanism can then be applied in a coarse-to-fine manner: starting from the highest level of
abstraction, the model progressively identifies the most relevant sub-contexts, unfolding from
coarse summaries down to the original raw tokens. This hierarchical design captures information at
multiple levels of granularity while maintaining computational efficiency that scales log-linearly
with context length.
Specifically, a key advantage of GSA in multi-document settings is that it eliminates the need for
cross-attention over raw tokens between documents. In standard full-attention, each query must
attend across all raw tokens from all documents, which is both computationally expensive and
susceptible to distraction from irrelevant passages. GSA replaces this with cross-attention over
gist tokens: tokens attend only to gist and meta-gist tokens from preceding documents, using
these high-density summaries as surrogates for full cross-document attention. This is both more
efficient—the number of gist tokens is a small fraction of the raw token count—and more effective,
as the compressed representations filter out noise and surface the most salient information. Selective
unfolding then recovers fine-grained detail only for the segments identified as relevant.
We evaluate our approach through comprehensive experiments on five RAG benchmarks and
LongBench. Across both settings, our method consistently delivers substantial gains over prior
compression and sparse attention baselines, often by large margins under the same token budget—
for example, improving average RAG performance by over 11 points under continued pretraining,
and by 8-12 points after finetuning.
3

g1 g2 g3qt
Query T oken Sink T oken Raw T oken Gist T oken0.2 0.7 0.1g1 g2 g3
g1 g2 g3
g1 g3g1 g2 g3
g1 g3
g2qtg2
g2GSA NSA
DSA
MoBA
qt
Mean Pooling Mean Pooling Mean PoolingCompressCompressed Attention Selected Attention Sliding Attention
Selection Slidingqt
Top-k Selector
ChunkLightning
Indexerqt
qt
SelectedSparse Attention
Sparse Attention
Sparse
AttentionGist
Encoding
Gist
Scoring
Top-k
Selection
Selective
Unfolding
g1 g2G1 g3 G2 g4H-GSAx1x2 x3x4 x5 g1 g2 g3
x1
x2
g1
x3
x4
g2
x5
g32(a)
2(b) 2(c)Figure 2:(a)Comparison of trainable sparse attention methods (left) and GSA pipeline (right). Left:
NSA replaces standard attention with three parallel architectural branches (compress, select, sliding
window); DSA relies on an external lightning indexer to select tokens before sparse attention; MoBA
uses non-differentiable mean pooling over blocks as routing gates. Right: GSA operates in four
stages within the standard Transformer: (1) interleaved gist tokens are trained to compress each
chunk via a causal mask; (2) the query scores all gist tokens and selects the top- kmost relevant
(starred); (3) the selected chunk is unfolded by restoring its raw tokens alongside its gist token
(red dashed box); (4) sparse attention is computed over the selected gist-chunk pairs only. Unlike
NSA, DSA, and MoBA, GSA requires no architectural modifications, external modules, or non-
differentiable operations.(b)The gist causal mask Mgistfor a sequence with chunk size L= 2 .
Rows represent queries; columns represent keys. Dark cells indicate allowed attention.(c)Attention
patterns with a two-level gist hierarchy. The grayed-out region represents compressed history
accessible only via the meta-gist. Raw and gist tokens (light/dark blue) attend to local context and
preceding meta-gists; and the meta-gist (red) attends to its child gists and the preceding meta-gists.
4

Our main contributions are summarized as follows:
•We observe that interleaved gist tokens serve as effective pivots for context selection, and
propose a selective unfolding mechanism that significantly outperforms standard compression
and sparse attention baselines under the same token budget.
•We integrate selective unfolding into training, yielding an end-to-end framework that bridges
gist-based compression with training-time sparse attention, without architectural modifications,
external indexers, or non-differentiable gates.
•We extend GSA to hierarchical gist-of-gist compression, achieving logarithmic per-step decod-
ing complexity through coarse-to-fine selective unfolding over multi-resolution representations.
2 Problem Setup
We consider autoregressive language modeling, where a model predicts the next token xtgiven all
preceding tokensx <t= (x 1, . . . , x t−1). The prediction is computed as:
P(xt|x<t) = Softmax 
Wvocab·ht
,(1)
where Wvocab∈RV×dis the output projection matrix and htis the hidden state at position t. In
the standard Transformer, htis computed via full attention over all preceding key-value pairs
(K<t,V<t):
ht= Attention(q t,K<t,V<t) = softmaxqtK⊤
<t√
d
V<t.(2)
As the context length tgrows, the cost of computing htand the memory required to store (K<t,V<t)
both scale linearly per step and quadratically over the full sequence, making long-context modeling
computationally prohibitive.
Goal.Our goal is to construct, at each decoding step t, a compact context Vtwith|Vt| ≪t key-value
pairs, such that the model’s prediction quality is preserved:
ht≈Attention(q t,KVt,VVt).(3)
We seek a method that satisfies three desiderata. First, the compact context should bequery-adaptive:
different queries may require different subsets of the history, and the selection should be dynamic
rather than fixed. Second, the selection mechanism should beend-to-end trainable, so that the model
can learn to identify relevant context during pretraining or finetuning, rather than relying on
heuristic or non-differentiable operations applied post hoc. Third, the approach should requireno
architectural modificationsto the standard Transformer, enabling straightforward integration with
existing pretrained models and infrastructures.
3 Method
We present GSA, a framework that bridges learnable gist-based context compression with sparse
attention through a selective unfolding mechanism. The key idea is that interleaved gist tokens,
trained to summarize local chunks, can simultaneously serve aslearned routing signalsfor identifying
thekmost relevant sub-contexts. When a chunk is deemed relevant, its raw tokens are selectively
reintroduced into the attention context, thereby recovering fine-grained detail precisely where
needed. This results in a bounded context size of |Vt|=k·(1 +L) , where Lis the chunk length and
is independent of the full sequence length. The framework further supports two complementary
5

stages: continued pretraining alone already enables effective selective unfolding at inference time,
while optional selective finetuning further improves performance, making GSA both effective and
flexible in practice.
Our framework is inspired by how humans process long contexts. When reading a novel or navigat-
ing a large codebase, readers do not retain every detail uniformly. Instead, they compress earlier
content into high-level summaries—the gist—which is usually sufficient for ongoing comprehen-
sion. Only when a specific detail becomes necessary (e.g., a character’s exact words or a function’s
signature) do they selectively revisit the relevant passage. GSA instantiates this strategy: gist tokens
maintain compact, continuously updated summaries of prior context, while selective unfolding
enables targeted recovery of fine-grained information precisely when needed.
We first review the sparse attention formulation and the interleaved gist token setup (§3.1). We
then describe the core selective unfolding mechanism at a single level of compression (§3.2–§3.3),
followed by the training procedure (§3.4). Finally, we show how the framework naturally extends to
a hierarchical gist-of-gist structure for log-linear complexity (§3.5).
3.1 Preliminaries
Sparse Attention.In standard attention (Eq. equation 2), each query qt∈Rdattends to all
preceding key-value pairs. As the sequence length grows, this scales quadratically in t. Sparse
attention methods reduce this cost by restricting each query to attend to a subset St⊂ {1, . . . , t} of
positions:
ot=X
i∈Stαt,ivi, α t,i=exp(q⊤
tki/√
d)P
j∈Stexp(q⊤
tkj/√
d),(4)
where |St| ≪t . The core challenge lies in how to construct St: fixed patterns such as sliding windows
or stride-based selection impose structural priors that may miss relevant tokens, while dynamic
methods that adapt Stto each query require an efficient mechanism for estimating token relevance
without computing full attention.
Interleaved Gist Tokens.Following prior work on gist-based context compression (Deng et al.,
2025, Mu et al., 2023, Petrov et al., 2025, Zhang et al., 2024), we partition the input sequence
X= (x 1, . . . , x n)intoM=⌈n/L⌉ non-overlapping chunks of length L:C={C 1, . . . , C M}, where
Cm= (x (m−1)L+1 , . . . , x mL). A learnable gist token gmis appended to each chunk, yielding the
augmented sequence:
˜X= (C 1, g1, C2, . . . , C M, gM).(5)
As illustrated in Figure 2(b), a causal mask Mgistis applied such that tokens within chunk Cmcan
attend to tokens within Cmand to all preceding gist tokens g<m, while tokens following gmcannot
attend to raw tokens within Cm’s scope—only to gmand the preceding gist tokens g<m. This creates
an information bottleneck that forces each gist token to learn a compressed representation of its
corresponding chunk. After a forward pass, the embedding gmsummarizes Cm, and its associated
KV pair(k gm,vgm)can then replace the full KV-cache ofC mduring decoding.
3.2 Gist-Based Relevance Scoring
The central observation of this work is that interleaved gist tokens, beyond their role as compressed
summaries, can naturally serve asselection proxiesfor their associated chunks. Because each gist
token gmencodes the semantic content of chunk Cm, the attention affinity between a query and a
gist token provides a direct, learned estimate of the chunk’s relevance.
6

Concretely, at each decoding step t, given the current query qt, we compute a relevance score for
each chunkC mvia the dot product betweenq tand the gist keyk gm:
st,m=q⊤
tkgm.(6)
This scoring mechanism is conceptually related to the gating function in MoBA (Lu et al., 2025),
which computes the affinity between a query and themean-pooledkeys of each block: si=q⊤K[Ii].
However, there is a crucial difference. Mean pooling is a fixed, non-parametric operation: the block
representation is simply the arithmetic average of the raw keys, which may wash out semantically
important signals and has no learnable parameters. In contrast, gist tokens arelearnedrepresentations
optimized through training. This enables two advantages: (i) the gist token can capture rich semantic
content beyond what a simple average reflects, and (ii) gradients flow through the gist tokens,
allowing the model to jointly optimize thequality of compressionand theaccuracy of selection.
This scoring is also related to NSA’s compressed attention branch (Yuan et al., 2025), which generates
coarse-grained summary tokens via a dedicated MLP and uses the resulting attention scores to
identify important fine-grained chunks. Our approach achieves a similar effect without requiring
additional architectural components: the gist tokens are a part of the standard input sequence and
reuse the existing attention mechanism.
3.3 Selective Unfolding and Hybrid Attention
Given the relevance scores, we select the top-kchunks with the highest scores:
It= Top-k 
{st,m}M
m=1
.(7)
For selected chunks m∈ I t, weunfoldthe compressed representation by reintroducing both the
gist KV pair (kgm,vgm)and the full raw KV pairs (KCm,VCm)into the attention context. Unselected
chunks areentirely excludedfrom the decoding context. The hybrid key and value matrices are thus
constructed as:
Khybrid =[
m∈I t 
kgm∪K Cm
,V hybrid =[
m∈I t 
vgm∪VCm
.(8)
The output is then computed via standard attention:
ht= Attention(q t,K hybrid,V hybrid ).(9)
It is worth noting that we skip selective unfolding in the first attention layer, as all gist tokens share
identical input hidden embeddings at this stage, rendering top-kselection uninformative.
We experimented with several variants of the hybrid context, including (i) retaining gist tokens for
all chunks (both selected and unselected) as compact surrogates, and (ii) using only the raw tokens
from selected chunks without any gist tokens in the context. Empirically, we found that attending
exclusively to the selected gist-chunk pairs yields the best performance (see §4.5). We attribute this
to the fact that, under a fixed token budget, replacing unselected gist tokens with unfolded raw
tokens from selected gists provides a more informative attention context. The gist tokens of selected
chunks are still retained alongside their raw tokens, as they provide complementary compressed
information that captures cross-token patterns within the chunk.
During the experiments, rather than using a fixed Top- kacross all settings, we set the kadaptively
based on the sequence length, compression ratio, and GQA group structure, ensuring a consistent
compression ratio across different configurations. Details of the adaptive top- kformula are provided
in Appendix §A.
7

Grouped Unfolding for GQA.In modern architectures employing grouped-query attention
(GQA) (Ainslie et al., 2023), multiple query heads share the same key-value cache. Since different
heads within the same KV group may prefer different chunks, we perform top- kselection separately
for each query head and then take the union of the selected chunk indices within the group. Let H(u)
denote the set of query heads associated with KV group u. For each head h∈ H(u) , we compute
I(h)
t= Top-k 
{s(h)
t,m}M
m=1
and define the group-level unfolded set as:I(u)
t=S
h∈H(u) I(h)
t.
All chunks in I(u)
tare then unfolded into the shared KV-cache for group u. This design preserves
head-specific routing while maintaining consistency with the GQA sharing pattern, ensuring that the
unfolded KV-cache can be loaded once and reused across all heads in the group. This is analogous to
NSA’s group-centric kernel design (Yuan et al., 2025), which loads shared KV blocks once per GQA
group to maximize memory reuse and data locality. Additionally, by batching queries within each
group and reusing KV blocks, it increases arithmetic intensity and enables more efficient utilization
of Tensor Core matrix operations.
3.4 Training
GSA is trained using standard teacher forcing with the autoregressive language modeling objective.
The training consists of two stages: continued pretraining, which is required, and selective finetuning,
which is optional. Both stages share the same sequence structure: each training sequence is split into
two contiguous regions—acompressed context(the prefix) and ageneration context(the suffix). The
loss is computed only over the suffix. Both stages are implemented through attention mask design,
with no separate prefill or decoding phases.
Stage 1: Continued Pretraining (Required).In the continued pretraining stage, the model learns
to compress context into gist tokens. The compressed context (the prefix) is augmented with
interleaved gist tokens and processed under the standard gist causal mask Mgistdescribed in §3.1.
The generation context tokens (the suffix) attend to the gist tokens produced by the compressed
context, and to each other under the standard causal mask. The next token prediction loss over
the generation context encourages the gist tokens to encode sufficient information for accurate
next-token prediction. Since the entire forward pass is a single teacher-forced computation with a
designed causal mask, training is fully parallelizable and requires no specialized kernels.
Importantly, after this stage alone, the model already supports selective unfolding at inference time.
Because the gist tokens have learned to encode meaningful chunk-level summaries, they can be used
directly as routing signals via Eq. equation 6–equation 7 without any further training. As shown
in §4, this inference-time selective unfolding already yields substantial gains over standard gist
compression (both appended and interleaved) as well as other inference-time-only sparse attention
approaches.
Stage 2: Selective Finetuning (Optional).While continued pretraining alone enables effective
inference-time selection, exposing the model to the selective unfolding mechanism during training
can further improve performance. In the finetuning stage, the sequence is again split into compressed
and generation contexts, but the attention mask for the generation context is now modified to
incorporate top- kselection. Concretely:Compressed context:Processed identically to continued
pretraining, using the standard interleaved gist mask Mgist. This region produces the gist token
representations.Generation context:For each query position tin the generation context, we
compute relevance scores against all gist tokens via Eq. equation 6 and select the top- kchunks via
8

Eq. equation 7. The attention mask for position tis then set to true only for the gist tokens and raw
tokens of the selected chunks; all other compressed-context tokens are masked out.
Unlike continued pretraining, which employs a fixed, static attention mask, selective finetuning
requires a position-dependent sparse mask where each query attends to a different subset of
keys. Implementing this efficiently may require customized CUDA kernels. However, the sparse
blockwise attention patterns involved are structurally similar to those in NSA (Yuan et al., 2025) and
MoBA (Lu et al., 2025), and can leverage the same kernel design principles. We emphasize that this
stage is optional: practitioners who wish to avoid the additional implementation effort can skip
selective finetuning entirely and still benefit from inference-time selective unfolding after continued
pretraining alone.
3.5 Extension to Hierarchical Gist-of-Gist Compression
The single-level framework described above achieves linear per-step complexity of O(M+kL)
where M=⌈n/L⌉ . We now show that the same principle—compressing representations and using
them as routing signals—can be applied recursively, yielding a hierarchical structure with log-linear
complexity. We refer to this extension as H-GSA.
The motivation is again grounded in how humans organize and retrieve information. Long docu-
ments are rarely flat sequences—they are structured hierarchically: a book is divided into chapters,
chapters into paragraphs. When searching for a specific detail, a reader does not scan every sentence
from the beginning. Instead, retrieval proceeds hierarchically: one first recalls a high-level memory
(e.g., the chapter or topic), then refines it to a more specific section, and finally reconstructs the
precise detail. Each level acts as a progressively sharper filter over memory. H-GSA mirrors this
structure: meta-gist tokens encode coarse summaries, gist tokens represent finer summaries, and
raw tokens preserve exact details. Hierarchical selection then implements this coarse-to-fine retrieval
process, progressively narrowing attention from abstract memory to precise token-level information.
We define the meta-gist tokens as follows:
Meta-Gist Tokens.Just as gist tokens summarize chunks of raw tokens,meta-gisttokens can
be defined to summarize groups of gist tokens. We introduce a meta-gist token Gkafter every J
consecutive gist-chunk pairs. The meta-gist Gksummarizes the segment Sk={(C j, gj)}kJ
j=(k−1)J+1,
yielding⌈M/J⌉meta-gist tokens in total. The augmented sequence becomes:
˜X= 
C1, g1, . . . , C J, gJ| {z }
Segment1, G1, . . . , C M, gM, G⌈M/J⌉
.(10)
Figure 2(c) illustrates the hierarchical attention pattern for a two-level gist hierarchy: tokens fol-
lowing a meta-gist Gkcannot attend to the raw tokens or gist tokens within the scope of Gk; they
can only attend to Gkitself. This blocking mechanism forces long-range history to be accessed
exclusively through the meta-gist representations. The construction generalizes to arbitrary depth
tby introducing additional levels of summary tokens, each summarizing Jgroups from the level
below. This means GSA can theoretically support unbounded context lengths: as the sequence
grows, additional hierarchy levels are added to maintain a fixed memory footprint at the top level,
with the full detail always recoverable through selective unfolding. Throughout this paper, we use
the termsummary tokensto refer collectively to gist tokens and meta-gist tokens at all levels of the
hierarchy, distinguishing them from the raw tokens of the original sequence.
Coarse-to-Fine Selection.The hierarchical structure enables a coarse-to-fine selection strategy that
mirrors the single-level mechanism at each scale. Instead of computing relevance scores against all
9

Mgist tokens, the model first scores the ⌈M/J⌉ meta-gist tokens, selects the top- k2segments, and
then scores only the k2·Jgist tokens within those segments to identify the final top- k1chunks for
unfolding. This reduces the per-query routing cost from O(M) toO(M/J+k 2·J). For a hierarchy
of depth t, the per-query routing cost generalizes to O(t·M1/t). When t=O(logJM), this reduces
toO(J·logJM) =O(logM) —that is, each query’s selection cost islogarithmicin the context length.
After selection, the query attends to the selected tokens ateverylevel of the hierarchy: the ktselected
top-level summary tokens, the kt−1selected tokens at the next level, down to the k1selected gist
tokens and their k1·Lunfolded raw tokens. The total attention context thus containsPt
ℓ=1kℓ+k1·L
tokens. Since each kℓis a small constant and t=O(logJM), the summary tokens contribute O(logM)
and the raw tokens contribute O(k 1L). The per-query attention cost is therefore O(k 1L+ logM) ,
which matches the routing cost in order. The total per-query decoding cost is O(logM) =O(logn) ,
logarithmic in the context length. Over the full sequence, the aggregate decoding cost is O(nlogn) ,
which is log-linear.
Table 1 summarizes the complexity comparison. We provide a more detailed analysis in Appendix §B.
Table 1: Complexity comparison. n: context length, L: chunk size, k: number of selected chunks, J:
meta-gist grouping factor. All quantities treat d(head dimension) as a constant. For the hierarchical
variant, the depth is set tot=⌈logJM⌉whereM=n/L.
Method Per-step Decoding Prefill
Full AttentionO(n)O(n2)
GSA (single-level)O(n/L+kL)O(nL+n2/L2)
GSA (hierarchical)O(logn+kL)O(nL+nJ/L)
4 Experiments
4.1 Experiment Setup
We evaluate GSA on two model families of different scales: Qwen2-7B-Instruct (Yang et al., 2024a)1
and Llama3.2-1B (Meta AI, 2024)2. This allows us to assess the generality of our approach across
both a large instruction-tuned model and a smaller base model.
Following the two-stage procedure described in §3.4, we first perform continued pretraining with
interleaved gist tokens to learn context compression, then optionally apply selective finetuning
to incorporate the unfolding mechanism into training. Following ActivationBeacon (Zhang et al.,
2024), we use data sampled from RedPajama (Weber et al., 2024) for continued pretraining of
Qwen2-7B-Instruct, with sequence lengths ranging from 7K to 20K. Following KVLink (Yang et al.,
2025), we use data sampled from FineWeb (Penedo et al., 2024) for pretraining of Llama-3.2-1B,
with sequence lengths ranging from 2K to 4K. For finetuning data, we use the dataset provided
by ActivationBeacon (Zhang et al., 2024) for Qwen2-7B-Instruct, which contains samples from
LongAlpaca (Chen et al., 2023), BookSum (Kry´ sci ´ nski et al., 2022), and a synthesized QA dataset. For
Llama-3.2-1B, we use the dataset provided by KVLink (Yang et al., 2025), which contains samples
from the training sets of TriviaQA (Joshi et al., 2017) and 2WikiMQA (Ho et al., 2020), with answers
generated by DeepSeek-V3.2 (Liu et al., 2025). All training is conducted on 8 NVIDIA H100 GPUs.
Table 2 summarizes the hyperparameters for each model and training stage.
1We choose Qwen2 because ActivationBeacon—an important baseline—has been evaluated on it.
2We choose Llama3.2-1B because KVLink—an important baseline—has been evaluated on it.
10

Table 2: Training hyperparameters for each model and stage.
Model Stage Batch Size Learning Rate Weight Decay LR Schedule #Epoch
Qwen2-7B-Inst.Pretraining 8 1e-5✓Cosine w/ warmup 1
Finetuning 8 5e-6✓Cosine w/ warmup 1
Llama3.2-1BPretraining 128 1e-5✓Cosine w/ warmup 1
Finetuning 64 5e-6✓Cosine w/ warmup 2
We compare against three categories of methods:(1) Inference-time sparse attention:LongLLMLin-
gua (Jiang et al., 2024), H 2O (Zhang et al., 2023b), StreamingLLM (Xiao et al., 2023), and Quest (Tang
et al., 2024). These methods apply KV-cache eviction or selection to models pretrained with full
attention, without any additional training.(2) Gist-based compression:ActivationBeacon (Zhang
et al., 2024) (interleaved gist), UniGist (Deng et al., 2025) (interleaved gist), and KVLink (Yang
et al., 2025) (appended gist). These methods require continued pretraining and/or finetuning.(3)
Full attention references:We include the original pretrained model (no compression), continued
pretraining with full attention (Full-PT), and finetuning with full attention (Full-FT) as the references.
We do not include comparisons with NSA, DSA, or MoBA, as these approaches are not directly
compatible with our setting: NSA requires architectural modifications, DSA depends on an external
indexer, and MoBA is limited to prefill-only scenarios (Lu et al., 2025). Consequently, they are not
applicable to the continued pretraining and sparse decoding framework we consider.
We evaluate our method under two primary settings: (1)Long-context compressionon Long-
Bench (Bai et al., 2024); and (2)Retrieval-Augmented Generation (RAG)on five multi-document
QA benchmarks, along with aKV-cache reusetask. Following NSA (Yuan et al., 2025), we exclude a
subset of LongBench tasks that consistently yield low scores across models, as they offer limited
discriminatory power.
The key distinction between these settings lies in how context is structured. The long-context setting
treats each sample as a single continuous sequence. In contrast, in the RAG setting, each sample
consists of multiple documents with explicit boundaries, and gist tokens are inserted independently
within each document.
Beyond these evaluations, we conduct extensive ablation studies to analyze the design of the hybrid
context, including comparisons between adaptive top- kselection and top- pthresholding. We further
assess GSA’s fundamental retrieval capability through a passkey retrieval task across varying context
lengths (Appendix §C).
4.2 Long-context Results
Table 3 reports results on LongBench using Qwen2-7B-Instruct across three compression ratios ( 8×,
16×,32×), under both continued pretraining and finetuning. We omit H-GSA at 8×compression, as
the chunk size is already small. For H-GSA at 16×, we set L= 4 andJ= 4 —inserting a gist token
every 4 tokens and a meta-gist token every 4 gist tokens. At 32×, we use L= 8 andJ= 4 —inserting
a gist token every 8 tokens.
GSA consistently outperforms gist baselines.Under continued pretraining, GSA achieves an
average of 46.20 across tasks at 8×compression, outperforming ActivationBeacon (42.52) and
UniGist (43.40) by 3.7 and 2.8 points respectively, and nearly matching the Full-PT model (47.78).
The advantage is maintained across higher compression ratios: at 16×, GSA scores 45.39 versus
40.64 (ActivationBeacon) and 40.89 (UniGist); at 32×, GSA achieves an average of 44.07 versus 38.30
11

and 38.16. This demonstrates that selective unfolding recovers critical fine-grained information that
standard compression discards uniformly.
Selective finetuning further improves performance.With finetuning, GSA continues to lead
across all settings. At 8×, it achieves 46.74 (vs. 42.67 for ActivationBeacon and 43.33 for UniGist),
approaching the Full-FT model (47.32). Notably, on several individual tasks GSA surpasses the
full-attention baseline: for instance, on MF-en task at 8×finetuning, GSA scores 54.24 compared
to Full-FT model’s 50.33, suggesting that the selective unfolding mechanism can act as a beneficial
inductive bias.
Hierarchical variant excels at high compression.H-GSA demonstrates its advantage at higher
compression ratios where the single-level variant’s routing cost becomes more significant. At
16× with finetuning, H-GSA achieves 46.48, surpassing GSA’s 45.26. At 32× with finetuning, the
gap widens: H-GSA scores 44.94 versus GSA’s 43.35. This validates that hierarchical gist-of-gist
compression and coarse-to-fine selection become more valuable as the compression ratio grows.
4.3 RAG Results
Table 4 reports results on five multi-document QA benchmarks using Llama-3.2-1B across two
compression ratios (8×,16×), under both continued pretraining and finetuning.
The RAG setting reveals the most striking advantage of GSA. At 8×compression with continued
pretraining only, GSA achieves 33.68 average score, not only outperforms KVLink (21.58) and
UniGist (22.53) by over 11 points, buteven surpasses the vanilla (27.99) and Full-PT model (27.14).
We attribute this to the nature of RAG: each query is relevant to only one of the retrieved documents,
while the remaining documents act as distractors. Full attention distributes capacity uniformly
across all documents, diluting focus on the relevant passage. In contrast, GSA’s selective unfolding
mechanism concentrates attention on the most query-relevant chunks, effectively filtering out
distracting documents.
After finetuning, GSA at 8×achieves 53.39, approaching Full-FT (57.76). This represents an improve-
ment of 11.94 points over KVLink (41.45) and 8.12 points over UniGist (45.27). At 16×, GSA scores
49.29, outperforming KVLink’s 38.67 and UniGist’s 42.24.
Consistent with the long-context results, H-GSA outperforms the single-level variant at 16× com-
pression. Under continued pretraining, H-GSA achieves 30.19, compared to 27.74 for GSA. After
finetuning, H-GSA reaches 50.72 versus 49.29 for GSA, suggesting that the coarse-to-fine hierarchical
selection more effectively allocates the limited token budget.
A notable pattern in the RAG results is that Full-PT underperforms GSA under continued pretraining,
yet surpasses it after finetuning. This is because continued pretraining optimizes a generic next-
token prediction objective over the given context, without any notion of a query or a target answer.
The model has no signal to distinguish relevant from irrelevant documents, and thus learns to attend
to all content indiscriminately. Finetuning, by contrast, trains the model on the downstream task
format—given multiple documents and a question, produce an answer—which explicitly teaches it
to focus on query-relevant passages and suppress distractors.
4.4 Parallel Computing with KV-cache Reuse
KV-cache reuse—where KV caches of long documents are stored and reused across repeated queries
to avoid redundant prefilling and support parallel computing—is a natural fit for GSA. In this
setting, GSA encodes documents independently with interleaved gist tokens and reuses their KV
caches at query time.
12

Table 3: Performance comparison on LongBench under different compression ratios ( 8×,16×,32×)
using Qwen2-7B-Instruct. Best results are in bold, second-best are underlined.
MethodsSingle-Document QA Multi-Document QA Few-shot SummarizationAvg.
NrtvQA Qasper MF-en HotpotQA 2WikiMQA Musique SAMSum TriviaQA GovRep
–Qwen2-7B-Inst. 26.33 42.30 44.50 54.39 55.41 36.19 45.19 89.91 36.16 47.82
Inference-time Only
LongLLML. 15.10 20.21 22.16 23.74 28.81 9.94 39.77 85.98 20.51 29.58
H2O 23.17 37.99 40.48 50.66 50.60 32.05 41.33 88.49 32.99 44.20
StreamLLM 22.95 29.16 25.31 44.78 44.56 27.37 42.84 88.99 23.30 38.81x8
QUEST 11.56 13.35 24.13 10.89 15.06 11.06 19.34 36.91 13.56 17.32
Continued Pretraining
– Full-PT 26.59 38.37 46.39 57.56 56.28 36.12 45.83 89.06 33.78 47.78
ActBeacon 18.23 35.40 31.69 48.87 54.45 30.89 45.30 87.59 30.24 42.52
UniGist 22.88 32.92 36.52 55.14 51.10 31.73 43.02 89.88 27.45 43.40
GSA 23.55 36.84 42.21 61.02 54.23 32.47 44.59 90.56 30.36 46.20x8
H-GSA – – – – – – – – – –
ActBeacon 19.08 32.45 28.57 46.47 47.93 31.67 45.50 85.02 29.06 40.64
UniGist 21.68 28.90 35.57 51.97 46.95 27.55 43.56 89.01 22.80 40.89
GSA 24.30 33.86 43.76 55.70 51.46 34.17 44.32 91.02 29.94 45.39x16
H-GSA 21.89 31.11 37.46 56.01 50.98 30.25 44.63 89.41 31.05 43.64
ActBeacon 19.90 27.92 24.71 43.81 46.95 28.13 44.70 83.16 25.42 38.30
UniGist 18.68 28.89 26.19 48.67 43.21 25.73 43.43 88.49 20.17 38.16
GSA 22.61 34.03 37.08 54.19 52.03 30.53 44.42 90.40 31.31 44.07x32
H-GSA 16.16 30.82 39.30 53.66 49.08 30.62 44.35 89.17 29.89 42.56
Finetuning
– Full-FT 27.84 38.56 50.33 52.26 54.87 35.25 43.70 87.99 35.07 47.32
ActBeacon 26.85 36.95 47.93 48.91 41.27 22.54 44.61 87.32 27.62 42.67
UniGist 24.17 38.27 47.28 54.02 41.26 24.84 43.63 89.92 26.59 43.33
GSA 26.24 43.54 54.24 54.26 48.18 28.95 44.31 90.11 30.79 46.74x8
H-GSA – – – – – – – – – –
ActBeacon 23.81 35.77 38.91 44.29 41.69 22.76 46.39 86.06 27.80 40.83
UniGist 23.15 34.91 40.54 46.79 38.67 22.82 43.82 89.97 23.15 40.42
GSA 24.38 42.47 51.98 54.69 42.91 28.59 42.66 88.43 31.27 45.26x16
H-GSA 22.72 43.98 53.40 56.11 48.31 29.04 43.28 90.36 31.12 46.48
ActBeacon 21.32 32.52 33.18 41.05 42.44 19.10 43.04 85.66 28.33 38.52
UniGist 19.96 31.98 33.87 40.34 37.63 20.47 43.43 89.56 21.32 37.62
GSA 24.58 40.24 45.68 50.64 39.38 27.55 42.69 89.10 30.30 43.35x32
H-GSA 22.03 43.32 50.79 52.47 45.18 27.93 42.56 89.47 30.68 44.94
Following the KVLink (Yang et al., 2025) protocol, we insert link tokens between documents. These
Link tokens attend only to gist and meta-gist tokens from preceding documents, enabling efficient
cross-document routing, while selective unfolding recovers full-resolution detail only for the most
relevant segments. This avoids traversing full raw KV caches and improves efficiency without
sacrificing quality. In Table 5, GSA achieves an average RAG performance of 48.07 at 8×compression
and 41.59 at 16×, substantially outperforming KVLink and UniGist; H-GSA further improves the
16× compression result to 46.34. These results support gist tokens as effective routing signals for
cross-document reasoning under KV-cache reuse.
13

Table 4: Performance comparison on RAG under different compression ratios ( 8×,16×) using
Llama3.2-1B. Best results are in bold, second-best are underlined.
Methods HotpotQA TriviaQA 2WikiMQA MuSiQue NQ Avg.
– Llama3.2-1b 29.98 40.82 25.86 9.10 34.20 27.99
Continued Pretraining
– Full-PT 29.64 39.73 25.45 9.18 31.72 27.14
KVLink 17.54 44.36 25.03 4.01 16.96 21.58
UniGist 22.77 44.44 22.84 3.72 18.86 22.53
GSA 41.09 52.06 38.65 12.99 23.62 33.68x8
H-GSA – – – – – –
KVLink 10.67 33.10 21.22 0.74 4.92 14.13
UniGist 21.51 41.62 30.10 3.23 14.56 22.20
GSA 30.56 46.28 32.67 8.11 21.10 27.74x16
H-GSA 34.34 50.98 33.61 9.06 22.98 30.19
Finetuning
– Full-FT 63.82 71.38 75.83 24.20 53.56 57.76
KVLink 41.15 61.65 62.68 10.47 31.28 41.45
UniGist 49.25 64.00 64.17 12.37 36.56 45.27
GSA 59.88 68.62 71.99 20.27 46.20 53.39x8
H-GSA – – – – – –
KVLink 36.92 58.85 62.21 8.11 27.24 38.67
UniGist 43.50 61.83 60.72 10.88 34.26 42.24
GSA 54.29 66.16 67.93 16.30 41.78 49.29x16
H-GSA 56.43 66.76 69.56 18.37 42.48 50.72
Table 5: RAG benchmarks with KV-cache Reuse under different compression ratios ( 8×,16×) using
Llama3.2-1B. Best results are in bold.
Methods HotpotQA TriviaQA 2WikiMQA MuSiQue NQ Avg.
KVLink 41.55 60.00 63.91 9.68 30.86 41.20
UniGist 42.74 59.98 61.51 9.02 28.48 40.35
GSA 54.23 64.81 70.21 13.98 37.10 48.07x8
H-GSA – – – – – –
KVLink 37.11 57.97 61.43 8.03 27.84 38.48
UniGist 37.20 57.65 58.14 7.53 25.42 37.19
GSA 44.58 59.32 61.88 9.68 32.50 41.59x16
H-GSA 50.95 62.77 68.26 12.25 37.46 46.34
14

0.80 0.85 0.90 0.95
top-p45464748495051Accuracy (%)
2wikimqa
Top-p Varaints
Top-k (Adaptive)
0.80 0.85 0.90 0.95
top-p2828292930
gov_report
Top-p Varaints
Top-k (Adaptive)
0.80 0.85 0.90 0.95
top-p394041424344
multifieldqa_en
Top-p Varaints
Top-k (Adaptive)
0.80 0.85 0.90 0.95
top-p21222223232424
narrativeqa
Top-p Varaints
Top-k (Adaptive)
0.80 0.85 0.90 0.95
top-p32323333
qasper
Top-p Varaints
Top-k (Adaptive)
0.80 0.85 0.90 0.95
top-p192020202021
NQ
Top-p Varaints
Top-k (Adaptive)Figure 3: Top- kvs. top- pselection across five LongBench tasks (on Qwen2) and NQ (on Llama3.2).
The red dashed line indicates our adaptive top- kselection; green/blue solid lines show top- pvariants
with p∈ {0.80,0.85,0.90,0.95} . Adaptive top- kconsistently outperforms all top- pthresholds and
exhibits greater stability across tasks.
4.5 Ablation Analysis
Selective Rules Comparison.Table 6 ablates the design of the hybrid attention context (Eq. equa-
tion 8). We compare three variants: (1)SR-only: the attention context includes only the selected
raw tokens, without their corresponding gist tokens; (2)AG+SR: all gist tokens are retained in the
context alongside the selected raw tokens; and (3)SG+SR: only the selected gist tokens are retained
together with their raw tokens (our default). SG+SR achieves the best average score (53.39), out-
performing SR-only (52.76) and slightly surpassing AG+SR (53.27). The result confirms our design
choice in §3.3: retaining selected gist tokens alongside their raw tokens provides complementary
compressed information, while including all gist tokens (AG+SR) dilutes the attention with less
relevant summaries. Dropping gist tokens entirely (SR-only) loses the compressed cross-chunk
patterns that gist tokens capture.
Table 6: Ablation Study on different selective rules. We adopt SG+SR as the default configuration.
Variants HotpotQA TriviaQA 2WikiMQA MuSiQue NQ Avg.
SR-only 58.74 68.33 71.7020.7644.28 52.76
AG+SR 59.53 68.7471.72 20.68 45.70 53.27
SG+SR 59.8868.62 71.9920.2746.20 53.39
Top-kvs. Top- p.Figure 3 compares our adaptive top- kselection (red dashed line) against top-
p(nucleus) selection with varying thresholds p∈ {0.80,0.85,0.90,0.95} (green/blue solid lines)
across six LongBench tasks. Top- pselection dynamically adjusts the number of unfolded chunks by
including chunks until their cumulative score mass reaches p. Across all six tasks, adaptive top- k
consistently outperforms every top- pthreshold, often by a substantial margin. For instance, on
2WikiMQA, top- kachieves over 51% while the best top- pvariant ( p= 0.80 ) reaches only 47.5%. On
GovReport, top- kscores approximately 30 compared to 28–29 for all top- psettings. The performance
of top- pis also notably unstable across tasks: on NarrativeQA it peaks at p= 0.85 but drops sharply
at other thresholds, and on NQ it fluctuates between 19 and 21 with no consistent trend. This
instability arises because the score distribution varies across queries and tasks, making a fixed
probability threshold a poor proxy for the number of relevant chunks. In contrast, adaptive top- k
directly controls the token budget, providing a more reliable and stable selection mechanism.
15

5 Related Works
Sparse Attention Mechanisms.Addressing the quadratic complexity of the self-attention mecha-
nism has been a central theme in scaling Transformers to long contexts. Early approaches introduced
fixed or heuristic sparsity patterns. Sparse Transformer (Child, 2019) and Longformer (Beltagy et al.,
2020) utilize fixed local windows combined with global attention tokens to reduce complexity to
O(N√
N)orO(N) . Many recent KV-cache management works such as H 2O (Zhang et al., 2023b),
StreamingLLM (Xiao et al., 2023), LongLLMLingua (Jiang et al., 2024), and Quest (Tang et al., 2024)
aim to evict less important KV pairs. More recently, hardware-aware and learnable sparse attention
mechanisms—such as Native Sparse Attention (Yuan et al., 2025), DeepSeek Sparse Attention (Liu
et al., 2025), and MoBA (Lu et al., 2025)—have been proposed to dynamically identify and attend to
the most relevant blocks in a hardware-efficient manner.
Context Compression.An alternative approach to efficient long-context processing is compressing
the context into compact representations. Gist Tokens (Mu et al., 2023) and AutoCompressors (Cheva-
lier et al., 2023) train models to compress prompt instructions into special tokens via an attention
mask bottleneck. KVLink (Yang et al., 2025) appends learnable compression tokens to the end of each
document, where each token can attend to the entire preceding context to produce a global summary.
ActivationBeacon (Zhang et al., 2024), UniGist (Deng et al., 2025), and GistPool (Petrov et al., 2025)
interleave gist tokens throughout the sequence, associating each with a local chunk. While these
methods effectively reduce sequence length, they typically treat compression as a one-way process:
once the context is compressed, the raw tokens are discarded and irreversibly lost.
Alternative Sequence Modeling Architectures.A separate line of research seeks to replace the
softmax attention mechanism entirely with architectures that achieve linear or near-linear complexity.
Linear attention methods (Katharopoulos et al., 2020) approximate the softmax kernel with feature
maps, enabling recurrent computation with a fixed-size state. State space models such as S4 (Gu
et al., 2021) and Mamba (Gu and Dao, 2023) model sequences through structured recurrences with
selective gating, achieving linear-time inference with competitive language modeling performance.
Recurrent alternatives including RWKV (Peng et al., 2023) and RetNet (Sun et al., 2023) combine
recurrent architectures with attention-like mechanisms, offering constant-memory inference. More
recent gated linear attention variants such as GLA (Yang et al., 2024b), refine selective forgetting
through per-token gates. These approaches represent a fundamentally different design axis from
our work. They modify or replace the core attention mechanism with alternative sequence modeling
primitives, often requiring pretraining from scratch on the new architecture.
6 Conclusions
We presented GSA, a framework that unifies gist-based context compression with sparse attention
through selective unfolding. By leveraging interleaved gist tokens as both compressed summaries
and learned routing signals, GSA enables query-adaptive recovery of fine-grained detail without
architectural modifications, external indexers, or non-differentiable gates. The framework naturally
extends to hierarchical gist-of-gist compression, achieving logarithmic per-step decoding complexity
through coarse-to-fine selection. Experiments on LongBench and RAG benchmarks demonstrate
consistent gains over both gist compression and sparse attention baselines across compression ratios
from8×to32×.
7 Acknowledgment
This work was supported in part by ONR Grant N00014-22-1-2110, NSF Grant 2205084, and the
Stanford Institute for Human-Centered Artificial Intelligence (HAI). EBF is a Biohub, San Francisco,
Investigator.
16

References
Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages
4895–4901, 2023.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context
understanding. InProceedings of the 62nd annual meeting of the association for computational linguistics
(volume 1: Long papers), pages 3119–3137, 2024.
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer.
arXiv preprint arXiv:2004.05150, 2020.
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora:
Efficient fine-tuning of long-context large language models.arXiv preprint arXiv:2309.12307, 2023.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to
compress contexts. InProceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, pages 3829–3846, 2023.
Rewon Child. Generating long sequences with sparse transformers.arXiv preprint arXiv:1904.10509,
2019.
Chenlong Deng, Zhisong Zhang, Kelong Mao, Shuaiyi Li, Tianqing Fang, Hongming Zhang, Haitao
Mi, Dong Yu, and Zhicheng Dou. Unigist: Towards general and hardware-aligned sequence-level
long context compression.arXiv preprint arXiv:2509.15763, 2025.
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arxiv.
arXiv preprint arXiv:2312.00752, 10, 2023.
Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured
state spaces.arXiv preprint arXiv:2111.00396, 2021.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu
Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning.arXiv preprint arXiv:2501.12948, 2025.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the 28th International
Conference on Computational Linguistics, pages 6609–6625, 2020.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili
Qiu. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt
compression, 2024. URLhttps://arxiv.org/abs/2310.06839.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are
rnns: Fast autoregressive transformers with linear attention. InInternational conference on machine
learning, pages 5156–5165. PMLR, 2020.
17

Wojciech Kry´ sci´ nski, Nazneen Rajani, Divyansh Agarwal, Caiming Xiong, and Dragomir Radev.
Booksum: A collection of datasets for long-form narrative summarization. InFindings of the
association for computational linguistics: EMNLP 2022, pages 6536–6558, 2022.
Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei
Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language
models.arXiv preprint arXiv:2512.02556, 2025.
Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He,
Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms.arXiv
preprint arXiv:2502.13189, 2025.
Meta AI. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. https:
//ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ ,
September 2024. Accessed: 2026-03-29.
Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens.Advances
in Neural Information Processing Systems, 36:19327–19352, 2023.
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S
Bernstein. Generative agents: Interactive simulacra of human behavior. InProceedings of the 36th
annual acm symposium on user interface software and technology, pages 1–22, 2023.
Guilherme Penedo, Hynek Kydlíˇ cek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raf-
fel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest
text data at scale. InThe Thirty-eight Conference on Neural Information Processing Systems Datasets
and Benchmarks Track, 2024. URLhttps://openreview.net/forum?id=n6SCkn2QaG.
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi
Cao, Xin Cheng, Michael Chung, Leon Derczynski, et al. Rwkv: Reinventing rnns for the
transformer era. InFindings of the association for computational linguistics: EMNLP 2023, pages
14048–14077, 2023.
Aleksandar Petrov, Mark Sandler, Andrey Zhmoginov, Nolan Miller, and Max Vladymyrov. Long
context in-context compression by getting to the gist of gisting.arXiv preprint arXiv:2504.08934,
2025.
Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
Furu Wei. Retentive network: A successor to transformer for large language models, 2023.
Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest:
Query-aware sparsity for efficient long-context llm inference.arXiv preprint arXiv:2406.10774, 2024.
Maurice Weber, Daniel Y. Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov,
Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala,
Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang.
Redpajama: an open dataset for training large language models.NeurIPS Datasets and Benchmarks
Track, 2024.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks.arXiv preprint arXiv:2309.17453, 2023.
18

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang,
Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren
Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li,
Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang,
Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan
Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao,
Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and
Zhihao Fan. Qwen2 technical report, 2024a. URLhttps://arxiv.org/abs/2407.10671.
Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. Kvlink: Accelerating large language
models via efficient kv cache reuse.arXiv preprint arXiv:2502.16002, 2025.
Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with
delta rule.arXiv preprint arXiv:2412.06464, 2024b.
Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie,
Yuxing Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and
natively trainable sparse attention. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 23078–23097, 2025.
Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with
reasoning.Advances in Neural Information Processing Systems, 35:15476–15488, 2022.
Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou,
and Weizhu Chen. Repocoder: Repository-level code completion through iterative retrieval and
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 2471–2484, 2023a.
Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context
compression with activation beacon.arXiv preprint arXiv:2401.03462, 2024.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient
generative inference of large language models.Advances in Neural Information Processing Systems,
36:34661–34710, 2023b.
19

Supplementary Material
A More Details on Adaptive Top-k.
Concretely, let nkvdenote the total number of KV positions available for the generation context,
Leffthe effective compression factor (equal to the chunk size Lfor single-level GSA, or L·J for
the hierarchical variant H-GSA), and Gthe number of query heads per KV group. The per-level
selection budget is computed as:
k=nkv
Leff·G·L
+ 1,(A.1)
where the division by Gaccounts for the grouped unfolding mechanism (§3.3): since each head
within a GQA group may select different chunks and their union is unfolded into the shared KV-
cache, the effective number of unfolded chunks per group scales with G. The +1ensures that at least
one chunk is always selected. For the hierarchical variant, the same kis used at both the meta-gist
and gist levels. This adaptive scheme ensures that the total number of unfolded tokens remains
approximately proportional to nkvregardless of the compression ratio, providing a fair comparison
across settings.
B Complexity Analysis
We provide a detailed complexity analysis for both the single-level and hierarchical variants of GSA.
Letndenote the input length, anddthe head dimension (treated as a constant).
B.1 Single-Level Complexity
Given chunk size L, the number of chunks (and thus gist tokens) is M=⌈n/L⌉ , and let kdenote the
number of chunks selected for unfolding.
Prefill.During prefill, the computation decomposes into two parts:
1.Intra-chunk attention.Each chunk CmhasLraw tokens plus one gist token. Under the causal
mask, tokens within Cmattend to at most L+ 1 positions. Across all Mchunks, this costs
O(M·L·(L+ 1)·d) =O(nL), sinceM=⌈n/L⌉anddis constant.
2.Gist-to-gist attention.Each gist token gmcan attend to all preceding gist tokens g1, . . . , g m−1.
The total cost across allMgist tokens isPM
m=1m=O(M2) =O(n2/L2).
The overall prefill cost is therefore O(nL+n2/L2). For typical settings where L=O(√n), both terms
areO(n√n), yielding sub-quadratic prefill. For fixed L, the gist-to-gist term O(n2/L2)dominates,
which is quadratic but reduced by a factor ofL2compared to full attention.
Decoding (per step).At each decoding stept, the computation involves:
1.Relevance scoring.The query qtcomputes dot products with all Mgist keys: st,m=q⊤
tkgm/√
d
form= 1, . . . , M. This costsO(M·d) =O(M).
2.Top-kselection.Selecting the khighest scores from Mcandidates costs O(M) (via a partial
sort).
3.Hybrid attention.The query attends to kgist tokens and k·L raw tokens from the selected
chunks, totaling k(1 +L) key-value pairs. The attention computation costs O(k(1 +L)·d) =
O(kL).
20

The total per-step decoding cost is:
O(M)|{z}
scoring+O(M)|{z}
top-k+O(kL)|{z}
attention=O(M+kL) =O(n/L+kL).(B.2)
Since kandLare fixed constants independent of n, this islinearin the context length n. Over a full
sequence of ndecoding steps, the aggregate decoding cost is O(n2/L+nkL) , which is quadratic in
nbut reduced by a factor ofLcompared to full attention’sO(n2).
B.2 Hierarchical Complexity
We now analyze the hierarchical variant with meta-gist grouping factor Jand hierarchy depth t.
At each level ℓ∈ {1, . . . , t} , there are Mℓ=⌈M/Jℓ−1⌉summary tokens: M1=M gist tokens at the
bottom, M2=⌈M/J⌉ meta-gist tokens at level 2, and so on up to Mt=⌈M/Jt−1⌉tokens at the top.
Prefill.The blocking mechanism ensures that tokens at level ℓcan only attend to summary tokens
at the same or higher levels within their local scope. The computation decomposes as:
1.Intra-chunk attention.Same as the single-level case:O(nL).
2.Level- ℓsummary attention.At level ℓ, each of the Mℓsummary tokens attends to at most J
tokens from level ℓ−1 (its children) plus the preceding summary tokens at level ℓ. Due to the
blocking mechanism, each level- ℓtoken’s effective attention span is bounded by O(J+M ℓ).
However, the blocking ensures that level- ℓtokens only see level- ℓpeers within their parent’s
scope at level ℓ+ 1 , bounding the span to O(J) per token. The total cost at level ℓisO(M ℓ·J).
Summing across all levels:
tX
ℓ=1O(M ℓ·J) =O 
JtX
ℓ=1M
Jℓ−1!
=O
MJ·1−J−t
1−J−1
=O(MJ).(B.3)
The geometric series converges since J≥2 , so the total summary attention cost is O(MJ) =
O(nJ/L), independent of the deptht.
The overall prefill cost is therefore O(nL+nJ/L) , which islinearin n—a substantial improvement
over the single-level prefill cost ofO(nL+n2/L2).
Decoding: Routing.Coarse-to-fine selection proceeds top-down through the hierarchy. At each
level ℓ, the model selects kℓsummary tokens from the candidates exposed by the previous level’s
selection:
1.Levelt(top).Score allM t=⌈M/Jt−1⌉top-level tokens and select the top-k t. Cost:O(M t).
2.Level ℓ < t .The kℓ+1selected tokens at level ℓ+ 1 expose kℓ+1·Jcandidate tokens at level ℓ.
Score these and select the top-k ℓ. Cost:O(k ℓ+1·J).
The total routing cost is:
O(M t)|{z}
top level+t−1X
ℓ=1O(kℓ+1·J)|{z}
levelℓ=OM
Jt−1+ (t−1)·k max·J
,(B.4)
where kmax= max ℓkℓ. When t=⌈logJM⌉, the first term becomes O(M/JlogJM−1) =O(J) , a
constant. The second term is O(k max·J·logJM). Since kmaxandJare fixed constants, the total
routing cost is:
O(J+k maxJlogJM) =O(logM) =O(log(n/L)) =O(logn).(B.5)
21

Decoding: Attention.After routing, the query attends to the selected tokens at every level of
the hierarchy: kℓselected summary tokens at each level ℓ∈ {2, . . . , t} , plus k1selected gist tokens
and their k1·Lunfolded raw tokens at the bottom level. The total number of key-value pairs in
the attention context isPt
ℓ=1kℓ+k1·L. Since each kℓis a small constant and t=O(logJM), the
summary tokens contribute O(logM) and the raw tokens contribute O(k 1L). The attention cost is
thereforeO(k 1L+ logM).
Decoding: Total.Combining routing and attention, the per-step decoding cost is:
O(logn+k 1L) =O(logn),(B.6)
which islogarithmicin the context length. Over a full sequence of ndecoding steps, the aggregate
cost isO(nlogn), which islog-linear.
C Experiments on Passkey Retrieval
To evaluate fine-grained retrieval under compression, we test on the passkey retrieval benchmark,
where a short passkey is inserted at a random position within a long distractor context and the
model must locate and reproduce it exactly. Figure 4 reports retrieval accuracy as a function of
context length (horizontal axis) and relative insertion position (vertical axis) for both after continue
pretrained GSA and H-GSA on Qwen2-7B-Instruct (up to 50K words, 2.5× the maximum sequence
length seen during continued pretraining) and Llama-3.2-1B (up to 40K words, 10× the maximum
sequence length seen during continued pretraining).
Both GSA and H-GSA achieveperfect 100% accuracyacross all context lengths, all insertion positions,
and both model families. The uniformly green heatmaps confirm that the selective unfolding
mechanism reliably identifies and recovers the passkey regardless of where it is placed—beginning,
middle, or end—and regardless of how long the surrounding context is. This holds even for the
hierarchical variant, where the passkey must be located through multi-level coarse-to-fine routing.
The result demonstrates that GSA’s selective unfolding mechanism is robust and position-unbiased.
It also demonstrates that GSA generalizes robustly to sequence lengths far exceeding those seen
during continued pretraining.
5K10K15K20K25K30K35K40K45K50K
Word Count5
15
25
30
45
55
65
75
85
95Position
020406080100
(a) GSA on Qwen2
5K10K15K20K25K30K35K40K45K50K
Word Count5
15
25
30
45
55
65
75
85
95Position
020406080100 (b) H-GSA on Qwen2
4K8K12K16K20K24K28K32K36K40K
Word Count5
15
25
30
45
55
65
75
85
95Position
020406080100 (c) GSA on Llama3.2
4K8K12K16K20K24K28K32K36K40K
Word Count5
15
25
30
45
55
65
75
85
95Position
020406080100 (d) H-GSA on Llama3.2
Figure 4: Passkey retrieval accuracy of GSA and H-GSA on Qwen2-7B-Instruct (left two panels) and
Llama3.2-1B (right two panels). The vertical axis denotes the context length, while the horizontal
axis represents the relative insertion position (%) of the passkey.
22