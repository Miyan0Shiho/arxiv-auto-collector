# InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context

**Authors**: Xin Teng, Canyu Zhang, Shaoyi Zheng, Danyang Zhuo, Tianyi Zhou, Shengjie Wang

**Published**: 2026-03-05 16:33:20

**PDF URL**: [https://arxiv.org/pdf/2603.05353v1](https://arxiv.org/pdf/2603.05353v1)

## Abstract
Retrieval-augmented generation (RAG) for long-context question answering is bottlenecked by inference-time prefilling over large retrieved contexts. A common strategy is to precompute key-value (KV) caches for individual documents and selectively recompute a small subset of tokens to restore global causal dependencies, but existing methods rely on heuristics or representation discrepancies without modeling whether selected tokens can effectively influence generation. We cast selective KV recomputation as an information flow problem and show that a simple attention-norm signal from the query reliably identifies tokens that are both semantically relevant and structurally positioned to propagate information, when computed under an inference-consistent RoPE geometry. We therefore reconstruct global positional assignments for retrieved chunks and introduce an information-flow-guided chunk reordering strategy. Experiments on LLM and VLM benchmarks demonstrate consistent gains over prior methods under comparable efficiency budgets.

## Full Text


<!-- PDF content starts -->

InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context
Xin Teng1 *Canyu Zhang1 *Shaoyi Zheng1 *Danyang Zhuo2Tianyi Zhou3Shengjie Wang1
Abstract
Retrieval-augmented generation (RAG) for long-
context question answering is bottlenecked by
inference-time prefilling over large retrieved con-
texts. A common strategy is to precompute
key–value (KV) caches for individual documents
and selectively recompute a small subset of to-
kens to restore global causal dependencies, but
existing methods rely on heuristics or represen-
tation discrepancies without modeling whether
selected tokens can effectively influence genera-
tion. We cast selective KV recomputation as an
information flow problem and show that a sim-
ple attention-norm signal from the query reliably
identifies tokens that are both semantically rele-
vant and structurally positioned to propagate in-
formation, when computed under an inference-
consistent RoPE geometry. We therefore recon-
struct global positional assignments for retrieved
chunks and introduce an information-flow–guided
chunk reordering strategy. Experiments on LLM
and VLM benchmarks demonstrate consistent
gains over prior methods under comparable ef-
ficiency budgets.
1. Introduction
Retrieval-augmented generation (RAG) (Lewis et al., 2020)
has become a dominant paradigm for question answer-
ing (QA) with large language models (LLMs) and vi-
sion–language models (VLMs). In RAG, the model re-
peatedly retrieves and conditions on large sets of external
documents, images, or multimodal evidence, often span-
ning tens or even hundreds of thousands of tokens, while
producing only short answers. In such settings, inference ef-
ficiency is primarily constrained by the prefilling stage that
computes key–value (KV) caches over the retrieved context,
rather than by autoregressive decoding. This bottleneck is
*Equal contribution1New York University2Duke University
3Mohamed bin Zayed University of Artificial Intelligence. Corre-
spondence to: Shengjie Wang<sw5973@nyu.edu>.
Preprint. March 6, 2026.further exacerbated in interactive or multi-query scenarios,
where retrieval results change across queries and prefilling
must be performed repeatedly, rendering na ¨ıve full-context
inference prohibitively expensive.
A natural efficiency strategy in RAG is to precompute KV
caches for individual documents or evidence chunks offline
and assemble the retrieved context at query time by reusing
these cached states. However, after retrieval, autoregres-
sive decoding requires a single globally ordered sequence
with global causal dependencies across retrieved evidence,
whereas each document’s KV cache is computed indepen-
dently under a local causal mask.
To mitigate this mismatch, existing methods introduce selec-
tive KV recomputation, in which a small subset of tokens is
recomputed under the global causal mask to partially restore
cross-document interactions while retaining most of the ef-
ficiency gains of offline precomputation. Representative
approaches differ primarily in how recomputation targets
are selected. CacheBlend (Yao et al., 2025) identifies to-
kens whose representations change the most by measuring
output discrepancies between cached and full-context runs
in early transformer layers; however, for efficiency reasons,
this comparison is limited to only a few shallow layers and
may not reflect a token’s ultimate influence on generation.
In contrast, EPIC (Hu et al., 2024) adopts fixed positional
heuristics, selecting tokens at predefined positions (e.g.,
document boundaries) for recomputation, independent of
content or query. As a result, neither approach fully cap-
tures token importance from two critical perspectives: (i)
semantic relevance, particularly with respect to the query,
and (ii) effective information flow, i.e., whether a token is
structurally positioned to influence downstream decoding
under the global causal attention graph. This limitation
motivates a more principled recomputation criterion that ex-
plicitly aligns token selection with the information pathways
required for answer generation.
In this work, we view selective KV recomputation as an
information flow problem, where the goal is to restore the
pathways through which retrieved evidence can effectively
influence answer generation. We find that a simple attention-
norm criterion from the query to context tokens is highly
effective for identifying recomputation targets. This signal
simultaneously captures semantic relevance, by reflecting
1arXiv:2603.05353v1  [cs.LG]  5 Mar 2026

Information-Flow-Aware KV Recomputation for Long Context
Figure 1.Context chunks are prefetched independently using chunk-local RoPE. At inference time, retrieved chunks are concatenated
with the prompt, global RoPE positions are reconstructed, and prompt-conditioned attention norms are used to select high-impact tokens
for full-context KV recomputation. The recomputed KV states are concatenated with cached chunks, restoring cross-chunk interactions.
An optional chunk reordering step places more informative chunks closer to the prompt.
query–token affinity, and effective information flow, by iden-
tifying tokens that are structurally positioned to propagate
information under the global causal attention induced by
retrieval and decoding. Crucially, this property holds only
when attention norms are computed under a RoPE order-
ing consistent with inference-time decoding, using a global
positional reconstruction of retrieved chunks; chunk-local
or mismatched positional configurations yield unstable and
misleading token rankings. Building on this insight, we
further propose a retrieved-chunk reordering strategy based
on identified information-flow–critical tokens, improving
the effectiveness of downstream attention and providing
additional validation of our information-flow perspective.
Extensive experiments on both LLM and VLM benchmarks
show that our approach consistently outperforms existing
recomputation methods under comparable budgets.
While our primary focus is retrieval-augmented generation
with precomputed document-level KV caches, the proposed
chunking-and-recompute strategy also applies when such
caches are unavailable. In non-RAG long-context inference,
the input can be partitioned into chunks whose KV caches
are computed independently, followed by selective recom-
putation under the global causal mask. This yields two
practical benefits: (i) in multi-query or interactive settings,
prefetched KV caches can be stored and reused to amor-
tize prefilling cost, and (ii) chunk-wise prefilling naturally
enables parallel computation across multiple GPUs, since
chunks are processed independently prior to recombination
and recomputation.
We summarize our contributions as follows.
•We propose a simple and effectiveattention-
norm–basedcriterion from the query to context to-
kens that jointly captures semantic relevance and effec-tive information flow under the global causal attention
graph.
•We show that reliable recomputation target selection
requires aninference-consistent RoPE ordering, and
introduce a global positional reconstruction of retrieved
chunks to enable stable and meaningful token ranking.
•Building on the identified information-flow–critical
tokens, we further propose aretrieved-chunk reorder-
ing strategythat improves downstream attention ef-
fectiveness, providing additional validation of our
information-flow perspective.
•We evaluate our approach on both LLM and VLM RAG
benchmarks, demonstrating consistent improvements
over existing selective recomputation methods under
comparable efficiency budgets.
2. Related Work
2.1. KV Eviction & Compression
KV eviction and compression methods reduce long-context
inference cost by limiting the size or precision of KV
caches during decoding. Eviction-based approaches such as
H2O (Zhang et al., 2023) and StreamingLLM (Xiao et al.,
2023) discard cached KV states based on heuristics, includ-
ing recency or attention importance. Compression methods
such as Pyramid KV (Cai et al., 2024) and memorization-
based approaches (Wu et al., 2022) approximate histori-
cal KV representations through hierarchical compression
or summary memory. These methods primarily target the
memory footprint and access cost during autoregressive
decoding. In contrast, our work addresses the prefilling
bottleneck under chunk-wise processing, where KV states
are computed independently, and cross-chunk interactions
2

Information-Flow-Aware KV Recomputation for Long Context
are lost. Rather than shrinking or discarding KV caches,
we retain token-level KV states and selectively recompute a
small subset of tokens to restore global information flow.
2.2. Chunked Input Processing
Chunked input processing has been explored to extend effec-
tive context length beyond the model’s native window. Con-
text Expansion with Parallel Encoding (CEPE) (Yen et al.,
2024) encodes long contexts in parallel using a lightweight
encoder and enables a decoder to attend to chunk repre-
sentations via cross-attention. A key limitation of such
approaches is that chunks are encoded independently, and
cross-chunk interactions are not explicitly modeled within
the chunk representations. Our method instead targets inter-
action across chunks by selectively recomputing KV states
under the full context. Moreover, CEPE requires addi-
tional training to align the parallel encoder with the decoder,
whereas our approach operates in a plug-and-play manner
at inference time without modifying the pretrained model.
2.3. KV Recomputation
KV recomputation methods aim to recover accuracy when
reusing independently prefetched KV caches by selectively
recomputing a small subset of tokens under the full context.
CacheBlend identifies tokens whose cached KV states devi-
ate most from full-context attention, while EPIC mitigates
positional mismatch by recomputing fixed chunk-initial to-
kens. These approaches select recomputation targets based
on token-level deviation or positional heuristics, without
explicitly modeling how information propagates to gener-
ated tokens. In contrast, our method selects tokens based on
their capacity to transmit information across chunks, jointly
accounting for semantic relevance and positional influence
under inference-time attention geometry.
3. Preliminary
3.1. KV Cache
We consider an autoregressive Transformer decoder. Let
hℓ
t∈Rddenote the hidden state at layer ℓand position t.
For each attention head, keys and values are computed as
linear projections of the hidden states,
kℓ
t=Wℓ
Khℓ
t,vℓ
t=Wℓ
Vhℓ
t,(1)
where the head index is omitted for clarity.
During autoregressive decoding, to avoid recomputing keys
and values for previously processed tokens, Transformer de-
coders maintain a key–value (KV) cache that stores all past
{kℓ
s,vℓ
s}s≤tat each layer. We denote the cached tensors up
to positiontby
Kℓ
≤t∈Rt×dh,Vℓ
≤t∈Rt×dh,(2)where dhis the per-head dimension. In practice, inference
consists of aprefillingphase that computes KV states for an
input context of length Lin a single forward pass, followed
by an iterativedecodingphase that appends one new KV pair
per generated token. In long-context question answering,
where the generated answer is short, inference cost is often
dominated by the prefilling stage.
3.2. Rotary Positional Embedding (RoPE)
Rotary positional embedding (RoPE) encodes positional
information by applying position-dependent rotations to
queries and keys. For vectors decomposed into 2D sub-
spaces, RoPE applies
RoPE(x, t) =cos(θ it)−sin(θ it)
sin(θ it) cos(θ it)
x,(3)
where the angular frequencies are defined as
θi= 10000−2i/d, i= 0, . . . ,d
2−1.(4)
Lower-index dimensions correspond to higher frequencies
and are more sensitive to positional differences, while
higher-index dimensions encode lower-frequency compo-
nents. Consequently, the effectiveness of attention interac-
tions depends on the absolute positions at which tokens are
placed, which becomes particularly relevant when KV states
are computed independently and later reused or recomputed
under different positional assignments.
4. Method
In this section, we introduce our method for selecting re-
computation targets by jointly considering token semantic
relevance and positional influence, with the goal of facilitat-
ing information propagation during decoding.
4.1. Input Chunking and Prefilling
Let the input consist of Ntokens, which we partition into
Kdisjoint chunks {C1, . . . , C K}. Each chunk serves as a
basic unit for prefilling and can correspond to a naturally
independent segment (e.g., a document or an image), or
a contiguous partition of a single long input (e.g., a text
sequence or a visual token grid).
During chunk-wise prefilling, each chunk is processed inde-
pendently to compute its key–value (KV) cache. Within a
chunk Ci, positional indices are assigned locally: tokens are
indexed from 0to|Ci| −1 according to their order inside
the chunk, regardless of their absolute position in the full
input. Keys and queries are then computed using RoPE
based on these chunk-local positions, and the resulting KV
states are stored in the cache associated with Ci. As a result,
chunk-wise prefilling encodes KV states under chunk-local
3

Information-Flow-Aware KV Recomputation for Long Context
positional geometry. When chunks are later recombined
or recomputed under global positional assignments, this
mismatch in positional encoding motivates the need for
selective KV recomputation.
4.2. Token Selection and Recomputation
Selection MethodToken selection is performed after the
prefetched chunks are concatenated with the prompt, using
the KV states obtained from the prefilling stage.
To formalize token selection, we distinguish between chunk-
local positions used during prefilling and global positions
used during token selection and decoding. We assume that
the global ordering of chunks is known and fixed, e.g., deter-
mined by retrieval ranking in RAG settings or by the natural
ordering of documents and images. For a context token
t∈C i, its global position during token selection is given by
g(t) = ∆ ctx+pi(t),(5)
where ∆ctxspecifies the starting position of the chunk in
the whole context. Similarly, prompt tokens are assigned
global positions
g(p) = ∆ pr+ppr(p),(6)
where ppr(p)denotes the prompt-internal ordering and ∆pr
is the prompt offset.
Under a given positional configuration, recomputation tar-
gets are selected based on prompt-conditioned attention
norms. Let A∈RM×Ndenote the prompt-to-context at-
tention matrix at a given layer, where Mis the prompt size
andAijrepresents the attention weight from the i-th prompt
token to the j-th context token. For each context token j,
we define its importance score as the aggregated attention
mass received from all prompt tokens,
sj=MX
i=1Aij.(7)
Recomputation targets are then selected as the top- kcontext
tokens with the largest importance scores,
S= Top-k 
{sj}N
j=1
.(8)
We use prompt-to-token attention rather than token-to-
prompt attention as the selection signal, because prompt-to-
token attention directly determines how much information
is retrieved from each context token during decoding, and
therefore has an immediate impact on next-token prediction.
This choice is also consistent with the autoregressive nature
of language models, where each generated token attends
to previously cached keys and values. Intuitively, tokens
that receive larger attention mass from the prompt are more
likely to contribute to subsequent generation, making them
natural candidates for recomputation.KV RecomputationGiven the selected set of recomputa-
tion targets, we perform KV recomputation under the full
global context. Specifically, for each selected token, we
recompute its key and value representations by running a
standard forward pass with the token placed at its global
position in the concatenated sequence. The recomputed
KV states then replace their chunk-local cache, while all
non-selected tokens reuse their prefetched KV states.
RoPE GeometryBased on the formulation above, token
selection under chunk-wise prefilling is influenced by how
the prompt and the chunked context are positionally encoded
after they are concatenated. In particular, the effectiveness
of information transmission from context tokens to down-
stream autoregressive decoding depends on two intrinsic ge-
ometric properties of the resulting RoPE assignment: (i) the
RoPE frequency range in which context tokens are placed,
and (ii) the relative RoPE positional proximity between the
prompt and the context. The first property determines how
sensitively relative positional differences between tokens
are encoded by RoPE, while the second governs whether
prompt-to-context attention can be effectively established
under causal decoding. These properties are not treated as
experimental factors, but as structural aspects of how the
selection method is instantiated under different positional
assignments.
Within this formulation, we proposed four representative
RoPE allocation configurations that correspond to distinct
positional arrangements under chunk-wise processing.(1)
Global Positioning (GLOBAL).Both context and prompt
tokens are assigned RoPE positions according to their abso-
lute indices in the full global sequence. This configuration
reflects the positional geometry encountered in full-context
prefilling.(2) Head-Local Context + Head Prompt (HL–
HP).Context chunks are assigned chunk-local RoPE posi-
tions with ∆ctxstarting from zero, and the prompt is placed
immediately after the chunked context. This configuration
places both context and prompt tokens in the low-index
(high-frequency) RoPE range while maintaining close posi-
tional proximity.(3) Head-Local Context + Tail Prompt
(HL–TP).Context chunks are the same as method (2), while
the prompt is placed at its global position index. This con-
figuration increases positional distance between the prompt
and the context and places them in different frequency re-
gions.(4) Tail-Local Context + Tail Prompt (TL–TP).
Prompt tokens are assigned RoPE positions according to
their global indices, and all context chunks are placed imme-
diately before the prompt. This configuration pushes context
tokens toward higher-index (low-frequency) RoPE regions,
while preserving their relative ordering with respect to the
prompt. Taken together, these four configurations define a
compact yet expressive set of RoPE allocation patterns un-
der chunk-wise processing, covering the principal ways in
4

Information-Flow-Aware KV Recomputation for Long Context
which frequency placement and prompt–context proximity
can arise in practice.
4.3. KV reorder
Beyond KV prefilling and token selection, we further con-
sider the ordering of chunks at inference time as an addi-
tional degree of freedom for facilitating information flow.
Under RoPE-based causal decoding, tokens placed closer to
the prompt in the sequence are more likely to interact with
prompt queries through attention, which facilitates more
effective propagation of contextual information to down-
stream generated tokens. We therefore introduce an optional
chunk reordering step for settings where input chunks corre-
spond toindependent segments, such as multi-document
retrieval, and where no intrinsic sequential order needs to
be preserved. For inputs with intrinsic sequential structure,
such as a single long document, reordering may disrupt
coherence and is therefore not used. To enable chunk re-
ordering, we first derive chunk-level importance scores.
Concretely, we perform an initial round of token selection
within each chunk independently, using local RoPE rather
than global positions. This makes token selection within
each chunk comparable and prevents chunks closer to the
prompt from being systematically favored. After reordering,
we reselect tokens under the updated chunk order to obtain
the final set of recomputation tokens. This second selection
pass accounts for the fact that reordering alters the original
prompt–context RoPE geometry, and ensures the selected
tokens remain important under the new layout. Autoregres-
sive decoding is then performed using the reordered chunks
and the corresponding recomputed KV states.
Thus, chunk reordering leverages the freedom in chunk or-
dering to further align RoPE geometry with information
flow in the decoding phase, complementing token selection
under chunk-wise prefilling. More details on KV recompu-
tation implementation can be found at Appendix. B
5. Ablation Result
5.1. RoPE Geometry
We ablate different RoPE geometry configurations to study
how positional allocation during token selection affects
downstream performance.
Table 1 shows that RoPE geometry has a substantial impact
on recomputation effectiveness. Among the tested configura-
tions, GLOBAL consistently achieves the best performance
across all four benchmarks. In contrast, configurations that
assign prompt and context tokens to separate or truncated
positional ranges (e.g., HL–HP and TL–TP) result in notice-
ably lower scores. This suggests that mismatches between
the positional layout used for selection and the actual decod-ing order can degrade the quality of selected recomputation
targets.
These results indicate that effective selective recomputation
benefits from evaluating token importance under a positional
layout that closely matches inference-time decoding. Based
on this observation, we adopt GLOBAL as the default RoPE
geometry for token selection in our method. In the reorder-
ing setting, we use HL–TP for the first-stage selection to
encourage diversity across chunks, followed by GLOBAL
for the second-stage selection to refine recomputation tar-
gets under the final positional layout.
Table 1.Ablation of RoPE geometry configurations on Qwen
under the passage-split setting. Higher is better.
Method 2WikiMQA MuSiQue HotpotQA NarrativeQA
HL–HP 0.4455 0.2871 0.5529 0.1481
TL–TP 0.4458 0.2970 0.5693 0.1923
HL–TP 0.4722 0.3072 0.5651 0.2106
GLOBAL0.5019 0.3386 0.5954 0.2288
5.2. RoPE Similarity
To further understand why our selection strategy facilitates
effective information propagation, we analyze the RoPE
similarity between prompt tokens and the selected context
tokens. Importantly, this analysis blocks the contribution
of token semantics: similarities are computed purely from
the RoPE embedding matrices. Specifically, we measure
the similarity between the RoPE embeddings of prompt
tokens and those of the selected context tokens, and report
both the Mean-of-Max (MoM) similarity and the maximum
similarity across selected tokens. Higher values indicate
that selected tokens occupy RoPE frequency bands that are
more closely aligned with the prompt, and are therefore
more positionally reachable under causal decoding.
2WikiMQA HotpotQA
Model Method MoM Max MoM Max
LLaMANorm-based0.5324 0.9773 0.5219 0.9766
CacheBlend 0.5243 0.9133 0.5191 0.8570
EPIC 0.5049 0.6734 0.4985 0.6852
QwenNorm-based0.4548 0.98050.41790.9805
CacheBlend 0.4505 0.80780.42260.8891
EPIC 0.4129 0.5656 0.3835 0.6555
Table 2.RoPE similarity statistics on long-context QA benchmarks.
We report Mean-of-Max (MoM) and Max scores across different
inference strategies. Higher is better.
As shown in Table 2, norm-based selection consistently
yields higher RoPE similarity scores than prior methods
across both datasets and model families, under both the
Mean-of-Max and Max metrics. This result reveals an
alignment between attention salience and RoPE-induced
5

Information-Flow-Aware KV Recomputation for Long Context
Table 3.Task performance comparison across different LLMs. Results are reported under a fixed chunk size of 2048 and passage-split
settings. Higher is better. Best and second-best results are highlighted excluding baseline and no-recompute methods.
Fixed Chunk (2048) Passage Split
Model Method 2WikiMQA MuSiQue HotpotQA NarrativeQA 2WikiMQA MuSiQue HotpotQA NarrativeQA
QwenBaseline 0.5161 0.3718 0.5922 0.1654 0.5161 0.3718 0.5922 0.1654
No Recompute 0.3948 0.1342 0.4633 0.1137 0.1162 0.1012 0.3059 0.2078
Our 0.5089 0.3384 0.5967 0.2110 0.5019 0.3386 0.5954 0.2288
Our + Reorder 0.4773 0.2872 0.5053 0.2251 0.5058 0.3285 0.5972 0.2310
CacheBlend 0.4417 0.2611 0.5352 0.2170 0.4330 0.2765 0.5738 0.2197
EPIC (15%) 0.4321 0.2368 0.5284 0.1999 0.3697 0.2480 0.5443 0.2291
LLaMABaseline 0.4588 0.3285 0.5410 0.1623 0.4588 0.3285 0.5410 0.1862
No Recompute 0.3969 0.2523 0.4671 0.2639 0.3066 0.2462 0.4253 0.2810
Our 0.4635 0.3104 0.5150 0.2891 0.4208 0.2996 0.5123 0.3141
Our + Reorder 0.4455 0.3044 0.5053 0.2957 0.4417 0.2793 0.5202 0.3243
CacheBlend 0.4131 0.2823 0.4720 0.2685 0.3976 0.2708 0.4872 0.3095
EPIC (15%) 0.4087 0.2638 0.4755 0.2701 0.3885 0.2852 0.4898 0.3102
ChatGLMBaseline 0.5253 0.3946 0.6003 0.3264 0.5253 0.3946 0.6003 0.3264
No Recompute 0.4370 0.2833 0.5024 0.2758 0.3523 0.2474 0.4388 0.3181
Our 0.5064 0.3688 0.5739 0.3239 0.4890 0.3758 0.5614 0.3100
Our + Reorder 0.5176 0.3786 0.5820 0.3140 0.4666 0.3635 0.5567 0.3180
CacheBlend 0.4226 0.2624 0.5177 0.2970 0.3757 0.3188 0.5164 0.3179
EPIC (15%) 0.4401 0.2902 0.5362 0.2962 0.4521 0.3091 0.5481 0.3142
positional structure. Specifically, tokens with high prompt-
conditioned attention norms tend to occupy positions that
are also favorable for information transmission in RoPE
space. This observation suggests that norm-based selection
naturally identifies recomputation targets that are efficient
from a purely geometric perspective, even without explicitly
incorporating RoPE similarity into the selection criterion.
6. Experiment
6.1. Experiment Setup
Methods.We compare our approach against a set of rep-
resentative baselines and prior KV recomputation methods.
Across all experiments, we evaluate the following infer-
ence strategies: (i)Baseline, which performs full-context
prefilling without chunking; (ii)No Recompute, which ap-
plies chunk-wise prefilling without any KV recomputation;
(iii)Our, the proposed semantic- and position-aware selec-
tive KV recomputation method; (iv)Our + Reorder, which
further incorporates the segment reordering strategy; (v)
CacheBlend; and (vi)EPIC (15%), which recomputes a
fixed proportion of tokens.
LLM Benchmarks.We conduct long-context question
answering experiments on three large language mod-
els: Qwen3-14B (Yang et al., 2025), Llama-3.1-8B-
Instruct (Grattafiori et al., 2024), and GLM-4-9B (GLM
et al., 2024). Experiments are performed on four
datasets from LongBench: 2WikiMQA (Ho et al., 2020),MuSiQue (Trivedi et al., 2022), HotpotQA (Yang et al.,
2018), and NarrativeQA (Ko ˇcisk`y et al., 2018). All datasets
are evaluated using their official evaluation protocols, and
we report the standard metrics specified by each benchmark.
We also include the Needle-in-the-Haystack analysis in Ap-
pendix A.
VLM Benchmarks.To evaluate the generality of our
method in multimodal settings, we further conduct experi-
ments on the vision–language model Qwen3-VL-8B (Bai
et al., 2025). We evaluate on three representative vision–
language benchmarks: OCRBench (Liu et al., 2024),
ChartQA (Masry et al., 2022), RealWorldQA (xAI, 2024),
HRBench4K (Wang et al., 2025), and InfoVQA (Mathew
et al., 2022). All results are obtained following the official
evaluation protocols of the respective datasets.
Efficiency Measurement.All efficiency measurements
are conducted on a single NVIDIA A100 GPU. For each
method, we measure end-to-end inference latency and report
the median runtime over 10 independent runs.
6.2. LLM Result
Analysis on LongBench.Table 3 reports long-context QA
performance on LongBench across three LLM backbones
under both fixed-chunk and passage-split settings.
Under both evaluation settings,Our Methodachieves
the strongest and most consistent performance among
6

Information-Flow-Aware KV Recomputation for Long Context
Model Method RealWorldQA ChartQA OCRBench HRBench4K infoVQA val
Qwen3-VL-8B (k= 0)Baseline (No Recompute) 0.7059 83.08 878 0.74875 83.07
Qwen3-VL-8B (k= 2)No Recompute 0.6745 71.32 839 0.72750 71.64
Our0.6810 73.488420.72625 73.07
CacheBlend 0.6758 71.728450.72375 72.00
EPIC 0.6745 70.92 836 0.72500 71.51
Qwen3-VL-8B (k= 4)No Recompute 0.6588 62.00 781 0.68875 57.88
Our0.6667 65.68 802 0.69125 62.23
CacheBlend 0.6562 62.68 786 0.68625 58.56
EPIC 0.6549 62.48 785 0.67250 57.82
Table 4.Performance comparison on Qwen3-VL-8B across multiple vision–language QA benchmarks under different recomputation
budgets k. We compare our method with prior recomputation strategies while keeping the number of recomputed tokens fixed. The k= 0
setting corresponds to standard inference without recomputation.
recomputation-based methods. It attains the best or second-
best results across all four benchmarks on Qwen, LLaMA,
and ChatGLM, compared to previous methods, with par-
ticularly strong gains on multi-hop reasoning tasks such
as 2WikiMQA, MuSiQue, and HotpotQA. These tasks re-
quire aggregating evidence distributed across distant context
segments, making them especially sensitive to cross-chunk
information loss.
We further observe thatOur + Reordermaintains compa-
rable performance and yields additional gains in several
passage-split and narrative-style settings, suggesting that
reordering helps when chunks are independent, because it
places informative chunks closer to the prompt for better
information propagation.
Overall, these results validate the effectiveness of our ap-
proach on LLMs and underscore the importance of selec-
tively recomputing tokens that facilitate information flow
across chunks.
6.3. VLM Result
VLM Results Analysis.Table 4 summarizes performance
on Qwen3-VL-8B across multiple vision–language QA
benchmarks under different recomputation budgets. Across
all datasets, increasing the recomputation budget leads to
consistent performance gains over the no-recompute setting,
indicating that restoring cross-token interactions is also crit-
ical for long-context inference in multimodal models.
Under both k= 2 andk= 4 chunk numbers,Ourmethod
achieves the strongest or most stable performance across
benchmarks. In particular, it consistently outperforms
CacheBlend and EPIC on RealWorldQA, ChartQA, and
OCRBench. The improvements are most evident on struc-
turally demanding tasks such as ChartQA and OCRBench,
where successful reasoning requires integrating dispersedvisual elements with textual cues over long contexts.
Taken together, these results show that the proposed recom-
putation strategy extends beyond language-only models and
remains effective in vision–language settings, providing ro-
bust and consistent improvements across diverse multimodal
reasoning tasks.
7. Efficiency Analysis
Latency with Prepared ContextWe first consider the
inference setting in which the context KV cache is already
prepared. This corresponds to scenarios where long contexts
are prefetched offline or reused across multiple queries, so
that inference primarily consists of selective KV recomputa-
tion and autoregressive decoding. Under this setting, latency
is dominated by recomputation rather than full-context pre-
filling, allowing us to directly examine the speed–accuracy
trade-off induced by different recomputation strategies. As
shown in Fig. 2, we observe a clear Pareto frontier between
TTFT and downstream performance. Within curves, our
method consistently achieves higher accuracy at comparable
latency, or equivalently lower latency at similar performance
levels, indicating that recomputation budget is allocated to
tokens that most effectively support downstream generation.
Latency with On-the-Fly Context PrefillingWe next
consider a more general setting in which no context cache
is available at inference time, and the TTFT includes chunk-
wise prefilling followed by selective KV recomputation and
decoding. In this regime, the cost of attention computation
during prefilling becomes a dominant factor. Empirically,
attention execution on a single device falls short of ideal
quadratic scaling in practice, limiting the efficiency gains
from chunk-wise prefilling. As a result, the prefilling over-
head cannot be effectively amortized without parallelization.
7

Information-Flow-Aware KV Recomputation for Long Context
Baseline No-recompute Our Our+Reorder CacheBlend EPIC (15%)
0 200 400 60030354045
Better
TTFT (ms)Avg F1 (%)(a) LLaMA / 2WikiMQA
0 250 500 750 1,000455055
Better
TTFT (ms)(b) LLaMA / HotpotQA
0 250 500 750 1,0002040
Better
TTFT (ms)(c) Qwen / 2WikiMQA
0 500 1,000 1,500 2,00030405060
Better
TTFT (ms)(d) Qwen / HotpotQA
Figure 2.Speed–accuracy trade-off on LLaMA and Qwen across long-context QA benchmarks. Each curve corresponds to a recomputation
budget sweep. Upper-left indicates a better trade-off.
This motivates the use of multi-GPU sequence parallelism,
which distributes attention computation across devices dur-
ing prefilling. By improving the scaling behavior of atten-
tion, sequence parallelism substantially reduces prefilling
latency and exposes a more favorable efficiency regime for
long-context inference. We therefore evaluate recomputa-
tion strategies under multi-GPU sequence-parallel (Shoeybi
et al., 2019) execution to characterize end-to-end latency
behavior when context must be constructed on the fly.
For long-context settings ( ≥8K tokens), the overhead intro-
duced by KV recomputation becomes negligible compared
to the cost of full-context attention. As shown in Table 5,
our method achieves competitive or better TTFT than ring
attention (Liu et al., 2023) from 8K onward, and consistently
outperforms it at sequence lengths of 16K and above. In
particular, at 16K our method reduces TTFT from 707.8,ms
(ring attention) to 427.6,ms, and at 32K from 2350.1,ms
to 914.0,ms, corresponding to a 2.57× speedup over ring
attention and up to a 3.49× speedup over the single-GPU
full-prefill baseline. These results indicate that selectively
recomputing a small set of information-critical tokens is
substantially more efficient than attending over the full con-
text, and that the recomputation cost is quickly amortized
as sequence length increases.
We note that, under the ring-attention setting, our method
avoids all-gathering the full KV cache across GPUs for
cross-chunk attention. Instead, we communicate only the
small subset of tokens selected for recomputation, and most
recomputation remains local (e.g., when a token lies in the
same sequence-parallel chunk). With a recomputation ratio
of 0.15, a substantial fraction of the selected tokens fall
within the first chunk, further reducing the need for ring-
attention-style cross-GPU attention. As a result, inter-GPU
communication is significantly reduced, and the latency ad-
vantage becomes more pronounced as context length grows,
leading to faster prefilling in long-context regimes.
As shown in Table 6, our method also outperforms ring at-Table 5.TTFT latency and speedup comparison under different
sequence lengths with sequence parallel on 4 NVIDIA H100 GPUs.
Speedup is reported relative to single-GPU prefilling.
Seq Len Method Recompute Ratio TTFT (ms) Speedup
8192 Single-GPU Prefill – 566.7 1.00x
Ring Attention – 247.5 2.29x
Ours 0.15 232.0 2.44x
16384 Single-GPU Prefill – 1285.8 1.00x
Ring Attention – 707.8 1.82x
Ours 0.15 427.6 3.01x
32768 Single-GPU Prefill – 3190.5 1.00x
Ring Attention – 2350.1 1.36x
Ours 0.15 914.0 3.49x
tention in answer quality under the same sequence-parallel
execution setting, improving F1 on all three QA benchmarks.
Specifically, we observe gains of +1.30 on HotpotQA, +2.55
on 2WikiMQA, and +1.44 on Musique. These results indi-
cate that selectively recomputing information-critical tokens
not only preserves accuracy under sequence parallelism, but
can further enhance reasoning performance.
Task Method F1 (%)
HotpotQA Ring Attention 56.53
Ours57.83
2WikiMQA Ring Attention 48.94
Ours51.49
Musique Ring Attention 32.10
Ours33.54
Table 6.Comparison between ring attention and our method across
three long-context QA benchmarks. Under identical sequence-
parallel settings, our method achieves consistent F1 improvements
on all tasks.
8

Information-Flow-Aware KV Recomputation for Long Context
8. Discussion
Recomputation Efficiency under Irregular Attention
MasksSelective KV recomputation requires attending
a dynamically selected subset of tokens to the full context
under a causal constraint, resulting in an irregular attention
mask that is neither fully dense nor strictly causal. Such
patterns are not efficiently supported by existing optimized
attention kernels (e.g., FlashAttention), leading to subop-
timal hardware utilization. In practice, we observe that
recomputation can incur up to 2×overhead relative to its
ideal compute cost, even at small recomputation ratios. This
limitation is shared by selective recomputation methods in
general and is primarily a kernel-level issue. Designing
customized kernels for sparse, index-based causal attention
is a promising direction for future work.
9. Conclusion
We revisit selective KV recomputation from an information-
flow perspective for chunk-wise long-context inference. By
selecting recomputation targets using prompt-conditioned
attention norms under inference-consistent RoPE geome-
try, our method restores effective cross-chunk information
transmission without modifying the pretrained model. Ex-
periments on both language and vision–language models
demonstrate consistent improvements under fixed recom-
putation budgets, highlighting the importance of positional
alignment in efficient long-context inference.
9

Information-Flow-Aware KV Recomputation for Long Context
Impact Statement
This work studies how to identify which tokens should be
recomputed when chunk-wise prefilling is used for long-
context inference in large language and vision–language
models. Rather than introducing a new efficiency mecha-
nism, our contribution lies in improving the way recomputa-
tion targets are selected, by grounding token importance in
prompt-conditioned attention and positional structure.
By providing a principled and empirically validated to-
ken selection strategy, this work can help make existing
recomputation-based inference methods more reliable and
predictable across different models, tasks, and input struc-
tures. A better understanding of which tokens facilitate
information flow may also inform future designs of long-
context inference systems and guide more interpretable and
controllable approximations.
This work does not introduce new model capabilities, train-
ing data, or deployment settings. As a result, it does not
create additional risks related to privacy, bias, or misuse
beyond those already present in pretrained language and
vision–language models. Any downstream impact, there-
fore, depends on how such models are applied, and existing
responsible-use practices remain applicable.
Overall, this work aims to clarify and strengthen the design
principles behind token-level recomputation, contributing
to more robust and well-understood long-context inference
methods.
References
Bai, S., Cai, Y ., Chen, R., Chen, K., Chen, X., Cheng, Z.,
Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z.,
Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li,
Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu,
J., Liu, C., Liu, Y ., Liu, D., Liu, S., Lu, D., Luo, R., Lv,
C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun,
Y ., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang,
Q., Wang, Y ., Xie, T., Xu, Y ., Xu, H., Xu, J., Yang, Z.,
Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang,
H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F.,
Zhou, J., Zhu, Y ., and Zhu, K. Qwen3-vl technical report.
arXiv preprint arXiv:2511.21631, 2025.
Cai, Z., Zhang, Y ., Gao, B., Liu, Y ., Li, Y ., Liu, T., Lu, K.,
Xiong, W., Dong, Y ., Hu, J., et al. Pyramidkv: Dynamic
kv cache compression based on pyramidal information
funneling.arXiv preprint arXiv:2406.02069, 2024.
GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D.,
Zhang, D., Rojas, D., Feng, G., Zhao, H., et al. Chatglm:
A family of large language models from glm-130b to
glm-4 all tools.arXiv preprint arXiv:2406.12793, 2024.Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.
Ho, X., Duong Nguyen, A.-K., Sugawara, S., and Aizawa,
A. Constructing a multi-hop QA dataset for compre-
hensive evaluation of reasoning steps. InProceedings
of the 28th International Conference on Computational
Linguistics, pp. 6609–6625, Barcelona, Spain (Online),
December 2020. International Committee on Compu-
tational Linguistics. URL https://www.aclweb.
org/anthology/2020.coling-main.580.
Hu, J., Huang, W., Wang, W., Wang, H., Hu, T., Zhang, Q.,
Feng, H., Chen, X., Shan, Y ., and Xie, T. Epic: Efficient
position-independent caching for serving large language
models.arXiv preprint arXiv:2410.15332, 2024.
Koˇcisk`y, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann,
K. M., Melis, G., and Grefenstette, E. The narrativeqa
reading comprehension challenge.Transactions of the
Association for Computational Linguistics, 6:317–328,
2018.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Liu, H., Zaharia, M., and Abbeel, P. Ring attention with
blockwise transformers for near-infinite context.arXiv
preprint arXiv:2310.01889, 2023.
Liu, Y ., Li, Z., Huang, M., Yang, B., Yu, W., Li, C.,
Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. Ocr-
bench: on the hidden mystery of ocr in large multi-
modal models.Science China Information Sciences, 67
(12), December 2024. ISSN 1869-1919. doi: 10.1007/
s11432-024-4235-6. URL http://dx.doi.org/
10.1007/s11432-024-4235-6.
Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E.
Chartqa: A benchmark for question answering about
charts with visual and logical reasoning. InFindings of
the association for computational linguistics: ACL 2022,
pp. 2263–2279, 2022.
Mathew, M., Bagal, V ., Tito, R., Karatzas, D., Valveny,
E., and Jawahar, C. Infographicvqa. InProceedings
of the IEEE/CVF Winter Conference on Applications of
Computer Vision, pp. 1697–1706, 2022.
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper,
J., and Catanzaro, B. Megatron-lm: Training multi-
billion parameter language models using model paral-
lelism.arXiv preprint arXiv:1909.08053, 2019.
10

Information-Flow-Aware KV Recomputation for Long Context
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Musique: Multihop questions via single-hop ques-
tion composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y .,
Yu, W., and Tao, D. Divide, conquer and combine: A
training-free framework for high-resolution image percep-
tion in multimodal large language models. InProceed-
ings of the AAAI Conference on Artificial Intelligence,
volume 39, pp. 7907–7915, 2025.
Wu, Y ., Rabe, M. N., Hutchins, D., and Szegedy, C. Mem-
orizing transformers.arXiv preprint arXiv:2203.08913,
2022.
xAI. Grok-1.5 vision preview. https://x.ai/blog/
grok-1.5v, 2024. Accessed: 2025.
Xiao, G., Tian, Y ., Chen, B., Han, S., and Lewis, M. Ef-
ficient streaming language models with attention sinks.
arXiv, 2023.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W., Salakhut-
dinov, R., and Manning, C. D. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering. In
Proceedings of the 2018 conference on empirical methods
in natural language processing, pp. 2369–2380, 2018.
Yao, J., Li, H., Liu, Y ., Ray, S., Cheng, Y ., Zhang, Q., Du,
K., Lu, S., and Jiang, J. Cacheblend: Fast large language
model serving for rag with cached knowledge fusion. In
Proceedings of the Twentieth European Conference on
Computer Systems, pp. 94–109, 2025.
Yen, H., Gao, T., and Chen, D. Long-context language
modeling with parallel context encoding. InProceed-
ings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
2588–2610, 2024.
Zhang, Z., Sheng, Y ., Zhou, T., Chen, T., Zheng, L., Cai,
R., Song, Z., Tian, Y ., R ´e, C., Barrett, C., et al. H2o:
Heavy-hitter oracle for efficient generative inference of
large language models.Advances in Neural Information
Processing Systems, 36:34661–34710, 2023.
11

Information-Flow-Aware KV Recomputation for Long Context
A. Needle-in-a-Haystack Analysis.
Figure 3 visualizes the robustness of different inference strategies on the Needle-in-a-Haystack task, where a single relevant
fact is placed at varying depths within increasingly long contexts. The baseline model maintains near-perfect retrieval
accuracy across all depths and context lengths, while chunk-wise prefilling without recomputation exhibits severe degradation
as context length increases, indicating a near-complete loss of long-range information access.
Selective KV recomputation substantially alleviates this failure mode. Both Our and Our + Reorder recover high retrieval
accuracy across most depths, demonstrating that recomputing a small number of carefully selected tokens is sufficient to
restore effective long-range information flow. Compared to CacheBlend and EPIC, our method exhibits fewer failure regions,
particularly at larger context lengths, suggesting more stable information propagation under causal decoding. These results
further highlight the role of positional allocation and reordering in long-context inference. By implicitly prioritizing tokens
that are both prompt-relevant and positionally effective, our method creates a more favorable prompt–context geometry,
enabling reliable retrieval even when relevant information is deeply buried within the context.
Figure 3.Needle-in-a-Haystack accuracy heatmaps on Qwen3-14B under varying context lengths and needle depths.
12

Information-Flow-Aware KV Recomputation for Long Context
B. More Implementation Details
Norm Layer Selection.To determine which Transformer layer to use for identifying important tokens, we perform a layer-
wise analysis on Qwen models. Specifically, we evaluate prompt–context attention norms extracted from different layers
and measure their downstream impact on long-context retrieval accuracy. We find that attention norms from intermediate-to-
late layers, particularly layers 22–25, consistently yield the strongest performance across context lengths. Based on this
observation, we use a 22-25 layer to compute prompt-conditioned attention norms and select recomputation targets for all
models in our experiments. This choice reflects a practical trade-off between semantic maturity and positional sensitivity,
and we observe that performance is relatively stable within this layer range.
Figure 4.Needle-in-a-Haystack accuracy heatmaps on Qwen3-14B under varying context lengths and needle depths, using attention
norms extracted from different Transformer layers.
Recomputation Layers.Once recomputation targets are selected, we recompute the corresponding key–value states
across all Transformer layers. This ensures that recomputed tokens are fully consistent with the global causal attention
geometry throughout the model, rather than correcting positional mismatches at only a subset of layers. While partial-layer
recomputation is possible, we find full-layer recomputation to be more stable and easier to integrate without introducing
additional hyperparameters.
Discrepancy between the single-GPU baseline and ring attention.In exact arithmetic, ring attention computes the
same attention output as the single-GPU full-prefill baseline. In practice, however, we observe small numerical differences.
The primary reason is that floating-point arithmetic is not associative: ring attention changes the order in which partial
statistics (e.g., row-wise maxima and normalizers for softmax, and the subsequent weighted-sum reductions) are accumulated
across sequence-parallel chunks, leading to slightly different rounding behavior compared to the single-GPU kernel. These
deviations can be amplified under mixed precision, since inputs are typically stored in bf16 and intermediate reductions
may use different accumulation precision depending on the kernel implementation. Thus, the mismatch is a numerical
artifact of reduction order and precision, rather than an algorithmic difference.
13