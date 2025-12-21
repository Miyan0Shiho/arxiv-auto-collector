# Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference

**Authors**: Dhruv Deshmukh, Saurabh Goyal, Nipun Kwatra, Ramachandran Ramjee

**Published**: 2025-12-18 10:37:14

**PDF URL**: [https://arxiv.org/pdf/2512.16391v1](https://arxiv.org/pdf/2512.16391v1)

## Abstract
Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight keys is stable across nearby layers. Kascade computes exact Top-k indices in a small set of anchor layers, then reuses those indices in intermediate reuse layers. The anchor layers are selected algorithmically, via a dynamic-programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across both prefill and decode attention. The Top-k selection and reuse in Kascade is head-aware and we show in our experiments that this is critical for high accuracy. Kascade achieves up to 4.1x speedup in decode attention and 2.2x speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense attention accuracy on long-context benchmarks such as LongBench and AIME-24.

## Full Text


<!-- PDF content starts -->

KASCADE: A PRACTICALSPARSEATTENTIONMETHOD FOR
LONG-CONTEXTLLM INFERENCE
Dhruv Deshmukh1Saurabh Goyal1Nipun Kwatra1Ramachandran Ramjee1
ABSTRACT
Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload
with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages
known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight
keys is stable across nearby layers. Kascade computes exact Top- kindices in a small set ofanchorlayers, then
reuses those indices in intermediatereuselayers. Theanchorlayers are selected algorithmically, via a dynamic-
programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment
across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across
both prefill and decode attention. The Top- kselection and reuse in Kascade ishead-aware and we show in our
experiments that this is critical for high accuracy. Kascade achieves up to 4.1× speedup in decode attention and
2.2× speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense
attention accuracy on long-context benchmarks such as LongBench and AIME-24. The source code of Kascade
will be available athttps://github.com/microsoft/kascade.
1 INTRODUCTION
Large language models are increasingly deployed in settings
that demand long contexts: chain-of-thought style reason-
ing, multi-step tool use, retrieval-augmented generation over
multi-document corpora, coding agents, etc. In long context
inference, the computation cost is dominated by the atten-
tion operation in both the prefill (where attention is O(n2)
for context length n, compared to O(n) MLP operation) and
decode ( O(n) attention vs O(1) MLP) phases. Moreover,
decode attention is memory bandwidth bound and therefore
does not benefit much from batching, making it inefficient
on modern GPUs.
The attention operation is expensive because each token
has to attend to all previous context tokens. A common
method to decrease the cost is sparse attention, where the
attention function is approximated by using only a subset
of the context tokens. Numerous sparse attention methods
have been proposed, including fixed-pattern (Beltagy et al.,
2020; Xiao et al., 2023; Zaheer et al., 2020; Jiang et al.,
2024), workload-aware (Gim et al., 2024; Yao et al., 2025;
Lu et al., 2024; Ma et al., 2025), and dynamic sparsity
variants (Singhania et al., 2024; Zhang et al., 2023; Tang
et al., 2024; Yang et al., 2025c; Gao et al., 2024; 2025).
However, some of these methods require model retraining,
1Microsoft Research India. Correspondence to: Saurabh Goyal
<saurabh.goyal@microsoft.com>.or sacrifice generality across tasks.
In this paper, we present Kascade, a dynamic sparsity based
technique which reduces the cost of attention significantly
while retaining the accuracy of dense attention. Compared
to other training free sparse attention schemes, we find that
Kascade achieves the best accuracy on AIME-24, at a given
sparsity ratio, as shown in Table 2.
Kascade leverages two known observations: 1) the post-
softmax attention scores are inherently sparse, and 2) the
sparsity structure is stable across nearby layers. Figure 1
shows the sparsity inherent in attention operation. As shown,
only 256 (about 10%) of the tokens contribute to over 95%
of the softmax output. This is intuitive, as the softmax
operation exponentially amplifies the relative magnitude of
larger values compared to the smaller ones. Thus, if we have
an oracle that determines the Top- ktokens, which contribute
most to the attention operation, we can get a very accurate
approximation of the operation, at a fraction of the cost.
Figure 2 shows the accuracy of Oracle Top- kwith varying
values of k. As shown, with just 2.5% of tokens, one can
recover almost the full accuracy of dense attention.
However, computing these Top- kvalues efficiently is a fun-
damental challenge as accurate computation will entail read-
ing all the O(n) keysof the tokens and computing the soft-
max. This is where we leverage the second observation —
the exact Top- koflayer iis very close to the exact Top- k
oflayer i+m for reasonable values of m. Figure 3 illus-arXiv:2512.16391v1  [cs.LG]  18 Dec 2025

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
trates this observation. For example, the Top- kof layer 16
captures 99% of the Top-kattention of layers 17 and 18.
These observations motivate our solution: we compute full
attention, and identify Top- ktokens, only on a subset of
layers, which we callanchorlayers, and reuse those Top- k
tokens to compute sparse attention in intermediate layers.
In order to identify the best subset ofanchorlayers, we
propose an automated dynamic programming scheme that
maximizes cross layer similarity scores. This makes it easy
to deploy Kascade on new models. We also observe that
the Top- ktokens vary across heads, and propose a head
remapping technique when reusing these Top- ktokens. We
find this is critical for high accuracy. Therefore, by algo-
rithmically choosing a good set ofanchorlayers, and being
head-aware, Kascade achieves the best accuracy on AIME-
24 among similar techniques for a given sparsity ratio.
Kascade incorporates design decisions based on low-level
kernel implementation of the attention operation to get
strong performance gains in both prefill and decode. For
example, in the prefill attention kernel, the QKToperation
is performed over tiles for maximizing parallelization and
efficiency. As a result, consecutive tokens of a sequence in a
Q-tile, share the samekeytokens. Thus, Kascade performs
the Top- kselection at a tile level and not independently
for each token. Kascade kernels are implemented in Tile-
Lang (Wang et al., 2025) and deliver substantial efficiency
gains on H100s (up to 4.1× for decode attention compared
to FlashAttention-3 baseline) with negligible impact on task
accuracy across models and benchmarks.
In summary, we make the following contributions:
•Kascade introduces three efficient mechanisms: tiled
top-K, head remapping and automatic anchor layer
selection, for making sparse attention practical and
accurate.
•Compared to previous techniques, Kascade achieves
higher accuracy at the same Top- k. For example, in
AIME-24, Kascade delivers substantially higher accu-
racy ( 8–10 % absolute) compared to previous schemes
with two different models at 10% Top-k.
•By automating anchor layer selection and head remap-
ping, and with a performant kernel for H100s, we be-
lieve Kascade is the first kernel that makes it easy to
deploy sparse attention for different models.
•On H100s, Kascade delivers up to 4.1× faster perfor-
mance than FlashAttention-3 decode kernel and up to
2.1× faster performance than FlashAttention-3 prefill
kernel, thereby ensuring significant performance gains
while achieving comparable accuracy in long-context
tasks.2 BACKGROUND ANDRELATEDWORK
2.1 Scaled Dot-Product Attention
In scaled dot-product attention (Vaswani et al., 2017), each
new token must attend to all previous tokens. Formally, for
a query qt∈Rd, corresponding to the current token, and
key-value pairs K,V∈RN×dof past tokens, the attention
outputyis computed as:
p= softmaxqt·K⊤
√
d
∈RN(1)
y=p·V∈Rd(2)
HereNis the current sequence length, and dis the hidden
dimension. Equation 1 computes a weight for every token,
and Equation 2 computes the output as a weighted sum of
values in V. As we’ll see in Section 3.1, the pvector is
sparse, and most of the weights are close to 0.
The above operation results in O(N) computation and mem-
ory access per token, making attention the dominant con-
tributor to latency in long-context inference.
2.2 Grouped Query Attention
In Multi-Head Attention (Vaswani et al., 2017), each
head performs independent attention with its own learned
projections of Q, K, V , and their outputs are concate-
nated and linearly transformed. Grouped Query Attention
(GQA) (Ainslie et al., 2023) generalizes this formulation
by allowing multiple query heads to share a common set
of key-value projections, reducing memory bandwidth and
improving efficiency in both training and inference. GQA
is widely adopted in recent models, but imposes additional
constraints on efficient implementation of sparse attention
schemes, which we explore in Section 3.4.
2.3 Sparse Attention Methods
As we observed in Section 2.1, the attention operation is
expensive because each token has to attend to all previous
context tokens. To address this bottleneck, various sparse
attention mechanisms have been proposed.
Fixed pattern sparsityapproaches fix a connectivity pat-
tern (e.g., sliding windows plus a small set of global to-
kens) (Beltagy et al., 2020; Xiao et al., 2023; Zaheer et al.,
2020; Jiang et al., 2024) and compute attention only over
these tokens. Some models deploy sliding window atten-
tion on a subset of layers to limit the attention cost (Team
et al., 2024; Agarwal et al., 2025). However, these ap-
proaches work best when baked into the architecture before
pre-training, as the model needs to learn to attend within
this connectivity pattern; or require some amount of post-
training.

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
Another class of techniques isworkload-aware sparsity
which primarily targets RAG like scenarios and limits the
attention operation of document tokens to within the context
of the same document (Gim et al., 2024; Yao et al., 2025;
Lu et al., 2024; Ma et al., 2025). These techniques may also
require some post-training to maintain accuracy, and they
only optimize the prefill phase of inference.
The final family of approaches isdynamic sparsity, where a
subset ofktokens is dynamically selected for the attention
computation (Ribar et al., 2024; Singhania et al., 2024;
Zhang et al., 2023; Tang et al., 2024; Yang et al., 2025c;
Gao et al., 2024; 2025). Selecting the best tokens efficiently
is, however, an open research problem.
2.4 Cross-Layer Similarity in Attention
One key observation we use, described in Section 3.2, is
that the sparsity patterns of attention weights exhibit strong
correlations across neighboring layers. Prior works (Ying
et al., 2021; Bhojanapalli et al., 2021; Xiao et al., 2019)
have evaluated sharing full attention weights across layers,
which we found to degrade accuracy.
Some recent works like OmniKV (Hao et al., 2025), TidaDe-
code (Yang et al., 2024), and LessIsMore (Yang et al.,
2025b) have also used this observation to design an ap-
proximate Top- kattention scheme. OmniKV has a focus
on reducing memory capacity requirements, and hence of-
floads the KV cache to the CPU. The benefit in memory
capacity, however, comes at the cost of reduced performance
because of CPU-GPU transfers. LessIsMore, built on top
of TidalDecode, computes Top- kindices on fewer layers
chosen manually. A key challenge with these schemes is
that there is not automated way to identify theanchorlayers
which makes it difficult to deploy to new models. These
schemes use a shared set of Top- kindices across all heads,
while we find separate Top- kindices for every key head to
improve accuracy, as described in Section 3.5.
3 KASCADE
We now outline the various insights and techniques needed
for a complete design and implementation of Kascade.
3.1 Oracle Top-kSelection
We first ask a feasibility question: can we approximate atten-
tion using only a small subset of past tokens without losing
task accuracy? Since the softmax operation in Equation 1 ex-
ponentially amplifies the relative magnitude of larger values
compared to the smaller ones, it has an inherent sparsifi-
cation. To exploit this sparsification, we define anOracle
Top-k, where we compute the attention, in Equation 2, only
overktokens with the highest pvalues. Since computing
these ktokens requires the full softmax operation, we use
0 4 8 12 16 20 24 28
Head ID0
4
8
12
16
20
24
28Layer ID
0.750.800.850.900.951.00
SparsityFigure 1. Attention weight covered by top 256 keys across lay-
ers and heads. Except for layer 0 rest of layers have high spar-
sity across majority of the heads. Model=Llama-3.1-8b-Instruct,
Dataset=MuSiQue.
the termOracle, and this provides an accuracy upper bound.
Figure 1 confirms that the output of softmax is indeed sparse.
For example, 95% of the total attention mass in almost all
layers and heads is captured by the top 256 tokens. The only
exception is layer 0, where the distribution is considerably
flatter. Kascade therefore always computes full dense atten-
tion in layer 0 and applies sparsity only layer 1 onwards.
Figure 2 evaluates how aggressively we can sparsify while
preserving task quality. We replace dense attention with
Oracle Top- kattention, and measure end-task accuracy (F1
on 2WikiMultihopQA(Ho et al., 2020) for Llama-3.1-8b-
Instruct). Even at k/N= 2.5% , Oracle Top- kattention
matches the accuracy of full attention. This shows that, if
Top-kcan be estimated efficiently, the performance upside
can be significant without compromising accuracy.
Kascade is designed to approximate this oracle efficiently.
The next subsections address two challenges of doing this
at runtime: (1) computing the Top- kset without first ma-
terializing full attention, and (2) enforcing GPU-friendly
sharing of Top- kindices across tiles, layers, and heads for
high throughput. We address (1) using cross-layer reuse
of Top- kindices and (2) using head remapping, tile-level
pooling, and GQA-aware constraints.
3.2 Cross-Layer Similarity
We now ask whether we can avoid recomputing the Top- k
set independently in every layer by reusing it across nearby
layers. For a query token q, letP(l,h)
q∈RNdenote the
post-softmax attention distribution, at layer land head h,

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
2 4 6 8 10
T opK Percentage0510152025303540455055F1 Score
Baseline
Oracle T opk
Oracle T opk (Full Attn in Layer 0)
Figure 2. Oracle Top- kattention results with varying Top- kper-
centage. With layer 0 doing full attention, Oracle Top- kmatches
baseline score even with Top- kas 5%. Model=Llama-3.1-8b-
Instruct, Dataset=2WikiMultihopQA.
where Nis the current sequence length. We define the layer
attention distribution Pl
qas the average of P(l,h)
q across all
heads. We then define the Top- kindex set for token qat
layerlasIl
q=topk(Pl
q, k).
To quantify how well the Top- kset from one layer can be
reused in another layer, we define a similarity score between
two layers a, bwhere a < b . For token q, we measure how
much of layer b’s oracle attention mass would be recovered
if we were to force layer bto use the Top- kkeys selected at
layera:
sim(a, b) q=Pk
i=1Pb
q[Ia
q[i]]
Pk
i=1Pbq[Ibq[i]],(3)
wherea > band|Ia
q|=|Ib
q|=k
Values near 1 indicate that the identity of the high-
importance keys is stable across layers. We compute this
similarity for each query token in a prompt, then average
across all tokens in that prompt. We then average again
across a development set of multiple prompts to obtain a
layer-by-layer similarity matrixS∈RL×L, whereLis the
total number of layers.
Figure 3 shows this cross-layer similarity matrix for Llama-
3.1-8b-Instruct using MuSiQue as the development set, with
k= 256 (the average context length in MuSiQue is 2.3K).
Most adjacent layer pairs achieve similarity scores close
to 1. Similarity generally decays with layer distance, but
remains high across short ranges. For example, similarity
score of most nearby pairs stays above 0.98, meaning that
more than 98% of the oracle Top- kattention mass at layer b
is already covered by the Top-kkeys chosen at layera.
With this observation, Kascade computes Top- kindices on
only a small set ofanchor layers. These indices are then
0 4 8 12 16 20 24 28
Layer ID j0
4
8
12
16
20
24
28Layer ID i
0.880.900.920.940.960.981.00
SimilarityFigure 3. Cross layer similarity using top 256 keys. Bright cell in-
dicates that Top- kindices of layer i cover high fraction of attention
covered by Top- kindices of layer j itself. Model=Llama-3.1-8b-
Instruct, Dataset=MuSiQue
used to compute Top- kattention on for the next fewreuse
layers.
3.3 Anchor Layer Selection
Given a budget for the number of anchor layers, we want
to select the set of anchor layers, such that it maximizes
the similarity between the chosen anchor layers and the
corresponding reuse layers. Kascade performs this selec-
tion using the dynamic programming algorithm, shown in
Algorithm 1 which uses the similarity matrix Sas input.
To construct S, we evaluate the similarity scores sim(a, b) q
(equation 3) for every token qin a prompt, then take the
minimumacross tokens in that prompt, rather than the mean.
This makes the score conservative and ensures that the sim-
ilarity is determined by the worst token in a prompt. We
observed that this resulted in a more robust anchor selec-
tion. We used k= 64 for computing the similarity scores
and found it to work well across experiments. The similar-
ity matrix also incorporates the modifications described in
Section 3.4, and Section 3.5.
Prior work has observed that attention in deeper layers can
be less important than attention in earlier layers (He et al.,
2024). We account for this observation by assigning each
layerlan importance weight wl. Ifxi
landyi
lare an (input,
output) pair of attention at layer l, we define the importance
scorewi
las:
wi
l= 1−CosineSim(xi
l,yi
l)
Intuitively, if attention barely changes the representation
(high cosine similarity), that layer’s attention block matters

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
Algorithm 1Anchor Layer Selection
Require:Similarity matrixS, budgetM, layers1. . . L
1:Initializedp[][] =−∞,path[][] = 0
2:dp[1][1] =S[1][1]
3:form= 2→M+ 1do
4:forj=m→L+ 1do
5:dp[m][j] = maxj−1
i=m−1 (dp[m−1][i] +Pj−1
l=iS[i][l])
6:path[m][j] =argmax(.)
7:end for
8:end for
9:Backtrack onpath[M+ 1][L+ 1]to recover{ℓ}
less. The layer’s weight is computed by aggregating this
score over the same development set. The similarity matrix
is then weighted by this importance measure.
sim[i][j] =w j·sim[i][j]
Figure 4 shows the importance score for all layers in the
Llama-3.1-8b-Instruct model showing a sharp decrease in
importance of deeper layers.
0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
Layer ID0.000.050.100.150.200.250.300.35Importance Score
Figure 4. Importance scores of attention blocks of all layers.
Deeper layers have lower importance than the initial layers.
Layer 0 has highest importance. Model=Llama-3.1-8b-Instruct,
Dataset=MuSiQue.
We now discuss more enhancements to Kascade to incorpo-
rate finer attention implementation details.
3.4 Query pooling
Most modern LLMs use GQA (Ainslie et al., 2023) where
multiple query heads share the same KV-heads. For effi-
ciency and maximizing parallelization, the decode GQA
attention kernels construct Q-tiles (for the QKTcomputa-
tion) by combining query values of all query heads sharing
the KV-heads. Similarly, prefill kernels construct Q-tiles
by combining consecutive tokens of the prompt as they all
share the same prefix for the QKToperation. This allows
0 64 128 192 256
Tile Size05101520253035F1 Score
Pre-Softmax
Post-SoftmaxFigure 5. Comparison of Top- kattention accuracy, when pooling
with Pre vs Post Softmax attention scores, across different tile
sizes. Top- kpercentage here is 10%. The smallest tile size is 4
where only the queries corresponding to the same key head are
pooled. Post Softmax is more robust to changes in tile size and
does consistently well across all tile sizes. Model=Llama-3.1-8b-
Instruct, Dataset=MuSiQue
batching of memory loads and reusing the fetched Kvalues
across multiple queries, resulting in higher GPU utilization.
In order to maintain this efficiency, Kascade needs to ensure
that all query tokens in a tile share the same Top- kindices.
We must thus construct a single “pooled” attention score for
these tiles during the Top- kcomputation in the anchor layers.
We consider two pooling strategies.Pre-Softmaxpooling
constructs a pooled query representation (by averaging the
query vectors in a tile), and computes attention once using
this pooled query.Post-Softmaxpooling instead computes
the full post-softmax attention distribution independently
for each query in the tile, and then pools these attention
distributions across the tile. We evaluate these pooling
strategies in the Oracle setting in Figure 5. As shown, Post-
Softmax pooling maintains accuracy even for large tiles,
while Pre-Softmax pooling degrades as tile size increases.
Based on these results, Kascade adopts Post-Softmax pool-
ing. In decode, we pool only across the query heads that
share a key head (GQA pooling). In prefill, we pool across
full tiles of 128 queries (including the GQA grouping),
which matches the tile size used in our dense FlashAttention-
style baselines. This choice lets Kascade reuse a single
Top-kindex set per GQA group in decode and per tile in
prefill, and therefore aligns sparse attention with the kernel
structure used by high-throughput implementations.
3.5 Head Remapping and Reuse
Kascade computes Top- kindices at the granularity of a key
head. Thus, each anchor layer will have HTop-ksets of
indices, where His the number of key heads. This raises the
question — which head’s Top- kindex set in ananchorlayer

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
3 4 5 6 7 8 9 10
T opK Percentage05101520253035F1 Score
Kascade
Kascade (All Heads Pooled)Kascade (No Head Remapping)
Figure 6. Comparison of Kascade variants with head remapping,
without head remapping and pooling across all heads, for differ-
ent Top- kpercentages. The tile size is 128, which is our default
tile size for prefill. No remapping is the worst. Head remap-
ping gives consistent scores across all Top- kpercentages thus
providing a larger operating range than pooling across all heads.
Model=Llama-3.1-8b-Instruct, Dataset=MuSiQue
should be mapped to a given head in a correspondingreuse
layer? One option is to perform a simple 1:1 mapping where
thei’th heads in theanchorandreuselayers are mapped to
each other. However, nothing in the transformer architecture
requires that the head iof one layer be similar to the head i
of another layer. We explore two strategies to handle this.
One strategy is to use a shared set of Top- kindices for all
heads, by pooling the attention weights across all heads in a
layer. This method will not account for any variation in the
Top-kindices across heads. In the second strategy, instead
of forcing a single shared Top- kacross all heads, we build
an explicit mapping from heads in eachreuselayer to the
most similar head in the correspondinganchorlayer. For
computing this mapping, we use the same similarity score
defined earlier, but at a head level, and find a head mapping
that maximizes the similarity. Note that this mapping can
be a many-to-one mapping. A similar technique is proposed
in (Bhojanapalli et al., 2021) for reusing complete attention
weights across layers.
Figure 6 compares these two strategies for Llama-3.1-8b-
Instruct on MuSiQue across a range of Top- kbudgets. We
observe that the head-remapping approach is consistently
more robust, especially at smaller values of Top- k. Kascade
therefore uses head remapping by default, but we also report
results for the variant with a shared Top- k, across all heads,
in Section 4.2 for completeness.
3.6 Efficient Kernel Implementation
We implement Kascade by modifying the FlashAttention
(Shah et al., 2024) kernels for prefill and decode, in Tile-
Lang (Wang et al., 2025). TileLang is a tile-level program-ming language for GPU kernels, which also takes care of
various optimizations like pipelining, memory layouts and
specialized instructions. We use both the original FA3, as
well as TileLang FlashAttention kernels as our baseline in
performance evaluations.
For intermediatereuselayers, we pass the Top- kindices
from the previousanchorlayer, and the head mapping com-
puted on the development set. During attention computation,
we load the keys by consulting the Top- kindices. The key
loads that make a key tile are not contiguous, but given that
each key is large, about 256 bytes, we do not notice any
overhead with this. This is in contrast to claims by block
sparse attention approaches (Tang et al., 2024; Gao et al.,
2024).
Foranchorlayers, we use a multi pass approach. Since
we use Post-Softmax pooling, we need to first compute
the post-softmax attention weights for each qseparately,
and then pool them across a tile. We can not compute
the pooled attention weights in one pass, since softmax
operation requires knowing the full row sum.
•The first pass computes the full attention weight matrix
QKT, as well as the row sum vector (Pm
j=1QKT
ij).
This does about half the work of full attention. In
decodes, we write out both these to HBM. In prefill,
since the attention weight matrix is large, we only
output the row sum vector.
•The second pass outputs the pooled post-softmax atten-
tion weights for each Q-tile. For decodes, we read the
weights from the first pass, do a softmax, and pool. For
prefill, we have to recompute the attention weights, but
as we know the row sum vector, we can also compute
the post-softmax weights and pool them.
•The third pass computes the Top- kindices over the
output of the second pass.
•In the final pass, we compute Top- kattention similar
toreuselayers.
Foranchorlayer 0, we need full dense attention as described
in Section 3.1, so we compute dense attention in the first
pass, and omit the last pass. Figure 8 shows the time split
across these passes. We see that the recomputation in the
second pass, for prefill, is a significant cost. We report
performance numbers in Section 4.3
4 EVALUATION
4.1 Setup
We evaluate Kascade on two popular long context bench-
marks, LongBench and AIME-24. We also choose two,

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
Table 1. Results on Longbench. For StreamingLLM, sliding window is set to 30% with 4 sink tokens. For the Top- kattention methods
Top-kis set to 10%. Note that for Quest, OmniKV , and LessIsMore, the prefill phase uses full attention as they only optimize the decode.
Since longbench is prefill-heavy, the high accuracy obtained by these schemes is not unexpected while Kascade achieves high accuracy
while optimizing both the prefill and decode for this benchmark.
Model Strategy SQA MQA Summ. Fewshot Synthetic Code Avg.Meta-Llama-3.1
-8B-InstructBaseline (Dense) 48.43 43.18 25.99 63.22 34.83 59.89 45.92
StreamingLLM 24.83 25.05 22.41 56.33 12.00 59.89 33.42
LessIsMore (decode-only) 48.15 42.71 25.38 63.05 34.67 59.16 45.52
OmniKV (decode-only) 48.22 43.05 25.97 63.22 34.72 59.33 45.75
Quest (decode-only) 46.97 42.82 25.71 62.33 34.14 54.36 44.39
Kascade 47.41 39.84 25.21 61.32 33.67 62.70 45.02
Kascade (All Heads Pooled) 47.83 40.50 25.34 63.09 34.50 62.95 45.70Qwen3-8BBaseline (Dense) 47.56 41.35 24.15 64.32 34.83 65.90 46.35
StreamingLLM 24.33 28.44 21.01 56.82 12.50 64.32 34.57
LessIsMore (decode-only) 40.87 38.47 23.03 62.38 34.67 63.21 43.77
Quest (decode-only) 44.71 40.46 24.34 62.72 33.94 57.53 43.95
Kascade 44.19 40.38 23.02 60.83 35.00 63.98 44.57
Kascade (All Heads Pooled) 44.87 42.34 23.74 61.99 34.50 62.71 45.02
Table 2. Results on AIME-24. For StreamingLLM, sliding win-
dow is set to 30% with 4 sink tokens. For the Top- kattention
methods, Top- kis set to 10%. Kascade gives the best accuracy
across the board performing well even for decode heavy tasks.
StreamingLLM fails to solve even a single question correctly.
Strategy Avg. Pass@1 (Decode Length)
DeepSeek-R1-
Distill-Llama-8BQwen3-8B
Baseline (Dense) 50.42 (11.3k) 73.75 (14.4k)
StreamingLLM 0.00 ( 7.5k) 0.00 ( 6.9k)
LessIsMore 36.25 (14.8k) 60.83 (17.9k)
OmniKV 39.58 (12.5k) -
Quest 7.50 (22.9k) 25.33 (28.8k)
Kascade 47.92(14.6k) 70.42(15.9k)
Kascade (All Heads Pooled) 41.25 (14.0k) 65.83 (17.9k)
widely used, long context models for evaluation: Llama-3.1-
8b-Instruct (Grattafiori et al., 2024), and Qwen3-8b (Yang
et al., 2025a). Both these models use GQA and can sup-
port up to 128k context length. For AIME-24 evaluations,
instead of Llama-3.1-8b-Instruct, we use DeepSeek-R1-
Distill-Llama-8b (Guo et al., 2025) which is a fine-tuned
version of Llama-3.1-8b for reasoning tasks. The original
Llama-3.1-8b-Instruct has very low baseline accuracy on
AIME-24.
We do accuracy comparisons with dense attention,
Quest (Tang et al., 2024), StreamingLLM (Xiao et al., 2023),OmniKV (Hao et al., 2025) and LessIsMore (Yang et al.,
2025b). We have implemented these algorithms in our own
code-base, borrowing from any publicly available code. We
also evaluate a variant of Kascade where we use a shared
Top-kacross all heads (Section 3.5.
Quest LessIsMore OmniKV Kascade010203040506070Pass@1 Accuracy
7.517.936.242.9
39.646.747.949.2Base Acc: 50.42
0510152025
Decode Length (k)
Base Len: 11.3k
T op-k=10% T op-k=20% Base Acc Decode Length Base Len
Figure 7. Accuracy and Decode lengths for Topk- k10% and 20%.
At 20%, Kascade is very close to the baseline and decode length
also reduces significantly. Model=DeepSeek-R1-Distill-Llama-8b,
Dataset=AIME-24.
Kascade requires choosing a set ofanchorlayers. We use
MuSiQue (Trivedi et al., 2022) as a development set and
the anchor layer selection algorithm in section 3.3 to choose
theanchor layers. Llama-3.1-8b-Instruct has 32 layers,
of which we choose 5 anchor layers, which are [0, 2, 8,

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
Table 3. The time for Kascade is weighted average of times foranchorlayer 0,anchorand reuse columns where the weights are1
32,4
32
and27
32. For decodes, batch size is 64, except at 512k, where it is 32. For prefills, batch size is 1. Kascade gives best speedups at Top- kset
to 10%. Speedups on both original FA3 and Tilelang(TL) implementation of FA3 are shown.
Step Seqlen Topk% FA3Tilelang
(TL)Anchor layer 0 Anchor Reuse Kascade Attn Speedup
Time
(ms)Time
(ms)Time
(ms)Ratio
w.r.t. TLTime
(ms)Ratio
w.r.t. TLTime
(ms)Ratio
w.r.t. TLTime
(ms)FA3 TLDecode8192 10 0.7 0.71 0.92 1.30 0.82 1.15 0.13 0.18 0.24 2.91 2.95
16384 10 1.4 1.39 1.71 1.23 1.45 1.04 0.21 0.15 0.41 3.40 3.37
32768 10 2.93 2.94 3.35 1.14 2.71 0.92 0.35 0.12 0.74 3.97 3.98
65536 10 5.85 5.83 6.74 1.16 5.39 0.92 0.65 0.11 1.43 4.08 4.07
131072 10 11.68 11.64 14.08 1.21 10.78 0.93 1.24 0.11 2.83 4.12 4.11
262144 10 21.77 21.63 25.95 1.20 20.63 0.95 2.31 0.11 5.34 4.08 4.05
524288 10 21.85 21.73 25.78 1.19 20.65 0.95 2.3 0.11 5.33 4.10 4.08
8192 20 0.7 0.71 0.91 1.28 0.89 1.25 0.2 0.28 0.31 2.27 2.30
16384 20 1.4 1.39 1.73 1.24 1.61 1.16 0.36 0.26 0.56 2.50 2.49
32768 20 2.93 2.94 3.58 1.22 3.22 1.10 0.65 0.22 1.06 2.76 2.77
65536 20 5.85 5.83 6.97 1.20 6.22 1.07 1.24 0.21 2.04 2.87 2.86
131072 20 11.68 11.64 13.85 1.19 12.25 1.05 2.42 0.21 4.01 2.92 2.91
262144 20 21.77 21.63 27.79 1.28 24.39 1.13 4.54 0.21 7.75 2.81 2.79
524288 20 21.85 21.73 28.79 1.32 24.93 1.15 4.55 0.21 7.86 2.78 2.77
8192 30 0.7 0.71 0.93 1.31 0.98 1.38 0.27 0.38 0.38 1.85 1.87
16384 30 1.4 1.39 1.9 1.37 1.93 1.39 0.48 0.35 0.71 1.98 1.97
32768 30 2.93 2.94 3.69 1.26 3.66 1.24 0.95 0.32 1.37 2.13 2.14
65536 30 5.85 5.83 7.19 1.23 7.06 1.21 1.82 0.31 2.64 2.21 2.21
131072 30 11.68 11.64 14.61 1.26 15.1 1.30 3.61 0.31 5.39 2.17 2.16
262144 30 21.77 21.63 28.65 1.32 27.63 1.28 6.77 0.31 10.06 2.16 2.15
524288 30 21.85 21.73 28.59 1.32 28.4 1.31 6.79 0.31 10.17 2.15 2.14Prefill8192 10 0.76 1 2.01 2.01 2.01 2.01 0.36 0.36 0.62 1.23 1.62
16384 10 2.96 3.98 7.28 1.83 6.69 1.68 0.94 0.24 1.86 1.59 2.14
32768 10 12.28 17.13 28.97 1.69 25.11 1.47 2.81 0.16 6.42 1.91 2.67
65536 10 53.77 64.65 120.36 1.86 103.77 1.61 9.36 0.14 24.63 2.18 2.62
131072 10 215.76 262.21 483.69 1.84 416.53 1.59 37.18 0.14 98.55 2.19 2.66
262144 10 864.02 1048.01 1955.47 1.87 1696.55 1.62 160.14 0.15 408.30 2.12 2.57
8192 20 0.76 1 2.04 2.04 2.17 2.17 0.47 0.47 0.73 1.04 1.37
16384 20 2.96 3.98 7.5 1.88 7.35 1.85 1.42 0.36 2.35 1.26 1.69
32768 20 12.28 17.13 31.25 1.82 29.78 1.74 4.68 0.27 8.65 1.42 1.98
65536 20 53.77 64.65 128.18 1.98 119.07 1.84 17.07 0.26 33.29 1.62 1.94
131072 20 215.76 262.21 507.71 1.94 476.05 1.82 72.2 0.28 136.29 1.58 1.92
262144 20 864.02 1048.01 2067.97 1.97 1949.78 1.86 308.36 0.29 568.53 1.52 1.84
8192 30 0.76 1 2.12 2.12 2.36 2.36 0.59 0.59 0.86 0.88 1.16
16384 30 2.96 3.98 8.45 2.12 8.82 2.22 1.87 0.47 2.94 1.01 1.35
32768 30 12.28 17.13 32.68 1.91 33.58 1.96 6.5 0.38 10.70 1.15 1.60
65536 30 53.77 64.65 134.21 2.08 132.42 2.05 24.93 0.39 41.78 1.29 1.55
131072 30 215.76 262.21 532.39 2.03 534.61 2.04 106.12 0.40 173.00 1.25 1.52
262144 30 864.02 1048.01 2158.44 2.06 2192.39 2.09 457.54 0.44 727.55 1.19 1.44
13, 14]. Qwen3-8b has 36 layers, of which we choose 5
anchor layers - [0, 2, 7, 14, 23]. For DeepSeek-R1-Distill-
Llama-8b, we use the same layers as Llama-3.1-8b-Instruct,
for Kascade, OmniKV and LessIsMore. OmniKV hasn’t
reported thefilterlayers for Qwen3-8b, so we remove it
from Qwen3-8b comparisons.
For all accuracy results, we use a Top- kof 10%, with a
minimum size of 128. So, if the current sequence length is
L, during decode, the number of Top-kselected is,
k=min(max(0.1·L,128), L)
For Kascade, we also use the Top- kin the prefill phase
in a rolling manner, where each tile attends only to 10%
of previous tokens. For Quest, OmniKV , and LessIsMore,
prefill phase uses full attention. For StreamingLLM, given
it is a weaker comparison, we use a sliding window size of
30% and 4 sink tokens. For AIME-24, we also show how
the accuracy changes as we increase Top-kto 20%.4.2 Accuracy results
LongBenchis composed of 21 long context tasks across 6
categories, covering multi-document QA, single-document
QA, summarization, few-shot learning, code completion,
and synthetic tasks. Almost all the tasks are prefill
heavy, with very few decodes. All techniques except
StreamingLLM and Kascade variants, do not use sparse
attention in the prefill phase. Table 1 presents the results.
We find that all techniques, including Kascade variants, per-
form very well on this benchmark. StreamingLLM is the
only exception that doesn’t perform well.
AIME-24consists of 30 challenging mathematical prob-
lems, which typically require long chain-of-thought reason-
ing to get to the correct answer. Models that have not been
trained for reasoning perform very poorly on these tasks.
Table 2 presents the average ofpass@1score, across 8 runs,
for each method. The attention patterns on these tasks can
be complex, and we see that StreamingLLM is unable to

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
0 100 200 300 400 500
Time (ms)Layer 0 Layer > 0Anchor LayerPass1 Pass2 T opK Reuse
(a) Prefill
0 2 4 6 8 10 12 14
Time (ms)Layer 0 Layer > 0Anchor LayerPass1 Pass2 T opK Reuse
(b) Decode
Figure 8. Time split for attention and Top- kindices computation in anchor layers, in prefill and decode phase, at 128k context length for
Llama-3.1-8b-Instruct setting.
solve any of the problems. We find that on this complex
reasoning task, Kascade performs much better than other
schemes. We also show the average decode length for each
method. For Kascade, the average decode length is about
29% higher than baseline on DeepSeek-R1-Distill-Llama-
8b, and 10% higher on Qwen3-8b. We also evaluate the
shared Top- kacross all heads variant of Kascade, and it
performs worse than default Kascade across both models,
but better than other schemes.
In Figure 7, we evaluate the effect of increasing Top- kto
20%. Kascade continues to have the highest accuracy and is
very close to the baseline. The decode length also reduces,
and is about 13% higher than baseline for Kascade.
4.3 Efficiency results
As discussed in section 3.6, we implemented Kascade ker-
nels in TileLang for both prefill and decode. We ran at-tention microbenchmarks on a single Nvidia H100 GPU,
for varying context lengths. The settings used for attention
are similar to that of Llama-3.1-8b-Instruct, with 32 total
number of heads, 8 key heads, 128 head dimension, in fp16.
For decode benchmarks we use a batch size of 64, except
at context length 512k, which uses a batch size of 32 be-
cause of insufficient memory on a single gpu. Table 3 shows
the results of these benchmarks on a combination of dif-
ferent Top- kpercentages and context lengths. We’ll focus
primarily on Top- kpercentage of 10%, and longer context
lengths. We find that ourreusekernels see the expected
speedup, and take about 10% of the time of full attention.
Anchorlayers take up more time, similar to the time of full
attention. The firstanchorlayer additionally does dense
attention, so takes even more time. Since we have 5anchor
layers, we compute the overall speedup accordingly. For
decode phase, the speedup is about 4.1× wrt both FA3 and
TileLang flashattention baselines, on Llama-3.1-8b-Instruct
settings. For prefill phase, TileLang baseline kernels are

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
about 20% slower than FA3. Further, as discussed in sec-
tion 3.6, our prefill kernels incur some extra costs in Top- k
computation, so our speedup wrt to TileLang is up to2.6×
and wrt to FA3 is up to 2.2× . For Qwen3-8b settings, since
the ratio ofanchorlayers is lower (5 of 36), we’d expect a
higher speedup.
Figure 8 shows the time split of the multiple passes required
foranchorlayers. The time for layer 0is higher because it
does dense attention in addition to computing Top- kindices.
Prefill speedup takes a hit primarily because of the recom-
putation of attention weights in the second pass ofanchor
layers.
5 CONCLUSION
We have presented Kascade as an efficient approximate Top-
kattention mechanism. It computes Top- kindices in a
subset of layers (anchorlayers), and uses them to compute
Top-kattention in the next fewreuselayers. To make this
scheme accurate and practically deployable across models,
we propose an automated way of choosing a good set of
anchorlayers, and make the algorithm head-aware. We also
implement efficient kernels for this scheme for both prefill
and decode, which requires sharing Top- kindices across a
tile of tokens. Kascade is able to achieve the best accuracy
on AIME-24, among other training free sparse attention
schemes, at a given sparsity ratio.
There are a few limitations of this work. First, this technique
requires a development set to compute theanchorlayers and
head mappings. It is possible that this biases the technique
towards the data in the development set. However, in the ex-
periments we have done, we have found the selections to be
robust to different datasets. Second, while Kascade reduces
attention latency, it doesn’t reduce the memory capacity re-
quirements for attention. The KV caches of long sequences
can be large and limit batch sizes which leads to reduced
performance. Some attention works, therefore, target both
capacity and latency benefits. Last, architectures which are
trained with sparsity like (Team et al., 2024; Agarwal et al.,
2025) will benefit less with this scheme.
REFERENCES
Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A.,
Arbus, E., Arora, R. K., Bai, Y ., Baker, B., Bao, H., et al.
gpt-oss-120b & gpt-oss-20b model card.arXiv preprint
arXiv:2508.10925, 2025.
Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y .,
Lebr ´on, F., and Sanghai, S. Gqa: Training generalized
multi-query transformer models from multi-head check-
points.arXiv preprint arXiv:2305.13245, 2023.
Beltagy, I., Peters, M. E., and Cohan, A. Long-former: The long-document transformer.arXiv preprint
arXiv:2004.05150, 2020.
Bhojanapalli, S., Chakrabarti, A., Veit, A., Lukasik, M.,
Jain, H., Liu, F., Chang, Y .-W., and Kumar, S. Leveraging
redundancy in attention with reuse transformers.arXiv
preprint arXiv:2110.06821, 2021.
Gao, Y ., Zeng, Z., Du, D., Cao, S., Zhou, P., Qi, J., Lai,
J., So, H. K.-H., Cao, T., Yang, F., et al. Seerattention:
Learning intrinsic sparse attention in your llms.arXiv
preprint arXiv:2410.13276, 2024.
Gao, Y ., Guo, S., Cao, S., Xia, Y ., Cheng, Y ., Wang, L., Ma,
L., Sun, Y ., Ye, T., Dong, L., et al. Seerattention-r: Sparse
attention adaptation for long reasoning.arXiv preprint
arXiv:2506.08889, 2025.
Gim, I., Chen, G., Lee, S.-s., Sarda, N., Khandelwal, A.,
and Zhong, L. Prompt cache: Modular attention reuse for
low-latency inference.Proceedings of Machine Learning
and Systems, 6:325–338, 2024.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783, 2024.
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,
Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: In-
centivizing reasoning capability in llms via reinforcement
learning.arXiv preprint arXiv:2501.12948, 2025.
Hao, J., Zhu, Y ., Wang, T., Yu, J., Xin, X., Zheng, B.,
Ren, Z., and Guo, S. OmniKV: Dynamic context selec-
tion for efficient long-context LLMs. InThe Thirteenth
International Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=ulCAPXYXfa.
He, S., Sun, G., Shen, Z., and Li, A. What matters in
transformers? not all attention is needed.arXiv preprint
arXiv:2406.15786, 2024.
Ho, X., Duong Nguyen, A.-K., Sugawara, S., and Aizawa,
A. Constructing a multi-hop QA dataset for compre-
hensive evaluation of reasoning steps. InProceedings
of the 28th International Conference on Computational
Linguistics, pp. 6609–6625, Barcelona, Spain (Online),
December 2020. International Committee on Compu-
tational Linguistics. URL https://www.aclweb.
org/anthology/2020.coling-main.580.
Jiang, H., Li, Y ., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han,
Z., Abdi, A. H., Li, D., Lin, C.-Y ., et al. Minference
1.0: Accelerating pre-filling for long-context llms via dy-
namic sparse attention.Advances in Neural Information
Processing Systems, 37:52481–52515, 2024.

Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference
Lu, S., Wang, H., Rong, Y ., Chen, Z., and Tang, Y . Turborag:
Accelerating retrieval-augmented generation with pre-
computed kv caches for chunked text.arXiv preprint
arXiv:2410.07590, 2024.
Ma, D., Wang, Y ., and Lan, T. Block-attention for efficient
prefilling. InThe Thirteenth International Conference
on Learning Representations, 2025. URL https://
openreview.net/forum?id=7zNYY1E2fq.
Ribar, L., Chelombiev, I., Hudlass-Galley, L., Blake, C.,
Luschi, C., and Orr, D. Sparq attention: bandwidth-
efficient llm inference. InForty-first International Con-
ference on Machine Learning, pp. 42558–42583, 2024.
Shah, J., Bikshandi, G., Zhang, Y ., Thakkar, V ., Ramani, P.,
and Dao, T. Flashattention-3: Fast and accurate attention
with asynchrony and low-precision.Advances in Neural
Information Processing Systems, 37:68658–68685, 2024.
Singhania, P., Singh, S., He, S., Feizi, S., and Bhatele,
A. Loki: Low-rank keys for efficient sparse attention.
Advances in Neural Information Processing Systems, 37:
16692–16723, 2024.
Tang, J., Zhao, Y ., Zhu, K., Xiao, G., Kasikci, B., and Han,
S. Quest: Query-aware sparsity for efficient long-context
llm inference.arXiv preprint arXiv:2406.10774, 2024.
Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju,
S., Pathak, S., Sifre, L., Rivi `ere, M., Kale, M. S., Love,
J., et al. Gemma: Open models based on gemini research
and technology.arXiv preprint arXiv:2403.08295, 2024.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Musique: Multihop questions via single-hop ques-
tion composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At-
tention is all you need.Advances in neural information
processing systems, 30, 2017.
Wang, L., Cheng, Y ., Shi, Y ., Tang, Z., Mo, Z., Xie, W.,
Ma, L., Xia, Y ., Xue, J., Yang, F., et al. Tilelang: A com-
posable tiled programming model for ai systems.arXiv
preprint arXiv:2504.17577, 2025.
Xiao, G., Tian, Y ., Chen, B., Han, S., and Lewis, M. Ef-
ficient streaming language models with attention sinks.
arXiv preprint arXiv:2309.17453, 2023.
Xiao, T., Li, Y ., Zhu, J., Yu, Z., and Liu, T. Sharing
attention weights for fast transformer.arXiv preprint
arXiv:1906.11024, 2019.Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025a.
Yang, L., Zhang, Z., Chen, Z., Li, Z., and Jia, Z. Tidalde-
code: Fast and accurate llm decoding with position per-
sistent sparse attention, 2024. URL https://arxiv.
org/abs/2410.05076.
Yang, L., Zhang, Z., Jain, A., Cao, S., Yuan, B., Chen,
Y ., Jia, Z., and Netravali, R. Less is more: Training-
free sparse attention with global locality for efficient rea-
soning, 2025b. URL https://arxiv.org/abs/
2508.07101.
Yang, S., Guo, J., Tang, H., Hu, Q., Xiao, G., Tang, J., Lin,
Y ., Liu, Z., Lu, Y ., and Han, S. Lserve: Efficient long-
sequence llm serving with unified sparse attention.arXiv
preprint arXiv:2502.14866, 2025c.
Yao, J., Li, H., Liu, Y ., Ray, S., Cheng, Y ., Zhang, Q., Du,
K., Lu, S., and Jiang, J. Cacheblend: Fast large language
model serving for rag with cached knowledge fusion. In
Proceedings of the Twentieth European Conference on
Computer Systems, pp. 94–109, 2025.
Ying, C., Ke, G., He, D., and Liu, T.-Y . Lazy-
former: Self attention with lazy update.arXiv preprint
arXiv:2102.12702, 2021.
Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Al-
berti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q.,
Yang, L., et al. Big bird: Transformers for longer se-
quences.Advances in neural information processing
systems, 33:17283–17297, 2020.
Zhang, Z., Sheng, Y ., Zhou, T., Chen, T., Zheng, L., Cai,
R., Song, Z., Tian, Y ., R ´e, C., Barrett, C., et al. H2o:
Heavy-hitter oracle for efficient generative inference of
large language models.Advances in Neural Information
Processing Systems, 36:34661–34710, 2023.