# LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation

**Authors**: Qixin Hu, Shuai Yang, Wei Huang, Song Han, Yukang Chen

**Published**: 2026-06-01 17:50:49

**PDF URL**: [https://arxiv.org/pdf/2606.02553v1](https://arxiv.org/pdf/2606.02553v1)

## Abstract
Autoregressive (AR) video diffusion enables variable-length synthesis, but long-horizon generation often suffers from accumulated errors and identity drift. For efficiency, existing methods commonly adopt sliding-window attention during generation. This creates an irreversible generation trajectory: once the active window accumulates appearance errors, subsequent generations can only condition on this degraded trajectory and drift further away. We address this limitation by formulating long video generation as a retrieval-augmented generation (RAG) problem. Rather than relying solely on the recent window, we treat previously generated latents as a dynamic, searchable history. We propose LongLive-RAG, a general retrieval framework for AR video generation. At each new block, LongLive-RAG uses a query embedding to retrieve relevant historical latents. This lightweight retrieval step adds only a small overhead relative to generation and lets the generator condition on non-local context instead of only the recent window. To make retrieval more discriminative, we introduce the Window Temporal Delta Loss that suppresses redundant local similarity and encourages embeddings to capture meaningful temporal changes. Together, these components help reduce error accumulation caused by sliding-window attention. Experiments across multiple AR backbones and generation lengths show improved long-video quality and the best average VBench-Long rank. To our knowledge, among open-ended AR long video generation methods, LongLive-RAG is the first to formulate self-generated latent history as content-addressable retrieval memory. Code is available at https://github.com/qixinhu11/LongLive-RAG.

## Full Text


<!-- PDF content starts -->

LongLive-RAG: A General Retrieval-Augmented
Framework for Long Video Generation
Qixin Hu1,2∗Shuai Yang1Wei Huang1Song Han1,3Yukang Chen1
1NVIDIA2USC3MIT
Project page:https://longlive-rag.github.io/
T = 0sT = 15sT = 30s
(2) Prompt: A rabbit wearing glasses sits on a chair, reading a newspaper. The rabbit has soft fur, big round eyes, and a small nose. It is sitting upright with its paws on its lap. The background shows a cozy living room with a …
(1) Prompt: A cute panda sitting on a stool, playing the guitar with its front paws, looking joyful and focused. The background is a green bamboo forest with dappled sunlight. Soft and whimsical illustration style. Close-up shot of the panda ...
Selection Trajectory (K=4)
Figure 1:LongLive-RAGlets a video generator look back at useful parts of the video it has already
generated. Left: for each new segment, LongLive-RAG actively searches the generated history and
retrieves relevant past context. Right: using this retrieved context helps long videos keep subjects and
scenes more stable and consistent over time.
Abstract
Autoregressive (AR) video diffusion enables variable-length synthesis, but long-
horizon generation often suffers from accumulated errors and identity drift. For
efficiency, existing methods commonly adopt sliding-window attention during gen-
eration. This creates an irreversible generation trajectory: once the active window
accumulates appearance errors, subsequent generations can only condition on this
degraded trajectory and drift further away. We address this limitation by formu-
∗Work done during an internship at NVIDIA.
Preprint.arXiv:2606.02553v1  [cs.CV]  1 Jun 2026

lating long video generation as a retrieval-augmented generation (RAG) problem.
Rather than relying solely on the recent window, we treat previously generated
latents as a dynamic, searchable history. We proposeLongLive-RAG, a general
retrieval framework for AR video generation. At each new block, LongLive-RAG
uses a query embedding to retrieve relevant historical latents. This lightweight
retrieval step adds only a small overhead relative to generation and lets the gen-
erator condition on non-local context instead of only the recent window. To
make retrieval more discriminative, we introduce the Window Temporal Delta
Loss that suppresses redundant local similarity and encourages embeddings to
capture meaningful temporal changes. Together, these components help reduce
error accumulation caused by sliding-window attention. Experiments across mul-
tiple AR backbones and generation lengths show improved long-video quality
and the best average VBench-Long rank. To our knowledge, among open-ended
AR long video generation methods, LongLive-RAG is the first to formulate self-
generated latent history as content-addressable retrieval memory. Code is available
at https://github.com/qixinhu11/LongLive-RAG.
1 Introduction
Recent video diffusion models can synthesize short clips with strong visual quality [ 48,57,58,26,11,
23,42,36,27,51]. Many applications, however, require longer videos in which subjects, backgrounds,
and scene layouts remain consistent across tens or hundreds of seconds [ 3,13,19,40,50,20,38]. AR
video diffusion is a practical formulation for this setting because it generates latent blocks causally
and can continue beyond a fixed clip length [ 5,55,21,66,32,49,25,18]. In practice, long AR
generation often suffers from error accumulation. To keep computation bounded, AR generators
usually keep only the most recent blocks as context and discard earlier blocks. Once these recent
blocks contain appearance drift, identity changes, or background artifacts, subsequent blocks may
condition on these errors and further propagate them [33, 12, 3].
Existing methods for long video generation address this problem by changing how history is kept.
Attention-sink methods retain fixed early tokens or frames as anchors [ 46,49,29,34]. Methods based
on positional extrapolation extend the usable temporal range of the model [ 39,65,52]. Methods based
on compressed-history tokens summarize older states into substitute tokens or recurrent memory
[53,59,24,62]. These methods can reduce error and improve stability, but they still have limitations.
Fixed anchors may not match the current content, positional extrapolation does not prevent error
accumulation once the visible context has drifted, and compressed-history tokens may lose critical
native visual details for subsequent generations.
Our observation is that long video generation needs a way to use generated states when they are
useful, instead of relying only on the recent window or a fixed summary of the past, as illustrated
in Figure 1. The generated history may contain earlier context that better preserves the subject
appearance, background layout, or scene structure needed by the current block. Retrieval provides
a direct way to use this history. Before generating a new block, the model can search previously
generated latents and bring back relevant historical context. When the recent window has drifted, this
retrieved context can provide a reference and reduce dependence on the corrupted local context.
Based on this observation, we proposeLongLive-RAG, a general retrieval-augmented framework
for AR video generation. LongLive-RAG stores previously generated latents and builds compact
embeddings for search. For each new block, an encoder maps the latest completed latent to a query
embedding. The method retrieves the top- Khistorical latents, combines them with the local latents,
and uses them as the generator’s attention context. The generator attends to the retrieved historical
latents and local latents rather than compressed-history tokens. This gives the generator access to
useful history outside the sliding window. It also keeps the base AR generator unchanged and adds
only lightweight retrieval overhead relative to transformer attention.
The retrieval embedding must be suitable for search. Adjacent video latents are often very similar,
so a reconstruction-only encoder can map many nearby blocks to almost identical embeddings. In
that case, top- Kretrieval may return blocks that add little beyond the local window. We introduce
a Window Temporal Delta Loss to reduce excessive similarity among nearby latents, and add a
2

smoothing term to keep embeddings stable over time. Together, these losses make retrieval more
discriminative while preserving visual information.
Our main contribution is to formulate open-ended AR video generation as retrieval over self-generated
latents, together with a lightweight retrieval framework and an embedding objective that selects useful
non-local context. LongLive-RAG retrieves useful past latents with a lightweight encoder and uses
them as extra attention context. We compare it with two alternatives for addressing sliding-window
limits: ∞-RoPE for positional extrapolation and Deep Forcing for compressed-history tokens. Across
three AR backbones, Causal-Forcing, Self-Forcing, and LongLive, and three generation lengths, 30s,
60s, and 120s, LongLive-RAG achieves the best average VBench-Long rank and improves subject
consistency, background consistency, motion smoothness, and imaging quality. Qualitative results
and ablations further support these results.
2 Related Work
This section briefly reviews related work; Appendix C gives the full discussion.
AR long video generation.AR video diffusion emits frames or latents causally for streaming and
variable-length synthesis [ 5,55,21,32,49,25,18,41,6,16,60]. Recent methods improve local
denoising, self-generated context, or streaming attention [ 55,21,9,66,8,32,49]. They mainly
improve the quality or causality of each local rollout step; LongLive-RAG is orthogonal to these
advances and keeps the generator unchanged.
Context visibility and memory.Long video generation systems expose history through sliding
windows, fixed anchors, positional extrapolation, or compressed-history tokens [ 46,49,29,34,39,
65,52,53,8,59,24,62,61]. These strategies differ in how much history is kept and how it is stored;
LongLive-RAG instead searches the generated history and brings back context for attention.
Retrieval memory in video generation.Retrieval is natural in world models when geometry,
camera pose, or scene coordinates are available [ 20,13,30,56,47,40,50,45]. Open-ended text-to-
video generation does not provide such explicit retrieval cues by default, so useful history must be
found from the generated content itself; LongLive-RAG searches the generated history and uses the
matched context during rollout.
3 Method
LongLive-RAG augments AR video diffusion with retrieval over self-generated latents while keeping
the base generator fixed. Section 3.1 defines the sliding-window and LongLive-RAG context assembly
rules. Section 3.2 describes the paired embedding/context history banks and the retrieval rule.
Section 3.3 explains retrieval embedding space training. Section 3.4 summarizes the inference.
3.1 AR Context Assembly
Sliding-window context.At block t, an AR video diffusion model denoises the current latent
representation while attending to context from previously completed blocks [ 55,21,32,49]. Standard
inference typically keeps a bounded sliding-window context
Asw
t= [C sink∥Cloc],(1)
where Csinkstores optional fixed early context and Clocstores context from recent completed blocks
[46,49,59]. Once earlier context is dropped, the generator cannot use it again; if the recent context
has drifted, the next block is still generated from this drifted context.
LongLive-RAG context.LongLive-RAG does not update the base generator or add trainable layers
inside the denoising backbone. Instead, it uses an offline-trained retrieval encoder and adds relevant
historical context entries directly to the attention context:
Arag
t= [C sink∥Mt∥Cloc].(2)
HereMtdenotes the retrieved non-local context entries for block t. The current block is then
denoised with the retrieved history and local context. Importantly, LongLive-RAG does not add
attention layers or change the denoising rule.
3

1. Autoregressive Video Generation PipelineTPromptsRetrieved Latents(Adaptive Selection)Recent Latents(Sliding Window)Noise TokensContext Tokens2. Adaptive Selection via Latent Similarity
Compressed Retrieval Embedding Space (1024-dim)
3. Error Trajectory: Without vs. With Latent RetrievalWithout Retrievaltt + 1t + 2t + H
Error Accumulates Over TimeWith Retrieval (Ours)tt + 1...t + H
Retrieval Provides Implicit Error Correction at Each StepRetrieval pulls latent to video manifoldt + 2...Similarity Computation(e.g., Cosine Similarity)Top-K SelectionAutoregressive Video Diffusion TransformerDenoised Tokens
Accurate Latent(on Manifold)Drifted Latent(off Manifold)GenerationTrajectoryVideo Manifold(good region)Generation Latent[C, H, W]Latent EncoderQuery EmbeddingAll Past LatentsRetrieval Embed. Pool
Sink......
Latent Encoder
Sliding WindowAdaptive SelectionAttention SinkFigure 2: Overview of LongLive-RAG. (1) The AR video diffusion transformer produces a generated
latent and assembles context tokens. (2) A latent encoder maps the latent to a compact query
embedding and searches the retrieval pool. (3) Retrieved latents provide context; when drift enters
the trajectory and errors accumulate over time, retrieval provides implicit correction at each step.
3.2 Indexing the History
Compact embeddings for search.Let ˆxt∈RC×H×Wbe the clean latent at block t. This
latent remains spatially dense because the generator needs it to preserve visual quality and fine
details during denoising, as in WAN [ 42]. Retrieval has a different requirement: it only needs a
discriminative key for finding useful history, not the full representation used for synthesis. LongLive-
RAG therefore performs search in a compressed retrieval embedding space instead of directly
comparing all historical latents with spatial shape [C, H, W] . For a completed block i, an encoder
produces a 1024-dimensional embedding vi, and the base backbone provides the corresponding
context entry Ci. The historical search bank is Hv={(i, v i)}, paired with a context bank Hctx=
{(i,C i)}. The embeddings are used for retrieval, while the generator attends to the matched context.
We train a latent autoencoder with encoderE ψand decoderD ψusing reconstruction loss,
Lrec=∥D ψ(Eψ(ˆxt))−ˆx t∥2
2,(3)
and use vt=E ψ(ˆxt)as the retrieval embedding. Here Eψis the encoder and vt∈R1024lies in the
compact search space. The encoder is trained offline on clean latents with the base generator fixed;
the decoder is used only to shape this space and is not inserted into the generator at inference. After
block tis completed, LongLive-RAG computes vtand associates it with the block context Ct. Entries
remain in the rolling local cache while they are recent; when they leave the local window, their paired
(vi,Ci)entries are added toH vandH ctx.
Similarity retrieval.For block t, LongLive-RAG performs retrieval once before denoising and
reuses the selected context for all Ndenoising steps. The query is the embedding of the most recent
completed latent, i.e., vq=vt−1after the previous block has been finalized. LongLive-RAG ranks
stored embeddings with cosine similarity,
ai= cos(v q, vi), i∈ P t,(4)
where Ptskips the most recent history entries to avoid retrieving near-duplicate recent context. The
top-KindicesI t= Top-K({a i}i∈P t)select matched entries fromH ctx:
Mt= [C i]i∈I t.(5)
These retrieved context entries are combined with the local cache in the current attention context.
4

Avg-Pool
+ Autoencoder
+ 𝐿!"#$"%&'+ 𝐿!())*+latent 𝑖latent 𝑗
AE only
AE + 𝐿!"#$"%&'
AE + 𝐿()*+),-.+ 𝐿/011-2
Figure 3: Embedding-space analysis. Darker green indicates a higher cosine similarity between
the embeddings. Relying on reconstruction alone preserves content but leaves temporally nearby
latents overly similar; the temporal delta separates redundant local states, and smoothing effectively
stabilizes the embedding trajectory along the entire generation trajectory.
Context handling.Each entry in Hctxfollows the same interface used by the local window.
LongLive-RAG does not introduce new attention layers; it only changes which historical context
entries are exposed to attention at each block. Retrieved entries are therefore handled in the same
way as local context entries inside the original generator.
Table 1: 120s retrieval overhead.
Component ms/block Total (ms)
Latent encoding 3.96 480
Top-Ksearch 0.08 10
Total 4.08 490Retrieval overhead.Table 1 reports the addi-
tional runtime introduced by LongLive-RAG re-
trieval over a 120s rollout. LongLive-RAG per-
forms retrieval once per AR block and adds 4.08
ms per block, totaling 490 ms of retrieval overhead:
480 ms from latent encoding and 10 ms from top- K
search. The overhead is dominated by encoding;
similarity search is negligible at this scale. For long
video generation, this cost is small compared with the diffusion rollout itself.
3.3 Learning the Embedding Space
Why reconstruction is not enough.The search space has a different requirement from the V AE
latent space used by video generators such as WAN [ 42]. For synthesis, adjacent latents should change
smoothly and preserve dense visual detail. For retrieval, however, the embedding must also decide
which history is worth bringing back into attention. If the embedding only optimizes reconstruction, it
tends to preserve the local continuity of video too well: nearby latents become almost interchangeable
search keys. Figure 3 illustrates this behavior. With AE-only training, the similarity map contains a
broad high-similarity band around the diagonal, meaning that many neighboring blocks look equally
valid to the retriever. In inference, such an index can waste the top- Kbudget on locally redundant
states, adding little beyond the recent cache.
This motivates an anti-collapse term at the temporal scale where redundancy is most common. We
do not want to repel all frames from each other, since the same subject or scene may legitimately
reappear later. Instead, we only discourage excessive similarity within a short window, where frames
are likely to be visually redundant and already covered by the local cache. The loss is therefore local
and margin-based: pairs below the margin are not pushed apart, and long-range repetitions are not
treated as negatives. We first define a pairwise temporal delta penalty,
L∆(vt, vref) =λ ∆max(0,cos(v t, vref)−m),(6)
which penalizes local pairs whose cosine similarity exceeds a margin m. The Window Temporal
Delta Loss averages this penalty over a local temporal window. For a sequence of length Tand
window sizew, let¯w= min(w, T−1).
LSeqDelta =1
¯w¯wX
τ=11
T−τTX
t=τ+1L∆(vt, vt−τ).(7)
5

Stable embedding trajectory.Temporal separation alone is still insufficient. If the embedding
moves too sharply from one block to the next, the top- Kset can change for reasons unrelated to
meaningful content changes, which may make the retrieved context unstable. The right behavior
is visible in Figure 3: after local redundancy is reduced, the embedding trajectory should remain
organized rather than noisy. We therefore add a second-order smoothness penalty
LSmooth =λsmooth1
T−2TX
t=3∥vt−2v t−1+vt−2∥2,(8)
Combining these terms gives the final training objective:
LTotal =L rec+LSeqDelta +LSmooth .(9)
The three terms play distinct roles. Reconstruction keeps the embedding tied to visual content;
temporal delta makes nearby redundant states less likely to dominate top- Kretrieval; and smoothing
keeps the embedding trajectory stable across the generation trajectory. This is the design principle
behind Figure 3: LongLive-RAG does not seek a generic compressed representation, but a search
geometry that matches the needs of non-local context selection.
3.4 Inference
Algorithm 1 LongLive-RAG inference
Require:G θ, Eψ;T, N, L, K, R
State:C sink,Cloc,Hctx,Hv
1Csink,Cloc,Hctx,Hv← ∅
2forblockt= 1, . . . , Tdo
3// Step 1: query searches pool
4I t← ∅
5if|H v|> Rthen
6v q←vt−1 ▷latest embedding
7I t←top-Kmatches inH voutside recentR
8end if
9// Step 2: denoise latentt
10M t←[H ctx[i]]i∈I t
11A t←[C sink∥Mt∥Cloc]
12forsteps= 1, . . . , Ndo
13xs+1
t←G θ(xs
t;At)
14end for
15// Step 3: encoder updates pool
16ˆx t←xN+1
t;vt←E ψ(ˆxt)
17 Append context andv ttoCloc
18if|C loc|> Lthen
19 Add oldest entry to(H ctx,Hv)
20 Keep the latestLentries inC loc
21end if
22end forAlgorithm 1 gives the inference procedure
with a reduced notation set. Here Gθis
the frozen AR generator and Eψis the en-
coder. Tis the number of latents, Nis the
number of denoising steps per block, Lis
the local-window size, Kis the retrieval
budget, and Ris the recency guard that
prevents the search from selecting near-
duplicate recent context. The state con-
sists of an optional sink cache Csink, a
rolling local cache Clocfor the latest con-
text, the pool Hv, and a paired historical
context pool Hctx. Entry iofHvis the
compact search key for entry iofHctx. At
block t,vq=vt−1is the query from the
latest completed latent, Itis the selected
top-Khistory indices, Mtis the matched
context, and At= [C sink∥Mt∥Cloc]is the
attention context. When no eligible his-
torical entry exists, Itis empty and the
method reduces to the base sink-plus-local
context. The latent xs
tdenotes block tat
denoising step s,ˆxt=xN+1
t is the clean
completed latent, and vt=E ψ(ˆxt)is the
embedding stored with the completed lo-
cal entry before later offloading to the his-
torical pool. The notation | · |denotes the number of entries in a bank, and ∥denotes concatenation
along the context dimension. As illustrated in Figure 2(3), this gives later blocks access to content-
relevant non-local context while retaining the original AR generator.
4 Experiments
We evaluate whether LongLive-RAG improves long-video quality across three AR backbones and
three generation lengths, 30s, 60s, and 120s, and whether learned retrieval contributes to the gains.
4.1 Setup
Implementation details.We evaluate LongLive-RAG on three AR backbones: Causal-Forcing [ 66],
Self-Forcing [ 21], and LongLive [ 49]. For each backbone, we compare the base model, ∞-RoPE [ 52],
6

Self Forcing
+ ∞-RoPE
+ Deep Forcing
+ LongLive-RAGt = 1st = 5st = 15st = 30s
t = 1st = 5st = 15st = 30s
t = 1st = 5st = 15st = 30s
t = 1st = 5st = 15st = 30s
Causal Forcing
+ ∞-RoPE
+ Deep Forcing
+ LongLive-RAGt = 1st = 10st = 20st = 30s
t = 1st = 10st = 20st = 30s
t = 1st = 10st = 20st = 30s
t = 1st = 10st = 20st = 30sFigure 4: Qualitative comparison on 30s generations. Rows compare the base model, ∞-RoPE,
Deep Forcing, and LongLive-RAG; columns show later timestamps. The displayed baselines show
color shifts, duplicated subjects, or background artifacts.
Deep Forcing [ 53], and LongLive-RAG. ∞-RoPE represents positional extrapolation, while Deep
Forcing represents compressed-history tokens. All inference runs use the same base sampling settings,
sink size 1, and total attention window size 12. LongLive-RAG uses K= 6 , giving the context
layout [1 sink∥6 retrieved context∥5 local window] . Under this fixed context budget, the compared
methods differ only in how the available slots are filled: the base model and ∞-RoPE use sink-plus-
local context, while Deep Forcing uses compressed-history context following its inference rule. The
reported inference and evaluation runs for the quantitative tables take about one week on 6 NVIDIA
RTX A6000 GPUs, excluding retrieval-encoder data construction and autoencoder training. More
implementation details are provided in Appendix E.
7

Table 2: VBench-Long results for 30s, 60s, and 120s generation. Each block fixes a base model
and compares it with ∞-RoPE [ 52], Deep Forcing [ 53], and LongLive-RAG. Bold/underline mark
best/second-best values; Avg. Rank is averaged over the six metrics. Avg. Rank is computed from
unrounded metric values.
MethodSubject
Consistency↑Background
Consistency↑Motion
Smoothness↑Dynamic
Degree↑Aesthetic
Quality↑Imaging
Quality↑Avg.
Rank↓
30s generation
Self-Forcing [21] 96.21 95.39 98.39 52.03 56.69 63.31 3.17
+∞-RoPE [52] 97.32 96.38 98.59 46.82 56.78 63.93 2.00
+ Deep Forcing [53] 97.04 96.02 98.57 38.85 56.44 61.91 3.50
+ LongLive-RAG (Ours) 97.57 96.56 98.7642.2457.17 65.43 1.33
LongLive [49] 97.35 96.15 98.70 44.74 59.38 68.15 2.67
+∞-RoPE [52] 97.27 96.19 98.6848.1858.69 67.99 3.17
+ Deep Forcing [53] 97.52 96.43 98.8241.46 59.00 67.61 2.50
+ LongLive-RAG (Ours) 97.5396.39 98.77 44.84 59.24 68.42 1.67
Causal-Forcing [66] 94.60 94.68 96.56 73.96 54.58 65.53 3.00
+∞-RoPE [52] 93.93 94.11 96.2190.8355.42 68.26 2.33
+ Deep Forcing [53] 93.52 93.86 95.84 84.79 55.03 66.07 3.33
+ LongLive-RAG (Ours) 95.43 94.79 97.1682.2957.31 70.07 1.33
60s generation
Self-Forcing [21] 95.84 95.27 98.20 51.72 56.05 62.22 3.33
+∞-RoPE [52] 97.24 96.24 98.58 46.64 56.09 63.28 2.17
+ Deep Forcing [53] 96.08 95.38 98.24 41.44 56.68 60.81 3.17
+ LongLive-RAG (Ours) 97.60 96.51 98.7044.6957.19 64.97 1.33
LongLive [49] 97.13 95.89 98.61 44.56 58.17 67.56 2.83
+∞-RoPE [52] 97.00 95.85 98.5353.3657.48 66.94 3.33
+ Deep Forcing [53] 97.17 96.04 98.7345.13 57.48 67.27 2.50
+ LongLive-RAG (Ours) 97.32 96.0898.62 49.90 58.30 67.79 1.33
Causal-Forcing [66] 93.52 94.12 95.74 72.32 51.24 62.30 3.83
+∞-RoPE [52] 93.81 93.78 96.0992.4754.42 67.50 2.50
+ Deep Forcing [53] 94.27 94.18 96.6278.59 52.12 64.25 2.33
+ LongLive-RAG (Ours) 94.29 94.2496.48 88.20 54.95 68.16 1.33
120s generation
Self-Forcing [21] 96.12 95.32 98.27 43.39 55.64 61.57 3.33
+∞-RoPE [52] 97.15 96.09 98.5546.2955.11 61.81 2.33
+ Deep Forcing [53] 96.9296.83 98.9715.23 52.84 57.93 2.83
+ LongLive-RAG (Ours) 97.6496.40 98.75 44.10 56.30 64.16 1.50
LongLive [49] 96.93 95.64 98.58 47.12 57.90 66.95 2.50
+∞-RoPE [52] 96.81 95.65 98.4853.5956.73 66.19 3.17
+ Deep Forcing [53] 97.17 95.72 98.57 46.03 56.98 66.03 3.00
+ LongLive-RAG (Ours) 97.22 95.88 98.6250.25 57.57 66.95 1.33
Causal-Forcing [66] 92.98 94.66 95.41 63.79 47.31 58.23 3.33
+∞-RoPE [52] 93.45 93.56 95.95 93.4453.47 66.81 2.17
+ Deep Forcing [53] 93.19 94.04 95.60 76.45 46.86 61.58 3.17
+ LongLive-RAG (Ours) 94.3894.08 96.5690.21 54.82 68.23 1.33
Evaluation protocol.We use all 128 prompts from MovieGenBench [ 36]. Following Self-
Forcing [ 21], all prompts are refined using Qwen2.5-7B-Instruct [ 37]. Table 2 reports VBench-Long
metrics [ 22] for 30s, 60s, and 120s generations. Table 3 reports ablations on 30s Causal-Forcing and
auxiliary VLM scores. The VLM prompt is provided in Appendix F.
4.2 Results
Qualitative results.Figure 4 compares LongLive-RAG with alternative history-handling methods,
including ∞-RoPE and Deep Forcing, on 30s rollouts. LongLive-RAG better preserves subject and
background appearance, while other methods can show appearance shifts, duplicated subjects, or
8

(a) Embedding space.
MethodSubj.
Cons.↑Bg.
Cons.↑Motion
Smooth.↑Imaging
Quality↑
Random retrieval 94.54 94.32 96.81 68.79
Avg-pool desc. 94.77 94.49 96.76 69.11
AE only 94.82 94.49 96.87 69.48
AE + SeqDelta 94.76 94.54 97.04 69.14
Ours 95.43 94.79 97.16 70.07
(b) Auxiliary VLM evaluation. (Max: 5)
Variant Causal Forc. Self Forc. LongLive
Base 2.60 3.50 4.65
+∞-RoPE 4.10 4.15 4.35
+ Deep Forcing 3.55 4.35 4.70
+ LongLive-RAG 4.70 4.45 4.75(c) Retrieval budget.
2 4 6 8
Retrieval Budget K90929496Score
94.4394.7895.43
90.56Subject Consistency
2 4 6 8
Retrieval Budget K92.893.694.495.2Score
94.2894.4594.79
93.07Background Consistency
2 4 6 8
Retrieval Budget K96.597.097.5Score
 96.5896.8597.16
96.53Motion Smoothness
2 4 6 8
Retrieval Budget K606570Score
69.14 69.2170.07
60.02Imaging Quality
Table 3: Ablation study and auxiliary VLM evaluation. Left: embedding-space ablation and auxiliary
VLM scores on 30s generations. Right: retrieval-budget trends under the same total attention budget.
color artifacts. These examples support the same trend measured by Table 2: retrieving selected
non-local latents provides useful context beyond the fixed local window or compressed history.
Quantitative results.Table 2 shows that LongLive-RAG obtains the lowest Avg. Rank across
all base-model and duration blocks. The gains are consistent across 30s, 60s, and 120s rollouts,
suggesting that retrieval remains useful as generation length increases. Improvements are most visible
on quality and consistency metrics, including subject consistency, background consistency, motion
smoothness, and imaging quality. These results indicate that selected non-local latents help the
generator preserve appearance and scene structure beyond the recent window.
Auxiliary VLM evaluation.Table 3 also reports VLM exposure scores on 30s generations as an
auxiliary check. The scores show the same overall trend as the VBench-Long results.
4.3 Ablations
Embedding space.Table 3 first tests the retrieval embedding space. Learned embeddings outper-
form random retrieval, average-pooled descriptors, and reconstruction-only embeddings. Adding
LSeqDelta improves background consistency and motion smoothness over AE-only, while the full
objective gives the strongest overall results across the retained metrics. These trends match the
design in Section 3.3: retrieval benefits from a learned space that is both content-preserving and
discriminative for non-local context.
Retrieval budget.With the embedding fixed, the retrieval-budget plots in Table 3 show that K= 6
gives the strongest consistency and imaging quality under the same total attention budget. This
setting allocates enough slots to retrieved non-local context while preserving local context for smooth
continuation. It therefore provides a practical balance between long-range reference and video
continuity. Appendix G provides additional examples.
5 Conclusion
We presented LongLive-RAG, a simple way for an AR video generator to look back at useful parts of
the video it has already generated. Instead of relying only on the recent window, LongLive-RAG
searches the generated history and brings back relevant context for the next block. We train the
retrieval embeddings with reconstruction, Window Temporal Delta Loss, and smoothing so that the
search keys preserve visual content, avoid redundant nearby matches, and remain stable over time.
Experiments across multiple AR backbones and video lengths show that this improves long-video
quality with little retrieval overhead. Limitations are discussed in Appendix A.
9

References
[1]bloc97. NTK-aware scaled RoPE allows LLaMA models to have extended (8k+) context
size without any fine-tuning and minimal perplexity degradation. Reddit post, 2023. URL
https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/.
[2]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al.
Improving language models by retrieving from trillions of tokens. InProceedings of the 39th
International Conference on Machine Learning, pages 2206–2240, 2022.
[3]Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr,
Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators.
OpenAI Blog, 1(8):1, 2024.
[4]Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski,
and Armand Joulin. Emerging properties in self-supervised vision transformers. InProceedings
of the IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021.
[5]Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent
Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion.Advances in
Neural Information Processing Systems, 37:24081–24125, 2024.
[6]Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao
Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film
generative model.arXiv preprint arXiv:2504.13074, 2025.
[7]Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context
window of large language models via positional interpolation.arXiv preprint arXiv:2306.15595,
2023.
[8]Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu
Chen. Context forcing: Consistent autoregressive video generation with long context.arXiv
preprint arXiv:2602.06028, 2026.
[9]Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and
Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation.arXiv
preprint arXiv:2510.02283, 2025.
[10] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban,
and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour.arXiv preprint
arXiv:2601.16914, 2026.
[11] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang
Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector
quantization. InThe Thirteenth International Conference on Learning Representations, 2025.
URLhttps://openreview.net/forum?id=JE9tCwe3lp.
[12] Mohamed Elmoghany, Ryan Rossi, Seunghyun Yoon, Subhojyoti Mukherjee, Eslam Mohamed
Bakr, Puneet Mathur, Gang Wu, Viet Dac Lai, Nedim Lipka, Ruiyi Zhang, et al. A survey
on long-video storytelling generation: architectures, consistency, and cinematic quality. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7023–7035,
2025.
[13] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun
Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with
real-time moving control.arXiv preprint arXiv:2412.03568, 2024.
[14] Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, and Poulami Das. Dialogue
without limits: Constant-sized kv caches for extended responses in llms.arXiv preprint
arXiv:2503.00979, 2025.
[15] Google DeepMind. Gemini 3.1 pro model card, 2026. URL https://deepmind.google/
models/model-cards/gemini-3-1-pro/.
10

[16] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling
with next-frame prediction.arXiv preprint arXiv:2503.19325, 2025.
[17] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
Retrieval-augmented language model pre-training. InProceedings of the 37th International
Conference on Machine Learning, pages 3929–3938, 2020.
[18] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tade-
vosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent,
dynamic, and extendable long video generation from text. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages 2568–2577, 2025.
[19] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy,
Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with
long-horizon memory.arXiv preprint arXiv:2512.04040, 2025.
[20] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie
Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving.
arXiv preprint arXiv:2309.17080, 2023.
[21] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing:
Bridging the train-test gap in autoregressive video diffusion. InThe Thirty-ninth Annual
Conference on Neural Information Processing Systems, 2025. URL https://openreview.
net/forum?id=mSiN7i0BYH.
[22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang,
Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark
suite for video generative models. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21807–21818, 2024.
[23] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang,
Yang Song, Yadong MU, and Zhouchen Lin. Pyramidal flow matching for efficient video
generative modeling. InThe Thirteenth International Conference on Learning Representations,
2025. URLhttps://openreview.net/forum?id=66NzcRQuOq.
[24] Youngrae Kim, Qixin Hu, C-C Jay Kuo, and Peter A Beerel. Memrope: Training-free infinite
video generation via evolving memory tokens.arXiv preprint arXiv:2603.12513, 2026.
[25] Akio Kodaira, Tingbo Hou, Ji Hou, Markos Georgopoulos, Felix Juefei-Xu, Masayoshi
Tomizuka, and Yue Zhao. Streamdit: Real-time streaming text-to-video generation.arXiv
preprint arXiv:2507.03745, 2025.
[26] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel
Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language
model for zero-shot video generation.arXiv preprint arXiv:2312.14125, 2023.
[27] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin
Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video
generative models.arXiv preprint arXiv:2412.03603, 2024.
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. InAdvances
in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.
[29] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging
limited-horizon training and open-ended testing in autoregressive video diffusion.arXiv preprint
arXiv:2602.07775, 2026.
[30] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive
video scene generation with surfel-indexed view memory. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pages 25690–25699, 2025.
11

[31] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye,
Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for
before generation.Advances in Neural Information Processing Systems, 37:22947–22970, 2024.
[32] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive
long video diffusion in real time.arXiv preprint arXiv:2509.25161, 2025.
[33] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue
Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations,
and opportunities of large vision models.arXiv preprint arXiv:2402.17177, 2024.
[34] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng
Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video
generation with rewarded distribution matching distillation.arXiv preprint arXiv:2512.04678,
2025.
[35] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context
window extension of large language models.arXiv preprint arXiv:2309.00071, 2023.
[36] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv
Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media
foundation models.arXiv preprint arXiv:2410.13720, 2024.
[37] Qwen Team. Qwen2.5 technical report.arXiv preprint arXiv:2412.15115, 2024.
[38] Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour,
Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, et al. Cosmos-drive-dreams:
Scalable synthetic driving data generation with world foundation models.arXiv preprint
arXiv:2506.09042, 2025.
[39] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer:
Enhanced transformer with rotary position embedding.Neurocomputing, 568:127063, 2024.
[40] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong
Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric
consistency for real-time interactive world modeling.arXiv preprint arXiv:2512.14614, 2025.
[41] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning
Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale.arXiv
preprint arXiv:2505.13211, 2025.
[42] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu,
Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative
models.arXiv preprint arXiv:2503.20314, 2025.
[43] Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan Tao, Zhihong Zhu, Xin Wang, Siqi Luo,
Jing Xiong, and Mi Zhang. D2o: Dynamic discriminative operations for efficient generative
inference of large language models.arXiv preprint arXiv:2406.13035, 2, 2024.
[44] Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia Zhang. Model tells you where to merge:
Adaptive kv cache merging for llms on long-context tasks.arXiv preprint arXiv:2407.08454,
2024.
[45] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xun-
liang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al. Infinite-world: Scaling interactive
world models to 1000-frame horizons via pose-free hierarchical memory.arXiv preprint
arXiv:2602.02393, 2026.
[46] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. InThe Twelfth International Conference on Learning
Representations, 2024. URLhttps://openreview.net/forum?id=NG7sS51zVF.
[47] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xin-
gang Pan. Worldmem: Long-term consistent world simulation with memory.arXiv preprint
arXiv:2504.12369, 2025.
12

[48] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation
using vq-vae and transformers.arXiv preprint arXiv:2104.10157, 2021.
[49] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang
Li, Enze Xie, Ying-Cong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time
interactive long video generation. InThe Fourteenth International Conference on Learning
Representations, 2026. URLhttps://openreview.net/forum?id=nCAODkpsPJ.
[50] Ying Yang, Zhengyao Lv, Tianlin Pan, Haofan Wang, Binxin Yang, Hubery Yin, Chen Li,
Ziwei Liu, and Chenyang Si. Stableworld: Towards stable and consistent long interactive video
generation.arXiv preprint arXiv:2601.15281, 2026.
[51] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming
Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion
models with an expert transformer.arXiv preprint arXiv:2408.06072, 2024.
[52] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag.
Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-
rollout.arXiv preprint arXiv:2511.20649, 2025.
[53] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep
forcing: Training-free long video generation with deep sink and participative compression.
arXiv preprint arXiv:2512.05081, 2025.
[54] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T
Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
6613–6623, 2024.
[55] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman,
and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
22963–22974, 2025.
[56] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and
Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory
retrieval. InProceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.
[57] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G
Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video
transformer. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10459–10469, 2023.
[58] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David
Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong,
Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion
- tokenizer is key to visual generation. InThe Twelfth International Conference on Learning
Representations, 2024. URLhttps://openreview.net/forum?id=gzqrANCF4g.
[59] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin
Ma, Yuewen Ma, Zhongrui Wang, et al. Videossm: Autoregressive long video generation with
hybrid state-space memory.arXiv preprint arXiv:2512.04519, 2025.
[60] Shangjin Zhai, Zhichao Ye, Jialin Liu, Weijian Xie, Jiaqi Hu, Zhen Peng, Hua Xue, Danpeng
Chen, Xiaomeng Wang, Lei Yang, et al. Stargen: A spatiotemporal autoregression framework
with video diffusion model for scalable and controllable scene generation. InProceedings of
the Computer Vision and Pattern Recognition Conference, pages 26822–26833, 2025.
[61] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction
models for video generation.arXiv e-prints, pages arXiv–2504, 2025.
[62] Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Beijia Lu, Anyi Rao, Song Han, Gordon
Wetzstein, and Maneesh Agrawala. Pretraining frame preservation in autoregressive video
memory compression.arXiv preprint arXiv:2512.23851, 2025.
13

[63] Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, and Rongrong
Ji. Cam: Cache merging for memory-efficient llms inference. InForty-first international
conference on machine learning, 2024.
[64] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao
Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient
generative inference of large language models.Advances in Neural Information Processing
Systems, 36:34661–34710, 2023.
[65] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. RIFLEx:
A free lunch for length extrapolation in video diffusion transformers. InForty-second Inter-
national Conference on Machine Learning, 2025. URLhttps://openreview.net/forum?
id=v3B79m7t8Z.
[66] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing:
Autoregressive diffusion distillation done right for high-quality real-time interactive video
generation.arXiv preprint arXiv:2602.02214, 2026.
14

A Limitations
LongLive-RAG builds on a frozen base checkpoint. It improves how the model selects and reuses
generated history, but it does not change the generator itself. As a result, the final video quality is still
bounded by the capability of the base AR model.
B Broader Impacts
LongLive-RAG is a research method for improving long-horizon consistency in AR text-to-video
generation. It may support positive applications that require stable long video synthesis, such as
creative tools, simulation, and interactive content generation. At the same time, improvements in
long-video quality can inherit the misuse risks of the underlying video generator, including deceptive
or misleading generated media. LongLive-RAG does not introduce a new base generator or a new data
source; it changes how a frozen AR generator selects and reuses its generated history. Responsible
use should therefore follow the license, usage terms, and safety practices of the underlying video
generation models.
C Additional Related Work Discussion
Video generation and AR rollout.Modern video generators include token-based models, diffusion
transformers, and large-scale text-to-video systems [ 48,57,58,26,11,23,42,36,27,51,41,6].
Many of these models denoise a fixed clip jointly, which ties computation to clip length. AR video
generation instead emits frames or latent blocks causally, enabling streaming, interactive prompting,
and variable-length output [ 5,55,21,32,49,25,18,16,60]. LongLive-RAG is designed for this AR
setting, where the generated trajectory itself becomes the source of future context.
Causal adaptation and self-generated context.Several methods convert or adapt pretrained
video diffusion models into causal generators through distillation, causal teacher–student training,
or self-generated context exposure [ 54,55,21,9,66,8,32,49,10]. These techniques reduce
exposure bias and improve each local denoising step under causal rollout. LongLive-RAG addresses a
complementary axis: given a causal generator and its self-generated trajectory, how should non-local
context be selected at inference time? It leaves the denoiser unchanged and modifies only the context
assembly policy.
Context windows, anchors, and positional extrapolation.The simplest scalable policy is a
sliding window, which bounds memory by keeping only recent context [ 55,21,32,49]. Attention
sinks and anchor-based variants preserve early tokens or frames to improve stability, but the chosen
anchors are not necessarily the content needed by a later query [ 46,49,29,34]. A related line extends
positional encodings so that models can operate beyond their trained length, including RoPE-based
positional extrapolation and video-specific variants [ 39,7,35,1,65,52]. These methods increase the
feasible context horizon or stabilize attention, but their selection rule is still primarily position-based.
LongLive-RAG instead uses content-addressed retrieval to decide which historical entries are exposed
to attention.
Compressed-history tokens and recurrent history.Methods based on compressed-history tokens
and cache management summarize, select, or evict long histories through substitute tokens, recurrent
states, compact sequence representations, or KV-cache policies [ 64,31,43,14,53,59,24,62,61,
44,63]. Such designs are attractive because they decouple memory cost from the full history length.
Their limitation for generation is that the model often attends to the summary itself rather than to
the original context produced by the generator. If the summary drops a rare object, identity cue, or
background detail, a later denoising step cannot recover it directly. LongLive-RAG also uses compact
representations, but only as search keys; after retrieval, the generator receives matched context entries
in the backbone’s native cache format. This separates the compression needed for indexing from the
information used for denoising.
Retrieval-augmented generation.Retrieval-augmented generation retrieves external evidence or
memory and conditions a generator on the retrieved content [ 28,17,2]. These methods are typically
15

used with a fixed text corpus or database to improve factual grounding or long-context access.
LongLive-RAG uses the same high-level retrieval principle, but the retrieval source is different: it
searches self-generated video latents produced during the current AR rollout. The retrieved items are
also inserted as native generator context rather than as external text evidence. This makes retrieval a
context-selection mechanism for long video generation.
Retrieval memory in video generation.Retrieval memory has been explored in world models
and embodied video settings where explicit structure is available. Memory can be organized by
camera pose, 3D geometry, scene coordinates, or field-of-view overlap, making it possible to identify
relevant observations in a physically grounded way [ 20,13,30,56,47,40,50,45]. Open-ended
text-to-video generation does not provide such explicit retrieval cues by default: the camera can move
freely, objects can reappear after occlusion, and no pose graph indicates which historical state should
be reused. LongLive-RAG therefore retrieves from the generator’s own latent trajectory and learns
the search geometry from generated content.
D Why Retrieval in Latent Space?
Most contemporary video diffusion systems denoise in a V AE latent space and decode only after the
latent rollout is complete [42, 36, 27, 51]. A raw-pixel retrieval pipeline would break this execution
pattern. After every completed AR block, the system would need to immediately run V AE decoding
before search, then either store decoded frames or extract pixel-space features for all historical
candidates. This inserts an extra decode-and-transfer path into each generation step, increasing
memory traffic and communication overhead relative to the standard latent-first pipeline.
Latent retrieval also gives a simpler learning problem. The generator’s latents already carry visual
and semantic information in a representation aligned with the denoising model and its attention
context. Training a lightweight encoder on these latents can therefore focus on making the search
space discriminative for recurrence. In contrast, a pixel-space retriever must learn from decoded
frames, which are higher-dimensional and less directly tied to the generator’s internal state. Using an
off-the-shelf image feature space is not a clean substitute: such features can be semantically broad but
insufficiently discriminative for long generated videos, causing the nearest neighbors to remain overly
local in time, similar to the collapse observed with reconstruction-only latent compression. Figure 5
illustrates this behavior with DINO features [ 4] computed from decoded frames: the similarity
structure is dominated by local neighborhoods rather than clean long-range matches.
Figure 5: Raw-pixel retrieval analysis using DINO features [ 4] from decoded frames. The similarity
map shows that off-the-shelf image features tend to retrieve temporally local neighbors, making them
less suitable for long generated videos where useful context may be far outside the recent window.
Finally, the mapping between latent states and decoded pixels is not one-to-one for retrieval. The
same or similar decoded appearance can correspond to different latent states that carry different
denoising histories, context associations, or future trajectory implications. A pixel-space match can
therefore be ambiguous when mapped back to the latent/context objects needed by the AR generator.
LongLive-RAG avoids this mismatch by indexing the objects the generator actually produces and
reuses: self-generated latents and their original context.
16

E Experiment Details
This appendix section records the experimental details used for LongLive-RAG.
Training data construction.We use the prompt pool from Self-Forcing [ 21]. We randomly sample
10% of the prompts with seed 0. For each selected prompt, we generate 90-latent LongLive rollouts
with a frozen generator [ 49]. We use LongLive because it provides long AR rollout latents with
history-dependent states in the target latent space. This choice is used only for collecting training
latents for the retrieval encoder; it is not used to filter evaluation prompts or select reported samples.
We save the clean denoised latent from each completed AR block and use these latents to train the
autoencoder. Generating the resulting 20,000+ training latents takes about 7 days on 6 NVIDIA
RTX A6000 GPUs. This data-construction stage is separate from the inference and evaluation runs
reported in Section 4.1. Decoded videos are used only for sanity checking.
Why single base model? Our choice is to train one retrieval encoder and use it unchanged
across all evaluated backbones.A natural alternative is to train a separate retrieval encoder for each
target AR backbone, or to train on rollouts collected from all base models. We deliberately avoid
this setting. LongLive-RAG is designed as a general RAG framework; per-backbone retrieval would
make the method tied to each new AR generator. We therefore train one encoder in the shared WAN
V AE latent space used by many recent AR video backbones, including Causal-Forcing, Self-Forcing,
and LongLive [ 66,21,49]. Although these backbones may have different rollout distributions, their
generated latents are decoded by the same WAN V AE and therefore share a common latent coordinate
system. The same frozen encoder is used for all evaluated backbones in Table 2, without per-backbone
retraining or data-size tuning.
Latent autoencoder training.The base video generators are frozen, and only the latent autoen-
coder is trained. The autoencoder is a convolutional encoder–decoder applied to each latent block
independently. The encoder uses stride-2 convolutions, GroupNorm, SiLU activations, residual
blocks, global average pooling, and a linear projection to the retrieval embedding. The decoder maps
the embedding back to a bottleneck feature map and reconstructs the latent with nearest-neighbor
upsampling, convolutions, GroupNorm, SiLU activations, and residual blocks. The decoder has no
output activation, and the output is cropped to the target latent size. For retrieval, we use the encoder
output with L2 normalization.
We train with AdamW. The objective follows Section 3.3: mean-squared reconstruction loss, Window
Temporal Delta Loss, and trajectory-smoothing loss. We split the latent dataset into training and
validation sets, using 10% for validation. We track reconstruction, sequence-delta, smoothness, and
total loss on both splits. The best checkpoint is selected by the lowest validation total loss. The
retrieval autoencoder training job takes less than 100 NVIDIA RTX A6000 GPU hours after the
training latents have been generated.
Table 4 summarizes the main hyperparameters. The values are read from the AE training config and
the LongLive-RAG inference configs.
Evaluation protocol.We evaluate generated videos with the official VBench-Long scripts [ 22].
The reported inference and evaluation runs for Table 2 and Table 3 take about one week on 6 NVIDIA
RTX A6000 GPUs, excluding the retrieval-encoder data-construction and autoencoder-training stages
described above. Runtime overhead is measured on one NVIDIA RTX A6000, with latent encoding
and top- Ksearch timed separately. We also conduct an auxiliary VLM evaluation on 30s generations.
17

Table 4: Main hyperparameters used in LongLive-RAG.
parameter value description
Optimizer AdamW optimizer for training the latent autoen-
coder
Sequence length 8 continuous latent blocks per training chunk
Embedding dimension 1024 retrieval embedding dimension
Hidden dimensions [64, 128, 256] encoder and decoder channel dimensions
Learning rate3×10−4learning rate for AdamW
Weight decay1×10−4optimizer weight decay
Batch size 128 training batch size
Training epochs 400 total training epochs
λrec 1 reconstruction loss weight
λ∆ 1.0 Window Temporal Delta Loss weight
Delta windoww3 local window for temporal delta pairs
Delta marginm0.85 cosine-similarity margin
λsmooth 1 trajectory-smoothing loss weight
Gradient clipping 1.0 maximum gradient norm
Mixed precision enabled mixed-precision training
Checkpoint selection lowest val. total loss validation total is the sum of tracked loss
terms
Recency guardR5 number of recent blocks excluded from
retrieval
F Auxiliary VLM Evaluation
VLM evaluation.We use Gemini 3.1-Pro [ 15] as the VLM judge to score each generated video.
Following Deep Forcing and Self-Forcing++ [ 53,9], we randomly sample 20 cases from the 30s
generations for this auxiliary evaluation. The VLM is given the generated video and returns an
aggregate score following the prompt below. Based on the original VLM scoring setup, we revise the
system prompt by adding system instructions, a scoring example, and an explicit output format. We
average scores over the evaluated prompt set and report the mean in Table 3 as an auxiliary check.
VLM Evaluation Prompt
You are an expert video quality assessor. Your task is to evaluate the “Overall Exposure Quality” of the
provided video file.
Since exposure can change over time, you must watch the entire video and provide a single aggregate
score based on the severity and duration of any exposure issues.
Scoring Scale
0: Catastrophic.The video is dominated by pure white (blown out) or pure black (crushed) frames for
a significant duration, rendering the content largely unreadable.
1: Severe.Large portions of the video suffer from severe over- or under-exposure, substantially
impairing visibility and ruining the viewer’s experience.
2: Noticeable.Persistent clipping in highlights or shadows throughout the video, or frequent noticeable
shifts that cause a significant loss of detail.
3: Moderate.Exposure is generally acceptable, but contains brief periods of over/under-exposure, or
constant but minor highlight/shadow clipping that preserves core visual information.
4: Minor.Exposure is mostly excellent. Small regions are occasionally too bright or dark, or there is a
very brief, negligible fluctuation, but it does not disrupt visibility.
5: Perfect.Rock-solid, well-balanced lighting across the entire frame and duration. No distracting
over-exposure, darkening, or unnatural shifts.
Evaluation Guidelines
If the video has mixed quality (e.g., mostly a “5” but has a 2-second period of “1”), penalize the overall
score based on how distracting that bad period is (e.g., an overall score of 2 or 3).
18

Pay attention to both the spatial extent (how much of the screen is affected) and the temporal extent
(how long the issue lasts).
Output Format
First, provide a brief analysis summarizing the exposure across the video’s duration (mentioning any
specific moments of over/under-exposure if they exist). Then, output the final integer score wrapped in
<score></score>tags.
Example:
Analysis: The first 5 seconds are perfectly exposed, but when the camera pans to the window, the sky
becomes completely blown out for the remainder of the video, losing all highlight detail.
Score:<score>3</score>
G Ablation Examples
We provide qualitative examples for the retrieval-budget and embedding-space ablations.
TimeK=2K=4K=6K=8
Figure 6: Retrieval-budget comparison. The balanced setting preserves both long-range references
and video continuity.
19

OursRandomAvg PoolAE
TimeAE + TemporalFigure 7: Embedding-space comparison. The learned LongLive-RAG embedding retrieves more
useful historical context than random or hand-crafted alternatives.
20