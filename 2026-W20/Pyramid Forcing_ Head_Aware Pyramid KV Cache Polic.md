# Pyramid Forcing: Head-Aware Pyramid KV Cache Policy for High-Quality Long Video Generation

**Authors**: Jiayu Chen, Junbei Tang, Wenbiao Zhao, Maoliang Li, Jiayi Luo, Zihao Zheng, Jiawei Yang, Guojie Luo, Xiang Chen

**Published**: 2026-05-13 07:23:02

**PDF URL**: [https://arxiv.org/pdf/2605.13111v1](https://arxiv.org/pdf/2605.13111v1)

## Abstract
Autoregressive video generation enables streaming and open-ended long video synthesis, but still suffers from long-term degradation caused by accumulated errors. Existing KVCache strategies usually apply unified historical-frame retention, implicitly assuming homogeneous historical dependencies across attention heads. We revisit historical-frame attention and reveal three distinct head types: Anchor Heads require broad long-range context, Wave Heads exhibit periodic temporal dependencies, and Veil Heads focus on initial and adjacent frames. Based on this finding, we propose Pyramid Forcing, a head-aware pyramidal KVCache framework that identifies head types offline, assigns behavior-specific cache policies, and supports heterogeneous cache lengths via efficient ragged-cache attention. Experiments on Self Forcing and Causal Forcing show that Pyramid Forcing consistently improves long-horizon generation quality on VBench-Long, increasing the 60-second Self Forcing score from 77.87 to 81.21 while enhancing motion dynamics, visual fidelity, and semantic consistency. Project: https://if-lab-pku.github.io/Pyramid-Forcing/.

## Full Text


<!-- PDF content starts -->

Pyramid Forcing: Head-Aware Pyramid KV Cache
Policy for High-Quality Long Video Generation
Jiayu Chen1Junbei Tang2∗Wenbiao Zhao3∗Maoliang Li1
Jiayi Luo4,5Zihao Zheng1Jiawei Yang1Guojie Luo1Xiang Chen1†
1Peking University2South China University of Technology3Xinjiang University
4Beihang University5Zhongguancun Academy
Rolling Forcing| Vbench-Long = 79.80 
A vibrant and dynamic illustration in the style of a futuristic sci-fi comic, depicting two playful dogs, a Samoyed and a Golden Retriever, running through a neon-lit city at night. The dogs‘ fur gleams under the vibrant glow of the city’s …Frame 1Frame 241Frame 481Frame 721Frame 9610s60sSelf Forcing| Vbench-Long = 77.87 Self Forcing + Deep Forcing| Vbench-Long = 80.08 Self Forcing + Pyramid Forcing| Vbench-Long = 81.21   
Figure 1: Pyramid Forcing mitigates long-video degradation, including appearance drift and subject
inconsistency, achieving a state-of-the-art VBench-Long score of 81.21 among all baselines.
Abstract
Autoregressive video generation enables streaming and open-ended long video
synthesis, but still suffers from long-term degradation caused by accumulated
errors. Existing KVCache strategies usually apply unified historical-frame re-
tention, implicitly assuming homogeneous historical dependencies across atten-
tion heads. We revisit historical-frame attention and reveal three distinct head
types: Anchor Heads require broad long-range context, Wave Heads exhibit peri-
odic temporal dependencies, and Veil Heads focus on initial and adjacent frames.
Based on this finding, we proposePyramid Forcing, a head-aware pyramidal
KVCache framework that identifies head types offline, assigns behavior-specific
cache policies, and supports heterogeneous cache lengths via efficient ragged-
∗Work performed during an internship at Peking University.
†Corresponding author.
Preprint.arXiv:2605.13111v1  [cs.CV]  13 May 2026

Sink FrameHigh attention Current FrameRecent FrameLow attention Compressed FrameUnselected Frame
(c) Pyramid Forcing(b) Deep Forcing(a) Self Forcing211516171819212021151617181921202115161718192120Head 1Head 2Head 3Cache Capacity16172021201918172117202120191817211720212019181721AR Generation with History KVCacheAnchor HeadWave HeadVeil Head2120212121StridedSliding WindowPeriodic SamplingCache Merge212121202020Head-awarePyramid KVCache PolicyUnified KVCache Compression PolicyKVCache FIFO Evict PolicyFigure 2: Comparison of KVCache policies. Unlike Self Forcing and Deep Forcing with unified
cache compression, Pyramid Forcing assigns head-aware strategies to Anchor, Wave, and Veil Heads
for differentiated historical information allocation.
cache attention. Experiments on Self Forcing and Causal Forcing show that Pyra-
mid Forcing consistently improves long-horizon generation quality on VBench-
Long, increasing the 60-second Self Forcing score from 77.87 to 81.21 while
enhancing motion dynamics, visual fidelity, and semantic consistency. Project:
https://if-lab-pku.github.io/Pyramid-Forcing/.
1 Introduction
Autoregressive (AR) video generation models [ 1,2,3] provide a natural pathway for long video
generation by iteratively denoising future frames conditioned on previously generated frames. Unlike
diffusion-based video generation models [ 4,5,6] constrained by a fixed temporal scope, AR video
generation supports streaming generation, continuation and open-ended generation, making it promis-
ing for applications such as interactive generation[ 7,8], world simulation[ 9,10], and embodied
intelligence[ 11,12]. However, in practical scenarios, long video generation still suffers from long-
term degradation. As illustrated in Fig. 1, as the generation process progresses, visual appearance
tends to drift, motion dynamics weaken, and object identity gradually becomes blurred.
Existing works [ 13] have suggested that this degradation primarily arises from error accumulation
during autoregressive generation, especially when full historical information is retained. To address
this issue, prior methods attempt to mitigate error accumulation by retaining recent frames[ 14,15],
preserving early target frames[ 16,17], or selecting important historical information[ 18]. However,
these approaches remain limited in alleviating error accumulation. As shown in Fig. 2, we further
observe that most of them implicitly assume a uniform dependency pattern on historical information
within the model, thereby adopting a unified coarse-grained cache compression strategy.
To examine this assumption, we revisit the attention mechanism in autoregressive video generation
models represented by Self Forcing, and analyze the dependency patterns and temporal behaviors of
different attention heads with respect to historical information during long-term generation.
Our analysis reveals significant heterogeneity in how attention heads utilize historical information:
One group of heads maintains strong and stable attention over a wide range of historical frames,
exhibiting long-range anchoring capability, which we termAnchor Heads. Another group shows
periodic temporal fluctuations in attention over historical frames, demonstrating dynamic and oscil-
latory temporal preferences, which we termWave Heads. A small subset of heads exhibits weak
attention to recent history but is highly sensitive to the initial frame, where excessive historical input
can even suppress their focus on the first frame, which we termVeil Heads.
These findings indicate that the implicit assumption of uniform compression in existing methods does
not hold. Different attention heads exhibit distinct patterns of dependency on historical information.
Therefore, historical information should be allocated in a differentiated manner based on head-specific
temporal behaviors to more effectively preserve critical context and suppress error accumulation.
2

Based on this observation, we propose Pyramid Forcing, a head-aware pyramidal KVCache framework
for long video generation, as shown in Fig. 2(c). Specifically, we first introduceOffline Tri-Pattern
Head Classification, which uses sign-rate statistics and frequency-domain periodicity to identify
Anchor, Wave, and Veil Heads from a small calibration set. Then, we introduce head-specific
Pyramid KVCache Policiestailored to distinct historical-frame attention patterns, using strided
retention for Anchor Heads, period-aligned sampling for Wave Heads, and compact local merging
for Veil Heads to balance long-range coverage, periodic dependency, and local continuity. We
further introduceHeterogeneous KVCache Computation Optimization, enabling low-overhead
ragged-cache attention for variable head-level cache lengths.
We evaluate Pyramid Forcing on Self Forcing and Causal Forcing using VBench-Long over 30s and
60s generations. Pyramid Forcing consistently improves long-horizon quality across both backbones,
increasing the 60s total score of Self Forcing from 77.87 to 81.21. It also yields substantial gains in
dynamic degree, with improvements of up to 29.36 points on Causal Forcing, demonstrating stronger
motion dynamics, visual fidelity, and semantic consistency in long video generation.
2 Preliminaries
2.1 Autoregressive Video Diffusion
Autoregressive video diffusion models generate videos frame by frame during the denoising diffusion
process, conditioned on previously generated frames. Given a video sequence of Nframes, its joint
probability distribution can be factorized in an autoregressive manner as p(x1:N) =QN
t=1p(xt|x<t).
Specifically, the previously generated frames x<t= (x1, x2, . . . , xt−1)are modeled by the self-
attention mechanism and used to predict the current framext.
The query of the current frame, Q∈RB×H×L q×D, attends to the keys and values of historical
frames,K, V∈RB×H×L kv×D. The attention operation is computed as
Attn(Q, K, V) = softmaxQK⊤
√
D
V.(1)
Here, Hdenotes the number of attention heads, and D=C/H is the channel dimension of each
head. LqandLkvdenote the sequence lengths of the query and key/value tokens, respectively. The
resulting attention logits typically have shapeQK⊤∈RB×H×L q×Lkv.
2.2 KVCache Policy in Autoregressive Video Diffusion
During autoregressive video generation, historical frames are encoded into latent tokens and stored as
KVCaches. To reduce computation and mitigate error accumulation, existing methods maintain a
rolling cache by feeding only selected historical frames into the model. When generating the t-th
frame, a subset I(t)⊂ H(t) is selected from the candidate history, typically consisting of sink frames
S(t) for global anchoring and recent frames R(t) for local temporal context. Their KVCaches are
concatenated asK′= [K sink|K recent], V′= [V sink|Vrecent],and attention is computed by
Attn(Q t, K′
t, V′
t) = softmaxQt[K⊤
sink, K⊤
recent]√
D
[Vsink, Vrecent].(2)
After each generation step, the new keys and values are written into H(t+ 1) and used to update
I(t+ 1) . Existing methods usually apply a unified cache policy across attention heads; in contrast,
we analyze head-specific historical dependencies and optimize KV caching at a finer granularity.
3 Motivation
Autoregressive video generation predicts future frames by attending to the KVCache of historical
frames. However, irrelevant historical frames may introduce noise and exacerbate error accumulation,
leading to pronounced long-term degradation and temporal drift in extended generation. To better
inform KVCache policy design, we empirically analyze historical-frame attention patterns in Self
Forcing and summarize the observations as follows.
3

Figure 3: Historical-frame attention patterns of three head types in Self Forcing.Anchor Heads
capture broad long-range dependencies,Wave Headsexhibit periodic temporal selectivity, andVeil
Headsfocus on the first and adjacent frames while suppressing mid-history.
Vbench-Long ScorePeriod of the Head
(b) Impact of Recent-Frame Length on Different Head(a) Periodicity of 120 attention heads across layers 0-9Recent Frame
6Wave Heads exhibit a small and stable perioddue to regular attention fluctuations.Layer/HeadAnchor and Veil Heads exhibit unstable periods due to uniformly high or low attention.Different head types exhibit distinct historical-information requirements.
Figure 4: Analysis of head-wise periodicity and historical information demands. (a) Wave Heads
exhibit a small and stable fluctuation period under FFT analysis, whereas others lack stable periodicity.
(b) Different head types prefer distinct recent history, motivating a head-aware KVCache policy.
Observation 1: Attention heads exhibit significant heterogeneity in their historical-frame
attention patterns.Based on the pre-softmax attention logits and their corresponding historical-
frame attention patterns, we categorize attention heads into three types:Anchor Heads,Wave Heads,
andVeil Heads, as illustrated in Fig. 3.
Anchor Heads.As shown in Fig. 3(a), Anchor Heads assign high attention scores across broad
historical ranges, forming globally activated attention maps. This pattern suggests a long-range
anchoring role that preserves historical information and supports temporal consistency. Accordingly,
Anchor Heads require cache policies that retain wider temporal coverage under limited cache budgets.
Wave Heads.As shown in Fig. 3(b), Wave Heads exhibit regular positive–negative alternations along
the temporal dimension, forming wave-like attention maps. This pattern indicates strong temporal
fluctuations and selective attention to intermediate historical frames, which helps preserve temporal
consistency in long video generation. Thus, Wave Heads only need to retain temporally patterned key
frames, reducing errors from non-critical history.
Veil Heads.As shown in Fig. 3(c), Veil Heads concentrate attention on the first and adjacent frames,
while assigning negative scores to most intermediate frames, yielding globally low attention-score
maps. This pattern suggests reliance on initial and local neighboring frames for structural information
and core semantics, with little mid-history dependence. Thus, retaining excessive intermediate frames
may suppress key signals, introduce errors, and degrade generation quality.
Moreover, despite variations in input prompts, the three types of attention heads consistently exhibit
stable attention patterns. This consistency suggests that attention heads in the autoregressive video
generation model possess stable functional roles that are intrinsic to the model, making them reliably
classifiable. More attention visualization results are provided in the supplementary material.
4

Anchor Head(a) Offline Tri-Pattern Head Classification minmax
(c) KV Cache Computation Optimization(b) Pyramid KV Cache Policies for Heterogeneous Heads
𝐾!𝑄𝑉
𝐿"𝐿#𝐿$𝐿%𝑐𝑢_𝑠𝑒𝑞𝑙𝑒𝑛ce_q𝒔𝒆𝒒𝒍𝒆𝒏𝒄𝒆𝐿%𝐿$𝐿#𝐿"Flatten
FlashAttention Varlen Kernel𝑐𝑢_𝑠𝑒𝑞𝑙𝑒𝑛ce_kv𝒔𝒆𝒒𝒍𝒆𝒏𝒄𝒆𝑄𝐾!𝑉Attention Matrix  𝐴!",$1.SignRateAnchor HeadVeil Head2. PeriodicityWave Head3. Mean-score Fallback
Wave HeadVeil HeadSink Frame 𝑆···0···D𝑦𝑛𝑎𝑚𝑖𝑐𝑠𝑡𝑟𝑖𝑑𝑒𝑠&=𝑡/𝐶𝑎𝑝KV𝐶𝑎𝑐ℎ𝑒=𝑲𝑽[𝑆:ℛ&'={𝑡−𝑖⋅𝑡Cap−𝑡2Cap}:𝑅]StridedSliding Window Policy
Force-preserved Recent Frame 𝑆Frame 𝑡Historical-frame KVCache𝑀𝑒𝑟𝑔𝑒𝑅𝑎𝑛𝑔𝑒𝑚KV𝐶𝑎𝑐ℎ𝑒=𝑲𝑽[𝑆:ℛ&(=Merge𝑡−𝑚:𝑡}:𝑅Merged-frame KVCacheCandidateRange···𝑃𝑒𝑟𝑖𝑜𝑑𝑃KV𝐶𝑎𝑐ℎ𝑒=𝑲𝑽[𝑆:ℛ&)=𝑡−𝑖⋅P}:𝑅Periodic-frame KVCachePeriodic Sampling PolicyCache Merging PolicyUnselected FramesCapacity𝐶𝑎𝑝High AttentionLow Attention···𝐹𝑟𝑎𝑚𝑒	iFigure 5: Overview of Pyramid Forcing. (a) Offline Tri-Pattern Head Classification identifies Anchor,
Wave, and Veil Heads. (b) Pyramid KVCache Policies assign head-specific cache strategies. (c)
KVCache Computation Optimization enables efficient ragged-cache attention with optimized kernels.
Observation 2: Wave Heads exhibit periodic attention patterns over historical frames.To
characterize the temporal fluctuations of Wave Heads, we apply FFT to the attention weights from
the last generated frame to all historical frames in Self Forcing. As shown in Fig. 4(a), Wave Heads
exhibit a small and stable fluctuation period, unlike Anchor and Veil Heads. This suggests that, when
generating the t-th frame, Wave Heads consistently attend to frames at positions t−P×i, i≥1 ,
where Pdenotes the estimated period. Therefore, Wave Heads can be identified by periodicity, and
their key historical frames can be retained accordingly. See Appendix B.4 for theoretical analysis.
Observation 3: Different attention heads require different amounts of historical information.
As shown in Fig 4(b), we evaluate how the length of recent frame R(t) affects 30-second video
generation quality across the three head types. Anchor Heads improve consistently with longer
windows, confirming their dependence on long-range history. Wave Heads perform best with 10
retained frames, and Veil Heads peak at only 5 frames, with quality degrading as the window further
expands. These results reveal distinct head-wise demands for historical information, suggesting that
AR video generation should adopt head-aware KVCache allocation instead of the unified policies.
4 Method
4.1 Offline Tri-Pattern Head Classification
Motivated by the heterogeneous yet stable historical attention distributions in video generation, we
propose an offline head classification method(Fig. 5(a)). We first sample a small set of prompts,
generate videos, and classify attention heads according to their historical-frame attention patterns.
As discussed inObservation 1andObservation 2, Anchor Heads assign broadly positive scores to
historical frames, Veil Heads mainly attend to the first and nearby frames while suppressing most
history, and Wave Heads exhibit periodic positive–negative oscillations. We therefore classify heads
using two complementary criteria: sign-rate statistics and frequency-domain periodicity.
Specifically, when generating the T-th frame, for the h-th head in the l-th layer, we denote the
pre-softmax attention-score matrix as A(l,h)
T∈RT×Tand extract the historical attention sequence as
a(l,h)(t) =A(l,h)
T[T, t],t= 1, . . . , T−1. We then compute the sign rates of sequencea(l,h)(t)as
rpos=1
T−1T−1X
t=11{a(l,h)(t)>0}, r neg= 1−r pos.(3)
Given a threshold αadjusted according to the empirical attention-score distribution, heads with
rpos≥αare classified as Anchor Heads, while those withr neg≥αare classified as Veil Heads.
5

For the remaining heads, we detect periodicity in the frequency domain. After preprocessing a(l,h)(t),
we apply FFT and estimate the dominant period P= Period(FFT(a(l,h)(t))) . Since Wave Heads
concentrate within a small and stable period range, a head is classified as a Wave Head if P < β ,
whereβis determined by theoretical and experimental analysis.
For any still-unclassified heads, we use a mean-score fallback: ¯a(l,h)=1
T−1PT−1
t=1a(l,h)(t). If
¯a(l,h)>0, the head is assigned to Anchor Heads; otherwise, it is assigned to Veil Heads. Finally, we
obtain the layer-wise head sets HA,HW, andHV, which are mutually exclusive and jointly cover all
attention heads. For notational simplicity, we use superscripts (A),(W) , and (V)to denote variables
associated with the three head types.
4.2 Pyramid KVCache Policies for Heterogeneous Heads
As shown in Fig. 5(b), we introduce Pyramid KVCache Policies, which adapt historical-frame caching
to Anchor, Wave, and Veil Heads, enabling fine-grained head-aware allocation for inference.
Anchor Heads KVCache Policy.Anchor Heads attend to a broad range of historical frames,
requiring a cache policy that preserves long-range information under a limited budget. We therefore
adopt anAdaptive Strided Sliding Windowstrategy, which adjusts the stride according to the current
history length and expands the effective coverage to the full generated sequence.
Specifically, when generating the t-th frame, we set the dynamic stride as st=t/Cap and uniformly
sampleCaprepresentative historical frames by selecting the center of each strided window:
R(A)
(t)=
t−i·t
Cap−t
2Cap
|i= 0,1, . . . ,Cap−1
.(4)
The corresponding KVCache is then K(A)
recent =K
R(A)
(t),:
,V(A)
recent =V
R(A)
(t),:
.This
centered strided sampling preserves broad historical semantics in a sparse yet representative manner.
Wave Heads KVCache Policy.Wave Heads exhibit stable periodic attention over historical frames,
requiring cache retention aligned with their intrinsic oscillation. We therefore adopt aPeriodic
Samplingpolicy, which selects representative historical frames across multiple periods under a
limited budget.
Specifically, when generating thet-th frame, we use the dominant periodPidentified by the offline
analysis in Sec. 3 and retainCaphistorical frames at intervalP:
R(W)
(t)={t−i·P|i= 0,1, . . . ,Cap−1}.(5)
The corresponding KVCache is then K(W)
recent =K
R(W)
(t),:
,V(W)
recent =V
R(W)
(t),:
.This
period-aligned sampling captures the fluctuating long-range dependency of Wave Heads.
Veil Heads KVCache Policy.Veil Heads concentrate attention on the first and nearby frames, while
assigning negative scores to intermediate history. To avoid errors from excessive historical context,
we adopt aCache Mergingstrategy that compresses local information within a small window.
Specifically, when generating the t-th frame, we split each frame in a local window of size minto
mfeature subsets. We then take the corresponding 1/m subset from each frame and concatenate
them into a representative feature: R(V)
(t)={merged(t−m:t)}. The merged KVCache is defined
asK(V)
merged= Concatm−1
i=0K
t−m+i,C i
,V(V)
merged= Concatm−1
i=0V
t−m+i,C i
,where
Cidenotes the selected feature subset of the i-th frame. This local merging preserves core nearby
information while suppressing irrelevant intermediate history.
In addition, all heads share the same sink frames Sand forced recent frames R, as both receive high
attention across head types. We further apply Dynamic RoPE to sink frames following prior works [ 18,
19]. Pyramid Forcing differs by applying head-aware policies to intermediate historical frames,
allocating retention ranges and cache strategies according to heterogeneous temporal dependencies.
6

Table 1: Quantitative comparison on long video generation. We evaluate Pyramid Forcing against
autoregressive video diffusion generation baselines for 30- and 60-second videos using VBench-Long.
ModelDynamic Motion Overall Imaging Aesthetic Quality Semantic Total
Degree↑Smoothness↑Consistency↑Quality↑Quality↑Score↑Score↑Score↑
30 seconds
CausVid 45.70 98.25 22.39 66.11 59.63 86.66 50.93 79.51
Rolling Forcing 30.91 98.72 24.85 71.43 61.69 86.18 55.20 79.98
LongLive 41.70 98.79 24.54 68.79 61.36 86.97 54.31 80.44
Self Forcing 44.34 98.52 24.66 70.66 63.22 87.14 54.31 80.57
+ Deep Forcing 46.48 98.43 25.10 71.70 63.77 87.23 54.00 80.59
+ Pyramid Forcing 55.07 98.82 25.26 72.17 66.55 88.93 55.52 82.25
Causal Forcing 61.3898.4823.0369.6658.22 86.45 53.12 79.78
+ Pyramid Forcing 82.67 97.23 23.21 68.64 60.03 86.98 54.07 80.40
60 seconds
CausVid 43.67 98.28 21.65 65.54 59.34 86.58 49.87 79.23
Rolling Forcing 32.08 98.73 24.20 70.55 61.06 86.19 54.20 79.80
LongLive 40.93 98.77 24.54 68.83 61.70 86.89 54.79 80.47
Self Forcing 43.75 97.85 22.33 64.28 55.62 84.84 50.01 77.87
+ Deep Forcing 43.10 98.35 24.48 68.43 60.48 86.37 54.91 80.08
+ Pyramid Forcing 53.68 98.53 24.70 70.03 62.00 87.58 55.71 81.21
Causal Forcing 57.0398.49 22.51 68.6356.87 85.84 52.36 79.14
+ Pyramid Forcing 86.39 97.03 22.41 68.38 58.91 86.64 53.05 79.92
4.3 Heterogeneous KVCache Computation Optimization
Ragged-Cache Attention.In standard autoregressive video diffusion, attention assumes a shared
dense KVCache, K, V∈RB×H×L×D. Our head-specific cache policies break this uniform-length
assumption by assigning each batch–head pair an effective history length L⟨b,h⟩, yielding a head-level
ragged cache. Direct dense attention is thus inapplicable, while padding incurs redundant memory
and computation. To support heterogeneous head-level caches, we propose Ragged-Cache Attention,
which stores variable-length KVCaches as flattened sequences with boundary indices, as shown in
Fig. 5(c). Specifically, given N=B×H sequences with lengths Li, we define cu_seqlensi=Pi−1
j=0Ljfor1≤i≤N , such that the i-th sequence occupies [cu_seqlensi,cu_seqlensi+1). With
queries and KVCaches flattened in batch–head order, variable-length FlashAttention[ 20] computes
attention directly from flattenedQ,K/V, andcu_seqlenswithout padding.
Efficient kernel customization.Although variable-length FlashAttention[ 20] supports packed
ragged sequences, separate per-frame calls introduce nontrivial launch overhead. We therefore
customize execution with multi-frame attention fusion, which merges three per-layer FlashAttention-
varlen calls into one ragged-segment invocation after permuting queries to match the KV layout.
We further vectorize workspace packing by replacing many per-head slice writes with batched
writes. These optimizations reduce attention-kernel launches by more than two-thirds and lower GPU
scheduling calls from about 60 to 6, enabling efficient heterogeneous KVCache computation.
5 Experiments
5.1 Experimental Setup
Evaluation.Following Self Forcing [ 1], we use 128 MovieGen [ 21] prompts for 30-second and
60-second video generation. All results are evaluated on VBench-Long [ 22,23], with representative
metrics reported in the main paper and full results provided in the appendix A.1. We compare
Pyramid Forcing with autoregressive video diffusion baselines, including distillation-based methods
7

A movie trailer in a classic cinematic style, featuring the adventurous 
journey of a 30-year-old space man wearing a vibrant red …0s 30s+Pyramid
 Forcing+Deep 
ForcingSelf
ForcingCausal
 Forcing+Pyramid 
Forcing
A dynamic scene captured in the style of a vibrant food photography, 
showcasing a chef skillfully chopping onions in a bustling kitchen …0s 60s
Figure 6: Qualitative comparison of Pyramid Forcing and baseline methods on 30-second and 60-
second video generation. Pyramid Forcing better preserves subject consistency, temporal coherence,
and visual fidelity over long horizons.
Table 2: Ablation study on the key components
of Pyramid Forcing across six variants.
Method Dynamic Degree↑Image Quality↑VBench-Long↑Lat.↓Mem.↓
Self Forcing 37.89 71.51 80.44 54.12s 35.80GB
+ Dynamic ROPE 41.79 70.74 80.96 54.17s 35.80GB
+ Ragged-Cache 41.01 71.08 80.4352.05s35.80GB
+ Head 39.84 71.16 80.99 54.03s34.87GB
+ Head & Pyramid 45.8972.0681.58 139.70s 36.06GB
Pyramid Forcing 50.58 71.73 82.16 52.31s 36.32GBTable 3: Efficiency and peak GPU memory usage.
FPS denotes generated frames per second.
Method VBench-Long↑Latency↓FPS↑Memory↓
Self Forcing 80.57 54.01s 11.85 35.80GB
+ Deep Forcing 80.59 52.85s 12.19 37.17GB
+ Pyramid Forcing 82.25 52.18s 11.98 36.32GB
Causal Forcing 79.78 46.75s 13.31 47.93GB
+ Pyramid Forcing 80.40 45.21s 13.78 45.59GB
such as CausVid [ 24], Rolling Forcing [ 15], and LongLive [ 14], as well as the training-free Deep
Forcing [ 18]. Unless otherwise specified, all methods share the same inference settings, sampling
hyperparameters, and generation lengths, differing only in KVCache mechanisms.
Implementation Details.We implement Pyramid Forcing on Self Forcing [ 1] and Causal Forc-
ing [3]. We set the sink size to 3, the force-preserved recent window to 4, and the cache capacities of
Anchor, Wave, and Veil Heads to 4,4, and 2, respectively, with the Veil merge range set to m= 2 .
For Tri-Pattern Head Classification, we sample 32 VBench-Long prompts and extract attention
sequences from the generated 15-second videos. The sign-rate and period thresholds are set to
80% and6.4, respectively. Customized kernels are implemented based on FlashInfer/FlashAttention
backends [25, 20], and all benchmarks are conducted on NVIDIA H200 GPUs.
5.2 Main Results
Quantitative Results.We evaluate 30-second and 60-second autoregressive video generation on
VBench-Long, with results reported in Tab. 1. Pyramid Forcing achieves the highest Total Score at
both lengths, improving Self Forcing from 80.57 to 82.25 on 30-second videos and from 77.87 to
81.21 on 60-second videos. It also consistently improves over Deep Forcing and further boosts Causal
Forcing, demonstrating its effectiveness in mitigating long-horizon degradation. Beyond overall
quality, Pyramid Forcing shows clear gains in Dynamic Degree, especially under Causal Forcing,
increasing it from 61.38 to 82.67 on 30-second videos and from 57.03 to 86.39 on 60-second videos.
We attribute these gains to head-aware cache allocation: unlike unified cache policies, Pyramid
Forcing assigns differentiated caches according to head-specific temporal dependencies, preserving
long-range consistency while avoiding overly strong motion constraints.
Qualitative Results.The visual comparisons in Fig. 6 show that existing methods suffer from
varying degrees of long-term degradation during long video generation. Self Forcing produces
noticeable appearance drift and noise artifacts as generation progresses. Deep Forcing and Rolling
Forcing can mitigate part of the degradation, but their subject details and motion dynamics remain
unstable. In contrast, Pyramid Forcing better preserves subject appearance, scene structure, and local
details in 60-second videos, while retaining richer camera and subject motion. These qualitative
8

Table 4: Effect of prompt number and video dura-
tion on offline tri-pattern head classification.
Prompts Duration VBench-Long↑Overhead↓Peak Mem.↓
1 60s 81.83 507.32 s 80.14 GB
1 30s 81.81 249.58 s 62.20 GB
1 15s 81.83 103.26 s 38.81 GB
1 10s 81.60 70.10 s 32.75 GB
1 5s 81.51 33.95 s 26.11 GB
128 15s 81.83 3.67 h 38.81 GB
64 15s 81.83 1.83 h 38.81 GB
32 15s 81.83 0.91 h 38.79 GB
16 15s 81.82 0.46 h 38.80 GBTable 5: Effect of head-specific policy parameters
in Pyramid KVCache policies.
Head Type Setup Dynamic Degree↑Image Quality↑VBench-Long↑
Self Forcing - 37.89 71.51 80.44
Pyramid Forcing - 50.58 71.73 82.16
Anchor
Headst= 350.98 72.24 81.92
st= 651.76 72.10 82.05
st= 1253.32 72.80 82.08
Wave
HeadP= 348.05 72.40 81.95
P= 653.32 72.49 82.06
P= 1250.78 72.01 81.96
Veil
Headm= 250.39 72.23 82.00
m= 348.63 72.23 81.94
results further verify that our method achieves a better balance among long-range consistency, visual
quality, and dynamic expressiveness.
5.3 Ablation Studies
A A vibrant illustration in a whimsical cartoon style depicting 
a flock of paper airplanes fluttering through a dense …0s 30s
+ Ragged
Cache + Dynamic
 ROPE Self
Forcing+ Head + Head & 
Pyramid KVPyramid
Forcing
Figure 7: Qualitative ablation of components.Ablation Study on Key Components.We con-
duct a component-wise ablation study over six vari-
ants: Self Forcing, variants with only Dynamic RoPE,
Ragged-Cache Attention, or Head Classification, the
combination of Head Classification and Pyramid KV-
Cache policies, and the full Pyramid Forcing. For the
Head Classification variant, the type-specific neigh-
boring windows are set to R(A)= 12 ,R(W)= 10 ,
andR(V)= 5. As shown in Tab. 2 and Fig. 7, Dy-
namic RoPE improves motion dynamics, and Ragged-
Cache Attention reduces latency with negligible qual-
ity loss. Head Classification improves VBench-Long,
while Pyramid KVCache policies further enhance
quality at the cost of extra heterogeneous-cache over-
head. The full Pyramid Forcing achieves the best
Dynamic Degree and VBench-Long with low latency,
confirming the complementarity of all components.
Efficiency and Peak GPU Memory Usage Study.We evaluate the computational efficiency and
peak GPU memory usage of Pyramid Forcing against other KVCache strategies, with results reported
in Tab. 3. Pyramid Forcing achieves comparable efficiency on both Self Forcing and Causal Forcing
while consistently improving VBench-Long performance.
Effect of Prompt Number and Video Duration on Offline Tri-pattern Head Classification.We
study how prompt number and video duration affect classification quality and overhead on a single
H200 GPU. As shown in Tab. 4, 32 prompts with 15-second videos achieve comparable quality to
longer or larger settings while substantially reducing time and memory cost. Since classification is
performed only once per model, its overhead is negligible compared with training or distillation.
Effect of head-specific policy parameters in Pyramid KVCache policies.We evaluate the
sensitivity of Pyramid KVCache policies to head-specific parameters. As shown in Tab. 5, all
configurations outperform Self Forcing and remain close to the full Pyramid Forcing performance.
The small variation across settings indicates that the method is robust to hyperparameter choices.
6 Related Work
Autoregressive Video Diffusion.Diffusion-based video generators achieve strong visual quality
through large-scale training [ 4,5,6], but full-sequence denoising with bidirectional attention limits
them to fixed temporal windows. Recent autoregressive video diffusion methods extend generation
9

by conditioning future clips on synthesized history: CausVid [ 24] distills causal generators from
bidirectional DiTs, Self Forcing [ 1] reduces train-test mismatch via self-generated conditioning,
and Causal Forcing [ 3] improves distillation with ODE initialization. Self Forcing++ [ 13], Rolling
Forcing [ 15], and LongLive [ 14] further improve long-horizon stability through error correction or
training-inference alignment. Nevertheless, long video generation still suffers from appearance drift,
motion decay, and identity inconsistency.
KVCache Policy.KVCache optimization is crucial for improving the quality and efficiency of
autoregressive generation. Existing LLM studies mainly focus on cache selection [ 26,27,28,29],
budget allocation [ 30,31,32], and cache compression [ 33,34,35,36,37]. However, these methods
are tailored to textual sequences and are difficult to directly transfer to autoregressive video generation,
where KVCaches must preserve semantic context, temporal continuity, spatial structure, and cross-
frame dependencies. Moreover, attention heads often exhibit heterogeneous temporal behaviors,
requiring video-specific cache policies.
7 Conclusion
We propose Pyramid Forcing, a training-free head-aware KVCache framework for autoregressive long
video generation. By identifying three temporal dependency patterns of attention heads, i.e., Anchor,
Wave, and Veil Heads, Pyramid Forcing applies differentiated cache policies with Ragged-Cache
Attention. Experiments show improved subject consistency, temporal coherence, and visual fidelity
while preserving inference efficiency, providing a new direction for fine-grained cache optimization
in long-horizon video generation.
10

References
[1]Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the
train-test gap in autoregressive video diffusion.arXiv preprint arXiv:2506.08009, 2025.
[2]Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang,
WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale.arXiv preprint
arXiv:2505.13211, 2025.
[3]Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregres-
sive diffusion distillation done right for high-quality real-time interactive video generation.arXiv preprint
arXiv:2602.02214, 2026.
[4]William Peebles and Saining Xie. Scalable diffusion models with transformers. InICCV, pages 4195–4205,
2023.
[5]Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming
Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models.arXiv preprint
arXiv:2503.20314, 2025.
[6]Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi
Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert
transformer.arXiv preprint arXiv:2408.06072, 2024.
[7]Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang,
Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive
world model.arXiv preprint arXiv:2508.13009, 2025.
[8]Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin
Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model.arXiv preprint
arXiv:2507.17744, 2025.
[9]Jianhao Yuan, Xiaofeng Zhang, Felix Friedrich, Nicolas Beltran-Velez, Melissa Hall, Reyhane Askari-
Hemmat, Xiaochuang Han, Nicolas Ballas, Michal Drozdzal, and Adriana Romero-Soriano. Inference-time
physics alignment of video generative models with latent world models.arXiv preprint arXiv:2601.10553,
2026.
[10] Linxi Xie, Lisong C Sun, Ashley Neall, Tong Wu, Shengqu Cai, and Gordon Wetzstein. Generated reality:
Human-centric world simulation using interactive video generation with hand and camera control.arXiv
preprint arXiv:2602.18422, 2026.
[11] Yao Feng, Chendong Xiang, Xinyi Mao, Hengkai Tan, Zuyue Zhang, Shuhe Huang, Kaiwen Zheng,
Haitian Liu, Hang Su, and Jun Zhu. Vidarc: Embodied video diffusion model for closed-loop control.
arXiv preprint arXiv:2512.17661, 2025.
[12] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru,
You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies.arXiv
preprint arXiv:2602.15922, 2026.
[13] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui
Hsieh. Self-forcing++: Towards minute-scale high-quality video generation.ICLR, 2025.
[14] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze
Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation.ICLR, 2025.
[15] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video
diffusion in real time.ICLR, 2025.
[16] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limited-horizon
training and open-ended testing in autoregressive video diffusion.arXiv preprint arXiv:2602.07775, 2026.
[17] Youngrae Kim, Qixin Hu, C-C Jay Kuo, and Peter A Beerel. Memrope: Training-free infinite video
generation via evolving memory tokens.arXiv preprint arXiv:2603.12513, 2026.
[18] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forc-
ing: Training-free long video generation with deep sink and participative compression.arXiv preprint
arXiv:2512.05081, 2025.
11

[19] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope:
Action-controllable infinite video generation emerges from autoregressive self-rollout.arXiv preprint
arXiv:2511.20649, 2025.
[20] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-
efficient exact attention with io-awareness.NeurIPS, 35:16344–16359, 2022.
[21] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas,
Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models.
arXiv preprint arXiv:2410.13720, 2024.
[22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu,
Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative
models. InCVPR, pages 21807–21818, 2024.
[23] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit,
Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video
generative models.IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
[24] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and
Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. InCVPR, pages
22963–22974, 2025.
[25] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris
Kasikci, Vinod Grover, Arvind Krishnamurthy, et al. Flashinfer: Efficient and customizable attention
engine for llm inference serving.Proceedings of Machine Learning and Systems, 7, 2025.
[26] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation.
NeurIPS, 37:22947–22970, 2024.
[27] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong
Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of
large language models.Advances in Neural Information Processing Systems, 36:34661–34710, 2023.
[28] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language
models with attention sinks.ICLR, 20234.
[29] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. {InfiniGen }: Efficient generative inference
of large language models with dynamic {KV}cache management. In18th USENIX Symposium on
Operating Systems Design and Implementation (OSDI 24), pages 155–172, 2024.
[30] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong,
Yue Dong, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information
funneling.arXiv preprint arXiv:2406.02069, 2024.
[31] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. Ada-kv: Optimizing kv cache eviction by
adaptive budget allocation for efficient llm inference.arXiv preprint arXiv:2407.11550, 2024.
[32] Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv
cache compression for high-throughput llm inference. InACL, pages 3258–3270, 2024.
[33] Wen Cheng, Shichen Dong, Jiayu Qin, and Wei Wang. Qaq: Quality adaptive quantization for llm kv
cache. InICCV, pages 2542–2550, 2025.
[34] Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. Compressed context memory for
online language model interaction.ICLR, 2023.
[35] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and
Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache.ICML, 2024.
[36] Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. Minicache: Kv
cache compression in depth dimension for large language models.NeurIPS, 37:139997–140031, 2024.
[37] Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. Get more with
less: Synthesizing recurrence with kv cache compression for efficient llm inference.ICML, 2024.
12

A Additional Visualizations and Experimental Results.
A.1 Additional Evaluation Metrics of the Main Experiment
Table 6 reports the remaining VBench-Long metrics for the main comparison, complementing the
qualitative examples above and the ablations in Section A.3.
Table 6: Supplementary quantitative comparison on long video generation. This table includes the
remaining 8 VBench-Long metrics not shown in the main text.
ModelApp. Background HumanScene↑Subject Temporal Temporal Temporal
Style↑Consist.↑Action↑Consist.↑Flickering↓Style↑(2D)↑
30 seconds
CausVid 24.19 97.04 18.70 26.76 98.05 97.12 22.39 56.36
Rolling Forcing 23.91 96.66 20.0234.77 97.7197.69 24.85 56.93
LongLive 23.61 96.62 21.14 31.50 97.61 97.69 24.54 57.33
Self Forcing 23.89 95.84 21.14 30.90 96.31 97.18 24.18 57.14
+ Deep Forcing24.0895.7821.4430.62 96.1996.8624.35 57.66
+ Pyramid Forcing 23.82 96.30 20.75 32.61 97.28 97.35 24.55 58.41
Causal Forcing 23.3995.2719.7834.36 96.6295.45 23.03 58.21
+ Pyramid Forcing 23.89 94.88 23.83 33.23 95.25 94.50 23.21 60.05
60 seconds
CausVid 24.16 97.18 18.80 25.43 98.18 97.24 21.65 55.93
Rolling Forcing 23.89 96.64 20.49 32.51 97.74 97.72 24.20 56.65
LongLive 23.80 96.49 20.8233.6497.44 97.6824.5457.43
Self Forcing 23.56 95.53 18.22 23.95 95.7596.8522.33 55.00
+ Deep Forcing24.1596.15 21.0933.2496.93 97.4124.4857.36
+ Pyramid Forcing 23.94 96.20 21.62 29.90 97.30 97.30 24.16 58.09
Causal Forcing 23.3395.0919.63 33.2096.5795.3922.5157.44
+ Pyramid Forcing 23.69 94.58 22.96 33.44 94.78 94.00 22.41 59.91
A.2 Further Visualizations of the Main Experiment
To further facilitate visual comparison, we provide additional qualitative results in Figures 8–11,
covering both 30-second and 60-second generation. These examples complement the main qualitative
comparison by showing object-centric scenes, large camera/object motion, underwater environments,
and human portraits. Overall, Pyramid Forcing preserves subject identity, scene layout, and local
visual details more consistently across long horizons, while several baseline cache strategies exhibit
either appearance drift, abrupt background changes, or weakened temporal continuity.
13

A charming illustration in a watercolor style of a young white rabbit wearing glasses and reading a newspaper. The rabbit 
has soft fur, large round ears, and gentle, curious eyes. It sits upright on a cozy armchair, one paw holding.Frame 1 Frame 8 Frame 64 Frame 88 Frame 116
0s 30s
LongLive Rolling Forcing CasuVid Self Forcing +Deep Forcing
 Causal Forcing +Pyramid Forcing +Pyramid ForcingFigure 8: Additional Visualization A
A cinematic scene from a classic western movie, featuring a rugged man riding a powerful horse through the vast Gobi 
Desert at sunset. The man, dressed in a dusty cowboy hat and a worn leather jacket, reins tightly on the horse‘s neck as he 
gallops across the golden sands…Frame 1 Frame 24 Frame 60 Frame 92 Frame 108
0s 30s
LongLive Rolling Forcing CasuVid Self Forcing +Deep Forcing Causal Forcing +Pyramid Forcing +Pyramid Forcing
Figure 9: Visualization B
14

A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and 
yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a va riety of 
marine life, including small schools of colorful fish and sea turtles gliding by…Frame 8 Frame 52 Frame 116 Frame 164 Frame 184
0s 60s
LongLive Rolling Forcing CasuVid Self Forcing +Deep Forcing Causal Forcing +Pyramid Forcing +Pyramid ForcingFigure 10: Visualization C
A zoom -in shot focusing on the face of a young woman sitting on a bench in the middle of an empty school gym. The woman 
has long wavy brown hair cascading down her shoulders and soft, warm hazel eyes. She wears a simple white t -shirt and 
blue jeans, her hands resting gently on her knees. Her expression is serene, with a slight smile playing on her lips…Frame 8 Frame 44 Frame 124 Frame 160 Frame 212
0s 60s
LongLive Rolling Forcing CasuVid Self Forcing +Deep Forcing Causal Forcing +Pyramid Forcing +Pyramid Forcing
Figure 11: Visualization D
15

Figure 11 further evaluates a 60-second human portrait in an indoor gymnasium, where face details,
hair, pose, and background elements are all sensitive to identity drift. Pyramid Forcing better preserves
the person and the indoor scene across time, whereas several baselines show more noticeable changes
in facial appearance, body configuration, or background consistency. These visual results motivate
the following ablation analyses, which isolate how persistent sink-frame context and recent-frame
retention contribute to stable long-horizon generation.
A.3 Impact of Sink Frames and Forced Recent Frame Retention
We evaluated the impact of varying numbers of sink frames ( S) and forced recent frames ( R) on
video quality using a single NVIDIA H200 GPU. These ablations complement the qualitative results
above by testing two cache design choices that directly affect long-range temporal stability: persistent
sink-frame context and forced retention of the most recent frames. As shown in Table 7, with S= 3
fixed, R= 4 yields the optimal generation quality. As indicated in Table 8, when R= 4 is fixed,
although S= 3 is not the absolute best, the gap in the total VBench score is negligible compared to
the optimal configuration.
Table 7: Ablation Study on Different Recent Window SizeR(FixedS= 3).
Recent Dynamic↑Motion↑Overall↑Imaging↑Aesthetic↑Quality↑Semantic↑Total↑
Frames Degree Smooth. Consist. Quality Quality Score Score Score
2 19.1799.0925.9773.0165.40 86.79 54.34 80.30
3 40.00 98.7326.1872.7965.6287.6855.2281.19
455.6398.78 25.52 72.27 64.5888.6154.3081.75
5 52.71 98.81 25.68 72.86 64.86 88.51 54.38 81.68
6 53.96 98.79 25.48 72.79 64.42 88.48 54.38 81.66
Table 8: Ablation Study on Different Sink FramesS(FixedR= 4).
Sink Dynamic↑Motion↑Overall↑Imaging↑Aesthetic↑Quality↑Semantic↑Total↑
Frames Degree Smooth. Consist. Quality Quality Score Score Score
167.2998.64 25.27 71.38 62.9388.7753.35 81.69
2 58.54 98.7725.8172.18 64.28 88.6854.91 81.93
3 55.63 98.78 25.52 72.27 64.58 88.61 54.30 81.75
4 56.46 98.79 25.5272.27 64.5988.66 54.15 81.76
5 57.0898.7925.52 72.27 64.59 88.72 54.26 81.83
A.4 Ablation Study on Period and Sign Rate Thresholds
As shown in Table 9, we conducted a comprehensive ablation study on thePeriod Thresholdwithin
our attention head classification method. The results indicate that with a fixed sign rate threshold
of 80%, the model achieves its peak performance of 82.16 when the period threshold is set to 6.4.
Simultaneously, as illustrated in Table 10, we analyzed the impact of theSign Rate Threshold. The
experimental data demonstrate that, under a fixed period threshold of 6.4, a sign rate threshold of 80%
yields the optimal overall score for the generated videos. These findings suggest that this specific
parameter configuration provides the superior classification of attention heads, establishing an ideal
balance for distinguishing between Anchor/Veil heads and Wave heads.
16

Table 9: Ablation Study on Period Threshold (Fixed Symbol Rate Threshold = 80%).
Period Dynamic↑Motion↑Overall↑Imaging↑Aesthetic↑Quality↑Semantic↑Total↑
Threshold Degree Smooth. Consist. Quality Quality Score Score Score
6.450.5998.72 26.00 71.74 65.2388.32 57.54 82.16
6.5 46.48 98.78 26.23 72.7265.4388.07 56.93 81.85
7.0 46.0998.8026.1472.7465.50 88.11 56.26 81.74
8.0 46.29 98.7826.2772.64 65.39 88.06 56.12 81.67
9.0 46.29 98.79 26.19 72.77 65.30 88.08 56.61 81.79
Table 10: Ablation Study on Symbol Rate Threshold (Fixed Period Threshold = 6.4).
Symbol Rate Dynamic↑Motion↑Overall↑Imaging↑Aesthetic↑Quality↑Semantic↑Total↑
Threshold Degree Smooth. Consist. Quality Quality Score Score Score
70% 45.31 98.74 25.9072.7865.12 87.86 56.32 81.55
80%50.5998.7226.0071.7465.23 88.32 57.54 82.16
90% 49.2298.8425.72 71.34 64.38 88.09 56.84 81.84
A.5 Ablation Study on Head-Specific Capacities
As shown in Table 11, we conducted an ablation study on theCapacityparameters for the three
distinct types of attention heads: Anchor, Wave, and Veil.
•Anchor Head (Stride):Anchor heads benefit from larger capacity, achieving the best
performance at capacity 4, consistent with their long-range dependency requirement.
•Wave Head (Periodic):Wave heads also peak at capacity 4, indicating that sufficient
capacity helps capture periodic motion patterns.
•Veil Head (Merge):Veil heads perform best with a smaller capacity, reaching 82.242,
suggesting that excessive history may introduce noise rather than improve fusion.
This group of experiments fully demonstrates that adjusting the capacity specifically for different
types of attention heads can maximize the optimization of video generation quality.
Table 11: Quantitative comparison based on different Head Types and Capacities. We evaluate the
performance across multiple quality metrics on VBench-Long Lite.
CapacityDynamic Motion Overall Imaging Aesthetic Quality Semantic Total
Degree↑Smoothness↑Consistency↑Quality↑Quality↑Score↑Score↑Score↑
Anchor Head (Stride)
1 51.953 98.694 25.984 71.43065.50288.37857.07682.118
2 52.148 98.729 26.04271.45165.352 88.386 57.017 82.112
4 52.734 98.751 26.069 71.429 65.267 88.482 56.894 82.165
Wave Head (Periodic)
3 50.00098.756 26.05771.512 65.457 88.28757.29082.088
4 53.125 98.748 26.024 71.629 65.496 88.570 56.718 82.200
5 52.344 98.754 25.95971.70865.223 88.409 56.775 82.082
Veil Head (Merge)
2 53.711 98.716 26.106 71.466 65.512 88.520 57.131 82.242
3 52.734 98.751 25.96271.71965.428 88.492 56.540 82.102
4 50.19598.78626.035 71.49065.61088.409 56.592 82.046
17

A.6 Visualizing Attention Patterns
Attention heads exhibit stable, repeatable functional patterns across diverse prompts and lengths.
As shown in Figures 13–16, despite varying scene semantics and dynamics, specific head groups
consistently demonstrate long-range, oscillatory, or local attention. This suggests that functional
specialization is an intrinsic structural property of autoregressive video models.
Moreover, this phenomenon is not limited to Self Forcing. As illustrated in Figure 17, Causal Forcing
also demonstrates clear functional differentiation among attention heads. Similar to Self Forcing, the
attention heads can likewise be broadly categorized into these three functional types.
03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
0 36 7103671H8
0 36 71H9
0 36 71H10
0 36 71H11
00.02.55.0H0 H1 H2 H3
05H4 H5 H6 H7
0 710510H8
0 71H9
0 71H10
0 71H11
(a) Frame attention logits (b) Last-block attention distribution
Figure 12: Attention visualization of the 72-frame Self Forcing model at Layer 23: “A majestic eagle
soaring through a cloudy sky, cinematic lighting.”
03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
0 36 7103671H8
0 36 71H9
0 36 71H10
0 36 71H11
002H0 H1 H2 H3
024H4 H5 H6 H7
0 710510H8
0 71H9
0 71H10
0 71H11
(a) Frame attention logits (b) Last-block attention distribution
Figure 13: Attention visualization of the 72-frame Self Forcing model at Layer 23: “A FPV drone
shot through a castle on a cliff.”
03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
0 36 7103671H8
0 36 71H9
0 36 71H10
0 36 71H11
002H0 H1 H2 H3
024H4 H5 H6 H7
0 7105H8
0 71H9
0 71H10
0 71H11
(a) Frame attention logits (b) Last-block attention distribution
Figure 14: Attention visualization of the 72-frame Self Forcing model at Layer 23: “Super fast zoom
out from the peak of a frozen mountain where a lonely hiker is arriving to the summit.”
18

03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
0 36 7103671H8
0 36 71H9
0 36 71H10
0 36 71H11
002H0 H1 H2 H3
0.02.55.0H4 H5 H6 H7
0 7105H8
0 71H9
0 71H10
0 71H11
(a) Frame attention logits (b) Last-block attention distributionFigure 15: Attention visualization of the 72-frame Self Forcing model at Layer 23: “A white and
orange tabby cat is seen happily darting through a dense garden, cinematic warm tones, shallow depth
of field.”
03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
036 7103671H8
036 71H9
036 71H10
036 71H11
002H0 H1 H2 H3
05H4 H5 H6 H7
0 1190510H8
0 119H9
0 119H10
0 119H11
(a) Frame attention logits (b) Last-block attention distribution
Figure 16: Attention visualization of the 120-frame Self Forcing model at Layer 23: “A majestic
eagle soaring through a cloudy sky, cinematic lighting.”
03671H0
 H1
 H2
 H3
03671H4
 H5
 H6
 H7
0 36 7103671H8
0 36 71H9
0 36 71H10
0 36 71H11
0−202H0 H1 H2 H3
−20H4 H5 H6 H7
0 7102H8
0 71H9
0 71H10
0 71H11
(a) Frame attention logits (b) Last-block attention distribution
Figure 17: Attention visualization of the 72-frame Causal Forcing model at Layer 23: “A majestic
eagle soaring through a cloudy sky, cinematic lighting.”
19

A.7 Visualization of Classification Results
Figure 18: Aggregated classification results obtained via majority voting across 256 prompts (15s
video duration), using a period threshold of 6.4 and a sign rate threshold of 80%.
A.8 Periodicity Distribution across Additional Layers
During the FFT analysis stage, we process attention sequences for each of the 256 prompts across
30 layers and 12 heads per layer, using the frame range from 0 to 68. For each sequence, we first
apply a first-order difference to emphasize inter-frame variation, followed by mean removal and a
Hanning window. We then compute the spectrum with rfft , discard the zero-frequency component,
and convert frequencies into periods as period = 1/f . The spectral peak is not selected solely by
raw amplitude. Instead, we perform harmonic folding, where the energy from multiple harmonics
is folded back to the fundamental frequency with decaying weights. The period with the highest
folded score is selected as the top-1 period estimate for the corresponding head under each prompt.
Finally, the 256 prompt-level estimates are aggregated to obtain summary statistics, including the
mean, standard deviation, minimum/maximum values, and dominant period.
Distribution of Periodicity across Layers (L10 -L19) Distribution of Periodicity across Layers (L20 -L29)
Figure 19: The periodicity distribution across layers L10–L29 reveals that most attention heads
consistently concentrate around a dominant period of 6.0, providing a stable statistical basis for our
head-aware classification.
B Algorithmic Analysis
B.1 Pre-Softmax Logits for Head Classification
In constructing a classification system to identify attention heads, the choice of classification metrics
is crucial. In the standard attention mechanism, features are determined by normalized weights:
Attn Score=Softmax(z i) =ezi/X
jezj(6)
20

However, to better enhance the classification performance, we select theraw logits before the
Softmax operation(i.e.,z=QKT/√dk) as the basis for attention head classification.
From a geometric representation perspective, the logit space preserves thepolarity information
of the vectors, which is absent in normalized probability distributions. In the original dot-product
space, if the Query and Key vectors point in opposite directions (with an angle greater than 90◦), the
corresponding logit value is negative. Physically, this represents the model’s explicit inhibition of
specific redundant information.
However, the Softmax function restricts the codomain to (0,1) , forcibly converting this directional
“negative correlation inhibition” into positive probability weights. This transition from “directional
vector projection” to “positive probability measure” obscures the original intent of the attention
head to actively suppress invalid information. Therefore, utilizing logits for classification can more
authentically restore the polarity of the model during feature extraction, enabling a more accurate
functional definition of attention heads in long video generation.
B.2 Failure Cases of Head Classification
Classification relying solely on the FFT dominant period has inherent limitations. SomeAnchor
HeadsandVeil Headsmay exhibit minor periodic peaks; however, their logit sequences do not
possess stable alternating positive and negative values, manifesting instead as one-sided bias or
non-stationary fluctuations, as shown in Fig. 20a. If a simple periodicity threshold is applied, these
“pseudo-periodic” heads are easily misclassified asWave Heads, subsequently impacting the KVCache
strategy. To address this, we introduce theSign Rateto measure the balance of positive and negative
logits. GenuineWave Headstypically exhibit a low and relatively balanced sign rate, whereas
pseudo-periodicAnchor/Veil Headsare often dominated by a single sign, leading to a significantly
higher sign rate. Therefore, the sign rate threshold serves as a pre-filtering condition to effectively
exclude non-oscillatory low-period heads.
Nevertheless, a small number of heads remain near the classification boundaries for both period and
sign rate, making stable determination difficult based on these two metrics alone, as shown in Fig. 20b.
In response, we introduce theMean Logitas a fallback criterion.Anchor Headstypically maintain
strong positive attention toward historical frames, resulting in a higher average logit;Veil Heads
show weaker or even inhibitory attention toward intermediate history, leading to a lower average
logit; while the alternating positive and negative nature ofWave Headsbrings their mean value closer
to neutrality. Consequently, when both periodicity and sign rate are uncertain, performing fallback
classification based on the mean logit can reduce misclassification of boundary samples and enhance
the robustness of the classification system.
(a) Pseudo-periodic logit sequences showing one-
sided bias.
 (b) Boundary samples near the classification threshold.
Figure 20: Visual analysis of head classification failure cases. (a) shows heads with periodic peaks
but unbalanced sign rates; (b) illustrates samples requiring the Mean Logit fallback mechanism.
B.3 Error Accumulation in Autoregressive Video Diffusion Models
Autoregressive video diffusion models are typically modeled as a hybrid generative framework that
integrates chain decomposition with a denoising diffusion process. Formally, given a video frame
sequence x1:N= (x1, x2, . . . , xN)of length N, its joint distribution can be decomposed into the
product of conditional probabilities as follows:
p(x1:N) =NY
i=1p(xi|x<i)(7)
21

In the actual inference process, the generation of each frame xiis conditioned on the sequence of
previously generated historical frames x<i. However, since the model cannot access the ground-truth
observations from the training phase during inference and must instead rely on its own previously
produced predictions ˆx<i, this inevitably introduces the problem oferror accumulation. Under
this mechanism, a significant distributional discrepancy exists between the inference and training
phases: tiny numerical biases ϵgenerated at each step are propagated and accumulated along the
autoregressive chain, leading to non-linear amplification. As the generation sequence Textends,
these biases force the latent space representation to gradually deviate from the original data manifold
established during pre-training, resulting in severe semantic drift and visual degradation in long-range
video dependencies.
This error accumulation phenomenon is particularly pronounced in long video generation tasks. When
the generated temporal window exceeds the observational scope of the model’s pre-training, the noise
accumulated in the KVCache interferes with the precise retrieval of the attention mechanism, thereby
inducing object distortion or visual collapse.
B.4 Theoretical Analysis of Wave-Head Periodicity
To explain why Wave Heads exhibit a stable periodicity of approximately 6, we analyze the dominant
Rotary Positional Embedding (RoPE) mechanism. In mainstream autoregressive video models, the
relative position interaction between query and key is encoded via RoPE. For the i-th two-dimensional
subspace, the attention score incorporates a term related to the relative distance(t−s):
Score(t, s)∝cos(θ i(t−s)), θ i= 10000−2i/d(8)
where tandsdenote the current and historical frame indices. This structure inherently introduces an
oscillatory component into the attention weights along the temporal dimension.
Derivation of the Dominant PeriodIn practical models, low-frequency components ( θiwhere i
is small) dominate the attention distribution because they provide smoother gradients and are less
susceptible to noise during training. Specifically, for the first two-dimensional subspace ( i= 0 ), we
haveθ 0= 1. The corresponding oscillation term is:
f(t, s) = cos(t−s)(9)
The theoretical period of this continuous function is T= 2π≈6.283 . Under discrete temporal
sampling (integer frame indices), the model perceives the closest integer period:
Tperceived ≈round(2π) = 6(10)
While RoPE contains multiple frequency components cos(θ i(t−s)) , the high-frequency terms tend
to cancel out through averaging, leaving the low-frequency θ0as the primary determinant of the
attention signal.
Implications for KVCacheThis analysis confirms that Wave Heads are a direct manifestation of
RoPE’s low-frequency periodic structure in the attention space. This theoretical grounding directly
supports our KVCache strategy:
•Selection:Periodicity≈6serves as a robust criterion for identifying Wave Heads.
•Efficiency:For Wave Heads, retaining only key frames at intervals of T≈6 captures
the essential periodic features while eliminating redundant noise and reducing memory
accumulation.
C Limitations and Discussion
Pyramid Forcing designs differentiated KVCache policies for three types of attention heads, but it
still relies on manually specified rules, such as cache capacity, stride, and merge range. Future work
will explore online head identification, adaptive cache budget allocation, and learnable KVCache
retrieval and compression strategies, enabling the cache mechanism to dynamically adapt to attention
patterns and video content.
22