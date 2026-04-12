# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

**Authors**: Honghao Fu, Miao Xu, Yiwei Wang, Dailing Zhang, Liu Jun, Yujun Cai

**Published**: 2026-04-07 04:26:59

**PDF URL**: [https://arxiv.org/pdf/2604.05418v1](https://arxiv.org/pdf/2604.05418v1)

## Abstract
Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query's intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query's reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame-query intent alignment. Experiments show that VideoStir is competitive with state-of-the-art baselines without relying on auxiliary information, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning. Codes and checkpoints are available at Github.

## Full Text


<!-- PDF content starts -->

VideoStir: Understanding Long Videos via Spatio-Temporally Structured
and Intent-Aware RAG
Honghao Fu1, Miao Xu1, Yiwei Wang2, Dailing Zhang3, Liu Jun4, Yujun Cai1,5*
1University of Queensland,2University of California, Merced
3Institute of Automation, CAS,4Lancaster University,5Ant Group
{honghao.fu, miao.xu, yujun.cai}@uq.edu.au,yiweiwang2@ucmerced.edu
zhangdailing2023@ia.ac.cn,j.liu81@lancaster.ac.uk
Abstract
Scaling multimodal large language models
(MLLMs) to long videos is constrained by
limited context windows. While retrieval-
augmented generation (RAG) is a promising
remedy by organizing query-relevant visual ev-
idence into a compact context, most existing
methods (i) flatten videos into independent seg-
ments, breaking their inherent spatio-temporal
structure, and (ii) depend on explicit semantic
matching, which can miss cues that are implic-
itly relevant to the query’s intent. To overcome
these limitations, we propose VideoStir, a struc-
tured and intent-aware long-video RAG frame-
work. It firstly structures a video as a spatio-
temporal graph at clip level, and then performs
multi-hop retrieval to aggregate evidence across
distant yet contextually related events. Further-
more, it introduces an MLLM-backed intent-
relevance scorer that retrieves frames based on
their alignment with the query’s reasoning in-
tent. To support this capability, we curate IR-
600K, a large-scale dataset tailored for learning
frame–query intent alignment. Experiments
show that VideoStir is competitive with state-
of-the-art baselines without relying on auxil-
iary information, highlighting the promise of
shifting long-video RAG from flattened seman-
tic matching to structured, intent-aware reason-
ing. Codes and checkpoints are available at
https://github.com/RomGai/VideoStir.
1 Introduction
Video understanding has emerged as a core frontier
in multimodal intelligence [ 1]. While recent Mul-
timodal Large Language Models (MLLMs) have
achieved remarkable success in interpreting short
clips, extending this capability to long videos re-
mains a formidable challenge [ 2]. The primary
difficulty lies not merely in the video’s length, but
in its information density and structural complex-
ity [3]. Unlike short clips that typically encapsulate
*Corresponding Author.
Flatten Frames/Clips
Semantic -based
Retrieval
I can’t  answer  it from  current  cues. I only see 
someone  using  a printer,  not why. (Incorrect)
MLLMQuery: What is the purpose of the person 
using the printer  in the video?
…
Query Structured Intent -aware
Cues missing (could retrieve only “printer”)
…Structured Intent -aware
Cues with noises from weakly related contexts
Intent -aware
Retrieval
MLLM…
Query Structured Intent -aware
Clean cues from spatiotemporal contexts
Spatiotemporal
Graph
Flatten Frames/ClipsIntent -aware
Retrieval
MLLM
 Query 
It shows  him printing  test papers  and answer  
sheets  to distribute  in a classroom . (Incorrect)
It shows  him using  the printer  to produce  m-
aterials  for a poster . (Correct)(Ground Truth: Make a poster.)Figure 1:Paradigm shift from flat semantic matching to struc-
tured, intent-aware long-video RAG.(Top)Semantic-based
retrieval relies on explicit semantic overlap, often missing cues
that are only implicitly relevant to the query intent.(Middle)
Intent-aware retrieval utilizes MLLM’s reasoning capability
to identify cues that may be relevant to the query intent (de-
tailed in Sec. 3.4); however, it can be distracted by noisy cues
when operating over flattened contexts scattered across long
temporal ranges.(Bottom)VideoStir structures the video as a
spatio-temporal graph to retrieve cues from coherent contexts,
enabling more reliable evidence aggregation. We further dis-
cuss the benefits of structured retrieval in Fig. 3.
a single event, long videos consist of interleaved
narratives where critical cues could be sparsely
scattered across the timeline. To handle the exten-
sive contexts, state-of-the-art (SOTA) models often
resort to expanding context windows to support
uniform sampling at second level [ 4]. However,
such approaches face a trade-off: they could either
miss fine-grained details due to temporally sparse
sampling or struggle with accurate reasoning when
overwhelmed by redundant visual noise.
To address the contextual intractability of pro-
cessing long videos, Retrieval-Augmented Genera-arXiv:2604.05418v1  [cs.CV]  7 Apr 2026

tion (RAG) has emerged as a promising solution by
retrieving a small subset of segments to fit within
limited context windows [ 5]. However, current
long-video RAG methods typically flatten a video
into a set of independent segments for query-based
retrieval [ 6,7], and rank candidates primarily by
embedding similarity computed with contrastive
vision–language models (e.g., CLIP) [2, 8, 9]. Yet
this flattened semantic matching paradigm leaves
two critical gaps that hinder downstream reasoning.
The first gap isspatio-temporal structure de-
coupling. Flattening a video into ndependent seg-
ments disrupts its intrinsic spatio-temporal struc-
ture, decoupling contexts that should remain linked.
Because text queries may lack the fine-grained
cues needed to reconnect dispersed evidence, di-
rect matching often struggles to retrieve contexts
that are spatio-temporally relevant yet share lit-
tle explicit semantic overlap with the query, lim-
iting downstream models’ ability to form coher-
ent reasoning chains [ 3,10]. The second gap
isinsufficient intent modeling. Similarity com-
puted from contrastive embeddings is optimized
for vision–language alignment and tends to capture
“what looks similar” rather than “why it matters”
for addressing the query’s reasoning intent [ 11].
As shown in Figure 1, the query “For whatpurpose
does the recorder use theprinter?” leads a seman-
tic retriever to select frames showing the “printer”
due to high semantic overlap, whereas the actual
“purpose” is revealed in a different event (e.g., mak-
ing a poster), which semantic overlap alone fails to
surface.
In light of these gaps, we advocate two shifts
in long-video RAG paradigms:(i) From Flattened
to Structured: moving beyond isolated segment
retrieval to reconstruct the intrinsic spatio-temporal
topology of the video; and(ii) From Semantic to
Intent: going beyond surface semantic to model the
alignment between query’s intent and visual cues.
To realize these shifts, we proposeVideoStir, a
framework designed to advance long-video RAG
from flattened semantic matching toSTructured,
Intent-awareReasoning. Inspired by human
episodic memory where recall proceeds coarse-
to-fine (locating relevant episodes before exam-
ining details), VideoStir operates in two phases,
from clips to frames. First, a coarse clip-retrieval
phase addresses structure decoupling by represent-
ing the video as a spatio-temporal graph, whose
nodes are semantically coherent short clips seg-
mented by an event boundary detector. Temporaledges preserve narrative continuity, while spatial
edges bridge distant yet semantically related clips
via inter-clip similarity. Traversing this topology
enables multi-hop retrieval from query-matched
anchors to their spatio-temporal neighbors, lever-
aging intrinsic correlations among semantically
dense clips to aggregate dependencies and miti-
gate the semantic sparsity of direct query matching.
Second, a fine-grained frame-retrieval phase ad-
dresses insufficient intent modeling with anIntent-
Relevance scorer trained on our curatedIR-600K
dataset. Rather than considering semantic similar-
ity alone, this lightweight MLLM-based scorer es-
timates frame–intent alignment, ensuring retrieved
frames are not only semantically relevant but also
pivotal for downstream reasoning.
Consequently, VideoStir provides the down-
stream MLLM with spatio-temporally context-
coherent and intent-aligned visual cues, enabling
more accurate long-video understanding. The
scorer and curated IR-600K dataset further pro-
vide a reusable foundation for future research on
intent-oriented long-video RAG systems. Our con-
tributions are summarized as follows:
•We propose VideoStir, a novel long-video
RAG framework that leverages structured re-
trieval to reconstruct video’s spatio-temporal
topology and aggregate contextually coher-
ent evidence, which overcomes the spatio-
temporal decoupling of flattened retrieval.
•We introduce an intent-relevance scorer and
IR-600K, a large-scale dataset modeling the
relevance between frames and query intent.
This shifts retrieval from semantic matching
to intent awareness, establishing a foundation
for intent-oriented long-video RAG.
•Extensive experiments show that VideoStir
is competitive with SOTA baselines without
relying on auxiliary information, highlight-
ing the promise of the paradigm shift from
flattened semantic matching to structured and
intent-aligned reasoning in long-video RAG.
2 Related Works
2.1 Video-Language Models
Long-video understanding has attracted increas-
ing attention in multimodal intelligence. Re-
cent general-purpose MLLMs such as Gem-
ini [20] and Long-VITA [ 21], along with special-
ized video models including Video-XL [ 22] and

LongVLM [ 23], extend visual context to enable
hour-scale video reasoning via uniform sampling.
However, this paradigm faces a major bottleneck:
limited context windows necessitate sparse sam-
pling across the timeline, which can yield semanti-
cally redundant frames while simultaneously risk-
ing the loss of fleeting, query-critical cues. In par-
allel, contrastive video–language models [ 24] (e.g.,
Video-CLIP [ 25], X-CLIP [ 26], and PE [ 27]) ex-
tend CLIP-style contrastive embeddings to video,
enabling query-conditioned retrieval of salient clips
and frames. Nevertheless, contrastive objectives
primarily optimize for semantic similarity and may
miss cues that are implicitly relevant to the query
intent but lack an explicit semantic match.
2.2 Agentic Long-Video Understanding
To alleviate the contextual intractability inherent in
processing long-duration videos, a growing body
of recent work has shifted towards leveraging ex-
ternal systems. These systems are designed to
reconstruct and distill extensive long-range con-
text into compact, semantically rich cues tailored
for downstream MLLMs. A prominent trajec-
tory within this domain involves the exploration
of agentic frameworks [ 28,29], which employ
(M)LLMs as decision-making policies equipped
with advanced modules for captioning, multi-step
reasoning, and external memory [ 30]. By simulat-
ing human-like curation processes [ 31,32], these
agents dynamically filter irrelevant data. For exam-
ple, DrVideo [ 33] synthesizes disjoint video frames
into coherent long-form textual narratives to fa-
cilitate precise query answering, while Vgent [ 3]
abstracts complex entity interactions into struc-
tured symbolic representations to bolster reasoning.
However, despite their demonstrated efficacy in
enhancing understanding, deploying such sophis-
ticated MLLM-based policies incurs substantial
inference overhead.
2.3 RAG for Long-Video Understanding
To reduce the procedural inference overhead of
agentic workflows, another line of work explores
long-video retrieval-augmented generation (RAG)
frameworks [ 5,34,35], which externally select
cues to fit limited context windows [ 7,9,36]. For
example, Video-RAG [ 2] retrieves keyframes via
semantic similarity, and enriches entity information
with external tools like optical character recogni-
tion and object detectors. E-VRAG [ 37] introduces
a retrieval-reranking pipeline, leveraging Qwen3-Reranker-style binary judgments for fine-grained
selection. while AKS [ 6] selects keyframes by
jointly optimizing query-frame semantic similarity
and temporal uniformity. However, existing meth-
ods often treat videos as independent segments,
decoupling intrinsic spatio-temporal structure and
limiting coherent reasoning. Moreover, current
paradigms favor explicit semantic matching over
intent-level alignment, risking the omission of cues
implicitly relevant to the query intent.
3 Method
3.1 Overview
We propose VideoStir, a long-video RAG frame-
work that explicitly models video spatio-temporal
topology and query intent to retrieve pivotal ev-
idence, thereby advancing downstream MLLMs’
long-video understanding.
As illustrated in Figure 2, VideoStir comprises
three phases: (1) Spatio-temporal topology mod-
eling. It converts the input video into a struc-
tured graph. An event boundary detector parti-
tions the video into clip-level nodes connected by
temporal edges (chronological adjacency) and spa-
tial edges (inter-clip semantic similarity). This
design reconstructs the topology of the video’s
spatio-temporal context. (2) Graph-based clip re-
trieval. It first identifies anchor clips semantically
aligned with the query, then expands the context via
multi-hop traversal on the spatio-temporal graph.
Consequently, the retrieved evidence includes not
only query-matched moments but also their spatio-
temporally relevant neighbors that are critical for
reasoning. (3) Intent-aware frame retrieval. It re-
fines evidence at the frame level. Since semantic
similarity often fails to capture the query’s underly-
ing intent, we introduce an intent-relevance scorer.
Specifically, we leverage a lightweight MLLM fine-
tuned on our curated IR-600K dataset to assign
each frame a relevance score, computed as the
weighted expectation over predicted relevance lev-
els. It mitigates the loss of implicit cues inherent
to explicit semantic matching, yielding an intent-
aligned evidence set for downstream MLLMs.
3.2 Spatio-Temporal Topology Modeling
To structurally encode a long video’s spatio-
temporal topology, we model it as a graph G=
(V,E) , where each node vk∈ V corresponds to
a clip sk⊂S . These clips are obtained via an
event boundary detector, which preprocesses the

(b) Graph -based Clip Retrieval 
Multi -hop
TraversalText QueryVideo -Language
EncoderEvent Boundary
DetectorInput Video
Clips
Clip-level Emb.······ ···
·········
Spatio -Temporal
GraphTemporal Edges (chronological adjacency)Spatial Edges (inter -clip similarity)(a) Spatio -Temporal Topology Modeling
Video -Language
Encoder··· Query Emb.
Retrieved Anchor Node(s): 
Top-Nquery -matched clip(s)
Anchor
Retrieval
Retrieved Multi -hop Nodes :
Anchor neighbors filtered by distance/weight(c) Intent -aware Frame Retrieval
Frames
Downstream
MLLM (Any)AnswerIntent -Relevance Scorer
Logits + SoftmaxRelevance Level Tokens
Weighted Sum.Relevance 
Level Prob.
Relevance ScoreText QueryRetrieved 
Nodes (Clips)Dense Sampling
Score -based
Filtering
Text
QueryRelevance Scoring Prompt- How  intent -relevant  this 
frame  is for answering  the 
question : ‘{query}’?  
- Output  only one number  
from  1 to 5, where :
[1] = Completely  Irrelevant
[2] = Slightly  Relevant
[3] = Moderately  Relevant  
[4] = Mostly  Relevant  
[5] = Highly  Relevant
··
·
··
·
··
···
···
·
···
······
······
···
 …
… Intent -Aligned
FramesLightweight MLLMFigure 2:Overview of VideoStir.VideoStir achieves long-video understanding via spatio-temporal structuring and two-stage
retrieval.(a) Spatio-Temporal Topology Modeling.An event boundary detector segments input video into semantically
coherent clips. Their embeddings are used to construct a spatio-temporal graph with temporal edges linking adjacent clips and
spatial edges weighted by inter-clip similarity.(b) Graph-based Clip Retrieval. Input query is embedded in the video-language
space to retrieve top- Nmatched anchor nodes; multi-hop traversal then expands along spatio-temporal links under hop-distance
and edge-weight constraints to aggregate contextual clips.(c) Intent-aware Frame Retrieval.Frames densely sampled from the
retrieved clips are assessed by an intent-relevance scorer (LoRA-tuned on IR-600K), which outputs a probability distribution over
discrete relevance levels. These distributions are then aggregated via probability-weighted expectation to produce a continuous
relevance score; high-scoring, intent-aligned frames are retained and fed to a downstream MLLM to produce the final answer.
long video into semantically coherent segments
by identifying event transitions through seman-
tic change-point detection. This process is im-
plemented by applying PELT [ 45] to the frame
embeddings for adaptive segmentation. For each
clip, we compute an embedding hk=E vl(sk)
with a video–language encoder Evl. The edge
setE={E Temp,ESp}consists of temporal edges
ETemp ={(v k, vk+1)|w k,k+1 = 1} and spa-
tial edges ESp={(v i, vj)|w i,j= cos(h i, hj)},
where wdenotes the edge weight. Here, we refer
to the embedding space as the “spatial” dimension
of the topology. Temporal edges connect chrono-
logically adjacent clips, preserving the video’s tem-
poral backbone. Spatial edges are weighted by
the similarity between clip embeddings, captur-
ing inter-clip correlations when different moments
share similar video-level features. By jointly mod-
eling these edges, Ginduces a spatio-temporal
topology over clips: temporally adjacent events are
linked along timeline, while semantically coherent
yet temporally distant events are bridged via spatial
edges. This graph re-entangles contextual relations
that may be lost in a flattened representation and
provides the backbone for multi-hop retrieval.3.3 Graph-based Clip Retrieval
Given a query qand a spatio-temporal graph G, we
aim to coarsely retrieve clips that are semantically
relevant to qand topologically supported by inter-
clip correlations. This design addresses seman-
tically sparse query–clip matching: a query may
refer to only a small facet of an event (e.g., the main
subject), whereas long videos often contain recur-
ring scenes/actions/entities and temporally adjacent
evidence that constitute critical spatio-temporal
contexts for the underlying event, which can be
missed by simply computing semantic similarity
between the query and individual moments. As
shown in Figure 3, with our spatio-temporal graph,
these contexts can be recovered via inter-clip sim-
ilarity and chronological adjacency encoded by
spatio-temporal edges.
Specifically, we first compute the query embed-
dinghq=E vl(q)using the same video-language
encoder as for clips, placing queries and clip nodes
in a shared embedding space. We then measure
query-node similarity via cos(h q, hk), and select
the top- Nsimilar nodes as the anchor set Vanc.
While these anchors are matched to the query, they
may not fully cover their implicitly contextual de-
pendencies. To recover broader context, we expand

Vancalong spatio-temporal links in G, traversing
edges according to their weights and collecting
nodes withinLhops of the anchors:
Vhop={v j|d(v j,Vanc)≤L, w ij≥η}, (1)
where d(vj,Vanc)is the shortest hop distance from
vjto any anchor in Vanc, and threshold ηfilters out
weak connections. In this way, multi-hop traver-
sal yields a coarse spatio-temporal neighborhood
around the event hinted by the query, providing
contextual candidates for subsequent fine-grained
intent-level filtering.
3.4 Intent-aware Frame Retrieval
Intent Relevance Reasoning and Retrieval.
While multi-hop retrieval yields a set of contex-
tually related clips, the candidates may include re-
dundant or merely co-occurring frames that are se-
mantically relevant but misaligned with the query’s
underlying intent. We therefore introduce an intent-
relevance scorer Rθ, which filters retrieved clips
at frame level using the intent-reasoning capability
of a lightweight MLLM backbone πθ. Specifically,
Rθestimates frame-query intent relevance, sepa-
rating frames that genuinely support the query’s
intent from those with only superficial similarity.
Given the retrieved node set Vhop={v k}, where
each node represents a video clip, we expand them
into a dense frame pool: Fret=S
vk∈Vhop{xt|
xt∈v k}. For a query–frame pair (q, x t), the
scorer outputs a softmax-normalized probability
distribution over discrete intent-relevance level to-
kensℓ∈ {1,2,3,4,5}:
Pθ(ℓ|q, x t,Pintent) =exp(π θ(ℓ|q, x t,Pintent))P5
k=1exp(π θ(k|q, x t,Pintent)),
(2)
where πθ(·)denotes the logits produced by the
MLLM backbone of Rθ, andPintent (detailed in
Appendix E) denotes the prompt that guides the
MLLM backbone πθto reason and score intent rel-
evance. The score of frame xtis computed as the
weighted expectation over relevance level tokens:
rt=R θ(q, x t,Pintent) =P5
ℓ=1ℓ·P θ(ℓ|q, x t,Pintent).
(3)
This converts the model’s internal reasoning into a
continuous relevance score, distinguishing frames
that provide intent-level support for the query from
those that are merely semantically similar. After
obtaining scores for Fret, we retain frames exceed-
ing a relevance threshold κs. The resulting setFintent ={x t|rt> κ s}provides an intent-
aligned visual context for downstream reasoning.
Distillation-based Scorer Training.To strengthen
the scorer’s intent-relevance assessment, we adopt
a distillation strategy to fine-tune its MLLM back-
bone. Specifically, given a query-frame pair (q, x t)
and the intent-relevance scoring prompt Pintent,
a powerful teacher MLLM Tis asked to rate
the intent relevance between qandxton the
discrete level (1-5). The resulting pseudo-label
yt=T(q, x t,Pintent)serves as supervision for
the student scorer Rθ. During fine-tuning, we in-
ject low-rank adaptation (LoRA) parameters [ 56]
into the MLLM backbone of Rθ, enabling effi-
cient adaptation without substantially disrupting
its pretrained priors. Given the student predic-
tionPθ(ℓ|q, x t,Pintent), the training objective
minimizes the cross-entropy between the student-
predicted distribution and the teacher-provided dis-
crete label:
LCE=−5X
ℓ=11[ℓ=y t] logP θ(ℓ|q, x t,Pintent).(4)
The LoRA parameters are optimized via gradient
descent, enabling the student to inherit the teacher’s
relevance-assessment patterns and yield more ac-
curate relevance scores, while retaining the compu-
tational efficiency of its lightweight backbone.
4 Experiment
4.1 Dataset and Implementation
Dataset.We curate IR-600K, a dataset designed
to enhance small-scale MLLMs’ ability to judge
intent relevance between video frames and queries,
supporting intent-aware long-video RAG. To the
best of our knowledge, IR-600K is the first dataset
targeting intent-level frame-query alignment, i.e.,
whether a frame provides evidence that satisfies
the query’s underlying information need. This dif-
fers from prior datasets such as TVSum [ 62] and
QVHighlights [ 63], which primarily evaluate the
semantic saliency of frames with respect to descrip-
tive statements.
Specifically, we sample 4,678 videos from three
representative video QA datasets (ActivityNet-
QA [ 64], NExT-QA [ 65], and STAR [ 66]). For
each video, we uniformly extract frames at 3 fps
to form query–frame pairs. We then distill supervi-
sion from a powerful MLLM: Qwen2.5-VL-72B-
Instruct [ 67] serves as the teacher to produce frame-
level intent-relevance labels. Guided by a carefully

Table 1:Comparison with long-video RAG baselines. “Native Input” indicates whether the model processes video directly
without relying on external auxiliary text or tools. Bold and underlined numbers indicate the best and second-best accuracies,
respectively. ‘–’ indicates not applicable.
Model Native InputLV-Bench (Val) MLVU Video-MME-Long (w/o Sub)
Overall↑Gain (%)↑Overall↑Gain (%)↑Overall↑Gain (%)↑
LLaV A-Video (7B)✓56.6 - 70.8 - - -
+ Video-RAG✗58.7 3.7 72.4 2.3 - -
+ TV-RAG✗58.8 2.1 72.1 1.8 - -
+ VideoStir (Ours) ✓ 60.3 6.5 73.1 3.2 - -
mPLUG-Owl3 (8B)✓52.1 - 63.7 - 50.1 -
+ Video-RAG✗54.5 4.6 64.4 1.1 50.6 1.0
+ TV-RAG✗54.8 5.2 64.0 0.5 50.9 1.6
+ VideoStir (Ours) ✓ 55.1 5.8 64.6 1.4 51.0 1.8
Aria (25B)✓64.2 - 70.6 - 58.8 -
+ Video-RAG✗66.4 3.4 72.1 2.1 59.6 1.4
+ TV-RAG✗65.5 2.0 71.2 0.8 59.0 0.3
+ VideoStir (Ours) ✓ 65.8 2.5 71.7 1.6 59.4 1.0
InternVL-1.5 (26B)✓51.2 - 50.4 - 45.6 -
+ Video-RAG✗52.2 2.051.8 2.846.7 2.4
+ TV-RAG✗52.4 2.3 51.0 1.2 46.3 1.5
+ VideoStir (Ours) ✓ 52.7 2.9 51.6 2.4 46.9 2.9
LLaV A-Video (72B)✓61.9 - 73.1 - 61.5 -
+ Video-RAG✗65.4 5.7 73.8 1.0 62.3 1.3
+ TV-RAG✗64.6 4.3 73.4 0.4 61.8 0.5
+ VideoStir (Ours) ✓ 66.0 6.6 74.1 1.4 62.1 1.0
Table 2:System-level comparison on EgoSchema’s test set
with SOTA methods built upon fixed (M)LLMs. These ap-
proaches can be categorized into two groups: one employs
video-to-text tools (e.g., captioners or object detectors) to pro-
vide auxiliary textual inputs for downstream (M)LLMs, while
the other leverages the native multimodal understanding capa-
bility (Native Input) of MLLMs without auxiliary text. Bold
and underlined numbers indicate the best and second-best ac-
curacies, respectively. ‘–’ indicates not applicable.
(M)LLM Backbone Method Overall↑
Methods with Video-to-Text Tools
GPT-3.5LLoVi [57] 57.6
DrVideo [33] 62.6
GPT-4VideoAgent [58] 60.2
LLoVi [57] 61.2
VidAgent [59] 62.8
VideoTree [60] 66.2
DrVideo [33] 66.4
Qwen2-VL-7B Vgent [3]68.0
Methods with Native Input
Qwen2-VL-7BIG-VLM [61] 66.2
VideoStir (Ours) 67.2
designed system prompt specifying explicit intent-
relevance criteria, the teacher assesses intent-level
alignment between each frame and the query, yield-
ing intent-relevance scores. IR-600K comprises
605,676 training samples and 21,791 validation
samples. Further details on the intent-relevance
scoring prompt, dataset statistics and annotation
format are provided in Appendix D and E.Implementation.We adopt Qwen2.5-VL-3B-
Instruct [ 67] as the backbone of the intent-
relevance scorer and fine-tune it with LoRA
adapters with a rank of 16, a scaling factor of 32,
and a dropout of 0.05. We optimize the LoRA
using AdamW with a learning rate of 5×10−5
and a weight decay of 0.05, together with a cosine
learning-rate schedule and a warm-up ratio of 0.05.
Training is performed for 1 epoch with a batch
size of 128 under bf16 precision. For the event
boundary detector, we use the vision tower from
the Qwen2.5-VL series as the embedding backbone.
For spatio-temporal graph construction and multi-
hop retrieval, we use the Perception Encoder [ 27]
as the video-language encoder Evland set N= 3 ,
L= 2 ,η= 0.4 , and κs= 3.25 ; an ablation study
on these retrieval hyperparameters is provided in
Appendix C. All training runs and comparative ex-
periments are conducted on 8 ×A100 GPUs, while
the remaining experiments are conducted on a sin-
gle A100 GPU. Unless otherwise specified, results
are reported as the median of three trials, and ab-
lations are conducted on LLaV A-Video (7B) using
LongVideoBench [76].
4.2 Comparison Against Other Methods
Table 1 compares VideoStir with the native MLLM
and state-of-the-art (SOTA) long-video RAG
baselines on LongVideoBench (LV-Bench) [ 76],
MLVU [ 77], and Video-MME-Long [ 78], follow-

Table 3:Component-level ablations on the expanded 1k vali-
dation set. “Overall” denotes overall accuracy, and “Retrieval
Acc.” measures the accuracy of successfully retrieving key
cues. “w/o Intent-relevance Scorer” replaces the scorer with
similarity-based frame retrieval; “w/o Weighted Expectation”
replaces probability-weighted expectation with discrete scores;
“w/o Spatio-Temporal Graph” disables structured retrieval,
which bypasses the graph and directly selects the top-14 clips
by PE-embedding similarity, following the best-performing
setting in Table 6. Bold indicates the best results.
Method Overall↑Retrieval Acc.↑
w/o Intent-relevance Scorer
+ CLIP L/14 52.8 69.2
+ Video-CLIP 56.3 75.4
+ PE 58.1 79.8
w/o Weighted Expectation 54.2 71.6
w/o Event-Boundary Detector 62.3 88.2
w/o Spatio-Temporal Graph 56.4 74.8
w/o Spatial Edges 57.2 79.3
w/o Temporal Edges 59.8 83.4
Full 64.5 92.2
ing the same experimental setup as Video-RAG [ 2]
and TV-RAG [ 9]. Baseline MLLMs include GPT-
4o [79], LLaV A-Video (7B/72B) [ 80], mPLUG-
Owl3 (8B) [ 81], Aria (25B) [ 82], and InternVL-1.5
(26B) [ 83]. Results are compiled from the bench-
marks’ leaderboards and the original papers, with
the remaining results obtained from our reproduc-
tions based on their open-sourced codes and origi-
nal settings. In most cases, VideoStir outperforms
both the native model and prior long-video RAG
baselines, demonstrating highly competitive per-
formance. Notably, VideoStir preserves the na-
tive MLLM input setting by operating solely on
retrieved visual evidence: it introduces no auxil-
iary text, tool calls, or extra modalities, while still
achieving SOTA performance.
Table 2 compares VideoStir on EgoSchema
against specialized long-video systems built atop
fixed (M)LLMs [ 84]; baseline results are taken
from DrVideo [ 33] and Vgent [ 3]. Relying solely
on native multimodal inputs without external video-
to-text tools (e.g., captioners or object detectors),
VideoStir achieves performance comparable to
these specialized systems that leverage such auxil-
iary tools.
4.3 Ablation Study
Component-level ablations.Table 3 validates the
contributions of VideoStir’s key components. Re-
placing the intent–relevance scorer with similarity-
based frame retrieval results in substantial drops
in both overall performance and retrieval accuracy,
even when using strong vision–language embed-Table 4:Ablations of VideoStir’s video-language embed-
ding backbones with N= 1000 . “Overall” denotes overall
accuracy, and “Retrieval Acc.” refers to the accuracy of suc-
cessfully retrieving the key evidence for answering the query.
The bold numbers represent the best results.
Embedding Backbone Overall↑Retrieval Acc.↑
Video-CLIP [25] 62.1 88.3
X-CLIP [26] 62.8 89.4
PE[27]64.5 92.2
dings from a powerful encoder such as PE. This
indicates that pure semantic similarity matching
can overlook the critical evidence required for long-
video understanding, underscoring the importance
of intent-aware retrieval. Meanwhile, removing the
probability-weighted expectation (i.e., directly us-
ing discrete relevance levels) significantly degrades
performance, suggesting that distributional scoring
provides a smoother and better-calibrated ranking
signal for discriminating frames. Moreover, dis-
abling the spatio-temporal graph (i.e., flattening
cues instead of performing structured retrieval), or
removing spatial/temporal edges, markedly reduces
retrieval accuracy, confirming that multi-hop aggre-
gation over structured contexts is essential for ex-
tending evidence beyond the semantically matched
anchor clips. In addition, replacing the event-
boundary detector with uniform sampling that pro-
duces the same number of clips slightly degrades
performance, suggesting that event-boundary seg-
mentation isn’t the main performance driver, but it
still improves evidence retrieval accuracy.
Embedding backbones.Table 4 examines the
impact of embedding choices on graph-based clip
retrieval. Stronger video–language encoders yield
consistent improvements on downstream perfor-
mance. However, these gains are comparatively
modest relative to the contributions from the frame-
work’s core components, suggesting that Video-
Stir’s effectiveness is driven primarily by its struc-
tured design rather than by reliance on increasingly
powerful embedding models.
Scorer training strategy.Table 5 compares zero-
shot inference, LoRA tuning, and full-parameter
fine-tuning for the intent-relevance scorer. We eval-
uate different student-teacher pairs, including In-
ternVL2.5 [ 83], Qwen2.5-VL [ 67], and Qwen2-
VL [ 85]. For Qwen2-VL-2B, we increase the
LoRA rank to 32 (vs. 16 for others) to align train-
able parameter counts for fair comparison.
In the zero-shot setting, student’s performance
significantly behind teachers. Conversely, LoRA

Table 5:Ablations on the intent-relevant scorer’s distillation strategies. “IR-600K CE” and “LV-Bench CE” denote the
cross-entropy loss computed on the corresponding datasets. To reduce distributional bias, the student model is distilled using
teacher models from the same family. “Retrieval Acc.” refers to the accuracy of successfully retrieving the key evidence.
Base Model Method IR-600K CE↓LV-Bench CE↓Overall↑Retrieval Acc.↑Trainable Params↓
Teacher Models
InternVL2.5-78B Zero-Shot - - 64.8 92.6 -
Qwen2.5-VL-72B Zero-Shot - - 65.4 95.8 -
Qwen2-VL-72B Zero-Shot - - 64.9 94.7 -
Student Models
InternVL2.5-4BZero-Shot 8.8755 8.9423 59.3 82.1 -
LoRA 3.4614 3.8151 62.3 87.8 3.7M
Full Params 3.0238 3.2578 63.4 90.7 3.0B
Qwen2.5-VL-3BZero-Shot 8.0909 8.1742 61.2 85.8 -
LoRA 3.3101 3.6461 64.5 92.2 3.7M
Full Params 2.9127 3.1017 64.7 93.2 3.0B
Qwen2-VL-2BZero-Shot 9.3210 9.3872 58.7 80.4 -
LoRA 3.8127 4.3093 61.3 86.8 4.4M
Full Params 3.1846 3.5204 62.4 88.9 1.5B
tuning markedly reduces validation cross-entropy
on IR-600K and consistently improves LV-Bench
performance, demonstrating robust generalization.
Notably, LoRA captures the majority of full-param
tuning gains using only about 4M parameters, in
contrast to the billions required for full updates,
with the downstream performance of LoRA-tuned
Qwen2.5-VL-3B nearly matching the fully fine-
tuned model. Consequently, we adopt the Qwen2.5-
VL family for both teacher and student roles to max-
imize performance. These results confirm that IR-
600K provides effective, transferable supervision,
while LoRA offers an optimal accuracy–efficiency
trade-off for practical long-video RAG systems.
4.4 Discussion
Tables 1 and 2 present a comprehensive comparison
between VideoStir and SOTA baselines, including
those that utilize external expert modules (e.g., cap-
tioners, object detectors, and OCR models) to aug-
ment video with auxiliary text. Our results indicate
that VideoStir achieves superior performance using
only video frames as native inputs, effectively by-
passing the need for external textual augmentation.
This suggests that one of the primary bottlenecks in
long-video understanding still lies in the retrieval
and organization of pivotal visual evidence. There-
fore, refining the selection of visual evidence is es-
sential for scaling MLLM’s long-video understand-
ing capabilities. VideoStir addresses this by en-
hancing both the relevance and spatio-temporal co-
herence of retrieved evidence, thereby facilitating
more reliable reasoning in downstream MLLMs.5 Conclusion
We propose VideoStir, a novel long-video RAG
framework designed to address the limitations of
contextual structure decoupling and intent misalign-
ment prevalent in current paradigms. VideoStir
models long videos as spatio-temporal graphs and
utilizes multi-hop traversal to recover the intrin-
sic topology of their contexts, thereby aggregating
spatio-temporally coherent evidence. Additionally,
we introduce an intent-relevance scorer trained on
our newly curated IR-600K dataset, which further
filters visual content by capturing the query’s under-
lying reasoning intent beyond surface-level seman-
tics. Both the dataset and the scorer provide a solid
foundation for future research into intent-oriented
long-video RAG. Extensive experiments show that
VideoStir is competitive with SOTA baselines with-
out relying on auxiliary information, underscoring
the effectiveness of shifting from flattened seman-
tic matching to structured, intent-aware reasoning,
and further revealing that one of the main bottle-
necks in long-video understanding lies in retrieving
and organizing pivotal visual evidence.
Limitations
As VideoStir introduces an additional step to trans-
form long videos into structured representations, it
inevitably incurs system latency. Indeed, system
latency remains a broad challenge for long-video
understanding frameworks that rely on external
components like agents or RAG pipelines. We
regard further reducing end-to-end latency as a piv-
otal direction for future long-video RAG research.

References
[1]KunChang Li, Yinan He, Yi Wang, Yizhuo Li,
Wenhai Wang, Ping Luo, Yali Wang, Limin Wang,
and Yu Qiao. Videochat: Chat-centric video un-
derstanding. Science China Information Sciences ,
68(10):200102, 2025.
[2]Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang
Yin, Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi
Ji, Fei Chao, Jiebo Luo, et al. Video-rag: Visually-
aligned retrieval-augmented long video compre-
hension. NeurIPS 2025, 2025.
[3]Xiaoqian Shen, Wenxuan Zhang, Jun Chen,
and Mohamed Elhoseiny. Vgent: Graph-
based retrieval-reasoning-augmented generation
for long video understanding. arXiv preprint
arXiv:2510.14032, 2025.
[4]Gheorghe Comanici, Eric Bieber, Mike Schaek-
ermann, Ice Pasupat, Noveen Sachdeva, Inderjit
Dhillon, Marcel Blistein, Ori Ram, Dan Zhang,
Evan Rosen, et al. Gemini 2.5: Pushing the fron-
tier with advanced reasoning, multimodality, long
context, and next generation agentic capabilities.
arXiv preprint arXiv:2507.06261, 2025.
[5]Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang
Wang, Dawei Yin, and Chao Huang. Vide-
orag: Retrieval-augmented generation with ex-
treme long-context videos. arXiv preprint
arXiv:2502.01549, 2025.
[6]Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jian-
bin Jiao, and Qixiang Ye. Adaptive keyframe
sampling for long video understanding. In
Proceedings oftheComputer Vision andPattern
Recognition Conference , pages 29118–29128,
2025.
[7]Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kan-
chan Sarkar, Zhenheng Yang, and Yang You. Fo-
cus: Efficient keyframe selection for long video
understanding. arXiv preprint arXiv:2510.27280 ,
2025.
[8]Alec Radford, Jong Wook Kim, Chris Hallacy,
Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin,
Jack Clark, et al. Learning transferable vi-
sual models from natural language supervision.
InInternational conference onmachine learning ,
pages 8748–8763. PmLR, 2021.
[9]Zongsheng Cao, Yangfan He, Anran Liu, Jun
Xie, Feng Chen, and Zhepeng Wang. Tv-rag: A
temporal-aware and semantic entropy-weighted
framework for long video retrieval and under-
standing. In Proceedings ofthe33rd ACM
International Conference onMultimedia , pages
9071–9079, 2025.
[10] Ruotong Liao, Max Erler, Huiyu Wang, Guangyao
Zhai, Gengyuan Zhang, Yunpu Ma, and V olkerTresp. Videoinsta: Zero-shot long video under-
standing via informative spatial-temporal reason-
ing with llms. In Findings oftheAssociation for
Computational Linguistics: EMNLP 2024 , pages
6577–6602, 2024.
[11] Kaihang Pan, Juncheng Li, Wenjie Wang, Hao Fei,
Hongye Song, Wei Ji, Jun Lin, Xiaozhong Liu,
Tat-Seng Chua, and Siliang Tang. I3: I ntent-i
ntrospective retrieval conditioned on i nstructions.
InProceedings ofthe47th International ACM
SIGIR Conference onResearch andDevelopment
inInformation Retrieval , pages 1839–1849, 2024.
[12] Yubin Gu, Yuan Meng, Siting Chen, Jiayi Ji, Xi-
aoshuai Sun, Weijian Ruan, and Rongrong Ji.
Sfir: Optimizing spatial and frequency domains
for image restoration. Pattern Recognition , page
112188, 2025.
[13] Yubin Gu, Yuan Meng, Kaihang Zheng, Xiaoshuai
Sun, Jiayi Ji, Weijian Ruan, Liujuan Cao, and
Rongrong Ji. An efficient and mixed heteroge-
neous model for image restoration. arXiv preprint
arXiv:2504.10967, 2025.
[14] Honghao Fu, Yilang Shen, Yuxuan Liu, Jingzhong
Li, and Xiang Zhang. Sgcn: a multi-order
neighborhood feature fusion landform classifi-
cation method based on superpixel and graph
convolutional network. International Journal of
Applied Earth Observation andGeoinformation ,
122:103441, 2023.
[15] Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong
Bi, Yuyao Ge, Jun Wan, Yurong Wu, and Xueqi
Cheng. a1: Steep test-time scaling law via en-
vironment augmented generation. arXiv preprint
arXiv:2504.14597, 2025.
[16] Hang Wu, Yujun Cai, Zehao Li, Haonan Ge,
Bowen Sun, Junsong Yuan, and Yiwei Wang.
Camreasoner: Reinforcing camera movement un-
derstanding via structured spatial reasoning. arXiv
preprint arXiv:2602.00181, 2026.
[17] Jianing Chen, Zehao Li, Yujun Cai, Hao Jiang,
Shuqin Gao, Honglong Zhao, Tianlu Mao, and
Yucheng Zhang. From tokens to nodes: Semantic-
guided motion control for dynamic 3d gaussian
splatting, 2025. URL https://arxiv.org/abs/
2510.02732.
[18] Lingrui Mei, Shenghua Liu, Yiwei Wang, Bao-
long Bi, Jiayi Mao, and Xueqi Cheng. "not
aligned" is not" malicious": Being careful about
hallucinations of large language models’ jailbreak.
COLING 2025, 2024.
[19] Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, and
Yujun Cai. Mrfd: Multi-region fusion decoding
with self-consistency for mitigating hallucinations
in lvlms, 2025. URL https://arxiv.org/abs/
2508.10264.

[20] Rohan Anil, Sebastian Borgeaud, Jean-Baptiste
Alayrac, Jiahui Yu, Radu Soricut, Johan Schalk-
wyk, Andrew M Dai, Anja Hauth, Katie Millican,
et al. Gemini: a family of highly capable multi-
modal models. arXiv preprint arXiv:2312.11805 ,
2023.
[21] Yunhang Shen, Chaoyou Fu, Shaoqi Dong, Xiong
Wang, Yi-Fan Zhang, Peixian Chen, Mengdan
Zhang, Haoyu Cao, Ke Li, Shaohui Lin, et al.
Long-vita: Scaling large multi-modal models to 1
million tokens with leading short-context accuracy.
arXiv preprint arXiv:2502.05177, 2025.
[22] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin,
Junjie Zhou, Zhengyang Liang, Tiejun Huang,
and Bo Zhao. Video-xl: Extra-long vision lan-
guage model for hour-scale video understand-
ing. In Proceedings oftheComputer Vision and
Pattern Recognition Conference , pages 26160–
26169, 2025.
[23] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun
Chang, and Bohan Zhuang. Longvlm: Efficient
long video understanding via large language mod-
els. In European Conference onComputer Vision ,
pages 453–470. Springer, 2024.
[24] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen,
Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An
empirical study of clip for end to end video clip
retrieval. arXiv preprint arXiv:2104.08860, 2021.
[25] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro
Okhonko, Armen Aghajanyan, Florian Metze,
Luke Zettlemoyer, and Christoph Feichtenhofer.
Videoclip: Contrastive pre-training for zero-shot
video-text understanding. In Proceedings of
the2021 Conference onEmpirical Methods in
Natural Language Processing , pages 6787–6800,
2021.
[26] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming
Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-
end multi-grained contrastive learning for video-
text retrieval. In Proceedings ofthe30th ACM
international conference onmultimedia , pages
638–647, 2022.
[27] Daniel Bolya, Po-Yao Huang, Peize Sun,
Jang Hyun Cho, Andrea Madotto, Chen Wei,
Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran,
Hanoona Rasheed, et al. Perception encoder: The
best visual embeddings are not at the output of the
network. arXiv preprint arXiv:2504.13181, 2025.
[28] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching
Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng
Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al.
Mm-vid: Advancing video understanding with
gpt-4v (ision). arXiv preprint arXiv:2310.19773 ,
2023.
[29] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo
Ma, and Kyusong Lee. Omagent: A multi-modalagent framework for complex video understand-
ing with task divide-and-conquer. In Proceedings
ofthe2024 Conference onEmpirical Methods
inNatural Language Processing , pages 10031–
10045, 2024.
[30] Hongbo Jin, Qingyuan Wang, Wenhao Zhang,
Yang Liu, and Sijie Cheng. Videomem: En-
hancing ultra-long video understanding via adap-
tive memory management. arXiv preprint
arXiv:2512.04540, 2025.
[31] Shijie Wang, Qi Zhao, Minh Quan Do, Nakul
Agarwal, Kwonjoon Lee, and Chen Sun. Va-
mos: Versatile action models for video under-
standing. In European Conference onComputer
Vision, pages 142–160. Springer, 2024.
[32] Juhong Min, Shyamal Buch, Arsha Nagrani,
Minsu Cho, and Cordelia Schmid. Morevqa:
Exploring modular reasoning models for video
question answering. In Proceedings ofthe
IEEE/CVF Conference onComputer Vision and
Pattern Recognition, pages 13235–13245, 2024.
[33] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin
Sun, Shutao Li, Hamid Rezatofighi, and Jian-
fei Cai. Drvideo: Document retrieval based
long video understanding. In Proceedings of
theComputer Vision and Pattern Recognition
Conference, pages 18936–18946, 2025.
[34] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yux-
uan Cai, Yong Liu, Xiangtai Li, and Dacheng
Tao. Omni-adavideorag: Omni-contextual adap-
tive retrieval-augmented for efficient long video
understanding. arXiv preprint arXiv:2506.13589 ,
2025.
[35] Nianbo Zeng, Haowen Hou, Fei Richard Yu,
Si Shi, and Ying Tiffany He. Scenerag: Scene-
level retrieval-augmented generation for video un-
derstanding. arXiv preprint arXiv:2506.07600 ,
2025.
[36] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and
Sung Ju Hwang. Videorag: Retrieval-augmented
generation over video corpus. arXiv preprint
arXiv:2501.05874, 2025.
[37] Zeyu Xu, Junkang Zhang, Qiang Wang, and Yi Liu.
E-vrag: Enhancing long video understanding with
resource-efficient retrieval augmented generation.
arXiv preprint arXiv:2508.01546, 2025.
[38] Yubin Gu, Honghui Xu, Yueqian Quan, Wan-
jun Chen, and Jianwei Zheng. Orsi salient ob-
ject detection via bidimensional attention and full-
stage semantic guidance. IEEE Transactions on
Geoscience andRemote Sensing, 61:1–13, 2023.
[39] Yuyao Ge, Shenghua Liu, Yiwei Wang, Lin-
grui Mei, Lizhe Chen, Baolong Bi, and Xueqi
Cheng. Innate reasoning is not enough: In-
context learning enhances reasoning large lan-
guage models with less overthinking. arXiv
preprint arXiv:2503.19602, 2025.

[40] Yubin Gu, Yuan Meng, Xiaoshuai Sun, Jiayi Ji,
Weijian Ruan, and Rongrong Ji. Mixed degrada-
tion image restoration via local dynamic optimiza-
tion and conditional embedding. arXiv preprint
arXiv:2411.16217, 2024.
[41] Honghao Fu, Hao Wang, Jing Jih Chin, and Zhiqi
Shen. Brainvis: Exploring the bridge between
brain and visual signals via image reconstruc-
tion. In ICASSP 2025-2025 IEEE International
Conference onAcoustics, Speech and Signal
Processing (ICASSP), pages 1–5. IEEE, 2025.
[42] Junlong Ren, Gangjian Zhang, Honghao Fu,
Pengcheng Wu, and Hao Wang. Wamo: Wavelet-
enhanced multi-frequency trajectory analysis for
fine-grained text-motion retrieval. arXiv preprint
arXiv:2508.03343, 2025.
[43] Honghao Fu, Yongli Gu, Yidong Yan, Yilang Shen,
Yiwen Wu, and Libo Sun. Sdr-gain: A high real-
time occluded pedestrian pose completion method
for autonomous driving. IEEE Transactions
onIntelligent Transportation Systems , 27(1):972–
982, 2025.
[44] Zhen Xiong, Yujun Cai, Zhecheng Li, Junsong
Yuan, and Yiwei Wang. Thinking with sound:
Audio chain-of-thought enables multimodal rea-
soning in large audio-language models. arXiv
preprint arXiv:2509.21749, 2025.
[45] Rebecca Killick, Paul Fearnhead, and Idris A
Eckley. Optimal detection of changepoints with
a linear computational cost. Journal ofthe
American Statistical Association , 107(500):1590–
1598, 2012.
[46] Yubin Gu, Siting Chen, Xiaoshuai Sun, Jiayi Ji,
Yiyi Zhou, and Rongrong Ji. Optical remote sens-
ing image salient object detection via bidirectional
cross-attention and attention restoration. Pattern
Recognition, 164:111478, 2025.
[47] Zhen Xiong, Yujun Cai, Zhecheng Li, and Yiwei
Wang. Unveiling the potential of diffusion large
language model in controllable generation. arXiv
preprint arXiv:2507.04504, 2025.
[48] Hang Wu, Yujun Cai, Haonan Ge, Hongkai Chen,
Ming-Hsuan Yang, and Yiwei Wang. Refineshot:
Rethinking cinematography understanding with
foundational skill evaluation. arXiv preprint
arXiv:2510.02423, 2025.
[49] Yubin Gu, Yuan Meng, Jiayi Ji, and Xiaoshuai
Sun. Acl: Activating capability of linear at-
tention for image restoration. In Proceedings
oftheComputer Vision andPattern Recognition
Conference, pages 17913–17923, 2025.
[50] Honghao Fu, Yuan Ouyang, Kai-Wei Chang, Yi-
wei Wang, Zi Huang, and Yujun Cai. Contextnav:
Towards agentic multimodal in-context learning.
arXiv preprint arXiv:2510.04560, 2025.[51] Xin Liu, Weijia Zhang, Wei Tang, Thuc Duy Le,
Jiuyong Li, Lin Liu, and Min-Ling Zhang. From
correlation to causation: Max-pooling-based multi-
instance learning leads to more robust whole slide
image classification, 2025. URL https://arxiv.
org/abs/2408.09449.
[52] Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang
Wu, and Yujun Cai. Framemind: Frame-
interleaved video reasoning via reinforcement
learning. arXiv preprint arXiv:2509.24008, 2025.
[53] Xin Liu, Weijia Zhang, and Min-Ling Zhang.
Hacsurv: A hierarchical copula-based approach
for survival analysis with dependent competing
risks. In International Conference onArtificial
Intelligence and Statistics , pages 3079–3087.
PMLR, 2025.
[54] Zhifang Zhang, Shuo He, Haobo Wang, Bingquan
Shen, and Lei Feng. Defending multimodal back-
doored models by repulsive visual prompt tuning.
NeurIPS, 2025.
[55] Zhifang Zhang, Yuwei Niu, Xin Liu, and Beibei
Li. Tuning vision-language models with candidate
labels by prompt alignment. In DASFAA, 2025.
[56] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of
large language models. ICLR, 1(2):3, 2022.
[57] Ce Zhang, Taixi Lu, Md Mohaiminul Islam,
Ziyang Wang, Shoubin Yu, Mohit Bansal, and
Gedas Bertasius. A simple llm framework
for long-range video question-answering. In
Proceedings ofthe2024 Conference onEmpirical
Methods inNatural Language Processing , pages
21715–21737, 2024.
[58] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and
Serena Yeung-Levy. Videoagent: Long-form
video understanding with large language model
as agent. In European Conference onComputer
Vision, pages 58–76. Springer, 2024.
[59] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi
Li, Zhi Gao, and Qing Li. Videoagent: A memory-
augmented multimodal agent for video under-
standing. In European Conference onComputer
Vision, pages 75–92. Springer, 2024.
[60] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin,
Jaehong Yoon, Feng Cheng, Gedas Bertasius, and
Mohit Bansal. Videotree: Adaptive tree-based
video representation for llm reasoning on long
videos. In Proceedings oftheComputer Vision
andPattern Recognition Conference , pages 3272–
3283, 2025.
[61] Wonkyun Kim, Changin Choi, Wonseok Lee, and
Wonjong Rhee. An image grid can be worth a
video: Zero-shot video question answering using
a vlm. IEEE Access, 2024.

[62] Yale Song, Jordi Vallmitjana, Amanda Stent,
and Alejandro Jaimes. Tvsum: Summarizing
web videos using titles. In Proceedings ofthe
IEEE conference oncomputer vision andpattern
recognition, pages 5179–5187, 2015.
[63] Jie Lei, Tamara L Berg, and Mohit Bansal.
Detecting moments and highlights in videos
via natural language queries. Advances in
Neural Information Processing Systems , 34:
11846–11858, 2021.
[64] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao,
Yueting Zhuang, and Dacheng Tao. Activitynet-qa:
A dataset for understanding complex web videos
via question answering. In Proceedings ofthe
AAAI Conference onArtificial Intelligence , vol-
ume 33, pages 9127–9134, 2019.
[65] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-
Seng Chua. Next-qa: Next phase of question-
answering to explaining temporal actions. In
Proceedings oftheIEEE/CVF conference on
computer vision andpattern recognition , pages
9777–9786, 2021.
[66] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B
Tenenbaum, and Chuang Gan. Star: A benchmark
for situated reasoning in real-world videos. arXiv
preprint arXiv:2405.09711, 2024.
[67] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang,
Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shi-
jie Wang, Jun Tang, et al. Qwen2. 5-vl technical
report. arXiv preprint arXiv:2502.13923, 2025.
[68] Honghao Fu, Yufei Wang, Wenhan Yang, Alex C
Kot, and Bihan Wen. Dp-iqa: Utilizing diffusion
prior for blind image quality assessment in the
wild. arXiv preprint arXiv:2405.19996, 2024.
[69] Yongli Gu, Xiang Yan, Hanlin Qin, Naveed Akhtar,
Shuai Yuan, Honghao Fu, Shuowen Yang, and Aj-
mal Mian. Hdtcnet: A hybrid-dimensional convo-
lutional network for multivariate time series classi-
fication. Pattern Recognition, 168:111837, 2025.
[70] Zhifang Zhang, Qiqi Tao, Jiaqi Lv, Na Zhao, Lei
Feng, and Joey Tianyi Zhou. Tokenswap: Back-
door attack on the compositional understanding
of large vision-language models. arXiv preprint
arXiv:2509.24566, 2025.
[71] Zhecheng Li, Yiwei Wang, Bryan Hooi, Yujun Cai,
Nanyun Peng, and Kai-Wei Chang. Drs: Deep
question reformulation with structured output. In
Association forComputational Linguistics ACL,
2025., 2024.
[72] Zhecheng Li, Guoxian Song, Yujun Cai, Zhen
Xiong, Junsong Yuan, and Yiwei Wang. Texture
or semantics? vision-language models get lost
in font recognition. In Conference onLanguage
Modeling COLM, 2025., 2025.[73] Jianing Chen, Zehao Li, Yujun Cai, Hao Jiang,
Chengxuan Qian, Juyuan Kang, Shuqin Gao, Hon-
glong Zhao, Tianlu Mao, and Yucheng Zhang.
Haif-gs: Hierarchical and induced flow-guided
gaussian splatting for dynamic scene. In NeurIPS
2025, 2025.
[74] Zhifang Zhang, Jiahan Zhang, Shengjie Zhou,
Qi Wei, Shuo He, Feng Liu, and Lei Feng. Im-
proving generalizability and undetectability for
targeted adversarial attacks on multimodal pre-
trained models. arXiv preprint arXiv:2509.19994 ,
2025.
[75] Lingrui Mei, Shenghua Liu, Yiwei Wang, Yuyao
Ge, Baolong Bi, Jiayu Yao, Jun Wan, Ziling Yin,
Jiafeng Guo, and Xueqi Cheng. Gated differ-
entiable working memory for long-context lan-
guage modeling, 2026. URL https://arxiv.
org/abs/2601.12906.
[76] Haoning Wu, Dongxu Li, Bei Chen, and Junnan
Li. Longvideobench: A benchmark for long-
context interleaved video-language understand-
ing. Advances inNeural Information Processing
Systems, 37:28828–28857, 2024.
[77] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu,
Zhengyang Liang, Shitao Xiao, Minghao Qin,
Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu:
Benchmarking multi-task long video understand-
ing. In Proceedings oftheComputer Vision and
Pattern Recognition Conference , pages 13691–
13701, 2025.
[78] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li,
Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu
Zhou, Yunhang Shen, Mengdan Zhang, et al.
Video-mme: The first-ever comprehensive evalua-
tion benchmark of multi-modal llms in video anal-
ysis. In Proceedings oftheComputer Vision and
Pattern Recognition Conference , pages 24108–
24118, 2025.
[79] Aaron Hurst, Adam Lerer, Adam P Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, et al. Gpt-4o system card. arXiv preprint
arXiv:2410.21276, 2024.
[80] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun
Ma, Ziwei Liu, and Chunyuan Li. Video instruc-
tion tuning with synthetic data. arXiv preprint
arXiv:2410.02713, 2024.
[81] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu,
Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and
Jingren Zhou. mplug-owl3: Towards long image-
sequence understanding in multi-modal large lan-
guage models. arXiv preprint arXiv:2408.04840 ,
2024.
[82] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang,
Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou,
Chengen Huang, Yanpeng Li, et al. Aria: An
open multimodal native mixture-of-experts model.
arXiv preprint arXiv:2410.05993, 2024.

[83] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong
Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong,
Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al.
How far are we to gpt-4v? closing the gap to
commercial multimodal models with open-source
suites. Science China Information Sciences , 67
(12):220101, 2024.
[84] Karttikeya Mangalam, Raiymbek Akshulakov,
and Jitendra Malik. Egoschema: A diagnostic
benchmark for very long-form video language
understanding. Advances inNeural Information
Processing Systems, 36:46212–46244, 2023.
[85] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang,
Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing
Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-
vl: Enhancing vision-language model’s percep-
tion of the world at any resolution. arXiv preprint
arXiv:2409.12191, 2024.
[86] Yuyao Ge, Shenghua Liu, Baolong Bi, Yiwei
Wang, Lingrui Mei, Wenjie Feng, Lizhe Chen,
and Xueqi Cheng. Can graph descriptive order
affect solving graph problems with llms? ACL
2025, 2024.
[87] Honghao Fu, Junlong Ren, Qi Chai, Deheng Ye,
Yujun Cai, and Hao Wang. Vistawise: Building
cost-effective agent with cross-modal knowledge
graph for minecraft. In Proceedings ofthe2025
Conference onEmpirical Methods inNatural
Language Processing, pages 21895–21909, 2025.
[88] Zhifang Zhang, Bojun Yang, Shuo He, Weitong
Chen, Wei Emma Zhang, Olaf Maennel, Lei Feng,
, and Miao Xu. Test-time attention purification
for backdoored large vision language models. In
CVPR, 2026.
[89] Chang Liu, Hongkai Chen, Yujun Cai, Hang
Wu, Qingwen Ye, Ming-Hsuan Yang, and Yiwei
Wang. Structured attention matters to multimodal
llms in document understanding. arXiv preprint
arXiv:2506.21600, 2025.
[90] Xinpeng Ti, Wentao Ye, Zhifang Zhang, Junbo
Zhao, Chang Yao, Lei Feng, and Haobo Wang.
Towards reverse engineering of language mod-
els: A survey. In Findings oftheAssociation for
Computational Linguistics: EMNLP 2025, 2025.
[91] Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang,
Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li,
Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi
Mao, Tianze Xia, Jiafeng Guo, and Shenghua Liu.
A survey of context engineering for large language
models, 2025. URL https://arxiv.org/abs/
2507.13334.
[92] Hang Wu, Hongkai Chen, Yujun Cai, Chang Liu,
Qingwen Ye, Ming-Hsuan Yang, and Yiwei Wang.
Dimo-gui: Advancing test-time scaling in gui
grounding via modality-aware visual reasoning.
InEMNLP 2025, 2025.
Query: What is the purpose of the person 
using the printer in the video?
Spatiotemporal Contexts are Missing
 Flatten 
Clips
···
Spatiotemporal
Graph
Multi -hop
TraversalAnchor
RetrievalSemantic -based 
Retrieval
Query 
Reconstructed Spatiotemporal Context
Structured
Structured
Query (Ground Truth: Make a poster.)Figure 3:Advantages of structured retrieval over flat clip re-
trieval. Flattened retrieval mainly returns clips with explicit se-
mantic overlap with the query, while overlooking clips that are
contextually relevant but do not contain direct query-matching
content. VideoStir structures long videos to reconstruct spatio-
temporal context, allowing multi-hop traversal to aggregate
contextually related clips around key moments. The resulting
evidence is cleaner and more coherent, providing stronger
support for subsequent fine-grained retrieval.
[93] Lingrui Mei, Shenghua Liu, Yiwei Wang, Bao-
long Bi, Ruibin Yuan, and Xueqi Cheng. Hid-
denguard: Fine-grained safe generation with spe-
cialized representation router. arXiv preprint
arXiv:2410.02684, 2024.
[94] Zhecheng Li, Yiwei Wang, Bryan Hooi, Yujun Cai,
Zhen Xiong, Nanyun Peng, and Kai-Wei Chang.
Vulnerability of llms to vertically aligned text ma-
nipulations. In Association forComputational
Linguistics ACL, 2025., 2024.
[95] Zhecheng Li, Yiwei Wang, Bryan Hooi, Yujun
Cai, Naifan Cheung, Nanyun Peng, and Kai-
Wei Chang. Think carefully and check again!
meta-generation unlocking llms for low-resource
cross-lingual summarization. arXiv preprint
arXiv:2410.20021, 2024.
[96] Yuyao Ge, Shenghua Liu, Yiwei Wang, Lingrui
Mei, Baolong Bi, Xuanshan Zhou, Jiayu Yao, Ji-
afeng Guo, and Xueqi Cheng. Focusing by con-
trastive attention: Enhancing vlms’ visual reason-
ing, 2025. URL https://arxiv.org/abs/2509.
06461.
A The Use of LLMs
The paper employs LLMs for language polishing.
B Extended discussion for Structured
Retrieval
In Figure 3, we further illustrate and discuss the
advantages of structured retrieval via visualizations.

C Hyperparameter Ablations
We determine the hyperparameters for the graph-
based clip retrieval procedure based on the ablation
study in Table 6.
Table 6:Hyperparameter ablations on Graph-based Clip
Retrieval. We investigate the impact of the number of anchors
N, hop distance L, and spatial edge threshold η. “Avg. Clips”
denotes the average number of retrieved clips per query. The
bold numbers represent the best results.
Hyperparameter Value Avg. Clips Retrieval Acc.↑
Number of Anchors (N)
(FixedL= 2, η= 0.6)1 7.2 88.0
213.692.0
3 21.9 90.5
4 25.4 90.0
Hop Distance (L)
(FixedN= 2, η= 0.6)1 8.4 87.0
213.692.0
3 16.6 92.0
4 18.1 91.0
Edge Weight Threshold (η)
(FixedN= 2, L= 2)0.2 31.4 84.0
0.4 20.6 90.5
0.613.692.0
0.8 6.8 82.5
D Supplemental Details for IR-600K
D.1 Annotation Example
We provide an example annotation below; during
training, the text portion of the user content will be
substituted into the intent-relevance scoring prompt
as {query}.
Annotation Example
[{"role": "system", "content": "You are a helpful assis-
tant."}, {"role": "user", "content": [{ "type": "image", "im-
age": "8480330992/frame_000310.jpg"}, {"type": "text",
"text": "why do the ladies tap the children s heads
with some soft item"}]}, { "role": "assistant", "content":
[{"type": "text","text": "2"}]}]
D.2 Dataset Distribution Statistics
We summarize the data distributions of the IR-
600K training and validation sets in Figures 4.
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni0000002c/uni00000051/uni00000057/uni00000048/uni00000051/uni00000057/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni00000048/uni00000003/uni0000002f/uni00000048/uni00000059/uni00000048/uni0000004f/uni00000013/uni00000011/uni00000013/uni00000013/uni00000013/uni00000011/uni00000013/uni00000018/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000016/uni00000013/uni00000033/uni00000055/uni00000052/uni00000053/uni00000052/uni00000055/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000052/uni00000049/uni00000003/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f/uni00000015/uni00000017/uni00000011/uni00000015/uni00000008
/uni0000001b/uni00000011/uni0000001a/uni00000008/uni00000015/uni0000001b/uni00000011/uni0000001b/uni00000008
/uni00000014/uni00000019/uni00000011/uni0000001b/uni00000008/uni00000015/uni00000014/uni00000011/uni00000018/uni00000008/uni00000015/uni0000001b/uni00000011/uni0000001b/uni00000008
/uni0000001c/uni00000011/uni0000001b/uni00000008/uni00000015/uni0000001b/uni00000011/uni0000001a/uni00000008
/uni00000014/uni00000017/uni00000011/uni00000017/uni00000008/uni00000014/uni0000001b/uni00000011/uni00000017/uni00000008/uni0000002c/uni00000051/uni00000057/uni00000048/uni00000051/uni00000057/uni00000003/uni00000035/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni00000048/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000003/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000055/uni0000004c/uni00000045/uni00000058/uni00000057/uni0000004c/uni00000052/uni00000051/uni0000001d/uni00000003/uni00000037/uni00000055/uni00000044/uni0000004c/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000059/uni00000056/uni00000011/uni00000003/uni00000039/uni00000044/uni0000004f/uni0000004c/uni00000047/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051
/uni00000037/uni00000055/uni00000044/uni0000004c/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000036/uni00000048/uni00000057
/uni00000039/uni00000044/uni0000004f/uni0000004c/uni00000047/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000036/uni00000048/uni00000057
Figure 4:Distribution of Intent Relevance Scores across the
IR-600K training and validation sets. The results demonstrate
that the validation set maintains a consistent class distribution
with the training set across all relevance levels.E Intent Relevance Scoring Prompt
Intent-Relevance Scoring Prompt
Infer the query’s intent and evaluate how likely it is that
this frame is intent-relevant for answering: ‘{query}’.
Output only one number from 1 to 5, where:
[1] = completely irrelevant — the frame provides no visual
or contextual information related to the question or its
answer.
[2] = slightly relevant — the frame shows general back-
ground or context, but it is unlikely to contribute to answer-
ing.
[3] = moderately relevant — the frame includes partial
clues or indirect context that might help infer the answer,
but the key evidence is missing.
[4] = mostly relevant — the frame provides substantial
visual or contextual information that can be used to answer
the question, though not fully decisive.
[5] = highly relevant — the frame clearly contains the
decisive evidence or strong contextual cues that directly or
indirectly support the correct answer.