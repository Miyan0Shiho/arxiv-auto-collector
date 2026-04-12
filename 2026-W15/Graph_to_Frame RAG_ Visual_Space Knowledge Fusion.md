# Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning

**Authors**: Songyuan Yang, Weijiang Yu, Ziyu Liu, Guijian Tang, Wenjing Yang, Huibin Tan, Nong Xiao

**Published**: 2026-04-06 02:43:03

**PDF URL**: [https://arxiv.org/pdf/2604.04372v1](https://arxiv.org/pdf/2604.04372v1)

## Abstract
When video reasoning requires external knowledge, many systems with large multimodal models (LMMs) adopt retrieval augmentation to supply the missing context. Appending textual or multi-clip evidence, however, forces heterogeneous signals into a single attention space. We observe diluted attention and higher cognitive load even on non-long videos. The bottleneck is not only what to retrieve but how to represent and fuse external knowledge with the video backbone.We present Graph-to-Frame RAG (G2F-RAG), a training free and auditable paradigm that delivers knowledge in the visual space. On the offline stage, an agent builds a problem-agnostic video knowledge graph that integrates entities, events, spatial relations, and linked world knowledge. On the online stage, a hierarchical multi-agent controller decides whether external knowledge is needed, retrieves a minimal sufficient subgraph, and renders it as a single reasoning frame appended to the video. LMMs then perform joint reasoning in a unified visual domain. This design reduces cognitive load and leaves an explicit, inspectable evidence trail.G2F-RAG is plug-and-play across backbones and scales. It yields consistent gains on diverse public benchmarks, with larger improvements in knowledge-intensive settings. Ablations further confirm that knowledge representation and delivery matter. G2F-RAG reframes retrieval as visual space knowledge fusion for robust and interpretable video reasoning.

## Full Text


<!-- PDF content starts -->

Graph-to-Frame RAG: Visual-Space Knowledge Fusion
for Training-Free and Auditable Video Reasoning
Songyuan Yang1∗Weijiang Yu2†Ziyu Liu2Guijian Tang1Wenjing Yang1†
Huibin Tan1∗Nong Xiao2
1National University of Defense Technology2Sun Yat-sen University
Abstract
When video reasoning requires external knowledge, many
systems with large multimodal models (LMMs) adopt re-
trieval augmentation to supply the missing context. Ap-
pending textual or multi-clip evidence, however, forces het-
erogeneous signals into a single attention space. We ob-
serve diluted attention and higher cognitive load even on
non-long videos. The bottleneck is not only what to re-
trieve but how to represent and fuse external knowledge
with the video backbone. We presentGraph-to-Frame RAG
(G2F-RAG), a training free and auditable paradigm that
delivers knowledge in the visual space. On the offline
stage, an agent builds a problem-agnostic video knowledge
graph that integrates entities, events, spatial relations, and
linked world knowledge. On the online stage, a hierarchical
multi-agent controller decides whether external knowledge
is needed, retrieves a minimal sufficient subgraph, and ren-
ders it as a single reasoning frame appended to the video.
LMMs then perform joint reasoning in a unified visual do-
main. This design reduces cognitive load and leaves an ex-
plicit, inspectable evidence trail. G2F-RAG is plug-and-
play across backbones and scales. It yields consistent gains
on diverse public benchmarks, with larger improvements
in knowledge-intensive settings. Ablations further confirm
that knowledge representation and delivery matter. G2F-
RAG reframes retrieval as visual space knowledge fusion
for robust and interpretable video reasoning.
1. Introduction
Recent progress in large multimodal models (LMMs) [4,
21, 25, 46] has advanced cross-modal representation, long-
range temporal modeling, and open-domain reasoning,
pushing video understanding from perception toward rea-
soning. Despite these gains, complex video reasoning still
faces three fundamental challenges. First, compositional
and multi-step reasoning [53], such as cross-shot causality,
navigation, and assembly, requires repeated alignment be-
∗Equal contribution.†Corresponding author.
A: She felt cold. LMMVideo RAG
LMM(a) Traditional Video RAG
(b) Graph-to-Frame RAG (Ours)Video
Auxiliary Text Video
A: She got ready for an online meeting.
G2F-RAG Agents
VideoQ:  Why did the girl put on clothes?
Q:  Why did the girl put on clothes?
 Video Graph 
Knowledge
RAG Input
Figure 1. Traditional Video RAG appends auxiliary text, mixing
heterogeneous tokens in one attention space, While G2F-RAG ren-
ders retrieved knowledge as a single reasoning frame appended
to the video, keeping evidence in the visual space and producing
more grounded causal answers.
tween local evidence and global structure. Second, external
knowledge [54] is often required, including commonsense,
object functionality, and spatial topology, yet such informa-
tion is not directly observable at the pixel level. Third, under
constraints of computation and interpretability [16], we aim
to solve complex problems reliably with small models with-
out additional training, while providing an auditable evi-
dence trail. This landscape motivates retrieval-augmented
video understanding: retrieve information relevant to the
question, then reason jointly with the video model. How-
ever, a key bottleneck remains how to feed external knowl-
edge into the model with an appropriate way.
Mainstream video retrieval-augmented methods typi-
cally follow a retrieve-then-append paradigm with three
common routes. The first [37] appends textual context such
as ASR, OCR, or descriptions to the prompt. The second
[2, 43] retrieves candidate clips or keyframes and providesarXiv:2604.04372v1  [cs.CV]  6 Apr 2026

them as additional visual inputs. The third [22, 42] struc-
tures video into graphs or event chains, then injects the se-
lected results as text or multi-segment context into the gen-
erator. These methods share an implicit assumption: ob-
taining more relevant content and providing a longer con-
text will improve reasoning. In practice, we repeatedly ob-
serve performance degradation even when videos are short.
Heterogeneous information sources must share a single at-
tention space [13]. Continuous low-level visual signals and
discrete high-level text or dense visual text blocks compete
for attention. The model must frequently switch and rec-
oncile between two distinct representations. This leads to
diluted attention and increased cognitive load [35, 43]. The
observation suggests that what to retrieve is important, but
how to represent and fuse it with the backbone is equally
crucial. When semantics are misaligned and the load is not
controlled, retrieval can hinder the capability of models.
Guided by these observations, we shift the focus from
extending videos or enlarging contexts to redesigning how
knowledge is presented to the model. Video models are
strongest when aggregating and reasoning within the vi-
sual space. External knowledge should therefore enter the
same space with a visual grammar, which avoids cross-
modal competition and reduces redundant tokens. Recent
studies [48] also suggest that the visual modality can serve
as an efficient compression medium for textual informa-
tion, representing rich content with far fewer tokens. In-
spired by this idea, we apply it to retrieval augmentation.
Instead of appending text or many retrieved segments, we
convert the retrieved structured knowledge into consumable
visual tokens so that the model operates in its most familiar
and well suited domain of spatiotemporal visual reasoning
rather than in long context assimilation.
Building on this principle, we present Graph-to-Frame
Retrieval-Augmented Generation (G2F-RAG) as a novel
paradigm for delivering retrieved knowledge in the visual
space, as shown in Figure 1. To instantiate this paradigm for
video reasoning, we design a training-free two-stage multi-
agent workflow, as shown in Figure 3. Offline, a graph-
construction agent builds and caches a problem-agnostic,
full video knowledge graph for each video. The graph
covers entities, events, spatial relations, and essential links
to external knowledge. It is constructed once and reused
across questions. Online, an orchestration agent performs
dynamic decisions via difficulty-aware hierarchical routing.
If a question can be answered directly from the video ev-
idence, a small LMMs answers it to preserve base per-
formance. If a question requires external knowledge or
multi-step reasoning, a retrieval agent retrieves a mini-
mal sufficient reasoning chain or subgraph from the offline
graph, a rendering agent renders it into a small number
of lightweight and layout-controlled reasoning frames, and
then feeds these frames together with the original video intothe LMMs for joint reasoning. The reasoning frames are in-
serted or concatenated as real video frames so that temporal
attention properly attends to them. The entire pipeline keeps
the backbone frozen and changes only how knowledge is
represented and fused, which addresses cross-modal com-
petition and cognitive load at the source. Extensive experi-
ments show that, under the same backbone, framing struc-
tured knowledge and fusing it in the visual modality signif-
icantly outperforms baselines.
Our contributions are as follows: (1) To the best of our
knowledge, we are the first to convert retrieved knowledge
into the visual modality for video reasoning. Specifically,
we propose Graph-to-Frame Retrieval-Augmented Genera-
tion (G2F-RAG), which renders retrieved knowledge into a
video frame and feeds it as part of the video, mitigating cog-
nitive load and retrieval-context explosion at the source. (2)
To instantiate this paradigm, we design a training-free, two-
stage multi-agent workflow: offline a graph-construction
agent builds and caches a reusable, problem-agnostic video
knowledge graph; online a controller performs difficulty-
aware routing, retrieves a minimal subgraph, and renders a
single layout-controlled reasoning frame appended to the
video for joint inference with frozen LMM backbones.
(3) We demonstrate strong performance and generalization
across benchmarks and backbones. Comprehensive anal-
yses reveal a central conclusion: compared to unbounded
context expansion, delivering necessary knowledge as vi-
sual frames is better aligned with the inductive biases of
video models for retrieval-augmented reasoning.
2. Related Work
2.1. Large Multimodal Models
Building on the success of LLMs [1, 12, 38, 52, 56], Large
Multimodal Models (LMMs) have advanced visual lan-
guage understanding rapidly [24]. LLaV A [32–34] popu-
larized instruction tuning on GPT-4 [1] curated data and es-
tablished a widely adopted recipe for constructing LMMs,
inspiring many follow up systems [6, 7, 63]. Fueled by
short-video applications and video generation, video un-
derstanding [55, 59, 62] has emerged as a focal area, yet
spatio-temporal joint modeling remains challenging, de-
manding fine-grained spatial grounding and multi-hop rea-
soning, especially in long-video settings [8, 27, 40, 61].
To improve efficiency and scalability, Vamba [41] intro-
duces a hybrid Mamba–Transformer architecture with near-
linear token complexity, while recent efforts [16, 29, 51]
explore GRPO-based reinforcement fine-tuning for video
LMMs and agentic pipelines for planning and tool use. In
practice, closed-source commercial systems (e.g., GPT-4o
[21], Gemini [44]) currently lead on video understanding,
whereas open-source families such as Qwen-VL Series [3–
5, 45], InternVL Series [9, 10, 46, 65], and VideoLLaMA
Series [11, 58, 59] provide widely used, reproducible re-

Q:  Why did the girl put on clothes?
Frame   14 Frame 0Video-RAG
G2F-RAG
1.0
0.0Key FrameFrame      7
...Graph Frame Text Contextual Token Graph Frame  Question Token
(a) Attention analysis of Video-RAG and G2F-RAG. (b) Comparison between Video-RAG and G2F-RAG.
Video-RAG G2F-RAG Baseline
Figure 2. (a) Attention Analysis. Text-append Video-RAG diverts attention from key frames toward contextual tokens, while G2F-RAG
keeps focus on key frames and the appended graph frame. (b) Comparison of the performance. Video-RAG causes consistent drops,
whereas G2F-RAG yields clear gains over the baseline on all three benchmarks.
search baselines. However, current LMMs still lack a prin-
cipled way to ingest external knowledge without diluting
attention or requiring retraining, and our Graph-to-Frame
RAG addresses this by fusing knowledge in the visual space
for training-free, auditable reasoning.
2.2. Video Augmented Understanding
Retrieval-Augmented Generation (RAG) is a primary
paradigm for Video Augmented Understanding [23, 37, 42]
in LMMs. It [18, 28] indexes clips, subtitles, and struc-
tured cues, retrieves evidence at query time, and conditions
generation on the retrieved context, which improves factual
grounding, reduces hallucination, and mitigates long con-
text limits. Progress [2, 19, 23] spans finer retrieval units,
learned aggregation across clips, and knowledge representa-
tions that range from chunk based stores to graph structures,
sometimes with adaptive routing that balances accuracy and
efficiency. Remaining challenges [42, 43, 50] include tem-
poral dispersion in long video, multi hop dependencies, re-
trieval noise, and the cost of constructing and maintaining
large structured memories. Complementing retrieval, re-
cent Video Augmented Understanding introduces light vi-
sual scaffolds that make temporal and structural informa-
tion explicit, for example numbering frames [49], draw-
ing progress indicators to surface relevant segments [60],
or translating images into symbolic SVG for compositional
reasoning [31]. However, appending retrieved text or multi-
clip evidence often creates attention competition and ampli-
fies noise, and our approach resolves this by converting the
retrieved subgraph into a single reasoning frame for same-
modality fusion.
3. Method
3.1. Attention Analysis
Current video LMMs encode a video as frame tokens and
align them with query tokens in a single attention space.
Traditional text-based video RAG methods appends re-
trieved passages as additional tokens. Using Qwen-2.5VL[5] as baseline, we aggregate the attention scores from all
the visual tokens corresponding to each frame across all
attention heads and separate query tokens from retrieved-
context tokens. In Figure 2.(a), text-based Video-RAG [37]
disperses attention toward the long retrieved context and
non-salient frames, reducing focus on key frames. With our
G2F-RAG, attention concentrates on the key segment and
on the single graph frame appended at the end.
The quantitative trends in Figure 2.(b) mirror this behav-
ior. On MLVU [64], Video-RAG reduces accuracy by 5.4
points relative to the baseline, while G2F-RAG improves by
4.6 points over the baseline and by 10.0 points over Video-
RAG. On WildVideo [54], the drops and gains are 4.1 and
3.9 points. On VideoMME [17], they are 4.6 and 5.5 points.
Appending retrieved text consistently harms performance,
whereas delivering the same knowledge as a single reason-
ing frame in the visual domain yields stable gains under the
same backbone. The key factor is not only what to retrieve,
but how to represent and fuse it with the video.
3.2. Multi-Agent Framework
We present a training free multi-agent framework that fuses
external knowledge with video in a single visual domain.
As shown in Figure 3, The workflow has an offline stage
and an online stage. In the offline stage, a networked graph-
construction agent generates and caches a full knowledge
graph for each video. In the online stage, an orchestration
agent receives a query, performs difficulty-aware routing,
invokes a retrieval agent to extract a minimal sufficient sub-
graph, and invokes a rendering agent to convert this sub-
graph into a single reasoning frame. The video and the rea-
soning frame are then provided to LMMs for joint reasoning
and answer generation.
Formally, given a videoVand a queryq, the offline
graph-construction agent is denoted byBuildand outputs
a problem-agnostic knowledge graphG=Build(V). The
graphG= (V,E)covers entities, events, actions, spatial
locations, affordances, and concept-level knowledge, with
cross-layer links. In the online stage, an orchestration agent

Text Token
External KnowledgeOffline Stage
    Full  Graph  Online Stage (Retrieval)
Orchestration 
Agent
Because she felt cold. Large Multimodal Model
Problem-agnostic Knowledge Graph  
Rendering 
AgentGraph  Construction Agent
Event Causality Extraction
Scene Affordance Analysis
Cross Links
Event Causality 
Entities:
  Girl
  ...Scene AffordanceCross Links
Objects:
  Clothes
  Bed
  PC
  ...
Video 
Subgraph S*
Edges:
  from a0 to a1
  from a0 to L1
...
Agent 
Retrieval Agent
Entity
Relations
Causal Flow
t
       Query          Video
Actions:
  sneeze
  pick_up
  put_on
  ...Locations:
  Bedroom
  ...
a1
a0a2
Nodes:
  sneeze
  pick_up
  put_on
  ...
Online Stage (Reasoning)from a0 to a1
from a1 to a2
from a1 to O3
...
a0
L1a1a2O3
a1
a0a2
O1O2O3
L1
L1
Video Frame   Video
Causality from a0 to a2 via a1a0
a2L1
a1Q:  Why did the 
girl put on clothes?
Hard
Orchestration 
Agent
   Videoif hard
if easy
Figure 3.Method overview.Offline: a graph-construction agent builds a problem-agnostic video knowledge graph with optional external
knowledge. Online: an orchestration agent routes the query; a retrieval agent extracts a minimal subgraphS∗; a rendering agent converts
it into one reasoning frame and appends it to the video to form ˜V= [V;I RF]. Finally, the LMM answers fromVfor easy cases, or from
˜Vfor hard cases. The pipeline is training free, auditable, and fuses knowledge in the visual space.
Routedecides a pathRoute(q, V,G)∈easy,hard. If the
decision is easy, LMMs answer directly on the video to pre-
serve base performance. If the decision is hard, a retrieval
agent selects a compact subgraphS⋆fromGthat maximizes
relevance under a complexity budget:
S⋆= arg max
S⊆G[R(q, S)−λ, C(S)], (1)
whereR(q, S)measures relevance,C(S)regularizes com-
plexity, andλ >0balances the two terms. A rendering
agent convertsS⋆into a single high-readability reasoning
frameI RF=Render(S⋆). We appendI RFas the last
frame of the video to obtain the fused input ˜V= [V;I RF].
LMMs then produce the final answera=LMMs( ˜V , q).
The framework keeps the backbone frozen and changes
only how knowledge is represented and delivered. It pre-
serves performance on easy cases and unlocks the benefits
of modality-aligned fusion for complex reasoning.
3.3. Graph-to-Frame Knowledge Fusion
Graph-to-frame fusion is the core of the method, as shown
in Figure 4. In the offline stage, a networked graph-
construction agent analyzes the video under a curated
prompt and produces a queryable full graphG. The de-
sign unifies two complementary views. The event–causal
LMM
Video 
Subgraph S*Graph  
Construction 
Agent
   Query Retrieval AgentRendering 
Agent
Video 
QFull  Graph  
interpolate
a1
a0 a2
(a) Event Causal View
O1O2O3
L1(b) Scene Affordance View
a0
L1a1a2O3(c) Cross Links
Offline Stage Online Stage
Figure 4.G2F-RAG pipline.Build a full video graph, retrieve a
minimal subgraph for the query, render it as a single frame, and
append it to the video for visual-space reasoning with the LMM.
view models participants, actions, intent, preconditions,
postconditions, and event-level abstractions connected by
causal links. The scene–affordance view models objects
and their affordances, functional areas and connectivity,
and abstract concept knowledge. Dense cross-links bind
the two views, which allows seamless transitions between
“what happened” and “where or with what it happened.”
The resulting graph covers visible facts and attaches essen-
tial world knowledge. It provides a reliable substrate for
downstream retrieval and reasoning.

In the online stage, the retrieval agent maps the natural-
language queryqto a structured subgraph query overGand
returns a compact subgraphS⋆that is minimal yet suffi-
cient for the target reasoning. We impose explicit complex-
ity constraints to control the visual token budget after ren-
dering, such as|V(S⋆)| ≤N max and|E(S⋆)| ≤E max.
A rendering agent then convertsS⋆into a single reasoning
frameI RFthat uses clear visual grammar to depict key en-
tities, relations, and causal flow. The frame employs simple
geometric primitives and short labels to ensure high read-
ability and low noise within the visual domain. The reason-
ing frame does not encode temporal spans or timestamps.
It focuses on structure and mechanism, which helps LMMs
internalize external knowledge without being distracted by
redundant details.
We appendI RFas the last frame to form ˜V= [V;I RF].
Placing the reasoning frame at the end avoids interference
with the model’s learning of the original content while en-
suring that temporal attention can still attend to the inserted
frame. A concise instruction enforces consistency, with
original video context treated as authoritative. Conceptu-
ally, the process maps “retrieved structured knowledge” to
“a small number of visual tokens” that LMMs can consume
efficiently.
3.4. Hierarchical Mechanism
The hierarchical mechanism uses an agent to classify ques-
tion difficulty and route computation accordingly. The goal
is to protect performance on easy cases while directing com-
plex cases to the knowledge-enhanced, modality-aligned
path. Given(q, V,G), the agent outputs a binary decision
d(q, V,G)∈easy,hard. We view the decision as an ap-
proximate utility comparison. The hard branch is chosen
when the expected utility of graph-to-frame fusion exceeds
that of direct video reasoning by a safe margin:
d(q, V,G) =(
hard,∆U(q, V,G)≥τ,
easy,otherwise.(2)
∆U(q, V,G) = bUG2F(q, V,G)− bUBase(q, V). (3)
HerebUG2Fand bUBaseare proxy scores produced by the
agent under different prompts. They summarize factors
such as locality of evidence, cross-scene dependency, ex-
ternal concept triggers, and estimated reasoning steps. The
thresholdτacts as a guard to protect easy-case perfor-
mance. The agent also exposes uncertainty and a fallback
signal. If confidence is low or the selected subgraph ex-
ceeds the budget, the system falls back to the easy branch
or retries with tighter constraints.
The mechanism does not learn new parameters. It relies
on prompt design to let the agent perform task decomposi-tion and strategy selection in the input space. Since the deci-
sion depends on(q, V,G)and on explicit detection of exter-
nal knowledge needs, it favors problems that require struc-
tured, cross-entity, or cross-location reasoning and routes
them to retrieval and graph-to-frame fusion. For questions
that can be answered directly from the video, it keeps the
shortest path and avoids unnecessary knowledge injection.
Combined with the multi-agent orchestration and graph-to-
frame fusion, the hierarchical mechanism achieves “direct
for easy, enhanced for hard” without modifying LMMs.
4. Experiments
4.1. Experimental Setup
Baselines.We evaluate G2F-RAG on representative open-
source LMMs, including InternVL3.5 [46], LLaV A-Video
[62], and Qwen2.5-VL [5], across different parameter
scales. For broader context, we report results on strong
open-source families such as QwenVL [4], InternVL [9,
65], MiniCPM-V [55, 57], VideoLLaMA 3 [58], and VILA-
1.5 [30], as well as commercial models GPT-4o [21] and
Gemini 1.5 Pro [44]. We further compare against state-
of-the-art video RAG baselines, including Video-RAG [37]
and Vgent [43]. All models use public checkpoints and of-
ficial inference configurations.
Benchmarks.We choose 8 datasets that span complemen-
tary skills: general multi-task ability with MVBench [26],
MMBench-Video [15], and VideoMME [17]; temporal and
spatial reasoning with TempCompass [36] and VSIBench
[53]; open world setting required external knowledge with
WildVideo [54] and VideoMMMU [20]; and long-form rea-
soning with MLVU [64]. We adopt the no-subtitle setting
for VideoMME and the single-turn setting for WildVideo.
Implementation Details.The multi-agent pipeline uses
LMMs without additional training. The offline graph-
construction agent employs GPT-4o [21] (API Version:
gpt-4o-2024-05-13). Difficulty routing and sub-
graph extraction use GPT-4o-mini [39] (API Version:
gpt-4o-mini-2024-07-18). The reasoning frame is
produced by a lightweight renderer Graphviz [14]. Prompts
are provided in the Appendix. All experiments run on
NVIDIA A100 40GB GPUs under the official decoding and
preprocessing settings of each model.
4.2. Main Result
As shown in Table 1, G2F-RAG delivers consistent and sig-
nificant gains across diverse LMM families and parameter
scales without retraining the backbones. These results show
that graph-to-frame fusion is portable across architectures
and additive to existing capabilities.
Analysis among Different Tasks.Grouping benchmarks
by skill reveals a clear pattern. On temporal–spatial rea-
soning, G2F-RAG strengthens frame-level focus and yields

Table 1.Performance comparison on sota LMMs.All scores are average accuracy (%) with 3 significant digits, except MMBench-Video
which is graded by an LLM evaluator. G2F-RAG’s absolute gains on each baseline are shown in superscripts.
General Multi Task Temporal Spatial Reasoning Open World
Model Size MVBenchVideoMME
(w/o sub)MMBench-Video TempCompass VSIBench WildVideo VideoMMMU
Proprietary LMMs
Gemini1.5 Pro [44] - 60.5 77.4 1.80 68.4 48.8 53.7 53.5
GPT-4o (2024-05-13) [21] - 57.5 67.9 1.87 69.5 34 62.1 61.2
Open-Source LMMs (5B<)
Qwen2.5-VL [5] 3B 67.0 61.5 1.63 64.4 27.9 38.6 42.3
Qwen3-VL [4] 4B 68.9 69.3 - - 58.4 - 56.2
InternVL3.5 [46] 4B 71.2 65.4 1.59 65.4 54.9 45.2 58.3
+ G2F-RAG (Ours) 4B 74.8+3.670.1+4.71.66+0.0768.7+3.359.0+4.147.1+1.962.4+3.9
Open-Source LMMs (>5B, 10B<)
InternVideo2.5 [47] 7B 75.7 65.1 - - - - 43.0
VideoLLaMA 3 [58] 7B 69.7 61.0 - - - - 46.0
VILA-1.5 [30] 8B - - - 58.8 28.9 - 20.9
MiniCPM-V 2.6 [55] 8B 44.7 59.7 1.70 59.6 - 46.4 -
MiniCPM-V 4.5 [57] 8B - 67.9 - - - - -
InternVL2.5 [9] 8B 70.5 63.7 1.68 68.7 41.6 - -
InternVL3 [65] 8B 73.2 66.0 1.69 70.4 41.6 - 48.9
Qwen3-VL [4] 8B 68.7 71.4 - - 59.4 - 65.3
LLaV A-Video [62] 7B 62.1 63.7 1.60 65.5 35.7 53.4 36.1
+ G2F-RAG (Ours) 7B 68.5+6.464.5+0.81.66+0.0669.0+3.542.5+6.857.0+3.644.0+7.9
Qwen2.5-VL [5] 7B 67.5 65.1 1.79 69.2 37.4 51.3 47.4
+ G2F-RAG (Ours) 7B 73.2+5.770.6+5.51.85+0.0672.0+2.844.8+7.455.4+4.153.1+5.7
InternVL3.5 [46] 8B 72.1 66.0 1.74 71.0 56.3 53.0 63.4
+ G2F-RAG (Ours) 8B 78.5+6.472.0+6.01.82+0.0875.5+4.560.1+3.860.1+7.168.2+4.8
Open-Source LMMs (>10B)
InternVL3 [65] 14B 76.6 70.4 1.73 - 48.9 - -
InternVL3.5 [46] 14B 72.8 67.9 1.73 71.2 60.8 60.1 65.0
+ G2F-RAG (Ours) 14B 77.4+4.670.6+2.71.80+0.0776.8+5.662.9+2.163.2+3.171.2+6.2
large gains, exemplified by a +7.4% improvement on
VSIBench with Qwen2.5-VL-7B. On external-knowledge
and open-world settings, visual-space delivery is even
more beneficial, as shown by 7.1% on WildVideo with
InternVL3.5-8B. General multi-task evaluations also im-
prove reliably: VideoMME typically rises by about +5–6%
across backbones (e.g., Qwen2.5-VL-7B +5.5%), while
MMBench-Video shows small but consistent positive shifts
on its 0–3 scale. These results indicate that converting re-
trieved knowledge into a single reasoning frame transfers
across models and task types.
Analysis among Different Model Sizes.Smaller and
medium models achieve larger relative gains, while larger
models also benefit. Absolute improvements for 4B and
7B models often fall in the range of three to seven points.Table 2.Performance comparison with sota RAG methods.We
compare 2 sota video RAG methods (Video-RAG, Vgent) with our
G2F-RAG. G2F-RAG consistently outperforms alternatives. The
best results arebold.All scores are average accuracy (%).
Method Size MLVU WildVideoVideoMME
(w/o sub)
LLaV A-Video 7B 69.5 53.4 63.7
+ Video-RAG [37] 7B 71.3 48.5 64.8
+ Vgent [43] 7B 72.5 51.6 66.7
+ G2F-RAG (Ours) 7B 75.5 57.0 64.5
Qwen2.5-VL 7B 68.8 51.3 65.1
+ Video-RAG [37] 7B 63.4 47.2 60.5
+ Vgent [43] 7B 72.1 50.1 68.9
+ G2F-RAG (Ours) 7B 73.4 55.4 70.6

Agent Workflow
Text Contextual Token
  Question: Why did she carry the cup after drinking, and where is it 
at the end？
Query 
Frame 0Video-RAG
G2F-RAG1.0
0.0Key FrameFrame    11
...Graph Frame
Orchestration 
Agent
Render Agent
Graph Construction 
Agent
Retrieval Agent
LMM
Final Answer: She carried 
the cup to drink again, and it 
still remained in her hand near 
the doorway at the end.External Knowledge
Event Causal View
  entities
E1: woman 
 actions  
Scene Affordance View
objects
O1: dishes
O2: faucet
O3: cup
O4: water purifier
L1: kitchen
L2: doorway  
locations Subgraph S* 
 nodes
edges
Frame    25
A1
A2 A3
A4
A5
O1
O2O3
O4
L1
L2
O3
A4A5
L2A3
A2L1A1: wash
A2: pick_up
A3: drink1
A4: walk
A5: drink2
Offline Stage
Online Stage 
(Retrieval)
Attention analysisCross LinksA2–A5–L2 causal link established.
Key FrameOnline Stage 
(Reasoning)I will build a knowledge graph 
for a complex reasoning task.
It’s a complex question. I need 
external knowledge. I should visualize the causality 
for complex reasoning.
I can see external knowledge! It 
indicates she continues drinking 
at the doorway, cup in hand.
Graph FrameQuestion Token(a) Event Causal View
(b) Scene Affordance View
  what happened
where it happenedwhat it happened with Video Frame
I need to extract the relevant 
information and obtain the 
subgraph S*.
Figure 5.Case study.We builds an offline full graph, retrieves a compact subgraph online, renders one reasoning frame, and appends it to
the video for reasoning. Further analysis show attention concentrates on key video frames and the graph frame.
The gains for 8B and 14B are slightly smaller but remain
steady, especially on knowledge- and structure-intensive
tasks. The pattern indicates that the effect comes from rep-
resentation alignment rather than capacity. Feeding knowl-
edge as frames aligns with the video attention space and
reduces the loss from cross-modal competition, which is or-
thogonal to size.
Comparison with Different RAG Method.Text-based
Video-RAG frequently degrades performance, for example
Qwen2.5-VL-7B on MLVU, WildVideo, and VideoMME
(w/o sub) changes from 68.8% to 63.4%, 51.3% to 47.2%,
and 65.1% to 60.5%. Vgent mitigates overload with struc-
tured retrieval and validation but still falls short of graph-to-
frame fusion. On Qwen2.5-VL-7B, MLVU and VideoMME
(w/o sub) rise from 72.1% and 68.9% with Vgent to 73.4%
and 70.6% with G2F-RAG. On LLaV A-Video-7B, Wild-
Video increases from 53.4% to 57.0%. These findings sug-
gest that, what to retrieve matters, but how to represent and
fuse knowledge with the video is decisive.
Case Study.As shown in Figure 5, the offline graph en-
codes actions, objects, and locations with event–scene links.
The orchestrator agent marks the query as hard; retrieval re-
turns a minimal subgraph that ties the pick-up and repeated-
drinking actions with the doorway location and the cup en-
tity. The rendered reasoning frame depicts this causal and
spatial chain and is appended as the last frame. Finally, the
LMM correctly outputs the grounded answer based multi-modal augmented input. The attention heatmap highlights
focus on the key temporal segments and the reasoning frame
under G2F-RAG, illustrating how visual-space fusion pro-
vides auditable evidence and reduces cognitive load.
4.3. Ablation Study
Representation and Delivery.We compare textual json
injection (G2J-RAG) with visual-space fusion (G2F-RAG).
Both variants use the same retrieved subgraph; only the
delivery differs. Rendering the subgraph as a single rea-
soning frame consistently outperforms textual json concate-
nation, notably on VideoMME (w/o sub) with an absolute
gain of +7.6% over G2J-RAG and +5.5% over the no-RAG
baseline, and on WildVideo with +6.6% over G2J-RAG.
On MLVU, G2F-RAG improves by +7.2% relative to G2J-
RAG. These results indicate that visual delivery avoids the
long retrieved context that dilutes attention and make the
consistent video augmented reasoning.
Frame Placement and Count.We vary where and how
many reasoning frames are inserted while fixing retrieval
and rendering. Appending a single frame at the end (End-
1) is best across datasets. Mid-sequence insertion disrupts
temporal aggregation and yields the largest drop, for ex-
ample MLVU 73.4% to 67.9% and VideoMME 70.6% to
64.0%. Placing the frame at the start is less harmful but
still below the default, likely due to a primacy effect that
biases the model before it has formed a stable video repre-
sentation. Increasing the number of end-appended frames

Table 3.Ablation Study on Qwen2.5-VL-7B.Representa-
tion/Delivery: G2J-RAG injects a textual graph json; G2F-RAG
renders the retrieved subgraph into a frame and fuses it visually.
Frame Placement/Count: End-1 appends a single frame at the end
of the video sequence; other placements or more frames reduce
performance. Frame Visual Style: Minimal uses icons and short
labels; text-heavy or single-aspect styles degrade results. Rout-
ing/Fallback: Off applies RAG to all queries and hurts easy cases.
Graph & Retrieval: Coarse removes intent and affordances; Full-
Loose selects larger subgraphs; Full-NoExt disables external web
tools; Full-Compact selects a compact subgraph.
Ablation Variant MLVU WildVideoVideoMME
(w/o sub)
Baseline
Qwen2.5-VL-7B 68.8 51.3 65.1
Representation / Delivery
G2J-RAG 66.2 48.8 63.0
G2F-RAG (Ours) 73.4 55.4 70.6
Frame Placement / Count
Start-1 72.4 54.3 69.4
Mid-1 67.9 49.8 64.0
End-4 69.0 51.6 66.0
End-2 72.4 54.6 69.8
End-1 (Ours) 73.4 55.4 70.6
Frame Visual Style
Text-Heavy 68.4 50.5 66.2
Topology 71.0 53.3 69.4
Causality 70.5 53.0 69.2
Icons-Only 69.6 52.4 68.5
Minimal (Ours) 73.4 55.4 70.6
Routing / Fallback
Off 69.9 52.0 66.8
On (no FB) 71.2 53.7 69.0
On + FB (Ours) 73.4 55.4 70.6
Graph & Retrieval
Coarse 70.2 52.1 68.8
Full-Loose 71.6 53.5 69.7
Full-NoExt 71.0 52.4 69.3
Full-Compact (Ours) 73.4 55.4 70.6
raises the visual token budget and slightly erodes accuracy,
as seen with End-2 on VideoMME (69.8%) and End-4 on
MLVU (69.0%). A single end frame provides the best bal-
ance between saliency and non-interference.
Frame Visual Style.We evaluate five renderings for the
same subgraph. “Minimal” uses icons and short labels
with compact layout. “Text-Heavy” uses dense textual de-
scriptions. “Topology” keeps spatial relations and removes
causal links. “Causality” keeps causal chains and removes
spatial topology. “Icons-Only” removes all text. The Min-
imal design (icons plus short labels) performs best, while
dense text reintroduces context burden and degrades Wild-
Video from 55.4% to 50.5%. Removing structural cues also
hurts: on MLVU, Topology and Causality variants reduceaccuracy from 73.4% to 71.0% and 70.5%. Icons-Only
further confirms that a small amount of text is needed for
disambiguation. The guideline is to keep diagrams com-
pact, retain both spatial topology and directed causality, and
avoid verbosity.
Routing and Fallback.We evaluate when to invoke G2F-
RAG. Forcing the frame for all queries (Off) hurts easy
cases, yielding 66.8% on VideoMME and 69.9% on MLVU.
Enabling routing without fallback(FB) improves results but
remains vulnerable to occasional misrouting, for example
69.0% on VideoMME. Routing with a conservative fallback
(On + FB) achieves the best accuracy on all datasets, 70.6%
on VideoMME and 73.4% on MLVU, by reserving fram-
ing for questions that benefit from external knowledge while
protecting easy ones.
Graph and Retrieval.We further ablate the offline graph
and the online retrieval. “Coarse” removes intent and affor-
dances from the graph. “Full-Loose” keeps the full graph
but selects larger subgraphs. “Full-NoExt” keeps the full
graph but disables external tools during construction. “Full-
Compact” is our default with the full graph and compact
subgraph selection. Removing intent and affordances low-
ers MLVU from 73.4% to 70.2% and VideoMME from
70.6% to 68.8%, showing that these fields capture useful
preconditions and functionality. Using larger subgraphs
modestly reduces precision, for example on VideoMME
69.7% compared with 70.6%. Disabling external tools dur-
ing construction weakens world knowledge and reduces
WildVideo from 55.4% to 52.4%. Our default setting pairs a
rich offline graph with compact, relevance-focused retrieval
and visual delivery.
To further assess robustness of G2F-RAG, we deliber-
ately feed incorrect and adversarial reasoning frames and
observe that, because the prompt consistently prioritizes
grounding in the original video while our method con-
tributes only a single frame, the overall performance drop
is almost negligible, more details are in the Appendix.
5. Conclusion
We addressed a central bottleneck in retrieval-augmented
video reasoning: how external knowledge is represented
and fused with the backbone. We introduced Graph-to-
Frame RAG (G2F-RAG), a training-free paradigm that con-
verts retrieved structured knowledge into a single reason-
ing frame and delivers it in the visual space. To the best
of our knowledge, this is the first work to deliver retrieved
knowledge in the visual space for video reasoning. Instanti-
ated with a two-stage multi-agent workflow, G2F-RAG re-
duces the cross-modal competition and token burden, and
provides an auditable evidence trail. Extensive experiments
across datasets and backbones show consistent gains and in-
dicate that visual-space fusion is a more compatible path for
augmenting video LMMs than unbounded textual context.

Acknowledgement.This work is supported by the Young
Scientists Fund of the Hunan Natural Science Foundation
(Grant No.2024JJ6474), the Youth Independent Innovation
Science Fund Project of NUDT (Grant No.ZK24-08).
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
Gpt-4 technical report.arXiv preprint arXiv:2303.08774,
2023. 2
[2] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Es-
sam Sleiman, Mingchen Zhuge, Jian Ding, Deyao Zhu,
J¨urgen Schmidhuber, and Mohamed Elhoseiny. Goldfish:
Vision-language understanding of arbitrarily long videos. In
European Conference on Computer Vision, pages 251–267.
Springer, 2024. 1, 3
[3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A versatile vision-language model for un-
derstanding, localization, text reading, and beyond.arXiv
preprint arXiv:2308.12966, 2023. 2
[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui
Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao,
Chunjiang Ge, et al. Qwen3-vl technical report.arXiv
preprint arXiv:2511.21631, 2025. 1, 5, 6
[5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhao-
hai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren
Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen
Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Jun-
yang Lin. Qwen2.5-vl technical report.arXiv preprint
arXiv:2502.13923, 2025. 2, 3, 5, 6
[6] Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara
Sarto, Marcella Cornia, Lorenzo Baraldi, and Rita Cuc-
chiara. Wiki-llava: Hierarchical retrieval-augmented gener-
ation for multimodal llms. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 1818–1826, 2024. 2
[7] Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He,
Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong
Liu, and Xiang Bai. Llava-kd: A framework of distilling
multimodal large language models. InProceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 239–249, 2025. 2
[8] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu,
Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang
Yang, Zhijian Liu, et al. Longvila: Scaling long-context
visual language models for long videos.arXiv preprint
arXiv:2408.10188, 2024. 2
[9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhang-
wei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,
Zhaoyang Liu, et al. Expanding performance boundaries of
open-source multimodal models with model, data, and test-
time scaling.arXiv preprint arXiv:2412.05271, 2024. 2, 5,
6[10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen,
Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu,
Lewei Lu, et al. Internvl: Scaling up vision foundation mod-
els and aligning for generic visual-linguistic tasks. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 24185–24198, 2024. 2
[11] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin
Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang
Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-
temporal modeling and audio understanding in video-llms.
arXiv preprint arXiv:2406.07476, 2024. 2
[12] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhang-
hao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yong-
hao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P.
Xing. Vicuna: An open-source chatbot impressing gpt-4
with 90%* chatgpt quality, 2023. 2
[13] Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi. Words or
vision: Do vision-language models have blind faith in text?
InProceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 3867–3876, 2025. 2
[14] John Ellson, Emden Gansner, Lefteris Koutsofios, Stephen C
North, and Gordon Woodhull. Graphviz—open source graph
drawing tools. InInternational symposium on graph draw-
ing, pages 483–484. Springer, 2001. 5
[15] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao,
Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A
long-form multi-shot benchmark for holistic video under-
standing.Advances in Neural Information Processing Sys-
tems, 37:89098–89124, 2024. 5
[16] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo,
Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang,
Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing
video reasoning in mllms.arXiv preprint arXiv:2503.21776,
2025. 1, 2
[17] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai
Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang
Shen, Mengdan Zhang, et al. Video-mme: The first-ever
comprehensive evaluation benchmark of multi-modal llms in
video analysis. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 24108–24118, 2025.
3, 5
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and
Haofen Wang. Retrieval-augmented generation for large lan-
guage models: A survey.arXiv preprint arXiv:2312.10997,
2(1), 2023. 3
[19] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A Rossi,
Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-
augmented generation with graphs (graphrag).arXiv
preprint arXiv:2501.00309, 2024. 3
[20] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan
Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu:
Evaluating knowledge acquisition from multi-discipline pro-
fessional videos.arXiv preprint arXiv:2501.13826, 2025. 5
[21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perel-
man, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli-

hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.
arXiv preprint arXiv:2410.21276, 2024. 1, 2, 5, 6
[22] Noureldien Hussein, Efstratios Gavves, and Arnold WM
Smeulders. Videograph: Recognizing minutes-long human
activities in videos.arXiv preprint arXiv:1905.05143, 2019.
2
[23] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju
Hwang. Videorag: Retrieval-augmented generation over
video corpus.arXiv preprint arXiv:2501.05874, 2025. 3
[24] Shixin Jiang, Jiafeng Liang, Jiyuan Wang, Xuan Dong,
Heng Chang, Weijiang Yu, Jinhua Du, Ming Liu, and Bing
Qin. From specific-MLLMs to omni-MLLMs: A survey on
MLLMs aligned with multi-modalities. InFindings of the
Association for Computational Linguistics: ACL 2025, 2025.
2
[25] Long Lan, Xiao Teng, Jing Zhang, Xiang Zhang, and
Dacheng Tao. Learning to purification for unsupervised per-
son re-identification.IEEE Transactions on Image Process-
ing, 32:3338–3353, 2023. 1
[26] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang,
Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.
Mvbench: A comprehensive multi-modal video understand-
ing benchmark. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 22195–
22206, 2024. 5
[27] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan
Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He,
Chenting Wang, et al. Videochat-flash: Hierarchical com-
pression for long-context video modeling.arXiv preprint
arXiv:2501.00574, 2024. 2
[28] Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yao-
jie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han, Le Sun, and
Yongbin Li. Structrag: Boosting knowledge intensive rea-
soning of llms via inference-time hybrid information struc-
turization.arXiv preprint arXiv:2410.08815, 2024. 3
[29] Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu
Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing
Huang. Star-r1: Spatial transformation reasoning by rein-
forcing multimodal llms.arXiv preprint arXiv:2505.15804,
2025. 2
[30] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Moham-
mad Shoeybi, and Song Han. Vila: On pre-training for vi-
sual language models. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
26689–26699, 2024. 5, 6
[31] Kevin Qinghong Lin, Yuhao Zheng, Hangyu Ran, Dantong
Zhu, Dongxing Mao, Linjie Li, Philip Torr, and Alex Jin-
peng Wang. Vcode: a multimodal coding benchmark
with svg as symbolic visual representation.arXiv preprint
arXiv:2511.02778, 2025. 3
[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023. 2
[33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning. InPro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 26296–26306, 2024.[34] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Im-
proved reasoning, ocr, and world knowledge, 2024. 2
[35] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape,
Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in
the middle: How language models use long contexts.Trans-
actions of the Association for Computational Linguistics, 12:
157–173, 2024. 2
[36] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai
Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcom-
pass: Do video llms really understand videos?arXiv preprint
arXiv:2403.00476, 2024. 5
[37] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin,
Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji, Fei Chao,
Jiebo Luo, and Rongrong Ji. Video-RAG: Visually-aligned
retrieval-augmented long video comprehension. InThe
Thirty-ninth Annual Conference on Neural Information Pro-
cessing Systems, 2025. 1, 3, 5, 6
[38] AI Meta. The llama 4 herd: The beginning of a new
era of natively multimodal ai innovation.https://ai. meta.
com/blog/llama-4-multimodal-intelligence/, checked on, 4
(7):2025, 2025. 2
[39] OpenAI. Gpt-4o mini: advancing cost-efficient intelligence.
https://openai.com/index/gpt- 4o- mini-
advancing - cost - efficient - intelligence,
2024. 5
[40] Ziqi Pang and Yu-Xiong Wang. Mr. video:” mapreduce” is
the principle for long video understanding.arXiv preprint
arXiv:2504.16082, 2025. 2
[41] Weiming Ren, Wentao Ma, Huan Yang, Cong Wei, Ge
Zhang, and Wenhu Chen. Vamba: Understanding hour-
long videos with hybrid mamba-transformers.arXiv preprint
arXiv:2503.11579, 2025. 2
[42] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei
Yin, and Chao Huang. Videorag: Retrieval-augmented gen-
eration with extreme long-context videos.arXiv preprint
arXiv:2502.01549, 2025. 2, 3
[43] Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mo-
hamed Elhoseiny. Vgent: Graph-based retrieval-reasoning-
augmented generation for long video understanding. InThe
Thirty-ninth Annual Conference on Neural Information Pro-
cessing Systems, 2025. 1, 2, 3, 5, 6
[44] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell,
Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent,
Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking
multimodal understanding across millions of tokens of con-
text.arXiv preprint arXiv:2403.05530, 2024. 2, 5, 6
[45] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui
Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-
yang Lin. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution.arXiv preprint
arXiv:2409.12191, 2024. 2
[46] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long
Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Sheng-
long Ye, Jie Shao, et al. Internvl3.5: Advancing open-source

multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025. 1, 2, 5, 6
[47] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xi-
angyu Zeng, Chenting Wang, Changlian Ma, Haian Huang,
Jianfei Gao, et al. Internvideo2. 5: Empowering video
mllms with long and rich context modeling.arXiv preprint
arXiv:2501.12386, 2025. 6
[48] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-
ocr: Contexts optical compression.arXiv preprint
arXiv:2510.18234, 2025. 2
[49] Yongliang Wu, Xinting Hu, Yuyang Sun, Yizhou Zhou,
Wenbo Zhu, Fengyun Rao, Bernt Schiele, and Xu Yang.
Number it: Temporal grounding videos like flipping manga.
InProceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 13754–13765, 2025. 3
[50] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yong Liu,
Xiangtai Li, Dacheng Tao, et al. Adavideorag: Omni-
contextual adaptive retrieval-augmented efficient long video
understanding. InThe Thirty-ninth Annual Conference on
Neural Information Processing Systems, 2025. 3
[51] Ziang Yan, Xinhao Li, Yinan He, Zhengrong Yue, Xiangyu
Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang.
Videochat-r1. 5: Visual test-time scaling to reinforce mul-
timodal reasoning by iterative perception.arXiv preprint
arXiv:2509.21100, 2025. 2
[52] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv
preprint arXiv:2505.09388, 2025. 2
[53] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han,
Li Fei-Fei, and Saining Xie. Thinking in space: How mul-
timodal large language models see, remember, and recall
spaces. InProceedings of the Computer Vision and Pattern
Recognition Conference, pages 10632–10643, 2025. 1, 5
[54] Songyuan Yang, Weijiang Yu, Wenjing Yang, Xinwang Liu,
Huibin Tan, Long Lan, and Nong Xiao. Wildvideo: Bench-
marking lmms for understanding video-language interaction.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 2025. 1, 3, 5
[55] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui,
Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He,
et al. Minicpm-v: A gpt-4v level mllm on your phone.arXiv
preprint arXiv:2408.01800, 2024. 2, 5, 6
[56] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang,
Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu,
Jianqun Chen, et al. Yi: Open foundation models by 01. ai.
arXiv preprint arXiv:2403.04652, 2024. 2
[57] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang,
Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang
Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking effi-
cient mllms via architecture, data, and training recipe.arXiv
preprint arXiv:2509.18154, 2025. 5, 6
[58] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu,
Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang,
Hang Zhang, Xin Li, et al. Videollama 3: Frontier multi-
modal foundation models for image and video understand-
ing.arXiv preprint arXiv:2501.13106, 2025. 2, 5, 6[59] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An
instruction-tuned audio-visual language model for video un-
derstanding.arXiv preprint arXiv:2306.02858, 2023. 2
[60] Jinglei Zhang, Yuanfan Guo, Rolandos Alexandros
Potamias, Jiankang Deng, Hang Xu, and Chao Ma. Vtime-
cot: Thinking by drawing for video temporal grounding and
reasoning. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 24203–24213, 2025.
3
[61] Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin
Li, Houqiang Li, and Yan Lu. Deep video discovery: Agen-
tic search with tool use for long-form video understanding.
arXiv preprint arXiv:2505.18079, 2025. 2
[62] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Zi-
wei Liu, and Chunyuan Li. Video instruction tuning with
synthetic data.arXiv preprint arXiv:2410.02713, 2024. 2, 5,
6
[63] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo,
Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A frame-
work of small-scale large multimodal models.arXiv preprint
arXiv:2402.14289, 2024. 2
[64] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang
Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping
Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task
long video understanding. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages 13691–
13701, 2025. 3, 5
[65] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shen-
glong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su,
Jie Shao, et al. Internvl3: Exploring advanced training and
test-time recipes for open-source multimodal models.arXiv
preprint arXiv:2504.10479, 2025. 2, 5, 6