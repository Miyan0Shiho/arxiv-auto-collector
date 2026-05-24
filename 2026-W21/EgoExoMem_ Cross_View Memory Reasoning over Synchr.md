# EgoExoMem: Cross-View Memory Reasoning over Synchronized Egocentric and Exocentric Videos

**Authors**: Ruiping Liu, Junwei Zheng, Yufan Chen, Di Wen, Shaofang Quan, Chengzhi Wu, Jiaming Zhang, Kailun Yang, Kunyu Peng, Rainer Stiefelhagen

**Published**: 2026-05-18 17:54:55

**PDF URL**: [https://arxiv.org/pdf/2605.18734v1](https://arxiv.org/pdf/2605.18734v1)

## Abstract
Egocentric memory is widely used in embodied intelligence, but it may be insufficient for comprehensive spatial-temporal reasoning. Inspired by human recall from both field and observer perspectives, we introduce EgoExoMem, the first benchmark for cross-view memory reasoning over synchronized egocentric and exocentric videos. EgoExoMem contains $2.6K$ high-quality MCQs across eight temporal, spatial, and cross-view QA types. To support dual-view retrieval, we propose E$^2$-Select, a training-free frame selection method for synchronized ego-exo videos. It combines relevance-based budget allocation with per-view k-DPP sampling to handle view asymmetry and cross-view temporal consistency. Experiments show that ego and exo views provide complementary memory cues, while existing MLLMs remain far from solving the benchmark: the best model reaches only $55.3\%$. E$^2$-Select achieves state-of-the-art performance of $58.2\%$ over frame-selection and RAG-based memory baselines. Further analysis reveals systematic view-preference conflicts between question framing and answer grounding, underscoring the novelty and challenge of cross-view memory reasoning.

## Full Text


<!-- PDF content starts -->

EgoExoMem: Cross-View Memory Reasoning over
Synchronized Egocentric and Exocentric Videos
Ruiping Liu1, Junwei Zheng1,2, Yufan Chen1,3, Di Wen1, Shaofang Quan1, Chengzhi Wu1,
Jiaming Zhang4,Kailun Yang4,Kunyu Peng1,5∗,Rainer Stiefelhagen1
1Karlsruhe Institute of Technology2ETH Zurich3University of Oxford
4Hunan University5INSAIT, Sofia University “St. Kliment Ohridski”
Figure 1: EgoExoMem requires reasoning over synchronized ego-exo memory streams to answer
questions unanswerable from either viewpoint alone. Blue indicates the temporal condition, green
indicates keyframes or the correct answer, and red indicates distractors.
Abstract
Egocentric memory is widely used in embodied intelligence, but it may be insuffi-
cient for comprehensive spatial-temporal reasoning. Inspired by human recall from
both field and observer perspectives, we introduce EgoExoMem, the first bench-
mark for cross-view memory reasoning over synchronized egocentric and exocen-
tric videos. EgoExoMem contains 2.6K high-quality MCQs across eight temporal,
spatial, and cross-view QA types. To support dual-view retrieval, we propose E2-
Select, a training-free frame selection method for synchronized ego-exo videos. It
combines relevance-based budget allocation with per-view k-DPP sampling to han-
dle view asymmetry and cross-view temporal consistency. Experiments show that
ego and exo views provide complementary memory cues, while existing MLLMs
remain far from solving the benchmark: the best model reaches only 55.3% . E2-
Select achieves state-of-the-art performance of 58.2% over frame-selection and
RAG-based memory baselines. Further analysis reveals systematic view-preference
conflicts between question framing and answer grounding, underscoring the novelty
and challenge of cross-view memory reasoning. The source code and dataset can
be found athttps://github.com/RuipingL/EgoExoMem.
∗Corresponding author: kunyu.peng@kit.edu;First author: ruiping.liu@kit.edu
Preprint.arXiv:2605.18734v1  [cs.CV]  18 May 2026

1 Introduction
With the rapid advancement of multimodal large language models, embodied agents are increasingly
expected to perceive, remember, and reason about complex real-world environments [ 55]. Central
to this capability is memory: the ability to retain and retrieve past observations to support informed
decision-making and long-horizon task execution [ 99,19]. However, existing memory frameworks
for embodied perception are predominantly built upon egocentric observations [ 25,19], capturing the
world exclusively from the agent’s own first-person viewpoint. While egocentric memory has been
widely studied for tasks such as episodic event recall [ 102,103,104], scene understanding [ 101,63],
and navigation [ 111], it remains fundamentally limited by the agent’s physical vantage point, leaving
out-of-view regions and their spatial relationships inconsistently observed. This raises a critical
question:is egocentric memory alone sufficient for comprehensive spatial understanding and temporal
reasoning in embodied environments?
Real-world scenarios reveal fundamental limitations of purely egocentric memory. For holistic scene
understanding, an agent’s egocentric stream captures its own interactions but offers limited visibility
into other agents’ interactions within the shared space. While equipping all agents with smart
glasses has been proposed [ 102,42], this is impractical due to device cost and wearable burden [ 53].
Egocentric recording is also ill-suited for capturing whole-body movements, particularly in healthcare
contexts such as fall detection [ 100]. Given the growing prevalence of exocentric infrastructure,
including hospital surveillance [ 81,23] and smart home cameras [ 9,1], we propose complementing
egocentric memory with exocentric observations to address these limitations.
Humans naturally encode experiences from two complementary perspectives: field memory, which
relives events through a first-person viewpoint, and observer memory, which recalls the same event
from a third-person vantage point [ 66]. These perspectives mirror the brain’s parallel egocentric
and allocentric reference frames [ 7], and their interplay supports situation awareness [ 17]. Yet,
existing ego-exo works focus on imitation learning or unidirectional knowledge transfer [ 108], with
methods treating the two streams independently [ 32] and benchmarks evaluating only local view-clip
matching [ 28] or view-invariant temporal consistency [ 39,78], rather than reasoning over the full,
synergistic memory formed by both views.
To address this gap, we introduceEgoExoMem, a large-scale benchmarkcomprising 2.6K human-
verified Multiple Choice Questions (MCQs) designed to evaluate memory-based reasoning over
synchronized egocentric and exocentric videos sourced from EgoExo4D [ 26] and LEMMA [ 34].
We characterize memory-based reasoning along eight QA types covering temporal, spatial, and
cross-view dimensions: Habitual Location, Instantaneous Position, Resulting Location, Egocentric
Direction, Object State, Allocentric Relation, Third-Person Activity, and Temporal Ordering, with
an example illustrated in Fig. 1. A text-only check is applied to ensure the vision-dependency of all
answers. Motivated by the view-asymmetric and temporally synchronized nature of EgoExoMem,
we further proposeE2-Select, a simple yet effective frame selection methodwith relevance-based
budget allocation and k-DPP sampling, explicitly designed to handle view asymmetry and cross-view
temporal consistency beyond single-view memory methods.
Extensive experimentsare conducted to validate the reasonableness of the EgoExoMem task,
benchmark the dataset, demonstrate the effectiveness of E2-Select, and analyze failure cases. Propri-
etary [ 12] and open-source MLLMs [ 45,91,2] are evaluated on EgoExoMem as zero-shot baselines,
confirming the task’s challenge. Memory mechanisms spanning frame selection [ 88,54], text re-
trieval [ 79,41], visually-grounded retrieval [ 61,33], and structured memory retrieval [ 42,104] are
used to benchmark EgoExoMem, while our E2-Select achieves state-of-the-art performance with an
averaged MCQ accuracy of 58.2% . Comprehensive ablation studies support the design choices of
E2-Select. Failure case analysis reveals a view-preference conflict between question framing and
answer grounding, highlighting the complementary potential of synergistic ego-exo views.
2 Related Work
Ego-Exo Video Understanding and Reasoning.Cross-view understanding spans view transla-
tion [ 60,108,62,67,82], correspondence learning [ 30,32], cross-view grounding [ 22,85,59], and
audio-visual association [ 31,36]. View-conditioned generation has also advanced rapidly, including
ego-exo synthesis [ 97,72,70], viewpoint-aware anticipation [ 83], and analysis of active versus pas-
2

Table 1: Comparison of related video QA benchmarks. “–” means not available.
Benchmark Venue #QA #Videos #Tasks View Sync Memory QA Cross-view QA
EgoSchema [64] NeurIPS’23 5.1k 5.1k 1 Ego –× ×
VSI-Bench [101] CVPR’25 5.0k 288 8 Ego –✓×
EgoMemoria [103] ICLR’25 7.0k 629 – Ego –✓×
EgoPlan-Bench [10] ECCV’24 4.9k – 1 Ego –× ×
EgoExoLearn [32] CVPR’24 2.2k 747 3 Ego+Exo Async×✓
EgoExoBench [28] NeurIPS’25 7.3k – 11 Ego+Exo Both×✓
MA-EgoQA [42] arXiv’26 1.7k – 5 Multi-ego –×✓
EgoExoMem (Ours)2026 2.6k 390 8 Ego+Exo Sync✓ ✓
sive perspectives [ 57]. Beyond generation, view-invariant representation learning [ 71] and imitation
error detection [ 50] further demonstrate the value of cross-view alignment for fine-grained activity
understanding. Recent benchmarks have also expanded the ego-exo scope: EPFL-Smart-Kitchen [ 6]
evaluates action and motion understanding in multi-modal ego-exo settings, and SAW-Bench [ 47]
probes situated awareness in real-world environments. Closest to our setting, Raviet al.[ 77] show
that off-screen context can be recovered from egocentric observations, and EgoExoLearn [ 32] bench-
marks skill assessment across paired viewpoints. Yet, existing methods treat the two streams as inputs
to a shared representation rather than as asymmetric sources of complementary evidence, a distinction
that prior cross-view benchmarks do not evaluate. Relevant benchmarks are summarized in Tab. 1.
Memory-Augmented Video Understanding.Episodic memory QA over egocentric video is well
established [ 5,14,35,37,4]. Recent work pushes toward longer temporal horizons [ 102,44,18] and
richer memory structures, including spatial mind palaces [ 24], dynamic memory refinement [ 29], and
multi-agent memory [ 65,42]. Fine-grained spatiotemporal grounding [ 52,20,96] and egocentric
captioning [ 15,40] further underpin retrieval capabilities. However, egocentric streams capture fine-
grained interactions but miss the spatial layout and third-person activities. Exocentric streams offer
the reverse perspective but lack close-up manipulation detail. EgoExoMem exploits this asymmetry
with QA types that are unresolvable from either view alone.
Spatial and Procedural Video QA Benchmarks.Spatial QA benchmarks probe object size, distance,
room layout [ 101], embodied cognition [ 13], and broad multi-modal egocentric perception [ 58,74,
38]. Procedural and task-oriented QA is addressed across activity sequences [ 35], instructional
video settings [ 76], and long-horizon retrieval tasks [ 102,18]. Chain-of-thought and cross-modal
reasoning in egocentric settings are targeted by Egothinker [ 73] and EgoCross [ 51]. Object-centric
and referential understanding [ 89,46,87] connects directly to EgoExoMem’s localization and relation-
change question types, while visual query localization [ 37] formalizes thewhere did I last see X
task that underlies our displacement queries. Scene graph understanding [ 80,84,11], egocentric text
QA [ 109,110], low-light settings [ 107], and assistive scenarios [ 94,92,93,90,49,86,105,106,95]
further round out the evaluation landscape. Taken together, these benchmarks span a rich set of
reasoning skills but do not require joint ego-exo memory, which is the gap EgoExoMem targets.
3 EgoExoMem
3.1 Video Selection
We select synchronized videos from two datasets, LEMMA [ 34] and EgoExo4D [ 26]. Since our
goal is to capture spatial changes over time through memory, the camera wearer must interact
extensively with the environment. We therefore retain only thecookingandbike repairscenarios
from EgoExo4D, which involve substantial environmental interaction, and exclude the remaining
skilled activity categories. To capture activities of persons other than the camera wearer, we use all
multi-agent videos from LEMMA, supplemented by single-agent videos from its test set. The video
length distributions are shown in Fig. 4 (a): LEMMA ( M= 111.3s ,σ= 74.8s ) and EgoExo4D
(M= 425.0s,σ= 339.3s).
3.2 QA Types
We refer to QA types from egocentric episodic memory [ 25,102] and spatial intelligence bench-
marks [101, 8, 63], adapting them to the synergistic nature of ego-exo memory as shown in Fig. 2.
3

Figure 2: Illustrative examples of the eight QA types (Q1–Q8) in EgoExoMem, covering object
location, spatial direction, object state, human activity, and temporal ordering. Each task requires
grounding answers in visual evidence across frames, highlighting fine-grained spatial-temporal
reasoning in real-world scenes.
Habitual Location (HL)tests persistent spatial map retention by asking where an object is habitually
located, requiring semantic spatial memory integrated over the full video. Ego and exo views con-
tribute complementarily: the egocentric stream resolves fine-grained proximity, while the exocentric
view anchors answers in global scene layout.Instantaneous Position (IP)probes the model’s ability
to bridge egocentric action observations and allocentric spatial positions, requiring a latent mapping
between first-person and world-centered reference frames.Resulting Location (RL)evaluates spatial
memory across an event boundary, combining temporal grounding with spatial memory update to
track an object’s location after manipulation.Egocentric Direction (ED)assesses fine-grained spatial
awareness within the first-person reference frame, asking where a target lies relative to the camera
wearer (e.g., back-left). While ego-view-anchored, exocentric frames provide complementary global
context to resolve directions occluded from the wearer’s perspective. Models must maintain a contin-
uous spatial representation as the wearer moves.Object State (OS)tests whether the model identifies
the physical state of a manipulated object at a specified moment (e.g., uncut/sliced, whole/eaten).
Object states change incrementally as procedural actions unfold, demanding frame-accurate under-
standing of manipulation sequences from either or both views.Allocentric Relation (AR)probes
allocentric spatial reasoning by asking where one object lies relative to another (e.g., left of, directly
behind). Both ego and exo views are viable but imperfect, as each provides a different vantage on
the same configuration, potentially requiring cross-view integration.Third Person Activity (TPA)
requires simultaneous spatial tracking of two people: while one agent performs a stated action, the
model must locate the second agent in the scene. It benefits substantially from the exocentric view,
which captures the full room layout, including agents outside the egocentric field of view.Temporal
Ordering (TO)evaluates episodic memory for procedural sequences by requiring models to arrange
sub-actions into their correct temporal order. The primary challenge is temporal-episodic: models
must recall event ordering across potentially long video segments, where the egocentric view provides
richer action-level detail and the exocentric view aids in disambiguating simultaneous or spatially
separated sub-events.
4

3.3 Benchmark Construction
Fig. 3 illustrates the benchmark construction pipeline: MCQs are first generated, then human-edited
and filtered for accuracy, and finally subjected to a text-only check to ensure vision dependency.
Figure 3: Overview of the EgoExoMem
benchmark construction pipeline.MCQ Generation.QA pairs are generated programmat-
ically from structured annotations and then verbalized
using GPT-5.4 [ 69]. The pipeline is shared across both
datasets but draws on different annotation sources. Each
question is formatted as a four-option multiple-choice
question with exactly one correct answer.
For LEMMA, the primary annotation source is per-frame
HOI labels, which record tuples of (action verb, inter-
acted object, furniture anchor location) alongside person
bounding boxes in the exocentric view. We extract contigu-
ous action segments and derive structured ground-truth
facts (object resting location, person position, object dis-
placement, egocentric object direction, state transitions)
as seeds for question generation. For EgoExo4D, we use
keystep annotations (step name, step description, start/end
timestamps) to identify activity segments, and relation
annotations (per-frame object bounding box tracks) to ac-
curately localize objects in the scene. Egocentric object
directions are computed from the median of bounding box
centroid across a track sequence.
For both datasets, each candidate is passed to GPT-5.4 with
up to two keyframe images and a structured ground-truth
context, producing a natural-language question and four
answer options. People are identified by visual appearance
rather than abstract labels. The keyframe selection varies
by question type:HLuses a representative exocentric
frame at the dominant get/put action;IPuses a window of ±5 surrounding exocentric frames around
the action timestamp;RLuses an egocentric frame at pickup and an exocentric frame at placement;
EDuses an egocentric frame, with the egocentric direction derived from the interacted object’s
bounding box centroid;OSuses keyframes at both the state-changing event and the later query action;
ARuses a single exocentric keyframe, with the spatial relation between two objects determined
entirely by the model from the image;TOis derived directly from annotation timestamps.TPA
(simultaneous two-person localization) is generated for LEMMA only, as EgoExo4D lacks concurrent
multi-person spatial annotations.
Human Editing and Filtering.The generated MCQs are not guaranteed to be fully accurate. To
ensure dataset quality, three human annotators reviewed all generated MCQs and either corrected
them (for instance, fixing confusion over human appearance, location ambiguity, or incorrect give/get
relations) or discarded them entirely (such as those with ambiguous reference objects or objects that
never appear in the video). The interface used for human verification is shown in Fig. 6.
Text-only Check.The label noise in the MCQs is minimized through human editing and filtering. To
ensure the dataset remains challenging and vision-dependent, we prompt GPT-4o [ 68] at temperature
values of 0 and 1 to answer each MCQ based solely on the question text, without any video input. If
GPT-4o answers correctly at both temperatures, the question is deemed common-sense answerable; if
incorrect at both, vision-dependent; if mixed, potentially chance-correct. We discard all common-
sense answerable questions and remove a subset of chance-correct ones to ensure that the text-only
accuracy of GPT-4o ( Temperature= 0 ) for each QA type remains below 45%. The resulting
distribution of QA types and the word cloud of questions are shown in Fig. 4 (b-c).
4 E2-Select: EgoExo Frame Selection
Unlike prior ego-exo benchmarks that focus on view matching or local clip understanding, EgoEx-
oMem treats synchronized egocentric and exocentric videos as a persistent dual-view memory. It
requires models to integrate complementary evidence across time: ego views provide fine-grained
5

Figure 4: Dataset statistics of EgoExoMem. (a) Video length distribution for LEMMA and EgoExo4D
subsets. (b) Distribution of QA types across the benchmark. (c) Word cloud of question text,
illustrating the diversity of spatial and temporal vocabulary.
hand-object and wearer-relative cues, whereas exo views offer global layout, other-agent activity, and
out-of-view context.
To address this setting, we propose E2-Select, a training-free frame selection method with independent
view scoring, relevance-based budget allocation, and per-view k-DPP sampling. This design preserves
view-specific evidence, reduces temporal redundancy, and implicitly routes each query to the most
informative ego-exo memory evidence.
Independent View Scoring.We first conduct pre-selection of individual views according to the
query-visual relevance. We compute query-frame relevance independently for each view to avoid
cross-view interference. Given a query q, ego frames {fe
i}Ne
i=1, and exo frames {fx
j}Nx
j=1where Ne
denotes the number of frames of egocentric video and Nxdenotes the number of frames of exocentric
video, we compute CLIP [75] cosine similarities separately according to Eq. 1 and Eq. 2.
se(i) =sim(ϕ visual (fe
i), ϕtext(q)), i= [1, . . . , N e],(1)
sx(j) =sim 
ϕvisual (fx
j), ϕtext(q)
, j= [1, . . . , N x],(2)
where ϕvisual (·)andϕtext(·)denote the CLIP visual encoder and text encoder. Each view is evaluated
independently in its own semantic subspace, which avoids cross-view score interference and prevents
a more dominant view from overwhelming the other during joint retrieval. When the two views are
synchronized,N eis equal toN x.
Relevance-Based Budget Allocation.To address view asymmetry, we allocate the total frame
budget Kproportionally to the aggregated query relevance of each view (Eq. 3): a view whose
frames are collectively more relevant to the query receives a larger frame budget, providing a soft and
query-dependent alternative to hard view selection.
Ke=$
K·P
ise(i)P
ise(i) +P
jsx(j)'
, K x=K−K e,(3)
where⌊·⌉denotes rounding to the nearest integer. This formulation implicitly performs view selection
while maintaining a better balance than hard-selecting between ego and exo views at each timestamp.
Per-View k-DPP Sampling.To address temporal redundancy within each view, we replace coverage-
based selection with a Determinantal Point Process (DPP), which provides a principled diversity
guarantee via determinant geometry.
For the ego view, we construct a quality-diversity kernel matrixLe∈RNe×Ne, as shown in Eq. 4.
Le
ij=se(i)·sim 
ϕvisual (fe
i), ϕvisual (fe
j)
·se(j),(4)
The diagonal entries Le
ii=se(i)2encode per-frame relevance; off-diagonal entries penalize the
simultaneous selection of pairs of frames with similar visual content. We then draw a subset Seof
sizeK efrom the k-DPP defined byLeas in Eq. 5.
P(Se)∝det 
Le
Se
,|S e|=K e,(5)
where Le
Sedenotes the principal submatrix of Lerestricted to the frame indices in Se. Because
the determinant measures the volume spanned by the selected frame embeddings, maximizing it
encourages a subset that is both relevant to the query and visually diverse, rather than merely
well-spaced in time.
6

Table 2: Per-category accuracy (%) of MLLMs on EgoExoMem across different input views.
Model Frame Input Views HL IP RL ED OS AR TPA TO Avg
Gemini 2.5 Flash [12] concat Ego 58.1 62.5 63.538.9 56.454.448.934.7 52.2
Gemini 2.5 Flash [12] concat Exo 51.472.259.6 34.6 48.8 44.6 45.5 38.9 49.5
Gemini 2.5 Flash [12] concat Ego+Exo 62.9 70.9 66.7 38.7 53.7 56.3 48.0 44.9 55.3
InternVL3.5 [91] concat Ego 50.7 55.5 55.138.255.5 43.3 47.7 39.3 48.2
InternVL3.5 [91] interleaved Ego53.857.7 57.7 36.1 56.7 42.549.541.9 49.5
InternVL3.5 [91] concat Exo 47.2 65.4 50.6 33.0 50.0 39.2 36.0 37.0 44.8
InternVL3.5 [91] interleaved Exo 46.966.351.3 34.0 44.5 37.9 35.7 38.0 44.3
InternVL3.5 [91] concat Ego+Exo 53.1 65.2 64.1 36.4 57.9 42.1 44.8 41.9 50.7
InternVL3.5 [91] interleaved Ego+Exo 50.7 63.1 58.3 35.3 55.5 44.2 39.2 42.2 48.6
LLaV A-OV [45] concat Ego 48.3 53.6 52.6 32.7 50.0 37.9 36.9 36.3 43.5
LLaV A-OV [45] interleaved Ego48.655.356.4 35.651.8 38.337.035.3 44.8
LLaV A-OV [45] concat Exo 47.264.242.9 31.2 50.0 37.9 34.136.643.0
LLaV A-OV [45] interleaved Exo 46.2 62.5 42.3 29.6 50.6 36.7 34.8 35.0 42.2
LLaV A-OV [45] concat Ego+Exo 51.0 63.6 50.0 34.8 52.4 41.2 34.6 35.6 45.4
LLaV A-OV [45] interleaved Ego+Exo 47.6 60.1 53.2 35.3 53.7 41.2 33.1 36.0 45.0
Qwen2.5-VL [3] concat Ego 50.0 49.3 53.236.943.3 38.338.5 37.043.3
Qwen2.5-VL [3] interleaved Ego 51.7 50.4 57.1 34.0 40.9 38.3 35.3 35.0 42.8
Qwen2.5-VL [3] concat Exo 50.7 57.1 51.9 30.4 41.5 40.0 35.7 36.6 43.0
Qwen2.5-VL [3] interleaved Exo 50.0 56.1 53.2 30.1 37.8 40.4 35.637.042.5
Qwen2.5-VL [3] concat Ego+Exo 53.1 58.2 58.3 31.7 42.7 40.4 34.3 36.0 44.3
Qwen2.5-VL [3] interleaved Ego+Exo 53.5 57.4 54.5 34.0 45.7 43.3 32.9 36.0 44.7
We apply the same k-DPP sampling procedure to the exocentric stream, obtaining a subset Sxwith
|Sx|=K x. For efficiency, we use approximate Cholesky-based k-DPP [ 16] inference, which has
complexityO(N2K).
Timestamp-Ordered Merge.After sampling view-specific subsets, we merge the selected ego and
exo frames according to their original timestamps as shown in Eq. 6.
F= Sort t 
{fe
i|i∈S e} ∪ {fx
j|j∈S x}
,(6)
where tdenotes the original video timestamps. This chronological merge preserves the temporal
structure of the synchronized ego-exo memory while retaining view-specific evidence. The resulting
sequenceFcontainsKframes and is used as the visual input to the MLLM.
5 Experiments
Since we propose the first benchmark dedicated to egocentric and exocentric memory, EgoExoMem,
we dedicate ourselves to answering four research questions: (1)Can egocentric and exocentric
streams serve as complementary memory sources for spatial and temporal reasoning?(2)How do
existing MLLMs perform on EgoExoMem?(3)How does E2-Select compare to existing memory
retrieval strategies on EgoExoMem?(4)under what conditions does joint ego-exo memory retrieval
deteriorate, and what are the underlying causes?
5.1 Implementation Details
We evaluate a proprietary MLLM, Gemini 2.5 Flash [ 12], and open-source ones, including In-
ternVL3.5 [ 91], LLaV A-OneVision [ 45], and Qwen2.5-VL [ 2], to investigate the synergy of ego-
centric and exocentric memory and their performance on EgoExoMem. For this part, we explore
two strategies for inputting frames from two video sources: concatenation [ 42,28], which preserves
intra-video temporal consistency but provides weaker inter-video frame correspondence; and inter-
leaving [ 48], which improves inter-video frame correspondence but disrupts intra-video temporal
consistency. Based on the best-performing open-source MLLM, InternVL3.5, we further experiment
with different memory mechanisms, RAG-based [ 79,41,61,33] and frame selection for single
video [88, 54], and our E2-Select for two synchronized videos.
For fair evaluation, all reasoning models receive 32frames as input. For MLLMs, 16frames are
uniformly sampled from either or both videos. For memory mechanisms, all videos are first processed
at1FPS. For the RAG-based methods, videos are segmented into clips, and the top- k(k= 8 )
most relevant clips are retrieved, each represented by 4frames. The captions used for retrieval are
7

Table 3: Per-category accuracy (%) of different memory mechanisms on EgoExoMem.
Method Views HL IP RL ED OS AR TPA TO Avg
AKS [88] Ego64.764.2 62.8 37.7 56.1 47.955.944.8 54.2
AKS [88] Exo 50.0 68.5 54.5 33.3 46.3 42.9 42.2 38.3 47.0
BOLT [54] Ego 62.9 63.6 62.240.358.5 51.3 55.6 45.9 55.0
BOLT [54] Exo 51.0 69.5 59.6 34.3 44.5 43.8 41.7 39.6 48.0
BM25 [79] Ego 51.7 59.6 62.8 34.3 51.8 44.6 46.0 45.2 49.5
BM25 [79] Exo 47.9 66.0 57.7 30.6 43.9 44.2 39.8 37.6 46.0
BM25 [79] Ego+Exo 54.2 66.3 62.8 37.4 51.8 45.8 43.8 42.9 50.6
DPR [41] Ego 52.8 60.6 64.7 34.5 48.2 44.2 45.8 44.9 49.5
DPR [41] Exo 48.6 66.3 54.5 29.1 42.1 41.2 39.7 37.6 44.9
DPR [41] Ego+Exo 56.3 68.2 67.3 36.9 45.7 40.8 45.8 41.9 50.4
VideoRAG [33] Ego 47.2 53.6 62.8 36.6 53.0 40.8 48.0 40.3 47.8
VideoRAG [33] Exo 47.2 69.0 60.3 29.6 46.3 37.9 41.4 36.0 46.0
VideoRAG [33] Ego+Exo 49.0 66.6 62.2 33.5 57.9 43.8 41.0 39.6 49.2
Video-RAG [61] Ego 56.3 55.3 50.0 36.9 50.6 47.1 48.8 35.6 47.6
Video-RAG [61] Exo 49.3 63.9 51.9 34.0 42.7 43.8 40.6 35.3 45.2
Video-RAG [61] Ego+Exo 53.1 66.0 57.1 35.8 50.6 45.8 43.0 35.0 48.3
EgoMAS [42] Ego+Exo 45.5 56.6 51.3 28.6 62.8 35.8 37.0 38.3 44.5
WorldMM [104] Ego+Exo 36.5 44.8 39.4 23.1 36.5 36.3 35.1 31.5 35.4
E2-Select (ours) Ego+Exo 59.8 72.8 64.1 34.3 57.9 51.7 47.9 46.2 56.3
Table 4: Ablation study of E2-Select.
Model Frame Input Views HL IP RL ED OS AR TPA TO Avg
Gemini 2.5 Flash [12] concat Ego+Exo 62.9 70.9 66.738.753.756.348.0 44.9 55.3
Gemini 2.5 Flash [12] k-DPP Ego 67.8 69.5 68.6 38.4 57.956.348.6 45.9 56.6
Gemini 2.5 Flash [12] k-DPP+hard selection Ego+Exo 67.1 73.069.034.1 57.3 54.452.346.7 56.7
Gemini 2.5 Flash [12] k-DPP+soft allocation(E2-Select) Ego+Exo 72.5 77.0 66.0 36.8 60.4 54.2 51.6 47.4 58.2
InternVL3.5 [91] concat Ego+Exo 53.1 65.264.136.457.942.1 44.8 41.9 50.7
InternVL3.5 [91] k-DPP Ego63.363.3 62.2 35.3 54.9 52.155.144.6 53.8
InternVL3.5 [91] AKS [88]+soft allocation Ego+Exo 58.0 71.7 68.6 37.1 58.5 46.2 45.8 42.6 53.6
InternVL3.5 [91] BOLT [54]+soft allocation Ego+Exo 56.6 68.2 64.1 37.9 54.3 47.9 44.1 43.6 52.1
InternVL3.5 [91] k-DPP+hard selection Ego+Exo 60.5 71.7 63.538.252.453.845.1 43.6 53.6
InternVL3.5 [91] k-DPP+soft allocation(E2-Select) Ego+Exo 59.8 72.8 64.1 34.3 57.9 51.7 47.9 46.2 56.3
generated with Gemini 2.5 Flash [ 12]. All other settings follow the original configurations of the
methods. Our experiments are conducted on 4 NVIDIA A100 (40GB) GPUs.
5.2 Results
Reasonability of EgoExoMem and Performance of MLLMs.Tab. 2 evaluates MLLMs with two
simple video frame input strategies: concatenation and interleaving. To assess the complementarity
of the two views, we keep the total number of frames constant by duplicating the same video for
single-video input, ensuring any performance gain stems from view complementarity rather than
additional temporal information. Overall, combining both views consistently outperforms single-
view input across all evaluated models.This supports the reasonability of leveraging both views
in EgoExoMem.For instance, Gemini 2.5 Flash improves from 52.2% (Ego) and 51.4% (Exo) to
55.3% with Ego+Exo, and InternVL3.5 improves from 48.2% (Ego) and 44.3% (Exo) to 50.7%
with Ego+Exo under concatenation. For Habitual Location, Resulting Location, Object State, and
Allocentric Relation, the cross-view combination is clearly preferred across models. Instantaneous
Position favors the exocentric view, and Egocentric Direction favors the egocentric view. Notably,
Third Person Activity consistently favors the egocentric view across all models, with InternVL3.5
showing a gap as large as 13.8% (49.5% vs.35.7% ), which is counterintuitive and will be analyzed
in the following section. Temporal Ordering shows no significant difference across egocentric,
exocentric, and mixed inputs, with scores remaining comparable across all input strategies.
Despite being state-of-the-art MLLMs, all models achieve relatively low average scores, with the
best-performing model, Gemini 2.5 Flash, reaching only 55.3% , highlighting the challenge of
8

Figure 5: Failure case analysis. (a) Question-aware view dependency measured by CLIP similarity
across all QA types. (b) View-specific emphasis on different cues, such as action in the ego view and
appearance in the exo view, even when the answers are visible in both views.
EgoExoMem. Among open-source models, InternVL3.5 achieves the best performance and is
therefore adopted as the reasoning model for the subsequent memory mechanism evaluation.
Performance of Memory Mechanisms.Tab. 3 demonstrates the performance of memory mech-
anisms in order of increasing complexity, including frame selection (AKS [ 88] and BOLT [ 54]),
text retrieval (BM25 [ 79] and DPR [ 41]), visually-grounded retrieval (Video-RAG [ 61] and Vide-
oRAG [ 33]), and structured memory retrieval (EgoMAS [ 42] and WorldMM [ 104]). Consistent with
the frame combination results in Tab. 2, retrieving from both egocentric and exocentric streams yields
better performance than single-view inputs across RAG-based methods. However, despite their strong
performance on long-video benchmarks, structured retrieval methods underperform on EgoExoMem,
whereas simpler approaches, frame selection and text retrieval, prove more effective. We attribute this
to the minute-level duration of our videos: for such short clips, complex retrieval pipelines introduce
unnecessary overhead and may be redundant [98].
As frame selection methods demonstrate superior performance on single-view understanding, we
propose E2-Select, the first frame selection method for dual-view ego-exo inputs. As shown in Tab. 3,
E2-Select achieves superior performance over all baselines. A comprehensive ablation study is further
conducted to verify the design choices in Tab. 4. Though AKS [ 88] and BOLT [ 54] achieve better
performance on single-view ego input, k-DPP [ 43] combines the two views more effectively. As
AKS and BOLT rely on temporal structure estimation and single-video saliency calibration, their
performance degrades under cross-view budget allocation. In contrast, k-DPP natively supports any
frame budget kand selects frames by maximizing diversity in feature space, making it distribution-
agnostic and robust to the domain shift between ego and exo views. We also compare soft allocation
with hard selection, which picks the more query-similar view at each timestep and then applies k-DPP.
It performs on par with single-view k-DPP, failing to fully exploit the complementarity of dual views.
Failure Cases and Potential Reasons.Among all baselines, it is counterintuitive that Third Person
Activity relies significantly on the egocentric view. By examining the keyframes used to generate the
MCQs, we find that many questions can be answered from either view due to the small room settings
in LEMMA [ 34] and the collaborative nature of the tasks, as shown in Fig. 5 (b). This raises the
question ofwhether the view most relevant to the answer is also the most relevant to the question. To
investigate this, we report the question-aware view-dependency measured by CLIP for all question
types in Fig. 5 (a). We observe that the view preferences of the question and the answer for Third
Person Activity (Tab. 2 and Tab. 3) are significantly different: the question favors the exocentric
view, whereas the answer favors the egocentric view. This discrepancy causes severe degradation of
TPA performance when frame selection is based solely on question-aware similarity, highlighting the
necessity of synergy between both views.
6 Conclusion
We present EgoExoMem, the first benchmark for memory-based reasoning over synchronized ego-
exo video. Spanning eight QA types across spatial, temporal, and cross-view memory, it reveals
that neither view alone suffices for comprehensive understanding, and that existing MLLMs and
memory mechanisms fail to fully exploit dual-view complementarity. To fill the gap in multi-view
frame selection, we propose E2-Select, which achieves superior performance via relevance-based
budget allocation and k-DPP sampling that accounts for view asymmetry and cross-view temporal
9

consistency. Failure analysis further exposes a systematic view-dependency mismatch for Third
Person Activity, motivating joint query-answer view routing in future work. We hope EgoExoMem
and E2-Select serve as a foundation for cross-view memory reasoning in embodied AI.
Acknowledgment
This work was performed on the HoreKa supercomputer funded by the Ministry of Science, Research
and the Arts Baden-Württemberg and by the Federal Ministry of Education and Research. The
authors also acknowledge support by the state of Baden-Württemberg through bwHPC and the
German Research Foundation (DFG) through grant INST 35/1597-1 FUGG. The project is funded
by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – SFB 1574 –
471687386. This project is also supported in part by the National Natural Science Foundation of
China under Grant No. 62473139, in part by the Hunan Provincial Research and Development
Project (Grant No. 2025QK3019), and in part by the State Key Laboratory of Autonomous Intelligent
Unmanned Systems (the opening project number ZZKF2025-2-10). This research was partially
funded by the Ministry of Education and Science of Bulgaria (support for INSAIT, part of the
Bulgarian National Roadmap for Research Infrastructure).
10

References
[1] Ring home security systems.https://ring.com, 2024.
[2]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao
Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report.arXiv preprint
arXiv:2511.21631, 2025.
[3]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang,
Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report.arXiv preprint
arXiv:2502.13923, 2025.
[4]Ziyi Bai, Ruiping Wang, and Xilin Chen. Glance and focus: Memory prompting for multi-event
video question answering. InNeurIPS, 2023.
[5]Leonard Bärmann and Alex Waibel. Where did I leave my keys?—Episodic-memory-based
question answering on egocentric videos. InCVPRW, 2022.
[6]Andy Bonnetto, Haozhe Qi, Franklin Leong, Matea Tashkovska, Mahdi Rad, Solaiman Shokur,
Friedhelm Hummel, Silvestro Micera, Marc Pollefeys, and Alexander Mathis. EPFL-Smart-
Kitchen: An ego-exo multi-modal dataset for challenging action and motion understanding in
video-language models. InNeurIPS, 2025.
[7]Neil Burgess. Spatial memory: how egocentric and allocentric combine.Trends in Cognitive
Sciences, 2006.
[8]Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, and Eli
Shlizerman. SA VVY: Spatial awareness via audio-visual LLMs through seeing and hearing.
InNeurIPS, 2025.
[9]Xi Chen, Julien Cumin, Fano Ramparany, and Dominique Vaufreydaz. MuRAL: A multi-
resident ambient sensor dataset annotated with natural language for activities of daily living.
InICIE, 2026.
[10] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying
Shan, and Xihui Liu. EgoPlan-Bench: Benchmarking multimodal large language models for
human-level planning.International Journal of Computer Vision, 2026.
[11] Anoop Cherian, Chiori Hori, Tim K Marks, and Jonathan Le Roux. (2.5+ 1) D spatio-temporal
scene graphs for video question answering. InAAAI, 2022.
[12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit
Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long context, and next generation agentic
capabilities.arXiv preprint arXiv:2507.06261, 2025.
[13] Ronghao Dang, Yuqian Yuan, Wenqi Zhang, Yifei Xin, Boqiang Zhang, Long Li, Liuyi Wang,
Qinyang Zeng, Xin Li, and Lidong Bing. ECBench: Can multi-modal foundation models
understand the egocentric world? A holistic embodied cognition benchmark. InCVPR, 2025.
[14] Samyak Datta, Sameer Dharur, Vincent Cartillier, Ruta Desai, Mukul Khanna, Dhruv Batra,
and Devi Parikh. Episodic memory question answering. InCVPR, 2022.
[15] Anna Deichler and Jonas Beskow. Look and tell: A dataset for multimodal grounding across
egocentric and exocentric views. InNeurIPS 2025 Workshop on Space in Vision, Language,
and Embodied AI, 2025.
[16] Michal Derezinski, Daniele Calandriello, and Michal Valko. Exact sampling of determinantal
point processes with sublinear time preprocessing. InNeurIPS, 2019.
[17] Mica R. Endsley. Toward a theory of situation awareness in dynamic systems.Human Factors:
The Journal of the Human Factors and Ergonomics Society, 1995.
11

[18] Bing Fan, Yunhe Feng, Yapeng Tian, James Chenhao Liang, Yuewei Lin, Yan Huang, and
Heng Fan. PRVQL: Progressive knowledge-guided refinement for robust egocentric visual
query localization. InICCV, 2025.
[19] Yue Fan, Xiaojian Ma, Rongpeng Su, Jun Guo, Rujie Wu, Xi Chen, and Qing Li. Embodied
VideoAgent: Persistent memory from egocentric videos and embodied sensors enables dynamic
scene understanding. InICCV, 2025.
[20] Yisen Feng, Haoyu Zhang, Meng Liu, Weili Guan, and Liqiang Nie. Object-shot enhanced
grounding network for egocentric video. InCVPR, 2025.
[21] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang,
Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehen-
sive evaluation benchmark of multi-modal llms in video analysis. InCVPR, 2025.
[22] Yuqian Fu, Runze Wang, Bin Ren, Guolei Sun, Biao Gong, Yanwei Fu, Danda Pani Paudel,
Xuanjing Huang, and Luc Van Gool. ObjectRelator: Enabling cross-view object relation
understanding across ego-centric and exo-centric perspectives. InICCV, 2025.
[23] Paolo Gabriel, Peter Rehani, Tyler Troy, Tiffany Wyatt, Michael Choma, and Narinder Singh.
Continuous patient monitoring with AI: Real-time analysis of video in hospital care settings.
Frontiers in Imaging, 2025.
[24] Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai
Krishna, Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim,
Mykel J. Kochenderfer, Ali-akbar Agha-mohammadi, and Shayegan Omidshafiei. Enter the
mind palace: Reasoning and planning for long-term active embodied question answering. In
CoRL, 2026.
[25] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit
Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4D: Around the
world in 3,000 hours of egocentric video. InCVPR, 2022.
[26] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Tri-
antafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al.
Ego-Exo4D: Understanding skilled human activity from first-and third-person perspectives. In
CVPR, 2024.
[27] Yuping He, Yifei Huang, Guo Chen, Lidong Lu, Baoqi Pei, Jilan Xu, Tong Lu, and Yoichi
Sato. Bridging perspectives: A survey on cross-view collaborative intelligence with egocentric-
exocentric vision.International Journal on Computer Vision, 2026.
[28] Yuping He, Yifei Huang, Guo Chen, Baoqi Pei, Jilan Xu, Tong Lu, and Jiangmiao Pang.
EgoExoBench: A benchmark for first-and third-person view video understanding in MLLMs.
InNeurIPS, 2025.
[29] Weipeng Hu, Jiun Tian Hoe, Jianhui Li, Haifeng Hu, Xudong Jiang, and Yap-Peng Tan.
Cascaded dynamic memory refinement and semantic alignment for exo-to-ego cross-view
video generation.IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
[30] Yijun Hu, Bing Fan, Xin Gu, Haiqing Ren, Dongfang Liu, Heng Fan, and Libo Zhang. Robust
ego-exo correspondence with long-term memory. InNeurIPS, 2025.
[31] Sihong Huang, Jiaxin Wu, Xiaoyong Wei, Yi Cai, Dongmei Jiang, and Yaowei Wang. Sound
bridge: Associating egocentric and exocentric videos via audio cues. InCVPR, 2025.
[32] Yifei Huang, Guo Chen, Jilan Xu, Mingfang Zhang, Lijin Yang, Baoqi Pei, Hongjie Zhang,
Lu Dong, Yali Wang, Limin Wang, and Qiao Yu. EgoExoLearn: A dataset for bridging
asynchronous ego-and exo-centric view of procedural activities in real world. InCVPR, 2024.
[33] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. VideoRAG: Retrieval-
augmented generation over video corpus. InACL (Findings), 2025.
12

[34] Baoxiong Jia, Yixin Chen, Siyuan Huang, Yixin Zhu, and Song-Chun Zhu. LEMMA: A
multi-view dataset for learning multi-agent multi-task activities. InECCV, 2020.
[35] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. EgoTaskQA: Understanding
human tasks in egocentric videos. InNeurIPS, 2022.
[36] Wenqi Jia, Miao Liu, Hao Jiang, Ishwarya Ananthabhotla, James M. Rehg, Vamsi Krishna
Ithapu, and Ruohan Gao. The audio-visual conversational graph: From an egocentric-
exocentric perspective. InCVPR, 2024.
[37] Hanwen Jiang, Santhosh Kumar Ramakrishnan, and Kristen Grauman. Single-stage visual
query localization in egocentric videos. InNeurIPS, 2023.
[38] Ji Hyeok Jung, Eun Tae Kim, Seoyeon Kim, Joo Ho Lee, Bumsoo Kim, and Buru Chang. Is
‘right’ right? Enhancing object orientation understanding in multimodal large language models
through egocentric instruction tuning. InCVPR, 2025.
[39] Minjoon Jung, Junbin Xiao, Junghyun Kim, Byoung-Tak Zhang, and Angela Yao.
EgoExo-Con: Exploring view-invariant video temporal understanding.arXiv preprint
arXiv:2510.26113, 2025.
[40] Soo-Han Kang and Ji-Hyeong Han. Video captioning based on both egocentric and exocentric
views of robot vision for human-robot interaction.International Journal of Social Robotics,
2023.
[41] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InEMNLP, 2020.
[42] Kangsan Kim, Yanlai Yang, Suji Kim, Woongyeong Yeo, Youngwan Lee, Mengye Ren, and
Sung Ju Hwang. MA-EgoQA: Question answering over egocentric videos from multiple
embodied agents.arXiv preprint arXiv:2603.09827, 2026.
[43] Alex Kulesza and Ben Taskar. k-DPPs: Fixed-size determinantal point processes. InICML,
2011.
[44] Yogesh Kulkarni and Pooyan Fazli. EgoVITA: Learning to plan and verify for egocentric
video reasoning.arXiv preprint arXiv:2511.18242, 2025.
[45] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang,
Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. LLaV A-OneVision: Easy visual task transfer.
arXiv preprint arXiv:2408.03326, 2024.
[46] Chentao Li, Zirui Gao, Mingze Gao, Yinglian Ren, Jianjiang Feng, and Jie Zhou. Do MLLMs
understand pointing? Benchmarking and enhancing referential reasoning in egocentric vision.
InACL, 2026.
[47] Chuhan Li, Ruilin Han, Joy Hsu, Yongyuan Liang, Rajiv Dhawan, Jiajun Wu, Ming-Hsuan
Yang, and Xin Eric Wang. Learning situated awareness in the real world. InICML, 2026.
[48] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan
Li. LLaV A-NeXT-Interleave: Tackling multi-image, video, and 3D in large multimodal models.
arXiv preprint arXiv:2407.07895, 2024.
[49] Shenshen Li, Xing Xu, Fumin Shen, Zhe Sun, Andrzej Cichocki, and Heng Tao Shen. Col-
laborated with hallucination: Enhancing egocentric grounded question answering via error
demonstrations.IEEE Transactions on Image Processing, 2026.
[50] Xiang Li, Heqian Qiu, Lanxiao Wang, Benliu Qiu, Fanman Meng, Linfeng Xu, and Hongliang
Li. SA V A-X: Ego-to-exo imitation error detection via scene-adaptive view alignment and
bidirectional cross view fusion. InCVPR, 2026.
[51] Yanjun Li, Yuqian Fu, Tianwen Qian, Qi’Ao Xu, Silong Dai, Danda Pani Paudel, Luc Van Gool,
and Xiaoling Wang. EgoCross: Benchmarking multimodal large language models for cross-
domain egocentric video question answering. InAAAI, 2026.
13

[52] Shuo Liang, Yiwu Zhong, Zi-Yuan Hu, Yeyao Tao, and Liwei Wang. Fine-grained spatiotem-
poral grounding on egocentric videos. InICCV, 2025.
[53] Ruiping Liu, Jiaming Zhang, Angela Schön, Karin Müller, Junwei Zheng, Kailun Yang,
Anhong Guo, Kathrin Gerling, and Rainer Stiefelhagen. ObjectFinder: An open-vocabulary
assistive system for interactive object search by blind people.arXiv preprint arXiv:2412.03118,
2024.
[54] Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem. BOLT: Boost large vision-
language model without training for long-form video understanding. InCVPR, 2025.
[55] Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang, Guanbin Li, Wen Gao, and Liang
Lin. Aligning cyber space with physical world: A comprehensive survey on embodied AI.
IEEE/ASME Transactions on Mechatronics, 2025.
[56] Yihao Liu, Xu Cao, Tingting Chen, Yankai Jiang, Junjie You, Minghua Wu, Xiaosong Wang,
Mengling Feng, Yaochu Jin, and Jintai Chen. From screens to scenes: A survey of embodied
ai in healthcare.Information Fusion, 119:103033, 2025.
[57] Tao Lu, Qian Zhu, Tiffany Ma, Wong Kam-Kwai, Anlan Xie, Alex Endert, and Yalong Yang.
Ego vs. exo and active vs. passive: Investigating the individual and combined effects of
viewpoint and navigation on spatial immersion and understanding in immersive storytelling.
InCHI, 2025.
[58] Hao Luo, Zihao Yue, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Deheng Ye, and Zongqing
Lu. OpenMMEgo: Enhancing egocentric understanding for LMMs with open weights and
data. InNeurIPS, 2025.
[59] Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, and Dacheng Tao. Grounded affordance
from exocentric view.International Journal of Computer Vision, 2024.
[60] Mi Luo, Zihui Xue, Alex Dimakis, and Kristen Grauman. Put myself in your shoes: Lifting
the egocentric perspective from exocentric videos. InECCV, 2024.
[61] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei
Chao, Jiebo Luo, and Rongrong Ji. Video-RAG: Visually-aligned retrieval-augmented long
video comprehension. InNeurIPS, 2025.
[62] Mohammad Mahdi, Yuqian Fu, Nedko Savov, Jiancheng Pan, Danda Pani Paudel, and Luc
Van Gool. Exo2EgoSyn: Unlocking foundation video generation models for exocentric-to-
egocentric video synthesis.arXiv preprint arXiv:2511.20186, 2025.
[63] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael
Henaff, Sneha Silwal, Paul McVay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav,
Qiyang Li, Ben Newman, Mohit Sharma, Vincent-Pierre Berges, Shiqi Zhang, Pulkit Agrawal,
Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Alexan-
der Sax, and Aravind Rajeswaran. OpenEQA: Embodied question answering in the era of
foundation models. InCVPR, 2024.
[64] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic
benchmark for very long-form video language understanding. InNeurIPS, 2023.
[65] Lorenzo Mur-Labadia, Maria Santos-Villafranca, Jesus Bermudez-Cameo, Alejandro Perez-
Yus, Ruben Martinez-Cantin, and Jose J. Guerrero. O-MaMa: Learning object mask matching
between egocentric and exocentric views. InICCV, 2025.
[66] Georgia Nigro and Ulric Neisser. Point of view in personal memories.Cognitive Psychology,
1983.
[67] Takehiko Ohkawa, Takuma Yagi, Taichi Nishimura, Ryosuke Furuta, Atsushi Hashimoto,
Yoshitaka Ushiku, and Yoichi Sato. Exo2EgoDVC: Dense video captioning of egocentric
procedural activities using web instructional videos. InWACV, 2025.
14

[68] OpenAI. Hello GPT-4o. https://openai.com/index/hello-gpt-4o , May 2024. Ac-
cessed: 2026-05-05.
[69] OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4 ,
March 2026. Accessed: 2026-05-05.
[70] Jiancheng Pan, Runze Wang, Tianwen Qian, Mohammad Mahdi, Yanwei Fu, Xiangyang Xue,
Xiaomeng Huang, Luc Van Gool, Danda Pani Paudel, and Yuqian Fu. V2-SAM: Marrying
SAM2 with multi-prompt experts for cross-view object correspondence. InCVPR, 2026.
[71] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Bootstrap your own views: Masked ego-exo
modeling for fine-grained view-invariant video representations. InCVPR, 2025.
[72] Junho Park, Andrew Sangwoo Ye, and Taein Kwon. EgoWorld: Translating exocentric view
to egocentric view using rich exocentric observations. InICLR, 2026.
[73] Baoqi Pei, Yifei Huang, Jilan Xu, Yuping He, Guo Chen, Fei Wu, Jiangmiao Pang, and
Yu Qiao. EgoThinker: Unveiling egocentric reasoning with spatio-temporal CoT. InNeurIPS,
2025.
[74] Taiying Peng, Jiacheng Hua, Miao Liu, and Feng Lu. In the eye of MLLM: Benchmarking
egocentric video intent understanding with gaze-guided prompting. InNeurIPS, 2025.
[75] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini
Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and
Ilya Sutskever. Learning transferable visual models from natural language supervision. In
ICML, 2021.
[76] Francesco Ragusa, Michele Mazzamuto, Rosario Forte, Irene D’Ambra, James Fort, Jakob En-
gel, Antonino Furnari, and Giovanni Maria Farinella. Ego-EXTRA: video-language egocentric
dataset for expert-trainee assistance. InWACV, 2026.
[77] Sahithya Ravi, Gabriel Herbert Sarch, Vibhav Vineet, Andrew D. Wilson, and Balasara-
vanan Thoravi Kumaravel. Out of sight, not out of context? Egocentric spatial reasoning in
vlms across disjoint frames. InEMNLP, 2025.
[78] Dominick Reilly, Manish Kumar Govind, Le Xue, and Srijan Das. From my view to yours:
Ego-to-exo transfer in vlms for understanding activities of daily living.arXiv preprint
arXiv:2501.05711, 2025.
[79] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and
beyond.Information Retrieval, 2009.
[80] Ivan Rodin, Tz-Ying Wu, Kyle Min, Sharath Nittur Sridhar, Antonino Furnari, Subarna Tripathi,
and Giovanni Maria Farinella. EASG-Bench: Video Q&A benchmark with egocentric action
scene graphs. InICCVW, 2025.
[81] David Schneider, Zdravko Marinov, Rafael Baur, Zeyun Zhong, Rodi Düger, and Rainer
Stiefelhagen. OmniFall: A unified staged-to-wild benchmark for human fall detection.arXiv
preprint arXiv:2505.19889, 2025.
[82] Zhaofeng Shi, Heqian Qiu, Lanxiao Wang, Qingbo Wu, Fanman Meng, and Hongliang Li.
Unsupervised ego-and exo-centric dense procedural activity captioning via gaze consensus
adaptation. InAAAI, 2025.
[83] Zhaofeng Shi, Heqian Qiu, Lanxiao Wang, Qingbo Wu, Fanman Meng, Lili Pan, and Hongliang
Li. Test-time ego-exo-centric adaptation for action anticipation via multi-label prototype
growing and dual-clue consistency. InCVPR, 2026.
[84] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens
Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. DriveLM: Driving with graph
visual question answering. InECCV, 2024.
15

[85] Yuhao Su and Ehsan Elhamifar. RegionAligner: Bridging ego-exo views for object correspon-
dence via unified text-visual learning. InWACV, 2026.
[86] Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli,
and Dongxiao Zhu. WalkGPT: Grounded vision-language conversation with depth-aware
segmentation for pedestrian navigation. InCVPR, 2026.
[87] Pengzhan Sun, Junbin Xiao, Tze Ho Elden Tse, Yicong Li, Arjun Akula, and Angela Yao.
Visual intention grounding for egocentric assistants. InICCV, 2025.
[88] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe
sampling for long video understanding. InCVPR, 2025.
[89] Haochen Wang, Qirui Chen, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, Weidi Xie, and
Stratis Gavves. Object-centric video question answering with visual grounding and referring.
InICCV, 2025.
[90] Runmin Wang, Xingdong Song, Zukun Wan, Han Xu, Congzhen Yu, Tianming Ma, Yajun
Ding, and Shengyou Qian. Dual-space intervention for mitigating bias in robust visual question
answering.Expert Systems with Applications, 2026.
[91] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang
Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source
multimodal models in versatility, reasoning, and efficiency.arXiv preprint arXiv:2508.18265,
2025.
[92] Yifei Wang, Zhenkai Li, Tianwen Qian, Huanran Zheng, Zheng Wang, Yuqian Fu, and Xiaoling
Wang. StreamEQA: Towards streaming video understanding for embodied scenarios.arXiv
preprint arXiv:2512.04451, 2025.
[93] Zeqing Wang, Wentao Wan, Qiqing Lao, Runmeng Chen, Minjie Lang, Xiao Wang, Feng
Gao, Keze Wang, and Liang Lin. Towards top-down reasoning: An explainable multi-agent
approach for visual question answering.IEEE Transactions on Multimedia, 2026.
[94] Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang,
and Angela Yao. EgoBlind: Towards egocentric visual assistance for the blind. InNeurIPS,
2025.
[95] Junbin Xiao, Shenglang Zhang, Pengxiang Zhu, and Angela Yao. Ego-grounding for personal-
ized question-answering in egocentric videos. InCVPR, 2026.
[96] Jilan Xu, Yifei Huang, Junlin Hou, Guo Chen, Yuejie Zhang, Rui Feng, and Weidi Xie.
Retrieval-augmented egocentric video captioning. InCVPR, 2024.
[97] Jilan Xu, Yifei Huang, Baoqi Pei, Junlin Hou, Qingqiu Li, Guo Chen, Yuejie Zhang, Rui Feng,
and Weidi Xie. EgoExo-Gen: Ego-centric video prediction by watching exo-centric videos. In
ICLR, 2025.
[98] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai Li, and Dacheng
Tao. AdaVideoRAG: Omni-contextual adaptive retrieval-augmented efficient long video
understanding. InNeurIPS, 2025.
[99] Karmesh Yadav, Yusuf Ali, Gunshi Gupta, Yarin Gal, and Zsolt Kira. FindingDory: A
benchmark to evaluate memory in embodied agents.arXiv preprint arXiv:2506.15635, 2025.
[100] Chenhongyi Yang, Anastasia Tkach, Shreyas Hampali, Linguang Zhang, Elliot J. Crowley,
and Cem Keskin. EgoPoseFormer: A simple baseline for stereo egocentric 3D human pose
estimation. InECCV, 2024.
[101] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking
in space: How multimodal large language models see, remember, and recall spaces. InCVPR,
2025.
16

[102] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang,
Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco
Cominelli, Zhongang Cai, Bo Li, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg
Widmer, Francesco Gringoli, Lei Yang, and Ziwei Liu. EgoLife: Towards egocentric life
assistant. InCVPR, 2025.
[103] Hanrong Ye, Haotian Zhang, Erik Daxberger, Lin Chen, Zongyu Lin, Yanghao Li, Bowen
Zhang, Haoxuan You, Dan Xu, Zhe Gan, Jiasen Lu, and Yinfei Yang. MM-Ego: Towards
building egocentric multimodal LLMs for video QA. InICLR, 2025.
[104] Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. WorldMM: Dynamic
multimodal memory agent for long video reasoning. InCVPR, 2026.
[105] Zhiqiang Yuan, Ting Zhang, Yeshuang Zhu, Jiapei Zhang, Ying Deng, Zexi Jia, Peixiang
Luo, Xiaoyue Duan, Jie Zhou, and Jinchao Zhang. WalkVLM: Aid visually impaired people
walking by vision language model. InICCV, 2025.
[106] Farnaz Zamiri Zeraati, Yang Cao, Yuehan Qiao, Hal Daumé III, and Hernisa Kacorri. Say it
my way: Exploring control in conversational visual question answering with blind users. In
CHI, 2026.
[107] Deheng Zhang, Yuqian Fu, Runyi Yang, Yang Miao, Tianwen Qian, Xu Zheng, Guolei Sun,
Ajad Chhatkuli, Xuanjing Huang, Yu-Gang Jiang, Luc Van Gool, and Danda Pani Paudel.
EgoNight: Towards egocentric vision understanding at night with a challenging benchmark.
InICLR, 2026.
[108] Haoyu Zhang, Qiaohui Chu, Meng Liu, Haoxiang Shi, Yaowei Wang, and Liqiang Nie.
Exo2Ego: Exocentric knowledge guided MLLM for egocentric video understanding. InAAAI,
2026.
[109] Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng Wang, Tat-Seng
Chua, and Angela Yao. EgoTextVQA: Towards egocentric scene-text aware video question
answering. InCVPR, 2025.
[110] Sheng Zhou, Junbin Xiao, Xun Yang, Peipei Song, Dan Guo, Angela Yao, Meng Wang,
and Tat-Seng Chua. Scene-text grounding for text-based video question answering.IEEE
Transactions on Multimedia, 2025.
[111] Siyuan Zhou, Yilun Du, Yuncong Yang, Lei Han, Peihao Chen, Dit-Yan Yeung, and Chuang
Gan. Learning 3D persistent embodied world models. InNeurIPS, 2025.
17

A Social Impact and Limitations
Memory is essential for reasoning with MLLMs as a form of auxiliary cognition. However, existing
memory-based approaches rely solely on the egocentric stream of mobile agents and their interactions
with the environment. This limits their ability to capture full-body movements of other agents and
track their interactions with the environment, both of which are critical to the holistic understanding
of the scene. With the advancement of surveillance systems, wearable devices for human agents, and
visual sensors for embodied agents, memory can increasingly be constructed from cross-view sources
to enable more comprehensive retrieval,e.g., in homes [ 34], hospitals [ 56], and parking facilities [ 27].
As the first benchmark for ego-exo memory, EgoExoMem has several limitations. First, the minute-
level video duration limits the scope of our benchmark, where simple retrieval methods may already
suffice. Since real-world observations can span weeks or months, datasets supporting long-term
ego-exo memory are necessary, and structured retrieval methods could prove more beneficial in
such settings. Second, to standardize the task, we currently restrict input to one egocentric and one
exocentric stream. However, the underlying datasets contain multiple egocentric and exocentric views,
which means that the same MCQs could be repurposed for multi-view settings. Future work could
explore this direction and investigate how multiple egocentric or exocentric streams individually
contribute to benchmark performance.
B Human Editing and Filtering
Fig. 6 shows the verification user interface.
Figure 6: Verification tool for human annotator editing and filtering.
C Evaluation Prompts
The prompt used to generate captions for RAG-based methods with Gemini 2.5 Flash is shown in
Fig. 7. The evaluation template is provided in Fig. 8.
D Further Ablation Studies
Following standard practice [ 21], we uniformly sample 32frames as input. For RAG-based methods,
we retrieve the top- kclips ( k= 8 ) and sample 4frames per clip. An ablation study on the effect of k
is provided in Tab. 5, which demonstrates that performance is largely insensitive to the choice of k.
18

system_prompt =
“You are a video captioning assistant. Given a few sampled frames
from a short video clip, write ONE concise sentence describing the
main activity or scene. Focus on actions and objects visible.”
Figure 7: Caption generation used for retrieval in RAG-based methods.
system_prompt =
“You are answering multiple-choice questions about a video. Selected
keyframes from egocentric and exocentric views are provided, ordered
by timestamp. Each frame is labeled with its view [Ego/Exo]
and time. Answer with a single letter (A, B, C, or D) only. No
explanation.”
user_prompt =
“Question: {question}
Options:
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}
Answer with the option letter only (A, B, C, or D).”
Figure 8: Evaluation template.
Table 5: Ablation study on the effect of the top-kvalue in VideoRAG [33].
Top-k Views HL IP RL ED OS AR TPA TO Avg
k= 5Ego 46.9 53.9 59.6 37.1 50.6 45.0 47.3 38.0 47.3
Exo 49.0 66.6 55.1 31.2 48.8 40.8 39.8 35.0 45.8
Ego+Exo 53.5 65.8 60.9 33.2 54.3 46.7 41.9 39.9 49.5
k= 8Ego 47.2 53.6 62.8 36.6 53.0 40.8 48.0 40.3 47.8
Exo 47.2 69.0 60.3 29.6 46.3 37.9 41.4 36.0 46.0
Ego+Exo 49.0 66.6 62.2 33.5 57.9 43.8 41.0 39.6 49.2
19