# Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios

**Authors**: Peizheng Yan, Yu Zhao, Liang Xie, Juntong Qi, Mingming Wang, Erwei Yin

**Published**: 2026-05-07 13:01:28

**PDF URL**: [https://arxiv.org/pdf/2605.06185v1](https://arxiv.org/pdf/2605.06185v1)

## Abstract
Recent large vision-language models have achieved strong performance on short- and medium-length video understanding, yet they remain inadequate for ultra-long or even infinite video reasoning, where models must preserve coherent memory over extended durations and infer causal dependencies across temporally distant events. Existing end-to-end video understanding methods are fundamentally limited by the $O(n^2)$ complexity of self-attention, while recent retrieval-augmented generation (RAG) approaches still suffer from fragmented clip-level memory, weak modeling of temporal and causal structure, and high storage and online inference costs. We present Event-Causal RAG, a lightweight retrieval-augmented framework for infinite long-video reasoning. Instead of indexing fixed-length clips, our method segments streaming videos into semantically coherent events and represents each event as a structured State-Event-State (SES) graph, capturing the event together with its surrounding state transitions. These graphs are merged into a global Event Knowledge Graph and stored in a dual-store memory that supports both semantic matching and causal-topological retrieval. On top of this memory, we design a bidirectional retrieval strategy to efficiently identify the most relevant event causal chains and provide them, together with the associated video evidence, to a backbone video foundation model for answer generation. Experiments on long-video understanding benchmarks demonstrate that Event-Causal RAG consistently outperforms strong clip-based retrieval baselines and long-context video models, particularly on questions requiring multi-event integration and causal inference across long temporal gaps, while also achieving improved memory efficiency and robust streaming performance.

## Full Text


<!-- PDF content starts -->

Event-Causal RAG: A Retrieval-Augmented
Generation Framework for Long Video Reasoning in
Complex Scenarios
Peizheng Yan
Tianjin Key Lab of Intelligent Unmanned
Swarm Tech & System Tianjin University
Tianjin, China
ypzyh@tju.edu.cnYu Zhao∗
Institute of Computing and Intelligence
Harbin Institute of Technology (Shenzhen)
Shenzhen, China
zhaoyu9067@live.cn
Liang Xie†
Tianjin Artificial Intelligence Innovation Center
Tianjin, China
Defense Innovation Institute
Academy of Military Sciences
Beijing, ChinaJuntong Qi
School of Future Technology
Shanghai University, Shanghai, China
qijt@shu.edu.cn
Mingming Wang
Tianjin Key Lab of Intelligent Unmanned
Swarm Tech& System Tianjin University
Tianjin, China
wangmm19@tju.edu.cnErwei Yin
Tianjin Artificial Intelligence Innovation Center
Tianjin, China
Defense Innovation Institute
Academy of Military Sciences
Beijing, China
Abstract
Recent large vision-language models have achieved strong performance on short-
and medium-length video understanding, yet they remain inadequate for ultra-long
or even infinite video reasoning, where models must preserve coherent memory over
extended durations and infer causal dependencies across temporally distant events.
Existing end-to-end video understanding methods are fundamentally limited by the
O(n2)complexity of self-attention, while recent retrieval-augmented generation
(RAG) approaches still suffer from fragmented clip-level memory, weak modeling
of temporal and causal structure, and high storage and online inference costs.
We present Event-Causal RAG, a lightweight retrieval-augmented framework for
infinite long-video reasoning. Instead of indexing fixed-length clips, our method
segments streaming videos into semantically coherent events and represents each
event as a structured State-Event-State (SES) graph, capturing the event together
with its surrounding state transitions. These graphs are merged into a global Event
Knowledge Graph and stored in a dual-store memory that supports both semantic
matching and causal-topological retrieval. On top of this memory, we design
a bidirectional retrieval strategy to efficiently identify the most relevant event
causal chains and provide them, together with the associated video evidence, to
a backbone video foundation model for answer generation. Experiments on long-
video understanding benchmarks demonstrate that Event-Causal RAG consistently
outperforms strong clip-based retrieval baselines and long-context video models,
∗Corresponding authors.
†Corresponding authors.
Preprint.arXiv:2605.06185v1  [cs.AI]  7 May 2026

particularly on questions requiring multi-event integration and causal inference
across long temporal gaps, while also achieving improved memory efficiency and
robust streaming performance.
1 Introduction
Recent advances in large vision-language models (LVLMs) have substantially improved the perfor-
mance of video understanding systems [ 1,2,3,4,5]. In particular, mainstream video foundation
models have become increasingly capable of recognizing actions, summarizing content, and an-
swering questions over short to medium-length videos [ 1,2,6,3]. These successes, however, do
not directly translate to ultra-long or even infinite video reasoning. Unlike short-form video under-
standing, long-horizon reasoning requires a model to track evolving storylines, maintain coherent
memory over extended time spans, and infer causal dependencies among events that may be separated
by minutes or even hours [ 7,8,9,10,11]. Such requirements are central to real-world scenarios
including surveillance [ 12,13], medical monitoring, and traffic analysis [ 14,15], where decisions
often depend not only onwhathappened, but also onwhyit happened and how it unfolded over time.
Current end-to-end VLM-based video understanding solutions face a severe computational bottleneck.
The dominant Transformer architecture relies on self-attention, whose time and memory complexity
increase rapidly with the number of visual tokens [ 16]. As video duration grows, the tokenized
representation quickly becomes prohibitively large, leading to excessive latency, memory overflow,
and unstable inference. Although recent efforts have explored temporal pooling, token compression,
sparse attention, and context window extension to alleviate this issue [ 17,18,19,20,21,22,23,24],
these techniques do not fundamentally resolve the challenge of reasoning over extremely long
sequences. In practice, models still suffer from the well-known lost-in-the-middle phenomenon [ 25],
where information located far from the beginning or end of the context is poorly utilized. As a result,
simply extending the context window does not yield reliable long-horizon reasoning; instead, it
often increases computational cost while degrading accuracy on causal and temporally compositional
queries [26, 27, 11].
To avoid processing an entire long video at once, recent works have adopted retrieval-augmented
generation (RAG) for long-video understanding [ 28,29,30,31,32,33]. Nevertheless, existing video-
RAG approaches still exhibit three critical limitations:(1)They oftenfragment spatiotemporal
contextby constructing memory over fixed-duration clips rather than semantically complete events.
In most existing pipelines, a long video is mechanically segmented into windows of equal length,
e.g., every 5 or 10 seconds, and each segment is independently embedded. While convenient, this
design ignores the natural boundaries of events. A physically and semantically coherent causal
chain may therefore be split across multiple clips [ 10,34,27].(2)Existing retrieval strategies
primarily rely onglobal semantic similaritybetween the query and individual clips, while largely
ignoring explicit modeling of temporal dependency and causal relations between events. This design
is sufficient when the goal is to find a visually similar scene or a directly matching clip, but it is
inadequate for questions that require multi-step reasoning across distant observations [ 35,7,36,37].
(3)Current RAG-based methodsremain costly in both storage and online processing. Because
they store dense representations for large numbers of low-level clips, the memory overhead grows
rapidly with video duration. This issue becomes more severe in streaming environments, where the
system must continuously process and retrieve new content as the video arrives [ 38,39,40,41,42].
Meanwhile, large-model inference is often too slow to keep pace with high-throughput streams,
forcing practitioners to rely on expensive parallel deployment for near-real-time performance.
In this paper, we propose Event-Causal RAG, a retrieval-augmented framework for INFINITE long-
video reasoning. Our key idea is to replace clip-level memory with a State-Event-State (SES) graph
memory that aligns retrieval units with semantically complete events rather than arbitrary temporal
windows. Specifically, we first perform asynchronous event-driven segmentation on streaming
videos using joint visual and audio cues, so that each memory unit preserves a naturally coherent
event whenever possible. Next, we parse each clip into structured event graphs, which capture the
core events with their pre- and post-event states. The event graphs are unified into a global Event
Knowledge Graph and stored in a dual-store memory (a Vector DB for node semantics and a Graph
DB for topology). Compared with conventional vector databases that mainly support similarity
retrieval, our memory explicitly retains higher-level relational information, allowing the system
to retrieve not only semantically relevant events but also causally connected context. Finally, we
2

use a bidirectional retrieval algorithm for RAG that can rapidly find the most relevant event causal
chains. The retrieved event graphs are serialized and provided, together with the corresponding event
clips, to the backbone foundation VLM, thereby enhancing its event-causal reasoning capability and
enabling the generation of detailed responses. Importantly, Event-Causal RAG is lightweight and can
be integrated with existing video foundation models without expensive end-to-end retraining over
ultra-long sequences.
We evaluate our approach on long-video understanding benchmarks, and experimental results show
that Event-Causal RAG consistently outperforms strong clip-based retrieval baselines and long-
context video models, especially on questions that require multi-event integration and causal in-
ference across extended temporal gaps. Moreover, our method achieves favorable efficiency by
reducing memory footprint while maintaining robust performance in streaming scenarios. In sum-
mary, EC-RAG addresses the three core challenges identified above:(a)It mitigates spatiotemporal
fragmentation by constructing event-level memory,(b)strengthens long-horizon reasoning through
explicit State-Event-State causal structure, and(c)reduces storage and online inference overhead
through compact dual-store retrieval. These properties make EC-RAG suitable for ultra-long and
streaming video scenarios where robust reasoning requires both efficient memory management and
temporally grounded causal evidence.
2 Related Work
Long Video Understanding via LVLMs.Recent large vision-language models (LVLMs) have
substantially advanced video understanding, enabling action recognition, temporal description,
and question answering over short and medium-length inputs [ 1,2,3,6]. To scale LVLMs to
longer videos, existing studies explore frame sampling, temporal pooling, visual token compression,
context-window extension, positional encoding adaptation, KV-cache compression, and streaming
inference [ 17,18,19,20,21,38,39,5]. These methods improve the computational feasibility
of processing long visual sequences. However, merely extending the accessible context does not
resolve the core challenge of long-video understanding, since the context window of LLMs is
ultimately bounded and cannot scale to ultra-long video inputs. Retrieval-augmented generation
(RAG) with external memory offer an alternative by decoupling video processing from question
answering [ 2,28,29,30]. Recent scene-level and graph-based RAG variants further show that
structured memory can improve multi-hop retrieval and long-range reasoning [ 31,43,44,45].
Nevertheless, most existing RAG pipelines still construct memory over fixed-duration clips, coarse
scenes, or generic semantic graphs. Such memory units can break semantically complete events,
while embedding-based retrieval mainly captures query similarity and lacks explicit modeling of
directional state changes and causal chains.
Temporal-causal Reasoning in Long Videos.Prior research on video understanding has exten-
sively studied temporal reasoning, including event ordering, temporal localization, and long-range
dependency modeling [ 35,10,7,9,8,27,11]. Early approaches mainly focus on short clips or tempo-
rally localized events, where temporal dynamics can be captured within limited contexts. To handle
longer videos, subsequent works explore hierarchical representations, memory-based architectures,
and retrieval-based frameworks to improve the modeling of extended temporal structure [ 7,9,8].
Meanwhile, another line of work investigates causal reasoning in videos, such as event causality
modeling, causal temporal grounding, and counterfactual analysis [ 10,27,46,47,11], aiming to
move beyond temporal correlation toward a deeper understanding of event relations. However,
these methods do not effectively capture long-range causal dependencies in extremely long videos.
Unlike prior approaches, EC-RAG represents streaming videos as a global Event Knowledge Graph.
This design explicitly preserves overall pre-event and post-event state transitions, and supports long
causal-topological retrieval, thereby facilitating ultra-long or streaming video reasoning.
3 Methodology
The overview of the EC-RAG framework is illustrated in Figure 1, consisting of three modules. First,
the Dual-Sentinel Event Segmentation module (Section 3.1) cuts the input videos into full-event
clips. Next, the SES Event Memory Constructor (Section 3.2) extracts the SES graphs and stores the
3

Bidirectional Retrieval RAG
 SES Memory Constructor
 Dual -Sentinel Event Segmentation
… …Video 
Stream
SigLIP𝒇𝒕𝒇𝒕+𝟏𝒇𝒕+𝟐𝒇𝒕+𝟑𝒇𝒕+𝟒
𝑺(𝒕)
𝝉𝒆𝒗𝒕𝝉𝒃𝒈
Event Anchor
Audio 
TrackASR
Speech Transactions…t-a t+b t
𝒕𝒂𝒔 𝒕𝒂𝒆Center -out 
Expansion from t
… …
Full-event Clip 
Full-event Clip VLMCoT
SES Triplets 
Events: [{
Temporal_order:1, Location: “mountain”, 
Description: “the boys climbing the mountain.”,
Entities：[“boy in white ”, “boy in blue jeans”,…]
PreState : “the boys laughing”, PostState : “the boys taking 
photos”}, …]
Embedding 
Model
GraphDB
… …
…>𝜸𝒆𝒏𝒕>𝜸𝒆𝒗𝒕VectorDB
Add Edge
Entity 
MergeState Event
Entity
Temporal EdgeUser Query VectorDBSTEP 1: Entry Anchoring 
Top K Anchors
STEP 2: Bidirectional Walk Selection……
N Hops
STEP 3: Semantic Refinement
Timeline Flattening Semantic Refinement
>𝝉𝒅𝒖𝒑
CoT Prompts：
Questions
[Rules]
[Memories]Large Visual -
Language ModelResponsem consecutive  S( i) > 𝝉𝒃𝒈
by timestampEvent Clip Figure 1:Overview of the Event-Causal RAG framework.
topology and node representations. With the event memory, the RAG module (Section 3.3) can query
event causal chains and enhance LVLM generation.
3.1 Dual-Sentinel Event Segmentation
Visual Sentinel.The foundational step of EC-RAG is determining where an event begins and ends.
Let a video stream be V={f 1, f2, ..., f T}, where fiis a visual frame. We first utilize a pre-trained
SigLIP [ 48] visual encoder Evto extract frame-level embeddings and then calculate the similarity
between adjacent frames:
S(t) = cos(E v(ft), Ev(ft+1)),(1)
where cos(·,·) denotes cosine similarity. We then define a trigger to anchor an event by detecting
fluctuations in S(t). When S(t) falls below a highly sensitive threshold ( τevt= 0.97 ), it signifies a
micro-action or state transition. To prevent over-segmentation from high-frequency noise, we add a
local minima constraint. Thus, an event is anchored attif:
S(t)< τ evt∧S(t)< S(t−1)∧S(t)< S(t+ 1).(2)
From these event anchors, we employ a “center-out” expansion until a stable background threshold
(τbg= 0.99) is exceeded to confirm an event windowT(t) ={f t−a, ..., f t, ..., f t+b}:
S(e)| t−a<e<t+b < τbg∧S(t−a−1)> τ bg∧S(t+b)> τ bg.(3)
To avoid overly fine-grained or excessively long clip segmentation, we also set lower and upper
bounds ( pl, pu) on the event window length: pl< a+b < p u. Finally, by applying the segmentation
algorithm, we obtain video chunks containing complete events.
Audio Sentinel.Simultaneously, to capture audio events that are not reflected in the visual signal,
we adopt an off-the-shelf ASR tool to extract video speech, projecting transcriptions (with strict
punctuation preservation) directly into the corresponding temporal segments. This cross-modal
projection preserves the emotional and tonal context crucial for downstream LLM reasoning. The
result is a strict, seamless temporal partition of Dynamic Events (sampled at high FPS) and Static
Backgrounds (sampled at low FPS). The detailed algorithm is available in Appendix 1.
3.2 State-Event-State Memory Constructor
Entity-First SES Triplets Extraction.We prompt a VLM to extract (State-Event-State) SES
triplets from each event clip T(t). Specifically, we use a two-step CoT guidance, where the VLM is
explicitly constrained to first output a scene_inventory , enumerating all visible interacting entities
(e.g., ["red-shirt man", "glass door", "military helicopter"] ). This step acts as a
4

spatial attention dispersal mechanism, forcing the VLM to track each listed entity independently.
Subsequently, the VLM generates structured SES triplets in the form of:
State pre[:PRECONDITION_OF]− − − − − − − − − − − − − − − − − →Event[:RESULTS_IN]− − − − − − − − − − − →State post (4)
Each event node contains the natural-language description of the event, the location, and the involved
entities, while each state node describes the pre- and post-event states. Additionally, an explicit
temporal_order attribute is assigned to each event triplet to enforce chronological logic and prevent
causal hallucination.
Dual-Store Memory System.We utilize a Dual-Store Memory system comprising a Vector DB
(e.g. Milvus) for high-dimensional semantic routing and a Graph DB (e.g. Neo4j) for topological
mapping. The extracted triplets from individual event clips must be continuously stored and unified
with previous ones. We employ an embedding model (e.g., Qwen3-Embedding-4B) to compute node
representations. First, for each newly arrived SES triplet, if its entity similarity with an existing
node exceeds γent= 0.85 , the two are merged into a single global node. We then enforceState
Continuityto construct causal chains: if the PostState of chunk Event iand the PreState of
chunk Event jexhibit a semantic similarity γevt= 0.85 , they are fused, and a [:TEMPORAL_NEXT]
edge is established, weaving parallel action chains into a dense, interconnected mesh.
3.3 Bidirectional Retrieval RAG
During the RAG QA phase, standard path-finding algorithms suffer from exponential traversal
explosion in dense graphs. We mitigate this via a three-step dimension-reduction algorithm:
Step 1: Entry Anchoring.We query the vector database to retrieve the most semantically relevant
event nodes as entry points.
Step 2: Bidirectional Walk Selection.Within the graph database, we abandon directed path
enumeration. Instead, we collect all distinct nodes within Nhops (typically N= 2 ) of the anchors,
capturing both causes and effects without path duplication.
Step 3: Semantic Refinement.To present the LLM with a clean context, all retrieved nodes are
flattened chronologically based on physical timestamps. To eliminate the “Context Redundancy”
caused by VLMs repeatedly extracting identical continuous actions across adjacent chunks, we
introduce an embedding-based semantic deduplication mechanism. Let nidenote a newly retrieved
candidate node, and let ni.Text represent its extracted textual description. We utilize the embedding
model Ψeto project this text into a semantic vector Ψe(ni.Text) . We then compute its maximum
cosine similarity, Smax(ni), against Mseen, which is the set of embeddings for all previously
preserved nodes:
Smax(ni) = max
v∈M seen{cos(Ψ e(ni.Text), v)},(5)
where cos(·,·) denotes the cosine similarity function. If Smax(ni)> τ dup, where τdupis a predefined
deduplication threshold, the candidate node is considered redundant and is pruned. Otherwise, its
semantic vector Ψe(ni.Text) is appended to Mseen. This semantic state collapse automatically
compresses prolonged, unchanging scenes into a single script entry, reducing the context volume by
>50% and drastically increasing information density. Finally, the LLM executes Chain-of-Thought
(CoT) reasoning based on this purified prompt. The relevant event clips are also provided as visual
inputs.
4 Experiments and Analysis
4.1 Settings
Multi-scale Video Benchmarks.To comprehensively verify the system’s robustness across dif-
ferent temporal spans, we constructed a hierarchical evaluation framework ranging from seconds
to a full day: 1)NExT-QA(short videos regarding the causal and temporal relationships between
actions, 5-180s); 2)EventBench(medium-to-long videos, multi-level event understanding and
causal reasoning, 60-1800s); 3)Video-MME Long(extremely long videos, including two chapters:
event action understanding and action reasoning, 1800-3600s); 4)24-hour+ streaming surveillance
video(the video uses real-world surveillance footage with a time span of over a day to verify the
framework’s ability to process unlimited streams of videos).
5

Table 1:Video-MME Long (1800s-3600s) subset testing.Under extreme length challenges, EC-RAG achieves
substantial improvements (+12.50%) in capturing dynamic features like action recognition through semantic
dimensionality reduction and graph abstraction.
TaskQwen3-VL-8B(Baseline)EC-RAG (Ours)∆(Improvement)
Action Reasoning 43.8946.67+ 2.78
Action Recognition 35.0047.50 + 12.50
Table 2:Quantitative evaluation on EventBench and NExT-QA.EC-RAG improves all open-source
backbones on NExT-QA and overall EventBench accuracy, with the main gains coming from contextual and
episodic reasoning Across overall metrics, EC-RAG enables all three open-source backbones to match or surpass
the GPT-4o overall score on EventBench.
NExT-QA (5–180s) EventBench (60–1800s)
MethodAccuracy (%) Contextual Episodic Counter-intuitive Overall
GPT-4o (Closed-source) 76.71 50.13 37.33 63.44 53.33
Gemini1.5-pro (Closed-source) 75.74 32.15 38.67 52.86 43.24
Qwen3-VL-8B 73.08 53.16 45.67 59.91 52.39
+EC-RAG 75.54 (+2.46) 61.77 (+8.61) 48.33 (+2.66)53.3055.31 (+2.92)
VideoLlama3-7B 76.69 56.71 42.00 55.95 51.74
+EC-RAG 78.48 (+1.79) 61.01 (+4.30) 49.67 (+7.67)55.5155.97 (+4.23)
InternVL3.5-8B 77.54 56.96 45.00 60.35 53.90
+EC-RAG 80.54 (+3.00) 62.28 (+5.32) 46.67 (+1.67)57.2755.97 (+2.07)
Implementation Details.EC-RAG utilizes SigLIP [ 48] for visual encoding, adaptively sampling
dynamic events at 8 FPS and static backgrounds at 2 FPS to optimize computational overhead. The
Dual-Modal Sentinel strictly bounds streaming chunks to 12 seconds to prevent context overflow.
Within the dual-store memory, nodes are embedded via Qwen3-Embedding-4B into ChromaDB for
semantic routing, while causal topologies are preserved in Neo4j. Bidirectional traversal is constrained
to a maximum depth of N= 2 , utilizing a semantic deduplication threshold of τdup= 0.85 . All
evaluations, including the 24-hour infinite stream test, are executed entirely on a single NVIDIA RTX
5090 (32GB VRAM).
4.2 Verification of causal reasoning ability under ultra-long videos (1800s-3600s)
In the extremely long video subset of Video-MME (Long), which lasts up to 1 hour, native large
models face severe context window constraints. As shown in Table 1, EC-RAG demonstrates signifi-
cant advantages. In the Action Reasoning task, EC-RAG achieves a score of 46.67%, representing a
+2.78% improvement over the native end-to-end no-RAG baseline (43.89%).
Notably, in the Action Recognition task, EC-RAG increases the accuracy from 35.00% to47.50%,
achieving a substantial growth of+12.50%. This indicates that when the video length exceeds 1800s,
native models are prone to missing key dynamics due to attention dispersion. In contrast, EC-RAG,
based on the dual-sentinel diffusion chunking and EKG state transition abstraction, effectively local-
izes the causal anchors of actions, thereby significantly enhancing dynamic perception capabilities in
extremely long sequences.
4.3 Causal reasoning ability within the framework of short and medium length videos
Robustness in Short-Horizon Micro-Dynamics.As shown in Table 2, on the NExT-QA benchmark
(5–180s), EC-RAG consistently improves accuracy across all evaluated open-source backbones,
with gains ranging from +1.79% to +3.00%. These consistent gains indicate that the proposed
State-Event-State (SES) abstraction does not only benefit long videos, but also strengthens short-
horizon micro-dynamic reasoning by preserving fine-grained spatiotemporal evidence while filtering
visually redundant context. This is important because short-video QA still often requires identifying
subtle action transitions and local temporal dependencies, which can be weakened by naive text-only
filtering or coarse clip summarization.
Efficacy in Long-Horizon Causal Reasoning.The advantage of EC-RAG becomes more evident on
EventBench (60–1800s), where questions require integrating temporally distributed evidence. In the
6

Contextualreasoning split, EC-RAG improves all three open-source models, with gains ranging from
+4.30% to +8.61%. These results show that dual-store state merging and bidirectional graph retrieval
help recover contextual evidence that is distributed across multiple events rather than concentrated in a
single clip. A similar trend appears in theEpisodicsplit, where EC-RAG improves all three backbones
with gains ranging from +1.67% to +7.67%. The particularly large episodic gain for VideoLlama3-7B
suggests that event-level topological memory can mitigate feature amnesia and logical fragmentation
when the backbone model alone struggles to preserve long-range event continuity.
Performance Variance in Subjective Queries.While EC-RAG consistently improves objective and
temporally grounded reasoning, it exhibits a different trend on theCounter-intuitivesubset. This
subset contains more subjective, aesthetic, or abstract questions (e.g., “What makes this video so
magical?”), for which the answer often depends on implicit visual impression rather than explicit
physical state transitions. Because EC-RAG converts videos into fact-grounded SES structures, it
prioritizes observable entities, actions, and state changes, but may suppress ambiguous artistic cues
or high-level affective semantics that are useful for such subjective interpretation. This explains why
EC-RAG can slightly reduce performance on theCounter-intuitivesplit while still improving the
overall reasoning capability.
Importantly, the overall results remain strongly aligned with the intended design of EC-RAG. On
EventBench, EC-RAG improves overall accuracy across all three backbones, with gains ranging
from +2.07% to +4.23%. Thus, all three parameter-efficient open-source models match or surpass
the GPT-4o overall score of 53.33%, while relying on compact event-causal memory rather than
full-context video ingestion or proprietary model scale. These results support the central claim of this
paper: structured event-level memory and causal-topological retrieval provide a practical mechanism
for improving long-video reasoning, especially when answers require multi-event integration and
temporally grounded causal evidence.
4.4 Ablation Study
To verify the systematic necessity of each core component of EC-RAG, we conducted an in-depth
module ablation on EventBench (see Table 3). To explicitly eliminate the performance variance
introduced by raw video frame parsing and strictly isolate the structural effectiveness of the framework
itself, this ablation study adopts a "pure RAG with text-based reasoning" paradigm. Under this
controlled setting, the complete EC-RAG architecture utilizing Qwen3-VL-8B establishes a strong
baseline accuracy of49.37%.
The experimental data reveals a clear logic: lacking the elastic sentinel with context redundancy
wrapper (degrading to a fixed chunking with mechanical truncation) severs the integrity of underlying
actions, leading to a 2.53% decline. Removing the local SES abstraction based on event transitions
causes the model to lose its dynamic causal focus, resulting in a noticeable drop of6.33%. Failing
to perform dual-store topological merging and relying solely on isolated chunks completely breaks
long-range logic, causing a 10.08% decrease. Finally, removing the lightweight semantic filtering
deduplication causes graph traversal redundancy due to synonymous static descriptions, which dilutes
the model’s attention and leads to a 3.04% degradation.
Table 3:System Ablation Study on EventBench (Contextual).Each component provides significant and
indispensable performance gains in architecture integration, macroscopic causal reconstruction, and redundancy
suppression.
Model Variant Accuracy (%)
EC-RAG (Full System Architecture) 49.37
w/oElastic bidirectional diffusion sentinel
(degrades to fixed chunks without perception detection)46.84 (-2.53)
w/oLocal SES abstraction
(degrades to unstructured text summaries without specific dynamic structures)43.04 (-6.33)
w/oDual-store topological state merging
(degrades to single-hop blind matching with pure dense vectors [29])38.99 (-10.08)
w/oSemantic embedding filtering deduplication
(degrades to redundant direct information acquired solely by hop count)46.33 (-3.04)
7

4.5 24-Hour Infinite Stream Processing: Industrial-grade Quantitative Tracing Test
To thoroughly evaluate the framework’s deployment capability on extremely long real-world data, we
fed the system a continuous 24-hour unedited industrial security surveillance stream.
We extracted the independent event segments automatically chunked by the perception layer, input
them into a high-performance closed-source large model (Qwen3.5), and conducted rigorous blind
testing combined with human review. The evaluation adopts "Strict Accuracy," requiring the extraction
to simultaneously satisfy: (1) no hallucination in semantic actions; (2) the time window covers the
complete event cycle; (3) the entity feature descriptions match; (4) the location is correct.
00.51
8:00 9:00 10:00 11:00 12:00 13:00 14:00 15:00 16:00 17:00 18:00 19:00 20:00 21:00 22:00 23:00 0:00 1:00 2:00 3:00 4:00 5:00 6:00 7:00Acc
Figure 2: The hourly strict accuracy rate of the 24-hour security surveillance stream.
Utilizing a single RTX 5090 with 32GB of VRAM, the framework continuously processed the 24-hour
video. The system adaptively generated 2514 distinguishable event chunk samples, taking a total of
11 hours and 14 minutes, fully meeting the requirements for streaming processing. Among these,
2277 events fully met the four-dimensional strict matching criteria. Under the condition of using
only an 8B base model for cognitive extraction, EC-RAG achieved astrict information extraction
accuracy of 90.57%, and the hourly accuracy is shown in Figure 2. This large-scale quantitative
data confirms that by leveraging the diffusion sentinel and state-merged extraction, EC-RAG not only
breaks the physical constraints of extreme contexts but also maintains highly stable event recognition
over extensive spatiotemporal spans. By using only a single RTX 5090 and an 8B-scale VLM,
high-precision video event causal graph extraction can be achieved.
4.6 Qualitative Analysis: Graph Construction and Dual-Store Retrieval
Figure 3 details the EC-RAG pipeline for processing a cooking instructional video, illustrating both
Event Knowledge Graph (EKG) construction and dual-store retrieval.
SES Graph Construction (Figure 3a).The left panel depicts spatiotemporal discretization into a
State-Event-State (SES) chain. Instead of encoding redundant frames, EC-RAG abstracts critical
physical transformations (e.g., S0: intact shellE1:crack egg− − − − − − − →S 1: liquid yolk). Linking these tuples via
[:TEMPORAL_NEXT] relations based on post-state similarities successfully compresses the continuous
video into a discrete causal graph.
Dual-Store Retrieval and Generation (Figure 3b).The right panel demonstrates dual-store retrieval
for the query"How did the chef make custard sauce?". Pure vector retrieval yields only fragmented
semantic anchors ( A1: custard, A2: eggs, A3: cream), inevitably causing standard RAG systems
to hallucinate. Conversely, EC-RAG uses these anchors to initiate a bidirectional graph traversal
(max depth = 2), recovering missing intermediate causal steps ( E1, E2, E4). Following temporal
merging and semantic deduplication, the compiled chronological prompt enables the LLM to generate
a precise, hallucination-free answer.
4.7 VRAM Consumption and Infinite Stream Processing
To evaluate the system-level efficiency and hardware requirements of processing long-form videos,
we analyze the Video RAM (VRAM) consumption of an 8B-parameter VLM under both the native
unoptimized setting and the EC-RAG framework.
As illustrated in Figure 4(a), processing continuous video tokens natively poses a severe memory
bottleneck. Even with Grouped-Query Attention (GQA) optimizations (8 KV Heads, ∼0.000122
GB/Token), the KV-cache accumulates linearly with the input video length. Consequently, the native
VLM rapidly exhausts the standard 32GB hardware limit, reaching a critical Out-Of-Memory (OOM)
point at approximately162 seconds. This hardware wall fundamentally prohibits standard VLMs
from executing medium-to-long video reasoning tasks.
8

Figure 3:Qualitative pipeline of EC-RAG. (a) SES Graph Construction:Continuous video frames are
parsed into alternating nodes of entity States ( S) and Actions ( E), compressing pixels into a discrete causal
chain.(b) Dual-Store Retrieval and QA:For a complex causal query, pure vector retrieval only locates
isolated semantic anchors ( A1,A2,A3). EC-RAG utilizes these anchors to trigger a bidirectional graph traversal,
reconstructing the complete chronological evidence chain required for accurate LLM generation.
In stark contrast, Figure 4(b) demonstrates the bounded memory footprint achieved by EC-RAG. By
implementing an elastic streaming architecture with a maximum chunking window of 12 seconds, the
framework entirely decouples VRAM consumption from the total video duration. The peak memory
utilization strictly plateaus at ∼17.6 GB. Once a 12-second visual chunk is processed and abstracted
into a discrete State-Event-State (SES) textual node, the heavy visual KV-cache is flushed, and the
semantic state is persistently stored in the Graph DB.
This architectural paradigm successfully prevents OOM crashes, enabling 8B-scale open-source
models to process infinitely long video streams (e.g., 3600 seconds and beyond) with zero memory
degradation, making it highly deployable on standard consumer-grade GPUs.
(a) Unoptimized GQA VRAM Consumption
 (b) EC-RAG VRAM Consumption
Figure 4:VRAM consumption comparison during video processing. (a)The native 8B VLM accumulates
KV-cache linearly, hitting a 32GB OOM wall at ∼162 seconds.(b)EC-RAG, bounded by a 12s maximum
chunking strategy, plateaus at∼17.6 GB, enabling infinite stream processing without memory explosion.
9

5 Conclusion and Limitations
In this paper, we introduced EC-RAG, a lightweight retrieval-augmented framework for infinite long-
video reasoning. By replacing fixed clip-level memory with a graph memory, EC-RAG preserves
semantically coherent events and explicitly models temporal and causal dependencies across long
horizons. Built upon asynchronous event-driven segmentation, a compact dual-store event memory,
and bidirectional retrieval, our framework enables video foundation models to retrieve not only
semantically relevant content but also causally connected context, without expensive end-to-end
retraining over ultra-long sequences. Experimental results on long-video benchmarks show that EC-
RAG consistently outperforms strong clip-based retrieval baselines and long-context video models,
especially on questions requiring causal reasoning over large temporal gaps, while also achieving fa-
vorable efficiency in streaming settings. While EC-RAG systematically enhances objective reasoning
accuracy, the framework exhibits the limitation. Due to the limited availability of existing benchmark
resources, the applicability of the method to broader scenarios remains to be verified. In future work,
we will focus on building evaluation datasets for ultra-long video scenarios.
References
[1]Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li.
LLaV A-NeXT-Interleave: Tackling multi-image, video, and 3d in large multimodal models.arXiv preprint
arXiv:2407.07895, 2024.
[2]Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and
Ser-Nam Lim. MA-LMM: Memory-augmented large multimodal model for long-term video understanding.
InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
13504–13514, 2024.
[3]Kanchana Ranasinghe, Xiang Li, Kumara Kahatapitiya, and Michael S. Ryoo. Understanding long videos
with multimodal language models. InInternational Conference on Learning Representations (ICLR), 2025.
[4]Heqing Zou, Tianze Luo, Guiyang Xie, Victor Zhang, Fengmao Lv, Guangcong Wang, Junyang Chen,
Zhuochen Wang, Hansheng Zhang, and Huaijian Zhang. From seconds to hours: Reviewing multimodal
large language models on comprehensive long video understanding.arXiv preprint arXiv:2409.18938,
2024.
[5]Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and
Bo Zhao. Video-XL: Extra-long vision language model for hour-scale video understanding. InProceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26160–26169,
2025.
[6]Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. LongVLM: Efficient long
video understanding via large language models. InProceedings of the European Conference on Computer
Vision (ECCV), pages 453–470, 2024.
[7]Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark
for very long-form video language understanding. InAdvances in Neural Information Processing Systems
36 (NeurIPS 2023), Datasets and Benchmarks Track, 2023.
[8]Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context
interleaved video-language understanding. InAdvances in Neural Information Processing Systems 37
(NeurIPS 2024), Datasets and Benchmarks Track, 2024.
[9]Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou,
Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu
Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The first-ever comprehensive
evaluation benchmark of multi-modal llms in video analysis. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, 2025.
[10] Yifan Du, Kun Zhou, Yuqi Huo, Yifan Li, Wayne Xin Zhao, Haoyu Lu, Zijia Zhao, Bingning Wang,
Weipeng Chen, and Ji-Rong Wen. Towards event-oriented long video understanding, 2024.
[11] Yumeng Shi, Quanyu Long, Yin Wu, and Wenya Wang. Causality matters: How temporal information
emerges in video language models.arXiv preprint arXiv:2508.11576, 2026. AAAI 2026.
10

[12] Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang,
Ziyang Yan, Ao Ma, Xiaochen Wang, Hao Tang, Yan Wang, and Shuyan Li. EventV AD: Training-free
event-aware video anomaly detection. InProceedings of the 33rd ACM International Conference on
Multimedia (MM), pages 2586–2595, 2025.
[13] Sarfaraz Natha, Fareed Ahmed, Mohammad Siraj, Mehwish Lagari, Majid Altamimi, and Asghar Ali
Chandio. Deep bilstm attention model for spatial and temporal anomaly detection in video surveillance.
Sensors, 25(1):251, 2025.
[14] Sardar Waqar Khan, Qasim Hafeez, Muhammad Irfan Khalid, Roobaea Alroobaea, Saddam Hussain,
Jawaid Iqbal, Jasem Almotiri, and Syed Sajid Ullah. Anomaly detection in traffic surveillance videos using
deep learning.Sensors, 22(17), 2022.
[15] Ketan Pawar and Vahida Z. Attar. Deep learning based detection and localization of road accidents from
traffic surveillance videos.ICT Express, 8(3):379–387, 2022.
[16] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need. InAdvances in Neural Information Processing
Systems, volume 30, 2017.
[17] Hongchen Wei and Zhenzhong Chen. Visual context window extension: A new perspective for long video
understanding.arXiv preprint arXiv:2409.20018, 2024.
[18] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2PE: Improving
multimodal long-context capability of vision-language models with variable visual position encoding. In
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 21070–21084,
2025.
[19] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to
discard: Adaptive kv cache compression for llms. InInternational Conference on Learning Representations
(ICLR), 2024. Oral.
[20] Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. MiniCache: Kv
cache compression in depth dimension for large language models. InAdvances in Neural Information
Processing Systems 37 (NeurIPS 2024), Main Conference Track, 2024.
[21] Minsoo Kim, Kyuhong Shim, Jungwook Choi, and Simyung Chang. InfiniPot-V: Memory-constrained
kv cache compression for streaming video understanding. InAdvances in Neural Information Processing
Systems (NeurIPS), 2025.
[22] Jincheng Dai, Zhuowei Huang, Haiyun Jiang, Chen Chen, Deng Cai, Wei Bi, and Shuming Shi.
CORM: Cache optimization with recent message for large language model inference.arXiv preprint
arXiv:2404.15949, 2024.
[23] Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv
cache compression for high-throughput llm inference. InFindings of the Association for Computational
Linguistics: ACL 2024, pages 3258–3270, 2024.
[24] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao,
Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, and Junchen Jiang.
CacheGen: Kv cache compression and streaming for fast large language model serving. InProceedings of
the ACM SIGCOMM Conference, 2024.
[25] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lost in the middle: How language models use long contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024.
[26] Rafael Souza, Jia-Hao Lim, and Alexander Davis. Temporal contrastive learning for video temporal
reasoning in large vision-language models.arXiv preprint arXiv:2412.11391, 2024.
[27] Tieyuan Chen, Huabin Liu, Tianyao He, Yihang Chen, Chaofan Gan, Xiao Ma, Cheng Zhong, Yang
Zhang, Yingxue Wang, Hui Lin, and Weiyao Lin. MECD: Unlocking multi-event causal discovery in video
reasoning. InAdvances in Neural Information Processing Systems (NeurIPS), 2024. Spotlight paper.
[28] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. VideoRAG: Retrieval-augmented
generation over video corpus. InFindings of the Association for Computational Linguistics: ACL 2025,
pages 21278–21298, Vienna, Austria, 2025. Association for Computational Linguistics.
11

[29] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji,
Fei Chao, Jiebo Luo, and Rongrong Ji. Video-RAG: Visually-aligned retrieval-augmented long video
comprehension. InAdvances in Neural Information Processing Systems (NeurIPS), 2025.
[30] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-
augmented generation with extreme long-context videos.arXiv preprint arXiv:2502.01549, 2025.
[31] Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, and Chu-Song Chen. MegaRAG:
Multimodal knowledge graph-based retrieval augmented generation.arXiv preprint arXiv:2512.20626,
2025. Listed as ACL 2026 in the arXiv metadata available during verification.
[32] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani,
Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. Ask in
any modality: A comprehensive survey on multimodal retrieval-augmented generation. InFindings of
the Association for Computational Linguistics: ACL 2025, pages 16776–16809, Vienna, Austria, 2025.
Association for Computational Linguistics.
[33] Vladana Perlic, Stephane Lebailly, Vadim Malvone, Van-Tam Nguyen, and Pascal Urard. MMA-RAG: A
survey on multimodal agentic retrieval-augmented generation. SSRN preprint / HAL preprint hal-05322313,
2025.
[34] Ivan Rodin, Antonino Furnari, Kyle Min, Subarna Tripathi, and Giovanni Maria Farinella. Action scene
graphs for long-form understanding of egocentric videos. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 18622–18632, 2024.
[35] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. NExT-QA: Next phase of question-answering
to explaining temporal actions. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2021.
[36] Jintao Liu, Kaiwen Wei, and Chenglong Liu. Multimodal event causality reasoning with scene graph
enhanced interaction network.Proceedings of the AAAI Conference on Artificial Intelligence, 38(8):8778–
8786, 2024.
[37] Shuming Liu, Lin Sui, Chen-Lin Zhang, Fangzhou Mu, Chen Zhao, and Bernard Ghanem. Harnessing
temporal causality for advanced temporal action detection.arXiv preprint arXiv:2407.17792, 2024.
[38] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming
long video understanding with large language models. InAdvances in Neural Information Processing
Systems 37 (NeurIPS 2024), Main Conference Track, 2024.
[39] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Stream-
ingVLM: Real-time understanding for infinite video streams. InInternational Conference on Learning
Representations (ICLR), 2026.
[40] Yibin Yan, Jilan Xu, Shangzhe Di, Yikun Liu, Yudi Shi, Qirui Chen, Zeqian Li, Yifei Huang, and Weidi
Xie. Learning streaming video representation via multitask training. InProceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pages 9900–9912, 2025.
[41] Dibyadip Chatterjee, Edoardo Remelli, Yale Song, Bugra Tekin, Abhay Mittal, Bharat Bhatnagar, Necati Ci-
han Camgoz, Shreyas Hampali, Eric Sauser, Shugao Ma, Angela Yao, and Fadime Sener. Streaming
videollms for real-time procedural video understanding. InProceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pages 22586–22598, 2025.
[42] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu.
Streaming video understanding and multi-round interaction with memory-enhanced knowledge. In
International Conference on Learning Representations (ICLR), 2025.
[43] Nianbo Zeng, Haowen Hou, Fei Richard Yu, Si Shi, and Ying Tiffany He. Scenerag: Scene-level retrieval-
augmented generation for video understanding.arXiv preprint arXiv:2506.07600, 2025.
[44] Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mohamed Elhoseiny. Vgent: Graph-based retrieval-
reasoning-augmented generation for long video understanding. InAdvances in Neural Information
Processing Systems, 2025. Spotlight.
[45] Zongsheng Cao, Anran Liu, Yangfan He, Jing Li, Bo Zhang, and Zigan Wang. ViG-RAG: Video-aware
graph retrieval-augmented generation via temporal and semantic hybrid reasoning. InProceedings of the
AAAI Conference on Artificial Intelligence, volume 40, pages 48–56, 2026.
12

[46] Anurag Arnab, Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. Temporal chain of
thought: Long-video understanding by thinking in frames.arXiv preprint arXiv:2507.02001, 2025.
[47] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu.
Video-of-thought: Step-by-step video reasoning from perception to cognition. InProceedings of the
41st International Conference on Machine Learning, volume 235 ofProceedings of Machine Learning
Research, pages 13109–13125. PMLR, 2024.
[48] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image
pre-training. InProceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
13

A Implementation Details
A.1 A. Algorithm Pseudocode
Algorithm 1:EC-RAG: Spatiotemporal Graph Construction and Retrieval Strategy
Input:Streaming videoV, Vision encoderΨ v, VLM, EmbedderΨ e, QueryQ, ASR Speech
IslandsA, Thresholdsτ event, τbg, γ, τ dup, Patiencem= 2
Output:Constructed EKGG, Final answerA
// Phase 1: Perception Layer (Dual-Modal Sentinel Chunking)
1B← ∅,C← ∅;
2foreach framef t∈ Vdo
3v t←Ψ v(ft);B←B∪ {v t};
4if|B| ≥2then
5D smooth (t)←GaussianFilter(1−cos(v t−1, vt));
6ifD smooth (t)> τ event andis local optimumthen
7t core←t;
8t start←TraverseBack(t core,untilD smooth < τbgformconsecutive frames);
9t end←TraverseForward(t core,untilD smooth < τbgformconsecutive frames);
10foreach speech islandI audio = [tas, tae]∈ Ado
11ifI audio intersects[t start, tend]then
12t start←min(t start, tas);// Extend left boundary if needed
13t end←max(t end, tae);// Extend right boundary if needed
14C←C∪ {Sample(V[t start :tend],8.0FPS)};
// Phase 2: Cognition Layer (SES Abstraction & Graph Merging)
15G ← ∅,S prev←NULL ;
16foreach chunkc∈Cdo
17SES c←VLM.Generate(c,Prompt=“Extract SES”);
18G ← G ∪SES c, InsertSES cto VectorDB ;
19ifS prev̸=NULLthen
20v post←Ψ e(Sprev.Post_State);
21v pre←Ψ e(SES c.Pre_State);
22ifcos(v post, vpre)> γthen
23Create edgeS prev[:TEMPORAL_NEXT]− − − − − − − − − − − − →SES cinG;
24S prev←SES c;
// Phase 3: Retrieval & Generation (Hop Traversal & Deduplication)
25Anchors←VectorDB.Search(Ψ e(Q), TopK= 3);
26M seen← ∅,Ctx← ∅;
27foreach anchora∈Anchorsdo
28SubGraph←GraphDB.BFS_Traversal(a, MaxHops= 2);
29foreach noden∈SubGraphdo
30ifM seen̸=∅then
31S max←max v∈M seencos(Ψ e(n.Text), v);
32else
33S max←0;
34ifS max≤τdupthen
35Ctx←Ctx∪ {n.Text};
36M seen← M seen∪ {Ψ e(n.Text)};
37A←LLM.Generate(Q, Ctx);
38returnA;
14

B Prompt Engineering Details
To ensure the full reproducibility of the EC-RAG framework, we detail the exact prompt templates
employed in both the State-Event-State (SES) graph construction and the final Dual-Store generation
phases.
B.1 SES Abstraction Prompt (Perception & Cognition)
During the offline perception and cognition phases, a VLM is tasked with discretizing continuous
video chunks into chronological causal graphs. To prevent entity confusion and action hallucination,
we design strict instruction rules. Most notably, we introduce the“Goldilocks Granularity”rule to
prevent actions from being either too vague or too kinematic, and the“Visual Attribute Injection”
rule to strictly forbid the use of pronouns or generic IDs (e.g., forcing “the man in red” instead of “he”
or “E1”). This ensures seamless downstream topological merging.
Prompt 1: SES Graph Construction
[SYSTEM PROMPT]
You are a state-of-the-art Video-to-Graph Parser designed for high-fidelity
Event-Causal Reasoning.
[Strict Graph Construction Rules]:
1. Task-Level Physical Actions (The ’Goldilocks’ Granularity): Describe
specific, observable physical tasks and object interactions...
- DO NOT use vague umbrella terms (e.g., ’performing’).
- DO NOT over-decompose into meaningless joint kinematics.
2. Visual Attribute Injection (CRITICAL): NEVER use generic IDs or pronouns.
You MUST refer to entities by distinct visual attributes (e.g., write ’The
man in the black t-shirt’, NOT ’E1’).
3. Micro-Detail Exhaustion: Capture secondary background events, holding
props, and screen text.
4. Direct Visual Evidence (CRITICAL): Preserve only directly visible
evidence. If text is blurry or uncertain, output an empty string.
5. Strict Causality: An ’Event’ is a directed edge bridging a ’Pre-State’
and ’Post-State’.
6. Output Format: Output ONLY pure JSON.
[USER PROMPT TEMPLATE]
Target Timestamp: {start_time} - {end_time}.
{audio_context}
Task: Deconstruct this clip into a chronological causal graph.
Step 1: Inventory ALL interacting entities, detailing visual attributes.
Step 2: Map the specific, task-level physical actions.
Step 3: Preserve direct visual evidence useful for downstream QA.
Step 4: If any text is unreadable, write an empty string.
[Mandatory JSON Schema]: {scene_inventory: [...], events: [...]}
B.2 Dual-Store RAG Generation Prompt (Retrieval)
In the final QA generation phase, the LLM integrates both the original raw video frames and the
retrieved multihop text subgraph ( ctx). To mitigate potential hallucinations derived from the text
graph, we explicitly establish a decision hierarchy (Rule 4), instructing the model to prioritize directly
visible video evidence in the event of a conflict.
15

Prompt 2: Final QA Generation
You must answer a multiple-choice question about the attached original video.
Choose exactly one option index from 0 to 4.
### [Question & Choices]
{q}
### [Decision Rules]
1) Use the original video frames as the primary evidence.
2) Use the retrieved graph memory as complementary evidence from the same
video.
3) Compare all five options against the visible action, object, person, and
purpose.
4) If video evidence and graph memory disagree, prefer the directly visible
video evidence.
5) If evidence is incomplete, make the best forced choice from the available
options.
### [Retrieved RAG Memory]
The following graph memory was extracted from the same video and may help
identify temporal actions, objects, participants, and causal context.
{ctx}
Output exactly one line and nothing else: [FINAL ANSWER: X]
16